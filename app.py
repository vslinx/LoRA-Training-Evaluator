"""LoRA Training Evaluator — FastAPI backend.

Serves the web UI and provides API endpoints for trainer-guided workflow,
running face comparison, and serving sample images.
"""

import os
import sys
# Reduce CUDA fragmentation for the large (12.8B) sampling models; must be set
# before torch initializes CUDA. expandable_segments is Linux-only — setting it on
# Windows just triggers a per-run "not supported on this platform" warning, so skip
# it there.
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import traceback
import asyncio
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# Ensure CUDA runtime DLLs are on PATH before anything imports onnxruntime, so
# onnxruntime-gpu's CUDAExecutionProvider can load instead of silently falling
# back to CPU. torch/lib bundles the full CUDA 12 set; nvidia/* are fallbacks.
_site = Path(sys.prefix) / "Lib" / "site-packages"
for _bin in (_site / "torch" / "lib", _site / "nvidia" / "cudnn" / "bin", _site / "nvidia" / "cublas" / "bin"):
    if _bin.exists() and str(_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")

import webbrowser
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from face_analyzer import FaceAnalyzer
from style_analyzer import StyleAnalyzer
from trainers import TRAINERS, SAMPLE_MODELS, empty_sampling_info
from trainers import onetrainer
from trainers import aitoolkit
from trainers import anima

import samplers
from samplers.base import ModelFiles, SamplerSettings
from sampling import run_sample_generation

app = FastAPI(title="LoRA Training Evaluator")


@app.on_event("startup")
async def open_browser():
    webbrowser.open("http://127.0.0.1:8384")
analyzer = FaceAnalyzer()
style_analyzer = StyleAnalyzer()

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Persisted user settings (gitignored). Remembers per-model-family model files
# and sampling LoRAs so they don't have to be re-entered for every run.
CONFIG_DIR = Path(__file__).parent / "config"
SAMPLE_SETTINGS_FILE = CONFIG_DIR / "sample_settings.json"


def _load_sample_settings() -> dict:
    if SAMPLE_SETTINGS_FILE.is_file():
        try:
            return json.loads(SAMPLE_SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_sample_settings(data: dict) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    SAMPLE_SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# Saved prompt gallery (gitignored) — reusable prompts the user can pull into
# any run's sample generation instead of retyping them.
PROMPT_GALLERY_FILE = CONFIG_DIR / "prompt_gallery.json"


def _load_prompt_gallery() -> list:
    if PROMPT_GALLERY_FILE.is_file():
        try:
            data = json.loads(PROMPT_GALLERY_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else data.get("prompts", [])
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_prompt_gallery(prompts: list) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    PROMPT_GALLERY_FILE.write_text(json.dumps(prompts, indent=2), encoding="utf-8")


# ── Models ──────────────────────────────────────────────────────────────────────

class SelectFolderRequest(BaseModel):
    title: str = "Select Folder"

class SelectFileRequest(BaseModel):
    title: str = "Select File"
    filetypes: list[list[str]] | None = None  # [["Models", "*.safetensors"], ...]

class InspectSamplingRequest(BaseModel):
    trainer: str
    run_dir: str
    config_file: str

class SamplePrompt(BaseModel):
    prompt: str
    negative_prompt: str = ""

class SampleLora(BaseModel):
    path: str
    weight: float = 1.0

class PromptGalleryRequest(BaseModel):
    prompts: list[SamplePrompt] = []

class SaveFamilySettingsRequest(BaseModel):
    family: str
    # All optional: only the fields actually provided are merged into the saved
    # entry, so saving model files doesn't clobber a saved sampler and vice versa.
    model_path: str | None = None
    clip_path: str | None = None
    vae_path: str | None = None
    loras: list[SampleLora] | None = None
    sampler: dict | None = None

class GenerateSamplesRequest(BaseModel):
    trainer: str
    run_dir: str
    config_file: str
    model: str
    model_path: str = ""
    clip_path: str = ""
    vae_path: str = ""
    loras: list[SampleLora] = []
    sampler: dict = {}
    prompts: list[SamplePrompt] = []
    mode: str = "new"  # "new" = (re)generate all; "continue" = skip existing samples

class CheckExistingSamplesRequest(BaseModel):
    trainer: str
    run_dir: str
    config_file: str
    num_prompts: int = 1

class DeleteSamplesRequest(BaseModel):
    files: list[str] = []

class ValidateWorkspaceRequest(BaseModel):
    trainer: str
    path: str
    for_sampling: bool = False

class ListConfigsRequest(BaseModel):
    trainer: str
    run_dir: str

class RunConfigEntry(BaseModel):
    config_file: str
    selected_steps: list[int]
    dataset_folder: str | None = None
    label: str | None = None


class MultiRunRequest(BaseModel):
    trainer: str
    run_dir: str
    runs: list[RunConfigEntry]
    comparison_mode: str = "likeness"


# ── Progress state ──────────────────────────────────────────────────────────────

_progress: dict = {"current": 0, "total": 0, "label": "", "phase": ""}
_sampling_progress: dict = {"current": 0, "total": 0, "label": "", "phase": "idle"}
_sampling_cancel: bool = False  # set by /api/stop-sampling, checked by the orchestrator


# ── Endpoints ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/trainers")
async def get_trainers():
    return {"trainers": TRAINERS}


@app.post("/api/select-folder")
async def select_folder(req: SelectFolderRequest):
    """Open a native folder picker dialog."""
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, _open_folder_dialog, req.title)
    return {"path": path}


def _open_folder_dialog(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder or ""


@app.post("/api/select-file")
async def select_file(req: SelectFileRequest):
    """Open a native file picker dialog."""
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, _open_file_dialog, req.title, req.filetypes)
    return {"path": path}


def _open_file_dialog(title: str, filetypes: list | None) -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ft = [tuple(f) for f in filetypes] if filetypes else [("All files", "*.*")]
    path = filedialog.askopenfilename(title=title, filetypes=ft)
    root.destroy()
    return path or ""


@app.get("/api/sample-models")
async def get_sample_models():
    """Return the list of supported base models for sample generation."""
    return {"models": SAMPLE_MODELS}


@app.get("/api/sample-settings")
async def get_sample_settings():
    """Return persisted per-family model files + sampling LoRAs."""
    return {"families": _load_sample_settings()}


@app.post("/api/sample-settings")
async def save_sample_settings(req: SaveFamilySettingsRequest):
    """Merge model files / LoRAs / sampler settings into a model family's saved
    defaults (only the provided fields are updated) so they persist across runs."""
    if not req.family:
        raise HTTPException(400, "family is required")
    data = _load_sample_settings()
    entry = data.get(req.family, {})
    if req.model_path is not None:
        entry["model_path"] = req.model_path
    if req.clip_path is not None:
        entry["clip_path"] = req.clip_path
    if req.vae_path is not None:
        entry["vae_path"] = req.vae_path
    if req.loras is not None:
        entry["loras"] = [l.model_dump() for l in req.loras]
    if req.sampler is not None:
        entry["sampler"] = req.sampler
    data[req.family] = entry
    _save_sample_settings(data)
    return {"saved": True, "family": req.family}


@app.get("/api/prompt-gallery")
async def get_prompt_gallery():
    """Return the saved reusable prompts (positive/negative pairs)."""
    return {"prompts": _load_prompt_gallery()}


@app.post("/api/prompt-gallery")
async def save_prompt_gallery(req: PromptGalleryRequest):
    """Replace the saved prompt gallery with the provided list."""
    prompts = [p.model_dump() for p in req.prompts]
    _save_prompt_gallery(prompts)
    return {"saved": True, "count": len(prompts)}


TRAINER_MODULES = {
    "onetrainer": onetrainer,
    "ai-toolkit": aitoolkit,
    "anima": anima,
}


@app.post("/api/validate-workspace")
async def validate_workspace(req: ValidateWorkspaceRequest):
    mod = TRAINER_MODULES.get(req.trainer)
    if mod:
        if req.for_sampling and hasattr(mod, "validate_workspace_sampling"):
            valid = mod.validate_workspace_sampling(req.path)
        else:
            valid = mod.validate_workspace(req.path)
        return {"valid": valid}
    return {"valid": False, "error": "Trainer not yet supported"}


@app.post("/api/list-runs-sampling")
async def list_runs_sampling(req: ListConfigsRequest):
    """List training runs available for sample generation, with their saved
    LoRA checkpoint steps (does not require existing sample images)."""
    mod = TRAINER_MODULES.get(req.trainer)
    if not mod or not hasattr(mod, "list_runs_for_sampling"):
        raise HTTPException(400, "Trainer not yet supported")

    runs = mod.list_runs_for_sampling(req.run_dir)
    return {
        "configs": [
            {
                "config_file": r.config_file,
                "start_time": r.start_time,
                "base_model": r.base_model,
                "output_name": r.output_name,
                "total_sample_images": r.total_sample_images,
                "steps": r.steps,
            }
            for r in runs
        ]
    }


@app.post("/api/list-configs")
async def list_configs(req: ListConfigsRequest):
    mod = TRAINER_MODULES.get(req.trainer)
    if not mod:
        raise HTTPException(400, "Trainer not yet supported")

    runs = mod.list_configs(req.run_dir)
    return {
        "configs": [
            {
                "config_file": r.config_file,
                "start_time": r.start_time,
                "base_model": r.base_model,
                "output_name": r.output_name,
                "dataset_path": r.dataset_path,
                "num_samples": r.num_samples,
                "total_sample_images": r.total_sample_images,
                "steps": r.steps,
            }
            for r in runs
        ]
    }


@app.post("/api/inspect-sampling")
async def inspect_sampling(req: InspectSamplingRequest):
    """Inspect a training run's config to recognize the model and pre-fill
    sample generation settings (model, paths, sampler, prompts)."""
    mod = TRAINER_MODULES.get(req.trainer)
    if not mod or not hasattr(mod, "inspect_for_sampling"):
        # Unsupported trainer (e.g. kohya_ss / musubi_tuner): return blank form.
        return empty_sampling_info()
    return mod.inspect_for_sampling(req.run_dir, req.config_file)


@app.post("/api/check-existing-samples")
async def check_existing_samples(req: CheckExistingSamplesRequest):
    """Report how many of a run's target checkpoints already have generated
    samples, so the UI can offer to resume an interrupted run instead of
    regenerating everything."""
    mod = TRAINER_MODULES.get(req.trainer)
    if not mod or not hasattr(mod, "get_checkpoints_for_run"):
        return {"supported": False, "existing_images": 0, "total_images": 0}

    steps = sorted(mod.get_checkpoints_for_run(req.run_dir, req.config_file).keys())
    n = max(1, req.num_prompts)
    total_images = len(steps) * n

    existing = {}
    if hasattr(mod, "existing_samples"):
        existing = mod.existing_samples(req.run_dir, req.config_file)

    existing_images = 0
    done_steps = 0
    for step in steps:
        present = sum(1 for pidx in range(n) if (step, pidx) in existing)
        existing_images += present
        if present >= n:
            done_steps += 1

    return {
        "supported": hasattr(mod, "existing_samples"),
        "existing_images": existing_images,
        "total_images": total_images,
        "done_steps": done_steps,
        "total_steps": len(steps),
        "remaining_images": max(0, total_images - existing_images),
    }


@app.post("/api/generate-samples")
async def generate_samples(req: GenerateSamplesRequest):
    """Generate sample images natively across every saved LoRA checkpoint."""
    global _sampling_progress, _sampling_cancel
    _sampling_cancel = False  # clear any prior stop request

    if not samplers.is_supported(req.model):
        raise HTTPException(
            400,
            f"Native sampling for '{req.model}' is not implemented yet. "
            f"Currently supported: {', '.join(sorted(samplers.supported_families()))}.",
        )
    mod = TRAINER_MODULES.get(req.trainer)
    if not mod or not hasattr(mod, "get_checkpoints_for_run"):
        raise HTTPException(400, f"Trainer '{req.trainer}' does not support sampling yet.")
    if not req.prompts:
        raise HTTPException(400, "At least one prompt is required.")

    files = ModelFiles(model_path=req.model_path, clip_path=req.clip_path, vae_path=req.vae_path)
    s = req.sampler or {}
    settings = SamplerSettings(
        name=s.get("name", ""), scheduler=s.get("scheduler", ""),
        steps=int(s.get("steps", 20)), cfg=float(s.get("cfg", 7.0)),
        width=int(s.get("width", 1024)), height=int(s.get("height", 1024)),
        seed=int(s.get("seed", 42)), shift=float(s.get("shift", 3.0)),
    )
    extra_loras = [(l.path, l.weight) for l in req.loras if l.path]

    def _progress_cb(p: dict):
        global _sampling_progress
        _sampling_progress = {**_sampling_progress, **p}

    _sampling_progress = {"current": 0, "total": 0, "label": "Starting…", "phase": "loading"}

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: run_sample_generation(
                trainer_module=mod, family=req.model, run_dir=req.run_dir,
                config_file=req.config_file, model_files=files, settings=settings,
                prompts=req.prompts, extra_loras=extra_loras, mode=req.mode,
                progress=_progress_cb, should_stop=lambda: _sampling_cancel,
            ),
        )
    except ModuleNotFoundError as e:
        msg = (f"Missing dependency: {e}. Install the sample-generation extras:\n"
               "    pip install torch transformers diffusers safetensors einops accelerate")
        _sampling_progress = {**_sampling_progress, "phase": "error", "label": msg}
        raise HTTPException(500, msg)
    except Exception as e:
        # Print the full traceback to the console so the failing line is visible.
        tb = traceback.format_exc()
        print("\n=== Sample generation failed ===\n" + tb, file=sys.stderr, flush=True)
        # Include the last couple of frames in the UI message too.
        tail = "".join(traceback.format_exc().strip().splitlines(keepends=True)[-4:])
        _sampling_progress = {**_sampling_progress, "phase": "error", "label": str(e)}
        raise HTTPException(500, f"Sample generation failed: {e}\n\n{tail}")

    return result


@app.get("/api/sampling-progress")
async def sampling_progress():
    return _sampling_progress


@app.post("/api/stop-sampling")
async def stop_sampling():
    """Request the running generation to stop after the current image."""
    global _sampling_cancel
    _sampling_cancel = True
    return {"stopping": True}


@app.post("/api/delete-samples")
async def delete_samples(req: DeleteSamplesRequest):
    """Delete generated sample files (e.g. after stopping a bad run). For safety
    only deletes existing image files that live under a samples/sample folder."""
    deleted = 0
    for f in req.files:
        p = Path(f)
        parts = {x.lower() for x in p.parts}
        safe = p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} \
            and ("samples" in parts or "sample" in parts)
        if safe:
            try:
                p.unlink()
                deleted += 1
            except OSError:
                pass
    return {"deleted": deleted}


@app.get("/api/sampler-options")
async def sampler_options(model: str = ""):
    """Which sampler/scheduler values the selected model family actually supports.
    The UI shows only these (hiding the rest). Falls back to the SDXL set when no
    (or an unknown) model is given, for back-compat."""
    from samplers import get_supported_options
    opts = get_supported_options(model) if model else None
    if opts is not None:
        return opts
    from samplers.sdxl.sampler import SUPPORTED_SAMPLERS, SUPPORTED_SCHEDULERS
    return {"samplers": SUPPORTED_SAMPLERS, "schedulers": SUPPORTED_SCHEDULERS}


@app.post("/api/run-multi")
async def run_multi_comparison(req: MultiRunRequest):
    """Run comparison across multiple training runs."""
    mod = TRAINER_MODULES.get(req.trainer)
    if not mod:
        raise HTTPException(400, "Trainer not yet supported")

    run_inputs = []
    for entry in req.runs:
        steps_map = mod.get_samples_for_run(req.run_dir, entry.config_file)
        dataset_folder = entry.dataset_folder or mod.get_dataset_path(req.run_dir, entry.config_file)

        if not dataset_folder or not Path(dataset_folder).is_dir():
            raise HTTPException(400, f"Dataset folder not found: {dataset_folder}")
        if not steps_map:
            raise HTTPException(400, f"No sample images found for run: {entry.label or entry.config_file}")

        run_inputs.append({
            "label": entry.label or Path(entry.config_file).stem,
            "dataset_folder": dataset_folder,
            "steps_map": steps_map,
            "selected_steps": entry.selected_steps,
        })

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _run_multi_comparison, run_inputs, req.comparison_mode
    )
    return result


def _run_multi_comparison(run_inputs: list[dict], comparison_mode: str = "likeness") -> dict:
    global _progress

    if comparison_mode == "style":
        style_analyzer.initialize()
        active = style_analyzer
        no_embed_msg = "Could not extract style from any image in dataset for"
    else:
        analyzer.initialize()
        active = analyzer
        no_embed_msg = "No faces detected in dataset for"

    total_runs = len(run_inputs)
    all_runs = []

    # Cache dataset embeddings by folder path to avoid re-computing
    dataset_cache: dict[str, tuple] = {}

    for run_idx, run in enumerate(run_inputs):
        dataset_folder = run["dataset_folder"]
        label = run["label"]

        # Extract or reuse reference embeddings
        if dataset_folder not in dataset_cache:
            _progress = {
                "phase": "base",
                "current": run_idx + 1,
                "total": total_runs,
                "label": f"Loading dataset for {label}...",
            }
            ref_embeddings, ref_skipped = active.get_folder_embeddings(dataset_folder)
            if not ref_embeddings:
                raise HTTPException(400, f"{no_embed_msg} {label}")
            dataset_cache[dataset_folder] = (ref_embeddings, ref_skipped)
        else:
            ref_embeddings, ref_skipped = dataset_cache[dataset_folder]

        selected_steps = run["selected_steps"]
        steps_map = run["steps_map"]
        results = []
        total_steps = len(selected_steps)

        for i, step_num in enumerate(selected_steps):
            if step_num not in steps_map:
                continue

            _progress = {
                "phase": "compare",
                "current_run": run_idx + 1,
                "total_runs": total_runs,
                "run_label": label,
                "current": i + 1,
                "total": total_steps,
                "label": f"[{label}] Step {step_num}",
            }

            image_paths = steps_map[step_num]
            result = active.compare_images_to_reference(ref_embeddings, image_paths)
            results.append({"step": step_num, "name": f"Step {step_num}", **result})

        results.sort(key=lambda r: r["average_similarity"], reverse=True)
        all_runs.append({
            "label": label,
            "results": results,
            "ref_count": len(ref_embeddings),
            "ref_skipped": len(ref_skipped),
        })

    return {"runs": all_runs}


@app.get("/api/progress")
async def get_progress():
    return _progress


@app.get("/api/image")
async def serve_image(path: str):
    p = Path(path)
    if not p.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(p), media_type=f"image/{p.suffix.lstrip('.').replace('jpg', 'jpeg')}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8384)

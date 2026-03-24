"""LoRA Training Evaluator — FastAPI backend.

Serves the web UI and provides API endpoints for trainer-guided workflow,
running face comparison, and serving sample images.
"""

import os
import sys
import asyncio
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# Ensure nvidia DLLs (cuDNN, cuBLAS) are on PATH before anything imports onnxruntime
for _pkg in ("cudnn", "cublas"):
    _bin = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / _pkg / "bin"
    if _bin.exists() and str(_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from face_analyzer import FaceAnalyzer
from trainers import TRAINERS
from trainers import onetrainer
from trainers import aitoolkit

app = FastAPI(title="LoRA Training Evaluator")
analyzer = FaceAnalyzer()

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Models ──────────────────────────────────────────────────────────────────────

class SelectFolderRequest(BaseModel):
    title: str = "Select Folder"

class ValidateWorkspaceRequest(BaseModel):
    trainer: str
    path: str

class ListConfigsRequest(BaseModel):
    trainer: str
    run_dir: str

class RunRequest(BaseModel):
    trainer: str
    run_dir: str
    config_file: str
    selected_steps: list[int]
    dataset_folder: str | None = None  # override auto-detected dataset


class RunConfigEntry(BaseModel):
    config_file: str
    selected_steps: list[int]
    dataset_folder: str | None = None
    label: str | None = None


class MultiRunRequest(BaseModel):
    trainer: str
    run_dir: str
    runs: list[RunConfigEntry]


# ── Progress state ──────────────────────────────────────────────────────────────

_progress: dict = {"current": 0, "total": 0, "label": "", "phase": ""}


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


TRAINER_MODULES = {
    "onetrainer": onetrainer,
    "ai-toolkit": aitoolkit,
}


@app.post("/api/validate-workspace")
async def validate_workspace(req: ValidateWorkspaceRequest):
    mod = TRAINER_MODULES.get(req.trainer)
    if mod:
        valid = mod.validate_workspace(req.path)
        return {"valid": valid}
    return {"valid": False, "error": "Trainer not yet supported"}


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


@app.post("/api/run")
async def run_comparison(req: RunRequest):
    # Get samples from trainer
    mod = TRAINER_MODULES.get(req.trainer)
    if not mod:
        raise HTTPException(400, "Trainer not yet supported")
    steps_map = mod.get_samples_for_run(req.run_dir, req.config_file)
    dataset_folder = req.dataset_folder or mod.get_dataset_path(req.run_dir, req.config_file)

    if not dataset_folder or not Path(dataset_folder).is_dir():
        raise HTTPException(400, f"Dataset folder not found: {dataset_folder}")

    if not steps_map:
        raise HTTPException(400, "No sample images found for this training run")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _run_comparison, dataset_folder, steps_map, req.selected_steps
    )
    return result


def _run_comparison(dataset_folder: str, steps_map: dict, selected_steps: list[int]) -> dict:
    global _progress
    analyzer.initialize()

    _progress = {"phase": "base", "current": 0, "total": 0, "label": "Loading base dataset..."}
    ref_embeddings, ref_skipped = analyzer.get_folder_embeddings(dataset_folder)

    if not ref_embeddings:
        raise HTTPException(400, "No faces detected in dataset")

    results = []
    total_steps = len(selected_steps)

    for i, step_num in enumerate(selected_steps):
        if step_num not in steps_map:
            continue

        _progress = {
            "phase": "compare",
            "current": i + 1,
            "total": total_steps,
            "label": f"Step {step_num}",
        }

        image_paths = steps_map[step_num]
        result = analyzer.compare_images_to_reference(ref_embeddings, image_paths)

        results.append({
            "step": step_num,
            "name": f"Step {step_num}",
            **result,
        })

    results.sort(key=lambda r: r["average_similarity"], reverse=True)

    return {
        "results": results,
        "ref_count": len(ref_embeddings),
        "ref_skipped": len(ref_skipped),
    }


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
    result = await loop.run_in_executor(None, _run_multi_comparison, run_inputs)
    return result


def _run_multi_comparison(run_inputs: list[dict]) -> dict:
    global _progress
    analyzer.initialize()

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
            ref_embeddings, ref_skipped = analyzer.get_folder_embeddings(dataset_folder)
            if not ref_embeddings:
                raise HTTPException(400, f"No faces detected in dataset for {label}")
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
            result = analyzer.compare_images_to_reference(ref_embeddings, image_paths)
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
    uvicorn.run(app, host="127.0.0.1", port=8000)

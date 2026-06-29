"""AI Toolkit adapter.

Workspace structure (the output folder):
    output/                        <- this is what the user selects as workspace
        run_name_a/
            config.yaml            — training configuration
            samples/               — sample images generated during training
            *.safetensors          — checkpoint files
        run_name_b/
            config.yaml
            samples/
            ...

Sample filename format:
    {timestamp}__{step:09d}_{prompt_index}.jpg
"""

import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from trainers import TrainingRun, detect_sdxl_variant, empty_sampling_info

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
SAMPLE_RE = re.compile(r"^\d+__(\d{9})_(\d+)\.\w+$")
# Checkpoint files are named "{name}_{step:09d}.safetensors"; the final save
# has no step suffix and is ignored for the step list.
CHECKPOINT_RE = re.compile(r"_(\d+)\.safetensors$")


def validate_workspace(run_dir: str) -> bool:
    """Check that a path looks like an AI Toolkit output folder.

    Valid if it contains at least one subfolder with config.yaml and samples/.
    """
    p = Path(run_dir)
    if not p.is_dir():
        return False
    for child in p.iterdir():
        if child.is_dir() and (child / "config.yaml").is_file() and (child / "samples").is_dir():
            return True
    return False


def _parse_config(run_folder: Path) -> dict:
    """Parse the config.yaml from a run folder. Falls back to .job_config.json."""
    if HAS_YAML:
        cfg_path = run_folder / "config.yaml"
        if cfg_path.is_file():
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    import json
    json_path = run_folder / ".job_config.json"
    if json_path.is_file():
        return json.loads(json_path.read_text(encoding="utf-8"))

    return {}


def _get_process_config(data: dict) -> dict:
    """Extract the first process config from the parsed config."""
    processes = data.get("config", {}).get("process", [])
    if processes and isinstance(processes, list):
        return processes[0]
    return {}


def _config_name(data: dict) -> str:
    return data.get("config", {}).get("name", "") or data.get("meta", {}).get("name", "")


def _final_checkpoint(run_folder: Path, data: dict, numbered_steps) -> tuple[int, Path] | None:
    """Resolve AI Toolkit's final checkpoint.

    AI Toolkit saves intermediate checkpoints as ``{name}_{step:09d}.safetensors``
    but the *final* save drops the step suffix (just ``{name}.safetensors``), so it
    would otherwise be missed. Map it to the configured total step count
    (``train.steps``) so it's included as the last step; if that's unknown, fall
    back to one save interval past the highest numbered checkpoint.
    """
    name = _config_name(data)
    if not name:
        return None
    final = run_folder / f"{name}.safetensors"
    if not final.is_file():
        return None
    proc = _get_process_config(data)
    try:
        total = int(proc.get("train", {}).get("steps"))
    except (TypeError, ValueError):
        total = None
    if total is None:
        steps = list(numbered_steps)
        try:
            save_every = int(proc.get("save", {}).get("save_every"))
        except (TypeError, ValueError):
            save_every = 1
        total = (max(steps) + save_every) if steps else None
    if total is None:
        return None
    return total, final


def list_configs(run_dir: str) -> list[TrainingRun]:
    """List all training runs found in the output folder.

    Each subfolder with config.yaml + samples/ is treated as a separate run.
    """
    p = Path(run_dir)
    runs = []

    for child in sorted(p.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        if not (child / "config.yaml").is_file() or not (child / "samples").is_dir():
            continue

        data = _parse_config(child)
        proc = _get_process_config(data)

        # Run name
        name = data.get("config", {}).get("name", "") or data.get("meta", {}).get("name", "") or child.name

        # Base model
        model_cfg = proc.get("model", {})
        base_model_path = model_cfg.get("name_or_path", "unknown")
        base_model = Path(base_model_path).stem if base_model_path else "unknown"

        # Dataset path (first dataset)
        datasets = proc.get("datasets", [])
        dataset_path = ""
        if datasets:
            dataset_path = datasets[0].get("folder_path", "")

        # Sample prompts count
        sample_cfg = proc.get("sample", {})
        sample_prompts = sample_cfg.get("samples", [])
        num_samples = len(sample_prompts)

        # Scan samples folder
        samples_dir = child / "samples"
        total_images = 0
        discovered_steps = set()

        for img in samples_dir.iterdir():
            if img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            match = SAMPLE_RE.match(img.name)
            if match:
                total_images += 1
                discovered_steps.add(int(match.group(1)))

        # Start time from config modification time
        try:
            mtime = (child / "config.yaml").stat().st_mtime
            start_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            start_time = "unknown"

        # config_file stores the subfolder name to identify this run
        runs.append(TrainingRun(
            config_file=child.name,
            config_path=child / "config.yaml",
            start_time=start_time,
            base_model=base_model,
            output_name=name,
            dataset_path=dataset_path,
            num_samples=num_samples,
            total_sample_images=total_images,
            steps=sorted(discovered_steps),
        ))

    return runs


def validate_workspace_sampling(run_dir: str) -> bool:
    """Lenient check for sample generation: a run with a config is enough,
    even if no sample images exist yet (we generate those)."""
    p = Path(run_dir)
    if not p.is_dir():
        return False
    for child in p.iterdir():
        if child.is_dir() and (child / "config.yaml").is_file():
            return True
    return False


def list_runs_for_sampling(run_dir: str) -> list[TrainingRun]:
    """List runs available for sample generation, with their saved LoRA
    checkpoint steps (independent of whether sample images exist)."""
    p = Path(run_dir)
    runs = []

    for child in sorted(p.iterdir(), reverse=True):
        if not child.is_dir() or not (child / "config.yaml").is_file():
            continue

        data = _parse_config(child)
        proc = _get_process_config(data)
        name = data.get("config", {}).get("name", "") or data.get("meta", {}).get("name", "") or child.name

        model_cfg = proc.get("model", {})
        base_model_path = model_cfg.get("name_or_path", "unknown")
        base_model = Path(base_model_path).stem if base_model_path else "unknown"

        steps = sorted({
            int(m.group(1))
            for f in child.glob("*.safetensors")
            if (m := CHECKPOINT_RE.search(f.name))
        })
        final = _final_checkpoint(child, data, steps)
        if final and final[0] not in steps:
            steps = sorted(steps + [final[0]])

        try:
            mtime = (child / "config.yaml").stat().st_mtime
            start_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            start_time = "unknown"

        runs.append(TrainingRun(
            config_file=child.name,
            config_path=child / "config.yaml",
            start_time=start_time,
            base_model=base_model,
            output_name=name,
            dataset_path="",
            num_samples=0,
            total_sample_images=len(steps),
            steps=steps,
        ))

    return runs


def get_samples_for_run(run_dir: str, config_file: str) -> dict[int, list[Path]]:
    """Get sample images grouped by step number.

    config_file is the run subfolder name within the output directory.
    """
    samples_dir = Path(run_dir) / config_file / "samples"
    if not samples_dir.is_dir():
        return {}

    steps_map: dict[int, list[Path]] = defaultdict(list)

    for img_path in sorted(samples_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = SAMPLE_RE.match(img_path.name)
        if match:
            step_num = int(match.group(1))
            steps_map[step_num].append(img_path)

    return dict(sorted(steps_map.items()))


# AI Toolkit "arch" field → our sample-model key. SDXL variants get refined
# from the model filename; Z-Image base vs turbo is encoded in the arch itself.
def _map_arch(arch: str, model_ref: str) -> str:
    arch = (arch or "").lower()
    if arch == "sdxl":
        return detect_sdxl_variant(model_ref)
    if arch in ("zimage:turbo", "zimage_turbo"):
        return "zimage_turbo"
    if arch in ("zimage", "zimage:base", "zimage_base"):
        return "zimage_base"
    if arch == "krea2":
        return "krea2"
    return ""


def inspect_for_sampling(run_dir: str, config_file: str) -> dict:
    """Inspect a run's config.yaml to pre-fill the sample generation form."""
    info = empty_sampling_info()
    run_folder = Path(run_dir) / config_file
    data = _parse_config(run_folder)
    proc = _get_process_config(data)
    if not proc:
        return info

    model_cfg = proc.get("model", {}) or {}
    model_ref = model_cfg.get("name_or_path", "") or ""
    info["model"] = _map_arch(model_cfg.get("arch", ""), model_ref)
    info["model_path"] = model_ref
    info["vae_path"] = model_cfg.get("vae_path", "") or model_cfg.get("vae", "") or ""
    info["clip_path"] = model_cfg.get("clip_path", "") or ""

    sample_cfg = proc.get("sample", {}) or {}
    prompts = []
    for s in sample_cfg.get("samples", []) or []:
        if isinstance(s, dict) and s.get("prompt"):
            prompts.append(s["prompt"])
        elif isinstance(s, str) and s:
            prompts.append(s)
    info["prompts"] = prompts
    info["negative_prompt"] = sample_cfg.get("neg", "") or ""
    info["sampler"].update({
        "name": sample_cfg.get("sampler", "") or "",
        "scheduler": sample_cfg.get("scheduler", "") or "",
        "steps": sample_cfg.get("sample_steps", info["sampler"]["steps"]),
        "cfg": sample_cfg.get("guidance_scale", info["sampler"]["cfg"]),
        "width": sample_cfg.get("width", info["sampler"]["width"]),
        "height": sample_cfg.get("height", info["sampler"]["height"]),
        "seed": sample_cfg.get("seed", info["sampler"]["seed"]),
        "shift": sample_cfg.get("shift") or info["sampler"]["shift"],
    })

    return info


def get_checkpoints_for_run(run_dir: str, config_file: str) -> dict[int, Path]:
    """Map each saved step to its LoRA checkpoint file for sample generation."""
    run_folder = Path(run_dir) / config_file
    out: dict[int, Path] = {}
    if not run_folder.is_dir():
        return out
    for f in run_folder.glob("*.safetensors"):
        m = CHECKPOINT_RE.search(f.name)
        if m:
            out[int(m.group(1))] = f
    final = _final_checkpoint(run_folder, _parse_config(run_folder), out.keys())
    if final and final[0] not in out:
        out[final[0]] = final[1]
    return dict(sorted(out.items()))


def get_samples_output_dir(run_dir: str, config_file: str) -> Path:
    """Folder where generated samples are written (the run's samples/ folder)."""
    return Path(run_dir) / config_file / "samples"


def sample_output_path(run_dir: str, config_file: str, step: int,
                       prompt_idx: int, settings) -> Path:
    """Path for a natively-generated sample, in AI Toolkit's read convention
    (``{timestamp}__{step:09d}_{prompt_idx}.png``, flat in the run's samples/)."""
    out_dir = Path(run_dir) / config_file / "samples"
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return out_dir / f"{ts}__{step:09d}_{prompt_idx}.png"


def existing_samples(run_dir: str, config_file: str) -> dict[tuple[int, int], list[Path]]:
    """Map each already-present ``(step, prompt_idx)`` to its sample image(s),
    so the orchestrator can resume an interrupted run."""
    samples_dir = Path(run_dir) / config_file / "samples"
    out: dict[tuple[int, int], list[Path]] = defaultdict(list)
    if not samples_dir.is_dir():
        return {}
    for img in samples_dir.iterdir():
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        m = SAMPLE_RE.match(img.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)))].append(img)
    return dict(out)


def get_dataset_path(run_dir: str, config_file: str) -> str:
    """Extract the dataset path from a run's config."""
    run_folder = Path(run_dir) / config_file
    data = _parse_config(run_folder)
    proc = _get_process_config(data)
    datasets = proc.get("datasets", [])
    if datasets:
        return datasets[0].get("folder_path", "")
    return ""

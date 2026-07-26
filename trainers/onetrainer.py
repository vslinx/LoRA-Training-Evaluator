"""OneTrainer adapter.

Workspace structure:
    run/
        config/       — one JSON per training run, named YYYY-MM-DD_HH-MM-SS.json
        samples/      — numbered prompt folders with timestamped sample images
        backup/
        save/
        tensorboard/

Sample filename format:
    YYYY-MM-DD_HH-MM-SS-training-sample-{steps}-{epoch}-{index}.{ext}
"""

import json
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from trainers import TrainingRun, detect_sdxl_variant, empty_sampling_info

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
SAMPLE_RE = re.compile(r"training-sample-(\d+)-\d+-\d+\.\w+$")
# Like SAMPLE_RE but also captures the trailing prompt index (used for resume).
SAMPLE_STEP_IDX_RE = re.compile(r"training-sample-(\d+)-\d+-(\d+)\.\w+$")
TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
# Saved LoRA checkpoints in save/ are named
# "{ts}-save-{step}-{epoch}-{idx}.safetensors".
CHECKPOINT_RE = re.compile(r"-save-(\d+)-\d+-\d+\.safetensors$")


def validate_workspace(run_dir: str) -> bool:
    """Check that a path looks like a OneTrainer workspace/run folder."""
    p = Path(run_dir)
    return (
        p.is_dir()
        and (p / "config").is_dir()
        and (p / "samples").is_dir()
    )


def list_configs(run_dir: str) -> list[TrainingRun]:
    """List all training run configs with metadata, newest first."""
    config_dir = Path(run_dir) / "config"
    samples_dir = Path(run_dir) / "samples"

    configs = sorted(config_dir.glob("*.json"), reverse=True)
    runs = []

    for cfg_path in configs:
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        # Parse start time from filename
        ts_match = TIMESTAMP_RE.search(cfg_path.stem)
        if ts_match:
            start_time = ts_match.group(1).replace("_", " ")
        else:
            start_time = cfg_path.stem

        # Get dataset from last concept
        concepts = data.get("concepts", [])
        dataset_path = ""
        if concepts:
            dataset_path = concepts[-1].get("path", "")

        # Base model
        base_model = data.get("base_model_name", "unknown")

        # Output
        output = data.get("output_model_destination", "")
        output_name = Path(output).stem if output else cfg_path.stem

        # Sample count
        sample_defs = data.get("samples", [])
        num_samples = len(sample_defs)

        # Count matching sample images
        cfg_dt = _parse_config_timestamp(cfg_path.name)
        total_images = 0
        discovered_steps = set()
        if cfg_dt and samples_dir.is_dir():
            next_dt = _find_next_config_time(config_dir, cfg_path.name)
            for prompt_dir in samples_dir.iterdir():
                if not prompt_dir.is_dir():
                    continue
                for img in prompt_dir.iterdir():
                    if img.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    img_dt = _parse_file_timestamp(img.name)
                    if img_dt and img_dt >= cfg_dt and (next_dt is None or img_dt < next_dt):
                        total_images += 1
                        step_match = SAMPLE_RE.search(img.name)
                        if step_match:
                            discovered_steps.add(int(step_match.group(1)))

        runs.append(TrainingRun(
            config_file=cfg_path.name,
            config_path=cfg_path,
            start_time=start_time,
            base_model=base_model,
            output_name=output_name,
            dataset_path=dataset_path,
            num_samples=num_samples,
            total_sample_images=total_images,
            steps=sorted(discovered_steps),
        ))

    return runs


def get_samples_for_run(run_dir: str, config_file: str) -> dict[int, list[Path]]:
    """Get sample images grouped by step number for a specific training run.

    Filters images by timestamp to only include those from this run.
    """
    config_dir = Path(run_dir) / "config"
    samples_dir = Path(run_dir) / "samples"
    cfg_path = config_dir / config_file

    cfg_dt = _parse_config_timestamp(config_file)
    if not cfg_dt:
        return {}

    next_dt = _find_next_config_time(config_dir, config_file)

    steps_map: dict[int, list[Path]] = defaultdict(list)

    for prompt_dir in sorted(samples_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        for img_path in prompt_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img_dt = _parse_file_timestamp(img_path.name)
            if not img_dt or img_dt < cfg_dt:
                continue
            if next_dt and img_dt >= next_dt:
                continue
            step_match = SAMPLE_RE.search(img_path.name)
            if step_match:
                step_num = int(step_match.group(1))
                steps_map[step_num].append(img_path)

    return dict(sorted(steps_map.items()))


def validate_workspace_sampling(run_dir: str) -> bool:
    """Lenient check for sample generation: a OneTrainer run only needs a
    config/ folder (saved checkpoints live in save/, samples are optional)."""
    p = Path(run_dir)
    return p.is_dir() and (p / "config").is_dir()


def list_runs_for_sampling(run_dir: str) -> list[TrainingRun]:
    """List runs available for sample generation, counting saved LoRA
    checkpoints in save/ within each run's timestamp window."""
    config_dir = Path(run_dir) / "config"
    save_dir = Path(run_dir) / "save"
    runs = []

    for cfg_path in sorted(config_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        ts_match = TIMESTAMP_RE.search(cfg_path.stem)
        start_time = ts_match.group(1).replace("_", " ") if ts_match else cfg_path.stem
        base_model = data.get("base_model_name", "unknown")
        output = data.get("output_model_destination", "")
        output_name = Path(output).stem if output else cfg_path.stem

        cfg_dt = _parse_config_timestamp(cfg_path.name)
        next_dt = _find_next_config_time(config_dir, cfg_path.name) if cfg_dt else None
        steps = set()
        if cfg_dt and save_dir.is_dir():
            for f in save_dir.iterdir():
                m = CHECKPOINT_RE.search(f.name)
                if not m:
                    continue
                f_dt = _parse_file_timestamp(f.name)
                if f_dt and f_dt >= cfg_dt and (next_dt is None or f_dt < next_dt):
                    steps.add(int(m.group(1)))
        steps = sorted(steps)

        runs.append(TrainingRun(
            config_file=cfg_path.name,
            config_path=cfg_path,
            start_time=start_time,
            base_model=base_model,
            output_name=output_name,
            dataset_path="",
            num_samples=0,
            total_sample_images=len(steps),
            steps=steps,
        ))

    return runs


# OneTrainer model_type → our sample-model key mapping.
# Variants that need a name heuristic (SDXL) or base/turbo split (Z-Image)
# are resolved below in inspect_for_sampling.
_MODEL_TYPE_MAP = {
    "STABLE_DIFFUSION_XL_10_BASE": "sdxl",
    "STABLE_DIFFUSION_XL": "sdxl",
    "SDXL": "sdxl",
}


def inspect_for_sampling(run_dir: str, config_file: str) -> dict:
    """Inspect a training run's config to pre-fill the sample generation form."""
    info = empty_sampling_info()
    cfg_path = Path(run_dir) / "config" / config_file
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return info

    model_type = (data.get("model_type") or "").upper()
    base_model = data.get("base_model_name", "") or ""

    if model_type == "Z_IMAGE":
        info["model"] = "zimage_turbo" if "turbo" in base_model.lower() else "zimage_base"
    elif model_type.startswith("STABLE_DIFFUSION_XL") or model_type == "SDXL":
        info["model"] = detect_sdxl_variant(base_model)
    elif model_type in _MODEL_TYPE_MAP:
        info["model"] = _MODEL_TYPE_MAP[model_type]

    info["model_path"] = base_model
    info["vae_path"] = (data.get("vae", {}) or {}).get("model_name", "") or ""

    # Prompts and sampler defaults come from the saved sample definitions.
    samples = data.get("samples", []) or []
    prompts = [s.get("prompt", "") for s in samples if s.get("prompt")]
    info["prompts"] = prompts
    if samples:
        first = samples[0]
        info["negative_prompt"] = first.get("negative_prompt", "") or ""
        info["sampler"].update({
            "scheduler": first.get("noise_scheduler", "") or "",
            "steps": first.get("diffusion_steps", info["sampler"]["steps"]),
            "cfg": first.get("cfg_scale", info["sampler"]["cfg"]),
            "width": first.get("width", info["sampler"]["width"]),
            "height": first.get("height", info["sampler"]["height"]),
            "seed": first.get("seed", info["sampler"]["seed"]),
        })

    return info


def get_checkpoints_for_run(run_dir: str, config_file: str) -> dict[int, Path]:
    """Map each saved step to its LoRA checkpoint in save/ (timestamp-windowed)."""
    config_dir = Path(run_dir) / "config"
    save_dir = Path(run_dir) / "save"
    out: dict[int, Path] = {}
    cfg_dt = _parse_config_timestamp(config_file)
    if not cfg_dt or not save_dir.is_dir():
        return out
    next_dt = _find_next_config_time(config_dir, config_file)
    for f in save_dir.iterdir():
        m = CHECKPOINT_RE.search(f.name)
        if not m:
            continue
        f_dt = _parse_file_timestamp(f.name)
        if f_dt and f_dt >= cfg_dt and (next_dt is None or f_dt < next_dt):
            out[int(m.group(1))] = f
    return dict(sorted(out.items()))


def get_samples_output_dir(run_dir: str, config_file: str) -> Path:
    """Folder where generated samples are written (the workspace samples/ folder)."""
    return Path(run_dir) / "samples"


def sample_output_path(run_dir: str, config_file: str, step: int,
                       prompt_idx: int, settings) -> Path:
    """Path for a natively-generated sample, in OneTrainer's read convention.

    OneTrainer's evaluator only scans per-prompt subfolders of ``samples/`` and
    matches ``...-training-sample-{step}-{epoch}-{idx}.ext``, attributing images
    to a run by a ``YYYY-MM-DD_HH-MM-SS`` timestamp falling inside that run's
    config window. We anchor the timestamp to this run's own config time so the
    samples are attributed to it even if newer runs exist in the workspace.
    """
    samples_dir = Path(run_dir) / "samples"
    cfg_dt = _parse_config_timestamp(config_file) or datetime.now()
    ts = cfg_dt.strftime("%Y-%m-%d_%H-%M-%S")
    sub = samples_dir / str(prompt_idx)
    return sub / f"{ts}-training-sample-{step}-0-{prompt_idx}.png"


def existing_samples(run_dir: str, config_file: str) -> dict[tuple[int, int], list[Path]]:
    """Map each already-present ``(step, prompt_idx)`` to its sample image(s),
    so the orchestrator can resume an interrupted run. Restricted to this run's
    config timestamp window (same filtering as ``get_samples_for_run``)."""
    config_dir = Path(run_dir) / "config"
    samples_dir = Path(run_dir) / "samples"
    cfg_dt = _parse_config_timestamp(config_file)
    out: dict[tuple[int, int], list[Path]] = defaultdict(list)
    if not cfg_dt or not samples_dir.is_dir():
        return {}
    next_dt = _find_next_config_time(config_dir, config_file)
    for prompt_dir in samples_dir.iterdir():
        if not prompt_dir.is_dir():
            continue
        for img in prompt_dir.iterdir():
            if img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img_dt = _parse_file_timestamp(img.name)
            if not img_dt or img_dt < cfg_dt or (next_dt and img_dt >= next_dt):
                continue
            m = SAMPLE_STEP_IDX_RE.search(img.name)
            if m:
                out[(int(m.group(1)), int(m.group(2)))].append(img)
    return dict(out)


def get_dataset_path(run_dir: str, config_file: str) -> str:
    """Extract the dataset path from a config file (last concept)."""
    cfg_path = Path(run_dir) / "config" / config_file
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    concepts = data.get("concepts", [])
    if concepts:
        return concepts[-1].get("path", "")
    return ""


# ── Internal helpers ─────────────────────────────────────────────────────────


def _parse_config_timestamp(filename: str) -> datetime | None:
    """Parse datetime from config filename like '2026-03-23_03-19-07.json'."""
    ts_match = TIMESTAMP_RE.search(filename)
    if not ts_match:
        return None
    try:
        return datetime.strptime(ts_match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def _parse_file_timestamp(filename: str) -> datetime | None:
    """Parse datetime from sample filename like '2026-03-23_03-20-38-training-sample-...'."""
    ts_match = TIMESTAMP_RE.search(filename)
    if not ts_match:
        return None
    try:
        return datetime.strptime(ts_match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def _find_next_config_time(config_dir: Path, current_config: str) -> datetime | None:
    """Find the start time of the config that comes after the current one."""
    configs = sorted(config_dir.glob("*.json"))
    current_dt = _parse_config_timestamp(current_config)
    if not current_dt:
        return None

    for cfg in configs:
        cfg_dt = _parse_config_timestamp(cfg.name)
        if cfg_dt and cfg_dt > current_dt:
            return cfg_dt
    return None

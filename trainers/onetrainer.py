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

from trainers import TrainingRun

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
SAMPLE_RE = re.compile(r"training-sample-(\d+)-\d+-\d+\.\w+$")
TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")


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
        ts_match = TIMESTAMP_RE.match(cfg_path.stem)
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
    ts_match = TIMESTAMP_RE.match(filename)
    if not ts_match:
        return None
    try:
        return datetime.strptime(ts_match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def _parse_file_timestamp(filename: str) -> datetime | None:
    """Parse datetime from sample filename like '2026-03-23_03-20-38-training-sample-...'."""
    ts_match = TIMESTAMP_RE.match(filename)
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

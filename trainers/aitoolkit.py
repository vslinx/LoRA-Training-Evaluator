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

from trainers import TrainingRun

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
SAMPLE_RE = re.compile(r"^\d+__(\d{9})_(\d+)\.\w+$")


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


def get_dataset_path(run_dir: str, config_file: str) -> str:
    """Extract the dataset path from a run's config."""
    run_folder = Path(run_dir) / config_file
    data = _parse_config(run_folder)
    proc = _get_process_config(data)
    datasets = proc.get("datasets", [])
    if datasets:
        return datasets[0].get("folder_path", "")
    return ""

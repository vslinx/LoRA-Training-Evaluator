"""Anima Standalone Trainer adapter.

Workspace structure (the training-ui/jobs folder):
    training-ui/jobs/              <- user selects this as workspace
        run_name_a/
            config.toml            — training config
            dataset.toml           — dataset config with image_dir
            _merged_config.toml    — full merged config with model paths
            output/
                sample/            — sample images generated during training
                *.safetensors      — checkpoint files
        run_name_b/
            ...

Sample filename format:
    {output_name}_{step:06d}_{sample_idx:02d}_{timestamp}_{seed}.png
"""

import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

from trainers import TrainingRun

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
# Match step (6 digits), sample index (2 digits), timestamp (14 digits), seed from end of filename
SAMPLE_RE = re.compile(r"_(\d{6})_(\d{2})_\d{14}_\d+\.\w+$")


def validate_workspace(run_dir: str) -> bool:
    """Check that a path looks like an Anima jobs folder.

    Valid if it contains at least one subfolder with config.toml and output/sample/.
    """
    p = Path(run_dir)
    if not p.is_dir():
        return False
    for child in p.iterdir():
        if child.is_dir() and (child / "config.toml").is_file():
            sample_dir = child / "output" / "sample"
            if sample_dir.is_dir():
                return True
    return False


def _parse_toml(path: Path) -> dict:
    """Parse a TOML file."""
    if not path.is_file() or tomllib is None:
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def list_configs(run_dir: str) -> list[TrainingRun]:
    """List all training runs found in the jobs folder."""
    p = Path(run_dir)
    runs = []

    for child in sorted(p.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        config_path = child / "config.toml"
        if not config_path.is_file():
            continue

        sample_dir = child / "output" / "sample"
        if not sample_dir.is_dir():
            continue

        config = _parse_toml(config_path)
        dataset_config = _parse_toml(child / "dataset.toml")
        merged_config = _parse_toml(child / "_merged_config.toml")

        # Output name
        output_name = config.get("training_arguments", {}).get("output_name", child.name)

        # Base model from merged config
        dit_path = merged_config.get("model_arguments", {}).get("dit_path", "unknown")
        base_model = Path(dit_path).stem if dit_path else "unknown"

        # Dataset path from dataset.toml
        dataset_path = ""
        datasets = dataset_config.get("datasets", [])
        if datasets:
            subsets = datasets[0].get("subsets", [])
            if subsets:
                dataset_path = subsets[0].get("image_dir", "")

        # Sample prompts count
        sample_prompts_file = child / "sample_prompts.txt"
        num_samples = 0
        if sample_prompts_file.is_file():
            lines = sample_prompts_file.read_text(encoding="utf-8").strip().splitlines()
            num_samples = len([l for l in lines if l.strip() and not l.strip().startswith("#")])

        # Scan sample images
        total_images = 0
        discovered_steps = set()
        for img in sample_dir.iterdir():
            if img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            match = SAMPLE_RE.search(img.name)
            if match:
                total_images += 1
                discovered_steps.add(int(match.group(1)))

        # Start time from config mtime
        try:
            mtime = config_path.stat().st_mtime
            start_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            start_time = "unknown"

        runs.append(TrainingRun(
            config_file=child.name,
            config_path=config_path,
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
    """Get sample images grouped by step number.

    config_file is the run subfolder name within the jobs directory.
    """
    sample_dir = Path(run_dir) / config_file / "output" / "sample"
    if not sample_dir.is_dir():
        return {}

    steps_map: dict[int, list[Path]] = defaultdict(list)

    for img_path in sorted(sample_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = SAMPLE_RE.search(img_path.name)
        if match:
            step_num = int(match.group(1))
            steps_map[step_num].append(img_path)

    return dict(sorted(steps_map.items()))


def get_dataset_path(run_dir: str, config_file: str) -> str:
    """Extract the dataset path from a run's dataset.toml."""
    dataset_toml = Path(run_dir) / config_file / "dataset.toml"
    dataset_config = _parse_toml(dataset_toml)
    datasets = dataset_config.get("datasets", [])
    if datasets:
        subsets = datasets[0].get("subsets", [])
        if subsets:
            return subsets[0].get("image_dir", "")
    return ""

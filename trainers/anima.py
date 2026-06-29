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

from trainers import TrainingRun, empty_sampling_info

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
# Match step/epoch (optional 'e' prefix + 6 digits), sample index (2 digits), timestamp (14 digits), seed
SAMPLE_RE = re.compile(r"_e?(\d{6})_(\d{2})_\d{14}_\d+\.\w+$")
# Checkpoints in output/ are named "{output_name}-step{step:08d}.safetensors".
CHECKPOINT_RE = re.compile(r"step0*(\d+)\.safetensors$")


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


def validate_workspace_sampling(run_dir: str) -> bool:
    """Lenient check for sample generation: a run folder with config.toml is
    enough, even if no sample images exist yet."""
    p = Path(run_dir)
    if not p.is_dir():
        return False
    for child in p.iterdir():
        if child.is_dir() and (child / "config.toml").is_file():
            return True
    return False


def list_runs_for_sampling(run_dir: str) -> list[TrainingRun]:
    """List runs available for sample generation, with their saved LoRA
    checkpoint steps from the output/ folder."""
    p = Path(run_dir)
    runs = []

    for child in sorted(p.iterdir(), reverse=True):
        config_path = child / "config.toml"
        if not child.is_dir() or not config_path.is_file():
            continue

        config = _parse_toml(config_path)
        merged_config = _parse_toml(child / "_merged_config.toml")
        output_name = config.get("training_arguments", {}).get("output_name", child.name)
        dit_path = merged_config.get("model_arguments", {}).get("dit_path", "unknown")
        base_model = Path(dit_path).stem if dit_path else "unknown"

        output_dir = child / "output"
        steps = set()
        if output_dir.is_dir():
            for f in output_dir.glob("*.safetensors"):
                m = CHECKPOINT_RE.search(f.name)
                if m:
                    steps.add(int(m.group(1)))
        steps = sorted(steps)

        try:
            start_time = datetime.fromtimestamp(config_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            start_time = "unknown"

        runs.append(TrainingRun(
            config_file=child.name,
            config_path=config_path,
            start_time=start_time,
            base_model=base_model,
            output_name=output_name,
            dataset_path="",
            num_samples=0,
            total_sample_images=len(steps),
            steps=steps,
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


# Anima sample prompts embed generation params as CLI-style flags:
#   <positive prompt> --w 832 --h 1216 --s 28 --d 40917 --l 3.5 --n <negative>
_FLAG_RE = re.compile(r"\s--([whsdln])\s+(.*?)(?=\s--[whsdln]\s|$)", re.DOTALL)


def _parse_prompt_line(line: str) -> dict:
    """Split an Anima sample prompt line into positive text + generation flags."""
    # The positive prompt is everything before the first ' --x ' flag.
    first_flag = re.search(r"\s--[whsdln]\s", line)
    positive = line[: first_flag.start()].strip() if first_flag else line.strip()

    flags = {m.group(1): m.group(2).strip() for m in _FLAG_RE.finditer(line)}
    return {
        "prompt": positive,
        "negative": flags.get("n", ""),
        "width": flags.get("w"),
        "height": flags.get("h"),
        "steps": flags.get("s"),
        "seed": flags.get("d"),
        "guidance": flags.get("l"),
    }


def inspect_for_sampling(run_dir: str, config_file: str) -> dict:
    """Inspect an Anima run to pre-fill the sample generation form.

    Anima always trains the Anima base model, so the model key is fixed; the
    DiT/Qwen3/VAE paths come from the merged config.
    """
    info = empty_sampling_info()
    info["model"] = "anima"

    child = Path(run_dir) / config_file
    merged = _parse_toml(child / "_merged_config.toml")
    model_args = merged.get("model_arguments", {})
    info["model_path"] = model_args.get("dit_path", "") or ""
    info["clip_path"] = model_args.get("qwen3_path", "") or ""
    info["vae_path"] = model_args.get("vae_path", "") or ""

    prompts_file = child / "sample_prompts.txt"
    if prompts_file.is_file():
        lines = [
            l for l in prompts_file.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        parsed = [_parse_prompt_line(l) for l in lines]
        info["prompts"] = [p["prompt"] for p in parsed if p["prompt"]]
        if parsed:
            first = parsed[0]
            info["negative_prompt"] = first["negative"]
            if first["width"]:
                info["sampler"]["width"] = int(first["width"])
            if first["height"]:
                info["sampler"]["height"] = int(first["height"])
            if first["steps"]:
                info["sampler"]["steps"] = int(first["steps"])
            if first["seed"]:
                info["sampler"]["seed"] = int(first["seed"])
            if first["guidance"]:
                info["sampler"]["cfg"] = float(first["guidance"])

    return info


def get_checkpoints_for_run(run_dir: str, config_file: str) -> dict[int, Path]:
    """Map each saved step to its LoRA checkpoint in output/ for sampling."""
    output_dir = Path(run_dir) / config_file / "output"
    out: dict[int, Path] = {}
    if not output_dir.is_dir():
        return out
    for f in output_dir.glob("*.safetensors"):
        m = CHECKPOINT_RE.search(f.name)
        if m:
            out[int(m.group(1))] = f
    return dict(sorted(out.items()))


def get_samples_output_dir(run_dir: str, config_file: str) -> Path:
    """Folder where generated samples are written (the run's output/sample folder)."""
    return Path(run_dir) / config_file / "output" / "sample"


def sample_output_path(run_dir: str, config_file: str, step: int,
                       prompt_idx: int, settings) -> Path:
    """Path for a natively-generated sample, in Anima's read convention
    (``{output_name}_{step:06d}_{idx:02d}_{timestamp:14}_{seed}.png``)."""
    child = Path(run_dir) / config_file
    out_dir = child / "output" / "sample"
    config = _parse_toml(child / "config.toml")
    output_name = config.get("training_arguments", {}).get("output_name", config_file)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    seed = getattr(settings, "seed", 0)
    return out_dir / f"{output_name}_{step:06d}_{prompt_idx:02d}_{ts}_{seed}.png"


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

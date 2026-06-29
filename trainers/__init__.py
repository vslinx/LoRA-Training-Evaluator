"""Trainer adapters for different LoRA training tools.

Each trainer module provides functions to:
- Scan a workspace for training run configs
- Extract dataset paths and sample image mappings from configs
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingRun:
    """Parsed training run metadata."""
    config_file: str                          # filename of the config
    config_path: Path                         # full path to config file
    start_time: str                           # human-readable start time
    base_model: str                           # base model name
    output_name: str                          # output model filename
    dataset_path: str                         # path to the reference dataset
    num_samples: int                          # number of sample prompts
    total_sample_images: int                  # total sample images found
    steps: list[int] = field(default_factory=list)  # discovered step numbers


TRAINERS = {
    "onetrainer": "OneTrainer",
    "ai-toolkit": "AI Toolkit",
    "anima": "Anima Standalone Trainer (gazingstars123)",
    "kohya_ss": "Kohya SS",
    "musubi_tuner": "MusubiTuner",
}


# Supported base models for sample generation, grouped for the UI dropdown.
# Keys are stable identifiers; labels are display names. The "group" field
# lets the frontend render SDXL variants under a single optgroup.
SAMPLE_MODELS = {
    "sdxl":          {"label": "SDXL",          "group": "SDXL"},
    "pony":          {"label": "Pony",          "group": "SDXL"},
    "illustrious":   {"label": "Illustrious",   "group": "SDXL"},
    "noobai":        {"label": "NoobAI",        "group": "SDXL"},
    "anima":         {"label": "Anima",         "group": None},
    "zimage_base":   {"label": "Z-Image Base",  "group": None},
    "zimage_turbo":  {"label": "Z-Image Turbo", "group": None},
    "krea2":         {"label": "Krea2",         "group": None},
}


def detect_sdxl_variant(model_ref: str) -> str:
    """Guess the SDXL family variant from a model name/path.

    Returns one of: 'pony', 'illustrious', 'noobai', 'sdxl' (fallback).
    """
    ref = (model_ref or "").lower()
    if "noob" in ref:
        return "noobai"
    if "illustrious" in ref or "illust" in ref or "ilxl" in ref:
        return "illustrious"
    if "pony" in ref:
        return "pony"
    return "sdxl"


# Default empty result for inspect_for_sampling, so every trainer returns the
# same shape regardless of whether anything was recognized.
def empty_sampling_info() -> dict:
    return {
        "model": "",
        "model_path": "",
        "clip_path": "",
        "vae_path": "",
        "sampler": {
            "name": "",
            "scheduler": "",
            "steps": 20,
            "cfg": 7.0,
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "shift": 3.0,
        },
        "prompts": [],
        "negative_prompt": "",
    }

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
    "kohya_ss": "Kohya SS",
    "musubi_tuner": "MusubiTuner",
}

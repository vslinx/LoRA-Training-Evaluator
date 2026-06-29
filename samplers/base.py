"""Base interface for native sample-generation backends.

A ``Sampler`` loads a base model once, then the orchestrator merges the per-step
trained LoRA (+ any extra LoRAs) in and out around each batch of prompts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SamplerSettings:
    name: str = ""
    scheduler: str = ""
    steps: int = 20
    cfg: float = 7.0
    width: int = 1024
    height: int = 1024
    seed: int = 42
    shift: float = 3.0  # flow-matching shift (used by flow models like Z-Image)


@dataclass
class ModelFiles:
    model_path: str = ""
    clip_path: str = ""
    vae_path: str = ""


class Sampler(ABC):
    """Native diffusion sampler for one model family."""

    #: human-readable name of the family this sampler handles
    family: str = ""

    @abstractmethod
    def load(self, files: ModelFiles, device: str = "cuda", dtype: str = "bfloat16") -> None:
        """Load the base model, text encoder and VAE into memory."""

    @abstractmethod
    def prepare(self, prompts: list[tuple[str, str]]) -> None:
        """Pre-encode all (prompt, negative) pairs up front. This lets a sampler
        free the text encoder before the (large) denoiser takes the GPU."""

    @abstractmethod
    def apply_loras(self, loras: list[tuple[str, float]]):
        """Merge LoRAs (``(path, weight)``) and return an opaque handle with
        ``.unmerge()``. The orchestrator calls this per checkpoint."""

    @abstractmethod
    def generate(self, index: int, settings: SamplerSettings):
        """Generate the image for pre-encoded prompt ``index``. Returns a PIL.Image."""

    def unload(self) -> None:
        """Release model resources (optional)."""

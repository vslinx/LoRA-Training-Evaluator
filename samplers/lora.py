"""LoRA loading + merging for native sampling.

Supports the two common LoRA key conventions found in our trainers' outputs:

  - PEFT / diffusers style:  ``<prefix>.<module>.lora_A.weight`` / ``.lora_B.weight``
    (AI Toolkit saves this for Krea2, no ``alpha`` -> scale 1.0)
  - kohya / "lora_down" style: ``<module>.lora_down.weight`` / ``.lora_up.weight``
    with an optional ``.alpha``.

Rather than patch forward methods, we *merge* the low-rank delta directly into
each target ``nn.Linear`` weight, then keep enough state to *unmerge* it again.
That lets the orchestrator load the heavy base model once and swap the per-step
trained LoRA in and out cheaply (merge -> sample all prompts -> unmerge).
"""

from __future__ import annotations

import re
from pathlib import Path

# Prefixes a saved LoRA may use for the denoiser; all map onto the bare module
# path inside our transformer (e.g. "blocks.0.attn.wq").
_STRIP_PREFIXES = ("diffusion_model.", "transformer.", "lora_unet_", "lora_te_")


def _module_path_from_key(key: str) -> str | None:
    """Reduce a LoRA tensor key to the dotted module path it targets, or None."""
    k = key
    for p in _STRIP_PREFIXES:
        if k.startswith(p):
            k = k[len(p):]
            break
    for marker in (".lora_A.weight", ".lora_B.weight", ".lora_down.weight",
                   ".lora_up.weight", ".alpha"):
        if k.endswith(marker):
            return k[: -len(marker)]
    return None


def _resolve_module(root, dotted: str):
    """Resolve a dotted path like 'blocks.0.attn.wq' to a submodule (or None)."""
    obj = root
    for part in dotted.split("."):
        if part.isdigit():
            try:
                obj = obj[int(part)]
            except (IndexError, TypeError, KeyError):
                return None
        else:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
    return obj


def parse_lora_state(state_dict: dict) -> dict:
    """Group a flat LoRA state dict into ``{module_path: {A, B, alpha?}}``."""
    grouped: dict[str, dict] = {}
    for key, tensor in state_dict.items():
        path = _module_path_from_key(key)
        if path is None:
            continue
        entry = grouped.setdefault(path, {})
        if key.endswith("lora_A.weight") or key.endswith("lora_down.weight"):
            entry["A"] = tensor
        elif key.endswith("lora_B.weight") or key.endswith("lora_up.weight"):
            entry["B"] = tensor
        elif key.endswith(".alpha"):
            entry["alpha"] = tensor
    # Keep only complete A/B pairs.
    return {p: e for p, e in grouped.items() if "A" in e and "B" in e}


class MergedLora:
    """Handle for a set of LoRAs applied to a model; can fully reverse them.

    Two kinds of targets:
      - plain ``nn.Linear`` (e.g. bf16 models): the low-rank delta is merged into
        ``.weight`` and subtracted back on unmerge.
      - ``Int8Linear`` (weight-only int8): can't merge into int8 weights, so the
        LoRA is pushed as an additive branch and cleared on unmerge.
    """

    def __init__(self):
        self._applied: list = []   # (module, A, B, factor) merged into weight
        self._int8_mods: list = []  # Int8Linear modules with pushed branches
        self.matched = 0
        self.skipped: list[str] = []

    def unmerge(self):
        import torch
        with torch.no_grad():
            for module, A, B, factor in self._applied:
                delta = (B.float() @ A.float()) * factor
                module.weight.data -= delta.to(module.weight.dtype).to(module.weight.device)
        for module in self._int8_mods:
            module.clear_loras()
        self._applied.clear()
        self._int8_mods.clear()


def merge_loras(model, specs: list[tuple[dict, float]]) -> MergedLora:
    """Apply one or more LoRAs to ``model``. ``specs`` is ``(state_dict, mult)``.
    Returns a handle whose ``.unmerge()`` exactly reverses everything. Modules a
    LoRA references but that don't exist on the model go in ``handle.skipped``."""
    import torch
    from samplers.krea2.int8 import Int8Linear

    handle = MergedLora()
    with torch.no_grad():
        for state_dict, multiplier in specs:
            grouped = parse_lora_state(state_dict)
            for path, e in grouped.items():
                module = _resolve_module(model, path)
                A, B = e["A"], e["B"]
                rank = A.shape[0]
                alpha = float(e["alpha"]) if "alpha" in e else float(rank)
                factor = multiplier * (alpha / rank)
                if isinstance(module, Int8Linear):
                    module.push_lora(A, B, factor)   # additive branch (dequant-time)
                    handle._int8_mods.append(module)
                    handle.matched += 1
                elif module is not None and hasattr(module, "weight"):
                    delta = (B.float() @ A.float()) * factor
                    module.weight.data += delta.to(module.weight.dtype).to(module.weight.device)
                    handle._applied.append((module, A, B, factor))
                    handle.matched += 1
                else:
                    handle.skipped.append(path)
    return handle


def load_lora_file(path: str | Path) -> dict:
    """Load a LoRA ``.safetensors`` file into a flat state dict."""
    from safetensors.torch import load_file
    return load_file(str(path))

"""LoRA merging for the Anima sampler.

Anima saves LoRAs in kohya/ComfyUI convention: keys look like
``lora_unet_blocks_0_self_attn_q_proj.lora_down.weight`` (DiT) or
``lora_te1_...`` (Qwen3 text encoder), i.e. the dotted module path is flattened
to underscores behind a fixed prefix, with ``lora_down`` / ``lora_up`` / ``alpha``.

Underscore-flattening isn't reversible by string surgery (``self_attn`` itself
contains an underscore), so we can't reuse the generic dotted-path resolver in
``samplers/lora.py``. Instead we mirror the trainer's ``LoRANetwork``: walk the
model, flatten each ``nn.Linear``'s path the same way to reconstruct its LoRA key,
and merge the low-rank delta straight into the weight (keeping enough state to
unmerge again around each checkpoint).
"""

from __future__ import annotations


def _build_name_map(dit, text_encoder):
    """Map flattened LoRA names -> Linear modules for the DiT and text encoder.

    ``("lora_unet." + path).replace(".", "_")`` reproduces exactly the name the
    trainer assigns (prefix has no dots, so this equals prefix + "_" + flattened
    path). Only ``nn.Linear`` modules are targeted, which covers every attention /
    MLP / modulation / llm-adapter projection a LoRA can train.
    """
    import torch.nn as nn

    name_map: dict = {}
    for prefix, root in (("lora_unet", dit), ("lora_te1", text_encoder)):
        if root is None:
            continue
        for path, module in root.named_modules():
            if isinstance(module, nn.Linear):
                lora_name = (prefix + ("." + path if path else "")).replace(".", "_")
                name_map[lora_name] = module
    return name_map


def _group_lora(state_dict: dict) -> dict:
    """Group a flat Anima LoRA state dict into ``{name: {down, up, alpha?}}``."""
    grouped: dict = {}
    for key, tensor in state_dict.items():
        if "." not in key:
            continue
        name = key.split(".")[0]
        entry = grouped.setdefault(name, {})
        if key.endswith(".lora_down.weight"):
            entry["down"] = tensor
        elif key.endswith(".lora_up.weight"):
            entry["up"] = tensor
        elif key.endswith(".alpha"):
            entry["alpha"] = tensor
    return {n: e for n, e in grouped.items() if "down" in e and "up" in e}


class AnimaMergedLora:
    """Handle for merged Anima LoRAs; ``unmerge()`` reverses them.

    Only the low-rank ``down`` / ``up`` factors are kept (not the full-rank delta),
    so a whole checkpoint's worth of merges costs almost no extra VRAM; the delta is
    recomputed at unmerge time, exactly mirroring the shared ``samplers.lora`` helper.
    """

    def __init__(self):
        self._applied: list = []   # (module, down, up, factor)
        self.matched = 0
        self.skipped: list[str] = []

    def unmerge(self):
        import torch
        with torch.no_grad():
            for module, down, up, factor in self._applied:
                delta = (up.float() @ down.float()) * factor
                w = module.weight
                w.data -= delta.to(dtype=w.dtype, device=w.device)
        self._applied.clear()


def merge_loras(dit, text_encoder, specs: list[tuple[dict, float]]) -> AnimaMergedLora:
    """Merge ``(state_dict, multiplier)`` LoRAs into the DiT + text encoder.

    Returns a handle whose ``unmerge()`` restores the original weights. Names a
    LoRA references but the model doesn't have are collected in ``handle.skipped``.
    """
    import torch

    name_map = _build_name_map(dit, text_encoder)
    handle = AnimaMergedLora()
    with torch.no_grad():
        for state_dict, multiplier in specs:
            for name, e in _group_lora(state_dict).items():
                module = name_map.get(name)
                if module is None:
                    handle.skipped.append(name)
                    continue
                down, up = e["down"], e["up"]
                rank = down.shape[0]
                alpha = float(e["alpha"]) if "alpha" in e else float(rank)
                factor = multiplier * (alpha / rank)
                w = module.weight
                delta = (up.float() @ down.float()) * factor
                w.data += delta.to(dtype=w.dtype, device=w.device)
                handle._applied.append((module, down, up, factor))
                handle.matched += 1
    return handle


def load_lora_file(path: str) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))

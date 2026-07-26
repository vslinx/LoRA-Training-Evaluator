"""LoRA merging for the Anima sampler.

Anima LoRAs show up in two conventions and we must handle both, because the
trained checkpoint and a stacked helper LoRA can each use a different one:

- **kohya / ComfyUI** — keys like
  ``lora_unet_blocks_0_self_attn_q_proj.lora_down.weight`` (DiT) or ``lora_te1_…``
  (Qwen3 TE), i.e. the dotted module path flattened to underscores behind a fixed
  prefix, with ``lora_down`` / ``lora_up`` / ``alpha``. This is what the Anima
  Standalone Trainer saves. Underscore-flattening isn't reversible by string
  surgery (``self_attn`` itself contains an underscore), so we mirror the trainer's
  ``LoRANetwork``: walk the model and flatten each ``nn.Linear``'s path the same way
  to reconstruct its LoRA key.

- **native / PEFT** — keys like
  ``diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight`` / ``.lora_B.weight``
  (no ``alpha`` → scale 1.0). This is what ComfyUI-side LoRAs use — e.g. the Anima
  **turbo** distillation LoRA, which also targets ``llm_adapter`` blocks. Here the
  dotted path *is* the module path, so we resolve it directly.

Either way we merge the low-rank delta straight into the target weight, keeping the
low-rank factors so we can subtract it back around each checkpoint.
"""

from __future__ import annotations

# Prefixes a *dotted* LoRA key may carry in front of the real module path.
_DOTTED_PREFIXES = ("diffusion_model.", "transformer.", "net.")

# LoRA tensor suffixes -> which slot they fill. ``lora_A``/``lora_down`` are the
# down-projection; ``lora_B``/``lora_up`` the up-projection.
_SUFFIXES = {
    ".lora_down.weight": "down", ".lora_A.weight": "down",
    ".lora_up.weight": "up", ".lora_B.weight": "up",
    ".alpha": "alpha",
}


def _build_name_map(dit, text_encoder):
    """Map kohya flattened LoRA names -> Linear modules for DiT + text encoder.

    ``("lora_unet." + path).replace(".", "_")`` reproduces exactly the name the
    trainer assigns (the prefix has no dots, so this equals ``lora_unet_`` + the
    flattened path). Only ``nn.Linear`` modules are targeted, covering every
    attention / MLP / modulation / llm-adapter projection a LoRA can train.
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


def _resolve_dotted(root, dotted: str):
    """Resolve a dotted path like ``blocks.0.self_attn.q_proj`` to a submodule."""
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


def _split_suffix(key: str):
    """Return ``(base, slot)`` for a LoRA key, or ``(None, None)`` if it isn't one."""
    for suffix, slot in _SUFFIXES.items():
        if key.endswith(suffix):
            return key[: -len(suffix)], slot
    return None, None


def _group_lora(state_dict: dict) -> dict:
    """Group a flat Anima LoRA state dict into ``{module_base: {down, up, alpha?}}``.

    ``module_base`` is the whole key minus the tensor suffix (e.g.
    ``lora_unet_blocks_0_self_attn_q_proj`` or
    ``diffusion_model.blocks.0.self_attn.q_proj``) — resolution to an actual module
    happens in ``merge_loras``.
    """
    grouped: dict = {}
    for key, tensor in state_dict.items():
        base, slot = _split_suffix(key)
        if base is None:
            continue
        grouped.setdefault(base, {})[slot] = tensor
    return {b: e for b, e in grouped.items() if "down" in e and "up" in e}


def _target_module(base: str, name_map: dict, dit, text_encoder):
    """Resolve a grouped LoRA ``base`` to its target module, or ``None``.

    kohya names (``lora_unet_…`` / ``lora_te1_…``) come straight from the flattened
    name map; a dotted ``base`` is resolved against the DiT (then the text encoder)
    after stripping any ``diffusion_model.`` / ``transformer.`` / ``net.`` prefix.
    """
    if base.startswith("lora_unet") or base.startswith("lora_te"):
        return name_map.get(base)

    path = base
    for prefix in _DOTTED_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    module = _resolve_dotted(dit, path)
    if module is None and text_encoder is not None:
        # e.g. "text_encoder.model.layers.0.…" style TE keys.
        module = _resolve_dotted(text_encoder, path.split("text_encoder.")[-1])
    return module


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

    Handles both the kohya (``lora_unet_…`` + ``alpha``) and native/PEFT
    (``diffusion_model.…`` + ``lora_A``/``lora_B``) conventions. Returns a handle
    whose ``unmerge()`` restores the original weights; names a LoRA references but
    the model doesn't have are collected in ``handle.skipped``.
    """
    import torch

    name_map = _build_name_map(dit, text_encoder)
    handle = AnimaMergedLora()
    with torch.no_grad():
        for state_dict, multiplier in specs:
            for base, e in _group_lora(state_dict).items():
                module = _target_module(base, name_map, dit, text_encoder)
                if module is None or not hasattr(module, "weight"):
                    handle.skipped.append(base)
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

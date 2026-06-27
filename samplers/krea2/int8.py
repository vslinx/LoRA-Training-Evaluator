"""Weight-only quantized linear for fitting the 12.8B Krea2 transformer in 24 GB.

Dequantizing the whole model to bf16 needs ~25.6 GB — more than a 24 GB card.
Instead we keep the quantized weights resident (~12.8 GB) and dequantize each
weight to the activation dtype *on the fly* in the forward pass (the bf16 copy of
one layer is transient and freed immediately). Mirrors how ComfyUI runs the same
checkpoint, trading a little speed for fitting in VRAM.

Two quantizations are supported:
  - **INT8** with a per-output-row ``scale`` (``weight = int8 * scale``). NOTE:
    INT8 files quantized with ``convrot`` (a rotation) are NOT supported here and
    are rejected at load — use the fp8 file instead.
  - **fp8** (``float8_e4m3fn``) with no scale — a direct cast to the compute dtype.

LoRAs are applied as additive low-rank branches (not merged into the quantized
weights), so the per-step trained LoRA can be pushed/cleared cheaply.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantLinear(nn.Module):
    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor | None = None, bias=None):
        super().__init__()
        # Non-persistent: these aren't part of state_dict (we load them directly).
        self.register_buffer("qweight", qweight, persistent=False)
        if scale is not None:
            self.register_buffer("scale", scale, persistent=False)
        else:
            self.scale = None
        self.bias = bias
        self.out_features, self.in_features = qweight.shape
        self._loras: list[tuple] = []  # (A[r,in], B[out,r], scale)

    def push_lora(self, A: torch.Tensor, B: torch.Tensor, scale: float):
        self._loras.append((A, B, float(scale)))

    def clear_loras(self):
        self._loras = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize this layer's weight just for this matmul.
        w = self.qweight.to(x.dtype)
        if self.scale is not None:
            w = w * self.scale.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        out = F.linear(x, w, bias)
        for A, B, scale in self._loras:
            a = A.to(device=x.device, dtype=x.dtype)
            b = B.to(device=x.device, dtype=x.dtype)
            out = out + F.linear(F.linear(x, a), b) * scale
        return out


# Backwards-compatible alias (the class used to be INT8-only).
Int8Linear = QuantLinear


def set_submodule(root: nn.Module, dotted: str, new_module: nn.Module):
    """Replace the submodule at a dotted path (e.g. 'blocks.0.attn.wq')."""
    parts = dotted.split(".")
    obj = root
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new_module
    else:
        setattr(obj, last, new_module)

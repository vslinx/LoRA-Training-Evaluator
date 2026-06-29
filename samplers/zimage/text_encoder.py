"""Load the Z-Image Qwen3 text encoder from a single file (safetensors or GGUF).

Z-Image conditions on a ``Qwen3ForCausalLM`` text encoder. When the user points
the CLIP field at a single file rather than a diffusers folder, the model config
and tokenizer come from the vendored ``assets/text_encoder`` (small JSON +
tokenizer files, so this is fully offline — mirroring how Krea2 vendors its
Qwen3-VL assets), and only the *weights* come from the user's file.

Two single-file formats are supported:
- ``.safetensors`` — HF-style keys (``model.layers.*``), optionally prefixed
  (``text_encoders.``/``transformer.``); the prefix is stripped.
- ``.gguf`` — a llama.cpp Qwen3 export (``blk.N.attn_q.weight`` …). Tensors are
  dequantized via the ``gguf`` package and the keys are mapped to HF naming.
"""

from __future__ import annotations

import os
from pathlib import Path

# Vendored config + tokenizer so single-file loading needs no download. Override
# with ZIMAGE_TE_REPO (a local dir or HF repo id) to use a different Qwen3.
_TE_ASSETS = Path(__file__).parent / "assets" / "text_encoder"
ZIMAGE_TE_REPO = (
    os.getenv("ZIMAGE_TE_REPO")
    or (str(_TE_ASSETS) if (_TE_ASSETS / "config.json").is_file() else "Qwen/Qwen3-4B")
)

# Strip these prefixes off single-file state-dict keys to reach HF naming.
_PREFIXES = ("text_encoders.qwen3.transformer.", "text_encoders.", "text_encoder.",
             "transformer.", "module.")


def _strip_prefix(key: str) -> str:
    for p in _PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def _gguf_key_to_hf(name: str) -> str | None:
    """Map a llama.cpp Qwen3 tensor name to its HF ``Qwen3ForCausalLM`` key."""
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if name == "output.weight":  # only present when embeddings are untied
        return "lm_head.weight"
    if not name.startswith("blk."):
        return None
    _, idx, rest = name.split(".", 2)
    sub = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }.get(rest)
    return f"model.layers.{idx}.{sub}" if sub else None


def _load_gguf_state_dict(path: str, torch_dtype):
    """Read + dequantize a GGUF Qwen3 into an HF-keyed state dict."""
    import gguf
    import numpy as np
    import torch

    reader = gguf.GGUFReader(path)
    sd = {}
    for t in reader.tensors:
        hf_key = _gguf_key_to_hf(t.name)
        if hf_key is None:
            continue
        arr = gguf.dequantize(t.data, t.tensor_type)  # float32, HF [out, in] shape
        # gguf returns a read-only array; copy so the tensor owns writable memory.
        sd[hf_key] = torch.from_numpy(np.array(arr, copy=True)).to(torch_dtype)
    return sd


def _load_safetensors_state_dict(path: str, torch_dtype):
    from safetensors import safe_open
    import torch

    sd = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            sd[_strip_prefix(k)] = f.get_tensor(k).to(torch_dtype)
    return sd


def load_zimage_text_encoder(clip_path: str, torch_dtype):
    """Build a ``Qwen3ForCausalLM`` from the user's single file (kept on CPU).

    Returns ``(tokenizer, text_encoder)``. The caller places it on the device /
    hands it to the pipeline (which offloads it).
    """
    import torch
    from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

    p = Path(clip_path)
    if not p.is_file():
        raise ValueError(f"Z-Image text encoder file not found: {clip_path}")
    ext = p.suffix.lower()
    if ext not in (".safetensors", ".gguf"):
        raise ValueError(
            f"Z-Image text encoder must be a .safetensors or .gguf file — got '{p.name}'."
        )

    tokenizer = AutoTokenizer.from_pretrained(ZIMAGE_TE_REPO)
    config = AutoConfig.from_pretrained(ZIMAGE_TE_REPO)

    # Build on CPU (not meta) so computed buffers like RoPE inv_freq get real
    # values; default dtype set so the transiently-full model is bf16, not fp32.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        te = Qwen3ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)

    if ext == ".gguf":
        sd = _load_gguf_state_dict(clip_path, torch_dtype)
    else:
        sd = _load_safetensors_state_dict(clip_path, torch_dtype)

    # assign=True swaps in the file tensors; built buffers (RoPE) are kept.
    te.load_state_dict(sd, strict=False, assign=True)
    te.tie_weights()  # re-tie lm_head <-> embed_tokens (config ties them)
    meta_left = [n for n, t in te.named_parameters() if t.is_meta]
    if meta_left:
        raise RuntimeError(
            f"Z-Image text encoder has unloaded weights (file missing tensors): {meta_left[:5]}"
        )
    return tokenizer, te.eval().requires_grad_(False)

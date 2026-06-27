# Krea2 architecture — attribution

`mmdit.py`, `text_encoder.py` and `pipeline.py` in this directory are vendored
from **AI Toolkit** (https://github.com/ostris/ai-toolkit),
`extensions_built_in/diffusion_models/krea2/`.

AI Toolkit is distributed under the MIT License, Copyright (c) 2024 Ostris, LLC.
These files are included here under those terms to provide native Krea2
(SingleStreamDiT) inference without depending on the full AI Toolkit package.

`wan_vae.py` is vendored from **ComfyUI-VAE-Utils**
(https://github.com/spacepxl/ComfyUI-VAE-Utils, `src/wan/vae.py`), which adapts
the **Wan2.1 VAE** (https://github.com/Wan-Video/Wan2.1, Copyright 2024-2025 The
Alibaba Wan Team). Its two ComfyUI-internal dependencies (`comfy.ops`,
`vae_attention`) are shimmed out to plain `torch.nn` / SDPA so it runs standalone.
Used for the optional Wan2.1 "upscale2x" VAE decode path.

`qwen3vl_assets/` contains the **Qwen3-VL-4B-Instruct** `config.json` + tokenizer
files (from https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct, Apache-2.0,
Copyright Alibaba Qwen Team). They are vendored so the single-file ComfyUI text
encoder can be loaded fully offline (no config/tokenizer download), the way
ComfyUI bundles them. Override with the `QWEN3VL_REPO` env var.

`sampler.py` and `__init__.py` are original to this project and wrap the above
modules with a self-contained loader, LoRA merger, and sampling entry point.

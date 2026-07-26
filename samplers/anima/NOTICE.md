# Anima architecture — attribution

`anima_models.py` (the `MiniTrainDIT` denoiser — a Cosmos-Predict2 style DiT with
3D RoPE, AdaLN-LoRA modulation and the optional LLM adapter) originates with
**NVIDIA CORPORATION & AFFILIATES** and is licensed under the **Apache-2.0**
License. It is vendored here via the **Anima Standalone Trainer**
(gazingstars123), `library/anima_models.py`. The upstream trainer's block-swap
offloading / IPEX helpers and its custom logging setup are replaced with light
stubs, since single-GPU sample generation never enables block swap.

`anima_vae.py` is the **WanVAE**, adapted from the **Wan2.1 VAE**
(https://github.com/Wan-Video/Wan2.1, Copyright 2024-2025 The Alibaba Wan Team).
It is self-contained (torch + einops only) and used to decode Anima's 16-channel
5-D latents.

`assets/qwen3_06b/` contains the **Qwen3-0.6B** `config.json` + tokenizer files
(from https://huggingface.co/Qwen/Qwen3-0.6B, Apache-2.0, Copyright Alibaba Qwen
Team), vendored so a single-file `qwen_3_06b_base.safetensors` text encoder loads
fully offline. `assets/t5_old/` contains a **T5** tokenizer (SentencePiece model +
`tokenizer.json`) whose token IDs are fed to the DiT's LLM adapter as target tokens
(the T5 model itself is never run).

`sampler.py`, `text_encoder.py`, `lora.py` and `__init__.py` are original to this
project and wrap the above modules with a self-contained loader, dual-tokenizer
text-encoding path, kohya-style LoRA merger, and rectified-flow Euler sampling
entry point (ported from the trainer's `do_sample` / `_sample_image_inference`).

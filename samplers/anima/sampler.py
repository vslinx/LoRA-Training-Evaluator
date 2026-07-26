"""Native Anima sampler.

Anima is a Cosmos-Predict2 style flow-matching image model with three pieces:

- **DiT** (``MiniTrainDIT``) — the denoiser. ``baseV10`` is the 2048-channel /
  28-block variant and additionally carries an **LLM adapter**, so conditioning is
  a two-stage process (see below). Weights are stored under ``net.*`` (trainer) or
  ``model.diffusion_model.*`` (ComfyUI); the config is derived from the tensor
  shapes by ``get_dit_config``.
- **Qwen3-0.6B text encoder** — we feed it the prompt and take its *hidden states*
  (not logits). The base checkpoint also needs a **T5 tokenization** whose
  ``input_ids`` become the LLM adapter's target tokens (the T5 model is never run).
  ``crossattn_emb = dit.llm_adapter(source=qwen3_hidden, target=t5_ids, …)``.
- **WanVAE** (the Qwen-Image VAE, ``z_dim=16``, 8× spatial downscale) — decodes the
  5-D ``(B,16,1,H/8,W/8)`` latents. Normalisation is ``scale=[mean, 1/std]``.

Sampling is rectified-flow Euler: sigmas run linearly ``1 → 0`` (no shift is applied
during sampling), ``x += model_output * dt`` each step, with a doubled-batch CFG.
This ports ``library/anima_train_utils.do_sample`` / ``_sample_image_inference``
from the Anima Standalone Trainer, minus the multi-GPU / block-swap machinery that
never applies to single-GPU sampling.

The orchestrator pre-encodes every prompt through Qwen3 in ``prepare`` (before any
LoRA merge) and frees the text encoder; the LLM adapter — which a LoRA *can* target
— is run per prompt in ``generate`` (after the merge).

Heavy imports (torch / transformers) live inside ``load`` so importing this module
stays cheap.
"""

from __future__ import annotations

import os

from samplers.base import Sampler, SamplerSettings, ModelFiles
from samplers.anima.text_encoder import (
    load_qwen3_text_encoder, load_t5_tokenizer, AnimaTokenizers, encode_qwen3,
)
from samplers.anima.lora import merge_loras, load_lora_file

_DTYPES = {"bfloat16": "bfloat16", "bf16": "bfloat16", "float16": "float16",
           "fp16": "float16", "half": "float16", "float32": "float32", "fp32": "float32"}

# Weights kept at the (high-precision) base dtype rather than transformer dtype,
# mirroring the trainer. We load everything at one dtype, so this only matters if a
# separate transformer dtype is ever introduced; kept for parity/clarity.
_KEEP_IN_HIGH_PRECISION = ["x_embedder", "t_embedder", "t_embedding_norm", "final_layer"]

# WanVAE config is fixed (Qwen-Image VAE).
_VAE_CONFIG = dict(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                   attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)

# Height/width must be a multiple of 16 (VAE 8× × patch 2).
_BUCKET = 16


class _AnimaLoraHandle:
    def __init__(self, merged):
        self._merged = merged
        self.matched = getattr(merged, "matched", 0)
        self.skipped = getattr(merged, "skipped", [])

    def unmerge(self):
        if self._merged is not None:
            self._merged.unmerge()


class AnimaSampler(Sampler):
    family = "anima"
    # Anima only samples with rectified-flow Euler (linear 1→0 sigmas); there is no
    # sigma-spacing choice, so we expose the single euler/normal combo.
    SUPPORTED_SAMPLERS = ["euler"]
    SUPPORTED_SCHEDULERS = ["normal"]

    def __init__(self):
        self.dit = None
        self.vae = None
        self.vae_scale = None
        self.text_encoder = None
        self.tokenizers = None
        self._device = None
        self._dtype = None
        # Per-prompt pre-encoded conditioning: (qwen3_hidden, qwen3_mask,
        # t5_ids, t5_mask) for positive and negative.
        self._encoded: list = []

    # ---- loading -----------------------------------------------------------

    def load(self, files: ModelFiles, device: str = "cuda", dtype: str = "bfloat16") -> None:
        import torch

        base = files.model_path
        if not base or not (base.lower().endswith(".safetensors") and os.path.isfile(base)):
            raise ValueError(
                "Anima needs a Model path pointing at a DiT .safetensors "
                f"(e.g. anima_baseV10.safetensors) — got '{base}'."
            )
        if not files.clip_path:
            raise ValueError(
                "Anima needs a CLIP path (the Qwen3-0.6B text encoder, "
                "e.g. qwen_3_06b_base.safetensors)."
            )
        if not files.vae_path:
            raise ValueError("Anima needs a VAE path (the WanVAE, e.g. qwen_image_vae.safetensors).")

        td = getattr(torch, _DTYPES.get(dtype, "bfloat16"))
        self._device, self._dtype = device, td

        self.dit = self._load_dit(base, td, device)
        self.vae, self.vae_scale = self._load_vae(files.vae_path, device)
        self.text_encoder, qwen3_tokenizer = load_qwen3_text_encoder(files.clip_path, td, device)
        self.tokenizers = AnimaTokenizers(qwen3_tokenizer, load_t5_tokenizer())

    def _load_dit(self, dit_path: str, dtype, device: str):
        import torch
        from safetensors.torch import load_file
        from samplers.anima.anima_models import MiniTrainDIT, get_dit_config

        state_dict = load_file(dit_path, device="cpu")
        # Normalise checkpoint prefixes (trainer 'net.' / ComfyUI 'model.diffusion_model.').
        clean = {}
        for k, v in state_dict.items():
            for prefix in ("net.", "model.diffusion_model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            clean[k] = v
        state_dict = clean

        if "x_embedder.proj.1.weight" not in state_dict:
            raise RuntimeError(
                f"'{dit_path}' does not look like an Anima DiT checkpoint "
                "(missing x_embedder weights). Point the Model field at the Anima "
                "DiT .safetensors (e.g. anima_baseV10.safetensors)."
            )

        dit_config = get_dit_config(state_dict)
        # baseV10 embeds the LLM adapter directly in the DiT checkpoint.
        if "llm_adapter.out_proj.weight" in state_dict:
            dit_config["use_llm_adapter"] = True

        dit = MiniTrainDIT(**dit_config)
        dit.load_state_dict(state_dict, strict=False)

        # Per-parameter dtype: 1-D tensors + critical modules stay high precision.
        for name, p in dit.named_parameters():
            p.data = p.data.to(dtype=dtype)
        for name, b in dit.named_buffers():
            if b.is_floating_point():
                b.data = b.data.to(dtype=dtype)

        dit.blocks_to_swap = None
        return dit.requires_grad_(False).eval().to(device)

    def _load_vae(self, vae_path: str, device: str):
        import torch
        from safetensors.torch import load_file
        from samplers.anima.anima_vae import WanVAE_
        from samplers.anima.anima_models import ANIMA_VAE_MEAN, ANIMA_VAE_STD

        # VAE stays in float32 for decode fidelity (it's small, ~0.5 GB).
        vdtype = torch.float32
        with torch.device("meta"):
            vae = WanVAE_(**_VAE_CONFIG)
        if vae_path.endswith(".safetensors"):
            vae_sd = load_file(vae_path, device="cpu")
        else:
            vae_sd = torch.load(vae_path, map_location="cpu", weights_only=True)
        vae.load_state_dict(vae_sd, assign=True)
        vae = vae.eval().requires_grad_(False).to(device, dtype=vdtype)

        mean = torch.tensor(ANIMA_VAE_MEAN, dtype=vdtype, device=device)
        std = torch.tensor(ANIMA_VAE_STD, dtype=vdtype, device=device)
        return vae, [mean, 1.0 / std]

    # ---- conditioning ------------------------------------------------------

    def prepare(self, prompts: list[tuple[str, str]]) -> None:
        """Pre-encode every (prompt, negative) pair through Qwen3, then free the
        text encoder. Only the tokenizer-side T5 ids are kept for the adapter,
        which runs later (post-LoRA-merge) in ``generate``."""
        import torch

        self._encoded = []
        with torch.no_grad():
            for prompt, negative in prompts:
                pos = self._encode_one(prompt)
                neg = self._encode_one(negative) if (negative and negative.strip()) else None
                self._encoded.append((pos, neg))

        # The (large) DiT owns the GPU from here; drop the Qwen3 encoder.
        self.text_encoder = None
        self._cleanup()

    def _encode_one(self, text: str):
        """Return CPU-held ``(qwen3_hidden, qwen3_mask, t5_ids, t5_mask)`` for one
        prompt (padding zeroed in the hidden states, matching the trainer)."""
        q_ids, q_mask, t5_ids, t5_mask = self.tokenizers.tokenize(text or "")
        hidden, mask = encode_qwen3(self.text_encoder, q_ids, q_mask)
        return (hidden.detach().cpu(), mask.detach().cpu(),
                t5_ids.detach().cpu(), t5_mask.detach().cpu())

    def _crossattn(self, encoded):
        """Run the LLM adapter (if present) for one pre-encoded prompt and return
        the DiT cross-attention embedding on the compute device."""
        import torch

        hidden, mask, t5_ids, t5_mask = encoded
        comp_dtype = self.dit.t_embedding_norm.weight.dtype
        hidden = hidden.to(self._device, dtype=comp_dtype)
        mask = mask.to(self._device)
        t5_ids = t5_ids.to(self._device, dtype=torch.long)
        t5_mask = t5_mask.to(self._device)

        if getattr(self.dit, "use_llm_adapter", False) and hasattr(self.dit, "llm_adapter"):
            crossattn = self.dit.llm_adapter(
                source_hidden_states=hidden,
                target_input_ids=t5_ids,
                target_attention_mask=t5_mask,
                source_attention_mask=mask,
            )
            crossattn[~t5_mask.bool()] = 0
        else:
            crossattn = hidden
        return crossattn

    # ---- LoRA --------------------------------------------------------------

    def apply_loras(self, loras: list[tuple[str, float]]):
        specs = [(load_lora_file(path), float(weight)) for path, weight in loras if path]
        return _AnimaLoraHandle(merge_loras(self.dit, None, specs))

    # ---- generation --------------------------------------------------------

    def generate(self, index: int, settings: SamplerSettings):
        import torch

        pos, neg = self._encoded[index]
        comp_dtype = self.dit.t_embedding_norm.weight.dtype

        w = max(64, int(settings.width) - int(settings.width) % _BUCKET)
        h = max(64, int(settings.height) - int(settings.height) % _BUCKET)
        scale = float(settings.cfg)

        with torch.no_grad():
            crossattn = self._crossattn(pos)
            neg_crossattn = None
            if scale > 1.0 and neg is not None:
                neg_crossattn = self._crossattn(neg)

            latents = self._do_sample(
                h, w, int(settings.seed), crossattn, int(settings.steps),
                comp_dtype, scale, neg_crossattn,
            )
            image = self._decode(latents)
        self._cleanup()
        return image

    def _do_sample(self, height, width, seed, crossattn_emb, steps, dtype, guidance_scale,
                   neg_crossattn_emb):
        """Rectified-flow Euler sampling (single GPU). Ports ``do_sample``."""
        import torch

        device = torch.device(self._device)
        latent_h, latent_w = height // 8, width // 8

        generator = torch.manual_seed(seed) if seed is not None else None
        noise = torch.randn(
            (1, 16, 1, latent_h, latent_w), dtype=torch.float32,
            generator=generator, device="cpu",
        ).to(dtype).to(device)

        sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
        x = noise.clone()
        padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=dtype, device=device)

        use_cfg = guidance_scale > 1.0 and neg_crossattn_emb is not None
        if use_cfg:
            crossattn_doubled = torch.cat([crossattn_emb, neg_crossattn_emb], dim=0)
            padding_doubled = torch.cat([padding_mask, padding_mask], dim=0)

        with torch.autocast(device_type=device.type, enabled=False):
            for i in range(steps):
                sigma = sigmas[i]
                if use_cfg:
                    x_doubled = torch.cat([x, x], dim=0)
                    t_doubled = torch.stack([sigma, sigma])
                    out = self.dit(x_doubled, t_doubled, crossattn_doubled, padding_mask=padding_doubled)
                    pos_out, neg_out = out.chunk(2)
                    model_output = neg_out + guidance_scale * (pos_out - neg_out)
                else:
                    model_output = self.dit(x, sigma.unsqueeze(0), crossattn_emb, padding_mask=padding_mask)
                x = x + model_output * (sigmas[i + 1] - sigma)
        return x

    def _decode(self, latents):
        import numpy as np
        import torch
        from PIL import Image

        vae_p = next(self.vae.parameters())
        decoded = self.vae.decode(latents.to(vae_p.device, dtype=vae_p.dtype), self.vae_scale)
        image = torch.clamp((decoded.float() + 1.0) / 2.0, 0.0, 1.0)[0]
        if image.ndim == 4:  # drop the temporal dim (T=1) for images
            image = image[:, 0, :, :]
        arr = (255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)).astype(np.uint8)
        return Image.fromarray(arr)

    # ---- teardown ----------------------------------------------------------

    def _cleanup(self):
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload(self) -> None:
        self.dit = None
        self.vae = None
        self.text_encoder = None
        self.tokenizers = None
        self._encoded = []
        self._cleanup()

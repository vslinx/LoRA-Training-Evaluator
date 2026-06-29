"""Native Krea2 (SingleStreamDiT) sampler.

Loads the raw Krea2 transformer + the Qwen3-VL text encoder + a VAE, applies
LoRAs by merging, and runs AI Toolkit's flow-matching sampling loop (vendored in
``pipeline.py``). No dependency on the AI Toolkit package itself.

Two VAE backends are auto-detected from the VAE the user points at:
  - **Qwen-Image VAE** (diffusers ``AutoencoderKLQwenImage``) — the standard path.
  - **Wan2.1 upscale2x VAE** (a raw ComfyUI-format safetensors whose decoder head
    emits ``3 * r²`` channels) — decoded then ``pixel_shuffle(r)`` to give an
    image at r× the requested resolution. Vendored in ``wan_vae.py``.

Heavy imports (torch / transformers / diffusers) happen inside ``load`` so the
module can be imported for registry/inspection without them installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from samplers.base import Sampler, SamplerSettings, ModelFiles
from samplers.lora import merge_loras, load_lora_file

# The reference "single_mmdit_large_wide" architecture (oss_raw / oss_turbo share it).
KREA2_MMDIT_CONFIG = dict(
    features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12, multiplier=4,
    layers=28, patch=2, channels=16, txtheads=20, txtkvheads=20, txtlayers=12,
)

_SCHEDULE_KWARGS = {
    "schedule_y1": 0.5, "schedule_y2": 1.15,
    "schedule_min_res": 256, "schedule_max_res": 1280, "schedule_mu": None,
}

# Qwen-Image VAE latent statistics (from Qwen/Qwen-Image vae/config.json). The
# Krea2 transformer operates in this normalized latent space, so latents are
# de-normalized with these before VAE decode — for BOTH VAE backends, since the
# Wan upscale2x VAE consumes the same Qwen-Image latents.
QWEN_IMAGE_LATENTS_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653,
                           -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632,
                           -0.1922, -0.9497, 0.2503, -0.2921]
QWEN_IMAGE_LATENTS_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052,
                          2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253,
                          2.8251, 1.916]

# Wan2.1 VAE architecture (dim/z_dim fixed; in/out channels read from weights).
WAN21_VAE_CONFIG = dict(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                        attn_scales=[], temperal_downsample=[False, True, True],
                        dropout=0.0)

_DTYPES = {
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float16": "float16", "fp16": "float16", "half": "float16",
    "float32": "float32", "fp32": "float32",
}


_TRANSFORMER_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "model.")


def _strip_prefix(key: str) -> str:
    for p in _TRANSFORMER_PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def _split_transformer_state_dict(sd: dict, compute_dtype):
    """Split a raw / INT8 / fp8 Krea2 checkpoint into (plain, quant).

    ``quant`` maps bare module paths -> ``(int8_weight, scale)`` for
    ComfyUI-INT8-Fast quantized linears (``<m>.weight`` int8 + ``<m>.weight_scale``
    per-output-row + ``<m>.comfy_quant`` meta). These stay int8 (kept resident at
    ~12.8 GB; dequantized per-forward by Int8Linear). ``plain`` holds everything
    else (norms, modulation, biases, non-quant linears) cast to ``compute_dtype``,
    with the ``model.diffusion_model.`` prefix stripped. fp8 weights are upcast.
    """
    import torch

    scales = {
        k[: -len(".weight_scale")]: v
        for k, v in sd.items() if k.endswith(".weight_scale")
    }
    fp8_dtypes = tuple(
        d for d in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
        if d is not None
    )

    # Reject convrot-rotated INT8 (we can't dequantize it with a plain scale).
    import json
    for k, v in sd.items():
        if k.endswith(".comfy_quant"):
            try:
                conf = json.loads(bytes(v.tolist()).decode("utf-8"))
            except Exception:
                conf = {}
            if conf.get("convrot"):
                raise RuntimeError(
                    "This INT8 checkpoint uses 'convrot' rotated quantization, which "
                    "isn't supported. Use the fp8 file instead (e.g. krea2_turbo_fp8.safetensors)."
                )
            break

    plain, quant = {}, {}
    for k, v in sd.items():
        if k.endswith(".weight_scale") or k.endswith(".comfy_quant"):
            continue
        bare = _strip_prefix(k)
        if k.endswith(".weight") and k[: -len(".weight")] in scales:
            path = bare[: -len(".weight")]
            quant[path] = (v, scales[k[: -len(".weight")]].to(compute_dtype))   # int8 + scale
        elif k.endswith(".weight") and v.dtype in fp8_dtypes:
            quant[bare[: -len(".weight")]] = (v, None)                          # fp8, no scale
        else:
            plain[bare] = v.to(compute_dtype)
    return plain, quant


def _load_transformer_state_dict(name_or_path: str, compute_dtype):
    """Load + split MMDiT weights from a .safetensors file, a dir containing one,
    or an HF repo id (auto-detect raw vs INT8/fp8 — 'whatever the Model field
    points to'). Returns ``(plain_state_dict, quant_entries)``."""
    from safetensors.torch import load_file

    if name_or_path.endswith(".safetensors") and os.path.isfile(name_or_path):
        sd = load_file(name_or_path)
    elif os.path.isdir(name_or_path):
        candidates = [f for f in os.listdir(name_or_path) if f.endswith(".safetensors")]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Could not pick a Krea2 checkpoint in {name_or_path}: found {candidates}."
            )
        sd = load_file(os.path.join(name_or_path, candidates[0]))
    else:
        import huggingface_hub
        fname = name_or_path.split("/")[-1].split("-")[-1].lower() + ".safetensors"
        local = huggingface_hub.hf_hub_download(
            repo_id=name_or_path, filename=fname, token=os.getenv("HF_TOKEN")
        )
        sd = load_file(local)

    return _split_transformer_state_dict(sd, compute_dtype)


def _is_wan_vae_state_dict(sd: dict) -> bool:
    """Recognize a raw Wan2.1-family VAE (the alternative decode path)."""
    return "decoder.middle.0.residual.0.gamma" in sd and "decoder.head.2.weight" in sd


class _WanUpscaleVAE:
    """Wraps the vendored WanVAE: decodes Qwen-Image latents and pixel-shuffles
    the 3·r²-channel decoder output up to an r× resolution RGB image in [-1, 1]."""

    def __init__(self, state_dict: dict, device, dtype):
        import torch
        from .wan_vae import WanVAE

        out_ch = state_dict["decoder.head.2.weight"].shape[0]
        in_ch = state_dict["encoder.conv1.weight"].shape[1]
        self.upscale = max(1, round((out_ch // 3) ** 0.5)) if out_ch % 3 == 0 else 1
        self.z_dim = WAN21_VAE_CONFIG["z_dim"]

        cfg = dict(WAN21_VAE_CONFIG, in_channels=in_ch, out_channels=out_ch)
        model = WanVAE(**cfg)
        missing, _ = model.load_state_dict(state_dict, strict=False)
        if any("decoder" in k for k in missing):
            raise RuntimeError(f"Wan VAE missing decoder weights: {missing[:5]}")
        self.model = model.to(device=device, dtype=dtype).eval()
        self.model.requires_grad_(False)

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    def eval(self):
        self.model.eval()
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    def decode(self, latents_5d):
        """latents_5d: (B, 16, T, h, w) already de-normalized. Returns (B, 3, T, Hf, Wf)."""
        import torch
        import torch.nn.functional as F
        out = self.model.decode(latents_5d)            # (B, 3·r², T, H, W), ~[-1, 1]
        if self.upscale > 1:
            b, c, t, h, w = out.shape
            out = out.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            out = F.pixel_shuffle(out, self.upscale)   # (B·T, 3, H·r, W·r)
            out = out.reshape(b, t, 3, h * self.upscale, w * self.upscale).permute(0, 2, 1, 3, 4)
        return out


# Source for the Qwen3-VL config + tokenizer when the user points the CLIP field
# at a bare single-file checkpoint (the ComfyUI distribution format). We vendor
# these small files in qwen3vl_assets/ so generation is fully local (no download),
# mirroring how ComfyUI bundles them. Override with QWEN3VL_REPO (dir or repo id).
_QWEN3VL_ASSETS = Path(__file__).parent / "qwen3vl_assets"
QWEN3VL_REPO = (
    os.getenv("QWEN3VL_REPO")
    or (str(_QWEN3VL_ASSETS) if (_QWEN3VL_ASSETS / "config.json").is_file()
        else "Qwen/Qwen3-VL-4B-Instruct")
)


def _load_text_encoder(clip_path: str, device, torch_dtype, max_len: int):
    """Load the Qwen3-VL text encoder + tokenizers, dropping the vision tower.

    Supports a full HF directory / repo id, OR a single-file ComfyUI checkpoint
    (``qwen3vl_4b_bf16.safetensors``) — in which case the config + tokenizer come
    from QWEN3VL_REPO and the weights from the local file.
    """
    import torch
    from transformers import (
        AutoConfig, AutoTokenizer, Qwen2TokenizerFast, Qwen3VLForConditionalGeneration,
    )
    token = os.getenv("HF_TOKEN")
    p = Path(clip_path)
    single_file = clip_path.endswith(".safetensors") and p.is_file()
    repo = QWEN3VL_REPO if single_file else clip_path

    tokenizer = AutoTokenizer.from_pretrained(repo, max_length=max_len, token=token)
    processor = Qwen2TokenizerFast.from_pretrained(repo, max_length=max_len, token=token)

    if single_file:
        from safetensors import safe_open
        config = AutoConfig.from_pretrained(repo, token=token)
        # Build on CPU (not meta) so computed buffers like RoPE inv_freq get real
        # values; default dtype set so the (transiently full) model is bf16, not
        # float32. The vision tower is dropped right after construction.
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
        try:
            te = Qwen3VLForConditionalGeneration(config)
        finally:
            torch.set_default_dtype(prev_dtype)
        te.model.visual = None  # text-only conditioning; skip the vision tower
        sd = {}
        with safe_open(clip_path, framework="pt") as f:
            for k in f.keys():
                if ".visual." in k or k.startswith("model.visual"):
                    continue
                sd[k] = f.get_tensor(k)
        # assign=True swaps in the file tensors; built buffers (RoPE) are kept.
        te.load_state_dict(sd, strict=False, assign=True)
        te.tie_weights()  # re-tie lm_head <-> embed_tokens after assign
        meta_left = [n for n, t in te.named_parameters() if t.is_meta]
        if meta_left:
            raise RuntimeError(f"Qwen3-VL has unloaded weights: {meta_left[:5]}")
    else:
        te = Qwen3VLForConditionalGeneration.from_pretrained(repo, torch_dtype=torch_dtype, token=token)
        if getattr(te.model, "visual", None) is not None:
            te.model.visual = None

    te = te.to(device=device, dtype=torch_dtype).eval()
    te.requires_grad_(False)
    return tokenizer, processor, te


def _load_vae(vae_path: str, device, torch_dtype):
    """Return (vae_obj, kind, latents_mean, latents_std). kind is 'qwen' or 'wan'."""
    from safetensors.torch import load_file

    vp = Path(vae_path)
    # A single safetensors file might be the Wan upscale2x VAE.
    if vae_path.endswith(".safetensors") and vp.is_file():
        sd = load_file(vae_path)
        if _is_wan_vae_state_dict(sd):
            vae = _WanUpscaleVAE(sd, device, torch_dtype)
            return vae, "wan", QWEN_IMAGE_LATENTS_MEAN, QWEN_IMAGE_LATENTS_STD

    # Otherwise treat it as a diffusers AutoencoderKLQwenImage (dir / repo id).
    from diffusers import AutoencoderKLQwenImage
    kwargs = {"torch_dtype": torch_dtype, "token": os.getenv("HF_TOKEN")}
    if (vp / "vae").is_dir():
        vae = AutoencoderKLQwenImage.from_pretrained(vae_path, subfolder="vae", **kwargs)
    elif (vp / "config.json").is_file():
        vae = AutoencoderKLQwenImage.from_pretrained(vae_path, **kwargs)
    else:
        vae = AutoencoderKLQwenImage.from_pretrained(vae_path, subfolder="vae", **kwargs)
    vae = vae.to(device).eval()
    vae.requires_grad_(False)
    return vae, "qwen", list(vae.config.latents_mean), list(vae.config.latents_std)


class _ModelShim:
    """Minimal object exposing what the vendored Krea2Pipeline expects."""

    def __init__(self, transformer, vae, vae_kind, mean, std, device, dtype):
        self.transformer = transformer
        self.vae = vae
        self.vae_kind = vae_kind
        self.latents_mean = mean
        self.latents_std = std
        self.device_torch = device
        self.torch_dtype = dtype
        self.vae_torch_dtype = dtype
        self.vae_device_torch = device
        self.patch_size = KREA2_MMDIT_CONFIG["patch"]
        self.vae_scale_factor = 8
        self.model_config = SimpleNamespace(model_kwargs=dict(_SCHEDULE_KWARGS), low_vram=False)

    def decode_latents(self, latents, device=None, dtype=None):
        import torch
        device = device or self.vae_device_torch
        dtype = dtype or self.vae_torch_dtype
        z = len(self.latents_mean)
        latents = latents.to(device, dtype=dtype).unsqueeze(2)  # add frame dim
        mean = torch.tensor(self.latents_mean, device=latents.device, dtype=latents.dtype).view(1, z, 1, 1, 1)
        std = torch.tensor(self.latents_std, device=latents.device, dtype=latents.dtype).view(1, z, 1, 1, 1)
        latents = latents * std + mean
        if self.vae_kind == "wan":
            images = self.vae.decode(latents)            # (B, 3, T, Hf, Wf) in [-1, 1]
        else:
            images = self.vae.decode(latents).sample     # (B, 3, T, H, W) in [-1, 1]
        return images.squeeze(2).to(device, dtype=dtype)


class Krea2Sampler(Sampler):
    family = "krea2"
    # Experimental: Krea2 sampling is a fixed flow-matching Euler loop.
    SUPPORTED_SAMPLERS = ["euler"]
    SUPPORTED_SCHEDULERS = ["normal"]

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.text_encoder = None
        self._device = None
        self._dtype = None
        self._max_text_length = 512
        self._embeds = []      # pre-encoded [(cond, uncond_or_None), ...]

    def load(self, files: ModelFiles, device: str = "cuda", dtype: str = "bfloat16") -> None:
        import torch
        from .mmdit import SingleMMDiTConfig, SingleStreamDiT
        from .int8 import Int8Linear, set_submodule

        if not files.clip_path:
            raise ValueError("Krea2 needs a CLIP / text encoder path (Qwen3-VL).")
        if not files.vae_path:
            raise ValueError("Krea2 needs a VAE path (Qwen-Image or Wan upscale2x VAE).")

        torch_dtype = getattr(torch, _DTYPES.get(dtype, "bfloat16"))
        self._device = device
        self._dtype = torch_dtype

        # 1) Transformer — build on meta, swap quantized linears for Int8Linear,
        #    assign the remaining (bf16) params. Kept on CPU until prepare() frees
        #    the text encoder from the GPU (they don't fit together on 24 GB).
        config = SingleMMDiTConfig(**KREA2_MMDIT_CONFIG)
        with torch.device("meta"):
            transformer = SingleStreamDiT(config)
        plain, quant = _load_transformer_state_dict(files.model_path, torch_dtype)
        for path, (w8, scale) in quant.items():
            set_submodule(transformer, path, Int8Linear(w8, scale))
        missing, _ = transformer.load_state_dict(plain, strict=False, assign=True)
        meta_left = [n for n, p in transformer.named_parameters() if p.is_meta]
        if meta_left:
            raise RuntimeError(
                f"Krea2 transformer has {len(meta_left)} unloaded weights after loading "
                f"'{files.model_path}' (file may be a different architecture). "
                f"First few: {meta_left[:5]}"
            )
        transformer = transformer.eval().to("cpu")
        transformer.requires_grad_(False)
        self._quant = bool(quant)

        # 2) Text encoder (Qwen3-VL) on GPU; drop the vision tower (text-only).
        #    Accepts a full HF dir/repo or a single-file ComfyUI checkpoint.
        self.tokenizer, self.processor, self.text_encoder = _load_text_encoder(
            files.clip_path, device, torch_dtype, self._max_text_length
        )

        # 3) VAE (auto-detect Qwen-Image vs Wan upscale2x), kept on CPU for now
        vae, kind, mean, std = _load_vae(files.vae_path, "cpu", torch_dtype)
        self.model = _ModelShim(transformer, vae, kind, mean, std, device, torch_dtype)

    def prepare(self, prompts: list[tuple[str, str]]) -> None:
        """Encode every (prompt, negative) up front, then free the text encoder
        and move the denoiser + VAE onto the GPU."""
        import torch, gc
        self._embeds = []
        for prompt, negative in prompts:
            cond = self._encode(prompt)
            uncond = self._encode(negative)
            self._embeds.append((cond.to(self._device), uncond.to(self._device)))

        # Free the text encoder from the GPU before the big denoiser lands there.
        self.text_encoder = None
        self.tokenizer = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.transformer = self.model.transformer.to(self._device)
        self.model.vae = self.model.vae.to(self._device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def apply_loras(self, loras: list[tuple[str, float]]):
        specs = [(load_lora_file(path), float(weight)) for path, weight in loras if path]
        return merge_loras(self.model.transformer, specs)

    def _encode(self, prompt: str):
        from .text_encoder import encode_krea_prompt, SELECT_LAYERS
        feats = encode_krea_prompt(
            self.text_encoder, self.tokenizer, self.processor, prompt or "",
            max_length=self._max_text_length, select_layers=SELECT_LAYERS,
        )  # (L, 12, 2560)
        return feats.reshape(feats.shape[0], -1)  # flatten layer axis; pipeline restores it

    def generate(self, index: int, settings: SamplerSettings):
        import torch
        from .pipeline import Krea2Pipeline

        cond_feat, uncond_feat = self._embeds[index]
        do_cfg = bool(settings.cfg and settings.cfg > 0)
        cond = SimpleNamespace(text_embeds=[cond_feat])
        uncond = SimpleNamespace(text_embeds=[uncond_feat]) if do_cfg else None

        generator = torch.Generator(device=self._device).manual_seed(int(settings.seed))
        pipeline = Krea2Pipeline(self.model)
        images = pipeline(
            conditional_embeds=cond,
            unconditional_embeds=uncond,
            height=int(settings.height),
            width=int(settings.width),
            num_inference_steps=int(settings.steps),
            guidance_scale=float(settings.cfg),
            generator=generator,
        )
        return images[0]

    def unload(self) -> None:
        self.model = None
        self.text_encoder = None
        self._embeds = []
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

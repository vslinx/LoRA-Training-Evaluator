"""Native Z-Image (Base / Turbo) sampler.

Z-Image is diffusers-native: a ``ZImageTransformer2DModel`` denoiser, a Qwen3
(``Qwen3ForCausalLM``) text encoder, and a standard ``AutoencoderKL`` VAE, driven
by ``ZImagePipeline``. The pipeline is assembled from components two ways:

- **Folder mode** — the Model field points at a diffusers checkpoint *directory*
  with ``transformer/`` ``text_encoder/`` ``tokenizer/`` ``vae/`` subfolders (no
  ``model_index.json``), the same way AI Toolkit loads it.
- **Single-file mode** — the Model field points at a single ``.safetensors``
  transformer (ComfyUI ``model.diffusion_model.*`` keys, converted by diffusers'
  ``from_single_file``). The **VAE** field then takes a single ``.safetensors``
  VAE and the **CLIP** field a single ``.safetensors``/``.gguf`` Qwen3 text
  encoder. Config + tokenizer come from the vendored ``assets/`` so single-file
  loading needs no diffusers folder at all.

Z-Image is a flow-matching model and ``ZImagePipeline`` always drives the
scheduler with a ``mu`` (computed from the scheduler's ``base_shift``/``max_shift``),
so we use ``FlowMatchEulerDiscreteScheduler`` (its native scheduler). The user's
static ``shift`` is applied exactly via ``mu = ln(shift)`` with exponential
time-shift (``exp(mu) == shift`` makes the dynamic and static shift formulas
identical); the scheduler dropdown selects the sigma spacing (beta/karras/…).
LoRAs are applied via the shared ``merge_loras`` helper (trained checkpoints use
``diffusion_model.*`` keys).

Heavy imports happen inside ``load`` so the module imports without diffusers.
"""

from __future__ import annotations

import os
from pathlib import Path

from samplers.base import Sampler, SamplerSettings, ModelFiles
from samplers.lora import merge_loras, load_lora_file
from samplers.zimage.text_encoder import load_zimage_text_encoder

_DTYPES = {"bfloat16": "bfloat16", "bf16": "bfloat16", "float16": "float16",
           "fp16": "float16", "half": "float16", "float32": "float32", "fp32": "float32"}

# Width/height must be divisible by this (8 for the VAE × 2 for the patch size).
_BUCKET = 16

# Vendored config dirs (config.json) so single-file from_single_file / Qwen3 build
# need no diffusers folder.
_ASSETS = Path(__file__).parent / "assets"


# Samplers we support and the sigma spacings ("scheduler" dropdown) each one can
# actually use. ``ZImagePipeline`` always passes a ``mu`` to ``set_timesteps`` and
# forces ``sigma_min=0``, so a scheduler is usable only if its set_timesteps takes
# ``mu`` and still yields sane flow sigmas (1→0). Verified combos:
#  - euler (FlowMatch Euler, static shift): normal/beta/karras all fine.
#  - dpmpp_2m (DPMSolver flow, dynamic shift): only ``normal`` (beta/karras blow the
#    sigmas up to ~157). Shift is resolution-derived (the manual Shift is ignored).
#  - uni_pc (UniPC flow, static flow_shift): normal/karras (beta blows up). Honors
#    the manual Shift.
# (heun/ddpm have no ``mu`` param; exponential needs log(sigma_min=0) → domain error.)
SUPPORTED_SAMPLERS = ["euler", "dpmpp_2m", "uni_pc"]
SUPPORTED_SCHEDULERS = ["normal", "beta", "karras"]
_ALLOWED_SPACING = {
    "euler": {"normal", "beta", "karras"},
    "dpmpp_2m": {"normal"},
    "uni_pc": {"normal", "karras"},
}


def _sigma_kwargs(spacing: str) -> dict:
    return {
        "beta": {"use_beta_sigmas": True},
        "karras": {"use_karras_sigmas": True},
    }.get(spacing, {})


def _make_unipc(**kwargs):
    """UniPC scheduler that keeps ``sigmas`` on the timestep device.

    Stock ``UniPCMultistepScheduler.set_timesteps`` parks ``self.sigmas`` on the
    CPU, but its ``multistep_uni_*_bh_update`` methods index those CPU sigmas and
    then ``torch.stack`` them together with ``torch.ones((), device=sample.device)``.
    On CUDA that raises "Expected all tensors to be on the same device" from the
    second step on (order ≥ 2). ``FlowMatchEulerDiscreteScheduler`` dodges this by
    moving sigmas onto the sample device inside ``step``; UniPC never does. We
    restore parity by moving ``sigmas`` back onto the timestep device after
    ``set_timesteps`` (the timesteps are already placed there by diffusers).
    """
    from diffusers import UniPCMultistepScheduler

    class _DeviceUniPCMultistepScheduler(UniPCMultistepScheduler):
        def set_timesteps(self, *args, **kw):
            super().set_timesteps(*args, **kw)
            self.sigmas = self.sigmas.to(self.timesteps.device)

    return _DeviceUniPCMultistepScheduler(**kwargs)


def _build_scheduler(sampler: str, scheduler: str, shift: float):
    """Build the scheduler for the chosen Z-Image sampler.

    The ``sampler`` selects the ODE solver (FlowMatch Euler / DPMSolver / UniPC),
    the ``scheduler`` the sigma spacing. Spacings unsupported by the chosen solver
    fall back to ``normal`` so we never feed the pipeline blown-up (157-scale)
    sigmas. ``shift`` is applied as a static shift for euler/uni_pc; dpmpp_2m uses
    diffusers' resolution-derived dynamic shift.
    """
    from diffusers import (FlowMatchEulerDiscreteScheduler,
                           DPMSolverMultistepScheduler)

    smp = (sampler or "euler").lower()
    if smp not in _ALLOWED_SPACING:
        smp = "euler"
    sch = (scheduler or "normal").lower()
    if sch not in _ALLOWED_SPACING[smp]:
        sch = "normal"
    shift = float(shift) if (shift and shift > 0) else 3.0
    sigma_kw = _sigma_kwargs(sch)

    try:
        if smp == "dpmpp_2m":
            # DPMSolver-flow only runs with dynamic shifting (static flow_shift
            # asserts); the pipeline derives mu from base/max_shift defaults.
            return DPMSolverMultistepScheduler(
                num_train_timesteps=1000, use_flow_sigmas=True,
                prediction_type="flow_prediction", use_dynamic_shifting=True,
                solver_order=2, **sigma_kw)
        if smp == "uni_pc":
            return _make_unipc(
                num_train_timesteps=1000, use_flow_sigmas=True,
                prediction_type="flow_prediction", use_dynamic_shifting=False,
                flow_shift=shift, **sigma_kw)
        # euler (default): FlowMatch Euler with a static shift; the pipeline's mu
        # is accepted and ignored.
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, use_dynamic_shifting=False, shift=shift, **sigma_kw)
    except Exception:
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, use_dynamic_shifting=False, shift=shift)


class _ZImageLoraHandle:
    def __init__(self, merged):
        self._merged = merged
        self.matched = getattr(merged, "matched", 0)
        self.skipped = getattr(merged, "skipped", [])

    def unmerge(self):
        if self._merged is not None:
            self._merged.unmerge()


class ZImageSampler(Sampler):
    family = "zimage"
    SUPPORTED_SAMPLERS = SUPPORTED_SAMPLERS
    SUPPORTED_SCHEDULERS = SUPPORTED_SCHEDULERS

    def __init__(self):
        self.pipe = None
        self._device = None
        self._dtype = None
        self._prompts: list = []

    def load(self, files: ModelFiles, device: str = "cuda", dtype: str = "bfloat16") -> None:
        import torch
        from diffusers import ZImagePipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from diffusers.models.transformers import ZImageTransformer2DModel
        from transformers import AutoTokenizer, Qwen3ForCausalLM

        base = files.model_path
        if not base:
            raise ValueError("Z-Image needs a Model path (a diffusers folder or a single .safetensors).")
        td = getattr(torch, _DTYPES.get(dtype, "bfloat16"))
        self._device, self._dtype = device, td

        is_folder = os.path.isdir(base) and (Path(base) / "transformer").is_dir()
        if is_folder:
            # Folder mode: assemble from the diffusers checkpoint subfolders.
            transformer = ZImageTransformer2DModel.from_pretrained(base, subfolder="transformer", torch_dtype=td)
            tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
            text_encoder = Qwen3ForCausalLM.from_pretrained(base, subfolder="text_encoder", torch_dtype=td)
            vae_src = files.vae_path if (files.vae_path and os.path.isdir(files.vae_path)) else base
            vae_sub = None if vae_src != base else "vae"
            vae = AutoencoderKL.from_pretrained(vae_src, subfolder=vae_sub, torch_dtype=td)
        else:
            # Single-file mode: transformer + VAE + Qwen3 TE each from one file,
            # config/tokenizer from the vendored assets/.
            if not (base.lower().endswith(".safetensors") and os.path.isfile(base)):
                raise ValueError(
                    f"Z-Image Model must be a diffusers folder or a single .safetensors transformer — got '{base}'."
                )
            if not files.clip_path:
                raise ValueError(
                    "Single-file Z-Image needs a CLIP path (a .safetensors or .gguf Qwen3 text encoder)."
                )
            if not files.vae_path:
                raise ValueError("Single-file Z-Image needs a VAE path (a .safetensors VAE).")
            transformer = ZImageTransformer2DModel.from_single_file(
                base, config=str(_ASSETS / "transformer"), torch_dtype=td)
            vae = AutoencoderKL.from_single_file(
                files.vae_path, config=str(_ASSETS / "vae"), torch_dtype=td)
            tokenizer, text_encoder = load_zimage_text_encoder(files.clip_path, td)

        pipe = ZImagePipeline(
            scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0),
            text_encoder=text_encoder, tokenizer=tokenizer, vae=vae, transformer=transformer,
        )
        pipe.set_progress_bar_config(disable=True)
        # Keep VRAM in check on a 24 GB card: offload components to CPU and bring
        # only the active one onto the GPU, and tile the (large) VAE decode.
        try:
            pipe.enable_model_cpu_offload(device=device)
        except Exception:
            pipe.to(device)
        for _enable in ("enable_vae_tiling", "enable_vae_slicing"):
            try:
                getattr(pipe, _enable)()
            except Exception:
                pass
        self.pipe = pipe

    def prepare(self, prompts: list[tuple[str, str]]) -> None:
        self._prompts = list(prompts)

    def apply_loras(self, loras: list[tuple[str, float]]):
        specs = [(load_lora_file(path), float(weight)) for path, weight in loras if path]
        return _ZImageLoraHandle(merge_loras(self.pipe.transformer, specs))

    def generate(self, index: int, settings: SamplerSettings):
        import torch
        self.pipe.scheduler = _build_scheduler(settings.name, settings.scheduler, settings.shift)
        prompt, negative = self._prompts[index]
        w = max(_BUCKET, int(settings.width) // _BUCKET * _BUCKET)
        h = max(_BUCKET, int(settings.height) // _BUCKET * _BUCKET)
        generator = torch.Generator(self._device).manual_seed(int(settings.seed))
        out = self.pipe(
            prompt=prompt,
            negative_prompt=(negative or None),
            height=h,
            width=w,
            num_inference_steps=int(settings.steps),
            guidance_scale=float(settings.cfg),
            generator=generator,
        )
        return out.images[0]

    def unload(self) -> None:
        self.pipe = None
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

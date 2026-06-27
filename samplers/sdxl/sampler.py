"""Native SDXL sampler (covers SDXL / Pony / Illustrious / NoobAI).

Standard diffusers ``StableDiffusionXLPipeline``: the checkpoint is all-in-one
(UNet + CLIP + VAE), so it fits comfortably on a 24 GB card and there's no need
for the CPU staging the Krea2 sampler does. LoRAs (trained checkpoint + extras)
are applied via diffusers' adapter API and removed between checkpoints.

Heavy imports happen inside ``load`` so the module imports without diffusers.
"""

from __future__ import annotations

import os

from samplers.base import Sampler, SamplerSettings, ModelFiles

_DTYPES = {"bfloat16": "bfloat16", "bf16": "bfloat16", "float16": "float16",
           "fp16": "float16", "half": "float16", "float32": "float32", "fp32": "float32"}

# Sampler/scheduler dropdown values that map to a real diffusers scheduler (the
# rest of ComfyUI's list has no diffusers equivalent). The UI greys those out.
SUPPORTED_SAMPLERS = [
    "euler", "euler_ancestral", "euler_cfg_pp", "heun", "lms", "ddim", "ddpm",
    "deis", "uni_pc", "uni_pc_bh2", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde",
    "dpmpp_sde", "dpmpp_2s_ancestral", "lcm",
]
SUPPORTED_SCHEDULERS = [
    "normal", "karras", "exponential", "beta", "sgm_uniform", "simple", "ddim_uniform",
]


def _build_scheduler(pipe, name: str, scheduler: str):
    """Map our sampler + scheduler dropdown values to a diffusers scheduler.
    Returns None to keep the pipeline's default.

    The ComfyUI "scheduler" maps to sigma spacing: karras/exponential/beta map to
    the matching ``use_*_sigmas`` flag; sgm_uniform/simple/ddim_uniform map to a
    timestep_spacing. LCM (used by DMD2 few-step LoRAs) gets its own scheduler —
    using a normal multi-step scheduler for DMD2 produces noise.
    """
    from diffusers import (
        EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler,
        DDIMScheduler, UniPCMultistepScheduler, DEISMultistepScheduler,
        HeunDiscreteScheduler, LMSDiscreteScheduler, DDPMScheduler, LCMScheduler,
    )
    cfg = pipe.scheduler.config
    name = (name or "").lower()
    sch = (scheduler or "").lower()

    # Sigma spacing flags (only some schedulers accept these).
    sigma_kw = {}
    if sch == "karras":
        sigma_kw = {"use_karras_sigmas": True}
    elif sch == "exponential":
        sigma_kw = {"use_exponential_sigmas": True}
    elif sch == "beta":
        sigma_kw = {"use_beta_sigmas": True}
    # Timestep spacing for the "uniform"-style schedulers.
    spacing_kw = {}
    if sch in ("sgm_uniform", "simple"):
        spacing_kw = {"timestep_spacing": "trailing"}
    elif sch == "ddim_uniform":
        spacing_kw = {"timestep_spacing": "linspace"}

    table = {
        "euler": lambda: EulerDiscreteScheduler.from_config(cfg, **sigma_kw, **spacing_kw),
        "euler_ancestral": lambda: EulerAncestralDiscreteScheduler.from_config(cfg, **spacing_kw),
        "euler_cfg_pp": lambda: EulerDiscreteScheduler.from_config(cfg, **sigma_kw, **spacing_kw),
        "heun": lambda: HeunDiscreteScheduler.from_config(cfg, **sigma_kw),
        "lms": lambda: LMSDiscreteScheduler.from_config(cfg, **sigma_kw),
        "ddim": lambda: DDIMScheduler.from_config(cfg),
        "ddpm": lambda: DDPMScheduler.from_config(cfg),
        "deis": lambda: DEISMultistepScheduler.from_config(cfg),
        "uni_pc": lambda: UniPCMultistepScheduler.from_config(cfg, **sigma_kw),
        "uni_pc_bh2": lambda: UniPCMultistepScheduler.from_config(cfg, solver_type="bh2", **sigma_kw),
        "dpmpp_2m": lambda: DPMSolverMultistepScheduler.from_config(cfg, **sigma_kw),
        "dpmpp_2m_sde": lambda: DPMSolverMultistepScheduler.from_config(cfg, algorithm_type="sde-dpmsolver++", **sigma_kw),
        "dpmpp_3m_sde": lambda: DPMSolverMultistepScheduler.from_config(cfg, algorithm_type="sde-dpmsolver++", solver_order=3, **sigma_kw),
        "dpmpp_sde": lambda: DPMSolverSinglestepScheduler.from_config(cfg, **sigma_kw),
        "dpmpp_2s_ancestral": lambda: DPMSolverSinglestepScheduler.from_config(cfg, **sigma_kw),
        "lcm": lambda: LCMScheduler.from_config(cfg),
    }
    factory = table.get(name)
    if factory is None:
        return None
    try:
        return factory()
    except Exception:
        # Some sigma/spacing combos aren't valid for every scheduler — fall back.
        return None


class _SdxlLoraHandle:
    def __init__(self, pipe, names):
        self.pipe = pipe
        self.names = names
        self.matched = len(names)
        self.skipped: list[str] = []

    def unmerge(self):
        try:
            self.pipe.delete_adapters(self.names)
        except Exception:
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass


class SdxlSampler(Sampler):
    family = "sdxl"

    def __init__(self):
        self.pipe = None
        self._device = None
        self._dtype = None
        self._prompts: list = []
        self._lora_counter = 0

    def load(self, files: ModelFiles, device: str = "cuda", dtype: str = "float16") -> None:
        import torch
        from diffusers import StableDiffusionXLPipeline, AutoencoderKL

        if not files.model_path:
            raise ValueError("SDXL needs a model / checkpoint path.")
        td = getattr(torch, _DTYPES.get(dtype, "float16"))
        self._device, self._dtype = device, td

        mp = files.model_path
        if mp.endswith(".safetensors") and os.path.isfile(mp):
            pipe = StableDiffusionXLPipeline.from_single_file(mp, torch_dtype=td)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(mp, torch_dtype=td)

        # Optional external VAE (SDXL usually has one baked in).
        if files.vae_path:
            vp = files.vae_path
            if vp.endswith(".safetensors") and os.path.isfile(vp):
                pipe.vae = AutoencoderKL.from_single_file(vp, torch_dtype=td)
            elif os.path.isdir(vp):
                pipe.vae = AutoencoderKL.from_pretrained(vp, torch_dtype=td)

        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        self.pipe = pipe

    def prepare(self, prompts: list[tuple[str, str]]) -> None:
        self._prompts = list(prompts)

    def apply_loras(self, loras: list[tuple[str, float]]):
        names, weights = [], []
        for path, w in loras:
            if not path:
                continue
            nm = f"lora{self._lora_counter}"
            self._lora_counter += 1
            try:
                self.pipe.load_lora_weights(path, adapter_name=nm)
                names.append(nm)
                weights.append(float(w))
            except Exception as e:
                print(f"SDXL: failed to load LoRA {path}: {e}", flush=True)
        if names:
            self.pipe.set_adapters(names, adapter_weights=weights)
        return _SdxlLoraHandle(self.pipe, names)

    def generate(self, index: int, settings: SamplerSettings):
        import torch
        sch = _build_scheduler(self.pipe, settings.name, settings.scheduler)
        if sch is not None:
            self.pipe.scheduler = sch
        prompt, negative = self._prompts[index]
        generator = torch.Generator(self._device).manual_seed(int(settings.seed))
        out = self.pipe(
            prompt=prompt,
            negative_prompt=(negative or None),
            num_inference_steps=int(settings.steps),
            guidance_scale=float(settings.cfg),
            width=int(settings.width),
            height=int(settings.height),
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

"""Native sample-generation backends, one per model family.

Samplers are imported lazily so the rest of the app keeps running even when the
heavy inference dependencies (torch / transformers / diffusers) aren't installed.
"""

from samplers.base import Sampler, SamplerSettings, ModelFiles

# Model-family key -> "module:Class" of its Sampler implementation.
_SAMPLER_REGISTRY: dict[str, str] = {
    "sdxl": "samplers.sdxl.sampler:SdxlSampler",
    "pony": "samplers.sdxl.sampler:SdxlSampler",
    "illustrious": "samplers.sdxl.sampler:SdxlSampler",
    "noobai": "samplers.sdxl.sampler:SdxlSampler",
    "zimage_base": "samplers.zimage.sampler:ZImageSampler",
    "zimage_turbo": "samplers.zimage.sampler:ZImageSampler",
    "krea2": "samplers.krea2.sampler:Krea2Sampler",
}


def supported_families() -> set[str]:
    return set(_SAMPLER_REGISTRY)


def is_supported(family: str) -> bool:
    return family in _SAMPLER_REGISTRY


def get_supported_options(family: str) -> dict | None:
    """Return the sampler/scheduler names a family's backend actually supports,
    as ``{"samplers": [...], "schedulers": [...]}``. Read from the Sampler class's
    ``SUPPORTED_SAMPLERS`` / ``SUPPORTED_SCHEDULERS`` attributes without running any
    heavy inference imports. Returns None if the family or its module is unavailable.
    """
    target = _SAMPLER_REGISTRY.get(family)
    if target is None:
        return None
    module_path, class_name = target.split(":")
    import importlib
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except Exception:
        return None
    return {
        "samplers": list(getattr(cls, "SUPPORTED_SAMPLERS", [])),
        "schedulers": list(getattr(cls, "SUPPORTED_SCHEDULERS", [])),
    }


def get_sampler(family: str) -> Sampler:
    """Instantiate the sampler for a model family.

    Raises NotImplementedError for families without a native sampler yet, and
    surfaces a clear message if inference dependencies are missing.
    """
    target = _SAMPLER_REGISTRY.get(family)
    if target is None:
        raise NotImplementedError(
            f"Native sampling for '{family}' is not implemented yet."
        )
    module_path, class_name = target.split(":")
    import importlib
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise RuntimeError(
            "Sample generation needs extra dependencies. Install them with:\n"
            "    pip install torch transformers diffusers safetensors einops accelerate\n"
            f"(import error: {e})"
        ) from e
    return getattr(module, class_name)()

"""Sample-generation orchestration.

Given a generation request, this loads the right native sampler once, then walks
every saved LoRA checkpoint of the run and renders each prompt — merging the
trained checkpoint (+ any extra LoRAs) in and out around each step — writing the
images into the trainer's samples-folder layout.

Image filenames follow AI Toolkit's pattern ``{ts}__{step:09d}_{idx}.png`` so the
generated samples are themselves readable by the evaluator afterwards.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable


def run_sample_generation(
    trainer_module,
    family: str,
    run_dir: str,
    config_file: str,
    model_files,            # samplers.base.ModelFiles
    settings,               # samplers.base.SamplerSettings
    prompts: list,          # list of objects with .prompt / .negative_prompt
    extra_loras: list,      # list of (path, weight)
    device: str = "cuda",
    dtype: str = "bfloat16",
    progress: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    from samplers import get_sampler

    def _stop() -> bool:
        return bool(should_stop and should_stop())

    def report(**kw):
        if progress:
            progress(kw)

    if not hasattr(trainer_module, "get_checkpoints_for_run"):
        raise RuntimeError(f"Trainer '{getattr(trainer_module, '__name__', '?')}' "
                           "does not support sample generation yet.")

    checkpoints = trainer_module.get_checkpoints_for_run(run_dir, config_file)
    if not checkpoints:
        raise RuntimeError("No saved LoRA checkpoints found for this run.")

    out_dir: Path = trainer_module.get_samples_output_dir(run_dir, config_file)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = sorted(checkpoints)
    total = len(steps) * len(prompts)
    report(phase="loading", current=0, total=total, label="Loading model…")

    sampler = get_sampler(family)
    sampler.load(model_files, device=device, dtype=dtype)

    # Pre-encode all prompts so the text encoder can be freed before sampling.
    report(phase="loading", current=0, total=total, label="Encoding prompts…")
    sampler.prepare([(p.prompt, getattr(p, "negative_prompt", "")) for p in prompts])

    done = 0
    written: list[str] = []
    skipped_lora_warned = False
    stopped = False
    try:
        for step in steps:
            if _stop():
                stopped = True
                break
            specs = [(str(checkpoints[step]), 1.0)] + [(p, w) for p, w in extra_loras]
            handle = sampler.apply_loras(specs)
            if handle is not None and getattr(handle, "skipped", None) and not skipped_lora_warned:
                # Surface once if a LoRA's keys didn't match the model.
                report(phase="warn", label=f"{len(handle.skipped)} LoRA module(s) didn't match the model")
                skipped_lora_warned = True
            try:
                for pidx, p in enumerate(prompts):
                    if _stop():
                        stopped = True
                        break
                    report(phase="sampling", current=done, total=total,
                            label=f"step {step} · prompt {pidx + 1}/{len(prompts)}", step=step)
                    image = sampler.generate(pidx, settings)
                    ts = time.strftime("%Y%m%d%H%M%S")
                    fname = f"{ts}__{step:09d}_{pidx}.png"
                    fpath = out_dir / fname
                    image.save(fpath)
                    written.append(str(fpath))
                    done += 1
                    report(phase="sampling", current=done, total=total,
                            label=f"step {step} · prompt {pidx + 1}/{len(prompts)}", step=step)
            finally:
                if handle is not None:
                    handle.unmerge()
    finally:
        sampler.unload()

    report(phase="stopped" if stopped else "done", current=done, total=total,
           label="Stopped" if stopped else "Done")
    return {
        "status": "stopped" if stopped else "ok",
        "generated": done,
        "output_dir": str(out_dir),
        "steps": steps,
        "prompts": len(prompts),
        "files": written,
    }

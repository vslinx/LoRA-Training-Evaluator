"""Sample-generation orchestration.

Given a generation request, this loads the right native sampler once, then walks
every saved LoRA checkpoint of the run and renders each prompt — merging the
trained checkpoint (+ any extra LoRAs) in and out around each step — writing the
images into the trainer's samples-folder layout.

Each trainer names and places generated samples in *its own* convention via an
optional ``sample_output_path(run_dir, config_file, step, prompt_idx, settings)``
hook, so the generated images are immediately readable by that trainer's
evaluator. Trainers that don't define the hook fall back to AI Toolkit's flat
``{ts}__{step:09d}_{idx}.png`` pattern.
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
    mode: str = "new",      # "new" = (re)generate everything; "continue" = skip existing
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

    # Map of already-present (step, prompt_idx) -> existing image paths. Used to
    # resume an interrupted run ("continue", skip them) or to replace cleanly
    # ("new", delete the old image before writing the fresh one).
    existing: dict = {}
    if hasattr(trainer_module, "existing_samples"):
        try:
            existing = trainer_module.existing_samples(run_dir, config_file)
        except Exception:
            existing = {}

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
    skipped = 0
    skipped_lora_warned = False
    stopped = False
    try:
        for step in steps:
            if _stop():
                stopped = True
                break
            # In continue-mode, skip the whole checkpoint (and its LoRA merge) if
            # every prompt already has a sample.
            if mode == "continue" and all((step, pidx) in existing for pidx in range(len(prompts))):
                done += len(prompts)
                skipped += len(prompts)
                report(phase="sampling", current=done, total=total,
                        label=f"step {step} · already generated, skipping", step=step)
                continue
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
                    if mode == "continue" and (step, pidx) in existing:
                        done += 1
                        skipped += 1
                        report(phase="sampling", current=done, total=total,
                                label=f"step {step} · prompt {pidx + 1}/{len(prompts)} (skipped)", step=step)
                        continue
                    report(phase="sampling", current=done, total=total,
                            label=f"step {step} · prompt {pidx + 1}/{len(prompts)}", step=step)
                    image = sampler.generate(pidx, settings)
                    if hasattr(trainer_module, "sample_output_path"):
                        fpath = trainer_module.sample_output_path(
                            run_dir, config_file, step, pidx, settings)
                    else:
                        ts = time.strftime("%Y%m%d%H%M%S")
                        fpath = out_dir / f"{ts}__{step:09d}_{pidx}.png"
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    # Regenerating (mode "new"): remove any prior image(s) for this
                    # (step, prompt) so we replace rather than accumulate duplicates.
                    for old in existing.get((step, pidx), []):
                        try:
                            if Path(old) != fpath:
                                Path(old).unlink()
                        except OSError:
                            pass
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
        "generated": len(written),
        "skipped": skipped,
        "output_dir": str(out_dir),
        "steps": steps,
        "prompts": len(prompts),
        "files": written,
    }

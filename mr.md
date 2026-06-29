# Add native Sample Generation + GPU-accelerated comparison fixes

## Overview

This PR adds a **Sample Generation** feature to the evaluator and hardens the
existing comparison pipeline so it reliably runs on the GPU. You can now generate
sample images for every saved LoRA checkpoint of a run — natively, without
ComfyUI — and hand the run straight to the evaluator for ranking. This is
especially useful when you trained with sampling disabled, or want fresh samples
with specific prompts/settings.

The evaluator itself is unchanged in behaviour; the additions live behind a new
**Sample Generation** toggle in the header, and the sampling dependencies are
optional so an evaluator-only checkout stays lean.

## What's new

### Native Sample Generation

A new tab (top-right of the header) that mirrors the evaluator's trainer →
workspace → run selection, then renders samples for every saved checkpoint:

- **Model recognition** — the run's training config is inspected to pre-select
  the base model family and pre-fill model/CLIP/VAE paths, sampler settings and
  prompts where possible.
- **Per-checkpoint sampling** — one image is rendered per *checkpoint × prompt*;
  the trained LoRA for each step (plus any extra "LoRAs for sampling" such as a
  turbo/DMD2 LoRA) is applied, and outputs are written in **each trainer's own
  samples-folder layout and filename convention** so the evaluator reads them
  immediately (see "Trainer-aware sample output" below).
- **Live progress + Stop** — a progress bar polls the backend; a **Stop** button
  cancels mid-run and offers to delete the partial samples.
- **Resume an interrupted run** — before generating, the run is checked for
  samples that already exist. If any are found (e.g. a previous run was cut short
  by a crash or power loss), you're asked whether to **Continue** — skipping the
  checkpoints/prompts already rendered and generating only what's missing — or
  **Regenerate all**, which replaces existing images cleanly (no duplicates).
- **Evaluate hand-off** — once a run finishes, an **Evaluate →** button jumps to
  the evaluator pre-filled with that trainer/workspace/run, ready to compare.
- **Remembered settings + Prompt Gallery** — model files, sampling LoRAs and
  (optionally, via a prompt on Generate) sampler settings are persisted per model
  family in a gitignored `config/sample_settings.json`. Reusable prompts can be
  saved to a **Prompt Gallery** and pulled into any run with one click.

**Supported model families**

| Model family | Status |
|--------------|--------|
| SDXL / Pony / Illustrious / NoobAI | Supported (diffusers `StableDiffusionXLPipeline`) |
| Z-Image (Base / Turbo) | Supported (diffusers `ZImagePipeline`) |
| Krea2 | Experimental / WIP — loads, runs and saves end-to-end, but output quality is not yet validated. Included intentionally; **not** considered done. |
| Anima | Planned |

**Supported trainers for sampling**: OneTrainer, AI Toolkit and Anima all expose
their saved checkpoints, recognized model, prompts and output layout to the
sampler. (`kohya_ss` / `musubi_tuner` remain listed as planned.)

SDXL highlights:
- Loads all-in-one single-file checkpoints; CLIP/VAE are optional (baked in).
- LoRAs applied via diffusers' adapter API (PEFT), including the per-step trained
  checkpoint and optional extra LoRAs, cleanly added/removed between checkpoints.
- Only sampler/scheduler combinations that map to a real diffusers scheduler are
  shown in the dropdowns. DMD2 few-step LoRAs use the **lcm** sampler.

Z-Image highlights:
- Flow-matching model loadable two ways: a **single `.safetensors`** transformer
  (ComfyUI `diffusion_model` format) paired with a single-file VAE and a Qwen3
  text encoder as either `.safetensors` **or `.gguf`** — the model config and
  tokenizer are vendored, so no diffusers folder is required — **or** a full
  **diffusers folder** with `transformer/`, `text_encoder/`, `tokenizer/`, `vae/`.
- The **Shift** field is honored by `euler` and `uni_pc`; `dpmpp_2m` uses
  diffusers' resolution-derived shift. The sampler dropdown offers `euler`,
  `dpmpp_2m` and `uni_pc`, with `beta`/`karras` sigma spacing limited to the
  cases where it produces sane flow sigmas.

Per-family sampler/scheduler dropdowns are now served by
`/api/sampler-options?model=...`, so the UI only lists what the selected family
can actually run instead of the full ComfyUI list.

### Trainer-aware sample output

Generated images are named and placed by a per-trainer hook
(`sample_output_path`) so each trainer's evaluator can read its own samples back:

- **OneTrainer** writes timestamped `…-training-sample-{step}-{epoch}-{idx}` files
  into per-prompt subfolders, with the timestamp anchored to the run's config
  window so samples are attributed to the correct run even when newer runs exist.
- **AI Toolkit** keeps its flat `{ts}__{step:09d}_{idx}` convention.
- **Anima** writes `{name}_{step:06d}_{idx:02d}_{ts}_{seed}` into `output/sample/`.

Previously all trainers received AI Toolkit's naming, so OneTrainer and Anima runs
generated images the evaluator couldn't find. The same per-(step, prompt) index is
what powers the resume/skip logic above.

Architecture: an arch-agnostic orchestrator (`sampling.py`) iterates checkpoints
× prompts and handles saving/progress/cancellation/resume, while pluggable
`Sampler` backends under `samplers/` implement each model family. Heavy ML imports
are lazy, so the app and evaluator run fine without the sampling extras installed.

### Trainer fixes surfaced by sampling

- **Final checkpoint included.** AI Toolkit's last save drops the step suffix
  (`{name}.safetensors` instead of `{name}_{step:09d}.safetensors`); it's now
  resolved to the configured total step count so the final checkpoint is analyzed
  and sampled like the rest.
- **Optional negative prompt.** Generating no longer requires typing (then
  clearing) a negative prompt — the form re-validates whenever prompts change, so
  an empty negative works (e.g. CFG 1 on a turbo model/LoRA).

### GPU-accelerated comparison fixes

- **onnxruntime now actually uses CUDA.** `onnxruntime-gpu` is pinned to a
  CUDA-12 build (`1.22.0`); newer releases target CUDA 13 and silently fall back
  to CPU on CUDA-12 systems. The startup hook also adds `torch/lib` to `PATH`
  (it bundles the full CUDA-12 DLL set, including `cudart`, which the previous
  hook missed) so the CUDA execution provider loads instead of falling back.
  Result: face detection + ArcFace recognition run on the GPU.

### Cleanup & docs

- Quieted noisy startup/inference warnings (Linux-only `expandable_segments`
  skipped on Windows; non-writable NumPy array from GGUF dequant copied first).
- Removed the dead single-run `/api/run` path (the UI has always used
  `/api/run-multi`) and other unused definitions.
- Split dependencies into core `requirements.txt` (the evaluator) and optional
  `requirements-sampling.txt` (Sample Generation), so a default checkout is light.
- README updated with the Sample Generation workflow, supported models/trainers,
  Z-Image single-file/GGUF loading, the Prompt Gallery, resume behaviour, the
  optional install step, and the onnxruntime/CUDA notes.

## Dependencies & setup

- Core install is unchanged: `run.bat` (PyTorch cu126 + `requirements.txt`).
- Sample Generation is opt-in:
  ```
  venv\Scripts\activate
  pip install -r requirements-sampling.txt
  ```
  These are pinned to a known-good combo (**transformers 5.5.3** +
  **diffusers** from a specific git commit + **peft** + **gguf**). The released
  diffusers + newest transformers break SDXL single-file loading (`CLIPTextModel`
  was refactored); transformers 5.5.3 keeps the structure diffusers needs and
  still ships Qwen3-VL. The evaluator does not require any of these.

## Notes / known limitations

- **Krea2 is experimental** and intentionally left in this PR (not merged as a
  finished feature). It loads (incl. INT8 weight-only / fp8), applies LoRAs,
  encodes via Qwen3-VL, and decodes via the Qwen-Image / Wan VAEs, but the
  generated image quality still needs validation against a reference. SDXL and
  Z-Image are the production-ready paths.
- **Z-Image's `dpmpp_2m` / `uni_pc` samplers** are wired and produce sane flow
  sigmas; `euler` is the visually-validated path.
- Resume matches existing samples by `(step, prompt index)`, so it assumes the
  prompt list is in the same order as the interrupted run.
- `kohya_ss` and `musubi_tuner` remain listed as planned trainers.
- Do not install plain `onnxruntime` alongside `onnxruntime-gpu` — they share a
  namespace and the CPU build disables the CUDA provider.

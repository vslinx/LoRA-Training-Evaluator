# LoRA Training Evaluator

Automatically compare face likeness or style similarity across LoRA/checkpoint training steps to find the best iteration — no more manual image-by-image comparison. Optionally **generate sample images natively** for every saved checkpoint first, then evaluate them.

Two comparison modes are available:
- **Person's Likeness** — extracts face embeddings (ArcFace) from your training dataset and compares them against sample images using cosine similarity
- **Style Similarity** — extracts style embeddings (CSD) from your training dataset and compares artistic style consistency across steps

Results are displayed as a ranked tier list with per-image breakdowns and a similarity-over-steps chart.

A separate **Sample Generation** tab (top-right of the header) can render sample images for every saved LoRA checkpoint of a run — useful when you trained with sampling disabled — and then hand the run straight to the evaluator. See [Sample Generation](#sample-generation).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-orange)
![CSD](https://img.shields.io/badge/CSD-Style_Similarity-purple)

## Features

- **Two comparison modes** — Person's Likeness (face) or Style Similarity (artistic style)
- **Multi-run comparison** — select multiple training runs to compare different settings side by side
- **Guided wizard UI** — step-by-step setup: select trainer, browse workspace, pick training runs, choose comparison mode, choose steps
- **Native folder picker** — no manual path typing needed
- **Auto-detection** — dataset path and sample mappings are parsed from training configs
- **Tier list results** — S/A/B/C/D ranking with expandable image previews, collapsible with "show more"
- **Multi-line similarity chart** — each run gets its own colored line for easy visual comparison
- **GPU accelerated** — uses CUDA for both face analysis (onnxruntime-gpu) and style analysis (PyTorch), falls back to CPU
- **Modular trainer support** — pluggable architecture for different training tools
- **Native sample generation** *(optional)* — generate samples for every saved checkpoint of a run with your own model/sampler/prompts, with live progress, a Stop button, and a one-click hand-off to the evaluator

## Supported Trainers

| Trainer | Status |
|---------|--------|
| OneTrainer | Fully supported |
| AI Toolkit | Fully supported |
| Anima Standalone Trainer | Fully supported |
| Kohya SS | Planned |
| MusubiTuner | Planned |

## Requirements

- Python 3.10+
- Windows (uses native folder dialogs via tkinter)
- NVIDIA GPU recommended (CUDA for GPU acceleration)

## Quick Start

1. **Clone or download** this repository

2. **Run the application:**
   ```
   run.bat
   ```
   This automatically creates a virtual environment, installs PyTorch with CUDA support, installs remaining dependencies, and starts the server.

3. **Open** http://127.0.0.1:8384 in your browser

### GPU Acceleration

`run.bat` installs PyTorch with CUDA 12.6 support automatically. The default `requirements.txt` also installs `onnxruntime-gpu` for face detection.

> **Important:** `onnxruntime-gpu` is pinned to `1.22.0` (a CUDA-12 build). Newer
> releases (1.27+) target CUDA 13 and silently fall back to CPU on a CUDA-12
> setup. Also never install plain `onnxruntime` alongside it — they share a
> namespace and the CPU build disables the CUDA provider. The app adds
> `torch/lib` to `PATH` at startup so onnxruntime finds the CUDA-12 DLLs.

If you don't have an NVIDIA GPU, replace `onnxruntime-gpu` with `onnxruntime` in `requirements.txt` and change the `pip install torch torchvision --index-url ...` line in `run.bat` to `pip install torch torchvision` before running.

### Optional: Sample Generation

The Sample Generation tab needs extra (heavy) dependencies. After the first
`run.bat`, install them into the venv:

```
venv\Scripts\activate
pip install -r requirements-sampling.txt
```

The evaluator works without these; the Sample Generation tab will show a clear
"install the extras" message until they're present. See [Sample Generation](#sample-generation).

## Usage (OneTrainer)

1. Select **OneTrainer** from the trainer dropdown
2. Click **Browse** and select your `workspace\run` folder
3. Pick one or **multiple training runs** from the list (click to toggle selection) — dataset paths are auto-detected from configs
4. Choose **comparison mode** — Person's Likeness or Style Similarity
5. Choose which **steps** to compare (all or a specific range)
6. Click **Run Comparison**

## Usage (AI Toolkit)

1. Select **AI Toolkit** from the trainer dropdown
2. Click **Browse** and select the `output/` folder (e.g., `ai-toolkit/output`) — each subfolder is detected as a separate run
3. Pick one or **multiple training runs** from the list — dataset paths and sample mappings are auto-detected from each run's `config.yaml`
4. Choose **comparison mode** — Person's Likeness or Style Similarity
5. Choose which **steps** to compare (all or a specific range)
6. Click **Run Comparison**

## Usage (Anima Standalone Trainer)

1. Select **Anima Standalone Trainer** from the trainer dropdown
2. Click **Browse** and select the `training-ui/jobs` folder
3. Pick one or **multiple training runs** from the list — dataset paths are auto-detected from each run's `dataset.toml`
4. Choose **comparison mode** — Person's Likeness or Style Similarity
5. Choose which **steps** to compare (all or a specific range)
6. Click **Run Comparison**

## Sample Generation

If you trained with sampling disabled (or want fresh samples with specific
prompts/settings), the **Sample Generation** tab renders an image for every
saved LoRA checkpoint of a run, natively (no ComfyUI), then lets you jump
straight into the evaluator.

**Requires the optional extras** — `pip install -r requirements-sampling.txt`.

1. Click **Sample Generation** (top-right of the header)
2. Pick **trainer → workspace → training run** (same as the evaluator); the run's
   config is inspected to recognize the base model
3. Confirm the **model family**, set the **model / CLIP / VAE** files, **sampler
   settings**, and **prompts** (add as many as you like). Model files, LoRAs, and
   (optionally) sampler settings are remembered per model family. Reusable prompts
   can be saved to the **Prompt Gallery** and pulled into any run with one click
4. Optionally add **LoRAs for sampling** (e.g. a turbo/DMD2 LoRA)
5. Click **Generate Samples** — one image is rendered per checkpoint × prompt and
   written into the run's samples folder. A **Stop** button cancels mid-run (with
   an option to delete the partial samples). If samples for this run already exist
   (e.g. an earlier run was interrupted), you're asked whether to **Continue**
   (skip what's already there and render only the missing images) or **Regenerate
   all** (replace everything)
6. Click **Evaluate →** to open the evaluator pre-filled with that run

### Supported models

| Model family | Status |
|--------------|--------|
| SDXL / Pony / Illustrious / NoobAI | Supported (via diffusers) |
| Z-Image (Base/Turbo) | Supported (via diffusers `ZImagePipeline`) |
| Krea2 | Supported (single-file MMDiT incl. fp8/int8 weight-only; Qwen3-VL text encoder + Qwen-Image VAE) |
| Anima | Planned |

Notes:
- SDXL loads all-in-one single-file checkpoints; CLIP/VAE are optional (baked in).
  Only sampler/scheduler combinations that map to a real diffusers scheduler are
  shown in the dropdowns. DMD2 few-step LoRAs need the **lcm** sampler.
- Z-Image is a flow-matching model and loads two ways:
  - **Single file** — point **Model** at a single `.safetensors` transformer
    (ComfyUI `diffusion_model` format), **VAE** at a single `.safetensors` VAE,
    and **CLIP** at a single `.safetensors` *or* `.gguf` Qwen3 text encoder. The
    model config + tokenizer are bundled (`samplers/zimage/assets/`), so no
    diffusers folder is needed. (Override the Qwen3 config/tokenizer with the
    `ZIMAGE_TE_REPO` env var if you use a different Qwen3.)
  - **Diffusers folder** — point **Model** at the base folder (with
    `transformer/`, `text_encoder/`, `tokenizer/`, `vae/` subfolders); CLIP/VAE
    are taken from the folder.
  Set the **Shift** field (e.g. 6 for Z-Image Turbo); it's honored by `euler`
  and `uni_pc` (`dpmpp_2m` uses diffusers' resolution-derived shift). The
  **sampler** dropdown offers `euler` (FlowMatch Euler), `dpmpp_2m` (DPMSolver)
  and `uni_pc` (UniPC); the **scheduler** dropdown is the sigma spacing
  (`normal`/`beta`/`karras`). `beta`/`karras` only apply to `euler` — the
  multistep solvers fall back to `normal` (other spacings blow up flow sigmas).
- The sampler/scheduler dropdowns now list **only what the selected model family
  supports** (served per-model by `/api/sampler-options?model=...`).
- Generated samples are written with the trainer's filename convention, so they
  are immediately readable by the evaluator.
- Per-family settings persist in `config/sample_settings.json` (gitignored).

## How It Works

### Person's Likeness Mode
1. **Face Detection** — InsightFace's RetinaFace detector finds faces in each image
2. **Embedding Extraction** — ArcFace (buffalo_l model) generates a 512-dimensional face embedding
3. **Reference Identity** — All dataset face embeddings are averaged into a centroid vector
4. **Comparison** — Each sample's face embedding is compared to the centroid via cosine similarity
5. **Ranking** — Steps are ranked by average similarity across all sample images

### Style Similarity Mode
1. **Style Extraction** — CSD (Contrastive Style Descriptors) ViT-L/14 model extracts style embeddings from each image
2. **Reference Style** — All dataset style embeddings are averaged into a centroid vector
3. **Comparison** — Each sample's style embedding is compared to the centroid via cosine similarity
4. **Ranking** — Steps are ranked by average style similarity across all sample images

CSD model weights (~2.4 GB) are downloaded automatically from HuggingFace on first use and cached locally.

## Project Structure

```
LoRA Training Evaluator/
  app.py                  — FastAPI backend (API endpoints, progress tracking)
  face_analyzer.py        — Face detection and embedding comparison (InsightFace/ArcFace)
  style_analyzer.py       — Style embedding extraction and comparison (CSD)
  sampling.py             — Sample-generation orchestrator (iterate checkpoints × prompts)
  trainers/
    __init__.py           — Trainer registry, shared data models, sample-model registry
    onetrainer.py         — OneTrainer config parsing and sample mapping
    aitoolkit.py          — AI Toolkit config parsing and sample mapping
    anima.py              — Anima Standalone Trainer config parsing and sample mapping
  samplers/               — Native sample-generation backends (optional feature)
    base.py               — Sampler interface + settings dataclasses
    lora.py               — LoRA loading / merging (PEFT + kohya formats)
    sdxl/sampler.py       — SDXL/Pony/Illustrious/NoobAI sampler (diffusers)
    zimage/               — Z-Image (Base/Turbo) sampler; single-file + folder
                            loading, GGUF/safetensors Qwen3 TE, vendored configs
    krea2/                — Krea2 sampler (single-file MMDiT, fp8/int8; vendored arch, see NOTICE.md)
  static/
    index.html            — Web UI (single-page app)
  config/                 — persisted per-family sample settings (gitignored)
  requirements.txt        — core dependencies (the evaluator)
  requirements-sampling.txt — optional dependencies (Sample Generation)
  run.bat                 — One-click launcher (venv + dependencies + server)
```

## Adding a New Trainer

Create a new file in `trainers/` (e.g., `trainers/kohya.py`). For the **evaluator**:

- `validate_workspace(path)` — check if a folder is a valid workspace
- `list_configs(path)` — return a list of `TrainingRun` objects
- `get_samples_for_run(path, config)` — return `dict[int, list[Path]]` mapping step numbers to image paths
- `get_dataset_path(path, config)` — return the reference dataset path

To also support **Sample Generation** for that trainer, add:

- `validate_workspace_sampling(path)` — lenient validation (samples may not exist yet)
- `list_runs_for_sampling(path)` — runs with their saved-checkpoint step counts
- `get_checkpoints_for_run(path, config)` — `dict[int, Path]` mapping steps to checkpoint files
- `get_samples_output_dir(path, config)` — where generated samples should be written
- `inspect_for_sampling(path, config)` — recognize the model + pre-fill sampler/prompts

Then register it in `TRAINER_MODULES` in `app.py` and `TRAINERS` in `trainers/__init__.py`.

To add a new **model family** for sampling, implement a `Sampler` subclass under
`samplers/` and register it in `samplers/__init__.py`.

# LoRA Training Evaluator

Automatically compare face likeness or style similarity across LoRA/checkpoint training steps to find the best iteration — no more manual image-by-image comparison.

Two comparison modes are available:
- **Person's Likeness** — extracts face embeddings (ArcFace) from your training dataset and compares them against sample images using cosine similarity
- **Style Similarity** — extracts style embeddings (CSD) from your training dataset and compares artistic style consistency across steps

Results are displayed as a ranked tier list with per-image breakdowns and a similarity-over-steps chart.

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

3. **Open** http://127.0.0.1:8000 in your browser

### GPU Acceleration

`run.bat` installs PyTorch with CUDA 12.6 support automatically. The default `requirements.txt` also installs `onnxruntime-gpu` for face detection.

If you don't have an NVIDIA GPU, replace `onnxruntime-gpu` with `onnxruntime` in `requirements.txt` and change the `pip install torch torchvision --index-url ...` line in `run.bat` to `pip install torch torchvision` before running.

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
  app.py              — FastAPI backend (API endpoints, progress tracking)
  face_analyzer.py    — Face detection and embedding comparison (InsightFace/ArcFace)
  style_analyzer.py   — Style embedding extraction and comparison (CSD)
  trainers/
    __init__.py       — Trainer registry and shared data models
    onetrainer.py     — OneTrainer config parsing and sample mapping
    aitoolkit.py      — AI Toolkit config parsing and sample mapping
    anima.py          — Anima Standalone Trainer config parsing and sample mapping
  static/
    index.html        — Web UI (single-page app)
  requirements.txt
  run.bat             — One-click launcher (venv + dependencies + server)
```

## Adding a New Trainer

Create a new file in `trainers/` (e.g., `trainers/kohya.py`) implementing:

- `validate_workspace(path)` — check if a folder is a valid workspace
- `list_configs(path)` — return a list of `TrainingRun` objects
- `get_samples_for_run(path, config)` — return `dict[int, list[Path]]` mapping step numbers to image paths
- `get_dataset_path(path, config)` — return the reference dataset path

Then register it in `trainers/__init__.py` and add the routing in `app.py`.

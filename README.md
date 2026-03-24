# LoRA Training Evaluator

Automatically compare face likeness across LoRA/checkpoint training steps to find the best iteration — no more manual image-by-image comparison.

The tool extracts face embeddings from your training dataset (reference identity) and compares them against sample images generated at each training step using ArcFace cosine similarity. Results are displayed as a ranked tier list with per-image breakdowns and a similarity-over-steps chart.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-orange)

## Features

- **Multi-run comparison** — select multiple training runs to compare different settings side by side
- **Guided wizard UI** — step-by-step setup: select trainer, browse workspace, pick training runs, choose steps
- **Native folder picker** — no manual path typing needed
- **Auto-detection** — dataset path and sample mappings are parsed from training configs
- **Tier list results** — S/A/B/C/D ranking with expandable image previews, collapsible with "show more"
- **Multi-line similarity chart** — each run gets its own colored line for easy visual comparison
- **GPU accelerated** — uses CUDA via onnxruntime-gpu when available, falls back to CPU
- **Modular trainer support** — pluggable architecture for different training tools

## Supported Trainers

| Trainer | Status |
|---------|--------|
| OneTrainer | Fully supported |
| AI Toolkit | Planned |
| Kohya SS | Planned |
| MusubiTuner | Planned |

## Requirements

- Python 3.10+
- Windows (uses native folder dialogs via tkinter)
- NVIDIA GPU recommended (CUDA + cuDNN for GPU acceleration)

## Quick Start

1. **Clone or download** this repository

2. **Run the application:**
   ```
   run.bat
   ```
   This automatically creates a virtual environment, installs dependencies, and starts the server.

3. **Open** http://127.0.0.1:8000 in your browser

### GPU Acceleration

The default `requirements.txt` installs `onnxruntime-gpu`. If you don't have an NVIDIA GPU, replace it with `onnxruntime` in `requirements.txt` before running.

GPU support dependencies (cuDNN and cuBLAS) are included in `requirements.txt` and installed automatically.

## Usage (OneTrainer)

1. Select **OneTrainer** from the trainer dropdown
2. Click **Browse** and select your `workspace\run` folder
3. Pick one or **multiple training runs** from the list (click to toggle selection) — dataset paths are auto-detected from configs
4. Choose which **steps** to compare (all or a specific range)
5. Click **Run Comparison**

The tool will:
- Extract face embeddings from your reference dataset (cached if shared across runs)
- For each selected run and step, collect sample images across all prompt folders
- Compare each sample face against the reference identity
- Rank all steps across all runs by average similarity
- Display a combined tier list and a chart with one colored line per run

## How It Works

1. **Face Detection** — InsightFace's RetinaFace detector finds faces in each image
2. **Embedding Extraction** — ArcFace (buffalo_l model) generates a 512-dimensional face embedding
3. **Reference Identity** — All dataset face embeddings are averaged into a centroid vector
4. **Comparison** — Each sample's face embedding is compared to the centroid via cosine similarity
5. **Ranking** — Steps are ranked by average similarity across all sample images

## Project Structure

```
LoRA Training Evaluator/
  app.py              — FastAPI backend (API endpoints, progress tracking)
  face_analyzer.py    — Face detection and embedding comparison (InsightFace/ArcFace)
  trainers/
    __init__.py       — Trainer registry and shared data models
    onetrainer.py     — OneTrainer config parsing and sample mapping
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

## License

MIT

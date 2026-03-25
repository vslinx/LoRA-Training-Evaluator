"""Style analysis module using CSD (Contrastive Style Descriptors) for style comparison.

Uses the CSD-ViT-L model from https://github.com/learn2phoenix/CSD
Weights are auto-downloaded from HuggingFace on first use.
"""

import copy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Standard OpenAI CLIP image preprocessing
_CLIP_PREPROCESS = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


# ── CLIP VisionTransformer (matches OpenAI CLIP architecture for state_dict compat) ──

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln_1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


# ── CSD Model ────────────────────────────────────────────────────────────────

class CSD_CLIP(nn.Module):
    """CSD: CLIP ViT-L/14 backbone with separate style and content projection heads."""

    def __init__(self):
        super().__init__()
        self.backbone = VisionTransformer(
            input_resolution=224, patch_size=14, width=1024,
            layers=24, heads=16, output_dim=768,
        )
        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        self.last_layer_content = copy.deepcopy(self.backbone.proj)
        self.backbone.proj = None

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        style = features @ self.last_layer_style
        style = F.normalize(style, dim=-1)
        content = features @ self.last_layer_content
        content = F.normalize(content, dim=-1)
        return features, content, style


# ── Style Analyzer ────────────────────────────────────────────────────────────

def _load_and_preprocess(image_path: Path) -> tuple[Path, torch.Tensor | None]:
    """Load and preprocess a single image (runs on thread pool)."""
    try:
        img = Image.open(str(image_path)).convert("RGB")
        return image_path, _CLIP_PREPROCESS(img)
    except Exception:
        return image_path, None


class StyleAnalyzer:
    """Analyze artistic style similarity using CSD embeddings."""

    HF_REPO = "tomg-group-umd/CSD-ViT-L"
    HF_FILENAME = "pytorch_model.bin"
    BATCH_SIZE = 16

    def __init__(self):
        self.model = None
        self.device = None
        self.use_fp16 = False

    def initialize(self):
        """Load the CSD model (downloads weights from HuggingFace on first use)."""
        if self.model is not None:
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = self.device.type == "cuda"

        from huggingface_hub import hf_hub_download
        print("Downloading CSD model weights (first run only)...")
        checkpoint_path = hf_hub_download(
            repo_id=self.HF_REPO,
            filename=self.HF_FILENAME,
        )

        self.model = CSD_CLIP()
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        if self.use_fp16:
            self.model.half()
        self.model.eval()
        print(f"CSD model loaded on {self.device}" + (" (fp16)" if self.use_fp16 else ""))

    def _extract_batch(self, tensors: list[torch.Tensor]) -> np.ndarray:
        """Run model on a batch of preprocessed tensors, return style embeddings."""
        batch = torch.stack(tensors).to(self.device)
        if self.use_fp16:
            batch = batch.half()
        with torch.no_grad():
            _, _, style = self.model(batch)
        return style.float().cpu().numpy()

    def get_embeddings_for_paths(self, image_paths: list[Path], progress_callback=None):
        """Extract style embeddings with batched GPU inference and threaded I/O."""
        embeddings = []
        skipped = []

        # Pre-load and preprocess images on thread pool (overlaps with GPU work)
        with ThreadPoolExecutor(max_workers=4) as pool:
            loaded = list(pool.map(_load_and_preprocess, image_paths))

        # Collect valid tensors, tracking original paths
        valid_paths = []
        valid_tensors = []
        for img_path, tensor in loaded:
            if tensor is not None:
                valid_paths.append(img_path)
                valid_tensors.append(tensor)
            else:
                skipped.append(img_path)

        # Process in batches
        total = len(valid_tensors)
        for batch_start in range(0, total, self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, total)
            batch_tensors = valid_tensors[batch_start:batch_end]
            batch_paths = valid_paths[batch_start:batch_end]

            if progress_callback:
                progress_callback(batch_start, total, batch_paths[0].name)

            style_embs = self._extract_batch(batch_tensors)

            for j, (img_path, emb) in enumerate(zip(batch_paths, style_embs)):
                embeddings.append((img_path, emb))

        if progress_callback and total > 0:
            progress_callback(total, total, "done")

        return embeddings, skipped

    def get_folder_embeddings(self, folder_path: str, progress_callback=None):
        """Extract style embeddings from all images in a folder."""
        folder = Path(folder_path)
        image_files = sorted(
            f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        return self.get_embeddings_for_paths(image_files, progress_callback)

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def compare_images_to_reference(
        self,
        reference_embeddings: list[tuple[str, np.ndarray]],
        image_paths: list[Path],
        progress_callback=None,
    ) -> dict:
        """Compare images against the reference style centroid.

        Returns dict with average_similarity, per_image, skipped, sample_count.
        """
        if not reference_embeddings:
            return {"average_similarity": 0.0, "per_image": [], "skipped": [], "sample_count": 0}

        ref_vectors = np.array([emb for _, emb in reference_embeddings])
        centroid = np.mean(ref_vectors, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        sample_embeddings, skipped = self.get_embeddings_for_paths(image_paths, progress_callback)

        per_image = []
        for img_path, emb in sample_embeddings:
            sim = self.cosine_similarity(centroid, emb)
            sim_pct = max(0.0, min(1.0, (sim + 1) / 2))
            per_image.append((str(img_path), img_path.name, sim_pct))

        per_image.sort(key=lambda x: x[1])
        avg = float(np.mean([s for _, _, s in per_image])) if per_image else 0.0

        return {
            "average_similarity": avg,
            "per_image": per_image,
            "skipped": [str(p) for p in skipped],
            "sample_count": len(per_image),
        }

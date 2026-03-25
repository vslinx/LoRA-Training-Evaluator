"""Face analysis module using InsightFace (ArcFace) for face detection and embedding comparison."""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add nvidia cuDNN/cuBLAS DLLs to PATH before importing onnxruntime/insightface
for _pkg in ("cudnn", "cublas"):
    _bin = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / _pkg / "bin"
    if _bin.exists() and str(_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")

import cv2
import numpy as np
import insightface

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


class FaceAnalyzer:
    def __init__(self):
        self.app = None

    def initialize(self):
        """Load the InsightFace model (buffalo_l includes ArcFace for recognition)."""
        if self.app is not None:
            return
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_face_embedding(self, image_path: str) -> np.ndarray | None:
        """Extract the face embedding from an image. Returns the largest detected face."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        faces = self.app.get(img)
        if not faces:
            return None
        # Pick the largest face by bounding box area
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest.embedding

    def get_embeddings_for_paths(self, image_paths: list[Path], progress_callback=None):
        """Extract face embeddings from a list of image paths.

        Uses threaded image preloading so disk I/O overlaps with GPU inference.

        Returns:
            list of (path, embedding) tuples and list of skipped paths.
        """
        embeddings = []
        skipped = []

        # Preload all images on threads while GPU processes them
        def _load(path):
            return path, cv2.imread(str(path))

        with ThreadPoolExecutor(max_workers=4) as pool:
            loaded = list(pool.map(_load, image_paths))

        for i, (img_path, img) in enumerate(loaded):
            if progress_callback:
                progress_callback(i, len(loaded), img_path.name)
            if img is None:
                skipped.append(img_path)
                continue
            faces = self.app.get(img)
            if not faces:
                skipped.append(img_path)
                continue
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embeddings.append((img_path, largest.embedding))

        return embeddings, skipped

    def get_folder_embeddings(self, folder_path: str, progress_callback=None):
        """Extract face embeddings from all images in a folder."""
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
        """Compare a list of images against the reference identity.

        Returns a dict with:
            - average_similarity: mean similarity across all sample images (0-1)
            - per_image: list of (filename, similarity) sorted descending
            - skipped: list of filenames where no face was detected
            - sample_count: number of samples successfully compared
        """
        if not reference_embeddings:
            return {"average_similarity": 0.0, "per_image": [], "skipped": [], "sample_count": 0}

        # Compute reference centroid
        ref_vectors = np.array([emb for _, emb in reference_embeddings])
        centroid = np.mean(ref_vectors, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        sample_embeddings, skipped = self.get_embeddings_for_paths(image_paths, progress_callback)

        per_image = []
        for img_path, emb in sample_embeddings:
            sim = self.cosine_similarity(centroid, emb)
            # Map from [-1, 1] cosine range to [0, 1] for display
            sim_pct = max(0.0, min(1.0, (sim + 1) / 2))
            per_image.append((str(img_path), img_path.name, sim_pct))

        per_image.sort(key=lambda x: x[1])

        avg = float(np.mean([s for _, _, s in per_image])) if per_image else 0.0

        return {
            "average_similarity": avg,
            "per_image": per_image,  # list of (full_path, filename, similarity)
            "skipped": [str(p) for p in skipped],
            "sample_count": len(per_image),
        }

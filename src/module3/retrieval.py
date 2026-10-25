"""Pretrained visual embedding retrieval for chest X-rays."""
from __future__ import annotations

import csv
import gc
from contextlib import contextmanager
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = PROJECT_ROOT.parent / "10000_JPG_IMAGES"
DEFAULT_CSV_PATH = PROJECT_ROOT.parent / "02_dataset" / "cxr_10k_master.csv"
DEFAULT_INDEX_PATH = PROJECT_ROOT / ".runtime" / "module3" / "image_index.npz"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


@contextmanager
def _limited_cpu_threads():
    """Limit temporary CPU workspaces for Module 3 without changing other modules."""
    previous = torch.get_num_threads()
    limited = min(previous, 2)
    if limited != previous:
        torch.set_num_threads(limited)
    try:
        yield
    finally:
        if limited != previous:
            torch.set_num_threads(previous)


def _configured_path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_paths() -> tuple[Path, Path, Path]:
    return (
        _configured_path("CXR_IMAGE_DIR", DEFAULT_IMAGE_DIR),
        _configured_path("CXR_CSV_PATH", DEFAULT_CSV_PATH),
        _configured_path("CXR_M3_INDEX_PATH", DEFAULT_INDEX_PATH),
    )


_LABEL_PREFIX = re.compile(
    r"(?i)\b(?:findings?|impressions?|report|chest\s+x[- ]?ray|"
    r"portable\s+chest|exam(?:ination)?|comparison|clinical\s+history)\s*:\s*"
)
_CLUTTER = re.compile(r"(?i)\b(?:findings?|impressions?|report|exam(?:ination)?)\s*:\s*")


def clean_text(value: Any) -> str:
    """Remove report formatting and empty placeholders without rewriting findings."""
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"_{2,}|-{3,}|\.{3,}", " ", text)
    text = _LABEL_PREFIX.sub("", text)
    text = _CLUTTER.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" \t;|-")
    return text


def load_encoder():
    """Load the frozen pretrained encoder on CPU; no training is performed."""
    import open_clip

    gc.collect()
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device="cpu"
    )
    model.eval()
    return model, preprocess


def _embed_images(model, preprocess, paths: list[Path], batch_size: int = 16) -> np.ndarray:
    vectors: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    with _limited_cpu_threads(), torch.inference_mode():
        for path in paths:
            with Image.open(path) as image:
                batch.append(preprocess(image.convert("RGB")))
            if len(batch) >= batch_size:
                vectors.extend(_encode_batch(model, batch))
                batch.clear()
        if batch:
            vectors.extend(_encode_batch(model, batch))
    return np.asarray(vectors, dtype=np.float32)


def _encode_batch(model, batch: list[torch.Tensor]) -> list[np.ndarray]:
    features = model.encode_image(torch.stack(batch))
    features = torch.nn.functional.normalize(features.float(), dim=-1)
    return [row for row in features.cpu().numpy()]


def build_index() -> dict[str, Any]:
    image_dir, csv_path, index_path = get_paths()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    valid: list[dict[str, str]] = []
    missing_images = invalid_images = invalid_text = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = (row.get("image_id") or "").strip()
            text = clean_text(row.get("text"))
            if not image_id or not text:
                invalid_text += 1
                continue
            image_path = image_dir / f"{image_id}.jpg"
            if not image_path.is_file():
                missing_images += 1
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                invalid_images += 1
                continue
            valid.append({"image_id": image_id, "text": text, "path": str(image_path)})

    if not valid:
        raise ValueError("No rows had a valid image_id, matching JPG, and usable original text.")

    model, preprocess = load_encoder()
    embeddings = _embed_images(model, preprocess, [Path(row["path"]) for row in valid])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "image_dir": str(image_dir),
        "csv_path": str(csv_path),
        "rows": [{k: row[k] for k in ("image_id", "text", "path")} for row in valid],
        "stats": {
            "indexed": len(valid), "missing_images": missing_images,
            "invalid_images": invalid_images, "invalid_text_or_id": invalid_text,
        },
    }
    temporary_path = index_path.with_suffix(index_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, embeddings=embeddings,
                            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    temporary_path.replace(index_path)
    return metadata["stats"]


class ImageRetriever:
    def __init__(self, index_path: Path | None = None):
        self.index_path = index_path or get_paths()[2]
        if not self.index_path.is_file():
            raise FileNotFoundError(
                f"Module 3 index not found at {self.index_path}. Build it with: "
                "python -m src.module3.build_index"
            )
        with np.load(self.index_path, allow_pickle=False) as saved:
            self.embeddings = saved["embeddings"].astype(np.float32, copy=False)
            self.metadata = json.loads(str(saved["metadata"].item()))
        self.model, self.preprocess = load_encoder()

    def search(self, image_path: str | Path, limit: int = 5) -> list[dict[str, Any]]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Query image not found: {path}")
        with _limited_cpu_threads(), Image.open(path) as image, torch.inference_mode():
            pixels = self.preprocess(image.convert("RGB")).unsqueeze(0)
            features = self.model.encode_image(pixels).float()
            query = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()[0]
        del pixels, features
        scores = self.embeddings @ query
        count = min(max(int(limit), 1), 5, len(scores))
        best = np.argpartition(scores, -count)[-count:]
        best = best[np.argsort(scores[best])[::-1]]
        results = []
        for i in best:
            row = self.metadata["rows"][int(i)]
            results.append({**row, "similarity": float(scores[i])})
        return results


"""Shared preprocessing entry point used by both classifier pipelines.

Both pipelines must begin with `load_image()`. Classifier-specific
downstream steps (ImageNet normalisation for the CNN, equalisation for
the classical pipeline) happen after this call, not inside it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from shared.config import (
    CLASS_TO_IDX,
    DATASET_ROOT,
    IMAGE_SIZE,
    SPLIT_CSV,
)


def load_image(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    img = img.crop((left, top, left + m, top + m))
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_split(split_name: str) -> pd.DataFrame:
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"{SPLIT_CSV} not found. Run `python -m shared.split` first."
        )
    if split_name not in {"train", "val", "test"}:
        raise ValueError(
            f"split_name must be one of 'train'/'val'/'test', got {split_name!r}"
        )
    df = pd.read_csv(SPLIT_CSV)
    sub = df[df["split"] == split_name].copy()
    sub["abs_path"] = sub["filename"].map(lambda p: str(DATASET_ROOT / p))
    sub["class_idx"] = sub["class"].map(CLASS_TO_IDX)
    return sub.reset_index(drop=True)

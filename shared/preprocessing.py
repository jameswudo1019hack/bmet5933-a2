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


def load_split(
    split_name: str,
    split_csv: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load one split ('train' | 'val' | 'test') from a stratified split CSV.

    Defaults to the primary split.csv + DATASET_ROOT (medium dataset).
    For the Sprint 2 full-dataset ConvNeXt V2 run, pass
    `split_csv=REPO_ROOT/"split_full.csv"` and `dataset_root=
    REPO_ROOT/"Dataset"/"CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"`.
    """
    csv_path = Path(split_csv) if split_csv else SPLIT_CSV
    root = Path(dataset_root) if dataset_root else DATASET_ROOT
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python -m shared.split` first."
        )
    if split_name not in {"train", "val", "test"}:
        raise ValueError(
            f"split_name must be one of 'train'/'val'/'test', got {split_name!r}"
        )
    df = pd.read_csv(csv_path)
    sub = df[df["split"] == split_name].copy()
    sub["abs_path"] = sub["filename"].map(lambda p: str(root / p))
    sub["class_idx"] = sub["class"].map(CLASS_TO_IDX)
    return sub.reset_index(drop=True)

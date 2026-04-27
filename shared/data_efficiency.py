"""Deterministic stratified subsets of the training split.

Both the classical and deep-learning pipelines call stratified_train_indices()
with the same (fraction, seed) to guarantee they train on identical subsets
when running a data-efficiency sweep. The returned indices are sorted and
index into the DataFrame returned by shared.preprocessing.load_split('train'),
whose row order is itself deterministic.

Stratification preserves per-class ratios to within ±1 image per class.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from shared.config import CLASSES, SEED
from shared.preprocessing import load_split


def stratified_train_indices(
    fraction: float,
    seed: int = SEED,
    split_csv: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> list[int]:
    """Return sorted indices into the train split containing `fraction` of rows.

    Stratified by class. Same (fraction, seed, split_csv) => same indices.
    `split_csv` and `dataset_root` default to the medium-split values configured
    in shared.config; pass `split_csv=REPO_ROOT/"split_full.csv"` for the full
    dataset sweep.
    """
    if not 0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    train_df = load_split("train", split_csv=split_csv, dataset_root=dataset_root)
    rng = np.random.default_rng(seed)

    chosen: list[int] = []
    for cls in CLASSES:
        cls_indices = train_df.index[train_df["class"] == cls].to_numpy()
        n_keep = int(round(len(cls_indices) * fraction))
        # Guarantee at least one sample per class even at tiny fractions
        n_keep = max(n_keep, 1)
        picked = rng.choice(cls_indices, size=n_keep, replace=False)
        chosen.extend(int(i) for i in picked)

    return sorted(chosen)


if __name__ == "__main__":
    for frac in (0.1, 0.25, 0.5, 1.0):
        idx = stratified_train_indices(frac)
        train_df = load_split("train").iloc[idx]
        by_class = train_df["class"].value_counts().to_dict()
        total = len(idx)
        by_class_str = "  ".join(f"{c}={by_class.get(c, 0)}" for c in CLASSES)
        print(f"frac={frac:>5}  total={total:>4}  {by_class_str}")

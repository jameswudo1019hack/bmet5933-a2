"""Deterministic stratified train/val/test splitter.

Emits `split.csv` at the repo root with columns (filename, class, split).
Run once with `python -m shared.split`; the CSV is committed as a
shared artefact. Both Person A and Person B pipelines read from this
CSV and never re-split.

Splitting is two-stage stratified:
  1. (train+val) vs test, preserving class prior.
  2. train vs val within (train+val), preserving class prior.
Both stages use the same seed from shared.config.
"""
from __future__ import annotations

import csv
from collections import Counter

from sklearn.model_selection import train_test_split

from shared.config import (
    CLASSES,
    DATASET_ROOT,
    SEED,
    SPLIT_CSV,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _collect() -> tuple[list[str], list[str]]:
    paths: list[str] = []
    labels: list[str] = []
    for cls in CLASSES:
        cls_dir = DATASET_ROOT / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {cls_dir}\n"
                f"Check DATASET_ROOT ({DATASET_ROOT}) or override it via "
                f"BMET5933_DATASET_ROOT / config.local.yaml."
            )
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in _IMAGE_SUFFIXES:
                paths.append(str(p.relative_to(DATASET_ROOT)))
                labels.append(cls)
    return paths, labels


def make_split() -> None:
    paths, labels = _collect()
    if not paths:
        raise RuntimeError(f"No images found under {DATASET_ROOT}")

    trainval_paths, test_paths, trainval_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=TEST_FRAC,
        stratify=labels,
        random_state=SEED,
    )
    val_share_of_trainval = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        trainval_paths,
        trainval_labels,
        test_size=val_share_of_trainval,
        stratify=trainval_labels,
        random_state=SEED,
    )

    rows: list[tuple[str, str, str]] = []
    for p, l in zip(train_paths, train_labels, strict=True):
        rows.append((p, l, "train"))
    for p, l in zip(val_paths, val_labels, strict=True):
        rows.append((p, l, "val"))
    for p, l in zip(test_paths, test_labels, strict=True):
        rows.append((p, l, "test"))

    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SPLIT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "class", "split"])
        w.writerows(rows)

    print(f"Wrote {SPLIT_CSV}  ({len(rows)} rows)")
    _print_summary(rows)


def _print_summary(rows: list[tuple[str, str, str]]) -> None:
    by_split_class = Counter((s, c) for _, c, s in rows)
    splits = ("train", "val", "test")
    header = f"{'class':<8}" + "".join(f"{s:>8}" for s in splits) + f"{'total':>8}"
    print(header)
    print("-" * len(header))
    totals = {s: 0 for s in splits}
    for cls in CLASSES:
        counts = [by_split_class[(s, cls)] for s in splits]
        line = f"{cls:<8}" + "".join(f"{c:>8}" for c in counts) + f"{sum(counts):>8}"
        print(line)
        for s, c in zip(splits, counts, strict=True):
            totals[s] += c
    print("-" * len(header))
    total_line = (
        f"{'total':<8}"
        + "".join(f"{totals[s]:>8}" for s in splits)
        + f"{sum(totals.values()):>8}"
    )
    print(total_line)


if __name__ == "__main__":
    make_split()

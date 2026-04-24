"""Deterministic stratified train/val/test splitter.

Emits `split.csv` at the repo root with columns (filename, class, split).
Run once with `python -m shared.split`; the CSV is committed as a
shared artefact. Both Person A and Person B pipelines read from this
CSV and never re-split.

Splitting is two-stage stratified:
  1. (train+val) vs test, preserving class prior.
  2. train vs val within (train+val), preserving class prior.
Both stages use the same seed from shared.config.

CLI flags allow generating a separate split for the full dataset
(`--dataset-root .../CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone
 --output split_full.csv`) without disturbing the committed `split.csv`
for the medium-dataset comparison.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from shared.config import (
    CLASSES,
    DATASET_ROOT,
    REPO_ROOT,
    SEED,
    SPLIT_CSV,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _collect(dataset_root: Path) -> tuple[list[str], list[str]]:
    paths: list[str] = []
    labels: list[str] = []
    for cls in CLASSES:
        cls_dir = dataset_root / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {cls_dir}\n"
                f"Check dataset_root ({dataset_root}) or override it via "
                f"BMET5933_DATASET_ROOT / config.local.yaml / --dataset-root."
            )
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in _IMAGE_SUFFIXES:
                paths.append(str(p.relative_to(dataset_root)))
                labels.append(cls)
    return paths, labels


def make_split(dataset_root: Path | None = None, output: Path | None = None) -> None:
    root = dataset_root or DATASET_ROOT
    out = output or SPLIT_CSV
    paths, labels = _collect(root)
    if not paths:
        raise RuntimeError(f"No images found under {root}")

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

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "class", "split"])
        w.writerows(rows)

    print(f"Wrote {out}  ({len(rows)} rows)  from  {root}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a stratified split CSV")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help=f"override dataset root (default resolved via shared.config: {DATASET_ROOT})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"output CSV path (default: {SPLIT_CSV})",
    )
    args = parser.parse_args()
    root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else None
    out = Path(args.output).expanduser().resolve() if args.output else None
    make_split(dataset_root=root, output=out)


if __name__ == "__main__":
    main()

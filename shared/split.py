"""Deterministic per-class group-aware train/val/test splitter.

Emits `split.csv` at the repo root with columns (filename, class, split).
Run once with `python -m shared.split`; the CSV is committed as a
shared artefact. Both Person A and Person B pipelines read from this
CSV and never re-split.

The default splitting strategy is **per-class group-based** (--group-size 50):
filenames encode a numeric slice ID, e.g. "Normal- (1234).jpg".
Images whose IDs fall in the same 50-number block are treated as one
patient group. Each class is split independently so all four classes
hit the target 70/15/15 proportions. This prevents near-identical CT
slices from the same scan leaking across train/test.
Use --group-size 0 to fall back to plain stratified random split.

CLI flags allow generating a separate split for the full dataset
(`--dataset-root .../CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone
 --output split_full.csv`) without disturbing the committed `split.csv`.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, train_test_split

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
_ID_RE = re.compile(r"\((\d+)\)")


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


def _group_id(relative_path: str, class_label: str, group_size: int) -> str:
    """Assign a patient-group label by bucketing the filename's numeric ID.

    E.g. "Normal- (1234).jpg" with group_size=50 -> "Normal_24"
    Images in the same bucket are assumed to be slices from the same scan
    and are kept together in the same split.
    """
    m = _ID_RE.search(relative_path)
    num = int(m.group(1)) if m else 0
    return f"{class_label}_{num // group_size}"


def group_ids_from_filenames(
    filenames: list[str],
    class_labels: list[str],
    group_size: int = 50,
) -> list[str]:
    """Return the same patient-group IDs used when generating the split CSV.

    Call with train_df['filename'].tolist() and train_df['class'].tolist()
    to get groups that align row-for-row with X_train.
    """
    return [_group_id(fn, cls, group_size) for fn, cls in zip(filenames, class_labels)]


def make_split(
    dataset_root: Path | None = None,
    output: Path | None = None,
    group_size: int = 50,
) -> None:
    root = dataset_root or DATASET_ROOT
    out = output or SPLIT_CSV
    paths, labels = _collect(root)
    if not paths:
        raise RuntimeError(f"No images found under {root}")

    rows: list[tuple[str, str, str]] = []

    if group_size > 0:
        val_share = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
        total_groups = 0

        for cls in CLASSES:
            cls_paths  = [p for p, l in zip(paths, labels) if l == cls]
            cls_labels = [cls] * len(cls_paths)
            cls_groups = [_group_id(p, cls, group_size) for p in cls_paths]
            n_grp = len(set(cls_groups))
            total_groups += n_grp

            gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
            tv_idx, test_idx = next(gss1.split(cls_paths, cls_labels, cls_groups))

            tv_paths  = [cls_paths[i]  for i in tv_idx]
            tv_labels = [cls_labels[i] for i in tv_idx]
            tv_groups = [cls_groups[i] for i in tv_idx]

            gss2 = GroupShuffleSplit(n_splits=1, test_size=val_share, random_state=SEED)
            tr_idx, va_idx = next(gss2.split(tv_paths, tv_labels, tv_groups))

            for i in tr_idx:
                rows.append((tv_paths[i], cls, "train"))
            for i in va_idx:
                rows.append((tv_paths[i], cls, "val"))
            for i in test_idx:
                rows.append((cls_paths[i], cls, "test"))

        print(f"[split] per-class group_size={group_size}  ->  {total_groups} groups from {len(paths)} images")
    else:
        print("[split] group_size=0  ->  plain stratified random split (no leakage protection)")
        trainval_paths, test_paths, trainval_labels, test_labels = train_test_split(
            paths, labels, test_size=TEST_FRAC, stratify=labels, random_state=SEED,
        )
        val_share = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            trainval_paths, trainval_labels,
            test_size=val_share, stratify=trainval_labels, random_state=SEED,
        )
        for p, l in zip(train_paths, train_labels):
            rows.append((p, l, "train"))
        for p, l in zip(val_paths, val_labels):
            rows.append((p, l, "val"))
        for p, l in zip(test_paths, test_labels):
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
    parser = argparse.ArgumentParser(description="Generate a per-class group-aware split CSV")
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
    parser.add_argument(
        "--group-size", type=int, default=50,
        help="numeric-ID block size for patient grouping (default 50; use 0 for plain stratified split)",
    )
    args = parser.parse_args()
    root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else None
    out = Path(args.output).expanduser().resolve() if args.output else None
    make_split(dataset_root=root, output=out, group_size=args.group_size)


if __name__ == "__main__":
    main()

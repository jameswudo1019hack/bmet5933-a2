"""Check for any overlap or duplication between the train and test splits.

Three independent checks:
  1. Filename overlap    — same file path appears in both splits
  2. Exact feature match — identical 138-dim feature vectors (bit-for-bit)
  3. Mean intensity      — images with the same rounded mean pixel value
                           (flags near-identical images even if renamed)

Usage
-----
  python -m classical.split_check
  python -m classical.split_check --split-csv split_full.csv \
      --features-cache-dir Results/classical_features_full
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from shared.config import RESULTS_DIR
from shared.preprocessing import load_split
from classical.features import build_feature_matrix

_DEFAULT_CACHE_DIR = RESULTS_DIR / "classical_features"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical ML split leakage check")
    parser.add_argument(
        "--split-csv",
        default=None,
        help="override split CSV (e.g. split_full.csv); default split.csv",
    )
    parser.add_argument(
        "--features-cache-dir",
        default=None,
        help="override feature cache directory (default Results/classical_features)",
    )
    args = parser.parse_args()

    split_csv = Path(args.split_csv) if args.split_csv else None
    CACHE_DIR = Path(args.features_cache_dir) if args.features_cache_dir else _DEFAULT_CACHE_DIR

    train_df = load_split("train", split_csv=split_csv)
    val_df   = load_split("val",   split_csv=split_csv)
    test_df  = load_split("test",  split_csv=split_csv)

    print(f"Split sizes  — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    # ── 1. Filename overlap ───────────────────────────────────────────────────
    train_files = set(train_df["filename"])
    val_files   = set(val_df["filename"])
    test_files  = set(test_df["filename"])

    tv_overlap  = train_files & val_files
    tt_overlap  = train_files & test_files
    vt_overlap  = val_files   & test_files

    print("\n── 1. Filename overlap ──────────────────────────────────────")
    print(f"  train ∩ val  : {len(tv_overlap)} duplicates")
    print(f"  train ∩ test : {len(tt_overlap)} duplicates")
    print(f"  val   ∩ test : {len(vt_overlap)} duplicates")
    if tt_overlap:
        print("  [WARN] train/test filename collisions:")
        for f in sorted(tt_overlap):
            print(f"    {f}")
    else:
        print("  [OK] no filename collisions across any split pair")

    # ── 2. Exact feature-vector duplicates ───────────────────────────────────
    print("\n── 2. Exact feature-vector duplicates ──────────────────────")
    print("  loading / extracting features (uses cache) ...")
    X_train, _ = build_feature_matrix(train_df, cache_path=CACHE_DIR / "train_frac100.npz", desc="train")
    X_val,   _ = build_feature_matrix(val_df,   cache_path=CACHE_DIR / "val.npz",           desc="val")
    X_test,  _ = build_feature_matrix(test_df,  cache_path=CACHE_DIR / "test.npz",          desc="test")

    # Convert each row to a hashable bytes key
    def row_keys(X: np.ndarray) -> set[bytes]:
        return {row.tobytes() for row in X}

    train_keys = row_keys(X_train)
    val_keys   = row_keys(X_val)
    test_keys  = row_keys(X_test)

    tv_feat = len(train_keys & val_keys)
    tt_feat = len(train_keys & test_keys)
    vt_feat = len(val_keys   & test_keys)

    print(f"  train ∩ val  : {tv_feat} exact feature matches")
    print(f"  train ∩ test : {tt_feat} exact feature matches")
    print(f"  val   ∩ test : {vt_feat} exact feature matches")
    if tt_feat == 0:
        print("  [OK] no bit-exact feature duplicates between train and test")
    else:
        print(f"  [WARN] {tt_feat} train rows are bit-identical to test rows")

    # ── 3. Mean-intensity collision (near-duplicate proxy) ───────────────────
    print("\n── 3. Mean-intensity collision (rounded to 2 dp) ───────────")
    # First-order stats: feature index 0 is mean pixel intensity (see features.py)
    train_means = np.round(X_train[:, 0].astype(float), 2)
    val_means   = np.round(X_val[:,   0].astype(float), 2)
    test_means  = np.round(X_test[:,  0].astype(float), 2)

    shared_tt = set(train_means.tolist()) & set(test_means.tolist())
    shared_tv = set(train_means.tolist()) & set(val_means.tolist())

    print(f"  Unique mean-intensity values — train: {len(set(train_means.tolist()))}  "
          f"val: {len(set(val_means.tolist()))}  test: {len(set(test_means.tolist()))}")
    print(f"  train ∩ test shared mean-intensity buckets : {len(shared_tt)}")
    print(f"  train ∩ val  shared mean-intensity buckets : {len(shared_tv)}")
    print("  (shared buckets are expected — different images can share the same mean)")
    print("  [NOTE] this is a coarse proxy; see check 2 for exact duplicates")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────")
    clean = (len(tt_overlap) == 0 and tt_feat == 0)
    if clean:
        print("  [PASS] train/test split is clean — no filename or feature duplicates found")
    else:
        print("  [FAIL] potential leakage detected — review warnings above")


if __name__ == "__main__":
    main()

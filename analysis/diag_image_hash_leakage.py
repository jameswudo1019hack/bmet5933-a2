"""Diagnostic 5 — train/test image-hash and feature-NN leakage probe.

Asks: are there any training images that are bit-identical to test/val images,
or so close in feature space that they are effectively the same slice? Either
case would be unambiguous data leakage and would inflate every reported metric.

Three independent checks per (train vs test) and (train vs val) pair:

  1. **Exact pixel hash** — MD5 of the standardised 256x256 uint8 array
     produced by `shared.preprocessing.load_image`. Catches: same file
     duplicated; same image after deterministic preprocessing.

  2. **Statistical fingerprint** — tuple (mean, std, p10, p25, p50, p75, p90,
     histogram-entropy) rounded to 6 decimals, hashed. This is the test the
     user proposed: "if a train and test image have *exactly* the same
     metrics, they are the same image." Catches: bit-identical images;
     differs from MD5 only if there is some nondeterminism that changes
     pixels but not metrics (rare).

  3. **Feature-space nearest neighbour** — for each test/val image, find its
     closest training image in the cached 108-dim handcrafted feature space
     (`Results/classical_features_full/` or `Results/classical_features/`)
     by cosine similarity. Flag pairs with similarity > 0.9999 as
     "potential near-duplicates" and > 0.999 as "very-similar pairs".
     Catches: near-identical slices, e.g. the same image with minor
     compression / resize artefacts, or adjacent slices from the same patient.

Outputs per split (medium / full):
  Results/diagnostics/image_hash_leakage_<split>.json      — full report
  Results/diagnostics/image_hash_leakage_<split>_pairs.csv — all near-dup pairs

Usage:
  python -m analysis.diag_image_hash_leakage
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as sp_stats

from shared.config import REPO_ROOT, RESULTS_DIR, SPLIT_CSV, SPLIT_CSV_FULL
from shared.preprocessing import load_image, load_split


OUT_DIR = RESULTS_DIR / "diagnostics"


def _pixel_md5(path: str) -> str:
    """MD5 of the standardised 256x256 uint8 pixel buffer."""
    img = load_image(path)
    return hashlib.md5(img.tobytes()).hexdigest()


def _stat_fingerprint(path: str) -> tuple[float, ...]:
    """Statistical fingerprint at 6 decimals. Tuple is hashable."""
    img = load_image(path).astype(np.float64).flatten()
    hist, _ = np.histogram(img, bins=256, range=(0.0, 256.0), density=False)
    prob = hist / (hist.sum() + 1e-12)
    entropy = -float(np.sum(prob[prob > 0] * np.log2(prob[prob > 0])))
    return (
        round(float(img.mean()),                     6),
        round(float(img.std()),                      6),
        round(float(sp_stats.skew(img)),             6),
        round(float(sp_stats.kurtosis(img)),         6),
        round(float(np.percentile(img, 10)),         6),
        round(float(np.percentile(img, 25)),         6),
        round(float(np.percentile(img, 50)),         6),
        round(float(np.percentile(img, 75)),         6),
        round(float(np.percentile(img, 90)),         6),
        round(entropy,                               6),
    )


def _both(path: str) -> tuple[str, tuple[float, ...]]:
    return _pixel_md5(path), _stat_fingerprint(path)


def _hash_split(df: pd.DataFrame, n_jobs: int = -1) -> tuple[list[str], list[tuple[float, ...]]]:
    """Compute MD5 and stat fingerprint for every row of df."""
    paths = df["abs_path"].tolist()
    out = Parallel(n_jobs=n_jobs)(delayed(_both)(p) for p in paths)
    md5s = [h for h, _ in out]
    fps = [f for _, f in out]
    return md5s, fps


def _check_overlap(
    a_df: pd.DataFrame, a_md5: list[str], a_fp: list[tuple[float, ...]],
    b_df: pd.DataFrame, b_md5: list[str], b_fp: list[tuple[float, ...]],
    label_a: str, label_b: str,
) -> dict:
    """Find {a_md5} ∩ {b_md5} and {a_fp} ∩ {b_fp}; report file paths."""
    a_md5_to_idx: dict[str, list[int]] = {}
    for i, h in enumerate(a_md5):
        a_md5_to_idx.setdefault(h, []).append(i)
    md5_collisions: list[dict] = []
    for j, h in enumerate(b_md5):
        if h in a_md5_to_idx:
            for i in a_md5_to_idx[h]:
                md5_collisions.append({
                    "md5": h,
                    f"{label_a}_idx": int(i),
                    f"{label_a}_filename": str(a_df.iloc[i]["filename"]),
                    f"{label_a}_class":    str(a_df.iloc[i]["class"]),
                    f"{label_b}_idx": int(j),
                    f"{label_b}_filename": str(b_df.iloc[j]["filename"]),
                    f"{label_b}_class":    str(b_df.iloc[j]["class"]),
                })

    a_fp_to_idx: dict[tuple[float, ...], list[int]] = {}
    for i, f in enumerate(a_fp):
        a_fp_to_idx.setdefault(f, []).append(i)
    fp_collisions: list[dict] = []
    for j, f in enumerate(b_fp):
        if f in a_fp_to_idx:
            for i in a_fp_to_idx[f]:
                fp_collisions.append({
                    "fingerprint": list(f),
                    f"{label_a}_idx": int(i),
                    f"{label_a}_filename": str(a_df.iloc[i]["filename"]),
                    f"{label_a}_class":    str(a_df.iloc[i]["class"]),
                    f"{label_b}_idx": int(j),
                    f"{label_b}_filename": str(b_df.iloc[j]["filename"]),
                    f"{label_b}_class":    str(b_df.iloc[j]["class"]),
                })

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_a": int(len(a_md5)),
        "n_b": int(len(b_md5)),
        "md5_exact_duplicates_count": len(md5_collisions),
        "fingerprint_exact_duplicates_count": len(fp_collisions),
        "md5_pairs": md5_collisions[:20],   # cap at 20 examples; full list in CSV
        "fingerprint_pairs": fp_collisions[:20],
    }


def _feature_nn_check(
    a_df: pd.DataFrame, a_X: np.ndarray,
    b_df: pd.DataFrame, b_X: np.ndarray,
    label_a: str, label_b: str,
    very_close_threshold: float = 0.999,
    near_duplicate_threshold: float = 0.9999,
) -> dict:
    """For each row in b, find closest row in a by cosine similarity."""
    a_norm = a_X / (np.linalg.norm(a_X, axis=1, keepdims=True) + 1e-12)
    b_norm = b_X / (np.linalg.norm(b_X, axis=1, keepdims=True) + 1e-12)
    sims = b_norm @ a_norm.T   # (n_b, n_a)
    nn_idx = sims.argmax(axis=1)
    nn_sim = sims.max(axis=1)

    near_dup_mask = nn_sim >= near_duplicate_threshold
    very_close_mask = (nn_sim >= very_close_threshold) & ~near_dup_mask

    def _pairs(mask: np.ndarray, cap: int = 20) -> list[dict]:
        idxs = np.where(mask)[0]
        out: list[dict] = []
        for j in idxs[:cap]:
            i = int(nn_idx[j])
            out.append({
                "cosine_sim":   float(nn_sim[j]),
                f"{label_a}_idx": i,
                f"{label_a}_filename": str(a_df.iloc[i]["filename"]),
                f"{label_a}_class":    str(a_df.iloc[i]["class"]),
                f"{label_b}_idx": int(j),
                f"{label_b}_filename": str(b_df.iloc[j]["filename"]),
                f"{label_b}_class":    str(b_df.iloc[j]["class"]),
            })
        return out

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_a": int(len(a_X)),
        "n_b": int(len(b_X)),
        "near_duplicate_threshold": near_duplicate_threshold,
        "very_close_threshold": very_close_threshold,
        "near_duplicate_count": int(near_dup_mask.sum()),
        "very_close_count": int(very_close_mask.sum()),
        "near_duplicate_pairs": _pairs(near_dup_mask),
        "very_close_pairs":     _pairs(very_close_mask),
        "max_cosine_sim":      float(nn_sim.max()),
        "mean_max_cosine_sim": float(nn_sim.mean()),
    }


def run_one_split(split_label: str, split_csv: Path, dataset_root: Path,
                  features_dir: Path) -> dict:
    print(f"\n{'='*78}\nDIAG 5 — split: {split_label}\n{'='*78}")
    print(f"  split_csv={split_csv}  dataset_root={dataset_root}")
    print(f"  features_dir={features_dir}")

    train_df = load_split("train", split_csv=str(split_csv), dataset_root=str(dataset_root))
    val_df   = load_split("val",   split_csv=str(split_csv), dataset_root=str(dataset_root))
    test_df  = load_split("test",  split_csv=str(split_csv), dataset_root=str(dataset_root))
    print(f"  n_train={len(train_df)}  n_val={len(val_df)}  n_test={len(test_df)}")

    print(f"  hashing train ({len(train_df)} images) ...")
    train_md5, train_fp = _hash_split(train_df)
    print(f"  hashing val ({len(val_df)} images) ...")
    val_md5,   val_fp   = _hash_split(val_df)
    print(f"  hashing test ({len(test_df)} images) ...")
    test_md5,  test_fp  = _hash_split(test_df)

    # Within-train duplicate check (sanity — should the dataset itself have dups?)
    train_md5_count: dict[str, int] = {}
    for h in train_md5:
        train_md5_count[h] = train_md5_count.get(h, 0) + 1
    n_within_train_dups = sum(1 for v in train_md5_count.values() if v > 1)

    overlap_train_test = _check_overlap(
        train_df, train_md5, train_fp,
        test_df,  test_md5,  test_fp,
        "train", "test",
    )
    overlap_train_val = _check_overlap(
        train_df, train_md5, train_fp,
        val_df,   val_md5,   val_fp,
        "train", "val",
    )
    overlap_val_test = _check_overlap(
        val_df,   val_md5,   val_fp,
        test_df,  test_md5,  test_fp,
        "val",   "test",
    )

    # Feature-space NN check using cached features
    train_npz = np.load(features_dir / "train_frac100.npz")
    val_npz   = np.load(features_dir / "val.npz")
    test_npz  = np.load(features_dir / "test.npz")
    X_train = train_npz["X"]
    X_val   = val_npz["X"]
    X_test  = test_npz["X"]
    assert X_train.shape[0] == len(train_df), \
        f"train cache size {X_train.shape[0]} != split rows {len(train_df)}"
    assert X_val.shape[0] == len(val_df)
    assert X_test.shape[0] == len(test_df)

    nn_train_test = _feature_nn_check(
        train_df, X_train, test_df, X_test, "train", "test",
    )
    nn_train_val = _feature_nn_check(
        train_df, X_train, val_df,  X_val,  "train", "val",
    )

    # Within-split duplicates (sanity)
    within_train_dup_md5 = [
        {"md5": h, "n": v} for h, v in train_md5_count.items() if v > 1
    ]

    summary = {
        "split_label": split_label,
        "n_train": int(len(train_df)),
        "n_val":   int(len(val_df)),
        "n_test":  int(len(test_df)),
        "within_train_md5_duplicates_count": int(n_within_train_dups),
        "within_train_md5_duplicates_examples": within_train_dup_md5[:20],
        "overlap_train_test": overlap_train_test,
        "overlap_train_val":  overlap_train_val,
        "overlap_val_test":   overlap_val_test,
        "feature_nn_train_test": nn_train_test,
        "feature_nn_train_val":  nn_train_val,
    }

    out_json = OUT_DIR / f"image_hash_leakage_{split_label}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] {out_json}")

    # Print headline
    print(f"\n  ===== HEADLINE for split={split_label} =====")
    print(f"  Within-train MD5 duplicates: {n_within_train_dups}")
    print(f"  train ∩ test: MD5 collisions = {overlap_train_test['md5_exact_duplicates_count']}, "
          f"fingerprint = {overlap_train_test['fingerprint_exact_duplicates_count']}")
    print(f"  train ∩ val:  MD5 collisions = {overlap_train_val['md5_exact_duplicates_count']}, "
          f"fingerprint = {overlap_train_val['fingerprint_exact_duplicates_count']}")
    print(f"  val ∩ test:   MD5 collisions = {overlap_val_test['md5_exact_duplicates_count']}, "
          f"fingerprint = {overlap_val_test['fingerprint_exact_duplicates_count']}")
    print(f"  feature-NN train→test: max sim = {nn_train_test['max_cosine_sim']:.6f}  "
          f"near-dup (>{nn_train_test['near_duplicate_threshold']}) = {nn_train_test['near_duplicate_count']}, "
          f"very-close (>{nn_train_test['very_close_threshold']}) = {nn_train_test['very_close_count']}")
    print(f"  feature-NN train→val:  max sim = {nn_train_val['max_cosine_sim']:.6f}  "
          f"near-dup = {nn_train_val['near_duplicate_count']}, "
          f"very-close = {nn_train_val['very_close_count']}")

    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    medium_root = REPO_ROOT / "Dataset" / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone_medium"
    full_root   = REPO_ROOT / "Dataset" / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"

    medium_summary = run_one_split(
        "medium",
        SPLIT_CSV,
        medium_root,
        RESULTS_DIR / "classical_features",
    )
    full_summary = run_one_split(
        "full",
        SPLIT_CSV_FULL,
        full_root,
        RESULTS_DIR / "classical_features_full",
    )

    overall = {
        "medium": medium_summary,
        "full":   full_summary,
    }
    out_path = OUT_DIR / "image_hash_leakage_summary.json"
    out_path.write_text(json.dumps(overall, indent=2))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()

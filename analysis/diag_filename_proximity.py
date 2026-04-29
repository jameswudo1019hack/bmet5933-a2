"""Diagnostic 1 — slice-level leakage probe via filename-numerical-proximity.

Hypothesis (Yagis et al. 2021, Veetil et al. 2024): the Islam kidney-CT dataset
has no patient IDs. Stratified random split on slice level may put adjacent
slices (which look near-identical) in train and test, inflating accuracy.

Filenames are like:
    Cyst-  (3051).jpg
    Tumor- (651).jpg
    Stone- (791).jpg
    Normal-(1795).jpg

If numerically adjacent files within the same class are slices from the same
patient, then for each test image, its numerically-nearest training image
should be feature-space-similar to it (encoding "this is patient X") far above
random baseline.

Test (per class):
  For each test image with class C and numeric ID i:
    nearest-by-ID = mean cosine similarity in 108-dim feature space between
                    the test image and the K=5 train images of class C with
                    smallest |id_train - i|.
    random-baseline = mean cosine similarity to K random train images of class C.
  Compare distributions per-class with one-sided Mann-Whitney U
  (alternative: nearest > random).

Verdict heuristics:
  ratio = mean(nearest)/mean(random)
    >= 1.5  -> strong evidence of patient grouping by filename order
    1.05-1.5 -> moderate evidence
    <  1.05 -> no evidence; filenames are random w.r.t. patient identity

Outputs:
  Results/diagnostics/filename_proximity.json   (per-class statistics)
  Results/diagnostics/filename_proximity.png    (4 KDE subplots)

Usage:
  python -m analysis.diag_filename_proximity
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from shared.config import CLASSES, REPO_ROOT, RESULTS_DIR
from shared.preprocessing import load_split


FEATURES_DIR = RESULTS_DIR / "classical_features_full"
OUT_DIR = RESULTS_DIR / "diagnostics"
SPLIT_CSV = REPO_ROOT / "split_full.csv"
DATASET_ROOT = REPO_ROOT / "Dataset" / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"

K_NEAREST = 5
N_RANDOM_TRIALS = 5  # average random baseline over this many random pulls per test image
RNG_SEED = 42


# Filenames look like e.g. "Cyst- (3051).jpg" — extract the integer in parentheses.
_ID_RE = re.compile(r"\((\d+)\)")


def _parse_id(filename: str) -> int | None:
    m = _ID_RE.search(filename)
    return int(m.group(1)) if m else None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. a: (n, d), b: (n, d) -> (n,)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a_norm * b_norm, axis=1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_npz = np.load(FEATURES_DIR / "train_frac100.npz")
    test_npz  = np.load(FEATURES_DIR / "test.npz")
    X_train, y_train = train_npz["X"], train_npz["y"]
    X_test,  y_test  = test_npz["X"],  test_npz["y"]
    print(f"[load] train={X_train.shape}  test={X_test.shape}")

    train_df = load_split("train", split_csv=str(SPLIT_CSV), dataset_root=str(DATASET_ROOT))
    test_df  = load_split("test",  split_csv=str(SPLIT_CSV), dataset_root=str(DATASET_ROOT))

    train_ids_all = np.array([_parse_id(fn) for fn in train_df["filename"]])
    test_ids_all  = np.array([_parse_id(fn) for fn in test_df["filename"]])
    assert (train_ids_all != None).all() and (test_ids_all != None).all(), \
        "Some filenames did not parse a numeric ID"
    train_ids_all = train_ids_all.astype(int)
    test_ids_all  = test_ids_all.astype(int)

    rng = np.random.default_rng(RNG_SEED)
    per_class_stats: dict[str, dict] = {}
    nearest_dists: dict[str, np.ndarray] = {}
    random_dists:  dict[str, np.ndarray] = {}

    for c_idx, c_name in enumerate(CLASSES):
        train_mask = y_train == c_idx
        test_mask  = y_test  == c_idx
        if test_mask.sum() == 0 or train_mask.sum() < K_NEAREST + 1:
            continue

        X_train_c = X_train[train_mask]
        X_test_c  = X_test[test_mask]
        train_ids_c = train_ids_all[train_mask]
        test_ids_c  = test_ids_all[test_mask]
        n_train_c = train_mask.sum()
        n_test_c  = test_mask.sum()

        nearest_sims_per_test: list[float] = []
        random_sims_per_test:  list[float] = []
        nearest_id_gap_per_test: list[float] = []  # mean |delta| for nearest K

        for i in range(n_test_c):
            t_id = test_ids_c[i]
            x_test_one = X_test_c[i:i + 1]  # (1, 108)

            # K nearest by id within this class's train pool
            id_diffs = np.abs(train_ids_c - t_id)
            order = np.argsort(id_diffs, kind="stable")
            nearest_idx = order[:K_NEAREST]
            x_train_nearest = X_train_c[nearest_idx]
            sim_nearest = _cosine_sim(
                np.repeat(x_test_one, K_NEAREST, axis=0), x_train_nearest
            ).mean()
            nearest_sims_per_test.append(float(sim_nearest))
            nearest_id_gap_per_test.append(float(id_diffs[nearest_idx].mean()))

            # Random baseline: average over N_RANDOM_TRIALS independent draws of K
            random_means: list[float] = []
            for _ in range(N_RANDOM_TRIALS):
                rand_idx = rng.choice(n_train_c, size=K_NEAREST, replace=False)
                x_train_rand = X_train_c[rand_idx]
                sim_rand = _cosine_sim(
                    np.repeat(x_test_one, K_NEAREST, axis=0), x_train_rand
                ).mean()
                random_means.append(float(sim_rand))
            random_sims_per_test.append(float(np.mean(random_means)))

        nearest_arr = np.array(nearest_sims_per_test)
        random_arr  = np.array(random_sims_per_test)

        # One-sided Mann-Whitney U: alternative = "nearest > random"
        try:
            u, p = sp_stats.mannwhitneyu(
                nearest_arr, random_arr, alternative="greater",
            )
        except ValueError:
            u, p = float("nan"), 1.0

        ratio = float(nearest_arr.mean() / max(random_arr.mean(), 1e-12))
        verdict = (
            "strong leakage signal" if ratio >= 1.5
            else "moderate signal"   if ratio >= 1.05
            else "no signal"
        )

        per_class_stats[c_name] = {
            "n_test": int(n_test_c),
            "n_train_pool": int(n_train_c),
            "nearest_mean_sim": float(nearest_arr.mean()),
            "nearest_std_sim":  float(nearest_arr.std()),
            "random_mean_sim":  float(random_arr.mean()),
            "random_std_sim":   float(random_arr.std()),
            "ratio":            ratio,
            "delta_mean_sim":   float(nearest_arr.mean() - random_arr.mean()),
            "mannwhitney_u":    float(u),
            "mannwhitney_pvalue": float(p),
            "mean_id_gap_to_nearest_K": float(np.mean(nearest_id_gap_per_test)),
            "verdict": verdict,
        }
        nearest_dists[c_name] = nearest_arr
        random_dists[c_name]  = random_arr

        print(f"[{c_name}]  n_test={n_test_c}  "
              f"nearest_sim={nearest_arr.mean():.4f}+-{nearest_arr.std():.4f}  "
              f"random_sim={random_arr.mean():.4f}+-{random_arr.std():.4f}  "
              f"ratio={ratio:.3f}  p={p:.2e}  verdict={verdict}")

    # ── Aggregate verdict ────────────────────────────────────────────────────
    n_strong   = sum(1 for s in per_class_stats.values() if s["verdict"] == "strong leakage signal")
    n_moderate = sum(1 for s in per_class_stats.values() if s["verdict"] == "moderate signal")
    n_no       = sum(1 for s in per_class_stats.values() if s["verdict"] == "no signal")

    overall = (
        "STRONG leakage suspected (>=1 class with ratio >= 1.5)" if n_strong >= 1
        else "MODERATE leakage suspected (>=2 classes with ratio >= 1.05)" if n_moderate >= 2
        else "WEAK / no leakage signal"
    )

    summary = {
        "k_nearest": K_NEAREST,
        "n_random_trials": N_RANDOM_TRIALS,
        "rng_seed": RNG_SEED,
        "per_class": per_class_stats,
        "overall_verdict": overall,
        "n_classes_strong_signal": n_strong,
        "n_classes_moderate_signal": n_moderate,
        "n_classes_no_signal": n_no,
    }
    out_json = OUT_DIR / "filename_proximity.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] {out_json}")
    print(f"[verdict] {overall}")

    # ── KDE figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False, sharey=False)
    for ax, c_name in zip(axes.ravel(), CLASSES):
        if c_name not in per_class_stats:
            ax.set_title(f"{c_name} — insufficient data")
            ax.axis("off")
            continue
        n = nearest_dists[c_name]
        r = random_dists[c_name]
        try:
            n_kde = sp_stats.gaussian_kde(n)
            r_kde = sp_stats.gaussian_kde(r)
            x_lo = float(min(n.min(), r.min())) - 0.02
            x_hi = float(max(n.max(), r.max())) + 0.02
            xs = np.linspace(x_lo, x_hi, 256)
            ax.fill_between(xs, n_kde(xs), color="#c44e52", alpha=0.45,
                            label=f"nearest by ID (mean={n.mean():.3f})")
            ax.fill_between(xs, r_kde(xs), color="#4c72b0", alpha=0.45,
                            label=f"random baseline (mean={r.mean():.3f})")
        except Exception:
            ax.hist(n, bins=30, color="#c44e52", alpha=0.5,
                    label=f"nearest by ID (mean={n.mean():.3f})")
            ax.hist(r, bins=30, color="#4c72b0", alpha=0.5,
                    label=f"random baseline (mean={r.mean():.3f})")

        s = per_class_stats[c_name]
        ax.set_title(
            f"{c_name}  ·  n_test={s['n_test']}  ·  ratio={s['ratio']:.3f}  "
            f"·  p={s['mannwhitney_pvalue']:.1e}\n{s['verdict']}",
            fontsize=10,
        )
        ax.set_xlabel("cosine similarity to K=5 train images", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Filename-numerical-proximity slice-leakage probe\n"
        f"Overall: {overall}",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()
    out_png = OUT_DIR / "filename_proximity.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}")


if __name__ == "__main__":
    main()

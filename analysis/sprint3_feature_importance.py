"""Sprint 3 — classical XGBoost feature importance (deployed pipeline + raw-XGB sanity).

Two analyses on the *full classical* run (Results/classical_run_full/):

1. Permutation importance on the deployed pipeline (StandardScaler -> PCA(50) -> XGB)
   over the n=1867 test set. Computed per-individual feature (108 features) and
   per-group (stats / GLCM / LBP / Gabor). This answers the question:
   "in the deployed model, which raw features carry the predictive signal?"

2. Parallel raw-XGB sanity check: fit a separate XGBoost on the unscaled,
   un-PCA'd 108-dim features using the same best_params, then read its
   built-in gain importance. This answers: "if PCA were not in the way,
   which features would XGB pick?". Cross-checks the permutation result.

The 108-dim feature vector layout (must match classical/features.py order):
  0   - 9   : 10 first-order intensity statistics
  10  - 21  : 12 GLCM features (6 props x 2 distances, angle-averaged)
  22  - 75  : 54 LBP histogram bins (10 + 18 + 26 across (P,R) = (8,1)/(16,2)/(24,3))
  76  - 107 : 32 Gabor features (16 kernels x {mean, std})

Outputs:
  Results/classical_run_full/feature_importance.json
  Results/classical_run_full/feature_importance.csv
  Results/classical_run_full/feature_importance_group.png    -> Paper Figure 2
  Results/classical_run_full/feature_importance_top20.png    -> supplementary

Usage:
  python -m analysis.sprint3_feature_importance
"""
from __future__ import annotations

import csv
import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from shared.config import RESULTS_DIR, SEED


PIPELINE_PATH = RESULTS_DIR / "classical_run_full" / "classical_pipeline.pkl"
FEATURES_DIR = RESULTS_DIR / "classical_features_full"
OUT_DIR = RESULTS_DIR / "classical_run_full"


# ── Feature layout ────────────────────────────────────────────────────────────
def _feature_names() -> tuple[list[str], dict[str, list[int]]]:
    """Return (per-feature names, group -> indices map)."""
    names: list[str] = []

    # Stats (10)
    stat_labels = [
        "mean", "std", "skew", "kurt", "entropy",
        "p10", "p25", "p50", "p75", "p90",
    ]
    names.extend(f"stat:{s}" for s in stat_labels)

    # GLCM (12) = 6 props x 2 distances, angle-averaged
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    distances = [1, 3]
    for prop in props:
        for d in distances:
            names.append(f"glcm:{prop}_d{d}")

    # LBP (54) = 10 + 18 + 26
    for P, R in ((8, 1), (16, 2), (24, 3)):
        for b in range(P + 2):
            names.append(f"lbp:P{P}R{R}_b{b}")

    # Gabor (32) = 16 kernels x {mean, std}
    freqs = [0.1, 0.2, 0.3, 0.4]
    thetas = ["0", "pi/4", "pi/2", "3pi/4"]
    for f in freqs:
        for t in thetas:
            names.append(f"gabor:f{f}_t{t}_mean")
            names.append(f"gabor:f{f}_t{t}_std")

    assert len(names) == 108, f"feature name count {len(names)} != 108"

    groups = {
        "stats": list(range(0, 10)),
        "GLCM":  list(range(10, 22)),
        "LBP":   list(range(22, 76)),
        "Gabor": list(range(76, 108)),
    }
    return names, groups


# ── Group permutation importance (sklearn doesn't ship with one) ─────────────
def _group_permutation_importance(
    pipe,
    X: np.ndarray,
    y: np.ndarray,
    groups: dict[str, list[int]],
    n_repeats: int = 10,
    rng_seed: int = SEED,
) -> dict[str, dict[str, float]]:
    """Permute all features in each group together and measure macro-F1 drop.

    Reuses the same pipeline.predict pathway as deployment; macro-F1 because
    that is the project's canonical metric (Phase 0 §5).
    """
    rng = np.random.default_rng(rng_seed)
    baseline = f1_score(y, pipe.predict(X), average="macro", zero_division=0)
    out: dict[str, dict[str, float]] = {"_baseline_macro_f1": {"value": float(baseline)}}

    for group_name, idx in groups.items():
        diffs: list[float] = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            shuffle = rng.permutation(len(X))
            X_perm[:, idx] = X[shuffle][:, :][:, idx]
            permuted_score = f1_score(
                y, pipe.predict(X_perm), average="macro", zero_division=0,
            )
            diffs.append(baseline - permuted_score)
        out[group_name] = {
            "mean_drop": float(np.mean(diffs)),
            "std_drop": float(np.std(diffs)),
            "n_repeats": n_repeats,
        }
    return out


# ── Plotting helpers ──────────────────────────────────────────────────────────
def _plot_groups(group_results: dict[str, dict[str, float]], out_path: Path) -> None:
    """Bar chart of per-group permutation-importance mean drop in macro-F1."""
    groups = [g for g in group_results if not g.startswith("_")]
    means = [group_results[g]["mean_drop"] for g in groups]
    stds = [group_results[g]["std_drop"] for g in groups]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    bars = ax.bar(
        groups, means, yerr=stds,
        color=colors[: len(groups)], alpha=0.85,
        capsize=5, edgecolor="black", linewidth=0.8,
    )
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=10,
        )
    baseline_f1 = group_results.get("_baseline_macro_f1", {}).get("value", float("nan"))
    ax.set_xlabel("Feature group", fontsize=11)
    ax.set_ylabel("macro-F1 drop when permuted\n(higher = more important)", fontsize=11)
    ax.set_title(
        f"Classical XGBoost (full): per-group permutation importance\n"
        f"deployed pipeline · baseline macro-F1 = {baseline_f1:.4f} · "
        f"n_test = 1867 · n_repeats = {next(iter([v.get('n_repeats') for k, v in group_results.items() if not k.startswith('_')]))}",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] groups -> {out_path}")


def _plot_top_n(
    importances: np.ndarray,
    names: list[str],
    n: int,
    out_path: Path,
    title_suffix: str,
) -> None:
    order = np.argsort(importances)[::-1][:n]
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.32)))
    ax.barh(
        range(n)[::-1],
        importances[order],
        color="#4c72b0", alpha=0.85, edgecolor="black", linewidth=0.6,
    )
    ax.set_yticks(range(n)[::-1])
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel("Importance (mean macro-F1 drop)", fontsize=11)
    ax.set_title(f"Top {n} individual features — {title_suffix}", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] top-{n} -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # Load deployed pipeline
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)
    deployed = Pipeline([
        ("scaler", pipeline["scaler"]),
        ("pca",    pipeline["pca"]),
        ("clf",    pipeline["classifier"]),
    ])
    best_params = pipeline["best_params"]
    print(f"[load] deployed pipeline: {pipeline['model_name']}  best_params={best_params}")

    # Load cached features
    train_npz = np.load(FEATURES_DIR / "train_frac100.npz")
    val_npz   = np.load(FEATURES_DIR / "val.npz")
    test_npz  = np.load(FEATURES_DIR / "test.npz")
    X_train, y_train = train_npz["X"], train_npz["y"]
    X_val,   y_val   = val_npz["X"],   val_npz["y"]
    X_test,  y_test  = test_npz["X"],  test_npz["y"]
    print(f"[load] train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    names, groups = _feature_names()

    # ── 1. Per-individual permutation importance (deployed pipeline) ────────
    print("\n[perm] running per-individual permutation importance "
          "(108 features x n_repeats=10) ...")
    perm_indiv = permutation_importance(
        deployed, X_test, y_test,
        n_repeats=10,
        random_state=SEED,
        scoring="f1_macro",
        n_jobs=-1,
    )
    indiv_mean = perm_indiv.importances_mean
    indiv_std  = perm_indiv.importances_std
    print(f"[perm] done. max indiv importance = {indiv_mean.max():.4f}  "
          f"min = {indiv_mean.min():.4f}")

    # ── 2. Group permutation importance (deployed pipeline) ─────────────────
    print("\n[perm] running per-group permutation importance ...")
    group_perm = _group_permutation_importance(
        deployed, X_test, y_test, groups, n_repeats=10, rng_seed=SEED,
    )
    print("[perm] group importance:")
    for g, v in group_perm.items():
        if g.startswith("_"):
            continue
        print(f"  {g:<6s}  mean_drop={v['mean_drop']:.4f} +- {v['std_drop']:.4f}")

    # ── 3. Raw-XGB sanity check (no PCA) ─────────────────────────────────────
    print("\n[raw-xgb] fitting parallel XGB on raw 108-dim features ...")
    X_tv = np.concatenate([X_train, X_val], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)
    sw = compute_sample_weight("balanced", y_tv)
    raw_xgb = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        verbosity=0,
        random_state=SEED,
        n_jobs=-1,
        **best_params,
    )
    raw_xgb.fit(X_tv, y_tv, sample_weight=sw)
    raw_pred = raw_xgb.predict(X_test)
    raw_macro_f1 = f1_score(y_test, raw_pred, average="macro", zero_division=0)
    print(f"[raw-xgb] test macro-F1 = {raw_macro_f1:.4f} (cf. deployed = "
          f"{group_perm['_baseline_macro_f1']['value']:.4f})")

    raw_imp = raw_xgb.feature_importances_

    # ── 4. Save per-feature CSV + JSON summary ───────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "feature_importance.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx", "name", "group",
            "perm_importance_mean", "perm_importance_std",
            "raw_xgb_gain_importance",
        ])
        for i, name in enumerate(names):
            group = next(g for g, idxs in groups.items() if i in idxs)
            w.writerow([
                i, name, group,
                f"{indiv_mean[i]:.6f}", f"{indiv_std[i]:.6f}",
                f"{raw_imp[i]:.6f}",
            ])
    print(f"[save] per-feature CSV -> {csv_path}")

    summary = {
        "deployed_pipeline_baseline_macro_f1": float(
            group_perm["_baseline_macro_f1"]["value"]
        ),
        "raw_xgb_test_macro_f1": float(raw_macro_f1),
        "n_features": len(names),
        "n_test": int(len(y_test)),
        "n_repeats": 10,
        "group_permutation_importance": {
            g: v for g, v in group_perm.items() if not g.startswith("_")
        },
        "top10_individual_permutation": [
            {
                "rank": rank + 1,
                "idx": int(idx),
                "name": names[idx],
                "group": next(g for g, idxs in groups.items() if idx in idxs),
                "mean_drop": float(indiv_mean[idx]),
                "std_drop": float(indiv_std[idx]),
            }
            for rank, idx in enumerate(np.argsort(indiv_mean)[::-1][:10])
        ],
        "top10_individual_raw_xgb_gain": [
            {
                "rank": rank + 1,
                "idx": int(idx),
                "name": names[idx],
                "group": next(g for g, idxs in groups.items() if idx in idxs),
                "gain_importance": float(raw_imp[idx]),
            }
            for rank, idx in enumerate(np.argsort(raw_imp)[::-1][:10])
        ],
    }
    json_path = OUT_DIR / "feature_importance.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[save] JSON summary -> {json_path}")

    # ── 5. Plots ─────────────────────────────────────────────────────────────
    _plot_groups(group_perm, OUT_DIR / "feature_importance_group.png")
    _plot_top_n(
        indiv_mean, names, 20,
        OUT_DIR / "feature_importance_top20.png",
        title_suffix="permutation importance (deployed pipeline)",
    )

    print("\n" + "=" * 60)
    print("Top 10 individual features by permutation importance:")
    print("=" * 60)
    for r, idx in enumerate(np.argsort(indiv_mean)[::-1][:10], 1):
        group = next(g for g, idxs in groups.items() if idx in idxs)
        print(f"  {r:>2}. {names[idx]:<28s} ({group:<5s})  "
              f"drop = {indiv_mean[idx]:.4f}")

    print("\n" + "=" * 60)
    print("Top 10 by raw-XGB gain (sanity check, no PCA):")
    print("=" * 60)
    for r, idx in enumerate(np.argsort(raw_imp)[::-1][:10], 1):
        group = next(g for g, idxs in groups.items() if idx in idxs)
        print(f"  {r:>2}. {names[idx]:<28s} ({group:<5s})  "
              f"gain = {raw_imp[idx]:.4f}")


if __name__ == "__main__":
    main()

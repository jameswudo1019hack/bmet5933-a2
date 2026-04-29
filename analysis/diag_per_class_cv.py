"""Diagnostic 3 — per-class 5-fold stratified CV metrics for classical XGBoost.

The original `classical/train.py` runs 5-fold stratified CV but reports only
the aggregate `cv_f1`. Sandhya's note "do CV class-wise too" asks for the
per-class breakdown: F1 / precision / recall mean ± std across folds. High
variance in a particular class (especially Stone, the universal weak class)
would indicate that classical's apparent saturation hides per-class instability.

Protocol:
  - Train+val combined (10,579 samples) — same pool used by the deployed
    pipeline's final fit, so per-class CV variance reflects what the deployed
    model "would see" if held-out folds were drawn from the deployment data.
  - StratifiedKFold(5, seed=42) on the deployed pipeline's PCA(50)-projected
    features. Same scaler/PCA fit per-fold (re-fit on each train portion
    so the metric is honestly held-out).
  - For each fold: train XGB(deployed_best_params), predict on held-out fold,
    record per-class F1 / precision / recall.
  - Aggregate: per-class mean ± std across 5 folds.

Why train+val and not train only? The deployed model is fit on train+val for
its final classifier; reflecting that here gives the best apples-to-apples
"how stable is the deployed config?" answer. We do *not* touch the test set —
test stays sacred for the final reported numbers.

Outputs:
  Results/diagnostics/per_class_cv.json
  Results/diagnostics/per_class_cv.png    (4-bar per-class F1 with error bars
                                            + held-out test F1 overlay)

Usage:
  python -m analysis.diag_per_class_cv
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from shared.config import CLASSES, RESULTS_DIR, SEED
from classical.config import PCA_N_COMPONENTS


import json as _json
import pickle as _pickle

PIPELINE_PATH = RESULTS_DIR / "classical_run_full" / "classical_pipeline.pkl"
RESULTS_JSON_PATH = RESULTS_DIR / "classical_run_full" / "classical_results.json"
FEATURES_DIR = RESULTS_DIR / "classical_features_full"
OUT_DIR = RESULTS_DIR / "diagnostics"
N_SPLITS = 5


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load best_params from deployed pipeline
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = _pickle.load(f)
    best_params = pipeline["best_params"]
    print(f"[load] deployed best_params = {best_params}")

    # Load held-out test per-class F1 for overlay reference
    deployed_test = _json.loads(RESULTS_JSON_PATH.read_text())
    deployed_per_class_f1 = {c: deployed_test["per_class"][c]["f1"] for c in CLASSES}
    print(f"[load] held-out test per-class F1: {deployed_per_class_f1}")

    # Load cached features and concatenate train+val
    train_npz = np.load(FEATURES_DIR / "train_frac100.npz")
    val_npz   = np.load(FEATURES_DIR / "val.npz")
    X_tv = np.concatenate([train_npz["X"], val_npz["X"]], axis=0)
    y_tv = np.concatenate([train_npz["y"], val_npz["y"]], axis=0)
    print(f"[load] train+val pool: {X_tv.shape}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    n_classes = len(CLASSES)

    per_fold_f1   = np.zeros((N_SPLITS, n_classes))
    per_fold_prec = np.zeros((N_SPLITS, n_classes))
    per_fold_rec  = np.zeros((N_SPLITS, n_classes))
    per_fold_macro_f1 = np.zeros(N_SPLITS)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tv, y_tv)):
        X_tr, X_va = X_tv[tr_idx], X_tv[va_idx]
        y_tr, y_va = y_tv[tr_idx], y_tv[va_idx]

        # Scaler + PCA fit on this fold's train portion only
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        pca_n = min(PCA_N_COMPONENTS, X_tr_sc.shape[0] - 1, X_tr_sc.shape[1])
        pca = PCA(n_components=pca_n, svd_solver="full", random_state=SEED)
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_va_pca = pca.transform(scaler.transform(X_va))

        # Train XGB with deployed best_params
        sw = compute_sample_weight("balanced", y_tr)
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            verbosity=0,
            random_state=SEED,
            n_jobs=-1,
            **best_params,
        )
        clf.fit(X_tr_pca, y_tr, sample_weight=sw)
        y_pred = clf.predict(X_va_pca)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_va, y_pred,
            labels=list(range(n_classes)),
            zero_division=0,
        )
        per_fold_f1[fold_idx]   = f1
        per_fold_prec[fold_idx] = prec
        per_fold_rec[fold_idx]  = rec
        per_fold_macro_f1[fold_idx] = float(f1.mean())

        print(f"[fold {fold_idx + 1}/{N_SPLITS}]  macro-F1={f1.mean():.4f}  "
              f"per-class F1: " +
              "  ".join(f"{c}={f1[i]:.4f}" for i, c in enumerate(CLASSES)))

    # Aggregate
    f1_mean   = per_fold_f1.mean(axis=0)
    f1_std    = per_fold_f1.std(axis=0)
    prec_mean = per_fold_prec.mean(axis=0)
    prec_std  = per_fold_prec.std(axis=0)
    rec_mean  = per_fold_rec.mean(axis=0)
    rec_std   = per_fold_rec.std(axis=0)

    summary = {
        "n_splits": N_SPLITS,
        "deployed_best_params": best_params,
        "macro_f1_mean": float(per_fold_macro_f1.mean()),
        "macro_f1_std":  float(per_fold_macro_f1.std()),
        "per_class": {
            c: {
                "f1_mean":        float(f1_mean[i]),
                "f1_std":         float(f1_std[i]),
                "precision_mean": float(prec_mean[i]),
                "precision_std":  float(prec_std[i]),
                "recall_mean":    float(rec_mean[i]),
                "recall_std":     float(rec_std[i]),
                "deployed_held_out_test_f1": float(deployed_per_class_f1[c]),
            }
            for i, c in enumerate(CLASSES)
        },
        "per_fold_macro_f1": per_fold_macro_f1.tolist(),
        "per_fold_per_class_f1":   per_fold_f1.tolist(),
        "per_fold_per_class_prec": per_fold_prec.tolist(),
        "per_fold_per_class_rec":  per_fold_rec.tolist(),
    }
    out_json = OUT_DIR / "per_class_cv.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] {out_json}")

    print("\n" + "=" * 60)
    print(f"5-fold stratified CV (n=10579, seed=42)")
    print("=" * 60)
    print(f"{'class':<7s} {'F1 mean ± std':<20s} {'min':<8s} {'max':<8s} "
          f"{'held-out test F1':<18s}")
    for i, c in enumerate(CLASSES):
        f1_min = float(per_fold_f1[:, i].min())
        f1_max = float(per_fold_f1[:, i].max())
        print(f"{c:<7s} {f1_mean[i]:.4f} ± {f1_std[i]:.4f}    "
              f"{f1_min:.4f}  {f1_max:.4f}    {deployed_per_class_f1[c]:.4f}")
    print(f"\nmacro-F1 (CV): {per_fold_macro_f1.mean():.4f} ± "
          f"{per_fold_macro_f1.std():.4f}")
    print(f"macro-F1 (deployed held-out test): {deployed_test['macro_f1']:.4f}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(n_classes)
    width = 0.35

    cv_bars = ax.bar(
        x - width / 2,
        f1_mean, yerr=f1_std,
        width=width, color="#4c72b0", alpha=0.85, capsize=5,
        edgecolor="black", linewidth=0.6, label="5-fold CV F1 (mean ± std)",
    )
    test_vals = [deployed_per_class_f1[c] for c in CLASSES]
    test_bars = ax.bar(
        x + width / 2,
        test_vals,
        width=width, color="#dd8452", alpha=0.85,
        edgecolor="black", linewidth=0.6, label="held-out test F1 (deployed)",
    )

    for bar, val in zip(cv_bars, f1_mean):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(test_bars, test_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # σ-distance annotations per class — the headline diagnostic signal.
    for i, c in enumerate(CLASSES):
        if f1_std[i] < 1e-9:
            continue
        sigma_dist = (test_vals[i] - f1_mean[i]) / f1_std[i]
        if abs(sigma_dist) >= 1.5:
            color = "#c44e52"          # red — concerning
            tag = "structural mismatch"
        elif abs(sigma_dist) >= 1.0:
            color = "#dd8452"          # orange — borderline
            tag = "borderline"
        else:
            color = "#55a868"          # green — within bounds
            tag = "within bounds"
        sign = "+" if sigma_dist >= 0 else "−"
        label = f"{sign}{abs(sigma_dist):.1f} σ"
        ax.text(
            i, 0.86, label,
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.2),
        )
        ax.text(
            i, 0.852, tag,
            ha="center", va="top", fontsize=8, color=color, style="italic",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_ylabel("F1 score", fontsize=11)
    ax.set_ylim(0.83, 1.02)
    ax.set_title(
        "Classical XGBoost — 5-fold stratified CV per-class F1 vs held-out test F1\n"
        f"CV macro-F1 = {per_fold_macro_f1.mean():.4f} ± {per_fold_macro_f1.std():.4f}  ·  "
        f"test macro-F1 = {deployed_test['macro_f1']:.4f}\n"
        "σ-distance = (test F1 − CV mean) / CV std  ·  "
        "|σ| ≥ 1.5 flagged as structural mismatch",
        fontsize=10,
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_png = OUT_DIR / "per_class_cv.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[save] {out_png}")


if __name__ == "__main__":
    main()

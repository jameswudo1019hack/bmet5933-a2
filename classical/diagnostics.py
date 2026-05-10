"""Overfitting diagnostics for the classical ML pipeline (Person A).

Three analyses:
  1. Split integrity check  — verifies no image appears in more than one split
  2. Training loss convergence — XGBoost mlogloss per boosting round (train + val)
  3. Per-class metrics table — Precision / Recall / F1 across train, 5-fold CV, and test

All analyses use a model fitted on the training split only (not train+val combined)
so that validation and test results reflect genuine out-of-sample performance and the
train/val gap is a meaningful overfitting indicator.

Outputs
-------
  Results/classical_run/convergence_curve.png
  Results/classical_run/metrics_table.csv
  Results/classical_run/split_integrity.json

Usage
-----
  python -m classical.diagnostics
  python -m classical.diagnostics --output-dir Results/classical_run
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from shared.config import CLASSES, RESULTS_DIR, SEED
from shared.preprocessing import load_split
from classical.config import (
    CV_FOLDS,
    FEATURES_CACHE_SUBDIR,
)
from classical.features import build_feature_matrix

CLASS_NAMES: list[str] = list(CLASSES)  # ['Cyst', 'Normal', 'Stone', 'Tumor']

_XGB_BASE = dict(
    objective="multi:softprob",
    eval_metric="mlogloss",
    verbosity=0,
    random_state=SEED,
    n_jobs=-1,
)


# ── 1. Split integrity ────────────────────────────────────────────────────────

def check_split_integrity(output_dir: Path) -> dict:
    print("\n[diagnostics] === 1. Split Integrity Check ===")

    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")

    train_paths = set(train_df["abs_path"])
    val_paths   = set(val_df["abs_path"])
    test_paths  = set(test_df["abs_path"])

    tv_overlap  = train_paths & val_paths
    tt_overlap  = train_paths & test_paths
    vt_overlap  = val_paths   & test_paths
    ok = not any([tv_overlap, tt_overlap, vt_overlap])

    print(f"  Train : {len(train_df):>5} images")
    print(f"  Val   : {len(val_df):>5} images")
    print(f"  Test  : {len(test_df):>5} images")
    print(f"  Total : {len(train_df)+len(val_df)+len(test_df):>5} images")
    print(f"  Train/Val overlap  : {len(tv_overlap)}")
    print(f"  Train/Test overlap : {len(tt_overlap)}")
    print(f"  Val/Test overlap   : {len(vt_overlap)}")
    print(f"  Result : {'PASS - no leakage between splits' if ok else 'FAIL - overlapping images found!'}")

    result = {
        "n_train": len(train_df),
        "n_val":   len(val_df),
        "n_test":  len(test_df),
        "train_val_overlap":  len(tv_overlap),
        "train_test_overlap": len(tt_overlap),
        "val_test_overlap":   len(vt_overlap),
        "integrity_ok": ok,
    }
    (output_dir / "split_integrity.json").write_text(json.dumps(result, indent=2))
    return result


# ── 2. Convergence curve ──────────────────────────────────────────────────────

def plot_convergence(
    X_tr_pca: np.ndarray,
    y_tr: np.ndarray,
    X_val_pca: np.ndarray,
    y_val: np.ndarray,
    best_params: dict,
    output_dir: Path,
) -> None:
    print("\n[diagnostics] === 2. Training Loss Convergence ===")

    sw = compute_sample_weight("balanced", y_tr)
    clf = xgb.XGBClassifier(**{**_XGB_BASE, **best_params})
    clf.fit(
        X_tr_pca, y_tr,
        sample_weight=sw,
        eval_set=[(X_tr_pca, y_tr), (X_val_pca, y_val)],
        verbose=False,
    )

    evals      = clf.evals_result()
    train_loss = evals["validation_0"]["mlogloss"]
    val_loss   = evals["validation_1"]["mlogloss"]
    rounds     = list(range(1, len(train_loss) + 1))

    min_val       = min(val_loss)
    min_val_round = val_loss.index(min_val) + 1
    final_gap     = val_loss[-1] - train_loss[-1]

    print(f"  Boosting rounds    : {len(rounds)}")
    print(f"  Final train loss   : {train_loss[-1]:.4f}")
    print(f"  Final val loss     : {val_loss[-1]:.4f}")
    print(f"  Best val loss      : {min_val:.4f}  (round {min_val_round})")
    print(f"  Train/val gap      : {final_gap:.4f}")
    if final_gap > 0.05:
        print("  [!] Notable gap — potential overfitting")
    else:
        print("  Gap is small — no strong overfitting signal")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, train_loss, label="Training loss",   color="steelblue", linewidth=2)
    ax.plot(rounds, val_loss,   label="Validation loss", color="tomato",    linewidth=2)
    ax.axvline(min_val_round, color="tomato", linestyle="--", alpha=0.5,
               label=f"Best val (round {min_val_round})")
    ax.set_xlabel("Boosting round", fontsize=12)
    ax.set_ylabel("Multiclass log-loss (mlogloss)", fontsize=12)
    ax.set_title("XGBoost training convergence", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    out = output_dir / "convergence_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ── 3. Per-class metrics table ────────────────────────────────────────────────

def compute_metrics_table(
    X_tr_pca: np.ndarray,
    y_tr: np.ndarray,
    X_val_pca: np.ndarray,
    y_val: np.ndarray,
    X_test_pca: np.ndarray,
    y_test: np.ndarray,
    best_params: dict,
    output_dir: Path,
) -> None:
    print("\n[diagnostics] === 3. Per-Class Metrics Table ===")

    # Train model on training split only
    sw = compute_sample_weight("balanced", y_tr)
    clf = xgb.XGBClassifier(**{**_XGB_BASE, **best_params})
    clf.fit(X_tr_pca, y_tr, sample_weight=sw)

    y_pred_tr   = clf.predict(X_tr_pca)
    y_pred_val  = clf.predict(X_val_pca)
    y_pred_test = clf.predict(X_test_pca)

    tr_rep   = classification_report(y_tr,   y_pred_tr,   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    val_rep  = classification_report(y_val,  y_pred_val,  target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    test_rep = classification_report(y_test, y_pred_test, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    # 5-fold CV: collect per-class metrics at each fold
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_prec = {c: [] for c in CLASS_NAMES}
    cv_rec  = {c: [] for c in CLASS_NAMES}
    cv_f1   = {c: [] for c in CLASS_NAMES}

    for fold_tr_idx, fold_val_idx in cv.split(X_tr_pca, y_tr):
        X_f_tr, X_f_val = X_tr_pca[fold_tr_idx], X_tr_pca[fold_val_idx]
        y_f_tr, y_f_val = y_tr[fold_tr_idx],      y_tr[fold_val_idx]
        sw_f = compute_sample_weight("balanced", y_f_tr)
        clf_f = xgb.XGBClassifier(**{**_XGB_BASE, **best_params})
        clf_f.fit(X_f_tr, y_f_tr, sample_weight=sw_f)
        fold_rep = classification_report(
            y_f_val, clf_f.predict(X_f_val),
            target_names=CLASS_NAMES, output_dict=True, zero_division=0,
        )
        for cls in CLASS_NAMES:
            cv_prec[cls].append(fold_rep[cls]["precision"])
            cv_rec[cls].append(fold_rep[cls]["recall"])
            cv_f1[cls].append(fold_rep[cls]["f1-score"])

    # ── Print table ───────────────────────────────────────────────────────────
    rows = CLASS_NAMES + ["Macro"]
    metrics = ["Precision", "Recall", "F1"]

    # Header
    print()
    print(f"{'':12} {'--- Precision ---':^33}  {'--- Recall ---':^33}  {'--- F1 ---':^33}")
    print(f"{'Class':12} {'Train':>10} {'CV (mean)':>10} {'Test':>10}  "
          f"{'Train':>10} {'CV (mean)':>10} {'Test':>10}  "
          f"{'Train':>10} {'CV (mean)':>10} {'Test':>10}")
    print("-" * 107)

    csv_rows = []
    for cls in CLASS_NAMES:
        tr_p  = tr_rep[cls]["precision"];    cv_p  = float(np.mean(cv_prec[cls])); te_p  = test_rep[cls]["precision"]
        tr_r  = tr_rep[cls]["recall"];       cv_r  = float(np.mean(cv_rec[cls]));  te_r  = test_rep[cls]["recall"]
        tr_f  = tr_rep[cls]["f1-score"];     cv_f  = float(np.mean(cv_f1[cls]));   te_f  = test_rep[cls]["f1-score"]
        print(f"{cls:12} {tr_p:>10.4f} {cv_p:>10.4f} {te_p:>10.4f}  "
              f"{tr_r:>10.4f} {cv_r:>10.4f} {te_r:>10.4f}  "
              f"{tr_f:>10.4f} {cv_f:>10.4f} {te_f:>10.4f}")
        csv_rows.append({
            "class": cls,
            "prec_train": round(tr_p, 4), "prec_cv": round(cv_p, 4), "prec_test": round(te_p, 4),
            "rec_train":  round(tr_r, 4), "rec_cv":  round(cv_r, 4), "rec_test":  round(te_r, 4),
            "f1_train":   round(tr_f, 4), "f1_cv":   round(cv_f, 4), "f1_test":   round(te_f, 4),
        })

    # Macro row
    tr_mp  = tr_rep["macro avg"]["precision"];   te_mp  = test_rep["macro avg"]["precision"]
    tr_mr  = tr_rep["macro avg"]["recall"];      te_mr  = test_rep["macro avg"]["recall"]
    tr_mf  = tr_rep["macro avg"]["f1-score"];    te_mf  = test_rep["macro avg"]["f1-score"]
    cv_mp  = float(np.mean([np.mean(cv_prec[c]) for c in CLASS_NAMES]))
    cv_mr  = float(np.mean([np.mean(cv_rec[c])  for c in CLASS_NAMES]))
    cv_mf  = float(np.mean([np.mean(cv_f1[c])   for c in CLASS_NAMES]))
    print("-" * 107)
    print(f"{'Macro':12} {tr_mp:>10.4f} {cv_mp:>10.4f} {te_mp:>10.4f}  "
          f"{tr_mr:>10.4f} {cv_mr:>10.4f} {te_mr:>10.4f}  "
          f"{tr_mf:>10.4f} {cv_mf:>10.4f} {te_mf:>10.4f}")
    csv_rows.append({
        "class": "Macro",
        "prec_train": round(tr_mp, 4), "prec_cv": round(cv_mp, 4), "prec_test": round(te_mp, 4),
        "rec_train":  round(tr_mr, 4), "rec_cv":  round(cv_mr, 4), "rec_test":  round(te_mr, 4),
        "f1_train":   round(tr_mf, 4), "f1_cv":   round(cv_mf, 4), "f1_test":   round(te_mf, 4),
    })

    # Accuracy line
    acc_tr = accuracy_score(y_tr, y_pred_tr)
    acc_val = accuracy_score(y_val, y_pred_val)
    acc_te = accuracy_score(y_test, y_pred_test)
    print(f"\n  Accuracy  Train={acc_tr:.4f}  Val={acc_val:.4f}  Test={acc_te:.4f}")

    # Save CSV
    csv_path = output_dir / "metrics_table.csv"
    fieldnames = ["class",
                  "prec_train", "prec_cv", "prec_test",
                  "rec_train",  "rec_cv",  "rec_test",
                  "f1_train",   "f1_cv",   "f1_test"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  Saved -> {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_diagnostics(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = RESULTS_DIR / FEATURES_CACHE_SUBDIR

    # Load best model params from full training run
    log_path = output_dir / "run_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"{log_path} not found — run classical.train first.")
    with open(log_path) as f:
        run_log = json.load(f)

    model_name  = run_log["best_model"]
    best_params = run_log["best_params"]
    print(f"[diagnostics] best model={model_name}  params={best_params}")

    # 1. Split integrity
    check_split_integrity(output_dir)

    # 2. Load feature caches
    print("\n[diagnostics] loading feature caches ...")
    X_tr, y_tr     = build_feature_matrix(load_split("train"), cache_path=cache_dir / "train_frac100.npz", desc="train")
    X_val, y_val   = build_feature_matrix(load_split("val"),   cache_path=cache_dir / "val.npz",           desc="val")
    X_test, y_test = build_feature_matrix(load_split("test"),  cache_path=cache_dir / "test.npz",          desc="test")

    # Fit StandardScaler on training data only
    scaler     = StandardScaler()
    X_tr_pca   = scaler.fit_transform(X_tr)
    X_val_pca  = scaler.transform(X_val)
    X_test_pca = scaler.transform(X_test)

    # 3. Convergence curve (XGBoost only)
    if model_name == "xgb":
        plot_convergence(X_tr_pca, y_tr, X_val_pca, y_val, best_params, output_dir)
    else:
        print(f"\n[diagnostics] Convergence curve skipped (model={model_name}, only available for xgb)")

    # 4. Per-class metrics table
    compute_metrics_table(
        X_tr_pca, y_tr,
        X_val_pca, y_val,
        X_test_pca, y_test,
        best_params, output_dir,
    )

    print(f"\n[diagnostics] done. All outputs in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfitting diagnostics for classical ML")
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "classical_run"),
        help="directory containing run_log.json (same as classical.train output)",
    )
    args = parser.parse_args()
    run_diagnostics(Path(args.output_dir))


if __name__ == "__main__":
    main()
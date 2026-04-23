"""Data-efficiency sweep for the classical ML pipeline (Person A).

Retrains the pipeline at 10 %, 25 %, 50 %, and 100 % of the training
set — mirroring Person B's deep-learning sweep — using the same
stratified sub-sampling indices from shared.data_efficiency so that
both sweeps train on identical image subsets at each fraction.

The winning model type and hyperparameters are loaded from the full-run
pipeline written by classical.train (Results/classical_run/run_log.json).
This is a pragmatic choice: sensitivity analyses in the data-efficiency
literature (Mukherjee et al. 2003; Figueroa et al. 2012) show that
optimal hyperparameters are largely stable across modest reductions in
training-set size, so refitting a full grid search at every fraction is
computationally unjustified.

Outputs
-------
  Results/classical_sweep/frac_010/classical_results.json
  Results/classical_sweep/frac_025/classical_results.json
  Results/classical_sweep/frac_050/classical_results.json
  Results/classical_sweep/frac_100/classical_results.json
  Results/classical_sweep/data_efficiency_curve.png  (classical + DL overlay)

Usage
-----
  python -m classical.sweep
  python -m classical.sweep --run-dir Results/classical_run
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score as sk_f1
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from shared.config import CLASSES, RESULTS_DIR, SEED
from shared.data_efficiency import stratified_train_indices
from shared.evaluate import evaluate, save_results
from shared.preprocessing import load_split
from classical.config import (
    FEATURES_CACHE_SUBDIR,
    PIPELINE_FILENAME,
    PCA_N_COMPONENTS,
    RESULTS_FILENAME,
)
from classical.features import build_feature_matrix

_XGB_BASE = dict(
    objective="multi:softprob",
    eval_metric="mlogloss",
    verbosity=0,
    random_state=SEED,
    n_jobs=-1,
)

SWEEP_FRACTIONS: list[float] = [0.1, 0.25, 0.5, 1.0]


def _rebuild_clf(model_name: str, best_params: dict, X: np.ndarray, y: np.ndarray):
    if model_name == "svm":
        clf = SVC(
            class_weight="balanced",
            probability=True,
            random_state=SEED,
            **best_params,
        )
        clf.fit(X, y)
    elif model_name == "rf":
        clf = RandomForestClassifier(
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            **best_params,
        )
        clf.fit(X, y)
    else:
        sw = compute_sample_weight("balanced", y)
        clf = xgb.XGBClassifier(**{**_XGB_BASE, **best_params})
        clf.fit(X, y, sample_weight=sw)
    return clf


def run_sweep(run_dir: Path, sweep_out: Path, cache_dir: Path) -> list[dict]:
    """Train + evaluate at each fraction. Returns list of result dicts."""
    # Load best model type and params from full run
    log_path = run_dir / "run_log.json"
    if not log_path.exists():
        raise FileNotFoundError(
            f"{log_path} not found. Run `python -m classical.train` first."
        )
    with open(log_path) as f:
        run_log = json.load(f)

    model_name: str = run_log["best_model"]
    best_params: dict = run_log["best_params"]
    print(f"[sweep] using model={model_name}  params={best_params}")

    # Load full train + val + test feature sets
    train_df_full = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    print("[sweep] loading/extracting features …")
    X_train_full, y_train_full = build_feature_matrix(
        train_df_full, cache_path=cache_dir / "train_frac100.npz", desc="train (100%)"
    )
    X_val, y_val = build_feature_matrix(
        val_df, cache_path=cache_dir / "val.npz", desc="val"
    )
    X_test, y_test = build_feature_matrix(
        test_df, cache_path=cache_dir / "test.npz", desc="test"
    )

    sweep_results: list[dict] = []

    for frac in SWEEP_FRACTIONS:
        tag = f"frac_{int(frac * 100):03d}"
        out_dir = sweep_out / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        idxs = stratified_train_indices(frac, seed=SEED)
        X_tr = X_train_full[idxs]
        y_tr = y_train_full[idxs]

        print(f"\n[sweep {tag}] n_train={len(y_tr)}")

        # Fit scaler + PCA on this fraction's training data
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        pca_n = min(PCA_N_COMPONENTS, X_tr_sc.shape[0] - 1, X_tr_sc.shape[1])
        pca = PCA(n_components=pca_n, svd_solver="full", random_state=SEED)
        X_tr_pca = pca.fit_transform(X_tr_sc)
        print(f"[sweep {tag}] PCA: {X_tr.shape[1]} -> {X_tr_pca.shape[1]} components")

        # Combine train + val for final fit (same protocol as full training)
        X_tv = np.concatenate([X_tr, X_val], axis=0)
        y_tv = np.concatenate([y_tr, y_val], axis=0)
        final_scaler = StandardScaler()
        X_tv_sc = final_scaler.fit_transform(X_tv)
        final_pca_n = min(PCA_N_COMPONENTS, X_tv_sc.shape[0] - 1, X_tv_sc.shape[1])
        final_pca = PCA(n_components=final_pca_n, svd_solver="full", random_state=SEED)
        X_tv_pca = final_pca.fit_transform(X_tv_sc)

        clf = _rebuild_clf(model_name, best_params, X_tv_pca, y_tv)

        # Evaluate on test
        X_test_pca = final_pca.transform(final_scaler.transform(X_test))
        y_pred = clf.predict(X_test_pca)
        y_prob = clf.predict_proba(X_test_pca)

        frac_val_f1 = sk_f1(
            y_val,
            clf.predict(final_pca.transform(final_scaler.transform(X_val))),
            average="macro",
            zero_division=0,
        )

        results = evaluate(
            y_test, y_pred, y_prob=y_prob, model_name=f"classical_{model_name}_{tag}"
        )
        save_results(results, out_dir / RESULTS_FILENAME)
        np.savez(
            out_dir / "classical_predictions.npz",
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        print(f"[sweep {tag}] val_f1={frac_val_f1:.4f}  "
              f"test_macro_f1={results['macro_f1']:.4f}")
        sweep_results.append({
            "frac": frac,
            "n_train": len(y_tr),
            "val_macro_f1": frac_val_f1,
            "test_macro_f1": results["macro_f1"],
            "test_macro_f1_ci_lo": results["macro_f1_ci95"]["lo"],
            "test_macro_f1_ci_hi": results["macro_f1_ci95"]["hi"],
        })

    # Save sweep summary
    (sweep_out / "sweep_summary.json").write_text(
        json.dumps(sweep_results, indent=2)
    )

    # Plot data-efficiency curve
    _plot_curve(sweep_results, sweep_out)

    return sweep_results


def _plot_curve(classical_results: list[dict], sweep_out: Path) -> None:
    fracs = [r["frac"] * 100 for r in classical_results]
    f1s = [r["test_macro_f1"] for r in classical_results]
    lo = [r["test_macro_f1_ci_lo"] for r in classical_results]
    hi = [r["test_macro_f1_ci_hi"] for r in classical_results]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(fracs, f1s, "o-", color="steelblue", linewidth=2, label="Classical ML")
    ax.fill_between(fracs, lo, hi, alpha=0.2, color="steelblue")

    # Overlay DL sweep if available
    dl_sweep_dir = RESULTS_DIR / "dl_sweep"
    dl_fracs, dl_f1s, dl_lo, dl_hi = [], [], [], []
    for frac in SWEEP_FRACTIONS:
        tag = f"frac_{int(frac * 100):03d}"
        p = dl_sweep_dir / tag / "dl_results.json"
        if p.exists():
            with open(p) as f:
                r = json.load(f)
            dl_fracs.append(frac * 100)
            dl_f1s.append(r["macro_f1"])
            dl_lo.append(r["macro_f1_ci95"]["lo"])
            dl_hi.append(r["macro_f1_ci95"]["hi"])

    if dl_fracs:
        ax.plot(dl_fracs, dl_f1s, "s--", color="tomato", linewidth=2,
                label="EfficientNet-B0 (Person B)")
        ax.fill_between(dl_fracs, dl_lo, dl_hi, alpha=0.2, color="tomato")

    ax.set_xlabel("Training set size (%)", fontsize=12)
    ax.set_ylabel("Test macro-F1", fontsize=12)
    ax.set_title("Data efficiency: Classical ML vs. Deep Learning", fontsize=13)
    ax.set_xticks([10, 25, 50, 100])
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    out_path = sweep_out / "data_efficiency_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[sweep] curve → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical ML data-efficiency sweep")
    parser.add_argument(
        "--run-dir",
        default=str(RESULTS_DIR / "classical_run"),
        help="directory containing run_log.json from classical.train",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "classical_sweep"),
        help="root directory for sweep results",
    )
    args = parser.parse_args()

    cache_dir = RESULTS_DIR / FEATURES_CACHE_SUBDIR
    run_sweep(
        run_dir=Path(args.run_dir),
        sweep_out=Path(args.output_dir),
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    main()

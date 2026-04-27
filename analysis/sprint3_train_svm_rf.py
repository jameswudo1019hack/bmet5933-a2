"""Sprint 3 — re-fit SVM and RF on full classical features for per-classifier comparison.

Why this exists: classical/train.py picks the val-best classifier (XGBoost won at full)
and only saves *that* pipeline. To do per-classifier failure-mode comparison on the
full test set, we need pipelines for SVM and RF too. This script reuses the cached
features and the best_params already searched in classical_run_full/run_log.json.

Inputs:
  Results/classical_features_full/{train_frac100,val,test}.npz
  Results/classical_run_full/run_log.json    (contains best_params for each classifier)

Outputs:
  Results/classical_run_full_svm/{classical_pipeline.pkl, classical_predictions.npz, classical_results.json}
  Results/classical_run_full_rf/{classical_pipeline.pkl, classical_predictions.npz, classical_results.json}

Usage:
  python -m analysis.sprint3_train_svm_rf
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from shared.config import CLASSES, RESULTS_DIR, SEED
from shared.evaluate import evaluate, print_summary, save_results
from classical.config import PCA_N_COMPONENTS, PIPELINE_FILENAME, RESULTS_FILENAME
from classical.train import _fit_final


FEATURES_DIR = RESULTS_DIR / "classical_features_full"
RUN_LOG = RESULTS_DIR / "classical_run_full" / "run_log.json"


def main() -> None:
    train_npz = np.load(FEATURES_DIR / "train_frac100.npz")
    val_npz   = np.load(FEATURES_DIR / "val.npz")
    test_npz  = np.load(FEATURES_DIR / "test.npz")
    X_train, y_train = train_npz["X"], train_npz["y"]
    X_val,   y_val   = val_npz["X"],   val_npz["y"]
    X_test,  y_test  = test_npz["X"],  test_npz["y"]

    print(f"[load] train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    with open(RUN_LOG) as f:
        run_log = json.load(f)

    # Same protocol as classical/train.py for the winner: refit scaler + PCA on
    # train+val combined, then fit the classifier on train+val.
    X_tv = np.concatenate([X_train, X_val], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)

    final_scaler = StandardScaler()
    X_tv_sc = final_scaler.fit_transform(X_tv)
    pca_n = min(PCA_N_COMPONENTS, X_tv_sc.shape[0] - 1, X_tv_sc.shape[1])
    final_pca = PCA(n_components=pca_n, svd_solver="full", random_state=SEED)
    final_pca.fit(X_tv_sc)
    print(f"[fit] PCA: {X_tv.shape[1]} -> {final_pca.n_components_} components")

    for model_name in ("svm", "rf"):
        out_dir = RESULTS_DIR / f"classical_run_full_{model_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        best_params = run_log["classifiers"][model_name]["best_params"]
        print(f"\n[{model_name}] best_params = {best_params}")

        clf = _fit_final(model_name, best_params, X_tv, y_tv, final_scaler, final_pca)

        X_test_pca = final_pca.transform(final_scaler.transform(X_test))
        y_pred = clf.predict(X_test_pca)
        y_prob = clf.predict_proba(X_test_pca)

        np.savez(
            out_dir / "classical_predictions.npz",
            y_true=y_test, y_pred=y_pred, y_prob=y_prob,
        )

        results = evaluate(
            y_test, y_pred, y_prob=y_prob, model_name=f"classical_{model_name}",
        )
        print_summary(results)
        save_results(results, out_dir / RESULTS_FILENAME)

        pipeline = {
            "scaler": final_scaler,
            "pca": final_pca,
            "classifier": clf,
            "model_name": model_name,
            "best_params": best_params,
            "n_raw_features": int(X_train.shape[1]),
            "n_pca_components": int(final_pca.n_components_),
            "classes": list(CLASSES),
        }
        with open(out_dir / PIPELINE_FILENAME, "wb") as f:
            pickle.dump(pipeline, f)

        print(f"[{model_name}] saved -> {out_dir}")


if __name__ == "__main__":
    main()

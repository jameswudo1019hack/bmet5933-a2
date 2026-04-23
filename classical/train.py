"""Train the classical ML pipeline (Person A).

Pipeline
--------
  shared.preprocessing.load_image()
  -> CLAHE + feature extraction  (classical.features)
  -> StandardScaler
  -> PCA (50 components, capacity-constrained to reduce slice memorisation)
  -> Grid-searched classifier

Three classifiers are evaluated:
  • SVM (linear kernel, small C)  class_weight='balanced'  (Cortes & Vapnik 1995)
  • Random Forest  class_weight='balanced'  (Breiman 2001)
  • XGBoost        sample_weight per fold   (Chen & Guestrin 2016)

Selection criterion: macro-F1 on the held-out validation split (Phase 0 §5).
The winning model is retrained on train + val combined before writing the
pipeline dict to disk.

Usage
-----
  python -m classical.train                   # full run
  python -m classical.train --smoke           # tiny 64-sample sanity check
  python -m classical.train --output-dir ...  # custom output location
  python -m classical.train --train-frac 0.5  # data-efficiency sweep point
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score as sk_f1
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from shared.config import CLASSES, RESULTS_DIR, SEED
from shared.data_efficiency import stratified_train_indices
from shared.preprocessing import load_split
from classical.config import (
    CV_FOLDS,
    CV_SCORING,
    FEATURES_CACHE_SUBDIR,
    PIPELINE_FILENAME,
    PCA_N_COMPONENTS,
    RF_PARAM_GRID,
    SVM_PARAM_GRID,
    XGB_PARAM_GRID,
)
from classical.features import build_feature_matrix


# ── XGBoost base parameters (non-tuned) ───────────────────────────────────────
_XGB_BASE = dict(
    objective="multi:softprob",
    eval_metric="mlogloss",
    verbosity=0,
    random_state=SEED,
    n_jobs=-1,
)


def _cv_xgb(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_splits: int = CV_FOLDS,
) -> float:
    """Stratified k-fold CV for XGBoost with per-fold balanced sample weights.

    Passes sample_weight directly to XGBClassifier.fit() inside each fold,
    avoiding sklearn's metadata-routing complexity entirely.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores: list[float] = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        sw = compute_sample_weight("balanced", y_tr)
        clf = xgb.XGBClassifier(**{**_XGB_BASE, **params})
        clf.fit(X_tr, y_tr, sample_weight=sw)
        y_pred = clf.predict(X_va)
        scores.append(sk_f1(y_va, y_pred, average="macro", zero_division=0))
    return float(np.mean(scores))


def _grid_search_svm(
    X: np.ndarray, y: np.ndarray, param_grid: dict | None = None
) -> tuple[SVC, dict, float]:
    """Grid search for SVM; returns (best_estimator, best_params, cv_score).

    Kernel is specified inside the param_grid so the grid can include both
    'linear' and 'rbf' combinations without hardcoding here.
    """
    grid = param_grid if param_grid is not None else SVM_PARAM_GRID
    # Base SVC with no kernel pre-set; kernel comes from the grid
    clf = SVC(class_weight="balanced", probability=True, random_state=SEED)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    gs = GridSearchCV(clf, grid, scoring=CV_SCORING, cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


def _grid_search_rf(
    X: np.ndarray, y: np.ndarray, param_grid: dict | None = None
) -> tuple[RandomForestClassifier, dict, float]:
    """Grid search for Random Forest; returns (best_estimator, best_params, cv_score)."""
    grid = param_grid if param_grid is not None else RF_PARAM_GRID
    clf = RandomForestClassifier(
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    gs = GridSearchCV(clf, grid, scoring=CV_SCORING, cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


def _grid_search_xgb(
    X: np.ndarray, y: np.ndarray, param_grid: dict | None = None
) -> tuple[dict, float]:
    """Manual grid search for XGBoost; returns (best_params, best_cv_score)."""
    grid = param_grid if param_grid is not None else XGB_PARAM_GRID
    best_score = -1.0
    best_params: dict = {}

    param_list = list(ParameterGrid(grid))
    print(f"[xgb grid search] {len(param_list)} combinations × {CV_FOLDS} folds")

    for i, params in enumerate(param_list, 1):
        score = _cv_xgb(X, y, params)
        print(f"  [{i}/{len(param_list)}] {params}  cv_macro_f1={score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params.copy()

    return best_params, best_score


def _fit_final(
    model_name: str,
    best_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
) -> object:
    """Fit the winning classifier on scaler+PCA-transformed data."""
    X_pca = pca.transform(scaler.transform(X))
    if model_name == "svm":
        clf = SVC(
            class_weight="balanced",
            probability=True,
            random_state=SEED,
            **best_params,
        )
        clf.fit(X_pca, y)
    elif model_name == "rf":
        clf = RandomForestClassifier(
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            **best_params,
        )
        clf.fit(X_pca, y)
    else:  # xgb
        sw = compute_sample_weight("balanced", y)
        clf = xgb.XGBClassifier(**{**_XGB_BASE, **best_params})
        clf.fit(X_pca, y, sample_weight=sw)
    return clf


def train(
    output_dir: Path,
    smoke: bool = False,
    train_frac: float = 1.0,
    seed: int = SEED,
    cache_dir: Path | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = RESULTS_DIR / FEATURES_CACHE_SUBDIR

    print(f"[train] output={output_dir}  smoke={smoke}  train_frac={train_frac}")

    # ── Load split DataFrames ─────────────────────────────────────────────────
    train_df = load_split("train")
    val_df = load_split("val")

    if train_frac < 1.0:
        idxs = stratified_train_indices(train_frac, seed=seed)
        train_df = train_df.iloc[idxs].reset_index(drop=True)
        print(f"[train] subsampled train to {len(train_df)} images ({train_frac:.0%})")

    if smoke:
        train_df = train_df.iloc[:64].reset_index(drop=True)
        val_df = val_df.iloc[:32].reset_index(drop=True)
        print("[train] smoke mode: 64 train / 32 val samples")

    # ── Feature extraction (with disk cache unless smoke) ─────────────────────
    frac_tag = f"frac{int(train_frac * 100):03d}"
    train_cache = (
        None if smoke else cache_dir / f"train_{frac_tag}.npz"
    )
    val_cache = None if smoke else cache_dir / "val.npz"

    print("[train] extracting train features …")
    X_train, y_train = build_feature_matrix(
        train_df, cache_path=train_cache, desc="train features"
    )
    print("[train] extracting val features …")
    X_val, y_val = build_feature_matrix(
        val_df, cache_path=val_cache, desc="val features"
    )

    n_raw = X_train.shape[1]
    print(f"[train] raw feature dims: {n_raw}")

    # ── Preprocessing: StandardScaler + PCA fitted on train only ─────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    pca_n = min(PCA_N_COMPONENTS, X_train_sc.shape[0] - 1, X_train_sc.shape[1]) if not smoke else min(20, X_train_sc.shape[0] - 1)
    pca = PCA(n_components=pca_n, svd_solver="full", random_state=seed)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca = pca.transform(scaler.transform(X_val))

    n_components = X_train_pca.shape[1]
    print(f"[train] PCA: {n_raw} -> {n_components} components (capacity-constrained)")

    # ── Grid search ───────────────────────────────────────────────────────────
    t0 = time.time()
    log: dict = {
        "train_frac": train_frac,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_raw_features": n_raw,
        "n_pca_components": n_components,
        "seed": seed,
        "smoke": smoke,
        "classifiers": {},
    }

    if smoke:
        # Tiny grid for smoke test only
        svm_grid = {"C": [1.0], "gamma": ["scale"]}
        rf_grid = {"n_estimators": [10], "max_depth": [5], "min_samples_split": [2]}
        xgb_grid = {"n_estimators": [10], "max_depth": [3], "learning_rate": [0.1]}
    else:
        svm_grid = SVM_PARAM_GRID
        rf_grid = RF_PARAM_GRID
        xgb_grid = XGB_PARAM_GRID

    # SVM
    print("\n[train] === SVM grid search ===")
    svm_clf, svm_params, svm_cv = _grid_search_svm(X_train_pca, y_train, svm_grid)
    svm_val_f1 = sk_f1(y_val, svm_clf.predict(X_val_pca), average="macro", zero_division=0)
    print(f"[svm]  best params={svm_params}  cv_f1={svm_cv:.4f}  val_f1={svm_val_f1:.4f}")
    log["classifiers"]["svm"] = {
        "best_params": svm_params, "cv_f1": svm_cv, "val_f1": svm_val_f1
    }

    # Random Forest
    print("\n[train] === Random Forest grid search ===")
    rf_clf, rf_params, rf_cv = _grid_search_rf(X_train_pca, y_train, rf_grid)
    rf_val_f1 = sk_f1(y_val, rf_clf.predict(X_val_pca), average="macro", zero_division=0)
    print(f"[rf]   best params={rf_params}  cv_f1={rf_cv:.4f}  val_f1={rf_val_f1:.4f}")
    log["classifiers"]["rf"] = {
        "best_params": rf_params, "cv_f1": rf_cv, "val_f1": rf_val_f1
    }

    # XGBoost
    print("\n[train] === XGBoost grid search ===")
    xgb_params, xgb_cv = _grid_search_xgb(X_train_pca, y_train, xgb_grid)

    xgb_clf = xgb.XGBClassifier(**{**_XGB_BASE, **xgb_params})
    sw = compute_sample_weight("balanced", y_train)
    xgb_clf.fit(X_train_pca, y_train, sample_weight=sw)
    xgb_val_f1 = sk_f1(y_val, xgb_clf.predict(X_val_pca), average="macro", zero_division=0)
    print(f"[xgb]  best params={xgb_params}  cv_f1={xgb_cv:.4f}  val_f1={xgb_val_f1:.4f}")
    log["classifiers"]["xgb"] = {
        "best_params": xgb_params, "cv_f1": xgb_cv, "val_f1": xgb_val_f1
    }

    # ── Select winner by val macro-F1 ─────────────────────────────────────────
    candidates = {
        "svm": (svm_val_f1, svm_params),
        "rf":  (rf_val_f1,  rf_params),
        "xgb": (xgb_val_f1, xgb_params),
    }
    best_name = max(candidates, key=lambda k: candidates[k][0])
    best_val_f1, best_params = candidates[best_name]
    print(f"\n[train] winner: {best_name}  val_macro_f1={best_val_f1:.4f}")
    log["best_model"] = best_name
    log["best_params"] = best_params
    log["best_val_f1"] = best_val_f1

    # ── Retrain on train + val combined ───────────────────────────────────────
    X_tv = np.concatenate([X_train, X_val], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)

    final_scaler = StandardScaler()
    X_tv_sc = final_scaler.fit_transform(X_tv)
    final_pca = PCA(n_components=min(pca_n, len(y_tv) - 1), svd_solver="full", random_state=seed)
    final_pca.fit(X_tv_sc)

    print(f"[train] retraining {best_name} on train+val ({len(y_tv)} samples) …")
    final_clf = _fit_final(best_name, best_params, X_tv, y_tv, final_scaler, final_pca)

    # ── Save pipeline ─────────────────────────────────────────────────────────
    pipeline = {
        "scaler": final_scaler,
        "pca": final_pca,
        "classifier": final_clf,
        "model_name": best_name,
        "best_params": best_params,
        "n_raw_features": n_raw,
        "n_pca_components": final_pca.n_components_,
        "classes": list(CLASSES),
    }
    pipeline_path = output_dir / PIPELINE_FILENAME
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    log["wall_time_sec"] = time.time() - t0
    log["pipeline_path"] = str(pipeline_path)
    log_path = output_dir / "run_log.json"
    log_path.write_text(json.dumps(log, indent=2))

    print(f"\n[train] done  best={best_name}  val_f1={best_val_f1:.4f}  "
          f"wall={log['wall_time_sec']:.1f}s")
    print(f"[train] pipeline -> {pipeline_path}")
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical ML pipeline")
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "classical_run"),
        help="directory for pipeline pickle and run_log.json",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run (64 train / 32 val) to verify the pipeline runs",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=1.0,
        help="stratified fraction of train split (for data-efficiency sweep)",
    )
    args = parser.parse_args()

    train(
        output_dir=Path(args.output_dir),
        smoke=args.smoke,
        train_frac=args.train_frac,
    )


if __name__ == "__main__":
    main()

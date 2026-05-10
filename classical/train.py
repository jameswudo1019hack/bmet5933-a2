"""Train the classical ML pipeline (Person A).

Pipeline
--------
  shared.preprocessing.load_image()
  -> CLAHE + feature extraction  (classical.features)
  -> StandardScaler
  -> Grid-searched classifier (SVM / XGBoost)

Two classifiers are evaluated:
  * SVM (linear kernel, small C)  class_weight='balanced'  (Cortes & Vapnik 1995)
  * XGBoost        sample_weight per fold   (Chen & Guestrin 2016)

Selection criterion: macro-F1 on the held-out validation split (Phase 0 §5).
The winning model is retrained on train + val combined before writing the
pipeline dict to disk.

After winner selection, per-class Precision/Recall/F1 is printed for both
the training set and 5-fold cross-validation so overfitting can be assessed
without running a separate diagnostics script.

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
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score as sk_f1
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from shared.config import CLASSES, GROUP_SIZE, RESULTS_DIR, SEED
from shared.data_efficiency import stratified_train_indices
from shared.preprocessing import load_split
from shared.split import group_ids_from_filenames
from classical.config import (
    CV_FOLDS,
    CV_SCORING,
    FEATURES_CACHE_SUBDIR,
    PIPELINE_FILENAME,
    PCA_N_COMPONENTS,
    RF_CLASS_WEIGHT,
    RF_PARAM_GRID,
    SVM_PARAM_GRID,
    XGB_PARAM_GRID,
)
from classical.features import build_feature_matrix
from classical.metrics_utils import (
    get_train_metrics,
    print_metrics_table,
    run_cv_per_class,
)


def _apply_smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upsample minority classes using BorderlineSMOTE (targets boundary-region samples)."""
    sm = BorderlineSMOTE(sampling_strategy="not majority", random_state=SEED, kind="borderline-1")
    return sm.fit_resample(X, y)


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
    groups: list[str] | None = None,
    n_splits: int = CV_FOLDS,
    use_smote: bool = False,
) -> float:
    """Group-aware k-fold CV for XGBoost with per-fold balanced sample weights."""
    from sklearn.model_selection import StratifiedKFold
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        split_iter = cv.split(X, y, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        split_iter = cv.split(X, y)
    scores: list[float] = []
    for train_idx, val_idx in split_iter:
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        if use_smote:
            X_tr, y_tr = _apply_smote(X_tr, y_tr)
        sw = compute_sample_weight("balanced", y_tr)
        clf = xgb.XGBClassifier(**{**_XGB_BASE, **params})
        clf.fit(X_tr, y_tr, sample_weight=sw)
        y_pred = clf.predict(X_va)
        scores.append(sk_f1(y_va, y_pred, average="macro", zero_division=0))
    return float(np.mean(scores))


def _grid_search_svm(
    X: np.ndarray, y: np.ndarray,
    groups: list[str] | None = None,
    param_grid: dict | None = None,
    use_smote: bool = False,
) -> tuple[object, dict, float]:
    from sklearn.model_selection import StratifiedKFold
    grid = param_grid if param_grid is not None else SVM_PARAM_GRID

    if use_smote:
        # Wrap in imblearn Pipeline so SMOTE runs inside each CV fold.
        # Prefix param names with "svm__" to match Pipeline step naming.
        smote = BorderlineSMOTE(sampling_strategy="not majority", random_state=SEED, kind="borderline-1")
        estimator = ImbPipeline([("smote", smote),
                                  ("svm", SVC(class_weight="balanced", probability=True, random_state=SEED))])
        if isinstance(grid, list):
            pg = [{f"svm__{k}": v for k, v in g.items()} for g in grid]
        else:
            pg = {f"svm__{k}": v for k, v in grid.items()}
    else:
        estimator = SVC(class_weight="balanced", probability=True, random_state=SEED)
        pg = grid

    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        gs = GridSearchCV(estimator, pg, scoring=CV_SCORING, cv=cv, n_jobs=-1, verbose=1)
        gs.fit(X, y, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        gs = GridSearchCV(estimator, pg, scoring=CV_SCORING, cv=cv, n_jobs=-1, verbose=1)
        gs.fit(X, y)

    # Strip "svm__" prefix so downstream code always sees plain param names.
    clean_params = {k.replace("svm__", ""): v for k, v in gs.best_params_.items()}
    return gs.best_estimator_, clean_params, float(gs.best_score_)


def _grid_search_xgb(
    X: np.ndarray, y: np.ndarray,
    groups: list[str] | None = None,
    param_grid: dict | None = None,
    use_smote: bool = False,
) -> tuple[dict, float]:
    grid = param_grid if param_grid is not None else XGB_PARAM_GRID
    best_score = -1.0
    best_params: dict = {}
    param_list = list(ParameterGrid(grid))
    print(f"[xgb grid search] {len(param_list)} combinations x {CV_FOLDS} folds")
    for i, params in enumerate(param_list, 1):
        score = _cv_xgb(X, y, params, groups=groups, use_smote=use_smote)
        print(f"  [{i}/{len(param_list)}] {params}  cv_macro_f1={score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params.copy()
    return best_params, best_score


def _cv_rf(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    groups: list[str] | None = None,
    n_splits: int = CV_FOLDS,
    use_smote: bool = False,
) -> float:
    """Group-aware k-fold CV for Random Forest."""
    from sklearn.model_selection import StratifiedKFold
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        split_iter = cv.split(X, y, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        split_iter = cv.split(X, y)
    scores: list[float] = []
    for train_idx, val_idx in split_iter:
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        if use_smote:
            X_tr, y_tr = _apply_smote(X_tr, y_tr)
        clf = RandomForestClassifier(
            class_weight=RF_CLASS_WEIGHT, random_state=SEED, n_jobs=-1, **params,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_va)
        scores.append(sk_f1(y_va, y_pred, average="macro", zero_division=0))
    return float(np.mean(scores))


def _grid_search_rf(
    X: np.ndarray, y: np.ndarray,
    groups: list[str] | None = None,
    param_grid: dict | None = None,
    use_smote: bool = False,
) -> tuple[RandomForestClassifier, dict, float]:
    grid = param_grid if param_grid is not None else RF_PARAM_GRID
    best_score = -1.0
    best_params: dict = {}
    param_list = list(ParameterGrid(grid))
    print(f"[rf grid search] {len(param_list)} combinations x {CV_FOLDS} folds")
    for i, params in enumerate(param_list, 1):
        score = _cv_rf(X, y, params, groups=groups, use_smote=use_smote)
        print(f"  [{i}/{len(param_list)}] {params}  cv_macro_f1={score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params.copy()
    X_fit, y_fit = (_apply_smote(X, y) if use_smote else (X, y))
    best_clf = RandomForestClassifier(
        class_weight="balanced", random_state=SEED, n_jobs=-1, **best_params,
    )
    best_clf.fit(X_fit, y_fit)
    return best_clf, best_params, best_score


def _fit_final(
    model_name: str,
    best_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    pca: PCA | None = None,
    use_smote: bool = False,
) -> object:
    """Fit the winning classifier on scaler-transformed (and optionally PCA-reduced) data."""
    X_sc = scaler.transform(X)
    X_sc = pca.transform(X_sc) if pca is not None else X_sc
    if use_smote and model_name != "svm":
        X_sc, y = _apply_smote(X_sc, y)
    if model_name == "svm":
        clf = SVC(
            class_weight="balanced", probability=True,
            random_state=SEED, **best_params,
        )
        clf.fit(X_sc, y)
    elif model_name == "rf":
        clf = RandomForestClassifier(
            class_weight=RF_CLASS_WEIGHT, random_state=SEED, n_jobs=-1, **best_params,
        )
        clf.fit(X_sc, y)
    else:
        sw = compute_sample_weight("balanced", y)
        clf = xgb.XGBClassifier(**{**_XGB_BASE, **best_params})
        clf.fit(X_sc, y, sample_weight=sw)
    return clf


def train(
    output_dir: Path,
    smoke: bool = False,
    train_frac: float = 1.0,
    seed: int = SEED,
    cache_dir: Path | None = None,
    split_csv: Path | None = None,
    dataset_root: Path | None = None,
    n_jobs: int = 1,
    use_pca: bool = False,
    use_smote: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = RESULTS_DIR / FEATURES_CACHE_SUBDIR

    print(f"[train] output={output_dir}  smoke={smoke}  train_frac={train_frac}  use_smote={use_smote}")

    # ── Load split DataFrames ─────────────────────────────────────────────────
    train_df = load_split("train", split_csv=split_csv, dataset_root=dataset_root)
    val_df   = load_split("val",   split_csv=split_csv, dataset_root=dataset_root)

    if train_frac < 1.0:
        idxs = stratified_train_indices(
            train_frac, seed=seed,
            split_csv=split_csv, dataset_root=dataset_root,
        )
        train_df = train_df.iloc[idxs].reset_index(drop=True)
        print(f"[train] subsampled train to {len(train_df)} images ({train_frac:.0%})")

    if smoke:
        train_df = train_df.iloc[:64].reset_index(drop=True)
        val_df   = val_df.iloc[:32].reset_index(drop=True)
        print("[train] smoke mode: 64 train / 32 val samples")

    # ── Feature extraction (with disk cache unless smoke) ─────────────────────
    frac_tag    = f"frac{int(train_frac * 100):03d}"
    train_cache = None if smoke else cache_dir / f"train_{frac_tag}.npz"
    val_cache   = None if smoke else cache_dir / "val.npz"

    print("[train] extracting train features ...")
    X_train, y_train = build_feature_matrix(
        train_df, cache_path=train_cache, desc="train features", n_jobs=n_jobs,
    )
    print("[train] extracting val features ...")
    X_val, y_val = build_feature_matrix(
        val_df, cache_path=val_cache, desc="val features", n_jobs=n_jobs,
    )

    n_raw = X_train.shape[1]
    print(f"[train] raw feature dims: {n_raw}")

    # Patient group IDs aligned row-for-row with X_train (None in smoke mode)
    train_groups: list[str] | None = None if smoke else group_ids_from_filenames(
        train_df["filename"].tolist(), train_df["class"].tolist(), GROUP_SIZE,
    )

    # ── Preprocessing: StandardScaler (+ optional PCA) fitted on train only ──
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    print(f"[train] StandardScaler fitted on {len(y_train)} training samples")

    pca = None
    if use_pca:
        pca_n = min(PCA_N_COMPONENTS, X_train_sc.shape[0] - 1, X_train_sc.shape[1]) if not smoke else min(20, X_train_sc.shape[0] - 1)
        pca = PCA(n_components=pca_n, svd_solver="full", random_state=seed)
        X_train_sc = pca.fit_transform(X_train_sc)
        X_val_sc   = pca.transform(X_val_sc)
        print(f"[train] PCA: {n_raw} -> {X_train_sc.shape[1]} components")

    # ── Grid search ───────────────────────────────────────────────────────────
    t0  = time.time()
    log: dict = {
        "train_frac": train_frac,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_raw_features": n_raw,
        "use_pca": use_pca,
        "n_pca_components": X_train_sc.shape[1] if use_pca else None,
        "seed": seed,
        "smoke": smoke,
        "use_smote": use_smote,
        "classifiers": {},
    }

    if smoke:
        svm_grid = {"C": [1.0], "kernel": ["linear"]}
        xgb_grid = {"n_estimators": [10], "max_depth": [3], "learning_rate": [0.1]}
        rf_grid  = {"n_estimators": [10], "max_depth": [3]}
    else:
        svm_grid = SVM_PARAM_GRID
        xgb_grid = XGB_PARAM_GRID
        rf_grid  = RF_PARAM_GRID

    # SVM — class_weight='balanced' handles imbalance natively; SMOTE not applied
    print("\n[train] === SVM grid search ===")
    svm_clf, svm_params, svm_cv = _grid_search_svm(X_train_sc, y_train, train_groups, svm_grid, use_smote=False)
    svm_val_f1 = sk_f1(y_val, svm_clf.predict(X_val_sc), average="macro", zero_division=0)
    print(f"[svm]  best params={svm_params}  cv_f1={svm_cv:.4f}  val_f1={svm_val_f1:.4f}")
    log["classifiers"]["svm"] = {"best_params": svm_params, "cv_f1": svm_cv, "val_f1": svm_val_f1}

    # XGBoost — temporarily skipped; re-enable by uncommenting for final comparison
    # print("\n[train] === XGBoost grid search ===")
    # xgb_params, xgb_cv = _grid_search_xgb(X_train_sc, y_train, train_groups, xgb_grid, use_smote=use_smote)
    # xgb_clf = xgb.XGBClassifier(**{**_XGB_BASE, **xgb_params})
    # X_xgb, y_xgb = (_apply_smote(X_train_sc, y_train) if use_smote else (X_train_sc, y_train))
    # sw = compute_sample_weight("balanced", y_xgb)
    # xgb_clf.fit(X_xgb, y_xgb, sample_weight=sw)
    # xgb_val_f1 = sk_f1(y_val, xgb_clf.predict(X_val_sc), average="macro", zero_division=0)
    # print(f"[xgb]  best params={xgb_params}  cv_f1={xgb_cv:.4f}  val_f1={xgb_val_f1:.4f}")
    # log["classifiers"]["xgb"] = {"best_params": xgb_params, "cv_f1": xgb_cv, "val_f1": xgb_val_f1}

    # Random Forest
    print("\n[train] === Random Forest grid search ===")
    rf_clf, rf_params, rf_cv = _grid_search_rf(X_train_sc, y_train, train_groups, rf_grid, use_smote=use_smote)
    rf_val_f1 = sk_f1(y_val, rf_clf.predict(X_val_sc), average="macro", zero_division=0)
    print(f"[rf]   best params={rf_params}  cv_f1={rf_cv:.4f}  val_f1={rf_val_f1:.4f}")
    log["classifiers"]["rf"] = {"best_params": rf_params, "cv_f1": rf_cv, "val_f1": rf_val_f1}

    # ── Select winner by val macro-F1 ─────────────────────────────────────────
    candidates = {
        "svm": (svm_val_f1, svm_params, svm_clf),
        # "xgb": (xgb_val_f1, xgb_params, xgb_clf),  # re-enable for final comparison
        "rf":  (rf_val_f1,  rf_params,  rf_clf),
    }
    best_name = max(candidates, key=lambda k: candidates[k][0])
    best_val_f1, best_params, best_clf_train = candidates[best_name]
    print(f"\n[train] winner: {best_name}  val_macro_f1={best_val_f1:.4f}")
    log["best_model"]  = best_name
    log["best_params"] = best_params
    log["best_val_f1"] = best_val_f1

    # ── Per-class metrics table (Train + CV) ──────────────────────────────────
    print("\n[train] computing per-class metrics (train + 5-fold CV) ...")
    train_metrics, train_acc = get_train_metrics(best_clf_train, X_train_sc, y_train)
    cv_metrics, cv_acc       = run_cv_per_class(best_name, best_params, X_train_sc, y_train, groups=train_groups, use_smote=use_smote)

    print("\n[train] === Per-class metrics: Train / Cross-Validation ===")
    print_metrics_table(train_metrics, cv_metrics, train_acc=train_acc, cv_acc=cv_acc)

    # Save so predict.py can load and display the full 3-column table
    train_cv_data = {
        "model_name":     best_name,
        "train_metrics":  train_metrics,
        "cv_metrics":     cv_metrics,
        "train_accuracy": train_acc,
        "cv_accuracy":    cv_acc,
    }
    (output_dir / "train_cv_metrics.json").write_text(json.dumps(train_cv_data, indent=2))

    # ── Retrain on train + val combined ───────────────────────────────────────
    X_tv         = np.concatenate([X_train, X_val], axis=0)
    y_tv         = np.concatenate([y_train, y_val], axis=0)
    final_scaler = StandardScaler()
    X_tv_sc      = final_scaler.fit_transform(X_tv)

    final_pca = None
    if use_pca:
        final_pca_n = min(PCA_N_COMPONENTS, X_tv_sc.shape[0] - 1, X_tv_sc.shape[1])
        final_pca = PCA(n_components=final_pca_n, svd_solver="full", random_state=seed)
        final_pca.fit(X_tv_sc)

    print(f"[train] retraining {best_name} on train+val ({len(y_tv)} samples) ...")
    final_clf = _fit_final(best_name, best_params, X_tv, y_tv, final_scaler, final_pca, use_smote=use_smote)

    # ── Save pipeline ─────────────────────────────────────────────────────────
    pipeline = {
        "scaler":         final_scaler,
        "pca":            final_pca,
        "classifier":     final_clf,
        "model_name":     best_name,
        "best_params":    best_params,
        "n_raw_features": n_raw,
        "classes":        list(CLASSES),
    }
    pipeline_path = output_dir / PIPELINE_FILENAME
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    log["wall_time_sec"] = time.time() - t0
    log["pipeline_path"] = str(pipeline_path)
    (output_dir / "run_log.json").write_text(json.dumps(log, indent=2))

    print(f"\n[train] done  best={best_name}  val_f1={best_val_f1:.4f}  "
          f"wall={log['wall_time_sec']:.1f}s")
    print(f"[train] pipeline -> {pipeline_path}")
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical ML pipeline")
    parser.add_argument(
        "--output-dir", default=str(RESULTS_DIR / "classical_run"),
        help="directory for pipeline pickle and run_log.json",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run (64 train / 32 val) to verify the pipeline runs")
    parser.add_argument("--train-frac", type=float, default=1.0,
                        help="stratified fraction of train split (for data-efficiency sweep)")
    parser.add_argument("--split-csv", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--features-cache-dir", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--use-pca", action="store_true",
                        help="apply PCA (50 components) after StandardScaler")
    parser.add_argument("--use-smote", action="store_true",
                        help="apply SMOTE inside each CV fold to balance minority classes")
    args = parser.parse_args()

    train(
        output_dir=Path(args.output_dir),
        smoke=args.smoke,
        train_frac=args.train_frac,
        cache_dir=Path(args.features_cache_dir) if args.features_cache_dir else None,
        split_csv=Path(args.split_csv) if args.split_csv else None,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        n_jobs=args.n_jobs,
        use_pca=args.use_pca,
        use_smote=args.use_smote,
    )


if __name__ == "__main__":
    main()
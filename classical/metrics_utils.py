"""Shared per-class metrics helpers for the classical pipeline.

Used by train.py (train + CV metrics), predict.py (test metrics + full table),
and diagnostics.py.  Centralising here avoids duplicating the table-printing
logic across files.
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from shared.config import CLASSES, SEED
from classical.config import CV_FOLDS, RF_CLASS_WEIGHT

CLASS_NAMES: list[str] = list(CLASSES)

_XGB_BASE = dict(
    objective="multi:softprob",
    eval_metric="mlogloss",
    verbosity=0,
    random_state=SEED,
    n_jobs=-1,
)


def report_to_dict(report: dict) -> dict:
    """Normalise classification_report output to {cls: {precision, recall, f1}}."""
    return {
        cls: {
            "precision": report[cls]["precision"],
            "recall":    report[cls]["recall"],
            "f1":        report[cls]["f1-score"],
        }
        for cls in CLASS_NAMES + ["macro avg"]
    }


def get_train_metrics(clf, X: np.ndarray, y: np.ndarray) -> tuple[dict, float]:
    """Per-class metrics + accuracy on the training set."""
    y_pred = clf.predict(X)
    report = classification_report(
        y, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    return report_to_dict(report), float(accuracy_score(y, y_pred))


def run_cv_per_class(
    model_name: str,
    best_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str] | None = None,
    n_folds: int = CV_FOLDS,
    use_smote: bool = False,
) -> tuple[dict, float]:
    """Group-aware k-fold CV; returns ({cls: {precision, recall, f1}}, mean_accuracy).

    When groups is provided (patient group IDs aligned with X rows) uses
    StratifiedGroupKFold so no patient's slices appear in both fold-train
    and fold-val.  Falls back to StratifiedKFold if groups is None.
    """
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        split_iter = cv.split(X, y, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        split_iter = cv.split(X, y)
    fold_prec = {c: [] for c in CLASS_NAMES}
    fold_rec  = {c: [] for c in CLASS_NAMES}
    fold_f1   = {c: [] for c in CLASS_NAMES}
    fold_acc: list[float] = []

    for tr_idx, val_idx in split_iter:
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if use_smote and model_name != "svm":
            sm = BorderlineSMOTE(sampling_strategy="not majority", random_state=SEED, kind="borderline-1")
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

        if model_name == "svm":
            clf_f = SVC(
                class_weight="balanced", probability=True,
                random_state=SEED, **best_params,
            )
            clf_f.fit(X_tr, y_tr)
        elif model_name == "rf":
            clf_f = RandomForestClassifier(
                class_weight=RF_CLASS_WEIGHT, random_state=SEED, n_jobs=-1, **best_params,
            )
            clf_f.fit(X_tr, y_tr)
        else:
            sw = compute_sample_weight("balanced", y_tr)
            clf_f = xgb.XGBClassifier(**{**_XGB_BASE, **best_params})
            clf_f.fit(X_tr, y_tr, sample_weight=sw)

        y_pred = clf_f.predict(X_val)
        rep = classification_report(
            y_val, y_pred,
            target_names=CLASS_NAMES, output_dict=True, zero_division=0,
        )
        fold_acc.append(float(accuracy_score(y_val, y_pred)))
        for cls in CLASS_NAMES:
            fold_prec[cls].append(rep[cls]["precision"])
            fold_rec[cls].append(rep[cls]["recall"])
            fold_f1[cls].append(rep[cls]["f1-score"])

    result: dict = {}
    for cls in CLASS_NAMES:
        result[cls] = {
            "precision": float(np.mean(fold_prec[cls])),
            "recall":    float(np.mean(fold_rec[cls])),
            "f1":        float(np.mean(fold_f1[cls])),
        }
    result["macro avg"] = {
        "precision": float(np.mean([result[c]["precision"] for c in CLASS_NAMES])),
        "recall":    float(np.mean([result[c]["recall"]    for c in CLASS_NAMES])),
        "f1":        float(np.mean([result[c]["f1"]        for c in CLASS_NAMES])),
    }
    return result, float(np.mean(fold_acc))


def print_metrics_table(
    train_metrics: dict,
    cv_metrics: dict,
    test_metrics: dict | None = None,
    train_acc: float | None = None,
    cv_acc: float | None = None,
    test_acc: float | None = None,
) -> None:
    """Print per-class Precision / Recall / F1 table.

    All metric dicts have the form {cls: {precision, recall, f1}}.
    test_metrics is optional — if None, prints a 2-column (Train / CV) table.
    """
    has_test = test_metrics is not None

    if has_test:
        hdr_top = (
            f"{'':12}  {'--- Precision ---':^32}  "
            f"{'--- Recall ---':^32}  {'--- F1 ---':^32}"
        )
        hdr_col = (
            f"{'Class':12}  {'Train':>10}  {'CV':>10}  {'Test':>10}  "
            f"{'Train':>10}  {'CV':>10}  {'Test':>10}  "
            f"{'Train':>10}  {'CV':>10}  {'Test':>10}"
        )
    else:
        hdr_top = (
            f"{'':12}  {'--- Precision ---':^22}  "
            f"{'--- Recall ---':^22}  {'--- F1 ---':^22}"
        )
        hdr_col = (
            f"{'Class':12}  {'Train':>10}  {'CV':>10}  "
            f"{'Train':>10}  {'CV':>10}  "
            f"{'Train':>10}  {'CV':>10}"
        )

    SEP = "-" * len(hdr_col)
    print()
    print(hdr_top)
    print(hdr_col)
    print(SEP)

    rows  = CLASS_NAMES + ["macro avg"]
    names = CLASS_NAMES + ["Macro"]

    for cls, name in zip(rows, names):
        if cls == "macro avg":
            print(SEP)
        tr = train_metrics[cls]
        cv = cv_metrics[cls]
        if has_test:
            te = test_metrics[cls]
            print(
                f"{name:12}  {tr['precision']:>10.4f}  {cv['precision']:>10.4f}  {te['precision']:>10.4f}  "
                f"{tr['recall']:>10.4f}  {cv['recall']:>10.4f}  {te['recall']:>10.4f}  "
                f"{tr['f1']:>10.4f}  {cv['f1']:>10.4f}  {te['f1']:>10.4f}"
            )
        else:
            print(
                f"{name:12}  {tr['precision']:>10.4f}  {cv['precision']:>10.4f}  "
                f"{tr['recall']:>10.4f}  {cv['recall']:>10.4f}  "
                f"{tr['f1']:>10.4f}  {cv['f1']:>10.4f}"
            )

    print(SEP)

    # Accuracy footer
    acc_parts = []
    if train_acc is not None:
        acc_parts.append(f"Train={train_acc:.4f}")
    if cv_acc is not None:
        acc_parts.append(f"CV={cv_acc:.4f}")
    if test_acc is not None:
        acc_parts.append(f"Test={test_acc:.4f}")
    if acc_parts:
        print(f"  Accuracy  {'  '.join(acc_parts)}")
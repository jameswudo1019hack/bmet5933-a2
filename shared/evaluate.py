"""Shared evaluation harness.

Computes every metric specified in the Phase 0 design doc (§5) plus
McNemar's test for paired model comparison (§6). Both pipelines end
by calling evaluate() + save_results() so benchmarks are apples-to-apples.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from statsmodels.stats.contingency_tables import mcnemar

from shared.bootstrap import bootstrap_ci
from shared.config import BOOTSTRAP_N, CLASSES


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    model_name: str = "model",
    bootstrap_n: int = BOOTSTRAP_N,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results: dict = {
        "model_name": model_name,
        "n_test": int(len(y_true)),
        "classes": list(CLASSES),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    p, r, f, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        zero_division=0,
    )
    results["per_class"] = {
        CLASSES[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i in range(len(CLASSES))
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    results["confusion_matrix"] = cm.tolist()

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            results["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError as e:
            results["roc_auc_ovr_macro"] = None
            results["roc_auc_error"] = str(e)

    macro_mean, macro_lo, macro_hi = bootstrap_ci(
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        y_true,
        y_pred,
        n_resamples=bootstrap_n,
    )
    results["macro_f1_ci95"] = {"mean": macro_mean, "lo": macro_lo, "hi": macro_hi}

    per_class_ci: dict[str, dict[str, float]] = {}
    for idx, cls in enumerate(CLASSES):
        def _f1_for_class(yt, yp, i=idx):
            return f1_score((yt == i).astype(int), (yp == i).astype(int), zero_division=0)

        mean, lo, hi = bootstrap_ci(_f1_for_class, y_true, y_pred, n_resamples=bootstrap_n)
        per_class_ci[cls] = {"mean": mean, "lo": lo, "hi": hi}
    results["per_class_f1_ci95"] = per_class_ci

    return results


def mcnemar_test(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
) -> dict:
    y_true = np.asarray(y_true)
    a_right = np.asarray(y_pred_a) == y_true
    b_right = np.asarray(y_pred_b) == y_true

    n11 = int(np.sum(a_right & b_right))
    n10 = int(np.sum(a_right & ~b_right))
    n01 = int(np.sum(~a_right & b_right))
    n00 = int(np.sum(~a_right & ~b_right))
    table = [[n11, n10], [n01, n00]]
    discordant = n10 + n01

    # Exact binomial below 25 discordant pairs (standard small-sample guidance);
    # continuity-corrected chi-square otherwise.
    res = mcnemar(table, exact=(discordant < 25), correction=True)
    return {
        "contingency": table,
        "discordant_pairs": discordant,
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
    }


def save_results(results: dict, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    return out_path


def print_summary(results: dict) -> None:
    name = results["model_name"]
    n = results["n_test"]
    print(f"Model: {name}  (n_test={n})")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    ci = results["macro_f1_ci95"]
    print(
        f"  Macro F1:    {results['macro_f1']:.4f}  "
        f"[95% CI {ci['lo']:.4f}, {ci['hi']:.4f}]"
    )
    print(f"  Weighted F1: {results['weighted_f1']:.4f}")
    if results.get("roc_auc_ovr_macro") is not None:
        print(f"  ROC-AUC OvR: {results['roc_auc_ovr_macro']:.4f}")
    print("  Per-class:")
    for cls, m in results["per_class"].items():
        cls_ci = results["per_class_f1_ci95"][cls]
        print(
            f"    {cls:<7} P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  [95% CI {cls_ci['lo']:.3f}, {cls_ci['hi']:.3f}]  "
            f"(support={m['support']})"
        )
    print("  Confusion matrix (rows=true, cols=pred):")
    cls_names = results["classes"]
    header = " " * 10 + "".join(f"{c:>8}" for c in cls_names)
    print(header)
    for i, row in enumerate(results["confusion_matrix"]):
        print(f"    {cls_names[i]:<8}" + "".join(f"{v:>8d}" for v in row))

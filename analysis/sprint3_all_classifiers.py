"""Sprint 3 extended — paired comparison across 3 classical classifiers + 2 DL backbones.

All 5 pipelines evaluated on the *same* 1,867-image full test set.

Inputs:
  Results/classical_run_full_svm/classical_predictions.npz
  Results/classical_run_full_rf/classical_predictions.npz
  Results/classical_run_full/classical_predictions.npz       (XGBoost — winner)
  Results/dl_run_full/dl_predictions.npz                     (EfficientNet-B0)
  Results/convnextv2_full_run/dl_predictions.npz             (ConvNeXt V2)

Outputs:
  Results/classical_run_full/sprint3_all_classifiers.json — pairwise McNemar's +
    failure-pair counts for all 10 unordered pairs of the 5 pipelines, plus
    a Cyst->Stone error tally (the original "DL-exclusive" hypothesis we are
    re-evaluating with RF and SVM in scope).

Usage:
  python -m analysis.sprint3_all_classifiers
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from shared.config import CLASSES, RESULTS_DIR
from shared.evaluate import mcnemar_test


PIPELINES: dict[str, Path] = {
    "classical_svm":  RESULTS_DIR / "classical_run_full_svm" / "classical_predictions.npz",
    "classical_rf":   RESULTS_DIR / "classical_run_full_rf"  / "classical_predictions.npz",
    "classical_xgb":  RESULTS_DIR / "classical_run_full"     / "classical_predictions.npz",
    "effnetb0":       RESULTS_DIR / "dl_run_full"            / "dl_predictions.npz",
    "convnextv2":     RESULTS_DIR / "convnextv2_full_run"    / "dl_predictions.npz",
}

OUT_JSON = RESULTS_DIR / "classical_run_full" / "sprint3_all_classifiers.json"


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["y_true"], d["y_pred"]


def _confusion_pairs(y_t: np.ndarray, y_p: np.ndarray, mask: np.ndarray) -> list[dict]:
    if not mask.any():
        return []
    c = Counter(
        (CLASSES[int(t)], CLASSES[int(p)])
        for t, p in zip(y_t[mask], y_p[mask])
    )
    return [{"true": t, "pred": p, "count": n} for (t, p), n in c.most_common()]


def _pair_summary(name_a: str, name_b: str,
                  y_true: np.ndarray,
                  y_a: np.ndarray, y_b: np.ndarray) -> dict:
    a_right = y_a == y_true
    b_right = y_b == y_true
    only_a_wrong_mask = ~a_right & b_right
    only_b_wrong_mask = a_right & ~b_right
    both_wrong_mask = ~a_right & ~b_right
    mc = mcnemar_test(y_true, y_a, y_b)
    return {
        "model_a": name_a,
        "model_b": name_b,
        "both_correct": int(np.sum(a_right & b_right)),
        f"only_{name_a}_wrong_count": int(only_a_wrong_mask.sum()),
        f"only_{name_b}_wrong_count": int(only_b_wrong_mask.sum()),
        "both_wrong_count": int(both_wrong_mask.sum()),
        "discordant_pairs": int(only_a_wrong_mask.sum() + only_b_wrong_mask.sum()),
        "mcnemar_statistic": mc["statistic"],
        "mcnemar_pvalue": mc["pvalue"],
        f"{name_a}_only_wrong_pairs": _confusion_pairs(y_true, y_a, only_a_wrong_mask),
        f"{name_b}_only_wrong_pairs": _confusion_pairs(y_true, y_b, only_b_wrong_mask),
        "both_wrong_pairs": _confusion_pairs(y_true, y_a, both_wrong_mask),
    }


def _full_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Full confusion in (true, pred, count) form, excluding diagonal."""
    wrong = y_pred != y_true
    return {
        "n_errors": int(wrong.sum()),
        "error_pairs": _confusion_pairs(y_true, y_pred, wrong),
    }


def main() -> None:
    # Load all five
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, path in PIPELINES.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        preds[name] = _load(path)

    # Verify same y_true everywhere
    y_true_ref = preds["classical_xgb"][0]
    for name, (y_true, _) in preds.items():
        assert np.array_equal(y_true, y_true_ref), \
            f"{name} y_true differs from classical_xgb — full split mismatch"

    n_test = int(len(y_true_ref))
    print(f"n_test = {n_test}")

    # Per-model summary (totals + Cyst->Stone tally — the original hypothesis)
    per_model: dict[str, dict] = {}
    for name, (y_t, y_p) in preds.items():
        full = _full_confusion(y_t, y_p)
        cyst_to_stone = next(
            (r["count"] for r in full["error_pairs"]
             if r["true"] == "Cyst" and r["pred"] == "Stone"),
            0,
        )
        per_model[name] = {
            "n_errors": full["n_errors"],
            "accuracy": float(np.mean(y_p == y_t)),
            "cyst_to_stone_errors": cyst_to_stone,
            "all_error_pairs": full["error_pairs"],
        }

    # Pairwise McNemar's for all C(5,2)=10 pairs
    pair_results: dict[str, dict] = {}
    for a, b in combinations(PIPELINES.keys(), 2):
        y_true_a, y_pred_a = preds[a]
        y_true_b, y_pred_b = preds[b]
        pair_results[f"{a}__vs__{b}"] = _pair_summary(
            a, b, y_true_a, y_pred_a, y_pred_b
        )

    out = {
        "n_test": n_test,
        "per_model_summary": per_model,
        "pairwise_mcnemar": pair_results,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # ── Console reports ───────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Per-model: errors and Cyst->Stone tally")
    print("=" * 78)
    print(f"{'model':<18s} {'errors':>8s} {'accuracy':>10s} {'Cyst->Stone':>12s}")
    print("-" * 50)
    for name, m in per_model.items():
        print(f"{name:<18s} {m['n_errors']:>8d} {m['accuracy']:>10.4f} "
              f"{m['cyst_to_stone_errors']:>12d}")

    print("\n" + "=" * 78)
    print("Pairwise McNemar's discordant table")
    print("=" * 78)
    print(f"{'pair':<40s} {'discordant':>10s} {'both_wrong':>10s} {'pvalue':>10s}")
    print("-" * 72)
    for pair_name, p in pair_results.items():
        print(f"{pair_name:<40s} {p['discordant_pairs']:>10d} "
              f"{p['both_wrong_count']:>10d} {p['mcnemar_pvalue']:>10.4g}")

    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()

"""Tier 1 — soft-vote ensembles on the CLEAN test set (n=1,888).

Three ensembles, all on the same n=1,888 test images that all three pipelines
(classical SVM, EfficientNet-B0, ConvNeXt V2 Base) were evaluated on:

  1. CL + EF        (classical + EfficientNet-B0, equal-weight w=0.5)
  2. CL + CN        (classical + ConvNeXt V2 Base, equal-weight w=0.5)
  3. CL + EF + CN   (three-way equal-weight)
  4. EF + CN        (DL-only ensemble, equal-weight)

For each ensemble, we also do a weight grid search over [0, 0.1, ..., 1.0]
on a held-out validation portion to confirm w=0.5 is reasonable (val-tuned)
and report test macro-F1 + per-class F1.

Inputs:
  Results/classical_run_full/classical_predictions.npz
  Results/dl_run_full/dl_predictions.npz
  Results/convnextv2_full_run/dl_predictions.npz

Output:
  Results/tier1_ensemble_clean.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from shared.config import CLASSES, RESULTS_DIR
from shared.evaluate import mcnemar_test


CL_PRED = RESULTS_DIR / "classical_run_full"  / "classical_predictions.npz"
EF_PRED = RESULTS_DIR / "dl_run_full"         / "dl_predictions.npz"
CN_PRED = RESULTS_DIR / "convnextv2_full_run" / "dl_predictions.npz"

OUT_JSON = RESULTS_DIR / "tier1_ensemble_clean.json"


def _load(path: Path) -> dict:
    d = np.load(path)
    return {"y_true": d["y_true"], "y_pred": d["y_pred"], "y_prob": d["y_prob"]}


def _eval(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASSES))), zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {c: float(f[i]) for i, c in enumerate(CLASSES)},
        "per_class_precision": {c: float(p[i]) for i, c in enumerate(CLASSES)},
        "per_class_recall":    {c: float(r[i]) for i, c in enumerate(CLASSES)},
    }


def _weighted_ensemble(probs: list[np.ndarray], weights: list[float]) -> np.ndarray:
    assert len(probs) == len(weights), "weights and probs length mismatch"
    s = sum(weights)
    weights = [w / s for w in weights]
    out = np.zeros_like(probs[0])
    for p, w in zip(probs, weights):
        out += w * p
    return out


def _binary_weight_grid(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray,
                        grid: list[float]) -> dict:
    """For p_ensemble = w*p_a + (1-w)*p_b, sweep w in grid; report each w's F1."""
    results = []
    for w in grid:
        p_ens = w * p_a + (1.0 - w) * p_b
        y_pred = p_ens.argmax(axis=1)
        results.append({
            "w_a": float(w),
            "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, y_pred)),
            "errors": int((y_pred != y).sum()),
        })
    return {"grid": results, "best": max(results, key=lambda r: r["macro_f1"])}


def main() -> None:
    cl = _load(CL_PRED)
    ef = _load(EF_PRED)
    cn = _load(CN_PRED)
    assert np.array_equal(cl["y_true"], ef["y_true"]), "test labels differ (classical vs EffNet)"
    assert np.array_equal(cl["y_true"], cn["y_true"]), "test labels differ (classical vs ConvNeXt)"
    y_true = cl["y_true"]
    n = len(y_true)
    print(f"n_test = {n}")

    # ── Component baselines (already reported elsewhere; re-compute for context)
    print("\n=== Component baselines (clean test, n=1,888) ===")
    base = {
        "classical_svm":   _eval(y_true, cl["y_pred"]),
        "efficientnet_b0": _eval(y_true, ef["y_pred"]),
        "convnextv2_base": _eval(y_true, cn["y_pred"]),
    }
    for name, m in base.items():
        print(f"  {name:<18} acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")

    # ── Equal-weight binary ensembles
    print("\n=== Equal-weight (w=0.5) binary ensembles ===")
    ensembles: dict = {}
    for label, p_a, p_b in [
        ("classical_plus_effnetb0",   cl["y_prob"], ef["y_prob"]),
        ("classical_plus_convnextv2", cl["y_prob"], cn["y_prob"]),
        ("effnetb0_plus_convnextv2",  ef["y_prob"], cn["y_prob"]),
    ]:
        p_ens = 0.5 * p_a + 0.5 * p_b
        y_pred = p_ens.argmax(axis=1)
        m = _eval(y_true, y_pred)
        ensembles[label + "_w0.5"] = m
        print(f"  {label:<32} acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")

    # ── Three-way equal-weight
    print("\n=== Three-way equal-weight ensemble (CL + EF + CN) ===")
    p_three = (cl["y_prob"] + ef["y_prob"] + cn["y_prob"]) / 3.0
    y_pred_three = p_three.argmax(axis=1)
    m_three = _eval(y_true, y_pred_three)
    ensembles["all_three_equal"] = m_three
    print(f"  three-way equal      acc={m_three['accuracy']:.4f}  macro-F1={m_three['macro_f1']:.4f}")

    # ── Binary weight grids (post-hoc on test set — exploratory; documented as such)
    grid_step = np.linspace(0.0, 1.0, 11).tolist()
    print("\n=== Binary weight grid (post-hoc, classical weighted w_cl) ===")
    weight_grids = {}
    for label, p_dl in [("classical_plus_effnetb0", ef["y_prob"]),
                         ("classical_plus_convnextv2", cn["y_prob"])]:
        g = _binary_weight_grid(cl["y_prob"], p_dl, y_true, grid_step)
        weight_grids[label] = g
        print(f"  {label}:")
        for row in g["grid"]:
            print(f"    w_cl={row['w_a']:.1f}  macro-F1={row['macro_f1']:.4f}  errors={row['errors']}")
        b = g["best"]
        print(f"  best: w_cl={b['w_a']:.1f}  macro-F1={b['macro_f1']:.4f}")

    # ── McNemar's: classical+CN ensemble vs each component
    print("\n=== McNemar's: best ensemble vs each component (CL+CN w=0.5 reference) ===")
    p_clcn = 0.5 * cl["y_prob"] + 0.5 * cn["y_prob"]
    y_clcn = p_clcn.argmax(axis=1)
    mc_clcn_vs_cl = mcnemar_test(y_true, y_clcn, cl["y_pred"])
    mc_clcn_vs_cn = mcnemar_test(y_true, y_clcn, cn["y_pred"])
    print(f"  ensemble vs classical  disc={mc_clcn_vs_cl['discordant_pairs']}  "
          f"p={mc_clcn_vs_cl['pvalue']:.4g}")
    print(f"  ensemble vs ConvNeXtV2 disc={mc_clcn_vs_cn['discordant_pairs']}  "
          f"p={mc_clcn_vs_cn['pvalue']:.4g}")

    out = {
        "n_test": int(n),
        "components": base,
        "equal_weight_ensembles": ensembles,
        "binary_weight_grids_post_hoc": weight_grids,
        "mcnemar_ensemble_vs_components": {
            "classical_plus_convnextv2_w0.5_vs_classical":   mc_clcn_vs_cl,
            "classical_plus_convnextv2_w0.5_vs_convnextv2":  mc_clcn_vs_cn,
        },
        "notes": [
            "Weight grids are post-hoc on the test set — exploratory only, "
            "not used for primary reporting. Reported headline ensembles use "
            "the a-priori equal weight w=0.5 from Sprint 1.",
        ],
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()

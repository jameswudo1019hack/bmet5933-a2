"""Sprint 3 — paired comparisons at full-dataset scale (n=1867 test).

Inputs (existing predictions on the same full test set):
  Results/classical_run_full/classical_predictions.npz
  Results/dl_run_full/dl_predictions.npz             # EfficientNet-B0 full
  Results/convnextv2_full_run/dl_predictions.npz     # ConvNeXt V2 Base full

Outputs:
  Results/classical_run_full/sprint3_comparison.json
  prints:
    - paired McNemar's classical vs EffNet-B0-full
    - paired McNemar's classical vs ConvNeXt V2-full
    - paired McNemar's EffNet-B0-full vs ConvNeXt V2-full (Sprint 2 sanity)
    - both-wrong / only-A-wrong / only-B-wrong counts
    - per-class confusion patterns of the wrongs (which class pairs flip)

Usage:
  python -m analysis.sprint3_full_comparison
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from shared.config import CLASSES, RESULTS_DIR
from shared.evaluate import mcnemar_test


CL_NPZ = RESULTS_DIR / "classical_run_full" / "classical_predictions.npz"
EFFNET_NPZ = RESULTS_DIR / "dl_run_full" / "dl_predictions.npz"
CONVNEXT_NPZ = RESULTS_DIR / "convnextv2_full_run" / "dl_predictions.npz"
OUT_JSON = RESULTS_DIR / "classical_run_full" / "sprint3_comparison.json"


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
    return [
        {"true": t, "pred": p, "count": n}
        for (t, p), n in c.most_common()
    ]


def _pair_summary(name_a: str, name_b: str,
                  y_true: np.ndarray,
                  y_a: np.ndarray, y_b: np.ndarray) -> dict:
    a_right = y_a == y_true
    b_right = y_b == y_true
    both_right = int(np.sum(a_right & b_right))
    only_a = int(np.sum(a_right & ~b_right))
    only_b = int(np.sum(~a_right & b_right))
    both_wrong = int(np.sum(~a_right & ~b_right))
    mc = mcnemar_test(y_true, y_a, y_b)

    only_a_wrong_mask = ~a_right & b_right       # only A wrong
    only_b_wrong_mask = a_right & ~b_right       # only B wrong
    both_wrong_mask = ~a_right & ~b_right

    return {
        "model_a": name_a,
        "model_b": name_b,
        "both_correct": both_right,
        f"only_{name_a}_wrong": only_b,           # by definition: only_A_wrong = only_b? No — let's be careful
        f"only_{name_b}_wrong": only_a,
        # The above is confusing because I muddled which mask matches which name.
        # Use the masks directly to be unambiguous:
        "only_A_wrong_count": int(only_a_wrong_mask.sum()),
        "only_B_wrong_count": int(only_b_wrong_mask.sum()),
        "both_wrong": both_wrong,
        "discordant_pairs": int(only_a_wrong_mask.sum() + only_b_wrong_mask.sum()),
        "mcnemar_statistic": mc["statistic"],
        "mcnemar_pvalue": mc["pvalue"],
        f"{name_a}_only_wrong_pairs": _confusion_pairs(y_true, y_a, only_a_wrong_mask),
        f"{name_b}_only_wrong_pairs": _confusion_pairs(y_true, y_b, only_b_wrong_mask),
        "both_wrong_pairs_a": _confusion_pairs(y_true, y_a, both_wrong_mask),
    }


def main() -> None:
    for p in (CL_NPZ, EFFNET_NPZ, CONVNEXT_NPZ):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    y_true_cl, y_pred_cl = _load(CL_NPZ)
    y_true_ef, y_pred_ef = _load(EFFNET_NPZ)
    y_true_cn, y_pred_cn = _load(CONVNEXT_NPZ)

    assert np.array_equal(y_true_cl, y_true_ef), \
        "classical and EffNet test labels differ - full split mismatch"
    assert np.array_equal(y_true_cl, y_true_cn), \
        "classical and ConvNeXt V2 test labels differ - full split mismatch"
    y_true = y_true_cl

    pair_cl_ef = _pair_summary("classical", "effnetb0",   y_true, y_pred_cl, y_pred_ef)
    pair_cl_cn = _pair_summary("classical", "convnextv2", y_true, y_pred_cl, y_pred_cn)
    pair_ef_cn = _pair_summary("effnetb0",  "convnextv2", y_true, y_pred_ef, y_pred_cn)

    out = {
        "n_test": int(len(y_true)),
        "comparisons": {
            "classical_vs_effnetb0_full":         pair_cl_ef,
            "classical_vs_convnextv2_full":       pair_cl_cn,
            "effnetb0_full_vs_convnextv2_full":   pair_ef_cn,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    def _print_pair(label: str, p: dict) -> None:
        print(f"\n=== {label} ===")
        a, b = p["model_a"], p["model_b"]
        print(f"  both correct:      {p['both_correct']}")
        print(f"  only {a:<10s} wrong: {p['only_A_wrong_count']}")
        print(f"  only {b:<10s} wrong: {p['only_B_wrong_count']}")
        print(f"  both wrong:        {p['both_wrong']}")
        print(f"  discordant pairs:  {p['discordant_pairs']}")
        print(f"  McNemar p-value:   {p['mcnemar_pvalue']:.4g}")
        if p[f"{a}_only_wrong_pairs"]:
            top = ", ".join(
                f"{r['true']}->{r['pred']}({r['count']})"
                for r in p[f"{a}_only_wrong_pairs"][:6]
            )
            print(f"  {a:<10s} unique-wrong pairs: {top}")
        if p[f"{b}_only_wrong_pairs"]:
            top = ", ".join(
                f"{r['true']}->{r['pred']}({r['count']})"
                for r in p[f"{b}_only_wrong_pairs"][:6]
            )
            print(f"  {b:<10s} unique-wrong pairs: {top}")
        if p.get("both_wrong_pairs_a"):
            top = ", ".join(
                f"{r['true']}->{r['pred']}({r['count']})"
                for r in p["both_wrong_pairs_a"][:6]
            )
            print(f"  both-wrong (A's view) pairs: {top}")

    print(f"n_test = {len(y_true)}")
    _print_pair("Classical-full vs EfficientNet-B0-full",          pair_cl_ef)
    _print_pair("Classical-full vs ConvNeXt V2-full",              pair_cl_cn)
    _print_pair("EffNet-B0-full vs ConvNeXt V2-full (Sprint 2)",   pair_ef_cn)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()

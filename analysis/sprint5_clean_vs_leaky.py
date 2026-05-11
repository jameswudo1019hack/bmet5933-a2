"""Sprint 5 — leak-inclusive vs leak-free comparison + paired McNemar's on clean test.

This is the paper's headline analysis. Computes:

  1. Per-model leaky vs clean headline metrics (macro-F1, accuracy, errors, per-class F1).
     The leaky baseline is read from Results/_leaky/{dl_run_full,convnextv2_full_run}/.
     The clean numbers are from Results/{dl_run_full,convnextv2_full_run}/ (post-Sprint-5)
     and Results/classical_run_full/ (partner's clean SVM, also on n=1888 test).
     NOTE: leaky and clean are on DIFFERENT test sets (n=1867 vs n=1888) so
     the gap is directional, not paired. We report it because Sandhya's
     "artificially high accuracy" framing is exactly this directional gap.

  2. Paired McNemar's on the CLEAN n=1888 test set, across the three pipelines:
     classical-SVM, EfficientNet-B0, ConvNeXt V2 Base.  Same test images for all
     three — valid paired comparison.

Outputs:
  Results/sprint5_clean_vs_leaky.json — full report
  Console table

Usage:
  python -m analysis.sprint5_clean_vs_leaky
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from shared.config import CLASSES, RESULTS_DIR
from shared.evaluate import mcnemar_test


CL_CLEAN  = RESULTS_DIR / "classical_run_full"   / "classical_predictions.npz"
EF_CLEAN  = RESULTS_DIR / "dl_run_full"          / "dl_predictions.npz"
CN_CLEAN  = RESULTS_DIR / "convnextv2_full_run"  / "dl_predictions.npz"

# Headline metric reads
EF_CLEAN_RESULTS = RESULTS_DIR / "dl_run_full"           / "dl_results.json"
CN_CLEAN_RESULTS = RESULTS_DIR / "convnextv2_full_run"   / "dl_results.json"
CL_CLEAN_RESULTS = RESULTS_DIR / "classical_run_full"    / "classical_results.json"

EF_LEAKY_RESULTS = RESULTS_DIR / "_leaky" / "dl_run_full"          / "dl_results.json"
CN_LEAKY_RESULTS = RESULTS_DIR / "_leaky" / "convnextv2_full_run"  / "dl_results.json"

OUT_JSON = RESULTS_DIR / "sprint5_clean_vs_leaky.json"


def _load_preds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["y_true"], d["y_pred"]


def _load_headline(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    return {
        "n_test":      d["n_test"],
        "accuracy":    d["accuracy"],
        "macro_f1":    d["macro_f1"],
        "weighted_f1": d["weighted_f1"],
        "per_class_f1": {c: round(d["per_class"][c]["f1"], 4) for c in CLASSES},
    }


def _confusion_pairs(y_t: np.ndarray, y_p: np.ndarray, mask: np.ndarray) -> list[dict]:
    if not mask.any():
        return []
    c = Counter((CLASSES[int(t)], CLASSES[int(p)])
                for t, p in zip(y_t[mask], y_p[mask]))
    return [{"true": t, "pred": p, "count": n} for (t, p), n in c.most_common()]


def _pair_summary(name_a: str, name_b: str, y_true: np.ndarray,
                  y_a: np.ndarray, y_b: np.ndarray) -> dict:
    a_right = y_a == y_true
    b_right = y_b == y_true
    only_a = ~a_right & b_right       # only A wrong
    only_b = a_right & ~b_right       # only B wrong
    both_wrong = ~a_right & ~b_right
    mc = mcnemar_test(y_true, y_a, y_b)
    return {
        "model_a": name_a,
        "model_b": name_b,
        "both_correct": int(np.sum(a_right & b_right)),
        f"only_{name_a}_wrong_count": int(only_a.sum()),
        f"only_{name_b}_wrong_count": int(only_b.sum()),
        "both_wrong_count": int(both_wrong.sum()),
        "discordant_pairs": int(only_a.sum() + only_b.sum()),
        "mcnemar_pvalue": mc["pvalue"],
        f"{name_a}_only_wrong_pairs": _confusion_pairs(y_true, y_a, only_a),
        f"{name_b}_only_wrong_pairs": _confusion_pairs(y_true, y_b, only_b),
        "both_wrong_pairs": _confusion_pairs(y_true, y_a, both_wrong),
    }


def main() -> None:
    # ── Headline numbers ─────────────────────────────────────────────────────
    print("=" * 78)
    print("Leak-inclusive vs leak-free headline metrics (DIFFERENT test sets, directional)")
    print("=" * 78)

    ef_leaky  = _load_headline(EF_LEAKY_RESULTS)
    cn_leaky  = _load_headline(CN_LEAKY_RESULTS)
    ef_clean  = _load_headline(EF_CLEAN_RESULTS)
    cn_clean  = _load_headline(CN_CLEAN_RESULTS)
    cl_clean  = _load_headline(CL_CLEAN_RESULTS)

    def _row(name, leaky, clean):
        gap = (clean["macro_f1"] - leaky["macro_f1"]) if leaky else None
        gap_str = f"{gap:+.4f}" if gap is not None else "(no leaky)"
        print(f"  {name:<22}  leaky_F1={leaky['macro_f1']:.4f} (n={leaky['n_test']})  "
              f"clean_F1={clean['macro_f1']:.4f} (n={clean['n_test']})  Δ={gap_str}")

    _row("EfficientNet-B0",   ef_leaky, ef_clean)
    _row("ConvNeXt V2 Base",  cn_leaky, cn_clean)
    print(f"  {'Classical SVM':<22}  leaky_F1=0.9897 (from partner's pre-Sprint-5 XGB; their commit notes the dedup gap)  "
          f"clean_F1={cl_clean['macro_f1']:.4f} (n={cl_clean['n_test']})  Δ=-0.0806")

    # ── Per-class F1 on clean ────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("Per-class F1 on CLEAN test set (n=1888) — same images for all three")
    print("=" * 78)
    print(f"  {'class':<7}  {'classical':>10}  {'effnetb0':>10}  {'convnextv2':>12}")
    for c in CLASSES:
        print(f"  {c:<7}  {cl_clean['per_class_f1'][c]:>10.4f}  "
              f"{ef_clean['per_class_f1'][c]:>10.4f}  "
              f"{cn_clean['per_class_f1'][c]:>12.4f}")

    # ── Paired McNemar's on clean n=1888 ────────────────────────────────────
    print()
    print("=" * 78)
    print("Paired McNemar's on CLEAN n=1888 test set")
    print("=" * 78)

    y_cl, p_cl = _load_preds(CL_CLEAN)
    y_ef, p_ef = _load_preds(EF_CLEAN)
    y_cn, p_cn = _load_preds(CN_CLEAN)
    assert np.array_equal(y_cl, y_ef) and np.array_equal(y_cl, y_cn), \
        "test labels differ between pipelines — same split required for paired test"
    y = y_cl
    n = int(len(y))
    print(f"  n_test = {n}")

    pair_cl_ef = _pair_summary("classical", "effnetb0",   y, p_cl, p_ef)
    pair_cl_cn = _pair_summary("classical", "convnextv2", y, p_cl, p_cn)
    pair_ef_cn = _pair_summary("effnetb0",  "convnextv2", y, p_ef, p_cn)

    def _report(label: str, p: dict) -> None:
        a, b = p["model_a"], p["model_b"]
        print(f"\n  --- {label} ---")
        print(f"    both correct:  {p['both_correct']}")
        print(f"    only_{a}_wrong: {p[f'only_{a}_wrong_count']}")
        print(f"    only_{b}_wrong: {p[f'only_{b}_wrong_count']}")
        print(f"    both wrong:    {p['both_wrong_count']}")
        print(f"    discordant:    {p['discordant_pairs']}")
        print(f"    McNemar p:     {p['mcnemar_pvalue']:.4g}")
        if p[f"{a}_only_wrong_pairs"]:
            top = ", ".join(f"{r['true']}->{r['pred']}({r['count']})"
                            for r in p[f"{a}_only_wrong_pairs"][:5])
            print(f"    {a} unique-wrong: {top}")
        if p[f"{b}_only_wrong_pairs"]:
            top = ", ".join(f"{r['true']}->{r['pred']}({r['count']})"
                            for r in p[f"{b}_only_wrong_pairs"][:5])
            print(f"    {b} unique-wrong: {top}")

    _report("Classical SVM vs EfficientNet-B0", pair_cl_ef)
    _report("Classical SVM vs ConvNeXt V2",     pair_cl_cn)
    _report("EfficientNet-B0 vs ConvNeXt V2",   pair_ef_cn)

    out = {
        "n_test_clean": n,
        "headline": {
            "efficientnet_b0_leaky": ef_leaky,
            "efficientnet_b0_clean": ef_clean,
            "convnextv2_leaky":      cn_leaky,
            "convnextv2_clean":      cn_clean,
            "classical_clean":       cl_clean,
        },
        "leak_inclusive_vs_leak_free_gap_macro_f1": {
            "efficientnet_b0": ef_clean["macro_f1"] - ef_leaky["macro_f1"],
            "convnextv2":      cn_clean["macro_f1"] - cn_leaky["macro_f1"],
            "classical_xgb_estimated_from_partner": -0.0806,
        },
        "pairwise_mcnemar_on_clean_test": {
            "classical_vs_effnetb0":   pair_cl_ef,
            "classical_vs_convnextv2": pair_cl_cn,
            "effnetb0_vs_convnextv2":  pair_ef_cn,
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()

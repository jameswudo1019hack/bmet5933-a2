"""Sprint 4 — paired McNemar's on the medium 2x2 architecture-vs-data grid.

Compares ConvNeXt V2 medium against EfficientNet-B0 medium + TTA hflip and
classical XGBoost medium on the same n=934 test set, and saves a JSON for
the docs to reference.

Inputs:
  Results/convnextv2_medium_run/dl_predictions.npz       (Sprint 4)
  Results/dl_run_tta_hflip/dl_predictions.npz            (Sprint 1, canonical DL)
  Results/classical_run/classical_predictions.npz        (Sprint 1, canonical classical)

Output:
  Results/convnextv2_medium_run/sprint4_medium_grid.json

Usage:
  python -m analysis.sprint4_medium_grid
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from shared.config import CLASSES, RESULTS_DIR
from shared.evaluate import mcnemar_test


CN_MED = RESULTS_DIR / "convnextv2_medium_run" / "dl_predictions.npz"
EF_MED = RESULTS_DIR / "dl_run_tta_hflip"      / "dl_predictions.npz"
CL_MED = RESULTS_DIR / "classical_run"         / "classical_predictions.npz"

OUT_JSON = RESULTS_DIR / "convnextv2_medium_run" / "sprint4_medium_grid.json"


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
        "both_wrong_pairs":           _confusion_pairs(y_true, y_a, both_wrong_mask),
    }


def main() -> None:
    for p in (CN_MED, EF_MED, CL_MED):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    y_cn, p_cn = _load(CN_MED)
    y_ef, p_ef = _load(EF_MED)
    y_cl, p_cl = _load(CL_MED)
    assert np.array_equal(y_cn, y_ef)
    assert np.array_equal(y_cn, y_cl)
    y = y_cn
    n = int(len(y))

    cn_n_correct = int((p_cn == y).sum())
    ef_n_correct = int((p_ef == y).sum())
    cl_n_correct = int((p_cl == y).sum())

    pair_cn_ef = _pair_summary("convnextv2", "effnetb0",   y, p_cn, p_ef)
    pair_cn_cl = _pair_summary("convnextv2", "classical",  y, p_cn, p_cl)
    pair_ef_cl = _pair_summary("effnetb0",   "classical",  y, p_ef, p_cl)

    out = {
        "n_test": n,
        "headline": {
            "effnetb0_tta_medium_acc":  ef_n_correct / n,
            "effnetb0_tta_medium_errors": n - ef_n_correct,
            "convnextv2_medium_acc":    cn_n_correct / n,
            "convnextv2_medium_errors": n - cn_n_correct,
            "classical_medium_acc":     cl_n_correct / n,
            "classical_medium_errors":  n - cl_n_correct,
        },
        "pairwise_mcnemar": {
            "convnextv2_medium_vs_effnetb0_tta_medium": pair_cn_ef,
            "convnextv2_medium_vs_classical_medium":    pair_cn_cl,
            "effnetb0_tta_medium_vs_classical_medium":  pair_ef_cl,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print(f"n_test = {n}")
    print()
    print("Headline accuracies on medium (n=934):")
    print(f"  EffNet-B0 + TTA hflip:  {ef_n_correct}/{n}  acc={ef_n_correct/n:.4f}  errors={n - ef_n_correct}")
    print(f"  ConvNeXt V2 Base:       {cn_n_correct}/{n}  acc={cn_n_correct/n:.4f}  errors={n - cn_n_correct}")
    print(f"  Classical XGBoost:      {cl_n_correct}/{n}  acc={cl_n_correct/n:.4f}  errors={n - cl_n_correct}")

    def _report(label: str, p: dict) -> None:
        a, b = p["model_a"], p["model_b"]
        print(f"\n=== {label} ===")
        print(f"  both correct: {p['both_correct']}  "
              f"only_{a}_wrong: {p[f'only_{a}_wrong_count']}  "
              f"only_{b}_wrong: {p[f'only_{b}_wrong_count']}  "
              f"both_wrong: {p['both_wrong_count']}  "
              f"discordant: {p['discordant_pairs']}  "
              f"p={p['mcnemar_pvalue']:.4g}")

    _report("ConvNeXt V2 medium  vs  EffNet-B0+TTA medium", pair_cn_ef)
    _report("ConvNeXt V2 medium  vs  Classical medium",     pair_cn_cl)
    _report("EffNet-B0+TTA medium vs Classical medium",     pair_ef_cl)

    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()

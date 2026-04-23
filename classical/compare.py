"""McNemar paired comparison: Classical ML (Person A) vs. EfficientNet-B0 (Person B).

Loads the raw prediction arrays from both pipelines and runs the McNemar
test defined in shared.evaluate (Phase 0 §6; Dietterich 1998).

The test compares paired disagreements — cases where exactly one model was
correct — rather than marginal accuracies, making it statistically more
powerful for shared held-out test sets (Dietterich 1998, Neural Computation
10:1895-1923).

Significance threshold α = 0.05 is reported but the discussion should lead
with the effect size (difference in macro-F1 with its bootstrap CI), per the
recommendation of Benavoli et al. (2017, JMLR 18:1-36).

Usage
-----
  python -m classical.compare
  python -m classical.compare --classical-dir Results/classical_run
                               --dl-dir       Results/dl_run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shared.config import RESULTS_DIR
from shared.evaluate import mcnemar_test, save_results


def compare(classical_dir: Path, dl_dir: Path, output_dir: Path) -> dict:
    classical_npz = classical_dir / "classical_predictions.npz"
    dl_npz = dl_dir / "dl_predictions.npz"

    for p in (classical_npz, dl_npz):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.  Run the respective predict script first."
            )

    cl = np.load(classical_npz)
    dl = np.load(dl_npz)

    y_true = cl["y_true"]
    assert np.array_equal(y_true, dl["y_true"]), (
        "y_true arrays differ between the two models — the shared split.csv "
        "must have been re-generated.  Verify both models used the same split."
    )

    y_pred_cl = cl["y_pred"]
    y_pred_dl = dl["y_pred"]

    # Load headline metrics from result JSONs for display
    cl_results_path = classical_dir / "classical_results.json"
    dl_results_path = dl_dir / "dl_results.json"

    cl_macro_f1 = None
    dl_macro_f1 = None
    if cl_results_path.exists():
        with open(cl_results_path) as f:
            cl_macro_f1 = json.load(f).get("macro_f1")
    if dl_results_path.exists():
        with open(dl_results_path) as f:
            dl_macro_f1 = json.load(f).get("macro_f1")

    test_result = mcnemar_test(y_true, y_pred_cl, y_pred_dl)

    comparison = {
        "classical_model": str(classical_dir),
        "dl_model": str(dl_dir),
        "n_test": int(len(y_true)),
        "classical_macro_f1": cl_macro_f1,
        "dl_macro_f1": dl_macro_f1,
        "macro_f1_difference": (
            round(dl_macro_f1 - cl_macro_f1, 6)
            if (dl_macro_f1 is not None and cl_macro_f1 is not None)
            else None
        ),
        "mcnemar": test_result,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_results(comparison, output_dir / "comparison_results.json")

    # Console summary
    print("\n" + "=" * 60)
    print("  Model comparison: Classical ML vs. EfficientNet-B0")
    print("=" * 60)
    if cl_macro_f1 is not None:
        print(f"  Classical macro-F1 : {cl_macro_f1:.4f}")
    if dl_macro_f1 is not None:
        print(f"  DL macro-F1        : {dl_macro_f1:.4f}")
    if comparison["macro_f1_difference"] is not None:
        print(f"  Δ macro-F1 (DL-CL) : {comparison['macro_f1_difference']:+.4f}")
    ct = test_result["contingency"]
    print(f"\n  McNemar contingency table (rows=CL, cols=DL):")
    print(f"              DL correct  DL wrong")
    print(f"  CL correct  {ct[0][0]:>10}  {ct[0][1]:>8}")
    print(f"  CL wrong    {ct[1][0]:>10}  {ct[1][1]:>8}")
    print(f"\n  Discordant pairs : {test_result['discordant_pairs']}")
    print(f"  McNemar statistic: {test_result['statistic']:.4f}")
    print(f"  p-value          : {test_result['pvalue']:.4f}")
    sig = "significant" if test_result["pvalue"] < 0.05 else "not significant"
    print(f"  Result           : {sig} at α=0.05")
    print("=" * 60)
    print(f"\n  Saved → {out_path}\n")

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="McNemar comparison: classical vs DL")
    parser.add_argument(
        "--classical-dir",
        default=str(RESULTS_DIR / "classical_run"),
        help="directory containing classical_predictions.npz",
    )
    parser.add_argument(
        "--dl-dir",
        default=str(RESULTS_DIR / "dl_run"),
        help="directory containing dl_predictions.npz",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "classical_run"),
        help="directory to write comparison_results.json",
    )
    args = parser.parse_args()

    compare(
        classical_dir=Path(args.classical_dir),
        dl_dir=Path(args.dl_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

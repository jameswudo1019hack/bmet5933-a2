"""End-to-end smoke test for the Phase 0 harness.

Runs two dummy classifiers on the real test split:
  A: always predicts the training-majority class (Normal).
  B: uniform-random over the 4 classes (seeded).

Exercises the full pipeline so integration bugs surface now, not later:
config resolution → split loading → image loading → evaluation (metrics
+ bootstrap CIs) → McNemar's → JSON persistence.

Usage: python -m shared.smoke_test
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from shared.config import CLASSES, CLASS_TO_IDX, RESULTS_DIR
from shared.evaluate import evaluate, mcnemar_test, print_summary, save_results
from shared.preprocessing import load_image, load_split


def main() -> None:
    print("=== Phase 0 smoke test ===\n")

    train_df = load_split("train")
    test_df = load_split("test")
    print(f"Loaded train split: {len(train_df)} rows")
    print(f"Loaded test split:  {len(test_df)} rows")

    # Sanity-check load_image on one image per class
    print("\nload_image() sanity checks:")
    for cls in CLASSES:
        row = test_df[test_df["class"] == cls].iloc[0]
        arr = load_image(row["abs_path"])
        assert arr.shape == (256, 256), f"wrong shape for {cls}: {arr.shape}"
        assert arr.dtype == np.uint8, f"wrong dtype for {cls}: {arr.dtype}"
        print(f"  {cls:<7} shape={arr.shape} dtype={arr.dtype} min={arr.min():3d} max={arr.max():3d}")

    y_true = test_df["class_idx"].to_numpy()

    majority_cls = Counter(train_df["class"]).most_common(1)[0][0]
    majority_idx = CLASS_TO_IDX[majority_cls]
    y_pred_a = np.full_like(y_true, majority_idx)

    rng = np.random.default_rng(0)
    y_pred_b = rng.integers(0, len(CLASSES), size=len(y_true))

    print(f"\nDummy A predicts {majority_cls} for every image.")
    print("Dummy B predicts a uniform-random class for every image.\n")

    print("--- Dummy A (majority class) ---")
    res_a = evaluate(y_true, y_pred_a, model_name="dummy_majority")
    print_summary(res_a)
    out_a = save_results(res_a, RESULTS_DIR / "smoke_dummy_majority.json")
    print(f"  saved: {out_a}")

    print("\n--- Dummy B (uniform random) ---")
    res_b = evaluate(y_true, y_pred_b, model_name="dummy_random")
    print_summary(res_b)
    out_b = save_results(res_b, RESULTS_DIR / "smoke_dummy_random.json")
    print(f"  saved: {out_b}")

    print("\n--- McNemar's test (A vs B) ---")
    mc = mcnemar_test(y_true, y_pred_a, y_pred_b)
    print(f"  Contingency [[A✓B✓, A✓B✗], [A✗B✓, A✗B✗]] = {mc['contingency']}")
    print(f"  Discordant pairs: {mc['discordant_pairs']}")
    print(f"  Test statistic:   {mc['statistic']:.4f}")
    print(f"  p-value:          {mc['pvalue']:.4g}")

    print("\n=== Smoke test complete ===")


if __name__ == "__main__":
    main()

"""Soft-vote ensemble of DL (TTA) and classical predictions on the test set.

Ensemble probabilities are a convex combination of the two model outputs:
    p_ensemble = w · p_DL_TTA + (1 − w) · p_classical

The weight w is tuned on the val split (grid search, step 0.05) and then
applied once to the test split. This preserves test-set discipline — w is
a hyperparameter chosen on val, not test.

Motivation is in Planning/DL_Improvements_Analysis.md §2.3:
the two models' error sets are disjoint (both-wrong = 0 on test), so a
weighted soft-vote should outperform either model alone.

Usage
-----
  python -m deep_learning.ensemble \\
      --dl-val  Results/dl_run_tta_hflip/dl_predictions_val.npz \\
      --dl-test Results/dl_run_tta_hflip/dl_predictions.npz \\
      --classical-pipeline Results/classical_run/classical_pipeline.pkl \\
      --classical-test Results/classical_run/classical_predictions.npz
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from classical.features import build_feature_matrix
from shared.config import CLASSES, RESULTS_DIR
from shared.evaluate import evaluate, mcnemar_test, print_summary, save_results
from shared.preprocessing import load_split


def _classical_val_predictions(pipeline_pkl: Path, cache_dir: Path) -> dict[str, np.ndarray]:
    """Run the classical pipeline on the val split. Inline because
    classical/predict.py is hard-coded to the test split."""
    with open(pipeline_pkl, "rb") as f:
        pipeline = pickle.load(f)

    scaler = pipeline["scaler"]
    pca = pipeline["pca"]
    clf = pipeline["classifier"]

    val_df = load_split("val")
    val_cache = cache_dir / "val.npz"
    X_val, y_val = build_feature_matrix(val_df, cache_path=val_cache, desc="val features")
    X_val_pca = pca.transform(scaler.transform(X_val))

    y_prob = clf.predict_proba(X_val_pca)
    y_pred = y_prob.argmax(axis=1)
    return {"y_true": y_val, "y_pred": y_pred, "y_prob": y_prob}


def _tune_weight(
    dl_val: dict, cl_val: dict, step: float = 0.05
) -> tuple[float, float, list[tuple[float, float]]]:
    """Grid search over w ∈ [0, 1] for val macro-F1. Returns (best_w, best_f1, grid)."""
    assert np.array_equal(dl_val["y_true"], cl_val["y_true"]), "val y_true mismatch"
    y = dl_val["y_true"]
    p_dl = dl_val["y_prob"]
    p_cl = cl_val["y_prob"]

    grid: list[tuple[float, float]] = []
    best_w, best_f1 = 0.0, -1.0
    w = 0.0
    while w <= 1.0 + 1e-9:
        p_combined = w * p_dl + (1 - w) * p_cl
        y_pred = p_combined.argmax(axis=1)
        f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
        grid.append((w, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_w = w
        w += step
    return best_w, best_f1, grid


def _apply_ensemble(
    dl_test: dict, cl_test: dict, w: float
) -> dict[str, np.ndarray]:
    assert np.array_equal(dl_test["y_true"], cl_test["y_true"]), "test y_true mismatch"
    p_combined = w * dl_test["y_prob"] + (1 - w) * cl_test["y_prob"]
    return {
        "y_true": dl_test["y_true"],
        "y_pred": p_combined.argmax(axis=1),
        "y_prob": p_combined,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dl-val", default=str(RESULTS_DIR / "dl_run_tta_hflip" / "dl_predictions_val.npz")
    )
    parser.add_argument(
        "--dl-test", default=str(RESULTS_DIR / "dl_run_tta_hflip" / "dl_predictions.npz")
    )
    parser.add_argument(
        "--classical-pipeline",
        default=str(RESULTS_DIR / "classical_run" / "classical_pipeline.pkl"),
    )
    parser.add_argument(
        "--classical-test",
        default=str(RESULTS_DIR / "classical_run" / "classical_predictions.npz"),
    )
    parser.add_argument(
        "--classical-features-cache",
        default=str(RESULTS_DIR / "classical_features"),
    )
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "ensemble"))
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--fixed-w",
        type=float,
        default=None,
        help="if given, skip val tuning and use this DL weight (a-priori default)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[ensemble] loading DL val + test preds (TTA hflip)")
    dl_val = dict(np.load(args.dl_val))
    dl_test = dict(np.load(args.dl_test))

    print("[ensemble] generating classical val preds (inline)")
    cl_val = _classical_val_predictions(
        Path(args.classical_pipeline),
        Path(args.classical_features_cache),
    )
    np.savez(
        out_dir / "classical_predictions_val.npz",
        y_true=cl_val["y_true"],
        y_pred=cl_val["y_pred"],
        y_prob=cl_val["y_prob"],
    )

    print("[ensemble] loading classical test preds")
    cl_test = dict(np.load(args.classical_test))

    # Always compute the full val grid for reporting, even if --fixed-w is given
    print(f"[ensemble] computing val weight grid (step={args.step}) …")
    tuned_w, tuned_val_f1, grid = _tune_weight(dl_val, cl_val, step=args.step)
    print(f"[ensemble] val-tuned best w = {tuned_w:.2f}  val_macro_f1 = {tuned_val_f1:.4f}")
    plateau = [w for w, f1 in grid if abs(f1 - tuned_val_f1) < 1e-9]
    if len(plateau) > 1:
        print(f"[ensemble] WARNING: val grid saturated — {len(plateau)} weights tie at {tuned_val_f1:.4f}  (w∈[{plateau[0]:.2f}, {plateau[-1]:.2f}])")
    print("[ensemble] full val curve:")
    for w, f1 in grid:
        marker = "   ← tuned" if abs(w - tuned_w) < 1e-9 else ""
        print(f"    w={w:.2f}  val_f1={f1:.4f}{marker}")

    if args.fixed_w is not None:
        used_w = float(args.fixed_w)
        weight_source = "fixed (a-priori default, not val-tuned)"
        print(f"\n[ensemble] using FIXED w_dl = {used_w} ({weight_source})")
    else:
        used_w = tuned_w
        weight_source = "val-tuned"

    # Apply once to test
    ensemble_test = _apply_ensemble(dl_test, cl_test, used_w)

    # Save predictions and evaluate
    np.savez(
        out_dir / "ensemble_predictions.npz",
        y_true=ensemble_test["y_true"],
        y_pred=ensemble_test["y_pred"],
        y_prob=ensemble_test["y_prob"],
    )

    results = evaluate(
        ensemble_test["y_true"],
        ensemble_test["y_pred"],
        y_prob=ensemble_test["y_prob"],
        model_name=f"ensemble_w{int(round(used_w * 100)):03d}",
    )
    results["ensemble_weight_dl"] = float(used_w)
    results["weight_source"] = weight_source
    results["val_tuned_weight"] = float(tuned_w)
    results["val_tuned_macro_f1"] = float(tuned_val_f1)
    results["val_weight_grid"] = [
        {"w_dl": float(w), "val_macro_f1": float(f1)} for w, f1 in grid
    ]

    print_summary(results)
    save_results(results, out_dir / "ensemble_results.json")

    # Paired McNemar's vs each component on the test set
    print("\n[ensemble] paired comparison on test:")
    for name, cmp_preds in [
        ("DL (TTA hflip)", dl_test["y_pred"]),
        ("Classical",      cl_test["y_pred"]),
    ]:
        mc = mcnemar_test(ensemble_test["y_true"], ensemble_test["y_pred"], cmp_preds)
        print(f"  vs {name:<18} discordant={mc['discordant_pairs']:<3}  "
              f"p={mc['pvalue']:.4g}")


if __name__ == "__main__":
    main()

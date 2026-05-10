"""Run the trained classical pipeline on the held-out test set (Person A).

Loads the pipeline dict written by classical.train, extracts (or loads
cached) test features, and emits:
  * classical_results.json  — full metric report via shared.evaluate
  * classical_predictions.npz — raw (y_true, y_pred, y_prob) arrays for
    downstream McNemar comparison against Person B's deep learning model

Also prints the full Train / CV / Test per-class metrics table by loading
train_cv_metrics.json written during classical.train.

Usage
-----
  python -m classical.predict
  python -m classical.predict --pipeline Results/classical_run/classical_pipeline.pkl
  python -m classical.predict --output-dir Results/classical_run
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from shared.config import CLASSES, RESULTS_DIR
from shared.evaluate import evaluate, print_summary, save_results
from shared.preprocessing import load_split
from classical.config import (
    FEATURES_CACHE_SUBDIR,
    PIPELINE_FILENAME,
    RESULTS_FILENAME,
)
from classical.features import build_feature_matrix
from classical.metrics_utils import (
    CLASS_NAMES,
    print_metrics_table,
    report_to_dict,
)


def predict(
    pipeline_path: Path,
    output_dir: Path,
    cache_dir: Path,
    split_csv: Path | None = None,
    dataset_root: Path | None = None,
    n_jobs: int = 1,
) -> dict:
    # Load pipeline
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    scaler     = pipeline["scaler"]
    pca        = pipeline.get("pca")
    clf        = pipeline["classifier"]
    model_name = pipeline["model_name"]
    n_raw      = pipeline["n_raw_features"]
    pca_info   = f" -> {pca.n_components_} PCA components" if pca is not None else ""
    print(f"[predict] loaded pipeline: {model_name}  ({n_raw} raw features{pca_info})")

    # Load / extract test features
    test_df = load_split("test", split_csv=split_csv, dataset_root=dataset_root)
    X_test, y_true = build_feature_matrix(
        test_df, cache_path=cache_dir / "test.npz",
        desc="test features", n_jobs=n_jobs,
    )

    # Inference
    X_test_sc = scaler.transform(X_test)
    if pca is not None:
        X_test_sc = pca.transform(X_test_sc)
    y_pred = clf.predict(X_test_sc)
    y_prob = clf.predict_proba(X_test_sc)

    # Save raw arrays for McNemar comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "classical_predictions.npz"
    np.savez(npz_path, y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    print(f"[predict] predictions -> {npz_path}")

    # Evaluate and save results
    results = evaluate(y_true, y_pred, y_prob=y_prob, model_name=model_name)
    print_summary(results)
    results_path = save_results(results, output_dir / RESULTS_FILENAME)
    print(f"[predict] results -> {results_path}")

    # ── Full Train / CV / Test per-class metrics table ────────────────────────
    train_cv_path = output_dir / "train_cv_metrics.json"
    if train_cv_path.exists():
        with open(train_cv_path) as f:
            saved = json.load(f)

        test_report_raw = classification_report(
            y_true, y_pred, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )
        test_metrics = report_to_dict(test_report_raw)
        test_acc     = float(accuracy_score(y_true, y_pred))

        print("\n[predict] === Per-class metrics: Train / Cross-Validation / Test ===")
        print_metrics_table(
            train_metrics=saved["train_metrics"],
            cv_metrics=saved["cv_metrics"],
            test_metrics=test_metrics,
            train_acc=saved.get("train_accuracy"),
            cv_acc=saved.get("cv_accuracy"),
            test_acc=test_acc,
        )
    else:
        print(
            "\n[predict] train_cv_metrics.json not found — run classical.train first "
            "to see the full Train/CV/Test table."
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical pipeline test-set inference")
    parser.add_argument(
        "--output-dir", default=str(RESULTS_DIR / "classical_run"),
    )
    parser.add_argument(
        "--pipeline", default=None,
        help="path to pipeline .pkl — defaults to <output-dir>/classical_pipeline.pkl",
    )
    parser.add_argument("--split-csv", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--features-cache-dir", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pipeline_path = (
        Path(args.pipeline) if args.pipeline
        else output_dir / PIPELINE_FILENAME
    )
    cache_dir = (
        Path(args.features_cache_dir)
        if args.features_cache_dir
        else RESULTS_DIR / FEATURES_CACHE_SUBDIR
    )
    predict(
        pipeline_path=pipeline_path,
        output_dir=output_dir,
        cache_dir=cache_dir,
        split_csv=Path(args.split_csv) if args.split_csv else None,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()

"""Run the trained classical pipeline on the held-out test set (Person A).

Loads the pipeline dict written by classical.train, extracts (or loads
cached) test features, and emits:
  • classical_results.json  — full metric report via shared.evaluate
  • classical_predictions.npz — raw (y_true, y_pred, y_prob) arrays for
    downstream McNemar comparison against Person B's deep learning model

Usage
-----
  python -m classical.predict
  python -m classical.predict --pipeline Results/classical_run/classical_pipeline.pkl
  python -m classical.predict --output-dir Results/classical_run
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from shared.config import RESULTS_DIR
from shared.evaluate import evaluate, print_summary, save_results
from shared.preprocessing import load_split
from classical.config import (
    FEATURES_CACHE_SUBDIR,
    PIPELINE_FILENAME,
    RESULTS_FILENAME,
)
from classical.features import build_feature_matrix


def predict(pipeline_path: Path, output_dir: Path, cache_dir: Path) -> dict:
    # Load pipeline
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    scaler = pipeline["scaler"]
    pca = pipeline["pca"]
    clf = pipeline["classifier"]
    model_name = pipeline["model_name"]
    print(f"[predict] loaded pipeline: {model_name}  "
          f"({pipeline['n_raw_features']} → {pipeline['n_pca_components']} dims)")

    # Load / extract test features
    test_df = load_split("test")
    test_cache = cache_dir / "test.npz"
    X_test, y_true = build_feature_matrix(
        test_df, cache_path=test_cache, desc="test features"
    )

    # Inference
    X_test_pca = pca.transform(scaler.transform(X_test))
    y_pred = clf.predict(X_test_pca)
    y_prob = clf.predict_proba(X_test_pca)   # SVC(probability=True), RF, XGB all support this

    # Save raw arrays for McNemar comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "classical_predictions.npz"
    np.savez(npz_path, y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    print(f"[predict] predictions → {npz_path}")

    # Evaluate and save results
    results = evaluate(y_true, y_pred, y_prob=y_prob, model_name=model_name)
    print_summary(results)
    results_path = save_results(results, output_dir / RESULTS_FILENAME)
    print(f"[predict] results → {results_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical pipeline test-set inference")
    default_run = str(RESULTS_DIR / "classical_run")
    parser.add_argument(
        "--pipeline",
        default=str(RESULTS_DIR / "classical_run" / PIPELINE_FILENAME),
        help="path to classical_pipeline.pkl written by classical.train",
    )
    parser.add_argument(
        "--output-dir",
        default=default_run,
        help="directory for classical_results.json and classical_predictions.npz",
    )
    args = parser.parse_args()

    cache_dir = RESULTS_DIR / FEATURES_CACHE_SUBDIR
    predict(
        pipeline_path=Path(args.pipeline),
        output_dir=Path(args.output_dir),
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    main()

"""Absolute baseline: no scaling, no PCA, no class balancing.

Loads the 108-dim feature vectors (from cache if available), trains a
default SVM (linear) and XGBoost on the train split, and evaluates each
on the test split.  Nothing tuned, nothing balanced — just a sanity floor.

Usage
-----
  python -m classical.baseline
  python -m classical.baseline --split-csv split_full.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import SVC
import xgboost as xgb

from shared.config import CLASSES, RESULTS_DIR, SEED
from shared.preprocessing import load_split
from classical.features import build_feature_matrix

CACHE_DIR = RESULTS_DIR / "classical_features"


def _report(name: str, y_test, y_pred) -> None:
    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print("\n" + "=" * 55)
    print(f"  {name}  (no scaling / PCA / balancing)")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro-F1  : {macro_f1:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=list(CLASSES), zero_division=0))
    print("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(description="Absolute baseline (no tuning/balancing)")
    parser.add_argument("--split-csv", default=None)
    parser.add_argument("--dataset-root", default=None)
    args = parser.parse_args()

    split_csv    = Path(args.split_csv)    if args.split_csv    else None
    dataset_root = Path(args.dataset_root) if args.dataset_root else None

    print("[baseline] loading splits ...")
    train_df = load_split("train", split_csv=split_csv, dataset_root=dataset_root)
    test_df  = load_split("test",  split_csv=split_csv, dataset_root=dataset_root)

    print("[baseline] extracting / loading features ...")
    X_train, y_train = build_feature_matrix(
        train_df, cache_path=CACHE_DIR / "train_frac100.npz", desc="train",
    )
    X_test, y_test = build_feature_matrix(
        test_df, cache_path=CACHE_DIR / "test.npz", desc="test",
    )
    print(f"[baseline] train={X_train.shape}  test={X_test.shape}")

    print("\n[baseline] fitting SVM (linear, default C=1) ...")
    svm = SVC(kernel="linear", random_state=SEED)
    svm.fit(X_train, y_train)
    _report("SVM — linear kernel, C=1", y_test, svm.predict(X_test))

    print("\n[baseline] fitting XGBoost (default params) ...")
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    )
    xgb_clf.fit(X_train, y_train)
    _report("XGBoost — default params", y_test, xgb_clf.predict(X_test))

    print("\n[baseline] fitting AdaBoost (default params, 50 estimators) ...")
    ada = AdaBoostClassifier(random_state=SEED)
    ada.fit(X_train, y_train)
    _report("AdaBoost — default params", y_test, ada.predict(X_test))


if __name__ == "__main__":
    main()

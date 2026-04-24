"""Central configuration for the Phase 0 shared infrastructure.

Defaults resolve the medium kidney-CT dataset at `Assignment2/Dataset/...`.
Override DATASET_ROOT by either:
  - setting the env var BMET5933_DATASET_ROOT, or
  - creating `config.local.yaml` at the repo root with key `dataset_root`.

Both are gitignored; partners with different layouts can override locally
without affecting the committed split.csv.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

SEED: int = 42

CLASSES: tuple[str, ...] = ("Cyst", "Normal", "Stone", "Tumor")
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}

TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15

IMAGE_SIZE: tuple[int, int] = (256, 256)

BOOTSTRAP_N: int = 1000
CI_LEVEL: float = 0.95

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
RESULTS_DIR: Path = REPO_ROOT / "Results"
SPLIT_CSV: Path = REPO_ROOT / "split.csv"             # medium dataset (primary comparison)
SPLIT_CSV_FULL: Path = REPO_ROOT / "split_full.csv"   # full 12,446-image dataset (scale validation only)
LOCAL_CONFIG: Path = REPO_ROOT / "config.local.yaml"

_DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "Dataset" / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone_medium"
)


def _resolve_dataset_root() -> Path:
    env = os.environ.get("BMET5933_DATASET_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if LOCAL_CONFIG.exists():
        with LOCAL_CONFIG.open() as f:
            cfg = yaml.safe_load(f) or {}
        root = cfg.get("dataset_root")
        if root:
            return Path(root).expanduser().resolve()
    return _DEFAULT_DATASET_ROOT.resolve()


DATASET_ROOT: Path = _resolve_dataset_root()


if __name__ == "__main__":
    print(f"Repo root:     {REPO_ROOT}")
    print(f"Dataset root:  {DATASET_ROOT}")
    print(f"  exists:      {DATASET_ROOT.exists()}")
    print(f"Split CSV:     {SPLIT_CSV}  (exists: {SPLIT_CSV.exists()})")
    print(f"Results dir:   {RESULTS_DIR}")
    print(f"Seed:          {SEED}")
    print(f"Classes:       {CLASSES}")
    print(f"Split ratios:  train={TRAIN_FRAC}  val={VAL_FRAC}  test={TEST_FRAC}")
    print(f"Image size:    {IMAGE_SIZE}")
    print(f"Bootstrap n:   {BOOTSTRAP_N}  (CI level {CI_LEVEL})")

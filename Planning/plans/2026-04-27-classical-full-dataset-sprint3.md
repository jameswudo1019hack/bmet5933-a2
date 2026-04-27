# Sprint 3 — Classical ML on Full Dataset (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Person A's classical ML pipeline (handcrafted features + grid-searched SVM/RF/XGB) on the 12,446-image full kidney CT dataset using Colab Pro+, then perform paired McNemar's vs EfficientNet-B0-full and ConvNeXt V2-full at matched scale (n=1867 test set), plus a data-efficiency sweep on the full split, with all artefacts logged on Obsidian.

**Architecture:** Add `--split-csv` / `--dataset-root` / `--features-cache-dir` CLI flags to `classical/train.py`, `classical/predict.py`, `classical/sweep.py`. Plumb them through to `shared.preprocessing.load_split(split_csv=…, dataset_root=…)` (which already supports both) and through `shared.data_efficiency.stratified_train_indices(split_csv=…)` for the sweep. Add optional joblib parallel feature extraction in `classical/features.build_feature_matrix`. Keep classical hyperparameters identical to the medium run for a matched-protocol comparison. All full-dataset outputs go to `Results/classical_run_full/`, `Results/classical_features_full/`, and `Results/classical_sweep_full/` so the medium artefacts are preserved.

**Tech Stack:** Python 3.11, scikit-learn 1.4+, scikit-image 0.22+, XGBoost 2.0+, joblib (already a sklearn dependency), statsmodels McNemar, Colab Pro+ high-CPU runtime.

**Out of scope:** retraining DL on a different split; new XGBoost hyperparameters (must match medium grid for matched-protocol claim); patient-level leakage mitigation (still flagged in paper as caveat).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `shared/data_efficiency.py` | modify | accept optional `split_csv=` so the sweep can stratify on full split |
| `classical/train.py` | modify | add `--split-csv`, `--dataset-root`, `--features-cache-dir`, `--n-jobs` |
| `classical/predict.py` | modify | same four flags |
| `classical/sweep.py` | modify | same four flags + pass `split_csv` into `stratified_train_indices` |
| `classical/features.py` | modify | add `n_jobs` parameter to `build_feature_matrix` for parallel extraction |
| `notebooks/colab_classical_full.ipynb` | create | Colab Pro+ orchestration: clone repo, install deps, run train/predict/sweep, push results |
| `analysis/sprint3_full_comparison.py` | create | paired McNemar's: classical-full vs EffNet-B0-full and vs ConvNeXt V2-full + disjoint-error analysis on n=1867 |
| `Results/classical_run_full/` | output | new run outputs (pickle, predictions.npz, results.json, run_log.json) |
| `Results/classical_features_full/` | output | feature cache for full split |
| `Results/classical_sweep_full/` | output | data-efficiency sweep on full |
| `Planning/experiments/Sprint3_classical_on_full.md` | create | Obsidian sprint log |
| `Planning/Home.md` | update | status table + headline numbers |
| `Planning/Results_Summary.md` | update | classical-full tables + new McNemar's table |
| `Planning/Tutor_Meeting_Brief.md` | update | refresh findings if Sprint 3 changes them |
| `Planning/Project_Framing_v2.md` | update | confirm/disconfirm paradigm-stable claim at full scale |

---

## Task 1: Wire `split_csv` parameter into `shared.data_efficiency.stratified_train_indices`

**Files:**
- Modify: `shared/data_efficiency.py:19-41`

**Why first:** the sweep on full needs to stratify against `split_full.csv`, not `split.csv`. The function currently calls `load_split("train")` with default args (medium). Backwards compatible: existing callers stay untouched.

- [ ] **Step 1: Add the optional parameter and pass it through to `load_split`**

```python
def stratified_train_indices(
    fraction: float,
    seed: int = SEED,
    split_csv: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> list[int]:
    """Return sorted indices into the train split containing `fraction` of rows.

    Stratified by class. Same (fraction, seed, split_csv) => same indices.
    `split_csv` and `dataset_root` default to the medium-split values configured
    in shared.config; pass `split_csv=REPO_ROOT/"split_full.csv"` for the full
    dataset sweep.
    """
    if not 0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    train_df = load_split("train", split_csv=split_csv, dataset_root=dataset_root)
    rng = np.random.default_rng(seed)

    chosen: list[int] = []
    for cls in CLASSES:
        cls_indices = train_df.index[train_df["class"] == cls].to_numpy()
        n_keep = int(round(len(cls_indices) * fraction))
        n_keep = max(n_keep, 1)
        picked = rng.choice(cls_indices, size=n_keep, replace=False)
        chosen.extend(int(i) for i in picked)

    return sorted(chosen)
```

Add `from pathlib import Path` at the top if not already present (it currently isn't).

- [ ] **Step 2: Verify medium behaviour unchanged**

Run from repo root:

```bash
python -m shared.data_efficiency
```

Expected: same output as before — four lines like `frac=  0.1  total= 436  Cyst=125  Normal=178  Stone=48  Tumor=85` (medium 4,353 → fractions). Numbers must match what was used in the existing medium sweep.

- [ ] **Step 3: Verify full-split behaviour by passing the new flag**

Run:

```bash
python -c "
from pathlib import Path
from shared.config import REPO_ROOT
from shared.data_efficiency import stratified_train_indices
idx = stratified_train_indices(1.0, split_csv=REPO_ROOT / 'split_full.csv',
                                dataset_root=REPO_ROOT / 'Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone')
print(f'full @ 1.0: n={len(idx)}')
idx10 = stratified_train_indices(0.1, split_csv=REPO_ROOT / 'split_full.csv',
                                  dataset_root=REPO_ROOT / 'Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone')
print(f'full @ 0.1: n={len(idx10)}')
"
```

Expected: `full @ 1.0: n=8712` and `full @ 0.1: n≈871` (sum of per-class round(0.1 * count_per_class)).

- [ ] **Step 4: Commit**

```bash
git add shared/data_efficiency.py
git commit -m "shared/data_efficiency: accept split_csv for full-split sweeps"
```

---

## Task 2: Add `n_jobs` joblib parallelism to `classical.features.build_feature_matrix`

**Files:**
- Modify: `classical/features.py:167-209`

**Why before training plumbing:** extraction dominates wall time on Colab CPU. Wiring this first lets every later run benefit. Default `n_jobs=1` keeps existing call sites identical.

- [ ] **Step 1: Add joblib import**

In `classical/features.py`, near the existing `from skimage…` imports add:

```python
from joblib import Parallel, delayed
```

(joblib ships with scikit-learn — already in `requirements.txt` indirectly.)

- [ ] **Step 2: Add `n_jobs` parameter and the parallel branch**

Replace the body of `build_feature_matrix` (currently lines 167–209). New full function:

```python
def build_feature_matrix(
    df,
    cache_path: Path | None = None,
    desc: str = "extracting features",
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for all rows in df.

    If cache_path is given and the file exists, loads from disk instead of
    recomputing.  On first run the result is written to cache_path (.npz).
    Subsequent calls return instantly from cache.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'abs_path' and 'class_idx' as returned by
        shared.preprocessing.load_split().
    cache_path : Path or None
        Optional path to a .npz cache file.
    desc : str
        tqdm progress-bar description (used only when n_jobs == 1).
    n_jobs : int
        joblib worker count. 1 = serial with tqdm; -1 = all CPUs (Colab Pro+).
    """
    if cache_path is not None and Path(cache_path).exists():
        data = np.load(cache_path)
        print(f"[features] loaded cached features from {cache_path}  "
              f"(shape {data['X'].shape})")
        return data["X"], data["y"]

    paths = df["abs_path"].tolist()
    labels = df["class_idx"].tolist()

    if n_jobs == 1:
        X_list: list[np.ndarray] = []
        for path in tqdm.tqdm(paths, desc=desc, unit="img"):
            X_list.append(_extract_row(path))
    else:
        print(f"[features] {desc}: parallel extraction n_jobs={n_jobs} "
              f"on {len(paths)} images …")
        X_list = Parallel(n_jobs=n_jobs)(
            delayed(_extract_row)(p) for p in paths
        )

    X = np.stack(X_list, axis=0)
    y = np.array(labels, dtype=np.int64)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"[features] saved feature cache -> {cache_path}")

    return X, y
```

- [ ] **Step 3: Equivalence check — parallel must match serial bit-exactly**

Run from repo root:

```bash
python -c "
import numpy as np
from shared.preprocessing import load_split
from classical.features import build_feature_matrix
df = load_split('train').iloc[:16].reset_index(drop=True)
X1, y1 = build_feature_matrix(df, cache_path=None, n_jobs=1)
X4, y4 = build_feature_matrix(df, cache_path=None, n_jobs=4)
np.testing.assert_array_equal(y1, y4)
np.testing.assert_allclose(X1, X4, rtol=1e-6, atol=1e-6)
print('PASS: parallel == serial on 16 rows')
"
```

Expected: `PASS: parallel == serial on 16 rows`. If it fails, the parallel branch is wrong — investigate before continuing.

- [ ] **Step 4: Commit**

```bash
git add classical/features.py
git commit -m "classical/features: add joblib parallel extraction (n_jobs)"
```

---

## Task 3: Add CLI flags to `classical/train.py`

**Files:**
- Modify: `classical/train.py:182-345` (function signature + body) and `347-375` (argparse).

- [ ] **Step 1: Extend the `train()` signature**

Replace the current `def train(...)` signature (line 182) and the cache-dir resolution lines (192–193) with:

```python
def train(
    output_dir: Path,
    smoke: bool = False,
    train_frac: float = 1.0,
    seed: int = SEED,
    cache_dir: Path | None = None,
    split_csv: Path | None = None,
    dataset_root: Path | None = None,
    n_jobs: int = 1,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = RESULTS_DIR / FEATURES_CACHE_SUBDIR

    print(f"[train] output={output_dir}  smoke={smoke}  train_frac={train_frac}")
    print(f"[train] split_csv={split_csv or '(default split.csv)'}  "
          f"dataset_root={dataset_root or '(default DATASET_ROOT)'}  "
          f"n_jobs={n_jobs}")
```

- [ ] **Step 2: Plumb the new params through to `load_split` and `stratified_train_indices`**

Replace lines 198–204 (the `load_split` and `stratified_train_indices` calls) with:

```python
    train_df = load_split("train", split_csv=split_csv, dataset_root=dataset_root)
    val_df = load_split("val", split_csv=split_csv, dataset_root=dataset_root)

    if train_frac < 1.0:
        idxs = stratified_train_indices(
            train_frac, seed=seed,
            split_csv=split_csv, dataset_root=dataset_root,
        )
        train_df = train_df.iloc[idxs].reset_index(drop=True)
        print(f"[train] subsampled train to {len(train_df)} images ({train_frac:.0%})")
```

- [ ] **Step 3: Plumb `n_jobs` through to both `build_feature_matrix` calls**

Replace lines 218–225 (the two `build_feature_matrix` calls) with:

```python
    print("[train] extracting train features …")
    X_train, y_train = build_feature_matrix(
        train_df, cache_path=train_cache, desc="train features", n_jobs=n_jobs,
    )
    print("[train] extracting val features …")
    X_val, y_val = build_feature_matrix(
        val_df, cache_path=val_cache, desc="val features", n_jobs=n_jobs,
    )
```

- [ ] **Step 4: Add the four argparse flags**

Replace `main()` (currently lines 347–371) with:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical ML pipeline")
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "classical_run"),
        help="directory for pipeline pickle and run_log.json",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run (64 train / 32 val) to verify the pipeline runs",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=1.0,
        help="stratified fraction of train split (for data-efficiency sweep)",
    )
    parser.add_argument(
        "--split-csv",
        default=None,
        help="override split CSV (e.g. split_full.csv); default split.csv",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="override image root directory (must match --split-csv)",
    )
    parser.add_argument(
        "--features-cache-dir",
        default=None,
        help="override feature cache directory (default Results/classical_features)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="joblib worker count for feature extraction (-1 = all CPUs)",
    )
    args = parser.parse_args()

    train(
        output_dir=Path(args.output_dir),
        smoke=args.smoke,
        train_frac=args.train_frac,
        cache_dir=Path(args.features_cache_dir) if args.features_cache_dir else None,
        split_csv=Path(args.split_csv) if args.split_csv else None,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        n_jobs=args.n_jobs,
    )
```

- [ ] **Step 5: Smoke-run the new CLI on the full split**

Run from repo root:

```bash
python -m classical.train \
  --smoke \
  --split-csv split_full.csv \
  --dataset-root Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone \
  --output-dir Results/_smoke_classical_full \
  --features-cache-dir Results/_smoke_classical_features_full \
  --n-jobs 4
```

Expected: completes in ≲60 s, prints `[train] done  best=…  val_f1=…`, writes `Results/_smoke_classical_full/classical_pipeline.pkl`, `run_log.json`.

- [ ] **Step 6: Clean up smoke artefacts**

```bash
rm -rf Results/_smoke_classical_full Results/_smoke_classical_features_full
```

- [ ] **Step 7: Commit**

```bash
git add classical/train.py
git commit -m "classical/train: add --split-csv, --dataset-root, --features-cache-dir, --n-jobs"
```

---

## Task 4: Add CLI flags to `classical/predict.py`

**Files:**
- Modify: `classical/predict.py:34-97`

- [ ] **Step 1: Extend the `predict()` signature**

Replace the current `def predict(...)` (line 34) and its `load_split` call (line 47) with:

```python
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

    scaler = pipeline["scaler"]
    pca = pipeline["pca"]
    clf = pipeline["classifier"]
    model_name = pipeline["model_name"]
    print(f"[predict] loaded pipeline: {model_name}  "
          f"({pipeline['n_raw_features']} → {pipeline['n_pca_components']} dims)")
    print(f"[predict] split_csv={split_csv or '(default split.csv)'}  "
          f"dataset_root={dataset_root or '(default DATASET_ROOT)'}  "
          f"n_jobs={n_jobs}")

    # Load / extract test features
    test_df = load_split("test", split_csv=split_csv, dataset_root=dataset_root)
    test_cache = cache_dir / "test.npz"
    X_test, y_true = build_feature_matrix(
        test_df, cache_path=test_cache, desc="test features", n_jobs=n_jobs,
    )
```

(The rest of `predict()` is unchanged.)

- [ ] **Step 2: Add the four argparse flags**

Replace `main()` (currently lines 73–93) with:

```python
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
    parser.add_argument(
        "--split-csv",
        default=None,
        help="override split CSV (e.g. split_full.csv); default split.csv",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="override image root directory (must match --split-csv)",
    )
    parser.add_argument(
        "--features-cache-dir",
        default=None,
        help="override feature cache directory (default Results/classical_features)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="joblib worker count for feature extraction (-1 = all CPUs)",
    )
    args = parser.parse_args()

    cache_dir = (
        Path(args.features_cache_dir)
        if args.features_cache_dir
        else RESULTS_DIR / FEATURES_CACHE_SUBDIR
    )
    predict(
        pipeline_path=Path(args.pipeline),
        output_dir=Path(args.output_dir),
        cache_dir=cache_dir,
        split_csv=Path(args.split_csv) if args.split_csv else None,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        n_jobs=args.n_jobs,
    )
```

- [ ] **Step 3: Verify import + argparse parses**

```bash
python -m classical.predict --help
```

Expected: prints help text including `--split-csv`, `--dataset-root`, `--features-cache-dir`, `--n-jobs`.

- [ ] **Step 4: Commit**

```bash
git add classical/predict.py
git commit -m "classical/predict: add --split-csv, --dataset-root, --features-cache-dir, --n-jobs"
```

---

## Task 5: Add CLI flags to `classical/sweep.py`

**Files:**
- Modify: `classical/sweep.py:95-272`

- [ ] **Step 1: Extend the `run_sweep()` signature**

Replace the current `def run_sweep(...)` (line 95) and the three `load_split` calls (lines 111–113) and the three `build_feature_matrix` calls (lines 116–124) with:

```python
def run_sweep(
    run_dir: Path,
    sweep_out: Path,
    cache_dir: Path,
    split_csv: Path | None = None,
    dataset_root: Path | None = None,
    n_jobs: int = 1,
) -> list[dict]:
    """Train + evaluate at each fraction. Returns list of result dicts."""
    log_path = run_dir / "run_log.json"
    if not log_path.exists():
        raise FileNotFoundError(
            f"{log_path} not found. Run `python -m classical.train` first."
        )
    with open(log_path) as f:
        run_log = json.load(f)

    model_name: str = run_log["best_model"]
    best_params: dict = run_log["best_params"]
    print(f"[sweep] using model={model_name}  params={best_params}")
    print(f"[sweep] split_csv={split_csv or '(default split.csv)'}  "
          f"dataset_root={dataset_root or '(default DATASET_ROOT)'}  "
          f"n_jobs={n_jobs}")

    train_df_full = load_split("train", split_csv=split_csv, dataset_root=dataset_root)
    val_df = load_split("val", split_csv=split_csv, dataset_root=dataset_root)
    test_df = load_split("test", split_csv=split_csv, dataset_root=dataset_root)

    print("[sweep] loading/extracting features …")
    X_train_full, y_train_full = build_feature_matrix(
        train_df_full,
        cache_path=cache_dir / "train_frac100.npz",
        desc="train (100%)",
        n_jobs=n_jobs,
    )
    X_val, y_val = build_feature_matrix(
        val_df, cache_path=cache_dir / "val.npz", desc="val", n_jobs=n_jobs,
    )
    X_test, y_test = build_feature_matrix(
        test_df, cache_path=cache_dir / "test.npz", desc="test", n_jobs=n_jobs,
    )
```

- [ ] **Step 2: Plumb `split_csv`/`dataset_root` into `stratified_train_indices`**

Replace line 133 (the `idxs = stratified_train_indices(frac, seed=SEED)` line) with:

```python
        idxs = stratified_train_indices(
            frac, seed=SEED,
            split_csv=split_csv, dataset_root=dataset_root,
        )
```

- [ ] **Step 3: Add the four argparse flags + dispatch**

Replace `main()` (currently lines 248–267) with:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Classical ML data-efficiency sweep")
    parser.add_argument(
        "--run-dir",
        default=str(RESULTS_DIR / "classical_run"),
        help="directory containing run_log.json from classical.train",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "classical_sweep"),
        help="root directory for sweep results",
    )
    parser.add_argument(
        "--split-csv",
        default=None,
        help="override split CSV (e.g. split_full.csv); default split.csv",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="override image root directory (must match --split-csv)",
    )
    parser.add_argument(
        "--features-cache-dir",
        default=None,
        help="override feature cache directory (default Results/classical_features)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="joblib worker count for feature extraction (-1 = all CPUs)",
    )
    args = parser.parse_args()

    cache_dir = (
        Path(args.features_cache_dir)
        if args.features_cache_dir
        else RESULTS_DIR / FEATURES_CACHE_SUBDIR
    )
    run_sweep(
        run_dir=Path(args.run_dir),
        sweep_out=Path(args.output_dir),
        cache_dir=cache_dir,
        split_csv=Path(args.split_csv) if args.split_csv else None,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        n_jobs=args.n_jobs,
    )
```

- [ ] **Step 4: Verify argparse parses**

```bash
python -m classical.sweep --help
```

Expected: help text including all four new flags.

- [ ] **Step 5: Commit**

```bash
git add classical/sweep.py
git commit -m "classical/sweep: add --split-csv, --dataset-root, --features-cache-dir, --n-jobs"
```

---

## Task 6: End-to-end smoke test on full split (local)

**Why:** before going to Colab we must confirm every flag wires through correctly on a local 64/32-sample run.

- [ ] **Step 1: Run the smoke train**

From repo root:

```bash
python -m classical.train \
  --smoke \
  --split-csv split_full.csv \
  --dataset-root Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone \
  --output-dir Results/_smoke_full \
  --features-cache-dir Results/_smoke_features_full \
  --n-jobs 4
```

Expected: `[train] done  best=… val_f1=… wall=…` and `Results/_smoke_full/classical_pipeline.pkl` exists.

- [ ] **Step 2: Run the smoke predict**

```bash
python -m classical.predict \
  --pipeline Results/_smoke_full/classical_pipeline.pkl \
  --output-dir Results/_smoke_full \
  --split-csv split_full.csv \
  --dataset-root Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone \
  --features-cache-dir Results/_smoke_features_full \
  --n-jobs 4
```

Expected: prints macro-F1 + per-class breakdown for the full 1,867-image test set, writes `classical_predictions.npz` and `classical_results.json`.

- [ ] **Step 3: Sanity-check the predictions npz**

```bash
python -c "
import numpy as np
d = np.load('Results/_smoke_full/classical_predictions.npz')
print('y_true shape:', d['y_true'].shape, 'unique:', sorted(set(d['y_true'].tolist())))
print('y_pred shape:', d['y_pred'].shape)
print('y_prob shape:', d['y_prob'].shape)
"
```

Expected: `y_true shape: (1867,)`, `unique: [0, 1, 2, 3]`, `y_pred shape: (1867,)`, `y_prob shape: (1867, 4)`. (The smoke model itself will be inaccurate — that's fine; we're verifying the wiring, not the metric.)

- [ ] **Step 4: Clean up smoke artefacts**

```bash
rm -rf Results/_smoke_full Results/_smoke_features_full
```

- [ ] **Step 5: No commit** — nothing was added; this is verification only.

---

## Task 7: Build `notebooks/colab_classical_full.ipynb`

**Files:**
- Create: `notebooks/colab_classical_full.ipynb`

**Why a new notebook:** `notebooks/colab_train.ipynb` is the DL notebook and we want each notebook single-purpose for the .ipynb submission later. Mirrors the same clone-pull-run-push pattern.

- [ ] **Step 1: Create the notebook with the cells listed below**

Use a small Python script to build the JSON, since notebook JSON is verbose. Save the script as `/tmp/build_classical_notebook.py`:

```python
import json, pathlib

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src})

md("""# Sprint 3 — Classical ML on Full Dataset (Colab Pro+)

**Person B running Person A's pipeline on the full 12,446-image dataset.**

Runtime: choose **Runtime → Change runtime type → CPU + High-RAM** (A100 GPU is irrelevant for skimage feature extraction). Wall time estimate: 10–25 min depending on vCPU count.

Outputs go to `Results/classical_run_full/` and `Results/classical_sweep_full/` and are pushed back to GitHub at the end.
""")

md("## 1. Authenticate to GitHub (PAT via getpass — never paste in a chat)")
code("""import getpass, os, subprocess

# Prefer Colab userdata if available
try:
    from google.colab import userdata
    PAT = userdata.get('GITHUB_PAT')
    print('Loaded PAT from Colab userdata.')
except Exception:
    PAT = None

if not PAT:
    PAT = getpass.getpass('GitHub PAT (will not be echoed): ').strip()

assert PAT and PAT.startswith(('ghp_', 'github_pat_')), 'PAT looks malformed.'
os.environ['GITHUB_PAT'] = PAT
print('PAT length:', len(PAT))""")

md("## 2. Clone repository")
code("""import subprocess, os, shutil
REPO_URL = f"https://x-access-token:{os.environ['GITHUB_PAT']}@github.com/jameswudo1019hack/bmet5933-a2.git"
REPO_DIR = '/content/bmet5933-a2'

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)

result = subprocess.run(['git', 'clone', REPO_URL, REPO_DIR],
                        capture_output=True, text=True)
print(result.stdout); print(result.stderr)
clone_ok = result.returncode == 0
assert clone_ok, 'git clone failed — check PAT scope (repo) and repo URL.'
%cd /content/bmet5933-a2
""")

md("## 3. Install dependencies")
code("""!pip install -q -r requirements.txt
import sklearn, skimage, xgboost, joblib
print('sklearn', sklearn.__version__,
      ' skimage', skimage.__version__,
      ' xgboost', xgboost.__version__,
      ' joblib', joblib.__version__)""")

md("""## 4. Mount Drive and copy/extract the full dataset

We expect `MyDrive/BMET5933/full.zip` to already be in Drive (uploaded once for Sprint 2). Otherwise upload it now or unzip from a local source.""")
code("""from google.colab import drive
drive.mount('/content/drive')""")

code("""import os, zipfile, shutil, pathlib
DATASET_ROOT = '/content/bmet5933-a2/Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'

if not os.path.exists(DATASET_ROOT):
    src_zip = '/content/drive/MyDrive/BMET5933/full.zip'
    assert os.path.exists(src_zip), f'Expected {src_zip}; upload full.zip first.'
    print('Extracting full.zip …')
    with zipfile.ZipFile(src_zip) as z:
        z.extractall('/content/bmet5933-a2/Dataset')
    print('Extraction done.')

# Verify
for cls in ['Cyst', 'Normal', 'Stone', 'Tumor']:
    n = len(os.listdir(os.path.join(DATASET_ROOT, cls)))
    print(f'{cls}: {n} images')""")

md("## 5. Smoke run — verify wiring before the full job")
code("""!python -m classical.train \\
    --smoke \\
    --split-csv split_full.csv \\
    --dataset-root Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone \\
    --output-dir Results/_smoke_full \\
    --features-cache-dir Results/_smoke_features_full \\
    --n-jobs -1
!rm -rf Results/_smoke_full Results/_smoke_features_full""")

md("""## 6. Full classical training on the 12,446-image split

Same hyperparameter grids as the medium run (`classical/config.py`) — matched-protocol comparison.""")
code("""!python -m classical.train \\
    --split-csv split_full.csv \\
    --dataset-root Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone \\
    --output-dir Results/classical_run_full \\
    --features-cache-dir Results/classical_features_full \\
    --n-jobs -1 2>&1 | tee Results/classical_run_full/train_log.txt""")

md("## 7. Predict on the full test split (n=1,867)")
code("""!python -m classical.predict \\
    --pipeline Results/classical_run_full/classical_pipeline.pkl \\
    --output-dir Results/classical_run_full \\
    --split-csv split_full.csv \\
    --dataset-root Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone \\
    --features-cache-dir Results/classical_features_full \\
    --n-jobs -1 2>&1 | tee Results/classical_run_full/predict_log.txt""")

md("## 8. Data-efficiency sweep on full (10 / 25 / 50 / 100 % of 8,712 train)")
code("""!python -m classical.sweep \\
    --run-dir Results/classical_run_full \\
    --output-dir Results/classical_sweep_full \\
    --split-csv split_full.csv \\
    --dataset-root Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone \\
    --features-cache-dir Results/classical_features_full \\
    --n-jobs -1 2>&1 | tee Results/classical_sweep_full/sweep_log.txt""")

md("## 9. Push results back to GitHub")
code("""!git config user.email 'colab@bmet5933.local'
!git config user.name  'Colab Pro+ runner'

# Drop large feature caches before commit (kept locally on Colab only)
!ls -la Results/classical_run_full/ Results/classical_sweep_full/

# Don't add features cache — it's regenerable and ~MB
!git add Results/classical_run_full Results/classical_sweep_full

!git commit -m "Sprint 3: classical ML on full dataset (train+predict+sweep)"
!git push origin main""")

md("## 10. (Optional) Pack a results zip for download as a backup")
code("""import shutil, datetime
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
out = f'/content/sprint3_results_{stamp}.zip'
shutil.make_archive(out.replace('.zip',''), 'zip',
                    'Results', 'classical_run_full')
print('Backup zip:', out)""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path('notebooks/colab_classical_full.ipynb')
out.write_text(json.dumps(nb, indent=1))
print('wrote', out)
```

Then run it from repo root:

```bash
python /tmp/build_classical_notebook.py
```

Expected: `wrote notebooks/colab_classical_full.ipynb`.

- [ ] **Step 2: Validate the notebook is well-formed JSON**

```bash
python -c "import json; json.load(open('notebooks/colab_classical_full.ipynb')); print('valid JSON')"
```

Expected: `valid JSON`.

- [ ] **Step 3: Confirm no PAT or other secrets are baked in**

```bash
grep -E "ghp_|github_pat_" notebooks/colab_classical_full.ipynb || echo "no PAT in notebook (good)"
```

Expected: `no PAT in notebook (good)`. The notebook prompts via `getpass`/`userdata`; nothing literal.

- [ ] **Step 4: Commit**

```bash
git add notebooks/colab_classical_full.ipynb
git commit -m "notebooks: add Sprint 3 Colab notebook for classical on full dataset"
```

---

## Task 8 [USER ACTION]: Run the notebook on Colab Pro+

**This task is executed by the user, not the agent.** The agent should pause here and wait for the user to confirm the run finished and pushed.

- [ ] **Step 1:** User opens `notebooks/colab_classical_full.ipynb` on Colab Pro+ (CPU + High-RAM runtime).
- [ ] **Step 2:** User runs all cells. Expected wall time 10–25 min depending on vCPU allocation.
- [ ] **Step 3:** User verifies the auto-pushed commit appears on `origin/main` and reports the run finished.
- [ ] **Step 4:** Agent does `git pull` locally so artefacts at `Results/classical_run_full/` are available for analysis.

```bash
git pull origin main
ls Results/classical_run_full Results/classical_sweep_full
```

Expected: directory listings show `classical_pipeline.pkl`, `classical_predictions.npz`, `classical_results.json`, `run_log.json`, `train_log.txt`, `predict_log.txt` (in `classical_run_full`) and `sweep_summary.json`, `data_efficiency_curve.png`, four `frac_*` subdirs (in `classical_sweep_full`).

- [ ] **Step 5:** Quick sanity-check the headline numbers locally:

```bash
python -c "
import json
r = json.load(open('Results/classical_run_full/classical_results.json'))
print(f\"model: {r['model_name']}  acc: {r['accuracy']:.4f}  macro-F1: {r['macro_f1']:.4f}  errors: {r['n_test'] - int(round(r['accuracy'] * r['n_test']))}\")
for cls, m in r['per_class'].items():
    print(f'  {cls}: F1={m[\"f1\"]:.4f}  support={m[\"support\"]}')
"
```

Expected (rough): macro-F1 ≥ 0.99 (consistent with medium = 0.9976; full has more data); per-class supports = Cyst 556, Normal 762, Stone 207, Tumor 342.

---

## Task 9: Comparative analysis — paired McNemar's + disjoint-error sets

**Files:**
- Create: `analysis/__init__.py` (empty, makes the dir a package)
- Create: `analysis/sprint3_full_comparison.py`

- [ ] **Step 1: Create the `analysis/` package directory**

```bash
mkdir -p analysis
touch analysis/__init__.py
```

- [ ] **Step 2: Write `analysis/sprint3_full_comparison.py`**

```python
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
    - paired McNemar's EffNet-B0-full vs ConvNeXt V2-full (sanity, already in Sprint 2 log)
    - both-wrong / only-classical-wrong / only-DL-wrong counts (disjoint-error analysis)
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

    # Confusion of "the wrongs" — which true→pred pairs are the failures?
    def confusion_pairs(y_t: np.ndarray, y_p: np.ndarray, mask: np.ndarray):
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

    only_a_wrong_mask = a_right == False  # noqa: E712
    only_b_wrong_mask = b_right == False  # noqa: E712

    return {
        "model_a": name_a,
        "model_b": name_b,
        "both_correct": both_right,
        f"only_{name_a}_wrong": int(np.sum(~a_right & b_right)),
        f"only_{name_b}_wrong": int(np.sum(a_right & ~b_right)),
        "both_wrong": both_wrong,
        "discordant_pairs": only_a + only_b,
        "mcnemar_statistic": mc["statistic"],
        "mcnemar_pvalue": mc["pvalue"],
        f"{name_a}_wrong_pairs": confusion_pairs(y_true, y_a, only_a_wrong_mask),
        f"{name_b}_wrong_pairs": confusion_pairs(y_true, y_b, only_b_wrong_mask),
    }


def main() -> None:
    for p in (CL_NPZ, EFFNET_NPZ, CONVNEXT_NPZ):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    y_true_cl, y_pred_cl = _load(CL_NPZ)
    y_true_ef, y_pred_ef = _load(EFFNET_NPZ)
    y_true_cn, y_pred_cn = _load(CONVNEXT_NPZ)

    assert np.array_equal(y_true_cl, y_true_ef), \
        "classical and EffNet test labels differ — full split mismatch"
    assert np.array_equal(y_true_cl, y_true_cn), \
        "classical and ConvNeXt V2 test labels differ — full split mismatch"
    y_true = y_true_cl

    pair_cl_ef = _pair_summary("classical", "effnetb0", y_true, y_pred_cl, y_pred_ef)
    pair_cl_cn = _pair_summary("classical", "convnextv2", y_true, y_pred_cl, y_pred_cn)
    pair_ef_cn = _pair_summary("effnetb0", "convnextv2", y_true, y_pred_ef, y_pred_cn)

    out = {
        "n_test": int(len(y_true)),
        "comparisons": {
            "classical_vs_effnetb0_full": pair_cl_ef,
            "classical_vs_convnextv2_full": pair_cl_cn,
            "effnetb0_full_vs_convnextv2_full": pair_ef_cn,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    def _print_pair(label: str, p: dict) -> None:
        print(f"\n=== {label} ===")
        a, b = p["model_a"], p["model_b"]
        print(f"  both correct: {p['both_correct']}")
        print(f"  only {a} wrong: {p[f'only_{a}_wrong']}")
        print(f"  only {b} wrong: {p[f'only_{b}_wrong']}")
        print(f"  both wrong: {p['both_wrong']}")
        print(f"  discordant pairs: {p['discordant_pairs']}")
        print(f"  McNemar p-value: {p['mcnemar_pvalue']:.4g}")
        if p[f"{a}_wrong_pairs"]:
            top = ", ".join(
                f"{r['true']}->{r['pred']}({r['count']})"
                for r in p[f"{a}_wrong_pairs"][:5]
            )
            print(f"  {a} failure pairs: {top}")
        if p[f"{b}_wrong_pairs"]:
            top = ", ".join(
                f"{r['true']}->{r['pred']}({r['count']})"
                for r in p[f"{b}_wrong_pairs"][:5]
            )
            print(f"  {b} failure pairs: {top}")

    print(f"n_test = {len(y_true)}")
    _print_pair("Classical-full vs EfficientNet-B0-full", pair_cl_ef)
    _print_pair("Classical-full vs ConvNeXt V2-full", pair_cl_cn)
    _print_pair("EffNet-B0-full vs ConvNeXt V2-full (Sprint 2 sanity)", pair_ef_cn)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

```bash
python -m analysis.sprint3_full_comparison
```

Expected: prints three pair summaries with McNemar p-values and failure pair counts; writes `Results/classical_run_full/sprint3_comparison.json`. The third comparison (EffNet vs ConvNeXt) should reproduce the Sprint 2 result (p ≈ 0.0021, discordant 27).

- [ ] **Step 4: Commit**

```bash
git add analysis/__init__.py analysis/sprint3_full_comparison.py Results/classical_run_full/sprint3_comparison.json
git commit -m "Sprint 3 analysis: paired McNemars + disjoint-error pairs at full scale"
```

---

## Task 10: Sprint 3 log on Obsidian

**Files:**
- Create: `Planning/experiments/Sprint3_classical_on_full.md`

- [ ] **Step 1: Author the sprint log**

Use the same shape as `Planning/experiments/Sprint2_ConvNeXtV2_on_full.md`:

```markdown
# Sprint 3 — Classical ML on the full dataset

**Date**: 2026-04-27
**Status**: complete
Related: [[Sprint2_ConvNeXtV2_on_full]], [[Sprint1_log]], [[DL_Improvements_Analysis]], [[Phase0_Design]], [[Project_Framing_v2]]

> **Framing.** Sprint 2 closed with classical-on-medium (0.9976) and DL-on-full (EffNet-B0 0.9819, ConvNeXt V2 0.9953). The medium-vs-full asymmetry left "classical fails on Cyst↔Tumor / DL fails on Cyst↔Stone" as a *medium-set* claim. Sprint 3 closes that asymmetry: classical retrained on the same full split (8,712 train, 1,867 test) so all three pipelines are now matched on the n=1867 test set. The scientific question: *does the paradigm-stable error pattern survive at full scale?*

## Decision

Reverses the Sprint 2 scoping note ("not in scope: rebuilding classical on full"). Justification: the matched-test-set comparison is the actual paradigm claim; without it, classical-vs-DL is medium-only. Person A asked Person B to run on Colab Pro+ given the compute access.

## Configuration

| Parameter | Value | Notes |
|---|---|---|
| Pipeline | `classical/{train,predict,sweep}.py` | identical to medium run after CLI flag plumbing |
| Hyperparameter grids | unchanged from `classical/config.py` | matched-protocol — no refitted grids |
| Split | `split_full.csv` (12,446 images, 70/15/15 stratified seed=42) | same split CSV used for EffNet-full and ConvNeXt V2-full |
| Train / val / test | 8,712 / 1,867 / 1,867 | same as DL Sprint 2 |
| Feature extraction | joblib `n_jobs=-1` (Colab Pro+ vCPUs) | new in Sprint 3 |
| Classifiers | SVM (C∈{0.001…1.0}, linear), RF, XGB | same grids as medium |
| Final model selector | val macro-F1 (same criterion as medium) | matches `train.py` selector |

## Results

(fill in actual numbers from `Results/classical_run_full/classical_results.json`)

| Model | Dataset | n_test | Accuracy | Macro-F1 [95% CI] | Errors |
|---|---|---|---|---|---|
| Classical (medium) | medium | 934 | 0.9979 | 0.9976 [0.9939, 1.0000] | 2 |
| **Classical (full)** | **full** | **1867** | **TBD** | **TBD** | **TBD** |
| EfficientNet-B0 + TTA (medium) | medium | 934 | 0.9861 | 0.9829 [0.9728, 0.9918] | 13 |
| EfficientNet-B0 (full) | full | 1867 | 0.9877 | 0.9819 [0.975, 0.989] | 23 |
| ConvNeXt V2 (full) | full | 1867 | 0.9968 | 0.9953 [0.991, 0.998] | 6 |

(Numbers in **bold/italic** to be filled by the agent after Task 9 prints them.)

## Paired McNemar's at full scale (n=1867)

(fill from `Results/classical_run_full/sprint3_comparison.json`)

| Comparison | both correct | only A wrong | only B wrong | both wrong | discordant | p-value |
|---|---|---|---|---|---|---|
| Classical vs EffNet-B0-full | TBD | TBD | TBD | TBD | TBD | TBD |
| Classical vs ConvNeXt V2-full | TBD | TBD | TBD | TBD | TBD | TBD |
| EffNet-B0-full vs ConvNeXt V2-full *(Sprint 2 sanity)* | 1839 | 22 | 5 | 1 | 27 | 0.0021 |

## Disjoint-error analysis at full scale

For the paradigm-stable claim to survive at full scale we need *both* Cyst↔Tumor (classical) and Cyst↔Stone (DL) failure pairs to dominate the *only-X-wrong* sets.

(fill in from `…wrong_pairs` arrays in `sprint3_comparison.json`)

- Top failure pairs of classical-full (only-classical-wrong vs ConvNeXt V2): `TBD`
- Top failure pairs of ConvNeXt V2-full (only-ConvNeXt-wrong vs classical): `TBD`
- Top failure pairs of EffNet-B0-full (only-EffNet-wrong vs classical): `TBD`

**Interpretation rules** (decided up front, before seeing the numbers):

| Outcome | Verdict | Paper handling |
|---|---|---|
| Both DL backbones still dominated by Cyst↔Stone *and* classical still dominated by Cyst↔Tumor | paradigm-stable claim survives | lead with it as Finding 1 (no caveat shift needed) |
| Classical Cyst↔Tumor still dominant, DL pattern shifts to a different pair | partially survives | reframe to "classical pattern is paradigm-stable; DL pattern is dataset-scale-dependent" |
| Classical pattern shifts to match DL (Cyst↔Stone) | does *not* survive | major reframe — paradigm-stable claim retracted; report as a saturation effect |
| Sample sizes too small to claim *any* pattern (≤3 errors per pipeline) | underpowered | report as "all three near-perfect at full scale; failure-mode analysis underpowered" |

## Data-efficiency sweep on the full split

`Results/classical_sweep_full/sweep_summary.json`:

| Train fraction | n_train | Macro-F1 [95% CI] | Errors / 1867 |
|---|---|---|---|
| 10 % | ~871 | TBD | TBD |
| 25 % | ~2178 | TBD | TBD |
| 50 % | ~4356 | TBD | TBD |
| 100 % | 8712 | TBD | TBD |

Compare to the medium-set sweep (Results_Summary.md): does the classical curve plateau at the same place, or does it keep rising?

## Implications for the paper

- **Confirms or invalidates** the "paradigm-stable error patterns" claim from `Project_Framing_v2.md` and `Tutor_Meeting_Brief.md`. Update those docs accordingly in Task 11.
- The full-set ensemble (classical + DL) is now possible — *but* deferred unless tutor explicitly requests it (current Sprint 1 ensemble is on medium and is the canonical headline; rerunning ensemble at full would be a 4th matched-data comparison, out of scope here).
- If classical-full < classical-medium, that itself is a finding ("classical's edge over DL on this dataset shrinks at scale, consistent with Bingol 2023 hybrid 99.37 % and the saturation framing").

## Limitations specific to Sprint 3

1. Same patient-level leakage caveat as all prior runs (no patient IDs available).
2. Single seed (consistent with medium and DL Sprint 2 — no variance characterisation).
3. Hyperparameter grids unchanged from medium — by design (matched protocol), but means the full-set classical may be sub-optimal (a wider grid could have helped). Flag in paper Discussion.
```

Save to `/Users/jameswu/Desktop/University/Year_5/Semester_1/BMET5933/Assignment2/Planning/experiments/Sprint3_classical_on_full.md`.

- [ ] **Step 2: Fill the TBDs**

After running Task 9, replace each `TBD` in the file using actual numbers from:
- `Results/classical_run_full/classical_results.json` → table 1
- `Results/classical_run_full/sprint3_comparison.json` → McNemar's table + disjoint-error pairs
- `Results/classical_sweep_full/sweep_summary.json` → sweep table

Use `Edit` tool, one TBD at a time. The numbers must come from disk — do not invent.

- [ ] **Step 3: Commit**

```bash
git add Planning/experiments/Sprint3_classical_on_full.md
git commit -m "Sprint 3: classical-on-full sprint log with McNemars and disjoint-error analysis"
```

---

## Task 11: Update vault index pages with Sprint 3 results

**Files:**
- Modify: `Planning/Home.md`
- Modify: `Planning/Results_Summary.md`
- Modify: `Planning/Tutor_Meeting_Brief.md`
- Modify: `Planning/Project_Framing_v2.md`

- [ ] **Step 1: Home.md — status table + headline numbers**

Add a row to the status table (`Planning/Home.md` around line 31–43):

```markdown
| Sprint 3 — Classical on full + paired McNemars | Person B (running) | ✅ Complete |
```

Add a row to the headline numbers table (around line 50–55):

```markdown
| Classical (full, supplementary) | Full (n=1867) | **<filled>** | <filled> / 1867 |
```

Add a Sprint 3 link in Analysis section (around line 17–19):

```markdown
- [[experiments/Sprint3_classical_on_full]] — classical on full + paired McNemars at matched scale
```

- [ ] **Step 2: Results_Summary.md — add classical-full section**

Append after the existing "EfficientNet-B0 on full dataset (matched-data control)" section (which ends around line 178). New section:

```markdown
---

## Classical ML on full dataset (Sprint 3)

Same full-dataset split as DL runs, n = 1,867 test. Trained with the same hyperparameter grids and selection criterion as the medium run (`classical/config.py`); only training data and feature cache differ. Hyperparameters were **not** re-tuned to keep the comparison matched-protocol.

| Metric | Value |
|---|---|
| Best model | <fill> |
| Accuracy | <fill> |
| Macro-F1 | <fill> [<lo>, <hi>] |
| Errors | <fill> / 1867 |
| Wall time (Colab Pro+) | <fill> |

### Per-class F1 — Classical (full)

| Class | F1 | Support |
|---|---|---|
| Cyst | <fill> | 556 |
| Normal | <fill> | 762 |
| Stone | <fill> | 207 |
| Tumor | <fill> | 342 |

### Paired McNemar's at full scale (same 1,867 test set)

| Comparison | Both correct | Only A wrong | Only B wrong | Both wrong | Discordant | p-value |
|---|---|---|---|---|---|---|
| Classical vs EfficientNet-B0-full | <fill> | <fill> | <fill> | <fill> | <fill> | <fill> |
| Classical vs ConvNeXt V2-full | <fill> | <fill> | <fill> | <fill> | <fill> | <fill> |

### Disjoint-error analysis at full scale

Top failure pairs of each pipeline (true → pred, count) on the n=1,867 test set:

- Classical: <fill>
- EfficientNet-B0-full: <fill>
- ConvNeXt V2-full: <fill>

(Compare to medium-set patterns: classical Cyst↔Tumor, DL Cyst↔Stone — does the pattern survive at full scale? See `Sprint3_classical_on_full.md` interpretation table.)

### Data-efficiency sweep — classical on full

| Train fraction | n_train | Macro-F1 [95 % CI] | Errors / 1867 |
|---|---|---|---|
| 10 % | <~871> | <fill> | <fill> |
| 25 % | <~2178> | <fill> | <fill> |
| 50 % | <~4356> | <fill> | <fill> |
| 100 % | 8712 | <fill> | <fill> |
```

Use `Edit` tool with the existing section as anchor. Fill `<fill>` placeholders using disk numbers (same files as Task 10 step 2).

- [ ] **Step 3: Tutor_Meeting_Brief.md — refresh findings if Sprint 3 changes them**

Read `Planning/Tutor_Meeting_Brief.md` and update **only if Sprint 3 changes the underlying claim**:

- If classical-full's dominant failure pair is still Cyst↔Tumor and DL backbones still dominated by Cyst↔Stone → **no change**, just append a parenthetical "(now confirmed at full scale, n=1867)" to Finding 1.
- If the pattern shifts → rewrite Finding 1 accordingly and update the "one-liner contribution" at the end.

Specifically, the line currently reading

> "(both DL backbones share the Cyst↔Stone failure mode that classical does not have)"

becomes one of:

```markdown
(both DL backbones share the Cyst↔Stone failure mode that classical does not have; **confirmed on the full 1,867-image test set in Sprint 3**: classical Cyst↔Tumor dominant, both DL Cyst↔Stone dominant, p-values <fill>)
```

or, if it doesn't survive:

```markdown
(this medium-set finding **did not survive at full scale**: see Sprint 3 — classical's failure pair shifted to <fill>, narrowing the paradigm-stability claim)
```

- [ ] **Step 4: Project_Framing_v2.md — confirm or retract the paradigm-stable claim**

Read `Planning/Project_Framing_v2.md`. If Sprint 3 confirms the claim, append a one-paragraph "Sprint 3 update" near the end. If it retracts, edit the thesis paragraph to soften the claim.

Skeleton paragraph:

```markdown
## Sprint 3 update (2026-04-27)

The paradigm-stable error claim was tested at full scale (n=1,867 test) by retraining the classical pipeline on `split_full.csv` and running paired McNemar's against EfficientNet-B0-full and ConvNeXt V2-full. <Outcome>: classical's dominant failure pair was <fill> (vs Cyst↔Tumor on medium), and ConvNeXt V2's was <fill> (vs Cyst↔Stone on medium). The p-values for paired discordance were classical-vs-EffNet <fill>, classical-vs-ConvNeXt <fill>. We <retain / partially retain / retract> the paradigm-stable claim and update Finding 1 in `Tutor_Meeting_Brief.md` accordingly.
```

- [ ] **Step 5: Commit the four-doc update**

```bash
git add Planning/Home.md Planning/Results_Summary.md Planning/Tutor_Meeting_Brief.md Planning/Project_Framing_v2.md
git commit -m "Sprint 3: vault index updates with full-scale classical results"
```

---

## Task 12: Final push and PR-style summary

- [ ] **Step 1: Push everything**

```bash
git push origin main
```

- [ ] **Step 2: Print a one-paragraph summary for the user**

In chat, summarise the Sprint 3 outcome: which model won (XGBoost? same as medium?), the headline macro-F1 vs both DL backbones at matched scale, the paradigm-stable verdict, and the data-efficiency curve direction. Reference `Planning/experiments/Sprint3_classical_on_full.md` as the authoritative log. ≤6 sentences.

- [ ] **Step 3: Update `Planning/Tutor_Meeting_Brief.md` to mention the Sprint 3 timing**

Add one line near the top of the brief: "Sprint 3 (classical on full + paired McNemar's) completed 2026-04-27 — see Sprint3_classical_on_full.md for details." This makes the update visible to the tutor without forcing them to dig.

- [ ] **Step 4: Final commit**

```bash
git add Planning/Tutor_Meeting_Brief.md
git commit -m "Sprint 3: tutor brief mentions Sprint 3 completion"
git push origin main
```

---

## Self-Review Checklist (run before handing off)

- [x] **Spec coverage:** every line in the user's brief — option C, same hyperparameters, new ML notebook — has a task.
  - "Same hyperparameters" → Task 7 step 6 explicitly does **not** override the grids; classical/config.py is unchanged; sweep is reused from medium parameters per `classical.sweep` design (loads `best_params` from `run_log.json`).
  - "New notebook for ML" → Task 7 creates `notebooks/colab_classical_full.ipynb`, distinct from `colab_train.ipynb`.
  - "Run on full" → Tasks 6 (smoke) + 8 (Colab actual) + 9 (analysis).
  - "Sweep on full" (Option C) → Task 8 step 8 (sweep cell) + Task 11 step 2 (sweep table).
  - "Logged on Obsidian" → Tasks 10 + 11.
- [x] **Placeholders scan:** all `<fill>` markers are post-run data substitutions explicitly tied to disk files in Task 10 step 2 / Task 11 step 2 — not implementation gaps.
- [x] **Type consistency:** `split_csv: Path | None` and `dataset_root: Path | None` and `n_jobs: int` used identically in Tasks 1, 2, 3, 4, 5. CLI flags `--split-csv`, `--dataset-root`, `--features-cache-dir`, `--n-jobs` consistent across `train.py`, `predict.py`, `sweep.py`. The `n_jobs` parameter to `build_feature_matrix` matches the `--n-jobs` CLI exactly.
- [x] **No PAT in plan or notebook:** Task 7 step 3 explicitly greps for `ghp_|github_pat_` and rejects.

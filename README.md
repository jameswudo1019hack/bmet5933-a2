# BMET 5933 Assignment 2 — Kidney CT Classification

Pairwise assignment comparing a classical-ML pipeline (handcrafted features) against a transfer-learned CNN on the Islam et al. (2022) kidney CT dataset. Both pipelines share a fixed data split and a single evaluation harness so the comparison is apples-to-apples.

Design rationale is in [Planning/Phase0_Justification.docx](Planning/Phase0_Justification.docx); a mirrored markdown version is in [Planning/Phase0_Design.md](Planning/Phase0_Design.md). Read the justification document before running anything — it is the source of truth for split ratios, preprocessing, metrics, and reporting decisions.

## Directory layout

```
Assignment2/
├── shared/              # Phase 0: split, preprocessing, evaluate, bootstrap, config
├── classical/           # Phase 1: Person A — classical ML with handcrafted features
├── deep_learning/       # Phase 2: Person B — transfer-learned CNN
├── notebooks/           # Submission notebooks (one per team member)
├── Planning/            # Design justification, project docs
├── Results/             # Metrics JSONs, figures, confusion matrices
├── Dataset/             # (gitignored) raw images — download separately
├── Lecture_Slides/      # (gitignored) reference only
├── split.csv            # (generated) fixed stratified split, committed once created
├── requirements.txt
└── README.md
```

Everything outside `shared/`, `classical/`, and `deep_learning/` is either documentation, generated output, or ignored data.

## Setup

Tested on macOS and Linux with Python 3.11. Windows should work via WSL.

```bash
# From the Assignment2/ root:
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are on Colab, skip the venv step — Colab already ships Python 3.11 with most dependencies preinstalled. `!pip install -r requirements.txt` at notebook start is enough.

### CUDA PyTorch

`requirements.txt` pulls the default PyTorch wheel (CPU on most local machines, CUDA on Colab). If you have a local CUDA GPU and want GPU training outside Colab, install torch separately following the picker at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) *before* running `pip install -r requirements.txt`.

## Dataset

The dataset is not committed (too large). Pull it once and point the config at it:

1. Download from the Kaggle release of Islam et al. (2022) — see the assignment brief for the link.
2. Extract so that `Dataset/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone_medium/` exists inside this repo (or anywhere on disk — see config override below).
3. If your copy lives outside the repo, create `config.local.yaml` at the repo root:

```yaml
dataset_root: /absolute/path/to/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone_medium
```

`config.local.yaml` is gitignored, so you and your partner can have different dataset paths without conflict.

## How to reproduce

The shared infrastructure is the bottleneck — once these are built, the per-person pipelines just consume them.

```bash
# 1. Generate the fixed stratified split (run once; the CSV is committed)
python -m shared.split

# 2. Smoke-test the evaluation harness
python -m shared.evaluate --smoke-test

# 3. Run Person A's classical pipeline
python -m classical.train
python -m classical.evaluate

# 4. Run Person B's deep-learning pipeline (Colab recommended)
#    See notebooks/deep_learning.ipynb for the Colab flow.
```

Modules marked above do not yet exist — they are the Phase 0 / Phase 1 / Phase 2 deliverables. This README will be updated as each lands.

## Team convention

- Person A owns `classical/` and `notebooks/<name>_classical.ipynb`.
- Person B owns `deep_learning/` and `notebooks/<name>_deep_learning.ipynb`.
- Both consume `shared/` and `split.csv`. Neither re-splits, re-preprocesses, or re-defines metrics.
- Test set (`split == "test"`) is touched exactly once per final model. All tuning uses train + val only.

## Citations

Primary dataset: Islam MN et al., *Scientific Reports* 12, 11440 (2022). [doi:10.1038/s41598-022-15634-4](https://doi.org/10.1038/s41598-022-15634-4). Full reference list is in the Phase 0 justification document.

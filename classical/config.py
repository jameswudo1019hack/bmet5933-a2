"""Classical ML pipeline hyperparameters (Person A).

All choices are justified in the Phase 2 design document with citations.
Separate from shared.config so deep-learning and classical parameters
cannot accidentally cross-contaminate.
"""
from __future__ import annotations

import math

# ── CLAHE (Pizer et al., 1987, CVGIP 39:355-368) ─────────────────────────────
# Contrast-limited adaptive histogram equalisation applied before all feature
# extraction.  clip_limit is normalised [0, 1]; 0.02 is equivalent to
# OpenCV's clip_limit=2.0 (the standard clinical default).
CLAHE_CLIP_LIMIT: float = 0.02
CLAHE_KERNEL_SIZE: int = 32       # 32-px tiles → 8×8 grid on 256×256 input

# ── GLCM / Haralick (Haralick et al., 1973, IEEE Trans SMC 3:610-621) ────────
# Two distances capture near- and medium-range spatial dependencies.
# Four canonical angles; features are averaged over angles for rotation
# invariance per Haralick's original recommendation.
GLCM_DISTANCES: list[int] = [1, 3]
GLCM_ANGLES: list[float] = [
    0.0,
    math.pi / 4,
    math.pi / 2,
    3 * math.pi / 4,
]
GLCM_LEVELS: int = 64             # quantise 0-255 → 0-63 for speed
GLCM_PROPS: list[str] = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
]

# ── LBP (Ojala et al., 2002, IEEE TPAMI 24:971-987) ──────────────────────────
# Three (P, R) scales follow Ojala's multi-resolution framework: fine (1 px),
# medium (2 px), coarse (3 px) neighbourhoods.
# method='uniform' keeps only the 2-transition patterns, reducing noise.
LBP_PARAMS: list[tuple[int, int]] = [(8, 1), (16, 2), (24, 3)]  # (P, R)

# ── Gabor filter bank (Arivazhagan & Ganesan 2003; Nanni et al. 2013) ────────
# Frequencies span fine-to-coarse texture (wavelengths ≈ 10, 5, 3, 2.5 px).
# Four orientations match the GLCM angle set for consistency.
GABOR_FREQUENCIES: list[float] = [0.1, 0.2, 0.3, 0.4]
GABOR_THETAS: list[float] = [
    0.0,
    math.pi / 4,
    math.pi / 2,
    3 * math.pi / 4,
]

# ── HOG (Dalal & Triggs, 2005, CVPR) ─────────────────────────────────────────
# 32×32-px cells → 8×8 cell grid on 256×256; 2×2 cell blocks;
# L2-Hys normalisation per the original paper.
HOG_ORIENTATIONS: int = 9
HOG_PIXELS_PER_CELL: tuple[int, int] = (32, 32)
HOG_CELLS_PER_BLOCK: tuple[int, int] = (2, 2)

# ── RF Stone class weight ─────────────────────────────────────────────────────
# Stone is confused with Normal at the decision boundary even after SMOTE.
# Boosting Stone's misclassification cost forces RF to prioritise Stone splits.
# Cyst=0, Normal=1, Stone=2, Tumor=3  (matches shared.config.CLASSES order)
RF_CLASS_WEIGHT: str = "balanced"

# ── Intensity histogram (Haralick 1979; Unser et al. 1986) ───────────────────
# 16 coarse bins capture the full pixel-intensity distribution shape.
# Kidney stones appear as bright calcifications in CT (high HU); bins in the
# upper range and p95/p99 directly encode this radiodensity signature.
HIST_BINS: int = 16

# ── Dimensionality reduction ──────────────────────────────────────────────────
# Capped at 50 components (not 95 % variance) to prevent the classifier from
# memorising patient-specific fine-texture patterns that arise from slice-level
# splitting (Yagis et al. 2021, Sci. Rep. 11:22544).  Fewer components force
# the model to rely on coarser, more pathology-specific structure.
PCA_N_COMPONENTS: int = 50

# ── Grid-search CV ────────────────────────────────────────────────────────────
CV_FOLDS: int = 5                 # stratified k-fold, Phase 0 §2
CV_SCORING: str = "f1_macro"      # primary metric, Phase 0 §5

SVM_PARAM_GRID: list = [
    {"kernel": ["linear"], "C": [0.1, 1.0, 10.0, 100.0]},
    {"kernel": ["rbf"], "C": [1.0, 10.0, 100.0, 1000.0], "gamma": ["scale", "auto", 0.001, 0.01]},
]

RF_PARAM_GRID: dict = {
    "n_estimators": [300, 500],
    "max_depth": [10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [1, 2, 4],
}

XGB_PARAM_GRID: dict = {
    "n_estimators": [500, 700],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.05, 0.1],
    "colsample_bytree": [0.7, 0.8],
    "subsample": [0.8],
    "min_child_weight": [1],
    "gamma": [0],
}

# ── Artefact filenames ────────────────────────────────────────────────────────
PIPELINE_FILENAME: str = "classical_pipeline.pkl"
RESULTS_FILENAME: str = "classical_results.json"
FEATURES_CACHE_SUBDIR: str = "classical_features"

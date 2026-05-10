"""Classical feature extraction for kidney CT images (Person A).

Seven complementary feature groups applied to the 256×256 grayscale image
returned by shared.preprocessing.load_image().  Groups 1–5 use the
CLAHE-enhanced image; groups 6–7 use the raw (pre-CLAHE) image to preserve
Stone's absolute radiodensity signature that CLAHE suppresses.

  1. First-order intensity statistics (10 features)
  2. Haralick GLCM texture, rotation-averaged (12 features)
  3. Multi-scale uniform LBP histograms (54 features)
  4. Gabor filter-bank mean + std (32 features)
  5. Intensity histogram + high-end percentiles (18 features)
  6. Pre-CLAHE raw intensity statistics (7 features)
  7. Morphological bright-region features (5 features)

Total raw feature vector: 138 dimensions.
HOG excluded: its 1764-dim shape descriptor dominates the feature space and
achieves near-perfect but leakage-inflated separation on this dataset.
StandardScaler in train.py normalises this before the classifier sees it.

Key references
--------------
Haralick et al. (1973) IEEE Trans SMC 3(6):610-621
Ojala et al. (2002) IEEE TPAMI 24(7):971-987
Pizer et al. (1987) CVGIP 39(3):355-368
Arivazhagan & Ganesan (2003) Pattern Recognit. Lett. 24(9-10):1513-1521
Nanni et al. (2013) Expert Syst. Appl. 40(4):1186-1191
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tqdm
from joblib import Parallel, delayed
from scipy import stats as sp_stats
from scipy.signal import fftconvolve
from skimage.exposure import equalize_adapthist
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor_kernel
from skimage.measure import label, regionprops

from shared.preprocessing import load_image
from classical.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_KERNEL_SIZE,
    GABOR_FREQUENCIES,
    GABOR_THETAS,
    GLCM_ANGLES,
    GLCM_DISTANCES,
    GLCM_LEVELS,
    GLCM_PROPS,
    HIST_BINS,
    LBP_PARAMS,
)


# ── Pre-compute constants once at module load ─────────────────────────────────
_GLCM_STEP: int = 256 // GLCM_LEVELS   # 4 for GLCM_LEVELS=64

# Gabor kernels (real, imag) pre-built so every call to _gabor_features avoids
# re-computing them.  FFT convolution (O(N log N)) is ~30x faster than
# real-domain convolution (O(N * K^2)) for the kernel sizes used here.
_GABOR_KERNELS: list[tuple[np.ndarray, np.ndarray]] = [
    (np.real(gabor_kernel(f, theta=t)), np.imag(gabor_kernel(f, theta=t)))
    for f in GABOR_FREQUENCIES
    for t in GABOR_THETAS
]


def _clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement → uint8."""
    enhanced = equalize_adapthist(
        img,
        kernel_size=CLAHE_KERNEL_SIZE,
        clip_limit=CLAHE_CLIP_LIMIT,
        nbins=256,
    )
    return (enhanced * 255).astype(np.uint8)


def _stat_features(img: np.ndarray) -> np.ndarray:
    """10 first-order intensity statistics."""
    flat = img.flatten().astype(np.float64)
    hist, _ = np.histogram(flat, bins=256, range=(0.0, 256.0), density=False)
    prob = hist / (hist.sum() + 1e-12)
    entropy = -float(np.sum(prob[prob > 0] * np.log2(prob[prob > 0])))
    return np.array([
        flat.mean(),
        flat.std(),
        float(sp_stats.skew(flat)),
        float(sp_stats.kurtosis(flat)),
        entropy,
        float(np.percentile(flat, 10)),
        float(np.percentile(flat, 25)),
        float(np.percentile(flat, 50)),
        float(np.percentile(flat, 75)),
        float(np.percentile(flat, 90)),
    ], dtype=np.float32)


def _glcm_features(img: np.ndarray) -> np.ndarray:
    """12 Haralick GLCM features, rotation-averaged over 4 angles."""
    quantized = (img // _GLCM_STEP).astype(np.uint8)
    cm = graycomatrix(
        quantized,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    feats: list[float] = []
    for prop in GLCM_PROPS:
        vals = graycoprops(cm, prop)          # shape (n_dist, n_angles)
        feats.extend(vals.mean(axis=1).tolist())   # average angles → (n_dist,)
    return np.array(feats, dtype=np.float32)


def _lbp_features(img: np.ndarray) -> np.ndarray:
    """54 multi-scale uniform LBP histogram features."""
    feats: list[float] = []
    for P, R in LBP_PARAMS:
        lbp_img = local_binary_pattern(img, P=P, R=R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(
            lbp_img, bins=n_bins, range=(0.0, float(n_bins)), density=False
        )
        hist_norm = hist.astype(np.float32) / (hist.sum() + 1e-12)
        feats.extend(hist_norm.tolist())
    return np.array(feats, dtype=np.float32)


def _hist_features(img: np.ndarray) -> np.ndarray:
    """18 intensity histogram + high-percentile features.

    16-bin normalized histogram captures the full intensity distribution shape.
    p95 and p99 target the bright-pixel tail where Stone's calcifications sit.
    """
    flat = img.flatten().astype(np.float64)
    hist, _ = np.histogram(flat, bins=HIST_BINS, range=(0.0, 256.0), density=False)
    hist_norm = (hist / (hist.sum() + 1e-12)).astype(np.float32)
    extra = np.array([
        float(np.percentile(flat, 95)),
        float(np.percentile(flat, 99)),
    ], dtype=np.float32)
    return np.concatenate([hist_norm, extra])


def _gabor_features(img_float: np.ndarray) -> np.ndarray:
    """32 Gabor filter-bank features (mean + std of response magnitude).

    Uses FFT convolution (scipy.signal.fftconvolve) which is O(N log N) and
    ~30x faster than real-domain convolution for the kernel sizes here.
    """
    feats: list[float] = []
    for kr, ki in _GABOR_KERNELS:
        magnitude = np.hypot(
            fftconvolve(img_float, kr, mode="same"),
            fftconvolve(img_float, ki, mode="same"),
        )
        feats.append(float(magnitude.mean()))
        feats.append(float(magnitude.std()))
    return np.array(feats, dtype=np.float32)


def _raw_intensity_features(img_raw: np.ndarray) -> np.ndarray:
    """7 features from the pre-CLAHE image preserving absolute radiodensity.

    CLAHE equalises the histogram and suppresses Stone's bright calcification
    signal.  Max, p95/p99, bright-pixel fraction, and statistics of the bright
    subset are computed on the original image before any enhancement.
    """
    flat = img_raw.flatten().astype(np.float64)
    bright = flat[flat > 200]
    return np.array([
        float(flat.max()),
        float(np.percentile(flat, 95)),
        float(np.percentile(flat, 99)),
        float(len(bright) / len(flat)),                          # bright fraction
        float(bright.mean()) if len(bright) > 0 else 0.0,       # mean of bright pixels
        float(bright.std())  if len(bright) > 1 else 0.0,       # std of bright pixels
        float(flat.std()),                                        # overall raw std
    ], dtype=np.float32)


def _morphological_features(img_raw: np.ndarray) -> np.ndarray:
    """5 features from bright connected regions — classical calcification detector.

    Stone produces small, compact, very bright blobs (calcifications) that are
    absent in Normal tissue.  Thresholding at p90 of the raw image isolates
    these high-density regions for structural analysis.
    """
    threshold = float(np.percentile(img_raw, 90))
    binary = img_raw >= threshold
    labeled = label(binary)
    regions = regionprops(labeled)

    if not regions:
        return np.zeros(5, dtype=np.float32)

    areas = np.array([r.area for r in regions], dtype=np.float64)
    return np.array([
        float(len(regions)),                                      # number of bright blobs
        float(areas.max()),                                       # largest blob area
        float(areas.mean()),                                      # average blob area
        float(areas.std()) if len(areas) > 1 else 0.0,           # variation in blob sizes
        float((areas < 500).sum() / len(areas)),                  # fraction of small blobs
    ], dtype=np.float32)


def extract_features(img: np.ndarray) -> np.ndarray:
    """Extract the 108-dimensional feature vector from one image.

    Input must be a 256x256 uint8 grayscale array as returned by
    shared.preprocessing.load_image().
    """
    enhanced = _clahe(img)
    img_float = enhanced.astype(np.float64) / 255.0

    return np.concatenate([
        _stat_features(enhanced),
        _glcm_features(enhanced),
        _lbp_features(enhanced),
        _gabor_features(img_float),
        _hist_features(enhanced),
        _raw_intensity_features(img),
        _morphological_features(img),
    ])


def _extract_row(abs_path: str) -> np.ndarray:
    """Module-level helper so joblib can pickle it for parallel extraction."""
    return extract_features(load_image(abs_path))


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

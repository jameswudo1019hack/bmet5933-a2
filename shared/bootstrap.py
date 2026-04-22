"""Non-parametric percentile bootstrap for metric confidence intervals.

Defaults (1000 resamples, 95 % CI) match the Phase 0 design document.
The seed is forwarded from shared.config so every CI is reproducible.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from shared.config import BOOTSTRAP_N, CI_LEVEL, SEED


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = BOOTSTRAP_N,
    ci: float = CI_LEVEL,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) of `metric_fn` across `n_resamples` bootstrap replicas.

    Sampling is with replacement at the test-set level. `lo` and `hi` are the
    (1-ci)/2 and (1+ci)/2 percentiles of the replicate scores.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot bootstrap over empty arrays")

    rng = np.random.default_rng(seed)
    scores = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(scores, [100 * alpha, 100 * (1 - alpha)])
    return float(np.mean(scores)), float(lo), float(hi)

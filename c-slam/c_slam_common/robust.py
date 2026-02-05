from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import gtsam  # type: ignore
except Exception:
    gtsam = None  # type: ignore


def make_spd(cov: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    cov = np.array(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    jitter = eps
    for _ in range(8):
        try:
            np.linalg.cholesky(cov + np.eye(6) * jitter)
            return cov + np.eye(6) * jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return cov + np.eye(6) * jitter


def gaussian_from_covariance(cov: np.ndarray):
    if gtsam is None:
        raise RuntimeError("GTSAM not available; cannot build noise model")
    cov = make_spd(cov)
    cov = np.array(cov, dtype=np.float64, order="C")
    return gtsam.noiseModel.Gaussian.Covariance(cov)


def robustify(base, kind: Optional[str] = None, k: Optional[float] = None):
    if gtsam is None:
        raise RuntimeError("GTSAM not available; cannot build robust model")
    if not kind:
        return base
    kind = kind.lower()
    if kind == "huber":
        k = 1.345 if k is None else k
        loss = gtsam.noiseModel.mEstimator.Huber(k)
    elif kind == "cauchy":
        k = 1.0 if k is None else k
        loss = gtsam.noiseModel.mEstimator.Cauchy(k)
    else:
        raise ValueError(f"Unsupported robust kernel: {kind}")
    return gtsam.noiseModel.Robust.Create(loss, base)


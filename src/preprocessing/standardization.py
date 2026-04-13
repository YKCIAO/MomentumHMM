from __future__ import annotations

import numpy as np


def _safe_std(x: np.ndarray) -> float:
    value = np.std(x, ddof=0)
    return value if value > 1e-12 else 1.0


def _safe_mad(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return mad if mad > 1e-12 else 1.0


def standardize_1d(x: np.ndarray, method: str) -> np.ndarray:
    if method == "zscore":
        mu = np.mean(x)
        sd = _safe_std(x)
        return (x - mu) / sd
    if method == "robust":
        med = np.median(x)
        mad = _safe_mad(x)
        return (x - med) / mad
    raise ValueError(f"Unknown standardization method: {method}")


def standardize_timeseries(data: np.ndarray, method: str) -> np.ndarray:
    out = np.empty_like(data, dtype=np.float64)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            out[i, j] = standardize_1d(data[i, j], method)
    return out
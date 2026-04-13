from __future__ import annotations

import numpy as np


def ensure_3d(data: np.ndarray) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 3:
        raise ValueError(
            f"Expected data shape [subjects, rois, time], got {data.shape}."
        )
    return data.astype(np.float64, copy=False)


def validate_threshold(threshold: float) -> None:
    if threshold < 0:
        raise ValueError(f"Threshold must be non-negative, got {threshold}.")


def validate_alpha_beta(alpha: float, beta: float) -> None:
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"alpha and beta must be positive, got {alpha}, {beta}.")
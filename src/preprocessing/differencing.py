from __future__ import annotations

import numpy as np


def first_difference(data: np.ndarray, fill_mode: str) -> np.ndarray:
    diff = np.diff(data, axis=-1)

    if fill_mode == "zero":
        first = np.zeros((data.shape[0], data.shape[1], 1), dtype=np.float64)
    elif fill_mode == "repeat":
        first = diff[..., :1].copy()
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")

    return np.concatenate([first, diff], axis=-1)
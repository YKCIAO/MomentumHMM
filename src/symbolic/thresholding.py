from __future__ import annotations

import numpy as np


def trinarize(data: np.ndarray, threshold: float) -> np.ndarray:
    out = np.zeros_like(data, dtype=np.int8)
    out[data > threshold] = 1
    out[data < -threshold] = -1
    return out
from __future__ import annotations

import numpy as np


def moving_average_1d(
    x: np.ndarray,
    window: int,
) -> np.ndarray:

    if window <= 1:
        return x.copy()

    if window % 2 == 0:
        raise ValueError(
            "Smoothing window must be an odd integer."
        )

    if window > len(x):
        raise ValueError(
            "Smoothing window cannot exceed sequence length."
        )

    pad = window // 2

    x_pad = np.pad(
        x,
        pad_width=pad,
        mode="reflect",
    )

    kernel = (
        np.ones(window, dtype=np.float64)
        / float(window)
    )

    return np.convolve(
        x_pad,
        kernel,
        mode="valid",
    )


def smooth_timeseries(data: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(data, dtype=np.float64)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            out[i, j] = moving_average_1d(data[i, j], window)
    return out
from __future__ import annotations

import numpy as np


def temporal_block_average_1d(
    x: np.ndarray,
    window: int,
    drop_remainder: bool = True,
) -> np.ndarray:
    """
    Downsample a 1D time series by averaging non-overlapping temporal blocks.

    Example:
        length = 10, window = 2 -> length = 5
        length = 11, window = 2:
            drop_remainder=True  -> length = 5, discard last point
            drop_remainder=False -> length = 6, keep last incomplete block
    """

    if window <= 1:
        return x.astype(np.float64, copy=True)

    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]

    if drop_remainder:
        T_new = T // window

        if T_new == 0:
            raise ValueError(
                f"Time series length {T} is shorter than window={window}."
            )

        x_trimmed = x[:T_new * window]
        return x_trimmed.reshape(T_new, window).mean(axis=1)

    chunks = [
        x[start:start + window].mean()
        for start in range(0, T, window)
    ]

    return np.asarray(chunks, dtype=np.float64)


def temporal_block_average_timeseries(
    data: np.ndarray,
    window: int,
    drop_remainder: bool = True,
) -> np.ndarray:
    """
    Downsample 3D fMRI time series by averaging adjacent time points.

    Input:
        data shape = [subjects, rois, time]

    Output:
        data shape = [subjects, rois, new_time]

    Example:
        data.shape = [607, 278, 1200], window = 2
        output.shape = [607, 278, 600]
    """

    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError(
            f"Expected data with shape [subjects, rois, time], got {data.shape}"
        )

    if window <= 1:
        return data.astype(np.float64, copy=True)

    n_subjects, n_rois, T = data.shape

    if drop_remainder:
        T_new = T // window
    else:
        T_new = int(np.ceil(T / window))

    if T_new == 0:
        raise ValueError(
            f"Time dimension {T} is shorter than window={window}."
        )

    out = np.empty(
        (n_subjects, n_rois, T_new),
        dtype=np.float64
    )

    for i in range(n_subjects):
        for j in range(n_rois):
            out[i, j] = temporal_block_average_1d(
                data[i, j],
                window=window,
                drop_remainder=drop_remainder,
            )

    return out
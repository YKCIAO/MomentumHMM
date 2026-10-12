from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def resample_timeseries(
    data: np.ndarray,
    source_tr_seconds: float,
    target_tr_seconds: float,
    max_denominator: int = 1000,
) -> np.ndarray:
    """
    Resample subject x ROI x time data to a common temporal resolution.

    scipy.signal.resample_poly applies an anti-aliasing FIR filter before
    downsampling. The resampling ratio is derived from the sampling-rate ratio:

        target_fs / source_fs = source_TR / target_TR

    Parameters
    ----------
    data
        Array shaped (subjects, rois, time).
    source_tr_seconds
        Original repetition time in seconds.
    target_tr_seconds
        Desired repetition time in seconds.
    max_denominator
        Maximum denominator used to approximate the rational resampling ratio.
    """
    x = np.asarray(data, dtype=np.float64)

    if x.ndim != 3:
        raise ValueError(
            f"Expected 3D data (subjects, rois, time), got shape {x.shape}."
        )

    if source_tr_seconds <= 0 or target_tr_seconds <= 0:
        raise ValueError("TR values must be positive.")

    if max_denominator < 1:
        raise ValueError("max_denominator must be >= 1.")

    if np.isclose(source_tr_seconds, target_tr_seconds, rtol=0.0, atol=1e-12):
        return x.copy()

    ratio = Fraction(
        source_tr_seconds / target_tr_seconds
    ).limit_denominator(max_denominator)

    up = ratio.numerator
    down = ratio.denominator

    return resample_poly(
        x,
        up=up,
        down=down,
        axis=-1,
        padtype="line",
    )

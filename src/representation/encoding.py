from __future__ import annotations

import numpy as np

from src.preprocessing.differencing import first_difference
from src.preprocessing.smoothing import smooth_timeseries
from src.preprocessing.standardization import standardize_timeseries
from src.utils.validation import ensure_3d, validate_threshold


def validate_channel_weights(
    alpha: float,
    beta: float,
    use_activation: bool,
    use_trend: bool,
) -> None:
    if alpha < 0 or beta < 0:
        raise ValueError(f"alpha and beta must be non-negative, got alpha={alpha}, beta={beta}")

    if not use_activation and not use_trend:
        raise ValueError("At least one of use_activation or use_trend must be True.")

    if use_activation and alpha == 0 and (not use_trend or beta == 0):
        raise ValueError("Activation is enabled, but alpha=0 and no effective trend channel exists.")

    if use_trend and beta == 0 and (not use_activation or alpha == 0):
        raise ValueError("Trend is enabled, but beta=0 and no effective activation channel exists.")


def ternary_encode(values: np.ndarray, threshold: float) -> np.ndarray:
    """
    Encode continuous values into {-1, 0, +1} using symmetric threshold.
    """
    validate_threshold(threshold)
    code = np.zeros_like(values, dtype=np.int8)
    code[values > threshold] = 1
    code[values < -threshold] = -1
    return code


def maybe_standardize_features(X: np.ndarray) -> np.ndarray:
    """
    Standardize each feature column globally across all timepoints.
    """
    X = X.astype(np.float64, copy=True)
    for j in range(X.shape[1]):
        col = X[:, j]
        mu = np.mean(col)
        sd = np.std(col, ddof=0)
        if sd <= 1e-12:
            sd = 1.0
        X[:, j] = (col - mu) / sd
    return X


def flatten_subject_roi_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten [subjects, rois, time, channels] into [T_total, channels],
    where each subject contributes rois * time observations.

    lengths therefore has shape [subjects], and each subject length = rois * time.
    """
    if features.ndim != 4:
        raise ValueError(
            f"Expected feature tensor with shape [subjects, rois, time, channels], got {features.shape}"
        )

    n_subjects, n_rois, n_timepoints, n_channels = features.shape
    X = features.transpose(0, 1, 2, 3).reshape(n_subjects * n_rois * n_timepoints, n_channels)
    lengths = np.full(n_subjects, n_rois * n_timepoints, dtype=np.int32)
    return X, lengths


def build_gaussian_2d_representation(
    data: np.ndarray,
    standardize_method: str,
    smooth: bool,
    smooth_window: int,
    center_diff_on_diff_series: bool,
    fill_first_diff: str,
    activation_threshold: float,
    trend_threshold: float,
    alpha: float,
    beta: float,
    use_activation: bool,
    use_trend: bool,
    feature_standardize: bool = False,
) -> dict:
    """
    Build 2.0 representation for Gaussian HMM.

    Steps:
      1) ensure input is [subjects, rois, time]
      2) optional smoothing
      3) standardize original time series
      4) compute first difference
      5) optionally standardize diff series
      6) ternary encode activation/trend separately
      7) construct feature channels with alpha/beta as weights or switches
      8) flatten to [T_total, D]
    """
    validate_threshold(activation_threshold)
    validate_threshold(trend_threshold)
    validate_channel_weights(alpha, beta, use_activation, use_trend)

    x = ensure_3d(data)

    if smooth:
        x = smooth_timeseries(x, smooth_window)

    x_std = standardize_timeseries(x, standardize_method)

    dx = first_difference(x_std, fill_first_diff)

    if center_diff_on_diff_series:
        dx_std = standardize_timeseries(dx, standardize_method)
    else:
        dx_std = dx.astype(np.float64, copy=False)

    activation_code = ternary_encode(x_std, activation_threshold)
    trend_code = ternary_encode(dx_std, trend_threshold)

    channels = []
    feature_names = []

    if use_activation and alpha > 0:
        channels.append((alpha * activation_code.astype(np.float64))[..., None])
        feature_names.append("activation")

    if use_trend and beta > 0:
        channels.append((beta * trend_code.astype(np.float64))[..., None])
        feature_names.append("trend")

    if len(channels) == 0:
        raise ValueError("No feature channels were constructed. Check alpha/beta and use_activation/use_trend.")

    feature_tensor = np.concatenate(channels, axis=-1)  # [subjects, rois, time, channels]
    X, lengths = flatten_subject_roi_features(feature_tensor)

    if feature_standardize:
        X = maybe_standardize_features(X)

    return {
        "x_std": x_std,
        "dx_std": dx_std,
        "activation_code": activation_code,
        "trend_code": trend_code,
        "feature_tensor": feature_tensor,
        "X": X,
        "lengths": lengths,
        "feature_names": np.asarray(feature_names, dtype=object),
        "n_subjects": x.shape[0],
        "n_rois": x.shape[1],
        "n_timepoints": x.shape[2],
        "n_features": X.shape[1],
    }
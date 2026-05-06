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
    activation_threshold: float | None,
    trend_threshold: float | None,
    alpha: float,
    beta: float,
    use_activation: bool,
    use_trend: bool,
    activation_encoding: str = "ternary",
    trend_encoding: str = "ternary",
    feature_standardize: bool = False,
) -> dict:
    """
    Build 2.0 representation for Gaussian HMM.

    Rules:
        activation_encoding = "ternary":
            HMM uses activation_code {-1, 0, +1}

        activation_encoding = "continuous":
            HMM uses x_std, i.e., standardized signal

        trend_encoding = "ternary":
            HMM uses trend_code {-1, 0, +1}

        trend_encoding = "continuous":
            HMM uses dx_std, i.e., standardized first difference

    Important:
        If both channels are continuous, thresholds are not used for HMM input.
        Ternary codes are only computed when needed.
    """

    activation_encoding = activation_encoding.lower()
    trend_encoding = trend_encoding.lower()

    if activation_encoding not in {"ternary", "continuous"}:
        raise ValueError(f"Unknown activation_encoding: {activation_encoding}")

    if trend_encoding not in {"ternary", "continuous"}:
        raise ValueError(f"Unknown trend_encoding: {trend_encoding}")

    validate_channel_weights(alpha, beta, use_activation, use_trend)

    x = ensure_3d(data)

    if smooth:
        x = smooth_timeseries(x, smooth_window)

    # continuous activation feature
    x_std = standardize_timeseries(x, standardize_method)

    # continuous trend feature
    dx = first_difference(x_std, fill_first_diff)

    if center_diff_on_diff_series:
        dx_std = standardize_timeseries(dx, standardize_method)
    else:
        dx_std = dx.astype(np.float64, copy=False)

    # --------------------------------------------------
    # Ternary codes are only computed when needed.
    # --------------------------------------------------
    activation_code = None
    trend_code = None

    if activation_encoding == "ternary":
        if activation_threshold is None:
            raise ValueError("activation_threshold is required when activation_encoding='ternary'.")
        validate_threshold(activation_threshold)
        activation_code = ternary_encode(x_std, activation_threshold)

    if trend_encoding == "ternary":
        if trend_threshold is None:
            raise ValueError("trend_threshold is required when trend_encoding='ternary'.")
        validate_threshold(trend_threshold)
        trend_code = ternary_encode(dx_std, trend_threshold)

    # --------------------------------------------------
    # Select actual HMM input per channel.
    # --------------------------------------------------
    if activation_encoding == "continuous":
        activation_feature = x_std.astype(np.float64, copy=False)
    else:
        activation_feature = activation_code.astype(np.float64, copy=False)

    if trend_encoding == "continuous":
        trend_feature = dx_std.astype(np.float64, copy=False)
    else:
        trend_feature = trend_code.astype(np.float64, copy=False)

    channels = []
    feature_names = []

    if use_activation and alpha > 0:
        channels.append((alpha * activation_feature)[..., None])
        feature_names.append(f"activation_{activation_encoding}")

    if use_trend and beta > 0:
        channels.append((beta * trend_feature)[..., None])
        feature_names.append(f"trend_{trend_encoding}")

    if len(channels) == 0:
        raise ValueError("No feature channels were constructed. Check alpha/beta and use_activation/use_trend.")

    feature_tensor = np.concatenate(channels, axis=-1)
    X, lengths = flatten_subject_roi_features(feature_tensor)

    if feature_standardize:
        X = maybe_standardize_features(X)

    # For downstream compatibility, save codes as empty arrays when not used.
    if activation_code is None:
        activation_code_to_save = np.empty((0,), dtype=np.int8)
    else:
        activation_code_to_save = activation_code

    if trend_code is None:
        trend_code_to_save = np.empty((0,), dtype=np.int8)
    else:
        trend_code_to_save = trend_code

    return {
        "x_std": x_std,
        "dx_std": dx_std,
        "activation_code": activation_code_to_save,
        "trend_code": trend_code_to_save,
        "feature_tensor": feature_tensor,
        "X": X,
        "lengths": lengths,
        "feature_names": np.asarray(feature_names, dtype=object),
        "activation_encoding": np.asarray([activation_encoding], dtype=object),
        "trend_encoding": np.asarray([trend_encoding], dtype=object),
        "n_subjects": x.shape[0],
        "n_rois": x.shape[1],
        "n_timepoints": x.shape[2],
        "n_features": X.shape[1],
    }
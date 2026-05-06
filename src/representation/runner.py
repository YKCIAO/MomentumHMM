from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.config import ExperimentConfig
from src.representation.encoding import build_gaussian_2d_representation
from src.utils.io_utils import ensure_dir, save_json, save_npz


def _threshold_grid_for_encoding(encoding: str, thresholds: list[float]) -> list[float | None]:
    """
    If encoding is continuous, thresholds are irrelevant.
    Return [None] to avoid unnecessary threshold combinations.
    """
    encoding = encoding.lower()

    if encoding == "continuous":
        return [None]

    if encoding == "ternary":
        return thresholds

    raise ValueError(f"Unknown encoding: {encoding}")


def _format_threshold_for_tag(name: str, value: float | None, encoding: str) -> str:
    if encoding == "continuous":
        return f"{name}_continuous"
    return f"{name}_{value:.4f}"


def exhaustive_representation_search(
    data: np.ndarray,
    cfg: ExperimentConfig,
) -> None:
    """
    Build all 2.0 feature runs.

    New behavior:
        - continuous channel: no threshold sweep
        - ternary channel: threshold sweep
        - mixed encoding is supported
    """
    out_root = ensure_dir(cfg.paths.symbolic_output_root)
    records = []

    rep_cfg = cfg.representation

    activation_encoding = rep_cfg.activation_encoding.lower()
    trend_encoding = rep_cfg.trend_encoding.lower()

    activation_threshold_grid = _threshold_grid_for_encoding(
        activation_encoding,
        rep_cfg.activation_thresholds,
    )

    trend_threshold_grid = _threshold_grid_for_encoding(
        trend_encoding,
        rep_cfg.trend_thresholds,
    )

    for activation_threshold in activation_threshold_grid:
        for trend_threshold in trend_threshold_grid:
            for alpha in rep_cfg.alpha_values:
                for beta in rep_cfg.beta_values:
                    if alpha == 0 and beta == 0:
                        continue

                    act_tag = _format_threshold_for_tag(
                        "act",
                        activation_threshold,
                        activation_encoding,
                    )
                    trend_tag = _format_threshold_for_tag(
                        "trend",
                        trend_threshold,
                        trend_encoding,
                    )

                    tag = (
                        f"{act_tag}"
                        f"__{trend_tag}"
                        f"__a_{alpha:.4f}"
                        f"__b_{beta:.4f}"
                    )

                    out_dir = ensure_dir(out_root / tag)

                    result = build_gaussian_2d_representation(
                        data=data,
                        standardize_method=cfg.preprocess.standardize_method,
                        smooth=cfg.preprocess.smooth,
                        smooth_window=cfg.preprocess.smooth_window,
                        center_diff_on_diff_series=cfg.preprocess.center_diff_on_diff_series,
                        fill_first_diff=cfg.preprocess.fill_first_diff,
                        activation_threshold=activation_threshold,
                        trend_threshold=trend_threshold,
                        alpha=alpha,
                        beta=beta,
                        use_activation=rep_cfg.use_activation,
                        use_trend=rep_cfg.use_trend,
                        activation_encoding=rep_cfg.activation_encoding,
                        trend_encoding=rep_cfg.trend_encoding,
                        feature_standardize=rep_cfg.feature_standardize,
                    )

                    save_npz(
                        out_dir / "representation_outputs.npz",
                        x_std=result["x_std"],
                        dx_std=result["dx_std"],
                        activation_code=result["activation_code"],
                        trend_code=result["trend_code"],
                        feature_tensor=result["feature_tensor"],
                        feature_names=result["feature_names"],
                        activation_encoding=result["activation_encoding"],
                        trend_encoding=result["trend_encoding"],
                    )

                    save_items = {
                        "X": result["X"],
                        "lengths": result["lengths"],
                        "feature_names": result["feature_names"],
                        "activation_encoding": result["activation_encoding"],
                        "trend_encoding": result["trend_encoding"],
                        "alpha": np.asarray([alpha], dtype=np.float64),
                        "beta": np.asarray([beta], dtype=np.float64),
                    }

                    if activation_threshold is not None:
                        save_items["activation_threshold"] = np.asarray([activation_threshold], dtype=np.float64)

                    if trend_threshold is not None:
                        save_items["trend_threshold"] = np.asarray([trend_threshold], dtype=np.float64)

                    save_npz(
                        out_dir / "hmm_ready_features.npz",
                        **save_items,
                    )

                    meta = {
                        "preprocess": asdict(cfg.preprocess),
                        "representation": {
                            **asdict(cfg.representation),
                            "activation_threshold": activation_threshold,
                            "trend_threshold": trend_threshold,
                            "alpha": alpha,
                            "beta": beta,
                            "activation_encoding": activation_encoding,
                            "trend_encoding": trend_encoding,
                        },
                        "summary": {
                            "n_subjects": int(result["n_subjects"]),
                            "n_rois": int(result["n_rois"]),
                            "n_timepoints": int(result["n_timepoints"]),
                            "n_features": int(result["n_features"]),
                            "feature_names": [str(x) for x in result["feature_names"]],
                        },
                    }

                    save_json(out_dir / "representation_meta.json", meta)

                    records.append(
                        {
                            "output_dir": str(out_dir),
                            "activation_threshold": activation_threshold,
                            "trend_threshold": trend_threshold,
                            "activation_encoding": activation_encoding,
                            "trend_encoding": trend_encoding,
                            "alpha": alpha,
                            "beta": beta,
                        }
                    )

    save_json(out_root / "search_manifest.json", {"runs": records})
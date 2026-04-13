from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import ExperimentConfig
from preprocessing.differencing import first_difference
from preprocessing.smoothing import smooth_timeseries
from preprocessing.standardization import standardize_timeseries
from symbolic.encoding import (
    compute_weighted_values,
    pair_to_fixed_category,
    weighted_values_to_rank_category,
)
from symbolic.sequence_builder import flatten_subject_roi_as_observation_sequence
from symbolic.thresholding import trinarize
from utils.io_utils import ensure_dir, save_json, save_npz
from utils.validation import ensure_3d, validate_alpha_beta, validate_threshold


def build_symbolic_representation(
    data: np.ndarray,
    standardize_method: str,
    smooth: bool,
    smooth_window: int,
    center_diff_on_diff_series: bool,
    fill_first_diff: str,
    deviation_threshold: float,
    momentum_threshold: float,
    alpha: float,
    beta: float,
    category_mode: str,
) -> dict:
    validate_threshold(deviation_threshold)
    validate_threshold(momentum_threshold)
    validate_alpha_beta(alpha, beta)

    x = ensure_3d(data)

    if smooth:
        x = smooth_timeseries(x, smooth_window)

    x_std = standardize_timeseries(x, standardize_method)

    dx = first_difference(x_std, fill_first_diff)

    if center_diff_on_diff_series:
        dx_std = standardize_timeseries(dx, standardize_method)
    else:
        dx_std = dx

    deviation_code = trinarize(x_std, deviation_threshold)
    momentum_code = trinarize(dx_std, momentum_threshold)

    weighted_values = compute_weighted_values(
        deviation_code=deviation_code,
        momentum_code=momentum_code,
        alpha=alpha,
        beta=beta,
    )

    if category_mode == "pair_index":
        category_9 = pair_to_fixed_category(deviation_code, momentum_code)
        weighted_value_map = {}
    elif category_mode == "weighted_rank":
        category_9, weighted_value_map = weighted_values_to_rank_category(weighted_values)
    else:
        raise ValueError(f"Unknown category_mode: {category_mode}")

    obs, lengths = flatten_subject_roi_as_observation_sequence(category_9)

    return {
        "x_std": x_std,
        "dx_std": dx_std,
        "deviation_code": deviation_code,
        "momentum_code": momentum_code,
        "weighted_values": weighted_values,
        "category_9": category_9,
        "obs": obs,
        "lengths": lengths,
        "weighted_value_map": weighted_value_map,
    }


def exhaustive_symbolic_search(
    data: np.ndarray,
    cfg: ExperimentConfig,
) -> None:
    out_root = ensure_dir(cfg.paths.symbolic_output_root)
    records = []

    for deviation_threshold in cfg.symbolic.deviation_thresholds:
        for momentum_threshold in cfg.symbolic.momentum_thresholds:
            for alpha in cfg.symbolic.alpha_values:
                for beta in cfg.symbolic.beta_values:
                    tag = (
                        f"dev_{deviation_threshold:.4f}"
                        f"__mom_{momentum_threshold:.4f}"
                        f"__a_{alpha:.4f}"
                        f"__b_{beta:.4f}"
                    )
                    out_dir = ensure_dir(out_root / tag)

                    result = build_symbolic_representation(
                        data=data,
                        standardize_method=cfg.preprocess.standardize_method,
                        smooth=cfg.preprocess.smooth,
                        smooth_window=cfg.preprocess.smooth_window,
                        center_diff_on_diff_series=cfg.preprocess.center_diff_on_diff_series,
                        fill_first_diff=cfg.preprocess.fill_first_diff,
                        deviation_threshold=deviation_threshold,
                        momentum_threshold=momentum_threshold,
                        alpha=alpha,
                        beta=beta,
                        category_mode=cfg.symbolic.category_mode,
                    )

                    save_npz(
                        out_dir / "symbolic_outputs.npz",
                        x_std=result["x_std"],
                        dx_std=result["dx_std"],
                        deviation_code=result["deviation_code"],
                        momentum_code=result["momentum_code"],
                        weighted_values=result["weighted_values"],
                        category_9=result["category_9"],
                    )

                    save_npz(
                        out_dir / "hmm_ready_sequence.npz",
                        obs=result["obs"],
                        lengths=result["lengths"],
                    )

                    meta = {
                        "preprocess": asdict(cfg.preprocess),
                        "symbolic": {
                            **asdict(cfg.symbolic),
                            "deviation_threshold": deviation_threshold,
                            "momentum_threshold": momentum_threshold,
                            "alpha": alpha,
                            "beta": beta,
                        },
                        "weighted_value_map": result["weighted_value_map"],
                    }
                    save_json(out_dir / "symbolic_meta.json", meta)

                    records.append(
                        {
                            "output_dir": str(out_dir),
                            "deviation_threshold": deviation_threshold,
                            "momentum_threshold": momentum_threshold,
                            "alpha": alpha,
                            "beta": beta,
                        }
                    )

    save_json(out_root / "search_manifest.json", {"runs": records})
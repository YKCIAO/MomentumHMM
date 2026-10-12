from __future__ import annotations

from dataclasses import asdict

import numpy as np

from src.config import ExperimentConfig
from src.preprocessing.differencing import first_difference
from src.preprocessing.downsampling import resample_timeseries
from src.preprocessing.smoothing import smooth_timeseries
from src.preprocessing.standardization import standardize_timeseries
from src.symbolic.encoding import pair_to_fixed_category
from src.symbolic.sequence_builder import flatten_subject_roi_as_observation_sequence
from src.symbolic.thresholding import trinarize
from src.utils.io_utils import ensure_dir, save_json, save_npz
from src.utils.validation import ensure_3d, validate_threshold


def build_symbolic_representation(
    data: np.ndarray,
    standardize_method: str,
    smooth: bool,
    smooth_window: int,
    center_diff_on_diff_series: bool,
    fill_first_diff: str,
    deviation_threshold: float,
    momentum_threshold: float,
    downsample: bool = False,
    source_tr_seconds: float = 2.0,
    target_tr_seconds: float = 2.0,
    resample_max_denominator: int = 1000,
) -> dict:
    validate_threshold(deviation_threshold)
    validate_threshold(momentum_threshold)

    x = ensure_3d(data)
    n_timepoints_input = int(x.shape[-1])

    # Temporal harmonization is intentionally independent of smoothing.
    # If enabled, it is applied first so both datasets enter the remaining
    # preprocessing steps at the same physical sampling interval.
    if downsample:
        x = resample_timeseries(
            x,
            source_tr_seconds=source_tr_seconds,
            target_tr_seconds=target_tr_seconds,
            max_denominator=resample_max_denominator,
        )

    n_timepoints_after_resampling = int(x.shape[-1])

    if smooth:
        x = smooth_timeseries(x, smooth_window)

    x_std = standardize_timeseries(x, standardize_method)

    dx = first_difference(x_std, fill_first_diff)

    if center_diff_on_diff_series:
        dx_std = standardize_timeseries(dx, standardize_method)
    else:
        dx_std = dx

    deviation_code = trinarize(
        x_std,
        deviation_threshold,
    )

    momentum_code = trinarize(
        dx_std,
        momentum_threshold,
    )

    category_9 = pair_to_fixed_category(
        deviation_code,
        momentum_code,
    )

    (
        obs,
        lengths,
        sequence_subject_ids,
        sequence_roi_ids,
    ) = flatten_subject_roi_as_observation_sequence(
        category_9
    )

    return {
        "x_std": x_std,
        "dx_std": dx_std,
        "deviation_code": deviation_code,
        "momentum_code": momentum_code,
        "category_9": category_9,
        "obs": obs,
        "lengths": lengths,
        "sequence_subject_ids": sequence_subject_ids,
        "sequence_roi_ids": sequence_roi_ids,
        "n_timepoints_input": n_timepoints_input,
        "n_timepoints_after_resampling": n_timepoints_after_resampling,
    }


def exhaustive_symbolic_search(
    data: np.ndarray,
    cfg: ExperimentConfig,
) -> None:

    out_root = ensure_dir(cfg.paths.symbolic_output_root)
    records = []

    for deviation_threshold in cfg.symbolic.deviation_thresholds:
        for momentum_threshold in cfg.symbolic.momentum_thresholds:

            tag = (
                f"dev_{deviation_threshold:.4f}"
                f"__mom_{momentum_threshold:.4f}"
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
                downsample=cfg.preprocess.downsample,
                source_tr_seconds=cfg.preprocess.source_tr_seconds,
                target_tr_seconds=cfg.preprocess.target_tr_seconds,
                resample_max_denominator=cfg.preprocess.resample_max_denominator,
            )

            save_npz(
                out_dir / "symbolic_outputs.npz",
                x_std=result["x_std"],
                dx_std=result["dx_std"],
                deviation_code=result["deviation_code"],
                momentum_code=result["momentum_code"],
                category_9=result["category_9"],
            )

            save_npz(
                out_dir / "hmm_ready_sequence.npz",
                obs=result["obs"],
                lengths=result["lengths"],
                sequence_subject_ids=result["sequence_subject_ids"],
                sequence_roi_ids=result["sequence_roi_ids"],
            )

            effective_tr_seconds = (
                cfg.preprocess.target_tr_seconds
                if cfg.preprocess.downsample
                else cfg.preprocess.source_tr_seconds
            )

            meta = {
                "preprocess": asdict(cfg.preprocess),
                "temporal_sampling": {
                    "n_timepoints_input": result["n_timepoints_input"],
                    "n_timepoints_after_resampling": result[
                        "n_timepoints_after_resampling"
                    ],
                    "effective_tr_seconds": effective_tr_seconds,
                },
                "symbolic": {
                    "category_mode": "deviation_momentum_pair_index",
                    "deviation_threshold": deviation_threshold,
                    "momentum_threshold": momentum_threshold,
                },
            }

            save_json(
                out_dir / "symbolic_meta.json",
                meta,
            )

            records.append(
                {
                    "output_dir": str(out_dir),
                    "deviation_threshold": deviation_threshold,
                    "momentum_threshold": momentum_threshold,
                    "n_timepoints_input": result["n_timepoints_input"],
                    "n_timepoints_after_resampling": result[
                        "n_timepoints_after_resampling"
                    ],
                    "effective_tr_seconds": effective_tr_seconds,
                }
            )

    save_json(
        out_root / "search_manifest.json",
        {"runs": records},
    )

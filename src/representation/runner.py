from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.config import ExperimentConfig
from src.representation.encoding import build_gaussian_2d_representation
from src.utils.io_utils import ensure_dir, save_json, save_npz


def exhaustive_representation_search(
    data: np.ndarray,
    cfg: ExperimentConfig,
) -> None:
    """
    Build all 2.0 feature runs across:
      activation_threshold x trend_threshold x alpha x beta
    and save each run under cfg.paths.symbolic_output_root.

    We keep the existing root path name for compatibility, but contents are now
    2.0 feature outputs instead of 1.0 symbolic outputs.
    """
    out_root = ensure_dir(cfg.paths.symbolic_output_root)
    records = []

    rep_cfg = cfg.representation

    for activation_threshold in rep_cfg.activation_thresholds:
        for trend_threshold in rep_cfg.trend_thresholds:
            for alpha in rep_cfg.alpha_values:
                for beta in rep_cfg.beta_values:
                    # skip fully disabled configuration
                    if alpha == 0 and beta == 0:
                        continue

                    tag = (
                        f"act_{activation_threshold:.4f}"
                        f"__trend_{trend_threshold:.4f}"
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
                    )

                    save_npz(
                        out_dir / "hmm_ready_features.npz",
                        X=result["X"],
                        lengths=result["lengths"],
                        feature_names=result["feature_names"],
                        activation_threshold=np.asarray([activation_threshold], dtype=np.float64),
                        trend_threshold=np.asarray([trend_threshold], dtype=np.float64),
                        alpha=np.asarray([alpha], dtype=np.float64),
                        beta=np.asarray([beta], dtype=np.float64),
                    )

                    meta = {
                        "preprocess": asdict(cfg.preprocess),
                        "representation": {
                            **asdict(cfg.representation),
                            "activation_threshold": activation_threshold,
                            "trend_threshold": trend_threshold,
                            "alpha": alpha,
                            "beta": beta,
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
                            "alpha": alpha,
                            "beta": beta,
                        }
                    )

    save_json(out_root / "search_manifest.json", {"runs": records})
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.hmm.decode import decode_hmm, split_sequence_by_lengths
from src.hmm.metrics import compute_subject_level_metrics, extract_dwell_times
from src.hmm.model import fit_gaussian_hmm
from src.hmm.runner import load_subject_metadata
from src.preprocessing.differencing import first_difference
from src.preprocessing.downsampling import resample_timeseries
from src.preprocessing.smoothing import smooth_timeseries
from src.preprocessing.standardization import standardize_timeseries
from src.utils.io_utils import ensure_dir, load_npy, save_json, save_npz
from src.utils.validation import ensure_3d


def log_step(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def preprocess_continuous_data(
    data: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    x = ensure_3d(data)
    n_timepoints_input = int(x.shape[-1])

    if cfg.preprocess.downsample:
        x = resample_timeseries(
            x,
            source_tr_seconds=cfg.preprocess.source_tr_seconds,
            target_tr_seconds=cfg.preprocess.target_tr_seconds,
            max_denominator=cfg.preprocess.resample_max_denominator,
        )

    n_timepoints_after_resampling = int(x.shape[-1])

    if cfg.preprocess.smooth:
        x = smooth_timeseries(
            x,
            cfg.preprocess.smooth_window,
        )

    x_std = standardize_timeseries(
        x,
        cfg.preprocess.standardize_method,
    )

    dx = first_difference(
        x_std,
        cfg.preprocess.fill_first_diff,
    )

    if cfg.preprocess.center_diff_on_diff_series:
        dx_std = standardize_timeseries(
            dx,
            cfg.preprocess.standardize_method,
        )
    else:
        dx_std = dx

    meta = {
        "n_timepoints_input": n_timepoints_input,
        "n_timepoints_after_resampling": n_timepoints_after_resampling,
    }

    return x_std, dx_std, meta


def build_gaussian_observations(
    data: np.ndarray,
    cfg: ExperimentConfig,
) -> dict:
    """
    Construct continuous observations for Gaussian-HMM baselines.

    sequence_layout = "subject_roi"
        Same sequence organization as MomentumHMM. Each subject x ROI is an
        independent sequence. This baseline isolates the effect of replacing
        symbolic observations with continuous Gaussian emissions.

    sequence_layout = "whole_brain"
        Conventional neuroimaging Gaussian HMM. Each subject is one sequence
        and all ROIs form a multivariate observation vector at each time point.
        For this layout, observation_mode must be "deviation".
    """
    x_std, dx_std, temporal_meta = preprocess_continuous_data(
        data,
        cfg,
    )

    layout = cfg.gaussian_hmm.sequence_layout
    mode = cfg.gaussian_hmm.observation_mode

    n_subjects, n_rois, n_timepoints = x_std.shape

    if layout == "whole_brain":
        if mode != "deviation":
            raise ValueError(
                "whole_brain Gaussian HMM currently requires "
                "gaussian_hmm.observation_mode='deviation'. "
                "This gives the conventional multivariate ROI baseline."
            )

        seqs = [
            x_std[subj_idx].T
            for subj_idx in range(n_subjects)
        ]

        obs = np.concatenate(
            seqs,
            axis=0,
        ).astype(np.float64, copy=False)

        return {
            "obs": obs,
            "lengths": np.full(
                n_subjects,
                n_timepoints,
                dtype=np.int64,
            ),
            "sequence_subject_ids": np.arange(
                n_subjects,
                dtype=np.int64,
            ),
            "sequence_roi_ids": None,
            "feature_names": [
                f"ROI_{roi_idx}"
                for roi_idx in range(n_rois)
            ],
            "n_features": n_rois,
            "n_rois": n_rois,
            "sequence_layout": layout,
            **temporal_meta,
        }

    if layout != "subject_roi":
        raise ValueError(
            "Unsupported gaussian_hmm.sequence_layout: "
            f"{layout}"
        )

    if mode == "deviation":
        features = x_std[..., np.newaxis]
        feature_names = ["deviation"]
    elif mode == "momentum":
        features = dx_std[..., np.newaxis]
        feature_names = ["momentum"]
    elif mode == "deviation_momentum":
        features = np.stack(
            [x_std, dx_std],
            axis=-1,
        )
        feature_names = ["deviation", "momentum"]
    else:
        raise ValueError(
            "Unsupported gaussian_hmm.observation_mode: "
            f"{mode}"
        )

    _, _, _, n_features = features.shape

    seqs = []
    lengths = []
    sequence_subject_ids = []
    sequence_roi_ids = []

    for subj_idx in range(n_subjects):
        for roi_idx in range(n_rois):
            seqs.append(
                features[subj_idx, roi_idx, :, :]
            )
            lengths.append(n_timepoints)
            sequence_subject_ids.append(subj_idx)
            sequence_roi_ids.append(roi_idx)

    obs = np.concatenate(
        seqs,
        axis=0,
    ).astype(np.float64, copy=False)

    return {
        "obs": obs,
        "lengths": np.asarray(lengths, dtype=np.int64),
        "sequence_subject_ids": np.asarray(
            sequence_subject_ids,
            dtype=np.int64,
        ),
        "sequence_roi_ids": np.asarray(
            sequence_roi_ids,
            dtype=np.int64,
        ),
        "feature_names": feature_names,
        "n_features": n_features,
        "n_rois": n_rois,
        "sequence_layout": layout,
        **temporal_meta,
    }


def _compute_whole_brain_subject_metrics(
    state_seqs: list[np.ndarray],
    n_hidden_states: int,
) -> dict:
    n_subjects = len(state_seqs)

    fo = np.zeros(
        (n_subjects, n_hidden_states),
        dtype=np.float64,
    )
    mdt = np.zeros_like(fo)
    visits = np.zeros(
        (n_subjects, n_hidden_states),
        dtype=np.int64,
    )

    for subj_idx, seq in enumerate(state_seqs):
        for state_idx in range(n_hidden_states):
            fo[subj_idx, state_idx] = np.mean(
                seq == state_idx
            )

            runs = extract_dwell_times(
                seq,
                state_idx,
            )
            visits[subj_idx, state_idx] = len(runs)

            if runs:
                mdt[subj_idx, state_idx] = float(
                    np.mean(runs)
                )

    return {
        "FO": fo,
        "MDT": mdt,
        "Visits": visits,
    }


def run_single_gaussian_hmm_task(
    obs: np.ndarray,
    lengths: np.ndarray,
    sequence_subject_ids: np.ndarray,
    sequence_roi_ids: np.ndarray | None,
    metadata_records: list[dict],
    out_root_str: str,
    n_hidden_states: int,
    gaussian_cfg: dict,
    feature_names: list[str],
    temporal_meta: dict,
    n_rois: int,
    save_posterior: bool = True,
) -> str:
    metadata_df = pd.DataFrame(metadata_records)
    out_root = Path(out_root_str)

    model = fit_gaussian_hmm(
        obs=obs,
        lengths=lengths,
        n_hidden_states=n_hidden_states,
        n_iter=gaussian_cfg["n_iter"],
        tol=gaussian_cfg["tol"],
        random_state=gaussian_cfg["random_state"],
        verbose=gaussian_cfg["verbose"],
        covariance_type=gaussian_cfg["covariance_type"],
        min_covar=gaussian_cfg["min_covar"],
    )

    decoded = decode_hmm(
        model,
        obs,
        lengths,
    )

    state_seqs = split_sequence_by_lengths(
        decoded["state_sequence"],
        lengths,
    )

    n_subjects = len(metadata_df)
    layout = gaussian_cfg["sequence_layout"]

    if layout == "subject_roi":
        metrics = compute_subject_level_metrics(
            state_seqs=state_seqs,
            sequence_subject_ids=sequence_subject_ids,
            sequence_roi_ids=sequence_roi_ids,
            n_subjects=n_subjects,
            n_rois=n_rois,
            n_hidden_states=n_hidden_states,
        )
    else:
        metrics = _compute_whole_brain_subject_metrics(
            state_seqs=state_seqs,
            n_hidden_states=n_hidden_states,
        )

    out_dir = ensure_dir(
        out_root / f"K_{n_hidden_states}"
    )

    save_dict = {
        "startprob_": model.startprob_,
        "transmat_": model.transmat_,
        "means_": model.means_,
        "covars_": model.covars_,
        "state_sequence": decoded["state_sequence"],
        "FO": metrics["FO"],
        "MDT": metrics["MDT"],
        "Visits": metrics["Visits"],
        "lengths": lengths,
        "sequence_subject_ids": sequence_subject_ids,
        "subject_ids": metadata_df["ID"].astype(
            str
        ).to_numpy(dtype=object),
        "subject_age": metadata_df["Age"].to_numpy(),
        "subject_gender": metadata_df["Gender"].astype(
            str
        ).to_numpy(dtype=object),
    }

    if sequence_roi_ids is not None:
        save_dict["sequence_roi_ids"] = sequence_roi_ids

    if layout == "subject_roi":
        save_dict["FO_roi"] = metrics["FO_roi"]
        save_dict["MDT_roi"] = metrics["MDT_roi"]
        save_dict["Visits_roi"] = metrics["Visits_roi"]

    if save_posterior:
        save_dict["posterior"] = decoded[
            "posterior"
        ].astype("float32", copy=False)

    save_npz(
        out_dir / "hmm_results.npz",
        **save_dict,
    )

    rows = []

    for subj_idx in range(n_subjects):
        if layout == "subject_roi":
            subject_sequence_mask = (
                sequence_subject_ids == subj_idx
            )
            subject_total_timepoints = int(
                lengths[subject_sequence_mask].sum()
            )
            n_sequences_for_subject = int(
                subject_sequence_mask.sum()
            )
        else:
            subject_total_timepoints = int(
                lengths[subj_idx]
            )
            n_sequences_for_subject = 1

        row = {
            "subject_index": subj_idx,
            "ID": str(metadata_df.loc[subj_idx, "ID"]),
            "Age": metadata_df.loc[subj_idx, "Age"],
            "Gender": metadata_df.loc[subj_idx, "Gender"],
            "total_timepoints": subject_total_timepoints,
            "n_roi_sequences": n_sequences_for_subject,
        }

        for state_idx in range(n_hidden_states):
            row[f"FO_state_{state_idx}"] = float(
                metrics["FO"][subj_idx, state_idx]
            )
            row[f"MDT_state_{state_idx}"] = float(
                metrics["MDT"][subj_idx, state_idx]
            )
            row[f"Visits_state_{state_idx}"] = int(
                metrics["Visits"][subj_idx, state_idx]
            )

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        out_dir / "subject_metrics.csv",
        index=False,
    )

    # Export Gaussian state means in a human-readable table.
    # For whole_brain layout, columns correspond to ROI_0 ... ROI_(R-1).
    # For subject_roi layout, columns are the continuous observation features.
    state_mean_columns = feature_names
    state_means_df = pd.DataFrame(
        model.means_,
        columns=state_mean_columns,
    )
    state_means_df.insert(
        0,
        "state",
        np.arange(n_hidden_states, dtype=int),
    )
    state_means_df.to_csv(
        out_dir / "state_gaussian_means.csv",
        index=False,
    )

    save_json(
        out_dir / "hmm_meta.json",
        {
            "model_type": "GaussianHMM",
            "sequence_layout": layout,
            "observation_mode": gaussian_cfg[
                "observation_mode"
            ],
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "covariance_type": gaussian_cfg[
                "covariance_type"
            ],
            "min_covar": gaussian_cfg["min_covar"],
            "n_hidden_states": n_hidden_states,
            "n_iter": gaussian_cfg["n_iter"],
            "tol": gaussian_cfg["tol"],
            "random_state": gaussian_cfg[
                "random_state"
            ],
            "verbose": gaussian_cfg["verbose"],
            "logprob": float(decoded["logprob"]),
            "n_subjects": n_subjects,
            "n_sequences": int(len(lengths)),
            "n_rois": int(n_rois),
            "sequence_unit": (
                "subject_roi"
                if layout == "subject_roi"
                else "subject"
            ),
            "save_posterior": bool(save_posterior),
            **temporal_meta,
        },
    )

    return (
        f"Finished GaussianHMM | layout={layout} "
        f"| mode={gaussian_cfg['observation_mode']} "
        f"| K={n_hidden_states}"
    )


def fit_all_gaussian_hmm_runs_parallel(
    cfg: ExperimentConfig,
    max_workers: int = 1,
    save_posterior: bool = True,
) -> None:
    if not cfg.gaussian_hmm.enabled:
        log_step(
            "Gaussian HMM is disabled in config; nothing to run."
        )
        return

    data = load_npy(cfg.paths.input_data)

    continuous = build_gaussian_observations(
        data=data,
        cfg=cfg,
    )

    sequence_subject_ids = continuous[
        "sequence_subject_ids"
    ]

    n_subjects_expected = int(
        sequence_subject_ids.max()
    ) + 1

    metadata_df = load_subject_metadata(
        cfg,
        n_subjects_expected=n_subjects_expected,
    )
    metadata_records = metadata_df.to_dict(
        orient="records"
    )

    out_root = ensure_dir(
        Path(cfg.paths.hmm_output_root)
        / cfg.gaussian_hmm.output_subdir
        / cfg.gaussian_hmm.sequence_layout
        / cfg.gaussian_hmm.observation_mode
    )

    gaussian_cfg = {
        "sequence_layout": (
            cfg.gaussian_hmm.sequence_layout
        ),
        "observation_mode": (
            cfg.gaussian_hmm.observation_mode
        ),
        "covariance_type": (
            cfg.gaussian_hmm.covariance_type
        ),
        "min_covar": cfg.gaussian_hmm.min_covar,
        "n_iter": cfg.gaussian_hmm.n_iter,
        "tol": cfg.gaussian_hmm.tol,
        "random_state": (
            cfg.gaussian_hmm.random_state
        ),
        "verbose": cfg.gaussian_hmm.verbose,
    }

    temporal_meta = {
        "n_timepoints_input": continuous[
            "n_timepoints_input"
        ],
        "n_timepoints_after_resampling": continuous[
            "n_timepoints_after_resampling"
        ],
        "effective_tr_seconds": (
            cfg.preprocess.target_tr_seconds
            if cfg.preprocess.downsample
            else cfg.preprocess.source_tr_seconds
        ),
        "downsample": cfg.preprocess.downsample,
        "smooth": cfg.preprocess.smooth,
        "smooth_window": (
            cfg.preprocess.smooth_window
        ),
        "standardize_method": (
            cfg.preprocess.standardize_method
        ),
        "center_diff_on_diff_series": (
            cfg.preprocess.center_diff_on_diff_series
        ),
    }

    log_step(f"CPU count detected: {os.cpu_count()}")
    log_step(
        "Gaussian HMM observations prepared: "
        f"shape={continuous['obs'].shape}, "
        f"layout={cfg.gaussian_hmm.sequence_layout}, "
        f"mode={cfg.gaussian_hmm.observation_mode}"
    )
    log_step(f"Output root: {out_root}")
    log_step(f"Using max_workers = {max_workers}")

    if max_workers > 1:
        log_step(
            "Warning: Gaussian observations can be large; "
            "multiple workers may substantially increase memory use."
        )

    tasks = [
        int(k)
        for k in cfg.hmm.n_hidden_states_values
    ]

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = [
            executor.submit(
                run_single_gaussian_hmm_task,
                continuous["obs"],
                continuous["lengths"],
                sequence_subject_ids,
                continuous["sequence_roi_ids"],
                metadata_records,
                str(out_root),
                k,
                gaussian_cfg,
                continuous["feature_names"],
                temporal_meta,
                continuous["n_rois"],
                save_posterior,
            )
            for k in tasks
        ]

        for task_idx, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            result = future.result()
            log_step(
                f"[{task_idx}/{len(futures)}] {result}"
            )

    log_step(
        "All Gaussian HMM runs completed successfully"
    )

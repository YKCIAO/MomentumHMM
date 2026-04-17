from __future__ import annotations

from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import pandas as pd

from src.config import ExperimentConfig
from src.hmm.decode import decode_hmm, split_sequence_by_lengths
from src.hmm.metrics import compute_subject_level_metrics
from src.hmm.model import fit_categorical_hmm
from src.utils.io_utils import ensure_dir, load_npz, save_json, save_npz


def log_step(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def load_subject_metadata(cfg: ExperimentConfig, n_subjects_expected: int) -> pd.DataFrame:
    """
    Load subject metadata from metadata.csv or dataset.npz.

    Priority:
        1. cfg.paths.metadata_csv
        2. cfg.paths.dataset_npz

    The row order must match the subject order used to build the symbolic sequences.
    """
    metadata_csv = getattr(cfg.paths, "metadata_csv", None)
    dataset_npz = getattr(cfg.paths, "dataset_npz", None)

    df = None

    if metadata_csv is not None and Path(metadata_csv).exists():
        log_step(f"Loading subject metadata from CSV: {metadata_csv}")
        df = pd.read_csv(metadata_csv)

    elif dataset_npz is not None and Path(dataset_npz).exists():
        log_step(f"Loading subject metadata from NPZ: {dataset_npz}")
        data = load_npz(dataset_npz)

        required_keys = {"id", "age", "gender"}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise KeyError(f"dataset.npz missing keys: {missing_keys}")

        df = pd.DataFrame({
            "ID": data["id"],
            "Age": data["age"],
            "Gender": data["gender"],
        })

    else:
        raise FileNotFoundError(
            "No subject metadata source found. "
            "Please provide cfg.paths.metadata_csv or cfg.paths.dataset_npz."
        )

    required_cols = ["ID", "Age", "Gender"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Metadata missing required column: {col}")

    df = df.reset_index(drop=True)

    if len(df) != n_subjects_expected:
        raise ValueError(
            f"Metadata row count ({len(df)}) does not match "
            f"number of subjects in lengths ({n_subjects_expected})."
        )

    return df


def run_single_hmm_task(
    symbolic_dir_str: str,
    hmm_root_str: str,
    n_hidden_states: int,
    hmm_cfg: dict,
    metadata_records: list[dict],
    save_posterior: bool = True,
) -> str:
    """
    Worker for one independent HMM task:
        one symbolic_dir + one K
    """
    symbolic_dir = Path(symbolic_dir_str)
    hmm_root = Path(hmm_root_str)
    metadata_df = pd.DataFrame(metadata_records)

    symbolic_data = load_npz(symbolic_dir / "hmm_ready_sequence.npz")
    obs = symbolic_data["obs"]
    lengths = symbolic_data["lengths"]

    model = fit_categorical_hmm(
        obs=obs,
        lengths=lengths,
        n_hidden_states=n_hidden_states,
        n_iter=hmm_cfg["n_iter"],
        tol=hmm_cfg["tol"],
        random_state=hmm_cfg["random_state"],
        verbose=hmm_cfg["verbose"],
    )

    decoded = decode_hmm(model, obs, lengths)

    subject_state_seqs = split_sequence_by_lengths(
        decoded["state_sequence"],
        lengths,
    )

    metrics = compute_subject_level_metrics(
        subject_state_seqs=subject_state_seqs,
        n_hidden_states=n_hidden_states,
    )

    out_dir = ensure_dir(hmm_root / symbolic_dir.name / f"K_{n_hidden_states}")

    # ---------- Save main HMM outputs ----------
    save_dict = {
        "startprob_": model.startprob_,
        "transmat_": model.transmat_,
        "emissionprob_": model.emissionprob_,
        "state_sequence": decoded["state_sequence"],
        "FO": metrics["FO"],
        "MDT": metrics["MDT"],
        "lengths": lengths,
        "subject_ids": metadata_df["ID"].astype(str).to_numpy(dtype=object),
        "subject_age": metadata_df["Age"].to_numpy(),
        "subject_gender": metadata_df["Gender"].astype(str).to_numpy(dtype=object),
    }

    # posterior may be large; save only when needed
    if save_posterior:
        save_dict["posterior"] = decoded["posterior"].astype("float32", copy=False)

    save_npz(out_dir / "hmm_results.npz", **save_dict)

    # ---------- Save subject-level CSV ----------
    subject_rows = []
    n_subjects = len(metadata_df)

    for subj_idx in range(n_subjects):
        row = {
            "subject_index": subj_idx,
            "ID": str(metadata_df.loc[subj_idx, "ID"]),
            "Age": metadata_df.loc[subj_idx, "Age"],
            "Gender": metadata_df.loc[subj_idx, "Gender"],
            "sequence_length": int(lengths[subj_idx]),
        }

        for state_idx in range(n_hidden_states):
            row[f"FO_state_{state_idx}"] = float(metrics["FO"][subj_idx, state_idx])
            row[f"MDT_state_{state_idx}"] = float(metrics["MDT"][subj_idx, state_idx])

        subject_rows.append(row)

    subject_metrics_df = pd.DataFrame(subject_rows)
    subject_metrics_df.to_csv(out_dir / "subject_metrics.csv", index=False)

    # ---------- Save meta ----------
    save_json(
        out_dir / "hmm_meta.json",
        {
            "symbolic_source_dir": str(symbolic_dir),
            "n_hidden_states": n_hidden_states,
            "n_iter": hmm_cfg["n_iter"],
            "tol": hmm_cfg["tol"],
            "random_state": hmm_cfg["random_state"],
            "verbose": hmm_cfg["verbose"],
            "logprob": float(decoded["logprob"]),
            "n_subjects": int(len(lengths)),
            "save_posterior": bool(save_posterior),
            "subject_metrics_csv": str(out_dir / "subject_metrics.csv"),
        },
    )

    return f"Finished {symbolic_dir.name} | K={n_hidden_states}"


def fit_all_hmm_runs_parallel(
    cfg: ExperimentConfig,
    max_workers: int = 3,
    save_posterior: bool = True,
) -> None:
    """
    Parallel version:
        one worker = one symbolic_dir x one K
    """
    symbolic_root = Path(cfg.paths.symbolic_output_root)
    hmm_root = ensure_dir(cfg.paths.hmm_output_root)

    symbolic_dirs = sorted([p for p in symbolic_root.iterdir() if p.is_dir()])
    if len(symbolic_dirs) == 0:
        raise FileNotFoundError(f"No symbolic directories found under: {symbolic_root}")

    log_step(f"CPU count detected: {os.cpu_count()}")
    log_step(f"Found {len(symbolic_dirs)} symbolic runs under: {symbolic_root}")
    log_step(f"Using max_workers = {max_workers}")

    # We need one sample lengths first to verify metadata count.
    first_symbolic_data = load_npz(symbolic_dirs[0] / "hmm_ready_sequence.npz")
    first_lengths = first_symbolic_data["lengths"]

    metadata_df = load_subject_metadata(cfg, n_subjects_expected=len(first_lengths))
    metadata_records = metadata_df.to_dict(orient="records")
    log_step(f"Subject metadata loaded successfully: {len(metadata_df)} subjects")

    hmm_cfg = {
        "n_iter": cfg.hmm.n_iter,
        "tol": cfg.hmm.tol,
        "random_state": cfg.hmm.random_state,
        "verbose": cfg.hmm.verbose,
    }

    tasks = []
    for symbolic_dir in symbolic_dirs:
        for n_hidden_states in cfg.hmm.n_hidden_states_values:
            tasks.append(
                {
                    "symbolic_dir_str": str(symbolic_dir),
                    "hmm_root_str": str(hmm_root),
                    "n_hidden_states": int(n_hidden_states),
                }
            )

    log_step(f"Submitting {len(tasks)} HMM tasks")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in tasks:
            futures.append(
                executor.submit(
                    run_single_hmm_task,
                    task["symbolic_dir_str"],
                    task["hmm_root_str"],
                    task["n_hidden_states"],
                    hmm_cfg,
                    metadata_records,
                    save_posterior,
                )
            )

        for task_idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            log_step(f"[{task_idx}/{len(futures)}] {result}")

    log_step("All parallel HMM runs completed successfully")
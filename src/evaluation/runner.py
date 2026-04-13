from __future__ import annotations

from pathlib import Path

import numpy as np

from config import ExperimentConfig
from evaluation.score import (
    compute_run_metrics,
    minmax_normalize_metric_table,
    weighted_score,
)
from utils.io_utils import ensure_dir, load_npz, save_json


def score_all_hmm_runs(cfg: ExperimentConfig) -> None:
    hmm_root = Path(cfg.paths.hmm_output_root)
    score_root = ensure_dir(cfg.paths.score_output_root)

    run_records = []
    metric_rows = []

    candidate_dirs = sorted(hmm_root.glob("*/*"))  # symbolic_dir / K_x

    for run_dir in candidate_dirs:
        hmm_file = run_dir / "hmm_results.npz"
        meta_file = run_dir / "hmm_meta.json"

        if not hmm_file.exists():
            continue

        hmm_data = load_npz(hmm_file)

        state_sequence = hmm_data["state_sequence"]
        transmat = hmm_data["transmat_"]
        FO = hmm_data["FO"]
        MDT = hmm_data["MDT"]

        # 从 symbolic source 反读 obs
        symbolic_source_dir = run_dir.parent.name
        symbolic_dir = Path(cfg.paths.symbolic_output_root) / symbolic_source_dir
        symbolic_data = load_npz(symbolic_dir / "hmm_ready_sequence.npz")
        obs = symbolic_data["obs"]

        n_hidden_states = transmat.shape[0]

        metrics = compute_run_metrics(
            obs=obs,
            state_sequence=state_sequence,
            transmat=transmat,
            FO=FO,
            MDT=MDT,
            n_hidden_states=n_hidden_states,
            n_categories=9,
        )

        run_info = {
            "run_dir": str(run_dir),
            "symbolic_dir": str(symbolic_dir),
            "n_hidden_states": int(n_hidden_states),
        }

        run_records.append(run_info)
        metric_rows.append(metrics)

    if len(run_records) == 0:
        raise FileNotFoundError("No HMM run results found for scoring.")

    if cfg.score.normalize_scores_across_runs:
        metric_rows_for_scoring = minmax_normalize_metric_table(metric_rows)
    else:
        metric_rows_for_scoring = metric_rows

    scored_runs = []
    for run_info, raw_metrics, norm_metrics in zip(run_records, metric_rows, metric_rows_for_scoring):
        final_score = weighted_score(norm_metrics, cfg.score.weights)

        scored_runs.append(
            {
                **run_info,
                "raw_metrics": raw_metrics,
                "normalized_metrics": norm_metrics,
                "final_score": final_score,
            }
        )

    scored_runs = sorted(scored_runs, key=lambda x: x["final_score"], reverse=True)

    save_json(score_root / "score_ranking.json", {"runs": scored_runs})
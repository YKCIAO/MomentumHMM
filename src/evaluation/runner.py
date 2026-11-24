from __future__ import annotations

from pathlib import Path

from src.config import ExperimentConfig
from src.evaluation.score import (
    compute_run_metrics,
    compute_age_relevance_score,
    minmax_normalize_metric_table,
    weighted_score,
)
from src.utils.io_utils import (
    ensure_dir,
    load_npz,
    save_json,
    save_score_excel,
)


def get_score_output_dir(cfg: ExperimentConfig) -> Path:
    root = Path(cfg.paths.score_output_root)
    if cfg.score.model_family == "gaussian":
        return ensure_dir(root / "gaussian")
    return ensure_dir(root)


def discover_hmm_runs(cfg: ExperimentConfig) -> list[Path]:
    hmm_root = Path(cfg.paths.hmm_output_root)

    if cfg.score.model_family == "gaussian":
        base = (
            hmm_root
            / cfg.gaussian_hmm.output_subdir
            / cfg.gaussian_hmm.sequence_layout
            / cfg.gaussian_hmm.observation_mode
        )
        return sorted(
            p for p in base.glob("K_*")
            if (p / "hmm_results.npz").exists()
        )

    return sorted(
        p for p in hmm_root.glob("*/*")
        if (p / "hmm_results.npz").exists()
        and cfg.gaussian_hmm.output_subdir not in p.parts
    )


def score_all_hmm_runs(cfg: ExperimentConfig) -> None:
    score_root = get_score_output_dir(cfg)

    run_records = []
    metric_rows = []

    candidate_dirs = discover_hmm_runs(cfg)

    for run_dir in candidate_dirs:
        hmm_file = run_dir / "hmm_results.npz"
        hmm_data = load_npz(hmm_file)

        state_sequence = hmm_data["state_sequence"]
        transmat = hmm_data["transmat_"]
        FO = hmm_data["FO"]
        MDT = hmm_data["MDT"]

        n_hidden_states = int(transmat.shape[0])

        if cfg.score.model_family == "categorical":
            symbolic_source_dir = run_dir.parent.name
            symbolic_dir = (
                Path(cfg.paths.symbolic_output_root)
                / symbolic_source_dir
            )
            symbolic_data = load_npz(
                symbolic_dir / "hmm_ready_sequence.npz"
            )
            obs = symbolic_data["obs"]

            parts = symbolic_source_dir.split("__")
            deviation_threshold = float(
                parts[0].replace("dev_", "")
            )
            momentum_threshold = float(
                parts[1].replace("mom_", "")
            )

            run_label = (
                f"{symbolic_source_dir}/K_{n_hidden_states}"
            )
            extra_info = {
                "symbolic_dir": str(symbolic_dir),
                "deviation_threshold": deviation_threshold,
                "momentum_threshold": momentum_threshold,
            }
            n_categories = 9
        else:
            obs = None
            n_categories = None
            run_label = (
                f"Gaussian/"
                f"{cfg.gaussian_hmm.sequence_layout}/"
                f"{cfg.gaussian_hmm.observation_mode}/"
                f"K_{n_hidden_states}"
            )
            extra_info = {
                "sequence_layout":
                    cfg.gaussian_hmm.sequence_layout,
                "observation_mode":
                    cfg.gaussian_hmm.observation_mode,
                "covariance_type":
                    cfg.gaussian_hmm.covariance_type,
                "deviation_threshold": None,
                "momentum_threshold": None,
            }

        metrics = compute_run_metrics(
            obs=obs,
            state_sequence=state_sequence,
            transmat=transmat,
            FO=FO,
            MDT=MDT,
            n_hidden_states=n_hidden_states,
            n_categories=n_categories,
            min_state_fo=cfg.score.min_state_fo,
            short_mdt_threshold=cfg.score.short_mdt_threshold,
        )

        age = hmm_data["subject_age"]
        age_score, age_details = compute_age_relevance_score(
            age=age,
            FO=FO,
            MDT=MDT,
            top_k=cfg.score.age_top_k,
        )

        run_info = {
            "model_family": cfg.score.model_family,
            "run_label": run_label,
            "run_dir": str(run_dir),
            "n_hidden_states": n_hidden_states,
            "age_score": float(age_score),
            "age_details": age_details,
            **extra_info,
        }

        run_records.append(run_info)
        metric_rows.append(metrics)

    if len(run_records) == 0:
        raise FileNotFoundError(
            f"No {cfg.score.model_family} HMM run results found "
            "for scoring."
        )

    if cfg.score.normalize_scores_across_runs:
        metric_rows_for_scoring = (
            minmax_normalize_metric_table(metric_rows)
        )
    else:
        metric_rows_for_scoring = metric_rows

    scored_runs = []

    for run_info, raw_metrics, norm_metrics in zip(
        run_records,
        metric_rows,
        metric_rows_for_scoring,
    ):
        quality_score_base = weighted_score(
            norm_metrics,
            cfg.score.weights,
        )

        effective_fraction = raw_metrics[
            "effective_state_fraction"
        ]

        quality_score = (
            quality_score_base * effective_fraction
        )

        age_score = run_info["age_score"]

        final_score = (
            cfg.score.quality_weight * quality_score
            + cfg.score.age_weight * age_score
        )

        scored_runs.append(
            {
                **run_info,
                "raw_metrics": raw_metrics,
                "normalized_metrics": norm_metrics,
                "quality_score": float(quality_score),
                "age_score": float(age_score),
                "final_score": float(final_score),
            }
        )

    scored_runs = sorted(
        scored_runs,
        key=lambda x: x["final_score"],
        reverse=True,
    )

    save_json(
        score_root / "score_ranking.json",
        {"runs": scored_runs},
    )
    save_score_excel(
        scored_runs=scored_runs,
        output_path=score_root / "score_ranking.xlsx",
    )

from __future__ import annotations

import json
from pathlib import Path

from src.config import ExperimentConfig
from src.evaluation.runner import (
    discover_hmm_runs,
    get_score_output_dir,
)
from src.utils.io_utils import load_npz
from src.visualization.common import (
    prepare_figure_dir,
    save_figure,
)
from src.visualization.hmm_plots import (
    plot_mean_fo,
    plot_mean_mdt,
    plot_transition_matrix,
)
from src.visualization.score_plots import (
    plot_top_score_runs,
)
from src.visualization.symbolic_plots import (
    plot_symbolic_distribution,
    plot_symbolic_timeseries_example,
    plot_symbolic_category_heatmap,
    plot_hmm_ready_sequences,
)


def visualize_all(cfg: ExperimentConfig) -> None:
    fig_root = prepare_figure_dir(
        cfg.paths.figure_output_root
    )

    # Symbolic plots only apply to categorical MomentumHMM.
    if (
        cfg.score.model_family == "categorical"
        and cfg.visualization.save_symbolic_distribution
    ):
        symbolic_root = Path(
            cfg.paths.symbolic_output_root
        )

        symbolic_dirs = sorted(
            p for p in symbolic_root.iterdir()
            if p.is_dir()
        )

        for symbolic_dir in symbolic_dirs:
            symbolic_data = load_npz(
                symbolic_dir / "symbolic_outputs.npz"
            )
            hmm_ready_data = load_npz(
                symbolic_dir / "hmm_ready_sequence.npz"
            )

            fig = plot_symbolic_distribution(
                symbolic_data["category_9"],
                show_titles=cfg.visualization.show_titles,
            )
            save_figure(
                fig,
                fig_root / "symbolic"
                / (
                    f"{symbolic_dir.name}__distribution."
                    f"{cfg.visualization.fig_format}"
                ),
                dpi=cfg.visualization.dpi,
            )

            fig = plot_symbolic_timeseries_example(
                x_std=symbolic_data["x_std"],
                dx_std=symbolic_data["dx_std"],
                deviation_code=symbolic_data["deviation_code"],
                momentum_code=symbolic_data["momentum_code"],
                category_9=symbolic_data["category_9"],
                subject_idx=0,
                roi_idx=0,
                max_timepoints=300,
                show_titles=cfg.visualization.show_titles,
            )
            save_figure(
                fig,
                fig_root / "symbolic"
                / (
                    f"{symbolic_dir.name}__timeseries_example."
                    f"{cfg.visualization.fig_format}"
                ),
                dpi=cfg.visualization.dpi,
            )

            fig = plot_symbolic_category_heatmap(
                category_9=symbolic_data["category_9"],
                subject_idx=0,
                max_rois=30,
                max_timepoints=300,
                show_titles=cfg.visualization.show_titles,
            )
            save_figure(
                fig,
                fig_root / "symbolic"
                / (
                    f"{symbolic_dir.name}__roi_time_heatmap."
                    f"{cfg.visualization.fig_format}"
                ),
                dpi=cfg.visualization.dpi,
            )

            fig = plot_hmm_ready_sequences(
                obs=hmm_ready_data["obs"],
                lengths=hmm_ready_data["lengths"],
                sequence_subject_ids=hmm_ready_data[
                    "sequence_subject_ids"
                ],
                sequence_roi_ids=hmm_ready_data[
                    "sequence_roi_ids"
                ],
                max_sequences=20,
                max_timepoints=300,
                show_titles=cfg.visualization.show_titles,
            )
            save_figure(
                fig,
                fig_root / "symbolic"
                / (
                    f"{symbolic_dir.name}__hmm_ready_sequences."
                    f"{cfg.visualization.fig_format}"
                ),
                dpi=cfg.visualization.dpi,
            )

    # HMM plots for the selected model family.
    for run_dir in discover_hmm_runs(cfg):
        hmm_data = load_npz(
            run_dir / "hmm_results.npz"
        )

        model_tag = (
            "gaussian"
            if cfg.score.model_family == "gaussian"
            else run_dir.parent.name
        )
        base_name = (
            f"{model_tag}__{run_dir.name}"
        )

        if cfg.visualization.save_transition_matrix:
            fig = plot_transition_matrix(
                hmm_data["transmat_"],
                show_titles=cfg.visualization.show_titles,
            )
            save_figure(
                fig,
                fig_root / "hmm"
                / (
                    f"{base_name}__transmat."
                    f"{cfg.visualization.fig_format}"
                ),
                dpi=cfg.visualization.dpi,
            )

        if cfg.visualization.save_fo_bar:
            fig = plot_mean_fo(
                hmm_data["FO"],
                show_titles=cfg.visualization.show_titles,
            )
            save_figure(
                fig,
                fig_root / "hmm"
                / (
                    f"{base_name}__FO."
                    f"{cfg.visualization.fig_format}"
                ),
                dpi=cfg.visualization.dpi,
            )

        if cfg.visualization.save_mdt_bar:
            fig = plot_mean_mdt(
                hmm_data["MDT"],
                show_titles=cfg.visualization.show_titles,
            )
            save_figure(
                fig,
                fig_root / "hmm"
                / (
                    f"{base_name}__MDT."
                    f"{cfg.visualization.fig_format}"
                ),
                dpi=cfg.visualization.dpi,
            )

    if cfg.visualization.save_score_bar:
        score_file = (
            get_score_output_dir(cfg)
            / "score_ranking.json"
        )

        if score_file.exists():
            with open(
                score_file,
                "r",
                encoding="utf-8",
            ) as f:
                scored_runs = json.load(f)["runs"]

            fig = plot_top_score_runs(
                scored_runs=scored_runs,
                top_n=cfg.visualization.top_n_score_runs,
                show_titles=cfg.visualization.show_titles,
            )
            save_figure(
                fig,
                fig_root / "score"
                / (
                    f"{cfg.score.model_family}__top_"
                    f"{cfg.visualization.top_n_score_runs}_runs."
                    f"{cfg.visualization.fig_format}"
                ),
                dpi=cfg.visualization.dpi,
            )

from __future__ import annotations

import json
from pathlib import Path

from src.config import ExperimentConfig
from src.utils.io_utils import ensure_dir, load_npz
from src.visualization.common import prepare_figure_dir, save_figure
from src.visualization.hmm_plots import (
    plot_mean_fo,
    plot_mean_mdt,
    plot_transition_matrix,
)
from src.visualization.score_plots import plot_top_score_runs
from src.visualization.symbolic_plots import (
    plot_symbolic_distribution,
    plot_symbolic_timeseries_example,
    plot_symbolic_category_heatmap,
    plot_hmm_ready_sequences,
)


def visualize_all(cfg: ExperimentConfig) -> None:
    fig_root = prepare_figure_dir(cfg.paths.figure_output_root)

    # 1) symbolic visualizations
    if cfg.visualization.save_symbolic_distribution:

        symbolic_root = Path(
            cfg.paths.symbolic_output_root
        )

        symbolic_dirs = sorted(
            [
                p
                for p in symbolic_root.iterdir()
                if p.is_dir()
            ]
        )

        for symbolic_dir in symbolic_dirs:
            symbolic_data = load_npz(
                symbolic_dir / "symbolic_outputs.npz"
            )

            hmm_ready_data = load_npz(
                symbolic_dir / "hmm_ready_sequence.npz"
            )

            # ---------------------------------
            # 1. Global category distribution
            # ---------------------------------

            fig = plot_symbolic_distribution(
                symbolic_data["category_9"],
                show_titles=cfg.visualization.show_titles,
            )

            save_figure(
                fig,
                fig_root
                / "symbolic"
                / f"{symbolic_dir.name}__distribution."
                  f"{cfg.visualization.fig_format}",
                dpi=cfg.visualization.dpi,
            )

            # ---------------------------------
            # 2. Single subject-ROI example
            # ---------------------------------

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
                fig_root
                / "symbolic"
                / f"{symbolic_dir.name}__timeseries_example."
                  f"{cfg.visualization.fig_format}",
                dpi=cfg.visualization.dpi,
            )

            # ---------------------------------
            # 3. ROI × time symbolic heatmap
            # ---------------------------------

            fig = plot_symbolic_category_heatmap(
                category_9=symbolic_data["category_9"],
                subject_idx=0,
                max_rois=30,
                max_timepoints=300,
                show_titles=cfg.visualization.show_titles,
            )

            save_figure(
                fig,
                fig_root
                / "symbolic"
                / f"{symbolic_dir.name}__roi_time_heatmap."
                  f"{cfg.visualization.fig_format}",
                dpi=cfg.visualization.dpi,
            )

            # ---------------------------------
            # 4. HMM-ready sequence heatmap
            # ---------------------------------

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
                fig_root
                / "symbolic"
                / f"{symbolic_dir.name}__hmm_ready_sequences."
                  f"{cfg.visualization.fig_format}",
                dpi=cfg.visualization.dpi,
            )

    # 2) HMM plots
    hmm_root = Path(cfg.paths.hmm_output_root)
    hmm_candidate_dirs = sorted(hmm_root.glob("*/*"))

    for run_dir in hmm_candidate_dirs:
        hmm_file = run_dir / "hmm_results.npz"
        if not hmm_file.exists():
            continue

        hmm_data = load_npz(hmm_file)

        if cfg.visualization.save_transition_matrix:
            fig = plot_transition_matrix(
                hmm_data["transmat_"],
                show_titles=cfg.visualization.show_titles,
            )
            save_figure(
                fig,
                fig_root / "hmm" / f"{run_dir.parent.name}__{run_dir.name}__transmat.{cfg.visualization.fig_format}",
                dpi=cfg.visualization.dpi,
            )

        if cfg.visualization.save_fo_bar:
            fig = plot_mean_fo(
                hmm_data["FO"],
                show_titles=cfg.visualization.show_titles,
            )
            save_figure(
                fig,
                fig_root / "hmm" / f"{run_dir.parent.name}__{run_dir.name}__FO.{cfg.visualization.fig_format}",
                dpi=cfg.visualization.dpi,
            )

        if cfg.visualization.save_mdt_bar:
            fig = plot_mean_mdt(
                hmm_data["MDT"],
                show_titles=cfg.visualization.show_titles,
            )
            save_figure(
                fig,
                fig_root / "hmm" / f"{run_dir.parent.name}__{run_dir.name}__MDT.{cfg.visualization.fig_format}",
                dpi=cfg.visualization.dpi,
            )

    # 3) score plot
    if cfg.visualization.save_score_bar:
        score_file = Path(cfg.paths.score_output_root) / "score_ranking.json"
        if score_file.exists():
            with open(score_file, "r", encoding="utf-8") as f:
                score_json = json.load(f)

            scored_runs = score_json["runs"]
            fig = plot_top_score_runs(
                scored_runs=scored_runs,
                top_n=cfg.visualization.top_n_score_runs,
                show_titles=cfg.visualization.show_titles,
            )
            save_figure(
                fig,
                fig_root / "score" / f"top_{cfg.visualization.top_n_score_runs}_runs.{cfg.visualization.fig_format}",
                dpi=cfg.visualization.dpi,
            )
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
from src.visualization.symbolic_plots import plot_symbolic_distribution


def visualize_all(cfg: ExperimentConfig) -> None:
    fig_root = prepare_figure_dir(cfg.paths.figure_output_root)

    # 1) symbolic distributions
    if cfg.visualization.save_symbolic_distribution:
        symbolic_root = Path(cfg.paths.symbolic_output_root)
        for symbolic_dir in sorted([p for p in symbolic_root.iterdir() if p.is_dir()]):
            symbolic_data = load_npz(symbolic_dir / "symbolic_outputs.npz")
            fig = plot_symbolic_distribution(
                symbolic_data["category_9"],
                show_titles=cfg.visualization.show_titles,
            )
            save_figure(
                fig,
                fig_root / "symbolic" / f"{symbolic_dir.name}_symbolic_dist.{cfg.visualization.fig_format}",
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
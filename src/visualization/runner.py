from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.visualization.common import ensure_dir, set_academic_style, load_json
from src.visualization.hmm_plots import (
    plot_state_means_2d,
    plot_fo_vs_age,
    plot_mdt_vs_age,
    plot_global_dynamics_vs_age,
    plot_transition_matrices_by_age,
    plot_transition_graph_by_age,
    plot_visits_vs_age,
    plot_parameter_robustness_heatmap,
    compute_run_age_effect_summary,
)


def _extract_k_from_dir(k_dir: Path) -> int:
    match = re.search(r"K_(\d+)", k_dir.name)
    if match:
        return int(match.group(1))
    return -1


def _find_hmm_result_dirs(hmm_root: Path) -> list[Path]:
    """
    Find all directories containing hmm_results.npz and subject_metrics.csv.
    """
    result_dirs = []
    for p in hmm_root.rglob("*"):
        if not p.is_dir():
            continue
        if (p / "hmm_results.npz").exists() and (p / "subject_metrics.csv").exists():
            result_dirs.append(p)
    return sorted(result_dirs)


def _safe_run_name(result_dir: Path, hmm_root: Path) -> str:
    rel = result_dir.relative_to(hmm_root)
    return "__".join(rel.parts)


def visualize_single_hmm_result(
    result_dir: Path,
    fig_root: Path,
    cfg: ExperimentConfig,
    run_name: str,
) -> dict:
    """
    Generate all Sprint 3 figures for one HMM result folder.
    """
    dpi = cfg.visualization.dpi
    fig_format = cfg.visualization.fig_format

    hmm_path = result_dir / "hmm_results.npz"
    subject_csv = result_dir / "subject_metrics.csv"

    hmm_npz = np.load(hmm_path, allow_pickle=True)
    k_value = _extract_k_from_dir(result_dir)

    out_dir = ensure_dir(fig_root / run_name)

    title_prefix = run_name.replace("__", " / ")

    # Figure 1
    plot_state_means_2d(
        hmm_npz=hmm_npz,
        out_path=out_dir / f"fig01_state_means_2d.{fig_format}",
        title=f"State Means in Activation-Trend Space\n{title_prefix}",
        dpi=dpi,
    )

    # Figure 2
    plot_fo_vs_age(
        subject_metrics_csv=subject_csv,
        out_path=out_dir / f"fig02_fo_vs_age.{fig_format}",
        title=f"Fractional Occupancy vs Age\n{title_prefix}",
        dpi=dpi,
    )

    # Figure 3
    plot_mdt_vs_age(
        subject_metrics_csv=subject_csv,
        out_path=out_dir / f"fig03_mdt_vs_age.{fig_format}",
        title=f"Mean Dwell Time vs Age\n{title_prefix}",
        dpi=dpi,
    )

    # Figure 4
    plot_global_dynamics_vs_age(
        subject_metrics_csv=subject_csv,
        out_path=out_dir / f"fig04_global_dynamics_vs_age.{fig_format}",
        title=f"Global HMM Dynamics vs Age\n{title_prefix}",
        dpi=dpi,
    )

    # Figure 5
    plot_transition_matrices_by_age(
        hmm_npz=hmm_npz,
        out_path=out_dir / f"fig05_transition_matrices_by_age.{fig_format}",
        title=f"Transition Matrices by Age Group\n{title_prefix}",
        dpi=dpi,
    )

    # Figure 6
    plot_transition_graph_by_age(
        hmm_npz=hmm_npz,
        out_path=out_dir / f"fig06_transition_graph_by_age.{fig_format}",
        title=f"Transition Graph by Age Group\n{title_prefix}",
        dpi=dpi,
    )

    # Figure 7
    plot_visits_vs_age(
        subject_metrics_csv=subject_csv,
        out_path=out_dir / f"fig07_visits_vs_age.{fig_format}",
        title=f"Visit Count vs Age\n{title_prefix}",
        dpi=dpi,
    )

    # Compact age-effect summary for robustness figure
    age_summary = compute_run_age_effect_summary(subject_csv)
    age_summary.update({
        "run_name": result_dir.parent.name,
        "K": k_value,
        "result_dir": str(result_dir),
    })

    return age_summary


def visualize_all(cfg: ExperimentConfig) -> None:
    """
    Sprint 3 visualization entry.

    Generates 8 academic-style figures:
        1. State means in 2D activation-trend space
        2. FO vs Age
        3. MDT vs Age
        4. SwitchingRate / StateEntropy / NTransitions vs Age
        5. Transition matrices for younger vs older subjects
        6. Transition graph for younger vs older subjects
        7. Visit count vs Age
        8. Parameter robustness heatmap across runs

    Each HMM result folder gets Figures 1-7.
    Figure 8 is generated once at the root level.
    """
    set_academic_style(font_size=16)

    hmm_root = Path(cfg.paths.hmm_output_root)
    fig_root = ensure_dir(cfg.paths.figure_output_root)

    result_dirs = _find_hmm_result_dirs(hmm_root)
    if len(result_dirs) == 0:
        raise FileNotFoundError(f"No HMM result folders found under: {hmm_root}")

    summaries = []

    for result_dir in result_dirs:
        run_name = _safe_run_name(result_dir, hmm_root)
        print(f"[Visualization] Processing: {run_name}", flush=True)

        try:
            summary = visualize_single_hmm_result(
                result_dir=result_dir,
                fig_root=fig_root,
                cfg=cfg,
                run_name=run_name,
            )
            summaries.append(summary)

        except Exception as e:
            print(f"[Visualization] FAILED: {run_name} | {repr(e)}", flush=True)

    summary_df = pd.DataFrame(summaries)
    summary_csv = fig_root / "age_effect_robustness_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    plot_parameter_robustness_heatmap(
        summary_df=summary_df,
        out_path=fig_root / f"fig08_parameter_robustness_heatmap.{cfg.visualization.fig_format}",
        title="Parameter Robustness of Age-Related HMM Effects",
        dpi=cfg.visualization.dpi,
    )

    print(f"[Visualization] All figures saved to: {fig_root}", flush=True)
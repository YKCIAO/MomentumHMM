from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import ExperimentConfig
from src.visualization.common import ensure_dir, save_figure, set_academic_style


# ---------------------------------------------------------------------
# Score Figure 1: Top-ranked model solutions
# ---------------------------------------------------------------------
# Scientific purpose:
#     This figure shows which HMM solutions are most defensible according
#     to the composite model-selection score. It helps avoid selecting a
#     model only because it has one strong but potentially unstable age effect.
# ---------------------------------------------------------------------
def plot_top_model_scores(
    ranking: pd.DataFrame,
    out_path: str | Path,
    top_n: int = 20,
    dpi: int = 300,
) -> None:
    df = ranking.head(top_n).copy()
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * len(df))))

    ax.barh(df["run_name"], df["final_score"])
    ax.set_xlabel("Composite Score")
    ax.set_ylabel("HMM Run")
    ax.set_title(f"Top {top_n} HMM Solutions")

    for i, val in enumerate(df["final_score"]):
        ax.text(
            val,
            i,
            f" {val:.3f}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------
# Score Figure 2: Age effect vs fragmentation
# ---------------------------------------------------------------------
# Scientific purpose:
#     This figure tests whether strong age effects are accompanied by
#     stable state solutions or whether they arise from fragmented models.
#     The ideal model has high age effect and low fragmentation.
# ---------------------------------------------------------------------
def plot_age_effect_vs_fragmentation(
    ranking: pd.DataFrame,
    out_path: str | Path,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    x = ranking["fragmentation_score"].to_numpy(dtype=float)
    y = ranking["max_abs_age_r"].to_numpy(dtype=float)

    sizes = 80 + 500 * ranking["state_usage_balance"].fillna(0).to_numpy(dtype=float)

    sc = ax.scatter(
        x,
        y,
        s=sizes,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.8,
    )

    best = ranking.iloc[0]
    ax.scatter(
        best["fragmentation_score"],
        best["max_abs_age_r"],
        s=350,
        marker="*",
        edgecolor="black",
        linewidth=1.5,
        zorder=5,
    )

    ax.text(
        best["fragmentation_score"],
        best["max_abs_age_r"],
        " Best",
        fontsize=13,
        fontweight="bold",
        va="center",
    )

    ax.set_xlabel("Fragmentation Score\n(lower is better)")
    ax.set_ylabel("Maximum |r(age)|")
    ax.set_title("Age Effect vs State Fragmentation")

    ax.grid(alpha=0.25)
    save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------
# Score Figure 3: Model score by number of states
# ---------------------------------------------------------------------
# Scientific purpose:
#     This figure evaluates whether increasing the number of hidden states
#     improves model quality or mainly increases fragmentation.
# ---------------------------------------------------------------------
def plot_score_by_k(
    ranking: pd.DataFrame,
    out_path: str | Path,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    ks = sorted(ranking["K"].dropna().unique())
    data = [ranking.loc[ranking["K"] == k, "final_score"].dropna().to_numpy() for k in ks]

    ax.boxplot(data, labels=[str(k) for k in ks], showmeans=True)

    ax.set_xlabel("Number of Hidden States (K)")
    ax.set_ylabel("Composite Score")
    ax.set_title("Model Score by K")

    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------
# Score Figure 4: Threshold robustness summary
# ---------------------------------------------------------------------
# Scientific purpose:
#     This figure shows how model score changes across activation and trend
#     thresholds. A robust analysis should not depend on one arbitrary
#     threshold combination.
# ---------------------------------------------------------------------
def plot_threshold_score_heatmap(
    ranking: pd.DataFrame,
    out_path: str | Path,
    dpi: int = 300,
) -> None:
    df = ranking.copy()

    if "activation_threshold" not in df.columns or "trend_threshold" not in df.columns:
        return

    df = df.dropna(subset=["activation_threshold", "trend_threshold", "final_score"])
    if df.empty:
        return

    pivot = df.pivot_table(
        index="activation_threshold",
        columns="trend_threshold",
        values="final_score",
        aggfunc="max",
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(pivot.to_numpy(), aspect="auto")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], fontweight="bold")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"{x:.2f}" for x in pivot.index], fontweight="bold")

    ax.set_xlabel("Trend Threshold")
    ax.set_ylabel("Activation Threshold")
    ax.set_title("Best Composite Score Across Thresholds")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.to_numpy()[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white" if val > np.nanmax(pivot.to_numpy()) * 0.6 else "black",
                )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, out_path, dpi=dpi)


def plot_score_outputs(cfg: ExperimentConfig, ranking: pd.DataFrame) -> None:
    set_academic_style(font_size=16)

    out_dir = ensure_dir(Path(cfg.paths.figure_output_root) / "model_selection")
    dpi = cfg.visualization.dpi
    fmt = cfg.visualization.fig_format
    top_n = cfg.visualization.top_n_score_runs

    plot_top_model_scores(
        ranking=ranking,
        out_path=out_dir / f"score_fig01_top_model_scores.{fmt}",
        top_n=top_n,
        dpi=dpi,
    )

    plot_age_effect_vs_fragmentation(
        ranking=ranking,
        out_path=out_dir / f"score_fig02_age_effect_vs_fragmentation.{fmt}",
        dpi=dpi,
    )

    plot_score_by_k(
        ranking=ranking,
        out_path=out_dir / f"score_fig03_score_by_k.{fmt}",
        dpi=dpi,
    )

    plot_threshold_score_heatmap(
        ranking=ranking,
        out_path=out_dir / f"score_fig04_threshold_score_heatmap.{fmt}",
        dpi=dpi,
    )
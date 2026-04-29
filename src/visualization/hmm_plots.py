from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

from src.visualization.common import (
    ensure_dir,
    save_figure,
    add_regression_line,
    annotate_r_p,
    pearson_r_p,
)


def _state_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols = sorted(cols, key=lambda x: int(x.split("_")[-1]))
    return cols


def _get_state_ids_from_columns(cols: list[str]) -> list[int]:
    return [int(c.split("_")[-1]) for c in cols]


def _grid_shape(n: int) -> tuple[int, int]:
    n_cols = min(4, max(1, n))
    n_rows = int(math.ceil(n / n_cols))
    return n_rows, n_cols


def _covariance_ellipse(mean, cov, ax, n_std: float = 2.0):
    """
    Draw covariance ellipse for a 2D Gaussian state.
    """
    if cov.shape != (2, 2):
        return

    cov = np.asarray(cov, dtype=float)

    # numerical safety
    cov = (cov + cov.T) / 2.0

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return

    eigvals = np.maximum(eigvals, 1e-12)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=2.0,
        alpha=0.8,
    )
    ax.add_patch(ellipse)


# ---------------------------------------------------------------------
# Figure 1: State means in 2D activation-trend space
# ---------------------------------------------------------------------
# Scientific purpose:
#     This figure explains what each hidden state represents.
#     In the 2.0 Gaussian HMM, each state has a mean activation value
#     and a mean trend value. Plotting these means in 2D space makes the
#     latent states interpretable, e.g., high activation + upward trend,
#     neutral activation + stable trend, or negative activation + recovery.
# ---------------------------------------------------------------------
def plot_state_means_2d(
    hmm_npz,
    out_path: str | Path,
    title: str,
    dpi: int = 300,
) -> None:
    means = hmm_npz["means_"]
    covars = hmm_npz["covars_"] if "covars_" in hmm_npz.files else None
    fo = hmm_npz["FO"] if "FO" in hmm_npz.files else None

    if means.shape[1] < 2:
        # activation-only or trend-only model cannot be shown in 2D
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(
            0.5,
            0.5,
            "2D state plot skipped:\nmodel has only one feature channel",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )
        ax.axis("off")
        save_figure(fig, out_path, dpi=dpi)
        return

    x = means[:, 0]
    y = means[:, 1]
    K = means.shape[0]

    if fo is not None:
        mean_fo = fo.mean(axis=0)
        sizes = 300 + 3000 * mean_fo / max(mean_fo.max(), 1e-12)
    else:
        sizes = np.full(K, 700)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(x, y, s=sizes, alpha=0.85, edgecolor="black", linewidth=1.2)

    if covars is not None and means.shape[1] == 2:
        for k in range(K):
            cov = covars[k]
            if cov.ndim == 1:
                cov = np.diag(cov)
            _covariance_ellipse(means[k], cov, ax)

    for k in range(K):
        ax.text(
            x[k],
            y[k],
            f"S{k}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    ax.axhline(0, linewidth=1.5, linestyle="--", alpha=0.5)
    ax.axvline(0, linewidth=1.5, linestyle="--", alpha=0.5)

    ax.set_xlabel("Activation")
    ax.set_ylabel("Trend")
    ax.set_title(title)

    ax.grid(alpha=0.25)
    save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------
# Figure 2: Fractional Occupancy vs Age
# ---------------------------------------------------------------------
# Scientific purpose:
#     FO indicates how much time each subject spends in each hidden state.
#     A state with FO increasing with age suggests that older subjects
#     spend more time in that brain dynamic regime.
# ---------------------------------------------------------------------
def plot_fo_vs_age(
    subject_metrics_csv: str | Path,
    out_path: str | Path,
    title: str,
    dpi: int = 300,
) -> None:
    df = pd.read_csv(subject_metrics_csv)
    age = df["Age"].to_numpy(dtype=float)

    cols = _state_columns(df, "FO_state_")
    K = len(cols)
    n_rows, n_cols = _grid_shape(K)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )

    for idx, col in enumerate(cols):
        ax = axes[idx // n_cols][idx % n_cols]
        y = df[col].to_numpy(dtype=float)

        ax.scatter(age, y, s=28, alpha=0.65)
        add_regression_line(ax, age, y)
        annotate_r_p(ax, age, y)

        state_id = col.split("_")[-1]
        ax.set_title(f"State {state_id}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Fractional Occupancy")

    for idx in range(K, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(title, fontsize=20, fontweight="bold")
    save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------
# Figure 3: Mean Dwell Time vs Age
# ---------------------------------------------------------------------
# Scientific purpose:
#     MDT reflects how long a state persists once it is entered.
#     If MDT increases with age, that state may become more stable or
#     attractor-like in older subjects. If MDT decreases, the state becomes
#     more transient with aging.
# ---------------------------------------------------------------------
def plot_mdt_vs_age(
    subject_metrics_csv: str | Path,
    out_path: str | Path,
    title: str,
    dpi: int = 300,
) -> None:
    df = pd.read_csv(subject_metrics_csv)
    age = df["Age"].to_numpy(dtype=float)

    cols = _state_columns(df, "MDT_state_")
    K = len(cols)
    n_rows, n_cols = _grid_shape(K)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )

    for idx, col in enumerate(cols):
        ax = axes[idx // n_cols][idx % n_cols]
        y = df[col].to_numpy(dtype=float)

        ax.scatter(age, y, s=28, alpha=0.65)
        add_regression_line(ax, age, y)
        annotate_r_p(ax, age, y)

        state_id = col.split("_")[-1]
        ax.set_title(f"State {state_id}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Mean Dwell Time")

    for idx in range(K, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(title, fontsize=20, fontweight="bold")
    save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------
# Figure 4: Global dynamics vs Age
# ---------------------------------------------------------------------
# Scientific purpose:
#     Switching rate and state entropy summarize whole-brain temporal dynamics.
#     SwitchingRate tests whether aging is associated with more frequent or
#     less frequent state changes. StateEntropy tests whether aging makes the
#     state repertoire more diverse or more restricted.
# ---------------------------------------------------------------------
def plot_global_dynamics_vs_age(
    subject_metrics_csv: str | Path,
    out_path: str | Path,
    title: str,
    dpi: int = 300,
) -> None:
    df = pd.read_csv(subject_metrics_csv)
    age = df["Age"].to_numpy(dtype=float)

    metrics = []
    if "SwitchingRate" in df.columns:
        metrics.append("SwitchingRate")
    if "StateEntropy" in df.columns:
        metrics.append("StateEntropy")
    if "NTransitions" in df.columns:
        metrics.append("NTransitions")

    if not metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No global dynamic metrics found", ha="center", va="center")
        ax.axis("off")
        save_figure(fig, out_path, dpi=dpi)
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.6))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        y = df[metric].to_numpy(dtype=float)
        ax.scatter(age, y, s=30, alpha=0.65)
        add_regression_line(ax, age, y)
        annotate_r_p(ax, age, y)

        ax.set_title(metric)
        ax.set_xlabel("Age")
        ax.set_ylabel(metric)

    fig.suptitle(title, fontsize=20, fontweight="bold")
    save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------
# Figure 5: Transition matrices by age group
# ---------------------------------------------------------------------
# Scientific purpose:
#     Transition matrices reveal how subjects move between states.
#     Comparing young and old groups shows whether aging changes transition
#     preferences, not only state occupancy or duration.
# ---------------------------------------------------------------------
def plot_transition_matrices_by_age(
    hmm_npz,
    out_path: str | Path,
    title: str,
    dpi: int = 300,
) -> None:
    age = hmm_npz["subject_age"].astype(float)
    trans = hmm_npz["TransitionProbs"]

    median_age = np.nanmedian(age)
    young_mask = age <= median_age
    old_mask = age > median_age

    young_mat = np.nanmean(trans[young_mask], axis=0)
    old_mat = np.nanmean(trans[old_mask], axis=0)
    diff_mat = old_mat - young_mat

    vmax = max(np.nanmax(young_mat), np.nanmax(old_mat))
    absmax = np.nanmax(np.abs(diff_mat))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    im0 = axes[0].imshow(young_mat, vmin=0, vmax=vmax)
    axes[0].set_title(f"Younger group\nAge ≤ {median_age:.1f}")
    axes[0].set_xlabel("To state")
    axes[0].set_ylabel("From state")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(old_mat, vmin=0, vmax=vmax)
    axes[1].set_title(f"Older group\nAge > {median_age:.1f}")
    axes[1].set_xlabel("To state")
    axes[1].set_ylabel("From state")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff_mat, vmin=-absmax, vmax=absmax)
    axes[2].set_title("Older - Younger")
    axes[2].set_xlabel("To state")
    axes[2].set_ylabel("From state")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    K = young_mat.shape[0]
    for ax in axes:
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))

    fig.suptitle(title, fontsize=20, fontweight="bold")
    save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------
# Figure 6: Transition graph
# ---------------------------------------------------------------------
# Scientific purpose:
#     This graph visualizes transition organization as a network.
#     Nodes are hidden states and directed edges are transition probabilities.
#     It provides a more intuitive depiction of the dynamic flow between states.
# ---------------------------------------------------------------------
def plot_transition_graph_by_age(
    hmm_npz,
    out_path: str | Path,
    title: str,
    dpi: int = 300,
    edge_threshold: float = 0.05,
) -> None:
    age = hmm_npz["subject_age"].astype(float)
    trans = hmm_npz["TransitionProbs"]

    median_age = np.nanmedian(age)
    young_mat = np.nanmean(trans[age <= median_age], axis=0)
    old_mat = np.nanmean(trans[age > median_age], axis=0)

    matrices = [young_mat, old_mat]
    subtitles = [f"Younger\nAge ≤ {median_age:.1f}", f"Older\nAge > {median_age:.1f}"]

    K = young_mat.shape[0]
    theta = np.linspace(0, 2 * np.pi, K, endpoint=False)
    pos = np.column_stack([np.cos(theta), np.sin(theta)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, mat, subtitle in zip(axes, matrices, subtitles):
        ax.scatter(pos[:, 0], pos[:, 1], s=900, edgecolor="black", linewidth=1.5, zorder=3)

        for k in range(K):
            ax.text(
                pos[k, 0],
                pos[k, 1],
                f"S{k}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                zorder=4,
            )

        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                p = mat[i, j]
                if p < edge_threshold:
                    continue

                start = pos[i]
                end = pos[j]
                arrow = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=1.0 + 8.0 * p,
                    alpha=0.45,
                    shrinkA=22,
                    shrinkB=22,
                    connectionstyle="arc3,rad=0.15",
                )
                ax.add_patch(arrow)

        ax.set_title(subtitle)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(title, fontsize=20, fontweight="bold")
    save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------
# Figure 7: Visits vs Age
# ---------------------------------------------------------------------
# Scientific purpose:
#     Visit count reflects how often a state is entered.
#     It complements FO and MDT by distinguishing frequent short visits from
#     fewer but longer visits. This helps clarify whether aging affects state
#     entry frequency or persistence.
# ---------------------------------------------------------------------
def plot_visits_vs_age(
    subject_metrics_csv: str | Path,
    out_path: str | Path,
    title: str,
    dpi: int = 300,
) -> None:
    df = pd.read_csv(subject_metrics_csv)
    age = df["Age"].to_numpy(dtype=float)

    cols = _state_columns(df, "Visits_state_")
    K = len(cols)
    n_rows, n_cols = _grid_shape(K)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )

    for idx, col in enumerate(cols):
        ax = axes[idx // n_cols][idx % n_cols]
        y = df[col].to_numpy(dtype=float)

        ax.scatter(age, y, s=28, alpha=0.65)
        add_regression_line(ax, age, y)
        annotate_r_p(ax, age, y)

        state_id = col.split("_")[-1]
        ax.set_title(f"State {state_id}")
        ax.set_xlabel("Age")
        ax.set_ylabel("Visit Count")

    for idx in range(K, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(title, fontsize=20, fontweight="bold")
    save_figure(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------
# Figure 8: Robustness heatmap across threshold/K runs
# ---------------------------------------------------------------------
# Scientific purpose:
#     This figure evaluates whether age-related effects are stable across
#     parameter settings. A robust result should not depend on a single
#     arbitrary threshold. The heatmap shows the maximum absolute age
#     correlation across FO/MDT/switching metrics for each run.
# ---------------------------------------------------------------------
def plot_parameter_robustness_heatmap(
    summary_df: pd.DataFrame,
    out_path: str | Path,
    title: str,
    dpi: int = 300,
) -> None:
    if summary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No robustness summary available", ha="center", va="center")
        ax.axis("off")
        save_figure(fig, out_path, dpi=dpi)
        return

    # combine run name and K as row label
    summary_df = summary_df.copy()
    summary_df["label"] = summary_df["run_name"] + " | K=" + summary_df["K"].astype(str)

    values = summary_df["max_abs_age_r"].to_numpy(dtype=float)[:, None]

    fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * len(summary_df))))

    im = ax.imshow(values, aspect="auto")
    ax.set_yticks(np.arange(len(summary_df)))
    ax.set_yticklabels(summary_df["label"], fontsize=10, fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels(["max |r(age)|"], fontweight="bold")

    for i, val in enumerate(summary_df["max_abs_age_r"].to_numpy(dtype=float)):
        ax.text(
            0,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white" if val > np.nanmax(values) * 0.6 else "black",
        )

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(title)
    save_figure(fig, out_path, dpi=dpi)


def compute_run_age_effect_summary(subject_metrics_csv: str | Path) -> dict:
    """
    Compute a compact age-effect summary for robustness visualization.
    """
    df = pd.read_csv(subject_metrics_csv)
    age = df["Age"].to_numpy(dtype=float)

    candidate_cols = []
    for prefix in ["FO_state_", "MDT_state_", "Visits_state_"]:
        candidate_cols.extend(_state_columns(df, prefix))

    for c in ["SwitchingRate", "StateEntropy", "NTransitions"]:
        if c in df.columns:
            candidate_cols.append(c)

    best_col = None
    best_r = np.nan
    best_abs = -np.inf

    for col in candidate_cols:
        y = df[col].to_numpy(dtype=float)
        r, p = pearson_r_p(age, y)
        if np.isfinite(r) and abs(r) > best_abs:
            best_abs = abs(r)
            best_r = r
            best_col = col

    if best_abs == -np.inf:
        best_abs = np.nan

    return {
        "best_metric": best_col,
        "best_age_r": best_r,
        "max_abs_age_r": best_abs,
    }
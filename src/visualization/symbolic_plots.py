from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


CATEGORY_LABELS = [
    "(-1,-1)",
    "(-1,0)",
    "(-1,+1)",
    "(0,-1)",
    "(0,0)",
    "(0,+1)",
    "(+1,-1)",
    "(+1,0)",
    "(+1,+1)",
]


def plot_symbolic_distribution(
    category_9: np.ndarray,
    show_titles: bool,
):
    """
    category_9: [subjects, rois, time], values in {0..8}
    """
    values = category_9.reshape(-1)
    counts = np.bincount(values, minlength=9)
    probs = counts / counts.sum() if counts.sum() > 0 else counts

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(np.arange(9), probs)

    ax.set_xlabel("Observation category")
    ax.set_ylabel("Proportion")

    ax.set_xticks(np.arange(9))
    ax.set_xticklabels(CATEGORY_LABELS)

    if show_titles:
        ax.set_title(
            "Distribution of deviation-momentum symbolic observations"
        )

    return fig


def plot_symbolic_timeseries_example(
    x_std: np.ndarray,
    dx_std: np.ndarray,
    deviation_code: np.ndarray,
    momentum_code: np.ndarray,
    category_9: np.ndarray,
    subject_idx: int = 0,
    roi_idx: int = 0,
    max_timepoints: int | None = 300,
    show_titles: bool = True,
):
    """
    Visualize the full symbolic transformation for one subject-ROI sequence.

    Arrays have shape:
        [subjects, rois, time]
    """

    n_time = x_std.shape[-1]

    if max_timepoints is None:
        end = n_time
    else:
        end = min(n_time, max_timepoints)

    t = np.arange(end)

    x = x_std[subject_idx, roi_idx, :end]
    dx = dx_std[subject_idx, roi_idx, :end]

    dev = deviation_code[subject_idx, roi_idx, :end]
    mom = momentum_code[subject_idx, roi_idx, :end]

    cat = category_9[subject_idx, roi_idx, :end]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 9),
        sharex=True,
    )

    # standardized signal
    axes[0].plot(t, x)
    axes[0].axhline(0, linewidth=0.8)
    axes[0].set_ylabel("x")
    axes[0].set_title("Standardized ROI signal")

    # standardized derivative
    axes[1].plot(t, dx)
    axes[1].axhline(0, linewidth=0.8)
    axes[1].set_ylabel("Δx")
    axes[1].set_title("Standardized temporal derivative")

    # ternary codes
    axes[2].step(
        t,
        dev,
        where="mid",
        label="Deviation",
    )

    axes[2].step(
        t,
        mom,
        where="mid",
        label="Momentum",
    )

    axes[2].set_yticks([-1, 0, 1])
    axes[2].set_ylabel("Code")
    axes[2].legend()
    axes[2].set_title("Ternary symbolic components")

    # final nine-category observations
    axes[3].step(
        t,
        cat,
        where="mid",
    )

    axes[3].set_yticks(np.arange(9))
    axes[3].set_yticklabels(CATEGORY_LABELS)

    axes[3].set_xlabel("Time point")
    axes[3].set_ylabel("Category")
    axes[3].set_title("Final 9-category observation sequence")

    if show_titles:
        fig.suptitle(
            f"Symbolic encoding | Subject {subject_idx} | ROI {roi_idx}",
            y=1.02,
        )

    return fig


def plot_symbolic_category_heatmap(
    category_9: np.ndarray,
    subject_idx: int = 0,
    max_rois: int | None = 30,
    max_timepoints: int | None = 300,
    show_titles: bool = True,
):
    """
    Plot ROI × time symbolic categories for one subject.
    """

    subject_data = category_9[subject_idx]

    n_rois, n_time = subject_data.shape

    if max_rois is not None:
        n_rois = min(n_rois, max_rois)

    if max_timepoints is not None:
        n_time = min(n_time, max_timepoints)

    data = subject_data[:n_rois, :n_time]

    fig, ax = plt.subplots(
        figsize=(14, max(5, n_rois * 0.22))
    )

    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=8,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        ticks=np.arange(9),
    )

    cbar.ax.set_yticklabels(CATEGORY_LABELS)
    cbar.set_label("Symbolic category")

    ax.set_xlabel("Time point")
    ax.set_ylabel("ROI")

    if show_titles:
        ax.set_title(
            f"Symbolic observations across ROIs | Subject {subject_idx}"
        )

    return fig


def plot_hmm_ready_sequences(
    obs: np.ndarray,
    lengths: np.ndarray,
    sequence_subject_ids: np.ndarray,
    sequence_roi_ids: np.ndarray,
    max_sequences: int = 20,
    max_timepoints: int | None = 300,
    show_titles: bool = True,
):
    """
    Visualize individual HMM-ready subject-ROI sequences.

    Each row corresponds to one independent sequence.
    """

    obs = obs.reshape(-1)

    n_sequences = len(lengths)
    n_plot = min(n_sequences, max_sequences)

    sequence_arrays = []

    start = 0

    for seq_idx, length in enumerate(lengths):

        end = start + int(length)

        if seq_idx < n_plot:

            seq = obs[start:end]

            if max_timepoints is not None:
                seq = seq[:max_timepoints]

            sequence_arrays.append(seq)

        start = end

    max_len = max(len(x) for x in sequence_arrays)

    matrix = np.full(
        (n_plot, max_len),
        np.nan,
        dtype=np.float64,
    )

    labels = []

    for seq_idx, seq in enumerate(sequence_arrays):

        matrix[seq_idx, :len(seq)] = seq

        labels.append(
            f"S{int(sequence_subject_ids[seq_idx])}"
            f"-R{int(sequence_roi_ids[seq_idx])}"
        )

    fig, ax = plt.subplots(
        figsize=(14, max(5, n_plot * 0.35))
    )

    masked = np.ma.masked_invalid(matrix)

    im = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=8,
    )

    ax.set_yticks(np.arange(n_plot))
    ax.set_yticklabels(labels)

    ax.set_xlabel("Time within independent sequence")
    ax.set_ylabel("Subject-ROI sequence")

    cbar = fig.colorbar(
        im,
        ax=ax,
        ticks=np.arange(9),
    )

    cbar.ax.set_yticklabels(CATEGORY_LABELS)
    cbar.set_label("Observation category")

    if show_titles:
        ax.set_title(
            "HMM-ready independent subject × ROI sequences"
        )

    return fig

def plot_symbolic_distribution(
    category_9: np.ndarray,
    show_titles: bool,
):
    """
    category_9: [subjects, rois, time], values in {0..8}
    """
    values = category_9.reshape(-1)
    counts = np.bincount(values, minlength=9)
    probs = counts / counts.sum() if counts.sum() > 0 else counts

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(9), probs)
    ax.set_xlabel("Observation category")
    ax.set_ylabel("Proportion")
    ax.set_xticks(np.arange(9))
    if show_titles:
        ax.set_title("Distribution of 9 symbolic observation categories")
    return fig
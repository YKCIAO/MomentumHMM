from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


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
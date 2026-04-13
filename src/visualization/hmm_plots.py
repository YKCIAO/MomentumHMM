from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_transition_matrix(
    transmat: np.ndarray,
    show_titles: bool,
):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(transmat, aspect="auto")
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_xticks(np.arange(transmat.shape[1]))
    ax.set_yticks(np.arange(transmat.shape[0]))
    if show_titles:
        ax.set_title("Transition matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def plot_mean_fo(
    FO: np.ndarray,
    show_titles: bool,
):
    """
    FO shape: [subjects, n_states]
    """
    mean_fo = FO.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(mean_fo)), mean_fo)
    ax.set_xlabel("Hidden state")
    ax.set_ylabel("Mean FO")
    ax.set_xticks(np.arange(len(mean_fo)))
    if show_titles:
        ax.set_title("Mean fractional occupancy across subjects")
    return fig


def plot_mean_mdt(
    MDT: np.ndarray,
    show_titles: bool,
):
    """
    MDT shape: [subjects, n_states]
    """
    mean_mdt = MDT.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(mean_mdt)), mean_mdt)
    ax.set_xlabel("Hidden state")
    ax.set_ylabel("Mean MDT")
    ax.set_xticks(np.arange(len(mean_mdt)))
    if show_titles:
        ax.set_title("Mean dwell time across subjects")
    return fig
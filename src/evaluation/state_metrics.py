from __future__ import annotations

import numpy as np


def safe_entropy(prob: np.ndarray) -> float:
    prob = np.asarray(prob, dtype=np.float64)
    prob = prob[prob > 0]
    if prob.size == 0:
        return 0.0
    return float(-np.sum(prob * np.log(prob)))


def normalize_entropy(entropy_value: float, n_bins: int) -> float:
    if n_bins <= 1:
        return 0.0
    max_entropy = np.log(n_bins)
    if max_entropy <= 0:
        return 0.0
    return float(entropy_value / max_entropy)


def state_usage_from_sequence(
    state_sequence: np.ndarray,
    n_hidden_states: int,
) -> np.ndarray:
    counts = np.bincount(state_sequence, minlength=n_hidden_states).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.zeros(n_hidden_states, dtype=np.float64)
    return counts / total


def observation_usage_from_sequence(
    obs_sequence: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    obs_flat = obs_sequence.reshape(-1)
    counts = np.bincount(obs_flat, minlength=n_categories).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.zeros(n_categories, dtype=np.float64)
    return counts / total


def transition_entropy(transmat: np.ndarray) -> float:
    """
    Mean row entropy, normalized by row width.
    """
    row_entropies = []
    n_states = transmat.shape[0]
    for row in transmat:
        ent = safe_entropy(row)
        row_entropies.append(normalize_entropy(ent, n_states))
    return float(np.mean(row_entropies))


def used_fraction(prob: np.ndarray) -> float:
    prob = np.asarray(prob)
    return float(np.mean(prob > 0))

def effective_state_fraction(
    FO: np.ndarray,
    min_mean_fo: float = 0.01,
) -> float:
    """
    Fraction of hidden states with meaningful occupancy.

    A state is considered effective when its mean FO across
    subjects exceeds min_mean_fo.
    """
    mean_fo = np.mean(
        FO,
        axis=0,
    )

    return float(
        np.mean(mean_fo >= min_mean_fo)
    )
def non_fragmented_state_fraction(
    MDT: np.ndarray,
    short_mdt_threshold: float = 1.5,
) -> float:
    """
    Fraction of states whose median positive MDT across subjects
    is greater than the short-state threshold.

    MDT == 0 means that the state was not visited and is excluded.
    """
    good_states = []

    for state_idx in range(MDT.shape[1]):

        values = MDT[:, state_idx]

        valid = (
            np.isfinite(values)
            & (values > 0)
        )

        if valid.sum() == 0:
            good_states.append(False)
            continue

        median_mdt = np.median(
            values[valid]
        )

        good_states.append(
            median_mdt > short_mdt_threshold
        )

    return float(
        np.mean(good_states)
    )
def mean_self_transition(
    transmat: np.ndarray,
) -> float:

    return float(
        np.mean(
            np.diag(transmat)
        )
    )





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


def coefficient_of_variation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    std = np.std(x, ddof=0)
    if abs(mean) < 1e-12:
        return float(std)
    return float(std / abs(mean))


def inverse_cv_score(x: np.ndarray) -> float:
    """
    Higher is better. Bounded in (0,1].
    """
    cv = coefficient_of_variation(x)
    return float(1.0 / (1.0 + cv))


def mean_subject_stability(matrix: np.ndarray) -> float:
    """
    matrix shape: [subjects, features]
    Compute per-feature inverse CV, then average.
    """
    scores = []
    for j in range(matrix.shape[1]):
        scores.append(inverse_cv_score(matrix[:, j]))
    return float(np.mean(scores))
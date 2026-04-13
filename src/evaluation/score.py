from __future__ import annotations

from typing import Dict

import numpy as np

from evaluation.state_metrics import (
    mean_subject_stability,
    normalize_entropy,
    observation_usage_from_sequence,
    safe_entropy,
    state_usage_from_sequence,
    transition_entropy,
    used_fraction,
)


def compute_run_metrics(
    obs: np.ndarray,
    state_sequence: np.ndarray,
    transmat: np.ndarray,
    FO: np.ndarray,
    MDT: np.ndarray,
    n_hidden_states: int,
    n_categories: int = 9,
) -> Dict[str, float]:
    state_usage = state_usage_from_sequence(state_sequence, n_hidden_states)
    obs_usage = observation_usage_from_sequence(obs, n_categories)

    state_usage_entropy = normalize_entropy(
        safe_entropy(state_usage),
        n_hidden_states,
    )
    observation_entropy = normalize_entropy(
        safe_entropy(obs_usage),
        n_categories,
    )

    metrics = {
        "state_usage_entropy": state_usage_entropy,
        "transition_entropy": transition_entropy(transmat),
        "fo_stability": mean_subject_stability(FO),
        "mdt_stability": mean_subject_stability(MDT),
        "observation_entropy": observation_entropy,
        "used_state_fraction": used_fraction(state_usage),
        "used_observation_fraction": used_fraction(obs_usage),
    }
    return metrics


def weighted_score(
    metrics: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    score = 0.0
    total_weight = 0.0

    for key, weight in weights.items():
        if key not in metrics:
            continue
        score += weight * metrics[key]
        total_weight += weight

    if total_weight <= 0:
        return 0.0
    return float(score / total_weight)


def minmax_normalize_metric_table(
    metric_table: list[Dict[str, float]],
) -> list[Dict[str, float]]:
    if len(metric_table) == 0:
        return metric_table

    keys = list(metric_table[0].keys())
    values = {k: np.array([row[k] for row in metric_table], dtype=np.float64) for k in keys}

    normalized = []
    for i in range(len(metric_table)):
        row = {}
        for k in keys:
            v = values[k]
            vmin = np.min(v)
            vmax = np.max(v)
            if abs(vmax - vmin) < 1e-12:
                row[k] = 1.0
            else:
                row[k] = float((metric_table[i][k] - vmin) / (vmax - vmin))
        normalized.append(row)
    return normalized
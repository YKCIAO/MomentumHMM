from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import pearsonr

from src.evaluation.state_metrics import (
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

def compute_age_relevance_score(
    age: np.ndarray,
    FO: np.ndarray,
    MDT: np.ndarray,
    top_k: int = 3,
) -> tuple[float, dict]:

    age = np.asarray(age, dtype=np.float64)

    fo_r = []
    mdt_r = []

    for state_idx in range(FO.shape[1]):

        # ---------- FO ----------
        fo_values = FO[:, state_idx]

        if np.std(fo_values) > 1e-12:
            r_fo, _ = pearsonr(
                age,
                fo_values,
            )
        else:
            r_fo = 0.0

        fo_r.append(float(r_fo))

        # ---------- MDT ----------
        mdt_values = MDT[:, state_idx]
        valid = np.isfinite(mdt_values) & (mdt_values > 0)
        if valid.sum() >= 4 and np.std(mdt_values[valid]) > 1e-12:

            r_mdt, _ = pearsonr(
                age[valid],
                mdt_values[valid],
            )
        else:
            r_mdt = 0.0

        mdt_r.append(float(r_mdt))

    abs_effects = np.abs(
        np.concatenate([
            np.asarray(fo_r),
            np.asarray(mdt_r),
        ])
    )

    n_top = min(
        top_k,
        len(abs_effects),
    )

    if n_top == 0:
        age_score = 0.0
        top_effects = np.array([])
    else:
        top_effects = np.sort(
            abs_effects
        )[-n_top:]

        age_score = float(
            np.mean(top_effects)
        )

    details = {
        "fo_age_r": fo_r,
        "mdt_age_r": mdt_r,
        "top_effects": top_effects.tolist(),
        "top_k": int(n_top),
    }

    return age_score, details
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
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan, np.nan

    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan, np.nan

    r, p = pearsonr(x, y)
    return float(r), float(p)


def compute_age_effect_summary(
    subject_metrics: pd.DataFrame,
    top_k: int = 5,
) -> dict:
    """
    Purpose:
        Quantify how strongly HMM-derived subject-level metrics are associated with age.

    Why it matters:
        The best model should capture meaningful age-related variation, but model
        selection should not rely on one isolated correlation only.
    """
    age = subject_metrics["Age"].to_numpy(dtype=float)

    candidate_cols = []
    for prefix in ["FO_state_", "MDT_state_", "Visits_state_"]:
        candidate_cols.extend([c for c in subject_metrics.columns if c.startswith(prefix)])

    for col in ["SwitchingRate", "StateEntropy", "NTransitions"]:
        if col in subject_metrics.columns:
            candidate_cols.append(col)

    rows = []
    for col in candidate_cols:
        y = subject_metrics[col].to_numpy(dtype=float)
        r, p = safe_pearsonr(age, y)
        if np.isfinite(r):
            rows.append(
                {
                    "metric": col,
                    "age_r": r,
                    "age_p": p,
                    "abs_age_r": abs(r),
                }
            )

    if len(rows) == 0:
        return {
            "best_metric": None,
            "best_age_r": np.nan,
            "best_age_p": np.nan,
            "max_abs_age_r": np.nan,
            "mean_abs_age_r_topk": np.nan,
            "n_age_sensitive_metrics": 0,
        }

    df = pd.DataFrame(rows).sort_values("abs_age_r", ascending=False)
    top = df.head(top_k)

    return {
        "best_metric": str(df.iloc[0]["metric"]),
        "best_age_r": float(df.iloc[0]["age_r"]),
        "best_age_p": float(df.iloc[0]["age_p"]),
        "max_abs_age_r": float(df.iloc[0]["abs_age_r"]),
        "mean_abs_age_r_topk": float(top["abs_age_r"].mean()),
        "n_age_sensitive_metrics": int((df["age_p"] < 0.05).sum()),
    }


def compute_fragmentation_score(subject_metrics: pd.DataFrame) -> dict:
    """
    Purpose:
        Detect whether the HMM solution is overly fragmented.

    Why it matters:
        A model with many states having MDT ≈ 1 may be capturing transient noise
        or threshold artifacts rather than stable brain dynamic regimes.
    """
    mdt_cols = [c for c in subject_metrics.columns if c.startswith("MDT_state_")]
    if len(mdt_cols) == 0:
        return {
            "mean_mdt": np.nan,
            "min_state_mdt": np.nan,
            "n_fragmented_states": np.nan,
            "fragmented_state_ratio": np.nan,
            "fragmentation_score": np.nan,
        }

    mdt = subject_metrics[mdt_cols].to_numpy(dtype=float)
    state_mean_mdt = np.nanmean(mdt, axis=0)

    n_fragmented = int(np.sum(state_mean_mdt <= 1.05))
    ratio = n_fragmented / len(mdt_cols)

    # Higher score = worse fragmentation
    fragmentation_score = ratio

    return {
        "mean_mdt": float(np.nanmean(state_mean_mdt)),
        "min_state_mdt": float(np.nanmin(state_mean_mdt)),
        "n_fragmented_states": n_fragmented,
        "fragmented_state_ratio": float(ratio),
        "fragmentation_score": float(fragmentation_score),
    }


def compute_state_usage_summary(subject_metrics: pd.DataFrame) -> dict:
    """
    Purpose:
        Evaluate whether states are actually used and whether one state dominates.

    Why it matters:
        A useful HMM should not assign almost all time points to one state, and
        should have several meaningfully occupied states.
    """
    fo_cols = [c for c in subject_metrics.columns if c.startswith("FO_state_")]
    if len(fo_cols) == 0:
        return {
            "n_valid_states": np.nan,
            "max_mean_fo": np.nan,
            "min_mean_fo": np.nan,
            "state_usage_balance": np.nan,
        }

    fo = subject_metrics[fo_cols].to_numpy(dtype=float)
    mean_fo = np.nanmean(fo, axis=0)

    n_valid_states = int(np.sum(mean_fo >= 0.01))
    max_mean_fo = float(np.nanmax(mean_fo))
    min_mean_fo = float(np.nanmin(mean_fo))

    # Entropy normalized by log(K); higher = more balanced.
    p = np.clip(mean_fo, 1e-12, 1.0)
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(mean_fo))
    state_usage_balance = entropy / max_entropy if max_entropy > 0 else np.nan

    return {
        "n_valid_states": n_valid_states,
        "max_mean_fo": max_mean_fo,
        "min_mean_fo": min_mean_fo,
        "state_usage_balance": float(state_usage_balance),
    }


def compute_transition_summary(hmm_npz) -> dict:
    """
    Purpose:
        Summarize transition structure.

    Why it matters:
        HMM states should form interpretable temporal dynamics, not only static
        occupancy differences.
    """
    if "TransitionProbs" not in hmm_npz.files:
        return {
            "mean_switching_rate": np.nan,
            "mean_state_entropy": np.nan,
            "transition_entropy": np.nan,
            "mean_self_transition": np.nan,
        }

    trans = hmm_npz["TransitionProbs"].astype(float)
    mean_trans = np.nanmean(trans, axis=0)

    p = np.clip(mean_trans.flatten(), 1e-12, 1.0)
    transition_entropy = -np.sum(p * np.log(p))
    transition_entropy = transition_entropy / np.log(len(p))

    mean_self_transition = float(np.nanmean(np.diag(mean_trans)))

    mean_switching_rate = (
        float(np.nanmean(hmm_npz["SwitchingRate"]))
        if "SwitchingRate" in hmm_npz.files
        else np.nan
    )

    mean_state_entropy = (
        float(np.nanmean(hmm_npz["StateEntropy"]))
        if "StateEntropy" in hmm_npz.files
        else np.nan
    )

    return {
        "mean_switching_rate": mean_switching_rate,
        "mean_state_entropy": mean_state_entropy,
        "transition_entropy": float(transition_entropy),
        "mean_self_transition": mean_self_transition,
    }


def compute_final_score(row: dict, weights: dict | None = None) -> float:
    """
    Purpose:
        Combine age sensitivity and model quality into one ranking score.

    Interpretation:
        Higher final_score = better candidate model.

    Default logic:
        reward:
            - stronger age association
            - balanced state usage
            - valid transition structure
            - state persistence
        penalize:
            - fragmentation
            - dominance by one state
    """
    if weights is None:
        weights = {
            "age_effect": 0.40,
            "topk_age_effect": 0.20,
            "state_balance": 0.15,
            "transition_entropy": 0.10,
            "mean_mdt": 0.10,
            "fragmentation_penalty": 0.25,
            "dominance_penalty": 0.10,
        }

    age_effect = row.get("max_abs_age_r", 0.0)
    topk_age_effect = row.get("mean_abs_age_r_topk", 0.0)
    state_balance = row.get("state_usage_balance", 0.0)
    transition_entropy = row.get("transition_entropy", 0.0)

    # compress MDT to 0-1 scale; MDT around 3 or above is already reasonable
    mean_mdt = row.get("mean_mdt", 0.0)
    mean_mdt_score = min(mean_mdt / 3.0, 1.0) if np.isfinite(mean_mdt) else 0.0

    fragmentation = row.get("fragmentation_score", 1.0)
    dominance = row.get("max_mean_fo", 1.0)

    score = (
        weights["age_effect"] * age_effect
        + weights["topk_age_effect"] * topk_age_effect
        + weights["state_balance"] * state_balance
        + weights["transition_entropy"] * transition_entropy
        + weights["mean_mdt"] * mean_mdt_score
        - weights["fragmentation_penalty"] * fragmentation
        - weights["dominance_penalty"] * dominance
    )

    return float(score)
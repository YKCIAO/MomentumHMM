from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ACTIVE_FO_THRESHOLD = 0.01
FRAGMENTED_MDT_THRESHOLD = 1.01


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


def get_state_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda x: int(x.split("_")[-1]))


def get_active_state_mask(
    subject_metrics: pd.DataFrame,
    fo_threshold: float = ACTIVE_FO_THRESHOLD,
) -> np.ndarray:
    """
    Active states are defined by mean FO > threshold.

    A state with FO=0, MDT=0, Visits=0 across all subjects is inactive and
    should not be interpreted as a meaningful brain state.
    """
    fo_cols = get_state_columns(subject_metrics, "FO_state_")
    if len(fo_cols) == 0:
        return np.array([], dtype=bool)

    fo = subject_metrics[fo_cols].to_numpy(dtype=float)
    mean_fo = np.nanmean(fo, axis=0)

    return mean_fo > fo_threshold


def filter_metric_columns_by_active_states(
    subject_metrics: pd.DataFrame,
    prefix: str,
    active_mask: np.ndarray,
) -> list[str]:
    cols = get_state_columns(subject_metrics, prefix)

    if len(cols) == 0:
        return []

    if len(cols) != len(active_mask):
        return cols

    return [c for c, active in zip(cols, active_mask) if active]


def compute_age_effect_summary(
    subject_metrics: pd.DataFrame,
    top_k: int = 5,
) -> dict:
    """
    Compute age associations only for active states.

    This avoids selecting a model because of unstable or unused states.
    Constant all-zero columns are also ignored automatically by safe_pearsonr().
    """
    age = subject_metrics["Age"].to_numpy(dtype=float)
    active_mask = get_active_state_mask(subject_metrics)

    candidate_cols = []

    for prefix in ["FO_state_", "MDT_state_", "Visits_state_"]:
        candidate_cols.extend(
            filter_metric_columns_by_active_states(subject_metrics, prefix, active_mask)
        )

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
    Compute fragmentation among active states only.

    Inactive states are penalized separately.
    """
    fo_cols = get_state_columns(subject_metrics, "FO_state_")
    mdt_cols = get_state_columns(subject_metrics, "MDT_state_")

    if len(fo_cols) == 0 or len(mdt_cols) == 0:
        return {
            "mean_mdt": np.nan,
            "min_state_mdt": np.nan,
            "n_fragmented_states": np.nan,
            "fragmented_state_ratio": np.nan,
            "fragmentation_score": np.nan,
            "n_inactive_states": np.nan,
            "inactive_state_ratio": np.nan,
        }

    active_mask = get_active_state_mask(subject_metrics)

    mdt = subject_metrics[mdt_cols].to_numpy(dtype=float)
    state_mean_mdt = np.nanmean(mdt, axis=0)

    n_states = len(state_mean_mdt)
    n_active = int(np.sum(active_mask))
    n_inactive = int(n_states - n_active)
    inactive_ratio = n_inactive / n_states if n_states > 0 else np.nan

    if n_active == 0:
        return {
            "mean_mdt": 0.0,
            "min_state_mdt": 0.0,
            "n_fragmented_states": n_states,
            "fragmented_state_ratio": 1.0,
            "fragmentation_score": 1.0,
            "n_inactive_states": n_inactive,
            "inactive_state_ratio": inactive_ratio,
        }

    active_mdt = state_mean_mdt[active_mask]
    n_fragmented = int(np.sum(active_mdt <= FRAGMENTED_MDT_THRESHOLD))
    fragmented_ratio = n_fragmented / n_active

    return {
        "mean_mdt": float(np.nanmean(active_mdt)),
        "min_state_mdt": float(np.nanmin(active_mdt)),
        "n_fragmented_states": n_fragmented,
        "fragmented_state_ratio": float(fragmented_ratio),
        "fragmentation_score": float(fragmented_ratio),
        "n_inactive_states": n_inactive,
        "inactive_state_ratio": float(inactive_ratio),
    }


def compute_state_usage_summary(subject_metrics: pd.DataFrame) -> dict:
    """
    State usage summary based on active states.

    Balance is computed after excluding inactive states and renormalizing FO.
    """
    fo_cols = get_state_columns(subject_metrics, "FO_state_")
    if len(fo_cols) == 0:
        return {
            "n_states": np.nan,
            "n_valid_states": np.nan,
            "effective_k": np.nan,
            "max_mean_fo": np.nan,
            "min_mean_fo": np.nan,
            "state_usage_balance": np.nan,
        }

    fo = subject_metrics[fo_cols].to_numpy(dtype=float)
    mean_fo = np.nanmean(fo, axis=0)

    active_mask = mean_fo > ACTIVE_FO_THRESHOLD
    active_fo = mean_fo[active_mask]

    n_states = len(mean_fo)
    n_valid_states = int(np.sum(active_mask))

    if n_valid_states == 0:
        return {
            "n_states": n_states,
            "n_valid_states": 0,
            "effective_k": 0,
            "max_mean_fo": 0.0,
            "min_mean_fo": 0.0,
            "state_usage_balance": 0.0,
        }

    active_fo = active_fo / np.sum(active_fo)

    p = np.clip(active_fo, 1e-12, 1.0)
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(active_fo))
    balance = entropy / max_entropy if max_entropy > 0 else 1.0

    return {
        "n_states": n_states,
        "n_valid_states": n_valid_states,
        "effective_k": n_valid_states,
        "max_mean_fo": float(np.nanmax(active_fo)),
        "min_mean_fo": float(np.nanmin(active_fo)),
        "state_usage_balance": float(balance),
    }


def compute_transition_summary(hmm_npz) -> dict:
    """
    Transition summary after excluding inactive states.

    This avoids inactive rows/columns distorting transition entropy.
    """
    if "TransitionProbs" not in hmm_npz.files:
        return {
            "mean_switching_rate": np.nan,
            "mean_state_entropy": np.nan,
            "transition_entropy": np.nan,
            "mean_self_transition": np.nan,
        }

    trans = hmm_npz["TransitionProbs"].astype(float)

    if "FO" in hmm_npz.files:
        mean_fo = np.nanmean(hmm_npz["FO"].astype(float), axis=0)
        active_mask = mean_fo > ACTIVE_FO_THRESHOLD
    else:
        active_mask = np.ones(trans.shape[1], dtype=bool)

    if np.sum(active_mask) == 0:
        return {
            "mean_switching_rate": np.nan,
            "mean_state_entropy": np.nan,
            "transition_entropy": 0.0,
            "mean_self_transition": 0.0,
        }

    trans_active = trans[:, active_mask, :][:, :, active_mask]
    mean_trans = np.nanmean(trans_active, axis=0)

    row_sums = mean_trans.sum(axis=1, keepdims=True)
    valid_rows = row_sums.squeeze() > 0

    if np.any(valid_rows):
        mean_trans[valid_rows] = mean_trans[valid_rows] / row_sums[valid_rows]

    p = mean_trans[valid_rows].flatten() if np.any(valid_rows) else np.array([])
    p = p[p > 0]

    if len(p) == 0:
        transition_entropy = 0.0
    else:
        transition_entropy = -np.sum(p * np.log(p))
        transition_entropy = transition_entropy / np.log(len(p)) if len(p) > 1 else 0.0

    mean_self_transition = float(np.nanmean(np.diag(mean_trans))) if mean_trans.size > 0 else np.nan

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


def finite_or_zero(value) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0

    return value if np.isfinite(value) else 0.0


def compute_final_score(row: dict, weights: dict | None = None) -> float:
    """
    Composite score with explicit inactive-state penalty.

    This prevents nominal K=6/K=7 models with unused states from being favored
    only because they show a higher max age correlation.
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
            "inactive_penalty": 0.30,
        }

    age_effect = finite_or_zero(row.get("max_abs_age_r", 0.0))
    topk_age_effect = finite_or_zero(row.get("mean_abs_age_r_topk", 0.0))
    state_balance = finite_or_zero(row.get("state_usage_balance", 0.0))
    transition_entropy = finite_or_zero(row.get("transition_entropy", 0.0))

    mean_mdt = finite_or_zero(row.get("mean_mdt", 0.0))
    mean_mdt_score = min(mean_mdt / 3.0, 1.0)

    fragmentation = finite_or_zero(row.get("fragmentation_score", 1.0))
    dominance = finite_or_zero(row.get("max_mean_fo", 1.0))
    inactive_ratio = finite_or_zero(row.get("inactive_state_ratio", 0.0))

    score = (
        weights["age_effect"] * age_effect
        + weights["topk_age_effect"] * topk_age_effect
        + weights["state_balance"] * state_balance
        + weights["transition_entropy"] * transition_entropy
        + weights["mean_mdt"] * mean_mdt_score
        - weights["fragmentation_penalty"] * fragmentation
        - weights["dominance_penalty"] * dominance
        - weights["inactive_penalty"] * inactive_ratio
    )

    return float(score)
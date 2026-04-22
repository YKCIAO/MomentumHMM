from __future__ import annotations

import numpy as np


def _safe_row_normalize(mat: np.ndarray) -> np.ndarray:
    """
    Row-normalize a 2D matrix.
    Rows with sum=0 remain all zeros.
    """
    mat = mat.astype(np.float64, copy=False)
    row_sums = mat.sum(axis=1, keepdims=True)
    out = np.zeros_like(mat, dtype=np.float64)
    valid = row_sums.squeeze(-1) > 0
    if np.any(valid):
        out[valid] = mat[valid] / row_sums[valid]
    return out


def _extract_runs_for_state(seq: np.ndarray, state: int) -> list[int]:
    """
    Extract consecutive run lengths for one state from a 1D state sequence.
    """
    runs: list[int] = []
    current_len = 0

    for val in seq:
        if val == state:
            current_len += 1
        else:
            if current_len > 0:
                runs.append(current_len)
                current_len = 0

    if current_len > 0:
        runs.append(current_len)

    return runs


def compute_fractional_occupancy(
    subject_state_seqs: list[np.ndarray],
    n_hidden_states: int,
) -> np.ndarray:
    """
    FO[i, k] = fraction of time subject i spent in state k.
    """
    n_sub = len(subject_state_seqs)
    fo = np.zeros((n_sub, n_hidden_states), dtype=np.float64)

    for i, seq in enumerate(subject_state_seqs):
        total = len(seq)
        if total == 0:
            continue
        for k in range(n_hidden_states):
            fo[i, k] = np.mean(seq == k)

    return fo


def compute_mean_dwell_time(
    subject_state_seqs: list[np.ndarray],
    n_hidden_states: int,
) -> np.ndarray:
    """
    MDT[i, k] = mean consecutive run length of state k for subject i.
    If a state never occurs, MDT = 0.
    """
    n_sub = len(subject_state_seqs)
    mdt = np.zeros((n_sub, n_hidden_states), dtype=np.float64)

    for i, seq in enumerate(subject_state_seqs):
        for k in range(n_hidden_states):
            runs = _extract_runs_for_state(seq, k)
            mdt[i, k] = float(np.mean(runs)) if len(runs) > 0 else 0.0

    return mdt


def compute_visit_count(
    subject_state_seqs: list[np.ndarray],
    n_hidden_states: int,
) -> np.ndarray:
    """
    Visits[i, k] = number of entries into state k for subject i.
    Equivalent to the number of runs for state k.
    """
    n_sub = len(subject_state_seqs)
    visits = np.zeros((n_sub, n_hidden_states), dtype=np.int32)

    for i, seq in enumerate(subject_state_seqs):
        for k in range(n_hidden_states):
            runs = _extract_runs_for_state(seq, k)
            visits[i, k] = len(runs)

    return visits


def compute_transition_counts(
    subject_state_seqs: list[np.ndarray],
    n_hidden_states: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute subject-level transition count matrices and transition numbers.

    Returns
    -------
    trans_counts : np.ndarray
        Shape [n_subjects, K, K]
    n_transitions : np.ndarray
        Shape [n_subjects,]
        Total number of state changes (seq[t] != seq[t-1]).
    """
    n_sub = len(subject_state_seqs)
    trans_counts = np.zeros((n_sub, n_hidden_states, n_hidden_states), dtype=np.int32)
    n_transitions = np.zeros(n_sub, dtype=np.int32)

    for i, seq in enumerate(subject_state_seqs):
        if len(seq) <= 1:
            continue

        changes = 0
        for t in range(1, len(seq)):
            prev_s = int(seq[t - 1])
            curr_s = int(seq[t])

            trans_counts[i, prev_s, curr_s] += 1

            if curr_s != prev_s:
                changes += 1

        n_transitions[i] = changes

    return trans_counts, n_transitions


def compute_switching_rate(
    subject_state_seqs: list[np.ndarray],
    n_transitions: np.ndarray,
) -> np.ndarray:
    """
    SwitchingRate[i] = number of state changes / (T-1)
    """
    n_sub = len(subject_state_seqs)
    switching_rate = np.zeros(n_sub, dtype=np.float64)

    for i, seq in enumerate(subject_state_seqs):
        denom = max(len(seq) - 1, 1)
        switching_rate[i] = float(n_transitions[i]) / float(denom)

    return switching_rate


def compute_state_entropy(
    fo: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Shannon entropy over subject-level FO distribution.
    Higher = subject distributes time across states more evenly.
    """
    p = np.clip(fo, eps, 1.0)
    entropy = -np.sum(p * np.log(p), axis=1)
    return entropy.astype(np.float64)


def compute_subject_level_metrics(
    subject_state_seqs: list[np.ndarray],
    n_hidden_states: int,
) -> dict:
    """
    Expanded 2.0 subject-level HMM metrics.

    Returns
    -------
    dict with keys:
        FO                         [n_sub, K]
        MDT                        [n_sub, K]
        Visits                     [n_sub, K]
        TransitionCounts           [n_sub, K, K]
        TransitionProbs            [n_sub, K, K]
        NTransitions               [n_sub]
        SwitchingRate              [n_sub]
        StateEntropy               [n_sub]
    """
    fo = compute_fractional_occupancy(subject_state_seqs, n_hidden_states)
    mdt = compute_mean_dwell_time(subject_state_seqs, n_hidden_states)
    visits = compute_visit_count(subject_state_seqs, n_hidden_states)

    trans_counts, n_transitions = compute_transition_counts(
        subject_state_seqs,
        n_hidden_states,
    )
    trans_probs = np.zeros_like(trans_counts, dtype=np.float64)
    for i in range(trans_counts.shape[0]):
        trans_probs[i] = _safe_row_normalize(trans_counts[i])

    switching_rate = compute_switching_rate(subject_state_seqs, n_transitions)
    state_entropy = compute_state_entropy(fo)

    return {
        "FO": fo,
        "MDT": mdt,
        "Visits": visits,
        "TransitionCounts": trans_counts,
        "TransitionProbs": trans_probs,
        "NTransitions": n_transitions,
        "SwitchingRate": switching_rate,
        "StateEntropy": state_entropy,
    }
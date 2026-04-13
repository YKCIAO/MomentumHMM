from __future__ import annotations

import numpy as np


def compute_subject_level_metrics(
    subject_state_seqs: list[np.ndarray],
    n_hidden_states: int,
) -> dict:
    n_sub = len(subject_state_seqs)
    fo = np.zeros((n_sub, n_hidden_states), dtype=np.float64)
    mdt = np.zeros((n_sub, n_hidden_states), dtype=np.float64)

    for i, seq in enumerate(subject_state_seqs):
        total = len(seq)
        for k in range(n_hidden_states):
            idx = seq == k
            fo[i, k] = idx.mean() if total > 0 else 0.0

            runs = []
            current_len = 0
            for val in seq:
                if val == k:
                    current_len += 1
                else:
                    if current_len > 0:
                        runs.append(current_len)
                        current_len = 0
            if current_len > 0:
                runs.append(current_len)

            mdt[i, k] = float(np.mean(runs)) if len(runs) > 0 else 0.0

    return {"FO": fo, "MDT": mdt}
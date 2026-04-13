from __future__ import annotations

import numpy as np


def decode_hmm(model, obs: np.ndarray, lengths: np.ndarray) -> dict:
    logprob, state_sequence = model.decode(obs, lengths=lengths, algorithm="viterbi")
    posterior = model.predict_proba(obs, lengths=lengths)
    return {
        "logprob": float(logprob),
        "state_sequence": state_sequence.astype(np.int32),
        "posterior": posterior.astype(np.float64),
    }


def split_sequence_by_lengths(seq: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    out = []
    start = 0
    for length in lengths:
        end = start + int(length)
        out.append(seq[start:end])
        start = end
    return out
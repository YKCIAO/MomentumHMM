from __future__ import annotations

import numpy as np


def flatten_subject_roi_as_observation_sequence(
    category_9: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    [subjects, rois, time] -> concatenated observation sequence
    subject-wise lengths preserved
    """
    seqs = []
    lengths = []

    for i in range(category_9.shape[0]):
        subj_seq = category_9[i].reshape(-1)
        seqs.append(subj_seq)
        lengths.append(len(subj_seq))

    obs = np.concatenate(seqs).astype(np.int64).reshape(-1, 1)
    lengths = np.array(lengths, dtype=np.int64)
    return obs, lengths
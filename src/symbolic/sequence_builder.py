from __future__ import annotations

import numpy as np


def flatten_subject_roi_as_observation_sequence(
    category_9: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert [subjects, rois, time] into independent subject-ROI sequences.

    Each (subject, ROI) pair is treated as one independent HMM sequence.

    Returns
    -------
    obs : ndarray, shape [subjects * rois * time, 1]
        Concatenated observations.

    lengths : ndarray, shape [subjects * rois]
        Length of each independent subject-ROI sequence.

    sequence_subject_ids : ndarray
        Subject index corresponding to each sequence.

    sequence_roi_ids : ndarray
        ROI index corresponding to each sequence.
    """
    n_subjects, n_rois, _ = category_9.shape

    seqs = []
    lengths = []
    sequence_subject_ids = []
    sequence_roi_ids = []

    for subj_idx in range(n_subjects):
        for roi_idx in range(n_rois):

            seq = category_9[subj_idx, roi_idx, :]

            seqs.append(seq)

            lengths.append(len(seq))
            sequence_subject_ids.append(subj_idx)
            sequence_roi_ids.append(roi_idx)

    obs = np.concatenate(seqs).astype(np.int64).reshape(-1, 1)

    return (
        obs,
        np.asarray(lengths, dtype=np.int64),
        np.asarray(sequence_subject_ids, dtype=np.int64),
        np.asarray(sequence_roi_ids, dtype=np.int64),
    )
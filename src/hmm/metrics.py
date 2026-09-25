from __future__ import annotations

import numpy as np


def compute_subject_level_metrics(
    state_seqs: list[np.ndarray],
    sequence_subject_ids: np.ndarray,
    sequence_roi_ids: np.ndarray,
    n_subjects: int,
    n_rois: int,
    n_hidden_states: int,
) -> dict:

    fo_roi = np.zeros(
        (n_subjects, n_rois, n_hidden_states),
        dtype=np.float64,
    )

    mdt_roi = np.zeros(
        (n_subjects, n_rois, n_hidden_states),
        dtype=np.float64,
    )

    visits_roi = np.zeros(
        (n_subjects, n_rois, n_hidden_states),
        dtype=np.int64,
    )

    subject_dwells = [
        [[] for _ in range(n_hidden_states)]
        for _ in range(n_subjects)
    ]

    for seq_idx, seq in enumerate(state_seqs):

        subj_idx = int(sequence_subject_ids[seq_idx])
        roi_idx = int(sequence_roi_ids[seq_idx])

        for k in range(n_hidden_states):

            # ROI-level fractional occupancy
            fo_roi[subj_idx, roi_idx, k] = np.mean(seq == k)

            # ROI-internal dwell episodes only
            runs = extract_dwell_times(seq, k)

            visits_roi[subj_idx, roi_idx, k] = len(runs)

            if len(runs) > 0:
                mdt_roi[subj_idx, roi_idx, k] = float(
                    np.mean(runs)
                )

                subject_dwells[subj_idx][k].extend(runs)

    # Subject-level FO
    # All ROIs have equal length in the current dataset.
    fo_subject = np.mean(
        fo_roi,
        axis=1,
    )

    # Subject-level MDT:
    # pool all ROI-internal dwell episodes first
    mdt_subject = np.zeros(
        (n_subjects, n_hidden_states),
        dtype=np.float64,
    )

    for subj_idx in range(n_subjects):

        for k in range(n_hidden_states):

            runs = subject_dwells[subj_idx][k]

            if len(runs) > 0:
                mdt_subject[subj_idx, k] = float(
                    np.mean(runs)
                )

    # Subject-level visit count
    visits_subject = np.sum(
        visits_roi,
        axis=1,
    )

    return {
        "FO": fo_subject,
        "MDT": mdt_subject,
        "Visits": visits_subject,
        "FO_roi": fo_roi,
        "MDT_roi": mdt_roi,
        "Visits_roi": visits_roi,
    }

def extract_dwell_times(
    seq: np.ndarray,
    state: int,
) -> list[int]:

    runs = []
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
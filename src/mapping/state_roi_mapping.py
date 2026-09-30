from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io_utils import ensure_dir, load_npz

def load_roi_mapping(
    n_rois: int,
    mapping_xlsx: Path | None,
) -> pd.DataFrame:
    """
    Load atlas/ROI annotations.

    Python ROI indices are 0-based, while atlas labels
    are assumed to be 1-based.
    """

    if mapping_xlsx is None or not mapping_xlsx.exists():

        return pd.DataFrame({
            "roi_index": np.arange(n_rois),
            "Label": np.arange(1, n_rois + 1),
            "Gyrus": [
                f"ROI_{i}"
                for i in range(n_rois)
            ],
        })

    df = pd.read_excel(
        mapping_xlsx
    ).copy()

    if "Label" not in df.columns:
        raise ValueError(
            "ROI mapping file must contain a 'Label' column."
        )

    df["Label"] = df["Label"].astype(int)

    df["roi_index"] = (
        df["Label"] - 1
    )

    optional_columns = [
        "Gyrus",
        "subregion_name",
        "region",
        "exact_region",
        "Yeo_7network",
        "Yeo_17network",
        "Yeo_7network_name",
        "Yeo_17network_name",
    ]

    for column in optional_columns:

        if column not in df.columns:
            df[column] = "Unknown"

    keep_columns = [
        "roi_index",
        "Label",
        "Gyrus",
        "subregion_name",
        "region",
        "exact_region",
        "Yeo_7network",
        "Yeo_17network",
        "Yeo_7network_name",
        "Yeo_17network_name",
    ]

    return df[keep_columns]

def reconstruct_state_array(
    state_sequence: np.ndarray,
    lengths: np.ndarray,
    sequence_subject_ids: np.ndarray,
    sequence_roi_ids: np.ndarray,
    n_subjects: int,
    n_rois: int,
    n_timepoints: int,
) -> np.ndarray:
    """
    Reconstruct decoded HMM states into:

        subjects × ROIs × timepoints

    using explicit sequence metadata.
    """

    state_3d = np.full(
        (
            n_subjects,
            n_rois,
            n_timepoints,
        ),
        fill_value=-1,
        dtype=np.int32,
    )

    start = 0

    for seq_idx, length in enumerate(
        lengths
    ):

        length = int(length)

        end = start + length

        seq = state_sequence[
            start:end
        ]

        subject_idx = int(
            sequence_subject_ids[
                seq_idx
            ]
        )

        roi_idx = int(
            sequence_roi_ids[
                seq_idx
            ]
        )

        if length != n_timepoints:

            raise ValueError(
                f"Sequence length mismatch for "
                f"subject={subject_idx}, "
                f"ROI={roi_idx}: "
                f"{length} != {n_timepoints}"
            )

        state_3d[
            subject_idx,
            roi_idx,
            :
        ] = seq

        start = end

    if start != len(state_sequence):

        raise ValueError(
            "State sequence was not fully consumed."
        )

    if np.any(state_3d < 0):

        raise ValueError(
            "Some subject-ROI sequences were not reconstructed."
        )

    return state_3d

def compute_state_roi_mapping(
    cfg,
    symbolic_run_name: str,
    k_value: int,
    top_n: int = 15,
):

    symbolic_dir = (
        Path(
            cfg.paths.symbolic_output_root
        )
        / symbolic_run_name
    )

    hmm_dir = (
        Path(
            cfg.paths.hmm_output_root
        )
        / symbolic_run_name
        / f"K_{k_value}"
    )

    symbolic_path = (
        symbolic_dir
        / "symbolic_outputs.npz"
    )

    sequence_path = (
        symbolic_dir
        / "hmm_ready_sequence.npz"
    )

    hmm_path = (
        hmm_dir
        / "hmm_results.npz"
    )

    if not symbolic_path.exists():
        raise FileNotFoundError(
            symbolic_path
        )

    if not sequence_path.exists():
        raise FileNotFoundError(
            sequence_path
        )

    if not hmm_path.exists():
        raise FileNotFoundError(
            hmm_path
        )

    symbolic = load_npz(
        symbolic_path
    )

    sequence = load_npz(
        sequence_path
    )

    hmm = load_npz(
        hmm_path
    )

    x_std = np.asarray(
        symbolic["x_std"]
    )

    dx_std = np.asarray(
        symbolic["dx_std"]
    )

    deviation_code = np.asarray(
        symbolic["deviation_code"]
    )

    momentum_code = np.asarray(
        symbolic["momentum_code"]
    )

    (
        n_subjects,
        n_rois,
        n_timepoints,
    ) = x_std.shape

    if dx_std.shape != x_std.shape:
        raise ValueError(
            "dx_std and x_std shape mismatch."
        )

    lengths = sequence["lengths"]

    sequence_subject_ids = (
        sequence[
            "sequence_subject_ids"
        ]
    )

    sequence_roi_ids = (
        sequence[
            "sequence_roi_ids"
        ]
    )

    state_sequence = np.asarray(
        hmm["state_sequence"]
    )

    n_states = int(
        hmm["FO"].shape[1]
    )

    state_3d = reconstruct_state_array(
        state_sequence=state_sequence,
        lengths=lengths,
        sequence_subject_ids=sequence_subject_ids,
        sequence_roi_ids=sequence_roi_ids,
        n_subjects=n_subjects,
        n_rois=n_rois,
        n_timepoints=n_timepoints,
    )

    mapping_path = getattr(
        cfg.paths,
        "roi_mapping_xlsx",
        "",
    )

    mapping_path = (
        Path(mapping_path)
        if mapping_path
        else None
    )

    roi_mapping = load_roi_mapping(
        n_rois=n_rois,
        mapping_xlsx=mapping_path,
    )

    rows = []

    for state_idx in range(
        n_states
    ):

        for roi_idx in range(
            n_rois
        ):

            mask = (
                state_3d[
                    :,
                    roi_idx,
                    :
                ]
                == state_idx
            )

            n_points = int(
                mask.sum()
            )

            occupancy = float(
                mask.mean()
            )

            deviation_values = (
                x_std[
                    :,
                    roi_idx,
                    :
                ][mask]
            )

            momentum_values = (
                dx_std[
                    :,
                    roi_idx,
                    :
                ][mask]
            )

            deviation_codes = (
                deviation_code[
                    :,
                    roi_idx,
                    :
                ][mask]
            )

            momentum_codes = (
                momentum_code[
                    :,
                    roi_idx,
                    :
                ][mask]
            )

            if n_points > 0:

                mean_deviation = float(
                    np.mean(
                        deviation_values
                    )
                )

                mean_abs_deviation = float(
                    np.mean(
                        np.abs(
                            deviation_values
                        )
                    )
                )

                std_deviation = float(
                    np.std(
                        deviation_values
                    )
                )

                mean_momentum = float(
                    np.mean(
                        momentum_values
                    )
                )

                mean_abs_momentum = float(
                    np.mean(
                        np.abs(
                            momentum_values
                        )
                    )
                )

                std_momentum = float(
                    np.std(
                        momentum_values
                    )
                )

                positive_deviation_ratio = float(
                    np.mean(
                        deviation_codes == 1
                    )
                )

                negative_deviation_ratio = float(
                    np.mean(
                        deviation_codes == -1
                    )
                )

                neutral_deviation_ratio = float(
                    np.mean(
                        deviation_codes == 0
                    )
                )

                positive_momentum_ratio = float(
                    np.mean(
                        momentum_codes == 1
                    )
                )

                negative_momentum_ratio = float(
                    np.mean(
                        momentum_codes == -1
                    )
                )

                neutral_momentum_ratio = float(
                    np.mean(
                        momentum_codes == 0
                    )
                )

            else:

                mean_deviation = np.nan
                mean_abs_deviation = np.nan
                std_deviation = np.nan

                mean_momentum = np.nan
                mean_abs_momentum = np.nan
                std_momentum = np.nan

                positive_deviation_ratio = np.nan
                negative_deviation_ratio = np.nan
                neutral_deviation_ratio = np.nan

                positive_momentum_ratio = np.nan
                negative_momentum_ratio = np.nan
                neutral_momentum_ratio = np.nan

            label_row = roi_mapping[
                roi_mapping[
                    "roi_index"
                ]
                == roi_idx
            ]

            if len(label_row) > 0:

                label_info = (
                    label_row
                    .iloc[0]
                    .to_dict()
                )

            else:

                label_info = {
                    "Label": roi_idx + 1,
                    "Gyrus":
                        f"ROI_{roi_idx}",
                    "subregion_name":
                        "Unknown",
                    "region":
                        "Unknown",
                    "exact_region":
                        "Unknown",
                    "Yeo_7network":
                        "Unknown",
                    "Yeo_17network":
                        "Unknown",
                    "Yeo_7network_name":
                        "Unknown",
                    "Yeo_17network_name":
                        "Unknown",
                }

            rows.append(
                {
                    "state":
                        state_idx,

                    "roi_index":
                        roi_idx,

                    "Label":
                        label_info[
                            "Label"
                        ],

                    "Gyrus":
                        label_info[
                            "Gyrus"
                        ],

                    "subregion_name":
                        label_info.get(
                            "subregion_name",
                            "Unknown",
                        ),

                    "region":
                        label_info.get(
                            "region",
                            "Unknown",
                        ),

                    "exact_region":
                        label_info.get(
                            "exact_region",
                            "Unknown",
                        ),

                    "Yeo_7network":
                        label_info.get(
                            "Yeo_7network",
                            "Unknown",
                        ),

                    "Yeo_7network_name":
                        label_info.get(
                            "Yeo_7network_name",
                            "Unknown",
                        ),

                    "state_roi_occupancy":
                        occupancy,

                    "n_points":
                        n_points,

                    "mean_deviation":
                        mean_deviation,

                    "mean_abs_deviation":
                        mean_abs_deviation,

                    "std_deviation":
                        std_deviation,

                    "mean_momentum":
                        mean_momentum,

                    "mean_abs_momentum":
                        mean_abs_momentum,

                    "std_momentum":
                        std_momentum,

                    "positive_deviation_ratio":
                        positive_deviation_ratio,

                    "negative_deviation_ratio":
                        negative_deviation_ratio,

                    "neutral_deviation_ratio":
                        neutral_deviation_ratio,

                    "positive_momentum_ratio":
                        positive_momentum_ratio,

                    "negative_momentum_ratio":
                        negative_momentum_ratio,

                    "neutral_momentum_ratio":
                        neutral_momentum_ratio,
                }
            )

    full_df = pd.DataFrame(
        rows
    )

    top_occupancy_rows = []
    top_deviation_rows = []
    top_momentum_rows = []

    for state_idx in range(
        n_states
    ):

        tmp = full_df[
            full_df["state"]
            == state_idx
        ].copy()

        top_occ = (
            tmp
            .sort_values(
                [
                    "state_roi_occupancy",
                    "mean_abs_deviation",
                    "mean_abs_momentum",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .head(top_n)
        )

        top_occ.insert(
            1,
            "rank",
            range(
                1,
                len(top_occ) + 1
            ),
        )

        top_occupancy_rows.append(
            top_occ
        )

        top_dev = (
            tmp
            .sort_values(
                [
                    "mean_abs_deviation",
                    "state_roi_occupancy",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .head(top_n)
        )

        top_dev.insert(
            1,
            "rank",
            range(
                1,
                len(top_dev) + 1
            ),
        )

        top_deviation_rows.append(
            top_dev
        )

        top_mom = (
            tmp
            .sort_values(
                [
                    "mean_abs_momentum",
                    "state_roi_occupancy",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .head(top_n)
        )

        top_mom.insert(
            1,
            "rank",
            range(
                1,
                len(top_mom) + 1
            ),
        )

        top_momentum_rows.append(
            top_mom
        )
        top_occupancy_df = pd.concat(
            top_occupancy_rows,
            ignore_index=True,
        )

        top_deviation_df = pd.concat(
            top_deviation_rows,
            ignore_index=True,
        )

        top_momentum_df = pd.concat(
            top_momentum_rows,
            ignore_index=True,
        )
        network_summary = (
            full_df
            .groupby(
                [
                    "state",
                    "Yeo_7network",
                    "Yeo_7network_name",
                ],
                dropna=False,
                as_index=False,
            )
            .agg(
                mean_state_roi_occupancy=(
                    "state_roi_occupancy",
                    "mean",
                ),

                max_state_roi_occupancy=(
                    "state_roi_occupancy",
                    "max",
                ),

                mean_abs_deviation=(
                    "mean_abs_deviation",
                    "mean",
                ),

                mean_abs_momentum=(
                    "mean_abs_momentum",
                    "mean",
                ),

                n_rois=(
                    "roi_index",
                    "count",
                ),
            )
            .sort_values(
                [
                    "state",
                    "mean_state_roi_occupancy",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
        )
        gyrus_summary = (
            full_df
            .groupby(
                [
                    "state",
                    "Gyrus",
                ],
                dropna=False,
                as_index=False,
            )
            .agg(
                mean_state_roi_occupancy=(
                    "state_roi_occupancy",
                    "mean",
                ),

                max_state_roi_occupancy=(
                    "state_roi_occupancy",
                    "max",
                ),

                mean_abs_deviation=(
                    "mean_abs_deviation",
                    "mean",
                ),

                mean_abs_momentum=(
                    "mean_abs_momentum",
                    "mean",
                ),

                n_rois=(
                    "roi_index",
                    "count",
                ),
            )
            .sort_values(
                [
                    "state",
                    "mean_state_roi_occupancy",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
        )
        output_dir = ensure_dir(
            hmm_dir
            / "state_roi_mapping"
        )

        excel_path = (
                output_dir
                / "state_roi_mapping.xlsx"
        )

        with pd.ExcelWriter(
                excel_path,
                engine="openpyxl",
        ) as writer:

            full_df.to_excel(
                writer,
                sheet_name="FullMapping",
                index=False,
            )

            top_occupancy_df.to_excel(
                writer,
                sheet_name="TopOccupancy",
                index=False,
            )

            top_deviation_df.to_excel(
                writer,
                sheet_name="TopDeviation",
                index=False,
            )

            top_momentum_df.to_excel(
                writer,
                sheet_name="TopMomentum",
                index=False,
            )

            network_summary.to_excel(
                writer,
                sheet_name="Yeo7Summary",
                index=False,
            )

            gyrus_summary.to_excel(
                writer,
                sheet_name="GyrusSummary",
                index=False,
            )

            for ws in writer.book.worksheets:

                ws.freeze_panes = "A2"

                ws.auto_filter.ref = (
                    ws.dimensions
                )

                for column_cells in ws.columns:
                    max_length = max(
                        (
                            len(
                                str(
                                    cell.value
                                )
                            )
                            if cell.value
                               is not None
                            else 0
                        )
                        for cell
                        in column_cells
                    )

                    letter = (
                        column_cells[
                            0
                        ].column_letter
                    )

                    ws.column_dimensions[
                        letter
                    ].width = min(
                        max_length + 2,
                        30,
                    )
        return {
            "full": full_df,
            "top_occupancy":
                top_occupancy_df,
            "top_deviation":
                top_deviation_df,
            "top_momentum":
                top_momentum_df,
            "network_summary":
                network_summary,
            "gyrus_summary":
                gyrus_summary,
            "excel_path":
                excel_path,
        }
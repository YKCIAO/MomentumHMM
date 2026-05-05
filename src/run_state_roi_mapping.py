from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from config import load_experiment_config
from utils.io_utils import ensure_dir


CONFIG_PATH = "configs/experiment_config.json"

FEATURE_RUN_NAME = "act_0.7000__trend_0.8000__a_1.0000__b_1.0000"
K_VALUE = 6
TOP_N = 15


def load_roi_mapping(n_rois: int, mapping_xlsx: Path | None = None) -> pd.DataFrame:
    """
    Load ROI mapping table.

    Required/expected columns in Mapping.xlsx:
        Label
        Gyrus
        subregion_name
        region
        exact_region
        Yeo_7network
        Yeo_17network
        Yeo_7network_name
        Yeo_17network_name

    Important:
        ROI index in Python is 0-based.
        Label in atlas table is usually 1-based.
        So roi_index = Label - 1.
    """
    if mapping_xlsx is not None and mapping_xlsx.exists():
        df = pd.read_excel(mapping_xlsx)

        required = ["Label", "Gyrus", "Yeo_7network", "Yeo_7network_name"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Mapping file missing required columns: {missing}")

        df = df.copy()
        df["Label"] = df["Label"].astype(int)
        df["roi_index"] = df["Label"] - 1

        optional_cols = [
            "Gyrus",
            "subregion_name",
            "region",
            "exact_region",
            "Yeo_7network",
            "Yeo_17network",
            "Yeo_7network_name",
            "Yeo_17network_name",
        ]

        for c in optional_cols:
            if c not in df.columns:
                df[c] = "Unknown"

        df = df[
            [
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
        ]

        if len(df) < n_rois:
            print(
                f"[Warning] Mapping file has {len(df)} rows, "
                f"but data has {n_rois} ROIs."
            )

        return df

    return pd.DataFrame({
        "roi_index": np.arange(n_rois),
        "Label": np.arange(1, n_rois + 1),
        "Gyrus": [f"ROI_{i}" for i in range(n_rois)],
        "subregion_name": ["Unknown"] * n_rois,
        "region": ["Unknown"] * n_rois,
        "exact_region": ["Unknown"] * n_rois,
        "Yeo_7network": ["Unknown"] * n_rois,
        "Yeo_17network": ["Unknown"] * n_rois,
        "Yeo_7network_name": ["Unknown"] * n_rois,
        "Yeo_17network_name": ["Unknown"] * n_rois,
    })


def infer_mapping_xlsx(cfg) -> Path | None:
    mapping = getattr(cfg.paths, "roi_mapping_xlsx", "")
    if mapping is None or str(mapping).strip() == "":
        return None
    return Path(mapping)


def compute_state_roi_mapping(
    cfg,
    feature_run_name: str,
    k_value: int,
    top_n: int = 15,
):
    representation_dir = Path(cfg.paths.symbolic_output_root) / feature_run_name
    hmm_result_dir = Path(cfg.paths.hmm_output_root) / feature_run_name / f"K_{k_value}"

    representation_path = representation_dir / "representation_outputs.npz"
    hmm_results_path = hmm_result_dir / "hmm_results.npz"

    output_dir = ensure_dir(hmm_result_dir / "state_roi_mapping")

    if not representation_path.exists():
        raise FileNotFoundError(f"Missing representation file: {representation_path}")

    if not hmm_results_path.exists():
        raise FileNotFoundError(f"Missing HMM result file: {hmm_results_path}")

    hmm = np.load(hmm_results_path, allow_pickle=True)
    rep = np.load(representation_path, allow_pickle=True)

    state_sequence = hmm["state_sequence"]
    activation_code = rep["activation_code"]
    trend_code = rep["trend_code"]

    n_subjects, n_rois, n_timepoints = activation_code.shape
    K = hmm["FO"].shape[1]

    expected_len = n_subjects * n_rois * n_timepoints
    if len(state_sequence) != expected_len:
        raise ValueError(
            f"state_sequence length mismatch: got {len(state_sequence)}, "
            f"expected {expected_len}. Please check flattening order."
        )

    mapping_xlsx = infer_mapping_xlsx(cfg)
    roi_df = load_roi_mapping(n_rois=n_rois, mapping_xlsx=mapping_xlsx)

    state_3d = state_sequence.reshape(n_subjects, n_rois, n_timepoints)

    rows = []

    for state_id in range(K):
        state_mask = state_3d == state_id

        for roi in range(n_rois):
            roi_mask = state_mask[:, roi, :]
            n_points = int(roi_mask.sum())
            occupancy = float(roi_mask.mean())

            if n_points > 0:
                act_values = activation_code[:, roi, :][roi_mask]
                trend_values = trend_code[:, roi, :][roi_mask]

                mean_activation = float(np.mean(act_values))
                mean_trend = float(np.mean(trend_values))

                abs_mean_activation = abs(mean_activation)
                abs_mean_trend = abs(mean_trend)

                positive_activation_ratio = float(np.mean(act_values == 1))
                negative_activation_ratio = float(np.mean(act_values == -1))
                zero_activation_ratio = float(np.mean(act_values == 0))

                upward_trend_ratio = float(np.mean(trend_values == 1))
                downward_trend_ratio = float(np.mean(trend_values == -1))
                zero_trend_ratio = float(np.mean(trend_values == 0))
            else:
                mean_activation = np.nan
                mean_trend = np.nan
                abs_mean_activation = np.nan
                abs_mean_trend = np.nan

                positive_activation_ratio = np.nan
                negative_activation_ratio = np.nan
                zero_activation_ratio = np.nan

                upward_trend_ratio = np.nan
                downward_trend_ratio = np.nan
                zero_trend_ratio = np.nan

            label_row = roi_df.loc[roi_df["roi_index"] == roi]

            if len(label_row) == 0:
                label_info = {
                    "Label": roi + 1,
                    "Gyrus": f"ROI_{roi}",
                    "subregion_name": "Unknown",
                    "region": "Unknown",
                    "exact_region": "Unknown",
                    "Yeo_7network": "Unknown",
                    "Yeo_17network": "Unknown",
                    "Yeo_7network_name": "Unknown",
                    "Yeo_17network_name": "Unknown",
                }
            else:
                label_info = label_row.iloc[0].to_dict()

            rows.append({
                "state": state_id,
                "roi_index": roi,
                "Label": label_info["Label"],
                "Gyrus": label_info["Gyrus"],
                "subregion_name": label_info["subregion_name"],
                "region": label_info["region"],
                "exact_region": label_info["exact_region"],
                "Yeo_7network": label_info["Yeo_7network"],
                "Yeo_17network": label_info["Yeo_17network"],
                "Yeo_7network_name": label_info["Yeo_7network_name"],
                "Yeo_17network_name": label_info["Yeo_17network_name"],

                "state_roi_occupancy": occupancy,
                "n_points": n_points,

                "mean_activation": mean_activation,
                "abs_mean_activation": abs_mean_activation,
                "positive_activation_ratio": positive_activation_ratio,
                "negative_activation_ratio": negative_activation_ratio,
                "zero_activation_ratio": zero_activation_ratio,

                "mean_trend": mean_trend,
                "abs_mean_trend": abs_mean_trend,
                "upward_trend_ratio": upward_trend_ratio,
                "downward_trend_ratio": downward_trend_ratio,
                "zero_trend_ratio": zero_trend_ratio,
            })

    full_df = pd.DataFrame(rows)

    full_df.to_csv(output_dir / "state_roi_mapping_full.csv", index=False)

    top_occ_rows = []
    top_act_rows = []
    top_trend_rows = []

    for state_id in range(K):
        tmp = full_df[full_df["state"] == state_id].copy()

        top_occ = tmp.sort_values(
            ["state_roi_occupancy", "abs_mean_activation", "abs_mean_trend"],
            ascending=[False, False, False],
        ).head(top_n)
        top_occ.insert(1, "rank", range(1, len(top_occ) + 1))
        top_occ_rows.append(top_occ)

        top_act = tmp.sort_values(
            ["abs_mean_activation", "state_roi_occupancy"],
            ascending=[False, False],
        ).head(top_n)
        top_act.insert(1, "rank", range(1, len(top_act) + 1))
        top_act_rows.append(top_act)

        top_trend = tmp.sort_values(
            ["abs_mean_trend", "state_roi_occupancy"],
            ascending=[False, False],
        ).head(top_n)
        top_trend.insert(1, "rank", range(1, len(top_trend) + 1))
        top_trend_rows.append(top_trend)

    top_occ_df = pd.concat(top_occ_rows, axis=0)
    top_act_df = pd.concat(top_act_rows, axis=0)
    top_trend_df = pd.concat(top_trend_rows, axis=0)

    network_summary = (
        full_df
        .groupby(["state", "Yeo_7network", "Yeo_7network_name"], as_index=False)
        .agg(
            mean_state_roi_occupancy=("state_roi_occupancy", "mean"),
            max_state_roi_occupancy=("state_roi_occupancy", "max"),
            mean_abs_activation=("abs_mean_activation", "mean"),
            mean_abs_trend=("abs_mean_trend", "mean"),
            mean_activation=("mean_activation", "mean"),
            mean_trend=("mean_trend", "mean"),
            n_rois=("roi_index", "count"),
        )
        .sort_values(["state", "mean_state_roi_occupancy"], ascending=[True, False])
    )

    gyrus_summary = (
        full_df
        .groupby(["state", "Gyrus"], as_index=False)
        .agg(
            mean_state_roi_occupancy=("state_roi_occupancy", "mean"),
            max_state_roi_occupancy=("state_roi_occupancy", "max"),
            mean_abs_activation=("abs_mean_activation", "mean"),
            mean_abs_trend=("abs_mean_trend", "mean"),
            mean_activation=("mean_activation", "mean"),
            mean_trend=("mean_trend", "mean"),
            n_rois=("roi_index", "count"),
        )
        .sort_values(["state", "mean_state_roi_occupancy"], ascending=[True, False])
    )

    top_occ_df.to_csv(output_dir / "top_occupancy_rois_per_state.csv", index=False)
    top_act_df.to_csv(output_dir / "top_activation_rois_per_state.csv", index=False)
    top_trend_df.to_csv(output_dir / "top_trend_rois_per_state.csv", index=False)
    network_summary.to_csv(output_dir / "state_yeo7network_summary.csv", index=False)
    gyrus_summary.to_csv(output_dir / "state_gyrus_summary.csv", index=False)

    excel_path = output_dir / "state_roi_mapping_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        full_df.to_excel(writer, index=False, sheet_name="full_roi_mapping")
        top_occ_df.to_excel(writer, index=False, sheet_name="top_occupancy")
        top_act_df.to_excel(writer, index=False, sheet_name="top_activation")
        top_trend_df.to_excel(writer, index=False, sheet_name="top_trend")
        network_summary.to_excel(writer, index=False, sheet_name="yeo7_network_summary")
        gyrus_summary.to_excel(writer, index=False, sheet_name="gyrus_summary")

    print("Done.")
    print(f"Mapping file: {mapping_xlsx}")
    print(f"Output dir: {output_dir}")
    print(f"Excel summary: {excel_path}")

    return full_df, top_occ_df, top_act_df, top_trend_df, network_summary, gyrus_summary


def main():
    cfg = load_experiment_config(CONFIG_PATH)

    compute_state_roi_mapping(
        cfg=cfg,
        feature_run_name=FEATURE_RUN_NAME,
        k_value=K_VALUE,
        top_n=TOP_N,
    )


if __name__ == "__main__":
    main()
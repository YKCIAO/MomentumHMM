from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from config import load_experiment_config
from utils.io_utils import ensure_dir


# =========================
# User settings
# =========================
CONFIG_PATH = "configs/experiment_config.json"

# 选择你要映射的具体 HMM run
FEATURE_RUN_NAME = "act_0.7000__trend_0.8000__a_1.0000__b_1.0000"
K_VALUE = 6

TOP_N = 15


def load_roi_labels(n_rois: int, roi_labels_csv: Path | None = None) -> pd.DataFrame:
    if roi_labels_csv is not None and roi_labels_csv.exists():
        df = pd.read_csv(roi_labels_csv)

        if "roi_index" not in df.columns:
            df["roi_index"] = np.arange(len(df))

        if "roi_name" not in df.columns:
            df["roi_name"] = [f"ROI_{i}" for i in range(len(df))]

        if "network" not in df.columns:
            df["network"] = "Unknown"

        return df[["roi_index", "roi_name", "network"]]

    return pd.DataFrame({
        "roi_index": np.arange(n_rois),
        "roi_name": [f"ROI_{i}" for i in range(n_rois)],
        "network": ["Unknown"] * n_rois,
    })


def infer_representation_dir(cfg, feature_run_name: str) -> Path:
    return Path(cfg.paths.symbolic_output_root) / feature_run_name


def infer_hmm_result_dir(cfg, feature_run_name: str, k_value: int) -> Path:
    return Path(cfg.paths.hmm_output_root) / feature_run_name / f"K_{k_value}"


def infer_roi_labels_csv(cfg) -> Path | None:
    """
    Optional config support.

    If you later add this field to config:
        "roi_labels_csv": "../data/roi_labels.csv"

    under paths, this function will use it automatically.
    """
    roi_labels = getattr(cfg.paths, "roi_labels_csv", None)
    if roi_labels is None or str(roi_labels).strip() == "":
        return None
    return Path(roi_labels)


def compute_state_roi_mapping(
    cfg,
    feature_run_name: str,
    k_value: int,
    top_n: int = 15,
):
    representation_dir = infer_representation_dir(cfg, feature_run_name)
    hmm_result_dir = infer_hmm_result_dir(cfg, feature_run_name, k_value)

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

    if K != k_value:
        print(f"[Warning] K from file = {K}, but requested K = {k_value}")

    expected_len = n_subjects * n_rois * n_timepoints
    if len(state_sequence) != expected_len:
        raise ValueError(
            f"state_sequence length mismatch: got {len(state_sequence)}, "
            f"expected {expected_len}. Please check flattening order."
        )

    roi_labels_csv = infer_roi_labels_csv(cfg)
    roi_df = load_roi_labels(n_rois, roi_labels_csv)

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
                roi_name = f"ROI_{roi}"
                network = "Unknown"
            else:
                roi_name = label_row.iloc[0]["roi_name"]
                network = label_row.iloc[0]["network"]

            rows.append({
                "state": state_id,
                "roi_index": roi,
                "roi_name": roi_name,
                "network": network,

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
        .groupby(["state", "network"], as_index=False)
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
    network_summary.to_csv(output_dir / "state_network_summary.csv", index=False)

    excel_path = output_dir / "state_roi_mapping_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        full_df.to_excel(writer, index=False, sheet_name="full_roi_mapping")
        top_occ_df.to_excel(writer, index=False, sheet_name="top_occupancy")
        top_act_df.to_excel(writer, index=False, sheet_name="top_activation")
        top_trend_df.to_excel(writer, index=False, sheet_name="top_trend")
        network_summary.to_excel(writer, index=False, sheet_name="network_summary")

    print("Done.")
    print(f"HMM result: {hmm_results_path}")
    print(f"Representation: {representation_path}")
    print(f"Output dir: {output_dir}")
    print(f"Excel summary: {excel_path}")

    return full_df, top_occ_df, top_act_df, top_trend_df, network_summary


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
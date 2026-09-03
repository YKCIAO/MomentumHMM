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


# ============================================================
# 1. ROI mapping
# ============================================================

def load_roi_mapping(n_rois: int, mapping_xlsx: Path | None = None) -> pd.DataFrame:
    """
    Load ROI mapping table.

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


# ============================================================
# 2. Helper functions
# ============================================================

def get_npz_scalar_string(npz, key: str, default: str = "unknown") -> str:
    if key not in npz.files:
        return default
    value = np.asarray(npz[key]).ravel()
    if len(value) == 0:
        return default
    return str(value[0])


def get_feature_names(hmm, n_features: int) -> list[str]:
    if "feature_names" in hmm.files:
        names = [str(x) for x in np.asarray(hmm["feature_names"]).ravel()]
        if len(names) == n_features:
            return names

    return [f"feature_{i}" for i in range(n_features)]


def find_3d_array(rep, candidates: list[str]) -> tuple[str, np.ndarray] | None:
    """
    Find a 3D array in representation_outputs.npz by candidate keys.
    Expected shape: subjects × rois × timepoints.
    """
    for key in candidates:
        if key in rep.files:
            arr = np.asarray(rep[key])
            if arr.ndim == 3:
                return key, arr

    return None


def infer_subject_roi_time_shape(rep, hmm) -> tuple[int, int, int]:
    """
    Infer (n_subjects, n_rois, n_timepoints).

    Priority:
    1. activation_code / trend_code if present
    2. any 3D array in representation_outputs.npz
    3. use lengths + X is not enough to infer n_rois, so raise error
    """

    preferred_keys = [
        "activation_code",
        "trend_code",
        "activation",
        "trend",
        "activation_value",
        "trend_value",
        "activation_values",
        "trend_values",
        "activation_feature",
        "trend_feature",
    ]

    for key in preferred_keys:
        if key in rep.files:
            arr = np.asarray(rep[key])
            if arr.ndim == 3:
                return arr.shape

    for key in rep.files:
        arr = np.asarray(rep[key])
        if arr.ndim == 3:
            print(f"[Info] Inferred subject/ROI/time shape from rep['{key}']: {arr.shape}")
            return arr.shape

    raise ValueError(
        "Cannot infer (n_subjects, n_rois, n_timepoints). "
        "Please ensure representation_outputs.npz contains at least one 3D array "
        "with shape subjects × rois × timepoints."
    )


def reshape_state_sequence(state_sequence, n_subjects: int, n_rois: int, n_timepoints: int) -> np.ndarray:
    expected_len = n_subjects * n_rois * n_timepoints

    if len(state_sequence) != expected_len:
        raise ValueError(
            f"state_sequence length mismatch: got {len(state_sequence)}, "
            f"expected {expected_len}. Please check flattening order."
        )

    return state_sequence.reshape(n_subjects, n_rois, n_timepoints)


def build_feature_4d_from_categorical(rep) -> tuple[np.ndarray, list[str], dict]:
    """
    Build feature array for categorical branch.

    Output:
        feature_4d: subjects × rois × timepoints × features
        feature_names: usually ["activation", "trend"]
        extra: contains activation_code and trend_code for ratio calculation
    """
    if "activation_code" not in rep.files:
        raise KeyError("categorical branch requires rep['activation_code'].")

    activation_code = np.asarray(rep["activation_code"])

    if "trend_code" in rep.files:
        trend_code = np.asarray(rep["trend_code"])
    else:
        trend_code = np.zeros_like(activation_code)

    if activation_code.shape != trend_code.shape:
        raise ValueError(
            f"activation_code and trend_code shape mismatch: "
            f"{activation_code.shape} vs {trend_code.shape}"
        )

    feature_4d = np.stack([activation_code, trend_code], axis=-1).astype(float)

    extra = {
        "activation_code": activation_code,
        "trend_code": trend_code,
    }

    return feature_4d, ["activation", "trend"], extra


def build_feature_4d_from_gaussian(rep, hmm, n_subjects: int, n_rois: int, n_timepoints: int) -> tuple[np.ndarray, list[str], dict]:
    """
    Build feature array for gaussian branch.

    Priority:
    1. Use 3D activation/trend arrays from representation_outputs.npz if available.
    2. Use hmm_results.npz['X'] and reshape to subjects × rois × timepoints × features.

    Gaussian branch uses continuous features, so activation/trend ratios are not computed.
    """

    # Try to find continuous activation/trend arrays in representation_outputs.npz
    activation_candidates = [
        "activation",
        "activation_value",
        "activation_values",
        "activation_feature",
        "activation_continuous",
        "activation_z",
        "activation_ts",
        "activation_signal",
    ]

    trend_candidates = [
        "trend",
        "trend_value",
        "trend_values",
        "trend_feature",
        "trend_continuous",
        "trend_z",
        "trend_ts",
        "trend_signal",
    ]

    act_found = find_3d_array(rep, activation_candidates)
    trend_found = find_3d_array(rep, trend_candidates)

    arrays = []
    names = []

    if act_found is not None:
        key, arr = act_found
        arrays.append(arr.astype(float))
        names.append("activation")
        print(f"[Info] Gaussian feature loaded from rep['{key}'] as activation.")

    if trend_found is not None:
        key, arr = trend_found
        arrays.append(arr.astype(float))
        names.append("trend")
        print(f"[Info] Gaussian feature loaded from rep['{key}'] as trend.")

    if len(arrays) > 0:
        for arr in arrays:
            if arr.shape != (n_subjects, n_rois, n_timepoints):
                raise ValueError(
                    f"Gaussian feature shape mismatch: got {arr.shape}, "
                    f"expected {(n_subjects, n_rois, n_timepoints)}"
                )

        feature_4d = np.stack(arrays, axis=-1)
        return feature_4d, names, {}

    # Fallback: use X from hmm_results.npz
    if "X" not in hmm.files:
        raise KeyError(
            "Gaussian branch could not find continuous 3D features in representation_outputs.npz "
            "and hmm_results.npz does not contain 'X'."
        )

    X = np.asarray(hmm["X"])
    if X.ndim == 1:
        X = X[:, None]

    n_obs, n_features = X.shape
    expected_len = n_subjects * n_rois * n_timepoints

    if n_obs != expected_len:
        raise ValueError(
            f"hmm['X'] length mismatch: got {n_obs}, expected {expected_len}. "
            "Cannot reshape X back to subjects × rois × timepoints × features."
        )

    feature_names = get_feature_names(hmm, n_features)

    feature_4d = X.reshape(n_subjects, n_rois, n_timepoints, n_features).astype(float)

    print(
        "[Info] Gaussian features loaded from hmm['X'] and reshaped to "
        f"{feature_4d.shape} with feature_names={feature_names}"
    )

    return feature_4d, feature_names, {}


def identify_activation_trend_indices(feature_names: list[str]) -> tuple[int | None, int | None]:
    """
    Find activation and trend feature indices from feature names.
    """
    lower_names = [x.lower() for x in feature_names]

    activation_idx = None
    trend_idx = None

    for i, name in enumerate(lower_names):
        if activation_idx is None and ("act" in name or "activation" in name):
            activation_idx = i
        if trend_idx is None and ("trend" in name or "slope" in name):
            trend_idx = i

    return activation_idx, trend_idx


def safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    return float(np.nanmean(x))


def safe_abs_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    return abs(float(np.nanmean(x)))


# ============================================================
# 3. Main computation
# ============================================================

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

    emission_type = get_npz_scalar_string(hmm, "emission_type", default="unknown").lower()
    print(f"[Info] Detected emission_type = {emission_type}")

    state_sequence = np.asarray(hmm["state_sequence"])
    K = int(hmm["FO"].shape[1])

    n_subjects, n_rois, n_timepoints = infer_subject_roi_time_shape(rep, hmm)
    state_3d = reshape_state_sequence(state_sequence, n_subjects, n_rois, n_timepoints)

    if emission_type == "categorical":
        feature_4d, feature_names, extra = build_feature_4d_from_categorical(rep)

    elif emission_type == "gaussian":
        feature_4d, feature_names, extra = build_feature_4d_from_gaussian(
            rep=rep,
            hmm=hmm,
            n_subjects=n_subjects,
            n_rois=n_rois,
            n_timepoints=n_timepoints,
        )

    else:
        # Robust fallback
        if "activation_code" in rep.files:
            print("[Warning] Unknown emission_type, but activation_code found. Treating as categorical.")
            feature_4d, feature_names, extra = build_feature_4d_from_categorical(rep)
            emission_type = "categorical"
        else:
            print("[Warning] Unknown emission_type. Trying Gaussian-style feature loading.")
            feature_4d, feature_names, extra = build_feature_4d_from_gaussian(
                rep=rep,
                hmm=hmm,
                n_subjects=n_subjects,
                n_rois=n_rois,
                n_timepoints=n_timepoints,
            )
            emission_type = "gaussian"

    if feature_4d.shape[:3] != (n_subjects, n_rois, n_timepoints):
        raise ValueError(
            f"feature_4d shape mismatch: got {feature_4d.shape[:3]}, "
            f"expected {(n_subjects, n_rois, n_timepoints)}"
        )

    n_features = feature_4d.shape[-1]

    activation_idx, trend_idx = identify_activation_trend_indices(feature_names)

    mapping_xlsx = infer_mapping_xlsx(cfg)
    roi_df = load_roi_mapping(n_rois=n_rois, mapping_xlsx=mapping_xlsx)

    rows = []

    for state_id in range(K):
        state_mask = state_3d == state_id

        for roi in range(n_rois):
            roi_mask = state_mask[:, roi, :]
            n_points = int(roi_mask.sum())
            occupancy = float(roi_mask.mean())

            roi_feature_values = feature_4d[:, roi, :, :][roi_mask]

            feature_stats = {}

            if n_points > 0:
                for f_idx, f_name in enumerate(feature_names):
                    values = roi_feature_values[:, f_idx]
                    clean_name = str(f_name).replace(" ", "_")

                    feature_stats[f"mean_{clean_name}"] = safe_mean(values)
                    feature_stats[f"abs_mean_{clean_name}"] = safe_abs_mean(values)
                    feature_stats[f"std_{clean_name}"] = float(np.nanstd(values))

                # Backward-compatible columns
                if activation_idx is not None:
                    act_values = roi_feature_values[:, activation_idx]
                    mean_activation = safe_mean(act_values)
                    abs_mean_activation = safe_abs_mean(act_values)
                else:
                    mean_activation = np.nan
                    abs_mean_activation = np.nan

                if trend_idx is not None:
                    trend_values = roi_feature_values[:, trend_idx]
                    mean_trend = safe_mean(trend_values)
                    abs_mean_trend = safe_abs_mean(trend_values)
                else:
                    mean_trend = np.nan
                    abs_mean_trend = np.nan

                if emission_type == "categorical":
                    activation_code = extra["activation_code"]
                    trend_code = extra["trend_code"]

                    act_values_cat = activation_code[:, roi, :][roi_mask]
                    trend_values_cat = trend_code[:, roi, :][roi_mask]

                    positive_activation_ratio = float(np.mean(act_values_cat == 1))
                    negative_activation_ratio = float(np.mean(act_values_cat == -1))
                    zero_activation_ratio = float(np.mean(act_values_cat == 0))

                    upward_trend_ratio = float(np.mean(trend_values_cat == 1))
                    downward_trend_ratio = float(np.mean(trend_values_cat == -1))
                    zero_trend_ratio = float(np.mean(trend_values_cat == 0))
                else:
                    positive_activation_ratio = np.nan
                    negative_activation_ratio = np.nan
                    zero_activation_ratio = np.nan

                    upward_trend_ratio = np.nan
                    downward_trend_ratio = np.nan
                    zero_trend_ratio = np.nan

            else:
                for f_name in feature_names:
                    clean_name = str(f_name).replace(" ", "_")
                    feature_stats[f"mean_{clean_name}"] = np.nan
                    feature_stats[f"abs_mean_{clean_name}"] = np.nan
                    feature_stats[f"std_{clean_name}"] = np.nan

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

            row = {
                "emission_type": emission_type,
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

                # backward-compatible core columns
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
            }

            row.update(feature_stats)
            rows.append(row)

    full_df = pd.DataFrame(rows)

    full_df.to_csv(output_dir / "state_roi_mapping_full.csv", index=False)

    # ========================================================
    # Top ROI tables
    # ========================================================

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

    # ========================================================
    # Summary tables
    # ========================================================

    agg_dict = {
        "mean_state_roi_occupancy": ("state_roi_occupancy", "mean"),
        "max_state_roi_occupancy": ("state_roi_occupancy", "max"),
        "mean_abs_activation": ("abs_mean_activation", "mean"),
        "mean_abs_trend": ("abs_mean_trend", "mean"),
        "mean_activation": ("mean_activation", "mean"),
        "mean_trend": ("mean_trend", "mean"),
        "n_rois": ("roi_index", "count"),
    }

    for f_name in feature_names:
        clean_name = str(f_name).replace(" ", "_")
        agg_dict[f"mean_{clean_name}"] = (f"mean_{clean_name}", "mean")
        agg_dict[f"abs_mean_{clean_name}"] = (f"abs_mean_{clean_name}", "mean")
        agg_dict[f"std_{clean_name}"] = (f"std_{clean_name}", "mean")

    if emission_type == "categorical":
        agg_dict.update({
            "mean_zero_activation_ratio": ("zero_activation_ratio", "mean"),
            "mean_positive_activation_ratio": ("positive_activation_ratio", "mean"),
            "mean_negative_activation_ratio": ("negative_activation_ratio", "mean"),
            "mean_zero_trend_ratio": ("zero_trend_ratio", "mean"),
            "mean_upward_trend_ratio": ("upward_trend_ratio", "mean"),
            "mean_downward_trend_ratio": ("downward_trend_ratio", "mean"),
        })

    network_summary = (
        full_df
        .groupby(["state", "Yeo_7network", "Yeo_7network_name"], as_index=False)
        .agg(**agg_dict)
        .sort_values(["state", "mean_state_roi_occupancy"], ascending=[True, False])
    )

    gyrus_summary = (
        full_df
        .groupby(["state", "Gyrus"], as_index=False)
        .agg(**agg_dict)
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
    print(f"Emission type: {emission_type}")
    print(f"Feature names: {feature_names}")
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
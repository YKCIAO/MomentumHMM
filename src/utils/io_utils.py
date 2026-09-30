from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(path)


def save_npz(path: str | Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kwargs)


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_npz(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_score_excel(
    scored_runs: list[dict],
    output_path: Path,
) -> None:
    """
    Save human-readable score ranking and state-level age
    associations to an Excel workbook.
    """

    ranking_rows = []
    age_rows = []

    for rank, run in enumerate(scored_runs, start=1):

        raw = run["raw_metrics"]
        norm = run["normalized_metrics"]

        ranking_rows.append(
            {
                "Rank": rank,

                "Deviation_threshold":
                    run["deviation_threshold"],

                "Momentum_threshold":
                    run["momentum_threshold"],

                "K":
                    run["n_hidden_states"],

                # Main scores
                "Final_score":
                    run["final_score"],

                "Quality_score":
                    run["quality_score"],

                "Age_score":
                    run["age_score"],

                # Main quality components
                "State_usage_entropy_raw":
                    raw["state_usage_entropy"],

                "State_usage_entropy_norm":
                    norm["state_usage_entropy"],

                "Non_fragmented_fraction_raw":
                    raw["non_fragmented_state_fraction"],

                "Non_fragmented_fraction_norm":
                    norm["non_fragmented_state_fraction"],

                # Guardrail
                "Effective_state_fraction":
                    raw["effective_state_fraction"],

                # Symbolic diagnostics
                "Observation_entropy":
                    raw["observation_entropy"],

                "Used_observation_fraction":
                    raw["used_observation_fraction"],

                # Transition diagnostics
                "Mean_self_transition":
                    raw["mean_self_transition"],

                "Transition_entropy":
                    raw["transition_entropy"],

                "Run_directory":
                    run["run_dir"],
            }
        )

        age_details = run["age_details"]

        fo_r = age_details["fo_age_r"]
        mdt_r = age_details["mdt_age_r"]

        for state_idx in range(
            run["n_hidden_states"]
        ):

            age_rows.append(
                {
                    "Rank": rank,

                    "Deviation_threshold":
                        run["deviation_threshold"],

                    "Momentum_threshold":
                        run["momentum_threshold"],

                    "K":
                        run["n_hidden_states"],

                    "State":
                        state_idx,

                    "FO_age_r":
                        fo_r[state_idx],

                    "MDT_age_r":
                        mdt_r[state_idx],

                    "Abs_FO_age_r":
                        abs(fo_r[state_idx]),

                    "Abs_MDT_age_r":
                        abs(mdt_r[state_idx]),

                    "Final_score":
                        run["final_score"],
                }
            )

    ranking_df = pd.DataFrame(
        ranking_rows
    )

    age_df = pd.DataFrame(
        age_rows
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        ranking_df.to_excel(
            writer,
            sheet_name="Ranking",
            index=False,
        )

        age_df.to_excel(
            writer,
            sheet_name="AgeDetails",
            index=False,
        )

        # ----------------------------
        # Human-readable formatting
        # ----------------------------

        ranking_ws = writer.sheets["Ranking"]
        age_ws = writer.sheets["AgeDetails"]

        ranking_ws.freeze_panes = "A2"
        age_ws.freeze_panes = "A2"

        ranking_ws.auto_filter.ref = (
            ranking_ws.dimensions
        )

        age_ws.auto_filter.ref = (
            age_ws.dimensions
        )

        # Adjust column widths
        for ws in [ranking_ws, age_ws]:

            for column_cells in ws.columns:

                max_length = 0

                for cell in column_cells:

                    value = cell.value

                    if value is None:
                        continue

                    max_length = max(
                        max_length,
                        len(str(value)),
                    )

                column_letter = (
                    column_cells[0].column_letter
                )

                ws.column_dimensions[
                    column_letter
                ].width = min(
                    max_length + 2,
                    28,
                )
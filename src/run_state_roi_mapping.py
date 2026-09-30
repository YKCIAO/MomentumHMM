from __future__ import annotations

import json
from pathlib import Path

from config import (
    load_experiment_config,
)

from mapping.state_roi_mapping import (
    compute_state_roi_mapping,
)


CONFIG_PATH = (
    "configs/experiment_config.json"
)

RANK = 1
TOP_N = 15

def load_ranked_run(
    score_json: Path,
    rank: int = 1,
) -> dict:

    with open(
        score_json,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    runs = data["runs"]

    if rank < 1 or rank > len(runs):

        raise ValueError(
            f"Rank {rank} is outside "
            f"1..{len(runs)}"
        )

    return runs[
        rank - 1
    ]

def main():

    cfg = load_experiment_config(
        CONFIG_PATH
    )

    score_json = (
        Path(
            cfg.paths.score_output_root
        )
        / "score_ranking.json"
    )

    run = load_ranked_run(
        score_json,
        rank=RANK,
    )

    run_dir = Path(
        run["run_dir"]
    )

    symbolic_run_name = (
        run_dir.parent.name
    )

    k_value = int(
        run[
            "n_hidden_states"
        ]
    )

    print(
        f"Mapping rank {RANK}: "
        f"{symbolic_run_name}, "
        f"K={k_value}"
    )

    result = compute_state_roi_mapping(
        cfg=cfg,
        symbolic_run_name=
            symbolic_run_name,
        k_value=k_value,
        top_n=TOP_N,
    )

    print(
        "State-to-ROI mapping "
        "completed."
    )

    print(
        f"Excel: "
        f"{result['excel_path']}"
    )


if __name__ == "__main__":
    main()


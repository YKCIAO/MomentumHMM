from __future__ import annotations

from config import load_experiment_config
from evaluation.runner import score_all_hmm_runs


def main():
    cfg = load_experiment_config("configs/experiment_config.json")
    score_all_hmm_runs(cfg)


if __name__ == "__main__":
    main()
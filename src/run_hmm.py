from __future__ import annotations

from config import load_experiment_config
from hmm.runner import fit_all_hmm_runs


def main():
    cfg = load_experiment_config("configs/experiment_config.json")
    fit_all_hmm_runs(cfg)


if __name__ == "__main__":
    main()
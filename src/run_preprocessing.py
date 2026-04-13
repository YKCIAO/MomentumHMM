from __future__ import annotations

from config import load_experiment_config
from pipeline.preprocessing_pipeline import exhaustive_symbolic_search
from utils.io_utils import load_npy


def main():
    cfg = load_experiment_config("configs/experiment_config.json")
    data = load_npy(cfg.paths.input_data)
    exhaustive_symbolic_search(data=data, cfg=cfg)


if __name__ == "__main__":
    main()
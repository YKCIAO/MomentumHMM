from __future__ import annotations

from datetime import datetime
from pathlib import Path
import traceback

from config import load_experiment_config
from pipeline.preprocessing_pipeline import exhaustive_representation_search
from utils.io_utils import load_npy


def log_step(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def main():
    try:
        log_step("Step 1/5: Starting preprocessing pipeline")
        log_step(f"Current working directory: {Path.cwd()}")

        config_path = "configs/experiment_config.json"
        log_step(f"Step 2/5: Loading config from: {config_path}")
        cfg = load_experiment_config(config_path)
        log_step("Config loaded successfully")

        log_step(f"Resolved input data path: {cfg.paths.input_data}")
        input_path = Path(cfg.paths.input_data)
        log_step(f"Input file exists: {input_path.exists()}")

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        log_step("Step 3/5: Loading timeseries data")
        data = load_npy(cfg.paths.input_data)
        log_step(f"Data loaded successfully, shape = {data.shape}, dtype = {data.dtype}")

        log_step("Step 4/5: Running exhaustive 2D representation search")
        exhaustive_representation_search(data=data, cfg=cfg)
        log_step("2D representation search finished successfully")

        log_step("Step 5/5: Pipeline completed")

    except Exception as e:
        log_step("Pipeline failed with an error")
        log_step(f"Error type: {type(e).__name__}")
        log_step(f"Error message: {e}")
        print(traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()
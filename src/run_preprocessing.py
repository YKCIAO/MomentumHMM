from __future__ import annotations

from datetime import datetime
from pathlib import Path
import traceback

from config import load_experiment_config
from pipeline.preprocessing_pipeline import exhaustive_symbolic_search
from utils.io_utils import load_npy


def log_step(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def main():
    try:
        log_step("Step 1/5: Starting preprocessing pipeline")

        # 当前工作目录
        log_step(f"Current working directory: {Path.cwd()}")

        # 配置文件路径
        config_path = "configs/experiment_config.json"
        log_step(f"Step 2/5: Loading config from: {config_path}")
        cfg = load_experiment_config(config_path)
        log_step("Config loaded successfully")

        # 输入数据路径
        log_step(f"Resolved input data path: {cfg.paths.input_data}")

        # 检查文件是否存在
        input_path = Path(cfg.paths.input_data)
        log_step(f"Input file exists: {input_path.exists()}")
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # 加载数据
        log_step("Step 3/5: Loading timeseries data")
        data = load_npy(cfg.paths.input_data)
        log_step(f"Data loaded successfully, shape = {data.shape}, dtype = {data.dtype}")

        # 运行主流程
        log_step("Step 4/5: Running exhaustive symbolic search")
        exhaustive_symbolic_search(data=data, cfg=cfg)
        log_step("Symbolic search finished successfully")

        log_step("Step 5/5: Pipeline completed")

    except Exception as e:
        log_step("Pipeline failed with an error")
        log_step(f"Error type: {type(e).__name__}")
        log_step(f"Error message: {e}")
        print(traceback.format_exc(), flush=True)
        raise


if __name__ == "__main__":
    main()
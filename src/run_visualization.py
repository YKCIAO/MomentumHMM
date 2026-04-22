from __future__ import annotations

from datetime import datetime

from src.config import load_experiment_config
from src.visualization.runner import visualize_all


def log_step(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def main():
    log_step("Step 1/2: Loading config")
    cfg = load_experiment_config("configs/experiment_config.json")

    log_step("Step 2/2: Running visualization")
    visualize_all(cfg)

    log_step("Finished successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_step(f"FAILED: {repr(e)}")
        raise
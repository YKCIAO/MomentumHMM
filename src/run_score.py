from __future__ import annotations

from datetime import datetime

from config import load_experiment_config
from evaluation.runner import evaluate_all_hmm_runs
from visualization.score_plots import plot_score_outputs


def log_step(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def main():
    log_step("Step 1/3: Loading config")
    cfg = load_experiment_config("configs/experiment_config.json")

    log_step("Step 2/3: Evaluating all HMM runs")
    ranking = evaluate_all_hmm_runs(cfg)

    log_step("Step 3/3: Plotting score outputs")
    plot_score_outputs(cfg, ranking)

    log_step("Scoring finished successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_step(f"FAILED: {repr(e)}")
        raise
from __future__ import annotations
from datetime import datetime
from config import load_experiment_config
from hmm.runner import fit_all_hmm_runs_parallel

def log_step(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)
def main():
    log_step("Step 1/2: Loading config")
    cfg = load_experiment_config("configs/experiment_config.json")
    log_step("Step 2/2: Running hmm")
    fit_all_hmm_runs_parallel(cfg, max_workers=3, save_posterior=True)


if __name__ == "__main__":
    main()
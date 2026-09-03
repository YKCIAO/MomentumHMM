from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.evaluation.metrics import (
    compute_age_effect_summary,
    compute_fragmentation_score,
    compute_state_usage_summary,
    compute_transition_summary,
    compute_final_score,
)
from src.utils.io_utils import ensure_dir, save_json


def _extract_k_from_dir(k_dir: Path) -> int:
    match = re.search(r"K_(\d+)", k_dir.name)
    if match:
        return int(match.group(1))
    return -1


def _find_hmm_result_dirs(hmm_root: Path) -> list[Path]:
    result_dirs = []
    for p in hmm_root.rglob("*"):
        if not p.is_dir():
            continue
        if (p / "hmm_results.npz").exists() and (p / "subject_metrics.csv").exists():
            result_dirs.append(p)
    return sorted(result_dirs)


def _read_meta(result_dir: Path) -> dict:
    meta_path = result_dir / "hmm_meta.json"
    if meta_path.exists():
        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _safe_scalar_from_npz(npz, key: str):
    if key not in npz.files:
        return np.nan
    arr = npz[key]
    if arr.size == 1:
        return arr.item()
    return arr


def evaluate_single_hmm_result(result_dir: Path, hmm_root: Path, top_k: int = 5) -> dict:
    hmm_path = result_dir / "hmm_results.npz"
    csv_path = result_dir / "subject_metrics.csv"

    hmm_npz = np.load(hmm_path, allow_pickle=True)
    subject_metrics = pd.read_csv(csv_path)
    meta = _read_meta(result_dir)

    age_summary = compute_age_effect_summary(subject_metrics, top_k=top_k)
    frag_summary = compute_fragmentation_score(subject_metrics)
    usage_summary = compute_state_usage_summary(subject_metrics)
    trans_summary = compute_transition_summary(hmm_npz)

    rel = result_dir.relative_to(hmm_root)
    run_name = "__".join(rel.parts)

    row = {
        "run_name": run_name,
        "feature_run": result_dir.parent.name,
        "K": _extract_k_from_dir(result_dir),
        "result_dir": str(result_dir),
        "emission_type": meta.get("emission_type", None),
        "covariance_type": meta.get("covariance_type", None),
        "logprob": meta.get("logprob", np.nan),
        "n_subjects": meta.get("n_subjects", np.nan),
        "activation_threshold": _safe_scalar_from_npz(hmm_npz, "activation_threshold"),
        "trend_threshold": _safe_scalar_from_npz(hmm_npz, "trend_threshold"),
        "alpha": _safe_scalar_from_npz(hmm_npz, "alpha"),
        "beta": _safe_scalar_from_npz(hmm_npz, "beta"),
    }

    row.update(age_summary)
    row.update(frag_summary)
    row.update(usage_summary)
    row.update(trans_summary)

    row["final_score"] = compute_final_score(row)
    return row


def evaluate_all_hmm_runs(cfg: ExperimentConfig) -> pd.DataFrame:
    hmm_root = Path(cfg.paths.hmm_output_root)
    score_root = ensure_dir(cfg.paths.score_output_root)

    result_dirs = _find_hmm_result_dirs(hmm_root)
    if len(result_dirs) == 0:
        raise FileNotFoundError(f"No HMM result folders found under: {hmm_root}")

    rows = []

    for idx, result_dir in enumerate(result_dirs, start=1):
        print(f"[Scoring] [{idx}/{len(result_dirs)}] {result_dir}", flush=True)

        try:
            row = evaluate_single_hmm_result(result_dir, hmm_root=hmm_root, top_k=5)
            rows.append(row)
        except Exception as e:
            print(f"[Scoring] FAILED: {result_dir} | {repr(e)}", flush=True)

    ranking = pd.DataFrame(rows)

    if ranking.empty:
        raise RuntimeError("No valid HMM result could be scored.")

    ranking = ranking.sort_values("final_score", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)

    csv_path = score_root / "score_ranking.csv"
    ranking.to_csv(csv_path, index=False)

    best = ranking.iloc[0].to_dict()
    save_json(score_root / "best_model.json", best)

    summary = {
        "n_models_scored": int(len(ranking)),
        "best_run_name": str(best["run_name"]),
        "best_result_dir": str(best["result_dir"]),
        "best_final_score": float(best["final_score"]),
        "best_metric": str(best["best_metric"]),
        "best_age_r": float(best["best_age_r"]),
        "best_age_p": float(best["best_age_p"]),
    }
    save_json(score_root / "score_summary.json", summary)

    print(f"[Scoring] Saved ranking to: {csv_path}", flush=True)
    print(f"[Scoring] Best model: {best['run_name']}", flush=True)

    return ranking
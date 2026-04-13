from __future__ import annotations

from pathlib import Path

from config import ExperimentConfig
from hmm.decode import decode_hmm, split_sequence_by_lengths
from hmm.metrics import compute_subject_level_metrics
from hmm.model import fit_categorical_hmm
from utils.io_utils import ensure_dir, load_npz, save_json, save_npz


def fit_all_hmm_runs(cfg: ExperimentConfig) -> None:
    symbolic_root = Path(cfg.paths.symbolic_output_root)
    hmm_root = ensure_dir(cfg.paths.hmm_output_root)

    symbolic_dirs = sorted([p for p in symbolic_root.iterdir() if p.is_dir()])

    for symbolic_dir in symbolic_dirs:
        symbolic_data = load_npz(symbolic_dir / "hmm_ready_sequence.npz")
        obs = symbolic_data["obs"]
        lengths = symbolic_data["lengths"]

        for n_hidden_states in cfg.hmm.n_hidden_states_values:
            model = fit_categorical_hmm(
                obs=obs,
                lengths=lengths,
                n_hidden_states=n_hidden_states,
                n_iter=cfg.hmm.n_iter,
                tol=cfg.hmm.tol,
                random_state=cfg.hmm.random_state,
                verbose=cfg.hmm.verbose,
            )

            decoded = decode_hmm(model, obs, lengths)
            subject_state_seqs = split_sequence_by_lengths(
                decoded["state_sequence"],
                lengths,
            )
            metrics = compute_subject_level_metrics(
                subject_state_seqs=subject_state_seqs,
                n_hidden_states=n_hidden_states,
            )

            out_dir = ensure_dir(hmm_root / symbolic_dir.name / f"K_{n_hidden_states}")

            save_npz(
                out_dir / "hmm_results.npz",
                startprob_=model.startprob_,
                transmat_=model.transmat_,
                emissionprob_=model.emissionprob_,
                state_sequence=decoded["state_sequence"],
                posterior=decoded["posterior"],
                FO=metrics["FO"],
                MDT=metrics["MDT"],
                lengths=lengths,
            )

            save_json(
                out_dir / "hmm_meta.json",
                {
                    "symbolic_source_dir": str(symbolic_dir),
                    "n_hidden_states": n_hidden_states,
                    "n_iter": cfg.hmm.n_iter,
                    "tol": cfg.hmm.tol,
                    "random_state": cfg.hmm.random_state,
                    "verbose": cfg.hmm.verbose,
                    "logprob": decoded["logprob"],
                },
            )
from __future__ import annotations

import numpy as np
from hmmlearn.hmm import CategoricalHMM, GaussianHMM


def fit_hmm(
    X: np.ndarray,
    lengths: np.ndarray,
    n_hidden_states: int,
    emission_type: str,
    covariance_type: str,
    n_iter: int,
    tol: float,
    random_state: int,
    verbose: bool,
):
    emission_type = emission_type.lower()

    if emission_type == "categorical":
        # legacy compatibility:
        # categorical input should be integer labels with shape [T, 1]
        if X.ndim == 1:
            obs = X.reshape(-1, 1)
        elif X.ndim == 2 and X.shape[1] == 1:
            obs = X
        else:
            raise ValueError(
                f"Categorical HMM expects shape [T] or [T,1], got {X.shape}"
            )

        obs = obs.astype(np.int32, copy=False)

        model = CategoricalHMM(
            n_components=n_hidden_states,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
            init_params="ste",
            params="ste",
        )
        model.fit(obs, lengths=lengths)
        return model

    if emission_type == "gaussian":
        if X.ndim != 2:
            raise ValueError(f"Gaussian HMM expects shape [T, D], got {X.shape}")

        X = X.astype(np.float64, copy=False)

        model = GaussianHMM(
            n_components=n_hidden_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
            init_params="stmc",
            params="stmc",
        )
        model.fit(X, lengths=lengths)
        return model

    raise ValueError(f"Unknown emission_type: {emission_type}")
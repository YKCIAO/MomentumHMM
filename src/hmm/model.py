from __future__ import annotations

import numpy as np
from hmmlearn.hmm import CategoricalHMM, GaussianHMM


def fit_categorical_hmm(
    obs: np.ndarray,
    lengths: np.ndarray,
    n_hidden_states: int,
    n_iter: int,
    tol: float,
    random_state: int,
    verbose: bool,
) -> CategoricalHMM:
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


def fit_gaussian_hmm(
    obs: np.ndarray,
    lengths: np.ndarray,
    n_hidden_states: int,
    n_iter: int,
    tol: float,
    random_state: int,
    verbose: bool,
    covariance_type: str = "diag",
    min_covar: float = 1e-3,
) -> GaussianHMM:
    """
    Fit a conventional Gaussian HMM on continuous observations.

    The caller is responsible for constructing obs with shape
    [n_total_timepoints, n_features] and for supplying sequence lengths so
    that subject-ROI boundaries are preserved.
    """
    model = GaussianHMM(
        n_components=n_hidden_states,
        covariance_type=covariance_type,
        min_covar=min_covar,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        verbose=verbose,
        init_params="stmc",
        params="stmc",
    )
    model.fit(obs, lengths=lengths)
    return model

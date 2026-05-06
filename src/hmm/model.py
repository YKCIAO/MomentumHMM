from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM, CategoricalHMM


def _make_sticky_transmat(K: int, self_prob: float = 0.85) -> np.ndarray:
    if K <= 1:
        return np.ones((K, K))

    off_prob = (1.0 - self_prob) / (K - 1)
    transmat = np.full((K, K), off_prob, dtype=float)
    np.fill_diagonal(transmat, self_prob)
    return transmat


def _init_means_from_unique_patterns(X: np.ndarray, K: int) -> np.ndarray:
    unique, counts = np.unique(X, axis=0, return_counts=True)

    if unique.shape[0] < K:
        raise ValueError(
            f"Too few unique symbolic patterns ({unique.shape[0]}) for K={K}. "
            "This run is structurally unstable and should be skipped."
        )

    # choose most frequent patterns first
    order = np.argsort(counts)[::-1]
    selected = unique[order[:K]].astype(float)

    return selected


def _init_diag_covars(X: np.ndarray, K: int, min_var: float = 0.05) -> np.ndarray:
    global_var = np.var(X, axis=0)

    # symbolic data often has small variance; enforce floor
    global_var = np.maximum(global_var, min_var)

    covars = np.tile(global_var[None, :], (K, 1))
    return covars


def fit_symbolic_gaussian_hmm(
    X: np.ndarray,
    lengths: np.ndarray,
    n_hidden_states: int,
    covariance_type: str = "diag",
    n_iter: int = 200,
    tol: float = 1e-4,
    random_state: int = 42,
    verbose: bool = False,
    self_prob: float = 0.85,
    min_covar: float = 1e-2,
):
    """
    Stable Gaussian HMM for symbolic 2D input.

    Important:
        X is still symbolic / ternary, but GaussianHMM is initialized using
        real observed symbolic patterns rather than random means.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"Gaussian HMM expects X with shape [T, D], got {X.shape}")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or Inf.")

    unique_patterns = np.unique(X, axis=0).shape[0]
    if unique_patterns < n_hidden_states:
        raise ValueError(
            f"K={n_hidden_states} is larger than available unique patterns "
            f"({unique_patterns}). Skip this run."
        )

    K = n_hidden_states

    model = GaussianHMM(
        n_components=K,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        verbose=verbose,
        min_covar=min_covar,
        init_params="",      # important: we manually initialize
        params="stmc",       # allow EM to update startprob, transmat, means, covars
    )

    model.startprob_ = np.ones(K, dtype=float) / K
    model.transmat_ = _make_sticky_transmat(K, self_prob=self_prob)
    model.means_ = _init_means_from_unique_patterns(X, K)

    if covariance_type == "diag":
        model.covars_ = _init_diag_covars(X, K, min_var=min_covar)
    elif covariance_type == "full":
        D = X.shape[1]
        base_cov = np.cov(X.T)
        if base_cov.ndim == 0:
            base_cov = np.eye(D) * min_covar
        base_cov = np.asarray(base_cov, dtype=float)
        base_cov = base_cov + np.eye(D) * min_covar
        model.covars_ = np.tile(base_cov[None, :, :], (K, 1, 1))
    else:
        raise ValueError(
            "For symbolic Gaussian HMM, I recommend covariance_type='diag' or 'full'."
        )

    model.fit(X, lengths=lengths)

    _check_model_validity(model)

    return model


def _check_model_validity(model) -> None:
    arrays = {
        "startprob_": model.startprob_,
        "transmat_": model.transmat_,
    }

    if hasattr(model, "means_"):
        arrays["means_"] = model.means_

    if hasattr(model, "covars_"):
        arrays["covars_"] = model.covars_

    for name, arr in arrays.items():
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or Inf.")

    if not np.isclose(np.sum(model.startprob_), 1.0):
        raise ValueError(f"startprob_ does not sum to 1: {np.sum(model.startprob_)}")

    row_sums = np.sum(model.transmat_, axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError(f"transmat_ rows do not sum to 1: {row_sums}")


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
        if X.ndim == 1:
            obs = X.reshape(-1, 1)
        elif X.ndim == 2 and X.shape[1] == 1:
            obs = X
        else:
            raise ValueError(f"Categorical HMM expects [T] or [T,1], got {X.shape}")

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
        return fit_symbolic_gaussian_hmm(
            X=X,
            lengths=lengths,
            n_hidden_states=n_hidden_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
            self_prob=0.85,
            min_covar=1e-2,
        )

    raise ValueError(f"Unknown emission_type: {emission_type}")
from __future__ import annotations

import numpy as np
from hmmlearn.hmm import CategoricalHMM


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
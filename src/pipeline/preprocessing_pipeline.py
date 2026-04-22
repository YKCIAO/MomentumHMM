from __future__ import annotations

import numpy as np

from config import ExperimentConfig
from representation.runner import exhaustive_representation_search as run_representation_search


def build_gaussian_2d_representation_wrapper(
    data: np.ndarray,
    cfg: ExperimentConfig,
) -> None:
    """
    Thin wrapper kept for pipeline-layer compatibility.
    """
    run_representation_search(data=data, cfg=cfg)


def exhaustive_representation_search_pipeline(
    data: np.ndarray,
    cfg: ExperimentConfig,
) -> None:
    """
    Main 2.0 preprocessing pipeline entry.
    """
    run_representation_search(data=data, cfg=cfg)


# backward-compatible alias for callers you may still have during transition
exhaustive_representation_search = exhaustive_representation_search_pipeline
from __future__ import annotations

import numpy as np


def compute_weighted_values(
    deviation_code: np.ndarray,
    momentum_code: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    return alpha * deviation_code.astype(np.float64) + beta * momentum_code.astype(np.float64)


def pair_to_fixed_category(
    deviation_code: np.ndarray,
    momentum_code: np.ndarray,
) -> np.ndarray:
    """
    Fixed mapping:
    (-1,-1)->0, (-1,0)->1, (-1,1)->2,
    ( 0,-1)->3, ( 0,0)->4, ( 0,1)->5,
    ( 1,-1)->6, ( 1,0)->7, ( 1,1)->8
    """
    return ((deviation_code + 1) * 3 + (momentum_code + 1)).astype(np.int8)


def weighted_values_to_rank_category(
    weighted_values: np.ndarray,
) -> tuple[np.ndarray, dict]:
    unique_vals = np.unique(weighted_values)
    if len(unique_vals) != 9:
        raise ValueError(
            f"Expected 9 unique weighted values, got {len(unique_vals)}. "
            "Check alpha/beta or use pair_index mode."
        )

    unique_sorted = np.sort(unique_vals)
    value_to_cat = {float(v): int(i) for i, v in enumerate(unique_sorted)}

    category = np.empty_like(weighted_values, dtype=np.int8)
    for v, c in value_to_cat.items():
        category[np.isclose(weighted_values, v)] = c

    return category, {str(k): v for k, v in value_to_cat.items()}
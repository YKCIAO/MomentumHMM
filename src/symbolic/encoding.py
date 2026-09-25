from __future__ import annotations

import numpy as np


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


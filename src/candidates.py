import numpy as np
from typing import Tuple


def generate_random_candidates(
    n_candidates: int,
    dim: int,
) -> np.ndarray:
    """
    Generate random candidate points uniformly in [0, 1)^dim.

    Parameters
    ----------
    n_candidates : int
        Number of candidate points to generate.
    dim : int
        Dimensionality of the function (number of input variables).

    Returns
    -------
    X_candidates : np.ndarray
        Array of shape (n_candidates, dim).
    """
    X_candidates = np.random.uniform(
        low=0.0,
        high=1.0,
        size=(n_candidates, dim),
    )
    return X_candidates

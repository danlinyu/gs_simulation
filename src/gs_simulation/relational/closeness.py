"""Grey Closeness Relational Degree — Liu §5.6.2.

Level + shape similarity (Liu et al. 2011). Inverse of the absolute area
gap between *raw* (non-shifted) sequences::

    ρ̃_{ij} = 1 / (1 + |S_i − S_j|)

where ``S_i = ∫ X_i dt`` (trapezoidal sum of the raw sequence).

NOT translation-invariant — shifting either sequence changes the gap.
Use when both sequences are measured in similar units and absolute level
is meaningful.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§5.6.2 Definition 5.6.2.
Liu et al. (2011). Original closeness formulation.
"""
from __future__ import annotations

import numpy as np

from gs_simulation.relational.absolute import trapezoidal_signed_area

__all__ = ["closeness_relational_degree"]


def closeness_relational_degree(
    x_i: np.ndarray, x_j: np.ndarray,
) -> float:
    """Compute the closeness relational degree ``ρ̃_{ij}``.

    Parameters
    ----------
    x_i, x_j : array-like, 1-D, equal length

    Returns
    -------
    float
        ``ρ̃_{ij} ∈ (0, 1]``. Level + shape; NOT translation-invariant.

    References
    ----------
    Liu (2024) §5.6.2 Definition 5.6.2.
    """
    x_i_arr = np.asarray(x_i, dtype=float)
    x_j_arr = np.asarray(x_j, dtype=float)
    if x_i_arr.ndim != 1 or x_j_arr.ndim != 1:
        raise ValueError(
            "closeness_relational_degree requires 1-D sequences; got "
            f"shapes {x_i_arr.shape} and {x_j_arr.shape}"
        )
    if x_i_arr.size != x_j_arr.size:
        raise ValueError(
            f"Sequences must be the same length; got {x_i_arr.size} and {x_j_arr.size}"
        )
    S_i = trapezoidal_signed_area(x_i_arr)
    S_j = trapezoidal_signed_area(x_j_arr)
    return float(1.0 / (1.0 + abs(S_i - S_j)))

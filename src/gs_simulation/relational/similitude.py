"""Grey Similitude Relational Degree — Liu §5.6.1.

Pure-shape similarity (Liu et al. 2011). Inverse of the absolute area
between zero-shifted sequences::

    ε̃_{ij} = 1 / (1 + |s_i − s_j|)

Translation-invariant. Larger ⟺ more similar shape. Magnitudes ``|s_i|``
and ``|s_j|`` do *not* enter — only the gap between them. Distinct from
Liu §5.4's absolute degree, whose denominator includes ``|s_i| + |s_j|``.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§5.6.1 Definition 5.6.1.
Liu et al. (2011). Original similitude formulation.
"""
from __future__ import annotations

import numpy as np

from gs_simulation.relational.absolute import (
    trapezoidal_signed_area,
    zero_starting_point_image,
)

__all__ = ["similitude_relational_degree"]


def similitude_relational_degree(
    x_i: np.ndarray, x_j: np.ndarray,
) -> float:
    """Compute the similitude relational degree ``ε̃_{ij}``.

    Parameters
    ----------
    x_i, x_j : array-like, 1-D, equal length

    Returns
    -------
    float
        ``ε̃_{ij} ∈ (0, 1]``. Pure-shape, translation-invariant.

    References
    ----------
    Liu (2024) §5.6.1 Definition 5.6.1.
    """
    x_i_arr = np.asarray(x_i, dtype=float)
    x_j_arr = np.asarray(x_j, dtype=float)
    if x_i_arr.ndim != 1 or x_j_arr.ndim != 1:
        raise ValueError(
            "similitude_relational_degree requires 1-D sequences; got "
            f"shapes {x_i_arr.shape} and {x_j_arr.shape}"
        )
    if x_i_arr.size != x_j_arr.size:
        raise ValueError(
            f"Sequences must be the same length; got {x_i_arr.size} and {x_j_arr.size}"
        )

    s_i = trapezoidal_signed_area(zero_starting_point_image(x_i_arr))
    s_j = trapezoidal_signed_area(zero_starting_point_image(x_j_arr))
    return float(1.0 / (1.0 + abs(s_i - s_j)))

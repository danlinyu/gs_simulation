"""Grey Relative Relational Degree — Liu §5.5.1.

Captures *rate-of-change* similarity. Computed as the absolute relational
degree of the *initial images* (D₁-normalized sequences). Scale-invariant:
``r(X, Y) = r(aX, bY)`` for any positive ``a, b``. Per Liu §5.5.1 Theorem
5.5.1, ``r`` and the absolute degree ``ε`` are uncorrelated — high ε
does not imply high r and vice versa.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§5.5.1 Definition 5.5.1; Theorem 5.5.1.
"""
from __future__ import annotations

import numpy as np

from gs_simulation.operators.relational import apply_d1_initialing
from gs_simulation.relational.absolute import absolute_relational_degree

__all__ = ["relative_relational_degree"]


def relative_relational_degree(
    x_i: np.ndarray, x_j: np.ndarray,
) -> float:
    """Compute the relative grey relational degree ``r_{ij}``.

    Sequence-rate-of-change similarity. Both sequences must have non-zero
    first elements so the D₁ initialing operator is well-defined.

    Parameters
    ----------
    x_i, x_j : array-like, 1-D, equal length
        Two sequences sampled at equal intervals; first elements non-zero.

    Returns
    -------
    float
        ``r_{ij} ∈ (0, 1]``. Scale-invariant.

    Raises
    ------
    ValueError
        If either sequence has a zero first element (D₁ undefined) or the
        sequences are not 1-D / not the same length.

    References
    ----------
    Liu (2024) §5.5.1 Definition 5.5.1; Theorem 5.5.1.
    """
    x_i_init = apply_d1_initialing(x_i)
    x_j_init = apply_d1_initialing(x_j)
    return absolute_relational_degree(x_i_init, x_j_init)

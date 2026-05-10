"""Grey Synthetic Relational Degree — Liu §5.5.2.

Convex combination of the absolute and relative degrees. The mixing
weight ``θ ∈ [0, 1]`` selects emphasis: ``θ = 1`` is purely absolute
(magnitude/shape only), ``θ = 0`` is purely relative (rate-only),
``θ = 0.5`` is balanced.

Mathematics::

    ρ_{ij} = θ · ε_{ij} + (1 − θ) · r_{ij}

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§5.5.2 Definition 5.5.2.
"""
from __future__ import annotations

import numpy as np

from gs_simulation.relational.absolute import absolute_relational_degree
from gs_simulation.relational.relative import relative_relational_degree

__all__ = ["synthetic_relational_degree"]


def synthetic_relational_degree(
    x_i: np.ndarray, x_j: np.ndarray, *, theta: float = 0.5,
) -> float:
    """Compute the synthetic relational degree ``ρ_{ij}``.

    Parameters
    ----------
    x_i, x_j : array-like, 1-D, equal length, with non-zero first elements.
    theta : float, default 0.5
        Mixing weight in ``[0, 1]``. ``θ = 1`` selects pure absolute;
        ``θ = 0`` selects pure relative.

    Returns
    -------
    float
        ``ρ_{ij} ∈ (0, 1]``.

    Raises
    ------
    ValueError
        If ``θ ∉ [0, 1]`` or input contracts (length / 1-D / non-zero
        first) are violated.

    References
    ----------
    Liu (2024) §5.5.2 Definition 5.5.2.
    """
    if not (0.0 <= theta <= 1.0):
        raise ValueError(f"theta must be in [0, 1]; got {theta}")
    eps = absolute_relational_degree(x_i, x_j)
    r = relative_relational_degree(x_i, x_j)
    return float(theta * eps + (1.0 - theta) * r)

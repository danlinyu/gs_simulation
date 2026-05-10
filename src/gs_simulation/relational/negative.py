"""Negative Grey Relational Analysis — Liu §5.7 (Liu et al. 2022).

Captures *anti-correlation*: how strongly two sequences vary in opposite
directions. Returns values in ``(−1, 0]`` rather than the positive-side
``(0, 1]`` of standard GRA degrees.

Mathematics
-----------
For sequences with opposite monotonicity (one increasing, one decreasing),
the *negative grey similitude relational degree* (Liu et al. 2022,
Definition 5.7.5) is::

    φᴺ_{ij} = − |s_i − s_j| / (1 + |s_i − s_j|)

where ``s_i, s_j`` are the trapezoidal signed areas of the zero-shifted
sequences. Larger |φᴺ| ⟺ stronger inverse coupling.

Properties (Liu §5.7 Axioms 5.7.1, 5.7.2)
-----------------------------------------
- ``-1 < φᴺ_{ij} ≤ 0``.
- ``φᴺ_{ij} = 0 ⟺ X_i = X_j`` (no inverse relationship).
- Stronger inverse relationship → smaller (more negative) ``φᴺ``.

Use case
--------
On real-data domains where one variable's increase tends to coincide
with another's decrease (e.g., lead exposure vs. cognitive scores;
demolitions vs. employment in some neighborhoods). Standard GRA on
reverse sequences gives misleadingly low positive-side degrees;
negative GRA reads the anti-coupling directly.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§5.7 Definitions 5.7.1-5.7.6; Axioms 5.7.1, 5.7.2; Example 1 (Liu 2022).
Liu et al. (2022). Original negative GRA formulation.
"""
from __future__ import annotations

import numpy as np

from gs_simulation.relational.absolute import (
    trapezoidal_signed_area,
    zero_starting_point_image,
)

__all__ = ["negative_similitude_relational_degree"]


def negative_similitude_relational_degree(
    x_i: np.ndarray, x_j: np.ndarray,
) -> float:
    """Compute the negative similitude relational degree ``φᴺ_{ij}``.

    Parameters
    ----------
    x_i, x_j : array-like, 1-D, equal length

    Returns
    -------
    float
        ``φᴺ_{ij} ∈ (−1, 0]``. More negative ⟺ stronger anti-coupling.

    References
    ----------
    Liu (2024) §5.7.5 Definition 5.7.5; Axioms 5.7.1, 5.7.2.

    Examples
    --------
    Liu §5.7 Example 1: X₁ = (1, 2, 3, 3, 5), X₂ = (5, 4, 2, 2, 1) →
    s₁ = +7, s₂ = −9, |s₁ − s₂| = 16, φᴺ = −16/17 ≈ −0.9412.

    >>> import numpy as np
    >>> X1 = np.array([1.0, 2.0, 3.0, 3.0, 5.0])
    >>> X2 = np.array([5.0, 4.0, 2.0, 2.0, 1.0])
    >>> phi = negative_similitude_relational_degree(X1, X2)
    >>> bool(np.isclose(phi, -16/17, atol=1e-12))
    True
    """
    x_i_arr = np.asarray(x_i, dtype=float)
    x_j_arr = np.asarray(x_j, dtype=float)
    if x_i_arr.ndim != 1 or x_j_arr.ndim != 1:
        raise ValueError(
            "negative_similitude_relational_degree requires 1-D sequences; "
            f"got shapes {x_i_arr.shape} and {x_j_arr.shape}"
        )
    if x_i_arr.size != x_j_arr.size:
        raise ValueError(
            f"Sequences must be the same length; got {x_i_arr.size} and {x_j_arr.size}"
        )

    s_i = trapezoidal_signed_area(zero_starting_point_image(x_i_arr))
    s_j = trapezoidal_signed_area(zero_starting_point_image(x_j_arr))
    diff = abs(s_i - s_j)
    return float(-diff / (1.0 + diff))

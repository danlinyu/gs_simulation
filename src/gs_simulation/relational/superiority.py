"""Superiority Analysis — Liu §5.9.

Given ``s`` system outputs ``Y₁, …, Y_s`` and ``m`` factors ``X₁, …, X_m``,
compute the ``s × m`` relational matrix ``Γ`` and identify:

- *Favorable* outputs / factors that dominate per-row (Definition 5.9.2).
- *Quasi-favorable* outputs / factors that dominate row-sums (Definition 5.9.4).

Per Proposition 5.9.1, a most-favorable output may not always exist, but
a quasi-preferred output and factor always do.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§5.9 Definitions 5.9.1-5.9.4; Proposition 5.9.1; Example 5.9.1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gs_simulation.relational.absolute import absolute_relational_degree

__all__ = [
    "build_relational_matrix",
    "favorable_indices",
    "quasi_favorable_ranking",
    "SuperiorityResult",
    "superiority_analysis",
]


@dataclass(frozen=True)
class SuperiorityResult:
    """Container for superiority analysis output.

    Attributes
    ----------
    Gamma : numpy.ndarray, shape ``(s, m)``
        Relational matrix; ``Gamma[i, j] = γ(Y_i, X_j)``.
    favorable_outputs : list[int]
        Indices of outputs that dominate every other output per-element
        (i.e., ``Γ[k, j] ≥ Γ[i, j]`` for all ``j``). Empty if no
        most-favorable output exists.
    favorable_factors : list[int]
        Indices of factors that dominate per-element (across outputs).
    quasi_preferred_outputs : list[int]
        Outputs ranked by row-sum, descending.
    quasi_preferred_factors : list[int]
        Factors ranked by column-sum, descending.
    """

    Gamma: np.ndarray
    favorable_outputs: list[int]
    favorable_factors: list[int]
    quasi_preferred_outputs: list[int]
    quasi_preferred_factors: list[int]


def build_relational_matrix(
    Y: np.ndarray, X: np.ndarray,
) -> np.ndarray:
    """Compute the ``s × m`` absolute relational matrix.

    Each entry ``Γ[i, j] = ε(Y_i, X_j)`` is the Liu §5.4 absolute degree.

    Parameters
    ----------
    Y : numpy.ndarray, shape ``(s, n)``
        Output sequences (rows).
    X : numpy.ndarray, shape ``(m, n)``
        Factor sequences (rows).

    Returns
    -------
    numpy.ndarray, shape ``(s, m)``

    Raises
    ------
    ValueError
        If ``Y`` and ``X`` have different second-axis lengths or are not 2-D.
    """
    Y_arr = np.asarray(Y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    if Y_arr.ndim != 2 or X_arr.ndim != 2:
        raise ValueError(
            f"Y and X must be 2-D; got shapes {Y_arr.shape} and {X_arr.shape}"
        )
    if Y_arr.shape[1] != X_arr.shape[1]:
        raise ValueError(
            f"Y.shape[1] = {Y_arr.shape[1]} must equal X.shape[1] = {X_arr.shape[1]}"
        )
    s, m = Y_arr.shape[0], X_arr.shape[0]
    Gamma = np.empty((s, m), dtype=float)
    for i in range(s):
        for j in range(m):
            Gamma[i, j] = absolute_relational_degree(Y_arr[i], X_arr[j])
    return Gamma


def favorable_indices(
    matrix: np.ndarray, *, axis: int = 0,
) -> list[int]:
    """Identify rows (or columns) that dominate all others *element-wise*.

    Liu §5.9 Definition 5.9.2: row ``k`` is more favorable than row ``i``
    if ``matrix[k, j] >= matrix[i, j]`` for every ``j``. Most favorable
    if dominant over all other rows.

    Parameters
    ----------
    matrix : numpy.ndarray, shape ``(s, m)``
    axis : int, default 0
        ``0`` finds dominant rows (favorable outputs); ``1`` finds
        dominant columns (favorable factors).

    Returns
    -------
    list[int]
        Indices of dominant rows / columns. Often empty when no strict
        domination exists.
    """
    if axis == 1:
        matrix = matrix.T
    n = matrix.shape[0]
    out: list[int] = []
    for k in range(n):
        is_dominant = True
        for i in range(n):
            if i == k:
                continue
            if not np.all(matrix[k] >= matrix[i]):
                is_dominant = False
                break
        if is_dominant:
            out.append(k)
    return out


def quasi_favorable_ranking(
    matrix: np.ndarray, *, axis: int = 0,
) -> list[int]:
    """Rank rows (or columns) by sum, descending — Liu §5.9 Definition 5.9.4.

    Parameters
    ----------
    matrix : numpy.ndarray, shape ``(s, m)``
    axis : int, default 0
        ``0`` ranks rows by row-sum; ``1`` ranks columns by column-sum.

    Returns
    -------
    list[int]
        Indices ordered by sum descending.
    """
    if axis == 0:
        sums = matrix.sum(axis=1)
    else:
        sums = matrix.sum(axis=0)
    return list(np.argsort(-sums))


def superiority_analysis(
    Y: np.ndarray, X: np.ndarray,
) -> SuperiorityResult:
    """Run the Liu §5.9 superiority analysis on outputs ``Y`` and factors ``X``.

    Builds ``Γ``, then extracts (a) favorable indices (Definition 5.9.2)
    and (b) quasi-preferred ranking (Definition 5.9.4) for both axes.

    Parameters
    ----------
    Y : numpy.ndarray, shape ``(s, n)``
    X : numpy.ndarray, shape ``(m, n)``

    Returns
    -------
    SuperiorityResult

    References
    ----------
    Liu (2024) §5.9 Definitions 5.9.1-5.9.4; Proposition 5.9.1.
    """
    Gamma = build_relational_matrix(Y, X)
    return SuperiorityResult(
        Gamma=Gamma,
        favorable_outputs=favorable_indices(Gamma, axis=0),
        favorable_factors=favorable_indices(Gamma, axis=1),
        quasi_preferred_outputs=quasi_favorable_ranking(Gamma, axis=0),
        quasi_preferred_factors=quasi_favorable_ranking(Gamma, axis=1),
    )

"""Deng's Grey Relational Degree — Liu §5.3.

The original 1985 grey relational degree by Deng Julong. Measures the
geometric similarity of compared sequences ``X_i`` to a reference sequence
``X_0`` after normalization, via point-wise distances scaled by global
distance range.

Mathematics
-----------
Given a reference sequence ``X_0`` and ``m`` compared sequences ``X_1, …, X_m``:

1. Apply a relational operator (``D_1`` initialing, ``D_2`` averaging, or
   ``D_3`` interval; Liu §5.2) to each sequence to obtain non-dimensional
   forms ``X_0', X_i'``.
2. Compute point-wise absolute differences ``Δ_i(k) = |X_0'(k) − X_i'(k)|``.
3. Find global extremes ``M = max_i max_k Δ_i(k)`` and ``m = min_i min_k Δ_i(k)``.
4. Per-point relational coefficients (``ξ ∈ (0, 1]`` is the discrimination
   coefficient; default ``ξ = 0.5``):

       γ_{0i}(k) = (m + ξ M) / (Δ_i(k) + ξ M)

5. Relational degree by averaging over points::

       γ_{0i} = (1/n) Σ_k γ_{0i}(k)

Properties (Liu §5.3 Theorem 5.3.2)
-----------------------------------
- ``0 < γ_{0i} ≤ 1``; ``γ_{0i} = 1 ⟺ X_0' = X_i'``.
- Larger ``γ`` ⟺ closer similarity in normalized value-space.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§5.3 Definitions 5.3.1-5.3.2; Theorem 5.3.2.
Deng, Julong (1985). Original GRA formulation.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from gs_simulation.operators.relational import (
    apply_d1_initialing,
    apply_d2_averaging,
    apply_d3_interval,
)

__all__ = [
    "deng_relational_degree",
    "deng_relational_coefficient",
]

OperatorName = Literal["D1", "D2", "D3", "none"]


def _apply_operator(x: np.ndarray, op: OperatorName) -> np.ndarray:
    """Apply the named relational operator (or identity if ``op == 'none'``)."""
    if op == "D1":
        return apply_d1_initialing(x, axis=-1)
    if op == "D2":
        return apply_d2_averaging(x, axis=-1)
    if op == "D3":
        return apply_d3_interval(x, axis=-1)
    if op == "none":
        return np.asarray(x, dtype=float)
    raise ValueError(
        f"Unknown operator {op!r}; choose 'D1', 'D2', 'D3', or 'none'."
    )


def deng_relational_coefficient(
    x0: np.ndarray,
    Xi: np.ndarray,
    *,
    xi: float = 0.5,
    operator: OperatorName = "D1",
) -> np.ndarray:
    """Compute the per-point Deng grey relational coefficients.

    Parameters
    ----------
    x0 : array-like, shape ``(n,)``
        Reference sequence.
    Xi : array-like, shape ``(n,)`` or ``(m, n)``
        One or more compared sequences. If 1-D, treated as a single
        compared sequence; if 2-D, each row is one compared sequence.
    xi : float, default 0.5
        Discrimination coefficient. Liu §5.3 default value.
    operator : {"D1", "D2", "D3", "none"}, default "D1"
        Pre-normalization operator (Liu §5.2). Use ``"none"`` to skip
        normalization (caller has already pre-processed).

    Returns
    -------
    numpy.ndarray
        Relational coefficients per point. Shape ``(n,)`` when ``Xi`` is
        1-D; shape ``(m, n)`` when ``Xi`` is 2-D.

    Raises
    ------
    ValueError
        If ``xi ∉ (0, 1]`` or shapes are incompatible.

    References
    ----------
    Liu (2024) §5.3 Definition 5.3.1.
    """
    if not (0.0 < xi <= 1.0):
        raise ValueError(f"Discrimination coefficient ξ must be in (0, 1]; got {xi}")

    x0_arr = np.asarray(x0, dtype=float)
    Xi_arr = np.asarray(Xi, dtype=float)
    if x0_arr.ndim != 1:
        raise ValueError(f"x0 must be 1-D; got shape {x0_arr.shape}")
    if Xi_arr.ndim == 1:
        if Xi_arr.shape != x0_arr.shape:
            raise ValueError(
                f"Xi length {Xi_arr.size} must match x0 length {x0_arr.size}"
            )
        Xi_arr = Xi_arr[None, :]
        squeeze = True
    elif Xi_arr.ndim == 2:
        if Xi_arr.shape[1] != x0_arr.size:
            raise ValueError(
                f"Xi shape {Xi_arr.shape} second axis must match x0 length "
                f"{x0_arr.size}"
            )
        squeeze = False
    else:
        raise ValueError(f"Xi must be 1-D or 2-D; got shape {Xi_arr.shape}")

    # Step 1: Apply normalization operator.
    x0_d = _apply_operator(x0_arr, operator)
    Xi_d = _apply_operator(Xi_arr, operator)

    # Step 2: Per-point absolute differences (m, n).
    delta = np.abs(x0_d[None, :] - Xi_d)

    # Step 3: Global extremes across all (i, k).
    M = float(delta.max())
    m_min = float(delta.min())

    # Step 4: Per-point coefficients γ(k).
    denom = delta + xi * M
    # When denom is zero (Δ = 0 AND ξM = 0 ⟹ all sequences identical),
    # set γ = 1. Use a guarded denominator to avoid divide-by-zero warnings.
    safe_denom = np.where(denom > 0, denom, 1.0)
    coeff = np.where(denom > 0, (m_min + xi * M) / safe_denom, 1.0)

    if squeeze:
        return coeff[0]
    return coeff


def deng_relational_degree(
    x0: np.ndarray,
    Xi: np.ndarray,
    *,
    xi: float = 0.5,
    operator: OperatorName = "D1",
) -> np.ndarray | float:
    """Compute Deng's grey relational degree γ_{0i}.

    Average of the per-point relational coefficients (Liu §5.3 Step 5).

    Parameters
    ----------
    x0 : array-like, shape ``(n,)``
    Xi : array-like, shape ``(n,)`` or ``(m, n)``
    xi : float, default 0.5
    operator : {"D1", "D2", "D3", "none"}, default "D1"

    Returns
    -------
    float or numpy.ndarray
        Scalar ``γ_{0i}`` when ``Xi`` is 1-D; array of shape ``(m,)`` of
        per-sequence degrees when ``Xi`` is 2-D.

    References
    ----------
    Liu (2024) §5.3 Definition 5.3.2; Theorem 5.3.2.

    Examples
    --------
    Identical sequences yield γ = 1:

    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> bool(np.isclose(deng_relational_degree(x, x), 1.0))
    True
    """
    coeff = deng_relational_coefficient(x0, Xi, xi=xi, operator=operator)
    if coeff.ndim == 1:
        return float(coeff.mean())
    return coeff.mean(axis=-1)

"""Buffer operators — Liu §4.2-4.3.

Buffer operators recover the underlying system behavior ``X⁰`` from
shock-disturbed observed data ``X = X⁰ + ε``. Per the three Liu §4.2.2
axioms (fixed-point at ``x(n)``, in-accordance-with-information,
expressed-normality), an operator ``D`` is a buffer if it satisfies
all three.

A *weakening* operator slows down growth / decline / fluctuation; a
*strengthening* operator amplifies it. Liu §4.2.3 Theorems 4.2.1-4.2.3
characterize each by element-wise inequalities versus the input.

Operators implemented (Liu §4.3):

================  =================  ===================================
Function          Family             Mechanism
================  =================  ===================================
**AWBO**          weakening          arithmetic mean of tail [k, n]
**WAWBO**         weakening          weighted arithmetic mean of tail
**WGAWBO**        weakening          weighted geometric mean of tail
**ASBO**          strengthening      ``(n−k+1)·x(k)² / Σ_tail x(i)``
**WASBO**         strengthening      weighted ASBO with weights ``ω``
**GFBO**          general            ``α < 0`` weakening; ``α > 0`` strengthening
================  =================  ===================================

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§4.2 Definitions 4.2.3-4.2.5; Theorems 4.2.1-4.2.3.
§4.3 Theorems 4.3.1-4.3.7.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "apply_awbo",
    "apply_wawbo",
    "apply_wgawbo",
    "apply_asbo",
    "apply_wasbo",
    "apply_gfbo",
]


def apply_awbo(x: np.ndarray) -> np.ndarray:
    """Average Weakening Buffer Operator (AWBO). Liu §4.3 Theorem 4.3.1.

    ``x_d(k) = (1/(n−k+1)) Σ_{i=k}^{n} x(i)``

    Always weakening regardless of monotonicity.

    Parameters
    ----------
    x : array-like, 1-D

    Returns
    -------
    numpy.ndarray, same shape as input

    Examples
    --------
    Liu §4.3 Example 4.3.1: X = (36.5, 54.3, 80.1, 109.8, 143.2)
    → X_D ≈ (84.78, 96.85, 111.03, 126.50, 143.20).

    >>> import numpy as np
    >>> X = np.array([36.5, 54.3, 80.1, 109.8, 143.2])
    >>> apply_awbo(X)  # doctest: +NORMALIZE_WHITESPACE
    array([ 84.78,  96.85, 111.03, 126.5 , 143.2 ])
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"AWBO requires 1-D input; got shape {arr.shape}")
    n = arr.size
    out = np.empty(n, dtype=float)
    for k in range(n):
        out[k] = arr[k:].mean()
    return out


def apply_wawbo(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted Average Weakening Buffer Operator (WAWBO). Liu §4.3 Theorem 4.3.2.

    ``x_d(k) = Σ_{i=k}^{n} ω_i x(i) / Σ_{i=k}^{n} ω_i``

    Reduces to AWBO when all weights are equal.

    Parameters
    ----------
    x : array-like, 1-D, shape ``(n,)``
    weights : array-like, 1-D, shape ``(n,)``, all positive

    Returns
    -------
    numpy.ndarray
    """
    arr = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    if arr.ndim != 1 or w.ndim != 1:
        raise ValueError(
            f"WAWBO requires 1-D input and 1-D weights; got shapes "
            f"{arr.shape} and {w.shape}"
        )
    if arr.size != w.size:
        raise ValueError(
            f"weights length {w.size} must equal input length {arr.size}"
        )
    if np.any(w <= 0):
        raise ValueError("WAWBO weights must be strictly positive.")
    n = arr.size
    out = np.empty(n, dtype=float)
    for k in range(n):
        tail_w = w[k:]
        tail_x = arr[k:]
        out[k] = np.sum(tail_w * tail_x) / np.sum(tail_w)
    return out


def apply_wgawbo(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted Geometric Average Weakening Buffer Operator (WGAWBO).

    Liu §4.3 Theorem 4.3.3:

        ``x_d(k) = (∏_{i=k}^{n} x(i)^{ω_i})^{1 / Σ_{i=k}^{n} ω_i}``

    Always weakening; requires ``x > 0``.

    Parameters
    ----------
    x : array-like, 1-D, shape ``(n,)``, all strictly positive
    weights : array-like, 1-D, shape ``(n,)``, all strictly positive

    Returns
    -------
    numpy.ndarray
    """
    arr = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    if arr.ndim != 1 or w.ndim != 1:
        raise ValueError(
            f"WGAWBO requires 1-D input and 1-D weights; got "
            f"{arr.shape} and {w.shape}"
        )
    if arr.size != w.size:
        raise ValueError(
            f"weights length {w.size} must equal input length {arr.size}"
        )
    if np.any(arr <= 0):
        raise ValueError("WGAWBO requires strictly positive input values.")
    if np.any(w <= 0):
        raise ValueError("WGAWBO weights must be strictly positive.")
    n = arr.size
    out = np.empty(n, dtype=float)
    log_x = np.log(arr)
    for k in range(n):
        tail_w = w[k:]
        tail_log = log_x[k:]
        out[k] = float(np.exp(np.sum(tail_w * tail_log) / np.sum(tail_w)))
    return out


def apply_asbo(x: np.ndarray) -> np.ndarray:
    """Average Strengthening Buffer Operator (ASBO). Liu §4.3 Theorem 4.3.5.

    ``x_d(k) = (n − k + 1)·x(k)² / Σ_{i=k}^{n} x(i)``

    Always strengthening regardless of monotonicity.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"ASBO requires 1-D input; got shape {arr.shape}")
    n = arr.size
    out = np.empty(n, dtype=float)
    for k in range(n):
        denom = arr[k:].sum()
        if denom == 0:
            out[k] = arr[k]
        else:
            out[k] = (n - k) * arr[k] ** 2 / denom
            # Liu §4.3 indexing is 1-based; (n−k+1) in 1-based is (n−k+1).
            # In 0-based (our k starts at 0), the multiplier is (n−k).
            # But length of tail [k:] is n−k as well, so (n−k) is correct.
    # Fixed-point: x_d(n) = x(n) by Axiom 4.2.1.
    out[-1] = arr[-1]
    return out


def apply_wasbo(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted Average Strengthening Buffer Operator (WASBO).

    Liu §4.3 Theorem 4.3.6:

        ``x_d(k) = (Σ_tail ω) x(k)² / (Σ_tail ω·x)``

    Strengthening regardless of sequence type.
    """
    arr = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    if arr.ndim != 1 or w.ndim != 1:
        raise ValueError(
            f"WASBO requires 1-D input and 1-D weights; got "
            f"{arr.shape} and {w.shape}"
        )
    if arr.size != w.size:
        raise ValueError(
            f"weights length {w.size} must equal input length {arr.size}"
        )
    if np.any(w <= 0):
        raise ValueError("WASBO weights must be strictly positive.")
    n = arr.size
    out = np.empty(n, dtype=float)
    for k in range(n):
        tail_w = w[k:]
        tail_x = arr[k:]
        denom = float(np.sum(tail_w * tail_x))
        if denom == 0:
            out[k] = arr[k]
        else:
            out[k] = float(np.sum(tail_w)) * arr[k] ** 2 / denom
    out[-1] = arr[-1]  # fixed point
    return out


def apply_gfbo(
    x: np.ndarray, alpha: float, weights: np.ndarray | None = None,
) -> np.ndarray:
    """General Form of Buffer Operator (GFBO). Liu §4.3 Theorem 4.3.7.

    ``x_d(k) = x(k) · (x(k) / W_tail_mean)^α``

    where ``W_tail_mean = Σ_{i=k}^{n} ω_i x(i) / Σ_{i=k}^{n} ω_i`` is the
    weighted tail mean. ``α < 0`` is weakening; ``α > 0`` is strengthening;
    ``α = 0`` is identity. ``α = -1`` recovers WAWBO; ``α = +1`` recovers
    WASBO (Liu §4.3 Corollaries).

    Parameters
    ----------
    x : array-like, 1-D, all strictly positive
    alpha : float
        Strength parameter.
    weights : array-like, 1-D, optional
        Per-element weights. Defaults to uniform ``1.0``.

    Returns
    -------
    numpy.ndarray
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"GFBO requires 1-D input; got shape {arr.shape}")
    if np.any(arr <= 0):
        raise ValueError("GFBO requires strictly positive input values.")
    n = arr.size
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,) or np.any(w <= 0):
            raise ValueError(
                "GFBO weights must be 1-D, length n, strictly positive."
            )

    if alpha == 0:
        return arr.copy()

    out = np.empty(n, dtype=float)
    for k in range(n):
        tail_w = w[k:]
        tail_x = arr[k:]
        tail_mean = float(np.sum(tail_w * tail_x) / np.sum(tail_w))
        out[k] = arr[k] * (arr[k] / tail_mean) ** alpha
    return out

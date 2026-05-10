"""Accumulation Generation Operator (AGO) and Inverse AGO (IAGO).

Liu §4.6 — load-bearing for grey forecasting models. AGO transforms a possibly
chaotic input sequence ``X⁰`` into ``X^(r)`` whose stepwise ratios are
quasi-constant under mild conditions (Liu §4.7 Theorem 4.7.2), making the
accumulated sequence fittable by exponential-form models such as GM(1,1).

Mathematics
-----------
**1-AGO** (first-order accumulation):

    x^(1)(k) = Σ_{i=1}^{k} x^(0)(i),    k = 1, …, n

**r-AGO** (recursive application):

    x^(r)(k) = Σ_{i=1}^{k} x^(r-1)(i),  k = 1, …, n

**1-IAGO** (inverse accumulation):

    α^(1) x^(0)(k) = x^(0)(k) − x^(0)(k−1),    k = 2, …, n
    α^(1) x^(0)(1) = x^(0)(1)

**Recovery (Liu §4.6 Proposition 4.6.1):**

    α^(r) X^(r) = X^(0)

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis: Methods, Models and Applications*,
2nd Ed., Springer Singapore. §4.6 Definitions 4.6.1-4.6.2; Proposition 4.6.1.
"""
from __future__ import annotations

import numpy as np


def apply_ago(x: np.ndarray, order: int = 1, axis: int = -1) -> np.ndarray:
    """Apply the r-th-order accumulation generation operator.

    Maps ``X⁰`` to ``X^(r)`` along the given axis. ``order=0`` is the
    identity. Each successive order applies a running sum along ``axis``.

    Parameters
    ----------
    x : array-like
        Input sequence(s). Any shape with at least one element along
        ``axis``.
    order : int, default 1
        Non-negative AGO order ``r``.
    axis : int, default -1
        Axis along which to accumulate.

    Returns
    -------
    numpy.ndarray
        ``X^(r)``. Same shape as input.

    Raises
    ------
    ValueError
        If ``order`` is negative.

    References
    ----------
    Liu (2024) §4.6 Definition 4.6.1.

    Examples
    --------
    >>> apply_ago(np.array([5.3, 7.6, 10.4, 13.8, 18.1]), order=1)
    array([ 5.3, 12.9, 23.3, 37.1, 55.2])
    """
    if order < 0:
        raise ValueError(
            f"AGO order must be non-negative; got order={order}"
        )
    out = np.asarray(x, dtype=float).copy()
    for _ in range(order):
        out = np.cumsum(out, axis=axis)
    return out


def apply_iago(x: np.ndarray, order: int = 1, axis: int = -1) -> np.ndarray:
    """Apply the r-th-order inverse accumulation generation operator.

    Maps ``X^(r)`` to ``X^(0)`` along the given axis. ``order=0`` is the
    identity. Each successive order applies a discrete first-difference
    along ``axis`` while preserving the first element so the inverse
    recovers the AGO input exactly (Liu §4.6 Prop. 4.6.1).

    Parameters
    ----------
    x : array-like
        Input sequence(s). Any shape with at least one element along
        ``axis``.
    order : int, default 1
        Non-negative IAGO order ``r``.
    axis : int, default -1
        Axis along which to apply the operator.

    Returns
    -------
    numpy.ndarray
        ``α^(r) X``. Same shape as input.

    Raises
    ------
    ValueError
        If ``order`` is negative.

    References
    ----------
    Liu (2024) §4.6 Definition 4.6.2; Proposition 4.6.1 (recovery theorem).

    Examples
    --------
    >>> apply_iago(np.array([5.3, 12.9, 23.3, 37.1, 55.2]), order=1)
    array([ 5.3,  7.6, 10.4, 13.8, 18.1])
    """
    if order < 0:
        raise ValueError(
            f"IAGO order must be non-negative; got order={order}"
        )
    out = np.asarray(x, dtype=float).copy()
    for _ in range(order):
        # First difference with the first slice along ``axis`` preserved.
        diff = np.diff(out, n=1, axis=axis)
        first_slice = np.take(out, indices=[0], axis=axis)
        out = np.concatenate([first_slice, diff], axis=axis)
    return out


__all__ = [
    "apply_ago",
    "apply_iago",
]

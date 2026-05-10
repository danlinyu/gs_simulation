"""Average and Moving-Average operators — Liu §4.4-4.5.

Lightweight operators for noise reduction and quasi-smoothness diagnostics:

- ``mean_operator`` (Liu §4.4 Definition 4.4.3): trailing 2-item mean
  ``z(k) = 0.5(x(k) + x(k−1))``. Used in GM(1,1) background-value
  construction and as a 1-step denoising primitive.
- ``moving_average_denoise`` (Liu §4.4 Definition 4.4.4): centered
  ``(2m+1)``-item average. Low-pass filter; output spans ``[m+1, n−m]``.
- ``smoothness_ratio`` (Liu §4.5 Definition 4.5.1):
  ``ρ(k) = x(k) / Σ_{i=1}^{k−1} x(i)``. Smaller ⟺ smoother.
- ``stepwise_ratio`` (Liu §4.5 Definition 4.5.3):
  ``σ(k) = x(k) / x(k−1)`` — growth-rate sequence.
- ``is_quasi_smooth`` (Liu §4.5 Definition 4.5.2): three-condition
  diagnostic for whether a sequence is GM(1,1)-fittable.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§4.4 Definitions 4.4.1-4.4.4.
§4.5 Definitions 4.5.1-4.5.3; Propositions 4.5.1-4.5.3.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "mean_operator",
    "moving_average_denoise",
    "smoothness_ratio",
    "stepwise_ratio",
    "is_quasi_smooth",
]


def mean_operator(x: np.ndarray) -> np.ndarray:
    """Trailing 2-item mean ``z(k) = 0.5(x(k) + x(k−1))`` for ``k = 2..n``.

    Output has ``n−1`` elements (indices correspond to ``k = 2..n`` in
    Liu's 1-based convention).

    Parameters
    ----------
    x : array-like, 1-D, length n ≥ 2

    Returns
    -------
    numpy.ndarray, shape ``(n−1,)``

    References
    ----------
    Liu (2024) §4.4 Definition 4.4.3.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(
            f"mean_operator requires 1-D input with n ≥ 2; got shape {arr.shape}"
        )
    return 0.5 * (arr[1:] + arr[:-1])


def moving_average_denoise(x: np.ndarray, m: int = 1) -> np.ndarray:
    """Centered ``(2m+1)``-window moving average (Liu §4.4 Definition 4.4.4).

    Output covers indices ``k = m+1, …, n−m`` in Liu's 1-based notation;
    output length is ``n − 2m``.

    Parameters
    ----------
    x : array-like, 1-D
    m : int, default 1
        Half-window. ``m=1`` ⇒ 3-item window; ``m=2`` ⇒ 5-item.

    Returns
    -------
    numpy.ndarray, shape ``(n − 2m,)``

    References
    ----------
    Liu (2024) §4.4 Definition 4.4.4.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"moving_average_denoise requires 1-D; got shape {arr.shape}")
    if m < 1:
        raise ValueError(f"m must be ≥ 1; got {m}")
    n = arr.size
    if n < 2 * m + 1:
        raise ValueError(
            f"Window size 2m+1 = {2 * m + 1} exceeds input length {n}"
        )
    window = 2 * m + 1
    out = np.empty(n - 2 * m, dtype=float)
    for j, k in enumerate(range(m, n - m)):
        out[j] = arr[k - m: k + m + 1].mean()
    assert window  # silence unused-variable lint (window is documentary)
    return out


def smoothness_ratio(x: np.ndarray) -> np.ndarray:
    """Liu §4.5 Definition 4.5.1: ``ρ(k) = x(k) / Σ_{i=1}^{k−1} x(i)``.

    Smaller ``ρ(k)`` indicates smoother / less abrupt change at index ``k``.
    Output indices correspond to ``k = 2..n``.

    Parameters
    ----------
    x : array-like, 1-D, length n ≥ 2, all entries non-negative

    Returns
    -------
    numpy.ndarray, shape ``(n−1,)``
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(
            f"smoothness_ratio requires 1-D input with n ≥ 2; got shape {arr.shape}"
        )
    cumsum = np.cumsum(arr)
    # ρ(k) for k = 2..n in 1-based is arr[k-1] / cumsum[k-2]; in 0-based:
    # output[i] (i = 0..n-2) = arr[i+1] / cumsum[i].
    denom = cumsum[:-1]
    if np.any(denom <= 0):
        raise ValueError(
            "smoothness_ratio undefined when cumulative sum has non-positive "
            "values; ensure x is non-negative with positive partial sums."
        )
    return arr[1:] / denom


def stepwise_ratio(x: np.ndarray) -> np.ndarray:
    """Liu §4.5 Definition 4.5.3: ``σ(k) = x(k) / x(k−1)``.

    Output indices correspond to ``k = 2..n`` (length ``n−1``).

    Parameters
    ----------
    x : array-like, 1-D, length n ≥ 2, no zero entries

    Returns
    -------
    numpy.ndarray, shape ``(n−1,)``
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(
            f"stepwise_ratio requires 1-D input with n ≥ 2; got shape {arr.shape}"
        )
    if np.any(arr[:-1] == 0):
        raise ValueError(
            "stepwise_ratio undefined when x(k−1) = 0; check input."
        )
    return arr[1:] / arr[:-1]


def is_quasi_smooth(x: np.ndarray, *, epsilon: float = 0.5) -> bool:
    """Liu §4.5 Definition 4.5.2 quasi-smoothness diagnostic.

    Returns ``True`` iff:

    1. ``ρ(k+1) / ρ(k) < 1`` for ``k = 2, …, n−1`` (smoothness ratios decreasing).
    2. ``ρ(k) ∈ [0, ε]`` for ``k = 3, …, n``.
    3. ``ε < 0.5``.

    Quasi-smoothness is the primary fittability diagnostic for GM(1,1).

    Parameters
    ----------
    x : array-like, 1-D
    epsilon : float, default 0.5
        Threshold (Condition 3 requires ``ε < 0.5``).

    Returns
    -------
    bool
    """
    if epsilon >= 0.5:
        return False
    rho = smoothness_ratio(x)
    if rho.size < 2:
        return False
    # Condition 1: rho ratios decreasing
    if not np.all(rho[1:] / rho[:-1] < 1.0):
        return False
    # Condition 2: rho[k] ∈ [0, ε] for k = 3..n. In our 0-based output:
    # rho[i] = ρ(i + 2), so we check rho[1:] (corresponding to k = 3..n).
    if rho.size < 2:
        return False
    if not np.all((rho[1:] >= 0.0) & (rho[1:] <= epsilon)):
        return False
    return True

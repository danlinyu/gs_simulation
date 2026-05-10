"""Fractional Grey Model GM(p/q, 1) — Liu §7.6.

Generalises the integer-order accumulation in GM(1,1) to *fractional*
order ``r = p/q`` (typically ``r ∈ (0, 1)``). The fractional AGO is::

    x^(r)(k) = Σ_{i=1}^{k} C_{k−i, k−i+r−1} · x⁽⁰⁾(i)

with binomial coefficients generalised via the Gamma function::

    C_{j, j−1+r} = Γ(j + r) / (Γ(r) · j!)
                 = r·(r+1)·…·(r+j−1) / j!     (Pochhammer recurrence)

Special cases:

- ``r = 1`` recovers integer 1-AGO (cumulative sum).
- ``r = 0`` is the identity.
- ``r = −1`` recovers 1-IAGO (first difference).
- General ``r`` is recovered by negating: ``α^(r) X = AGO(X, −r)``.

The fractional GM model is then fit DGM-style on the fractional AGO::

    x^(r)(k+1) = β₁ · x^(r)(k) + β₂

with least-squares estimation of ``(β₁, β₂)``. Inverse fractional AGO
recovers the predicted ``x⁽⁰⁾`` for forecasting.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§7.6 Definitions 7.6.1-7.6.3; Theorem 7.6.1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "apply_fractional_ago",
    "FractionalGMFit",
    "fit_fractional_gm11",
    "simulate_fractional_gm11",
    "forecast_fractional_gm11",
]


def _fractional_coefficients(order: float, n: int) -> np.ndarray:
    """Pochhammer-recurrence coefficients ``c_0, c_1, …, c_{n−1}``.

    ``c_0 = 1; c_j = c_{j−1} · (order + j − 1) / j``.
    """
    coef = np.empty(n, dtype=float)
    coef[0] = 1.0
    for j in range(1, n):
        coef[j] = coef[j - 1] * (order + j - 1) / j
    return coef


def apply_fractional_ago(x: np.ndarray, order: float) -> np.ndarray:
    """Apply the fractional accumulation operator of given real ``order``.

    Generalises 1-AGO (``order=1``) and 1-IAGO (``order=−1``) to any real
    order. ``apply_fractional_ago(apply_fractional_ago(x, r), −r) ≈ x``
    (recovery to floating-point precision).

    Parameters
    ----------
    x : array-like, 1-D
    order : float
        Real order. ``0`` is identity; ``1`` is 1-AGO; ``−1`` is 1-IAGO.

    Returns
    -------
    numpy.ndarray, same shape as input

    References
    ----------
    Liu (2024) §7.6 Definition 7.6.1.

    Examples
    --------
    >>> import numpy as np
    >>> apply_fractional_ago(np.array([1.0, 1.0, 1.0, 1.0]), order=1.0)
    array([1., 2., 3., 4.])
    >>> apply_fractional_ago(np.array([1.0, 2.0, 3.0, 4.0]), order=-1.0)
    array([1., 1., 1., 1.])
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"apply_fractional_ago requires 1-D; got shape {arr.shape}")
    n = arr.size
    if n == 0:
        return arr.copy()
    coef = _fractional_coefficients(order, n)
    out = np.empty(n, dtype=float)
    for k in range(n):
        # x^(r)(k+1)_1based = Σ_{j=0}^{k} c_j · x[k - j]
        # In 0-based: out[k] = Σ_{j=0}^{k} c_j · x[k - j] = convolution.
        out[k] = float(np.dot(coef[: k + 1], arr[k::-1]))
    return out


@dataclass(frozen=True)
class FractionalGMFit:
    """Fitted fractional GM(p/q, 1) parameters.

    Attributes
    ----------
    beta1, beta2 : float
        Discrete-model parameters from ``x^(r)(k+1) = β₁ x^(r)(k) + β₂``.
    order : float
        Fractional accumulation order.
    x0_first : float
        ``x⁽⁰⁾(1)`` for the time-response formula.
    n_train : int
    """

    beta1: float
    beta2: float
    order: float
    x0_first: float
    n_train: int


def fit_fractional_gm11(x: np.ndarray, order: float) -> FractionalGMFit:
    """Fit a fractional GM(p/q, 1) by least squares on the fractional AGO.

    Parameters
    ----------
    x : array-like, 1-D, n ≥ 4
    order : float
        Fractional accumulation order; typically in ``(0, 1]``.

    Returns
    -------
    FractionalGMFit

    Raises
    ------
    ValueError
        If input non-1-D, n < 4, or the resulting B^T B is singular, or
        ``β₁ = 1`` (degenerate).

    References
    ----------
    Liu (2024) §7.6 Theorem 7.6.1.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"fit_fractional_gm11 requires 1-D; got shape {arr.shape}")
    if arr.size < 4:
        raise ValueError(f"Need n ≥ 4 observations; got n={arr.size}")

    x_frac = apply_fractional_ago(arr, order)
    n = arr.size
    B = np.column_stack([x_frac[:-1], np.ones(n - 1)])
    Y = x_frac[1:]
    try:
        params = np.linalg.solve(B.T @ B, B.T @ Y)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"Fractional GM least-squares failed: BᵀB singular. {exc}"
        ) from exc
    beta1 = float(params[0])
    beta2 = float(params[1])
    if beta1 == 1.0:
        raise ValueError(
            "Fractional GM degenerate: β₁ = 1 yields non-decaying time response."
        )
    return FractionalGMFit(
        beta1=beta1,
        beta2=beta2,
        order=order,
        x0_first=float(arr[0]),
        n_train=n,
    )


def _x_frac_at(fit: FractionalGMFit, k: int) -> float:
    """Time response ``x̂^(r)(k)`` (1-based) for the discrete model."""
    fixed_pt = fit.beta2 / (1.0 - fit.beta1)
    return float((fit.x0_first - fixed_pt) * fit.beta1 ** (k - 1) + fixed_pt)


def simulate_fractional_gm11(
    fit: FractionalGMFit, n_periods: int | None = None,
) -> np.ndarray:
    """Reconstruct ``x̂⁽⁰⁾`` for ``k = 1, …, n_periods``.

    Computes ``x̂^(r)`` from the time-response formula then applies inverse
    fractional AGO (negative-order accumulation) to recover ``x̂⁽⁰⁾``.

    Parameters
    ----------
    fit : FractionalGMFit
    n_periods : int | None, default None
        Defaults to ``fit.n_train``.

    Returns
    -------
    numpy.ndarray, shape ``(n_periods,)``
    """
    if n_periods is None:
        n_periods = fit.n_train
    if n_periods < 1:
        raise ValueError(f"n_periods must be ≥ 1; got {n_periods}")

    x_frac_hat = np.array(
        [_x_frac_at(fit, k) for k in range(1, n_periods + 1)],
        dtype=float,
    )
    # Invert the fractional AGO: apply with negated order.
    x0_hat = apply_fractional_ago(x_frac_hat, -fit.order)
    return x0_hat


def forecast_fractional_gm11(
    fit: FractionalGMFit, n_ahead: int,
) -> np.ndarray:
    """Forecast ``n_ahead`` future ``x̂⁽⁰⁾`` values.

    Parameters
    ----------
    fit : FractionalGMFit
    n_ahead : int

    Returns
    -------
    numpy.ndarray, shape ``(n_ahead,)``
    """
    if n_ahead < 1:
        raise ValueError(f"n_ahead must be ≥ 1; got {n_ahead}")
    # Build the FULL fractional simulation up through n_train + n_ahead so the
    # inverse-AGO has the correct cumulative history.
    total = fit.n_train + n_ahead
    full_sim = simulate_fractional_gm11(fit, n_periods=total)
    return full_sim[fit.n_train: fit.n_train + n_ahead]

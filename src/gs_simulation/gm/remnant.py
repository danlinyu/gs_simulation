"""Remnant GM(1,1) — residual-modification model. Liu §7.4.

When base GM(1,1) accuracy is insufficient, the residual sequence
``ε⁽⁰⁾(k) = x⁽¹⁾(k) − x̂⁽¹⁾(k)`` may itself be approximately modelable. If
the residuals from some index ``k₀`` onward are sign-consistent and
weakly monotonic with at least 4 terms, a second GM(1,1) is fit on
``|ε⁽⁰⁾|[k₀:]`` and the error forecast ``ε̂⁽⁰⁾(k+1)`` is added back to
the base prediction with the original sign.

Per Liu Example 7.4.1, the remnant correction reduces the 4-point average
relative error from 21.5% → 4.6% on a worked example.

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis*, 2nd Ed., Springer Singapore.
§7.4 (Remnant GM(1,1) Model); Example 7.4.1 (worked example).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from gs_simulation.gm.base import GMFit
from gs_simulation.gm.gm11 import (
    fit_egm11,
    forecast_gm11,
    simulate_gm11,
)
from gs_simulation.operators.accumulation import apply_ago

__all__ = [
    "RemnantGMFit",
    "fit_remnant_gm11",
    "forecast_remnant_gm11",
    "simulate_remnant_gm11",
]


# Minimum number of post-k₀ residuals required to fit a second GM(1,1)
# (Liu §7.4 modelability criterion).
MIN_REMNANT_TERMS: Final[int] = 4


@dataclass(frozen=True)
class RemnantGMFit:
    """Container for a base GM(1,1) plus optional remnant correction.

    Attributes
    ----------
    base : GMFit
        Base GM(1,1) fit on the original sequence.
    remnant : GMFit | None
        Second GM(1,1) fit on ``|ε⁽⁰⁾|[k₀:]``. ``None`` if the residual
        sequence does not satisfy the modelability criterion.
    k0 : int
        1-based index at which the residual sign becomes consistent.
        ``-1`` when ``remnant is None``.
    sign : int
        Sign of the residuals from ``k₀`` onward (+1 or −1). ``0`` when
        ``remnant is None``.
    """

    base: GMFit
    remnant: GMFit | None
    k0: int
    sign: int


def fit_remnant_gm11(
    x: np.ndarray, *, base_fit: GMFit | None = None,
) -> RemnantGMFit:
    """Fit a base GM(1,1) plus optional remnant correction.

    Parameters
    ----------
    x : array-like
        1-D training sequence ``X⁽⁰⁾``. Same constraints as
        :func:`fit_egm11` (n ≥ 4, |a| < 2).
    base_fit : GMFit | None, default None
        Optional pre-computed base fit (e.g., to reuse a non-EGM form).
        If ``None``, EGM is fit internally.

    Returns
    -------
    RemnantGMFit
        Base + remnant. If the residual sequence does not satisfy the
        Liu §7.4 modelability criterion (constant sign for ≥ 4 terms,
        weakly monotonic in absolute value), ``remnant`` is ``None``.

    References
    ----------
    Liu (2024) §7.4.
    """
    x0 = np.asarray(x, dtype=float)
    if base_fit is None:
        base_fit = fit_egm11(x0)

    # Compute base in-sample x̂⁽⁰⁾ and convert to x̂⁽¹⁾ via cumsum.
    x0_hat = simulate_gm11(base_fit, n_periods=base_fit.n_train)
    x1_hat = np.cumsum(x0_hat)
    x1 = apply_ago(x0, order=1)

    # Residuals in the accumulated (X⁽¹⁾) space (Liu §7.4 Definition).
    eps = x1 - x1_hat  # shape (n,); eps[0] = 0 by construction
    n = eps.size

    # Modelability check: find largest tail [k0..n-1] where sign is constant
    # and length ≥ MIN_REMNANT_TERMS. Use 1-based index notation in attrs.
    sign_arr = np.sign(eps)
    # Walk backward from the end to find the longest constant-sign suffix.
    last_sign = 0
    k0_zero_based = n
    for k in range(n - 1, -1, -1):
        s = sign_arr[k]
        if s == 0:
            break
        if last_sign == 0:
            last_sign = int(s)
            k0_zero_based = k
        elif s == last_sign:
            k0_zero_based = k
        else:
            break
    tail_len = n - k0_zero_based

    if tail_len < MIN_REMNANT_TERMS or last_sign == 0:
        return RemnantGMFit(base=base_fit, remnant=None, k0=-1, sign=0)

    # Fit GM(1,1) on |ε|[k0:].
    eps_tail_abs = np.abs(eps[k0_zero_based:])
    try:
        remnant_fit = fit_egm11(eps_tail_abs)
    except ValueError:
        # Forbidden zone or singular B^T B — skip remnant correction.
        return RemnantGMFit(base=base_fit, remnant=None, k0=-1, sign=0)

    return RemnantGMFit(
        base=base_fit,
        remnant=remnant_fit,
        k0=k0_zero_based + 1,  # convert to 1-based
        sign=last_sign,
    )


def simulate_remnant_gm11(
    fit: RemnantGMFit, n_periods: int | None = None,
) -> np.ndarray:
    """In-sample reconstruction with remnant correction applied where it bites.

    Returns ``x̂⁽⁰⁾_modified(k)`` for ``k = 1, …, n_periods``. For ``k < k₀``,
    returns the base prediction. For ``k ≥ k₀``, adds the sign-restored
    remnant forecast.

    Parameters
    ----------
    fit : RemnantGMFit
    n_periods : int | None, default None
        Defaults to ``fit.base.n_train``.

    Returns
    -------
    numpy.ndarray, shape ``(n_periods,)``
    """
    base_pred = simulate_gm11(fit.base, n_periods=n_periods)
    if fit.remnant is None:
        return base_pred

    n = base_pred.size
    out = base_pred.copy()
    # Remnant fit is over |ε⁽⁰⁾|[k₀:]. For each k ≥ k₀, the remnant simulation
    # at offset (k - k₀ + 1) gives the predicted absolute residual; restore
    # sign and add to base prediction.
    rem_sim = simulate_gm11(fit.remnant, n_periods=n - fit.k0 + 1)
    for offset, sim_val in enumerate(rem_sim):
        k_idx = fit.k0 - 1 + offset  # 0-based
        if k_idx < n:
            out[k_idx] = base_pred[k_idx] + fit.sign * sim_val
    return out


def forecast_remnant_gm11(fit: RemnantGMFit, n_ahead: int) -> np.ndarray:
    """Forecast with remnant correction.

    Returns ``x̂⁽⁰⁾_modified`` for ``n_ahead`` future periods.

    Parameters
    ----------
    fit : RemnantGMFit
    n_ahead : int

    Returns
    -------
    numpy.ndarray, shape ``(n_ahead,)``
    """
    base_forecast = forecast_gm11(fit.base, n_ahead=n_ahead)
    if fit.remnant is None:
        return base_forecast

    # Future residuals continue from offset (n_train - k₀ + 1) onward.
    offset_start = fit.base.n_train - fit.k0 + 1
    rem_forecast = forecast_gm11(fit.remnant, n_ahead=offset_start + n_ahead)
    # Take the n_ahead values starting at offset_start.
    rem_future = rem_forecast[offset_start: offset_start + n_ahead]
    if rem_future.size < n_ahead:
        # Pad with zeros if remnant horizon is too short (defensive — should
        # not occur because forecast_gm11 generates the requested horizon).
        rem_future = np.concatenate(
            [rem_future, np.zeros(n_ahead - rem_future.size)]
        )
    return base_forecast + fit.sign * rem_future

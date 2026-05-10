"""GM(1,1) — the canonical grey forecasting model. Liu §7.2.

Four basic forms are supported, each fit by ordinary least squares against
a different background-value formulation of the discrete grey model
``x⁽⁰⁾(k) + a·B(k) = b``:

============  =====================================================
Form          Background value ``B(k)``
============  =====================================================
**EGM**       ``z⁽¹⁾(k) = 0.5·(x⁽¹⁾(k) + x⁽¹⁾(k−1))``  (mean form, Liu §7.2.1)
**ODGM**      ``x⁽¹⁾(k−1)``  (original-difference form, Liu §7.2.1)
**DGM**       fitted directly in discrete time:
              ``x⁽¹⁾(k+1) = β₁·x⁽¹⁾(k) + β₂``  (Liu §7.2.8)
**EDGM**      ``z⁽¹⁾(k) = 0.5·(x⁽¹⁾(k) + x⁽¹⁾(k−1))`` but fit by even-difference (Liu §7.2.7)
============  =====================================================

All four forms share the :class:`GMFit` container; ``simulate_gm11`` and
``forecast_gm11`` dispatch on the ``form`` field to apply the form's own
time-response formula (Liu §7.2 Theorems 7.2.1, 7.2.2, 7.2.4).

References
----------
Liu, Sifeng (2024). *Grey Systems Analysis: Methods, Models and Applications*,
2nd Ed., Springer Singapore. §7.2 (forms), §7.3 (suitable ranges), §7.8 (worked
examples).
"""
from __future__ import annotations

import numpy as np

from gs_simulation.gm.base import GM11_FORBIDDEN_ABS_A, GMFit
from gs_simulation.operators.accumulation import apply_ago

__all__ = [
    "GMFit",
    "fit_egm11",
    "fit_odgm11",
    "fit_dgm11",
    "fit_edgm11",
    "simulate_gm11",
    "forecast_gm11",
]


# ---------------------------------------------------------------------------
# Common pre-fit infrastructure
# ---------------------------------------------------------------------------


def _validate_gm11_input(x: np.ndarray) -> np.ndarray:
    """Validate input sequence and return as float64 1-D array."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"GM(1,1) expects a 1-D sequence; got shape {arr.shape}"
        )
    if arr.size < 4:
        raise ValueError(
            f"GM(1,1) requires at least 4 observations to fit; got n={arr.size}"
        )
    return arr


def _solve_least_squares(B: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    """Solve normal equations ``â = (BᵀB)⁻¹ Bᵀ Y`` for two-parameter fits."""
    BtB = B.T @ B
    BtY = B.T @ Y
    try:
        params = np.linalg.solve(BtB, BtY)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"GM(1,1) least-squares failed: BᵀB is singular. {exc}"
        ) from exc
    return float(params[0]), float(params[1])


def _check_forbidden_zone(a: float) -> None:
    """Raise if ``|a| >= GM11_FORBIDDEN_ABS_A`` (Liu §7.3 Prop 7.3.2)."""
    if abs(a) >= GM11_FORBIDDEN_ABS_A:
        raise ValueError(
            f"GM(1,1) forbidden zone |a| >= {GM11_FORBIDDEN_ABS_A} "
            f"(Liu §7.3 Proposition 7.3.2); got a={a:.4f}. Model invalid."
        )


# ---------------------------------------------------------------------------
# Basic Form 1 — EGM (even / mean-background form, Liu §7.2.1)
# ---------------------------------------------------------------------------


def fit_egm11(x: np.ndarray) -> GMFit:
    """Fit the even-form GM(1,1) model.

    Discrete grey model ``x⁽⁰⁾(k) + a·z⁽¹⁾(k) = b`` with
    ``z⁽¹⁾(k) = 0.5·(x⁽¹⁾(k) + x⁽¹⁾(k−1))``.

    Parameters
    ----------
    x : array-like
        1-D training sequence ``X⁽⁰⁾``; needs ``n >= 4`` observations.

    Returns
    -------
    GMFit
        Fitted parameters with ``form="EGM"``.

    Raises
    ------
    ValueError
        If input is non-1-D, has fewer than 4 observations, the development
        coefficient lands in the forbidden zone, or ``BᵀB`` is singular.

    References
    ----------
    Liu (2024) §7.2 Definitions 7.2.3-7.2.4, Theorem 7.2.1.

    Examples
    --------
    >>> import numpy as np
    >>> fit = fit_egm11(np.array([27260.0, 29547.0, 32411.0, 35388.0]))
    >>> bool(np.isclose(fit.a, -0.089995, atol=1e-4))
    True
    """
    x0 = _validate_gm11_input(x)
    x1 = apply_ago(x0, order=1)
    n = x0.size

    # Background values: z⁽¹⁾(k) = 0.5(x⁽¹⁾(k) + x⁽¹⁾(k−1)) for k = 2..n.
    z1 = 0.5 * (x1[1:] + x1[:-1])  # shape (n-1,)
    B = np.column_stack([-z1, np.ones(n - 1)])
    Y = x0[1:]

    a, b = _solve_least_squares(B, Y)
    _check_forbidden_zone(a)

    return GMFit(
        a=a,
        b=b,
        x0_first=float(x0[0]),
        n_train=n,
        form="EGM",
    )


def _time_response_egm(fit: GMFit, k: int) -> float:
    """``x̂⁽¹⁾(k)`` for EGM. Liu §7.2 Theorem 7.2.1.

    ``x̂⁽¹⁾(k) = (x⁽⁰⁾(1) - b/a)·exp(-a·(k-1)) + b/a`` for ``k = 1, 2, …``.
    """
    a, b, x0_first = fit.a, fit.b, fit.x0_first
    return float((x0_first - b / a) * np.exp(-a * (k - 1)) + b / a)


# ---------------------------------------------------------------------------
# Basic Form 2 — ODGM (original-difference form, Liu §7.2.1)
# ---------------------------------------------------------------------------


def fit_odgm11(x: np.ndarray) -> GMFit:
    """Fit the original-difference GM(1,1) model.

    Discrete grey model ``x⁽⁰⁾(k) + a·x⁽¹⁾(k) = b`` (uses raw accumulation
    ``x⁽¹⁾(k)`` as background, NOT the mean ``z⁽¹⁾(k)``).

    Parameters
    ----------
    x : array-like

    Returns
    -------
    GMFit
        Fitted parameters with ``form="ODGM"``.

    References
    ----------
    Liu (2024) §7.2 Definition 7.2.1.
    """
    x0 = _validate_gm11_input(x)
    x1 = apply_ago(x0, order=1)
    n = x0.size

    # Background = x⁽¹⁾(k) for k = 2..n (no averaging).
    B = np.column_stack([-x1[1:], np.ones(n - 1)])
    Y = x0[1:]

    a, b = _solve_least_squares(B, Y)
    _check_forbidden_zone(a)

    return GMFit(
        a=a,
        b=b,
        x0_first=float(x0[0]),
        n_train=n,
        form="ODGM",
    )


def _time_response_odgm(fit: GMFit, k: int) -> float:
    """``x̂⁽¹⁾(k)`` for ODGM. Liu §7.2.

    ``x̂⁽¹⁾(k) = (x⁽⁰⁾(1) - b/a) · (1/(1+a))^k + b/a`` for ``k = 1, 2, …``.
    """
    a, b, x0_first = fit.a, fit.b, fit.x0_first
    return float((x0_first - b / a) * (1.0 / (1.0 + a)) ** k + b / a)


# ---------------------------------------------------------------------------
# Basic Form 3 — DGM (discrete grey model, Liu §7.2.8)
# ---------------------------------------------------------------------------


def fit_dgm11(x: np.ndarray) -> GMFit:
    """Fit the discrete GM(1,1) model.

    Discrete model ``x⁽¹⁾(k+1) = β₁·x⁽¹⁾(k) + β₂``, fit by least squares.
    For backward compatibility with the (a, b) representation we store the
    equivalent EGM-style ``(a, b)`` derived from ``(β₁, β₂)`` via
    ``a = -ln(β₁)``, ``b = β₂·(1 - β₁) / a`` when applicable; the
    DGM-specific time response uses ``β₁/β₂`` directly.

    Parameters
    ----------
    x : array-like

    Returns
    -------
    GMFit
        Fitted parameters with ``form="DGM"``. ``a = β₁`` and ``b = β₂``
        are stored directly (the dataclass slots are reused; the
        ``form`` discriminator selects the correct interpretation).

    References
    ----------
    Liu (2024) §7.2 Definition 7.2.8, Theorem 7.2.2.
    """
    x0 = _validate_gm11_input(x)
    x1 = apply_ago(x0, order=1)
    n = x0.size

    # Regress x⁽¹⁾(k+1) on x⁽¹⁾(k): rows [x⁽¹⁾(k), 1] for k = 1..n-1.
    B = np.column_stack([x1[:-1], np.ones(n - 1)])
    Y = x1[1:]

    beta1, beta2 = _solve_least_squares(B, Y)

    if beta1 == 1.0:
        raise ValueError(
            "DGM fit failed: β₁ = 1 yields a degenerate (non-decaying) "
            "time response. Try EGM or ODGM instead."
        )

    return GMFit(
        a=beta1,
        b=beta2,
        x0_first=float(x0[0]),
        n_train=n,
        form="DGM",
    )


def _time_response_dgm(fit: GMFit, k: int) -> float:
    """``x̂⁽¹⁾(k)`` for DGM. Liu §7.2 Theorem 7.2.2.

    ``x̂⁽¹⁾(k) = (x⁽⁰⁾(1) - β₂/(1-β₁)) · β₁^(k-1) + β₂/(1-β₁)``.

    Note: in our :class:`GMFit` storage, DGM uses ``a = β₁`` and ``b = β₂``.
    """
    beta1, beta2, x0_first = fit.a, fit.b, fit.x0_first
    fixed_pt = beta2 / (1.0 - beta1)
    return float((x0_first - fixed_pt) * beta1 ** (k - 1) + fixed_pt)


# ---------------------------------------------------------------------------
# Basic Form 4 — EDGM (even-difference form, Liu §7.2.7)
# ---------------------------------------------------------------------------


def fit_edgm11(x: np.ndarray) -> GMFit:
    """Fit the even-difference GM(1,1) model.

    Same B/Y construction as EGM (mean background) but a different
    time-response formula uses the discrete approximation
    ``(1 − 0.5a)/(1 + 0.5a)`` rather than ``exp(−a)`` (Liu §7.2.7).

    Parameters
    ----------
    x : array-like

    Returns
    -------
    GMFit
        Fitted parameters with ``form="EDGM"``.

    References
    ----------
    Liu (2024) §7.2 Definition 7.2.7, Theorem 7.2.4.
    """
    fit = fit_egm11(x)
    # Same (a, b) — discriminator differs.
    return GMFit(
        a=fit.a,
        b=fit.b,
        x0_first=fit.x0_first,
        n_train=fit.n_train,
        form="EDGM",
    )


def _time_response_edgm(fit: GMFit, k: int) -> float:
    """``x̂⁽¹⁾(k)`` for EDGM. Liu §7.2 Theorem 7.2.4.

    ``x̂⁽¹⁾(k) = (x⁽⁰⁾(1) - b/a) · ((1 - 0.5a)/(1 + 0.5a))^k + b/a``.
    """
    a, b, x0_first = fit.a, fit.b, fit.x0_first
    ratio = (1.0 - 0.5 * a) / (1.0 + 0.5 * a)
    return float((x0_first - b / a) * ratio ** k + b / a)


# ---------------------------------------------------------------------------
# Simulate / forecast (form-aware dispatch)
# ---------------------------------------------------------------------------


_TIME_RESPONSE_DISPATCH = {
    "EGM": _time_response_egm,
    "ODGM": _time_response_odgm,
    "DGM": _time_response_dgm,
    "EDGM": _time_response_edgm,
}


def _x1_hat_at(fit: GMFit, k: int) -> float:
    """Return ``x̂⁽¹⁾(k)`` (1-based) using the fit's form-specific formula."""
    try:
        fn = _TIME_RESPONSE_DISPATCH[fit.form]
    except KeyError as exc:
        raise ValueError(f"Unknown GM(1,1) form: {fit.form!r}") from exc
    return fn(fit, k)


def simulate_gm11(fit: GMFit, n_periods: int | None = None) -> np.ndarray:
    """Reconstruct ``x̂⁽⁰⁾`` for ``k = 1, …, n_periods``.

    For ``k = 1``, returns ``x⁽⁰⁾(1)`` (the initial value, by convention).
    For ``k >= 2``, applies IAGO to the time-response sequence:
    ``x̂⁽⁰⁾(k) = x̂⁽¹⁾(k) − x̂⁽¹⁾(k−1)``.

    Parameters
    ----------
    fit : GMFit
    n_periods : int | None, default None
        If ``None``, defaults to ``fit.n_train`` (full in-sample
        reconstruction).

    Returns
    -------
    numpy.ndarray, shape ``(n_periods,)``
        Reconstructed ``X̂⁽⁰⁾`` values.
    """
    if n_periods is None:
        n_periods = fit.n_train
    if n_periods < 1:
        raise ValueError(f"n_periods must be >= 1; got {n_periods}")

    x1_hat = np.array(
        [_x1_hat_at(fit, k) for k in range(1, n_periods + 1)],
        dtype=float,
    )
    x0_hat = np.empty(n_periods, dtype=float)
    x0_hat[0] = fit.x0_first
    if n_periods > 1:
        x0_hat[1:] = x1_hat[1:] - x1_hat[:-1]
    return x0_hat


def forecast_gm11(fit: GMFit, n_ahead: int) -> np.ndarray:
    """Forecast ``x̂⁽⁰⁾`` for ``n_ahead`` future periods.

    Returns predictions for ``k = n_train + 1, …, n_train + n_ahead``.

    Parameters
    ----------
    fit : GMFit
    n_ahead : int
        Number of future periods to forecast (>= 1).

    Returns
    -------
    numpy.ndarray, shape ``(n_ahead,)``
    """
    if n_ahead < 1:
        raise ValueError(f"n_ahead must be >= 1; got {n_ahead}")

    n = fit.n_train
    # Need x̂⁽¹⁾ at k = n, n+1, …, n+n_ahead so the (n_ahead) differences
    # x̂⁽⁰⁾(n+h) = x̂⁽¹⁾(n+h) - x̂⁽¹⁾(n+h-1) can be computed.
    ks = range(n, n + n_ahead + 1)
    x1_hat = np.array([_x1_hat_at(fit, k) for k in ks], dtype=float)
    return x1_hat[1:] - x1_hat[:-1]

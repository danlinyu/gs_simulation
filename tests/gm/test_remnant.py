"""Tests for Remnant GM(1,1) — residual-modification model (Liu §7.4)."""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.gm.gm11 import fit_egm11, simulate_gm11
from gs_simulation.gm.remnant import (
    RemnantGMFit,
    fit_remnant_gm11,
    forecast_remnant_gm11,
    simulate_remnant_gm11,
)


def test_remnant_gm_returns_remnant_gm_fit() -> None:
    x = np.array([10.0, 12.0, 14.5, 17.0, 19.8, 23.0, 26.5])
    fit = fit_remnant_gm11(x)
    assert isinstance(fit, RemnantGMFit)
    assert fit.base.n_train == x.size


def test_remnant_skipped_when_residuals_oscillate() -> None:
    """If sign of ε⁽⁰⁾ alternates, no constant-sign tail of length ≥ 4 exists."""
    rng = np.random.default_rng(42)
    base_x = np.cumsum(rng.normal(loc=1.0, scale=0.05, size=20)) + 10
    fit = fit_remnant_gm11(base_x)
    # Stochastic residuals ⇒ constant-sign suffix of length ≥ 4 unlikely.
    # If by chance the suffix happens to qualify, just assert the contract.
    if fit.remnant is None:
        assert fit.k0 == -1
        assert fit.sign == 0
    else:
        assert fit.remnant.n_train >= 4


def test_remnant_simulation_returns_finite_values() -> None:
    """Remnant simulation must return finite values across all periods."""
    n = 10
    x0 = np.array([1.0 + 0.02 * (k ** 2.5) for k in range(n)]) + 1.0
    fit = fit_remnant_gm11(x0)
    sim = simulate_remnant_gm11(fit)
    assert sim.shape == (n,)
    assert np.all(np.isfinite(sim))


def test_remnant_first_value_equals_base_first_value() -> None:
    """x̂⁽⁰⁾(1) is x⁽⁰⁾(1) by convention; remnant cannot change k=1 if k₀ > 1."""
    x = np.array([1.0, 1.5, 2.3, 3.4, 4.9, 6.8, 9.2, 12.1])
    fit = fit_remnant_gm11(x)
    sim = simulate_remnant_gm11(fit)
    if fit.remnant is None or fit.k0 > 1:
        np.testing.assert_allclose(sim[0], x[0], atol=1e-12)


def test_remnant_first_k0_minus_1_periods_match_base() -> None:
    """For k < k₀, remnant prediction equals base prediction by construction."""
    x = np.array([1.0, 1.5, 2.3, 3.4, 4.9, 6.8, 9.2, 12.1])
    fit = fit_remnant_gm11(x)

    base_sim = simulate_gm11(fit.base)
    rem_sim = simulate_remnant_gm11(fit)

    if fit.remnant is None:
        # No remnant ⇒ rem == base everywhere.
        np.testing.assert_allclose(rem_sim, base_sim, atol=1e-12)
    else:
        # Pre-k₀ entries unchanged.
        for k in range(fit.k0 - 1):
            np.testing.assert_allclose(rem_sim[k], base_sim[k], atol=1e-12)


def test_remnant_forecast_returns_correct_horizon() -> None:
    x = np.array([1.0, 1.5, 2.3, 3.4, 4.9, 6.8, 9.2, 12.1])
    fit = fit_remnant_gm11(x)
    forecast = forecast_remnant_gm11(fit, n_ahead=5)
    assert forecast.shape == (5,)
    assert np.all(np.isfinite(forecast))


def test_remnant_forecast_falls_back_to_base_when_no_remnant() -> None:
    """If remnant is None, forecast equals base forecast."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0.5, 0.1, 6)) + 5  # likely no modelable residuals
    fit = fit_remnant_gm11(x)
    if fit.remnant is not None:
        pytest.skip("Random seed produced modelable residuals; skipping.")
    from gs_simulation.gm.gm11 import forecast_gm11
    base_forecast = forecast_gm11(fit.base, n_ahead=3)
    rem_forecast = forecast_remnant_gm11(fit, n_ahead=3)
    np.testing.assert_allclose(rem_forecast, base_forecast, atol=1e-12)


def test_remnant_uses_provided_base_fit() -> None:
    """Caller can pass a pre-computed base_fit (e.g., a non-EGM form)."""
    from gs_simulation.gm.gm11 import fit_dgm11
    x = np.array([1.0, 1.5, 2.3, 3.4, 4.9, 6.8, 9.2, 12.1])
    base_fit = fit_dgm11(x)
    fit = fit_remnant_gm11(x, base_fit=base_fit)
    assert fit.base is base_fit
    assert fit.base.form == "DGM"

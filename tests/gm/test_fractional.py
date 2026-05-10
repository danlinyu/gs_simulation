"""Tests for Fractional GM(p/q, 1) — Liu §7.6."""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.gm.fractional import (
    FractionalGMFit,
    apply_fractional_ago,
    fit_fractional_gm11,
    forecast_fractional_gm11,
    simulate_fractional_gm11,
)
from gs_simulation.operators.accumulation import apply_ago, apply_iago


# ---------------------------------------------------------------------------
# Fractional AGO operator — special cases against integer AGO/IAGO
# ---------------------------------------------------------------------------


def test_fractional_ago_order_1_equals_integer_ago() -> None:
    x = np.array([5.3, 7.6, 10.4, 13.8, 18.1])
    np.testing.assert_allclose(
        apply_fractional_ago(x, order=1.0),
        apply_ago(x, order=1),
        atol=1e-10,
    )


def test_fractional_ago_order_neg1_equals_integer_iago() -> None:
    x = np.array([5.3, 12.9, 23.3, 37.1, 55.2])
    np.testing.assert_allclose(
        apply_fractional_ago(x, order=-1.0),
        apply_iago(x, order=1),
        atol=1e-10,
    )


def test_fractional_ago_order_0_is_identity() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        apply_fractional_ago(x, order=0.0), x, atol=1e-12,
    )


@pytest.mark.parametrize("order", [0.1, 0.25, 0.5, 0.75, 1.5, -0.5, -2.0])
def test_fractional_ago_inverts_with_negated_order(order: float) -> None:
    """apply(apply(x, r), -r) ≈ x for any real r."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(8) * 5 + 3
    recovered = apply_fractional_ago(apply_fractional_ago(x, order=order), order=-order)
    np.testing.assert_allclose(recovered, x, atol=1e-9)


def test_fractional_ago_first_element_unchanged() -> None:
    """At any order r, x^(r)[0] = x[0] (since c_0 = 1)."""
    x = np.array([7.5, 1.0, 2.0])
    for r in (0.3, 0.7, 1.0, -0.5):
        assert apply_fractional_ago(x, order=r)[0] == x[0]


def test_fractional_ago_handles_empty() -> None:
    out = apply_fractional_ago(np.array([], dtype=float), order=0.5)
    assert out.size == 0


# ---------------------------------------------------------------------------
# Fractional GM(p/q, 1) fit + simulate + forecast
# ---------------------------------------------------------------------------


def test_fit_fractional_gm11_returns_dataclass() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    fit = fit_fractional_gm11(x, order=0.5)
    assert isinstance(fit, FractionalGMFit)
    assert fit.order == 0.5
    assert fit.n_train == 5
    assert fit.x0_first == 1.0


def test_fit_fractional_gm11_at_order_1_is_close_to_dgm() -> None:
    """At order=1, fractional AGO = 1-AGO; the fractional GM fit should
    match a direct DGM fit on x⁽¹⁾.
    """
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    fit_frac = fit_fractional_gm11(x, order=1.0)
    sim_frac = simulate_fractional_gm11(fit_frac)
    # Sanity: simulated values should be close to original (exponential growth
    # is GM(1,1)-friendly).
    rel_err = np.max(np.abs(sim_frac - x) / np.abs(x))
    assert rel_err < 0.05


def test_simulate_fractional_returns_correct_length() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    fit = fit_fractional_gm11(x, order=0.7)
    sim = simulate_fractional_gm11(fit)
    assert sim.shape == (fit.n_train,)
    sim_extended = simulate_fractional_gm11(fit, n_periods=8)
    assert sim_extended.shape == (8,)


def test_forecast_fractional_returns_correct_horizon() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    fit = fit_fractional_gm11(x, order=0.5)
    forecast = forecast_fractional_gm11(fit, n_ahead=3)
    assert forecast.shape == (3,)
    assert np.all(np.isfinite(forecast))


def test_fit_fractional_too_few_observations_raises() -> None:
    with pytest.raises(ValueError, match="n ≥ 4"):
        fit_fractional_gm11(np.array([1.0, 2.0, 3.0]), order=0.5)


def test_fit_fractional_2d_input_raises() -> None:
    with pytest.raises(ValueError, match="1-D"):
        fit_fractional_gm11(np.array([[1.0, 2.0], [3.0, 4.0]]), order=0.5)


def test_simulate_negative_n_periods_raises() -> None:
    fit = fit_fractional_gm11(np.array([1.0, 2.0, 4.0, 8.0]), order=0.5)
    with pytest.raises(ValueError, match="n_periods"):
        simulate_fractional_gm11(fit, n_periods=0)


def test_forecast_negative_horizon_raises() -> None:
    fit = fit_fractional_gm11(np.array([1.0, 2.0, 4.0, 8.0]), order=0.5)
    with pytest.raises(ValueError, match="n_ahead"):
        forecast_fractional_gm11(fit, n_ahead=0)

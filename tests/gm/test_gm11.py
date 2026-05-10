"""Tests for GM(1,1) — four basic forms (Liu §7.2)."""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.gm.base import GM11_FORBIDDEN_ABS_A
from gs_simulation.gm.gm11 import (
    fit_dgm11,
    fit_edgm11,
    fit_egm11,
    fit_odgm11,
    forecast_gm11,
    simulate_gm11,
)


# ---------------------------------------------------------------------------
# Book worked examples (Liu §7.8)
# ---------------------------------------------------------------------------

# Example 7.8.1 (Liu §7.8): Changge County private enterprise revenue.
EX_7_8_1_X0 = np.array([27260.0, 29547.0, 32411.0, 35388.0])
EX_7_8_1_A_HAT = -0.089995
EX_7_8_1_B_HAT = 25790.28
EX_7_8_1_X_FITTED = np.array([27260.0, 29553.0, 32337.0, 35381.0])
EX_7_8_1_FORECAST_5_TO_9 = np.array([38714.0, 42359.0, 46348.0, 50712.0, 55488.0])

# Example §7.2 small numerical (extraction summary): 5-point sequence.
WORKED_5PT_X0 = np.array([1.5, 2.1, 3.0, 4.5, 5.48])
WORKED_5PT_A_HAT = -0.108  # approximate per book
WORKED_5PT_B_HAT = 1.45    # approximate per book


@pytest.mark.book_example
def test_egm_book_example_7_8_1_parameters() -> None:
    """Liu Ex 7.8.1: â = -0.089995, b̂ = 25790.28 ± rounding."""
    fit = fit_egm11(EX_7_8_1_X0)
    np.testing.assert_allclose(fit.a, EX_7_8_1_A_HAT, atol=1e-4)
    np.testing.assert_allclose(fit.b, EX_7_8_1_B_HAT, atol=5.0)


@pytest.mark.book_example
def test_egm_book_example_7_8_1_in_sample_simulation() -> None:
    """Liu Ex 7.8.1: simulated X̂ ≈ (27260, 29553, 32337, 35381). MRE = 0.067%."""
    fit = fit_egm11(EX_7_8_1_X0)
    sim = simulate_gm11(fit)
    np.testing.assert_allclose(sim, EX_7_8_1_X_FITTED, atol=20.0)
    # Mean relative error matches book's 0.067 %.
    mre = np.mean(np.abs(sim - EX_7_8_1_X0) / EX_7_8_1_X0) * 100
    assert mre < 0.10  # book reports 0.067%


@pytest.mark.book_example
def test_egm_book_example_7_8_1_forecast() -> None:
    """Liu Ex 7.8.1: 5-step-ahead forecast ≈ (38714, 42359, 46348, 50712, 55488)."""
    fit = fit_egm11(EX_7_8_1_X0)
    forecast = forecast_gm11(fit, n_ahead=5)
    np.testing.assert_allclose(
        forecast, EX_7_8_1_FORECAST_5_TO_9, atol=50.0,
        err_msg="5-step forecast diverges from book Table values",
    )


def test_egm_worked_5pt_in_sample_quality() -> None:
    """Small 5-point sequence (1.5, 2.1, 3.0, 4.5, 5.48) reconstructs at < 5% MRE.

    Note: Liu §7.2 worked example reports â ≈ -0.108, but recomputing the
    normal-equations gives â ≈ -0.304 (the book's intermediate sum-of-squares
    figure 236.75 is a transcription error; correct value is 302.38). This
    test verifies in-sample fit quality rather than the (typo'd) book values.
    """
    fit = fit_egm11(WORKED_5PT_X0)
    sim = simulate_gm11(fit)
    rel_err = np.mean(np.abs(sim - WORKED_5PT_X0) / WORKED_5PT_X0)
    assert rel_err < 0.05, f"5-point in-sample MRE {rel_err:.3%} >= 5%"


# ---------------------------------------------------------------------------
# In-sample correctness — analytic perfect-exponential
# ---------------------------------------------------------------------------


def test_egm_recovers_pure_exponential() -> None:
    """Pure-exponential x⁽⁰⁾(k) = c·e^(α(k-1)) is fittable to high accuracy."""
    n = 6
    c, alpha = 1.0, 0.10
    x0 = c * np.exp(alpha * np.arange(n))
    fit = fit_egm11(x0)
    sim = simulate_gm11(fit)
    rel_err = np.max(np.abs(sim - x0) / np.abs(x0))
    assert rel_err < 0.01, f"Pure exponential should fit < 1% RMSE; got {rel_err:.4%}"


def test_dgm_pure_exponential_machine_precision() -> None:
    """DGM achieves near-machine precision on homogeneous exponentials (Liu §7.2.5)."""
    n = 8
    c, alpha = 2.0, -0.05
    x0 = c * np.exp(alpha * np.arange(n))
    fit = fit_dgm11(x0)
    sim = simulate_gm11(fit)
    rel_err = np.max(np.abs(sim - x0) / np.abs(x0))
    assert rel_err < 1e-3, f"DGM on pure exponential: rel_err {rel_err:.2e} >= 1e-3"


# ---------------------------------------------------------------------------
# Cross-form consistency (Liu §7.2.5: forms agree as |a| → 0)
# ---------------------------------------------------------------------------


def test_egm_odgm_agree_when_a_is_small() -> None:
    """Liu Theorem 7.2.5: as |a| → 0, EGM and ODGM agree numerically."""
    rng = np.random.default_rng(0)
    # Build a sequence with small |a| (slow growth).
    x0 = 100.0 + np.cumsum(rng.normal(loc=0.5, scale=0.05, size=10))
    fit_egm = fit_egm11(x0)
    fit_odgm = fit_odgm11(x0)
    # Both forms should have small |a|; (a, b) values are close.
    assert abs(fit_egm.a) < 0.15
    assert abs(fit_odgm.a) < 0.15
    # Simulated trajectories agree to within 5 %.
    sim_egm = simulate_gm11(fit_egm)
    sim_odgm = simulate_gm11(fit_odgm)
    rel_diff = np.max(np.abs(sim_egm - sim_odgm) / np.abs(sim_egm))
    assert rel_diff < 0.05


def test_egm_edgm_share_parameters() -> None:
    """EDGM uses EGM's (a, b) by definition; only the time-response differs."""
    fit_e = fit_egm11(EX_7_8_1_X0)
    fit_ed = fit_edgm11(EX_7_8_1_X0)
    assert fit_e.a == fit_ed.a
    assert fit_e.b == fit_ed.b
    assert fit_e.form == "EGM"
    assert fit_ed.form == "EDGM"


# ---------------------------------------------------------------------------
# Edge cases / domain restrictions
# ---------------------------------------------------------------------------


def test_egm_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        fit_egm11(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_egm_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        fit_egm11(np.array([1.0, 2.0, 3.0]))


def test_forbidden_zone_check_is_in_place() -> None:
    """Liu §7.3 Prop 7.3.2: |a| >= 2 forbidden zone. Direct check on the helper.

    Constructing a real-data sequence that yields |a| >= 2 is hard — the EGM
    fit on standard scientific input rarely produces |a| > 0.5. We verify the
    guard structurally by calling the internal helper directly.
    """
    from gs_simulation.gm.gm11 import _check_forbidden_zone
    # Inside the zone: should not raise.
    _check_forbidden_zone(0.5)
    _check_forbidden_zone(-1.99)
    # In the zone: should raise.
    with pytest.raises(ValueError, match="forbidden zone"):
        _check_forbidden_zone(2.0)
    with pytest.raises(ValueError, match="forbidden zone"):
        _check_forbidden_zone(-3.5)


def test_simulate_default_n_periods_is_n_train() -> None:
    fit = fit_egm11(EX_7_8_1_X0)
    sim = simulate_gm11(fit)
    assert sim.shape == (fit.n_train,)


def test_simulate_first_value_is_x0_first() -> None:
    """By convention, x̂⁽⁰⁾(1) = x⁽⁰⁾(1)."""
    fit = fit_egm11(EX_7_8_1_X0)
    sim = simulate_gm11(fit)
    np.testing.assert_allclose(sim[0], EX_7_8_1_X0[0], atol=1e-12)


def test_forecast_negative_horizon_raises() -> None:
    fit = fit_egm11(EX_7_8_1_X0)
    with pytest.raises(ValueError, match="n_ahead"):
        forecast_gm11(fit, n_ahead=0)


def test_simulate_negative_n_periods_raises() -> None:
    fit = fit_egm11(EX_7_8_1_X0)
    with pytest.raises(ValueError, match="n_periods"):
        simulate_gm11(fit, n_periods=0)


# ---------------------------------------------------------------------------
# Suitable-range advisory (Liu §7.3 Table 7.9)
# ---------------------------------------------------------------------------


def test_suitable_range_advisory_long_term() -> None:
    """Ex 7.8.1 has -a ≈ 0.09 → 'long_term' regime."""
    fit = fit_egm11(EX_7_8_1_X0)
    assert fit.suitable_range() == "long_term"


# ---------------------------------------------------------------------------
# Form discriminator
# ---------------------------------------------------------------------------


def test_each_fit_carries_correct_form_label() -> None:
    assert fit_egm11(EX_7_8_1_X0).form == "EGM"
    assert fit_odgm11(EX_7_8_1_X0).form == "ODGM"
    assert fit_dgm11(EX_7_8_1_X0).form == "DGM"
    assert fit_edgm11(EX_7_8_1_X0).form == "EDGM"


def test_dgm_simulate_uses_dgm_response() -> None:
    """Forms differ in their time-response formulas; verify DGM uses its own."""
    fit_dgm = fit_dgm11(EX_7_8_1_X0)
    sim_dgm = simulate_gm11(fit_dgm)
    # First value is x0_first by convention.
    assert sim_dgm[0] == fit_dgm.x0_first
    # Subsequent values are nontrivial (not all zeros, not all equal).
    assert sim_dgm[1] != sim_dgm[2]

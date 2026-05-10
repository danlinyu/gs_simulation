"""Tests for Deng's Grey Relational Degree (Liu §5.3)."""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.relational.deng import (
    deng_relational_coefficient,
    deng_relational_degree,
)


# ---------------------------------------------------------------------------
# Identity / extreme cases — the property-based correctness floor
# ---------------------------------------------------------------------------


def test_identical_sequences_yield_unity() -> None:
    """Liu §5.3 Property 1: γ_{0i} = 1 ⟺ X_0' = X_i' (Theorem 5.3.2)."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert deng_relational_degree(x, x) == 1.0


def test_proportional_sequences_yield_unity_under_d1() -> None:
    """D₁ normalizes by first value; (1,2,3) and (10,20,30) become equal."""
    x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x1 = 10.0 * x0
    np.testing.assert_allclose(
        deng_relational_degree(x0, x1, operator="D1"), 1.0, atol=1e-12,
    )


def test_relational_degree_in_unit_interval() -> None:
    """Liu §5.3 Property 1: 0 < γ_{0i} ≤ 1 always."""
    rng = np.random.default_rng(42)
    x0 = rng.standard_normal(8)
    x1 = rng.standard_normal(8)
    gamma = deng_relational_degree(x0, x1, operator="D2")
    assert 0.0 < gamma <= 1.0


def test_reverse_sequence_has_lower_degree_than_identity() -> None:
    """Reversed sequence is geometrically less similar than identity."""
    x0 = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    x_rev = x0[::-1].copy()
    gamma_id = deng_relational_degree(x0, x0, operator="D1")
    gamma_rev = deng_relational_degree(x0, x_rev, operator="D1")
    assert gamma_rev < gamma_id


# ---------------------------------------------------------------------------
# Multi-sequence (2-D) input
# ---------------------------------------------------------------------------


def test_multiple_sequences_returned_as_array() -> None:
    """When Xi is 2-D, return one γ per row."""
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    Xi = np.array([
        [1.0, 2.0, 3.0, 4.0],   # identical
        [2.0, 4.0, 6.0, 8.0],   # proportional (γ=1 under D1)
        [4.0, 3.0, 2.0, 1.0],   # reversed (lower γ)
    ])
    gammas = deng_relational_degree(x0, Xi, operator="D1")
    assert gammas.shape == (3,)
    np.testing.assert_allclose(gammas[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(gammas[1], 1.0, atol=1e-12)
    assert gammas[2] < 1.0


def test_multi_sequence_ranking_consistent() -> None:
    """A more-similar sequence ranks higher than a less-similar one."""
    x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Xi = np.array([
        [1.1, 2.05, 3.02, 4.01, 5.005],  # very close to x0
        [5.0, 4.0, 3.0, 2.0, 1.0],       # reversed
    ])
    gammas = deng_relational_degree(x0, Xi, operator="D1")
    assert gammas[0] > gammas[1]


# ---------------------------------------------------------------------------
# Operator selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["D1", "D2", "D3", "none"])
def test_all_operators_produce_unit_interval_results(op: str) -> None:
    rng = np.random.default_rng(0)
    x0 = rng.uniform(1.0, 5.0, size=10)
    x1 = rng.uniform(1.0, 5.0, size=10)
    gamma = deng_relational_degree(x0, x1, operator=op)
    assert 0.0 < gamma <= 1.0 + 1e-12


def test_unknown_operator_raises() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="Unknown operator"):
        deng_relational_degree(x, x, operator="DX")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Discrimination coefficient ξ
# ---------------------------------------------------------------------------


def test_discrimination_coefficient_default_is_half() -> None:
    """Default ξ = 0.5 per Liu §5.3 Step 4."""
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    x1 = np.array([1.5, 2.5, 3.5, 4.5])
    gamma_default = deng_relational_degree(x0, x1, operator="D1")
    gamma_explicit_half = deng_relational_degree(x0, x1, xi=0.5, operator="D1")
    assert gamma_default == gamma_explicit_half


def test_xi_out_of_range_raises() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match=r"ξ must be in"):
        deng_relational_degree(x, x, xi=0.0)
    with pytest.raises(ValueError, match=r"ξ must be in"):
        deng_relational_degree(x, x, xi=1.5)


def test_smaller_xi_amplifies_differences() -> None:
    """Smaller ξ → coefficients more sensitive to per-point differences."""
    x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x1 = np.array([1.0, 2.5, 3.0, 5.0, 5.0])
    gamma_lo = deng_relational_degree(x0, x1, xi=0.1, operator="D1")
    gamma_hi = deng_relational_degree(x0, x1, xi=0.9, operator="D1")
    # Smaller ξ → more discrimination → smaller γ for the same difference.
    assert gamma_lo < gamma_hi


# ---------------------------------------------------------------------------
# Coefficient (per-point) variant
# ---------------------------------------------------------------------------


def test_coefficient_returns_per_point_array() -> None:
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    x1 = np.array([1.0, 2.5, 3.0, 5.0])
    coeff = deng_relational_coefficient(x0, x1, operator="D1")
    assert coeff.shape == (4,)
    assert np.all((coeff > 0) & (coeff <= 1.0 + 1e-12))


def test_coefficient_average_equals_degree() -> None:
    """γ_{0i} = mean over k of γ_{0i}(k) by construction (Liu §5.3 Step 5)."""
    rng = np.random.default_rng(7)
    x0 = rng.uniform(1.0, 5.0, size=8)
    x1 = rng.uniform(1.0, 5.0, size=8)
    coeff = deng_relational_coefficient(x0, x1, operator="D1")
    degree = deng_relational_degree(x0, x1, operator="D1")
    np.testing.assert_allclose(coeff.mean(), degree, atol=1e-12)


def test_coefficient_2d_input_returns_2d_array() -> None:
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    Xi = np.array([
        [1.5, 2.5, 3.5, 4.5],
        [2.0, 1.0, 0.5, 0.25],
    ])
    coeff = deng_relational_coefficient(x0, Xi, operator="D1")
    assert coeff.shape == (2, 4)


# ---------------------------------------------------------------------------
# Shape / type
# ---------------------------------------------------------------------------


def test_x0_2d_raises() -> None:
    with pytest.raises(ValueError, match="x0 must be 1-D"):
        deng_relational_degree(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([1.0, 2.0]),
        )


def test_xi_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must match x0 length"):
        deng_relational_degree(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_degree_returns_float_for_1d_xi() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    out = deng_relational_degree(x, x)
    assert isinstance(out, float)


def test_degree_returns_array_for_2d_xi() -> None:
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    Xi = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    out = deng_relational_degree(x0, Xi)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)

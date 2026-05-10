"""Tests for Accumulation Generation Operator (AGO) and Inverse AGO (IAGO).

Liu §4.6 — the load-bearing transformation that turns chaotic sequences into
quasi-exponential ones suitable for GM(1,1) fitting.
"""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.operators.accumulation import apply_ago, apply_iago


# ---------------------------------------------------------------------------
# Book worked example (Liu §4.6, Example 4.6.1)
# ---------------------------------------------------------------------------

# Liu §4.6.4 Example 4.6.1: X⁰ = (5.3, 7.6, 10.4, 13.8, 18.1)
BOOK_X0 = np.array([5.3, 7.6, 10.4, 13.8, 18.1])
BOOK_X1 = np.array([5.3, 12.9, 23.3, 37.1, 55.2])
BOOK_X2 = np.array([5.3, 18.2, 41.5, 78.6, 133.8])
BOOK_IAGO = np.array([5.3, 2.3, 2.8, 3.4, 4.3])


@pytest.mark.book_example
def test_ago_1_book_example() -> None:
    """Liu §4.6 Ex 4.6.1: 1-AGO of (5.3, 7.6, 10.4, 13.8, 18.1) = (5.3, 12.9, 23.3, 37.1, 55.2)."""
    actual = apply_ago(BOOK_X0, order=1)
    np.testing.assert_allclose(actual, BOOK_X1, atol=1e-10)


@pytest.mark.book_example
def test_ago_2_book_example() -> None:
    """Liu §4.6 Ex 4.6.1: 2-AGO of same input = (5.3, 18.2, 41.5, 78.6, 133.8)."""
    actual = apply_ago(BOOK_X0, order=2)
    np.testing.assert_allclose(actual, BOOK_X2, atol=1e-10)


@pytest.mark.book_example
def test_iago_1_book_example() -> None:
    """Liu §4.6 Ex 4.6.1: 1-IAGO of (5.3, 7.6, 10.4, 13.8, 18.1) = (5.3, 2.3, 2.8, 3.4, 4.3).

    IAGO is the discrete first-difference with the first element preserved.
    """
    actual = apply_iago(BOOK_X0, order=1)
    np.testing.assert_allclose(actual, BOOK_IAGO, atol=1e-10)


# ---------------------------------------------------------------------------
# Invertibility — AGO and IAGO are mathematical inverses (Prop. 4.6.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_iago_inverts_ago(order: int) -> None:
    """Liu §4.6.3 Prop 4.6.1: α^(r) X^(r) = X^(0)."""
    rng = np.random.default_rng(42)
    x0 = rng.standard_normal(10) * 5 + 3
    x_r = apply_ago(x0, order=order)
    x_recovered = apply_iago(x_r, order=order)
    np.testing.assert_allclose(x_recovered, x0, atol=1e-10)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_ago_inverts_iago(order: int) -> None:
    """AGO ∘ IAGO is identity at any order (the other direction)."""
    rng = np.random.default_rng(7)
    x0 = rng.standard_normal(8) * 2 + 5
    x_iago = apply_iago(x0, order=order)
    x_recovered = apply_ago(x_iago, order=order)
    np.testing.assert_allclose(x_recovered, x0, atol=1e-10)


# ---------------------------------------------------------------------------
# Element-by-element correctness
# ---------------------------------------------------------------------------


def test_ago_is_running_sum() -> None:
    """1-AGO of x is the cumulative sum: x_AGO[k] = Σ_{i=0}^{k} x[i]."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.array([1.0, 3.0, 6.0, 10.0])
    np.testing.assert_allclose(apply_ago(x, order=1), expected, atol=1e-12)


def test_iago_is_first_difference_with_initial() -> None:
    """1-IAGO of x is (x[0], x[1]-x[0], x[2]-x[1], ...)."""
    x = np.array([1.0, 3.0, 6.0, 10.0])
    expected = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(apply_iago(x, order=1), expected, atol=1e-12)


def test_ago_first_element_unchanged() -> None:
    """x_AGO[0] always equals x[0] (running sum of one element)."""
    x = np.array([7.5, 1.0, 2.0])
    assert apply_ago(x)[0] == 7.5


def test_ago_order_zero_is_identity() -> None:
    """0-AGO is the identity (no transformation)."""
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(apply_ago(x, order=0), x)


def test_iago_order_zero_is_identity() -> None:
    """0-IAGO is the identity (no transformation)."""
    x = np.array([1.0, 3.0, 6.0])
    np.testing.assert_array_equal(apply_iago(x, order=0), x)


# ---------------------------------------------------------------------------
# Shape / axis handling
# ---------------------------------------------------------------------------


def test_ago_preserves_shape_1d() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert apply_ago(x).shape == x.shape
    assert apply_iago(x).shape == x.shape


def test_ago_2d_axis_minus1() -> None:
    """For 2-D input, AGO accumulates along axis=-1 (rows independently)."""
    X = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    expected = np.array([[1.0, 3.0, 6.0], [10.0, 30.0, 60.0]])
    np.testing.assert_allclose(apply_ago(X, order=1, axis=-1), expected, atol=1e-12)


def test_iago_2d_axis_0() -> None:
    """For 2-D input with axis=0, IAGO works column-by-column."""
    X = np.array([[1.0, 10.0], [3.0, 30.0], [6.0, 60.0]])
    expected = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    np.testing.assert_allclose(apply_iago(X, order=1, axis=0), expected, atol=1e-12)


def test_invertibility_holds_2d() -> None:
    """AGO ∘ IAGO and IAGO ∘ AGO are identity for 2-D arrays too."""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((3, 7))
    np.testing.assert_allclose(
        apply_iago(apply_ago(X, order=2), order=2), X, atol=1e-10
    )


# ---------------------------------------------------------------------------
# Edge cases / dtype
# ---------------------------------------------------------------------------


def test_ago_integer_input_promoted_to_float() -> None:
    x_int = np.array([1, 2, 3, 4])
    out = apply_ago(x_int)
    assert out.dtype.kind == "f"
    np.testing.assert_allclose(out, np.array([1.0, 3.0, 6.0, 10.0]), atol=1e-12)


def test_ago_order_negative_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        apply_ago(np.array([1.0, 2.0]), order=-1)


def test_iago_order_negative_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        apply_iago(np.array([1.0, 2.0]), order=-1)


def test_ago_handles_negative_values() -> None:
    """AGO is defined for any real-valued input (Liu §4.6 edge-case table)."""
    x = np.array([-1.0, 2.0, -3.0, 4.0])
    expected = np.array([-1.0, 1.0, -2.0, 2.0])
    np.testing.assert_allclose(apply_ago(x), expected, atol=1e-12)


def test_ago_single_element() -> None:
    """1-element sequence: AGO and IAGO are both identity."""
    x = np.array([42.0])
    np.testing.assert_array_equal(apply_ago(x), x)
    np.testing.assert_array_equal(apply_iago(x), x)


def test_apply_ago_does_not_mutate_input() -> None:
    """Operators must not modify the input array in place."""
    x = np.array([1.0, 2.0, 3.0])
    x_copy = x.copy()
    apply_ago(x)
    apply_iago(x)
    np.testing.assert_array_equal(x, x_copy)

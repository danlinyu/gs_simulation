"""Tests for grey relational operators D₁ / D₂ / D₃ (Liu §5.2)."""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.operators.relational import (
    apply_d1_initialing,
    apply_d2_averaging,
    apply_d3_interval,
)


# ----------------------------------------------------------------------------
# Book worked examples (Liu 2024 §5.2)
# ----------------------------------------------------------------------------

# Liu §5.2 illustration sequence: X = (3.2, 3.7, 4.5, 4.9, 5.6).
BOOK_INPUT = np.array([3.2, 3.7, 4.5, 4.9, 5.6])


@pytest.mark.book_example
def test_d1_book_example() -> None:
    """Liu §5.2: D₁(X) = (1, 1.15625, 1.40625, 1.53125, 1.75)."""
    expected = np.array([1.0, 1.15625, 1.40625, 1.53125, 1.75])
    actual = apply_d1_initialing(BOOK_INPUT)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.book_example
def test_d2_book_example() -> None:
    """Liu §5.2: mean(X) = 4.38; D₂(X) ≈ (0.7306, 0.8447, 1.0274, 1.1187, 1.2785)."""
    mean_X = 21.9 / 5  # 4.38
    expected = BOOK_INPUT / mean_X
    actual = apply_d2_averaging(BOOK_INPUT)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.book_example
def test_d3_book_example() -> None:
    """Liu §5.2: min=3.2, max=5.6, span=2.4; D₃(X) ≈ (0, 0.208, 0.542, 0.708, 1)."""
    expected = np.array([0.0, 0.208333, 0.541667, 0.708333, 1.0])
    actual = apply_d3_interval(BOOK_INPUT)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


# ----------------------------------------------------------------------------
# Property-based correctness checks
# ----------------------------------------------------------------------------


def test_d1_first_value_is_unity() -> None:
    x = np.array([2.5, 3.1, 4.7])
    assert apply_d1_initialing(x)[0] == 1.0


def test_d2_image_mean_is_unity() -> None:
    """D₂'s image has mean 1 by construction."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = apply_d2_averaging(x)
    np.testing.assert_allclose(out.mean(), 1.0, atol=1e-12)


def test_d3_range_is_unit_interval() -> None:
    """D₃ image is bounded in [0, 1] with both endpoints attained."""
    x = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
    out = apply_d3_interval(x)
    assert out.min() == 0.0
    assert out.max() == 1.0
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_d1_scale_invariance() -> None:
    """D₁ is invariant under positive scalar multiplication of input."""
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        apply_d1_initialing(x), apply_d1_initialing(2.5 * x), atol=1e-12
    )


def test_d2_scale_invariance() -> None:
    """D₂ is invariant under positive scalar multiplication of input."""
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        apply_d2_averaging(x), apply_d2_averaging(7.0 * x), atol=1e-12
    )


def test_d3_translation_and_scale() -> None:
    """D₃ is invariant under positive affine transformations a*x + b."""
    x = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
    np.testing.assert_allclose(
        apply_d3_interval(x), apply_d3_interval(2.0 * x + 10.0), atol=1e-12
    )


# ----------------------------------------------------------------------------
# Edge cases / undefined inputs
# ----------------------------------------------------------------------------


def test_d1_raises_on_zero_first_value() -> None:
    with pytest.raises(ValueError, match="first value"):
        apply_d1_initialing(np.array([0.0, 1.0, 2.0]))


def test_d2_raises_on_zero_mean() -> None:
    with pytest.raises(ValueError, match="mean"):
        apply_d2_averaging(np.array([-1.0, 0.0, 1.0]))


def test_d3_raises_on_constant_sequence() -> None:
    with pytest.raises(ValueError, match="constant"):
        apply_d3_interval(np.array([5.0, 5.0, 5.0]))


# ----------------------------------------------------------------------------
# Shape and axis handling
# ----------------------------------------------------------------------------


def test_shape_preservation_1d() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    for op in (apply_d1_initialing, apply_d2_averaging, apply_d3_interval):
        assert op(x).shape == x.shape


def test_shape_preservation_2d() -> None:
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    for op in (apply_d1_initialing, apply_d2_averaging, apply_d3_interval):
        assert op(X).shape == X.shape


def test_d1_2d_axis_minus1() -> None:
    """For 2-D input with axis=-1, each row is normalized independently."""
    X = np.array([[1.0, 2.0, 4.0], [3.0, 6.0, 9.0]])
    expected = np.array([[1.0, 2.0, 4.0], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(apply_d1_initialing(X, axis=-1), expected, atol=1e-12)


def test_d1_2d_axis_0() -> None:
    """For 2-D input with axis=0, each column is normalized independently."""
    X = np.array([[1.0, 4.0], [2.0, 8.0], [3.0, 12.0]])
    expected = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    np.testing.assert_allclose(apply_d1_initialing(X, axis=0), expected, atol=1e-12)


def test_d3_2d_per_row() -> None:
    """D₃ along axis=-1 normalizes each row to [0, 1] independently."""
    X = np.array([[1.0, 2.0, 3.0], [10.0, 50.0, 90.0]])
    out = apply_d3_interval(X, axis=-1)
    np.testing.assert_allclose(out[0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[0, -1], 1.0, atol=1e-12)
    np.testing.assert_allclose(out[1, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[1, -1], 1.0, atol=1e-12)


def test_dtype_promoted_to_float() -> None:
    """Integer input is promoted to float to permit division."""
    x_int = np.array([2, 4, 6, 8])
    out = apply_d1_initialing(x_int)
    assert out.dtype.kind == "f"
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0, 4.0]), atol=1e-12)

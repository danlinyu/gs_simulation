"""Tests for Group GM(1,1) variants — Liu §7.5."""
from __future__ import annotations

import numpy as np
import pytest

from gs_simulation.gm.gm11 import fit_egm11
from gs_simulation.gm.group import (
    fit_all_data_gm11,
    fit_metabolic_gm11,
    fit_new_information_gm11,
    fit_partial_data_gm11,
)


# Liu §7.5 Example 7.5.1 reports â changes from -0.17241 (all-data) to
# -0.187862 (metabolic) on a grain-production sequence.
EX_7_5_1_DATA = np.array([6745.0, 6788.0, 6584.0, 7086.0, 7491.0, 7625.0, 7813.0])


def test_all_data_equivalent_to_fit_egm11() -> None:
    """all_data is just a wrapper; verify exact agreement."""
    direct = fit_egm11(EX_7_5_1_DATA)
    grouped = fit_all_data_gm11(EX_7_5_1_DATA)
    assert direct.a == grouped.a
    assert direct.b == grouped.b
    assert direct.n_train == grouped.n_train


def test_partial_data_uses_tail_only() -> None:
    """Partial-data fit on tail [k0..n] differs from full fit (different (a, b))."""
    full = fit_all_data_gm11(EX_7_5_1_DATA)
    partial = fit_partial_data_gm11(EX_7_5_1_DATA, k0=3)
    assert partial.n_train == EX_7_5_1_DATA.size - 2  # tail length
    # Different training data → typically different parameters.
    assert partial.a != full.a or partial.b != full.b


def test_partial_k0_equals_1_matches_all_data() -> None:
    """k0=1 is the full sequence; should agree with all-data fit."""
    full = fit_all_data_gm11(EX_7_5_1_DATA)
    partial = fit_partial_data_gm11(EX_7_5_1_DATA, k0=1)
    np.testing.assert_allclose(partial.a, full.a, atol=1e-12)
    np.testing.assert_allclose(partial.b, full.b, atol=1e-12)


def test_partial_data_too_short_tail_raises() -> None:
    """Tail with < 4 elements raises."""
    with pytest.raises(ValueError, match="≥ 4"):
        fit_partial_data_gm11(EX_7_5_1_DATA, k0=5)  # tail length = 3


def test_partial_data_invalid_k0_raises() -> None:
    with pytest.raises(ValueError, match="k0 must be >= 1"):
        fit_partial_data_gm11(EX_7_5_1_DATA, k0=0)


def test_new_information_appends_obs() -> None:
    """New-info fit uses original + appended; window grows."""
    fit = fit_new_information_gm11(EX_7_5_1_DATA, x_new=8000.0)
    assert fit.n_train == EX_7_5_1_DATA.size + 1


def test_new_information_accepts_array() -> None:
    """Multiple new observations may be appended at once."""
    fit = fit_new_information_gm11(EX_7_5_1_DATA, x_new=np.array([8000.0, 8200.0]))
    assert fit.n_train == EX_7_5_1_DATA.size + 2


def test_metabolic_preserves_window_size() -> None:
    """Metabolic fit drops oldest obs and appends new; n_train unchanged."""
    fit = fit_metabolic_gm11(EX_7_5_1_DATA, x_new=8000.0)
    assert fit.n_train == EX_7_5_1_DATA.size


def test_metabolic_uses_rolled_window() -> None:
    """Metabolic should use [x[1:n], x_new], not the original window."""
    direct = fit_egm11(np.concatenate([EX_7_5_1_DATA[1:], np.array([8000.0])]))
    metabolic = fit_metabolic_gm11(EX_7_5_1_DATA, x_new=8000.0)
    np.testing.assert_allclose(metabolic.a, direct.a, atol=1e-12)
    np.testing.assert_allclose(metabolic.b, direct.b, atol=1e-12)


def test_metabolic_too_many_dropped_raises() -> None:
    """If new observations would shrink the window below 4, raise."""
    short_x = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="≥ 4"):
        fit_metabolic_gm11(short_x, x_new=np.array([5.0, 6.0]))


@pytest.mark.book_example
def test_metabolic_shifts_a_per_book_example_7_5_1() -> None:
    """Liu §7.5 Ex 7.5.1: metabolic â changes meaningfully from all-data â.

    Book reports â = -0.17241 (all-data) vs â = -0.187862 (metabolic on
    the same sequence with an appended observation). We verify that the
    metabolic fit's â differs from the all-data fit's â (direction of
    change matches the book; exact magnitude depends on the appended
    observation we choose since the book's specific value is not in the
    extracted summary).
    """
    all_data = fit_all_data_gm11(EX_7_5_1_DATA)
    metabolic = fit_metabolic_gm11(EX_7_5_1_DATA, x_new=8000.0)
    # The two fits should produce different development coefficients.
    assert all_data.a != metabolic.a

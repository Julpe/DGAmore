# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

from unittest.mock import patch, MagicMock

import pytest

from dgamore.interaction import LocalInteraction, Interaction
from dgamore.local_four_point import LocalFourPoint
import numpy as np

from dgamore.n_point_base import SpinChannel


@pytest.mark.parametrize("n", [1, 2, 3])
def test_exponentiation_with_positive_power_1(n):
    """Exponentiation with a positive power contracts the object with itself (variant 1)."""
    mat = np.random.rand(2, 2, 2, 2, 21, 20) + 1j * np.random.rand(2, 2, 2, 2, 21, 20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=1)
    identity = LocalFourPoint.identity_like(obj)
    result = obj.pow(n, identity)
    expected = obj
    for _ in range(n - 1):
        expected = expected @ obj
    assert np.allclose(result.mat, expected.mat, rtol=1e-4)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_exponentiation_with_positive_power_2(n):
    """Exponentiation with a positive power contracts the object with itself (variant 2)."""
    mat = np.random.rand(2, 2, 2, 2, 21, 20, 20) + 1j * np.random.rand(2, 2, 2, 2, 21, 20, 20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=2)
    identity = LocalFourPoint.identity_like(obj)
    result = obj.pow(n, identity)
    expected = obj
    for _ in range(n - 1):
        expected = expected @ obj
    assert np.allclose(result.mat, expected.mat, rtol=1e-4)


def test_exponentiation_with_zero_power_returns_identity_1():
    """Exponentiation with power zero returns the identity (variant 1)."""
    mat = np.random.rand(2, 2, 2, 2, 21, 20) + 1j * np.random.rand(2, 2, 2, 2, 21, 20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=1)
    identity = LocalFourPoint.identity_like(obj)
    result = obj.pow(0, identity)
    assert np.allclose(result.mat, identity.mat, rtol=1e-4)


def test_exponentiation_with_zero_power_returns_identity_2():
    """Exponentiation with power zero returns the identity (variant 2)."""
    mat = np.random.rand(2, 2, 2, 2, 21, 20, 20) + 1j * np.random.rand(2, 2, 2, 2, 21, 20, 20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=2)
    identity = LocalFourPoint.identity_like(obj)
    result = obj.pow(0, identity)
    assert np.allclose(result.mat, identity.mat, rtol=1e-4)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_exponentiation_with_negative_power_1(n):
    """Exponentiation with a negative power inverts and powers the object (variant 1)."""
    mat = np.random.rand(2, 2, 2, 2, 21, 20) + 1j * np.random.rand(2, 2, 2, 2, 21, 20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=1)
    identity = LocalFourPoint.identity_like(obj)
    result = obj.pow(-n, identity)
    expected = obj.invert()
    for _ in range(n - 1):
        expected = expected @ obj.invert()
    assert np.allclose(result.mat, expected.mat, rtol=1e-4)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_exponentiation_with_negative_power_2(n):
    """Exponentiation with a negative power inverts and powers the object (variant 2)."""
    mat = np.random.rand(2, 2, 2, 2, 21, 20, 20) + 1j * np.random.rand(2, 2, 2, 2, 21, 20, 20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=2)
    identity = LocalFourPoint.identity_like(obj)
    result = obj.pow(-n, identity)
    expected = obj.invert()
    for _ in range(n - 1):
        expected = expected @ obj.invert()
    assert np.allclose(result.mat, expected.mat, rtol=1e-2)


def test_exponentiation_with_non_integer_power_raises_error():
    """Exponentiation with a non-integer power raises."""
    mat = np.random.rand(2, 2, 2, 2, 21, 20, 20) + 1j * np.random.rand(2, 2, 2, 2, 21, 20, 20)
    obj = LocalFourPoint(mat, channel=SpinChannel.NONE, num_vn_dimensions=2)
    identity = LocalFourPoint.identity_like(obj)
    with pytest.raises(ValueError):
        obj.pow(2.5, identity)


def test_symmetrizes_square_matrix_correctly():
    """symmetrize symmetrizes a square compound matrix correctly."""
    mat = np.array([[[[[[1, 2.5], [2.5, 4]]]]]])
    obj = LocalFourPoint(mat)
    result = obj.symmetrize_v_vp()
    expected = np.array([[[[[[1, 2.5], [2.5, 4]]]]]])
    assert np.allclose(result.mat, expected, rtol=1e-4)


def test_symmetrizes_random_matrix_correctly():
    """symmetrize symmetrizes a random matrix correctly."""
    mat = np.random.rand(2, 2, 2, 2, 5, 3, 3)
    obj = LocalFourPoint(mat)
    result = obj.symmetrize_v_vp()
    expected = 0.5 * (mat + mat.swapaxes(0, 3).swapaxes(1, 2).swapaxes(-1, -2))
    assert np.allclose(result.mat, expected, rtol=1e-4)


def test_take_first_wn_selects_first_entry_and_returns_independent_copy():
    """take_first_wn removes the bosonic axis (keeping its first entry) and returns an independent copy."""
    mat = np.random.rand(2, 2, 2, 2, 5, 6, 6) + 1j * np.random.rand(2, 2, 2, 2, 5, 6, 6)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    expected = obj.mat[..., 0, :, :].copy()
    result = obj.take_first_wn()
    assert result.num_wn_dimensions == 0
    assert np.allclose(result.mat, expected, rtol=1e-4)
    result.mat[0, 0, 0, 0, 0, 0] = 12345.0
    assert not np.allclose(obj.mat[0, 0, 0, 0, 0, 0, 0], 12345.0)


def test_handles_symmetric_matrix_without_modification():
    """symmetrize leaves an already-symmetric matrix unchanged."""
    mat = np.array([[[[[[1, 2], [2, 4]]]]]])
    obj = LocalFourPoint(mat)
    result = obj.symmetrize_v_vp()
    assert np.allclose(result.mat, mat, rtol=1e-4)


def test_raises_error_for_non_square_last_two_axes():
    """symmetrize raises for non-square last two axes."""
    mat = np.random.rand(2, 2, 2, 2, 5, 3, 4)
    obj = LocalFourPoint(mat)
    with pytest.raises(ValueError):
        obj.symmetrize_v_vp()


def test_raises_error_for_not_having_two_vn_dimensions():
    """symmetrize raises without two fermionic-frequency dimensions."""
    mat = np.random.rand(2, 2, 2, 2, 5, 3)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    with pytest.raises(ValueError):
        obj.symmetrize_v_vp()


def test_sums_over_orbitals_correctly_1():
    """sum_over_orbitals contracts the orbital indices correctly (variant 1)."""
    mat = np.random.rand(2, 2, 2, 2, 5, 3)
    obj = LocalFourPoint(mat)
    result = obj.sum_over_orbitals("abcd->ad")
    assert result.mat.shape == (2, 2, 5, 3)
    assert np.allclose(result.mat, np.sum(mat, axis=(1, 2)), rtol=1e-4)


def test_sums_over_orbitals_correctly_2():
    """sum_over_orbitals contracts the orbital indices correctly (variant 2)."""
    mat = np.random.rand(2, 2, 2, 2, 5, 3, 3)
    obj = LocalFourPoint(mat)
    result = obj.sum_over_orbitals("abcd->ad")
    assert result.mat.shape == (2, 2, 5, 3, 3)
    assert np.allclose(result.mat, np.sum(mat, axis=(1, 2)), rtol=1e-4)


def test_raises_error_for_invalid_orbital_contraction_format():
    """sum_over_orbitals raises for an invalid contraction format."""
    mat = np.random.rand(2, 2, 2, 2, 5, 3, 3)
    obj = LocalFourPoint(mat)
    with pytest.raises(ValueError):
        obj.sum_over_orbitals("abc->ad")


def test_handles_no_orbital_reduction():
    """sum_over_orbitals leaves the object unchanged when no orbitals are reduced."""
    mat = np.random.rand(2, 2, 2, 2, 5, 3, 3)
    obj = LocalFourPoint(mat)
    result = obj.sum_over_orbitals("abcd->abcd")
    assert np.allclose(result.mat, mat, rtol=1e-4)


def test_reduces_orbital_dimensions_correctly_1():
    """sum_over_orbitals reduces the orbital dimensions correctly (variant 1)."""
    mat = np.random.rand(3, 3, 3, 3, 5, 4, 4)
    obj = LocalFourPoint(mat)
    result = obj.sum_over_orbitals("abcd->a")
    assert result.mat.shape == (3, 5, 4, 4)
    assert np.allclose(result.mat, np.sum(mat, axis=(1, 2, 3)), rtol=1e-4)


def test_reduces_orbital_dimensions_correctly_2():
    """sum_over_orbitals reduces the orbital dimensions correctly (variant 2)."""
    mat = np.random.rand(3, 3, 3, 3, 5, 4, 4)
    obj = LocalFourPoint(mat)
    result = obj.sum_over_orbitals("abcd->ab")
    assert result.mat.shape == (3, 3, 5, 4, 4)
    assert np.allclose(result.mat, np.sum(mat, axis=(2, 3)), rtol=1e-4)


def test_reduces_orbital_dimensions_correctly_3():
    """sum_over_orbitals reduces the orbital dimensions correctly (variant 3)."""
    mat = np.random.rand(3, 3, 3, 3, 5, 4, 4)
    obj = LocalFourPoint(mat)
    result = obj.sum_over_orbitals("abcd->abc")
    assert result.mat.shape == (3, 3, 3, 5, 4, 4)
    assert np.allclose(result.mat, np.sum(mat, axis=(3,)), rtol=1e-4)


def test_sums_over_single_vn_dimension_correctly_1():
    """sum_over_vn sums over a single fermionic-frequency dimension (variant 1)."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    beta = 10.0
    result = obj.sum_over_vn(beta, axis=(-1,))
    expected_mat = 1 / beta * np.sum(mat, axis=-1)
    assert np.allclose(result.mat, expected_mat, rtol=1e-4)
    assert result.num_vn_dimensions == 0


@pytest.mark.parametrize("n", [1, 2])
def test_sums_over_single_vn_dimension_correctly_2(n):
    """sum_over_vn sums over a single fermionic-frequency dimension (variant 2)."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    beta = 10.0
    result = obj.sum_over_vn(beta, axis=(-n,))
    expected_mat = 1 / beta * np.sum(mat, axis=(-n,))
    assert np.allclose(result.mat, expected_mat, rtol=1e-4)
    assert result.num_vn_dimensions == 1


def test_sums_over_multiple_vn_dimensions_correctly():
    """sum_over_vn sums over multiple fermionic-frequency dimensions."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    beta = 10.0
    result = obj.sum_over_vn(beta, axis=(-2, -1))
    expected_mat = 1 / beta**2 * np.sum(mat, axis=(-2, -1))
    assert np.allclose(result.mat, expected_mat, rtol=1e-4)
    assert result.num_vn_dimensions == 0


def test_raises_error_when_summing_over_too_many_vn_dimensions():
    """sum_over_vn raises when summing over too many fermionic dimensions."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    beta = 10.0
    with pytest.raises(ValueError):
        obj.sum_over_vn(beta, axis=(-2, -1))


def test_sums_over_all_vn_with_double_vn_dimensions_correctly():
    """sum_over_all_vn sums over both fermionic dimensions."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    beta = 10.0
    result = obj.sum_over_all_vn(beta)
    expected_mat = 1 / beta**2 * np.sum(mat, axis=(-2, -1))
    assert np.allclose(result.mat, expected_mat, rtol=1e-4)
    assert result.num_vn_dimensions == 0


def test_sums_over_all_vn_with_single_vn_dimension_correctly():
    """sum_over_all_vn sums over the single fermionic dimension."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    beta = 10.0
    result = obj.sum_over_all_vn(beta)
    expected_mat = 1 / beta * np.sum(mat, axis=-1)
    assert np.allclose(result.mat, expected_mat, rtol=1e-4)
    assert result.num_vn_dimensions == 0


def test_handles_no_vn_dimensions_without_modification_for_sum():
    """sum_over_all_vn leaves an object with no fermionic dimensions unchanged."""
    mat = np.random.rand(2, 2, 2, 2, 5)
    obj = LocalFourPoint(mat, num_vn_dimensions=0)
    beta = 10.0
    result = obj.sum_over_all_vn(beta)
    assert np.allclose(result.mat, mat, rtol=1e-4)
    assert result.num_vn_dimensions == 0


def test_contracts_legs_correctly_with_two_vn_dimensions():
    """contract_legs contracts the legs correctly with two fermionic dimensions."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    beta = 10.0
    result = obj.contract_legs(beta)
    assert result.mat.shape == (2, 2, 5)
    assert np.allclose(result.mat, 1.0 / beta**2 * np.einsum("abcdefg->ade", mat), rtol=1e-4)


def test_raises_error_when_contracting_legs_with_invalid_vn_dimensions():
    """contract_legs raises for an invalid number of fermionic dimensions."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    beta = 10.0
    with pytest.raises(ValueError):
        obj.contract_legs(beta)


def test_calls_sum_over_all_vn_and_sum_over_orbitals():
    """contract_legs calls sum_over_all_vn and sum_over_orbitals."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    beta = 10.0

    with (
        patch.object(LocalFourPoint, "sum_over_all_vn", autospec=True) as mock_sum_vn,
        patch.object(LocalFourPoint, "sum_over_orbitals", autospec=True) as mock_sum_orb,
    ):
        mock_sum_vn.return_value = obj
        mock_sum_orb.return_value = obj

        obj.contract_legs(beta)

        mock_sum_vn.assert_called_once_with(obj, beta)
        mock_sum_orb.assert_called_once_with(obj, "abcd->ad")


def test_converts_to_compound_indices_with_no_vn_dimensions():
    """to_compound_indices converts an object with no fermionic dimensions."""
    mat = np.random.rand(2, 2, 2, 2, 5)
    obj = LocalFourPoint(mat, num_vn_dimensions=0)
    result = obj.to_compound_indices()
    assert result.mat.shape == (5, 4, 4)
    assert np.allclose(result.mat, mat.transpose(4, 0, 1, 3, 2).reshape(5, 4, 4), rtol=1e-4)


def test_converts_to_compound_indices_with_one_vn_dimension():
    """to_compound_indices converts an object with one fermionic dimension."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    result = obj.to_compound_indices()
    assert result.mat.shape == (5, 16, 16)


def test_calls_extend_vn_to_diagonal_with_one_vn_dimension_and_executes_original():
    """to_compound_indices extends a single fermionic dimension to the diagonal."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    with patch.object(
        LocalFourPoint, "extend_vn_to_diagonal", autospec=True, wraps=LocalFourPoint.extend_vn_to_diagonal
    ) as mock_extend:
        result = obj.to_compound_indices()
        mock_extend.assert_called_once_with(obj)
        assert result.mat.shape == (5, 16, 16)


def test_converts_to_compound_indices_with_two_vn_dimensions():
    """to_compound_indices converts an object with two fermionic dimensions."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    result = obj.to_compound_indices()
    assert np.allclose(result.mat, mat.transpose(4, 0, 1, 5, 3, 2, 6).reshape(5, 16, 16), rtol=1e-4)


def test_raises_error_for_missing_bosonic_frequencies():
    """to_compound_indices raises for missing bosonic frequencies."""
    mat = np.random.rand(2, 2, 2, 2, 4, 4)
    obj = LocalFourPoint(mat, num_wn_dimensions=0)
    with pytest.raises(ValueError):
        obj.to_compound_indices()


def test_handles_already_compound_indices_without_modification():
    """to_compound_indices leaves an already-compound object unchanged."""
    mat = np.random.rand(5, 4, 4)
    obj = LocalFourPoint(mat, num_wn_dimensions=1, num_vn_dimensions=2)
    result = obj.to_compound_indices()
    assert np.allclose(result.mat, mat, rtol=1e-4)


@pytest.mark.parametrize(
    "num_vn_dimensions,expected_shape,compound_shape",
    [(0, (2, 2, 2, 2, 5), (5, 4, 4)), (2, (2, 2, 2, 2, 5, 4, 4), (5, 16, 16))],
)
def test_converts_compound_indices_to_full_indices_correctly(num_vn_dimensions, expected_shape, compound_shape):
    """to_full_indices converts compound indices back to full indices."""
    mat = np.random.rand(*expected_shape)
    obj = LocalFourPoint(mat, num_vn_dimensions=num_vn_dimensions)
    obj = obj.to_compound_indices()
    assert obj.mat.shape == compound_shape
    result = obj.to_full_indices()
    assert result.mat.shape == expected_shape


def test_converts_compound_indices_to_full_indices_correctly_for_one_vn_dimension():
    """to_full_indices converts compound indices back for one fermionic dimension."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    obj = obj.to_compound_indices()
    assert obj.mat.shape == (5, 16, 16)
    result = obj.to_full_indices()
    assert result.mat.shape == (2, 2, 2, 2, 5, 4, 4)
    assert result.num_vn_dimensions == 2
    assert np.allclose(mat, result.take_vn_diagonal().mat, rtol=1e-4)


def test_raises_error_for_invalid_current_shape():
    """to_full_indices raises for an invalid current shape."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    with pytest.raises(ValueError, match="Converting to full indices with shape .* not supported."):
        obj.to_full_indices()


@pytest.mark.parametrize(
    "num_wn_dimensions,num_vn_dimensions,shape",
    [
        (0, 0, (2, 2, 2, 2)),
        (1, 0, (2, 2, 2, 2, 5)),
        (0, 1, (2, 2, 2, 2, 4)),
        (1, 1, (2, 2, 2, 2, 5, 4)),
        (0, 2, (2, 2, 2, 2, 4, 4)),
        (1, 2, (2, 2, 2, 2, 5, 4, 4)),
    ],
)
def test_returns_original_object_when_already_in_full_indices(num_wn_dimensions, num_vn_dimensions, shape):
    """to_full_indices returns the original object when already in full indices."""
    mat = np.random.rand(*shape)
    obj = LocalFourPoint(mat, num_wn_dimensions=num_wn_dimensions, num_vn_dimensions=num_vn_dimensions)
    result = obj.to_full_indices()
    assert result.mat.shape == shape
    assert np.allclose(result.mat, mat, rtol=1e-4)
    assert result.num_wn_dimensions == num_wn_dimensions
    assert result.num_vn_dimensions == num_vn_dimensions


def test_handles_diagonal_extraction_for_single_vn_dimension():
    """to_full_indices extracts the diagonal for a single fermionic dimension."""
    mat = np.random.rand(5, 16, 16)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    obj._original_shape = (2, 2, 2, 2, 5, 4)
    result = obj.to_full_indices()
    assert result.mat.shape == (2, 2, 2, 2, 5, 4)

    mat = mat.reshape((5,) + (2, 2, 4) * 2).transpose(1, 2, 5, 4, 0, 3, 6).diagonal(axis1=-2, axis2=-1)
    assert np.allclose(result.mat, mat, rtol=1e-4)


def test_raises_error_for_invalid_bosonic_frequency_dimensions():
    """to_full_indices raises for invalid bosonic-frequency dimensions."""
    mat = np.random.rand(1, 16, 16)
    obj = LocalFourPoint(mat, num_wn_dimensions=0, num_vn_dimensions=2)
    with pytest.raises(ValueError):
        obj.to_full_indices()


@pytest.mark.parametrize("full_niw_range", [True, False])
def test_assures_invert_calls_to_half_niw_range_to_compound_indices_and_to_full_indices(full_niw_range):
    """invert routes through to_half_niw_range, to_compound_indices and to_full_indices."""
    mat = np.random.rand(2, 2, 2, 2, 11, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2, full_niw_range=full_niw_range)
    with (
        patch.object(
            LocalFourPoint, "to_half_niw_range", autospec=True, wraps=LocalFourPoint.to_half_niw_range
        ) as mock_half_niw,
        patch.object(
            LocalFourPoint, "to_compound_indices", autospec=True, wraps=LocalFourPoint.to_compound_indices
        ) as mock_compound,
        patch.object(
            LocalFourPoint, "to_full_indices", autospec=True, wraps=LocalFourPoint.to_full_indices
        ) as mock_full,
    ):
        obj.invert()
        mock_half_niw.assert_called_once()
        mock_compound.assert_called_once()
        mock_full.assert_called_once()


def test_assures_invert_always_returns_half_niw_range():
    """invert always returns a half bosonic range."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    with patch.object(
        LocalFourPoint, "to_half_niw_range", autospec=True, wraps=LocalFourPoint.to_half_niw_range
    ) as mock_half_niw:
        result = obj.invert()
        mock_half_niw.assert_called()
        assert not result.full_niw_range


def test_assures_invert_calls_to_full_indices_with_default_shape():
    """invert calls to_full_indices with the default shape."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    with patch.object(
        LocalFourPoint, "to_full_indices", autospec=True, wraps=LocalFourPoint.to_full_indices
    ) as mock_full:
        obj.invert()
        args, kwargs = mock_full.call_args
        # shape=None is default
        assert kwargs.get("shape", None) is None


def test_multiplies_two_objects_with_no_vn_dimensions_correctly():
    """matmul multiplies two objects with no fermionic dimensions."""
    mat1 = np.random.rand(2, 2, 2, 2, 5)
    mat2 = np.random.rand(2, 2, 2, 2, 5)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=0)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=0)
    result1 = obj1 @ obj2
    result2 = obj2 @ obj1
    expected1 = np.einsum("abcdw,dcefw->abefw", mat1, mat2, optimize=True)
    expected2 = np.einsum("abcdw,dcefw->abefw", mat2, mat1, optimize=True)
    assert np.allclose(result1.mat, expected1[..., 2:], rtol=1e-4)
    assert np.allclose(result2.mat, expected2[..., 2:], rtol=1e-4)


def test_multiplies_two_objects_with_one_vn_dimension_correctly():
    """matmul multiplies two objects with one fermionic dimension."""
    mat1 = np.random.rand(2, 2, 2, 2, 5, 4)
    mat2 = np.random.rand(2, 2, 2, 2, 5, 4)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=1)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=1)
    result1 = obj1 @ obj2
    result2 = obj2 @ obj1
    expected1 = np.einsum("abcdwv,dcefwv->abefwv", mat1, mat2, optimize=True)
    expected2 = np.einsum("abcdwv,dcefwv->abefwv", mat2, mat1, optimize=True)
    assert np.allclose(result1.mat, expected1[..., 2:, :], rtol=1e-4)
    assert np.allclose(result2.mat, expected2[..., 2:, :], rtol=1e-4)


@pytest.mark.parametrize(
    "full_niw_range1,full_niw_range2", [(False, False), (True, True), (False, True), (True, False)]
)
def test_assures_matmul_calls_to_compound_indices_for_two_vn_dimensions(full_niw_range1, full_niw_range2):
    """matmul routes through to_compound_indices for two fermionic dimensions."""
    mat1 = np.random.rand(2, 2, 2, 2, 21 if full_niw_range1 else 11, 4, 4)
    mat2 = np.random.rand(2, 2, 2, 2, 21 if full_niw_range2 else 11, 4, 4)
    count_full_niw_range = [full_niw_range1, full_niw_range2].count(True)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=2, full_niw_range=full_niw_range1)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=2, full_niw_range=full_niw_range2)
    with (
        patch.object(
            LocalFourPoint, "to_compound_indices", autospec=True, wraps=LocalFourPoint.to_compound_indices
        ) as mock_compound,
        patch.object(
            LocalFourPoint, "to_half_niw_range", autospec=True, wraps=LocalFourPoint.to_half_niw_range
        ) as mock_half_niw,
        patch.object(
            LocalFourPoint, "to_full_niw_range", autospec=True, wraps=LocalFourPoint.to_full_niw_range
        ) as mock_full_niw,
        patch.object(
            LocalFourPoint, "to_full_indices", autospec=True, wraps=LocalFourPoint.to_full_indices
        ) as mock_to_full_indices,
        patch("numpy.matmul", autospec=True, wraps=np.matmul) as mock_matmul,
    ):
        obj1 @ obj2
        assert mock_half_niw.call_count == 2
        mock_matmul.assert_called_once()
        assert mock_compound.call_count == 2
        assert mock_full_niw.call_count == count_full_niw_range
        assert mock_to_full_indices.call_count == 3


def test_raises_error_for_invalid_multiplication_with_non_local_four_point():
    """matmul raises for an invalid multiplication with a non-local four point."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    with pytest.raises(ValueError, match="Multiplication .* not supported."):
        obj @ np.random.rand(4, 4)


def test_handles_multiplication_with_local_interaction_correctly():
    """matmul handles multiplication with a LocalInteraction."""
    mat1 = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    mat2 = np.random.rand(2, 2, 2, 2)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=2)
    obj2 = LocalInteraction(mat2)
    result1 = obj1 @ obj2
    result2 = obj2 @ obj1
    expected1 = np.einsum("abcdwvp,dcef->abefwvp", mat1, mat2, optimize=True)
    expected2 = np.einsum("abcd,dcefwvp->abefwvp", mat2, mat1, optimize=True)
    assert np.allclose(result1.mat, expected1, rtol=1e-4)
    assert np.allclose(result2.mat, expected2, rtol=1e-4)


def test_multiplies_objects_with_mixed_vn_dimensions_correctly():
    """matmul multiplies objects with mixed fermionic dimensions."""
    mat1 = np.random.rand(2, 2, 2, 2, 5, 4)
    mat2 = np.random.rand(2, 2, 2, 2, 5)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=1)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=0)
    result1 = obj1 @ obj2
    result2 = obj2 @ obj1
    expected1 = np.einsum("abcdwv,dcefw->abefwv", mat1, mat2, optimize=True)
    expected2 = np.einsum("abcdw,dcefwv->abefwv", mat2, mat1, optimize=True)
    assert np.allclose(result1.mat, expected1[..., 2:, :], rtol=1e-4)
    assert np.allclose(result2.mat, expected2[..., 2:, :], rtol=1e-4)
    assert result1.num_vn_dimensions == 1
    assert result2.num_vn_dimensions == 1


def test_multiplies_with_full_niw_range_and_restores_shape():
    """matmul with a full bosonic range restores the original shape."""
    mat1 = np.random.rand(2, 2, 2, 2, 21, 4, 4)
    mat2 = np.random.rand(2, 2, 2, 2, 21, 4, 4)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=2, full_niw_range=True)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=2, full_niw_range=True)
    result = obj1 @ obj2
    assert result.mat.shape == (2, 2, 2, 2, 11, 4, 4)
    assert not result.full_niw_range


def test_multiplies_with_scalar_correctly():
    """Multiplication with a scalar scales the matrix."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    scalar = 2.5
    result = obj * scalar
    expected = mat * scalar
    assert np.allclose(result.mat, expected, rtol=1e-4)


def test_multiplies_with_numpy_array_correctly():
    """Multiplication with a numpy array multiplies elementwise."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    array = np.random.rand(2, 2, 2, 2, 5, 4)
    result = obj * array
    expected = mat * array
    assert np.allclose(result.mat, expected, rtol=1e-4)


def test_raises_error_for_invalid_multiplication_type():
    """Multiplication raises for an unsupported operand type."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1)
    with pytest.raises(
        ValueError, match="Multiplication only supported with numbers, numpy arrays or LocalFourPoint objects."
    ):
        obj * "invalid_type"


def test_raises_error_for_invalid_vn_dimensions():
    """matmul raises for an invalid number of fermionic dimensions."""
    mat1 = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    mat2 = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    mat3 = np.random.rand(2, 2, 2, 2, 5, 4)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=2)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=2)
    obj3 = LocalFourPoint(mat3, num_vn_dimensions=1)
    with pytest.raises(ValueError, match="Both objects must have only one fermionic frequency dimension."):
        obj1 * obj2
    with pytest.raises(ValueError, match="Both objects must have only one fermionic frequency dimension."):
        obj2 * obj3
    with pytest.raises(ValueError, match="Both objects must have only one fermionic frequency dimension."):
        obj1 * obj3


def test_multiplies_two_objects_with_one_vn_dimension_and_generates_two_vn_dimensions():
    """matmul of two one-vn objects generates a two-vn result."""
    mat1 = np.random.rand(2, 2, 2, 2, 21, 4)
    mat2 = np.random.rand(2, 2, 2, 2, 21, 4)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=1, full_niw_range=True)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=1, full_niw_range=True)
    result = obj1 * obj2
    expected = np.einsum("abcdwv,dcefwp->abefwvp", mat1, mat2, optimize=True)
    assert np.allclose(result.mat, expected[..., 10:, :, :], rtol=1e-4)


def test_converts_to_half_bosonic_range_correctly_1():
    """to_half_niw_range keeps the positive bosonic half (variant 1)."""
    mat = np.random.rand(2, 2, 2, 2, 21) + 1j * np.random.rand(2, 2, 2, 2, 21)
    obj = LocalFourPoint(mat, num_vn_dimensions=0, full_niw_range=True)
    result = obj.to_half_niw_range()
    assert result is obj
    assert result.mat.shape == (2, 2, 2, 2, 11)
    assert np.allclose(result.mat, np.take(mat, np.arange(10, 21), axis=-1), rtol=1e-4)


def test_converts_to_half_bosonic_range_correctly_2():
    """to_half_niw_range keeps the positive bosonic half (variant 2)."""
    mat = np.random.rand(2, 2, 2, 2, 21, 20) + 1j * np.random.rand(2, 2, 2, 2, 21, 20)
    obj = LocalFourPoint(mat, num_vn_dimensions=1, full_niw_range=True)
    result = obj.to_half_niw_range()
    assert result is obj
    assert result.mat.shape == (2, 2, 2, 2, 11, 20)
    assert np.allclose(result.mat, np.take(mat, np.arange(10, 21), axis=-2), rtol=1e-4)


def test_converts_to_half_bosonic_range_correctly_3():
    """to_half_niw_range keeps the positive bosonic half (variant 3)."""
    mat = np.random.rand(2, 2, 2, 2, 21, 10, 10) + 1j * np.random.rand(2, 2, 2, 2, 21, 10, 10)
    obj = LocalFourPoint(mat, num_vn_dimensions=2, full_niw_range=True)
    result = obj.to_half_niw_range()
    assert result is obj
    assert result.mat.shape == (2, 2, 2, 2, 11, 10, 10)
    assert np.allclose(result.mat, np.take(mat, np.arange(10, 21), axis=-3), rtol=1e-4)


def test_to_full_niw_range_to_half_niw_range_should_reproduce_original_1():
    """full->half bosonic range round-trips to the original (variant 1)."""
    mat = np.random.rand(2, 2, 2, 2, 21) + 1j * np.random.rand(2, 2, 2, 2, 21)
    obj = LocalFourPoint(mat, num_vn_dimensions=0, full_niw_range=False)
    obj = obj.to_full_niw_range().to_half_niw_range()
    assert np.allclose(obj.mat, mat, rtol=1e-4)
    assert obj.full_niw_range is False
    assert obj.num_vn_dimensions == 0


def test_to_full_niw_range_to_half_niw_range_should_reproduce_original_2():
    """full->half bosonic range round-trips to the original (variant 2)."""
    mat = np.random.rand(2, 2, 2, 2, 21, 4) + 1j * np.random.rand(2, 2, 2, 2, 21, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1, full_niw_range=False)
    obj = obj.to_full_niw_range().to_half_niw_range()
    assert np.allclose(obj.mat, mat, rtol=1e-4)
    assert obj.full_niw_range is False
    assert obj.num_vn_dimensions == 1


def test_to_full_niw_range_to_half_niw_range_should_reproduce_original_3():
    """full->half bosonic range round-trips to the original (variant 3)."""
    mat = np.random.rand(2, 2, 2, 2, 21, 4, 4) + 1j * np.random.rand(2, 2, 2, 2, 21, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2, full_niw_range=False)
    obj = obj.to_full_niw_range().to_half_niw_range()
    assert np.allclose(obj.mat, mat, rtol=1e-4)
    assert obj.full_niw_range is False
    assert obj.num_vn_dimensions == 2


def test_adds_two_local_four_point_objects_correctly():
    """Adding two LocalFourPoint objects adds their matrices."""
    mat1 = np.random.rand(2, 2, 2, 2, 21, 4, 4) + 1j * np.random.rand(2, 2, 2, 2, 21, 4, 4)
    mat2 = np.random.rand(2, 2, 2, 2, 21, 4, 4) + 1j * np.random.rand(2, 2, 2, 2, 21, 4, 4)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=2)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=2)
    result = obj1 + obj2
    expected = mat1 + mat2
    assert result.full_niw_range == False
    assert np.allclose(result.mat, expected[..., 10:, :, :], rtol=1e-4)


def test_adds_two_local_four_point_objects_with_different_vn_dimensions():
    """Adding LocalFourPoint objects with different fermionic dimensions promotes correctly."""
    mat1 = np.random.rand(2, 2, 2, 2, 21, 4, 4)
    mat2 = np.random.rand(2, 2, 2, 2, 21, 4)
    mat2_diagonal = np.einsum("...i,ij->...ij", mat2, np.eye(mat2.shape[-1]))
    mat3 = np.random.rand(2, 2, 2, 2, 21)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=2)
    obj2 = LocalFourPoint(mat2, num_vn_dimensions=1)
    obj3 = LocalFourPoint(mat3, num_vn_dimensions=0)

    with patch.object(
        LocalFourPoint, "extend_vn_to_diagonal", autospec=True, wraps=LocalFourPoint.extend_vn_to_diagonal
    ) as mock_extend:
        result1 = obj1 + obj2
        assert mock_extend.call_count == 1
        result2 = obj1 + obj3
        result3 = obj2 + obj3
        result4 = obj2 + obj2
        result5 = obj3 + obj3
        result6 = obj1 + obj1
        assert mock_extend.call_count == 1
        assert result1.mat.shape == (2, 2, 2, 2, 11, 4, 4)
        assert result2.mat.shape == (2, 2, 2, 2, 11, 4, 4)
        assert result3.mat.shape == (2, 2, 2, 2, 11, 4)
        assert result4.mat.shape == (2, 2, 2, 2, 11, 4)
        assert result5.mat.shape == (2, 2, 2, 2, 11)
        assert result6.mat.shape == (2, 2, 2, 2, 11, 4, 4)

        assert result1.full_niw_range is False
        assert result2.full_niw_range is False
        assert result3.full_niw_range is False
        assert result4.full_niw_range is False
        assert result5.full_niw_range is False
        assert result6.full_niw_range is False

        assert np.allclose(result1.mat, (mat1 + mat2_diagonal)[..., 10:, :, :], rtol=1e-4)
        assert np.allclose(result2.mat, (mat1 + mat3[..., None, None])[..., 10:, :, :], rtol=1e-4)
        assert np.allclose(result3.mat, (mat2 + mat3[..., None])[..., 10:, :], rtol=1e-4)
        assert np.allclose(result4.mat, (mat2 + mat2)[..., 10:, :], rtol=1e-4)
        assert np.allclose(result5.mat, (mat3 + mat3)[..., 10:], rtol=1e-4)
        assert np.allclose(result6.mat, (mat1 + mat1)[..., 10:, :, :], rtol=1e-4)


def test_adds_local_four_point_and_scalar_correctly():
    """Adding a scalar to a LocalFourPoint adds elementwise."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    scalar = 2.5
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    result = obj + scalar
    expected = mat + scalar
    assert np.allclose(result.mat, expected, rtol=1e-4)


def test_adds_local_four_point_and_numpy_array_correctly():
    """Adding a numpy array to a LocalFourPoint adds elementwise."""
    mat1 = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    mat2 = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat1, num_vn_dimensions=2)
    result = obj + mat2
    expected = mat1 + mat2
    assert np.allclose(result.mat, expected, rtol=1e-4)


def test_adds_local_four_point_and_local_interaction_correctly():
    """Adding a LocalInteraction to a LocalFourPoint broadcasts over frequencies."""
    mat1 = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    mat2 = np.random.rand(2, 2, 2, 2)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=2)
    obj2 = LocalInteraction(mat2)
    result = obj1 + obj2
    expected = mat1 + mat2[..., None, None, None]
    assert np.allclose(result.mat, expected, rtol=1e-4)


def test_raises_error_for_unsupported_addition_type():
    """Addition raises for an unsupported operand type."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    with pytest.raises(ValueError, match="Operations '\\+/-' for .* not supported."):
        obj + "invalid_type"


def test_adds_local_four_point_and_interaction_with_compressed_q_dimension():
    """Adding a compressed Interaction to a LocalFourPoint returns an Interaction."""
    mat1 = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    mat2 = np.random.rand(5, 2, 2, 2, 2)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=2)
    interaction = Interaction(mat2, has_compressed_q_dimension=True)
    result = obj1 + interaction
    assert isinstance(result, np.ndarray)
    expected = mat1[None, ...] + mat2[..., None, None, None]
    assert np.allclose(result, expected, rtol=1e-4)
    assert result.shape[1:] == mat1.shape
    assert result.shape[0] == 5


def test_adds_local_four_point_and_interaction_with_decompressed_q_dimension():
    """Adding a decompressed Interaction to a LocalFourPoint returns an Interaction."""
    mat1 = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    mat2 = np.random.rand(4, 4, 1, 2, 2, 2, 2)
    obj1 = LocalFourPoint(mat1, num_vn_dimensions=2)
    interaction = Interaction(mat2, has_compressed_q_dimension=False, nq=(4, 4, 1))
    result = obj1 + interaction
    assert isinstance(result, np.ndarray)
    expected = mat1[None, None, None, ...] + mat2[..., None, None, None]
    assert np.allclose(result, expected, rtol=1e-4)
    assert result.shape[3:] == mat1.shape
    assert result.shape[:2] == mat2.shape[:2]


def test_permutes_orbitals_correctly():
    """permute_orbitals applies the orbital permutation correctly."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    result = obj.permute_orbitals("abcd->cdab")
    expected = np.einsum("abcd...->cdab...", mat, optimize=True)
    assert np.allclose(result.mat, expected)


def test_raises_error_for_invalid_permutation_format():
    """permute_orbitals raises for an invalid permutation format."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    with pytest.raises(ValueError, match="Invalid permutation."):
        obj.permute_orbitals("abc->abcd")


def test_raises_error_for_mismatched_orbital_dimensions():
    """permute_orbitals raises for mismatched orbital dimensions."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    with pytest.raises(ValueError, match="Invalid permutation."):
        obj.permute_orbitals("abcd->abc")


def test_converts_to_full_niw_range_correctly_with_no_vn_dimensions():
    """to_full_niw_range converts an object with no fermionic dimensions."""
    mat = np.random.rand(2, 2, 2, 2, 11) + 1j * np.random.rand(2, 2, 2, 2, 11)
    obj = LocalFourPoint(mat, num_vn_dimensions=0, full_niw_range=False)
    result = obj.to_full_niw_range()
    expected = np.conj(np.flip(np.take(mat, np.arange(1, mat.shape[-1]), axis=-1), axis=-1))
    expected = np.concatenate((expected, mat), axis=-1)
    assert np.allclose(result.mat, expected, rtol=1e-4)
    assert result.full_niw_range is True


def test_converts_to_full_niw_range_correctly_with_one_vn_dimension():
    """to_full_niw_range converts an object with one fermionic dimension."""
    mat = np.random.rand(2, 2, 2, 2, 11, 4) + 1j * np.random.rand(2, 2, 2, 2, 11, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=1, full_niw_range=False)
    result = obj.to_full_niw_range()
    expected = np.conj(np.flip(np.take(mat, np.arange(1, mat.shape[-2]), axis=-2), axis=(-2, -1)))
    expected = np.concatenate((expected, mat), axis=-2)
    assert np.allclose(result.mat, expected, rtol=1e-4)
    assert result.full_niw_range is True


def test_converts_to_full_niw_range_correctly_with_two_vn_dimensions():
    """to_full_niw_range converts an object with two fermionic dimensions."""
    mat = np.random.rand(2, 2, 2, 2, 11, 4, 4) + 1j * np.random.rand(2, 2, 2, 2, 11, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2, full_niw_range=False)
    result = obj.to_full_niw_range()
    expected = np.conj(np.flip(np.take(mat, np.arange(1, mat.shape[-3]), axis=-3), axis=(-3, -2, -1)))
    expected = np.concatenate((expected, mat), axis=-3)
    assert np.allclose(result.mat, expected, rtol=1e-4)
    assert result.full_niw_range is True


def test_handles_already_full_niw_range_without_modification():
    """to_full_niw_range leaves an already-full object unchanged."""
    mat = np.random.rand(2, 2, 2, 2, 21, 4, 4) + 1j * np.random.rand(2, 2, 2, 2, 21, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2, full_niw_range=True)
    result = obj.to_full_niw_range()
    assert np.allclose(result.mat, mat, rtol=1e-4)
    assert result.full_niw_range is True


def test_identity_returns_correct_shape_for_vn_2():
    """identity returns the correct shape for two fermionic dimensions."""
    obj = LocalFourPoint.identity(2, 3, 4, num_vn_dimensions=2)
    assert obj.mat.shape == (2, 2, 2, 2, 7, 8, 8)


def test_identity_returns_correct_shape_for_vn_1():
    """identity returns the correct shape for one fermionic dimension."""
    obj = LocalFourPoint.identity(2, 3, 4, num_vn_dimensions=1)
    assert obj.mat.shape == (2, 2, 2, 2, 7, 8)


def test_identity_raises_for_invalid_vn_dimensions():
    """identity raises for an invalid number of fermionic dimensions."""
    with pytest.raises(ValueError):
        LocalFourPoint.identity(2, 3, 4, num_vn_dimensions=0)


def test_identity_matrix_is_eye_in_compound_indices():
    """The identity is the unit matrix in compound indices."""
    obj = LocalFourPoint.identity(2, 1, 1, num_vn_dimensions=2)
    obj_comp = obj.to_compound_indices()
    for mat in obj_comp.mat:
        assert np.allclose(mat, np.eye(mat.shape[0]))


def test_identity_like_returns_same_shape_as_other():
    """identity_like returns the same shape as its operand."""
    other = LocalFourPoint.identity(2, 2, 2, num_vn_dimensions=2)
    ident = LocalFourPoint.identity_like(other)
    assert ident.mat.shape == other.mat.shape


def test_identity_like_returns_eye_in_compound_indices():
    """identity_like is the unit matrix in compound indices."""
    other = LocalFourPoint.identity(2, 1, 1, num_vn_dimensions=2)
    ident = LocalFourPoint.identity_like(other)
    ident_comp = ident.to_compound_indices()
    for mat in ident_comp.mat:
        assert np.allclose(mat, np.eye(mat.shape[0]))


def test_identity_like_works_for_vn_1():
    """identity_like works for one fermionic dimension."""
    other = LocalFourPoint.identity(2, 1, 1, num_vn_dimensions=1)
    ident = LocalFourPoint.identity_like(other)
    assert ident.mat.shape == other.mat.shape


def test_add_dunder_calls_add():
    """__add__ delegates to add."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj1 = LocalFourPoint(mat, num_vn_dimensions=2)
    obj2 = LocalFourPoint(mat, num_vn_dimensions=2)
    with patch.object(LocalFourPoint, "add", wraps=obj1.add) as mock_add:
        _ = obj1 + obj2
        mock_add.assert_called_once_with(obj2)


def test_sub_method_and_dunder():
    """sub and __sub__ subtract correctly."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj1 = LocalFourPoint(mat, num_vn_dimensions=2)
    obj2 = LocalFourPoint(mat, num_vn_dimensions=2)
    with (
        patch.object(LocalFourPoint, "sub", wraps=obj1.sub) as mock_sub,
        patch.object(LocalFourPoint, "add", wraps=obj1.add) as mock_add,
    ):
        _ = obj1.sub(obj2)
        mock_sub.assert_called_once_with(obj2)
        mock_add.assert_called_once()
    with (
        patch.object(LocalFourPoint, "sub", wraps=obj1.sub) as mock_sub,
        patch.object(LocalFourPoint, "add", wraps=obj1.add) as mock_add,
    ):
        _ = obj1 - obj2
        mock_sub.assert_called_once_with(obj2)
        mock_add.assert_called_once()


def test_sub_operator():
    """The - operator subtracts two objects."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    fp = LocalFourPoint(mat, num_vn_dimensions=2)
    res = fp - 1.0
    assert np.allclose(res.mat, fp.mat - 1.0)

    res2 = 1.0 - fp
    assert np.allclose(res2.mat, 1.0 - fp.mat)


def test_mul_dunder_calls_mul():
    """__mul__ delegates to mul."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    scalar = 2.0
    with patch.object(LocalFourPoint, "mul", wraps=obj.mul) as mock_mul:
        _ = obj * scalar
        mock_mul.assert_called_once_with(scalar)


def test_matmul_dunder_calls_matmul():
    """__matmul__ delegates to matmul."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj1 = LocalFourPoint(mat, num_vn_dimensions=2)
    obj2 = LocalFourPoint(mat, num_vn_dimensions=2)
    with patch.object(LocalFourPoint, "matmul", wraps=obj1.matmul) as mock_matmul:
        _ = obj1 @ obj2
        mock_matmul.assert_called_once_with(obj2, left_hand_side=True)
    with patch.object(LocalFourPoint, "matmul", wraps=obj1.matmul) as mock_matmul:
        _ = obj1.__rmatmul__(obj2)
        mock_matmul.assert_called_once_with(obj2, left_hand_side=False)


def test_pow_dunder_calls_pow():
    """__pow__ delegates to pow."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    exponent = 2
    with patch.object(LocalFourPoint, "pow", wraps=obj.pow) as mock_pow:
        _ = obj**exponent
        mock_pow.assert_called_once()


def test_neg_dunder_calls_neg():
    """__neg__ negates the matrix."""
    mat = np.random.rand(2, 2, 2, 2, 5, 4, 4)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    with patch.object(LocalFourPoint, "__neg__", wraps=obj.__neg__) as mock_neg:
        result = -obj
        mock_neg.assert_called()
        assert np.allclose(result.mat, -obj.mat, rtol=1e-4)


@pytest.mark.parametrize("num_vn_dimensions", [0, 1, 2])
def test_creates_bosonic_dimension_when_not_present(num_vn_dimensions):
    """A bosonic dimension is created when not present."""
    shape = (2,) * 4 + (4,) * num_vn_dimensions
    mat = np.random.rand(*shape)
    obj = LocalFourPoint(mat, num_vn_dimensions=num_vn_dimensions, num_wn_dimensions=0)
    result = obj.create_wn_dimension()
    assert result.num_wn_dimensions == 1
    assert result.mat.shape == (2,) * 4 + (1,) + (4,) * num_vn_dimensions
    assert np.allclose(result.mat, np.expand_dims(mat, axis=-(num_vn_dimensions + 1)), rtol=1e-4)


@pytest.mark.parametrize("num_vn_dimensions", [0, 1, 2])
def test_raises_error_when_bosonic_dimension_already_exists(num_vn_dimensions):
    """Creating a bosonic dimension raises when one already exists."""
    shape = (2,) * 4 + (4,) * num_vn_dimensions
    mat = np.random.rand(*shape)
    obj = LocalFourPoint(mat, num_vn_dimensions=num_vn_dimensions, num_wn_dimensions=1)
    with pytest.raises(ValueError, match="Object already has bosonic frequency dimensions."):
        obj.create_wn_dimension()


@pytest.mark.parametrize("num_vn_dimensions", [0, 1, 2])
def test_removes_bosonic_dimension_correctly(num_vn_dimensions):
    """The bosonic dimension is removed correctly."""
    shape = (2,) * 4 + (1,) + (4,) * num_vn_dimensions
    mat = np.random.rand(*shape)
    obj = LocalFourPoint(mat, num_vn_dimensions=num_vn_dimensions, num_wn_dimensions=1)
    result = obj.take_first_wn()
    assert result.num_wn_dimensions == 0
    assert result.mat.shape == (2,) * 4 + (4,) * num_vn_dimensions
    assert np.allclose(result.mat, np.take(mat, 0, axis=-(num_vn_dimensions + 1)), rtol=1e-4)


@pytest.mark.parametrize("num_vn_dimensions", [0, 1, 2])
def test_raises_error_when_no_bosonic_dimension(num_vn_dimensions):
    """Removing the bosonic dimension raises when none exists."""
    shape = (2,) * 4 + (1,) + (4,) * num_vn_dimensions
    mat = np.random.rand(*shape)
    obj = LocalFourPoint(mat, num_vn_dimensions=num_vn_dimensions, num_wn_dimensions=0)
    with pytest.raises(ValueError, match="Object must have exactly one bosonic frequency dimension."):
        obj.take_first_wn()


@pytest.mark.parametrize("niv_pad", [5, 10, 15])
def test_pads_with_u_correctly(niv_pad):
    """pad_with_u pads the fermionic box with the interaction tail."""
    mat = np.random.rand(2, 2, 2, 2, 11, 8, 8)
    u_mat = np.random.rand(2, 2, 2, 2)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    u = LocalInteraction(u_mat)
    result = obj.pad_with_u(u, niv_pad)
    assert result.mat.shape == (2, 2, 2, 2, 11, 2 * niv_pad, 2 * niv_pad)
    assert result.original_shape == result.mat.shape
    assert np.allclose(result.mat[..., niv_pad - 4 : niv_pad + 4, niv_pad - 4 : niv_pad + 4], mat, rtol=1e-4)

    assert np.allclose(result.mat[..., : niv_pad - 4, :], u_mat[..., None, None, None], rtol=1e-4)
    assert np.allclose(result.mat[..., : niv_pad - 4], u_mat[..., None, None, None], rtol=1e-4)
    assert np.allclose(result.mat[..., niv_pad + 4 :, :], u_mat[..., None, None, None], rtol=1e-4)
    assert np.allclose(result.mat[..., niv_pad + 4 :], u_mat[..., None, None, None], rtol=1e-4)

    assert np.allclose(result.mat[..., niv_pad + 4 :, : niv_pad - 4], u_mat[..., None, None, None], rtol=1e-4)
    assert np.allclose(result.mat[..., : niv_pad - 4, niv_pad + 4 :], u_mat[..., None, None, None], rtol=1e-4)
    assert np.allclose(result.mat[..., : niv_pad - 4, : niv_pad - 4], u_mat[..., None, None, None], rtol=1e-4)
    assert np.allclose(result.mat[..., niv_pad + 4 :, niv_pad + 4 :], u_mat[..., None, None, None], rtol=1e-4)


@pytest.mark.parametrize("niv", [5, 10, 15])
def test_does_not_pad_when_niv_pad_is_less_or_equal(niv):
    """pad_with_u is a no-op when the target niv does not exceed the current one."""
    mat = np.random.rand(2, 2, 2, 2, 5, 2 * niv, 2 * niv)
    u_mat = np.random.rand(2, 2, 2, 2)
    obj = LocalFourPoint(mat, num_vn_dimensions=2)
    u = LocalInteraction(u_mat)
    result = obj.pad_with_u(u, niv)
    assert np.allclose(result.mat, mat, rtol=1e-4)
    assert result.mat.shape == mat.shape


def test_pad_with_u_noop_returns_independent_copy():
    """pad_with_u no-op returns an independent copy."""
    mat = np.random.rand(2, 2, 2, 2, 5, 8, 8)
    obj = LocalFourPoint(mat.copy(), num_vn_dimensions=2)
    u = LocalInteraction(np.random.rand(2, 2, 2, 2))
    result = obj.pad_with_u(u, 4)  # niv_pad == niv -> no-op branch
    assert result is not obj
    result.mat[0, 0, 0, 0, 0, 0, 0] = -999.0
    assert obj.mat[0, 0, 0, 0, 0, 0, 0] != -999.0  # no-op must still hand back an independent copy


def test_pad_with_u_does_not_mutate_self():
    """pad_with_u does not mutate the original object."""
    mat = np.random.rand(2, 2, 2, 2, 5, 8, 8)
    obj = LocalFourPoint(mat.copy(), num_vn_dimensions=2)
    u = LocalInteraction(np.random.rand(2, 2, 2, 2))
    obj.pad_with_u(u, 8)
    assert obj.mat.shape == (2, 2, 2, 2, 5, 8, 8)
    assert np.allclose(obj.mat, mat, rtol=1e-4)  # source array left untouched


def test_symmetrize_orbitals_already_symmetrized():
    """symmetrize_orbitals returns self when already symmetrized."""
    obj = LocalFourPoint(np.zeros((2, 2, 2, 2)))

    obj.is_orbitally_symmetrized = MagicMock(return_value=True)
    obj._symmetrize_orbitals = MagicMock()

    orbitals = [1, 2, 3]

    result = obj.symmetrize_orbitals(orbitals)

    assert result is obj
    obj.is_orbitally_symmetrized.assert_called_once_with(orbitals)
    obj._symmetrize_orbitals.assert_not_called()


def test_symmetrize_orbitals_calls_private_method():
    """symmetrize_orbitals delegates to the private method when not yet symmetrized."""
    obj = LocalFourPoint(np.zeros((2, 2, 2, 2)))

    obj.is_orbitally_symmetrized = MagicMock(return_value=False)
    obj._symmetrize_orbitals = MagicMock(return_value="symmetrized_obj")

    orbitals = [1, 3]

    result = obj.symmetrize_orbitals(orbitals)

    obj.is_orbitally_symmetrized.assert_called_once_with(orbitals)
    obj._symmetrize_orbitals.assert_called_once_with(orbitals, (0, 1, 2, 3))
    assert result == "symmetrized_obj"


def test_is_orbitally_symmetrized_delegates():
    """is_orbitally_symmetrized delegates to the private check."""
    obj = LocalFourPoint(np.zeros((2, 2, 2, 2)))

    obj._is_orbitally_symmetrized = MagicMock(return_value=True)

    orbitals = np.array([1, 2])

    result = obj.is_orbitally_symmetrized(orbitals)

    obj._is_orbitally_symmetrized.assert_called_once_with(orbitals, (0, 1, 2, 3))
    assert result is True


def test_symmetrize_orbitals_empty_list():
    """symmetrize_orbitals on an empty list returns self."""
    obj = LocalFourPoint(np.zeros((2, 2, 2, 2)))

    obj.is_orbitally_symmetrized = MagicMock(return_value=True)
    obj._symmetrize_orbitals = MagicMock()

    orbitals = []

    result = obj.symmetrize_orbitals(orbitals)

    assert result is obj
    obj._symmetrize_orbitals.assert_not_called()


def test_from_constant_passes_complex64_dtype_to_np_full(monkeypatch):
    """from_constant builds the array directly in complex64."""
    seen = {"dtype": None}
    real_full = np.full

    def spy_full(shape, value, *args, **kwargs):
        seen["dtype"] = kwargs.get("dtype", args[0] if args else None)
        return real_full(shape, value, *args, **kwargs)

    monkeypatch.setattr(np, "full", spy_full)
    fp = LocalFourPoint.from_constant(1, 1, 2, value=1.0)
    assert fp.mat.dtype == np.complex64
    assert seen["dtype"] == np.complex64


def test_local_four_point_identity_is_valid_and_complex64():
    """identity is built directly in the complex64 storage dtype."""
    ident = LocalFourPoint.identity(1, 1, 2, num_vn_dimensions=2)
    assert ident.mat.dtype == np.complex64
    # in compound-index space each bosonic-frequency slice must be the unit matrix
    compound = ident.to_compound_indices().mat
    n = compound.shape[-1]
    for w_slice in compound:
        assert np.allclose(w_slice, np.eye(n), atol=1e-5)


def test_identity_like_matches_its_own_operand_shape():
    """identity_like uses its own operand's shape."""
    magn = LocalFourPoint.from_constant(1, 1, 3, num_vn_dimensions=2, channel=SpinChannel.MAGN, value=1.0)
    ident = LocalFourPoint.identity_like(magn)
    result = ident + magn  # must not raise on shape mismatch
    assert result.niv == magn.niv

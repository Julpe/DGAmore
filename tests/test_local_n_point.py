# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import itertools
from unittest.mock import MagicMock

import numpy as np
import pytest

from dgamore.local_n_point import LocalNPoint


def test_initializes_with_valid_parameters():
    """LocalNPoint initializes with valid dimension parameters and full ranges."""
    mat = np.zeros((4, 4))
    obj = LocalNPoint(mat, 2, 1, 1)
    assert obj.num_orbital_dimensions == 2
    assert obj.num_wn_dimensions == 1
    assert obj.num_vn_dimensions == 1
    assert obj.full_niw_range is True
    assert obj.full_niv_range is True


def test_raises_error_for_invalid_orbital_dimensions():
    """LocalNPoint rejects an invalid number of orbital dimensions."""
    mat = np.zeros((4, 4))
    with pytest.raises(AssertionError):
        LocalNPoint(mat, 3, 1, 1)


def test_raises_error_for_invalid_fermionic_dimensions():
    """LocalNPoint rejects an invalid number of fermionic dimensions."""
    mat = np.zeros((4, 4))
    with pytest.raises(AssertionError):
        LocalNPoint(mat, 2, 1, 3)


def test_raises_error_for_invalid_bosonic_dimensions():
    """LocalNPoint rejects an invalid number of bosonic dimensions."""
    mat = np.zeros((4, 4))
    with pytest.raises(AssertionError):
        LocalNPoint(mat, 2, 2, 1)


def test_initializes_with_partial_frequency_ranges():
    """LocalNPoint initializes with half bosonic and fermionic ranges."""
    mat = np.zeros((4, 4))
    obj = LocalNPoint(mat, 4, 0, 2, full_niw_range=False, full_niv_range=False)
    assert obj.full_niw_range is False
    assert obj.full_niv_range is False


def test_returns_correct_number_of_bands_for_higher_dimensional_matrix():
    """n_bands is read from the orbital axes of a higher-dimensional matrix."""
    mat = np.zeros((4, 4, 9, 10, 10))
    obj = LocalNPoint(mat, 2, 1, 2)
    assert obj.n_bands == 4


def test_returns_zero_bosonic_frequencies_when_no_wn_dimensions():
    """niw is zero when there are no bosonic-frequency dimensions."""
    mat = np.zeros((4, 4))
    obj = LocalNPoint(mat, 2, 0, 1)
    assert obj.niw == 0


def test_calculates_correct_bosonic_frequencies_with_full_range():
    """niw is computed correctly for a full bosonic range."""
    mat = np.zeros((4, 5, 10))
    obj = LocalNPoint(mat, 2, 1, 1, full_niw_range=True)
    assert obj.niw == 2


def test_calculates_correct_bosonic_frequencies_with_half_range():
    """niw is computed correctly for a half bosonic range."""
    mat = np.zeros((4, 4, 10))
    obj = LocalNPoint(mat, 2, 1, 1, full_niw_range=False)
    assert obj.niw == 2


def test_returns_zero_fermionic_frequencies_when_no_vn_dimensions():
    """niv is zero when there are no fermionic-frequency dimensions."""
    mat = np.zeros((4, 4))
    obj = LocalNPoint(mat, 2, 1, 0)
    assert obj.niv == 0


def test_calculates_correct_fermionic_frequencies_with_full_range():
    """niv is computed correctly for a full fermionic range."""
    mat = np.zeros((4, 4, 10))
    obj = LocalNPoint(mat, 2, 1, 1, full_niv_range=True)
    assert obj.niv == 5


def test_calculates_correct_fermionic_frequencies_with_half_range():
    """niv is computed correctly for a half fermionic range."""
    mat = np.zeros((4, 4, 10))
    obj = LocalNPoint(mat, 2, 1, 1, full_niv_range=False)
    assert obj.niv == 5


def test_raises_error_when_cutting_bosonic_frequencies_with_no_wn_dimensions():
    """cut_niw raises when there are no bosonic-frequency dimensions."""
    mat = np.zeros((4, 4))
    obj = LocalNPoint(mat, 2, 0, 1)
    with pytest.raises(ValueError):
        obj.cut_niw(1)


def test_does_not_raise_error_when_cutting_more_bosonic_frequencies_than_available():
    """cut_niw is a no-op when asked to cut more bosonic frequencies than available."""
    mat = np.zeros((4, 4, 10, 10))
    obj = LocalNPoint(mat, 2, 1, 1)
    res = obj.cut_niw(6)
    assert res is obj


def test_cuts_bosonic_frequencies_correctly_with_full_range():
    """cut_niw trims the bosonic axis for a full range."""
    mat = np.zeros((4, 4, 10))
    obj = LocalNPoint(mat, 2, 1, 1, full_niw_range=True)
    result = obj.cut_niw(2)
    assert result.mat.shape[-3] == 4


def test_cuts_bosonic_frequencies_correctly_with_half_range():
    """cut_niw trims the bosonic axis for a half range."""
    mat = np.zeros((4, 4, 10))
    obj = LocalNPoint(mat, 2, 1, 1, full_niw_range=False)
    result = obj.cut_niw(2)
    assert result.mat.shape[-3] == 4


def test_preserves_matrix_shape_when_cutting_with_no_vn_dimensions():
    """cut_niw preserves the shape when there are no fermionic dimensions."""
    mat = np.zeros((4, 4, 10))
    obj = LocalNPoint(mat, 2, 1, 0)
    result = obj.cut_niw(2)
    assert result.mat.shape == (4, 4, 5)


def test_raises_error_when_cutting_fermionic_frequencies_with_no_vn_dimensions():
    """cut_niv raises when there are no fermionic-frequency dimensions."""
    mat = np.zeros((4, 4))
    obj = LocalNPoint(mat, 2, 1, 0)
    with pytest.raises(ValueError):
        obj.cut_niv(1)


def test_does_not_raise_error_when_cutting_more_fermionic_frequencies_than_available():
    """cut_niv is a no-op when asked to cut more fermionic frequencies than available."""
    mat = np.zeros((4, 4, 10, 10))
    obj = LocalNPoint(mat, 2, 1, 1)
    res = obj.cut_niv(6)
    assert res is obj


def test_cuts_fermionic_frequencies_correctly_with_full_range():
    """cut_niv trims both fermionic axes for a full range."""
    mat = np.zeros((4, 4, 10, 10))
    obj = LocalNPoint(mat, 2, 1, 2, full_niv_range=True)
    result = obj.cut_niv(3)
    assert result.mat.shape[-1] == 6
    assert result.mat.shape[-2] == 6


def test_cuts_fermionic_frequencies_correctly_with_half_range():
    """cut_niv trims the fermionic axis for a half range."""
    mat = np.zeros((4, 4, 10))
    obj = LocalNPoint(mat, 2, 1, 1, full_niv_range=False)
    result = obj.cut_niv(3)
    assert result.mat.shape[-1] == 3


def test_preserves_matrix_shape_when_cutting_with_no_wn_dimensions():
    """cut_niv preserves the shape when there are no bosonic dimensions."""
    mat = np.zeros((4, 4, 10))
    obj = LocalNPoint(mat, 2, 0, 1)
    result = obj.cut_niv(2)
    assert result.mat.shape == (4, 4, 4)


def test_does_not_raise_error_when_cutting_both_frequencies_with_invalid_bosonic_cut():
    """cut_niw_and_niv ignores an over-large bosonic cut but applies the fermionic one."""
    mat = np.zeros((4, 4, 10, 10))
    obj = LocalNPoint(mat, 2, 1, 1)
    res = obj.cut_niw_and_niv(6, 3)
    assert res.niv == 3
    assert res.niw == 5


def test_does_not_raise_error_when_cutting_both_frequencies_with_invalid_fermionic_cut():
    """cut_niw_and_niv ignores an over-large fermionic cut but applies the bosonic one."""
    mat = np.zeros((4, 4, 10, 10))
    obj = LocalNPoint(mat, 2, 1, 1)
    res = obj.cut_niw_and_niv(3, 6)
    assert res.niw == 3
    assert res.niv == 5


def test_cuts_both_frequencies_correctly_with_full_ranges():
    """cut_niw_and_niv trims both bosonic and fermionic axes for full ranges."""
    mat = np.zeros((4, 4, 10, 10))
    obj = LocalNPoint(mat, 2, 1, 2, full_niw_range=True, full_niv_range=True)
    result = obj.cut_niw_and_niv(2, 3)
    assert result.mat.shape[-3] == 4
    assert result.mat.shape[-1] == 6
    assert result.mat.shape[-2] == 6


def test_cuts_both_frequencies_correctly_with_half_ranges():
    """cut_niw_and_niv trims both axes for half ranges."""
    mat = np.zeros((1, 1, 1, 1, 5, 10, 10))
    obj = LocalNPoint(mat, 2, 1, 2, full_niw_range=False, full_niv_range=False)
    result = obj.cut_niw_and_niv(2, 3)
    assert result.mat.shape[-3] == 3
    assert result.mat.shape[-1] == 3
    assert result.mat.shape[-2] == 3


def test_raises_error_when_extending_with_no_fermionic_dimensions():
    """extend_vn_to_diagonal raises with no fermionic dimensions."""
    mat = np.zeros((4, 4))
    obj = LocalNPoint(mat, 2, 1, 0)
    with pytest.raises(ValueError):
        obj.extend_vn_to_diagonal()


def test_returns_self_when_extending_with_two_fermionic_dimensions():
    """extend_vn_to_diagonal returns self when already at two fermionic dimensions."""
    mat = np.zeros((4, 4, 4, 4, 4))
    obj = LocalNPoint(mat, 2, 1, 2)
    result = obj.extend_vn_to_diagonal()
    assert result is obj
    assert result.mat.shape == (4, 4, 4, 4, 4)


def test_extends_correctly_with_one_fermionic_dimension():
    """extend_vn_to_diagonal places the single fermionic axis on the diagonal of a new one."""
    mat = np.zeros((4, 4, 4, 4))
    obj = LocalNPoint(mat, 2, 1, 1)
    result = obj.extend_vn_to_diagonal()
    assert result is obj
    assert result.mat.shape == (4, 4, 4, 4, 4)
    assert np.allclose(result.mat[..., 0, 0], mat[..., 0], atol=1e-2)
    assert np.allclose(result.mat[..., 0, 1], 0, atol=1e-2)
    assert np.allclose(result.mat[..., 1, 0], 0, atol=1e-2)
    assert np.allclose(result.mat[..., 1, 1], mat[..., 1], atol=1e-2)
    assert np.allclose(result.mat[..., 2, 0], 0, atol=1e-2)
    assert np.allclose(result.mat[..., 0, 2], 0, atol=1e-2)
    assert np.allclose(result.mat[..., 2, 1], 0, atol=1e-2)
    assert np.allclose(result.mat[..., 1, 2], 0, atol=1e-2)
    assert np.allclose(result.mat[..., 2, 2], mat[..., 2], atol=1e-2)


def test_raises_error_when_taking_diagonal_with_no_fermionic_dimensions():
    """take_vn_diagonal raises with no fermionic dimensions."""
    mat = np.zeros((4, 4))
    obj = LocalNPoint(mat, 2, 1, 0)
    with pytest.raises(ValueError):
        obj.take_vn_diagonal()


def test_returns_self_when_taking_diagonal_with_one_fermionic_dimension():
    """take_vn_diagonal returns self when already at one fermionic dimension."""
    mat = np.zeros((4, 4, 4, 4))
    obj = LocalNPoint(mat, 2, 1, 1)
    result = obj.take_vn_diagonal()
    assert result is obj
    assert result.mat.shape == (4, 4, 4, 4)


def test_compresses_correctly_with_two_fermionic_dimensions():
    """take_vn_diagonal collapses two fermionic axes to their diagonal."""
    mat = np.zeros((4, 4, 4, 4, 4))
    for i in range(4):
        mat[..., i, i] = i + 1
    obj = LocalNPoint(mat, 2, 1, 2)
    result = obj.take_vn_diagonal()
    assert result is obj
    assert result.mat.shape == (4, 4, 4, 4)
    assert np.allclose(result.mat[..., 0], 1, atol=1e-2)
    assert np.allclose(result.mat[..., 1], 2, atol=1e-2)
    assert np.allclose(result.mat[..., 2], 3, atol=1e-2)
    assert np.allclose(result.mat[..., 3], 4, atol=1e-2)


def test_flips_matrix_along_valid_single_axis():
    """flip_frequency_axis flips a single valid axis."""
    mat = np.zeros((4, 4, 9, 10))
    obj = LocalNPoint(mat, 2, 1, 1)
    result = obj.flip_frequency_axis(axis=(-1,))
    assert np.allclose(result.mat, np.flip(mat, axis=-1), atol=1e-2)


def test_flips_matrix_along_valid_multiple_axes():
    """flip_frequency_axis flips multiple valid axes."""
    mat = np.zeros((4, 4, 9, 10))
    obj = LocalNPoint(mat, 2, 1, 1)
    result = obj.flip_frequency_axis(axis=(-2, -1))
    assert np.allclose(result.mat, np.flip(mat, axis=(-2, -1)), atol=1e-2)


def test_raises_error_when_flipping_with_no_frequency_dimensions():
    """flip_frequency_axis raises with no frequency dimensions."""
    mat = np.zeros((4, 4))
    obj = LocalNPoint(mat, 2, 0, 0)
    with pytest.raises(ValueError):
        obj.flip_frequency_axis(axis=(-1,))
        obj.flip_frequency_axis(axis=-1)


def test_raises_error_for_invalid_axis_outside_possible_range():
    """flip_frequency_axis raises for an axis outside the frequency range."""
    mat = np.zeros((4, 4, 9, 10))
    obj = LocalNPoint(mat, 2, 1, 1)
    with pytest.raises(ValueError):
        obj.flip_frequency_axis(axis=(-3,))
        obj.flip_frequency_axis(axis=-3)
        obj.flip_frequency_axis(axis=(-3, -2))


def test_handles_single_axis_as_integer():
    """flip_frequency_axis accepts a single axis given as an integer."""
    mat = np.zeros((4, 4, 9, 10))
    obj = LocalNPoint(mat, 2, 1, 1)
    result = obj.flip_frequency_axis(axis=-1)
    assert np.allclose(result.mat, np.flip(mat, axis=-1), atol=1e-2)


def test_aligns_frequency_dimensions_correctly_when_self_has_one_and_other_has_two_fermionic_dimensions():
    """_align_frequency_dimensions_for_operation extends self when it has fewer fermionic dimensions."""
    mat_self = np.zeros((4, 4, 4, 4))
    mat_other = np.zeros((4, 4, 4, 4, 4))
    obj_self = LocalNPoint(mat_self, 2, 1, 1)
    obj_other = LocalNPoint(mat_other, 2, 1, 2)
    result_other, self_extended, other_extended = obj_self._align_frequency_dimensions_for_operation(obj_other)
    assert self_extended is True
    assert other_extended is False
    assert obj_self.mat.shape == (4, 4, 4, 4, 4)
    assert obj_other.mat.shape == (4, 4, 4, 4, 4)
    assert result_other is obj_other


def test_aligns_frequency_dimensions_correctly_when_self_has_two_and_other_has_one_fermionic_dimensions():
    """_align_frequency_dimensions_for_operation extends the other when it has fewer fermionic dimensions."""
    mat_self = np.zeros((4, 4, 4, 4, 4))
    mat_other = np.zeros((4, 4, 4, 4))
    obj_self = LocalNPoint(mat_self, 2, 1, 2)
    obj_other = LocalNPoint(mat_other, 2, 1, 1)
    result_other, self_extended, other_extended = obj_self._align_frequency_dimensions_for_operation(obj_other)
    assert self_extended is False
    assert other_extended is True
    assert obj_self.mat.shape == (4, 4, 4, 4, 4)
    assert obj_other.mat.shape == (4, 4, 4, 4, 4)
    assert result_other.mat.shape == (4, 4, 4, 4, 4)


def test_does_not_extend_frequency_dimensions_when_both_have_two_fermionic_dimensions():
    """_align_frequency_dimensions_for_operation extends nothing when both have two fermionic dimensions."""
    mat_self = np.zeros((4, 4, 4, 4, 4))
    mat_other = np.zeros((4, 4, 4, 4, 4))
    obj_self = LocalNPoint(mat_self, 2, 1, 2)
    obj_other = LocalNPoint(mat_other, 2, 1, 2)
    result_other, self_extended, other_extended = obj_self._align_frequency_dimensions_for_operation(obj_other)
    assert self_extended is False
    assert other_extended is False
    assert obj_self.mat.shape == (4, 4, 4, 4, 4)
    assert result_other.mat.shape == (4, 4, 4, 4, 4)


def test_does_not_extend_frequency_dimensions_when_both_have_one_fermionic_dimension():
    """_align_frequency_dimensions_for_operation extends nothing when both have one fermionic dimension."""
    mat_self = np.zeros((4, 4, 4, 4))
    mat_other = np.zeros((4, 4, 4, 4))
    obj_self = LocalNPoint(mat_self, 2, 1, 1)
    obj_other = LocalNPoint(mat_other, 2, 1, 1)
    result_other, self_extended, other_extended = obj_self._align_frequency_dimensions_for_operation(obj_other)
    assert self_extended is False
    assert other_extended is False
    assert obj_self.mat.shape == (4, 4, 4, 4)
    assert result_other.mat.shape == (4, 4, 4, 4)


def test_returns_self_when_already_in_full_bosonic_range():
    """to_full_niw_range returns self when already in the full bosonic range."""
    mat = np.zeros((4, 4, 10))
    obj = LocalNPoint(mat, 2, 1, 1, full_niw_range=True)
    result = obj.to_full_niw_range()
    assert result is obj
    assert result.mat.shape == mat.shape


def test_converts_to_half_bosonic_range_correctly():
    """to_half_niw_range keeps the positive bosonic half."""
    mat = np.random.rand(4, 4, 21, 20) + 1j * np.random.rand(4, 4, 21, 20)
    obj = LocalNPoint(mat, 2, 1, 1, full_niw_range=True)
    result = obj.to_half_niw_range()
    assert result is obj
    assert result.mat.shape == (4, 4, 11, 20)
    assert np.allclose(result.mat, np.take(mat, np.arange(10, 21), axis=-2), atol=1e-2)


def test_returns_self_when_already_in_half_bosonic_range():
    """to_half_niw_range returns self when already in the half bosonic range."""
    mat = np.random.rand(4, 4, 11, 10)
    obj = LocalNPoint(mat, 2, 1, 1, full_niw_range=False)
    result = obj.to_half_niw_range()
    assert result is obj
    assert result.mat.shape == mat.shape


def test_swaps_two_fermionic_frequency_axes_correctly():
    """swap_fermionic_frequency_axes exchanges the two fermionic axes."""
    mat = np.zeros((2, 2, 2, 2, 5, 4, 4))
    random_1 = np.random.rand()
    random_2 = np.random.rand()
    mat[..., 0, 1] = random_1
    mat[..., 1, 0] = random_2
    obj = LocalNPoint(mat, 4, 1, 2)
    result = obj.swap_fermionic_frequency_axes()
    assert np.allclose(result.mat[..., 0, 1], random_2)
    assert np.allclose(result.mat[..., 1, 0], random_1)


@pytest.mark.parametrize("num_vn_dimensions", [0, 1])
def test_raises_error_when_swapping_with_less_than_two_fermionic_dimensions(num_vn_dimensions):
    """swap_fermionic_frequency_axes raises with fewer than two fermionic dimensions."""
    shape = (4, 4, 4, 4, 1) + (4,) * num_vn_dimensions
    mat = np.zeros(shape)
    obj = LocalNPoint(mat, 4, 1, num_vn_dimensions)
    with pytest.raises(ValueError):
        obj.swap_fermionic_frequency_axes()


def test_saves_matrix_calls_to_full_niw_range_when_full_range(monkeypatch):
    """save converts to full then half niw range before writing for a full-range object."""
    mat = np.zeros((4, 4, 10))
    obj = LocalNPoint(mat, 2, 1, 1, full_niw_range=True)
    mock_full, mock_half, mock_save = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(obj, "to_full_niw_range", mock_full)
    monkeypatch.setattr(obj, "to_half_niw_range", mock_half)
    monkeypatch.setattr(np, "save", mock_save)
    obj.save(output_dir="dir", name="full_range")
    mock_full.assert_called_once()
    mock_half.assert_called_once()
    mock_save.assert_called_once()


def test_saves_matrix_calls_to_half_niw_range_when_half_range(monkeypatch):
    """save only converts to half niw range before writing for a half-range object."""
    mat = np.zeros((4, 4, 10))
    obj = LocalNPoint(mat, 2, 1, 1, full_niw_range=False)
    mock_full, mock_half, mock_save = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(obj, "to_full_niw_range", mock_full)
    monkeypatch.setattr(obj, "to_half_niw_range", mock_half)
    monkeypatch.setattr(np, "save", mock_save)
    obj.save(output_dir="dir", name="half_range")
    mock_full.assert_not_called()
    mock_half.assert_called_once()
    mock_save.assert_called_once()


def test_symmetrizes_orbitals_correctly():
    """_symmetrize_orbitals averages the requested orbitals and leaves the rest untouched."""
    mat = np.random.rand(4, 4, 4, 4)
    obj = LocalNPoint(mat, 4, 0, 0)
    orbitals = [1, 2]
    orbital_axes = (0, 1, 2, 3)
    symmetrized_obj = obj._symmetrize_orbitals(orbitals, orbital_axes)

    assert np.allclose(0.5 * (obj[0, 0, 0, 0] + obj[1, 1, 1, 1]), symmetrized_obj[0, 0, 0, 0])
    assert np.allclose(symmetrized_obj[0, 0, 0, 0], symmetrized_obj[1, 1, 1, 1])
    assert np.allclose(symmetrized_obj[1, 1, 0, 0], symmetrized_obj[0, 0, 1, 1])
    assert np.allclose(symmetrized_obj[1, 0, 0, 0], symmetrized_obj[0, 1, 1, 1])

    assert not np.allclose(symmetrized_obj[2, 2, 2, 2], symmetrized_obj[3, 3, 3, 3])
    assert not np.allclose(symmetrized_obj[2, 2, 0, 0], symmetrized_obj[3, 3, 1, 1])
    assert not np.allclose(symmetrized_obj[2, 0, 0, 0], symmetrized_obj[3, 1, 1, 1])


def test_raises_error_for_invalid_orbitals():
    """_symmetrize_orbitals raises for out-of-range orbital indices."""
    mat = np.random.rand(4, 4, 4, 4)
    obj = LocalNPoint(mat, 4, 0, 0)
    orbital_axes = (0, 1, 2, 3)

    with pytest.raises(ValueError):
        obj._symmetrize_orbitals([0, 5], orbital_axes)


def test_returns_self_for_single_orbital():
    """_symmetrize_orbitals returns self for a single orbital."""
    mat = np.random.rand(4, 4, 4, 4)
    obj = LocalNPoint(mat, 4, 0, 0)
    orbital_axes = (0, 1, 2, 3)
    result = obj._symmetrize_orbitals([1], orbital_axes)
    assert result is obj


def test_checks_if_orbitals_are_symmetrized():
    """_is_orbitally_symmetrized reports True after symmetrization."""
    mat = np.random.rand(4, 4, 4, 4)
    obj = LocalNPoint(mat, 4, 0, 0)
    orbitals = [1, 3]
    orbital_axes = (0, 1, 2, 3)
    obj._symmetrize_orbitals(orbitals, orbital_axes)
    assert obj._is_orbitally_symmetrized(orbitals, orbital_axes) is True


def test_detects_unsymmetrized_orbitals():
    """_is_orbitally_symmetrized reports False for unsymmetrized orbitals."""
    mat = np.random.rand(4, 4, 4, 4)
    obj = LocalNPoint(mat, 4, 0, 0)
    orbitals = [1, 3]
    orbital_axes = (0, 1, 2, 3)
    assert obj._is_orbitally_symmetrized(orbitals, orbital_axes) is False


def test_symmetrize_single_orbital_is_noop_and_returns_self():
    """_symmetrize_orbitals on a single orbital is a no-op returning self."""
    mat = np.random.rand(4, 4, 4, 4)
    original = mat.copy()
    obj = LocalNPoint(mat, 4, 0, 0)
    result = obj._symmetrize_orbitals([1], (0, 1, 2, 3))
    assert result is obj
    assert np.allclose(result.mat, original)


@pytest.mark.parametrize("orbitals", [[1], [1, 2], [1, 3], [1, 2, 3], [1, 2, 3, 4]])
def test_symmetrize_multiple_orbital_sets(orbitals):
    """_symmetrize_orbitals collapses the diagonal, pair and 3-1 patterns of a four-axis tensor to single values."""
    nb = 4
    mat = np.random.rand(nb, nb, nb, nb)

    obj = LocalNPoint(mat.copy(), 4, 0, 0)
    orbital_axes = (0, 1, 2, 3)

    sym_obj = obj._symmetrize_orbitals(orbitals, orbital_axes)
    sym_mat = sym_obj.mat

    orbitals_idx = np.array(orbitals) - 1

    if len(orbitals_idx) <= 1:
        return

    ref = sym_mat[orbitals_idx[0], orbitals_idx[0], orbitals_idx[0], orbitals_idx[0]]

    for o in orbitals_idx[1:]:
        assert np.allclose(sym_mat[o, o, o, o], ref)

    vals = []
    for i in orbitals_idx:
        for j in orbitals_idx:
            if i != j:
                vals.append(sym_mat[i, i, j, j])

    ref = vals[0]
    for v in vals:
        assert np.allclose(v, ref)

    vals = []
    for i in orbitals_idx:
        for j in orbitals_idx:
            if i != j:
                vals.append(sym_mat[i, j, j, i])

    if vals:
        ref = vals[0]
        for v in vals:
            assert np.allclose(v, ref)

    vals = []
    for i in orbitals_idx:
        for j in orbitals_idx:
            if i != j:
                vals.append(sym_mat[i, j, i, j])

    if vals:
        ref = vals[0]
        for v in vals:
            assert np.allclose(v, ref)

    vals = []

    for i in orbitals_idx:
        for j in orbitals_idx:
            if i != j:
                base = [i, j, j, j]
                for perm in set(itertools.permutations(base)):
                    vals.append(sym_mat[perm])

    if vals:
        ref = vals[0]
        for v in vals:
            assert np.allclose(v, ref)


@pytest.mark.parametrize(
    "orbital_groups", [[[1, 2], [3, 4]], [[1, 2, 3], [4]], [[1, 3], [2, 4]], [[1, 2, 3, 4]], [[1], [2], [3], [4]]]
)
def test_symmetrize_multiple_groups(orbital_groups):
    """_symmetrize_orbitals collapses each group's patterns onto single values without forcing distinct groups equal."""
    nb = 4
    mat = np.random.rand(nb, nb, nb, nb)

    obj = LocalNPoint(mat.copy(), 4, 0, 0)
    orbital_axes = (0, 1, 2, 3)

    sym_obj = obj._symmetrize_orbitals(orbital_groups, orbital_axes)
    sym_mat = sym_obj.mat

    for group in orbital_groups:
        group_idx = np.array(group) - 1

        if len(group_idx) <= 1:
            continue

        ref = sym_mat[group_idx[0], group_idx[0], group_idx[0], group_idx[0]]

        for o in group_idx[1:]:
            assert np.allclose(sym_mat[o, o, o, o], ref)

            vals = []
        for i in group_idx:
            for j in group_idx:
                if i != j:
                    vals.append(sym_mat[i, i, j, j])

        ref = vals[0]
        for v in vals:
            assert np.allclose(v, ref)

            vals = []
        for i in group_idx:
            for j in group_idx:
                if i != j:
                    vals.append(sym_mat[i, j, j, i])

        if vals:
            ref = vals[0]
            for v in vals:
                assert np.allclose(v, ref)

            vals = []
        for i in group_idx:
            for j in group_idx:
                if i != j:
                    vals.append(sym_mat[i, j, i, j])

        if vals:
            ref = vals[0]
            for v in vals:
                assert np.allclose(v, ref)

        vals = []
        for i in group_idx:
            for j in group_idx:
                if i != j:
                    base = [i, j, j, j]
                    for perm in set(itertools.permutations(base)):
                        vals.append(sym_mat[perm])

        if vals:
            ref = vals[0]
            for v in vals:
                assert np.allclose(v, ref)

    nontrivial_groups = [g for g in orbital_groups if len(g) > 1]

    if len(nontrivial_groups) >= 2:
        g1 = nontrivial_groups[0][0] - 1
        g2 = nontrivial_groups[1][0] - 1

        assert not np.array_equal(
            sym_mat[g1, g1, g1, g1],
            sym_mat[g2, g2, g2, g2],
        )


def test_orbital_symmetrization_patterns():
    """_symmetrize_orbitals equalizes the diagonal and the [i,i,j,j] and [i,j,j,i] pair patterns over the set."""
    mat = np.random.rand(3, 3, 3, 3)
    obj = LocalNPoint(mat.copy(), 4, 0, 0)

    orbitals = [[1, 2, 3]]
    obj._symmetrize_orbitals(orbitals, orbital_axes=(0, 1, 2, 3))

    assert obj.mat[0, 0, 0, 0] == obj.mat[1, 1, 1, 1] == obj.mat[2, 2, 2, 2]

    orbitals = [0, 1, 2]
    vals_iijj = [obj.mat[i, i, j, j] for i in orbitals for j in orbitals if i != j]
    ref_iijj = vals_iijj[0]
    for v in vals_iijj:
        assert np.allclose(v, ref_iijj)

    vals_ijji = [obj.mat[i, j, j, i] for i in orbitals for j in orbitals if i != j]
    ref_ijji = vals_ijji[0]
    for v in vals_ijji:
        assert np.allclose(v, ref_ijji)


def test_symmetrize_raises_for_orbitals_out_of_range_negative_and_large():
    """_symmetrize_orbitals raises for negative or too-large orbital indices."""
    mat = np.random.rand(4, 4, 4, 4)
    obj = LocalNPoint(mat, 4, 0, 0)
    with pytest.raises(ValueError):
        obj._symmetrize_orbitals([0, 2], (0, 1, 2, 3))
    with pytest.raises(ValueError):
        obj._symmetrize_orbitals([1, 10], (0, 1, 2, 3))


@pytest.mark.parametrize("orbitals", [[1], [1, 2], [1, 3], [1, 2, 3], [1, 2, 3, 4]])
def test_symmetrize_two_orbital_axes_single_set(orbitals):
    """_symmetrize_orbitals collapses the diagonal and off-diagonal elements of a two-axis tensor to single values."""
    nb = 4
    mat = np.random.rand(nb, nb)

    obj = LocalNPoint(mat.copy(), 2, 0, 0)
    orbital_axes = (0, 1)

    sym_obj = obj._symmetrize_orbitals(orbitals, orbital_axes)
    sym_mat = sym_obj.mat

    orbitals_idx = np.array(orbitals) - 1

    if len(orbitals_idx) <= 1:
        return

    ref = sym_mat[orbitals_idx[0], orbitals_idx[0]]

    for o in orbitals_idx[1:]:
        assert np.allclose(sym_mat[o, o], ref)

    vals = []
    for i in orbitals_idx:
        for j in orbitals_idx:
            if i != j:
                vals.append(sym_mat[i, j])

    if vals:
        ref = vals[0]
        for v in vals:
            assert np.allclose(v, ref)


@pytest.mark.parametrize(
    "orbital_groups", [[[1, 2], [3, 4]], [[1, 2, 3], [4]], [[1, 3], [2, 4]], [[1, 2, 3, 4]], [[1], [2], [3], [4]]]
)
def test_symmetrize_two_orbital_axes_multiple_groups(orbital_groups):
    """_symmetrize_orbitals holds each group's two-axis degeneracies without forcing distinct groups equal."""
    nb = 4
    mat = np.random.rand(nb, nb)

    obj = LocalNPoint(mat.copy(), 2, 0, 0)
    orbital_axes = (0, 1)

    sym_obj = obj._symmetrize_orbitals(orbital_groups, orbital_axes)
    sym_mat = sym_obj.mat

    for group in orbital_groups:
        group_idx = np.array(group) - 1

        if len(group_idx) <= 1:
            continue

        ref = sym_mat[group_idx[0], group_idx[0]]

        for o in group_idx[1:]:
            assert np.allclose(sym_mat[o, o], ref)

        vals = []
        for i in group_idx:
            for j in group_idx:
                if i != j:
                    vals.append(sym_mat[i, j])

        if vals:
            ref = vals[0]
            for v in vals:
                assert np.allclose(v, ref)

    nontrivial_groups = [g for g in orbital_groups if len(g) > 1]

    if len(nontrivial_groups) >= 2:
        g1 = nontrivial_groups[0][0] - 1
        g2 = nontrivial_groups[1][0] - 1

        assert not np.array_equal(
            sym_mat[g1, g1],
            sym_mat[g2, g2],
        )


def test_extend_vn_to_diagonal_stays_native_dtype_without_c128_temporary(monkeypatch):
    """extend_vn_to_diagonal stays in the native dtype with no complex128 temporary."""
    seen = {"c128": False}
    real_einsum = np.einsum

    def spy_einsum(subscripts, *operands, **kwargs):
        out = real_einsum(subscripts, *operands, **kwargs)
        if hasattr(out, "dtype") and out.dtype == np.complex128:
            seen["c128"] = True
        return out

    monkeypatch.setattr(np, "einsum", spy_einsum)
    obj = LocalNPoint(np.ones((1, 1, 3, 4), dtype=np.complex64), 2, 1, 1)  # [o1, o2, w, v]
    obj.extend_vn_to_diagonal()
    assert obj.num_vn_dimensions == 2
    assert seen["c128"] is False


def test_cut_niv_copy_false_mutates_self_in_place():
    """cut_niv(copy=False) mutates and returns self."""
    obj = LocalNPoint(np.arange(2 * 2 * 4).reshape(2, 2, 4).astype(complex), 2, 0, 1)  # [o1,o2,v], niv=2
    returned = obj.cut_niv(1, copy=False)
    assert returned is obj
    assert obj.niv == 1


def test_cut_niv_copy_true_leaves_original_untouched():
    """cut_niv(copy=True) returns a new object and leaves the original untouched."""
    obj = LocalNPoint(np.arange(2 * 2 * 4).reshape(2, 2, 4).astype(complex), 2, 0, 1)
    cut = obj.cut_niv(1, copy=True)
    assert cut is not obj
    assert obj.niv == 2
    assert cut.niv == 1


def test_cut_niw_copy_false_mutates_self_in_place():
    """cut_niw(copy=False) mutates and returns self."""
    obj = LocalNPoint(np.arange(2 * 2 * 5).reshape(2, 2, 5).astype(complex), 2, 1, 0)  # [o1,o2,w], niw=2
    returned = obj.cut_niw(1, copy=False)
    assert returned is obj
    assert obj.niw == 1


def test_cut_niw_releases_parent_array():
    """cut_niw releases the trimmed parent array instead of keeping a view."""
    obj = LocalNPoint(np.arange(2 * 2 * 10).reshape(2, 2, 10).astype(complex), 2, 1, 0, full_niw_range=True)
    result = obj.cut_niw(2)
    # A bare slice would keep the full pre-cut array alive via ``mat.base``; the copy releases it.
    assert result.mat.base is None
    assert result.mat.shape[-1] == 5


def test_cut_niv_releases_parent_array_and_preserves_values():
    """cut_niv releases the parent array and keeps the central frequency block."""
    mat = np.arange(2 * 2 * 8 * 8).reshape(2, 2, 8, 8).astype(complex)
    obj = LocalNPoint(mat.copy(), 2, 0, 2, full_niv_range=True)
    result = obj.cut_niv(2)
    assert result.mat.base is None
    # niv goes 4 -> 2, keeping the central [2:6, 2:6] block; the original stays untouched (copy=True default)
    assert np.array_equal(result.mat, mat[..., 2:6, 2:6])
    assert obj.niv == 4


def test_cut_niv_copy_false_releases_parent_array():
    """cut_niv(copy=False) releases the parent array."""
    obj = LocalNPoint(np.arange(2 * 2 * 8 * 8).reshape(2, 2, 8, 8).astype(complex), 2, 0, 2)
    result = obj.cut_niv(2, copy=False)
    assert result is obj
    assert result.mat.base is None


def test_cut_niw_and_niv_matches_separate_cuts_and_releases_parent():
    """cut_niw_and_niv yields the same result as chaining cut_niw and cut_niv, in a single freed-parent copy."""
    mat = np.arange(2 * 2 * 11 * 8 * 8).reshape(2, 2, 11, 8, 8).astype(complex)
    obj = LocalNPoint(mat.copy(), 2, 1, 2, full_niw_range=True, full_niv_range=True)
    result = obj.cut_niw_and_niv(2, 3)
    expected = LocalNPoint(mat.copy(), 2, 1, 2, full_niw_range=True, full_niv_range=True).cut_niw(2).cut_niv(3)
    assert np.array_equal(result.mat, expected.mat)
    assert result.mat.base is None


def test_cut_niw_and_niv_copy_false_mutates_self_in_place():
    """cut_niw_and_niv(copy=False) mutates and returns self."""
    obj = LocalNPoint(np.arange(2 * 2 * 11 * 8 * 8).reshape(2, 2, 11, 8, 8).astype(complex), 2, 1, 2)
    result = obj.cut_niw_and_niv(2, 3, copy=False)
    assert result is obj
    assert obj.niw == 2 and obj.niv == 3


def test_cut_niw_and_niv_copy_true_leaves_original_untouched():
    """cut_niw_and_niv(copy=True) does not modify the original object."""
    obj = LocalNPoint(np.arange(2 * 2 * 11 * 8 * 8).reshape(2, 2, 11, 8, 8).astype(complex), 2, 1, 2)
    _ = obj.cut_niw_and_niv(2, 3)
    assert obj.niw == 5 and obj.niv == 4


def test_cut_niw_and_niv_no_op_returns_self_unchanged():
    """cut_niw_and_niv with both cutoffs not smaller than the range returns self unchanged."""
    obj = LocalNPoint(np.arange(2 * 2 * 11 * 8 * 8).reshape(2, 2, 11, 8, 8).astype(complex), 2, 1, 2)
    assert obj.cut_niw_and_niv(10, 10) is obj


def test_cut_niw_and_niv_raises_without_bosonic_dimension():
    """cut_niw_and_niv raises when there is no bosonic frequency dimension."""
    obj = LocalNPoint(np.zeros((2, 2, 8, 8)), 2, 0, 2)
    with pytest.raises(ValueError):
        obj.cut_niw_and_niv(2, 3)


def test_cut_niw_and_niv_raises_without_fermionic_dimension():
    """cut_niw_and_niv raises when there is no fermionic frequency dimension."""
    obj = LocalNPoint(np.zeros((2, 2, 11)), 2, 1, 0)
    with pytest.raises(ValueError):
        obj.cut_niw_and_niv(2, 3)


def test_to_half_niv_range_releases_parent_and_keeps_positive_half():
    """to_half_niv_range releases the parent and keeps the positive fermionic half."""
    mat = np.arange(2 * 2 * 8).reshape(2, 2, 8).astype(complex)
    obj = LocalNPoint(mat.copy(), 2, 0, 1, full_niv_range=True)  # niv=4
    result = obj.to_half_niv_range()
    assert result.mat.base is None
    assert result.full_niv_range is False
    assert np.array_equal(result.mat, mat[..., 4:])


def test_take_vn_diagonal_is_writeable_and_releases_parent():
    """take_vn_diagonal returns a writeable array that releases the parent."""
    mat = np.arange(2 * 2 * 3 * 4 * 4).reshape(2, 2, 3, 4, 4).astype(complex)
    obj = LocalNPoint(mat.copy(), 2, 1, 2)
    result = obj.take_vn_diagonal()
    assert result.num_vn_dimensions == 1
    assert result.mat.flags.writeable  # diagonal() alone would yield a read-only view
    assert result.mat.base is None
    assert np.array_equal(result.mat, np.diagonal(mat, axis1=-2, axis2=-1))
    result.mat[0, 0, 0, 0] = 123.0
    assert result.mat[0, 0, 0, 0] == 123.0


@pytest.mark.parametrize("num_vn", [0, 1, 2])
def test_to_negative_niw_range_twice_returns_original(num_vn):
    """to_negative_niw_range is its own inverse, returns a new object, keeps the niw+1 half-range entries."""
    nb, niw, niv = 2, 3, 2
    shape = (nb, nb, niw + 1) + (2 * niv,) * num_vn
    mat = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    obj = LocalNPoint(mat.copy(), 2, 1, num_vn, full_niw_range=False)

    once = obj.to_negative_niw_range()
    twice = once.to_negative_niw_range()

    assert twice is not obj
    assert once.mat.shape == obj.mat.shape
    assert np.allclose(twice.mat, obj.mat)
    assert np.allclose(obj.mat, mat)


def test_to_negative_niw_range_raises_when_in_full_bosonic_range():
    """to_negative_niw_range raises when in the full bosonic range."""
    mat = np.random.rand(2, 2, 7, 4) + 1j * np.random.rand(2, 2, 7, 4)
    obj = LocalNPoint(mat, 2, 1, 1, full_niw_range=True)
    with pytest.raises(ValueError):
        obj.to_negative_niw_range()


def test_to_negative_niw_range_raises_without_bosonic_dimension():
    """to_negative_niw_range raises without a bosonic-frequency dimension."""
    mat = np.random.rand(2, 2, 4) + 1j * np.random.rand(2, 2, 4)
    obj = LocalNPoint(mat, 2, 0, 1, full_niw_range=False)  # no bosonic frequency dimension
    with pytest.raises(ValueError):
        obj.to_negative_niw_range()


@pytest.mark.parametrize("num_vn", [0, 1])
def test_to_negative_niw_range_matches_full_niw_range_negative_block(num_vn):
    """to_negative_niw_range index k (w=-k) equals the full-range object's slot niw-k for k = 1..niw."""
    nb, niw, niv = 2, 3, 2
    shape = (nb, nb, niw + 1) + (2 * niv,) * num_vn
    mat = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    full = LocalNPoint(mat.copy(), 2, 1, num_vn, full_niw_range=False).to_full_niw_range()
    neg = LocalNPoint(mat.copy(), 2, 1, num_vn, full_niw_range=False).to_negative_niw_range()

    w_axis = -(1 + num_vn)
    for k in range(1, niw + 1):
        neg_slice = np.take(neg.mat, k, axis=w_axis)
        full_slice = np.take(full.mat, niw - k, axis=w_axis)
        assert np.allclose(neg_slice, full_slice)


@pytest.mark.parametrize("num_vn", [0, 1, 2])
def test_niv_first_and_niv_second_agree_on_a_symmetric_box(num_vn):
    """Both fermionic-axis accessors coincide with niv whenever the box is symmetric or has at most one axis."""
    nb, niw, niv = 2, 2, 3
    shape = (nb, nb, 2 * niw + 1) + (2 * niv,) * num_vn
    obj = LocalNPoint(np.zeros(shape, dtype=np.complex64), 2, 1, num_vn)
    assert obj.niv_first == obj.niv_second == obj.niv == (niv if num_vn else 0)


@pytest.mark.parametrize("full_niv_range", [True, False])
def test_niv_first_and_niv_second_report_their_own_axis_of_an_asymmetric_box(full_niv_range):
    """On an asymmetric nu x nu' box each accessor reports its own axis, with niv_second aliasing niv."""
    nb, niw, niv_first, niv_second = 2, 2, 5, 2
    shape = (nb, nb, 2 * niw + 1, 2 * niv_first, 2 * niv_second)
    obj = LocalNPoint(np.zeros(shape, dtype=np.complex64), 2, 1, 2, full_niv_range=full_niv_range)
    assert obj.niv_first == niv_first and obj.niv_second == niv_second == obj.niv


def test_cut_niv_cuts_each_fermionic_axis_against_its_own_width():
    """cut_niv slices an asymmetric nu x nu' box per axis instead of reusing the last axis width."""
    nb, niw, niv_first, niv, niv_cut = 2, 1, 5, 3, 2
    shape = (nb, nb, 2 * niw + 1, 2 * niv_first, 2 * niv)
    mat = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    obj = LocalNPoint(mat.copy(), 2, 1, 2)

    out = obj.cut_niv(niv_cut)

    expected = mat[..., niv_first - niv_cut : niv_first + niv_cut, niv - niv_cut : niv + niv_cut]
    assert out.mat.shape == (nb, nb, 2 * niw + 1, 2 * niv_cut, 2 * niv_cut)
    assert np.allclose(out.mat, expected, atol=1e-12)


def test_cut_niv_still_trims_the_first_axis_when_only_it_is_wider():
    """A cutoff between nu' and nu leaves nu' untouched and trims nu down to the cutoff."""
    nb, niw, niv_first, niv, niv_cut = 2, 1, 6, 2, 3
    shape = (nb, nb, 2 * niw + 1, 2 * niv_first, 2 * niv)
    mat = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    out = LocalNPoint(mat.copy(), 2, 1, 2).cut_niv(niv_cut)

    assert out.mat.shape == (nb, nb, 2 * niw + 1, 2 * niv_cut, 2 * niv)
    assert np.allclose(out.mat, mat[..., niv_first - niv_cut : niv_first + niv_cut, :], atol=1e-12)


def test_cut_niw_and_niv_cuts_each_fermionic_axis_against_its_own_width():
    """The fused bosonic/fermionic cut slices an asymmetric nu x nu' box per axis."""
    nb, niw, niv_first, niv, niw_cut, niv_cut = 2, 3, 5, 3, 1, 2
    shape = (nb, nb, 2 * niw + 1, 2 * niv_first, 2 * niv)
    mat = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    out = LocalNPoint(mat.copy(), 2, 1, 2).cut_niw_and_niv(niw_cut, niv_cut)

    expected = mat[
        ...,
        niw - niw_cut : niw + niw_cut + 1,
        niv_first - niv_cut : niv_first + niv_cut,
        niv - niv_cut : niv + niv_cut,
    ]
    assert out.mat.shape == (nb, nb, 2 * niw_cut + 1, 2 * niv_cut, 2 * niv_cut)
    assert np.allclose(out.mat, expected, atol=1e-12)


@pytest.mark.parametrize("num_vn", [1, 2])
def test_cut_niv_on_a_half_fermionic_range_truncates_at_an_equal_cutoff(num_vn):
    """On a half fermionic range a cutoff equal to niv still truncates the axis rather than keeping it whole."""
    nb, niw, niv = 2, 1, 3
    shape = (nb, nb, 2 * niw + 1) + (2 * niv,) * num_vn
    mat = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    obj = LocalNPoint(mat.copy(), 2, 1, num_vn, full_niv_range=False)

    out = obj.cut_niv(obj.niv)

    assert out.current_shape == (nb, nb, 2 * niw + 1) + (niv,) * num_vn
    assert np.allclose(out.mat, mat[(...,) + (slice(0, niv),) * num_vn], atol=1e-12)


@pytest.mark.parametrize("num_vn", [1, 2])
def test_take_wn_slice_restricts_a_half_range_object_to_a_bosonic_window(num_vn):
    """take_wn_slice returns an independent copy holding only the requested bosonic index window."""
    nb, n_w, niv = 2, 5, 3
    shape = (nb, nb, n_w) + (2 * niv,) * num_vn
    mat = (np.random.rand(*shape) + 1j * np.random.rand(*shape)).astype(np.complex64)
    obj = LocalNPoint(mat.copy(), 2, 1, num_vn, full_niw_range=False)

    out = obj.take_wn_slice(1, 3)

    assert out.current_shape == (nb, nb, 2) + (2 * niv,) * num_vn
    assert np.array_equal(out.mat, mat[(..., slice(1, 3)) + (slice(None),) * num_vn])
    assert np.array_equal(obj.mat, mat)


def test_take_wn_slice_rejects_a_full_bosonic_range_object():
    """take_wn_slice refuses a full-range object, whose bosonic index 0 is not omega = 0."""
    obj = LocalNPoint(np.zeros((2, 2, 5, 6)), 2, 1, 1, full_niw_range=True)
    with pytest.raises(ValueError):
        obj.take_wn_slice(1, 3)

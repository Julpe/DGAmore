# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

from unittest.mock import MagicMock

import numpy as np
import pytest

import dgamore.brillouin_zone as bz
from dgamore.brillouin_zone import KPath, KnownKPoints, Labels


def test_applies_inversion_symmetry_along_x_axis():
    """Inversion symmetry along the x axis maps the matrix correctly."""
    mat = np.random.rand(6, 4, 4)
    bz.inv_sym(mat, axis=0)
    assert np.allclose(mat[4:, :, :], mat[1:3, :, :][::-1])


def test_applies_inversion_symmetry_along_y_axis():
    """Inversion symmetry along the y axis maps the matrix correctly."""
    mat = np.random.rand(4, 6, 4)
    bz.inv_sym(mat, axis=1)
    assert np.allclose(mat[:, 4:, :], mat[:, 1:3, :][:, ::-1])


def test_applies_inversion_symmetry_along_z_axis():
    """Inversion symmetry along the z axis maps the matrix correctly."""
    mat = np.random.rand(4, 4, 6)
    bz.inv_sym(mat, axis=2)
    assert np.allclose(mat[:, :, 4:], mat[:, :, 1:3][:, :, ::-1])


def test_raises_error_for_invalid_axis():
    """Inversion symmetry raises for an invalid axis."""
    mat = np.random.rand(4, 4, 4)
    with pytest.raises(AssertionError, match=r"axis = 3 but must be in \[0,1,2\]"):
        bz.inv_sym(mat, axis=3)


def test_raises_error_for_insufficient_dimensions_on_inv_sym():
    """Inversion symmetry raises for insufficient dimensions."""
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError, match=r"dim\(mat\) = 2 but must be at least 3 dimensional"):
        bz.inv_sym(mat, axis=0)


def test_applies_x_y_symmetry_to_square_matrix():
    """The x-y exchange symmetry maps a square matrix correctly."""
    mat = np.random.rand(4, 4, 6)
    bz.x_y_sym(mat)
    assert np.allclose(mat, np.minimum(mat, mat.swapaxes(0, 1)))


def test_applies_x_z_symmetry_to_square_matrix():
    """The x-z exchange symmetry maps a square matrix correctly."""
    mat = np.random.rand(4, 6, 4)
    bz.x_z_sym(mat)
    assert np.allclose(mat, np.minimum(mat, mat.swapaxes(0, 2)))


def test_applies_y_z_symmetry_to_square_matrix():
    """The y-z exchange symmetry maps a square matrix correctly."""
    mat = np.random.rand(6, 4, 4)
    bz.y_z_sym(mat)
    assert np.allclose(mat, np.minimum(mat, mat.swapaxes(1, 2)))


def test_does_nothing_for_non_square_matrix():
    """The exchange symmetry leaves a non-square matrix unchanged."""
    mat = np.random.rand(4, 5, 6)
    original_mat = mat.copy()
    bz.x_y_sym(mat)
    assert np.allclose(mat, original_mat)


def test_raises_error_for_insufficient_dimensions_on_x_y_sym():
    """The x-y symmetry raises for insufficient dimensions."""
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError):
        bz.x_y_sym(mat)


def test_raises_error_for_insufficient_dimensions_on_x_z_sym():
    """The x-z symmetry raises for insufficient dimensions."""
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError):
        bz.x_z_sym(mat)


def test_raises_error_for_insufficient_dimensions_on_y_z_sym():
    """The y-z symmetry raises for insufficient dimensions."""
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError):
        bz.y_z_sym(mat)


def test_applies_simultaneous_inversion_in_x_and_y_directions():
    """Simultaneous x and y inversion maps the matrix correctly."""
    mat = np.random.rand(6, 6, 4)
    bz.x_y_inv(mat)
    assert np.allclose(mat[4:, 4:, :], mat[1:3, 1:3, :][::-1, ::-1, :])


def test_raises_error_for_insufficient_dimensions_on_x_y_inv():
    """Simultaneous x-y inversion raises for insufficient dimensions."""
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError, match=r"dim\(mat\) = 2 but must be at least 3 dimensional"):
        bz.x_y_inv(mat)


def test_applies_x_inversion_symmetry_correctly_with_mock(monkeypatch):
    """apply_symmetry dispatches the x inversion to the correct helper."""
    mat = np.random.rand(6, 4, 4)
    mock_inv_sym = MagicMock()
    with monkeypatch.context() as mp:
        mp.setattr(bz, "inv_sym", mock_inv_sym)
        bz.apply_symmetry(mat, bz.KnownSymmetries.X_INV)
    mock_inv_sym.assert_called_once_with(mat, 0)


def test_applies_y_inversion_symmetry_correctly_with_mock(monkeypatch):
    """apply_symmetry dispatches the y inversion to the correct helper."""
    mat = np.random.rand(4, 6, 4)
    mock_inv_sym = MagicMock()
    with monkeypatch.context() as mp:
        mp.setattr(bz, "inv_sym", mock_inv_sym)
        bz.apply_symmetry(mat, bz.KnownSymmetries.Y_INV)
    mock_inv_sym.assert_called_once_with(mat, 1)


def test_applies_z_inversion_symmetry_correctly_with_mock(monkeypatch):
    """apply_symmetry dispatches the z inversion to the correct helper."""
    mat = np.random.rand(4, 4, 6)
    mock_inv_sym = MagicMock()
    with monkeypatch.context() as mp:
        mp.setattr(bz, "inv_sym", mock_inv_sym)
        bz.apply_symmetry(mat, bz.KnownSymmetries.Z_INV)
    mock_inv_sym.assert_called_once_with(mat, 2)


def test_applies_x_y_symmetry_correctly_with_mock(monkeypatch):
    """apply_symmetry dispatches the x-y exchange to the correct helper."""
    mat = np.random.rand(4, 4, 6)
    mock_x_y_sym = MagicMock()
    with monkeypatch.context() as mp:
        mp.setattr(bz, "x_y_sym", mock_x_y_sym)
        bz.apply_symmetry(mat, bz.KnownSymmetries.X_Y_SYM)
    mock_x_y_sym.assert_called_once_with(mat)


def test_applies_x_z_symmetry_correctly_with_mock(monkeypatch):
    """apply_symmetry dispatches the x-z exchange to the correct helper."""
    mat = np.random.rand(4, 6, 4)
    mock_x_z_sym = MagicMock()
    with monkeypatch.context() as mp:
        mp.setattr(bz, "x_z_sym", mock_x_z_sym)
        bz.apply_symmetry(mat, bz.KnownSymmetries.X_Z_SYM)
    mock_x_z_sym.assert_called_once_with(mat)


def test_applies_y_z_symmetry_correctly_with_mock(monkeypatch):
    """apply_symmetry dispatches the y-z exchange to the correct helper."""
    mat = np.random.rand(6, 4, 4)
    mock_y_z_sym = MagicMock()
    with monkeypatch.context() as mp:
        mp.setattr(bz, "y_z_sym", mock_y_z_sym)
        bz.apply_symmetry(mat, bz.KnownSymmetries.Y_Z_SYM)
    mock_y_z_sym.assert_called_once_with(mat)


def test_applies_x_y_inversion_symmetry_correctly_with_mock(monkeypatch):
    """apply_symmetry dispatches the simultaneous x-y inversion to the correct helper."""
    mat = np.random.rand(6, 6, 4)
    mock_x_y_inv = MagicMock()
    with monkeypatch.context() as mp:
        mp.setattr(bz, "x_y_inv", mock_x_y_inv)
        bz.apply_symmetry(mat, bz.KnownSymmetries.X_Y_INV)
    mock_x_y_inv.assert_called_once_with(mat)


def test_raises_error_for_unknown_symmetry_with_mock(monkeypatch):
    """apply_symmetry raises for an unknown symmetry."""
    mat = np.random.rand(4, 4, 4)
    mock_known_symmetries = MagicMock()
    with monkeypatch.context() as mp:
        mp.setattr(bz, "KnownSymmetries", mock_known_symmetries)
        with pytest.raises(AssertionError, match="sym = .* not in known symmetries .*"):
            bz.apply_symmetry(mat, "unknown_symmetry")
    mock_known_symmetries.__contains__.assert_called()


def test_applies_multiple_symmetries_in_order(monkeypatch):
    """apply_symmetries applies multiple symmetries in order."""
    mat = np.random.rand(6, 6, 6)
    mock_apply_symmetry = MagicMock()
    with monkeypatch.context() as mp:
        mp.setattr(bz, "apply_symmetry", mock_apply_symmetry)
        bz.apply_symmetries(mat, [bz.KnownSymmetries.X_INV, bz.KnownSymmetries.Y_INV, bz.KnownSymmetries.Z_INV])
    mock_apply_symmetry.assert_any_call(mat, bz.KnownSymmetries.X_INV)
    mock_apply_symmetry.assert_any_call(mat, bz.KnownSymmetries.Y_INV)
    mock_apply_symmetry.assert_any_call(mat, bz.KnownSymmetries.Z_INV)
    assert mock_apply_symmetry.call_count == 3


def test_does_nothing_when_no_symmetries_provided(monkeypatch):
    """apply_symmetries does nothing when no symmetries are provided."""
    mat = np.random.rand(6, 6, 6)
    mock_apply_symmetry = MagicMock()
    with monkeypatch.context() as mp:
        mp.setattr(bz, "apply_symmetry", mock_apply_symmetry)
        bz.apply_symmetries(mat, [])
    mock_apply_symmetry.assert_not_called()


def test_raises_error_for_insufficient_dimensions_on_apply_symmetries():
    """apply_symmetries raises for insufficient dimensions."""
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError, match=r"dim\(mat\) = 2 but must at least 3 dimensional"):
        bz.apply_symmetries(mat, [bz.KnownSymmetries.X_INV])


def test_returns_correct_symmetries_for_two_dimensional_square():
    """two_dimensional_square_symmetries returns the expected symmetry list."""
    result = bz.get_lattice_symmetries_from_string("two_dimensional_square")
    assert result == bz.two_dimensional_square_symmetries()


def test_returns_correct_symmetries_for_three_dimensional_cubic():
    """three_dimensional_cubic_symmetries returns the expected symmetry list."""
    result = bz.get_lattice_symmetries_from_string("three_dimensional_cubic")
    assert result == bz.three_dimensional_cubic_symmetries()


def test_returns_correct_symmetries_for_quasi_one_dimensional_square():
    """quasi_one_dimensional_square_symmetries returns the expected symmetry list."""
    result = bz.get_lattice_symmetries_from_string("quasi_one_dimensional_square")
    assert result == bz.quasi_one_dimensional_square_symmetries()


def test_returns_correct_symmetries_for_simultaneous_x_y_inversion():
    """The simultaneous x-y inversion symmetry list is built correctly."""
    result = bz.get_lattice_symmetries_from_string("simultaneous_x_y_inversion")
    assert result == bz.simultaneous_x_y_inversion()


def test_returns_correct_symmetries_for_quasi_two_dimensional_square_symmetries():
    """quasi_two_dimensional_square_symmetries returns the expected symmetry list."""
    result = bz.get_lattice_symmetries_from_string("quasi_two_dimensional_square_symmetries")
    assert result == bz.quasi_two_dimensional_square_symmetries()


def test_returns_empty_list_for_none_or_empty_string():
    """get_lattice_symmetries_from_string returns an empty list for None or an empty string."""
    result_none = bz.get_lattice_symmetries_from_string(None)
    result_empty = bz.get_lattice_symmetries_from_string("")
    assert result_none == []
    assert result_empty == []


def test_raises_error_for_unsupported_symmetry_string():
    """get_lattice_symmetries_from_string raises for an unsupported symmetry string."""
    with pytest.raises(ValueError, match="Symmetry does not exist or input cannot be parsed as a Python literal."):
        bz.get_lattice_symmetries_from_string("unsupported_symmetry")


def test_raises_error_for_unsupported_symmetry_in_list():
    """get_lattice_symmetries_from_string raises for an unsupported symmetry in a list."""
    with pytest.raises(NotImplementedError, match="Symmetry unsupported_symmetry not supported."):
        bz.get_lattice_symmetries_from_string(["x-inv", "unsupported_symmetry"])


def test_returns_correct_symmetries_for_list_of_valid_symmetries():
    """get_lattice_symmetries_from_string parses a list of valid symmetries."""
    result = bz.get_lattice_symmetries_from_string(["x-inv", "y-inv"])
    assert result == [bz.KnownSymmetries.X_INV, bz.KnownSymmetries.Y_INV]


def test_maps_full_bz_to_irreducible_correctly(monkeypatch):
    """The full BZ is mapped to the irreducible BZ correctly."""
    nk = (4, 4, 4)
    symmetries = [bz.KnownSymmetries.X_INV, bz.KnownSymmetries.Y_INV]
    kgrid = bz.KGrid(nk=nk, symmetries=symmetries)
    mock_apply_symmetries = MagicMock()
    with monkeypatch.context() as mp:
        mp.setattr(bz, "apply_symmetries", mock_apply_symmetries)
        kgrid.set_fbz2irrk()
    mock_apply_symmetries.assert_called_once_with(kgrid.fbz2irrk, symmetries)


def test_handles_empty_symmetry_list_without_error(monkeypatch):
    """Mapping handles an empty symmetry list without error."""
    nk = (4, 4, 4)
    symmetries = []
    kgrid = bz.KGrid(nk=nk, symmetries=symmetries)
    mock_apply_symmetries = MagicMock()
    with monkeypatch.context() as mp:
        mp.setattr(bz, "apply_symmetries", mock_apply_symmetries)
        kgrid.set_fbz2irrk()
    mock_apply_symmetries.assert_called_once_with(kgrid.fbz2irrk, symmetries)


def test_maps_unique_elements_correctly_to_indices(monkeypatch):
    """Unique elements are mapped correctly to indices."""
    kgrid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    mock_unique = MagicMock(wraps=np.unique)
    with monkeypatch.context() as mp:
        mp.setattr(np, "unique", mock_unique)
        kgrid.set_fbz2irrk()
        kgrid.set_irrk_maps()
    mock_unique.assert_called_once_with(kgrid.fbz2irrk, return_index=True, return_inverse=True, return_counts=True)


def test_handles_empty_input_without_error():
    """Mapping handles an empty input without error."""
    fbz2irrk = np.array([])
    kgrid = bz.KGrid(nk=(0, 0, 0), symmetries=[])
    kgrid.fbz2irrk = fbz2irrk
    kgrid.set_irrk_maps()
    assert kgrid.irrk_ind.size == 0
    assert kgrid.irrk_inv.size == 0
    assert kgrid.irrk_count.size == 0


def test_sets_irrk_mesh_correctly_for_valid_input():
    """set_irrk_mesh builds the irreducible mesh for valid input."""
    nk = (4, 4, 4)
    symmetries = [bz.KnownSymmetries.X_INV, bz.KnownSymmetries.Y_INV]
    kgrid = bz.KGrid(nk=nk, symmetries=symmetries)
    kgrid.set_irrk_mesh()
    assert kgrid.irr_kmesh.shape == (3, kgrid.nk_irr)


def test_returns_correct_kx_shift_for_valid_input():
    """The kx shift is computed correctly."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    expected_shift = kgrid.kx - np.pi
    assert np.allclose(kgrid.kx_shift, expected_shift)


def test_returns_correct_ky_shift_for_valid_input():
    """The ky shift is computed correctly."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    expected_shift = kgrid.ky - np.pi
    assert np.allclose(kgrid.ky_shift, expected_shift)


def test_returns_correct_kz_shift_for_valid_input():
    """The kz shift is computed correctly."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    expected_shift = kgrid.kz - np.pi
    assert np.allclose(kgrid.kz_shift, expected_shift)


def test_returns_correct_kx_shift_closed_for_valid_input():
    """The closed kx shift is computed correctly."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    expected_shift_closed = np.array([*(kgrid.kx - np.pi), -kgrid.kx[0] + np.pi])
    assert np.allclose(kgrid.kx_shift_closed, expected_shift_closed)


def test_returns_correct_ky_shift_closed_for_valid_input():
    """The closed ky shift is computed correctly."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    expected_shift_closed = np.array([*(kgrid.ky - np.pi), -kgrid.ky[0] + np.pi])
    assert np.allclose(kgrid.ky_shift_closed, expected_shift_closed)


def test_returns_correct_kz_shift_closed_for_valid_input():
    """The closed kz shift is computed correctly."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    expected_shift_closed = np.array([*(kgrid.kz - np.pi), -kgrid.kz[0] + np.pi])
    assert np.allclose(kgrid.kz_shift_closed, expected_shift_closed)


def test_returns_correct_k_grid_as_tuple():
    """The k-grid is returned as the expected tuple."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    kx, ky, kz = kgrid.grid
    assert np.array_equal(kx, kgrid.kx)
    assert np.array_equal(ky, kgrid.ky)
    assert np.array_equal(kz, kgrid.kz)


def test_calculates_total_number_of_k_points_correctly():
    """nk_tot is the total number of k-points."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    assert kgrid.nk_tot == 64


def test_calculates_number_of_irreducible_k_points_correctly():
    """nk_irr is the number of irreducible k-points."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[bz.KnownSymmetries.X_INV])
    assert kgrid.nk_irr == len(np.unique(kgrid.fbz2irrk))


def test_returns_correct_k_meshgrid():
    """The k meshgrid is built correctly."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    kmesh = kgrid.kmesh
    assert kmesh.shape == (3, 4, 4, 4)
    assert np.array_equal(kmesh[0], np.meshgrid(kgrid.kx, kgrid.ky, kgrid.kz, indexing="ij")[0])
    assert np.array_equal(kmesh[1], np.meshgrid(kgrid.kx, kgrid.ky, kgrid.kz, indexing="ij")[1])
    assert np.array_equal(kmesh[2], np.meshgrid(kgrid.kx, kgrid.ky, kgrid.kz, indexing="ij")[2])


def test_returns_correct_kmesh_list_for_valid_input():
    """The kmesh list is built correctly."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    kmesh_list = kgrid.kmesh_list
    assert kmesh_list.shape == (3, 64)
    assert np.array_equal(kmesh_list[0], kgrid.kmesh[0].flatten())
    assert np.array_equal(kmesh_list[1], kgrid.kmesh[1].flatten())
    assert np.array_equal(kmesh_list[2], kgrid.kmesh[2].flatten())


def test_sets_k_axes_correctly_for_valid_input():
    """The k axes are set correctly."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    assert np.allclose(kgrid.kx, np.linspace(0, 2 * np.pi, 4, endpoint=False))
    assert np.allclose(kgrid.ky, np.linspace(0, 2 * np.pi, 4, endpoint=False))
    assert np.allclose(kgrid.kz, np.linspace(0, 2 * np.pi, 4, endpoint=False))


def test_returns_correct_q_list_for_valid_input():
    """get_q_list returns the expected q-list."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    q_list = kgrid.get_q_list()
    assert q_list.shape == (64, 3)
    assert np.array_equal(q_list[:, 0], kgrid.kmesh_ind[0].flatten())
    assert np.array_equal(q_list[:, 1], kgrid.kmesh_ind[1].flatten())
    assert np.array_equal(q_list[:, 2], kgrid.kmesh_ind[2].flatten())


def test_returns_correct_irrq_list_for_valid_input():
    """get_irrq_list returns the expected irreducible q-list."""
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[bz.KnownSymmetries.X_INV])
    irrq_list = kgrid.get_irrq_list()
    assert irrq_list.shape == (kgrid.nk_irr, 3)
    assert np.array_equal(irrq_list[:, 0], kgrid.kmesh_ind[0].flatten()[kgrid.irrk_ind])
    assert np.array_equal(irrq_list[:, 1], kgrid.kmesh_ind[1].flatten()[kgrid.irrk_ind])
    assert np.array_equal(irrq_list[:, 2], kgrid.kmesh_ind[2].flatten()[kgrid.irrk_ind])


def test_corner_k_points_and_label_mapping_for_known_labels():
    """Corner k-points and label mapping resolve known high-symmetry labels."""
    kx = np.arange(4)
    kp = KPath(nk=(4, 4, 4), path="gamma-x", kx=kx, ky=kx, kz=kx)

    assert kp.ckps == ["gamma", "x"]

    ckp = kp.corner_k_points()

    assert np.allclose(ckp[0], np.array(KnownKPoints.GAMMA.value))
    assert np.allclose(ckp[1], np.array(KnownKPoints.X.value))

    assert kp.labels == [Labels.GAMMA.latex, Labels.X.latex]


def test_map_to_kpath_and_get_kpoints_return_expected_values():
    """map_to_kpath and get_kpoints return the expected path points."""
    kx = np.arange(4)
    kp = KPath(nk=(4, 4, 4), path="gamma-x", kx=kx, ky=kx, kz=kx)

    mat = np.arange(4 * 4 * 4).reshape(4, 4, 4)
    mapped = kp.map_to_kpath(mat)

    expected = np.array([mat[tuple(kp.kpts[i])] for i in range(kp.kpts.shape[0])])
    assert np.array_equal(mapped, expected)

    kpoints = kp.get_kpoints()
    assert kpoints.shape == (kp.kpts.shape[0], 3)

    assert np.array_equal(kpoints[:, 0], kp.kpts[:, 0])
    assert np.array_equal(kpoints[:, 1], kp.kpts[:, 1])
    assert np.array_equal(kpoints[:, 2], kp.kpts[:, 2])


def test_corner_k_points_accepts_numeric_string_points():
    """Corner k-points accept numeric string coordinates."""
    kx = np.arange(4)
    kp = KPath(nk=(4, 4, 4), path="gamma-0.25 0.25 0", kx=kx, ky=kx, kz=kx)

    ckp = kp.corner_k_points()
    assert np.allclose(ckp[0], np.array(KnownKPoints.GAMMA.value))
    assert np.allclose(ckp[1], np.array([0.25, 0.25, 0.0]))


def test_nk_tot_returns_sum_of_nkp():
    """KPath.nk_tot returns the sum of the per-segment point counts."""
    kp = KPath(nk=(4, 4, 4), path="gamma-x")
    kp.nkp = [2, 3, 1]
    assert kp.nk_tot == 6


def test_nk_seg_returns_diff_of_cind():
    """KPath.nk_seg returns the per-segment point counts."""
    kp = KPath(nk=(4, 4, 4), path="gamma-x")
    kp.nkp = [2, 3, 1]
    cind = np.concatenate(([0], np.cumsum(kp.nkp) - 1))
    expected = np.diff(cind)
    assert np.array_equal(kp.nk_seg, expected)


def test_k_axis_normalized_positions_and_length():
    """The k-axis normalizes cumulative distances to [0,1]: four unit-step points give [0, 1/3, 2/3, 1]."""
    kp = KPath(nk=(4, 4, 4), path="gamma-x")
    kp.kpts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    kp.nkp = [2, 2]
    expected = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    assert np.allclose(kp.k_axis, expected, atol=1e-12)
    assert kp.k_axis.size == kp.nk_tot


def test_build_k_path_single_segment_gamma_to_x():
    """For nk=(4,4,4) the 'gamma-x' segment yields indices [[0,0,0],[1,0,0]] and nkp [2]."""
    kp = KPath(nk=(4, 4, 4), path="gamma-x")
    k_path, nkp = kp.build_k_path()

    expected = np.array([[0, 0, 0], [1, 0, 0]])
    assert isinstance(k_path, np.ndarray)
    assert np.array_equal(k_path, expected)
    assert nkp == [2]


def test_get_kpath_val_reads_each_axis_for_its_own_column():
    """get_kpath_val maps the x/y/z path columns through kx/ky/kz (regression: kx was used for all three)."""
    kp = KPath(nk=(4, 4, 4), path="gamma-x")
    kp.kx = np.array([10.0, 11.0, 12.0])
    kp.ky = np.array([20.0, 21.0, 22.0])
    kp.kz = np.array([30.0, 31.0, 32.0])
    kp.kpts = np.array([[0, 1, 2], [2, 0, 1]])

    kx_vals, ky_vals, kz_vals = kp.get_kpath_val()

    assert np.array_equal(kx_vals, np.array([10.0, 12.0]))
    assert np.array_equal(ky_vals, np.array([21.0, 20.0]))
    assert np.array_equal(kz_vals, np.array([32.0, 31.0]))


def test_get_bands_returns_sorted_real_eigenvalues(monkeypatch):
    """get_bands returns sorted real eigenvalues for each k-point of the object KPath.map_to_kpath yields."""
    kp = KPath(nk=(4, 4, 4), path="gamma-x")
    # two diagonal 2x2 matrices with eigenvalues [2, 1] and [4, 3], expected sorted to [1, 2] and [3, 4]
    mats = [np.array([[2.0, 0.0], [0.0, 1.0]]), np.array([[4.0, 0.0], [0.0, 3.0]])]
    mock_ek = MagicMock(current_shape=(len(mats), mats[0].shape[0], mats[0].shape[0]))
    mock_ek.__iter__.side_effect = lambda: iter(mats)

    with monkeypatch.context() as mp:
        mp.setattr(KPath, "map_to_kpath", MagicMock(return_value=mock_ek))
        bands = kp.get_bands(ek=None)

    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(bands, expected)


def test_is_auto_symmetries_true_for_auto_flag():
    """is_auto_symmetries is True for a list/tuple carrying the KnownSymmetries.AUTO flag."""
    assert bz.is_auto_symmetries([bz.KnownSymmetries.AUTO]) is True
    assert bz.is_auto_symmetries((bz.KnownSymmetries.AUTO,)) is True


def test_is_auto_symmetries_false_for_plain_symmetry_list():
    """is_auto_symmetries is False for a symmetry list without the AUTO flag."""
    assert bz.is_auto_symmetries(bz.two_dimensional_square_symmetries()) is False
    assert bz.is_auto_symmetries(bz.three_dimensional_cubic_symmetries()) is False


def test_is_auto_symmetries_false_for_empty_list_none_or_other():
    """is_auto_symmetries is False for an empty list, None, or non-list values (incl. the raw "auto" string)."""
    assert bz.is_auto_symmetries([]) is False
    assert bz.is_auto_symmetries(None) is False
    # The raw string "auto" is NOT auto - callers must go through get_lattice_symmetries_from_string.
    assert bz.is_auto_symmetries("auto") is False
    assert bz.is_auto_symmetries(0) is False
    assert bz.is_auto_symmetries({}) is False


def test_get_lattice_symmetries_from_string_returns_auto_flag_for_auto():
    """The "auto" string is the public entry point for auto-discovery; it resolves to [KnownSymmetries.AUTO]."""
    assert bz.get_lattice_symmetries_from_string("auto") == [bz.KnownSymmetries.AUTO]


def test_get_lattice_symmetries_from_string_auto_is_case_insensitive():
    """Lowercase normalization is applied to all string inputs; "auto" / "AUTO" / "Auto" all work."""
    for s in ("auto", "AUTO", "Auto", "AuTo"):
        assert bz.get_lattice_symmetries_from_string(s) == [bz.KnownSymmetries.AUTO]


def _make_small_real_cubic_h(nx=4, ny=4, nz=4, nb=1):
    """A small real Hermitian cubic-symmetric single-band H; the auto-discovered group is the spatial cubic group."""
    j1, j2, j3 = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    k1 = 2 * np.pi * j1 / nx
    k2 = 2 * np.pi * j2 / ny
    k3 = 2 * np.pi * j3 / nz
    H = np.zeros((nx, ny, nz, nb, nb), dtype=complex)
    e = -2.0 * (np.cos(k1) + np.cos(k2) + np.cos(k3))
    for o in range(nb):
        H[..., o, o] = e
    return H


def test_kgrid_with_auto_sentinel_sets_auto_mode_flag():
    """A KGrid built with the auto sentinel sets the auto-mode flag."""
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=[bz.KnownSymmetries.AUTO])
    assert grid._auto_mode is True


def test_kgrid_with_plain_symmetries_does_not_set_auto_mode():
    """A KGrid built with predefined symmetries must NOT enter auto mode."""
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.three_dimensional_cubic_symmetries())
    assert grid._auto_mode is False


def test_kgrid_with_no_symmetries_does_not_set_auto_mode():
    """A KGrid built with no symmetries does not set auto mode."""
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    assert grid._auto_mode is False


def test_kgrid_auto_mode_starts_with_trivial_ibz_and_no_auto_data():
    """Before specify_auto_symmetries an auto KGrid has IBZ == FBZ and unset auto-data slots."""
    nx, ny, nz = 4, 4, 4
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=[bz.KnownSymmetries.AUTO])
    assert grid._auto_us is None
    assert grid._auto_sigmas is None
    assert grid._auto_conjs is None
    # Trivial IBZ = FBZ before discovery
    assert grid.nk_irr == nx * ny * nz


def test_kgrid_is_auto_property_is_false_before_specify_auto_symmetries():
    """is_auto is False between construction and specify_auto_symmetries and True afterward."""
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=[bz.KnownSymmetries.AUTO])
    assert grid.is_auto is False


def test_kgrid_is_auto_property_is_false_for_plain_symmetry_grid():
    """KGrid.is_auto is False for a plain-symmetry grid."""
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.three_dimensional_cubic_symmetries())
    assert grid.is_auto is False


def test_kgrid_is_auto_property_is_true_after_specify_auto_symmetries():
    """KGrid.is_auto becomes True after specify_auto_symmetries."""
    nx, ny, nz = 4, 4, 4
    H = _make_small_real_cubic_h(nx, ny, nz, nb=1)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=[bz.KnownSymmetries.AUTO])
    grid.specify_auto_symmetries(H)
    assert grid.is_auto is True


def test_specify_auto_symmetries_populates_all_expected_arrays():
    """After a successful call every cached IBZ field and auto-data field is populated and consistent."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=[bz.KnownSymmetries.AUTO])
    grid.specify_auto_symmetries(H)

    assert grid.fbz2irrk.shape == (nx, ny, nz)
    assert grid.irrk_ind is not None
    assert grid.irrk_inv is not None
    assert grid.irrk_count is not None
    assert grid.irr_kmesh is not None
    assert grid.nk_irr == len(grid.irrk_ind)
    assert grid.nk_irr <= nx * ny * nz

    assert grid._auto_us.shape == (nx, ny, nz, nb, nb)
    assert grid._auto_sigmas.shape == (nx, ny, nz)
    assert grid._auto_conjs.shape == (nx, ny, nz)
    assert grid._auto_us.dtype == complex
    assert grid._auto_conjs.dtype == bool


def test_specify_auto_symmetries_produces_consistent_fbz2irrk_and_irrk_inv():
    """irrk_inv inverts irrk_ind w.r.t. fbz2irrk: irrk_ind[irrk_inv[k]] == fbz2irrk.flat[k] for every k."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=[bz.KnownSymmetries.AUTO])
    grid.specify_auto_symmetries(H)

    fbz_flat = grid.fbz2irrk.ravel()
    inv_flat = grid.irrk_inv.ravel()
    # For each FBZ point k: fbz2irrk[k] is the flat IBZ index, irrk_ind[irrk_inv[k]] should equal it
    assert np.array_equal(grid.irrk_ind[inv_flat], fbz_flat)


def test_specify_auto_symmetries_irrk_count_sums_to_full_bz():
    """The duplicity counts must sum to the total number of FBZ points."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=[bz.KnownSymmetries.AUTO])
    grid.specify_auto_symmetries(H)

    assert grid.irrk_count.sum() == nx * ny * nz


def test_specify_auto_symmetries_us_are_unitary():
    """Every stored per-k transformation must be unitary: U U^dag = I."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=[bz.KnownSymmetries.AUTO])
    grid.specify_auto_symmetries(H)
    Us = grid._auto_us.reshape(-1, nb, nb)
    identity = np.eye(nb, dtype=complex)
    products = np.einsum("...ij,...kj->...ik", Us, Us.conj())
    assert np.allclose(products, identity[None, ...], atol=1e-10)


def test_specify_auto_symmetries_sigmas_are_plus_or_minus_one():
    """specify_auto_symmetries yields sigma factors of +/-1."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=[bz.KnownSymmetries.AUTO])
    grid.specify_auto_symmetries(H)
    sig = grid._auto_sigmas.ravel()
    assert np.all(np.isin(sig, [-1.0, +1.0]))


def test_specify_auto_symmetries_default_drops_antiunitary_ops():
    """The default include_antiunitary=False drops time-reversal-like ops, so no per-k transform carries conj=True."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=[bz.KnownSymmetries.AUTO])
    grid.specify_auto_symmetries(H)
    assert int(grid._auto_conjs.sum()) == 0


def test_specify_auto_symmetries_with_include_antiunitary_admits_conj_ops():
    """include_antiunitary=True gives a larger group; for a real H some FBZ points carry conj=True."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=[bz.KnownSymmetries.AUTO])
    grid.specify_auto_symmetries(H, include_antiunitary=True)
    assert int(grid._auto_conjs.sum()) > 0


def test_specify_auto_symmetries_with_include_antiunitary_yields_smaller_or_equal_ibz():
    """Adding TR ops can only shrink the IBZ (or leave it unchanged)."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    g_spatial = bz.KGrid(nk=(nx, ny, nz), symmetries=[bz.KnownSymmetries.AUTO])
    g_spatial.specify_auto_symmetries(H, include_antiunitary=False)
    g_full = bz.KGrid(nk=(nx, ny, nz), symmetries=[bz.KnownSymmetries.AUTO])
    g_full.specify_auto_symmetries(H, include_antiunitary=True)
    assert g_full.nk_irr <= g_spatial.nk_irr


def test_specify_auto_symmetries_raises_when_kgrid_is_not_in_auto_mode():
    """specify_auto_symmetries on a plain-symmetry (non-auto) KGrid raises rather than clobbering it."""
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.three_dimensional_cubic_symmetries())
    H = _make_small_real_cubic_h(4, 4, 4, 1)
    with pytest.raises(RuntimeError, match="auto mode"):
        grid.specify_auto_symmetries(H)


def test_specify_auto_symmetries_raises_on_grid_shape_mismatch():
    """specify_auto_symmetries raises on a grid-shape mismatch."""
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=[bz.KnownSymmetries.AUTO])
    H_wrong = _make_small_real_cubic_h(4, 4, 2, 1)  # nz=2 instead of 4
    with pytest.raises(ValueError, match="k-grid shape"):
        grid.specify_auto_symmetries(H_wrong)


def test_specify_auto_symmetries_raises_on_wrong_ndim():
    """specify_auto_symmetries raises on the wrong number of dimensions."""
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=[bz.KnownSymmetries.AUTO])
    bad = np.zeros((4, 4, 4), dtype=complex)  # missing orbital axes
    with pytest.raises(ValueError, match="must have shape"):
        grid.specify_auto_symmetries(bad)


def test_specify_auto_symmetries_raises_on_non_square_orbital_axes():
    """specify_auto_symmetries raises on non-square orbital axes."""
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=[bz.KnownSymmetries.AUTO])
    bad = np.zeros((4, 4, 4, 2, 3), dtype=complex)  # mismatched orbital dims
    with pytest.raises(ValueError, match="must have shape"):
        grid.specify_auto_symmetries(bad)


def test_specify_auto_symmetries_accepts_non_contiguous_input():
    """Non-contiguous or non-complex128 input is accepted (the routine casts to complex128 explicitly)."""
    H = _make_small_real_cubic_h(4, 4, 4, 1).astype(np.complex64)
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=[bz.KnownSymmetries.AUTO])
    # Should not raise
    grid.specify_auto_symmetries(H)
    assert grid.is_auto is True


def test_plain_symmetry_kgrid_two_dimensional_square_unchanged():
    """The plain-symmetry path keeps its IBZ: 4x4x1 square symmetry gives Gamma, X, M and one interior point."""
    grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    assert grid.nk_irr <= 16
    assert grid.irrk_count.sum() == 16
    assert grid.fbz2irrk.shape == (4, 4, 1)


def test_plain_symmetry_kgrid_three_dimensional_cubic_unchanged():
    """A plain-symmetry 3D cubic KGrid behaves unchanged."""
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.three_dimensional_cubic_symmetries())
    assert grid.nk_irr <= 64
    assert grid.irrk_count.sum() == 64


def test_specify_auto_symmetries_finds_at_least_explicit_cubic_symmetries_for_cubic_h():
    """For a real cubic H the auto IBZ is no larger than the explicit three_dimensional_cubic IBZ."""
    H = _make_small_real_cubic_h(4, 4, 4, 1)
    g_auto = bz.KGrid(nk=(4, 4, 4), symmetries=[bz.KnownSymmetries.AUTO])
    g_auto.specify_auto_symmetries(H)
    g_explicit = bz.KGrid(nk=(4, 4, 4), symmetries=bz.three_dimensional_cubic_symmetries())
    assert g_auto.nk_irr <= g_explicit.nk_irr
    # auto refines the cubic orbits: fbz2irrk_auto must be constant on each explicit-group orbit
    fbz_auto = g_auto.fbz2irrk.ravel()
    fbz_explicit = g_explicit.fbz2irrk.ravel()
    for explicit_rep in np.unique(fbz_explicit):
        members = np.where(fbz_explicit == explicit_rep)[0]
        assert len(np.unique(fbz_auto[members])) == 1

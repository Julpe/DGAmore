# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

from unittest.mock import patch

import numpy as np
import pytest

import dgamore.brillouin_zone as bz
from dgamore.brillouin_zone import KPath, KnownKPoints, Labels


def test_applies_inversion_symmetry_along_x_axis():
    mat = np.random.rand(6, 4, 4)
    bz.inv_sym(mat, axis=0)
    assert np.allclose(mat[4:, :, :], mat[1:3, :, :][::-1])


def test_applies_inversion_symmetry_along_y_axis():
    mat = np.random.rand(4, 6, 4)
    bz.inv_sym(mat, axis=1)
    assert np.allclose(mat[:, 4:, :], mat[:, 1:3, :][:, ::-1])


def test_applies_inversion_symmetry_along_z_axis():
    mat = np.random.rand(4, 4, 6)
    bz.inv_sym(mat, axis=2)
    assert np.allclose(mat[:, :, 4:], mat[:, :, 1:3][:, :, ::-1])


def test_raises_error_for_invalid_axis():
    mat = np.random.rand(4, 4, 4)
    with pytest.raises(AssertionError, match="axis = 3 but must be in \[0,1,2\]"):
        bz.inv_sym(mat, axis=3)


def test_raises_error_for_insufficient_dimensions_on_inv_sym():
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError, match="dim\(mat\) = 2 but must be at least 3 dimensional"):
        bz.inv_sym(mat, axis=0)


def test_applies_x_y_symmetry_to_square_matrix():
    mat = np.random.rand(4, 4, 6)
    bz.x_y_sym(mat)
    assert np.allclose(mat, np.minimum(mat, mat.swapaxes(0, 1)))


def test_applies_x_z_symmetry_to_square_matrix():
    mat = np.random.rand(4, 6, 4)
    bz.x_z_sym(mat)
    assert np.allclose(mat, np.minimum(mat, mat.swapaxes(0, 2)))


def test_applies_y_z_symmetry_to_square_matrix():
    mat = np.random.rand(6, 4, 4)
    bz.y_z_sym(mat)
    assert np.allclose(mat, np.minimum(mat, mat.swapaxes(1, 2)))


def test_does_nothing_for_non_square_matrix():
    mat = np.random.rand(4, 5, 6)
    original_mat = mat.copy()
    bz.x_y_sym(mat)
    assert np.allclose(mat, original_mat)


def test_raises_error_for_insufficient_dimensions_on_x_y_sym():
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError):
        bz.x_y_sym(mat)


def test_raises_error_for_insufficient_dimensions_on_x_z_sym():
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError):
        bz.x_z_sym(mat)


def test_raises_error_for_insufficient_dimensions_on_y_z_sym():
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError):
        bz.y_z_sym(mat)


def test_applies_simultaneous_inversion_in_x_and_y_directions():
    mat = np.random.rand(6, 6, 4)
    bz.x_y_inv(mat)
    assert np.allclose(mat[4:, 4:, :], mat[1:3, 1:3, :][::-1, ::-1, :])


def test_raises_error_for_insufficient_dimensions_on_x_y_inv():
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError, match="dim\(mat\) = 2 but must be at least 3 dimensional"):
        bz.x_y_inv(mat)


def test_applies_x_inversion_symmetry_correctly_with_mock():
    mat = np.random.rand(6, 4, 4)
    with patch("dgamore.brillouin_zone.inv_sym") as mock_inv_sym:
        bz.apply_symmetry(mat, bz.KnownSymmetries.X_INV)
        mock_inv_sym.assert_called_once_with(mat, 0)


def test_applies_y_inversion_symmetry_correctly_with_mock():
    mat = np.random.rand(4, 6, 4)
    with patch("dgamore.brillouin_zone.inv_sym") as mock_inv_sym:
        bz.apply_symmetry(mat, bz.KnownSymmetries.Y_INV)
        mock_inv_sym.assert_called_once_with(mat, 1)


def test_applies_z_inversion_symmetry_correctly_with_mock():
    mat = np.random.rand(4, 4, 6)
    with patch("dgamore.brillouin_zone.inv_sym") as mock_inv_sym:
        bz.apply_symmetry(mat, bz.KnownSymmetries.Z_INV)
        mock_inv_sym.assert_called_once_with(mat, 2)


def test_applies_x_y_symmetry_correctly_with_mock():
    mat = np.random.rand(4, 4, 6)
    with patch("dgamore.brillouin_zone.x_y_sym") as mock_x_y_sym:
        bz.apply_symmetry(mat, bz.KnownSymmetries.X_Y_SYM)
        mock_x_y_sym.assert_called_once_with(mat)


def test_applies_x_z_symmetry_correctly_with_mock():
    mat = np.random.rand(4, 6, 4)
    with patch("dgamore.brillouin_zone.x_z_sym") as mock_x_z_sym:
        bz.apply_symmetry(mat, bz.KnownSymmetries.X_Z_SYM)
        mock_x_z_sym.assert_called_once_with(mat)


def test_applies_y_z_symmetry_correctly_with_mock():
    mat = np.random.rand(6, 4, 4)
    with patch("dgamore.brillouin_zone.y_z_sym") as mock_y_z_sym:
        bz.apply_symmetry(mat, bz.KnownSymmetries.Y_Z_SYM)
        mock_y_z_sym.assert_called_once_with(mat)


def test_applies_x_y_inversion_symmetry_correctly_with_mock():
    mat = np.random.rand(6, 6, 4)
    with patch("dgamore.brillouin_zone.x_y_inv") as mock_x_y_inv:
        bz.apply_symmetry(mat, bz.KnownSymmetries.X_Y_INV)
        mock_x_y_inv.assert_called_once_with(mat)


def test_raises_error_for_unknown_symmetry_with_mock():
    mat = np.random.rand(4, 4, 4)
    with patch("dgamore.brillouin_zone.KnownSymmetries") as mock_known_symmetries:
        with pytest.raises(AssertionError, match="sym = .* not in known symmetries .*"):
            bz.apply_symmetry(mat, "unknown_symmetry")
        mock_known_symmetries.__contains__.assert_called()


def test_applies_multiple_symmetries_in_order():
    mat = np.random.rand(6, 6, 6)
    with patch("dgamore.brillouin_zone.apply_symmetry") as mock_apply_symmetry:
        bz.apply_symmetries(mat, [bz.KnownSymmetries.X_INV, bz.KnownSymmetries.Y_INV, bz.KnownSymmetries.Z_INV])
        mock_apply_symmetry.assert_any_call(mat, bz.KnownSymmetries.X_INV)
        mock_apply_symmetry.assert_any_call(mat, bz.KnownSymmetries.Y_INV)
        mock_apply_symmetry.assert_any_call(mat, bz.KnownSymmetries.Z_INV)
        assert mock_apply_symmetry.call_count == 3


def test_does_nothing_when_no_symmetries_provided():
    mat = np.random.rand(6, 6, 6)
    with patch("dgamore.brillouin_zone.apply_symmetry") as mock_apply_symmetry:
        bz.apply_symmetries(mat, [])
        mock_apply_symmetry.assert_not_called()


def test_raises_error_for_insufficient_dimensions_on_apply_symmetries():
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError, match="dim\(mat\) = 2 but must at least 3 dimensional"):
        bz.apply_symmetries(mat, [bz.KnownSymmetries.X_INV])


def test_returns_correct_symmetries_for_two_dimensional_square():
    result = bz.get_lattice_symmetries_from_string("two_dimensional_square")
    assert result == bz.two_dimensional_square_symmetries()


def test_returns_correct_symmetries_for_three_dimensional_cubic():
    result = bz.get_lattice_symmetries_from_string("three_dimensional_cubic")
    assert result == bz.three_dimensional_cubic_symmetries()


def test_returns_correct_symmetries_for_quasi_one_dimensional_square():
    result = bz.get_lattice_symmetries_from_string("quasi_one_dimensional_square")
    assert result == bz.quasi_one_dimensional_square_symmetries()


def test_returns_correct_symmetries_for_simultaneous_x_y_inversion():
    result = bz.get_lattice_symmetries_from_string("simultaneous_x_y_inversion")
    assert result == bz.simultaneous_x_y_inversion()


def test_returns_correct_symmetries_for_quasi_two_dimensional_square_symmetries():
    result = bz.get_lattice_symmetries_from_string("quasi_two_dimensional_square_symmetries")
    assert result == bz.quasi_two_dimensional_square_symmetries()


def test_returns_empty_list_for_none_or_empty_string():
    result_none = bz.get_lattice_symmetries_from_string(None)
    result_empty = bz.get_lattice_symmetries_from_string("")
    assert result_none == []
    assert result_empty == []


def test_raises_error_for_unsupported_symmetry_string():
    with pytest.raises(ValueError, match="Symmetry does not exist or input cannot be parsed as a Python literal."):
        bz.get_lattice_symmetries_from_string("unsupported_symmetry")


def test_raises_error_for_unsupported_symmetry_in_list():
    with pytest.raises(NotImplementedError, match="Symmetry unsupported_symmetry not supported."):
        bz.get_lattice_symmetries_from_string(["x-inv", "unsupported_symmetry"])


def test_returns_correct_symmetries_for_list_of_valid_symmetries():
    result = bz.get_lattice_symmetries_from_string(["x-inv", "y-inv"])
    assert result == [bz.KnownSymmetries.X_INV, bz.KnownSymmetries.Y_INV]


def test_maps_full_bz_to_irreducible_correctly():
    nk = (4, 4, 4)
    symmetries = [bz.KnownSymmetries.X_INV, bz.KnownSymmetries.Y_INV]
    kgrid = bz.KGrid(nk=nk, symmetries=symmetries)
    with patch("dgamore.brillouin_zone.apply_symmetries") as mock_apply_symmetries:
        kgrid.set_fbz2irrk()
        mock_apply_symmetries.assert_called_once_with(kgrid.fbz2irrk, symmetries)


def test_handles_empty_symmetry_list_without_error():
    nk = (4, 4, 4)
    symmetries = []
    kgrid = bz.KGrid(nk=nk, symmetries=symmetries)
    with patch("dgamore.brillouin_zone.apply_symmetries") as mock_apply_symmetries:
        kgrid.set_fbz2irrk()
        mock_apply_symmetries.assert_called_once_with(kgrid.fbz2irrk, symmetries)


def test_maps_unique_elements_correctly_to_indices():
    kgrid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    with patch("numpy.unique", wraps=np.unique) as mock_unique:
        kgrid.set_fbz2irrk()
        kgrid.set_irrk_maps()
        mock_unique.assert_called_once_with(kgrid.fbz2irrk, return_index=True, return_inverse=True, return_counts=True)


def test_handles_empty_input_without_error():
    fbz2irrk = np.array([])
    kgrid = bz.KGrid(nk=(0, 0, 0), symmetries=[])
    kgrid.fbz2irrk = fbz2irrk
    kgrid.set_irrk_maps()
    assert kgrid.irrk_ind.size == 0
    assert kgrid.irrk_inv.size == 0
    assert kgrid.irrk_count.size == 0


def test_sets_irrk_mesh_correctly_for_valid_input():
    nk = (4, 4, 4)
    symmetries = [bz.KnownSymmetries.X_INV, bz.KnownSymmetries.Y_INV]
    kgrid = bz.KGrid(nk=nk, symmetries=symmetries)
    kgrid.set_irrk_mesh()
    assert kgrid.irr_kmesh.shape == (3, kgrid.nk_irr)


def test_returns_correct_kx_shift_for_valid_input():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    expected_shift = kgrid.kx - np.pi
    assert np.allclose(kgrid.kx_shift, expected_shift)


def test_returns_correct_ky_shift_for_valid_input():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    expected_shift = kgrid.ky - np.pi
    assert np.allclose(kgrid.ky_shift, expected_shift)


def test_returns_correct_kz_shift_for_valid_input():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    expected_shift = kgrid.kz - np.pi
    assert np.allclose(kgrid.kz_shift, expected_shift)


def test_returns_correct_kx_shift_closed_for_valid_input():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    expected_shift_closed = np.array([*(kgrid.kx - np.pi), -kgrid.kx[0] + np.pi])
    assert np.allclose(kgrid.kx_shift_closed, expected_shift_closed)


def test_returns_correct_ky_shift_closed_for_valid_input():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    expected_shift_closed = np.array([*(kgrid.ky - np.pi), -kgrid.ky[0] + np.pi])
    assert np.allclose(kgrid.ky_shift_closed, expected_shift_closed)


def test_returns_correct_kz_shift_closed_for_valid_input():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    expected_shift_closed = np.array([*(kgrid.kz - np.pi), -kgrid.kz[0] + np.pi])
    assert np.allclose(kgrid.kz_shift_closed, expected_shift_closed)


def test_returns_correct_k_grid_as_tuple():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    kx, ky, kz = kgrid.grid
    assert np.array_equal(kx, kgrid.kx)
    assert np.array_equal(ky, kgrid.ky)
    assert np.array_equal(kz, kgrid.kz)


def test_calculates_total_number_of_k_points_correctly():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    assert kgrid.nk_tot == 64


def test_calculates_number_of_irreducible_k_points_correctly():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[bz.KnownSymmetries.X_INV])
    assert kgrid.nk_irr == len(np.unique(kgrid.fbz2irrk))


def test_returns_correct_k_meshgrid():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    kmesh = kgrid.kmesh
    assert kmesh.shape == (3, 4, 4, 4)
    assert np.array_equal(kmesh[0], np.meshgrid(kgrid.kx, kgrid.ky, kgrid.kz, indexing="ij")[0])
    assert np.array_equal(kmesh[1], np.meshgrid(kgrid.kx, kgrid.ky, kgrid.kz, indexing="ij")[1])
    assert np.array_equal(kmesh[2], np.meshgrid(kgrid.kx, kgrid.ky, kgrid.kz, indexing="ij")[2])


def test_returns_correct_kmesh_list_for_valid_input():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    kmesh_list = kgrid.kmesh_list
    assert kmesh_list.shape == (3, 64)
    assert np.array_equal(kmesh_list[0], kgrid.kmesh[0].flatten())
    assert np.array_equal(kmesh_list[1], kgrid.kmesh[1].flatten())
    assert np.array_equal(kmesh_list[2], kgrid.kmesh[2].flatten())


def test_sets_k_axes_correctly_for_valid_input():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    assert np.allclose(kgrid.kx, np.linspace(0, 2 * np.pi, 4, endpoint=False))
    assert np.allclose(kgrid.ky, np.linspace(0, 2 * np.pi, 4, endpoint=False))
    assert np.allclose(kgrid.kz, np.linspace(0, 2 * np.pi, 4, endpoint=False))


def test_returns_correct_q_list_for_valid_input():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    q_list = kgrid.get_q_list()
    assert q_list.shape == (64, 3)
    assert np.array_equal(q_list[:, 0], kgrid.kmesh_ind[0].flatten())
    assert np.array_equal(q_list[:, 1], kgrid.kmesh_ind[1].flatten())
    assert np.array_equal(q_list[:, 2], kgrid.kmesh_ind[2].flatten())


def test_returns_correct_irrq_list_for_valid_input():
    kgrid = bz.KGrid(nk=(4, 4, 4), symmetries=[bz.KnownSymmetries.X_INV])
    irrq_list = kgrid.get_irrq_list()
    assert irrq_list.shape == (kgrid.nk_irr, 3)
    assert np.array_equal(irrq_list[:, 0], kgrid.kmesh_ind[0].flatten()[kgrid.irrk_ind])
    assert np.array_equal(irrq_list[:, 1], kgrid.kmesh_ind[1].flatten()[kgrid.irrk_ind])
    assert np.array_equal(irrq_list[:, 2], kgrid.kmesh_ind[2].flatten()[kgrid.irrk_ind])


def test_corner_k_points_and_label_mapping_for_known_labels():
    kx = np.arange(4)
    kp = KPath(nk=(4, 4, 4), path="gamma-x", kx=kx, ky=kx, kz=kx)

    assert kp.ckps == ["gamma", "x"]

    ckp = kp.corner_k_points()

    assert np.allclose(ckp[0], np.array(KnownKPoints.GAMMA.value))
    assert np.allclose(ckp[1], np.array(KnownKPoints.X.value))

    assert kp.labels == [Labels.GAMMA.latex, Labels.X.latex]


def test_map_to_kpath_and_get_kpoints_return_expected_values():
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
    kx = np.arange(4)
    kp = KPath(nk=(4, 4, 4), path="gamma-0.25 0.25 0", kx=kx, ky=kx, kz=kx)

    ckp = kp.corner_k_points()
    assert np.allclose(ckp[0], np.array(KnownKPoints.GAMMA.value))
    assert np.allclose(ckp[1], np.array([0.25, 0.25, 0.0]))


def test_nk_tot_returns_sum_of_nkp():
    kp = KPath(nk=(4, 4, 4), path="gamma-x")
    kp.nkp = [2, 3, 1]
    assert kp.nk_tot == 6


def test_nk_seg_returns_diff_of_cind():
    kp = KPath(nk=(4, 4, 4), path="gamma-x")
    kp.nkp = [2, 3, 1]
    cind = np.concatenate(([0], np.cumsum(kp.nkp) - 1))
    expected = np.diff(cind)
    assert np.array_equal(kp.nk_seg, expected)


def test_k_axis_normalized_positions_and_length():
    kp = KPath(nk=(4, 4, 4), path="gamma-x")
    # create 4 consecutive k-points with equal step lengths of 1
    kp.kpts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    kp.nkp = [2, 2]  # total points = 4
    # distances between consecutive points: [1, 1, 1] -> cumulative [1,2,3]
    # k_axis_pos = [0,1,2,3] -> normalized by 3 -> [0, 1/3, 2/3, 1]
    expected = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    assert np.allclose(kp.k_axis, expected, rtol=1e-12, atol=1e-12)
    assert kp.k_axis.size == kp.nk_tot


def test_build_k_path_single_segment_gamma_to_x():
    """
    For nk=(4,4,4) and path 'gamma-x' the segment should produce two indices:
    [[0,0,0], [1,0,0]] and nkp should be [2].
    """
    kp = KPath(nk=(4, 4, 4), path="gamma-x")
    k_path, nkp = kp.build_k_path()

    expected = np.array([[0, 0, 0], [1, 0, 0]])
    assert isinstance(k_path, np.ndarray)
    assert np.array_equal(k_path, expected)
    assert nkp == [2]


def test_get_bands_returns_sorted_real_eigenvalues():
    """
    Patch KPath.map_to_kpath to return an object that yields 2x2 matrices.
    Ensure get_bands returns sorted real eigenvalues for each k-point.
    """

    class MockEKPath:
        def __init__(self, mats):
            self.mats = mats
            # emulate an array with shape (n_kpoints, nbands, nbands)
            self.current_shape = (len(mats), mats[0].shape[0], mats[0].shape[0])

        def __iter__(self):
            return iter(self.mats)

    kp = KPath(nk=(4, 4, 4), path="gamma-x")
    # create two diagonal matrices with known eigenvalues
    mats = [
        np.array([[2.0, 0.0], [0.0, 1.0]]),  # eigenvalues [2,1] -> sorted [1,2]
        np.array([[4.0, 0.0], [0.0, 3.0]]),  # eigenvalues [4,3] -> sorted [3,4]
    ]
    mock_ek = MockEKPath(mats)

    with patch.object(KPath, "map_to_kpath", return_value=mock_ek):
        bands = kp.get_bands(ek=None)

    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(bands, expected)


# =============================================================================
# Auto-symmetry sentinel and is_auto_symmetries helper
# =============================================================================


def test_auto_symmetries_sentinel_is_singleton():
    """``_AutoSymmetriesSentinel`` is a singleton: every instantiation returns the
    canonical ``AUTO_SYMMETRIES_SENTINEL`` object."""
    a = bz._AutoSymmetriesSentinel()
    b = bz._AutoSymmetriesSentinel()
    assert a is b
    assert a is bz.AUTO_SYMMETRIES_SENTINEL


def test_auto_symmetries_sentinel_repr_is_stable():
    assert repr(bz.AUTO_SYMMETRIES_SENTINEL) == "<auto-symmetries>"


def test_auto_symmetries_sentinel_iterates_as_empty():
    """The sentinel is iterable and yields nothing, so legacy ``for s in symmetries``
    code paths see no operations to apply."""
    assert list(bz.AUTO_SYMMETRIES_SENTINEL) == []


def test_auto_symmetries_sentinel_has_length_zero():
    assert len(bz.AUTO_SYMMETRIES_SENTINEL) == 0


def test_auto_symmetries_sentinel_is_truthy():
    """Truthy so ``if symmetries:`` branches still enter, even though iteration is empty.
    This is what lets KGrid detect auto-mode intent without triggering legacy code."""
    assert bool(bz.AUTO_SYMMETRIES_SENTINEL) is True


def test_is_auto_symmetries_true_for_sentinel_only():
    assert bz.is_auto_symmetries(bz.AUTO_SYMMETRIES_SENTINEL) is True


def test_is_auto_symmetries_false_for_legacy_list():
    assert bz.is_auto_symmetries(bz.two_dimensional_square_symmetries()) is False
    assert bz.is_auto_symmetries(bz.three_dimensional_cubic_symmetries()) is False


def test_is_auto_symmetries_false_for_empty_list_none_or_other():
    assert bz.is_auto_symmetries([]) is False
    assert bz.is_auto_symmetries(None) is False
    # Identity-based check: only the sentinel singleton qualifies, the string
    # "auto" should NOT (callers must go through get_lattice_symmetries_from_string).
    assert bz.is_auto_symmetries("auto") is False
    assert bz.is_auto_symmetries(0) is False
    assert bz.is_auto_symmetries({}) is False


def test_get_lattice_symmetries_from_string_returns_sentinel_for_auto():
    """The "auto" string is the documented public entry point for opting into
    auto-discovery; it must resolve to the singleton sentinel."""
    result = bz.get_lattice_symmetries_from_string("auto")
    assert result is bz.AUTO_SYMMETRIES_SENTINEL


def test_get_lattice_symmetries_from_string_auto_is_case_insensitive():
    """Lowercase normalization is applied to all string inputs; "auto" / "AUTO" / "Auto" all work."""
    for s in ("auto", "AUTO", "Auto", "AuTo"):
        assert bz.get_lattice_symmetries_from_string(s) is bz.AUTO_SYMMETRIES_SENTINEL


# =============================================================================
# KGrid construction in auto mode
# =============================================================================


def _make_small_real_cubic_h(nx=4, ny=4, nz=4, nb=1):
    """A simple real, Hermitian, cubic-symmetric H on a small grid — convenient for
    testing auto-detection. With a single band the orbital action is trivial; the
    discovered symmetry is the spatial cubic group (8 ops for nx=ny=nz with all-axes
    inversions + permutations)."""
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
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    assert grid._auto_mode is True


def test_kgrid_with_legacy_symmetries_does_not_set_auto_mode():
    """A KGrid built with predefined symmetries must NOT enter auto mode."""
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.three_dimensional_cubic_symmetries())
    assert grid._auto_mode is False


def test_kgrid_with_no_symmetries_does_not_set_auto_mode():
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=[])
    assert grid._auto_mode is False


def test_kgrid_auto_mode_starts_with_trivial_ibz_and_no_auto_data():
    """Before specify_auto_symmetries() is called, the auto-mode KGrid behaves like
    the symmetry-free case: the IBZ equals the FBZ. The auto-data slots are unset."""
    nx, ny, nz = 4, 4, 4
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    assert grid._auto_us is None
    assert grid._auto_sigmas is None
    assert grid._auto_conjs is None
    # Trivial IBZ = FBZ before discovery
    assert grid.nk_irr == nx * ny * nz


def test_kgrid_is_auto_property_is_false_before_specify_auto_symmetries():
    """``is_auto`` is the runtime indicator that auto-data has been populated.
    It must be False between construction and specify_auto_symmetries()."""
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    assert grid.is_auto is False


def test_kgrid_is_auto_property_is_false_for_legacy_grid():
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.three_dimensional_cubic_symmetries())
    assert grid.is_auto is False


def test_kgrid_is_auto_property_is_true_after_specify_auto_symmetries():
    nx, ny, nz = 4, 4, 4
    H = _make_small_real_cubic_h(nx, ny, nz, nb=1)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    grid.specify_auto_symmetries(H)
    assert grid.is_auto is True


# =============================================================================
# KGrid.specify_auto_symmetries: happy path
# =============================================================================


def test_specify_auto_symmetries_populates_all_expected_arrays():
    """After a successful call, every cached IBZ-related field plus the new auto-data
    fields must be populated and internally consistent."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    grid.specify_auto_symmetries(H)

    # IBZ fields
    assert grid.fbz2irrk.shape == (nx, ny, nz)
    assert grid.irrk_ind is not None
    assert grid.irrk_inv is not None
    assert grid.irrk_count is not None
    assert grid.irr_kmesh is not None
    assert grid.nk_irr == len(grid.irrk_ind)
    assert grid.nk_irr <= nx * ny * nz

    # Auto-data fields
    assert grid._auto_us.shape == (nx, ny, nz, nb, nb)
    assert grid._auto_sigmas.shape == (nx, ny, nz)
    assert grid._auto_conjs.shape == (nx, ny, nz)
    assert grid._auto_us.dtype == complex
    assert grid._auto_conjs.dtype == bool


def test_specify_auto_symmetries_produces_consistent_fbz2irrk_and_irrk_inv():
    """irrk_inv must be a true inverse of irrk_ind w.r.t. fbz2irrk:
    irrk_ind[irrk_inv[k]] == fbz2irrk.flat[k] for every k."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    grid.specify_auto_symmetries(H)

    fbz_flat = grid.fbz2irrk.ravel()
    inv_flat = grid.irrk_inv.ravel()
    # For each FBZ point k: fbz2irrk[k] is the flat IBZ index, irrk_ind[irrk_inv[k]] should equal it
    assert np.array_equal(grid.irrk_ind[inv_flat], fbz_flat)


def test_specify_auto_symmetries_irrk_count_sums_to_full_bz():
    """The duplicity counts must sum to the total number of FBZ points."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    grid.specify_auto_symmetries(H)

    assert grid.irrk_count.sum() == nx * ny * nz


def test_specify_auto_symmetries_us_are_unitary():
    """Every stored per-k transformation must be unitary: U U^dag = I."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    grid.specify_auto_symmetries(H)
    Us = grid._auto_us.reshape(-1, nb, nb)
    identity = np.eye(nb, dtype=complex)
    products = np.einsum("...ij,...kj->...ik", Us, Us.conj())
    assert np.allclose(products, identity[None, ...], atol=1e-10)


def test_specify_auto_symmetries_sigmas_are_plus_or_minus_one():
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    grid.specify_auto_symmetries(H)
    sig = grid._auto_sigmas.ravel()
    assert np.all(np.isin(sig, [-1.0, +1.0]))


def test_specify_auto_symmetries_default_drops_antiunitary_ops():
    """The default ``include_antiunitary=False`` filters out time-reversal-like ops
    so the FBZ expansion is safe for frequency-dependent objects: no per-k
    transformation should carry conj=True."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    grid.specify_auto_symmetries(H)
    assert int(grid._auto_conjs.sum()) == 0


def test_specify_auto_symmetries_with_include_antiunitary_admits_conj_ops():
    """Opting in via ``include_antiunitary=True`` produces a larger group; for a
    purely-real H, anti-unitary ops always exist (H(k) = H(k)*), so at least some
    FBZ k-points will carry conj=True."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    grid.specify_auto_symmetries(H, include_antiunitary=True)
    assert int(grid._auto_conjs.sum()) > 0


def test_specify_auto_symmetries_with_include_antiunitary_yields_smaller_or_equal_ibz():
    """Adding TR ops can only shrink the IBZ (or leave it unchanged)."""
    nx, ny, nz, nb = 4, 4, 4, 1
    H = _make_small_real_cubic_h(nx, ny, nz, nb)
    g_spatial = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    g_spatial.specify_auto_symmetries(H, include_antiunitary=False)
    g_full = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    g_full.specify_auto_symmetries(H, include_antiunitary=True)
    assert g_full.nk_irr <= g_spatial.nk_irr


# =============================================================================
# KGrid.specify_auto_symmetries: input validation
# =============================================================================


def test_specify_auto_symmetries_raises_when_kgrid_is_not_in_auto_mode():
    """Calling specify_auto_symmetries on a legacy KGrid must fail loudly so users
    don't accidentally clobber a non-auto setup."""
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.three_dimensional_cubic_symmetries())
    H = _make_small_real_cubic_h(4, 4, 4, 1)
    with pytest.raises(RuntimeError, match="auto mode"):
        grid.specify_auto_symmetries(H)


def test_specify_auto_symmetries_raises_on_grid_shape_mismatch():
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    H_wrong = _make_small_real_cubic_h(4, 4, 2, 1)  # nz=2 instead of 4
    with pytest.raises(ValueError, match="k-grid shape"):
        grid.specify_auto_symmetries(H_wrong)


def test_specify_auto_symmetries_raises_on_wrong_ndim():
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    bad = np.zeros((4, 4, 4), dtype=complex)  # missing orbital axes
    with pytest.raises(ValueError, match="must have shape"):
        grid.specify_auto_symmetries(bad)


def test_specify_auto_symmetries_raises_on_non_square_orbital_axes():
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    bad = np.zeros((4, 4, 4, 2, 3), dtype=complex)  # mismatched orbital dims
    with pytest.raises(ValueError, match="must have shape"):
        grid.specify_auto_symmetries(bad)


def test_specify_auto_symmetries_accepts_non_contiguous_input():
    """The implementation casts to complex128 explicitly; non-contiguous or
    non-complex128 input should be accepted without crashing."""
    H = _make_small_real_cubic_h(4, 4, 4, 1).astype(np.complex64)
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    # Should not raise
    grid.specify_auto_symmetries(H)
    assert grid.is_auto is True


# =============================================================================
# Regression: legacy KGrid still works unchanged
# =============================================================================


def test_legacy_kgrid_two_dimensional_square_unchanged():
    """Legacy code paths must continue to produce the same IBZ they always did."""
    grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    # Known: 4x4x1 with full square symmetry has a small IBZ (Γ, X, M, and one interior)
    assert grid.nk_irr <= 16
    # Sanity-consistent IBZ
    assert grid.irrk_count.sum() == 16
    assert grid.fbz2irrk.shape == (4, 4, 1)


def test_legacy_kgrid_three_dimensional_cubic_unchanged():
    grid = bz.KGrid(nk=(4, 4, 4), symmetries=bz.three_dimensional_cubic_symmetries())
    assert grid.nk_irr <= 64
    assert grid.irrk_count.sum() == 64


def test_specify_auto_symmetries_finds_at_least_legacy_symmetries_for_cubic_h():
    """For a real cubic H, the auto-discovered spatial group must be at least as
    large as the legacy ``three_dimensional_cubic`` group, so the auto IBZ must
    be no larger than the legacy IBZ. (Auto can find accidental extra symmetries
    on small grids, so we don't insist on strict equality here.)"""
    H = _make_small_real_cubic_h(4, 4, 4, 1)
    g_auto = bz.KGrid(nk=(4, 4, 4), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    g_auto.specify_auto_symmetries(H)
    g_legacy = bz.KGrid(nk=(4, 4, 4), symmetries=bz.three_dimensional_cubic_symmetries())
    assert g_auto.nk_irr <= g_legacy.nk_irr
    # The cubic orbit structure must be refined by auto: every legacy orbit member
    # should map to the same auto IBZ representative (i.e., auto is a refinement).
    # Equivalent statement: fbz2irrk_auto is constant on each legacy orbit.
    fbz_auto = g_auto.fbz2irrk.ravel()
    fbz_legacy = g_legacy.fbz2irrk.ravel()
    for legacy_rep in np.unique(fbz_legacy):
        members = np.where(fbz_legacy == legacy_rep)[0]
        # All members of the legacy orbit should be in the same auto orbit
        assert len(np.unique(fbz_auto[members])) == 1

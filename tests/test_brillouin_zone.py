import numpy as np
import pytest
import moldga.brillouin_zone as bz
from unittest.mock import patch


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
    with pytest.raises(AssertionError, match="dim\(mat\) = 2 but must be at least 3 dimensional"):
        bz.x_y_sym(mat)


def test_raises_error_for_insufficient_dimensions_on_x_z_sym():
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError, match="dim\(mat\) = 2 but must be at least 3 dimensional"):
        bz.x_z_sym(mat)


def test_raises_error_for_insufficient_dimensions_on_y_z_sym():
    mat = np.random.rand(4, 4)
    with pytest.raises(AssertionError, match="dim\(mat\) = 2 but must be at least 3 dimensional"):
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
    with patch("moldga.brillouin_zone.inv_sym") as mock_inv_sym:
        bz.apply_symmetry(mat, bz.KnownSymmetries.X_INV)
        mock_inv_sym.assert_called_once_with(mat, 0)


def test_applies_y_inversion_symmetry_correctly_with_mock():
    mat = np.random.rand(4, 6, 4)
    with patch("moldga.brillouin_zone.inv_sym") as mock_inv_sym:
        bz.apply_symmetry(mat, bz.KnownSymmetries.Y_INV)
        mock_inv_sym.assert_called_once_with(mat, 1)


def test_applies_z_inversion_symmetry_correctly_with_mock():
    mat = np.random.rand(4, 4, 6)
    with patch("moldga.brillouin_zone.inv_sym") as mock_inv_sym:
        bz.apply_symmetry(mat, bz.KnownSymmetries.Z_INV)
        mock_inv_sym.assert_called_once_with(mat, 2)


def test_applies_x_y_symmetry_correctly_with_mock():
    mat = np.random.rand(4, 4, 6)
    with patch("moldga.brillouin_zone.x_y_sym") as mock_x_y_sym:
        bz.apply_symmetry(mat, bz.KnownSymmetries.X_Y_SYM)
        mock_x_y_sym.assert_called_once_with(mat)


def test_applies_x_z_symmetry_correctly_with_mock():
    mat = np.random.rand(4, 6, 4)
    with patch("moldga.brillouin_zone.x_z_sym") as mock_x_z_sym:
        bz.apply_symmetry(mat, bz.KnownSymmetries.X_Z_SYM)
        mock_x_z_sym.assert_called_once_with(mat)


def test_applies_y_z_symmetry_correctly_with_mock():
    mat = np.random.rand(6, 4, 4)
    with patch("moldga.brillouin_zone.y_z_sym") as mock_y_z_sym:
        bz.apply_symmetry(mat, bz.KnownSymmetries.Y_Z_SYM)
        mock_y_z_sym.assert_called_once_with(mat)


def test_applies_x_y_inversion_symmetry_correctly_with_mock():
    mat = np.random.rand(6, 6, 4)
    with patch("moldga.brillouin_zone.x_y_inv") as mock_x_y_inv:
        bz.apply_symmetry(mat, bz.KnownSymmetries.X_Y_INV)
        mock_x_y_inv.assert_called_once_with(mat)


def test_raises_error_for_unknown_symmetry_with_mock():
    mat = np.random.rand(4, 4, 4)
    with patch("moldga.brillouin_zone.KnownSymmetries") as mock_known_symmetries:
        with pytest.raises(AssertionError, match="sym = .* not in known symmetries .*"):
            bz.apply_symmetry(mat, "unknown_symmetry")
        mock_known_symmetries.__contains__.assert_called()


def test_applies_multiple_symmetries_in_order():
    mat = np.random.rand(6, 6, 6)
    with patch("moldga.brillouin_zone.apply_symmetry") as mock_apply_symmetry:
        bz.apply_symmetries(mat, [bz.KnownSymmetries.X_INV, bz.KnownSymmetries.Y_INV, bz.KnownSymmetries.Z_INV])
        mock_apply_symmetry.assert_any_call(mat, bz.KnownSymmetries.X_INV)
        mock_apply_symmetry.assert_any_call(mat, bz.KnownSymmetries.Y_INV)
        mock_apply_symmetry.assert_any_call(mat, bz.KnownSymmetries.Z_INV)
        assert mock_apply_symmetry.call_count == 3


def test_does_nothing_when_no_symmetries_provided():
    mat = np.random.rand(6, 6, 6)
    with patch("moldga.brillouin_zone.apply_symmetry") as mock_apply_symmetry:
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
    with pytest.raises(NotImplementedError, match="Symmetry unsupported_symmetry not supported."):
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
    with patch("moldga.brillouin_zone.apply_symmetries") as mock_apply_symmetries:
        kgrid.set_fbz2irrk()
        mock_apply_symmetries.assert_called_once_with(kgrid.fbz2irrk, symmetries)


def test_handles_empty_symmetry_list_without_error():
    nk = (4, 4, 4)
    symmetries = []
    kgrid = bz.KGrid(nk=nk, symmetries=symmetries)
    with patch("moldga.brillouin_zone.apply_symmetries") as mock_apply_symmetries:
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

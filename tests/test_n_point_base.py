# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
import scipy as sp
from scipy.signal import resample

import dgamore.symmetry_reduction as sr
from dgamore import brillouin_zone as bz
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.n_point_base import IHaveChannel, IHaveMat, IAmNonLocal, SpinChannel, FrequencyNotation, DTYPE


def test_initializes_with_correct_matrix_and_shape():
    """IHaveMat initializes with the given matrix and records its shape."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    assert np.allclose(obj.mat, mat, atol=1e-6)
    assert obj.original_shape == mat.shape


def test_updates_matrix_and_preserves_dtype():
    """Setting mat updates the array and coerces it to the storage dtype."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    assert obj.mat.dtype == np.complex64
    new_mat = np.array([[5, 6], [7, 8]], dtype=np.float64)
    obj.mat = new_mat
    assert obj.mat.dtype == np.complex64


def test_calculates_correct_memory_usage():
    """The reported memory usage matches the backing array."""
    mat = np.zeros((1000, 1000), dtype=np.complex64)
    obj = IHaveMat(mat)
    assert obj.memory_usage_in_gb == pytest.approx(mat.nbytes / (1024**3))


def test_multiplies_with_scalar_correctly():
    """Multiplication by a scalar scales the matrix."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    result = obj * 2
    assert np.allclose(result.mat, mat * 2, atol=1e-6)


def test_raises_error_when_multiplying_with_invalid_type():
    """Multiplication raises for an unsupported operand type."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    with pytest.raises(ValueError):
        obj * "invalid"


def test_performs_right_multiplication_with_scalar_correctly():
    """Right multiplication by a scalar scales the matrix."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    result = 2 * obj
    assert np.allclose(result.mat, mat * 2, atol=1e-6)


def test_negates_matrix_correctly():
    """Negation flips the sign of the matrix."""
    mat = np.array([[1, -2], [-3, 4]])
    obj = IHaveMat(mat)
    result = -obj
    assert np.allclose(result.mat, -mat, atol=1e-6)


def test_divides_by_scalar_correctly():
    """Division by a scalar scales the matrix."""
    mat = np.array([[2, 4], [6, 8]])
    obj = IHaveMat(mat)
    result = obj / 2
    assert np.allclose(result.mat, mat / 2, atol=1e-6)


def test_raises_error_when_dividing_by_invalid_type():
    """Division raises for an unsupported operand type."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    with pytest.raises(ValueError):
        obj / "invalid"


def test_reshapes_matrix_and_updates_original_shape():
    """Reshaping updates the matrix and tracks the original shape."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    obj.mat = obj.mat.reshape(4, 1)
    obj.update_original_shape()
    assert obj.original_shape == (4, 1)


def test_performs_einsum_contraction_correctly():
    """times performs an einsum contraction correctly."""
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    obj1 = IHaveMat(mat1)
    obj2 = IHaveMat(mat2)
    result = obj1.times("ij,jk->ik", obj2)
    assert np.allclose(result, np.dot(mat1, mat2), atol=1e-6)


def test_performs_einsum_contraction_with_multiple_matrices():
    """times contracts multiple operands via einsum."""
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    mat3 = np.array([[1, 0], [0, 1]])
    obj1 = IHaveMat(mat1)
    obj2 = IHaveMat(mat2)
    obj3 = IHaveMat(mat3)
    result = obj1.times("ij,jk,kl->il", obj2, obj3)
    assert np.allclose(result, np.dot(np.dot(mat1, mat2), mat3), atol=1e-6)


def test_raises_error_when_contraction_argument_is_invalid():
    """times raises for an invalid contraction argument."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    with pytest.raises(ValueError):
        obj.times("ij,jk->ik", "invalid_argument")


def test_handles_empty_matrices_in_contraction():
    """times handles empty matrices in a contraction."""
    mat1 = np.array([], dtype=np.float64).reshape(0, 0)
    mat2 = np.array([], dtype=np.float64).reshape(0, 0)
    obj1 = IHaveMat(mat1)
    obj2 = IHaveMat(mat2)
    result = obj1.times("ij,jk->ik", obj2)
    assert result.size == 0


def test_performs_einsum_contraction_with_numpy_array():
    """times contracts against a raw numpy array operand."""
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    obj = IHaveMat(mat1)
    result = obj.times("ij,jk->ik", mat2)
    assert np.allclose(result, np.dot(mat1, mat2), atol=1e-6)


def test_raises_error_when_contraction_string_is_invalid():
    """times raises for an invalid contraction string."""
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    obj = IHaveMat(mat1)
    with pytest.raises(ValueError):
        obj.times("invalid_contraction", mat2)


def test_retrieves_correct_value_for_valid_index():
    """Indexing returns the correct matrix element."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    assert obj[0, 1] == 2


def test_sets_value_correctly_for_valid_index():
    """Item assignment sets the correct matrix element."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    obj[0, 1] = 5
    assert obj[0, 1] == 5


def test_raises_error_for_invalid_index_retrieval():
    """Indexing raises for an invalid index."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    with pytest.raises(IndexError):
        _ = obj[2, 2]


def test_raises_error_for_invalid_index_assignment():
    """Item assignment raises for an invalid index."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    with pytest.raises(IndexError):
        obj[2, 2] = 5


def test_initializes_with_default_channel_and_frequency_notation():
    """IHaveChannel initializes with the default channel and frequency notation."""
    obj = IHaveChannel()
    assert obj.channel == SpinChannel.NONE
    assert obj.frequency_notation == FrequencyNotation.PH


def test_initializes_with_provided_channel_and_frequency_notation():
    """IHaveChannel initializes with a provided channel and frequency notation."""
    obj = IHaveChannel(channel=SpinChannel.DENS, frequency_notation=FrequencyNotation.PP)
    assert obj.channel == SpinChannel.DENS
    assert obj.frequency_notation == FrequencyNotation.PP


def test_updates_channel_to_valid_value():
    """set_channel updates the spin channel to a valid value."""
    obj = IHaveChannel()
    obj.channel = SpinChannel.MAGN
    assert obj.channel == SpinChannel.MAGN
    obj.set_channel(SpinChannel.DENS)
    assert obj.channel == SpinChannel.DENS


def test_raises_error_when_setting_invalid_channel():
    """set_channel raises for an invalid channel."""
    obj = IHaveChannel()
    with pytest.raises(ValueError):
        obj.channel = "invalid_channel"
    with pytest.raises(ValueError):
        obj.set_channel("invalid_channel")


def test_updates_frequency_notation_to_valid_value():
    """set_frequency_notation updates to a valid notation."""
    obj = IHaveChannel()
    obj.frequency_notation = FrequencyNotation.PP
    assert obj.frequency_notation == FrequencyNotation.PP
    obj.set_frequency_notation(FrequencyNotation.PH)
    assert obj.frequency_notation == FrequencyNotation.PH


def test_raises_error_when_setting_invalid_frequency_notation():
    """set_frequency_notation raises for an invalid notation."""
    obj = IHaveChannel()
    with pytest.raises(ValueError):
        obj.frequency_notation = "invalid_notation"
    with pytest.raises(ValueError):
        obj.set_frequency_notation("invalid_notation")


def test_initializes_with_correct_matrix_and_momentum_dimensions():
    """IAmNonLocal initializes with the given matrix and momentum dimensions."""
    mat = np.zeros((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    assert np.allclose(obj.mat, mat, atol=1e-6)
    assert obj.nq == nq
    assert obj.has_compressed_q_dimension is False


def test_initializes_with_compressed_q_dimension():
    """IAmNonLocal initializes with a compressed q dimension."""
    mat = np.zeros((64,))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    assert np.allclose(obj.mat, mat, atol=1e-6)
    assert obj.nq == nq
    assert obj.has_compressed_q_dimension is True


def test_shifts_momentum_by_zero_correctly():
    """shift_k_by_q with zero shift leaves the matrix unchanged."""
    mat = np.zeros((4, 4, 4))
    obj = IAmNonLocal(mat, (4, 4, 4))
    shifted = obj.shift_k_by_q((0, 0, 0))
    assert np.allclose(shifted.mat, mat, atol=1e-6)


def test_shifts_momentum_by_positive_values_correctly():
    """shift_k_by_q shifts the momentum by positive values correctly."""
    mat = np.arange(64).reshape((4, 4, 4))
    obj = IAmNonLocal(mat, (4, 4, 4))
    shifted = obj.shift_k_by_q((1, 1, 1))
    expected = np.roll(mat, shift=(-1, -1, -1), axis=(0, 1, 2))
    assert np.allclose(shifted.mat, expected, atol=1e-6)


def test_shifts_momentum_by_negative_values_correctly():
    """shift_k_by_q shifts the momentum by negative values correctly."""
    mat = np.arange(64).reshape((4, 4, 4))
    obj = IAmNonLocal(mat, (4, 4, 4))
    shifted = obj.shift_k_by_q((-1, -1, -1))
    expected = np.roll(mat, shift=(1, 1, 1), axis=(0, 1, 2))
    assert np.allclose(shifted.mat, expected, atol=1e-6)


def test_shifts_momentum_with_compressed_q_dimension_correctly():
    """shift_k_by_q shifts a compressed-q object correctly."""
    mat = np.zeros((64))
    obj = IAmNonLocal(mat, (4, 4, 4), has_compressed_q_dimension=True)
    shifted = obj.shift_k_by_q((1, 1, 1))
    assert shifted.current_shape == (64,)


def test_raises_error_when_shifting_with_invalid_q_length():
    """shift_k_by_q raises for an invalid q length."""
    mat = np.zeros((4, 4, 4))
    obj = IAmNonLocal(mat, (4, 4, 4))
    with pytest.raises(ValueError):
        obj.shift_k_by_q((1, 1))


def test_shifts_momentum_by_pi_correctly():
    """shift_k_by_pi shifts the momentum by pi correctly."""
    mat = np.arange(64).reshape((4, 4, 4))
    obj = IAmNonLocal(mat, (4, 4, 4))
    shifted = obj.shift_k_by_pi()
    expected = np.roll(mat, shift=(2, 2, 2), axis=(0, 1, 2))
    assert np.allclose(shifted.mat, expected, atol=1e-6)


def test_shifts_momentum_by_pi_with_compressed_q_dimension():
    """shift_k_by_pi shifts a compressed-q object by pi."""
    mat = np.arange(64)
    obj = IAmNonLocal(mat, (4, 4, 4), has_compressed_q_dimension=True)
    shifted = obj.shift_k_by_pi()
    assert shifted.has_compressed_q_dimension is True
    assert shifted.mat.shape == mat.shape


def test_raises_error_when_shifting_by_pi_with_invalid_matrix_shape():
    """shift_k_by_pi raises for an invalid matrix shape."""
    mat = np.zeros((4, 4))
    obj = IAmNonLocal(mat, (4, 4, 4))
    with pytest.raises(ValueError):
        obj.shift_k_by_pi()


def test_compresses_q_dimension_correctly():
    """compress_q_dimension folds the momentum axes correctly."""
    mat = np.zeros((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    obj.compress_q_dimension()
    assert obj.mat.shape == (64,)
    assert obj.has_compressed_q_dimension is True


def test_does_not_compress_already_compressed_q_dimension():
    """compress_q_dimension is a no-op when already compressed."""
    mat = np.zeros((64,))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    obj.compress_q_dimension()
    assert obj.mat.shape == (64,)
    assert obj.has_compressed_q_dimension is True


def test_compresses_q_dimension_with_additional_dimensions():
    """compress_q_dimension preserves trailing dimensions."""
    mat = np.zeros((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    obj.compress_q_dimension()
    assert obj.mat.shape == (64, 2)
    assert obj.has_compressed_q_dimension is True


def test_decompresses_q_dimension_correctly():
    """decompress_q_dimension unfolds the momentum axis correctly."""
    mat = np.zeros((64,))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    obj.decompress_q_dimension()
    assert obj.mat.shape == (4, 4, 4)
    assert obj.has_compressed_q_dimension is False


def test_does_not_decompress_if_already_decompressed():
    """decompress_q_dimension is a no-op when already decompressed."""
    mat = np.zeros((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    obj.decompress_q_dimension()
    assert obj.mat.shape == (4, 4, 4)
    assert obj.has_compressed_q_dimension is False


def test_decompresses_q_dimension_with_additional_dimensions():
    """decompress_q_dimension preserves trailing dimensions."""
    mat = np.zeros((64, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    obj.decompress_q_dimension()
    assert obj.mat.shape == (4, 4, 4, 2)
    assert obj.has_compressed_q_dimension is False


def test_reduces_q_dimension_to_specified_momenta():
    """reduce_q keeps only the specified momenta."""
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    q_list = np.array([[1, 1, 1], [2, 2, 2]])
    reduced = obj.reduce_q(q_list)
    assert reduced.mat.shape == (2,)
    assert reduced.has_compressed_q_dimension is True


def test_reduces_q_dimension_with_compressed_input():
    """reduce_q works on a compressed-q input."""
    mat = np.arange(64)
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    q_list = np.array([[0, 0, 0], [3, 3, 3]])
    reduced = obj.reduce_q(q_list)
    assert reduced.mat.shape == (2,)
    assert reduced.has_compressed_q_dimension is True


def test_reduce_q_raises_error_when_q_list_has_invalid_shape():
    """reduce_q raises when the q-list has an invalid shape."""
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    q_list = np.array([[0, 0], [3, 3]])
    with pytest.raises(ValueError):
        obj.reduce_q(q_list)


def test_reduces_q_dimension_to_specified_momenta_and_values():
    """reduce_q keeps the specified momenta and their values."""
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    q_list = np.array([[1, 1, 1], [2, 2, 2]])
    reduced = obj.reduce_q(q_list)
    expected_values = mat[1, 1, 1], mat[2, 2, 2]
    assert reduced.mat.shape == (2,)
    assert np.allclose(reduced.mat, expected_values, atol=1e-6)
    assert reduced.has_compressed_q_dimension is True


def test_finds_correct_matrix_element_for_given_momentum():
    """find_q returns the matrix element for a given momentum."""
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.find_q((1, 1, 1))
    assert result.mat.shape == (1,)
    assert result.mat[0] == mat[1, 1, 1]
    assert result.nq == (1, 1, 1)


def test_finds_matrix_element_for_valid_momentum():
    """find_q returns the matrix element for a valid momentum."""
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.find_q((2, 2, 2))
    assert result.mat.shape == (1,)
    assert result.mat[0] == mat[2, 2, 2]
    assert result.nq == (1, 1, 1)


def test_raises_error_for_invalid_momentum_shape():
    """find_q raises for an invalid momentum shape."""
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    with pytest.raises(ValueError):
        obj.find_q((1, 1))


def test_raises_error_for_out_of_bounds_momentum():
    """find_q raises for an out-of-bounds momentum."""
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    with pytest.raises(ValueError):
        obj.find_q((5, 5, 5))


def test_maps_to_full_bz_correctly_with_valid_inverse_map():
    """map_to_full_bz expands the IBZ using a valid inverse map."""
    mat = np.arange(64)
    nq = (4, 4, 4)
    np.array([0, 1, 2, 3, 4, 5, 6, 7])
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    grid = bz.KGrid(nk=(2, 2, 2), symmetries=[])
    obj.map_to_full_bz(grid, nq=(2, 2, 2))
    assert obj.mat.shape == (8,)
    assert obj.nq == (2, 2, 2)


def test_raises_error_when_mapping_to_full_bz_without_compressed_q_dimension():
    """map_to_full_bz raises without a compressed q dimension."""
    mat = np.zeros((4, 4, 4))
    nq = (4, 4, 4)
    inverse_map = np.array([0, 1, 2, 3])
    obj = IAmNonLocal(mat, nq)
    with pytest.raises(ValueError):
        obj.map_to_full_bz(inverse_map)


def test_updates_nq_correctly_when_provided():
    """map_to_full_bz updates nq when provided."""
    mat = np.arange(64)
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    grid = bz.KGrid(nk=(2, 2, 2), symmetries=[])
    obj.map_to_full_bz(grid, nq=(2, 2, 2))
    assert obj.nq == (2, 2, 2)


def test_retains_original_nq_when_not_provided():
    """map_to_full_bz retains the original nq when none is provided."""
    mat = np.arange(64)
    nq = (4, 4, 4)
    grid = bz.KGrid(nk=nq, symmetries=[])
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    obj.map_to_full_bz(grid)
    assert obj.nq == (4, 4, 4)


def test_performs_fft_correctly_on_decompressed_matrix():
    """fft transforms a decompressed matrix correctly."""
    mat = np.random.random((4, 4, 4)) + 1j * np.random.random((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.fft()
    expected = sp.fft.fftn(mat, axes=(0, 1, 2))
    assert np.allclose(result.mat, expected, atol=1e-6)
    assert result.has_compressed_q_dimension is False


def test_performs_fft_correctly_on_compressed_matrix():
    """fft transforms a compressed matrix correctly."""
    mat = np.random.random((64,)) + 1j * np.random.random((64,))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    result = obj.fft()
    decompressed_mat = mat.reshape(nq)
    expected = sp.fft.fftn(decompressed_mat, axes=(0, 1, 2)).reshape(64)
    assert np.allclose(result.mat, expected, atol=1e-6)
    assert result.has_compressed_q_dimension is True


def test_retains_original_shape_after_fft():
    """fft retains the original shape bookkeeping."""
    mat = np.random.random((4, 4, 4)) + 1j * np.random.random((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.fft()
    assert result.original_shape == (4, 4, 4)


def test_performs_ifft_correctly_on_decompressed_matrix():
    """ifft transforms a decompressed matrix correctly."""
    mat = np.random.random((4, 4, 4)) + 1j * np.random.random((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.ifft()
    expected = sp.fft.ifftn(mat, axes=(0, 1, 2))
    assert np.allclose(result.mat, expected, atol=1e-6)
    assert result.has_compressed_q_dimension is False


def test_performs_ifft_correctly_on_compressed_matrix():
    """ifft transforms a compressed matrix correctly."""
    mat = np.random.random((64,)) + 1j * np.random.random((64,))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    result = obj.ifft()
    decompressed_mat = mat.reshape(nq)
    expected = sp.fft.ifftn(decompressed_mat, axes=(0, 1, 2)).reshape(64)
    assert np.allclose(result.mat, expected, atol=1e-6)
    assert result.has_compressed_q_dimension is True


def test_retains_original_shape_after_ifft():
    """ifft retains the original shape bookkeeping."""
    mat = np.random.random((4, 4, 4)) + 1j * np.random.random((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.ifft()
    assert result.original_shape == (4, 4, 4)


def test_flips_momentum_axis_correctly_for_decompressed_matrix():
    """flip_momentum_axis flips a decompressed matrix correctly."""
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    flipped = obj.flip_momentum_axis()
    expected = np.roll(np.flip(mat, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
    assert np.allclose(flipped.mat, expected, atol=1e-6)
    assert flipped.has_compressed_q_dimension is False


def test_flips_momentum_axis_correctly_for_compressed_matrix():
    """flip_momentum_axis flips a compressed matrix correctly."""
    mat = np.arange(64)
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    flipped = obj.flip_momentum_axis()
    decompressed_mat = mat.reshape(nq)
    expected = np.roll(np.flip(decompressed_mat, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2)).reshape(64)
    assert np.allclose(flipped.mat, expected, atol=1e-6)
    assert flipped.has_compressed_q_dimension is True


def test_retains_original_shape_after_flipping_momentum_axis():
    """flip_momentum_axis retains the original shape bookkeeping."""
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    flipped = obj.flip_momentum_axis()
    assert flipped.original_shape == (4, 4, 4)


def test_aligns_q_dimensions_when_both_are_decompressed():
    """_align_q_dimensions_for_operation aligns two decompressed objects."""
    mat1 = np.zeros((4, 4, 4))
    mat2 = np.zeros((4, 4, 4))
    obj1 = IAmNonLocal(mat1, (4, 4, 4))
    obj2 = IAmNonLocal(mat2, (4, 4, 4))
    aligned = obj1._align_q_dimensions_for_operations(obj2)
    assert not obj1.has_compressed_q_dimension
    assert not aligned.has_compressed_q_dimension


def test_aligns_q_dimensions_when_both_are_compressed():
    """_align_q_dimensions_for_operation aligns two compressed objects."""
    mat1 = np.zeros((64,))
    mat2 = np.zeros((64,))
    obj1 = IAmNonLocal(mat1, (4, 4, 4), has_compressed_q_dimension=True)
    obj2 = IAmNonLocal(mat2, (4, 4, 4), has_compressed_q_dimension=True)
    aligned = obj1._align_q_dimensions_for_operations(obj2)
    assert obj1.has_compressed_q_dimension
    assert aligned.has_compressed_q_dimension


def test_compresses_self_when_other_is_compressed():
    """_align_q_dimensions_for_operation compresses self when the other is compressed."""
    mat1 = np.zeros((4, 4, 4))
    mat2 = np.zeros((64,))
    obj1 = IAmNonLocal(mat1, (4, 4, 4))
    obj2 = IAmNonLocal(mat2, (4, 4, 4), has_compressed_q_dimension=True)
    aligned = obj1._align_q_dimensions_for_operations(obj2)
    assert obj1.has_compressed_q_dimension
    assert aligned.has_compressed_q_dimension


def test_compresses_other_when_self_is_compressed():
    """_align_q_dimensions_for_operation compresses the other when self is compressed."""
    mat1 = np.zeros((64,))
    mat2 = np.zeros((4, 4, 4))
    obj1 = IAmNonLocal(mat1, (4, 4, 4), has_compressed_q_dimension=True)
    obj2 = IAmNonLocal(mat2, (4, 4, 4))
    aligned = obj1._align_q_dimensions_for_operations(obj2)
    assert obj1.has_compressed_q_dimension
    assert aligned.has_compressed_q_dimension


def test_filter_small_values_sets_tiny_entries_to_zero():
    """filter_small_values zeroes tiny entries."""
    mat = np.array(
        [
            [1e-13 + 1e-13j, 1e-11 + 1e-13j],
            [1e-13 + 1e-11j, 1.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    obj = IHaveMat(mat)
    returned = obj.filter_small_values()  # default threshold 1e-12

    # method returns self
    assert returned is obj

    res = obj.mat
    assert res[0, 0] == 0.0 + 0.0j  # both real and imag below threshold -> zeroed
    assert res[0, 1] != 0.0 + 0.0j  # imag above threshold -> not zeroed
    assert res[1, 0] != 0.0 + 0.0j  # imag above threshold -> not zeroed
    assert res[1, 1] == 1.0 + 0.0j  # large value preserved


def test_filter_small_values_respects_custom_threshold():
    """filter_small_values respects a custom threshold."""
    mat = np.array([1e-6 + 1e-6j, 2e-6 + 0.0j, 5e-5 + 1e-8j], dtype=np.complex128)
    obj = IHaveMat(mat)
    obj.filter_small_values(threshold=1e-5)

    # first two entries have both components < 1e-5 -> zeroed
    assert obj.mat[0] == 0.0 + 0.0j
    assert obj.mat[1] == 0.0 + 0.0j
    # last entry has real component above threshold -> preserved
    assert not (obj.mat[2].real == 0.0 and obj.mat[2].imag == 0.0)


def test_filter_small_values_preserves_values_with_one_large_component():
    """filter_small_values preserves entries with one large component."""
    mat = np.array([1e-13 + 1e-8j, 1e-8 + 1e-13j], dtype=np.complex128)
    obj = IHaveMat(mat)
    obj.filter_small_values(threshold=1e-12)

    # entries with at least one component above threshold must be preserved
    assert not (obj.mat[0].real == 0.0 and obj.mat[0].imag == 0.0)
    assert not (obj.mat[1].real == 0.0 and obj.mat[1].imag == 0.0)


def test_free_releases_underlying_matrix():
    """free releases the underlying matrix."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)

    # ensure matrix is set initially
    assert obj.mat is not None

    # free without trim should release the array
    obj.free(trim=False)
    assert obj.mat is None


def test_free_with_trim_calls_malloc_trim(monkeypatch):
    """free with trim calls malloc_trim."""
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)

    # prepare a fake libc with a malloc_trim that records calls
    class FakeLibc:
        def __init__(self):
            self.called = False

        def malloc_trim(self, arg):
            # record that the function was invoked
            self.called = True

    fake = FakeLibc()

    # make the class think malloc_trim is available and supply our fake libc
    monkeypatch.setattr(IHaveMat, "_malloc_trim_available", True)
    monkeypatch.setattr(IHaveMat, "_libc", fake)

    # call free with trim and ensure the libc's malloc_trim was invoked
    obj.free(trim=True)
    assert fake.called is True
    assert obj.mat is None


def test__malloc_trim_is_noop_when_unavailable(monkeypatch):
    """_malloc_trim is a no-op when malloc_trim is unavailable."""
    # ensure that when _malloc_trim_available is False, calling _malloc_trim does nothing
    monkeypatch.setattr(IHaveMat, "_malloc_trim_available", False)

    # set a libc that would raise if called to ensure it's not invoked
    class ExplodingLibc:
        def malloc_trim(self, arg):
            raise RuntimeError("should not be called")

    monkeypatch.setattr(IHaveMat, "_libc", ExplodingLibc())

    # should not raise
    IHaveMat._malloc_trim()

    # also ensure free(trim=True) will not try to call malloc_trim when availability is False
    mat = np.array([1.0, 2.0, 3.0])
    obj = IHaveMat(mat)
    obj.free(trim=True)
    assert obj.mat is None


def test_enter_returns_self():
    """The context-manager __enter__ returns self."""
    mat = np.array([[1.0]])
    obj = IHaveMat(mat)
    assert obj.__enter__() is obj


def test_exit_calls_free_without_trim(monkeypatch):
    """__exit__ frees the matrix without trimming."""
    mat = np.array([[1.0]])
    obj = IHaveMat(mat)

    called = {}

    def fake_free(self, trim=False):
        called["called"] = True
        called["trim"] = trim
        self._mat = None

    monkeypatch.setattr(IHaveMat, "free", fake_free)

    # simulate context manager exit
    obj.__exit__(None, None, None)

    assert called.get("called") is True
    assert called.get("trim") is True
    assert obj.mat is None


def test_del_calls_free_without_trim(monkeypatch):
    """The destructor frees the matrix without trimming."""
    mat = np.array([[1.0]])
    obj = IHaveMat(mat)

    called = {}

    def fake_free(self, trim=False):
        called["called"] = True
        called["trim"] = trim
        self._mat = None

    monkeypatch.setattr(IHaveMat, "free", fake_free)

    # call destructor implementation directly
    obj.__del__()

    assert called.get("called") is True
    assert called.get("trim") is False
    assert obj.mat is None


def test_skip_on_non_posix_or_no_proc(monkeypatch):
    """malloc_trim detection is skipped on non-POSIX or missing /proc."""
    # simulate non-posix or missing /proc -> should mark unavailable and return
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    monkeypatch.setattr(IHaveMat, "_malloc_trim_available", None, raising=False)

    IHaveMat._malloc_trim()

    assert IHaveMat._malloc_trim_available is False


def test_loads_libc_and_calls_malloc_trim(monkeypatch):
    """malloc_trim loads libc and calls malloc_trim when available."""

    # simulate posix with /proc and a working ctypes.CDLL returning a libc with malloc_trim
    class FakeLib:
        def __init__(self):
            self.called = False

        def malloc_trim(self, arg):
            self.called = True
            return 1

    fake_lib = FakeLib()
    fake_ctypes = types.ModuleType("ctypes")
    fake_ctypes.CDLL = lambda name: fake_lib

    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    monkeypatch.setattr(IHaveMat, "_malloc_trim_available", None, raising=False)

    IHaveMat._malloc_trim()

    assert IHaveMat._malloc_trim_available is True
    assert getattr(IHaveMat, "_libc") is fake_lib
    assert fake_lib.called is True


def test_ctypes_cdll_failure_sets_unavailable(monkeypatch):
    """A ctypes CDLL failure marks malloc_trim unavailable."""

    # simulate posix with /proc but CDLL raises -> should mark unavailable and not raise
    def failing_cdll(name):
        raise OSError("no libc")

    fake_ctypes = types.ModuleType("ctypes")
    fake_ctypes.CDLL = failing_cdll

    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    monkeypatch.setattr(IHaveMat, "_malloc_trim_available", None, raising=False)

    IHaveMat._malloc_trim()

    assert IHaveMat._malloc_trim_available is False


def test_malloc_trim_exception_is_suppressed(monkeypatch):
    """An exception from malloc_trim is suppressed."""

    # simulate libc present but malloc_trim itself raises -> should be suppressed (no exception)
    class BadLib:
        def malloc_trim(self, arg):
            raise RuntimeError("boom")

    fake_ctypes = types.ModuleType("ctypes")
    fake_ctypes.CDLL = lambda name: BadLib()

    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    monkeypatch.setattr(IHaveMat, "_malloc_trim_available", None, raising=False)

    # must not raise
    IHaveMat._malloc_trim()

    # when ctypes loaded successfully, availability should be True even if malloc_trim raised
    assert IHaveMat._malloc_trim_available is True


def test_filter_q_index_returns_correct_index():
    """filter_q_index returns the requested momentum index."""
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index(5)
    assert result.mat.shape == (1, 2)
    assert np.allclose(result.mat[0], mat.reshape(64, 2)[5], atol=1e-6)


def test_filter_q_index_default_index_is_zero():
    """filter_q_index defaults to index zero."""
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index()
    assert result.mat.shape == (1, 2)
    assert np.allclose(result.mat[0], mat.reshape(64, 2)[0], atol=1e-6)


def test_filter_q_index_compresses_q_dimension_if_not_already_compressed():
    """filter_q_index compresses the q dimension when not already compressed."""
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    assert not obj.has_compressed_q_dimension
    _ = obj.filter_q_index(0)
    assert obj.has_compressed_q_dimension


def test_filter_q_index_does_not_modify_original_when_already_compressed():
    """filter_q_index does not modify an already-compressed original."""
    mat = np.arange(64 * 2).reshape((64, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    original_mat = obj.mat.copy()
    _ = obj.filter_q_index(3)
    assert np.allclose(obj.mat, original_mat, atol=1e-6)


def test_filter_q_index_sets_nq_to_one():
    """filter_q_index sets nq to one."""
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index(0)
    assert result.nq == (1, 1, 1)


def test_filter_q_index_result_has_compressed_q_dimension():
    """The filter_q_index result has a compressed q dimension."""
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index(0)
    assert result.has_compressed_q_dimension


def test_filter_q_index_result_original_shape_is_updated():
    """filter_q_index updates the result's original shape."""
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index(0)
    assert result.original_shape == (1, 2)


def test_filter_q_index_returns_deep_copy():
    """filter_q_index returns a deep copy."""
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index(0)
    result.mat[0, 0] = 9999
    assert not np.allclose(obj.mat.reshape(64, 2)[0, 0], 9999, atol=1e-6)


def test_filter_q_index_last_index():
    """filter_q_index works for the last momentum index."""
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index(63)
    assert np.allclose(result.mat[0], mat.reshape(64, 2)[63], atol=1e-6)


def test_filter_q_index_raises_for_out_of_bounds_index():
    """filter_q_index raises for an out-of-bounds index."""
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    with pytest.raises(IndexError):
        obj.filter_q_index(64)


def test_filter_q_index_does_not_retain_parent_array():
    """filter_q_index copies the single q-slice instead of keeping the full parent array alive via a view."""
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    obj = IAmNonLocal(mat, (4, 4, 4))
    result = obj.filter_q_index(5)
    assert result.mat.base is None


def test_nq_tot_decompressed_is_product_of_nq():
    """nq_tot multiplies the per-direction momentum counts (as a plain int) when decompressed."""
    obj = IAmNonLocal(np.arange(2 * 3 * 4 * 2).reshape((2, 3, 4, 2)), (2, 3, 4))
    assert obj.nq_tot == 24
    assert isinstance(obj.nq_tot, int)


def test_nq_tot_compressed_uses_stored_leading_axis():
    """nq_tot uses the stored leading axis length when compressed (which may be IBZ-reduced below prod(nq))."""
    obj = IAmNonLocal(np.arange(10 * 2).reshape((10, 2)), (2, 3, 4), has_compressed_q_dimension=True)
    assert obj.nq_tot == 10


def test_q_mean_averages_full_momentum_grid_to_single_q_point():
    """q_mean averages the full momentum grid to a single q-point."""
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)

    result = obj.q_mean()

    expected = np.mean(mat, axis=(0, 1, 2))[None, None, None, ...]
    assert result.mat.shape == (1, 1, 1, 2)
    assert np.allclose(result.mat, expected, atol=1e-6)
    assert result.nq == (1, 1, 1)
    assert result.original_shape == (1, 1, 1, 2)
    assert np.allclose(obj.mat, mat, atol=1e-6)


def test_q_mean_averages_compressed_momentum_grid_to_single_q_point():
    """q_mean averages a compressed momentum grid to a single q-point."""
    mat = np.arange(64 * 2).reshape((64, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)

    result = obj.q_mean()

    expected = np.mean(mat, axis=0)[None, ...]
    assert result.mat.shape == (1, 2)
    assert np.allclose(result.mat, expected, atol=1e-6)
    assert result.nq == (1, 1, 1)
    assert result.original_shape == (1, 2)
    assert np.allclose(obj.mat, mat, atol=1e-6)


def test_q_mean_preserves_values_for_single_momentum_point():
    """q_mean preserves the value for a single momentum point."""
    mat = np.array([[[[3.0, -2.0j]]]], dtype=complex)
    nq = (1, 1, 1)
    obj = IAmNonLocal(mat, nq)

    result = obj.q_mean()

    assert result.mat.shape == (1, 1, 1, 2)
    assert np.allclose(result.mat[0, 0, 0], mat[0, 0, 0], atol=1e-6)
    assert result.nq == (1, 1, 1)
    assert np.allclose(obj.mat, mat, atol=1e-6)


def _rng_payload(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_interpolate_q_grid_same_grid_returns_unchanged_copy():
    """interpolate_q_grid on the same grid returns an unchanged copy."""
    mat = _rng_payload((4, 4, 4, 1, 1, 3))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4))
    out = obj.interpolate_q_grid((4, 4, 4))
    assert out is not obj
    assert np.array_equal(out.mat, mat)
    assert out.nq == (4, 4, 4)


def test_interpolate_q_grid_same_grid_no_copy_returns_self():
    """interpolate_q_grid on the same grid with copy=False returns self."""
    mat = _rng_payload((4, 4, 4, 1, 1, 2))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4))
    out = obj.interpolate_q_grid((4, 4, 4), copy=False)
    assert out is obj


def test_interpolate_q_grid_commensurate_downsample_is_exact_slice():
    """interpolate_q_grid commensurate downsampling is an exact slice."""
    mat = _rng_payload((4, 4, 4, 2, 2, 3))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4))
    out = obj.interpolate_q_grid((2, 2, 2))
    assert out.current_shape == (2, 2, 2, 2, 2, 3)
    assert np.array_equal(out.mat, mat[::2, ::2, ::2])
    assert out.nq == (2, 2, 2)


def test_interpolate_q_grid_collapse_z_takes_kz0_plane():
    """interpolate_q_grid collapsing z takes the kz=0 plane."""
    mat = _rng_payload((4, 4, 16, 1, 1, 2))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 16))
    out = obj.interpolate_q_grid((4, 4, 1))
    assert out.current_shape == (4, 4, 1, 1, 1, 2)
    assert np.array_equal(out.mat, mat[:, :, 0:1])
    assert out.nq == (4, 4, 1)


def test_interpolate_q_grid_upsample_shape_and_nq():
    """interpolate_q_grid upsampling gives the right shape and nq."""
    mat = _rng_payload((4, 4, 4, 1, 1, 3))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4))
    out = obj.interpolate_q_grid((8, 8, 8))
    assert out.current_shape == (8, 8, 8, 1, 1, 3)
    assert out.nq == (8, 8, 8)


def test_interpolate_q_grid_upsample_preserves_values_at_shared_points():
    """interpolate_q_grid upsampling preserves values at shared grid points."""
    mat = _rng_payload((4, 4, 4, 1, 1, 2))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4))
    out = obj.interpolate_q_grid((8, 8, 8))
    assert np.allclose(out.mat[::2, ::2, ::2], mat, atol=1e-10)


def test_interpolate_q_grid_up_then_commensurate_down_round_trips():
    """interpolate_q_grid up- then commensurate down-sampling round-trips."""
    mat = _rng_payload((4, 4, 4, 2, 2, 3))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4))
    out = obj.interpolate_q_grid((8, 8, 8)).interpolate_q_grid((4, 4, 4))
    assert np.allclose(out.mat, mat, atol=1e-10)
    assert out.nq == (4, 4, 4)


def test_interpolate_q_grid_incommensurate_matches_scipy_resample():
    """interpolate_q_grid incommensurate resampling matches scipy.signal.resample."""
    mat = _rng_payload((12, 12, 4, 1, 1, 2))
    obj = _DoublePrecisionNonLocal(mat.copy(), (12, 12, 4))
    out = obj.interpolate_q_grid((8, 8, 4))
    ref = resample(resample(mat, 8, axis=0), 8, axis=1)
    assert out.current_shape == (8, 8, 4, 1, 1, 2)
    assert np.allclose(out.mat, ref, atol=1e-10)


def test_interpolate_q_grid_incommensurate_is_not_a_slice():
    """interpolate_q_grid incommensurate resampling is not a plain slice."""
    mat = _rng_payload((12, 4, 4, 1, 1, 1))
    obj = _DoublePrecisionNonLocal(mat.copy(), (12, 4, 4))
    out = obj.interpolate_q_grid((8, 4, 4))
    # no integer stride from 12 -> 8; the band-limited result must differ from a naive slice
    assert not np.allclose(out.mat[:, 0, 0, 0, 0, 0], mat[:8, 0, 0, 0, 0, 0])


def test_interpolate_q_grid_mixed_axes_slice_and_resample():
    """interpolate_q_grid mixes slicing and resampling across axes."""
    mat = _rng_payload((8, 6, 4, 1, 1, 2))
    obj = _DoublePrecisionNonLocal(mat.copy(), (8, 6, 4))
    out = obj.interpolate_q_grid((4, 12, 4))
    assert out.current_shape == (4, 12, 4, 1, 1, 2)
    assert out.nq == (4, 12, 4)
    # axis 0 is commensurate-down (exact slice), axis 1 is upsampled (Fourier)
    ref = resample(mat[::2], 12, axis=1)
    assert np.allclose(out.mat, ref, atol=1e-10)


def test_interpolate_q_grid_compressed_in_compressed_out():
    """interpolate_q_grid keeps a compressed input compressed."""
    mat = _rng_payload((64, 2, 2, 3))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4), has_compressed_q_dimension=True)
    out = obj.interpolate_q_grid((2, 2, 2))
    assert out.has_compressed_q_dimension is True
    assert out.current_shape == (8, 2, 2, 3)
    assert out.nq == (2, 2, 2)


def test_interpolate_q_grid_compressed_values_match_decompressed_path():
    """interpolate_q_grid compressed and decompressed paths agree."""
    mat = _rng_payload((4, 4, 4, 1, 1, 2))
    dec = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4))
    comp = _DoublePrecisionNonLocal(mat.reshape(64, 1, 1, 2).copy(), (4, 4, 4), has_compressed_q_dimension=True)
    out_dec = dec.interpolate_q_grid((2, 2, 2))
    out_comp = comp.interpolate_q_grid((2, 2, 2))
    assert np.array_equal(out_comp.mat, out_dec.mat.reshape(8, 1, 1, 2))


def test_interpolate_q_grid_decompressed_in_decompressed_out():
    """interpolate_q_grid keeps a decompressed input decompressed."""
    mat = _rng_payload((4, 4, 4, 1, 1, 2))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4))
    out = obj.interpolate_q_grid((2, 2, 2))
    assert out.has_compressed_q_dimension is False


def test_interpolate_q_grid_updates_original_shape():
    """interpolate_q_grid updates the original shape."""
    mat = _rng_payload((4, 4, 4, 1, 1, 2))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4))
    out = obj.interpolate_q_grid((2, 2, 2))
    assert out.original_shape == out.current_shape == (2, 2, 2, 1, 1, 2)


def test_interpolate_q_grid_leaves_trailing_axes_untouched():
    """interpolate_q_grid leaves the trailing axes untouched."""
    mat = _rng_payload((4, 4, 4, 2, 3, 5))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4))
    out = obj.interpolate_q_grid((8, 8, 8))
    assert out.current_shape[3:] == (2, 3, 5)


def test_interpolate_q_grid_copy_true_leaves_original_intact():
    """interpolate_q_grid with copy=True leaves the original intact."""
    mat = _rng_payload((4, 4, 4, 1, 1, 2))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4))
    obj.interpolate_q_grid((2, 2, 2))
    assert obj.current_shape == (4, 4, 4, 1, 1, 2)
    assert obj.nq == (4, 4, 4)


def test_interpolate_q_grid_copy_false_mutates_self():
    """interpolate_q_grid with copy=False mutates self."""
    mat = _rng_payload((4, 4, 4, 1, 1, 2))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4))
    out = obj.interpolate_q_grid((2, 2, 2), copy=False)
    assert out is obj
    assert obj.current_shape == (2, 2, 2, 1, 1, 2)
    assert obj.nq == (2, 2, 2)


def test_interpolate_q_grid_production_class_returns_complex64():
    """interpolate_q_grid returns complex64 for a production class."""
    mat = _rng_payload((4, 4, 4, 1, 1, 2))
    obj = IAmNonLocal(mat.copy(), (4, 4, 4))
    out = obj.interpolate_q_grid((8, 8, 8))
    assert out.mat.dtype == np.complex64


def test_interpolate_q_grid_rejects_non_positive_sizes():
    """interpolate_q_grid rejects non-positive grid sizes."""
    mat = _rng_payload((4, 4, 4, 1, 1, 2))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4))
    with pytest.raises(ValueError):
        obj.interpolate_q_grid((4, 4, 0))
    with pytest.raises(ValueError):
        obj.interpolate_q_grid((4, -2, 4))


def test_interpolate_q_grid_rejects_ibz_reduced_input():
    """interpolate_q_grid rejects an IBZ-reduced input."""
    # compressed first dim (10) != prod(nq) = 64 -> IBZ-like, must be rejected
    mat = _rng_payload((10, 1, 1, 2))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 4), has_compressed_q_dimension=True)
    with pytest.raises(ValueError, match="full-BZ"):
        obj.interpolate_q_grid((2, 2, 2))


def test_interpolate_q_grid_single_kpoint_axis_upsamples_constant():
    """interpolate_q_grid upsamples a single-k-point axis as a constant."""
    # n_old = 1 -> resample replicates the constant along that axis
    mat = _rng_payload((4, 4, 1, 1, 1, 2))
    obj = _DoublePrecisionNonLocal(mat.copy(), (4, 4, 1))
    out = obj.interpolate_q_grid((4, 4, 3))
    assert out.current_shape == (4, 4, 3, 1, 1, 2)
    assert np.allclose(out.mat[:, :, 0], out.mat[:, :, 1], atol=1e-10)
    assert np.allclose(out.mat[:, :, 0], out.mat[:, :, 2], atol=1e-10)


def _build_auto_kgrid(nx=4, ny=4, nz=4, nb=1, hopping=1.0, include_antiunitary=False):
    """Build an auto-mode KGrid populated with a small real cubic Hamiltonian.
    Returns (kgrid, H_full[nx,ny,nz,nb,nb])."""
    j1, j2, j3 = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    k1 = 2 * np.pi * j1 / nx
    k2 = 2 * np.pi * j2 / ny
    k3 = 2 * np.pi * j3 / nz
    H = np.zeros((nx, ny, nz, nb, nb), dtype=complex)
    eps = -2.0 * hopping * (np.cos(k1) + np.cos(k2) + np.cos(k3))
    for o in range(nb):
        H[..., o, o] = eps + 0.1 * o
    grid = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    grid.specify_auto_symmetries(H, include_antiunitary=include_antiunitary)
    return grid, H


class _DoublePrecisionNonLocal(IAmNonLocal):
    """IAmNonLocal subclass that preserves the input matrix dtype instead of
    casting to complex64. Lets us verify the mapping logic against double-precision
    references; the production class deliberately downcasts for memory savings."""

    @IAmNonLocal.mat.setter
    def mat(self, value):
        if value is None:
            self._mat = None
            return
        self._mat = np.asarray(value)


def test_map_to_full_bz_legacy_kgrid_pure_replication():
    """With a legacy (non-auto) KGrid, ``_map_to_full_bz`` reduces to a bare IBZ→FBZ index expansion via ``irrk_inv``: each FBZ point gets the IBZ value at the index pointed to by ``irrk_inv``, with no orbital transformation."""
    grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    nb = 1
    nq_tot = 16
    # Make a clearly-non-trivial IBZ payload
    ibz_payload = (np.arange(grid.nk_irr) + 1).astype(np.complex128).reshape(grid.nk_irr, nb, nb)
    obj = _DoublePrecisionNonLocal(mat=ibz_payload.copy(), nq=(4, 4, 1), has_compressed_q_dimension=True)
    obj._map_to_full_bz(grid, num_orbital_dimensions=2)

    assert obj.mat.shape == (nq_tot, nb, nb)
    # Every FBZ k must hold the IBZ value at irrk_inv[k]
    inv = grid.irrk_inv.ravel()
    expected = ibz_payload[inv]
    assert np.array_equal(obj.mat, expected)


def test_map_to_full_bz_auto_2idx_reconstructs_H_exactly():
    """End-to-end: pick auto IBZ slice of H, _map_to_full_bz should reproduce H."""
    grid, H = _build_auto_kgrid(nx=4, ny=4, nz=4, nb=1)
    nb = 1
    H_flat = H.reshape(-1, nb, nb)
    H_ibz = H_flat[grid.irrk_ind].copy()
    obj = _DoublePrecisionNonLocal(mat=H_ibz, nq=(4, 4, 4), has_compressed_q_dimension=True)
    obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    H_rec = obj.mat.reshape(4, 4, 4, nb, nb)
    assert np.allclose(H_rec, H, atol=1e-12)


def test_map_to_full_bz_auto_2idx_reconstructs_H_for_multiorbital_case():
    """Same as above but with multiple orbitals - exercises the orbital einsum path."""
    grid, H = _build_auto_kgrid(nx=4, ny=4, nz=4, nb=2)
    nb = 2
    H_flat = H.reshape(-1, nb, nb)
    H_ibz = H_flat[grid.irrk_ind].copy()
    obj = _DoublePrecisionNonLocal(mat=H_ibz, nq=(4, 4, 4), has_compressed_q_dimension=True)
    obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    H_rec = obj.mat.reshape(4, 4, 4, nb, nb)
    assert np.allclose(H_rec, H, atol=1e-12)


def test_map_to_full_bz_auto_4idx_reconstructs_HotimesH_exactly():
    """For Γ = H ⊗ H (which inherits H's symmetry trivially), reconstruction must be exact under the 4-orbital-index code path."""
    grid, H = _build_auto_kgrid(nx=3, ny=3, nz=3, nb=2)
    nb = 2
    Gamma_full = np.einsum("...ab,...cd->...abcd", H, H)
    Gamma_flat = Gamma_full.reshape(-1, nb, nb, nb, nb)
    Gamma_ibz = Gamma_flat[grid.irrk_ind].copy()
    obj = _DoublePrecisionNonLocal(mat=Gamma_ibz, nq=(3, 3, 3), has_compressed_q_dimension=True)
    obj._map_to_full_bz(grid, num_orbital_dimensions=4)
    G_rec = obj.mat.reshape(3, 3, 3, nb, nb, nb, nb)
    assert np.allclose(G_rec, Gamma_full, atol=1e-12)


def test_map_to_full_bz_auto_preserves_trailing_frequency_dimensions():
    """The mapping is shape-polymorphic in the trailing axes (e.g. frequency axes). A 1-band IBZ payload with 2 frequency axes after the orbital pair must come back to the full BZ unmodified beyond the index expansion."""
    grid, _ = _build_auto_kgrid(nx=4, ny=4, nz=1, nb=1)
    nb = 1
    n_freq = 5
    # Distinct payload at every IBZ slot so missing/wrong indices show up
    rng = np.random.default_rng(0)
    ibz_payload = rng.standard_normal((grid.nk_irr, nb, nb, n_freq)) + 1j * rng.standard_normal(
        (grid.nk_irr, nb, nb, n_freq)
    )
    obj = _DoublePrecisionNonLocal(mat=ibz_payload.copy(), nq=(4, 4, 1), has_compressed_q_dimension=True)
    obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    # For 1-band the orbital transform is identity, so the result is pure replication
    inv = grid.irrk_inv.ravel()
    expected = ibz_payload[inv]
    assert obj.mat.shape == expected.shape
    assert np.allclose(obj.mat, expected, atol=1e-14)


def test_map_to_full_bz_auto_default_no_antiunitary_does_no_conjugation():
    """Default (include_antiunitary=False): no FBZ point should ever be conjugated, so a complex IBZ payload reconstructs as a pure index replication. This is the safe semantics for frequency-dependent objects."""
    nb = 1
    grid, _ = _build_auto_kgrid(nx=4, ny=4, nz=1, nb=nb, include_antiunitary=False)
    assert int(grid._auto_conjs.sum()) == 0

    rng = np.random.default_rng(2)
    ibz_payload = rng.standard_normal((grid.nk_irr, nb, nb)) + 1j * rng.standard_normal((grid.nk_irr, nb, nb))
    obj = _DoublePrecisionNonLocal(mat=ibz_payload.copy(), nq=(4, 4, 1), has_compressed_q_dimension=True)
    obj._map_to_full_bz(grid, num_orbital_dimensions=2)

    inv = grid.irrk_inv.ravel()
    expected = ibz_payload[inv]
    assert np.allclose(obj.mat, expected, atol=1e-14)


def test_map_to_full_bz_auto_delegates_to_apply_auto_orbital_transform():
    """The auto branch must call ``symmetry_reduction.apply_auto_orbital_transform`` with the correctly-sliced (Us, sigmas, conjs) arrays and the right ndim."""
    grid, H = _build_auto_kgrid(nx=4, ny=4, nz=4, nb=2)
    nb = 2
    nktot = 4 * 4 * 4
    H_flat = H.reshape(-1, nb, nb)
    H_ibz = H_flat[grid.irrk_ind].copy()
    obj = _DoublePrecisionNonLocal(mat=H_ibz, nq=(4, 4, 4), has_compressed_q_dimension=True)

    # Patch so we can assert it gets called with the right shapes and args
    with patch.object(sr, "apply_auto_orbital_transform", wraps=sr.apply_auto_orbital_transform) as spy:
        obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    assert spy.call_count == 1
    _, kwargs = spy.call_args
    # The function was called with keyword arguments matching the signature
    assert kwargs["num_orbital_dimensions"] == 2
    assert kwargs["us"].shape == (nktot, nb, nb)
    assert kwargs["sigmas"].shape == (nktot,)
    assert kwargs["conjs"].shape == (nktot,)


def test_map_to_full_bz_auto_passes_num_orbital_dimensions_4_for_vertex():
    """map_to_full_bz auto mode passes num_orbital_dimensions=4 for a vertex."""
    grid, H = _build_auto_kgrid(nx=3, ny=3, nz=3, nb=2)
    nb = 2
    Gamma_full = np.einsum("...ab,...cd->...abcd", H, H)
    Gamma_ibz = Gamma_full.reshape(-1, nb, nb, nb, nb)[grid.irrk_ind].copy()
    obj = _DoublePrecisionNonLocal(mat=Gamma_ibz, nq=(3, 3, 3), has_compressed_q_dimension=True)
    with patch.object(sr, "apply_auto_orbital_transform", wraps=sr.apply_auto_orbital_transform) as spy:
        obj._map_to_full_bz(grid, num_orbital_dimensions=4)
    _, kwargs = spy.call_args
    assert kwargs["num_orbital_dimensions"] == 4


def test_map_to_full_bz_legacy_kgrid_does_not_call_orbital_transform():
    """For a legacy KGrid (not auto-mode), the orbital transform helper must NOT be called: only the IBZ→FBZ replication runs."""
    grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    nb = 1
    ibz_payload = np.arange(grid.nk_irr).astype(np.complex128).reshape(grid.nk_irr, nb, nb)
    obj = _DoublePrecisionNonLocal(mat=ibz_payload.copy(), nq=(4, 4, 1), has_compressed_q_dimension=True)
    with patch.object(sr, "apply_auto_orbital_transform", wraps=sr.apply_auto_orbital_transform) as spy:
        obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    assert spy.call_count == 0


def test_map_to_full_bz_raises_for_invalid_num_orbital_dimensions():
    """Only ``num_orbital_dimensions`` in {2, 4} are supported."""
    grid, H = _build_auto_kgrid(nx=4, ny=4, nz=4, nb=1)
    H_ibz = H.reshape(-1, 1, 1)[grid.irrk_ind].copy()
    obj = _DoublePrecisionNonLocal(mat=H_ibz, nq=(4, 4, 4), has_compressed_q_dimension=True)
    with pytest.raises(AssertionError, match="2 or 4"):
        obj._map_to_full_bz(grid, num_orbital_dimensions=3)
    with pytest.raises(AssertionError, match="2 or 4"):
        obj._map_to_full_bz(grid, num_orbital_dimensions=1)


def test_map_to_full_bz_raises_when_not_compressed():
    """The compressed-q convention is required: an already-expanded matrix is not a valid input to ``_map_to_full_bz``."""
    grid, H = _build_auto_kgrid(nx=4, ny=4, nz=4, nb=1)
    obj = _DoublePrecisionNonLocal(mat=H, nq=(4, 4, 4), has_compressed_q_dimension=False)
    with pytest.raises(ValueError, match="compressed momentum dimension"):
        obj._map_to_full_bz(grid, num_orbital_dimensions=2)


def test_map_to_full_bz_auto_uses_supplied_nq_override():
    """The optional ``nq`` argument must override the object's stored ``nq``."""
    grid, H = _build_auto_kgrid(nx=4, ny=4, nz=4, nb=1)
    H_ibz = H.reshape(-1, 1, 1)[grid.irrk_ind].copy()
    obj = _DoublePrecisionNonLocal(mat=H_ibz, nq=(2, 2, 16), has_compressed_q_dimension=True)
    obj._map_to_full_bz(grid, num_orbital_dimensions=2, nq=(4, 4, 4))
    assert obj.nq == (4, 4, 4)
    H_rec = obj.mat.reshape(4, 4, 4, 1, 1)
    assert np.allclose(H_rec, H, atol=1e-12)


def test_map_to_full_bz_auto_returns_self_for_method_chaining():
    """For ergonomic chaining the method returns ``self``."""
    grid, H = _build_auto_kgrid(nx=4, ny=4, nz=4, nb=1)
    H_ibz = H.reshape(-1, 1, 1)[grid.irrk_ind].copy()
    obj = _DoublePrecisionNonLocal(mat=H_ibz, nq=(4, 4, 4), has_compressed_q_dimension=True)
    result = obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    assert result is obj


def test_map_to_full_bz_legacy_returns_self_for_method_chaining():
    """map_to_full_bz legacy mode returns self for chaining."""
    grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    nb = 1
    ibz_payload = np.arange(grid.nk_irr).astype(np.complex128).reshape(grid.nk_irr, nb, nb)
    obj = _DoublePrecisionNonLocal(mat=ibz_payload.copy(), nq=(4, 4, 1), has_compressed_q_dimension=True)
    result = obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    assert result is obj


def test_map_to_full_bz_auto_1x1x1_trivial_grid_is_identity():
    """Edge case: a 1×1×1 grid has a single k-point, so the FBZ trivially equals the IBZ and the mapping returns the input unchanged in value."""
    nb = 2
    H = np.zeros((1, 1, 1, nb, nb), dtype=complex)
    H[0, 0, 0] = np.array([[1.0, 0.3], [0.3, 2.0]])
    grid = bz.KGrid(nk=(1, 1, 1), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    grid.specify_auto_symmetries(H)
    H_ibz = H.reshape(-1, nb, nb)[grid.irrk_ind].copy()
    obj = _DoublePrecisionNonLocal(mat=H_ibz, nq=(1, 1, 1), has_compressed_q_dimension=True)
    obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    assert np.allclose(obj.mat.reshape(1, 1, 1, nb, nb), H, atol=1e-14)


def test_map_to_full_bz_auto_preserves_dtype():
    """The output matrix has the same dtype as the input (the function does not silently cast within the auto branch - the cast to complex64 happens elsewhere in ``IHaveMat.mat = value``)."""
    grid, H = _build_auto_kgrid(nx=4, ny=4, nz=4, nb=1)
    H_ibz_64 = H.reshape(-1, 1, 1)[grid.irrk_ind].astype(np.complex64).copy()
    obj = _DoublePrecisionNonLocal(mat=H_ibz_64, nq=(4, 4, 4), has_compressed_q_dimension=True)
    obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    assert obj.mat.dtype == np.complex64


def test_map_to_full_bz_auto_irrk_inv_consistency_at_every_fbz_point():
    """Every FBZ k must end up with the value at irrk_inv[k] transformed by the stored (U_k, sigma_k, conj_k). Check this explicitly point-by-point."""
    grid, H = _build_auto_kgrid(nx=3, ny=3, nz=3, nb=2)
    nb = 2
    H_flat = H.reshape(-1, nb, nb)
    H_ibz = H_flat[grid.irrk_ind].copy()
    obj = _DoublePrecisionNonLocal(mat=H_ibz, nq=(3, 3, 3), has_compressed_q_dimension=True)
    obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    H_rec = obj.mat.reshape(-1, nb, nb)

    inv = grid.irrk_inv.ravel()
    Us = grid._auto_us.reshape(-1, nb, nb)
    sigmas = grid._auto_sigmas.reshape(-1)
    conjs = grid._auto_conjs.reshape(-1)
    for k in range(H_rec.shape[0]):
        block = H_ibz[inv[k]]
        if conjs[k]:
            block = block.conj()
        expected = sigmas[k] * Us[k] @ block @ Us[k].conj().T
        assert np.allclose(H_rec[k], expected, atol=1e-12), f"mismatch at FBZ k={k}"


def test_mul_rejects_ndarray_with_accurate_message():
    """Multiplication rejects an ndarray with an accurate error message."""
    u = LocalInteraction(np.ones((2, 2, 2, 2)))
    with pytest.raises(ValueError, match=r"only supported with numbers\."):
        u * np.ones((2, 2, 2, 2))


def test_mat_setter_does_not_copy_complex64():
    """The mat setter does not copy an already-complex64 array."""
    u = LocalInteraction(np.ones((2, 2, 2, 2)))
    arr = np.zeros((2, 2, 2, 2), dtype=np.complex64)
    u.mat = arr
    assert u.mat is arr  # no redundant copy for already-complex64 input


def test_mat_setter_still_casts_other_dtypes():
    """The mat setter still casts other dtypes to the storage dtype."""
    u = LocalInteraction(np.ones((2, 2, 2, 2)))
    arr = np.zeros((2, 2, 2, 2), dtype=np.float64)
    u.mat = arr
    assert u.mat.dtype == np.complex64
    assert u.mat is not arr


def test_fft_ifft_round_trip():
    """fft followed by ifft round-trips to the original matrix."""
    rng = np.random.default_rng(0)
    mat = (rng.standard_normal((2, 2, 1, 1, 1, 1, 1)) + 1j * rng.standard_normal((2, 2, 1, 1, 1, 1, 1))).astype(
        np.complex64
    )
    obj = Interaction(mat, SpinChannel.NONE, (2, 2, 1), has_compressed_q_dimension=False)
    original = obj.mat.copy()
    out = obj.fft(copy=True).ifft(copy=True)
    assert np.allclose(out.mat, original, atol=1e-4)


def test_fft_matches_reference_fftn():
    """fft matches a reference numpy fftn."""
    rng = np.random.default_rng(1)
    mat = (rng.standard_normal((3, 2, 1, 1, 1, 1, 1)) + 1j * rng.standard_normal((3, 2, 1, 1, 1, 1, 1))).astype(
        np.complex64
    )
    obj = Interaction(mat.copy(), SpinChannel.NONE, (3, 2, 1), has_compressed_q_dimension=False)
    out = obj.fft(copy=True)
    assert np.allclose(out.mat, np.fft.fftn(mat, axes=(0, 1, 2)), atol=1e-3)


def test_dtype_default_is_complex64():
    """The default storage dtype is complex64."""
    assert DTYPE == np.complex64
    assert IHaveMat.DTYPE == np.complex64


def test_dtype_constant_drives_storage_precision(monkeypatch):
    """The DTYPE constant drives the storage precision."""
    # The mat setter is the single enforcement point: every IHaveMat-derived object is coerced to the
    # module-level DTYPE. (The canonical way to switch precision is editing the DTYPE constant in source,
    # so all modules import the new value; here we patch the setter's global to verify the indirection.)
    import dgamore.n_point_base as npb

    monkeypatch.setattr(npb, "DTYPE", np.complex128)
    u = LocalInteraction(np.ones((2, 2, 2, 2)))
    assert u.mat.dtype == np.complex128


def test_shift_k_by_q_does_not_mutate_self_and_owns_result():
    """shift_k_by_q does not mutate self and owns its result."""
    mat = np.arange(64, dtype=complex).reshape((4, 4, 4))
    obj = IAmNonLocal(mat.copy(), (4, 4, 4))
    shifted = obj.shift_k_by_q((1, 1, 1))
    assert np.array_equal(obj.mat, mat)  # self untouched despite the shared-reference clone
    assert np.array_equal(shifted.mat, np.roll(mat, (-1, -1, -1), axis=(0, 1, 2)))
    assert shifted.mat.base is None


def test_shift_k_by_pi_does_not_mutate_self():
    """shift_k_by_pi does not mutate self."""
    mat = np.arange(64, dtype=complex).reshape((4, 4, 4))
    obj = IAmNonLocal(mat.copy(), (4, 4, 4))
    shifted = obj.shift_k_by_pi()
    assert np.array_equal(obj.mat, mat)
    assert np.array_equal(shifted.mat, np.roll(mat, (2, 2, 2), axis=(0, 1, 2)))


def test_q_mean_does_not_mutate_self():
    """q_mean does not mutate self."""
    mat = np.arange(64, dtype=complex).reshape((4, 4, 4))
    obj = IAmNonLocal(mat.copy(), (4, 4, 4))
    averaged = obj.q_mean()
    assert np.array_equal(obj.mat, mat)
    assert averaged.nq == (1, 1, 1)
    assert np.allclose(averaged.mat[0, 0, 0], mat.mean())


def test_flip_momentum_axis_copy_does_not_mutate_self():
    """flip_momentum_axis with copy=True does not mutate self."""
    mat = np.arange(64, dtype=complex).reshape((4, 4, 4))
    obj = IAmNonLocal(mat.copy(), (4, 4, 4))
    flipped = obj.flip_momentum_axis(copy=True)
    assert np.array_equal(obj.mat, mat)
    expected = np.roll(np.flip(mat, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
    assert np.array_equal(flipped.mat, expected)


def test_reduce_q_does_not_mutate_self_and_owns_result():
    """reduce_q does not mutate self and owns its result."""
    mat = np.arange(64, dtype=complex).reshape((4, 4, 4))
    obj = IAmNonLocal(mat.copy(), (4, 4, 4))
    reduced = obj.reduce_q(np.array([[1, 1, 1], [2, 2, 2]]))
    assert np.array_equal(obj.mat, mat)  # self untouched
    assert reduced.mat.base is None  # advanced indexing copies, releasing the parent
    assert np.allclose(reduced.mat, [mat[1, 1, 1], mat[2, 2, 2]])


def test_reduce_q_selects_in_ascending_flat_order_regardless_of_input_order():
    """reduce_q selects momenta in ascending flat order regardless of input order."""
    mat = np.arange(64, dtype=complex).reshape((4, 4, 4))
    obj = IAmNonLocal(mat.copy(), (4, 4, 4))
    reduced = obj.reduce_q(np.array([[2, 2, 2], [1, 1, 1]]))
    # selection follows ascending flat index, not the order given (matches the previous mask-based behaviour)
    assert np.allclose(reduced.mat, [mat[1, 1, 1], mat[2, 2, 2]])


def test_reduce_q_silently_drops_out_of_range_momenta():
    """reduce_q silently drops out-of-range momenta."""
    mat = np.arange(64, dtype=complex).reshape((4, 4, 4))
    obj = IAmNonLocal(mat.copy(), (4, 4, 4))
    reduced = obj.reduce_q(np.array([[1, 1, 1], [9, 9, 9]]))
    assert reduced.mat.shape == (1,)
    assert np.allclose(reduced.mat, [mat[1, 1, 1]])


def test_find_q_returns_independent_copy():
    """find_q returns an independent copy."""
    mat = np.arange(64, dtype=complex).reshape((4, 4, 4))
    obj = IAmNonLocal(mat.copy(), (4, 4, 4))
    result = obj.find_q((1, 1, 1))
    result.mat[0] = -777.0
    assert obj.mat[1, 1, 1] == mat[1, 1, 1]  # mutating the result must not touch self


def _filter_reference(mat: np.ndarray, threshold: float) -> np.ndarray:
    """Single-shot reference implementation of filter_small_values for parity checks."""
    out = mat.copy()
    out[(np.abs(out.real) < threshold) & (np.abs(out.imag) < threshold)] = 0.0
    return out


def test_filter_small_values_chunked_path_matches_single_pass(monkeypatch):
    """filter_small_values chunked path matches the single-pass result."""
    rng = np.random.default_rng(7)
    mat = (rng.standard_normal((50, 4)) + 1j * rng.standard_normal((50, 4))).astype(np.complex64)
    mat[::3] = 1e-15 + 1e-15j  # make the mask non-trivial and span several chunks
    reference = _filter_reference(mat, 1e-12)

    obj = IHaveMat(mat.copy())
    # Tiny budget forces step == 1, i.e. the multi-chunk branch (one element per chunk).
    monkeypatch.setattr(IHaveMat, "_FILTER_CHUNK_BYTES", 8)
    returned = obj.filter_small_values(threshold=1e-12)

    assert returned is obj  # chainable
    assert np.array_equal(obj.mat, reference)


def test_filter_small_values_handles_non_contiguous_input(monkeypatch):
    """filter_small_values handles a non-contiguous input."""
    rng = np.random.default_rng(11)
    base = (rng.standard_normal((6, 8)) + 1j * rng.standard_normal((6, 8))).astype(np.complex64)
    base[:, ::2] = 1e-15 + 1e-15j
    non_contiguous = base.T  # transpose -> non-contiguous, axis-0 length 8
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    reference = _filter_reference(non_contiguous, 1e-12)

    obj = IHaveMat(non_contiguous)  # already complex64, so the setter keeps the non-contiguous view
    assert not obj.mat.flags["C_CONTIGUOUS"]
    monkeypatch.setattr(IHaveMat, "_FILTER_CHUNK_BYTES", 8)  # force axis-0 chunking on the non-contiguous branch
    obj.filter_small_values(threshold=1e-12)

    assert np.array_equal(obj.mat, reference)


def test_fft_ifft_preserve_complex64_dtype():
    """fft and ifft preserve the complex64 dtype."""
    # the BZ FFTs must keep complex64 (scipy.fft + overwrite_x); a regression to np.fft would upcast/spike.
    rng = np.random.default_rng(3)
    shape = (4, 4, 2, 1, 1, 6)
    mat = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    obj = IAmNonLocal(mat.copy(), (4, 4, 2))

    forward = obj.fft()
    assert forward.mat.dtype == np.complex64
    back = forward.ifft()
    assert back.mat.dtype == np.complex64
    assert np.allclose(back.mat, mat, atol=1e-4)  # round-trip recovers the input


def test_q_mean_preserves_complex64_dtype():
    """q_mean preserves the complex64 dtype."""
    mat = np.ones((4, 4, 2, 3), dtype=np.complex64)
    obj = IAmNonLocal(mat, (4, 4, 2))
    assert obj.q_mean().mat.dtype == np.complex64

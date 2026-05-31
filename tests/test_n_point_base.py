# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
import sys
import types

import numpy as np
import pytest

from moldga import brillouin_zone
from moldga import brillouin_zone as bz
from moldga.n_point_base import IHaveChannel, IHaveMat, IAmNonLocal, SpinChannel, FrequencyNotation


# ----- Tests for IHaveMat -----
def test_initializes_with_correct_matrix_and_shape():
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    assert np.allclose(obj.mat, mat, rtol=1e-2)
    assert obj.original_shape == mat.shape


def test_updates_matrix_and_preserves_dtype():
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    assert obj.mat.dtype == np.complex64
    new_mat = np.array([[5, 6], [7, 8]], dtype=np.float64)
    obj.mat = new_mat
    assert obj.mat.dtype == np.complex64


def test_calculates_correct_memory_usage():
    mat = np.zeros((1000, 1000), dtype=np.complex64)
    obj = IHaveMat(mat)
    assert obj.memory_usage_in_gb == pytest.approx(mat.nbytes / (1024**3))


def test_multiplies_with_scalar_correctly():
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    result = obj * 2
    assert np.allclose(result.mat, mat * 2, rtol=1e-2)


def test_raises_error_when_multiplying_with_invalid_type():
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    with pytest.raises(ValueError):
        obj * "invalid"


def test_performs_right_multiplication_with_scalar_correctly():
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    result = 2 * obj
    assert np.allclose(result.mat, mat * 2, rtol=1e-2)


def test_negates_matrix_correctly():
    mat = np.array([[1, -2], [-3, 4]])
    obj = IHaveMat(mat)
    result = -obj
    assert np.allclose(result.mat, -mat, rtol=1e-2)


def test_divides_by_scalar_correctly():
    mat = np.array([[2, 4], [6, 8]])
    obj = IHaveMat(mat)
    result = obj / 2
    assert np.allclose(result.mat, mat / 2, rtol=1e-2)


def test_raises_error_when_dividing_by_invalid_type():
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    with pytest.raises(ValueError):
        obj / "invalid"


def test_reshapes_matrix_and_updates_original_shape():
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    obj.mat = obj.mat.reshape(4, 1)
    obj.update_original_shape()
    assert obj.original_shape == (4, 1)


def test_performs_einsum_contraction_correctly():
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    obj1 = IHaveMat(mat1)
    obj2 = IHaveMat(mat2)
    result = obj1.times("ij,jk->ik", obj2)
    assert np.allclose(result, np.dot(mat1, mat2), rtol=1e-2)


def test_performs_einsum_contraction_with_multiple_matrices():
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    mat3 = np.array([[1, 0], [0, 1]])
    obj1 = IHaveMat(mat1)
    obj2 = IHaveMat(mat2)
    obj3 = IHaveMat(mat3)
    result = obj1.times("ij,jk,kl->il", obj2, obj3)
    assert np.allclose(result, np.dot(np.dot(mat1, mat2), mat3), rtol=1e-2)


def test_raises_error_when_contraction_argument_is_invalid():
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    with pytest.raises(ValueError):
        obj.times("ij,jk->ik", "invalid_argument")


def test_handles_empty_matrices_in_contraction():
    mat1 = np.array([], dtype=np.float64).reshape(0, 0)
    mat2 = np.array([], dtype=np.float64).reshape(0, 0)
    obj1 = IHaveMat(mat1)
    obj2 = IHaveMat(mat2)
    result = obj1.times("ij,jk->ik", obj2)
    assert result.size == 0


def test_performs_einsum_contraction_with_numpy_array():
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    obj = IHaveMat(mat1)
    result = obj.times("ij,jk->ik", mat2)
    assert np.allclose(result, np.dot(mat1, mat2), rtol=1e-2)


def test_raises_error_when_contraction_string_is_invalid():
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    obj = IHaveMat(mat1)
    with pytest.raises(ValueError):
        obj.times("invalid_contraction", mat2)


def test_retrieves_correct_value_for_valid_index():
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    assert obj[0, 1] == 2


def test_sets_value_correctly_for_valid_index():
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    obj[0, 1] = 5
    assert obj[0, 1] == 5


def test_raises_error_for_invalid_index_retrieval():
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    with pytest.raises(IndexError):
        _ = obj[2, 2]


def test_raises_error_for_invalid_index_assignment():
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)
    with pytest.raises(IndexError):
        obj[2, 2] = 5


# ----- Tests for IHaveChannel -----
def test_initializes_with_default_channel_and_frequency_notation():
    obj = IHaveChannel()
    assert obj.channel == SpinChannel.NONE
    assert obj.frequency_notation == FrequencyNotation.PH


def test_initializes_with_provided_channel_and_frequency_notation():
    obj = IHaveChannel(channel=SpinChannel.DENS, frequency_notation=FrequencyNotation.PP)
    assert obj.channel == SpinChannel.DENS
    assert obj.frequency_notation == FrequencyNotation.PP


def test_updates_channel_to_valid_value():
    obj = IHaveChannel()
    obj.channel = SpinChannel.MAGN
    assert obj.channel == SpinChannel.MAGN
    obj.set_channel(SpinChannel.DENS)
    assert obj.channel == SpinChannel.DENS


def test_raises_error_when_setting_invalid_channel():
    obj = IHaveChannel()
    with pytest.raises(ValueError):
        obj.channel = "invalid_channel"
    with pytest.raises(ValueError):
        obj.set_channel("invalid_channel")


def test_updates_frequency_notation_to_valid_value():
    obj = IHaveChannel()
    obj.frequency_notation = FrequencyNotation.PP
    assert obj.frequency_notation == FrequencyNotation.PP
    obj.set_frequency_notation(FrequencyNotation.PH)
    assert obj.frequency_notation == FrequencyNotation.PH


def test_raises_error_when_setting_invalid_frequency_notation():
    obj = IHaveChannel()
    with pytest.raises(ValueError):
        obj.frequency_notation = "invalid_notation"
    with pytest.raises(ValueError):
        obj.set_frequency_notation("invalid_notation")


# ----- Tests for IAmNonLocal -----
def test_initializes_with_correct_matrix_and_momentum_dimensions():
    mat = np.zeros((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    assert np.allclose(obj.mat, mat, rtol=1e-2)
    assert obj.nq == nq
    assert obj.has_compressed_q_dimension is False


def test_initializes_with_compressed_q_dimension():
    mat = np.zeros((64,))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    assert np.allclose(obj.mat, mat, rtol=1e-2)
    assert obj.nq == nq
    assert obj.has_compressed_q_dimension is True


def test_shifts_momentum_by_zero_correctly():
    mat = np.zeros((4, 4, 4))
    obj = IAmNonLocal(mat, (4, 4, 4))
    shifted = obj.shift_k_by_q((0, 0, 0))
    assert np.allclose(shifted.mat, mat, rtol=1e-2)


def test_shifts_momentum_by_positive_values_correctly():
    mat = np.arange(64).reshape((4, 4, 4))
    obj = IAmNonLocal(mat, (4, 4, 4))
    shifted = obj.shift_k_by_q((1, 1, 1))
    expected = np.roll(mat, shift=(-1, -1, -1), axis=(0, 1, 2))
    assert np.allclose(shifted.mat, expected, rtol=1e-2)


def test_shifts_momentum_by_negative_values_correctly():
    mat = np.arange(64).reshape((4, 4, 4))
    obj = IAmNonLocal(mat, (4, 4, 4))
    shifted = obj.shift_k_by_q((-1, -1, -1))
    expected = np.roll(mat, shift=(1, 1, 1), axis=(0, 1, 2))
    assert np.allclose(shifted.mat, expected, rtol=1e-2)


def test_shifts_momentum_with_compressed_q_dimension_correctly():
    mat = np.zeros((64))
    obj = IAmNonLocal(mat, (4, 4, 4), has_compressed_q_dimension=True)
    shifted = obj.shift_k_by_q((1, 1, 1))
    assert shifted.current_shape == (64,)


def test_raises_error_when_shifting_with_invalid_q_length():
    mat = np.zeros((4, 4, 4))
    obj = IAmNonLocal(mat, (4, 4, 4))
    with pytest.raises(ValueError):
        obj.shift_k_by_q((1, 1))


def test_shifts_momentum_by_pi_correctly():
    mat = np.arange(64).reshape((4, 4, 4))
    obj = IAmNonLocal(mat, (4, 4, 4))
    shifted = obj.shift_k_by_pi()
    expected = np.roll(mat, shift=(2, 2, 2), axis=(0, 1, 2))
    assert np.allclose(shifted.mat, expected, rtol=1e-2)


def test_shifts_momentum_by_pi_with_compressed_q_dimension():
    mat = np.arange(64)
    obj = IAmNonLocal(mat, (4, 4, 4), has_compressed_q_dimension=True)
    shifted = obj.shift_k_by_pi()
    assert shifted.has_compressed_q_dimension is True
    assert shifted.mat.shape == mat.shape


def test_raises_error_when_shifting_by_pi_with_invalid_matrix_shape():
    mat = np.zeros((4, 4))
    obj = IAmNonLocal(mat, (4, 4, 4))
    with pytest.raises(ValueError):
        obj.shift_k_by_pi()


def test_compresses_q_dimension_correctly():
    mat = np.zeros((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    obj.compress_q_dimension()
    assert obj.mat.shape == (64,)
    assert obj.has_compressed_q_dimension is True


def test_does_not_compress_already_compressed_q_dimension():
    mat = np.zeros((64,))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    obj.compress_q_dimension()
    assert obj.mat.shape == (64,)
    assert obj.has_compressed_q_dimension is True


def test_compresses_q_dimension_with_additional_dimensions():
    mat = np.zeros((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    obj.compress_q_dimension()
    assert obj.mat.shape == (64, 2)
    assert obj.has_compressed_q_dimension is True


def test_decompresses_q_dimension_correctly():
    mat = np.zeros((64,))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    obj.decompress_q_dimension()
    assert obj.mat.shape == (4, 4, 4)
    assert obj.has_compressed_q_dimension is False


def test_does_not_decompress_if_already_decompressed():
    mat = np.zeros((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    obj.decompress_q_dimension()
    assert obj.mat.shape == (4, 4, 4)
    assert obj.has_compressed_q_dimension is False


def test_decompresses_q_dimension_with_additional_dimensions():
    mat = np.zeros((64, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    obj.decompress_q_dimension()
    assert obj.mat.shape == (4, 4, 4, 2)
    assert obj.has_compressed_q_dimension is False


def test_reduces_q_dimension_to_specified_momenta():
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    q_list = np.array([[1, 1, 1], [2, 2, 2]])
    reduced = obj.reduce_q(q_list)
    assert reduced.mat.shape == (2,)
    assert reduced.has_compressed_q_dimension is True


def test_reduces_q_dimension_with_compressed_input():
    mat = np.arange(64)
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    q_list = np.array([[0, 0, 0], [3, 3, 3]])
    reduced = obj.reduce_q(q_list)
    assert reduced.mat.shape == (2,)
    assert reduced.has_compressed_q_dimension is True


def test_reduce_q_raises_error_when_q_list_has_invalid_shape():
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    q_list = np.array([[0, 0], [3, 3]])
    with pytest.raises(ValueError):
        obj.reduce_q(q_list)


def test_reduces_q_dimension_to_specified_momenta_and_values():
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    q_list = np.array([[1, 1, 1], [2, 2, 2]])
    reduced = obj.reduce_q(q_list)
    expected_values = mat[1, 1, 1], mat[2, 2, 2]
    assert reduced.mat.shape == (2,)
    assert np.allclose(reduced.mat, expected_values, rtol=1e-2)
    assert reduced.has_compressed_q_dimension is True


def test_finds_correct_matrix_element_for_given_momentum():
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.find_q((1, 1, 1))
    assert result.mat.shape == (1,)
    assert result.mat[0] == mat[1, 1, 1]
    assert result.nq == (1, 1, 1)


def test_finds_matrix_element_for_valid_momentum():
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.find_q((2, 2, 2))
    assert result.mat.shape == (1,)
    assert result.mat[0] == mat[2, 2, 2]
    assert result.nq == (1, 1, 1)


def test_raises_error_for_invalid_momentum_shape():
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    with pytest.raises(ValueError):
        obj.find_q((1, 1))


def test_raises_error_for_out_of_bounds_momentum():
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    with pytest.raises(ValueError):
        obj.find_q((5, 5, 5))


def test_maps_to_full_bz_correctly_with_valid_inverse_map():
    mat = np.arange(64)
    nq = (4, 4, 4)
    np.array([0, 1, 2, 3, 4, 5, 6, 7])
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    grid = brillouin_zone.KGrid(nk=(2, 2, 2), symmetries=[])
    obj.map_to_full_bz(grid, nq=(2, 2, 2))
    assert obj.mat.shape == (8,)
    assert obj.nq == (2, 2, 2)


def test_raises_error_when_mapping_to_full_bz_without_compressed_q_dimension():
    mat = np.zeros((4, 4, 4))
    nq = (4, 4, 4)
    inverse_map = np.array([0, 1, 2, 3])
    obj = IAmNonLocal(mat, nq)
    with pytest.raises(ValueError):
        obj.map_to_full_bz(inverse_map)


def test_updates_nq_correctly_when_provided():
    mat = np.arange(64)
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    grid = brillouin_zone.KGrid(nk=(2, 2, 2), symmetries=[])
    obj.map_to_full_bz(grid, nq=(2, 2, 2))
    assert obj.nq == (2, 2, 2)


def test_retains_original_nq_when_not_provided():
    mat = np.arange(64)
    nq = (4, 4, 4)
    grid = brillouin_zone.KGrid(nk=nq, symmetries=[])
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    obj.map_to_full_bz(grid)
    assert obj.nq == (4, 4, 4)


def test_performs_fft_correctly_on_decompressed_matrix():
    mat = np.random.random((4, 4, 4)) + 1j * np.random.random((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.fft()
    expected = np.fft.fftn(mat, axes=(0, 1, 2))
    assert np.allclose(result.mat, expected, rtol=1e-2)
    assert result.has_compressed_q_dimension is False


def test_performs_fft_correctly_on_compressed_matrix():
    mat = np.random.random((64,)) + 1j * np.random.random((64,))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    result = obj.fft()
    decompressed_mat = mat.reshape(nq)
    expected = np.fft.fftn(decompressed_mat, axes=(0, 1, 2)).reshape(64)
    assert np.allclose(result.mat, expected, rtol=1e-2)
    assert result.has_compressed_q_dimension is True


def test_retains_original_shape_after_fft():
    mat = np.random.random((4, 4, 4)) + 1j * np.random.random((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.fft()
    assert result.original_shape == (4, 4, 4)


def test_performs_ifft_correctly_on_decompressed_matrix():
    mat = np.random.random((4, 4, 4)) + 1j * np.random.random((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.ifft()
    expected = np.fft.ifftn(mat, axes=(0, 1, 2))
    assert np.allclose(result.mat, expected, rtol=1e-2)
    assert result.has_compressed_q_dimension is False


def test_performs_ifft_correctly_on_compressed_matrix():
    mat = np.random.random((64,)) + 1j * np.random.random((64,))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    result = obj.ifft()
    decompressed_mat = mat.reshape(nq)
    expected = np.fft.ifftn(decompressed_mat, axes=(0, 1, 2)).reshape(64)
    assert np.allclose(result.mat, expected, rtol=1e-2)
    assert result.has_compressed_q_dimension is True


def test_retains_original_shape_after_ifft():
    mat = np.random.random((4, 4, 4)) + 1j * np.random.random((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.ifft()
    assert result.original_shape == (4, 4, 4)


def test_flips_momentum_axis_correctly_for_decompressed_matrix():
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    flipped = obj.flip_momentum_axis()
    expected = np.roll(np.flip(mat, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
    assert np.allclose(flipped.mat, expected, rtol=1e-2)
    assert flipped.has_compressed_q_dimension is False


def test_flips_momentum_axis_correctly_for_compressed_matrix():
    mat = np.arange(64)
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    flipped = obj.flip_momentum_axis()
    decompressed_mat = mat.reshape(nq)
    expected = np.roll(np.flip(decompressed_mat, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2)).reshape(64)
    assert np.allclose(flipped.mat, expected, rtol=1e-2)
    assert flipped.has_compressed_q_dimension is True


def test_retains_original_shape_after_flipping_momentum_axis():
    mat = np.arange(64).reshape((4, 4, 4))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    flipped = obj.flip_momentum_axis()
    assert flipped.original_shape == (4, 4, 4)


def test_aligns_q_dimensions_when_both_are_decompressed():
    mat1 = np.zeros((4, 4, 4))
    mat2 = np.zeros((4, 4, 4))
    obj1 = IAmNonLocal(mat1, (4, 4, 4))
    obj2 = IAmNonLocal(mat2, (4, 4, 4))
    aligned = obj1._align_q_dimensions_for_operations(obj2)
    assert not obj1.has_compressed_q_dimension
    assert not aligned.has_compressed_q_dimension


def test_aligns_q_dimensions_when_both_are_compressed():
    mat1 = np.zeros((64,))
    mat2 = np.zeros((64,))
    obj1 = IAmNonLocal(mat1, (4, 4, 4), has_compressed_q_dimension=True)
    obj2 = IAmNonLocal(mat2, (4, 4, 4), has_compressed_q_dimension=True)
    aligned = obj1._align_q_dimensions_for_operations(obj2)
    assert obj1.has_compressed_q_dimension
    assert aligned.has_compressed_q_dimension


def test_compresses_self_when_other_is_compressed():
    mat1 = np.zeros((4, 4, 4))
    mat2 = np.zeros((64,))
    obj1 = IAmNonLocal(mat1, (4, 4, 4))
    obj2 = IAmNonLocal(mat2, (4, 4, 4), has_compressed_q_dimension=True)
    aligned = obj1._align_q_dimensions_for_operations(obj2)
    assert obj1.has_compressed_q_dimension
    assert aligned.has_compressed_q_dimension


def test_compresses_other_when_self_is_compressed():
    mat1 = np.zeros((64,))
    mat2 = np.zeros((4, 4, 4))
    obj1 = IAmNonLocal(mat1, (4, 4, 4), has_compressed_q_dimension=True)
    obj2 = IAmNonLocal(mat2, (4, 4, 4))
    aligned = obj1._align_q_dimensions_for_operations(obj2)
    assert obj1.has_compressed_q_dimension
    assert aligned.has_compressed_q_dimension


def test_filter_small_values_sets_tiny_entries_to_zero():
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
    mat = np.array([1e-6 + 1e-6j, 2e-6 + 0.0j, 5e-5 + 1e-8j], dtype=np.complex128)
    obj = IHaveMat(mat)
    obj.filter_small_values(threshold=1e-5)

    # first two entries have both components < 1e-5 -> zeroed
    assert obj.mat[0] == 0.0 + 0.0j
    assert obj.mat[1] == 0.0 + 0.0j
    # last entry has real component above threshold -> preserved
    assert not (obj.mat[2].real == 0.0 and obj.mat[2].imag == 0.0)


def test_filter_small_values_preserves_values_with_one_large_component():
    mat = np.array([1e-13 + 1e-8j, 1e-8 + 1e-13j], dtype=np.complex128)
    obj = IHaveMat(mat)
    obj.filter_small_values(threshold=1e-12)

    # entries with at least one component above threshold must be preserved
    assert not (obj.mat[0].real == 0.0 and obj.mat[0].imag == 0.0)
    assert not (obj.mat[1].real == 0.0 and obj.mat[1].imag == 0.0)


def test_free_releases_underlying_matrix():
    mat = np.array([[1, 2], [3, 4]])
    obj = IHaveMat(mat)

    # ensure matrix is set initially
    assert obj.mat is not None

    # free without trim should release the array
    obj.free(trim=False)
    assert obj.mat is None


def test_free_with_trim_calls_malloc_trim(monkeypatch):
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
    mat = np.array([[1.0]])
    obj = IHaveMat(mat)
    assert obj.__enter__() is obj


def test_exit_calls_free_without_trim(monkeypatch):
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
    assert called.get("trim") is True
    assert obj.mat is None


def test_skip_on_non_posix_or_no_proc(monkeypatch):
    # simulate non-posix or missing /proc -> should mark unavailable and return
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    monkeypatch.setattr(IHaveMat, "_malloc_trim_available", None, raising=False)

    IHaveMat._malloc_trim()

    assert IHaveMat._malloc_trim_available is False


def test_loads_libc_and_calls_malloc_trim(monkeypatch):
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
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index(5)
    assert result.mat.shape == (1, 2)
    assert np.allclose(result.mat[0], mat.reshape(64, 2)[5], rtol=1e-2)


def test_filter_q_index_default_index_is_zero():
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index()
    assert result.mat.shape == (1, 2)
    assert np.allclose(result.mat[0], mat.reshape(64, 2)[0], rtol=1e-2)


def test_filter_q_index_compresses_q_dimension_if_not_already_compressed():
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    assert not obj.has_compressed_q_dimension
    _ = obj.filter_q_index(0)
    assert obj.has_compressed_q_dimension


def test_filter_q_index_does_not_modify_original_when_already_compressed():
    mat = np.arange(64 * 2).reshape((64, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    original_mat = obj.mat.copy()
    _ = obj.filter_q_index(3)
    assert np.allclose(obj.mat, original_mat, rtol=1e-2)


def test_filter_q_index_sets_nq_to_one():
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index(0)
    assert result.nq == (1, 1, 1)


def test_filter_q_index_result_has_compressed_q_dimension():
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index(0)
    assert result.has_compressed_q_dimension


def test_filter_q_index_result_original_shape_is_updated():
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index(0)
    assert result.original_shape == (1, 2)


def test_filter_q_index_returns_deep_copy():
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index(0)
    result.mat[0, 0] = 9999
    assert not np.allclose(obj.mat.reshape(64, 2)[0, 0], 9999, rtol=1e-2)


def test_filter_q_index_last_index():
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    result = obj.filter_q_index(63)
    assert np.allclose(result.mat[0], mat.reshape(64, 2)[63], rtol=1e-2)


def test_filter_q_index_raises_for_out_of_bounds_index():
    mat = np.arange(64 * 2).reshape((4, 4, 4, 2))
    nq = (4, 4, 4)
    obj = IAmNonLocal(mat, nq)
    with pytest.raises(IndexError):
        obj.filter_q_index(64)


# =============================================================================
# _map_to_full_bz: auto-mode branch
# =============================================================================
from unittest.mock import patch

import moldga.symmetry_reduction as _sr


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
    """With a legacy (non-auto) KGrid, ``_map_to_full_bz`` reduces to a bare
    IBZ→FBZ index expansion via ``irrk_inv``: each FBZ point gets the IBZ value
    at the index pointed to by ``irrk_inv``, with no orbital transformation."""
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
    """Same as above but with multiple orbitals — exercises the orbital einsum path."""
    grid, H = _build_auto_kgrid(nx=4, ny=4, nz=4, nb=2)
    nb = 2
    H_flat = H.reshape(-1, nb, nb)
    H_ibz = H_flat[grid.irrk_ind].copy()
    obj = _DoublePrecisionNonLocal(mat=H_ibz, nq=(4, 4, 4), has_compressed_q_dimension=True)
    obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    H_rec = obj.mat.reshape(4, 4, 4, nb, nb)
    assert np.allclose(H_rec, H, atol=1e-12)


def test_map_to_full_bz_auto_4idx_reconstructs_HotimesH_exactly():
    """For Γ = H ⊗ H (which inherits H's symmetry trivially), reconstruction must
    be exact under the 4-orbital-index code path."""
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
    """The mapping is shape-polymorphic in the trailing axes (e.g. frequency axes).
    A 1-band IBZ payload with 2 frequency axes after the orbital pair must come
    back to the full BZ unmodified beyond the index expansion."""
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


def test_map_to_full_bz_auto_with_antiunitary_does_apply_conjugation():
    """Opting into ``include_antiunitary=True`` produces conj=True at some FBZ
    points, and the FBZ expansion then conjugates orbital values at those points.
    For a complex IBZ payload, the result at those points must equal the
    conjugate of the corresponding IBZ value."""
    nb = 1
    grid, _ = _build_auto_kgrid(nx=4, ny=4, nz=1, nb=nb, include_antiunitary=True)
    assert int(grid._auto_conjs.sum()) > 0, "expected at least one conj=True point"

    # Build a complex IBZ payload so conjugation has a visible effect
    rng = np.random.default_rng(1)
    ibz_payload = rng.standard_normal((grid.nk_irr, nb, nb)) + 1j * rng.standard_normal((grid.nk_irr, nb, nb))
    obj = _DoublePrecisionNonLocal(mat=ibz_payload.copy(), nq=(4, 4, 1), has_compressed_q_dimension=True)
    obj._map_to_full_bz(grid, num_orbital_dimensions=2)

    inv = grid.irrk_inv.ravel()
    conjs = grid._auto_conjs.reshape(-1)
    # Expected: IBZ-replicated, with conj applied where conjs is True (since U=[[1]] for nb=1)
    expected = ibz_payload[inv].copy()
    expected[conjs] = expected[conjs].conj()
    assert np.allclose(obj.mat, expected, atol=1e-14)


def test_map_to_full_bz_auto_default_no_antiunitary_does_no_conjugation():
    """Default (include_antiunitary=False): no FBZ point should ever be conjugated,
    so a complex IBZ payload reconstructs as a pure index replication. This is
    the safe semantics for frequency-dependent objects."""
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
    """The auto branch must call ``symmetry_reduction.apply_auto_orbital_transform``
    with the correctly-sliced (Us, sigmas, conjs) arrays and the right ndim."""
    grid, H = _build_auto_kgrid(nx=4, ny=4, nz=4, nb=2)
    nb = 2
    nktot = 4 * 4 * 4
    H_flat = H.reshape(-1, nb, nb)
    H_ibz = H_flat[grid.irrk_ind].copy()
    obj = _DoublePrecisionNonLocal(mat=H_ibz, nq=(4, 4, 4), has_compressed_q_dimension=True)

    # Patch so we can assert it gets called with the right shapes and args
    with patch.object(_sr, "apply_auto_orbital_transform", wraps=_sr.apply_auto_orbital_transform) as spy:
        obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    assert spy.call_count == 1
    _, kwargs = spy.call_args
    # The function was called with keyword arguments matching the signature
    assert kwargs["num_orbital_dimensions"] == 2
    assert kwargs["us"].shape == (nktot, nb, nb)
    assert kwargs["sigmas"].shape == (nktot,)
    assert kwargs["conjs"].shape == (nktot,)


def test_map_to_full_bz_auto_passes_num_orbital_dimensions_4_for_vertex():
    grid, H = _build_auto_kgrid(nx=3, ny=3, nz=3, nb=2)
    nb = 2
    Gamma_full = np.einsum("...ab,...cd->...abcd", H, H)
    Gamma_ibz = Gamma_full.reshape(-1, nb, nb, nb, nb)[grid.irrk_ind].copy()
    obj = _DoublePrecisionNonLocal(mat=Gamma_ibz, nq=(3, 3, 3), has_compressed_q_dimension=True)
    with patch.object(_sr, "apply_auto_orbital_transform", wraps=_sr.apply_auto_orbital_transform) as spy:
        obj._map_to_full_bz(grid, num_orbital_dimensions=4)
    _, kwargs = spy.call_args
    assert kwargs["num_orbital_dimensions"] == 4


def test_map_to_full_bz_legacy_kgrid_does_not_call_orbital_transform():
    """For a legacy KGrid (not auto-mode), the orbital transform helper must NOT
    be called: only the IBZ→FBZ replication runs."""
    grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    nb = 1
    ibz_payload = np.arange(grid.nk_irr).astype(np.complex128).reshape(grid.nk_irr, nb, nb)
    obj = _DoublePrecisionNonLocal(mat=ibz_payload.copy(), nq=(4, 4, 1), has_compressed_q_dimension=True)
    with patch.object(_sr, "apply_auto_orbital_transform", wraps=_sr.apply_auto_orbital_transform) as spy:
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
    """The compressed-q convention is required: an already-expanded matrix is
    not a valid input to ``_map_to_full_bz``."""
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
    grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    nb = 1
    ibz_payload = np.arange(grid.nk_irr).astype(np.complex128).reshape(grid.nk_irr, nb, nb)
    obj = _DoublePrecisionNonLocal(mat=ibz_payload.copy(), nq=(4, 4, 1), has_compressed_q_dimension=True)
    result = obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    assert result is obj


def test_map_to_full_bz_auto_1x1x1_trivial_grid_is_identity():
    """Edge case: a 1×1×1 grid has a single k-point, so the FBZ trivially equals the
    IBZ and the mapping returns the input unchanged in value."""
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
    """The output matrix has the same dtype as the input (the function does not
    silently cast within the auto branch — the cast to complex64 happens elsewhere
    in ``IHaveMat.mat = value``)."""
    grid, H = _build_auto_kgrid(nx=4, ny=4, nz=4, nb=1)
    H_ibz_64 = H.reshape(-1, 1, 1)[grid.irrk_ind].astype(np.complex64).copy()
    obj = _DoublePrecisionNonLocal(mat=H_ibz_64, nq=(4, 4, 4), has_compressed_q_dimension=True)
    obj._map_to_full_bz(grid, num_orbital_dimensions=2)
    assert obj.mat.dtype == np.complex64


def test_map_to_full_bz_auto_irrk_inv_consistency_at_every_fbz_point():
    """Every FBZ k must end up with the value at irrk_inv[k] transformed by the
    stored (U_k, sigma_k, conj_k). Check this explicitly point-by-point."""
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

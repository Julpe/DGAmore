import numpy as np
import pytest

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
    mat = np.zeros((64,))
    obj = IAmNonLocal(mat, (4, 4, 4), has_compressed_q_dimension=True)
    shifted = obj.shift_k_by_q((1, 1, 1))
    assert shifted.current_shape == (4, 4, 4)


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
    inverse_map = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    obj.map_to_full_bz(inverse_map, nq=(2, 2, 2))
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
    inverse_map = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    obj.map_to_full_bz(inverse_map, nq=(2, 2, 2))
    assert obj.nq == (2, 2, 2)


def test_retains_original_nq_when_not_provided():
    mat = np.arange(64)
    nq = (4, 4, 4)
    inverse_map = np.arange(64) - 64
    obj = IAmNonLocal(mat, nq, has_compressed_q_dimension=True)
    obj.map_to_full_bz(inverse_map)
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

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

from copy import deepcopy

import numpy as np
import pytest

from dgamore.four_point import FourPoint
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.n_point_base import SpinChannel, FrequencyNotation, IAmNonLocal


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def small_fourpoint(rng):
    nq = (4, 4, 1)
    o = 2
    niw = 3
    niv = 3
    shape = (*nq, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    fp = FourPoint(
        mat=mat,
        channel=SpinChannel.DENS,
        nq=nq,
        num_wn_dimensions=1,
        num_vn_dimensions=2,
        full_niw_range=True,
        full_niv_range=True,
        has_compressed_q_dimension=False,
        frequency_notation=FrequencyNotation.PH,
    )
    return fp


@pytest.fixture
def small_fourpoint_compressed(rng):
    nq = (4, 4, 1)
    qtot = int(np.prod(nq))
    o = 2
    niw = 3
    niv = 3
    shape = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    fp = FourPoint(
        mat=mat,
        channel=SpinChannel.MAGN,
        nq=nq,
        num_wn_dimensions=1,
        num_vn_dimensions=2,
        full_niw_range=True,
        full_niv_range=True,
        has_compressed_q_dimension=True,
        frequency_notation=FrequencyNotation.PH,
    )
    return fp


def test_basic_init_and_properties(small_fourpoint):
    """A FourPoint exposes its momentum grid, dimension counts and band count."""
    fp = small_fourpoint
    assert isinstance(fp, IAmNonLocal)
    assert fp.nq == (4, 4, 1)
    assert fp.nq_tot == 16
    assert fp.num_wn_dimensions == 1
    assert fp.num_vn_dimensions == 2
    assert fp.n_bands == 2


def test_add_scalar_and_numpy(small_fourpoint):
    """Adding a scalar or a numpy array adds elementwise to the FourPoint matrix."""
    fp = small_fourpoint
    val = 2.5
    res = fp + val
    assert isinstance(res, FourPoint)
    assert np.allclose(res.mat, fp.mat + val)

    arr = np.ones_like(fp.mat)
    res2 = fp + arr
    assert np.allclose(res2.mat, fp.mat + arr)


def test_sub_operator(small_fourpoint):
    """Subtraction works in both fp - scalar and scalar - fp orders."""
    fp = small_fourpoint
    res = fp - 1.0
    assert np.allclose(res.mat, fp.mat - 1.0)

    res2 = 1.0 - fp
    assert np.allclose(res2.mat, 1.0 - fp.mat)


def test_mul_with_scalar_and_array(small_fourpoint):
    """Multiplying by a scalar or array scales the FourPoint matrix elementwise."""
    fp = small_fourpoint
    res = fp * 3.0
    assert np.allclose(res.mat, fp.mat * 3.0)

    arr = np.full_like(fp.mat, 2.0)
    res2 = 2.0 * fp
    assert np.allclose(res2.mat, fp.mat * 2.0)


def test_add_with_localinteraction_and_interaction(rng):
    """A FourPoint adds a LocalInteraction and a momentum-dependent Interaction, preserving its shape."""
    nq = (4, 4, 1)
    qtot = 4
    o = 2
    niw = 2
    niv = 2
    shape_fp = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    fp = FourPoint(
        rng.standard_normal(shape_fp) + 1j * rng.standard_normal(shape_fp),
        nq=nq,
        num_wn_dimensions=1,
        num_vn_dimensions=2,
        has_compressed_q_dimension=True,
    )

    u_loc = rng.standard_normal((o, o, o, o)) + 1j * rng.standard_normal((o, o, o, o))
    u_loc = LocalInteraction(u_loc)
    res1 = fp + u_loc
    assert res1.current_shape == fp.current_shape

    u_q = rng.standard_normal((qtot, o, o, o, o)) + 1j * rng.standard_normal((qtot, o, o, o, o))
    u_q = Interaction(u_q, nq=nq, has_compressed_q_dimension=True)
    res2 = fp + u_q
    assert res2.current_shape == fp.current_shape


def test_add_two_fourpoints_same_shape_and_ranges(rng):
    """Adding two FourPoints of equal shape and range adds their matrices."""
    nq = (2, 2, 1)
    qtot = 4
    o = 2
    niw = 2
    niv = 2
    shape = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    mat1 = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    mat2 = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    fp1 = FourPoint(mat1, nq=nq, num_vn_dimensions=2, has_compressed_q_dimension=True, full_niw_range=False)
    fp2 = FourPoint(mat2, nq=nq, num_vn_dimensions=2, has_compressed_q_dimension=True, full_niw_range=False)

    res = fp1 + fp2
    assert np.allclose(res.mat, mat1 + mat2, atol=1e-6)
    assert isinstance(res, FourPoint)
    assert res.current_shape == fp1.current_shape


def test_add_two_fourpoints_different_full_half_ranges(rng):
    """Adding a full-niw and a half-niw FourPoint yields a half-niw result."""
    nq = (2, 2, 1)
    qtot = 4
    o = 2
    niw = 2
    niv = 2
    shape_full = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    mat1 = rng.standard_normal(shape_full) + 1j * rng.standard_normal(shape_full)
    mat2 = rng.standard_normal(shape_full) + 1j * rng.standard_normal(shape_full)
    fp1 = FourPoint(mat1, nq=nq, num_vn_dimensions=2, full_niw_range=True, has_compressed_q_dimension=True)
    fp2 = FourPoint(
        mat2[:, :, :, :, :, niw:, :, :],
        nq=nq,
        num_vn_dimensions=2,
        full_niw_range=False,
        has_compressed_q_dimension=True,
    )

    res = fp1 + fp2
    assert res.full_niw_range is False
    assert res.current_shape[5] == niw + 1


def test_add_two_fourpoints_mismatched_vn_dims_promotes_correctly(rng):
    """Adding FourPoints with 1 and 2 fermionic-frequency dimensions promotes to 2."""
    nq = (1, 1, 1)
    o = 2
    niw = 2
    niv = 2
    shape1 = (1, o, o, o, o, 2 * niw + 1, 2 * niv)
    shape2 = (1, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    mat1 = rng.standard_normal(shape1) + 1j * rng.standard_normal(shape1)
    mat2 = rng.standard_normal(shape2) + 1j * rng.standard_normal(shape2)
    fp1 = FourPoint(mat1, nq=nq, num_vn_dimensions=1, has_compressed_q_dimension=True)
    fp2 = FourPoint(mat2, nq=nq, num_vn_dimensions=2, has_compressed_q_dimension=True)

    res = fp1 + fp2
    assert res.num_vn_dimensions == 2
    assert res.current_shape[0] == 1


def test_add_two_fourpoints_different_q_compression(rng):
    """Adding a compressed and a decompressed FourPoint compresses and adds correctly."""
    nq = (2, 2, 1)
    o = 2
    niw = 2
    niv = 2
    shape_compr = (np.prod(nq), o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    shape_decomp = (*nq, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)

    mat1 = rng.standard_normal(shape_compr) + 1j * rng.standard_normal(shape_compr)
    mat2 = rng.standard_normal(shape_decomp) + 1j * rng.standard_normal(shape_decomp)

    fp1 = FourPoint(mat1, nq=nq, num_vn_dimensions=2, has_compressed_q_dimension=True, full_niw_range=False)
    fp2 = FourPoint(mat2, nq=nq, num_vn_dimensions=2, has_compressed_q_dimension=False, full_niw_range=False)

    res = fp1 + fp2
    assert res.has_compressed_q_dimension
    assert res.current_shape[0] == np.prod(nq)
    assert np.allclose(res.mat, fp1.mat + fp2.compress_q_dimension().mat, atol=1e-6)


def test_sum_over_vn_reduces_dims(small_fourpoint):
    """sum_over_vn reduces a fermionic-frequency dimension with the 1/beta factor."""
    fp = small_fourpoint.to_half_niw_range()
    beta = 10.0
    out = fp.sum_over_vn(beta=beta, axis=(-1,))
    assert isinstance(out, FourPoint)
    assert out.num_vn_dimensions == 1

    sl = fp.mat[0, 0, 0, 0, 0, 0, :, :, :]
    expect = (1 / beta) * np.sum(sl, axis=-1)
    got = out.mat[0, 0, 0, 0, 0, 0, :, :]
    assert np.allclose(got, expect, rtol=1e-6, atol=1e-6)


def test_sum_over_vn_raises_on_too_many_axes(small_fourpoint):
    """sum_over_vn raises when asked to sum over more axes than exist."""
    with pytest.raises(ValueError):
        _ = small_fourpoint.sum_over_vn(beta=1.0, axis=(-1, -2, -3))


def test_sum_over_orbitals_valid_and_invalid(small_fourpoint):
    """sum_over_orbitals contracts a valid orbital string and rejects malformed ones."""
    fp = small_fourpoint.to_half_niw_range()
    out = fp.sum_over_orbitals("abcd->ad")
    assert out._num_orbital_dimensions == 2

    with pytest.raises(ValueError):
        _ = small_fourpoint.sum_over_orbitals("abc->a")
    with pytest.raises(ValueError):
        _ = small_fourpoint.sum_over_orbitals("abcd->abcde")


def test_permute_orbitals_noop_and_swap(small_fourpoint):
    """permute_orbitals short-circuits the identity and applies an orbital swap."""
    fp = small_fourpoint
    out = fp.permute_orbitals("abcd->abcd")
    assert out is fp
    assert np.allclose(out.mat, fp.mat)

    out2 = fp.permute_orbitals("abcd->badc")
    idx_src = (0, 0, 0, 1, 0, 1, 0, slice(None), slice(None), slice(None))
    idx_dst = (0, 0, 0, 0, 1, 0, 1, slice(None), slice(None), slice(None))
    assert np.allclose(out2.mat[idx_dst], fp.mat[idx_src])


def test_permute_orbitals_invalid_strings_raise(small_fourpoint):
    """permute_orbitals rejects malformed permutation strings."""
    with pytest.raises(ValueError):
        _ = small_fourpoint.permute_orbitals("abc->abc")
    with pytest.raises(ValueError):
        _ = small_fourpoint.permute_orbitals("abcd->abc")


def test_to_compound_indices_and_back_with_two_vn_dims(small_fourpoint_compressed):
    """to_compound_indices / to_full_indices round-trip a two-vn FourPoint."""
    fp = small_fourpoint_compressed
    fp_half = fp.to_half_niw_range()
    fp_ci = fp_half.to_compound_indices()
    assert fp_ci.has_compressed_q_dimension
    assert len(fp_ci.current_shape) == 4
    back = fp_ci.to_full_indices(fp.original_shape)
    assert back.has_compressed_q_dimension is True
    assert back.current_shape == fp.original_shape


def test_to_compound_indices_vn2_vs_vn1_vs_vn0(rng):
    """to_compound_indices handles 0, 1 and 2 fermionic-frequency dimensions."""
    nq = (1, 1, 1)
    qtot = 1
    o = 2
    niw = 2
    niv = 2

    shape2 = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    mat2 = rng.standard_normal(shape2) + 1j * rng.standard_normal(shape2)
    fp2 = FourPoint(mat2, nq=nq, num_vn_dimensions=2, has_compressed_q_dimension=True)
    fp2_ci = fp2.to_compound_indices()
    assert len(fp2_ci.current_shape) == 4
    assert fp2_ci.current_shape[2] == fp2_ci.current_shape[3]

    shape1 = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv)
    mat1 = rng.standard_normal(shape1) + 1j * rng.standard_normal(shape1)
    fp1 = FourPoint(mat1, nq=nq, num_vn_dimensions=1, has_compressed_q_dimension=True)
    fp1_ci = fp1.to_compound_indices()
    assert len(fp1_ci.current_shape) == 4
    assert fp1_ci.num_vn_dimensions == 2

    shape0 = (qtot, o, o, o, o, 2 * niw + 1)
    mat0 = rng.standard_normal(shape0) + 1j * rng.standard_normal(shape0)
    fp0 = FourPoint(mat0, nq=nq, num_vn_dimensions=0, has_compressed_q_dimension=True)
    fp0_ci = fp0.to_compound_indices()
    assert len(fp0_ci.current_shape) == 4
    assert fp0_ci.num_vn_dimensions == 0


def test_to_compound_indices_decompressed_vs_compressed(rng):
    """to_compound_indices compresses the q dimension of a decompressed FourPoint."""
    nq = (2, 2, 1)
    o = 2
    niw = 2
    niv = 2
    shape_decomp = (*nq, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    mat = rng.standard_normal(shape_decomp) + 1j * rng.standard_normal(shape_decomp)

    fp = FourPoint(mat, nq=nq, num_vn_dimensions=2, has_compressed_q_dimension=False)
    fp_ci = fp.to_half_niw_range().to_compound_indices()
    assert fp_ci.has_compressed_q_dimension
    assert fp_ci.current_shape[2] == fp_ci.current_shape[3]


def test_to_compound_indices_no_wn_dim_raises_if_vn_not_2(rng):
    """to_compound_indices with no bosonic dimension requires exactly two fermionic ones."""
    nq = (1, 1, 1)
    o = 2
    niv = 2
    mat = rng.standard_normal((1, o, o, o, o, 2 * niv)) + 1j * rng.standard_normal((1, o, o, o, o, 2 * niv))
    fp = FourPoint(
        mat=mat,
        channel=SpinChannel.DENS,
        nq=nq,
        num_wn_dimensions=0,
        num_vn_dimensions=1,
        full_niw_range=True,
        full_niv_range=True,
        has_compressed_q_dimension=True,
        frequency_notation=FrequencyNotation.PH,
    )
    with pytest.raises(ValueError):
        _ = fp.to_compound_indices()


def test_to_full_indices_round_trip_with_explicit_shape(rng):
    """to_full_indices restores the original shape from an explicit shape argument."""
    nq = (2, 2, 1)
    qtot = 4
    o = 2
    niw = 2
    niv = 2
    shape = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    fp = FourPoint(mat, nq=nq, num_vn_dimensions=2, has_compressed_q_dimension=True)
    fp_ci = fp.to_half_niw_range().to_compound_indices()
    shape = fp_ci.original_shape
    fp_full = fp_ci.to_full_indices(shape)
    assert fp_full.current_shape == shape


def test_to_full_indices_with_incorrect_shape_argument(rng):
    """to_full_indices raises for an incorrect explicit shape argument."""
    nq = (2, 2, 1)
    qtot = 4
    o = 2
    niw = 2
    niv = 2
    shape = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    fp = FourPoint(mat, nq=nq, num_vn_dimensions=2, has_compressed_q_dimension=True)
    fp_ci = fp.to_half_niw_range().to_compound_indices()
    wrong_shape = shape[:-1]
    with pytest.raises(ValueError):
        fp_ci.to_full_indices(wrong_shape)


def test_to_full_indices_invalid_shape_raises(small_fourpoint_compressed):
    """to_full_indices raises when the q-compression flag is inconsistent."""
    fp = small_fourpoint_compressed.to_compound_indices()
    fp._has_compressed_q_dimension = False
    with pytest.raises(ValueError):
        _ = fp.to_full_indices()


def test_to_full_indices_requires_one_wn_dim(small_fourpoint_compressed):
    """to_full_indices requires exactly one bosonic-frequency dimension."""
    fp = small_fourpoint_compressed.to_half_niw_range().to_compound_indices()
    fp._num_wn_dimensions = 0
    with pytest.raises(ValueError):
        _ = fp.to_full_indices()


def test_to_full_indices_sets_real_compression_flag_no_vn():
    """to_full_indices sets the real q-compression flag when num_vn_dimensions == 0."""
    nb, niw = 1, 1
    mat = np.ones((2, 2, 1, nb, nb, nb, nb, 2 * niw + 1))
    fp = FourPoint(mat, SpinChannel.DENS, (2, 2, 1), num_wn_dimensions=1, num_vn_dimensions=0)
    fp.to_compound_indices().to_full_indices()
    assert fp.has_compressed_q_dimension is True
    assert not hasattr(fp, "_has_compressed_momentum_dimension")  # typo attribute must not exist


def test_matmul_with_localinteraction_left_and_right(rng):
    """A FourPoint matrix-multiplies a LocalInteraction from the left and the right."""
    nq = (2, 2, 1)
    qtot = 4
    o = 2
    niw = 2
    niv = 2
    shape_fp = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv)
    fp_mat = rng.standard_normal(shape_fp) + 1j * rng.standard_normal(shape_fp)
    fp = FourPoint(
        fp_mat,
        channel=SpinChannel.DENS,
        nq=nq,
        num_wn_dimensions=1,
        num_vn_dimensions=1,
        has_compressed_q_dimension=True,
    )

    u = rng.standard_normal((o, o, o, o)) + 1j * rng.standard_normal((o, o, o, o))
    u_loc = LocalInteraction(u)

    out1 = fp @ u_loc
    assert isinstance(out1, FourPoint)
    assert out1.current_shape[:1] == (qtot,)
    out2 = u_loc @ fp
    assert isinstance(out2, FourPoint)
    assert out2.current_shape[:1] == (qtot,)


def test_matmul_fourpoint_vs_fourpoint_mixed_vn_dims(rng):
    """Matrix-multiplying FourPoints with mixed fermionic dimensions yields a two-vn result."""
    nq = (2, 2, 1)
    qtot = 4
    o = 2
    niw = 2
    niv = 2
    shape_left = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv)
    shape_right = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    lhs = FourPoint(
        rng.standard_normal(shape_left) + 1j * rng.standard_normal(shape_left),
        nq=nq,
        num_wn_dimensions=1,
        num_vn_dimensions=1,
        has_compressed_q_dimension=True,
    )
    rhs = FourPoint(
        rng.standard_normal(shape_right) + 1j * rng.standard_normal(shape_right),
        nq=nq,
        num_wn_dimensions=1,
        num_vn_dimensions=2,
        has_compressed_q_dimension=True,
    )
    out = lhs @ rhs
    assert out.num_vn_dimensions == 2
    assert out.current_shape[0] == qtot
    assert len(out.current_shape) == 1 + 4 + 1 + 2


def test_identity_shapes_and_like(rng):
    """identity and identity_like build a FourPoint matching the requested or target metadata."""
    n_bands = 2
    niw = 2
    niv = 2
    nq = (4, 4, 1)
    qtot = 4
    I = FourPoint.identity(n_bands=n_bands, niw=niw, niv=niv, nq_tot=qtot, nq=nq, num_vn_dimensions=2)
    assert isinstance(I, FourPoint)
    assert I.current_shape[1 + 4] == niw + 1

    shape = (qtot, n_bands, n_bands, n_bands, n_bands, 2 * niw + 1, 2 * niv, 2 * niv)
    target = FourPoint(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        nq=nq,
        num_vn_dimensions=2,
        has_compressed_q_dimension=True,
    )
    I2 = FourPoint.identity_like(target)
    assert I2.n_bands == target.n_bands
    assert I2.nq_tot == target.nq_tot
    assert I2.num_vn_dimensions == target.num_vn_dimensions


def test_four_point_identity_is_valid_and_complex64():
    """FourPoint.identity is built directly in the complex64 storage dtype."""
    ident = FourPoint.identity(1, 1, 2, nq_tot=1, nq=(1, 1, 1), num_vn_dimensions=2)
    assert ident.mat.dtype == np.complex64


def test_flip_axes_helpers_from_base_do_not_break(small_fourpoint_compressed):
    """The base shift_k_by_pi and flip_momentum_axis helpers preserve the shape."""
    fp = small_fourpoint_compressed
    out = fp.shift_k_by_pi()
    assert out.current_shape == fp.current_shape
    out2 = fp.flip_momentum_axis()
    assert out2.current_shape == fp.current_shape


def test_invert_num_vn1_per_q_matches_batched_reference(rng):
    """The per-q num_vn==1 invert matches a single batched np.linalg.inv reference."""
    nq = (3, 2, 1)
    qtot = int(np.prod(nq))
    o, niw, niv = 2, 2, 2
    shape = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv)
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    fp = FourPoint(
        mat,
        nq=nq,
        num_wn_dimensions=1,
        num_vn_dimensions=1,
        has_compressed_q_dimension=True,
        full_niw_range=True,
    )

    result = deepcopy(fp).invert(copy=True)

    # Reference: replicate the production compound transform but with a single batched np.linalg.inv.
    ref = deepcopy(fp).to_half_niw_range()
    w_dim = ref.original_shape[5]
    ref.compress_q_dimension()
    comp = ref.mat.transpose(0, 5, 6, 1, 2, 4, 3).reshape((ref.current_shape[0], w_dim, 2 * ref.niv, o**2, o**2))
    comp = np.linalg.inv(comp)
    ref_mat = comp.reshape((ref.current_shape[0], w_dim, 2 * ref.niv, o, o, o, o)).transpose(0, 3, 4, 6, 5, 1, 2)

    assert result.num_vn_dimensions == 1
    assert not result.full_niw_range  # invert returns the half niw range
    assert result.mat.dtype == np.complex64
    assert np.allclose(result.mat, ref_mat)


def test_invert_and_sum_methods_agree_on_decompressed_fp(small_fourpoint):
    """Both invert_and_sum_over_last_vn variants agree on a decompressed FourPoint."""
    fp = deepcopy(small_fourpoint)
    beta = 7.0

    out1 = fp.invert_and_sum_over_last_vn(beta)
    fp2 = deepcopy(small_fourpoint)
    out2 = fp2.invert_and_sum_over_last_vn_v2(beta)

    assert out1._num_vn_dimensions == 1
    assert out2._num_vn_dimensions == 1
    assert out1.current_shape == out2.current_shape
    assert np.allclose(out1.mat, out2.mat, rtol=1e-6, atol=1e-8)


def test_invert_and_sum_methods_agree_on_compressed_fp(small_fourpoint_compressed):
    """Both invert_and_sum_over_last_vn variants agree on a compressed FourPoint."""
    fp = deepcopy(small_fourpoint_compressed)
    beta = 11.0

    out1 = fp.invert_and_sum_over_last_vn(beta)
    fp2 = deepcopy(small_fourpoint_compressed)
    out2 = fp2.invert_and_sum_over_last_vn_v2(beta)

    assert out1._num_vn_dimensions == 1
    assert out2._num_vn_dimensions == 1
    assert out1.current_shape == out2.current_shape
    assert np.allclose(out1.mat, out2.mat, rtol=1e-6, atol=1e-8)


def test_invert_and_sum_scales_with_beta(small_fourpoint_compressed):
    """invert_and_sum_over_last_vn_v2 scales as 1/beta."""
    fp = deepcopy(small_fourpoint_compressed)

    beta1 = 1.0
    beta2 = 5.0

    out_beta1 = deepcopy(fp).invert_and_sum_over_last_vn_v2(beta1)
    out_beta2 = deepcopy(fp).invert_and_sum_over_last_vn_v2(beta2)

    scale = beta1 / beta2
    assert np.allclose(out_beta2.mat, out_beta1.mat * scale, rtol=1e-8, atol=1e-10)


def test_invert_and_sum_v1_scales_with_beta(small_fourpoint_compressed):
    """invert_and_sum_over_last_vn keeps the exact 1/beta scaling."""
    fp = small_fourpoint_compressed
    out_beta1 = deepcopy(fp).invert_and_sum_over_last_vn(1.0)
    out_beta2 = deepcopy(fp).invert_and_sum_over_last_vn(2.0)
    assert np.allclose(out_beta1.mat, 2.0 * out_beta2.mat)


def test_invert_and_sum_matches_manual_small_case():
    """invert_and_sum variants match a manual invert-then-sum on a small case."""
    rng_local = np.random.default_rng(42)
    nq_tot = 1
    o = 2
    niw = 1
    niv = 1
    shape = (nq_tot, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    mat = rng_local.standard_normal(shape) + 1j * rng_local.standard_normal(shape)

    fp = FourPoint(
        mat,
        nq=(1, 1, 1),
        num_wn_dimensions=1,
        num_vn_dimensions=2,
        has_compressed_q_dimension=True,
        full_niw_range=True,
    )

    beta = 3.0

    computed = deepcopy(fp).invert(False).sum_over_vn(beta, axis=(-1,))
    computed_v1 = deepcopy(fp).invert_and_sum_over_last_vn(beta)
    computed_v2 = deepcopy(fp).invert_and_sum_over_last_vn_v2(beta)
    assert np.allclose(computed.mat, computed_v1.mat)
    assert np.allclose(computed.mat, computed_v2.mat)
    assert computed._num_vn_dimensions == 1


def test_invert_and_sum_and_invert_preserve_complex64(small_fourpoint_compressed):
    """invert and invert_and_sum variants preserve the complex64 dtype."""
    fp = small_fourpoint_compressed
    assert deepcopy(fp).invert_and_sum_over_last_vn(2.0).mat.dtype == np.complex64
    assert deepcopy(fp).invert_and_sum_over_last_vn_v2(2.0).mat.dtype == np.complex64
    assert deepcopy(fp).invert().mat.dtype == np.complex64

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
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


def test_sub_fourpoint_equals_add_of_negated(rng):
    """a - b (direct subtract via _add, no negated copy) bit-equals the reference a + (-b) for two vertices."""
    nq = (4, 4, 1)
    o, niw, niv = 2, 3, 3
    shape = (*nq, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)

    def _fp():
        mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        return FourPoint(mat, SpinChannel.DENS, nq, 1, 2, True, True, False, FrequencyNotation.PH)

    a, b = _fp(), _fp()
    res_ref = deepcopy(a) + (-deepcopy(b))
    res_sub = a - b
    assert np.array_equal(res_sub.mat, res_ref.mat)


def test_mul_with_scalar_and_array(small_fourpoint):
    """Multiplying by a scalar or array scales the FourPoint matrix elementwise."""
    fp = small_fourpoint
    res = fp * 3.0
    assert np.allclose(res.mat, fp.mat * 3.0)

    arr = np.full_like(fp.mat, 2.0)
    res2 = 2.0 * fp
    assert np.allclose(res2.mat, fp.mat * 2.0)


def test_scale_in_place_multiplies_and_returns_self(small_fourpoint):
    """scale(factor) multiplies mat in place (copy=False default), returns self, and stays complex64."""
    fp = small_fourpoint
    before = fp.mat.copy()
    out = fp.scale(3.0)
    assert out is fp
    assert np.array_equal(fp.mat, (before * 3.0).astype(np.complex64))
    assert fp.mat.dtype == np.complex64


def test_scale_copy_true_leaves_original_untouched(small_fourpoint):
    """scale(factor, copy=True) returns a new scaled object and leaves self unchanged."""
    fp = small_fourpoint
    before = fp.mat.copy()
    out = fp.scale(-2.0, copy=True)
    assert out is not fp
    assert np.array_equal(out.mat, (before * -2.0).astype(np.complex64))
    assert np.array_equal(fp.mat, before)


def test_scale_rejects_non_scalar(small_fourpoint):
    """scale only accepts numbers (mirrors __mul__)."""
    with pytest.raises(ValueError):
        small_fourpoint.scale(np.ones(3))


def test_copy_returns_independent_deep_copy(small_fourpoint):
    """copy() returns a new, independent deep copy whose mutation does not affect the original."""
    fp = small_fourpoint
    c = fp.copy()
    assert c is not fp
    assert c.mat is not fp.mat
    assert np.array_equal(c.mat, fp.mat)
    before = fp.mat.copy()
    c.mat[...] = 0.0
    assert np.array_equal(fp.mat, before)


def _kernel_block(rng, channel, num_vn=1):
    """Builds a half-niw, compressed-q FourPoint matching the self-energy-kernel layout (num_vn 1 or 2)."""
    nq = (4, 4, 1)
    qtot, o, niw, niv = 16, 2, 3, 3
    shape = (qtot, o, o, o, o, niw + 1) + (2 * niv,) * num_vn
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return FourPoint(mat, channel, nq, 1, num_vn, False, True, True, FrequencyNotation.PH)


def test_add_inplace_equals_out_of_place_and_returns_self(rng):
    """add(other, copy=False) accumulates into self in place, returns self, bit-equal to self + other."""
    a, b = _kernel_block(rng, SpinChannel.NONE), _kernel_block(rng, SpinChannel.DENS)
    ref = deepcopy(a) + deepcopy(b)
    out = a.add(b, copy=False)
    assert out is a
    assert np.array_equal(a.mat, ref.mat)
    assert a.channel == ref.channel == SpinChannel.DENS


def test_sub_inplace_equals_out_of_place_and_returns_self(rng):
    """sub(other, copy=False) subtracts into self in place, returns self, bit-equal to self - other."""
    a, b = _kernel_block(rng, SpinChannel.DENS), _kernel_block(rng, SpinChannel.MAGN)
    ref = deepcopy(a) - deepcopy(b)
    out = a.sub(b, copy=False)
    assert out is a
    assert np.array_equal(a.mat, ref.mat)


def test_add_inplace_with_scaled_other_matches_reference(rng):
    """The kernel-accumulation idiom k.add(other.scale(3), copy=False) equals k + 3 * other."""
    a, b = _kernel_block(rng, SpinChannel.NONE), _kernel_block(rng, SpinChannel.MAGN)
    ref = deepcopy(a) + 3 * deepcopy(b)
    a.add(b.scale(3.0), copy=False)
    assert np.array_equal(a.mat, ref.mat)


def test_add_copy_true_is_nondestructive(rng):
    """add(other) (copy=True default) returns a new object and leaves self unchanged (unchanged behavior)."""
    a, b = _kernel_block(rng, SpinChannel.DENS), _kernel_block(rng, SpinChannel.DENS)
    before = a.mat.copy()
    out = a.add(b)
    assert out is not a
    assert np.array_equal(a.mat, before)


def test_add_inplace_rejects_non_fourpoint(rng):
    """copy=False is only defined for FourPoint and (Local)Interaction operands; a scalar operand raises."""
    a = _kernel_block(rng, SpinChannel.DENS)
    with pytest.raises(NotImplementedError):
        a.add(2.0, copy=False)


def test_add_inplace_rejects_vn_extension(rng):
    """copy=False refuses to diagonally extend self (num_vn=1 += num_vn=2), which would allocate a larger array."""
    a, b = _kernel_block(rng, SpinChannel.NONE, num_vn=1), _kernel_block(rng, SpinChannel.DENS, num_vn=2)
    with pytest.raises(ValueError):
        a.add(b, copy=False)


def test_add_inplace_reverts_other_niw_range(rng):
    """copy=False converts a full-niw other to half for the op, then restores its range (non-destructive to other)."""
    nq, qtot, o, niw, niv = (4, 4, 1), 16, 2, 3, 3
    a = _kernel_block(rng, SpinChannel.DENS)  # half niw range
    full_shape = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv)
    full_mat = rng.standard_normal(full_shape) + 1j * rng.standard_normal(full_shape)
    b = FourPoint(full_mat, SpinChannel.MAGN, nq, 1, 1, True, True, True, FrequencyNotation.PH)
    ref = deepcopy(a) + deepcopy(b)
    a.add(b, copy=False)
    assert np.array_equal(a.mat, ref.mat)
    assert b.full_niw_range


def _channel_interactions(rng, o=2, qtot=16, nq=(4, 4, 1)):
    """Builds a random LocalInteraction [o, o, o, o] and Interaction [q, o, o, o, o] pair (compressed q)."""
    u_mat = rng.standard_normal((o,) * 4)
    v_mat = rng.standard_normal((qtot,) + (o,) * 4)
    return LocalInteraction(u_mat, SpinChannel.DENS), Interaction(v_mat, SpinChannel.DENS, nq, True)


def test_sub_inplace_with_local_interaction_matches_copy(rng):
    """sub(u_loc, copy=False) broadcast-subtracts the local interaction into self in place, bit-equal to the
    copying branch, and returns self."""
    a, ref = _kernel_block(rng, SpinChannel.DENS, num_vn=2), None
    u_loc, _ = _channel_interactions(rng)
    ref = deepcopy(a).sub(u_loc)
    out = a.sub(u_loc, copy=False)
    assert out is a
    assert np.array_equal(a.mat, ref.mat)


def test_add_inplace_with_nonlocal_interaction_matches_copy(rng):
    """add(v_nonloc, copy=False) broadcast-adds the q-dependent interaction into self in place, bit-equal to the
    copying branch."""
    a = _kernel_block(rng, SpinChannel.DENS, num_vn=2)
    _, v_nonloc = _channel_interactions(rng)
    ref = deepcopy(a).add(v_nonloc)
    out = a.add(v_nonloc, copy=False)
    assert out is a
    assert np.array_equal(a.mat, ref.mat)


def _local_block(rng, num_vn=2, o=2, niw=3, niv=3):
    """Builds a half-niw LocalFourPoint [o, o, o, o, niw + 1, (2*niv,) * num_vn] with random complex entries."""
    from dgamore.local_four_point import LocalFourPoint

    shape = (o, o, o, o, niw + 1) + (2 * niv,) * num_vn
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return LocalFourPoint(mat, SpinChannel.NONE, 1, num_vn, False, True, FrequencyNotation.PH)


def test_add_inplace_with_local_fourpoint_matches_copy(rng):
    """add(local_other, copy=False) broadcast-accumulates a momentum-independent operand into self in place,
    bit-equal to the copying branch, leaving the operand untouched."""
    a, b = _kernel_block(rng, SpinChannel.DENS, num_vn=2), _local_block(rng, num_vn=2)
    b_before = b.mat.copy()
    ref = deepcopy(a).add(deepcopy(b))
    out = a.add(b, copy=False)
    assert out is a
    assert np.array_equal(a.mat, ref.mat)
    assert np.array_equal(b.mat, b_before)


def test_sub_inplace_with_local_fourpoint_rejects_vn_extension(rng):
    """copy=False with a 2-vn local operand refuses to diagonally extend a 1-vn self."""
    a, b = _kernel_block(rng, SpinChannel.DENS, num_vn=1), _local_block(rng, num_vn=2)
    with pytest.raises(ValueError):
        a.sub(b, copy=False)


def test_add_on_vn_diagonal_matches_extend_and_add(rng):
    """add_on_vn_diagonal(other, factor) equals extending the 1-vn other to the fermionic diagonal and adding the
    scaled result, without allocating the extended block; mutates and returns self."""
    a = _kernel_block(rng, SpinChannel.DENS, num_vn=2)
    b = _kernel_block(rng, SpinChannel.NONE, num_vn=1)
    ref = deepcopy(a).add(deepcopy(b).scale(2.5).extend_vn_to_diagonal())
    out = a.add_on_vn_diagonal(b, factor=2.5)
    assert out is a
    assert np.allclose(a.mat, ref.mat, atol=1e-6)
    assert b.num_vn_dimensions == 1


def test_add_on_vn_diagonal_with_local_other_broadcasts_over_q(rng):
    """A momentum-independent 1-vn LocalFourPoint other is broadcast over the momentum axis of self."""
    from dgamore.local_four_point import LocalFourPoint

    a = _kernel_block(rng, SpinChannel.DENS, num_vn=2)
    o, niw, niv = 2, 3, 3
    b_shape = (o, o, o, o, niw + 1, 2 * niv)
    b_mat = rng.standard_normal(b_shape) + 1j * rng.standard_normal(b_shape)
    b = LocalFourPoint(b_mat, SpinChannel.NONE, 1, 1, False, True, FrequencyNotation.PH)
    ref = deepcopy(a).add(deepcopy(b).extend_vn_to_diagonal())
    out = a.add_on_vn_diagonal(b)
    assert out is a
    assert np.allclose(a.mat, ref.mat, atol=1e-6)


def test_add_on_vn_diagonal_rejects_mismatched_operands(rng):
    """Mismatched vn counts, niw ranges or fermionic box sizes raise instead of silently mis-adding."""
    a2, b1 = _kernel_block(rng, SpinChannel.DENS, num_vn=2), _kernel_block(rng, SpinChannel.NONE, num_vn=1)
    with pytest.raises(ValueError):
        b1.add_on_vn_diagonal(b1)
    with pytest.raises(ValueError):
        a2.add_on_vn_diagonal(a2)
    with pytest.raises(ValueError):
        a2.add_on_vn_diagonal(deepcopy(b1).to_full_niw_range())


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
    assert np.allclose(got, expect, atol=1e-6)


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
    assert not result.full_niw_range
    assert result.mat.dtype == np.complex64
    assert np.allclose(result.mat, ref_mat)


def test_invert_num_vn2_per_q_matches_batched_compound_reference(rng):
    """The num_vn==2 invert matches a batched np.linalg.inv on the explicit compound layout [q, w, (1, 2, v),
    (4, 3, v')] for both frequency notations (pp pairs rows (1, 3, v) x cols (4, 2, v') via the acbd permute), and
    inverting twice returns the original."""
    nq = (3, 2, 1)
    qtot, o, niw, niv = 6, 2, 2, 2
    size = o * o * 2 * niv
    for notation in (FrequencyNotation.PH, FrequencyNotation.PP):
        shape = (qtot, o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
        mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        fp = FourPoint(mat, SpinChannel.DENS, nq, 1, 2, True, True, True, notation)

        ref = fp.copy().to_half_niw_range()
        ref_mat = ref.mat if notation == FrequencyNotation.PH else np.einsum("qabcdwvp->qacbdwvp", ref.mat)
        comp = np.linalg.inv(ref_mat.transpose(0, 5, 1, 2, 6, 4, 3, 7).reshape(qtot, niw + 1, size, size))
        ref_mat = comp.reshape(qtot, niw + 1, o, o, 2 * niv, o, o, 2 * niv).transpose(0, 2, 3, 6, 5, 1, 4, 7)
        if notation == FrequencyNotation.PP:
            ref_mat = np.einsum("qacbdwvp->qabcdwvp", ref_mat)

        result = fp.copy().invert(copy=True)
        assert not result.full_niw_range and result.mat.dtype == np.complex64
        assert np.allclose(result.mat, ref_mat, atol=1e-3)

        roundtrip = result.invert(copy=True)
        assert np.allclose(roundtrip.mat, fp.copy().to_half_niw_range().mat, atol=1e-3)


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
    assert np.allclose(out1.mat, out2.mat, atol=1e-6)


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
    assert np.allclose(out1.mat, out2.mat, atol=1e-6)


def test_invert_and_sum_scales_with_beta(small_fourpoint_compressed):
    """invert_and_sum_over_last_vn_v2 scales as 1/beta."""
    fp = deepcopy(small_fourpoint_compressed)

    beta1 = 1.0
    beta2 = 5.0

    out_beta1 = deepcopy(fp).invert_and_sum_over_last_vn_v2(beta1)
    out_beta2 = deepcopy(fp).invert_and_sum_over_last_vn_v2(beta2)

    scale = beta1 / beta2
    assert np.allclose(out_beta2.mat, out_beta1.mat * scale, atol=1e-8)


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


def _compound_product_reference_q(mat1: np.ndarray, mat2: np.ndarray, notation: FrequencyNotation) -> np.ndarray:
    """Per-momentum compound-space matrix product of two full-index tensors [q,o,o,o,o,v,v'] in the given frequency
    notation (ph: rows {1,2,v}, cols {4,3,v'}; pp: rows {1,3,v}, cols {4,2,v'})."""
    nq_tot, o, n2 = mat1.shape[0], mat1.shape[1], mat1.shape[-1]
    dim = o * o * n2
    order = (0, 1, 4, 3, 2, 5) if notation == FrequencyNotation.PH else (0, 2, 4, 3, 1, 5)
    order_q = (0,) + tuple(i + 1 for i in order)
    compound_shape = (nq_tot,) + tuple(np.array(mat1.shape[1:])[list(order)])
    prod = np.transpose(mat1, order_q).reshape(nq_tot, dim, dim) @ np.transpose(mat2, order_q).reshape(nq_tot, dim, dim)
    return np.transpose(prod.reshape(compound_shape), np.argsort(order_q))


@pytest.mark.parametrize("notation", [FrequencyNotation.PH, FrequencyNotation.PP])
def test_matmul_propagates_frequency_notation_and_compound_pairing(notation):
    """Matmul contracts each momentum slice in the compound space of the operands' notation and the result carries
    the frequency notation of self, so pp results unravel with the acbd back-permute."""
    rng = np.random.default_rng(13)
    nq, o, niv = (2, 2, 1), 2, 3
    nq_tot = int(np.prod(nq))
    shape = (nq_tot, o, o, o, o, 1, 2 * niv, 2 * niv)
    mat1 = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    mat2 = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    x = FourPoint(mat1.copy(), SpinChannel.DENS, nq, 1, 2, True, True, True, notation)
    y = FourPoint(mat2.copy(), SpinChannel.DENS, nq, 1, 2, True, True, True, notation)
    z = x @ y
    ref = _compound_product_reference_q(
        mat1[:, :, :, :, :, 0].astype(np.complex64), mat2[:, :, :, :, :, 0].astype(np.complex64), notation
    )
    assert z.frequency_notation == notation
    assert np.allclose(z.mat[:, :, :, :, :, 0], ref, atol=1e-4)


@pytest.mark.parametrize("notation", [FrequencyNotation.PH, FrequencyNotation.PP])
def test_matmul_mixed_vn_respects_frequency_notation(notation):
    """The memory-saving 2vn @ 1vn matmul branch contracts each momentum slice with the notation's orbital pairing
    (the 1vn operand acts nu-diagonally on the result's second frequency) and keeps the frequency notation of
    self."""
    rng = np.random.default_rng(16)
    nq, o, niv = (2, 2, 1), 2, 3
    nq_tot = int(np.prod(nq))
    shape2 = (nq_tot, o, o, o, o, 1, 2 * niv, 2 * niv)
    shape1 = (nq_tot, o, o, o, o, 1, 2 * niv)
    mat2v = rng.standard_normal(shape2) + 1j * rng.standard_normal(shape2)
    mat1v = rng.standard_normal(shape1) + 1j * rng.standard_normal(shape1)
    x = FourPoint(mat2v.copy(), SpinChannel.DENS, nq, 1, 2, True, True, True, notation)
    y = FourPoint(mat1v.copy(), SpinChannel.DENS, nq, 1, 1, True, True, True, notation)
    z = x @ y
    y_diag = np.zeros(shape2, dtype=np.complex64)
    idx = np.arange(2 * niv)
    y_diag[..., idx, idx] = mat1v
    ref = _compound_product_reference_q(
        mat2v[:, :, :, :, :, 0].astype(np.complex64), y_diag[:, :, :, :, :, 0], notation
    )
    assert z.frequency_notation == notation
    assert z.num_vn_dimensions == 2
    assert np.allclose(z.mat[:, :, :, :, :, 0], ref, atol=1e-4)


@pytest.mark.parametrize("notation", [FrequencyNotation.PH, FrequencyNotation.PP])
def test_matmul_with_local_interaction_respects_frequency_notation(notation):
    """FourPoint @ LocalInteraction contracts the frequency-constant bare interaction with the notation's orbital
    pairing on every momentum slice and keeps the frequency notation of the four-point operand."""
    rng = np.random.default_rng(17)
    nq, o, niv = (2, 2, 1), 2, 3
    nq_tot = int(np.prod(nq))
    shape = (nq_tot, o, o, o, o, 1, 2 * niv, 2 * niv)
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    umat = rng.standard_normal((o, o, o, o)) + 1j * rng.standard_normal((o, o, o, o))
    x = FourPoint(mat.copy(), SpinChannel.DENS, nq, 1, 2, True, True, True, notation)
    u = LocalInteraction(umat.copy())
    u_diag = np.zeros(shape, dtype=np.complex64)
    idx = np.arange(2 * niv)
    u_diag[..., idx, idx] = umat[None, :, :, :, :, None, None].astype(np.complex64)
    z = x @ u
    ref = _compound_product_reference_q(mat[:, :, :, :, :, 0].astype(np.complex64), u_diag[:, :, :, :, :, 0], notation)
    assert z.frequency_notation == notation
    assert np.allclose(z.mat[:, :, :, :, :, 0], ref, atol=1e-4)


def test_pow_pp_squares_in_pp_compound_space_without_explicit_identity():
    """fp ** 2 on a pp object squares each momentum slice in the pp compound space (rows {1,3,v}, cols {4,2,v'})
    and keeps the PP notation, with the matching identity derived internally via identity_like."""
    rng = np.random.default_rng(20)
    nq, o, niv = (2, 2, 1), 2, 3
    nq_tot = int(np.prod(nq))
    shape = (nq_tot, o, o, o, o, 1, 2 * niv, 2 * niv)
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    fp = FourPoint(mat.copy(), SpinChannel.DENS, nq, 1, 2, True, True, True, FrequencyNotation.PP)
    result = fp**2
    mat64 = mat[:, :, :, :, :, 0].astype(np.complex64)
    ref = _compound_product_reference_q(mat64, mat64, FrequencyNotation.PP)
    assert result.frequency_notation == FrequencyNotation.PP
    assert np.allclose(result.mat[:, :, :, :, :, 0], ref, atol=1e-4)

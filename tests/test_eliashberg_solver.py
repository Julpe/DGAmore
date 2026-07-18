# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

from unittest.mock import MagicMock

import numpy as np
import pytest

import dgamore.config as config
from dgamore import brillouin_zone as bz
from dgamore.bubble_gen import BubbleGenerator
from dgamore.eliashberg_solver import (
    _apply_gamma_pp,
    _apply_gchi0_pp,
    _chi0_to_matmul_layout,
    _frequency_parity_sectors,
    gap_parity_diagnostics,
    _project_gap_to_sector,
    classify_gap_symmetry,
    create_local_gamma_ud_pp_w0,
    create_local_gamma_ud_pp_w0_per_ineq,
    _gamma_to_matmul_layout,
    solve_eliashberg_lanczos,
    symmetrize_degenerate_gaps,
    transform_vertex_loc_frequencies_w0,
)
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction
from dgamore.local_four_point import LocalFourPoint
from dgamore.n_point_base import FrequencyNotation, SpinChannel


def test_apply_gchi0_pp_matches_einsum():
    """_apply_gchi0_pp reproduces einsum('xyzabcdv,xyzcdv->xyzabv') (the chi0*gap multiplication in the matvec)."""
    rng = np.random.default_rng(0)
    nqx, nqy, nqz, o, v = 3, 4, 2, 3, 6
    chi0 = rng.standard_normal((nqx, nqy, nqz, o, o, o, o, v)) + 1j * rng.standard_normal(
        (nqx, nqy, nqz, o, o, o, o, v)
    )
    gap = rng.standard_normal((nqx, nqy, nqz, o, o, v)) + 1j * rng.standard_normal((nqx, nqy, nqz, o, o, v))
    ref = np.einsum("xyzabcdv,xyzcdv->xyzabv", chi0, gap, optimize=True)
    got = _apply_gchi0_pp(_chi0_to_matmul_layout(chi0), gap.ravel(), o)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-10)


@pytest.mark.parametrize("nv", [6, 3])
def test_apply_gamma_pp_matches_einsum(nv):
    """_apply_gamma_pp reproduces einsum('xyzacbdvp,xyzcdp->xyzabv') for full (nv==np) and frequency-sliced (nv<np) v."""
    rng = np.random.default_rng(1)
    nqx, nqy, nqz, o, npp = 2, 3, 2, 3, 6
    shape = (nqx, nqy, nqz, o, o, o, o, nv, npp)
    gamma = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    gap_gg = rng.standard_normal((nqx, nqy, nqz, o, o, npp)) + 1j * rng.standard_normal((nqx, nqy, nqz, o, o, npp))
    ref = np.einsum("xyzacbdvp,xyzcdp->xyzabv", gamma, gap_gg, optimize=True)
    got = _apply_gamma_pp(_gamma_to_matmul_layout(gamma), gap_gg, o)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-10)


def _make_pp_chi_and_bubble(
    o: int, niv_pp: int, beta: float, seed: int
) -> tuple[LocalFourPoint, LocalFourPoint, GreensFunction]:
    """Builds a random crossing-symmetric pp susceptibility (J chi J = chi) and the matching diagonal bare pp bubble
    from a random symmetric local Green's function (G_12(v) = G_21(v), no SOC)."""
    rng = np.random.default_rng(seed)
    g_mat = rng.standard_normal((o, o, 2 * (niv_pp + 2))) + 1j * rng.standard_normal((o, o, 2 * (niv_pp + 2)))
    g_mat = 0.5 * (g_mat + g_mat.transpose(1, 0, 2)) + 2.0 * np.eye(o)[:, :, None]
    g = GreensFunction(g_mat)
    chi0 = BubbleGenerator.create_generalized_chi0_pp_w0(g, niv_pp, beta).extend_vn_to_diagonal()
    shape = (o, o, o, o, 1, 2 * niv_pp, 2 * niv_pp)
    chi_mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    chi_mat = 0.5 * (chi_mat + _j_conjugate(chi_mat))
    # compound-identity shift: J-symmetric, keeps every inversion well-conditioned for any band count
    ident = np.einsum("ad,bc,vp->abcdvp", np.eye(o), np.eye(o), np.eye(2 * niv_pp))
    chi_mat += 4.0 * o * ident[:, :, :, :, None]
    chi = LocalFourPoint(chi_mat, SpinChannel.UD, 1, 2, True, True, FrequencyNotation.PP)
    return chi, chi0, g


def _j_conjugate(mat: np.ndarray) -> np.ndarray:
    """Conjugates a full-index pp tensor with the crossing operator J (swap both orbital pairs, flip both fermionic
    frequencies)."""
    return np.einsum("abcdwvp->cdabwvp", mat)[..., ::-1, ::-1]


def test_crossed_term_reuses_direct_vertex_via_index_shuffles():
    """The flipped-vertex contraction Gamma_flip[K] @ gap_flip[K] (Gamma_flip = momentum-flip + v'-flip + adcb of
    the direct vertex, sign-folded) must equal the K-flipped, row-orbital-swapped image of the DIRECT vertex applied
    to the p-flipped gap: sign * flip_K[swap_ab[Gamma @ flip_p(gap_gg)]] - the identity that lets the matvec drop
    the second stored vertex."""
    from dgamore.n_point_base import FrequencyNotation

    rng = np.random.default_rng(37)
    nqx, nqy, nqz, o, n2 = 3, 4, 2, 2, 4
    shape = (nqx, nqy, nqz, o, o, o, o, n2, n2)
    gam = FourPoint(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        SpinChannel.SING,
        (nqx, nqy, nqz),
        0,
        2,
        True,
        True,
        False,
        FrequencyNotation.PP,
    )
    gap_gg = rng.standard_normal((nqx, nqy, nqz, o, o, n2)) + 1j * rng.standard_normal((nqx, nqy, nqz, o, o, n2))
    gap_gg = gap_gg.astype(np.complex64)
    for sign in (1.0, -1.0):
        flipped = gam.flip_momentum_axis().flip_frequency_axis(-1, False).permute_orbitals("abcd->adcb", False)
        flipped_mm = _gamma_to_matmul_layout(flipped.permute_orbitals("abcd->badc", False).mat) * sign
        gap_gg_flipped = np.roll(np.flip(gap_gg, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
        ref = _apply_gamma_pp(flipped_mm, gap_gg_flipped, o)

        direct_mm = _gamma_to_matmul_layout(gam.copy().permute_orbitals("abcd->badc", False).mat)
        crossed = _apply_gamma_pp(direct_mm, np.flip(gap_gg, axis=-1), o)
        crossed = sign * np.roll(np.flip(crossed.swapaxes(3, 4), axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
        assert np.allclose(crossed, ref, atol=1e-4)


def test_transform_vertex_q_frequencies_w0_matches_untrimmed_reference():
    """The trimmed-bosonic-window pp transform must equal the untrimmed reference gather (full niw restoration plus
    the omega = v - v' selection over wn(niw)) for niw far larger than the read window 2*niv_pp - 1."""
    from dgamore.eliashberg_solver import transform_vertex_q_frequencies_w0
    from dgamore.matsubara_frequencies import MFHelper

    rng = np.random.default_rng(33)
    o, nqi, niw, niv, niv_pp = 2, 2, 9, 4, 2
    config.lattice.k_grid = bz.KGrid((nqi, 1, 1), symmetries=[])
    shape = (nqi, o, o, o, o, niw + 1, 2 * niv, 2 * niv)
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    vertex = FourPoint(mat, SpinChannel.DENS, (nqi, 1, 1), 1, 2, False, True, True)

    ref_v = (
        vertex.copy()
        .cut_niv(niv_pp)
        .to_full_niw_range()
        .permute_orbitals("abcd->adcb", copy=False)
        .flip_frequency_axis(-1, False)
    )
    vn = MFHelper.vn(niv_pp)
    omega = vn[:, None] - vn[None, :]
    ref = np.zeros((*ref_v.current_shape[:-3], 2 * niv_pp, 2 * niv_pp), dtype=ref_v.mat.dtype)
    for idx, w in enumerate(MFHelper.wn(niw)):
        ref[..., omega == w] = -ref_v.mat[..., idx, omega == w]

    out = transform_vertex_q_frequencies_w0(vertex, niv_pp)
    assert np.array_equal(out.mat, ref)


def test_full_vertex_first_term_restructure_matches_original_expression():
    """The eager-rebound first term of the full ladder vertex, F_1 = -beta^2 (chi0^-1 chi* chi0^-1) plus beta^2
    chi0^-1 added on the fermionic diagonal, must equal the original single-expression form
    beta^2 (chi0^-1 delta - chi0^-1 chi* chi0^-1) within complex64 rounding, without mutating the bubble."""
    rng = np.random.default_rng(31)
    o, nqi, nw, niv, beta = 2, 3, 2, 2, 12.5
    config.sys.beta = beta
    aux_shape = (nqi, o, o, o, o, nw, 2 * niv, 2 * niv)
    aux_mat = rng.standard_normal(aux_shape) + 1j * rng.standard_normal(aux_shape)
    chi_aux = FourPoint(aux_mat, SpinChannel.DENS, (nqi, 1, 1), 1, 2, False, True, True)
    chi0_shape = (nqi, o, o, o, o, nw, 2 * niv)
    chi0_mat = rng.standard_normal(chi0_shape) + 1j * rng.standard_normal(chi0_shape)
    gchi0_inv = FourPoint(chi0_mat, SpinChannel.NONE, (nqi, 1, 1), 1, 1, False, True, True)
    chi0_before = gchi0_inv.mat.copy()
    ref = (gchi0_inv - gchi0_inv @ chi_aux.copy() @ gchi0_inv).scale(beta**2)
    new = gchi0_inv @ chi_aux
    new = new @ gchi0_inv
    new = new.scale(-(beta**2)).add_on_vn_diagonal(gchi0_inv, factor=beta**2)
    assert np.allclose(new.mat, ref.mat, atol=0.05)
    assert np.array_equal(gchi0_inv.mat, chi0_before)


def test_local_gamma_ud_pp_w0_matches_b26_for_single_band():
    """For one band and crossing-symmetric chi the J-decorated BSE inversion is identical to the old flipped-bubble
    B.26 form, locking backwards compatibility of the single-orbital results."""
    beta, niv_pp = 12.5, 4
    chi, chi0, _ = _make_pp_chi_and_bubble(1, niv_pp, beta, seed=7)
    chi0_flipped = chi0.flip_frequency_axis(-1)
    gamma_old = ((chi - chi0_flipped).invert() + chi0_flipped.invert()).scale(beta**2)
    gamma_new = create_local_gamma_ud_pp_w0(chi, chi0, beta)
    assert gamma_new.mat.shape == gamma_old.mat.shape
    assert np.allclose(gamma_new.mat, gamma_old.mat, atol=1e-3)


@pytest.mark.parametrize("o", [2, 3, 4, 5])
def test_local_gamma_ud_pp_w0_multiorbital_preserves_crossing_symmetry(o):
    """For more than one band the J-decorated inversion keeps Gamma crossing-symmetric (J Gamma J = Gamma) and
    deviates from the old flipped-bubble form, whose bubble misses the orbital pair permutation of the crossing
    operator."""
    beta, niv_pp = 10.0, 3
    chi, chi0, _ = _make_pp_chi_and_bubble(o, niv_pp, beta, seed=8)
    gamma_new = create_local_gamma_ud_pp_w0(chi, chi0, beta)
    assert gamma_new.mat.shape == (o, o, o, o, 1, 2 * niv_pp, 2 * niv_pp)
    assert np.allclose(gamma_new.mat, _j_conjugate(gamma_new.mat), atol=1e-3)
    chi0_flipped = chi0.flip_frequency_axis(-1)
    gamma_old = ((chi - chi0_flipped).invert() + chi0_flipped.invert()).scale(beta**2)
    assert not np.allclose(gamma_old.mat, gamma_new.mat, atol=1e-3)


@pytest.mark.parametrize("o", [2, 3, 4, 5])
def test_local_gamma_ud_pp_w0_satisfies_pp_bse(o):
    """Locks every orbital and frequency index of create_local_gamma_ud_pp_w0: the helper matches a float64 compound
    reference of Gamma^{vv'}_1234 = beta^2 [chi0 J - chi0 chi^{-1} chi0]^{-1;vv'}_1234, the reference F (leg-amputated
    chi) rebuilds chi through the raw leg einsum, and (Gamma, F) satisfy the crossing-decoupled pp BSE written with
    explicit indices (free orbital indices 1234, summed indices alphabetical from the left-most object)."""
    beta, niv_pp = 10.0, 3
    chi_obj, chi0_obj, g = _make_pp_chi_and_bubble(o, niv_pp, beta, seed=11)
    n2 = 2 * niv_pp
    dim = o * o * n2

    def to_c(x):
        return np.transpose(x, (0, 2, 4, 3, 1, 5)).reshape(dim, dim)

    def to_t(m):
        return np.transpose(m.reshape(o, o, n2, o, o, n2), (0, 4, 1, 3, 2, 5))

    chi = chi_obj.mat[:, :, :, :, 0].astype(np.complex128)
    chi0 = chi0_obj.mat[:, :, :, :, 0].astype(np.complex128)
    gv = g.mat[:, :, 2:-2].astype(np.complex128)
    gm = gv[:, :, ::-1]
    chi0_j = np.einsum("abcdvp->adcbvp", chi0)[..., ::-1]
    assert np.allclose(chi0_j, -beta * np.einsum("abv,cdv,vp->abcdvp", gv, gm, np.eye(n2)[:, ::-1]), atol=1e-4)
    gamma_ref = to_t(beta**2 * np.linalg.inv(to_c(chi0_j) - to_c(chi0) @ np.linalg.inv(to_c(chi)) @ to_c(chi0)))
    gamma = create_local_gamma_ud_pp_w0(chi_obj, chi0_obj, beta)
    assert np.allclose(gamma.mat[:, :, :, :, 0], gamma_ref, atol=1e-3)
    f = to_t(-(beta**2) * np.linalg.inv(to_c(chi0)) @ to_c(chi) @ np.linalg.inv(to_c(chi0)))
    chi_back = -np.einsum("xav,ycv,abcdvp,dup,bzp->xzyuvp", gv, gm, f, gv, gm, optimize=True)
    assert np.allclose(chi_back, chi, atol=1e-4)
    # BSE: F^{vv'}_1234 = Gamma^{vv'}_1234 - (1/beta) sum_{v1,abcd} Gamma^{v v1}_{1a3b} G_bc(v1) G_ad(-v1) F^{(-v1)v'}_{d2c4}
    ladder = np.einsum("xfyevw,ehw,fgw,gzhuwp->xzyuvp", gamma_ref, gv, gm, f[..., ::-1, :], optimize=True)
    assert np.allclose(f, gamma_ref - ladder / beta, atol=1e-3)


@pytest.mark.parametrize("o", [5, 4, 3, 2, 1])
def test_local_gamma_ud_pp_w0_satisfies_decoupled_singlet_triplet_bse(o):
    """Gamma from create_local_gamma_ud_pp_w0 fulfills the decoupled singlet/triplet pp BSEs of thesis Eqs.
    (3.51)/(3.52) on the J-even/odd blocks, F_s/t = Gamma_s/t +/- 1/(2 beta^2) Gamma_s/t chi0 F_s/t, in compound
    pp space (rows {1,3,v}, cols {4,2,v'}); J is the crossing operator, chi0*J its exact matrix realization."""
    beta, niv_pp = 10.0, 3
    chi_obj, chi0_obj, _ = _make_pp_chi_and_bubble(o, niv_pp, beta, seed=23)
    n2 = 2 * niv_pp
    dim = o * o * n2

    def to_c(x):
        return np.transpose(x, (0, 2, 4, 3, 1, 5)).reshape(dim, dim)

    chi = to_c(chi_obj.mat[:, :, :, :, 0].astype(np.complex128))
    chi0 = to_c(chi0_obj.mat[:, :, :, :, 0].astype(np.complex128))
    chi0_j = to_c(np.einsum("abcdvp->adcbvp", chi0_obj.mat[:, :, :, :, 0].astype(np.complex128))[..., ::-1])
    j = to_c(np.einsum("ab,cd,vp->abcdvp", np.eye(o), np.eye(o), np.eye(n2)[:, ::-1]))
    # the crossing operator squares to one, realizes chi0*J and commutes with the (crossing-symmetric) inputs
    assert np.allclose(j @ j, np.eye(dim), atol=1e-12)
    assert np.allclose(chi0 @ j, chi0_j, atol=1e-12)
    assert np.allclose(j @ chi0, chi0 @ j, atol=1e-12)
    assert np.allclose(j @ chi, chi @ j, atol=1e-4)

    gamma_ref = beta**2 * np.linalg.inv(chi0_j - chi0 @ np.linalg.inv(chi) @ chi0)
    gamma_code = to_c(create_local_gamma_ud_pp_w0(chi_obj, chi0_obj, beta).mat[:, :, :, :, 0].astype(np.complex128))
    assert np.allclose(gamma_code, gamma_ref, atol=1e-3)

    f = -(beta**2) * np.linalg.inv(chi0) @ chi @ np.linalg.inv(chi0)
    p_plus = 0.5 * (np.eye(dim) + j)
    p_minus = 0.5 * (np.eye(dim) - j)
    f_s, gamma_s = 2 * f @ p_plus, 2 * gamma_ref @ p_plus
    f_t, gamma_t = 2 * f @ p_minus, 2 * gamma_ref @ p_minus
    assert np.allclose(f_s, gamma_s + gamma_s @ chi0 @ f_s / (2 * beta**2), atol=1e-3)
    assert np.allclose(f_t, gamma_t - gamma_t @ chi0 @ f_t / (2 * beta**2), atol=1e-3)


@pytest.mark.parametrize("o", [5, 4, 3, 2, 1])
def test_transform_vertex_loc_frequencies_w0_is_crossed_slot_form(o):
    """transform_vertex_loc_frequencies_w0 returns -F^{(v-v')v(-v')}_{1432} (crossed slot of thesis Eq. 4.49); the
    orbital permutation to 1432 only shows up for more than one band."""
    rng = np.random.default_rng(6)
    niw, niv, niv_pp = 6, 5, 3
    config.box.niw_core = niw
    shape = (o, o, o, o, 2 * niw + 1, 2 * niv, 2 * niv)
    mat = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    f_loc = LocalFourPoint(mat.copy(), SpinChannel.UD, 1, 2, True, True)
    got = transform_vertex_loc_frequencies_w0(f_loc, niv_pp)
    vn = np.arange(-niv_pp, niv_pp)
    expected = np.zeros((o, o, o, o, 2 * niv_pp, 2 * niv_pp), dtype=mat.dtype)
    for i, n in enumerate(vn):
        for j, m in enumerate(vn):
            # crossed slot: orbitals 1432 ('abcd->adcb'), bosonic index v-v', second fermionic frequency flipped
            expected[..., i, j] = -mat[:, :, :, :, niw + n - m, niv + n, niv - m - 1].transpose(0, 3, 2, 1)
    assert got.mat.shape == expected.shape
    assert got.frequency_notation == FrequencyNotation.PP
    assert np.allclose(got.mat, expected, atol=1e-6)


@pytest.mark.parametrize("o", [2, 1])
def test_pairing_vertex_contraction_uses_triqs_leg_order(o):
    """The 'abcd->badc' permute makes the vertex contraction implement Delta_ab = sum_cd Gamma_cadb X_cd (TRIQS leg
    order); the raw w2dynamics-order layout only agrees for a single band."""
    rng = np.random.default_rng(3)
    nq, nv = (2, 2, 1), 4
    shape = (*nq, o, o, o, o, nv, nv)
    gamma = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    x = (rng.standard_normal((*nq, o, o, nv)) + 1j * rng.standard_normal((*nq, o, o, nv))).astype(np.complex64)
    ref = np.einsum("xyzcadbvp,xyzcdp->xyzabv", gamma, x, optimize=True)
    badc = FourPoint(
        gamma.copy(), SpinChannel.SING, nq, 0, 2, True, True, False, FrequencyNotation.PP
    ).permute_orbitals("abcd->badc")
    fixed = _apply_gamma_pp(_gamma_to_matmul_layout(badc.mat), x, o)
    raw = _apply_gamma_pp(_gamma_to_matmul_layout(gamma), x, o)
    assert np.allclose(fixed, ref, atol=1e-4)
    assert (o == 1) == np.allclose(raw, ref, atol=1e-4)


def _graded_orbital_symmetrization(gamma: np.ndarray, generators: list[tuple[tuple[int, ...], int]]) -> np.ndarray:
    """Averages a rank-4 orbital tensor over the group generated by the given (permutation, character) pairs. With all
    characters +1 this is a plain symmetrization; a character -1 makes the vertex odd under that permutation."""
    group = {(0, 1, 2, 3): 1.0}
    changed = True
    while changed:
        changed = False
        for p, s in list(group.items()):
            for q, sq in generators:
                composition = tuple(p[q[i]] for i in range(4))
                if composition not in group:
                    group[composition] = s * sq
                    changed = True
    return sum(s * np.transpose(gamma, p) for p, s in group.items()) / len(group)


def _matvec_direct_term_single_slice(
    gamma: np.ndarray, g_plus: np.ndarray, g_minus: np.ndarray, gap: np.ndarray, n_bands: int
) -> np.ndarray:
    """Applies the direct term of the Eliashberg matvec for a single momentum/frequency slice, exactly as in
    :func:`solve_eliashberg_lanczos`: bare pp bubble times gap (see :func:`_apply_gchi0_pp`), then the
    ``"abcd->badc"``-permuted pairing vertex contraction (see :func:`_apply_gamma_pp`)."""
    chi0 = np.einsum("ad,bc->abcd", g_plus, g_minus)[None, None, None, ..., None]
    gap_gg = _apply_gchi0_pp(_chi0_to_matmul_layout(chi0), gap[None, None, None, ..., None].ravel(), n_bands)
    vertex = FourPoint(
        gamma[None, None, None, ..., None, None].copy(),
        SpinChannel.SING,
        (1, 1, 1),
        0,
        2,
        True,
        True,
        False,
        FrequencyNotation.PP,
    ).permute_orbitals("abcd->badc")
    return _apply_gamma_pp(_gamma_to_matmul_layout(vertex.mat), gap_gg, n_bands)[0, 0, 0, :, :, 0]


@pytest.mark.parametrize("channel", [SpinChannel.SING, SpinChannel.TRIP])
def test_matvec_direct_term_matches_thesis_eq_4_40_on_physical_sector(channel):
    """On the physical symmetry class -- pairing vertex with its particle-swap (1432, 3214) and static time-reversal
    (dcba) symmetries, G(k) symmetric (time reversal plus inversion) and the gap in the SPOT sector (Delta = +/-
    Delta^T for singlet/triplet) -- the matvec direct term is identical to Eq. (4.40) of the thesis and preserves the
    sector. For a generic (unsymmetrized) vertex the two wirings differ, so the equivalence is a property of the
    symmetry class, not of the contraction pattern."""
    rng = np.random.default_rng(4)
    o = 3
    sign = 1 if channel == SpinChannel.SING else -1

    gamma = (rng.standard_normal((o,) * 4) + 1j * rng.standard_normal((o,) * 4)).astype(np.complex64)
    gamma_sym = _graded_orbital_symmetrization(gamma, [((0, 3, 2, 1), sign), ((2, 1, 0, 3), sign), ((3, 2, 1, 0), 1)])

    g_k = rng.standard_normal((o, o)) + 1j * rng.standard_normal((o, o))
    g_k = (0.5 * (g_k + g_k.T)).astype(np.complex64)

    for _ in range(4):
        gap = rng.standard_normal((o, o)) + 1j * rng.standard_normal((o, o))
        gap = (0.5 * (gap + sign * gap.T)).astype(np.complex64)

        got = _matvec_direct_term_single_slice(gamma_sym, g_k, g_k, gap, o)
        ref = np.einsum("xbya,ad,cb,dc->xy", gamma_sym, g_k, g_k, gap, optimize=True)
        assert np.allclose(got, ref, atol=1e-4)
        assert np.allclose(got, sign * got.T, atol=1e-4)

        got_generic = _matvec_direct_term_single_slice(gamma, g_k, g_k, gap, o)
        ref_generic = np.einsum("xbya,ad,cb,dc->xy", gamma, g_k, g_k, gap, optimize=True)
        assert not np.allclose(got_generic, ref_generic, atol=1e-4)


@pytest.mark.parametrize("o", [2, 1])
def test_badc_permute_is_noop_for_swap_and_tr_symmetric_pairing_vertex(o):
    """'abcd->badc' composes from the physical vertex symmetries (badc = dcba composed with cdab), so for a pairing
    vertex carrying the particle-swap (1432, 3214) and static time-reversal (dcba) symmetries the w2dynamics and TRIQS
    leg orders give identical contractions for any gap; for a generic tensor they differ (except for a single band)."""
    rng = np.random.default_rng(5)
    nq, nv = (1, 1, 1), 1

    gamma = (rng.standard_normal((o,) * 4) + 1j * rng.standard_normal((o,) * 4)).astype(np.complex64)
    gamma_sym = _graded_orbital_symmetrization(gamma, [((0, 3, 2, 1), 1), ((2, 1, 0, 3), 1), ((3, 2, 1, 0), 1)])
    x = (rng.standard_normal((*nq, o, o, nv)) + 1j * rng.standard_normal((*nq, o, o, nv))).astype(np.complex64)

    def contract(mat):
        padded = mat[None, None, None, ..., None, None].copy()
        return _apply_gamma_pp(_gamma_to_matmul_layout(padded), x, o)

    def badc(mat):
        return (
            FourPoint(
                mat[None, None, None, ..., None, None].copy(),
                SpinChannel.SING,
                nq,
                0,
                2,
                True,
                True,
                False,
                FrequencyNotation.PP,
            )
            .permute_orbitals("abcd->badc")
            .mat[0, 0, 0, :, :, :, :, 0, 0]
        )

    assert np.allclose(contract(badc(gamma_sym)), contract(gamma_sym), atol=1e-4)
    assert (o == 1) == np.allclose(contract(badc(gamma)), contract(gamma), atol=1e-4)


@pytest.mark.parametrize("o", [2, 3, 4, 5])
@pytest.mark.parametrize("channel", [SpinChannel.SING, SpinChannel.TRIP])
def test_degenerate_decoupled_bands_reproduce_single_band_kernel(monkeypatch, channel, o):
    """o decoupled, degenerate bands (same-band-only pairing vertex, orbital-diagonal Green's function) must give
    a pairing kernel that is o identical copies of the single-band kernel with vanishing inter-band blocks, so the
    eigenvalue spectrum is the single-band one, o-fold degenerate; run through the production solve_eliashberg_lanczos
    matvec (eigsh is intercepted and densified)."""
    nq, niv_pp, beta = (2, 2, 1), 2, 10.0
    nq_tot, n2 = int(np.prod(nq)), 2 * niv_pp
    config.lattice.nk = nq
    config.lattice.k_grid = bz.KGrid(nq, symmetries=[])
    config.sys.beta = beta
    config.eliashberg.n_eig = 2
    config.eliashberg.epsilon = 1e-10
    config.eliashberg.symmetry = "random"
    config.eliashberg.resolve_frequency_parity = False
    config.logger = MagicMock()

    rng = np.random.default_rng(21)
    shape_g1 = (nq_tot, 1, 1, 1, 1, n2, n2)
    gamma1 = rng.standard_normal(shape_g1) + 1j * rng.standard_normal(shape_g1)
    chi1 = rng.standard_normal(shape_g1[:-1]) + 1j * rng.standard_normal(shape_g1[:-1])

    gamma2 = np.zeros((nq_tot, o, o, o, o, n2, n2), dtype=complex)
    chi2 = np.zeros((nq_tot, o, o, o, o, n2), dtype=complex)
    for band in range(o):
        gamma2[:, band, band, band, band] = gamma1[:, 0, 0, 0, 0]
    for a in range(o):
        for c in range(o):
            chi2[:, a, c, c, a] = chi1[:, 0, 0, 0, 0]

    captured = []

    def fake_eigsh(op, k, tol, v0, which, maxiter):
        n = op.shape[0]
        dense = np.column_stack([op.matvec(np.eye(n, dtype=np.complex64)[:, i]) for i in range(n)])
        captured.append(dense)
        lam, vec = np.linalg.eig(dense)
        order = np.argsort(lam.real)[::-1][:k]
        return lam.real[order], vec[:, order]

    with monkeypatch.context() as mp:
        mp.setattr("dgamore.eliashberg_solver.sp.sparse.linalg.eigsh", fake_eigsh)
        solve_eliashberg_lanczos(
            FourPoint(gamma1.copy(), channel, nq, 0, 2, True, True, True, FrequencyNotation.PP),
            FourPoint(chi1.copy(), SpinChannel.NONE, nq, 0, 1, True, True, True, FrequencyNotation.PP),
            (0, 0),
        )
        solve_eliashberg_lanczos(
            FourPoint(gamma2, channel, nq, 0, 2, True, True, True, FrequencyNotation.PP),
            FourPoint(chi2, SpinChannel.NONE, nq, 0, 1, True, True, True, FrequencyNotation.PP),
            (0, 0),
        )
    m1, m2 = captured

    # gap flattening [k, a, b, v]: build the (a, b) sector masks of the multi-band gap space
    idx = np.arange(nq_tot * o * o * n2)
    a_idx = (idx // (o * n2)) % o
    b_idx = idx // n2 % o
    expected = np.zeros_like(m2)
    for band in range(o):
        sector = np.flatnonzero((a_idx == band) & (b_idx == band))
        expected[np.ix_(sector, sector)] = m1
    assert np.allclose(m2, expected, atol=1e-4)

    ev1 = np.linalg.eigvals(m1)
    ev2 = np.linalg.eigvals(m2)
    expected_ev = np.concatenate([np.tile(ev1, o), np.zeros(len(ev2) - o * len(ev1))])
    assert np.allclose(np.sort_complex(ev2), np.sort_complex(expected_ev), atol=1e-3)


def test_solver_thread_budget_derives_from_affinity(monkeypatch):
    """The solver thread budget equals the size of the process affinity mask (at least 1) and falls back to 1
    where the affinity API does not exist."""
    import os

    import dgamore.eliashberg_solver as es

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    assert es._solver_thread_budget() == 4
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert es._solver_thread_budget() == 1


def test_apply_gamma_pp_momentum_parallel_path_is_bit_equal():
    """The momentum-batch-parallel contraction (contiguous k-chunks, one output slice per worker) must be
    bit-equal to the serial path, including worker counts that do not divide the batch."""
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(6)
    nqx, nqy, nqz, o, n2 = 3, 5, 1, 2, 4
    shape = (nqx, nqy, nqz, o, o, o, o, n2, n2)
    gamma_mm = _gamma_to_matmul_layout(
        (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    )
    gap_gg = (
        rng.standard_normal((nqx, nqy, nqz, o, o, n2)) + 1j * rng.standard_normal((nqx, nqy, nqz, o, o, n2))
    ).astype(np.complex64)
    serial = _apply_gamma_pp(gamma_mm, gap_gg, o)
    with ThreadPoolExecutor(max_workers=4) as executor:
        threaded = _apply_gamma_pp(gamma_mm, gap_gg, o, executor, 4)
    assert np.array_equal(threaded, serial)


def test_apply_gchi0_pp_momentum_parallel_path_is_bit_equal():
    """The momentum-batch-parallel bubble multiplication (contiguous k-chunks, one output slice per worker) must
    be bit-equal to the serial path, including worker counts that do not divide the batch."""
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(7)
    nqx, nqy, nqz, o, v = 3, 5, 1, 2, 6
    chi0 = (
        rng.standard_normal((nqx, nqy, nqz, o, o, o, o, v)) + 1j * rng.standard_normal((nqx, nqy, nqz, o, o, o, o, v))
    ).astype(np.complex64)
    gap = (rng.standard_normal((nqx, nqy, nqz, o, o, v)) + 1j * rng.standard_normal((nqx, nqy, nqz, o, o, v))).astype(
        np.complex64
    )
    serial = _apply_gchi0_pp(_chi0_to_matmul_layout(chi0), gap.ravel(), o)
    with ThreadPoolExecutor(max_workers=4) as executor:
        threaded = _apply_gchi0_pp(_chi0_to_matmul_layout(chi0), gap.ravel(), o, executor, 4)
    assert np.array_equal(threaded, serial)


def _make_budget_comm(size, rank, infos):
    """Builds a communicator stub for the v2 thread-budget tests: fixed size/rank and a preset allgather result."""
    return MagicMock(size=size, rank=rank, **{"allgather.return_value": infos})


@pytest.mark.parametrize(
    "rank, active_ranks, masks, hosts, expected",
    [
        (0, [0, 1, 2, 3], [set(range(8))] * 4, ["n0"] * 4, 2),
        (0, [0, 1], [set(range(8))] * 4, ["n0"] * 4, 4),
        (2, [0, 1], [set(range(8))] * 4, ["n0"] * 4, 1),
        (0, [0, 1, 2, 3], [set(range(8))] * 2 + [set(range(8))] * 2, ["n0", "n0", "n1", "n1"], 4),
        (0, [0, 1, 2, 3], [{0, 1, 2, 3}, {0, 1, 2, 3}, {4, 5, 6, 7}, {4, 5, 6, 7}], ["n0"] * 4, 2),
    ],
)
def test_v2_thread_budget_divides_mask_among_active_node_ranks(monkeypatch, rank, active_ranks, masks, hosts, expected):
    """The frequency-distributed solver's budget is this rank's affinity-mask size divided by the number of
    ACTIVE ranks on its node whose masks overlap with it (1 for inactive ranks): full occupancy halves/quarters
    the mask, idle ranks free their share, other nodes and disjoint (per-socket) masks do not count."""
    import dgamore.eliashberg_solver as es

    monkeypatch.setattr(es.os, "sched_getaffinity", lambda pid: masks[rank], raising=False)
    infos = [(hosts[r], frozenset(masks[r])) for r in range(len(masks))]
    comm = _make_budget_comm(len(masks), rank, infos)
    assert es._v2_solver_thread_budget(comm, active_ranks) == expected


def test_v2_thread_budget_falls_back_without_affinity_api_and_single_rank(monkeypatch):
    """Without the affinity API the budget is 1; a single-rank communicator gets the whole mask (nothing else
    runs on the node) without any collective call."""
    import dgamore.eliashberg_solver as es

    monkeypatch.delattr(es.os, "sched_getaffinity", raising=False)
    assert es._v2_solver_thread_budget(_make_budget_comm(4, 0, None), [0, 1, 2, 3]) == 1
    monkeypatch.setattr(es.os, "sched_getaffinity", lambda pid: set(range(6)), raising=False)
    assert es._v2_solver_thread_budget(_make_budget_comm(1, 0, None), [0]) == 6


def test_openblas_thread_slot_cap_reads_back_compiled_maximum(monkeypatch):
    """The slot-cap probe reads an oversized clamped limit back as the smallest compiled OpenBLAS capacity."""
    import dgamore.eliashberg_solver as es

    selection = MagicMock(lib_controllers=[MagicMock(num_threads=64), MagicMock(num_threads=128)])
    monkeypatch.setattr(
        es, "ThreadpoolController", MagicMock(return_value=MagicMock(**{"select.return_value": selection}))
    )
    assert es._openblas_thread_slot_cap.__wrapped__() == 64
    assert selection.limit.call_args.kwargs["limits"] > 8192


def test_openblas_thread_slot_cap_is_none_without_openblas(monkeypatch):
    """The slot-cap probe returns None (no clamp) when no loaded library exposes the OpenBLAS API."""
    import dgamore.eliashberg_solver as es

    selection = MagicMock(lib_controllers=[])
    monkeypatch.setattr(
        es, "ThreadpoolController", MagicMock(return_value=MagicMock(**{"select.return_value": selection}))
    )
    assert es._openblas_thread_slot_cap.__wrapped__() is None
    assert selection.limit.call_count == 0
    monkeypatch.setattr(es, "_openblas_thread_slot_cap", MagicMock(return_value=None))
    assert es._clamp_to_openblas_slot_cap(96) == 96


def test_solver_thread_budgets_clamp_to_openblas_slot_cap(monkeypatch):
    """Both solver thread budgets are clamped to the compiled OpenBLAS thread capacity on wide affinity masks."""
    import dgamore.eliashberg_solver as es

    monkeypatch.setattr(es.os, "sched_getaffinity", lambda pid: set(range(256)), raising=False)
    monkeypatch.setattr(es, "_openblas_thread_slot_cap", MagicMock(return_value=64))
    assert es._solver_thread_budget() == 64
    assert es._v2_solver_thread_budget(_make_budget_comm(1, 0, None), [0]) == 64
    infos = [("n0", frozenset(range(256)))] * 2
    assert es._v2_solver_thread_budget(_make_budget_comm(2, 0, infos), [0, 1]) == 64


def test_solve_eliashberg_lanczos_v2_threaded_matches_serial():
    """The frequency-distributed solver with a multi-thread budget must return bit-equal eigenvalues and gap
    functions to its serial path (per-k chunk independence, no reordered reduction), driven end to end through
    the real eigsh on a single-rank distributor with a deterministic d-wave seed."""
    from types import SimpleNamespace

    from dgamore.mpi_utils import MpiDistributor
    from tests.conftest import create_comm_mock

    nq, niv_pp, o = (4, 4, 1), 3, 2
    nq_tot, n2 = int(np.prod(nq)), 2 * niv_pp
    config.lattice.nk = nq
    config.lattice.k_grid = bz.KGrid(nq, symmetries=[])
    config.sys.beta = 10.0
    config.eliashberg.n_eig = 2
    config.eliashberg.epsilon = 1e-10
    config.eliashberg.symmetry = "d-wave"
    config.eliashberg.resolve_frequency_parity = False
    config.logger = SimpleNamespace(
        info=lambda *a, **k: None, log_memory_usage=lambda *a, **k: None, warning=lambda *a, **k: None
    )

    rng = np.random.default_rng(11)
    shape = (nq_tot, o, o, o, o, n2, n2)
    gamma_mat = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    chi0_shape = (nq_tot, o, o, o, o, n2)
    chi0 = FourPoint(
        (rng.standard_normal(chi0_shape) + 1j * rng.standard_normal(chi0_shape)),
        SpinChannel.NONE,
        nq,
        0,
        1,
        True,
        True,
        True,
        FrequencyNotation.PP,
    ).decompress_q_dimension()

    def run(n_threads):
        from threadpoolctl import threadpool_limits

        gamma = FourPoint(gamma_mat.copy(), SpinChannel.SING, nq, 0, 2, True, True, True, FrequencyNotation.PP)
        dist = MpiDistributor(ntasks=n2, comm=create_comm_mock())
        from dgamore.eliashberg_solver import solve_eliashberg_lanczos_v2

        # pin BLAS to one thread for both runs so the comparison isolates the momentum-batch threading from
        # nondeterministic multi-threaded BLAS reductions in the serial (n_threads == 1) ARPACK path
        with threadpool_limits(limits=1):
            return solve_eliashberg_lanczos_v2(gamma, chi0, dist, [0], n_threads)["none"]

    lambdas_serial, gaps_serial = run(1)
    lambdas_threaded, gaps_threaded = run(4)
    assert np.array_equal(lambdas_threaded, lambdas_serial)
    for g_threaded, g_serial in zip(gaps_threaded, gaps_serial):
        assert np.array_equal(g_threaded.mat, g_serial.mat)


def test_solve_eliashberg_lanczos_v2_with_inactive_ranks_runs_on_restricted_distributor(monkeypatch):
    """With more ranks than frequency columns the active ranks run the solve on the sub-communicator-restricted
    distributor (its matvec collectives span only them) while the inactive ranks skip it entirely: the solve
    completes without collective mismatch and reproduces the single-rank matvec results (eigsh is replaced by a
    deterministic power loop over the real matvec, since ARPACK's global lock cannot interleave with lockstep
    collectives across threads)."""
    from types import SimpleNamespace

    import dgamore.eliashberg_solver as es
    from dgamore.mpi_utils import MpiDistributor
    from tests.conftest import create_comm_mock, run_parallel

    nq, o, niv_pp = (2, 2, 1), 1, 1
    nq_tot, n2 = int(np.prod(nq)), 2 * niv_pp
    config.lattice.nk = nq
    config.lattice.k_grid = bz.KGrid(nq, symmetries=[])
    config.sys.beta = 10.0
    config.eliashberg.n_eig = 3
    config.eliashberg.epsilon = 1e-10
    config.eliashberg.symmetry = "d-wave"
    config.eliashberg.resolve_frequency_parity = False
    config.eliashberg.symmetrize_degenerate_gaps = False
    config.logger = SimpleNamespace(
        info=lambda *a, **k: None, log_memory_usage=lambda *a, **k: None, warning=lambda *a, **k: None
    )

    rng = np.random.default_rng(17)
    shape = (nq_tot, o, o, o, o, n2, n2)
    gamma_mat = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    chi0_shape = (nq_tot, o, o, o, o, n2)
    chi0 = FourPoint(
        (rng.standard_normal(chi0_shape) + 1j * rng.standard_normal(chi0_shape)),
        SpinChannel.NONE,
        nq,
        0,
        1,
        True,
        True,
        True,
        FrequencyNotation.PP,
    ).decompress_q_dimension()

    def fake_eigsh(op, k, tol, v0, which, maxiter):
        x = v0.astype(np.complex64)
        norms = []
        for _ in range(k):
            x = op.matvec(x).astype(np.complex64)
            nrm = float(np.linalg.norm(x))
            x = x / nrm
            norms.append(nrm)
        return np.array(norms), np.tile(x[:, None], (1, k)).astype(np.complex64)

    monkeypatch.setattr(es.sp.sparse.linalg, "eigsh", fake_eigsh)

    def make_gamma(v_slice):
        mat = gamma_mat[..., v_slice, :].copy()
        return FourPoint(mat, SpinChannel.SING, nq, 0, 2, True, True, True, FrequencyNotation.PP)

    dist_ref = MpiDistributor(ntasks=n2, comm=create_comm_mock())
    lambdas_ref, gaps_ref = es.solve_eliashberg_lanczos_v2(make_gamma(slice(None)), chi0, dist_ref, [0], 1)["none"]

    def fn(comm, rank):
        dist_full = MpiDistributor(ntasks=n2, comm=comm)
        active = [q for q in range(comm.size) if (dist_full.slices[q].stop - dist_full.slices[q].start) > 0]
        sub = comm.Split(0 if rank in active else 1, rank)
        if rank not in active:
            return None
        dist = dist_full.restricted_to(sub, active)
        chi0_arg = chi0 if sub.Get_rank() == 0 else None
        lambdas, gaps = es.solve_eliashberg_lanczos_v2(make_gamma(dist.my_slice), chi0_arg, dist, active, 1)["none"]
        return lambdas, gaps[0].mat

    _, res = run_parallel(4, fn)
    assert res[0] is None and res[1] is None
    for out in res[2:]:
        assert np.allclose(out[0], lambdas_ref, atol=1e-5)
        assert np.allclose(out[1], gaps_ref[0].mat, atol=1e-6)


def test_solve_eliashberg_lanczos_runs_eigsh_inside_thread_budget(monkeypatch):
    """With a multi-core budget the in-memory solver pins the live BLAS pool to one thread around the eigsh call
    via threadpool_limits (the momentum-batch executor threads must not nest BLAS threads underneath; a
    mid-process environment-variable change would be ignored by the already-initialized pool)."""
    import dgamore.eliashberg_solver as es

    nq, niv_pp = (2, 2, 1), 2
    nq_tot, n2 = int(np.prod(nq)), 2 * niv_pp
    config.lattice.nk = nq
    config.lattice.k_grid = bz.KGrid(nq, symmetries=[])
    config.sys.beta = 10.0
    config.eliashberg.n_eig = 1
    config.eliashberg.epsilon = 1e-10
    config.eliashberg.symmetry = "random"
    config.eliashberg.resolve_frequency_parity = False
    config.logger = MagicMock()

    rng = np.random.default_rng(4)
    shape = (nq_tot, 1, 1, 1, 1, n2, n2)
    gamma = FourPoint(
        (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)),
        SpinChannel.SING,
        nq,
        0,
        2,
        True,
        True,
        True,
        FrequencyNotation.PP,
    )
    chi0 = FourPoint(
        (rng.standard_normal(shape[:-1]) + 1j * rng.standard_normal(shape[:-1])),
        SpinChannel.NONE,
        nq,
        0,
        1,
        True,
        True,
        True,
        FrequencyNotation.PP,
    )

    def fake_eigsh(op, k, tol, v0, which, maxiter):
        return np.ones(k), np.ones((op.shape[0], k), dtype=np.complex64)

    limits_seen = []
    from contextlib import contextmanager

    @contextmanager
    def fake_limits(limits):
        limits_seen.append(limits)
        yield

    with monkeypatch.context() as mp:
        mp.setattr(es, "_solver_thread_budget", MagicMock(return_value=3))
        mp.setattr(es, "threadpool_limits", fake_limits)
        mp.setattr("dgamore.eliashberg_solver.sp.sparse.linalg.eigsh", fake_eigsh)
        solve_eliashberg_lanczos(gamma, chi0, (0, 0))
    assert limits_seen == [1]


def _make_p_wave_doublet(nk: int = 6, n2: int = 4) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Builds orthonormal p_x/p_y-like gap columns sin(k_x) g(v) and sin(k_y) g(v) on a small single-band grid."""
    gap_shape = (nk, nk, 1, 1, 1, n2)
    k = 2 * np.pi * np.arange(nk) / nk
    g_v = np.linspace(1.0, 0.5, n2)
    px = (np.sin(k)[:, None, None, None, None, None] * g_v * np.ones(gap_shape)).ravel()
    py = (np.sin(k)[None, :, None, None, None, None] * g_v * np.ones(gap_shape)).ravel()
    px = px / np.linalg.norm(px)
    py = py / np.linalg.norm(py)
    return px.astype(np.complex128), py, gap_shape


def _mirror_y_column(column: np.ndarray, gap_shape: tuple) -> np.ndarray:
    """Applies the mirror k_y -> -k_y to a flattened gap column."""
    idx = (gap_shape[1] - np.arange(gap_shape[1])) % gap_shape[1]
    return column.reshape(gap_shape)[:, idx].ravel()


def test_symmetrize_degenerate_gaps_recovers_mirror_partners():
    """An obliquely mixed, complex-phased degenerate doublet is orthonormalized and rotated back to the
    mirror-adapted p_x-like (even) and p_y-like (odd) partners with deterministic phases; the vector of the
    non-degenerate eigenvalue is only phase-fixed."""
    px, py, gap_shape = _make_p_wave_doublet()
    rng = np.random.default_rng(24)
    extra = rng.standard_normal(px.size) + 1j * rng.standard_normal(px.size)
    extra /= np.linalg.norm(extra)
    mix = np.array([[1.0, 0.6 + 0.3j], [0.2 - 0.4j, 0.8]])
    doublet = np.stack([px, py], axis=1) @ mix
    gaps = np.concatenate([doublet, extra[:, None]], axis=1)
    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5, 0.2]), gaps, gap_shape)
    overlap = fixed[:, :2].conj().T @ fixed[:, :2]
    assert np.allclose(overlap, np.eye(2), atol=1e-12)
    assert np.allclose(_mirror_y_column(fixed[:, 0], gap_shape), fixed[:, 0], atol=1e-12)
    assert np.allclose(_mirror_y_column(fixed[:, 1], gap_shape), -fixed[:, 1], atol=1e-12)
    assert abs(np.vdot(fixed[:, 0], px)) > 1 - 1e-12
    assert abs(np.vdot(fixed[:, 1], py)) > 1 - 1e-12
    assert abs(np.vdot(fixed[:, 2], extra)) > 1 - 1e-12


def test_symmetrize_degenerate_gaps_is_idempotent():
    """Applying the symmetrization twice gives the same result as applying it once (fixed point)."""
    px, py, gap_shape = _make_p_wave_doublet()
    mix = np.array([[1.0, 0.5 - 0.2j], [0.3 + 0.1j, 1.0]])
    gaps = np.stack([px, py], axis=1) @ mix
    once = symmetrize_degenerate_gaps(np.array([0.7, 0.7]), gaps, gap_shape)
    twice = symmetrize_degenerate_gaps(np.array([0.7, 0.7]), once, gap_shape)
    assert np.allclose(once, twice, atol=1e-12)


def test_symmetrize_degenerate_gaps_skips_dependent_vectors():
    """A cluster of (numerically) linearly dependent vectors is left untouched instead of amplifying noise."""
    px, _, gap_shape = _make_p_wave_doublet()
    gaps = np.stack([px, px * (1 + 1e-15)], axis=1)
    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5]), gaps, gap_shape)
    assert np.allclose(fixed, gaps, atol=1e-12)


def _assemble_two_atom_system(niv_pp: int, beta: float) -> tuple:
    """Builds a two-atom (1 + 2 bands) block-structured full chi and Green's function from independent per-atom
    blocks, plus the per-atom inputs for reference."""
    chi_a, _, g_a = _make_pp_chi_and_bubble(1, niv_pp, beta, seed=31)
    chi_b, _, g_b = _make_pp_chi_and_bubble(2, niv_pp, beta, seed=32)
    n2, nvg = 2 * niv_pp, g_a.mat.shape[-1]
    chi_full = np.zeros((3, 3, 3, 3, 1, n2, n2), dtype=complex)
    chi_full[:1, :1, :1, :1] = chi_a.mat
    chi_full[1:, 1:, 1:, 1:] = chi_b.mat
    g_full = np.zeros((3, 3, nvg), dtype=complex)
    g_full[:1, :1] = g_a.mat
    g_full[1:, 1:] = g_b.mat
    chi_full_obj = LocalFourPoint(chi_full, SpinChannel.UD, 1, 2, True, True, FrequencyNotation.PP)
    return chi_full_obj, GreensFunction(g_full), (chi_a, g_a), (chi_b, g_b)


def test_local_gamma_ud_pp_w0_per_ineq_assembles_block_structure():
    """For two inequivalent atoms the full multi-band chi is block-structured and its compound pp matrix singular
    (the La3Ni2O7 crash); the per-ineq driver inverts each atom's block separately, reproduces the direct per-block
    result, and leaves all cross-atom components zero."""
    beta, niv_pp = 10.0, 3
    config.dmft.n_ineq = 2
    config.dmft.ineq_ordering = [1, 2]
    config.dmft.n_bands_per_ineq = [1, 2]
    chi_full_obj, g_full, (chi_a, g_a), (chi_b, g_b) = _assemble_two_atom_system(niv_pp, beta)
    chi0_full = BubbleGenerator.create_generalized_chi0_pp_w0(g_full, niv_pp, beta).extend_vn_to_diagonal()
    with pytest.raises(np.linalg.LinAlgError):
        create_local_gamma_ud_pp_w0(chi_full_obj, chi0_full, beta)
    gamma_full = create_local_gamma_ud_pp_w0_per_ineq(chi_full_obj, g_full, beta)
    chi0_a = BubbleGenerator.create_generalized_chi0_pp_w0(g_a, niv_pp, beta).extend_vn_to_diagonal()
    chi0_b = BubbleGenerator.create_generalized_chi0_pp_w0(g_b, niv_pp, beta).extend_vn_to_diagonal()
    assert np.allclose(gamma_full.mat[:1, :1, :1, :1], create_local_gamma_ud_pp_w0(chi_a, chi0_a, beta).mat, atol=1e-6)
    assert np.allclose(gamma_full.mat[1:, 1:, 1:, 1:], create_local_gamma_ud_pp_w0(chi_b, chi0_b, beta).mat, atol=1e-6)
    cross = gamma_full.mat.copy()
    cross[:1, :1, :1, :1] = 0.0
    cross[1:, 1:, 1:, 1:] = 0.0
    assert np.allclose(cross, 0.0, atol=1e-12)


def test_local_gamma_ud_pp_w0_per_ineq_reuses_repeated_atoms():
    """Repeated entries in ineq_ordering (the same inequivalent atom at several positions) are computed once and
    written identically into every position."""
    beta, niv_pp = 10.0, 3
    config.dmft.n_ineq = 1
    config.dmft.ineq_ordering = [1, 1]
    config.dmft.n_bands_per_ineq = [2]
    chi_a, _, g_a = _make_pp_chi_and_bubble(2, niv_pp, beta, seed=33)
    n2, nvg = 2 * niv_pp, g_a.mat.shape[-1]
    chi_full = np.zeros((4, 4, 4, 4, 1, n2, n2), dtype=complex)
    g_full = np.zeros((4, 4, nvg), dtype=complex)
    for sl in (slice(0, 2), slice(2, 4)):
        chi_full[sl, sl, sl, sl] = chi_a.mat
        g_full[sl, sl] = g_a.mat
    chi_full_obj = LocalFourPoint(chi_full, SpinChannel.UD, 1, 2, True, True, FrequencyNotation.PP)
    gamma_full = create_local_gamma_ud_pp_w0_per_ineq(chi_full_obj, GreensFunction(g_full), beta)
    assert np.allclose(gamma_full.mat[:2, :2, :2, :2], gamma_full.mat[2:, 2:, 2:, 2:], atol=1e-12)
    chi0_a = BubbleGenerator.create_generalized_chi0_pp_w0(g_a, niv_pp, beta).extend_vn_to_diagonal()
    assert np.allclose(gamma_full.mat[:2, :2, :2, :2], create_local_gamma_ud_pp_w0(chi_a, chi0_a, beta).mat, atol=1e-6)


# --- frequency-parity gap sectors (physical-gap projection) ---
def _flip_nu(g: np.ndarray) -> np.ndarray:
    """T involution on a [kx, ky, kz, o1, o2, v] gap array (nu -> -nu)."""
    return np.flip(g, axis=-1)


def _flip_po(g: np.ndarray) -> np.ndarray:
    """P.O involution on a [kx, ky, kz, o1, o2, v] gap array (k -> -k and o1 <-> o2)."""
    return np.roll(np.flip(g.swapaxes(3, 4), axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))


@pytest.mark.parametrize("resolve, expected", [(True, [("even", 1), ("odd", -1)]), (False, [("none", None)])])
def test_frequency_parity_sectors_maps_boolean(resolve, expected):
    """_frequency_parity_sectors returns the even and odd sectors when resolving frequency parity, else the raw sector."""
    assert _frequency_parity_sectors(resolve) == expected


@pytest.mark.parametrize("eps_t, eps_po", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_project_gap_to_sector_is_idempotent_and_selects_parities(eps_t, eps_po):
    """The sector projector is idempotent and its image has T-parity eps_t and (P.O)-parity eps_po."""
    shape = (2, 2, 1, 2, 2, 4)
    rng = np.random.default_rng(7)
    vec = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).ravel()
    proj = _project_gap_to_sector(vec, shape, eps_t, eps_po)
    assert np.allclose(_project_gap_to_sector(proj, shape, eps_t, eps_po), proj, atol=1e-12)
    g = proj.reshape(shape)
    assert np.allclose(_flip_nu(g), eps_t * g, atol=1e-12)
    assert np.allclose(_flip_po(g), eps_po * g, atol=1e-12)


def test_project_gap_to_sector_is_hermitian_and_partitions_identity():
    """The four sector projectors are Hermitian and sum to the identity over a channel (even/odd x its forced P.O)."""
    shape = (2, 2, 1, 2, 2, 4)
    n = int(np.prod(shape))
    basis = np.eye(n, dtype=np.complex128)
    total = np.zeros((n, n), dtype=np.complex128)
    for eps_t, eps_po in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        proj = np.column_stack([_project_gap_to_sector(basis[:, i], shape, eps_t, eps_po) for i in range(n)])
        assert np.allclose(proj, proj.conj().T, atol=1e-12)
        total += proj
    assert np.allclose(total, np.eye(n), atol=1e-12)


@pytest.mark.parametrize("eps_t, eps_po", [(1, -1), (-1, 1)])
def test_gap_parity_diagnostics_reports_projected_parities(eps_t, eps_po):
    """_gap_parity_diagnostics returns the T and P.O Rayleigh quotients of a gap projected into a known sector."""
    shape = (2, 2, 1, 2, 2, 4)
    rng = np.random.default_rng(3)
    vec = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).ravel()
    diag = gap_parity_diagnostics(_project_gap_to_sector(vec, shape, eps_t, eps_po), shape)
    assert np.isclose(diag["T"].real, eps_t, atol=1e-10) and np.isclose(diag["T"].imag, 0.0, atol=1e-10)
    assert np.isclose(diag["PO"].real, eps_po, atol=1e-10) and np.isclose(diag["PO"].imag, 0.0, atol=1e-10)


def test_gap_parity_diagnostics_zero_gap_returns_zeros():
    """_gap_parity_diagnostics returns zero Rayleigh quotients (no division by zero) for an all-zero gap."""
    shape = (2, 2, 1, 1, 1, 4)
    diag = gap_parity_diagnostics(np.zeros(int(np.prod(shape)), dtype=np.complex128), shape)
    assert all(diag[key] == 0 for key in ("T", "P", "O", "PO"))


def _wave_gap(form: np.ndarray, parity: str, niv: int = 2) -> np.ndarray:
    """Builds a [kx, ky, 1, 1, 1, 2*niv] gap carrying the [kx, ky] momentum form with the given frequency parity."""
    gap = np.zeros(form.shape + (1, 1, 1, 2 * niv), dtype=np.complex128)
    positive = form[:, :, None, None, None, None]
    gap[..., niv:] = positive
    gap[..., :niv] = positive if parity == "even" else -positive
    return gap


_K4 = 2 * np.pi * np.arange(4) / 4
_S_FORM = np.cos(_K4)[:, None] + np.cos(_K4)[None, :]
_D_FORM = np.cos(_K4)[:, None] - np.cos(_K4)[None, :]
_P_FORM = np.broadcast_to(np.sin(_K4)[:, None], (4, 4)).astype(float).copy()


@pytest.mark.parametrize(
    "form, parity, expected",
    [
        (_S_FORM, "even", "s+"),
        (_D_FORM, "even", "d+"),
        (_D_FORM, "odd", "d-"),
        (_P_FORM, "odd", "p-"),
        (_P_FORM, "even", "p+"),
    ],
)
def test_classify_gap_symmetry_labels_wave_and_frequency_parity(form, parity, expected):
    """classify_gap_symmetry labels the spatial wave (s/d/p) and the frequency parity (+/-) of a gap."""
    assert classify_gap_symmetry(_wave_gap(form.astype(complex), parity)) == expected


def test_classify_gap_symmetry_returns_unknown_for_zero_gap():
    """classify_gap_symmetry returns 'unknown' when the positive-frequency momentum slice is identically zero."""
    assert classify_gap_symmetry(np.zeros((4, 4, 1, 1, 1, 4), dtype=complex)) == "unknown"


def _single_band_pp_operands(nq: tuple, niv_pp: int, seed: int) -> tuple[FourPoint, FourPoint]:
    """Builds a random single-band pp pairing vertex and bare pp bubble on the given momentum grid for solver tests."""
    config.lattice.nk = nq
    config.lattice.k_grid = bz.KGrid(nq, symmetries=[])
    config.sys.beta = 10.0
    config.logger = MagicMock()
    nq_tot, n2 = int(np.prod(nq)), 2 * niv_pp
    rng = np.random.default_rng(seed)
    gshape = (nq_tot, 1, 1, 1, 1, n2, n2)
    gamma = FourPoint(
        rng.standard_normal(gshape) + 1j * rng.standard_normal(gshape),
        SpinChannel.SING,
        nq,
        0,
        2,
        True,
        True,
        True,
        FrequencyNotation.PP,
    )
    chi0 = FourPoint(
        rng.standard_normal(gshape[:-1]) + 1j * rng.standard_normal(gshape[:-1]),
        SpinChannel.NONE,
        nq,
        0,
        1,
        True,
        True,
        True,
        FrequencyNotation.PP,
    )
    return gamma, chi0


def test_solve_eliashberg_lanczos_both_projects_matvec_and_v0_into_each_sector(monkeypatch):
    """With resolve_frequency_parity set the singlet solve returns an even and an odd sector, and each sector's eigsh
    operator and seed are projected: the matvec output and v0 carry that sector's T-parity and P.O-parity exactly."""
    nq, niv_pp = (4, 4, 1), 2
    gap_shape = nq + (1, 1) + (2 * niv_pp,)
    config.eliashberg.n_eig = 2
    config.eliashberg.epsilon = 1e-10
    config.eliashberg.symmetry = "random"
    config.eliashberg.resolve_frequency_parity = True
    gamma, chi0 = _single_band_pp_operands(nq, niv_pp, seed=11)

    seen = []

    def fake_eigsh(op, k, tol, v0, which, maxiter):
        n = op.shape[0]
        rng = np.random.default_rng(1)
        probe = op.matvec(rng.standard_normal(n) + 1j * rng.standard_normal(n))
        seen.append((v0.copy(), probe))
        return np.arange(k, 0, -1).astype(float), np.ones((n, k), dtype=np.complex64)

    with monkeypatch.context() as mp:
        mp.setattr("dgamore.eliashberg_solver.sp.sparse.linalg.eigsh", fake_eigsh)
        result = solve_eliashberg_lanczos(gamma, chi0, (0, 0))

    assert set(result) == {"even", "odd"}
    for parity, (lambdas, gaps) in result.items():
        assert len(gaps) == config.eliashberg.n_eig and len(lambdas) == config.eliashberg.n_eig
    for (parity, eps_t, eps_po), (v0, probe) in zip([("even", 1, 1), ("odd", -1, -1)], seen):
        for vec in (v0, probe):
            g = vec.reshape(gap_shape)
            assert np.allclose(_flip_nu(g), eps_t * g, atol=1e-5)
            assert np.allclose(_flip_po(g), eps_po * g, atol=1e-5)


def _densify_pairing_kernel(monkeypatch, gamma_arr: np.ndarray, chi0_arr: np.ndarray, nq: tuple, channel) -> np.ndarray:
    """Densifies the production pairing-kernel matvec (unprojected) by intercepting eigsh and hitting the operator
    with every standard-basis column."""
    config.lattice.nk = nq
    config.lattice.k_grid = bz.KGrid(nq, symmetries=[])
    config.sys.beta = 10.0
    config.eliashberg.n_eig = 1
    config.eliashberg.epsilon = 1e-10
    config.eliashberg.symmetry = "random"
    config.eliashberg.resolve_frequency_parity = False
    config.eliashberg.symmetrize_degenerate_gaps = False
    config.logger = MagicMock()
    dense = []

    def fake_eigsh(op, k, tol, v0, which, maxiter):
        n = op.shape[0]
        dense.append(np.column_stack([op.matvec(np.eye(n, dtype=np.complex128)[:, i]) for i in range(n)]))
        return np.ones(k), np.ones((n, k), dtype=np.complex128)

    with monkeypatch.context() as mp:
        mp.setattr("dgamore.eliashberg_solver.sp.sparse.linalg.eigsh", fake_eigsh)
        solve_eliashberg_lanczos(
            FourPoint(gamma_arr.copy(), channel, nq, 0, 2, True, True, True, FrequencyNotation.PP),
            FourPoint(chi0_arr.copy(), SpinChannel.NONE, nq, 0, 1, True, True, True, FrequencyNotation.PP),
            (0, 0),
        )
    return dense[0]


def _symmetrize_tr_inversion(arr: np.ndarray, nq: tuple, two_fermion: bool) -> np.ndarray:
    """Averages a compressed-q pp array with its Gamma(q, v, v') = Gamma(-q, -v, -v') image (time reversal plus
    inversion), the symmetry that makes the pairing kernel conserve fermionic-frequency parity."""
    grid = arr.reshape(nq + arr.shape[1:])
    flipped = np.roll(np.flip(grid, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
    flipped = np.flip(flipped, axis=(-1, -2)) if two_fermion else np.flip(flipped, axis=-1)
    return (0.5 * (grid + flipped)).reshape(arr.shape)


@pytest.mark.parametrize("channel", [SpinChannel.SING, SpinChannel.TRIP])
def test_kernel_conserves_frequency_parity_only_for_tr_inversion_symmetric_vertex(monkeypatch, channel):
    """The densified pairing kernel commutes with the T (nu -> -nu) involution exactly when the pairing vertex and
    bubble carry the time-reversal-plus-inversion symmetry Gamma(q, v, v') = Gamma(-q, -v, -v') - the precondition
    for splitting the gap into frequency-even and frequency-odd sectors - and fails to for a generic vertex."""
    nq, o, niv_pp = (2, 2, 1), 2, 2
    nq_tot, n2 = int(np.prod(nq)), 2 * niv_pp
    rng = np.random.default_rng(0)
    gamma = rng.standard_normal((nq_tot, o, o, o, o, n2, n2)) + 1j * rng.standard_normal((nq_tot, o, o, o, o, n2, n2))
    chi0 = rng.standard_normal((nq_tot, o, o, o, o, n2)) + 1j * rng.standard_normal((nq_tot, o, o, o, o, n2))
    gap_shape = nq + (o, o) + (n2,)

    m_generic = _densify_pairing_kernel(monkeypatch, gamma, chi0, nq, channel)
    m_symm = _densify_pairing_kernel(
        monkeypatch,
        _symmetrize_tr_inversion(gamma, nq, two_fermion=True),
        _symmetrize_tr_inversion(chi0, nq, two_fermion=False),
        nq,
        channel,
    )
    n = m_generic.shape[0]
    t_mat = np.column_stack(
        [np.flip(np.eye(n, dtype=complex)[:, i].reshape(gap_shape), axis=-1).reshape(-1) for i in range(n)]
    )
    generic_commutator = np.abs(m_generic @ t_mat - t_mat @ m_generic).max() / np.abs(m_generic).max()
    symm_commutator = np.abs(m_symm @ t_mat - t_mat @ m_symm).max() / np.abs(m_symm).max()
    assert symm_commutator < 1e-5
    assert generic_commutator > 1e-1


def test_solve_eliashberg_lanczos_reseeds_when_symmetry_seed_orthogonal_to_sector(monkeypatch):
    """When the configured momentum symmetry seeds a frequency parity orthogonal to the requested sector (a d-wave
    even-nu singlet seed projected onto the odd sector collapses), the solver reseeds so the eigsh start is nonzero
    and carries the sector's parities."""
    nq, niv_pp = (4, 4, 1), 2
    gap_shape = nq + (1, 1) + (2 * niv_pp,)
    config.eliashberg.n_eig = 1
    config.eliashberg.epsilon = 1e-10
    config.eliashberg.symmetry = "d-wave"
    config.eliashberg.resolve_frequency_parity = True
    gamma, chi0 = _single_band_pp_operands(nq, niv_pp, seed=5)

    seeds = []

    def fake_eigsh(op, k, tol, v0, which, maxiter):
        seeds.append(v0.copy())
        n = op.shape[0]
        return np.arange(k, 0, -1).astype(float), np.ones((n, k), dtype=np.complex64)

    with monkeypatch.context() as mp:
        mp.setattr("dgamore.eliashberg_solver.sp.sparse.linalg.eigsh", fake_eigsh)
        result = solve_eliashberg_lanczos(gamma, chi0, (0, 0))

    assert set(result) == {"even", "odd"}
    v0_odd = seeds[1].reshape(gap_shape)
    assert np.linalg.norm(v0_odd) > 0
    assert np.allclose(_flip_nu(v0_odd), -v0_odd, atol=1e-6) and np.allclose(_flip_po(v0_odd), -v0_odd, atol=1e-6)


def test_solve_eliashberg_lanczos_seeds_empty_sector_with_nonzero_base(monkeypatch):
    """On a self-dual grid where every k equals -k (2x2) the P-odd singlet sector is empty, so the projected seed
    and its random reseed both collapse; the solver still hands eigsh the nonzero base seed rather than a zero
    vector."""
    nq, niv_pp = (2, 2, 1), 2
    config.eliashberg.n_eig = 1
    config.eliashberg.epsilon = 1e-10
    config.eliashberg.symmetry = "random"
    config.eliashberg.resolve_frequency_parity = True
    config.eliashberg.symmetrize_degenerate_gaps = False
    gamma, chi0 = _single_band_pp_operands(nq, niv_pp, seed=8)

    seeds = []

    def fake_eigsh(op, k, tol, v0, which, maxiter):
        seeds.append(v0.copy())
        n = op.shape[0]
        return np.zeros(k), np.zeros((n, k), dtype=np.complex64)

    with monkeypatch.context() as mp:
        mp.setattr("dgamore.eliashberg_solver.sp.sparse.linalg.eigsh", fake_eigsh)
        solve_eliashberg_lanczos(gamma, chi0, (0, 0))

    assert np.linalg.norm(seeds[1]) > 0


def test_solve_eliashberg_lanczos_none_returns_single_unprojected_sector(monkeypatch):
    """With resolve_frequency_parity disabled the solve returns one 'none' sector whose matvec is the raw
    (unprojected) pairing kernel."""
    nq, niv_pp = (2, 2, 1), 2
    config.eliashberg.n_eig = 2
    config.eliashberg.epsilon = 1e-10
    config.eliashberg.symmetry = "random"
    config.eliashberg.resolve_frequency_parity = False
    gamma, chi0 = _single_band_pp_operands(nq, niv_pp, seed=12)

    calls = []

    def fake_eigsh(op, k, tol, v0, which, maxiter):
        calls.append(op)
        n = op.shape[0]
        return np.arange(k, 0, -1).astype(float), np.ones((n, k), dtype=np.complex64)

    with monkeypatch.context() as mp:
        mp.setattr("dgamore.eliashberg_solver.sp.sparse.linalg.eigsh", fake_eigsh)
        result = solve_eliashberg_lanczos(gamma, chi0, (0, 0))

    assert set(result) == {"none"}
    assert len(calls) == 1 and len(result["none"][1]) == config.eliashberg.n_eig

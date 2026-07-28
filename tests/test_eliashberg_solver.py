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
    _compute_once_per_node,
    _frequency_parity_sectors,
    _gap_orbital_mirrors,
    _mirror_acts_on_orbitals,
    _mirror_operator,
    _orient_cluster_by_mirrors,
    _sector_log_label,
    _validated_orbital_mirrors,
    gap_parity_diagnostics,
    _project_gap_to_sector,
    classify_gap_symmetry,
    get_initial_gap_function,
    create_local_f_ud_transformed_w0,
    create_local_gamma_ud_pp_w0,
    create_local_gamma_ud_pp_w0_per_ineq,
    _gamma_to_matmul_layout,
    get_ranks_for_lanczos,
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
    """_apply_gamma_pp reproduces einsum('xyzacbdvp,xyzcdp->xyzabv') for full (nv==np) and sliced (nv<np) v."""
    rng = np.random.default_rng(1)
    nqx, nqy, nqz, o, npp = 2, 3, 2, 3, 6
    shape = (nqx, nqy, nqz, o, o, o, o, nv, npp)
    gamma = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    gap_gg = rng.standard_normal((nqx, nqy, nqz, o, o, npp)) + 1j * rng.standard_normal((nqx, nqy, nqz, o, o, npp))
    ref = np.einsum("xyzacbdvp,xyzcdp->xyzabv", gamma, gap_gg, optimize=True)
    got = _apply_gamma_pp(_gamma_to_matmul_layout(gamma), gap_gg, o)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-10)


@pytest.mark.parametrize("o", [1, 2])
def test_apply_gamma_pp_is_insensitive_to_flipped_rhs_contiguity(o):
    """Materializing the crossed term's flipped-gap RHS contiguously must not change _apply_gamma_pp vs the raw view."""
    rng = np.random.default_rng(7)
    nqx, nqy, nqz, npp = 3, 4, 2, 6
    shape = (nqx, nqy, nqz, o, o, o, o, npp, npp)
    gamma_mm = _gamma_to_matmul_layout(
        (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    )
    gap_gg = (
        rng.standard_normal((nqx, nqy, nqz, o, o, npp)) + 1j * rng.standard_normal((nqx, nqy, nqz, o, o, npp))
    ).astype(np.complex64)
    view = _apply_gamma_pp(gamma_mm, np.flip(gap_gg, axis=-1), o)
    contiguous = _apply_gamma_pp(gamma_mm, np.ascontiguousarray(np.flip(gap_gg, axis=-1)), o)
    assert np.allclose(view, contiguous, atol=1e-5)


def _make_pp_chi_and_bubble(
    o: int, niv_pp: int, beta: float, seed: int
) -> tuple[LocalFourPoint, LocalFourPoint, GreensFunction]:
    """Builds a random crossing-symmetric pp susceptibility (J chi J = chi) and its matching diagonal bare pp bubble."""
    rng = np.random.default_rng(seed)
    g_mat = rng.standard_normal((o, o, 2 * (niv_pp + 2))) + 1j * rng.standard_normal((o, o, 2 * (niv_pp + 2)))
    g_mat = 0.5 * (g_mat + g_mat.transpose(1, 0, 2)) + 2.0 * np.eye(o)[:, :, None]
    g = GreensFunction(g_mat[None, None, None, ...])
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
    """Conjugates a full-index pp tensor with the crossing operator J (swap both orbital pairs, flip both v)."""
    return np.einsum("abcdwvp->cdabwvp", mat)[..., ::-1, ::-1]


def test_crossed_term_reuses_direct_vertex_via_index_shuffles():
    """The crossed matvec term equals sign*flip_K[swap_ab[Gamma @ flip_p(gap)]], reusing the direct vertex."""
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
    """The trimmed-window pp transform equals the untrimmed reference gather for niw far above the read window."""
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
    """The eager-rebound first term of the full ladder vertex equals the original expression, bubble untouched."""
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
    """For one band and crossing-symmetric chi the J-decorated BSE inversion equals the old flipped-bubble B.26 form."""
    beta, niv_pp = 12.5, 4
    chi, chi0, _ = _make_pp_chi_and_bubble(1, niv_pp, beta, seed=7)
    chi0_flipped = chi0.flip_frequency_axis(-1)
    gamma_old = ((chi - chi0_flipped).invert() + chi0_flipped.invert()).scale(beta**2)
    gamma_new = create_local_gamma_ud_pp_w0(chi, chi0, beta)
    assert gamma_new.mat.shape == gamma_old.mat.shape
    assert np.allclose(gamma_new.mat, gamma_old.mat, atol=1e-3)


@pytest.mark.parametrize("o", [2, 3, 4, 5])
def test_local_gamma_ud_pp_w0_multiorbital_preserves_crossing_symmetry(o):
    """For >1 band the J-decorated inversion keeps Gamma crossing-symmetric and differs from the old form."""
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
    """create_local_gamma_ud_pp_w0 matches the float64 compound Gamma and satisfies the crossing-decoupled pp BSE."""
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
    gv = g.mat[0, 0, 0][:, :, 2:-2].astype(np.complex128)
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
    """Gamma's J-even/odd blocks fulfill the decoupled singlet/triplet pp BSEs of thesis Eqs. (3.51)/(3.52)."""
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
    """transform_vertex_loc_frequencies_w0 returns the crossed slot -F_{1432} (the permute shows only for >1 band)."""
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
    """The 'abcd->badc' permute makes the contraction TRIQS-order; w2dynamics order agrees only for one band."""
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
    """Averages a rank-4 orbital tensor over the group generated by the given (permutation, character) pairs."""
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
    """Applies the Eliashberg matvec direct term for one slice: pp bubble times gap, then the badc-permuted vertex."""
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
    """On the physical symmetry class the matvec direct term equals thesis Eq. (4.40); a generic vertex differs."""
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
    """For a swap- and TR-symmetric pairing vertex the badc permute is a no-op, so w2dynamics and TRIQS orders agree."""
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
    """o decoupled degenerate bands give o single-band kernel copies with an o-fold spectrum."""
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
    """The solver thread budget is the affinity-mask size (at least 1) and falls back to 1 without the affinity API."""
    import os

    import dgamore.eliashberg_solver as es

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    assert es._solver_thread_budget() == 4
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert es._solver_thread_budget() == 1


def test_apply_gamma_pp_momentum_parallel_path_is_bit_equal():
    """The momentum-parallel _apply_gamma_pp is bit-equal to serial, even when workers don't divide the batch."""
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
    """The momentum-parallel _apply_gchi0_pp is bit-equal to serial, even when workers don't divide the batch."""
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
    """The v2 budget is this rank's mask size divided by the active node ranks sharing it; idle ranks free theirs."""
    import dgamore.eliashberg_solver as es

    monkeypatch.setattr(es.os, "sched_getaffinity", lambda pid: masks[rank], raising=False)
    infos = [(hosts[r], frozenset(masks[r])) for r in range(len(masks))]
    comm = _make_budget_comm(len(masks), rank, infos)
    assert es._v2_solver_thread_budget(comm, active_ranks) == expected


def test_v2_thread_budget_falls_back_without_affinity_api_and_single_rank(monkeypatch):
    """Without the affinity API the v2 budget is 1; a single-rank comm gets the whole mask with no collective call."""
    import dgamore.eliashberg_solver as es

    monkeypatch.delattr(es.os, "sched_getaffinity", raising=False)
    assert es._v2_solver_thread_budget(_make_budget_comm(4, 0, None), [0, 1, 2, 3]) == 1
    monkeypatch.setattr(es.os, "sched_getaffinity", lambda pid: set(range(6)), raising=False)
    assert es._v2_solver_thread_budget(_make_budget_comm(1, 0, None), [0]) == 6


def test_v2_thread_budget_needs_no_initialized_mpi(monkeypatch):
    """The v2 budget groups ranks by socket.gethostname, so it needs no MPI.Get_processor_name or MPI_Init."""
    import dgamore.eliashberg_solver as es

    monkeypatch.setattr(es.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)
    monkeypatch.setattr(es, "_openblas_thread_slot_cap", MagicMock(return_value=None))
    monkeypatch.setattr(
        es.MPI, "Get_processor_name", MagicMock(side_effect=RuntimeError("MPI not initialized")), raising=False
    )
    infos = [("n0", frozenset(range(8)))] * 2
    assert es._v2_solver_thread_budget(_make_budget_comm(2, 0, infos), [0, 1]) == 4


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
    """The threaded frequency-distributed solver reproduces the eigenvalues and gaps of its serial path."""
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
        info=lambda *a, **k: None,
        log_memory_usage=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
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

        # pin BLAS for both runs where threadpoolctl can (it does not reach Apple's Accelerate), so the momentum
        # batching is compared under as close a numerical stack as the platform allows
        with threadpool_limits(limits=1):
            return solve_eliashberg_lanczos_v2(gamma, chi0, dist, [0], n_threads)["none"]

    lambdas_serial, gaps_serial = run(1)
    lambdas_threaded, gaps_threaded = run(4)
    assert np.allclose(lambdas_threaded, lambdas_serial, atol=1e-5)
    for g_threaded, g_serial in zip(gaps_threaded, gaps_serial):
        assert np.allclose(g_threaded.mat, g_serial.mat, atol=1e-3)


def test_solve_eliashberg_lanczos_v2_with_inactive_ranks_runs_on_restricted_distributor(monkeypatch):
    """With more ranks than frequency columns the active ranks solve on the restricted distributor without mismatch."""
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
        info=lambda *a, **k: None,
        log_memory_usage=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
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
    """The in-memory solver pins the BLAS pool to one thread around eigsh so the executor doesn't nest BLAS."""
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
    """An oblique degenerate doublet is orthonormalized and rotated back to mirror-adapted p_x/p_y partners."""
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
    """A cluster of (numerically) linearly dependent vectors is not orthonormalized or rotated, only normalized."""
    px, _, gap_shape = _make_p_wave_doublet()
    gaps = np.stack([px, px * (1 + 1e-15)], axis=1)
    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5]), gaps, gap_shape)
    # each column is the input up to normalization and a global phase: no S^{-1/2} amplification, no mirror rotation
    for col in range(2):
        overlap = np.vdot(gaps[:, col], fixed[:, col])
        assert np.isclose(abs(overlap), np.linalg.norm(gaps[:, col]) * np.linalg.norm(fixed[:, col]), rtol=1e-12)


def test_symmetrize_degenerate_gaps_phase_fixes_dependent_vectors():
    """Even the linearly dependent cluster obeys the phase convention: largest-magnitude element real and positive."""
    px, _, gap_shape = _make_p_wave_doublet()
    gaps = np.stack([px, px * (1 + 1e-15)], axis=1) * np.exp(1j * 0.7)
    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5]), gaps, gap_shape)
    for col in range(2):
        assert np.isclose(np.linalg.norm(fixed[:, col]), 1.0, atol=1e-12)
        leading = fixed[np.argmax(np.abs(fixed[:, col])), col]
        assert leading.real > 0 and np.isclose(leading.imag, 0.0, atol=1e-12)


def _make_p_wave_triplet(nk: int = 4, n2: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Builds orthonormal p_x/p_y/p_z-like gap columns sin(k_i) g(v) on a small three-dimensional single-band grid."""
    gap_shape = (nk, nk, nk, 1, 1, n2)
    k = 2 * np.pi * np.arange(nk) / nk
    g_v = np.linspace(1.0, 0.5, n2)
    px = np.sin(k)[:, None, None, None, None, None] * g_v * np.ones(gap_shape)
    py = np.sin(k)[None, :, None, None, None, None] * g_v * np.ones(gap_shape)
    pz = np.sin(k)[None, None, :, None, None, None] * g_v * np.ones(gap_shape)
    columns = [(c.ravel() / np.linalg.norm(c)).astype(np.complex128) for c in (px, py, pz)]
    return columns[0], columns[1], columns[2], gap_shape


def _mirror_axis_column(column: np.ndarray, gap_shape: tuple, axis: int) -> np.ndarray:
    """Applies the single-axis mirror k_axis -> -k_axis to a flattened gap column."""
    idx = (gap_shape[axis] - np.arange(gap_shape[axis])) % gap_shape[axis]
    take = [slice(None)] * len(gap_shape)
    take[axis] = idx
    return column.reshape(gap_shape)[tuple(take)].ravel()


def test_symmetrize_degenerate_gaps_recovers_triplet_partners():
    """An oblique degenerate triplet is rotated to p_x/p_y/p_z partners ordered by the axis each one is odd under."""
    px, py, pz, gap_shape = _make_p_wave_triplet()
    rng = np.random.default_rng(7)
    mix = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    gaps = np.stack([px, py, pz], axis=1) @ mix
    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5, 0.5]), gaps, gap_shape)
    assert np.allclose(fixed.conj().T @ fixed, np.eye(3), atol=1e-10)
    for col, (parent, odd_axis) in enumerate(((px, 0), (py, 1), (pz, 2))):
        assert abs(np.vdot(fixed[:, col], parent)) > 1 - 1e-10
        for axis in range(3):
            sign = -1.0 if axis == odd_axis else 1.0
            assert np.allclose(_mirror_axis_column(fixed[:, col], gap_shape, axis), sign * fixed[:, col], atol=1e-10)


def test_symmetrize_degenerate_gaps_triplet_is_idempotent():
    """Applying the symmetrization twice to a degenerate triplet gives the same result as applying it once."""
    px, py, pz, gap_shape = _make_p_wave_triplet()
    rng = np.random.default_rng(11)
    gaps = np.stack([px, py, pz], axis=1) @ (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))
    lambdas = np.array([0.6, 0.6, 0.6])
    once = symmetrize_degenerate_gaps(lambdas, gaps, gap_shape)
    twice = symmetrize_degenerate_gaps(lambdas, once, gap_shape)
    assert np.allclose(once, twice, atol=1e-10)


def _make_d_wave_triplet(nk: int = 4, n2: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Builds orthonormal d_xy/d_xz/d_yz-like gap columns sin(k_i) sin(k_j) g(v) on a small three-dimensional grid."""
    gap_shape = (nk, nk, nk, 1, 1, n2)
    s = np.sin(2 * np.pi * np.arange(nk) / nk)
    g_v = np.linspace(1.0, 0.5, n2)
    dxy = s[:, None, None, None, None, None] * s[None, :, None, None, None, None] * g_v * np.ones(gap_shape)
    dxz = s[:, None, None, None, None, None] * s[None, None, :, None, None, None] * g_v * np.ones(gap_shape)
    dyz = s[None, :, None, None, None, None] * s[None, None, :, None, None, None] * g_v * np.ones(gap_shape)
    columns = [(c.ravel() / np.linalg.norm(c)).astype(np.complex128) for c in (dxy, dxz, dyz)]
    return columns[0], columns[1], columns[2], gap_shape


def test_symmetrize_degenerate_gaps_orders_two_axis_gaps_by_odd_axis_pair():
    """A degenerate d_xy/d_xz/d_yz triplet is resolved and ordered by the pair of axes each partner is odd under."""
    dxy, dxz, dyz, gap_shape = _make_d_wave_triplet()
    rng = np.random.default_rng(7)
    mix = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    gaps = np.stack([dxy, dxz, dyz], axis=1) @ mix
    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5, 0.5]), gaps, gap_shape)
    assert np.allclose(fixed.conj().T @ fixed, np.eye(3), atol=1e-10)
    for col, (parent, odd_axes) in enumerate(((dxy, {0, 1}), (dxz, {0, 2}), (dyz, {1, 2}))):
        assert abs(np.vdot(fixed[:, col], parent)) > 1 - 1e-10
        for axis in range(3):
            sign = -1.0 if axis in odd_axes else 1.0
            assert np.allclose(_mirror_axis_column(fixed[:, col], gap_shape, axis), sign * fixed[:, col], atol=1e-10)


def test_orient_cluster_by_mirrors_falls_back_for_non_mirror_cluster():
    """_orient_cluster_by_mirrors returns None when the cluster vectors are not clean +/-1 mirror eigenstates."""
    gap_shape = (4, 4, 4, 1, 1, 4)
    rng = np.random.default_rng(3)
    n = int(np.prod(gap_shape))
    block, _ = np.linalg.qr(rng.standard_normal((n, 3)) + 1j * rng.standard_normal((n, 3)))
    assert _orient_cluster_by_mirrors(block, gap_shape) is None


def test_orient_cluster_by_mirrors_falls_back_without_resolved_axes():
    """_orient_cluster_by_mirrors returns None on a single-k-point grid where no momentum axis is reflected."""
    gap_shape = (1, 1, 1, 1, 1, 4)
    block, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((4, 3)) + 0j)
    assert _orient_cluster_by_mirrors(block, gap_shape) is None


def _make_local_orbital_triplet(nk: int = 4, nb: int = 3, n2: int = 4) -> tuple[np.ndarray, tuple]:
    """Builds an orthonormal momentum-independent triplet of antisymmetric orbital gap functions g(v) e_{o1 o2}."""
    gap_shape = (nk, nk, nk, nb, nb, n2)
    g_v = np.linspace(1.0, 0.5, n2)
    columns = []
    for o1, o2 in ((0, 1), (0, 2), (1, 2)):
        c = np.zeros(gap_shape, dtype=np.complex128)
        c[:, :, :, o1, o2] = g_v
        c[:, :, :, o2, o1] = -g_v
        columns.append(c.ravel() / np.linalg.norm(c))
    return np.stack(columns, axis=1), gap_shape


def test_orient_cluster_by_mirrors_falls_back_for_degenerate_local_cluster():
    """A momentum-independent triplet is even under every mirror, so the mirrors cannot resolve it."""
    block, gap_shape = _make_local_orbital_triplet()
    assert _orient_cluster_by_mirrors(block, gap_shape) is None


def test_orient_cluster_by_mirrors_falls_back_when_one_axis_cannot_separate_three_partners():
    """With a single resolved axis two of three partners must share a sign pattern, which stays unresolved."""
    gap_shape = (6, 1, 1, 1, 1, 4)
    k = 2 * np.pi * np.arange(gap_shape[0]) / gap_shape[0]
    g_v = np.linspace(1.0, 0.5, gap_shape[-1])
    even = np.ones(gap_shape) * g_v
    odd = np.sin(k)[:, None, None, None, None, None] * g_v * np.ones(gap_shape)
    even_2 = np.cos(k)[:, None, None, None, None, None] * g_v * np.ones(gap_shape)
    cols = [(c.ravel() / np.linalg.norm(c)).astype(np.complex128) for c in (even, odd, even_2)]
    block, _ = np.linalg.qr(np.stack(cols, axis=1))
    assert _orient_cluster_by_mirrors(block, gap_shape) is None


def test_symmetrize_degenerate_gaps_keeps_loewdin_basis_for_local_cluster():
    """An unresolvable momentum-independent triplet is kept as is instead of rotated arbitrarily within itself."""
    block, gap_shape = _make_local_orbital_triplet()
    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5, 0.5]), block, gap_shape)
    assert np.allclose(fixed, block, atol=1e-12)


def test_symmetrize_degenerate_gaps_keeps_contaminated_local_cluster_unrotated():
    """The k-dependent noise a real solve leaves on a local triplet must not tip it into an arbitrary rotation."""
    block, gap_shape = _make_local_orbital_triplet()
    rng = np.random.default_rng(17)
    noise = rng.standard_normal(block.shape) + 1j * rng.standard_normal(block.shape)
    block = block + 1e-7 * noise / np.linalg.norm(noise, axis=0)
    eigs, u = np.linalg.eigh(block.conj().T @ block)
    block = block @ (u @ np.diag(eigs**-0.5) @ u.conj().T)  # the Loewdin basis the caller must keep
    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5, 0.5]), block, gap_shape)
    for col in range(3):
        assert abs(np.vdot(fixed[:, col], block[:, col])) > 1 - 1e-12


def test_orient_cluster_by_mirrors_resolves_a_p_triplet_without_diagonal_mirrors():
    """On a non-square grid the coordinate mirrors alone (no diagonal family) still resolve a p_x/p_y/p_z triplet."""
    gap_shape = (4, 6, 5, 1, 1, 4)
    s = [np.sin(2 * np.pi * np.arange(n) / n) for n in gap_shape[:3]]
    g_v = np.linspace(1.0, 0.5, gap_shape[-1])
    px = s[0][:, None, None, None, None, None] * g_v * np.ones(gap_shape)
    py = s[1][None, :, None, None, None, None] * g_v * np.ones(gap_shape)
    pz = s[2][None, None, :, None, None, None] * g_v * np.ones(gap_shape)
    cols = [(c.ravel() / np.linalg.norm(c)).astype(np.complex128) for c in (px, py, pz)]
    rng = np.random.default_rng(2)
    block, _ = np.linalg.qr(np.stack(cols, axis=1) @ (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))))
    fixed = _orient_cluster_by_mirrors(block, gap_shape)
    assert fixed is not None
    for col, odd_axis in enumerate((0, 1, 2)):
        for axis in range(3):
            sign = -1.0 if axis == odd_axis else 1.0
            assert np.allclose(_mirror_axis_column(fixed[:, col], gap_shape, axis), sign * fixed[:, col], atol=1e-10)


def test_symmetrize_degenerate_gaps_canonicalizes_diagonal_p_doublet_to_axis_basis():
    """An obliquely mixed p_{x+y}/p_{x-y} doublet is canonicalized to the coordinate p_x/p_y basis by the M_y mirror."""
    px, py, gap_shape = _make_p_wave_doublet()
    pxy = (px + py) / np.linalg.norm(px + py)
    pxmy = (px - py) / np.linalg.norm(px - py)
    rng = np.random.default_rng(9)
    gaps = np.stack([pxy, pxmy], axis=1) @ (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)))
    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5]), gaps, gap_shape)
    assert abs(np.vdot(fixed[:, 0], px)) > 1 - 1e-10
    assert abs(np.vdot(fixed[:, 1], py)) > 1 - 1e-10


def _assemble_two_atom_system(niv_pp: int, beta: float) -> tuple:
    """Builds a two-atom (1+2 band) block-structured full chi and G plus the per-atom reference inputs."""
    chi_a, _, g_a = _make_pp_chi_and_bubble(1, niv_pp, beta, seed=31)
    chi_b, _, g_b = _make_pp_chi_and_bubble(2, niv_pp, beta, seed=32)
    n2, nvg = 2 * niv_pp, g_a.mat.shape[-1]
    chi_full = np.zeros((3, 3, 3, 3, 1, n2, n2), dtype=complex)
    chi_full[:1, :1, :1, :1] = chi_a.mat
    chi_full[1:, 1:, 1:, 1:] = chi_b.mat
    g_full = np.zeros((1, 1, 1, 3, 3, nvg), dtype=complex)
    g_full[..., :1, :1, :] = g_a.mat
    g_full[..., 1:, 1:, :] = g_b.mat
    chi_full_obj = LocalFourPoint(chi_full, SpinChannel.UD, 1, 2, True, True, FrequencyNotation.PP)
    return chi_full_obj, GreensFunction(g_full), (chi_a, g_a), (chi_b, g_b)


def test_local_gamma_ud_pp_w0_per_ineq_assembles_block_structure():
    """For two inequivalent atoms the per-ineq driver inverts each atom's block separately, cross-atom zero."""
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
    """Repeated ineq_ordering entries are computed once and written identically into every position."""
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
    """_frequency_parity_sectors returns the even and odd sectors when resolving parity, else the raw sector."""
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


def _orbital_gap(blocks: dict, n_orb: int, nk: int = 6, niv: int = 2) -> np.ndarray:
    """Builds an even-in-nu [k,k,k,n_orb,n_orb,2*niv] gap from a {(o1, o2): [kx,ky,kz] form} mapping."""
    gap = np.zeros((nk, nk, nk, n_orb, n_orb, 2 * niv), dtype=complex)
    for (o1, o2), form in blocks.items():
        gap[:, :, :, o1, o2, niv:] = form[..., None]
    gap[..., :niv] = gap[..., niv:]
    return gap


def _cubic_forms(nk: int = 6) -> tuple:
    """Returns (cos kx, cos ky, cos kz) broadcast over a [nk, nk, nk] grid."""
    k = 2 * np.pi * np.arange(nk) / nk
    return np.cos(k)[:, None, None], np.cos(k)[None, :, None], np.cos(k)[None, None, :]


def test_classify_gap_symmetry_reads_every_orbital_block():
    """A gap living entirely in an orbital-off-diagonal block is classified, not written off as undetermined."""
    cx, cy, _ = _cubic_forms()
    gap = _orbital_gap({(0, 1): cx - cy, (1, 0): cx - cy}, n_orb=2)
    assert classify_gap_symmetry(gap) == "d+"


def test_classify_gap_symmetry_collapses_per_block_d_wave_partners():
    """The cubic t2g Eg gap carries a different d-wave in each orbital block; all are d, so the label collapses."""
    cx, cy, cz = _cubic_forms()
    gap = _orbital_gap({(0, 0): cx - cy, (1, 1): cz - cx, (2, 2): cy - cz}, n_orb=3)
    assert classify_gap_symmetry(gap) == "d+"


def test_classify_gap_symmetry_tolerates_small_admixture():
    """A gap that is 95 % extended s-wave is labeled s, where an element-wise comparison would reject it."""
    cx, cy, cz = _cubic_forms()
    gap = _orbital_gap({(0, 0): 1.0 + 0.05 * (cx + cy + cz)}, n_orb=1)
    assert classify_gap_symmetry(gap) == "s+"


def test_classify_gap_symmetry_lists_blocks_when_waves_disagree():
    """Blocks carrying genuinely different waves are reported separately, ordered by weight."""
    cx, cy, cz = _cubic_forms()
    gap = _orbital_gap({(0, 0): cx - cy, (1, 1): cx + cy + cz}, n_orb=2)
    label = classify_gap_symmetry(gap)
    assert "d+[00]" in label and "s+[11]" in label and "|" in label


def test_classify_gap_symmetry_ignores_negligible_blocks():
    """A block far below the weight floor carries no physical symmetry and must not enter the label."""
    cx, cy, cz = _cubic_forms()
    gap = _orbital_gap({(0, 0): cx - cy, (1, 1): 1e-6 * (cx + cy + cz)}, n_orb=2)
    assert classify_gap_symmetry(gap) == "d+"


@pytest.mark.parametrize("symmetry", ["random", "d-wave"])
def test_get_initial_gap_function_is_deterministic(symmetry):
    """The start vector is reproducible across calls, so a run (and the basis of a degenerate multiplet) repeats."""
    config.lattice.nk = (4, 4, 1)
    config.lattice.k_grid = bz.KGrid((4, 4, 1), symmetries=[])
    config.eliashberg.symmetry = symmetry
    shape = (4, 4, 1, 1, 1, 4)
    first = get_initial_gap_function(shape, SpinChannel.SING)
    second = get_initial_gap_function(shape, SpinChannel.SING)
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    "symmetry, channel, eps_t",
    [
        ("d-wave", SpinChannel.SING, 1),
        ("d-wave", SpinChannel.TRIP, -1),
        ("p-wave-x", SpinChannel.SING, -1),
        ("p-wave-x", SpinChannel.TRIP, 1),
    ],
)
def test_get_initial_gap_function_seeds_the_requested_frequency_parity(symmetry, channel, eps_t):
    """The seeded start vector is an eigenvector of the frequency flip T with the parity its channel requires."""
    config.lattice.nk = (4, 4, 1)
    config.lattice.k_grid = bz.KGrid((4, 4, 1), symmetries=[])
    config.eliashberg.symmetry = symmetry
    gap = get_initial_gap_function((4, 4, 1, 1, 1, 4), channel)
    assert np.allclose(np.flip(gap, axis=-1), eps_t * gap, atol=1e-12)


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
    """With parity resolved the singlet solve returns even and odd sectors whose matvec and v0 carry the parities."""
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
    """Densifies the unprojected pairing-kernel matvec by intercepting eigsh and hitting every basis column."""
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
    """Averages a compressed-q pp array with its Gamma(q,v,v')=Gamma(-q,-v,-v') (time-reversal-plus-inversion) image."""
    grid = arr.reshape(nq + arr.shape[1:])
    flipped = np.roll(np.flip(grid, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
    flipped = np.flip(flipped, axis=(-1, -2)) if two_fermion else np.flip(flipped, axis=-1)
    return (0.5 * (grid + flipped)).reshape(arr.shape)


@pytest.mark.parametrize("channel", [SpinChannel.SING, SpinChannel.TRIP])
def test_kernel_conserves_frequency_parity_only_for_tr_inversion_symmetric_vertex(monkeypatch, channel):
    """The densified kernel commutes with nu -> -nu for a TR-plus-inversion-symmetric vertex, not for a generic one."""
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
    """When the symmetry seed is orthogonal to the sector the solver reseeds so eigsh starts nonzero with parity."""
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
    """On a self-dual 2x2 grid the P-odd singlet sector is empty, so the solver seeds eigsh with the nonzero base."""
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
    """With parity resolution off the solve returns one 'none' sector whose matvec is the raw unprojected kernel."""
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


@pytest.mark.parametrize(
    "info, per_sector, giwk, expected",
    [
        # 4 nodes (one rank each), memory fits one vertex per node -> the four sectors spread over the four nodes
        ([("n0", 100), ("n1", 100), ("n2", 100), ("n3", 100)], 90, 0, ([0, 1], [2, 3])),
        # one node with four ranks and ample memory -> all four sectors run concurrently on that node
        ([("n0", 10000)] * 4, 10, 0, ([0, 1], [2, 3])),
        # one node whose memory only fits two vertices -> the singlet/triplet 2-way (sectors sequential per channel)
        ([("n0", 250)] * 4, 100, 0, ([0, 0], [1, 1])),
        # the bubble-node giwk reservation eats the memory down to one vertex -> everything sequential on one rank
        ([("n0", 1000)] * 4, 100, 800, ([0, 0], [0, 0])),
    ],
)
def test_get_ranks_for_lanczos_packs_sectors_by_node_memory(info, per_sector, giwk, expected):
    """get_ranks_for_lanczos packs as many concurrent sector solves per node as free memory fits (giwk reserved)."""
    comm = MagicMock()
    comm.allgather.side_effect = lambda payload: info
    assert get_ranks_for_lanczos(comm, 2, info[0][1], per_sector, giwk) == expected


@pytest.mark.parametrize(
    "hostnames, n_parities, expected",
    [
        (["n0", "n0", "n1", "n1"], 2, ([0, 0], [2, 2])),
        (["n0", "n0", "n0", "n0"], 2, ([0, 0], [1, 1])),
        (["n0"], 2, ([0, 0], [0, 0])),
        (["n0", "n1"], 1, ([0], [1])),
    ],
)
def test_get_ranks_for_lanczos_falls_back_to_two_way_without_memory_info(hostnames, n_parities, expected):
    """Without a per-sector memory estimate get_ranks_for_lanczos returns the 2-way split, parities sequential."""
    comm = MagicMock()
    comm.allgather.side_effect = lambda payload: [(h, None) for h in hostnames]
    assert get_ranks_for_lanczos(comm, n_parities) == expected


def test_solve_eliashberg_lanczos_parities_subset_solves_only_requested(monkeypatch):
    """Passing an explicit parities subset restricts the solve (and eigsh calls) to just those sectors."""
    nq, niv_pp = (4, 4, 1), 2
    config.eliashberg.n_eig = 2
    config.eliashberg.epsilon = 1e-10
    config.eliashberg.symmetry = "random"
    config.eliashberg.resolve_frequency_parity = True
    gamma, chi0 = _single_band_pp_operands(nq, niv_pp, seed=13)

    calls = []

    def fake_eigsh(op, k, tol, v0, which, maxiter):
        calls.append(1)
        n = op.shape[0]
        return np.arange(k, 0, -1).astype(float), np.ones((n, k), dtype=np.complex64)

    with monkeypatch.context() as mp:
        mp.setattr("dgamore.eliashberg_solver.sp.sparse.linalg.eigsh", fake_eigsh)
        result = solve_eliashberg_lanczos(gamma, chi0, (0, 0), parities=["odd"])

    assert set(result) == {"odd"} and len(calls) == 1


def test_solve_sectors_in_memory_runs_each_sector_on_its_assigned_rank(monkeypatch):
    """The in-memory sector distribution solves every (channel, parity) sector on its assigned rank."""
    import dgamore.eliashberg_solver as es
    from dgamore.bubble_gen import BubbleGenerator
    from dgamore.gap_function import GapFunction
    from dgamore.greens_function import GreensFunction
    from dgamore.mpi_utils import MpiDistributor
    from tests.conftest import FAKE_MPI, _tls, run_parallel

    monkeypatch.setattr("dgamore.mpi_utils.MPI", FAKE_MPI)
    nq, o, niv_pp, ntasks = (2, 2, 1), 1, 2, 8
    n2 = 2 * niv_pp
    config.lattice.nk = nq
    config.lattice.k_grid = bz.KGrid(nq, symmetries=[])
    config.sys.beta = 10.0
    config.eliashberg.n_eig = 2
    config.logger = MagicMock()

    def fake_solver(gamma, gchi0, ranks, parities=None):
        rank = getattr(_tls, "rank", 0)
        n_eig = config.eliashberg.n_eig
        return {
            p: (
                np.array([rank + 1] * n_eig, dtype=float),
                [GapFunction(np.zeros(nq + (o, o, n2), dtype=np.complex64)) for _ in range(n_eig)],
            )
            for p in parities
        }

    def fake_bubble(giwk, niv, k_grid):
        return FourPoint(
            np.zeros((ntasks, 1, 1, 1, 1, n2)), SpinChannel.NONE, nq, 0, 1, True, True, True, FrequencyNotation.PP
        )

    monkeypatch.setattr(es, "solve_eliashberg_lanczos", fake_solver)
    monkeypatch.setattr(BubbleGenerator, "create_generalized_chi0_q_pp_w0", fake_bubble)

    def fn(comm, rank):
        dist = MpiDistributor(ntasks=ntasks, comm=comm)
        shape = (dist.my_size, o, o, o, o, n2, n2)
        gsing = FourPoint(
            np.zeros(shape, dtype=np.complex64), SpinChannel.SING, nq, 0, 2, True, True, True, FrequencyNotation.PP
        )
        gtrip = FourPoint(
            np.zeros(shape, dtype=np.complex64), SpinChannel.TRIP, nq, 0, 2, True, True, True, FrequencyNotation.PP
        )
        giwk = GreensFunction(np.zeros(nq + (o, o, 10), dtype=np.complex64), nk=nq)
        return es._solve_sectors_in_memory(dist, gsing, gtrip, giwk, niv_pp, [0, 2], [1, 3], 0, ["even", "odd"])

    _, results = run_parallel(4, fn)
    owners = {
        (SpinChannel.SING, "even"): 1,
        (SpinChannel.SING, "odd"): 3,
        (SpinChannel.TRIP, "even"): 2,
        (SpinChannel.TRIP, "odd"): 4,
    }
    for res in results:
        assert set(res) == set(owners)
        for key, expected_marker in owners.items():
            assert res[key][0][0] == expected_marker and len(res[key][1]) == config.eliashberg.n_eig


def test_compute_once_per_node_evaluates_the_builder_only_on_the_node_root():
    """_compute_once_per_node runs the builder once per node and hands that result to the node's other ranks."""
    from tests.conftest import FAKE_MPI, run_parallel

    def fn(comm, r):
        node_comm = comm.Split_type(FAKE_MPI.COMM_TYPE_SHARED)
        return _compute_once_per_node(node_comm, lambda: f"built-by-{r}")

    _, results = run_parallel(4, fn, hostnames=["a", "a", "b", "b"])
    assert results == ["built-by-0", "built-by-0", "built-by-2", "built-by-2"]


def test_compute_once_per_node_evaluates_the_builder_without_a_node_communicator():
    """_compute_once_per_node falls back to a plain local evaluation when no node communicator is given."""
    assert _compute_once_per_node(None, lambda: "built") == "built"


def test_create_local_f_ud_transformed_w0_is_the_w0_transform_of_the_half_difference(monkeypatch):
    """create_local_f_ud_transformed_w0 returns the pp-w0 transform of 0.5 (F_dens - F_magn) read from file."""
    rng = np.random.default_rng(3)
    shape = (5, 1, 1, 1, 1, 8, 8)
    f_dens = LocalFourPoint(rng.random(shape) + 1j * rng.random(shape), SpinChannel.DENS, 1, 2, False, True)
    f_magn = LocalFourPoint(rng.random(shape) + 1j * rng.random(shape), SpinChannel.MAGN, 1, 2, False, True)
    stored = {SpinChannel.DENS: f_dens, SpinChannel.MAGN: f_magn}
    monkeypatch.setattr(
        LocalFourPoint, "load", staticmethod(lambda filename, channel=SpinChannel.NONE: stored[channel].copy())
    )
    expected = transform_vertex_loc_frequencies_w0((f_dens - f_magn).set_channel(SpinChannel.UD).scale(0.5), 2)
    assert np.allclose(create_local_f_ud_transformed_w0(2).mat, expected.mat, atol=1e-6)


@pytest.mark.parametrize("o", [1, 2])
def test_local_vertex_pp_transforms_are_blind_to_a_nu_prime_restricted_vertex(o):
    """Both pp transforms of the local full vertex ignore the nu' columns outside the core box they cut to."""
    niw, niv_core, niv_full, niv_pp = 6, 8, 12, 3
    rng = np.random.default_rng(5)
    shape = (o, o, o, o, niw + 1, 2 * niv_full, 2 * niv_full)
    full = LocalFourPoint(
        (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64),
        SpinChannel.DENS,
        1,
        2,
        False,
        True,
    )
    cut = LocalFourPoint(
        full.mat[..., niv_full - niv_core : niv_full + niv_core].copy(), SpinChannel.DENS, 1, 2, False, True
    )

    assert cut.niv_first == niv_full and cut.niv_second == niv_core
    assert np.array_equal(
        transform_vertex_loc_frequencies_w0(full.copy(), niv_pp).mat,
        transform_vertex_loc_frequencies_w0(cut.copy(), niv_pp).mat,
    )
    assert np.array_equal(
        full.copy().cut_niv(niv_pp).change_frequency_notation_ph_to_pp_w0().mat,
        cut.copy().cut_niv(niv_pp).change_frequency_notation_ph_to_pp_w0().mat,
    )


@pytest.mark.parametrize(
    "resolve, parities, expected",
    [
        (False, None, "the singlet channel"),
        (True, None, "the singlet/even & odd sectors"),
        (True, ["odd"], "the singlet/odd sector"),
        (True, ["even"], "the singlet/even sector"),
    ],
)
def test_sector_log_label_names_the_sectors_a_call_covers(resolve, parities, expected):
    """The log label names the covered sectors, so concurrently solving ranks never emit identical lines."""
    config.eliashberg.resolve_frequency_parity = resolve
    assert _sector_log_label(SpinChannel.SING, parities) == expected


def test_sector_log_label_distinguishes_the_channels():
    """Both channels get their own label, so the four concurrent sector solves are told apart in the log."""
    config.eliashberg.resolve_frequency_parity = True
    labels = [_sector_log_label(c, [p]) for c in (SpinChannel.SING, SpinChannel.TRIP) for p in ("even", "odd")]
    assert len(set(labels)) == 4


# ============================================================================
# Combined momentum-plus-orbital mirrors

# orbital order d_xy, d_xz, d_yz; a mirror k_i -> -k_i flips the two orbitals that extend along k_i
_T2G_MIRROR_SIGNS = {0: (-1.0, -1.0, 1.0), 1: (-1.0, 1.0, -1.0), 2: (1.0, -1.0, -1.0)}


def _t2g_orbital_mirrors() -> dict:
    """The orbital part {axis: U} of the coordinate mirrors for three t2g orbitals."""
    return {axis: np.diag(signs).astype(np.complex128) for axis, signs in _T2G_MIRROR_SIGNS.items()}


def _combined_mirror_column(column: np.ndarray, gap_shape: tuple, axis: int, u: np.ndarray = None) -> np.ndarray:
    """Applies Delta(k) -> U Delta(M_axis k) U^dag to a flattened gap column (momentum only when ``u`` is None)."""
    idx = (gap_shape[axis] - np.arange(gap_shape[axis])) % gap_shape[axis]
    take = [slice(None)] * len(gap_shape)
    take[axis] = idx
    reflected = column.reshape(gap_shape)[tuple(take)]
    if u is None:
        return reflected.ravel()
    return np.einsum("ap,xyzpqv,bq->xyzabv", u, reflected, np.conj(u)).ravel()


def _mirror_signature(column: np.ndarray, gap_shape: tuple, mirrors: dict = None) -> np.ndarray:
    """The three mirror Rayleigh quotients <Delta|T_axis|Delta> of a normalized gap column."""
    mirrors = mirrors or {}
    return np.array(
        [
            float(np.vdot(column, _combined_mirror_column(column, gap_shape, axis, mirrors.get(axis))).real)
            for axis in range(3)
        ]
    )


def _make_t2g_mixed_triplet(nk: int = 4, n2: int = 4, off_weight: float = 0.8) -> tuple[np.ndarray, tuple]:
    """
    Builds an orthonormal t2g triplet of known mirror parity whose partners are exact +/-1 eigenstates of the
    combined momentum-plus-orbital mirrors but not of the momentum-only ones. Each partner carries an
    orbital-diagonal piece (s_o s_o = +1, so its sign comes from the momentum parity alone) plus an off-diagonal
    piece in one orbital pair (s_{o1} s_{o2} = -1 on two of the three axes), chosen so that both pieces end up with
    the same combined sign. The three partners have mirror signatures (+,-,-), (-,+,-) and (-,-,+).
    """
    gap_shape = (nk, nk, nk, 3, 3, n2)
    s = np.sin(2 * np.pi * np.arange(nk) / nk)
    g_v = np.linspace(1.0, 0.5, n2)
    sx, sy, sz = s[:, None, None, None], s[None, :, None, None], s[None, None, :, None]
    columns = []
    for diag_f, (o1, o2) in ((sy * sz, (0, 1)), (sx * sz, (0, 2)), (sx * sy, (1, 2))):
        c = np.zeros(gap_shape, dtype=np.complex128)
        c[:, :, :, 0, 0] = diag_f * g_v
        c[:, :, :, o1, o2] = off_weight * g_v
        c[:, :, :, o2, o1] = off_weight * g_v
        columns.append(c.ravel() / np.linalg.norm(c))
    return np.stack(columns, axis=1), gap_shape


def _make_eg_like_doublet(nk: int = 4, n2: int = 4) -> tuple[np.ndarray, tuple]:
    """
    Builds an orthonormal orbital-diagonal doublet that is even under every coordinate mirror - the E_g situation
    the coordinate mirrors provably cannot split, since an orbital-diagonal state picks up s_o s_o = +1 everywhere.
    """
    gap_shape = (nk, nk, nk, 2, 2, n2)
    c_k = np.cos(2 * np.pi * np.arange(nk) / nk)
    g_v = np.linspace(1.0, 0.5, n2)
    forms = (np.ones((nk, nk, nk, 1)), (c_k[:, None, None] * c_k[None, :, None])[..., None])
    columns = []
    for orbital, form in enumerate(forms):
        c = np.zeros(gap_shape, dtype=np.complex128)
        c[:, :, :, orbital, orbital] = form * g_v
        columns.append(c.ravel() / np.linalg.norm(c))
    return np.stack(columns, axis=1), gap_shape


def _cubic_t2g_hamiltonian(nk: int = 8, t: float = 0.25, tp: float = 0.03, ti: float = 0.02) -> np.ndarray:
    """
    Cubic t2g Hamiltonian in the orbital order d_xy, d_xz, d_yz (kept in sync with the copy in
    test_symmetry_reduction.py). The inter-orbital terms carry the only momentum dependence compatible with the t2g
    mirror signs, which is what lets the mirror solver pin the signs down.
    """
    k = 2 * np.pi * np.arange(nk) / nk
    kx, ky, kz = k[:, None, None], k[None, :, None], k[None, None, :]
    c, s = np.cos, np.sin
    h = np.zeros((nk, nk, nk, 3, 3), dtype=complex)
    h[..., 0, 0] = -2 * t * (c(kx) + c(ky)) - 2 * tp * c(kz)
    h[..., 1, 1] = -2 * t * (c(kx) + c(kz)) - 2 * tp * c(ky)
    h[..., 2, 2] = -2 * t * (c(ky) + c(kz)) - 2 * tp * c(kx)
    h[..., 0, 1] = h[..., 1, 0] = ti * s(ky) * s(kz)
    h[..., 0, 2] = h[..., 2, 0] = ti * s(kx) * s(kz)
    h[..., 1, 2] = h[..., 2, 1] = ti * s(kx) * s(ky)
    return h


def _mix(block: np.ndarray, seed: int) -> np.ndarray:
    """Scrambles a cluster into an oblique, non-orthogonal basis the way ARPACK may hand one back."""
    rng = np.random.default_rng(seed)
    n = block.shape[1]
    return block @ (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))


def _loewdin(block: np.ndarray) -> np.ndarray:
    """The Loewdin basis symmetrize_degenerate_gaps falls back to: column-normalize, then apply S^{-1/2}."""
    block = block / np.linalg.norm(block, axis=0)
    eigs, u = np.linalg.eigh(block.conj().T @ block)
    return block @ (u @ np.diag(eigs**-0.5) @ u.conj().T)


def test_mirror_acts_on_orbitals_only_for_non_scalar_matrices():
    """Conjugation by U is the identity map exactly when U is a multiple of the identity (a global phase cancels)."""
    assert not _mirror_acts_on_orbitals(None)
    assert not _mirror_acts_on_orbitals(np.eye(1, dtype=complex))
    assert not _mirror_acts_on_orbitals(np.eye(3, dtype=complex))
    assert not _mirror_acts_on_orbitals(-np.eye(3, dtype=complex))
    assert not _mirror_acts_on_orbitals(np.exp(0.7j) * np.eye(3))
    assert _mirror_acts_on_orbitals(np.diag([-1.0, -1.0, 1.0]).astype(complex))


def test_validated_orbital_mirrors_keeps_only_matching_orbital_dimensions():
    """Mirrors discovered for a different orbital count must never be applied silently."""
    gap_shape = (4, 4, 4, 3, 3, 4)
    mirrors = {0: np.eye(3), 1: np.eye(2), 2: None}

    validated = _validated_orbital_mirrors(gap_shape, mirrors)

    assert sorted(validated) == [0]
    assert validated[0].dtype == np.complex128
    assert _validated_orbital_mirrors(gap_shape, None) == {}


def test_mirror_operator_applies_the_orbital_signs_on_top_of_the_reflection():
    """The operator is Delta_{o1 o2}(k) -> s_{o1} s_{o2} Delta_{o1 o2}(M k) for a diagonal-sign orbital mirror."""
    gap_shape = (4, 4, 4, 3, 3, 2)
    rng = np.random.default_rng(4)
    column = (rng.standard_normal(gap_shape) + 1j * rng.standard_normal(gap_shape)).ravel()

    for axis, signs in _T2G_MIRROR_SIGNS.items():
        got = _mirror_operator(gap_shape, axis, np.diag(signs).astype(np.complex128))(column)
        reflected = _combined_mirror_column(column, gap_shape, axis).reshape(gap_shape)
        expected = (np.array(signs)[:, None] * np.array(signs)[None, :])[None, None, None, :, :, None] * reflected
        assert np.allclose(got, expected.ravel(), atol=1e-12)


def test_mirror_operator_is_an_involution_and_norm_preserving():
    """Applying the combined mirror twice is the identity, so its eigenvalues can only be +/-1."""
    gap_shape = (4, 4, 4, 3, 3, 2)
    rng = np.random.default_rng(6)
    column = (rng.standard_normal(gap_shape) + 1j * rng.standard_normal(gap_shape)).ravel()

    for axis, u in _t2g_orbital_mirrors().items():
        op = _mirror_operator(gap_shape, axis, u)
        assert np.allclose(op(op(column)), column, atol=1e-12)
        assert np.isclose(np.linalg.norm(op(column)), np.linalg.norm(column))


def test_mirror_operator_without_an_orbital_matrix_is_a_pure_momentum_reflection():
    """A missing or scalar orbital matrix leaves the historical momentum-only reflection untouched."""
    gap_shape = (4, 4, 4, 2, 2, 2)
    rng = np.random.default_rng(8)
    column = (rng.standard_normal(gap_shape) + 1j * rng.standard_normal(gap_shape)).ravel()

    for u in (None, np.eye(2, dtype=complex), np.exp(1.1j) * np.eye(2)):
        got = _mirror_operator(gap_shape, 1, u)(column)
        assert np.allclose(got, _combined_mirror_column(column, gap_shape, 1), atol=1e-12)


def test_momentum_only_mirror_measures_diagonal_minus_off_diagonal_weight():
    """Without the orbital factor the mirror quotient reads the diagonal-minus-off-diagonal weight, not +/-1."""
    block, gap_shape = _make_t2g_mixed_triplet()

    for col in range(3):
        column = block[:, col]
        weights = np.abs(column.reshape(gap_shape)) ** 2
        diagonal = float(np.einsum("xyzoov->", weights))
        difference = abs(diagonal - (1.0 - diagonal))
        signature = _mirror_signature(column, gap_shape)
        assert 0.1 < diagonal < 0.9  # substantial weight in both orbital sectors
        assert np.allclose(sorted(np.abs(signature)), sorted([1.0, difference, difference]), atol=1e-10), signature


def test_orient_cluster_by_mirrors_falls_back_without_orbital_mirrors_for_a_multi_orbital_multiplet():
    """Momentum-only mirrors are not a symmetry of a multi-orbital kernel, so the multiplet is left alone."""
    block, gap_shape = _make_t2g_mixed_triplet()

    assert _orient_cluster_by_mirrors(np.linalg.qr(_mix(block, 5))[0], gap_shape) is None


def test_orient_cluster_by_mirrors_resolves_a_multi_orbital_multiplet_with_orbital_mirrors():
    """With the orbital factor the partners are clean +/-1 eigenstates, ordered by the axes they are odd under."""
    block, gap_shape = _make_t2g_mixed_triplet()
    mirrors = _t2g_orbital_mirrors()

    fixed = _orient_cluster_by_mirrors(np.linalg.qr(_mix(block, 5))[0], gap_shape, orbital_mirrors=mirrors)

    assert fixed is not None
    assert np.allclose(fixed.conj().T @ fixed, np.eye(3), atol=1e-10)
    # sorted lexicographically by odd axes: (0,1), (0,2), (1,2) - i.e. the input triplet in reverse
    for col, (parent, expected) in enumerate(
        ((block[:, 2], (-1.0, -1.0, 1.0)), (block[:, 1], (-1.0, 1.0, -1.0)), (block[:, 0], (1.0, -1.0, -1.0)))
    ):
        signature = _mirror_signature(fixed[:, col], gap_shape, mirrors)
        assert np.allclose(signature, expected, atol=1e-10), signature
        for axis in range(3):
            mirrored = _combined_mirror_column(fixed[:, col], gap_shape, axis, mirrors[axis])
            assert np.allclose(mirrored, expected[axis] * fixed[:, col], atol=1e-10)
        assert abs(np.vdot(fixed[:, col], parent)) > 1 - 1e-10


def test_orient_cluster_by_mirrors_resolves_a_purely_local_triplet_with_orbital_mirrors():
    """A momentum-independent triplet is invisible to momentum-only mirrors but split by the orbital factor."""
    block, gap_shape = _make_local_orbital_triplet()
    mirrors = _t2g_orbital_mirrors()

    assert _orient_cluster_by_mirrors(block, gap_shape) is None  # momentum-only mirrors act as the identity here

    fixed = _orient_cluster_by_mirrors(block, gap_shape, orbital_mirrors=mirrors)

    assert fixed is not None
    for col, (parent, expected) in enumerate(
        ((block[:, 2], (-1.0, -1.0, 1.0)), (block[:, 1], (-1.0, 1.0, -1.0)), (block[:, 0], (1.0, -1.0, -1.0)))
    ):
        assert np.allclose(_mirror_signature(fixed[:, col], gap_shape, mirrors), expected, atol=1e-12)
        assert abs(np.vdot(fixed[:, col], parent)) > 1 - 1e-12


def test_symmetrize_degenerate_gaps_returns_clean_mirror_eigenstates_for_a_multi_orbital_triplet():
    """End to end: an oblique multi-orbital triplet comes back orthonormal and exactly +/-1 under every mirror."""
    block, gap_shape = _make_t2g_mixed_triplet()
    mirrors = _t2g_orbital_mirrors()

    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5, 0.5]), _mix(block, 12), gap_shape, orbital_mirrors=mirrors)

    assert np.allclose(fixed.conj().T @ fixed, np.eye(3), atol=1e-10)
    for col in range(3):
        signature = _mirror_signature(fixed[:, col], gap_shape, mirrors)
        assert np.allclose(np.abs(signature), 1.0, atol=1e-10), signature
        for axis in range(3):
            mirrored = _combined_mirror_column(fixed[:, col], gap_shape, axis, mirrors[axis])
            assert np.allclose(mirrored, signature[axis] * fixed[:, col], atol=1e-10)
    patterns = {tuple(np.sign(_mirror_signature(fixed[:, c], gap_shape, mirrors))) for c in range(3)}
    assert len(patterns) == 3  # no two partners share a sign pattern


def test_symmetrize_degenerate_gaps_without_orbital_mirrors_keeps_the_loewdin_basis():
    """The same triplet, symmetrized with momentum-only mirrors, must be left in its Loewdin basis."""
    block, gap_shape = _make_t2g_mixed_triplet()
    oblique = _mix(block, 12)
    loewdin = _loewdin(oblique)

    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5, 0.5]), oblique, gap_shape)

    for col in range(3):
        assert abs(np.vdot(fixed[:, col], loewdin[:, col])) > 1 - 1e-10


def test_symmetrize_degenerate_gaps_is_idempotent_for_a_multi_orbital_triplet():
    """Symmetrizing twice with the orbital mirrors reproduces the first result exactly."""
    block, gap_shape = _make_t2g_mixed_triplet()
    mirrors = _t2g_orbital_mirrors()
    lambdas = np.array([0.5, 0.5, 0.5])

    once = symmetrize_degenerate_gaps(lambdas, _mix(block, 3), gap_shape, orbital_mirrors=mirrors)
    twice = symmetrize_degenerate_gaps(lambdas, once, gap_shape, orbital_mirrors=mirrors)

    assert np.allclose(once, twice, atol=1e-10)


def test_symmetrize_degenerate_gaps_resolves_a_multi_orbital_doublet_with_orbital_mirrors():
    """The two-fold cluster path needs the orbital factor too: M_y splits the pair only once it is included."""
    block, gap_shape = _make_t2g_mixed_triplet()
    mirrors = _t2g_orbital_mirrors()
    pair = block[:, :2]  # partner 0 is odd and partner 1 even under M_y, so M_y alone resolves this pair

    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5]), _mix(pair, 21), gap_shape, orbital_mirrors=mirrors)

    assert np.allclose(fixed.conj().T @ fixed, np.eye(2), atol=1e-10)
    for col, (parent, sign) in enumerate(((block[:, 1], 1.0), (block[:, 0], -1.0))):
        mirrored = _combined_mirror_column(fixed[:, col], gap_shape, 1, mirrors[1])
        assert np.allclose(mirrored, sign * fixed[:, col], atol=1e-10)
        assert abs(np.vdot(fixed[:, col], parent)) > 1 - 1e-10


def test_symmetrize_degenerate_gaps_leaves_a_multi_orbital_doublet_alone_without_orbital_mirrors():
    """Without the orbital factor M_y is not a symmetry of the pair, so the Loewdin basis is kept."""
    block, gap_shape = _make_t2g_mixed_triplet()
    oblique = _mix(block[:, :2], 21)
    loewdin = _loewdin(oblique)

    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5]), oblique, gap_shape)

    for col in range(2):
        assert abs(np.vdot(fixed[:, col], loewdin[:, col])) > 1 - 1e-10


def test_symmetrize_degenerate_gaps_keeps_the_loewdin_basis_for_an_eg_like_doublet():
    """Coordinate mirrors cannot split an E_g doublet, so the pair keeps its Loewdin basis (not a noise rotation)."""
    block, gap_shape = _make_eg_like_doublet()
    mirrors = {axis: np.diag([1.0, -1.0]).astype(np.complex128) for axis in range(3)}
    for col in range(2):  # both partners are even under every mirror, with or without the orbital factor
        assert np.allclose(_mirror_signature(block[:, col], gap_shape, mirrors), 1.0, atol=1e-12)

    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5]), block, gap_shape, orbital_mirrors=mirrors)

    assert np.allclose(np.abs(fixed.conj().T @ block), np.eye(2), atol=1e-10)


def test_symmetrize_degenerate_gaps_keeps_a_doublet_that_is_not_a_clean_mirror_eigenspace():
    """A doublet whose M_y projection is not +/-1 is kept as is instead of rotated into a meaningless basis."""
    gap_shape = (6, 6, 1, 2, 2, 4)
    rng = np.random.default_rng(19)
    n = int(np.prod(gap_shape))
    block, _ = np.linalg.qr(rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2)))
    mirrors = {axis: np.diag([1.0, -1.0]).astype(np.complex128) for axis in range(3)}

    fixed = symmetrize_degenerate_gaps(np.array([0.5, 0.5]), block, gap_shape, orbital_mirrors=mirrors)

    assert np.allclose(np.abs(fixed.conj().T @ block), np.eye(2), atol=1e-10)


def test_symmetrize_degenerate_gaps_ignores_orbital_mirrors_of_the_wrong_orbital_size():
    """Mirrors whose orbital dimension does not match the gap are dropped, reducing to momentum-only reflections."""
    px, py, gap_shape = _make_p_wave_doublet()
    gaps = np.stack([px, py], axis=1) @ np.array([[1.0, 0.5 - 0.2j], [0.3 + 0.1j, 1.0]])
    lambdas = np.array([0.7, 0.7])

    with_bad = symmetrize_degenerate_gaps(lambdas, gaps, gap_shape, orbital_mirrors=_t2g_orbital_mirrors())

    assert np.allclose(with_bad, symmetrize_degenerate_gaps(lambdas, gaps, gap_shape), atol=1e-12)


def test_gap_orbital_mirrors_returns_nothing_for_a_single_orbital_gap():
    """A 1x1 orbital unitary is a phase and cancels in U Delta U^dag, so there is nothing to apply."""
    assert _gap_orbital_mirrors(1) == {}


def test_gap_orbital_mirrors_derives_the_t2g_signs_from_the_configured_hamiltonian():
    """The mirrors are read off the configured H(k) - nothing about the orbital set is hard-coded in the solver."""
    from dgamore.hamiltonian import Hamiltonian

    nk = 8
    config.lattice.nk = (nk, nk, nk)
    config.lattice.k_grid = bz.KGrid(config.lattice.nk, symmetries=[])
    config.lattice.hamiltonian = Hamiltonian().set_ek(_cubic_t2g_hamiltonian(nk=nk))

    mirrors = _gap_orbital_mirrors(3)

    assert sorted(mirrors) == [0, 1, 2]
    for axis, signs in _T2G_MIRROR_SIGNS.items():
        expected = np.diag(signs)
        assert np.allclose(mirrors[axis], expected, atol=1e-8) or np.allclose(mirrors[axis], -expected, atol=1e-8)


def test_gap_orbital_mirrors_returns_nothing_when_the_hamiltonian_is_unavailable():
    """A failing symmetry probe must degrade to momentum-only mirrors, never abort the solve."""
    from types import SimpleNamespace

    config.logger = MagicMock()
    config.lattice.hamiltonian = SimpleNamespace(get_ek=MagicMock(side_effect=RuntimeError("no hopping set")))

    assert _gap_orbital_mirrors(3) == {}


def test_solve_pairing_sectors_passes_the_orbital_mirrors_into_the_symmetrization(monkeypatch):
    """The solver hands the discovered mirrors to symmetrize_degenerate_gaps instead of leaving it momentum-only."""
    import dgamore.eliashberg_solver as es

    gap_shape = (2, 2, 1, 3, 3, 2)
    mirrors = _t2g_orbital_mirrors()
    seen = {}

    def fake_symmetrize(lambdas, gaps, shape, tol=1e-4, orbital_mirrors=None):
        seen["orbital_mirrors"] = orbital_mirrors
        return gaps

    n = int(np.prod(gap_shape))
    monkeypatch.setattr(es, "_gap_orbital_mirrors", lambda n_bands: mirrors)
    monkeypatch.setattr(es, "symmetrize_degenerate_gaps", fake_symmetrize)
    monkeypatch.setattr(es.sp.sparse.linalg, "eigsh", lambda *a, **kw: (np.array([0.3]), np.ones((n, 1))))
    config.eliashberg.n_eig = 1
    config.eliashberg.symmetrize_degenerate_gaps = True
    config.eliashberg.resolve_frequency_parity = None
    config.logger = MagicMock()

    es._solve_pairing_sectors(
        mv=lambda gap: gap,
        gap_shape=gap_shape,
        sign=1,
        channel=SpinChannel.SING,
        nq=gap_shape[:3],
        executor=None,
        ranks=(0, 0),
        base_seed=np.ones(n),
    )

    assert seen["orbital_mirrors"] is mirrors

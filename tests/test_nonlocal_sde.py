# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import dgamore.brillouin_zone as bz
import dgamore.config as config
import dgamore.mpi_utils as mpi_utils
import dgamore.nonlocal_sde as nonlocal_sde
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction
from dgamore.hamiltonian import Hamiltonian
from dgamore.interaction import Interaction, LocalInteraction
from dgamore.local_four_point import LocalFourPoint
from dgamore.local_sde import get_local_hartree_fock
from dgamore.n_point_base import SpinChannel
from tests.conftest import FAKE_MPI, run_parallel
from dgamore.nonlocal_sde import (
    _build_giwk_full,
    _cut_and_reshare_giwk,
    _free_shared_window,
    _init_mu_history,
    _release_shared_giwk,
    get_hartree_fock,
    perform_ornstein_zernike_fit,
)
from dgamore.self_energy import SelfEnergy
from tests.conftest import FAKE_MPI, create_comm_mock, run_parallel

LOCAL_SDE_DATA = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/local_sde"


def test_init_mu_history_fresh_uses_current_mu():
    """A fresh run (starting_iter == 0) seeds the history with the current (DMFT) chemical potential and leaves it."""
    config.sys.mu = 0.7
    config.self_consistency.previous_sc_path = ""

    mu_history = _init_mu_history(0)

    assert mu_history == [0.7]
    assert config.sys.mu == 0.7  # unchanged on a fresh run


def test_init_mu_history_from_previous_syncs_global_mu(monkeypatch, tmp_path):
    """Resuming seeds the history with, and syncs the stale global config.sys.mu to, the previous last converged mu."""
    config.sys.mu = 0.3
    config.self_consistency.previous_sc_path = str(tmp_path)
    previous_mu = 1.5
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: np.array([0.9, 1.2, previous_mu]))

    mu_history = _init_mu_history(3)

    assert mu_history == [previous_mu]
    assert config.sys.mu == previous_mu


def test_nonlocal_hartree_fock_matches_local_reference():
    """The non-local Hartree-Fock reduces to the local reference for V=0 and k-independent occupation."""
    nb = 2
    nk = (2, 2, 1)
    nq_tot = int(np.prod(nk))

    config.lattice.nk = nk
    config.sys.n_bands = nb

    occ = np.load(f"{LOCAL_SDE_DATA}/occ.npy", allow_pickle=False)
    config.sys.occ = occ
    config.sys.occ_k = np.broadcast_to(occ, nk + (nb, nb)).copy()

    u_loc = Hamiltonian().read_umatrix(f"{LOCAL_SDE_DATA}/u_matrix.dat").get_local_u()
    v_nonloc = Interaction(
        np.zeros((nq_tot, nb, nb, nb, nb), dtype=u_loc.mat.dtype),
        SpinChannel.NONE,
        nk,
        has_compressed_q_dimension=True,
    )
    hartree, fock = get_hartree_fock(u_loc, v_nonloc)
    hf_nonlocal = (hartree + fock)[..., 0]

    sigma_hf_ref = np.load(f"{LOCAL_SDE_DATA}/sigma_HF.npy", allow_pickle=False)
    assert hf_nonlocal.shape == (nq_tot, nb, nb)
    assert np.allclose(hf_nonlocal, sigma_hf_ref[None, ...])
    # the same reference is the local Hartree-Fock, so the two SDE paths agree
    assert np.allclose(hf_nonlocal, get_local_hartree_fock(u_loc, occ)[None, ...])


def test_nonlocal_fock_fft_matches_explicit_momentum_sum():
    """The FFT Fock term matches the explicit q-loop n^{k-q} sum for multi-band non-zero V and k-dependent occ."""
    nb, nk = 2, (3, 4, 2)
    nk_tot = int(np.prod(nk))
    config.lattice.nk = nk
    config.sys.n_bands = nb

    rng = np.random.default_rng(3)
    u_loc = LocalInteraction(rng.standard_normal((nb, nb, nb, nb)).astype(np.complex64), SpinChannel.NONE)
    v_mat = (rng.standard_normal((nk_tot, nb, nb, nb, nb)) + 1j * rng.standard_normal((nk_tot, nb, nb, nb, nb))).astype(
        np.complex64
    )
    v_nonloc = Interaction(v_mat, SpinChannel.NONE, nk, has_compressed_q_dimension=True)
    occ_k = (rng.standard_normal((*nk, nb, nb)) + 1j * rng.standard_normal((*nk, nb, nb))).astype(np.complex64)
    config.sys.occ_k = occ_k
    config.sys.occ = occ_k.mean(axis=(0, 1, 2))

    uq = (u_loc + v_nonloc).permute_orbitals("abcd->adcb").mat.reshape(nk_tot, nb, nb, nb, nb)  # [q,a,d,c,b]
    q_list = np.array([np.unravel_index(i, nk) for i in range(nk_tot)])
    fock_ref = np.zeros((nk_tot, nb, nb), dtype=np.complex64)
    for d in range(nb):
        for c in range(nb):
            occ_qk = np.array([np.roll(occ_k[..., d, c], tuple(q), axis=(0, 1, 2)) for q in q_list]).reshape(
                nk_tot, nk_tot
            )
            fock_ref += np.einsum("qab,qk->kab", uq[:, :, d, c, :], occ_qk, optimize=True)
    fock_ref *= -1.0 / nk_tot

    _, fock = get_hartree_fock(u_loc, v_nonloc)
    assert np.allclose(fock[..., 0], fock_ref, atol=1e-4)


def _constant_chi(mat: np.ndarray):
    """Builds a physical-susceptibility stand-in whose copy and BZ/frequency reductions are identities."""
    chi = MagicMock(mat=mat)
    for name in ("copy", "map_to_full_bz", "to_half_niw_range", "take_first_wn"):
        getattr(chi, name).return_value = chi
    return chi


def test_ornstein_zernike_fit_aggregates_nonconverged_warnings(monkeypatch):
    """All non-converging OZ fits collapse into a single aggregated warning instead of one log per orbital."""
    config.sys.n_bands = 2
    logger = MagicMock()
    monkeypatch.setattr(config, "logger", logger, raising=False)
    monkeypatch.setattr(nonlocal_sde.opt, "curve_fit", MagicMock(side_effect=RuntimeError("forced non-convergence")))

    perform_ornstein_zernike_fit(_constant_chi(np.ones((2, 2, 1, 2, 2, 2, 2), dtype=np.complex64)))

    logger.warning.assert_called_once()
    msg = logger.warning.call_args.args[0]
    assert "16 orbital combination(s)" in msg
    assert "(1, 1, 1, 1)" in msg and "(2, 2, 2, 2)" in msg  # 1-based orbital labels, not 0-based


def test_ornstein_zernike_fit_logs_no_warning_when_all_converge(monkeypatch):
    """A fully converging set of OZ fits emits no warning at all (the aggregation guard stays silent)."""
    config.sys.n_bands = 2
    logger = MagicMock()
    monkeypatch.setattr(config, "logger", logger, raising=False)
    monkeypatch.setattr(nonlocal_sde.opt, "curve_fit", MagicMock(return_value=(np.array([1.0, 2.0]), None)))

    perform_ornstein_zernike_fit(_constant_chi(np.ones((2, 2, 1, 2, 2, 2, 2), dtype=np.complex64)))

    logger.warning.assert_not_called()


def _tiny_sigma_and_ek(nb=1, niv=4):
    """Builds a minimal single-k self-energy and dispersion for the Dyson build."""
    sigma = SelfEnergy(np.zeros((1, 1, 1, nb, nb, 2 * niv), dtype=np.complex64), calc_smom=False, beta=10.0)
    ek = np.zeros((1, 1, 1, nb, nb), dtype=np.complex64)
    return sigma, ek


def test_build_giwk_full_shared_single_rank_matches_direct_dyson():
    """With node-sharing on but a single-rank node, the giwk is bit-parity with the direct Dyson build (no window)."""
    sigma, ek = _tiny_sigma_and_ek()
    comm = create_comm_mock()

    giwk, win, node_comm = _build_giwk_full(comm, sigma, 0.3, ek, 10.0)

    assert win is None
    assert np.array_equal(giwk.mat, GreensFunction.get_g_full(sigma, 0.3, ek, 10.0).mat)
    _release_shared_giwk(win, node_comm)  # must not raise on the single-rank / mock path


def test_release_shared_giwk_without_sharing_is_noop():
    """Releasing a non-shared giwk (both handles None) is a safe no-op."""
    _release_shared_giwk(None, None)


def test_release_shared_giwk_frees_window_and_communicator():
    """With a window allocated, _release barriers (so no rank still reads), frees the window, then the node comm."""
    win, node_comm = MagicMock(), MagicMock()
    _release_shared_giwk(win, node_comm)
    node_comm.Barrier.assert_called_once()
    win.Free.assert_called_once()
    node_comm.Free.assert_called_once()


def test_free_shared_window_frees_window_but_keeps_communicator():
    """_free_shared_window barriers and frees the window but leaves the node communicator alive (reused for the cut)."""
    win, node_comm = MagicMock(), MagicMock()
    _free_shared_window(win, node_comm)
    node_comm.Barrier.assert_called_once()
    win.Free.assert_called_once()
    node_comm.Free.assert_not_called()


def test_cut_and_reshare_giwk_without_sharing_is_a_plain_cut():
    """Without a node communicator, _cut_and_reshare_giwk is a plain per-rank cut and allocates no window."""
    sigma, ek = _tiny_sigma_and_ek(niv=8)
    giwk = GreensFunction.get_g_full(sigma, 0.3, ek, 10.0)
    cut, win = _cut_and_reshare_giwk(giwk, None, None, 4)
    assert win is None
    assert cut.niv == 4
    assert np.array_equal(cut.mat, GreensFunction.get_g_full(sigma, 0.3, ek, 10.0).cut_niv(4).mat)


def test_cut_and_reshare_giwk_shared_single_rank_matches_plain_cut():
    """With a single-rank node communicator the cut giwk matches the plain cut and needs no window."""
    sigma, ek = _tiny_sigma_and_ek(niv=8)
    giwk = GreensFunction.get_g_full(sigma, 0.3, ek, 10.0)
    cut, win = _cut_and_reshare_giwk(giwk, None, create_comm_mock(), 4)
    assert win is None
    assert cut.niv == 4
    assert np.array_equal(cut.mat, GreensFunction.get_g_full(sigma, 0.3, ek, 10.0).cut_niv(4).mat)


def _bse_assembly_inputs(rng, o=2, nqi=3, nw=3, niv=2, beta=12.5):
    """Builds (gamma [full niw], gchi0_q_inv [half niw, 1 vn], u_loc, v_nonloc) for the BSE-matrix assembly tests."""
    config.sys.beta = beta
    gamma_shape = (o, o, o, o, 2 * nw - 1, 2 * niv, 2 * niv)
    gamma_mat = rng.standard_normal(gamma_shape) + 1j * rng.standard_normal(gamma_shape)
    gamma = LocalFourPoint(gamma_mat, SpinChannel.DENS, 1, 2, True, True)
    chi0_shape = (nqi, o, o, o, o, nw, 2 * niv)
    chi0_mat = rng.standard_normal(chi0_shape) + 1j * rng.standard_normal(chi0_shape)
    gchi0_q_inv = FourPoint(chi0_mat, SpinChannel.NONE, (nqi, 1, 1), 1, 1, False, True, True)
    u_loc = LocalInteraction(rng.standard_normal((o,) * 4), SpinChannel.NONE)
    v_nonloc = Interaction(rng.standard_normal((nqi,) + (o,) * 4), SpinChannel.NONE, (nqi, 1, 1), True)
    return gamma, gchi0_q_inv, u_loc, v_nonloc


def _bse_assembly_reference(gamma, gchi0_q_inv, u_loc, v_nonloc):
    """Evaluates the pre-fusion two-block expression (chi0^-1 + Gamma/beta^2) - (U + V^q)/beta^2 via the object API."""
    beta = config.sys.beta
    return (gchi0_q_inv.copy() + 1.0 / beta**2 * gamma.copy()) - 1.0 / beta**2 * (
        v_nonloc.as_channel(gamma.channel) + u_loc.as_channel(gamma.channel)
    )


def test_create_inverse_auxiliary_chi_r_q_matches_two_block_expression():
    """The fused single-block BSE assembly is bit-equal to the former add/extend/subtract chain, inputs untouched."""
    rng = np.random.default_rng(21)
    gamma, gchi0_q_inv, u_loc, v_nonloc = _bse_assembly_inputs(rng)
    gamma_before, chi0_before = gamma.mat.copy(), gchi0_q_inv.mat.copy()
    ref = _bse_assembly_reference(gamma, gchi0_q_inv, u_loc, v_nonloc)
    u_r = v_nonloc.as_channel(gamma.channel) + u_loc.as_channel(gamma.channel)
    fused = nonlocal_sde.create_inverse_auxiliary_chi_r_q(gamma, gchi0_q_inv, u_r)
    assert np.array_equal(fused.mat, ref.mat)
    assert fused.channel == SpinChannel.DENS
    assert not fused.full_niw_range and fused.num_vn_dimensions == 2
    assert np.array_equal(gamma.mat, gamma_before) and gamma.full_niw_range
    assert np.array_equal(gchi0_q_inv.mat, chi0_before) and gchi0_q_inv.num_vn_dimensions == 1


def test_create_inverse_auxiliary_chi_r_q_accepts_half_niw_gamma():
    """A gamma already in the half bosonic range assembles identically to its full-range twin."""
    rng = np.random.default_rng(22)
    gamma, gchi0_q_inv, u_loc, v_nonloc = _bse_assembly_inputs(rng)
    u_r = v_nonloc.as_channel(gamma.channel) + u_loc.as_channel(gamma.channel)
    full = nonlocal_sde.create_inverse_auxiliary_chi_r_q(gamma, gchi0_q_inv, u_r)
    half = nonlocal_sde.create_inverse_auxiliary_chi_r_q(gamma.copy().to_half_niw_range(), gchi0_q_inv, u_r)
    assert np.array_equal(full.mat, half.mat)


def test_create_auxiliary_chi_r_q_matches_explicit_reference():
    """The full-inversion variant reproduces the explicit two-block expression inverted in compound space."""
    rng = np.random.default_rng(24)
    gamma, gchi0_q_inv, u_loc, v_nonloc = _bse_assembly_inputs(rng)
    ref = _bse_assembly_reference(gamma, gchi0_q_inv, u_loc, v_nonloc).invert(False)
    out = nonlocal_sde.create_auxiliary_chi_r_q(gamma, gchi0_q_inv, u_loc, v_nonloc)
    assert np.allclose(out.mat, ref.mat, atol=1e-10)


def test_calculate_kernel_r_q_matches_rewired_reference():
    """The kernel equals [gamma(1 - U chi) - 1] right-multiplied by U_r, in the sigma-contraction layout.

    The reference realizes Sigma_{12} ~ sum_{ab} [K U_r]_{1ab2} G_{ab} == sum_{ab} K'_{a12b} G_{ab}, i.e. the
    interaction attaches to the chi*-side pair of the three-leg vertex (Worm's placement) and the stored layout
    is the "abcd->badc" permutation of [K U_r].
    """
    rng = np.random.default_rng(25)
    o, nqi, nw, niv, beta = 2, 3, 3, 2, 12.5
    config.sys.beta = beta
    vrg_shape = (nqi, o, o, o, o, nw, 2 * niv)
    chi_shape = (nqi, o, o, o, o, nw)
    for channel in (SpinChannel.MAGN, SpinChannel.DENS):
        vrg = FourPoint(
            rng.standard_normal(vrg_shape) + 1j * rng.standard_normal(vrg_shape),
            channel,
            (nqi, 1, 1),
            1,
            1,
            False,
            True,
            True,
        )
        chi = FourPoint(
            rng.standard_normal(chi_shape) + 1j * rng.standard_normal(chi_shape),
            channel,
            (nqi, 1, 1),
            1,
            0,
            False,
            True,
            True,
        )
        u_loc = nonlocal_sde.LocalInteraction(rng.standard_normal((o,) * 4), SpinChannel.NONE)
        v_nonloc = Interaction(rng.standard_normal((nqi,) + (o,) * 4), SpinChannel.NONE, (nqi, 1, 1), True)
        u_r = v_nonloc.as_channel(channel) + u_loc.as_channel(channel)
        inner = vrg.copy() - vrg.copy() @ u_r @ chi.copy()
        inner = inner - FourPoint.identity_like(inner)
        ref = inner @ u_r
        # "abcd->badc" on the orbital axes of [q, o1, o2, o3, o4, w, v]
        ref_mat = ref.mat.transpose(0, 2, 1, 4, 3, 5, 6)
        out = nonlocal_sde.calculate_kernel_r_q(vrg, chi, v_nonloc, u_loc)
        assert np.allclose(out.mat, ref_mat, atol=1e-10)


def test_vrg_right_is_first_frequency_summed_three_leg_vertex():
    """The right three-leg vertex equals its first-frequency-summed definition via the dcba permute of chi*."""
    o, nqi, nw, n2, beta = 2, 3, 3, 4, 12.5
    config.sys.beta = beta
    rng = np.random.default_rng(11)
    shape = (nqi, o, o, o, o, nw, n2, n2)
    chi_star = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    chi_star = 0.5 * (chi_star + np.transpose(chi_star, (0, 4, 3, 2, 1, 5, 7, 6)))
    chi0_inv = rng.standard_normal(shape[:-1]) + 1j * rng.standard_normal(shape[:-1])

    sum_last = FourPoint(chi_star.sum(axis=-1) / beta, SpinChannel.DENS, (nqi, 1, 1), 1, 1, False, True, True)
    chi0_inv_fp = FourPoint(chi0_inv.copy(), SpinChannel.NONE, (nqi, 1, 1), 1, 1, False, True, True)

    vrg_left = nonlocal_sde.create_vrg_r_q(sum_last.copy(), chi0_inv_fp)
    ref_left = np.einsum("qabefwv,qfecdwv->qabcdwv", chi0_inv, chi_star.sum(axis=-1), optimize=True)
    assert np.allclose(vrg_left.mat, ref_left, atol=1e-10)

    vrg_right = nonlocal_sde.create_vrg_r_q_right(sum_last, chi0_inv_fp)
    ref_right = np.einsum("qabefwv,qfecdwv->qabcdwv", chi_star.sum(axis=-2), chi0_inv, optimize=True)
    assert np.allclose(vrg_right.mat, ref_right, atol=1e-10)


def test_unused_qloop_sigma_variants_agree():
    """The two unused q-loop self-energy variants (plain and buffered) agree on synthetic data, locking them."""
    nk, o, niw, niv = (4, 4, 1), 2, 3, 4
    config.lattice.nk = nk
    config.lattice.k_grid = bz.KGrid(nk, symmetries=[])
    config.box.niw_core = niw
    config.box.niv_core = niv
    config.sys.n_bands = o
    config.sys.beta = 12.5
    config.logger = MagicMock()

    rng = np.random.default_rng(7)
    niv_g = niv + niw + 2
    g_shape = (*nk, o, o, 2 * niv_g)
    k_shape = (int(np.prod(nk)), o, o, o, o, niw + 1, 2 * niv)
    g_mat = (rng.standard_normal(g_shape) + 1j * rng.standard_normal(g_shape)).astype(np.complex64)
    kernel_mat = (rng.standard_normal(k_shape) + 1j * rng.standard_normal(k_shape)).astype(np.complex64)
    giwk = GreensFunction(g_mat, calc_filling=False, nk=nk, beta=config.sys.beta)
    q_list = config.lattice.k_grid.get_q_list()

    def make_kernel():
        return FourPoint(
            kernel_mat.copy(), SpinChannel.NONE, nk, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
        )

    sigma_ref = nonlocal_sde.calculate_sigma_from_kernel(make_kernel(), giwk, q_list)
    sigma_loop = nonlocal_sde.calculate_sigma_from_kernel_loop(make_kernel(), giwk, q_list)

    assert np.allclose(sigma_loop.mat, sigma_ref.mat, atol=1e-6)


def _chi_from_compound(comp: np.ndarray, o: int):
    """Builds a 0-vn FourPoint [q, o, o, o, o, w] from a compound array [q, w, o^2, o^2] (rows (12), cols (43))."""
    nq, nw = comp.shape[:2]
    mat = np.transpose(comp.reshape(nq, nw, o, o, o, o), (0, 2, 3, 5, 4, 1))
    return FourPoint(mat.copy(), SpinChannel.DENS, (nq, 1, 1), 1, 0, False, True, True)


def _compound_of(chi, o: int) -> np.ndarray:
    """Extracts the compound array [q, w, o^2, o^2] from a 0-vn FourPoint in full-index layout."""
    return chi.mat.transpose(0, 5, 1, 2, 4, 3).reshape(chi.mat.shape[0], chi.mat.shape[-1], o * o, o * o)


def test_restrict_chi_phys_floors_negative_inverse_eigenvalues():
    """A chi block with a negative eigenvalue comes back with its inverse eigenvalue floored, positive pairs kept."""
    rng = np.random.default_rng(4)
    o, nq, nw, floor = 2, 2, 3, 1e-4
    q_mat = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))[0]
    eigs_in = np.array([-5.0, 0.5, 1.0, 2.0])
    comp = np.tile((q_mat * eigs_in) @ q_mat.conj().T, (nq, nw, 1, 1))
    chi = _chi_from_compound(comp, o)
    out, n_floored = nonlocal_sde.restrict_chi_phys_to_positive_eigenvalues(chi, floor=floor)
    assert n_floored == nq * nw
    out_comp = _compound_of(out, o).astype(np.complex128)
    ev = np.sort(np.linalg.eigvalsh(0.5 * (out_comp + np.conj(np.transpose(out_comp, (0, 1, 3, 2))))), axis=-1)
    expected = np.sort(np.array([1.0 / floor, 0.5, 1.0, 2.0]))
    assert np.allclose(ev / expected[None, None, :], 1.0, atol=1e-3)
    proj_pos = (q_mat[:, 1:] * eigs_in[1:]) @ q_mat[:, 1:].conj().T
    proj_out = out_comp[0, 0] - (out_comp[0, 0] @ q_mat[:, :1]) @ q_mat[:, :1].conj().T
    assert np.allclose(proj_out @ q_mat[:, 1:], proj_pos @ q_mat[:, 1:], atol=1e-3)


def test_restrict_chi_phys_leaves_positive_definite_input_unchanged():
    """A positive-definite chi with negative off-diagonal entries passes through unchanged."""
    rng = np.random.default_rng(5)
    o, nq, nw = 2, 2, 3
    a = rng.standard_normal((nq, nw, 4, 4)) + 1j * rng.standard_normal((nq, nw, 4, 4))
    comp = a @ np.conj(np.transpose(a, (0, 1, 3, 2))) + 0.1 * np.eye(4)
    assert (comp.real < 0).any()
    chi = _chi_from_compound(comp, o)
    out, n_floored = nonlocal_sde.restrict_chi_phys_to_positive_eigenvalues(chi, floor=1e-4)
    assert n_floored == 0
    assert np.allclose(_compound_of(out, o), comp, atol=1e-3)


def test_restrict_chi_phys_matches_scalar_clamp_for_single_band():
    """For a single band the eigenvalue floor reproduces the elementwise clamp: negative chi -> 1/floor, else kept."""
    floor = 1e-4
    mat = np.array([-0.5, 0.3], dtype=complex).reshape(2, 1, 1, 1, 1, 1)
    chi = FourPoint(mat.copy(), SpinChannel.DENS, (2, 1, 1), 1, 0, False, True, True)
    out, n_floored = nonlocal_sde.restrict_chi_phys_to_positive_eigenvalues(chi, floor=floor)
    assert n_floored == 1
    assert np.allclose(out.mat.ravel() / np.array([1.0 / floor, 0.3]), 1.0, atol=1e-3)


def test_min_static_compound_eigenvalue_reports_definiteness():
    """A positive-definite chi gives a positive minimum static eigenvalue, and a planted negative one is reported."""
    rng = np.random.default_rng(6)
    o, nq, nw = 2, 3, 4
    a = rng.standard_normal((nq, nw, 4, 4)) + 1j * rng.standard_normal((nq, nw, 4, 4))
    comp = a @ np.conj(np.transpose(a, (0, 1, 3, 2))) + 0.5 * np.eye(4)
    assert nonlocal_sde.min_static_compound_eigenvalue(_chi_from_compound(comp, o)) > 0.0
    q_vec = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))[0]
    comp[1, 0] = (q_vec * np.array([-3.0, 0.5, 1.0, 2.0])) @ q_vec.conj().T
    assert np.allclose(nonlocal_sde.min_static_compound_eigenvalue(_chi_from_compound(comp, o)), -3.0, atol=1e-3)


def test_effective_epsilon_is_relaxed_while_restriction_is_active():
    """The threshold is 10x epsilon while the susceptibility restriction is active, plain after release."""
    config.self_consistency.epsilon = 1e-5
    config.stabilization.use_chi_phys_restriction = True
    assert np.allclose(nonlocal_sde._effective_epsilon(), 1e-4, atol=1e-18)
    config.stabilization.use_chi_phys_restriction = False
    assert np.allclose(nonlocal_sde._effective_epsilon(), 1e-5, atol=1e-18)


def test_effective_epsilon_is_relaxed_while_lambda_correction_is_active():
    """The threshold is 10x epsilon while the per-iteration lambda correction is active, never by the one-shot."""
    config.self_consistency.epsilon = 1e-5
    config.stabilization.use_chi_phys_restriction = False
    config.stabilization.use_lambda_correction = True
    assert np.allclose(nonlocal_sde._effective_epsilon(), 1e-4, atol=1e-18)
    config.stabilization.use_lambda_correction = False
    config.lambda_correction.perform_lambda_correction = True
    assert np.allclose(nonlocal_sde._effective_epsilon(), 1e-5, atol=1e-18)


def test_build_rspace_giwk_pencil_node_sharing_matches_private_build():
    """_build_rspace_giwk_pencil routes the R-space G through the node-shared window builder, matching private."""
    nk, o, niw, niv = (4, 4, 1), 2, 3, 4
    config.lattice.nk = nk
    config.lattice.k_grid = bz.KGrid(nk, symmetries=[])
    config.box.niw_core = niw
    config.box.niv_core = niv
    config.sys.n_bands = o
    config.sys.beta = 12.5
    config.logger = MagicMock()

    rng = np.random.default_rng(9)
    g_shape = (*nk, o, o, 2 * (niv + niw + 2))
    g_mat = (rng.standard_normal(g_shape) + 1j * rng.standard_normal(g_shape)).astype(np.complex64)
    giwk = GreensFunction(g_mat, calc_filling=False, nk=nk, beta=config.sys.beta)
    mpi_dist = MagicMock()
    mpi_dist.comm = create_comm_mock()

    node_comm = MagicMock(**{"Get_rank.return_value": 0, "Get_size.return_value": 1})
    g_r_plain = nonlocal_sde._build_rspace_giwk_pencil(giwk, mpi_dist)
    g_r_shared = nonlocal_sde._build_rspace_giwk_pencil(giwk, mpi_dist, node_comm=node_comm)
    assert np.array_equal(g_r_shared, g_r_plain)


def test_load_node_shared_local_vertex_private_path_applies_transform(monkeypatch):
    """Without a node communicator the helper loads privately, applies the transform and returns win=None."""
    rng = np.random.default_rng(81)
    mat = (rng.standard_normal((2, 2, 2, 2, 3, 4, 4)) + 1j * rng.standard_normal((2, 2, 2, 2, 3, 4, 4))).astype(
        np.complex64
    )
    monkeypatch.setattr(np, "load", lambda *a, **k: mat.copy())
    ref = LocalFourPoint(mat.copy(), SpinChannel.NONE, 1, 2, False, True).permute_orbitals("abcd->cbad").scale(2.0)
    out, win = nonlocal_sde._load_node_shared_local_vertex(
        None,
        "unused.npy",
        SpinChannel.NONE,
        transform=lambda o: o.permute_orbitals("abcd->cbad", copy=False).scale(2.0),
    )
    assert win is None
    assert np.array_equal(out.mat, ref.mat)
    assert out.mat.flags["C_CONTIGUOUS"]


def test_load_node_shared_local_vertex_loads_once_per_node(monkeypatch):
    """With a node communicator the file is read once per node (the root) and every rank maps the same values."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    rng = np.random.default_rng(82)
    mat = (rng.standard_normal((2, 2, 2, 2, 3, 4, 4)) + 1j * rng.standard_normal((2, 2, 2, 2, 3, 4, 4))).astype(
        np.complex64
    )
    load_calls = []

    def fake_load(*a, **k):
        load_calls.append(1)
        return mat.copy()

    monkeypatch.setattr(np, "load", fake_load)

    def fn(comm, rank):
        node_comm = comm.Split_type(FAKE_MPI.COMM_TYPE_SHARED)
        out, win = nonlocal_sde._load_node_shared_local_vertex(node_comm, "unused.npy", SpinChannel.DENS)
        res = out.mat.copy()
        out.mat = None
        nonlocal_sde._free_shared_window(win, node_comm)
        return res

    _, res = run_parallel(2, fn, hostnames=["n0", "n0"])
    assert len(load_calls) == 1  # one read per node, not per rank
    assert np.array_equal(res[0], mat) and np.array_equal(res[1], mat)


@pytest.fixture
def stab_logger(monkeypatch):
    """Installs a MagicMock logger as config.logger for the duration of a test (stabilization glue tests)."""
    logger = MagicMock()
    monkeypatch.setattr(config, "logger", logger, raising=False)
    return logger


def test_mixing_history_cap_uses_most_recent_reset_event():
    """The history cap counts iterations since the later of the restriction release and the annealing-mass change."""
    assert nonlocal_sde._mixing_history_cap(10, None, None) is None
    assert nonlocal_sde._mixing_history_cap(10, 7, None) == 2
    assert nonlocal_sde._mixing_history_cap(10, None, 8) == 1
    assert nonlocal_sde._mixing_history_cap(10, 7, 9) == 0
    assert nonlocal_sde._mixing_history_cap(9, 4, 9) == 0


def _make_resid_sigma(mat):
    """Wraps a compressed-layout array into a SelfEnergy for the residual helper tests."""
    return SelfEnergy(mat, (mat.shape[0], 1, 1), has_compressed_q_dimension=True, calc_smom=False, beta=10.0)


def test_relative_sigma_residual_zero_for_identical_input():
    """Identical self-energies give a vanishing relative residual."""
    config.box.niv_core = 3
    rng = np.random.default_rng(0)
    mat = (rng.standard_normal((4, 1, 1, 10)) + 1j * rng.standard_normal((4, 1, 1, 10))).astype(np.complex64)
    assert nonlocal_sde._relative_sigma_residual(_make_resid_sigma(mat), _make_resid_sigma(mat.copy())) == 0.0


def test_relative_sigma_residual_matches_explicit_core_window_formula():
    """The helper equals the explicit [Re; Im]-stacked L2 quotient over the positive core frequencies."""
    config.box.niv_core = 3
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((4, 1, 1, 10)) + 1j * rng.standard_normal((4, 1, 1, 10))).astype(np.complex64)
    b = (rng.standard_normal((4, 1, 1, 10)) + 1j * rng.standard_normal((4, 1, 1, 10))).astype(np.complex64)
    new_c, old_c = a[..., 5:8], b[..., 5:8]
    diff = (new_c - old_c).ravel()
    ref = np.linalg.norm(np.concatenate([diff.real, diff.imag])) / np.linalg.norm(
        np.concatenate([old_c.real.ravel(), old_c.imag.ravel()])
    )
    out = nonlocal_sde._relative_sigma_residual(_make_resid_sigma(a), _make_resid_sigma(b))
    assert np.allclose(out, ref, atol=1e-12)


def test_relative_sigma_residual_broadcasts_local_old_against_full_bz():
    """A local (nq=1) previous iterate is broadcast against a full-BZ proposal without shape errors."""
    config.box.niv_core = 3
    rng = np.random.default_rng(2)
    new = (rng.standard_normal((6, 1, 1, 10)) + 1j * rng.standard_normal((6, 1, 1, 10))).astype(np.complex64)
    old_local = new[:1].copy()
    out = nonlocal_sde._relative_sigma_residual(_make_resid_sigma(new), _make_resid_sigma(old_local))
    assert np.isfinite(out) and out > 0.0
    assert nonlocal_sde._relative_sigma_residual(_make_resid_sigma(new[:1].copy()), _make_resid_sigma(old_local)) == 0.0


def test_relative_sigma_residual_scales_with_mixing_step():
    """A linear-mixing step alpha*(S(x)-x) yields exactly alpha times the raw proposal residual."""
    config.box.niv_core = 3
    rng = np.random.default_rng(3)
    old = (rng.standard_normal((4, 1, 1, 10)) + 1j * rng.standard_normal((4, 1, 1, 10))).astype(np.complex128)
    prop = (rng.standard_normal((4, 1, 1, 10)) + 1j * rng.standard_normal((4, 1, 1, 10))).astype(np.complex128)
    alpha = 0.2
    mixed = alpha * prop + (1 - alpha) * old
    raw = nonlocal_sde._relative_sigma_residual(_make_resid_sigma(prop), _make_resid_sigma(old))
    step = nonlocal_sde._relative_sigma_residual(_make_resid_sigma(mixed), _make_resid_sigma(old))
    assert np.allclose(step, alpha * raw, atol=1e-12)


def _single_rank_dist():
    """Builds a minimal single-rank distributor stand-in for the annealing shift (only .comm.size is read)."""
    return SimpleNamespace(comm=SimpleNamespace(size=1))


def _seeded_annealer(mass=0.0, gaps=None, initialized=True):
    """Builds a LambdaAnnealer with a prescribed shared mass and per-channel gaps for the schedule tests."""
    annealer = nonlocal_sde.LambdaAnnealer()
    annealer._initialized = initialized
    annealer._mass = mass
    if gaps:
        annealer._gaps.update(gaps)
    return annealer


def test_annealer_apply_shifts_inverse_by_shared_mass(stab_logger):
    """With a nonzero shared mass the result equals 1/(1/chi + lambda) exactly and the static gap is stored."""
    rng = np.random.default_rng(11)
    o, nq, nw, lam = 2, 2, 3, 0.7
    q_mat = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))[0]
    comp_inv = np.tile((q_mat * np.array([-0.5, 0.5, 1.0, 2.0])) @ q_mat.conj().T, (nq, nw, 1, 1))
    chi = _chi_from_compound(np.linalg.inv(comp_inv), o)
    annealer = _seeded_annealer(mass=lam)
    out = annealer.apply(chi, _single_rank_dist())
    assert np.allclose(_compound_of(out, o), np.linalg.inv(comp_inv + lam * np.eye(4)), atol=1e-4)
    assert np.allclose(annealer._gaps["dens"], -0.5, atol=1e-5)


def test_annealer_apply_measures_without_shift_at_zero_mass():
    """At zero shared mass the channel is measured (static gap) but returned unchanged."""
    rng = np.random.default_rng(12)
    o, nq, nw = 2, 2, 2
    q_mat = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))[0]
    comp_inv = np.tile((q_mat * np.array([0.3, 0.5, 1.0, 2.0])) @ q_mat.conj().T, (nq, nw, 1, 1))
    chi = _chi_from_compound(np.linalg.inv(comp_inv), o)
    ref = chi.mat.copy()
    annealer = _seeded_annealer(mass=0.0, initialized=False)
    out = annealer.apply(chi, _single_rank_dist())
    assert np.array_equal(out.mat, ref)
    assert np.allclose(annealer._gaps["dens"], 0.3, atol=1e-5)


def test_annealer_apply_measure_false_skips_measurement():
    """apply(measure=False) applies the current shared mass but never (re-)measures the gap."""
    rng = np.random.default_rng(14)
    o, nq, nw, lam = 2, 2, 2, 0.3
    q_mat = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))[0]
    comp_inv = np.tile((q_mat * np.array([0.4, 0.5, 1.0, 2.0])) @ q_mat.conj().T, (nq, nw, 1, 1))
    chi = _chi_from_compound(np.linalg.inv(comp_inv), o)
    annealer = _seeded_annealer(mass=lam, gaps={"dens": None})
    out = annealer.apply(chi, _single_rank_dist(), measure=False)
    assert annealer._gaps["dens"] is None
    assert np.allclose(_compound_of(out, o), np.linalg.inv(comp_inv + lam * np.eye(4)), atol=1e-4)


def test_annealer_init_damped_bump_from_worst_gap_and_inert_when_healthy(stab_logger):
    """Init ramps the shared mass a damped step (0.5) toward 1.5x the WORST gap; a healthy pair stays inert."""
    annealer = _seeded_annealer(gaps={"dens": -0.4, "magn": -0.1}, initialized=False)
    assert annealer.update(converged=False) is True and annealer._initialized is True
    assert np.allclose(annealer._mass, 0.5 * 1.5 * 0.4, atol=1e-12)  # 0.5 * target(worst gap -0.4)
    healthy = _seeded_annealer(gaps={"dens": 0.2, "magn": 0.1}, initialized=False)
    assert healthy.update(converged=False) is False and healthy._mass == 0.0


def test_annealer_bump_is_damped_toward_target(stab_logger):
    """A bump moves the shared mass halfway from its current value to 1.5x the worst violation, not all the way."""
    annealer = _seeded_annealer(mass=0.2, gaps={"dens": -0.3, "magn": -0.1})
    assert annealer.update(converged=False) is True
    assert np.allclose(annealer._mass, 0.2 + 0.5 * (1.5 * 0.3 - 0.2), atol=1e-12)  # worst gap -0.3


def test_annealer_converged_phase_halves_and_snaps_to_zero(stab_logger):
    """A converged phase with a healthy shifted gap halves the shared mass; below the floor it snaps to zero."""
    annealer = _seeded_annealer(mass=0.4, gaps={"dens": 0.01, "magn": 0.02})
    assert annealer.update(converged=True) is True
    assert np.allclose(annealer._mass, 0.2, atol=1e-12)
    annealer._mass = 1.5 * nonlocal_sde.LambdaAnnealer._LAMBDA_FLOOR
    assert annealer.update(converged=True) is True
    assert annealer._mass == 0.0


def test_annealer_bump_takes_precedence_over_halving_on_converged_phase(stab_logger):
    """A converged-but-still-violated shared gap is bumped, never halved: no tug-of-war in one iteration."""
    annealer = _seeded_annealer(mass=0.1, gaps={"dens": -0.3, "magn": 0.2})
    assert annealer.update(converged=True) is True
    assert np.allclose(annealer._mass, 0.1 + 0.5 * (1.5 * 0.3 - 0.1), atol=1e-12)


def test_annealer_rearms_when_pole_reopens_after_annealed_to_zero(stab_logger):
    """A scaffold annealed to zero bumps again if a static pole reopens (shared mass re-arms)."""
    annealer = _seeded_annealer(mass=0.0, gaps={"dens": -0.2, "magn": 0.01})
    assert annealer.update(converged=False) is True
    assert np.allclose(annealer._mass, 0.5 * 1.5 * 0.2, atol=1e-12)


def test_annealer_mass_capped_at_ceiling_with_warning(stab_logger):
    """The shared mass never exceeds the ceiling; hitting it emits a warning (warm-start advice)."""
    annealer = _seeded_annealer(mass=0.0, gaps={"dens": -1e6, "magn": 0.0})
    assert annealer.update(converged=False) is True
    assert annealer._mass == nonlocal_sde.LambdaAnnealer._MAX_LAMBDA
    assert stab_logger.warning.call_count == 1


def test_annealer_active_and_mass_present_flags():
    """active is True while uninitialized or a mass is present; mass_present tracks the shared mass only."""
    fresh = nonlocal_sde.LambdaAnnealer()
    assert fresh.active is True and fresh.mass_present is False
    healthy = _seeded_annealer(mass=0.0)
    assert healthy.active is False and healthy.mass_present is False
    massive = _seeded_annealer(mass=0.1)
    assert massive.active is True and massive.mass_present is True


def test_effective_epsilon_relaxed_while_annealing_active():
    """The convergence threshold is relaxed tenfold while a mass is present and full once annealed (or off)."""
    config.self_consistency.epsilon, config.stabilization.use_chi_phys_restriction = 1e-5, False
    assert np.allclose(nonlocal_sde._effective_epsilon(_seeded_annealer(mass=0.1)), 1e-4, atol=1e-15)
    assert np.allclose(nonlocal_sde._effective_epsilon(_seeded_annealer(mass=0.0)), 1e-5, atol=1e-15)
    assert np.allclose(nonlocal_sde._effective_epsilon(None), 1e-5, atol=1e-15)


def test_relative_sigma_residual_layout_mismatch_is_normalized():
    """A decompressed previous iterate against a compressed proposal compares matching momenta, not wrong ones."""
    config.box.niv_core = 3
    rng = np.random.default_rng(4)
    nk = (3, 2, 1)
    mat_dec = (rng.standard_normal((*nk, 1, 1, 10)) + 1j * rng.standard_normal((*nk, 1, 1, 10))).astype(np.complex64)
    sig_dec = SelfEnergy(mat_dec.copy(), nk, has_compressed_q_dimension=False, calc_smom=False, beta=10.0)
    sig_comp = SelfEnergy(
        mat_dec.reshape(6, 1, 1, 10).copy(), nk, has_compressed_q_dimension=True, calc_smom=False, beta=10.0
    )
    assert nonlocal_sde._relative_sigma_residual(sig_comp, sig_dec) == 0.0
    assert nonlocal_sde._relative_sigma_residual(sig_dec, sig_comp) == 0.0


def test_relative_sigma_residual_local_old_normalization_is_k_count_independent():
    """A constant offset between a local iterate and its full-BZ tiling gives the same residual for any nk."""
    config.box.niv_core = 3
    old_local = np.ones((1, 1, 1, 10), dtype=np.complex64)
    for nk_tot in (4, 16):
        new_full = 2.0 * np.ones((nk_tot, 1, 1, 10), dtype=np.complex64)
        out = nonlocal_sde._relative_sigma_residual(_make_resid_sigma(new_full), _make_resid_sigma(old_local))
        assert np.allclose(out, 1.0, atol=1e-12)


def test_select_and_apply_lambda_correction_dispatch(monkeypatch):
    """The rank-0 lambda selector dispatches by band count for both flags and returns chi unchanged when off."""
    sentinel, chi = object(), object()
    config.lambda_correction.perform_lambda_correction = False
    config.stabilization.use_lambda_correction = False
    assert nonlocal_sde._select_and_apply_lambda_correction(chi) is chi

    with monkeypatch.context() as mp:
        single = MagicMock(return_value=sentinel)
        mp.setattr(nonlocal_sde.LambdaCorrection, "perform", single)
        config.stabilization.use_lambda_correction = True
        config.sys.n_bands = 1
        assert nonlocal_sde._select_and_apply_lambda_correction(chi) is sentinel
        single.assert_called_once_with(chi)

    with monkeypatch.context() as mp:
        multi = MagicMock(return_value=sentinel)
        mp.setattr(nonlocal_sde.MultiOrbitalLambdaCorrection, "perform", multi)
        config.sys.n_bands = 2
        assert nonlocal_sde._select_and_apply_lambda_correction(chi) is sentinel
        multi.assert_called_once_with(chi)

    config.stabilization.use_lambda_correction = False
    config.lambda_correction.perform_lambda_correction = True
    with monkeypatch.context() as mp:
        single = MagicMock(return_value=sentinel)
        mp.setattr(nonlocal_sde.LambdaCorrection, "perform", single)
        config.sys.n_bands = 1
        assert nonlocal_sde._select_and_apply_lambda_correction(chi) is sentinel
        single.assert_called_once_with(chi)
    with monkeypatch.context() as mp:
        multi = MagicMock(return_value=sentinel)
        mp.setattr(nonlocal_sde.MultiOrbitalLambdaCorrection, "perform", multi)
        config.sys.n_bands = 2
        assert nonlocal_sde._select_and_apply_lambda_correction(chi) is sentinel
        multi.assert_called_once_with(chi)


def _setup_self_energy_loop(monkeypatch, tmp_path, proposal_step, max_iter=10, epsilon=1e-3):
    """Minimal single-k single-band environment for calculate_self_energy_q with a synthetic map and frozen mu."""
    config.lattice.k_grid = bz.KGrid((1, 1, 1), symmetries=[])
    config.lattice.hamiltonian = SimpleNamespace(get_ek=lambda: np.zeros((1, 1, 1, 1, 1)))
    config.box.niw_core, config.box.niv_core, config.box.niv_full, config.box.niv_dmft = 1, 2, 2, 8
    config.sys.beta, config.sys.mu, config.sys.n = 10.0, 0.5, 1.0
    config.output.output_path = str(tmp_path)
    config.self_consistency.previous_sc_path = ""
    config.self_consistency.mixing_strategy = "linear"
    config.self_consistency.mixing = 0.5
    config.self_consistency.epsilon = epsilon
    config.self_consistency.max_iter = max_iter
    config.self_energy_interpolation.do_interpolation = False
    logger = MagicMock()
    monkeypatch.setattr(config, "logger", logger, raising=False)

    gf_stub = SimpleNamespace(
        get_fill_nonlocal=lambda: (1.0, np.eye(1, dtype=np.complex128), np.zeros((1, 1, 1, 1, 1))),
        get_ekin=lambda: 0.0,
        get_epot=lambda: 0.0,
        save=lambda *a, **k: None,
        free=lambda: None,
    )
    monkeypatch.setattr(nonlocal_sde, "GreensFunction", SimpleNamespace(get_g_full=lambda *a, **k: gf_stub))
    monkeypatch.setattr(nonlocal_sde, "update_mu", lambda *a, **k: 0.5)
    monkeypatch.setattr(
        nonlocal_sde,
        "_update_occ_and_energies_distributed",
        lambda *a: (1.0, np.zeros((1, 1)), np.zeros((1, 1, 1, 1, 1)), 0.0, 0.0),
    )

    calls = []

    def fake_proposal(sigma_in, *args, annealer=None, **kwargs):
        calls.append(len(calls) + 1)
        return proposal_step(sigma_in, len(calls), annealer)

    monkeypatch.setattr(nonlocal_sde, "calculate_sigma_proposal", fake_proposal)

    mat = np.full((1, 1, 1, 16), 1.0 + 0.1j, dtype=np.complex64)
    sigma_dmft = SelfEnergy(mat, (1, 1, 1), has_compressed_q_dimension=True, beta=10.0)

    v_nonloc = MagicMock()
    v_nonloc.copy.return_value = v_nonloc
    v_nonloc.reduce_q.return_value = v_nonloc

    def run():
        return nonlocal_sde.calculate_self_energy_q(create_comm_mock(), None, v_nonloc, sigma_dmft, sigma_dmft.copy())

    return run, calls, logger


def test_loop_annealing_runs_pure_phase_after_mass_snaps_to_zero(monkeypatch, tmp_path):
    """When the annealing mass snaps to zero on a converged phase the loop runs one pure iteration before finishing."""
    seen = {}

    def step(sigma_in, n_call, annealer):
        seen["annealer"] = annealer
        annealer._gaps.update({"dens": -4e-2, "magn": 0.01} if n_call == 1 else {"dens": 0.01, "magn": 0.01})
        return sigma_in.copy()

    run, calls, _ = _setup_self_energy_loop(monkeypatch, tmp_path, step)
    config.stabilization.use_lambda_annealing = True
    run()

    assert len(calls) == 4
    assert seen["annealer"].mass_present is False


def test_loop_lambda_correction_release_runs_pure_phase_before_finishing(monkeypatch, tmp_path):
    """The lambda-correction scaffold releases on first relaxed convergence, then converges the pure map."""

    def step(sigma_in, n_call, annealer):
        return sigma_in.copy()

    run, calls, logger = _setup_self_energy_loop(monkeypatch, tmp_path, step)
    config.stabilization.use_lambda_correction = True
    run()

    assert len(calls) == 3
    assert config.stabilization.use_lambda_correction is False
    assert any(
        "Self-consistency with the lambda correction reached" in str(c.args[0]) for c in logger.info.call_args_list
    )


def test_loop_one_shot_lambda_correction_never_fires_release(monkeypatch, tmp_path):
    """The one-shot perform_lambda_correction never relaxes epsilon or fires the release; the flag stays enabled."""

    def step(sigma_in, n_call, annealer):
        return sigma_in.copy()

    run, calls, logger = _setup_self_energy_loop(monkeypatch, tmp_path, step)
    config.lambda_correction.perform_lambda_correction = True
    run()

    assert len(calls) == 2
    assert config.lambda_correction.perform_lambda_correction is True
    assert not any("lambda correction reached" in str(c.args[0]) for c in logger.info.call_args_list)


def test_annealer_update_without_measured_gaps_is_inert():
    """update is a no-op (and the scaffold stays uninitialized) while no channel gap has been measured yet."""
    annealer = nonlocal_sde.LambdaAnnealer()
    assert annealer.update(converged=True) is False
    assert annealer.active is True and annealer.mass_present is False


def test_annealer_steady_state_and_ceiling_pin_change_nothing(stab_logger):
    """A healthy unconverged phase keeps the mass untouched and a ceiling-pinned mass reports no change."""
    steady = _seeded_annealer(mass=0.2, gaps={"dens": 0.01, "magn": 0.3})
    assert steady.update(converged=False) is False and steady._mass == 0.2
    pinned = _seeded_annealer(mass=nonlocal_sde.LambdaAnnealer._MAX_LAMBDA, gaps={"dens": -1e6, "magn": 0.0})
    pinned._capped = True
    assert pinned.update(converged=False) is False
    assert pinned._mass == nonlocal_sde.LambdaAnnealer._MAX_LAMBDA


def test_annealer_static_gap_uses_omega_zero_slice_in_both_niw_ranges():
    """The static gap is read from the w=0 slice (niw full-range, 0 half-range) despite deeper non-static ones."""
    o, nq = 2, 2
    healthy = np.diag([0.7, 1.0, 2.0, 3.0]).astype(np.complex128)
    poled = np.diag([-5.0, 1.0, 2.0, 3.0]).astype(np.complex128)
    comp_full = np.tile(np.linalg.inv(np.stack([poled, healthy, poled])), (nq, 1, 1, 1))
    chi_full = _chi_from_compound(comp_full, o)
    chi_full._full_niw_range = True
    assert np.allclose(nonlocal_sde.LambdaAnnealer._static_gap(chi_full, _single_rank_dist()), 0.7, atol=1e-5)
    comp_half = np.tile(np.linalg.inv(np.stack([healthy, poled])), (nq, 1, 1, 1))
    chi_half = _chi_from_compound(comp_half, o)
    assert np.allclose(nonlocal_sde.LambdaAnnealer._static_gap(chi_half, _single_rank_dist()), 0.7, atol=1e-5)


def test_annealer_static_gap_reduces_min_across_ranks():
    """The measured static gap is the MPI.MIN of the per-rank q-slice minima, identical on every rank."""

    def fn(comm, rank):
        eigs = np.array([0.5, 1.0, 2.0, 3.0]) if rank == 0 else np.array([-0.7, 1.0, 2.0, 3.0])
        comp = np.tile(np.linalg.inv(np.diag(eigs).astype(np.complex128)), (2, 2, 1, 1))
        chi = _chi_from_compound(comp, 2)
        return nonlocal_sde.LambdaAnnealer._static_gap(chi, SimpleNamespace(comm=comm))

    _, results = run_parallel(2, fn)
    assert np.allclose(results[0], -0.7, atol=1e-5)
    assert np.allclose(results[1], -0.7, atol=1e-5)


def _dc_kernel_inputs(o, nq, niw, niv_core, niv_full, beta=8.0, seed=23):
    """Builds (f_dc_loc on the full nu x nu' box, gchi0_q, u_loc) for the double-counting kernel tests."""
    config.sys.beta = beta
    config.box.niw_core, config.box.niv_core = niw, niv_core
    config.box.niv_shell, config.box.niv_full = niv_full - niv_core, niv_full
    rng = np.random.default_rng(seed)
    f_shape = (o, o, o, o, niw + 1, 2 * niv_full, 2 * niv_full)
    f_dc_loc = LocalFourPoint(
        (rng.standard_normal(f_shape) + 1j * rng.standard_normal(f_shape)).astype(np.complex64),
        SpinChannel.MAGN,
        1,
        2,
        False,
        True,
    )
    b_shape = (nq, o, o, o, o, niw + 1, 2 * niv_full)
    gchi0_q = FourPoint(
        (rng.standard_normal(b_shape) + 1j * rng.standard_normal(b_shape)).astype(np.complex64),
        SpinChannel.NONE,
        nq=(nq, 1, 1),
        num_wn_dimensions=1,
        num_vn_dimensions=1,
        full_niw_range=False,
        has_compressed_q_dimension=True,
    )
    u_mat = rng.standard_normal((o,) * 4).astype(np.complex64)
    u_mat = 0.5 * (u_mat + u_mat.transpose(2, 3, 0, 1))  # pair-swap symmetry of physical tensors
    u_loc = LocalInteraction(u_mat, SpinChannel.NONE)
    return f_dc_loc, gchi0_q, u_loc


@pytest.mark.parametrize("o", [1, 2])
def test_sigma_dc_kernel_matches_explicit_reference(o):
    """The dc kernel equals -2/b^2 [sum_{nu'} F_magn chi0] @ U^{cbad} in the sigma-contraction layout.

    Only the local interaction enters (the V^q-attached local-vertex piece is not contained in DMFT and is kept).
    The reference reads the summed nu' off the stored first fermionic index via the compound symmetry
    F_{1dfe}(nu, nu') = F_{efd1}(nu', nu) of the symmetrized local vertex.
    """
    nq, niw, niv_core, niv_full = 3, 2, 2, 5
    f_dc_loc, gchi0_q, u_loc = _dc_kernel_inputs(o, nq, niw, niv_core, niv_full)
    beta = config.sys.beta

    out = nonlocal_sde.calculate_sigma_dc_kernel(f_dc_loc, gchi0_q.copy(), u_loc)

    window = slice(niv_full - niv_core, niv_full + niv_core)
    t = np.einsum("efgiwvp,qefdcwv->qigdcwp", f_dc_loc.mat, gchi0_q.mat, optimize=True)[..., window]
    u_perm = u_loc.mat.transpose(2, 1, 0, 3)  # "abcd->cbad": the transversal attachment U_{acb2}
    m = np.einsum("qigdcwp,cdaj->qigajwp", t, u_perm, optimize=True)
    ref = -2.0 / beta**2 * m.transpose(0, 2, 1, 4, 3, 5, 6)  # "abcd->badc" into the contraction layout
    assert out.niv == niv_core
    assert np.allclose(out.mat, ref, atol=1e-4)


def test_sigma_dc_kernel_sums_the_stored_first_fermionic_index_over_the_full_box():
    """The kernel contracts the vertex's stored first index (the summed nu') over the whole asymptotic box."""
    nq, niw, niv_core, niv_full = 2, 1, 2, 5
    f_dc_loc, gchi0_q, u_loc = _dc_kernel_inputs(1, nq, niw, niv_core, niv_full)
    f_shell_zeroed = f_dc_loc.copy()
    f_shell_zeroed.mat[..., : niv_full - niv_core, :] = 0.0

    full = nonlocal_sde.calculate_sigma_dc_kernel(f_dc_loc, gchi0_q.copy(), u_loc)
    zeroed = nonlocal_sde.calculate_sigma_dc_kernel(f_shell_zeroed, gchi0_q.copy(), u_loc)

    assert not np.allclose(zeroed.mat, full.mat, atol=1e-6)


def test_create_auxiliary_chi_r_q_sum_matches_full_inversion_reference():
    """The chunked auxiliary-susceptibility sum equals the full compound inversion summed over the last frequency."""
    rng = np.random.default_rng(31)
    gamma, gchi0_q_inv, u_loc, v_nonloc = _bse_assembly_inputs(rng)
    ref = nonlocal_sde.create_auxiliary_chi_r_q(gamma, gchi0_q_inv, u_loc, v_nonloc).sum_over_vn(config.sys.beta)
    out = nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_q_inv, u_loc, v_nonloc)
    assert np.allclose(out.mat, ref.mat, atol=1e-5)
    assert out.channel == gamma.channel and not out.full_niw_range and out.num_vn_dimensions == 1


def test_create_auxiliary_chi_r_q_sum_is_chunk_size_invariant(monkeypatch):
    """A one-element chunk budget reproduces the whole-box result of the chunked auxiliary-susceptibility sum."""
    rng = np.random.default_rng(32)
    gamma, gchi0_q_inv, u_loc, v_nonloc = _bse_assembly_inputs(rng)
    whole = nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_q_inv, u_loc, v_nonloc)
    monkeypatch.setattr(nonlocal_sde, "SLICE_CHUNK_BYTES", 1)
    chunked = nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_q_inv, u_loc, v_nonloc)
    assert np.allclose(chunked.mat, whole.mat, atol=1e-6)


def test_update_occ_and_energies_distributed_matches_the_full_box_evaluation(monkeypatch):
    """The k-distributed occupation/energy evaluation matches the single-rank full-box reference."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, o, niv, niv_dmft, beta, mu = (4, 2, 1), 2, 3, 8, 9.0, 0.4
    nk_tot = int(np.prod(nk))
    rng = np.random.default_rng(9)
    config.sys.beta, config.sys.n_bands, config.sys.mu = beta, o, mu
    config.lattice.nk = nk
    config.lattice.k_grid = SimpleNamespace(nk_tot=nk_tot, nk=nk)
    ek = rng.standard_normal((*nk, o, o))
    ek = ek + ek.swapaxes(-1, -2)
    config.lattice.hamiltonian = MagicMock(get_ek=MagicMock(return_value=ek))
    sig_mat = (rng.standard_normal((nk_tot, o, o, 2 * niv)) * 0.1 + 0.3j).astype(np.complex64)
    dmft_mat = (rng.standard_normal((1, 1, 1, o, o, 2 * niv_dmft)) * 0.1 + 0.2j).astype(np.complex64)
    sigma_new = SelfEnergy(sig_mat.copy(), nk, has_compressed_q_dimension=True, beta=beta)
    sigma_dmft_full = SelfEnergy(dmft_mat.copy(), (1, 1, 1), beta=beta)

    sigma_ref = sigma_new.copy().concatenate_self_energies(sigma_dmft_full)
    giwk_ref = GreensFunction.get_g_full(sigma_ref, mu, ek, beta)
    _, occ_ref, occ_k_ref = giwk_ref.get_fill_nonlocal()
    ekin_ref, epot_ref = giwk_ref.get_ekin(), giwk_ref.get_epot()

    def fn(comm, rank):
        d_full = mpi_utils.MpiDistributor(ntasks=nk_tot, comm=comm)
        return nonlocal_sde._update_occ_and_energies_distributed(sigma_new, sigma_dmft_full, d_full, mu)

    _, res = run_parallel(2, fn)
    # occupations reproduce the full-box reference bit-for-bit; only the energy scalars regroup their k-sums
    for _, occ, occ_k, ekin, epot in res:
        assert np.array_equal(occ, occ_ref) and np.array_equal(occ_k, occ_k_ref)
        assert np.allclose([ekin, epot], [ekin_ref, epot_ref], atol=1e-5)


def test_update_occ_and_energies_distributed_pins_the_occupation_dtype_across_ranks(monkeypatch):
    """Ranks whose fill comes out float (all-real eigenvalues) and ranks with complex fill reduce to one complex128 occ_k."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, o, niv, niv_dmft, beta, mu = (4, 2, 1), 2, 3, 8, 9.0, 0.4
    nk_tot = int(np.prod(nk))
    rng = np.random.default_rng(11)
    config.sys.beta, config.sys.n_bands, config.sys.mu = beta, o, mu
    config.lattice.nk = nk
    config.lattice.k_grid = SimpleNamespace(nk_tot=nk_tot, nk=nk)
    ek = rng.standard_normal((*nk, o, o))
    config.lattice.hamiltonian = MagicMock(get_ek=MagicMock(return_value=ek + ek.swapaxes(-1, -2)))
    sigma_new = SelfEnergy(
        (rng.standard_normal((nk_tot, o, o, 2 * niv)) * 0.1 + 0.3j).astype(np.complex64),
        nk,
        has_compressed_q_dimension=True,
        beta=beta,
    )
    sigma_dmft_full = SelfEnergy(
        (rng.standard_normal((1, 1, 1, o, o, 2 * niv_dmft)) * 0.1 + 0.2j).astype(np.complex64), (1, 1, 1), beta=beta
    )

    def fake_get_g_full(sigma_occ, mu_in, ek_slice, beta_in):
        # the first slice (rows starting at ek row 0) plays the all-real-eigenvalue rank and returns a float fill
        n_my = ek_slice.shape[0]
        is_first = np.allclose(ek_slice[0, 0, 0], config.lattice.hamiltonian.get_ek().reshape(nk_tot, o, o)[0])
        dtype, value = (np.float64, 0.25) if is_first else (np.complex128, 0.75 + 0.5j)
        g = MagicMock()
        g.get_fill_nonlocal.return_value = (0.0, None, np.full((n_my, 1, 1, o, o), value, dtype=dtype))
        g.get_ekin.return_value = 1.0
        g.get_epot.return_value = 2.0
        return g

    def fn(comm, rank):
        d_full = mpi_utils.MpiDistributor(ntasks=nk_tot, comm=comm)
        return nonlocal_sde._update_occ_and_energies_distributed(sigma_new, sigma_dmft_full, d_full, mu)

    with monkeypatch.context() as mp:
        mp.setattr(nonlocal_sde.GreensFunction, "get_g_full", fake_get_g_full)
        _, res = run_parallel(2, fn)
    for _, occ, occ_k, ekin, epot in res:
        assert occ_k.dtype == np.complex128
        assert np.allclose(occ_k.reshape(nk_tot, o, o)[0], 0.25) and np.allclose(
            occ_k.reshape(nk_tot, o, o)[-1], 0.75 + 0.5j
        )


def test_share_sigma_per_node_gives_each_node_one_shared_buffer_with_rank0_values(monkeypatch):
    """_share_sigma_per_node leaves every rank viewing its node's single shared buffer holding rank 0's array."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    mixed = (np.arange(2 * 2 * 6).reshape(2, 2, 6) + 1j).astype(np.complex64)

    def fn(comm, rank):
        sigma = SimpleNamespace(mat=mixed.copy() if rank == 0 else np.zeros_like(mixed))
        node_comm = comm.Split_type(0)
        roots_comm = comm.Split(0 if node_comm.rank == 0 else 1)
        win = nonlocal_sde._share_sigma_per_node(sigma, node_comm, roots_comm)
        return sigma.mat.copy(), sigma.mat.__array_interface__["data"][0], win is not None

    _, res = run_parallel(4, fn, hostnames=["n0", "n0", "n1", "n1"])
    pointers = {r[1] for r in res}
    assert all(np.array_equal(r[0], mixed) for r in res)
    assert len(pointers) == 2 and all(r[2] for r in res)


@pytest.mark.parametrize("negative_w", [False, True])
def test_fft_sde_pass_is_invariant_under_the_w_chunk_size(negative_w, monkeypatch):
    """A one-byte chunk budget reproduces the all-w-at-once result of the chunked FFT self-energy pass."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, o, niw, niv, beta = (4, 4, 1), 2, 3, 4, 12.5
    rng = np.random.default_rng(21)
    niv_g = niv + niw
    g_shape = (*nk, o, o, 2 * niv_g)
    g_mat = (rng.standard_normal(g_shape) + 1j * rng.standard_normal(g_shape)).astype(np.complex64)
    n_irr = bz.KGrid(nk, bz.two_dimensional_square_symmetries()).nk_irr
    k_shape = (n_irr, o, o, o, o, niw + 1, 2 * niv)
    kernel_mat = (rng.standard_normal(k_shape) + 1j * rng.standard_normal(k_shape)).astype(np.complex64)

    def fn(comm, rank):
        config.lattice.nk = nk
        config.lattice.k_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
        config.box.niw_core, config.box.niv_core = niw, niv
        config.sys.n_bands, config.sys.beta = o, beta
        config.logger = MagicMock()
        d_irr = mpi_utils.MpiDistributor(ntasks=n_irr, comm=comm)
        d_full = mpi_utils.MpiDistributor(ntasks=int(np.prod(nk)), comm=comm)
        giwk = GreensFunction(g_mat.copy(), calc_filling=False, nk=nk, beta=beta)
        g_r_local = nonlocal_sde._build_rspace_giwk_pencil(giwk, d_irr)
        pairs = [(i, -i) for i in range(1, niw + 1)] if negative_w else [(i, i) for i in range(niw + 1)]
        results = []
        for chunk_bytes in (2**62, 1):
            kernel = FourPoint(
                kernel_mat[d_irr.my_slice].copy(),
                SpinChannel.NONE,
                nk,
                1,
                1,
                full_niw_range=False,
                has_compressed_q_dimension=True,
            )
            sigma = nonlocal_sde._run_fft_sde_pass(
                kernel, d_irr, d_full, g_r_local, niv_g, pairs, negative_w, chunk_bytes
            )
            results.append(sigma.mat.copy())
        return results

    _, res = run_parallel(3, fn)
    for whole, chunked in res:
        assert np.allclose(chunked, whole, atol=1e-5)

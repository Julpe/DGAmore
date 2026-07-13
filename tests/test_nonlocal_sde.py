# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import itertools
import os
from unittest import mock

import numpy as np
import pytest

import dgamore.config as config
import dgamore.nonlocal_sde as nonlocal_sde
from dgamore.greens_function import GreensFunction
from dgamore.hamiltonian import Hamiltonian
from dgamore.jacobian_stabilization import PhysicalSolutionStabilizer
from dgamore.interaction import Interaction
from dgamore.local_sde import get_local_hartree_fock
from dgamore.n_point_base import SpinChannel
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
from tests.conftest import create_comm_mock

LOCAL_SDE_DATA = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/local_sde"


def test_init_mu_history_fresh_uses_current_mu():
    """A fresh run (starting_iter == 0) seeds the history with the current (DMFT) chemical potential and leaves it."""
    config.sys.mu = 0.7
    config.self_consistency.previous_sc_path = ""

    mu_history = _init_mu_history(0)

    assert mu_history == [0.7]
    assert config.sys.mu == 0.7  # unchanged on a fresh run


def test_init_mu_history_from_previous_syncs_global_mu(monkeypatch, tmp_path):
    """Resuming from a previous run seeds the history with, and syncs config.sys.mu to, that run's last mu."""
    config.sys.mu = 0.3  # stale DMFT value that must be overwritten
    config.self_consistency.previous_sc_path = str(tmp_path)
    previous_mu = 1.5
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: np.array([0.9, 1.2, previous_mu]))

    mu_history = _init_mu_history(3)

    assert mu_history == [previous_mu]
    assert config.sys.mu == previous_mu  # global synced to the previous run's converged mu


def test_nonlocal_hartree_fock_matches_local_reference():
    """
    The non-local Hartree-Fock term must reduce to the (DMFT-validated) local Hartree-Fock at every k when the
    non-local interaction vanishes (V=0) and the occupation is k-independent. This covers the same orbital-index
    convention as the local test (:func:`dgamore.local_sde.get_local_hartree_fock`): the Hartree must pick up the
    inter-orbital density U' (stored at U_{abab}), not U_{aabb}. Reverting the ``qacbd`` contraction in
    ``get_hartree_fock`` makes this fail.
    """
    nb = 2
    nk = (2, 2, 1)
    nq_tot = int(np.prod(nk))

    config.lattice.nk = nk
    config.sys.n_bands = nb

    occ = np.load(f"{LOCAL_SDE_DATA}/occ.npy", allow_pickle=False)
    config.sys.occ = occ
    config.sys.occ_k = np.broadcast_to(occ, nk + (nb, nb)).copy()  # k-independent occupation

    u_loc = Hamiltonian().read_umatrix(f"{LOCAL_SDE_DATA}/u_matrix.dat").get_local_u()
    v_nonloc = Interaction(
        np.zeros((nq_tot, nb, nb, nb, nb), dtype=u_loc.mat.dtype),
        SpinChannel.NONE,
        nk,
        has_compressed_q_dimension=True,
    )
    q_list = np.array(list(itertools.product(*[range(n) for n in nk])))

    hartree, fock = get_hartree_fock(u_loc, v_nonloc, q_list)
    hf_nonlocal = (hartree + fock)[..., 0]  # [nk_tot, nb, nb]

    sigma_hf_ref = np.load(f"{LOCAL_SDE_DATA}/sigma_HF.npy", allow_pickle=False)
    assert hf_nonlocal.shape == (nq_tot, nb, nb)
    assert np.allclose(hf_nonlocal, sigma_hf_ref[None, ...])
    # the same reference is the local Hartree-Fock, so the two SDE paths agree
    assert np.allclose(hf_nonlocal, get_local_hartree_fock(u_loc, occ)[None, ...])


class _ConstantChi:
    """Minimal physical-susceptibility stand-in whose BZ and frequency reductions are identities."""

    def __init__(self, mat: np.ndarray):
        """Stores the orbital-resolved matrix that the reduction chain returns unchanged."""
        self._mat = mat

    def copy(self):
        """Identity copy; the reduction chain is non-mutating so a fresh wrapper suffices."""
        return _ConstantChi(self._mat)

    def map_to_full_bz(self, grid):
        """Identity unfolding to the full BZ."""
        return self

    def to_half_niw_range(self):
        """Identity reduction to the half niw range."""
        return self

    def take_first_wn(self):
        """Identity selection of the first bosonic frequency."""
        return self

    @property
    def mat(self) -> np.ndarray:
        """The backing orbital-resolved matrix."""
        return self._mat


def test_ornstein_zernike_fit_aggregates_nonconverged_warnings(monkeypatch):
    """All non-converging OZ fits collapse into a single aggregated warning instead of one log per orbital."""
    config.sys.n_bands = 2
    logger = mock.Mock()
    monkeypatch.setattr(config, "logger", logger, raising=False)
    monkeypatch.setattr(nonlocal_sde.opt, "curve_fit", mock.Mock(side_effect=RuntimeError("forced non-convergence")))

    perform_ornstein_zernike_fit(_ConstantChi(np.ones((2, 2, 1, 2, 2, 2, 2), dtype=np.complex64)))

    logger.warning.assert_called_once()
    msg = logger.warning.call_args.args[0]
    assert "16 orbital combination(s)" in msg
    assert "(1, 1, 1, 1)" in msg and "(2, 2, 2, 2)" in msg  # 1-based orbital labels, not 0-based


def test_ornstein_zernike_fit_logs_no_warning_when_all_converge(monkeypatch):
    """A fully converging set of OZ fits emits no warning at all (the aggregation guard stays silent)."""
    config.sys.n_bands = 2
    logger = mock.Mock()
    monkeypatch.setattr(config, "logger", logger, raising=False)
    monkeypatch.setattr(nonlocal_sde.opt, "curve_fit", mock.Mock(return_value=(np.array([1.0, 2.0]), None)))

    perform_ornstein_zernike_fit(_ConstantChi(np.ones((2, 2, 1, 2, 2, 2, 2), dtype=np.complex64)))

    logger.warning.assert_not_called()


def _tiny_sigma_and_ek(nb=1, niv=4):
    """Builds a minimal single-k self-energy and dispersion for the Dyson build."""
    sigma = SelfEnergy(np.zeros((1, 1, 1, nb, nb, 2 * niv), dtype=np.complex64), calc_smom=False, beta=10.0)
    ek = np.zeros((1, 1, 1, nb, nb), dtype=np.complex64)
    return sigma, ek


def test_build_giwk_full_disabled_matches_direct_dyson_and_skips_split():
    """With node-sharing off, _build_giwk_full is the plain Dyson build and never touches the communicator."""
    config.memory.use_shared_memory_common_obj = False
    sigma, ek = _tiny_sigma_and_ek()
    comm = create_comm_mock()

    giwk, win, node_comm = _build_giwk_full(comm, sigma, 0.3, ek, 10.0)

    assert win is None and node_comm is None
    assert comm.Split_type.called is False
    assert np.allclose(giwk.mat, GreensFunction.get_g_full(sigma, 0.3, ek, 10.0).mat, atol=1e-6)


def test_build_giwk_full_shared_single_rank_matches_direct_dyson():
    """With node-sharing on but a single-rank node, the giwk is bit-parity with the direct Dyson build (no window)."""
    config.memory.use_shared_memory_common_obj = True
    sigma, ek = _tiny_sigma_and_ek()
    comm = create_comm_mock()

    giwk, win, node_comm = _build_giwk_full(comm, sigma, 0.3, ek, 10.0)

    assert win is None  # a single-rank node needs no shared window
    assert np.array_equal(giwk.mat, GreensFunction.get_g_full(sigma, 0.3, ek, 10.0).mat)
    _release_shared_giwk(win, node_comm)  # must not raise on the single-rank / mock path


def test_release_shared_giwk_without_sharing_is_noop():
    """Releasing a non-shared giwk (both handles None) is a safe no-op."""
    _release_shared_giwk(None, None)


def test_release_shared_giwk_frees_window_and_communicator():
    """With a window allocated, _release barriers (so no rank still reads), frees the window, then the node comm."""
    win, node_comm = mock.Mock(), mock.Mock()
    _release_shared_giwk(win, node_comm)
    node_comm.Barrier.assert_called_once()
    win.Free.assert_called_once()
    node_comm.Free.assert_called_once()


def test_free_shared_window_frees_window_but_keeps_communicator():
    """_free_shared_window barriers and frees the window but leaves the node communicator alive (reused for the cut)."""
    win, node_comm = mock.Mock(), mock.Mock()
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
    from dgamore.four_point import FourPoint
    from dgamore.interaction import Interaction, LocalInteraction
    from dgamore.local_four_point import LocalFourPoint

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


def test_assemble_bse_matrix_matches_two_block_expression():
    """The fused single-block BSE-matrix assembly must be bit-equal to the former add/extend/subtract chain
    (broadcast gamma fill + diagonal chi0^-1 add + in-place interaction subtract commute elementwise with it)
    and must leave gamma and gchi0_q_inv untouched."""
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


def test_assemble_bse_matrix_accepts_half_niw_gamma():
    """A gamma already in the half bosonic range assembles identically to its full-range twin."""
    rng = np.random.default_rng(22)
    gamma, gchi0_q_inv, u_loc, v_nonloc = _bse_assembly_inputs(rng)
    u_r = v_nonloc.as_channel(gamma.channel) + u_loc.as_channel(gamma.channel)
    full = nonlocal_sde.create_inverse_auxiliary_chi_r_q(gamma, gchi0_q_inv, u_r)
    half = nonlocal_sde.create_inverse_auxiliary_chi_r_q(gamma.copy().to_half_niw_range(), gchi0_q_inv, u_r)
    assert np.array_equal(full.mat, half.mat)


def test_create_auxiliary_chi_r_q_sum_v1_matches_explicit_reference():
    """The fused-assembly v1 auxiliary susceptibility must reproduce the explicit two-block expression followed by
    the fused invert-and-sum, locking the rewired create_auxiliary_chi_r_q_sum_v1 against the pre-fusion result."""
    rng = np.random.default_rng(23)
    gamma, gchi0_q_inv, u_loc, v_nonloc = _bse_assembly_inputs(rng)
    ref = _bse_assembly_reference(gamma, gchi0_q_inv, u_loc, v_nonloc).invert_and_sum_over_last_vn(config.sys.beta)
    out = nonlocal_sde.create_auxiliary_chi_r_q_sum_v1(gamma, gchi0_q_inv, u_loc, v_nonloc)
    assert np.allclose(out.mat, ref.mat, atol=1e-10)
    assert out.channel == SpinChannel.DENS


def test_create_auxiliary_chi_r_q_matches_explicit_reference():
    """The fused-assembly full inversion variant must reproduce the explicit two-block expression inverted in
    compound space, locking the rewired create_auxiliary_chi_r_q (also used by the Eliashberg vertex build)."""
    rng = np.random.default_rng(24)
    gamma, gchi0_q_inv, u_loc, v_nonloc = _bse_assembly_inputs(rng)
    ref = _bse_assembly_reference(gamma, gchi0_q_inv, u_loc, v_nonloc).invert(False)
    out = nonlocal_sde.create_auxiliary_chi_r_q(gamma, gchi0_q_inv, u_loc, v_nonloc)
    assert np.allclose(out.mat, ref.mat, atol=1e-10)


def test_calculate_kernel_r_q_matches_identity_like_reference():
    """The self-energy kernel U_r (gamma - gamma U_r chi - 2/3 identity)_magn must equal the explicit expression
    built with the full identity_like block, locking the in-place orbital-diagonal subtraction (the identity is
    nonzero at o1 == o4, o2 == o3 for every frequency); the density channel carries no identity term."""
    from dgamore.four_point import FourPoint

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
        ref = vrg.copy() - vrg.copy() @ u_r @ chi.copy()
        if channel == SpinChannel.MAGN:
            ref = ref - FourPoint.identity_like(ref).scale(2.0 / 3.0)
        ref = u_r @ ref
        out = nonlocal_sde.calculate_kernel_r_q(vrg, chi, v_nonloc, u_loc)
        assert np.allclose(out.mat, ref.mat, atol=1e-10)


def test_vrg_right_is_first_frequency_summed_three_leg_vertex():
    """The right-sided three-leg vertex must equal its definition gamma-tilde^{qv}_{1234} = beta sum_{ab} sum_{v_1}
    chi*^{q v_1 v}_{12ab} (chi^{qv}_{0;ba34})^{-1}, which the dcba orbital permutation of the last-frequency-summed
    chi* provides for a time-reversal-symmetric chi* (chi*^{q v v'}_{1234} = chi*^{q v' v}_{4321}); the left vertex
    gamma^{qv}_{1234} = beta sum_{ab} sum_{v'} (chi^{qv}_{0;12ab})^{-1} chi*^{q v v'}_{ba34} is locked alongside."""
    from dgamore.four_point import FourPoint
    from dgamore.n_point_base import SpinChannel

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
    """The unused q-loop self-energy contraction variants - the plain reference, the Fortran-buffered CPU one, the
    GPU one (run through a numpy-backed cupy stub) and the auto dispatcher (falling back to the CPU) - produce
    matching self-energies on synthetic full-BZ kernel and Green's-function data, locking the kept functions."""
    import types

    import dgamore.brillouin_zone as bz
    from dgamore.four_point import FourPoint

    nk, o, niw, niv = (4, 4, 1), 2, 3, 4
    config.lattice.nk = nk
    config.lattice.k_grid = bz.KGrid(nk, symmetries=[])
    config.box.niw_core = niw
    config.box.niv_core = niv
    config.sys.n_bands = o
    config.sys.beta = 12.5
    config.logger = mock.MagicMock()

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
    sigma_cpu = nonlocal_sde.calculate_sigma_from_kernel_cpu(make_kernel(), giwk, q_list)

    cupy_stub = types.ModuleType("cupy")
    cupy_stub.zeros, cupy_stub.asarray, cupy_stub.arange = np.zeros, np.asarray, np.arange
    cupy_stub.einsum, cupy_stub.asnumpy = np.einsum, lambda x: x
    with mock.patch.dict("sys.modules", {"cupy": cupy_stub}):
        sigma_gpu = nonlocal_sde.calculate_sigma_from_kernel_gpu(make_kernel(), giwk, q_list)
    with mock.patch.dict("sys.modules", {"cupy": None}):
        sigma_auto = nonlocal_sde.calculate_sigma_from_kernel_auto(mock.MagicMock(), make_kernel(), giwk, q_list)

    assert np.allclose(sigma_cpu.mat, sigma_ref.mat, atol=1e-6)
    assert np.allclose(sigma_gpu.mat, sigma_ref.mat, atol=1e-6)
    assert np.array_equal(sigma_auto.mat, sigma_cpu.mat)


def _chi_from_compound(comp: np.ndarray, o: int):
    """Builds a 0-vn FourPoint [q, o, o, o, o, w] from a compound array [q, w, o^2, o^2] (rows (12), cols (43))."""
    from dgamore.four_point import FourPoint

    nq, nw = comp.shape[:2]
    mat = np.transpose(comp.reshape(nq, nw, o, o, o, o), (0, 2, 3, 5, 4, 1))
    return FourPoint(mat.copy(), SpinChannel.DENS, (nq, 1, 1), 1, 0, False, True, True)


def _compound_of(chi, o: int) -> np.ndarray:
    """Extracts the compound array [q, w, o^2, o^2] from a 0-vn FourPoint in full-index layout."""
    return chi.mat.transpose(0, 5, 1, 2, 4, 3).reshape(chi.mat.shape[0], chi.mat.shape[-1], o * o, o * o)


def test_restrict_chi_phys_floors_negative_inverse_eigenvalues():
    """A compound block of chi^{qw} with a negative eigenvalue must come back with that eigenvalue replaced by
    1/floor (the inverse eigenvalue is floored at +floor) while all positive eigenpairs are preserved exactly."""
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
    """A Hermitian positive-definite chi with negative real off-diagonal ENTRIES must pass through unchanged - the
    eigenvalue floor must not clamp legitimately negative matrix elements like the old elementwise version did."""
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
    """For a single band the compound block is a scalar, so the eigenvalue floor must reproduce the elementwise
    behavior: negative chi maps to 1/floor, positive chi is unchanged."""
    floor = 1e-4
    mat = np.array([-0.5, 0.3], dtype=complex).reshape(2, 1, 1, 1, 1, 1)
    from dgamore.four_point import FourPoint

    chi = FourPoint(mat.copy(), SpinChannel.DENS, (2, 1, 1), 1, 0, False, True, True)
    out, n_floored = nonlocal_sde.restrict_chi_phys_to_positive_eigenvalues(chi, floor=floor)
    assert n_floored == 1
    assert np.allclose(out.mat.ravel() / np.array([1.0 / floor, 0.3]), 1.0, atol=1e-3)


def test_min_static_compound_eigenvalue_reports_definiteness():
    """The static (w = 0) compound blocks of a positive-definite chi give a positive minimum eigenvalue, and a
    single negative eigenvalue planted in one w = 0 block is reported exactly."""
    rng = np.random.default_rng(6)
    o, nq, nw = 2, 3, 4
    a = rng.standard_normal((nq, nw, 4, 4)) + 1j * rng.standard_normal((nq, nw, 4, 4))
    comp = a @ np.conj(np.transpose(a, (0, 1, 3, 2))) + 0.5 * np.eye(4)
    assert nonlocal_sde.min_static_compound_eigenvalue(_chi_from_compound(comp, o)) > 0.0
    q_vec = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))[0]
    comp[1, 0] = (q_vec * np.array([-3.0, 0.5, 1.0, 2.0])) @ q_vec.conj().T
    assert np.allclose(nonlocal_sde.min_static_compound_eigenvalue(_chi_from_compound(comp, o)), -3.0, atol=1e-3)


def test_effective_epsilon_is_relaxed_while_restriction_is_active():
    """The self-energy convergence threshold is 10x epsilon while the susceptibility restriction is active (the
    restricted phase is only a scaffold for the released phase) and the plain epsilon after the release."""
    config.self_consistency.epsilon = 1e-5
    config.stabilization.use_chi_phys_restriction = True
    assert np.allclose(nonlocal_sde._effective_epsilon(), 1e-4, atol=1e-18)
    config.stabilization.use_chi_phys_restriction = False
    assert np.allclose(nonlocal_sde._effective_epsilon(), 1e-5, atol=1e-18)


def test_effective_epsilon_is_relaxed_while_lambda_correction_is_active():
    """The convergence threshold is 10x epsilon while the per-iteration lambda correction is active (a releasing
    scaffold, like use_chi_phys_restriction), the plain epsilon once it is disabled, and never relaxed by the
    one-shot perform_lambda_correction."""
    config.self_consistency.epsilon = 1e-5
    config.stabilization.use_chi_phys_restriction = False
    config.stabilization.use_lambda_correction = True
    assert np.allclose(nonlocal_sde._effective_epsilon(), 1e-4, atol=1e-18)
    config.stabilization.use_lambda_correction = False
    config.lambda_correction.perform_lambda_correction = True
    assert np.allclose(nonlocal_sde._effective_epsilon(), 1e-5, atol=1e-18)


def test_sde_fft_rspace_greens_function_node_sharing_matches_private_build():
    """calculate_sigma_from_kernel_fft_cpu with a node communicator routes the R-space Green's function through the
    node-shared window builder and reproduces the private per-rank build bit-identically."""
    import dgamore.brillouin_zone as bz
    from dgamore.four_point import FourPoint
    from tests import conftest

    nk, o, niw, niv = (4, 4, 1), 2, 3, 4
    config.lattice.nk = nk
    config.lattice.k_grid = bz.KGrid(nk, symmetries=[])
    config.box.niw_core = niw
    config.box.niv_core = niv
    config.sys.n_bands = o
    config.sys.beta = 12.5
    config.memory.use_shared_memory_common_obj = True
    config.logger = mock.MagicMock()

    rng = np.random.default_rng(9)
    g_shape = (*nk, o, o, 2 * (niv + niw + 2))
    k_shape = (int(np.prod(nk)), o, o, o, o, niw + 1, 2 * niv)
    g_mat = (rng.standard_normal(g_shape) + 1j * rng.standard_normal(g_shape)).astype(np.complex64)
    kernel_mat = (rng.standard_normal(k_shape) + 1j * rng.standard_normal(k_shape)).astype(np.complex64)
    giwk = GreensFunction(g_mat, calc_filling=False, nk=nk, beta=config.sys.beta)
    mpi_dist = mock.MagicMock()
    mpi_dist.comm = conftest.create_comm_mock()
    pairs = [(i, i) for i in range(niw + 1)]

    def make_kernel():
        return FourPoint(
            kernel_mat.copy(), SpinChannel.NONE, nk, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
        )

    class _SingleRankComm:
        @staticmethod
        def Get_rank():
            return 0

        @staticmethod
        def Get_size():
            return 1

    sigma_plain = nonlocal_sde.calculate_sigma_from_kernel_fft_cpu(mpi_dist, make_kernel(), giwk, pairs)
    sigma_shared = nonlocal_sde.calculate_sigma_from_kernel_fft_cpu(
        mpi_dist, make_kernel(), giwk, pairs, node_comm=_SingleRankComm()
    )
    assert np.array_equal(sigma_shared.mat, sigma_plain.mat)


def test_load_node_shared_local_vertex_private_path_applies_transform(monkeypatch):
    """Without a node communicator the helper loads privately, applies the transform and returns win=None, matching
    the former per-rank load + permute + scale chain."""
    from dgamore.local_four_point import LocalFourPoint

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
    import dgamore.mpi_utils as mpi_utils_mod
    from tests.conftest import FAKE_MPI, run_parallel

    monkeypatch.setattr(mpi_utils_mod, "MPI", FAKE_MPI)
    config.memory.use_shared_memory_common_obj = True
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


class FakeSelfEnergyWindow:
    """Duck-typed SelfEnergy stand-in exposing only what apply_modified_preconditioner touches."""

    def __init__(self, mat, niv):
        self.mat = mat
        self.niv = niv
        self.compress_calls = 0

    def compress_q_dimension(self):
        self.compress_calls += 1
        return self


class RecordingStabLogger:
    """Counting logger stub for the stabilizer glue tests."""

    def __init__(self):
        self.calls = {"info": 0, "debug": 0, "log_memory_usage": 0, "warning": 0}

    def info(self, *a, **k):
        self.calls["info"] += 1

    def debug(self, *a, **k):
        self.calls["debug"] += 1

    def log_memory_usage(self, *a, **k):
        self.calls["log_memory_usage"] += 1

    def warning(self, *a, **k):
        self.calls["warning"] += 1


def _vec_inner(m):
    """Flattens a complex window tensor into the stabilizer's real [Re; Im] vector."""
    f = m.reshape(-1)
    return np.concatenate((f.real, f.imag))


def _build_window_stabilizer(inner_shape, p, *, unstable):
    """Builds a real PhysicalSolutionStabilizer on an affine inner-window map with or without one unstable mode."""
    nc = int(np.prod(inner_shape))
    n = 2 * nc

    def to_mat(v):
        return (v[:nc] + 1j * v[nc:]).reshape(inner_shape)

    rng = np.random.default_rng(0)
    q_mat, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = rng.uniform(-0.4, 0.4, n)
    if unstable:
        s[0] = (1.30 - (1.0 - p)) / p
    jac = q_mat @ np.diag(s) @ q_mat.T
    b = rng.standard_normal(n)

    def proposal(mat):
        return to_mat(jac @ _vec_inner(mat) + b)

    xstar = np.linalg.solve(np.eye(n) - jac, b)
    return PhysicalSolutionStabilizer(proposal, to_mat(xstar), p, inner_shape[-1] // 2, n_modes=6)


@pytest.fixture
def stab_logger(monkeypatch):
    """Installs a RecordingStabLogger as config.logger for the duration of a test."""
    rec = RecordingStabLogger()
    monkeypatch.setattr(config, "logger", rec, raising=False)
    return rec


def test_preconditioner_reflects_inner_window_only(stab_logger):
    """The reflection changes exactly the inner Jacobian window of the proposal and leaves the rest untouched."""
    nk, nb, niv, niv_jac = 4, 2, 10, 4
    stab = _build_window_stabilizer((nk, nb, nb, 2 * niv_jac), p=0.3, unstable=True)
    assert stab.n_unstable >= 1
    rng = np.random.default_rng(1)
    full = (nk, nb, nb, 2 * niv)
    new_mat = rng.standard_normal(full) + 1j * rng.standard_normal(full)
    old_mat = rng.standard_normal(full) + 1j * rng.standard_normal(full)
    sig_new = FakeSelfEnergyWindow(new_mat.copy(), niv)
    sig_old = FakeSelfEnergyWindow(old_mat.copy(), niv)
    sl = slice(niv - niv_jac, niv + niv_jac)
    expected_inner = stab.reflect_proposal(new_mat[..., sl], old_mat[..., sl])

    out = nonlocal_sde.apply_modified_preconditioner(sig_new, sig_old, stab)

    assert np.allclose(out.mat[..., sl], expected_inner)
    assert not np.allclose(out.mat[..., sl], new_mat[..., sl])
    assert np.allclose(out.mat[..., : niv - niv_jac], new_mat[..., : niv - niv_jac])
    assert np.allclose(out.mat[..., niv + niv_jac :], new_mat[..., niv + niv_jac :])
    assert sig_new.compress_calls == 1 and sig_old.compress_calls == 1


def test_preconditioner_is_identity_when_no_unstable_modes(stab_logger):
    """With an empty projector the whole proposal passes through unchanged."""
    nk, nb, niv, niv_jac = 4, 2, 10, 4
    stab = _build_window_stabilizer((nk, nb, nb, 2 * niv_jac), p=0.3, unstable=False)
    assert stab.n_unstable == 0
    rng = np.random.default_rng(2)
    full = (nk, nb, nb, 2 * niv)
    new_mat = rng.standard_normal(full) + 1j * rng.standard_normal(full)
    old_mat = rng.standard_normal(full) + 1j * rng.standard_normal(full)

    out = nonlocal_sde.apply_modified_preconditioner(
        FakeSelfEnergyWindow(new_mat.copy(), niv), FakeSelfEnergyWindow(old_mat.copy(), niv), stab
    )
    assert np.allclose(out.mat, new_mat)


def test_preconditioner_preserves_dtype_and_shape(stab_logger):
    """The reflected window is written back in the proposal's own dtype (no silent upcast of complex64)."""
    nk, nb, niv, niv_jac = 4, 2, 10, 4
    stab = _build_window_stabilizer((nk, nb, nb, 2 * niv_jac), p=0.3, unstable=True)
    rng = np.random.default_rng(3)
    full = (nk, nb, nb, 2 * niv)
    new_mat = (rng.standard_normal(full) + 1j * rng.standard_normal(full)).astype(np.complex64)
    old_mat = (rng.standard_normal(full) + 1j * rng.standard_normal(full)).astype(np.complex64)

    out = nonlocal_sde.apply_modified_preconditioner(
        FakeSelfEnergyWindow(new_mat.copy(), niv), FakeSelfEnergyWindow(old_mat.copy(), niv), stab
    )
    assert out.mat.dtype == np.complex64
    assert out.mat.shape == full


def test_preconditioner_broadcasts_local_old_to_full_bz(stab_logger):
    """A local (nq=1) previous iterate is broadcast across the BZ before reflecting against the full-BZ proposal."""
    nk, nb, niv, niv_jac = 8, 2, 10, 4
    stab = _build_window_stabilizer((nk, nb, nb, 2 * niv_jac), p=0.3, unstable=True)
    assert stab.n_unstable >= 1
    rng = np.random.default_rng(7)
    full = (nk, nb, nb, 2 * niv)
    new_mat = rng.standard_normal(full) + 1j * rng.standard_normal(full)
    old_mat = rng.standard_normal((1, nb, nb, 2 * niv)) + 1j * rng.standard_normal((1, nb, nb, 2 * niv))

    out = nonlocal_sde.apply_modified_preconditioner(
        FakeSelfEnergyWindow(new_mat.copy(), niv), FakeSelfEnergyWindow(old_mat.copy(), niv), stab
    )

    sl = slice(niv - niv_jac, niv + niv_jac)
    old_bcast = np.broadcast_to(old_mat[..., sl], (nk, nb, nb, 2 * niv_jac))
    expected_inner = stab.reflect_proposal(new_mat[..., sl], old_bcast)
    assert out.mat.shape == full
    assert np.allclose(out.mat[..., sl], expected_inner)
    assert np.allclose(out.mat[..., : niv - niv_jac], new_mat[..., : niv - niv_jac])


def test_preconditioner_logs_n_unstable(stab_logger):
    """Applying the reflection emits exactly one info line stating the unstable-subspace dimension."""
    nk, nb, niv, niv_jac = 4, 2, 10, 4
    stab = _build_window_stabilizer((nk, nb, nb, 2 * niv_jac), p=0.3, unstable=True)
    z = np.zeros((nk, nb, nb, 2 * niv), dtype=np.complex128)
    nonlocal_sde.apply_modified_preconditioner(
        FakeSelfEnergyWindow(z.copy(), niv), FakeSelfEnergyWindow(z.copy(), niv), stab
    )
    assert stab_logger.calls["info"] == 1


def test_suppressed_logging_silences_info_debug_memory(stab_logger):
    """Inside the context, info/debug/memory logging is a no-op."""
    with nonlocal_sde._suppressed_logging():
        config.logger.info("x")
        config.logger.debug("x")
        config.logger.log_memory_usage("x")
    assert stab_logger.calls["info"] == 0
    assert stab_logger.calls["debug"] == 0
    assert stab_logger.calls["log_memory_usage"] == 0


def test_suppressed_logging_preserves_warning(stab_logger):
    """Warnings stay audible inside the suppression context."""
    with nonlocal_sde._suppressed_logging():
        config.logger.warning("kept")
    assert stab_logger.calls["warning"] == 1


def test_suppressed_logging_restores_after_block(stab_logger):
    """The original logger methods are restored once the context exits."""
    with nonlocal_sde._suppressed_logging():
        config.logger.info("suppressed")
    config.logger.info("counted")
    config.logger.debug("counted")
    assert stab_logger.calls["info"] == 1
    assert stab_logger.calls["debug"] == 1


def test_suppressed_logging_restores_on_exception(stab_logger):
    """The logger is restored even when the suppressed block raises."""
    with pytest.raises(RuntimeError):
        with nonlocal_sde._suppressed_logging():
            raise RuntimeError("boom")
    config.logger.info("after")
    assert stab_logger.calls["info"] == 1


def _run_probe(residuals, probe_iters=3):
    """Drives the trigger detector over a residual sequence; returns the 1-based trigger iteration or None."""
    best, growth, stall = float("inf"), 0, 0
    for i, r in enumerate(residuals, start=1):
        best, growth, stall, trig = nonlocal_sde._update_stabilizer_probe(r, best, growth, stall, probe_iters)
        if trig:
            return i
    return None


def test_probe_never_triggers_while_residual_decreases():
    """A contracting (converging) run must never arm the stabilizer."""
    assert _run_probe([1e-1, 3e-2, 8e-3, 2e-3, 5e-4, 1e-4], probe_iters=3) is None


def test_probe_triggers_on_sustained_divergence():
    """Sustained residual growth (a factor 3 above the best for probe_iters iterations) arms the stabilizer."""
    assert _run_probe([1e-1, 2e-1, 4e-1, 8e-1, 1.6], probe_iters=3) == 5


def test_probe_short_plateau_does_not_trigger():
    """A plateau shorter than three probe windows is not enough evidence to arm."""
    assert _run_probe([5e-2] * 9, probe_iters=3) is None


def test_probe_triggers_on_long_plateau_far_above_epsilon():
    """A plateau of three probe windows far above epsilon signals a repelling fixed point and arms."""
    config.self_consistency.epsilon = 1e-4
    assert _run_probe([5e-2] * 10, probe_iters=3) == 10


def test_probe_plateau_near_epsilon_never_triggers():
    """A plateau close to the convergence threshold is a slowly converging run, never a trigger."""
    config.self_consistency.epsilon = 1e-4
    assert _run_probe([5e-4] * 20, probe_iters=3) is None


def test_probe_resets_on_renewed_improvement():
    """An improvement after some stalling resets both counters, so a run that recovers never triggers."""
    assert _run_probe([1e-1, 2e-1, 3e-1, 1e-2, 1e-3, 1e-4], probe_iters=3) is None


def test_probe_growth_must_be_consecutive():
    """Growth iterations interleaved with mere stalls do not accumulate toward the divergence trigger."""
    config.self_consistency.epsilon = 1e-4
    assert _run_probe([1e-1, 4e-1, 1.5e-1, 4e-1, 1.5e-1, 4e-1, 1.5e-1], probe_iters=3) is None


def test_probe_respects_probe_iters_setting():
    """A larger probe window requires more sustained growth iterations before arming."""
    residuals = [1e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1]
    assert _run_probe(residuals, probe_iters=1) == 2
    assert _run_probe(residuals, probe_iters=5) == 6


def _run_watchdog(residuals, arming_residual, probe_iters=3):
    """Drives the watchdog over post-arming residuals; returns ('passed'|'revert', 1-based iteration) or None."""
    count, passed = 0, False
    for i, r in enumerate(residuals, start=1):
        count, passed, revert = nonlocal_sde._update_stabilizer_watchdog(r, arming_residual, count, probe_iters)
        if passed:
            return "passed", i
        if revert:
            return "revert", i
    return None


def test_watchdog_passes_on_improvement():
    """The watchdog ends as soon as the reflected iteration improves on the arming residual."""
    assert _run_watchdog([6e-2, 5.5e-2, 4e-2], arming_residual=5e-2, probe_iters=3) == ("passed", 3)


def test_watchdog_reverts_when_reflection_does_not_help():
    """Without any improvement over the arming level within three probe windows, the reflection is reverted."""
    assert _run_watchdog([6e-2] * 12, arming_residual=5e-2, probe_iters=3) == ("revert", 9)


def test_watchdog_requires_meaningful_improvement():
    """A sub-0.1-percent dip below the arming residual does not count as improvement."""
    assert _run_watchdog([5e-2 * (1.0 - 1e-5)] * 12, arming_residual=5e-2, probe_iters=3) == ("revert", 9)


def test_stabilizer_probe_paused_while_restriction_active():
    """The stall detector must not run while use_chi_phys_restriction is active: the restricted map is a scaffold."""
    config.stabilization.use_chi_phys_restriction = True
    assert nonlocal_sde._stabilizer_probe_active(True, None, False, None) is False
    config.stabilization.use_chi_phys_restriction = False
    assert nonlocal_sde._stabilizer_probe_active(True, None, False, None) is True


def test_stabilizer_probe_inactive_when_disarmed_deployed_or_converged():
    """The stall detector is off when the stabilizer is disarmed, already deployed, or the cycle converged."""
    config.stabilization.use_chi_phys_restriction = False
    assert nonlocal_sde._stabilizer_probe_active(False, None, False, None) is False
    assert nonlocal_sde._stabilizer_probe_active(True, object(), False, None) is False
    assert nonlocal_sde._stabilizer_probe_active(True, None, True, None) is False


def test_mixing_history_cap_uses_most_recent_reset_event():
    """The history cap counts iterations since the later of the restriction release and the stabilizer arming."""
    assert nonlocal_sde._mixing_history_cap(10, None, None) is None
    assert nonlocal_sde._mixing_history_cap(10, 7, None) == 2
    assert nonlocal_sde._mixing_history_cap(10, None, 8) == 1
    assert nonlocal_sde._mixing_history_cap(10, 7, 9) == 0
    assert nonlocal_sde._mixing_history_cap(9, 4, 9) == 0


def _setup_stabilizer_build(monkeypatch, nk=(2, 1, 1), nb=1, niv_core=4, niv=6, unstable=True, sigma_nq=None):
    """
    Prepares config, a window-affine fake proposal map with a known fixed point and the sigma_star whose inner
    window sits exactly on it; returns (sigma_star, build_kwargs) for build_stabilization_projector.
    """
    import dgamore.brillouin_zone as bz
    from types import SimpleNamespace

    p = 0.3
    nk_tot = int(np.prod(nk))
    config.lattice.k_grid = bz.KGrid(nk, symmetries=[])
    config.lattice.hamiltonian = SimpleNamespace(get_ek=lambda k_grid=None: np.zeros((*nk, nb, nb)))
    config.box.niv_core = niv_core
    config.sys.beta = 10.0
    config.sys.n = 1.0
    config.sys.occ, config.sys.occ_k = np.eye(nb), np.zeros((*nk, nb, nb))
    config.self_consistency.mixing = p
    config.stabilization.stabilizer_n_modes = 4
    config.stabilization.max_stabilizer_base_residual = 0.5
    monkeypatch.setattr(config, "logger", RecordingStabLogger())

    niv_jac = max(niv_core // 2, min(15, niv_core))
    sl = slice(niv - niv_jac, niv + niv_jac)
    n_real = 2 * nk_tot * nb * nb * 2 * niv_jac
    rng = np.random.default_rng(0)
    q_mat, _ = np.linalg.qr(rng.standard_normal((n_real, n_real)))
    s = rng.uniform(-0.4, 0.4, n_real)
    if unstable:
        s[0] = (1.30 - (1.0 - p)) / p
    jac = q_mat @ np.diag(s) @ q_mat.T
    b = rng.standard_normal(n_real)
    xstar = np.linalg.solve(np.eye(n_real) - jac, b)
    nc = n_real // 2

    def fake_proposal(sigma_in, *args, **kwargs):
        out = sigma_in.copy().compress_q_dimension()
        win = out.mat[..., sl]
        v = np.concatenate((win.reshape(-1).real, win.reshape(-1).imag))
        v = jac @ v + b
        out.mat[..., sl] = (v[:nc] + 1j * v[nc:]).reshape(win.shape).astype(out.mat.dtype)
        return out

    monkeypatch.setattr(nonlocal_sde, "calculate_sigma_proposal", fake_proposal)
    monkeypatch.setattr(nonlocal_sde, "update_mu", lambda *a, **k: 0.5)
    fill = SimpleNamespace(get_fill_nonlocal=lambda: (1.0, np.eye(nb), np.zeros((*nk, nb, nb))))
    monkeypatch.setattr(nonlocal_sde, "GreensFunction", SimpleNamespace(get_g_full=lambda *a, **k: fill))

    star_nq = nk_tot if sigma_nq is None else sigma_nq
    mat = np.zeros((star_nq, nb, nb, 2 * niv), dtype=np.complex64)
    if star_nq == nk_tot:
        mat[..., sl] = (xstar[:nc] + 1j * xstar[nc:]).reshape((nk_tot, nb, nb, 2 * niv_jac)).astype(np.complex64)
    star_nk = (star_nq, 1, 1)
    sigma_star = SelfEnergy(mat, star_nk, has_compressed_q_dimension=True, beta=10.0)
    sigma_dmft_full = SelfEnergy(mat.copy(), star_nk, has_compressed_q_dimension=True, beta=10.0)

    build_kwargs = dict(
        sigma_star=sigma_star,
        mu_star=0.5,
        u_loc=None,
        v_nonloc=None,
        v_nonloc_full=None,
        sigma_dmft=None,
        sigma_dmft_full=sigma_dmft_full,
        delta_sigma=None,
        my_irr_q_list=None,
        my_full_q_list=None,
        mpi_dist_irrk=None,
        mpi_dist_fullbz=None,
        comm=create_comm_mock(),
    )
    return sigma_star, build_kwargs


def test_build_stabilization_projector_returns_none_for_stable_map(monkeypatch):
    """A stable proposal map yields no projector and the constraint state is restored after the build."""
    _, kwargs = _setup_stabilizer_build(monkeypatch, unstable=False)
    config.sys.mu = 0.123
    out = nonlocal_sde.build_stabilization_projector(**kwargs)
    assert out is None
    assert config.sys.mu == 0.5 and config.sys.n == 1.0


def test_build_stabilization_projector_detects_unstable_mode(monkeypatch):
    """An unstable window map produces a projector with the expected window size and no mixing change."""
    _, kwargs = _setup_stabilizer_build(monkeypatch, unstable=True)
    out = nonlocal_sde.build_stabilization_projector(**kwargs)
    assert out is not None and out.n_unstable >= 1
    assert out.niv_jac == 4
    assert config.self_consistency.mixing == 0.3


def test_build_stabilization_projector_raises_on_kgrid_mismatch(monkeypatch):
    """A non-local warm start on a different k-grid must raise instead of being silently tiled."""
    _, kwargs = _setup_stabilizer_build(monkeypatch, unstable=False, sigma_nq=3)
    with pytest.raises(ValueError):
        nonlocal_sde.build_stabilization_projector(**kwargs)


def test_build_stabilization_projector_tiles_local_start(monkeypatch):
    """A purely local (nq=1) warm start is broadcast to the full BZ and the build runs through the guard."""
    _, kwargs = _setup_stabilizer_build(monkeypatch, unstable=False, sigma_nq=1)
    config.stabilization.max_stabilizer_base_residual = 1e9
    out = nonlocal_sde.build_stabilization_projector(**kwargs)
    assert out is None


def test_build_stabilization_projector_aborts_on_cold_start(monkeypatch):
    """A warm start far from any fixed point trips the base-residual guard by design."""
    _, kwargs = _setup_stabilizer_build(monkeypatch, unstable=False)
    kwargs["sigma_star"].mat[:] = 0.0
    with pytest.raises(nonlocal_sde.jstab.PhysicalSolutionStabilizerError):
        nonlocal_sde.build_stabilization_projector(**kwargs)


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


class _SingleRankDist:
    """Minimal single-rank distributor stand-in for the annealing shift (only .comm.size is read)."""

    def __init__(self):
        from types import SimpleNamespace

        self.comm = SimpleNamespace(size=1)


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
    out = annealer.apply(chi, _SingleRankDist())
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
    out = annealer.apply(chi, _SingleRankDist())
    assert np.array_equal(out.mat, ref)
    assert np.allclose(annealer._gaps["dens"], 0.3, atol=1e-5)


def test_annealer_apply_quiet_probe_skips_measurement():
    """Quiet Jacobian probes apply the current shared mass but never (re-)measure the gap."""
    rng = np.random.default_rng(14)
    o, nq, nw, lam = 2, 2, 2, 0.3
    q_mat = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))[0]
    comp_inv = np.tile((q_mat * np.array([0.4, 0.5, 1.0, 2.0])) @ q_mat.conj().T, (nq, nw, 1, 1))
    chi = _chi_from_compound(np.linalg.inv(comp_inv), o)
    annealer = _seeded_annealer(mass=lam, gaps={"dens": None})
    out = annealer.apply(chi, _SingleRankDist(), measure=False)
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
    assert stab_logger.calls["warning"] == 1


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


def test_stabilizer_probe_paused_while_annealing_active():
    """The stabilizer stall detector must not run while the annealing scaffold shapes the map."""
    config.stabilization.use_chi_phys_restriction = False
    assert nonlocal_sde._stabilizer_probe_active(True, None, False, _seeded_annealer(mass=0.1)) is False
    assert nonlocal_sde._stabilizer_probe_active(True, None, False, _seeded_annealer(mass=0.0)) is True
    assert nonlocal_sde._stabilizer_probe_active(True, None, False, None) is True


def test_mixing_history_cap_includes_anneal_reset():
    """The history cap also counts from the most recent annealing-mass change."""
    assert nonlocal_sde._mixing_history_cap(10, None, None, 9) == 0
    assert nonlocal_sde._mixing_history_cap(10, 7, None, 5) == 2
    assert nonlocal_sde._mixing_history_cap(10, None, None, None) is None


def test_relative_sigma_residual_layout_mismatch_is_normalized():
    """A decompressed previous iterate against a compressed proposal compares matching momenta (regression:
    rank 0's iterate is left decompressed by the save path and raw broadcasting paired wrong momenta)."""
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


def test_probe_triggers_on_nonfinite_residual():
    """Inf/NaN residuals count as divergence evidence instead of silently resetting the growth counter."""
    assert _run_probe([1e-1, np.inf, np.nan, np.nan], probe_iters=3) == 4


def test_select_and_apply_lambda_correction_dispatch():
    """The rank-0 lambda selector dispatches by the band count (single-band scalar vs. multi-orbital matrix
    correction) for both the one-shot and the per-iteration flag, and returns the susceptibility unchanged when
    neither is enabled."""
    sentinel, chi = object(), object()
    config.lambda_correction.perform_lambda_correction = False
    config.stabilization.use_lambda_correction = False
    assert nonlocal_sde._select_and_apply_lambda_correction(chi, quiet=False) is chi

    with mock.patch.object(nonlocal_sde.LambdaCorrection, "perform", return_value=sentinel) as single:
        config.stabilization.use_lambda_correction = True
        config.sys.n_bands = 1
        assert nonlocal_sde._select_and_apply_lambda_correction(chi, quiet=True) is sentinel
        single.assert_called_once_with(chi, quiet=True)

    with mock.patch.object(nonlocal_sde.MultiOrbitalLambdaCorrection, "perform", return_value=sentinel) as multi:
        config.sys.n_bands = 2
        assert nonlocal_sde._select_and_apply_lambda_correction(chi, quiet=True) is sentinel
        multi.assert_called_once_with(chi, quiet=True)

    config.stabilization.use_lambda_correction = False
    config.lambda_correction.perform_lambda_correction = True
    with mock.patch.object(nonlocal_sde.LambdaCorrection, "perform", return_value=sentinel) as single:
        config.sys.n_bands = 1
        assert nonlocal_sde._select_and_apply_lambda_correction(chi, quiet=False) is sentinel
        single.assert_called_once_with(chi, quiet=False)
    with mock.patch.object(nonlocal_sde.MultiOrbitalLambdaCorrection, "perform", return_value=sentinel) as multi:
        config.sys.n_bands = 2
        assert nonlocal_sde._select_and_apply_lambda_correction(chi, quiet=False) is sentinel
        multi.assert_called_once_with(chi, quiet=False)


class _RecordingTextLogger:
    """Logger stub recording full info/warning texts for the loop sequencing tests."""

    def __init__(self):
        self.infos, self.warnings = [], []

    def info(self, msg, *a, **k):
        self.infos.append(str(msg))

    def warning(self, msg, *a, **k):
        self.warnings.append(str(msg))

    def debug(self, *a, **k):
        pass

    def log_memory_usage(self, *a, **k):
        pass


def _setup_self_energy_loop(monkeypatch, tmp_path, proposal_step, max_iter=10, epsilon=1e-3):
    """
    Minimal single-k single-band environment for calculate_self_energy_q: the heavy pipeline is replaced by the
    synthetic per-iteration map ``proposal_step(sigma_in, n_call, annealer)`` and the mu/occupation solves are
    frozen, so the tests exercise exactly the loop sequencing (mixing, convergence gate, scaffold releases and the
    stabilizer arm/watchdog wiring). Returns ``(run, calls, logger)`` with ``run()`` executing the loop and
    ``calls`` the proposal invocation list.
    """
    import dgamore.brillouin_zone as bz
    from types import SimpleNamespace

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
    logger = _RecordingTextLogger()
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

    calls = []

    def fake_proposal(sigma_in, *args, quiet=False, annealer=None, **kwargs):
        calls.append(len(calls) + 1)
        return proposal_step(sigma_in, len(calls), annealer)

    monkeypatch.setattr(nonlocal_sde, "calculate_sigma_proposal", fake_proposal)

    mat = np.full((1, 1, 1, 16), 1.0 + 0.1j, dtype=np.complex64)
    sigma_dmft = SelfEnergy(mat, (1, 1, 1), has_compressed_q_dimension=True, beta=10.0)

    class _VStub:
        def copy(self):
            return self

        def reduce_q(self, q_list):
            return self

    def run():
        return nonlocal_sde.calculate_self_energy_q(create_comm_mock(), None, _VStub(), sigma_dmft, sigma_dmft.copy())

    return run, calls, logger


def test_loop_annealing_runs_pure_phase_after_mass_snaps_to_zero(monkeypatch, tmp_path):
    """When the annealing mass snaps to zero on a converged phase the loop must not break on that same iteration:
    the converged verdict belongs to the scaffolded map, so at least one pure (mass-zero) iteration must follow
    before the run counts as converged. Schedule on the identity map: iteration 1 measures a poled gap (mass arms
    at 3e-2), iterations 2 and 3 converge and halve the mass to zero, iteration 4 converges the pure map."""
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
    """The per-iteration lambda-correction scaffold releases on the first relaxed convergence (flag disabled, no
    break) and the loop then converges the pure map at full epsilon on a later iteration."""

    def step(sigma_in, n_call, annealer):
        return sigma_in.copy()

    run, calls, logger = _setup_self_energy_loop(monkeypatch, tmp_path, step)
    config.stabilization.use_lambda_correction = True
    run()

    assert len(calls) == 3
    assert config.stabilization.use_lambda_correction is False
    assert any("Self-consistency with the lambda correction reached" in m for m in logger.infos)


def test_loop_one_shot_lambda_correction_never_fires_release(monkeypatch, tmp_path):
    """The one-shot perform_lambda_correction neither relaxes epsilon nor triggers the release branch: the loop
    converges at full epsilon on the identity map and the flag stays enabled."""

    def step(sigma_in, n_call, annealer):
        return sigma_in.copy()

    run, calls, logger = _setup_self_energy_loop(monkeypatch, tmp_path, step)
    config.lambda_correction.perform_lambda_correction = True
    run()

    assert len(calls) == 2
    assert config.lambda_correction.perform_lambda_correction is True
    assert not any("lambda correction reached" in m for m in logger.infos)


def test_loop_stabilizer_plateau_triggers_build_and_watchdog_reverts(monkeypatch, tmp_path):
    """A far-above-epsilon residual plateau (constant relative step on the map S(x) = 1.1 x) triggers exactly one
    projector build at the warm-start sigma after three probe windows; the deployed (here inert) reflection fails
    to improve the arming residual, so the watchdog reverts it after three more windows and plain mixing resumes."""

    def step(sigma_in, n_call, annealer):
        out = sigma_in.copy()
        out.mat = out.mat * 1.1
        return out

    run, calls, logger = _setup_self_energy_loop(monkeypatch, tmp_path, step, max_iter=22, epsilon=1e-8)
    config.stabilization.use_jacobian_stabilization = True
    config.stabilization.stabilizer_probe_iters = 3
    build_calls = []
    monkeypatch.setattr(
        nonlocal_sde,
        "build_stabilization_projector",
        lambda sigma_star, *a, **k: build_calls.append(sigma_star.mat.copy()) or mock.MagicMock(n_unstable=1),
    )
    precond_calls = []
    monkeypatch.setattr(
        nonlocal_sde, "apply_modified_preconditioner", lambda new, old, stab: precond_calls.append(1) or new
    )
    run()

    assert len(calls) == 22
    assert len(build_calls) == 1 and np.allclose(build_calls[0], 1.0 + 0.1j, atol=1e-6)
    assert len(precond_calls) == 9
    assert any("did not improve its arming residual" in w for w in logger.warnings)


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
    """The static gap is read from the w=0 slice - index niw for full-range, index 0 for half-range objects - on a
    frequency-varying susceptibility whose non-static slices carry a much deeper (pole-like) eigenvalue."""
    o, nq = 2, 2
    healthy = np.diag([0.7, 1.0, 2.0, 3.0]).astype(np.complex128)
    poled = np.diag([-5.0, 1.0, 2.0, 3.0]).astype(np.complex128)
    comp_full = np.tile(np.linalg.inv(np.stack([poled, healthy, poled])), (nq, 1, 1, 1))
    chi_full = _chi_from_compound(comp_full, o)
    chi_full._full_niw_range = True
    assert np.allclose(nonlocal_sde.LambdaAnnealer._static_gap(chi_full, _SingleRankDist()), 0.7, atol=1e-5)
    comp_half = np.tile(np.linalg.inv(np.stack([healthy, poled])), (nq, 1, 1, 1))
    chi_half = _chi_from_compound(comp_half, o)
    assert np.allclose(nonlocal_sde.LambdaAnnealer._static_gap(chi_half, _SingleRankDist()), 0.7, atol=1e-5)


def test_annealer_static_gap_reduces_min_across_ranks():
    """The measured static gap is the MPI.MIN of the per-rank q-slice minima, identical on every rank."""
    from types import SimpleNamespace

    from tests.conftest import run_parallel

    def fn(comm, rank):
        eigs = np.array([0.5, 1.0, 2.0, 3.0]) if rank == 0 else np.array([-0.7, 1.0, 2.0, 3.0])
        comp = np.tile(np.linalg.inv(np.diag(eigs).astype(np.complex128)), (2, 2, 1, 1))
        chi = _chi_from_compound(comp, 2)
        return nonlocal_sde.LambdaAnnealer._static_gap(chi, SimpleNamespace(comm=comm))

    _, results = run_parallel(2, fn)
    assert np.allclose(results[0], -0.7, atol=1e-5)
    assert np.allclose(results[1], -0.7, atol=1e-5)

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import itertools
import os
from unittest import mock

import numpy as np

import dgamore.config as config
import dgamore.nonlocal_sde as nonlocal_sde
from dgamore.greens_function import GreensFunction
from dgamore.hamiltonian import Hamiltonian
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
    config.lattice.nq = nk
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
    config.lattice.q_grid = config.lattice.k_grid
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
    q_list = config.lattice.q_grid.get_q_list()

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
    config.self_consistency.restrict_chi_phys = True
    assert np.allclose(nonlocal_sde._effective_epsilon(), 1e-4, atol=1e-18)
    config.self_consistency.restrict_chi_phys = False
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
    config.lattice.q_grid = config.lattice.k_grid
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

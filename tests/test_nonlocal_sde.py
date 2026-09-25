# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import contextlib
import fnmatch
import gc
import os
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest

import dgamore.brillouin_zone as bz
import dgamore.config as config
import dgamore.memory_estimator as memory_estimator
import dgamore.mpi_utils as mpi_utils
import dgamore.nonlocal_sde as nonlocal_sde
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction
from dgamore.hamiltonian import Hamiltonian
from dgamore.interaction import Interaction, LocalInteraction
from dgamore.jacobian_stabilization import (
    JACOBIAN_FILE,
    JacobianTracker,
    load_spectrum,
    TRACKER_PAIRS,
    to_mat,
    to_vec,
)
from dgamore.local_four_point import LocalFourPoint
from dgamore.local_sde import get_local_hartree_fock
from dgamore.n_point_base import DTYPE, SpinChannel
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
from dgamore.matsubara_frequencies import MFHelper
from dgamore.self_energy import SelfEnergy
from tests.conftest import FAKE_MPI, create_comm_mock, patch_mini_pole_with_exact_single_pole_fit, run_parallel

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


def _previous_run_folder(tmp_path, monkeypatch, names, beta=25.0):
    """Creates empty predecessor files and a np.load stub returning a constant self-energy tagged by the file name."""
    config.logger = MagicMock()
    config.sys.beta = beta
    config.box.niv_core = 2
    config.lattice.nk = (2, 2, 1)
    config.lattice.k_grid = bz.KGrid(config.lattice.nk, symmetries=bz.two_dimensional_square_symmetries())
    config.self_consistency.previous_sc_path = str(tmp_path)
    for name in names:
        (tmp_path / name).touch()
    loaded = []

    def fake_load(path, *args, **kwargs):
        loaded.append(os.path.basename(path))
        return np.full((2, 2, 1, 1, 1, 8), float(len(loaded)), dtype=complex)

    monkeypatch.setattr(np, "load", fake_load)
    return loaded


def test_get_starting_sigma_picks_the_interpolated_file_closest_in_beta_and_counts_the_raw_iterates(
    monkeypatch, tmp_path
):
    """With use_interpolated_sigma the interpolated file closest in beta is loaded and the iterate count is kept."""
    names = ("sigma_dga_iteration_2.npy", "sigma_dga_iteration_7.npy", "sigma_dga_interpolated_beta20.0_niv4.npy")
    loaded = _previous_run_folder(tmp_path, monkeypatch, names + ("sigma_dga_interpolated_beta25.0_niv4.npy",))
    config.self_consistency.use_interpolated_sigma = True

    sigma, starting_iter = nonlocal_sde.get_starting_sigma(MagicMock())

    assert loaded == ["sigma_dga_interpolated_beta25.0_niv4.npy"]
    assert starting_iter == 7
    assert sigma.mat.shape == (2, 2, 1, 1, 1, 4)


def test_get_starting_sigma_without_interpolation_loads_the_highest_raw_iterate(monkeypatch, tmp_path):
    """Without use_interpolated_sigma the raw iterate with the highest number is loaded."""
    names = ("sigma_dga_iteration_2.npy", "sigma_dga_iteration_7.npy", "sigma_dga_interpolated_beta25.0_niv4.npy")
    loaded = _previous_run_folder(tmp_path, monkeypatch, names)
    config.self_consistency.use_interpolated_sigma = False

    _, starting_iter = nonlocal_sde.get_starting_sigma(MagicMock())

    assert loaded == ["sigma_dga_iteration_7.npy"]
    assert starting_iter == 7


def test_get_starting_sigma_warns_when_use_interpolated_sigma_has_no_previous_path():
    """With use_interpolated_sigma but an empty previous_sc_path the default self-energy is returned with a warning."""
    config.logger = MagicMock()
    config.self_consistency.previous_sc_path = ""
    config.self_consistency.use_interpolated_sigma = True
    default = MagicMock()

    sigma, starting_iter = nonlocal_sde.get_starting_sigma(default)

    assert sigma is default
    assert starting_iter == 0
    config.logger.warning.assert_called_once()


def test_get_starting_sigma_falls_back_to_the_default_and_warns_without_an_interpolated_file(monkeypatch, tmp_path):
    """With use_interpolated_sigma but no interpolated file the default self-energy is returned with a warning."""
    loaded = _previous_run_folder(tmp_path, monkeypatch, ("sigma_dga_iteration_3.npy",))
    config.self_consistency.use_interpolated_sigma = True
    default = MagicMock()

    sigma, starting_iter = nonlocal_sde.get_starting_sigma(default)

    assert sigma is default
    assert starting_iter == 0
    assert loaded == []
    config.logger.warning.assert_called_once()


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

    uq = (u_loc + v_nonloc).mat.reshape(nk_tot, nb, nb, nb, nb)
    q_list = np.array([np.unravel_index(i, nk) for i in range(nk_tot)])
    fock_ref = np.zeros((nk_tot, nb, nb), dtype=np.complex64)
    for d in range(nb):
        for c in range(nb):
            occ_qk = np.array([np.roll(occ_k[..., c, d], tuple(q), axis=(0, 1, 2)) for q in q_list]).reshape(
                nk_tot, nk_tot
            )
            # -sum_q (U + V^q)_{1ab2} n^{k-q}_{ab}: external orbitals on the outer slots of uq[q, o1, o2, o3, o4]
            fock_ref += np.einsum("qab,qk->kab", uq[:, :, c, d, :], occ_qk, optimize=True)
    fock_ref *= -1.0 / nk_tot

    _, fock = get_hartree_fock(u_loc, v_nonloc)
    assert np.allclose(fock[..., 0], fock_ref, atol=1e-4)


def test_nonlocal_hartree_fock_matches_the_local_one_for_a_generic_tensor():
    """For V = 0 and a generic (non-Kanamori) local tensor the non-local Hartree-Fock equals the local SDE's."""
    nb, nk = 2, (2, 2, 1)
    nk_tot = int(np.prod(nk))
    config.lattice.nk = nk
    config.sys.n_bands = nb
    rng = np.random.default_rng(5)
    occ = rng.standard_normal((nb, nb))
    occ = occ + occ.T
    config.sys.occ = occ
    config.sys.occ_k = np.broadcast_to(occ, nk + (nb, nb)).copy()
    u_loc = LocalInteraction(rng.standard_normal((nb, nb, nb, nb)), SpinChannel.NONE)
    v_zero = Interaction(np.zeros((nk_tot, nb, nb, nb, nb)), SpinChannel.NONE, nk, has_compressed_q_dimension=True)
    hartree, fock = get_hartree_fock(u_loc, v_zero)
    assert np.allclose((hartree + fock)[0, ..., 0], get_local_hartree_fock(u_loc, occ), atol=1e-6)


def test_hartree_fock_of_explicit_occupations_equals_the_config_path_and_is_linear():
    """Explicit occupations give the config-path term, and the term is linear in the occupations it is handed."""
    nb, nk = 2, (3, 2, 1)
    nk_tot = int(np.prod(nk))
    config.lattice.nk = nk
    config.sys.n_bands = nb
    rng = np.random.default_rng(6)
    u_loc = LocalInteraction(rng.standard_normal((nb, nb, nb, nb)), SpinChannel.NONE)
    v_nonloc = Interaction(rng.standard_normal((nk_tot, nb, nb, nb, nb)), SpinChannel.NONE, nk, True)
    occ_k_a, occ_k_b = rng.standard_normal((2, *nk, nb, nb)) + 1j * rng.standard_normal((2, *nk, nb, nb))
    config.sys.occ_k, config.sys.occ = occ_k_a, occ_k_a.mean(axis=(0, 1, 2))
    via_config = sum(get_hartree_fock(u_loc, v_nonloc))
    explicit = sum(get_hartree_fock(u_loc, v_nonloc, occ=occ_k_a.mean(axis=(0, 1, 2)), occ_k=occ_k_a))
    both = sum(get_hartree_fock(u_loc, v_nonloc, occ=(occ_k_a + occ_k_b).mean(axis=(0, 1, 2)), occ_k=occ_k_a + occ_k_b))
    only_b = sum(get_hartree_fock(u_loc, v_nonloc, occ=occ_k_b.mean(axis=(0, 1, 2)), occ_k=occ_k_b))
    assert np.array_equal(explicit, via_config)
    assert np.allclose(both, explicit + only_b, atol=1e-5)


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


def test_ornstein_zernike_fit_fits_the_static_slice_of_the_unfolded_susceptibility(monkeypatch, tmp_path):
    """Selecting the static slice before the unfold hands curve_fit the same data as unfolding every slice first."""
    config.sys.n_bands = 2
    config.output.output_path = str(tmp_path)
    config.lattice.k_grid = bz.KGrid((4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
    rng = np.random.default_rng(5)
    shape = (config.lattice.k_grid.nk_irr, 2, 2, 2, 2, 3)
    chi = FourPoint(
        (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64),
        SpinChannel.MAGN,
        (4, 4, 1),
        1,
        0,
        full_niw_range=False,
        has_compressed_q_dimension=True,
    )
    fitted = []
    monkeypatch.setattr(
        nonlocal_sde.opt, "curve_fit", lambda f, x, y, p0: fitted.append(y.copy()) or (np.array([1.0, 2.0]), None)
    )

    perform_ornstein_zernike_fit(chi)

    ref = chi.copy().map_to_full_bz(config.lattice.k_grid).to_half_niw_range().take_first_wn().mat.real
    expected = [ref[..., idx[0], idx[1], idx[2], idx[3]].flatten() for idx in np.ndindex((2, 2, 2, 2))]
    assert len(fitted) == 16
    assert all(np.array_equal(f, e) for f, e in zip(fitted, expected))


def test_rpa_susceptibility_dresses_the_frequency_summed_bubble_with_the_channel_interaction(monkeypatch):
    """For one band the saved RPA susceptibility is chi0 / (1 + (U_r + V_r) chi0) of the frequency-summed bubble."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
    saved = {}
    monkeypatch.setattr(FourPoint, "save", lambda self, name, output_dir: saved.__setitem__(name, self.mat.copy()))
    rng = np.random.default_rng(12)
    nq, nw = 3, 2
    chi0 = FourPoint(
        (0.2 + rng.random((nq, 1, 1, 1, 1, nw)) + 0.1j * rng.standard_normal((nq, 1, 1, 1, 1, nw))),
        SpinChannel.NONE,
        (nq, 1, 1),
        1,
        0,
        False,
        True,
        True,
    )
    u_loc = LocalInteraction(np.full((1, 1, 1, 1), 1.3), SpinChannel.NONE)
    v_nonloc = Interaction(0.2 * rng.standard_normal((nq, 1, 1, 1, 1)), SpinChannel.NONE, (nq, 1, 1), True)
    dist = mpi_utils.MpiDistributor(ntasks=nq, comm=create_comm_mock())
    nonlocal_sde.calculate_and_save_chi_q_r_rpa(chi0, u_loc, v_nonloc, dist)
    for channel in (SpinChannel.DENS, SpinChannel.MAGN):
        u = u_loc.as_channel(channel).mat.ravel()[0] + v_nonloc.as_channel(channel).mat.reshape(nq, 1)
        expected = chi0.mat.reshape(nq, nw) / (1.0 + u * chi0.mat.reshape(nq, nw))
        assert np.allclose(saved[f"chi_rpa_q_{channel.value}"].reshape(nq, nw), expected, atol=1e-6)


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


def test_build_giwk_full_keeps_no_self_energy_so_its_copies_never_duplicate_sigma():
    """The proposal's Green's function holds the dispersion but no self-energy, so a frequency cut copies no Sigma."""
    sigma, ek = _tiny_sigma_and_ek(niv=8)
    giwk, win, node_comm = _build_giwk_full(create_comm_mock(), sigma, 0.3, ek, 10.0)
    cut = giwk.cut_niv(4)
    assert giwk._sigma is None and cut._sigma is None and np.array_equal(cut._ek, ek)
    _release_shared_giwk(win, node_comm)


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


def _bse_assembly_reference(gamma, gchi0_q_inv, u_loc):
    """Evaluates the pre-fusion two-block expression (chi0^-1 + Gamma/beta^2) - U_r/beta^2 via the object API."""
    beta = config.sys.beta
    return (gchi0_q_inv.copy() + 1.0 / beta**2 * gamma.copy()) - 1.0 / beta**2 * u_loc.as_channel(gamma.channel)


def test_create_inverse_auxiliary_chi_r_q_matches_two_block_expression():
    """The fused single-block BSE assembly is bit-equal to the former add/extend/subtract chain, inputs untouched."""
    rng = np.random.default_rng(21)
    gamma, gchi0_q_inv, u_loc, v_nonloc = _bse_assembly_inputs(rng)
    gamma_before, chi0_before = gamma.mat.copy(), gchi0_q_inv.mat.copy()
    ref = _bse_assembly_reference(gamma, gchi0_q_inv, u_loc)
    u_r = u_loc.as_channel(gamma.channel)
    fused = nonlocal_sde.create_inverse_auxiliary_chi_r_q(gamma, gchi0_q_inv, u_r)
    assert np.array_equal(fused.mat, ref.mat)
    assert fused.channel == SpinChannel.DENS
    assert not fused.full_niw_range and fused.num_vn_dimensions == 2
    assert np.array_equal(gamma.mat, gamma_before) and gamma.full_niw_range
    assert np.array_equal(gchi0_q_inv.mat, chi0_before) and gchi0_q_inv.num_vn_dimensions == 1


def test_create_inverse_auxiliary_chi_r_q_accepts_half_niw_gamma():
    """A gamma already in the half bosonic range assembles identically to its full-range twin."""
    rng = np.random.default_rng(22)
    gamma, gchi0_q_inv, u_loc, _ = _bse_assembly_inputs(rng)
    u_r = u_loc.as_channel(gamma.channel)
    full = nonlocal_sde.create_inverse_auxiliary_chi_r_q(gamma, gchi0_q_inv, u_r)
    half = nonlocal_sde.create_inverse_auxiliary_chi_r_q(gamma.copy().to_half_niw_range(), gchi0_q_inv, u_r)
    assert np.array_equal(full.mat, half.mat)


def test_create_auxiliary_chi_r_q_matches_explicit_reference():
    """The full-inversion variant reproduces the explicit two-block expression inverted in compound space."""
    rng = np.random.default_rng(24)
    gamma, gchi0_q_inv, u_loc, _ = _bse_assembly_inputs(rng)
    ref = _bse_assembly_reference(gamma, gchi0_q_inv, u_loc).invert(False)
    out = nonlocal_sde.create_auxiliary_chi_r_q(gamma, gchi0_q_inv, u_loc)
    assert np.allclose(out.mat, ref.mat, atol=1e-10)


def test_auxiliary_chi_r_q_chain_resums_to_the_ladder_with_the_nonlocal_rung():
    """[(sum chi*)^-1 + U_r + V_r]^-1 equals the summed BSE with Gamma^w_r + V^q_r (the rung enters via chi_phys)."""
    rng = np.random.default_rng(25)
    gamma, gchi0_q_inv, u_loc, v_nonloc = _bse_assembly_inputs(rng)
    beta, channel = config.sys.beta, gamma.channel
    chi_star_sum = nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_q_inv, u_loc).sum_over_all_vn(beta)
    # zero shell correction (same sum added and subtracted); direct = chi0^-1 + (Gamma^w + V_r)/beta^2 via -V_r
    chain = nonlocal_sde.create_generalized_chi_q_with_shell_correction(
        chi_star_sum, chi_star_sum, chi_star_sum, u_loc, v_nonloc
    )
    direct = (
        nonlocal_sde.create_inverse_auxiliary_chi_r_q(gamma, gchi0_q_inv, -v_nonloc.as_channel(channel))
        .invert(False)
        .sum_over_all_vn(beta)
    )
    assert np.allclose(chain.mat, direct.mat, atol=1e-8)


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


def test_vrg_is_the_inverse_bubble_applied_to_the_last_frequency_sum_of_chi_aux():
    """The three-leg vertex equals the inverse bubble contracted with the last-frequency sum of chi*."""
    o, nqi, nw, n2, beta = 2, 3, 3, 4, 12.5
    config.sys.beta = beta
    rng = np.random.default_rng(11)
    shape = (nqi, o, o, o, o, nw, n2, n2)
    chi_star = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    chi0_inv = rng.standard_normal(shape[:-1]) + 1j * rng.standard_normal(shape[:-1])

    sum_last = FourPoint(chi_star.sum(axis=-1) / beta, SpinChannel.DENS, (nqi, 1, 1), 1, 1, False, True, True)
    chi0_inv_fp = FourPoint(chi0_inv.copy(), SpinChannel.NONE, (nqi, 1, 1), 1, 1, False, True, True)

    vrg_left = nonlocal_sde.create_vrg_r_q(sum_last.copy(), chi0_inv_fp)
    ref_left = np.einsum("qabefwv,qfecdwv->qabcdwv", chi0_inv, chi_star.sum(axis=-1), optimize=True)
    assert np.allclose(vrg_left.mat, ref_left, atol=1e-10)


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
    """A static chi block with a negative eigenvalue gets its inverse eigenvalue floored, positive pairs kept."""
    rng = np.random.default_rng(4)
    o, nq, floor = 2, 2, 1e-4
    q_mat = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))[0]
    eigs_in = np.array([-5.0, 0.5, 1.0, 2.0])
    comp = np.tile((q_mat * eigs_in) @ q_mat.conj().T, (nq, 1, 1, 1))
    chi = _chi_from_compound(comp, o)
    out, n_floored = nonlocal_sde.restrict_chi_phys_to_positive_eigenvalues(chi, floor=floor)
    assert n_floored == nq
    out_comp = _compound_of(out, o).astype(np.complex128)
    ev = np.sort(np.linalg.eigvalsh(0.5 * (out_comp + np.conj(np.transpose(out_comp, (0, 1, 3, 2))))), axis=-1)
    expected = np.sort(np.array([1.0 / floor, 0.5, 1.0, 2.0]))
    assert np.allclose(ev / expected[None, None, :], 1.0, atol=1e-3)
    proj_pos = (q_mat[:, 1:] * eigs_in[1:]) @ q_mat[:, 1:].conj().T
    proj_out = out_comp[0, 0] - (out_comp[0, 0] @ q_mat[:, :1]) @ q_mat[:, :1].conj().T
    assert np.allclose(proj_out @ q_mat[:, 1:], proj_pos @ q_mat[:, 1:], atol=1e-3)


def test_restrict_chi_phys_leaves_a_healthy_input_bit_identical():
    """A positive-definite static chi and smaller finite-frequency blocks pass through bit for bit."""
    rng = np.random.default_rng(5)
    o, nq, nw = 2, 2, 3
    a = rng.standard_normal((nq, 1, 4, 4)) + 1j * rng.standard_normal((nq, 1, 4, 4))
    static = a @ np.conj(np.transpose(a, (0, 1, 3, 2))) + 0.1 * np.eye(4)
    comp = np.concatenate([static / (1.0 + w) for w in range(nw)], axis=1)
    comp[:, 2] -= 0.05 * np.eye(4) * np.linalg.eigvalsh(static[:, 0])[:, :1, None]
    assert (comp.real < 0).any()
    chi = _chi_from_compound(comp, o)
    out, n_restricted = nonlocal_sde.restrict_chi_phys_to_positive_eigenvalues(chi, floor=1e-4)
    assert n_restricted == 0
    assert np.array_equal(out.mat, chi.mat)


def test_restrict_chi_phys_matches_scalar_clamp_for_single_band():
    """For a single band the eigenvalue floor reproduces the elementwise clamp: negative chi -> 1/floor, else kept."""
    floor = 1e-4
    mat = np.array([-0.5, 0.3], dtype=complex).reshape(2, 1, 1, 1, 1, 1)
    chi = FourPoint(mat.copy(), SpinChannel.DENS, (2, 1, 1), 1, 0, False, True, True)
    out, n_floored = nonlocal_sde.restrict_chi_phys_to_positive_eigenvalues(chi, floor=floor)
    assert n_floored == 1
    assert np.allclose(out.mat.ravel() / np.array([1.0 / floor, 0.3]), 1.0, atol=1e-3)


def test_restrict_chi_phys_clips_a_first_frequency_pole_to_the_static_bound():
    """A finite-frequency chi outside [-chi(q, 0), chi(q, 0)] is clipped to it, real and imaginary part alike."""
    comp = np.array(
        [[[[0.10]], [[7.0 + 0.2j]], [[-0.005]]], [[[0.05]], [[-6.0 - 0.1j]], [[0.02]]]], dtype=np.complex128
    )
    chi = _chi_from_compound(comp, 1)
    out, n_restricted = nonlocal_sde.restrict_chi_phys_to_positive_eigenvalues(chi, floor=1e-4)
    expected = np.array([[0.10, 0.10 + 0.10j, -0.005], [0.05, -0.05 - 0.05j, 0.02]])
    assert n_restricted == 4
    assert np.allclose(_compound_of(out, 1)[..., 0, 0], expected, atol=1e-6)


def test_restrict_chi_phys_clips_finite_frequency_eigenvalues_on_their_eigenvectors():
    """Multi-orbital finite-frequency eigenvalues are clipped to the largest static one, eigenvectors unchanged."""
    rng = np.random.default_rng(7)
    q_mat = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))[0]
    static = np.diag([1.0, 0.6, 0.3, 0.2]).astype(np.complex128)
    finite = (q_mat * np.array([8.0, 0.3, -0.2, -9.0])) @ q_mat.conj().T
    chi = _chi_from_compound(np.stack([static, finite])[None], 2)
    out, n_restricted = nonlocal_sde.restrict_chi_phys_to_positive_eigenvalues(chi, floor=1e-4)
    expected = (q_mat * np.array([1.0, 0.3, -0.2, -1.0])) @ q_mat.conj().T
    assert n_restricted == 2
    assert np.allclose(_compound_of(out, 2)[0, 1], expected, atol=1e-5)
    assert np.allclose(_compound_of(out, 2)[0, 0], static, atol=1e-6)


def test_restrict_chi_phys_returns_the_half_range_of_a_full_range_input():
    """A full-range input is restricted on its non-negative frequencies and returned in the half range."""
    comp = np.array([[[[0.05]], [[7.0]], [[0.10]], [[7.0]], [[0.05]]]], dtype=np.complex128)
    chi = _chi_from_compound(comp, 1)
    chi._full_niw_range = True
    out, n_restricted = nonlocal_sde.restrict_chi_phys_to_positive_eigenvalues(chi, floor=1e-4)
    assert not out.full_niw_range and n_restricted == 1
    assert np.allclose(_compound_of(out, 1)[0, :, 0, 0], [0.10, 0.10, 0.05], atol=1e-6)


def test_restrict_chi_phys_pins_a_crossed_static_mode_at_twice_the_largest_healthy_one():
    """Without an explicit floor a crossed static mode is pinned at twice the channel's largest healthy value."""
    comp = np.array([[[[0.5]]], [[[0.2]]], [[[-0.3]]]], dtype=np.complex128)
    out, n_restricted = nonlocal_sde.restrict_chi_phys_to_positive_eigenvalues(_chi_from_compound(comp, 1))
    assert n_restricted == 1
    assert np.allclose(_compound_of(out, 1)[:, 0, 0, 0], [0.5, 0.2, 1.0], atol=1e-6)


def test_restrict_chi_phys_takes_the_largest_healthy_value_over_all_ranks():
    """The healthy maximum is reduced over the ranks, so every rank pins its crossed modes at the same value."""
    comm = MagicMock()
    comm.size = 2
    comm.allreduce.return_value = 0.25
    comp = np.array([[[[0.5]]], [[[-0.3]]]], dtype=np.complex128)
    out, _ = nonlocal_sde.restrict_chi_phys_to_positive_eigenvalues(_chi_from_compound(comp, 1), comm)
    assert np.allclose(comm.allreduce.call_args.args[0], 2.0, atol=1e-6)
    assert np.allclose(_compound_of(out, 1)[:, 0, 0, 0], [0.5, 8.0], atol=1e-5)


def test_restrict_chi_phys_falls_back_to_a_fixed_floor_without_a_healthy_block():
    """With every static block crossed the inverse is floored at the fallback, pinning chi at its inverse."""
    comp = np.array([[[[-0.5]]], [[[-0.3]]]], dtype=np.complex128)
    out, n_restricted = nonlocal_sde.restrict_chi_phys_to_positive_eigenvalues(_chi_from_compound(comp, 1))
    pinned = 1.0 / nonlocal_sde._RESTRICTION_FALLBACK_FLOOR
    assert n_restricted == 2
    assert np.allclose(_compound_of(out, 1)[:, 0, 0, 0], [pinned, pinned], atol=1e-3)


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


def test_load_node_shared_local_vertex_exposes_the_requested_storage_order_as_a_view(monkeypatch):
    """With an axis order the private copy and the node window store that order and expose the logical layout."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    rng = np.random.default_rng(83)
    mat = (rng.standard_normal((2, 2, 2, 2, 3, 4, 4)) + 1j * rng.standard_normal((2, 2, 2, 2, 3, 4, 4))).astype(
        np.complex64
    )
    monkeypatch.setattr(np, "load", lambda *a, **k: mat.copy())
    axes = (4, 0, 1, 5, 2, 3, 6)

    def fn(comm, rank):
        node_comm = comm.Split_type(FAKE_MPI.COMM_TYPE_SHARED)
        out, win = nonlocal_sde._load_node_shared_local_vertex(node_comm, "unused.npy", SpinChannel.DENS, axes=axes)
        res = (out.mat.copy(), out.mat.transpose(axes).flags["C_CONTIGUOUS"])
        out.mat = None
        nonlocal_sde._free_shared_window(win, node_comm)
        return res

    private, _ = nonlocal_sde._load_node_shared_local_vertex(None, "unused.npy", SpinChannel.DENS, axes=axes)
    assert np.array_equal(private.mat, mat) and private.mat.transpose(axes).flags["C_CONTIGUOUS"]
    _, res = run_parallel(2, fn, hostnames=["n0", "n0"])
    assert all(np.array_equal(r[0], mat) and r[1] for r in res)


@pytest.fixture
def stab_logger(monkeypatch):
    """Installs a MagicMock logger as config.logger for the duration of a test (stabilization glue tests)."""
    logger = MagicMock()
    monkeypatch.setattr(config, "logger", logger, raising=False)
    return logger


def test_mixing_history_cap_uses_most_recent_reset_event():
    """The history cap counts iterations since the latest of the release, the mass change and the tracker event."""
    assert nonlocal_sde._mixing_history_cap(10, None, None) is None
    assert nonlocal_sde._mixing_history_cap(10, 7, None) == 2
    assert nonlocal_sde._mixing_history_cap(10, None, 8) == 1
    assert nonlocal_sde._mixing_history_cap(10, 7, 9) == 0
    assert nonlocal_sde._mixing_history_cap(9, 4, 9) == 0
    assert nonlocal_sde._mixing_history_cap(10, None, None, 8) == 1
    assert nonlocal_sde._mixing_history_cap(10, 7, None, 9) == 0
    assert nonlocal_sde._mixing_history_cap(10, 9, 5, 6) == 0


def test_update_jacobian_tracker_windows_and_allow_flip():
    """The tracker sees the raw pairs since the last scaffold event and pauses flips while a scaffold is active."""
    tracker = create_autospec(JacobianTracker, instance=True)
    tracker.update.return_value = True
    history = [(np.full(2, float(i)), np.full(2, i + 0.5), np.full(2, i + 0.25)) for i in range(6)]

    assert nonlocal_sde._update_jacobian_tracker(tracker, history, 10, None, None, None) is True
    assert len(tracker.update.call_args.args[0]) == 6
    assert tracker.update.call_args.kwargs["allow_flip"] is True

    nonlocal_sde._update_jacobian_tracker(tracker, history, 10, 7, None, None)
    assert [float(iterate[0]) for iterate in tracker.update.call_args.args[0]] == [3.0, 4.0, 5.0]
    assert [float(proposal[0]) for proposal in tracker.update.call_args.args[1]] == [3.5, 4.5, 5.5]

    config.stabilization.use_lambda_correction = True
    nonlocal_sde._update_jacobian_tracker(tracker, history, 10, None, None, None)
    assert tracker.update.call_args.kwargs["allow_flip"] is False

    config.stabilization.use_lambda_correction = False
    config.stabilization.use_chi_phys_restriction = True
    nonlocal_sde._update_jacobian_tracker(tracker, history, 10, None, None, None)
    assert tracker.update.call_args.kwargs["allow_flip"] is False

    config.stabilization.use_chi_phys_restriction = False
    nonlocal_sde._update_jacobian_tracker(tracker, history, 10, None, 8, None)
    assert [float(iterate[0]) for iterate in tracker.update.call_args.args[0]] == [4.0, 5.0]
    assert [float(proposal[0]) for proposal in tracker.update.call_args.args[1]] == [4.5, 5.5]

    tracker.update.return_value = False
    annealer = SimpleNamespace(mass_present=True)
    assert nonlocal_sde._update_jacobian_tracker(tracker, history, 10, None, None, annealer) is False
    assert tracker.update.call_args.kwargs["allow_flip"] is False


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
        single.assert_called_once_with(chi, lambda_previous=None)

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
        single.assert_called_once_with(chi, lambda_previous=None)
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
    monkeypatch.setattr(gc, "collect", lambda *args, **kwargs: 0)

    gf_stub = SimpleNamespace(
        get_fill_nonlocal=lambda: (1.0, np.eye(1, dtype=np.complex128), np.zeros((1, 1, 1, 1, 1))),
        get_ekin=lambda: 0.0,
        get_epot=lambda: 0.0,
        save=lambda *a, **k: None,
        free=lambda: None,
    )
    monkeypatch.setattr(
        nonlocal_sde,
        "GreensFunction",
        SimpleNamespace(
            get_g_full=lambda *a, **k: gf_stub, get_fill_nonlocal_from_sigma=lambda *a, **k: gf_stub.get_fill_nonlocal()
        ),
    )
    monkeypatch.setattr(nonlocal_sde, "update_mu", lambda *a, **k: 0.5)
    monkeypatch.setattr(
        nonlocal_sde,
        "_update_occ_and_energies_distributed",
        lambda *a: (1.0, np.zeros((1, 1)), np.zeros((1, 1, 1, 1, 1)), 0.0, 0.0),
    )

    calls = []

    def fake_proposal(sigma_in, *args, annealer=None, **kwargs):
        calls.append(len(calls) + 1)
        proposal = proposal_step(sigma_in, len(calls), annealer)
        return proposal if args[-2].rank == 0 else None  # like the real map: comm precedes current_iter

    monkeypatch.setattr(nonlocal_sde, "calculate_sigma_proposal", fake_proposal)

    mat = np.full((1, 1, 1, 16), 1.0 + 0.1j, dtype=np.complex64)
    sigma_dmft = SelfEnergy(mat, (1, 1, 1), has_compressed_q_dimension=True, beta=10.0)

    v_nonloc = MagicMock()
    v_nonloc.copy.return_value = v_nonloc
    v_nonloc.reduce_q.return_value = v_nonloc

    def run(comm=None, **kwargs):
        comm = create_comm_mock() if comm is None else comm
        return nonlocal_sde.calculate_self_energy_q(comm, None, v_nonloc, sigma_dmft, sigma_dmft.copy(), **kwargs)

    return run, calls, logger


def _run_loop_on_two_node_ranks(monkeypatch, tmp_path, spied_name, spied_owner):
    """Runs the loop on two fake ranks of one node, returning the thread names of the in-loop calls of the spy."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    monkeypatch.setattr(nonlocal_sde, "MPI", FAKE_MPI)
    run, calls, _ = _setup_self_energy_loop(monkeypatch, tmp_path, lambda s, n, a: s.copy(), max_iter=3, epsilon=0.0)
    original = getattr(spied_owner, spied_name)
    seen = []

    def spy(*args, **kwargs):
        if calls:  # the loop has started: everything before the first proposal is setup, not the iteration
            seen.append(threading.current_thread().name)
        return original(*args, **kwargs)

    monkeypatch.setattr(spied_owner, spied_name, spy)
    run_parallel(2, lambda comm, rank: run(comm), hostnames=["n0", "n0"])
    return seen


def test_loop_returns_sigma_on_rank0_only_after_interpolating_the_final_sigma_on_every_rank(monkeypatch, tmp_path):
    """Only rank 0 gets the loop's result back, and the interpolation still reads the final sigma on every rank."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    monkeypatch.setattr(nonlocal_sde, "MPI", FAKE_MPI)
    run, _, _ = _setup_self_energy_loop(monkeypatch, tmp_path, lambda s, n, a: s.copy(), max_iter=2, epsilon=0.0)
    config.self_energy_interpolation.do_interpolation = True
    seen = []

    def interpolate(sigma, beta_target, niv_target, comm):
        seen.append(sigma.mat.copy())
        return sigma.copy() if comm.rank == 0 else None

    monkeypatch.setattr(nonlocal_sde, "interpolate_sigma", interpolate)
    _, res = run_parallel(2, lambda comm, rank: run(comm), hostnames=["n0", "n0"])
    assert isinstance(res[0], SelfEnergy) and res[1] is None
    assert len(seen) == 2 and all(np.array_equal(m.reshape(-1), res[0].mat.reshape(-1)) for m in seen)


def test_loop_forwards_the_chunk_budgets_to_every_proposal(monkeypatch, tmp_path):
    """The chunk budgets handed to the loop reach every proposal evaluation; None is forwarded as None."""
    run, calls, _ = _setup_self_energy_loop(monkeypatch, tmp_path, lambda s, n, a: s.copy(), max_iter=2, epsilon=0.0)
    fake, seen = nonlocal_sde.calculate_sigma_proposal, []

    def spy(*args, **kwargs):
        seen.append(kwargs["chunk_budgets"])
        return fake(*args, **kwargs)

    monkeypatch.setattr(nonlocal_sde, "calculate_sigma_proposal", spy)
    budgets = memory_estimator.ChunkBudgets(12345, 678, 9)
    run(chunk_budgets=budgets)
    run()
    assert seen == [budgets, budgets, None, None]


def test_resumed_run_shares_the_starting_iterate_per_node_before_the_first_proposal(monkeypatch, tmp_path):
    """A resumed run reads its iterate on rank 0 only and hands it to the node-shared window before iteration one."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    monkeypatch.setattr(nonlocal_sde, "MPI", FAKE_MPI)
    run, calls, _ = _setup_self_energy_loop(monkeypatch, tmp_path, lambda s, n, a: s.copy(), max_iter=2, epsilon=0.0)
    monkeypatch.setattr(nonlocal_sde, "_init_mu_history", lambda starting_iter: [config.sys.mu])
    original, before, loads, broadcasts = nonlocal_sde._share_sigma_per_node, [], [], []

    def spy(sigma, comm, node_comm, roots_comm):
        first = not calls  # read before the collective, which lets the other rank race ahead into the proposal
        shared, win = original(sigma, comm, node_comm, roots_comm)
        if first:
            before.append((threading.current_thread().name, complex(shared.mat.reshape(-1)[0])))
        return shared, win

    monkeypatch.setattr(nonlocal_sde, "_share_sigma_per_node", spy)
    monkeypatch.setattr(mpi_utils.MpiDistributor, "bcast_npoint", lambda self, obj, root=0: broadcasts.append(obj))

    mat = np.full((1, 1, 1, 16), 0.5 + 0.2j, dtype=np.complex64)

    def loaded(default):
        loads.append(threading.current_thread().name)
        return SelfEnergy(mat.copy(), (1, 1, 1), has_compressed_q_dimension=True, beta=10.0), 3

    monkeypatch.setattr(nonlocal_sde, "get_starting_sigma", loaded)
    run_parallel(2, lambda comm, rank: run(comm), hostnames=["n0", "n0"])
    assert loads == ["rank0"]
    value = complex(np.complex64(0.5 + 0.2j))
    assert sorted(before) == [("rank0", value), ("rank1", value)]
    assert broadcasts == []

    before.clear()
    calls.clear()
    monkeypatch.setattr(nonlocal_sde, "get_starting_sigma", lambda default: (default, 0))
    run_parallel(2, lambda comm, rank: run(comm), hostnames=["n0", "n0"])
    assert before == []


def test_warm_start_holds_the_dmft_filling_and_resolves_mu(monkeypatch, tmp_path):
    """A resumed run holds the DMFT lattice filling, not its start's filling at the predecessor's mu, and re-solves
    the start's mu for it before the first proposal."""
    run, calls, logger = _setup_self_energy_loop(
        monkeypatch, tmp_path, lambda s, n, a: s.copy(), max_iter=1, epsilon=0.0
    )
    config.sys.mu_dmft = 0.5
    start = SelfEnergy(
        np.full((1, 1, 1, 16), 0.5 + 0.2j, dtype=np.complex64), (1, 1, 1), has_compressed_q_dimension=True, beta=10.0
    )
    monkeypatch.setattr(nonlocal_sde, "get_starting_sigma", lambda default: (start, 3))
    monkeypatch.setattr(nonlocal_sde, "_init_mu_history", lambda starting_iter: [0.7])  # the predecessor's mu

    def g_full(siw, mu, ek, beta):
        n = 0.85 if np.isclose(siw.mat.reshape(-1)[0], 1.0 + 0.1j) and mu == 0.5 else 0.8621
        fill = (n, np.eye(1, dtype=np.complex128), np.zeros((1, 1, 1, 1, 1)))
        return SimpleNamespace(get_fill_nonlocal=lambda: fill, save=lambda *a, **k: None, free=lambda: None)

    greens = SimpleNamespace(get_g_full=g_full, get_fill_nonlocal_from_sigma=lambda *a: g_full(*a).get_fill_nonlocal())
    monkeypatch.setattr(nonlocal_sde, "GreensFunction", greens)
    solves = []
    monkeypatch.setattr(nonlocal_sde, "update_mu", lambda mu0, n, *a, **k: solves.append((mu0, n)) or 0.42)
    fake, proposal_mus = nonlocal_sde.calculate_sigma_proposal, []

    def spy(sigma_in, mu, *args, **kwargs):
        proposal_mus.append(mu)
        return fake(sigma_in, mu, *args, **kwargs)

    monkeypatch.setattr(nonlocal_sde, "calculate_sigma_proposal", spy)
    run()
    assert solves[0] == (0.7, 0.85)  # the start's mu re-solved for the DMFT lattice filling
    assert proposal_mus[0] == 0.42 and config.sys.n == 0.85
    assert all(n == 0.85 for _, n in solves)  # every later mu update holds the same filling
    infos = [str(c.args[0]) for c in logger.info.call_args_list]
    assert "Filling of the updated Green's function: 1.000000 (target 0.850000)." in infos  # the stub reports 1.0


def test_loop_concatenates_the_previous_iterate_on_rank0_only(monkeypatch, tmp_path):
    """Only rank 0 rebuilds the previous iterate on the DMFT tail each iteration; the other ranks never copy it."""
    seen = _run_loop_on_two_node_ranks(monkeypatch, tmp_path, "concatenate_self_energies", SelfEnergy)
    assert seen == ["rank0"] * 3


def test_loop_measures_the_step_residual_on_rank0_only(monkeypatch, tmp_path):
    """The step residual only decides convergence on rank 0, so no other rank evaluates it."""
    seen = _run_loop_on_two_node_ranks(monkeypatch, tmp_path, "_relative_sigma_residual", nonlocal_sde)
    assert seen == ["rank0"] * 3


_FIXED_POINT = 0.3 - 0.2j
_STABLE_GAINS = np.linspace(-0.4, -0.04, 8)
_UNSTABLE_GAINS = np.array([-0.40, -0.34, 1.30, -0.28, -0.22, -0.16, -0.10, -0.04])
_DEGENERATE_STABLE_GAINS = np.full(8, 0.5)


def _affine_core_step(gains: np.ndarray):
    """Returns a synthetic proposal map S(x) = x* + A (x - x*) on the core window, A = diag(gains) in [Re; Im]."""

    def step(sigma_in, n_call, annealer):
        sigma_out = sigma_in.copy()
        niv_core = config.box.niv_core
        window = sigma_out.mat[..., sigma_out.niv - niv_core : sigma_out.niv + niv_core]
        target = to_vec(np.full(window.shape, _FIXED_POINT))
        window[...] = to_mat(target + gains * (to_vec(window) - target), window.shape)
        return sigma_out

    return step


def _core_distance(sigma):
    """Returns the relative L2 distance of a self-energy's core window from the synthetic map's fixed point."""
    window = sigma.mat[..., sigma.niv - config.box.niv_core : sigma.niv + config.box.niv_core]
    target = to_vec(np.full(window.shape, _FIXED_POINT))
    return float(np.linalg.norm(to_vec(window) - target) / np.linalg.norm(target))


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


def test_loop_flag_off_never_builds_tracker_or_history(monkeypatch, tmp_path):
    """With the Jacobian flag off the loop mixes without a tracker and allocates no pair history for linear mixing."""

    def step(sigma_in, n_call, annealer):
        return sigma_in.copy()

    run, calls, _ = _setup_self_energy_loop(monkeypatch, tmp_path, step)
    spy = create_autospec(nonlocal_sde.apply_mixing_strategy, wraps=nonlocal_sde.apply_mixing_strategy)
    monkeypatch.setattr(nonlocal_sde, "apply_mixing_strategy", spy)
    run()

    assert len(calls) == 2 and len(spy.call_args_list) == 2
    assert all(call.args[3] is None and call.args[4] is None for call in spy.call_args_list)


@pytest.mark.parametrize("use_exact_jacobian, max_iter, expected", [(True, 80, 1), (False, 80, 0), (True, 3, 0)])
def test_loop_evaluates_the_exact_jacobian_once_after_pure_convergence(
    monkeypatch, tmp_path, use_exact_jacobian, max_iter, expected
):
    """The exact spectrum is evaluated once, at the converged iterate, only with the flag on and pure convergence."""
    run, calls, _ = _setup_self_energy_loop(
        monkeypatch, tmp_path, _affine_core_step(_STABLE_GAINS), max_iter=max_iter, epsilon=5e-4
    )
    config.stabilization.use_jacobian_stabilization = True
    config.stabilization.use_exact_jacobian = use_exact_jacobian
    seen = []
    monkeypatch.setattr(nonlocal_sde, "_install_exact_jacobian", lambda tracker, sigma, mu, *a: seen.append(sigma))
    result = run()

    assert len(seen) == expected
    if expected:
        assert len(calls) < max_iter and seen[0] is result


def test_install_exact_jacobian_frees_the_operator_and_installs_rank_0s_spectrum(monkeypatch):
    """The helper builds the operator, solves, frees it even when the solve raises, and installs on rank 0 only."""
    import dgamore.sigma_jacobian as sigma_jacobian

    spectrum = (np.array([0.5 + 0.0j]), np.array([1e-10]), np.ones((4, 1), dtype=np.complex128))

    def solve_before_free(jac, comm):
        assert not jac.free.called
        return spectrum

    operator, solve, tracker = MagicMock(), MagicMock(side_effect=solve_before_free), MagicMock()
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
    monkeypatch.setattr(sigma_jacobian, "ExactJacobian", operator)
    monkeypatch.setattr(sigma_jacobian, "leading_eigenpairs", solve)
    args = ("sigma", 1.5, "u", "v", None, "dmft", "irr", "fbz")
    nonlocal_sde._install_exact_jacobian(tracker, *args, create_comm_mock())
    assert operator.call_args.args[1] == 1.5 and solve.call_args.args[0] is operator.return_value
    assert operator.return_value.free.call_count == 1
    tracker.install_exact_spectrum.assert_called_once_with(*spectrum)

    operator.return_value.free.reset_mock()
    nonlocal_sde._install_exact_jacobian(tracker, *args, MagicMock(rank=1))
    assert operator.return_value.free.call_count == 1 and tracker.install_exact_spectrum.call_count == 1

    operator.return_value.free.reset_mock()
    solve.side_effect = RuntimeError("ARPACK")
    with pytest.raises(RuntimeError):
        nonlocal_sde._install_exact_jacobian(tracker, *args, create_comm_mock())
    assert operator.return_value.free.call_count == 1


def test_loop_tracker_stabilizes_unstable_synthetic_map(monkeypatch, tmp_path):
    """A map with one unstable direction runs away under plain mixing and reaches its fixed point with the flag on."""
    # the unstable gain sits on the real part of the first positive core frequency, the window the step residual reads
    run, calls, _ = _setup_self_energy_loop(
        monkeypatch, tmp_path, _affine_core_step(_UNSTABLE_GAINS), max_iter=55, epsilon=1e-3
    )
    diverged = run()

    assert len(calls) == 55
    assert _core_distance(diverged) > 1e3

    run, calls, _ = _setup_self_energy_loop(
        monkeypatch, tmp_path, _affine_core_step(_UNSTABLE_GAINS), max_iter=80, epsilon=5e-4
    )
    config.stabilization.use_jacobian_stabilization = True
    config.sys.n_bands = 1
    stabilized = run()

    assert len(calls) < 80
    assert _core_distance(stabilized) < 1e-2


def test_loop_tracker_stabilizes_map_with_degenerate_stable_gains(monkeypatch, tmp_path):
    """A map sharing one stable gain across all directions leaves a rank-deficient secant window but still converges."""
    run, calls, _ = _setup_self_energy_loop(
        monkeypatch, tmp_path, _affine_core_step(_DEGENERATE_STABLE_GAINS), max_iter=80, epsilon=5e-4
    )
    config.stabilization.use_jacobian_stabilization = True
    stabilized = run()

    assert len(calls) < 80
    assert _core_distance(stabilized) < 1e-2


def test_loop_tracker_pauses_flips_while_scaffold_active(monkeypatch, tmp_path):
    """Flips stay paused while the lambda-correction scaffold is active and are allowed again after its release."""
    run, _, _ = _setup_self_energy_loop(
        monkeypatch, tmp_path, _affine_core_step(_STABLE_GAINS), max_iter=30, epsilon=1e-3
    )
    config.stabilization.use_lambda_correction = True
    config.stabilization.use_jacobian_stabilization = True
    spy = create_autospec(JacobianTracker.update, wraps=JacobianTracker.update)
    monkeypatch.setattr(nonlocal_sde.JacobianTracker, "update", spy)
    run()

    flags = [call.kwargs["allow_flip"] for call in spy.call_args_list]
    assert flags.count(False) > 0 and flags.count(True) > 0
    assert flags.index(True) == flags.count(False)


def test_loop_tracker_event_feeds_history_cap(monkeypatch, tmp_path):
    """A tracker event reaches the accelerated-mixing cap while the tracker's own raw window is never shortened."""
    run, _, _ = _setup_self_energy_loop(
        monkeypatch, tmp_path, _affine_core_step(_UNSTABLE_GAINS), max_iter=15, epsilon=1e-3
    )
    config.stabilization.use_jacobian_stabilization = True
    config.sys.n_bands = 1
    cap_spy = create_autospec(nonlocal_sde._mixing_history_cap, wraps=nonlocal_sde._mixing_history_cap)
    update_spy = create_autospec(JacobianTracker.update, wraps=JacobianTracker.update)
    monkeypatch.setattr(nonlocal_sde, "_mixing_history_cap", cap_spy)
    monkeypatch.setattr(nonlocal_sde.JacobianTracker, "update", update_spy)
    run()

    events = [call.args[3] for call in cap_spy.call_args_list if len(call.args) == 4 and call.args[3] is not None]
    lengths = [len(call.args[1]) for call in update_spy.call_args_list]
    assert events and min(events) > 1
    assert lengths == sorted(lengths) and lengths[-1] == TRACKER_PAIRS + 1


def test_loop_tracker_runs_on_rank0_only_on_two_node_ranks(monkeypatch, tmp_path):
    """On two ranks of one node only rank 0 updates the tracker, saves the proposal and returns sigma, and rank 1
    evaluates exactly the iterate rank 0's tracker-processed mixing produced."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    monkeypatch.setattr(nonlocal_sde, "MPI", FAKE_MPI)
    step, inputs, buffers = _affine_core_step(_STABLE_GAINS), {"rank0": [], "rank1": []}, {"rank0": [], "rank1": []}

    def recording_step(sigma_in, n, annealer):
        inputs[threading.current_thread().name].append(sigma_in.mat.reshape(-1).copy())
        buffers[threading.current_thread().name].append(sigma_in.mat.__array_interface__["data"][0])
        return step(sigma_in, n, annealer)

    run, calls, _ = _setup_self_energy_loop(monkeypatch, tmp_path, recording_step, max_iter=4, epsilon=0.0)
    config.stabilization.use_jacobian_stabilization = True
    config.sys.n_bands = 1
    update, save_iteration = JacobianTracker.update, nonlocal_sde._save_sigma_iteration
    updates, estimates, saves = [], [], []

    def update_spy(self, *args, **kwargs):
        updates.append(threading.current_thread().name)
        changed = update(self, *args, **kwargs)
        estimates.append(self.estimated)
        return changed

    def save_spy(sigma, base_name, current_iter):
        if base_name == "sigma_dga_proposal":
            saves.append(threading.current_thread().name)
        return save_iteration(sigma, base_name, current_iter)

    monkeypatch.setattr(nonlocal_sde.JacobianTracker, "update", update_spy)
    monkeypatch.setattr(nonlocal_sde, "_save_sigma_iteration", save_spy)
    monkeypatch.setattr(nonlocal_sde, "_save_jacobian_spectrum", lambda tracker, converged, *rows: None)
    mix, mixed = nonlocal_sde.apply_mixing_strategy, []

    def mix_spy(*args, **kwargs):
        out = mix(*args, **kwargs)
        mixed.append(out.mat.reshape(-1).copy())  # copied here: sharing the window drops rank 0's own array
        return out

    monkeypatch.setattr(nonlocal_sde, "apply_mixing_strategy", mix_spy)
    _, results = run_parallel(2, lambda comm, rank: run(comm), hostnames=["n0", "n0"])

    assert len(calls) == 8 and updates == ["rank0"] * 4 and saves == ["rank0"] * 4
    assert any(estimates)
    assert results[0] is not None and results[1] is None
    # values, not layout: rank 0's own sigma_dga save decompresses its momentum axis
    assert len(mixed) == 4 and len(inputs["rank0"]) == len(inputs["rank1"]) == 4
    assert all(np.array_equal(a, b) for a, b in zip(inputs["rank0"], inputs["rank1"]))
    assert all(np.array_equal(m, x) for m, x in zip(mixed[:-1], inputs["rank1"][1:]))
    assert buffers["rank0"][1:] == buffers["rank1"][1:]  # from iteration 2 on: one shared buffer per node, no copies


def test_loop_tracker_logs_at_convergence(monkeypatch, tmp_path):
    """At convergence the loop logs the tracker's flipped-direction count, effective damping and predicted rate."""
    run, calls, logger = _setup_self_energy_loop(
        monkeypatch, tmp_path, _affine_core_step(_STABLE_GAINS), max_iter=30, epsilon=1e-3
    )
    config.stabilization.use_jacobian_stabilization = True
    run()

    assert len(calls) < 30
    assert any(
        "Jacobian tracker at convergence" in str(c.args[0]) and "flipped" in str(c.args[0])
        for c in logger.info.call_args_list
    )


def test_loop_warns_when_convergence_beats_the_minimum_iteration_count(monkeypatch, tmp_path):
    """A run converging in fewer iterations than the certified spectrum's minimum logs the flat-direction warning."""
    run, calls, logger = _setup_self_energy_loop(
        monkeypatch, tmp_path, _affine_core_step(_STABLE_GAINS), max_iter=30, epsilon=1e-3
    )
    config.stabilization.use_jacobian_stabilization = True
    monkeypatch.setattr(nonlocal_sde.JacobianTracker, "minimum_iterations", lambda self: 999)
    run()

    assert len(calls) < 30
    assert any("below the minimum 999" in str(c.args[0]) for c in logger.warning.call_args_list)


@pytest.mark.parametrize(
    "strategy, applied",
    [("linear", "linearly mixed"), ("pulay", "Pulay mixing applied"), ("anderson", "Anderson acceleration applied")],
)
def test_loop_stabilizes_the_damped_step_of_every_strategy(monkeypatch, tmp_path, strategy, applied):
    """Every strategy hands exactly its damped steps and its linear fallbacks to the stabilized step."""
    run, calls, logger = _setup_self_energy_loop(
        monkeypatch, tmp_path, _affine_core_step(_UNSTABLE_GAINS), max_iter=8, epsilon=1e-3
    )
    config.stabilization.use_jacobian_stabilization = True
    config.sys.n_bands = 1
    config.self_consistency.mixing_strategy = strategy
    config.self_consistency.mixing_history_length = 2
    monkeypatch.setattr(nonlocal_sde.JacobianTracker, "active", property(lambda self: True))
    spy = create_autospec(JacobianTracker.stabilize_step, wraps=JacobianTracker.stabilize_step)
    monkeypatch.setattr(nonlocal_sde.JacobianTracker, "stabilize_step", spy)
    run()

    fallbacks = sum("falling back to linear mixing" in str(c.args[0]) for c in logger.warning.call_args_list)
    damped = sum("linearly mixed" in str(c.args[0]) for c in logger.info.call_args_list) + fallbacks
    assert len(calls) > 3 and 0 < damped <= len(calls) and len(spy.call_args_list) == damped
    assert any(applied in str(c.args[0]) for c in logger.info.call_args_list)


@pytest.mark.parametrize(
    "strategy, warned",
    [("pulay", "Pulay SVD ill-conditioned"), ("anderson", "Anderson SVD failed")],
)
def test_loop_stabilizes_the_mixed_step_of_an_ill_conditioned_accelerated_fallback(
    monkeypatch, tmp_path, strategy, warned
):
    """An accelerated step of either scheme falling back to linear mixing still hands its result to the step."""
    run, calls, logger = _setup_self_energy_loop(
        monkeypatch, tmp_path, _affine_core_step(_UNSTABLE_GAINS), max_iter=8, epsilon=1e-3
    )
    config.stabilization.use_jacobian_stabilization = True
    config.sys.n_bands = 1
    config.self_consistency.mixing_strategy = strategy
    config.self_consistency.mixing_history_length = 2
    monkeypatch.setattr(nonlocal_sde.JacobianTracker, "active", property(lambda self: True))
    real_svd = np.linalg.svd

    def rank_deficient_svd(matrix, *args, **kwargs):
        """Returns an all-zero decomposition for the accelerated least-squares problem and the true one elsewhere."""
        if kwargs.get("full_matrices") is False:
            width = matrix.shape[1]
            return np.zeros_like(matrix), np.zeros(width), np.zeros((width, width))
        return real_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", rank_deficient_svd)
    spy = create_autospec(JacobianTracker.stabilize_step, wraps=JacobianTracker.stabilize_step)
    monkeypatch.setattr(nonlocal_sde.JacobianTracker, "stabilize_step", spy)
    run()

    assert len(spy.call_args_list) == len(calls)
    assert any(warned in str(c.args[0]) for c in logger.warning.call_args_list)


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
    gamma, gchi0_q_inv, u_loc, _ = _bse_assembly_inputs(rng)
    ref = nonlocal_sde.create_auxiliary_chi_r_q(gamma, gchi0_q_inv, u_loc).sum_over_vn(config.sys.beta)
    out = nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_q_inv, u_loc)
    assert np.allclose(out.mat, ref.mat, atol=1e-5)
    assert out.channel == gamma.channel and not out.full_niw_range and out.num_vn_dimensions == 1


def test_create_auxiliary_chi_r_q_sum_is_bit_invariant_under_the_chunk_budget(monkeypatch):
    """Single slices, w-chunks, q-groups and the whole box give the same bits, so the budget may follow free memory."""
    rng = np.random.default_rng(32)
    o, nqi, nw, niv = 2, 4, 3, 2
    gamma, gchi0_q_inv, u_loc, _ = _bse_assembly_inputs(rng, o=o, nqi=nqi, nw=nw, niv=niv)
    whole = nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_q_inv, u_loc, 2**62)
    one_wn = (2 * niv) ** 2 * o**4 * gamma.mat.itemsize
    for budget in (1, 2 * one_wn, nw * one_wn, 2 * nw * one_wn):  # single slice, w-chunk, one q, q-group of two
        chunked = nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_q_inv, u_loc, budget)
        assert np.array_equal(chunked.mat, whole.mat)
    monkeypatch.setattr(nonlocal_sde, "SLICE_CHUNK_BYTES", 1)
    assert np.array_equal(nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_q_inv, u_loc).mat, whole.mat)


def test_create_auxiliary_chi_r_q_sum_with_rhs_solves_the_bethe_salpeter_system(monkeypatch):
    """With rhs every chunk solves the Bethe-Salpeter system, the full auxiliary susceptibility applied to rhs."""
    monkeypatch.setattr("dgamore.n_point_base.DTYPE", np.complex128)
    rng = np.random.default_rng(34)
    gamma, gchi0_q_inv, u_loc, _ = _bse_assembly_inputs(rng, o=2, nqi=3, nw=2, niv=2)
    shape = gchi0_q_inv.current_shape
    rhs_mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    rhs = FourPoint(rhs_mat, SpinChannel.NONE, gchi0_q_inv.nq, 1, 1, False, True, True)
    chi_star = nonlocal_sde.create_auxiliary_chi_r_q(gamma, gchi0_q_inv, u_loc).to_half_niw_range().mat
    ref = np.einsum("qijabwvp,qbaklwp->qijklwv", chi_star, rhs.mat)
    for budget in (None, 1):
        out = nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_q_inv, u_loc, budget, rhs=rhs)
        assert np.allclose(out.mat, ref, atol=1e-10)


def test_create_auxiliary_chi_r_q_sum_eliminates_the_pairs_without_vertex_of_a_two_atom_cell(monkeypatch):
    """For two atoms built atom by atom the free pairs are logged and eliminated; the sum matches the full inversion."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
    gamma, gchi0_q_inv, u_loc, _ = _bse_assembly_inputs(np.random.default_rng(33), o=4, nqi=2, nw=2, niv=2)
    atom = np.array([0, 0, 1, 1])
    same = atom[:, None, None, None] == atom[None, :, None, None]
    per_atom = same & (same.transpose(0, 2, 1, 3)) & (same.transpose(0, 2, 3, 1))
    gamma.mat[~per_atom] = 0.0
    u_loc.mat[~per_atom] = 0.0
    out = nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_q_inv, u_loc)
    ref = nonlocal_sde.create_auxiliary_chi_r_q(gamma, gchi0_q_inv, u_loc).sum_over_vn(config.sys.beta)
    assert "8 of 16 orbital pairs" in config.logger.info.call_args.args[0]
    assert np.allclose(out.mat, ref.mat, atol=1e-5)
    assert np.array_equal(nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_q_inv, u_loc, 1).mat, out.mat)


def test_update_occ_and_energies_distributed_matches_the_full_box_evaluation(monkeypatch):
    """The k-distributed occupation/energy evaluation matches the single-rank full-box reference with its offset."""
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

    sigma_ref = sigma_new.copy().concatenate_self_energies(
        sigma_dmft_full, shell_offset=sigma_new.shell_offset_from(sigma_dmft_full)
    )
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


def test_update_occ_and_energies_distributed_carries_the_shell_offset_of_the_mixed_sigma(monkeypatch):
    """A momentum-dependent constant in sigma's shell is carried into the DMFT-box extension on every rank."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, o, niv_core, niv, niv_dmft, beta, mu = (4, 2, 1), 2, 2, 4, 8, 9.0, 0.4
    nk_tot = int(np.prod(nk))
    rng = np.random.default_rng(13)
    config.sys.beta, config.sys.n_bands, config.sys.mu = beta, o, mu
    config.lattice.nk = nk
    config.lattice.k_grid = SimpleNamespace(nk_tot=nk_tot, nk=nk)
    ek = rng.standard_normal((*nk, o, o))
    config.lattice.hamiltonian = MagicMock(get_ek=MagicMock(return_value=ek + ek.swapaxes(-1, -2)))
    dmft_mat = (rng.standard_normal((1, 1, 1, o, o, 2 * niv_dmft)) * 0.1 + 0.2j).astype(np.complex64)
    offset = rng.standard_normal((nk_tot, o, o))
    # the mixed sigma: DMFT shell plus a k-dependent constant everywhere, a random core box inside
    sig_mat = np.broadcast_to(dmft_mat[0, 0, 0, ..., niv_dmft - niv : niv_dmft + niv], (nk_tot, o, o, 2 * niv)).copy()
    sig_mat += offset[..., None]
    sig_mat[..., niv - niv_core : niv + niv_core] = rng.standard_normal((nk_tot, o, o, 2 * niv_core)) * 0.1 + 0.3j
    sigma_new = SelfEnergy(sig_mat.astype(np.complex64), nk, has_compressed_q_dimension=True, beta=beta)
    sigma_dmft_full = SelfEnergy(dmft_mat.copy(), (1, 1, 1), beta=beta)

    sigma_ref = sigma_new.copy().concatenate_self_energies(sigma_dmft_full, shell_offset=offset)
    giwk_ref = GreensFunction.get_g_full(sigma_ref, mu, ek + ek.swapaxes(-1, -2), beta)
    _, occ_ref, occ_k_ref = giwk_ref.get_fill_nonlocal()
    ekin_ref, epot_ref = giwk_ref.get_ekin(), giwk_ref.get_epot()

    def fn(comm, rank):
        d_full = mpi_utils.MpiDistributor(ntasks=nk_tot, comm=comm)
        return nonlocal_sde._update_occ_and_energies_distributed(sigma_new, sigma_dmft_full, d_full, mu)

    _, res = run_parallel(2, fn)
    for _, occ, occ_k, ekin, epot in res:
        assert np.allclose(occ, occ_ref, atol=1e-6) and np.allclose(occ_k, occ_k_ref, atol=1e-6)
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


def test_share_sigma_per_node_hands_rank0_sigma_with_its_metadata_to_every_rank_once_per_node(monkeypatch):
    """Rank 0's Sigma reaches every rank, metadata included, as a view of its node's single shared buffer."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    mixed = (np.arange(2 * 2 * 12).reshape(2, 2, 1, 1, 1, 12) + 1j).astype(np.complex64)

    def fn(comm, rank):
        sigma = SelfEnergy(mixed.copy(), (2, 2, 1), calc_smom=False, beta=10.0) if rank == 0 else None
        node_comm = comm.Split_type(0)
        roots_comm = comm.Split(0 if node_comm.rank == 0 else 1)
        shared, win = nonlocal_sde._share_sigma_per_node(sigma, comm, node_comm, roots_comm)
        meta = (shared.has_compressed_q_dimension, shared.nq, shared.niv, shared._beta)
        return shared.mat.copy(), shared.mat.__array_interface__["data"][0], win is not None, meta

    _, res = run_parallel(4, fn, hostnames=["n0", "n0", "n1", "n1"])
    assert all(np.array_equal(r[0], mixed.reshape(4, 1, 1, 12)) for r in res)
    assert len({r[1] for r in res}) == 2 and all(r[2] for r in res)
    assert all(r[3] == (True, (2, 2, 1), 6, 10.0) for r in res)


def test_assemble_occupation_from_momentum_slices_matches_the_full_grid_filling(monkeypatch):
    """The slice-wise filling of a momentum-local sigma assembles to the full-grid filling bit for bit."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, o, niv, beta, mu = (4, 2, 1), 2, 6, 9.0, 0.4
    nk_tot = int(np.prod(nk))
    rng = np.random.default_rng(17)
    config.sys.beta, config.sys.n_bands = beta, o
    config.lattice.nk = nk
    config.lattice.k_grid = SimpleNamespace(nk_tot=nk_tot, nk=nk)
    ek = rng.standard_normal((*nk, o, o))
    ek = ek + ek.swapaxes(-1, -2)
    sigma = SelfEnergy((rng.standard_normal((1, 1, 1, o, o, 2 * niv)) * 0.1 + 0.3j).astype(np.complex64), beta=beta)
    n_ref, occ_ref, occ_k_ref = GreensFunction.get_g_full(sigma, mu, ek, beta).get_fill_nonlocal()

    def fn(comm, rank):
        d_full = mpi_utils.MpiDistributor(ntasks=nk_tot, comm=comm)
        n_my = d_full.my_size
        ek_my = ek.reshape(-1, o, o)[d_full.my_slice].reshape(n_my, 1, 1, o, o)
        occ_k_my = GreensFunction.get_g_full(sigma, mu, ek_my, beta).get_fill_nonlocal()[2]
        return nonlocal_sde._assemble_occupation(occ_k_my, d_full)

    _, res = run_parallel(3, fn)
    for n_el, occ, occ_k in res:
        assert n_el == n_ref and np.array_equal(occ, occ_ref) and np.array_equal(occ_k, occ_k_ref)


def _column_sde_setup(auto: bool, nk=(4, 4, 2), o=2, niw=3, niv=2, seed=41):
    """Configures a small grid (optionally in auto mode with random complex unitaries) and returns kernel and G."""
    rng = np.random.default_rng(seed)
    grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    if auto:  # a real grid in auto mode with complex unitaries and anti-unitary members
        grid._auto_mode = True
        grid._auto_us = np.array(
            [
                np.linalg.qr(rng.standard_normal((o, o)) + 1j * rng.standard_normal((o, o)))[0]
                for _ in range(grid.nk_tot)
            ]
        ).reshape(*nk, o, o)
        grid._auto_sigmas = np.ones(nk)
        grid._auto_conjs = rng.random(nk) < 0.3
    config.lattice.nk, config.lattice.k_grid = nk, grid
    config.box.niw_core, config.box.niv_core = niw, niv
    config.sys.n_bands, config.sys.beta = o, 12.5
    config.logger = MagicMock()
    k_shape = (grid.nk_irr, o, o, o, o, niw + 1, 2 * niv)
    kernel = (rng.standard_normal(k_shape) + 1j * rng.standard_normal(k_shape)).astype(np.complex64)
    g_shape = (*nk, o, o, 2 * (niv + niw))
    g_mat = (rng.standard_normal(g_shape) + 1j * rng.standard_normal(g_shape)).astype(np.complex64)
    return kernel, GreensFunction(g_mat, calc_filling=False, nk=nk, beta=config.sys.beta)


def _run_column_sde_parallel(size, kernel, giwk, chunk_bytes=None, hostnames=None):
    """Runs the column-distributed contraction on fake ranks and returns rank 0's real-space result."""

    def fn(comm, rank):
        d_irr = mpi_utils.MpiDistributor(ntasks=config.lattice.k_grid.nk_irr, comm=comm)
        node_comm = comm.Split_type(FAKE_MPI.COMM_TYPE_SHARED)
        g_r, win = nonlocal_sde._build_rspace_giwk_window(giwk, node_comm)
        kern = FourPoint(kernel[d_irr.my_slice].copy(), SpinChannel.NONE, config.lattice.nk, 1, 1, False, True, True)
        sigma = nonlocal_sde._run_column_sde(kern, d_irr, g_r, giwk.niv, chunk_bytes)
        g_r = None
        nonlocal_sde._free_shared_window(win, node_comm)
        return None if sigma is None else sigma.mat.copy()

    _, res = run_parallel(size, fn, hostnames=hostnames)
    assert all(r is None for r in res[1:])
    return res[0]


@pytest.mark.parametrize("auto", [False, True])
@pytest.mark.parametrize("size", [1, 3])
def test_column_sde_matches_the_qloop_reference(auto, size, monkeypatch):
    """The column-distributed contraction reproduces the q-loop sum over the kernel mapped to the full BZ."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    kernel, giwk = _column_sde_setup(auto)
    k_grid = config.lattice.k_grid
    full_kernel = FourPoint(kernel.copy(), SpinChannel.NONE, config.lattice.nk, 1, 1, False, True, True)
    ref = nonlocal_sde.calculate_sigma_from_kernel(full_kernel.map_to_full_bz(k_grid), giwk, k_grid.get_q_list())

    mat = _run_column_sde_parallel(size, kernel, giwk, hostnames=["n0"] * size)
    sigma = SelfEnergy(mat, config.lattice.nk, False, True, calc_smom=False, beta=config.sys.beta)
    assert np.allclose(sigma.ifft().to_full_niv_range().mat, ref.mat, atol=1e-5)


@pytest.mark.parametrize("auto", [False, True])
def test_column_sde_is_bit_invariant_under_the_round_budget_and_the_contraction_chunk(auto, monkeypatch):
    """Single-task rounds, one-row contraction chunks and the whole block give the same bits; rank counts agree."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    kernel, giwk = _column_sde_setup(auto)
    whole = _run_column_sde_parallel(3, kernel, giwk, chunk_bytes=2**40, hostnames=["n0", "n0", "n1"])
    rounds = _run_column_sde_parallel(3, kernel, giwk, chunk_bytes=1)
    monkeypatch.setattr(memory_estimator, "SDE_CONTRACTION_CHUNK_BYTES", 1)
    rows = _run_column_sde_parallel(3, kernel, giwk, chunk_bytes=2**40)
    assert np.array_equal(rounds, whole) and np.array_equal(rows, whole)
    for size in (1, 2, 4):  # the rank count moves the fold's partial-sum boundaries, a rounding-level change
        assert np.allclose(_run_column_sde_parallel(size, kernel, giwk), whole, atol=1e-5)


def test_column_sde_single_rank_mock_path_equals_the_fake_mpi_run(monkeypatch):
    """The single-rank mock communicator (no messages, private window) gives the fake one-rank result exactly."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    kernel, giwk = _column_sde_setup(True)
    comm = create_comm_mock()
    d_irr = mpi_utils.MpiDistributor(ntasks=config.lattice.k_grid.nk_irr, comm=comm)
    g_r, win = nonlocal_sde._build_rspace_giwk_window(giwk, comm)
    kern = FourPoint(kernel.copy(), SpinChannel.NONE, config.lattice.nk, 1, 1, False, True, True)
    sigma = nonlocal_sde._run_column_sde(kern, d_irr, g_r, giwk.niv)
    assert win is None
    assert np.array_equal(sigma.mat, _run_column_sde_parallel(1, kernel, giwk))


@pytest.mark.parametrize("size", [5, 7])
def test_column_sde_with_more_ranks_than_tasks_matches_one_rank(size, monkeypatch):
    """Ranks without tasks or irreducible momenta join every exchange and leave the result at the one-rank value."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    kernel, giwk = _column_sde_setup(True, niw=1, niv=1)
    one = _run_column_sde_parallel(1, kernel, giwk)
    assert np.allclose(_run_column_sde_parallel(size, kernel, giwk), one, atol=1e-5)


def test_build_rspace_giwk_window_is_the_frequency_first_transform_in_one_buffer_per_node(monkeypatch):
    """Every rank of a node maps one shared buffer holding the momentum FFT of G in frequency-first layout."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    _, giwk = _column_sde_setup(False)
    nb, n_r = config.sys.n_bands, config.lattice.k_grid.nk_tot
    expected = np.moveaxis(giwk.fft().mat.reshape(n_r, nb, nb, -1), -1, 0)

    def fn(comm, rank):
        node_comm = comm.Split_type(FAKE_MPI.COMM_TYPE_SHARED)
        g_r, win = nonlocal_sde._build_rspace_giwk_window(giwk, node_comm)
        res = (g_r.copy(), g_r.__array_interface__["data"][0], g_r.flags["C_CONTIGUOUS"])
        g_r = None
        nonlocal_sde._free_shared_window(win, node_comm)
        return res

    _, res = run_parallel(2, fn, hostnames=["n0", "n0"])
    assert all(np.array_equal(r[0], expected) and r[2] for r in res)
    assert res[0][1] == res[1][1]


@pytest.mark.parametrize("chunk_bytes, n_lines", [(1, 2), (2**40, 1)])
def test_column_sde_logs_progress_at_the_halfway_and_the_last_round(chunk_bytes, n_lines, monkeypatch):
    """Single-task rounds log the halfway and the last round, a single round logs once."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    kernel, giwk = _column_sde_setup(False)
    _run_column_sde_parallel(1, kernel, giwk, chunk_bytes=chunk_bytes)
    lines = [c.args[0] for c in config.logger.info.call_args_list if "column rounds" in c.args[0]]
    assert len(lines) == n_lines and lines[-1].split(":")[1].split()[0] == lines[-1].split()[-4]


def test_sde_chunk_budget_is_the_job_wide_minimum(monkeypatch):
    """Every rank gets the smallest node budget, so all ranks walk the same collective chunk schedule."""
    monkeypatch.setattr(mpi_utils, "job_memory_total", lambda: 2**40)
    node_sizes = (8, 64, 512)  # per-node budgets: capped at 4 GiB, 2 GiB, floored at 256 MiB

    def fn(comm, rank):
        return nonlocal_sde._sde_chunk_budget(comm, SimpleNamespace(size=node_sizes[rank]))

    _, budgets = run_parallel(3, fn)
    assert budgets == [memory_estimator.SLICE_CHUNK_BYTES] * 3


@pytest.mark.parametrize("size", [2, 5])
def test_interpolate_sigma_unfolds_distributed_pole_fits_to_the_full_bz(monkeypatch, size):
    """interpolate_sigma fits the flagged irreducible momenta across ranks and unfolds them to the full BZ on rank 0."""
    config.logger = MagicMock()
    beta, x, hartree, niv = 10.0, -0.3, 1.5, 200
    config.lattice.nk = (4, 4, 1)
    config.lattice.k_grid = bz.KGrid(config.lattice.nk, symmetries=bz.two_dimensional_square_symmetries())
    kx, ky = np.meshgrid(2 * np.pi * np.arange(4) / 4, 2 * np.pi * np.arange(4) / 4, indexing="ij")
    # square-symmetric pole weight, negative (non-causal, hence flagged) around the M point only
    weight = (0.3 * (np.cos(kx) + np.cos(ky)) - 0.2)[..., None, None, None, None]
    vn = MFHelper.vn(niv, beta)
    sigma = SelfEnergy(hartree + weight / (1j * vn - x), nk=(4, 4, 1), has_compressed_q_dimension=False, beta=beta)
    patch_mini_pole_with_exact_single_pole_fit(monkeypatch, x)
    monkeypatch.setattr(contextlib, "redirect_stdout", lambda target: contextlib.nullcontext())

    def fn(comm, rank):
        out = nonlocal_sde.interpolate_sigma(sigma.copy(), beta_target=2.0 * beta, niv_target=niv, comm=comm)
        return None if out is None else out.mat

    _, results = run_parallel(size, fn)
    single = nonlocal_sde.interpolate_sigma(sigma, 2.0 * beta, niv, create_comm_mock()).mat

    target_vn = MFHelper.vn(niv, 2.0 * beta)
    inner = np.abs(target_vn) < vn[niv]
    flagged = sigma.pole_fit_mask().reshape(4, 4, 1)
    expected_inner = hartree + weight / (1j * target_vn[inner] - x)
    plain = sigma.interpolate(2.0 * beta, niv).mat
    assert 0 < flagged.sum() < flagged.size
    assert all(r is None for r in results[1:])
    assert np.array_equal(results[0], single)
    assert np.allclose(results[0][flagged][..., inner], expected_inner[flagged], atol=1e-5)
    assert np.allclose(results[0][~flagged][..., inner], plain[~flagged][..., inner])
    assert np.allclose(results[0][..., ~inner], plain[..., ~inner])


def test_interpolate_sigma_without_flagged_momenta_equals_the_plain_interpolation():
    """interpolate_sigma returns the plain interpolation when no momentum needs a pole fit."""
    config.lattice.nk = (2, 2, 1)
    config.lattice.k_grid = bz.KGrid(config.lattice.nk, symmetries=bz.two_dimensional_square_symmetries())
    vn = MFHelper.vn(6, 1.0)
    signal = 1.0 - 1j * np.sign(vn) * (0.1 + 0.2 * np.abs(vn))
    sigma = SelfEnergy(np.broadcast_to(signal, (2, 2, 1, 1, 1, vn.size)).copy(), nk=(2, 2, 1), beta=1.0)

    result = nonlocal_sde.interpolate_sigma(sigma, beta_target=2.0, niv_target=6, comm=create_comm_mock())

    assert not sigma.pole_fit_mask().any()
    assert np.array_equal(result.mat, sigma.interpolate(2.0, 6).mat)


def _grid_config(nk=(2, 2, 1), niv_core=4, beta=10.0):
    """Sets the target grid the expander reads from the config."""
    config.lattice.k_grid = bz.KGrid(nk, symmetries=[])
    config.box.niv_core = niv_core
    config.sys.beta = beta
    config.sys.n_bands = 1


def test_jacobian_expander_is_the_identity_on_equal_grids():
    """Equal window, grid and temperature expand a stored column to the plain real vector."""
    _grid_config()
    window = (np.arange(32) + 1j).reshape(2, 2, 1, 1, 1, 8)
    state = {"shape": np.array([2, 2, 1, 1, 1, 8]), "beta": np.float64(10.0)}
    out = nonlocal_sde._jacobian_expander(state)(window.reshape(-1).astype(np.complex64))
    assert np.array_equal(out, to_vec(window))


def test_jacobian_expander_regrids_frequencies_and_zeroes_beyond_the_source_range():
    """A cooling step re-grids a linear column exactly onto the new frequencies and zeroes it above the source range."""
    _grid_config(nk=(1, 1, 1), niv_core=8, beta=20.0)
    state = {"shape": np.array([1, 1, 1, 1, 1, 8]), "beta": np.float64(10.0)}
    vn_src = MFHelper.vn(4, 10.0)
    column = (1.0 + 1j * vn_src).astype(np.complex64)  # PCHIP reproduces a linear function exactly
    out = to_mat(nonlocal_sde._jacobian_expander(state)(column), (1, 1, 1, 1, 1, 16))[0, 0, 0, 0, 0]
    vn_tgt = MFHelper.vn(8, 20.0)
    inside = np.abs(vn_tgt) <= np.abs(vn_src).max()
    assert np.allclose(out[inside], 1.0 + 1j * vn_tgt[inside], atol=1e-6)
    assert np.array_equal(out[~inside], np.zeros_like(out[~inside]))


def test_jacobian_expander_resamples_a_refined_grid():
    """A refined momentum grid is reached through interpolate_q_grid, doubling the peak momentum index."""
    _grid_config(nk=(4, 4, 1))
    state = {"shape": np.array([2, 2, 1, 1, 1, 8]), "beta": np.float64(10.0)}
    column = np.zeros((2, 2, 1, 1, 1, 8), dtype=np.complex64)
    column[1, 0] = 1.0
    out = nonlocal_sde._jacobian_expander(state)(column.reshape(-1))
    out_mat = to_mat(out, (4, 4, 1, 1, 1, 8))
    peak = np.unravel_index(np.argmax(out_mat.real.sum(axis=(2, 3, 4, 5))), (4, 4))
    assert out.shape == (2 * 16 * 8,) and peak == (2, 0)


def test_jacobian_expander_drops_a_column_that_leaves_this_runs_window(monkeypatch):
    """A re-gridded column of another length is dropped with a warning where it is built, not where it is read."""
    _grid_config(nk=(1, 1, 1), niv_core=8, beta=20.0)
    logger = MagicMock()
    monkeypatch.setattr(config, "logger", logger, raising=False)
    state = {"shape": np.array([1, 1, 1, 1, 1, 8]), "beta": np.float64(10.0)}
    monkeypatch.setattr(nonlocal_sde, "to_vec", lambda mat: np.zeros(4))
    column = (1.0 + 1j * MFHelper.vn(4, 10.0)).astype(np.complex64)

    assert nonlocal_sde._jacobian_expander(state)(column).size == 0
    assert any(
        "re-grids to 4 entries" in str(c.args[0]) and "it is dropped" in str(c.args[0])
        for c in logger.warning.call_args_list
    )


def test_jacobian_expander_refuses_a_different_orbital_count():
    """A predecessor with another orbital count cannot be carried."""
    _grid_config()
    config.logger = MagicMock()
    assert nonlocal_sde._jacobian_expander({"shape": np.array([2, 2, 1, 2, 2, 8]), "beta": np.float64(10.0)}) is None
    config.logger.warning.assert_called_once()


def test_carry_jacobian_spectrum_reads_the_predecessor_file(monkeypatch, tmp_path):
    """The predecessor's spectrum file is loaded, expanded and handed to the tracker with flips allowed."""
    config.logger = MagicMock()
    config.self_consistency.previous_sc_path = str(tmp_path)
    config.stabilization.use_chi_phys_restriction = False
    config.stabilization.use_lambda_correction = False
    state = {"converged": np.bool_(True), "lam_pi": np.zeros(0, dtype=np.complex128)}
    monkeypatch.setattr(nonlocal_sde.os.path, "isfile", lambda path: path.endswith(JACOBIAN_FILE))
    monkeypatch.setattr(nonlocal_sde, "load_spectrum", lambda path: state)
    monkeypatch.setattr(nonlocal_sde, "_jacobian_expander", lambda st: (lambda column: column))
    tracker = MagicMock()
    nonlocal_sde._carry_jacobian_spectrum(tracker, None)
    tracker.carry_in.assert_called_once()
    assert tracker.carry_in.call_args.kwargs["allow_flip"] is True
    assert tracker.carry_in.call_args.kwargs["beta"] == config.sys.beta
    config.logger.warning.assert_not_called()


def test_carry_jacobian_spectrum_warns_for_an_unconverged_predecessor_and_pauses_under_a_scaffold(
    monkeypatch, tmp_path
):
    """An unconverged predecessor is carried with a warning, and an active scaffold keeps the flip pending."""
    config.logger = MagicMock()
    config.self_consistency.previous_sc_path = str(tmp_path)
    config.stabilization.use_chi_phys_restriction = True
    config.stabilization.use_lambda_correction = False
    monkeypatch.setattr(nonlocal_sde.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(
        nonlocal_sde, "load_spectrum", lambda path: {"converged": np.bool_(False), "lam_pi": np.zeros(0, dtype=complex)}
    )
    monkeypatch.setattr(nonlocal_sde, "_jacobian_expander", lambda st: (lambda column: column))
    tracker = MagicMock()
    nonlocal_sde._carry_jacobian_spectrum(tracker, None)
    assert tracker.carry_in.call_args.kwargs["allow_flip"] is False
    config.logger.warning.assert_called_once()
    config.stabilization.use_chi_phys_restriction = False


def test_carry_jacobian_spectrum_skips_a_missing_file_and_a_refused_expansion(monkeypatch, tmp_path):
    """No file means nothing carried with an info line; a refused expander means nothing carried."""
    config.logger = MagicMock()
    config.self_consistency.previous_sc_path = str(tmp_path)
    tracker = MagicMock()
    monkeypatch.setattr(nonlocal_sde.os.path, "isfile", lambda path: False)
    nonlocal_sde._carry_jacobian_spectrum(tracker, None)
    tracker.carry_in.assert_not_called()
    monkeypatch.setattr(nonlocal_sde.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(
        nonlocal_sde, "load_spectrum", lambda path: {"converged": np.bool_(True), "lam_pi": np.zeros(0, dtype=complex)}
    )
    monkeypatch.setattr(nonlocal_sde, "_jacobian_expander", lambda st: None)
    nonlocal_sde._carry_jacobian_spectrum(tracker, None)
    tracker.carry_in.assert_not_called()


def test_save_jacobian_spectrum_passes_the_window_shape_and_the_path(tmp_path):
    """The loop's writer hands the window shape, beta and the output path to the tracker's writer."""
    config.output.output_path = str(tmp_path)
    config.lattice.k_grid = bz.KGrid((2, 2, 1), symmetries=[])
    config.box.niv_core = 3
    config.sys.beta = 10.0
    config.sys.n_bands = 1
    tracker = MagicMock()
    width = TRACKER_PAIRS - 1
    rows = ([np.ones(width, dtype=np.complex128)], [np.zeros(width)], [0.4])
    nonlocal_sde._save_jacobian_spectrum(tracker, True, *rows)
    args, kwargs = tracker.save_spectrum.call_args
    assert args == (os.path.join(str(tmp_path), JACOBIAN_FILE), (2, 2, 1, 1, 1, 6), 10.0, True, DTYPE)
    assert set(kwargs["traces"]) == {"eigenvalues", "eigenvalue_residuals", "damping"}
    assert np.array_equal(kwargs["traces"]["damping"], np.array([0.4]))
    assert kwargs["traces"]["eigenvalues"].shape == (1, width) and kwargs["traces"]["eigenvalue_residuals"].shape == (
        1,
        width,
    )


def test_append_jacobian_eigenvalues_writes_nan_rows_without_an_estimate(monkeypatch, tmp_path):
    """The writer appends an all-nan row while the tracker produced no estimate, and the tracker's row otherwise."""
    config.output.output_path = str(tmp_path)
    saved = {}
    monkeypatch.setattr(np, "savez_compressed", lambda path, **arrays: saved.setdefault(path, []).append(arrays))
    rows_lam, rows_res, rows_damping = [], [], []
    tracker = MagicMock()
    tracker.estimated = False
    tracker.p_eff = 0.2
    nonlocal_sde._append_jacobian_eigenvalues(tracker, rows_lam, rows_res, rows_damping)
    lam_known = np.array([2.0 + 1j, 1.5 - 0.5j, 0.5 + 0j, 0.2 + 0j, 0.1 - 0.1j, 0.1 + 0.1j])
    res_known = np.array([1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3])
    assert lam_known.size == TRACKER_PAIRS - 1
    tracker.estimated, tracker.p_eff = True, 0.1
    tracker.leading.return_value = (lam_known, res_known)
    nonlocal_sde._append_jacobian_eigenvalues(tracker, rows_lam, rows_res, rows_damping)
    path = os.path.join(config.output.output_path, JACOBIAN_FILE)
    assert list(saved) == [path] and len(saved[path]) == 2
    lam_rows, res_rows = saved[path][-1]["eigenvalues"], saved[path][-1]["eigenvalue_residuals"]
    assert np.isnan(lam_rows[0]).all() and np.isnan(res_rows[0]).all()
    assert np.array_equal(lam_rows[1], lam_known) and np.array_equal(res_rows[1], res_known)
    assert np.array_equal(saved[path][-1]["damping"], np.array([0.2, 0.1]))
    assert set(saved[path][-1]) == {"eigenvalues", "eigenvalue_residuals", "damping"}


def test_loop_writes_the_spectrum_file_at_the_end(monkeypatch, tmp_path):
    """With the flag on, the loop ends by saving the tracker's spectrum next to the mu history."""
    config.stabilization.use_jacobian_stabilization = True
    saved = []

    def step(sigma_in, n_call, annealer):
        return sigma_in.copy()

    # epsilon=0.0 forces non-convergence: a zero step residual is not below zero, so the identity map cannot converge
    run, _, _ = _setup_self_energy_loop(monkeypatch, tmp_path, step, max_iter=2, epsilon=0.0)
    monkeypatch.setattr(
        nonlocal_sde, "_save_jacobian_spectrum", lambda tracker, converged, *rows: saved.append(converged)
    )
    run()
    config.stabilization.use_jacobian_stabilization = False
    assert saved == [False]


def test_loop_writes_the_spectrum_file_as_converged(monkeypatch, tmp_path):
    """With the flag on, a run that reaches convergence saves the tracker's spectrum with the flag set."""
    config.stabilization.use_jacobian_stabilization = True
    saved = []

    def step(sigma_in, n_call, annealer):
        return sigma_in.copy()

    run, _, _ = _setup_self_energy_loop(monkeypatch, tmp_path, step, max_iter=2)
    monkeypatch.setattr(
        nonlocal_sde, "_save_jacobian_spectrum", lambda tracker, converged, *rows: saved.append(converged)
    )
    run()
    config.stabilization.use_jacobian_stabilization = False
    assert saved == [True]


def test_loop_writes_the_eigenvalue_rows_every_iteration(monkeypatch, tmp_path):
    """With the flag on, the loop calls the per-iteration eigenvalue writer once for every iteration."""
    config.stabilization.use_jacobian_stabilization = True

    def step(sigma_in, n_call, annealer):
        return sigma_in.copy()

    run, _, _ = _setup_self_energy_loop(monkeypatch, tmp_path, step, max_iter=2)
    append = MagicMock()
    monkeypatch.setattr(nonlocal_sde, "_append_jacobian_eigenvalues", append)
    monkeypatch.setattr(nonlocal_sde, "_save_jacobian_spectrum", lambda tracker, converged, *rows: None)
    run()
    config.stabilization.use_jacobian_stabilization = False
    assert append.call_count == 2


def test_loop_marks_a_scaffolded_phase_exit_as_not_converged(monkeypatch, tmp_path):
    """A restriction released on the final iteration exits without the pure branch, so the saved flag is False."""
    config.stabilization.use_jacobian_stabilization = True
    config.stabilization.use_chi_phys_restriction = True
    config.stabilization.use_lambda_correction = False
    config.stabilization.use_lambda_annealing = False
    saved = []

    def step(sigma_in, n_call, annealer):
        return sigma_in.copy()

    # relaxed threshold (10x epsilon) plus a zero step residual converges the identity map on the last iteration
    run, _, logger = _setup_self_energy_loop(monkeypatch, tmp_path, step, max_iter=2)
    monkeypatch.setattr(
        nonlocal_sde, "_save_jacobian_spectrum", lambda tracker, converged, *rows: saved.append(converged)
    )
    run()
    config.stabilization.use_jacobian_stabilization = False
    config.stabilization.use_chi_phys_restriction = False
    assert any("no unrestricted iterations remain" in str(c.args[0]) for c in logger.warning.call_args_list)
    assert saved == [False]


def test_loop_does_not_carry_the_spectrum_on_a_fresh_start(monkeypatch, tmp_path):
    """A fresh run starting at iteration zero never asks a predecessor for its spectrum."""
    config.stabilization.use_jacobian_stabilization = True
    carried = []

    def step(sigma_in, n_call, annealer):
        return sigma_in.copy()

    run, _, _ = _setup_self_energy_loop(monkeypatch, tmp_path, step, max_iter=1)
    monkeypatch.setattr(nonlocal_sde, "_carry_jacobian_spectrum", lambda tracker, annealer: carried.append(tracker))
    monkeypatch.setattr(nonlocal_sde, "_save_jacobian_spectrum", lambda tracker, converged, *rows: None)
    run()
    config.stabilization.use_jacobian_stabilization = False
    assert carried == []


def test_loop_carries_the_spectrum_when_resuming(monkeypatch, tmp_path):
    """A run that starts from a previous one asks for its spectrum before the first iteration."""
    config.stabilization.use_jacobian_stabilization = True
    carried = []

    def step(sigma_in, n_call, annealer):
        return sigma_in.copy()

    run, _, _ = _setup_self_energy_loop(monkeypatch, tmp_path, step, max_iter=1)
    # the harness leaves previous_sc_path empty; a starting iteration above zero is what triggers the carry-over
    monkeypatch.setattr(nonlocal_sde, "get_starting_sigma", lambda sigma: (sigma, 5))
    monkeypatch.setattr(nonlocal_sde, "_init_mu_history", lambda starting_iter: [0.5])
    monkeypatch.setattr(nonlocal_sde, "_carry_jacobian_spectrum", lambda tracker, annealer: carried.append(tracker))
    monkeypatch.setattr(nonlocal_sde, "_save_jacobian_spectrum", lambda tracker, converged, *rows: None)
    run()
    config.stabilization.use_jacobian_stabilization = False
    assert len(carried) == 1 and isinstance(carried[0], nonlocal_sde.JacobianTracker)


def _fake_previous_run(monkeypatch, tmp_path, iterate_iters, niv=4, subfolder=True, run_folder_iters=()):
    """Fakes the raw per-iteration sigma files of a previous run through glob and np.load."""
    folder = (
        os.path.join(str(tmp_path), config.self_consistency.sigma_iterates_subfolder_name)
        if subfolder
        else str(tmp_path)
    )
    files = {os.path.join(folder, f"sigma_dga_iteration_{it}.npy"): it for it in iterate_iters}
    files.update({os.path.join(str(tmp_path), f"sigma_dga_iteration_{it}.npy"): it for it in run_folder_iters})
    monkeypatch.setattr(nonlocal_sde.glob, "glob", lambda pattern: [f for f in files if fnmatch.fnmatch(f, pattern)])
    monkeypatch.setattr(np, "load", lambda f: np.full((1, 1, 1, 1, 1, 2 * niv), files[f], dtype=np.complex64))
    config.logger = MagicMock()
    config.self_consistency.previous_sc_path = str(tmp_path)
    config.self_consistency.use_interpolated_sigma = False
    config.self_consistency.mixing_history_length = 2
    config.box.niv_core = 2
    config.lattice.k_grid = bz.KGrid((1, 1, 1), symmetries=[])
    config.sys.beta = 10.0
    return files


def test_loop_saves_the_raw_proposal_only_with_jacobian_stabilization(monkeypatch, tmp_path):
    """The un-mixed proposal lands next to the mixed iterate in the iterates subfolder only with the tracking on."""

    def step(sigma_in, n_call, annealer):
        return sigma_in.copy()

    run, _, _ = _setup_self_energy_loop(monkeypatch, tmp_path, step, max_iter=2)
    save = MagicMock()
    monkeypatch.setattr(SelfEnergy, "save", save)
    monkeypatch.setattr(nonlocal_sde, "_save_jacobian_spectrum", lambda tracker, converged, *rows: None)
    run()
    untracked = [call.kwargs["name"] for call in save.call_args_list]
    save.reset_mock()
    config.stabilization.use_jacobian_stabilization = True
    run()
    tracked = [call.kwargs["name"] for call in save.call_args_list]
    assert "sigma_dga_iteration_1" in untracked and "sigma_dga_proposal_iteration_1" not in untracked
    assert "sigma_dga_iteration_1" in tracked and "sigma_dga_proposal_iteration_1" in tracked
    assert all(call.kwargs["output_dir"] == config.output.sigma_iterates_path for call in save.call_args_list)


def test_loop_saves_the_proposal_as_the_map_returned_it(monkeypatch, tmp_path):
    """The saved proposal is the raw map output, neither reflected nor mixed, and differs from the next iterate."""
    run, _, _ = _setup_self_energy_loop(
        monkeypatch, tmp_path, _affine_core_step(_STABLE_GAINS), max_iter=3, epsilon=0.0
    )
    config.stabilization.use_jacobian_stabilization = True
    config.sys.n_bands = 1
    proposal_map, save_iteration, mixing = (
        nonlocal_sde.calculate_sigma_proposal,
        nonlocal_sde._save_sigma_iteration,
        nonlocal_sde.apply_mixing_strategy,
    )
    produced, saved, mixed_in, mixed_out = [], [], [], []

    def proposal_spy(*args, **kwargs):
        sigma = proposal_map(*args, **kwargs)
        produced.append(sigma.mat.reshape(-1).copy())
        return sigma

    def save_spy(sigma, base_name, current_iter):
        if base_name == "sigma_dga_proposal":
            saved.append(sigma.mat.reshape(-1).copy())
        return save_iteration(sigma, base_name, current_iter)

    def mixing_spy(sigma_new, *args, **kwargs):
        mixed_in.append(sigma_new.mat.reshape(-1).copy())
        mixed = mixing(sigma_new, *args, **kwargs)
        mixed_out.append(mixed.mat.reshape(-1).copy())
        return mixed

    monkeypatch.setattr(nonlocal_sde, "calculate_sigma_proposal", proposal_spy)
    monkeypatch.setattr(nonlocal_sde, "_save_sigma_iteration", save_spy)
    monkeypatch.setattr(nonlocal_sde, "apply_mixing_strategy", mixing_spy)
    reflect = MagicMock(side_effect=lambda proposal, iterate: np.zeros_like(proposal))
    monkeypatch.setattr(nonlocal_sde.JacobianTracker, "active", property(lambda self: True))
    monkeypatch.setattr(nonlocal_sde.JacobianTracker, "reflect", reflect)
    monkeypatch.setattr(nonlocal_sde, "_save_jacobian_spectrum", lambda tracker, converged, *rows: None)
    run()

    assert len(saved) == 3 and len(produced) == 3 and reflect.call_count == 3
    assert all(np.array_equal(s, p) for s, p in zip(saved, produced))
    assert all(np.array_equal(s, m) for s, m in zip(saved, mixed_in))
    assert not any(np.array_equal(s, m) for s, m in zip(saved, mixed_out))


def test_get_starting_sigma_reads_the_iterates_subfolder_before_the_run_folder(monkeypatch, tmp_path):
    """The highest iterate of the Sigma_Iterates subfolder is the primary read path; run-folder files never win."""
    _fake_previous_run(monkeypatch, tmp_path, [3, 9], run_folder_iters=(4, 12))
    default = SelfEnergy(np.zeros((1, 1, 1, 1, 1, 8), dtype=np.complex64), (1, 1, 1), beta=10.0)

    sigma, starting_iter = nonlocal_sde.get_starting_sigma(default)

    # the loaded array carries the iteration number of the file it came from
    assert starting_iter == 9
    assert np.array_equal(sigma.mat, np.full(sigma.mat.shape, 9.0, dtype=sigma.mat.dtype))


def test_get_starting_sigma_falls_back_to_files_in_the_run_folder(monkeypatch, tmp_path):
    """A run written before the Sigma_Iterates subfolder existed is still resumed from its run folder."""
    _fake_previous_run(monkeypatch, tmp_path, [2, 4], subfolder=False)
    default = SelfEnergy(np.zeros((1, 1, 1, 1, 1, 8), dtype=np.complex64), (1, 1, 1), beta=10.0)
    assert nonlocal_sde.get_starting_sigma(default)[1] == 4


def test_get_starting_sigma_without_previous_run_returns_default(monkeypatch, tmp_path):
    """Without a usable previous run the DMFT self-energy starts the loop at iteration zero."""
    _fake_previous_run(monkeypatch, tmp_path, [])
    default = SelfEnergy(np.zeros((1, 1, 1, 1, 1, 8), dtype=np.complex64), (1, 1, 1), beta=10.0)
    assert nonlocal_sde.get_starting_sigma(default) == (default, 0)


def test_loop_threads_one_persistent_lambda_dict_to_every_proposal(monkeypatch, tmp_path):
    """The per-iteration lambda correction gets one dict for the whole run, while the one-shot correction gets None."""
    run, _, _ = _setup_self_energy_loop(monkeypatch, tmp_path, lambda s, n, a: s.copy(), max_iter=2, epsilon=0.0)
    fake, seen = nonlocal_sde.calculate_sigma_proposal, []

    def spy(*args, **kwargs):
        seen.append(kwargs["lambda_previous"])
        return fake(*args, **kwargs)

    monkeypatch.setattr(nonlocal_sde, "calculate_sigma_proposal", spy)
    config.stabilization.use_lambda_correction = True
    run()
    config.lambda_correction.perform_lambda_correction = True
    run()

    assert seen[0] == {} and seen[0] is seen[1]
    assert seen[2] is None and seen[3] is None


def _drifting_pairs(value: float, k: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Three (iterate, proposal) pairs of a (1, 1, 1, 1, 1, 2) window, lambda_Pi = value on the first coordinate."""
    x = np.array([4.0 * 0.7**k, 4.0 * 0.7**k], dtype=np.complex128)
    gains = np.diag([1.0 - value, 0.5])
    iterates, proposals = [], []
    for _ in range(3):
        proposal = (gains @ x.real).astype(np.complex128)
        iterates.append(x.reshape(1, 1, 1, 1, 1, 2).copy())
        proposals.append(proposal.reshape(1, 1, 1, 1, 1, 2).copy())
        x = x + 0.07 * (proposal - x)
    return iterates, proposals


def test_the_predecessor_columns_reach_the_saved_spectrum_of_a_carried_run(tmp_path):
    """A carried run matches its final certified vectors against the predecessor's columns and writes lam_prev."""
    _grid_config(nk=(1, 1, 1), niv_core=1, beta=10.0)
    config.logger = MagicMock()
    config.self_consistency.previous_sc_path = str(tmp_path / "rung1")
    config.output.output_path = str(tmp_path / "rung2")
    (tmp_path / "rung1").mkdir()
    (tmp_path / "rung2").mkdir()
    first = JacobianTracker(0.4)
    for k in range(2):
        first.update(*_drifting_pairs(0.09, k))
    config.output.output_path = str(tmp_path / "rung1")
    nonlocal_sde._save_jacobian_spectrum(first, True, [], [], [])
    config.output.output_path = str(tmp_path / "rung2")
    second = JacobianTracker(0.4, logger=config.logger)
    nonlocal_sde._carry_jacobian_spectrum(second, None)
    for k in range(2):
        second.update(*_drifting_pairs(0.03, k))
    nonlocal_sde._save_jacobian_spectrum(second, True, [], [], [])
    saved = load_spectrum(str(tmp_path / "rung2" / JACOBIAN_FILE))
    stored = saved["stored"]
    assert np.allclose(saved["lam_pi"][stored], [0.03], atol=1e-8)
    assert np.allclose(saved["lam_prev"][stored], [0.09], atol=1e-8) and saved["beta_prev"][stored][0] == 10.0
    assert np.isnan(saved["lam_prev"][~stored]).all()


def test_a_carried_run_logs_the_cross_rung_pre_flip(tmp_path):
    """A predecessor file carrying two rungs of a falling mode makes the rung pre-flip it and log the prediction."""
    _grid_config(nk=(1, 1, 1), niv_core=4, beta=25.0)
    config.logger = MagicMock()
    config.self_consistency.previous_sc_path = str(tmp_path)
    shape = (1, 1, 1, 1, 1, 8)
    state = {
        "lam_pi": np.array([0.03 + 0.0j]),
        "res": np.zeros(1),
        "flip": np.zeros(1, dtype=bool),
        "predicted": np.zeros(1, dtype=bool),
        "stored": np.ones(1, dtype=bool),
        "lam_prev": np.array([0.09 + 0.0j]),
        "res_prev": np.zeros(1),
        "beta_prev": np.array([15.0]),
        "u_re": np.ones((8, 1), dtype=np.complex64),
        "u_im": np.zeros((8, 1), dtype=np.complex64),
        "shape": np.asarray(shape, dtype=np.int64),
        "beta": np.float64(20.0),
        "p_eff": np.float64(0.4),
        "converged": np.bool_(True),
    }
    np.savez_compressed(str(tmp_path / JACOBIAN_FILE), **state)
    tracker = JacobianTracker(0.4, logger=config.logger)
    nonlocal_sde._carry_jacobian_spectrum(tracker, None)
    assert tracker.q is not None and tracker.q.shape[1] == 1 and tracker._flip.tolist() == [True]
    assert any("cross-rung prediction" in str(c.args[0]) for c in config.logger.info.call_args_list)


def _constant_direction_pairs(value: float, k: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Three pairs of an eight-frequency window with lambda_Pi = value on the frequency-constant direction."""
    v = np.ones(8) / np.sqrt(8.0)
    gains = 0.5 * np.eye(8) + (0.5 - value) * np.outer(v, v)
    x = (4.0 * 0.7**k) * (v + 0.3 * np.eye(8)[:, 1]).astype(np.complex128)
    iterates, proposals = [], []
    for _ in range(3):
        proposal = (gains @ x.real).astype(np.complex128)
        iterates.append(x.reshape(1, 1, 1, 1, 1, 8).copy())
        proposals.append(proposal.reshape(1, 1, 1, 1, 1, 8).copy())
        x = x + 0.07 * (proposal - x)
    return iterates, proposals


def test_the_predecessor_match_survives_the_frequency_re_gridding_of_a_cooling_rung(tmp_path):
    """A column re-gridded from beta 10 to beta 20 still matches the rung's own vector, so lam_prev is written."""
    _grid_config(nk=(1, 1, 1), niv_core=4, beta=10.0)
    config.logger = MagicMock()
    config.self_consistency.previous_sc_path = str(tmp_path / "rung1")
    (tmp_path / "rung1").mkdir()
    (tmp_path / "rung2").mkdir()
    first = JacobianTracker(0.4)
    for k in range(2):
        first.update(*_constant_direction_pairs(0.09, k))
    config.output.output_path = str(tmp_path / "rung1")
    nonlocal_sde._save_jacobian_spectrum(first, True, [], [], [])
    config.sys.beta = 20.0
    config.output.output_path = str(tmp_path / "rung2")
    second = JacobianTracker(0.4, logger=config.logger)
    nonlocal_sde._carry_jacobian_spectrum(second, None)
    for k in range(2):
        second.update(*_constant_direction_pairs(0.03, k))
    nonlocal_sde._save_jacobian_spectrum(second, True, [], [], [])
    saved = load_spectrum(str(tmp_path / "rung2" / JACOBIAN_FILE))
    stored = saved["stored"]
    assert np.allclose(saved["lam_pi"][stored], [0.03], atol=1e-8)
    assert np.allclose(saved["lam_prev"][stored], [0.09], atol=1e-8) and saved["beta_prev"][stored][0] == 10.0


def test_carry_jacobian_spectrum_skips_an_unfinished_predecessor_file(tmp_path):
    """A jacobian.npz holding only the per-iteration traces, the mark of a run that never ended, carries nothing."""
    config.logger = MagicMock()
    config.self_consistency.previous_sc_path = str(tmp_path)
    np.savez_compressed(
        str(tmp_path / JACOBIAN_FILE),
        eigenvalues=np.full((2, 3), np.nan, dtype=np.complex128),
        eigenvalue_residuals=np.full((2, 3), np.nan),
        damping=np.array([0.4, 0.4]),
    )
    tracker = MagicMock()
    nonlocal_sde._carry_jacobian_spectrum(tracker, None)
    tracker.carry_in.assert_not_called()
    assert any("no certified spectrum" in str(c.args[0]) for c in config.logger.info.call_args_list)


def test_carry_jacobian_spectrum_reads_only_the_jacobian_file(tmp_path):
    """A folder holding a file under any other name, the former jacobian_spectrum.npz included, carries nothing."""
    config.logger = MagicMock()
    config.self_consistency.previous_sc_path = str(tmp_path)
    np.savez_compressed(str(tmp_path / "jacobian_spectrum.npz"), lam_pi=np.array([-0.5 + 0j]), converged=np.bool_(True))
    tracker = MagicMock()
    nonlocal_sde._carry_jacobian_spectrum(tracker, None)
    tracker.carry_in.assert_not_called()
    assert any(f"No {JACOBIAN_FILE}" in str(c.args[0]) for c in config.logger.info.call_args_list)


def test_max_compound_norm_reads_the_requested_bosonic_frequency():
    """The largest compound spectral norm over momenta is read at the static and at the first bosonic frequency."""
    rng = np.random.default_rng(5)
    comp = rng.standard_normal((3, 2, 4, 4)) + 1j * rng.standard_normal((3, 2, 4, 4))
    chi = _chi_from_compound(comp, 2)
    for w in (0, 1):
        expected = np.linalg.norm(comp[:, w], ord=2, axis=(-2, -1)).max()
        assert np.allclose(nonlocal_sde.max_compound_norm(chi, w), expected, atol=1e-5)
    assert nonlocal_sde.max_compound_norm(chi, 2) == 0.0


def test_monitor_chi_phys_warns_when_the_first_frequency_exceeds_the_static_value(monkeypatch):
    """A first-frequency susceptibility above twice the static one warns, a decreasing one does not."""
    logger = MagicMock()
    monkeypatch.setattr(config, "logger", logger, raising=False)
    healthy = _chi_from_compound(np.array([[[[0.10]], [[0.08]]], [[[0.05]], [[-0.07]]]], dtype=complex), 1)
    nonlocal_sde._monitor_chi_phys(healthy, _single_rank_dist())
    logger.warning.assert_not_called()
    broken = _chi_from_compound(np.array([[[[0.10]], [[0.08]]], [[[0.05]], [[-5.3]]]], dtype=complex), 1)
    nonlocal_sde._monitor_chi_phys(broken, _single_rank_dist())
    assert "first bosonic frequency" in logger.warning.call_args.args[0]


def test_monitor_chi_phys_logs_the_smallest_static_inverse_eigenvalue(monkeypatch):
    """The smallest eigenvalue of the static inverse compound susceptibility over all momenta is logged."""
    logger = MagicMock()
    monkeypatch.setattr(config, "logger", logger, raising=False)
    chi = _chi_from_compound(np.array([[[[0.10]], [[0.08]]], [[[0.05]], [[0.01]]]], dtype=complex), 1)
    nonlocal_sde._monitor_chi_phys(chi, _single_rank_dist())
    lines = [str(call.args[0]) for call in logger.info.call_args_list]
    assert any("eigenvalue of chi_phys (dens): 0.050000, of 1/chi_phys: 10.000000." in line for line in lines)


def _single_band_sigma(nk, niv, beta, seed):
    """A decaying single-band self-energy on a k-grid (compressed momentum layout) and a dispersion for it."""
    rng = np.random.default_rng(seed)
    iv = 1j * MFHelper.vn(niv, beta)
    mat = 0.2 * rng.standard_normal((int(np.prod(nk)), 1, 1, 2 * niv)) + 0.3 + 1.0 / iv
    ek = rng.standard_normal((*nk, 1, 1))
    return SelfEnergy(mat.astype(np.complex128), nk=nk, has_compressed_q_dimension=True, beta=beta), ek


def test_first_frequency_pole_ratio_is_the_local_vertex_times_the_q0_bubble_element(monkeypatch):
    """R = gamma_loc beta Re<G(k, pi T) G(k, -pi T)>_k, from the iterate's own nu = +-pi T values."""
    beta, mu, gamma_loc = 10.0, 0.4, 0.07
    sigma, ek = _single_band_sigma((4, 4, 1), 8, beta, seed=5)
    monkeypatch.setattr(config.sys, "beta", beta)
    monkeypatch.setattr(config.lattice, "hamiltonian", SimpleNamespace(get_ek=lambda: ek))
    nu0, s = np.pi / beta, sigma.mat[:, 0, 0]
    g_plus = 1.0 / (1j * nu0 + mu - ek.reshape(-1) - s[:, 8])
    g_minus = 1.0 / (-1j * nu0 + mu - ek.reshape(-1) - s[:, 7])
    expected = gamma_loc * beta * np.mean(g_plus * g_minus).real
    assert np.isclose(nonlocal_sde.first_frequency_pole_ratio(sigma, mu, gamma_loc), expected, rtol=1e-12)

    # a momentum-local self-energy (the DMFT start) enters every momentum alike
    local = SelfEnergy(sigma.mat[:1].copy(), nk=(1, 1, 1), has_compressed_q_dimension=True, beta=beta)
    tiled = SelfEnergy(np.repeat(sigma.mat[:1], 16, axis=0), nk=(4, 4, 1), has_compressed_q_dimension=True, beta=beta)
    assert np.isclose(
        nonlocal_sde.first_frequency_pole_ratio(local, mu, gamma_loc),
        nonlocal_sde.first_frequency_pole_ratio(tiled, mu, gamma_loc),
        rtol=1e-12,
    )


def _write_npy(path, arr):
    """Writes a real .npy file (the autouse fixture turns np.save into a no-op)."""
    with open(path, "wb") as f:
        np.lib.format.write_array(f, np.asarray(arr))


def test_first_frequency_local_vertex_reads_the_saved_local_files(monkeypatch, tmp_path):
    """gamma_loc = 1/gchi_loc(w1; pi T, pi T) - 1/(-beta G(pi T) G(-pi T)) from gchi_dens_loc.npy and g_dmft.npy;
    None for more than one band."""
    beta, niw, niv, niv_dmft = 12.5, 3, 4, 10
    monkeypatch.setattr(config.sys, "beta", beta)
    monkeypatch.setattr(config.box, "niw_core", niw)
    monkeypatch.setattr(config.output, "output_path", str(tmp_path))
    rng = np.random.default_rng(2)
    gchi = rng.standard_normal((1, 1, 1, 1, niw + 1, 2 * niv, 2 * niv)) + 1j * rng.standard_normal(
        (1,) * 4 + (niw + 1,) + (2 * niv,) * 2
    )
    g = rng.standard_normal((1, 1, 1, 1, 1, 2 * niv_dmft)) + 1j * rng.standard_normal((1, 1, 1, 1, 1, 2 * niv_dmft))
    _write_npy(tmp_path / "gchi_dens_loc.npy", gchi)  # np.save is mocked in every test
    _write_npy(tmp_path / "g_dmft.npy", g)
    expected = 1.0 / gchi[0, 0, 0, 0, 1, niv, niv] - 1.0 / (
        -beta * g[0, 0, 0, 0, 0, niv_dmft] * g[0, 0, 0, 0, 0, niv_dmft - 1]
    )
    assert np.isclose(nonlocal_sde.first_frequency_local_vertex(), expected.real, rtol=1e-12)

    _write_npy(tmp_path / "gchi_dens_loc.npy", np.zeros((2, 2, 2, 2, niw + 1, 2 * niv, 2 * niv)))
    assert nonlocal_sde.first_frequency_local_vertex() is None


def test_loop_logs_the_first_frequency_pole_ratio_of_every_evaluated_iterate(monkeypatch, tmp_path):
    """Rank 0 logs R of the iterate each proposal evaluates and warns once R reaches 1."""
    run, calls, logger = _setup_self_energy_loop(
        monkeypatch, tmp_path, lambda s, n, a: s.copy(), max_iter=2, epsilon=0.0
    )
    monkeypatch.setattr(nonlocal_sde, "first_frequency_local_vertex", lambda: 0.07)
    ratios = iter([0.95, 1.02])
    seen = []

    def ratio(sigma, mu, gamma_loc):
        seen.append((mu, gamma_loc, len(calls)))
        return next(ratios)

    monkeypatch.setattr(nonlocal_sde, "first_frequency_pole_ratio", ratio)
    run()
    assert seen == [(0.5, 0.07, 0), (0.5, 0.07, 1)]  # measured before each proposal, at the iterate's mu
    infos = [str(c.args[0]) for c in logger.info.call_args_list]
    warnings = [str(c.args[0]) for c in logger.warning.call_args_list]
    assert any("First-frequency density pole ratio R = 0.9500" in line for line in infos)
    assert sum("R = 1.0200" in line and "pole ring" in line for line in warnings) == 1

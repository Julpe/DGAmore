# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import gc
import os
import threading
import tracemalloc
import types
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
import scipy.sparse.linalg as spla

import dgamore.brillouin_zone as bz
import dgamore.n_point_base as npb
from dgamore import config, dga_io, four_point, memory_estimator, mpi_utils, nonlocal_sde
from dgamore.dga_logger import DgaLogger
from dgamore.four_point import FourPoint
from dgamore.local_four_point import LocalFourPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore.n_point_base import SpinChannel, deferred_collection
from dgamore.greens_function import GreensFunction, update_mu
from dgamore.interaction import Interaction, LocalInteraction
from dgamore.jacobian_stabilization import _RITZ_GATE, JacobianTracker, _schur_ritz
from dgamore.mpi_utils import MpiDistributor
from dgamore.self_energy import SelfEnergy
from dgamore.sigma_jacobian import (
    _CHANNELS,
    _CHECK_BASIS,
    _F_DC,
    _INVERSE_ORDER,
    ExactJacobian,
    _block_extension,
    _merge_modes,
    _squared_box_sum,
    _thick_restart,
    certify_subspace,
    leading_eigenpairs,
    sector_projector,
    tracker_sector_maps,
)
from tests import conftest

pytestmark = pytest.mark.usefixtures("one_blas_thread")


def _p_orbital_grid(nk: tuple[int, int, int] = (4, 4, 1)) -> tuple[bz.KGrid, np.ndarray]:
    """The auto-symmetry grid of the p_x/p_y dispersion, returned with the dispersion."""
    hk = conftest.p_orbital_hk(nk)
    grid = bz.KGrid(nk=nk, symmetries=[bz.KnownSymmetries.AUTO])
    grid.specify_auto_symmetries(hk)
    return grid, hk


def _complex_hopping_hk(nk: tuple[int, int, int]) -> np.ndarray:
    """The p_x/p_y bands of unequal orbitals joined by the real hoppings 0.3 + 0.4 e^{ik_x}, so H(-k) = H(k)^*."""
    kx = 2 * np.pi * np.arange(nk[0]) / nk[0]
    hk = conftest.p_orbital_hk(nk, equal_orbitals=False)
    hk[..., 0, 1] = (0.3 + 0.4 * np.exp(1j * kx))[:, None, None]
    hk[..., 1, 0] = np.conj(hk[..., 0, 1])
    return hk


def _nearest_neighbor_v(nk: tuple[int, int, int], v: float) -> Interaction:
    """The nearest-neighbor density-density interaction V(q) = 2 v (cos q_x + cos q_y) between every orbital pair."""
    qx, qy, _ = np.meshgrid(*(2 * np.pi * np.arange(n) / n for n in nk), indexing="ij")
    vq = np.zeros((*nk, 2, 2, 2, 2), dtype=complex)
    for a in range(2):
        for b in range(2):
            vq[..., a, a, b, b] = 2.0 * v * (np.cos(qx) + np.cos(qy))
    return Interaction(vq, SpinChannel.NONE, nk)


@pytest.fixture
def loop_state(monkeypatch):
    """The fresh-start loop state of the two-band end_2_end fixture in complex128, on the box of its stored vertices."""
    return _loop_state(monkeypatch)


def _first_orbital(s_dmft: SelfEnergy, u_loc: LocalInteraction, folder) -> tuple[SelfEnergy, LocalInteraction]:
    """
    Restricts the loaded fixture to its first orbital: DMFT self-energy, dispersion, U and the stored vertices, which
    are written to ``folder`` and read from there, so held and per-window (streamed) loads both get one band.
    """
    assert not os.path.samefile(folder, config.output.output_path)  # never overwrite the fixture's own vertices
    for name in ("gamma_dens_loc.npy", "gamma_magn_loc.npy", _F_DC[0]):
        first = np.load(os.path.join(config.output.output_path, name))[:1, :1, :1, :1]
        with open(os.path.join(folder, name), "wb") as file:
            np.lib.format.write_array(file, np.ascontiguousarray(first))
    config.output.output_path = str(folder)
    config.sys.n_bands = 1
    config.sys.occ_dmft = config.sys.occ_dmft[:1, :1]
    config.lattice.hamiltonian.set_ek(config.lattice.hamiltonian.get_ek()[..., :1, :1].copy())
    s_one = SelfEnergy(s_dmft.mat[..., :1, :1, :].copy(), (1, 1, 1), True, False, beta=config.sys.beta)
    return s_one, LocalInteraction(u_loc.mat[:1, :1, :1, :1].copy())


def _loop_state(monkeypatch, lattice: str = "fixture", tmp_path=None) -> types.SimpleNamespace:
    """
    The end_2_end fixture's fresh-start loop state on its own, the p_x/p_y, complex-hopping or one-band lattice, or
    the one band as the correlated orbital of the Emery dispersion; the one-band lattices write their vertices to
    ``tmp_path``.
    """
    monkeypatch.setattr(npb, "DTYPE", np.complex128)
    monkeypatch.setattr(gc, "collect", lambda *args, **kwargs: 0)
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/end_2_end"
    comm = conftest.create_comm_mock()
    monkeypatch.setattr("mpi4py.MPI.COMM_WORLD", comm)
    monkeypatch.setattr(config, "logger", DgaLogger(comm, "./"), raising=False)
    conftest.create_default_config(config, folder)
    config.box.niw_core, config.box.niv_core, config.box.niv_shell = 20, 20, 10

    _, s_dmft, _, _ = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())
    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]
    config.output.output_path = folder
    u_loc = config.lattice.hamiltonian.get_local_u()
    if lattice in ("one_band", "one_band_hybridized"):
        s_dmft, u_loc = _first_orbital(s_dmft, u_loc, tmp_path)
    if lattice == "one_band_hybridized":
        config.lattice.hamiltonian.set_ek(conftest.emery_hk(config.lattice.nk))
        config.sys.correlated_orbitals = np.array([0])
        config.lattice.k_grid = bz.KGrid(config.lattice.nk, symmetries=[bz.KnownSymmetries.AUTO])
        config.lattice.k_grid.specify_auto_symmetries(config.lattice.hamiltonian.get_ek(), orbitals=np.array([0]))
    if lattice == "p_orbitals":
        # unequal orbitals keep the diagonal mirrors, which swap the two, out of the lattice's symmetries
        config.lattice.hamiltonian.set_ek(conftest.p_orbital_hk(config.lattice.nk, equal_orbitals=False))
    elif lattice == "complex_hopping":
        config.lattice.hamiltonian.set_ek(_complex_hopping_hk(config.lattice.nk))
    if lattice in ("p_orbitals", "complex_hopping"):
        config.lattice.k_grid = bz.KGrid(config.lattice.nk, symmetries=[bz.KnownSymmetries.AUTO])
        config.lattice.k_grid.specify_auto_symmetries(config.lattice.hamiltonian.get_ek())
    k_grid = config.lattice.k_grid
    dist_irrk = MpiDistributor.create_distributor(ntasks=k_grid.nk_irr, comm=comm, name="Q")
    dist_fullbz = MpiDistributor.create_distributor(ntasks=k_grid.nk_tot, comm=comm, name="FBZ")
    niv_cut = min(config.box.niw_core + config.box.niv_full + 10, config.box.niv_dmft)
    ek = config.lattice.hamiltonian.get_ek()
    start = s_dmft.concatenate_self_energies(s_dmft)
    nb, o = ek.shape[-1], config.sys.correlated_orbitals
    occ_k = GreensFunction.get_g_full(start, config.sys.mu, ek.reshape(-1, 1, 1, nb, nb), config.sys.beta, o)
    config.sys.n, config.sys.occ, config.sys.occ_k = nonlocal_sde._assemble_occupation(
        occ_k.get_fill_nonlocal()[2], dist_fullbz, o
    )
    sigma_dmft = s_dmft.cut_niv(niv_cut)
    v_nonloc = config.lattice.hamiltonian.get_vq(k_grid)
    if lattice == "complex_hopping":
        v_nonloc = _nearest_neighbor_v(k_grid.nk, 0.3)
    elif lattice in ("one_band", "one_band_hybridized"):
        v_nonloc = Interaction(v_nonloc.mat[..., :1, :1, :1, :1].copy(), SpinChannel.NONE, k_grid.nk)
    return types.SimpleNamespace(
        comm=comm,
        u_loc=u_loc,
        v_irr=v_nonloc.reduce_q(k_grid.get_irrq_list()),
        v_full=v_nonloc,
        sigma_dmft=sigma_dmft,
        sigma_dmft_full=s_dmft,
        delta_sigma=sigma_dmft.cut_niv(config.box.niv_core) - sigma_dmft.cut_niv(config.box.niv_core),
        irr_q_list=k_grid.get_irrq_list(),
        dist_irrk=dist_irrk,
        dist_fullbz=dist_fullbz,
        ek=ek,
    )


def _proposal(state, sigma, mu: float):
    """The raw proposal at the occupations held in config.sys."""
    return nonlocal_sde.calculate_sigma_proposal(
        sigma.copy(),
        mu,
        state.u_loc,
        state.v_irr,
        state.v_full,
        state.sigma_dmft,
        state.delta_sigma,
        state.irr_q_list,
        state.dist_irrk,
        state.comm,
        2,
    ).compress_q_dimension()


def _mu_and_occupations(state, sigma, mu_start: float, fixed: bool = False) -> float:
    """The held filling's chemical potential, or ``mu_start`` if ``fixed``, and in config.sys the occupations at it."""
    s = sigma.copy().compress_q_dimension()
    mu = mu_start
    if not fixed:
        smom0, orbitals = s.fit_smom()[0], config.sys.correlated_orbitals
        mu = update_mu(mu_start, config.sys.n, state.ek, s.mat, config.sys.beta, smom0, tol=1e-14, orbitals=orbitals)
    _, config.sys.occ, config.sys.occ_k, _, _ = nonlocal_sde._update_occ_and_energies_distributed(
        s, state.sigma_dmft_full, state.dist_fullbz, mu
    )
    return mu


def _with_core(sigma, window: np.ndarray):
    """A copy of ``sigma`` whose core window is ``window`` (compressed momentum axis)."""
    out = sigma.copy().compress_q_dimension()
    niv_core = config.box.niv_core
    out.mat[..., out.niv - niv_core : out.niv + niv_core] = window
    return out


def _core(sigma) -> np.ndarray:
    """The core window of ``sigma`` (compressed momentum axis)."""
    niv_core = config.box.niv_core
    return sigma.compress_q_dimension().mat[..., sigma.niv - niv_core : sigma.niv + niv_core]


def _sector_maps():
    """The weighted sector maps of the configured loop (see tracker_sector_maps)."""
    return tracker_sector_maps(config.lattice.k_grid, config.box.niv_core, config.sys.n_bands, config.sys.beta)


def _sector(window: np.ndarray) -> np.ndarray:
    """The sector vector of a core window in the tracker's weighted coordinates."""
    return _sector_maps()[0](window)


def _window(y: np.ndarray) -> np.ndarray:
    """The core window (compressed momentum axis) a sector vector unfolds to."""
    return _sector_maps()[1](y)


def _start(state, mu: float | None = None) -> tuple[SelfEnergy, float]:
    """The proposal of the DMFT start and the chemical potential of the held filling at it, or ``mu`` when given."""
    base = _proposal(state, state.sigma_dmft, config.sys.mu)
    return base, _mu_and_occupations(state, base, config.sys.mu if mu is None else mu, fixed=mu is not None)


def _jacobian(state, base, mu0: float, budgets: memory_estimator.ChunkBudgets | None = None) -> ExactJacobian:
    """The exact Jacobian of the loop map at ``base`` on the state's own distributors and communicator."""
    held = (state.u_loc, state.v_irr, state.v_full, state.sigma_dmft_full, state.dist_irrk, state.dist_fullbz)
    return ExactJacobian(base, mu0, *held, state.comm, budgets)


def _rank_jacobian(state, base, mu0: float, budgets, comm, dist_irrk, dist_fullbz) -> ExactJacobian:
    """The exact Jacobian on a rank's communicator and distributors, from its own V(q) and copies of the inputs."""
    # one V(q) per fake rank, as every MPI rank holds its own: reduce_q and find_q park their source's mat at None
    # while they clone it (_clone_without_mat), so ranks sharing one object race on it
    v_full = state.v_full.copy()
    v_irr = v_full.reduce_q(state.irr_q_list[dist_irrk.my_slice])
    held = (state.u_loc, v_irr, v_full, state.sigma_dmft_full.copy(), dist_irrk, dist_fullbz)
    return ExactJacobian(base.copy(), mu0, *held, comm, budgets)


@pytest.fixture
def quiet_logger(monkeypatch):
    """A MagicMock in place of config.logger."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)


@pytest.mark.parametrize("lattice", ["fixture", "p_orbitals", "complex_hopping", "one_band", "one_band_hybridized"])
def test_exact_jacobian_matches_central_differences_of_the_loop_map(lattice, monkeypatch, tmp_path):
    """J y equals the central difference of the loop's map along a reachable direction, mu and occupations included."""
    state = _loop_state(monkeypatch, lattice, tmp_path)
    base, mu0 = _start(state)
    jac = _jacobian(state, base, mu0)
    assert jac._tr_average == (lattice not in ("one_band", "one_band_hybridized"))
    x0 = _core(base).copy()
    y = jac.project(np.random.default_rng(3).standard_normal(jac.n_real))
    dx = np.reshape(_window(y), x0.shape)
    y *= 1e-2 * np.linalg.norm(x0) / np.linalg.norm(dx)
    dx = np.reshape(_window(y), x0.shape)
    assert np.allclose(_sector(dx), y, atol=1e-14 * np.abs(y).max())

    jy = jac.matvec(y)
    jac.free()

    h = 1e-5  # five-point difference: 1e-9 of max |fd| on the fixture (two-point 3e-6), rounding at most 3e-8
    sides = {}
    for step in (2, 1, -1, -2):
        sigma = _with_core(base, x0 + step * h * dx)
        sides[step] = _core(_proposal(state, sigma, _mu_and_occupations(state, sigma, mu0)))
    fd = _sector((8 * (sides[1] - sides[-1]) - (sides[2] - sides[-2])) / (12 * h))

    assert np.allclose(jy, fd, rtol=0, atol=1e-7 * np.abs(fd).max())


def _make_matrix_operator(a: np.ndarray, comm) -> types.SimpleNamespace:
    """A fixed real matrix as a collective operator whose every product broadcasts its vector or block from rank 0."""

    def product(y):
        y = comm.bcast(y, root=0)
        return a @ y if comm.rank == 0 else None

    return types.SimpleNamespace(
        n_real=a.shape[0],
        matvec=MagicMock(side_effect=product),
        matvec_block=MagicMock(side_effect=product),
        project=lambda y: y,
    )


def _normal_matrix(n: int, seed: int) -> np.ndarray:
    """A real normal matrix with a leading real value, a complex pair, a stiff negative value and a small bulk."""
    rng = np.random.default_rng(seed)
    blocks = np.zeros((n, n))
    blocks[0, 0], blocks[3, 3] = 2.0, -3.0
    blocks[1:3, 1:3] = [[1.2, 0.4], [-0.4, 1.2]]
    blocks[np.arange(4, n), np.arange(4, n)] = rng.uniform(-0.3, 0.3, n - 4)
    q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    return q @ blocks @ q.T


def test_exact_jacobian_block_products_share_one_bethe_salpeter_pass_per_channel_and_piece(loop_state, monkeypatch):
    """Three columns at a block width of two run as pieces of two and one, each channel solved once per piece."""
    jac = _jacobian(loop_state, *_start(loop_state), memory_estimator.ChunkBudgets(exact_block=2))
    rng = np.random.default_rng(11)
    ys = np.column_stack([jac.project(rng.standard_normal(jac.n_real)) for _ in range(3)])
    singles = np.column_stack([jac.matvec(y) for y in ys.T])
    solve = MagicMock(side_effect=nonlocal_sde.create_auxiliary_chi_r_q_sum)
    monkeypatch.setattr(nonlocal_sde, "create_auxiliary_chi_r_q_sum", solve)
    block = jac.matvec_block(ys)
    jac.free()
    assert [len(call.kwargs["rhs"]) for call in solve.call_args_list] == [2, 2, 1, 1]
    assert np.allclose(block, singles, atol=1e-12 * np.abs(singles).max())


@pytest.mark.parametrize("width", [1, 4, None])
def test_exact_jacobian_solves_each_channel_once_per_window_and_piece_with_the_whole_range_products(
    width, loop_state, monkeypatch
):
    """Windows of one, four (the last one ragged) or all frequencies give one product, one solve per channel each."""
    base, mu0 = _start(loop_state)
    budgets = memory_estimator.ChunkBudgets(exact_block=2)
    jac = _jacobian(loop_state, base.copy(), mu0, budgets)
    ys = np.column_stack([jac.project(y) for y in np.random.default_rng(5).standard_normal((2, jac.n_real))])
    whole = jac.matvec_block(ys)
    jac.free()
    jac = _jacobian(loop_state, base.copy(), mu0, memory_estimator.ChunkBudgets(exact_block=2, exact_window=width))
    solve = MagicMock(side_effect=nonlocal_sde.create_auxiliary_chi_r_q_sum)
    monkeypatch.setattr(nonlocal_sde, "create_auxiliary_chi_r_q_sum", solve)
    products, windows = jac.matvec_block(ys), jac._w_windows
    jac.free()
    n_w = config.box.niw_core + 1
    assert windows[-1][1] == n_w and len(windows) == -(-n_w // (width or n_w)) and solve.call_count == 2 * len(windows)
    assert np.allclose(products, whole, rtol=0, atol=conftest.ten_times_the_floor(whole))


def test_exact_jacobian_loads_its_local_vertices_per_window_with_the_same_products(loop_state, monkeypatch):
    """Vertices loaded per window give the held vertices' products to 10x the floor, one load per vertex and piece."""
    base, mu0 = _start(loop_state)
    real_load, loads, products, held = nonlocal_sde._load_node_shared_local_vertex, [], {}, {}

    def load(node_comm, path, channel, **kwargs):
        loads.append(os.path.basename(path))
        return real_load(node_comm, path, channel, **kwargs)

    monkeypatch.setattr(nonlocal_sde, "_load_node_shared_local_vertex", load)
    for per_window in (False, True):
        budgets = memory_estimator.ChunkBudgets(exact_block=2, exact_vertices_per_window=per_window)
        jac = _jacobian(loop_state, base.copy(), mu0, budgets)
        ys = np.column_stack([jac.project(y) for y in np.random.default_rng(6).standard_normal((3, jac.n_real))])
        loads.clear()
        products[per_window], held[per_window] = (jac.matvec_block(ys), sorted(loads)), sorted(jac._vertices)
        jac.free()
    names = ["f_dc_loc.npy", "gamma_dens_loc.npy", "gamma_magn_loc.npy"]
    held_products = products[False][0]
    assert np.allclose(products[True][0], held_products, rtol=0, atol=conftest.ten_times_the_floor(held_products))
    assert products[False][1] == [] and products[True][1] == sorted(names * 2) and held == {False: names, True: []}


@pytest.mark.parametrize("hostnames", [["n0", "n0"], ["n0", "n1"]])
def test_exact_jacobian_products_agree_across_rank_layouts(hostnames, loop_state, monkeypatch):
    """J y and J Y on rank 0 of two ranks, on one node or two, equal the single-rank products; rank 1 gets None."""
    base, mu0 = _start(loop_state)
    budgets = memory_estimator.ChunkBudgets(exact_block=2)
    single = _rank_jacobian(
        loop_state, base, mu0, budgets, loop_state.comm, loop_state.dist_irrk, loop_state.dist_fullbz
    )
    rng = np.random.default_rng(12)
    ys = np.column_stack([single.project(rng.standard_normal(single.n_real)) for _ in range(2)])
    expected = (single.matvec(ys[:, 0]), single.matvec_block(ys))
    single.free()
    monkeypatch.setattr(mpi_utils, "MPI", conftest.FAKE_MPI)
    monkeypatch.setattr(nonlocal_sde, "MPI", conftest.FAKE_MPI)

    k_grid = config.lattice.k_grid

    def fn(comm, rank):
        dist_irrk = MpiDistributor.create_distributor(ntasks=k_grid.nk_irr, comm=comm, name="Q")
        dist_fullbz = MpiDistributor.create_distributor(ntasks=k_grid.nk_tot, comm=comm, name="FBZ")
        jac = _rank_jacobian(loop_state, base, mu0, budgets, comm, dist_irrk, dist_fullbz)
        held = (jac._dn_dmu, jac._kappa)
        out = (jac.matvec(ys[:, 0] if rank == 0 else None), jac.matvec_block(ys if rank == 0 else None))
        jac.free()
        return out, held

    _, results = conftest.run_parallel(2, fn, hostnames=hostnames)
    (jy, jys), held = results[0]
    assert all(x is None for x in (*results[1][0], *results[1][1])) and held[0] is not None and held[1] is not None
    assert np.allclose(jy, expected[0], rtol=0, atol=1e-12 * np.abs(expected[0]).max())
    assert np.allclose(jys, expected[1], rtol=0, atol=1e-12 * np.abs(expected[1]).max())


def test_exact_jacobian_evaluates_each_columns_change_once(loop_state, monkeypatch):
    """A product unfolds its column and evaluates the occupation change once, inside a block too."""
    jac = _jacobian(loop_state, *_start(loop_state), memory_estimator.ChunkBudgets(exact_block=3))
    unfold = MagicMock(side_effect=jac._from_sector)
    occupation = MagicMock(side_effect=jac._occupation_change)
    monkeypatch.setattr(jac, "_from_sector", unfold)
    monkeypatch.setattr(jac, "_occupation_change", occupation)
    ys = np.column_stack([jac.project(y) for y in np.random.default_rng(4).standard_normal((3, jac.n_real))])
    singles = np.column_stack([jac.matvec(y) for y in ys.T])
    counts = (unfold.call_count, occupation.call_count)
    block = jac.matvec_block(ys)
    jac.free()
    assert counts == (3, 3) and (unfold.call_count, occupation.call_count) == (3 + 3, 3 + 3)
    assert np.allclose(block, singles, rtol=0, atol=1e-12 * np.abs(singles).max())


def test_exact_jacobian_warns_when_a_moment_fit_window_reaches_into_the_core(loop_state, monkeypatch):
    """A fit window that reaches into the core window, whose moment the products hold fixed, is reported."""
    base, mu0 = _start(loop_state)
    logger = MagicMock()
    monkeypatch.setattr(config, "logger", logger)
    monkeypatch.setattr(SelfEnergy, "_n_freq_fit", staticmethod(lambda niv: niv))
    _jacobian(loop_state, base, mu0).free()
    assert any("moment-fit window" in str(call.args[0]) for call in logger.warning.call_args_list)


def test_squared_box_sum_of_a_real_slice_keeps_the_whole_dispersions_complex_rows():
    """A momentum slice that is real alone gives the full complex dispersion's rows when told the whole is complex."""
    rng = np.random.default_rng(9)
    kx = 2 * np.pi * np.arange(4) / 4
    ek = np.zeros((4, 2, 2), dtype=complex)
    ek[:, 0, 1] = 0.4 * (1.0 + np.exp(1j * kx))
    ek[:, 1, 0] = ek[:, 0, 1].conj()
    sigma_mat = 0.2 * (rng.standard_normal((1, 2, 2, 8)) + 1j * rng.standard_normal((1, 2, 2, 8)))
    full = _squared_box_sum(sigma_mat, 0.3, ek, 10.0, True)
    sliced = _squared_box_sum(sigma_mat, 0.3, ek[:1], 10.0, True)
    assert np.array_equal(np.real_if_close(ek[:1]), ek[:1].real) and np.abs(full[0].imag).max() > 1e-3
    assert np.array_equal(sliced, full[:1])


def test_kernel_responses_of_a_rank_without_momenta_are_empty():
    """A rank that owns no irreducible momentum (more ranks than momenta) solves no slice and raises nothing."""
    jac = ExactJacobian.__new__(ExactJacobian)
    empty = np.zeros((0, 1, 1, 1, 1, 3, 4), dtype=np.complex64)
    jac._gchi0_inv = FourPoint(empty, SpinChannel.DENS, (1, 1, 1), 1, 1, False, has_compressed_q_dimension=True)
    jac._beta, jac._channels, jac._chi, jac._vrg = 10.0, [], {}, {}
    assert jac._kernel_responses([], (0, 2)) == []


def test_leading_eigenpairs_serves_the_products_of_every_rank_and_finds_both_targets(quiet_logger):
    """Under two ranks, rank 0 gets the leading and the stiff eigenpairs and both ranks take part in every product."""
    a = _normal_matrix(40, seed=1)
    operators = {}

    def fn(comm, rank):
        operators[rank] = _make_matrix_operator(a, comm)
        return leading_eigenpairs(operators[rank], comm)

    _, results = conftest.run_parallel(2, fn)

    assert results[1] is None
    assert operators[0].matvec.call_count == operators[1].matvec.call_count > 0
    lam_pi, res, u = results[0]
    expected = 1.0 - np.array([2.0, 1.2 + 0.4j, 1.2 - 0.4j, -3.0])
    assert np.allclose([np.min(np.abs(lam_pi - value)) for value in expected], 0.0, atol=1e-4)
    assert np.all(np.diff(lam_pi.real) >= 0) and np.all(res > 0) and np.all(res < 1e-3)
    assert np.allclose(np.linalg.norm(u, axis=0), 1.0, atol=1e-12)
    assert np.allclose(a @ u, u * (1.0 - lam_pi), atol=1e-4)


def _record_eigs_requests(monkeypatch) -> dict:
    """Replaces scipy's eigs by one that records the vectors each target applies the operator to."""
    real_eigs, requests = spla.eigs, {}

    def eigs(operator, k, which, **kwargs):
        requests[which] = []

        def record(x):
            requests[which].append(np.array(x, copy=True))
            return operator.matvec(x)

        recorder = spla.LinearOperator(operator.shape, matvec=record, dtype=operator.dtype)
        return real_eigs(recorder, k=k, which=which, **kwargs)

    monkeypatch.setattr(spla, "eigs", eigs)
    return requests


def test_leading_eigenpairs_serves_the_repeated_first_factorization_without_products(quiet_logger, monkeypatch):
    """The second target repeats the first one's first ncv + 1 vectors, whose kept products save as many products."""
    requests = _record_eigs_requests(monkeypatch)
    comm = conftest.create_comm_mock()
    a = _normal_matrix(100, seed=3)
    operator = _make_matrix_operator(a, comm)
    lam_pi, _, u = leading_eigenpairs(operator, comm)
    ncv = memory_estimator.EXACT_JACOBIAN_NCV
    shared = all(np.array_equal(x, y) for x, y in zip(requests["LR"][: ncv + 1], requests["LM"][: ncv + 1]))
    assert shared and len(requests["LM"]) > ncv
    assert operator.matvec.call_count == len(requests["LR"]) + len(requests["LM"]) - (ncv + 1)
    assert np.allclose(a @ u, u * (1.0 - lam_pi), atol=1e-4)


def test_leading_eigenpairs_starts_both_targets_from_the_given_directions(quiet_logger, monkeypatch):
    """Both targets start from one vector holding the given least-stable and stiff directions beside a random one."""
    requests = _record_eigs_requests(monkeypatch)
    comm = conftest.create_comm_mock()
    a = _normal_matrix(100, seed=3)
    theta, vectors = np.linalg.eig(a)
    lead, stiff = (vectors[:, np.argmax(f(theta))].real for f in (np.real, np.abs))
    lam_pi, _, _ = leading_eigenpairs(_make_matrix_operator(a, comm), comm, lead, stiff)
    start = requests["LR"][0] / np.linalg.norm(requests["LR"][0])
    assert np.array_equal(requests["LR"][0], requests["LM"][0])
    assert all(abs(start @ v) / np.linalg.norm(v) > 0.4 for v in (lead, stiff))
    assert np.min(np.abs(lam_pi - (1.0 - 2.0))) < 1e-6 and np.min(np.abs(lam_pi - 4.0)) < 1e-6


def test_leading_eigenpairs_warns_when_every_leading_pair_is_unstable(monkeypatch):
    """Eight unstable directions fill the six pairs of largest real part and are flagged; one unstable value is not."""
    rng = np.random.default_rng(5)
    q = np.linalg.qr(rng.standard_normal((60, 60)))[0]
    comm = conftest.create_comm_mock()
    warned = []
    for unstable in (8, 1):
        logger = MagicMock()
        monkeypatch.setattr(config, "logger", logger, raising=False)
        diagonal = np.concatenate((np.linspace(1.5, 3.0, unstable), rng.uniform(-0.5, 0.5, 60 - unstable)))
        leading_eigenpairs(_make_matrix_operator(q @ np.diag(diagonal) @ q.T, comm), comm)
        warned.append(any("are unstable" in str(call.args[0]) for call in logger.warning.call_args_list))
    assert warned == [True, False]


def test_sector_projector_averages_every_irreducible_momentum_over_its_little_group():
    """The projector is idempotent, keeps a symmetric array and imposes the orbital constraints of each stabilizer."""
    grid, hk = _p_orbital_grid()
    project = sector_projector(grid, np.ones((2, 2), dtype=bool))
    rng = np.random.default_rng(0)
    y = rng.standard_normal((grid.nk_irr, 2, 2, 3)) + 1j * rng.standard_normal((grid.nk_irr, 2, 2, 3))
    py = project(y)
    gamma = int(np.flatnonzero(grid.irrk_ind == 0)[0])
    assert np.allclose(project(py), py, atol=1e-14)
    assert np.allclose(py[gamma, 0, 0], py[gamma, 1, 1]) and np.allclose(py[gamma, 0, 1], 0.0)
    symmetric = hk.reshape(-1, 2, 2)[grid.irrk_ind][..., None] * np.array([1.0, 0.5j, -2.0])
    assert np.allclose(project(symmetric), symmetric, atol=1e-12)


def test_sector_projector_zeroes_the_orbital_pairs_outside_keep():
    """The orbital pairs the self-energy never populates are zeroed on every momentum, the others kept."""
    grid, _ = _p_orbital_grid()
    out = sector_projector(grid, np.eye(2, dtype=bool))(np.ones((grid.nk_irr, 2, 2, 2), dtype=complex))
    assert np.allclose(out[:, 0, 1], 0.0) and np.allclose(out[:, 1, 0], 0.0)
    assert np.allclose(out[:, 0, 0], 1.0) and np.allclose(out[:, 1, 1], 1.0)


def test_sector_projector_leaves_the_orbitals_alone_where_the_symmetries_act_on_momenta_only():
    """Symmetries given by name act on the momenta alone, so the projector is the identity there."""
    grid = bz.KGrid((4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    y = np.random.default_rng(1).standard_normal((grid.nk_irr, 2, 2, 2)).astype(complex)
    assert np.array_equal(sector_projector(grid, np.ones((2, 2), dtype=bool))(y), y)


def test_exact_jacobian_acts_on_the_sector_the_loop_reaches(monkeypatch):
    """The products stay in the sector (a symmetry-breaking input gives zero), in the tracker's coordinates."""
    state = _loop_state(monkeypatch, "p_orbitals")
    jac = _jacobian(state, *_start(state))
    y = np.random.default_rng(5).standard_normal(jac.n_real)
    breaking = y - jac.project(y)
    jy, jb = jac.matvec(y), jac.matvec(breaking)
    round_trip = _sector(_window(y - breaking))
    jac.free()
    assert np.linalg.norm(breaking) > 0.05 * np.linalg.norm(y)
    assert np.allclose(round_trip, y - breaking, atol=1e-12 * np.abs(y).max())
    assert np.allclose(jb, 0.0, atol=1e-12 * np.abs(jy).max())
    assert np.allclose(jac.project(jy), jy, atol=1e-12 * np.abs(jy).max())


@pytest.mark.parametrize("block", [1, 2])
def test_leading_eigenpairs_reports_only_modes_of_the_projected_sector(quiet_logger, block):
    """Eigenpairs outside the projected sector never reach the result, although theta = 0 there tops a negative bulk."""
    rng = np.random.default_rng(2)
    inner = np.diag(np.concatenate(([2.0, -3.0], rng.uniform(-0.9, -0.4, 26))))
    inner[2:4, 2:4] = [[1.2, 0.4], [-0.4, 1.2]]
    q = np.linalg.qr(rng.standard_normal((28, 28)))[0]
    a = np.zeros((40, 40))
    a[:28, :28] = q @ inner @ q.T
    keep = np.arange(40) < 28
    comm = conftest.create_comm_mock()
    operator = types.SimpleNamespace(
        n_real=40,
        matvec=lambda y: keep * (a @ (keep * y)),
        matvec_block=lambda ys: keep[:, None] * (a @ (keep[:, None] * ys)),
        expand=lambda y: y,
        project=lambda y: keep * y,
    )
    lam_pi, _, u = leading_eigenpairs(operator, comm, block=block)
    assert np.allclose(u[~keep], 0.0, atol=1e-12)
    assert not np.any(np.isclose(lam_pi, 1.0, atol=1e-6))
    assert np.min(np.abs(lam_pi - (1.0 - 2.0))) < 1e-6


def _start_near(vector: np.ndarray, overlap: float, seed: int) -> np.ndarray:
    """A unit start column with the given overlap with the unit ``vector``, the rest random and orthogonal to it."""
    noise = np.random.default_rng(seed).standard_normal(vector.size)
    noise -= (noise @ vector) * vector
    return (overlap * vector + np.sqrt(1.0 - overlap**2) * noise / np.linalg.norm(noise))[:, None]


def _leading_vector(a: np.ndarray) -> np.ndarray:
    """The unit eigenvector of the eigenvalue of ``a`` with the largest real part (a real one)."""
    theta, vectors = np.linalg.eig(a)
    lead = vectors[:, np.argmax(theta.real)].real
    return lead / np.linalg.norm(lead)


def test_certify_subspace_certifies_the_leading_pair_from_a_nearby_start(quiet_logger):
    """A start with overlap 0.9 to the unstable eigenvector certifies it within the budget, with its exact residual."""
    a = _normal_matrix(40, seed=1)
    comm = conftest.create_comm_mock()
    lam_pi, res, u, n_products, _ = certify_subspace(
        _make_matrix_operator(a, comm), _start_near(_leading_vector(a), 0.9, seed=4), comm
    )
    j = int(np.argmin(lam_pi.real))
    assert np.isclose(lam_pi[j], -1.0, atol=1e-2) and res[j] <= _RITZ_GATE * 2.0 and 0 < n_products <= _CHECK_BASIS
    assert np.isclose(res[j], np.linalg.norm(a @ u[:, j] - (1.0 - lam_pi[j]) * u[:, j]), atol=1e-12)


def test_certify_subspace_returns_the_condition_numbers_of_a_non_normal_pair(quiet_logger):
    """An invariant 2x2 block with coupling 1 over the gap 0.5 certifies both values with kappa sqrt(1 + 2^2)."""
    rng = np.random.default_rng(3)
    blocks = np.diag(np.r_[2.0, 1.5, rng.uniform(-0.3, 0.3, 38)])
    blocks[0, 1] = 1.0
    q = np.linalg.qr(rng.standard_normal((40, 40)))[0]
    comm = conftest.create_comm_mock()
    lam_pi, res, _, n_products, kappa = certify_subspace(_make_matrix_operator(q @ blocks @ q.T, comm), q[:, :2], comm)
    assert n_products == 2 and np.allclose(lam_pi, [-1.0, -0.5], atol=1e-12) and np.all(res < 1e-12)
    assert np.allclose(kappa, np.sqrt(5.0), atol=1e-10)


def test_certify_subspace_returns_an_uncertified_pair_when_the_budget_runs_out(quiet_logger, monkeypatch):
    """With a budget of one product a random start comes back after that product, its residual above the gate."""
    monkeypatch.setattr("dgamore.sigma_jacobian._CHECK_BASIS", 1)
    comm = conftest.create_comm_mock()
    operator = _make_matrix_operator(_normal_matrix(40, seed=1), comm)
    lam_pi, res, _, n_products, _ = certify_subspace(operator, np.random.default_rng(7).standard_normal((40, 1)), comm)
    assert n_products == operator.matvec_block.call_count == 1 and lam_pi.size == 1
    assert res[0] > _RITZ_GATE * max(1.0, abs(1.0 - lam_pi[0]))


def test_certify_subspace_applies_its_start_columns_and_each_extension_as_one_block(quiet_logger):
    """The start columns go through one block product, and every extension of the basis through one more."""
    comm = conftest.create_comm_mock()
    operator = _make_matrix_operator(_normal_matrix(40, seed=1), comm)
    *_, n_products, _ = certify_subspace(operator, np.random.default_rng(8).standard_normal((40, 3)), comm)
    widths = [call.args[0].shape[1] for call in operator.matvec_block.call_args_list]
    assert widths[0] == 3 and all(width in (1, 2) for width in widths[1:]) and sum(widths) == n_products
    assert operator.matvec.call_count == 0


def test_certify_subspace_drops_a_start_column_that_lies_outside_the_sector(quiet_logger):
    """A start column whose sector part is at rounding level relative to it costs no product."""
    comm = conftest.create_comm_mock()
    keep = np.arange(40) < 28
    a = np.zeros((40, 40))
    a[:28, :28] = _normal_matrix(28, seed=1)
    operator = _make_matrix_operator(a, comm)
    operator.project = lambda y: keep * y
    rng = np.random.default_rng(2)
    outside = np.where(keep, 1e-12, 1.0) * rng.standard_normal(40)
    certify_subspace(operator, np.column_stack([keep * rng.standard_normal(40), outside]), comm)
    assert operator.matvec_block.call_args_list[0].args[0].shape[1] == 1


def test_certify_subspace_serves_the_products_of_every_rank(quiet_logger):
    """Under two ranks both take part in every product and only rank 0 gets the certified pairs back."""
    a = _normal_matrix(40, seed=1)
    start = _start_near(_leading_vector(a), 0.9, seed=4)
    operators = {}

    def fn(comm, rank):
        operators[rank] = _make_matrix_operator(a, comm)
        return certify_subspace(operators[rank], start if rank == 0 else None, comm)

    _, results = conftest.run_parallel(2, fn)
    columns = [call.args[0].shape[1] for call in operators[0].matvec_block.call_args_list]
    assert results[1] is None and results[0][3] == sum(columns)
    assert operators[0].matvec_block.call_count == operators[1].matvec_block.call_count > 0


def test_leading_eigenpairs_runs_on_a_sector_of_eight_entries(quiet_logger):
    """ARPACK's basis fits a sector of eight real entries, where k + 1 < ncv <= n leaves little room."""
    comm = conftest.create_comm_mock()
    lam_pi, res, u = leading_eigenpairs(_make_matrix_operator(_normal_matrix(8, seed=1), comm), comm)
    assert lam_pi.size > 0 and np.all(res > 0)


def test_leading_eigenpairs_keeps_the_converged_pairs_of_a_target_that_gives_up(monkeypatch):
    """A target ARPACK gives up on keeps its converged pairs with a warning, and the serving ranks are released."""
    logger = MagicMock()
    monkeypatch.setattr(config, "logger", logger, raising=False)
    comm = conftest.create_comm_mock()
    real_eigs = spla.eigs

    def eigs(operator, k, which, **kwargs):
        theta, vectors = real_eigs(operator, k=k, which=which, **kwargs)
        if which == "LR":
            raise spla.ArpackNoConvergence("stopped", theta[:2], vectors[:, :2])
        return theta, vectors

    monkeypatch.setattr(spla, "eigs", eigs)
    lam_pi, res, u = leading_eigenpairs(_make_matrix_operator(_normal_matrix(40, seed=1), comm), comm)
    assert any("ARPACK (LR) converged 2 of" in str(c.args[0]) for c in logger.warning.call_args_list)
    assert lam_pi.size >= 2 and comm.bcast.call_args_list[-1].args[0] is False


@pytest.mark.parametrize("block", [2, 4])
def test_leading_eigenpairs_block_search_serves_every_rank_and_finds_both_targets(quiet_logger, block):
    """Every rank serves each product of at most block columns, and rank 0 gets the leading and the stiff pairs."""
    a = _normal_matrix(40, seed=1)
    operators = {}

    def fn(comm, rank):
        operators[rank] = _make_matrix_operator(a, comm)
        return leading_eigenpairs(operators[rank], comm, block=block)

    _, results = conftest.run_parallel(2, fn)

    widths = [c.args[0].shape[1] for c in operators[0].matvec_block.call_args_list]
    assert results[1] is None and operators[0].matvec.call_count == operators[1].matvec.call_count == 0
    assert operators[1].matvec_block.call_count == len(widths) > 0 and max(widths) <= block
    lam_pi, res, u = results[0]
    expected = 1.0 - np.array([2.0, 1.2 + 0.4j, 1.2 - 0.4j, -3.0])
    assert np.allclose([np.min(np.abs(lam_pi - value)) for value in expected], 0.0, atol=1e-4)
    assert np.all(np.diff(lam_pi.real) >= 0) and np.allclose(np.linalg.norm(u, axis=0), 1.0, atol=1e-12)
    assert np.all(np.linalg.norm(a @ u - u * (1.0 - lam_pi), axis=0) <= res)


def test_leading_eigenpairs_block_search_starts_with_the_given_directions_as_columns(quiet_logger):
    """The first block spans the least-stable and the stiff direction beside the shared random start vector."""
    comm = conftest.create_comm_mock()
    a = _normal_matrix(100, seed=3)
    theta, vectors = np.linalg.eig(a)
    lead, stiff = (vectors[:, np.argmax(f(theta))].real for f in (np.real, np.abs))
    operator, blocks = _make_matrix_operator(a, comm), []
    product = operator.matvec_block.side_effect
    operator.matvec_block.side_effect = lambda ys: blocks.append(np.array(ys)) or product(ys)
    lam_pi, _, _ = leading_eigenpairs(operator, comm, lead, stiff, block=4)
    first = blocks[0]
    assert first.shape[1] == 4 and np.allclose(first.T @ first, np.eye(4), atol=1e-12)
    assert all(np.linalg.norm(first.T @ v) > (1 - 1e-12) * np.linalg.norm(v) for v in (lead, stiff))
    assert np.min(np.abs(lam_pi - (1.0 - 2.0))) < 1e-6 and np.min(np.abs(lam_pi - 4.0)) < 1e-6


def test_leading_eigenpairs_block_search_keeps_its_converged_pairs_when_the_passes_run_out(monkeypatch):
    """A block search out of passes warns, returns only its converged pairs and releases the serving ranks."""
    logger = MagicMock()
    monkeypatch.setattr(config, "logger", logger, raising=False)
    monkeypatch.setattr("dgamore.sigma_jacobian._MAX_MATVECS", 8)
    comm = conftest.create_comm_mock()
    a = _normal_matrix(200, seed=3)
    lam_pi, res, u = leading_eigenpairs(_make_matrix_operator(a, comm), comm, block=2)
    assert any("the block search converged" in str(c.args[0]) for c in logger.warning.call_args_list)
    assert lam_pi.size > 0 and np.all(np.linalg.norm(a @ u - u * (1.0 - lam_pi), axis=0) <= res)
    assert comm.bcast.call_args_list[-1].args[0] is False


def test_leading_eigenpairs_block_search_stays_inside_rank_0s_modeled_eigensolver_vectors(monkeypatch):
    """Rank 0's traced peak of a block search of four columns stays inside the estimator's float64 sector vectors."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
    monkeypatch.setattr("dgamore.sigma_jacobian._SWEEP_ROWS", 2048)
    n, rng = 50_000, np.random.default_rng(0)
    diagonal = np.concatenate(([2.0, -3.0, 1.5], rng.uniform(-0.3, 0.3, n - 3)))[:, None]
    low = rng.standard_normal((2, n, 3)) / np.sqrt(n)
    operator = types.SimpleNamespace(
        n_real=n, matvec_block=lambda ys: diagonal * ys + low[0] @ (low[1].T @ ys), project=lambda y: y.copy()
    )
    starts = rng.standard_normal((2, n))
    tracemalloc.start()
    lam_pi, _, _ = leading_eigenpairs(operator, conftest.create_comm_mock(), *starts, block=4)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    modeled = 2 * (memory_estimator.EXACT_JACOBIAN_NCV + 1) + 3 * memory_estimator.EXACT_JACOBIAN_MODES + 11
    assert lam_pi.size > 0 and peak <= modeled * 8 * n


def test_block_extension_redoes_a_cancelling_column_against_the_whole_basis():
    """A block column nearly parallel to the one before it keeps the basis orthonormal to rounding."""
    rng = np.random.default_rng(4)
    basis = np.empty((500, 12), order="F")
    basis[:, :8] = np.linalg.qr(rng.standard_normal((500, 8)))[0]
    fresh = rng.standard_normal(500)
    near = fresh + 1e-7 * rng.standard_normal(500) + 1e-3 * basis[:, :8] @ rng.standard_normal(8)
    block = np.asfortranarray(np.column_stack([fresh, near]))
    size = _block_extension(basis, 8, block, np.linalg.norm(block, axis=0))
    assert size == 10 and np.allclose(basis[:, :size].T @ basis[:, :size], np.eye(size), rtol=0, atol=1e-13)


def test_thick_restart_keeps_the_leading_ritz_values_with_the_images_and_the_map_of_the_kept_basis():
    """The compacted basis holds the 2k leading Ritz values of both targets, J times it and its projected map."""
    a = _normal_matrix(60, seed=5)
    basis = np.asfortranarray(np.linalg.qr(np.random.default_rng(6).standard_normal((60, 30)))[0])
    images = np.asfortranarray(a @ basis)
    h = basis.T @ images
    theta = np.linalg.eigvals(h)
    kept = _thick_restart(basis, images, h, 30, theta, 3)
    leading = [*np.argsort(-theta.real)[:6], *np.argsort(-np.abs(theta))[:6]]
    kept_theta = np.linalg.eigvals(h[:kept, :kept])
    assert kept < 30 and all(np.min(np.abs(kept_theta - theta[i])) < 1e-10 for i in leading)
    q, w = basis[:, :kept], images[:, :kept]
    assert np.allclose(q.T @ q, np.eye(kept), rtol=0, atol=1e-12) and np.allclose(w, a @ q, rtol=0, atol=1e-12)
    assert np.allclose(q.T @ w, h[:kept, :kept], rtol=0, atol=1e-12)


def test_merge_modes_keeps_both_partners_of_a_degenerate_value():
    """Two eigenpairs of one value with orthogonal vectors are two modes; a repeated pair is one."""
    theta = np.array([2.0 + 0j, 2.0 + 0j, 2.0 + 0j])
    vectors = np.eye(4)[:, [0, 1, 0]].astype(complex)
    kept, kept_vectors = _merge_modes(theta, vectors, 1e-5)
    assert kept.size == 2 and abs(np.vdot(kept_vectors[:, 0], kept_vectors[:, 1])) < 1e-12


def test_merge_modes_keeps_both_partners_of_a_near_real_complex_pair():
    """A lone member of a near-real complex pair with almost real eigenvectors comes back with its partner."""
    vector = np.array([1.0, -0.05j, 0.0]) / np.sqrt(1.0025)
    for theta, vectors in (
        ([0.6 - 3e-4j], vector[:, None]),
        ([0.6 + 3e-4j, 0.6 - 3e-4j], np.column_stack([vector.conj(), vector])),
    ):
        kept, kept_vectors = _merge_modes(np.array(theta), np.array(vectors), 100 * np.finfo(np.float32).eps)
        assert np.allclose(kept, [0.6 + 3e-4j, 0.6 - 3e-4j], atol=1e-15)
        assert np.allclose(kept_vectors, np.column_stack([vector.conj(), vector]), atol=1e-15)


def test_merge_modes_keeps_one_vector_per_dimension_of_a_degenerate_space():
    """A two-dimensional eigenspace returned by both targets in different bases is kept as two modes."""
    e0, e1 = np.eye(4)[:, 0], np.eye(4)[:, 1]
    vectors = np.column_stack([e0, e1, (e0 + e1) / np.sqrt(2), (e0 - e1) / np.sqrt(2)]).astype(complex)
    kept, kept_vectors = _merge_modes(np.full(4, 2.0 + 0j), vectors, 1e-5)
    assert kept.size == 2 and np.allclose(kept_vectors, np.column_stack([e0, e1]), atol=1e-15)


def test_merge_modes_allocates_only_the_kept_columns_and_their_partners():
    """Merging complex eigenpairs costs about its result, no normalized or conjugated copy of the input."""
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((20000, 12)) + 1j * rng.standard_normal((20000, 12))
    theta = 1.0 + np.arange(12) + 0.5j
    (kept, _), peak = conftest.traced_peak(lambda: _merge_modes(theta, vectors, 1e-10))
    assert kept.size == 24 and peak < 2.2 * vectors.nbytes


def test_exact_product_allocates_between_half_and_all_of_the_estimated_per_rank_transient(loop_state, monkeypatch):
    """The tracemalloc peak of one exact product lies between half and all of the estimate's per-rank transient."""
    jac = _jacobian(loop_state, *_start(loop_state))
    y = jac.project(np.random.default_rng(3).standard_normal(jac.n_real))
    _, peak = conftest.traced_peak(lambda: jac.matvec(y))
    jac.free()
    monkeypatch.setattr(memory_estimator, "DTYPE_BYTES", np.dtype(np.complex128).itemsize)
    box, k_grid = config.box, config.lattice.k_grid
    bp = memory_estimator.estimate_peaks(
        n_bands=config.sys.n_bands,
        nk_tot=k_grid.nk_tot,
        nk_irr=k_grid.nk_irr,
        niw_core=box.niw_core,
        niv_core=box.niv_core,
        niv_full=box.niv_full,
        niv_cut=min(box.niw_core + box.niv_full + 10, box.niv_dmft),
        niv_dmft=box.niv_dmft,
        niv_pp=1,
        n_ranks=1,
        with_eliashberg=False,
        with_exact_jacobian=True,
    )["exact_jacobian"]
    assert 0.5 * bp.off_distributed < peak <= bp.off_distributed


@pytest.mark.parametrize("block", [1, 3])
@pytest.mark.parametrize("dtype, tol", [(np.complex64, 100 * np.finfo(np.float32).eps), (np.complex128, 1e-10)])
def test_leading_eigenpairs_bounds_the_residual_by_the_tolerance_of_the_storage_precision(
    dtype, tol, block, monkeypatch
):
    """The residual bound of every pair is max(100 eps, 1e-10) max(1, |theta|), eps of the storage precision."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
    monkeypatch.setattr(npb, "DTYPE", dtype)
    comm = conftest.create_comm_mock()
    a = _normal_matrix(40, seed=1)
    lam_pi, res, u = leading_eigenpairs(_make_matrix_operator(a, comm), comm, block=block)
    assert np.allclose(res, tol * np.maximum(1.0, np.abs(1.0 - lam_pi)), atol=1e-20)
    assert np.all(np.linalg.norm(a @ u - u * (1.0 - lam_pi), axis=0) <= res)


def _sector_grid(auto: bool) -> bz.KGrid:
    """The p_x/p_y auto-symmetry grid, whose elements rotate the orbitals, or a grid of named momentum symmetries."""
    return _p_orbital_grid()[0] if auto else bz.KGrid((4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())


@pytest.mark.parametrize("auto", [True, False])
def test_tracker_sector_maps_are_an_isometry_of_the_symmetric_windows(auto, monkeypatch):
    """A window unfolded from the sector returns from its weighted sector vector unchanged, with the same norm."""
    monkeypatch.setattr(npb, "DTYPE", np.complex128)
    grid, niv = _sector_grid(auto), 3
    to_sector, from_sector = tracker_sector_maps(grid, niv, 2, 10.0)
    window = from_sector(np.random.default_rng(0).standard_normal(2 * grid.nk_irr * 4 * niv))
    sector = to_sector(window)
    assert window.size == grid.nk_tot * 4 * 2 * niv and sector.size == 2 * grid.nk_irr * 4 * niv
    assert np.allclose(np.linalg.norm(sector), np.linalg.norm(window), atol=1e-12 * np.linalg.norm(window))
    assert np.allclose(from_sector(sector), window, atol=1e-12 * np.abs(window).max())


@pytest.mark.parametrize("auto", [True, False])
def test_tracker_sector_maps_keep_the_window_of_a_lattice_self_energy(auto, monkeypatch):
    """A self-energy built from the lattice Hamiltonian as the loop builds its proposal survives the round trip."""
    monkeypatch.setattr(npb, "DTYPE", np.complex128)
    grid, niv, beta = _sector_grid(auto), 3, 10.0
    hk = _p_orbital_grid()[1] if auto else np.zeros((*grid.nk, 2, 2), dtype=complex)
    if not auto:
        kx, ky, _ = np.meshgrid(*(2 * np.pi * np.arange(n) / n for n in grid.nk), indexing="ij")
        hk[..., 0, 0] = -2.0 * (np.cos(kx) + np.cos(ky))
        hk[..., 1, 1] = 0.5 * hk[..., 0, 0] + 0.2
        hk[..., 0, 1] = hk[..., 1, 0] = 0.3
    nu = (2 * np.arange(niv) + 1) * np.pi / beta
    mat = np.moveaxis(np.linalg.inv((1j * nu + 0.5)[:, None, None] * np.eye(2) - hk[..., None, :, :]), -3, -1)
    window = SelfEnergy(mat, grid.nk, False, beta=beta).compress_q_dimension().to_full_niv_range().mat
    to_sector, from_sector = tracker_sector_maps(grid, niv, 2, beta)
    assert np.allclose(from_sector(to_sector(window)), window, atol=1e-12 * np.abs(window).max())


def _symmetric_iteration(grid: bz.KGrid, niv: int, gains: np.ndarray, n_iter: int, maps):
    """The window pairs of a damped linear iteration that stays on the symmetric windows, from a sector-space map."""
    to_sector, from_sector = maps
    n = 2 * grid.nk_irr * 4 * niv
    rng = np.random.default_rng(7)
    basis = np.linalg.qr(rng.standard_normal((n, gains.size)))[0]
    x = to_sector(from_sector(rng.standard_normal(n)))
    iterates, proposals = [], []
    for _ in range(n_iter):
        f = x + basis @ ((gains - 1.0) * (basis.T @ x))
        iterates.append(from_sector(x))
        proposals.append(from_sector(f))
        x = x + 0.3 * (f - x)
    return iterates, proposals


@pytest.mark.parametrize("auto", [True, False])
def test_a_tracker_on_sector_coordinates_decides_as_one_on_the_whole_window(auto, monkeypatch):
    """Ritz values, flips and the reflected proposal agree between a sector-coordinate tracker and a full-window one."""
    monkeypatch.setattr(npb, "DTYPE", np.complex128)
    grid, niv = _sector_grid(auto), 3
    maps = tracker_sector_maps(grid, niv, 2, 10.0)
    iterates, proposals = _symmetric_iteration(grid, niv, np.array([2.5, 0.6, 0.3]), 7, maps)
    whole, sector = JacobianTracker(0.4), JacobianTracker(0.4, to_sector=maps[0], from_sector=maps[1])
    for tracker in (whole, sector):
        tracker.update(iterates, proposals)
    assert np.allclose(np.sort_complex(sector.last_lam_pi), np.sort_complex(whole.last_lam_pi), atol=1e-8)
    assert sector._flip.sum() == whole._flip.sum() == 1 and sector.active and whole.active
    reflected = [tracker.reflect(proposals[-1], iterates[-1]) for tracker in (whole, sector)]
    assert np.allclose(reflected[1], reflected[0], atol=1e-10 * np.abs(reflected[0]).max())


def test_exact_jacobian_scans_each_channel_once_and_reuses_the_real_space_interaction(loop_state, monkeypatch):
    """Pairs without vertex are scanned once per channel, and every product reuses the held real-space interaction."""
    base, mu0 = _start(loop_state)
    scan = create_autospec(
        LocalFourPoint.orbital_pairs_without_vertex, wraps=LocalFourPoint.orbital_pairs_without_vertex
    )
    monkeypatch.setattr(LocalFourPoint, "orbital_pairs_without_vertex", scan)
    hartree_fock = MagicMock(side_effect=nonlocal_sde.get_hartree_fock)
    monkeypatch.setattr(nonlocal_sde, "get_hartree_fock", hartree_fock)
    jac = _jacobian(loop_state, base, mu0)
    y = jac.project(np.random.default_rng(2).standard_normal(jac.n_real))
    first, second, w_r = jac.matvec(y), jac.matvec(y), jac._w_r
    jac.free()
    assert scan.call_count == 2 and np.array_equal(first, second)
    assert hartree_fock.call_count == 2 and all(call.args[4] is w_r for call in hartree_fock.call_args_list)


def test_exact_jacobian_frees_every_full_box_bubble_before_attaching_the_dc_interaction(loop_state, monkeypatch):
    """The build and a product split and free every window's full-box bubble before attaching the dc interaction."""
    base, mu0 = _start(loop_state)
    bubbles, freed = [], []
    contract, attach = nonlocal_sde._contract_dc_vertex, nonlocal_sde._attach_dc_interaction

    def contract_dc_vertex(f_dc_loc, gchi0_q):
        bubbles.append(gchi0_q)
        return contract(f_dc_loc, gchi0_q)

    def attach_dc_interaction(contraction, u_loc, per_wn=False):
        freed.append(bubbles[-1].mat is None and per_wn)
        return attach(contraction, u_loc, per_wn)

    monkeypatch.setattr(nonlocal_sde, "_contract_dc_vertex", contract_dc_vertex)
    monkeypatch.setattr(nonlocal_sde, "_attach_dc_interaction", attach_dc_interaction)
    jac = _jacobian(loop_state, base, mu0, memory_estimator.ChunkBudgets(exact_window=4))
    jac.matvec(jac.project(np.random.default_rng(3).standard_normal(jac.n_real)))
    jac.free()
    assert freed == [True] * 2 * len(jac._w_windows) and len(jac._w_windows) == 6


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_exact_product_finalized_in_place_has_the_bits_of_the_out_of_place_sums(loop_state, monkeypatch, dtype):
    """Rank 0's in-place transform and Hartree-Fock terms of a column's contraction give the out-of-place bits."""
    base, mu0 = _start(loop_state)
    monkeypatch.setattr(npb, "DTYPE", dtype)
    jac = _jacobian(loop_state, base, mu0)
    contractions, terms = [], []
    real_finish, real_hartree_fock = nonlocal_sde._ColumnSdeStream.finish, nonlocal_sde.get_hartree_fock

    def finish(stream, out):
        contractions.append(real_finish(stream, out))
        return contractions[-1].copy()

    def hartree_fock(*args):
        terms.append(real_hartree_fock(*args))
        return terms[-1]

    monkeypatch.setattr(nonlocal_sde._ColumnSdeStream, "finish", finish)
    monkeypatch.setattr(nonlocal_sde, "get_hartree_fock", hartree_fock)
    product = jac.matvec(jac.project(np.random.default_rng(3).standard_normal(jac.n_real)))
    reference = contractions[0].ifft().to_full_niv_range() + terms[0][0] + terms[0][1]
    reference.compress_q_dimension().symmetrize_time_reversal()
    expected = jac.project(jac._to_sector(reference.mat))
    jac.free()
    assert jac._tr_average and len(contractions) == 1 and np.array_equal(product, expected)


def test_exact_product_finalize_holds_two_core_windows_on_rank_0(monkeypatch):
    """Transforming and adding Hartree-Fock in place leaves the full-range completion as the finalize peak."""
    nk, nb, niv = (32, 32, 1), 2, 20
    config.lattice.nk, config.sys.n_bands, config.box.niv_core = nk, nb, niv
    rng = np.random.default_rng(0)
    nk_tot = int(np.prod(nk))
    half = rng.standard_normal((nk_tot, nb, nb, niv)) + 1j * rng.standard_normal((nk_tot, nb, nb, niv))
    u = LocalInteraction(rng.standard_normal((nb,) * 4).astype(complex))
    v = Interaction(rng.standard_normal((*nk, nb, nb, nb, nb)).astype(complex), SpinChannel.NONE, nk)
    jac = types.SimpleNamespace(
        _comm=types.SimpleNamespace(rank=0),
        _u_loc=u,
        _v_nonloc_full=v,
        _w_r=(u + v).fft(copy=False).decompress_q_dimension().mat,
        _tr_average=False,
        _to_sector=lambda mat: mat[:1].real.ravel(),
        project=lambda y: y,
    )
    docc, docc_k = rng.standard_normal((nb, nb)) + 0j, rng.standard_normal((*nk, nb, nb)) + 0j
    contraction = SelfEnergy(half, nk, False, True, calc_smom=False, beta=10.0)
    sigma_core = nk_tot * nb * nb * 2 * niv * np.dtype(npb.DTYPE).itemsize
    _, peak = conftest.traced_peak(lambda: ExactJacobian._finalize(jac, contraction, docc, docc_k))
    assert npb.DTYPE == np.complex64 and half.astype(npb.DTYPE).nbytes + peak <= 2.05 * sigma_core


def _diagonal_operator(n: int) -> types.SimpleNamespace:
    """A diagonal operator with a leading, a stiff and a spread bulk of values; a product allocates only its result."""
    d = np.random.default_rng(0).uniform(-0.9, 0.9, n)
    d[:3] = 1.6, 1.3, -2.5

    def product(y):
        return d.reshape((n,) + (1,) * (np.ndim(y) - 1)) * y

    return types.SimpleNamespace(n_real=n, matvec=product, matvec_block=product, project=lambda y: y)


def test_eigen_solvers_on_rank_0_stay_within_the_sector_vectors_the_memory_model_gives_them(quiet_logger):
    """ARPACK beside its two start directions and a full-basis exact check peak within their modeled sector vectors."""
    comm, n = conftest.create_comm_mock(), 40000
    operator, start = _diagonal_operator(n), np.random.default_rng(1).standard_normal((n, 6))
    solvers = (
        lambda: leading_eigenpairs(operator, comm, *np.random.default_rng(2).standard_normal((2, n))),
        lambda: certify_subspace(operator, start, comm),
    )
    traced = [conftest.traced_peak(solve) for solve in solvers]
    result, peaks = traced[1][0], [peak / (8 * n) for _, peak in traced]
    modes, ncv = memory_estimator.EXACT_JACOBIAN_MODES, memory_estimator.EXACT_JACOBIAN_NCV
    assert result[3] == _CHECK_BASIS and peaks[1] <= 4 * _CHECK_BASIS + 4
    assert peaks[0] <= 2 * (ncv + 1) + 3 * modes + 11


def _certify_reference(a: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The block Arnoldi of certify_subspace on a dense matrix with growing column stacks and complex Ritz products."""

    def extend(basis: np.ndarray, block: np.ndarray) -> np.ndarray:
        """The basis with the orthonormalized columns of block appended, cut to the check's basis size."""
        for column in block.T:
            remainder = column.copy()
            for _ in range(2):
                remainder -= basis @ (basis.T @ remainder)
            if np.linalg.norm(remainder) > 1e-8 * np.linalg.norm(column):
                basis = np.column_stack([basis, remainder / np.linalg.norm(remainder)])
        return basis[:, :_CHECK_BASIS]

    basis, images = extend(np.zeros((a.shape[0], 0)), start), np.zeros((a.shape[0], 0))
    while basis.shape[1] > images.shape[1]:
        images = np.column_stack([images, a @ basis[:, images.shape[1] :]])
        theta, y, _ = _schur_ritz(basis.T @ images)
        ritz = basis @ y
        residuals = images @ y - ritz * theta
        res = np.linalg.norm(residuals, axis=0) / np.linalg.norm(ritz, axis=0)
        lead = int(np.argmax(theta.real))
        if res[lead] <= _RITZ_GATE * max(1.0, abs(theta[lead])):
            break
        basis = extend(basis, np.column_stack([residuals[:, lead].real, residuals[:, lead].imag]))
    order = np.argsort((1.0 - theta).real)
    return (1.0 - theta)[order], res[order], (ritz / np.linalg.norm(ritz, axis=0))[:, order]


@pytest.mark.parametrize("overlap", [0.9, None])
def test_certify_subspace_matches_the_column_stack_arnoldi_to_rounding(overlap, quiet_logger):
    """Preallocated arrays and real products give the column-stack Arnoldi's Ritz pairs to 1e-12, vectors to sign."""
    a = _normal_matrix(40, seed=1)
    rng = np.random.default_rng(6)
    start = rng.standard_normal((40, 3)) if overlap is None else _start_near(_leading_vector(a), overlap, seed=4)
    comm = conftest.create_comm_mock()
    lam_pi, res, u, n_products, _ = certify_subspace(_make_matrix_operator(a, comm), start, comm)
    lam_ref, res_ref, u_ref = _certify_reference(a, start)
    assert lam_pi.size == lam_ref.size == n_products and np.allclose(lam_pi, lam_ref, atol=1e-12)
    assert np.allclose(res, res_ref, atol=1e-12)
    assert np.allclose(np.abs(np.sum(u_ref.conj() * u, axis=0)), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "size, dtype, inactive",
    [(1, np.complex128, None), (1, np.complex64, None), (2, np.complex128, None), (1, np.complex128, [1, 2])],
)
def test_exact_jacobian_recomputing_three_leg_vertices_per_window_gives_the_held_products(
    size, dtype, inactive, loop_state, monkeypatch
):
    """Three-leg vertices recomputed per window give the held products to 10x the floor, reusing the build's scan."""
    base, mu0 = _start(loop_state)
    monkeypatch.setattr(npb, "DTYPE", dtype)
    k_grid = config.lattice.k_grid
    ys = np.random.default_rng(13).standard_normal((2 * k_grid.nk_irr * config.sys.n_bands**2 * config.box.niv_core, 2))
    monkeypatch.setattr(mpi_utils, "MPI", conftest.FAKE_MPI)
    monkeypatch.setattr(nonlocal_sde, "MPI", conftest.FAKE_MPI)
    real_scan, scans = LocalFourPoint.orbital_pairs_without_vertex, []

    def scan(gamma, u_r):
        scans.append(1)
        return real_scan(gamma, u_r) if inactive is None else np.array(inactive)

    monkeypatch.setattr(LocalFourPoint, "orbital_pairs_without_vertex", scan)

    def fn(comm, rank):
        dist_irrk = MpiDistributor.create_distributor(ntasks=k_grid.nk_irr, comm=comm, name="Q")
        dist_fullbz = MpiDistributor.create_distributor(ntasks=k_grid.nk_tot, comm=comm, name="FBZ")
        products, held = [], []
        for per_window in (False, True):
            budgets = memory_estimator.ChunkBudgets(exact_block=2, exact_window=4, exact_vrg_per_window=per_window)
            jac = _rank_jacobian(loop_state, base, mu0, budgets, comm, dist_irrk, dist_fullbz)
            held.append(sorted(channel.value for channel in jac._vrg))
            products.append(jac.matvec_block(ys if rank == 0 else None))
            jac.free()
        return products, held

    _, results = conftest.run_parallel(size, fn)
    (held_products, window_products), held = results[0]
    assert held == [["dens", "magn"], []] and len(scans) == 4 * size
    atol = conftest.ten_times_the_floor(held_products, dtype)
    assert np.allclose(window_products, held_products, rtol=0, atol=atol)


@pytest.mark.parametrize("size, dtype", [(1, np.complex128), (1, np.complex64), (2, np.complex128), (2, np.complex64)])
def test_exact_jacobian_solves_recomputed_three_leg_vertices_fused_or_in_two_passes_with_the_same_products(
    size, dtype, loop_state, monkeypatch
):
    """Recomputed three-leg vertices give the same products, to 10x the floor, fused per slice or in two passes."""
    base, mu0 = _start(loop_state)
    monkeypatch.setattr(npb, "DTYPE", dtype)
    k_grid = config.lattice.k_grid
    ys = np.random.default_rng(15).standard_normal((2 * k_grid.nk_irr * config.sys.n_bands**2 * config.box.niv_core, 2))
    monkeypatch.setattr(mpi_utils, "MPI", conftest.FAKE_MPI)
    monkeypatch.setattr(nonlocal_sde, "MPI", conftest.FAKE_MPI)
    hooks = MagicMock(side_effect=ExactJacobian._response_rhs)
    monkeypatch.setattr(ExactJacobian, "_response_rhs", hooks)

    def fn(comm, rank):
        dist_irrk = MpiDistributor.create_distributor(ntasks=k_grid.nk_irr, comm=comm, name="Q")
        dist_fullbz = MpiDistributor.create_distributor(ntasks=k_grid.nk_tot, comm=comm, name="FBZ")
        budgets = memory_estimator.ChunkBudgets(exact_block=2, exact_window=4, exact_vrg_per_window=True)
        jac = _rank_jacobian(loop_state, base, mu0, budgets, comm, dist_irrk, dist_fullbz)
        products = jac.matvec_block(ys if rank == 0 else None)
        jac.free()
        return products

    products, hook_calls = [], []
    for crossover in (0, 2**40):
        monkeypatch.setattr("dgamore.sigma_jacobian._FUSED_SOLVE_MIN_COMPOUND", crossover)
        hooks.reset_mock()
        products.append(conftest.run_parallel(size, fn)[1][0])
        hook_calls.append(hooks.call_count)
    assert hook_calls[0] > 0 and hook_calls[1] == 0
    assert np.allclose(products[0], products[1], rtol=0, atol=conftest.ten_times_the_floor(products[1], dtype))


_REFERENCE = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/end_2_end/exact_jacobian_reference.npz"


def _reference_start(state, reference, lattice: str) -> tuple[SelfEnergy, float]:
    """The reference's stored DMFT-start proposal and chemical potential, with the loop's occupations there."""
    # stored, not recomputed: another machine's BLAS rounding moves the proposal by 1e-14, the products by 1e-11
    base = SelfEnergy(reference[f"{lattice}_base"], config.lattice.nk, True, True, beta=config.sys.beta)
    return base, _mu_and_occupations(state, base, float(reference[f"{lattice}_mu0"]), fixed=True)


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize(
    "lattice, budgets",
    [
        ("fixture", {}),
        ("fixture", {"exact_block": 2, "exact_window": 4, "exact_vrg_per_window": True}),
        ("fixture", {"exact_block": 3, "exact_window": 7, "exact_vertices_per_window": True}),
        ("p_orbitals", {"exact_block": 2, "exact_window": 4}),
        ("complex_hopping", {"exact_block": 2, "exact_window": 4, "exact_vrg_per_window": True}),
        ("one_band", {"exact_window": 1}),
        ("one_band", {"exact_block": 2, "exact_window": 3, "exact_vertices_per_window": True}),
    ],
)
def test_exact_jacobian_products_match_the_reference_operator_within_ten_times_the_floating_point_floor(
    lattice, budgets, dtype, monkeypatch, tmp_path
):
    """J y is the reference's to 10x its floor: its BLAS-thread spread, or 1e-14 (c128) or its c128 gap (c64)."""
    reference = np.load(_REFERENCE)
    state = _loop_state(monkeypatch, lattice, tmp_path)
    base, mu0 = _reference_start(state, reference, lattice)
    monkeypatch.setattr(npb, "DTYPE", dtype)
    name = "c128" if dtype == np.complex128 else "c64"
    jac = _jacobian(state, base, mu0, memory_estimator.ChunkBudgets(**budgets))
    products = jac.matvec_block(reference[f"{lattice}_{name}_ys"])
    jac.free()
    exact, stored = reference[f"{lattice}_c128_products"], reference[f"{lattice}_{name}_products"]
    gap = 1e-14 * np.abs(exact).max() if name == "c128" else np.abs(stored - exact).max()
    assert np.allclose(products, stored, rtol=0, atol=10 * max(gap, float(reference[f"{lattice}_{name}_spread"])))


def test_exact_jacobian_keeps_the_reference_operators_eigenpairs_within_twice_their_residuals(loop_state, monkeypatch):
    """The reference operator's leading eigenpairs stay eigenpairs within twice their residual there or their bound."""
    # double precision only: in single precision the residuals depend on the BLAS kernel (5e-3 to 0.26)
    reference = np.load(_REFERENCE)
    base, mu0 = _reference_start(loop_state, reference, "fixture")
    monkeypatch.setattr(npb, "DTYPE", np.complex128)
    jac = _jacobian(loop_state, base, mu0, memory_estimator.ChunkBudgets(exact_block=4, exact_window=4))
    lam_pi, res, u, before = (reference[f"fixture_c128_{key}"] for key in ("lam_pi", "res", "u", "residual"))
    images = jac.matvec_block(np.column_stack([u.real, u.imag]))
    jac.free()
    n = u.shape[1]
    residual = np.linalg.norm(images[:, :n] + 1j * images[:, n:] - (1.0 - lam_pi) * u, axis=0)
    assert n > 0 and np.all(residual <= 2 * np.maximum(res, before))


def test_exact_jacobian_in_single_precision_keeps_the_reference_eigenvalues(loop_state, monkeypatch):
    """The complex64 operator's Rayleigh quotients on the exact eigenvectors keep the eigenvalues to 5e-3."""
    # projecting on the eigenvector drops the BLAS-dependent rounding: within 6.5e-4 on five OpenBLAS kernels
    reference = np.load(_REFERENCE)
    base, mu0 = _reference_start(loop_state, reference, "fixture")
    monkeypatch.setattr(npb, "DTYPE", np.complex64)
    jac = _jacobian(loop_state, base, mu0, memory_estimator.ChunkBudgets(exact_block=4, exact_window=4))
    lam_pi, u = reference["fixture_c128_lam_pi"], reference["fixture_c128_u"]
    images = jac.matvec_block(np.column_stack([u.real, u.imag]))
    jac.free()
    n = u.shape[1]
    rayleigh = np.sum(u.conj() * (images[:, :n] + 1j * images[:, n:]), axis=0) / np.sum(np.abs(u) ** 2, axis=0)
    error = np.abs(rayleigh - (1.0 - lam_pi)) / np.abs(1.0 - lam_pi)
    assert n > 0 and np.all(error < 5e-3)


@pytest.mark.slow
@pytest.mark.parametrize("block", [1, 4])
def test_exact_jacobian_leading_eigenpairs_are_the_reference_operators(loop_state, monkeypatch, block):
    """ARPACK and the block search on the windowed operator find the reference's leading eigenvalues to tolerance."""
    reference = np.load(_REFERENCE)
    base, mu0 = _reference_start(loop_state, reference, "fixture")
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
    jac = _jacobian(loop_state, base, mu0, memory_estimator.ChunkBudgets(exact_window=4, exact_vrg_per_window=True))
    lam_pi, res, _ = leading_eigenpairs(jac, loop_state.comm, block=block)
    jac.free()
    expected = reference["fixture_c128_lam_pi"]
    assert lam_pi.size == expected.size and np.allclose(lam_pi, expected, rtol=0, atol=10 * res.max())


@pytest.mark.parametrize("size, hostnames", [(2, ["n0", "n0"]), (2, ["n0", "n1"]), (7, ["n0"] * 4 + ["n1"] * 3)])
def test_exact_jacobian_products_on_any_rank_layout_do_not_depend_on_the_window_the_residency_or_the_block(
    size, hostnames, loop_state, monkeypatch
):
    """On one or two nodes, empty ranks included, windows, residencies and blocks move no product past 10x the floor."""
    base, mu0 = _start(loop_state)
    k_grid = config.lattice.k_grid
    ys = np.random.default_rng(14).standard_normal((2 * k_grid.nk_irr * config.sys.n_bands**2 * config.box.niv_core, 3))
    monkeypatch.setattr(mpi_utils, "MPI", conftest.FAKE_MPI)
    monkeypatch.setattr(nonlocal_sde, "MPI", conftest.FAKE_MPI)
    specs = [{}, {"exact_block": 2, "exact_window": 4, "exact_vrg_per_window": True}]
    specs.append({"exact_block": 3, "exact_window": 1, "exact_vertices_per_window": True})

    def fn(comm, rank):
        dist_irrk = MpiDistributor.create_distributor(ntasks=k_grid.nk_irr, comm=comm, name="Q")
        dist_fullbz = MpiDistributor.create_distributor(ntasks=k_grid.nk_tot, comm=comm, name="FBZ")
        products = []
        for spec in specs:
            budgets = memory_estimator.ChunkBudgets(**spec)
            jac = _rank_jacobian(loop_state, base, mu0, budgets, comm, dist_irrk, dist_fullbz)
            products.append(jac.matvec_block(ys if rank == 0 else None))
            jac.free()
        return products, dist_irrk.my_size

    _, results = conftest.run_parallel(size, fn, hostnames=hostnames)
    products, sizes = results[0][0], [my_size for _, my_size in results]
    assert all(p is None for products_r, _ in results[1:] for p in products_r) and (min(sizes) == 0) == (size == 7)
    # the settings change the shapes and the concurrency of the BLAS calls, whose bits a BLAS need not keep
    atol = conftest.ten_times_the_floor(products[0])
    assert all(np.allclose(p, products[0], rtol=0, atol=atol) for p in products[1:])


def test_rank_jacobians_clone_no_interaction_the_fake_ranks_share(loop_state, monkeypatch):
    """No fake rank reduces or finalizes on the loop state's V(q): cloning it parks its mat at None for the others."""
    base, mu0 = _start(loop_state)
    k_grid = config.lattice.k_grid
    ys = np.random.default_rng(16).standard_normal(2 * k_grid.nk_irr * config.sys.n_bands**2 * config.box.niv_core)
    monkeypatch.setattr(mpi_utils, "MPI", conftest.FAKE_MPI)
    monkeypatch.setattr(nonlocal_sde, "MPI", conftest.FAKE_MPI)
    shared, real_clone, cloned_by = loop_state.v_full, npb.IHaveMat._clone_without_mat, []

    def clone(self):
        """Records a rank thread's clone of the shared V(q)."""
        if self is shared and threading.current_thread() is not threading.main_thread():
            cloned_by.append(threading.current_thread().name)
        return real_clone(self)

    monkeypatch.setattr(npb.IHaveMat, "_clone_without_mat", clone)

    def fn(comm, rank):
        dist_irrk = MpiDistributor.create_distributor(ntasks=k_grid.nk_irr, comm=comm, name="Q")
        dist_fullbz = MpiDistributor.create_distributor(ntasks=k_grid.nk_tot, comm=comm, name="FBZ")
        jac = _rank_jacobian(loop_state, base, mu0, memory_estimator.ChunkBudgets(), comm, dist_irrk, dist_fullbz)
        jac.matvec(ys if rank == 0 else None)
        jac.free()

    conftest.run_parallel(2, fn, hostnames=["n0", "n1"])
    assert cloned_by == []


def _window_case(
    nb: int, n_q: int = 3, wp: int = 5, niv: int = 3, niv_full: int = 5, m: int = 2
) -> types.SimpleNamespace:
    """The bubble of G and m bubble responses (full box), local vertices, U and V(q) of a synthetic window test."""
    rng = np.random.default_rng(7)
    c = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    config.sys.beta, config.sys.n_bands, config.box.niw_core = 10.0, nb, wp - 1
    config.box.niv_core, config.box.niv_full = niv, niv_full
    config.lattice.k_grid = bz.KGrid((max(n_q, 1), 1, 1), symmetries=[])
    vc, vf, eye = 2 * niv, 2 * niv_full, np.eye(nb)
    identity = np.einsum("ad,bc->abcd", eye, eye)[None, ..., None, None]
    bubbles = [-0.5 * identity + 0.05 * c(n_q, nb, nb, nb, nb, wp, vf)]
    bubbles += [0.05 * c(n_q, nb, nb, nb, nb, wp, vf) for _ in range(m)]
    gammas = [0.3 * c(nb, nb, nb, nb, wp, vc, vc) for _ in _CHANNELS]
    f_dc = 0.3 * c(nb, nb, nb, nb, wp, vf, vc)
    u_loc, v = rng.standard_normal((nb,) * 4) + 0.3j * rng.standard_normal((nb,) * 4), 0.2 * c(n_q, nb, nb, nb, nb)
    return types.SimpleNamespace(n_q=n_q, wp=wp, bubbles=bubbles, gammas=gammas, f_dc=f_dc, u_loc=u_loc, v=v)


def _window_operator(case, width: int, per_window: bool = False) -> ExactJacobian:
    """An operator on the case's held vertices, channels and windows, its residents built window by window."""
    jac = ExactJacobian.__new__(ExactJacobian)
    u_loc = LocalInteraction(case.u_loc)
    jac._beta, jac._u_loc, jac._vertices_per_window, jac._vertices = config.sys.beta, u_loc, False, {}
    jac._node_comm, jac._dist = conftest.create_comm_mock(), types.SimpleNamespace(my_size=case.n_q)
    jac._v_nonloc = Interaction(case.v, SpinChannel.NONE, (case.n_q, 1, 1), True)
    stored = np.ascontiguousarray(case.f_dc.transpose(_F_DC[2]))
    f_dc = LocalFourPoint(stored.transpose(np.argsort(_F_DC[2])), SpinChannel.MAGN, 1, 2, False, True)
    jac._vertices[_F_DC[0]] = (f_dc, None)
    for (channel, _), gamma in zip(_CHANNELS, case.gammas):
        jac._vertices[f"gamma_{channel.value}_loc.npy"] = (LocalFourPoint(gamma, channel, 1, 2, False, True), None)
    jac._w_windows = [(w0, min(case.wp, w0 + width)) for w0 in range(0, case.wp, width)]
    jac._channels = []
    for channel, weight in _CHANNELS:
        u_r = u_loc.as_channel(channel)
        jac._channels.append(
            (weight, channel, jac._v_nonloc.as_channel(channel) + u_r, jac._inactive_pairs(channel, u_r))
        )
    jac._kernel, jac._gchi0_inv = jac._held(SpinChannel.NONE, 1), jac._held(SpinChannel.NONE, 1, _INVERSE_ORDER)
    jac._chi = {channel: jac._held(channel, 0) for channel, _ in _CHANNELS}
    jac._vrg = {} if per_window else {channel: jac._held(channel, 1) for channel, _ in _CHANNELS}
    with deferred_collection():
        for w0, w1 in jac._w_windows:
            jac._build_window(FourPoint(case.bubbles[0][..., w0:w1, :].copy(), *_window_meta(case)), (w0, w1))
    return jac


def _window_meta(case) -> tuple:
    """The metadata of a compressed one-fermion object on the case's momenta (half niw range)."""
    return SpinChannel.NONE, (max(case.n_q, 1), 1, 1), 1, 1, False, True, True


def _window_parts(jac: ExactJacobian, case, w_window: tuple[int, int]) -> list[tuple[FourPoint, ...]]:
    """The double-counting split of the case's bubble responses on one window, as a product runs it."""
    w0, w1 = w_window
    responses = [FourPoint(r[..., w0:w1, :].copy(), *_window_meta(case)) for r in case.bubbles[1:]]
    with jac._local_vertex(*_F_DC, w_window=w_window) as f_dc:
        return [jac._split_bubble(r, nonlocal_sde._contract_dc_vertex(f_dc, r)) for r in responses]


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("nb, n_q", [(1, 3), (2, 3), (2, 0)])
def test_window_residents_and_kernel_responses_do_not_depend_on_the_window_or_the_residency(
    dtype, nb, n_q, monkeypatch
):
    """Windows of one, two (last ragged) or five frequencies, held or recomputed vertices: one result, 10x the floor."""
    monkeypatch.setattr(npb, "DTYPE", dtype)
    case, results = _window_case(nb, n_q=n_q), []
    for width, per_window in ((5, False), (1, False), (2, True), (5, True)):
        jac = _window_operator(case, width, per_window)
        dkernels = [[] for _ in case.bubbles[1:]]
        with deferred_collection():
            for window in jac._w_windows:
                for j, dkernel in enumerate(jac._kernel_responses(_window_parts(jac, case, window), window)):
                    dkernels[j].append(dkernel.mat.copy())
        held = [jac._kernel.mat, jac._gchi0_inv.mat, *(chi.mat for chi in jac._chi.values())]
        results.append([*(x.copy() for x in held), *(np.concatenate(d, axis=-2) for d in dkernels)])
    shapes = [x.shape for x in results[0]]
    assert shapes[0] == (n_q, nb, nb, nb, nb, 5, 6) and shapes[-1] == shapes[0] and results[0][0].dtype == dtype
    pairs = [(a, b) for result in results[1:] for a, b in zip(result, results[0])]
    assert all(np.allclose(a, b, rtol=0, atol=conftest.ten_times_the_floor(b)) for a, b in pairs)


@pytest.mark.parametrize("per_window, crossover, passes", [(False, 320, 1), (True, 0, 1), (True, 320, 2)])
def test_window_bethe_salpeter_pass_factorizes_every_slice_once_per_channel_and_pass(
    per_window, crossover, passes, monkeypatch
):
    """Held or fused three-leg vertices factorize each slice once per frequency and channel, two passes twice."""
    monkeypatch.setattr("dgamore.sigma_jacobian._FUSED_SOLVE_MIN_COMPOUND", crossover)
    case = _window_case(2, m=2)
    jac = _window_operator(case, 2, per_window)
    factorizations, factorization, solve = [], four_point._factorization, four_point._factorize_and_solve
    monkeypatch.setattr(
        four_point, "_factorization", lambda buf, sym: factorizations.append(sym) or factorization(buf, sym)
    )
    monkeypatch.setattr(
        four_point, "_factorize_and_solve", lambda buf, rhs, sym: factorizations.append(sym) or solve(buf, rhs, sym)
    )
    with deferred_collection():
        for window in jac._w_windows:
            jac._kernel_responses(_window_parts(jac, case, window), window)
    assert len(factorizations) == passes * len(_CHANNELS) * case.n_q * case.wp


@pytest.mark.parametrize(
    "nb, n_q, wp, m, per_window, crossover",
    [(1, 8, 32, 2, False, 320), (2, 6, 16, 1, True, 320), (2, 6, 16, 3, True, 0), (2, 6, 16, 3, False, 320)],
)
def test_window_stages_lie_between_their_parts_and_the_modeled_bytes(
    nb, n_q, wp, m, per_window, crossover, monkeypatch
):
    """The traced dc split and Bethe-Salpeter pass of a window exceed their parts and stay within the model + 64 KiB."""
    monkeypatch.setattr("dgamore.sigma_jacobian._FUSED_SOLVE_MIN_COMPOUND", crossover)
    monkeypatch.setattr(npb, "DTYPE", np.complex128)
    monkeypatch.setattr(memory_estimator, "DTYPE_BYTES", 16)
    monkeypatch.setattr(gc, "collect", lambda *args, **kwargs: 0)
    case = _window_case(nb, n_q=n_q, wp=wp, niv=8, niv_full=10, m=m)
    jac = _window_operator(case, wp, per_window)
    parts, traced = {}, {}
    with deferred_collection():
        traced["dc"] = conftest.traced_peak(lambda: parts.setdefault("dc", _window_parts(jac, case, (0, wp))))[1]
        traced["bse"] = conftest.traced_peak(lambda: jac._kernel_responses(parts["dc"], (0, wp)))[1]
    dc, bse = memory_estimator.exact_window_stages(n_q, nb, wp, 8, 10, m, per_window)
    core = 16 * n_q * nb**4 * wp * 16
    assert m * 2 * core < traced["dc"] <= dc + 2**16 and m * core < traced["bse"] <= bse + 2**16


@pytest.mark.parametrize("nb, n_q, wp, per_window", [(1, 8, 32, False), (2, 6, 16, True), (2, 6, 16, False)])
def test_window_build_stays_within_the_modeled_bethe_salpeter_pass_of_one_column(nb, n_q, wp, per_window, monkeypatch):
    """The traced build of one window's residents exceeds a core block and stays within a one-column pass + 64 KiB."""
    monkeypatch.setattr(npb, "DTYPE", np.complex128)
    monkeypatch.setattr(memory_estimator, "DTYPE_BYTES", 16)
    monkeypatch.setattr(gc, "collect", lambda *args, **kwargs: 0)
    case = _window_case(nb, n_q=n_q, wp=wp, niv=8, niv_full=10, m=0)
    jac = _window_operator(case, wp, per_window)
    bubble = FourPoint(case.bubbles[0].copy(), *_window_meta(case))
    with deferred_collection():
        peak = conftest.traced_peak(lambda: jac._build_window(bubble, (0, wp)))[1]
    bse = memory_estimator.exact_window_stages(n_q, nb, wp, 8, 10, 1, per_window)[1]
    assert 16 * n_q * nb**4 * wp * 16 < peak <= bse + 2**16


def test_unfolding_a_sector_vector_peaks_at_two_core_windows(loop_state):
    """The traced unfolding of a sector vector, which every node root runs per column, stays within two core windows."""
    k_grid, nb, niv = config.lattice.k_grid, config.sys.n_bands, config.box.niv_core
    y = np.random.default_rng(16).standard_normal(2 * k_grid.nk_irr * nb**2 * niv)
    window, peak = conftest.traced_peak(lambda: _window(y))
    assert window.nbytes < peak <= 2 * window.nbytes + 2**16


def test_squared_box_sum_embeds_the_self_energy_of_the_correlated_orbitals():
    """The squared box sum with orbitals equals the plain one of the embedded self-energy (Emery, n_d = 1)."""
    ek = conftest.emery_hk((2, 2, 1)).reshape(-1, 3, 3)
    rows = (0.3 - 0.4j / (1j * MFHelper.vn(6, 7.0)))[None, None, None, :] * np.ones((4, 1, 1, 1))
    embedded = np.zeros((4, 3, 3, 12), dtype=complex)
    embedded[:, :1, :1] = rows
    expected = _squared_box_sum(embedded, 0.2, ek, 7.0, True)
    got = _squared_box_sum(rows, 0.2, ek, 7.0, True, orbitals=np.array([0]))
    assert got.shape == (4, 3, 3) and np.allclose(got, expected, rtol=0, atol=1e-13)

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import gc
import os
import tracemalloc
import types
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
import scipy.sparse.linalg as spla

import dgamore.brillouin_zone as bz
import dgamore.n_point_base as npb
from dgamore import config, dga_io, memory_estimator, mpi_utils, nonlocal_sde
from dgamore.bubble_gen import BubbleGenerator
from dgamore.dga_logger import DgaLogger
from dgamore.four_point import FourPoint
from dgamore.local_four_point import LocalFourPoint
from dgamore.n_point_base import SpinChannel
from dgamore.greens_function import GreensFunction, update_mu
from dgamore.interaction import Interaction, LocalInteraction
from dgamore.jacobian_stabilization import _RITZ_GATE, JacobianTracker, _schur_ritz
from dgamore.mpi_utils import MpiDistributor
from dgamore.self_energy import SelfEnergy
from dgamore.sigma_jacobian import (
    _CHECK_BASIS,
    ExactJacobian,
    _merge_modes,
    _squared_box_sum,
    certify_subspace,
    leading_eigenpairs,
    sector_projector,
    tracker_sector_maps,
)
from tests import conftest


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


def _first_orbital(monkeypatch, s_dmft: SelfEnergy, u_loc: LocalInteraction) -> tuple[SelfEnergy, LocalInteraction]:
    """Restricts the loaded fixture to its first orbital: DMFT self-energy, dispersion, U and the stored vertices."""
    real_load = LocalFourPoint.load

    def load(filename: str, channel: SpinChannel = SpinChannel.NONE) -> LocalFourPoint:
        """The first-orbital block of a stored local vertex."""
        v = real_load(filename, channel)
        mat = v.mat[:1, :1, :1, :1].copy()
        return LocalFourPoint(
            mat, channel, v.num_wn_dimensions, v.num_vn_dimensions, v.full_niw_range, v.full_niv_range
        )

    monkeypatch.setattr(LocalFourPoint, "load", staticmethod(load))
    config.sys.n_bands = 1
    config.sys.occ_dmft = config.sys.occ_dmft[:1, :1]
    config.lattice.hamiltonian.set_ek(config.lattice.hamiltonian.get_ek()[..., :1, :1].copy())
    s_one = SelfEnergy(s_dmft.mat[..., :1, :1, :].copy(), (1, 1, 1), True, False, beta=config.sys.beta)
    return s_one, LocalInteraction(u_loc.mat[:1, :1, :1, :1].copy())


def _loop_state(monkeypatch, lattice: str = "fixture") -> types.SimpleNamespace:
    """The end_2_end fixture's fresh-start loop state on its own, the p_x/p_y, complex-hopping or one-band lattice."""
    monkeypatch.setattr(npb, "DTYPE", np.complex128)
    monkeypatch.setattr(gc, "collect", lambda *args, **kwargs: 0)
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/end_2_end"
    comm = conftest.create_comm_mock()
    monkeypatch.setattr("mpi4py.MPI.COMM_WORLD", comm)
    monkeypatch.setattr(config, "logger", DgaLogger(comm, "./"), raising=False)
    conftest.create_default_config(config, folder)
    config.box.niw_core, config.box.niv_core, config.box.niv_shell = 20, 20, 10
    config.dmft.symmetrize_orbitals = []

    _, s_dmft, _, _ = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())
    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]
    config.output.output_path = folder
    u_loc = config.lattice.hamiltonian.get_local_u()
    if lattice == "one_band":
        s_dmft, u_loc = _first_orbital(monkeypatch, s_dmft, u_loc)
    if lattice == "p_orbitals":
        # the stored vertices keep the sign of either orbital exactly but not the swap of the two, so no swap here
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
    nb = ek.shape[-1]
    occ_k = GreensFunction.get_g_full(start, config.sys.mu, ek.reshape(-1, 1, 1, nb, nb), config.sys.beta)
    config.sys.n, config.sys.occ, config.sys.occ_k = nonlocal_sde._assemble_occupation(
        occ_k.get_fill_nonlocal()[2], dist_fullbz
    )
    sigma_dmft = s_dmft.cut_niv(niv_cut)
    v_nonloc = config.lattice.hamiltonian.get_vq(k_grid)
    if lattice == "complex_hopping":
        v_nonloc = _nearest_neighbor_v(k_grid.nk, 0.3)
    elif lattice == "one_band":
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


def _mu_and_occupations(state, sigma, mu_start: float) -> float:
    """The chemical potential of the held filling and, in config.sys, the occupations of the loop at it."""
    s = sigma.copy().compress_q_dimension()
    mu = update_mu(mu_start, config.sys.n, state.ek, s.mat, config.sys.beta, s.fit_smom()[0], tol=1e-14)
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


def _start(state) -> tuple[SelfEnergy, float]:
    """The proposal of the DMFT start and the chemical potential of the held filling at it (see _mu_and_occupations)."""
    base = _proposal(state, state.sigma_dmft, config.sys.mu)
    return base, _mu_and_occupations(state, base, config.sys.mu)


def _jacobian(state, base, mu0: float, budgets: memory_estimator.ChunkBudgets | None = None) -> ExactJacobian:
    """The exact Jacobian of the loop map at ``base`` on the state's own distributors and communicator."""
    held = (state.u_loc, state.v_irr, state.v_full, state.sigma_dmft_full, state.dist_irrk, state.dist_fullbz)
    return ExactJacobian(base, mu0, *held, state.comm, budgets)


def _rank_jacobian(state, base, mu0: float, budgets, comm, dist_irrk, dist_fullbz) -> ExactJacobian:
    """The exact Jacobian on a rank's communicator and distributors, from its slice of V(q) and copies of the inputs."""
    v_irr = state.v_full.reduce_q(state.irr_q_list[dist_irrk.my_slice])
    held = (state.u_loc, v_irr, state.v_full, state.sigma_dmft_full.copy(), dist_irrk, dist_fullbz)
    return ExactJacobian(base.copy(), mu0, *held, comm, budgets)


@pytest.fixture
def quiet_logger(monkeypatch):
    """A MagicMock in place of config.logger."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)


@pytest.mark.parametrize("lattice", ["fixture", "p_orbitals", "complex_hopping", "one_band"])
def test_exact_jacobian_matches_central_differences_of_the_loop_map(lattice, monkeypatch):
    """J y equals the central difference of the loop's map along a reachable direction, mu and occupations included."""
    state = _loop_state(monkeypatch, lattice)
    base, mu0 = _start(state)
    jac = _jacobian(state, base, mu0)
    assert jac._tr_average == (lattice != "one_band")
    x0 = _core(base).copy()
    y = jac.project(np.random.default_rng(3).standard_normal(jac.n_real))
    dx = np.reshape(_window(y), x0.shape)
    y *= 1e-2 * np.linalg.norm(x0) / np.linalg.norm(dx)
    dx = np.reshape(_window(y), x0.shape)
    assert np.allclose(_sector(dx), y, atol=1e-14 * np.abs(y).max())

    jy = jac.matvec(y)
    jac.free()

    h = 1e-5  # the central difference converges as h^2 to 1e-9 here; its truncation is 2e-7 at h = 1e-4
    sides = []
    for sign in (1.0, -1.0):
        sigma = _with_core(base, x0 + sign * h * dx)
        sides.append(_core(_proposal(state, sigma, _mu_and_occupations(state, sigma, mu0))))
    fd = _sector((sides[0] - sides[1]) / (2 * h))

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


def test_exact_jacobian_streams_its_kernel_phase_one_momentum_group_at_a_time(loop_state, monkeypatch):
    """A budget of one momentum per group solves each channel once per momentum and gives the whole-block products."""
    base, mu0 = _start(loop_state)
    products, calls = {}, {}
    solve = MagicMock(side_effect=nonlocal_sde.create_auxiliary_chi_r_q_sum)
    monkeypatch.setattr(nonlocal_sde, "create_auxiliary_chi_r_q_sum", solve)
    for budget in (1, memory_estimator.MAX_CHUNK_BUDGET_BYTES):
        jac = _jacobian(loop_state, base.copy(), mu0, memory_estimator.ChunkBudgets(exact=budget, exact_block=2))
        ys = np.column_stack([jac.project(y) for y in np.random.default_rng(5).standard_normal((2, jac.n_real))])
        solve.reset_mock()
        products[budget], calls[budget] = jac.matvec_block(ys), solve.call_count
        jac.free()
    assert calls == {1: 2 * loop_state.dist_irrk.my_size, memory_estimator.MAX_CHUNK_BUDGET_BYTES: 2}
    whole = products[memory_estimator.MAX_CHUNK_BUDGET_BYTES]
    assert np.allclose(products[1], whole, atol=1e-12 * np.abs(whole).max())


def test_exact_jacobian_loads_its_local_vertices_per_phase_with_the_same_products(loop_state, monkeypatch):
    """Vertices loaded for their phase give the held vertices' products bit for bit, one load per vertex and piece."""
    base, mu0 = _start(loop_state)
    real_load, loads, products, held = nonlocal_sde._load_node_shared_local_vertex, [], {}, {}

    def load(node_comm, path, channel, **kwargs):
        loads.append(os.path.basename(path))
        return real_load(node_comm, path, channel, **kwargs)

    monkeypatch.setattr(nonlocal_sde, "_load_node_shared_local_vertex", load)
    for per_phase in (False, True):
        budgets = memory_estimator.ChunkBudgets(exact_block=2, exact_vertices_per_phase=per_phase)
        jac = _jacobian(loop_state, base.copy(), mu0, budgets)
        ys = np.column_stack([jac.project(y) for y in np.random.default_rng(6).standard_normal((3, jac.n_real))])
        loads.clear()
        products[per_phase], held[per_phase] = (jac.matvec_block(ys), sorted(loads)), sorted(jac._vertices)
        jac.free()
    names = ["f_dc_loc.npy", "gamma_dens_loc.npy", "gamma_magn_loc.npy"]
    assert np.array_equal(products[True][0], products[False][0])
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
    """A product unfolds its column and evaluates the occupation change once; a block rebuilds all but its last."""
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
    assert counts == (3, 3) and (unfold.call_count, occupation.call_count) == (3 + 5, 3 + 3)
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
    """A rank that owns no irreducible momentum (more ranks than momenta) walks no group and raises nothing."""
    jac = ExactJacobian.__new__(ExactJacobian)
    empty = np.zeros((0, 1, 1, 1, 1, 3, 4), dtype=np.complex64)
    jac._gchi0_inv = FourPoint(empty, SpinChannel.DENS, (1, 1, 1), 1, 1, False, has_compressed_q_dimension=True)
    jac._aux_chunk, jac._beta, jac._channels = 2**20, 10.0, []
    assert jac._kernel_responses([]) == []


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


def test_leading_eigenpairs_reports_only_modes_of_the_projected_sector(quiet_logger):
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
        n_real=40, matvec=lambda y: keep * (a @ (keep * y)), expand=lambda y: y, project=lambda y: keep * y
    )
    lam_pi, _, u = leading_eigenpairs(operator, comm)
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
    lam_pi, res, u, n_products = certify_subspace(
        _make_matrix_operator(a, comm), _start_near(_leading_vector(a), 0.9, seed=4), comm
    )
    j = int(np.argmin(lam_pi.real))
    assert np.isclose(lam_pi[j], -1.0, atol=1e-2) and res[j] <= _RITZ_GATE * 2.0 and 0 < n_products <= _CHECK_BASIS
    assert np.isclose(res[j], np.linalg.norm(a @ u[:, j] - (1.0 - lam_pi[j]) * u[:, j]), atol=1e-12)


def test_certify_subspace_returns_an_uncertified_pair_when_the_budget_runs_out(quiet_logger, monkeypatch):
    """With a budget of one product a random start comes back after that product, its residual above the gate."""
    monkeypatch.setattr("dgamore.sigma_jacobian._CHECK_BASIS", 1)
    comm = conftest.create_comm_mock()
    operator = _make_matrix_operator(_normal_matrix(40, seed=1), comm)
    lam_pi, res, _, n_products = certify_subspace(operator, np.random.default_rng(7).standard_normal((40, 1)), comm)
    assert n_products == operator.matvec_block.call_count == 1 and lam_pi.size == 1
    assert res[0] > _RITZ_GATE * max(1.0, abs(1.0 - lam_pi[0]))


def test_certify_subspace_applies_its_start_columns_and_each_extension_as_one_block(quiet_logger):
    """The start columns go through one block product, and every extension of the basis through one more."""
    comm = conftest.create_comm_mock()
    operator = _make_matrix_operator(_normal_matrix(40, seed=1), comm)
    *_, n_products = certify_subspace(operator, np.random.default_rng(8).standard_normal((40, 3)), comm)
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


@pytest.mark.parametrize("dtype, tol", [(np.complex64, 100 * np.finfo(np.float32).eps), (np.complex128, 1e-10)])
def test_leading_eigenpairs_bounds_the_residual_by_the_tolerance_of_the_storage_precision(dtype, tol, monkeypatch):
    """The residual bound of every pair is max(100 eps, 1e-10) max(1, |theta|), eps of the storage precision."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
    monkeypatch.setattr(npb, "DTYPE", dtype)
    comm = conftest.create_comm_mock()
    lam_pi, res, _ = leading_eigenpairs(_make_matrix_operator(_normal_matrix(40, seed=1), comm), comm)
    assert np.allclose(res, tol * np.maximum(1.0, np.abs(1.0 - lam_pi)), atol=1e-20)


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


def test_exact_product_attaches_the_dc_interaction_after_the_full_box_response_is_freed(loop_state, monkeypatch):
    """A product's bubble stage peaks near the modeled dc attachment beside the column's parts, the full box freed."""
    distributed = BubbleGenerator._create_generalized_chi0_q_fft_distributed

    def column_path(dist, giwk, niw, niv, k_grid, beta, node_comm=None, subtract_from=None):
        return distributed(dist, giwk, niw, niv, k_grid, beta, None, subtract_from)

    monkeypatch.setattr(BubbleGenerator, "create_generalized_chi0_q_fft", staticmethod(column_path))
    jac = _jacobian(loop_state, *_start(loop_state))
    stage_peak, solve = [], jac._kernel_responses

    def kernel_responses(parts):
        stage_peak.append(tracemalloc.get_traced_memory()[1])
        return solve(parts)

    monkeypatch.setattr(jac, "_kernel_responses", kernel_responses)
    y = jac.project(np.random.default_rng(3).standard_normal(jac.n_real))
    tracemalloc.start()
    entry = tracemalloc.get_traced_memory()[0]
    jac._piece_products([y])
    tracemalloc.stop()
    core, vc = jac._gchi0_inv.mat.nbytes, 2 * config.box.niv_core
    jac.free()
    assert config.box.niv_full > config.box.niv_core and stage_peak[0] - entry <= 1.02 * (4 + 2 / vc) * core


def test_exact_jacobian_build_frees_the_full_box_bubble_before_attaching_the_dc_interaction(loop_state, monkeypatch):
    """The operator's build splits and frees the full-box bubble before the dc kernel's interaction is attached."""
    base, mu0 = _start(loop_state)
    bubbles, freed = [], []
    contract, attach = nonlocal_sde._contract_dc_vertex, nonlocal_sde._attach_dc_interaction

    def contract_dc_vertex(f_dc_loc, gchi0_q):
        bubbles.append(gchi0_q)
        return contract(f_dc_loc, gchi0_q)

    def attach_dc_interaction(contraction, u_loc):
        freed.append(bubbles[-1].mat is None)
        return attach(contraction, u_loc)

    monkeypatch.setattr(nonlocal_sde, "_contract_dc_vertex", contract_dc_vertex)
    monkeypatch.setattr(nonlocal_sde, "_attach_dc_interaction", attach_dc_interaction)
    _jacobian(loop_state, base, mu0).free()
    assert freed == [True]


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_exact_product_finalized_in_place_has_the_bits_of_the_out_of_place_sums(loop_state, monkeypatch, dtype):
    """Rank 0's in-place sum, transform and Hartree-Fock terms of both contractions give the out-of-place bits."""
    base, mu0 = _start(loop_state)
    monkeypatch.setattr(npb, "DTYPE", dtype)
    jac = _jacobian(loop_state, base, mu0)
    halves, terms = [], []
    real_sde, real_hartree_fock = nonlocal_sde._run_column_sde, nonlocal_sde.get_hartree_fock

    def column_sde(*args):
        halves.append(real_sde(*args))
        return halves[-1].copy()

    def hartree_fock(*args):
        terms.append(real_hartree_fock(*args))
        return terms[-1]

    monkeypatch.setattr(nonlocal_sde, "_run_column_sde", column_sde)
    monkeypatch.setattr(nonlocal_sde, "get_hartree_fock", hartree_fock)
    product = jac.matvec(jac.project(np.random.default_rng(3).standard_normal(jac.n_real)))
    reference = (halves[0] + halves[1]).ifft().to_full_niv_range() + terms[0][0] + terms[0][1]
    reference.compress_q_dimension().symmetrize_time_reversal()
    expected = jac.project(jac._to_sector(reference.mat))
    jac.free()
    assert jac._tr_average and np.array_equal(product, expected)


def test_exact_product_finalize_holds_two_core_windows_on_rank_0(monkeypatch):
    """Summing, transforming and adding Hartree-Fock in place leaves the full-range completion as the finalize peak."""
    nk, nb, niv = (32, 32, 1), 2, 20
    config.lattice.nk, config.sys.n_bands, config.box.niv_core = nk, nb, niv
    rng = np.random.default_rng(0)
    nk_tot = int(np.prod(nk))
    halves = iter(rng.standard_normal((2, nk_tot, nb, nb, niv)) + 1j * rng.standard_normal((2, nk_tot, nb, nb, niv)))
    u = LocalInteraction(rng.standard_normal((nb,) * 4).astype(complex))
    v = Interaction(rng.standard_normal((*nk, nb, nb, nb, nb)).astype(complex), SpinChannel.NONE, nk)
    monkeypatch.setattr(
        nonlocal_sde,
        "_run_column_sde",
        lambda *a: SelfEnergy(next(halves), nk, False, True, calc_smom=False, beta=10.0),
    )
    monkeypatch.setattr(nonlocal_sde, "_cut_and_reshare_giwk", lambda giwk, win, node_comm, niv_cut: (giwk, None))
    monkeypatch.setattr(nonlocal_sde, "_build_rspace_giwk_window", lambda giwk, node_comm: (None, None))
    monkeypatch.setattr(nonlocal_sde, "_free_shared_window", lambda win, node_comm: None)
    jac = types.SimpleNamespace(
        _greens_function=lambda mat: types.SimpleNamespace(niv=niv, mat=None),
        _comm=types.SimpleNamespace(rank=0),
        _u_loc=u,
        _v_nonloc_full=v,
        _w_r=(u + v).fft(copy=False).decompress_q_dimension().mat,
        _tr_average=False,
        _to_sector=lambda mat: mat[:1].real.ravel(),
        project=lambda y: y,
        **dict.fromkeys(("_dg", "_dg_win", "_node_comm", "_kernel", "_dist", "_sde_chunk", "_g_r", "_g_r_niv")),
    )
    docc, docc_k = rng.standard_normal((nb, nb)) + 0j, rng.standard_normal((*nk, nb, nb)) + 0j
    sigma_core = nk_tot * nb * nb * 2 * niv * np.dtype(npb.DTYPE).itemsize
    _, peak = conftest.traced_peak(lambda: ExactJacobian._contraction(jac, MagicMock(), docc, docc_k))
    assert npb.DTYPE == np.complex64 and peak <= 2.05 * sigma_core


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
    lam_pi, res, u, n_products = certify_subspace(_make_matrix_operator(a, comm), start, comm)
    lam_ref, res_ref, u_ref = _certify_reference(a, start)
    assert lam_pi.size == lam_ref.size == n_products and np.allclose(lam_pi, lam_ref, atol=1e-12)
    assert np.allclose(res, res_ref, atol=1e-12)
    assert np.allclose(np.abs(np.sum(u_ref.conj() * u, axis=0)), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "size, dtype, inactive",
    [(1, np.complex128, None), (1, np.complex64, None), (2, np.complex128, None), (1, np.complex128, [1, 2])],
)
def test_exact_jacobian_recomputing_three_leg_vertices_per_group_gives_the_held_products(
    size, dtype, inactive, loop_state, monkeypatch
):
    """Three-leg vertices recomputed per momentum group give the held products bit for bit, reusing the build's scan."""
    base, mu0 = _start(loop_state)
    monkeypatch.setattr(npb, "DTYPE", dtype)
    k_grid = config.lattice.k_grid
    ys = np.random.default_rng(13).standard_normal((2 * k_grid.nk_irr * config.sys.n_bands**2 * config.box.niv_core, 2))
    monkeypatch.setattr(mpi_utils, "MPI", conftest.FAKE_MPI)
    monkeypatch.setattr(nonlocal_sde, "MPI", conftest.FAKE_MPI)
    vrg_builds = MagicMock(side_effect=nonlocal_sde.create_vrg_r_q)
    monkeypatch.setattr(nonlocal_sde, "create_vrg_r_q", vrg_builds)
    real_scan = nonlocal_sde._inactive_orbital_pairs
    scan = MagicMock(side_effect=lambda gamma, u_r: real_scan(gamma, u_r) if inactive is None else np.array(inactive))
    monkeypatch.setattr(nonlocal_sde, "_inactive_orbital_pairs", scan)

    def fn(comm, rank):
        dist_irrk = MpiDistributor.create_distributor(ntasks=k_grid.nk_irr, comm=comm, name="Q")
        dist_fullbz = MpiDistributor.create_distributor(ntasks=k_grid.nk_tot, comm=comm, name="FBZ")
        products, held = [], []
        for per_group in (False, True):
            budgets = memory_estimator.ChunkBudgets(exact=9 * 1024**2, exact_block=2, exact_vrg_per_group=per_group)
            jac = _rank_jacobian(loop_state, base, mu0, budgets, comm, dist_irrk, dist_fullbz)
            held.append([vrg is not None for _, _, vrg, *_ in jac._channels])
            products.append(jac.matvec_block(ys if rank == 0 else None))
            jac.free()
        return products, held

    _, results = conftest.run_parallel(size, fn)
    (held_products, group_products), held = results[0]
    # four at the two builds of a rank, then one per channel and momentum group, more than one group each here
    assert held == [[True, True], [False, False]] and vrg_builds.call_count > 6 * size and scan.call_count == 4 * size
    assert np.array_equal(held_products, group_products)

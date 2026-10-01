# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import gc
import os
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import dgamore.brillouin_zone as bz
import dgamore.n_point_base as npb
from dgamore import config, dga_io, memory_estimator, nonlocal_sde
from dgamore.dga_logger import DgaLogger
from dgamore.four_point import FourPoint
from dgamore.n_point_base import SpinChannel
from dgamore.greens_function import GreensFunction, update_mu
from dgamore.interaction import Interaction
from dgamore.jacobian_stabilization import _RITZ_GATE, JacobianTracker
from dgamore.mpi_utils import MpiDistributor
from dgamore.self_energy import SelfEnergy
from dgamore.sigma_jacobian import (
    _CHECK_BASIS,
    ExactJacobian,
    _merge_modes,
    certify_subspace,
    leading_eigenpairs,
    sector_projector,
    tracker_sector_maps,
)
from tests import conftest


def _p_orbital_hk(nk: tuple[int, int, int], equal_orbitals: bool = True) -> np.ndarray:
    """A p_x/p_y dispersion with an inter-orbital hopping, whose mirrors flip the sign of one orbital; with equal
    orbitals the diagonal mirrors also swap them."""
    kx, ky, _ = np.meshgrid(*(2 * np.pi * np.arange(n) / n for n in nk), indexing="ij")
    hk = np.zeros((*nk, 2, 2), dtype=complex)
    hk[..., 0, 0] = -2.0 * np.cos(kx) - 0.6 * np.cos(ky)
    hk[..., 1, 1] = -2.0 * np.cos(ky) - 0.6 * np.cos(kx) if equal_orbitals else -1.4 * np.cos(ky) - 0.5 * np.cos(kx)
    hk[..., 0, 1] = hk[..., 1, 0] = 0.3 * np.sin(kx) * np.sin(ky)
    return hk


def _p_orbital_grid(nk: tuple[int, int, int] = (4, 4, 1)) -> tuple[bz.KGrid, np.ndarray]:
    """The auto-symmetry grid of the p_x/p_y dispersion, returned with the dispersion."""
    hk = _p_orbital_hk(nk)
    grid = bz.KGrid(nk=nk, symmetries=[bz.KnownSymmetries.AUTO])
    grid.specify_auto_symmetries(hk)
    return grid, hk


def _complex_hopping_hk(nk: tuple[int, int, int]) -> np.ndarray:
    """The p_x/p_y bands of unequal orbitals joined by the real hoppings 0.3 + 0.4 e^{ik_x}, so H(-k) = H(k)^*."""
    kx = 2 * np.pi * np.arange(nk[0]) / nk[0]
    hk = _p_orbital_hk(nk, equal_orbitals=False)
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


@pytest.fixture
def p_orbital_state(monkeypatch):
    """The same loop state on a p_x/p_y lattice whose auto-discovered mirrors flip the sign of one orbital."""
    return _loop_state(monkeypatch, lattice="p_orbitals")


@pytest.fixture
def complex_hopping_state(monkeypatch):
    """The same loop state on a complex-hopping lattice with a nearest-neighbor V(q), whose occupations are complex."""
    return _loop_state(monkeypatch, lattice="complex_hopping")


def _loop_state(monkeypatch, lattice: str = "fixture") -> types.SimpleNamespace:
    """Builds the end_2_end fixture's fresh-start loop state on its own, the p_x/p_y or the complex-hopping lattice."""
    monkeypatch.setattr(npb, "DTYPE", np.complex128)
    monkeypatch.setattr(gc, "collect", lambda *args, **kwargs: 0)
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/end_2_end"
    comm = conftest.create_comm_mock()
    monkeypatch.setattr("mpi4py.MPI.COMM_WORLD", comm)
    config.logger = DgaLogger(comm, "./")
    conftest.create_default_config(config, folder)
    config.box.niw_core, config.box.niv_core, config.box.niv_shell = 20, 20, 10
    config.dmft.symmetrize_orbitals = []

    _, s_dmft, _, _ = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())
    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]
    config.output.output_path = folder
    if lattice == "p_orbitals":
        # the stored vertices keep the sign of either orbital exactly but not the swap of the two, so no swap here
        config.lattice.hamiltonian.set_ek(_p_orbital_hk(config.lattice.nk, equal_orbitals=False))
    elif lattice == "complex_hopping":
        config.lattice.hamiltonian.set_ek(_complex_hopping_hk(config.lattice.nk))
    if lattice != "fixture":
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
    return types.SimpleNamespace(
        comm=comm,
        u_loc=config.lattice.hamiltonian.get_local_u(),
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


@pytest.mark.parametrize("lattice", ["loop_state", "p_orbital_state", "complex_hopping_state"])
def test_exact_jacobian_matches_central_differences_of_the_loop_map(lattice, request):
    """J y equals the central difference of the loop's map along a reachable direction, mu and occupations included."""
    state = request.getfixturevalue(lattice)
    base = _proposal(state, state.sigma_dmft, config.sys.mu)
    mu0 = _mu_and_occupations(state, base, config.sys.mu)
    jac = ExactJacobian(
        base,
        mu0,
        state.u_loc,
        state.v_irr,
        state.v_full,
        state.sigma_dmft_full,
        state.dist_irrk,
        state.dist_fullbz,
        state.comm,
    )
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

    assert np.allclose(jy, fd, atol=1e-7 * np.abs(fd).max())


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
    state = loop_state
    base = _proposal(state, state.sigma_dmft, config.sys.mu)
    mu0 = _mu_and_occupations(state, base, config.sys.mu)
    jac = ExactJacobian(
        base,
        mu0,
        state.u_loc,
        state.v_irr,
        state.v_full,
        state.sigma_dmft_full,
        state.dist_irrk,
        state.dist_fullbz,
        state.comm,
        memory_estimator.ChunkBudgets(exact_block=2),
    )
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
    state = loop_state
    base = _proposal(state, state.sigma_dmft, config.sys.mu)
    mu0 = _mu_and_occupations(state, base, config.sys.mu)
    products, calls = {}, {}
    solve = MagicMock(side_effect=nonlocal_sde.create_auxiliary_chi_r_q_sum)
    monkeypatch.setattr(nonlocal_sde, "create_auxiliary_chi_r_q_sum", solve)
    for budget in (1, memory_estimator.MAX_CHUNK_BUDGET_BYTES):
        jac = ExactJacobian(
            base.copy(),
            mu0,
            state.u_loc,
            state.v_irr,
            state.v_full,
            state.sigma_dmft_full,
            state.dist_irrk,
            state.dist_fullbz,
            state.comm,
            memory_estimator.ChunkBudgets(exact=budget, exact_block=2),
        )
        ys = np.column_stack([jac.project(y) for y in np.random.default_rng(5).standard_normal((2, jac.n_real))])
        solve.reset_mock()
        products[budget], calls[budget] = jac.matvec_block(ys), solve.call_count
        jac.free()
    assert calls == {1: 2 * state.dist_irrk.my_size, memory_estimator.MAX_CHUNK_BUDGET_BYTES: 2}
    whole = products[memory_estimator.MAX_CHUNK_BUDGET_BYTES]
    assert np.allclose(products[1], whole, atol=1e-12 * np.abs(whole).max())


def test_exact_jacobian_loads_its_local_vertices_per_phase_with_the_same_products(loop_state, monkeypatch):
    """Vertices loaded for their phase give the held vertices' products bit for bit, one load per vertex and piece."""
    state = loop_state
    base = _proposal(state, state.sigma_dmft, config.sys.mu)
    mu0 = _mu_and_occupations(state, base, config.sys.mu)
    real_load, loads, products, held = nonlocal_sde._load_node_shared_local_vertex, [], {}, {}

    def load(node_comm, path, channel, **kwargs):
        loads.append(os.path.basename(path))
        return real_load(node_comm, path, channel, **kwargs)

    monkeypatch.setattr(nonlocal_sde, "_load_node_shared_local_vertex", load)
    for per_phase in (False, True):
        jac = ExactJacobian(
            base.copy(),
            mu0,
            state.u_loc,
            state.v_irr,
            state.v_full,
            state.sigma_dmft_full,
            state.dist_irrk,
            state.dist_fullbz,
            state.comm,
            memory_estimator.ChunkBudgets(exact_block=2, exact_vertices_per_phase=per_phase),
        )
        ys = np.column_stack([jac.project(y) for y in np.random.default_rng(6).standard_normal((3, jac.n_real))])
        loads.clear()
        products[per_phase], held[per_phase] = jac.matvec_block(ys), sorted(jac._vertices)
        products[per_phase] = (products[per_phase], sorted(loads))
        jac.free()
    names = ["f_dc_loc.npy", "gamma_dens_loc.npy", "gamma_magn_loc.npy"]
    assert np.array_equal(products[True][0], products[False][0])
    assert products[False][1] == [] and products[True][1] == sorted(names * 2) and held == {False: names, True: []}


def test_kernel_responses_of_a_rank_without_momenta_are_empty():
    """A rank that owns no irreducible momentum (more ranks than momenta) walks no group and raises nothing."""
    jac = ExactJacobian.__new__(ExactJacobian)
    empty = np.zeros((0, 1, 1, 1, 1, 3, 4), dtype=np.complex64)
    jac._gchi0_inv = FourPoint(empty, SpinChannel.DENS, (1, 1, 1), 1, 1, False, has_compressed_q_dimension=True)
    jac._aux_chunk, jac._beta, jac._channels = 2**20, 10.0, []
    assert jac._kernel_responses([]) == []


def test_leading_eigenpairs_serves_the_products_of_every_rank_and_finds_both_targets(monkeypatch):
    """Under two ranks, rank 0 gets the leading and the stiff eigenpairs and both ranks take part in every product."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
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


def test_exact_jacobian_acts_on_the_sector_the_loop_reaches(p_orbital_state):
    """The products stay in the sector (a symmetry-breaking input gives zero), in the tracker's coordinates."""
    state = p_orbital_state
    base = _proposal(state, state.sigma_dmft, config.sys.mu)
    mu0 = _mu_and_occupations(state, base, config.sys.mu)
    jac = ExactJacobian(
        base,
        mu0,
        state.u_loc,
        state.v_irr,
        state.v_full,
        state.sigma_dmft_full,
        state.dist_irrk,
        state.dist_fullbz,
        state.comm,
    )
    y = np.random.default_rng(5).standard_normal(jac.n_real)
    breaking = y - jac.project(y)
    jy, jb = jac.matvec(y), jac.matvec(breaking)
    round_trip = _sector(_window(y - breaking))
    jac.free()
    assert np.linalg.norm(breaking) > 0.05 * np.linalg.norm(y)
    assert np.allclose(round_trip, y - breaking, atol=1e-12 * np.abs(y).max())
    assert np.allclose(jb, 0.0, atol=1e-12 * np.abs(jy).max())
    assert np.allclose(jac.project(jy), jy, atol=1e-12 * np.abs(jy).max())


def test_leading_eigenpairs_reports_only_modes_of_the_projected_sector(monkeypatch):
    """Eigenpairs outside the projected sector never reach the result, although theta = 0 there tops a negative bulk."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
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


def test_certify_subspace_certifies_the_leading_pair_from_a_nearby_start(monkeypatch):
    """A start with overlap 0.9 to the unstable eigenvector certifies it within the budget, with its exact residual."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
    a = _normal_matrix(40, seed=1)
    comm = conftest.create_comm_mock()
    lam_pi, res, u, n_products = certify_subspace(
        _make_matrix_operator(a, comm), _start_near(_leading_vector(a), 0.9, seed=4), comm
    )
    j = int(np.argmin(lam_pi.real))
    assert np.isclose(lam_pi[j], -1.0, atol=1e-2) and res[j] <= _RITZ_GATE * 2.0 and 0 < n_products <= _CHECK_BASIS
    assert np.isclose(res[j], np.linalg.norm(a @ u[:, j] - (1.0 - lam_pi[j]) * u[:, j]), atol=1e-12)


def test_certify_subspace_returns_an_uncertified_pair_when_the_budget_runs_out(monkeypatch):
    """With a budget of one product a random start comes back after that product, its residual above the gate."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
    monkeypatch.setattr("dgamore.sigma_jacobian._CHECK_BASIS", 1)
    comm = conftest.create_comm_mock()
    operator = _make_matrix_operator(_normal_matrix(40, seed=1), comm)
    lam_pi, res, _, n_products = certify_subspace(operator, np.random.default_rng(7).standard_normal((40, 1)), comm)
    assert n_products == operator.matvec_block.call_count == 1 and lam_pi.size == 1
    assert res[0] > _RITZ_GATE * max(1.0, abs(1.0 - lam_pi[0]))


def test_certify_subspace_applies_its_start_columns_and_each_extension_as_one_block(monkeypatch):
    """The start columns go through one block product, and every extension of the basis through one more."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
    comm = conftest.create_comm_mock()
    operator = _make_matrix_operator(_normal_matrix(40, seed=1), comm)
    *_, n_products = certify_subspace(operator, np.random.default_rng(8).standard_normal((40, 3)), comm)
    widths = [call.args[0].shape[1] for call in operator.matvec_block.call_args_list]
    assert widths[0] == 3 and all(width in (1, 2) for width in widths[1:]) and sum(widths) == n_products
    assert operator.matvec.call_count == 0


def test_certify_subspace_serves_the_products_of_every_rank(monkeypatch):
    """Under two ranks both take part in every product and only rank 0 gets the certified pairs back."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
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


def test_leading_eigenpairs_runs_on_a_sector_of_eight_entries(monkeypatch):
    """ARPACK's basis fits a sector of eight real entries, where k + 1 < ncv <= n leaves little room."""
    monkeypatch.setattr(config, "logger", MagicMock(), raising=False)
    comm = conftest.create_comm_mock()
    lam_pi, res, u = leading_eigenpairs(_make_matrix_operator(_normal_matrix(8, seed=1), comm), comm)
    assert lam_pi.size > 0 and np.all(res > 0)


def test_leading_eigenpairs_keeps_the_converged_pairs_of_a_target_that_gives_up(monkeypatch):
    """A target ARPACK gives up on keeps its converged pairs with a warning, and the serving ranks are released."""
    import scipy.sparse.linalg as spla

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

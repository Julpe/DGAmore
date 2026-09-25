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

import dgamore.n_point_base as npb
from dgamore import config, dga_io, nonlocal_sde
from dgamore.dga_logger import DgaLogger
from dgamore.greens_function import GreensFunction, update_mu
from dgamore.jacobian_stabilization import to_mat, to_vec
from dgamore.mpi_utils import MpiDistributor
from dgamore.sigma_jacobian import ExactJacobian, leading_eigenpairs
from tests import conftest


@pytest.fixture
def loop_state(monkeypatch):
    """The fresh-start loop state of the two-band end_2_end fixture in complex128, on the box of its stored vertices."""
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


def _sector(window: np.ndarray) -> np.ndarray:
    """The sector vector of a core window: irreducible momenta, positive frequencies."""
    return to_vec(window[config.lattice.k_grid.irrk_ind][..., config.box.niv_core :])


def test_exact_jacobian_matches_central_differences_of_the_loop_map(loop_state):
    """J y equals the central difference of the loop's map, with mu re-solved and the occupations updated."""
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
    )
    x0 = _core(base).copy()
    y = np.random.default_rng(3).standard_normal(jac.n_real)
    dx = to_mat(jac.expand(y), x0.shape)
    y *= 1e-2 * np.linalg.norm(x0) / np.linalg.norm(dx)
    dx = to_mat(jac.expand(y), x0.shape)
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
    """A fixed real matrix as a collective operator whose every product broadcasts its vector from rank 0."""

    def product(y):
        y = comm.bcast(y, root=0)
        return a @ y if comm.rank == 0 else None

    return types.SimpleNamespace(n_real=a.shape[0], matvec=MagicMock(side_effect=product), expand=lambda y: y)


def _normal_matrix(n: int, seed: int) -> np.ndarray:
    """A real normal matrix with a leading real value, a complex pair, a stiff negative value and a small bulk."""
    rng = np.random.default_rng(seed)
    blocks = np.zeros((n, n))
    blocks[0, 0], blocks[3, 3] = 2.0, -3.0
    blocks[1:3, 1:3] = [[1.2, 0.4], [-0.4, 1.2]]
    blocks[np.arange(4, n), np.arange(4, n)] = rng.uniform(-0.3, 0.3, n - 4)
    q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    return q @ blocks @ q.T


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

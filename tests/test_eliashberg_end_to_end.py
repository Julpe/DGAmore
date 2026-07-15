# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from mpi4py import MPI as RealMPI

from dgamore import config, eliashberg_solver, dga_io
from dgamore.dga_logger import DgaLogger
from dgamore.greens_function import GreensFunction
from tests import conftest


@pytest.fixture
def setup(monkeypatch):
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/end_2_end"
    comm_mock = conftest.create_comm_mock()

    # patch the MPI reference inside each module that calls MPI.Request.Waitall,
    # keeping the real constants for logic checks
    for module_path in ("dgamore.mpi_utils", "dgamore.eliashberg_solver"):
        mock_mpi = MagicMock(
            COMM_WORLD=comm_mock,
            Request=MagicMock(Waitall=MagicMock(return_value=None)),
            IN_PLACE=RealMPI.IN_PLACE,
            SUM=RealMPI.SUM,
        )
        monkeypatch.setattr(f"{module_path}.MPI", mock_mpi)
    monkeypatch.setattr("mpi4py.MPI.COMM_WORLD", comm_mock)

    config.logger = DgaLogger(comm_mock, "./")
    conftest.create_default_config(config, folder)

    config.eliashberg.perform_eliashberg = False
    config.eliashberg.symmetry = "random"
    config.eliashberg.epsilon = 1e-12
    config.eliashberg.n_eig = 4

    # the comm mock returns itself for chained calls and node logic
    comm_mock.Split.return_value = comm_mock
    comm_mock.allgather.return_value = ["node1"]

    yield folder, comm_mock


@pytest.mark.parametrize("save_fq, save_memory", [(True, True), (False, True), (True, False), (False, False)])
def test_eliashberg_equation_without_local_part(setup, save_fq, save_memory):
    """The Eliashberg solve without the local vertex part reproduces the reference eigenvalues in all four
    save_fq x save_memory flag combinations."""
    folder, comm_mock = setup

    config.box.niw_core = 20
    config.box.niv_core = 20
    config.box.niv_shell = 10

    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = config.output.output_path
    config.eliashberg.include_local_part = False
    config.eliashberg.save_fq = save_fq
    config.memory.save_memory_for_fq = save_memory
    config.memory.save_memory_for_lanczos = save_memory

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)

    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"), nk=config.lattice.nk)

    lambdas_sing, lambdas_trip, gaps_sing, gaps_trip = eliashberg_solver.solve(
        g_dga, g_dmft, u_loc, v_nonloc, comm_mock
    )
    assert np.allclose(lambdas_sing, np.array([16.00752, 15.802559, 14.981937, 14.684908]), atol=1e-2)
    assert np.allclose(lambdas_trip, np.array([6.7059956, 6.705986, 6.456438, 6.4564347]), atol=1e-2)


@pytest.mark.parametrize("save_fq, save_memory", [(True, True), (False, True), (True, False), (False, False)])
def test_eliashberg_equation_with_local_part(setup, save_fq, save_memory):
    """The Eliashberg solve including the local vertex part reproduces the reference eigenvalues in all four
    save_fq x save_memory flag combinations."""
    folder, comm_mock = setup

    config.box.niw_core = 20
    config.box.niv_core = 20
    config.box.niv_shell = 10

    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = config.output.output_path
    config.eliashberg.include_local_part = True
    config.eliashberg.save_fq = save_fq
    config.memory.save_memory_for_fq = save_memory
    config.memory.save_memory_for_lanczos = save_memory

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)

    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"), nk=config.lattice.nk)

    lambdas_sing, lambdas_trip, gaps_sing, gaps_trip = eliashberg_solver.solve(
        g_dga, g_dmft, u_loc, v_nonloc, comm_mock
    )
    assert np.allclose(lambdas_sing, np.array([15.802544, 15.555848, 14.684908, 14.280717]), atol=1e-2)
    assert np.allclose(lambdas_trip, np.array([6.705995, 6.7059927, 6.45644, 6.4564347]), atol=1e-2)


def _fft_index_map(nq: tuple, f) -> np.ndarray:
    """Builds the [k, k'] index map on the flattened FFT grid from a per-axis index function f(i, j, n)."""
    m = np.empty((int(np.prod(nq)), int(np.prod(nq))), dtype=int)
    for a in range(nq[0]):
        for b in range(nq[1]):
            for c in range(nq[0]):
                for d in range(nq[1]):
                    m[a * nq[1] + b, c * nq[1] + d] = (f(a, c, nq[0])) * nq[1] + (f(b, d, nq[1]))
    return m


def test_kernel_matches_thesis_eliashberg_form_on_two_band_vertex(setup, monkeypatch):
    """Locks the full multi-orbital pairing kernel on the real two-band vertex against thesis Eq. (4.63): the
    densified production matvec must equal the transparent index formula norm * sum_{ef} [sign
    Gamma^{K-Q;vv'}_{e1f2} + Gamma^{(-K)-Q;v(-v')}_{e2f1}] G_{eh}(Q, v') conj(G_{gf}(Q, v')) Delta_{gh}(Q, v')
    (pinning every layout, permute, FFT and the bubble-gap contraction for orbitally off-diagonal G), and its
    leading eigenvalue spectrum must equal that of the independently densified thesis kernel
    K^{vv'}_{1b2a}(K - Q) chi^{Q v'}_{0;acbd} Delta_{cd} with chi_{0;acbd} = G_{ad}(Q) G_{cb}(-Q) and the vertex
    legs read as Gamma-thesis_{1234} = Gamma-stored_{2341}."""
    folder, comm_mock = setup

    config.box.niw_core = 20
    config.box.niv_core = 20
    config.box.niv_shell = 10

    g_dmft, _, _, _ = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = folder
    config.eliashberg.include_local_part = True
    config.eliashberg.save_fq = False
    config.memory.save_memory_for_fq = False
    config.memory.save_memory_for_lanczos = False

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)
    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"), nk=config.lattice.nk)

    captured = {}
    real_solver = eliashberg_solver.solve_eliashberg_lanczos

    def capture_solver(gamma_r_pp, gchi0_q0_pp, ranks):
        dense_holder = []

        def fake_eigsh(op, k, tol, v0, which, maxiter):
            n = op.shape[0]
            dense_holder.append(np.column_stack([op.matvec(np.eye(n, dtype=np.complex64)[:, i]) for i in range(n)]))
            lam, vec = np.linalg.eig(dense_holder[0])
            order = np.argsort(lam.real)[::-1][:k]
            return lam.real[order], vec[:, order]

        channel = gamma_r_pp.channel.value
        # the solver consumes the passed vertex (in-place BZ map, fft, sign fold, free), so the momentum-space
        # full-BZ vertex is captured from a copy before the solve
        gamma_full = (
            gamma_r_pp.copy().map_to_full_bz(config.lattice.k_grid, config.lattice.k_grid.nk).decompress_q_dimension()
        )
        captured[channel] = {"gamma": gamma_full.mat.copy()}
        gamma_full.free()
        with monkeypatch.context() as m:
            m.setattr("dgamore.eliashberg_solver.sp.sparse.linalg.eigsh", fake_eigsh)
            out = real_solver(gamma_r_pp, gchi0_q0_pp, ranks)
        captured[channel]["dense"] = dense_holder[0]
        return out

    monkeypatch.setattr(eliashberg_solver, "solve_eliashberg_lanczos", capture_solver)
    eliashberg_solver.solve(g_dga, g_dmft, u_loc, v_nonloc, comm_mock)

    nq = config.lattice.k_grid.nk
    nq_tot, o, beta = int(np.prod(nq)), config.sys.n_bands, config.sys.beta
    niv_pp = min(config.box.niw_core // 2, config.box.niv_core // 2)
    n2 = 2 * niv_pp
    giwk = g_dga.cut_niv(niv_pp).compress_q_dimension().mat.astype(np.complex128)
    kdiff = _fft_index_map(nq, lambda a, c, n: (a - c) % n)
    kncross = _fft_index_map(nq, lambda a, c, n: (-a - c) % n)
    norm = 0.5 / nq_tot / beta

    for ch, sign in (("sing", 1.0), ("trip", -1.0)):
        gam = captured[ch]["gamma"].astype(np.complex128).reshape(nq_tot, o, o, o, o, n2, n2)
        dense = captured[ch]["dense"].astype(np.complex128)

        gcm = np.transpose(gam, (0, 1, 4, 3, 2, 5, 6))[kncross][..., ::-1]
        m_rec = norm * np.einsum(
            "KQexfyvp,Qehp,Qgfp->KxyvQghp", sign * gam[kdiff] + gcm, giwk, np.conj(giwk), optimize=True
        )
        assert np.allclose(dense, m_rec.reshape(dense.shape), atol=1e-5 * np.abs(dense).max())

        gam_th = np.transpose(gam, (0, 2, 3, 4, 1, 5, 6))
        direct = np.einsum("KQxbyavp,Qadp,Qcbp->KxyvQcdp", gam_th[kdiff], giwk, np.conj(giwk), optimize=True)
        crossed = np.einsum(
            "KQxaybvp,Qadp,Qcbp->KxyvQcdp", gam_th[kncross][..., ::-1], giwk, np.conj(giwk), optimize=True
        )
        m_th = (norm * sign * (direct + sign * crossed)).reshape(dense.shape)
        ev_code = np.sort(np.linalg.eigvals(dense).real)[::-1][:10]
        ev_thesis = np.sort(np.linalg.eigvals(m_th).real)[::-1][:10]
        assert np.allclose(ev_code, ev_thesis, atol=1e-3)

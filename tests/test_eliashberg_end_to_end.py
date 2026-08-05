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
from dgamore.eliashberg_solver import gap_parity_diagnostics
from dgamore.greens_function import GreensFunction
from dgamore.n_point_base import SpinChannel
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


def _assert_gap_sector_parity(results: dict) -> None:
    """Asserts every returned Eliashberg gap carries its sector's frequency (T) and momentum-orbital (P.O) parity."""
    channel_sign = {SpinChannel.SING: 1, SpinChannel.TRIP: -1}
    parity_sign = {"even": 1, "odd": -1}
    for (channel, parity), (_lambdas, gaps) in results.items():
        for gap in gaps:
            diagnostics = gap_parity_diagnostics(gap.mat.flatten(), gap.mat.shape)
            assert np.isclose(diagnostics["T"].real, parity_sign[parity], atol=1e-3)
            assert np.isclose(diagnostics["PO"].real, channel_sign[channel] * parity_sign[parity], atol=1e-3)


@pytest.mark.parametrize("save_fq", [True, False])
def test_eliashberg_equation_with_local_part(setup, save_fq):
    """The Eliashberg solve matches reference eigenvalues and gap parity with and without save_fq."""
    folder, comm_mock = setup

    config.box.niw_core = 20
    config.box.niv_core = 20
    config.box.niv_shell = 10

    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = config.output.output_path
    config.eliashberg.resolve_frequency_parity = True
    config.eliashberg.save_fq = save_fq

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)

    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"), nk=config.lattice.nk)

    results = eliashberg_solver.solve(g_dga, g_dmft, u_loc, v_nonloc, comm_mock)
    assert set(results) == {
        (SpinChannel.SING, "even"),
        (SpinChannel.SING, "odd"),
        (SpinChannel.TRIP, "even"),
        (SpinChannel.TRIP, "odd"),
    }
    for (_channel, _parity), (lambdas, gaps) in results.items():
        assert len(gaps) == config.eliashberg.n_eig and len(lambdas) == config.eliashberg.n_eig
    # sing/even reproduces the unprojected leading singlet and trip/odd the unprojected leading triplet
    assert np.allclose(results[(SpinChannel.SING, "even")][0], [15.80255, 15.55585, 14.68491, 14.28071], atol=1e-2)
    assert np.allclose(results[(SpinChannel.SING, "odd")][0], [4.37008, 4.37007, 3.33718, 3.33718], atol=1e-2)
    assert np.allclose(results[(SpinChannel.TRIP, "even")][0], [2.84907, 2.84906, 2.40087, 2.40087], atol=1e-2)
    assert np.allclose(results[(SpinChannel.TRIP, "odd")][0], [7.33033, 7.26272, 6.64910, 6.20756], atol=1e-2)
    _assert_gap_sector_parity(results)


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
    """The densified two-band pairing kernel matches the index formula and the thesis Eq. (4.63) spectrum."""
    folder, comm_mock = setup

    config.box.niw_core = 20
    config.box.niv_core = 20
    config.box.niv_shell = 10

    g_dmft, _, _, _ = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = folder
    config.eliashberg.save_fq = False
    config.eliashberg.resolve_frequency_parity = False

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)
    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"), nk=config.lattice.nk)

    captured = {}
    real_solver = eliashberg_solver.solve_eliashberg_lanczos

    def capture_solver(gamma_r_pp, gchi0_q0_pp, ranks, parities=None):
        dense_holder = []

        def fake_eigsh(op, k, tol, v0, which, maxiter):
            n = op.shape[0]
            dense_holder.append(np.column_stack([op.matvec(np.eye(n, dtype=np.complex64)[:, i]) for i in range(n)]))
            lam, vec = np.linalg.eig(dense_holder[0])
            order = np.argsort(lam.real)[::-1][:k]
            return lam.real[order], vec[:, order]

        channel = gamma_r_pp.channel.value
        # the solver consumes the passed vertex (in-place BZ map, fft, free), so the momentum-space
        # full-BZ vertex is captured from a copy before the solve
        gamma_full = (
            gamma_r_pp.copy().map_to_full_bz(config.lattice.k_grid, config.lattice.k_grid.nk).decompress_q_dimension()
        )
        captured[channel] = {"gamma": gamma_full.mat.copy()}
        gamma_full.free()
        with monkeypatch.context() as m:
            m.setattr("dgamore.eliashberg_solver.sp.sparse.linalg.eigsh", fake_eigsh)
            out = real_solver(gamma_r_pp, gchi0_q0_pp, ranks, parities)
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
            "KQexfyvp,Qehp,Qgfp->KxyvQghp", gam[kdiff] + sign * gcm, giwk, np.conj(giwk), optimize=True
        )
        assert np.allclose(dense, m_rec.reshape(dense.shape), atol=1e-5 * np.abs(dense).max())

        gam_th = np.transpose(gam, (0, 2, 3, 4, 1, 5, 6))
        direct = np.einsum("KQxbyavp,Qadp,Qcbp->KxyvQcdp", gam_th[kdiff], giwk, np.conj(giwk), optimize=True)
        crossed = np.einsum(
            "KQxaybvp,Qadp,Qcbp->KxyvQcdp", gam_th[kncross][..., ::-1], giwk, np.conj(giwk), optimize=True
        )
        m_th = (norm * (direct + sign * crossed)).reshape(dense.shape)
        ev_code = np.sort(np.linalg.eigvals(dense).real)[::-1][:10]
        ev_thesis = np.sort(np.linalg.eigvals(m_th).real)[::-1][:10]
        assert np.allclose(ev_code, ev_thesis, atol=1e-3)


def _phase_align(gap: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Removes the global-phase ambiguity of an eigenvector by rotating it onto the reference's phase."""
    overlap = np.vdot(reference, gap)
    return gap if overlap == 0 else gap * (np.abs(overlap) / overlap)


def test_eliashberg_gap_functions_carry_sector_parity_and_match_reference(setup):
    """Each gap carries its sector's T and P.O parity and, with a fixed seed, matches the reference up to a phase."""
    folder, comm_mock = setup

    config.box.niw_core = 20
    config.box.niv_core = 20
    config.box.niv_shell = 10

    g_dmft, _, _, _ = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = folder
    config.eliashberg.n_eig = 4
    config.eliashberg.resolve_frequency_parity = True

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)
    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"), nk=config.lattice.nk)

    np.random.seed(0)
    results = eliashberg_solver.solve(g_dga, g_dmft, u_loc, v_nonloc, comm_mock)

    channel_sign = {SpinChannel.SING: 1, SpinChannel.TRIP: -1}
    parity_sign = {"even": 1, "odd": -1}
    for (channel, parity), (_lambdas, gaps) in results.items():
        for i, gap in enumerate(gaps):
            diagnostics = gap_parity_diagnostics(gap.mat.flatten(), gap.mat.shape)
            assert np.isclose(diagnostics["T"].real, parity_sign[parity], atol=1e-3)
            assert np.isclose(diagnostics["PO"].real, channel_sign[channel] * parity_sign[parity], atol=1e-3)
            reference = np.load(f"{folder}/gap_{channel.value}_{parity}_{i + 1}.npy")
            assert np.allclose(_phase_align(gap.mat, reference), reference, atol=1e-4)


def test_eliashberg_grid_solver_matches_reference(setup, monkeypatch):
    """The forced block-distributed solver reproduces the reference eigenvalues of the in-memory solve."""
    folder, comm_mock = setup

    config.box.niw_core = 20
    config.box.niv_core = 20
    config.box.niv_shell = 10

    g_dmft, _, _, _ = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = config.output.output_path
    config.eliashberg.resolve_frequency_parity = True
    monkeypatch.setattr(eliashberg_solver, "FORCE_GRID_SOLVER", True)

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)
    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"), nk=config.lattice.nk)

    results = eliashberg_solver.solve(g_dga, g_dmft, u_loc, v_nonloc, comm_mock)

    assert np.allclose(results[(SpinChannel.SING, "even")][0], [15.80255, 15.55585, 14.68491, 14.28071], atol=1e-2)
    assert np.allclose(results[(SpinChannel.SING, "odd")][0], [4.37008, 4.37007, 3.33718, 3.33718], atol=1e-2)
    assert np.allclose(results[(SpinChannel.TRIP, "even")][0], [2.84907, 2.84906, 2.40087, 2.40087], atol=1e-2)
    assert np.allclose(results[(SpinChannel.TRIP, "odd")][0], [7.33033, 7.26272, 6.64910, 6.20756], atol=1e-2)

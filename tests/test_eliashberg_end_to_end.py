# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
from mpi4py import MPI as RealMPI

from dgamore import config, eliashberg_solver, dga_io
from dgamore.dga_logger import DgaLogger
from dgamore.greens_function import GreensFunction
from tests import conftest


@pytest.fixture
def setup():
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/end_2_end"
    comm_mock = conftest.create_comm_mock()

    # 1. Create a mutable Python-native replacement for the immutable C-extension class
    class MockRequest:
        @staticmethod
        def Waitall(reqs):
            return None

    # 2. Identify all modules that call MPI.Request.Waitall
    # We must patch the 'MPI' name inside these specific modules to bypass the C-extension
    modules_to_patch = ["dgamore.mpi_utils", "dgamore.eliashberg_solver"]

    with ExitStack() as stack:
        # Patch the MPI module reference in each relevant file
        for module_path in modules_to_patch:
            mock_mpi = stack.enter_context(patch(f"{module_path}.MPI"))

            # Re-bind the necessary parts to our mocks
            mock_mpi.COMM_WORLD = comm_mock
            mock_mpi.Request = MockRequest
            mock_mpi.IN_PLACE = RealMPI.IN_PLACE  # Keep real constants for logic checks
            mock_mpi.SUM = RealMPI.SUM

        # 3. Apply the global COMM_WORLD patch for general use
        stack.enter_context(patch("mpi4py.MPI.COMM_WORLD", comm_mock))

        # 4. Standard DGAmore Configuration
        config.logger = DgaLogger(comm_mock, "./")
        conftest.create_default_config(config, folder)

        config.eliashberg.perform_eliashberg = False
        config.eliashberg.symmetry = "random"
        config.eliashberg.epsilon = 1e-12
        config.eliashberg.n_eig = 4

        # Ensure mocks return themselves for chained calls or node logic
        comm_mock.Split.return_value = comm_mock
        comm_mock.allgather.return_value = ["node1"]

        yield folder, comm_mock


@pytest.mark.parametrize(
    "niw_core, niv_core, niv_shell, save_fq, save_memory",
    [(20, 20, 10, True, True), (20, 20, 10, False, True), (20, 20, 10, True, False), (20, 20, 10, False, False)],
)
def test_eliashberg_equation_without_local_part(setup, niw_core, niv_core, niv_shell, save_fq, save_memory):
    folder, comm_mock = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell

    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = config.output.output_path
    config.eliashberg.include_local_part = False
    config.eliashberg.save_fq = save_fq
    config.eliashberg.construct_fq_cheap = False
    config.memory.save_memory_for_sde = save_memory
    config.memory.save_memory_for_fq = save_memory
    config.memory.save_memory_for_lanczos = save_memory

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"))

    lambdas_sing, lambdas_trip, gaps_sing, gaps_trip = eliashberg_solver.solve(
        g_dga, g_dmft, u_loc, v_nonloc, comm_mock
    )
    assert np.allclose(lambdas_sing, np.array([16.00998764, 15.8037398, 14.97882938, 14.68343997]), atol=1e-4)
    assert np.allclose(lambdas_trip, np.array([6.70800075, 6.70799438, 6.45388298, 6.45387878]), atol=1e-4)


@pytest.mark.parametrize(
    "niw_core, niv_core, niv_shell, save_fq, save_memory",
    [(20, 20, 10, True, True), (20, 20, 10, False, True), (20, 20, 10, True, False), (20, 20, 10, False, False)],
)
def test_eliashberg_equation_with_local_part(setup, niw_core, niv_core, niv_shell, save_fq, save_memory):
    folder, comm_mock = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell

    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = config.output.output_path
    config.eliashberg.include_local_part = True
    config.eliashberg.save_fq = save_fq
    config.eliashberg.construct_fq_cheap = False
    config.memory.save_memory_for_sde = save_memory
    config.memory.save_memory_for_fq = save_memory
    config.memory.save_memory_for_lanczos = save_memory

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"))

    lambdas_sing, lambdas_trip, gaps_sing, gaps_trip = eliashberg_solver.solve(
        g_dga, g_dmft, u_loc, v_nonloc, comm_mock
    )
    assert np.allclose(lambdas_sing, np.array([15.80373154, 14.68344217, 12.59137005, 10.81553018]), atol=1e-4)
    assert np.allclose(lambdas_trip, np.array([6.70800083, 6.70799431, 6.45388305, 6.45387905]), atol=1e-4)

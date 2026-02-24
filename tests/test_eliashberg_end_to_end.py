import os
from unittest.mock import patch

import numpy as np
import pytest

from moldga import config, eliashberg_solver, dga_io
from moldga.dga_logger import DgaLogger
from moldga.greens_function import GreensFunction
from moldga.local_four_point import LocalFourPoint
from moldga.n_point_base import SpinChannel
from tests import conftest


@pytest.fixture
def setup():
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/end_2_end"

    comm_mock = conftest.create_comm_mock()

    with patch("mpi4py.MPI.COMM_WORLD", comm_mock):
        config.logger = DgaLogger(comm_mock, "./")
        conftest.create_default_config(config, folder)
        config.eliashberg.perform_eliashberg = False
        config.eliashberg.symmetry = "random"
        config.eliashberg.epsilon = 1e-12
        config.eliashberg.n_eig = 4
        comm_mock.Split.return_value = comm_mock

        yield folder, comm_mock


@pytest.mark.parametrize("niw_core, niv_core, niv_shell, save_fq", [(20, 20, 10, True), (20, 20, 10, False)])
def test_eliashberg_equation_without_local_part(setup, niw_core, niv_core, niv_shell, save_fq):
    folder, comm_mock = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell

    g_dmft, s_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = config.output.output_path
    config.eliashberg.include_local_part = False
    config.eliashberg.save_fq = save_fq

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"))

    gamma_dens = LocalFourPoint.load(f"{folder}/gamma_dens_loc.npy", channel=SpinChannel.DENS)
    gamma_magn = LocalFourPoint.load(f"{folder}/gamma_magn_loc.npy", channel=SpinChannel.MAGN)

    lambdas_sing, lambdas_trip, gaps_sing, gaps_trip = eliashberg_solver.solve(
        g_dga, g_dmft, u_loc, v_nonloc, gamma_dens, gamma_magn, comm_mock
    )
    assert np.allclose(lambdas_sing, np.array([3.85828144, 3.70361068, 3.65005429, 3.5992988]), atol=1e-4)
    assert np.allclose(lambdas_trip, np.array([3.34166718, 2.9909934, 2.72114652, 2.72114537]), atol=1e-4)


@pytest.mark.parametrize("niw_core, niv_core, niv_shell, save_fq", [(20, 20, 10, True), (20, 20, 10, False)])
def test_eliashberg_equation_with_local_part(setup, niw_core, niv_core, niv_shell, save_fq):
    folder, comm_mock = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell

    g_dmft, s_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = config.output.output_path
    config.eliashberg.include_local_part = True
    config.eliashberg.save_fq = save_fq

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"))

    gamma_dens = LocalFourPoint.load(f"{folder}/gamma_dens_loc.npy", channel=SpinChannel.DENS)
    gamma_magn = LocalFourPoint.load(f"{folder}/gamma_magn_loc.npy", channel=SpinChannel.MAGN)

    lambdas_sing, lambdas_trip, gaps_sing, gaps_trip = eliashberg_solver.solve(
        g_dga, g_dmft, u_loc, v_nonloc, gamma_dens, gamma_magn, comm_mock
    )
    assert np.allclose(lambdas_sing, np.array([3.7036108, 3.5992989, 3.32485204, 3.32485072]), atol=1e-4)
    assert np.allclose(lambdas_trip, np.array([2.72114656, 2.72114542, 2.69452022, 2.69451905]), atol=1e-4)

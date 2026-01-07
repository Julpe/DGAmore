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


@pytest.mark.parametrize("niw_core, niv_core, niv_shell", [(20, 20, 10), (-1, 20, 10), (20, -1, 10), (-1, -1, 10)])
def test_eliashberg_equation_without_local_part(setup, niw_core, niv_core, niv_shell):
    folder, comm_mock = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell

    g_dmft, s_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = config.output.output_path
    config.eliashberg.include_local_part = False

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"))

    gamma_dens = LocalFourPoint.load(f"{folder}/gamma_dens_loc.npy", channel=SpinChannel.DENS)
    gamma_magn = LocalFourPoint.load(f"{folder}/gamma_magn_loc.npy", channel=SpinChannel.MAGN)

    lambdas_sing, lambdas_trip, gaps_sing, gaps_trip = eliashberg_solver.solve(
        g_dga, g_dmft, u_loc, v_nonloc, gamma_dens, gamma_magn, comm_mock
    )
    assert np.allclose(lambdas_sing, np.array([4.17936064, 3.98239343, 3.7661796, 3.6118379]), atol=1e-4)
    assert np.allclose(lambdas_trip, np.array([3.87211113, 3.27590262, 2.91889808, 2.85647941]), atol=1e-4)


@pytest.mark.parametrize("niw_core, niv_core, niv_shell", [(20, 20, 10), (-1, 20, 10), (20, -1, 10), (-1, -1, 10)])
def test_eliashberg_equation_with_local_part(setup, niw_core, niv_core, niv_shell):
    folder, comm_mock = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell

    g_dmft, s_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = config.output.output_path
    config.eliashberg.include_local_part = True
    config.eliashberg.save_fq = True

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"))

    gamma_dens = LocalFourPoint.load(f"{folder}/gamma_dens_loc.npy", channel=SpinChannel.DENS)
    gamma_magn = LocalFourPoint.load(f"{folder}/gamma_magn_loc.npy", channel=SpinChannel.MAGN)

    lambdas_sing, lambdas_trip, gaps_sing, gaps_trip = eliashberg_solver.solve(
        g_dga, g_dmft, u_loc, v_nonloc, gamma_dens, gamma_magn, comm_mock
    )
    assert np.allclose(lambdas_sing, np.array([6.19956223, 5.01912816, 3.9427516, 3.53948015]), atol=1e-4)
    assert np.allclose(lambdas_trip, np.array([5.4919969, 4.65633698, 2.87634353, 2.79390464]), atol=1e-4)

import os
from unittest.mock import patch

import numpy as np
import pytest

from moldga import config, dga_io, local_sde
from moldga import nonlocal_sde
from moldga.dga_logger import DgaLogger
from moldga.greens_function import GreensFunction
from tests import conftest


@pytest.fixture
def setup():
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/end_2_end"

    comm_mock = conftest.create_comm_mock()

    with patch("mpi4py.MPI.COMM_WORLD", comm_mock):
        config.logger = DgaLogger(comm_mock, "./")
        conftest.create_default_config(config, folder)
        yield folder, comm_mock


@pytest.mark.parametrize("niw_core, niv_core, niv_shell", [(20, 20, 10)])
def test_calculates_nonlocal_sde_correctly(setup, niw_core, niv_core, niv_shell):
    folder, comm_mock = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell

    g_dmft, s_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()

    config.output.output_path = folder

    ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
    g_loc = GreensFunction.create_g_loc(s_dmft.create_with_asympt_up_to_core(), ek)

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    (gamma_d, gamma_m, chi_d, chi_m, vrg_d, vrg_m, f_d, f_m, gchi_d, gchi_m, s_loc) = (
        local_sde.perform_local_schwinger_dyson(g_loc, g2_dens, g2_magn, u_loc)
    )

    sigma_dga_mat = (
        nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft, s_loc, gamma_d, gamma_m)
        .decompress_q_dimension()
        .mat
    )
    sigma_dga_ref = np.load(f"{folder}/sigma_dga.npy")

    assert np.allclose(sigma_dga_mat, sigma_dga_ref, atol=3e-5)

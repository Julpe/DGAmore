import os
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse.linalg

from moldga import config, eliashberg_solver, dga_io
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

    lambdas_sing, lambdas_trip, gaps_sing, gaps_trip, *_ = eliashberg_solver.solve(
        g_dga, g_dmft, u_loc, v_nonloc, comm_mock
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

    lambdas_sing, lambdas_trip, gaps_sing, gaps_trip, *_ = eliashberg_solver.solve(
        g_dga, g_dmft, u_loc, v_nonloc, comm_mock
    )
    assert np.allclose(lambdas_sing, np.array([3.7036108, 3.5992989, 3.32485204, 3.32485072]), atol=1e-4)
    assert np.allclose(lambdas_trip, np.array([2.72114656, 2.72114542, 2.69452022, 2.69451905]), atol=1e-4)


@pytest.mark.parametrize(
    "niw_core, niv_core, niv_shell, include_local_part, use_shift_invert_mode",
    [(20, 20, 10, True, True), (20, 20, 10, True, False), (20, 20, 10, False, True), (20, 20, 10, False, False)],
)
def test_eliashberg_equation_with_shift_invert_mode(
    setup, niw_core, niv_core, niv_shell, include_local_part, use_shift_invert_mode
):
    folder, comm_mock = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell

    g_dmft, s_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()

    config.eliashberg.perform_eliashberg = True
    config.output.output_path = folder
    config.output.eliashberg_path = config.output.output_path
    config.eliashberg.include_local_part = include_local_part
    config.eliashberg.use_shift_invert_mode = use_shift_invert_mode

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    g_dga = GreensFunction(np.load(f"{folder}/giwk_dga.npy"))

    _real_eigsh = scipy.sparse.linalg.eigsh

    def mock_eigsh(mat, k=1, tol=None, v0=None, sigma=None, which="LM", maxiter=None):
        if sigma == 1.0 and which.upper() == "LM":
            return np.array([1.0, 0.8, 0.9, 0.7]), np.random.rand(4, 4, 1, 2, 2, 20, 4)
        else:
            try:
                return _real_eigsh(mat, k=k, tol=tol, v0=v0, sigma=sigma, which=which, maxiter=maxiter)
            except Exception:
                raise RuntimeError()

    with patch.object(scipy.sparse.linalg, "eigsh", side_effect=mock_eigsh) as mock:
        (l_sing, l_trip, gaps_sing, gaps_trip, l_sing_si, l_trip_si, gaps_sing_si, gaps_trip_si) = (
            eliashberg_solver.solve(g_dga, g_dmft, u_loc, v_nonloc, comm_mock)
        )

        if use_shift_invert_mode:
            assert mock.call_count == 4
            assert any(
                call.kwargs.get("sigma") == 1.0 and call.kwargs.get("which") == "LM" for call in mock.call_args_list
            )
        else:
            assert mock.call_count == 2

    if include_local_part:
        assert np.allclose(l_sing, np.array([3.7036108, 3.5992989, 3.32485204, 3.32485072]), atol=1e-4)
        assert np.allclose(l_trip, np.array([2.72114656, 2.72114542, 2.69452022, 2.69451905]), atol=1e-4)
    else:
        assert np.allclose(l_sing, np.array([3.85828144, 3.70361068, 3.65005429, 3.5992988]), atol=1e-4)
        assert np.allclose(l_trip, np.array([3.34166718, 2.9909934, 2.72114652, 2.72114537]), atol=1e-4)

    if use_shift_invert_mode:
        assert not np.allclose(l_sing_si, l_sing, atol=1e-4)
        assert not np.allclose(l_trip_si, l_trip, atol=1e-4)

    if not use_shift_invert_mode:
        assert l_sing_si is None
        assert l_trip_si is None
        assert gaps_sing_si is None
        assert gaps_trip_si is None

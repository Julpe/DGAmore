import contextlib
import os
import types
from unittest import mock
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from moldga import config, dga_io, local_sde
from moldga import nonlocal_sde
from moldga.dga_logger import DgaLogger
from moldga.greens_function import GreensFunction
import moldga.brillouin_zone as bz
from tests import conftest


@pytest.fixture
def setup():
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/end_2_end"

    comm_mock = conftest.create_comm_mock()

    with patch("mpi4py.MPI.COMM_WORLD", comm_mock):
        config.logger = DgaLogger(comm_mock, "./")
        conftest.create_default_config(config, folder)
        yield folder, comm_mock


@pytest.fixture
def setup_srvo3_cubic():
    def create_srvo3_cubic_config(c, f: str):
        c.box.niw_core = -1
        c.box.niv_core = -1
        c.box.niv_shell = 0
        c.output.save_quantities = False
        c.output.do_plotting = False
        c.lattice.nk = (8, 8, 8)
        c.lattice.nq = config.lattice.nk
        c.lattice.symmetries = "three_dimensional_cubic"
        c.lattice.k_grid = bz.KGrid(c.lattice.nk, bz.three_dimensional_cubic_symmetries())
        c.lattice.q_grid = config.lattice.k_grid
        c.lattice.k_grid.specify_orbital_basis(3, "t2g")
        c.lattice.q_grid.specify_orbital_basis(3, "t2g")
        c.lattice.type = "from_wannier90"
        c.lattice.interaction_type = "kanamori_from_dmft"
        c.lattice.er_input = f"{f}/wan_hr.dat"
        c.dmft.input_path = f
        c.dmft.do_sym_v_vp = True
        c.dmft.symmetrize_orbitals = [1, 2, 3]
        c.eliashberg.perform_eliashberg = False
        c.self_consistency.mixing = 1
        c.self_consistency.max_iter = 1

    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/srvo3_end2end"

    comm_mock = conftest.create_comm_mock()

    with patch("mpi4py.MPI.COMM_WORLD", comm_mock):
        config.logger = DgaLogger(comm_mock, "./")
        create_srvo3_cubic_config(config, folder)
        yield folder, comm_mock


def make_cupy_mock():
    cp = types.ModuleType("cupy")

    cp.asarray = np.asarray
    cp.zeros = np.zeros
    cp.empty = np.empty
    cp.empty_like = np.empty_like

    cp.asnumpy = lambda x: x

    cp.arange = np.arange
    cp.take = np.take

    def einsum(*args, **kwargs):
        return np.einsum(*args, **kwargs)

    cp.einsum = mock.Mock(side_effect=einsum)

    cp.cuda = types.ModuleType("cupy.cuda")
    cp.cuda.is_available = mock.Mock(return_value=True)

    cp.cuda.runtime = types.ModuleType("cupy.cuda.runtime")
    cp.cuda.runtime.getDeviceCount = mock.Mock(return_value=1)

    class DummyDevice:
        def __init__(self, device_id):
            self.device_id = device_id

        def use(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    cp.cuda.Device = DummyDevice

    return cp


@contextlib.contextmanager
def gpu_cpu_context(use_gpu: bool):
    mock_gpu = False
    if use_gpu:
        try:  # real GPU is available
            import cupy as cp

            yield mock_gpu
        except ImportError:  # fallback to mocked GPU
            mock_gpu = True
            mock_cupy = make_cupy_mock()
            with mock.patch.dict(
                "sys.modules",
                {
                    "cupy": mock_cupy,
                    "cupy.cuda": mock_cupy.cuda,
                    "cupy.cuda.runtime": mock_cupy.cuda.runtime,
                },
            ):
                yield mock_gpu
                assert mock_cupy.einsum.called, "GPU path not taken (cp.einsum not called)"
    else:  # force CPU path
        with mock.patch.dict("sys.modules", {"cupy": None}):
            yield mock_gpu


@pytest.mark.parametrize("niw_core, niv_core, niv_shell, use_gpu", [(20, 20, 10, True), (20, 20, 10, False)])
def test_calculates_nonlocal_sde_correctly(setup, niw_core, niv_core, niv_shell, use_gpu):
    folder, comm_mock = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell
    config.dmft.symmetrize_orbitals = []

    g_dmft, s_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()

    config.output.output_path = folder

    ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
    g_loc = GreensFunction.create_g_loc(s_dmft.create_with_asympt_up_to_core(), ek)

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    (*_, s_loc) = local_sde.perform_local_schwinger_dyson(g_loc, g2_dens, g2_magn, u_loc)

    with gpu_cpu_context(use_gpu) as mock_gpu:
        sigma_dga = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft, s_loc)

    sigma_dga_mat = sigma_dga.decompress_q_dimension().cut_niv(50).mat
    sigma_dga_ref = np.load(f"{folder}/sigma_dga.npy")

    assert np.allclose(sigma_dga_mat, sigma_dga_ref, atol=3e-5 if not mock_gpu else 1e-3)

    sigma_interpolated_mat = sigma_dga.interpolate(12.5, 25, 30).cut_niv(50).mat
    sigma_interpolated_ref = np.load(f"{folder}/sigma_dga_interpolated.npy")

    assert np.allclose(sigma_interpolated_mat, sigma_interpolated_ref, atol=3e-5 if not mock_gpu else 1e-3)


def test_calculates_srvo3_correctly(setup_srvo3_cubic):
    folder_cubic, comm_mock = setup_srvo3_cubic

    g_dmft, s_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()

    config.output.output_path = folder_cubic

    ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
    perm = [1, 0, 2]  # swap orbital 0 and 1 because wan_hr.dat has ordering dxz, dxy and dyz instead of alphabetical
    ek = ek[..., perm, :][..., perm]  # permute both row and column indices
    ek.imag[np.abs(ek.imag) < 1e-9] = 0
    config.lattice.hamiltonian._ek = ek
    g_loc = GreensFunction.create_g_loc(s_dmft.create_with_asympt_up_to_core(), ek)

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    (*_, s_loc) = local_sde.perform_local_schwinger_dyson(g_loc, g2_dens, g2_magn, u_loc)

    sigma_dga_cubic = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft, s_loc)

    niv = sigma_dga_cubic.current_shape[-1] // 2
    s_cubic = sigma_dga_cubic.compress_q_dimension().mat.reshape(8, 8, 8, 3, 3, 2 * niv)  # (nkx, nky, nkz, nb, nb, niv)

    s_xy_cub = np.swapaxes(s_cubic, 0, 1)
    s_xz_cub = np.swapaxes(s_cubic, 0, 2)
    s_yz_cub = np.swapaxes(s_cubic, 1, 2)

    atol = 1e-2  # 8x8x8 grid is too coarse for tight mirror symmetry checks

    # X_Y_SYM (kx<->ky): dxy(0) invariant, dxz(1)<->dyz(2) swap
    assert np.allclose(s_cubic[..., 0, 0, :], s_xy_cub[..., 0, 0, :], atol=atol), "X_Y_SYM dxy failed"
    assert np.allclose(s_cubic[..., 1, 1, :], s_xy_cub[..., 2, 2, :], atol=atol), "X_Y_SYM dxz<->dyz failed"

    # X_Z_SYM (kx<->kz): dxz(1) invariant, dxy(0)<->dyz(2) swap
    assert np.allclose(s_cubic[..., 1, 1, :], s_xz_cub[..., 1, 1, :], atol=atol), "X_Z_SYM dxz failed"
    assert np.allclose(s_cubic[..., 0, 0, :], s_xz_cub[..., 2, 2, :], atol=atol), "X_Z_SYM dxy<->dyz failed"

    # Y_Z_SYM (ky<->kz): dyz(2) invariant, dxy(0)<->dxz(1) swap
    assert np.allclose(s_cubic[..., 2, 2, :], s_yz_cub[..., 2, 2, :], atol=atol), "Y_Z_SYM dyz failed"
    assert np.allclose(s_cubic[..., 0, 0, :], s_yz_cub[..., 1, 1, :], atol=atol), "Y_Z_SYM dxy<->dxz failed"

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import contextlib
import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import dgamore.brillouin_zone as bz
from dgamore import config, dga_io, local_sde
from dgamore import nonlocal_sde
from dgamore.brillouin_zone import KnownSymmetries
from dgamore.dga_logger import DgaLogger
from tests import conftest


@pytest.fixture
def setup(monkeypatch):
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/end_2_end"

    comm_mock = conftest.create_comm_mock()

    monkeypatch.setattr("mpi4py.MPI.COMM_WORLD", comm_mock)
    config.logger = DgaLogger(comm_mock, "./")
    conftest.create_default_config(config, folder)
    yield folder, comm_mock


@pytest.fixture
def setup_srvo3_cubic(monkeypatch):
    def create_srvo3_cubic_config(c, f: str):
        c.box.niw_core = -1
        c.box.niv_core = -1
        c.box.niv_shell = 0
        c.output.do_plotting = False
        c.lattice.nk = (12, 12, 12)
        c.lattice.k_grid = bz.KGrid(c.lattice.nk, [KnownSymmetries.AUTO])
        c.lattice.symmetries = [KnownSymmetries.AUTO]
        c.lattice.type = "from_wannier90"
        c.lattice.interaction_type = "kanamori_from_dmft"
        c.lattice.er_input = f"{f}/wan_hr.dat"
        c.dmft.input_path = f
        c.dmft.symmetrize_orbitals = [1, 2, 3]
        c.dmft.n_ineq = 1
        c.dmft.ineq_ordering = [1]
        c.dmft.n_bands_per_ineq = []
        c.eliashberg.perform_eliashberg = False
        c.self_consistency.mixing = 1
        c.self_consistency.max_iter = 1
        c.sys.occ_dmft_per_ineq = []

    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/srvo3_end2end"

    comm_mock = conftest.create_comm_mock()

    monkeypatch.setattr("mpi4py.MPI.COMM_WORLD", comm_mock)
    config.logger = DgaLogger(comm_mock, "./")
    create_srvo3_cubic_config(config, folder)
    yield folder, comm_mock


def make_cupy_mock():
    """Builds a numpy-backed cupy module stand-in with a single available (no-op) GPU device."""
    cp = types.ModuleType("cupy")

    cp.asarray = np.asarray
    cp.zeros = np.zeros
    cp.empty = np.empty
    cp.empty_like = np.empty_like

    cp.asnumpy = lambda x: x
    cp.conj = np.conj

    cp.arange = np.arange
    cp.take = np.take

    cp.einsum = np.einsum
    cp.multiply = np.multiply

    cp.fft = types.ModuleType("cupy.fft")
    cp.fft.ifftn = np.fft.ifftn
    cp.fft.fftn = np.fft.ifft

    cp.cuda = types.ModuleType("cupy.cuda")
    cp.cuda.is_available = MagicMock(return_value=True)

    cp.cuda.runtime = types.ModuleType("cupy.cuda.runtime")
    cp.cuda.runtime.getDeviceCount = MagicMock(return_value=1)

    cp.cuda.Device = MagicMock()

    return cp


@contextlib.contextmanager
def gpu_cpu_context(use_gpu: bool, monkeypatch):
    """Yields whether the GPU is mocked: real cupy, the numpy stand-in, or disabled when CPU-forced."""
    mock_gpu = False
    if use_gpu:
        try:  # real GPU is available
            import cupy as cp

            yield mock_gpu
        except:  # fallback to mocked GPU
            mock_gpu = True
            mock_cupy = make_cupy_mock()
            with monkeypatch.context() as m:
                m.setitem(sys.modules, "cupy", mock_cupy)
                m.setitem(sys.modules, "cupy.cuda", mock_cupy.cuda)
                m.setitem(sys.modules, "cupy.cuda.runtime", mock_cupy.cuda.runtime)
                yield mock_gpu
    else:  # force CPU path
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "cupy", None)
            yield mock_gpu


@pytest.mark.parametrize(
    "niw_core, niv_core, niv_shell, use_gpu",
    [(20, 20, 10, True), (20, 20, 10, False)],
)
def test_calculates_nonlocal_sde_correctly(setup, monkeypatch, niw_core, niv_core, niv_shell, use_gpu):
    """The non-local SDE reproduces the reference self-energy on the CPU and the (mocked) GPU path."""
    folder, comm_mock = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell
    config.dmft.symmetrize_orbitals = []

    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]

    config.output.output_path = folder

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)

    *_, s_loc = local_sde.perform_local_schwinger_dyson(g_dmft, g2_dens, g2_magn, u_loc)

    with gpu_cpu_context(use_gpu, monkeypatch) as mock_gpu:
        sigma_dga = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft, s_loc)

    sigma_dga_mat = sigma_dga.decompress_q_dimension().cut_niv(50).mat
    sigma_dga_ref = np.load(f"{folder}/sigma_dga.npy")

    assert np.allclose(sigma_dga_mat, sigma_dga_ref, atol=3e-5 if not mock_gpu else 1e-3)


def test_calculates_srvo3_correctly(setup_srvo3_cubic):
    """The SrVO3 cubic run reproduces the cubic point-group symmetry of the self-energy under the three axis swaps."""
    folder_cubic, comm_mock = setup_srvo3_cubic

    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    assert g_dmft.is_orbitally_symmetrized([1, 2, 3])
    assert s_dmft.is_orbitally_symmetrized([1, 2, 3])
    assert g2_dens.is_orbitally_symmetrized([1, 2, 3])
    assert g2_magn.is_orbitally_symmetrized([1, 2, 3])

    config.output.output_path = folder_cubic

    ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
    # swap orbitals 0 and 1 in rows and columns because wan_hr.dat orders dxz, dxy, dyz instead of alphabetical
    perm = [1, 0, 2]
    ek = ek[..., perm, :][..., perm]
    ek.imag[np.abs(ek.imag) < 1e-9] = 0
    config.lattice.hamiltonian._ek = ek
    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]

    config.lattice.k_grid.specify_auto_symmetries(ek, atol=1e-12)

    s_dmft_ref = np.load(f"{folder_cubic}/sigma_dmft.npy")
    assert np.allclose(s_dmft_ref, s_dmft_ref)
    g_dmft_ref = np.load(f"{folder_cubic}/g_dmft.npy")
    assert np.allclose(g_dmft_ref, g_dmft_ref)

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)

    *_, s_loc = local_sde.perform_local_schwinger_dyson(g_dmft, g2_dens, g2_magn, u_loc)

    sigma_dga_cubic = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft, s_loc)

    niv = sigma_dga_cubic.current_shape[-1] // 2
    s_cubic = sigma_dga_cubic.compress_q_dimension().mat.reshape(12, 12, 12, 3, 3, 2 * niv)

    s_xy_cub = np.swapaxes(s_cubic, 0, 1)
    s_xz_cub = np.swapaxes(s_cubic, 0, 2)
    s_yz_cub = np.swapaxes(s_cubic, 1, 2)

    atol = 1e-6

    assert np.allclose(s_cubic[..., 0, 0, :], s_xy_cub[..., 0, 0, :], atol=atol), "X_Y_SYM dxy failed"
    assert np.allclose(s_cubic[..., 1, 1, :], s_xy_cub[..., 2, 2, :], atol=atol), "X_Y_SYM dxz<->dyz failed"

    assert np.allclose(s_cubic[..., 1, 1, :], s_xz_cub[..., 1, 1, :], atol=atol), "X_Z_SYM dxz failed"
    assert np.allclose(s_cubic[..., 0, 0, :], s_xz_cub[..., 2, 2, :], atol=atol), "X_Z_SYM dxy<->dyz failed"

    assert np.allclose(s_cubic[..., 2, 2, :], s_yz_cub[..., 2, 2, :], atol=atol), "Y_Z_SYM dyz failed"
    assert np.allclose(s_cubic[..., 0, 0, :], s_yz_cub[..., 1, 1, :], atol=atol), "Y_Z_SYM dxy<->dxz failed"

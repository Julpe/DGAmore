# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import logging
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import moldga.config as config
from moldga import local_sde, dga_io
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
        yield folder


@pytest.fixture(autouse=True)
def mock_logger(monkeypatch):
    """
    Automatically mock logger.log for all tests in this file.
    """
    logger_mock = MagicMock()
    monkeypatch.setattr(logging, "getLogger", lambda name=None: logger_mock)
    monkeypatch.setattr(logging, "Logger", MagicMock(return_value=logger_mock))
    yield logger_mock


@pytest.mark.parametrize(
    "niw_core, niv_core, niv_shell",
    [
        (10, 20, 0),
        (20, 10, 0),
        (20, 20, 0),
        (10, 20, 10),
        (20, 10, 10),
        (20, 20, 10),
        (-1, 20, 10),
        (20, -1, 10),
        (-1, -1, 10),
    ],
)
def test_extracts_dmft_quantities_correctly(setup, niw_core, niv_core, niv_shell):
    folder = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell

    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    if niw_core == -1:
        assert config.box.niw_core == 20
    if niv_core == -1:
        assert config.box.niv_core == 20

    assert config.box.niv_full == config.box.niv_core + config.box.niv_shell

    g2_dens_ref = (
        LocalFourPoint.load(f"{folder}/g2_dens_loc.npy")
        .to_full_niw_range()
        .cut_niw_and_niv(config.box.niw_core, config.box.niv_core)
    )
    g2_magn_ref = (
        LocalFourPoint.load(f"{folder}/g2_magn_loc.npy")
        .to_full_niw_range()
        .cut_niw_and_niv(config.box.niw_core, config.box.niv_core)
    )

    niv = 100
    cut = config.box.niw_core + config.box.niv_full + 10

    g_dmft_ref_mat = np.load(f"{folder}/g_dmft.npy")[..., niv - cut : niv + cut]
    s_dmft_ref_mat = np.load(f"{folder}/sigma_dmft.npy")[..., niv - cut : niv + cut]

    niv = g_dmft.mat.shape[-1] // 2
    g_dmft.mat = g_dmft.mat[..., niv - cut : niv + cut]
    s_dmft.mat = s_dmft.mat[..., niv - cut : niv + cut]

    assert np.allclose(g2_dens.mat, g2_dens_ref.mat)
    assert np.allclose(g2_magn.mat, g2_magn_ref.mat)

    assert np.allclose(g_dmft.mat, g_dmft_ref_mat)
    assert np.allclose(s_dmft.mat, s_dmft_ref_mat)

    assert config.box.niw_core == g2_dens.niw
    assert config.box.niv_core == g2_dens.niv

    u_loc = config.lattice.hamiltonian.get_local_u()
    assert u_loc.mat.shape == (2, 2, 2, 2)
    indices = np.arange(2)
    assert np.all(u_loc.mat[indices, indices, indices, indices] == 8)

    mask = np.ones(u_loc.mat.shape, dtype=bool)
    mask[indices, indices, indices, indices] = False
    assert np.all(u_loc.mat[mask] == 0)

    vq = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)
    assert vq.mat.shape == (4, 4, 1, 2, 2, 2, 2)
    assert np.allclose(vq.mat, np.zeros_like(vq.mat))


@pytest.mark.parametrize(
    "niw_core, niv_core, niv_shell",
    [
        (10, 20, 0),
        (20, 10, 0),
        (20, 20, 0),
        (10, 20, 10),
        (20, 10, 10),
        (20, 20, 10),
        (-1, 20, 10),
        (20, -1, 10),
        (-1, -1, 10),
    ],
)
def test_extracts_dmft_quantities_correctly_with_symmetrization(setup, niw_core, niv_core, niv_shell):
    folder = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell
    config.dmft.symmetrize_orbitals = [1, 2]

    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    if niw_core == -1:
        assert config.box.niw_core == 20
    if niv_core == -1:
        assert config.box.niv_core == 20

    assert config.box.niv_full == config.box.niv_core + config.box.niv_shell

    g2_dens_ref = (
        LocalFourPoint.load(f"{folder}/g2_dens_loc.npy")
        .to_full_niw_range()
        .cut_niw_and_niv(config.box.niw_core, config.box.niv_core)
    ).symmetrize_orbitals([1, 2])
    g2_magn_ref = (
        LocalFourPoint.load(f"{folder}/g2_magn_loc.npy")
        .to_full_niw_range()
        .cut_niw_and_niv(config.box.niw_core, config.box.niv_core)
    ).symmetrize_orbitals([1, 2])

    assert np.allclose(g_dmft[0, 0], g_dmft[1, 1])
    assert np.allclose(g_dmft[0, 1], g_dmft[1, 0])

    s_dmft = s_dmft[0, 0, 0]
    assert np.allclose(s_dmft[0, 0], s_dmft[1, 1])
    assert np.allclose(s_dmft[0, 1], s_dmft[1, 0])

    assert np.allclose(g2_dens.mat, g2_dens_ref.mat)
    assert np.allclose(g2_magn.mat, g2_magn_ref.mat)

    for g2 in [g2_dens, g2_magn]:
        assert np.array_equal(g2[0, 0, 0, 0], g2[1, 1, 1, 1])
        assert np.array_equal(g2[0, 0, 1, 1], g2[1, 1, 0, 0])
        assert np.array_equal(g2[0, 1, 0, 1], g2[1, 0, 1, 0])
        assert np.array_equal(g2[0, 1, 1, 0], g2[1, 0, 0, 1])

        assert np.array_equal(g2[1, 0, 0, 0], g2[0, 1, 1, 1])
        assert np.array_equal(g2[0, 1, 0, 0], g2[1, 0, 1, 1])
        assert np.array_equal(g2[0, 0, 1, 0], g2[1, 1, 0, 1])
        assert np.array_equal(g2[0, 0, 0, 1], g2[1, 1, 1, 0])


@pytest.mark.parametrize("niw_core, niv_core, niv_shell", [(20, 20, 10)])
def test_calculates_local_sde_correctly(setup, niw_core, niv_core, niv_shell):
    folder = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell
    config.dmft.symmetrize_orbitals = []

    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())
    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]

    u_loc = config.lattice.hamiltonian.get_local_u()

    (gamma_d, gamma_m, chi_d, chi_m, vrg_d, vrg_m, f_d, f_m, gchi_d, gchi_m, sigma_loc) = (
        local_sde.perform_local_schwinger_dyson(g_dmft, g2_dens, g2_magn, u_loc)
    )

    def compare_quantity(obj_dens_sde, obj_magn_sde, dens_name: str, magn_name: str, num_vn_dimensions: int, niv: int):
        obj_dens_ref = LocalFourPoint.load(
            f"{folder}/{dens_name}.npy", SpinChannel.DENS, num_vn_dimensions=num_vn_dimensions
        )
        obj_magn_ref = LocalFourPoint.load(
            f"{folder}/{magn_name}.npy", SpinChannel.MAGN, num_vn_dimensions=num_vn_dimensions
        )

        assert np.allclose(obj_dens_sde.mat, obj_dens_ref.mat, atol=1e-3)
        assert np.allclose(obj_magn_sde.mat, obj_magn_ref.mat, atol=1e-3)

        assert obj_dens_sde.channel == obj_dens_ref.channel
        assert obj_magn_sde.channel == obj_magn_ref.channel

        assert obj_dens_sde.num_wn_dimensions == obj_dens_ref.num_wn_dimensions
        assert obj_magn_sde.num_wn_dimensions == obj_magn_ref.num_wn_dimensions

        assert obj_dens_sde.num_vn_dimensions == obj_dens_ref.num_vn_dimensions
        assert obj_magn_sde.num_vn_dimensions == obj_magn_ref.num_vn_dimensions

        assert obj_dens_sde.niw == obj_dens_ref.niw
        assert obj_magn_sde.niw == obj_magn_ref.niw

        assert obj_dens_sde.niv == obj_dens_ref.niv
        assert obj_magn_sde.niv == obj_magn_ref.niv

        assert obj_dens_sde.full_niw_range == obj_dens_ref.full_niw_range
        assert obj_magn_sde.full_niw_range == obj_magn_ref.full_niw_range

        assert obj_dens_sde.current_shape == (2, 2, 2, 2, 21) + (2 * niv,) * num_vn_dimensions
        assert obj_magn_sde.current_shape == (2, 2, 2, 2, 21) + (2 * niv,) * num_vn_dimensions

    compare_quantity(gamma_d, gamma_m, "gamma_dens_loc", "gamma_magn_loc", 2, config.box.niv_core)
    compare_quantity(chi_d, chi_m, "chi_dens_loc", "chi_magn_loc", 0, config.box.niv_core)
    compare_quantity(vrg_d, vrg_m, "vrg_dens_loc", "vrg_magn_loc", 1, config.box.niv_core)
    compare_quantity(f_d, f_m, "f_dens_loc", "f_magn_loc", 2, config.box.niv_full)
    compare_quantity(gchi_d, gchi_m, "gchi_dens_loc", "gchi_magn_loc", 2, config.box.niv_core)

    sigma_loc_ref = np.load(f"{folder}/siw_dga_local.npy")
    assert np.allclose(sigma_loc.mat, sigma_loc_ref, atol=1e-5)

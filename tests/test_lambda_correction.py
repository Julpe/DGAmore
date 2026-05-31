# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
from unittest.mock import patch

import mpi4py
import numpy as np
import pytest

import dgamore.lambda_correction as lc
from dgamore import nonlocal_sde
from dgamore.dga_logger import DgaLogger
from dgamore.four_point import FourPoint
from dgamore.local_four_point import LocalFourPoint
from dgamore.n_point_base import SpinChannel
import dgamore.config as config
import dgamore.brillouin_zone as bz


def load_four_point(lc_type: str, filename: str, channel: SpinChannel) -> FourPoint:
    return FourPoint.load(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/lambda_correction/"
        + lc_type
        + "/"
        + filename
        + ".npy",
        channel,
        num_vn_dimensions=0,
        has_compressed_q_dimension=True,
        nq=(4, 4, 1),
    )


def load_local_four_point(lc_type: str, filename: str, channel: SpinChannel) -> LocalFourPoint:
    return LocalFourPoint.load(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/lambda_correction/"
        + lc_type
        + "/"
        + filename
        + ".npy",
        channel,
        num_vn_dimensions=0,
    )


def test_lambda_correction_spch():
    config.lattice.q_grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.lattice.k_grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.sys.beta = 12.5

    def perform_lc(channel: SpinChannel):
        chi_r_before_lambda = load_four_point("spch", f"chi_phys_q_{channel.value}_before_lambda", channel)
        chi_r_loc = load_local_four_point("spch", f"chi_{channel.value}_loc", channel).to_full_niw_range()
        chi_r_loc_sum = chi_r_loc.mat.sum() / config.sys.beta
        return lc.perform_single_lambda_correction(chi_r_before_lambda, chi_r_loc_sum)

    corrected_chi_dens, lambda_dens = perform_lc(SpinChannel.DENS)
    corrected_chi_magn, lambda_magn = perform_lc(SpinChannel.MAGN)

    reference_chi_dens = load_four_point("spch", "chi_phys_q_dens", SpinChannel.DENS)
    reference_chi_magn = load_four_point("spch", "chi_phys_q_magn", SpinChannel.MAGN)

    assert np.allclose(corrected_chi_dens.mat, reference_chi_dens.mat)
    assert np.allclose(corrected_chi_magn.mat, reference_chi_magn.mat)

    assert np.allclose(lambda_dens, -37.450340)
    assert np.allclose(lambda_magn, 4.328781)


def test_lambda_correction_sp():
    config.lattice.q_grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.sys.beta = 12.5

    chi_dens_q_before_lambda = load_four_point(
        "sp", f"chi_phys_q_dens_before_lambda", SpinChannel.DENS
    ).to_full_niw_range()
    chi_phys_q_dens = load_four_point("sp", f"chi_phys_q_dens", SpinChannel.DENS).to_full_niw_range()
    assert np.allclose(chi_dens_q_before_lambda.mat, chi_phys_q_dens.mat)

    chi_dens_loc_sum = load_local_four_point("sp", f"chi_dens_loc", SpinChannel.DENS).to_full_niw_range().mat.sum()
    chi_magn_loc_sum = load_local_four_point("sp", f"chi_magn_loc", SpinChannel.MAGN).to_full_niw_range().mat.sum()

    chi_loc_sum = (
        chi_dens_loc_sum
        + chi_magn_loc_sum
        - 1.0 / 16 * (config.lattice.q_grid.irrk_count[:, None, None, None, None, None] * chi_phys_q_dens.mat).sum()
    )

    chi_magn_q_before_lambda = load_four_point("sp", f"chi_phys_q_magn_before_lambda", SpinChannel.MAGN)
    corrected_chi_magn, lambda_magn = lc.perform_single_lambda_correction(
        chi_magn_q_before_lambda, chi_loc_sum / config.sys.beta
    )

    reference_chi_magn = load_four_point("sp", "chi_phys_q_magn", SpinChannel.MAGN)

    assert np.allclose(corrected_chi_magn.mat, reference_chi_magn.mat)

    assert np.allclose(lambda_magn, 4.281153)


@pytest.mark.parametrize("lc_type", ["sp", "spch"])
def test_lambda_correction_in_sde_sp(lc_type):
    config.lattice.q_grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.sys.beta = 12.5
    config.lambda_correction.type = lc_type
    config.output.output_path = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/lambda_correction/{lc_type}"

    with patch("mpi4py.MPI.COMM_WORLD", wraps=mpi4py.MPI.COMM_WORLD) as comm_mock:
        config.logger = DgaLogger(comm_mock, "./")

        chi_dens_q_before_lambda = load_four_point(lc_type, f"chi_phys_q_dens_before_lambda", SpinChannel.DENS)
        chi_magn_q_before_lambda = load_four_point(lc_type, f"chi_phys_q_magn_before_lambda", SpinChannel.MAGN)

        chi_magn_corrected = nonlocal_sde.perform_lambda_correction(chi_magn_q_before_lambda)
        chi_dens_corrected = nonlocal_sde.perform_lambda_correction(chi_dens_q_before_lambda)

        reference_chi_dens = load_four_point(lc_type, "chi_phys_q_dens", SpinChannel.DENS)
        reference_chi_magn = load_four_point(lc_type, "chi_phys_q_magn", SpinChannel.MAGN)

        assert np.allclose(chi_dens_corrected.mat, reference_chi_dens.mat)
        assert np.allclose(chi_magn_corrected.mat, reference_chi_magn.mat)

        if lc_type == "sp":
            assert np.allclose(chi_dens_corrected.mat, chi_dens_q_before_lambda.mat)

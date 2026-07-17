# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import fnmatch
import os

from dgamore import output_files
from dgamore.n_point_base import SpinChannel


def test_sigma_iteration_name_matches_glob_and_regex():
    """The per-iteration sigma name is found by the resume glob and its regex captures the iteration number."""
    filename = output_files.npy_filename(output_files.sigma_iteration_name(17))
    assert fnmatch.fnmatch(filename, output_files.SIGMA_ITERATION_GLOB)
    match = output_files.SIGMA_ITERATION_REGEX.search(filename)
    assert match is not None and int(match.group(1)) == 17


def test_sigma_interpolated_name_matches_glob_and_regex():
    """The interpolated sigma name matches its glob and regex, and the plain-iteration regex does not misfire on it."""
    filename = output_files.npy_filename(output_files.sigma_interpolated_iteration_name(12.5, 40, 3))
    assert fnmatch.fnmatch(filename, output_files.SIGMA_INTERPOLATED_GLOB)
    match = output_files.SIGMA_INTERPOLATED_REGEX.search(filename)
    assert match is not None and int(match.group(1)) == 3
    assert output_files.SIGMA_ITERATION_REGEX.search(filename) is None


def test_npy_path_joins_directory_and_extension():
    """npy_path is the directory join of the bare name plus the .npy extension (the save/load convention)."""
    assert output_files.npy_filename("some_name") == "some_name.npy"
    assert output_files.npy_path("/out/dir", "some_name") == os.path.join("/out/dir", "some_name.npy")


def test_eliashberg_rank_names_lock_the_on_disk_contract():
    """The per-rank Eliashberg intermediate names are locked exactly so the SDE-to-Eliashberg handoff cannot break."""
    assert output_files.gchi0_q_inv_rank_name(4) == "gchi0_q_inv_rank_4"
    assert output_files.vrg_q_rank_name(SpinChannel.DENS, 0) == "vrg_q_dens_rank_0"
    assert output_files.vrg_q_right_rank_name(SpinChannel.MAGN, 2) == "vrg_q_magn_right_rank_2"
    assert output_files.chi_phys_q_rank_name(SpinChannel.MAGN, 7) == "chi_phys_q_magn_rank_7"


def test_local_vertex_and_susceptibility_names_lock_the_on_disk_contract():
    """The local-vertex and susceptibility names shared across writer and reader modules are locked exactly."""
    assert output_files.local_vertex_name("gamma", SpinChannel.DENS) == "gamma_dens_loc"
    assert output_files.local_vertex_name("f", SpinChannel.MAGN) == "f_magn_loc"
    assert output_files.chi_phys_q_name(SpinChannel.DENS) == "chi_phys_q_dens"
    assert output_files.chi_rpa_q_name(SpinChannel.MAGN) == "chi_rpa_q_magn"
    assert output_files.MU_HISTORY_FILENAME == "mu_history.npy"
    assert output_files.G_LATT_DMFT_NAME == "g_latt_dmft"

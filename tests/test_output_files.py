# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os

from dgamore import output_files
from dgamore.n_point_base import SpinChannel


def test_local_name_spells_the_local_vertex_files():
    """A local vertex file is named <kind>_<channel>_loc."""
    assert output_files.local_name("gamma", SpinChannel.DENS) == "gamma_dens_loc"
    assert output_files.local_name("chi", SpinChannel.MAGN) == "chi_magn_loc"


def test_channel_name_appends_the_channel_value():
    """A channel file is named <stem>_<channel>."""
    assert output_files.channel_name(output_files.CHI_PHYS_Q, SpinChannel.MAGN) == "chi_phys_q_magn"


def test_rank_name_spells_the_per_rank_files_with_and_without_a_channel():
    """A per-rank file is named <stem>_<channel>_rank_<r>, or <stem>_rank_<r> without a channel."""
    assert output_files.rank_name(output_files.VRG_Q, 3, SpinChannel.MAGN) == "vrg_q_magn_rank_3"
    assert output_files.rank_name(output_files.VRG_MIDDLE_Q, 0, SpinChannel.DENS) == "vrg_middle_q_dens_rank_0"
    assert output_files.rank_name(output_files.GCHI0_Q_INV, 0) == "gchi0_q_inv_rank_0"


def test_npy_path_appends_the_extension_inside_the_folder():
    """npy_path joins the folder and the bare name with the .npy extension."""
    assert output_files.npy_path("/run", "f_dc_loc") == os.path.join("/run", "f_dc_loc.npy")


def test_fixed_names_keep_their_on_disk_spelling():
    """The fixed stage-to-stage names keep the spelling the run folders already carry."""
    fixed = (output_files.ABORT_MARKER, output_files.GIWK_FILE_STEM, output_files.F_DC_LOC, output_files.G_DMFT)
    assert fixed == (".aborting", "giwk_core_rank_0", "f_dc_loc", "g_dmft")
    assert (output_files.G_LATT_DMFT, output_files.SIGMA_DGA) == ("g_latt_dmft", "sigma_dga")
    stems = (output_files.VRG_Q, output_files.VRG_MIDDLE_Q, output_files.CHI_PHYS_Q, output_files.GCHI0_Q_INV)
    assert stems == ("vrg_q", "vrg_middle_q", "chi_phys_q", "gchi0_q_inv")

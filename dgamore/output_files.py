# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Single source of truth for the names of the ``.npy`` files handed between the pipeline stages. The local
Schwinger-Dyson step writes the assembled local vertices, the non-local SDE step writes the per-rank ladder
intermediates consumed by the Eliashberg step and the per-iteration self-energies consumed by the mixing history,
and the lambda corrections read the local and physical susceptibilities back from file. Every writer and reader
builds the file name through the functions and constants here, so a name can never silently diverge between the
module that writes it and the module that reads it. All name builders return the bare file name **without** the
``.npy`` extension (the convention of :meth:`~dgamore.local_n_point.LocalNPoint.save`); :func:`npy_filename` and
:func:`npy_path` append it for the load/delete sites.
"""

import os
import re

from dgamore.n_point_base import SpinChannel

# Glob patterns and regexes for the per-iteration self-energy dumps (the regex group captures the iteration).
SIGMA_ITERATION_GLOB: str = "sigma_dga_iteration_*.npy"
SIGMA_ITERATION_REGEX: re.Pattern = re.compile(r"sigma_dga_iteration_(\d+)\.npy$")
SIGMA_INTERPOLATED_GLOB: str = "sigma_dga_interpolated_*_iteration_*.npy"
SIGMA_INTERPOLATED_REGEX: re.Pattern = re.compile(r"sigma_dga_interpolated_.+_iteration_(\d+)\.npy$")

# Fixed file names (written and read verbatim).
MU_HISTORY_FILENAME: str = "mu_history.npy"
G_LATT_DMFT_NAME: str = "g_latt_dmft"


def npy_filename(name: str) -> str:
    """
    Returns the ``.npy`` file name for a bare object name (as produced by the name builders here).

    :param name: The bare file name without extension.
    :return: The file name with the ``.npy`` extension.
    """
    return f"{name}.npy"


def npy_path(directory: str, name: str) -> str:
    """
    Returns the full path of a saved object: the directory joined with the bare name plus the ``.npy`` extension.

    :param directory: The directory the file lives in.
    :param name: The bare file name without extension.
    :return: The full file path.
    """
    return os.path.join(directory, npy_filename(name))


def local_vertex_name(kind: str, channel: SpinChannel) -> str:
    r"""
    Returns the file name of an assembled full multi-band local vertex written after the local Schwinger-Dyson
    step, e.g. ``gamma_dens_loc`` or ``f_magn_loc``.

    :param kind: The vertex kind: ``"gamma"``, ``"chi"``, ``"g2"``, ``"vrg"``, ``"gchi"`` or ``"f"``.
    :param channel: The spin channel.
    :return: The bare file name.
    """
    return f"{kind}_{channel.value}_loc"


def chi_phys_q_name(channel: SpinChannel) -> str:
    r"""
    Returns the file name of the physical susceptibility :math:`\chi^{q}_{r}` (irreducible BZ, rank-0 gathered).

    :param channel: The spin channel.
    :return: The bare file name.
    """
    return f"chi_phys_q_{channel.value}"


def chi_rpa_q_name(channel: SpinChannel) -> str:
    r"""
    Returns the file name of the RPA susceptibility :math:`\chi^{q}_{r;\mathrm{RPA}}`.

    :param channel: The spin channel.
    :return: The bare file name.
    """
    return f"chi_rpa_q_{channel.value}"


def gchi0_q_inv_rank_name(rank: int) -> str:
    r"""
    Returns the file name of a rank's slice of the inverse bare bubble :math:`(\chi_0^q)^{-1}` (an Eliashberg
    intermediate written per MPI rank by the non-local SDE step).

    :param rank: The MPI rank owning the q-slice.
    :return: The bare file name.
    """
    return f"gchi0_q_inv_rank_{rank}"


def vrg_q_rank_name(channel: SpinChannel, rank: int) -> str:
    r"""
    Returns the file name of a rank's slice of the three-leg vertex :math:`\gamma^q_{r}` (an Eliashberg
    intermediate written per MPI rank by the non-local SDE step).

    :param channel: The spin channel.
    :param rank: The MPI rank owning the q-slice.
    :return: The bare file name.
    """
    return f"vrg_q_{channel.value}_rank_{rank}"


def vrg_q_right_rank_name(channel: SpinChannel, rank: int) -> str:
    r"""
    Returns the file name of a rank's slice of the right-sided three-leg vertex :math:`\tilde{\gamma}^q_{r}` (an
    Eliashberg intermediate written per MPI rank by the non-local SDE step).

    :param channel: The spin channel.
    :param rank: The MPI rank owning the q-slice.
    :return: The bare file name.
    """
    return f"vrg_q_{channel.value}_right_rank_{rank}"


def chi_phys_q_rank_name(channel: SpinChannel, rank: int) -> str:
    r"""
    Returns the file name of a rank's slice of the physical susceptibility :math:`\chi^{q}_{r}` (an Eliashberg
    intermediate written per MPI rank by the non-local SDE step).

    :param channel: The spin channel.
    :param rank: The MPI rank owning the q-slice.
    :return: The bare file name.
    """
    return f"chi_phys_q_{channel.value}_rank_{rank}"


def sigma_iteration_name(iteration: int) -> str:
    r"""
    Returns the file name of the mixed self-energy of one self-consistency iteration (read back by the resume
    logic and the accelerated-mixing history; see :data:`SIGMA_ITERATION_GLOB` / :data:`SIGMA_ITERATION_REGEX`).

    :param iteration: The self-consistency iteration number.
    :return: The bare file name.
    """
    return f"sigma_dga_iteration_{iteration}"


def sigma_interpolated_iteration_name(beta_target: float, niv_target: int, iteration: int) -> str:
    r"""
    Returns the file name of the interpolated self-energy of one self-consistency iteration (read back by the
    resume logic; see :data:`SIGMA_INTERPOLATED_GLOB` / :data:`SIGMA_INTERPOLATED_REGEX`).

    :param beta_target: The target inverse temperature of the interpolation.
    :param niv_target: The target number of positive fermionic frequencies of the interpolation.
    :param iteration: The self-consistency iteration number.
    :return: The bare file name.
    """
    return f"sigma_dga_interpolated_beta{beta_target}_niv{niv_target}_iteration_{iteration}"

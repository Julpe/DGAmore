# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Names of the files one pipeline stage writes and a later stage reads: the local vertices of the local step, the
per-rank intermediates the non-local step hands to the Eliashberg step, the physical susceptibilities, the lattice
Green's functions and the self-energy. Writer and reader both take the name from here.

Names are bare (no extension), as :meth:`~dgamore.local_n_point.LocalNPoint.save` expects them; :func:`npy_path`
gives the path a load or a delete opens.
"""

import os

import dgamore.config as config
from dgamore.n_point_base import SpinChannel

ABORT_MARKER = ".aborting"  # placed in the Eliashberg folder by a failed rank before it deletes the rank files
GIWK_FILE_STEM = "giwk_core_rank_0"  # the kernel step's Green's function on the box of the core bubble, by rank 0
F_DC_LOC = "f_dc_loc"  # the local double-counting vertex
G_DMFT = "g_dmft"  # the local DMFT Green's function
G_LATT_DMFT = "g_latt_dmft"  # the lattice Green's function of the DMFT self-energy
SIGMA_DGA = "sigma_dga"  # the non-local DGA self-energy
VRG_Q = "vrg_q"  # per-rank three-leg vertex of the kernel step
VRG_MIDDLE_Q = "vrg_middle_q"  # per-rank middle vertex of the kernel step
CHI_PHYS_Q = "chi_phys_q"  # physical susceptibility, whole and per rank
GCHI0_Q_INV = "gchi0_q_inv"  # per-rank inverse bubble


def local_name(kind: str, channel: SpinChannel) -> str:
    """
    Returns the name of a local vertex file, ``<kind>_<channel>_loc``.

    :param kind: The local quantity, e.g. ``"gamma"``, ``"chi"``, ``"gchi"``, ``"vrg"``, ``"g2"`` or ``"f"``.
    :param channel: Spin channel of the object (see :class:`SpinChannel`).
    :return: The bare file name.
    """
    return f"{kind}_{channel.value}_loc"


def channel_name(stem: str, channel: SpinChannel) -> str:
    """
    Returns the name of a file that holds one spin channel, ``<stem>_<channel>``.

    :param stem: The file stem, e.g. :data:`CHI_PHYS_Q`.
    :param channel: Spin channel of the object (see :class:`SpinChannel`).
    :return: The bare file name.
    """
    return f"{stem}_{channel.value}"


def rank_name(stem: str, rank: int, channel: SpinChannel | None = None) -> str:
    """
    Returns the name of a per-rank file, ``<stem>_<channel>_rank_<rank>``, or ``<stem>_rank_<rank>`` without a
    channel.

    :param stem: The file stem, e.g. :data:`VRG_Q`.
    :param rank: The rank that writes the file.
    :param channel: Spin channel of the object (see :class:`SpinChannel`), or ``None``.
    :return: The bare file name.
    """
    if channel is None:
        return f"{stem}_rank_{rank}"
    return f"{stem}_{channel.value}_rank_{rank}"


def npy_path(folder: str, name: str) -> str:
    """
    Returns the path of the ``.npy`` file of a bare name inside a folder.

    :param folder: The folder.
    :param name: The bare file name.
    :return: The path ``<folder>/<name>.npy``.
    """
    return os.path.join(folder, f"{name}.npy")


def delete_files(filepath: str, *args) -> None:
    """
    Deletes files in the given directory. If a file is not found, it is ignored. The deleted files are usually
    temporary files that are no longer needed after the calculation is done.

    :param filepath: Directory containing the files.
    :param args: One or more file names (relative to ``filepath``) to delete.
    :return: None.
    :raises TypeError: If any of the given names is not a string.
    """
    for name in args:
        if not isinstance(name, str):
            raise TypeError(f"Expected string, got {type(name)}.")
        full_path = os.path.join(filepath, name)
        if os.path.isfile(full_path):
            try:
                os.remove(full_path)
            except OSError:
                config.logger.info(f"Error deleting file: {name}.")

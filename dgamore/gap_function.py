# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
The superconducting gap function :math:`\Delta`, i.e. the eigenvector of the linearized Eliashberg equation.
"""

import numpy as np

from dgamore.n_point_base import IHaveChannel, SpinChannel, FrequencyNotation
from dgamore.two_point import TwoPoint


class GapFunction(TwoPoint, IHaveChannel):
    """
    Represents the superconducting gap function. Has one momentum dimension, two orbital dimensions and one fermionic
    frequency dimension. Inherits the two-point orbital bookkeeping and the irreducible-to-full-BZ unfold from
    :class:`TwoPoint` and adds the pairing (singlet/triplet) channel.
    """

    def __init__(
        self,
        mat: np.ndarray,
        channel: SpinChannel = SpinChannel.NONE,
        nk: tuple[int, int, int] = (1, 1, 1),
        full_niv_range: bool = True,
        has_compressed_q_dimension: bool = False,
    ):
        r"""
        Initializes the gap function in the given pairing channel and momentum layout.

        :param mat: Gap-function array with one momentum dimension, two orbital axes and one fermionic frequency axis.
        :param channel: Pairing channel, i.e. singlet or triplet (see :class:`SpinChannel`).
        :param nk: Number of momenta per spatial direction ``(nkx, nky, nkz)``.
        :param full_niv_range: Whether the object spans the full (signed) fermionic range or only
            :math:`\nu \geq 0`.
        :param has_compressed_q_dimension: Whether the momentum is stored as a single compressed axis ``[q, ...]``
            (True) or as three separate axes ``[kx, ky, kz, ...]`` (False).
        """
        TwoPoint.__init__(self, mat, nk, full_niv_range, has_compressed_q_dimension)
        IHaveChannel.__init__(self, channel, FrequencyNotation.PP)

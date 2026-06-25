# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
The superconducting gap function :math:`\\Delta`, i.e. the eigenvector of the linearized Eliashberg equation.
"""

import numpy as np

from dgamore.brillouin_zone import KGrid
from dgamore.local_n_point import LocalNPoint
from dgamore.n_point_base import IAmNonLocal, IHaveChannel, SpinChannel, FrequencyNotation


class GapFunction(IAmNonLocal, LocalNPoint, IHaveChannel):
    """
    Represents the superconducting gap function. Has one momentum dimension, two orbital dimensions and one fermionic
    frequency dimension.
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
        :param full_niv_range: Whether the fermionic frequency axis spans the full (signed) range or only
            :math:`\nu \geq 0`.
        :param has_compressed_q_dimension: Whether the momentum is stored as a single compressed axis ``[q, ...]``
            (True) or as three separate axes ``[qx, qy, qz, ...]`` (False).
        """
        LocalNPoint.__init__(self, mat, 2, 0, 1, full_niv_range=full_niv_range)
        IAmNonLocal.__init__(self, mat, nk, has_compressed_q_dimension=has_compressed_q_dimension)
        IHaveChannel.__init__(self, channel, FrequencyNotation.PP)

    def map_to_full_bz(self, k_grid: KGrid, nq: tuple = None):
        """
        Maps the gap function from the irreducible to the full Brillouin zone using the grid's symmetry map.

        :param k_grid: The momentum grid carrying the irreducible-to-full-BZ mapping (and orbital rotations).
        :param nq: Optional override for the number of momenta; if None the object's own ``nq`` is used.
        :return: ``self`` expanded to the full BZ (with two orbital dimensions transformed).
        """
        return self._map_to_full_bz(k_grid, 2, nq)

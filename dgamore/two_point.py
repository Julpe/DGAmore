# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Momentum-dependent two-point objects. :class:`TwoPoint` extends :class:`LocalTwoPoint` with one momentum axis
(see :class:`IAmNonLocal`) and is the common parent of the single-particle Green's function, the self-energy and
the superconducting gap function. It overrides the orbital permutation to account for the leading momentum axis
and adds the irreducible-to-full-BZ unfold with two orbital dimensions. It mirrors how :class:`FourPoint` extends
:class:`LocalFourPoint`.
"""

from copy import deepcopy

import numpy as np

from dgamore.brillouin_zone import KGrid
from dgamore.local_two_point import LocalTwoPoint
from dgamore.n_point_base import IAmNonLocal


class TwoPoint(IAmNonLocal, LocalTwoPoint):
    """
    Base class for the momentum-dependent two-point objects (Green's function, self-energy, gap function). They carry
    one momentum dimension, two orbital dimensions, no bosonic and exactly one fermionic frequency dimension. The
    momentum-agnostic orbital symmetrization helpers are inherited from :class:`LocalTwoPoint`; the orbital
    permutation is overridden here to keep the leading momentum axis fixed, and the irreducible-to-full-BZ unfold is
    added.
    """

    def __init__(
        self,
        mat: np.ndarray,
        nk: tuple[int, int, int] = (1, 1, 1),
        full_niv_range: bool = True,
        has_compressed_q_dimension: bool = False,
    ):
        r"""
        Initializes the momentum-dependent two-point object.

        :param mat: Underlying array (one momentum axis -- compressed or not -- then two orbital axes and one
            fermionic frequency axis).
        :param nk: Number of momenta per spatial direction ``(nkx, nky, nkz)``.
        :param full_niv_range: Whether the object spans the full (signed) fermionic range or only :math:`\nu \geq 0`.
        :param has_compressed_q_dimension: Whether the momentum is stored as a single compressed axis ``[q, ...]``
            (True) or as three separate axes ``[kx, ky, kz, ...]`` (False).
        """
        LocalTwoPoint.__init__(self, mat, full_niv_range)
        IAmNonLocal.__init__(self, mat, nk, has_compressed_q_dimension=has_compressed_q_dimension)

    def permute_orbitals(self, permutation: str = "ab->ab") -> "TwoPoint":
        """
        Permutes the two orbital axes according to an einsum-style string, returning a copy (the identity
        permutation returns ``self``). The momentum and frequency axes are kept fixed.

        :param permutation: A permutation of the form ``"ab->..."`` using exactly the two orbital labels.
        :return: The orbital-permuted object (a copy), or ``self`` for the identity permutation.
        :raises ValueError: If the permutation is malformed or does not list both orbitals on each side.
        """
        split = permutation.split("->")
        if len(split) != 2 or len(split[0]) != 2 or len(split[1]) != 2:
            raise ValueError("Invalid permutation.")

        if split[0] == split[1]:
            return self

        copy = deepcopy(self)

        permutation = (
            (
                f"i{split[0]}...->i{split[1]}..."
                if self.has_compressed_q_dimension
                else f"ijk{split[0]}...->ijk{split[1]}..."
            )
            if len(self.current_shape) != 3
            else f"{split[0]}v->{split[1]}v"
        )

        copy.mat = np.einsum(permutation, copy.mat, optimize=True)
        return copy

    def map_to_full_bz(self, k_grid: KGrid, nq: tuple = None):
        """
        Unfolds the object from the irreducible BZ to the full BZ using the grid's symmetry index map (see
        :meth:`IAmNonLocal._map_to_full_bz`), with two orbital dimensions.

        :param k_grid: The :class:`KGrid` providing the irreducible-to-full BZ index mapping.
        :param nq: Optional number of momenta per direction for the unfolded grid; defaults to the object's ``nq``.
        :return: ``self`` defined on the full BZ.
        """
        return self._map_to_full_bz(k_grid, 2, nq)

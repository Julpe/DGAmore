# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Base class for the local (momentum-independent) two-point quantities. :class:`LocalTwoPoint` wraps a single array
carrying **two orbital indices and exactly one fermionic frequency axis** (no bosonic axis), e.g. a local Green's
function or self-energy, and collects the orbital bookkeeping shared by every two-point object: locating the
orbital axes for the current layout, orbital permutation/transposition and orbital symmetrization. It mirrors
:class:`LocalFourPoint` for the two-orbital case; the momentum-dependent counterpart :class:`TwoPoint` adds the
single momentum dimension on top (just as :class:`FourPoint` extends :class:`LocalFourPoint`).
"""

from copy import deepcopy

import numpy as np

from dgamore.local_n_point import LocalNPoint


class LocalTwoPoint(LocalNPoint):
    """
    Base class for the local (momentum-independent) two-point objects. They carry two orbital dimensions, no bosonic
    and exactly one fermionic frequency dimension. The orbital-axis lookup and the orbital permutation/symmetrization
    helpers are shared and live here; the momentum-dependent subclass :class:`TwoPoint` overrides the orbital
    permutation to account for the leading momentum axis and adds the irreducible-to-full-BZ unfold.
    """

    def __init__(self, mat: np.ndarray, full_niv_range: bool = True):
        r"""
        Initializes the local two-point object with two orbital axes, no bosonic axis and one fermionic frequency axis.

        :param mat: Underlying array laid out as ``[o1, o2, v]``.
        :param full_niv_range: Whether the object spans the full (signed) fermionic range or only :math:`\nu \geq 0`.
        """
        LocalNPoint.__init__(self, mat, 2, 0, 1, full_niv_range=full_niv_range)

    def _get_orbital_axes(self) -> tuple[int, int]:
        """
        Determines the axes carrying the two orbital indices for the current layout (local, compressed-q, or
        decompressed-q). The momentum-dependent layouts are handled here too so the :class:`TwoPoint` subclass can
        reuse this method unchanged.

        :return: The tuple of the two orbital axis indices.
        :raises ValueError: If the object does not have 3, 4, or 6 dimensions.
        """
        if len(self.current_shape) == 3:  # [o1,o2,v]
            return 0, 1
        elif len(self.current_shape) == 4:  # [k,o1,o2,v]
            return 1, 2
        elif len(self.current_shape) == 6:  # [kx,ky,kz,o1,o2,v]
            return 3, 4
        raise ValueError("The object has to have either 3, 4 or 6 dimensions.")

    def permute_orbitals(self, permutation: str = "ab->ab") -> "LocalTwoPoint":
        """
        Permutes the two orbital axes according to an einsum-style string, returning a copy (the identity
        permutation returns ``self``). The frequency axis is kept fixed.

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
        copy.mat = np.einsum(f"{split[0]}...->{split[1]}...", copy.mat, optimize=True)
        return copy

    def transpose_orbitals(self) -> "LocalTwoPoint":
        r"""
        Transposes the two orbital indices, :math:`M_{ab} \to M_{ba}` (see :meth:`permute_orbitals`).

        :return: The orbitally transposed object (a copy).
        """
        return self.permute_orbitals("ab->ba")

    def symmetrize_orbitals(self, orbitals: list | np.ndarray) -> "LocalTwoPoint":
        """
        Symmetrizes the object with respect to the given (equivalent) orbitals by averaging over all permutations of
        those orbitals applied to the two orbital axes. The orbital labels are 1-based, ranging from 1 to the number
        of bands; e.g. for a 3-band object ``orbitals=[1, 3]`` symmetrizes the first and third orbital.

        :param orbitals: 1-based orbital indices to treat as equivalent.
        :return: The symmetrized object (``self`` unchanged if already symmetrized).
        """
        orbital_axes = self._get_orbital_axes()
        if self.is_orbitally_symmetrized(orbitals):
            return self
        return self._symmetrize_orbitals(orbitals, orbital_axes)

    def is_orbitally_symmetrized(self, orbitals: list | np.ndarray) -> bool:
        """
        Checks whether the object is already symmetric under all permutations of the given orbitals.

        :param orbitals: 1-based orbital indices to test for equivalence.
        :return: True if the object is invariant under permutations of those orbitals.
        """
        return self._is_orbitally_symmetrized(orbitals, self._get_orbital_axes())

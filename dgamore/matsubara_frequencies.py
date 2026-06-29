# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Helpers for Matsubara frequency arithmetic. :class:`MFHelper` builds the integer index grids and the
corresponding real bosonic/fermionic Matsubara frequencies used throughout the code, and converts
particle-hole (ph) frequency triples to particle-particle (pp) notation.
"""

from enum import Enum

import numpy as np
from multimethod import multimethod


class FrequencyShift(Enum):
    """
    Enum for the direction of an asymptotic frequency shift (used when extending a quantity beyond its core box).

    :cvar MINUS: Shift towards negative frequencies.
    :cvar PLUS: Shift towards positive frequencies.
    :cvar CENTER: Centered (symmetric) shift.
    :cvar NONE: No shift.
    """

    MINUS: str = "minus"
    PLUS: str = "plus"
    CENTER: str = "center"
    NONE: str = "none"


class MFHelper:
    """
    Collection of static helpers for Matsubara frequency index and grid arithmetic.
    """

    @multimethod
    @staticmethod
    def wn(niw: int, shift: int = 0, return_only_positive: bool = False) -> np.ndarray:
        r"""
        Returns the integer bosonic Matsubara indices in the closed interval :math:`[-\mathrm{niw}, \mathrm{niw}]`,
        optionally shifted. This is the index-only overload (no temperature dependence).

        :param niw: Half-width of the bosonic frequency box (number of positive bosonic frequencies).
        :param shift: Integer offset added to the whole index range.
        :param return_only_positive: If True, return only the non-negative indices :math:`[0, \mathrm{niw}]`.
        :return: 1D integer array of bosonic Matsubara indices.
        """
        if return_only_positive:
            return np.arange(shift, niw + shift + 1)
        return np.arange(-niw + shift, niw + shift + 1)

    @multimethod
    @staticmethod
    def wn(niw: int, beta: float, shift: int = 0, return_only_positive: bool = False) -> np.ndarray:
        r"""
        Returns the real bosonic Matsubara frequencies :math:`\omega_n = 2 n \pi / \beta` for the index range
        :math:`n \in [-\mathrm{niw}, \mathrm{niw}]`, optionally shifted.

        :param niw: Half-width of the bosonic frequency box (number of positive bosonic frequencies).
        :param beta: Inverse temperature :math:`\beta`.
        :param shift: Integer offset added to the whole index range before scaling.
        :param return_only_positive: If True, return only the non-negative frequencies.
        :return: 1D real array of bosonic Matsubara frequencies.
        """
        return np.pi / beta * 2 * MFHelper.wn(niw, shift, return_only_positive)

    @multimethod
    @staticmethod
    def vn(niv: int, shift: int = 0, return_only_positive: bool = False) -> np.ndarray:
        r"""
        Returns the integer fermionic Matsubara indices in the half-open interval
        :math:`[-\mathrm{niv}, \mathrm{niv})`, optionally shifted. This is the index-only overload (no temperature
        dependence).

        :param niv: Half-width of the fermionic frequency box (number of positive fermionic frequencies).
        :param shift: Integer offset added to the whole index range.
        :param return_only_positive: If True, return only the non-negative indices :math:`[0, \mathrm{niv})`.
        :return: 1D integer array of fermionic Matsubara indices.
        """
        if return_only_positive:
            return np.arange(shift, niv + shift)
        return np.arange(-niv + shift, niv + shift)

    @multimethod
    @staticmethod
    def vn(niv: int, beta: float, shift: int = 0, return_only_positive: bool = False) -> np.ndarray:
        r"""
        Returns the real fermionic Matsubara frequencies :math:`\nu_n = (2 n + 1) \pi / \beta` for the index range
        :math:`n \in [-\mathrm{niv}, \mathrm{niv})`, optionally shifted.

        :param niv: Half-width of the fermionic frequency box (number of positive fermionic frequencies).
        :param beta: Inverse temperature :math:`\beta`.
        :param shift: Integer offset added to the whole index range before scaling.
        :param return_only_positive: If True, return only the positive frequencies.
        :return: 1D real array of fermionic Matsubara frequencies.
        """
        return np.pi / beta * (2 * MFHelper.vn(niv, shift, return_only_positive) + 1)

    @staticmethod
    def get_frequencies_for_ph_to_pp_w0_channel_conversion(
        niw: int, niv: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Returns the index arrays that map a particle-hole quantity onto the :math:`\omega = 0` particle-particle
        notation, i.e. the :math:`(\omega', \nu_1', \nu_2')` indices such that
        :math:`F_{pp}[\omega, \nu_1, \nu_2] = F_{ph}[\omega', \nu_1', \nu_2']` with the frequency shift
        :math:`(\omega = 0, \nu_1, \nu_2) \to (\nu_1 + \nu_2, \nu_1, \nu_2)`. The returned arrays are already
        offset so they can be used to index directly into the full (positive-and-negative) ph frequency axes.

        :param niw: Half-width of the bosonic frequency box of the source ph quantity.
        :param niv: Half-width of the fermionic frequency box of the source ph quantity.
        :return: Tuple ``(wn_pp, vn_pp, vpn_pp)`` of index arrays for the bosonic and the two fermionic axes.
        """
        niv_pp = min(niw // 2, niv)
        vn = MFHelper.vn(niv_pp)
        vn_pp, vpn_pp = np.meshgrid(vn, vn, indexing="ij")

        wn_pp = niw + vn_pp + vpn_pp + 1
        vn_pp += niv
        vpn_pp += niv

        return wn_pp, vn_pp, vpn_pp

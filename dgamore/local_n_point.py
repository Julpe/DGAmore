# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Base class for all local (momentum-independent) N-point quantities. :class:`LocalNPoint` adds the orbital and
frequency-axis bookkeeping (number of orbital/bosonic/fermionic axes, full vs. half frequency ranges) and the
shared frequency transformations (cut, diagonal extension, full/half range conversion, orbital symmetrization)
on top of :class:`IHaveMat`.
"""

import itertools
import os
from copy import deepcopy

import numpy as np

from dgamore.n_point_base import IHaveMat


class LocalNPoint(IHaveMat):
    """
    Base class for all (Local)NPoint objects, such as the (Full/Irreducible) Vertex functions, Susceptibilities,
    Fermi-Bose Vertices, Green's Function, Self-Energy and the like. Removes redundancy of a lot of methods to make
    the implementation more efficient.
    """

    def __init__(
        self,
        mat: np.ndarray,
        num_orbital_dimensions: int,
        num_wn_dimensions: int,
        num_vn_dimensions: int,
        full_niw_range: bool = True,
        full_niv_range: bool = True,
    ):
        r"""
        Stores the array together with its orbital- and frequency-axis bookkeeping.

        :param mat: The underlying numpy array laid out as ``[orbitals..., (w), (v), (v')]``.
        :param num_orbital_dimensions: Number of orbital axes; 2 (two-point) or 4 (three-/four-leg vertex).
        :param num_wn_dimensions: Number of bosonic frequency axes (0 or 1).
        :param num_vn_dimensions: Number of fermionic frequency axes (0, 1 or 2).
        :param full_niw_range: Whether the bosonic axis spans the full (signed) range or only :math:`\omega \geq 0`.
        :param full_niv_range: Whether the fermionic axes span the full (signed) range or only :math:`\nu \geq 0`.
        """
        IHaveMat.__init__(self, mat)

        assert num_orbital_dimensions in (2, 4), "2 or 4 orbital dimensions are supported."
        self._num_orbital_dimensions = num_orbital_dimensions

        assert num_vn_dimensions in (0, 1, 2), "0 - 2 fermionic frequency dimensions are supported."
        self._num_vn_dimensions = num_vn_dimensions

        assert num_wn_dimensions in (0, 1), "0 or 1 bosonic frequency dimensions are supported."
        self._num_wn_dimensions = num_wn_dimensions

        self._full_niv_range = full_niv_range
        self._full_niw_range = full_niw_range

    @property
    def n_bands(self) -> int:
        """
        Returns the number of bands. Since these objects are momentum-independent, the orbital dimension is always
        in the first dimension.

        :return: The number of bands (orbitals).
        """
        return self.original_shape[0]

    @property
    def num_orbital_dimensions(self) -> int:
        """
        Returns the number of orbital dimensions; two (for a two-point object) or four
        (for a three-leg or four-leg vertex) are allowed.

        :return: The number of orbital axes (2 or 4).
        """
        return self._num_orbital_dimensions

    @property
    def num_wn_dimensions(self) -> int:
        """
        Returns the number of bosonic frequency dimensions; none or one are allowed.

        :return: The number of bosonic frequency axes (0 or 1).
        """
        return self._num_wn_dimensions

    @property
    def num_vn_dimensions(self) -> int:
        """
        Returns the number of fermionic frequency dimensions; none, one or two are allowed.

        :return: The number of fermionic frequency axes (0, 1 or 2).
        """
        return self._num_vn_dimensions

    @property
    def niw(self) -> int:
        """
        Returns the number of bosonic frequencies in the object.

        :return: The half-width of the bosonic frequency box (0 if there is no bosonic axis).
        """
        if self.num_wn_dimensions == 0:
            return 0
        axis = -(self.num_wn_dimensions + self.num_vn_dimensions)
        return self.original_shape[axis] // 2

    @property
    def niv(self) -> int:
        """
        Returns the number of fermionic frequencies in the object.

        :return: The half-width of the fermionic frequency box (0 if there is no fermionic axis).
        """
        if self.num_vn_dimensions == 0:
            return 0
        return self.original_shape[-1] // 2

    @property
    def full_niw_range(self) -> bool:
        r"""
        Specifies whether the object is stored in the full bosonic frequency range or
        only a subset of it (only :math:`\omega \geq 0`). All vertices fulfill a certain symmetry against the sign change
        of :math:`\omega\to-\omega`, which can be taken advantage of. By exploiting this symmetry it allows us to almost
        half their memory usage.

        :return: True if the bosonic axis spans the full (signed) range.
        """
        return self._full_niw_range

    @property
    def full_niv_range(self) -> bool:
        r"""
        Specifies whether the object is stored in the full fermionic frequency range or
        only a subset of it (only :math:`\nu\geq0`). Same reasoning as already discussed in `full_niw_range`.

        :return: True if the fermionic axes span the full (signed) range.
        """
        return self._full_niv_range

    def cut_niw(self, niw_cut: int, copy: bool = True):
        """
        Allows to place a cutoff on the number of bosonic frequencies of the object. Returns a copy if ``copy`` is
        True, otherwise modifies and returns ``self`` in place. If the requested cutoff is not smaller than the
        available range this is a no-op and returns ``self`` (not a copy) regardless of ``copy`` -- do not mutate the
        result expecting the original to be unaffected.

        :param niw_cut: Number of bosonic frequencies to keep (per sign).
        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: The cut object (a copy, ``self``, or ``self`` unchanged for the no-op case).
        :raises ValueError: If the object has no bosonic frequency dimension.
        """
        if self.num_wn_dimensions == 0:
            raise ValueError("Cannot cut bosonic frequencies if there are none.")

        if niw_cut > self.niw:
            return self

        if copy:
            obj = self._clone_without_mat()
            obj.mat = self.mat  # shared; the slice is copied below, so self.mat is left untouched
        else:
            obj = self

        niw_slice = slice(obj.niw - niw_cut, obj.niw + niw_cut + 1) if obj.full_niw_range else slice(0, niw_cut + 1)

        # ``.copy()`` so the trimmed parent array is actually released; a bare slice would keep it alive via the view.
        if obj.num_vn_dimensions == 2:
            obj.mat = obj.mat[..., niw_slice, :, :].copy()
        elif obj.num_vn_dimensions == 1:
            obj.mat = obj.mat[..., niw_slice, :].copy()
        else:  # obj.num_vn_dimensions == 0
            obj.mat = obj.mat[..., niw_slice].copy()

        obj.update_original_shape()
        return obj

    def cut_niv(self, niv_cut: int, copy: bool = True):
        """
        Allows to place a cutoff on the number of fermionic frequencies of the object. Returns a copy if ``copy`` is
        True, otherwise modifies and returns ``self`` in place. If the requested cutoff is not smaller than the
        available range this is a no-op and returns ``self`` (not a copy) regardless of ``copy`` -- do not mutate the
        result expecting the original to be unaffected.

        :param niv_cut: Number of fermionic frequencies to keep (per sign).
        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: The cut object (a copy, ``self``, or ``self`` unchanged for the no-op case).
        :raises ValueError: If the object has no fermionic frequency dimension.
        """
        if self.num_vn_dimensions == 0:
            raise ValueError("Cannot cut fermionic frequencies if there are none.")

        if niv_cut > self.niv:
            return self

        if copy:
            obj = self._clone_without_mat()
            obj.mat = self.mat  # shared; the slice is copied below, so self.mat is left untouched
        else:
            obj = self

        niv_slice = slice(obj.niv - niv_cut, obj.niv + niv_cut) if obj.full_niv_range else slice(0, niv_cut)

        # ``.copy()`` so the trimmed parent array is actually released; a bare slice would keep it alive via the view.
        if obj.num_vn_dimensions == 2:
            obj.mat = obj.mat[..., niv_slice, niv_slice].copy()
        elif obj.num_vn_dimensions == 1:
            obj.mat = obj.mat[..., niv_slice].copy()

        obj.update_original_shape()
        return obj

    def cut_niw_and_niv(self, niw_cut: int, niv_cut: int, copy: bool = True):
        """
        Allows to place a cutoff on the number of bosonic and fermionic frequencies of the object. Returns a copy if
        ``copy`` is True, otherwise modifies and returns ``self`` in place.

        :param niw_cut: Number of bosonic frequencies to keep (per sign).
        :param niv_cut: Number of fermionic frequencies to keep (per sign).
        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: The cut object; see :meth:`cut_niw` and :meth:`cut_niv`.
        """
        return self.cut_niw(niw_cut, copy).cut_niv(niv_cut, copy)

    def extend_vn_to_diagonal(self):
        """
        Extends an object [...,w,v] to [...,w,v,v] by making a diagonal from the last dimension if the number of fermionic
        frequency dimensions is one. Returns the original object.

        :return: ``self`` with two fermionic frequency axes (a no-op if it already has two).
        :raises ValueError: If the object has no fermionic frequency dimension.
        """
        if self.num_vn_dimensions == 0:
            raise ValueError("No fermionic frequency dimensions available for extension.")
        if self.num_vn_dimensions == 2:
            return self
        self.mat = np.einsum(
            "...i,ij->...ij", self.mat, np.eye(self.current_shape[-1], dtype=self.mat.dtype), optimize=True
        )
        self._num_vn_dimensions = 2
        self.update_original_shape()
        return self

    def take_vn_diagonal(self):
        """
        Compresses an object [...w,v,v] to [...,w,v] by taking the diagonal of the last two dimensions and returns the
        original object.

        :return: ``self`` with one fermionic frequency axis (a no-op if it already has one).
        :raises ValueError: If the object has no fermionic frequency dimension.
        """
        if self.num_vn_dimensions == 0:
            raise ValueError("No fermionic frequency dimensions available for compression.")
        if self.num_vn_dimensions == 1:
            return self
        # ``.copy()`` since ``diagonal`` returns a read-only view that also keeps the larger [...,v,v] parent alive.
        self.mat = self.mat.diagonal(axis1=-2, axis2=-1).copy()
        self._num_vn_dimensions = 1
        self.update_original_shape()
        return self

    def to_full_niw_range(self):
        """
        Converts the object to the full bosonic frequency range and returns the original object. For details, we refer
        to Eq. (2.39) and the associated text in Georg Rohringer's PhD thesis. This corresponds to time-reversal
        symmetry.

        :return: ``self`` over the full (signed) bosonic range (a no-op if there is no bosonic axis or it is already full).
        """
        if self.num_wn_dimensions == 0 or self.full_niw_range:
            return self

        niw_axis = -(self.num_wn_dimensions + self.num_vn_dimensions)
        freq_axes = tuple(range(-(self.num_wn_dimensions + self.num_vn_dimensions), 0))
        n = self.mat.shape[niw_axis]

        out_shape = list(self.mat.shape)
        out_shape[niw_axis] = n * 2 - 1  # w=0 appears once, not twice

        full_mat = np.empty(out_shape, dtype=self.mat.dtype)

        neg_slice = [slice(None)] * self.mat.ndim
        neg_slice[niw_axis] = slice(None, n - 1)

        pos_slice = [slice(None)] * self.mat.ndim
        pos_slice[niw_axis] = slice(n - 1, None)

        src_slice = [slice(None)] * self.mat.ndim
        src_slice[niw_axis] = slice(1, None)

        np.copyto(full_mat[tuple(pos_slice)], self.mat)
        np.copyto(full_mat[tuple(neg_slice)], np.flip(self.mat[tuple(src_slice)], axis=freq_axes))
        np.conj(full_mat[tuple(neg_slice)], out=full_mat[tuple(neg_slice)])

        self.mat = full_mat
        self.update_original_shape()
        self._full_niw_range = True
        return self

    def to_half_niw_range(self):
        r"""
        Converts the object to the half bosonic frequency range by taking
        :math:`F^{\omega\nu\nu'}_{abcd}\to F^{\omega\geq0;\nu\nu'}_{abcd}`. Returns the original object.

        :return: ``self`` over the half bosonic range (a no-op if there is no bosonic axis or it is already half).
        """
        if self.num_wn_dimensions == 0 or not self.full_niw_range:
            return self

        axis = -(self.num_wn_dimensions + self.num_vn_dimensions)
        ind = np.arange(self.current_shape[axis] // 2, self.current_shape[axis])
        self.mat = np.take(self.mat, ind, axis=axis)
        self.update_original_shape()
        self._full_niw_range = False
        return self

    def to_negative_niw_range(self):
        r"""
        Returns a **new** object holding the negative bosonic frequency block :math:`\omega = 0, -1, \ldots, -niw`
        (``niw + 1`` entries, :math:`\omega = 0` included for consistency with :meth:`to_half_niw_range`), derived from
        a half (positive) niw-range object via the time-reversal symmetry
        :math:`F^{-\omega,-\nu,-\nu'}_{abcd} = (F^{\omega\nu\nu'}_{abcd})^{*}`. The bosonic axis order is kept so that
        index ``i`` corresponds to :math:`\omega = -i` (index 0 is :math:`\omega = 0`); only the fermionic axes are
        flipped and the whole array is conjugated. This is the negative-frequency counterpart of
        :meth:`to_full_niw_range`, and it is its own inverse (applying it twice returns the original object).

        **Only allowed for an object already in the half (positive) bosonic frequency range** (``full_niw_range`` must
        be ``False`` and a bosonic frequency dimension must exist); ``self`` is left unchanged. Memory-lean: the
        result is a single freshly allocated array (``np.flip`` is a view, ``np.conj`` allocates only the output) plus
        the (array-less) metadata clone.

        :return: A new object holding the negative bosonic frequency block (``num_vn``/orbital/momentum layout unchanged).
        :raises ValueError: If the object has no bosonic frequency dimension, or is not in the half (positive) niw range.
        """
        if self.num_wn_dimensions == 0:
            raise ValueError("Cannot build the negative bosonic frequency range without a bosonic frequency dimension.")
        if self.full_niw_range:
            raise ValueError(
                "Flipping to negative niw range requires the object to be "
                "in the half (positive) bosonic frequency range."
            )

        result = self._clone_without_mat()
        if self.num_vn_dimensions == 0:
            result.mat = np.conj(self.mat)
        else:
            vn_axes = tuple(range(-self.num_vn_dimensions, 0))
            result.mat = np.conj(np.flip(self.mat, axis=vn_axes))
        result.update_original_shape()
        return result

    def to_half_niv_range(self):
        r"""
        Converts the object to the half fermionic frequency range by taking
        :math:`F^{\omega\nu\nu'}_{abcd}\to F^{\omega;\nu\geq0,\nu'\geq0}_{abcd}`. Returns the original object.

        :return: ``self`` over the half fermionic range (a no-op if there is no fermionic axis or it is already half).
        """
        if self.num_vn_dimensions == 0 or not self.full_niv_range:
            return self

        # ``.copy()`` so the discarded negative-frequency half is actually released; a bare slice would keep the full
        # parent array alive via the view.
        if self.num_vn_dimensions == 1:
            self.mat = self.mat[..., self.niv :].copy()
        if self.num_vn_dimensions == 2:
            self.mat = self.mat[..., self.niv :, self.niv :].copy()

        self.update_original_shape()
        self._full_niv_range = False
        return self

    def flip_frequency_axis(self, axis: tuple | int, copy: bool = True):
        """
        Flips the matrix along the specified frequency axis and returns a copy if specified.

        :param axis: The frequency axis or axes to flip (negative indices into the trailing frequency axes).
        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: The flipped object.
        :raises ValueError: If the object has no frequency axes, or ``axis`` is not a valid frequency axis.
        """
        if self.num_wn_dimensions + self.num_vn_dimensions == 0:
            raise ValueError("Cannot flip the matrix if there are no frequency dimensions.")

        if isinstance(axis, int):
            axis = (axis,)

        axis_possible = tuple(range(-self.num_wn_dimensions - self.num_vn_dimensions, 0))
        if not set(axis).issubset(axis_possible):
            raise ValueError(f"Invalid axis {axis}. Possible axes are {axis_possible}.")

        if copy:
            return deepcopy(self).flip_frequency_axis(axis, copy=False)

        self.mat = np.flip(self.mat, axis=axis)
        return self

    def swap_fermionic_frequency_axes(self, copy: bool = True):
        """
        Swaps two frequency axes of the matrix and returns a copy if specified.

        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: The object with its two fermionic frequency axes swapped.
        :raises ValueError: If the object has fewer than two fermionic frequency dimensions.
        """
        if self.num_vn_dimensions < 2:
            raise ValueError("Cannot swap axes if there are less than two fermionic frequency dimensions.")

        if copy:
            return deepcopy(self).swap_fermionic_frequency_axes(copy=False)

        self.mat = np.swapaxes(self.mat, -1, -2)
        return self

    def save(self, output_dir: str = "./", name: str = "please_give_me_a_name") -> None:
        """
        Saves the content of the matrix to a numpy file. Always saves it in half the niw range to save storage space.

        :param output_dir: Directory to write the ``.npy`` file to.
        :param name: File name (without extension).
        :return: None.
        """
        is_self_full_niw_range = self.full_niw_range
        np.save(os.path.join(output_dir, f"{name}.npy"), self.to_half_niw_range().mat, allow_pickle=False)
        if is_self_full_niw_range:
            self.to_full_niw_range()

    def _symmetrize_orbitals(self, orbitals: list | np.ndarray, orbital_axes: tuple):
        """
        Enforce orbital degeneracy inside given groups along specified orbital_axes.

        Each pattern is averaged independently:
            1) [i,i,i,i]
            2) [i,i,j,j]
            3) [i,j,j,i]
            4) [i,j,i,j]
            5) 3–1 patterns [i,j,j,j]

        :param orbitals: A group (list of 1-based orbital indices) or a list of such groups to make degenerate.
        :param orbital_axes: The axes of ``mat`` that carry the orbital indices (length 2 or 4).
        :return: ``self`` with the requested orbital groups averaged (in place).
        :raises ValueError: If an orbital index is out of range, a pattern length mismatches, or the number of
            orbital axes is neither 2 nor 4.
        """
        nb = self.current_shape[orbital_axes[0]]
        mat_orig = self.mat.copy()

        # Normalize input: single group -> list of lists
        if all(isinstance(x, int) for x in orbitals):
            orbitals = [orbitals]

        for group in orbitals:
            if len(group) <= 1:
                continue

            group = np.array(group)
            if np.any(group < 1) or np.any(group > nb):
                raise ValueError(f"Invalid orbitals {group}. Orbitals should be between 1 and {nb}.")

            group -= 1  # zero-based

            def average_patterns(patterns):
                """
                Averages the object over a set of equivalent orbital-index patterns and writes the average back to
                every member pattern (the in-place symmetrization step).

                :param patterns: List of orbital-index tuples (one entry per orbital axis) to average together.
                :return: None.
                :raises ValueError: If a pattern's length does not match the number of orbital axes.
                """
                if not patterns:
                    return

                indexers = []
                for pattern in patterns:
                    if len(pattern) != len(orbital_axes):
                        raise ValueError(
                            f"Pattern length {len(pattern)} does not match number of orbital axes {len(orbital_axes)}."
                        )
                    idx = [slice(None)] * mat_orig.ndim
                    for ax, val in zip(orbital_axes, pattern):
                        idx[ax] = val
                    indexers.append(tuple(idx))

                avg = sum(mat_orig[idx] for idx in indexers) / len(indexers)

                for idx in indexers:
                    self.mat[idx] = avg

            if len(orbital_axes) == 4:
                # 1) diagonals [i,i,i,i]
                patterns = [[i, i, i, i] for i in group]
                average_patterns(patterns)

                # 2) double-diagonal [i,i,j,j]
                patterns = [[i, i, j, j] for i in group for j in group if i != j]
                average_patterns(patterns)

                # 3) exchange pattern [i,j,j,i]
                patterns = [[i, j, j, i] for i in group for j in group if i != j]
                average_patterns(patterns)

                # 4) alternating pattern [i,j,i,j]
                patterns = [[i, j, i, j] for i in group for j in group if i != j]
                average_patterns(patterns)

                # 5) 3–1 patterns [i,j,j,j] and permutations
                patterns = set()
                for i in group:
                    for j in group:
                        if i == j:
                            continue
                        base = [i, j, j, j]
                        for perm in itertools.permutations(base):
                            patterns.add(tuple(perm))
                patterns = [list(p) for p in patterns]
                average_patterns(patterns)

            elif len(orbital_axes) == 2:
                # 1) diagonals [i,i]
                patterns = [[i, i] for i in group]
                average_patterns(patterns)

                # 2) exchange pattern [i,j]
                patterns = [[i, j] for i in group for j in group if i != j]
                average_patterns(patterns)
            else:
                raise ValueError("Invalid number of orbital axes. Only 2 or 4 are supported.")

        return self

    def _is_orbitally_symmetrized(self, orbitals: list | np.ndarray, orbital_axes: tuple) -> bool:
        """
        Check whether the tensor is orbitally symmetrized within given groups along specified orbital_axes.

        Verifies degeneracy of:

            1) [i,i,i,i]
            2) [i,i,j,j]
            3) [i,j,j,i]
            4) [i,j,i,j]
            5) 3–1 patterns [i,j,j,j]

        :param orbitals: A group (list of 1-based orbital indices) or a list of such groups to check.
        :param orbital_axes: The axes of ``mat`` that carry the orbital indices (length 2 or 4).
        :return: True if all required orbital patterns are already degenerate.
        :raises ValueError: If an orbital index is out of range, a pattern length mismatches, or the number of
            orbital axes is neither 2 nor 4.
        """
        nb = self.current_shape[orbital_axes[0]]

        if all(isinstance(x, int) for x in orbitals):
            orbitals = [orbitals]

        for group in orbitals:
            if len(group) <= 1:
                continue

            group = np.array(group)
            if np.any(group < 1) or np.any(group > nb):
                raise ValueError(f"Invalid orbitals {group}. Orbitals should be between 1 and {nb}.")

            group -= 1  # zero-based

            def check_patterns(patterns):
                """
                Checks whether the object already has identical values across a set of equivalent orbital-index
                patterns (the test used to short-circuit symmetrization).

                :param patterns: List of orbital-index tuples (one entry per orbital axis) to compare.
                :return: True if all patterns hold equal values (or the list is empty).
                :raises ValueError: If a pattern's length does not match the number of orbital axes.
                """
                if not patterns:
                    return True

                values = []
                for pattern in patterns:
                    if len(pattern) != len(orbital_axes):
                        raise ValueError(
                            f"Pattern length {len(pattern)} does not match number of orbital axes {len(orbital_axes)}."
                        )
                    idx = [slice(None)] * self.mat.ndim
                    for ax, val in zip(orbital_axes, pattern):
                        idx[ax] = val
                    values.append(self.mat[tuple(idx)])

                ref = values[0]
                return all(np.array_equal(v, ref) for v in values)

            if len(orbital_axes) == 4:
                # 1) diagonals [i,i,i,i]
                patterns = [[i, i, i, i] for i in group]
                if not check_patterns(patterns):
                    return False

                # 2) double-diagonal [i,i,j,j]
                patterns = [[i, i, j, j] for i in group for j in group if i != j]
                if not check_patterns(patterns):
                    return False

                # 3) exchange pattern [i,j,j,i]
                patterns = [[i, j, j, i] for i in group for j in group if i != j]
                if not check_patterns(patterns):
                    return False

                # 4) alternating pattern [i,j,i,j]
                patterns = [[i, j, i, j] for i in group for j in group if i != j]
                if not check_patterns(patterns):
                    return False

                # 5) 3–1 patterns [i,j,j,j] and permutations
                patterns = set()
                for i in group:
                    for j in group:
                        if i == j:
                            continue
                        base = [i, j, j, j]
                        for perm in itertools.permutations(base):
                            patterns.add(tuple(perm))
                patterns = [list(p) for p in patterns]
                if not check_patterns(patterns):
                    return False

            elif len(orbital_axes) == 2:
                # 1) diagonals [i,i]
                patterns = [[i, i] for i in group]
                if not check_patterns(patterns):
                    return False

                # 2) exchange pattern [i,j]
                patterns = [[i, j] for i in group for j in group if i != j]
                if not check_patterns(patterns):
                    return False
            else:
                raise ValueError("Invalid number of orbital axes. Only 2 or 4 are supported.")

        return True

    def _align_frequency_dimensions_for_operation(self, other: "LocalNPoint"):
        """
        Adapts the frequency dimensions of two (Local)NPoint objects to fit each other for addition or multiplication.

        :param other: The other object to align with ``self``.
        :return: A tuple ``(other, self_extended, other_extended)`` where the booleans record whether ``self`` or
            ``other`` had a fermionic axis diagonally extended (so the caller can revert it afterwards).
        """
        self_extended = False
        other_extended = False
        if self.num_vn_dimensions == 1 and other.num_vn_dimensions == 2:
            self.extend_vn_to_diagonal()
            self_extended = True
        if self.num_vn_dimensions == 2 and other.num_vn_dimensions == 1:
            other = other.extend_vn_to_diagonal()
            other_extended = True
        return other, self_extended, other_extended

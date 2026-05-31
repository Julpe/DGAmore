# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import itertools
import os

from dgamore.n_point_base import *


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
        """
        return self.original_shape[0]

    @property
    def num_orbital_dimensions(self) -> int:
        """
        Returns the number of orbital dimensions; two (for a two-point object) or four
        (for a three-leg or four-leg vertex) are allowed.
        """
        return self._num_orbital_dimensions

    @property
    def num_wn_dimensions(self) -> int:
        """
        Returns the number of bosonic frequency dimensions; none or one are allowed.
        """
        return self._num_wn_dimensions

    @property
    def num_vn_dimensions(self) -> int:
        """
        Returns the number of fermionic frequency dimensions; none, one or two are allowed.
        """
        return self._num_vn_dimensions

    @property
    def niw(self) -> int:
        """
        Returns the number of bosonic frequencies in the object.
        """
        if self.num_wn_dimensions == 0:
            return 0
        axis = -(self.num_wn_dimensions + self.num_vn_dimensions)
        return self.original_shape[axis] // 2

    @property
    def niv(self) -> int:
        """
        Returns the number of fermionic frequencies in the object.
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
        """
        return self._full_niw_range

    @property
    def full_niv_range(self) -> bool:
        r"""
        Specifies whether the object is stored in the full fermionic frequency range or
        only a subset of it (only :math:`\nu\geq0`). Same reasoning as already discussed in `full_niw_range`.
        """
        return self._full_niv_range

    def cut_niw(self, niw_cut: int):
        """
        Allows to place a cutoff on the number of bosonic frequencies of the object. Returns a copy of the object.
        """
        if self.num_wn_dimensions == 0:
            raise ValueError("Cannot cut bosonic frequencies if there are none.")

        if niw_cut > self.niw:
            return self

        copy = deepcopy(self)

        niw_slice = slice(copy.niw - niw_cut, copy.niw + niw_cut + 1) if copy.full_niw_range else slice(0, niw_cut + 1)

        if copy.num_vn_dimensions == 2:
            copy.mat = copy.mat[..., niw_slice, :, :]
        elif copy.num_vn_dimensions == 1:
            copy.mat = copy.mat[..., niw_slice, :]
        else:  # copy.num_vn_dimensions == 0
            copy.mat = copy.mat[..., niw_slice]

        copy.update_original_shape()
        return copy

    def cut_niv(self, niv_cut: int):
        """
        Allows to place a cutoff on the number of fermionic frequencies of the object. Returns a copy of the object.
        """
        if self.num_vn_dimensions == 0:
            raise ValueError("Cannot cut fermionic frequencies if there are none.")

        if niv_cut > self.niv:
            return self

        copy = deepcopy(self)

        niv_slice = slice(copy.niv - niv_cut, copy.niv + niv_cut) if copy.full_niv_range else slice(0, niv_cut)

        if copy.num_vn_dimensions == 2:
            copy.mat = copy.mat[..., niv_slice, niv_slice]
        elif copy.num_vn_dimensions == 1:
            copy.mat = copy.mat[..., niv_slice]

        copy.update_original_shape()
        return copy

    def cut_niw_and_niv(self, niw_cut: int, niv_cut: int):
        """
        Allows to place a cutoff on the number of bosonic and fermionic frequencies of the object. Returns a copy of
        the object.
        """
        return self.cut_niw(niw_cut).cut_niv(niv_cut)

    def extend_vn_to_diagonal(self):
        """
        Extends an object [...,w,v] to [...,w,v,v] by making a diagonal from the last dimension if the number of fermionic
        frequency dimensions is one. Returns the original object.
        """
        if self.num_vn_dimensions == 0:
            raise ValueError("No fermionic frequency dimensions available for extension.")
        if self.num_vn_dimensions == 2:
            return self
        self.mat = np.einsum("...i,ij->...ij", self.mat, np.eye(self.current_shape[-1]), optimize=True)
        self._num_vn_dimensions = 2
        self.update_original_shape()
        return self

    def take_vn_diagonal(self):
        """
        Compresses an object [...w,v,v] to [...,w,v] by taking the diagonal of the last two dimensions and returns the
        original object.
        """
        if self.num_vn_dimensions == 0:
            raise ValueError("No fermionic frequency dimensions available for compression.")
        if self.num_vn_dimensions == 1:
            return self
        self.mat = self.mat.diagonal(axis1=-2, axis2=-1)
        self._num_vn_dimensions = 1
        self.update_original_shape()
        return self

    def to_full_niw_range(self):
        """
        Converts the object to the full bosonic frequency range and returns the original object. For details, we refer
        to Eq. (2.39) and the associated text in Georg Rohringer's PhD thesis. This corresponds to time-reversal
        symmetry.
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
        """
        if self.num_wn_dimensions == 0 or not self.full_niw_range:
            return self

        axis = -(self.num_wn_dimensions + self.num_vn_dimensions)
        ind = np.arange(self.current_shape[axis] // 2, self.current_shape[axis])
        self.mat = np.take(self.mat, ind, axis=axis)
        self.update_original_shape()
        self._full_niw_range = False
        return self

    def to_half_niv_range(self):
        r"""
        Converts the object to the half fermionic frequency range by taking
        :math:`F^{\omega\nu\nu'}_{abcd}\to F^{\omega;\nu\geq0,\nu'\geq0}_{abcd}`. Returns the original object.
        """
        if self.num_vn_dimensions == 0 or not self.full_niv_range:
            return self

        if self.num_vn_dimensions == 1:
            self.mat = self.mat[..., self.niv :]
        if self.num_vn_dimensions == 2:
            self.mat = self.mat[..., self.niv :, self.niv :]

        self.update_original_shape()
        self._full_niv_range = False
        return self

    def flip_frequency_axis(self, axis: tuple | int, copy: bool = True):
        """
        Flips the matrix along the specified frequency axis and returns a copy if specified.
        """
        if self.num_wn_dimensions + self.num_vn_dimensions == 0:
            raise ValueError("Cannot flip the matrix if there are no frequency dimensions.")

        if isinstance(axis, int):
            axis = (axis,)

        axis_possible = tuple(range(-self.num_wn_dimensions - self.num_vn_dimensions, 0))
        if not set(axis).issubset(axis_possible):
            raise ValueError(f"Invalid axis {axis}. Possible axes are {axis_possible}.")

        if copy:
            copy = deepcopy(self)
            copy.mat = np.flip(copy.mat, axis=axis)
            return copy

        self.mat = np.flip(self.mat, axis=axis)
        return self

    def swap_fermionic_frequency_axes(self, copy: bool = True):
        """
        Swaps two frequency axes of the matrix and returns a copy if specified.
        """
        if self.num_vn_dimensions < 2:
            raise ValueError("Cannot swap axes if there are less than two fermionic frequency dimensions.")

        if copy:
            copy = deepcopy(self)
            copy.mat = np.swapaxes(copy.mat, -1, -2)
            return copy

        self.mat = np.swapaxes(self.mat, -1, -2)
        return self

    def save(self, output_dir: str = "./", name: str = "please_give_me_a_name") -> None:
        """
        Saves the content of the matrix to a numpy file. Always saves it in half the niw range to save storage space.
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

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Momentum-dependent four-point objects. :class:`FourPoint` extends :class:`LocalFourPoint` with one momentum axis
(see :class:`IAmNonLocal`) to represent quantities such as the ladder susceptibility
:math:`\chi_{1234}^{q\omega\nu\nu'}` and vertex :math:`F^{q}` over the (irreducible) BZ. The arithmetic and
compound-index machinery mirrors :class:`LocalFourPoint` but accounts for the extra momentum dimension; these
operations are the performance and memory bottleneck of the non-local ladder DGA step, so several variants are
provided that trade speed for footprint. Notation mirrors the thesis (Chapters 3 & 4).
"""

import gc

import numpy as np

from dgamore.brillouin_zone import KGrid
from dgamore.interaction import Interaction, LocalInteraction
from dgamore.local_four_point import LocalFourPoint
from dgamore.n_point_base import IAmNonLocal, SpinChannel, FrequencyNotation, DTYPE


class FourPoint(IAmNonLocal, LocalFourPoint):
    """
    This class is used to represent a non-local four-point object in a given channel with one momentum dimension,
    four orbital dimensions and variable bosonic and fermionic frequency dimensions. Calculations using these objects
    are the bottleneck of the DGA algorithm and need to be optimized for performance and memory usage.
    """

    def __init__(
        self,
        mat: np.ndarray,
        channel: SpinChannel = SpinChannel.NONE,
        nq: tuple[int, int, int] = (1, 1, 1),
        num_wn_dimensions: int = 1,
        num_vn_dimensions: int = 2,
        full_niw_range: bool = True,
        full_niv_range: bool = True,
        has_compressed_q_dimension: bool = False,
        frequency_notation: FrequencyNotation = FrequencyNotation.PH,
    ):
        r"""
        Initializes the momentum-dependent four-point object from an array and its channel/momentum/frequency metadata.

        :param mat: Underlying array with one momentum axis (compressed or not), four orbital axes, then frequency axes.
        :param channel: Spin channel of the object (see :class:`SpinChannel`).
        :param nq: Number of momenta per spatial direction.
        :param num_wn_dimensions: Number of bosonic frequency axes (0–1).
        :param num_vn_dimensions: Number of fermionic frequency axes (0–2).
        :param full_niw_range: Whether the object spans the full (signed) bosonic range or only :math:`\omega \geq 0`.
        :param full_niv_range: Whether the object spans the full (signed) fermionic range or only :math:`\nu \geq 0`.
        :param has_compressed_q_dimension: Whether the momentum is stored as a single compressed axis ``[q, ...]``
            (True) or as ``[qx, qy, qz, ...]`` (False).
        :param frequency_notation: Frequency convention (see :class:`FrequencyNotation`).
        """
        LocalFourPoint.__init__(
            self,
            mat,
            channel,
            num_wn_dimensions,
            num_vn_dimensions,
            full_niw_range,
            full_niv_range,
            frequency_notation,
        )
        IAmNonLocal.__init__(self, mat, nq, has_compressed_q_dimension)

    def __add__(self, other) -> "FourPoint":
        """
        Operator form of :meth:`add` (``A + B``). See :meth:`add`.
        """
        return self.add(other)

    def __radd__(self, other) -> "FourPoint":
        """
        Reflected operator form of :meth:`add` (``B + A``). See :meth:`add`.
        """
        return self.add(other)

    def __sub__(self, other) -> "FourPoint":
        """
        Operator form of :meth:`sub` (``A - B``). See :meth:`sub`.
        """
        return self.sub(other)

    def __rsub__(self, other) -> "FourPoint":
        """
        Reflected operator form of :meth:`sub` (``B - A``), returning ``-(A - B)``. See :meth:`sub`.
        """
        return self.sub(other).scale(-1.0)

    def __mul__(self, other) -> "FourPoint":
        """
        Operator form of :meth:`mul` (``A * B``). See :meth:`mul` for the (element-wise, not matrix) semantics.
        """
        return self.mul(other)

    def __rmul__(self, other) -> "FourPoint":
        """
        Reflected operator form of :meth:`mul` (``B * A``). See :meth:`mul`.
        """
        return self.mul(other)

    def __matmul__(self, other) -> "FourPoint":
        """
        Operator form of :meth:`matmul` with ``self`` on the left (``A @ B``). See :meth:`matmul`.
        """
        return self.matmul(other, left_hand_side=True)

    def __rmatmul__(self, other) -> "FourPoint":
        """
        Operator form of :meth:`matmul` with ``self`` on the right (``B @ A``). See :meth:`matmul`.
        """
        return self.matmul(other, left_hand_side=False)

    def __invert__(self):
        """
        Operator form of :meth:`invert` (``~A``). See :meth:`invert`.
        """
        return self.invert()

    def __pow__(self, power, modulo=None) -> "FourPoint":
        """
        Operator form of :meth:`pow` (``A ** n``). See :meth:`pow`.
        """
        return self.pow(power)

    def sum_over_vn(self, beta: float, axis: tuple = (-1,), copy: bool = True) -> "FourPoint":
        r"""
        Sums over the given fermionic frequency axes and applies the Matsubara prefactor :math:`1/\beta^{n}` (with
        :math:`n` the number of summed axes). The in-place branch frees the old array before allocating the result to
        cap peak memory.

        :param beta: Inverse temperature :math:`\beta`.
        :param axis: Fermionic axes to sum over (negative indices into the frequency tail).
        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: A :class:`FourPoint` with the summed axes removed and ``num_vn_dimensions`` reduced accordingly.
        :raises ValueError: If more axes are requested than the object has fermionic frequency dimensions.
        """
        if len(axis) > self.num_vn_dimensions:
            raise ValueError(f"Cannot sum over more fermionic axes than available in {self.current_shape}.")

        if not copy:
            summed = np.sum(self.mat, axis=axis)
            summed *= 1.0 / beta ** len(axis)  # in-place scale avoids a second full-size temporary
            self.mat = None
            gc.collect()
            self.mat = summed
            self._num_vn_dimensions -= len(axis)
            self.update_original_shape()
            return self

        copy_mat = np.sum(self.mat, axis=axis)
        copy_mat *= 1.0 / beta ** len(axis)  # in-place scale avoids a second full-size temporary

        self.update_original_shape()
        return FourPoint(
            copy_mat,
            self.channel,
            self.nq,
            self.num_wn_dimensions,
            self.num_vn_dimensions - len(axis),
            self.full_niw_range,
            self.full_niv_range,
            self.has_compressed_q_dimension,
            self.frequency_notation,
        )

    def sum_over_orbitals(self, orbital_contraction: str = "abcd->ad") -> "FourPoint":
        """
        Sums over orbital indices according to an einsum-style contraction on the four orbital axes (the leading
        momentum axis and trailing frequency axes are preserved automatically). Mutates ``self`` in place.

        :param orbital_contraction: Contraction of the form ``"abcd->..."`` whose target side is a subset of ``abcd``.
        :return: ``self``, with the summed orbital axes removed.
        :raises ValueError: If the left side does not have four orbitals or the right side has more indices than the left.
        """
        split = orbital_contraction.split("->")
        if len(split[0]) != 4 or len(split[1]) > len(split[0]):
            raise ValueError("Invalid orbital contraction.")

        permutation = (
            f"i{split[0]}...->i{split[1]}..."
            if self.has_compressed_q_dimension
            else f"ijk{split[0]}...->ijk{split[1]}..."
        )

        self.mat = np.einsum(permutation, self.mat)
        diff = len(split[0]) - len(split[1])
        self.update_original_shape()
        self._num_orbital_dimensions -= diff
        return self

    def to_compound_indices(self) -> "FourPoint":
        r"""
        Converts the indices of the FourPoint object :math:`F^{\omega\nu\nu'}_{1234}` to compound indices :math:`F^{\omega}_{c_1, c_2}`
        by transposing the object to [q, w, o1, o2, v, o4, o3, v'] (if the object has any fermionic frequency dimension,
        otherwise the compound indices are built from orbital dimensions only) and grouping {o1, o2, v} and {o4, o3, v'}
        to the new compound index. Always returns the object with a compressed momentum dimension and in the same niw
        range as the original object.

        :return: ``self`` with shape ``[q, w, c1, c2]`` (compound indices, compressed momentum).
        :raises NotImplementedError: If the frequency notation is neither ph nor pp.
        """
        if self.frequency_notation == FrequencyNotation.PH:
            return self._to_compound_indices_ph()
        elif self.frequency_notation == FrequencyNotation.PP:
            return self._to_compound_indices_pp()
        else:
            raise NotImplementedError(
                f"Frequency notation {self.frequency_notation} not supported for transformation to "
                f"compound indices."
            )

    def _to_compound_indices_ph(self) -> "FourPoint":
        """
        Converts the indices of the FourPoint object in the ph notation to compound indices (see
        :meth:`to_compound_indices`).

        :return: ``self`` in compound-index layout with compressed momentum.
        :raises ValueError: If the object has no bosonic frequency dimension but not exactly two fermionic ones.
        """
        if len(self.current_shape) == 4:  # [q, w, x1, x2]
            return self

        if not self.has_compressed_q_dimension:
            self.compress_q_dimension()

        self.update_original_shape()

        if (
            self.num_wn_dimensions == 0  # [q, o1, o2, o3, o4, v, vp]
        ):  # special case for objects without any bosonic frequency dimension (such as the pairing vertex)
            if self.num_vn_dimensions != 2:
                raise ValueError(
                    "Object must have 2 fermionic frequency dimensions if it does not have any w dimension."
                )
            self.mat = self.mat.reshape(self.nq_tot, self.n_bands**2 * 2 * self.niv, self.n_bands**2 * 2 * self.niv)
            return self

        w_dim = self.current_shape[5]
        if self.num_vn_dimensions == 0:  # [q, o1, o2, o3, o4, w]
            self.mat = self.mat.transpose(0, 5, 1, 2, 4, 3).reshape(
                self.nq_tot, w_dim, self.n_bands**2, self.n_bands**2
            )  # reshaping to [q,w,o1,o2,o4,o3] and then collecting {o1,o2} and {o4,o3} into two indices
            return self

        if self.num_vn_dimensions == 1:  # [q, o1, o2, o3, o4, w, v]
            self.extend_vn_to_diagonal()

        # [q, o1, o2, o3, o4, w, v, vp]
        self.mat = self.mat.transpose(0, 5, 1, 2, 6, 4, 3, 7).reshape(
            self.nq_tot, w_dim, self.n_bands**2 * 2 * self.niv, self.n_bands**2 * 2 * self.niv
        )  # reshaping to [q,w,o1,o2,v,o4,o3,vp] and then collecting {o1,o2,v} and {o4,o3,vp} into two indices

        return self

    def _to_compound_indices_pp(self) -> "FourPoint":
        """
        Converts the indices of the FourPoint object in the pp notation to compound indices. The difference to the ph
        case is that the orbital indices are permuted first, since the ordering of pp quantities is not "1234" but
        rather "1324".

        :return: ``self`` in compound-index layout.
        """
        if len(self.current_shape) == 3 + self.num_wn_dimensions:  # [q, w, x1, x2] or [q, x1, x2]
            return self

        return self.permute_orbitals("abcd->acbd", copy=False)._to_compound_indices_ph()

    def to_full_indices(self, shape: tuple = None) -> "FourPoint":
        """
        Converts an object stored with compound indices to an object that has unraveled momentum,
        orbital and frequency axes. Always returns the object with a compressed momentum dimension. This is the inverse
        transformation of :meth:`to_compound_indices`. Will make use of the ``original_shape`` the object was
        created or last modified with. If the ``original_shape`` is not set or is hard to obtain, the ``shape``
        argument can be used to specify the original shape of the object.

        :param shape: Optional override for the stored ``original_shape`` used to unravel the compound axes.
        :return: ``self`` with unraveled orbital and frequency axes (compressed momentum).
        :raises NotImplementedError: If the frequency notation is neither ph nor pp.
        """
        if self.frequency_notation == FrequencyNotation.PH:
            return self._to_full_indices_ph(shape)
        elif self.frequency_notation == FrequencyNotation.PP:
            return self._to_full_indices_pp(shape)
        else:
            raise NotImplementedError(
                f"Frequency notation {self.frequency_notation} not supported for transformation to full indices."
            )

    def _to_full_indices_ph(self, shape: tuple = None) -> "FourPoint":
        """
        Converts the indices of the FourPoint object in the ph notation back to full indices (see
        :meth:`to_full_indices`).

        :param shape: Optional override for the stored ``original_shape``.
        :return: ``self`` with unraveled orbital and frequency axes.
        :raises ValueError: If the current shape is not a compound-index layout, or there is no bosonic frequency axis.
        """
        if (
            len(self.current_shape) == 1 + self.num_orbital_dimensions + self.num_wn_dimensions + self.num_vn_dimensions
            and self.has_compressed_q_dimension
        ):
            return self
        elif (
            len(self.current_shape) == 3 + self.num_orbital_dimensions + self.num_wn_dimensions + self.num_vn_dimensions
            and not self.has_compressed_q_dimension
        ):
            return self

        if (len(self.current_shape) != 4 and self.has_compressed_q_dimension) or (
            len(self.current_shape) != 6 and not self.has_compressed_q_dimension
        ):  # (q,w,x1,x2) or (qx,qy,qz,w,x1,x2)
            raise ValueError(f"Converting to full indices with shape {self.current_shape} not supported.")

        if self.num_wn_dimensions != 1:
            raise ValueError("Number of bosonic frequency dimensions must be 1.")

        self.original_shape = shape if shape is not None else self.original_shape
        w_dim = self.original_shape[5] if self.has_compressed_q_dimension else self.original_shape[7]

        if self.num_vn_dimensions == 0:  # original was [q,o1,o2,o4,o3,w]
            self.mat = self.mat.reshape(
                (self.nq_tot,) + (w_dim,) + (self.n_bands,) * self.num_orbital_dimensions
            ).transpose(0, 2, 3, 5, 4, 1)
            self._has_compressed_q_dimension = True
            return self

        compound_index_shape = (self.n_bands, self.n_bands, 2 * self.niv)

        # original was [q,o1,o2,o4,o3,w,v,v']
        self.mat = self.mat.reshape((self.nq_tot,) + (w_dim,) + compound_index_shape * 2).transpose(
            0, 2, 3, 6, 5, 1, 4, 7
        )

        if self.num_vn_dimensions == 1:  # original was [q,o1,o2,o4,o3,w,v]
            # ``.copy()`` since ``diagonal`` returns a read-only view that also keeps the larger parent alive.
            self.mat = self.mat.diagonal(axis1=-2, axis2=-1).copy()
        return self

    def _to_full_indices_pp(self, shape: tuple = None) -> "FourPoint":
        """
        Converts the indices of the FourPoint object in the pp notation back to full indices. The difference to the ph
        case is that the orbital indices are permuted back, since the ordering of pp quantities is not "1234" but
        rather "1324".

        :param shape: Optional override for the stored ``original_shape``.
        :return: ``self`` with unraveled orbital and frequency axes.
        """
        return self._to_full_indices_ph(shape).permute_orbitals("abcd->acbd", copy=False)

    def permute_orbitals(self, permutation: str = "abcd->abcd", copy: bool = True) -> "FourPoint":
        """
        Permutes the four orbital axes according to an einsum-style string (the momentum and frequency axes are kept
        fixed). Summing over orbitals is not allowed (both sides must list all four orbitals).

        :param permutation: A permutation of the form ``"abcd->..."`` using exactly the four orbital labels.
        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: The orbital-permuted :class:`FourPoint` (``self`` unchanged if the permutation is the identity).
        :raises ValueError: If the permutation is malformed or does not list all four orbitals on both sides.
        """
        split = permutation.split("->")
        if len(split) != 2 or len(split[0]) != 4 or len(split[1]) != 4:
            raise ValueError("Invalid permutation.")

        if split[0] == split[1]:
            return self

        if copy:
            return self.copy().permute_orbitals(permutation, copy=False)

        permutation = (
            f"i{split[0]}...->i{split[1]}..."
            if self.has_compressed_q_dimension
            else f"ijk{split[0]}...->ijk{split[1]}..."
        )
        self.mat = np.einsum(permutation, self.mat, optimize=True)
        return self

    def map_to_full_bz(self, grid: KGrid, nq: tuple = None):
        """
        Unfolds the object from the irreducible BZ to the full BZ using the grid's symmetry index map (see
        :meth:`IAmNonLocal._map_to_full_bz`), with four orbital dimensions.

        :param grid: The :class:`KGrid` providing the irreducible-to-full BZ index mapping.
        :param nq: Optional number of momenta per direction for the unfolded grid; defaults to the object's ``nq``.
        :return: ``self`` defined on the full BZ.
        """
        return self._map_to_full_bz(grid, 4, nq)

    def add(self, other, copy: bool = True) -> "FourPoint":
        """
        Adds ``other`` to this object (operator ``+``); see :meth:`_add` for the accepted operands and the niw-range
        handling.

        :param other: A :class:`FourPoint`, :class:`LocalFourPoint`, :class:`Interaction`, :class:`LocalInteraction`,
            numpy array, or number.
        :param copy: If True (default), return a new :class:`FourPoint`; if False, accumulate into ``self`` in place
            (only supported when ``other`` is a conforming :class:`FourPoint`, see :meth:`_add`).
        :return: A new :class:`FourPoint` holding the sum (or ``self`` when ``copy=False``).
        """
        return self._add(other, copy=copy)

    def _add(self, other, subtract: bool = False, copy: bool = True) -> "FourPoint":
        """
        Helper method that allows for addition of FourPoint objects and other FourPoint, LocalFourPoint, Interaction or
        LocalInteraction objects. Additions with numpy arrays, floats, ints or complex numbers are also supported.
        Depending on the number of frequency and momentum dimensions, the vertices have to be added slightly different.
        If the objects have different niw ranges, they will be converted to the half niw range before the addition.
        Objects will always be returned in the half niw range to save memory.

        :param other: A :class:`FourPoint`, :class:`LocalFourPoint`, :class:`Interaction`, :class:`LocalInteraction`,
            numpy array, or number. Local operands are broadcast over the momentum axis.
        :param subtract: If True, subtract ``other`` instead of adding it (used by :meth:`sub` to avoid a negated copy).
        :param copy: If True (default), return a new :class:`FourPoint`; if False, accumulate the result into
            ``self.mat`` in place and return ``self`` (no out-of-place result block). The in-place branch is
            supported between two :class:`FourPoint` objects whose fermionic frequency dimensions already match (it
            refuses to diagonally extend ``self``) - the self-energy-kernel accumulation case in
            :mod:`dgamore.nonlocal_sde` - and for (:class:`LocalInteraction`/:class:`Interaction`) operands, which
            broadcast-accumulate without a full-size result block (the BSE-matrix assembly case).
        :return: A new :class:`FourPoint` (in the half niw range for the vertex-vertex case), or ``self`` when
            ``copy=False``.
        :raises ValueError: If ``other`` has an unsupported type, or ``copy=False`` would have to diagonally extend
            ``self``.
        :raises NotImplementedError: If ``copy=False`` and ``other`` is not a :class:`FourPoint`.
        """
        if not isinstance(
            other, (FourPoint, LocalFourPoint, Interaction, LocalInteraction, np.ndarray, float, int, complex)
        ):
            raise ValueError(f"Operations '+/-' for {type(self)} and {type(other)} not supported.")

        op = np.subtract if subtract else np.add

        if not copy and not isinstance(other, (FourPoint, LocalFourPoint, Interaction, LocalInteraction)):
            raise NotImplementedError(
                "In-place addition/subtraction (copy=False) is only supported for (Local)FourPoint or "
                "(Local)Interaction operands."
            )

        if isinstance(other, (np.ndarray, float, int, complex)):
            return FourPoint(
                op(self.mat, other),
                self.channel,
                self.nq,
                self.num_wn_dimensions,
                self.num_vn_dimensions,
                self.full_niw_range,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                self.frequency_notation,
            )

        channel = self.channel if self.channel != SpinChannel.NONE else other.channel

        if isinstance(other, (Interaction, LocalInteraction)):
            self.compress_q_dimension()

            other_mat = other.mat[None, ...] if not isinstance(other, Interaction) else other.compress_q_dimension().mat
            other_mat = other_mat.reshape(other.mat.shape + (1,) * (self.num_wn_dimensions + self.num_vn_dimensions))
            if not copy:
                # broadcast-accumulate the (small) interaction into self in place: no full-size result block
                op(self.mat, other_mat, out=self.mat)
                return self
            return FourPoint(
                op(self.mat, other_mat),
                self.channel,
                self.nq,
                self.num_wn_dimensions,
                self.num_vn_dimensions,
                self.full_niw_range,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                self.frequency_notation,
            )

        self_full_niw_range = self.full_niw_range
        other_full_niw_range = other.full_niw_range

        self.to_half_niw_range()
        other = other.to_half_niw_range()

        if not isinstance(other, FourPoint):
            # if other is LocalFourPoint
            other, self_extended, other_extended = self._align_frequency_dimensions_for_operation(other)
            other_mat = other.mat[None, ...] if self.has_compressed_q_dimension else other.mat[None, None, None, ...]

            if not copy:
                if self_extended:
                    raise ValueError(
                        "In-place addition/subtraction (copy=False) cannot diagonally extend 'self'; both operands "
                        "must have the same number of fermionic frequency dimensions."
                    )
                # broadcast-accumulate the momentum-independent operand into self in place
                op(self.mat, other_mat, out=self.mat)
                self.channel = channel
                self._full_niw_range = False
                self.update_original_shape()
                if other_full_niw_range:
                    other = other.to_full_niw_range()
                self._revert_frequency_dimensions_after_operation(other, other_extended, False)
                return self

            result = FourPoint(
                op(self.mat, other_mat),
                channel,
                self.nq,
                self.num_wn_dimensions,
                max(self.num_vn_dimensions, other.num_vn_dimensions),
                False,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                self.frequency_notation,
            )

            if self_full_niw_range:
                self.to_full_niw_range()
            if other_full_niw_range:
                other = other.to_full_niw_range()

            other = self._revert_frequency_dimensions_after_operation(other, other_extended, self_extended)
            return result

        other = self._align_q_dimensions_for_operations(other)
        other, self_extended, other_extended = self._align_frequency_dimensions_for_operation(other)

        if not copy:
            if self_extended:
                raise ValueError(
                    "In-place addition/subtraction (copy=False) cannot diagonally extend 'self'; both operands "
                    "must have the same number of fermionic frequency dimensions."
                )
            # accumulate into self in place: no out-of-place result block (and the caller folds any scalar prefactor
            # via FourPoint.scale, so no negated/scaled copy of ``other`` either).
            op(self.mat, other.mat, out=self.mat)
            self.channel = channel
            self._full_niw_range = False
            self.update_original_shape()
            if other_full_niw_range:
                other = other.to_full_niw_range()
            self._revert_frequency_dimensions_after_operation(other, other_extended, False)
            return self

        result = FourPoint(
            op(self.mat, other.mat),
            channel,
            self.nq,
            self.num_wn_dimensions,
            self.num_vn_dimensions,
            False,
            self.full_niv_range,
            self.has_compressed_q_dimension,
            self.frequency_notation,
        )

        if self_full_niw_range:
            self.to_full_niw_range()
        if other_full_niw_range:
            other = other.to_full_niw_range()

        other = self._revert_frequency_dimensions_after_operation(other, other_extended, self_extended)
        return result

    def sub(self, other, copy: bool = True) -> "FourPoint":
        """
        Helper method that allows for subtraction of FourPoint objects and other FourPoint, LocalFourPoint, Interaction or
        LocalInteraction objects. Subtractions with numpy arrays, floats, ints or complex numbers are also supported.
        Depending on the number of frequency and momentum dimensions, the vertices have to be subtracted slightly different.
        If the objects have different niw ranges, they will be converted to the half niw range before the subtraction.
        Objects will always be returned in the half niw range to save memory.

        :param other: A :class:`FourPoint`, :class:`LocalFourPoint`, :class:`Interaction`, :class:`LocalInteraction`,
            numpy array, or number.
        :param copy: If True (default), return a new :class:`FourPoint`; if False, subtract into ``self`` in place
            and return ``self`` (only supported when ``other`` is a conforming :class:`FourPoint`, see :meth:`_add`).
        :return: The difference, implemented as ``self._add(other, subtract=True)`` (see :meth:`_add`), or ``self``
            when ``copy=False``.
        :raises ValueError: Propagated from :meth:`_add` for unsupported operands.
        """
        return self._add(other, subtract=True, copy=copy)

    def mul(self, other) -> "FourPoint":
        r"""
        Allows for the multiplication with a number, a numpy array or another FourPoint object. This is different from
        the :meth:`matmul` method, which is used for matrix multiplication.
        In the case the other object is a FourPoint object, we require that both objects have only one fermionic
        frequency dimension, such that :math:`\sum_{ab} A_{12ab}^{q\nu} * B_{ba34}^{q\nu'} = C_{1234}^{q\nu\nu'}`. This is needed to
        construct the full vertex, see Eq. (3.139) in my thesis. Returns the object in the half niw range.

        :param other: A number, numpy array, or :class:`FourPoint`.
        :return: A new :class:`FourPoint` (in the half niw range for the four-point case).
        :raises ValueError: If ``other`` has an unsupported type, or either four-point operand does not have exactly
            one fermionic frequency dimension.
        """
        if not isinstance(other, (int, float, complex, np.ndarray, FourPoint)):
            raise ValueError("Multiplication only supported with numbers, numpy arrays or FourPoint objects.")

        if not isinstance(other, FourPoint):
            copy = self.copy()
            copy.mat *= other
            return copy

        if self.num_vn_dimensions != 1 or other.num_vn_dimensions != 1:
            raise ValueError("Both objects must have only one fermionic frequency dimension.")

        is_self_full_niw_range = self.full_niw_range
        is_other_full_niw_range = other.full_niw_range

        self.to_half_niw_range()
        other = other.to_half_niw_range()
        result_mat = self.times("qabcdwv,qdcefwp->qabefwvp", other)

        if is_self_full_niw_range:
            self.to_full_niw_range()
        if is_other_full_niw_range:
            other = other.to_full_niw_range()

        return FourPoint(result_mat, self.channel, self.nq, 1, 2, False, True, True, self.frequency_notation)

    def matmul(self, other, left_hand_side: bool = True) -> "FourPoint":
        """
        Helper method that allows for a matrix multiplication between FourPoint and FourPoint, LocalFourPoint,
        Interaction and LocalInteraction objects. Depending on the number of frequency and momentum dimensions,
        the objects have to be multiplied differently. The use of einsum is very crucial for memory efficiency here,
        as a regular matrix multiplication in compound index space would create large intermediate arrays if one of both
        partaking objects has less than two fermionic frequency dimensions. Result objects will always be returned in
        half of their niw range to save memory.

        :param other: A :class:`FourPoint`, :class:`LocalFourPoint`, :class:`Interaction`, or :class:`LocalInteraction`.
            Local operands are broadcast over the momentum axis.
        :param left_hand_side: If True, compute ``self @ other``; if False, compute ``other @ self``.
        :return: A new :class:`FourPoint` in the half bosonic frequency range, carrying the non-NONE channel and the
            :attr:`~dgamore.n_point_base.IHaveChannel.frequency_notation` of ``self``. All branches contract in
            the compound space of that notation (ph: rows {1,2,v}, cols {4,3,v'}; pp: rows {1,3,v}, cols
            {4,2,v'}; see :meth:`to_compound_indices`).
        :raises ValueError: If ``other`` has an unsupported type, or if the operands' frequency notations differ.
        """
        if not isinstance(other, (FourPoint, LocalFourPoint, Interaction, LocalInteraction)):
            raise ValueError(f"Multiplication {type(self)} @ {type(other)} not supported.")

        if isinstance(other, LocalFourPoint) and other.frequency_notation != self.frequency_notation:
            raise ValueError("Cannot multiply two objects with different frequency notations.")

        if isinstance(other, (LocalInteraction, Interaction)):
            is_local = not isinstance(other, Interaction)
            q_prefix = "" if is_local else "q"

            self.compress_q_dimension()

            left_orbs, right_orbs, final_orbs = (
                ("abij", "jief", "abef")
                if self.frequency_notation == FrequencyNotation.PH
                else ("afce", "ebfd", "abcd")
            )
            suffix = {0: "w", 1: "wv", 2: "wvp"}.get(self.num_vn_dimensions, "")
            einsum_str = (
                f"q{left_orbs}{suffix},{q_prefix}{right_orbs}->q{final_orbs}{suffix}"
                if left_hand_side
                else f"{q_prefix}{left_orbs},q{right_orbs}{suffix}->q{final_orbs}{suffix}"
            )

            return FourPoint(
                (
                    np.einsum(einsum_str, self.mat, other.mat, optimize=True)
                    if left_hand_side
                    else np.einsum(einsum_str, other.mat, self.mat, optimize=True)
                ),
                self.channel,
                self.nq,
                self.num_wn_dimensions,
                self.num_vn_dimensions,
                self.full_niw_range,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                self.frequency_notation,
            )

        is_local = not isinstance(other, FourPoint)
        channel = self.channel if self.channel != SpinChannel.NONE else other.channel

        if self.num_vn_dimensions in (0, 1) or other.num_vn_dimensions in (0, 1):
            # special cases if both objects do not have two fermionic frequency dimensions each. Straightforward
            # contraction is saving memory as we do not have to add fermionic frequency dimensions to artificially
            # create compound indices
            q_prefix = "" if is_local else "q"

            self.compress_q_dimension()
            if not is_local:
                other = other.compress_q_dimension()

            self.to_half_niw_range()
            other.to_half_niw_range()

            suffix_other, suffix_result, suffix_self = self._get_frequency_suffixes_for_matmul(other, left_hand_side)

            left_orbs, right_orbs, final_orbs = (
                ("abcd", "dcef", "abef")
                if self.frequency_notation == FrequencyNotation.PH
                else ("afce", "ebfd", "abcd")
            )
            einsum_str = (
                f"q{left_orbs}{suffix_self},{q_prefix}{right_orbs}{suffix_other}->q{final_orbs}{suffix_result}"
                if left_hand_side
                else f"{q_prefix}{left_orbs}{suffix_other},q{right_orbs}{suffix_self}->q{final_orbs}{suffix_result}"
            )

            return FourPoint(
                (
                    np.einsum(einsum_str, self.mat, other.mat, optimize=True)
                    if left_hand_side
                    else np.einsum(einsum_str, other.mat, self.mat, optimize=True)
                ),
                channel,
                self.nq,
                self.num_wn_dimensions,
                max(self.num_vn_dimensions, other.num_vn_dimensions),
                self.full_niw_range,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                self.frequency_notation,
            )

        is_self_full_niw_range = self.full_niw_range
        is_other_full_niw_range = other.full_niw_range

        self.to_half_niw_range().to_compound_indices()
        other = other.to_half_niw_range().to_compound_indices()
        # for __matmul__ self needs to be the LHS object, for __rmatmul__ self needs to be the RHS object
        new_mat = (
            np.matmul(self.mat, other.mat[None, ...] if is_local else other.mat)
            if left_hand_side
            else np.matmul(other.mat[None, ...] if is_local else other.mat, self.mat)
        )

        self.to_full_indices()
        if is_self_full_niw_range:
            self.to_full_niw_range()
        other = other.to_full_indices()
        if is_other_full_niw_range:
            other = other.to_full_niw_range()

        return FourPoint(
            new_mat,
            channel,
            self.nq,
            self.num_wn_dimensions,
            2,
            False,
            self.full_niv_range,
            self.has_compressed_q_dimension,
            self.frequency_notation,
        ).to_full_indices(self.original_shape)

    def invert(self, copy: bool = True):
        r"""
        Inverts the object in compound-index (matrix) space, per momentum. The single-fermionic-frequency case is
        handled by a dedicated block-diagonal reshape; otherwise each momentum slice is inverted in a loop to keep
        intermediate arrays small. The result is always returned in the half bosonic frequency range.

        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: The inverted :class:`FourPoint` in the half niw range.
        """

        if copy:
            return self.copy().invert(copy=False)

        self.to_half_niw_range()
        if self.num_vn_dimensions == 1:
            w_dim = self.original_shape[5] if self.has_compressed_q_dimension else self.original_shape[7]
            self.compress_q_dimension()
            self.mat = self.mat.transpose(0, 5, 6, 1, 2, 4, 3).reshape(
                (self.current_shape[0], w_dim, 2 * self.niv, self.n_bands**2, self.n_bands**2)
            )  # transpose to [q,w,v,o1,o2,o4,o3] and collecting [q,w,v,x1,x2]
            # invert in [x1,x2] per q (still batched over [w,v]); writing each q-slice in place caps the peak at the
            # input size instead of allocating the full batched-inverse output (~one extra full array).
            for i in range(self.current_shape[0]):
                self.mat[i] = np.linalg.inv(self.mat[i])
            # reshape to [q,w,v,o1,o2,o4,o3] and transpose to [q,o1,o2,o3,o4,w,v]
            self.mat = self.mat.reshape(
                (self.current_shape[0], w_dim, 2 * self.niv, self.n_bands, self.n_bands, self.n_bands, self.n_bands)
            ).transpose(0, 3, 4, 6, 5, 1, 2)
            return self

        self.compress_q_dimension()
        # full-index layout check: one (compressed) momentum axis + four orbital axes + the frequency axes
        full_index_ndim = 1 + 4 + self.num_wn_dimensions + self.num_vn_dimensions
        if self.num_wn_dimensions == 1 and self.num_vn_dimensions == 2 and len(self.current_shape) == full_index_ndim:
            # per-q compound round trip for the full-index two-fermion layout: the former global
            # to_compound_indices/to_full_indices pair materialized a second full-size layout copy; here only one
            # [w, x1, x2] workspace per q-slice is live besides the object
            if self.frequency_notation == FrequencyNotation.PP:
                self.permute_orbitals("abcd->acbd", copy=False)  # pure permutation, returns a view

            n = self.n_bands
            w_dim = self.current_shape[-3]
            size = n * n * 2 * self.niv
            for i in range(self.current_shape[0]):
                compound = self.mat[i].transpose(4, 0, 1, 5, 3, 2, 6).reshape(w_dim, size, size)
                self.mat[i] = (
                    np.linalg.inv(compound)
                    .reshape(w_dim, n, n, 2 * self.niv, n, n, 2 * self.niv)
                    .transpose(1, 2, 5, 4, 0, 3, 6)
                )

            if self.frequency_notation == FrequencyNotation.PP:
                self.permute_orbitals("abcd->acbd", copy=False)  # the pp pairing permute is self-inverse
            return self

        self.to_compound_indices()
        for i in range(self.current_shape[0]):
            self.mat[i] = np.linalg.inv(self.mat[i])
        return self.to_full_indices()

    def invert_and_sum_over_last_vn(self, beta: float):
        r"""
        Inverts the object in compound-index space per momentum and bosonic frequency, then sums over the last
        fermionic frequency axis (with the :math:`1/\beta` prefactor). This computes the auxiliary susceptibility
        used in the ladder construction in a single fused pass. Mutates ``self`` in place.

        :param beta: Inverse temperature :math:`\beta`.
        :return: ``self`` with the last fermionic axis summed out (``num_vn_dimensions`` reduced to 1).
        :raises NotImplementedError: If the object does not have exactly two fermionic frequency dimensions.
        """
        if self.num_vn_dimensions != 2:
            raise NotImplementedError("Method only implemented for objects with two fermionic frequency dimensions.")

        compound_index_shape = (self.n_bands, self.n_bands, 2 * self.niv)
        size = np.prod(compound_index_shape)

        self.to_half_niw_range().compress_q_dimension()
        w_dim = self.original_shape[5] if self.has_compressed_q_dimension else self.original_shape[7]

        new_arr = np.empty(self.original_shape[:-1], dtype=self.mat.dtype)
        for i in range(self.current_shape[0]):
            compound_arr = self.mat[i].transpose(4, 0, 1, 5, 3, 2, 6).reshape(w_dim, size, size)
            new_arr[i] = (
                np.linalg.inv(compound_arr).reshape((w_dim,) + compound_index_shape * 2).transpose(1, 2, 5, 4, 0, 3, 6)
            ).sum(axis=-1)
        new_arr /= beta  # in-place scale: new_arr is freshly allocated and unaliased, so no full-size temporary
        self.mat = new_arr
        self._num_vn_dimensions = 1
        self.update_original_shape()
        return self

    def invert_and_sum_over_last_vn_v2(self, beta: float):
        r"""
        Helper method that explicitly handles the calculation of the sum over the auxiliary susceptibility while
        being highly memory-efficient. Does not invert the matrix directly but uses a linear solver to avoid the
        creation of large intermediate arrays. This is especially important for objects with a large number of
        orbital degrees of freedom, where the matrix in compound index space can become very large. Is up to
        numerical precision the same as :meth:`invert_and_sum_over_last_vn`.

        :param beta: Inverse temperature :math:`\beta`.
        :return: ``self`` with the last fermionic axis summed out (``num_vn_dimensions`` reduced to 1).
        """
        o = self.n_bands
        vn = 2 * self.niv
        compound_size = o * o * vn

        self.to_half_niw_range().compress_q_dimension()
        w_dim = self.original_shape[5] if self.has_compressed_q_dimension else self.original_shape[7]

        new_arr = np.empty(self.original_shape[:-1], dtype=self.mat.dtype)

        idx = np.arange(compound_size)

        # decode flat index -> (o4,o3)
        idx_o4 = idx // (o * vn)
        idx_o3 = (idx // vn) % o

        rhs = np.zeros((compound_size, o * o), dtype=self.mat.dtype)
        rhs[idx, idx_o4 * o + idx_o3] = 1.0
        rhs = np.ascontiguousarray(rhs)

        rhs_batched = np.broadcast_to(rhs, (w_dim, compound_size, o * o))

        for i in range(self.current_shape[0]):
            compound_arr = np.ascontiguousarray(self.mat[i].transpose(4, 0, 1, 5, 3, 2, 6)).reshape(
                w_dim, compound_size, compound_size
            )
            new_arr[i] = (
                np.linalg.solve(compound_arr, rhs_batched).reshape((w_dim, o, o, vn, o, o)).transpose(1, 2, 5, 4, 0, 3)
            )

        new_arr /= beta  # in-place scale: new_arr is freshly allocated and unaliased, so no full-size temporary
        self.mat = new_arr
        self._num_vn_dimensions = 1
        self.update_original_shape()
        return self

    @staticmethod
    def load(
        filename: str,
        channel: SpinChannel = SpinChannel.NONE,
        nq: tuple[int, int, int] = (1, 1, 1),
        num_wn_dimensions: int = 1,
        num_vn_dimensions: int = 2,
        full_niw_range: bool = False,
        full_niv_range: bool = True,
        has_compressed_q_dimension: bool = True,
        frequency_notation: FrequencyNotation = FrequencyNotation.PH,
    ) -> "FourPoint":
        r"""
        Loads a :class:`FourPoint` from a ``.npy`` file.

        :param filename: Path to the ``.npy`` file (loaded with ``allow_pickle=False``).
        :param channel: Spin channel of the object (see :class:`SpinChannel`).
        :param nq: Number of momenta per spatial direction.
        :param num_wn_dimensions: Number of bosonic frequency axes (0–1).
        :param num_vn_dimensions: Number of fermionic frequency axes (0–2).
        :param full_niw_range: Whether the stored array spans the full (signed) bosonic range or only :math:`\omega \geq 0`.
        :param full_niv_range: Whether the stored array spans the full (signed) fermionic range or only :math:`\nu \geq 0`.
        :param has_compressed_q_dimension: Whether the momentum is stored as a single compressed axis (True) or as
            ``[qx, qy, qz, ...]`` (False).
        :param frequency_notation: Frequency convention (see :class:`FrequencyNotation`).
        :return: The loaded :class:`FourPoint`.
        """
        return FourPoint(
            np.load(filename, allow_pickle=False),
            channel,
            nq,
            num_wn_dimensions,
            num_vn_dimensions,
            full_niw_range,
            full_niv_range,
            has_compressed_q_dimension,
            frequency_notation,
        )

    @staticmethod
    def identity(
        n_bands: int,
        niw: int,
        niv: int,
        nq_tot: int = 1,
        nq: tuple[int, int, int] = (1, 1, 1),
        num_vn_dimensions: int = 2,
        frequency_notation: FrequencyNotation = FrequencyNotation.PH,
    ) -> "FourPoint":
        r"""
        Creates a :class:`FourPoint` that is the identity in compound-index (matrix) space at each momentum, returned
        in the half bosonic frequency range.

        :param n_bands: Number of orbitals/bands per orbital axis.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param nq_tot: Total number of momenta (product over directions).
        :param nq: Number of momenta per spatial direction.
        :param num_vn_dimensions: Number of fermionic frequency axes (1 or 2).
        :param frequency_notation: Frequency convention (see :class:`FrequencyNotation`).
        :return: The identity :class:`FourPoint` (compressed momentum, half niw range).
        :raises ValueError: If ``num_vn_dimensions`` is not 1 or 2.
        """
        if num_vn_dimensions not in (1, 2):
            raise ValueError("Invalid number of fermionic frequency dimensions.")

        full_shape = (nq_tot,) + (n_bands,) * 4 + (2 * niw + 1,) + (2 * niv,) * num_vn_dimensions

        if num_vn_dimensions == 1:
            mat = (
                np.tile(np.eye(n_bands**2, dtype=DTYPE)[None, ..., None, None], (nq_tot, 1, 1, 2 * niw + 1, 2 * niv))
                .reshape(full_shape)
                .swapaxes(3, 4)
            )
            # swapping the last two orbital axes to get [q,o1,o2,o4,o3,w,v] since the matrix is unity in
            # [o1,o2,o4,o3] (remember last two orbital indices are swapped in the compound index notation)
            return FourPoint(
                mat,
                nq=nq,
                num_vn_dimensions=num_vn_dimensions,
                has_compressed_q_dimension=True,
                frequency_notation=frequency_notation,
            ).to_half_niw_range()

        compound_index_size = 2 * niv * n_bands**2
        mat = np.tile(np.eye(compound_index_size, dtype=DTYPE)[None, None, ...], (nq_tot, 2 * niw + 1, 1, 1))

        return (
            FourPoint(
                mat,
                nq=nq,
                num_vn_dimensions=num_vn_dimensions,
                has_compressed_q_dimension=True,
                frequency_notation=frequency_notation,
            )
            .to_full_indices(full_shape)
            .to_half_niw_range()
        )

    @staticmethod
    def identity_like(other: "FourPoint") -> "FourPoint":
        """
        Creates a compound-index identity matching the bands, frequency box, momenta, fermionic-axis count and
        frequency notation of ``other`` (see :meth:`identity`).

        :param other: The :class:`FourPoint` whose shape/attributes the identity should match.
        :return: The matching identity :class:`FourPoint`.
        """
        return FourPoint.identity(
            other.n_bands,
            other.niw if other.full_niw_range else 2 * other.niw,
            other.niv,
            other.nq_tot,
            other.nq,
            other.num_vn_dimensions,
            other.frequency_notation,
        )

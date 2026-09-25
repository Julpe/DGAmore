# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Momentum-dependent four-point objects. :class:`FourPoint` extends :class:`LocalFourPoint` with one momentum axis (see
:class:`IAmNonLocal`) to represent quantities such as the ladder susceptibility :math:`\chi^{\mathrm{q}\nu\nu'}_{1234}`
and vertex :math:`F^{\mathrm{q}}` over the (irreducible) BZ. The arithmetic and compound-index machinery mirrors
:class:`LocalFourPoint` but accounts for the extra momentum dimension; these operations are the performance and memory
bottleneck of the non-local ladder DGA step, so several variants are provided that trade speed for footprint. Notation
mirrors the thesis (Chapters 3 & 4).
"""

import gc
import warnings

import numpy as np
import scipy as sp
from scipy.linalg import LinAlgWarning

from dgamore.brillouin_zone import KGrid
from dgamore.interaction import Interaction, LocalInteraction
from dgamore.local_four_point import LocalFourPoint
from dgamore.n_point_base import IAmNonLocal, SpinChannel, FrequencyNotation, DTYPE


def _is_complex_symmetric(buf: np.ndarray) -> bool:
    """
    Tells whether a square compound slice is complex-symmetric, i.e. ``max|M - M^T|`` lies within 4 machine epsilons
    of its dtype times ``max|M|``. Both maxima come from 256-wide tiles (each upper tile against its mirror), so no
    slice-sized temporary is made, and the test stops at the first failing tile.

    :param buf: The square slice.
    :return: Whether the slice counts as complex-symmetric.
    """
    cols = [slice(a, a + 256) for a in range(0, buf.shape[0], 256)]
    bound = 4 * np.finfo(buf.dtype).eps * np.max([np.abs(buf[:, c]).max() for c in cols])
    return all(np.abs(buf[r, c] - buf[c, r].T).max() <= bound for t, r in enumerate(cols) for c in cols[t:])


class FourPoint(IAmNonLocal, LocalFourPoint):
    """
    A non-local four-point object in a given channel, carrying one momentum dimension, four orbital dimensions and
    a variable number of bosonic and fermionic frequency dimensions. Calculations on these objects are the
    bottleneck of the DGA algorithm, so they have to stay fast and memory-lean.
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
        :param nq: Number of momenta per spatial direction ``(nx, ny, nz)``.
        :param num_wn_dimensions: Number of bosonic frequency axes (0 or 1).
        :param num_vn_dimensions: Number of fermionic frequency axes (0, 1 or 2).
        :param full_niw_range: Whether the object spans the full (signed) bosonic range or only :math:`\omega \geq 0`.
        :param full_niv_range: Whether the object spans the full (signed) fermionic range or only :math:`\nu \geq 0`.
        :param has_compressed_q_dimension: Whether the momentum is stored as a single compressed axis ``[q, ...]``
            (True) or as three separate axes ``[qx, qy, qz, ...]`` (False).
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

    def map_to_full_bz(self, grid: KGrid, nq: tuple = None, conjugate: bool = False):
        """
        Unfolds the object from the irreducible BZ to the full BZ using the grid's symmetry index map (see
        :meth:`IAmNonLocal._map_to_full_bz`), with four orbital dimensions.

        :param grid: The :class:`KGrid` providing the irreducible-to-full BZ index mapping.
        :param nq: Optional number of momenta per direction for the unfolded grid; defaults to the object's ``nq``.
        :param conjugate: Whether the object holds the complex conjugate of the quantity the grid's orbital rotations
            were discovered for; the rotation then runs with the conjugate unitaries.
        :return: ``self`` defined on the full BZ.
        """
        return self._map_to_full_bz(grid, 4, nq, conjugate)

    def add(self, other, copy: bool = True) -> "FourPoint":
        """
        Adds ``other`` to this object (operator ``+``); see :meth:`_add` for the accepted operands and the niw-range
        handling.

        :param other: A :class:`FourPoint`, :class:`LocalFourPoint`, :class:`Interaction`, :class:`LocalInteraction`,
            numpy array, or number.
        :param copy: If True (default), return a new :class:`FourPoint`; if False, accumulate into ``self`` in place and
            return ``self`` (only supported when ``other`` is a conforming :class:`FourPoint`, see :meth:`_add`).
        :return: A new :class:`FourPoint` holding the sum (or ``self`` when ``copy=False``).
        """
        return self._add(other, copy=copy)

    def _add(self, other, subtract: bool = False, copy: bool = True) -> "FourPoint":
        """
        Adds a FourPoint, LocalFourPoint, Interaction or LocalInteraction object (or a numpy array, float, int or
        complex number) to this one. How the vertices are added depends on their frequency and momentum dimensions.
        Operands with different niw ranges are first brought to the half niw range, and the result stays in the half
        niw range to save memory.

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
        Subtracts a FourPoint, LocalFourPoint, Interaction or LocalInteraction object (or a numpy array, float, int
        or complex number) from this one. How the vertices are subtracted depends on their frequency and momentum
        dimensions. Operands with different niw ranges are first brought to the half niw range, and the result stays
        in the half niw range to save memory.

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
        Multiplies the object by a scalar/array (element-wise) or by another :class:`FourPoint`. Note this is **not**
        a matrix product (see :meth:`matmul`): for two four-point operands, each with a single fermionic frequency,
        it forms :math:`\sum_{ab} A^{\mathrm{q}\nu}_{12ab} \, B^{\mathrm{q}\nu'}_{ba34} =
        C^{\mathrm{q}\nu\nu'}_{1234}`, contracting the inner orbitals while keeping both fermionic frequencies as
        separate axes. This product builds the full vertex, see Eq. (3.139) in my thesis. Returns the object in the
        half niw range.

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
        Matrix-multiplies this object with a FourPoint, LocalFourPoint, Interaction or LocalInteraction operand. How
        the product is wired depends on the frequency and momentum dimensions of the two operands. einsum is
        essential for memory here: a plain matrix multiplication in compound index space would build large
        intermediates whenever one of the operands has fewer than two fermionic frequency dimensions. The result
        comes back in half its niw range to save memory.

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
            # special case if either object lacks two fermionic frequency dimensions: straightforward contraction saves
            # memory (no need to add fermionic dimensions to artificially create compound indices)
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
            # per-q compound round trip for the full-index two-fermion layout: the global to_compound/to_full pair
            # materialized a second full-size copy; here only one [w, x1, x2] workspace per q-slice is live
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

    def contract_first_pair_with_local_vertex(self, f_loc: LocalFourPoint, niv_out: int) -> "FourPoint":
        r"""
        Contracts the first orbital pair and the fermionic frequency of this one-fermion bubble with the first
        orbital pair and the first fermionic frequency of a local two-fermion vertex,

        .. math:: T^{\mathrm{q}\nu'}_{1234} = \sum_{ab\nu} F^{\omega\nu\nu'}_{ab21}\,\chi^{\mathrm{q}\nu}_{0;ab34},

        as one matrix product per momentum, ``[w, (34), (ab nu)] @ [w, (ab nu), (21 nu')]``, and keeps only the
        window ``|nu'| < niv_out`` of the vertex's second frequency. The vertex operand is a view when the vertex
        array is stored in the axis order ``(4, 0, 1, 5, 2, 3, 6)``, i.e. ``[w, o1, o2, nu, o3, o4, nu']``, and a
        single copy otherwise; the per-momentum product never materializes the vertex again.

        :param f_loc: The local two-fermion vertex ``[o1, o2, o3, o4, w, nu, nu']`` (half niw range) whose stored
            first frequency is summed.
        :param niv_out: Number of positive fermionic frequencies kept of the vertex's second frequency.
        :return: The contraction as a new one-fermion :class:`FourPoint` (half niw range, compressed momentum axis,
            channel ``NONE``).
        """
        nq, nb, n_w = self.current_shape[0], self.n_bands, self.mat.shape[-2]
        window = f_loc.vn_slice(f_loc.niv_second, niv_out)
        f_pair = f_loc.mat.transpose(4, 0, 1, 5, 2, 3, 6).reshape(n_w, nb * nb * f_loc.mat.shape[-2], -1)
        out = np.empty((nq, nb, nb, nb, nb, n_w, 2 * niv_out), dtype=self.mat.dtype)
        for q in range(nq):
            bubble = self.mat[q].transpose(4, 2, 3, 0, 1, 5).reshape(n_w, nb * nb, -1)
            out[q] = (bubble @ f_pair).reshape(n_w, nb, nb, nb, nb, -1).transpose(4, 3, 1, 2, 0, 5)[..., window]
        return FourPoint(
            out, SpinChannel.NONE, self.nq, 1, 1, self.full_niw_range, True, has_compressed_q_dimension=True
        )

    def invert_and_sum_over_last_vn_v2(
        self, beta: float, inactive_pairs: np.ndarray | None = None, rhs: "FourPoint | None" = None
    ):
        r"""
        Computes the sum over the auxiliary susceptibility with a very small memory footprint. Rather than inverting
        the full compound matrix, each momentum and bosonic frequency slice is copied once (strided) into a reused
        Fortran-order buffer, factorized in place and only the :math:`o^2` right-hand sides that select the
        last-fermionic-frequency sum grouped by :math:`(o_4, o_3)` are back-substituted. The single compound slice
        held live per iteration keeps the peak footprint far below the full inverse, which matters most for a large
        number of orbital degrees of freedom, where the compound-index matrix becomes very large.

        A slice whose compound matrix is complex-symmetric, :math:`M_{1234}^{\nu\nu'} = M_{4321}^{\nu'\nu}` (the
        time-reversal property the :math:`\nu\nu'`-symmetrized vertex and a real dispersion give the Bethe-Salpeter
        matrix), is factorized with the complex-symmetric Bunch-Kaufman routine (``?sytrf``/``?sytrs``, half the
        flops of an LU); the decision is taken per slice from its own data, so it does not depend on the chunking.
        Any other slice takes the LU (``scipy.linalg.lu_factor``/``lu_solve``). Both agree with
        :meth:`invert_and_sum_over_last_vn` up to numerical precision.

        Orbital pairs without vertex (``inactive_pairs``, see :meth:`LocalFourPoint.orbital_pairs_without_vertex`)
        label compound rows and columns that couple only at equal :math:`\nu`, through the inverse bubble. They are
        eliminated one fermionic frequency at a time: the matrix over the other pairs, :math:`S = M_{aa} - M_{ai}
        M_{ii}^{-1} M_{ia}` with the correction on its frequency diagonal, is assembled and factorized as above, and
        the eliminated pairs follow by back-substitution. With half of the pairs eliminated the factorization costs an
        eighth. With no pair, or every pair, eliminated the full slice is solved as before.

        With ``rhs`` the same factorization solves the compound system for those right-hand sides instead of the
        :math:`\nu'`-sum selector, :math:`x = M^{-1} b` per slice, and the result is that solution without the
        :math:`1/\beta` of the sum.

        :param beta: Inverse temperature :math:`\beta`.
        :param inactive_pairs: Flat indices ``x * n_bands + y`` of the orbital pairs without vertex, or None.
        :param rhs: Right-hand sides in the layout of the result, ``[q, o1, o2, o3, o4, w, v]`` (compressed
            momenta, the bosonic range of the result), or None for the sum over the last fermionic frequency.
        :return: ``self`` with the last fermionic axis summed out (``num_vn_dimensions`` reduced to 1), or holding
            the solution for ``rhs`` in the same layout.
        """
        o = self.n_bands
        vn = 2 * self.niv
        compound_size = o * o * vn

        self.to_half_niw_range().compress_q_dimension()
        w_dim = self.original_shape[5] if self.has_compressed_q_dimension else self.original_shape[7]

        new_arr = np.empty(self.original_shape[:-1], dtype=self.mat.dtype)
        sytrf, sytrs, sytrf_lwork = sp.linalg.get_lapack_funcs(("sytrf", "sytrs", "sytrf_lwork"), dtype=new_arr.dtype)

        def factorize_and_solve(buf: np.ndarray, rhs: np.ndarray) -> np.ndarray:
            """Solves ``buf @ x = rhs`` in place of the Fortran-order ``buf``: Bunch-Kaufman if symmetric, else LU."""
            if _is_complex_symmetric(buf):
                lwork = int(sytrf_lwork(buf.shape[0], lower=1)[0].real)
                ldu, ipiv, info = sytrf(buf, lower=1, lwork=lwork, overwrite_a=1)
                if info > 0:
                    warnings.warn(f"Diagonal number {info} is exactly zero. Singular matrix.", LinAlgWarning)
                return sytrs(ldu, ipiv, rhs, lower=1)[0]
            lu_and_piv = sp.linalg.lu_factor(buf, overwrite_a=True, check_finite=False)
            return sp.linalg.lu_solve(lu_and_piv, rhs, check_finite=False)

        rhs_mat = None if rhs is None else rhs.to_half_niw_range().compress_q_dimension().mat

        def slice_rhs(i: int, w: int) -> np.ndarray | None:
            """The right-hand sides of slice ``(i, w)`` as the matrix ``[(o1, o2, v), (o4, o3)]``, or None."""
            if rhs_mat is None:
                return None
            b = rhs_mat[i][:, :, :, :, w, :].transpose(0, 1, 4, 3, 2).reshape(compound_size, o * o)
            return b.astype(self.mat.dtype, copy=False)

        inactive = np.asarray([] if inactive_pairs is None else inactive_pairs, dtype=int)
        if not 0 < inactive.size < o * o:
            idx = np.arange(compound_size)

            # decode flat compound column index (o4,o3,v') -> the (o4,o3) group it contributes its v' sum to
            idx_o4 = idx // (o * vn)
            idx_o3 = (idx // vn) % o

            selector = np.zeros((compound_size, o * o), dtype=self.mat.dtype)
            selector[idx, idx_o4 * o + idx_o3] = 1.0

            # one Fortran-order buffer, reused for every slice; fview addresses it as [(o1,o2,v), (o4,o3,v')] in the
            # block's own index order, so each slice is filled by a single strided copy
            fbuf = np.empty((compound_size, compound_size), dtype=self.mat.dtype, order="F")
            fview = fbuf.T.reshape(o, o, vn, o, o, vn).transpose(3, 4, 5, 0, 1, 2)

            def solve_slice(src: np.ndarray, b: np.ndarray | None) -> np.ndarray:
                """Solves one slice ``src`` ``[o1, o2, o3, o4, v, v']`` as a whole, for ``b`` or the selector."""
                np.copyto(fview, src.transpose(0, 1, 4, 3, 2, 5))
                return factorize_and_solve(fbuf, selector if b is None else b)

        else:
            active = np.setdiff1d(np.arange(o * o), inactive)
            ax, ay = np.divmod(active, o)
            ix, iy = np.divmod(inactive, o)
            n_act, n_ina, v = active.size, inactive.size, np.arange(vn)
            sbuf = np.empty((n_act * vn, n_act * vn), dtype=self.mat.dtype, order="F")
            # sview addresses sbuf as [a, v, a', v'] over the active pairs, the way fview addresses the whole slice
            sview = sbuf.T.reshape(n_act, vn, n_act, vn).transpose(2, 3, 0, 1)
            rhs_act = np.zeros((n_act, vn, o * o), dtype=self.mat.dtype)
            rhs_act[np.arange(n_act), :, active] = 1.0
            rhs_ina = np.zeros((vn, n_ina, o * o), dtype=self.mat.dtype)
            rhs_ina[:, np.arange(n_ina), inactive] = 1.0

            def solve_slice(src: np.ndarray, b: np.ndarray | None) -> np.ndarray:
                """Solves one slice ``src`` ``[o1, o2, o3, o4, v, v']`` by eliminating the inactive pairs per v, for
                ``b`` or the selector."""
                for r, (x, y) in enumerate(zip(ax, ay)):
                    for c, (xx, yy) in enumerate(zip(ax, ay)):
                        sview[r, :, c, :] = src[x, y, yy, xx]
                diag = np.diagonal(src, axis1=4, axis2=5)  # [o1, o2, o3, o4, v]: the equal-frequency couplings
                d_ii = diag[ix[:, None], iy[:, None], iy, ix].transpose(2, 0, 1)
                d_ia = diag[ix[:, None], iy[:, None], ay, ax].transpose(2, 0, 1)
                d_ai = diag[ax[:, None], ay[:, None], iy, ix].transpose(2, 0, 1)
                if b is None:
                    b_act, b_ina = rhs_act, rhs_ina
                else:
                    rows = b.reshape(o * o, vn, o * o)
                    b_act, b_ina = rows[active], rows[inactive].transpose(1, 0, 2)
                y_a = np.linalg.solve(d_ii, d_ia)
                y_r = np.linalg.solve(d_ii, b_ina)
                sview[:, v, :, v] -= d_ai @ y_a
                b_schur = (b_act - (d_ai @ y_r).transpose(1, 0, 2)).reshape(n_act * vn, o * o)
                x_act = factorize_and_solve(sbuf, b_schur).reshape(n_act, vn, o * o)
                solution = np.empty((o * o, vn, o * o), dtype=self.mat.dtype)
                solution[active] = x_act
                solution[inactive] = (y_r - y_a @ x_act.transpose(1, 0, 2)).transpose(1, 0, 2)
                return solution.reshape(compound_size, o * o)

        for i in range(self.current_shape[0]):
            block = self.mat[i]
            for w in range(w_dim):
                solution = solve_slice(block[:, :, :, :, w], slice_rhs(i, w))
                new_arr[i][:, :, :, :, w, :] = solution.reshape((o, o, vn, o, o)).transpose(0, 1, 4, 3, 2)

        if rhs_mat is None:
            new_arr /= beta  # in-place scale: new_arr is freshly allocated and unaliased, so no full-size temporary
        self.mat = new_arr
        self._num_vn_dimensions = 1
        self.update_original_shape()
        return self

    def invert_on_anti_diagonal(
        self, niv_band: int, w_start: int, beta: float, inactive_pairs: np.ndarray | None = None
    ) -> tuple["FourPoint", "FourPoint"]:
        r"""
        Inverts the compound matrix :math:`M` per momentum and bosonic frequency, but returns the inverse
        :math:`X = M^{-1}` only on the anti-diagonal :math:`\nu + \nu' = \omega` of the centered box of ``niv_band``
        positive fermionic frequencies (zero everywhere else), together with the sum over its FIRST fermionic
        frequency on that box,

        .. math:: R^{\omega\nu}_{12ab} = \frac{1}{\beta} \sum_{\nu'} X^{\omega\nu'\nu}_{12ab},

        where :math:`\nu'` runs over the whole box of the object. :math:`R` comes from a solve with :math:`M^{T}`, so
        it is exact also where :math:`M` is not complex-symmetric (the lattice bubble is symmetric only where
        :math:`G^{\mathrm{k}} = (G^{\mathrm{k}})^{T}`, which complex hoppings or a lattice without inversion
        break); on a complex-symmetric slice it equals the orbital-reversed sum over the last frequency.

        Per slice the compound matrix is copied once into a reused Fortran-order buffer in the storage precision. A
        complex-symmetric slice, :math:`M_{1234}^{\nu\nu'} = M_{4321}^{\nu'\nu}` within 4 machine epsilons, is solved
        with ``?sysv`` (Bunch-Kaufman with a level-3 solve) for only the band columns on one side of the
        anti-diagonal; the other side follows from :math:`X = X^{T}`. Any other slice is reduced to the box by the
        Schur complement :math:`S = M_{PP} - M_{PQ} M_{QQ}^{-1} M_{QP}`, with :math:`P` the box and :math:`Q` the
        remaining frequencies, and :math:`(M^{-1})_{PP} = S^{-1}` is solved with an LU for the band columns. Orbital
        pairs without vertex (``inactive_pairs``, see :meth:`LocalFourPoint.orbital_pairs_without_vertex`) couple only
        at equal :math:`\nu`: they are eliminated one fermionic frequency at a time as in
        :meth:`invert_and_sum_over_last_vn_v2`, the band of the reduced matrix is solved as above, and the blocks of
        the eliminated pairs follow from the equal-frequency couplings.

        :param niv_band: Number of positive fermionic frequencies of the centered box the band lives on.
        :param w_start: Bosonic index of the object's first frequency (the object holds ``w >= 0``).
        :param beta: Inverse temperature :math:`\beta`.
        :param inactive_pairs: Flat indices ``x * n_bands + y`` of the orbital pairs without vertex, or None.
        :return: The tuple ``(band, first_sum)``: :math:`X` on the anti-diagonal of the box as a two-fermion
            :class:`FourPoint` and :math:`R` on the box as a one-fermion :class:`FourPoint` (half niw range,
            compressed momenta).
        """
        o, vn, nb2 = self.n_bands, 2 * self.niv, 2 * niv_band
        off = self.niv - niv_band
        inner = np.arange(off, off + nb2)
        outer = np.setdiff1d(np.arange(vn), inner)

        self.to_half_niw_range().compress_q_dimension()
        n_q, w_dim = self.current_shape[0], self.current_shape[-3]
        dtype = self.mat.dtype
        band = np.zeros((n_q, o, o, o, o, w_dim, nb2, nb2), dtype=dtype)
        first_sum = np.empty((n_q, o, o, o, o, w_dim, nb2), dtype=dtype)

        inactive = np.asarray([] if inactive_pairs is None else inactive_pairs, dtype=int)
        eliminate = 0 < inactive.size < o * o
        pairs = np.setdiff1d(np.arange(o * o), inactive) if eliminate else np.arange(o * o)
        n_p = pairs.size
        size = n_p * vn
        buf = np.empty((size, size), dtype=dtype, order="F")
        # sums over the first frequency per orbital pair, in the positions (pair, v) of the system
        selector = np.zeros((n_p, vn, o * o), dtype=dtype)
        selector[np.arange(n_p), :, pairs] = 1.0
        i_p = (np.arange(n_p)[:, None] * vn + inner).reshape(-1)
        i_q = (np.arange(n_p)[:, None] * vn + outer).reshape(-1)
        sysv, sysv_lwork = sp.linalg.get_lapack_funcs(("sysv", "sysv_lwork"), dtype=dtype)
        lwork = int(sysv_lwork(size, lower=1)[0].real)

        if eliminate:
            ax, ay = np.divmod(pairs, o)
            ix, iy = np.divmod(inactive, o)
            v = np.arange(vn)
            ina_sum = np.zeros((vn, inactive.size, o * o), dtype=dtype)
            ina_sum[:, np.arange(inactive.size), inactive] = 1.0
            # bview addresses buf as [a, v, a', v'] over the active pairs
            bview = buf.T.reshape(n_p, vn, n_p, vn).transpose(2, 3, 0, 1)
        else:
            # bview addresses buf as [(o1, o2, v), (o4, o3, v')] in the slice's own index order
            bview = buf.T.reshape(o, o, vn, o, o, vn).transpose(3, 4, 5, 0, 1, 2)

        def solve_band(w_abs: int, rhs_sum: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """Returns the band blocks of ``buf^-1`` and ``buf^-T @ rhs_sum`` on the box; ``buf`` is overwritten."""
            v2 = np.arange(off + w_abs, off + nb2)
            v1 = vn - 1 + w_abs - v2
            if _is_complex_symmetric(buf):
                keep = v2 >= v1
                cols = (np.arange(n_p)[None, :] * vn + v2[keep][:, None]).reshape(-1)
                rhs = np.zeros((size, cols.size + o * o), dtype=dtype, order="F")
                rhs[cols, np.arange(cols.size)] = 1.0
                rhs[:, cols.size :] = rhs_sum.reshape(size, o * o)
                _, _, x, info = sysv(buf, rhs, lwork=lwork, lower=1, overwrite_a=1, overwrite_b=1)
                if info > 0:
                    warnings.warn(f"Diagonal number {info} is exactly zero. Singular matrix.", LinAlgWarning)
                n_keep = int(keep.sum())
                half = x[:, : cols.size].reshape(n_p, vn, n_keep, n_p)[:, v1[keep], np.arange(n_keep)]
                blocks = np.empty((v2.size, n_p, n_p), dtype=dtype)
                blocks[keep] = half.transpose(1, 0, 2)
                # the other side of the anti-diagonal follows from X = X^T: the mirror of (v1, v2) is row v1 - v2[0]
                blocks[~keep] = blocks[v1[~keep] - v2[0]].transpose(0, 2, 1)
                return blocks, v1, v2, x[:, cols.size :].reshape(n_p, vn, o * o)[:, inner]
            lu_qq = sp.linalg.lu_factor(np.asfortranarray(buf[np.ix_(i_q, i_q)]), overwrite_a=True, check_finite=False)
            w_qp = sp.linalg.lu_solve(lu_qq, buf[np.ix_(i_q, i_p)], check_finite=False)
            del lu_qq
            schur = np.asfortranarray(buf[np.ix_(i_p, i_p)] - buf[np.ix_(i_p, i_q)] @ w_qp)
            lu_s = sp.linalg.lu_factor(schur, overwrite_a=True, check_finite=False)
            cols = (np.arange(n_p)[None, :] * nb2 + (v2 - off)[:, None]).reshape(-1)
            rhs = np.zeros((n_p * nb2, cols.size), dtype=dtype, order="F")
            rhs[cols, np.arange(cols.size)] = 1.0
            x = sp.linalg.lu_solve(lu_s, rhs, check_finite=False).reshape(n_p, nb2, v2.size, n_p)
            blocks = x[:, v1 - off, np.arange(v2.size)].transpose(1, 0, 2)
            # (M^-T rhs)_P = S^-T (rhs_P - W^T rhs_Q) with W = M_QQ^-1 M_QP
            rhs_t = rhs_sum.reshape(size, o * o)
            rhs_t = rhs_t[i_p] - w_qp.T @ rhs_t[i_q]
            y_box = sp.linalg.lu_solve(lu_s, rhs_t, trans=1, check_finite=False).reshape(n_p, nb2, o * o)
            return blocks, v1, v2, y_box

        for i in range(n_q):
            for w in range(w_dim):
                src = self.mat[i, :, :, :, :, w]  # [o1, o2, o3, o4, v, v']
                if not eliminate:
                    np.copyto(bview, src.transpose(0, 1, 4, 3, 2, 5))
                    blocks, v1, v2, y_box = solve_band(w_start + w, selector)
                else:
                    for r, (x, y) in enumerate(zip(ax, ay)):
                        for c, (xx, yy) in enumerate(zip(ax, ay)):
                            bview[r, :, c, :] = src[x, y, yy, xx]
                    diag = np.diagonal(src, axis1=4, axis2=5)  # [o1, o2, o3, o4, v]: the equal-frequency couplings
                    d_ii = diag[ix[:, None], iy[:, None], iy, ix].transpose(2, 0, 1)
                    d_ia = diag[ix[:, None], iy[:, None], ay, ax].transpose(2, 0, 1)
                    d_ai = diag[ax[:, None], ay[:, None], iy, ix].transpose(2, 0, 1)
                    h_ia = np.linalg.solve(d_ii, d_ia)  # M_ii^-1 M_ia per v
                    bview[:, v, :, v] -= d_ai @ h_ia
                    # the first-frequency sums solve with M^T, whose blocks are M_aa^T, M_ia^T, M_ai^T and M_ii^T
                    d_ii_t = d_ii.transpose(0, 2, 1)
                    y_ina = np.linalg.solve(d_ii_t, ina_sum)
                    rhs_sum = selector - (d_ia.transpose(0, 2, 1) @ y_ina).transpose(1, 0, 2)
                    blocks_a, v1, v2, y_act = solve_band(w_start + w, rhs_sum)
                    g_t = np.linalg.solve(d_ii_t, d_ai.transpose(0, 2, 1))  # (M_ai M_ii^-1)^T per v
                    g_ai = g_t.transpose(0, 2, 1)
                    j = np.arange(v2.size)[:, None, None]
                    blocks = np.empty((v2.size, o * o, o * o), dtype=dtype)
                    blocks[j, pairs[:, None], pairs] = blocks_a
                    blocks[j, pairs[:, None], inactive] = -blocks_a @ g_ai[v2]
                    blocks[j, inactive[:, None], pairs] = -h_ia[v1] @ blocks_a
                    x_ii = h_ia[v1] @ blocks_a @ g_ai[v2]
                    same = v1 == v2
                    x_ii[same] += np.linalg.inv(d_ii[v1[same]])
                    blocks[j, inactive[:, None], inactive] = x_ii
                    y_box = np.empty((o * o, nb2, o * o), dtype=dtype)
                    y_box[pairs] = y_act
                    y_box[inactive] = (y_ina[inner] - g_t[inner] @ y_act.transpose(1, 0, 2)).transpose(1, 0, 2)
                # blocks [j, (o1 o2), (o4 o3)] -> band[o1, o2, o3, o4, v1_j - off, v2_j - off]
                band[i, :, :, :, :, w][..., v1 - off, v2 - off] = blocks.reshape(-1, o, o, o, o).transpose(
                    1, 2, 4, 3, 0
                )
                # y_box [(b a), v, (1 2)] -> first_sum[1, 2, a, b, v]
                first_sum[i, :, :, :, :, w] = y_box.reshape(o, o, nb2, o, o).transpose(3, 4, 1, 0, 2)

        first_sum /= beta
        meta = (self.channel, self.nq, 1)
        band_obj = FourPoint(band, *meta, 2, False, True, True, self.frequency_notation)
        return band_obj, FourPoint(first_sum, *meta, 1, False, True, True, self.frequency_notation)

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
        :param nq: Number of momenta per spatial direction ``(nx, ny, nz)``.
        :param num_wn_dimensions: Number of bosonic frequency axes (0 or 1).
        :param num_vn_dimensions: Number of fermionic frequency axes (0, 1 or 2).
        :param full_niw_range: Whether the object spans the full (signed) bosonic range or only :math:`\omega \geq 0`.
        :param full_niv_range: Whether the object spans the full (signed) fermionic range or only :math:`\nu \geq 0`.
        :param has_compressed_q_dimension: Whether the momentum is stored as a single compressed axis ``[q, ...]``
            (True) or as three separate axes ``[qx, qy, qz, ...]`` (False).
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
        :param nq: Number of momenta per spatial direction ``(nx, ny, nz)``.
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

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Local (momentum-independent) four-point objects. :class:`LocalFourPoint` wraps a single array carrying four
orbital indices and a variable number of bosonic/fermionic Matsubara axes, e.g. the generalized susceptibility
:math:`\chi_{1234}^{\omega\nu\nu'}`, the vertex :math:`F`, the irreducible vertex :math:`\Gamma`, and the
physical susceptibility :math:`\chi`. It adds channel-/notation-aware arithmetic (``+``, ``-``, ``*``, ``@``,
inversion, integer powers), orbital and frequency contractions, and conversions to/from the compound-index
matrix layout used for inversion and matrix products. Notation mirrors the thesis (Chapters 3 & 4).
"""

import numpy as np

from dgamore.interaction import LocalInteraction, Interaction
from dgamore.local_n_point import LocalNPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore.n_point_base import IHaveChannel, SpinChannel, FrequencyNotation, DTYPE


class LocalFourPoint(LocalNPoint, IHaveChannel):
    """
    This class is used to represent a local four-point object in a given channel with four orbital dimensions and a
    variable number of bosonic and fermionic frequency dimensions.
    """

    def __init__(
        self,
        mat: np.ndarray,
        channel: SpinChannel = SpinChannel.NONE,
        num_wn_dimensions: int = 1,
        num_vn_dimensions: int = 2,
        full_niw_range: bool = True,
        full_niv_range: bool = True,
        frequency_notation: FrequencyNotation = FrequencyNotation.PH,
    ):
        r"""
        Initializes the local four-point object from an array and its channel/notation/frequency metadata.

        :param mat: Underlying array with four leading orbital axes followed by the frequency axes.
        :param channel: Spin channel of the object (see :class:`SpinChannel`).
        :param num_wn_dimensions: Number of bosonic frequency axes (0 or 1).
        :param num_vn_dimensions: Number of fermionic frequency axes (0, 1 or 2).
        :param full_niw_range: Whether the object spans the full (signed) bosonic range or only :math:`\omega \geq 0`.
        :param full_niv_range: Whether the object spans the full (signed) fermionic range or only :math:`\nu \geq 0`.
        :param frequency_notation: Frequency convention (see :class:`FrequencyNotation`).
        """
        LocalNPoint.__init__(
            self,
            mat,
            4,
            num_wn_dimensions,
            num_vn_dimensions,
            full_niw_range,
            full_niv_range,
        )
        IHaveChannel.__init__(self, channel, frequency_notation)

    def __add__(self, other):
        """
        Operator form of :meth:`add` (``A + B``). See :meth:`add` for the accepted operands and semantics.
        """
        return self.add(other)

    def __radd__(self, other):
        """
        Reflected operator form of :meth:`add` (``B + A``). See :meth:`add`.
        """
        return self.add(other)

    def __sub__(self, other):
        """
        Operator form of :meth:`sub` (``A - B``). See :meth:`sub`.
        """
        return self.sub(other)

    def __rsub__(self, other):
        """
        Reflected operator form of :meth:`sub` (``B - A``), returning ``-(A - B)``. See :meth:`sub`.
        """
        return self.sub(other).scale(-1.0)

    def __mul__(self, other):
        """
        Operator form of :meth:`mul` (``A * B``). See :meth:`mul` for the (element-wise, not matrix) semantics.
        """
        return self.mul(other)

    def __rmul__(self, other) -> "LocalFourPoint":
        """
        Reflected operator form of :meth:`mul` (``B * A``). See :meth:`mul`.
        """
        return self.mul(other)

    def __matmul__(self, other) -> "LocalFourPoint":
        """
        Operator form of :meth:`matmul` with ``self`` on the left (``A @ B``). See :meth:`matmul`.
        """
        return self.matmul(other, left_hand_side=True)

    def __rmatmul__(self, other) -> "LocalFourPoint":
        """
        Operator form of :meth:`matmul` with ``self`` on the right (``B @ A``). See :meth:`matmul`.
        """
        return self.matmul(other, left_hand_side=False)

    def __invert__(self):
        """
        Operator form of :meth:`invert` (``~A``). See :meth:`invert`.
        """
        return self.invert()

    def __pow__(self, power, modulo=None):
        """
        Operator form of :meth:`pow` (``A ** n``). See :meth:`pow`.
        """
        return self.pow(power)

    def pow(self, power: int, identity=None):
        r"""
        Raises the object to an integer power using exponentiation by squaring (in compound-index matrix space of
        the object's :attr:`~dgamore.n_point_base.IHaveChannel.frequency_notation`). For :math:`n < 0` the inverse
        (see :meth:`invert`) is exponentiated, i.e. :math:`A^{-n} = (A^{-1})^{n}`.

        :param power: Integer exponent :math:`n`.
        :param identity: Identity object matching ``self`` (see :meth:`identity_like`), returned for :math:`n = 0`.
            Built via :meth:`identity_like` (carrying the frequency notation of ``self``) if omitted.
        :return: The exponentiated object.
        :raises ValueError: If ``power`` is not an integer or ``identity`` does not share the frequency notation
            of ``self``.
        """
        if not isinstance(power, int):
            raise ValueError("Only integer powers are supported.")

        if identity is not None and identity.frequency_notation != self.frequency_notation:
            raise ValueError("Identity must be in the same frequency notation as the object.")

        if power == 1:
            return self
        if power < 0:
            return self.invert() ** abs(power)

        if identity is None:
            identity = type(self).identity_like(self)
        if power == 0:
            return identity

        result = identity
        base = self.copy()

        # Exponentiation by squaring
        while power > 0:
            if power % 2 == 1:
                result @= base
            base @= base
            power //= 2

        return result

    def sum_over_orbitals(self, orbital_contraction: str = "abcd->ad"):
        """
        Sums over orbital indices according to an einsum-style contraction acting on the four orbital axes (the
        frequency axes are appended automatically). Mutates ``self`` in place and updates the orbital-dimension count.

        :param orbital_contraction: Contraction of the form ``"abcd->..."`` whose target side is a subset of
            ``abcd``, e.g. ``"abcd->ad"`` to trace over the inner orbitals.
        :return: ``self``, with the summed orbital axes removed.
        :raises ValueError: If the left side does not have four orbitals or the right side has more indices than the left.
        """
        split = orbital_contraction.split("->")
        if len(split[0]) != 4 or len(split[1]) > len(split[0]):
            raise ValueError("Invalid orbital contraction.")

        self.mat = np.einsum(f"{split[0]}...->{split[1]}...", self.mat)
        diff = len(split[0]) - len(split[1])
        self.update_original_shape()
        self._num_orbital_dimensions -= diff
        return self

    def sum_over_vn(self, beta: float, axis: tuple = (-1,)):
        r"""
        Sums over the given fermionic frequency axes and applies the Matsubara prefactor
        :math:`1/\beta^{n}`, where :math:`n` is the number of summed axes.

        :param beta: Inverse temperature :math:`\beta`.
        :param axis: Fermionic axes to sum over (negative indices into the frequency tail).
        :return: A new :class:`LocalFourPoint` with the summed axes removed and ``num_vn_dimensions`` reduced accordingly.
        :raises ValueError: If more axes are requested than the object has fermionic frequency dimensions.
        """
        if len(axis) > self.num_vn_dimensions:
            raise ValueError(f"Cannot sum over more fermionic axes than available in {self.current_shape}.")
        copy_mat = np.sum(self.mat, axis=axis)
        copy_mat *= 1.0 / beta ** len(axis)  # in-place scale avoids a second full-size temporary
        self.update_original_shape()
        return LocalFourPoint(
            copy_mat,
            self.channel,
            self.num_wn_dimensions,
            self.num_vn_dimensions - len(axis),
            self.full_niw_range,
            self.full_niv_range,
        )

    def sum_over_all_vn(self, beta: float):
        r"""
        Sums over all fermionic frequency axes and applies the prefactor :math:`1/\beta^{n}` (with :math:`n` the
        number of fermionic axes). A no-op (returns ``self``) if there are no fermionic frequency dimensions.

        :param beta: Inverse temperature :math:`\beta`.
        :return: A new :class:`LocalFourPoint` without fermionic frequency axes (or ``self`` if there were none).
        :raises ValueError: If the object has more than two fermionic frequency dimensions.
        """
        if self.num_vn_dimensions == 0:
            return self
        elif self.num_vn_dimensions == 1:
            axis = (-1,)
        elif self.num_vn_dimensions == 2:
            axis = (-2, -1)
        else:
            raise ValueError(f"Cannot sum over more fermionic axes than available in {self.current_shape}.")
        return self.sum_over_vn(beta, axis=axis)

    def contract_legs(self, beta: float):
        r"""
        Contracts the outer legs of a four-point vertex: sums over both fermionic frequency axes (with the
        :math:`1/\beta^2` prefactor) and traces the inner two orbitals (``abcd->ad``). Used e.g. to obtain the
        physical susceptibility :math:`\chi_{14}^{\omega}` from a generalized susceptibility.

        :param beta: Inverse temperature :math:`\beta`.
        :return: A new :class:`LocalFourPoint` with two orbital indices and no fermionic frequency axes.
        :raises ValueError: If the object does not have exactly two fermionic frequency dimensions.
        """
        if self.num_vn_dimensions != 2:
            raise ValueError("This method is only implemented for objects with 2 fermionic frequency dimensions.")
        return self.sum_over_all_vn(beta).sum_over_orbitals("abcd->ad")

    def to_compound_indices(self) -> "LocalFourPoint":
        r"""
        Converts the indices of the LocalFourPoint object :math:`F^{\omega\nu\nu'}_{1234}` to compound indices :math:`F^{\omega}_{c_1, c_2}`
        by transposing the object to [w, o1, o2, v, o4, o3, v'] (if the object has any fermionic frequency dimension,
        otherwise the compound indices are built from orbital dimensions only) and grouping {o1, o2, v} and {o4, o3, v'}
        to the new compound index. Always returns the object in the same niw range as the original object.

        :return: ``self`` with shape ``[w, c1, c2]`` (compound indices).
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

    def _to_compound_indices_ph(self) -> "LocalFourPoint":
        """
        Converts the indices of the LocalFourPoint object in the ph notation to compound indices (see
        :meth:`to_compound_indices`).

        :return: ``self`` in compound-index layout.
        :raises ValueError: If the object has no bosonic frequency dimension.
        """
        if len(self.current_shape) == 3:  # [w,x1,x2]
            return self

        if self.num_wn_dimensions != 1:
            raise ValueError(f"Cannot convert to compound indices if there are no bosonic frequencies.")

        self.update_original_shape()

        w_dim = self.original_shape[4]
        if self.num_vn_dimensions == 0:  # [o1,o2,o3,o4,w]
            self.mat = self.mat.transpose(4, 0, 1, 3, 2).reshape(w_dim, self.n_bands**2, self.n_bands**2)
            return self

        if self.num_vn_dimensions == 1:  # [o1,o2,o3,o4,w,v]
            self.extend_vn_to_diagonal()

        self.mat = self.mat.transpose(4, 0, 1, 5, 3, 2, 6).reshape(
            w_dim, self.n_bands**2 * 2 * self.niv, self.n_bands**2 * 2 * self.niv
        )
        return self

    def _to_compound_indices_pp(self) -> "LocalFourPoint":
        """
        Converts the indices of the LocalFourPoint object in the pp notation to compound indices. The difference to
        the ph case is that the orbital indices are permuted first, since the ordering of pp quantities is not
        "1234" but rather "1324".

        :return: ``self`` in compound-index layout.
        """
        if len(self.current_shape) == 2 + self.num_wn_dimensions:  # [w, x1, x2] or [x1, x2]
            return self

        return self.permute_orbitals("abcd->acbd", copy=False)._to_compound_indices_ph()

    def to_full_indices(self, shape: tuple = None) -> "LocalFourPoint":
        """
        Converts an object stored with compound indices to an object that has unraveled orbital
        and frequency axes. This is the inverse
        transformation of :meth:`to_compound_indices`. Will make use of the ``original_shape`` the object was
        created or last modified with. If the ``original_shape`` is not set or is hard to obtain, the ``shape``
        argument can be used to specify the original shape of the object.

        :param shape: Optional override for the stored ``original_shape`` used to unravel the compound axes.
        :return: ``self`` with unraveled orbital and frequency axes.
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

    def _to_full_indices_ph(self, shape: tuple = None) -> "LocalFourPoint":
        """
        Converts the indices of the LocalFourPoint object in the ph notation back to full indices (see
        :meth:`to_full_indices`).

        :param shape: Optional override for the stored ``original_shape``.
        :return: ``self`` with unraveled orbital and frequency axes.
        :raises ValueError: If the current shape is not a compound-index layout, or there is no bosonic frequency axis.
        """
        if len(self.current_shape) == self.num_orbital_dimensions + self.num_wn_dimensions + self.num_vn_dimensions:
            return self

        if len(self.current_shape) != 3:
            raise ValueError(f"Converting to full indices with shape {self.current_shape} not supported.")

        if self.num_wn_dimensions != 1:
            raise ValueError("Number of bosonic frequency dimensions must be 1.")

        self.original_shape = shape if shape is not None else self.original_shape
        w_dim = self.original_shape[4]

        if self.num_vn_dimensions == 0:  # original was [o1,o2,o4,o3,w]
            self.mat = self.mat.reshape((w_dim,) + (self.n_bands,) * self.num_orbital_dimensions).transpose(
                1, 2, 4, 3, 0
            )
            return self

        compound_index_shape = (self.n_bands, self.n_bands, 2 * self.niv)

        # original was [o1,o2,o4,o3,w,v,v']
        self.mat = self.mat.reshape((w_dim,) + compound_index_shape * 2).transpose(1, 2, 5, 4, 0, 3, 6)

        if self.num_vn_dimensions == 1:  # original was [o1,o2,o4,o3,w,v]
            # ``.copy()`` since ``diagonal`` returns a read-only view that also keeps the larger parent alive.
            self.mat = self.mat.diagonal(axis1=-2, axis2=-1).copy()
        return self

    def _to_full_indices_pp(self, shape: tuple = None) -> "LocalFourPoint":
        """
        Converts the indices of the LocalFourPoint object in the pp notation back to full indices. The difference to
        the ph case is that the orbital indices are permuted back, since the ordering of pp quantities is not
        "1234" but rather "1324".

        :param shape: Optional override for the stored ``original_shape``.
        :return: ``self`` with unraveled orbital and frequency axes.
        """
        return self._to_full_indices_ph(shape).permute_orbitals("abcd->acbd", copy=False)

    def invert(self, copy: bool = True):
        r"""
        Inverts the object in compound-index (matrix) space. The result is always returned in the half bosonic
        frequency range to save memory. A ph object with a **single** fermionic dimension is block-diagonal in
        :math:`\nu`, so it is inverted per :math:`(\omega, \nu)` orbital block and **keeps one fermionic
        dimension** (the inverse of a block-diagonal matrix is block-diagonal); the former route diagonally
        extended it and dense-inverted the whole compound matrix - :math:`(2 n_{\nu})^2` more flops and a full
        two-fermion block for the same information.

        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: The inverted :class:`LocalFourPoint` in the half niw range.
        """

        if copy:
            return self.copy().invert(copy=False)

        self.to_half_niw_range()
        if self.num_vn_dimensions == 1 and self.frequency_notation == FrequencyNotation.PH:
            n = self.n_bands
            w_dim, v_dim = self.current_shape[-2], self.current_shape[-1]
            # [o1, o2, o3, o4, w, v] -> [w, v, (o1, o2), (o4, o3)] (the compound pairing) and back
            mat = self.mat.transpose(4, 5, 0, 1, 3, 2).reshape(w_dim, v_dim, n * n, n * n)
            mat = np.linalg.inv(mat)
            self.mat = mat.reshape(w_dim, v_dim, n, n, n, n).transpose(2, 3, 5, 4, 0, 1)
            self.update_original_shape()
            return self

        self.to_compound_indices()
        self.mat = np.linalg.inv(self.mat)
        return self.to_full_indices()

    def get_core_from_shell_inversion(self, coupling: "LocalInteraction", niv_core: int = -1) -> "LocalFourPoint":
        r"""
        Returns the core fermionic-frequency box of
        :math:`\big(\mathrm{diag}_\nu[(X_{1234}^{\omega\nu})^{-1}] + \mathcal{C}_{1234}\big)^{-1}`, where ``self`` is
        :math:`X` (one fermionic axis, so the inverted first term is block-diagonal in :math:`\nu`) and
        :math:`\mathcal{C}` is a frequency-independent coupling, which connects every pair :math:`(\nu, \nu')`.

        The inversion runs over the whole fermionic box ``self`` carries while only the core box is returned, so the
        shell contributes exclusively through the frequency sum below. The compound inversion over that full box is
        avoided by the Woodbury identity: the added term is a rank-:math:`\mathrm{orbital}^2` update
        :math:`\mathcal{C}\otimes\mathbf{1}\mathbf{1}^T` on a block-diagonal matrix whose own inverse is :math:`X`
        itself, so :math:`X` is never inverted. With :math:`S = \sum_\nu X^{\omega\nu}` over the full box and

        .. math:: K = \mathcal{C}\,(\mathbb{1} + S\,\mathcal{C})^{-1}

        (the :math:`\mathcal{C}^{-1}`-free form, so a singular coupling such as a density-density interaction is
        admissible), the returned block reads
        :math:`\delta_{\nu\nu'} X^{\omega\nu}_{1234} - \sum_{abcd} X^{\omega\nu}_{12ab} K_{badc} X^{\omega\nu'}_{cd34}`.
        This replaces an :math:`O((n_\nu\,\mathrm{orbital}^2)^3)` dense inversion by an
        :math:`\mathrm{orbital}^2` inverse plus an :math:`O(n_{\nu,\mathrm{core}}^2)` contraction, and never
        materializes a full-box compound block. ``self`` is not modified.

        The caller supplies :math:`\mathcal{C}` already scaled; the shell-corrected irreducible vertex of the local
        Schwinger-Dyson step, for instance, passes :math:`\mathcal{U}/\beta^2`.

        :param coupling: The frequency-independent coupling :math:`\mathcal{C}` as a :class:`LocalInteraction`; its
            channel is carried over to the result.
        :param niv_core: Number of positive fermionic frequencies of the returned core box; the default of ``-1``
            (or any value not smaller than ``self.niv``) returns the whole box.
        :return: The inverse restricted to the core box, as a two-fermion :class:`LocalFourPoint` (half niw range).
        """
        o = self.n_bands
        o2 = o * o
        niv_core = self.niv if niv_core < 0 or niv_core > self.niv else niv_core

        # both are fresh objects (the clamp keeps cut_niv off its no-op branch), so reducing them in place leaves
        # ``self`` untouched; the summed one ends up in the same compound layout that invert() pairs by
        x_core = self.cut_niv(niv_core).to_half_niw_range()
        s_compound = self.sum_over_vn(1.0).to_half_niw_range().to_compound_indices()
        c_compound = coupling.permute_orbitals("abcd->abdc").mat.reshape(o2, o2)

        # per-(w, v) orbital blocks: the block-diagonal layout the compound API deliberately does not carry, since
        # its one-fermion form extends to the dense nu diagonal, which is exactly what this method avoids
        w_dim = x_core.current_shape[4]
        blocks = x_core.mat.transpose(4, 5, 0, 1, 3, 2).reshape(w_dim, 2 * niv_core, o2, o2)

        identity = np.eye(o2, dtype=self.mat.dtype)
        diag = np.arange(2 * niv_core)
        core = np.empty((w_dim, 2 * niv_core, 2 * niv_core, o2, o2), dtype=self.mat.dtype)
        for iw in range(w_dim):
            k = c_compound @ np.linalg.inv(identity + s_compound.mat[iw] @ c_compound)
            block = -np.einsum("vpq,qr,wrs->vwps", blocks[iw], k, blocks[iw], optimize=True)
            block[diag, diag] += blocks[iw]  # + delta_{v v'} X
            core[iw] = block

        core = core.reshape(w_dim, 2 * niv_core, 2 * niv_core, o, o, o, o).transpose(3, 4, 6, 5, 0, 1, 2)
        return LocalFourPoint(core, coupling.channel, 1, 2, False, self.full_niv_range, self.frequency_notation)

    def matmul(self, other, left_hand_side: bool = True) -> "LocalFourPoint":
        """
        Helper method that allows for a matrix multiplication between LocalFourPoint and LocalFourPoint and LocalInteraction
        objects. Depending on the number of frequency and momentum dimensions,
        the objects have to be multiplied differently. The use of einsum is very crucial for memory efficiency here,
        as a regular matrix multiplication in compound index space would create large intermediate arrays if one of both
        partaking objects has less than two fermionic frequency dimensions. Result objects will always be returned in
        half of their niw range to save memory.

        :param other: The right/left operand, a :class:`LocalFourPoint` or :class:`LocalInteraction`.
        :param left_hand_side: If True, compute ``self @ other``; if False, compute ``other @ self``.
        :return: A new :class:`LocalFourPoint` in the half bosonic frequency range, carrying the non-NONE channel
            and the :attr:`~dgamore.n_point_base.IHaveChannel.frequency_notation` of ``self``. All branches
            contract in the compound space of that notation (ph: rows {1,2,v}, cols {4,3,v'}; pp: rows {1,3,v},
            cols {4,2,v'}; see :meth:`to_compound_indices`).
        :raises ValueError: If ``other`` is neither a :class:`LocalFourPoint` nor a :class:`LocalInteraction`, or
            if the operands' frequency notations differ.
        """
        if not isinstance(other, (LocalFourPoint, LocalInteraction)):
            raise ValueError(f"Multiplication {type(self)} @ {type(other)} not supported.")

        if isinstance(other, LocalFourPoint) and other.frequency_notation != self.frequency_notation:
            raise ValueError("Cannot multiply two objects with different frequency notations.")

        if isinstance(other, LocalInteraction):
            left_orbs, right_orbs, final_orbs = (
                ("abij", "jief", "abef")
                if self.frequency_notation == FrequencyNotation.PH
                else ("afce", "ebfd", "abcd")
            )
            suffix = {0: "w", 1: "wv", 2: "wvp"}.get(self.num_vn_dimensions)
            einsum_str = (
                f"{left_orbs}{suffix},{right_orbs}->{final_orbs}{suffix}"
                if left_hand_side
                else f"{left_orbs},{right_orbs}{suffix}->{final_orbs}{suffix}"
            )
            return LocalFourPoint(
                (
                    np.einsum(einsum_str, self.mat, other.mat, optimize=True)
                    if left_hand_side
                    else np.einsum(einsum_str, other.mat, self.mat, optimize=True)
                ),
                self.channel,
                self.num_wn_dimensions,
                self.num_vn_dimensions,
                self.full_niw_range,
                self.full_niv_range,
                self.frequency_notation,
            )

        channel = self.channel if self.channel != SpinChannel.NONE else other.channel

        if self.num_vn_dimensions in (0, 1) or other.num_vn_dimensions in (0, 1):
            # special case if either object lacks two fermionic frequency dimensions: straightforward contraction saves
            # memory (no need to add fermionic dimensions to artificially create compound indices)
            self.to_half_niw_range()
            other.to_half_niw_range()

            suffix_other, suffix_result, suffix_self = self._get_frequency_suffixes_for_matmul(other, left_hand_side)

            left_orbs, right_orbs, final_orbs = (
                ("abcd", "dcef", "abef")
                if self.frequency_notation == FrequencyNotation.PH
                else ("afce", "ebfd", "abcd")
            )
            einsum_str = (
                f"{left_orbs}{suffix_self},{right_orbs}{suffix_other}->{final_orbs}{suffix_result}"
                if left_hand_side
                else f"{left_orbs}{suffix_other},{right_orbs}{suffix_self}->{final_orbs}{suffix_result}"
            )

            return LocalFourPoint(
                (
                    np.einsum(einsum_str, self.mat, other.mat, optimize=True)
                    if left_hand_side
                    else np.einsum(einsum_str, other.mat, self.mat, optimize=True)
                ),
                channel,
                self.num_wn_dimensions,
                max(self.num_vn_dimensions, other.num_vn_dimensions),
                self.full_niw_range,
                self.full_niv_range,
                self.frequency_notation,
            )

        self_full_niw_range = self.full_niw_range
        other_full_niw_range = other.full_niw_range

        # we do not use np.einsum here because it is faster to use np.matmul with compound indices instead of np.einsum
        self.to_half_niw_range().to_compound_indices()
        other = other.to_half_niw_range().to_compound_indices()
        # for __matmul__ self needs to be the LHS object, for __rmatmul__ self needs to be the RHS object
        new_mat = np.matmul(self.mat, other.mat) if left_hand_side else np.matmul(other.mat, self.mat)

        shape = (
            self.original_shape
            if self.num_vn_dimensions == max(self.num_vn_dimensions, other.num_vn_dimensions)
            else other.original_shape
        )

        self.to_full_indices()
        if self_full_niw_range:
            self.to_full_niw_range()
        other = other.to_full_indices()
        if other_full_niw_range:
            other = other.to_full_niw_range()

        return LocalFourPoint(
            new_mat,
            channel,
            self.num_wn_dimensions,
            max(self.num_vn_dimensions, other.num_vn_dimensions),
            full_niw_range=False,
            full_niv_range=self.full_niv_range,
            frequency_notation=self.frequency_notation,
        ).to_full_indices(shape)

    def mul(self, other):
        r"""
        Multiplies the object by a scalar/array (element-wise) or by another :class:`LocalFourPoint`. Note this is
        **not** a matrix product (see :meth:`matmul`): for two four-point operands, each with a single fermionic
        frequency, it forms :math:`\sum_{ab} A_{12ab}^{\omega\nu} \, B_{ba34}^{\omega\nu'} = C_{1234}^{\omega\nu\nu'}`,
        contracting the inner orbitals while keeping both fermionic frequencies as separate axes.

        :param other: A number, numpy array, or :class:`LocalFourPoint`.
        :return: A new :class:`LocalFourPoint` (in the half niw range for the four-point case).
        :raises ValueError: If ``other`` has an unsupported type, or either four-point operand does not have exactly
            one fermionic frequency dimension.
        """
        if not isinstance(other, (int, float, complex, np.ndarray, LocalFourPoint)):
            raise ValueError("Multiplication only supported with numbers, numpy arrays or LocalFourPoint objects.")

        if not isinstance(other, LocalFourPoint):
            copy = self.copy()
            copy.mat *= other
            return copy

        if self.num_vn_dimensions != 1 or other.num_vn_dimensions != 1:
            raise ValueError("Both objects must have only one fermionic frequency dimension.")

        is_self_full_niw_range = self.full_niw_range
        is_other_full_niw_range = other.full_niw_range

        self.to_half_niw_range()
        other = other.to_half_niw_range()
        result_mat = self.times("abcdwv,dcefwp->abefwvp", other)

        if is_self_full_niw_range:
            self.to_full_niw_range()
        if is_other_full_niw_range:
            other = other.to_full_niw_range()

        return LocalFourPoint(result_mat, self.channel, 1, 2, False, True, self.frequency_notation)

    def add(self, other, copy: bool = True):
        """
        Adds ``other`` to this object (operator ``+``); see :meth:`_add` for the accepted operands and the niw-range
        handling.

        :param other: A :class:`LocalFourPoint`, :class:`LocalInteraction`, numpy array, or number.
        :param copy: If True (default), return a new :class:`LocalFourPoint`; if False, accumulate into ``self`` in
            place and return ``self`` (only supported for conforming :class:`LocalFourPoint` and
            :class:`LocalInteraction` operands, see :meth:`_add`).
        :return: The sum (a new :class:`LocalFourPoint`, ``self`` when ``copy=False``, or a raw numpy array for a
            non-local :class:`Interaction`).
        """
        return self._add(other, copy=copy)

    def _add(self, other, subtract: bool = False, copy: bool = True):
        """
        Helper method that allows for addition of LocalFourPoint objects and other LocalFourPoint or LocalInteraction
        objects. Additions with numpy arrays, floats, ints or complex numbers are also supported.
        Depending on the number of frequency and momentum dimensions, the vertices have to be added slightly different.
        If the objects have different niw ranges, they will be converted to the half niw range before the addition.
        Objects will always be returned in the half niw range to save memory.

        :param other: A :class:`LocalFourPoint`, :class:`LocalInteraction`, numpy array, or number.
        :param subtract: If True, subtract ``other`` instead of adding it (used by :meth:`sub` to avoid a negated copy).
        :param copy: If True (default), return a new :class:`LocalFourPoint`; if False, accumulate the result into
            ``self.mat`` in place and return ``self`` (no out-of-place result block). The in-place branch is
            supported for a local :class:`LocalInteraction` operand (broadcast accumulate) and for a
            :class:`LocalFourPoint` operand whose bosonic and fermionic dimension counts already match ``self``
            (it refuses to diagonally extend either operand).
        :return: The sum. For a non-local :class:`Interaction` operand a raw numpy array is returned (to avoid a
            dependency on the :class:`FourPoint` class); otherwise a new :class:`LocalFourPoint` (or ``self`` when
            ``copy=False``).
        :raises ValueError: If ``other`` has an unsupported type, the bosonic-frequency counts do not match, or
            ``copy=False`` would have to diagonally extend an operand.
        :raises NotImplementedError: If ``copy=False`` is requested for a scalar/array/non-local operand.
        """
        if not isinstance(other, (LocalFourPoint, LocalInteraction, np.ndarray, float, int, complex)):
            raise ValueError(f"Operations '+/-' for {type(self)} and {type(other)} not supported.")

        if not copy and (not isinstance(other, (LocalFourPoint, LocalInteraction)) or isinstance(other, Interaction)):
            raise NotImplementedError(
                "In-place addition/subtraction (copy=False) is only supported for LocalFourPoint or "
                "LocalInteraction operands."
            )

        op = np.subtract if subtract else np.add

        if isinstance(other, (np.ndarray, float, int, complex)):
            return LocalFourPoint(
                op(self.mat, other),
                self.channel,
                self.num_wn_dimensions,
                self.num_vn_dimensions,
                self.full_niw_range,
                self.full_niv_range,
                self.frequency_notation,
            )

        if isinstance(other, LocalInteraction):
            other_mat = other.mat.reshape(other.mat.shape + (1,) * (self.num_wn_dimensions + self.num_vn_dimensions))
            is_local = not isinstance(other, Interaction)

            if not is_local:
                # since we do not want to have a dependency on the FourPoint class, we simply return a numpy array
                return (
                    op(self.mat[None, ...], other_mat)
                    if other.has_compressed_q_dimension
                    else op(self.mat[None, None, None, ...], other_mat)
                )

            if not copy:
                # broadcast-accumulate the (small) interaction into self in place: no full-size result block
                op(self.mat, other_mat, out=self.mat)
                return self
            return LocalFourPoint(
                op(self.mat, other_mat),
                self.channel,
                self.num_wn_dimensions,
                self.num_vn_dimensions,
                self.full_niw_range,
                self.full_niv_range,
                self.frequency_notation,
            )

        if self.num_wn_dimensions != other.num_wn_dimensions:
            raise ValueError("Number of bosonic frequency dimensions do not match.")

        self_full_niw_range = self.full_niw_range
        other_full_niw_range = other.full_niw_range

        self.to_half_niw_range()
        other = other.to_half_niw_range()

        channel = self.channel if self.channel != SpinChannel.NONE else other.channel

        if self.num_vn_dimensions == 0 or other.num_vn_dimensions == 0:
            if not copy:
                raise ValueError(
                    "In-place addition/subtraction (copy=False) is not supported when a fermionic axis has to be "
                    "broadcast (an operand without fermionic frequency dimensions)."
                )
            self_mat = self.mat
            other_mat = other.mat

            if self.num_vn_dimensions != other.num_vn_dimensions:
                if self.num_vn_dimensions == 0:
                    self_mat = np.expand_dims(self_mat, axis=-1 if other.num_vn_dimensions == 1 else (-1, -2))
                elif other.num_vn_dimensions == 0:
                    other_mat = np.expand_dims(other_mat, axis=-1 if self.num_vn_dimensions == 1 else (-1, -2))

            result = LocalFourPoint(
                op(self_mat, other_mat),
                self.channel,
                max(self.num_wn_dimensions, other.num_wn_dimensions),
                max(self.num_vn_dimensions, other.num_vn_dimensions),
                self.full_niw_range,
                self.full_niv_range,
                self.frequency_notation,
            )

            if self_full_niw_range:
                self.to_full_niw_range()
            if other_full_niw_range:
                other = other.to_full_niw_range()

            return result

        other, self_extended, other_extended = self._align_frequency_dimensions_for_operation(other)

        if not copy:
            if self_extended or other_extended:
                raise ValueError(
                    "In-place addition/subtraction (copy=False) cannot diagonally extend an operand; both must "
                    "have the same number of fermionic frequency dimensions."
                )
            op(self.mat, other.mat, out=self.mat)
            self.channel = channel
            if other_full_niw_range:
                other = other.to_full_niw_range()
            return self

        result = LocalFourPoint(
            op(self.mat, other.mat),
            channel,
            self.num_wn_dimensions,
            max(self.num_vn_dimensions, other.num_vn_dimensions),
            self.full_niw_range,
            self.full_niv_range,
            self.frequency_notation,
        )

        if self_full_niw_range:
            self.to_full_niw_range()
        if other_full_niw_range:
            other = other.to_full_niw_range()

        other = self._revert_frequency_dimensions_after_operation(other, other_extended, self_extended)
        return result

    def sub(self, other, copy: bool = True):
        """
        Helper method that allows for subtraction of LocalFourPoint objects and other LocalFourPoint or LocalInteraction
        objects. Subtractions with numpy arrays, floats, ints or complex numbers are also supported.
        Depending on the number of frequency and momentum dimensions, the vertices have to be subtracted slightly different.
        If the objects have different niw ranges, they will be converted to the half niw range before the subtraction.
        Objects will always be returned in the half niw range to save memory.

        :param other: A :class:`LocalFourPoint`, :class:`LocalInteraction`, numpy array, or number.
        :param copy: If True (default), return a new :class:`LocalFourPoint`; if False, subtract into ``self`` in place
            and return ``self`` (only supported for conforming :class:`LocalFourPoint` and :class:`LocalInteraction`
            operands, see :meth:`_add`).
        :return: The difference, implemented as ``self._add(other, subtract=True)`` (see :meth:`_add`).
        :raises ValueError: Propagated from :meth:`_add` for unsupported operands.
        """
        return self._add(other, subtract=True, copy=copy)

    def add_on_vn_diagonal(self, other: "LocalFourPoint", factor: float = 1.0) -> "LocalFourPoint":
        r"""
        Adds ``factor * other`` onto the fermionic frequency diagonal of ``self`` in place, i.e.
        :math:`A^{\omega\nu\nu'}_{1234} \mathrel{+}= f\, \delta_{\nu\nu'}\, B^{\omega\nu}_{1234}`. This is the
        memory-lean equivalent of ``self.add(other.extend_vn_to_diagonal().scale(factor))``: the diagonally extended
        block (by far the largest allocation of that pattern) is never materialized, only a 1-vn temporary for the
        scaled ``other``. A momentum-independent ``other`` broadcasts over the momentum axis of a non-local ``self``.
        Mutates and returns ``self``.

        :param other: The object holding the diagonal data, with exactly one fermionic frequency dimension.
        :param factor: Scalar prefactor applied to ``other`` (``self`` is not scaled).
        :return: ``self`` with the scaled diagonal added.
        :raises ValueError: If the fermionic dimension counts, bosonic ranges or fermionic box sizes do not match.
        """
        if self.num_vn_dimensions != 2 or other.num_vn_dimensions != 1:
            raise ValueError("Diagonal addition requires two fermionic dimensions on 'self' and one on 'other'.")
        if self.full_niw_range != other.full_niw_range:
            raise ValueError("Both objects must be in the same bosonic frequency range.")
        if self.current_shape[-1] != other.current_shape[-1]:
            raise ValueError("Fermionic frequency box sizes do not match.")

        diag = np.arange(self.current_shape[-1])
        self.mat[..., diag, diag] += other.mat if factor == 1.0 else factor * other.mat
        return self

    def permute_orbitals(self, permutation: str = "abcd->abcd", copy: bool = True) -> "LocalFourPoint":
        """
        Permutes the four orbital axes according to an einsum-style string. Summing over orbitals is not allowed
        (both sides must list all four orbitals); the frequency axes are kept fixed.

        :param permutation: A permutation of the form ``"abcd->..."`` using exactly the four orbital labels.
        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: The orbital-permuted :class:`LocalFourPoint`.
        :raises ValueError: If the permutation is malformed or does not list all four orbitals on both sides.
        """
        split = permutation.split("->")
        if (
            len(split) != 2
            or len(split[0]) != self.num_orbital_dimensions
            or len(split[1]) != self.num_orbital_dimensions
        ):
            raise ValueError("Invalid permutation.")

        if copy:
            return self.copy().permute_orbitals(permutation, copy=False)

        permutation = f"{split[0]}...->{split[1]}..."
        self.mat = np.einsum(permutation, self.mat, optimize=True)
        return self

    def symmetrize_orbitals(self, orbitals: list | np.ndarray) -> "LocalFourPoint":
        """
        Symmetrizes the object with respect to the given (equivalent) orbitals by averaging over all permutations of
        those orbitals applied to the four orbital axes. The orbital labels are 1-based, ranging from 1 to the number
        of bands; e.g. for a 3-band object ``orbitals=[1, 3]`` symmetrizes the first and third orbital.

        :param orbitals: 1-based orbital indices to treat as equivalent.
        :return: The symmetrized :class:`LocalFourPoint` (``self`` unchanged if already symmetrized).
        """
        if self.is_orbitally_symmetrized(orbitals):
            return self
        return self._symmetrize_orbitals(orbitals, (0, 1, 2, 3))

    def is_orbitally_symmetrized(self, orbitals: list | np.ndarray) -> bool:
        """
        Checks whether the object is already symmetric under all permutations of the given orbitals.

        :param orbitals: 1-based orbital indices to test for equivalence.
        :return: True if the object is invariant under permutations of those orbitals.
        """
        return self._is_orbitally_symmetrized(orbitals, (0, 1, 2, 3))

    def symmetrize_v_vp(self):
        r"""
        Symmetrizes the vertex with respect to :math:`(\nu, \nu')` via time-reversal symmetry:
        :math:`G_{1234}(\nu, \nu', \omega) = G_{4321}(\nu', \nu, \omega)`. Valid for TR-symmetric (real-hopping,
        paramagnetic) systems with SU(2) symmetry. Mutates ``self`` in place.

        :return: ``self``, symmetrized in :math:`(\nu, \nu')`.
        :raises ValueError: If the object does not have exactly two fermionic frequency dimensions.
        """
        if self.num_vn_dimensions != 2:
            raise ValueError("This method is only implemented for objects with 2 fermionic frequency dimensions.")

        # ``times`` (einsum) returns a view for the pure orbital permutation, avoiding the full-array deepcopy in
        # ``permute_orbitals(copy=True)``; the sum is then scaled in place to avoid a second full-size temporary.
        permuted = self.times("abcd...->dcba...")
        result = self.mat + np.swapaxes(permuted, -1, -2)
        result *= 0.5
        self.mat = result
        return self

    def change_frequency_notation_ph_to_pp_w0(self):
        r"""
        Converts the frequency notation from ph to pp at the bosonic frequency :math:`\omega' = 0`, returning a copy.
        The frequency map is :math:`(\omega, \nu_1, \nu_2) \to (\omega', \nu_1', \nu_2') = (\nu_1 + \nu_2 - \omega,
        \nu_1, \nu_2)`. A pp object is returned unchanged.

        :return: A new :class:`LocalFourPoint` in pp notation at :math:`\omega' = 0`.
        :raises ValueError: If the object does not have one bosonic and one or two fermionic frequency dimensions.
        """
        if self.num_wn_dimensions != 1 or self.num_vn_dimensions not in (1, 2):
            raise ValueError("Object must have 1 bosonic and 1 or 2 fermionic frequency dimensions.")

        copy = self.copy()

        if copy.frequency_notation == FrequencyNotation.PP:
            return copy

        # the w0 pp map samples only |w| <= 2*min(niw//2, niv) ph slices, so the bosonic axis is trimmed to that (even,
        # preserving the map's fermionic box) before to_full_niw_range doubles it (cut_niw's no-op guard misjudges here)
        w_axis = -(copy.num_vn_dimensions + 1)
        niw_stored = copy.current_shape[w_axis] // 2 if copy.full_niw_range else copy.current_shape[w_axis] - 1
        niw_window = min(niw_stored, 2 * min(niw_stored // 2, copy.niv))
        if niw_window < niw_stored:
            slicer = [slice(None)] * copy.mat.ndim
            slicer[w_axis] = (
                slice(niw_stored - niw_window, niw_stored + niw_window + 1)
                if copy.full_niw_range
                else slice(0, niw_window + 1)
            )
            copy.mat = copy.mat[tuple(slicer)].copy()
            copy.update_original_shape()

        copy = copy.to_full_niw_range()

        if copy.num_vn_dimensions == 1:
            copy = copy.extend_vn_to_diagonal()

        iw_pp, iv_pp, ivp_pp = MFHelper.get_frequencies_for_ph_to_pp_w0_channel_conversion(copy.niw, copy.niv)
        copy.mat = copy.mat[..., iw_pp, iv_pp, ivp_pp][..., None, :, :]
        copy.update_original_shape()
        return copy.set_frequency_notation(FrequencyNotation.PP)

    def create_wn_dimension(self):
        """
        Adds a single-point bosonic frequency axis (niw=0) immediately before the fermionic axes, returning a copy.

        :return: A new :class:`LocalFourPoint` with one bosonic frequency dimension.
        :raises ValueError: If the object already has a bosonic frequency dimension.
        """
        if self.num_wn_dimensions != 0:
            raise ValueError("Object already has bosonic frequency dimensions.")

        copy = self.copy()
        # insert bosonic axis immediately before the last vn axes
        copy.mat = np.expand_dims(copy.mat, axis=-(copy.num_vn_dimensions + 1))
        copy._num_wn_dimensions = 1
        copy.update_original_shape()
        return copy

    def take_first_wn(self):
        """
        Selects the first entry along the bosonic frequency axis, removing that axis, and returns a copy.

        :return: A new :class:`LocalFourPoint` without a bosonic frequency dimension.
        :raises ValueError: If the object does not have exactly one bosonic frequency dimension.
        """
        if self.num_wn_dimensions != 1:
            raise ValueError("Object must have exactly one bosonic frequency dimension.")

        # ``np.take`` allocates a fresh array, so clone the metadata without duplicating ``mat`` first.
        copy = self._clone_without_mat()
        # select first entry of wn
        copy.mat = np.take(self.mat, 0, axis=-(self.num_vn_dimensions + 1))
        copy._num_wn_dimensions = 0
        copy.update_original_shape()
        return copy

    def pad_with_u(self, u: LocalInteraction, niv_pad: int):
        r"""
        Extends the fermionic frequency range to ``niv_pad`` by filling the shell (outside the core region) with the
        bare interaction ``u``, returning a copy. Mainly used to add the constant asymptotic value :math:`U` to the
        irreducible vertex :math:`\Gamma` beyond the core frequency box. A no-op (returns a copy) if
        ``niv_pad <= self.niv``.

        :param u: The bare local interaction to fill the frequency shell with.
        :param niv_pad: Target number of positive fermionic frequencies.
        :return: A new :class:`LocalFourPoint` padded to ``niv_pad``.
        """
        if niv_pad <= self.niv:
            return self.copy()

        copy = self._clone_without_mat()
        # Allocate the padded array once and broadcast-fill the shell with ``u`` instead of materializing a full
        # tiled copy; then write the core region from ``self.mat`` (which is left untouched).
        urange_mat = np.empty(self.current_shape[:4] + (2 * self.niw + 1, 2 * niv_pad, 2 * niv_pad), dtype=DTYPE)
        urange_mat[...] = u.mat[..., None, None, None]
        niv_diff = niv_pad - self.niv
        urange_mat[..., niv_diff : niv_diff + 2 * self.niv, niv_diff : niv_diff + 2 * self.niv] = self.mat
        copy.mat = urange_mat
        copy.update_original_shape()
        return copy

    @staticmethod
    def load(
        filename: str,
        channel: SpinChannel = SpinChannel.NONE,
        num_wn_dimensions: int = 1,
        num_vn_dimensions: int = 2,
        full_niw_range: bool = False,
        full_niv_range: bool = True,
        frequency_notation: FrequencyNotation = FrequencyNotation.PH,
    ) -> "LocalFourPoint":
        r"""
        Loads a :class:`LocalFourPoint` from a ``.npy`` file.

        :param filename: Path to the ``.npy`` file (loaded with ``allow_pickle=False``).
        :param channel: Spin channel of the object (see :class:`SpinChannel`).
        :param num_wn_dimensions: Number of bosonic frequency axes (0 or 1).
        :param num_vn_dimensions: Number of fermionic frequency axes (0, 1 or 2).
        :param full_niw_range: Whether the object spans the full (signed) bosonic range or only :math:`\omega \geq 0`.
        :param full_niv_range: Whether the object spans the full (signed) fermionic range or only :math:`\nu \geq 0`.
        :param frequency_notation: Frequency convention (see :class:`FrequencyNotation`).
        :return: The loaded :class:`LocalFourPoint`.
        """
        return LocalFourPoint(
            np.load(filename, allow_pickle=False),
            channel,
            num_wn_dimensions,
            num_vn_dimensions,
            full_niw_range,
            full_niv_range,
            frequency_notation,
        )

    @staticmethod
    def from_constant(
        n_bands: int,
        niw: int,
        niv: int,
        num_wn_dimensions: int = 1,
        num_vn_dimensions: int = 2,
        channel: SpinChannel = SpinChannel.NONE,
        frequency_notation: FrequencyNotation = FrequencyNotation.PH,
        value: float = 0.0,
    ) -> "LocalFourPoint":
        r"""
        Creates a :class:`LocalFourPoint` filled with a constant value.

        :param n_bands: Number of orbitals/bands per orbital axis.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param num_wn_dimensions: Number of bosonic frequency axes (0 or 1).
        :param num_vn_dimensions: Number of fermionic frequency axes (0, 1 or 2).
        :param channel: Spin channel of the object (see :class:`SpinChannel`).
        :param frequency_notation: Frequency convention (see :class:`FrequencyNotation`).
        :param value: Constant fill value.
        :return: The constant :class:`LocalFourPoint` (in the full niw/niv range).
        """
        shape = (n_bands,) * 4 + (2 * niw + 1,) * num_wn_dimensions + (2 * niv,) * num_vn_dimensions
        return LocalFourPoint(
            np.full(shape, value, dtype=DTYPE),
            channel,
            num_wn_dimensions,
            num_vn_dimensions,
            frequency_notation=frequency_notation,
        )

    @staticmethod
    def identity(
        n_bands: int,
        niw: int,
        niv: int,
        num_vn_dimensions: int = 2,
        full_niw_range: bool = False,
        frequency_notation: FrequencyNotation = FrequencyNotation.PH,
    ) -> "LocalFourPoint":
        r"""
        Creates a :class:`LocalFourPoint` that is the identity in compound-index (matrix) space.

        :param n_bands: Number of orbitals/bands per orbital axis.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param num_vn_dimensions: Number of fermionic frequency axes (1 or 2).
        :param full_niw_range: Whether the object spans the full (signed) bosonic range or only :math:`\omega \geq 0`.
        :param frequency_notation: Frequency convention (see :class:`FrequencyNotation`).
        :return: The identity :class:`LocalFourPoint`.
        :raises ValueError: If ``num_vn_dimensions`` is not 1 or 2.
        """
        if num_vn_dimensions not in (1, 2):
            raise ValueError("Invalid number of fermionic frequency dimensions.")
        full_shape = (n_bands,) * 4 + (2 * niw + 1,) + (2 * niv,) * num_vn_dimensions
        compound_index_size = 2 * niv * n_bands**2
        mat = np.tile(np.eye(compound_index_size, dtype=DTYPE)[None, ...], (2 * niw + 1, 1, 1))

        result = LocalFourPoint(
            mat,
            num_vn_dimensions=num_vn_dimensions,
            full_niw_range=full_niw_range,
            frequency_notation=frequency_notation,
        ).to_full_indices(full_shape)
        if num_vn_dimensions == 1:
            return result.take_vn_diagonal()
        return result

    @staticmethod
    def identity_like(other: "LocalFourPoint") -> "LocalFourPoint":
        """
        Creates a compound-index identity matching the bands, frequency box, fermionic-axis count, niw range and
        frequency notation of ``other`` (see :meth:`identity`).

        :param other: The :class:`LocalFourPoint` whose shape/attributes the identity should match.
        :return: The matching identity :class:`LocalFourPoint`.
        """
        return LocalFourPoint.identity(
            other.n_bands, other.niw, other.niv, other.num_vn_dimensions, other.full_niw_range, other.frequency_notation
        )

    def _revert_frequency_dimensions_after_operation(
        self, other: "LocalFourPoint", other_extended: bool, self_extended: bool
    ):
        """
        Undoes the temporary diagonal extension of fermionic axes applied by
        :meth:`_align_frequency_dimensions_for_operation`, collapsing extended objects back to a single
        fermionic axis via :meth:`take_vn_diagonal`.

        :param other: The other operand, possibly diagonally extended.
        :param other_extended: Whether ``other`` was extended and must be collapsed.
        :param self_extended: Whether ``self`` was extended and must be collapsed.
        :return: ``other``, collapsed back if it had been extended.
        """
        if self_extended:
            self.take_vn_diagonal()
        if other_extended:
            other = other.take_vn_diagonal()
        return other

    def _get_frequency_suffixes_for_matmul(self, other: "LocalFourPoint", left_hand_side: bool) -> tuple[str, str, str]:
        """
        Builds the einsum frequency-index suffixes for a :meth:`matmul` between operands with fewer than two
        fermionic frequency dimensions, picking the layout that keeps the larger fermionic axis count.

        :param other: The other operand of the matrix product.
        :param left_hand_side: Whether ``self`` is on the left of the product.
        :return: The tuple ``(suffix_other, suffix_result, suffix_self)`` of frequency-index strings.
        """
        base = {0: "w", 1: "wv", 2: "wvp"}
        suffix_self = base[self.num_vn_dimensions]
        suffix_other = base[other.num_vn_dimensions]
        suffix_result = base[max(self.num_vn_dimensions, other.num_vn_dimensions)]
        if left_hand_side and (self.num_vn_dimensions, other.num_vn_dimensions) == (2, 1):
            suffix_self, suffix_other, suffix_result = "wvp", "wp", "wvp"
        if not left_hand_side and (self.num_vn_dimensions, other.num_vn_dimensions) == (1, 2):
            suffix_self, suffix_other, suffix_result = "wp", "wvp", "wvp"
        return suffix_other, suffix_result, suffix_self

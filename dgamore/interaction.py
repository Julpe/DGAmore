# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Interaction tensors. :class:`LocalInteraction` wraps the momentum-independent (Hubbard/Kanamori) interaction
:math:`U_{abcd}`; :class:`Interaction` adds a momentum dimension for the non-local interaction :math:`V_{abcd}^q`.
Both provide the channel projections (density/magnetic/singlet/triplet) and the algebra used in the ladder
equations.
"""

from copy import deepcopy

import numpy as np

from dgamore.n_point_base import IHaveMat, IHaveChannel, IAmNonLocal, SpinChannel


class LocalInteraction(IHaveMat, IHaveChannel):
    r"""
    Class for the local (momentum-independent) interaction :math:`U_{abcd}`.
    """

    def __init__(self, mat: np.ndarray, channel: SpinChannel = SpinChannel.NONE):
        r"""
        Initializes the local interaction tensor in the given spin channel.

        :param mat: Interaction tensor :math:`U_{abcd}` with four orbital axes ``[o1, o2, o3, o4]``.
        :param channel: Spin channel the tensor is expressed in (see :class:`SpinChannel`); ``NONE`` for the
            bare, not-yet-projected interaction.
        """
        IHaveMat.__init__(self, mat)
        IHaveChannel.__init__(self, channel)

    @property
    def n_bands(self) -> int:
        """
        Returns the number of bands (orbitals).

        :return: Number of bands, i.e. the size of the first orbital axis.
        """
        return self.original_shape[0]

    def permute_orbitals(self, permutation: str = "abcd->abcd") -> "LocalInteraction":
        """
        Permutes the four orbital axes according to an einsum-style permutation string.

        :param permutation: einsum permutation of the four orbital labels, e.g. ``"abcd->adcb"``.
        :return: A new :class:`LocalInteraction` with permuted orbitals (a deep copy for the identity permutation).
        :raises ValueError: If the permutation string is not a valid four-orbital permutation.
        """
        split = permutation.split("->")
        if len(split) != 2 or len(split[0]) != 4 or len(split[1]) != 4:
            raise ValueError("Invalid permutation.")

        if split[0] == split[1]:
            return deepcopy(self)

        return LocalInteraction(np.einsum(permutation, self.mat, optimize=True), self.channel)

    def as_channel(self, channel: SpinChannel) -> "LocalInteraction":
        r"""
        Projects the bare interaction onto a given spin channel. With the crossing-permuted tensor
        :math:`\tilde U_{abcd} = U_{adcb}` the projections are
        :math:`U_d = 2U - \tilde U`, :math:`U_m = -\tilde U`, :math:`U_s = U + \tilde U`, :math:`U_t = U - \tilde U`.

        :param channel: Target spin channel (density, magnetic, singlet or triplet).
        :return: A new :class:`LocalInteraction` in the requested channel.
        :raises ValueError: If the object is already in a (non-``NONE``) channel, or the target channel is unsupported.
        """
        copy = deepcopy(self)

        if copy.channel == channel:
            return copy
        elif copy.channel != channel.NONE:
            raise ValueError(f"Cannot transform interaction from channel {copy.channel} to {channel}.")

        copy._channel = channel
        perm: str = "abcd->adcb"
        if channel == SpinChannel.DENS:
            return 2 * copy - copy.permute_orbitals(perm)
        elif channel == SpinChannel.MAGN:
            return -copy.permute_orbitals(perm)
        elif channel == SpinChannel.SING:
            return copy + copy.permute_orbitals(perm)
        elif channel == SpinChannel.TRIP:
            return copy - copy.permute_orbitals(perm)
        else:
            raise ValueError(f"Channel {channel} not supported.")

    def add(self, other) -> "LocalInteraction":
        """
        Adds another interaction or a raw numpy array elementwise.

        :param other: A :class:`LocalInteraction` or a numpy array broadcastable to ``mat``.
        :return: A new :class:`LocalInteraction` holding the sum (inheriting the non-``NONE`` channel of the operands).
        :raises ValueError: If ``other`` is neither a :class:`LocalInteraction` nor a numpy array.
        """
        if not isinstance(other, (LocalInteraction, np.ndarray)):
            raise ValueError(f"Operation {type(self)} +/- {type(other)} not supported.")

        if isinstance(other, np.ndarray):
            return LocalInteraction(self.mat + other, self.channel)

        return LocalInteraction(
            self.mat + other.mat, self.channel if self.channel != SpinChannel.NONE else other.channel
        )

    def sub(self, other) -> "LocalInteraction":
        """
        Subtracts another interaction or a raw numpy array elementwise.

        :param other: A :class:`LocalInteraction` or a numpy array broadcastable to ``mat``.
        :return: A new :class:`LocalInteraction` holding the difference ``self - other``.
        """
        return self.add(-other)

    def pow(self, power) -> "LocalInteraction":
        r"""
        Raises the interaction to an integer power via repeated orbital contraction
        :math:`U^{(n)}_{abef} = U_{abcd} U^{(n-1)}_{dcef}`.

        :param power: Positive integer exponent (must be greater than zero).
        :return: A new :class:`LocalInteraction` equal to ``self`` contracted with itself ``power`` times.
        :raises ValueError: If ``power`` is not a positive integer greater than zero.
        """
        if power <= 0:
            raise ValueError("Exponentiation of Interaction objects only supports positive powers greater than zero.")
        result = deepcopy(self)
        for _ in range(1, power):
            result = LocalInteraction(result.times("abcd,dcef->abef", self), self.channel)
        return result

    def __add__(self, other) -> "LocalInteraction":
        """
        Operator overload for ``self + other``; see :meth:`add`.

        :param other: A :class:`LocalInteraction` or a numpy array.
        :return: The sum as a new :class:`LocalInteraction`.
        """
        return self.add(other)

    def __radd__(self, other) -> "LocalInteraction":
        """
        Reflected operator overload for ``other + self``; see :meth:`add` (addition commutes).

        :param other: A :class:`LocalInteraction` or a numpy array.
        :return: The sum as a new :class:`LocalInteraction`.
        """
        return self.add(other)

    def __sub__(self, other) -> "LocalInteraction":
        """
        Operator overload for ``self - other``; see :meth:`sub`.

        :param other: A :class:`LocalInteraction` or a numpy array.
        :return: The difference ``self - other`` as a new :class:`LocalInteraction`.
        """
        return self.sub(other)

    def __rsub__(self, other) -> "LocalInteraction":
        """
        Reflected operator overload for ``other - self`` (returns ``-(self - other)``).

        :param other: A :class:`LocalInteraction` or a numpy array on the left-hand side.
        :return: The difference ``other - self`` as a new :class:`LocalInteraction`.
        """
        return -self.sub(other)

    def __pow__(self, power, modulo=None) -> "LocalInteraction":
        """
        Operator overload for ``self ** power``; see :meth:`pow`.

        :param power: Positive integer exponent.
        :param modulo: Unused (present for the ``__pow__`` protocol).
        :return: ``self`` raised to ``power`` as a new :class:`LocalInteraction`.
        """
        return self.pow(power)


class Interaction(IAmNonLocal, LocalInteraction):
    r"""
    Class for the non-local (momentum-dependent) interaction :math:`V_{abcd}^q`.
    """

    def __init__(
        self,
        mat: np.ndarray,
        channel: SpinChannel = SpinChannel.NONE,
        nq: tuple[int, int, int] = (1, 1, 1),
        has_compressed_q_dimension: bool = False,
    ):
        r"""
        Initializes the non-local interaction tensor in the given spin channel and momentum layout.

        :param mat: Interaction tensor :math:`V_{abcd}^q` with one momentum dimension and four orbital axes.
        :param channel: Spin channel the tensor is expressed in (see :class:`SpinChannel`).
        :param nq: Number of momenta per spatial direction ``(nqx, nqy, nqz)``.
        :param has_compressed_q_dimension: Whether the momentum is stored as a single compressed axis ``[q, ...]``
            (True) or as three separate axes ``[qx, qy, qz, ...]`` (False).
        """
        LocalInteraction.__init__(self, mat, channel)
        IAmNonLocal.__init__(self, mat, nq, has_compressed_q_dimension)

    def permute_orbitals(self, permutation: str = "abcd->abcd") -> "Interaction":
        """
        Permutes the four orbital axes (leaving the momentum axis untouched).

        :param permutation: einsum permutation of the four orbital labels, e.g. ``"abcd->adcb"``.
        :return: A new :class:`Interaction` with permuted orbitals (a deep copy for the identity permutation).
        :raises ValueError: If the permutation string is not a valid four-orbital permutation.
        """
        split = permutation.split("->")
        if len(split) != 2 or len(split[0]) != 4 or len(split[1]) != 4:
            raise ValueError("Invalid permutation.")

        if split[0] == split[1]:
            return deepcopy(self)

        permutation = f"...{split[0]}->...{split[1]}"
        return Interaction(
            np.einsum(permutation, self.mat, optimize=True), self.channel, self.nq, self.has_compressed_q_dimension
        )

    def as_channel(self, channel: SpinChannel) -> "Interaction":
        r"""
        Projects the bare non-local interaction onto a given spin channel. Only the non-local particle-hole
        contribution enters the ladder DGA equations and the ph-bar contribution to the spin channels vanishes,
        so the projections reduce to :math:`V_d = 2V`, :math:`V_m = 0`, :math:`V_s = V_t = V`.

        :param channel: Target spin channel (density, magnetic, singlet or triplet).
        :return: A new :class:`Interaction` in the requested channel.
        :raises ValueError: If the object is already in a (non-``NONE``) channel, or the target channel is unsupported.
        """
        copy = deepcopy(self)

        if copy.channel == channel:
            return copy
        elif copy.channel != channel.NONE:
            raise ValueError(f"Cannot transform interaction from channel {copy.channel} to {channel}.")

        copy._channel = channel
        if channel == SpinChannel.DENS:
            return 2 * copy
        elif channel == SpinChannel.MAGN:
            return 0 * copy
        elif channel == SpinChannel.SING:
            return copy
        elif channel == SpinChannel.TRIP:
            return copy
        else:
            raise ValueError(f"Channel {channel} not supported.")

    def add(self, other) -> "Interaction":
        """
        Adds another (non-)local interaction or a raw numpy array. A local operand is broadcast over the
        momentum axis.

        :param other: An :class:`Interaction`, a :class:`LocalInteraction`, or a numpy array.
        :return: A new :class:`Interaction` holding the sum (inheriting the non-``NONE`` channel of the operands).
        :raises ValueError: If ``other`` is not one of the supported types.
        """
        if not isinstance(other, (LocalInteraction, Interaction, np.ndarray)):
            raise ValueError(f"Operation {type(self)} +/- {type(other)} not supported.")

        if isinstance(other, np.ndarray):
            return Interaction(self.mat + other, self.channel, self.nq, self.has_compressed_q_dimension)

        if not isinstance(other, Interaction):
            other_mat = other.mat[None, ...] if self.has_compressed_q_dimension else other.mat[None, None, None, ...]
        else:
            other_mat = other.mat

        return Interaction(
            self.mat + other_mat,
            self.channel if self.channel != SpinChannel.NONE else other.channel,
            self.nq,
            self.has_compressed_q_dimension,
        )

    def sub(self, other) -> "Interaction":
        """
        Subtracts another (non-)local interaction or a raw numpy array.

        :param other: An :class:`Interaction`, a :class:`LocalInteraction`, or a numpy array.
        :return: A new :class:`Interaction` holding the difference ``self - other``.
        """
        return self.add(-other)

    def pow(self, power) -> "Interaction":
        r"""
        Raises the interaction to an integer power via repeated momentum-diagonal orbital contraction
        :math:`V^{(n);q}_{abef} = V^{q}_{abcd} V^{(n-1);q}_{dcef}`.

        :param power: Positive integer exponent (must be greater than zero).
        :return: A new :class:`Interaction` in the same momentum-compression state as ``self``.
        :raises ValueError: If ``power`` is not a positive integer greater than zero.
        """
        if power <= 0:
            raise ValueError("Exponentiation of Interaction objects only supports positive powers greater than zero.")
        is_self_compressed = self.has_compressed_q_dimension
        result = deepcopy(self).compress_q_dimension()
        for _ in range(1, power):
            result = Interaction(result.times("qabcd,qdcef->qabef", self), self.channel, self.nq, True)
        return result if is_self_compressed else result.decompress_q_dimension()

    def __add__(self, other) -> "Interaction":
        """
        Operator overload for ``self + other``; see :meth:`add`.

        :param other: An :class:`Interaction`, a :class:`LocalInteraction`, or a numpy array.
        :return: The sum as a new :class:`Interaction`.
        """
        return self.add(other)

    def __radd__(self, other) -> "Interaction":
        """
        Reflected operator overload for ``other + self``; see :meth:`add` (addition commutes).

        :param other: An :class:`Interaction`, a :class:`LocalInteraction`, or a numpy array.
        :return: The sum as a new :class:`Interaction`.
        """
        return self.add(other)

    def __sub__(self, other) -> "Interaction":
        """
        Operator overload for ``self - other``; see :meth:`sub`.

        :param other: An :class:`Interaction`, a :class:`LocalInteraction`, or a numpy array.
        :return: The difference ``self - other`` as a new :class:`Interaction`.
        """
        return self.sub(other)

    def __rsub__(self, other) -> "Interaction":
        """
        Reflected operator overload for ``other - self`` (returns ``-(self - other)``).

        :param other: An :class:`Interaction`, a :class:`LocalInteraction`, or a numpy array on the left-hand side.
        :return: The difference ``other - self`` as a new :class:`Interaction`.
        """
        return -self.sub(other)

    def __pow__(self, power, modulo=None) -> "Interaction":
        """
        Operator overload for ``self ** power``; see :meth:`pow`.

        :param power: Positive integer exponent.
        :param modulo: Unused (present for the ``__pow__`` protocol).
        :return: ``self`` raised to ``power`` as a new :class:`Interaction`.
        """
        return self.pow(power)

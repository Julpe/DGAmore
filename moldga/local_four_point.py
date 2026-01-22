from moldga.interaction import LocalInteraction, Interaction
from moldga.local_n_point import LocalNPoint
from moldga.matsubara_frequencies import MFHelper
from moldga.n_point_base import *


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
        Adds two vertex objects involving a LocalFourPoint object. Allows for A + B = C.
        """
        return self.add(other)

    def __radd__(self, other):
        """
        Adds two vertex objects involving a LocalFourPoint object. Allows for A + B = C.
        """
        return self.add(other)

    def __sub__(self, other):
        """
        Subtracts two vertex objects involving a LocalFourPoint object. Allows for A - B = C.
        """
        return self.sub(other)

    def __rsub__(self, other):
        """
        Subtracts two vertex objects involving a LocalFourPoint object. Allows for A - B = C.
        """
        return -self.sub(other)

    def __mul__(self, other):
        r"""
        Allows for the multiplication of two LocalFourPoint objects with both having only one fermionic frequency
        dimension or for the multiplication of a LocalFourPoint object with a number or numpy array. Attention: this
        does not do the same as `__matmul__` or `matmul`, but rather :math:`A_{abcd}^{wv} * B_{dcef}^{wv'} = C_{abef}^{wvv'}`.
        """
        return self.mul(other)

    def __rmul__(self, other) -> "LocalFourPoint":
        r"""
        Allows for the multiplication of two LocalFourPoint objects with both having only one fermionic frequency
        dimension or for the multiplication of a LocalFourPoint object with a number or numpy array. Attention: this
        does not do the same as `__matmul__` or `matmul`, but rather :math:`A_{abcd}^{wv} * B_{dcef}^{wv'} = C_{abef}^{wvv'}`.
        """
        return self.mul(other)

    def __matmul__(self, other) -> "LocalFourPoint":
        """
        Matrix multiplication for LocalFourPoint objects. Allows for A @ B = C using compound indices.
        """
        return self.matmul(other, left_hand_side=True)

    def __rmatmul__(self, other) -> "LocalFourPoint":
        """
        Matrix multiplication for LocalFourPoint objects. Allows for A @ B = C using compound indices.
        """
        return self.matmul(other, left_hand_side=False)

    def __invert__(self):
        """
        Inverts the LocalFourPoint object by transforming it to compound indices.
        """
        return self.invert()

    def __pow__(self, power, modulo=None):
        """
        Exponentiates for LocalFourPoint objects. Allows for A ** n = B, where n is an integer. If n < 0, then we
        exponentiate the inverse of A |n| times, i.e., A ** (-n) = A^(-1) ** n.
        """
        return self.pow(power, LocalFourPoint.identity_like(self))

    def pow(self, power: int, identity):
        """
        Exponentiation for LocalFourPoint objects. Allows for A ** n = B, where n is an integer. If n < 0, then we
        exponentiate the inverse of A |n| times, i.e., A ** (-n) = A^(-1) ** n. Requires the input of the identity.
        """
        if not isinstance(power, int):
            raise ValueError("Only integer powers are supported.")

        if power == 0:
            return identity
        if power == 1:
            return self
        if power < 0:
            return self.invert() ** abs(power)

        result = identity
        base = deepcopy(self)

        # Exponentiation by squaring
        while power > 0:
            if power % 2 == 1:
                result @= base
            base @= base
            power //= 2

        return result

    def sum_over_orbitals(self, orbital_contraction: str = "abcd->ad"):
        """
        Sums over the given orbitals with the contraction given. Raises an error if the contraction is not valid.
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
        Sums over a specific number of fermionic frequency dimensions and multiplies the with the correct prefactor
        :math:`1/\beta^{n_dim}`.
        """
        if len(axis) > self.num_vn_dimensions:
            raise ValueError(f"Cannot sum over more fermionic axes than available in {self.current_shape}.")
        copy_mat = 1 / beta ** len(axis) * np.sum(self.mat, axis=axis)
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
        Sums over all fermionic frequency dimensions and multiplies the with the correct prefactor
        :math:`1/\beta^{n_dim}`.
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
        """
        Contracts the outer legs of a four-point vertex. This is equivalent to a sum over all fermionic frequency
        dimensions if the object has 2 fermionic frequency dimensions and a sum over the inner two orbital dimensions.
        """
        if self.num_vn_dimensions != 2:
            raise ValueError("This method is only implemented for objects with 2 fermionic frequency dimensions.")
        return self.sum_over_all_vn(beta).sum_over_orbitals("abcd->ad")

    def to_compound_indices(self) -> "LocalFourPoint":
        r"""
        Converts the indices of the LocalFourPoint object :math:`F^{wvv'}_{lmm'l'}` to compound indices :math:`F^{w}_{c_1, c_2}`
        by transposing the object to [w, o1, o2, v, o4, o3, v'] (if the object has any fermionic frequency dimension,
        otherwise the compound indices are built from orbital dimensions only) and grouping {o1, o2, v} and {o4, o3, v'}
        to the new compound index. Always returns the object with a compressed momentum dimension and in the same niw
        range as the original object.
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
        Converts the indices of the LocalFourPoint object in ph channel to compound indices.
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
        Converts the indices of the LocalFourPoint object in pp channel to compound indices. The difference to the ph
        channel is that the orbital indices have to be permuted first since the ordering of pp quantities is not
        "1234" but rather "1324".
        """
        if len(self.current_shape) == 3:  # [w,x1,x2]
            return self

        return self.permute_orbitals("abcd->acbd", copy=False)._to_compound_indices_ph()

    def to_full_indices(self, shape: tuple = None) -> "LocalFourPoint":
        """
        Converts an object stored with compound indices to an object that has unraveled momentum,
        orbital and frequency axes. Always returns the object with a compressed momentum dimension. This is the inverse
        transformation as `to_compound_indices`. Will make use of the `original_shape` the object was created or last
        modified with. If the `original_shape` is not set or is hard to obtain, the `shape` argument can be used to
        specify the original shape of the object.
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
        Converts the indices of the LocalFourPoint object in ph channel to full indices.
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
            self.mat = self.mat.diagonal(axis1=-2, axis2=-1)
        return self

    def _to_full_indices_pp(self, shape: tuple = None) -> "LocalFourPoint":
        """
        Converts the indices of the LocalFourPoint object in pp channel to full indices. The difference to the ph
        channel is that the orbital indices have to be permuted back since the ordering of pp quantities is not
        "1234" but rather "1324".
        """
        return self._to_full_indices_ph(shape).permute_orbitals("abcd->acbd", copy=False)

    def invert(self, copy: bool = True):
        """
        Inverts the object by transforming it to compound indices. Returns the object always in half of their niw range.
        """

        if copy:
            copy = deepcopy(self.to_half_niw_range()).to_compound_indices()
            copy.mat = np.linalg.inv(copy.mat)
            return copy.to_full_indices()

        self.to_half_niw_range().to_compound_indices()
        self.mat = np.linalg.inv(self.mat)
        return self.to_full_indices()

    def matmul(self, other, left_hand_side: bool = True) -> "LocalFourPoint":
        """
        Helper method that allows for a matrix multiplication between LocalFourPoint and LocalFourPoint and LocalInteraction
        objects. Depending on the number of frequency and momentum dimensions,
        the objects have to be multiplied differently. The use of einsum is very crucial for memory efficiency here,
        as a regular matrix multiplication in compound index space would create large intermediate arrays if one of both
        partaking objects has less than two fermionic frequency dimensions. Result objects will always be returned in
        half of their niw range to save memory.
        """
        if not isinstance(other, (LocalFourPoint, LocalInteraction)):
            raise ValueError(f"Multiplication {type(self)} @ {type(other)} not supported.")

        if isinstance(other, LocalInteraction):
            einsum_str = {
                0: "abijw,jief->abefw" if left_hand_side else "abij,jiefw->abefw",
                1: "abijwv,jief->abefwv" if left_hand_side else "abij,jiefwv->abefwv",
                2: "abijwvp,jief->abefwvp" if left_hand_side else "abij,jiefwvp->abefwvp",
            }.get(self.num_vn_dimensions)
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
            # special cases if both objects do not have two fermionic frequency dimensions each. Straightforward
            # contraction is saving memory as we do not have to add fermionic frequency dimensions to artificially
            # create compound indices
            self.to_half_niw_range()
            other.to_half_niw_range()

            suffix_other, suffix_result, suffix_self = self._get_frequency_suffixes_for_matmul(other, left_hand_side)

            einsum_str = (
                f"abcd{suffix_self},dcef{suffix_other}->abef{suffix_result}"
                if left_hand_side
                else f"abcd{suffix_other},dcef{suffix_self}->abef{suffix_result}"
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
        ).to_full_indices(shape)

    def mul(self, other):
        r"""
        Allows for the multiplication of two LocalFourPoint objects with both having only one fermionic frequency
        dimension or for the multiplication of a LocalFourPoint object with a number or numpy array. Attention: this
        does not do the same as `__matmul__` or `matmul`, but rather :math:`A_{abcd}^{wv} * B_{dcef}^{wv'} = C_{abef}^{wvv'}`.
        """
        if not isinstance(other, (int, float, complex, np.ndarray, LocalFourPoint)):
            raise ValueError("Multiplication only supported with numbers, numpy arrays or LocalFourPoint objects.")

        if not isinstance(other, LocalFourPoint):
            copy = deepcopy(self)
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

    def add(self, other):
        """
        Helper method that allows for addition of LocalFourPoint objects and other LocalFourPoint or LocalInteraction
        objects. Additions with numpy arrays, floats, ints or complex numbers are also supported.
        Depending on the number of frequency and momentum dimensions, the vertices have to be added slightly different.
        If the objects have different niw ranges, they will be converted to the half niw range before the addition.
        Objects will always be returned in the half niw range to save memory.
        """
        if not isinstance(other, (LocalFourPoint, LocalInteraction, np.ndarray, float, int, complex)):
            raise ValueError(f"Operations '+/-' for {type(self)} and {type(other)} not supported.")

        if isinstance(other, (np.ndarray, float, int, complex)):
            return LocalFourPoint(
                self.mat + other,
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
                    (self.mat[None, ...] + other_mat)
                    if other.has_compressed_q_dimension
                    else (self.mat[None, None, None, ...] + other_mat)
                )

            return LocalFourPoint(
                self.mat + other_mat,
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
            self_mat = self.mat
            other_mat = other.mat

            if self.num_vn_dimensions != other.num_vn_dimensions:
                if self.num_vn_dimensions == 0:
                    self_mat = np.expand_dims(self_mat, axis=-1 if other.num_vn_dimensions == 1 else (-1, -2))
                elif other.num_vn_dimensions == 0:
                    other_mat = np.expand_dims(other_mat, axis=-1 if self.num_vn_dimensions == 1 else (-1, -2))

            result = LocalFourPoint(
                self_mat + other_mat,
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

        result = LocalFourPoint(
            self.mat + other.mat,
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

    def sub(self, other):
        """
        Helper method that allows for Subtraction of LocalFourPoint objects and other LocalFourPoint or LocalInteraction
        objects. Subtractions with numpy arrays, floats, ints or complex numbers are also supported.
        Depending on the number of frequency and momentum dimensions, the vertices have to be subtracted slightly different.
        If the objects have different niw ranges, they will be converted to the half niw range before the subtraction.
        Objects will always be returned in the half niw range to save memory.
        """
        return self.add(-other)

    def permute_orbitals(self, permutation: str = "abcd->abcd", copy: bool = True) -> "LocalFourPoint":
        """
        Permutes the orbitals of the LocalFourPoint object with the string given. It is not possible to sum over any
        orbitals with this method.
        """
        split = permutation.split("->")
        if (
            len(split) != 2
            or len(split[0]) != self.num_orbital_dimensions
            or len(split[1]) != self.num_orbital_dimensions
        ):
            raise ValueError("Invalid permutation.")

        permutation = f"{split[0]}...->{split[1]}..."

        if not copy:
            self.mat = np.einsum(permutation, self.mat, optimize=True)
            return self

        copy = deepcopy(self)
        copy.mat = np.einsum(permutation, copy.mat, optimize=True)
        return copy

    def symmetrize_v_vp(self):
        """
        Symmetrizes the vertex with respect to (v,v'). This is justified for SU(2) symmetric systems, see PhD Thesis
        of Georg Rohringer p. 72
        """
        if self.num_vn_dimensions != 2:
            raise ValueError("This method is only implemented for objects with 2 fermionic frequency dimensions.")

        self.mat = 0.5 * (self.mat + np.swapaxes(self.permute_orbitals("abcd->cdab").mat, -1, -2))
        return self

    def change_frequency_notation_ph_to_pp_w0(self):
        r"""
        Changes the frequency notation of the object from ph to pp and returns a copy in half the niw range.
        The frequency shifts are :math:`(w,v_1,v_2) -> (w',v_1',v_2') = (v_1 + v_2 - w, v_1, v_2)`.
        """
        if self.num_wn_dimensions != 1 or self.num_vn_dimensions not in (1, 2):
            raise ValueError("Object must have 1 bosonic and 1 or 2 fermionic frequency dimensions.")

        copy = deepcopy(self)

        if copy.frequency_notation == FrequencyNotation.PP:
            return copy

        copy = copy.to_full_niw_range()

        if copy.num_vn_dimensions == 1:
            copy = copy.extend_vn_to_diagonal()

        iw_pp, iv_pp, ivp_pp = MFHelper.get_frequencies_for_ph_to_pp_w0_channel_conversion(copy.niw, copy.niv)
        copy.mat = copy.mat[..., iw_pp, iv_pp, ivp_pp][..., None, :, :]
        copy.update_original_shape()
        return copy.set_frequency_notation(FrequencyNotation.PP)

    def create_wn_dimension(self):
        """
        Creates a bosonic frequency dimension for the LocalFourPoint object if it does not have one yet.
        The new bosonic frequency dimension will have niw=0 only.
        """
        if self.num_wn_dimensions != 0:
            raise ValueError("Object already has bosonic frequency dimensions.")

        copy = deepcopy(self)
        # insert bosonic axis immediately before the last vn axes
        copy.mat = np.expand_dims(copy.mat, axis=-(copy.num_vn_dimensions + 1))
        copy._num_wn_dimensions = 1
        copy.update_original_shape()
        return copy

    def take_first_wn(self):
        """
        Selects the first entry in the wn dimension of the LocalFourPoint object, effectively removing the bosonic
        frequency dimension.
        """
        if self.num_wn_dimensions != 1:
            raise ValueError("Object must have exactly one bosonic frequency dimension.")

        copy = deepcopy(self)
        # select first entry of wn
        copy.mat = np.take(copy.mat, 0, axis=-(copy.num_vn_dimensions + 1))
        copy._num_wn_dimensions = 0
        copy.update_original_shape()
        return copy

    def pad_with_u(self, u: LocalInteraction, niv_pad: int):
        """
        Used to pad a LocalFourPoint object with a LocalInteraction object outside the core frequency region.
        If niv_pad is less or equal self.niv, no padding will be done. Mainly used to add the interaction to the irreducible
        vertex out of the core frequency region.
        """
        copy = deepcopy(self)

        if niv_pad <= copy.niv:
            return copy

        urange_mat = np.tile(u.mat[..., None, None, None], (1, 1, 1, 1, 2 * copy.niw + 1, 2 * niv_pad, 2 * niv_pad))
        niv_diff = niv_pad - copy.niv
        urange_mat[..., niv_diff : niv_diff + 2 * copy.niv, niv_diff : niv_diff + 2 * copy.niv] = copy.mat
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
        """
        Loads a LocalFourPoint object from a file. The file must be of type '.npy'.
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
        """
        Initializes a LocalFourPoint object with a constant value.
        """
        shape = (n_bands,) * 4 + (2 * niw + 1,) * num_wn_dimensions + (2 * niv,) * num_vn_dimensions
        return LocalFourPoint(
            np.full(shape, value),
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
        """
        Creates an identity (matrix in compound index notation is unity) for the LocalFourPoint object.
        """
        if num_vn_dimensions not in (1, 2):
            raise ValueError("Invalid number of fermionic frequency dimensions.")
        full_shape = (n_bands,) * 4 + (2 * niw + 1,) + (2 * niv,) * num_vn_dimensions
        compound_index_size = 2 * niv * n_bands**2
        mat = np.tile(np.eye(compound_index_size)[None, ...], (2 * niw + 1, 1, 1))

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
        Creates an identity (matrix in compound index notation is unity) for the LocalFourPoint object from the shape
        of another LocalFourPoint object.
        """
        return LocalFourPoint.identity(
            other.n_bands, other.niw, other.niv, other.num_vn_dimensions, other.full_niw_range, other.frequency_notation
        )

    def _revert_frequency_dimensions_after_operation(
        self, other: "LocalFourPoint", other_extended: bool, self_extended: bool
    ):
        if self_extended:
            self.take_vn_diagonal()
        if other_extended:
            other = other.take_vn_diagonal()
        return other

    def _get_frequency_suffixes_for_matmul(self, other: "LocalFourPoint", left_hand_side: bool) -> tuple[str, str, str]:
        base = {0: "w", 1: "wv", 2: "wvp"}
        suffix_self = base[self.num_vn_dimensions]
        suffix_other = base[other.num_vn_dimensions]
        suffix_result = base[max(self.num_vn_dimensions, other.num_vn_dimensions)]
        if left_hand_side and (self.num_vn_dimensions, other.num_vn_dimensions) == (2, 1):
            suffix_self, suffix_other, suffix_result = "wvp", "wp", "wvp"
        if not left_hand_side and (self.num_vn_dimensions, other.num_vn_dimensions) == (1, 2):
            suffix_self, suffix_other, suffix_result = "wp", "wvp", "wvp"
        return suffix_other, suffix_result, suffix_self

from copy import deepcopy

import numpy as np

from moldga.interaction import Interaction, LocalInteraction
from moldga.local_four_point import LocalFourPoint
from moldga.n_point_base import IAmNonLocal, SpinChannel, FrequencyNotation


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
        Adds two vertex objects involving a FourPoint object. Allows for A + B = C.
        """
        return self.add(other)

    def __radd__(self, other) -> "FourPoint":
        """
        Adds two vertex objects involving a FourPoint object. Allows for A + B = C.
        """
        return self.add(other)

    def __sub__(self, other) -> "FourPoint":
        """
        Subtracts two vertex objects involving a FourPoint object. Allows for A + B = C.
        """
        return self.sub(other)

    def __rsub__(self, other) -> "FourPoint":
        """
        Subtracts two vertex objects involving a FourPoint object. Allows for A - B = C.
        """
        return -self.sub(other)

    def __mul__(self, other) -> "FourPoint":
        r"""
        Multiplies two FourPoint objects with both having only one fermionic frequency
        dimension or for the multiplication of a FourPoint object with a number or numpy array. Attention: this does not
        do the same as `__matmul__` or `matmul`, but rather :math:`A_{abcd}^{qv} * B_{dcef}^{qv'} = C_{abef}^{qvv'}`.
        """
        return self.mul(other)

    def __rmul__(self, other) -> "FourPoint":
        r"""
        Multiplies two FourPoint objects with both having only one fermionic frequency
        dimension or for the multiplication of a FourPoint object with a number or numpy array. Attention: this does not
        do the same as `__matmul__` or `matmul`, but rather :math:`A_{abcd}^{qv} * B_{dcef}^{qv'} = C_{abef}^{qvv'}`.
        """
        return self.mul(other)

    def __matmul__(self, other) -> "FourPoint":
        """
        Matrix multiplication for FourPoint objects. Allows for A @ B = C.
        """
        return self.matmul(other, left_hand_side=True)

    def __rmatmul__(self, other) -> "FourPoint":
        """
        Matrix multiplication for FourPoint objects. Allows for A @ B = C.
        """
        return self.matmul(other, left_hand_side=False)

    def __invert__(self):
        """
        Inverts the FourPoint object by transforming it to compound indices.
        """
        return self.invert()

    def __pow__(self, power, modulo=None) -> "FourPoint":
        """
        Exponentiates FourPoint objects. Allows for A ** n = B, where n is an integer. If n < 0, then we
        exponentiate the inverse of A |n| times, i.e., A ** (-3) = [A^(-1)] ** 3.
        """
        return self.pow(power, FourPoint.identity_like(self))

    def sum_over_vn(self, beta: float, axis: tuple = (-1,)) -> "FourPoint":
        r"""
        Sums over a specific number of fermionic frequency dimensions and multiplies the with the correct prefactor
        :math:`1/\beta^{n_dim}`.
        """
        if len(axis) > self.num_vn_dimensions:
            raise ValueError(f"Cannot sum over more fermionic axes than available in {self.current_shape}.")
        copy_mat = 1 / beta ** len(axis) * np.sum(self.mat, axis=axis)
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
        Sums over the given orbitals with the contraction given. Raises an error if the contraction is not valid.
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
        Converts the indices of the LocalFourPoint object :math:`F^{wvv'}_{lmm'l'}` to compound indices :math:`F^{w}_{c_1, c_2}`
        by transposing the object to [q, w, o1, o2, v, o4, o3, v'] (if the object has any fermionic frequency dimension,
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

    def _to_compound_indices_ph(self) -> "FourPoint":
        r"""
        Converts the indices of the FourPoint object in ph channel to compound indices.
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
        Converts the indices of the LocalFourPoint object in pp channel to compound indices. The difference to the ph
        channel is that the orbital indices have to be permuted first since the ordering of pp quantities is not
        "1234" but rather "1324".
        """
        if len(self.current_shape) == 3:  # [q, w, x1, x2]
            return self

        return self.permute_orbitals("abcd->acbd", copy=False)._to_compound_indices_ph()

    def to_full_indices(self, shape: tuple = None) -> "FourPoint":
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

    def _to_full_indices_ph(self, shape: tuple = None) -> "FourPoint":
        """
        Converts the indices of the FourPoint object in ph channel to full indices.
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
            self._has_compressed_momentum_dimension = True
            return self

        compound_index_shape = (self.n_bands, self.n_bands, 2 * self.niv)

        # original was [q,o1,o2,o4,o3,w,v,v']
        self.mat = self.mat.reshape((self.nq_tot,) + (w_dim,) + compound_index_shape * 2).transpose(
            0, 2, 3, 6, 5, 1, 4, 7
        )

        if self.num_vn_dimensions == 1:  # original was [q,o1,o2,o4,o3,w,v]
            self.mat = self.mat.diagonal(axis1=-2, axis2=-1)
        return self

    def _to_full_indices_pp(self, shape: tuple = None) -> "FourPoint":
        """
        Converts the indices of the FourPoint object in pp channel to full indices. The difference to the ph
        channel is that the orbital indices have to be permuted back since the ordering of pp quantities is not
        "1234" but rather "1324".
        """
        return self._to_full_indices_ph(shape).permute_orbitals("abcd->acbd", copy=False)

    def permute_orbitals(self, permutation: str = "abcd->abcd", copy: bool = True) -> "FourPoint":
        """
        Permutes the orbitals of the FourPoint object with the string given. It is not possible to sum over any
        orbitals with this method.
        """
        split = permutation.split("->")
        if len(split) != 2 or len(split[0]) != 4 or len(split[1]) != 4:
            raise ValueError("Invalid permutation.")

        if split[0] == split[1]:
            return self

        permutation = (
            f"i{split[0]}...->i{split[1]}..."
            if self.has_compressed_q_dimension
            else f"ijk{split[0]}...->ijk{split[1]}..."
        )

        if not copy:
            self.mat = np.einsum(permutation, self.mat, optimize=True)
            return self

        copy = deepcopy(self)
        copy.mat = np.einsum(permutation, copy.mat, optimize=True)
        return copy

    def add(self, other) -> "FourPoint":
        """
        Helper method that allows for addition of FourPoint objects and other FourPoint, LocalFourPoint, Interaction or
        LocalInteraction objects. Additions with numpy arrays, floats, ints or complex numbers are also supported.
        Depending on the number of frequency and momentum dimensions, the vertices have to be added slightly different.
        If the objects have different niw ranges, they will be converted to the half niw range before the addition.
        Objects will always be returned in the half niw range to save memory.
        """
        if not isinstance(
            other, (FourPoint, LocalFourPoint, Interaction, LocalInteraction, np.ndarray, float, int, complex)
        ):
            raise ValueError(f"Operations '+/-' for {type(self)} and {type(other)} not supported.")

        if isinstance(other, (np.ndarray, float, int, complex)):
            return FourPoint(
                self.mat + other,
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
            return FourPoint(
                self.mat + other_mat,
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
            result = FourPoint(
                (
                    self.mat
                    + (other.mat[None, ...] if self.has_compressed_q_dimension else other.mat[None, None, None, ...])
                ),
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

        result = FourPoint(
            self.mat + other.mat,
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

    def sub(self, other) -> "FourPoint":
        """
        Helper method that allows for subtracted of FourPoint objects and other FourPoint, LocalFourPoint, Interaction or
        LocalInteraction objects. Additions with numpy arrays, floats, ints or complex numbers are also supported.
        Depending on the number of frequency and momentum dimensions, the vertices have to be added slightly different.
        If the objects have different niw ranges, they will be converted to the half niw range before the subtraction.
        Objects will always be returned in the half niw range to save memory.
        """
        return self.add(-other)

    def mul(self, other) -> "FourPoint":
        r"""
        Allows for the multiplication with a number, a numpy array or another FourPoint object. This is different from
        the `matmul` method, which is used for matrix multiplication.
        In the case the other object is a FourPoint object, we require that both objects have only one fermionic
        frequency dimension, such that :math:`A_{abcd}^{qv} * B_{dcef}^{qv'} = C_{abef}^{qvv'}`. This is needed to
        construct the full vertex, see Eq. (3.139) in my thesis. Returns the object in the half niw range.
        """
        if not isinstance(other, (int, float, complex, np.ndarray, FourPoint)):
            raise ValueError("Multiplication only supported with numbers, numpy arrays or FourPoint objects.")

        if not isinstance(other, FourPoint):
            copy = deepcopy(self)
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
        """
        if not isinstance(other, (FourPoint, LocalFourPoint, Interaction, LocalInteraction)):
            raise ValueError(f"Multiplication {type(self)} @ {type(other)} not supported.")

        if isinstance(other, (LocalInteraction, Interaction)):
            is_local = not isinstance(other, Interaction)
            q_prefix = "" if is_local else "q"

            self.compress_q_dimension()

            left_orbs = "abij"
            right_orbs = "jief"
            final_orbs = "abef"
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

            einsum_str = (
                f"qabcd{suffix_self},{q_prefix}dcef{suffix_other}->qabef{suffix_result}"
                if left_hand_side
                else f"{q_prefix}abcd{suffix_other},qdcef{suffix_self}->qabef{suffix_result}"
            )

            return FourPoint(
                np.einsum(einsum_str, self.mat, other.mat, optimize=True),
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
        """
        Loads a FourPoint object from a file. The file must be of type '.npy'.
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
        """
        Creates an identity (matrix in compound index notation is unity in the last two dimensions) for the
        FourPoint object.
        """
        if num_vn_dimensions not in (1, 2):
            raise ValueError("Invalid number of fermionic frequency dimensions.")
        full_shape = (nq_tot,) + (n_bands,) * 4 + (2 * niw + 1,) + (2 * niv,) * num_vn_dimensions
        compound_index_size = 2 * niv * n_bands**2
        mat = np.tile(np.eye(compound_index_size)[None, None, ...], (nq_tot, 2 * niw + 1, 1, 1))

        result = FourPoint(
            mat,
            nq=nq,
            num_vn_dimensions=num_vn_dimensions,
            has_compressed_q_dimension=True,
            frequency_notation=frequency_notation,
        ).to_full_indices(full_shape)
        if num_vn_dimensions == 1:
            result = result.take_vn_diagonal()
        return result.to_half_niw_range()

    @staticmethod
    def identity_like(other: "FourPoint") -> "FourPoint":
        """
        Creates an identity (matrix in compound index notation is unity in the last two dimensions) for the FourPoint
        object from the shape of another FourPoint object.
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

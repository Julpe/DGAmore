# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import gc
from abc import ABC
from copy import deepcopy
from enum import Enum

import numpy as np
import scipy as sp

from moldga import symmetry_reduction
from moldga.brillouin_zone import KGrid


class SpinChannel(Enum):
    """
    Enum for the different spin combinations.
    """

    DENS = "dens"
    MAGN = "magn"
    SING = "sing"
    TRIP = "trip"
    UU = "uu"
    UD = "ud"
    UD_BAR = "ud_bar"
    NONE = "none"


class FrequencyNotation(Enum):
    """
    Enum for the different frequency notations. Is interchangeable with the channel reducibility.
    """

    PH = "ph"
    PH_BAR = "ph_bar"
    PP = "pp"


class IHaveMat(ABC):
    """
    Abstract interface for classes that have a mat attribute. Adds a couple of convenience methods for matrix operations.
    Also adds a way to easily delete the underlying matrix to free memory.
    """

    _libc = None
    _malloc_trim_available = None

    def __init__(self, mat: np.ndarray):
        self.mat = mat
        self._original_shape = self.mat.shape

    @property
    def mat(self) -> np.ndarray:
        """
        Returns the underlying matrix, i.e. the numpy array.
        """
        return self._mat

    @mat.setter
    def mat(self, value: np.ndarray) -> None:
        """
        Sets the underlying matrix, i.e. the numpy array with a complex64 or complex128 type. If someone wants to use
        higher precision, they can always change it to complex128 themselves. Per default, we use complex64 to save
        memory.
        """
        if value is None:
            self._mat = None
            return
        self._mat = value.astype(np.complex64)

    @property
    def current_shape(self) -> tuple:
        """
        Keeps track of the current shape of the underlying numpy array.
        """
        return self._mat.shape

    @property
    def original_shape(self) -> tuple:
        """
        Keeps track of the previous shape of the underlying numpy array for any reshaping process.
        E.g., it is needed when reshaping it to compound indices where the original shape would have been lost otherwise.
        """
        return self._original_shape

    @original_shape.setter
    def original_shape(self, value) -> None:
        """
        Sets the original shape of the matrix. Keeps track of the previous shape of the underlying numpy array for any
        reshaping process. E.g., it is needed when reshaping it to compound indices where the original shape would have
        been lost otherwise.
        """
        self._original_shape = value

    @property
    def memory_usage_in_gb(self) -> float:
        """
        Returns the memory usage of the numpy array in GigaBytes (GB).
        """
        return self.mat.nbytes / (1024**3)

    def __mul__(self, other) -> "IHaveMat":
        """
        Allows for the multiplication with a number. A * n = B
        """
        if not isinstance(other, (int, float, complex)):
            raise ValueError("Multiplication only supported with numbers or numpy arrays.")

        copy = deepcopy(self)
        copy.mat *= other
        return copy

    def __rmul__(self, other) -> "IHaveMat":
        """
        Allows for the multiplication with a number. A * n = B
        """
        return self.__mul__(other)

    def __neg__(self) -> "IHaveMat":
        """
        Negates the matrix.
        """
        return self.__mul__(-1.0)

    def __truediv__(self, other) -> "IHaveMat":
        """
        Allows for the division with a number. A / n = B
        """
        if not isinstance(other, (int, float, complex)):
            raise ValueError("Division only supported with numbers.")
        return self.__mul__(1.0 / other)

    def __getitem__(self, item):
        """
        Returns the value at position [item]. Allows for the use of obj[...] instead of obj.mat[...].
        """
        return self.mat[item]

    def __setitem__(self, key, value):
        """
        Sets the value at position [key]. Allows for the use of obj[...] = value instead of obj.mat[...] = value.
        """
        self.mat[key] = value

    def __del__(self):
        """
        Deletes the underlying numpy array to free memory. However, the memory might not be immediately freed by Python
        if you call this method directly, e.g. by del obj, because the object might still be part of a reference cycle.
        In this case, the memory will be freed when the reference cycle is collected by the garbage collector, which
        can be forced by calling gc.collect() after del obj. However, it is generally recommended to use the free()
        method instead of del obj to explicitly free memory, since it also allows for the option to return freed heap
        memory back to the OS on Linux systems. If you want to use the context manager, you can use "with obj:" which
        will automatically free memory after the block.
        """
        self.free(True)

    def __enter__(self):
        """
        Context manager for the object. Allows for the use of "with obj:" to automatically free memory after the block.
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """
        Context manager for the object. Allows for the use of "with obj:" to automatically free memory after the block.
        """
        self.free(True)

    def free(self, trim: bool = False):
        """
        Explicitly releases the underlying numpy array. If True and running on Linux, attempts to return freed heap
        memory back to the OS using malloc_trim.
        """
        if self._mat is not None:
            self._mat = None

        gc.collect()

        if trim:
            self._malloc_trim()

    def update_original_shape(self):
        """
        Updates the original shape of the numpy array to the current array. This is often needed when the matrix
        is reshaped.
        """
        self.original_shape = self.current_shape

    def times(self, contraction: str, *args) -> np.ndarray:
        """
        Multiplies the matrices of multiple objects with the contraction specified and returns the result as a
        numpy array.
        """
        if not all(isinstance(obj, (IHaveMat, np.ndarray)) for obj in args):
            raise ValueError("Args has atleast one object with the wrong type. Allowed are [IHaveMat] or [np.ndarray].")
        return np.einsum(
            contraction, self.mat, *[obj.mat if isinstance(obj, IHaveMat) else obj for obj in args], optimize=True
        )

    def filter_small_values(self, threshold: float = 1e-12):
        """
        Sets all values in the underlying matrix to zero which are smaller than the given threshold in absolute value.
        This can be used to save memory and speed up calculations by setting very small values to zero.
        """
        self.mat[(np.abs(self.mat.real) < threshold) & (np.abs(self.mat.imag) < threshold)] = 0.0
        return self

    @classmethod
    def _malloc_trim(cls):
        """
        Returns unused heap memory to the OS using glibc malloc_trim.
        Only available on Linux systems.
        """
        import os

        if cls._malloc_trim_available is False:
            return

        if cls._malloc_trim_available is None:
            if os.name != "posix" or not os.path.exists("/proc"):
                cls._malloc_trim_available = False
                return

            try:
                import ctypes

                cls._libc = ctypes.CDLL("libc.so.6")
                cls._malloc_trim_available = True
            except Exception:
                cls._malloc_trim_available = False
                return

        try:
            cls._libc.malloc_trim(0)
        except Exception:
            pass


class IHaveChannel(ABC):
    """
    Abstract interface for classes that have a channel attribute. Adds a property for the spin channel and the
    frequency notation.
    """

    def __init__(
        self, channel: SpinChannel = SpinChannel.NONE, frequency_notation: FrequencyNotation = FrequencyNotation.PH
    ):
        self._channel = channel
        self._frequency_notation = frequency_notation

    @property
    def channel(self) -> SpinChannel:
        """
        Returns the spin channel of the object. For a set of available channels, see the enum `SpinChannel`.
        """
        return self._channel

    @channel.setter
    def channel(self, value: SpinChannel) -> None:
        """
        Sets the spin channel of the object. For a set of available channels, see the enum `SpinChannel`.
        """
        if not isinstance(value, SpinChannel):
            raise ValueError("Channel must be of type SpinChannel.")
        self._channel = value

    def set_channel(self, channel: SpinChannel):
        """
        Sets the spin channel of the object. For a set of available channels, see the enum `SpinChannel`.
        """
        self.channel = channel
        return self

    @property
    def frequency_notation(self) -> FrequencyNotation:
        """
        Returns the frequency notation (not the channel reducibility) of the object.
        For a set of available frequency notations, see the enum `FrequencyNotation`.
        """
        return self._frequency_notation

    @frequency_notation.setter
    def frequency_notation(self, value: FrequencyNotation) -> None:
        """
        Sets the frequency notation of the object. For a set of available frequency notations,
        see the enum `FrequencyNotation`.
        """
        if not isinstance(value, FrequencyNotation):
            raise ValueError("Frequency notation must be of type FrequencyNotation.")
        self._frequency_notation = value

    def set_frequency_notation(self, value: FrequencyNotation):
        """
        Sets the frequency notation of the object. For a set of available frequency notations,
        see the enum `FrequencyNotation`.
        """
        self.frequency_notation = value
        return self


class IAmNonLocal(IHaveMat, ABC):
    """
    Abstract interface for objects that are momentum dependent. Since we focus on ladder objects, we do not
    need more than one momentum dimension for one- and two-particle quantities.
    """

    def __init__(self, mat: np.ndarray, nq: tuple[int, int, int], has_compressed_q_dimension: bool = False):
        IHaveMat.__init__(self, mat)
        self._nq = nq
        self._has_compressed_q_dimension = has_compressed_q_dimension

    @property
    def nq(self) -> tuple[int, int, int]:
        """
        Returns the number of momenta in the object. This should always be equal to the k- or q-point grid of the lattice.
        """
        return self._nq

    @property
    def nq_tot(self) -> int:
        """
        Returns the total number of momenta in the object. This might be lower than np.prod(self.nq) if the object is
        currently saved in the irreducible Brillouin zone.
        """
        return np.prod(self.nq).astype(int) if not self.has_compressed_q_dimension else self.original_shape[0]

    @property
    def has_compressed_q_dimension(self) -> bool:
        """
        Returns whether the underlying matrix has a compressed momentum dimension [q,...] or not [qx,qy,qz,...].
        """
        return self._has_compressed_q_dimension

    @property
    def n_bands(self) -> int:
        """
        Returns the number of bands in the nonlocal two- or four-point object. If the object has a compressed momentum
        dimension, the array has dimension [q, o1, o2, ... ], otherwise it has dimension [qx, qy, qz, o1, o2, ... ].
        """
        return self.original_shape[1] if self.has_compressed_q_dimension else self.original_shape[3]

    def shift_k_by_q(self, q: tuple | list[int] = (0, 0, 0)):
        """
        Shifts all momenta by the given values and returns a copy of the object with a decompressed momentum dimension.
        """
        copy = deepcopy(self)

        compress = False
        if copy.has_compressed_q_dimension:
            compress = True
            copy.decompress_q_dimension()

        copy.mat = np.roll(copy.mat, [-i for i in q], axis=(0, 1, 2))

        if compress:
            copy.compress_q_dimension()
        return copy

    def shift_k_by_pi(self):
        r"""
        Shifts all momenta by :math:`\pi` and returns the object with a decompressed momentum dimension.
        """
        copy = deepcopy(self)

        compress = False
        if copy.has_compressed_q_dimension:
            compress = True
            copy.decompress_q_dimension()

        shifts = np.array(copy.current_shape[:3]) // 2
        copy.mat = np.roll(copy.mat, shift=shifts, axis=(0, 1, 2))

        if compress:
            copy.compress_q_dimension()
        return copy

    def compress_q_dimension(self):
        """
        Converts the object from [qx,qy,qz,...] to [q,...], where len(q) = qx*qy*qz.
        """
        if self.has_compressed_q_dimension:
            return self

        self.mat = self.mat.reshape((self.nq_tot, *self.original_shape[3:]))
        self._has_compressed_q_dimension = True
        self.update_original_shape()
        return self

    def decompress_q_dimension(self):
        """
        Converts the object from [q,...] to [qx,qy,qz,...], where len(q) = qx*qy*qz.
        """
        if not self.has_compressed_q_dimension:
            return self

        self.mat = self.mat.reshape((*self.nq, *self.current_shape[1:]))
        self._has_compressed_q_dimension = False
        self.update_original_shape()
        return self

    def reduce_q(self, q_list: np.ndarray):
        r"""
        Reduces the object to the given list of momenta and returns a copy with a compressed momentum dimension. Acts
        like a filter. Makes it possible to use e.g. only the :math:`\vec{q}=0` component of a non-local object or
        filter the irreducible Brillouin zone from an object in the full Brillouin zone.
        """
        copy = deepcopy(self)

        if copy.has_compressed_q_dimension:
            copy.decompress_q_dimension()

        indices = np.indices(copy.current_shape[:3])
        mask = np.zeros(copy.current_shape[:3], dtype=bool)
        mask |= np.any(np.all(indices == np.array(q_list)[:, :, None, None, None], axis=1), axis=0)
        copy.mat = copy.mat[mask]

        copy.update_original_shape()
        copy._has_compressed_q_dimension = True
        return copy

    def find_q(self, q: tuple[int, int, int] = (0, 0, 0)):
        r"""
        Find the matrix element for a single momentum :math:`\vec{q}` and returns a compressed copy.
        Raises a ValueError if no element is found.
        """
        q_arr = np.atleast_2d(np.array(q, dtype=int))
        result = deepcopy(self).reduce_q(q_arr)
        result._nq = (1, 1, 1)

        if getattr(result, "mat", None) is None or result.mat.size == 0 or result.current_shape[0] == 0:
            raise ValueError("No matrix element found for the given momentum.")

        return result

    def filter_q_index(self, index: int = 0):
        r"""
        Filters the object to the given index of the momentum dimension and returns a copy. Acts like a filter.
        Makes it possible to use e.g. only the first component of a non-local object.
        """
        if not self.has_compressed_q_dimension:
            self.compress_q_dimension()

        copy = deepcopy(self)
        copy.mat = copy.mat[index][None, ...]
        copy.update_original_shape()
        copy._nq = (1, 1, 1)
        return copy

    def map_to_full_bz(self, k_grid: KGrid, nq: tuple = None):
        """
        Maps to full BZ using k_grid's inverse map and precomputed orbital rotation tensors.
        Call k_grid.set_orbital_rotations() before this if orbital mixing is needed,
        otherwise identity is assumed for all k-points.
        """
        return self._map_to_full_bz(k_grid, 4, nq)

    def _map_to_full_bz(self, k_grid: KGrid, num_orbital_dimensions: int, nq: tuple = None):
        """
        Maps the object from the irreducible to the full Brillouin zone.

        First expands the compressed IBZ momentum dimension to the full BZ by copying each IBZ
        value to all its symmetry-equivalent FBZ images via ``k_grid.irrk_inv``. Then applies the
        per-k orbital transformation stored on ``k_grid`` by ``specify_auto_symmetries(hk)``.

        The orbital transformation follows the ket/bra convention of the operator ordering
        G_abcd := <T[c_a c†_b c_c c†_d]> -- annihilation indices (positions 1, 3) transform with
        U, creation indices (positions 2, 4) with U^dagger -- combined with a per-k antisymmetry
        sign ``sigma_k`` and an optional complex conjugation ``conj_k``:

            2-index : M_ab(k)   = sigma_k     * U_aa' [M_a'b'(k_rep)]^{[*conj_k]} U^dag_b'b
            4-index : M_abcd(k) = sigma_k^2 * U_aa' [M_a'b'c'd'(k_rep)]^{[*conj_k]} U^dag_b'b U_cc' U^dag_d'd

        If ``k_grid`` is not yet in auto mode (``specify_auto_symmetries`` has not been called),
        only the momentum expansion is performed and orbital indices are left unchanged.
        """
        if not self.has_compressed_q_dimension:
            raise ValueError("Mapping to full BZ only possible for compressed momentum dimension.")

        assert num_orbital_dimensions in (2, 4), "Number of orbital dimensions must be 2 or 4."

        if nq is not None:
            self._nq = nq

        # Expand IBZ -> FBZ via the standard irrk_inv map (no orbital action yet).
        flat_inv = k_grid.irrk_inv.ravel()
        out_shape = (np.prod(self.nq), *self.current_shape[1:])
        expanded = np.empty(out_shape, dtype=self.mat.dtype)
        np.take(self.mat, flat_inv, axis=0, out=expanded)
        self.mat = expanded

        # Apply per-k orbital transformation if auto-mode data is present.
        if getattr(k_grid, "is_auto", False):
            self.mat = symmetry_reduction.apply_auto_orbital_transform(
                self.mat,
                us=k_grid._auto_us.reshape(np.prod(k_grid.nk), *k_grid._auto_us.shape[3:]),
                sigmas=k_grid._auto_sigmas.reshape(-1),
                conjs=k_grid._auto_conjs.reshape(-1),
                num_orbital_dimensions=num_orbital_dimensions,
            )

        self.update_original_shape()
        return self

    def fft(self, copy: bool = True):
        """
        Performs a discrete forward Fourier transform over the momentum dimension and returns a copy if specified.
        """
        if copy:
            copy = deepcopy(self)

            compress = False
            if copy.has_compressed_q_dimension:
                compress = True
                copy.decompress_q_dimension()

            sp.fft.fftn(copy.mat, axes=(0, 1, 2), overwrite_x=True)
            return copy.compress_q_dimension() if compress else copy

        compress = False
        if self.has_compressed_q_dimension:
            compress = True
            self.decompress_q_dimension()
        sp.fft.fftn(self.mat, axes=(0, 1, 2), overwrite_x=True)
        return self.compress_q_dimension() if compress else self

    def ifft(self, copy: bool = True):
        """
        Performs a discrete inverse Fourier transform over the momentum dimension and returns a copy if specified.
        """
        if copy:
            copy = deepcopy(self)

            compress = False
            if copy.has_compressed_q_dimension:
                compress = True
                copy.decompress_q_dimension()

            sp.fft.ifftn(copy.mat, axes=(0, 1, 2), overwrite_x=True)
            return copy.compress_q_dimension() if compress else copy

        compress = False
        if self.has_compressed_q_dimension:
            compress = True
            self.decompress_q_dimension()
        sp.fft.ifftn(self.mat, axes=(0, 1, 2), overwrite_x=True)
        return self.compress_q_dimension() if compress else self

    def flip_momentum_axis(self, copy: bool = True):
        r"""
        Flips the momentum axis :math:`F^{q}\to F^{-q}` of the object and returns a copy if specified.
        """
        if copy:
            copy = deepcopy(self)

            compress = False
            if copy.has_compressed_q_dimension:
                compress = True
                copy.decompress_q_dimension()

            copy.mat = np.roll(np.flip(copy.mat, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
            return copy.compress_q_dimension() if compress else copy

        compress = False
        if self.has_compressed_q_dimension:
            compress = True
            self.decompress_q_dimension()

        self.mat = np.roll(np.flip(self.mat, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
        return self.compress_q_dimension() if compress else self

    def _align_q_dimensions_for_operations(self, other: "IAmNonLocal"):
        """
        Helper method which adapts the frequency dimensions of two non-local objects to fit each other for
        addition or multiplication.
        """
        if not self.has_compressed_q_dimension and other.has_compressed_q_dimension:
            self.compress_q_dimension()
        if not other.has_compressed_q_dimension and self.has_compressed_q_dimension:
            other = other.compress_q_dimension()
        return other

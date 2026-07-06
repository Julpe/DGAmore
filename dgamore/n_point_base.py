# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Foundational mixins for every physical quantity in the code. :class:`IHaveMat` owns the underlying numpy array
(stored in the global precision :data:`DTYPE`) and the memory-management/arithmetic helpers; :class:`IHaveChannel`
adds the spin channel and frequency notation; :class:`IAmNonLocal` adds a single momentum dimension with the
compressed/decompressed and irreducible/full-BZ bookkeeping. Concrete classes (Green's function, self-energy,
vertices, interaction, gap function) compose these mixins.
"""

import gc
from abc import ABC
from copy import deepcopy
from enum import Enum

import numpy as np
import scipy as sp

from dgamore.brillouin_zone import KGrid

# Global storage precision for all N-point quantities. Switch this single constant to np.complex128 for
# double precision (at the cost of doubling the memory footprint). All objects deriving from IHaveMat store
# their underlying matrix in this dtype; it is also exposed as the class attribute IHaveMat.DTYPE.
DTYPE = np.complex64


class SpinChannel(Enum):
    """
    Enum for the different spin combinations.

    :cvar DENS: Density (charge) channel.
    :cvar MAGN: Magnetic (spin) channel.
    :cvar SING: Singlet pairing channel.
    :cvar TRIP: Triplet pairing channel.
    :cvar UU: Up-up spin combination.
    :cvar UD: Up-down spin combination.
    :cvar UD_BAR: Crossed (transverse) up-down spin combination.
    :cvar NONE: No / unspecified channel (e.g. a bare, not-yet-projected quantity).
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

    :cvar PH: Particle-hole notation.
    :cvar PH_BAR: Transverse (crossed) particle-hole notation.
    :cvar PP: Particle-particle notation.
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

    # Upper bound (in bytes) on the transient temporaries that filter_small_values may materialize per chunk. The
    # mask is evaluated in slices so the full-size |real|/|imag|/boolean temporaries are never built all at once.
    _FILTER_CHUNK_BYTES = 256 * 1024**2

    # Storage precision for the underlying matrix. Defaults to the module-level ``DTYPE`` (complex64).
    # Override the module constant (or this attribute) to switch the whole code base to double precision.
    DTYPE = DTYPE

    def __init__(self, mat: np.ndarray):
        """
        Stores the array and records its original shape.

        :param mat: The underlying numpy array; it is coerced to the global storage precision :data:`DTYPE`.
        """
        self.mat = mat
        self._original_shape = self.mat.shape

    @property
    def mat(self) -> np.ndarray:
        """
        Returns the underlying matrix, i.e. the numpy array.

        :return: The underlying numpy array (or None if it has been freed).
        """
        return self._mat

    @mat.setter
    def mat(self, value: np.ndarray) -> None:
        """
        Sets the underlying matrix, i.e. the numpy array, coerced to the global storage precision ``DTYPE``
        (complex64 by default to save memory). To switch the whole code base to higher precision, change the
        module-level ``DTYPE`` constant (or ``IHaveMat.DTYPE``) to e.g. np.complex128.

        Aliasing contract: ``astype(..., copy=False)`` means an array that is already of dtype ``DTYPE`` is
        stored **by reference, not copied**. Constructing an object from (or assigning) a view or an array that
        is still referenced elsewhere therefore aliases it -- a subsequent in-place mutation (e.g. ``obj[...] =``,
        ``filter_small_values``, a ``copy=False`` transformation) would affect the source. ``.copy()`` the input
        first if the source must stay independent.

        :param value: The new numpy array (or None to free the storage).
        """
        if value is None:
            self._mat = None
            return
        self._mat = value.astype(DTYPE, copy=False)

    @property
    def current_shape(self) -> tuple:
        """
        Keeps track of the current shape of the underlying numpy array.

        :return: The current shape of ``mat``.
        """
        return self._mat.shape

    @property
    def original_shape(self) -> tuple:
        """
        Keeps track of the previous shape of the underlying numpy array for any reshaping process.
        E.g., it is needed when reshaping it to compound indices where the original shape would have been lost otherwise.

        :return: The stored original (pre-reshape) shape.
        """
        return self._original_shape

    @original_shape.setter
    def original_shape(self, value) -> None:
        """
        Sets the original shape of the matrix. Keeps track of the previous shape of the underlying numpy array for any
        reshaping process. E.g., it is needed when reshaping it to compound indices where the original shape would have
        been lost otherwise.

        :param value: The shape tuple to store as the original shape.
        """
        self._original_shape = value

    @property
    def memory_usage_in_gb(self) -> float:
        """
        Returns the memory usage of the numpy array in GigaBytes (GB).

        :return: Memory footprint of ``mat`` in GB.
        """
        return self.mat.nbytes / (1024**3)

    def copy(self) -> "IHaveMat":
        """
        Returns an independent deep copy of the object -- a thin wrapper around :func:`copy.deepcopy` used in place of
        a bare ``deepcopy(obj)`` throughout the package. Mutating the returned object (including its ``mat``) does not
        affect the original.

        :return: A deep copy of ``self``.
        """
        return deepcopy(self)

    def __mul__(self, other) -> "IHaveMat":
        """
        Multiplies the object by a scalar number, returning a new (deep-copied) object.

        :param other: A scalar (int, float or complex) to multiply with.
        :return: A new object holding ``self * other``.
        :raises ValueError: If ``other`` is not a number.
        """
        if not isinstance(other, (int, float, complex)):
            raise ValueError("Multiplication only supported with numbers.")

        copy = self.copy()
        copy.mat *= other
        return copy

    def __rmul__(self, other) -> "IHaveMat":
        """
        Reflected scalar multiplication ``other * self``; see :meth:`__mul__`.

        :param other: A scalar to multiply with.
        :return: A new object holding ``self * other``.
        """
        return self.__mul__(other)

    def __neg__(self) -> "IHaveMat":
        """
        Negates the matrix (``-self``); see :meth:`__mul__`.

        :return: A new object holding ``-self``.
        """
        return self.__mul__(-1.0)

    def __truediv__(self, other) -> "IHaveMat":
        """
        Divides the object by a scalar number; see :meth:`__mul__`.

        :param other: A scalar (int, float or complex) to divide by.
        :return: A new object holding ``self / other``.
        :raises ValueError: If ``other`` is not a number.
        """
        if not isinstance(other, (int, float, complex)):
            raise ValueError("Division only supported with numbers.")
        return self.__mul__(1.0 / other)

    def scale(self, factor, copy: bool = False) -> "IHaveMat":
        """
        Multiplies the matrix by a scalar. The in-place branch (``copy=False``, the default here) mutates and
        returns ``self`` without allocating a copy -- the memory-lean counterpart of the non-destructive
        ``*``/:meth:`__mul__` operator (which already covers the copying case). Used to fold a scalar prefactor
        into an in-place accumulation (e.g. the self-energy-kernel assembly in :mod:`dgamore.nonlocal_sde`)
        without a throwaway scaled copy. ``copy=True`` returns a scaled deep copy and leaves ``self`` untouched.

        :param factor: A scalar (int, float or complex) to multiply with.
        :param copy: If True, return a scaled deep copy; if False (default), scale ``self.mat`` in place.
        :return: The scaled object (``self`` when ``copy=False``).
        :raises ValueError: If ``factor`` is not a number.
        """
        if not isinstance(factor, (int, float, complex)):
            raise ValueError("Multiplication only supported with numbers.")
        if copy:
            return self.copy().scale(factor, copy=False)
        self.mat *= factor
        return self

    def __getitem__(self, item):
        """
        Indexing shortcut: ``obj[item]`` is equivalent to ``obj.mat[item]``.

        :param item: Any numpy-compatible index.
        :return: The indexed view/value of ``mat``.
        """
        return self.mat[item]

    def __setitem__(self, key, value):
        """
        Assignment shortcut: ``obj[key] = value`` is equivalent to ``obj.mat[key] = value`` (in-place).

        :param key: Any numpy-compatible index.
        :param value: The value to assign.
        """
        self.mat[key] = value

    def __del__(self):
        """
        Deletes the underlying numpy array to free memory. However, the memory might not be immediately freed by Python
        if you call this method directly, e.g. by del obj, because the object might still be part of a reference cycle.
        In this case, the memory will be freed when the reference cycle is collected by the garbage collector, which
        can be forced by calling gc.collect() after del obj. However, it is generally recommended to use the free()
        method instead of del obj to explicitly free memory. The destructor does **not** trim the heap (it calls
        ``free()`` with the default ``trim=False``): ``malloc_trim`` walks the whole heap, so trimming on every object
        destruction in the tight per-q/per-w loops is wasted work for no memory benefit (the buffer is released
        regardless). Return freed pages to the OS explicitly with ``free(trim=True)`` at coarse, heavy boundaries, or
        via the ``with obj:`` context manager.
        """
        self.free()

    def __enter__(self):
        """
        Context manager entry; see :meth:`__exit__`.

        :return: ``self``.
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """
        Context manager exit: frees the underlying array (and trims heap memory) when leaving a ``with`` block.

        :param exc_type: Exception type if one was raised inside the block, else None.
        :param exc: Exception instance if one was raised, else None.
        :param tb: Traceback if an exception was raised, else None.
        """
        self.free(True)

    def free(self, trim: bool = False):
        """
        Explicitly releases the underlying numpy array. If True and running on Linux, attempts to return freed heap
        memory back to the OS using malloc_trim.

        ``trim`` defaults to **False** on purpose: ``malloc_trim`` walks the entire heap, so it should be requested
        only at coarse, heavy boundaries (e.g. after a large object is released at the end of a self-consistency
        iteration), never on every release in a hot loop.

        :param trim: If True, additionally attempt to return freed heap memory to the OS (Linux only).
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

    def _clone_without_mat(self):
        """
        Returns a deep copy of the object that carries all metadata but **not** the underlying array (``mat`` is
        ``None`` on the clone). This avoids the wasteful duplication of ``mat`` in transformation methods that
        deep-copy ``self`` only to overwrite ``copy.mat`` with a freshly computed array immediately afterwards. The
        caller is expected to assign a new array to ``clone.mat`` before using it.

        :return: A deep copy of ``self`` whose ``mat`` is ``None``.
        """
        mat = self._mat
        self._mat = None
        try:
            clone = self.copy()
        finally:
            self._mat = mat
        return clone

    def times(self, contraction: str, *args) -> np.ndarray:
        """
        Multiplies the matrices of multiple objects with the contraction specified and returns the result as a
        numpy array.

        :param contraction: An einsum contraction string applied to ``self.mat`` and the operands in ``args``.
        :param args: Further operands, each either an :class:`IHaveMat` or a numpy array.
        :return: The contracted numpy array.
        :raises ValueError: If any operand is neither an :class:`IHaveMat` nor a numpy array.
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

        The mask is evaluated in chunks so the full-size ``|real|``/``|imag|`` and boolean temporaries (~0.75x the
        array) are never materialized all at once -- on a very large array that transient spike would otherwise
        dominate the footprint right when ``mat`` is at its largest. Chunking is also faster (cache-resident
        temporaries, fewer page faults). A small array is filtered in a single pass to avoid the per-chunk overhead.

        :param threshold: Values whose real and imaginary parts are both below this magnitude are zeroed (in place).
        :return: ``self`` (for chaining).
        """
        mat = self.mat
        # Per-element transient cost of the mask: the two float halves (== itemsize total) plus two boolean arrays.
        temp_bytes_per_elem = mat.dtype.itemsize + 2
        budget_bytes = self._FILTER_CHUNK_BYTES  # cap the per-chunk temporaries (class attribute; patchable in tests)

        def _zero_below(view: np.ndarray) -> None:
            """Zeroes the entries of ``view`` whose real and imag parts are both below ``threshold`` (in place)."""
            view[(np.abs(view.real) < threshold) & (np.abs(view.imag) < threshold)] = 0.0

        if mat.flags["C_CONTIGUOUS"]:
            # A flat view is free for a C-contiguous array; chunk it along the single flat axis.
            flat = mat.reshape(-1)
            n = flat.shape[0]
            step = max(1, budget_bytes // temp_bytes_per_elem)
            if step >= n:
                _zero_below(flat)
            else:
                for i in range(0, n, step):
                    _zero_below(flat[i : i + step])
        else:
            # Non-contiguous: reshape(-1) would copy the whole array, so chunk axis 0 (slices stay views).
            n = mat.shape[0]
            rest_elems = max(1, mat.size // n)
            step = max(1, budget_bytes // (rest_elems * temp_bytes_per_elem))
            if step >= n:
                _zero_below(mat)
            else:
                for i in range(0, n, step):
                    _zero_below(mat[i : i + step])
        return self

    @classmethod
    def _malloc_trim(cls):
        """
        Returns unused heap memory to the OS using glibc malloc_trim. Only available on Linux systems; a no-op
        elsewhere. The availability is detected once and cached on the class.
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
        """
        Stores the spin channel and frequency notation of the object.

        :param channel: Spin channel of the object (see :class:`SpinChannel`).
        :param frequency_notation: Frequency notation of the object (see :class:`FrequencyNotation`).
        """
        self._channel = channel
        self._frequency_notation = frequency_notation

    @property
    def channel(self) -> SpinChannel:
        """
        Returns the spin channel of the object. For a set of available channels, see the enum :class:`SpinChannel`.

        :return: The current spin channel.
        """
        return self._channel

    @channel.setter
    def channel(self, value: SpinChannel) -> None:
        """
        Sets the spin channel of the object. For a set of available channels, see the enum :class:`SpinChannel`.

        :param value: The spin channel to set.
        :raises ValueError: If ``value`` is not a :class:`SpinChannel`.
        """
        if not isinstance(value, SpinChannel):
            raise ValueError("Channel must be of type SpinChannel.")
        self._channel = value

    def set_channel(self, channel: SpinChannel):
        """
        Sets the spin channel of the object (chainable). For a set of available channels, see the enum
        :class:`SpinChannel`.

        :param channel: The spin channel to set.
        :return: ``self`` (for chaining).
        """
        self.channel = channel
        return self

    @property
    def frequency_notation(self) -> FrequencyNotation:
        """
        Returns the frequency notation (not the channel reducibility) of the object.
        For a set of available frequency notations, see the enum :class:`FrequencyNotation`.

        :return: The current frequency notation.
        """
        return self._frequency_notation

    @frequency_notation.setter
    def frequency_notation(self, value: FrequencyNotation) -> None:
        """
        Sets the frequency notation of the object. For a set of available frequency notations,
        see the enum :class:`FrequencyNotation`.

        :param value: The frequency notation to set.
        :raises ValueError: If ``value`` is not a :class:`FrequencyNotation`.
        """
        if not isinstance(value, FrequencyNotation):
            raise ValueError("Frequency notation must be of type FrequencyNotation.")
        self._frequency_notation = value

    def set_frequency_notation(self, value: FrequencyNotation):
        """
        Sets the frequency notation of the object (chainable). For a set of available frequency notations,
        see the enum :class:`FrequencyNotation`.

        :param value: The frequency notation to set.
        :return: ``self`` (for chaining).
        """
        self.frequency_notation = value
        return self


class IAmNonLocal(IHaveMat, ABC):
    """
    Abstract interface for objects that are momentum dependent. Since we focus on ladder objects, we do not
    need more than one momentum dimension for one- and two-particle quantities.
    """

    def __init__(self, mat: np.ndarray, nq: tuple[int, int, int], has_compressed_q_dimension: bool = False):
        """
        Stores the array, the momentum-grid size and the momentum-layout flag.

        :param mat: The underlying numpy array with one (compressed) or three (decompressed) leading momentum axes.
        :param nq: Number of momenta per spatial direction ``(nqx, nqy, nqz)``.
        :param has_compressed_q_dimension: Whether the momentum is stored as a single compressed axis ``[q, ...]``
            (True) or as three separate axes ``[qx, qy, qz, ...]`` (False).
        """
        IHaveMat.__init__(self, mat)
        self._nq = nq
        self._has_compressed_q_dimension = has_compressed_q_dimension

    @property
    def nq(self) -> tuple[int, int, int]:
        """
        Returns the number of momenta in the object. This should always be equal to the k- or q-point grid of the lattice.

        :return: The number of momenta per spatial direction ``(nqx, nqy, nqz)``.
        """
        return self._nq

    @property
    def nq_tot(self) -> int:
        """
        Returns the total number of momenta in the object. This might be lower than np.prod(self.nq) if the object is
        currently saved in the irreducible Brillouin zone.

        :return: The total number of stored momenta.
        """
        return (
            int(self.nq[0] * self.nq[1] * self.nq[2]) if not self.has_compressed_q_dimension else self.original_shape[0]
        )

    @property
    def has_compressed_q_dimension(self) -> bool:
        """
        Returns whether the underlying matrix has a compressed momentum dimension [q,...] or not [qx,qy,qz,...].

        :return: True if the momentum is stored as a single compressed axis.
        """
        return self._has_compressed_q_dimension

    @property
    def n_bands(self) -> int:
        """
        Returns the number of bands in the nonlocal two- or four-point object. If the object has a compressed momentum
        dimension, the array has dimension [q, o1, o2, ... ], otherwise it has dimension [qx, qy, qz, o1, o2, ... ].

        :return: The number of bands (orbitals).
        """
        return self.original_shape[1] if self.has_compressed_q_dimension else self.original_shape[3]

    def shift_k_by_q(self, q: tuple | list[int] = (0, 0, 0)):
        r"""
        Shifts all momenta by the given values and returns a copy of the object with a decompressed momentum dimension.

        :param q: The momentum shift :math:`\vec{q}` as integer grid offsets ``(qx, qy, qz)``.
        :return: A copy shifted by :math:`\vec{q}`, in the same momentum-compression state as ``self``.
        """
        copy = self._clone_without_mat()
        copy.mat = self.mat  # shared reference; np.roll below allocates a fresh array, leaving self.mat untouched

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

        :return: A copy shifted by :math:`\pi` along every momentum axis, in the same compression state as ``self``.
        """
        copy = self._clone_without_mat()
        copy.mat = self.mat  # shared reference; np.roll below allocates a fresh array, leaving self.mat untouched

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

        :return: ``self`` with a compressed momentum dimension (a no-op if already compressed).
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

        :return: ``self`` with a decompressed momentum dimension (a no-op if already decompressed).
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

        :param q_list: Array of momentum grid indices to keep, shape ``[3, n_q]`` (one column per momentum).
        :return: A copy reduced to ``q_list``, with a compressed momentum dimension.
        """
        copy = self._clone_without_mat()
        copy.mat = self.mat  # shared reference; the boolean-mask indexing below copies, leaving self.mat untouched

        if copy.has_compressed_q_dimension:
            copy.decompress_q_dimension()

        # Build the keep-mask via flat indices instead of a (n_q, 3, nqx, nqy, nqz) broadcast comparison. ``q_list``
        # has shape [n_q, 3] (one momentum per row). Out-of-range momenta are silently dropped (matching the previous
        # behaviour of yielding no match for them).
        shape3 = copy.current_shape[:3]
        q_arr = np.asarray(q_list)
        if q_arr.ndim != 2 or q_arr.shape[1] != 3:
            raise ValueError("q_list must have shape [n_q, 3].")
        coords = (q_arr[:, 0], q_arr[:, 1], q_arr[:, 2])
        in_range = np.all([(c >= 0) & (c < s) for c, s in zip(coords, shape3)], axis=0)
        flat_idx = np.ravel_multi_index(tuple(c[in_range] for c in coords), shape3)
        mask = np.zeros(int(np.prod(shape3)), dtype=bool)
        mask[flat_idx] = True
        copy.mat = copy.mat.reshape((mask.size, *copy.current_shape[3:]))[mask]

        copy.update_original_shape()
        copy._has_compressed_q_dimension = True
        return copy

    def find_q(self, q: tuple[int, int, int] = (0, 0, 0)):
        r"""
        Finds the matrix element for a single momentum :math:`\vec{q}` and returns a compressed copy.

        :param q: The momentum grid index ``(qx, qy, qz)`` to select.
        :return: A compressed copy containing only the requested momentum (``nq = (1, 1, 1)``).
        :raises ValueError: If no matrix element is found for the given momentum.
        """
        q_arr = np.atleast_2d(np.array(q, dtype=int))
        result = self.reduce_q(q_arr)  # reduce_q already returns an independent copy; no extra deepcopy needed
        result._nq = (1, 1, 1)

        if getattr(result, "mat", None) is None or result.mat.size == 0 or result.current_shape[0] == 0:
            raise ValueError("No matrix element found for the given momentum.")

        return result

    def filter_q_index(self, index: int = 0):
        r"""
        Filters the object to the given index of the momentum dimension and returns a copy. Acts like a filter.
        Makes it possible to use e.g. only the n-th q-component of a non-local object.

        :param index: The position along the compressed momentum axis to keep.
        :return: A compressed copy containing only that single momentum (``nq = (1, 1, 1)``).
        """
        if not self.has_compressed_q_dimension:
            self.compress_q_dimension()

        # Copy only the single requested q-slice instead of deep-copying the whole multi-q array; ``.copy()`` so the
        # returned object does not keep the full parent array alive via a view.
        copy = self._clone_without_mat()
        copy.mat = self.mat[index][None, ...].copy()
        copy.update_original_shape()
        copy._nq = (1, 1, 1)
        return copy

    def q_mean(self):
        r"""
        Averages over the momentum dimension and returns a copy of the object with nq = (1,1,1).

        :return: A copy holding the momentum average (``nq = (1, 1, 1)``).
        """
        copy = self._clone_without_mat()
        if self.has_compressed_q_dimension:
            copy.mat = np.mean(self.mat, axis=0)[None, ...]
        else:
            copy.mat = np.mean(self.mat, axis=(0, 1, 2))[None, None, None, ...]
        copy.update_original_shape()
        copy._nq = (1, 1, 1)
        return copy

    def interpolate_q_grid(self, nq_new: tuple[int, int, int], copy: bool = True):
        r"""
        Re-samples the object onto a new Gamma-centered momentum grid (:math:`\Gamma` at index 0)
        and returns a copy if specified. Up- and downsampling are both supported, per axis and in
        any combination. Returns in the same momentum compression state as the input.

        For each axis the method chooses automatically:
          * if the target is a sub-lattice of the source (``n_new`` divides ``n_old``), the exact
            :math:`\Sigma(\vec{q})` at the retained points is obtained by striding -- no band-limiting;
          * otherwise (upsampling or incommensurate downsampling) a band-limited Fourier
            interpolation (FFT -> pad/truncate spectrum -> iFFT, with symmetric Nyquist handling) is used.

        Requires a full-BZ object. A compressed dimension that is not the full BZ (e.g. IBZ-reduced)
        is rejected.

        :param nq_new: Target number of momenta per spatial direction ``(nqx, nqy, nqz)``.
        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: The re-gridded object in the same momentum-compression state as the input.
        :raises ValueError: If a target size is not a positive integer, or the object is compressed but not full-BZ.
        """
        copy = self.copy() if copy else self

        nq_new = tuple(int(n) for n in nq_new)
        if any(n < 1 for n in nq_new):
            raise ValueError("Target grid sizes must be positive integers.")
        if nq_new == copy.nq:
            return copy

        compress = False
        if copy.has_compressed_q_dimension:
            if copy.current_shape[0] != int(np.prod(copy.nq)):
                raise ValueError("Re-gridding requires a full-BZ object.")
            compress = True
            copy.decompress_q_dimension()

        for axis, n_new in enumerate(nq_new):
            n_old = copy.current_shape[axis]
            if n_new == n_old:
                continue
            if n_new < n_old and n_old % n_new == 0:
                copy.mat = np.take(copy.mat, np.arange(0, n_old, n_old // n_new), axis=axis)
            else:
                copy.mat = sp.signal.resample(copy.mat, n_new, axis=axis)

        copy._nq = nq_new
        copy.update_original_shape()

        return copy.compress_q_dimension() if compress else copy

    def map_to_full_bz(self, k_grid: "KGrid", nq: tuple = None):
        """
        Maps to full BZ using k_grid's inverse map and precomputed orbital rotation tensors.

        :param k_grid: The momentum grid carrying the irreducible-to-full-BZ map and per-k orbital rotations.
        :param nq: Optional override for the number of momenta; if None the object's own ``nq`` is used.
        :return: ``self`` expanded to the full BZ (four orbital dimensions transformed).
        """
        return self._map_to_full_bz(k_grid, 4, nq)

    def _map_to_full_bz(self, k_grid: "KGrid", num_orbital_dimensions: int, nq: tuple = None):
        r"""
        Maps the object from the irreducible to the full Brillouin zone.

        First expands the compressed IBZ momentum dimension to the full BZ by copying each IBZ
        value to all its symmetry-equivalent FBZ images via ``k_grid.irrk_inv``. Then applies the
        per-k orbital transformation stored on ``k_grid`` by ``specify_auto_symmetries(hk)``.

        The orbital transformation follows the ket/bra convention of the operator ordering
        :math:`G_{1234} := \langle T[c_1 c^\dagger_2 c_3 c^\dagger_4]\rangle` -- annihilation indices
        (positions 1, 3) transform with :math:`U`, creation indices (positions 2, 4) with :math:`U^\dagger` --
        combined with a per-k antisymmetry sign :math:`\sigma_k` and an optional complex conjugation ``conj_k``:

            2-index : :math:`M_{12}(k)   = \sigma_k     \, U_{1a} [M_{ab}(k_{rep})]^{[*conj_k]} U^\dagger_{b2}`
            4-index : :math:`M_{1234}(k) = \sigma_k^2 \, U_{1a} [M_{abcd}(k_{rep})]^{[*conj_k]} U^\dagger_{b2} U_{3c} U^\dagger_{d4}`

        If ``k_grid`` is not yet in auto mode (``specify_auto_symmetries`` has not been called),
        only the momentum expansion is performed and orbital indices are left unchanged.

        :param k_grid: The momentum grid carrying the irreducible-to-full-BZ map and per-k orbital rotations.
        :param num_orbital_dimensions: Number of orbital axes to transform; must be 2 or 4.
        :param nq: Optional override for the number of momenta; if None the object's own ``nq`` is used.
        :return: ``self`` expanded to the full BZ.
        :raises ValueError: If the object does not have a compressed momentum dimension.
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
            from dgamore import symmetry_reduction

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

        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: The Fourier-transformed object, in the same momentum-compression state as the input.
        """
        if copy:
            return self.copy().fft(copy=False)

        compress = False
        if self.has_compressed_q_dimension:
            compress = True
            self.decompress_q_dimension()
        # scipy.fft + overwrite_x transforms the complex64 array in place (no extra buffer). Do NOT switch to
        # numpy.fft.fftn: it computes in double precision internally and peaks at ~5x the array size.
        self.mat = sp.fft.fftn(self.mat, axes=(0, 1, 2), overwrite_x=True)
        return self.compress_q_dimension() if compress else self

    def ifft(self, copy: bool = True):
        """
        Performs a discrete inverse Fourier transform over the momentum dimension and returns a copy if specified.

        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: The inverse-Fourier-transformed object, in the same momentum-compression state as the input.
        """
        if copy:
            return self.copy().ifft(copy=False)

        compress = False
        if self.has_compressed_q_dimension:
            compress = True
            self.decompress_q_dimension()
        # scipy.fft + overwrite_x transforms the complex64 array in place (no extra buffer). Do NOT switch to
        # numpy.fft.ifftn: it computes in double precision internally and peaks at ~5x the array size.
        self.mat = sp.fft.ifftn(self.mat, axes=(0, 1, 2), overwrite_x=True)
        return self.compress_q_dimension() if compress else self

    def flip_momentum_axis(self, copy: bool = True):
        r"""
        Flips the momentum axis :math:`F^{q}\to F^{-q}` of the object and returns a copy if specified.

        :param copy: If True, operate on and return a deep copy; if False, mutate and return ``self`` in place.
        :return: The momentum-flipped object, in the same momentum-compression state as the input.
        """
        if copy:
            clone = self._clone_without_mat()
            clone.mat = self.mat  # shared; flip/roll below allocate a fresh array, leaving self.mat untouched
            return clone.flip_momentum_axis(copy=False)

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

        :param other: The other non-local object to align with ``self``.
        :return: ``other``, compressed if necessary to match ``self``'s momentum layout.
        """
        if not self.has_compressed_q_dimension and other.has_compressed_q_dimension:
            self.compress_q_dimension()
        if not other.has_compressed_q_dimension and self.has_compressed_q_dimension:
            other = other.compress_q_dimension()
        return other

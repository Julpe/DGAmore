# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Multiprocessing (MPI) utilities for the non-local step - a single module covering everything parallel:

* low-level **message-chunking primitives** (``send_rows`` / ``recv_rows_into`` / ``recv_rows_alloc`` /
  ``bcast_rows`` / ``bcast_rows_into`` / ``send_bytes`` / ``recv_bytes`` and the ``row_chunks`` / ``chunk_step``
  bound math) that split any transfer below the 2 GB MPI per-message limit and write received data **directly into**
  the caller's preallocated contiguous buffer (no per-chunk staging copy);
* the **work distributor** :class:`MpiDistributor`, which splits a number of tasks (typically the q-points of the
  irreducible Brillouin zone) into per-rank contiguous slices and wraps the collective/point-to-point communication
  (scatter, gather, all-gather, all-reduce, broadcast, object send/recv); each rank owns a private HDF5 file for
  spilling intermediate results without write conflicts;
* higher-level **data-movement routines** built on top: mapping objects between the irreducible-BZ and full-BZ rank
  distributions (a simple gather/scatter version and a fully distributed peer-to-peer version that never assembles
  the full object on one rank), a node-aware frequency distribution, and a distributed 3D FFT over the BZ via pencil
  redistributions.

``MAX_MPI_BYTES`` is the single source of the 2 GB limit; the chunking helpers take it as an explicit ``limit``
argument (read at call time by the callers), so monkeypatching ``mpi_utils.MAX_MPI_BYTES`` in tests still forces the
chunked path.
"""

import gc
import os
import pickle
import socket

import h5py
import mpi4py.MPI as MPI
import numpy as np
import scipy.fft as fft

import dgamore.config as config
from dgamore import symmetry_reduction
from dgamore.brillouin_zone import KGrid
from dgamore.four_point import FourPoint
from dgamore.n_point_base import DTYPE

# Canonical 2 GB MPI per-message limit. The chunking helpers below take it as an explicit ``limit`` argument so the
# established test hook of monkeypatching ``mpi_utils.MAX_MPI_BYTES`` to force the chunked path keeps working.
MAX_MPI_BYTES = 2**31 - 1


def build_node_shared_array(node_comm, compute_fn, dtype=DTYPE):
    r"""
    Build an array once per node and expose it to every rank on that node through a single MPI shared-memory window,
    so a large replicated quantity (e.g. the full-grid Green's function ``giwk_full``) is stored **once per node
    instead of once per rank** and computed only on the node root.

    ``compute_fn`` is called **only on the node-local root rank** (rank 0 of ``node_comm``) and must return the fully
    built numpy array; the other ranks do not call it. Its result is copied into a shared-memory segment allocated by
    the root, and every rank receives a numpy view of that same physical buffer (read-only by convention - only the
    root writes it). ``node_comm`` must be a node-local communicator, e.g. ``comm.Split_type(MPI.COMM_TYPE_SHARED)``.

    When the node holds a single rank the shared window is pointless, so the freshly computed private array is
    returned unchanged with ``win = None`` (this also keeps single-rank / mock communicators working). The caller
    owns the returned window and must free it (``win.Free()``) once all ranks are done reading the array.

    :param node_comm: The node-local (shared-memory) communicator.
    :param compute_fn: Zero-argument callable returning the array; invoked only on the node root.
    :param dtype: Storage dtype of the shared buffer (defaults to the global ``DTYPE``, complex64).
    :return: The tuple ``(array, win)`` - the (shared) numpy array on every rank and the MPI window (``None`` for a
        single-rank node).
    """
    is_root = node_comm.Get_rank() == 0
    local = compute_fn() if is_root else None
    if node_comm.Get_size() == 1:
        return local, None
    shape = node_comm.bcast(local.shape if is_root else None)
    itemsize = np.dtype(dtype).itemsize
    nbytes = int(np.prod(shape)) * itemsize if is_root else 0
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=node_comm)
    buf, _ = win.Shared_query(0)
    shared = np.ndarray(buffer=buf, dtype=dtype, shape=shape)
    if is_root:
        shared[...] = local
    node_comm.Barrier()
    return shared, win


# ====================================================================================================================
# Message-chunking primitives (split a transfer's leading axis below the 2 GB MPI per-message limit).
def _items_per_row(shape: tuple) -> int:
    """
    Returns the number of scalars per leading-axis row (the product of the trailing dimensions, or 1 for a 1D array).

    :param shape: The array shape.
    :return: The number of scalars per axis-0 element.
    """
    return int(np.prod(shape[1:])) if len(shape) > 1 else 1


def chunk_step(itemsize: int, items_per_row: int, limit: int = MAX_MPI_BYTES) -> int:
    """
    Returns the number of leading-axis rows that fit into a single ``limit``-byte message (at least 1). This is the
    raw step used by callers that drive their own (e.g. non-blocking) send/receive loop.

    :param itemsize: Size in bytes of one array element.
    :param items_per_row: Number of scalars per axis-0 row (product of trailing dimensions).
    :param limit: Maximum message size in bytes.
    :return: The maximum number of rows per chunk (>= 1).
    """
    return max(1, limit // (itemsize * max(1, items_per_row)))


def row_chunks(n_rows: int, itemsize: int, items_per_row: int, limit: int = MAX_MPI_BYTES):
    """
    Yields ``(start, stop)`` row-index pairs splitting ``n_rows`` leading-axis rows into sub-``limit``-byte chunks.

    :param n_rows: Number of rows (axis-0 elements) to split.
    :param itemsize: Size in bytes of one array element.
    :param items_per_row: Number of scalars per axis-0 row (product of trailing dimensions).
    :param limit: Maximum message size in bytes.
    :return: A generator of ``(start, stop)`` row-index pairs.
    """
    step = chunk_step(itemsize, items_per_row, limit)
    for i in range(0, n_rows, step):
        yield i, min(n_rows, i + step)


def send_rows(comm, arr: np.ndarray, dest: int, base_tag: int = 0, limit: int = MAX_MPI_BYTES) -> None:
    """
    Sends ``arr`` to ``dest`` in sub-``limit``-byte chunks along axis 0 (``tag = base_tag + chunk_index``).

    :param comm: The MPI communicator.
    :param arr: The array to send.
    :param dest: Destination rank.
    :param base_tag: Base MPI tag; successive chunks use ``base_tag + chunk_index``.
    :param limit: Maximum message size in bytes.
    :return: None.
    """
    arr = np.ascontiguousarray(arr)
    for idx, (i, j) in enumerate(row_chunks(arr.shape[0], arr.dtype.itemsize, _items_per_row(arr.shape), limit)):
        comm.Send(arr[i:j], dest=dest, tag=base_tag + idx)


def recv_rows_into(comm, buf: np.ndarray, source: int, base_tag: int = 0, limit: int = MAX_MPI_BYTES) -> np.ndarray:
    """
    Receives rows from ``source`` **directly into** the contiguous buffer ``buf`` (no per-chunk staging buffer). The
    buffer's axis-0 slices must be contiguous (e.g. ``buf`` is C-contiguous or an axis-0 view of such an array).

    :param comm: The MPI communicator.
    :param buf: The destination buffer; its leading axis length determines the number of rows received.
    :param source: Source rank.
    :param base_tag: Base MPI tag; successive chunks use ``base_tag + chunk_index``.
    :param limit: Maximum message size in bytes.
    :return: ``buf`` (filled in place).
    """
    for idx, (i, j) in enumerate(row_chunks(buf.shape[0], buf.dtype.itemsize, _items_per_row(buf.shape), limit)):
        comm.Recv(buf[i:j], source=source, tag=base_tag + idx)
    return buf


def recv_rows_alloc(
    comm, shape: tuple, dtype, source: int, base_tag: int = 0, limit: int = MAX_MPI_BYTES
) -> np.ndarray:
    """
    Allocates an array of the given shape/dtype and receives into it (see :func:`recv_rows_into`).

    :param comm: The MPI communicator.
    :param shape: Shape of the array to receive.
    :param dtype: Dtype of the array to receive.
    :param source: Source rank.
    :param base_tag: Base MPI tag; successive chunks use ``base_tag + chunk_index``.
    :param limit: Maximum message size in bytes.
    :return: The received array.
    """
    return recv_rows_into(comm, np.empty(shape, dtype=dtype), source, base_tag=base_tag, limit=limit)


def _isend_rows(comm, arr: np.ndarray, dest: int, base_tag: int = 0, limit: int = MAX_MPI_BYTES) -> list:
    """
    Non-blocking counterpart of :func:`send_rows`: posts one ``Isend`` per sub-``limit``-byte axis-0 chunk
    (``tag = base_tag + chunk_index``) and returns the request list for a later ``Waitall``, so transfers to
    different peers (and the chunks of one transfer) overlap instead of serializing.

    ``arr`` must be C-contiguous and **kept alive by the caller** until the returned requests complete; the helper
    deliberately does not stage a private copy (that copy would be freed on return and corrupt an in-flight send).

    :param comm: The MPI communicator.
    :param arr: The C-contiguous array to send.
    :param dest: Destination rank.
    :param base_tag: Base MPI tag; successive chunks use ``base_tag + chunk_index``.
    :param limit: Maximum message size in bytes.
    :return: The list of MPI request handles for the posted sends.
    :raises ValueError: If ``arr`` is not C-contiguous.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("_isend_rows requires a C-contiguous array (the caller must keep it alive until Waitall)")
    reqs = []
    for idx, (i, j) in enumerate(row_chunks(arr.shape[0], arr.dtype.itemsize, _items_per_row(arr.shape), limit)):
        reqs.append(comm.Isend(arr[i:j], dest=dest, tag=base_tag + idx))
    return reqs


def _irecv_rows_into(comm, buf: np.ndarray, source: int, base_tag: int = 0, limit: int = MAX_MPI_BYTES) -> list:
    """
    Non-blocking counterpart of :func:`recv_rows_into`: posts one ``Irecv`` per sub-``limit``-byte axis-0 chunk
    **directly into** the contiguous buffer ``buf`` (no per-chunk staging) and returns the request list for a later
    ``Waitall``. The buffer's axis-0 slices must be contiguous (e.g. ``buf`` is C-contiguous).

    :param comm: The MPI communicator.
    :param buf: The destination buffer; its leading axis length determines the number of rows received.
    :param source: Source rank.
    :param base_tag: Base MPI tag; successive chunks use ``base_tag + chunk_index``.
    :param limit: Maximum message size in bytes.
    :return: The list of MPI request handles for the posted receives.
    """
    reqs = []
    for idx, (i, j) in enumerate(row_chunks(buf.shape[0], buf.dtype.itemsize, _items_per_row(buf.shape), limit)):
        reqs.append(comm.Irecv(buf[i:j], source=source, tag=base_tag + idx))
    return reqs


def bcast_rows(comm, arr: np.ndarray, root: int, limit: int = MAX_MPI_BYTES) -> np.ndarray:
    """
    Broadcasts a numpy array from ``root`` to all ranks, chunked along axis 0. Non-root ranks allocate the receive
    buffer from the broadcast shape/dtype.

    :param comm: The MPI communicator.
    :param arr: The array to broadcast (only read on ``root``).
    :param root: The broadcasting rank.
    :param limit: Maximum message size in bytes.
    :return: The broadcast array on every rank.
    """
    rank = comm.Get_rank()
    shape = comm.bcast(arr.shape if rank == root else None, root=root)
    dtype = comm.bcast(arr.dtype if rank == root else None, root=root)

    if rank != root:
        arr = np.empty(shape, dtype=dtype)
    arr = np.ascontiguousarray(arr)

    for i, j in row_chunks(shape[0], np.dtype(dtype).itemsize, _items_per_row(shape), limit):
        comm.Bcast(arr[i:j], root=root)
    return arr


def bcast_rows_into(comm, view: np.ndarray, root: int, limit: int = MAX_MPI_BYTES) -> np.ndarray:
    """
    Broadcasts **into** an existing contiguous buffer view, chunked along axis 0. This is a collective call: every
    rank must pass the matching view (same shape/dtype). Used to fill one rank's slice of an all-gather target.

    :param comm: The MPI communicator.
    :param view: The destination buffer view (axis-0 slices must be contiguous), identical in shape on all ranks.
    :param root: The rank whose data is broadcast.
    :param limit: Maximum message size in bytes.
    :return: ``view`` (filled in place).
    """
    for i, j in row_chunks(view.shape[0], view.dtype.itemsize, _items_per_row(view.shape), limit):
        comm.Bcast(view[i:j], root=root)
    return view


def send_bytes(comm, data: bytes, dest: int, base_tag: int = 0, limit: int = MAX_MPI_BYTES) -> None:
    """
    Sends a raw byte blob to ``dest`` in sub-``limit``-byte chunks, preceded by a small length message (at
    ``base_tag``); the chunks use ``base_tag + 1 + chunk_index``.

    :param comm: The MPI communicator.
    :param data: The bytes to send.
    :param dest: Destination rank.
    :param base_tag: Base MPI tag (length at ``base_tag``, chunks at ``base_tag + 1 + chunk_index``).
    :param limit: Maximum message size in bytes.
    :return: None.
    """
    total = len(data)
    comm.send(total, dest=dest, tag=base_tag)
    arr = np.frombuffer(data, dtype=np.uint8)
    for idx, (i, j) in enumerate(row_chunks(total, 1, 1, limit)):
        comm.Send(arr[i:j], dest=dest, tag=base_tag + 1 + idx)


def recv_bytes(comm, source: int, base_tag: int = 0, limit: int = MAX_MPI_BYTES) -> bytes:
    """
    Receives a chunked raw byte blob sent by :func:`send_bytes`.

    :param comm: The MPI communicator.
    :param source: Source rank.
    :param base_tag: Base MPI tag matching the one used by :func:`send_bytes`.
    :param limit: Maximum message size in bytes.
    :return: The reassembled bytes.
    """
    total = comm.recv(source=source, tag=base_tag)
    buf = np.empty(total, dtype=np.uint8)
    for idx, (i, j) in enumerate(row_chunks(total, 1, 1, limit)):
        comm.Recv(buf[i:j], source=source, tag=base_tag + 1 + idx)
    return buf.tobytes()


# ====================================================================================================================
# Work distribution.
class MpiDistributor:
    """
    Distributes tasks among all available cores. Uses the first (q) dimension to slice the vertex data into chunks
    and sends it to all active MPI processes. Saves intermediate computational results in rank files. Each rank
    has its own instance of an MPI distributor and hdf5-file to avoid write conflicts.
    """

    def __init__(self, ntasks: int = 1, comm: MPI.Comm = None, name: str = "", output_path: str = None):
        """
        Distributes the tasks across the communicator and opens this rank's HDF5 spill file.

        :param ntasks: Total number of tasks to distribute (e.g. the number of irreducible q-points).
        :param comm: The MPI communicator across which the tasks are distributed.
        :param name: Prefix for this rank's HDF5 spill file.
        :param output_path: Directory the per-rank HDF5 spill file is created in; if None, no spill file is opened.
        """
        self._comm = comm
        self._ntasks = ntasks
        self._file = None
        self._my_slice = None
        self._sizes = None
        self._my_size = None
        self._slices = None

        self._distribute_tasks()

        if output_path is not None:
            # creates rank file if it does not exist
            self._fname = os.path.join(output_path, f"{name}_Rank{self.my_rank:05d}.hdf5")
            self._file = h5py.File(self._fname, "a")
            self._file.close()

    def __del__(self):
        """
        Destructor to close the hdf5 file if it is still open.
        """
        if self._file is not None:
            try:
                self.close_file()
            except (OSError, AttributeError):
                pass

    def __enter__(self):
        """
        Context manager to open the hdf5 file.
        """
        self.open_file()
        return self._file

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit; closes the hdf5 file (see :meth:`close_file`).

        :param exc_type: Exception type if one was raised in the ``with`` block, else None.
        :param exc_value: Exception instance if one was raised, else None.
        :param traceback: Traceback if an exception was raised, else None.
        :return: None.
        """
        if self._file:
            self.close_file()

    @property
    def comm(self) -> MPI.Comm:
        """
        The MPI communicator this distributor operates on.

        :return: The MPI communicator.
        """
        return self._comm

    @property
    def is_root(self) -> bool:
        """
        Whether the current process is the root rank.

        :return: True if the current rank is the root rank (rank 0).
        """
        return self.my_rank == 0

    @property
    def ntasks(self) -> int:
        """
        The total number of distributed tasks.

        :return: The total number of tasks to be distributed (e.g. the number of irreducible-BZ q-points).
        """
        return self._ntasks

    @property
    def sizes(self) -> np.ndarray:
        """
        The per-rank task counts.

        :return: The per-rank chunk sizes (number of tasks assigned to each rank).
        """
        return self._sizes

    @property
    def slices(self) -> np.ndarray:
        """
        The per-rank slices into the full task list.

        :return: The per-rank ``slice`` objects into the full task list.
        """
        return self._slices

    @property
    def my_rank(self) -> int:
        """
        The current process's rank.

        :return: The rank of the current process.
        """
        return self._comm.Get_rank()

    @property
    def my_tasks(self) -> np.ndarray:
        """
        The task indices owned by the current rank.

        :return: The task indices assigned to the current rank (e.g. the q-points it processes).
        """
        return np.arange(0, self.ntasks)[self.my_slice]

    @property
    def mpi_size(self) -> int:
        """
        The communicator size.

        :return: The total number of MPI processes in the communicator.
        """
        return self._comm.size

    @property
    def my_size(self) -> int:
        """
        The number of tasks owned by the current rank.

        :return: The number of tasks assigned to the current rank.
        """
        return self._my_size

    @property
    def my_slice(self) -> int:
        """
        The current rank's slice into the full task list.

        :return: The ``slice`` object selecting the current rank's portion of the full task list.
        """
        return self._my_slice

    def open_file(self):
        """
        Opens this rank's hdf5 file for read/write. Silently does nothing if the file is missing.

        :return: None.
        """
        try:
            self._file = h5py.File(self._fname, "r+")
        except (OSError, AttributeError):
            pass

    def close_file(self):
        """
        Closes this rank's hdf5 file. Silently does nothing if it is not open.

        :return: None.
        """
        try:
            self._file.close()
        except (OSError, AttributeError):
            pass

    def delete_file(self):
        """
        Deletes this rank's hdf5 spill file. Silently does nothing if it does not exist.

        :return: None.
        """
        try:
            os.remove(self._fname)
        except (OSError, AttributeError):
            pass

    def barrier(self):
        """
        Synchronizes all ranks. Forces a garbage collection first so that all ranks free their memory before the
        barrier.

        :return: None.
        """
        gc.collect()
        self.comm.Barrier()

    def allgather(self, rank_result: np.ndarray = None) -> np.ndarray:
        """
        Gathers each rank's array slice (along axis 0) into the full array, replicated on every rank. The common case
        is a single bandwidth-optimal ``Allgatherv`` collective (a derived "row" count keeps the per-rank counts and
        displacements small, so the result is correct regardless of element size); only when a rank's slice would
        exceed the 2 GB per-message limit does it fall back to per-rank chunked broadcasts.

        :param rank_result: This rank's slice of the result (leading axis indexes the rank's tasks).
        :return: The full array of shape ``(ntasks, ...)`` on all ranks.
        """
        rank_result = np.ascontiguousarray(rank_result)
        tot_shape = (self.ntasks,) + rank_result.shape[1:]

        # Single rank: nothing to gather. Returning a copy avoids a needless collective and keeps the routine usable
        # on a minimal communicator that does not implement Allgatherv (e.g. the single-rank test mock).
        if self.mpi_size == 1:
            return rank_result.copy()

        tot_result = np.empty(tot_shape, dtype=rank_result.dtype)

        items = _items_per_row(rank_result.shape)
        max_rows = chunk_step(rank_result.dtype.itemsize, items, limit=MAX_MPI_BYTES)

        # Fast path: a single Allgatherv when every rank's slice fits one message and the whole result's element count
        # fits an MPI int displacement. Counts/displacements are in elements of the flattened buffers.
        if self._sizes.max(initial=0) <= max_rows and tot_result.size < 2**31:
            counts = (self._sizes * items).astype(int)
            displs = np.array([s.start for s in self._slices], dtype=int) * items
            self.comm.Allgatherv(
                [rank_result.reshape(-1), int(counts[self.my_rank])],
                [tot_result.reshape(-1), (counts, displs)],
            )
            return tot_result

        # Fallback for arrays exceeding the 2 GB per-message limit: broadcast each rank's contiguous slice of the
        # target buffer from that rank, chunked under the 2 GB limit.
        for r in range(self.mpi_size):
            sub = tot_result[self._slices[r]]
            if self.my_rank == r:
                sub[...] = rank_result
            bcast_rows_into(self.comm, sub, root=r, limit=MAX_MPI_BYTES)
        return tot_result

    def gather(self, rank_result: np.ndarray = None, root: int = 0) -> np.ndarray:
        """
        Gathers each rank's array slice into the full array, in correct task order, on the ``root`` rank only. Handles
        arrays exceeding the 2 GB MPI limit by chunking along axis 0.

        :param rank_result: This rank's slice of the result (leading axis indexes the rank's tasks).
        :param root: The rank that collects the full array.
        :return: The full array of shape ``(ntasks, ...)`` on ``root``, ``None`` on the other ranks.
        """
        rank_result = np.ascontiguousarray(rank_result)
        rest_shape = rank_result.shape[1:]

        tot_result = np.empty((self.ntasks,) + rest_shape, dtype=rank_result.dtype) if self.my_rank == root else None

        if self.my_rank == root:
            # copy own slice directly
            tot_result[self._slices[root]] = rank_result

            # Pre-post non-blocking receives into every rank's contiguous destination slice at once, so the incoming
            # transfers overlap instead of completing rank-by-rank; data lands straight in place (no staging buffer).
            reqs = []
            for r in range(self.mpi_size):
                if r == root or self._sizes[r] == 0:
                    continue
                reqs += _irecv_rows_into(self.comm, tot_result[self._slices[r]], source=r, limit=MAX_MPI_BYTES)
            MPI.Request.Waitall(reqs)
        else:
            if rank_result.shape[0] > 0:
                MPI.Request.Waitall(_isend_rows(self.comm, rank_result, dest=root, limit=MAX_MPI_BYTES))

        return tot_result

    def scatter(self, full_data: np.ndarray = None, root: int = 0):
        """
        Scatters the full array (held on ``root``) along axis 0 into the per-rank task slices. Handles the 2 GB MPI
        limit by chunking. The single-rank case where ``full_data`` already has the rank-local length is passed
        through directly.

        :param full_data: The full array on ``root`` (shape ``(ntasks, ...)``); ignored on non-root ranks.
        :param root: The rank holding ``full_data``.
        :return: This rank's slice of the data (shape ``(my_size, ...)``).
        :raises TypeError: If ``full_data`` is given but is not a numpy array.
        :raises ValueError: If ``full_data``'s leading length matches neither ``ntasks`` nor the single-rank case.
        """

        if full_data is not None and not isinstance(full_data, np.ndarray):
            raise TypeError("full_data must be a numpy array or None")

        if full_data is not None:
            data_len = full_data.shape[0]
            rest_shape = full_data.shape[1:]
            data_type = full_data.dtype
        else:
            data_len = None
            rest_shape = None
            data_type = None

        data_type, rest_shape = self.comm.bcast((data_type, rest_shape), root)

        rank_shape = (self._my_size,) + rest_shape if rest_shape else (self._my_size,)
        rank_data = np.empty(rank_shape, dtype=data_type)

        if self.my_rank == root:
            if full_data is None:
                return rank_data

            # Make the source contiguous once so each rank's axis-0 slice is itself contiguous and can be sent as a
            # view (no per-rank copy); this is also required for the non-blocking sends below.
            full_data = np.ascontiguousarray(np.asarray(full_data, dtype=data_type))

            if data_len == self.ntasks:
                # Post non-blocking sends to every other rank at once so the outgoing transfers overlap instead of
                # going rank-by-rank; full_data stays alive (local) until the Waitall.
                reqs = []
                for r in range(self.mpi_size):
                    n = self._sizes[r]
                    if n == 0:
                        continue
                    sl = self._slices[r]
                    if r == root:
                        rank_data[...] = full_data[sl]
                    else:
                        reqs += _isend_rows(self.comm, full_data[sl], dest=r, limit=MAX_MPI_BYTES)
                MPI.Request.Waitall(reqs)
            elif data_len == self._my_size and self.mpi_size == 1:
                rank_data[...] = full_data
            else:
                raise ValueError(f"Mismatch in scatter!")
        else:
            if self._my_size > 0:
                MPI.Request.Waitall(_irecv_rows_into(self.comm, rank_data, source=root, limit=MAX_MPI_BYTES))

        return rank_data

    def send_to_rank(self, obj, dest: int, base_tag: int = 0):
        """
        Sends an n-point-like object to a single rank. The large ``.mat`` array is sent as raw chunks (to avoid
        holding a full pickle blob in memory), while the rest of the object is pickled into a small metadata blob.

        :param obj: The object to send; must expose a ``.mat`` numpy array attribute.
        :param dest: Destination rank.
        :param base_tag: Base MPI tag (metadata uses ``base_tag``, array chunks ``base_tag + 500 + ...``).
        :return: None.
        """

        # Temporarily detach .mat so it is not included in the small pickled metadata blob.
        mat = obj.mat
        obj.mat = None
        try:
            meta_bytes = pickle.dumps(obj)
        finally:
            obj.mat = mat  # always restore, even if pickle raises

        # metadata blob (tags base_tag, base_tag+1, ...), then the raw array preceded by its shape/dtype
        # (meta at base_tag+500, chunks at base_tag+501, ...).
        send_bytes(self.comm, meta_bytes, dest, base_tag=base_tag, limit=MAX_MPI_BYTES)
        mat = np.ascontiguousarray(mat)
        self.comm.send({"shape": mat.shape, "dtype": mat.dtype}, dest=dest, tag=base_tag + 500)
        send_rows(self.comm, mat, dest=dest, base_tag=base_tag + 501, limit=MAX_MPI_BYTES)

    def recv_from_rank(self, source: int, base_tag: int = 0):
        """
        Receives an object sent by :meth:`send_to_rank`: reconstructs the pickled metadata object and reattaches the
        chunk-received ``.mat`` array.

        :param source: Source rank.
        :param base_tag: Base MPI tag matching the one used by :meth:`send_to_rank`.
        :return: The reconstructed object with its ``.mat`` array attached.
        """

        meta_bytes = recv_bytes(self.comm, source, base_tag=base_tag, limit=MAX_MPI_BYTES)
        obj = pickle.loads(meta_bytes)

        meta = self.comm.recv(source=source, tag=base_tag + 500)
        obj.mat = recv_rows_alloc(
            self.comm, meta["shape"], meta["dtype"], source=source, base_tag=base_tag + 501, limit=MAX_MPI_BYTES
        )
        return obj

    def bcast(self, data, root=0):
        """
        Broadcasts an arbitrary (picklable) object from ``root`` to all ranks.

        :param data: The object to broadcast (only read on ``root``).
        :param root: The broadcasting rank.
        :return: The broadcast object on every rank.
        """
        return self.comm.bcast(data, root=root)

    def bcast_chunked(self, arr: np.ndarray, root: int = 0) -> np.ndarray:
        """
        Broadcasts a large numpy array from ``root`` to all ranks, using raw MPI buffers and chunking along axis 0 to
        respect the 2 GB MPI message limit.

        :param arr: The array to broadcast (only read on ``root``; non-root ranks allocate from the broadcast metadata).
        :param root: The broadcasting rank.
        :return: The broadcast array on every rank.
        """
        return bcast_rows(self.comm, arr, root, limit=MAX_MPI_BYTES)

    def bcast_npoint(self, obj, root: int = 0):
        """
        Broadcasts an n-point-like object (one exposing a ``.mat`` numpy array) from ``root`` to all ranks. The large
        ``.mat`` is broadcast as raw sub-2 GB chunks (so there is no multi-gigabyte pickle blob and no >2 GB message),
        while the rest of the object travels as a small pickled metadata blob - the broadcast analog of
        :meth:`send_to_rank`/:meth:`recv_from_rank`. Prefer this over :meth:`bcast` for large objects such as a
        full-BZ self-energy or gap function, both to respect the 2 GB limit and to avoid the full in-memory pickle copy.

        :param obj: The object to broadcast; must expose a ``.mat`` numpy array attribute. Only read on ``root``.
        :param root: The broadcasting rank.
        :return: The broadcast object with its ``.mat`` attached, on every rank.
        """
        if self.mpi_size == 1:
            return obj

        if self.my_rank == root:
            # Detach .mat so the pickled metadata blob stays small; broadcast the array separately as raw chunks.
            mat = obj.mat
            obj.mat = None
            try:
                meta_bytes = pickle.dumps(obj)
            finally:
                obj.mat = mat  # always restore, even if pickle raises
            self.comm.bcast(meta_bytes, root=root)
            obj.mat = bcast_rows(self.comm, mat, root, limit=MAX_MPI_BYTES)
            return obj

        obj = pickle.loads(self.comm.bcast(None, root=root))
        obj.mat = bcast_rows(self.comm, None, root, limit=MAX_MPI_BYTES)
        return obj

    def allreduce(self, rank_result=None) -> np.ndarray:
        """
        Sums an array element-wise across all ranks in place and returns the result on every rank, chunked along axis 0
        so no single message exceeds the 2 GB MPI limit (consistent with the rest of the module).

        ``Allreduce`` is collective, so the chunk schedule must be identical on every rank. That holds here because the
        reduced arrays are always equally shaped across ranks (the callers reduce full, replicated quantities such as
        the full-k-space self-energy / Fock term - each rank holds a partial sum of the *same* array), so every rank
        derives the same chunk boundaries. The single-chunk case is byte-for-byte the previous behavior.

        :param rank_result: This rank's contribution; reduced in place. Must have the same shape on every rank.
        :return: The summed array (same buffer), identical on all ranks.
        """
        rows, itemsize, per_row = rank_result.shape[0], rank_result.dtype.itemsize, _items_per_row(rank_result.shape)
        for i, j in row_chunks(rows, itemsize, per_row, limit=MAX_MPI_BYTES):
            self.comm.Allreduce(MPI.IN_PLACE, rank_result[i:j])
        return rank_result

    @staticmethod
    def create_distributor(
        ntasks: int, comm: MPI.Comm = None, name: str = "", output_path: str = None
    ) -> "MpiDistributor":
        """
        Factory that creates an :class:`MpiDistributor`, defaulting to ``MPI.COMM_WORLD`` if no communicator is given.

        :param ntasks: Total number of tasks to distribute.
        :param comm: The MPI communicator (``MPI.COMM_WORLD`` if None).
        :param name: Prefix for the per-rank HDF5 spill file.
        :param output_path: Directory the per-rank HDF5 spill file is created in; if None, no spill file is opened.
        :return: The created :class:`MpiDistributor`.
        """
        if comm is None:
            comm = MPI.COMM_WORLD
        return MpiDistributor(ntasks=ntasks, comm=comm, name=name, output_path=output_path)

    def restricted_to(self, sub_comm, member_ranks: list) -> "MpiDistributor":
        """
        Returns a new distributor over ``sub_comm`` carrying only ``member_ranks``' task slices of this
        distributor, in the given order, which must match the sub-communicator's rank order (e.g. from a
        ``comm.Split`` keyed on the original rank). Collectives of the returned distributor then span exactly the
        member ranks, so ranks with an empty task slice need not enter them. No per-rank spill file is opened.

        :param sub_comm: The sub-communicator containing exactly the ``member_ranks``.
        :param member_ranks: The original-communicator ranks to keep, ordered like the sub-communicator's ranks.
        :return: The restricted :class:`MpiDistributor`.
        """
        restricted = MpiDistributor(ntasks=self.ntasks, comm=sub_comm)
        restricted._sizes = np.array([self._sizes[r] for r in member_ranks], dtype=int)
        restricted._slices = [self._slices[r] for r in member_ranks]
        restricted._my_size = restricted._sizes[restricted.my_rank]
        restricted._my_slice = restricted._slices[restricted.my_rank]
        return restricted

    def _distribute_tasks(self):
        """
        Computes the per-rank chunk sizes and slices, distributing the tasks as evenly as possible (excess tasks go to
        the highest ranks), and records this rank's own size and slice.

        :return: None.
        """
        n_per_rank = self.ntasks // self.mpi_size
        n_excess = self.ntasks - n_per_rank * self.mpi_size
        self._sizes = n_per_rank * np.ones(self.mpi_size, int)

        if n_excess:
            self._sizes[-n_excess:] += 1

        slice_ends = self._sizes.cumsum()
        self._slices = list(map(slice, slice_ends - self._sizes, slice_ends))
        self._my_size = self._sizes[self.my_rank]
        self._my_slice = self._slices[self.my_rank]


# ====================================================================================================================
# Higher-level data-movement routines (irreducible-BZ <-> full-BZ remap, node-aware frequency split, distributed FFT).
def _get_node_aware_v_dist(n_nu, comm):
    """
    Calculates frequency distribution based on physical node topology.
    Frequencies are split equally among nodes, then assigned to ranks within those nodes.

    Uses inverse rank-lookup to be robust against inconsistent hostname strings
    returned by the OS in local or cluster environments.

    :param n_nu: Total number of fermionic frequencies to distribute.
    :param comm: The MPI communicator.
    :return: The tuple ``(all_sizes, all_slices)`` of per-rank frequency counts and ``slice`` objects.
    :raises RuntimeError: If a rank cannot locate itself in the host map.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Group ranks by physical hostname
    local_hostname = socket.gethostname()
    all_hostnames = comm.allgather(local_hostname)

    # Map hostnames to the list of ranks living on them
    nodes_map = {}
    for r, h in enumerate(all_hostnames):
        h_clean = str(h).strip()
        if h_clean not in nodes_map:
            nodes_map[h_clean] = []
        nodes_map[h_clean].append(r)

    # Sorted list of unique nodes ensures every rank sees the same order
    sorted_node_names = sorted(nodes_map.keys())
    n_nodes = len(sorted_node_names)

    # 2. CANONICAL HOSTNAME RESOLUTION: instead of string matching, find which node list contains THIS rank.
    # This guarantees we find the correct key in nodes_map.
    canonical_hostname = None
    for name, ranks in nodes_map.items():
        if rank in ranks:
            canonical_hostname = name
            break

    if canonical_hostname is None:
        raise RuntimeError(f"Rank {rank} could not find itself in the host map.")

    # 3. Distribute frequencies to nodes
    v_per_node = n_nu // n_nodes
    extra_v_nodes = n_nu % n_nodes

    my_node_idx = sorted_node_names.index(canonical_hostname)
    v_on_this_node = v_per_node + (1 if my_node_idx < extra_v_nodes else 0)

    # 4. Split this node's frequencies among its local ranks
    ranks_on_my_node = nodes_map[canonical_hostname]
    rank_in_node = ranks_on_my_node.index(rank)

    v_per_rank = v_on_this_node // len(ranks_on_my_node)
    extra_v_ranks = v_on_this_node % len(ranks_on_my_node)

    my_size = v_per_rank + (1 if rank_in_node < extra_v_ranks else 0)

    # 5. Globalize the distribution for the Distributor
    # Allgather ensures every rank knows the frequency slices of every other rank.
    all_sizes = np.zeros(size, dtype=int)
    all_sizes[rank] = my_size
    comm.Allgather(MPI.IN_PLACE, all_sizes)

    # Calculate slices based on the gathered sizes
    all_offsets = np.insert(np.cumsum(all_sizes), 0, 0)
    all_slices = [slice(all_offsets[i], all_offsets[i + 1]) for i in range(size)]

    return all_sizes, all_slices


def _send_in_chunks(comm, arr, dest, base_tag=0):
    """
    Sends a numpy array to a destination rank in below-2 GB chunks along axis 0 (no handshake). Thin wrapper around
    :func:`send_rows`, passing this module's ``MAX_MPI_BYTES`` (read at call time so the test hook of monkeypatching
    ``mpi_utils.MAX_MPI_BYTES`` keeps forcing the chunked path).

    :param comm: The MPI communicator.
    :param arr: The array to send.
    :param dest: Destination rank.
    :param base_tag: Base MPI tag; successive chunks use ``base_tag + chunk_index``.
    :return: None.
    """
    send_rows(comm, arr, dest, base_tag=base_tag, limit=MAX_MPI_BYTES)


def _recv_in_chunks(comm, shape, dtype, source, base_tag=0):
    """
    Receives a numpy array from a source rank in below-2 GB chunks along axis 0 into a freshly allocated buffer. Thin
    wrapper around :func:`recv_rows_alloc` (see :func:`_send_in_chunks` for the ``MAX_MPI_BYTES`` handling).

    :param comm: The MPI communicator.
    :param shape: Shape of the array to receive.
    :param dtype: Dtype of the array to receive.
    :param source: Source rank.
    :param base_tag: Base MPI tag; successive chunks use ``base_tag + chunk_index``.
    :return: The received array.
    """
    return recv_rows_alloc(comm, shape, dtype, source, base_tag=base_tag, limit=MAX_MPI_BYTES)


def map_irrbz_fullbz(obj, mpi_dist_irrk, mpi_dist_fullbz):
    """
    Maps an object from the irreducible-BZ rank distribution to the full-BZ distribution the simple way: gather to
    rank 0, unfold to the full BZ there, then scatter back. Requires rank 0 to hold the full-BZ object transiently.

    :param obj: The object distributed over the irreducible BZ (must support :meth:`map_to_full_bz`).
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ (source layout).
    :param mpi_dist_fullbz: MPI distributor over the full BZ (target layout).
    :return: The object distributed over the full BZ.
    """
    obj.mat = mpi_dist_irrk.gather(obj.mat)
    if mpi_dist_irrk.comm.rank == 0:
        obj = obj.map_to_full_bz(config.lattice.k_grid)
    obj.mat = mpi_dist_fullbz.scatter(obj.mat)
    return obj


def exchange_and_map_irrbz_fullbz(
    obj: FourPoint, mpi_dist_irrk: MpiDistributor, mpi_dist_fullbz: MpiDistributor
) -> FourPoint:
    """
    Maps an object from the irreducible BZ distribution to the full BZ distribution without
    ever assembling the full object on any single rank.

    Each rank holds a slice of the object over the irreducible BZ (shape [q_irr_rank, ...]).
    This routine redistributes the data peer-to-peer so that each rank ends up with a slice
    over the full BZ (shape [q_full_rank, ...]), with symmetry-equivalent points correctly
    replicated according to the ``irrk_inv`` mapping.

    If ``config.lattice.k_grid`` is in auto-discovered symmetry mode (its
    ``specify_auto_symmetries`` has been called), the per-k orbital transformation
    ``(sigma_k, U_k, conj_k)`` is also applied locally on each rank, using only the
    transformation arrays sliced to that rank's FBZ range. No global gather is needed.

    This is a distributed replacement for the pattern (see also :func:`map_irrbz_fullbz`)::

        obj.mat = mpi_dist_irrk.gather(obj.mat)
        if comm.rank == 0:
            obj = obj.map_to_full_bz(q_grid)
        obj.mat = mpi_dist_fullbz.scatter(obj.mat)

    which would require rank 0 to hold the entire full-BZ object in memory.

    :param obj: The :class:`FourPoint` distributed over the irreducible BZ.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ (source layout).
    :param mpi_dist_fullbz: MPI distributor over the full BZ (target layout).
    :return: The :class:`FourPoint` distributed over the full BZ (compressed q dimension).
    """
    comm = mpi_dist_irrk.comm
    rank = comm.rank
    size = comm.size

    q_grid = config.lattice.k_grid

    # 1. Global mapping setup
    # irrk_inv[fbz_idx] = irrbz_idx
    irrk_inv_flat = q_grid.irrk_inv.ravel()

    # These are the global FBZ indices this specific rank is responsible for
    my_fbz_range = np.arange(mpi_dist_fullbz.my_slice.start, mpi_dist_fullbz.my_slice.stop)
    # These are the corresponding global IRBZ indices needed
    needed_irrk_indices = irrk_inv_flat[my_fbz_range]

    # 2. Identify Sources
    # Find which rank owns each needed IRBZ index
    irr_rank_starts = np.array([s.start for s in mpi_dist_irrk.slices])
    # owner_ranks[i] is the rank that has the data for my_fbz_range[i]
    owner_ranks = np.searchsorted(irr_rank_starts, needed_irrk_indices, side="right") - 1

    # 3. Request/Send Index Information: tell each rank exactly which of its LOCAL indices we need,
    # asking for each unique index only once.

    indices_to_send = [[] for _ in range(size)]
    # Mapping to help us rebuild the full_mat after receiving unique matrices
    # key: source_rank, value: (unique_local_indices, map_to_my_fbz_slice)
    receiving_info = {}

    for src in range(size):
        mask = owner_ranks == src
        if not np.any(mask):
            continue

        global_indices = needed_irrk_indices[mask]
        local_indices = global_indices - irr_rank_starts[src]

        # Uniqueify so we don't transfer the same matrix multiple times
        unique_local, inv_map = np.unique(local_indices, return_inverse=True)
        indices_to_send[src] = unique_local.astype(int)

        # Store how to put these unique received matrices back into our full_mat
        receiving_info[src] = {
            "full_mat_locations": np.where(mask)[0],
            "unique_map": inv_map,
            "count": unique_local.size,
        }

    # 4. Exchange Counts and Indices
    send_counts = np.array([len(indices_to_send[r]) for r in range(size)], dtype=int)
    recv_counts = np.empty(size, dtype=int)
    comm.Alltoall(send_counts, recv_counts)

    reqs = []
    remote_indices_needed_from_me = [np.empty(recv_counts[r], dtype=int) for r in range(size)]

    for r in range(size):
        if r == rank:
            continue
        if send_counts[r] > 0:
            reqs.append(comm.Isend(indices_to_send[r], dest=r, tag=11))
        if recv_counts[r] > 0:
            reqs.append(comm.Irecv(remote_indices_needed_from_me[r], source=r, tag=11))
    MPI.Request.Waitall(reqs)

    # 5. Data Exchange
    # Prepare buffers
    rest_shape = obj.mat.shape[1:]
    dtype = obj.mat.dtype
    full_mat = np.empty((mpi_dist_fullbz.my_size,) + rest_shape, dtype=dtype)

    data_reqs = []
    send_buffers = []  # Keep alive for Isend

    # Handle Self-Copy first (Avoids MPI latency for local data)
    if rank in receiving_info:
        info = receiving_info[rank]
        local_data = obj.mat[indices_to_send[rank]]
        full_mat[info["full_mat_locations"]] = local_data[info["unique_map"]]

    # Prepare Receives
    receive_buffers = {}
    for src, info in receiving_info.items():
        if src == rank:
            continue
        buf = np.empty((info["count"],) + rest_shape, dtype=dtype)
        receive_buffers[src] = buf
        # Chunk the receive under the 2 GB limit (the matching send chunks identically: same row count and shape).
        data_reqs += _irecv_rows_into(comm, buf, source=src, base_tag=12, limit=MAX_MPI_BYTES)

    # Prepare Sends
    for dest in range(size):
        if dest == rank or recv_counts[dest] == 0:
            continue
        # Extract the matrices the remote rank requested from my local IRBZ slice
        data_to_send = np.ascontiguousarray(obj.mat[remote_indices_needed_from_me[dest]])
        send_buffers.append(data_to_send)
        data_reqs += _isend_rows(comm, data_to_send, dest=dest, base_tag=12, limit=MAX_MPI_BYTES)

    MPI.Request.Waitall(data_reqs)

    # 6. Reconstruct full_mat from received unique buffers
    for src, buf in receive_buffers.items():
        info = receiving_info[src]
        # Duplicate the unique received matrices into their (multiple) FBZ target rows
        full_mat[info["full_mat_locations"]] = buf[info["unique_map"]]

    # 6b. Apply the per-k orbital transformation locally if q_grid is in auto mode. Each rank only needs the FBZ
    # slice of (Us, sigmas, conjs) for its own my_fbz_range - no gather, no scatter.
    if q_grid.is_auto and mpi_dist_fullbz.my_size > 0:
        # FourPoint always has 4 orbital indices contracted with the symmetry.
        num_orb_dims = 4
        nb_full = q_grid._auto_us.shape[-1]
        us_flat = q_grid._auto_us.reshape(-1, nb_full, nb_full)
        sigmas_flat = q_grid._auto_sigmas.reshape(-1)
        conjs_flat = q_grid._auto_conjs.reshape(-1)

        us_local = us_flat[my_fbz_range]
        sigmas_local = sigmas_flat[my_fbz_range]
        conjs_local = conjs_flat[my_fbz_range]

        full_mat = symmetry_reduction.apply_auto_orbital_transform(
            full_mat,
            us=us_local,
            sigmas=sigmas_local,
            conjs=conjs_local,
            num_orbital_dimensions=num_orb_dims,
        )

    # 7. Finalize Object
    return FourPoint(
        full_mat,
        obj.channel,
        obj.nq,
        obj.num_wn_dimensions,
        obj.num_vn_dimensions,
        obj.full_niw_range,
        obj.full_niv_range,
        True,
        obj.frequency_notation,
    )


def gather_full_ibz_for_vslice(
    gamma_r: FourPoint, mpi_dist_irrq: MpiDistributor, mpi_dist_v: MpiDistributor, q_grid: KGrid
) -> FourPoint:
    """
    Re-lays out a q-distributed pairing vertex into a fermionic-frequency-distributed one for the Eliashberg solver:
    each rank ends up with the full BZ but only its node-aware slice of the (second) fermionic frequency. The
    momentum is unfolded to the full BZ locally. Ranks with an empty frequency slice receive ``None``.

    :param gamma_r: The :class:`FourPoint` pairing vertex distributed over the irreducible BZ.
    :param mpi_dist_irrq: MPI distributor over the irreducible BZ q-points (source layout).
    :param mpi_dist_v: MPI distributor over the fermionic frequency axis (updated in place to the node-aware split).
    :param q_grid: The :class:`KGrid` used to unfold to the full BZ.
    :return: The full-BZ :class:`FourPoint` for this rank's frequency slice, or ``None`` if the slice is empty.
    """
    # 1. Distribution Update (Node-Aware)
    sizes, slices = _get_node_aware_v_dist(mpi_dist_v.ntasks, mpi_dist_v.comm)
    mpi_dist_v._sizes, mpi_dist_v._slices = sizes, slices
    mpi_dist_v._my_size = sizes[mpi_dist_v.my_rank]

    comm = mpi_dist_irrq.comm
    size = mpi_dist_irrq.mpi_size
    dtype = gamma_r.mat.dtype

    orb_dims = gamma_r.mat.shape[1:-2]
    n_vp = gamma_r.mat.shape[-1]
    items_per_q_v = int(np.prod(orb_dims)) * mpi_dist_v.my_size * n_vp

    # 2. Pre-allocate Buffer
    if mpi_dist_v.my_size > 0:
        full_ibz_mat = np.zeros((mpi_dist_irrq.ntasks,) + orb_dims + (mpi_dist_v.my_size, n_vp), dtype=dtype)
    else:
        full_ibz_mat = None

    # 3. Non-Blocking Exchange (The "Fast" Way)
    reqs = []
    send_buffers = []  # Protect from Garbage Collection

    # A. Pre-post Receives (Matches the exchange_and_map logic)
    if mpi_dist_v.my_size > 0:
        for r_src in range(size):
            q_src_count = mpi_dist_irrq.sizes[r_src]
            if q_src_count == 0:
                continue

            # Same per-chunk row count as the shared chunking helpers (drives the manual non-blocking loop below).
            max_q_recv = chunk_step(dtype.itemsize, items_per_q_v, limit=MAX_MPI_BYTES)
            q_offset = mpi_dist_irrq.slices[r_src].start

            for chunk_idx, i in enumerate(range(0, q_src_count, max_q_recv)):
                j = min(q_src_count, i + max_q_recv)
                # MPI matching is (source, tag)-scoped, so the tag only needs to distinguish the chunks of ONE pair
                # transfer; a rank-dependent base (former r_src*size+rank) overflows MPI_TAG_UB above ~180 ranks.
                tag = 700 + chunk_idx
                reqs.append(comm.Irecv(full_ibz_mat[q_offset + i : q_offset + j], source=r_src, tag=tag))

    # B. Post Sends
    for r_dst in range(size):
        v_dst_size = mpi_dist_v.sizes[r_dst]
        if v_dst_size == 0 or mpi_dist_irrq.my_size == 0:
            continue

        v_dst_slice = mpi_dist_v.slices[r_dst]
        items_per_q_send = int(np.prod(orb_dims)) * v_dst_size * n_vp
        max_q_send = chunk_step(dtype.itemsize, items_per_q_send, limit=MAX_MPI_BYTES)

        for chunk_idx, i in enumerate(range(0, mpi_dist_irrq.my_size, max_q_send)):
            j = min(mpi_dist_irrq.my_size, i + max_q_send)
            tag = 700 + chunk_idx  # rank-independent, see the matching receive above

            # Payload must be contiguous for Send
            payload = np.ascontiguousarray(gamma_r.mat[i:j, ..., v_dst_slice, :])
            send_buffers.append(payload)
            reqs.append(comm.Isend(payload, dest=r_dst, tag=tag))

    # 4. Wait for All to complete
    MPI.Request.Waitall(reqs)

    # 5. Local Expansion
    if mpi_dist_v.my_size > 0:
        gamma_r.mat = full_ibz_mat
        return gamma_r.map_to_full_bz(q_grid)
    else:
        return None


def get_pencil_indices(rank: int, size: int, nq: tuple[int, int, int], layout: str) -> np.ndarray:
    """
    Calculates which global flattened q-indices (0 to ``n_tot - 1``) a rank owns under a given decomposition layout.
    The ``"flat"`` layout matches :meth:`MpiDistributor._distribute_tasks` (excess on the last ranks); the pencil
    layouts assign whole lines along one axis so a subsequent 1D FFT along that axis is rank-local.

    :param rank: The rank whose indices to compute.
    :param size: Total number of ranks.
    :param nq: Number of momenta per spatial direction ``(nx, ny, nz)``.
    :param layout: One of ``"flat"``, ``"z_pencil"``, ``"y_pencil"``, ``"x_pencil"``.
    :return: The global flattened q-indices owned by ``rank``.
    :raises ValueError: If ``layout`` is not one of the supported layouts.
    """
    nx, ny, nz = nq
    n_tot = nx * ny * nz

    if layout == "flat":
        # Same convention as MpiDistributor._distribute_tasks: excess on the LAST ranks.
        n_per, rem = divmod(n_tot, size)
        sizes = np.full(size, n_per, dtype=int)
        if rem:
            sizes[-rem:] += 1
        start = int(sizes[:rank].sum())
        count = int(sizes[rank])
        return np.arange(start, start + count)
    elif layout == "z_pencil":
        # A Z-pencil owns all nz points for a specific (x, y) coordinate.
        # Total number of such pencils is nx * ny.
        n_pencils = nx * ny
        n_per, rem = divmod(n_pencils, size)
        start_p = rank * n_per + min(rank, rem)
        count_p = n_per + (1 if rank < rem else 0)

        # In a flattened array [x,y,z], a Z-pencil is a contiguous block of length nz.
        # The global start index of pencil 'p' is p * nz.
        indices = []
        for p in range(start_p, start_p + count_p):
            indices.append(np.arange(p * nz, (p + 1) * nz))
        return np.concatenate(indices) if indices else np.array([], dtype=int)
    elif layout == "y_pencil":
        # A Y-pencil owns all ny points for a specific (x, z) coordinate.
        # Total number of such pencils is nx * nz.
        n_pencils = nx * nz
        n_per, rem = divmod(n_pencils, size)
        start_p = rank * n_per + min(rank, rem)
        count_p = n_per + (1 if rank < rem else 0)

        indices = []
        for p in range(start_p, start_p + count_p):
            # Decompose pencil index p into x and z
            ix = p // nz
            iz = p % nz
            # A Y-pencil starts at (ix, 0, iz) and jumps by nz for ny steps.
            # Global index q = ix*(ny*nz) + iy*nz + iz
            start_q = ix * (ny * nz) + iz
            indices.append(start_q + np.arange(ny) * nz)
        return np.concatenate(indices) if indices else np.array([], dtype=int)
    elif layout == "x_pencil":
        # An X-pencil owns all nx points for a specific (y, z) coordinate.
        # Total number of such pencils is ny * nz.
        n_pencils = ny * nz
        n_per, rem = divmod(n_pencils, size)
        start_p = rank * n_per + min(rank, rem)
        count_p = n_per + (1 if rank < rem else 0)

        indices = []
        for p in range(start_p, start_p + count_p):
            # p represents the (y, z) coordinate
            iy = p // nz
            iz = p % nz
            # An X-pencil starts at (0, iy, iz) and jumps by (ny*nz) for nx steps.
            # Global index q = ix*(ny*nz) + iy*nz + iz
            start_q = iy * nz + iz
            indices.append(start_q + np.arange(nx) * (ny * nz))
        return np.concatenate(indices) if indices else np.array([], dtype=int)
    else:
        raise ValueError(f"Unknown layout: {layout}")


def _redistribute_p2p(mat, nq, comm, source_layout, target_layout):
    """
    Peer-to-peer redistributes the rows of ``mat`` (indexed by flattened q) from one pencil/flat layout to another,
    exchanging only the rows each rank pair shares (in below-2 GB byte chunks).

    :param mat: The local array slice, with the q-index on axis 0.
    :param nq: Number of momenta per spatial direction ``(nx, ny, nz)``.
    :param comm: The MPI communicator.
    :param source_layout: The current layout of ``mat`` (see :func:`get_pencil_indices`).
    :param target_layout: The desired layout of the result.
    :return: The local array slice in the target layout.
    """
    size = comm.Get_size()
    rank = comm.Get_rank()

    src_indices = get_pencil_indices(rank, size, nq, source_layout)
    tgt_indices = get_pencil_indices(rank, size, nq, target_layout)

    res_mat = np.empty((len(tgt_indices),) + mat.shape[1:], dtype=mat.dtype)
    src_map = {g_idx: l_idx for l_idx, g_idx in enumerate(src_indices)}
    tgt_map = {g_idx: l_idx for l_idx, g_idx in enumerate(tgt_indices)}

    for shift in range(size):
        if shift == 0:
            # Self-overlap: rows this rank both owns (source layout) and needs (target layout). Copy locally instead
            # of round-tripping the data through MPI to itself.
            common = np.intersect1d(src_indices, tgt_indices, assume_unique=True)
            if len(common) > 0:
                res_mat[[tgt_map[g] for g in common]] = mat[[src_map[g] for g in common]]
            continue

        target_rank = (rank + shift) % size
        source_rank = (rank - shift) % size

        remote_tgt_indices = get_pencil_indices(target_rank, size, nq, target_layout)
        to_send_g = np.intersect1d(src_indices, remote_tgt_indices, assume_unique=True)

        remote_src_indices = get_pencil_indices(source_rank, size, nq, source_layout)
        to_recv_g = np.intersect1d(tgt_indices, remote_src_indices, assume_unique=True)

        reqs = []
        send_buf = None  # keep alive until Waitall
        recv_staging = None

        if len(to_send_g) > 0:
            send_l = [src_map[g] for g in to_send_g]
            send_buf = np.ascontiguousarray(mat[send_l])
            send_view = send_buf.view(np.byte).reshape(-1)
            for i in range(0, send_view.nbytes, MAX_MPI_BYTES):
                reqs.append(comm.Isend(send_view[i : i + MAX_MPI_BYTES], dest=target_rank, tag=shift))

        if len(to_recv_g) > 0:
            recv_staging = np.empty((len(to_recv_g),) + mat.shape[1:], dtype=mat.dtype)
            recv_view = recv_staging.view(np.byte).reshape(-1)
            for i in range(0, recv_view.nbytes, MAX_MPI_BYTES):
                reqs.append(comm.Irecv(recv_view[i : i + MAX_MPI_BYTES], source=source_rank, tag=shift))

        MPI.Request.Waitall(reqs)

        # Now copy from staging into res_mat at the right rows
        if len(to_recv_g) > 0:
            recv_l = [tgt_map[g] for g in to_recv_g]
            res_mat[recv_l] = recv_staging

    return res_mat


def execute_distributed_fft(obj: FourPoint, comm: MPI.Comm) -> FourPoint:
    """
    Main routine: Call this for objects that are local to a rank but in the respective full BZ slice. E.g., after
    a call to :func:`exchange_and_map_irrbz_fullbz`.
    This routine performs a distributed 3D FFT by redistributing the data into pencil decompositions for
    each dimension, performing local FFTs, and then redistributing back to the original layout.
    The final result is that ``obj.mat`` is transformed in place to the Fourier space representation
    corresponding to the full BZ.
    Attention: modifies the object in place!

    :param obj: The :class:`FourPoint` distributed over the full BZ (``flat`` layout), transformed in place.
    :param comm: The MPI communicator.
    :return: The same :class:`FourPoint`, now holding the BZ Fourier transform (back in the ``flat`` layout).
    """
    nq = obj.nq

    # The 3D BZ FFT factorizes into independent, commuting 1D FFTs along z, y and x. A singleton axis (n == 1, e.g.
    # nz == 1 on a 2D/layered lattice) has an identity 1D FFT, so its pencil stage is skipped entirely - one fewer
    # distributed redistribution per skipped axis; the remaining axes give the identical result.
    active_stages = [(n, layout) for n, layout in zip(nq[::-1], ("z_pencil", "y_pencil", "x_pencil")) if n > 1]
    if not active_stages:
        return obj  # a single momentum on every axis: the BZ FFT is the identity

    current_layout = "flat"
    for n, layout in active_stages:
        # Move to this axis's pencil layout (whole lines rank-local), FFT along it, keep the pencil shape.
        obj.mat = _redistribute_p2p(obj.mat, nq, comm, current_layout, layout)
        pencil_shape = obj.mat.shape
        obj.mat = obj.mat.reshape(-1, n, *pencil_shape[1:])
        obj.mat = fft.fftn(obj.mat, axes=(1,), overwrite_x=True)
        obj.mat = obj.mat.reshape(pencil_shape)
        current_layout = layout

    obj.mat = _redistribute_p2p(obj.mat, nq, comm, current_layout, "flat")
    return obj

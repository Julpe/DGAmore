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
* higher-level **data-movement routines** built on top: the per-node shared-memory windows, the ring transpose that
  turns a row-distributed array into a column-distributed one (the kernel columns of the self-energy contraction),
  and the ordered point-to-point delivery of numbered blocks.

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
import psutil
import numpy as np

from dgamore.n_point_base import DTYPE

# Canonical 2 GB MPI per-message limit. The chunking helpers below take it as an explicit ``limit`` argument so the
# established test hook of monkeypatching ``mpi_utils.MAX_MPI_BYTES`` to force the chunked path keeps working.
MAX_MPI_BYTES = 2**31 - 1


def build_node_shared_array(node_comm, compute_fn, dtype=None):
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
    :param dtype: Storage dtype of the shared buffer; ``None`` takes the dtype of the array the root computed.
    :return: The tuple ``(array, win)`` - the (shared) numpy array on every rank and the MPI window (``None`` for a
        single-rank node).
    """
    is_root = node_comm.Get_rank() == 0
    local = compute_fn() if is_root else None
    if node_comm.Get_size() == 1:
        return local, None
    shape, dtype = node_comm.bcast((local.shape, dtype or local.dtype) if is_root else None)
    shared, win = allocate_node_shared_array(node_comm, shape, dtype)
    if is_root:
        shared[...] = local
    node_comm.Barrier()
    return shared, win


def allocate_node_shared_array(node_comm, shape: tuple, dtype=DTYPE):
    r"""
    Allocates one MPI shared-memory window holding an array of the given shape on a shared-memory communicator and
    returns an (uninitialized) numpy view of it on every rank; the caller fills the array and owns the window (free
    it once every rank is done reading, after a barrier). A single-rank communicator, or ``None``, gets a private
    array and ``win = None`` instead, which keeps single-rank and mock communicators working.

    :param node_comm: The node-local (shared-memory) communicator, or ``None``.
    :param shape: Shape of the shared array.
    :param dtype: Storage dtype of the shared buffer (defaults to the global ``DTYPE``, complex64).
    :return: The tuple ``(array, win)`` - the shared numpy array on every rank and the MPI window (``None`` for a
        single-rank communicator).
    """
    if node_comm is None or node_comm.Get_size() == 1:
        return np.empty(shape, dtype=dtype), None
    itemsize = np.dtype(dtype).itemsize
    nbytes = int(np.prod(shape)) * itemsize if node_comm.Get_rank() == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=node_comm)
    buf, _ = win.Shared_query(0)
    return np.ndarray(buffer=buf, dtype=dtype, shape=shape), win


def cgroup_memory_limit(proc_file: str = "/proc/self/cgroup", cgroup_root: str = "/sys/fs/cgroup") -> int | None:
    """
    Returns this process's effective cgroup memory limit in bytes, or ``None`` when no limit is set (or none is
    readable). Batch schedulers such as slurm enforce a job's memory request through a cgroup, which can be far
    below the node's physical memory, so every memory budget must honor it. The cgroup path is taken from
    ``proc_file``; on cgroup v2 every ancestor's ``memory.max`` is read up to the root (the limit may sit on the
    job level rather than the process's own leaf) and the smallest set value wins, on v1 the memory controller's
    ``memory.limit_in_bytes`` is read directly. Values at or above ``2**62`` mean "unlimited" and are ignored.

    :param proc_file: Path of the process's cgroup membership file.
    :param cgroup_root: Mount point of the cgroup filesystem.
    :return: The smallest configured limit in bytes, or ``None`` if unlimited or undeterminable.
    """
    limits = []
    try:
        entries = dict((line.split(":", 2)[1], line.split(":", 2)[2].strip()) for line in open(proc_file, "r"))
        if "" in entries:  # cgroup v2: one unified hierarchy, limits possibly on an ancestor
            path = os.path.normpath(cgroup_root + entries[""])
            while path.startswith(cgroup_root):
                try:
                    value = open(os.path.join(path, "memory.max"), "r").read().strip()
                    if value != "max":
                        limits.append(int(value))
                except OSError:
                    pass
                path = os.path.dirname(path)
        elif "memory" in entries:  # cgroup v1: the memory controller's own hierarchy
            value = open(cgroup_root + "/memory" + entries["memory"] + "/memory.limit_in_bytes", "r").read()
            limits.append(int(value))
    except (OSError, ValueError, IndexError):
        return None
    limits = [limit for limit in limits if limit < 2**62]
    return min(limits) if limits else None


def job_memory_total() -> int:
    """
    Returns the memory a job may plan with on this node: the hardware total, capped by the cgroup limit when the
    scheduler sets one (see :func:`cgroup_memory_limit`). Chunk budgets derive from this instead of the free
    memory, so the chunking - and with it the floating-point reduction order - stays reproducible across reruns
    (the cgroup limit is part of the job specification, unlike the machine's momentary load).

    :return: The plannable memory of this node in bytes.
    """
    total = psutil.virtual_memory().total
    limit = cgroup_memory_limit()
    return total if limit is None else min(total, limit)


def count_nodes(comm, node_comm) -> int:
    r"""
    Returns the number of distinct nodes ``comm`` spans, i.e. how many copies of a per-node shared array (see
    :func:`build_node_shared_array`) exist across the job - the multiplicity such a quantity must be logged with
    (see :meth:`dgamore.dga_logger.DgaLogger.log_memory_usage`).

    Counted by summing one contribution per node-local root rank, so it is **collective over** ``comm`` and every
    rank must call it. A communicator that sits on a single node short-circuits without communication; that
    condition holds on every rank at once (``node_comm`` is a subset of ``comm``), so no rank is left in the
    reduction alone.

    :param comm: The MPI communicator.
    :param node_comm: The node-local (shared-memory) communicator, e.g. from
        ``comm.Split_type(MPI.COMM_TYPE_SHARED)``; ``None`` means no node topology was determined and counts as one.
    :return: The number of nodes.
    """
    if node_comm is None or node_comm.size >= comm.size:
        return 1
    return comm.allreduce(1 if node_comm.rank == 0 else 0)


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

    def __init__(
        self,
        ntasks: int = 1,
        comm: MPI.Comm = None,
        name: str = "",
        output_path: str = None,
        sizes: np.ndarray | None = None,
    ):
        """
        Distributes the tasks across the communicator and opens this rank's HDF5 spill file.

        :param ntasks: Total number of tasks to distribute (e.g. the number of irreducible q-points).
        :param comm: The MPI communicator across which the tasks are distributed.
        :param name: Prefix for this rank's HDF5 spill file.
        :param output_path: Directory the per-rank HDF5 spill file is created in; if None, no spill file is opened.
        :param sizes: Optional per-rank task counts (contiguous runs in rank order, summing to ``ntasks``) replacing
            the default as-even-as-possible split.
        """
        self._comm = comm
        self._ntasks = ntasks
        self._file = None
        self._my_slice = None
        self._sizes = None
        self._my_size = None
        self._slices = None

        self._distribute_tasks(sizes)

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

    def _distribute_tasks(self, sizes: np.ndarray | None = None):
        """
        Computes the per-rank chunk sizes and slices, distributing the tasks as evenly as possible (excess tasks go to
        the highest ranks) unless explicit sizes are given, and records this rank's own size and slice.

        :param sizes: Optional per-rank task counts summing to ``ntasks``.
        :return: None.
        :raises ValueError: If ``sizes`` does not hold one count per rank summing to ``ntasks``.
        """
        if sizes is not None:
            if len(sizes) != self.mpi_size or int(np.sum(sizes)) != self.ntasks:
                raise ValueError("The explicit task counts must hold one entry per rank and sum to ntasks.")
            self._sizes = np.asarray(sizes, dtype=int)
        else:
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
# Higher-level data-movement routines (column transpose, ordered block exchange).
def transpose_columns(
    local: np.ndarray, mpi_dist: MpiDistributor, cols_of: list[np.ndarray], base_tag: int = 0
) -> np.ndarray:
    r"""
    Transposes a row-distributed array into a column-distributed one. ``local`` holds this rank's rows
    ``[rows_rank, ..., n_a, n_b]`` of an array whose rows are split over the ranks as in ``mpi_dist`` and whose
    columns are the index pairs into its last two axes; ``cols_of[r]`` lists the columns of rank ``r`` as two index
    rows ``[2, n_cols]`` (into the second-to-last and the last axis). Every rank receives all rows of its own
    columns, ``[n_rows, ..., n_cols]`` in the order of ``cols_of[rank]``. The rows travel in a ring, one rank
    distance per step (the step's single send copy is freed before the next step), straight into the result, so no
    staging buffer is held.

    :param local: This rank's rows of the array; only read.
    :param mpi_dist: MPI distributor of the rows (its communicator and per-rank row slices).
    :param cols_of: The column index rows of every rank (identical on every rank).
    :param base_tag: Base MPI tag; successive chunks use ``base_tag + chunk_index``.
    :return: All rows of this rank's columns.
    """
    comm, rank, size = mpi_dist.comm, mpi_dist.my_rank, mpi_dist.mpi_size
    slices = mpi_dist.slices
    mine = cols_of[rank]
    out = np.empty((mpi_dist.ntasks, *local.shape[1:-2], len(mine[0])), dtype=local.dtype)
    out[slices[rank]] = local[..., mine[0], mine[1]]
    for step in range(1, size):
        dst, src = (rank + step) % size, (rank - step) % size
        reqs, send = [], None
        if local.shape[0] and len(cols_of[dst][0]):
            send = np.ascontiguousarray(local[..., cols_of[dst][0], cols_of[dst][1]])
            reqs += _isend_rows(comm, send, dst, base_tag=base_tag, limit=MAX_MPI_BYTES)
        if slices[src].stop > slices[src].start and len(mine[0]):
            reqs += _irecv_rows_into(comm, out[slices[src]], src, base_tag=base_tag, limit=MAX_MPI_BYTES)
        MPI.Request.Waitall(reqs)
        del send
    return out


def exchange_blocks(
    comm, sends: list[tuple[int, np.ndarray]], sources: list[int], shape: tuple, dtype, base_tag: int = 0
) -> list[np.ndarray]:
    r"""
    Delivers numbered array blocks point to point: every send ``(destination, array)`` and every receive (one per
    entry of ``sources``, each into a fresh array of ``shape`` and ``dtype``) is posted at once and all complete in a
    single ``Waitall``. Blocks between one pair of ranks travel on the same tags, so they match in posting order (MPI
    non-overtaking): both sides must list them in the same order. A single-rank communicator exchanges nothing.

    :param comm: The MPI communicator.
    :param sends: The outgoing blocks as ``(destination, C-contiguous array)`` pairs, in delivery order; the arrays
        stay alive until the exchange completes.
    :param sources: The source rank of every incoming block, in delivery order.
    :param shape: Shape of every incoming block.
    :param dtype: Element type of every incoming block.
    :param base_tag: Base MPI tag; successive chunks use ``base_tag + chunk_index``.
    :return: The received blocks, in the order of ``sources``.
    """
    if comm.size == 1:
        return []
    received = [np.empty(shape, dtype=dtype) for _ in sources]
    reqs = []
    for dst, block in sends:
        reqs += _isend_rows(comm, block, dst, base_tag=base_tag, limit=MAX_MPI_BYTES)
    for src, buf in zip(sources, received):
        reqs += _irecv_rows_into(comm, buf, src, base_tag=base_tag, limit=MAX_MPI_BYTES)
    MPI.Request.Waitall(reqs)
    return received


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

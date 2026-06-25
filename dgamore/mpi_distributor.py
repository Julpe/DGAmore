# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
MPI work distribution. :class:`MpiDistributor` splits a number of tasks (typically the q-points of the irreducible
Brillouin zone) into per-rank contiguous slices and provides the collective/point-to-point communication used
throughout the non-local step — scatter, gather, all-gather, all-reduce and broadcast — all of which transparently
chunk arrays that exceed the 2 GB MPI message limit. Each rank also owns a private HDF5 file for spilling
intermediate results without write conflicts.
"""

import gc
import os
import pickle

import h5py
import mpi4py.MPI as MPI
import numpy as np

import dgamore.config as config

MAX_MPI_BYTES = 2**31 - 1


class MpiDistributor:
    """
    Distributes tasks among all available cores. Uses the first (q) dimension to slice the vertex data into chunks
    and sends it to all active MPI processes. Saves intermediate computational results in rank files. Each rank
    has their own instance of an MPI distributor and hdf5-file to avoid write conflicts.
    """

    def __init__(self, ntasks: int = 1, comm: MPI.Comm = None, name: str = ""):
        """
        Distributes the tasks across the communicator and opens this rank's HDF5 spill file.

        :param ntasks: Total number of tasks to distribute (e.g. the number of irreducible q-points).
        :param comm: The MPI communicator across which the tasks are distributed.
        :param name: Prefix for this rank's HDF5 spill file (created under ``config.output.output_path`` if set).
        """
        self._comm = comm
        self._ntasks = ntasks
        self._file = None
        self._my_slice = None
        self._sizes = None
        self._my_size = None
        self._slices = None

        self._distribute_tasks()

        if config.output.output_path is not None:
            # creates rank file if it does not exist
            self._fname = os.path.join(config.output.output_path, f"{name}_Rank{self.my_rank:05d}.hdf5")
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
        Gathers each rank's array slice (along axis 0) into the full array, replicated on every rank. Handles the 2 GB
        MPI message limit by chunking the broadcasts.

        :param rank_result: This rank's slice of the result (leading axis indexes the rank's tasks).
        :return: The full array of shape ``(ntasks, ...)`` on all ranks.
        """
        rank_result = np.ascontiguousarray(rank_result)
        tot_shape = (self.ntasks,) + rank_result.shape[1:]
        tot_result = np.empty(tot_shape, dtype=rank_result.dtype)

        itemsize = rank_result.dtype.itemsize
        items_per_q = int(np.prod(rank_result.shape[1:]))
        max_q_per_chunk = max(1, MAX_MPI_BYTES // (itemsize * items_per_q))

        for r in range(self.mpi_size):
            n_q = self.sizes[r]
            start_idx = self._slices[r].start

            for i in range(0, n_q, max_q_per_chunk):
                j = min(n_q, i + max_q_per_chunk)
                chunk_view = tot_result[start_idx + i : start_idx + j]

                if self.my_rank == r:
                    chunk_view[...] = rank_result[i:j]
                self.comm.Bcast(chunk_view, root=r)
        return tot_result

    def gather(self, rank_result: np.ndarray = None, root: int = 0) -> np.ndarray:
        """
        Gathers each rank's array slice into the full array, in correct task order, on the ``root`` rank only. Handles
        arrays exceeding the 2 GB MPI limit by chunking along axis 0.

        :param rank_result: This rank's slice of the result (leading axis indexes the rank's tasks).
        :param root: The rank that collects the full array.
        :return: The full array of shape ``(ntasks, ...)`` on ``root``, ``None`` on the other ranks.
        """

        def chunk_bounds(n_items: int, itemsize: int, items_per_element: int):
            """
            Yields ``(start, stop)`` index pairs splitting ``n_items`` rows into below-2 GB message chunks.

            :param n_items: Number of rows (axis-0 elements) to split.
            :param itemsize: Size in bytes of one array element.
            :param items_per_element: Number of scalars per axis-0 element (product of trailing dimensions).
            :return: A generator of ``(start, stop)`` row-index pairs.
            """
            max_elems = max(1, MAX_MPI_BYTES // (itemsize * items_per_element))
            for i in range(0, n_items, max_elems):
                yield i, min(n_items, i + max_elems)

        def send_in_chunks(arr: np.ndarray, dest: int, base_tag: int = 0):
            """
            Sends ``arr`` to ``dest`` in below-2 GB chunks along axis 0.

            :param arr: The array to send.
            :param dest: Destination rank.
            :param base_tag: Base MPI tag; successive chunks use ``base_tag + idx``.
            :return: None.
            """
            arr = np.ascontiguousarray(arr)
            itemsize = arr.dtype.itemsize
            items_per_element = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
            for idx, (i, j) in enumerate(chunk_bounds(arr.shape[0], itemsize, items_per_element)):
                self.comm.Send(arr[i:j], dest=dest, tag=base_tag + idx)

        def recv_in_chunks(buf: np.ndarray, offset: int, n_items: int, source: int, base_tag: int = 0):
            """
            Receives ``n_items`` rows from ``source`` in below-2 GB chunks and writes them into ``buf`` at ``offset``.

            :param buf: Destination buffer to write the received rows into.
            :param offset: Row offset in ``buf`` at which to start writing.
            :param n_items: Number of rows to receive.
            :param source: Source rank.
            :param base_tag: Base MPI tag; successive chunks use ``base_tag + idx``.
            :return: None.
            """
            itemsize = buf.dtype.itemsize
            items_per_element = int(np.prod(buf.shape[1:])) if buf.ndim > 1 else 1
            for idx, (i, j) in enumerate(chunk_bounds(n_items, itemsize, items_per_element)):
                tmp = np.empty((j - i,) + buf.shape[1:], dtype=buf.dtype)
                self.comm.Recv(tmp, source=source, tag=base_tag + idx)
                buf[offset + i : offset + j] = tmp

        rank_result = np.ascontiguousarray(rank_result)
        rest_shape = rank_result.shape[1:]

        tot_result = np.empty((self.ntasks,) + rest_shape, dtype=rank_result.dtype) if self.my_rank == root else None

        if self.my_rank == root:
            # copy own slice directly
            sl = self._slices[root]
            tot_result[sl] = rank_result

            # receive from all other ranks
            for r in range(self.mpi_size):
                if r == root:
                    continue
                n = self._sizes[r]
                if n == 0:
                    continue
                recv_in_chunks(tot_result, self._slices[r].start, n, source=r, base_tag=0)
        else:
            if rank_result.shape[0] > 0:
                send_in_chunks(rank_result, dest=root, base_tag=0)

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

        def chunk_bounds(n_items: int, itemsize: int, items_per_element: int):
            """
            Yields ``(start, stop)`` index pairs splitting ``n_items`` rows into below-2 GB message chunks.

            :param n_items: Number of rows (axis-0 elements) to split.
            :param itemsize: Size in bytes of one array element.
            :param items_per_element: Number of scalars per axis-0 element (product of trailing dimensions).
            :return: A generator of ``(start, stop)`` row-index pairs.
            """
            max_elems = max(1, MAX_MPI_BYTES // (itemsize * items_per_element))
            for i in range(0, n_items, max_elems):
                yield i, min(n_items, i + max_elems)

        def send_in_chunks(arr: np.ndarray, dest: int, base_tag: int = 0):
            """
            Sends ``arr`` to ``dest`` in below-2 GB chunks along axis 0.

            :param arr: The array to send.
            :param dest: Destination rank.
            :param base_tag: Base MPI tag; successive chunks use ``base_tag + idx``.
            :return: None.
            """
            arr = np.ascontiguousarray(arr)
            itemsize = arr.dtype.itemsize
            items_per_element = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
            for idx, (i, j) in enumerate(chunk_bounds(arr.shape[0], itemsize, items_per_element)):
                self.comm.Send(arr[i:j], dest=dest, tag=base_tag + idx)

        def recv_in_chunks(shape, dtype, source: int, base_tag: int = 0):
            """
            Receives an array of the given shape/dtype from ``source`` in below-2 GB chunks along axis 0.

            :param shape: Shape of the array to receive.
            :param dtype: Dtype of the array to receive.
            :param source: Source rank.
            :param base_tag: Base MPI tag; successive chunks use ``base_tag + idx``.
            :return: The received array.
            """
            out = np.empty(shape, dtype=dtype)
            itemsize = np.dtype(dtype).itemsize
            items_per_element = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            for idx, (i, j) in enumerate(chunk_bounds(shape[0], itemsize, items_per_element)):
                tmp = np.empty((j - i,) + tuple(shape[1:]), dtype=dtype)
                self.comm.Recv(tmp, source=source, tag=base_tag + idx)
                out[i:j] = tmp
            return out

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

        data_type = self.comm.bcast(data_type, root)
        rest_shape = self.comm.bcast(rest_shape, root)

        rank_shape = (self._my_size,) + rest_shape if rest_shape else (self._my_size,)
        rank_data = np.empty(rank_shape, dtype=data_type)

        if self.my_rank == root:
            if full_data is None:
                return rank_data

            full_data = np.asarray(full_data, dtype=data_type)

            if data_len == self.ntasks:
                for r in range(self.mpi_size):
                    n = self._sizes[r]
                    if n == 0:
                        continue
                    sl = self._slices[r]
                    if r == root:
                        rank_data[...] = full_data[sl]
                    else:
                        send_in_chunks(full_data[sl], dest=r, base_tag=0)
            elif data_len == self._my_size and self.mpi_size == 1:
                rank_data[...] = np.ascontiguousarray(full_data)
            else:
                raise ValueError(f"Mismatch in scatter!")
        else:
            if self._my_size > 0:
                rank_data = recv_in_chunks(rank_shape, data_type, source=root, base_tag=0)

        return rank_data

    def send_to_rank(self, obj, dest: int, base_tag: int = 0):
        """
        Sends an N-point-like object to a single rank. The large ``.mat`` array is sent as raw chunks (to avoid
        holding a full pickle blob in memory), while the rest of the object is pickled into a small metadata blob.

        :param obj: The object to send; must expose a ``.mat`` numpy array attribute.
        :param dest: Destination rank.
        :param base_tag: Base MPI tag (metadata uses ``base_tag``, array chunks ``base_tag + 500 + ...``).
        :return: None.
        """

        def send_bytes(data: bytes, tag_offset: int):
            """
            Sends a raw byte blob to ``dest`` in below-2 GB chunks (with a leading length message).

            :param data: The bytes to send.
            :param tag_offset: Tag offset added to ``base_tag`` for this blob's messages.
            :return: None.
            """
            total = len(data)
            self.comm.send(total, dest=dest, tag=base_tag + tag_offset)
            offset = 0
            chunk_idx = 1
            while offset < total:
                end = min(offset + MAX_MPI_BYTES, total)
                chunk = np.frombuffer(data[offset:end], dtype=np.uint8)
                self.comm.Send(chunk, dest=dest, tag=base_tag + tag_offset + chunk_idx)
                offset = end
                chunk_idx += 1

        def send_array(arr: np.ndarray, tag_offset: int):
            """
            Sends a numpy array to ``dest`` in below-2 GB chunks along axis 0 (preceded by its shape/dtype).

            :param arr: The array to send.
            :param tag_offset: Tag offset added to ``base_tag`` for this array's messages.
            :return: None.
            """
            arr = np.ascontiguousarray(arr)
            itemsize = arr.dtype.itemsize
            items_per_element = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
            max_elems = max(1, MAX_MPI_BYTES // (itemsize * items_per_element))
            # Send shape/dtype so receiver can allocate
            self.comm.send({"shape": arr.shape, "dtype": arr.dtype}, dest=dest, tag=base_tag + tag_offset)
            for idx, i in enumerate(range(0, arr.shape[0], max_elems)):
                j = min(arr.shape[0], i + max_elems)
                self.comm.Send(np.ascontiguousarray(arr[i:j]), dest=dest, tag=base_tag + tag_offset + 1 + idx)

        # Temporarily detach .mat so it is not included in the pickle
        mat = obj.mat
        obj.mat = None
        try:
            meta_bytes = pickle.dumps(obj)
        finally:
            obj.mat = mat  # always restore, even if pickle raises

        send_bytes(meta_bytes, tag_offset=0)  # tag_offset 0    : metadata blob
        send_array(mat, tag_offset=500)  # tag_offset 500  : raw array chunks

    def recv_from_rank(self, source: int, base_tag: int = 0):
        """
        Receives an object sent by :meth:`send_to_rank`: reconstructs the pickled metadata object and reattaches the
        chunk-received ``.mat`` array.

        :param source: Source rank.
        :param base_tag: Base MPI tag matching the one used by :meth:`send_to_rank`.
        :return: The reconstructed object with its ``.mat`` array attached.
        """

        def recv_bytes(tag_offset: int) -> bytes:
            """
            Receives a chunked raw byte blob from ``source``.

            :param tag_offset: Tag offset added to ``base_tag`` for this blob's messages.
            :return: The reassembled bytes.
            """
            total = self.comm.recv(source=source, tag=base_tag + tag_offset)
            buf = bytearray(total)
            offset = 0
            chunk_idx = 1
            while offset < total:
                end = min(offset + MAX_MPI_BYTES, total)
                chunk = np.empty(end - offset, dtype=np.uint8)
                self.comm.Recv(chunk, source=source, tag=base_tag + tag_offset + chunk_idx)
                buf[offset:end] = chunk.tobytes()
                offset = end
                chunk_idx += 1
            return bytes(buf)

        def recv_array(tag_offset: int) -> np.ndarray:
            """
            Receives a chunked numpy array from ``source`` (shape/dtype received first).

            :param tag_offset: Tag offset added to ``base_tag`` for this array's messages.
            :return: The received array.
            """
            meta = self.comm.recv(source=source, tag=base_tag + tag_offset)
            shape, dtype = meta["shape"], meta["dtype"]
            out = np.empty(shape, dtype=dtype)
            itemsize = np.dtype(dtype).itemsize
            items_per_element = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            max_elems = max(1, MAX_MPI_BYTES // (itemsize * items_per_element))
            for idx, i in enumerate(range(0, shape[0], max_elems)):
                j = min(shape[0], i + max_elems)
                tmp = np.empty((j - i,) + shape[1:], dtype=dtype)
                self.comm.Recv(tmp, source=source, tag=base_tag + tag_offset + 1 + idx)
                out[i:j] = tmp
            return out

        obj = pickle.loads(recv_bytes(tag_offset=0))
        obj.mat = recv_array(tag_offset=500)
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
        # 1. Share metadata (shape and dtype) using lowercase bcast
        shape = self.comm.bcast(arr.shape if self.my_rank == root else None, root=root)
        dtype = self.comm.bcast(arr.dtype if self.my_rank == root else None, root=root)

        # 2. Prepare the buffer on non-root ranks
        if self.my_rank != root:
            arr = np.empty(shape, dtype=dtype)

        # Ensure the array is contiguous for the MPI buffer
        # This is a view if already contiguous, otherwise a copy
        arr = np.ascontiguousarray(arr)

        # 3. Calculate chunking bounds based on MAX_MPI_BYTES
        itemsize = arr.dtype.itemsize
        # Number of items along the non-slicing dimensions
        items_per_element = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        max_q_per_chunk = max(1, MAX_MPI_BYTES // (itemsize * items_per_element))

        # 4. Perform chunked collective Broadcast
        # Since Bcast is collective, ALL ranks must enter this loop
        for i in range(0, shape[0], max_q_per_chunk):
            j = min(shape[0], i + max_q_per_chunk)
            # Use a slice view to broadcast piece by piece
            self.comm.Bcast(arr[i:j], root=root)

        return arr

    def allreduce(self, rank_result=None) -> np.ndarray:
        """
        Sums an array element-wise across all ranks in place and returns the result on every rank.

        :param rank_result: This rank's contribution; reduced in place.
        :return: The summed array (same buffer), identical on all ranks.
        """
        self.comm.Allreduce(MPI.IN_PLACE, rank_result)
        return rank_result

    @staticmethod
    def create_distributor(ntasks: int, comm: MPI.Comm, name: str = "") -> "MpiDistributor":
        """
        Factory that creates an :class:`MpiDistributor`, defaulting to ``MPI.COMM_WORLD`` if no communicator is given.

        :param ntasks: Total number of tasks to distribute.
        :param comm: The MPI communicator (``MPI.COMM_WORLD`` if None).
        :param name: Prefix for the per-rank HDF5 spill file.
        :return: The created :class:`MpiDistributor`.
        """
        if comm is None:
            comm = MPI.COMM_WORLD
        return MpiDistributor(ntasks=ntasks, comm=comm, name=name)

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

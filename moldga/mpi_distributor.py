# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import gc
import os
import pickle

import h5py
import mpi4py.MPI as MPI
import numpy as np

import moldga.config as config

MAX_MPI_BYTES = 2**31 - 1


class MpiDistributor:
    """
    Distributes tasks among all available cores. Uses the first (q) dimension to slice the vertex data into chunks
    and sends it to all active MPI processes. Saves intermediate computational results in rank files. Each rank
    has their own instance of an MPI distributor and hdf5-file to avoid write conflicts.
    """

    def __init__(self, ntasks: int = 1, comm: MPI.Comm = None, name: str = ""):
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
            except:
                pass

    def __enter__(self):
        """
        Context manager to open the hdf5 file.
        """
        self.open_file()
        return self._file

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager to close the hdf5 file.
        """
        if self._file:
            self.close_file()

    @property
    def comm(self) -> MPI.Comm:
        """
        Returns the MPI communicator.
        """
        return self._comm

    @property
    def is_root(self) -> bool:
        """
        Returns True if the current rank is the root rank (rank 0).
        """
        return self.my_rank == 0

    @property
    def ntasks(self) -> int:
        """
        Returns the total number of tasks to be distributed, i.e. in our case the total number of q-points in the
        irreducible Brillouin zone.
        """
        return self._ntasks

    @property
    def sizes(self) -> np.ndarray:
        """
        Returns the sizes of the chunks for each rank.
        """
        return self._sizes

    @property
    def slices(self) -> np.ndarray:
        """
        Returns the slices for each rank.
        """
        return self._slices

    @property
    def my_rank(self) -> int:
        """
        Returns the rank of the current process.
        """
        return self._comm.Get_rank()

    @property
    def my_tasks(self) -> np.ndarray:
        """
        Returns the tasks assigned to the current rank, i.e. the q-points the current rank has to process.
        """
        return np.arange(0, self.ntasks)[self.my_slice]

    @property
    def mpi_size(self) -> int:
        """
        Returns the total number of MPI processes.
        """
        return self._comm.size

    @property
    def my_size(self) -> int:
        """
        Returns the number of tasks assigned to the current rank, i.e. the number of q-points the current rank has to
        process.
        """
        return self._my_size

    @property
    def my_slice(self) -> int:
        """
        Returns the slice object for the current rank to slice the full q-list to the q-list for that rank.
        """
        return self._my_slice

    def open_file(self):
        """
        Opens the hdf5 file for the current rank.
        """
        try:
            self._file = h5py.File(self._fname, "r+")
        except:
            pass

    def close_file(self):
        """
        Closes the hdf5 file for the current rank.
        """
        try:
            self._file.close()
        except:
            pass

    def delete_file(self):
        """
        Deletes the hdf5 file for the current rank.
        """
        try:
            os.remove(self._fname)
        except:
            pass

    def barrier(self):
        """
        Waits for all ranks until each MPI process has hit this statement. Explicitly calls garbage collection before
        to make sure that all ranks have freed their memory before synchronization.
        """
        gc.collect()
        self.comm.Barrier()

    def allgather(self, rank_result: np.ndarray = None) -> np.ndarray:
        """
        Gathers the numpy array from all ranks into a full array on all ranks.
        Handles the 2GB MPI limit by chunking broadcasts across the communicator.
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
        Gathers the numpy array from all ranks in the correct q-list order to the root rank.
        Handles arrays exceeding the 2 GB MPI limit by chunking along axis 0.
        """

        def chunk_bounds(n_items: int, itemsize: int, items_per_element: int):
            max_elems = max(1, MAX_MPI_BYTES // (itemsize * items_per_element))
            for i in range(0, n_items, max_elems):
                yield i, min(n_items, i + max_elems)

        def send_in_chunks(arr: np.ndarray, dest: int, base_tag: int = 0):
            arr = np.ascontiguousarray(arr)
            itemsize = arr.dtype.itemsize
            items_per_element = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
            for idx, (i, j) in enumerate(chunk_bounds(arr.shape[0], itemsize, items_per_element)):
                self.comm.Send(arr[i:j], dest=dest, tag=base_tag + idx)

        def recv_in_chunks(buf: np.ndarray, offset: int, n_items: int, source: int, base_tag: int = 0):
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
        Scatters the data along the first axis.
        """

        def chunk_bounds(n_items: int, itemsize: int, items_per_element: int):
            max_elems = max(1, MAX_MPI_BYTES // (itemsize * items_per_element))
            for i in range(0, n_items, max_elems):
                yield i, min(n_items, i + max_elems)

        def send_in_chunks(arr: np.ndarray, dest: int, base_tag: int = 0):
            arr = np.ascontiguousarray(arr)
            itemsize = arr.dtype.itemsize
            items_per_element = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
            for idx, (i, j) in enumerate(chunk_bounds(arr.shape[0], itemsize, items_per_element)):
                self.comm.Send(arr[i:j], dest=dest, tag=base_tag + idx)

        def recv_in_chunks(shape, dtype, source: int, base_tag: int = 0):
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
        Send an object to a single rank. The .mat array is sent as raw bytes
        in chunks to avoid holding the full pickle blob in memory.
        Everything else is pickled and sent as a small metadata blob.
        """

        def send_bytes(data: bytes, tag_offset: int):
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
        Receive an object sent by send_to_rank.
        Reconstructs the metadata object then reattaches the .mat array.
        """

        def recv_bytes(tag_offset: int) -> bytes:
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
        Broadcasts data from the root rank to all other ranks.
        """
        return self.comm.bcast(data, root=root)

    def bcast_chunked(self, arr: np.ndarray, root: int = 0) -> np.ndarray:
        """
        Broadcasts a large numpy array from root to all ranks.
        Handles the 2GB MPI limit by chunking along the first axis and
        utilizes raw MPI buffers for maximum performance.
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
        Reduces the numpy array from all ranks by summing it up and returns the result on all ranks.
        """
        self.comm.Allreduce(MPI.IN_PLACE, rank_result)
        return rank_result

    @staticmethod
    def create_distributor(ntasks: int, comm: MPI.Comm, name: str = "") -> "MpiDistributor":
        """
        Factory method to create an MpiDistributor instance.
        """
        if comm is None:
            comm = MPI.COMM_WORLD
        return MpiDistributor(ntasks=ntasks, comm=comm, name=name)

    def _distribute_tasks(self):
        """
        Distributes the tasks among all ranks. Calculates the sizes and slices for each rank.
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

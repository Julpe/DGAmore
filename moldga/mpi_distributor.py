# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import gc
import os

import h5py
import mpi4py.MPI as MPI
import numpy as np

import moldga.config as config


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
        Gathers the numpy array from all ranks in the correct q-list order.
        """
        tot_shape = (self.ntasks,) + rank_result.shape[1:]
        tot_result = np.empty(tot_shape, rank_result.dtype)
        # tot_result[...] = np.nan
        other_dims = np.prod(rank_result.shape[1:])

        # The sizes argument needs the total number of elements rather than
        # just the first axis. The type argument is inferred.
        self.comm.Allgatherv(rank_result, [tot_result, self.sizes * other_dims])
        return tot_result

    def gather(self, rank_result: np.ndarray = None, root: int = 0) -> np.ndarray:
        """
        Gathers the numpy array from all ranks in the correct q-list order to the root rank.
        """
        rank_result = np.ascontiguousarray(rank_result)
        other_dims = int(np.prod(rank_result.shape[1:]))
        tot_result = (
            np.empty((self.ntasks,) + rank_result.shape[1:], dtype=rank_result.dtype)
            if self.comm.rank == root
            else None
        )
        self.comm.Gatherv(
            rank_result, [tot_result, self.sizes * other_dims] if self.comm.rank == root else None, root=root
        )
        return tot_result

    def scatter(self, full_data: np.ndarray = None, root: int = 0) -> np.ndarray:
        """
        Scatters the data along the first axis.
        """
        if self.my_rank == root:
            rest_shape = full_data.shape[1:]
            data_type = full_data.dtype
        else:

            rest_shape = None
            data_type = None

        rest_shape = self.comm.bcast(rest_shape, root=root)
        data_type = self.comm.bcast(data_type, root=root)

        rank_shape = (self._my_size,) + rest_shape
        rank_data = np.empty(rank_shape, dtype=data_type)

        other_dims = int(np.prod(rest_shape)) if rest_shape else 1
        itemsize = np.dtype(data_type).itemsize
        MAX_MPI_BYTES = 2**31 - 1
        max_rows = max(1, MAX_MPI_BYTES // (itemsize * other_dims))

        if self.my_rank == root:
            full_data = np.ascontiguousarray(full_data, dtype=data_type)

        # track how many rows have been sent/received per rank
        sent = np.zeros(self.mpi_size, dtype=int)
        received = 0

        while received < self._my_size:
            # compute chunk sizes respecting _sizes and _slices per rank
            chunk_sizes = None
            if self.my_rank == root:
                chunk_sizes = np.array([min(max_rows, self._sizes[r] - sent[r]) for r in range(self.mpi_size)])
                chunk_sizes = np.maximum(chunk_sizes, 0)

            chunk_sizes = self.comm.bcast(chunk_sizes, root=root)
            my_chunk = chunk_sizes[self.my_rank]

            if my_chunk == 0:
                break

            send_buf = None
            if self.my_rank == root:
                # use _slices to get the correct data for each rank
                rows = [
                    full_data[self._slices[r].start + sent[r] : self._slices[r].start + sent[r] + chunk_sizes[r]]
                    for r in range(self.mpi_size)
                ]
                send_buf = np.ascontiguousarray(np.concatenate(rows, axis=0))

            recv_buf = np.empty((my_chunk,) + rest_shape, dtype=data_type)
            self.comm.Scatterv(
                [send_buf, chunk_sizes * other_dims] if self.my_rank == root else None, recv_buf, root=root
            )

            rank_data[received : received + my_chunk] = recv_buf
            received += my_chunk
            if self.my_rank == root:
                sent += chunk_sizes

        return rank_data

    def bcast(self, data, root=0):
        """
        Broadcasts data from the root rank to all other ranks.
        """
        return self.comm.bcast(data, root=root)

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

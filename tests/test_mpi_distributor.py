"""Unit tests for dgamore.mpi_distributor.MpiDistributor.

All MPI behaviour is provided by an in-process thread-based fake communicator
(set up in conftest.py), so no MPI installation is required.
"""

import os

import numpy as np
import pytest

import dgamore.config as config
import dgamore.mpi_distributor as md
from dgamore.mpi_distributor import MpiDistributor

from tests.conftest import run_parallel, FAKE_MPI as MPI

# The shared conftest installs an autouse fixture that no-ops os.remove (so the
# rest of the suite never deletes real files). Capture the genuine os.remove
# here, at import time, so the rank-file lifecycle test can opt back in to real
# deletion for its own assertion without disturbing that fixture.
_REAL_OS_REMOVE = os.remove


@pytest.fixture(autouse=True)
def _use_fake_mpi(monkeypatch):
    # Inject the thread-based fake communicator into the module under test for
    # the duration of these tests only; real mpi4py is left untouched elsewhere.
    monkeypatch.setattr(md, "MPI", MPI)


# A picklable object carrying a `.mat` array, matching what send_to_rank expects.
class Holder:
    def __init__(self, mat, label="obj"):
        self.mat = mat
        self.label = label


def comm1():
    """A size-1 communicator usable directly on the main thread."""
    return MPI.Comm(1)


@pytest.fixture(autouse=True)
def _default_no_output_path(monkeypatch):
    # The real config defaults output_path to "" (not None), which makes the
    # distributor create a rank file on construction. Default it to None for the
    # whole module so plain distributor tests create no files; the file-lifecycle
    # tests opt back in by setting output_path to a tmp_path in their own body.
    monkeypatch.setattr(config.output, "output_path", None)


# --------------------------------------------------------------------------- #
# Task distribution & basic properties
# --------------------------------------------------------------------------- #
def test_distribute_tasks_with_excess():
    d = MpiDistributor(ntasks=7, comm=MPI.Comm(3))
    # 7 tasks over 3 ranks -> excess of 1 lands on the last rank.
    assert list(d.sizes) == [2, 2, 3]
    assert [(s.start, s.stop) for s in d.slices] == [(0, 2), (2, 4), (4, 7)]


def test_distribute_tasks_even_no_excess():
    d = MpiDistributor(ntasks=6, comm=MPI.Comm(3))
    assert list(d.sizes) == [2, 2, 2]


def test_distribute_tasks_fewer_than_ranks():
    d = MpiDistributor(ntasks=1, comm=MPI.Comm(3))
    assert list(d.sizes) == [0, 0, 1]


def test_properties_single_rank():
    d = MpiDistributor(ntasks=5, comm=comm1())
    assert d.ntasks == 5
    assert d.mpi_size == 1
    assert d.my_rank == 0
    assert d.is_root is True
    assert d.my_size == 5
    assert isinstance(d.my_slice, slice)
    assert np.array_equal(d.my_tasks, np.arange(5))
    assert d.comm is not None
    assert np.array_equal(d.sizes, [5])
    assert len(d.slices) == 1


def test_is_root_false_on_nonzero_rank():
    def fn(comm, rank):
        d = MpiDistributor(ntasks=4, comm=comm)
        return d.is_root, d.my_rank

    _, res = run_parallel(3, fn)
    assert res[0] == (True, 0)
    assert res[1] == (False, 1)
    assert res[2] == (False, 2)


# --------------------------------------------------------------------------- #
# HDF5 rank-file lifecycle (h5py is faked)
# --------------------------------------------------------------------------- #
def test_rankfile_created_and_context_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(config.output, "output_path", str(tmp_path))
    d = MpiDistributor(ntasks=3, comm=comm1(), name="green")
    assert d._fname.endswith("green_Rank00000.hdf5")
    # context manager opens then closes the file
    with d as f:
        assert f is not None
    # explicit open/close/delete cycle (opt back in to real deletion, since the
    # shared conftest's autouse fixture otherwise no-ops os.remove)
    monkeypatch.setattr(os, "remove", _REAL_OS_REMOVE)
    d.open_file()
    d.close_file()
    d.delete_file()
    assert not os.path.exists(d._fname)


def test_open_close_delete_are_safe_without_file():
    # No output_path -> no _fname attribute; all file ops must swallow errors.
    d = MpiDistributor(ntasks=2, comm=comm1())
    assert d._file is None
    d.open_file()  # AttributeError on missing _fname -> swallowed
    d.close_file()  # None.close() -> swallowed
    d.delete_file()  # os.remove(missing) -> swallowed


def test_del_closes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config.output, "output_path", str(tmp_path))
    d = MpiDistributor(ntasks=1, comm=comm1(), name="del")
    fname = d._fname
    d.open_file()
    d.__del__()  # exercise destructor path directly
    assert fname.endswith("del_Rank00000.hdf5")


def test_exit_without_open_file():
    d = MpiDistributor(ntasks=1, comm=comm1())
    # _file is None -> __exit__ must not attempt to close
    d.__exit__(None, None, None)


# --------------------------------------------------------------------------- #
# barrier
# --------------------------------------------------------------------------- #
def test_barrier_runs_on_all_ranks():
    def fn(comm, rank):
        d = MpiDistributor(ntasks=3, comm=comm)
        d.barrier()
        return rank

    _, res = run_parallel(3, fn)
    assert sorted(res) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# allgather
# --------------------------------------------------------------------------- #
def test_allgather_reassembles_full_array():
    full = (np.arange(7 * 2).reshape(7, 2) + 0.25).astype(np.float64)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=7, comm=comm)
        return d.allgather(full[d.my_slice])

    _, res = run_parallel(3, fn)
    for r in res:
        assert np.allclose(r, full)


def test_allgather_single_rank():
    d = MpiDistributor(ntasks=4, comm=comm1())
    local = np.arange(4, dtype=float)[:, None]
    out = d.allgather(local)
    assert np.allclose(out, local)


def test_allgather_chunked(monkeypatch):
    full = (np.arange(6 * 3).reshape(6, 3) + 1j).astype(np.complex128)
    # Force several chunks per rank: one complex row = 48 bytes.
    monkeypatch.setattr(md, "MAX_MPI_BYTES", 16)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=6, comm=comm)
        return d.allgather(full[d.my_slice])

    _, res = run_parallel(3, fn)
    for r in res:
        assert np.allclose(r, full)


# --------------------------------------------------------------------------- #
# gather
# --------------------------------------------------------------------------- #
def test_gather_to_root():
    full = (np.arange(7 * 3).reshape(7, 3) + 1j).astype(np.complex128)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=7, comm=comm)
        return d.gather(full[d.my_slice], root=0)

    _, res = run_parallel(3, fn)
    assert np.allclose(res[0], full)
    assert res[1] is None and res[2] is None


def test_gather_with_empty_ranks():
    # ntasks=1, size=3 -> sizes [0, 0, 1]: a non-root rank has zero tasks.
    full = np.arange(1 * 2).reshape(1, 2) + 3.0

    def fn(comm, rank):
        d = MpiDistributor(ntasks=1, comm=comm)
        return d.gather(full[d.my_slice], root=0)

    _, res = run_parallel(3, fn)
    assert np.allclose(res[0], full)


def test_gather_chunked(monkeypatch):
    full = (np.arange(7 * 3).reshape(7, 3) + 1j).astype(np.complex128)
    monkeypatch.setattr(md, "MAX_MPI_BYTES", 16)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=7, comm=comm)
        return d.gather(full[d.my_slice], root=0)

    _, res = run_parallel(3, fn)
    assert np.allclose(res[0], full)


def test_gather_single_rank():
    d = MpiDistributor(ntasks=3, comm=comm1())
    arr = np.arange(3 * 2).reshape(3, 2).astype(float)
    out = d.gather(arr, root=0)
    assert np.allclose(out, arr)


# --------------------------------------------------------------------------- #
# scatter
# --------------------------------------------------------------------------- #
def test_scatter_distributes_rows():
    full = (np.arange(7 * 3).reshape(7, 3) + 1j).astype(np.complex128)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=7, comm=comm)
        return d.scatter(full if rank == 0 else None, root=0)

    _, res = run_parallel(3, fn)
    assert np.allclose(np.concatenate(res, axis=0), full)


def test_scatter_chunked(monkeypatch):
    full = (np.arange(7 * 3).reshape(7, 3) + 1j).astype(np.complex128)
    monkeypatch.setattr(md, "MAX_MPI_BYTES", 16)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=7, comm=comm)
        return d.scatter(full if rank == 0 else None, root=0)

    _, res = run_parallel(3, fn)
    assert np.allclose(np.concatenate(res, axis=0), full)


def test_scatter_with_empty_rank():
    # ntasks=1, size=3 -> rank 1 receives nothing (my_size 0).
    full = np.arange(1 * 2).reshape(1, 2) + 2.0

    def fn(comm, rank):
        d = MpiDistributor(ntasks=1, comm=comm)
        out = d.scatter(full if rank == 0 else None, root=0)
        return out.shape[0]

    _, res = run_parallel(3, fn)
    assert res == [0, 0, 1]


def test_scatter_type_error():
    d = MpiDistributor(ntasks=3, comm=comm1())
    with pytest.raises(TypeError):
        d.scatter([1, 2, 3], root=0)  # not a numpy array


def test_scatter_value_error_on_mismatch():
    d = MpiDistributor(ntasks=3, comm=comm1())
    bad = np.zeros((4, 2))  # length 4 != ntasks(3) and != my_size(3)
    with pytest.raises(ValueError):
        d.scatter(bad, root=0)


def test_scatter_none_on_root_single_rank():
    d = MpiDistributor(ntasks=3, comm=comm1())
    out = d.scatter(None, root=0)
    assert out.shape[0] == 3


# --------------------------------------------------------------------------- #
# send_to_rank / recv_from_rank
# --------------------------------------------------------------------------- #
def test_send_recv_object_roundtrip():
    arr = np.arange(5 * 4).reshape(5, 4) + 0.5

    def fn(comm, rank):
        d = MpiDistributor(ntasks=5, comm=comm)
        if rank == 0:
            h = Holder(arr.copy(), label="hello")
            d.send_to_rank(h, dest=1)
            return ("sent", h.mat)  # mat must be restored after send
        if rank == 1:
            h = d.recv_from_rank(source=0)
            return ("recv", h.label, h.mat)
        return None

    _, res = run_parallel(2, fn)
    assert res[0][0] == "sent"
    assert np.allclose(res[0][1], arr)  # restored on sender
    assert res[1][1] == "hello"
    assert np.allclose(res[1][2], arr)


def test_send_recv_object_chunked(monkeypatch):
    arr = np.arange(6 * 2).reshape(6, 2) + 1.0
    monkeypatch.setattr(md, "MAX_MPI_BYTES", 8)  # forces chunking of meta blob and mat

    def fn(comm, rank):
        d = MpiDistributor(ntasks=6, comm=comm)
        if rank == 0:
            d.send_to_rank(Holder(arr.copy(), "chunky"), dest=1)
            return "sent"
        if rank == 1:
            h = d.recv_from_rank(source=0)
            return (h.label, h.mat)
        return None

    _, res = run_parallel(2, fn)
    assert res[1][0] == "chunky"
    assert np.allclose(res[1][1], arr)


# --------------------------------------------------------------------------- #
# bcast / bcast_chunked / allreduce
# --------------------------------------------------------------------------- #
def test_bcast_object():
    def fn(comm, rank):
        d = MpiDistributor(ntasks=3, comm=comm)
        payload = {"v": 42} if rank == 0 else None
        return d.bcast(payload, root=0)

    _, res = run_parallel(3, fn)
    assert all(r == {"v": 42} for r in res)


def test_bcast_chunked():
    arr = (np.arange(10 * 2).reshape(10, 2) + 1j).astype(np.complex128)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=4, comm=comm)
        local = arr.copy() if rank == 0 else np.empty((1, 1), dtype=np.complex128)
        return d.bcast_chunked(local, root=0)

    _, res = run_parallel(3, fn)
    for r in res:
        assert np.allclose(r, arr)


def test_bcast_chunked_multi_chunk(monkeypatch):
    arr = (np.arange(8 * 3).reshape(8, 3) + 1j).astype(np.complex128)
    monkeypatch.setattr(md, "MAX_MPI_BYTES", 24)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=4, comm=comm)
        local = arr.copy() if rank == 0 else np.empty((1, 1), dtype=np.complex128)
        return d.bcast_chunked(local, root=0)

    _, res = run_parallel(3, fn)
    for r in res:
        assert np.allclose(r, arr)


def test_allreduce_sums_across_ranks():
    def fn(comm, rank):
        d = MpiDistributor(ntasks=3, comm=comm)
        return d.allreduce(np.array([float(rank + 1), 10.0]))

    _, res = run_parallel(3, fn)
    # ranks contribute (1,10),(2,10),(3,10) -> sum (6,30) on every rank.
    for r in res:
        assert np.allclose(r, [6.0, 30.0])


# --------------------------------------------------------------------------- #
# create_distributor factory
# --------------------------------------------------------------------------- #
def test_create_distributor_with_comm():
    d = MpiDistributor.create_distributor(ntasks=4, comm=comm1(), name="f")
    assert isinstance(d, MpiDistributor)
    assert d.ntasks == 4


def test_create_distributor_defaults_to_comm_world():
    d = MpiDistributor.create_distributor(ntasks=2, comm=None)
    assert d.comm is MPI.COMM_WORLD
    assert d.mpi_size == 1

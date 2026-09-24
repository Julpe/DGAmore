# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import dgamore.config as config
import dgamore.mpi_utils as mu
from dgamore.mpi_utils import MpiDistributor

from tests.conftest import run_parallel, FAKE_MPI as MPI, FAKE_SOCKET

# The shared conftest autouse-fixture no-ops os.remove (so the suite never deletes real files). Capture the genuine
# os.remove here at import time, so the rank-file lifecycle test can opt back into real deletion without disturbing it.
_REAL_OS_REMOVE = os.remove


@pytest.fixture(autouse=True)
def _use_fake_mpi(monkeypatch):
    # Inject the thread-based fake communicator into the module under test (the distributor, the chunking primitives
    # and the data-movement routines all live in mpi_utils now); real mpi4py is left untouched elsewhere.
    monkeypatch.setattr(mu, "MPI", MPI)
    monkeypatch.setattr(mu, "socket", FAKE_SOCKET)


@pytest.fixture(autouse=True)
def _default_no_output_path(monkeypatch):
    # The real config defaults output_path to "" (not None), which makes every MpiDistributor create a rank file on
    # construction. None -> no files; the file-lifecycle tests opt back in by passing a tmp_path explicitly.
    monkeypatch.setattr(config.output, "output_path", None)


def _holder(mat, label="obj"):
    """Builds a picklable object carrying a .mat array, matching what MpiDistributor.send_to_rank expects."""
    return SimpleNamespace(mat=mat, label=label)


def comm1():
    """A size-1 communicator usable directly on the main thread."""
    return MPI.Comm(1)


def test_send_recv_in_chunks_roundtrip():
    """_send_in_chunks / _recv_in_chunks round-trip a 2D complex array."""
    arr = (np.arange(5 * 3).reshape(5, 3) + 1j).astype(np.complex128)

    def fn(comm, rank):
        if rank == 0:
            mu._send_in_chunks(comm, arr, dest=1)
            return None
        if rank == 1:
            return mu._recv_in_chunks(comm, arr.shape, arr.dtype, source=0)
        return None

    _, res = run_parallel(2, fn)
    assert np.allclose(res[1], arr)


def test_send_recv_in_chunks_multichunk(monkeypatch):
    """_send_in_chunks / _recv_in_chunks round-trip across multiple chunks."""
    arr = (np.arange(6 * 2).reshape(6, 2) + 2j).astype(np.complex128)
    monkeypatch.setattr(mu, "MAX_MPI_BYTES", 16)

    def fn(comm, rank):
        if rank == 0:
            mu._send_in_chunks(comm, arr, dest=1)
            return None
        if rank == 1:
            return mu._recv_in_chunks(comm, arr.shape, arr.dtype, source=0)
        return None

    _, res = run_parallel(2, fn)
    assert np.allclose(res[1], arr)


def test_send_recv_in_chunks_1d():
    """_send_in_chunks / _recv_in_chunks round-trip a 1D array."""
    arr = np.arange(10, dtype=np.float64)

    def fn(comm, rank):
        if rank == 0:
            mu._send_in_chunks(comm, arr, dest=1)
            return None
        if rank == 1:
            return mu._recv_in_chunks(comm, arr.shape, arr.dtype, source=0)
        return None

    _, res = run_parallel(2, fn)
    assert np.allclose(res[1], arr)


def test_cgroup_memory_limit_walks_v2_ancestors_to_the_smallest_set_limit(tmp_path):
    """The v2 reader skips a "max" leaf and returns the smallest configured ancestor memory.max."""
    (tmp_path / "proc_cgroup").write_text("0::/a/b\n")
    (tmp_path / "cg" / "a" / "b").mkdir(parents=True)
    (tmp_path / "cg" / "a" / "b" / "memory.max").write_text("max\n")
    (tmp_path / "cg" / "a" / "memory.max").write_text("1234567\n")
    assert mu.cgroup_memory_limit(str(tmp_path / "proc_cgroup"), str(tmp_path / "cg")) == 1234567


def test_cgroup_memory_limit_reads_v1_controller_and_ignores_unlimited(tmp_path):
    """The v1 fallback reads memory.limit_in_bytes and treats huge sentinel values as unlimited."""
    (tmp_path / "proc_cgroup").write_text("9:memory:/slurm/job1\n")
    (tmp_path / "cg" / "memory" / "slurm" / "job1").mkdir(parents=True)
    (tmp_path / "cg" / "memory" / "slurm" / "job1" / "memory.limit_in_bytes").write_text("2222\n")
    assert mu.cgroup_memory_limit(str(tmp_path / "proc_cgroup"), str(tmp_path / "cg")) == 2222
    (tmp_path / "cg" / "memory" / "slurm" / "job1" / "memory.limit_in_bytes").write_text(str(2**63 - 4096))
    assert mu.cgroup_memory_limit(str(tmp_path / "proc_cgroup"), str(tmp_path / "cg")) is None


def test_cgroup_memory_limit_returns_none_without_cgroup_information(tmp_path):
    """A missing cgroup file (or one without limits) yields None, so the caller falls back to host memory."""
    assert mu.cgroup_memory_limit(str(tmp_path / "absent"), str(tmp_path / "cg")) is None


def test_job_memory_total_caps_the_hardware_total_by_the_cgroup_limit(monkeypatch):
    """job_memory_total returns the hardware total capped by the cgroup limit, or the plain total without one."""
    monkeypatch.setattr(mu.psutil, "virtual_memory", lambda: SimpleNamespace(total=1000))
    monkeypatch.setattr(mu, "cgroup_memory_limit", lambda: 200)
    assert mu.job_memory_total() == 200
    monkeypatch.setattr(mu, "cgroup_memory_limit", lambda: None)
    assert mu.job_memory_total() == 1000


def test_distribute_tasks_with_excess():
    """MpiDistributor places the task excess on the last rank."""
    d = MpiDistributor(ntasks=7, comm=MPI.Comm(3))
    # 7 tasks over 3 ranks -> excess of 1 lands on the last rank.
    assert list(d.sizes) == [2, 2, 3]
    assert [(s.start, s.stop) for s in d.slices] == [(0, 2), (2, 4), (4, 7)]


def test_distribute_tasks_even_no_excess():
    """MpiDistributor splits tasks evenly when there is no excess."""
    d = MpiDistributor(ntasks=6, comm=MPI.Comm(3))
    assert list(d.sizes) == [2, 2, 2]


def test_distributor_takes_explicit_task_counts_and_gathers_them_in_rank_order():
    """Explicit per-rank task counts replace the even split, and gather assembles the uneven slices in rank order."""
    rows = np.arange(7 * 3, dtype=float).reshape(7, 3)
    sizes = [0, 5, 2]
    bounds = np.concatenate([[0], np.cumsum(sizes)])

    def fn(comm, rank):
        d = MpiDistributor(ntasks=7, comm=comm, sizes=sizes)
        return list(d.sizes), d.gather(rows[bounds[rank] : bounds[rank + 1]].copy())

    _, res = run_parallel(3, fn)
    assert all(r[0] == sizes for r in res)
    assert np.array_equal(res[0][1], rows) and res[1][1] is None
    with pytest.raises(ValueError):
        MpiDistributor(ntasks=7, comm=MPI.Comm(3), sizes=[3, 3, 3])


@pytest.mark.parametrize("size", [1, 3, 4])
@pytest.mark.parametrize("limit", [None, 64])
def test_transpose_columns_delivers_all_rows_of_each_ranks_columns(size, limit, monkeypatch):
    """Every rank receives all rows of its columns, in its column order, also through the chunked message path."""
    if limit is not None:
        monkeypatch.setattr(mu, "MAX_MPI_BYTES", limit)
    rng = np.random.default_rng(size)
    full = rng.standard_normal((7, 2, 3, 4)) + 1j * rng.standard_normal((7, 2, 3, 4))
    pairs = [(a, b) for a in range(3) for b in range(4)]
    order = rng.permutation(len(pairs))
    split = np.array_split(order, size)
    cols_of = [
        (np.array([pairs[i][0] for i in part], dtype=int), np.array([pairs[i][1] for i in part], dtype=int))
        for part in split
    ]

    def fn(comm, rank):
        d = MpiDistributor(ntasks=7, comm=comm)
        return mu.transpose_columns(full[d.my_slice].copy(), d, cols_of, base_tag=5)

    _, res = run_parallel(size, fn)
    for rank, out in enumerate(res):
        assert np.array_equal(out, full[..., cols_of[rank][0], cols_of[rank][1]])


def test_exchange_blocks_matches_blocks_between_one_pair_in_posting_order():
    """Several blocks from one source arrive in the order both sides list them; one rank exchanges nothing."""
    blocks = [np.full((3, 2), float(i)) for i in range(4)]

    def fn(comm, rank):
        if rank == 0:
            return mu.exchange_blocks(comm, [], [1, 1, 2], (3, 2), float, base_tag=7)
        sends = [(0, blocks[0]), (0, blocks[1])] if rank == 1 else [(0, blocks[2])]
        return mu.exchange_blocks(comm, sends, [], (3, 2), float, base_tag=7)

    _, res = run_parallel(3, fn)
    assert [b[0, 0] for b in res[0]] == [0.0, 1.0, 2.0] and res[1] == [] and res[2] == []
    assert mu.exchange_blocks(MPI.Comm(1), [], [], (3, 2), float) == []


def test_distribute_tasks_fewer_than_ranks():
    """MpiDistributor gives empty slices to ranks beyond the task count."""
    d = MpiDistributor(ntasks=1, comm=MPI.Comm(3))
    assert list(d.sizes) == [0, 0, 1]


def test_properties_single_rank():
    """MpiDistributor exposes consistent properties on a single rank."""
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
    """MpiDistributor.is_root is True only on rank 0."""

    def fn(comm, rank):
        d = MpiDistributor(ntasks=4, comm=comm)
        return d.is_root, d.my_rank

    _, res = run_parallel(3, fn)
    assert res[0] == (True, 0)
    assert res[1] == (False, 1)
    assert res[2] == (False, 2)


def test_rankfile_created_and_context_manager(tmp_path, monkeypatch):
    """MpiDistributor creates a rank file and supports the context-manager and delete cycle."""
    d = MpiDistributor(ntasks=3, comm=comm1(), name="green", output_path=str(tmp_path))
    assert d._fname.endswith("green_Rank00000.hdf5")
    with d as f:
        assert f is not None
    # opt back in to real deletion; the shared conftest's autouse fixture otherwise no-ops os.remove
    monkeypatch.setattr(os, "remove", _REAL_OS_REMOVE)
    d.open_file()
    d.close_file()
    d.delete_file()
    assert not os.path.exists(d._fname)


def test_open_close_delete_are_safe_without_file():
    """MpiDistributor file ops are no-ops without an output path: open/close/delete swallow the missing errors."""
    d = MpiDistributor(ntasks=2, comm=comm1())
    assert d._file is None
    d.open_file()
    d.close_file()
    d.delete_file()


def test_output_path_is_injected_not_read_from_config(tmp_path, monkeypatch):
    """MpiDistributor uses the injected output_path, never config.output."""
    # config.output.output_path is poisoned; the injected output_path must win and
    # the distributor must never read the global config.
    monkeypatch.setattr(config.output, "output_path", "/should/not/be/used")
    d = MpiDistributor(ntasks=3, comm=comm1(), name="inj", output_path=str(tmp_path))
    assert str(tmp_path) in d._fname
    assert "/should/not/be/used" not in d._fname


def test_config_output_path_is_ignored_when_not_injected(monkeypatch):
    """MpiDistributor creates no spill file unless output_path is passed explicitly."""
    # Even with config.output.output_path set, no spill file is created unless
    # output_path is passed explicitly -> proves the global is no longer consulted.
    monkeypatch.setattr(config.output, "output_path", "/should/not/be/used")
    d = MpiDistributor(ntasks=3, comm=comm1(), name="ignored")
    assert d._file is None
    assert not hasattr(d, "_fname")


def test_create_distributor_threads_output_path(tmp_path, monkeypatch):
    """create_distributor forwards output_path to the rank file name."""
    monkeypatch.setattr(config.output, "output_path", None)
    d = MpiDistributor.create_distributor(ntasks=3, comm=comm1(), name="fac", output_path=str(tmp_path))
    assert d._fname.endswith("fac_Rank00000.hdf5")


def test_del_closes_file(tmp_path, monkeypatch):
    """MpiDistributor.__del__ closes an open rank file."""
    d = MpiDistributor(ntasks=1, comm=comm1(), name="del", output_path=str(tmp_path))
    fname = d._fname
    d.open_file()
    d.__del__()  # exercise destructor path directly
    assert fname.endswith("del_Rank00000.hdf5")


def test_exit_without_open_file():
    """MpiDistributor.__exit__ does nothing when no file is open."""
    d = MpiDistributor(ntasks=1, comm=comm1())
    # _file is None -> __exit__ must not attempt to close
    d.__exit__(None, None, None)


def test_barrier_runs_on_all_ranks():
    """MpiDistributor.barrier runs on every rank."""

    def fn(comm, rank):
        d = MpiDistributor(ntasks=3, comm=comm)
        d.barrier()
        return rank

    _, res = run_parallel(3, fn)
    assert sorted(res) == [0, 1, 2]


def test_allgather_reassembles_full_array():
    """MpiDistributor.allgather reassembles the full array on every rank."""
    full = (np.arange(7 * 2).reshape(7, 2) + 0.25).astype(np.float64)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=7, comm=comm)
        return d.allgather(full[d.my_slice])

    _, res = run_parallel(3, fn)
    for r in res:
        assert np.allclose(r, full)


def test_allgather_uses_allgatherv_collective(monkeypatch):
    """MpiDistributor.allgather uses a single Allgatherv (not per-rank broadcasts) when the data fits the limit."""
    full = (np.arange(6 * 2).reshape(6, 2) + 0.5).astype(np.float64)
    calls = {"agv": 0, "bcast": 0}
    orig_agv, orig_bcast = MPI.Comm.Allgatherv, MPI.Comm.Bcast

    def agv(self, *a, **k):
        calls["agv"] += 1
        return orig_agv(self, *a, **k)

    def bcast(self, *a, **k):
        calls["bcast"] += 1
        return orig_bcast(self, *a, **k)

    monkeypatch.setattr(MPI.Comm, "Allgatherv", agv)
    monkeypatch.setattr(MPI.Comm, "Bcast", bcast)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=6, comm=comm)
        return d.allgather(full[d.my_slice])

    _, res = run_parallel(3, fn)
    for r in res:
        assert np.allclose(r, full)
    assert calls["agv"] >= 1
    assert calls["bcast"] == 0


def test_allgather_1d_payload():
    """MpiDistributor.allgather reassembles a 1-D payload (items_per_row == 1)."""
    full = np.arange(7, dtype=np.float64)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=7, comm=comm)
        return d.allgather(full[d.my_slice])

    _, res = run_parallel(3, fn)
    for r in res:
        assert np.allclose(r, full)


def test_allgather_single_rank():
    """MpiDistributor.allgather returns the local data on a single rank."""
    d = MpiDistributor(ntasks=4, comm=comm1())
    local = np.arange(4, dtype=float)[:, None]
    out = d.allgather(local)
    assert np.allclose(out, local)


def test_allgather_single_rank_skips_collective(monkeypatch):
    """MpiDistributor.allgather short-circuits on a single rank (no collective, so it works on a minimal comm)."""
    called = {"agv": 0}
    orig = MPI.Comm.Allgatherv

    def agv(self, *a, **k):
        called["agv"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(MPI.Comm, "Allgatherv", agv)
    d = MpiDistributor(ntasks=4, comm=comm1())
    local = (np.arange(4 * 2).reshape(4, 2) + 1j).astype(np.complex128)
    out = d.allgather(local)
    assert np.allclose(out, local)
    assert called["agv"] == 0


def test_allgather_chunked(monkeypatch):
    """MpiDistributor.allgather reassembles the full array across multiple chunks."""
    full = (np.arange(6 * 3).reshape(6, 3) + 1j).astype(np.complex128)
    # Force several chunks per rank: one complex row = 48 bytes.
    monkeypatch.setattr(mu, "MAX_MPI_BYTES", 16)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=6, comm=comm)
        return d.allgather(full[d.my_slice])

    _, res = run_parallel(3, fn)
    for r in res:
        assert np.allclose(r, full)


def test_gather_to_root():
    """MpiDistributor.gather collects the full array on the root rank only."""
    full = (np.arange(7 * 3).reshape(7, 3) + 1j).astype(np.complex128)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=7, comm=comm)
        return d.gather(full[d.my_slice], root=0)

    _, res = run_parallel(3, fn)
    assert np.allclose(res[0], full)
    assert res[1] is None and res[2] is None


def test_gather_with_empty_ranks():
    """MpiDistributor.gather works when some ranks hold no tasks."""
    # ntasks=1, size=3 -> sizes [0, 0, 1]: a non-root rank has zero tasks.
    full = np.arange(1 * 2).reshape(1, 2) + 3.0

    def fn(comm, rank):
        d = MpiDistributor(ntasks=1, comm=comm)
        return d.gather(full[d.my_slice], root=0)

    _, res = run_parallel(3, fn)
    assert np.allclose(res[0], full)


def test_gather_chunked(monkeypatch):
    """MpiDistributor.gather collects the full array across multiple chunks."""
    full = (np.arange(7 * 3).reshape(7, 3) + 1j).astype(np.complex128)
    monkeypatch.setattr(mu, "MAX_MPI_BYTES", 16)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=7, comm=comm)
        return d.gather(full[d.my_slice], root=0)

    _, res = run_parallel(3, fn)
    assert np.allclose(res[0], full)


def test_gather_chunked_multidim_trailing(monkeypatch):
    """MpiDistributor.gather writes chunks straight into multi-dimensional axis-0 slices with no staging buffer."""
    full = (np.arange(5 * 2 * 3).reshape(5, 2, 3) + 1j).astype(np.complex128)
    monkeypatch.setattr(mu, "MAX_MPI_BYTES", 16)  # 1 row per chunk -> exercises the per-chunk Recv path

    def fn(comm, rank):
        d = MpiDistributor(ntasks=5, comm=comm)
        return d.gather(full[d.my_slice], root=0)

    _, res = run_parallel(3, fn)
    assert np.array_equal(res[0], full)


def test_gather_single_rank():
    """MpiDistributor.gather returns the local data on a single rank."""
    d = MpiDistributor(ntasks=3, comm=comm1())
    arr = np.arange(3 * 2).reshape(3, 2).astype(float)
    out = d.gather(arr, root=0)
    assert np.allclose(out, arr)


def test_scatter_distributes_rows():
    """MpiDistributor.scatter distributes the root's rows across ranks."""
    full = (np.arange(7 * 3).reshape(7, 3) + 1j).astype(np.complex128)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=7, comm=comm)
        return d.scatter(full if rank == 0 else None, root=0)

    _, res = run_parallel(3, fn)
    assert np.allclose(np.concatenate(res, axis=0), full)


def test_scatter_chunked(monkeypatch):
    """MpiDistributor.scatter distributes rows across multiple chunks."""
    full = (np.arange(7 * 3).reshape(7, 3) + 1j).astype(np.complex128)
    monkeypatch.setattr(mu, "MAX_MPI_BYTES", 16)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=7, comm=comm)
        return d.scatter(full if rank == 0 else None, root=0)

    _, res = run_parallel(3, fn)
    assert np.allclose(np.concatenate(res, axis=0), full)


def test_scatter_with_empty_rank():
    """MpiDistributor.scatter gives empty slices to ranks with no tasks."""
    # ntasks=1, size=3 -> rank 1 receives nothing (my_size 0).
    full = np.arange(1 * 2).reshape(1, 2) + 2.0

    def fn(comm, rank):
        d = MpiDistributor(ntasks=1, comm=comm)
        out = d.scatter(full if rank == 0 else None, root=0)
        return out.shape[0]

    _, res = run_parallel(3, fn)
    assert res == [0, 0, 1]


def test_scatter_type_error():
    """MpiDistributor.scatter raises TypeError for a non-array payload."""
    d = MpiDistributor(ntasks=3, comm=comm1())
    with pytest.raises(TypeError):
        d.scatter([1, 2, 3], root=0)  # not a numpy array


def test_scatter_value_error_on_mismatch():
    """MpiDistributor.scatter raises ValueError when the row count mismatches."""
    d = MpiDistributor(ntasks=3, comm=comm1())
    bad = np.zeros((4, 2))  # length 4 != ntasks(3) and != my_size(3)
    with pytest.raises(ValueError):
        d.scatter(bad, root=0)


def test_scatter_none_on_root_single_rank():
    """MpiDistributor.scatter on a single rank returns the full local size even for a None payload."""
    d = MpiDistributor(ntasks=3, comm=comm1())
    out = d.scatter(None, root=0)
    assert out.shape[0] == 3


def test_send_recv_object_roundtrip():
    """MpiDistributor.send_to_rank / recv_from_rank round-trip a .mat-carrying object."""
    arr = np.arange(5 * 4).reshape(5, 4) + 0.5

    def fn(comm, rank):
        d = MpiDistributor(ntasks=5, comm=comm)
        if rank == 0:
            h = _holder(arr.copy(), label="hello")
            d.send_to_rank(h, dest=1)
            return ("sent", h.mat)  # mat must be restored on the sender after the send
        if rank == 1:
            h = d.recv_from_rank(source=0)
            return ("recv", h.label, h.mat)
        return None

    _, res = run_parallel(2, fn)
    assert res[0][0] == "sent"
    assert np.allclose(res[0][1], arr)
    assert res[1][1] == "hello"
    assert np.allclose(res[1][2], arr)


def test_send_recv_object_chunked(monkeypatch):
    """MpiDistributor.send_to_rank / recv_from_rank round-trip across multiple chunks."""
    arr = np.arange(6 * 2).reshape(6, 2) + 1.0
    monkeypatch.setattr(mu, "MAX_MPI_BYTES", 8)  # forces chunking of meta blob and mat

    def fn(comm, rank):
        d = MpiDistributor(ntasks=6, comm=comm)
        if rank == 0:
            d.send_to_rank(_holder(arr.copy(), "chunky"), dest=1)
            return "sent"
        if rank == 1:
            h = d.recv_from_rank(source=0)
            return (h.label, h.mat)
        return None

    _, res = run_parallel(2, fn)
    assert res[1][0] == "chunky"
    assert np.allclose(res[1][1], arr)


def test_bcast_object():
    """MpiDistributor.bcast broadcasts a python object to all ranks."""

    def fn(comm, rank):
        d = MpiDistributor(ntasks=3, comm=comm)
        payload = {"v": 42} if rank == 0 else None
        return d.bcast(payload, root=0)

    _, res = run_parallel(3, fn)
    assert all(r == {"v": 42} for r in res)


def test_bcast_chunked():
    """MpiDistributor.bcast_chunked broadcasts an array to all ranks."""
    arr = (np.arange(10 * 2).reshape(10, 2) + 1j).astype(np.complex128)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=4, comm=comm)
        local = arr.copy() if rank == 0 else np.empty((1, 1), dtype=np.complex128)
        return d.bcast_chunked(local, root=0)

    _, res = run_parallel(3, fn)
    for r in res:
        assert np.allclose(r, arr)


def test_bcast_chunked_multi_chunk(monkeypatch):
    """MpiDistributor.bcast_chunked broadcasts an array across multiple chunks."""
    arr = (np.arange(8 * 3).reshape(8, 3) + 1j).astype(np.complex128)
    monkeypatch.setattr(mu, "MAX_MPI_BYTES", 24)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=4, comm=comm)
        local = arr.copy() if rank == 0 else np.empty((1, 1), dtype=np.complex128)
        return d.bcast_chunked(local, root=0)

    _, res = run_parallel(3, fn)
    for r in res:
        assert np.allclose(r, arr)


def test_bcast_npoint_roundtrips_object_and_mat():
    """bcast_npoint broadcasts an n-point-like object's metadata and chunk-broadcasts its .mat to every rank."""
    arr = (np.arange(5 * 4).reshape(5, 4) + 0.5 + 1j).astype(np.complex128)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=5, comm=comm)
        obj = _holder(arr.copy(), label="hello") if rank == 0 else _holder(np.empty((1, 1), np.complex128), label="x")
        out = d.bcast_npoint(obj, root=0)
        return out.label, out.mat

    _, res = run_parallel(3, fn)
    for label, mat in res:
        assert label == "hello"
        assert np.allclose(mat, arr)


def test_bcast_npoint_chunked(monkeypatch):
    """bcast_npoint chunk-broadcasts the .mat under the 2 GB limit (multiple chunks)."""
    arr = (np.arange(6 * 2).reshape(6, 2) + 2j).astype(np.complex128)
    monkeypatch.setattr(mu, "MAX_MPI_BYTES", 16)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=6, comm=comm)
        obj = _holder(arr.copy(), label="chunky") if rank == 0 else _holder(None, label="x")
        out = d.bcast_npoint(obj, root=0)
        return out.label, out.mat

    _, res = run_parallel(3, fn)
    for label, mat in res:
        assert label == "chunky"
        assert np.array_equal(mat, arr)


def test_bcast_npoint_single_rank_returns_object():
    """bcast_npoint short-circuits on a single rank (no collective, works on a minimal comm)."""
    d = MpiDistributor(ntasks=3, comm=comm1())
    obj = _holder((np.arange(6).reshape(3, 2) + 1j).astype(np.complex128), label="solo")
    out = d.bcast_npoint(obj, root=0)
    assert out is obj
    assert out.label == "solo"


def test_allreduce_sums_across_ranks():
    """MpiDistributor.allreduce sums contributions across ranks."""

    def fn(comm, rank):
        d = MpiDistributor(ntasks=3, comm=comm)
        return d.allreduce(np.array([float(rank + 1), 10.0]))

    _, res = run_parallel(3, fn)
    # ranks contribute (1,10),(2,10),(3,10) -> sum (6,30) on every rank.
    for r in res:
        assert np.allclose(r, [6.0, 30.0])


def test_allreduce_respects_message_limit(monkeypatch):
    """MpiDistributor.allreduce chunks the reduction so no message exceeds 2 GB (equal shapes chunk identically)."""
    base = (np.arange(6 * 2).reshape(6, 2) + 1j).astype(np.complex128)  # 32 bytes per row
    recorded = []
    orig = MPI.Comm.Allreduce

    def rec(self, sendbuf, recvbuf=None):
        buf = recvbuf if sendbuf is MPI.IN_PLACE else sendbuf
        recorded.append(np.asarray(buf).nbytes)
        return orig(self, sendbuf, recvbuf)

    monkeypatch.setattr(MPI.Comm, "Allreduce", rec)
    monkeypatch.setattr(mu, "MAX_MPI_BYTES", 64)

    def fn(comm, rank):
        d = MpiDistributor(ntasks=3, comm=comm)
        return d.allreduce((base * (rank + 1)).copy())

    _, res = run_parallel(3, fn)
    expected = base * (1 + 2 + 3)
    for r in res:
        assert np.allclose(r, expected)
    assert recorded and max(recorded) <= 64


def test_allreduce_empty_leading_axis():
    """MpiDistributor.allreduce is a no-op on an array with a zero-length leading axis."""

    def fn(comm, rank):
        d = MpiDistributor(ntasks=3, comm=comm)
        return d.allreduce(np.empty((0, 2), dtype=np.complex128))

    _, res = run_parallel(3, fn)
    for r in res:
        assert r.shape == (0, 2)


def test_create_distributor_with_comm():
    """create_distributor builds an MpiDistributor from a given communicator."""
    d = MpiDistributor.create_distributor(ntasks=4, comm=comm1(), name="f")
    assert isinstance(d, MpiDistributor)
    assert d.ntasks == 4


def test_create_distributor_defaults_to_comm_world():
    """create_distributor defaults to COMM_WORLD when no comm is given."""
    d = MpiDistributor.create_distributor(ntasks=2, comm=None)
    assert d.comm is MPI.COMM_WORLD
    assert d.mpi_size == 1


def test_close_file_propagates_non_os_errors():
    """close_file swallows OSError but propagates other (programming) errors."""
    dist = MpiDistributor.__new__(MpiDistributor)

    dist._file = MagicMock(close=MagicMock(side_effect=ValueError("not an OSError")))
    try:
        with pytest.raises(ValueError):
            dist.close_file()
    finally:
        dist._file = None  # avoid the mocked close() raising again during __del__ on GC


def test_chunk_step_floors_to_one_and_divides():
    """chunk_step floors to one row and divides by the per-row item count safely."""
    assert mu.chunk_step(16, 3, limit=16) == 1  # one row (48 B) exceeds the limit -> floor to 1
    assert mu.chunk_step(1, 1, limit=10) == 10
    assert mu.chunk_step(2, 1, limit=10) == 5
    assert mu.chunk_step(4, 0, limit=100) >= 1


@pytest.mark.parametrize(
    "n, limit, expected",
    [
        (5, 2, [(0, 2), (2, 4), (4, 5)]),  # remainder
        (4, 2, [(0, 2), (2, 4)]),  # exact multiple
        (0, 2, []),  # empty
        (3, 10_000, [(0, 3)]),  # single chunk when limit is huge
    ],
)
def test_row_chunks_boundaries(n, limit, expected):
    """row_chunks yields the correct chunk boundaries including remainder and empty cases."""
    assert list(mu.row_chunks(n, 1, 1, limit=limit)) == expected


@pytest.mark.parametrize("shape", [(7, 3), (5, 2, 3), (6,)])
def test_send_recv_rows_roundtrip_multichunk(shape):
    """send_rows / recv_rows_alloc round-trip arrays of various shapes across chunks."""
    arr = (np.arange(int(np.prod(shape))).reshape(shape) + 1j).astype(np.complex128)

    def fn(comm, rank):
        if rank == 0:
            mu.send_rows(comm, arr, dest=1, limit=16)  # 16 B < one row -> 1 row/chunk
            return None
        return mu.recv_rows_alloc(comm, arr.shape, arr.dtype, source=0, limit=16)

    _, res = run_parallel(2, fn)
    assert np.array_equal(res[1], arr)


def test_recv_rows_into_writes_into_given_buffer():
    """recv_rows_into fills the caller's preallocated buffer in place."""
    # Regression: recv_rows_into must fill the caller's preallocated buffer in place (no staging copy).
    arr = (np.arange(5 * 2 * 3).reshape(5, 2, 3) + 1j).astype(np.complex128)

    def fn(comm, rank):
        if rank == 0:
            mu.send_rows(comm, arr, dest=1, limit=16)
            return None
        buf = np.zeros_like(arr)
        ret = mu.recv_rows_into(comm, buf, source=0, limit=16)
        return ret is buf, buf

    _, res = run_parallel(2, fn)
    same_object, buf = res[1]
    assert same_object
    assert np.array_equal(buf, arr)


def test_send_recv_rows_default_limit_single_chunk():
    """send_rows / recv_rows_alloc round-trip in a single chunk at the default limit."""
    arr = np.arange(5 * 2).reshape(5, 2).astype(float)

    def fn(comm, rank):
        if rank == 0:
            mu.send_rows(comm, arr, dest=1)  # default (2 GB) limit -> single chunk
            return None
        return mu.recv_rows_alloc(comm, arr.shape, arr.dtype, source=0)

    _, res = run_parallel(2, fn)
    assert np.allclose(res[1], arr)


def test_bcast_rows_roundtrip_multichunk():
    """bcast_rows broadcasts an array across multiple chunks."""
    arr = (np.arange(6 * 2).reshape(6, 2) + 1j).astype(np.complex128)

    def fn(comm, rank):
        return mu.bcast_rows(comm, arr if rank == 0 else None, root=0, limit=16)

    _, res = run_parallel(3, fn)
    for r in range(3):
        assert np.array_equal(res[r], arr)


def test_bcast_rows_into_fills_view():
    """bcast_rows_into fills the receiver's buffer view."""
    arr = (np.arange(6 * 2).reshape(6, 2) + 1j).astype(np.complex128)

    def fn(comm, rank):
        buf = arr.copy() if rank == 0 else np.empty_like(arr)
        mu.bcast_rows_into(comm, buf, root=0, limit=16)
        return buf

    _, res = run_parallel(3, fn)
    for r in range(3):
        assert np.array_equal(res[r], arr)


def test_send_recv_bytes_roundtrip_multichunk():
    """send_bytes / recv_bytes round-trip a raw byte blob across chunks."""
    data = bytes(range(256)) * 5  # 1280 bytes

    def fn(comm, rank):
        if rank == 0:
            mu.send_bytes(comm, data, dest=1, limit=64)
            return None
        return mu.recv_bytes(comm, source=0, limit=64)

    _, res = run_parallel(2, fn)
    assert res[1] == data


def test_isend_irecv_rows_roundtrip_multichunk():
    """_isend_rows / _irecv_rows_into round-trip an array across multiple chunks via non-blocking requests."""
    arr = (np.arange(6 * 2).reshape(6, 2) + 1j).astype(np.complex128)

    def fn(comm, rank):
        if rank == 0:
            MPI.Request.Waitall(mu._isend_rows(comm, arr, dest=1, limit=16))
            return None
        buf = np.empty_like(arr)
        MPI.Request.Waitall(mu._irecv_rows_into(comm, buf, source=0, limit=16))
        return buf

    _, res = run_parallel(2, fn)
    assert np.array_equal(res[1], arr)


def test_isend_rows_rejects_non_contiguous():
    """_isend_rows refuses a non-C-contiguous array (it must not silently stage a throwaway copy)."""
    arr = (np.arange(6 * 2).reshape(6, 2) + 1j).astype(np.complex128).T  # F-contiguous view
    with pytest.raises(ValueError):
        mu._isend_rows(comm1(), arr, dest=0)


def test_allocate_node_shared_array_single_rank_returns_private_array():
    """A single-rank (or absent) communicator gets a private array of the requested shape and dtype and no window."""
    arr, win = mu.allocate_node_shared_array(None, (2, 3), np.complex128)
    assert arr.shape == (2, 3) and arr.dtype == np.complex128 and win is None

    def fn(comm, rank):
        arr, win = mu.allocate_node_shared_array(comm.Split_type(MPI.COMM_TYPE_SHARED), (4,), np.complex64)
        return arr.shape, arr.dtype, win

    _, res = run_parallel(1, fn)
    assert res[0] == ((4,), np.dtype(np.complex64), None)


def test_allocate_node_shared_array_exposes_one_buffer_to_every_node_rank():
    """What the node root writes into the allocated window is what every other rank of the node reads back."""

    def fn(comm, rank):
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        arr, win = mu.allocate_node_shared_array(node_comm, (2, 3), np.complex64)
        if node_comm.Get_rank() == 0:
            arr[...] = np.arange(6, dtype=np.complex64).reshape(2, 3) + 1j
        node_comm.Barrier()
        seen = np.array(arr)
        node_comm.Barrier()
        if win is not None:
            win.Free()
        node_comm.Free()
        return seen, win is not None

    _, res = run_parallel(3, fn, hostnames=["h", "h", "h"])
    expected = np.arange(6, dtype=np.complex64).reshape(2, 3) + 1j
    assert all(np.array_equal(seen, expected) for seen, _ in res)
    assert all(shared for _, shared in res)


def test_build_node_shared_array_single_rank_returns_private_array():
    """A single-rank node short-circuits to a private array (no window) and still computes once."""
    calls = []

    def fn(comm, rank):
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

        def compute():
            calls.append(rank)
            return np.full((2, 3), 5.0, dtype=np.complex64)

        arr, win = mu.build_node_shared_array(node_comm, compute)
        return np.array(arr), win

    _, res = run_parallel(1, fn)
    assert calls == [0]
    assert np.array_equal(res[0][0], np.full((2, 3), 5.0, dtype=np.complex64))
    assert res[0][1] is None


def test_build_node_shared_array_computes_once_and_shares_within_node():
    """On one node the root computes once and every rank maps the same populated buffer through a window."""
    calls = []

    def fn(comm, rank):
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

        def compute():
            calls.append(rank)
            return np.full((2, 3), 7.0, dtype=np.complex64)

        arr, win = mu.build_node_shared_array(node_comm, compute)
        seen = np.array(arr)
        node_comm.Barrier()
        if win is not None:
            win.Free()
        node_comm.Free()
        return seen, win is not None

    _, res = run_parallel(4, fn, hostnames=["h", "h", "h", "h"])
    assert calls == [0]
    for seen, has_win in res:
        assert has_win
        assert np.array_equal(seen, np.full((2, 3), 7.0, dtype=np.complex64))


def test_build_node_shared_array_takes_the_dtype_of_the_root_array():
    """Without an explicit dtype the window is allocated in the dtype the root computed, on every node rank."""

    def fn(comm, rank):
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        arr, win = mu.build_node_shared_array(node_comm, lambda: np.full((2, 2), 1 + 1j, dtype=np.complex128))
        seen = np.array(arr)
        node_comm.Barrier()
        win.Free()
        node_comm.Free()
        return seen

    _, res = run_parallel(3, fn, hostnames=["h", "h", "h"])
    for seen in res:
        assert seen.dtype == np.complex128 and np.array_equal(seen, np.full((2, 2), 1 + 1j))


def test_build_node_shared_array_isolates_between_nodes():
    """Two nodes each compute on their own root; ranks see only their own node's buffer."""
    calls = []

    def fn(comm, rank):
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

        def compute():
            calls.append(rank)
            return np.full((3,), float(rank + 1), dtype=np.complex64)

        arr, win = mu.build_node_shared_array(node_comm, compute)
        seen = np.array(arr)
        node_comm.Barrier()
        if win is not None:
            win.Free()
        node_comm.Free()
        return seen

    _, res = run_parallel(4, fn, hostnames=["n0", "n0", "n1", "n1"])
    assert sorted(calls) == [0, 2]  # one compute per node root (global ranks 0 and 2)
    assert np.array_equal(res[0], np.full((3,), 1.0, dtype=np.complex64))
    assert np.array_equal(res[1], np.full((3,), 1.0, dtype=np.complex64))
    assert np.array_equal(res[2], np.full((3,), 3.0, dtype=np.complex64))
    assert np.array_equal(res[3], np.full((3,), 3.0, dtype=np.complex64))


def test_build_node_shared_array_view_is_live_shared_memory():
    """A write by the root after construction is visible to the other node ranks (a true shared view)."""

    def fn(comm, rank):
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        arr, win = mu.build_node_shared_array(node_comm, lambda: np.zeros((4,), dtype=np.complex64))
        if node_comm.Get_rank() == 0:
            arr[:] = np.arange(4)  # mutate the shared buffer after it was built
        node_comm.Barrier()
        seen = np.array(arr)
        node_comm.Barrier()
        if win is not None:
            win.Free()
        node_comm.Free()
        return seen

    _, res = run_parallel(2, fn, hostnames=["h", "h"])
    for seen in res:
        assert np.array_equal(seen, np.arange(4).astype(np.complex64))


def test_count_nodes_single_node_short_circuits_without_communication():
    """All ranks on one node count a single node without entering the reduction."""

    def fn(comm, rank):
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        return mu.count_nodes(comm, node_comm)

    _, res = run_parallel(4, fn, hostnames=["h", "h", "h", "h"])
    assert res == [1, 1, 1, 1]


def test_count_nodes_counts_one_per_node_root():
    """Ranks spread over two nodes count two nodes on every rank."""

    def fn(comm, rank):
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        return mu.count_nodes(comm, node_comm)

    _, res = run_parallel(4, fn, hostnames=["n0", "n0", "n1", "n1"])
    assert res == [2, 2, 2, 2]


def test_count_nodes_counts_unevenly_populated_nodes():
    """An uneven rank-to-node mapping is counted by node roots, not by the node-local rank counts."""

    def fn(comm, rank):
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        return mu.count_nodes(comm, node_comm)

    _, res = run_parallel(4, fn, hostnames=["n0", "n1", "n1", "n2"])
    assert res == [3, 3, 3, 3]


def test_count_nodes_without_node_comm_is_one():
    """A missing node communicator means no topology was determined and counts as a single node."""
    assert mu.count_nodes(comm1(), None) == 1


def test_fake_comm_split_groups_by_color_ordered_by_key():
    """The fake Comm's Split mirrors MPI_Comm_split: one sub-comm per color, members shared, ranks ordered by key."""

    def fn(comm, rank):
        sub = comm.Split(0 if rank in (1, 3) else 1, rank)
        return sub.Get_size(), sub.Get_rank()

    _, res = run_parallel(4, fn)
    assert res == [(2, 0), (2, 0), (2, 1), (2, 1)]

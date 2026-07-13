# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import logging
import os
from unittest.mock import MagicMock

import mpi4py.MPI as MPI
import numpy as np
import pytest

import dgamore.brillouin_zone as bz
import dgamore.config as config


def pytest_addoption(parser):
    """Register the --runslow flag so individual tests can be marked @pytest.mark.slow
    and skipped by default. CI can opt in via `pytest --runslow`."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked as slow (large-grid auto-symmetry discovery, etc.).",
    )


def pytest_configure(config):
    """Register the 'slow' marker so pytest doesn't warn about an unknown marker."""
    config.addinivalue_line("markers", "slow: mark test as slow (deselect with '-m \"not slow\"')")


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.slow tests unless --runslow was given."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def mock_does_not_delete_files(monkeypatch):
    # Make os.remove do nothing
    monkeypatch.setattr(os, "remove", lambda path: None)


@pytest.fixture(autouse=True)
def mock_does_not_create_folders(monkeypatch):
    # Make os.remove do nothing
    monkeypatch.setattr(os, "makedirs", lambda path: None)


@pytest.fixture(autouse=True)
def mock_numpy_save(monkeypatch):
    # Automatically mock numpy.save for all tests.
    def fake_save(file, arr, **kwargs):
        pass

    monkeypatch.setattr(np, "save", fake_save)
    monkeypatch.setattr(np, "savetxt", fake_save)
    yield


@pytest.fixture(autouse=True)
def mock_logger(monkeypatch):
    # Automatically mock logger.log for all tests.
    logger_mock = MagicMock()
    monkeypatch.setattr(logging, "getLogger", lambda name=None: logger_mock)
    monkeypatch.setattr(logging, "Logger", MagicMock(return_value=logger_mock))
    yield logger_mock


@pytest.fixture(autouse=True)
def reset_config_singletons():
    # Restore the global config singletons to fresh defaults after every test: config.py is module-level mutable state
    # that e2e tests reconfigure, so without this reset mutations would leak into later tests. Runs on teardown.
    yield
    config.box = config.BoxConfig()
    config.lattice = config.LatticeConfig()
    config.lambda_correction = config.LambdaCorrectionConfig()
    config.dmft = config.DmftConfig()
    config.sys = config.SystemConfig()
    config.output = config.OutputConfig()
    config.self_energy_interpolation = config.SelfEnergyInterpolationConfig()
    config.self_consistency = config.SelfConsistencyConfig()
    config.stabilization = config.StabilizationConfig()
    config.eliashberg = config.EliashbergConfig()
    config.memory = config.MemoryConfig()
    config.ana_cont = config.AnaContConfig()


def create_default_config(config, folder: str):
    config.box.niw_core = -1
    config.box.niv_core = -1
    config.box.niv_shell = 10
    config.output.do_plotting = False
    config.lattice.nk = (4, 4, 1)
    config.lattice.k_grid = bz.KGrid(config.lattice.nk, symmetries=bz.two_dimensional_square_symmetries())
    config.lattice.type = "from_wannierHK"
    config.lattice.interaction_type = "kanamori_from_dmft"
    config.lattice.er_input = f"{folder}/wannier.hk"
    config.dmft.input_path = folder
    config.dmft.n_ineq = 1
    config.dmft.ineq_ordering = [1]
    config.dmft.n_bands_per_ineq = []
    config.eliashberg.perform_eliashberg = False
    config.self_consistency.mixing = 1
    config.self_consistency.max_iter = 1


def create_comm_mock():
    comm_mock = MagicMock()

    # Fundamental MPI properties
    comm_mock.Get_size.return_value = 1
    comm_mock.Get_rank.return_value = 0
    comm_mock.size = 1
    comm_mock.rank = 0
    comm_mock.IN_PLACE = MPI.IN_PLACE

    # Internal state to track pending non-blocking operations
    # {tag: buffer_pointer}
    pending_irecvs = {}
    pending_isends = {}

    def mock_Irecv(buf, source, tag=0):
        if tag in pending_isends:
            np.copyto(buf, pending_isends[tag])
            del pending_isends[tag]
        else:
            pending_irecvs[tag] = buf
        return MPI.REQUEST_NULL

    def mock_Isend(buf, dest, tag=0):
        if tag in pending_irecvs:
            np.copyto(pending_irecvs[tag], buf)
            del pending_irecvs[tag]
        else:
            pending_isends[tag] = np.copy(buf)
        return MPI.REQUEST_NULL

    comm_mock.Irecv.side_effect = mock_Irecv
    comm_mock.Isend.side_effect = mock_Isend

    # --- Lowercase (Object) Loopback ---
    # Used for bcast/scatter/send_to_rank
    comm_mock.bcast.side_effect = lambda obj, root=0: obj
    comm_mock.allgather.side_effect = lambda obj: [obj]

    # Point-to-point object exchange
    obj_bus = {}

    def mock_send(obj, dest, tag=0):
        obj_bus[tag] = obj

    def mock_recv(source=0, tag=0):
        return obj_bus.get(tag, None)

    comm_mock.send.side_effect = mock_send
    comm_mock.recv.side_effect = mock_recv

    # --- Collective Uppercase (NumPy) ---
    def bcast_numpy(buf, root=0):
        return None  # Data already in-place for 1-rank

    def allreduce_numpy(sendbuf, recvbuf, op=None):
        if sendbuf is not MPI.IN_PLACE:
            np.copyto(recvbuf, sendbuf)
        return None

    comm_mock.Bcast.side_effect = bcast_numpy
    comm_mock.Allreduce.side_effect = allreduce_numpy

    # Split / Split_type return the single-rank comm itself; Free is a no-op
    comm_mock.Split.return_value = comm_mock
    comm_mock.Split_type.return_value = comm_mock
    comm_mock.Free.return_value = None

    return comm_mock


import queue as _queue
import threading as _threading
import traceback as _traceback
import types as _types

_tls = _threading.local()
_BARRIER_TIMEOUT = _QUEUE_TIMEOUT = 20.0


class _InPlace:
    def __repr__(self):
        return "IN_PLACE"


IN_PLACE = _InPlace()
REQUEST_NULL = object()


def Get_processor_name():
    return getattr(_tls, "hostname", "node0")


class _Request:
    def __init__(self, wait_fn=None):
        self._wait_fn, self._done = wait_fn, False

    def wait(self):
        if not self._done and self._wait_fn is not None:
            self._wait_fn()
        self._done = True

    Wait = wait


class Request:
    @staticmethod
    def Waitall(reqs):
        for r in reqs:
            if r is not None:
                r.wait()


def _assign(buf, data):
    try:
        buf[...] = data
    except (ValueError, TypeError):
        buf.view(np.uint8).reshape(-1)[:] = np.ascontiguousarray(data).view(np.uint8).reshape(-1)


class Comm:
    def __init__(self, size=1, members=None):
        self._members = tuple(members) if members is not None else tuple(range(int(size)))
        self._size = len(self._members)
        self._barrier = _threading.Barrier(self._size) if self._size > 1 else None
        self._inboxes = [dict() for _ in range(self._size)]
        self._lock = _threading.Lock()
        self._store = [None] * self._size
        self._split_cache = {}

    def Get_rank(self):
        return self._members.index(getattr(_tls, "rank", 0))

    def Get_size(self):
        return self._size

    rank = property(Get_rank)
    size = property(Get_size)

    def _bw(self):
        if self._barrier is not None:
            self._barrier.wait(timeout=_BARRIER_TIMEOUT)

    def Barrier(self):
        self._bw()

    def barrier(self):
        self._bw()

    def abort_barrier(self):
        if self._barrier is not None:
            self._barrier.abort()

    def Split_type(self, split_type, key=0, info=None):
        # Group ranks by host (COMM_TYPE_SHARED -> one sub-communicator per node). Threads share memory, so same-node
        # ranks receive the SAME cached Comm (shared barrier/store), which makes the shared-memory window shared.
        host = getattr(_tls, "hostname", "node0")
        hosts = list(self._collective(host))
        group = tuple(self._members[i] for i in range(self._size) if hosts[i] == host)
        with self._lock:
            if host not in self._split_cache:
                self._split_cache[host] = Comm(members=group)
        return self._split_cache[host]

    def Free(self):
        pass

    def _collective(self, contribution):
        r = self.rank
        self._bw()
        if r == 0:
            self._store = [None] * self._size
        self._bw()
        self._store[r] = contribution
        self._bw()
        return self._store

    def bcast(self, obj, root=0):
        return self._collective(obj if self.rank == root else None)[root]

    def Bcast(self, buf, root=0):
        store = self._collective(np.array(buf, copy=True) if self.rank == root else None)
        if self.rank != root:
            _assign(buf, store[root])
        return buf

    def allreduce(self, obj, op=None):
        # Object-level allreduce over scalars: MPI.MIN/MAX map to min/max, everything else (incl. None) sums.
        values = list(self._collective(obj))
        if op is not None and op == MPI.MIN:
            return min(values)
        if op is not None and op == MPI.MAX:
            return max(values)
        return sum(values)

    def Allreduce(self, sendbuf, recvbuf=None):
        buf = recvbuf
        if sendbuf is not IN_PLACE:
            _assign(buf, sendbuf)
        total = None
        for part in self._collective(np.array(buf, copy=True)):
            total = part.copy() if total is None else total + part
        _assign(buf, total)
        return buf

    def Allgather(self, sendbuf, recvbuf):
        if sendbuf is IN_PLACE:
            store = self._collective(np.array(recvbuf, copy=True))
            for i in range(self._size):
                recvbuf[i] = store[i][i]
            return recvbuf
        store = self._collective(np.array(sendbuf, copy=True))
        for i in range(self._size):
            recvbuf[i] = store[i]
        return recvbuf

    def allgather(self, obj):
        return list(self._collective(obj))

    def Allgatherv(self, sendspec, recvspec):
        # Element-count semantics on flattened buffers: each rank contributes sendspec[1] elements, placed into the
        # flat recvbuf at the per-rank (counts, displs) given in recvspec[1]. Mirrors mpi4py's Allgatherv contract.
        sendbuf = sendspec[0] if isinstance(sendspec, (list, tuple)) else sendspec
        scount = (
            int(sendspec[1]) if isinstance(sendspec, (list, tuple)) and len(sendspec) > 1 else int(np.size(sendbuf))
        )
        recvbuf, (counts, displs) = recvspec[0], recvspec[1]
        flat_send = np.ascontiguousarray(sendbuf).reshape(-1)[:scount]
        store = self._collective(np.array(flat_send, copy=True))
        rflat = np.asarray(recvbuf).reshape(-1)
        for r in range(self._size):
            c, d = int(counts[r]), int(displs[r])
            if c:
                rflat[d : d + c] = store[r][:c]
        return recvbuf

    def Alltoall(self, sendbuf, recvbuf):
        store = self._collective(np.array(sendbuf, copy=True))
        for j in range(self._size):
            recvbuf[j] = store[j][self.rank]
        return recvbuf

    def _queue(self, owner, key):
        with self._lock:
            return self._inboxes[owner].setdefault(key, _queue.Queue())

    def Send(self, buf, dest, tag=0):
        self._queue(dest, (self.rank, tag)).put(np.array(np.ascontiguousarray(buf), copy=True))

    def Recv(self, buf, source, tag=0):
        _assign(buf, self._queue(self.rank, (source, tag)).get(timeout=_QUEUE_TIMEOUT))
        return buf

    def send(self, obj, dest, tag=0):
        self._queue(dest, ("py", self.rank, tag)).put(obj)

    def recv(self, source, tag=0):
        return self._queue(self.rank, ("py", source, tag)).get(timeout=_QUEUE_TIMEOUT)

    def Isend(self, buf, dest, tag=0):
        self.Send(buf, dest, tag)
        return _Request()

    def Irecv(self, buf, source, tag=0):
        return _Request(lambda: _assign(buf, self._queue(self.rank, (source, tag)).get(timeout=_QUEUE_TIMEOUT)))


class Win:
    # Minimal MPI shared-memory window over the threaded fake Comm. Threads share an address space, so the node root
    # allocates one bytearray and bcasts the SAME object to every node rank - a numpy view over it is genuinely shared.
    @staticmethod
    def Allocate_shared(size, disp_unit=1, info=None, comm=None):
        win = Win()
        buf = bytearray(int(size)) if comm.Get_rank() == 0 else None
        win._buf = comm.bcast(buf, root=0)
        return win

    def Shared_query(self, rank):
        return self._buf, 1

    def Free(self):
        self._buf = None


COMM_WORLD = Comm(1)

FAKE_MPI = _types.SimpleNamespace(
    Comm=Comm,
    Request=Request,
    Win=Win,
    IN_PLACE=IN_PLACE,
    REQUEST_NULL=REQUEST_NULL,
    COMM_TYPE_SHARED=1,
    COMM_WORLD=COMM_WORLD,
    Get_processor_name=Get_processor_name,
)


def run_parallel(size, fn, hostnames=None):
    comm = Comm(size)
    results, errors = [None] * size, [None] * size

    def worker(r):
        _tls.rank = r
        _tls.hostname = hostnames[r] if hostnames is not None else f"node{r}"
        try:
            results[r] = fn(comm, r)
        except BaseException as exc:
            errors[r] = (exc, _traceback.format_exc())
            comm.abort_barrier()

    threads = [_threading.Thread(target=worker, args=(r,), name=f"rank{r}") for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    for r, e in enumerate(errors):
        if e is not None:
            raise AssertionError(f"rank {r} raised:\n{e[1]}") from e[0]
    return comm, results

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import gc
import logging
import os
import tracemalloc
from unittest.mock import MagicMock

import mpi4py.MPI as MPI
import numpy as np
import pytest

import dgamore.brillouin_zone as bz
import dgamore.config as config
import dgamore.local_n_point as local_n_point


def pytest_addoption(parser):
    """Registers --runslow so @pytest.mark.slow tests are skipped unless --runslow is passed."""
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


def pytest_collection_finish(session):
    """Freezes the collected session so the gc.collect() of every IHaveMat.free() walks only what the tests create."""
    gc.collect()
    gc.freeze()


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
    monkeypatch.setattr(local_n_point, "_flush_and_drop", lambda path: None)  # nothing was written to flush
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
    reset_config()


def reset_config():
    """Restores every config singleton to fresh defaults."""
    config.box = config.BoxConfig()
    config.lattice = config.LatticeConfig()
    config.lambda_correction = config.LambdaCorrectionConfig()
    config.misc = config.MiscConfig()
    config.dmft = config.DmftConfig()
    config.sys = config.SystemConfig()
    config.output = config.OutputConfig()
    config.self_energy_interpolation = config.SelfEnergyInterpolationConfig()
    config.self_consistency = config.SelfConsistencyConfig()
    config.stabilization = config.StabilizationConfig()
    config.eliashberg = config.EliashbergConfig()
    config.ana_cont = config.AnaContConfig()


def create_default_config(config, folder: str):
    config.box.niw_core = -1
    config.box.niv_core = -1
    config.box.niv_shell = 10
    config.output.do_plotting = False
    config.lattice.nk = (4, 4, 1)
    config.lattice.k_grid = bz.KGrid(config.lattice.nk, symmetries=bz.two_dimensional_square_symmetries())
    config.lattice.type = "from_wannierHK"
    config.lattice.interaction_type = "from_dmft"
    config.lattice.er_input = f"{folder}/wannier.hk"
    config.dmft.input_path = folder
    config.dmft.n_ineq = 1
    config.dmft.ineq_ordering = [1]
    config.dmft.n_bands_per_ineq = []
    config.dmft.symmetrize_orbitals = [1, 2]
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
# a deadlock guard, not a timing check: one rank of the 7-rank Jacobian tests can trail the others by more than 20 s
# on a slow CI runner under coverage
_BARRIER_TIMEOUT = _QUEUE_TIMEOUT = 120.0


class _InPlace:
    def __repr__(self):
        return "IN_PLACE"


IN_PLACE = _InPlace()
REQUEST_NULL = object()


def Get_processor_name():
    return getattr(_tls, "hostname", "node0")


def gethostname():
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

    def Split(self, color, key=0):
        # MPI_Comm_split: one sub-communicator per color, member order by (key, member) as in the standard.
        pairs = list(self._collective((color, key)))
        group = tuple(m for (c, k), m in sorted(zip(pairs, self._members), key=lambda t: (t[0][1], t[1])) if c == color)
        # the group must be part of the key: successive Splits may reuse a color for different member sets
        cache_key = ("split", color, group)
        with self._lock:
            if cache_key not in self._split_cache:
                self._split_cache[cache_key] = Comm(members=group)
        return self._split_cache[cache_key]

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

FAKE_SOCKET = _types.SimpleNamespace(gethostname=gethostname)


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
    # every rank is waited for: the barrier and queue timeouts end a deadlock, while a join timeout would hand back a
    # slow rank's result as None
    for t in threads:
        t.join()
    for r, e in enumerate(errors):
        if e is not None:
            raise AssertionError(f"rank {r} raised:\n{e[1]}") from e[0]
    return comm, results


def ten_times_the_floor(reference: np.ndarray, dtype=None) -> float:
    """
    Ten times the floating-point floor of ``reference``: 1e-14 of its largest entry in double precision, scaled by
    the ratio of the machine epsilons (to 5.4e-6) in single precision; 0 for an empty array. Results whose
    computation changes the shapes or the concurrency of BLAS and LAPACK calls (windows, blocks, residency, fused
    solves, rank layouts) compare to this bar, not bit for bit: their bits depend on the BLAS library.

    :param reference: The result compared against.
    :param dtype: The precision the result was computed in, if not its own (an operator's real products of
        complex64 storage).
    :return: The absolute tolerance.
    """
    floor = 1e-14 * np.finfo(reference.dtype if dtype is None else dtype).eps / np.finfo(np.complex128).eps
    return 10 * floor * float(np.abs(reference).max(initial=0.0))


def patch_mini_pole_with_exact_single_pole_fit(monkeypatch, x: float):
    """Replaces SelfEnergy's MiniPole by an exact single-pole fitter at x; the real cal_G_vector is kept."""
    from dgamore import self_energy as self_energy_module

    def fit(G_w, w, **kwargs):
        return _types.SimpleNamespace(
            pole_location=np.array([x + 0.0j]), pole_weight=np.array([[[G_w[0] * (1j * w[0] - x)]]])
        )

    mini_pole = MagicMock(side_effect=fit)
    mini_pole.cal_G_vector = self_energy_module.MiniPole.cal_G_vector
    monkeypatch.setattr(self_energy_module, "MiniPole", mini_pole)


def traced_peak(fn):
    """Runs fn under tracemalloc and returns its result and the peak bytes traced above the level at its start."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        entry = tracemalloc.get_traced_memory()[0]
        result = fn()
        peak = tracemalloc.get_traced_memory()[1] - entry
    finally:
        tracemalloc.stop()
    return result, peak


def write_hk(path, ek: np.ndarray, kpoints: np.ndarray) -> None:
    """Writes ek [k, o, o] at the k-points [k, 3] as a wannier90 .hk file (the autouse fixture mutes np.savetxt)."""
    n = ek.shape[-1]
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"{ek.shape[0]} {n} {n}\n")
        for k, block in zip(kpoints, ek):
            file.write(" ".join(f"{x:.12f}" for x in k) + "\n")
            for row in block:
                file.write(" ".join(f"{z.real:.12f} {z.imag:.12f}" for z in row) + "\n")


def emery_hk(nk: tuple[int, int, int], e_p: float = -2.5, t_pd: float = 1.0, t_pp: float = 0.4) -> np.ndarray:
    """The three-orbital Emery dispersion (d, p_x, p_y) in the cell-periodic gauge, with the d level at zero."""
    kx, ky, _ = np.meshgrid(*(2 * np.pi * np.arange(n) / n for n in nk), indexing="ij")
    hk = np.zeros((*nk, 3, 3), dtype=complex)
    hk[..., 1, 1] = hk[..., 2, 2] = e_p
    hk[..., 0, 1] = t_pd * (1 - np.exp(-1j * kx))
    hk[..., 0, 2] = t_pd * (1 - np.exp(-1j * ky))
    hk[..., 1, 2] = t_pp * (1 - np.exp(1j * kx)) * (1 - np.exp(-1j * ky))
    return hk + np.conj(np.triu(hk, 1).swapaxes(-1, -2))


def p_orbital_hk(nk: tuple[int, int, int], equal_orbitals: bool = True) -> np.ndarray:
    """A p_x/p_y dispersion with an inter-orbital hopping, whose mirrors flip the sign of one orbital; with equal
    orbitals the diagonal mirrors also swap them."""
    kx, ky, _ = np.meshgrid(*(2 * np.pi * np.arange(n) / n for n in nk), indexing="ij")
    hk = np.zeros((*nk, 2, 2), dtype=complex)
    hk[..., 0, 0] = -2.0 * np.cos(kx) - 0.6 * np.cos(ky)
    hk[..., 1, 1] = -2.0 * np.cos(ky) - 0.6 * np.cos(kx) if equal_orbitals else -1.4 * np.cos(ky) - 0.5 * np.cos(kx)
    hk[..., 0, 1] = hk[..., 1, 0] = 0.3 * np.sin(kx) * np.sin(ky)
    return hk


def linear_pairs(
    mat: np.ndarray, shift: np.ndarray, p: float, n_iter: int, x0: list[float]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns the (iterate, proposal) lists of a damped linear iteration as complex core-window arrays."""
    x = np.asarray(x0, dtype=np.complex128)
    iterates, proposals = [], []
    for _ in range(n_iter):
        proposal = (mat @ x.real + shift).astype(np.complex128)
        iterates.append(x.copy())
        proposals.append(proposal.copy())
        x = x + p * (proposal - x)
    return iterates, proposals


def general_su2_u(n_bands: int, u: float, u_prime: float, j: float, j_prime: float) -> np.ndarray:
    """A general SU(2) local interaction in the equation layout: U on aaaa, the inter-orbital density U' on aabb, the
    Hund's exchange J on abba and the pair hopping J' on abab (Kanamori is U' = U - 2J, J' = J)."""
    u_mat = np.zeros((n_bands,) * 4)
    for a in range(n_bands):
        for b in range(n_bands):
            u_mat[a, a, b, b] = u if a == b else u_prime
            if a != b:
                u_mat[a, b, b, a], u_mat[a, b, a, b] = j, j_prime
    return u_mat


def graser_spin_charge_u(n_bands: int, u: float, u_prime: float, j: float, j_prime: float) -> tuple:
    """U^s and U^c of Graser et al., New J. Phys. 11, 025016 (2009), below eq. 20, in Kubo's notation (U~' = U',
    J~ = J, J~' = J'), stored as (U)^{l1 l2}_{l3 l4} -> [l1, l2, l3, l4]: U^s has U, U' (l1 = l3 != l2 = l4), J
    (l1 = l2 != l3 = l4) and J' (l1 = l4 != l2 = l3), U^c has U, 2J - U', 2U' - J and J' on the same patterns."""
    u_s, u_c = np.zeros((n_bands,) * 4), np.zeros((n_bands,) * 4)
    for a in range(n_bands):
        u_s[a, a, a, a] = u_c[a, a, a, a] = u
        for b in range(n_bands):
            if a != b:
                u_s[a, b, a, b], u_c[a, b, a, b] = u_prime, 2 * j - u_prime
                u_s[a, a, b, b], u_c[a, a, b, b] = j, 2 * u_prime - j
                u_s[a, b, b, a] = u_c[a, b, b, a] = j_prime
    return u_s, u_c


def chain_without_inversion_hk(nk: int) -> np.ndarray:
    """H(k) [kx, 1, 1, o, o] of a time-reversal symmetric two-orbital chain whose inter-orbital hopping differs along
    +x and -x: complex, H(-k) = H(k)^*, and no inversion (no orbital action maps H(k) onto H(-k))."""
    hop = np.array([[-1.0, 0.3], [0.1, -0.6]])
    phase = np.exp(2j * np.pi * np.arange(nk) / nk)[:, None, None]
    return (np.array([[0.2, 0.15], [0.15, -0.1]]) + hop * phase + hop.T * phase.conj())[:, None, None]


def bilayer_chain_hk(nk: int) -> np.ndarray:
    """H(k) [kx, 1, 1, o, o] of two layers of one orbital on a chain whose interlayer hopping reaches into the next
    cell: complex, H(-k) = H(k)^* != H(k), and inversion swaps the layers, H(-k) = S H(k) S."""
    phase = np.exp(2j * np.pi * np.arange(nk) / nk)
    hk = np.zeros((nk, 1, 1, 2, 2), dtype=complex)
    hk[..., 0, 0] = hk[..., 1, 1] = (-2.0 * phase.real)[:, None, None]
    hk[..., 0, 1] = (0.5 + 0.3 * phase)[:, None, None]
    hk[..., 1, 0] = np.conj(hk[..., 0, 1])
    return hk


def site_lattice_hk(nk: tuple[int, int, int], positions, onsite, hopping) -> np.ndarray:
    """H(k) [nx, ny, nz, o, o] in the cell-periodic gauge of one orbital per site at the fractional positions of a
    cubic cell: H_ab(k) = sum_R t_ab(|x_b + R - x_a|) e^{i k.R} over the neighboring cells along the axes with more
    than one k-point, t_ab(r) = hopping[a][b] exp(-2 r) for 0 < r < 1.01. Distance-only hoppings keep every symmetry
    of the site arrangement, so a site that inversion maps into another cell gives a momentum-dependent action."""
    positions, hopping = np.asarray(positions, dtype=float), np.asarray(hopping, dtype=float)
    k = np.stack(np.meshgrid(*(2 * np.pi * np.arange(n) / n for n in nk), indexing="ij"), axis=-1)
    hk = np.zeros((*nk, len(positions), len(positions)), dtype=complex)
    hk[..., np.arange(len(positions)), np.arange(len(positions))] = onsite
    for cell in np.stack(np.meshgrid(*([-1, 0, 1] if n > 1 else [0] for n in nk), indexing="ij"), -1).reshape(-1, 3):
        r = np.linalg.norm(positions[None, :] + cell - positions[:, None], axis=-1)
        hop = np.where((r > 1e-9) & (r < 1.01), hopping * np.exp(-2 * r), 0.0)
        hk += hop * np.exp(1j * k @ cell)[..., None, None]
    return hk


def chain_without_inversion_green(nk: int, beta: float, mu: float, niv: int) -> np.ndarray:
    """G(k, i nu) = (i nu + mu - H(k))^-1 [kx, 1, 1, o, o, v] on 2 niv frequencies of the chain of
    :func:`chain_without_inversion_hk`."""
    hk = chain_without_inversion_hk(nk)[:, 0, 0]
    nu = (2 * np.arange(-niv, niv) + 1) * np.pi / beta
    g = np.linalg.inv((1j * nu[:, None, None, None] + mu) * np.eye(2) - hk[None])  # [v, k, o, o]
    return g.transpose(1, 2, 3, 0)[:, None, None]


def graser_bubble(g: np.ndarray, q, w: int, niv: int) -> np.ndarray:
    """Graser's eq. 12, chi^{pq}_{st}(q, i w) = -T/N sum_k sum_nu G_sp(k, nu) G_qt(k + q, nu + w), as [p, q, s, t],
    with nu on [-niv, niv) (the box of DGAmore's summed bubble) and T = 1 / config.sys.beta; g holds G(k, i nu)
    [kx, ky, kz, o, o, v] on a centered box of at least niv + |w| positive frequencies."""
    ng = g.shape[-1] // 2
    g_k = g[..., ng - niv : ng + niv]
    g_kq = np.roll(g, shift=tuple(-np.asarray(q)), axis=(0, 1, 2))[..., ng - niv + w : ng + niv + w]
    return -np.einsum("xyzspv,xyzqtv->pqst", g_k, g_kq) / (config.sys.beta * np.prod(g.shape[:3]))

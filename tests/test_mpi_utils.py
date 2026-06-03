# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import numpy as np
import pytest

import dgamore.config as config
import dgamore.symmetry_reduction as symmetry_reduction
import dgamore.mpi_utils as mu
import dgamore.mpi_distributor as md
from dgamore.mpi_distributor import MpiDistributor
import dgamore.brillouin_zone as bz
from dgamore.brillouin_zone import KGrid
from dgamore.four_point import FourPoint
from dgamore.n_point_base import SpinChannel

from tests.conftest import run_parallel, FAKE_MPI as MPI


@pytest.fixture(autouse=True)
def _use_fake_mpi(monkeypatch):
    # Inject the thread-based fake communicator into the modules under test for
    # the duration of these tests only; real mpi4py is left untouched elsewhere.
    monkeypatch.setattr(mu, "MPI", MPI)
    monkeypatch.setattr(md, "MPI", MPI)


@pytest.fixture(autouse=True)
def _default_no_output_path(monkeypatch):
    # The real config defaults output_path to "" (not None), which makes every
    # MpiDistributor create a rank file on construction. None -> no files.
    monkeypatch.setattr(config.output, "output_path", None)


# --------------------------------------------------------------------------- #
# _get_node_aware_v_dist
# --------------------------------------------------------------------------- #
def test_node_aware_single_node():
    # 7 frequencies, 3 ranks, all on one node -> excess on the first ranks.
    def fn(comm, rank):
        sizes, slices = mu._get_node_aware_v_dist(7, comm)
        return list(sizes), [(s.start, s.stop) for s in slices]

    _, res = run_parallel(3, fn, hostnames=["host", "host", "host"])
    for sizes, slices in res:
        assert sizes == [3, 2, 2]
        assert slices == [(0, 3), (3, 5), (5, 7)]


def test_node_aware_multi_node():
    # 4 ranks split across 2 nodes (2 ranks each); 6 frequencies.
    # 3 freqs per node; within a node, excess on first local rank -> [2,1].
    hostnames = ["n0", "n1", "n0", "n1"]

    def fn(comm, rank):
        sizes, slices = mu._get_node_aware_v_dist(6, comm)
        return list(sizes)

    _, res = run_parallel(4, fn, hostnames=hostnames)
    sizes = res[0]
    assert sum(sizes) == 6
    # Every rank agrees on the global distribution.
    assert all(s == sizes for s in res)


def test_node_aware_uneven_nodes():
    # 3 nodes, 5 frequencies -> 2,2,1 across nodes; total preserved.
    hostnames = ["a", "b", "c", "a"]

    def fn(comm, rank):
        sizes, _ = mu._get_node_aware_v_dist(5, comm)
        return list(sizes)

    _, res = run_parallel(4, fn, hostnames=hostnames)
    assert sum(res[0]) == 5
    assert all(s == res[0] for s in res)


# --------------------------------------------------------------------------- #
# _send_in_chunks / _recv_in_chunks
# --------------------------------------------------------------------------- #
def test_send_recv_in_chunks_roundtrip():
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


# --------------------------------------------------------------------------- #
# Shared fixtures for BZ mapping tests
# --------------------------------------------------------------------------- #
# A real (2x2x1) square-lattice grid reduces 4 full-BZ q-points to 3
# irreducible ones; its inverse map is [0, 1, 1, 2] (point 1 is duplicated, so
# the IBZ->FBZ expansion is non-trivial). We derive the sizes and the reference
# expansion straight from the grid so the tests stay correct against the real
# KGrid (no hand-maintained mapping).
Q_NK = (2, 2, 1)
Q_SYMS = bz.two_dimensional_square_symmetries()
IRR_INV = KGrid(Q_NK, Q_SYMS).irrk_inv.ravel()
N_IRR = int(IRR_INV.max()) + 1
N_FULL = int(IRR_INV.size)
G_IRR = (np.arange(N_IRR * 2 * 2).reshape(N_IRR, 2, 2) + 1j).astype(np.complex128)
FULL_MAPPED = G_IRR[IRR_INV]  # shape (N_FULL, 2, 2)


def _q_grid():
    return KGrid(Q_NK, Q_SYMS)


def test_map_irrbz_fullbz():
    def fn(comm, rank):
        config.lattice.q_grid = _q_grid()
        d_irr = MpiDistributor(ntasks=N_IRR, comm=comm)
        d_full = MpiDistributor(ntasks=N_FULL, comm=comm)
        obj = FourPoint(G_IRR[d_irr.my_slice].copy(), nq=Q_NK, has_compressed_q_dimension=True)
        out = mu.map_irrbz_fullbz(obj, d_irr, d_full)
        return out.mat

    _, res = run_parallel(3, fn)
    rebuilt = np.concatenate(res, axis=0)
    assert np.allclose(rebuilt, FULL_MAPPED)


def test_exchange_and_map_matches_reference():
    def fn(comm, rank):
        config.lattice.q_grid = _q_grid()
        d_irr = MpiDistributor(ntasks=N_IRR, comm=comm)
        d_full = MpiDistributor(ntasks=N_FULL, comm=comm)
        obj = FourPoint(
            G_IRR[d_irr.my_slice].copy(), channel=SpinChannel.DENS, nq=Q_NK, has_compressed_q_dimension=True
        )
        out = mu.exchange_and_map_irrbz_fullbz(obj, d_irr, d_full)
        return None if out is None else out.mat

    _, res = run_parallel(3, fn)
    rebuilt = np.concatenate([r for r in res if r is not None], axis=0)
    assert np.allclose(rebuilt, FULL_MAPPED)


def test_exchange_and_map_single_rank():
    config.lattice.q_grid = _q_grid()
    comm = MPI.Comm(1)
    d_irr = MpiDistributor(ntasks=N_IRR, comm=comm)
    d_full = MpiDistributor(ntasks=N_FULL, comm=comm)
    obj = FourPoint(G_IRR.copy(), channel=SpinChannel.MAGN, nq=Q_NK, has_compressed_q_dimension=True)
    out = mu.exchange_and_map_irrbz_fullbz(obj, d_irr, d_full)
    assert np.allclose(out.mat, FULL_MAPPED)
    # metadata propagated to the new FourPoint
    assert out.channel == SpinChannel.MAGN
    assert out.nq == Q_NK


def test_exchange_and_map_auto_orbital_transform(monkeypatch):
    # Record every apply_auto_orbital_transform call. Patching the function on
    # the symmetry_reduction module works whether that module is the real one
    # or the in-process fake (the auto branch in mpi_utils calls it by name).
    calls = []

    def _recording_transform(full_mat, us, sigmas, conjs, num_orbital_dimensions):
        calls.append(
            {
                "n_us": None if us is None else len(us),
                "num_orbital_dimensions": num_orbital_dimensions,
            }
        )
        return full_mat  # identity, so the mapped result is unchanged

    monkeypatch.setattr(symmetry_reduction, "apply_auto_orbital_transform", _recording_transform)

    nb = 2
    auto_us = (np.arange(N_FULL * nb * nb).reshape(N_FULL, nb, nb) + 0.0).astype(np.complex128)
    auto_sigmas = np.arange(N_FULL)
    auto_conjs = np.zeros(N_FULL, dtype=bool)

    def _auto_grid():
        # Force a real KGrid into auto mode with controlled transform data,
        # without needing a Hamiltonian / specify_auto_symmetries().
        g = _q_grid()
        g._auto_mode = True
        g._auto_us = auto_us
        g._auto_sigmas = auto_sigmas
        g._auto_conjs = auto_conjs
        return g

    def fn(comm, rank):
        config.lattice.q_grid = _auto_grid()
        d_irr = MpiDistributor(ntasks=N_IRR, comm=comm)
        d_full = MpiDistributor(ntasks=N_FULL, comm=comm)
        obj = FourPoint(G_IRR[d_irr.my_slice].copy(), nq=Q_NK, has_compressed_q_dimension=True)
        out = mu.exchange_and_map_irrbz_fullbz(obj, d_irr, d_full)
        return out.mat, d_full.my_size

    _, res = run_parallel(3, fn)
    rebuilt = np.concatenate([m for m, _ in res], axis=0)
    # Fake transform is the identity, so the mapping result is unchanged.
    assert np.allclose(rebuilt, FULL_MAPPED)
    # The orbital transform was applied on each rank with my_size > 0, with the
    # rank-local slice sizes summing to the full BZ.
    applied_sizes = [c["n_us"] for c in calls]
    expected_sizes = [sz for _, sz in res if sz > 0]
    assert sorted(applied_sizes) == sorted(expected_sizes)
    assert sum(applied_sizes) == N_FULL
    assert all(c["num_orbital_dimensions"] == 4 for c in calls)


# --------------------------------------------------------------------------- #
# get_pencil_indices
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layout", ["flat", "z_pencil", "y_pencil", "x_pencil"])
@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_get_pencil_indices_partition(layout, size):
    nq = (2, 3, 2)
    n_tot = nq[0] * nq[1] * nq[2]
    parts = [mu.get_pencil_indices(r, size, nq, layout) for r in range(size)]
    allidx = np.concatenate(parts) if any(len(p) for p in parts) else np.array([], dtype=int)
    # Every global index is owned exactly once.
    assert np.array_equal(np.sort(allidx), np.arange(n_tot))


def test_get_pencil_indices_flat_matches_distributor():
    nq = (2, 2, 2)
    size = 3
    for r in range(size):
        idx = mu.get_pencil_indices(r, size, nq, "flat")
        d = MpiDistributor(ntasks=8, comm=MPI.Comm(size))
        # same excess-on-last convention
        sl = d.slices[r]
        assert np.array_equal(idx, np.arange(sl.start, sl.stop))


def test_get_pencil_indices_invalid_layout():
    with pytest.raises(ValueError):
        mu.get_pencil_indices(0, 1, (2, 2, 2), "bogus")


def test_get_pencil_indices_empty_partition():
    # More ranks than pencils -> some ranks own nothing.
    nq = (1, 1, 2)  # only 2 z-pencils? n_pencils for z = nx*ny = 1
    idx = mu.get_pencil_indices(3, 4, nq, "z_pencil")
    assert idx.size == 0


# --------------------------------------------------------------------------- #
# _redistribute_p2p
# --------------------------------------------------------------------------- #
def test_redistribute_flat_to_zpencil():
    nq = (2, 3, 2)
    n_tot = nq[0] * nq[1] * nq[2]
    G = (np.arange(n_tot * 2).reshape(n_tot, 2) + 1j).astype(np.complex128)

    def fn(comm, rank):
        src = mu.get_pencil_indices(rank, comm.size, nq, "flat")
        out = mu._redistribute_p2p(G[src].copy(), nq, comm, "flat", "z_pencil")
        return rank, out

    _, res = run_parallel(3, fn)
    for rank, out in res:
        expected = G[mu.get_pencil_indices(rank, 3, nq, "z_pencil")]
        assert np.allclose(out, expected)


def test_redistribute_roundtrip_multichunk(monkeypatch):
    nq = (2, 2, 2)
    n_tot = 8
    G = (np.arange(n_tot * 3).reshape(n_tot, 3) + 2j).astype(np.complex128)
    monkeypatch.setattr(mu, "MAX_MPI_BYTES", 16)

    def fn(comm, rank):
        src = mu.get_pencil_indices(rank, comm.size, nq, "flat")
        z = mu._redistribute_p2p(G[src].copy(), nq, comm, "flat", "z_pencil")
        back = mu._redistribute_p2p(z, nq, comm, "z_pencil", "flat")
        return rank, back

    _, res = run_parallel(2, fn)
    for rank, back in res:
        expected = G[mu.get_pencil_indices(rank, 2, nq, "flat")]
        assert np.allclose(back, expected)


# --------------------------------------------------------------------------- #
# execute_distributed_fft
# --------------------------------------------------------------------------- #
def _run_dist_fft(size, G, nq):
    """Run the distributed FFT across `size` ranks and reconstruct the global
    flat-layout result."""

    def fn(comm, rank):
        flat = mu.get_pencil_indices(rank, comm.size, nq, "flat")
        obj = FourPoint(G[flat].copy(), nq=nq)
        out = mu.execute_distributed_fft(obj, comm)
        return rank, flat, out.mat

    _, res = run_parallel(size, fn)
    rebuilt = np.empty_like(G)
    for rank, flat, mat in res:
        rebuilt[flat] = mat
    return rebuilt


def test_execute_distributed_fft_matches_numpy():
    nq = (2, 3, 2)
    nx, ny, nz = nq
    n_tot = nx * ny * nz
    rng = np.random.default_rng(0)
    G = (rng.standard_normal((n_tot, 2)) + 1j * rng.standard_normal((n_tot, 2))).astype(np.complex128)

    expected = np.fft.fftn(G.reshape(nx, ny, nz, 2), axes=(0, 1, 2)).reshape(n_tot, 2)

    for size in (1, 2, 3):
        rebuilt = _run_dist_fft(size, G.copy(), nq)
        assert np.allclose(rebuilt, expected), f"mismatch for size={size}"


# --------------------------------------------------------------------------- #
# gather_full_ibz_for_vslice
# --------------------------------------------------------------------------- #
def test_gather_full_ibz_for_vslice():
    n_irrq = 4
    n_v = 3
    n_vp = 2
    norb = 2
    # identity irr->full map so map_to_full_bz is a no-op
    q_grid = KGrid((n_irrq, 1, 1), [])
    G = (np.arange(n_irrq * norb * n_v * n_vp).reshape(n_irrq, norb, n_v, n_vp) + 1j).astype(np.complex128)

    def fn(comm, rank):
        config.lattice.q_grid = q_grid
        d_irrq = MpiDistributor(ntasks=n_irrq, comm=comm)
        d_v = MpiDistributor(ntasks=n_v, comm=comm)
        gamma = FourPoint(G[d_irrq.my_slice].copy(), nq=(2, 2, 1), has_compressed_q_dimension=True)
        out = mu.gather_full_ibz_for_vslice(gamma, d_irrq, d_v, q_grid)
        vslice = d_v.slices[rank]
        if out is None:
            return None
        return out.mat, (vslice.start, vslice.stop)

    _, res = run_parallel(2, fn, hostnames=["h", "h"])
    for r in res:
        assert r is not None  # both ranks own frequencies here
        mat, (vs, ve) = r
        # full IBZ q for this rank's v-slice
        assert np.allclose(mat, G[:, :, vs:ve, :])


def test_gather_full_ibz_for_vslice_empty_rank_returns_none():
    n_irrq = 3
    n_v = 1  # fewer frequencies than ranks -> rank 1 gets none
    n_vp = 2
    norb = 2
    q_grid = KGrid((n_irrq, 1, 1), [])
    G = (np.arange(n_irrq * norb * n_v * n_vp).reshape(n_irrq, norb, n_v, n_vp) + 1j).astype(np.complex128)

    def fn(comm, rank):
        config.lattice.q_grid = q_grid
        d_irrq = MpiDistributor(ntasks=n_irrq, comm=comm)
        d_v = MpiDistributor(ntasks=n_v, comm=comm)
        gamma = FourPoint(G[d_irrq.my_slice].copy(), nq=(3, 1, 1), has_compressed_q_dimension=True)
        out = mu.gather_full_ibz_for_vslice(gamma, d_irrq, d_v, q_grid)
        return rank, (out is None), d_v.my_size

    _, res = run_parallel(2, fn, hostnames=["h", "h"])
    # exactly one rank owns the single frequency; the other returns None
    none_flags = [is_none for _, is_none, _ in res]
    assert none_flags.count(True) == 1
    for _, is_none, my_v in res:
        assert is_none == (my_v == 0)

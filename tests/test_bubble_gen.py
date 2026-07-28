# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Parity tests for :class:`dgamore.bubble_gen.BubbleGenerator`. Each test compares the produced bubble against an
independent, explicit hand-rolled reference of the documented formula (small random Green's function + small grid),
which both validates correctness and locks the behavior of the memory-optimized methods.
"""

import numpy as np
import pytest

import dgamore.config as config
from dgamore import brillouin_zone as bz
from dgamore.bubble_gen import BubbleGenerator
from dgamore.greens_function import GreensFunction
from dgamore.matsubara_frequencies import MFHelper
from dgamore.n_point_base import FrequencyNotation


def _make_local_g(nb: int, niv: int, seed: int = 0) -> GreensFunction:
    """Builds a local Green's function ``[1, 1, 1, o1, o2, 2*niv]`` with reproducible random data."""
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((nb, nb, 2 * niv)) + 1j * rng.standard_normal((nb, nb, 2 * niv))
    return GreensFunction(mat[None, None, None, ...])


def _make_momentum_g(nk: tuple, nb: int, niv: int, seed: int = 0) -> GreensFunction:
    """Builds a decompressed momentum Green's function ``[kx, ky, kz, o1, o2, 2*niv]`` and aligns ``config.lattice``."""
    config.lattice.nk = nk
    rng = np.random.default_rng(seed)
    shape = (*nk, nb, nb, 2 * niv)
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return GreensFunction(mat, has_compressed_q_dimension=False, nk=nk)


def test_create_generalized_chi0_matches_reference():
    """create_generalized_chi0 matches an explicit hand-rolled reference of the documented formula."""
    nb, niv, niw, beta = 2, 1, 1, 2.0
    g = _make_local_g(nb, niv + niw + 1, seed=1)
    gm = g.mat[0, 0, 0]
    nivd = g.niv

    res = BubbleGenerator.create_generalized_chi0(g, niw, niv, beta)

    wn = MFHelper.wn(niw)
    niv_range = np.arange(-niv, niv)
    ref = np.zeros((nb, nb, nb, nb, len(wn), 2 * niv), dtype=np.complex128)
    for a in range(nb):
        for b in range(nb):
            for c in range(nb):
                for d in range(nb):
                    for iw in range(len(wn)):
                        for iv, v in enumerate(niv_range):
                            gl = gm[a, d, nivd + v]
                            gr = gm[c, b, nivd + v - wn[iw]]  # transpose_orbitals -> g[c,b]
                            ref[a, b, c, d, iw, iv] = -beta * gl * gr

    assert res.mat.shape == ref.shape
    assert np.allclose(res.mat, ref, atol=1e-4)


def test_create_generalized_chi0_q_matches_reference():
    """create_generalized_chi0_q matches an explicit per-q reference and stays q-compressed."""
    nk, nb, niv, niw, beta = (2, 2, 1), 2, 1, 1, 2.0
    g = _make_momentum_g(nk, nb, niv + niw + 1, seed=2)
    q_grid = bz.KGrid(nk, symmetries=[])
    q_list = q_grid.get_q_list()

    res = BubbleGenerator.create_generalized_chi0_q(g, niw, niv, q_list, q_grid, beta, use_gpu=False)

    gcut = g.cut_niv(niv + niw).mat  # [kx,ky,kz,o,o,2*(niv+niw)]
    niv_g = gcut.shape[-1] // 2
    gl = gcut[..., niv_g - niv : niv_g + niv]
    wn = MFHelper.wn(niw, return_only_positive=True)

    ref = np.zeros((len(q_list), nb, nb, nb, nb, len(wn), 2 * niv), dtype=np.complex128)
    for iq, q in enumerate(q_list):
        gr = np.roll(gcut, (q[0], q[1], q[2]), axis=(0, 1, 2))
        for iw, wn_i in enumerate(wn):
            s, e = niv_g - niv - wn_i, niv_g + niv - wn_i
            for a in range(nb):
                for b in range(nb):
                    for c in range(nb):
                        for d in range(nb):
                            ref[iq, a, b, c, d, iw, :] = (gl[..., a, d, :] * gr[..., c, b, s:e]).sum(axis=(0, 1, 2))
    ref *= -beta / q_grid.nk_tot

    assert res.mat.shape == ref.shape
    assert res.has_compressed_q_dimension is True
    assert np.allclose(res.mat, ref, atol=1e-4)


def test_create_generalized_chi0_q_no_copy_of_full_green_function():
    """create_generalized_chi0_q does not mutate its input Green's function."""
    nk, nb, niv, niw, beta = (2, 2, 1), 2, 1, 1, 2.0
    g = _make_momentum_g(nk, nb, niv + niw + 1, seed=3)
    g_before = g.mat.copy()
    q_grid = bz.KGrid(nk, symmetries=[])

    BubbleGenerator.create_generalized_chi0_q(g, niw, niv, q_grid.get_q_list(), q_grid, beta, use_gpu=False)

    assert np.array_equal(g.mat, g_before)  # input untouched


def test_create_generalized_chi0_pp_w0_matches_reference():
    """create_generalized_chi0_pp_w0 matches an explicit reference and carries PP notation."""
    nb, niv_pp, beta = 2, 2, 2.0
    g = _make_local_g(nb, niv_pp + 1, seed=4)
    gm = g.cut_niv(niv_pp).mat[0, 0, 0]
    n = 2 * niv_pp

    res = BubbleGenerator.create_generalized_chi0_pp_w0(g, niv_pp, beta)

    ref = np.zeros((nb, nb, nb, nb, 1, n), dtype=np.complex128)
    for a in range(nb):
        for b in range(nb):
            for c in range(nb):
                for d in range(nb):
                    for v in range(n):
                        # transpose_orbitals -> g[c,b]; flip last freq axis -> index n-1-v
                        ref[a, b, c, d, 0, v] = -beta * gm[a, d, v] * gm[c, b, n - 1 - v]

    assert res.mat.shape == ref.shape
    assert res.frequency_notation == FrequencyNotation.PP
    assert np.allclose(res.mat, ref, atol=1e-4)


def test_create_generalized_chi0_pp_w0_does_not_mutate_input():
    """create_generalized_chi0_pp_w0 flips only a private copy and leaves its input untouched."""
    nb, niv_pp, beta = 2, 2, 2.0
    g = _make_local_g(nb, niv_pp + 1, seed=5)
    g_before = g.mat.copy()
    BubbleGenerator.create_generalized_chi0_pp_w0(g, niv_pp, beta)
    assert np.array_equal(g.mat, g_before)


def test_create_generalized_chi0_q_pp_w0_matches_reference():
    """create_generalized_chi0_q_pp_w0 matches an explicit reference and carries PP notation."""
    nk, nb, niv_pp = (2, 2, 1), 2, 2
    g = _make_momentum_g(nk, nb, niv_pp + 1, seed=6)
    q_grid = bz.KGrid(nk, symmetries=[])

    res = BubbleGenerator.create_generalized_chi0_q_pp_w0(g, niv_pp, q_grid)

    gm = g.cut_niv(niv_pp).compress_q_dimension().mat
    nkt, n = gm.shape[0], 2 * niv_pp
    ref = np.zeros((nkt, nb, nb, nb, nb, n), dtype=np.complex128)
    for k in range(nkt):
        for a in range(nb):
            for b in range(nb):
                for c in range(nb):
                    for d in range(nb):
                        # G_14^{kv} * conj(G_32^{kv})   (transpose_orbitals -> g[c,b], conjugated)
                        ref[k, a, b, c, d, :] = gm[k, a, d, :] * np.conj(gm[k, c, b, :])

    assert res.mat.shape == ref.shape
    assert res.frequency_notation == FrequencyNotation.PP
    assert np.allclose(res.mat, ref, atol=1e-4)


def test_create_generalized_chi0_q_pp_w0_does_not_mutate_input():
    """create_generalized_chi0_q_pp_w0 conjugates only a private copy and leaves its input untouched."""
    nk, nb, niv_pp = (2, 2, 1), 2, 2
    g = _make_momentum_g(nk, nb, niv_pp + 1, seed=7)
    g_before = g.mat.copy()
    BubbleGenerator.create_generalized_chi0_q_pp_w0(g, niv_pp, bz.KGrid(nk, symmetries=[]))
    assert np.array_equal(g.mat, g_before)


def test_momentum_pp_bubble_is_local_pp_bubble_in_acbd_layout():
    """Momentum pp bubble equals the local pp bubble permuted 'abcd->acbd' (up to the -beta the momentum form omits)."""
    nb, niv_pp, beta = 2, 3, 2.0
    nk = (2, 2, 1)
    config.lattice.nk = nk
    q_grid = bz.KGrid(nk, symmetries=[])
    rng = np.random.default_rng(11)
    nvg = niv_pp + 2
    # asymmetric G_ij != G_ji obeying the physical conjugation symmetry G_ij(-nu) = conj(G_ji(nu))
    half = rng.standard_normal((nb, nb, nvg)) + 1j * rng.standard_normal((nb, nb, nvg))
    gloc = np.empty((nb, nb, 2 * nvg), dtype=np.complex64)
    gloc[:, :, nvg:] = half
    gloc[:, :, :nvg] = np.conj(np.swapaxes(half, 0, 1))[:, :, ::-1]
    g_loc = GreensFunction(gloc[None, None, None, ...])
    g_mom = GreensFunction(
        np.broadcast_to(gloc, (*nk, nb, nb, 2 * nvg)).copy(), has_compressed_q_dimension=False, nk=nk
    )
    local_acbd = BubbleGenerator.create_generalized_chi0_pp_w0(g_loc, niv_pp, beta).permute_orbitals("abcd->acbd")
    momentum = BubbleGenerator.create_generalized_chi0_q_pp_w0(g_mom, niv_pp, q_grid).decompress_q_dimension()
    assert np.allclose(momentum.mat, local_acbd.mat[..., 0, :][None, None, None] / (-beta), atol=1e-4)


def _fft_bubble_reference(g, niw, niv, q_grid, beta):
    """Computes the single-rank rank-0 FFT bubble (the mock-comm path) as the distributed-path reference."""
    from dgamore.mpi_utils import MpiDistributor
    from tests.conftest import create_comm_mock

    dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=create_comm_mock(), name="Q")
    return BubbleGenerator.create_generalized_chi0_q_fft(dist, g, niw, niv, q_grid, beta)


def test_fft_bubble_matches_direct_einsum_bubble():
    """The FFT bubble (scipy c64 ifft) matches the direct einsum bubble on the irr wedge and stays complex64."""
    nk, nb, niv, niw, beta = (4, 4, 1), 2, 2, 2, 2.5
    g = _make_momentum_g(nk, nb, niv + niw + 2, seed=7)
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    fft_bubble = _fft_bubble_reference(g, niw, niv, q_grid, beta)
    direct = BubbleGenerator.create_generalized_chi0_q(g, niw, niv, np.array(q_grid.get_irrq_list()), q_grid, beta)
    assert fft_bubble.mat.dtype == np.complex64
    assert fft_bubble.mat.shape == direct.mat.shape
    assert np.allclose(fft_bubble.mat, direct.mat, atol=1e-4)


def test_fft_bubble_distributed_matches_single_rank(monkeypatch):
    """The R-scattered multi-rank FFT bubble reproduces the single-rank rank-0 path on a non-trivial wedge."""
    import dgamore.mpi_utils as mpi_utils
    from dgamore.mpi_utils import MpiDistributor
    from tests.conftest import FAKE_MPI, run_parallel

    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, nb, niv, niw, beta = (4, 4, 1), 2, 2, 2, 2.5
    g = _make_momentum_g(nk, nb, niv + niw + 2, seed=11)
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    ref = _fft_bubble_reference(g, niw, niv, q_grid, beta)

    def fn(comm, rank):
        dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
        return BubbleGenerator.create_generalized_chi0_q_fft(dist, g, niw, niv, q_grid, beta).mat

    _, res = run_parallel(3, fn)
    assembled = np.concatenate(res, axis=0)
    assert assembled.shape == ref.mat.shape
    assert np.allclose(assembled, ref.mat, atol=1e-5)


def test_fft_bubble_distributed_with_node_shared_greens_function(monkeypatch):
    """The distributed bubble builds the R-space Green's function once per node in a shared window and matches."""
    import dgamore.mpi_utils as mpi_utils
    from dgamore.mpi_utils import MpiDistributor
    from tests.conftest import FAKE_MPI, run_parallel

    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, nb, niv, niw, beta = (2, 2, 1), 2, 1, 1, 2.0
    g = _make_momentum_g(nk, nb, niv + niw + 1, seed=12)
    q_grid = bz.KGrid(nk, symmetries=[])
    ref = _fft_bubble_reference(g, niw, niv, q_grid, beta)

    def fn(comm, rank):
        node_comm = comm.Split_type(FAKE_MPI.COMM_TYPE_SHARED)
        dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
        return BubbleGenerator.create_generalized_chi0_q_fft(dist, g, niw, niv, q_grid, beta, node_comm=node_comm).mat

    _, res = run_parallel(4, fn, hostnames=["n0", "n0", "n1", "n1"])
    assembled = np.concatenate(res, axis=0)
    assert np.allclose(assembled, ref.mat, atol=1e-5)


def test_fft_bubble_distributed_with_fewer_columns_than_ranks(monkeypatch):
    """With fewer (w,v) columns than ranks the idle ranks still receive their q-slice and the bubble matches."""
    import dgamore.mpi_utils as mpi_utils
    from dgamore.mpi_utils import MpiDistributor
    from tests.conftest import FAKE_MPI, run_parallel

    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, nb, niv, niw, beta = (5, 1, 1), 1, 1, 0, 2.0
    g = _make_momentum_g(nk, nb, niv + niw + 1, seed=13)
    q_grid = bz.KGrid(nk, symmetries=[])
    ref = _fft_bubble_reference(g, niw, niv, q_grid, beta)

    def fn(comm, rank):
        dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
        return BubbleGenerator.create_generalized_chi0_q_fft(dist, g, niw, niv, q_grid, beta).mat

    _, res = run_parallel(5, fn)
    assembled = np.concatenate(res, axis=0)
    assert np.allclose(assembled, ref.mat, atol=1e-5)

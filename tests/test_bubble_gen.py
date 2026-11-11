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

import types

import numpy as np
import pytest

import dgamore.config as config
import dgamore.mpi_utils as mpi_utils
import dgamore.n_point_base as n_point_base
from dgamore import brillouin_zone as bz
from dgamore import nonlocal_sde
from dgamore.bubble_gen import BubbleGenerator
from dgamore.greens_function import GreensFunction
from dgamore.matsubara_frequencies import MFHelper
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import FrequencyNotation
from tests.conftest import FAKE_MPI, create_comm_mock, run_parallel, traced_peak


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


def _make_hermitian_g(nk: tuple, nb: int, niv: int, seed: int = 0) -> GreensFunction:
    """Builds a momentum Green's function obeying G(k, -v) = G(k, v)^dagger exactly, from random data at v >= 0."""
    g = _make_momentum_g(nk, nb, niv, seed)
    g.mat[..., :niv] = np.conj(g.mat[..., niv:][..., ::-1].swapaxes(-2, -3))
    return g


def _make_crossing_g(nk: tuple, nb: int, niv: int, seed: int = 0, hermitian: bool = True) -> GreensFunction:
    """Builds a momentum Green's function obeying G(-k, v) = G(k, v)^T and G(k, -v) = G(k, v)^dagger exactly."""
    g = _make_momentum_g(nk, nb, niv, seed)
    flipped = np.roll(np.flip(g.mat, axis=(0, 1, 2)), 1, axis=(0, 1, 2)).swapaxes(-2, -3)
    g.mat[...] = (g.mat + flipped) / 2  # commutative sum: the -k partner gets the same bits
    if hermitian:
        g.mat[..., :niv] = np.conj(g.mat[..., niv:][..., ::-1].swapaxes(-2, -3))
    return g


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


def test_create_generalized_chi0_q_loop_matches_reference():
    """create_generalized_chi0_q_loop matches an explicit per-q reference and stays q-compressed."""
    nk, nb, niv, niw, beta = (2, 2, 1), 2, 1, 1, 2.0
    g = _make_momentum_g(nk, nb, niv + niw + 1, seed=2)
    q_grid = bz.KGrid(nk, symmetries=[])
    q_list = q_grid.get_q_list()

    res = BubbleGenerator.create_generalized_chi0_q_loop(g, niw, niv, q_list, q_grid, beta)

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


def test_create_generalized_chi0_q_loop_no_copy_of_full_green_function():
    """create_generalized_chi0_q_loop does not mutate its input Green's function."""
    nk, nb, niv, niw, beta = (2, 2, 1), 2, 1, 1, 2.0
    g = _make_momentum_g(nk, nb, niv + niw + 1, seed=3)
    g_before = g.mat.copy()
    q_grid = bz.KGrid(nk, symmetries=[])

    BubbleGenerator.create_generalized_chi0_q_loop(g, niw, niv, q_grid.get_q_list(), q_grid, beta)

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
    minus_k = [np.ravel_multi_index(tuple(-np.array(np.unravel_index(k, nk)) % nk), nk) for k in range(nkt)]
    ref = np.zeros((nkt, nb, nb, nb, nb, n), dtype=np.complex128)
    for k in range(nkt):
        for a in range(nb):
            for b in range(nb):
                for c in range(nb):
                    for d in range(nb):
                        # G_14^{k,v} * G_23^{-k,-v}
                        ref[k, a, b, c, d, :] = gm[k, a, d, :] * gm[minus_k[k], b, c, ::-1]

    assert res.mat.shape == ref.shape
    assert res.frequency_notation == FrequencyNotation.PP
    assert np.allclose(res.mat, ref, atol=1e-4)


def test_momentum_pp_bubble_transforms_like_a_four_point_object_under_a_unit_cell_relabeling():
    """Relabeling one orbital's cell (G -> D G D^+, D(-k) = D(k)^*) rephases the pp bubble by D_1 D_2^* D_3 D_4^*."""
    nk, nb, niv_pp = (4, 1, 1), 2, 3
    q_grid = bz.KGrid(nk, symmetries=[])
    g = _make_momentum_g(nk, nb, niv_pp, seed=12)
    d = np.ones((*nk, nb), dtype=np.complex128)
    d[..., 1] = np.exp(2j * np.pi * np.arange(nk[0]) / nk[0])[:, None, None]
    mat = d[..., :, None, None] * g.mat * d[..., None, :, None].conj()
    shifted = GreensFunction(mat, has_compressed_q_dimension=False, nk=nk)
    bubble, bubble_shifted = (
        BubbleGenerator.create_generalized_chi0_q_pp_w0(x, niv_pp, q_grid).decompress_q_dimension().mat
        for x in (g, shifted)
    )
    dc = d.conj()
    phase = d[..., :, None, None, None] * dc[..., None, :, None, None] * d[..., None, None, :, None]
    assert np.allclose(bubble_shifted, (phase * dc[..., None, None, None, :])[..., None] * bubble, atol=1e-5)


def test_create_generalized_chi0_q_pp_w0_does_not_mutate_input():
    """create_generalized_chi0_q_pp_w0 flips only a private copy and leaves its input untouched."""
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
    direct = BubbleGenerator.create_generalized_chi0_q_loop(g, niw, niv, np.array(q_grid.get_irrq_list()), q_grid, beta)
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

    # one Green's function per fake rank, as every MPI rank holds its own: cut_niv parks its source's mat at None
    # while it clones it (_clone_without_mat), so ranks sharing one object race on it
    g_ranks = [g.copy() for _ in range(3)]

    def fn(comm, rank):
        dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
        return BubbleGenerator.create_generalized_chi0_q_fft(dist, g_ranks[rank], niw, niv, q_grid, beta).mat

    _, res = run_parallel(3, fn)
    assembled = np.concatenate(res, axis=0)
    assert assembled.shape == ref.mat.shape
    assert np.allclose(assembled, ref.mat, atol=1e-5)


@pytest.mark.parametrize("one_column_budget", [False, True])
@pytest.mark.parametrize(
    "nb, ranks, hostnames, compressed", [(1, 2, None, False), (2, 3, ["n0", "n0", "n1"], True), (3, 2, None, False)]
)
def test_fft_bubble_distributed_round_buffer_budget_keeps_the_rank0_path_bits(
    nb, ranks, hostnames, compressed, one_column_budget, monkeypatch
):
    """A round transforms several columns within the byte budget, one column above it, and keeps the rank-0 bits."""
    import scipy as sp

    import dgamore.bubble_gen as bubble_gen
    import dgamore.mpi_utils as mpi_utils
    from dgamore.mpi_utils import MpiDistributor
    from tests.conftest import FAKE_MPI, run_parallel

    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, niv, niw, beta = (4, 4, 1), 4, 3, 2.5
    if one_column_budget:
        monkeypatch.setattr(bubble_gen, "ROUND_BUFFER_BYTES", int(np.prod(nk)) * nb**4 * 8)
    g = _make_momentum_g(nk, nb, niv + niw + 2, seed=20 + nb)
    g = g.compress_q_dimension() if compressed else g
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    ref = _fft_bubble_reference(g, niw, niv, q_grid, beta)
    shapes, ifftn = [], sp.fft.ifftn
    monkeypatch.setattr(sp.fft, "ifftn", lambda x, **kwargs: shapes.append(x.shape) or ifftn(x, **kwargs))

    # one Green's function per fake rank, as every MPI rank holds its own: cut_niv parks its source's mat at None
    # while it clones it (_clone_without_mat), so ranks sharing one object race on it
    g_ranks = [g.copy() for _ in range(ranks)]

    def fn(comm, rank):
        node_comm = comm.Split_type(FAKE_MPI.COMM_TYPE_SHARED) if hostnames else None
        dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
        return BubbleGenerator.create_generalized_chi0_q_fft(
            dist, g_ranks[rank], niw, niv, q_grid, beta, node_comm=node_comm
        ).mat

    _, res = run_parallel(ranks, fn, hostnames=hostnames)
    assert np.array_equal(np.concatenate(res, axis=0), ref.mat)
    assert (max(s[-1] for s in shapes) == 1) == one_column_budget and sum(s[-1] for s in shapes) == (niw + 1) * 2 * niv


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

    # one Green's function per fake rank, as every MPI rank holds its own: cut_niv parks its source's mat at None
    # while it clones it (_clone_without_mat), so ranks sharing one object race on it
    g_ranks = [g.copy() for _ in range(4)]

    def fn(comm, rank):
        node_comm = comm.Split_type(FAKE_MPI.COMM_TYPE_SHARED)
        dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
        return BubbleGenerator.create_generalized_chi0_q_fft(
            dist, g_ranks[rank], niw, niv, q_grid, beta, node_comm=node_comm
        ).mat

    _, res = run_parallel(4, fn, hostnames=["n0", "n0", "n1", "n1"])
    assembled = np.concatenate(res, axis=0)
    assert np.allclose(assembled, ref.mat, atol=1e-5)


@pytest.mark.parametrize("crossing", [False, True])
@pytest.mark.parametrize("nk, ranks", [((5, 1, 1), 5), ((2, 1, 1), 3)])
def test_fft_bubble_distributed_with_fewer_columns_than_ranks(nk, ranks, crossing, monkeypatch):
    """With fewer (w,v) columns or momenta than ranks the idle ranks still take part and the bubble matches."""
    import dgamore.mpi_utils as mpi_utils
    from dgamore.mpi_utils import MpiDistributor
    from tests.conftest import FAKE_MPI, run_parallel

    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nb, niv, niw, beta = 1, 1, 0, 2.0
    g = (_make_crossing_g if crossing else _make_momentum_g)(nk, nb, niv + niw + 1, seed=13)
    q_grid = bz.KGrid(nk, symmetries=[])
    ref = _fft_bubble_reference(g, niw, niv, q_grid, beta)

    # one Green's function per fake rank, as every MPI rank holds its own: cut_niv parks its source's mat at None
    # while it clones it (_clone_without_mat), so ranks sharing one object race on it
    g_ranks = [g.copy() for _ in range(ranks)]

    def fn(comm, rank):
        dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
        return BubbleGenerator.create_generalized_chi0_q_fft(dist, g_ranks[rank], niw, niv, q_grid, beta).mat

    _, res = run_parallel(ranks, fn)
    assembled = np.concatenate(res, axis=0)
    assert np.array_equal(assembled, ref.mat)


def test_crossing_bubble_matches_the_direct_bubble_and_derives_its_partners_from_their_own_momentum_bit_for_bit():
    """The derived column (w, w - v) is the conjugate of the computed (w, v) at the same q with orbitals 3412."""
    nk, nb, niv, niw, beta = (4, 3, 1), 2, 3, 3, 2.5
    n = niv + niw + 2
    g = _make_crossing_g(nk, nb, n, seed=75)
    # time reversal off by 1e-7, inside the guard
    g.mat[..., n:] += 1e-7 * _make_momentum_g(nk, nb, n, seed=76).mat[..., n:]
    g.mat[..., :n] = np.conj(g.mat[..., n:][..., ::-1].swapaxes(-2, -3))
    assert BubbleGenerator._has_crossing_symmetries(g, niv + niw)
    q_grid = bz.KGrid(nk, symmetries=[])
    bubble = _fft_bubble_reference(g, niw, niv, q_grid, beta).mat
    q_list = np.array(q_grid.get_irrq_list())
    direct = BubbleGenerator.create_generalized_chi0_q_loop(g, niw, niv, q_list, q_grid, beta).mat
    assert np.allclose(bubble, direct, atol=1e-4)
    columns, partners = BubbleGenerator._crossing_columns(MFHelper.wn(niw, return_only_positive=True), niv)
    for column, partner in zip(columns[partners >= 0], partners[partners >= 0]):
        (iw, iv), ip = divmod(column, 2 * niv), partner % (2 * niv)
        assert np.array_equal(bubble[..., iw, ip], np.conj(bubble[..., iw, iv].transpose(0, 3, 4, 1, 2)))


@pytest.mark.parametrize(
    "nb, ranks, hostnames, compressed, one_column_budget",
    [
        (2, 2, None, True, True),
        (2, 3, ["n0", "n0", "n1"], False, False),
        (3, 3, None, False, True),
        (1, 4, ["n0", "n0", "n1", "n1"], True, False),
    ],
)
def test_crossing_bubble_of_a_crossing_g_transforms_only_the_computed_columns_and_keeps_the_rank0_bits(
    nb, ranks, hostnames, compressed, one_column_budget, monkeypatch
):
    """On a symmetric G only the crossing's computed columns are transformed, with the rank-0 bits at any rank count."""
    import scipy as sp

    import dgamore.bubble_gen as bubble_gen

    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, niv, niw, beta = (4, 4, 1), 4, 3, 2.5
    if one_column_budget:
        monkeypatch.setattr(bubble_gen, "ROUND_BUFFER_BYTES", int(np.prod(nk)) * nb**4 * 8)
    g = _make_crossing_g(nk, nb, niv + niw + 2, seed=70 + nb)
    g = g.compress_q_dimension() if compressed else g
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    n_computed = len(BubbleGenerator._crossing_columns(MFHelper.wn(niw, return_only_positive=True), niv)[0])
    shapes, ifftn = [], sp.fft.ifftn
    monkeypatch.setattr(sp.fft, "ifftn", lambda x, **kwargs: shapes.append(x.shape) or ifftn(x, **kwargs))
    ref = _fft_bubble_reference(g, niw, niv, q_grid, beta).mat
    assert sum(s[-1] for s in shapes) == n_computed < (niw + 1) * 2 * niv
    shapes.clear()

    # one Green's function per fake rank, as every MPI rank holds its own: cut_niv parks its source's mat at None
    # while it clones it (_clone_without_mat), so ranks sharing one object race on it
    g_ranks = [g.copy() for _ in range(ranks)]

    def fn(comm, rank):
        node_comm = comm.Split_type(FAKE_MPI.COMM_TYPE_SHARED) if hostnames else None
        dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
        return BubbleGenerator.create_generalized_chi0_q_fft(
            dist, g_ranks[rank], niw, niv, q_grid, beta, node_comm=node_comm
        ).mat

    _, res = run_parallel(ranks, fn, hostnames=hostnames)
    assert np.array_equal(np.concatenate(res, axis=0), ref)
    assert sum(s[-1] for s in shapes) == n_computed


def test_crossing_bubble_exchanges_only_the_computed_columns(monkeypatch):
    """Every rank derives the partners of its own momenta, so only the computed columns are sent."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, nb, niv, niw, beta, ranks = (4, 4, 1), 2, 4, 3, 2.5, 3
    g = _make_crossing_g(nk, nb, niv + niw + 2, seed=81)
    q_grid = bz.KGrid(nk, symmetries=[])
    n_computed = len(BubbleGenerator._crossing_columns(MFHelper.wn(niw, return_only_positive=True), niv)[0])
    sent, isend = [], mpi_utils._isend_rows
    monkeypatch.setattr(
        mpi_utils,
        "_isend_rows",
        lambda comm, arr, *args, **kwargs: sent.append(arr.shape[-1]) or isend(comm, arr, *args, **kwargs),
    )
    g_ranks = [g.copy() for _ in range(ranks)]

    def fn(comm, rank):
        dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
        return BubbleGenerator.create_generalized_chi0_q_fft(dist, g_ranks[rank], niw, niv, q_grid, beta).mat

    run_parallel(ranks, fn)
    assert sum(sent) == n_computed * (ranks - 1)


@pytest.mark.parametrize("ranks", [1, 3])
def test_bubble_of_a_hermitian_g_without_time_reversal_transforms_every_column(ranks, monkeypatch):
    """A Hermitian G without time reversal skips the crossing: every column is transformed, with the plain bits."""
    import scipy as sp

    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, nb, niv, niw, beta = (4, 4, 1), 2, 4, 3, 2.5
    g = _make_hermitian_g(nk, nb, niv + niw + 2, seed=80)
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    g_ranks = [g.copy() for _ in range(2 * ranks)]

    def bubble(first: int) -> np.ndarray:
        if ranks == 1:
            return _fft_bubble_reference(g_ranks[first], niw, niv, q_grid, beta).mat

        def fn(comm, rank):
            dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
            return BubbleGenerator.create_generalized_chi0_q_fft(
                dist, g_ranks[first + rank], niw, niv, q_grid, beta
            ).mat

        return np.concatenate(run_parallel(ranks, fn)[1], axis=0)

    shapes, ifftn = [], sp.fft.ifftn
    monkeypatch.setattr(sp.fft, "ifftn", lambda x, **kwargs: shapes.append(x.shape) or ifftn(x, **kwargs))
    result = bubble(0)
    assert sum(s[-1] for s in shapes) == (niw + 1) * 2 * niv
    monkeypatch.setattr(BubbleGenerator, "_has_crossing_symmetries", staticmethod(lambda *args: False))
    assert np.array_equal(result, bubble(ranks))


@pytest.mark.parametrize("crossing", [False, True])
@pytest.mark.parametrize("size", [1, 2])
def test_fft_bubble_subtracted_in_place_has_the_bits_of_the_difference_of_both_bubbles(size, crossing, monkeypatch):
    """A bubble subtracted from another as its blocks arrive equals a.sub(b) of the two built bubbles bit for bit."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, nb, niv, niw, beta = (4, 4, 1), 2, 2, 2, 2.5
    make = _make_crossing_g if crossing else _make_momentum_g
    greens = [make(nk, nb, niv + niw + 2, seed=seed) for seed in (14, 15)]
    for g in greens:
        g.mat[..., 1, 0, :] *= 1e-7  # bubble entries below the filter threshold
        g.mat[..., 0, 1, :] *= 1e-7  # and the transposed partners, so both symmetries survive
    assert all(BubbleGenerator._has_crossing_symmetries(g, niv + niw) == crossing for g in greens)
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())

    def fn(comm, rank):
        dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
        a, b, target = (
            BubbleGenerator.create_generalized_chi0_q_fft(dist, g, niw, niv, q_grid, beta) for g in (*greens, greens[0])
        )
        out = BubbleGenerator.create_generalized_chi0_q_fft(dist, greens[1], niw, niv, q_grid, beta, subtract_from=a)
        return out is a, a.mat, target.sub(b, copy=False).mat

    _, results = run_parallel(size, fn)
    assert all(same and np.array_equal(out, ref) for same, out, ref in results)
    assert np.any(np.concatenate([ref for *_, ref in results]) == 0)


def test_distributed_fft_bubble_subtracted_in_place_never_allocates_its_own_array():
    """The column path subtracting into a bubble peaks below the bubble it would otherwise build, on one rank."""
    nk, nb, niv, niw, beta = (4, 4, 1), 2, 6, 6, 2.5
    g = _make_momentum_g(nk, nb, niv + niw + 2, seed=16)
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=create_comm_mock(), name="Q")
    peaks = []
    for subtract in (False, True):
        target = BubbleGenerator._create_generalized_chi0_q_fft_distributed(dist, g, niw, niv, q_grid, beta)
        _, peak = traced_peak(
            lambda: BubbleGenerator._create_generalized_chi0_q_fft_distributed(
                dist, g, niw, niv, q_grid, beta, subtract_from=target if subtract else None
            )
        )
        peaks.append(peak)
    assert peaks[0] - peaks[1] >= 0.95 * target.mat.nbytes


def test_distributed_fft_bubble_subtracts_received_blocks_in_the_precision_they_were_sent_in(monkeypatch):
    """On 2 ranks a complex64 bubble subtracted into a complex128 target keeps its own rounding in every block."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, nb, niv, niw, beta = (4, 4, 1), 2, 2, 2, 2.5
    greens = [_make_momentum_g(nk, nb, niv + niw + 2, seed=seed) for seed in (17, 18)]
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())

    def fn(comm, rank):
        dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
        a, b = (BubbleGenerator.create_generalized_chi0_q_fft(dist, g, niw, niv, q_grid, beta) for g in greens)
        target = types.SimpleNamespace(mat=a.mat.astype(np.complex128))
        BubbleGenerator.create_generalized_chi0_q_fft(dist, greens[1], niw, niv, q_grid, beta, subtract_from=target)
        return b.mat.dtype, target.mat, a.mat.astype(np.complex128) - b.mat

    _, results = run_parallel(2, fn)
    assert all(dtype == np.complex64 and np.array_equal(out, ref) for dtype, out, ref in results)


@pytest.mark.parametrize("niw, niv", [(0, 3), (3, 2), (2, 4), (5, 1), (40, 140)])
def test_crossing_columns_split_every_frequency_into_computed_columns_and_their_partners(niw, niv):
    """Every column is computed or the v' = w - v partner of exactly one computed column, which lies above it."""
    wn = MFHelper.wn(niw, return_only_positive=True)
    n_v = 2 * niv
    columns, partners = BubbleGenerator._crossing_columns(wn, niv)
    derived = partners[partners >= 0]
    assert np.array_equal(np.sort(np.concatenate([columns, derived])), np.arange(len(wn) * n_v))
    iw, iv = np.divmod(columns[partners >= 0], n_v)
    assert np.array_equal(derived // n_v, iw) and np.array_equal(derived % n_v, 2 * niv + wn[iw] - 1 - iv)
    assert np.all(derived % n_v > iv)
    assert np.sum(columns < n_v) == niv  # w = 0 pairs every v with -v
    if (niw, niv) == (40, 140):
        assert len(columns) == 6160


@pytest.mark.parametrize("compressed", [False, True])
def test_crossing_guard_holds_for_a_hermitian_time_reversal_g_and_fails_without_either(compressed, monkeypatch):
    """The guard agrees over the ranks, sees a perturbation in one rank's share and ignores one outside the window."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, nb, niv, niv_window = (4, 3, 1), 2, 5, 3
    g = _make_crossing_g(nk, nb, niv, seed=50)
    inside, outside = g.copy(), g.copy()
    inside.mat[3, 2, 0, 0, 1, niv - niv_window] += 1e-3  # flat k 11 is rank 2's share, its -k row 4 rank 1's
    outside.mat[3, 2, 0, 0, 1, niv - niv_window - 1] += 1e-3
    hermitian_only = _make_hermitian_g(nk, nb, niv, seed=51)
    time_reversal_only = _make_crossing_g(nk, nb, niv, seed=52, hermitian=False)
    greens = [g, inside, outside, hermitian_only, time_reversal_only]
    greens = [x.compress_q_dimension() if compressed else x for x in greens]

    def fn(comm, rank):
        return [BubbleGenerator._has_crossing_symmetries(x, niv_window, comm) for x in greens]

    expected = [True, False, True, False, False]
    assert run_parallel(3, fn)[1] == [expected] * 3
    assert [BubbleGenerator._has_crossing_symmetries(x, niv_window) for x in greens] == expected


@pytest.mark.parametrize("dtype, holds", [(np.complex64, True), (np.complex128, False)])
def test_crossing_guard_demands_the_storage_precision_below_one_in_a_million(dtype, holds, monkeypatch):
    """A 1e-7 break of time reversal or Hermiticity passes in complex64 and fails in complex128 (64 ulps)."""
    monkeypatch.setattr(n_point_base, "DTYPE", dtype)
    nk, nb, niv = (4, 3, 1), 2, 5
    exact = _make_crossing_g(nk, nb, niv, seed=53)
    noise = 1e-7 * np.abs(exact.mat).max()
    time_reversal, hermiticity = exact.copy(), exact.copy()
    time_reversal.mat[..., niv:] += noise * _make_momentum_g(nk, nb, niv, seed=54).mat[..., niv:] / 5
    time_reversal.mat[..., :niv] = np.conj(time_reversal.mat[..., niv:][..., ::-1].swapaxes(-2, -3))
    hermiticity.mat[1, 2, 0, 0, 1, 0] += noise
    assert exact.mat.dtype == dtype and BubbleGenerator._has_crossing_symmetries(exact, niv)
    assert [BubbleGenerator._has_crossing_symmetries(g, niv) for g in (time_reversal, hermiticity)] == [holds] * 2


def test_column_pass_peak_does_not_grow_with_the_number_of_rounds():
    """A rank drops a round's irreducible rows before it builds the next round, so four rounds peak as one does."""
    nk, nb, n_v = (16, 16, 1), 2, 8
    q_grid = bz.KGrid(nk, symmetries=[])
    dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=create_comm_mock(), name="Q")

    def peak(n_w: int) -> int:
        def run():
            fill = lambda out, part, iw, iv: out.fill(1.0)
            BubbleGenerator._run_columns(dist, q_grid, n_w, n_v, 1, nb, np.complex64, fill, lambda where, b: None)

        return traced_peak(run)[1]

    one_round = peak(1)
    assert peak(4) <= 1.05 * one_round


def _on_ranks(ranks: int, fn, hostnames=None) -> list:
    """Runs fn(comm, rank) on the single-rank mock communicator for one rank, else on threaded fake ranks."""
    return [fn(create_comm_mock(), 0)] if ranks == 1 else run_parallel(ranks, fn, hostnames=hostnames)[1]


def _window_bubbles(comm, greens, niv, niw, q_grid, beta, bounds) -> list:
    """Builds the windows of G and the changes, runs the window bubble per bosonic window and frees the windows."""
    node_comm = comm.Split_type(FAKE_MPI.COMM_TYPE_SHARED)
    dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
    built = [nonlocal_sde._build_rspace_giwk_window(g, node_comm, niv_window=niv + niw) for g in greens]
    groups = [
        BubbleGenerator.create_generalized_chi0_q_fft_window(
            dist, built[0][0], niv, q_grid, beta, (w0, w1), [g_r for g_r, _ in built[1:]]
        )
        for w0, w1 in zip(bounds[:-1], bounds[1:])
    ]
    wins = [win for _, win in built]
    built = None
    for win in wins:
        nonlocal_sde._free_shared_window(win, node_comm)
    return [np.concatenate([group[i].mat for group in groups], axis=-2) for i in range(len(greens))]


@pytest.mark.parametrize("bounds", [(0, 4), (0, 3, 4), (0, 1, 2, 3, 4)])
@pytest.mark.parametrize(
    "nb, ranks, hostnames, compressed",
    [(1, 1, None, False), (2, 1, None, True), (2, 3, ["n0", "n0", "n1"], True), (3, 2, None, False)],
)
def test_window_bubble_columns_have_the_bits_of_the_whole_bubble(bounds, nb, ranks, hostnames, compressed, monkeypatch):
    """Every bosonic window of the window bubble holds the bits of the rank-0 bubble's columns at any rank count."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, niv, niw, beta = (4, 4, 1), 3, 3, 2.5
    g = _make_momentum_g(nk, nb, niv + niw + 2, seed=30 + nb)
    g = g.compress_q_dimension() if compressed else g
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    ref = _fft_bubble_reference(g, niw, niv, q_grid, beta).mat

    def fn(comm, rank):
        return _window_bubbles(comm, [g], niv, niw, q_grid, beta, bounds)[0]

    assert np.array_equal(np.concatenate(_on_ranks(ranks, fn, hostnames), axis=0), ref)


@pytest.mark.parametrize("one_column_budget", [False, True])
@pytest.mark.parametrize("nb, ranks, hostnames", [(1, 1, None), (2, 1, None), (2, 3, ["n0", "n0", "n1"])])
def test_window_bubble_responses_are_window_invariant_and_match_the_polarization_pair(
    nb, ranks, hostnames, one_column_budget, monkeypatch
):
    """Each change's bilinear bubble keeps its bits in any bosonic window and equals the c128 polarization pair."""
    import dgamore.bubble_gen as bubble_gen

    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    monkeypatch.setattr(n_point_base, "DTYPE", np.complex128)
    nk, niv, niw, beta = (4, 4, 1), 3, 3, 2.5
    if one_column_budget:
        monkeypatch.setattr(bubble_gen, "ROUND_BUFFER_BYTES", int(np.prod(nk)) * nb**4 * 16)
    g, *dgs = (_make_momentum_g(nk, nb, niv + niw + 2, seed=seed) for seed in (40, 41, 42))
    for x in (g, *dgs):
        x.mat[..., nb - 1, 0, :] *= 1e-7 if nb > 1 else 1.0  # entries of the shifted bubbles below the filter threshold
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    polarizations = []
    for dg in dgs:
        s = float(np.linalg.norm(g.mat) / np.linalg.norm(dg.mat))
        shifted = (
            GreensFunction(g.mat + sign * s * dg.mat, has_compressed_q_dimension=False, nk=nk) for sign in (1, -1)
        )
        plus, minus = (_fft_bubble_reference(x, niw, niv, q_grid, beta).mat for x in shifted)
        polarizations.append((plus - minus) * (0.5 / s))

    def fn(comm, rank):
        whole = _window_bubbles(comm, [g, *dgs], niv, niw, q_grid, beta, (0, niw + 1))
        return whole, _window_bubbles(comm, [g, *dgs], niv, niw, q_grid, beta, (0, 1, 3, 4))

    results = _on_ranks(ranks, fn, hostnames)
    assert all(np.array_equal(w, s) for whole, split in results for w, s in zip(whole, split))
    for i, polarization in enumerate(polarizations):
        assert np.allclose(np.concatenate([whole[i + 1] for whole, _ in results]), polarization, rtol=0, atol=1e-12)


def test_window_bubble_response_in_complex64_stays_within_ten_times_the_floor_of_the_polarization_pair(monkeypatch):
    """The c64 bilinear bubble differs from the c64 polarization pair by at most 10x that pair's distance to c128."""
    nk, nb, niv, niw, beta = (4, 4, 1), 2, 3, 3, 2.5
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    results = {}
    for dtype in (np.complex128, np.complex64):
        monkeypatch.setattr(n_point_base, "DTYPE", dtype)
        g, dg = (_make_momentum_g(nk, nb, niv + niw + 2, seed=seed) for seed in (52, 53))
        s = float(np.linalg.norm(g.mat) / np.linalg.norm(dg.mat))
        shifted = (
            GreensFunction(g.mat + sign * s * dg.mat, has_compressed_q_dimension=False, nk=nk) for sign in (1, -1)
        )
        plus, minus = (_fft_bubble_reference(x, niw, niv, q_grid, beta).mat for x in shifted)
        response = _window_bubbles(create_comm_mock(), [g, dg], niv, niw, q_grid, beta, (0, 2, niw + 1))[1]
        results[dtype] = ((plus - minus) * (0.5 / s), response)
    (polarization128, _), (polarization64, response64) = results[np.complex128], results[np.complex64]
    floor = np.abs(polarization64 - polarization128).max()
    assert np.allclose(response64, polarization64, rtol=0, atol=10 * floor)


def test_window_bubble_gives_a_rank_without_momenta_empty_bubbles(monkeypatch):
    """With more ranks than irreducible momenta the empty rank gets empty bubbles and the others the rank-0 bits."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, nb, niv, niw, beta = (2, 2, 1), 2, 2, 1, 2.0
    g, dg = (_make_momentum_g(nk, nb, niv + niw + 1, seed=seed) for seed in (43, 44))
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    ref = _fft_bubble_reference(g, niw, niv, q_grid, beta).mat

    def fn(comm, rank):
        forward, response = _window_bubbles(comm, [g, dg], niv, niw, q_grid, beta, (0, 1, 2))
        return forward, response.shape

    results = run_parallel(4, fn)[1]
    assert results[0][0].shape == (0, nb, nb, nb, nb, niw + 1, 2 * niv) == results[0][1]
    assert np.array_equal(np.concatenate([forward for forward, _ in results], axis=0), ref)


@pytest.mark.parametrize("w_window", [(0, 3), (1, 1)])
def test_window_bubble_refuses_an_empty_window_or_one_beyond_the_rspace_frequencies(w_window):
    """A bosonic window that is empty or whose shifts leave the real-space window raises instead of wrapping around."""
    nk, nb, niv, niw, beta = (2, 2, 1), 1, 2, 1, 2.0
    g = _make_momentum_g(nk, nb, niv + niw + 1, seed=45)
    q_grid = bz.KGrid(nk, symmetries=[])
    dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=create_comm_mock(), name="Q")
    g_r, _ = nonlocal_sde._build_rspace_giwk_window(g, create_comm_mock(), niv_window=niv + niw)
    with pytest.raises(ValueError):
        BubbleGenerator.create_generalized_chi0_q_fft_window(dist, g_r, niv, q_grid, beta, w_window)


def test_window_bubble_peaks_at_its_bubbles_and_one_column_round_without_copying_a_window():
    """On one rank the window bubble peaks at its bubbles and one round of columns, far below a window."""
    nk, nb, niv, niw, beta = (8, 8, 1), 2, 4, 3, 2.5
    g, dg = (_make_momentum_g(nk, nb, 402, seed=seed) for seed in (46, 47))
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=create_comm_mock(), name="Q")
    g_r, dg_r = (nonlocal_sde._build_rspace_giwk_window(x, create_comm_mock(), niv_window=400)[0] for x in (g, dg))

    def run():
        return BubbleGenerator.create_generalized_chi0_q_fft_window(dist, g_r, niv, q_grid, beta, (0, niw + 1), [dg_r])

    run()
    bubbles, peak = traced_peak(run)
    column, ufunc_buffers = q_grid.nk_tot * nb**4 * g_r.itemsize, 3 * np.getbufsize() * g_r.itemsize
    model = sum(bubble.mat.nbytes for bubble in bubbles) + 2 * niv * column * (1 + 4 * q_grid.nk_irr / q_grid.nk_tot)
    assert peak <= model + ufunc_buffers + 4 * q_grid.nk_tot * nb**2 * g_r.itemsize < g_r.nbytes / 2


@pytest.mark.parametrize("ranks, hostnames", [(1, None), (3, ["n0", "n0", "n1"])])
def test_window_bubble_without_the_bubble_of_g_returns_the_bits_of_its_responses(ranks, hostnames, monkeypatch):
    """Without the bubble of G the pass returns only the bilinear bubbles, with the bits they have beside it."""
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, nb, niv, niw, beta = (4, 4, 1), 2, 3, 3, 2.5
    greens = [_make_momentum_g(nk, nb, niv + niw + 2, seed=seed) for seed in (60, 61, 62)]
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())

    def fn(comm, rank):
        node_comm = comm.Split_type(FAKE_MPI.COMM_TYPE_SHARED)
        dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=comm, name="Q")
        built = [nonlocal_sde._build_rspace_giwk_window(g, node_comm, niv_window=niv + niw) for g in greens]
        g_r, dg_rs = built[0][0], [g for g, _ in built[1:]]
        window = BubbleGenerator.create_generalized_chi0_q_fft_window
        both = [b.mat.copy() for b in window(dist, g_r, niv, q_grid, beta, (1, 4), dg_rs)[1:]]
        alone = [b.mat.copy() for b in window(dist, g_r, niv, q_grid, beta, (1, 4), dg_rs, with_forward=False)]
        wins = [win for _, win in built]
        built = g_r = dg_rs = None
        for win in wins:
            nonlocal_sde._free_shared_window(win, node_comm)
        return both, alone

    results = _on_ranks(ranks, fn, hostnames)
    assert all(len(alone) == 2 and all(np.array_equal(a, b) for a, b in zip(both, alone)) for both, alone in results)


@pytest.mark.parametrize("ranks", [1, 2])
def test_window_bubble_adds_the_second_product_of_a_change_in_blocks_of_momenta_with_the_same_bits(ranks, monkeypatch):
    """The bilinear bubble written in blocks of a few momenta has the bits of the bubble written in one block."""
    import dgamore.bubble_gen as bubble_gen

    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    nk, nb, niv, niw, beta = (4, 4, 1), 2, 3, 3, 2.5
    greens = [_make_momentum_g(nk, nb, niv + niw + 2, seed=seed) for seed in (63, 64)]
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())

    def fn(comm, rank):
        return _window_bubbles(comm, greens, niv, niw, q_grid, beta, (0, niw + 1))[1]

    whole = _on_ranks(ranks, fn)
    monkeypatch.setattr(bubble_gen, "ROUND_BUFFER_BYTES", 3 * nb**4 * np.dtype(n_point_base.DTYPE).itemsize)
    assert all(np.array_equal(a, b) for a, b in zip(_on_ranks(ranks, fn), whole))


def test_window_bubble_on_one_rank_transforms_all_fermionic_columns_of_a_frequency_at_once(monkeypatch):
    """A single rank skips the column pass and runs one transform per bosonic frequency and bubble."""
    import scipy as sp

    nk, nb, niv, niw, beta = (4, 4, 1), 2, 3, 3, 2.5
    g, dg = (_make_momentum_g(nk, nb, niv + niw + 2, seed=seed) for seed in (65, 66))
    q_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
    dist = MpiDistributor.create_distributor(ntasks=q_grid.nk_irr, comm=create_comm_mock(), name="Q")
    g_r, dg_r = (
        nonlocal_sde._build_rspace_giwk_window(x, create_comm_mock(), niv_window=niv + niw)[0] for x in (g, dg)
    )
    shapes, real = [], sp.fft.ifftn

    def column_pass(*args):
        raise AssertionError("the column pass runs on one rank")

    monkeypatch.setattr(BubbleGenerator, "_run_columns", staticmethod(column_pass))
    monkeypatch.setattr(sp.fft, "ifftn", lambda x, **kwargs: shapes.append(x.shape) or real(x, **kwargs))
    BubbleGenerator.create_generalized_chi0_q_fft_window(dist, g_r, niv, q_grid, beta, (0, niw + 1), [dg_r])
    assert shapes == [(*nk, nb, nb, nb, nb, 2 * niv)] * 2 * (niw + 1)

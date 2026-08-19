# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Pure, side-effect-free estimator of the peak host-memory of the memory-sensitive DGAmore operations. Every step runs a single
code path (the Eliashberg solver alone falls back from its in-memory to its block-distributed grid variant when a
sector does not fit on one rank); this module estimates the peak bytes of the dominant arrays so the driver can
verify upfront that a run fits its nodes. Apart from the global storage precision :data:`dgamore.n_point_base.DTYPE` (the single
source of truth for the per-element size), it pulls in no run-state from the package -- no MPI, no ``psutil``, no
config singleton: every input is passed as an argument, which keeps the formulas unit-testable in isolation.

All heavy quantities are backed by a single :data:`~dgamore.n_point_base.DTYPE` array, and q-points are distributed
across MPI ranks, so per-rank arrays scale with the per-rank q-count rather than the total. Only the dominant large
arrays of each branch are modeled; a single global ``OVERHEAD_FACTOR`` scales every estimate to absorb un-modeled
transients (known un-modeled cost: the mixing history of ``apply_mixing_strategy``, dominated by the modeled branch
peaks). It defaults to ``1.0`` (no extra margin); the residual headroom for OS/allocator overhead lives in the
driver's node-memory fraction (``NODE_MEMORY_FRACTION`` in :mod:`dgamore.DGAmore`), so the two margins do not compound.

Each branch carries its **own** persistent per-rank ``baseline`` (the full-grid two-point objects resident at that
branch's peak) and the portion of it (``giwk_shareable``) that the node-shared giwk window deduplicates
to a single copy per node; the node-total assembly lives in the driver
(:func:`dgamore.DGAmore.autodetect_memory_settings`).
"""

from dataclasses import dataclass

import numpy as np

from dgamore.n_point_base import DTYPE

# Bytes per stored element (from the global DTYPE, so it tracks a switch to e.g. complex128).
DTYPE_BYTES: int = np.dtype(DTYPE).itemsize
OVERHEAD_FACTOR: float = 1.0

# chiq_aux chunk transient: the assembled BSE window, its compound transpose copy inside the per-slice LU solve and
# the back-substituted columns are alive together at the chunk peak.
CHIQ_AUX_CHUNK_FACTOR: int = 3

# fq: single-block BSE assembly + eagerly rebound matmuls (f = gchi0_q_inv @ f, then f @ gchi0_q_inv), ~2 blocks live.
FQ_MATMUL_FACTOR: int = 2

# FFT SDE per-chunk transient: the exchanged full-BZ bosonic window plus the peer-to-peer exchange's in-flight
# send and receive staging copies of it.
SDE_CHUNK_FACTOR: int = 3

# scipy.fft.ifftn(overwrite_x=True) transforms the c64 full-grid bubble in place, so no ifftn transient is allocated
# beyond the multiply buffer (numpy.fft would allocate ~2x: returned array + work arrays). Per-iw peak = multiply buffer.
CHI0Q_IFFTN_TRANSIENT_FACTOR: int = 0

# In-memory Eliashberg solver holds the direct vertex + its matmul-layout copy at the layout-build peak (the flipped
# vertex is never stored - the matvec reuses the direct array via gap-sized index shuffles). Peak 2, residency 1.
LANCZOS_VERTEX_FACTOR: int = 2

# Gap-sized matvec temporaries beyond the ARPACK basis (chi0*gap, its flipped copy, the output) in both variants.
ARPACK_EXTRA_VECTORS: int = 3

# Local SDE dominant transient: the chi-tilde shell chain at niv_full (extended inverted bubble carrying the +U, its
# dense compound inversion output, LAPACK workspace) - ~3 niv_full two-fermion blocks beyond the persistent outputs.
LOCAL_SHELL_INVERT_FACTOR: int = 3


# Floor and cap of the per-chunk byte budget of the chunked builds (auxiliary susceptibility and pairing vertex):
# the floor keeps per-chunk Python and dispatch overhead negligible, the cap bounds the transient on fat nodes.
SLICE_CHUNK_BYTES: int = 2**28
MAX_SLICE_CHUNK_BYTES: int = 2**32


def dynamic_chunk_budget(total_bytes: float, node_ranks: int) -> int:
    r"""
    Returns the per-rank chunk byte budget of the chunked builds: an eighth of the rank's fair share of the node's
    total host memory, floored at :data:`SLICE_CHUNK_BYTES` and capped at :data:`MAX_SLICE_CHUNK_BYTES`. The
    transient of a chunked build stays below a few budgets, so the eighth leaves the bulk of the fair share to the
    step's persistent inputs and outputs while large nodes run correspondingly larger, faster chunks. Deriving the
    budget from the total rather than the currently free memory keeps the chunking - and with it the floating-point
    reduction order - reproducible across reruns on the same node layout.

    :param total_bytes: Total host memory of this node.
    :param node_ranks: Number of MPI ranks sharing the node.
    :return: The chunk budget in bytes.
    """
    return max(SLICE_CHUNK_BYTES, min(MAX_SLICE_CHUNK_BYTES, int(total_bytes // (node_ranks * 8))))


def solver_grid_shape(n_ranks: int, n_freq: int) -> tuple[int, int]:
    r"""
    Chooses the ``(rows, cols)`` block grid of the distributed Eliashberg solver: frequency rows :math:`\nu` first
    (up to one row per frequency), then whole-divisor column blocks of the :math:`\nu'` axis, so the column
    partition is always mirror-symmetric (the crossed matvec term reads each block's mirror). A 1x1 grid degenerates
    into the in-memory matvec.

    :param n_ranks: Number of available MPI ranks.
    :param n_freq: Number of fermionic frequencies ``2 niv_pp`` of the pp box.
    :return: ``(rows, cols)`` with ``rows * cols <= n_ranks`` and ``cols`` dividing ``n_freq``.
    """
    rows = min(n_ranks, n_freq)
    budget = min(n_ranks // rows, n_freq)
    cols = max(d for d in range(1, budget + 1) if n_freq % d == 0)
    return rows, cols


def lanczos_solver_bytes(
    n_bands: int, nk_tot: int, nk_irr: int, niv_pp: int, n_eig: int, n_ranks: int, overhead: float = OVERHEAD_FACTOR
) -> tuple[float, float]:
    r"""
    Per-rank host-memory residency of the two Eliashberg solver variants, in bytes: the in-memory solve (one full-BZ
    sector residency - the matmul-layout vertex and its build copy, the pp bubble and the ARPACK Lanczos basis) and
    the block-distributed grid solve (this rank's vertex block plus the bubble transient, basis and gather buffer).
    The single shared formula that both :func:`estimate_peaks` and the solver dispatch consume, so the two can never
    drift apart.

    :param n_bands: Number of bands :math:`B`.
    :param nk_tot: Total number of momentum points (full BZ).
    :param nk_irr: Number of momentum points in the irreducible BZ.
    :param niv_pp: Number of positive fermionic frequencies of the pp (Eliashberg) box.
    :param n_eig: Number of requested eigenpairs (sets the ARPACK basis size).
    :param n_ranks: Number of MPI ranks (sets the solver grid of the distributed variant).
    :param overhead: Safety factor multiplied onto the raw byte counts.
    :return: ``(per_sector_bytes, grid_per_rank_bytes)``.
    """
    scale = DTYPE_BYTES * overhead
    vpp = 2 * niv_pp
    vertex_pp_full = _two_fermion_block(nk_tot, n_bands, 1, vpp)
    chi0_pp_full = _bubble_block(nk_tot, n_bands, 1, vpp)
    ncv = max(2 * n_eig + 1, 20)
    arpack_ws = (ncv + ARPACK_EXTRA_VECTORS) * _giwk_rspace(nk_tot, n_bands, vpp)
    per_sector = LANCZOS_VERTEX_FACTOR * vertex_pp_full + chi0_pp_full + arpack_ws
    if n_ranks == 1:
        per_sector += _two_fermion_block(nk_irr, n_bands, 1, vpp)  # the waiting channel's vertex
    rows, cols = solver_grid_shape(n_ranks, vpp)
    grid_share = (
        LANCZOS_VERTEX_FACTOR * vertex_pp_full / (rows * cols)
        + chi0_pp_full
        + arpack_ws
        + 2 * _giwk_rspace(nk_tot, n_bands, vpp)
    )
    return scale * per_sector, scale * grid_share


@dataclass(frozen=True)
class BranchPeak:
    """
    Per-rank memory description of one memory-sensitive branch, split into the persistent baseline live at this
    branch's peak and the transient peaks of the fast (``off``) and lean (``on``) code paths, each split by how the
    transient is distributed across the MPI ranks of a node.

    For the node-total budget the memory on a node with ``r`` ranks at this branch's peak is
    ``r * (baseline + distributed) + single``: a *distributed* transient is held by every rank simultaneously (so it
    scales with ``r``), while a *single-rank* transient is built on one rank while the others idle (so it is counted
    once). The driver counts ``giwk_shareable`` once per node
    instead of once per rank (subtracting ``(r - 1) * giwk_shareable`` from the node total).

    :ivar baseline: Per-rank persistent bytes (full-grid two-point objects) live at this branch's peak.
    :ivar giwk_shareable: The ``giwk_full`` portion of ``baseline`` that the node-shared window deduplicates to one
        copy per node (0 for the Eliashberg branches, whose ``giwk_dga`` is a private per-rank object).
    :ivar off_distributed: Per-rank transient bytes held by every rank in the fast (flag-off) path.
    :ivar off_single: Transient bytes held by a single rank in the fast (flag-off) path.
    :ivar on_distributed: Per-rank transient bytes held by every rank in the lean (flag-on) path.
    :ivar on_single: Transient bytes held by a single rank in the lean (flag-on) path.
    """

    baseline: float
    giwk_shareable: float
    off_distributed: float
    off_single: float
    on_distributed: float
    on_single: float


def _ceil_div(a: int, b: int) -> int:
    """
    Returns the ceiling of ``a / b`` for non-negative integers (per-rank task counts).

    :param a: Numerator (e.g. the total number of q-points).
    :param b: Denominator (e.g. the number of MPI ranks).
    :return: ``ceil(a / b)`` as an int (at least 1 if ``a > 0``).
    """
    return -(-a // b)


def _two_fermion_block(q: int, nb: int, nw: int, nv: int, nv_second: int = -1) -> int:
    """
    Returns the element count of a two-fermion four-point block ``[q, nb^4, nw, nv, nv_second]``.

    :param q: Number of (rank-local) momentum points.
    :param nb: Number of bands.
    :param nw: Number of bosonic frequencies.
    :param nv: Number of fermionic frequencies of the first axis (single axis length).
    :param nv_second: Length of the second fermionic axis; defaults to ``nv``, i.e. a symmetric block.
    :return: The number of complex elements.
    """
    return q * nb**4 * nw * nv * (nv if nv_second < 0 else nv_second)


def _bubble_block(q: int, nb: int, nw: int, nv: int) -> int:
    """
    Returns the element count of a bubble / kernel block with a single fermionic axis ``[q, nb^4, nw, nv]``.

    :param q: Number of (rank-local) momentum points.
    :param nb: Number of bands.
    :param nw: Number of bosonic frequencies.
    :param nv: Number of fermionic frequencies (single axis length).
    :return: The number of complex elements.
    """
    return q * nb**4 * nw * nv


def _giwk_rspace(nk_tot: int, nb: int, nv: int) -> int:
    """
    Returns the element count of a momentum-space Green's function replicated over the full grid ``[nk_tot, nb^2, nv]``
    (the FFT paths and the persistent baselines hold such replicated buffers).

    :param nk_tot: Total number of momentum points (full BZ).
    :param nb: Number of bands.
    :param nv: Number of fermionic frequencies (single axis length).
    :return: The number of complex elements.
    """
    return nk_tot * nb**2 * nv


def estimate_peaks(
    *,
    n_bands: int,
    nk_tot: int,
    nk_irr: int,
    niw_core: int,
    niv_core: int,
    niv_full: int,
    niv_cut: int,
    niv_dmft: int,
    niv_pp: int,
    n_ranks: int,
    with_eliashberg: bool,
    save_pairing_vertex: bool = False,
    n_eig: int = 1,
    overhead: float = OVERHEAD_FACTOR,
) -> dict[str, BranchPeak]:
    r"""
    Estimates the per-rank peak host-memory (in bytes) of the fast and lean code path of each memory-sensitive
    branch, split by whether each transient is distributed across the ranks of a node or built on a single rank,
    together with the per-rank persistent baseline live at that branch's peak.

    The returned dict maps a branch key to a :class:`BranchPeak`. Every branch is single-path with identical off
    and on slots, except ``"lanczos"``, which carries the in-memory solve in its off slots and the block-distributed
    grid fallback in its on slots. ``"fq"`` and ``"lanczos"`` are present only when ``with_eliashberg`` is True. For a node with ``r`` ranks the memory at a branch's peak is
    ``r * (baseline + distributed) + single``, minus ``(r - 1) * giwk_shareable`` when the node-shared giwk window is
    active (the driver assembles this; see :func:`dgamore.DGAmore.autodetect_memory_settings`).

    The per-branch baselines track the actual giwk window of ``nonlocal_sde.calculate_self_energy_q``: the bubble
    (``chi0q``) runs on the ``niv_cut`` window, after which giwk is cut (and re-shared) to the
    ``niv_core + niw_core`` window for the kernel/SDE section (``chiq_aux``, ``sde``); ``sigma_old`` is cut to the
    core box before the kernel section. The ``sde`` baseline additionally holds the R-space Green's-function copy,
    which is node-shared like giwk itself when the shared window is active (its ``giwk_shareable`` covers both). The
    Eliashberg branches run after the self-consistency loop with ``sigma_dga`` freed on every rank and ``giwk_dga``
    surviving on the bubble-building rank only, so they carry no per-rank baseline (``giwk_shareable = 0``); the
    surviving copy is counted in their single-rank slots.

    :param n_bands: Number of bands :math:`B`.
    :param nk_tot: Total number of momentum points (full BZ).
    :param nk_irr: Number of momentum points in the irreducible BZ.
    :param niw_core: Number of positive bosonic core frequencies.
    :param niv_core: Number of positive fermionic core frequencies.
    :param niv_full: Number of positive fermionic full-region frequencies.
    :param niv_cut: Number of positive fermionic frequencies the full-grid ``giwk_full`` is built at
        (``min(niw_core + niv_full + 10, niv_dmft)`` in :func:`dgamore.nonlocal_sde.calculate_self_energy_q`).
    :param niv_dmft: Number of positive fermionic frequencies of the DMFT input box (the rank-0 occupation and
        energy step of every iteration concatenates the self-energy back to it).
    :param niv_pp: Number of positive fermionic frequencies of the pp (Eliashberg) box.
    :param n_ranks: Number of MPI ranks the q-points are distributed over.
    :param with_eliashberg: Whether the Eliashberg step runs (adds the ``"fq"`` and ``"lanczos"`` branches).
    :param save_pairing_vertex: Whether both pp pairing vertices are gathered on one rank for saving
        (``config.eliashberg.save_pairing_vertex``); a single-rank peak of the ``lanczos`` branch.
    :param n_eig: Number of requested eigenpairs (``config.eliashberg.n_eig``); sets the ARPACK Lanczos basis size
        ``ncv = max(2 * n_eig + 1, 20)`` held per solving rank.
    :param overhead: Global multiplicative factor accounting for un-modeled transient arrays.
    :return: A dict mapping each branch key to its :class:`BranchPeak`.
    """
    nb = n_bands
    wp = niw_core + 1  # half bosonic range, as the heavy objects are constructed
    vc = 2 * niv_core
    vf = 2 * niv_full
    vpp = 2 * niv_pp
    niv_sde = niv_core + niw_core  # giwk window through the kernel/SDE section (post-bubble cut/re-share)

    qi = _ceil_div(nk_irr, n_ranks)  # per-rank irreducible-BZ q-count
    qt = _ceil_div(nk_tot, n_ranks)  # per-rank full-BZ q-count

    scale = DTYPE_BYTES * overhead

    # Persistent per-rank baselines: giwk_full + sigma_old at niv_cut (bubble), cut to niv_core+niw_core / core box for
    # the kernel section; sde adds a node-shared giwk.fft() copy; Eliashberg frees sigma_dga, keeps one giwk_dga.
    giwk_bubble = scale * _giwk_rspace(nk_tot, nb, 2 * niv_cut)
    giwk_sde = scale * _giwk_rspace(nk_tot, nb, 2 * niv_sde)
    baseline_bubble = giwk_bubble + scale * _giwk_rspace(nk_tot, nb, 2 * niv_cut)  # + sigma_old at niv_cut
    baseline_kernel_section = giwk_sde + scale * _giwk_rspace(nk_tot, nb, vc)  # + sigma_old at the core box
    baseline_sde = baseline_kernel_section + giwk_sde  # + the (node-shareable) R-space Green's-function copy
    giwk_dga_single = scale * _giwk_rspace(nk_tot, nb, 2 * niv_cut)  # the single surviving giwk_dga copy

    peaks: dict[str, BranchPeak] = {}

    # Single-path FFT bubble (verify-only): multi-rank splits (w, v) columns across ranks (R-space G node-shared);
    # single-rank builds the whole irr-BZ bubble plus the full-grid multiply buffer per iw (in-place scipy ifftn).
    gf_window_bubble = 2 * (niv_full + niw_core)
    if n_ranks == 1:
        chi0q_baseline = baseline_bubble
        chi0q_shareable = giwk_bubble
        chi0q_off_distributed = 0.0
        chi0q_off_single = scale * (
            _bubble_block(nk_irr, nb, wp, vf)
            + (1 + CHI0Q_IFFTN_TRANSIENT_FACTOR) * _bubble_block(nk_tot, nb, 1, vf)
            + 2 * _giwk_rspace(nk_tot, nb, gf_window_bubble)
            + _giwk_rspace(nk_tot, nb, vf)
        )
    else:
        g_r_windows = scale * 2 * _giwk_rspace(nk_tot, nb, gf_window_bubble)  # node-shared like giwk itself
        chi0q_baseline = baseline_bubble + g_r_windows
        chi0q_shareable = giwk_bubble + g_r_windows
        chi0q_off_distributed = scale * 1.5 * _bubble_block(qi, nb, wp, vf)
        chi0q_off_single = 0.0
    peaks["chi0q"] = BranchPeak(
        baseline=chi0q_baseline,
        giwk_shareable=chi0q_shareable,
        off_distributed=chi0q_off_distributed,
        off_single=chi0q_off_single,
        on_distributed=chi0q_off_distributed,
        on_single=chi0q_off_single,
    )

    # Single-path chunked sum (verify-only): the accumulated 1-fermion result plus the byte-bounded chunk transient
    # (assembled window, its compound transpose copy and the solved columns). One node-shared local vertex resident:
    # f_dc_loc is niv_full x niv_core (summed index on the full box), the surviving one core-box square.
    local_vertex_shared = scale * max(_two_fermion_block(1, nb, wp, vf, vc), _two_fermion_block(1, nb, wp, vc))
    chiq_aux_chunk = min(SLICE_CHUNK_BYTES, DTYPE_BYTES * _two_fermion_block(qi, nb, wp, vc))
    chiq_aux_distributed = scale * _bubble_block(qi, nb, wp, vc) + overhead * CHIQ_AUX_CHUNK_FACTOR * chiq_aux_chunk
    peaks["chiq_aux"] = BranchPeak(
        baseline=baseline_kernel_section + local_vertex_shared,
        giwk_shareable=giwk_sde + local_vertex_shared,
        off_distributed=chiq_aux_distributed,
        off_single=0.0,
        on_distributed=chiq_aux_distributed,
        on_single=0.0,
    )

    # Single flag-less two-pass FFT contraction, walked in bounded w-chunks: the retained irr-BZ kernel plus the
    # chunk-capped exchange transients (floor modeled - the dynamic budget stays below an eighth of the fair share).
    sde_chunk = min(SLICE_CHUNK_BYTES, DTYPE_BYTES * _bubble_block(qt, nb, wp, vc))
    sde_distributed = scale * _bubble_block(qi, nb, wp, vc) + overhead * SDE_CHUNK_FACTOR * sde_chunk
    # rank-0 single: the sigma finalize buffers, or the occupation/energy step's DMFT-box sigma + giwk pair
    # (its concatenation and Dyson-build transients are broadcast-assigned and v-chunked, so only the pair counts)
    sde_single = scale * max(2 * _giwk_rspace(nk_tot, nb, vc), 2 * _giwk_rspace(nk_tot, nb, 2 * niv_dmft))
    peaks["sde"] = BranchPeak(
        baseline=baseline_sde,
        giwk_shareable=2 * giwk_sde,
        off_distributed=sde_distributed,
        off_single=sde_single,
        on_distributed=sde_distributed,
        on_single=sde_single,
    )

    if with_eliashberg:
        # Slice-direct pairing-vertex build: pp accumulator + three loaded one-fermion inputs + the chunk-bounded
        # transient (matmul pair plus the sliced local vertex = FQ_MATMUL_FACTOR + 1 chunks; the floor is modeled -
        # the dynamic budget stays below an eighth of the fair share by construction, see dynamic_chunk_budget).
        fq_chunk = min(SLICE_CHUNK_BYTES, DTYPE_BYTES * _two_fermion_block(qi, nb, wp, vc))
        fq_distributed = (
            scale * (_two_fermion_block(qi, nb, 1, vpp) + 3 * _bubble_block(qi, nb, wp, vc))
            + overhead * (FQ_MATMUL_FACTOR + 1) * fq_chunk
        )
        peaks["fq"] = BranchPeak(
            baseline=0.0,
            giwk_shareable=0.0,
            off_distributed=fq_distributed,
            off_single=giwk_dga_single,
            on_distributed=fq_distributed,
            on_single=giwk_dga_single,
        )

        # In-memory Lanczos solver in the off slots, block-distributed grid fallback in the on slots; both from the
        # one shared formula the solver dispatch reads as well (see lanczos_solver_bytes).
        solver_single, solver_grid = lanczos_solver_bytes(nb, nk_tot, nk_irr, niv_pp, n_eig, n_ranks, overhead)
        pairing_gather = scale * (2 * _two_fermion_block(nk_irr, nb, 1, vpp)) if save_pairing_vertex else 0.0
        peaks["lanczos"] = BranchPeak(
            baseline=0.0,
            giwk_shareable=0.0,
            off_distributed=0.0,
            off_single=max(solver_single, pairing_gather) + giwk_dga_single,
            on_distributed=solver_grid,
            on_single=pairing_gather + giwk_dga_single,
        )

    # Rank-0-serial local SDE (flag-less, verify-only): both channels' outputs (gamma + chi at the core box, full
    # vertex at niv_full) + the two halved inputs + the dominant chi-tilde shell transient at niv_full.
    l_core = _two_fermion_block(1, nb, wp, vc)
    l_full = _two_fermion_block(1, nb, wp, vf)
    local_single = scale * (2 * (2 * l_core + l_full) + 2 * l_core + LOCAL_SHELL_INVERT_FACTOR * l_full)
    peaks["local"] = BranchPeak(
        baseline=0.0,
        giwk_shareable=0.0,
        off_distributed=0.0,
        off_single=local_single,
        on_distributed=0.0,
        on_single=local_single,
    )

    return peaks

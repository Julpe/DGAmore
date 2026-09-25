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

The chunk budgets of the three chunked builds are sized from this estimate: the driver hands
:func:`max_chunk_budget` the fit check of a branch and receives the largest budget that check accepts, and
:func:`estimate_peaks` models each branch at the budget it was handed (:class:`ChunkBudgets`), so the fit check and
the builds agree. All three builds are bit-invariant under their chunking (the self-energy contraction associates
its bosonic sum in fixed blocks of :data:`SDE_W_BLOCK` frequencies folded in rank order, whatever its budget) and fill
the headroom below the currently available memory. Their per-chunk transients are modeled from their actual
temporaries (assembled window, sliced local vertex, per-slice copies, transposed kernel columns).

Each branch carries its **own** persistent per-rank ``baseline`` (the full-grid two-point objects resident at that
branch's peak) and the portion of it (``giwk_shareable``) that the node-shared giwk window deduplicates
to a single copy per node; the node-total assembly lives in the driver
(:func:`dgamore.DGAmore.autodetect_memory_settings`).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dgamore.jacobian_stabilization import TRACKER_PAIRS

from dgamore.n_point_base import DTYPE

# Bytes per stored element (from the global DTYPE, so it tracks a switch to e.g. complex128).
DTYPE_BYTES: int = np.dtype(DTYPE).itemsize
OVERHEAD_FACTOR: float = 1.0

# Per-rank footprint outside the modeled arrays (interpreter, numpy/scipy/MPI libraries and buffers, the DMFT input),
# ~170 MB per rank measured on a single-band run; counted in every branch's baseline.
RANK_BASELINE_BYTES: int = 2**28

# Consecutive bosonic frequencies one task of the column-distributed self-energy contraction sums; the bosonic sum is
# associated in these fixed blocks, so the chunk budget never changes the result (the rank count moves the fold).
SDE_W_BLOCK: int = 4

# Full-BZ kernel columns alive at once in the self-energy contraction: the unfolded column plus the orbital rotation's
# group copies on auto-symmetry grids or the in-place FFT's line buffers (measured up to 1.73 columns at four bands).
SDE_COLUMN_FACTOR: float = 1.75

# Bytes of kernel rows one call of the self-energy contraction reads; numpy reorders its operand into a copy of this
# size, so the contraction runs over the real-space rows in pieces of it (per-row results do not depend on it).
SDE_CONTRACTION_CHUNK_BYTES: int = 32 * 1024**2

# scipy.fft.ifftn(overwrite_x=True) transforms the c64 full-grid bubble in place, so no ifftn transient is allocated
# beyond the multiply buffer (numpy.fft would allocate ~2x: returned array + work arrays). Per-iw peak = multiply buffer.
CHI0Q_IFFTN_TRANSIENT_FACTOR: int = 0

# In-memory Eliashberg solver holds the direct vertex + its matmul-layout copy at the layout-build peak (the flipped
# vertex is never stored - the matvec reuses the direct array via gap-sized index shuffles). Peak 2, residency 1.
LANCZOS_VERTEX_FACTOR: int = 2

# Gap-sized matvec temporaries beyond the ARPACK basis (chi0*gap, its flipped copy, the output) in both variants.
ARPACK_EXTRA_VECTORS: int = 3

# Smallest Lanczos basis handed to the eigensolver (scipy's own floor is 20): ncv = max(2 n_eig + 1, this floor).
LANCZOS_NCV_FLOOR: int = 12

# Eigenpairs the converged-point exact Jacobian asks ARPACK for per target (largest real part, largest modulus) and
# its Arnoldi basis size, at least 2 * EXACT_JACOBIAN_MODES + 1
EXACT_JACOBIAN_MODES: int = 6
EXACT_JACOBIAN_NCV: int = 20

# Gap-sized scratch windows of one team matvec (the gap, its dressed transform, the direct and the crossed term).
TEAM_SCRATCH_VECTORS: int = 4

# Gap-sized temporaries of one team beyond its basis and eigenvectors, summed over the team's ranks: the four-block
# peak of the sector projection plus the Krylov iteration's input copy (measured 4.5 gap vectors).
TEAM_MATVEC_TRANSIENT_VECTORS: int = 5

# Bytes of one column block a node rank expands, transforms and writes at a time while the team solve builds a
# channel's full-BZ vertex window from the node-shared irreducible source.
TEAM_BUILD_CHUNK_BYTES: int = 64 * 1024**2

# Local SDE dominant transient: the chi-tilde shell chain at niv_full (extended inverted bubble carrying the +U, its
# dense compound inversion output, LAPACK workspace) - ~3 niv_full two-fermion blocks beyond the persistent outputs.
LOCAL_SHELL_INVERT_FACTOR: int = 3


# Floor and cap of the per-chunk byte budget of the chunked builds (auxiliary susceptibility and pairing vertex):
# the floor keeps per-chunk Python and dispatch overhead negligible, the cap bounds the transient on fat nodes.
SLICE_CHUNK_BYTES: int = 2**28
MAX_SLICE_CHUNK_BYTES: int = 2**32

# A budget at or above this value walks a build in a single chunk per rank block (upper bound of the budget search).
MAX_CHUNK_BUDGET_BYTES: int = 2**40


@dataclass(frozen=True)
class ChunkBudgets:
    """
    Per-rank chunk byte budgets of the three chunked builds, sized by the driver from the estimate (see
    :func:`max_chunk_budget`) and consumed both by :func:`estimate_peaks` and by the builds themselves.
    Every field defaults to the :data:`SLICE_CHUNK_BYTES` floor.

    :ivar chiq_aux: Budget of the auxiliary-susceptibility sum
        (:func:`dgamore.nonlocal_sde.create_auxiliary_chi_r_q_sum`).
    :ivar sde: Budget of one round of transposed irreducible kernel columns of the self-energy contraction
        (:func:`dgamore.nonlocal_sde._run_column_sde`).
    :ivar fq: Budget of the pairing-vertex build (:func:`dgamore.eliashberg_solver._build_pairing_vertex_pp`).
    """

    chiq_aux: int = SLICE_CHUNK_BYTES
    sde: int = SLICE_CHUNK_BYTES
    fq: int = SLICE_CHUNK_BYTES


def dynamic_chunk_budget(total_bytes: float, node_ranks: int) -> int:
    r"""
    Returns the fair-share per-rank chunk byte budget of a chunked build, used when no estimate-sized budget is
    handed down (see :class:`ChunkBudgets` and :func:`max_chunk_budget`): an eighth of the rank's fair share of
    the node's total host memory, floored at :data:`SLICE_CHUNK_BYTES` and capped at :data:`MAX_SLICE_CHUNK_BYTES`.
    The transient of a chunked build stays below a few budgets, so the eighth leaves the bulk of the fair share to
    the step's persistent inputs and outputs while large nodes run correspondingly larger, faster chunks. Deriving
    the budget from the total rather than the currently free memory keeps the chunking - and with it the
    floating-point reduction order - reproducible across reruns on the same node layout.

    :param total_bytes: Total host memory of this node.
    :param node_ranks: Number of MPI ranks sharing the node.
    :return: The chunk budget in bytes.
    """
    return max(SLICE_CHUNK_BYTES, min(MAX_SLICE_CHUNK_BYTES, int(total_bytes // (node_ranks * 8))))


def max_chunk_budget(fits: Callable[[int], bool], upper: int = MAX_CHUNK_BUDGET_BYTES) -> int:
    r"""
    Returns the largest chunk byte budget in ``[SLICE_CHUNK_BYTES, upper]`` that ``fits`` accepts, found by
    bisection to within 1 MiB; ``fits`` must be monotone (a budget that fits implies every smaller one fits), which
    holds for the node totals of :func:`estimate_peaks` since every modeled chunk grows with its budget. Returns
    ``upper`` outright when it fits (the build then runs one chunk per rank block) and the :data:`SLICE_CHUNK_BYTES`
    floor when nothing fits (the caller's fit check then fails the run).

    :param fits: Predicate telling whether a candidate budget keeps the branch inside the node budget.
    :param upper: Largest budget to consider.
    :return: The chunk budget in bytes.
    """
    if fits(upper):
        return upper
    low, high = SLICE_CHUNK_BYTES, upper  # invariant: high does not fit
    if not fits(low):
        return low
    while high - low > 2**20:
        mid = (low + high) // 2
        if fits(mid):
            low = mid
        else:
            high = mid
    return low


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


def lanczos_ncv(n_eig: int) -> int:
    """
    Returns the Lanczos basis size ``ncv`` for ``n_eig`` requested eigenpairs, ``max(2 n_eig + 1, LANCZOS_NCV_FLOOR)``
    (the solver caps it to the problem size).

    :param n_eig: Number of requested eigenpairs.
    :return: The basis size.
    """
    return max(2 * n_eig + 1, LANCZOS_NCV_FLOOR)


def lanczos_solver_bytes(
    n_bands: int, nk_tot: int, nk_irr: int, niv_pp: int, n_eig: int, n_ranks: int, overhead: float = OVERHEAD_FACTOR
) -> tuple[float, float]:
    r"""
    Per-rank host-memory residency of the two Eliashberg solver variants, in bytes: the in-memory solve (one full-BZ
    sector residency - the matmul-layout vertex and its build copy, the pp bubble and the ARPACK Lanczos basis) and
    the block-distributed grid solve (this rank's vertex block plus the bubble transient, basis and gather buffer).
    The single shared formula that both :func:`estimate_peaks` and the single-rank solve consume, so the two can
    never drift apart; a multi-rank job's team solve is planned per node from :func:`lanczos_team_bytes` instead.

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
    ncv = lanczos_ncv(n_eig)
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


def column_sde_schedule(niw: int, niv: int, n_ranks: int) -> tuple[list, np.ndarray, np.ndarray]:
    r"""
    Returns the task layout of the column-distributed self-energy contraction
    (:func:`dgamore.nonlocal_sde._run_column_sde`). Task ``t = v * n_blocks + b`` sums the bosonic block ``b`` of the
    fermionic frequency index ``v``; the blocks are the runs of :data:`SDE_W_BLOCK` consecutive frequencies
    :math:`\omega = 0, \ldots, n_\omega` followed by the runs of the negative frequencies :math:`\omega = -1, \ldots,
    -n_\omega`, in the order the bosonic sum is associated. The tasks are split into contiguous runs over the ranks, as
    equal as possible, and the owner of ``v``, the rank that reduces its blocks, is the rank holding its first block.
    The layout depends on the box and the rank count only, so every rank derives the same one.

    :param niw: Number of positive bosonic frequencies.
    :param niv: Number of positive fermionic frequencies.
    :param n_ranks: Number of MPI ranks.
    :return: The tuple ``(blocks, bounds, owner)``: the blocks as ``(negative, bosonic frequencies)`` pairs in summation
        order, the task bounds of the ranks (rank ``r`` holds tasks ``bounds[r]`` to ``bounds[r + 1] - 1``) and the
        owner rank of every ``v``.
    """
    w = SDE_W_BLOCK
    blocks = [(False, tuple(range(b, min(niw + 1, b + w)))) for b in range(0, niw + 1, w)]
    blocks += [(True, tuple(range(b, min(niw + 1, b + w)))) for b in range(1, niw + 1, w)]
    n_tasks = niv * len(blocks)
    bounds = np.arange(n_ranks + 1) * n_tasks // n_ranks
    owner = np.searchsorted(bounds, np.arange(niv) * len(blocks), side="right") - 1
    return blocks, bounds, owner


def column_sde_slabs(niw: int, niv: int, n_ranks: int) -> int:
    r"""
    Returns the largest number of real-space self-energy slabs ``[nk_tot, nb, nb]`` one rank holds in the
    column-distributed self-energy contraction (see :func:`column_sde_schedule`): the reduced result of every owned
    frequency, the running sum of the one frequency another rank owns (only the first frequency of a rank's run can
    have begun on another rank), the receive buffers for the other ranks' sums of its owned frequencies, and the partial
    and contraction buffers of the task in flight.

    :param niw: Number of positive bosonic frequencies.
    :param niv: Number of positive fermionic frequencies.
    :param n_ranks: Number of MPI ranks.
    :return: The slab count of the busiest rank.
    """
    blocks, bounds, owner = column_sde_schedule(niw, niv, n_ranks)
    n_b = len(blocks)
    slabs = np.bincount(owner, minlength=n_ranks)
    for rank in np.flatnonzero(np.diff(bounds) > 0):
        slabs[rank] += 2  # the partial and contraction buffers of the task in flight
        first_v = bounds[rank] // n_b
        if owner[first_v] != rank:
            slabs[rank] += 1  # the running sum handed to the owner
            slabs[owner[first_v]] += 1  # the owner's receive buffer for it
    return int(np.max(slabs))


def team_build_columns(nk_tot: int, n_bands: int, npp: int) -> int:
    r"""
    Returns the number of vertex columns one node rank expands, transforms and writes at a time while the team solve
    builds a channel's vertex window: as many as fit :data:`TEAM_BUILD_CHUNK_BYTES` per full-BZ column, at least one.

    :param nk_tot: Total number of momentum points (full BZ).
    :param n_bands: Number of bands :math:`B`.
    :param npp: Number of columns of one fermionic row of the pp vertex (``2 niv_pp``).
    :return: The number of columns per block.
    """
    return max(1, min(npp, TEAM_BUILD_CHUNK_BYTES // max(1, nk_tot * n_bands**4 * DTYPE_BYTES)))


def lanczos_team_bytes(
    n_bands: int,
    nk_tot: int,
    nk_irr: int,
    niv_pp: int,
    n_eig: int,
    n_channels: int,
    n_sectors: int,
    n_node_ranks: int,
    nk_window: int | None = None,
    overhead: float = OVERHEAD_FACTOR,
) -> float:
    r"""
    Node-peak host memory of the team solve of the Eliashberg equation, in bytes, with ``n_channels`` pairing
    vertices resident in node-shared windows of ``nk_window`` momentum points each (the full BZ, or the star
    representatives of the real-space grid) and ``n_sectors`` (channel x parity) sectors solved on the node: the
    windows, the node-shared irreducible-BZ source of the last window next to either the node root's private
    gathered copy of it or the column blocks the node's ranks expand at once (one block of
    :func:`team_build_columns` full-BZ columns per rank, counted twice for the expansion temporary, at most twice
    the full-BZ vertex), the node-shared pp bubble, and per sector the Krylov basis (``ncv`` gap vectors,
    split over the team), the scratch windows of the team matvec, the lead's assembled eigenvectors with their one
    reordering or symmetrization copy (``2 n_eig``) and the block-sized temporaries. Read by the team planner to
    decide whether a node may hold both channels at once, one at a time, or none.

    :param n_bands: Number of bands :math:`B`.
    :param nk_tot: Total number of momentum points (full BZ).
    :param nk_irr: Number of momentum points in the irreducible BZ.
    :param niv_pp: Number of positive fermionic frequencies of the pp (Eliashberg) box.
    :param n_eig: Number of requested eigenpairs (sets the ARPACK basis size).
    :param n_channels: Number of pairing vertices resident on the node at once (1 or 2).
    :param n_sectors: Number of sectors solved on the node while those vertices are resident.
    :param n_node_ranks: Number of MPI ranks on the node (each expands one column block at a time).
    :param nk_window: Number of momentum points each resident window holds; ``None`` means the full BZ.
    :param overhead: Safety factor multiplied onto the raw byte counts.
    :return: The node-peak bytes.
    """
    vpp = 2 * niv_pp
    ncv = lanczos_ncv(n_eig)
    per_sector = (ncv + TEAM_SCRATCH_VECTORS + 2 * n_eig + TEAM_MATVEC_TRANSIENT_VECTORS) * _giwk_rspace(
        nk_tot, n_bands, vpp
    )
    vertex_full = _two_fermion_block(nk_tot, n_bands, 1, vpp)
    vertex_irr = _two_fermion_block(nk_irr, n_bands, 1, vpp)
    block = nk_tot * n_bands**4 * team_build_columns(nk_tot, n_bands, vpp)
    build = min(2 * vertex_full, 2 * n_node_ranks * block)
    return (
        DTYPE_BYTES
        * overhead
        * (
            n_channels * _two_fermion_block(nk_tot if nk_window is None else nk_window, n_bands, 1, vpp)
            + vertex_irr
            + max(vertex_irr, build)
            + _bubble_block(nk_tot, n_bands, 1, vpp)
            + n_sectors * per_sector
        )
    )


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
    :ivar giwk_shareable: The portion of ``baseline`` held in per-node shared-memory windows (the Green's functions,
        the local vertex and the loop self-energy), deduplicated to one copy per node (0 for the Eliashberg branches,
        whose ``giwk_dga`` is a private object).
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


def _chiq_aux_transient(chunk: int, per_q_box: int, one_slice: int, vc: int) -> int:
    """
    Returns the per-rank transient bytes of one chunk of the auxiliary-susceptibility sum: the assembled
    Bethe-Salpeter window, the sliced local vertex (the window's bosonic range, at most one momentum's box), the
    Fortran-ordered copy of the compound slice being LU-factorized and the window's summed output.

    :param chunk: Bytes of the assembled two-fermion window.
    :param per_q_box: Bytes of one momentum's full two-fermion box.
    :param one_slice: Bytes of one ``(q, w)`` compound slice.
    :param vc: Number of fermionic frequencies of the core box (the summed output is the window over ``vc``).
    :return: The transient bytes.
    """
    return chunk + min(chunk, per_q_box) + one_slice + chunk // vc


def _fq_transient(chunk: int, one_slice: int, streaming: bool) -> int:
    """
    Returns the per-rank transient bytes of one chunk of the pairing-vertex build, as measured by RSS: the band build
    holds the sliced local vertex and the assembled Bethe-Salpeter window with its pp-box chain (1.6 windows) plus one
    slice's solve buffers (2 slices); the streamed full vertex inverts the whole window with numpy, which computes a
    complex64 window in complex128 (6.2 windows plus 4.2 slices; a complex128 window needs about 3.7 and 1.4).

    :param chunk: Bytes of the assembled two-fermion window.
    :param one_slice: Bytes of one ``(q, w)`` compound slice.
    :param streaming: Whether the full ladder vertex is streamed to disk (``save_fq``).
    :return: The transient bytes.
    """
    if streaming:
        return int(6.2 * chunk + 4.2 * one_slice)
    return int(1.6 * chunk + 2 * one_slice)


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


def jacobian_tracker_bytes(nk_tot: int, nb: int, nv: int) -> tuple[int, int]:
    """
    Bytes the rank-0 Jacobian tracker holds beyond its mixing history (the triples :func:`estimate_peaks` counts
    with the history), split into what stays resident through the whole loop and the transient of one update.
    Resident, between updates: the complex Ritz vectors of up to three sets (``TRACKER_PAIRS - 1`` columns each: the
    snapshot kept for the spectrum file, a newer uncertified estimate, and the predecessor's columns a carried run
    keeps on its own window until its spectrum is saved, which also serve as its carried set while that is
    installed or pending) and the four tall float64 arrays of the reflector's basis and dual rows and the map's span
    and dual rows (each ``[n_real, k]`` or its transpose with ``k <= TRACKER_PAIRS - 1``). Transient, in the mixing
    step after the mixing temporaries are freed: the secant step's nine ``[n_real, TRACKER_PAIRS - 1]`` float64
    arrays with ``n_real = 2 * nk_tot * nb^2 * nv`` (the two increment stacks, the tall QR factor together with the
    copies its factorization and the triangular solve make, the kept basis and its image, and the complex Ritz
    vectors built from them, a real-to-complex promotion of the basis included). The reflector and the map are
    rebuilt after that step, while the old four tall arrays are still alive, but below its peak.

    :param nk_tot: Total number of momentum points (full BZ).
    :param nb: Number of bands.
    :param nv: Number of fermionic frequencies (single axis length).
    :return: The tuple ``(resident, transient)`` in bytes.
    """
    core = nk_tot * nb**2 * nv
    # three resident Ritz sets: the snapshot kept for the spectrum file, a newer uncertified estimate and the
    # predecessor's re-gridded columns a carried run keeps until its own spectrum is saved
    ritz_vectors = 3 * np.dtype(np.complex128).itemsize * 2 * core * (TRACKER_PAIRS - 1)
    # the reflector's basis and dual rows and the map's span and dual rows, kept between updates
    tall = 4 * np.dtype(np.float64).itemsize * 2 * core * (TRACKER_PAIRS - 1)
    transient = 9 * np.dtype(np.float64).itemsize * 2 * core * (TRACKER_PAIRS - 1)
    return ritz_vectors + tall, transient


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
    save_fq: bool = False,
    n_eig: int = 1,
    mixing_pairs: int = 0,
    niv_interp: int = 0,
    with_jacobian_tracker: bool = False,
    mixing_history_length: int = 0,
    with_exact_jacobian: bool = False,
    overhead: float = OVERHEAD_FACTOR,
    chunk_budgets: ChunkBudgets = ChunkBudgets(),
) -> dict[str, BranchPeak]:
    r"""
    Estimates the per-rank peak host-memory (in bytes) of the fast and lean code path of each memory-sensitive
    branch, split by whether each transient is distributed across the ranks of a node or built on a single rank,
    together with the per-rank persistent baseline live at that branch's peak.

    The returned dict maps a branch key to a :class:`BranchPeak`. Every branch is single-path with identical off
    and on slots, except ``"lanczos"``, which carries the in-memory solve in its off slots and the block-distributed
    grid fallback in its on slots. ``"fq"`` and ``"lanczos"`` are present only when ``with_eliashberg`` is True,
    ``"exact_jacobian"`` only when ``with_exact_jacobian`` is. For a node with ``r`` ranks the memory at a branch's
    peak is ``r * (baseline + distributed) + single``, minus ``(r - 1) * giwk_shareable`` when the node-shared giwk
    window is active (the driver assembles this; see :func:`dgamore.DGAmore.autodetect_memory_settings`).

    The per-branch baselines track the actual giwk window of ``nonlocal_sde.calculate_self_energy_q``: the bubble
    (``chi0q``) runs on the ``niv_cut`` window, after which giwk is cut (and re-shared) to the
    ``niv_core + niw_core`` window for the kernel/SDE section (``chiq_aux``, ``sde``). The loop's self-energy stays
    node-shared at the ``niv_cut`` window through the whole proposal, and rank 0 holds the accelerated-mixing history
    (``mixing_pairs`` core-box (iterate, proposal) pairs) in the single-rank slots of all three branches. The ``sde``
    baseline additionally holds the R-space Green's-function copy, which is node-shared like giwk itself when the
    shared window is active (its ``giwk_shareable`` covers the node-shared arrays). The Eliashberg branches run after
    the self-consistency loop with ``sigma_dga`` freed and ``giwk_dga`` on rank 0 only, so their baseline is the
    per-rank footprint alone (``giwk_shareable = 0``); the surviving copy is counted in their single-rank slots. The
    ``sigma_loop`` branch is the loop's self-energy step after the SDE, which runs on rank 0 alone: the history plus
    the largest of the mixing point (the proposal and the rebuilt previous iterate with the three linear-mixing
    copies, or the accelerated least-squares solve) and the chemical-potential update (the previous iterate and the
    node's new window next to two complex128 Green's-function arrays), which dominate the proposal tail and the
    hand-over; the other node roots hold at most their received array and window then, which the single-rank slot
    covers on every node. The ``sigma_interp`` branch, present when ``niv_interp`` is set, is the final re-gridding:
    rank 0 interpolates the irreducible self-energy and unfolds the result next to the node-shared window, every rank
    re-grids at most its share of the momenta it fits with a pole. Every branch's baseline includes the per-rank
    :data:`RANK_BASELINE_BYTES` and the full-grid non-local interaction every rank keeps for the whole run.

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
    :param save_fq: Whether the full ladder vertex is streamed to disk (``config.eliashberg.save_fq``), which
        inverts every whole Bethe-Salpeter slice instead of its pp band.
    :param n_eig: Number of requested eigenpairs (``config.eliashberg.n_eig``); sets the ARPACK Lanczos basis size
        ``ncv`` of :func:`lanczos_ncv` held per solving rank.
    :param mixing_pairs: Number of (iterate, proposal) pairs rank 0's accelerated-mixing history reaches, i.e.
        ``min(mixing_history_length + 1, max_iter)`` for Pulay or Anderson mixing and 0 for linear mixing; the
        least-squares solve is modeled over ``mixing_pairs - 1`` secant columns.
    :param niv_interp: Number of positive fermionic frequencies of the final self-energy interpolation's target grid
        (``config.self_energy_interpolation.niv_target``), or 0 when the run does not interpolate (no
        ``"sigma_interp"`` branch).
    :param with_jacobian_tracker: Whether the Jacobian tracker runs
        (``config.stabilization.use_jacobian_stabilization``). Its history replaces the pairs of ``mixing_pairs``
        in every single-rank slot: (iterate, raw proposal, reflected proposal) triples of the core window kept for
        ``max(mixing_history_length, TRACKER_PAIRS) + 1`` entries, with linear mixing too. Its resident Ritz sets
        and tall arrays (:func:`jacobian_tracker_bytes`) join the history there, and its secant transient enters the
        ``sigma_loop`` maximum, since the update runs in the mixing step after the mixing temporaries are freed.
    :param mixing_history_length: The accelerated-mixing history length
        (``config.self_consistency.mixing_history_length``).
    :param with_exact_jacobian: Whether the converged-point exact Jacobian is evaluated
        (``config.stabilization.use_exact_jacobian``), which adds the ``"exact_jacobian"`` branch: the node-shared
        windows :class:`~dgamore.sigma_jacobian.ExactJacobian` holds, four resident core one-fermion blocks per rank
        and the largest phase of one product (bubble response, Bethe-Salpeter solve, contraction), and on rank 0 the
        Arnoldi basis and the complex128 eigenvectors on the whole window.
    :param overhead: Global multiplicative factor accounting for un-modeled transient arrays.
    :param chunk_budgets: Chunk byte budgets of the three chunked builds (see :class:`ChunkBudgets` and
        :func:`max_chunk_budget`); each modeled chunk is clamped to at least one slice of its build (a ``(q, w)``
        compound slice, or one task's irreducible kernel columns of the self-energy contraction) and at most the
        build's rank block. A zero budget yields that branch's residents plus a single slice's transient.
    :return: A dict mapping each branch key to its :class:`BranchPeak`.
    """
    nb = n_bands
    wp = niw_core + 1  # half bosonic range, as the heavy objects are constructed
    vc = 2 * niv_core
    vf = 2 * niv_full
    vpp = 2 * niv_pp
    niv_sde = niv_core + niw_core  # giwk window through the kernel/SDE section (post-bubble cut/re-share)

    qi = _ceil_div(nk_irr, n_ranks)  # per-rank irreducible-BZ q-count

    scale = DTYPE_BYTES * overhead

    # Persistent per-rank baselines: giwk_full (niv_cut for the bubble, niv_core+niw_core for the kernel section) next
    # to the node-shared loop Sigma at niv_cut; sde adds a node-shared giwk.fft() copy; Eliashberg keeps one giwk_dga.
    sigma_core = _giwk_rspace(nk_tot, nb, vc)
    sigma_full = _giwk_rspace(nk_tot, nb, 2 * niv_cut)
    giwk_bubble = scale * sigma_full
    giwk_sde = scale * _giwk_rspace(nk_tot, nb, 2 * niv_sde)
    sigma_shared = scale * sigma_full
    baseline_bubble = giwk_bubble + sigma_shared
    baseline_kernel_section = giwk_sde + sigma_shared
    baseline_sde = baseline_kernel_section + giwk_sde  # + the (node-shareable) R-space Green's-function copy
    giwk_dga_single = scale * sigma_full  # the single surviving giwk_dga copy
    # interpreter, libraries and buffers of every rank plus the driver's full-grid non-local interaction
    rank_base = overhead * RANK_BASELINE_BYTES + scale * nk_tot * nb**4
    # rank 0's accelerated-mixing history of core-box arrays, resident through every branch: (iterate, proposal) pairs,
    # or the Jacobian tracker's (iterate, raw proposal, reflected proposal) triples of max(m, TRACKER_PAIRS) + 1 entries
    # together with the tracker's resident Ritz sets and tall arrays
    history_arrays = 3 * (max(mixing_history_length, TRACKER_PAIRS) + 1) if with_jacobian_tracker else 2 * mixing_pairs
    tracker_resident, tracker_transient = jacobian_tracker_bytes(nk_tot, nb, vc) if with_jacobian_tracker else (0, 0)
    mixing_history = scale * history_arrays * sigma_core + overhead * tracker_resident

    peaks: dict[str, BranchPeak] = {}

    # Single-path FFT bubble (verify-only): multi-rank splits (w, v) columns across ranks (R-space G node-shared);
    # single-rank builds the whole irr-BZ bubble plus the full-grid multiply buffer per iw (in-place scipy ifftn).
    gf_window_bubble = 2 * (niv_full + niw_core)
    if n_ranks == 1:
        chi0q_baseline = baseline_bubble
        chi0q_shareable = giwk_bubble + sigma_shared
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
        chi0q_shareable = giwk_bubble + sigma_shared + g_r_windows
        chi0q_off_distributed = scale * 1.5 * _bubble_block(qi, nb, wp, vf)
        chi0q_off_single = 0.0
    peaks["chi0q"] = BranchPeak(
        baseline=chi0q_baseline + rank_base,
        giwk_shareable=chi0q_shareable,
        off_distributed=chi0q_off_distributed,
        off_single=chi0q_off_single + mixing_history,
        on_distributed=chi0q_off_distributed,
        on_single=chi0q_off_single + mixing_history,
    )

    # Chunked sum (verify-only): three resident one-fermion blocks (accumulated sum, kernel accumulator, inverse
    # bubble) + the chunk transient at the driver-sized budget, clamped to [one (q, w) compound slice, rank block].
    one_slice = DTYPE_BYTES * _two_fermion_block(1, nb, 1, vc)
    per_q_box = DTYPE_BYTES * _two_fermion_block(1, nb, wp, vc)
    rank_block = DTYPE_BYTES * _two_fermion_block(qi, nb, wp, vc)
    chiq_aux_chunk = min(max(chunk_budgets.chiq_aux, one_slice), rank_block)
    chiq_aux_distributed = scale * 3 * _bubble_block(qi, nb, wp, vc) + overhead * _chiq_aux_transient(
        chiq_aux_chunk, per_q_box, one_slice, vc
    )
    # One node-shared local vertex resident: f_dc_loc is niv_full x niv_core (summed index on the full box), the
    # surviving one a core-box square.
    local_vertex_shared = scale * max(_two_fermion_block(1, nb, wp, vf, vc), _two_fermion_block(1, nb, wp, vc))
    peaks["chiq_aux"] = BranchPeak(
        baseline=baseline_kernel_section + local_vertex_shared + rank_base,
        giwk_shareable=giwk_sde + sigma_shared + local_vertex_shared,
        off_distributed=chiq_aux_distributed,
        off_single=mixing_history,
        on_distributed=chiq_aux_distributed,
        on_single=mixing_history,
    )

    # Column-distributed contraction: the irr kernel, one round of irr columns (whole tasks) with the ring's send
    # copy, the full-BZ column work with the contraction's operand chunk, and the busiest rank's real-space slabs.
    _, sde_bounds, _ = column_sde_schedule(niw_core, niv_core, n_ranks)
    sde_task = DTYPE_BYTES * _bubble_block(nk_irr, nb, 1, SDE_W_BLOCK)
    sde_round = min(max(1, chunk_budgets.sde // sde_task), int(np.max(np.diff(sde_bounds)))) * sde_task
    sde_column = DTYPE_BYTES * _bubble_block(nk_tot, nb, 1, 1)
    sde_columns = SDE_COLUMN_FACTOR * sde_column + min(SDE_CONTRACTION_CHUNK_BYTES, sde_column)
    sde_slabs = column_sde_slabs(niw_core, niv_core, n_ranks) * DTYPE_BYTES * _giwk_rspace(nk_tot, nb, 1)
    sde_distributed = scale * _bubble_block(qi, nb, wp, vc) + overhead * (
        sde_round * (1 + qi / nk_irr) + sde_columns + sde_slabs
    )
    # rank-0 single: the sigma finalize buffers, or the occupation/energy step's DMFT-box sigma + giwk pair
    # (its concatenation and Dyson-build transients are broadcast-assigned and v-chunked, so only the pair counts)
    sde_single = scale * max(2 * sigma_core, 2 * _giwk_rspace(nk_tot, nb, 2 * niv_dmft)) + mixing_history
    peaks["sde"] = BranchPeak(
        baseline=baseline_sde + rank_base,
        giwk_shareable=2 * giwk_sde + sigma_shared,
        off_distributed=sde_distributed,
        off_single=sde_single,
        on_distributed=sde_distributed,
        on_single=sde_single,
    )

    if with_eliashberg:
        # Slice-direct pairing-vertex build: pp accumulator + the two loaded one-fermion inputs + the measured chunk
        # transient of the band build (or of the streamed full vertex) at the driver-sized budget, slice/block clamped.
        fq_chunk = min(max(chunk_budgets.fq, one_slice), rank_block)
        fq_distributed = scale * (
            _two_fermion_block(qi, nb, 1, vpp) + 2 * _bubble_block(qi, nb, wp, vc)
        ) + overhead * _fq_transient(fq_chunk, one_slice, save_fq)
        peaks["fq"] = BranchPeak(
            baseline=rank_base,
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
            baseline=rank_base,
            giwk_shareable=0.0,
            off_distributed=0.0,
            off_single=max(solver_single, pairing_gather) + giwk_dga_single,
            on_distributed=solver_grid,
            on_single=pairing_gather + giwk_dga_single,
        )

    # Rank-0 loop step (verify-only): the history + two niv_cut Sigmas with 3 linear-mix copies, the accelerated solve
    # (measured: 9 m + 10 complex64 core copies bound Anderson and Pulay in both precisions), the Jacobian tracker's
    # secant step (after the mixing temporaries are freed) or update_mu's G arrays.
    solve_width = mixing_pairs - 1
    mixing_solve = overhead * 8 * sigma_core * (9 * solve_width + 10) if solve_width > 0 else 0.0
    sigma_loop_single = (
        mixing_history
        + scale * 2 * sigma_full
        + max(scale * 3 * sigma_full, mixing_solve, overhead * 16 * 2 * sigma_full, overhead * tracker_transient)
    )
    peaks["sigma_loop"] = BranchPeak(
        baseline=rank_base,
        giwk_shareable=0.0,
        off_distributed=0.0,
        off_single=sigma_loop_single,
        on_distributed=0.0,
        on_single=sigma_loop_single,
    )

    if with_exact_jacobian:
        # Converged-point exact Jacobian (verify-only, see sigma_jacobian.ExactJacobian); the transients of one product
        # follow the phases of its matvec, the bubble on the path the chi0q branch models for this rank count.
        core_block, full_block = _bubble_block(qi, nb, wp, vc), _bubble_block(qi, nb, wp, vf)
        vertices = scale * (_two_fermion_block(1, nb, wp, vf, vc) + 2 * _two_fermion_block(1, nb, wp, vc))
        windows = sigma_shared + 3 * giwk_bubble + giwk_sde + vertices
        bubble_windows = scale * 2 * _giwk_rspace(nk_tot, nb, gf_window_bubble) if n_ranks > 1 else 0.0
        shared_extra = max(bubble_windows, 2 * giwk_sde)
        bubble_phase = scale * full_block + chi0q_off_distributed
        kernel_phase = max(
            scale * 4 * core_block + overhead * _chiq_aux_transient(chiq_aux_chunk, per_q_box, one_slice, vc),
            scale * 7 * core_block,
        )
        sde_phase = scale * core_block + overhead * (sde_round * (1 + qi / nk_irr) + sde_columns + sde_slabs)
        exact_distributed = scale * 4 * core_block + max(bubble_phase, kernel_phase, sde_phase)
        n_sector = 2 * _giwk_rspace(nk_irr, nb, niv_core)
        eigen = overhead * (
            np.dtype(np.float64).itemsize * (EXACT_JACOBIAN_NCV + 1) * n_sector
            + np.dtype(np.complex128).itemsize * 2 * sigma_core * 4 * EXACT_JACOBIAN_MODES
        )
        exact_single = mixing_history + chi0q_off_single + scale * 4 * sigma_core + eigen
        peaks["exact_jacobian"] = BranchPeak(
            baseline=windows + shared_extra + rank_base,
            giwk_shareable=windows + shared_extra,
            off_distributed=exact_distributed,
            off_single=exact_single,
            on_distributed=exact_distributed,
            on_single=exact_single,
        )

    if niv_interp:
        # Final re-gridding (verify-only): rank 0 interpolates the irreducible Sigma (measured: 10 source + 4 target
        # complex128 copies of PCHIP data), then unfolds it; every rank re-grids at most its share for the pole fits.
        irr_source = _giwk_rspace(nk_irr, nb, 2 * niv_cut)
        irr_target = _giwk_rspace(nk_irr, nb, 2 * niv_interp)
        share_source, share_target = _giwk_rspace(qi, nb, 2 * niv_cut), _giwk_rspace(qi, nb, 2 * niv_interp)
        interp_distributed = scale * share_source + overhead * 16 * (10 * share_source + 4 * share_target)
        interp_single = (
            mixing_history
            + scale * (sigma_full + irr_source)
            + max(
                overhead * 16 * (10 * irr_source + 4 * irr_target),
                scale * (irr_target + _giwk_rspace(nk_tot, nb, 2 * niv_interp)),
            )
        )
        peaks["sigma_interp"] = BranchPeak(
            baseline=rank_base,
            giwk_shareable=0.0,
            off_distributed=interp_distributed,
            off_single=interp_single,
            on_distributed=interp_distributed,
            on_single=interp_single,
        )

    # Rank-0-serial local SDE (flag-less, verify-only): both channels' outputs (gamma + chi at the core box, full
    # vertex at niv_full) + the two halved inputs + the dominant chi-tilde shell transient at niv_full.
    l_core = _two_fermion_block(1, nb, wp, vc)
    l_full = _two_fermion_block(1, nb, wp, vf)
    local_single = scale * (2 * (2 * l_core + l_full) + 2 * l_core + LOCAL_SHELL_INVERT_FACTOR * l_full)
    peaks["local"] = BranchPeak(
        baseline=rank_base,
        giwk_shareable=0.0,
        off_distributed=0.0,
        off_single=local_single,
        on_distributed=0.0,
        on_single=local_single,
    )

    return peaks

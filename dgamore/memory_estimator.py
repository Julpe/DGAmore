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

The chunk budgets of the two chunked builds are sized from this estimate: the driver hands
:func:`max_chunk_budget` the fit check of a branch and receives the largest budget that check accepts, and
:func:`estimate_peaks` models each branch at the budget it was handed (:class:`ChunkBudgets`), so the fit check and
the builds agree. Both builds are bit-invariant under their chunking (the self-energy contraction associates its
bosonic sum in fixed blocks of :data:`SDE_W_BLOCK` frequencies folded in rank order, whatever its budget) and fill
the headroom below the currently available memory. Their per-chunk transients are modeled from their actual
temporaries (the pairing vertex's window chain and per-slice solve buffers, transposed kernel columns); the
auxiliary-susceptibility sum takes no budget, its transient is a fixed two compound slices.

Each branch carries its **own** persistent per-rank ``baseline`` (the full-grid two-point objects resident at that
branch's peak) and the portion of it (``giwk_shareable``) that the node-shared giwk window deduplicates
to a single copy per node; :func:`node_total` assembles a node's total from the estimates of its ranks, which the
driver (:func:`dgamore.DGAmore.autodetect_memory_settings`) evaluates per node.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from dgamore.jacobian_stabilization import _CHECK_START, TRACKER_PAIRS

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

# Bytes of full-BZ kernel columns, with their tasks' partial sums, the self-energy contraction unfolds and transforms
# in one call (about one rank's L3 share, so a batch adds at most a few MiB); a column above it runs alone.
SDE_UNFOLD_BATCH_BYTES: int = 4 * 1024**2

# The chunk element budgets of dgamore.greens_function (_MOMENTUM_CHUNK_ELEMENTS of the occupation and in-box
# potential-energy sums, _MODEL_EPOT_CHUNK_ELEMENTS of the Dyson build and the analytic potential-energy tail) and the
# tail's default frequency count (GreensFunction.get_epot's niv_asympt); test-locked to those values.
GF_MOMENTUM_CHUNK_ELEMENTS: int = 2**22
GF_FREQUENCY_CHUNK_ELEMENTS: int = 2**20
EPOT_NIV_ASYMPT: int = 50000

# scipy.fft.ifftn(overwrite_x=True) transforms the c64 full-grid bubble in place, so no ifftn transient is allocated
# beyond the multiply buffer (numpy.fft would allocate ~2x: returned array + work arrays). Per-iw peak = multiply buffer.
CHI0Q_IFFTN_TRANSIENT_FACTOR: int = 0

# In-memory Eliashberg solver holds the direct vertex + its matmul-layout copy at the layout-build peak (the flipped
# vertex is never stored - the matvec reuses the direct array via gap-sized index shuffles). Peak 2, residency 1.
LANCZOS_VERTEX_FACTOR: int = 2

# Gap-sized matvec temporaries beyond the ARPACK basis (chi0*gap, its flipped copy, the output) in both variants, and
# the two of the sector projection's orbital rotation where the inversion has an orbital action.
ARPACK_EXTRA_VECTORS: int = 5

# Smallest Lanczos basis handed to the eigensolver (scipy's own floor is 20): ncv = max(2 n_eig + 1, this floor).
LANCZOS_NCV_FLOOR: int = 12

# Eigenpairs the converged-point exact Jacobian asks ARPACK for per target (largest real part, largest modulus) and
# its Arnoldi basis size (>= 2 * EXACT_JACOBIAN_MODES + 1; wide, so clustered stable modes converge within the cap)
EXACT_JACOBIAN_MODES: int = 6
EXACT_JACOBIAN_NCV: int = 40
# Basis vectors, i.e. products, of one in-loop exact check (sigma_jacobian.certify_subspace)
EXACT_CHECK_BASIS: int = 10
# Widest block of exact-Jacobian products sharing one Bethe-Salpeter factorization pass (ExactJacobian.matvec_block)
EXACT_JACOBIAN_BLOCK: int = 4
# Modeled extra cost of a block's pass that recomputes the three-leg vertices against the per-slice factorization
# (the selector solve beside the factorization, about 2 % at production slices)
EXACT_VRG_RECOMPUTE_COST: float = 0.02
# Bytes of full-grid columns one round of the window bubble fills (bubble_gen.ROUND_BUFFER_BYTES); a column above it
# runs alone
BUBBLE_ROUND_BYTES: int = 4 * 1024**2

# Momentum groups of a channel's three-leg vertex and kernel (bits do not depend on it) and one group's blocks alive
# beside the stage's residents: the in-place vertex's copies (measured 2.0 to 3.6), the kernel's products (3.2 to 3.6).
KERNEL_MOMENTUM_GROUPS: int = 16
KERNEL_GROUP_TRANSIENT: int = 4

# Gap-sized scratch windows of one team matvec (the gap, its dressed transform, the direct and the crossed term).
TEAM_SCRATCH_VECTORS: int = 4

# Gap-sized temporaries of one team beyond its basis and eigenvectors, summed over the team's ranks: the sector
# projection's peak and the Krylov input copy (measured 4.4 gap vectors, 6.0 with an orbital action of the inversion).
TEAM_MATVEC_TRANSIENT_VECTORS: int = 6

# Bytes of one column block a node rank expands, transforms and writes at a time while the team solve builds a
# channel's full-BZ vertex window from the node-shared irreducible source.
TEAM_BUILD_CHUNK_BYTES: int = 64 * 1024**2


# Floor and cap of the per-chunk byte budget of the chunked builds (self-energy contraction and pairing vertex):
# the floor keeps per-chunk Python and dispatch overhead negligible, the cap bounds the transient on fat nodes.
SLICE_CHUNK_BYTES: int = 2**28
MAX_SLICE_CHUNK_BYTES: int = 2**32

# A budget at or above this value walks a build in a single chunk per rank block (upper bound of the budget search).
MAX_CHUNK_BUDGET_BYTES: int = 2**40


@dataclass(frozen=True)
class ChunkBudgets:
    """
    Per-rank chunk byte budgets of the two chunked builds, sized by the driver from the estimate (see
    :func:`max_chunk_budget`) and consumed both by :func:`estimate_peaks` and by the builds themselves.
    The exact Jacobian's fields are sized the same way. Byte budgets default to the :data:`SLICE_CHUNK_BYTES` floor.

    :ivar sde: Budget of one round of transposed irreducible kernel columns of the self-energy contraction
        (:func:`dgamore.nonlocal_sde._run_column_sde`).
    :ivar fq: Budget of the pairing-vertex build (:func:`dgamore.eliashberg_solver._build_pairing_vertex_pp`).
    :ivar exact: Budget of the exact Jacobian's products (:class:`dgamore.sigma_jacobian.ExactJacobian`): one round of
        transposed kernel columns of its self-energy contraction.
    :ivar exact_block: Columns one block product of the exact Jacobian takes at once
        (:meth:`dgamore.sigma_jacobian.ExactJacobian.matvec_block`), at most :data:`EXACT_JACOBIAN_BLOCK`.
    :ivar exact_window: Stored bosonic frequencies of one window of the exact Jacobian's build and products, ``None``
        for all of them in one window.
    :ivar exact_vertices_per_window: Whether the exact Jacobian loads each local vertex's window for the phase that
        reads it instead of holding all three vertices between products.
    :ivar exact_vrg_per_window: Whether the exact Jacobian recomputes both channels' three-leg vertices for each window
        of its Bethe-Salpeter pass, against the factorization that solves the window's responses, instead of holding
        them between products.
    """

    sde: int = SLICE_CHUNK_BYTES
    fq: int = SLICE_CHUNK_BYTES
    exact: int = SLICE_CHUNK_BYTES
    exact_block: int = 1
    exact_window: int | None = None
    exact_vertices_per_window: bool = False
    exact_vrg_per_window: bool = False


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


def exact_jacobian_budgets(
    fits: Callable[[ChunkBudgets], bool], budgets: ChunkBudgets, n_frequencies: int
) -> ChunkBudgets:
    """
    Returns ``budgets`` with the exact Jacobian's five fields set: among the blocks of up to
    :data:`EXACT_JACOBIAN_BLOCK` columns in each residency (vertices held, the local ones loaded per window, the
    three-leg ones recomputed per window, or both) that ``fits`` accepts on windows of one stored bosonic frequency at
    the chunk floor, the one with the cheapest in-loop check, whose :data:`EXACT_CHECK_BASIS` products take
    ``ceil(EXACT_CHECK_BASIS / block)`` passes, each :data:`EXACT_VRG_RECOMPUTE_COST` dearer with recomputed three-leg
    vertices (held local vertices win a tie); then the widest window that fits and the largest chunk budget that fits
    at both (:func:`max_chunk_budget`). When nothing fits, one column on windows of one frequency at the floor with
    both vertex sets per window is returned, and the caller's fit check then disables the exact Jacobian.

    :param fits: Predicate telling whether the exact Jacobian's branch fits every node at the given budgets; it must
        be monotone in the window and the chunk budget.
    :param budgets: The budgets of the loop's chunked builds, returned unchanged in their own fields.
    :param n_frequencies: Number of stored bosonic frequencies of the products, ``niw_core + 1``.
    :return: The budgets with the exact Jacobian's fields set.
    """
    shapes = [
        replace(
            budgets,
            exact=SLICE_CHUNK_BYTES,
            exact_block=width,
            exact_window=1,
            exact_vertices_per_window=vertices_per_window,
            exact_vrg_per_window=vrg_per_window,
        )
        for vrg_per_window in (False, True)
        for vertices_per_window in (False, True)
        for width in range(1, EXACT_JACOBIAN_BLOCK + 1)
    ]
    shapes.sort(
        key=lambda shape: (
            _ceil_div(EXACT_CHECK_BASIS, shape.exact_block)
            * (1 + EXACT_VRG_RECOMPUTE_COST * shape.exact_vrg_per_window),
            shape.exact_vertices_per_window,
        )
    )
    # the dearest shape last: one column with both vertex sets per window
    shape = next((shape for shape in shapes if fits(shape)), shapes[-1])
    low, high = 1, n_frequencies + 1  # invariant: low fits unless nothing does, high does not
    while high - low > 1:
        mid = (low + high) // 2
        low, high = (mid, high) if fits(replace(shape, exact_window=mid)) else (low, mid)
    shape = replace(shape, exact_window=low)
    return replace(shape, exact=max_chunk_budget(lambda exact: fits(replace(shape, exact=exact))))


def solver_grid_shape(n_ranks: int, n_freq: int) -> tuple[int, int]:
    r"""
    Chooses the ``(rows, cols)`` block grid of the distributed Eliashberg solver: frequency rows :math:`\nu` first
    (up to one row per frequency), then whole-divisor column blocks of the :math:`\nu'` axis, so every column block
    has the same width. A 1x1 grid degenerates into the in-memory matvec.

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


def column_sde_schedule(
    niw: int, niv: int, n_ranks: int, by_frequency: bool = False
) -> tuple[list, np.ndarray, np.ndarray]:
    r"""
    Returns the task layout of the column-distributed self-energy contraction
    (:class:`dgamore.nonlocal_sde._ColumnSdeStream`). Task ``t = v * n_blocks + b`` sums the bosonic block ``b`` of the
    fermionic frequency index ``v``; the blocks are the runs of :data:`SDE_W_BLOCK` consecutive frequencies
    :math:`\omega = 0, \ldots, n_\omega` followed by the runs of the negative frequencies :math:`\omega = -1, \ldots,
    -n_\omega`, in the order the bosonic sum is associated. With ``by_frequency`` every block holds one frequency
    instead, in ascending :math:`\lvert\omega\rvert`, the block of :math:`\omega` before that of :math:`-\omega` (no
    second block for :math:`\omega = 0`), so the blocks of the stored frequencies below any bound come first. The
    tasks are split into contiguous runs over the ranks, as equal as possible, and the owner of ``v``, the rank that
    reduces its blocks, is the rank holding its first block. The layout depends on the box and the rank count only, so
    every rank derives the same one.

    :param niw: Number of positive bosonic frequencies.
    :param niv: Number of positive fermionic frequencies.
    :param n_ranks: Number of MPI ranks.
    :param by_frequency: Whether every block holds one bosonic frequency.
    :return: The tuple ``(blocks, bounds, owner)``: the blocks as ``(negative, bosonic frequencies)`` pairs in summation
        order, the task bounds of the ranks (rank ``r`` holds tasks ``bounds[r]`` to ``bounds[r + 1] - 1``) and the
        owner rank of every ``v``.
    """
    if by_frequency:
        blocks = [(negative, (w,)) for w in range(niw + 1) for negative in (False, True) if w > 0 or not negative]
    else:
        w = SDE_W_BLOCK
        blocks = [(False, tuple(range(b, min(niw + 1, b + w)))) for b in range(0, niw + 1, w)]
        blocks += [(True, tuple(range(b, min(niw + 1, b + w)))) for b in range(1, niw + 1, w)]
    n_tasks = niv * len(blocks)
    bounds = np.arange(n_ranks + 1) * n_tasks // n_ranks
    owner = np.searchsorted(bounds, np.arange(niv) * len(blocks), side="right") - 1
    return blocks, bounds, owner


def sde_unfold_batch(column_bytes: int, partial_bytes: int, n_columns: int) -> int:
    r"""
    Returns the number of full-BZ kernel columns the column-distributed self-energy contraction
    (:class:`dgamore.nonlocal_sde._ColumnSdeStream`) unfolds and transforms in one call: as many as fit
    :data:`SDE_UNFOLD_BATCH_BYTES` with the partial sums of their tasks, at least one and at most ``n_columns``.

    :param column_bytes: Bytes of one full-BZ kernel column.
    :param partial_bytes: Bytes of the partial sums of every output of one task.
    :param n_columns: Number of columns a rank contracts in one round.
    :return: The batch size in columns.
    """
    return max(1, min(n_columns, SDE_UNFOLD_BATCH_BYTES // (column_bytes + partial_bytes)))


def column_sde_slabs(
    niw: int, niv: int, n_ranks: int, n_out: int = 1, by_frequency: bool = False, carried: bool = False
) -> int:
    r"""
    Returns the largest number of real-space self-energy slabs ``[nk_tot, nb, nb]`` one rank holds in the
    column-distributed self-energy contraction of ``n_out`` outputs (see :func:`column_sde_schedule`): per output the
    reduced result of every owned frequency and the running sum of the one frequency another rank owns (only the first
    frequency of a rank's run can have begun on another rank), the partial buffer of every output and the contraction
    buffer of the task in flight, and the receive buffers for the other ranks' sums of its owned frequencies (one
    output at a time). With ``carried`` only the results and running sums are counted, which a contraction fed in
    bosonic groups holds from its first group to its end.

    :param niw: Number of positive bosonic frequencies.
    :param niv: Number of positive fermionic frequencies.
    :param n_ranks: Number of MPI ranks.
    :param n_out: Number of outputs.
    :param by_frequency: Whether every block holds one bosonic frequency (see :func:`column_sde_schedule`).
    :param carried: Whether to count only the slabs held across the groups.
    :return: The slab count of the busiest rank.
    """
    blocks, bounds, owner = column_sde_schedule(niw, niv, n_ranks, by_frequency)
    n_b = len(blocks)
    held = np.bincount(owner, minlength=n_ranks)
    work = np.zeros(n_ranks, dtype=int)
    for rank in np.flatnonzero(np.diff(bounds) > 0):
        work[rank] += n_out + 1  # every output's partial and the contraction buffer of the task in flight
        first_v = bounds[rank] // n_b
        if owner[first_v] != rank:
            held[rank] += 1  # the running sum handed to the owner
            work[owner[first_v]] += 1  # the owner's receive buffer for it
    return int(np.max(n_out * held + (0 if carried else work)))


def column_sde_stream_bytes(
    nk_irr: int,
    nk_tot: int,
    n_bands: int,
    niw: int,
    niv: int,
    n_ranks: int,
    q_rank: int,
    n_kernels: int,
    n_out: int,
    chunk_bytes: int,
) -> tuple[float, float]:
    r"""
    Returns the per-rank bytes of the self-energy contraction fed in bosonic groups
    (:class:`dgamore.nonlocal_sde._ColumnSdeStream` with ``by_frequency``) beside the kernels of a group: the slabs
    held from its first group to its end (see :func:`column_sde_slabs`), and the transient of a group or of the
    reduction beside them: one round of one-column tasks of every kernel with one kernel's ring send copy, the full-BZ
    column work of one batch (see :func:`sde_unfold_batch`) with the contraction's operand chunk, every output's
    partial of every task of the batch, the contraction buffer and one output's receive buffers.

    :param nk_irr: Number of momentum points in the irreducible BZ.
    :param nk_tot: Total number of momentum points (full BZ).
    :param n_bands: Number of bands :math:`B`.
    :param niw: Number of positive bosonic frequencies.
    :param niv: Number of positive fermionic frequencies.
    :param n_ranks: Number of MPI ranks.
    :param q_rank: Irreducible momenta of the rank (its rows of the ring's send copy).
    :param n_kernels: Number of kernels of a group.
    :param n_out: Number of outputs.
    :param chunk_bytes: Byte budget of one round of transposed irreducible columns of all kernels.
    :return: The tuple ``(carried_bytes, transient_bytes)``.
    """
    column = DTYPE_BYTES * _bubble_block(nk_irr, n_bands, 1, 1)
    _, bounds, _ = column_sde_schedule(niw, niv, n_ranks, by_frequency=True)
    per_round = min(max(1, chunk_bytes // (n_kernels * column)), int(np.max(np.diff(bounds))))
    full_column = DTYPE_BYTES * _bubble_block(nk_tot, n_bands, 1, 1)
    slab = DTYPE_BYTES * _giwk_rspace(nk_tot, n_bands, 1)
    carried = column_sde_slabs(niw, niv, n_ranks, n_out, by_frequency=True, carried=True)
    batch = sde_unfold_batch(full_column, n_out * slab, per_round)
    work = column_sde_slabs(niw, niv, n_ranks, n_out, by_frequency=True) - carried + (batch - 1) * n_out
    columns = SDE_COLUMN_FACTOR * batch * full_column + min(SDE_CONTRACTION_CHUNK_BYTES, full_column)
    return carried * slab, per_round * column * (n_kernels + q_rank / nk_irr) + columns + work * slab


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
    block: int = 1,
    overhead: float = OVERHEAD_FACTOR,
) -> float:
    r"""
    Node-peak host memory of the team solve of the Eliashberg equation, in bytes, with ``n_channels`` pairing
    vertices resident in node-shared windows of ``nk_window`` momentum points each (the full BZ, or the star
    representatives of the real-space grid) and ``n_sectors`` (channel x parity) sectors solved on the node: the
    windows, the node-shared irreducible-BZ source of the last window next to either the node root's private
    gathered copy of it or the column blocks the node's ranks expand at once (one block of
    :func:`team_build_columns` full-BZ columns per rank, counted twice for the expansion temporary, at most twice
    the full-BZ vertex), the node-shared pp bubble, and per sector the Krylov basis (``ncv + block - 1`` gap vectors,
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
    :param block: Number of starting vectors of the eigensolver's block iteration (1 for the plain one).
    :param overhead: Safety factor multiplied onto the raw byte counts.
    :return: The node-peak bytes.
    """
    vpp = 2 * niv_pp
    ncv = lanczos_ncv(n_eig)
    per_sector = (ncv + block - 1 + TEAM_SCRATCH_VECTORS + 2 * n_eig + TEAM_MATVEC_TRANSIENT_VECTORS) * _giwk_rspace(
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

    For the node-total budget (see :func:`node_total`) every rank of a node holds ``baseline + distributed`` at its
    own task counts: a *distributed* transient is held by every rank simultaneously, while a *single-rank* transient
    is built on rank 0 while the others idle (so it is counted once, on rank 0's node; a node without rank 0 holds
    ``remote_single`` instead). ``giwk_shareable`` is counted once per node instead of once per rank (subtracting
    ``(r - 1) * giwk_shareable`` from the total of an ``r``-rank node).

    :ivar baseline: Per-rank persistent bytes (full-grid two-point objects) live at this branch's peak.
    :ivar giwk_shareable: The portion of ``baseline`` held in per-node shared-memory windows (the Green's functions,
        the local vertex and the loop self-energy), deduplicated to one copy per node (for the Eliashberg branches
        only the local vertex of the pairing-vertex build: ``giwk_dga`` is a private object).
    :ivar off_distributed: Per-rank transient bytes held by every rank in the fast (flag-off) path.
    :ivar off_single: Transient bytes held by a single rank in the fast (flag-off) path.
    :ivar on_distributed: Per-rank transient bytes held by every rank in the lean (flag-on) path.
    :ivar on_single: Transient bytes held by a single rank in the lean (flag-on) path.
    :ivar remote_single: Single-rank bytes of either path on a node without rank 0 (what its node root holds of the
        step); ``None`` counts the single-rank slots on every node.
    """

    baseline: float
    giwk_shareable: float
    off_distributed: float
    off_single: float
    on_distributed: float
    on_single: float
    remote_single: float | None = None


def _ceil_div(a: int, b: int) -> int:
    """
    Returns the ceiling of ``a / b`` for non-negative integers (per-rank task counts).

    :param a: Numerator (e.g. the total number of q-points).
    :param b: Denominator (e.g. the number of MPI ranks).
    :return: ``ceil(a / b)`` as an int (at least 1 if ``a > 0``).
    """
    return -(-a // b)


def rank_tasks(n_tasks: int, n_ranks: int, rank: int) -> int:
    """
    Returns the number of tasks :class:`~dgamore.mpi_utils.MpiDistributor` gives ``rank``: an even split with the
    excess on the highest ranks.

    :param n_tasks: Total number of tasks (e.g. the irreducible q-points).
    :param n_ranks: Number of MPI ranks.
    :param rank: The rank.
    :return: The rank's task count.
    """
    return n_tasks // n_ranks + int(rank >= n_ranks - n_tasks % n_ranks)


def interaction_bytes(n_bands: int, nk_tot: int, overhead: float = OVERHEAD_FACTOR) -> float:
    r"""
    Returns the bytes of the full-grid non-local interaction :math:`V^{\mathbf{q}}` the driver builds on rank 0 and
    rank 0 holds until the end of the Eliashberg solve.

    :param n_bands: Number of bands.
    :param nk_tot: Total number of momentum points (full BZ).
    :param overhead: Global multiplicative factor accounting for un-modeled transient arrays.
    :return: The bytes.
    """
    return DTYPE_BYTES * overhead * nk_tot * n_bands**4


def node_total(rank_peaks: list[tuple[int, BranchPeak]], holds_rank_zero: bool, grid: bool = False) -> float:
    """
    Returns the memory held on one node at a branch's peak: every rank's ``baseline`` and distributed transient at
    the rank's own task counts, rank 0's single-rank transient on the node holding rank 0 (``remote_single`` on any
    other node, when the branch sets it) and the node-shared windows once instead of once per rank.

    :param rank_peaks: The pairs ``(number of the node's ranks, their BranchPeak)``, one per distinct per-rank estimate
        of the branch (the single-rank slots and the shared part do not depend on the rank).
    :param holds_rank_zero: Whether the node holds rank 0.
    :param grid: Whether to count the lean (``on``) path instead of the fast (``off``) one.
    :return: The bytes.
    """
    first = rank_peaks[0][1]
    n_ranks = sum(count for count, _ in rank_peaks)
    total = sum(count * (bp.baseline + (bp.on_distributed if grid else bp.off_distributed)) for count, bp in rank_peaks)
    single = first.on_single if grid else first.off_single
    if not holds_rank_zero and first.remote_single is not None:
        single = first.remote_single
    return total + single - (n_ranks - 1) * first.giwk_shareable


def kernel_momentum_group(n_q: int) -> int:
    """
    Returns the number of momenta per group of a rank's self-energy kernel step (see
    :data:`KERNEL_MOMENTUM_GROUPS`).

    :param n_q: Number of the rank's momenta.
    :return: The group size, at least 1.
    """
    return max(1, _ceil_div(n_q, KERNEL_MOMENTUM_GROUPS))


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


def _chiq_aux_transient(compound: int, n_bands: int) -> int:
    """
    Returns the per-rank transient bytes of the auxiliary-susceptibility sum beyond its summed output: two compound
    slices (the momentum-independent buffer of the current bosonic frequency and the factorized copy), the
    column-sized temporaries alive during the next bosonic frequency's symmetry scan (128 slice columns for the
    absolute values of one 256-column tile, plus four arrays of ``n_bands^2`` columns: the right-hand sides, the
    vertex diagonal and the previous slice's solution and diagonal blocks) and two 256 x 256 tiles of the scan. Up
    to eight bands this also covers the factorization step, whose workspace is 64 columns at the reference LAPACK
    block size. It is the bound with every orbital pair solved: with pairs eliminated the slices span only the
    other pairs.

    :param compound: Side length of a ``(q, w)`` compound slice, ``n_bands^2`` times the fermionic frequencies.
    :param n_bands: Number of bands :math:`B`.
    :return: The transient bytes.
    """
    return DTYPE_BYTES * (2 * compound**2 + (128 + 4 * n_bands**2) * compound + 2 * 256**2)


def _exact_solve_transient(compound: int, n_bands: int, n_rhs: int) -> int:
    """
    Returns the per-rank transient bytes of a Bethe-Salpeter pass of the exact Jacobian's products: the per-slice
    solve of the auxiliary-susceptibility sum (:func:`_chiq_aux_transient`) for ``n_rhs`` systems of given
    right-hand sides, plus ``4 n_rhs - 3`` slice columns of ``n_bands^2`` each (traced up to eight bands): per system
    the joined right-hand sides and their copy from before the join, beside the previous slice's right-hand sides and
    solution, which stay alive while the next slice's are built.

    :param compound: Side length of a ``(q, w)`` compound slice, ``n_bands^2`` times the fermionic frequencies.
    :param n_bands: Number of bands :math:`B`.
    :param n_rhs: Number of systems solved against each factorization, at least one.
    :return: The transient bytes.
    """
    return _chiq_aux_transient(compound, n_bands) + DTYPE_BYTES * (4 * n_rhs - 3) * n_bands**2 * compound


def exact_window_stages(
    qi: int,
    n_bands: int,
    n_w: int,
    niv_core: int,
    niv_full: int,
    block: int,
    vrg_per_window: bool,
    overhead: float = OVERHEAD_FACTOR,
) -> tuple[float, float]:
    r"""
    Returns the per-rank peak bytes of the double-counting split and of the Bethe-Salpeter pass of one window of
    ``n_w`` stored bosonic frequencies of an exact Jacobian product (:class:`dgamore.sigma_jacobian.ExactJacobian`),
    beyond its residents. The split takes the ``block`` bubble responses of the window one after the other: the one in
    work (its contraction beside one momentum's reordered response and product, its core cut beside the contraction
    and the full-box sum, or the attachment's operand copy and product beside its contraction, core and two sums)
    beside the parts of the earlier ones, ``2 + 2 / v_c`` core blocks of the window each, and the later responses. The
    pass holds the parts and, beside the per-slice solve (:func:`_exact_solve_transient`), six core blocks of the
    window, two per column (its right-hand side and solution) and the recomputed three-leg vertex with
    ``vrg_per_window``. A rank without momenta holds nothing.

    :param qi: Number of this rank's irreducible momenta.
    :param n_bands: Number of bands :math:`B`.
    :param n_w: Number of stored bosonic frequencies of the window.
    :param niv_core: Number of positive fermionic frequencies of the core box.
    :param niv_full: Number of positive fermionic frequencies of the full box.
    :param block: Columns of the piece, at least one.
    :param vrg_per_window: Whether the three-leg vertices are recomputed per window.
    :param overhead: Safety factor multiplied onto the raw byte counts.
    :return: The tuple ``(dc_stage, bse_stage)`` in bytes.
    """
    if qi == 0:
        return 0.0, 0.0
    vc, m = 2 * niv_core, max(1, block)
    core, full = _bubble_block(qi, n_bands, n_w, vc), _bubble_block(qi, n_bands, n_w, 2 * niv_full)
    part = (2 + 2 / vc) * core
    one_q = _bubble_block(1, n_bands, n_w, 2 * niv_full + 2 * vc)
    # the response i in work beside the parts of the i before it and the m - 1 - i after it (largest at i = 0 or m - 1)
    dc = max(
        max(
            i * part + (m - i) * full + max(core + one_q, (2 + 1 / vc) * core),
            i * part + (m - 1 - i) * full + (4 + 2 / vc) * core,
        )
        for i in (0, m - 1)
    )
    bse = DTYPE_BYTES * (m * part + (6 + 2 * m + vrg_per_window) * core) + _exact_solve_transient(
        n_bands**2 * vc, n_bands, m
    )
    return DTYPE_BYTES * overhead * dc, overhead * bse


def _fq_transient(chunk: int, one_fermion: int, one_slice: int, streaming: bool) -> int:
    """
    Returns the per-rank transient bytes of one chunk of the pairing-vertex build beyond its inputs, as traced: the
    chain on the chunk's two-fermion window peaks at three windows (the two matmul products and the einsum's operand
    copy, or the vertex and the negative-frequency half of the pp-band write) next to at most six of its one-fermion
    windows (the bubble and three-leg-vertex windows and cuts, the first-frequency sum, the right three-leg vertex).
    The Bethe-Salpeter slices are built and solved one at a time: the band solve holds four compound slices (the
    per-frequency buffer, the solved copy and at most two of the Schur reduction's temporaries), the whole-slice
    inverse of a streamed full vertex nine (beyond the one buffer it is built in, numpy inverts a complex64 slice in
    complex128: the cast input, LAPACK's matrix and right-hand-side copies and the output take two slices each).

    :param chunk: Bytes of the chunk's two-fermion window on the box its chain runs on (the pp box for the band, the
        core box for a streamed full vertex).
    :param one_fermion: Bytes of the chunk's one-fermion window on the core box.
    :param one_slice: Bytes of one ``(q, w)`` compound slice of the core box.
    :param streaming: Whether the full ladder vertex is streamed to disk (``save_fq``).
    :return: The transient bytes.
    """
    return 3 * chunk + 6 * one_fermion + (9 if streaming else 4) * one_slice


def _green_function_transients(n_k: int, nb: int, nv: int) -> tuple[int, int, int]:
    """
    Returns the transient bytes, beyond the Green's function and its self-energy, of the three steps of
    :mod:`dgamore.greens_function` that run over a Green's function of ``n_k`` momenta and ``nv`` fermionic
    frequencies, as traced: the Dyson build of :meth:`~dgamore.greens_function.GreensFunction.get_g_full` (the matrix
    and its inverse of one frequency chunk next to the frequency-independent part on every momentum), the occupation
    sum (3.5 complex128 arrays of one momentum chunk) and the
    potential energy's analytic tail (three complex128 arrays of one frequency chunk of :data:`EPOT_NIV_ASYMPT`
    frequencies).

    :param n_k: Number of momenta the Green's function holds.
    :param nb: Number of bands :math:`B`.
    :param nv: Number of fermionic frequencies.
    :return: The tuple ``(dyson, occupation, tail)`` of transient bytes.
    """
    elements = _giwk_rspace(n_k, nb, nv)
    dyson = 2 * min(max(GF_FREQUENCY_CHUNK_ELEMENTS, n_k * nb**2), elements) + n_k * nb**2
    occupation = 3.5 * min(max(GF_MOMENTUM_CHUNK_ELEMENTS, nb**2 * nv), elements)
    tail = 3 * min(max(GF_FREQUENCY_CHUNK_ELEMENTS, n_k * nb), n_k * nb * 2 * EPOT_NIV_ASYMPT)
    return 16 * dyson, int(16 * occupation), 16 * tail


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


def jacobian_tracker_bytes(
    nk_irr: int, nb: int, nv: int, with_exact_jacobian: bool = False, after_loop: bool = False
) -> tuple[int, int]:
    """
    Bytes the rank-0 Jacobian tracker holds, split into what stays resident through the whole loop and the transient of
    one update. Every vector lives on the weighted symmetry sector of the core window
    (:func:`~dgamore.sigma_jacobian.tracker_sector_maps`), ``n_real = nk_irr * nb^2 * nv`` real entries: the irreducible
    momenta and the positive half of the ``nv`` core frequencies. Every resident is counted at its largest width and all
    at once, with ``k = TRACKER_PAIRS - 1`` the columns of one secant estimate; the transient is the secant step's seven
    ``[n_real, k]`` float64 arrays. Not counted: the rebuild of the reflector and the map after that step (below its
    peak) and the unfold of one correction to the whole core window by the reflection, the per-direction step or an
    accelerated step (about two core windows in the storage precision, measured 2.1 to 2.3, and 2.6 to 3.1 for the
    accelerated step with its sector solve), which the mixing step's ``niv_cut`` copies cover. After the loop only the
    snapshot, the flips kept beside it, the predecessor's columns and the reflector's basis stay, with no transient (see
    :meth:`~dgamore.jacobian_stabilization.JacobianTracker.release_after_loop`).

    :param nk_irr: Number of irreducible momentum points.
    :param nb: Number of bands.
    :param nv: Number of fermionic frequencies of the core window (single axis length).
    :param with_exact_jacobian: Whether the tracker runs the exact checks of ``stabilization.use_exact_jacobian``,
        whose sets and an exact predecessor's columns widen its residents.
    :param after_loop: Whether to count what the tracker keeps after the loop instead of its residents in the loop.
    :return: The tuple ``(resident, transient)`` in bytes.
    """
    n_real = nk_irr * nb**2 * nv
    k = TRACKER_PAIRS - 1
    real_column, complex_column = np.dtype(np.float64).itemsize * n_real, DTYPE_BYTES * n_real
    snapshot = max(k, EXACT_CHECK_BASIS) if with_exact_jacobian else k
    carried = 2 * EXACT_JACOBIAN_MODES if with_exact_jacobian else k
    check_set = EXACT_CHECK_BASIS + k if with_exact_jacobian else 0
    window = 2 * TRACKER_PAIRS * (DTYPE_BYTES // 2) * n_real
    # the newest estimate, the snapshot, the reflected flips kept beside it, an exact check's extended set, and in
    # complex64 the pair of the mode that set p_eff and the pair whose lower bound waits for its second certification
    ritz_vectors = complex_column * (k + snapshot + k + check_set) + 4 * np.dtype(np.complex64).itemsize * n_real
    # the predecessor's columns in the precision of its file, shared by the carried set
    predecessor = DTYPE_BYTES * n_real * carried
    if after_loop:
        return complex_column * (snapshot + k) + predecessor + real_column * max(k, carried, check_set), 0
    # the estimate's basis, span and their dual rows, or an orthogonal set's basis and span with views as dual rows
    tall = 2 * real_column * max(2 * k, carried, check_set)
    # the rung-start check's real start columns and the complex columns of a later check's candidate mode
    checks = _CHECK_START * real_column + 2 * complex_column if with_exact_jacobian else 0
    transient = 7 * real_column * k
    return window + ritz_vectors + predecessor + tall + checks, transient


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
    rank: int | None = None,
    save_pairing_vertex: bool = False,
    save_fq: bool = False,
    n_eig: int = 1,
    mixing_pairs: int = 0,
    niv_interp: int = 0,
    symmetrize_orbitals: bool = False,
    do_spectrum_dmft: bool = False,
    warm_start: bool = False,
    previous_proposal: bool = False,
    with_jacobian_tracker: bool = False,
    with_exact_jacobian: bool = False,
    overhead: float = OVERHEAD_FACTOR,
    chunk_budgets: ChunkBudgets = ChunkBudgets(),
    n_orbitals: int | None = None,
    bare_vertex: bool = False,
) -> dict[str, BranchPeak]:
    r"""
    Estimates the per-rank peak host-memory (in bytes) of the fast and lean code path of each memory-sensitive
    branch, split by whether each transient is distributed across the ranks of a node or built on a single rank,
    together with the per-rank persistent baseline live at that branch's peak.

    The returned dict maps a branch key to a :class:`BranchPeak`. Every branch is single-path with identical off
    and on slots, except ``"lanczos"``, which carries the in-memory solve in its off slots and the block-distributed
    grid fallback in its on slots. ``"fq"`` and ``"lanczos"`` are present only when ``with_eliashberg`` is True,
    ``"exact_jacobian"`` only when ``with_exact_jacobian`` is. The distributed terms take ``rank``'s irreducible and
    full-BZ task counts (:func:`rank_tasks`; the busiest rank's when ``rank`` is None) and depend on ``rank`` through
    nothing else, so ranks with equal counts share one estimate; the self-energy contraction's column schedule (its
    round cap and real-space slabs, in ``sde`` and ``exact_jacobian``) stays the busiest rank's. :func:`node_total`
    assembles a node's total from the estimates of its ranks.

    The per-branch baselines track the actual giwk window of ``nonlocal_sde.calculate_self_energy_q``: the bubble
    (``chi0q``) runs on the ``niv_cut`` window, after which giwk is cut (and re-shared) to the
    ``niv_core + niw_core`` window for the kernel/SDE section (``chiq_aux``, ``sde``). The loop's self-energy stays
    node-shared at the ``niv_cut`` window through the whole proposal, and rank 0 holds the accelerated-mixing history
    (``mixing_pairs`` core-box (iterate, proposal) pairs) in the single-rank slots of all three branches. The ``sde``
    baseline additionally holds the R-space Green's-function copy, which is node-shared like giwk itself when the
    shared window is active (its ``giwk_shareable`` covers the node-shared arrays). The Eliashberg branches run after
    the self-consistency loop with ``sigma_dga`` freed and ``giwk_dga`` on rank 0 only, so their baseline is the
    per-rank footprint, plus for ``fq`` the node-shared local vertex the Bethe-Salpeter slices are built from (its
    ``giwk_shareable``, absent with ``bare_vertex``); the surviving copy is counted in their single-rank slots. The
    ``sigma_loop`` branch is the loop's mixing step after the SDE, which runs on rank 0 alone: the history plus the
    proposal and the rebuilt previous iterate with the three linear-mixing copies or the accelerated least-squares
    solve, which dominate the proposal tail and the hand-over; the other node roots hold at most their received array
    and window then, their ``remote_single``. The ``mu_update`` branch is the
    chemical-potential update after it: every rank holds two complex128 Green's-function arrays on its share of the
    full-BZ momenta, next to rank 0's history and previous iterate and the node's new window (the window alone on a
    node without rank 0). The ``energies`` branch
    is the occupation and energy step after it: every rank holds its share's self-energy and Green's function on the
    DMFT box and the largest of the Dyson, occupation and energy-tail transients of :mod:`dgamore.greens_function`
    (bounded by its chunk sizes), next to the same rank-0 residents. The ``occupation`` branch is the start of the
    non-local routine: every rank's fresh-start occupation on its share of the DMFT box (on a warm start its mu
    search), with ``do_spectrum_dmft`` rank 0's DMFT lattice Green's function of the whole grid on that box (built
    there, and reloaded with its core-box cut after the loop next to ``giwk_dga`` and ``sigma_dga``), and on a warm
    start rank 0's starting iterate extended onto it. The ``sigma_interp``
    branch, present when ``niv_interp`` is set, is the final re-gridding: rank 0 interpolates the irreducible
    self-energy and unfolds the result next to the node-shared window every node holds, every rank re-grids at most
    its share of the momenta it fits with a pole. The ``local`` branch is rank 0's local Schwinger-Dyson step, the
    larger of the second channel's inversions (three core blocks next to the first channel's outputs and its own
    generalized susceptibility) and its full vertex (next to both channels' irreducible vertices and generalized
    susceptibilities and the first full vertex), plus the orbital symmetrization's copy of one full vertex with
    ``symmetrize_orbitals``. Every branch's baseline includes the per-rank :data:`RANK_BASELINE_BYTES`, and every
    single-rank slot the full-grid non-local interaction (:func:`interaction_bytes`) rank 0 holds from the driver's
    build to the end of the Eliashberg solve.

    :param n_bands: Number of bands :math:`B`.
    :param nk_tot: Total number of momentum points (full BZ).
    :param nk_irr: Number of momentum points in the irreducible BZ.
    :param niw_core: Number of positive bosonic core frequencies.
    :param niv_core: Number of positive fermionic core frequencies.
    :param niv_full: Number of positive fermionic full-region frequencies.
    :param niv_cut: Number of positive fermionic frequencies the full-grid ``giwk_full`` is built at
        (``min(niw_core + niv_full + 10, niv_dmft)`` in :func:`dgamore.nonlocal_sde.calculate_self_energy_q`).
    :param niv_dmft: Number of positive fermionic frequencies of the DMFT input box (the fresh-start occupation and the
        occupation and energy step of every iteration run on it).
    :param niv_pp: Number of positive fermionic frequencies of the pp (Eliashberg) box.
    :param n_ranks: Number of MPI ranks the q-points are distributed over.
    :param with_eliashberg: Whether the Eliashberg step runs (adds the ``"fq"`` and ``"lanczos"`` branches).
    :param rank: The rank whose irreducible and full-BZ task counts (see :func:`rank_tasks`) the distributed terms
        take, or ``None`` for the busiest rank's; the column schedule of the self-energy contraction stays the busiest
        rank's.
    :param save_pairing_vertex: Whether both pp pairing vertices are gathered on one rank for saving
        (``config.eliashberg.save_pairing_vertex``); a single-rank peak of the ``lanczos`` branch.
    :param save_fq: Whether the full ladder vertex is streamed to disk (``config.eliashberg.save_fq``), which
        inverts every whole Bethe-Salpeter slice instead of its pp band.
    :param n_eig: Number of requested eigenpairs (``config.eliashberg.n_eig``); sets the ARPACK Lanczos basis size
        ``ncv`` of :func:`lanczos_ncv` held per solving rank.
    :param mixing_pairs: Number of (iterate, proposal) pairs rank 0's accelerated-mixing history reaches, i.e.
        ``min(mixing_history_length + 1, max_iter)`` for Pulay or Anderson mixing and 0 for linear mixing; the
        least-squares solve is modeled over ``mixing_pairs - 1`` secant columns, of the tracker's sector length when
        the tracker runs.
    :param niv_interp: Number of positive fermionic frequencies of the final self-energy interpolation's target grid
        (``config.self_energy_interpolation.niv_target``), or 0 when the run does not interpolate (no
        ``"sigma_interp"`` branch).
    :param symmetrize_orbitals: Whether the local vertices are symmetrized over orbitals
        (``config.dmft.symmetrize_orbitals`` is not empty), which copies one full vertex at a time.
    :param do_spectrum_dmft: Whether the DMFT spectrum is continued (``config.ana_cont.do_spectrum_dmft``), for which
        rank 0 builds the DMFT lattice Green's function of the whole grid on the DMFT box.
    :param warm_start: Whether the loop may start from a previous run's iterate
        (``config.self_consistency.previous_sc_path`` is set), which rank 0 extends onto the DMFT box on the whole
        grid while every rank re-solves mu on its momenta.
    :param previous_proposal: Whether the loop runs more than one iteration (``config.self_consistency.max_iter`` above
        one), so rank 0 keeps the positive core window of the previous proposal for the convergence test; it joins
        the single-rank slots of every branch that holds the mixing history.
    :param with_jacobian_tracker: Whether the tracker runs (``config.stabilization.use_jacobian_stabilization``): its
        residents (:func:`jacobian_tracker_bytes`) join the single-rank slots of the loop's branches (``chi0q``,
        ``chiq_aux``, ``sde``, ``sigma_loop``, ``mu_update``, ``energies``), of ``exact_jacobian`` (its in-loop check)
        and, as an upper bound, of ``sigma_interp``; its reflected proposal and raw residual the ``sigma_loop`` slot and
        its secant transient that slot's maximum.
    :param with_exact_jacobian: Whether the exact Jacobian is evaluated (``config.stabilization.use_exact_jacobian``),
        which adds the ``"exact_jacobian"`` branch: node-shared operator windows, per-rank product phases, on rank 0 a
        product beside the larger of the in-loop check and the end-of-rung ARPACK solve, and on another node's root a
        product's unfolding (two core windows) or its unfolded window change beside one momentum chunk of its change
        of :math:`G` (its ``remote_single``).
    :param overhead: Global multiplicative factor accounting for un-modeled transient arrays.
    :param chunk_budgets: Chunk byte budgets of the two chunked builds (see :class:`ChunkBudgets` and
        :func:`max_chunk_budget`); each modeled chunk is clamped to at least one slice of its build (a ``(q, w)``
        slice of the pairing vertex's window, or one task's irreducible kernel columns of the self-energy contraction)
        and at most the build's rank block. A zero budget yields that branch's residents plus a single slice's
        transient.
        The ``"exact_jacobian"`` branch reads the exact Jacobian's own budgets (see :func:`exact_jacobian_budgets`).
    :param n_orbitals: Number of orbitals of H(k), the size of the Dyson matrix in the chemical-potential search, the
        occupation and the energies (uncorrelated orbitals included); None takes ``n_bands``. With more orbitals than
        bands the exact Jacobian also holds the window of :math:`(G G)_{dd}`.
    :param bare_vertex: Whether a bare-vertex mode runs (``config.misc.bare_vertex``), whose pairing-vertex chunks
        are the RPA vertex of the physical susceptibility: ``"fq"`` then holds no local vertex, one-fermion inputs or
        Bethe-Salpeter slices.
    :return: A dict mapping each branch key to its :class:`BranchPeak`.
    """
    nb = n_bands
    n_orb = nb if n_orbitals is None else n_orbitals  # the Dyson matrix of the occupation, filling and energy steps
    wp = niw_core + 1  # half bosonic range, as the heavy objects are constructed
    vc = 2 * niv_core
    vf = 2 * niv_full
    vpp = 2 * niv_pp
    niv_sde = niv_core + niw_core  # giwk window through the kernel/SDE section (post-bubble cut/re-share)

    # the rank's irreducible-BZ q-count and full-BZ momentum share
    qi = _ceil_div(nk_irr, n_ranks) if rank is None else rank_tasks(nk_irr, n_ranks, rank)
    share_k = _ceil_div(nk_tot, n_ranks) if rank is None else rank_tasks(nk_tot, n_ranks, rank)

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
    # interpreter, libraries and buffers of every rank (rank 0's full-grid interaction joins the single slots below)
    rank_base = overhead * RANK_BASELINE_BYTES
    # rank 0's accelerated-mixing history of core-box (iterate, proposal) pairs, the tracker's real sector vectors
    # when it runs, and the tracker's residents, alive through every loop branch; the history is freed after the loop
    tracker_resident, tracker_transient = (
        jacobian_tracker_bytes(nk_irr, nb, vc, with_exact_jacobian) if with_jacobian_tracker else (0, 0)
    )
    sector_rows = nk_irr * nb**2 * vc
    history_entry = overhead * (DTYPE_BYTES // 2) * sector_rows if with_jacobian_tracker else scale * sigma_core
    mixing_history = 2 * mixing_pairs * history_entry + overhead * tracker_resident
    if previous_proposal:
        # the previous proposal's positive core window, which the convergence test compares the next proposal with
        mixing_history += scale * sigma_core / 2
    # the reflected proposal and the raw residual the stabilized damped step and the guard of an accelerated step read
    tracker_step = scale * 2 * sigma_core if with_jacobian_tracker else 0.0

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
        remote_single=0.0,
    )

    # Kernel section (verify-only), its largest stage: the cut beside the full-box bubble and dc contraction, the dc
    # attachment beside the cut, the aux sum's solve, or the grouped in-place three-leg vertex and kernel.
    core_block = _bubble_block(qi, nb, wp, vc)
    group_block = _bubble_block(kernel_momentum_group(qi), nb, wp, vc)
    chiq_aux_distributed = max(
        scale * (_bubble_block(qi, nb, wp, vf) + 2 * core_block),
        scale * 4 * core_block,
        scale * 3 * core_block + overhead * _chiq_aux_transient(nb * nb * vc, nb),
        scale * (3 * core_block + KERNEL_GROUP_TRANSIENT * group_block),
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
        remote_single=0.0,
    )

    # Column-distributed contraction: the irr kernel, one round of irr columns (whole tasks) with the ring's send
    # copy, an unfold batch of full-BZ columns with the operand chunk and partials, the busiest rank's real-space slabs.
    _, sde_bounds, _ = column_sde_schedule(niw_core, niv_core, n_ranks)
    sde_task = DTYPE_BYTES * _bubble_block(nk_irr, nb, 1, SDE_W_BLOCK)
    sde_tasks = min(max(1, chunk_budgets.sde // sde_task), int(np.max(np.diff(sde_bounds))))
    sde_round = sde_tasks * sde_task
    sde_column = DTYPE_BYTES * _bubble_block(nk_tot, nb, 1, 1)
    sde_slab = DTYPE_BYTES * _giwk_rspace(nk_tot, nb, 1)
    sde_batch = sde_unfold_batch(sde_column, sde_slab, sde_tasks * SDE_W_BLOCK)
    sde_columns = SDE_COLUMN_FACTOR * sde_batch * sde_column + min(SDE_CONTRACTION_CHUNK_BYTES, sde_column)
    sde_slabs = (column_sde_slabs(niw_core, niv_core, n_ranks) + sde_batch - 1) * sde_slab
    sde_distributed = scale * _bubble_block(qi, nb, wp, vc) + overhead * (
        sde_round * (1 + qi / nk_irr) + sde_columns + sde_slabs
    )
    # rank-0 single: the sigma finalize buffers
    sde_single = scale * 2 * sigma_core + mixing_history
    peaks["sde"] = BranchPeak(
        baseline=baseline_sde + rank_base,
        giwk_shareable=2 * giwk_sde + sigma_shared,
        off_distributed=sde_distributed,
        off_single=sde_single,
        on_distributed=sde_distributed,
        on_single=sde_single,
        remote_single=0.0,
    )

    if with_eliashberg:
        # Slice-direct pairing-vertex build: pp accumulator, two loaded one-fermion inputs and their momentum-group
        # copies, chunk transient at the driver-sized budget; the band's window is the pp box, a streamed one the core.
        one_slice = DTYPE_BYTES * _two_fermion_block(1, nb, 1, vc)
        fq_nv, fq_nw = (vc, wp) if save_fq else (vpp, vpp)
        window_slice = DTYPE_BYTES * _two_fermion_block(1, nb, 1, fq_nv)
        fq_chunk = min(max(chunk_budgets.fq, window_slice), qi * fq_nw * window_slice)
        q_group = max(1, fq_chunk // (fq_nw * window_slice))
        if bare_vertex:
            # the RPA chunks read the physical susceptibility alone (and its momentum-group copy): the chunk's window,
            # the pp-band write's cut and negative half, and one pp slice of the latter's transient; no slice solve
            pp_slice = DTYPE_BYTES * _two_fermion_block(1, nb, 1, vpp)
            fq_transient = fq_chunk + 2 * (fq_chunk * vpp**2 // fq_nv**2) + pp_slice
            fq_distributed = (
                scale * (_two_fermion_block(qi, nb, 1, vpp) + _bubble_block(qi + q_group, nb, wp, 1))
                + overhead * fq_transient
            )
            local_vertex_fq = 0.0
        else:
            fq_distributed = scale * (
                _two_fermion_block(qi, nb, 1, vpp) + 2 * _bubble_block(qi + q_group, nb, wp, vc)
            ) + overhead * _fq_transient(fq_chunk, fq_chunk * vc // fq_nv**2, one_slice, save_fq)
            # the local vertex the slices are built from, node-shared (counted once per node)
            local_vertex_fq = scale * _two_fermion_block(1, nb, wp, vc)
        peaks["fq"] = BranchPeak(
            baseline=rank_base + local_vertex_fq,
            giwk_shareable=local_vertex_fq,
            off_distributed=fq_distributed,
            off_single=giwk_dga_single,
            on_distributed=fq_distributed,
            on_single=giwk_dga_single,
            remote_single=0.0,
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
            remote_single=0.0,
        )

    # Rank-0 loop step (verify-only): the history + two niv_cut Sigmas with 3 linear-mix copies or the accelerated
    # solve (measured: 9 m + 10 complex64 core copies bound Anderson and Pulay in both precisions).
    solve_width = mixing_pairs - 1
    # the solve's copies at 4 B per real row: two rows per complex core entry, or the sector rows with the tracker
    solve_rows = sector_rows if with_jacobian_tracker else 2 * sigma_core
    mixing_solve = overhead * 4 * solve_rows * (9 * solve_width + 10) if solve_width > 0 else 0.0
    sigma_loop_single = mixing_history + scale * 2 * sigma_full + max(scale * 3 * sigma_full, mixing_solve)
    # the Jacobian tracker's secant step when it outgrows the mixing step, beside its reflected proposal and residual
    tracker_loop = mixing_history + scale * 2 * sigma_full + overhead * tracker_transient
    sigma_loop_single = max(sigma_loop_single, tracker_loop) + tracker_step
    peaks["sigma_loop"] = BranchPeak(
        baseline=rank_base,
        giwk_shareable=0.0,
        off_distributed=0.0,
        off_single=sigma_loop_single,
        on_distributed=0.0,
        on_single=sigma_loop_single,
        remote_single=2 * sigma_shared,
    )

    # Chemical-potential update (verify-only): every rank's two complex128 Green's-function arrays on its share of the
    # full-BZ momenta, next to rank 0's history, previous iterate and the node's window.
    mu_update_distributed = overhead * 16 * 2 * _giwk_rspace(share_k, n_orb, 2 * niv_cut)
    mu_update_single = mixing_history + scale * 2 * sigma_full
    peaks["mu_update"] = BranchPeak(
        baseline=rank_base,
        giwk_shareable=0.0,
        off_distributed=mu_update_distributed,
        off_single=mu_update_single,
        on_distributed=mu_update_distributed,
        on_single=mu_update_single,
        remote_single=sigma_shared,
    )

    # Occupation and energy step of every iteration (verify-only): every rank's self-energy and Green's function on its
    # momenta of the DMFT box and the largest of the Dyson, occupation and energy-tail transients, next to rank 0's
    # history, previous iterate and the node's window and its whole-grid moment-fit window (top fifth of the box).
    dmft_box = 2 * niv_dmft
    energies_distributed = scale * 2 * _giwk_rspace(share_k, nb, dmft_box) + overhead * max(
        _green_function_transients(share_k, n_orb, dmft_box)
    )
    energies_single = mu_update_single + scale * _giwk_rspace(nk_tot, nb, max(int(0.2 * niv_dmft), 4))
    peaks["energies"] = BranchPeak(
        baseline=rank_base,
        giwk_shareable=0.0,
        off_distributed=energies_distributed,
        off_single=energies_single,
        on_distributed=energies_distributed,
        on_single=energies_single,
        remote_single=sigma_shared,
    )

    # Start of the non-local routine (verify-only): every rank's fresh-start occupation on its momenta of the DMFT box,
    # on a warm start its mu search on its momenta of the niv_cut box instead; on rank 0 with do_spectrum_dmft the
    # DMFT lattice Green's function of the whole grid on the DMFT box with its Dyson transient, and on a warm start
    # the starting iterate extended onto the DMFT box on the whole grid next to its niv_cut copy and the occupation.
    whole_box = _giwk_rspace(nk_tot, nb, dmft_box)
    occupation_distributed = overhead * _green_function_transients(share_k, n_orb, dmft_box)[1]
    if warm_start:
        occupation_distributed = max(occupation_distributed, mu_update_distributed)
    # with do_spectrum_dmft rank 0 also reloads that Green's function after the loop, cut to the core box, next to
    # giwk_dga and sigma_dga
    spectrum_single = max(
        scale * whole_box + overhead * _green_function_transients(nk_tot, n_orb, dmft_box)[0],
        scale * (whole_box + 2 * sigma_full + _giwk_rspace(nk_tot, nb, vc)),
    )
    warm_single = scale * (whole_box + sigma_full) + overhead * _green_function_transients(nk_tot, n_orb, dmft_box)[1]
    occupation_single = max(spectrum_single if do_spectrum_dmft else 0.0, warm_single if warm_start else 0.0)
    peaks["occupation"] = BranchPeak(
        baseline=rank_base,
        giwk_shareable=0.0,
        off_distributed=occupation_distributed,
        off_single=occupation_single,
        on_distributed=occupation_distributed,
        on_single=occupation_single,
        remote_single=0.0,
    )

    if with_exact_jacobian:
        # Converged-point exact Jacobian (verify-only, see sigma_jacobian.ExactJacobian): its residents on this rank's
        # momenta beside the largest stage of one window of stored bosonic frequencies of a product
        m = max(1, chunk_budgets.exact_block)
        n_w = wp if chunk_budgets.exact_window is None else max(1, min(wp, chunk_budgets.exact_window))
        vrg_per_window = chunk_budgets.exact_vrg_per_window
        window_core, window_full = _bubble_block(qi, nb, n_w, vc), _bubble_block(qi, nb, n_w, vf)
        # the kernel and the inverse core bubble, both three-leg vertices unless recomputed, both susceptibilities
        residents = (2 if vrg_per_window else 4) * _bubble_block(qi, nb, wp, vc) + 2 * _bubble_block(qi, nb, wp, 1)
        carried, sde_transient = column_sde_stream_bytes(
            nk_irr, nk_tot, nb, niw_core, niv_core, n_ranks, qi, m + 1, m, chunk_budgets.exact
        )
        # the window's bubbles beside one round of full-grid columns with its parts' irreducible rows, the received rows
        # and the block of a change's second product, or beside the filter of the bubble of G
        column = DTYPE_BYTES * _bubble_block(nk_tot, nb, 1, 1)
        rounds = max(column, BUBBLE_ROUND_BYTES) * (1 + (m + 1) * (2 * nk_irr + qi) / nk_tot)
        rounds += 4 * DTYPE_BYTES * _giwk_rspace(nk_tot, nb, 1) + BUBBLE_ROUND_BYTES
        if n_ranks == 1:
            # one rank transforms every fermionic column of a frequency at once, beside its two gathered partners
            rounds = column * vf + 2 * DTYPE_BYTES * _giwk_rspace(nk_tot, nb, vf)
            rounds += max(BUBBLE_ROUND_BYTES, DTYPE_BYTES * nb**4 * vf)
        bubble = scale * (m + 1) * window_full + overhead * max(rounds, (DTYPE_BYTES + 3) * window_full)
        dc, bse = exact_window_stages(qi, nb, n_w, niv_core, niv_full, m, vrg_per_window, overhead)
        sde = scale * m * window_core + overhead * sde_transient
        exact_distributed = scale * residents + overhead * carried + max(bubble, dc, bse, sde)
        f_dc_vertex, gamma_vertex = _two_fermion_block(1, nb, wp, vf, vc), _two_fermion_block(1, nb, wp, vc)
        # the three local vertices together, or the larger one's window alone when each is loaded per window
        vertices_per_window = chunk_budgets.exact_vertices_per_window
        vertices = scale * (
            max(f_dc_vertex, gamma_vertex) * n_w / wp if vertices_per_window else f_dc_vertex + 2 * gamma_vertex
        )
        vector = np.dtype(np.float64).itemsize * sector_rows
        # a node root holds the block of sector vectors, never wider than a check's basis
        block_vectors = EXACT_CHECK_BASIS * vector
        # the loop Sigma, the loop-box G and change windows, and the real-space windows of G and of a piece's changes
        rspace = scale * _giwk_rspace(nk_tot, nb, gf_window_bubble)
        windows = sigma_shared + 2 * giwk_bubble + vertices + overhead * block_vectors + (m + 1) * rspace
        if n_orb != nb:
            windows += giwk_bubble  # the (G G)_dd window of the mu response with uncorrelated orbitals
        # rank 0's solvers in float64 sector vectors (measured): the in-loop check's basis, products, eigenvectors and
        # four temporaries; the end-of-rung basis with kept products or images, eigenvectors and two start vectors
        check = (4 * EXACT_CHECK_BASIS + 4) * vector
        arpack = (2 * (EXACT_JACOBIAN_NCV + 1) + 3 * EXACT_JACOBIAN_MODES + 11) * vector
        # a product's unfolded window change or its finalize (two core windows each), the real-space u_loc + V(q)
        product = scale * (2 * sigma_core + nk_tot * nb**4)
        # an in-loop check meets the history, the whole tracker and rank 0's private proposal; the end-of-rung solve
        # comes after the loop, which drops the history and what the tracker reads only in the loop
        tracker_after_loop = (
            jacobian_tracker_bytes(nk_irr, nb, vc, True, after_loop=True)[0] if with_jacobian_tracker else 0
        )
        in_loop = mixing_history + scale * sigma_full + overhead * check
        exact_single = product + max(in_loop, overhead * (tracker_after_loop + arpack))
        # another node's root holds a product's unfolding (two core windows) or its unfolded change beside one momentum
        # chunk of its dG build, which never exceeds the loop-box Green's function
        remote = scale * sigma_core + max(
            scale * sigma_core, overhead * min(SLICE_CHUNK_BYTES, DTYPE_BYTES * sigma_full)
        )
        peaks["exact_jacobian"] = BranchPeak(
            baseline=windows + rank_base,
            giwk_shareable=windows,
            off_distributed=exact_distributed,
            off_single=exact_single,
            on_distributed=exact_distributed,
            on_single=exact_single,
            remote_single=remote,
        )

    if niv_interp:
        # Final re-gridding (verify-only): rank 0 interpolates the irreducible Sigma (measured: 10 source + 4 target
        # complex128 copies of PCHIP data), then unfolds it; every rank re-grids at most its share for the pole fits.
        irr_source = _giwk_rspace(nk_irr, nb, 2 * niv_cut)
        irr_target = _giwk_rspace(nk_irr, nb, 2 * niv_interp)
        share_source, share_target = _giwk_rspace(qi, nb, 2 * niv_cut), _giwk_rspace(qi, nb, 2 * niv_interp)
        interp_distributed = scale * share_source + overhead * 16 * (10 * share_source + 4 * share_target)
        interp_single = (
            overhead * tracker_resident
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
            remote_single=sigma_shared,
        )

    # Rank-0-serial local SDE (verify-only): the first channel's gamma, gchi and F (niv_full x niv_core) next to the
    # second's gchi plus three core blocks of its inversions, or next to its gamma, gchi and F (measured)
    l_core = _two_fermion_block(1, nb, wp, vc)
    f_full = _two_fermion_block(1, nb, wp, vf, vc)
    local_single = scale * max(4 * l_core + (3 if symmetrize_orbitals else 2) * f_full, 6 * l_core + f_full)
    peaks["local"] = BranchPeak(
        baseline=rank_base,
        giwk_shareable=0.0,
        off_distributed=0.0,
        off_single=local_single,
        on_distributed=0.0,
        on_single=local_single,
        remote_single=0.0,
    )

    # rank 0 alone holds the driver's full-grid non-local interaction, through every branch
    v_full = interaction_bytes(nb, nk_tot, overhead)
    return {
        key: replace(bp, off_single=bp.off_single + v_full, on_single=bp.on_single + v_full)
        for key, bp in peaks.items()
    }

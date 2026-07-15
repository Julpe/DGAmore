# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Pure, side-effect-free estimator of the peak host-memory of the memory-sensitive DGAmore operations. Each
``save_memory_*`` switch in :class:`dgamore.config.MemoryConfig` selects between a fast (flag off) and a lean
(flag on) code path; this module estimates the peak bytes of the dominant arrays of both paths so the driver can set
the flags automatically. Apart from the global storage precision :data:`dgamore.n_point_base.DTYPE` (the single
source of truth for the per-element size), it pulls in no run-state from the package -- no MPI, no ``psutil``, no
config singleton: every input is passed as an argument, which keeps the formulas unit-testable in isolation.

All heavy quantities are backed by a single :data:`~dgamore.n_point_base.DTYPE` array, and q-points are distributed
across MPI ranks, so per-rank arrays scale with the per-rank q-count rather than the total. Only the dominant large
arrays of each branch are modeled; a single global ``OVERHEAD_FACTOR`` scales every estimate to absorb un-modeled
transients (known un-modeled costs: the mixing history of ``apply_mixing_strategy`` and the pp pairing-vertex assembly
in ``eliashberg_solver.solve``, both dominated by the modeled branch peaks). It defaults to ``1.0`` (no extra margin,
since those costs are dominated by the modeled peaks); the residual headroom for OS/allocator overhead lives in the
driver's node-memory fraction (``NODE_MEMORY_FRACTION`` in :mod:`dgamore.DGAmore`), so the two margins do not compound.

Each branch carries its **own** persistent per-rank ``baseline`` (the full-grid two-point objects resident at that
branch's peak) and the portion of it (``giwk_shareable``) that ``config.memory.use_shared_memory_common_obj`` deduplicates
to a single copy per node; the node-total assembly lives in the driver
(:func:`dgamore.DGAmore.autodetect_memory_settings`).
"""

from dataclasses import dataclass

import numpy as np

from dgamore.n_point_base import DTYPE

# Bytes per stored element (from the global DTYPE, so it tracks a switch to e.g. complex128).
DTYPE_BYTES: int = np.dtype(DTYPE).itemsize
OVERHEAD_FACTOR: float = 1.0

# chiq_aux builds its BSE block in ONE allocation (nonlocal_sde._assemble_bse_matrix), then holds it resident across
# the per-q invert_and_sum_over_last_vn (per-q inversion transient negligible). Peak ~1x the rank-local block.
CHIQ_AUX_INVERT_FACTOR: int = 1

# fq: single-block BSE assembly + eagerly rebound matmuls (f = gchi0_q_inv @ f, then f @ gchi0_q_inv), ~2 blocks live.
FQ_MATMUL_FACTOR: int = 2

# FFT SDE holds the full-BZ niw-half kernel + its half-niv copy through the conj/distributed-FFT round trip, ~2 blocks.
SDE_FFT_KERNEL_FACTOR: int = 2

# numpy.fft.ifftn on the c64 full-grid bubble transiently allocates ~2x the buffer beyond it (returned array + work
# arrays; numpy 2.x, dtype preserved). Per-iw peak = multiply buffer + this transient.
CHI0Q_IFFTN_TRANSIENT_FACTOR: int = 2

# In-memory Eliashberg solver holds the direct vertex + its matmul-layout copy at the layout-build peak (the flipped
# vertex is never stored - the matvec reuses the direct array via gap-sized index shuffles). Peak 2, residency 1.
LANCZOS_VERTEX_FACTOR: int = 2

# Gap-sized matvec temporaries beyond the ARPACK basis (chi0*gap, its flipped copy, the output) in both variants.
ARPACK_EXTRA_VECTORS: int = 3

# Local SDE dominant transient: the chi-tilde shell chain at niv_full (extended inverted bubble carrying the +U, its
# dense compound inversion output, LAPACK workspace) - ~3 niv_full two-fermion blocks beyond the persistent outputs.
LOCAL_SHELL_INVERT_FACTOR: int = 3


@dataclass(frozen=True)
class BranchPeak:
    """
    Per-rank memory description of one memory-sensitive branch, split into the persistent baseline live at this
    branch's peak and the transient peaks of the fast (``off``) and lean (``on``) code paths, each split by how the
    transient is distributed across the MPI ranks of a node.

    For the node-total budget the memory on a node with ``r`` ranks at this branch's peak is
    ``r * (baseline + distributed) + single``: a *distributed* transient is held by every rank simultaneously (so it
    scales with ``r``), while a *single-rank* transient is built on one rank while the others idle (so it is counted
    once). When ``config.memory.use_shared_memory_common_obj`` is on, the driver counts ``giwk_shareable`` once per node
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


def _two_fermion_block(q: int, nb: int, nw: int, nv: int) -> int:
    """
    Returns the element count of a full two-fermion four-point block ``[q, nb^4, nw, nv, nv]``.

    :param q: Number of (rank-local) momentum points.
    :param nb: Number of bands.
    :param nw: Number of bosonic frequencies.
    :param nv: Number of fermionic frequencies (single axis length).
    :return: The number of complex elements.
    """
    return q * nb**4 * nw * nv * nv


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
    niv_pp: int,
    n_ranks: int,
    with_eliashberg: bool,
    save_fq: bool = False,
    save_pairing_vertex: bool = False,
    n_eig: int = 1,
    overhead: float = OVERHEAD_FACTOR,
) -> dict[str, BranchPeak]:
    r"""
    Estimates the per-rank peak host-memory (in bytes) of the fast and lean code path of each memory-sensitive
    branch, split by whether each transient is distributed across the ranks of a node or built on a single rank,
    together with the per-rank persistent baseline live at that branch's peak.

    The returned dict maps a branch key to a :class:`BranchPeak`; the branch keys mirror the ``save_memory_for_*``
    switches plus the flag-less ``"sde"`` step (always the two-pass FFT contraction, so its off and on slots are
    identical and the driver only verifies the fit) and the flag-less ``"local"`` step (the rank-0-serial local
    Schwinger-Dyson pass, also verify-only): ``"chi0q"``, ``"chiq_aux"``, ``"sde"``, ``"local"`` are always present;
    ``"fq"`` and ``"lanczos"`` are added only when ``with_eliashberg`` is True. For a node with ``r`` ranks the memory at a branch's peak is
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
    :param niv_pp: Number of positive fermionic frequencies of the pp (Eliashberg) box.
    :param n_ranks: Number of MPI ranks the q-points are distributed over.
    :param with_eliashberg: Whether the Eliashberg step runs (adds the ``"fq"`` and ``"lanczos"`` branches).
    :param save_fq: Whether the full ladder vertex is kept in the full ph box (``config.eliashberg.save_fq``); when
        True the per-rank ``fq`` accumulator spans the full ``[wn, vc, vc]`` block instead of the small pp box AND the
        whole irreducible-BZ vertex is gathered on one rank for saving (a single-rank peak in both paths).
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

    # Fast (FFT) path: multi-rank splits (w, v) columns (DISTRIBUTED, R-space G node-shared); single-rank builds the
    # whole irr-BZ bubble + B^4 multiply buffer + ~2x ifftn transient per iw. Lean per-q einsum is DISTRIBUTED.
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
        on_distributed=scale * (_bubble_block(qi, nb, wp, vf) + 2 * _giwk_rspace(nk_tot, nb, gf_window_bubble)),
        on_single=0.0,
    )

    # Fast holds the whole rank-local two-fermion block, lean accumulates the 1-fermion result (both DISTRIBUTED; map
    # always p2p, no single-rank term). One node-shared local vertex (f_dc_loc at niv_full or core-box gamma) resident.
    local_vertex_shared = scale * max(_two_fermion_block(1, nb, wp, vf), _two_fermion_block(1, nb, wp, vc))
    peaks["chiq_aux"] = BranchPeak(
        baseline=baseline_kernel_section + local_vertex_shared,
        giwk_shareable=giwk_sde + local_vertex_shared,
        off_distributed=scale * CHIQ_AUX_INVERT_FACTOR * _two_fermion_block(qi, nb, wp, vc),
        off_single=0.0,
        on_distributed=scale
        * (CHIQ_AUX_INVERT_FACTOR * _two_fermion_block(1, nb, wp, vc) + _bubble_block(qi, nb, wp, vc)),
        on_single=0.0,
    )

    # Single flag-less two-pass FFT contraction: full-BZ niw-half kernel + its half-niv copy (~SDE_FFT_KERNEL_FACTOR
    # blocks) + one private giwk.fft() copy/rank; rank 0 finalizes. The old q-loop path is unused (peaked HIGHER).
    sde_distributed = scale * (SDE_FFT_KERNEL_FACTOR * _bubble_block(qt, nb, wp, vc) + _bubble_block(qi, nb, wp, vc))
    sde_single = scale * 2 * _giwk_rspace(nk_tot, nb, vc)
    peaks["sde"] = BranchPeak(
        baseline=baseline_sde,
        giwk_shareable=2 * giwk_sde,
        off_distributed=sde_distributed,
        off_single=sde_single,
        on_distributed=sde_distributed,
        on_single=sde_single,
    )

    if with_eliashberg:
        # ~FQ_MATMUL_FACTOR two-fermion blocks at the matmul peak (per rank fast, per q lean) + 1-fermion inputs + the
        # lean accumulator. save_fq gathers the whole irr-BZ vertex on one rank.
        vc_fq = vc
        fq_accumulator = _two_fermion_block(qi, nb, wp, vc) if save_fq else _two_fermion_block(qi, nb, 1, vpp)
        fq_gather_single = scale * _two_fermion_block(nk_irr, nb, wp, vc) if save_fq else 0.0
        peaks["fq"] = BranchPeak(
            baseline=0.0,
            giwk_shareable=0.0,
            off_distributed=scale
            * (FQ_MATMUL_FACTOR * _two_fermion_block(qi, nb, wp, vc_fq) + _bubble_block(qi, nb, wp, vc_fq)),
            off_single=fq_gather_single + giwk_dga_single,
            on_distributed=scale
            * (
                FQ_MATMUL_FACTOR * _two_fermion_block(1, nb, wp, vc_fq)
                + fq_accumulator
                + 3 * _bubble_block(qi, nb, wp, vc_fq)
            ),
            on_single=fq_gather_single + giwk_dga_single,
        )

        # Fast: whole-BZ pairing vertex on ONE rank (LANCZOS_VERTEX_FACTOR copies + full-BZ pp bubble + ARPACK ws;
        # sing/trip concurrent, driver doubles). Lean: full BZ per rank on its v'-slice, each with its own ARPACK ws.
        vertex_pp_full = _two_fermion_block(nk_tot, nb, 1, vpp)
        chi0_pp_full = _bubble_block(nk_tot, nb, 1, vpp)
        ncv = max(2 * n_eig + 1, 20)
        arpack_ws = (ncv + ARPACK_EXTRA_VECTORS) * _giwk_rspace(nk_tot, nb, vpp)
        pairing_gather = 2 * _two_fermion_block(nk_irr, nb, 1, vpp) if save_pairing_vertex else 0
        v_share = _ceil_div(vpp, n_ranks)  # per-rank share of the v'-distributed vertex (vpp tasks, not q-points)
        vertex_pp_slice = nk_tot * nb**4 * vpp * v_share
        solver_single_fast = LANCZOS_VERTEX_FACTOR * vertex_pp_full + chi0_pp_full + arpack_ws
        if n_ranks == 1:
            solver_single_fast += _two_fermion_block(nk_irr, nb, 1, vpp)  # the waiting channel's gathered vertex
        peaks["lanczos"] = BranchPeak(
            baseline=0.0,
            giwk_shareable=0.0,
            off_distributed=0.0,
            off_single=scale * max(solver_single_fast, pairing_gather) + giwk_dga_single,
            on_distributed=scale * (LANCZOS_VERTEX_FACTOR * vertex_pp_slice + arpack_ws),
            on_single=scale * max(chi0_pp_full, pairing_gather) + giwk_dga_single,
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

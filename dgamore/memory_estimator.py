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
arrays of each branch are modeled; a single global ``OVERHEAD_FACTOR`` accounts for the un-modeled transients (known
un-modeled costs: the mixing history of ``apply_mixing_strategy`` and the pp pairing-vertex assembly in
``eliashberg_solver.solve``, both dominated by the modeled branch peaks).

Each branch carries its **own** persistent per-rank ``baseline`` (the full-grid two-point objects resident at that
branch's peak) and the portion of it (``giwk_shareable``) that ``config.memory.use_shared_memory_common_obj`` deduplicates
to a single copy per node; the node-total assembly lives in the driver
(:func:`dgamore.DGAmore.autodetect_memory_settings`).
"""

from dataclasses import dataclass

import numpy as np

from dgamore.n_point_base import DTYPE

# Bytes per stored array element, taken from the global storage precision so this stays correct if DTYPE is switched
# (e.g. to complex128).
DTYPE_BYTES: int = np.dtype(DTYPE).itemsize
OVERHEAD_FACTOR: float = 1.1

# chiq_aux builds its two-fermion block via ``(gchi0_inv + gamma) - (v + u)``; each add/sub returns a new full block
# while its input block is still live, and the following invert_and_sum_over_last_vn keeps its input block resident
# while looping over q (the per-q inversion transient is single-q, hence negligible). The peak is therefore
# ~2x the rank-local block, not 1x.
CHIQ_AUX_INVERT_FACTOR: int = 2

# fq builds the block and combines it with whole-block compound-index matmuls (gchi0_q_inv @ f @ gchi0_q_inv) plus a
# second accumulated term, holding ~3 full blocks live at once.
FQ_MATMUL_FACTOR: int = 3

# The fast (FFT) SDE holds the mapped full-BZ niw-half kernel plus its half-niv copy in flight through the conj /
# distributed-FFT round trip (in- and output buffers of half the kernel each), so ~2 kernel-sized blocks are live.
SDE_FFT_KERNEL_FACTOR: int = 2

# ``numpy.fft.ifftn`` on the complex64 full-grid bubble buffer transiently allocates ~2x the buffer beyond it
# (the returned array plus internal work arrays; measured 2026-07-03 on numpy 2.x, dtype preserved), so the per-iw
# FFT moment holds the multiply buffer plus this transient.
CHI0Q_IFFTN_TRANSIENT_FACTOR: int = 2

# The in-memory Eliashberg solver holds the direct vertex, its momentum-/frequency-flipped copy and one freshly
# materialized matmul-layout copy at the layout-build peak (each einsum-layout source is freed right after its
# matmul-layout copy exists, so the peak is 3 vertices, the residency 2).
LANCZOS_VERTEX_FACTOR: int = 3

# Gap-vector-sized matvec temporaries live beyond the ARPACK Lanczos basis (chi0*gap, its flipped copy, the
# contracted output) in both solver variants.
ARPACK_EXTRA_VECTORS: int = 3


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
    construct_fq_cheap: bool = False,
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
    identical and the driver only verifies the fit): ``"chi0q"``, ``"chiq_aux"``, ``"sde"`` are always present;
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
    :param construct_fq_cheap: Whether the ``fq`` per-q blocks are built on the smaller pp frequency box
        (``config.eliashberg.construct_fq_cheap``), shrinking every per-q two-fermion block from ``vc`` to ``vpp``.
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

    # Persistent baselines (full-grid two-point objects live on every rank at the branch peak). During the bubble,
    # giwk_full and sigma_old are both at the niv_cut window; before the kernel section giwk is cut (and re-shared)
    # to the niv_core + niw_core window and sigma_old to the core box. The sde step additionally builds a full
    # R-space Green's-function copy (giwk.fft()), which is node-shared alongside giwk itself when the shared-giwk
    # window is active. The remaining self-energies (sigma_dmft, sigma_dmft_full, delta_sigma) are local (a single
    # k-point) and negligible. The Eliashberg step runs after the loop with sigma_dga freed on every rank and
    # giwk_dga surviving on the bubble-building rank only (a single-rank term, not a per-rank baseline).
    giwk_bubble = scale * _giwk_rspace(nk_tot, nb, 2 * niv_cut)
    giwk_sde = scale * _giwk_rspace(nk_tot, nb, 2 * niv_sde)
    baseline_bubble = giwk_bubble + scale * _giwk_rspace(nk_tot, nb, 2 * niv_cut)  # + sigma_old at niv_cut
    baseline_kernel_section = giwk_sde + scale * _giwk_rspace(nk_tot, nb, vc)  # + sigma_old at the core box
    baseline_sde = baseline_kernel_section + giwk_sde  # + the (node-shareable) R-space Green's-function copy
    giwk_dga_single = scale * _giwk_rspace(nk_tot, nb, 2 * niv_cut)  # the single surviving giwk_dga copy

    peaks: dict[str, BranchPeak] = {}

    # chi0q: fast path (FFT, create_generalized_chi0_q_fft) builds the WHOLE irreducible-BZ bubble on rank 0
    # (nk_irr, not the per-rank q-count) plus the full-grid B^4 multiply buffer ``chi_r_v_buffer`` AND the
    # ~2x-buffer-sized ``xp.fft.ifftn`` transient each iw, plus the replicated real-space Green's functions: two at
    # the (niv_full + niw_core) window (g_k stays bound through the loop, g_r_rev is its flipped/transposed copy) and
    # the central niv_full window slice (g_r). All on rank 0, so the whole fast path is a SINGLE-rank transient. The
    # lean per-q einsum builds only this rank's q-slice of the bubble plus its two shared-backing Green's-function
    # buffers (g_full + g_r_buf), so it is DISTRIBUTED.
    gf_window_bubble = 2 * (niv_full + niw_core)
    peaks["chi0q"] = BranchPeak(
        baseline=baseline_bubble,
        giwk_shareable=giwk_bubble,
        off_distributed=0.0,
        off_single=scale
        * (
            _bubble_block(nk_irr, nb, wp, vf)
            + (1 + CHI0Q_IFFTN_TRANSIENT_FACTOR) * _bubble_block(nk_tot, nb, 1, vf)
            + 2 * _giwk_rspace(nk_tot, nb, gf_window_bubble)
            + _giwk_rspace(nk_tot, nb, vf)
        ),
        on_distributed=scale * (_bubble_block(qi, nb, wp, vf) + 2 * _giwk_rspace(nk_tot, nb, gf_window_bubble)),
        on_single=0.0,
    )

    # chiq_aux: fast path (v1) materializes the whole rank-local two-fermion block on every rank (DISTRIBUTED) and
    # inverts it one q at a time, plus the full-BZ kernel assembled on a SINGLE rank (rank 0) by the gather/unfold
    # irr-to-full-BZ map this flag selects (map_irrbz_fullbz, run per FFT-SDE pass). The lean path (v3) builds one q
    # at a time and accumulates the (1-fermion) summed result, all DISTRIBUTED (p2p map, no single-rank assembly).
    peaks["chiq_aux"] = BranchPeak(
        baseline=baseline_kernel_section,
        giwk_shareable=giwk_sde,
        off_distributed=scale * CHIQ_AUX_INVERT_FACTOR * _two_fermion_block(qi, nb, wp, vc),
        off_single=scale * _bubble_block(nk_tot, nb, wp, vc),
        on_distributed=scale
        * (CHIQ_AUX_INVERT_FACTOR * _two_fermion_block(1, nb, wp, vc) + _bubble_block(qi, nb, wp, vc)),
        on_single=0.0,
    )

    # sde: a single (flag-less) path - the two-pass FFT contraction. It keeps the mapped full-BZ niw-half kernel
    # plus its half-niv copy in flight through the conj/distributed-FFT round trip (~SDE_FFT_KERNEL_FACTOR kernel
    # blocks), retains the irreducible-BZ kernel across both passes, and makes one private real-space
    # Green's-function copy (giwk.fft(copy=True)) per rank at the SDE window; rank 0 additionally finalizes the
    # gathered self-energy (ifft + to_full_niv_range, ~2 full-grid nb^2 objects at the core box). The old q-loop
    # path is unused (it restored the FULL bosonic range on the kernel and peaked HIGHER - see
    # nonlocal_sde.calculate_sigma_from_kernel), so no save_memory switch exists and both path slots carry the FFT
    # path; the driver only verifies the fit.
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
        # fq: both paths hold ~FQ_MATMUL_FACTOR two-fermion blocks at the matmul-chain peak (per rank for the fast
        # path, per q for the lean one) plus the loaded 1-fermion inputs (gchi0_q_inv for the fast path; gchi0_q_inv,
        # vrg_left and vrg_right for the lean one, which keeps all three alive through its q-loop) and, for the lean
        # path, the rank-local accumulator. ``construct_fq_cheap`` shrinks every per-q construction block from vc to
        # vpp; the accumulator keeps the full ph box [wn, vc, vc] when ``save_fq`` is set, otherwise the small pp box
        # [vpp, vpp]. ``save_fq`` additionally gathers the WHOLE irreducible-BZ two-fermion vertex on one rank for
        # saving - a single-rank peak in BOTH paths (usually the largest single object of a save_fq run).
        vc_fq = vpp if construct_fq_cheap else vc
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

        # lanczos: the fast (in-memory) path assembles the entire-BZ pairing vertex on ONE solving rank and holds
        # LANCZOS_VERTEX_FACTOR copies of it at the matmul-layout build (direct + flipped + one layout copy), plus
        # the full-BZ pp bubble and the ARPACK workspace (the ncv-column Lanczos basis + gap-sized matvec
        # temporaries). The singlet and triplet solves run CONCURRENTLY on two ranks - the driver doubles this
        # single-rank peak when both land on the same node; a single-rank run solves sequentially but holds the
        # waiting channel's gathered irreducible-BZ vertex alongside. The lean path (gather_full_ibz_for_vslice)
        # hands every active rank the FULL BZ with only its slice of the second fermionic frequency (the v-axis has
        # only 2*niv_pp tasks, hence the ceil-based share), also LANCZOS_VERTEX_FACTOR copies at the layout build,
        # plus the same per-rank ARPACK workspace (every active rank runs its own full-length eigsh); the root rank
        # additionally holds the full-BZ pp bubble. ``save_pairing_vertex`` gathers both irreducible-BZ pp vertices
        # on rank 0 before the solve (sequential, hence max() against the solver single-rank peak).
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

    return peaks

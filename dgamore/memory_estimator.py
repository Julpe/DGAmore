# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
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
arrays of each branch are modelled; a single global ``OVERHEAD_FACTOR`` accounts for the un-modelled transients.
"""

from dataclasses import dataclass

import numpy as np

from dgamore.n_point_base import DTYPE

# Bytes per stored array element, taken from the global storage precision so this stays correct if DTYPE is switched
# (e.g. to complex128).
DTYPE_BYTES: int = np.dtype(DTYPE).itemsize
OVERHEAD_FACTOR: float = 1.1

# chiq_aux builds its two-fermion block via ``(gchi0_inv + gamma) - (v + u)``; ``sub`` is ``add(-other)``, so the
# add-then-sub chain holds two full blocks live, and the following invert_and_sum_over_last_vn keeps its input block
# resident while looping over q (the per-q inversion transient is single-q, hence negligible). The peak is therefore
# ~2x the rank-local block, not 1x.
CHIQ_AUX_INVERT_FACTOR: int = 2

# fq builds the block and combines it with whole-block compound-index matmuls (gchi0_q_inv @ f @ gchi0_q_inv) plus a
# second accumulated term, holding ~3 full blocks live at once.
FQ_MATMUL_FACTOR: int = 3


@dataclass(frozen=True)
class BranchPeak:
    """
    Per-rank transient peak bytes of one memory-sensitive branch, split by how the peak is distributed across the MPI
    ranks of a node (the persistent baseline is reported separately and is *not* included here).

    For the node-total budget the memory on a node with ``r`` ranks at this branch's peak is
    ``r * (baseline + distributed) + single``: a *distributed* transient is held by every rank simultaneously (so it
    scales with ``r``), while a *single-rank* transient is built on one rank while the others idle (so it is counted
    once). Both the fast (``off``) and lean (``on``) code paths are described.

    :ivar off_distributed: Per-rank transient bytes held by every rank in the fast (flag-off) path.
    :ivar off_single: Transient bytes held by a single rank in the fast (flag-off) path.
    :ivar on_distributed: Per-rank transient bytes held by every rank in the lean (flag-on) path.
    :ivar on_single: Transient bytes held by a single rank in the lean (flag-on) path.
    """

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
    (the FFT paths and the persistent baseline hold such replicated buffers).

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
    overhead: float = OVERHEAD_FACTOR,
) -> tuple[float, dict[str, BranchPeak]]:
    """
    Estimates the per-rank transient peak host-memory (in bytes) of the fast and lean code path of each
    memory-sensitive branch, split by whether each transient is distributed across the ranks of a node or built on a
    single rank, together with the per-rank persistent baseline.

    The returned dict maps a branch key to a :class:`BranchPeak`; the branch keys mirror the ``save_memory_for_*``
    switches: ``"chi0q"``, ``"chiq_aux"``, ``"sde"`` are always present; ``"fq"`` and ``"lanczos"`` are added only
    when ``with_eliashberg`` is True. The first tuple element is the per-rank persistent baseline (the replicated
    full-grid Green's function and self-energies that stay live throughout the non-local routine); the caller adds it
    to the node total (every rank holds it). For a node with ``r`` ranks the memory at a branch's peak is
    ``r * (baseline + distributed) + single``.

    :param n_bands: Number of bands :math:`B`.
    :param nk_tot: Total number of momentum points (full BZ).
    :param nk_irr: Number of momentum points in the irreducible BZ.
    :param niw_core: Number of positive bosonic core frequencies.
    :param niv_core: Number of positive fermionic core frequencies.
    :param niv_full: Number of positive fermionic full-region frequencies.
    :param niv_cut: Number of positive fermionic frequencies the full-grid ``giwk_full`` is kept at through the
        kernel/SDE section (``min(niw_core + niv_full + 10, niv_dmft)`` in
        :func:`dgamore.nonlocal_sde.calculate_self_energy_q`); the SDE self-energy contraction needs the shell window,
        so giwk is not shrunk to the core box here.
    :param niv_pp: Number of positive fermionic frequencies of the pp (Eliashberg) box.
    :param n_ranks: Number of MPI ranks the q-points are distributed over.
    :param with_eliashberg: Whether the Eliashberg step runs (adds the ``"fq"`` and ``"lanczos"`` branches).
    :param save_fq: Whether the full ladder vertex is kept in the full ph box (``config.eliashberg.save_fq``); when
        True the per-rank ``fq`` accumulator spans the full ``[wn, vc, vc]`` block instead of the small pp box.
    :param construct_fq_cheap: Whether the ``fq`` per-q blocks are built on the smaller pp frequency box
        (``config.eliashberg.construct_fq_cheap``), shrinking every per-q two-fermion block from ``vc`` to ``vpp``.
    :param overhead: Global multiplicative factor accounting for un-modelled transient arrays.
    :return: A tuple ``(baseline_bytes, peaks)`` of the per-rank baseline and a dict mapping each branch key to its
        :class:`BranchPeak`.
    """
    nb = n_bands
    wp = niw_core + 1  # half bosonic range, as the heavy objects are constructed
    vc = 2 * niv_core
    vf = 2 * niv_full
    vpp = 2 * niv_pp

    qi = _ceil_div(nk_irr, n_ranks)  # per-rank irreducible-BZ q-count
    qt = _ceil_div(nk_tot, n_ranks)  # per-rank full-BZ q-count

    scale = DTYPE_BYTES * overhead

    # Per-rank persistent baseline at the heavy-section (kernel/SDE) peak: the two full-grid two-point objects that
    # stay live on every rank -- giwk_full and sigma_old, both kept at niv_cut through the SDE (giwk's shell window is
    # needed by the self-energy contraction; sigma_old keeps its DMFT shell for the mixing/residual). See
    # nonlocal_sde.calculate_self_energy_q. The remaining self-energies (sigma_dmft, sigma_dmft_full, delta_sigma) are
    # local (a single k-point) and negligible.
    baseline = scale * 2 * _giwk_rspace(nk_tot, nb, 2 * niv_cut)

    peaks: dict[str, BranchPeak] = {}

    # chi0q: fast path (FFT, create_generalized_chi0_q_fft) builds the WHOLE irreducible-BZ bubble on rank 0
    # (nk_irr, not the per-rank q-count) plus TWO full-grid B^4 buffers, each with one fermionic axis over nk_tot --
    # the preallocated ``chi_r_v_buffer`` multiply target and the equally large array returned by ``xp.fft.ifftn``
    # each iw -- and ~2 replicated real-space Green's functions over the (niv_full + niw_core) window. All on rank 0,
    # so the whole fast path is a SINGLE-rank transient. The lean per-q einsum builds only this rank's q-slice of
    # the bubble plus its own ~2 Green's-function buffers, so it is DISTRIBUTED.
    gf_copies = 2 * _giwk_rspace(nk_tot, nb, 2 * (niv_full + niw_core))
    peaks["chi0q"] = BranchPeak(
        off_distributed=0.0,
        off_single=scale * (_bubble_block(nk_irr, nb, wp, vf) + 2 * _bubble_block(nk_tot, nb, 1, vf) + gf_copies),
        on_distributed=scale * (_bubble_block(qi, nb, wp, vf) + gf_copies),
        on_single=0.0,
    )

    # chiq_aux: fast path (v1) materializes the whole rank-local two-fermion block on every rank (DISTRIBUTED) and
    # inverts it one q at a time, plus the full-BZ kernel gathered on a SINGLE rank (rank 0). The lean path (v3)
    # builds one q at a time and accumulates the (1-fermion) summed result, all DISTRIBUTED.
    peaks["chiq_aux"] = BranchPeak(
        off_distributed=scale * CHIQ_AUX_INVERT_FACTOR * _two_fermion_block(qi, nb, wp, vc),
        off_single=scale * _bubble_block(nk_tot, nb, wp, vc),
        on_distributed=scale
        * (CHIQ_AUX_INVERT_FACTOR * _two_fermion_block(1, nb, wp, vc) + _bubble_block(qi, nb, wp, vc)),
        on_single=0.0,
    )

    # sde: both paths are DISTRIBUTED. The fast path (FFT) materializes the full-BZ kernel on every rank plus a
    # replicated R-space Green's function (giwk is kept at niv_cut by this point) and the kernel's own R-space buffer.
    # The lean q-loop (calculate_sigma_from_kernel_cpu) maps the kernel once but then makes a full
    # ``np.asfortranarray(giwk.mat)`` copy of the Green's function (and a Fortran copy of the kernel of comparable
    # size to the mapped one), so beyond the mapped kernel it holds a replicated full-grid Green's function at
    # niv_cut. Both paths carry a comparable kernel transient (the FFT path redistributes it, the q-loop copies it),
    # so only the giwk copy is counted here.
    sde_kernel_full = _bubble_block(qt, nb, wp, vc)
    peaks["sde"] = BranchPeak(
        off_distributed=scale * (sde_kernel_full + 2 * _giwk_rspace(nk_tot, nb, 2 * niv_cut)),
        off_single=0.0,
        on_distributed=scale * (sde_kernel_full + _giwk_rspace(nk_tot, nb, 2 * niv_cut)),
        on_single=0.0,
    )

    if with_eliashberg:
        # fq: both paths are DISTRIBUTED. The fast path builds the whole rank-local two-fermion ph block and combines
        # it with whole-block compound matmuls, holding ~FQ_MATMUL_FACTOR full blocks live; the lean path does the
        # same one q at a time but additionally writes into a rank-local accumulator (f_q_r_mat) spanning ALL
        # rank-local q-points. ``construct_fq_cheap`` shrinks every per-q construction block from vc to vpp; the
        # accumulator keeps the full ph box [wn, vc, vc] when ``save_fq`` is set, otherwise the small pp box [vpp, vpp].
        vc_fq = vpp if construct_fq_cheap else vc
        fq_accumulator = _two_fermion_block(qi, nb, wp, vc) if save_fq else _two_fermion_block(qi, nb, 1, vpp)
        peaks["fq"] = BranchPeak(
            off_distributed=scale * FQ_MATMUL_FACTOR * _two_fermion_block(qi, nb, wp, vc_fq),
            off_single=0.0,
            on_distributed=scale * (FQ_MATMUL_FACTOR * _two_fermion_block(1, nb, wp, vc_fq) + fq_accumulator),
            on_single=0.0,
        )

        # lanczos: the fast (in-memory) path assembles the entire BZ pairing vertex on ONE rank AND a momentum-flipped
        # copy of it (flip_momentum_axis allocates a fresh full array), so two full-BZ vertices are live on a SINGLE
        # rank. The lean path (gather_full_ibz_for_vslice + map_to_full_bz) hands every rank the FULL BZ with only a
        # slice of the second fermionic frequency, so its per-rank share scales with the full-BZ per-rank q-count
        # (nk_tot/n_ranks), NOT the irreducible one -- also two copies (vertex + flipped), DISTRIBUTED.
        peaks["lanczos"] = BranchPeak(
            off_distributed=0.0,
            off_single=scale * 2 * _two_fermion_block(nk_tot, nb, 1, vpp),
            on_distributed=scale * 2 * _two_fermion_block(qt, nb, 1, vpp),
            on_single=0.0,
        )

    return baseline, peaks

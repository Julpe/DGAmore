# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Pairing vertex of the Eliashberg step: the particle-particle vertex at :math:`\omega = 0` in the singlet and triplet
channels, built per momentum slice from the ladder-DGA three-leg vertices the non-local step saved (or streamed into
the full vertex :math:`F` under ``eliashberg.save_fq``), and the local reducible diagrams the solver adds to it.
"""

import os

import mpi4py.MPI as MPI
import numpy as np

import dgamore.config as config
from dgamore import mpi_utils, output_files, sde_kernels
from dgamore.bubble_gen import BubbleGenerator
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.local_four_point import LocalFourPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore.memory_estimator import SLICE_CHUNK_BYTES
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel, FrequencyNotation, DTYPE, deferred_collection


# --- Frequency transform helpers (PH -> PP w0) ---
def _pp_w0_band(niv_pp: int, niw_stored: int) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Index map of the :math:`\omega' = 0` particle-particle band, shared by every consumer of the ph-to-pp map so that
    the map exists in exactly one place. For a pp entry :math:`(\nu, \nu')` the ph object is read at the bosonic
    frequency :math:`\omega = \nu - \nu'`, so each :math:`\omega` contributes one anti-diagonal of the
    :math:`(\nu, \nu')` plane and only :math:`|\omega| \leq 2 n_{\nu}^{\mathrm{pp}} - 1` is ever read.

    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param niw_stored: Number of positive bosonic frequencies available in the ph object.
    :return: ``(wn, omega)``: the bosonic indices of the readable window, and the matrix
        :math:`\omega = \nu - \nu'` whose entries select one anti-diagonal per bosonic frequency.
    """
    vn = MFHelper.vn(niv_pp)
    return MFHelper.wn(min(niw_stored, 2 * niv_pp - 1)), vn[:, None] - vn[None, :]


def _transform_vertex_frequencies_w0(vertex: LocalFourPoint | FourPoint, niv_pp: int) -> np.ndarray:
    r"""
    Transforms a vertex from particle-hole to particle-particle notation at :math:`\omega' = 0`, following Motoharu
    Kitatani's frequency convention: the fermionic frequency is flipped, the bosonic index is remapped via
    :math:`\omega = \nu - \nu'` and the orbitals are permuted to :math:`1432`. In full index notation the output is

    .. math:: \bar{F}^{\mathrm{pp};\nu\nu'}_{1234} = -F^{\mathrm{ph};\,\omega=\nu-\nu';\ \nu_1=\nu,\ \nu_2=-\nu'}_{1432}
        = -F^{\mathrm{ph};(\nu-\nu')\nu(-\nu')}_{1432},

    i.e. (minus) the crossed-slot form of the pairing vertex of Eq. (4.49) in my thesis: with the ph frequency
    convention of Eq. (3.28a) the four legs of :math:`\bar{F}^{\mathrm{pp};\nu\nu'}_{1234}` carry the frequencies
    :math:`(\nu, \nu', -\nu, -\nu')` on the orbitals :math:`(1, 4, 3, 2)`. The overall minus is the sign of the
    power-iteration matrix :math:`M = -\Gamma\chi` of Eq. (4.42). Used by :func:`transform_vertex_loc_frequencies_w0`; the direct-slot counterpart (:math:`\omega_{\mathrm{ph}} = \nu +
    \nu'`, no flip, orbitals :math:`1234`) is
    :meth:`~dgamore.local_four_point.LocalFourPoint.change_frequency_notation_ph_to_pp_w0`.

    :param vertex: The vertex to transform (:class:`LocalFourPoint` or :class:`FourPoint`) in ph notation.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :return: The transformed vertex as a raw numpy array with two fermionic axes ``[..., 2*niv_pp, 2*niv_pp]``.
    """
    vertex = vertex.cut_niv(niv_pp)
    # only the |w| <= 2*niv_pp - 1 anti-diagonals (omega = v - v') are read below, so the bosonic axis is trimmed to
    # that window before to_full_niw_range doubles it (cut_niw's no-op guard misjudges half-range objects here)
    w_axis = -3
    niw_stored = vertex.current_shape[w_axis] // 2 if vertex.full_niw_range else vertex.current_shape[w_axis] - 1
    wn, omega = _pp_w0_band(niv_pp, niw_stored)
    niw_window = (len(wn) - 1) // 2
    if niw_window < niw_stored:
        slicer = [slice(None)] * vertex.mat.ndim
        slicer[w_axis] = (
            slice(niw_stored - niw_window, niw_stored + niw_window + 1)
            if vertex.full_niw_range
            else slice(0, niw_window + 1)
        )
        vertex.mat = vertex.mat[tuple(slicer)].copy()
        vertex.update_original_shape()

    vertex = vertex.to_full_niw_range().permute_orbitals("abcd->adcb", copy=False).flip_frequency_axis(-1, False)
    f_q_r_pp_mat = np.zeros((*vertex.current_shape[:-3], 2 * niv_pp, 2 * niv_pp), dtype=vertex.mat.dtype)

    for idx, w in enumerate(wn):
        f_q_r_pp_mat[..., omega == w] = -vertex[..., idx, omega == w]
    return f_q_r_pp_mat


def transform_vertex_loc_frequencies_w0(f_r_loc: LocalFourPoint, niv_pp: int) -> LocalFourPoint:
    r"""
    Transforms a local vertex from particle-hole to the modified particle-particle notation at :math:`\omega' = 0`
    (see :func:`_transform_vertex_frequencies_w0`).

    :param f_r_loc: The local vertex :math:`F` in ph notation.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :return: The transformed vertex as a :class:`LocalFourPoint` (channel UD, pp notation, no bosonic axis).
    """
    mat = _transform_vertex_frequencies_w0(f_r_loc, niv_pp)
    return LocalFourPoint(mat, SpinChannel.UD, 0, 2, True, True, FrequencyNotation.PP)


def _wn_chunk_size(one_wn_bytes: int, chunk_bytes: int) -> int:
    """Number of bosonic frequencies assembled at once, bounded by the chunk byte budget (never below one)."""
    return max(1, int(chunk_bytes // max(one_wn_bytes, 1)))


def _build_ladder_vertex_chunk(
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    vrg_q_r_left: FourPoint,
    middle_q_r: FourPoint,
    w_start: int,
    w_stop: int,
    niv_pp: int | None,
    inactive_pairs: np.ndarray | None = None,
) -> FourPoint:
    r"""
    Builds the full ladder vertex :math:`F^{\mathrm{q}}_{r}` for the momenta of the inputs and the bosonic window
    ``[w_start, w_stop)`` from the ladder of :math:`\Gamma_{r}` alone: its amputated susceptibility plus the separable
    part :math:`(\gamma^{\Gamma}_{r} Y_{r}) \tilde\gamma^{\Gamma}_{r}` with the matrix :math:`Y_{r}` between the
    three-leg vertices (see :func:`~dgamore.sde_kernels.create_ladder_vertex_middle`). This is the vertex of the
    auxiliary ladder (shift :math:`U_{r}`), written so that no inversion meets the zero eigenvalues the auxiliary
    Bethe-Salpeter matrix can have where the physical susceptibility is regular. The right-sided three-leg vertex
    :math:`\tilde\gamma^{\Gamma;\mathrm{q}\nu'}_{r;1234} = \beta \sum_{ab}\sum_{\nu}
    \chi^{\Gamma;\mathrm{q}\nu\nu'}_{r;12ab} (\chi^{\mathrm{q}\nu'}_{0;ba34})^{-1}` sums the ladder susceptibility
    over its first frequency, which the build takes from the same inversion (exact also where the Bethe-Salpeter matrix
    is not complex-symmetric). The Bethe-Salpeter matrix is built one compound slice at a time from the local vertex
    and the inverse bubble, never as a whole window.

    With ``niv_pp`` given, only what the :math:`\omega' = 0` pp band reads is built: the ladder susceptibility on the
    anti-diagonal :math:`\nu + \nu' = \omega` of the pp box (see
    :meth:`FourPoint.invert_with_local_kernel_on_anti_diagonal`, with the inverse bubble as the frequency-diagonal
    part, the kernel :math:`\Gamma_{r}`, a zero shift and the scale :math:`1/\beta^2`), and the whole chain on the pp
    box, so the result is exact on that anti-diagonal only. With ``niv_pp`` None the whole core box is inverted (see
    :meth:`FourPoint.invert_with_local_kernel`) and every entry is exact (the streamed full vertex).

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel, from :math:`\omega = 0` on
        (half bosonic range) or on the full range.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` of the momenta.
    :param vrg_q_r_left: The three-leg vertex :math:`\gamma^{\Gamma;\mathrm{q}\nu}_{r}` of the ladder of
        :math:`\Gamma_{r}` of the momenta.
    :param middle_q_r: The matrix :math:`Y^{\mathrm{q}}_{r}` between the three-leg vertices of the momenta.
    :param w_start: First bosonic index of the window.
    :param w_stop: One past the last bosonic index of the window.
    :param niv_pp: Number of positive fermionic frequencies of the pp box, or None for the whole core box.
    :param inactive_pairs: Orbital pairs without vertex (see :meth:`LocalFourPoint.orbital_pairs_without_vertex`).
    :return: The ladder vertex over that window as a :class:`FourPoint` (half niw range, two fermionic dimensions).
    """
    beta = config.sys.beta
    zero = LocalInteraction(np.zeros((gamma_r.n_bands,) * 4), gamma_r.channel)
    gchi0_w = gchi0_q_inv.take_wn_slice(w_start, w_stop)
    vrg_left_w = vrg_q_r_left.take_wn_slice(w_start, w_stop)
    if niv_pp is None:
        chi_ladder = gchi0_w.invert_with_local_kernel(gamma_r, zero, 1.0 / beta**2, w_start)
        chi_ladder_first_sum = chi_ladder.sum_over_vn(beta, axis=(-2,))
    else:
        chi_ladder, chi_ladder_first_sum = gchi0_w.invert_with_local_kernel_on_anti_diagonal(
            gamma_r, zero, 1.0 / beta**2, beta, niv_pp, w_start, inactive_pairs
        )
        gchi0_w = gchi0_w.cut_niv(niv_pp)
        vrg_left_w = vrg_left_w.cut_niv(niv_pp)

    # chi is released right after the first matmul and its product by rebinding; the bubble term enters in place
    f_chunk = gchi0_w @ chi_ladder
    del chi_ladder
    f_chunk = f_chunk @ gchi0_w
    f_chunk = f_chunk.scale(-(beta**2)).add_on_vn_diagonal(gchi0_w, factor=beta**2)

    vrg_right_w = (chi_ladder_first_sum @ gchi0_w).scale(beta)
    return f_chunk.add((vrg_left_w @ middle_q_r.take_wn_slice(w_start, w_stop)) * vrg_right_w, copy=False)


def _build_rpa_vertex_chunk(chi_q_r: FourPoint, u_r: Interaction, w_start: int, w_stop: int, niv: int) -> FourPoint:
    r"""
    Builds the full vertex of the RPA ladder, :math:`F^{\mathrm{q}}_{r;1234} = \mathcal{U}^{\mathbf{q}}_{r;1234} -
    \sum_{abcd} \mathcal{U}^{\mathbf{q}}_{r;12ab}\,\chi^{\mathrm{q}}_{r;bacd}\,\mathcal{U}^{\mathbf{q}}_{r;dc34}`, for
    the momenta of the inputs and the bosonic window ``[w_start, w_stop)``: the vertex of
    :func:`_build_ladder_vertex_chunk` for :math:`\Gamma_{r} = U_{r}`, where the auxiliary susceptibility is the
    bubble, the bubble term vanishes and both three-leg vertices are one. It does not depend on the fermionic
    frequencies and is held constant on the box of ``niv`` positive frequencies.

    :param chi_q_r: The physical susceptibility :math:`\chi^{\mathrm{q}}_{r}` of the momenta.
    :param u_r: The channel-projected total interaction :math:`\mathcal{U}^{\mathbf{q}}_{r}`.
    :param w_start: First bosonic index of the window.
    :param w_stop: One past the last bosonic index of the window.
    :param niv: Number of positive fermionic frequencies of the box.
    :return: The vertex over that window as a :class:`FourPoint` (half niw range, two fermionic dimensions).
    """
    chi_w = chi_q_r.take_wn_slice(w_start, w_stop)
    return (u_r @ chi_w @ u_r).scale(-1.0).add(u_r, copy=False).repeat_over_vn(niv, 2)


def _write_pp_band(out: np.ndarray, f_chunk: FourPoint, niv_pp: int, omega: np.ndarray, w_start: int) -> None:
    r"""
    Writes the :math:`\omega' = 0` pp band of one ladder-vertex window into the pp accumulator of its momenta.

    Each bosonic frequency contributes the anti-diagonal :math:`\omega = \nu - \nu'` (see :func:`_pp_w0_band`), and
    the negative bosonic half is obtained from the positive one at the same momentum through the Hermiticity of the
    vertex, :math:`F^{(\mathbf{q},-\omega)\nu\nu'}_{1234} = (F^{(\mathbf{q},\omega)(-\nu')(-\nu)}_{4321})^*` (a plain
    conjugation would need :math:`-\mathbf{q}`). The orbital permutation, the fermionic flip and the overall minus
    are the ones of :func:`_transform_vertex_frequencies_w0`.

    :param out: The pp accumulator of this momentum group, shape ``[nq_group, no, no, no, no, 2 niv_pp, 2 niv_pp]``.
    :param f_chunk: The ladder vertex over a bosonic window, in ph notation and half niw range.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param omega: The :math:`\nu - \nu'` matrix returned by :func:`_pp_w0_band`.
    :param w_start: Bosonic index the window starts at.
    :return: None.
    """
    # cut to the pp box first so the negative-half copy is pp-sized, not core-sized (the cut is centered, so the
    # fermionic flips inside to_negative_niw_range commute with it); positive then mutates the cut copy in place
    cut = f_chunk.cut_niv(niv_pp)
    # Hermitian partner F^{-w}_{1234}(v, v') = conj(F^{w}_{4321}(-v', -v)), then the 1432 slot map: orbitals 2341
    negative = (
        cut.to_negative_niw_range()
        .swap_fermionic_frequency_axes(copy=False)
        .permute_orbitals("bcda->abcd", copy=False)
        .flip_frequency_axis(-1, False)
    )
    positive = cut.permute_orbitals("abcd->adcb", copy=False).flip_frequency_axis(-1, False)

    for index in range(positive.current_shape[-3]):
        w = w_start + index
        out[..., omega == w] = -positive.mat[..., index, omega == w]
        if w > 0:  # w = 0 is shared by both halves and belongs to the positive one
            out[..., omega == -w] = -negative.mat[..., index, omega == -w]


def _build_pairing_vertex_pp(
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    gamma_r: LocalFourPoint | None,
    niv_pp: int,
    mpi_dist_irrk: MpiDistributor,
    chunk_writer=None,
    chunk_bytes: int | None = None,
    channel: SpinChannel | None = None,
    gchi0_q_inv: FourPoint | None = None,
) -> FourPoint:
    r"""
    Shared loop of the slice-direct pairing-vertex construction: walks the rank-local momenta and, per momentum, the
    bosonic axis in byte-bounded chunks, building each ladder-vertex chunk once and reading its :math:`\omega' = 0`
    pp band (see :func:`_pp_w0_band`). When ``chunk_writer`` is given, every ph-notation chunk is handed to it before
    being released, so a caller can stream the full vertex to disk in the same pass. The ladder chunks read the inverse
    bubble, the saved three-leg vertex and vertex middle of the ladder of :math:`\Gamma_{r}` (see
    :func:`_build_ladder_vertex_chunk`). Without a local vertex the chunks are the RPA vertex of the saved physical
    susceptibility (see :func:`_build_rpa_vertex_chunk`), the only intermediate read then.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel, or ``None`` for the RPA vertex.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param chunk_writer: Optional callable ``(q_start, w_start, chunk_mat)`` receiving each ph-notation chunk (the
        momenta of the group as the leading axis).
    :param chunk_bytes: Chunk byte budget of the build (``None`` uses the floor).
    :param channel: Spin channel of the RPA vertex, i.e. density or magnetic; only read when ``gamma_r`` is ``None``.
    :param gchi0_q_inv: The inverse bare bubble of this rank's momenta on the core box (see
        :func:`~dgamore.eliashberg_solver.inverse_core_bubble`), read and kept; ``None`` loads the saved
        ``gchi0_q_inv_rank_<r>.npy`` and frees it.
    :return: The pairing vertex :math:`F^{\mathrm{q}}_{r}` in pp notation as a :class:`FourPoint`.
    """
    logger = config.logger
    channel = channel if gamma_r is None else gamma_r.channel
    path, rank = config.output.eliashberg_path, mpi_dist_irrk.my_rank

    loaded_bubble = gamma_r is not None and gchi0_q_inv is None
    if loaded_bubble:
        gchi0_q_inv = FourPoint.load(
            output_files.npy_path(path, output_files.rank_name(output_files.GCHI0_Q_INV, rank)), num_vn_dimensions=1
        )
    if gamma_r is not None:
        vrg_q_r_left = FourPoint.load(
            output_files.npy_path(path, output_files.rank_name(output_files.VRG_Q, rank, channel)),
            channel=channel,
            num_vn_dimensions=1,
        )
    bosonic_name = output_files.VRG_MIDDLE_Q if gamma_r is not None else output_files.CHI_PHYS_Q
    bosonic_q_r = FourPoint.load(
        output_files.npy_path(path, output_files.rank_name(bosonic_name, rank, channel)),
        channel=channel,
        num_vn_dimensions=0,
    )
    logger.info(f"Loaded the intermediates for the {channel.value} pairing vertex.")

    my_irr_q_list = config.lattice.k_grid.get_irrq_list()[mpi_dist_irrk.my_slice]
    niw_stored = bosonic_q_r.current_shape[-1] - 1
    _, omega = _pp_w0_band(niv_pp, niw_stored)
    # the pp band only reads |w| <= 2 niv_pp - 1, but a streamed full vertex must cover the whole stored box
    niw_build = niw_stored if chunk_writer is not None else int(np.max(omega))

    n_bands = config.sys.n_bands
    # orbital pairs without vertex are eliminated from every slice, as in the kernel step's auxiliary susceptibility
    inactive = None if gamma_r is None else gamma_r.orbital_pairs_without_vertex(u_loc.as_channel(channel))
    f_pp_mat = np.zeros((len(my_irr_q_list),) + (n_bands,) * 4 + (2 * niv_pp,) * 2, dtype=bosonic_q_r.mat.dtype)

    # the budget bounds a chunk's two-fermion window on the box its chain runs on (pp box for the band, core box for a
    # streamed full vertex): as many whole momenta as fit, else one momentum's bosonic axis is chunked
    budget = SLICE_CHUNK_BYTES if chunk_bytes is None else chunk_bytes
    niv_window = config.box.niv_core if chunk_writer is not None else niv_pp
    one_wn_bytes = n_bands**4 * (2 * niv_window) ** 2 * np.dtype(DTYPE).itemsize
    w_chunk = _wn_chunk_size(one_wn_bytes, budget)
    q_group = max(1, int(budget // max(one_wn_bytes * (niw_build + 1), 1)))

    with deferred_collection():
        for q_start in range(0, len(my_irr_q_list), q_group):
            q_stop = min(q_start + q_group, len(my_irr_q_list))
            if gamma_r is not None:
                gchi0_grp = gchi0_q_inv.take_q_index_slice(q_start, q_stop)
                vrg_left_grp = vrg_q_r_left.take_q_index_slice(q_start, q_stop)
            else:
                u_r = u_loc.as_channel(channel) + v_nonloc.take_q_index_slice(q_start, q_stop).as_channel(channel)
            bosonic_grp = bosonic_q_r.take_q_index_slice(q_start, q_stop)

            for w_start in range(0, niw_build + 1, w_chunk):
                w_stop = min(w_start + w_chunk, niw_build + 1)
                if gamma_r is None:
                    f_chunk = _build_rpa_vertex_chunk(bosonic_grp, u_r, w_start, w_stop, niv_window)
                else:
                    f_chunk = _build_ladder_vertex_chunk(
                        gamma_r,
                        gchi0_grp,
                        vrg_left_grp,
                        bosonic_grp,
                        w_start,
                        w_stop,
                        None if chunk_writer is not None else niv_pp,
                        inactive,
                    )
                if chunk_writer is not None:
                    chunk_writer(q_start, w_start, f_chunk.mat)
                _write_pp_band(f_pp_mat[q_start:q_stop], f_chunk, niv_pp, omega, w_start)
                f_chunk.free()

            if gamma_r is not None:
                gchi0_grp.free()
                vrg_left_grp.free()
            bosonic_grp.free()

    if loaded_bubble:
        gchi0_q_inv.free()
    if gamma_r is not None:
        vrg_q_r_left.free()
    bosonic_q_r.free()

    output_files.delete_files(
        path,
        *(
            f"{output_files.rank_name(stem, rank, channel)}.npy"
            for stem in (output_files.VRG_Q, output_files.VRG_MIDDLE_Q, output_files.CHI_PHYS_Q)
        ),
    )

    return FourPoint(f_pp_mat, channel, config.lattice.k_grid.nk, 0, 2, False, True, True, FrequencyNotation.PP)


def create_pairing_vertex_slice_q_r(
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    gamma_r: LocalFourPoint | None,
    niv_pp: int,
    mpi_dist_irrk: MpiDistributor,
    chunk_bytes: int | None = None,
    channel: SpinChannel | None = None,
    gchi0_q_inv: FourPoint | None = None,
) -> FourPoint:
    r"""
    Builds the pp pairing vertex at :math:`\omega' = 0` directly, without ever materializing the full
    three-frequency ladder vertex :math:`F^{\mathrm{q}\nu\nu'}_{r}` (see :func:`_build_pairing_vertex_pp`). The
    result is identical to the full-inversion construction up to floating-point accuracy, at a transient of one
    byte-bounded bosonic chunk instead of the whole rank-local box.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel, or ``None`` for the RPA vertex.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param chunk_bytes: Chunk byte budget of the build (``None`` uses the floor).
    :param channel: Spin channel of the RPA vertex, i.e. density or magnetic; only read when ``gamma_r`` is ``None``.
    :param gchi0_q_inv: The inverse core bubble of this rank's momenta, or ``None`` to load the saved one (see
        :func:`_build_pairing_vertex_pp`).
    :return: The pairing vertex :math:`F^{\mathrm{q}}_{r}` in pp notation as a :class:`FourPoint`.
    """
    return _build_pairing_vertex_pp(
        u_loc,
        v_nonloc,
        gamma_r,
        niv_pp,
        mpi_dist_irrk,
        chunk_bytes=chunk_bytes,
        channel=channel,
        gchi0_q_inv=gchi0_q_inv,
    )


def create_pairing_vertex_streaming_fq(
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    gamma_r: LocalFourPoint | None,
    niv_pp: int,
    mpi_dist_irrk: MpiDistributor,
    chunk_bytes: int | None = None,
    channel: SpinChannel | None = None,
    gchi0_q_inv: FourPoint | None = None,
) -> FourPoint:
    r"""
    Builds the pp pairing vertex like :func:`create_pairing_vertex_slice_q_r` while streaming the full ladder vertex
    in ph notation to ``f_irrq_<channel>.npy`` chunk by chunk, replacing the rank-0 gather of the whole
    irreducible-BZ vertex. Rank 0 creates the memory-mapped file with the layout of the gathered save (half bosonic
    range, global irreducible q-ordering); every rank then writes only its own disjoint ``(q, omega)`` slabs.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel, or ``None`` for the RPA vertex.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param chunk_bytes: Chunk byte budget of the build (``None`` uses the floor).
    :param channel: Spin channel of the RPA vertex, i.e. density or magnetic; only read when ``gamma_r`` is ``None``.
    :param gchi0_q_inv: The inverse core bubble of this rank's momenta, or ``None`` to load the saved one (see
        :func:`_build_pairing_vertex_pp`).
    :return: The pairing vertex :math:`F^{\mathrm{q}}_{r}` in pp notation as a :class:`FourPoint`.
    """
    channel = channel if gamma_r is None else gamma_r.channel
    n_bands = config.sys.n_bands
    nk_irr = config.lattice.k_grid.nk_irr
    file_path = os.path.join(config.output.output_path, f"f_irrq_{channel.value}.npy")
    # plain ints: numpy scalars in the shape end up as np.int64(...) in the npy header, which cannot be re-read
    shape = tuple(
        int(n) for n in (nk_irr,) + (n_bands,) * 4 + (config.box.niw_core + 1,) + (2 * config.box.niv_core,) * 2
    )

    if mpi_dist_irrk.comm.rank == 0:
        np.lib.format.open_memmap(file_path, mode="w+", dtype=DTYPE, shape=shape)
    mpi_dist_irrk.barrier()
    file_mat = np.lib.format.open_memmap(file_path, mode="r+")

    q_offset = mpi_dist_irrk.my_slice.indices(nk_irr)[0]

    def chunk_writer(q_start: int, w_start: int, chunk_mat: np.ndarray) -> None:
        w_stop = w_start + chunk_mat.shape[-3]
        file_mat[q_offset + q_start : q_offset + q_start + chunk_mat.shape[0], ..., w_start:w_stop, :, :] = chunk_mat

    f_pp = _build_pairing_vertex_pp(
        u_loc, v_nonloc, gamma_r, niv_pp, mpi_dist_irrk, chunk_writer, chunk_bytes, channel, gchi0_q_inv
    )
    file_mat.flush()
    del file_mat
    mpi_dist_irrk.barrier()
    config.logger.info(f"Streamed full ladder-vertex ({channel.value}) in the irreducible BZ to file.")
    return f_pp


# --- Local particle-particle reducible diagrams (w=0) ---
def create_local_gamma_ud_pp_w0(
    gchi_ud_pp_w0: LocalFourPoint, gchi0_pp_w0: LocalFourPoint, beta: float
) -> LocalFourPoint:
    r"""
    Returns the local pp-irreducible up-down vertex at :math:`\omega = 0` from the crossing-decoupled pp
    Bethe-Salpeter equation,

    .. math:: \Gamma^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234} = \beta^2 \left[\chi^{\mathrm{pp}}_0 J - \chi^{\mathrm{pp}}_0\,
        (\chi^{\mathrm{pp}}_{\uparrow\downarrow})^{-1}\, \chi^{\mathrm{pp}}_0\right]^{-1;\,\nu\nu'}_{1234}.

    All products and inverses live in compound pp index space, i.e. as matrices :math:`M_{(13\nu),(42\nu')} =
    X^{\mathrm{pp};\nu\nu'}_{1234}` with the product and unit element

    .. math:: (X Y)^{\mathrm{pp};\nu\nu'}_{1234} = \sum_{ab\nu_1} X^{\mathrm{pp};\nu\nu_1}_{1a3b}\, Y^{\mathrm{pp};\nu_1\nu'}_{b2a4}, \qquad
        \mathbb{1}^{\mathrm{pp};\nu\nu'}_{1234} = \delta_{14}\,\delta_{23}\,\delta_{\nu\nu'}.

    The ingredients in full index notation are the diagonal bare pp bubble, built from the local DMFT Green's function
    :math:`G^{\mathrm{DMFT}}_{12}(\nu)`, and its image under the crossing operator :math:`J` (:math:`\nu' \to -\nu'`
    combined with the orbital permutation :math:`1234 \to 1432`, i.e. :math:`(XJ)^{\mathrm{pp};\nu\nu'}_{1234} =
    X^{\mathrm{pp};\nu(-\nu')}_{1432}`),

    .. math:: \chi^{\mathrm{pp};\nu\nu'}_{0;1234} = -\beta\, G^{\mathrm{DMFT}}_{14}(\nu)\, G^{\mathrm{DMFT}}_{32}(-\nu)\,
        \delta_{\nu\nu'}, \qquad (\chi^{\mathrm{pp}}_0 J)^{\nu\nu'}_{1234} = -\beta\, G^{\mathrm{DMFT}}_{12}(\nu)\,
        G^{\mathrm{DMFT}}_{34}(-\nu)\, \delta_{\nu,-\nu'}.

    The returned :math:`\Gamma^{\mathrm{pp}}_{\uparrow\downarrow}` is equivalent to solving the crossing-decoupled pp
    BSE

    .. math:: F^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234} = \Gamma^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234}
        - \frac{1}{\beta} \sum_{\nu_1} \sum_{abcd} \Gamma^{\mathrm{pp};\nu\nu_1}_{\uparrow\downarrow;1a3b}\,
        G^{\mathrm{DMFT}}_{bc}(\nu_1)\, G^{\mathrm{DMFT}}_{ad}(-\nu_1)\,
        F^{\mathrm{pp};(-\nu_1)\nu'}_{\uparrow\downarrow;d2c4}

    for the full vertex :math:`F^{\mathrm{pp}}_{\uparrow\downarrow}` defined by amputating the DMFT legs of the
    susceptibility,

    .. math:: \chi^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234} = -\sum_{abcd} F^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;abcd}\,
        G^{\mathrm{DMFT}}_{1a}(\nu)\, G^{\mathrm{DMFT}}_{b2}(-\nu')\, G^{\mathrm{DMFT}}_{3c}(-\nu)\,
        G^{\mathrm{DMFT}}_{d4}(\nu').

    Note that :math:`\chi^{\mathrm{pp}}_{\uparrow\downarrow}` must be the CONNECTED susceptibility: the disconnected
    straight term :math:`\delta_{\omega_{\mathrm{ph}} 0}\, \beta\, G^{\mathrm{DMFT}}_{12}(\nu)\,
    G^{\mathrm{DMFT}}_{34}(\nu')` would land exactly on the pp anti-diagonal :math:`\nu' = -\nu` and corrupt the
    :math:`\chi^{\mathrm{pp}}_0 J` rung. The loader guarantees this: :func:`~dgamore.local_sde.create_generalized_chi`
    subtracts that term in the density channel, and the :math:`\frac{1}{2}(\chi^{\mathrm{ph}}_{d} -
    \chi^{\mathrm{ph}}_{m})` combination cancels both it and the vertical bubble exactly.

    In the pair basis that keeps the up and down lines apart, i.e. for the matrices
    :math:`S[X]_{(13\nu),(24\nu')} = X^{\mathrm{pp};\nu(-\nu')}_{1234}` with the ordinary matrix product and inverse,
    the expression is the up-down pp Bethe-Salpeter equation in the form of the ph one,

    .. math:: S[\Gamma^{\mathrm{pp}}_{\uparrow\downarrow}] = \beta^2 \left( S[\chi^{\mathrm{pp}}_{\uparrow\downarrow}
        - \chi^{\mathrm{pp}}_0 J]^{-1} - S[-\chi^{\mathrm{pp}}_0 J]^{-1} \right),

    where :math:`-\chi^{\mathrm{pp}}_0 J` is the disconnected straight term of the up-down susceptibility in pp
    notation. Only :math:`J \chi^{\mathrm{pp}}_0 J = \chi^{\mathrm{pp}}_0` enters, which holds for any Green's
    function: neither :math:`G^{\mathrm{DMFT}}_{12}(\nu) = G^{\mathrm{DMFT}}_{21}(\nu)` nor crossing symmetry of
    :math:`\chi^{\mathrm{pp}}_{\uparrow\downarrow}` is assumed. For a crossing-symmetric susceptibility :math:`J`
    commutes with every pp object, so this is also the full-space form of inverting the decoupled singlet/triplet BSEs
    (thesis Eqs. 3.51/3.52) on their :math:`J`-even/odd blocks. For a single band :math:`J` reduces to the plain
    frequency flip and the expression is equivalent to Eq. (B.26) of Rohringer's thesis.

    :param gchi_ud_pp_w0: The local connected up-down susceptibility
        :math:`\chi^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234}` in pp notation at :math:`\omega = 0`, see
        :meth:`~dgamore.local_four_point.LocalFourPoint.change_frequency_notation_ph_to_pp_w0`.
    :param gchi0_pp_w0: The local bare pp bubble :math:`\chi^{\mathrm{pp};\nu\nu'}_{0;1234}` (diagonal in
        :math:`\nu\nu'`), built from the DMFT Green's function via
        :meth:`~dgamore.bubble_gen.BubbleGenerator.create_generalized_chi0_pp_w0`.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The vertex :math:`\Gamma^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234}` as a :class:`LocalFourPoint` in
        pp notation.
    """
    # chi0 * J in tensor form: the bubble with the second fermionic frequency flipped and the orbitals permuted
    gchi0_j = gchi0_pp_w0.flip_frequency_axis(-1).permute_orbitals("abcd->adcb", copy=False).to_half_niw_range()
    return (
        (gchi0_j - gchi0_pp_w0 @ gchi_ud_pp_w0.invert() @ gchi0_pp_w0)
        .invert()
        .scale(beta**2)
        .set_channel(SpinChannel.UD)
    )


def create_local_gamma_ud_pp_w0_per_ineq(
    gchi_ud_pp_w0: LocalFourPoint, g_dmft: GreensFunction, beta: float
) -> LocalFourPoint:
    r"""
    Builds the local pp-irreducible up-down vertex :math:`\Gamma^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234}` per
    inequivalent atom and assembles the per-atom blocks into the full multi-band object (mirroring the local
    Schwinger-Dyson assembly). Local correlations do not connect orbitals of different atoms, so the assembled
    multi-band susceptibility is nonzero only when all four orbital indices belong to the same atom; the compound pp
    matrix of the FULL object is therefore singular for more than one atom and must never be inverted directly. Instead,
    :func:`create_local_gamma_ud_pp_w0` is evaluated on each atom's orbital block (with the bare pp bubble built from
    that atom's block of :math:`G^{\mathrm{DMFT}}_{12}(\nu)`), computing every inequivalent atom only once and writing
    the result into all of its positions in the compound band layout.

    :param gchi_ud_pp_w0: The full multi-band connected up-down susceptibility
        :math:`\chi^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234}` in pp notation at :math:`\omega = 0`
        (block-structured per inequivalent atom).
    :param g_dmft: The full multi-band local DMFT :class:`GreensFunction` :math:`G^{\mathrm{DMFT}}_{12}(\nu)`.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The assembled vertex :math:`\Gamma^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234}` as a
        :class:`LocalFourPoint` in pp notation (nonzero only on the same-atom orbital blocks).
    """
    n_bands = gchi_ud_pp_w0.n_bands
    if config.dmft.n_bands_per_ineq and config.dmft.ineq_ordering:
        layout = []
        n_start = 0
        for ineq in config.dmft.ineq_ordering:
            n_end = n_start + config.dmft.n_bands_per_ineq[ineq - 1]
            layout.append((ineq, slice(n_start, n_end)))
            n_start = n_end
    else:
        layout = [(1, slice(0, n_bands))]

    gamma_full = gchi_ud_pp_w0._clone_without_mat()
    gamma_full.mat = np.zeros(gchi_ud_pp_w0.current_shape, dtype=gchi_ud_pp_w0.mat.dtype)
    gamma_full.update_original_shape()

    gamma_per_ineq: dict[int, LocalFourPoint] = {}
    for ineq, sl in layout:
        if ineq not in gamma_per_ineq:
            gchi_block = LocalFourPoint(
                gchi_ud_pp_w0.mat[sl, sl, sl, sl].copy(),
                SpinChannel.UD,
                1,
                2,
                gchi_ud_pp_w0.full_niw_range,
                gchi_ud_pp_w0.full_niv_range,
                FrequencyNotation.PP,
            )
            g_mat_block = g_dmft.mat[..., sl, sl, :]
            g_block = GreensFunction(g_mat_block.reshape((1, 1, 1) + g_mat_block.shape[-3:]).copy())
            gchi0_block = BubbleGenerator.create_generalized_chi0_pp_w0(
                g_block, gchi_block.niv, beta
            ).extend_vn_to_diagonal()
            gamma_per_ineq[ineq] = create_local_gamma_ud_pp_w0(gchi_block, gchi0_block, beta)
        gamma_full.mat[sl, sl, sl, sl] = gamma_per_ineq[ineq].mat

    return gamma_full.set_channel(SpinChannel.UD)


def create_local_ud_diagrams_pp_w0(
    g_dmft: GreensFunction, niv_pp: int
) -> tuple[LocalFourPoint, LocalFourPoint, LocalFourPoint]:
    r"""
    Builds the local particle-particle reducible diagrams at :math:`\omega = 0` in the up-down channel: the full vertex
    :math:`F^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234}`, the pp-irreducible vertex
    :math:`\Gamma^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234}` (built per inequivalent atom and assembled into the
    full multi-band object, see :func:`create_local_gamma_ud_pp_w0_per_ineq`), and the reducible part

    .. math:: \Phi^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234} = F^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234}
        - \Gamma^{\mathrm{pp};\nu\nu'}_{\uparrow\downarrow;1234},

    with :math:`\chi^{\mathrm{pp}}_{\uparrow\downarrow} = \frac{1}{2}(\chi^{\mathrm{ph}}_{d} - \chi^{\mathrm{ph}}_{m})`
    mapped to pp notation at :math:`\omega_{\mathrm{pp}} = 0` via
    :meth:`~dgamore.local_four_point.LocalFourPoint.change_frequency_notation_ph_to_pp_w0` (ph legs evaluated at
    :math:`\omega_{\mathrm{ph}} = \nu + \nu'`) and the bare pp bubble built from the local DMFT Green's function
    :math:`G^{\mathrm{DMFT}}_{12}(\nu)` via :meth:`~dgamore.bubble_gen.BubbleGenerator.create_generalized_chi0_pp_w0`.
    The pairing vertex subtracts :math:`\Phi` from its ladder part, which makes it pp-irreducible, together with the
    local full vertex of :func:`create_local_f_ud_transformed_w0`, which both ladders contain (thesis Eqs. 4.49-4.52).

    :param g_dmft: The local DMFT :class:`GreensFunction` :math:`G^{\mathrm{DMFT}}_{12}(\nu)`.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex; the local diagrams are cut to this
        box so they always match the ladder pairing vertex, also when ``niw_core > niv_core``.
    :return: The tuple ``(f_ud_loc_pp_w0, gamma_ud_loc_pp_w0, phi_ud_loc_pp_w0)`` of local pp diagrams at
        :math:`\omega = 0`.
    """
    gchi_dens_loc = LocalFourPoint.load(
        output_files.npy_path(config.output.output_path, output_files.local_name("gchi", SpinChannel.DENS)),
        SpinChannel.DENS,
    )
    gchi_magn_loc = LocalFourPoint.load(
        output_files.npy_path(config.output.output_path, output_files.local_name("gchi", SpinChannel.MAGN)),
        SpinChannel.MAGN,
    )
    gchi_ud_loc = (gchi_dens_loc - gchi_magn_loc).set_channel(SpinChannel.UD).scale(0.5)
    # the fermionic cut commutes with the w0 pp map (it samples only the retained window), so cutting first shrinks
    # the transform transient from the full box to the pp box
    gchi_ud_loc_pp_w0 = gchi_ud_loc.cut_niv(niv_pp).change_frequency_notation_ph_to_pp_w0()
    del gchi_dens_loc, gchi_magn_loc, gchi_ud_loc

    gamma_ud_loc_pp_w0 = create_local_gamma_ud_pp_w0_per_ineq(gchi_ud_loc_pp_w0, g_dmft, config.sys.beta)
    del gchi_ud_loc_pp_w0

    gamma_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="gamma_ud_loc_pp_w0")

    f_dens_loc = LocalFourPoint.load(
        output_files.npy_path(config.output.output_path, output_files.local_name("f", SpinChannel.DENS)),
        SpinChannel.DENS,
    )
    f_magn_loc = LocalFourPoint.load(
        output_files.npy_path(config.output.output_path, output_files.local_name("f", SpinChannel.MAGN)),
        SpinChannel.MAGN,
    )
    f_ud_loc = (f_dens_loc - f_magn_loc).set_channel(SpinChannel.UD).scale(0.5)
    f_ud_loc_pp_w0 = f_ud_loc.cut_niv(niv_pp).change_frequency_notation_ph_to_pp_w0()

    del f_dens_loc, f_magn_loc, f_ud_loc

    phi_ud_loc_pp_w0 = f_ud_loc_pp_w0 - gamma_ud_loc_pp_w0
    phi_ud_loc_pp_w0 = phi_ud_loc_pp_w0.take_first_wn()
    f_ud_loc_pp_w0 = f_ud_loc_pp_w0.take_first_wn()

    return f_ud_loc_pp_w0, gamma_ud_loc_pp_w0, phi_ud_loc_pp_w0


def create_local_f_ud_transformed_w0(niv_pp: int) -> LocalFourPoint:
    r"""
    Loads the local full vertex of both particle-hole channels from file and returns the up-down combination

    .. math:: F^{\omega\nu\nu'}_{\uparrow\downarrow;1234} = \frac{1}{2}\left(F^{\omega\nu\nu'}_{d;1234}
        - F^{\omega\nu\nu'}_{m;1234}\right)

    in the modified particle-particle notation at :math:`\omega' = 0` (see
    :func:`transform_vertex_loc_frequencies_w0`). This is the local full vertex subtracted from each ladder slot of
    the pairing vertex (thesis Eqs. 4.49-4.52); it carries a different frequency notation than the other local pp
    diagrams of :func:`create_local_ud_diagrams_pp_w0`. The loaded vertices carry both fermionic indices on the core
    box (only the double-counting kernel needs the summed index on the full asymptotic box, and it reads its own
    file for that), yet they remain among the largest objects of this step, so the caller reduces them once per node.

    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :return: The transformed vertex as a :class:`LocalFourPoint` (channel UD, pp notation, no bosonic axis).
    """
    f_dens_loc = LocalFourPoint.load(
        output_files.npy_path(config.output.output_path, output_files.local_name("f", SpinChannel.DENS)),
        SpinChannel.DENS,
    )
    f_magn_loc = LocalFourPoint.load(
        output_files.npy_path(config.output.output_path, output_files.local_name("f", SpinChannel.MAGN)),
        SpinChannel.MAGN,
    )
    f_ud_loc = (f_dens_loc - f_magn_loc).set_channel(SpinChannel.UD).scale(0.5)
    del f_dens_loc, f_magn_loc
    return transform_vertex_loc_frequencies_w0(f_ud_loc, niv_pp)


def dispatch_full_vertex_calculation(
    channel: SpinChannel,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    niv_pp: int,
    mpi_dist: MpiDistributor,
    chunk_bytes: int | None = None,
    node_comm: MPI.Comm | None = None,
    gchi0_q_inv: FourPoint | None = None,
) -> FourPoint:
    r"""
    Loads the local irreducible vertex for ``channel`` and builds the full ladder pp vertex through the slice-direct
    construction, streaming the ph-notation vertex to disk when ``save_fq`` is set. Please note that Eq. (4.43) in
    my master's thesis is wrong. The correct formula is
    :math:`F^{\mathrm{q}\nu\nu'}_{r;1234}=F^{(1);\mathrm{q}\nu\nu'}_{r;1234}+F^{(2);\mathrm{q}\nu\nu'}_{r;1234}`, with
    :math:`F^{(1);\mathrm{q}\nu\nu'}_{r;1234} = \beta^2\Big[(\chi^{\mathrm{q}\nu\nu'}_{0;1234})^{-1}-
    \sum_{\nu_1\nu_2}\sum_{abcd}(\chi^{\mathrm{q}\nu\nu_1}_{0;12ab})^{-1}\chi^{*;\mathrm{q}\nu_1\nu_2}_{r;bacd}(\chi^{\mathrm{q}\nu_2\nu'}_{0;dc34})^{-1}\Big]`
    and :math:`F^{(2);\mathrm{q}\nu\nu'}_{r;1234} = \sum_{abcdgh}\gamma^{\mathrm{q}\nu}_{r;12ab}\Big(\mathbb{1}_{bacd} -
    \sum_{ef}\mathcal{U}^{\mathbf{q}}_{r;baef}\chi^{\mathrm{q}}_{r;fecd}\Big)\mathcal{U}^{\mathbf{q}}_{r;dcgh}\tilde\gamma^{\mathrm{q}\nu'}_{r;hg34}`,
    where :math:`\tilde\gamma^{\mathrm{q}\nu}_{r;1234}=\beta \sum_{ab}\sum_{\nu'} \chi^{*;\mathrm{q}\nu'\nu}_{r;12ab}
    (\chi^{\mathrm{q}\nu}_{0;ba34})^{-1}` sums over the FIRST frequency argument. It equals the orbital-reversed sum
    over the last one, :math:`\beta \sum_{ab}\sum_{\nu'} \chi^{*;\mathrm{q}\nu\nu'}_{r;ab21}
    (\chi^{\mathrm{q}\nu}_{0;ab34})^{-1}`, only where the Bethe-Salpeter matrix is complex-symmetric, i.e. where
    :math:`G^{\mathrm{k}} = (G^{\mathrm{k}})^{T}`; complex hoppings or a lattice without inversion break that symmetry
    of the lattice bubble, so the build takes the first-frequency sum from its own inversion (see
    :func:`_build_ladder_vertex_chunk`). No explicit factors of :math:`\beta`
    appear in :math:`F^{(2)}` because they are absorbed into the stored objects: :math:`\chi^{\mathrm{q}}_{r}` is the
    (:math:`U`-dressed, shell- (and sometimes :math:`\lambda`-corrected)) physical susceptibility normalized as
    :math:`\frac{1}{\beta^2}\sum_{\nu\nu'}\chi^{\mathrm{q}\nu\nu'}_{r}`, and the three-leg vertices carry the net
    normalization :math:`\gamma^{\mathrm{q}\nu}_{r} = (\chi^{\mathrm{q}\nu}_{0})^{-1}\sum_{\nu'}
    \chi^{*;\mathrm{q}\nu\nu'}_{r}` (the explicit :math:`\beta` in their construction cancels the :math:`1/\beta` of the
    fused frequency sum), such that :math:`\gamma^{\mathrm{q}\nu}_{r} \to \mathbb{1}` for :math:`\nu \to \infty`. In the
    bare-vertex modes (``config.misc.do_flex``, ``config.misc.do_rpa``) no local vertex is loaded and the vertex is the
    RPA one, :math:`F^{\mathrm{q}}_{r} = \mathcal{U}^{\mathbf{q}}_{r} -
    \mathcal{U}^{\mathbf{q}}_{r}\chi^{\mathrm{q}}_{r}\mathcal{U}^{\mathbf{q}}_{r}` (see
    :func:`_build_rpa_vertex_chunk`).

    :param channel: The spin channel (density or magnetic).
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param mpi_dist: MPI distributor over the irreducible BZ q-points.
    :param chunk_bytes: Chunk byte budget of the build (``None`` uses the floor).
    :param node_comm: Optional node-local communicator; when given, the multi-GB local vertex is loaded once per
        node into an MPI shared-memory window instead of once per rank (the build only reads it).
    :param gchi0_q_inv: The inverse core bubble of this rank's momenta (see
        :func:`~dgamore.eliashberg_solver.inverse_core_bubble`), or ``None`` to load the saved one; not read in the
        bare-vertex modes.
    :return: The full ladder pp vertex :math:`F^{\mathrm{q}}_{r}` as a :class:`FourPoint`.
    """
    if config.misc.bare_vertex:
        build = create_pairing_vertex_streaming_fq if config.eliashberg.save_fq else create_pairing_vertex_slice_q_r
        f_q_r = build(u_loc, v_nonloc, None, niv_pp, mpi_dist, chunk_bytes, channel=channel)
        mpi_dist.barrier()
        return f_q_r
    gamma_r, gamma_win = sde_kernels.load_node_shared_local_vertex(
        node_comm, output_files.npy_path(config.output.output_path, output_files.local_name("gamma", channel)), channel
    )
    build = create_pairing_vertex_streaming_fq if config.eliashberg.save_fq else create_pairing_vertex_slice_q_r
    f_q_r = build(u_loc, v_nonloc, gamma_r, niv_pp, mpi_dist, chunk_bytes, gchi0_q_inv=gchi0_q_inv)
    gamma_r.mat = None
    if gamma_win is None:
        gamma_r.free()
    mpi_utils.free_shared_window(gamma_win, node_comm)
    mpi_dist.barrier()
    return f_q_r

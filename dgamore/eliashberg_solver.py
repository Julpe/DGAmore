# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Linearized Eliashberg equation solver. Starting from the ladder-DGA full vertex (saved per channel by the non-local SDE
step), this module assembles the particle-particle pairing vertex in the singlet/triplet channels at :math:`\omega = 0`,
optionally adds the local reducible diagrams, and solves the linearized gap equation :math:`\lambda \Delta =
\pm\frac{1}{2\beta n_{\mathbf{k}}}\, \Gamma^{\mathrm{pp}}\, \chi_0^{\mathrm{pp}}\, \Delta` with a matrix-free
ARPACK/Lanczos eigensolver: in memory when a node holds one channel's full-BZ pairing vertex (as a team solve over
one node-shared vertex window per channel on a multi-rank job, sector after sector on a single rank) and on a
block-distributed frequency grid otherwise. The leading eigenvalue :math:`\lambda` signals the pairing instability
and the eigenvector is the gap function :math:`\Delta^{\mathrm{k}}_{12}`. Equation numbers refer to the author's
master's thesis (Chapter 4).
"""

import os
import socket
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import mpi4py.MPI as MPI
import numpy as np
import scipy as sp
from threadpoolctl import ThreadpoolController, threadpool_limits

import dgamore.config as config
from dgamore import nonlocal_sde, mpi_utils
from dgamore.brillouin_zone import KGrid, KnownSymmetries
from dgamore.bubble_gen import BubbleGenerator
from dgamore.four_point import FourPoint
from dgamore.gap_function import GapFunction
from dgamore.greens_function import GreensFunction
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.local_four_point import LocalFourPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore import memory_estimator
from dgamore.memory_estimator import (
    SLICE_CHUNK_BYTES,
    lanczos_ncv,
    lanczos_solver_bytes,
    lanczos_team_bytes,
    solver_grid_shape,
    team_build_columns,
)
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel, FrequencyNotation, DTYPE, deferred_collection
from dgamore.self_energy import SelfEnergy
from dgamore.symmetry_reduction import (
    _M_preserves_grid,
    _close_group,
    find_coordinate_mirror_orbital_unitaries,
    find_inversion_orbital_unitary,
    forced_multiplicity,
    point_group_orbits,
)

EVEN_SECTOR_DEGENERACY = 1e-2  # largest relative difference of the leading singlet-even and triplet-even eigenvalues
# at which even_sectors_degenerate reports them as coinciding
SECTOR_RESIDUAL_WARNING = 1e-4  # residual (over max(|rho|, 1)) of a sector's leading gap under the unprojected kernel
# above which (or above ten times a larger eigensolver tolerance) the parity sectors are reported as no symmetry of it
LANCZOS_RESTARTS = 3000  # restart cap of one eigensolver attempt, after which _solve_loosening retries the sector
TOLERANCE_LOOSENING = 1e2  # factor of each of the at most two looser retries of an unconverged sector


def delete_files(filepath: str, *args) -> None:
    """
    Deletes files in the given directory. If a file is not found, it is ignored. The deleted files are usually
    temporary files that are no longer needed after the calculation is done.

    :param filepath: Directory containing the files.
    :param args: One or more file names (relative to ``filepath``) to delete.
    :return: None.
    :raises TypeError: If any of the given names is not a string.
    """
    for name in args:
        if not isinstance(name, str):
            raise TypeError(f"Expected string, got {type(name)}.")
        full_path = os.path.join(filepath, name)
        if os.path.isfile(full_path):
            try:
                os.remove(full_path)
            except OSError:
                config.logger.info(f"Error deleting file: {name}.")


def _compute_once_per_node(node_comm: MPI.Comm | None, compute_fn):
    """
    Evaluates ``compute_fn`` on the node-local root rank only and broadcasts its (small) result to the node's other
    ranks. Used for quantities whose result is cheap but whose construction holds a multi-GB transient, so that
    transient exists once per node instead of once per rank. A ``None`` communicator evaluates locally.

    :param node_comm: The node-local communicator (e.g. from ``comm.Split_type(MPI.COMM_TYPE_SHARED)``), or ``None``.
    :param compute_fn: Zero-argument callable returning the result; invoked only on the node root.
    :return: The node root's result, on every rank of the node.
    """
    if node_comm is None:
        return compute_fn()
    result = compute_fn() if node_comm.Get_rank() == 0 else None
    return node_comm.bcast(result, root=0)


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
    chi_phys_q_r: FourPoint,
    u_loc: LocalInteraction,
    u_r: Interaction,
    w_start: int,
    w_stop: int,
    niv_pp: int | None,
    inactive_pairs: np.ndarray | None = None,
) -> FourPoint:
    r"""
    Builds the full ladder vertex :math:`F^{\mathrm{q}}_{r}` for the momenta of the inputs and the bosonic window
    ``[w_start, w_stop)``, i.e. the amputated auxiliary susceptibility plus the separable interaction part. The
    right-sided three-leg vertex :math:`\tilde\gamma^{\mathrm{q}\nu'}_{r;1234} = \beta \sum_{ab}\sum_{\nu}
    \chi^{*;\mathrm{q}\nu\nu'}_{r;12ab} (\chi^{\mathrm{q}\nu'}_{0;ba34})^{-1}` of the separable part sums the
    auxiliary susceptibility over its first frequency, which the build takes from the same inversion (exact also where
    the Bethe-Salpeter matrix is not complex-symmetric). The Bethe-Salpeter matrix is built one compound slice at a
    time from the local vertex and the inverse bubble, never as a whole window.

    With ``niv_pp`` given, only what the :math:`\omega' = 0` pp band reads is built: the auxiliary susceptibility on
    the anti-diagonal :math:`\nu + \nu' = \omega` of the pp box (see
    :meth:`FourPoint.invert_with_local_kernel_on_anti_diagonal`, with the inverse bubble as the frequency-diagonal
    part, the kernel :math:`\Gamma_{r}`, the shift :math:`U_{r}` and the scale :math:`1/\beta^2`), and the whole
    chain on the pp box, so the result is exact on that anti-diagonal only. With ``niv_pp`` None the whole core box
    is inverted (see :meth:`FourPoint.invert_with_local_kernel`) and every entry is exact (the streamed full
    vertex).

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel, from :math:`\omega = 0` on
        (half bosonic range) or on the full range.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` of the momenta.
    :param vrg_q_r_left: The three-leg vertex :math:`\gamma^{\mathrm{q}\nu}_{r}` of the momenta.
    :param chi_phys_q_r: The physical susceptibility :math:`\chi^{\mathrm{phys};\mathrm{q}}_{r}` of the momenta.
    :param u_loc: The bare local interaction :math:`U`.
    :param u_r: The channel-projected total interaction :math:`\mathcal{U}^{\mathbf{q}}_{r}`.
    :param w_start: First bosonic index of the window.
    :param w_stop: One past the last bosonic index of the window.
    :param niv_pp: Number of positive fermionic frequencies of the pp box, or None for the whole core box.
    :param inactive_pairs: Orbital pairs without vertex (see :meth:`LocalFourPoint.orbital_pairs_without_vertex`).
    :return: The ladder vertex over that window as a :class:`FourPoint` (half niw range, two fermionic dimensions).
    """
    beta = config.sys.beta
    u_r_loc = u_loc.as_channel(gamma_r.channel)
    gchi0_w = gchi0_q_inv.take_wn_slice(w_start, w_stop)
    vrg_left_w = vrg_q_r_left.take_wn_slice(w_start, w_stop)
    if niv_pp is None:
        chi_aux = gchi0_w.invert_with_local_kernel(gamma_r, u_r_loc, 1.0 / beta**2, w_start)
        chi_aux_first_sum = chi_aux.sum_over_vn(beta, axis=(-2,))
    else:
        chi_aux, chi_aux_first_sum = gchi0_w.invert_with_local_kernel_on_anti_diagonal(
            gamma_r, u_r_loc, 1.0 / beta**2, beta, niv_pp, w_start, inactive_pairs
        )
        gchi0_w = gchi0_w.cut_niv(niv_pp)
        vrg_left_w = vrg_left_w.cut_niv(niv_pp)

    # chi* is released right after the first matmul and its product by rebinding; the bubble term enters in place
    f_chunk = gchi0_w @ chi_aux
    del chi_aux
    f_chunk = f_chunk @ gchi0_w
    f_chunk = f_chunk.scale(-(beta**2)).add_on_vn_diagonal(gchi0_w, factor=beta**2)

    vrg_right_w = (chi_aux_first_sum @ gchi0_w).scale(beta)
    chi_phys_w = chi_phys_q_r.take_wn_slice(w_start, w_stop)
    return f_chunk.add((vrg_left_w @ u_r - vrg_left_w @ (u_r @ chi_phys_w @ u_r)) * vrg_right_w, copy=False)


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
) -> FourPoint:
    r"""
    Shared loop of the slice-direct pairing-vertex construction: walks the rank-local momenta and, per momentum, the
    bosonic axis in byte-bounded chunks, building each ladder-vertex chunk once and reading its :math:`\omega' = 0`
    pp band (see :func:`_pp_w0_band`). When ``chunk_writer`` is given, every ph-notation chunk is handed to it before
    being released, so a caller can stream the full vertex to disk in the same pass. Without a local vertex the
    chunks are the RPA vertex of the saved physical susceptibility (see :func:`_build_rpa_vertex_chunk`), the only
    intermediate read then.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel, or ``None`` for the RPA vertex.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param chunk_writer: Optional callable ``(q_start, w_start, chunk_mat)`` receiving each ph-notation chunk (the
        momenta of the group as the leading axis).
    :param chunk_bytes: Chunk byte budget of the build (``None`` uses the floor).
    :param channel: Spin channel of the RPA vertex, i.e. density or magnetic; only read when ``gamma_r`` is ``None``.
    :return: The pairing vertex :math:`F^{\mathrm{q}}_{r}` in pp notation as a :class:`FourPoint`.
    """
    logger = config.logger
    channel = channel if gamma_r is None else gamma_r.channel
    path, rank = config.output.eliashberg_path, mpi_dist_irrk.my_rank

    if gamma_r is not None:
        gchi0_q_inv = FourPoint.load(os.path.join(path, f"gchi0_q_inv_rank_{rank}.npy"), num_vn_dimensions=1)
        vrg_q_r_left = FourPoint.load(
            os.path.join(path, f"vrg_q_{channel.value}_rank_{rank}.npy"), channel=channel, num_vn_dimensions=1
        )
    chi_phys_q_r = FourPoint.load(
        os.path.join(path, f"chi_phys_q_{channel.value}_rank_{rank}.npy"), channel=channel, num_vn_dimensions=0
    )
    logger.info(f"Loaded the intermediates for the {channel.value} pairing vertex.")

    my_irr_q_list = config.lattice.k_grid.get_irrq_list()[mpi_dist_irrk.my_slice]
    niw_stored = chi_phys_q_r.current_shape[-1] - 1
    _, omega = _pp_w0_band(niv_pp, niw_stored)
    # the pp band only reads |w| <= 2 niv_pp - 1, but a streamed full vertex must cover the whole stored box
    niw_build = niw_stored if chunk_writer is not None else int(np.max(omega))

    n_bands = config.sys.n_bands
    # orbital pairs without vertex are eliminated from every slice, as in the kernel step's auxiliary susceptibility
    inactive = None if gamma_r is None else gamma_r.orbital_pairs_without_vertex(u_loc.as_channel(channel))
    f_pp_mat = np.zeros((len(my_irr_q_list),) + (n_bands,) * 4 + (2 * niv_pp,) * 2, dtype=chi_phys_q_r.mat.dtype)

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
            chi_phys_grp = chi_phys_q_r.take_q_index_slice(q_start, q_stop)
            v_nonloc_grp = v_nonloc.take_q_index_slice(q_start, q_stop)
            u_r = u_loc.as_channel(channel) + v_nonloc_grp.as_channel(channel)

            for w_start in range(0, niw_build + 1, w_chunk):
                w_stop = min(w_start + w_chunk, niw_build + 1)
                if gamma_r is None:
                    f_chunk = _build_rpa_vertex_chunk(chi_phys_grp, u_r, w_start, w_stop, niv_window)
                else:
                    f_chunk = _build_ladder_vertex_chunk(
                        gamma_r,
                        gchi0_grp,
                        vrg_left_grp,
                        chi_phys_grp,
                        u_loc,
                        u_r,
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
            chi_phys_grp.free()

    if gamma_r is not None:
        gchi0_q_inv.free()
        vrg_q_r_left.free()
    chi_phys_q_r.free()

    delete_files(
        path,
        f"vrg_q_{channel.value}_rank_{rank}.npy",
        f"chi_phys_q_{channel.value}_rank_{rank}.npy",
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
    :return: The pairing vertex :math:`F^{\mathrm{q}}_{r}` in pp notation as a :class:`FourPoint`.
    """
    return _build_pairing_vertex_pp(
        u_loc, v_nonloc, gamma_r, niv_pp, mpi_dist_irrk, chunk_bytes=chunk_bytes, channel=channel
    )


def create_pairing_vertex_streaming_fq(
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    gamma_r: LocalFourPoint | None,
    niv_pp: int,
    mpi_dist_irrk: MpiDistributor,
    chunk_bytes: int | None = None,
    channel: SpinChannel | None = None,
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
        u_loc, v_nonloc, gamma_r, niv_pp, mpi_dist_irrk, chunk_writer, chunk_bytes, channel=channel
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

    :math:`J` commutes with every pp object by crossing symmetry, so this is the full-space form of inverting the
    decoupled singlet/triplet BSEs (thesis Eqs. 3.51/3.52) on their :math:`J`-even/odd blocks. For a single band
    :math:`J` reduces to the plain frequency flip and the expression is equivalent to Eq. (B.26) of Rohringer's thesis.
    Assumes :math:`G^{\mathrm{DMFT}}_{12}(\nu) = G^{\mathrm{DMFT}}_{21}(\nu)` (real orbital basis, no spin-orbit
    coupling); with SOC the rung :math:`\chi^{\mathrm{pp}}_0 J` must be replaced by :math:`-\beta\,
    G^{\mathrm{DMFT}}_{12}(\nu)\, G^{\mathrm{DMFT}}_{43}(-\nu)\, \delta_{\nu,-\nu'}` (second Green's function
    transposed).

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
    These are the local diagrams always subtracted/added to the pairing vertex; see
    :class:`~dgamore.config.EliashbergConfig` is enabled, to avoid double counting the local pairing contribution
    (thesis Eqs. 4.49-4.52).

    :param g_dmft: The local DMFT :class:`GreensFunction` :math:`G^{\mathrm{DMFT}}_{12}(\nu)`.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex; the local diagrams are cut to this
        box so they always match the ladder pairing vertex, also when ``niw_core > niv_core``.
    :return: The tuple ``(f_ud_loc_pp_w0, gamma_ud_loc_pp_w0, phi_ud_loc_pp_w0)`` of local pp diagrams at
        :math:`\omega = 0`.
    """
    gchi_dens_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"gchi_dens_loc.npy"), SpinChannel.DENS)
    gchi_magn_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"gchi_magn_loc.npy"), SpinChannel.MAGN)
    gchi_ud_loc = (gchi_dens_loc - gchi_magn_loc).set_channel(SpinChannel.UD).scale(0.5)
    # the fermionic cut commutes with the w0 pp map (it samples only the retained window), so cutting first shrinks
    # the transform transient from the full box to the pp box
    gchi_ud_loc_pp_w0 = gchi_ud_loc.cut_niv(niv_pp).change_frequency_notation_ph_to_pp_w0()
    del gchi_dens_loc, gchi_magn_loc, gchi_ud_loc

    gamma_ud_loc_pp_w0 = create_local_gamma_ud_pp_w0_per_ineq(gchi_ud_loc_pp_w0, g_dmft, config.sys.beta)
    del gchi_ud_loc_pp_w0

    gamma_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="gamma_ud_loc_pp_w0")

    f_dens_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"f_dens_loc.npy"), SpinChannel.DENS)
    f_magn_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"f_magn_loc.npy"), SpinChannel.MAGN)
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
    f_dens_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"f_dens_loc.npy"), SpinChannel.DENS)
    f_magn_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"f_magn_loc.npy"), SpinChannel.MAGN)
    f_ud_loc = (f_dens_loc - f_magn_loc).set_channel(SpinChannel.UD).scale(0.5)
    del f_dens_loc, f_magn_loc
    return transform_vertex_loc_frequencies_w0(f_ud_loc, niv_pp)


# --- Gap initialization ---
def get_initial_gap_function(shape: tuple, channel: SpinChannel) -> np.ndarray:
    """
    Generates the initial gap-function guess for the power iteration, seeded with the configured momentum symmetry
    (d-wave / p-wave-x / p-wave-y) and the corresponding frequency parity for the singlet/triplet channel; falls back
    to a random guess if no symmetry is configured or recognized.

    The random guess is drawn from a generator with a fixed seed rather than from the global ``np.random``
    state: an unseeded start vector makes a run irreproducible, and inside a degenerate multiplet - where the
    eigensolver is free to return any basis of the eigenspace - it changes which partners come back from one run to
    the next. It also keeps the seed identical on every rank without a broadcast.

    For a configured symmetry the negative-frequency half is set from the positive one reflected under the
    fermionic-frequency flip T (``np.flip`` over the last axis, pairing ``niv + j`` with ``niv - 1 - j``) and signed
    by the channel's required parity, so the seed is a T eigenvector even for a frequency-dependent form factor.

    :param shape: Target array shape ``[kx, ky, kz, o1, o2, v]`` of the gap function.
    :param channel: Pairing channel, either :attr:`SpinChannel.SING` or :attr:`SpinChannel.TRIP`.
    :return: The initial gap-function array (dtype ``DTYPE`` either way).
    :raises ValueError: If ``channel`` is neither SING nor TRIP.
    """
    if channel not in {SpinChannel.SING, SpinChannel.TRIP}:
        raise ValueError("Channel must be either SING or TRIP.")

    symmetry = config.eliashberg.symmetry
    symm = {
        "d-wave": lambda k: -np.cos(k[0])[:, None, None] + np.cos(k[1])[None, :, None],
        "p-wave-x": lambda k: np.sin(k[0])[:, None, None],
        "p-wave-y": lambda k: np.sin(k[1])[None, :, None],
    }
    v_sym = {
        "d-wave": "even" if channel == SpinChannel.SING else "odd",
        "p-wave-x": "odd" if channel == SpinChannel.SING else "even",
        "p-wave-y": "odd" if channel == SpinChannel.SING else "even",
    }

    if symmetry not in symm:
        return np.random.default_rng(42).random(shape).astype(DTYPE)

    niv = shape[-1] // 2
    gap0 = np.zeros(shape, dtype=DTYPE)
    gap0[..., niv:] = np.repeat(symm[symmetry](config.lattice.k_grid.grid)[:, :, :, None, None, None], niv, axis=-1)
    # reflect the positive-frequency half under T (np.flip) and sign it with the channel's parity
    gap0[..., :niv] = (1 if v_sym[symmetry] == "even" else -1) * np.flip(gap0[..., niv:], axis=-1)
    return gap0


# --- Physical-gap symmetry sectors (parity under T O_U + forced inversion parity) ---
def even_sectors_degenerate(results: dict) -> bool:
    r"""
    Returns whether the leading even-frequency singlet and triplet eigenvalues agree to the relative tolerance
    ``EVEN_SECTOR_DEGENERACY``, i.e. to 1 %. A gap counts as even-frequency when most of its weight is even in the
    frequency (see :func:`frequency_parts`): for one band these are the gaps of the even sectors, while with hybridized
    orbitals either parity sector can hold one (an orbital-antisymmetric even-frequency triplet is odd under
    :math:`T O_U`). False without parity resolution or without an even-frequency gap in either channel.

    :param results: The solver output ``{(channel, parity): (lambdas, gaps)}`` with its gaps (rank 0).
    :return: Whether the two leading even-frequency eigenvalues coincide.
    """
    leading = []
    for channel in (SpinChannel.SING, SpinChannel.TRIP):
        even = [
            float(np.real(lam))
            for parity in ("even", "odd")
            for lam, gap in zip(*results.get((channel, parity), ((), ())))
            if _even_frequency_weight(gap.mat) > 0.5
        ]
        if not even:
            return False
        leading.append(max(even))
    return abs(leading[0] - leading[1]) <= EVEN_SECTOR_DEGENERACY * max(abs(leading[0]), abs(leading[1]))


def _parity_sectors(resolve_parity_sectors: bool) -> list[tuple[str, int | None]]:
    r"""
    Returns the list of gap sectors to solve, each a ``(label, eps_T)`` pair where ``eps_T`` is the requested parity
    under the frequency flip times the orbital exchange, :math:`T O_U` (:math:`+1` even, :math:`-1` odd; the frequency
    parity for one band, see :func:`_project_gap_to_sector`), or ``None`` for the unprojected case. When
    ``resolve_parity_sectors`` is set, the even and odd sectors are both returned; otherwise the single unprojected
    sector is returned. The paired inversion parity is fixed by the Pauli constraint ``eps_P = sign * eps_T`` inside
    the solver, so it is not carried here.

    :param resolve_parity_sectors: Whether to split the gap into the even and odd sectors.
    :return: The ``[(label, eps_T), ...]`` sector list.
    """
    return [("even", 1), ("odd", -1)] if resolve_parity_sectors else [("none", None)]


def _sector_log_label(channel: SpinChannel, parities: list[str] | None = None) -> str:
    """
    Returns the name of the sectors a solver call covers, for log messages. Sector-aware naming keeps concurrently
    solving ranks distinguishable in the log, where a channel-only name would repeat verbatim once per sector.

    :param channel: The spin channel being solved.
    :param parities: The subset of parity labels this call handles, or ``None`` for all of them.
    :return: ``"the singlet channel"`` when the parity projection is off, otherwise the covered sectors, e.g.
        ``"the singlet/even & odd sectors"`` or ``"the triplet/odd sector"``.
    """
    labels = [label for label, _ in _parity_sectors(config.eliashberg.resolve_parity_sectors)]
    if parities is not None:
        labels = [label for label in labels if label in parities]
    if labels == ["none"]:
        return f"the {channel.value}let channel"
    return f"the {channel.value}let/{' & '.join(labels)} sector{'' if len(labels) == 1 else 's'}"


def _local_dmft_self_energy(sigma_dmft_per_ineq: list[SelfEnergy]) -> np.ndarray:
    r"""
    Returns the local DMFT self-energy on the correlated orbitals, block diagonal over the atoms of
    ``config.dmft.ineq_ordering``.

    :param sigma_dmft_per_ineq: The DMFT :class:`SelfEnergy` of each inequivalent atom.
    :return: The self-energy, shape ``[o1, o2, v]``.
    """
    blocks = [sigma_dmft_per_ineq[ineq - 1].mat[0, 0, 0] for ineq in config.dmft.ineq_ordering]
    sigma = np.zeros((sum(len(block) for block in blocks),) * 2 + blocks[0].shape[-1:], dtype=complex)
    start = 0
    for block in blocks:
        sigma[start : start + len(block), start : start + len(block)] = block
        start += len(block)
    return sigma


def _inversion_action(sigma_dmft: np.ndarray | None = None) -> tuple[np.ndarray | None, str | None]:
    r"""
    Returns the orbital action of the inversion on the orbitals of the gap (the correlated ones), found on
    :math:`H(\mathbf{k})` by :func:`~dgamore.symmetry_reduction.find_inversion_orbital_unitary`, with the reason the
    parity sectors are not exact, ``None`` when they are. They need inversion to be a symmetry of the whole pairing
    problem: of :math:`H(\mathbf{k})` (of its correlated block), of the local interaction, which has to keep its form
    under the action and may only couple orbitals whose inversion images lie in one unit cell (else the inverted
    interaction is no local one), and of the local DMFT self-energy when it is given. The non-local interaction is
    taken to share the symmetry of :math:`H(\mathbf{k})`, as in the symmetry reduction of the momenta. A multiple of
    the identity comes back as ``None`` (no action).

    :param sigma_dmft: The local DMFT self-energy on the correlated orbitals, shape ``[o1, o2, v]`` (see
        :func:`_local_dmft_self_energy`); ``None`` skips its test.
    :return: ``(u, None)`` with the constant ``[o, o]`` or momentum-dependent ``[kx, ky, kz, o, o]`` action (``None``
        for none), or ``(None, reason)``.
    """
    hamiltonian = config.lattice.hamiltonian
    ek = hamiltonian.get_ek(config.lattice.k_grid)
    # to 1e-4 of the largest element, the tolerance of the self-energy's symmetry tests: a wannier90 H(k) keeps
    # inversion only to the precision of its print
    u = find_inversion_orbital_unitary(ek, 1e-4, orbitals=config.sys.correlated_orbitals)
    if u is None:
        return None, "inversion is no symmetry of H(k) to 1e-4 of its largest element"
    u_mat = hamiltonian.get_local_u().mat
    u0 = u if u.ndim == 2 else u[0, 0, 0]
    if u.ndim == 5:
        # U(0)^dagger U(k) = diag(e^{i k.s_a}): the Bloch phases of the cells the orbitals' images lie in
        phases = np.diagonal(u0.conj().T @ u, axis1=-2, axis2=-1).reshape(-1, len(u0))
        same_cell = np.isclose(phases[:, :, None], phases[:, None, :]).all(axis=0)
        a, b, c, d = np.nonzero(np.abs(u_mat) > 1e-12)
        if not (same_cell[a, b] & same_cell[a, c] & same_cell[a, d]).all():
            return None, "the local interaction couples orbitals whose inversion images lie in different unit cells"
    # the action carries the precision of H(k), so the interaction is compared to the same relative tolerance
    rotated = np.einsum("ia,jb,kc,ld,abcd->ijkl", u0, u0.conj(), u0, u0.conj(), u_mat)
    if not np.allclose(rotated, u_mat, rtol=0.0, atol=1e-4 * max(1.0, np.abs(u_mat).max())):
        return None, "the local interaction is not invariant under the orbital action of the inversion"
    if sigma_dmft is not None:
        rotated_sigma = np.einsum("ia,abv,jb->ijv", u0, sigma_dmft, u0.conj())
        if not np.allclose(rotated_sigma, sigma_dmft, rtol=0.0, atol=1e-4 * max(1.0, np.abs(sigma_dmft).max())):
            return None, "the DMFT self-energy is not invariant under the orbital action of the inversion"
    if u.ndim == 2 and np.allclose(u, u[0, 0] * np.eye(len(u))):
        return None, None
    return u, None


def check_parity_sectors(
    comm: MPI.Comm | None = None, sigma_dmft_per_ineq: list[SelfEnergy] | None = None, log: bool = True
) -> np.ndarray | None:
    r"""
    Prepares the parity sectors of the gap (``config.eliashberg.resolve_parity_sectors``): rank 0 looks for the
    orbital action of the inversion (see :func:`_inversion_action`) and broadcasts it, and every rank gets it back for
    the sector projectors (see :func:`_project_gap_to_sector`). Where inversion is no symmetry of the pairing problem
    the sectors would mix, so the flag is switched off with a warning and both channels are solved unprojected. The
    driver calls it before the memory estimate, which counts the sectors, when the run solves the gap, and
    :func:`solve` again for the action, so the momentum-dependent one is held only while the gap is solved. The
    driver's call also tests the DMFT self-energy (outside RPA mode, which reads none) and leaves the log line of the
    action to the second call.

    :param comm: The MPI communicator; ``None`` on a single rank.
    :param sigma_dmft_per_ineq: The DMFT :class:`SelfEnergy` of each inequivalent atom, read on rank 0 (see
        :func:`_local_dmft_self_energy`); ``None`` skips its test.
    :param log: Whether to log the action found.
    :return: The constant ``[o, o]`` or momentum-dependent ``[kx, ky, kz, o, o]`` action, ``None`` for none or without
        the sectors.
    """
    eliashberg = config.eliashberg
    if not eliashberg.resolve_parity_sectors:
        return None
    found = None
    if comm is None or comm.rank == 0:
        found = _inversion_action(None if sigma_dmft_per_ineq is None else _local_dmft_self_energy(sigma_dmft_per_ineq))
    if comm is not None and comm.size > 1:
        found = comm.bcast(found, root=0)
    u, reason = found
    if reason is not None:
        config.logger.warning(
            f"The parity sectors of the gap are switched off: {reason}, so the sectors would mix. Both channels are "
            "solved unprojected."
        )
        eliashberg.resolve_parity_sectors = False
        return None
    if log:
        action = (
            "leaves the orbitals unchanged"
            if u is None
            else (
                "acts on the orbitals by a constant unitary"
                if u.ndim == 2
                else "acts on the orbitals by a momentum-dependent unitary (it maps a site into another unit cell)"
            )
        )
        config.logger.info(f"Parity sectors of the gap: the inversion {action}.")
    return u


def frequency_parts(gap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Splits a gap into its even- and odd-frequency parts, :math:`\Delta^{\pm}(\nu) = \tfrac{1}{2}[\Delta(\nu) \pm
    \Delta(-\nu)]`. Once the orbitals hybridize, the frequency flip is no symmetry of the kernel and the gap of a
    parity sector holds both parts; their weights :math:`\lVert\Delta^{\pm}\rVert^2 / \lVert\Delta\rVert^2` add up to
    one.

    :param gap: The gap, the fermionic frequency (on a box symmetric about zero) as its last axis.
    :return: ``(even, odd)``, each of the shape of ``gap``.
    """
    mirror = np.flip(gap, axis=-1)
    return 0.5 * (gap + mirror), 0.5 * (gap - mirror)


def _even_frequency_weight(gap: np.ndarray) -> float:
    r"""The weight :math:`\lVert\Delta^{+}\rVert^2 / \lVert\Delta\rVert^2` of a gap's even-frequency part (see
    :func:`frequency_parts`), zero for a zero gap."""
    norm2 = np.vdot(gap, gap).real
    even = frequency_parts(gap)[0]
    return float(np.vdot(even, even).real / norm2) if norm2 > 0 else 0.0


def _log_sector_residual(gap: np.ndarray, raw: np.ndarray, label: str, ranks, reduce: Callable | None = None) -> None:
    r"""
    Logs the residual :math:`\lVert K\Delta - \rho\Delta\rVert / (\max(\lvert\rho\rvert, 1) \lVert\Delta\rVert)` of a
    projected sector's leading gap under the unprojected kernel :math:`K`, with the Rayleigh quotient
    :math:`\rho = \langle\Delta, K\Delta\rangle / \langle\Delta, \Delta\rangle` (the eigensolvers report only the real
    part of a complex eigenvalue). It is relative to the eigenvalue, but at most to the instability at 1: the rounding
    of a single-precision kernel is absolute, so a sector whose eigenvalue lies far below 1 cannot reach a small
    relative residual, and it matters no more for the instability. Where the sector is a symmetry of the kernel the
    residual vanishes up to rounding and the eigensolver tolerance ``eliashberg.epsilon``; above the larger of
    ``SECTOR_RESIDUAL_WARNING`` and ten times that tolerance a warning follows.

    :param gap: The leading gap of the sector (this rank's block of it on a team).
    :param raw: :math:`K\Delta` (the same block).
    :param label: The sector label for the log.
    :param ranks: The ranks that log.
    :param reduce: Sums a block quantity over the ranks holding the gap (a team's ``allreduce``); ``None`` for a whole
        gap.
    :return: None.
    """
    reduce = (lambda x: x) if reduce is None else reduce
    norm2 = float(np.real(reduce(np.vdot(gap, gap))))
    rayleigh = complex(reduce(np.vdot(gap, raw))) / norm2 if norm2 > 0 else 0j
    if rayleigh == 0:
        return  # an empty sector (every k equal to -k on a tiny grid) leaves no eigenvalue to check
    leak = raw - rayleigh * gap
    residual = np.sqrt(float(np.real(reduce(np.vdot(leak, leak)))) / norm2) / max(abs(rayleigh), 1.0)
    logger = config.logger
    logger.info(f"Leading gap of {label} under the unprojected kernel: residual {residual:.1e}.", allowed_ranks=ranks)
    # a Ritz vector keeps a residual of the eigensolver tolerance even where the sector is exact
    if residual > max(SECTOR_RESIDUAL_WARNING, 10 * config.eliashberg.epsilon):
        logger.warning(
            f"The leading gap of {label} is no eigenvector of the unprojected kernel (residual "
            f"{residual:.1e}): the parity sectors are no symmetry of this pairing kernel, and the sector eigenvalues "
            "belong to the projected kernel only. 'eliashberg.resolve_parity_sectors: False' solves the channels "
            "unprojected.",
            allowed_ranks=ranks,
        )


def _at_minus_k(a: np.ndarray) -> np.ndarray:
    r"""An array with the momentum grid on its three leading axes, read at :math:`-\mathbf{k}` on the periodic grid."""
    return np.roll(np.flip(a, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))


def _rotate_orbital_pair(g: np.ndarray, u: np.ndarray | None) -> np.ndarray:
    r"""
    Applies the orbital action of the inversion to both orbitals of a gap ``[kx, ky, kz, o1, o2, v]``,
    :math:`U(\mathbf{k})\, g(\mathbf{k})\, U(-\mathbf{k})^T`, for a constant ``u`` (an involution, so
    :math:`U(-\mathbf{k}) = U`) or a momentum-dependent one of shape ``[kx, ky, kz, o, o]``; ``None`` is no action.
    """
    if u is None:
        return g
    u = u.astype(g.dtype, copy=False)
    if u.ndim == 2:
        return np.einsum("ac,xyzcdv,bd->xyzabv", u, g, u, optimize=True)
    return np.einsum("xyzac,xyzcdv,xyzbd->xyzabv", u, g, _at_minus_k(u), optimize=True)


def _exchange_orbitals(g: np.ndarray, u: np.ndarray | None) -> np.ndarray:
    r""":math:`O_U`: the orbital transpose of a gap followed by the orbital action of the inversion,
    :math:`U(\mathbf{k})\, g(\mathbf{k})^T U(-\mathbf{k})^T`."""
    return _rotate_orbital_pair(g.swapaxes(3, 4), u)


def _invert(g: np.ndarray, u: np.ndarray | None) -> np.ndarray:
    r""":math:`P_U`: the inversion of a gap, :math:`\Delta(\mathbf{k}) \to U(\mathbf{k})\, \Delta(-\mathbf{k})\,
    U(-\mathbf{k})^T`."""
    return _rotate_orbital_pair(_at_minus_k(g), u)


def _project_gap_to_sector(
    vec: np.ndarray, gap_shape: tuple, eps_t: int, eps_p: int, u: np.ndarray | None = None
) -> np.ndarray:
    r"""
    Projects a flattened gap onto a physical symmetry sector by applying the two commuting Hermitian projectors
    :math:`\tfrac{1}{2}(1 + \varepsilon_T\, T O_U)` and :math:`\tfrac{1}{2}(1 + \varepsilon_P P_U)` in turn, where
    the involutions act on the orbital gap :math:`\Delta^{\nu}_{12}(\mathbf{k})` as
    :math:`(T\Delta)^{\nu}(\mathbf{k}) = \Delta^{-\nu}(\mathbf{k})` (fermionic-frequency flip),
    :math:`(O_U\Delta)^{\nu}(\mathbf{k}) = U(\mathbf{k}) (\Delta^{\nu}(\mathbf{k}))^T U(-\mathbf{k})^T` (orbital
    exchange) and :math:`(P_U\Delta)^{\nu}(\mathbf{k}) = U(\mathbf{k}) \Delta^{\nu}(-\mathbf{k}) U(-\mathbf{k})^T`
    (inversion), with :math:`U(\mathbf{k})` the orbital action of the inversion,
    :math:`H(-\mathbf{k}) = U(\mathbf{k}) H(\mathbf{k}) U(\mathbf{k})^\dagger` and :math:`U(\mathbf{k}) U(-\mathbf{k})
    = 1`, constant unless inversion maps a site into another unit cell (see :func:`check_parity_sectors`).
    The Pauli antisymmetry :math:`\hat{S}\,P\,O\,T\,\Delta = -\Delta` with the spin exchange :math:`\hat{S}` a scalar
    in the singlet/triplet basis fixes :math:`T O_U \Delta = \mathrm{sign}\, P_U \Delta` (``sign`` the channel sign),
    so once :math:`\varepsilon_T` is chosen the inversion parity is forced to :math:`\varepsilon_P = \mathrm{sign}\cdot
    \varepsilon_T`, and the sectors are the inversion-parity sectors of the channel. The pairing kernel conserves
    :math:`P_U` wherever inversion is a symmetry, so they are invariant subspaces; the frequency parity :math:`T` alone
    is no symmetry once the orbitals hybridize (even and odd frequencies mix), and for one band :math:`O_U` is trivial
    and :math:`\varepsilon_T` is the frequency parity.

    :param vec: The flattened gap vector.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param eps_t: The requested parity under :math:`T O_U`, :math:`+1` (even) or :math:`-1` (odd).
    :param eps_p: The forced inversion parity, :math:`\mathrm{sign}\cdot\varepsilon_T`.
    :param u: The orbital action of the inversion, or ``None`` for none (one band, inversion-even orbitals).
    :return: The projected flattened gap vector (dtype preserved).
    """
    g = vec.reshape(gap_shape)
    g = 0.5 * (g + eps_t * _exchange_orbitals(np.flip(g, axis=-1), u))
    g = 0.5 * (g + eps_p * _invert(g, u))
    return g.reshape(-1).astype(vec.dtype, copy=False)


def _project_frequency_block(
    gap: np.ndarray, v0: int, v1: int, eps_t: int, eps_p: int, u: np.ndarray | None = None
) -> np.ndarray:
    r"""
    Returns the fermionic-frequency block ``[..., v0:v1]`` of the sector projection of a gap ``[kx, ky, kz, o1, o2, v]``
    (see :func:`_project_gap_to_sector`), computed from that block and its frequency mirror alone: the :math:`T O_U`
    projector pairs :math:`\nu` with :math:`-\nu` and the :math:`P_U` projector acts within one frequency, so every
    element is formed by the same operations as in the projection of the whole gap: bit for bit without an orbital
    action, and to rounding with one, whose einsum picks its contraction path by the operand shapes.

    :param gap: The gap in shape ``[kx, ky, kz, o1, o2, v]``.
    :param v0: First frequency index of the block.
    :param v1: One past the last frequency index of the block.
    :param eps_t: The requested parity under :math:`T O_U`, :math:`+1` (even) or :math:`-1` (odd).
    :param eps_p: The forced inversion parity, :math:`\mathrm{sign}\cdot\varepsilon_T`.
    :param u: The orbital action of the inversion, or ``None`` for none.
    :return: The projected block, shape ``[kx, ky, kz, o1, o2, v1 - v0]``.
    """
    nv = gap.shape[-1]
    mirror = gap[..., nv - v1 : nv - v0][..., ::-1]  # np.flip(gap, axis=-1)[..., v0:v1]
    g = 0.5 * (gap[..., v0:v1] + eps_t * _exchange_orbitals(mirror, u))
    return 0.5 * (g + eps_p * _invert(g, u))


def _sector_seed(
    base_seed: np.ndarray, gap_shape: tuple, eps_t: int | None, eps_p: int | None, u: np.ndarray | None = None
) -> np.ndarray:
    r"""
    Returns the eigensolver's starting vector of one sector: the base seed projected onto the sector, with a
    deterministic random fallback when the projection of the seed collapses (a seed whose parity is orthogonal to
    the requested sector) and the nonzero base seed itself when the sector is empty on the grid (e.g. every
    :math:`\mathbf{k}` equals :math:`-\mathbf{k}`); the ``"none"`` sector (``eps_t`` of ``None``) uses the base
    seed unchanged. Identical on every rank for identical inputs.

    :param base_seed: The flattened initial gap seed.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param eps_t: The requested parity under :math:`T O_U`, or ``None`` for the raw kernel.
    :param eps_p: The forced inversion parity.
    :param u: The orbital action of the inversion (see :func:`_project_gap_to_sector`).
    :return: The flattened seed of the sector.
    """
    if eps_t is None:
        return base_seed
    seed = _project_gap_to_sector(base_seed, gap_shape, eps_t, eps_p, u)
    if np.linalg.norm(seed) >= 1e-10 * max(np.linalg.norm(base_seed), 1e-30):
        return seed
    rng = np.random.default_rng(0)
    fallback = (rng.standard_normal(gap_shape) + 1j * rng.standard_normal(gap_shape)).flatten()
    fallback = _project_gap_to_sector(fallback.astype(base_seed.dtype, copy=False), gap_shape, eps_t, eps_p, u)
    return fallback if np.linalg.norm(fallback) > 0 else base_seed


def _sector_seeds(
    base_seed: np.ndarray,
    gap_shape: tuple,
    eps_t: int | None,
    eps_p: int | None,
    n_seeds: int,
    u: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Returns the ``n_seeds`` starting vectors of one sector along a leading axis: the seed of :func:`_sector_seed`
    followed by random vectors of a fixed-seed generator projected onto the sector, identical on every rank.

    :param base_seed: The flattened initial gap seed.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param eps_t: The requested parity under :math:`T O_U`, or ``None`` for the raw kernel.
    :param eps_p: The forced inversion parity.
    :param n_seeds: Number of starting vectors.
    :param u: The orbital action of the inversion (see :func:`_project_gap_to_sector`).
    :return: The seeds, shape ``[n_seeds, n]``.
    """
    rng = np.random.default_rng(1)
    seeds = [_sector_seed(base_seed, gap_shape, eps_t, eps_p, u)]
    for _ in range(n_seeds - 1):
        extra = (rng.standard_normal(base_seed.size) + 1j * rng.standard_normal(base_seed.size)).astype(base_seed.dtype)
        seeds.append(extra if eps_t is None else _project_gap_to_sector(extra, gap_shape, eps_t, eps_p, u))
    return np.stack(seeds)


def _finish_sector(
    lambdas: np.ndarray,
    gaps: np.ndarray,
    gap_shape: tuple,
    channel: SpinChannel,
    nq: tuple,
    label: str,
    orbital_mirrors: dict,
    ranks: tuple,
) -> tuple[np.ndarray, list[GapFunction]]:
    r"""
    Orders one solved sector's eigenpairs by descending eigenvalue, symmetrizes degenerate gaps when configured,
    logs the eigenvalues and wraps the eigenvectors as :class:`GapFunction` objects (as many as the eigensolver
    returned, each a contiguous copy so the eigenvector matrix is released).

    :param lambdas: The eigenvalues.
    :param gaps: The eigenvectors as columns, ``[n, len(lambdas)]``.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param channel: The pairing channel (used to label outputs).
    :param nq: The momentum-grid shape carried onto each :class:`GapFunction`.
    :param label: The sector label for the log.
    :param orbital_mirrors: The orbital mirror unitaries for the degenerate-gap symmetrization.
    :param ranks: The ranks tuple used for logging.
    :return: ``(lambdas, [GapFunction, ...])``.
    """
    order = lambdas.argsort()[::-1]
    if not np.array_equal(order, np.arange(len(order))):
        lambdas, gaps = lambdas[order], gaps[:, order]
    if config.eliashberg.symmetrize_degenerate_gaps:
        gaps = symmetrize_degenerate_gaps(lambdas, gaps, gap_shape, orbital_mirrors=orbital_mirrors)
    plural = "" if config.eliashberg.n_eig == 1 else "s"
    config.logger.info(
        f"Largest eigenvalue{plural} for {label}: " + ", ".join(f"{lam:.6f}" for lam in lambdas), allowed_ranks=ranks
    )
    gaps = [np.ascontiguousarray(gaps[:, i]).reshape(gap_shape) for i in range(gaps.shape[1])]
    if gap_shape[3] > 1:
        weights = [_even_frequency_weight(gap) for gap in gaps]
        config.logger.info(
            f"Even-frequency weight of the gap{plural} for {label}: " + ", ".join(f"{w:.4f}" for w in weights),
            allowed_ranks=ranks,
        )
    return lambdas, [GapFunction(gap, channel, nq) for gap in gaps]


def _solve_loosening(solve: Callable[[float], tuple], tol: float, n_eig: int, label: str, ranks) -> tuple:
    r"""
    Returns the eigensolver's result at ``tol`` once it delivers ``n_eig`` eigenpairs within ``LANCZOS_RESTARTS``
    restarts, else retries at a tolerance ``TOLERANCE_LOOSENING`` times looser, twice at most, with a warning each
    time, and returns the last attempt whatever it delivered. In single precision a sector whose eigenvalues are small
    against the rest of the kernel cannot reach a relative tolerance below the kernel's absolute rounding, and the
    eigensolver would run to its restart cap for nothing.

    :param solve: Runs the eigensolver at a tolerance; the first entry of its result holds the converged eigenvalues.
    :param tol: The tolerance of the first attempt.
    :param n_eig: Number of eigenpairs asked for.
    :param label: The sector label for the log.
    :param ranks: The ranks that log.
    :return: The result of the last attempt.
    """
    for attempt in range(3):
        result = solve(tol)
        if len(result[0]) >= n_eig or attempt == 2:
            return result
        config.logger.warning(
            f"Lanczos for {label} did not converge to {tol:.0e} within {LANCZOS_RESTARTS} restarts; retrying at "
            f"{tol * TOLERANCE_LOOSENING:.0e}.",
            allowed_ranks=ranks,
        )
        tol *= TOLERANCE_LOOSENING


def _team_arnoldi(
    apply_block: Callable,
    seed_block: np.ndarray,
    n_eig: int,
    ncv: int,
    tol: float,
    maxiter: int,
    team_comm: MPI.Comm | None,
    n_block: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    r"""
    Krylov-Schur (restarted Arnoldi) iteration for the eigenpairs of largest real part of an operator whose vectors
    are split over the ranks of a team: every rank holds its own block of each basis vector, ``apply_block`` maps a
    rank's block of a vector to its block of the operator's image (a collective call over the team), and every
    inner product is a team allreduce, so all ranks walk the same iteration with the same scalars. The basis is
    orthogonalized by classical Gram-Schmidt, repeated once when the first pass removed more than a fixed share of
    the new vector (:data:`REORTHOGONALIZE_BELOW`, the criterion ARPACK uses), so a well-conditioned step streams
    the basis twice instead of four times. A vector the second pass also cuts below that share lies in the span of
    the basis (the Krylov space is invariant, e.g. a sector smaller than the basis): the iteration then continues
    from a random direction orthogonalized against the basis, as ARPACK does, with the vanishing norm kept in the
    Rayleigh quotient so the Arnoldi relation stays exact. When the ``ncv``-vector basis is full, the Ritz pairs of the
    Rayleigh quotient are formed; a pair counts as converged when its residual estimate :math:`|\beta_m|\,|y_m|` is
    at most ``tol`` times the modulus of its Ritz value. Otherwise the Schur form of the Rayleigh quotient is
    reordered so that the ``n_eig`` wanted Ritz values and half of the rest lead, and the basis is compacted to
    those Schur vectors plus the residual direction before the iteration continues. Stops when the wanted pairs have
    converged or after ``maxiter`` restarts, returning the converged pairs in either case; the eigenvalues are
    returned as real parts, the way scipy's ``eigsh`` reports a complex operator's spectrum.

    With ``n_block > 1`` it is the block (band) variant: the basis starts from ``n_block`` orthonormalized seeds, the
    image of basis vector :math:`j` is orthogonalized against the first :math:`j + n_\mathrm{block}` vectors and
    becomes vector :math:`j + n_\mathrm{block}`, and the last ``n_block`` vectors form the residual block whose
    coupling to the Ritz vectors gives the residual estimates and carries over the restart. Every eigenspace of
    dimension up to ``n_block`` is then reached in exact arithmetic, where a single start vector reaches one direction
    of each; ``n_block = 1`` is the plain iteration.

    :param apply_block: The operator, ``block -> block`` on this rank's slice.
    :param seed_block: This rank's block of the starting vector, or of the ``n_block`` starting vectors along a
        leading axis.
    :param n_eig: Number of wanted eigenpairs.
    :param ncv: Size of the Krylov basis (``n_eig < ncv``).
    :param tol: Relative residual tolerance of a Ritz pair.
    :param maxiter: Maximum number of restarts.
    :param team_comm: The team communicator, or ``None`` for a single process holding the whole vectors.
    :param n_block: Number of starting vectors, the largest eigenvalue multiplicity resolved in exact arithmetic.
    :return: ``(lambdas, ritz, basis, n_matvec, matvec_seconds)``: the converged eigenvalues in descending order,
        their Ritz coefficients as columns (this rank's block of eigenvector ``i`` is ``ritz[:, i] @ basis``, of unit
        norm up to rounding), this rank's blocks of the basis vectors as rows, the number of operator applications
        and the time spent in them.
    """
    dtype = seed_block.dtype
    seeds = seed_block.reshape(n_block, -1)
    basis = np.empty((ncv + n_block - 1, seeds.shape[1]), dtype=dtype)
    rayleigh = np.zeros((ncv + n_block, ncv), dtype=np.complex128)
    n_matvec, matvec_seconds = 0, 0.0

    reduce = (lambda x: x) if team_comm is None else team_comm.allreduce

    def dots(vectors: np.ndarray, w: np.ndarray) -> np.ndarray:
        # conjugating the one vector instead of the basis slice spares a basis-sized copy per call (bit-identical)
        return reduce((vectors @ w.conj()).conj())

    def norm(w: np.ndarray) -> float:
        return float(np.sqrt(reduce(float(np.vdot(w, w).real))))

    rng = np.random.default_rng(0 if team_comm is None else team_comm.Get_rank())

    def fresh_direction(n_basis: int) -> np.ndarray:
        # a random unit vector orthogonalized twice against the basis (all ranks draw their block together)
        w = (rng.standard_normal(basis.shape[1]) + 1j * rng.standard_normal(basis.shape[1])).astype(dtype)
        for _ in range(2):
            w = w - dots(basis[:n_basis], w) @ basis[:n_basis]
        return w / norm(w)

    basis[0] = seeds[0] / norm(seeds[0])
    for i in range(1, n_block):
        w = seeds[i] - dots(basis[:i], seeds[i]) @ basis[:i]
        w = w - dots(basis[:i], w) @ basis[:i]
        basis[i] = w / norm(w)
    del seed_block, seeds
    j, restarts = 0, 0
    while True:
        start = time.perf_counter()
        w = apply_block(basis[j].copy()).reshape(-1).astype(dtype, copy=False)
        matvec_seconds += time.perf_counter() - start
        n_matvec += 1
        w_norm = norm(w)
        top = j + n_block
        h = dots(basis[:top], w)
        w = w - h @ basis[:top]
        beta, spent = norm(w), False
        if beta < REORTHOGONALIZE_BELOW * w_norm:
            correction = dots(basis[:top], w)
            w = w - correction @ basis[:top]
            h += correction
            beta, previous = norm(w), beta
            spent = beta < REORTHOGONALIZE_BELOW * previous
        rayleigh[:top, j] = h
        rayleigh[top, j] = beta
        if j + 1 < ncv:
            basis[top] = fresh_direction(top) if spent else w / beta
            j += 1
            continue
        # the basis is full: Ritz pairs of the Rayleigh quotient, largest real part first
        theta, ritz = np.linalg.eig(rayleigh[:ncv])
        order = np.argsort(-theta.real)
        theta, ritz = theta[order], ritz[:, order]
        residual = beta * np.abs(ritz[-1]) if n_block == 1 else np.linalg.norm(rayleigh[ncv:] @ ritz, axis=0)
        converged = residual <= tol * np.abs(theta)
        if converged[:n_eig].all() or restarts >= maxiter:
            chosen = [i for i in range(n_eig) if converged[i]]
            return theta[chosen].real, ritz[:, chosen], basis[:ncv], n_matvec, matvec_seconds
        # Krylov-Schur restart: the wanted Ritz values and half of the rest lead the reordered Schur form
        threshold = theta.real[n_eig + (ncv - n_eig) // 2 - 1]
        schur, vectors, n_leading = sp.linalg.schur(
            rayleigh[:ncv], output="complex", sort=lambda z: z.real >= threshold
        )
        kept = min(n_leading, ncv - n_block)
        # compacted in column chunks, so the transient stays a fraction of the basis
        chunk = max(1, basis.shape[1] // 8)
        for c0 in range(0, basis.shape[1], chunk):
            basis[:kept, c0 : c0 + chunk] = vectors[:, :kept].T @ basis[:ncv, c0 : c0 + chunk]
        basis[kept : kept + n_block - 1] = basis[ncv:]
        basis[kept + n_block - 1] = w / beta
        coupling = beta * vectors[-1, :kept] if n_block == 1 else rayleigh[ncv:] @ vectors[:, :kept]
        rayleigh[...] = 0.0
        rayleigh[:kept, :kept] = schur[:kept, :kept]
        rayleigh[kept : kept + n_block, :kept] = coupling
        j = kept
        restarts += 1


def _crossed_term(direct: np.ndarray, sign: int, v0: int = 0, v1: int | None = None) -> np.ndarray:
    r"""
    Forms the crossed term of the pairing kernel from the direct term :math:`D`, as the exchange of its outgoing
    pair: the gap's first orbital sits at :math:`-\mathrm{k}` (see :func:`_apply_gchi0_pp`), so the exchange is
    :math:`(E X)_{ab}(\mathrm{k}) = \mathrm{sign}\,X_{ba}(-\mathrm{k})` with :math:`-\mathrm{k} = (-\mathbf{k}, -\nu)`
    and the kernel is :math:`(1 + E) D`. It maps every gap onto the Pauli-allowed form :math:`E\Delta = \Delta` and
    agrees with the exchange-symmetrized gap equation on it, for any vertex. The momentum flip is the same index map
    in real space and in momentum space, so either form of the direct term may be passed.

    :param direct: The direct term ``[x, y, z, o1, o2, v]`` on the full fermionic box.
    :param sign: The channel sign (:math:`+1` singlet, :math:`-1` triplet).
    :param v0: First fermionic index of the returned block.
    :param v1: One past the last fermionic index of the returned block (``None``: the whole axis).
    :return: The crossed term on the fermionic block ``[v0, v1)``, a new array.
    """
    nv = direct.shape[-1]
    v1 = nv if v1 is None else v1
    crossed = _at_minus_k(np.flip(direct.swapaxes(3, 4)[..., nv - v1 : nv - v0], axis=-1))
    if sign != 1:
        crossed *= sign
    return crossed


def gap_parity_diagnostics(gap: np.ndarray, gap_shape: tuple, u: np.ndarray | None = None) -> dict[str, complex]:
    r"""
    Reports the parity Rayleigh quotients :math:`\langle \Delta, X \Delta \rangle / \langle \Delta, \Delta \rangle`
    of a flattened gap for the involutions ``T`` (frequency flip), ``P`` (momentum flip), ``O`` (orbital transpose),
    their product ``PO``, and the two the parity sectors fix (see :func:`_project_gap_to_sector`): ``TOU``, the
    frequency flip times the orbital exchange :math:`O_U`, and ``PU``, the inversion :math:`P_U`, with ``u`` the
    orbital action of the inversion (``None``: none, so they are ``T.O`` and ``P``). A pure-parity gap returns
    :math:`\pm 1` for the involutions it is an eigenvector of; the values certify the parity of a returned gap and
    expose any leakage.

    :param gap: The flattened gap vector.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param u: The orbital action of the inversion (see :func:`check_parity_sectors`).
    :return: A dict mapping ``"T"``, ``"P"``, ``"O"``, ``"PO"``, ``"TOU"``, ``"PU"`` to the Rayleigh quotient.
    """
    g = gap.reshape(gap_shape)
    ops = {
        "T": np.flip(g, axis=-1),
        "P": _at_minus_k(g),
        "O": g.swapaxes(3, 4),
        "PO": _at_minus_k(g.swapaxes(3, 4)),
        "TOU": _exchange_orbitals(np.flip(g, axis=-1), u),
        "PU": _invert(g, u),
    }
    denom = np.vdot(g, g)
    if denom == 0:
        return {name: 0j for name in ops}
    return {name: complex(np.vdot(g, op) / denom) for name, op in ops.items()}


def _classify_momentum_block(block: np.ndarray, purity: float) -> str:
    r"""
    Classifies the wave symmetry of one orbital block's momentum profile, returning ``s``, ``d``, ``dxy`` (``dxz``,
    ``dyz``), ``g``, ``p`` or ``x``.

    The block is decomposed into symmetry sectors rather than compared element-wise: for an involution :math:`X`,
    the fraction of the block's weight that is even under it is
    :math:`\lVert \tfrac{1}{2}(b + Xb)\rVert^{2}/\lVert b\rVert^{2}`, and the odd fraction is its complement. A
    sector counts as realized when its fraction reaches ``purity``. This tolerates the few percent of admixture
    that a converged gap generally carries, where an element-wise ``allclose`` would reject it outright.

    The involutions are the three coordinate inversions :math:`k_i \to -k_i` (in the ``np.roll(np.flip(...), 1)``
    convention of the Gamma-at-index-0 grid) and the axis exchanges :math:`k_i \leftrightarrow k_j`, the latter only
    for pairs of axes of equal length. All three exchanges are tested, not just :math:`k_x \leftrightarrow k_y`: on
    a cubic lattice a :math:`d`-wave living in the :math:`xz` plane is antisymmetric under
    :math:`k_x \leftrightarrow k_z` and carries no definite symmetry under :math:`k_x \leftrightarrow k_y` at all.

    Only the momentum profile enters here; the orbital indices are fixed by the caller. This is a diagnostic label,
    so unlike the mirrors of :func:`_mirror_operator` no orbital transformation accompanies the reflections.

    :param block: The momentum profile ``[kx, ky, kz]`` of one orbital block.
    :param purity: Minimum weight fraction for a symmetry sector to count as realized.
    :return: ``s`` (even under every inversion, symmetric under every applicable exchange), ``d`` (even under every
        inversion, antisymmetric under some exchange: the :math:`d_{x^2-y^2}` type), ``dxy`` (odd under
        :math:`k_x \to -k_x` and :math:`k_y \to -k_y`, even under the third, symmetric under their exchange; ``dxz``,
        ``dyz`` in the other planes), ``g`` (odd under two inversions, antisymmetric under their exchange), ``p`` (odd
        under the full inversion) or ``x``.
    """

    def even_fraction(arr: np.ndarray, mirrored: np.ndarray) -> float:
        norm = float(np.vdot(arr, arr).real)
        return float(np.vdot(arr + mirrored, arr + mirrored).real) / (4.0 * norm) if norm > 0 else 0.0

    def invert(arr: np.ndarray, axes: tuple) -> np.ndarray:
        return np.roll(np.flip(arr, axis=axes), shift=(1,) * len(axes), axis=axes)

    if float(np.vdot(block, block).real) <= 0:
        return "x"

    even_axis = [even_fraction(block, invert(block, (axis,))) for axis in range(3)]
    if all(frac >= purity for frac in even_axis):
        exchanges = [
            (i, j) for i, j in ((0, 1), (0, 2), (1, 2)) if block.shape[i] == block.shape[j] and block.shape[i] > 1
        ]
        swapped = [np.swapaxes(block, i, j) for i, j in exchanges]
        if any(even_fraction(block, -swap) >= purity for swap in swapped):  # antisymmetric under some exchange
            return "d"
        if all(even_fraction(block, swap) >= purity for swap in swapped):
            return "s"
        return "x"
    odd_axes = [axis for axis in range(3) if 1.0 - even_axis[axis] >= purity]
    if len(odd_axes) == 2 and even_axis[3 - sum(odd_axes)] >= purity:
        i, j = odd_axes
        if block.shape[i] == block.shape[j]:
            swapped = np.swapaxes(block, i, j)
            if even_fraction(block, swapped) >= purity:
                return f"d{'xyz'[i]}{'xyz'[j]}"
            if even_fraction(block, -swapped) >= purity:
                return "g"
        return "x"
    if even_fraction(block, -invert(block, (0, 1, 2))) >= purity:  # odd under the full inversion
        return "p"
    return "x"


def fermi_surface_bands(ek: np.ndarray, mu: float, orbitals: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Returns the band energies relative to the chemical potential and the band eigenvectors on the gap's orbitals,
    :math:`\varepsilon_n(\mathbf{k}) - \mu` and :math:`\psi_{an}(\mathbf{k})`, the input of the Fermi-surface sign test
    of :func:`classify_gap_symmetry`.

    :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})` on the whole grid, shape ``[kx, ky, kz, o1, o2]``.
    :param mu: Chemical potential :math:`\mu`.
    :param orbitals: Indices of the orbitals the gap lives on (the correlated ones); ``None`` for every orbital.
    :return: ``(energies, vectors)`` of shapes ``[kx, ky, kz, n]`` and ``[kx, ky, kz, o, n]``.
    """
    energies, vectors = np.linalg.eigh(ek)
    return energies - mu, (vectors if orbitals is None else vectors[..., orbitals, :])


def _changes_sign_between_fermi_sheets(
    gap_slice: np.ndarray, bands: tuple[np.ndarray, np.ndarray], consistency: float = 0.5, floor: float = 0.05
) -> bool:
    r"""
    Whether a gap at one frequency has opposite signs on the Fermi sheets of two bands. The intraband gap
    :math:`\Delta_n(\mathbf{k}) = \psi_n(\mathbf{k})^\dagger \Delta(\mathbf{k})^{T} \psi_n(\mathbf{k})` (the band gap of
    the time-reversal gauge :math:`\psi(-\mathbf{k}) = \psi(\mathbf{k})^*` for the gap's first orbital at
    :math:`-\mathbf{k}`, independent of the eigenvector phases) is read on the momenta next to a sign change of
    :math:`\varepsilon_n - \mu` towards a grid neighbor, after the global phase that makes it most nearly real. A band
    counts when its sheet carries at least ``floor`` of the largest sheet's weight and keeps at least ``consistency`` of
    it in the signed sum (one sign on most of the sheet). Bands degenerate at a momentum to 1e-4 of the largest band
    energy all carry the trace over them there as their band gap, independent of the basis the eigensolver picked
    among them, so degenerate partners count with one sign, also where another band crosses them.

    :param gap_slice: The gap at one frequency, shape ``[kx, ky, kz, o1, o2]``.
    :param bands: The ``(energies, vectors)`` of :func:`fermi_surface_bands`.
    :param consistency: Minimum ratio of the signed to the absolute sum on a sheet.
    :param floor: Minimum weight of a sheet relative to the largest one.
    :return: True when two counted Fermi surfaces carry opposite signs.
    """
    energies, vectors = bands
    intra = np.einsum("xyzan,xyzba,xyzbn->xyzn", vectors.conj(), gap_slice, vectors)
    sheet = np.zeros(energies.shape, dtype=bool)
    for axis in range(3):
        if energies.shape[axis] > 1:
            crossing = energies * np.roll(energies, -1, axis=axis) < 0
            sheet |= crossing | np.roll(crossing, 1, axis=axis)
    if not sheet.any():
        return False
    # bands degenerate at a momentum all carry the trace over them there as their band gap
    degenerate = np.abs(energies[..., :, None] - energies[..., None, :]) <= 1e-4 * np.abs(energies).max()
    intra = np.einsum("xyznm,xyzm->xyzn", degenerate, intra)
    signed = (intra * np.exp(-0.5j * np.angle(np.sum(intra[sheet] ** 2)))).real
    sums = [
        (signed[..., n][sheet[..., n]].sum(), np.abs(intra[..., n][sheet[..., n]]).sum())
        for n in range(intra.shape[-1])
    ]
    largest = max(weight for _, weight in sums)
    signs = [
        np.sign(total) for total, weight in sums if weight > floor * largest and abs(total) >= consistency * weight
    ]
    return min(signs, default=0) < 0 < max(signs, default=0)


def classify_gap_symmetry(
    gap: np.ndarray,
    purity: float = 0.9,
    weight_floor: float = 0.01,
    bands: tuple[np.ndarray, np.ndarray] | None = None,
) -> str:
    r"""
    Classifies the momentum wave symmetry and Matsubara-frequency parity of a gap and returns a compact label of the
    form ``<wave><parity>``. The wave is ``s``, ``s±``, ``d`` (:math:`d_{x^2-y^2}` type), ``dxy`` (``dxz``, ``dyz``),
    ``g`` or ``p`` (``x`` if none of these match), and the parity sign is ``+`` (even in :math:`\nu`), ``-`` (odd) or
    empty (neither). The momentum tests are reflections of the grid axes, so the names presume the axes of a square or
    cubic lattice. With ``bands`` an ``s`` wave whose band gap has opposite signs on the Fermi sheets of two bands
    becomes ``s±`` (see :func:`_changes_sign_between_fermi_sheets`). The frequency parity is the sign of the
    global T Rayleigh quotient :math:`\langle \Delta, T\Delta \rangle / \langle \Delta, \Delta \rangle` (with
    :math:`T` the fermionic-frequency flip), so it is consistent with the parity diagnostics.

    Every orbital block :math:`(o_1, o_2)` carrying at least ``weight_floor`` of the heaviest block's weight is
    classified separately from its momentum profile at the first positive Matsubara frequency (see
    :func:`_classify_momentum_block`), because a multi-orbital gap need not carry the same wave in every block: the
    :math:`E_g` singlet of a cubic :math:`t_{2g}` system, for instance, is a :math:`d`-wave in each block but a
    *different* one each time, :math:`\cos k_x - \cos k_y` on :math:`d_{xy}` and its cyclic images on the other two.
    Blocks below the floor are skipped - the wave symmetry of a numerically negligible block is noise.

    Blocks that agree collapse to a single label, so the common case stays one token wide. When they disagree the
    label lists each distinct wave with the blocks realizing it, e.g. ``d+[00,11]|s+[22]``, ordered by weight; the
    number of entries is bounded by the number of distinct waves, never by the number of orbital blocks.

    :param gap: The gap array in the ``[kx, ky, kz, o1, o2, v]`` layout.
    :param purity: Minimum weight fraction for a symmetry sector to count as realized within a block.
    :param weight_floor: Minimum weight of a block relative to the heaviest one for it to be classified at all.
    :param bands: The band energies and eigenvectors of :func:`fermi_surface_bands`, for the ``s±`` test; ``None``
        skips it.
    :return: The ``<wave><parity>`` label, or ``"unknown"`` for an all-zero gap.
    """
    denom = np.vdot(gap, gap)
    if denom == 0:
        return "unknown"
    t = (np.vdot(gap, np.flip(gap, axis=-1)) / denom).real
    freq_label = "+" if t > 0.5 else ("-" if t < -0.5 else "")

    n_o1, n_o2 = gap.shape[3], gap.shape[4]
    weights = np.linalg.norm(gap.reshape(-1, n_o1, n_o2, gap.shape[-1]), axis=(0, -1))
    if weights.max() <= 0:
        return f"x{freq_label}"

    labeled: dict[str, list[tuple[float, str]]] = {}
    for o1 in range(n_o1):
        for o2 in range(n_o2):
            if weights[o1, o2] < weight_floor * weights.max():
                continue
            wave = _classify_momentum_block(gap[:, :, :, o1, o2, gap.shape[-1] // 2], purity)
            labeled.setdefault(wave, []).append((float(weights[o1, o2]), f"{o1}{o2}"))

    if not labeled:
        return f"x{freq_label}"
    if "s" in labeled and bands is not None and _changes_sign_between_fermi_sheets(gap[..., gap.shape[-1] // 2], bands):
        labeled["s±"] = labeled.pop("s")
    if len(labeled) == 1:
        return f"{next(iter(labeled))}{freq_label}"
    order = sorted(labeled.items(), key=lambda kv: -sum(w for w, _ in kv[1]))
    return "|".join(
        f"{wave}{freq_label}[{','.join(block for _, block in sorted(blocks, key=lambda wb: -wb[0]))}]"
        for wave, blocks in order
    )


# Testing/benchmark override: forces the block-distributed grid solver even when the in-memory one would fit.
FORCE_GRID_SOLVER: bool = False


def _gather_grid_vertex_block(
    gamma_r_pp: FourPoint, comm: MPI.Comm, row_slice: slice, col_slice: slice, keep: bool
) -> np.ndarray | None:
    r"""
    Redistributes the q-distributed pairing vertex into this rank's frequency block: every rank broadcasts its
    rank-local irreducible-BZ share once (chunked, one share in flight at a time), and each grid rank keeps only the
    ``[row_slice, col_slice]`` frequency window of every share. Afterwards a grid rank holds all irreducible momenta
    of its own ``(nu, nu')`` block and nothing else.

    :param gamma_r_pp: This rank's irreducible-BZ share of the pairing vertex (compressed momentum axis).
    :param comm: The MPI communicator.
    :param row_slice: The :math:`\nu` (row) window of this rank's block.
    :param col_slice: The :math:`\nu'` (column) window of this rank's block.
    :param keep: Whether this rank is part of the solver grid (idle ranks only feed the broadcasts).
    :return: The assembled block ``[nk_irr, o, o, o, o, nu_block, nu'_block]``, or ``None`` on idle ranks.
    """
    counts = comm.allgather(gamma_r_pp.current_shape[0])
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(int)

    block = None
    if keep:
        shape = gamma_r_pp.current_shape[1:5] + (
            row_slice.stop - row_slice.start,
            col_slice.stop - col_slice.start,
        )
        block = np.empty((int(offsets[-1]),) + shape, dtype=gamma_r_pp.mat.dtype)

    for src in range(comm.size):
        src_mat = mpi_utils.bcast_rows(comm, gamma_r_pp.mat, root=src)
        if keep:
            block[offsets[src] : offsets[src + 1]] = src_mat[..., row_slice, col_slice]
        if src != comm.rank:
            del src_mat
    return block


def solve_eliashberg_lanczos_grid(
    gamma_r_pp: FourPoint, gchi0_q0_pp: FourPoint | None, comm: MPI.Comm, bubble_rank: int, u: np.ndarray | None = None
) -> dict[str, tuple[np.ndarray, list[GapFunction]]] | None:
    r"""
    Solves the linearized Eliashberg equation with the pairing vertex distributed over a 2-D ``(nu, nu')`` block grid,
    for problems whose full-BZ vertex does not fit on a single rank. Semantics are identical to
    :func:`solve_eliashberg_lanczos`: the same matvec, projectors, seeding and post-processing, only the storage and
    the contraction are distributed.

    Every grid rank holds one full-BZ, Fourier-transformed, matmul-layout block of the vertex and runs the
    eigensolver in lockstep on the full gap vector (the collectives inside the matvec keep every rank's iterates
    identical). Per matvec each rank dresses its own :math:`\nu'`-column block of the gap and contracts it with its
    vertex block, so the only communication is one block-sized ``Allreduce`` over each row group (completing the
    :math:`\nu'` sum) and one gap-sized ``Allgatherv`` over each column group (reassembling the :math:`\nu` rows);
    the crossed term reads the mirror :math:`\nu` rows (see :func:`_crossed_term`), so every rank adds it to the
    reassembled direct term. Sectors and channels run sequentially on the whole grid; ranks beyond ``rows * cols``
    idle and return ``None``.

    :param gamma_r_pp: This rank's irreducible-BZ share of the pairing vertex; consumed by the solve.
    :param gchi0_q0_pp: The bare pp bubble :math:`\chi_0^{\mathrm{pp}}` (only read on ``bubble_rank``).
    :param comm: The MPI communicator.
    :param bubble_rank: The rank holding the pp bubble.
    :param u: The orbital action of the inversion for the parity sectors (see :func:`_project_gap_to_sector`).
    :return: ``{parity_label: (lambdas, gaps)}`` on grid ranks, ``None`` on idle ranks.
    """
    logger = config.logger
    channel = gamma_r_pp.channel
    n_freq = gamma_r_pp.current_shape[-1]
    n_bands = gamma_r_pp.n_bands
    k_grid = config.lattice.k_grid

    rows, cols = solver_grid_shape(comm.size, n_freq)
    in_grid = comm.rank < rows * cols
    logger.info(
        f"Starting to solve the Eliashberg equation for the {channel.value}let channel "
        f"on a {rows}x{cols} solver grid.",
        allowed_ranks=(0,),
    )

    i, j = divmod(comm.rank, cols) if in_grid else (0, 0)
    row_bounds = np.linspace(0, n_freq, rows + 1).astype(int)
    row_slice = slice(int(row_bounds[i]), int(row_bounds[i + 1]))
    width = n_freq // cols
    col_slice = slice(j * width, (j + 1) * width)

    # collective splits must involve every rank, including idle ones (distinct colors keep them out of the groups)
    reduce_comm = comm.Split(i if in_grid else rows, comm.rank)  # same nu rows, nu' columns vary
    gather_comm = comm.Split(j if in_grid else cols, comm.rank)  # same nu' columns, nu rows vary

    # w2dynamics G2 leg order (c cdag c cdag) -> TRIQS order (cdag c cdag c); orbital-only, so it commutes with the
    # frequency slicing and is applied to the small rank-local share before the exchange
    gamma_r_pp = gamma_r_pp.permute_orbitals("abcd->badc", False)
    block_mat = _gather_grid_vertex_block(gamma_r_pp, comm, row_slice, col_slice, in_grid)
    gamma_r_pp.free()

    # the ranks beyond the rows x cols grid never touch the bubble: ship it inside the grid only, so the idle
    # ranks do not each receive a full-BZ copy (Split keeps the rank order, so the grid root index is bubble_rank)
    grid_comm = comm.Split(0 if in_grid else 1, comm.rank)
    if bubble_rank < rows * cols:
        chi0_mat = (
            mpi_utils.bcast_rows(grid_comm, gchi0_q0_pp.mat if comm.rank == bubble_rank else None, root=bubble_rank)
            if in_grid
            else None
        )
    else:
        chi0_mat = mpi_utils.bcast_rows(
            comm, gchi0_q0_pp.mat if comm.rank == bubble_rank else np.empty(0), root=bubble_rank
        )
    if not in_grid:
        return None

    block = FourPoint(block_mat, channel, k_grid.nk, 0, 2, False, True, True, FrequencyNotation.PP)
    block = block.map_to_full_bz(k_grid, k_grid.nk).decompress_q_dimension().fft(False)
    logger.log_memory_usage(f"Gamma_pp_{channel.value} grid block", block, rows * cols, allowed_ranks=(0,))
    # the kernel prefactor is folded into the persistent vertex once, exactly as the in-memory solve does
    gamma_mm = _gamma_to_matmul_layout(block.mat, scale=0.5 / k_grid.nk_tot / config.sys.beta)
    block.free()

    chi0_full = FourPoint(chi0_mat, SpinChannel.NONE, k_grid.nk, 0, 1, False, True, True, FrequencyNotation.PP)
    chi0_full = chi0_full.decompress_q_dimension()
    chi0_own = _chi0_to_matmul_layout(np.ascontiguousarray(chi0_full.mat[..., col_slice]))
    chi0_full.free()

    gap_shape = k_grid.nk + 2 * (n_bands,) + (n_freq,)
    sign = 1 if channel == SpinChannel.SING else -1
    row_counts = (row_bounds[1:] - row_bounds[:-1]) * int(np.prod(gap_shape[:5]))
    row_displs = np.concatenate(([0], np.cumsum(row_counts[:-1])))

    def mv(gap: np.ndarray):
        r"""
        Applies the pairing kernel to a full flattened gap vector on the solver grid: this rank dresses its own
        :math:`\nu'`-column block of the gap, contracts it with its vertex block, completes the :math:`\nu'` sum by an
        ``Allreduce`` over the row group, reassembles the full :math:`\nu` axis by an ``Allgatherv`` over the column
        group and adds the crossed term, so every rank returns the identical full result.

        :param gap: The flattened gap vector (full length, identical on every grid rank).
        :return: The flattened result of applying the pairing kernel to ``gap``.
        """
        gap6 = gap.reshape(gap_shape)
        gap_gg = sp.fft.fftn(
            _apply_gchi0_pp(chi0_own, np.ascontiguousarray(gap6[..., col_slice]), n_bands),
            axes=(0, 1, 2),
            overwrite_x=True,
        )
        part = _apply_gamma_pp(gamma_mm, gap_gg, n_bands)
        if cols > 1:
            part = np.ascontiguousarray(part)
            reduce_comm.Allreduce(MPI.IN_PLACE, part)
        part = sp.fft.ifftn(part, axes=(0, 1, 2), overwrite_x=True)
        if rows > 1:
            send = np.ascontiguousarray(np.moveaxis(part, -1, 0))
            recv = np.empty((n_freq,) + send.shape[1:], dtype=send.dtype)
            gather_comm.Allgatherv(send, [recv, (row_counts, row_displs)])
            part = np.moveaxis(recv, 0, -1)
        return (part + _crossed_term(part, sign)).flatten()

    def sector_mv(gap: np.ndarray, eps_t: int | None, eps_p: int | None) -> np.ndarray:
        # a projected sector applies Pi mv Pi with the projector of _project_gap_to_sector
        if eps_t is None:
            return mv(gap)
        return _project_gap_to_sector(
            mv(_project_gap_to_sector(gap, gap_shape, eps_t, eps_p, u)), gap_shape, eps_t, eps_p, u
        )

    # get_initial_gap_function draws from a fixed-seed generator, so every lockstep rank computes the same seed
    seed = get_initial_gap_function(gap_shape, channel)
    return _solve_pairing_sectors(
        sector_mv,
        gap_shape,
        sign,
        channel,
        k_grid.nk,
        None,
        (0,),
        seed.flatten(),
        None,
        gamma_mm.dtype,
        u,
    )


# --- Eliashberg eigensolver (Lanczos / ARPACK) ---
@lru_cache(maxsize=1)
def _openblas_thread_slot_cap() -> int | None:
    r"""
    Returns the build-time thread capacity (``NUM_THREADS``) of the loaded OpenBLAS libraries, or ``None`` when no
    OpenBLAS is loaded. OpenBLAS reserves working-buffer slots for at most that many calling threads at build time;
    a process calling into it from more threads than that overflows into an auxiliary bookkeeping path
    ("precompiled NUM_THREADS exceeded" warning) that is unreliable under concurrency and crashes ("Bad memory
    unallocation!", segmentation faults), so every solver thread budget must stay at or below this capacity.
    ``openblas_set_num_threads`` clamps its argument to the build maximum, so probing with an oversized limit and
    reading the value back yields that maximum; the previous thread settings are restored afterwards. The result is
    cached - the capacity is a fixed property of the loaded libraries. Other BLAS implementations (MKL, BLIS) size
    their buffers per calling thread and need no cap.

    :return: The smallest build-time thread capacity among the loaded OpenBLAS libraries, ``None`` without OpenBLAS.
    """
    openblas_libs = ThreadpoolController().select(internal_api="openblas")
    if not openblas_libs.lib_controllers:
        return None
    with openblas_libs.limit(limits=1 << 15):
        return min(lib.num_threads for lib in openblas_libs.lib_controllers)


def _clamp_to_openblas_slot_cap(budget: int) -> int:
    r"""
    Clamps a thread budget to the loaded OpenBLAS build's thread capacity (see :func:`_openblas_thread_slot_cap`);
    a larger budget would call OpenBLAS from more threads than it has buffer slots for and crash. The budget passes
    through unchanged when no OpenBLAS is loaded.

    :param budget: The thread budget to clamp.
    :return: The budget, clamped to the OpenBLAS thread capacity if OpenBLAS is loaded.
    """
    cap = _openblas_thread_slot_cap()
    return budget if cap is None else min(budget, cap)


def _solver_thread_budget() -> int:
    r"""
    Returns the BLAS/FFT thread budget for the in-memory Lanczos solve: the size of this process's CPU affinity
    mask (at least 1), clamped to the OpenBLAS thread capacity (see :func:`_clamp_to_openblas_slot_cap`). During
    that solve only the one or two solver ranks work while the other ranks of the node wait at the post-solve
    broadcast, so the solver may use every core its affinity mask allows - the launcher's binding stays the single
    source of truth (under a strict one-core-per-rank binding this is 1 and the threading is a no-op). Falls back
    to 1 where the affinity API does not exist (non-Linux platforms).

    :return: The thread budget as an int.
    """
    try:
        budget = max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return 1
    return _clamp_to_openblas_slot_cap(budget)


def _chi0_to_matmul_layout(chi0_mat: np.ndarray) -> np.ndarray:
    r"""
    Returns the bare pp bubble in batched-matmul layout ``[x, y, z, v, o2, o2]`` (a view, no copy) for use with
    :func:`_apply_gchi0_pp`. ``chi0_mat`` is in the einsum layout ``[x, y, z, a, b, c, d, v]``.

    :param chi0_mat: The bubble array :math:`\chi_0^{\mathrm{pp}}`, shape ``[x, y, z, o, o, o, o, v]``.
    :return: A view reshaped/transposed to ``[x, y, z, v, o2, o2]`` (rows ``(a, b)``, columns ``(c, d)``).
    """
    nqx, nqy, nqz, nb = chi0_mat.shape[:4]
    v = chi0_mat.shape[-1]
    return np.moveaxis(chi0_mat.reshape(nqx, nqy, nqz, nb * nb, nb * nb, v), -1, 3)


_MIRROR_TOL = 1e-6  # tolerance on the deviation of a mirror eigenvalue from +/-1


def _gap_orbital_mirrors(n_bands: int) -> dict:
    r"""
    Collects the orbital part :math:`U_i` of the single-axis coordinate mirrors :math:`k_i \to -k_i` for the gap
    symmetrization (see :func:`_mirror_operator`) by solving for them from :math:`H(\mathbf{k})` with the same U-solver
    the automatic symmetry discovery uses (:func:`~dgamore.symmetry_reduction.find_coordinate_mirror_orbital_unitaries`,
    a handful of small eigen-solves on the cached :math:`H(\mathbf{k})`). Nothing is hard-coded per orbital set - the
    mirrors are read off the Hamiltonian, so :math:`t_{2g}`, :math:`e_g` and any other Wannier basis are covered alike,
    and no symmetry mode is assumed (the solve does not need ``symmetries: auto``).

    A single-orbital gap needs no orbital factor (a :math:`1 \times 1` unitary is a phase, and it cancels in
    :math:`U \Delta U^\dagger`), and a failure to determine the mirrors is not fatal: the symmetrization then
    reduces to momentum-only mirrors and falls back to the Loewdin basis wherever those do not resolve a multiplet.

    :param n_bands: Number of orbitals of the gap.
    :return: A dict ``{axis: U}`` (empty when there is nothing to apply or the mirrors cannot be determined; with
        uncorrelated orbitals an axis whose mirror is no symmetry of the correlated block maps to ``None``, with a
        warning).
    """
    if n_bands < 2:
        return {}

    orbitals = config.sys.correlated_orbitals
    try:
        ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
        mirrors = find_coordinate_mirror_orbital_unitaries(np.asarray(ek, dtype=np.complex128), orbitals=orbitals)
    except Exception as exc:  # never let a symmetry probe abort a solve; momentum-only mirrors still work
        log = config.logger.debug if orbitals is None else config.logger.warning
        log(f"Could not determine the orbital mirror matrices for the gap symmetrization: {exc}")
        return {}
    dropped = sorted(axis for axis, u in mirrors.items() if u is None)
    if dropped:
        config.logger.warning(
            f"The orbital mirrors of axes {dropped} are no symmetry of the correlated block; the gap symmetrization "
            f"reflects only the momenta there."
        )
    return mirrors


def _mirror_acts_on_orbitals(u: np.ndarray | None) -> bool:
    r"""
    Whether an orbital mirror matrix acts non-trivially on the gap. The gap transforms by conjugation,
    :math:`\Delta \to U \Delta U^\dagger`, which is the identity map exactly when :math:`U` is a multiple of the
    identity - a global phase cancels between the two factors.

    :param u: The orbital mirror matrix, or ``None``.
    :return: True if conjugation by ``u`` is not the identity map.
    """
    if u is None or u.shape[0] < 2:
        return False
    return not np.allclose(u, (np.trace(u) / u.shape[0]) * np.eye(u.shape[0]), atol=1e-8)


def _validated_orbital_mirrors(gap_shape: tuple, orbital_mirrors: dict | None) -> dict:
    r"""
    Normalizes the caller-supplied orbital mirror matrices, dropping entries that do not match the gap's orbital
    dimension (a mirror discovered on a different orbital set must never be applied silently).

    :param gap_shape: Full gap shape ``[kx, ky, kz, o1, o2, 2*niv_pp]``.
    :param orbital_mirrors: A dict ``{axis: U}`` of orbital mirror matrices, or ``None``.
    :return: The validated dict ``{axis: U}`` as complex arrays (empty when nothing usable is supplied).
    """
    n_orb = gap_shape[3]
    return {
        int(axis): np.asarray(u, dtype=np.complex128)
        for axis, u in (orbital_mirrors or {}).items()
        if u is not None and np.shape(u) == (n_orb, n_orb)
    }


def _mirror_operator(gap_shape: tuple, axis: int, u: np.ndarray | None):
    r"""
    Builds the single-axis mirror operator acting on a flattened gap column. A point-group mirror :math:`k_i \to -k_i`
    acts on the orbital indices as well as on the momenta, so with the orbital matrix :math:`U` of that mirror
    (:math:`H(M_i \mathbf{k}) = U H(\mathbf{k}) U^\dagger`, see
    :func:`~dgamore.symmetry_reduction.find_coordinate_mirror_orbital_unitaries`) the gap transforms as

    .. math:: \Delta_{o_1 o_2}(\mathbf{k}) \to \left[U\, \Delta(M_i \mathbf{k})\, U^\dagger\right]_{o_1 o_2} .

    For :math:`t_{2g}` orbitals :math:`U` is the diagonal sign matrix of the mirror and this reduces to the familiar
    :math:`s_{o_1} s_{o_2} \Delta_{o_1 o_2}(M_i \mathbf{k})`. Dropping the orbital factor (as a momentum-only reflection
    does) leaves an operator that is not a symmetry of the multi-orbital pairing kernel: a partner carrying weight in
    both orbital sectors then measures the orbital-diagonal minus the orbital-off-diagonal weight instead of :math:`\pm
    1`, and the cleanliness check in :func:`_orient_cluster_by_mirrors` rejects it.

    :param gap_shape: Full gap shape ``[kx, ky, kz, o1, o2, 2*niv_pp]``.
    :param axis: The reflected momentum axis (0, 1 or 2).
    :param u: The orbital matrix of that mirror, or ``None``/a multiple of the identity for a momentum-only mirror.
    :return: A callable mapping a flattened gap column to its mirrored image.
    """
    idx = (gap_shape[axis] - np.arange(gap_shape[axis])) % gap_shape[axis]
    take = [slice(None)] * len(gap_shape)
    take[axis] = idx
    take = tuple(take)

    if not _mirror_acts_on_orbitals(u):
        return lambda column: column.reshape(gap_shape)[take].ravel()

    return lambda column: np.einsum(
        "ap,...pqv,bq->...abv", u, column.reshape(gap_shape)[take], u.conj(), optimize=True
    ).ravel()


def _orient_cluster_by_mirrors(
    block: np.ndarray, gap_shape: tuple, tol: float = _MIRROR_TOL, orbital_mirrors: dict | None = None
) -> np.ndarray | None:
    r"""
    Rotates an orthonormal degenerate cluster onto the common eigenbasis of the single-axis coordinate mirrors
    :math:`k_i \to -k_i` and orders the partners lexicographically by the tuple of axes each one is odd under. The
    mirrors mutually commute, so one generic real combination of their projections into the cluster (weights
    :math:`1, \sqrt{2}, \sqrt{3}`, giving a distinct eigenvalue per sign pattern) shares their eigenvectors; each
    partner's mirror eigenvalues are read off as Rayleigh quotients. Single-axis :math:`p`-like partners sort
    ``x, y, z`` and two-axis :math:`d`-like partners (:math:`d_{xy}`, :math:`d_{xz}`, :math:`d_{yz}`) sort
    ``xy, xz, yz``.

    Each mirror acts on the orbital indices as well as on the momenta (see :func:`_mirror_operator`); an axis whose
    orbital matrix is non-trivial therefore resolves partners even when its momentum action does not, which is what
    makes purely local (momentum-independent) multiplets - a local :math:`T_{1g}` triplet, say - resolvable at all.

    Two partners sharing a sign pattern span a subspace the mirrors do not resolve: the combined matrix is
    degenerate there, so the eigenvectors within it are fixed by floating-point noise alone and the sort cannot
    separate them either. Coordinate mirrors alone cannot split an :math:`E_g` doublet, for instance, because an
    orbital-diagonal state picks up :math:`s_o s_o = +1` on every axis; separating those partners would take a
    three-fold rotation about :math:`[111]`. Rotating by noise-fixed eigenvectors would scramble the cluster, so
    such a cluster is rejected and the caller keeps its input basis.

    :param block: Orthonormal cluster, one flattened gap function per column.
    :param gap_shape: Full gap shape ``[kx, ky, kz, o1, o2, 2*niv_pp]``.
    :param tol: Tolerance on the deviation of a mirror Rayleigh quotient from :math:`\pm 1`.
    :param orbital_mirrors: A dict ``{axis: U}`` with the orbital part of each coordinate mirror; entries whose shape
        does not match the gap's orbital dimension are ignored, and a missing axis falls back to a momentum-only
        reflection (exact for a single-orbital gap, but not a kernel symmetry for a multi-orbital one).
    :return: The reordered cluster, or ``None`` when no axis is resolved, a partner is not a clean :math:`\pm 1`
        mirror eigenstate, or two partners share the same mirror sign pattern.
    """
    mirrors = _validated_orbital_mirrors(gap_shape, orbital_mirrors)
    # an axis resolves partners through its momentum reflection, its orbital matrix, or both
    axes = [axis for axis in (0, 1, 2) if gap_shape[axis] > 1 or _mirror_acts_on_orbitals(mirrors.get(axis))]
    if not axes:
        return None

    projected = []
    for axis in axes:
        reflect = _mirror_operator(gap_shape, axis, mirrors.get(axis))
        mirrored = np.stack([reflect(block[:, i]) for i in range(block.shape[1])], axis=1)
        m = block.conj().T @ mirrored
        projected.append(0.5 * (m + m.conj().T))

    weights = np.sqrt(np.arange(1, len(projected) + 1, dtype=float))  # 1, sqrt(2), sqrt(3): generic, deterministic
    _, vecs = np.linalg.eigh(sum(w * m for w, m in zip(weights, projected)))

    keys = []
    for col in range(vecs.shape[1]):
        c = vecs[:, col]
        signature = np.array([(c.conj() @ m @ c).real for m in projected])
        if np.any(np.abs(np.abs(signature) - 1.0) > tol):  # not a clean +/-1 mirror eigenstate
            return None
        keys.append(tuple(int(a) for a in np.flatnonzero(signature < 0.0)))

    if len(set(keys)) < len(keys):  # a shared sign pattern spans a subspace the mirrors leave unresolved
        return None

    block = block @ vecs
    return block[:, sorted(range(len(keys)), key=lambda col: keys[col])]


def symmetrize_degenerate_gaps(
    lambdas: np.ndarray, gaps: np.ndarray, gap_shape: tuple, tol: float = 1e-4, orbital_mirrors: dict | None = None
) -> np.ndarray:
    r"""
    Orthonormalizes the eigenvectors returned by the Lanczos solver within clusters of (near-)degenerate
    eigenvalues and rotates every cluster to a mirror-adapted basis. The pairing kernel is not Hermitian in the
    plain inner product, so ARPACK may return oblique (mutually non-orthogonal) combinations
    inside a degenerate cluster: the cluster subspace is symmetry-covariant, but the returned vectors then do not
    form the symmetry-adapted partners.

    Per cluster the following steps are applied: (i) Loewdin orthonormalization, i.e. :math:`S^{-1/2}` applied to
    the cluster overlap matrix :math:`S`, which yields the orthonormal basis closest to the input vectors. A cluster
    of (nearly) linearly dependent vectors is exempt from it and from the orientation of step (ii), since
    :math:`S^{-1/2}` would amplify exactly the noise that makes such vectors distinct; it is still normalized and
    phase-fixed, so step (iii) holds for every returned vector without exception; (ii)
    for doublets, the mirror operation

    .. math:: M_y: \Delta_{o_1 o_2}(k_x, k_y, k_z, \nu) \to
        \left[U_y\, \Delta(k_x, -k_y, k_z, \nu)\, U_y^\dagger\right]_{o_1 o_2}

    is diagonalized within the cluster, ordering the even (:math:`+1`, :math:`p_x`-like) partner first and the
    odd (:math:`-1`, :math:`p_y`-like) partner second, but only when the two partners come out as clean, oppositely
    signed :math:`\pm 1` eigenstates (an :math:`E_g` doublet, even under every coordinate mirror, is not resolved
    this way and keeps the Loewdin basis); every cluster of three or more members is handled by
    :func:`_orient_cluster_by_mirrors`, which diagonalizes the single-axis coordinate mirrors of the resolved axes
    simultaneously and orders the partners by the axes each one is odd under (:math:`p`-like as ``x, y, z``,
    two-axis :math:`d`-like as ``xy, xz, yz``), provided every partner is a clean :math:`\pm 1` eigenstate and no
    two partners share a sign pattern (otherwise the mirrors do not resolve the cluster and the Loewdin basis is
    kept); (iii) the global phase of every vector is fixed such that its largest-magnitude element is real and
    positive. Eigenvalues are not modified; vectors of non-degenerate eigenvalues are only phase-fixed. Enabled via
    ``symmetrize_degenerate_gaps`` of :class:`~dgamore.config.EliashbergConfig`.

    The mirrors act on the orbital indices as well as on the momenta (:math:`U_y` above, see
    :func:`_mirror_operator`). Without that orbital factor the operator is not a symmetry of a multi-orbital pairing
    kernel, and any multiplet with weight in both the orbital-diagonal and the orbital-off-diagonal sector fails the
    :math:`\pm 1` check and falls back to the Loewdin basis; with it, purely local (momentum-independent) multiplets
    become resolvable too, which a momentum-only mirror cannot do because it acts on them as the identity.

    :param lambdas: Eigenvalues sorted in descending order.
    :param gaps: Eigenvector matrix ``[n, n_eig]`` with one flattened gap function per column.
    :param gap_shape: Full gap shape ``[kx, ky, kz, o1, o2, 2*niv_pp]``, used to locate the momentum axes.
    :param tol: Relative tolerance for clustering neighboring eigenvalues as degenerate.
    :param orbital_mirrors: A dict ``{axis: U}`` with the orbital part of each coordinate mirror, as produced by
        :func:`_gap_orbital_mirrors`. Omitting it reduces the mirrors to momentum-only reflections, which is exact
        for a single-orbital gap only.
    :return: The symmetrized eigenvector matrix ``[n, n_eig]``.
    """
    mirrors = _validated_orbital_mirrors(gap_shape, orbital_mirrors)
    mirror_y = _mirror_operator(gap_shape, 1, mirrors.get(1))

    clusters = [[0]]
    for i in range(1, gaps.shape[1]):
        if abs(lambdas[i] - lambdas[i - 1]) <= tol * max(abs(lambdas[i]), 1e-12):
            clusters[-1].append(i)
        else:
            clusters.append([i])

    gaps = gaps.copy()
    for cluster in clusters:
        block = gaps[:, cluster].astype(np.complex128)
        block /= np.linalg.norm(block, axis=0)

        independent = True
        if len(cluster) > 1:
            overlap = block.conj().T @ block
            eigs, u = np.linalg.eigh(overlap)
            # dependent vectors keep only the stable normalization and phase fix; S^{-1/2} and the mirror step
            # would amplify the noise that makes them distinct (see the step (i) note in the docstring)
            independent = bool(eigs.min() >= 1e-12)
            if independent:
                block = block @ (u @ np.diag(eigs**-0.5) @ u.conj().T)

        if independent and len(cluster) == 2:
            mirrored = np.stack([mirror_y(block[:, i]) for i in range(2)], axis=1)
            mirror_block = block.conj().T @ mirrored
            mirror_block = 0.5 * (mirror_block + mirror_block.conj().T)
            mirror_eigs, mirror_vecs = np.linalg.eigh(mirror_block)
            # rotate only when M_y resolves the doublet into clean, oppositely signed +/-1 eigenstates; otherwise
            # keep the Loewdin basis (an E_g doublet is even under every mirror, so noise would fix the rotation)
            if np.all(np.abs(np.abs(mirror_eigs) - 1.0) <= _MIRROR_TOL) and mirror_eigs[0] * mirror_eigs[1] < 0.0:
                # order the even (+1, p_x-like) partner first and the odd (-1, p_y-like) partner second
                block = block @ mirror_vecs[:, ::-1]
        elif independent and len(cluster) >= 3:
            oriented = _orient_cluster_by_mirrors(block, gap_shape, orbital_mirrors=mirrors)
            if oriented is not None:
                block = oriented

        for col in range(block.shape[1]):
            mags = np.abs(block[:, col])
            if mags.max() == 0:  # an all-zero vector has no phase to fix (and dividing by it would yield nan)
                continue
            # tie-break on the first index among the maximal-modulus elements, stable against fp noise
            phase = block[np.flatnonzero(mags >= mags.max() * (1.0 - 1e-8))[0], col]
            block[:, col] *= phase.conjugate() / abs(phase)
        gaps[:, cluster] = block

    return gaps


def _apply_gchi0_pp(
    chi0_mm: np.ndarray, gap: np.ndarray, n_bands: int, executor: ThreadPoolExecutor = None, n_workers: int = 1
) -> np.ndarray:
    r"""
    Batched-matmul equivalent of ``np.einsum("xyzabcdv,xyzcdv->xyzabv", chi0, gap)`` (multiply the gap by the bare pp
    bubble per momentum and frequency). ``np.matmul`` is both faster than ``np.einsum`` and far leaner here: einsum
    materializes a vertex-sized internal temporary, the matmul allocates only the gap-sized output. With an
    ``executor`` the batch is split into up to ``n_workers`` contiguous chunks of the leading momentum axis and
    contracted concurrently: the chunks are pure slices of ``chi0_mm`` (no reshape of the bubble that could
    silently copy it) and every worker writes its own slice of the one gap-sized output buffer, so the threaded
    path allocates exactly what the serial path does and the result is bit-equal to it.

    :param chi0_mm: The bubble in matmul layout from :func:`_chi0_to_matmul_layout`, shape ``[x, y, z, v, o2, o2]``.
    :param gap: The gap vector, reshapeable to ``[x, y, z, o, o, v]``.
    :param n_bands: Number of orbitals ``o``.
    :param executor: Optional thread pool for the momentum-batch parallel path (``None`` runs serially).
    :param n_workers: Number of contiguous momentum chunks when ``executor`` is given.
    :return: ``chi0 @ gap`` in shape ``[x, y, z, o, o, v]``.
    """
    nqx, nqy, nqz, v = chi0_mm.shape[0], chi0_mm.shape[1], chi0_mm.shape[2], chi0_mm.shape[3]
    oo = n_bands * n_bands
    gap_r = np.moveaxis(gap.reshape(nqx, nqy, nqz, oo, v), -1, 3)[..., None]  # [x, y, z, v, o2, 1]
    if executor is None:
        out = np.matmul(chi0_mm, gap_r)[..., 0]  # [x, y, z, v, o2]
        return np.moveaxis(out, 3, -1).reshape(nqx, nqy, nqz, n_bands, n_bands, v)

    out = np.empty((nqx, nqy, nqz, v, oo, 1), dtype=np.result_type(chi0_mm.dtype, gap_r.dtype))
    bounds = np.linspace(0, nqx, n_workers + 1).astype(int)
    futures = [
        executor.submit(np.matmul, chi0_mm[i:j], gap_r[i:j], out[i:j]) for i, j in zip(bounds[:-1], bounds[1:]) if j > i
    ]
    for future in futures:
        future.result()
    return np.moveaxis(out[..., 0], 3, -1).reshape(nqx, nqy, nqz, n_bands, n_bands, v)


def _gamma_to_matmul_layout(gamma_mat: np.ndarray, scale: float = 1.0) -> np.ndarray:
    r"""
    Materializes the pp pairing vertex in batched-matmul layout ``[x, y, z, o2*nv, o2*np]`` (rows ``(a, b, v)``,
    columns ``(c, d, p)``) for :func:`_apply_gamma_pp`, scaled by ``scale``. The einsum layout is
    ``[x, y, z, a, c, b, d, v, p]`` with the orbitals interleaved (``a, c, b, d``), so a transpose to
    ``(a, b, v, c, d, p)`` precedes the copying reshape; a scale factor other than one rides along with that single
    pass (the same one multiply per element as scaling the layout afterwards). The per-matvec ``np.matmul`` then
    allocates only the gap-sized output.

    :param gamma_mat: The pp vertex, shape ``[x, y, z, a, c, b, d, v, p]`` (``v`` may be a frequency slice).
    :param scale: Factor multiplied onto every element, e.g. the kernel prefactor (one leaves the values untouched).
    :return: A contiguous array ``[x, y, z, o2*nv, o2*np]`` in matmul layout.
    """
    nqx, nqy, nqz, nb = gamma_mat.shape[:4]
    nv, npp = gamma_mat.shape[-2], gamma_mat.shape[-1]
    out = np.empty((nqx, nqy, nqz, nb * nb * nv, nb * nb * npp), dtype=gamma_mat.dtype)
    transposed = np.transpose(gamma_mat, (0, 1, 2, 3, 5, 7, 4, 6, 8))  # [x,y,z,a,b,v,c,d,p]
    np.multiply(transposed, scale, out=out.reshape(nqx, nqy, nqz, nb, nb, nv, nb, nb, npp))
    return out


def _apply_gamma_pp(
    gamma_mm: np.ndarray, gap_gg: np.ndarray, n_bands: int, executor: ThreadPoolExecutor = None, n_workers: int = 1
) -> np.ndarray:
    r"""
    Batched-matmul equivalent of ``np.einsum("xyzacbdvp,xyzcdp->xyzabv", gamma, gap_gg)`` (contract the pairing vertex
    with the gap over ``(c, d, p)``). Faster and leaner than ``np.einsum`` (see :func:`_apply_gchi0_pp`). With an
    ``executor`` the momentum batch is split into ``n_workers`` contiguous chunks contracted concurrently (each
    worker writes its own output slice, so the result is bit-equal to the serial path): the contraction is a batch
    of many small per-k GEMV products, which parallelizes over the batch but not inside one product (a raised BLAS
    thread pool pays per-call synchronization on every small GEMV and runs slower).

    :param gamma_mm: The vertex in matmul layout from :func:`_gamma_to_matmul_layout`, shape ``[x, y, z, o2*nv, o2*np]``.
    :param gap_gg: The transformed gap, shape ``[x, y, z, c, d, p]``.
    :param n_bands: Number of orbitals ``o``.
    :param executor: Optional thread pool for the momentum-batch parallel path (``None`` runs serially).
    :param n_workers: Number of contiguous momentum chunks when ``executor`` is given.
    :return: ``gamma @ gap_gg`` in shape ``[x, y, z, o, o, nv]``.
    """
    nqx, nqy, nqz = gamma_mm.shape[:3]
    oo = n_bands * n_bands
    npp = gap_gg.shape[-1]
    nv = gamma_mm.shape[3] // oo
    gg_r = gap_gg.reshape(nqx, nqy, nqz, oo * npp)[..., None]  # [x, y, z, o2*np, 1]
    if executor is None:
        out = np.matmul(gamma_mm, gg_r)[..., 0]  # [x, y, z, o2*nv]
        return out.reshape(nqx, nqy, nqz, n_bands, n_bands, nv)

    nk = nqx * nqy * nqz
    mm_flat = gamma_mm.reshape(nk, oo * nv, oo * npp)
    gg_flat = gg_r.reshape(nk, oo * npp, 1)
    out = np.empty((nk, oo * nv, 1), dtype=gamma_mm.dtype)
    bounds = np.linspace(0, nk, n_workers + 1).astype(int)
    futures = [
        executor.submit(np.matmul, mm_flat[i:j], gg_flat[i:j], out[i:j])
        for i, j in zip(bounds[:-1], bounds[1:])
        if j > i
    ]
    for future in futures:
        future.result()
    return out[..., 0].reshape(nqx, nqy, nqz, n_bands, n_bands, nv)


@lru_cache(maxsize=1)
def _wedge_orbits(k_grid: KGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    r"""
    Groups the flat indices of the real-space grid into the stars of the grid's point-like symmetry operations
    (index permutations with orbital rotations, no translations, no antiunitary members), under which the
    Fourier-transformed vertex obeys :math:`\Gamma(P\mathbf{R}) = U_P\,\Gamma(\mathbf{R})\,U_P^\dagger` with the same
    rotation as in momentum space. On an explicit-symmetry grid these are the grid's own irreducible-BZ maps; on an
    auto-symmetry grid they come from :func:`~dgamore.symmetry_reduction.point_group_orbits`. The result of the last
    grid is cached.

    :param k_grid: The momentum grid (its index maps apply to the real-space grid alike).
    :return: ``(points, offsets, order, us)``: the representative grid index of every star, the star boundaries
        into ``order``, every grid index sorted by star, and per grid index the rotation carrying the representative
        onto it (``None`` when the grid applies no orbital rotation).
    """
    if k_grid.is_auto:
        rep, us = point_group_orbits(k_grid._auto_group, tuple(k_grid.nk))
        reps, inv = np.unique(rep, return_inverse=True)
    else:
        reps, inv, us = k_grid.irrk_ind, k_grid.irrk_inv.ravel(), None
    order = np.argsort(inv, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(np.bincount(inv, minlength=len(reps)))))
    return reps, offsets, order, us


_SYMMETRY_MATRICES = {
    KnownSymmetries.X_INV: np.diag([-1, 1, 1]),
    KnownSymmetries.Y_INV: np.diag([1, -1, 1]),
    KnownSymmetries.Z_INV: np.diag([1, 1, -1]),
    KnownSymmetries.X_Y_SYM: np.eye(3, dtype=np.int64)[[1, 0, 2]],
    KnownSymmetries.X_Z_SYM: np.eye(3, dtype=np.int64)[[2, 1, 0]],
    KnownSymmetries.Y_Z_SYM: np.eye(3, dtype=np.int64)[[0, 2, 1]],
    KnownSymmetries.X_Y_INV: np.diag([-1, -1, 1]),
}


def block_size(k_grid: KGrid, parity: int | None = None) -> int:
    r"""
    Returns the number of starting vectors the Eliashberg eigensolver takes with ``resolve_degenerate_multiplets``, so
    that every eigenvalue multiplicity the grid's symmetry forces is found in exact arithmetic (see :func:`~dgamore.symmetry_reduction.forced_multiplicity`):
    the discovered group on an auto-symmetry grid, the group generated by the listed operations otherwise. A
    single-vector Krylov method reaches one direction of each eigenspace, so the partners of a forced multiplet only
    enter through rounding and can be missing from the converged eigenpairs.

    :param k_grid: The momentum grid.
    :param parity: The momentum-inversion parity of the sector (one band), or ``None`` for every representation.
    :return: The block size, 1 when the group forces no multiplet or ``resolve_degenerate_multiplets`` is off.
    """
    if not config.eliashberg.resolve_degenerate_multiplets:
        return 1
    nk = tuple(k_grid.nk)
    if k_grid.is_auto:
        return forced_multiplicity(k_grid._auto_group, nk, parity)
    matrices = [_SYMMETRY_MATRICES.get(sym) for sym in k_grid.symmetries or []]
    ops = [
        {"M": m, "q": np.zeros(3, dtype=np.int64), "U": np.eye(1), "sigma": 1, "conj": False}
        for m in matrices
        if m is not None and _M_preserves_grid(m, nk)
    ]
    return forced_multiplicity(_close_group(ops, 1, nk), nk, parity)


def wedge_window_points(k_grid: KGrid) -> int:
    r"""
    Returns the number of real-space points a channel's vertex window holds in the team solve: the star
    representatives of the grid's point-like operations (see :func:`_wedge_orbits`), i.e. the whole grid when they
    reduce nothing.

    :param k_grid: The momentum grid.
    :return: The number of window points.
    """
    return len(_wedge_orbits(k_grid)[0])


def _build_shared_vertex_window(
    gamma_r_pp: FourPoint, node_comm: MPI.Comm, norm: float, wedge_points: np.ndarray | None = None
) -> tuple[np.ndarray, MPI.Win | None]:
    r"""
    Puts one channel's pairing vertex, in matmul layout, into a single MPI shared-memory window of its node. The
    node root holds the gathered irreducible-BZ vertex on entry and publishes it through a temporary node-shared
    window; every rank of the node then builds its share of the full-BZ matmul layout straight into the vertex
    window, one column block at a time (one fermionic frequency :math:`\nu` and a range of :math:`\nu'` columns,
    sized by :data:`~dgamore.memory_estimator.TEAM_BUILD_CHUNK_BYTES`): each block runs through the same
    :class:`FourPoint` chain as the single-rank build (map to the full BZ, Fourier transform over the momenta,
    reorder the legs ``abcd->badc``) and is written scaled by the kernel prefactor. Every step acts on each column
    independently, so the window equals the layout the single-rank solve builds bit for bit. With ``wedge_points``
    only the rows of those real-space points are kept (the star representatives of :func:`_wedge_orbits`). The
    node holds the windows, the irreducible source and one block per rank; the vertex is **consumed** on the node
    root.

    :param gamma_r_pp: The channel's pairing vertex, the whole irreducible BZ on the node root (ignored elsewhere).
    :param node_comm: The node-local communicator.
    :param norm: The kernel prefactor :math:`1 / (2 \beta n_{\mathbf{k}})` folded into the vertex.
    :param wedge_points: Flat indices of the real-space points to keep, or ``None`` for the whole grid.
    :return: ``(gamma_mm, win)`` - the shared matmul-layout vertex, ``[x, y, z, o2*nv, o2*np]`` on the full grid or
        ``[r, o2*nv, o2*np]`` on the wedge, and its window.
    """
    k_grid = config.lattice.k_grid
    size, rank = node_comm.Get_size(), node_comm.Get_rank()
    irr_shape, dtype = node_comm.bcast((gamma_r_pp.mat.shape, gamma_r_pp.mat.dtype) if rank == 0 else None, root=0)
    irr, irr_win = gamma_r_pp.mat, None
    if size > 1:
        irr, irr_win = mpi_utils.allocate_node_shared_array(node_comm, irr_shape, dtype)
        if rank == 0:
            irr[...] = gamma_r_pp.mat
            gamma_r_pp.free()
        node_comm.Barrier()

    nb, nv, npp = irr_shape[1], irr_shape[-2], irr_shape[-1]
    rows, cols = nb * nb * nv, nb * nb * npp
    shape = tuple(k_grid.nk) + (rows, cols) if wedge_points is None else (len(wedge_points), rows, cols)
    gamma_mm, win = mpi_utils.allocate_node_shared_array(node_comm, shape, dtype)
    target = gamma_mm.reshape(-1, nb, nb, nv, nb, nb, npp)
    columns = team_build_columns(k_grid.nk_tot, nb, npp)
    chunks = [(v, p0, min(p0 + columns, npp)) for v in range(nv) for p0 in range(0, npp, columns)]
    for i in np.array_split(np.arange(len(chunks)), size)[rank]:
        v, p0, p1 = chunks[i]
        block = FourPoint(
            irr[..., v, p0:p1], gamma_r_pp.channel, k_grid.nk, 0, 1, False, True, True, FrequencyNotation.PP
        )
        block = block.map_to_full_bz(k_grid, k_grid.nk).decompress_q_dimension().fft(False)
        block = block.permute_orbitals("abcd->badc", False).mat.reshape(-1, nb, nb, nb, nb, p1 - p0)
        if wedge_points is not None:
            block = block[wedge_points]
        np.multiply(np.transpose(block, (0, 1, 3, 2, 4, 5)), norm, out=target[:, :, :, v, :, :, p0:p1])
    del irr, target
    node_comm.Barrier()
    nonlocal_sde._free_shared_window(irr_win, node_comm)
    if size == 1:
        gamma_r_pp.free()
    return gamma_mm, win


def _team_matvec(
    team_comm: MPI.Comm,
    gamma_by_channel: dict[SpinChannel, np.ndarray],
    chi0_mm: np.ndarray,
    scratch: list[np.ndarray],
    gap_shape: tuple,
    n_bands: int,
    orbits: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None = None,
    u: np.ndarray | None = None,
) -> tuple[Callable, tuple[int, int]]:
    r"""
    Builds the pairing-kernel matvec of a team of ranks that share their node's vertex windows. Every rank holds one
    frequency block of the vectors and every matvec runs in lockstep stages over the team's gap-sized scratch windows
    (the gap, its dressed transform, the direct and the crossed term), each rank working on its own share: the sector
    projection of the input (see :func:`_project_frequency_block`), the bubble multiply and the momentum FFT on a
    contiguous block of fermionic frequencies, the vertex contraction on a contiguous block of momenta, the crossed term
    of the frequency block (see :func:`_crossed_term`, which reads the mirror block of the direct term, so it is staged
    in its window before any block is overwritten), its addition and the inverse FFT, and for a projected sector the
    projection of the result on that block. Every element is computed by the same operations as in the single-rank
    sector matvec, so the result is bit-identical to it without an orbital action, and agrees to rounding with one (the
    orbital rotation's einsum picks its contraction path by the block's shape). With ``orbits`` the windows hold the
    irreducible wedge of the real-space grid and each rank contracts the stars of its irreducible points instead, one
    matrix product per star with the members' right-hand sides as columns (rotated into and out of the irreducible
    point's orbital frame on an auto-symmetry grid: the layout's row and column pairs both carry the conjugate unitary,
    so :math:`W_{(12),(ab)} = U^*_{1a} U^*_{2b}` enters as :math:`W^\dagger` on the right-hand sides and as :math:`W` on
    the result); the result then equals the single-rank one to rounding.

    :param team_comm: The team communicator (rank 0 is the lead).
    :param gamma_by_channel: The node-shared matmul-layout vertices per channel, ``[x, y, z, o2*nv, o2*np]`` on the
        full grid or ``[r, o2*nv, o2*np]`` on the wedge.
    :param chi0_mm: The node-shared bubble in matmul layout (see :func:`_chi0_to_matmul_layout`).
    :param scratch: Four gap-shaped team-shared arrays.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param n_bands: Number of orbitals ``o``.
    :param orbits: The stars of :func:`_wedge_orbits` for wedge windows, else ``None``.
    :param u: The orbital action of the inversion for the parity sectors (see :func:`_project_gap_to_sector`).
    :return: ``(apply_block, (v0, v1))``: the collective sector matvec on this rank's frequency block
        ``[..., v0:v1]`` of the gap, ``apply_block(channel, eps_t, eps_p, block) -> block`` (``eps_t`` of ``None``
        applies the raw kernel; every team rank calls it with the same sector), and the block bounds.
    """
    gap_w, gg_w, direct_w, crossed_w = scratch
    size, rank = team_comm.Get_size(), team_comm.Get_rank()
    nqx, nqy, nqz = gap_shape[:3]
    nk, nv, oo = nqx * nqy * nqz, gap_shape[-1], n_bands * n_bands
    rows = oo * nv
    columns = np.linspace(0, nv, size + 1).astype(int)
    v0, v1 = int(columns[rank]), int(columns[rank + 1])
    chunks = np.linspace(0, nk, size + 1).astype(int)
    k0, k1 = int(chunks[rank]), int(chunks[rank + 1])
    gg_flat = gg_w.reshape(nk, rows, 1)
    direct_flat = direct_w.reshape(nk, rows, 1)
    if orbits is not None:
        _, offsets, order, us = orbits
        # this rank's irreducible points, balanced by the number of star members (the matrix-product columns)
        bounds = np.searchsorted(offsets, np.linspace(0, nk, size + 1))
        r0, r1 = int(bounds[rank]), int(bounds[rank + 1])
        rotations = None
        if us is not None:
            members = us[order[offsets[r0] : offsets[r1]]]
            rotations = np.einsum("kab,kcd->kacbd", members.conj(), members.conj()).reshape(len(members), oo, oo)

    def contract_stars(gamma_irr: np.ndarray) -> None:
        for r in range(r0, r1):
            members = order[offsets[r] : offsets[r + 1]]
            rhs = gg_flat[members, :, 0].T  # [(c, d, p), m]
            if rotations is not None:
                w = rotations[offsets[r] - offsets[r0] : offsets[r + 1] - offsets[r0]]
                rhs = np.einsum("jab,apj->bpj", w.conj(), rhs.reshape(oo, nv, -1)).reshape(rows, -1)
            out = gamma_irr[r] @ rhs
            if rotations is not None:
                out = np.einsum("jab,bvj->avj", w, out.reshape(oo, nv, -1)).reshape(rows, -1)
            direct_flat[members, :, 0] = out.T

    def run(channel: SpinChannel, eps_t: int | None, eps_p: int | None) -> None:
        gamma_flat = gamma_by_channel[channel].reshape(-1, rows, rows)
        sign = 1 if channel == SpinChannel.SING else -1
        team_comm.Barrier()
        if v1 > v0:
            block = gap_w[..., v0:v1] if eps_t is None else _project_frequency_block(gap_w, v0, v1, eps_t, eps_p, u)
            dressed = _apply_gchi0_pp(chi0_mm[:, :, :, v0:v1], block, n_bands)
            gg_w[..., v0:v1] = sp.fft.fftn(dressed, axes=(0, 1, 2), overwrite_x=True)
            del block, dressed
        team_comm.Barrier()
        if orbits is not None:
            contract_stars(gamma_flat)
        elif k1 > k0:
            np.matmul(gamma_flat[k0:k1], gg_flat[k0:k1], out=direct_flat[k0:k1])
        team_comm.Barrier()
        if v1 > v0:
            crossed_w[..., v0:v1] = _crossed_term(direct_w, sign, v0, v1)
        team_comm.Barrier()
        if v1 > v0:
            direct_w[..., v0:v1] += crossed_w[..., v0:v1]
            direct_w[..., v0:v1] = sp.fft.ifftn(direct_w[..., v0:v1], axes=(0, 1, 2), overwrite_x=True)
        team_comm.Barrier()
        if eps_t is not None:
            # the projected result goes to the (no longer needed) gap window; every rank reads the mirror block
            if v1 > v0:
                gap_w[..., v0:v1] = _project_frequency_block(direct_w, v0, v1, eps_t, eps_p, u)
            team_comm.Barrier()

    def apply_block(channel: SpinChannel, eps_t: int | None, eps_p: int | None, block: np.ndarray) -> np.ndarray:
        gap_w[..., v0:v1] = block.reshape(gap_shape[:-1] + (v1 - v0,))
        run(channel, eps_t, eps_p)
        return (direct_w if eps_t is None else gap_w)[..., v0:v1].copy()

    return apply_block, (v0, v1)


@dataclass(frozen=True)
class LanczosTeam:
    """
    One team of the Eliashberg team solve: the ranks of one node that solve the given sectors together, sharing
    their node's vertex windows.

    :ivar host: The node's hostname.
    :ivar node_root: The lowest rank of the node (the rank that builds the node's vertex windows).
    :ivar ranks: The team's ranks in ascending order; the first is the lead that drives the eigensolver.
    :ivar sectors: The ``(channel, parity)`` sectors the lead solves, one after the other.
    """

    host: str
    node_root: int
    ranks: tuple[int, ...]
    sectors: tuple[tuple[SpinChannel, str], ...]


def plan_lanczos_teams(
    comm: MPI.Comm, node_comm: MPI.Comm, available_bytes: int, niv_pp: int
) -> tuple[tuple[LanczosTeam, ...], ...] | None:
    r"""
    Plans the team solve: the ``(channel, parity)`` sectors are spread as evenly as possible over the nodes (one
    node hosts all of them, two nodes one channel each, four nodes one sector each; nodes beyond the sector count
    host nothing), each channel's pairing vertex is built into a shared window on every node that hosts one of its
    sectors, and a node's ranks are split into one team per sector hosted there (a node with fewer ranks than
    sectors gives every rank a team of one that solves several sectors in turn). The nodes are the groups of
    ``node_comm``, keyed by the lowest rank of each, so the plan and the windows it drives can never disagree on
    who shares memory with whom. Every hosting node is checked against its own free memory with
    :func:`~dgamore.memory_estimator.lanczos_team_bytes`: when a node cannot hold its share at once the channels are
    solved in two rounds, each spread over the nodes the same way, and when even that does not fit the plan is
    ``None`` (the grid solver takes over).

    :param comm: The MPI communicator.
    :param node_comm: The node-local communicator.
    :param available_bytes: This rank's free host memory, allgathered and reduced (minimum) per node.
    :param niv_pp: Number of positive fermionic frequencies of the pp box.
    :return: The rounds of the solve, each a tuple of teams that run concurrently, or ``None`` when no round fits.
    """
    info = comm.allgather((node_comm.bcast(comm.rank, root=0), socket.gethostname(), available_bytes))
    node_ranks: dict[int, list[int]] = {}
    node_host: dict[int, str] = {}
    node_available: dict[int, int] = {}
    for rank, (node_root, host, avail) in enumerate(info):
        node_ranks.setdefault(node_root, []).append(rank)
        node_host[node_root] = host
        node_available[node_root] = min(node_available.get(node_root, avail), avail)
    nodes = list(node_ranks)
    parities = [label for label, _ in _parity_sectors(config.eliashberg.resolve_parity_sectors)]
    sectors = [(channel, parity) for channel in (SpinChannel.SING, SpinChannel.TRIP) for parity in parities]

    def spread(group: list[tuple[SpinChannel, str]]) -> dict[int, list[tuple[SpinChannel, str]]]:
        # ponytail: at most one node per sector; splitting one matvec across nodes is the grid solver's job
        chunks = np.array_split(np.arange(len(group)), min(len(nodes), len(group)))
        return {nodes[i]: [group[j] for j in chunk] for i, chunk in enumerate(chunks)}

    def fits(plan: list[dict[int, list[tuple[SpinChannel, str]]]]) -> bool:
        for hosted in plan:
            for node, node_sectors in hosted.items():
                need = lanczos_team_bytes(
                    config.sys.n_bands,
                    config.lattice.k_grid.nk_tot,
                    config.lattice.k_grid.nk_irr,
                    niv_pp,
                    config.eliashberg.n_eig,
                    len({channel for channel, _ in node_sectors}),
                    len(node_sectors),
                    len(node_ranks[node]),
                    wedge_window_points(config.lattice.k_grid),
                    block_size(config.lattice.k_grid),
                )
                if need > node_available[node] * NODE_MEMORY_FRACTION:
                    return False
        return True

    plan = [spread(sectors)]
    if not fits(plan):
        plan = [spread([s for s in sectors if s[0] == channel]) for channel in (SpinChannel.SING, SpinChannel.TRIP)]
        if not fits(plan):
            return None

    rounds = []
    for hosted in plan:
        teams = []
        for node, node_sectors in hosted.items():
            ranks = node_ranks[node]
            n_teams = min(len(node_sectors), len(ranks))
            for i, group in enumerate(np.array_split(np.array(ranks), n_teams)):
                teams.append(
                    LanczosTeam(node_host[node], node, tuple(int(r) for r in group), tuple(node_sectors[i::n_teams]))
                )
        rounds.append(tuple(teams))
    return tuple(rounds)


def _solve_sectors_in_teams(
    mpi_dist_irrk: MpiDistributor,
    comm: MPI.Comm,
    node_comm: MPI.Comm,
    gamma_sing_pp: FourPoint,
    gamma_trip_pp: FourPoint,
    giwk_dga: GreensFunction | None,
    niv_pp: int,
    rounds: tuple[tuple[LanczosTeam, ...], ...],
    u: np.ndarray | None = None,
) -> dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Solves every ``(channel, parity)`` sector with the teams of :func:`plan_lanczos_teams`. The pp bubble is built
    once on rank 0, shipped to the other hosting node roots and exposed through one shared window per node; per
    round, each channel's pairing vertex is gathered onto the root of every node hosting one of its sectors and
    written into a node-shared matmul-layout window there (see :func:`_build_shared_vertex_window`), so a node holds
    one copy per hosted channel instead of one per sector; every team then runs the lockstep matvec of
    :func:`_team_matvec` over that window while its lead drives the eigensolver. On a symmetry-reduced grid with
    unitary operations the windows hold the irreducible wedge of the real-space grid and the matvec contracts one
    star at a time (see :func:`_wedge_orbits`). Both pairing vertices are **consumed**. The eigenvalues of
    every sector reach every rank; the gap functions travel from the lead to rank 0 only (the rank that writes
    them), one blocking transfer per gap, their number following the eigenvalues actually returned.

    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param comm: The MPI communicator.
    :param node_comm: The node-local communicator.
    :param gamma_sing_pp: The singlet pairing vertex (irr-BZ q-distributed on entry; consumed).
    :param gamma_trip_pp: The triplet pairing vertex (irr-BZ q-distributed on entry; consumed).
    :param giwk_dga: The DGA Green's function (held on rank 0; used to build the bubble).
    :param niv_pp: Number of positive fermionic frequencies of the pp box.
    :param rounds: The team plan.
    :return: ``{(channel, parity): (lambdas, gaps)}`` for every sector; ``lambdas`` on every rank, ``gaps`` on rank 0
        (an empty list elsewhere).
    """
    logger = config.logger
    my_rank = comm.rank
    k_grid = config.lattice.k_grid
    norm = 0.5 / k_grid.nk_tot / config.sys.beta
    n_bands = gamma_sing_pp.n_bands
    gap_shape = k_grid.nk + (n_bands, n_bands, 2 * gamma_sing_pp.niv)
    my_node_root = node_comm.bcast(my_rank, root=0)
    hosting_roots = sorted({team.node_root for teams in rounds for team in teams})
    orbits = _wedge_orbits(k_grid) if wedge_window_points(k_grid) < k_grid.nk_tot else None
    wedge = orbits is not None

    # the pp bubble: built on rank 0, shipped to the other hosting node roots, one shared window per hosting node
    gchi0_q_pp = None
    if my_rank == 0:
        gchi0_q_pp = BubbleGenerator.create_generalized_chi0_q_pp_w0(giwk_dga, niv_pp, k_grid).decompress_q_dimension()
        giwk_dga.free()
        logger.info("Created the bare bubble susceptibility in pp notation.")
    for root in hosting_roots:
        if root == 0:
            continue
        if my_rank == 0:
            mpi_dist_irrk.send_to_rank(gchi0_q_pp, dest=root, base_tag=BUBBLE_TAG)
        elif my_rank == root:
            gchi0_q_pp = mpi_dist_irrk.recv_from_rank(source=0, base_tag=BUBBLE_TAG)
    chi0_mm, chi0_win = None, None
    if my_node_root in hosting_roots:
        chi0_shared, chi0_win = mpi_utils.build_node_shared_array(node_comm, lambda: gchi0_q_pp.mat)
        chi0_mm = _chi0_to_matmul_layout(chi0_shared)
        del chi0_shared
    if gchi0_q_pp is not None:
        gchi0_q_pp.free()

    results: dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]] = {}
    for teams in rounds:
        gamma_by_channel: dict[SpinChannel, np.ndarray] = {}
        vertex_wins = []
        for channel, gamma_pp in ((SpinChannel.SING, gamma_sing_pp), (SpinChannel.TRIP, gamma_trip_pp)):
            roots = sorted({team.node_root for team in teams if any(c == channel for c, _ in team.sectors)})
            if not roots:
                continue
            local = gamma_pp.mat
            for root in roots:
                gathered = mpi_dist_irrk.gather(local, root=root)
                if my_rank == root:
                    gamma_pp.mat = gathered
            # the gathers are complete; these names would otherwise keep the rank's share alive
            del gathered, local
            if my_rank not in roots:
                gamma_pp.free()
            if my_node_root in roots:
                gamma_by_channel[channel], win = _build_shared_vertex_window(
                    gamma_pp, node_comm, norm, None if orbits is None else orbits[0]
                )
                vertex_wins.append(win)
                logger.info(
                    f"Gamma_pp_{channel.value} window ({'irreducible wedge' if wedge else 'full grid'}): "
                    f"{gamma_by_channel[channel].nbytes / 1024**3:.3f} GB, shared by the {node_comm.Get_size()} "
                    f"rank(s) of its node.",
                    allowed_ranks=tuple(roots),
                )

        my_team = next((team for team in teams if my_rank in team.ranks), None)
        team_comm = comm.Split(teams.index(my_team) if my_team is not None else len(teams), my_rank)
        local: dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]] = {}
        if my_team is not None:
            dtype = next(iter(gamma_by_channel.values())).dtype
            scratch = [mpi_utils.allocate_node_shared_array(team_comm, gap_shape, dtype) for _ in range(4)]
            apply_block, (v0, v1) = _team_matvec(
                team_comm,
                gamma_by_channel,
                chi0_mm,
                [array for array, _ in scratch],
                gap_shape,
                n_bands,
                orbits,
                u,
            )
            # ponytail: one BLAS thread per team rank (production binds one rank per core); a per-rank executor
            # if few-rank jobs on wide nodes ever matter
            with threadpool_limits(limits=1):
                local = _solve_team_sectors(team_comm, my_team, apply_block, (v0, v1), scratch[0][0], gap_shape, u)
            # every view into a window dies before the window is freed (the matvec closure holds the scratch
            # arrays and the vertex dict)
            scratch_wins = [win for _, win in scratch]
            del apply_block, scratch
            for win in scratch_wins:
                nonlocal_sde._free_shared_window(win, team_comm)
        team_comm.Free()

        for team in teams:
            for sector in team.sectors:
                lead, mine, tag = team.ranks[0], local.get(sector), SECTOR_TAG * (1 + len(results))
                lambdas = mpi_dist_irrk.bcast(mine[0] if mine is not None else None, root=lead)
                gaps = mine[1] if mine is not None else []
                if my_rank == lead != 0:
                    for gap in gaps:
                        mpi_dist_irrk.send_to_rank(gap, dest=0, base_tag=tag)
                    gaps = []
                elif my_rank == 0 != lead:
                    gaps = [mpi_dist_irrk.recv_from_rank(source=lead, base_tag=tag) for _ in range(len(lambdas))]
                results[sector] = (lambdas, gaps)
        gamma_by_channel.clear()
        for win in vertex_wins:
            nonlocal_sde._free_shared_window(win, node_comm)
    del chi0_mm
    nonlocal_sde._free_shared_window(chi0_win, node_comm)
    return results


def _solve_team_sectors(
    team_comm: MPI.Comm,
    team: LanczosTeam,
    apply_block: Callable,
    bounds: tuple[int, int],
    gap_w: np.ndarray,
    gap_shape: tuple,
    u: np.ndarray | None = None,
) -> dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Solves the sectors of one team with the team-distributed Krylov-Schur iteration (:func:`_team_arnoldi`): every
    rank of the team runs the same iteration on its frequency block of the vectors, the sector seeds and projectors
    are those of the single-rank solve, and the converged eigenvectors are assembled through the team's gap window
    onto the lead, which orders, symmetrizes and wraps them (see :func:`_finish_sector`). The lead alone holds the
    results.

    :param team_comm: The team communicator (rank 0 is the lead).
    :param team: The team and its sectors.
    :param apply_block: The collective sector matvec of :func:`_team_matvec`.
    :param bounds: This rank's frequency block ``(v0, v1)``.
    :param gap_w: The team's gap-shaped scratch window used to assemble the eigenvectors.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param u: The orbital action of the inversion for the parity sectors (see :func:`_project_gap_to_sector`).
    :return: ``{(channel, parity): (lambdas, gaps)}`` on the lead, an empty dict elsewhere.
    """
    logger = config.logger
    lead = team_comm.Get_rank() == 0
    v0, v1 = bounds
    n_eig, tol = config.eliashberg.n_eig, config.eliashberg.epsilon
    ncv = min(lanczos_ncv(n_eig), int(np.prod(gap_shape)))
    orbital_mirrors = _gap_orbital_mirrors(gap_shape[3]) if config.eliashberg.symmetrize_degenerate_gaps else {}
    eps_by_parity = dict(_parity_sectors(config.eliashberg.resolve_parity_sectors))
    ranks = (team.ranks[0],)
    labels = [_sector_log_label(channel, [parity]) for channel, parity in team.sectors]
    logger.info(f"Lanczos team for {', '.join(labels)}: {len(team.ranks)} rank(s) on {team.host}.", allowed_ranks=ranks)
    results: dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]] = {}
    seeded_channel = None
    for (channel, parity), label in zip(team.sectors, labels):
        sign = 1 if channel == SpinChannel.SING else -1
        eps_t = eps_by_parity[parity]
        eps_p = None if eps_t is None else sign * eps_t
        logger.info(f"Starting Lanczos method for {label}.", allowed_ranks=ranks)
        if channel != seeded_channel:
            base_seed, seeded_channel = get_initial_gap_function(gap_shape, channel).flatten(), channel
        n_block = block_size(config.lattice.k_grid, eps_p if gap_shape[3] == 1 else None)
        seeds = _sector_seeds(base_seed, gap_shape, eps_t, eps_p, n_block, u)
        seed = np.ascontiguousarray(seeds.reshape((n_block,) + tuple(gap_shape))[..., v0:v1])
        del seeds
        attempts = []

        def attempt(t: float) -> tuple:
            attempts.append(
                _team_arnoldi(
                    lambda block: apply_block(channel, eps_t, eps_p, block),
                    seed,
                    n_eig,
                    ncv,
                    t,
                    LANCZOS_RESTARTS,
                    team_comm,
                    n_block,
                )
            )
            return attempts[-1]

        start = time.perf_counter()
        lambdas, ritz, basis, _, _ = _solve_loosening(attempt, tol, n_eig, label, ranks)
        # the log counts every attempt, as the single-rank solve does
        n_matvec, matvec_seconds = sum(a[3] for a in attempts), sum(a[4] for a in attempts)
        logger.info(
            f"Lanczos for {label}: {n_matvec} matvecs, {matvec_seconds:.1f} s in the matvec "
            f"({1e3 * matvec_seconds / max(n_matvec, 1):.0f} ms each), {time.perf_counter() - start:.1f} s "
            f"in the solve.",
            allowed_ranks=ranks,
        )
        if len(lambdas) < n_eig:
            logger.warning(
                f"Lanczos for {label} did not converge within the restart limit; keeping the {len(lambdas)} of "
                f"{n_eig} converged eigenpair(s).",
                allowed_ranks=ranks,
            )
        if eps_t is not None and len(lambdas):
            leading = ritz[:, int(np.argmax(lambdas))] @ basis
            raw = apply_block(channel, None, None, leading).reshape(-1)
            _log_sector_residual(leading, raw, label, ranks, team_comm.allreduce)
        # the eigenvectors travel block by block through the shared gap window to the lead
        gaps = np.empty((int(np.prod(gap_shape)), len(lambdas)), dtype=basis.dtype) if lead else None
        for i in range(len(lambdas)):
            block = ritz[:, i] @ basis
            block /= np.sqrt(team_comm.allreduce(float(np.vdot(block, block).real)))
            gap_w[..., v0:v1] = block.reshape(gap_shape[:-1] + (v1 - v0,))
            team_comm.Barrier()
            if lead:
                gaps[:, i] = gap_w.reshape(-1)
            team_comm.Barrier()
        if lead:
            results[(channel, parity)] = _finish_sector(
                lambdas, gaps, gap_shape, channel, tuple(gap_shape[:3]), label, orbital_mirrors, ranks
            )
    logger.info(f"Finished solving the Eliashberg equation for {', '.join(labels)}.", allowed_ranks=ranks)
    return results


def _solve_pairing_sectors(
    mv,
    gap_shape: tuple,
    sign: int,
    channel: SpinChannel,
    nq: tuple,
    executor,
    ranks,
    base_seed: np.ndarray,
    parities: list[str] | None = None,
    dtype: np.dtype = DTYPE,
    u: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Runs the ARPACK/Lanczos solve of the pairing kernel ``mv`` once per parity sector selected by
    ``config.eliashberg.resolve_parity_sectors`` and returns the leading ``n_eig`` eigenpairs of each. Every projected
    sector hands the matvec its Hermitian sector projector :math:`\Pi` (parity ``eps_T`` under :math:`T O_U` and the
    Pauli-forced inversion parity ``eps_P = sign * eps_T``, see :func:`_project_gap_to_sector`) and projects the seed
    with it, and the residual of its leading gap under the raw kernel is logged (see
    :func:`_log_sector_residual`); the ``"none"`` sector runs the raw kernel unchanged. The passed ``executor`` (the
    momentum-batch thread pool, or ``None``) is shut down before returning.

    :param mv: The sector matvec ``mv(gap, eps_t, eps_p)`` (maps a full-length gap vector to a full-length gap
        vector, projected onto the sector unless ``eps_t`` is ``None``).
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param sign: The channel sign (:math:`+1` singlet, :math:`-1` triplet).
    :param channel: The pairing channel (used to label outputs).
    :param nq: The momentum-grid shape carried onto each :class:`GapFunction`.
    :param executor: The momentum-batch thread pool (or ``None``); shut down on return.
    :param ranks: The ranks tuple used for logging.
    :param base_seed: The flattened initial gap seed, identical on every rank (drawn from a fixed-seed
        generator); projected into each sector by :func:`_sector_seed`.
    :param parities: An optional subset of parity labels to solve; ``None`` solves every configured sector.
    :param dtype: The dtype ``mv`` returns (that of the pairing vertex); declared on the eigensolver's operator so
        scipy does not probe it with an extra matvec.
    :param u: The orbital action of the inversion for the parity sectors (see :func:`_project_gap_to_sector`).
    :return: ``{parity_label: (lambdas, [GapFunction, ...])}`` for each solved sector.
    """
    logger = config.logger
    n_eig = config.eliashberg.n_eig
    shape_flat = int(np.prod(gap_shape))

    sectors = _parity_sectors(config.eliashberg.resolve_parity_sectors)
    if parities is not None:
        sectors = [(label, eps_t) for label, eps_t in sectors if label in parities]

    # the mirrors are a property of the orbital basis and the lattice, not of the sector: determine them once
    orbital_mirrors = _gap_orbital_mirrors(gap_shape[3]) if config.eliashberg.symmetrize_degenerate_gaps else {}

    results: dict[str, tuple[np.ndarray, list[GapFunction]]] = {}
    try:
        for parity, eps_t in sectors:
            eps_p = None if eps_t is None else sign * eps_t
            label = _sector_log_label(channel, [parity])
            logger.info(f"Starting Lanczos method for {label}.", allowed_ranks=ranks)
            n_matvec, matvec_seconds = 0, 0.0

            def counted_matvec(gap: np.ndarray, eps_t=eps_t, eps_p=eps_p) -> np.ndarray:
                nonlocal n_matvec, matvec_seconds
                start = time.perf_counter()
                result = mv(gap, eps_t, eps_p)
                n_matvec += 1
                matvec_seconds += time.perf_counter() - start
                return result

            mat = sp.sparse.linalg.LinearOperator(shape=(shape_flat, shape_flat), matvec=counted_matvec, dtype=dtype)
            n_block = block_size(config.lattice.k_grid, eps_p if gap_shape[3] == 1 else None)
            eigsh_start = time.perf_counter()
            # BLAS is pinned to one thread for the solve (threadpool_limits resizes the live pool; an environment
            # change would be ignored) so the momentum-batch threads never nest BLAS threads underneath.
            with threadpool_limits(limits=1 if executor is not None else None):
                if n_block > 1:
                    # forced multiplets need the block iteration, run on the full vectors of this rank
                    lambdas, ritz, basis, _, _ = _solve_loosening(
                        lambda t: _team_arnoldi(
                            counted_matvec,
                            _sector_seeds(base_seed, gap_shape, eps_t, eps_p, n_block, u),
                            n_eig,
                            min(lanczos_ncv(n_eig), shape_flat),
                            t,
                            LANCZOS_RESTARTS,
                            None,
                            n_block,
                        ),
                        config.eliashberg.epsilon,
                        n_eig,
                        label,
                        ranks,
                    )
                    gaps = (ritz.T @ basis).T
                    del basis
                    if len(lambdas) < n_eig:
                        logger.warning(
                            f"Lanczos for {label} did not converge within the restart limit; keeping the "
                            f"{len(lambdas)} of {n_eig} converged eigenpair(s).",
                            allowed_ranks=ranks,
                        )
                else:

                    def solve(tol: float) -> tuple[np.ndarray, np.ndarray]:
                        try:
                            return sp.sparse.linalg.eigsh(
                                mat,
                                k=n_eig,
                                tol=tol,
                                v0=_sector_seed(base_seed, gap_shape, eps_t, eps_p, u),
                                ncv=min(lanczos_ncv(n_eig), shape_flat),
                                which="LA",
                                maxiter=LANCZOS_RESTARTS,
                            )
                        except sp.sparse.linalg.ArpackNoConvergence as exc:
                            # the converged subset is kept; the delivery sizes the gap transfer off what came back
                            return exc.eigenvalues.real, exc.eigenvectors

                    lambdas, gaps = _solve_loosening(solve, config.eliashberg.epsilon, n_eig, label, ranks)
                    if len(lambdas) < n_eig:
                        logger.warning(
                            f"Lanczos for {label} did not converge within the iteration limit; keeping the "
                            f"{len(lambdas)} of {n_eig} converged eigenpair(s).",
                            allowed_ranks=ranks,
                        )
            logger.info(
                f"Lanczos for {label}: {n_matvec} matvecs, {matvec_seconds:.1f} s in the matvec "
                f"({1e3 * matvec_seconds / max(n_matvec, 1):.0f} ms each), {time.perf_counter() - eigsh_start:.1f} s "
                f"in eigsh.",
                allowed_ranks=ranks,
            )
            if eps_t is not None and len(lambdas):
                leading = gaps[:, int(np.argmax(lambdas))]
                # the check's matvec runs on the momentum-batch threads as well, so BLAS stays pinned
                with threadpool_limits(limits=1 if executor is not None else None):
                    _log_sector_residual(leading, mv(leading, None, None), label, ranks)
            results[parity] = _finish_sector(lambdas, gaps, gap_shape, channel, nq, label, orbital_mirrors, ranks)
    finally:
        if executor is not None:
            executor.shutdown()

    logger.info(f"Finished solving the Eliashberg equation for the {channel.value}let channel.", allowed_ranks=ranks[0])
    return results


def solve_eliashberg_lanczos(
    gamma_r_pp: FourPoint,
    gchi0_q0_pp: FourPoint,
    ranks: tuple[int, int],
    parities: list[str] | None = None,
    u: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Solves the linearized Eliashberg equation for the leading superconducting eigenvalue(s) and gap function(s) using
    an ARPACK/Lanczos eigensolver, with the pairing kernel applied matrix-free via FFTs over the BZ. This in-memory
    variant holds the full-BZ pairing vertex on the solving rank. The passed pairing vertex is **consumed** (mapped
    to the full BZ and Fourier transformed in place, then freed once its matmul-layout copy is built).

    When ``config.eliashberg.resolve_parity_sectors`` is set, the matvec and the starting vector are, for each
    parity sector, sandwiched in the sector projector :math:`\Pi` (see
    :func:`_project_gap_to_sector`), so the eigensolver returns the leading eigenpairs of :math:`\Pi M \Pi` restricted
    to that sector; otherwise the raw kernel :math:`M` is run unchanged.

    :param gamma_r_pp: The pairing vertex :math:`\Gamma^{\mathrm{pp}}_{r}` (irreducible BZ, pp notation) for one
        channel; consumed by the solve.
    :param gchi0_q0_pp: The bare pp bubble :math:`\chi_0^{\mathrm{pp}}` at :math:`\omega = 0`.
    :param ranks: The ranks used for logging.
    :param parities: An optional subset of parity labels to solve (``None`` solves every configured sector).
    :param u: The orbital action of the inversion for the parity sectors (see :func:`_project_gap_to_sector`).
    :return: A dict ``{parity_label: (lambdas, gaps)}`` of the leading eigenvalues and :class:`GapFunction` objects
        per solved parity sector (a single ``"none"`` key when no projection is requested).
    """
    logger = config.logger
    import psutil

    logger.info(
        f"Starting to solve the Eliashberg equation for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=ranks[0],
    )

    gamma_r_pp = gamma_r_pp.map_to_full_bz(config.lattice.k_grid, config.lattice.k_grid.nk).decompress_q_dimension()
    logger.log_memory_usage(f"Gamma_pp_{gamma_r_pp.channel.value}", gamma_r_pp, 1, allowed_ranks=ranks[0])

    gamma_r_pp = gamma_r_pp.fft(False)

    gap_shape = gamma_r_pp.nq + 2 * (gamma_r_pp.n_bands,) + (2 * gamma_r_pp.niv,)
    gchi0_q0_pp = gchi0_q0_pp.decompress_q_dimension()

    symmetry_label = config.eliashberg.symmetry.lower() if config.eliashberg.symmetry else "random"
    logger.info(
        f"Initialized the gap function as {symmetry_label} for {_sector_log_label(gamma_r_pp.channel, parities)}.",
        allowed_ranks=ranks,
    )

    n_bands = gamma_r_pp.n_bands
    norm = 0.5 / config.lattice.k_grid.nk_tot / config.sys.beta
    # the single rank spreads the bandwidth-bound matvec over every core its affinity mask allows (momentum-batch
    # threads, BLAS kept at 1)
    n_threads = _solver_thread_budget()
    executor = ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None
    logger.info(
        f"Solver thread budget for {_sector_log_label(gamma_r_pp.channel, parities)}: {n_threads} thread(s), from "
        f"the CPU affinity mask.",
        allowed_ranks=ranks,
    )

    chi0_mm = _chi0_to_matmul_layout(gchi0_q0_pp.mat)
    # The pairing vertex arrives in w2dynamics G2 leg order (c cdag c cdag), whereas _apply_gamma_pp expects the
    # TRIQS order (cdag c cdag c), see https://triqs.github.io/tprf/latest/theory/eliashberg.html
    # the kernel prefactor is folded into the persistent vertex once: both matvec terms inherit it by linearity, so
    # the per-matvec full-gap multiply is dropped.
    gamma_mm = _gamma_to_matmul_layout(gamma_r_pp.permute_orbitals("abcd->badc", False).mat, scale=norm)
    gamma_r_pp.free()
    resident_gb = psutil.Process().memory_info().rss / 1024**3
    logger.info(
        f"Solver rank resident set after the matmul-layout build: {resident_gb:.3f} GB "
        f"({_sector_log_label(gamma_r_pp.channel, parities)}).",
        allowed_ranks=ranks,
    )

    sign = 1 if gamma_r_pp.channel == SpinChannel.SING else -1

    def mv(gap: np.ndarray, eps_t: int | None, eps_p: int | None) -> np.ndarray:
        r"""
        Applies the pairing kernel of one sector to a flattened gap vector (the matrix-vector product for the
        eigensolver): projects onto the sector (unless ``eps_t`` is ``None``), multiplies by
        :math:`\chi_0^{\mathrm{pp}}`, FFTs to real space, contracts with the pairing vertex (the direct term, plus the
        crossed term formed from it by the exchange of the outgoing pair, see :func:`_crossed_term`), transforms back
        and projects again. The orbital contractions are batched ``np.matmul`` products and the BZ transforms run in
        place through ``scipy.fft`` (both threaded up to the solver thread budget).

        :param gap: The flattened gap vector.
        :param eps_t: The parity of the sector under :math:`T O_U`, or ``None`` for the raw kernel.
        :param eps_p: The forced inversion parity of the sector.
        :return: The flattened result of applying the sector kernel to ``gap``.
        """
        if eps_t is not None:
            gap = _project_gap_to_sector(gap, gap_shape, eps_t, eps_p, u)
        gap_gg = sp.fft.fftn(
            _apply_gchi0_pp(chi0_mm, gap, n_bands, executor, n_threads),
            axes=(0, 1, 2),
            overwrite_x=True,
            workers=n_threads,
        )
        gap_new = _apply_gamma_pp(gamma_mm, gap_gg, n_bands, executor, n_threads)
        gap_new += _crossed_term(gap_new, sign)
        gap_new = sp.fft.ifftn(gap_new, axes=(0, 1, 2), overwrite_x=True, workers=n_threads)
        out = gap_new.flatten()
        return out if eps_t is None else _project_gap_to_sector(out, gap_shape, eps_t, eps_p, u)

    base_seed = get_initial_gap_function(gap_shape, gamma_r_pp.channel).flatten()
    return _solve_pairing_sectors(
        mv,
        gap_shape,
        sign,
        gamma_r_pp.channel,
        gamma_r_pp.nq,
        executor,
        ranks,
        base_seed,
        parities,
        gamma_mm.dtype,
        u,
    )


# --- Eliashberg eigensolver (Lanczos / ARPACK) ---
def dispatch_full_vertex_calculation(
    channel: SpinChannel,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    niv_pp: int,
    mpi_dist: MpiDistributor,
    chunk_bytes: int | None = None,
    node_comm: MPI.Comm | None = None,
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
    :return: The full ladder pp vertex :math:`F^{\mathrm{q}}_{r}` as a :class:`FourPoint`.
    """
    if config.misc.bare_vertex:
        build = create_pairing_vertex_streaming_fq if config.eliashberg.save_fq else create_pairing_vertex_slice_q_r
        f_q_r = build(u_loc, v_nonloc, None, niv_pp, mpi_dist, chunk_bytes, channel=channel)
        mpi_dist.barrier()
        return f_q_r
    gamma_r, gamma_win = nonlocal_sde._load_node_shared_local_vertex(
        node_comm, os.path.join(config.output.output_path, f"gamma_{channel.value}_loc.npy"), channel
    )
    if config.eliashberg.save_fq:
        f_q_r = create_pairing_vertex_streaming_fq(u_loc, v_nonloc, gamma_r, niv_pp, mpi_dist, chunk_bytes)
    else:
        f_q_r = create_pairing_vertex_slice_q_r(u_loc, v_nonloc, gamma_r, niv_pp, mpi_dist, chunk_bytes)
    gamma_r.mat = None
    if gamma_win is None:
        gamma_r.free()
    nonlocal_sde._free_shared_window(gamma_win, node_comm)
    mpi_dist.barrier()
    return f_q_r


def _solve_sectors_in_memory(
    mpi_dist_irrk: MpiDistributor,
    gamma_sing_pp: FourPoint,
    gamma_trip_pp: FourPoint,
    giwk_dga: GreensFunction,
    niv_pp: int,
    parities: list[str],
    u: np.ndarray | None = None,
) -> dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Solves every ``(channel, parity)`` sector in turn on the single rank of the job (the in-memory path of a
    single-rank run): the pp bubble is built once from ``giwk_dga``, then the singlet and the triplet channel run
    through :func:`solve_eliashberg_lanczos` one after the other, each over the requested parity sectors. Both
    pairing vertices are **consumed**.

    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param gamma_sing_pp: The singlet pairing vertex (the whole irreducible BZ; consumed).
    :param gamma_trip_pp: The triplet pairing vertex (the whole irreducible BZ; consumed).
    :param giwk_dga: The DGA Green's function (used to build the bubble).
    :param niv_pp: Number of positive fermionic frequencies of the pp box.
    :param parities: The parity labels to solve.
    :param u: The orbital action of the inversion for the parity sectors (see :func:`_project_gap_to_sector`).
    :return: ``{(channel, parity): (lambdas, gaps)}`` for every sector.
    """
    gchi0_q_pp = BubbleGenerator.create_generalized_chi0_q_pp_w0(giwk_dga, niv_pp, config.lattice.k_grid)
    config.logger.info("Created the bare bubble susceptibility in pp notation.")
    results: dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]] = {}
    for gamma_pp in (gamma_sing_pp, gamma_trip_pp):
        sectors = solve_eliashberg_lanczos(gamma_pp, gchi0_q_pp, (0,), parities, u)
        results.update({(gamma_pp.channel, parity): sectors[parity] for parity in parities})
    mpi_dist_irrk.delete_file()
    return results


def solve(
    giwk_dga: GreensFunction | None,
    g_dmft: GreensFunction,
    u_loc: LocalInteraction,
    v_nonloc: Interaction | None,
    comm: MPI.Comm,
    chunk_budgets: memory_estimator.ChunkBudgets | None = None,
):
    r"""
    Drives the Eliashberg step: assembles the singlet and triplet pairing vertices from the saved
    ladder-DGA full vertices (optionally adding the local reducible diagrams; in the bare-vertex modes
    ``config.misc.do_flex`` and ``config.misc.do_rpa`` from the RPA vertices, counting the bare up-down interaction
    once instead), then solves the linearized gap equation for each channel and returns the leading eigenvalues and
    gap functions. A
    single rank solves in memory, sector after sector; a multi-rank job solves in memory as concurrent (channel x
    parity) sector teams that share one vertex window per channel and node when a node holds such a window (see
    :func:`plan_lanczos_teams`, decided from the available node memory), and on the block-distributed solver grid
    otherwise.

    :param giwk_dga: The converged momentum-dependent DGA :class:`GreensFunction` on rank 0 (the pp-bubble build
        there is its only reader); ``None`` on every other rank.
    :param g_dmft: The local (DMFT) :class:`GreensFunction` (used for the local diagrams).
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}` on the full q-grid; only read on rank 0.
    :param comm: The MPI communicator.
    :param chunk_budgets: Chunk byte budgets sized by the driver from the memory estimate, of which the pairing-vertex
        budget is used; ``None`` gives the fair-share budget of
        :func:`~dgamore.memory_estimator.dynamic_chunk_budget`.
    :return: ``(results, inversion)``: a dict keyed by ``(channel, parity_label)`` mapping to ``(lambdas, gaps)`` of
        the leading eigenvalues and :class:`GapFunction` objects for each solved parity sector, and the orbital action
        of the inversion the sectors were projected with (see :func:`check_parity_sectors`; ``None`` for none or
        without the sectors). The eigenvalues are present on every rank, the gap functions on rank 0 only (an empty
        list elsewhere). When ``config.eliashberg.resolve_parity_sectors`` is set the parity labels are ``"even"``
        and ``"odd"``, otherwise a single unprojected ``"none"`` sector is returned.
    """
    logger = config.logger
    import psutil

    mpi_dist_irrk = MpiDistributor.create_distributor(
        ntasks=config.lattice.k_grid.nk_irr, comm=comm, name="Q", output_path=config.output.output_path
    )
    v_nonloc = nonlocal_sde.scatter_irreducible_interaction(v_nonloc, mpi_dist_irrk)

    # the check may switch the sectors off, so it runs before they are fixed
    inversion = check_parity_sectors(comm)
    parities = [parity for parity, _ in _parity_sectors(config.eliashberg.resolve_parity_sectors)]
    niv_pp = min(config.box.niw_core // 2, config.box.niv_core // 2)

    # one sector's single-rank residency, from the memory_estimator formula the driver's fit check reads as well;
    # it is what a single rank holds, and the figure the solver log lines quote
    per_sector_bytes, _ = lanczos_solver_bytes(
        config.sys.n_bands,
        config.lattice.k_grid.nk_tot,
        config.lattice.k_grid.nk_irr,
        niv_pp,
        config.eliashberg.n_eig,
        comm.size,
    )
    # the node budget honors the scheduler's cgroup memory limit (e.g. slurm --mem), like the driver's fit check
    node_budget = psutil.virtual_memory().available
    cgroup_limit = mpi_utils.cgroup_memory_limit()
    if cgroup_limit is not None:
        node_budget = min(node_budget, cgroup_limit)

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) if comm.size > 1 else None
    # on a multi-rank job the team plan (an allgather, so identical on every rank) decides: it is None when no
    # node holds even one channel's windows, and the grid takes over; a single rank solves in memory
    rounds = None
    if comm.size > 1 and not FORCE_GRID_SOLVER:
        rounds = plan_lanczos_teams(comm, node_comm, node_budget, niv_pp)
    use_grid = FORCE_GRID_SOLVER or (comm.size > 1 and rounds is None)
    bubble_rank = 0
    if use_grid:
        _, grid_bytes = lanczos_solver_bytes(
            config.sys.n_bands,
            config.lattice.k_grid.nk_tot,
            config.lattice.k_grid.nk_irr,
            niv_pp,
            config.eliashberg.n_eig,
            comm.size,
        )
        rows, cols = solver_grid_shape(comm.size, 2 * niv_pp)
        logger.info(
            f"Eliashberg solver: no node holds one channel's vertex windows -> block-distributed {rows}x{cols} grid, "
            f"sectors sequential, {grid_bytes / 1024**3:.3f} GB per rank (one full-BZ sector residency is "
            f"{per_sector_bytes / 1024**3:.3f} GB; niv_pp = {niv_pp}, node budget {node_budget / 1024**3:.3f} GB)."
        )
    else:
        if rounds is None:
            logger.info(
                f"Eliashberg solver: {2 * len(parities)} (channel x parity) sector(s) in turn on the single rank, "
                f"holding at most {per_sector_bytes / 1024**3:.3f} GB (niv_pp = {niv_pp})."
            )
        else:
            hosts = sorted({team.host for teams in rounds for team in teams})
            sizes = sorted({len(team.ranks) for teams in rounds for team in teams})
            logger.info(
                f"Eliashberg solver: {2 * len(parities)} (channel x parity) sector(s) solved by "
                f"{sum(len(teams) for teams in rounds)} team(s) of {'-'.join(map(str, sizes))} rank(s) on "
                f"{len(hosts)} node(s), "
                f"{'all sectors at once' if len(rounds) == 1 else 'one channel at a time'}; one vertex window per "
                f"channel and hosting node (niv_pp = {niv_pp}, node budget {node_budget / 1024**3:.3f} GB)."
            )

    chunk_bytes = (
        memory_estimator.dynamic_chunk_budget(mpi_utils.job_memory_total(), node_comm.size if node_comm else 1)
        if chunk_budgets is None
        else chunk_budgets.fq
    )

    build_start = time.perf_counter()
    f_dens_pp = dispatch_full_vertex_calculation(
        SpinChannel.DENS, u_loc, v_nonloc, niv_pp, mpi_dist_irrk, chunk_bytes, node_comm
    )
    logger.info(f"Built the density full ladder vertex in pp notation in {time.perf_counter() - build_start:.1f} s.")
    build_start = time.perf_counter()
    f_magn_pp = dispatch_full_vertex_calculation(
        SpinChannel.MAGN, u_loc, v_nonloc, niv_pp, mpi_dist_irrk, chunk_bytes, node_comm
    )
    logger.info(f"Built the magnetic full ladder vertex in pp notation in {time.perf_counter() - build_start:.1f} s.")

    delete_files(config.output.eliashberg_path, f"gchi0_q_inv_rank_{comm.rank}.npy")

    mpi_dist_irrk.delete_file()

    gamma_sing_pp = f_dens_pp.scale(0.5).sub(f_magn_pp.scale(1.5, copy=True), copy=False)
    del f_dens_pp
    gamma_sing_pp.channel = SpinChannel.SING
    logger.info("Calculated full ladder-vertex (singlet) in pp notation.")

    gamma_trip_pp = gamma_sing_pp.add(f_magn_pp.scale(2.0))
    gamma_trip_pp.channel = SpinChannel.TRIP
    f_magn_pp.free()
    logger.info("Calculated full ladder-vertex (triplet) in pp notation.")

    if config.misc.bare_vertex:
        # Eqs. (4.49)-(4.52) with the bare up-down vertex U as the local full vertex and no pp-reducible diagrams:
        # its minus in crossed-slot form, -(-U_{1432}), keeps one copy of U, which both RPA ladders start with
        u_crossed = u_loc.permute_orbitals("abcd->adcb")
        gamma_sing_pp.add(u_crossed, copy=False)
        gamma_trip_pp.add(u_crossed, copy=False)
    else:
        # the local diagrams are reduced from local vertices on the full asymptotic fermionic box; one rank per node
        # reads and reduces them and broadcasts the pp-box-sized results, so that transient exists once per node
        f_ud_loc_pp_w0, gamma_ud_loc_pp_w0, phi_ud_loc_pp_w0 = _compute_once_per_node(
            node_comm, lambda: create_local_ud_diagrams_pp_w0(g_dmft, niv_pp)
        )

        if mpi_dist_irrk.my_rank == 0:
            f_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="f_ud_loc_pp_w0")
            phi_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="phi_ud_loc_pp_w0")
            gamma_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="gamma_ud_loc_pp_w0")
            logger.info("Saved local ud diagrams in pp notation to file.")

        del f_ud_loc_pp_w0, gamma_ud_loc_pp_w0

        # special treatment of local full vertex that is subtracted with a different frequency notation and is
        # different from the regular pp
        f_ud_loc_transf_w0 = _compute_once_per_node(node_comm, lambda: create_local_f_ud_transformed_w0(niv_pp))

        # Eqs. (4.49)-(4.52): the assembled vertex holds the negative crossed slot, so the local full vertex enters
        # with a relative minus, the pp-reducible diagrams phi with a plus, both in crossed-slot form ((v, -v'), 1432).
        phi_ud_loc_pp_w0 = phi_ud_loc_pp_w0.flip_frequency_axis(-1, copy=False).permute_orbitals(
            "abcd->adcb", copy=False
        )
        delta_loc = phi_ud_loc_pp_w0.sub(f_ud_loc_transf_w0)
        gamma_sing_pp.add(delta_loc, copy=False)
        gamma_trip_pp.add(delta_loc, copy=False)
        del phi_ud_loc_pp_w0, f_ud_loc_transf_w0, delta_loc

    if config.eliashberg.save_pairing_vertex:
        gamma_sing_pp.mat = mpi_dist_irrk.gather(gamma_sing_pp.mat)
        gamma_trip_pp.mat = mpi_dist_irrk.gather(gamma_trip_pp.mat)
        if comm.rank == 0:
            gamma_sing_pp.save(
                output_dir=config.output.eliashberg_path, name=f"gamma_irrq_{gamma_sing_pp.channel.value}_pp"
            )
            gamma_trip_pp.save(
                output_dir=config.output.eliashberg_path, name=f"gamma_irrq_{gamma_trip_pp.channel.value}_pp"
            )
        gamma_sing_pp.mat = mpi_dist_irrk.scatter(gamma_sing_pp.mat)
        gamma_trip_pp.mat = mpi_dist_irrk.scatter(gamma_trip_pp.mat)
        logger.info(f"Saved singlet and triplet pairing vertices in pp notation in the irreducible BZ to file.")

    if use_grid:
        gchi0_q_pp = None
        if comm.rank == bubble_rank:
            gchi0_q_pp = BubbleGenerator.create_generalized_chi0_q_pp_w0(giwk_dga, niv_pp, config.lattice.k_grid)
            logger.info("Created the bare bubble susceptibility in pp notation.", allowed_ranks=(bubble_rank,))
        results = {}
        for gamma_pp in (gamma_sing_pp, gamma_trip_pp):
            channel = gamma_pp.channel
            sectors = solve_eliashberg_lanczos_grid(gamma_pp, gchi0_q_pp, comm, bubble_rank, inversion)
            for parity in parities:
                local = sectors[parity] if sectors is not None else None
                lambdas = comm.bcast(local[0] if local is not None else None, root=0)
                # every grid rank holds the sector's gaps; only rank 0 writes them, the others drop theirs
                results[(channel, parity)] = (lambdas, local[1] if comm.rank == 0 else [])
            sectors = None
    elif rounds is None:
        results = _solve_sectors_in_memory(
            mpi_dist_irrk, gamma_sing_pp, gamma_trip_pp, giwk_dga, niv_pp, parities, inversion
        )
    else:
        results = _solve_sectors_in_teams(
            mpi_dist_irrk, comm, node_comm, gamma_sing_pp, gamma_trip_pp, giwk_dga, niv_pp, rounds, inversion
        )
    if node_comm is not None:
        node_comm.Free()

    return results, inversion


# Fraction of a node's available host memory the sector packing may occupy (mirrors DGAmore.NODE_MEMORY_FRACTION).
NODE_MEMORY_FRACTION: float = 0.95
# Base MPI tags of the point-to-point transfers of the sector solves: the pp bubble ship to the hosting node roots,
# and one block per delivered sector (its gaps at SECTOR_TAG * (1 + sector_index)), clear of the gathers' tags.
BUBBLE_TAG: int = 900
SECTOR_TAG: int = 1000

# The team Arnoldi step orthogonalizes a second time only when the first pass left less than this share of the
# vector's norm (ARPACK's DGKS test), so the basis is streamed twice per step instead of four times.
REORTHOGONALIZE_BELOW: float = 0.717

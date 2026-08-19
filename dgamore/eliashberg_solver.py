# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Linearized Eliashberg equation solver. Starting from the ladder-DGA full vertex (saved per channel by the non-local SDE
step), this module assembles the particle-particle pairing vertex in the singlet/triplet channels at :math:`\omega = 0`,
optionally adds the local reducible diagrams, and solves the linearized gap equation :math:`\lambda \Delta =
\pm\frac{1}{2\beta n_{\mathbf{q}}}\, \Gamma^{\mathrm{pp}}\, \chi_0^{\mathrm{pp}}\, \Delta` with a matrix-free
ARPACK/Lanczos eigensolver (in memory when one sector's full-BZ pairing vertex fits on a rank, and on a
block-distributed frequency grid otherwise). The leading
eigenvalue :math:`\lambda` signals the pairing instability and the eigenvector is the gap function
:math:`\Delta^{\mathrm{k}}_{12}`. Equation numbers refer to the author's master's thesis (Chapter 4).
"""

import os
import socket
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import mpi4py.MPI as MPI
import numpy as np
import scipy as sp
from threadpoolctl import ThreadpoolController, threadpool_limits

import dgamore.config as config
from dgamore import nonlocal_sde, mpi_utils
from dgamore.bubble_gen import BubbleGenerator
from dgamore.four_point import FourPoint
from dgamore.gap_function import GapFunction
from dgamore.greens_function import GreensFunction
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.local_four_point import LocalFourPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore import memory_estimator
from dgamore.memory_estimator import SLICE_CHUNK_BYTES, lanczos_solver_bytes, solver_grid_shape
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel, FrequencyNotation, DTYPE, deferred_collection
from dgamore.symmetry_reduction import find_coordinate_mirror_orbital_unitaries


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
    vrg_q_r_right: FourPoint,
    chi_phys_q_r: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    u_r: Interaction,
    w_start: int,
    w_stop: int,
) -> FourPoint:
    r"""
    Builds the full ladder vertex :math:`F^{\mathrm{q}}_{r}` for one momentum and the bosonic window
    ``[w_start, w_stop)``, i.e. the amputated auxiliary susceptibility plus the separable interaction part, on the
    restricted inputs.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel.
    :param gchi0_q_inv: The single-q inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}`.
    :param vrg_q_r_left: The single-q three-leg vertex :math:`\gamma^{\mathrm{q}\nu}_{r}`.
    :param vrg_q_r_right: The single-q "right-side" three-leg vertex :math:`\tilde\gamma^{\mathrm{q}\nu}_{r}`.
    :param chi_phys_q_r: The single-q physical susceptibility :math:`\chi^{\mathrm{phys};\mathrm{q}}_{r}`.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}` restricted to this momentum.
    :param u_r: The channel-projected total interaction :math:`\mathcal{U}^{\mathbf{q}}_{r}`.
    :param w_start: First bosonic index of the window.
    :param w_stop: One past the last bosonic index of the window.
    :return: The ladder vertex over that window as a :class:`FourPoint` (half niw range, two fermionic dimensions).
    """
    gchi0_w = gchi0_q_inv.take_wn_slice(w_start, w_stop)
    gamma_w = gamma_r.take_wn_slice(w_start, w_stop)

    # eager rebinding releases chi* right after the first matmul; the bubble term enters on the diagonal in place
    f_chunk = nonlocal_sde.create_auxiliary_chi_r_q(gamma_w, gchi0_w, u_loc, v_nonloc)
    f_chunk = gchi0_w @ f_chunk
    f_chunk = f_chunk @ gchi0_w
    f_chunk = f_chunk.scale(-config.sys.beta**2).add_on_vn_diagonal(gchi0_w, factor=config.sys.beta**2)

    vrg_left_w = vrg_q_r_left.take_wn_slice(w_start, w_stop)
    vrg_right_w = vrg_q_r_right.take_wn_slice(w_start, w_stop)
    chi_phys_w = chi_phys_q_r.take_wn_slice(w_start, w_stop)
    return f_chunk.add((vrg_left_w @ u_r - vrg_left_w @ (u_r @ chi_phys_w @ u_r)) * vrg_right_w, copy=False)


def _write_pp_band(out: np.ndarray, f_chunk: FourPoint, niv_pp: int, omega: np.ndarray, w_start: int) -> None:
    r"""
    Writes the :math:`\omega' = 0` pp band of one ladder-vertex window into the pp accumulator of its momenta.

    Each bosonic frequency contributes the anti-diagonal :math:`\omega = \nu - \nu'` (see :func:`_pp_w0_band`), and
    the negative bosonic half is obtained from the positive one through the complex-conjugation symmetry that
    :meth:`~dgamore.local_n_point.LocalNPoint.to_negative_niw_range` implements. The orbital permutation, the
    fermionic flip and the overall minus are the ones of :func:`_transform_vertex_frequencies_w0`.

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
    negative = cut.to_negative_niw_range().permute_orbitals("abcd->adcb", copy=False).flip_frequency_axis(-1, False)
    positive = cut.permute_orbitals("abcd->adcb", copy=False).flip_frequency_axis(-1, False)

    for index in range(positive.current_shape[-3]):
        w = w_start + index
        out[..., omega == w] = -positive.mat[..., index, omega == w]
        if w > 0:  # w = 0 is shared by both halves and belongs to the positive one
            out[..., omega == -w] = -negative.mat[..., index, omega == -w]


def _build_pairing_vertex_pp(
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    gamma_r: LocalFourPoint,
    niv_pp: int,
    mpi_dist_irrk: MpiDistributor,
    chunk_writer=None,
    chunk_bytes: int | None = None,
) -> FourPoint:
    r"""
    Shared loop of the slice-direct pairing-vertex construction: walks the rank-local momenta and, per momentum, the
    bosonic axis in byte-bounded chunks, building each ladder-vertex chunk once and reading its :math:`\omega' = 0`
    pp band (see :func:`_pp_w0_band`). When ``chunk_writer`` is given, every ph-notation chunk is handed to it before
    being released, so a caller can stream the full vertex to disk in the same pass.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param chunk_writer: Optional callable ``(q_start, w_start, chunk_mat)`` receiving each ph-notation chunk (the
        momenta of the group as the leading axis).
    :param chunk_bytes: Chunk byte budget of the build (``None`` uses the floor).
    :return: The pairing vertex :math:`F^{\mathrm{q}}_{r}` in pp notation as a :class:`FourPoint`.
    """
    logger = config.logger
    channel = gamma_r.channel
    path, rank = config.output.eliashberg_path, mpi_dist_irrk.my_rank

    gchi0_q_inv = FourPoint.load(os.path.join(path, f"gchi0_q_inv_rank_{rank}.npy"), num_vn_dimensions=1)
    vrg_q_r_left = FourPoint.load(
        os.path.join(path, f"vrg_q_{channel.value}_rank_{rank}.npy"), channel=channel, num_vn_dimensions=1
    )
    vrg_q_r_right = FourPoint.load(
        os.path.join(path, f"vrg_q_{channel.value}_right_rank_{rank}.npy"), channel=channel, num_vn_dimensions=1
    )
    chi_phys_q_r = FourPoint.load(
        os.path.join(path, f"chi_phys_q_{channel.value}_rank_{rank}.npy"), channel=channel, num_vn_dimensions=0
    )
    logger.info(f"Loaded the intermediates for the {channel.value} pairing vertex.")

    my_irr_q_list = config.lattice.k_grid.get_irrq_list()[mpi_dist_irrk.my_slice]
    niw_stored = gchi0_q_inv.current_shape[-2] - 1
    _, omega = _pp_w0_band(niv_pp, niw_stored)
    # the pp band only reads |w| <= 2 niv_pp - 1, but a streamed full vertex must cover the whole stored box
    niw_build = niw_stored if chunk_writer is not None else int(np.max(omega))

    n_bands = config.sys.n_bands
    f_pp_mat = np.zeros((len(my_irr_q_list),) + (n_bands,) * 4 + (2 * niv_pp,) * 2, dtype=gamma_r.mat.dtype)

    # one byte budget bounds the transient: as many whole-box momenta as fit (small problems degenerate to the
    # single batched build), and where even one momentum's box exceeds it, that momentum's bosonic axis is chunked
    budget = SLICE_CHUNK_BYTES if chunk_bytes is None else chunk_bytes
    one_wn_bytes = n_bands**4 * (2 * config.box.niv_core) ** 2 * np.dtype(DTYPE).itemsize
    w_chunk = _wn_chunk_size(one_wn_bytes, budget)
    q_group = max(1, int(budget // max(one_wn_bytes * (niw_build + 1), 1)))

    with deferred_collection():
        for q_start in range(0, len(my_irr_q_list), q_group):
            q_stop = min(q_start + q_group, len(my_irr_q_list))
            gchi0_grp = gchi0_q_inv.take_q_index_slice(q_start, q_stop)
            vrg_left_grp = vrg_q_r_left.take_q_index_slice(q_start, q_stop)
            vrg_right_grp = vrg_q_r_right.take_q_index_slice(q_start, q_stop)
            chi_phys_grp = chi_phys_q_r.take_q_index_slice(q_start, q_stop)
            v_nonloc_grp = v_nonloc.take_q_index_slice(q_start, q_stop)
            u_r = u_loc.as_channel(channel) + v_nonloc_grp.as_channel(channel)

            for w_start in range(0, niw_build + 1, w_chunk):
                f_chunk = _build_ladder_vertex_chunk(
                    gamma_r,
                    gchi0_grp,
                    vrg_left_grp,
                    vrg_right_grp,
                    chi_phys_grp,
                    u_loc,
                    v_nonloc_grp,
                    u_r,
                    w_start,
                    min(w_start + w_chunk, niw_build + 1),
                )
                if chunk_writer is not None:
                    chunk_writer(q_start, w_start, f_chunk.mat)
                _write_pp_band(f_pp_mat[q_start:q_stop], f_chunk, niv_pp, omega, w_start)
                f_chunk.free()

            gchi0_grp.free()
            vrg_left_grp.free()
            vrg_right_grp.free()
            chi_phys_grp.free()

    gchi0_q_inv.free()
    vrg_q_r_left.free()
    vrg_q_r_right.free()
    chi_phys_q_r.free()

    delete_files(
        path,
        f"vrg_q_{channel.value}_rank_{rank}.npy",
        f"vrg_q_{channel.value}_right_rank_{rank}.npy",
        f"chi_phys_q_{channel.value}_rank_{rank}.npy",
    )

    return FourPoint(f_pp_mat, channel, config.lattice.k_grid.nk, 0, 2, False, True, True, FrequencyNotation.PP)


def create_pairing_vertex_slice_q_r(
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    gamma_r: LocalFourPoint,
    niv_pp: int,
    mpi_dist_irrk: MpiDistributor,
    chunk_bytes: int | None = None,
) -> FourPoint:
    r"""
    Builds the pp pairing vertex at :math:`\omega' = 0` directly, without ever materializing the full
    three-frequency ladder vertex :math:`F^{\mathrm{q}\nu\nu'}_{r}` (see :func:`_build_pairing_vertex_pp`). The
    result is identical to the full-inversion construction up to floating-point accuracy, at a transient of one
    byte-bounded bosonic chunk instead of the whole rank-local box.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param chunk_bytes: Chunk byte budget of the build (``None`` uses the floor).
    :return: The pairing vertex :math:`F^{\mathrm{q}}_{r}` in pp notation as a :class:`FourPoint`.
    """
    return _build_pairing_vertex_pp(u_loc, v_nonloc, gamma_r, niv_pp, mpi_dist_irrk, chunk_bytes=chunk_bytes)


def create_pairing_vertex_streaming_fq(
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    gamma_r: LocalFourPoint,
    niv_pp: int,
    mpi_dist_irrk: MpiDistributor,
    chunk_bytes: int | None = None,
) -> FourPoint:
    r"""
    Builds the pp pairing vertex like :func:`create_pairing_vertex_slice_q_r` while streaming the full ladder vertex
    in ph notation to ``f_irrq_<channel>.npy`` chunk by chunk, replacing the rank-0 gather of the whole
    irreducible-BZ vertex. Rank 0 creates the memory-mapped file with the layout of the gathered save (half bosonic
    range, global irreducible q-ordering); every rank then writes only its own disjoint ``(q, omega)`` slabs.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param chunk_bytes: Chunk byte budget of the build (``None`` uses the floor).
    :return: The pairing vertex :math:`F^{\mathrm{q}}_{r}` in pp notation as a :class:`FourPoint`.
    """
    channel = gamma_r.channel
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

    f_pp = _build_pairing_vertex_pp(u_loc, v_nonloc, gamma_r, niv_pp, mpi_dist_irrk, chunk_writer, chunk_bytes)
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


# --- Physical-gap symmetry sectors (frequency parity + forced momentum/orbital parity) ---
def _frequency_parity_sectors(resolve_frequency_parity: bool) -> list[tuple[str, int | None]]:
    r"""
    Returns the list of gap sectors to solve, each a ``(label, eps_T)`` pair where ``eps_T`` is the requested
    T-parity (:math:`+1` even, :math:`-1` odd) or ``None`` for the unprojected case. When ``resolve_frequency_parity``
    is set, the frequency-even and frequency-odd sectors are both returned; otherwise the single unprojected sector
    is returned. The paired momentum/orbital parity is fixed by the Pauli constraint ``eps_{P.O} = sign * eps_T``
    inside the solver, so it is not carried here.

    :param resolve_frequency_parity: Whether to split the gap into the frequency-even and frequency-odd sectors.
    :return: The ``[(label, eps_T), ...]`` sector list.
    """
    return [("even", 1), ("odd", -1)] if resolve_frequency_parity else [("none", None)]


def _sector_log_label(channel: SpinChannel, parities: list[str] | None = None) -> str:
    """
    Returns the name of the sectors a solver call covers, for log messages. Sector-aware naming keeps concurrently
    solving ranks distinguishable in the log, where a channel-only name would repeat verbatim once per sector.

    :param channel: The spin channel being solved.
    :param parities: The subset of parity labels this call handles, or ``None`` for all of them.
    :return: ``"the singlet channel"`` when the parity projection is off, otherwise the covered sectors, e.g.
        ``"the singlet/even & odd sectors"`` or ``"the triplet/odd sector"``.
    """
    labels = [label for label, _ in _frequency_parity_sectors(config.eliashberg.resolve_frequency_parity)]
    if parities is not None:
        labels = [label for label in labels if label in parities]
    if labels == ["none"]:
        return f"the {channel.value}let channel"
    return f"the {channel.value}let/{' & '.join(labels)} sector{'' if len(labels) == 1 else 's'}"


def _project_gap_to_sector(vec: np.ndarray, gap_shape: tuple, eps_t: int, eps_po: int) -> np.ndarray:
    r"""
    Projects a flattened gap onto a physical symmetry sector by applying the two commuting Hermitian projectors
    :math:`\tfrac{1}{2}(1 + \varepsilon_T T)` and :math:`\tfrac{1}{2}(1 + \varepsilon_{PO}\, P O)` in turn, where the
    three involutions act on the orbital gap :math:`\Delta^{\nu}_{12}(\mathbf{k})` as
    :math:`(T\Delta)^{\nu}_{12}(\mathbf{k}) = \Delta^{-\nu}_{12}(\mathbf{k})` (fermionic-frequency flip),
    :math:`(P\Delta)^{\nu}_{12}(\mathbf{k}) = \Delta^{\nu}_{12}(-\mathbf{k})` (momentum flip) and
    :math:`(O\Delta)^{\nu}_{12}(\mathbf{k}) = \Delta^{\nu}_{21}(\mathbf{k})` (orbital transpose), realized by the same
    array operations the pairing-kernel matvec uses. The Pauli antisymmetry :math:`\hat{S}\,P\,O\,T\,\Delta = -\Delta`
    with the spin exchange :math:`\hat{S}` a scalar in the singlet/triplet basis fixes :math:`P\,O\,T\,\Delta =
    \mathrm{sign}\,\Delta` (``sign`` the channel sign), so once the frequency parity :math:`\varepsilon_T` is chosen the
    combined momentum-orbital parity is forced to :math:`\varepsilon_{PO} = \mathrm{sign}\cdot\varepsilon_T`; only the
    product :math:`P\,O` is fixed, never :math:`P` and :math:`O` separately.

    :param vec: The flattened gap vector.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param eps_t: The requested T-parity, :math:`+1` (even) or :math:`-1` (odd).
    :param eps_po: The forced combined ``P.O`` parity, :math:`\mathrm{sign}\cdot\varepsilon_T`.
    :return: The projected flattened gap vector (dtype preserved).
    """
    g = vec.reshape(gap_shape)
    g = 0.5 * (g + eps_t * np.flip(g, axis=-1))
    g = 0.5 * (g + eps_po * np.roll(np.flip(g.swapaxes(3, 4), axis=(0, 1, 2)), shift=1, axis=(0, 1, 2)))
    return g.reshape(-1).astype(vec.dtype, copy=False)


def gap_parity_diagnostics(gap: np.ndarray, gap_shape: tuple) -> dict[str, complex]:
    r"""
    Reports the parity Rayleigh quotients :math:`\langle \Delta, X \Delta \rangle / \langle \Delta, \Delta \rangle`
    of a flattened gap for the involutions ``T`` (frequency flip), ``P`` (momentum flip), ``O`` (orbital transpose)
    and their product ``P.O``. A pure-parity gap returns :math:`\pm 1` for the involutions it is an eigenvector of;
    the values certify the parity of a returned gap and expose any leakage.

    :param gap: The flattened gap vector.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :return: A dict mapping ``"T"``, ``"P"``, ``"O"``, ``"PO"`` to the corresponding Rayleigh quotient.
    """
    g = gap.reshape(gap_shape)
    ops = {
        "T": np.flip(g, axis=-1),
        "P": np.roll(np.flip(g, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2)),
        "O": g.swapaxes(3, 4),
        "PO": np.roll(np.flip(g.swapaxes(3, 4), axis=(0, 1, 2)), shift=1, axis=(0, 1, 2)),
    }
    denom = np.vdot(g, g)
    if denom == 0:
        return {name: 0j for name in ops}
    return {name: complex(np.vdot(g, op) / denom) for name, op in ops.items()}


def _classify_momentum_block(block: np.ndarray, purity: float) -> str:
    r"""
    Classifies the wave symmetry of one orbital block's momentum profile, returning ``s``, ``d``, ``p`` or ``x``.

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
        inversion, antisymmetric under some exchange), ``p`` (odd under the full inversion) or ``x``.
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
    if even_fraction(block, -invert(block, (0, 1, 2))) >= purity:  # odd under the full inversion
        return "p"
    return "x"


def classify_gap_symmetry(gap: np.ndarray, purity: float = 0.9, weight_floor: float = 0.01) -> str:
    r"""
    Classifies the momentum wave symmetry and Matsubara-frequency parity of a gap and returns a compact label of the
    form ``<wave><parity>``. The wave letter is ``s``, ``d`` or ``p`` (``x`` if none of these match), and the parity
    sign is ``+`` (even in :math:`\nu`), ``-`` (odd) or empty (neither). The frequency parity is the sign of the
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
    gamma_r_pp: FourPoint, gchi0_q0_pp: FourPoint | None, comm: MPI.Comm, bubble_rank: int
) -> dict[str, tuple[np.ndarray, list[GapFunction]]] | None:
    r"""
    Solves the linearized Eliashberg equation with the pairing vertex distributed over a 2-D ``(nu, nu')`` block grid,
    for problems whose full-BZ vertex does not fit on a single rank. Semantics are identical to
    :func:`solve_eliashberg_lanczos`: the same matvec, projectors, seeding and post-processing, only the storage and
    the contraction are distributed.

    Every grid rank holds one full-BZ, Fourier-transformed, matmul-layout block of the vertex and runs the
    eigensolver in lockstep on the full gap vector (the collectives inside the matvec keep every rank's iterates
    identical). Per matvec each rank dresses its own and its mirror
    :math:`\nu'`-column block locally (the crossed term reads the mirror block through the :math:`\nu'` flip), so
    the only communication is one block-sized ``Allreduce`` over each row group (completing the :math:`\nu'` sum)
    and one gap-sized ``Allgatherv`` over each column group (reassembling the :math:`\nu` rows). Sectors and
    channels run sequentially on the whole grid; ranks beyond ``rows * cols`` idle and return ``None``.

    :param gamma_r_pp: This rank's irreducible-BZ share of the pairing vertex; consumed by the solve.
    :param gchi0_q0_pp: The bare pp bubble :math:`\chi_0^{\mathrm{pp}}` (only read on ``bubble_rank``).
    :param comm: The MPI communicator.
    :param bubble_rank: The rank holding the pp bubble.
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
    mirror_j = cols - 1 - j
    col_slice = slice(j * width, (j + 1) * width)
    mirror_slice = slice(mirror_j * width, (mirror_j + 1) * width)

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
    gamma_mm = _gamma_to_matmul_layout(block.mat)
    block.free()
    # fold the kernel prefactor into the persistent vertex once, exactly as the in-memory solve does
    gamma_mm *= 0.5 / k_grid.nk_tot / config.sys.beta

    chi0_full = FourPoint(chi0_mat, SpinChannel.NONE, k_grid.nk, 0, 1, False, True, True, FrequencyNotation.PP)
    chi0_full = chi0_full.decompress_q_dimension()
    chi0_own = _chi0_to_matmul_layout(np.ascontiguousarray(chi0_full.mat[..., col_slice]))
    chi0_mir = (
        chi0_own if mirror_j == j else _chi0_to_matmul_layout(np.ascontiguousarray(chi0_full.mat[..., mirror_slice]))
    )
    chi0_full.free()

    gap_shape = k_grid.nk + 2 * (n_bands,) + (n_freq,)
    sign = 1 if channel == SpinChannel.SING else -1
    row_counts = (row_bounds[1:] - row_bounds[:-1]) * int(np.prod(gap_shape[:5]))
    row_displs = np.concatenate(([0], np.cumsum(row_counts[:-1])))

    def mv(gap: np.ndarray):
        r"""
        Applies the pairing kernel to a full flattened gap vector on the solver grid: this rank dresses its own and
        its mirror :math:`\nu'`-column block of the gap, contracts them with its vertex block (direct and crossed
        term), completes the :math:`\nu'` sum by an ``Allreduce`` over the row group, and reassembles the full
        :math:`\nu` axis by an ``Allgatherv`` over the column group, so every rank returns the identical full result.

        :param gap: The flattened gap vector (full length, identical on every grid rank).
        :return: The flattened result of applying the pairing kernel to ``gap``.
        """
        gap6 = gap.reshape(gap_shape)
        gap_gg = sp.fft.fftn(
            _apply_gchi0_pp(chi0_own, np.ascontiguousarray(gap6[..., col_slice]), n_bands),
            axes=(0, 1, 2),
            overwrite_x=True,
        )
        gg_mirror = (
            gap_gg
            if mirror_j == j
            else sp.fft.fftn(
                _apply_gchi0_pp(chi0_mir, np.ascontiguousarray(gap6[..., mirror_slice]), n_bands),
                axes=(0, 1, 2),
                overwrite_x=True,
            )
        )
        part = _apply_gamma_pp(gamma_mm, gap_gg, n_bands)
        # crossed term: flip_p of the FULL dressed gap restricted to this rank's columns equals the mirror block
        # flipped, so the mirror dressing above replaces any exchange (see solve_eliashberg_lanczos for the identity)
        crossed = _apply_gamma_pp(gamma_mm, np.ascontiguousarray(np.flip(gg_mirror, axis=-1)), n_bands)
        crossed = np.roll(np.flip(crossed.swapaxes(3, 4), axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
        if sign != 1:
            crossed *= sign
        part += crossed
        if cols > 1:
            part = np.ascontiguousarray(part)
            reduce_comm.Allreduce(MPI.IN_PLACE, part)
        part = sp.fft.ifftn(part, axes=(0, 1, 2), overwrite_x=True)
        if rows == 1:
            return part.flatten()
        send = np.ascontiguousarray(np.moveaxis(part, -1, 0))
        recv = np.empty((n_freq,) + send.shape[1:], dtype=send.dtype)
        gather_comm.Allgatherv(send, [recv, (row_counts, row_displs)])
        return np.moveaxis(recv, 0, -1).flatten()

    # get_initial_gap_function draws from a fixed-seed generator, so every lockstep rank computes the same seed
    seed = get_initial_gap_function(gap_shape, channel)
    return _solve_pairing_sectors(mv, gap_shape, sign, channel, k_grid.nk, None, (0,), seed.flatten())


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
    :return: A dict ``{axis: U}`` (empty when there is nothing to apply or the mirrors cannot be determined).
    """
    if n_bands < 2:
        return {}

    try:
        ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
        return find_coordinate_mirror_orbital_unitaries(np.asarray(ek, dtype=np.complex128))
    except Exception as exc:  # never let a symmetry probe abort a solve; momentum-only mirrors still work
        config.logger.debug(f"Could not determine the orbital mirror matrices for the gap symmetrization: {exc}")
        return {}


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
    eigenvalues and rotates every cluster to a mirror-adapted basis. The pairing kernel is only symmetrizable, not
    Hermitian in the plain inner product, so ARPACK may return oblique (mutually non-orthogonal) combinations
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


def _gamma_to_matmul_layout(gamma_mat: np.ndarray) -> np.ndarray:
    r"""
    Materializes the pp pairing vertex in batched-matmul layout ``[x, y, z, o2*nv, o2*np]`` (rows ``(a, b, v)``,
    columns ``(c, d, p)``) for :func:`_apply_gamma_pp`. The einsum layout is ``[x, y, z, a, c, b, d, v, p]`` with the
    orbitals interleaved (``a, c, b, d``), so a transpose to ``(a, b, v, c, d, p)`` precedes the (copying) reshape.
    The per-matvec ``np.matmul`` then allocates only the gap-sized output.

    :param gamma_mat: The pp vertex, shape ``[x, y, z, a, c, b, d, v, p]`` (``v`` may be a frequency slice).
    :return: A contiguous array ``[x, y, z, o2*nv, o2*np]`` in matmul layout.
    """
    nqx, nqy, nqz, nb = gamma_mat.shape[:4]
    nv, npp = gamma_mat.shape[-2], gamma_mat.shape[-1]
    transposed = np.ascontiguousarray(np.transpose(gamma_mat, (0, 1, 2, 3, 5, 7, 4, 6, 8)))  # [x,y,z,a,b,v,c,d,p]
    return transposed.reshape(nqx, nqy, nqz, nb * nb * nv, nb * nb * npp)


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
) -> dict[str, tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Runs the ARPACK/Lanczos solve of the pairing kernel ``mv`` once per physical frequency-parity sector selected by
    ``config.eliashberg.resolve_frequency_parity`` and returns the leading ``n_eig`` eigenpairs of each. Every projected
    sector wraps the matvec and the seed in the Hermitian sector projector :math:`\Pi` (T-parity ``eps_T`` and the
    Pauli-forced combined parity ``eps_PO = sign * eps_T``); the ``"none"`` sector runs the raw kernel unchanged. The
    passed ``executor`` (the momentum-batch thread pool, or ``None``) is shut down before returning.

    :param mv: The flattened pairing-kernel matvec (maps a full-length gap vector to a full-length gap vector).
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param sign: The channel sign (:math:`+1` singlet, :math:`-1` triplet).
    :param channel: The pairing channel (used to label outputs).
    :param nq: The momentum-grid shape carried onto each :class:`GapFunction`.
    :param executor: The momentum-batch thread pool (or ``None``); shut down on return.
    :param ranks: The ranks tuple used for logging.
    :param base_seed: The flattened initial gap seed, identical on every rank (drawn from a fixed-seed
        generator); projected into each sector, with a deterministic random fallback when the
        projection of the seed collapses (a seed whose parity is orthogonal to the requested sector).
    :param parities: An optional subset of parity labels to solve; ``None`` solves every configured sector. Used to
        hand different sectors to different ranks.
    :return: ``{parity_label: (lambdas, [GapFunction, ...])}`` for each solved sector.
    """
    logger = config.logger
    n_eig = config.eliashberg.n_eig
    plural = "" if n_eig == 1 else "s"
    shape_flat = int(np.prod(gap_shape))

    def sector_matvec(eps_t: int | None, eps_po: int | None):
        if eps_t is None:
            return mv
        return lambda gap: _project_gap_to_sector(
            mv(_project_gap_to_sector(gap, gap_shape, eps_t, eps_po)), gap_shape, eps_t, eps_po
        )

    def sector_seed(eps_t: int | None, eps_po: int | None) -> np.ndarray:
        if eps_t is None:
            return base_seed
        seed = _project_gap_to_sector(base_seed, gap_shape, eps_t, eps_po)
        if np.linalg.norm(seed) >= 1e-10 * max(np.linalg.norm(base_seed), 1e-30):
            return seed
        # the seed's parity is orthogonal to this sector; reseed deterministically so every rank agrees
        rng = np.random.default_rng(0)
        fallback = (rng.standard_normal(gap_shape) + 1j * rng.standard_normal(gap_shape)).flatten()
        fallback = _project_gap_to_sector(fallback, gap_shape, eps_t, eps_po)
        # a sector empty on this grid (e.g. every k equals -k) leaves nothing to project onto: fall back to the
        # nonzero base seed so the eigensolver gets a valid deterministic start (the projected operator returns ~0)
        return fallback if np.linalg.norm(fallback) > 0 else base_seed

    sectors = _frequency_parity_sectors(config.eliashberg.resolve_frequency_parity)
    if parities is not None:
        sectors = [(label, eps_t) for label, eps_t in sectors if label in parities]

    # the mirrors are a property of the orbital basis and the lattice, not of the sector: determine them once
    orbital_mirrors = _gap_orbital_mirrors(gap_shape[3]) if config.eliashberg.symmetrize_degenerate_gaps else {}

    results: dict[str, tuple[np.ndarray, list[GapFunction]]] = {}
    try:
        for parity, eps_t in sectors:
            eps_po = None if eps_t is None else sign * eps_t
            label = _sector_log_label(channel, [parity])
            logger.info(f"Starting Lanczos method for {label}.", allowed_ranks=ranks)
            mat = sp.sparse.linalg.LinearOperator(shape=(shape_flat, shape_flat), matvec=sector_matvec(eps_t, eps_po))
            # BLAS is pinned to one thread for the solve (threadpool_limits resizes the live pool; an environment
            # change would be ignored) so the momentum-batch threads never nest BLAS threads underneath.
            with threadpool_limits(limits=1 if executor is not None else None):
                lambdas, gaps = sp.sparse.linalg.eigsh(
                    mat,
                    k=n_eig,
                    tol=config.eliashberg.epsilon,
                    v0=sector_seed(eps_t, eps_po),
                    which="LA",
                    maxiter=10000,
                )
            order = lambdas.argsort()[::-1]  # sort eigenvalues in descending order
            lambdas = lambdas[order]
            gaps = gaps[:, order]
            if config.eliashberg.symmetrize_degenerate_gaps:
                gaps = symmetrize_degenerate_gaps(lambdas, gaps, gap_shape, orbital_mirrors=orbital_mirrors)
            logger.info(
                f"Largest eigenvalue{plural} for {label}: " + ", ".join(f"{lam:.6f}" for lam in lambdas),
                allowed_ranks=ranks,
            )
            # ARPACK may return fewer eigenpairs than requested, so size the list off what actually came back
            gap_list = [GapFunction(gaps[:, i].reshape(gap_shape), channel, nq) for i in range(gaps.shape[1])]
            results[parity] = (lambdas, gap_list)
    finally:
        if executor is not None:
            executor.shutdown()

    logger.info(f"Finished solving the Eliashberg equation for the {channel.value}let channel.", allowed_ranks=ranks[0])
    return results


def solve_eliashberg_lanczos(
    gamma_r_pp: FourPoint, gchi0_q0_pp: FourPoint, ranks: tuple[int, int], parities: list[str] | None = None
) -> dict[str, tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Solves the linearized Eliashberg equation for the leading superconducting eigenvalue(s) and gap function(s) using
    an ARPACK/Lanczos eigensolver, with the pairing kernel applied matrix-free via FFTs over the BZ. This in-memory
    variant holds the full-BZ pairing vertex on the solving rank. The passed pairing vertex is **consumed** (mapped
    to the full BZ and Fourier transformed in place, then freed once its matmul-layout copy is built).

    When ``config.eliashberg.resolve_frequency_parity`` is set, the matvec and the starting vector are, for each
    physical frequency-parity sector, sandwiched in the sector projector :math:`\Pi` (see
    :func:`_project_gap_to_sector`), so the eigensolver returns the leading eigenpairs of :math:`\Pi M \Pi` restricted
    to that sector; otherwise the raw kernel :math:`M` is run unchanged.

    :param gamma_r_pp: The pairing vertex :math:`\Gamma^{\mathrm{pp}}_{r}` (irreducible BZ, pp notation) for one
        channel; consumed by the solve.
    :param gchi0_q0_pp: The bare pp bubble :math:`\chi_0^{\mathrm{pp}}` at :math:`\omega = 0`.
    :param ranks: The ranks used for logging.
    :param parities: An optional subset of parity labels to solve on this rank (``None`` solves every configured
        sector); the caller assigns different parities to different ranks so the sectors solve concurrently.
    :return: A dict ``{parity_label: (lambdas, gaps)}`` of the leading eigenvalues and :class:`GapFunction` objects
        per solved physical frequency-parity sector (a single ``"none"`` key when no projection is requested).
    """
    logger = config.logger

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
    # the other ranks idle at the post-solve broadcast during this in-memory solve, so the solver rank may spread
    # the bandwidth-bound matvec over every core its affinity mask allows (momentum-batch threads, BLAS kept at 1)
    n_threads = _solver_thread_budget()
    executor = ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None

    chi0_mm = _chi0_to_matmul_layout(gchi0_q0_pp.mat)
    # The pairing vertex arrives in w2dynamics G2 leg order (c cdag c cdag), whereas _apply_gamma_pp expects the
    # TRIQS order (cdag c cdag c), see https://triqs.github.io/tprf/latest/theory/eliashberg.html
    gamma_mm = _gamma_to_matmul_layout(gamma_r_pp.permute_orbitals("abcd->badc", False).mat)
    gamma_r_pp.free()
    # fold the kernel prefactor into the persistent vertex once: both matvec terms inherit it by linearity, so the
    # per-matvec full-gap multiply is dropped.
    gamma_mm *= norm

    sign = 1 if gamma_r_pp.channel == SpinChannel.SING else -1

    def mv(gap: np.ndarray):
        r"""
        Applies the pairing kernel to a flattened gap vector (the matrix-vector product for the eigensolver): multiplies
        by :math:`\chi_0^{\mathrm{pp}}`, FFTs to real space, contracts with the pairing vertex (direct plus the crossed
        term, the latter reusing the direct vertex via gap-sized index shuffles), and transforms back. The orbital
        contractions are batched ``np.matmul`` products and the BZ transforms run in place through ``scipy.fft`` (both
        threaded up to the solver thread budget).

        :param gap: The flattened gap vector.
        :return: The flattened result of applying the pairing kernel to ``gap``.
        """
        gap_gg = sp.fft.fftn(
            _apply_gchi0_pp(chi0_mm, gap, n_bands), axes=(0, 1, 2), overwrite_x=True, workers=n_threads
        )
        gap_new = _apply_gamma_pp(gamma_mm, gap_gg, n_bands, executor, n_threads)
        # crossed term: Gamma_flip[K] @ gap_flip[K] == sign * flip_K[swap_ab[Gamma @ flip_p(gap_gg)]]; the flipped
        # RHS is materialized contiguously so np.matmul stays on the BLAS fast path (a single-band flip is a view).
        crossed = _apply_gamma_pp(
            gamma_mm, np.ascontiguousarray(np.flip(gap_gg, axis=-1)), n_bands, executor, n_threads
        )
        crossed = np.roll(np.flip(crossed.swapaxes(3, 4), axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
        if sign != 1:
            crossed *= sign
        gap_new += crossed
        gap_new = sp.fft.ifftn(gap_new, axes=(0, 1, 2), overwrite_x=True, workers=n_threads)
        return gap_new.flatten()

    base_seed = get_initial_gap_function(gap_shape, gamma_r_pp.channel).flatten()
    return _solve_pairing_sectors(
        mv, gap_shape, sign, gamma_r_pp.channel, gamma_r_pp.nq, executor, ranks, base_seed, parities
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
    (\chi^{\mathrm{q}\nu}_{0;ba34})^{-1}
    =\beta \sum_{ab}\sum_{\nu'} \chi^{*;\mathrm{q}\nu\nu'}_{r;ab21} (\chi^{\mathrm{q}\nu}_{0;ab34})^{-1}`, i.e. the sum
    over the first frequency argument equals the sum over the last one only up to the orbital reversal dictated by
    time-reversal symmetry, see :meth:`~dgamore.nonlocal_sde.create_vrg_r_q_right`. No explicit factors of :math:`\beta`
    appear in :math:`F^{(2)}` because they are absorbed into the stored objects: :math:`\chi^{\mathrm{q}}_{r}` is the
    (:math:`U`-dressed, shell- (and sometimes :math:`\lambda`-corrected)) physical susceptibility normalized as
    :math:`\frac{1}{\beta^2}\sum_{\nu\nu'}\chi^{\mathrm{q}\nu\nu'}_{r}`, and the three-leg vertices carry the net
    normalization :math:`\gamma^{\mathrm{q}\nu}_{r} = (\chi^{\mathrm{q}\nu}_{0})^{-1}\sum_{\nu'}
    \chi^{*;\mathrm{q}\nu\nu'}_{r}` (the explicit :math:`\beta` in their construction cancels the :math:`1/\beta` of the
    fused frequency sum), such that :math:`\gamma^{\mathrm{q}\nu}_{r} \to \mathbb{1}` for :math:`\nu \to \infty`.

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
    sing_ranks: list[int],
    trip_ranks: list[int],
    bubble_rank: int,
    parities: list[str],
) -> dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Distributes the singlet and triplet pairing vertices and the bare pp bubble across the sector ranks and solves
    every ``(channel, parity)`` sector on its assigned rank so they run concurrently (the in-memory Lanczos path).
    Each channel's vertex is gathered to every rank that owns one of its sectors (the distinct-node assignment of
    :func:`get_ranks_for_lanczos` keeps this at one copy per node); the bubble is built once on ``bubble_rank`` (the
    only rank still holding ``giwk_dga``) and shipped to the other solving ranks. The results are broadcast from each
    sector's owning rank so every rank returns the full dict.

    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param gamma_sing_pp: The singlet pairing vertex (irr-BZ q-distributed on entry; consumed).
    :param gamma_trip_pp: The triplet pairing vertex (irr-BZ q-distributed on entry; consumed).
    :param giwk_dga: The DGA Green's function (held only on ``bubble_rank``; used to build the bubble).
    :param niv_pp: The pp fermionic box size.
    :param sing_ranks: The rank owning each singlet parity sector (index ``i`` -> ``parities[i]``).
    :param trip_ranks: The rank owning each triplet parity sector.
    :param bubble_rank: The rank that builds the pp bubble (``sing_ranks[0]``).
    :param parities: The parity labels to solve.
    :return: ``{(channel, parity): (lambdas, gaps)}`` for every sector, identical on every rank.
    """
    logger = config.logger
    my_rank = mpi_dist_irrk.my_rank
    n_eig = config.eliashberg.n_eig
    distinct_sing = list(dict.fromkeys(sing_ranks))
    distinct_trip = list(dict.fromkeys(trip_ranks))

    local_sing = gamma_sing_pp.mat
    for target in distinct_sing:
        gathered = mpi_dist_irrk.gather(local_sing, root=target)
        if my_rank == target:
            gamma_sing_pp.mat = gathered
    if my_rank not in distinct_sing:
        gamma_sing_pp.free()
    local_trip = gamma_trip_pp.mat
    for target in distinct_trip:
        gathered = mpi_dist_irrk.gather(local_trip, root=target)
        if my_rank == target:
            gamma_trip_pp.mat = gathered
    if my_rank not in distinct_trip:
        gamma_trip_pp.free()

    gchi0_q_pp = None
    if my_rank == bubble_rank:
        gchi0_q_pp = BubbleGenerator.create_generalized_chi0_q_pp_w0(giwk_dga, niv_pp, config.lattice.k_grid)
        logger.info("Created the bare bubble susceptibility in pp notation.", allowed_ranks=(bubble_rank,))
    for target in dict.fromkeys(distinct_sing + distinct_trip):
        if target == bubble_rank:
            continue
        if my_rank == bubble_rank:
            mpi_dist_irrk.send_to_rank(gchi0_q_pp, dest=target, base_tag=0)
        elif my_rank == target:
            gchi0_q_pp = mpi_dist_irrk.recv_from_rank(source=bubble_rank, base_tag=0)

    my_sing = [parities[i] for i in range(len(parities)) if sing_ranks[i] == my_rank]
    my_trip = [parities[i] for i in range(len(parities)) if trip_ranks[i] == my_rank]
    sectors_sing = sectors_trip = None
    if my_sing:
        sectors_sing = solve_eliashberg_lanczos(gamma_sing_pp, gchi0_q_pp, tuple(distinct_sing), my_sing)
    if my_trip:
        sectors_trip = solve_eliashberg_lanczos(gamma_trip_pp, gchi0_q_pp, tuple(distinct_trip), my_trip)

    mpi_dist_irrk.delete_file()

    results: dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]] = {}
    for channel, sectors, ranks_list in (
        (SpinChannel.SING, sectors_sing, sing_ranks),
        (SpinChannel.TRIP, sectors_trip, trip_ranks),
    ):
        for i, parity in enumerate(parities):
            owner = ranks_list[i]
            local = sectors[parity] if (sectors is not None and parity in sectors) else None
            lambdas = mpi_dist_irrk.bcast(local[0] if local is not None else None, root=owner)
            gaps = local[1] if local is not None else [GapFunction(np.empty(0)) for _ in range(n_eig)]
            gaps = [mpi_dist_irrk.bcast_npoint(gap, root=owner) for gap in gaps]
            results[(channel, parity)] = (lambdas, gaps)
    return results


def solve(
    giwk_dga: GreensFunction, g_dmft: GreensFunction, u_loc: LocalInteraction, v_nonloc: Interaction, comm: MPI.Comm
):
    r"""
    Drives the Eliashberg step: assembles the singlet and triplet pairing vertices from the saved
    ladder-DGA full vertices (optionally adding the local reducible diagrams), then solves the linearized gap equation
    for each channel and returns the leading eigenvalues and gap functions. Solves in memory with concurrent
    (channel x parity) sectors when one sector's residency fits on a rank, and on the block-distributed solver grid
    otherwise (decided automatically from the available node memory).

    :param giwk_dga: The converged momentum-dependent DGA :class:`GreensFunction`.
    :param g_dmft: The local (DMFT) :class:`GreensFunction` (used for the local diagrams).
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param comm: The MPI communicator.
    :return: A dict keyed by ``(channel, parity_label)`` mapping to ``(lambdas, gaps)`` of the leading eigenvalues
        and :class:`GapFunction` objects for each solved physical frequency-parity sector. When
        ``config.eliashberg.resolve_frequency_parity`` is set the parity labels are ``"even"`` and ``"odd"``,
        otherwise a single unprojected ``"none"`` sector is returned.
    """
    logger = config.logger
    import psutil

    mpi_dist_irrk = MpiDistributor.create_distributor(
        ntasks=config.lattice.k_grid.nk_irr, comm=comm, name="Q", output_path=config.output.output_path
    )
    irrk_q_list = config.lattice.k_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    parities = [parity for parity, _ in _frequency_parity_sectors(config.eliashberg.resolve_frequency_parity)]
    niv_pp = min(config.box.niw_core // 2, config.box.niv_core // 2)

    # per-node memory drives the solver choice and how many (channel, parity) sectors run concurrently; the
    # residencies come from the one memory_estimator formula, so dispatch and estimate can never drift apart
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

    # one full-BZ sector residency per rank picks the in-memory solve, otherwise the grid takes over; rank 0
    # decides and broadcasts, so ranks on differently loaded nodes can never pick different solvers
    use_grid = FORCE_GRID_SOLVER or (
        comm.size > 1 and per_sector_bytes + giwk_dga.mat.nbytes > node_budget * NODE_MEMORY_FRACTION
    )
    use_grid = comm.bcast(use_grid, root=0)
    if use_grid:
        sing_ranks = trip_ranks = None
        bubble_rank = 0
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
            f"Eliashberg solver: one sector needs {per_sector_bytes / 1024**3:.3f} GB, exceeding the single-rank "
            f"budget -> block-distributed {rows}x{cols} grid, sectors sequential, {grid_bytes / 1024**3:.3f} GB "
            f"per rank."
        )
    else:
        sing_ranks, trip_ranks = get_ranks_for_lanczos(
            comm, len(parities), node_budget, per_sector_bytes, giwk_dga.mat.nbytes
        )
        bubble_rank = sing_ranks[0]
        n_concurrent = len(set(sing_ranks) | set(trip_ranks))
        n_sectors = 2 * len(parities)
        logger.info(
            f"Eliashberg solver: {n_sectors} (channel x parity) sector(s) on {n_concurrent} rank(s) "
            f"({'fully concurrent' if n_concurrent == n_sectors else 'partly sequential'}; singlet on rank(s) "
            f"{sorted(set(sing_ranks))}, triplet on rank(s) {sorted(set(trip_ranks))}), each holding at most "
            f"{per_sector_bytes / 1024**3:.3f} GB."
        )
    # giwk_dga is consumed only by the pp-bubble build on bubble_rank, so every other rank drops its copy
    if comm.rank != bubble_rank:
        giwk_dga.free()

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) if comm.size > 1 else None
    chunk_bytes = memory_estimator.dynamic_chunk_budget(
        mpi_utils.job_memory_total(), node_comm.size if node_comm is not None else 1
    )

    f_dens_pp = dispatch_full_vertex_calculation(
        SpinChannel.DENS, u_loc, v_nonloc, niv_pp, mpi_dist_irrk, chunk_bytes, node_comm
    )
    f_magn_pp = dispatch_full_vertex_calculation(
        SpinChannel.MAGN, u_loc, v_nonloc, niv_pp, mpi_dist_irrk, chunk_bytes, node_comm
    )

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
    if node_comm is not None:
        node_comm.Free()

    # Eqs. (4.49)-(4.52): the assembled vertex holds the negative crossed slot, so the local full vertex enters with
    # a relative minus and the pp-reducible diagrams phi with a plus, both in crossed-slot form ((v, -v'), 1432).
    phi_ud_loc_pp_w0 = phi_ud_loc_pp_w0.flip_frequency_axis(-1, copy=False).permute_orbitals("abcd->adcb", copy=False)
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
            sectors = solve_eliashberg_lanczos_grid(gamma_pp, gchi0_q_pp, comm, bubble_rank)
            for parity in parities:
                local = sectors[parity] if sectors is not None else None
                lambdas = comm.bcast(local[0] if local is not None else None, root=0)
                gaps = (
                    local[1]
                    if local is not None
                    else [GapFunction(np.empty(0)) for _ in range(config.eliashberg.n_eig)]
                )
                gaps = [mpi_dist_irrk.bcast_npoint(gap, root=0) for gap in gaps]
                results[(channel, parity)] = (lambdas, gaps)
    else:
        results = _solve_sectors_in_memory(
            mpi_dist_irrk, gamma_sing_pp, gamma_trip_pp, giwk_dga, niv_pp, sing_ranks, trip_ranks, bubble_rank, parities
        )

    return results


# Fraction of a node's available host memory the sector packing may occupy (mirrors DGAmore.NODE_MEMORY_FRACTION).
NODE_MEMORY_FRACTION: float = 0.95


def get_ranks_for_lanczos(
    comm: MPI.Comm,
    n_parities: int = 1,
    available_bytes: int | None = None,
    per_sector_bytes: int | None = None,
    giwk_bytes: int = 0,
) -> tuple[list[int], list[int]]:
    r"""
    Assigns MPI ranks to the singlet and triplet frequency-parity sectors so that as many as fit run concurrently.
    When a per-sector memory estimate is supplied, each node is packed with as many concurrent sector solves as its
    free memory holds (one full pairing vertex per solving rank, capped by the node's rank count), so several sectors
    may share a node when it has the headroom - never exceeding ``available_bytes * NODE_MEMORY_FRACTION`` per node,
    hence never overcommitting. The bubble node (the first solving rank) additionally reserves ``giwk_bytes``. Sectors
    beyond a node's capacity reuse an already-assigned rank of the same channel and are solved sequentially there
    (one vertex copy). Without the estimate (``per_sector_bytes is None``) it falls back to the proven 2-way: singlet
    on one node, triplet on another (or a second rank of the sole node), each channel's parities solved sequentially.

    :param comm: The MPI communicator.
    :param n_parities: The number of frequency-parity sectors per channel (2 when resolving parity, else 1).
    :param available_bytes: This rank's free host memory (:func:`psutil.virtual_memory().available`), allgathered and
        reduced (minimum) per node; ``None`` selects the memory-unaware 2-way fallback.
    :param per_sector_bytes: The estimated peak host memory of one in-memory sector solve (dominated by the full-BZ
        pairing vertex); ``None`` selects the fallback.
    :param giwk_bytes: The DGA Green's function size held on the bubble node while it builds the pp bubble.
    :return: ``(singlet_ranks, triplet_ranks)``, each a list of ``n_parities`` ranks (the rank that owns parity ``i``).
    """
    info = comm.allgather((socket.gethostname(), available_bytes))
    node_to_ranks: dict = {}
    node_available: dict = {}
    for r, (host, avail) in enumerate(info):
        node_to_ranks.setdefault(host, []).append(r)
        if avail is not None:
            node_available[host] = avail if host not in node_available else min(node_available[host], avail)
    nodes = list(node_to_ranks)

    if per_sector_bytes is None or not node_available:
        # no memory estimate: the proven 2-way (channels concurrent, a channel's parities sequential on its rank)
        if len(nodes) >= 2:
            singlet_rank, triplet_rank = node_to_ranks[nodes[0]][0], node_to_ranks[nodes[1]][0]
        else:
            ranks_on_node = node_to_ranks[nodes[0]]
            singlet_rank = ranks_on_node[0]
            triplet_rank = ranks_on_node[1] if len(ranks_on_node) > 1 else ranks_on_node[0]
        return [singlet_rank] * n_parities, [triplet_rank] * n_parities

    # concurrent vertices a node can hold (the first node also stores the giwk for the bubble build)
    capacity = {}
    for i, host in enumerate(nodes):
        budget = node_available[host] * NODE_MEMORY_FRACTION - (giwk_bytes if i == 0 else 0)
        capacity[host] = max(1, min(len(node_to_ranks[host]), int(budget // per_sector_bytes)))

    # distinct solving ranks (one vertex each), filled round-robin across nodes up to each node's capacity
    n_sectors = 2 * n_parities
    slots: list[int] = []
    used = {host: 0 for host in nodes}
    while len(slots) < n_sectors:
        progressed = False
        for host in nodes:
            if used[host] < capacity[host]:
                slots.append(node_to_ranks[host][used[host]])
                used[host] += 1
                progressed = True
                if len(slots) >= n_sectors:
                    break
        if not progressed:
            break

    if len(slots) <= 1:
        rank = slots[0] if slots else node_to_ranks[nodes[0]][0]
        return [rank] * n_parities, [rank] * n_parities

    singlet_count = min(n_parities, (len(slots) + 1) // 2)
    singlet_slots, triplet_slots = slots[:singlet_count], slots[singlet_count:]
    singlet_ranks = [singlet_slots[i % len(singlet_slots)] for i in range(n_parities)]
    triplet_ranks = [triplet_slots[i % len(triplet_slots)] for i in range(n_parities)]
    return singlet_ranks, triplet_ranks

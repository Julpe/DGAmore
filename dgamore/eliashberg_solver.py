# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Linearized Eliashberg equation solver. Starting from the ladder-DGA full vertex (saved per channel by the
non-local SDE step), this module assembles the particle-particle pairing vertex in the singlet/triplet channels at
:math:`\omega = 0`, optionally adds the local reducible diagrams, and solves the linearized gap equation
:math:`\lambda \Delta = \pm\frac{1}{2\beta N_q}\, \Gamma^{pp}\, \chi_0^{pp}\, \Delta` with a matrix-free
ARPACK/Lanczos eigensolver (two variants: an in-memory one and a memory-lean frequency-distributed one). The leading
eigenvalue :math:`\lambda` signals the pairing instability and the eigenvector is the gap function
:math:`\Delta(k, \nu)`. Equation numbers refer to the author's master's thesis (Chapter 4).
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
from dgamore.memory_estimator import LANCZOS_VERTEX_FACTOR
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel, FrequencyNotation, DTYPE


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
def _transform_vertex_frequencies_w0(vertex: LocalFourPoint | FourPoint, niv_pp: int) -> np.ndarray:
    r"""
    Transforms a vertex from particle-hole to particle-particle notation at :math:`\omega' = 0`, following Motoharu
    Kitatani's frequency convention: the fermionic frequency is flipped, the bosonic index is remapped via
    :math:`\omega = \nu - \nu'` and the orbitals are permuted to :math:`1432`. In full index notation the output is

    .. math:: \bar{F}^{pp;\nu\nu'}_{1234} = -F^{ph;\,\omega=\nu-\nu';\ \nu_1=\nu,\ \nu_2=-\nu'}_{1432}
        = -F^{ph;(\nu-\nu')\nu(-\nu')}_{1432},

    i.e. (minus) the crossed-slot form of the pairing vertex of Eq. (4.49) in my thesis: with the ph frequency
    convention of Eq. (3.28a) the four legs of :math:`\bar{F}^{pp;\nu\nu'}_{1234}` carry the frequencies
    :math:`(\nu, \nu', -\nu, -\nu')` on the orbitals :math:`(1, 4, 3, 2)`. The overall minus is the sign of the
    power-iteration matrix :math:`M = -\Gamma\chi` of Eq. (4.42). Used by
    :func:`transform_vertex_loc_frequencies_w0` and :func:`transform_vertex_q_frequencies_w0`; the direct-slot
    counterpart (:math:`\omega_{ph} = \nu + \nu'`, no flip, orbitals :math:`1234`) is
    :meth:`~dgamore.local_four_point.LocalFourPoint.change_frequency_notation_ph_to_pp_w0`.

    :param vertex: The vertex to transform (:class:`LocalFourPoint` or :class:`FourPoint`) in ph notation.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :return: The transformed vertex as a raw numpy array with two fermionic axes ``[..., 2*niv_pp, 2*niv_pp]``.
    """
    vn = MFHelper.vn(niv_pp)
    omega = vn[:, None] - vn[None, :]

    vertex = vertex.cut_niv(niv_pp)
    # only the |w| <= 2*niv_pp - 1 anti-diagonals (omega = v - v') are read below, so the bosonic axis is trimmed to
    # that window before to_full_niw_range doubles it (cut_niw's no-op guard misjudges half-range objects here)
    w_axis = -3
    niw_stored = vertex.current_shape[w_axis] // 2 if vertex.full_niw_range else vertex.current_shape[w_axis] - 1
    niw_window = min(niw_stored, 2 * niv_pp - 1)
    if niw_window < niw_stored:
        slicer = [slice(None)] * vertex.mat.ndim
        slicer[w_axis] = (
            slice(niw_stored - niw_window, niw_stored + niw_window + 1)
            if vertex.full_niw_range
            else slice(0, niw_window + 1)
        )
        vertex.mat = vertex.mat[tuple(slicer)].copy()
        vertex.update_original_shape()
    wn = MFHelper.wn(niw_window)

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


def transform_vertex_q_frequencies_w0(f_q_r: FourPoint, niv_pp: int) -> FourPoint:
    r"""
    Transforms a momentum-dependent vertex from particle-hole to the modified particle-particle notation at
    :math:`\omega' = 0` (see :func:`_transform_vertex_frequencies_w0`).

    :param f_q_r: The momentum-dependent vertex :math:`F^{q}` in ph notation.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :return: The transformed vertex as a :class:`FourPoint` (pp notation, no bosonic axis, compressed q).
    """
    mat = _transform_vertex_frequencies_w0(f_q_r, niv_pp)
    return FourPoint(mat, f_q_r.channel, config.lattice.k_grid.nk, 0, 2, True, True, True, FrequencyNotation.PP)


# --- Full q-dependent vertex creation and transformation ---
def create_full_vertex_q_r(
    u_loc: LocalInteraction, v_nonloc: Interaction, gamma_r: LocalFourPoint, niv_pp: int, mpi_dist: MpiDistributor
) -> FourPoint:
    r"""
    Calculates the momentum-dependent full ladder vertex in the given channel (density or magnetic) from the saved
    intermediates (inverse bubble, three-leg vertex, summed auxiliary susceptibility), and transforms it to pp
    notation unless ``save_fq`` requests keeping the ph form. Deletes the consumed intermediate files afterwards.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param mpi_dist: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :return: The full ladder vertex :math:`F^{q}_{r}` as a :class:`FourPoint`.
    """
    logger = config.logger
    logger.info(f"Starting to calculate the full {gamma_r.channel.value} vertex.")

    gchi0_q_inv = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"gchi0_q_inv_rank_{mpi_dist.my_rank}.npy"), num_vn_dimensions=1
    )

    logger.info(f"Loaded gchi0_q_inv from file.")
    f_q_r = nonlocal_sde.create_auxiliary_chi_r_q(gamma_r, gchi0_q_inv, u_loc, v_nonloc)
    logger.info(f"Non-Local auxiliary susceptibility ({gamma_r.channel.value}) calculated.")

    # eager rebinding releases chi* right after the first matmul, and the bubble term enters on the fermionic
    # diagonal in place - the former single-expression form held four two-fermion blocks at its subtraction peak
    f_q_r = gchi0_q_inv @ f_q_r
    f_q_r = f_q_r @ gchi0_q_inv
    f_q_r = f_q_r.scale(-config.sys.beta**2).add_on_vn_diagonal(gchi0_q_inv, factor=config.sys.beta**2)
    gchi0_q_inv.free()

    if not config.eliashberg.save_fq:
        f_q_r = transform_vertex_q_frequencies_w0(f_q_r, niv_pp)

    mpi_dist.barrier()

    logger.info(f"Calculated first part of full {gamma_r.channel.value} vertex.")

    vrg_q_r_left = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"vrg_q_{gamma_r.channel.value}_rank_{mpi_dist.my_rank}.npy"),
        channel=gamma_r.channel,
        num_vn_dimensions=1,
    )

    chi_phys_q_r = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"chi_phys_q_{gamma_r.channel.value}_rank_{mpi_dist.my_rank}.npy"),
        channel=gamma_r.channel,
        num_vn_dimensions=0,
    )
    logger.info(f"Loaded vrg_q_{gamma_r.channel.value} and chi_phys_q_{gamma_r.channel.value} from files.")

    u = u_loc.as_channel(gamma_r.channel) + v_nonloc.as_channel(gamma_r.channel)
    f_q_r_2 = vrg_q_r_left @ u - vrg_q_r_left @ (u @ chi_phys_q_r @ u)
    vrg_q_r_left.free()
    chi_phys_q_r.free()

    vrg_q_r_right = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"vrg_q_{gamma_r.channel.value}_right_rank_{mpi_dist.my_rank}.npy"),
        channel=gamma_r.channel,
        num_vn_dimensions=1,
    )

    f_q_r_2 = f_q_r_2 * vrg_q_r_right
    vrg_q_r_right.free()

    if not config.eliashberg.save_fq:
        f_q_r_2 = transform_vertex_q_frequencies_w0(f_q_r_2, niv_pp)
    f_q_r = f_q_r.add(f_q_r_2, copy=False)  # accumulate in place: no third full-size vertex block
    f_q_r_2.free()

    mpi_dist.barrier()

    logger.info(f"Calculated second part of full {f_q_r.channel.value} vertex.")

    delete_files(
        config.output.eliashberg_path,
        f"vrg_q_{gamma_r.channel.value}_rank_{mpi_dist.my_rank}.npy",
        f"vrg_q_{gamma_r.channel.value}_right_rank_{mpi_dist.my_rank}.npy",
        f"chi_phys_q_{gamma_r.channel.value}_rank_{mpi_dist.my_rank}.npy",
    )

    return f_q_r


def create_full_vertex_q_r_pp_w0(
    u_loc: LocalInteraction, v_nonloc: Interaction, gamma_r: LocalFourPoint, niv_pp: int, mpi_dist_irrk: MpiDistributor
):
    r"""
    Builds the full ladder vertex (see :func:`create_full_vertex_q_r`), optionally gathers and saves it in ph
    notation in the irreducible BZ, and returns it transformed to pp notation at :math:`\omega' = 0`.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :return: The full ladder vertex :math:`F^{q}_{r}` in pp notation as a :class:`FourPoint`.
    """
    logger = config.logger

    f_q_r = create_full_vertex_q_r(u_loc, v_nonloc, gamma_r, niv_pp, mpi_dist_irrk)

    logger.info(f"Full ladder-vertex ({f_q_r.channel.value}) calculated.")
    logger.log_memory_usage(
        f"Full ladder-vertex ({f_q_r.channel.value})",
        f_q_r,
        mpi_dist_irrk.comm.size * (1 if config.eliashberg.save_fq else 4 * (config.box.niw_core + 1)),
    )

    if config.eliashberg.save_fq:
        f_q_r.mat = mpi_dist_irrk.gather(f_q_r.mat)
        if mpi_dist_irrk.comm.rank == 0:
            f_q_r.save(output_dir=config.output.output_path, name=f"f_irrq_{f_q_r.channel.value}")
        f_q_r.mat = mpi_dist_irrk.scatter(f_q_r.mat)
        config.logger.info(f"Saved full ladder-vertex ({f_q_r.channel.value}) in the irreducible BZ to file.")

    if config.eliashberg.save_fq:
        return transform_vertex_q_frequencies_w0(f_q_r, niv_pp)
    return f_q_r


def create_full_vertex_q_r_v2(
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    vrg_q_r_left: FourPoint,
    vrg_q_r_right: FourPoint,
    chi_phys_q_r: FourPoint,
    niv_pp: int,
    q_index: int,
) -> FourPoint:
    r"""
    Calculates the full ladder vertex for a single q-point (memory-lean per-q variant of
    :func:`create_full_vertex_q_r`), transforming it to pp notation unless ``save_fq`` keeps the ph form.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi_0^q)^{-1}` over all rank-local q-points.
    :param vrg_q_r_left: The momentum-dependent three-leg vertex :math:`\gamma^q_{r}`.
    :param vrg_q_r_right: The momentum-dependent "right-side" three-leg vertex :math:`\gamma^q_{r}`.
    :param chi_phys_q_r: The physical susceptibility :math:`\chi^{phys;q}_{r}`.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param q_index: Index of the q-point (into the rank-local list) to compute.
    :return: The full ladder vertex :math:`F^{q}_{r}` for that q-point as a :class:`FourPoint`.
    """
    gchi0_q_inv_idx = gchi0_q_inv.filter_q_index(q_index)
    vrg_q_r_left_idx = vrg_q_r_left.filter_q_index(q_index)
    vrg_q_r_right_idx = vrg_q_r_right.filter_q_index(q_index)
    chi_phys_q_r_idx = chi_phys_q_r.filter_q_index(q_index)
    v_nonloc_idx = v_nonloc.filter_q_index(q_index)

    u = u_loc.as_channel(gamma_r.channel) + v_nonloc_idx.as_channel(gamma_r.channel)

    f_q_r_idx = nonlocal_sde.create_auxiliary_chi_r_q(gamma_r, gchi0_q_inv_idx, u_loc, v_nonloc_idx)
    # same eager-rebinding / diagonal-add construction as in create_full_vertex_q_r, per q-point
    f_q_r_idx = gchi0_q_inv_idx @ f_q_r_idx
    f_q_r_idx = f_q_r_idx @ gchi0_q_inv_idx
    f_q_r_idx = f_q_r_idx.scale(-config.sys.beta**2).add_on_vn_diagonal(gchi0_q_inv_idx, factor=config.sys.beta**2)
    f_q_r_idx = f_q_r_idx.add(
        (vrg_q_r_left_idx @ u - vrg_q_r_left_idx @ (u @ chi_phys_q_r_idx @ u)) * vrg_q_r_right_idx, copy=False
    )

    gchi0_q_inv_idx.free()
    vrg_q_r_left_idx.free()
    vrg_q_r_right_idx.free()
    chi_phys_q_r_idx.free()

    if not config.eliashberg.save_fq:
        f_q_r_idx = transform_vertex_q_frequencies_w0(f_q_r_idx, niv_pp)

    return f_q_r_idx


def create_full_vertex_q_r_pp_w0_v2(
    u_loc: LocalInteraction, v_nonloc: Interaction, gamma_r: LocalFourPoint, niv_pp: int, mpi_dist_irrk: MpiDistributor
):
    r"""
    Builds the full ladder vertex as a memory-lean variant of :func:`create_full_vertex_q_r_pp_w0`, looping over the
    rank-local q-points (see :func:`create_full_vertex_q_r_v2`), optionally saving it in ph notation, and returning it
    in pp notation at :math:`\omega' = 0`.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` for this channel.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :return: The full ladder vertex :math:`F^{q}_{r}` in pp notation as a :class:`FourPoint`.
    """
    logger = config.logger

    gchi0_q_inv = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"gchi0_q_inv_rank_{mpi_dist_irrk.my_rank}.npy"),
        num_vn_dimensions=1,
    )
    logger.info(f"Loaded gchi0_q_inv from file.")

    vrg_q_r_left = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"vrg_q_{gamma_r.channel.value}_rank_{mpi_dist_irrk.my_rank}.npy"),
        channel=gamma_r.channel,
        num_vn_dimensions=1,
    )

    vrg_q_r_right = FourPoint.load(
        os.path.join(
            config.output.eliashberg_path, f"vrg_q_{gamma_r.channel.value}_right_rank_{mpi_dist_irrk.my_rank}.npy"
        ),
        channel=gamma_r.channel,
        num_vn_dimensions=1,
    )

    chi_phys_q_r = FourPoint.load(
        os.path.join(
            config.output.eliashberg_path, f"chi_phys_q_{gamma_r.channel.value}_rank_{mpi_dist_irrk.my_rank}.npy"
        ),
        channel=gamma_r.channel,
        num_vn_dimensions=0,
    )

    logger.info(f"Loaded vrg_q_{gamma_r.channel.value} and chi_phys_q_{gamma_r.channel.value} from files.")

    irrk_q_list = config.lattice.k_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    if config.eliashberg.save_fq:
        f_q_r_mat = np.empty(
            (
                (len(my_irr_q_list),)
                + (config.sys.n_bands,) * 4
                + (gamma_r.current_shape[-3],)
                + (2 * config.box.niv_core,) * 2
            ),
            dtype=gamma_r.mat.dtype,
        )
    else:
        f_q_r_mat = np.empty(
            ((len(my_irr_q_list),) + (config.sys.n_bands,) * 4 + (2 * niv_pp,) * 2), dtype=gamma_r.mat.dtype
        )

    logger.info(f"Starting to calculate the full {gamma_r.channel.value} vertex.")

    for idx, q in enumerate(my_irr_q_list):
        f_q_r_mat[idx] = create_full_vertex_q_r_v2(
            u_loc, v_nonloc, gamma_r, gchi0_q_inv, vrg_q_r_left, vrg_q_r_right, chi_phys_q_r, niv_pp, idx
        ).mat

    logger.info(f"Full ladder-vertex ({gamma_r.channel.value}) calculated.")

    gchi0_q_inv.free()
    vrg_q_r_left.free()
    vrg_q_r_right.free()
    chi_phys_q_r.free()

    delete_files(
        config.output.eliashberg_path,
        f"vrg_q_{gamma_r.channel.value}_rank_{mpi_dist_irrk.my_rank}.npy",
        f"vrg_q_{gamma_r.channel.value}_right_rank_{mpi_dist_irrk.my_rank}.npy",
        f"chi_phys_q_{gamma_r.channel.value}_rank_{mpi_dist_irrk.my_rank}.npy",
    )

    if not config.eliashberg.save_fq:
        f_q_r = FourPoint(
            f_q_r_mat, gamma_r.channel, config.lattice.k_grid.nk, 0, 2, False, True, True, FrequencyNotation.PP
        )
        logger.log_memory_usage(
            f"Full ladder-vertex ({f_q_r.channel.value})",
            f_q_r,
            mpi_dist_irrk.comm.size * 4 * (config.box.niw_core + 1),
        )
        return f_q_r

    f_q_r = FourPoint(
        f_q_r_mat, gamma_r.channel, config.lattice.k_grid.nk, 1, 2, False, True, True, FrequencyNotation.PP
    )
    logger.log_memory_usage(f"Full ladder-vertex ({f_q_r.channel.value})", f_q_r, mpi_dist_irrk.comm.size)
    f_q_r.mat = mpi_dist_irrk.gather(f_q_r.mat)
    if mpi_dist_irrk.comm.rank == 0:
        f_q_r.save(output_dir=config.output.output_path, name=f"f_irrq_{f_q_r.channel.value}")
        logger.info(f"Saved full ladder-vertex ({f_q_r.channel.value}) in the irreducible BZ to file.")
    f_q_r.mat = mpi_dist_irrk.scatter(f_q_r.mat)
    return transform_vertex_q_frequencies_w0(f_q_r, niv_pp)


# --- Local particle-particle reducible diagrams (w=0) ---
def create_local_gamma_ud_pp_w0(
    gchi_ud_pp_w0: LocalFourPoint, gchi0_pp_w0: LocalFourPoint, beta: float
) -> LocalFourPoint:
    r"""
    Returns the local pp-irreducible up-down vertex at :math:`\omega = 0` from the crossing-decoupled pp
    Bethe-Salpeter equation,

    .. math:: \Gamma^{pp;\nu\nu'}_{\uparrow\downarrow;1234} = \beta^2 \left[\chi^{pp}_0 J - \chi^{pp}_0\,
        (\chi^{pp}_{\uparrow\downarrow})^{-1}\, \chi^{pp}_0\right]^{-1;\,\nu\nu'}_{1234}.

    All products and inverses live in compound pp index space, i.e. as matrices
    :math:`M_{(13\nu),(42\nu')} = X^{pp;\nu\nu'}_{1234}` with the product and unit element

    .. math:: (X Y)^{pp;\nu\nu'}_{1234} = \sum_{ab\nu_1} X^{pp;\nu\nu_1}_{1a3b}\, Y^{pp;\nu_1\nu'}_{b2a4}, \qquad
        \mathbb{1}^{pp;\nu\nu'}_{1234} = \delta_{14}\,\delta_{23}\,\delta_{\nu\nu'}.

    The ingredients in full index notation are the diagonal bare pp bubble, built from the local DMFT Green's
    function :math:`G^{\mathrm{DMFT}}_{12}(\nu)`, and its image under the crossing operator :math:`J`
    (:math:`\nu' \to -\nu'` combined with the orbital permutation :math:`1234 \to 1432`, i.e.
    :math:`(XJ)^{pp;\nu\nu'}_{1234} = X^{pp;\nu(-\nu')}_{1432}`),

    .. math:: \chi^{pp;\nu\nu'}_{0;1234} = -\beta\, G^{\mathrm{DMFT}}_{14}(\nu)\, G^{\mathrm{DMFT}}_{32}(-\nu)\,
        \delta_{\nu\nu'}, \qquad (\chi^{pp}_0 J)^{\nu\nu'}_{1234} = -\beta\, G^{\mathrm{DMFT}}_{12}(\nu)\,
        G^{\mathrm{DMFT}}_{34}(-\nu)\, \delta_{\nu,-\nu'}.

    The returned :math:`\Gamma^{pp}_{\uparrow\downarrow}` is equivalent to solving the crossing-decoupled pp BSE

    .. math:: F^{pp;\nu\nu'}_{\uparrow\downarrow;1234} = \Gamma^{pp;\nu\nu'}_{\uparrow\downarrow;1234}
        - \frac{1}{\beta} \sum_{\nu_1} \sum_{abcd} \Gamma^{pp;\nu\nu_1}_{\uparrow\downarrow;1a3b}\,
        G^{\mathrm{DMFT}}_{bc}(\nu_1)\, G^{\mathrm{DMFT}}_{ad}(-\nu_1)\,
        F^{pp;(-\nu_1)\nu'}_{\uparrow\downarrow;d2c4}

    for the full vertex :math:`F^{pp}_{\uparrow\downarrow}` defined by amputating the DMFT legs of the
    susceptibility,

    .. math:: \chi^{pp;\nu\nu'}_{\uparrow\downarrow;1234} = -\sum_{abcd} F^{pp;\nu\nu'}_{\uparrow\downarrow;abcd}\,
        G^{\mathrm{DMFT}}_{1a}(\nu)\, G^{\mathrm{DMFT}}_{b2}(-\nu')\, G^{\mathrm{DMFT}}_{3c}(-\nu)\,
        G^{\mathrm{DMFT}}_{d4}(\nu').

    Note that :math:`\chi^{pp}_{\uparrow\downarrow}` must be the CONNECTED susceptibility: the disconnected
    straight term :math:`\delta_{\omega_{ph} 0}\, \beta\, G^{\mathrm{DMFT}}_{12}(\nu)\, G^{\mathrm{DMFT}}_{34}(\nu')`
    would land exactly on the pp anti-diagonal :math:`\nu' = -\nu` and corrupt the :math:`\chi^{pp}_0 J` rung. The
    loader guarantees this: :func:`~dgamore.local_sde.create_generalized_chi` subtracts that term in the density
    channel, and the :math:`\frac{1}{2}(\chi^{ph}_{d} - \chi^{ph}_{m})` combination cancels both it and the
    vertical bubble exactly.

    :math:`J` commutes with every pp object by crossing symmetry, so this is the full-space form of inverting the
    decoupled singlet/triplet BSEs (thesis Eqs. 3.51/3.52) on their :math:`J`-even/odd blocks. For a single band
    :math:`J` reduces to the plain frequency flip and the expression is equivalent to Eq. (B.26) of Rohringer's
    thesis. Assumes :math:`G^{\mathrm{DMFT}}_{12}(\nu) = G^{\mathrm{DMFT}}_{21}(\nu)` (real orbital basis, no
    spin-orbit coupling); with SOC the rung :math:`\chi^{pp}_0 J` must be replaced by
    :math:`-\beta\, G^{\mathrm{DMFT}}_{12}(\nu)\, G^{\mathrm{DMFT}}_{43}(-\nu)\, \delta_{\nu,-\nu'}` (second
    Green's function transposed).

    :param gchi_ud_pp_w0: The local connected up-down susceptibility
        :math:`\chi^{pp;\nu\nu'}_{\uparrow\downarrow;1234}` in pp notation at :math:`\omega = 0`, see
        :meth:`~dgamore.local_four_point.LocalFourPoint.change_frequency_notation_ph_to_pp_w0`.
    :param gchi0_pp_w0: The local bare pp bubble :math:`\chi^{pp;\nu\nu'}_{0;1234}` (diagonal in :math:`\nu\nu'`),
        built from the DMFT Green's function via
        :meth:`~dgamore.bubble_gen.BubbleGenerator.create_generalized_chi0_pp_w0`.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The vertex :math:`\Gamma^{pp;\nu\nu'}_{\uparrow\downarrow;1234}` as a :class:`LocalFourPoint` in pp
        notation.
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
    Builds the local pp-irreducible up-down vertex :math:`\Gamma^{pp;\nu\nu'}_{\uparrow\downarrow;1234}` per
    inequivalent atom and assembles the per-atom blocks into the full multi-band object (mirroring the local
    Schwinger-Dyson assembly). Local correlations do not connect orbitals of different atoms, so the assembled
    multi-band susceptibility is nonzero only when all four orbital indices belong to the same atom; the compound
    pp matrix of the FULL object is therefore singular for more than one atom and must never be inverted directly.
    Instead, :func:`create_local_gamma_ud_pp_w0` is evaluated on each atom's orbital block (with the bare pp
    bubble built from that atom's block of :math:`G^{\mathrm{DMFT}}_{12}(\nu)`), computing every inequivalent atom
    only once and writing the result into all of its positions in the compound band layout.

    :param gchi_ud_pp_w0: The full multi-band connected up-down susceptibility
        :math:`\chi^{pp;\nu\nu'}_{\uparrow\downarrow;1234}` in pp notation at :math:`\omega = 0`
        (block-structured per inequivalent atom).
    :param g_dmft: The full multi-band local DMFT :class:`GreensFunction` :math:`G^{\mathrm{DMFT}}_{12}(\nu)`.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The assembled vertex :math:`\Gamma^{pp;\nu\nu'}_{\uparrow\downarrow;1234}` as a
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
            g_block = GreensFunction(g_mat_block.reshape(g_mat_block.shape[-3:]).copy())
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
    Builds the local particle-particle reducible diagrams at :math:`\omega = 0` in the up-down channel: the full
    vertex :math:`F^{pp;\nu\nu'}_{\uparrow\downarrow;1234}`, the pp-irreducible vertex
    :math:`\Gamma^{pp;\nu\nu'}_{\uparrow\downarrow;1234}` (built per inequivalent atom and assembled into the full
    multi-band object, see :func:`create_local_gamma_ud_pp_w0_per_ineq`), and the reducible part

    .. math:: \Phi^{pp;\nu\nu'}_{\uparrow\downarrow;1234} = F^{pp;\nu\nu'}_{\uparrow\downarrow;1234}
        - \Gamma^{pp;\nu\nu'}_{\uparrow\downarrow;1234},

    with :math:`\chi^{pp}_{\uparrow\downarrow} = \frac{1}{2}(\chi^{ph}_{d} - \chi^{ph}_{m})` mapped to pp notation
    at :math:`\omega_{pp} = 0` via :meth:`~dgamore.local_four_point.LocalFourPoint.change_frequency_notation_ph_to_pp_w0`
    (ph legs evaluated at :math:`\omega_{ph} = \nu + \nu'`) and the bare pp bubble built from the local DMFT
    Green's function :math:`G^{\mathrm{DMFT}}_{12}(\nu)` via
    :meth:`~dgamore.bubble_gen.BubbleGenerator.create_generalized_chi0_pp_w0`. These are the local diagrams
    subtracted/added when ``include_local_part`` of :class:`~dgamore.config.EliashbergConfig` is enabled, to avoid
    double counting the local pairing contribution (thesis Eqs. 4.49-4.52).

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

    :param shape: Target array shape ``[kx, ky, kz, o1, o2, v]`` of the gap function.
    :param channel: Pairing channel, either :attr:`SpinChannel.SING` or :attr:`SpinChannel.TRIP`.
    :return: The initial gap-function array.
    :raises ValueError: If ``channel`` is neither SING nor TRIP.
    """
    if channel not in {SpinChannel.SING, SpinChannel.TRIP}:
        raise ValueError("Channel must be either SING or TRIP.")

    gap0 = np.zeros(shape, dtype=DTYPE)
    niv = shape[-1] // 2
    k_grid = config.lattice.k_grid.grid

    symm = {
        "d-wave": lambda k: -np.cos(k[0])[:, None, None] + np.cos(k[1])[None, :, None],
        "p-wave-x": lambda k: np.sin(k[0])[:, None, None],
        "p-wave-y": lambda k: np.sin(k[1])[None, :, None],
    }

    if config.eliashberg.symmetry in symm:
        gap0[..., niv:] = np.repeat(symm[config.eliashberg.symmetry](k_grid)[:, :, :, None, None, None], niv, axis=-1)
    else:
        gap0 = np.random.random_sample(shape)

    v_sym = {
        "d-wave": "even" if channel == SpinChannel.SING else "odd",
        "p-wave-x": "odd" if channel == SpinChannel.SING else "even",
        "p-wave-y": "odd" if channel == SpinChannel.SING else "even",
    }.get(config.eliashberg.symmetry, "")

    if v_sym in {"even", "odd"}:
        gap0[..., :niv] = gap0[..., niv:] if v_sym == "even" else -gap0[..., niv:]
    else:
        gap0 = np.random.random_sample(shape)

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
    three involutions act on the orbital gap :math:`\Delta_{12}^{\nu}(k)` as
    :math:`(T\Delta)_{12}^{\nu}(k) = \Delta_{12}^{-\nu}(k)` (fermionic-frequency flip),
    :math:`(P\Delta)_{12}^{\nu}(k) = \Delta_{12}^{\nu}(-k)` (momentum flip) and
    :math:`(O\Delta)_{12}^{\nu}(k) = \Delta_{21}^{\nu}(k)` (orbital transpose), realized by the same array operations
    the pairing-kernel matvec uses. The Pauli antisymmetry :math:`\hat{S}\,P\,O\,T\,\Delta = -\Delta` with the spin
    exchange :math:`\hat{S}` a scalar in the singlet/triplet basis fixes :math:`P\,O\,T\,\Delta = \mathrm{sign}\,\Delta`
    (``sign`` the channel sign), so once the frequency parity :math:`\varepsilon_T` is chosen the combined
    momentum-orbital parity is forced to :math:`\varepsilon_{PO} = \mathrm{sign}\cdot\varepsilon_T`; only the product
    :math:`P\,O` is fixed, never :math:`P` and :math:`O` separately.

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


def classify_gap_symmetry(gap: np.ndarray) -> str:
    r"""
    Classifies the dominant momentum wave symmetry and Matsubara-frequency parity of a gap and returns a compact
    label of the form ``<wave><parity>``. The wave letter is ``s``, ``d`` or ``p`` (``x`` if none of these match),
    and the parity sign is ``+`` (even in :math:`\nu`), ``-`` (odd) or empty (neither). The frequency parity is the
    sign of the global T Rayleigh quotient :math:`\langle \Delta, T\Delta \rangle / \langle \Delta, \Delta \rangle`
    (with :math:`T` the fermionic-frequency flip), so it is consistent with the parity diagnostics and robust for
    multi-orbital gaps. The wave symmetry is read from the orbital-diagonal ``o1 = o2 = 0``, ``kz = 0`` block at the
    first positive Matsubara frequency under the lattice inversions (:math:`k_x \to -k_x`, :math:`k_y \to -k_y`, the
    full :math:`k \to -k`, realized with the ``np.roll(np.flip(...), 1)`` convention for the Gamma-at-index-0 grid)
    and the :math:`k_x \leftrightarrow k_y` exchange: ``s`` is even under both axis inversions and symmetric under
    exchange, ``d`` is even under both axis inversions and antisymmetric under exchange, and ``p`` is odd under the
    full inversion.

    :param gap: The gap array in the ``[kx, ky, kz, o1, o2, v]`` layout.
    :return: The ``<wave><parity>`` label, ``"unknown"`` for an all-zero gap, or ``x<parity>`` when the wave cannot
        be determined from the orbital-diagonal block.
    """
    atol = 1e-3
    denom = np.vdot(gap, gap)
    if denom == 0:
        return "unknown"
    t = (np.vdot(gap, np.flip(gap, axis=-1)) / denom).real
    freq_label = "+" if t > 0.5 else ("-" if t < -0.5 else "")

    d_plus = gap[
        ..., 0, 0, 0, gap.shape[-1] // 2
    ].real  # [kx, ky]: orbital-diagonal (0, 0) block at kz = 0, nu = +pi/beta
    scale = np.max(np.abs(d_plus))
    if scale == 0:
        return f"x{freq_label}"
    d_plus = d_plus / scale
    inv_x = np.roll(np.flip(d_plus, axis=0), shift=1, axis=0)
    inv_y = np.roll(np.flip(d_plus, axis=1), shift=1, axis=1)
    inv_full = np.roll(np.flip(d_plus, axis=(0, 1)), shift=(1, 1), axis=(0, 1))
    even_x = np.allclose(d_plus, inv_x, atol=atol, rtol=0)
    even_y = np.allclose(d_plus, inv_y, atol=atol, rtol=0)
    odd_full = np.allclose(d_plus, -inv_full, atol=atol, rtol=0)
    square = d_plus.shape[0] == d_plus.shape[1]
    xy_sym = square and np.allclose(d_plus, d_plus.T, atol=atol, rtol=0)
    xy_anti = square and np.allclose(d_plus, -d_plus.T, atol=atol, rtol=0)

    if even_x and even_y and xy_sym:
        return f"s{freq_label}"
    if even_x and even_y and xy_anti:
        return f"d{freq_label}"
    if odd_full:
        return f"p{freq_label}"
    return f"x{freq_label}"


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


def _v2_solver_thread_budget(comm: MPI.Comm, active_ranks: list) -> int:
    r"""
    Returns the momentum-batch/FFT thread budget of THIS rank for the frequency-distributed Lanczos solve. Unlike
    the in-memory solve (where all other ranks of the node wait and the one solver rank may claim its whole
    affinity mask, see :func:`_solver_thread_budget`), every rank with a non-empty frequency slice computes
    simultaneously here, so this rank's mask is divided by the number of active ranks on its node whose affinity
    masks overlap with it - threading them all at full mask would oversubscribe the shared cores. The result is
    clamped to the OpenBLAS thread capacity (see :func:`_clamp_to_openblas_slot_cap`). Under a strict
    one-core-per-rank binding the budget is 1 and the threading is a no-op; idle cores are only picked up when the
    launcher binding leaves masks wider than one core (e.g. socket binding) and some of the node's ranks hold
    empty frequency slices. This is a collective call - every rank of ``comm`` must enter it, active or not.

    :param comm: The MPI communicator.
    :param active_ranks: The ranks holding non-empty frequency slices (see :func:`solve_eliashberg_lanczos_v2`).
    :return: The thread budget of this rank as an int (1 for inactive ranks and where no affinity API exists).
    """
    try:
        my_mask = frozenset(os.sched_getaffinity(0))
    except AttributeError:
        my_mask = None

    if comm.size == 1:
        return _clamp_to_openblas_slot_cap(max(1, len(my_mask))) if my_mask else 1

    infos = comm.allgather((socket.gethostname(), my_mask))
    if my_mask is None or comm.rank not in active_ranks:
        return 1

    my_host = infos[comm.rank][0]
    sharing = sum(
        1 for r in active_ranks if infos[r][0] == my_host and infos[r][1] is not None and infos[r][1] & my_mask
    )
    return _clamp_to_openblas_slot_cap(max(1, len(my_mask) // max(1, sharing)))


def _chi0_to_matmul_layout(chi0_mat: np.ndarray) -> np.ndarray:
    r"""
    Returns the bare pp bubble in batched-matmul layout ``[x, y, z, v, o2, o2]`` (a view, no copy) for use with
    :func:`_apply_gchi0_pp`. ``chi0_mat`` is in the einsum layout ``[x, y, z, a, b, c, d, v]``.

    :param chi0_mat: The bubble array :math:`\chi_0^{pp}`, shape ``[x, y, z, o, o, o, o, v]``.
    :return: A view reshaped/transposed to ``[x, y, z, v, o2, o2]`` (rows ``(a, b)``, columns ``(c, d)``).
    """
    nqx, nqy, nqz, nb = chi0_mat.shape[:4]
    v = chi0_mat.shape[-1]
    return np.moveaxis(chi0_mat.reshape(nqx, nqy, nqz, nb * nb, nb * nb, v), -1, 3)


def _orient_cluster_by_mirrors(block: np.ndarray, gap_shape: tuple, tol: float = 1e-6) -> np.ndarray | None:
    r"""
    Rotates an orthonormal degenerate cluster onto the common eigenbasis of the single-axis coordinate mirrors
    :math:`k_i \to -k_i` and orders the partners lexicographically by the tuple of axes each one is odd under. The
    mirrors mutually commute, so one generic real combination of their projections into the cluster (weights
    :math:`1, \sqrt{2}, \sqrt{3}`, giving a distinct eigenvalue per sign pattern) shares their eigenvectors; each
    partner's mirror eigenvalues are read off as Rayleigh quotients. Single-axis :math:`p`-like partners sort
    ``x, y, z`` and two-axis :math:`d`-like partners (:math:`d_{xy}`, :math:`d_{xz}`, :math:`d_{yz}`) sort
    ``xy, xz, yz``.

    Two partners sharing a sign pattern span a subspace the mirrors do not resolve: the combined matrix is
    degenerate there, so the eigenvectors within it are fixed by floating-point noise alone and the sort cannot
    separate them either. This is the generic situation for a momentum-independent (purely local) cluster, where
    every partner is even under every mirror and the combined matrix is a multiple of the identity. Rotating by
    those eigenvectors would scramble the cluster, so such a cluster is rejected and the caller keeps its input
    basis.

    :param block: Orthonormal cluster, one flattened gap function per column.
    :param gap_shape: Full gap shape ``[kx, ky, kz, o1, o2, 2*niv_pp]``.
    :param tol: Tolerance on the deviation of a mirror Rayleigh quotient from :math:`\pm 1`.
    :return: The reordered cluster, or ``None`` when no momentum axis is resolved, a partner is not a clean
        :math:`\pm 1` mirror eigenstate, or two partners share the same mirror sign pattern.
    """

    def reflect_axis(column: np.ndarray, axis: int) -> np.ndarray:
        idx = (gap_shape[axis] - np.arange(gap_shape[axis])) % gap_shape[axis]
        take = [slice(None)] * len(gap_shape)
        take[axis] = idx
        return column.reshape(gap_shape)[tuple(take)].ravel()

    axes = [axis for axis in (0, 1, 2) if gap_shape[axis] > 1]
    if not axes:
        return None

    projected = []
    for axis in axes:
        mirrored = np.stack([reflect_axis(block[:, i], axis) for i in range(block.shape[1])], axis=1)
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
    lambdas: np.ndarray, gaps: np.ndarray, gap_shape: tuple, tol: float = 1e-4
) -> np.ndarray:
    r"""
    Orthonormalizes the eigenvectors returned by the Lanczos solver within clusters of (near-)degenerate
    eigenvalues and rotates every cluster to a mirror-adapted basis. The pairing kernel is only symmetrizable, not
    Hermitian in the plain inner product, so ARPACK may return oblique (mutually non-orthogonal) combinations
    inside a degenerate cluster: the cluster subspace is symmetry-covariant, but the returned vectors then do not
    form the symmetry-adapted partners.

    Per cluster the following steps are applied: (i) Loewdin orthonormalization, i.e. :math:`S^{-1/2}` applied to
    the cluster overlap matrix :math:`S`, which yields the orthonormal basis closest to the input vectors; (ii)
    for doublets, the mirror operation

    .. math:: M_y: \Delta_{12}(k_x, k_y, k_z, \nu) \to \Delta_{12}(k_x, -k_y, k_z, \nu)

    is diagonalized within the cluster, ordering the even (:math:`+1`, :math:`p_x`-like) partner first and the
    odd (:math:`-1`, :math:`p_y`-like) partner second; every cluster of three or more members is handled by
    :func:`_orient_cluster_by_mirrors`, which diagonalizes the single-axis coordinate mirrors of the resolved axes
    simultaneously and orders the partners by the axes each one is odd under (:math:`p`-like as ``x, y, z``,
    two-axis :math:`d`-like as ``xy, xz, yz``), provided every partner is a clean :math:`\pm 1` eigenstate and no
    two partners share a sign pattern (otherwise the mirrors do not resolve the cluster - a momentum-independent
    cluster is the generic such case - and the Loewdin basis is kept); (iii) the global phase of every vector is
    fixed such that its largest-magnitude element is real and positive. Eigenvalues are not modified; vectors of
    non-degenerate eigenvalues are only phase-fixed. Enabled via ``symmetrize_degenerate_gaps`` of
    :class:`~dgamore.config.EliashbergConfig`.

    :param lambdas: Eigenvalues sorted in descending order.
    :param gaps: Eigenvector matrix ``[n, n_eig]`` with one flattened gap function per column.
    :param gap_shape: Full gap shape ``[kx, ky, kz, o1, o2, 2*niv_pp]``, used to locate the momentum axes.
    :param tol: Relative tolerance for clustering neighboring eigenvalues as degenerate.
    :return: The symmetrized eigenvector matrix ``[n, n_eig]``.
    """
    n_ky = gap_shape[1]
    idx_neg = (n_ky - np.arange(n_ky)) % n_ky

    def mirror_y(column: np.ndarray) -> np.ndarray:
        return column.reshape(gap_shape)[:, idx_neg].ravel()

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

        if len(cluster) > 1:
            overlap = block.conj().T @ block
            eigs, u = np.linalg.eigh(overlap)
            if eigs.min() < 1e-12:  # (nearly) linearly dependent vectors cannot be orthonormalized meaningfully
                continue
            block = block @ (u @ np.diag(eigs**-0.5) @ u.conj().T)

        if len(cluster) == 2:
            mirrored = np.stack([mirror_y(block[:, i]) for i in range(2)], axis=1)
            mirror_block = block.conj().T @ mirrored
            mirror_block = 0.5 * (mirror_block + mirror_block.conj().T)
            _, mirror_vecs = np.linalg.eigh(mirror_block)
            # order the even (+1, p_x-like) partner first and the odd (-1, p_y-like) partner second
            block = block @ mirror_vecs[:, ::-1]
        elif len(cluster) >= 3:
            oriented = _orient_cluster_by_mirrors(block, gap_shape)
            if oriented is not None:
                block = oriented

        for col in range(block.shape[1]):
            mags = np.abs(block[:, col])
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
    :param base_seed: The flattened initial gap seed, identical on every rank (broadcast beforehand for the
        frequency-distributed solve); projected into each sector, with a deterministic random fallback when the
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
                gaps = symmetrize_degenerate_gaps(lambdas, gaps, gap_shape)
            logger.info(
                f"Largest eigenvalue{plural} for {label}: " + ", ".join(f"{lam:.6f}" for lam in lambdas),
                allowed_ranks=ranks,
            )
            gap_list = [GapFunction(gaps[:, i].reshape(gap_shape), channel, nq) for i in range(n_eig)]
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

    :param gamma_r_pp: The pairing vertex :math:`\Gamma^{pp}_{r}` (irreducible BZ, pp notation) for one channel;
        consumed by the solve.
    :param gchi0_q0_pp: The bare pp bubble :math:`\chi_0^{pp}` at :math:`\omega = 0`.
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
        Applies the pairing kernel to a flattened gap vector (the matrix-vector product for the eigensolver):
        multiplies by :math:`\chi_0^{pp}`, FFTs to real space, contracts with the pairing vertex (direct plus the
        crossed term, the latter reusing the direct vertex via gap-sized index shuffles), and transforms back. The
        orbital contractions are batched ``np.matmul`` products and the BZ transforms run in place through
        ``scipy.fft`` (both threaded up to the solver thread budget).

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
def solve_eliashberg_lanczos_v2(
    gamma_r_pp: FourPoint,
    gchi0_q0_pp: FourPoint,
    mpi_dist_v: MpiDistributor,
    active_ranks: list,
    n_threads: int = 1,
) -> dict[str, tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Solves the linearized Eliashberg equation for the leading superconducting eigenvalue(s) and gap function(s) using
    an ARPACK/Lanczos eigensolver. This variant distributes the gap function along the fermionic frequency axis across
    ranks (and performs the :math:`\chi_0^{pp}` multiplication only on the root rank), so it is more memory-efficient
    but slower than :func:`solve_eliashberg_lanczos`. The passed pairing vertex is **consumed** (Fourier transformed
    in place, then freed once its matmul-layout copy is built).

    The physical frequency-parity sectors of ``config.eliashberg.resolve_frequency_parity`` are handled exactly as in
    :func:`solve_eliashberg_lanczos`: the sector projector :math:`\Pi` (see :func:`_project_gap_to_sector`) wraps the
    matvec and the starting vector on the full (undistributed) gap vector the eigensolver sees, so the frequency
    distribution is transparent to the projection.

    :param gamma_r_pp: The pairing vertex :math:`\Gamma^{pp}_{r}` (frequency-distributed) for one channel; consumed
        by the solve.
    :param gchi0_q0_pp: The bare pp bubble :math:`\chi_0^{pp}` at :math:`\omega = 0` (held on the root rank).
    :param mpi_dist_v: MPI distributor over the fermionic frequency axis (see :class:`MpiDistributor`). Its
        communicator must span exactly the participating ranks - when some ranks hold empty frequency slices the
        caller passes the restricted distributor (see :meth:`~dgamore.mpi_utils.MpiDistributor.restricted_to`),
        whose rank 0 is the first active rank; ranks outside it must not enter this function.
    :param active_ranks: The original-communicator ranks participating in this solve (used for logging); the
        first is the root that owns the bubble.
    :param n_threads: This rank's momentum-batch/FFT thread budget (see :func:`_v2_solver_thread_budget`); the
        default 1 runs the serial path, results are bit-equal either way.
    :return: A dict ``{parity_label: (lambdas, gaps)}`` of the leading eigenvalues and :class:`GapFunction` objects
        per physical frequency-parity sector (a single ``"none"`` key when no projection is requested). The sector
        projectors act on the full (undistributed) gap vector the eigensolver sees, so the frequency distribution is
        transparent to them.
    """
    logger = config.logger
    root = active_ranks[0]
    # distributor operations are rooted at rank 0 of ITS communicator: the first active rank, whether the
    # distributor spans all ranks (all active) or the Split sub-communicator (keyed on the original rank)
    dist_root = 0

    logger.info(
        f"Starting to solve the Eliashberg equation for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=root,
    )
    logger.log_memory_usage(f"Gamma_pp_{gamma_r_pp.channel.value}", gamma_r_pp, len(active_ranks), allowed_ranks=root)

    gamma_r_pp = gamma_r_pp.fft(False).decompress_q_dimension()

    gap_shape = gamma_r_pp.nq + 2 * (gamma_r_pp.n_bands,) + (gamma_r_pp.current_shape[-1],)

    gap0 = get_initial_gap_function(gap_shape, gamma_r_pp.channel)
    gap0 = mpi_dist_v.bcast_chunked(gap0, root=dist_root)

    symmetry_label = config.eliashberg.symmetry.lower() if config.eliashberg.symmetry else "random"
    logger.info(
        f"Initialized the gap function as {symmetry_label} for {_sector_log_label(gamma_r_pp.channel)}.",
        allowed_ranks=root,
    )

    n_bands = gamma_r_pp.n_bands
    norm = 0.5 / config.lattice.k_grid.nk_tot / config.sys.beta
    # every active rank computes simultaneously here, so the budget divides this rank's affinity mask among the
    # node's active ranks (see _v2_solver_thread_budget); 1 under one-core-per-rank binding = serial path
    executor = ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None

    # Batched-matmul layout (see solve_eliashberg_lanczos): chi0 on the root rank (a view), the frequency-sliced
    # pairing vertex materialized once per rank; the flipped vertex is never stored (matvec reuses the direct array).
    chi0_mm = _chi0_to_matmul_layout(gchi0_q0_pp.mat) if mpi_dist_v.comm.rank == dist_root else None
    # w2dynamics G2 leg order (c cdag c cdag) vs the TRIQS order (cdag c cdag c) that _apply_gamma_pp expects: they
    # differ by the "abcd->badc" swap (o1<->o2, o3<->o4), which aligns the gap's creation legs; no-op for one band.
    gamma_mm = _gamma_to_matmul_layout(gamma_r_pp.permute_orbitals("abcd->badc", False).mat)
    gamma_r_pp.free()
    # fold the kernel prefactor into the persistent vertex once (see solve_eliashberg_lanczos): both matvec terms
    # inherit it by linearity, dropping the per-matvec full-gap multiply.
    gamma_mm *= norm

    sign = 1 if gamma_r_pp.channel == SpinChannel.SING else -1

    def mv(gap: np.ndarray):
        r"""
        Applies the pairing kernel to a flattened gap vector in the frequency-distributed scheme: the root rank
        multiplies by :math:`\chi_0^{pp}` and broadcasts, all ranks FFT and contract with their frequency slice of the
        pairing vertex, then the result is reassembled across the frequency axis via all-gather. The orbital
        contractions and the BZ transforms are threaded up to this rank's budget (serial for a budget of 1).

        :param gap: The flattened gap vector (full on root, sliced elsewhere).
        :return: The flattened result of applying the pairing kernel to ``gap``.
        """
        # 1. multiply chi0 * gap for the full BZ (only done by one rank, since memory would be an issue)
        gap_gg = (
            _apply_gchi0_pp(chi0_mm, gap, n_bands, executor, n_threads) if mpi_dist_v.comm.rank == dist_root else None
        )
        gap_gg = mpi_dist_v.bcast_chunked(gap_gg, root=dist_root)
        # 2. perform Fourier transform for the full chi0 * gap quantity
        gap_gg = sp.fft.fftn(gap_gg, axes=(0, 1, 2), overwrite_x=True, workers=n_threads)
        # 3. contract with the pairing vertex over (c, d, p) -> the gap for this rank's v slice; the crossed term
        #    reuses the direct vertex: Gamma_flip[K] @ gap_flip[K] == sign * flip_K[swap_ab[Gamma @ flip_p(gap_gg)]]
        gap_new = _apply_gamma_pp(gamma_mm, gap_gg, n_bands, executor, n_threads)
        crossed = _apply_gamma_pp(
            gamma_mm, np.ascontiguousarray(np.flip(gap_gg, axis=-1)), n_bands, executor, n_threads
        )
        crossed = np.roll(np.flip(crossed.swapaxes(3, 4), axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
        if sign != 1:
            crossed *= sign
        gap_new += crossed
        # 4. perform fourier transform on the local v slice
        gap_new = sp.fft.ifftn(gap_new, axes=(0, 1, 2), overwrite_x=True, workers=n_threads)
        # 5. assemble gap_new for the full v range through mpi_dist_v and allgather (remember we distributed v)
        gap_new = np.moveaxis(gap_new, -1, 0)  # (v_local, nq_tot, orb, orb)
        gap_new = mpi_dist_v.allgather(gap_new)  # (v_total, nq_tot, orb, orb)
        return np.moveaxis(gap_new, 0, -1).flatten()  # (nq_tot, orb, orb, v_total)

    return _solve_pairing_sectors(
        mv, gap_shape, sign, gamma_r_pp.channel, gamma_r_pp.nq, executor, (root,), gap0.flatten()
    )


def dispatch_full_vertex_calculation(
    channel: SpinChannel, u_loc: LocalInteraction, v_nonloc: Interaction, niv_pp: int, mpi_dist: MpiDistributor
) -> FourPoint:
    r"""
    Loads the local irreducible vertex for ``channel`` and builds the full ladder pp vertex, dispatching between
    the memory-lean and the regular construction routine based on the memory configuration. Please note that
    Eq. (4.43) in my master's thesis is wrong. The correct formula is
    :math:`F^{q\nu\nu'}_{r;1234}=F^{(1);q\nu\nu'}_{r;1234}+F^{(2);q\nu\nu'}_{r;1234}`, with
    :math:`F^{(1);q\nu\nu'}_{r;1234} = \beta^2\Big[(\chi_{0;1234}^{q\nu\nu'})^{-1}-
    \sum_{\nu_1\nu_2}\sum_{abcd}(\chi_{0;12ab}^{q\nu\nu_1})^{-1}\chi_{r;bacd}^{*;q\nu_1\nu_2}(\chi_{0;dc34}^{q\nu_2\nu'})^{-1}\Big]` and
    :math:`F^{(2);q\nu\nu'}_{r;1234} = \sum_{abcdgh}\gamma^{q\nu}_{r;12ab}\Big(\mathbb{1}_{bacd} -
    \sum_{ef}\mathcal{U}^{q}_{r;baef}\chi^{q}_{r;fecd}\Big)\mathcal{U}^{q}_{r;dcgh}\tilde\gamma^{q\nu'}_{r;hg34}`,
    where
    :math:`\tilde\gamma_{r;1234}^{q\nu}=\beta \sum_{ab}\sum_{\nu'} \chi^{*;q\nu'\nu}_{r;12ab} (\chi^{q\nu}_{0;ba34})^{-1}
    =\beta \sum_{ab}\sum_{\nu'} \chi^{*;q\nu\nu'}_{r;ab21} (\chi^{q\nu}_{0;ab34})^{-1}`, i.e. the sum over the first
    frequency argument equals the sum over the last one only up to the orbital reversal dictated by
    time-reversal symmetry, see :meth:`~dgamore.nonlocal_sde.create_vrg_r_q_right`. No explicit factors of
    :math:`\beta` appear in :math:`F^{(2)}` because they are absorbed into the stored objects:
    :math:`\chi^{q}_{r}` is the (:math:`U`-dressed, shell- (and sometimes :math:`\lambda`-corrected)) physical
    susceptibility normalized as :math:`\frac{1}{\beta^2}\sum_{\nu\nu'}\chi^{q\nu\nu'}_{r}`, and the three-leg
    vertices carry the net normalization :math:`\gamma^{q\nu}_{r} = (\chi^{q\nu}_{0})^{-1}\sum_{\nu'}
    \chi^{*;q\nu\nu'}_{r}` (the explicit :math:`\beta` in their construction cancels the :math:`1/\beta` of the
    fused frequency sum), such that :math:`\gamma^{q\nu}_{r} \to \mathbb{1}` for :math:`\nu \to \infty`.

    :param channel: The spin channel (density or magnetic).
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :param niv_pp: Number of positive fermionic frequencies of the pp vertex.
    :param mpi_dist: MPI distributor over the irreducible BZ q-points.
    :return: The full ladder pp vertex :math:`F^{q}_{r}` as a :class:`FourPoint`.
    """
    gamma_r = LocalFourPoint.load(os.path.join(config.output.output_path, f"gamma_{channel.value}_loc.npy"), channel)
    if config.memory.save_memory_for_fq:
        f_q_r = create_full_vertex_q_r_pp_w0_v2(u_loc, v_nonloc, gamma_r, niv_pp, mpi_dist)
    else:
        f_q_r = create_full_vertex_q_r_pp_w0(u_loc, v_nonloc, gamma_r, niv_pp, mpi_dist)
    gamma_r.free()
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
    for each channel and returns the leading eigenvalues and gap functions. Dispatches between the in-memory and the
    memory-lean Lanczos solvers depending on the memory configuration.

    :param giwk_dga: The converged momentum-dependent DGA :class:`GreensFunction`.
    :param g_dmft: The local (DMFT) :class:`GreensFunction` (used for the local diagrams).
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :param comm: The MPI communicator.
    :return: A dict keyed by ``(channel, parity_label)`` mapping to ``(lambdas, gaps)`` of the leading eigenvalues
        and :class:`GapFunction` objects for each solved physical frequency-parity sector. When
        ``config.eliashberg.resolve_frequency_parity`` is set the parity labels are ``"even"`` and ``"odd"``,
        otherwise a single unprojected ``"none"`` sector is returned.
    """
    logger = config.logger

    mpi_dist_irrk = MpiDistributor.create_distributor(
        ntasks=config.lattice.k_grid.nk_irr, comm=comm, name="Q", output_path=config.output.output_path
    )
    irrk_q_list = config.lattice.k_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    parities = [parity for parity, _ in _frequency_parity_sectors(config.eliashberg.resolve_frequency_parity)]
    niv_pp = min(config.box.niw_core // 2, config.box.niv_core // 2)

    # giwk_dga is consumed only by the pp-bubble build on a single rank (the in-memory bubble rank, or rank 0 of the
    # frequency-distributed variant), so every other rank drops its replicated full-grid copy for this phase.
    if config.memory.save_memory_for_lanczos:
        sing_ranks = trip_ranks = None
        bubble_rank = 0
    else:
        import psutil

        # per-node memory drives how many (channel, parity) sectors run concurrently: pack as many per node as its
        # free memory fits, so the sectors parallelize without overcommitting a node (the estimate mirrors the
        # memory_estimator lanczos peak - LANCZOS_VERTEX_FACTOR full-BZ pp vertices + the pp bubble + ARPACK basis)
        itemsize = np.dtype(DTYPE).itemsize
        n2 = 2 * niv_pp
        vertex_pp_full = config.lattice.k_grid.nk_tot * config.sys.n_bands**4 * n2**2 * itemsize
        chi0_pp_full = config.lattice.k_grid.nk_tot * config.sys.n_bands**4 * n2 * itemsize
        gap_bytes = config.lattice.k_grid.nk_tot * config.sys.n_bands**2 * n2 * itemsize
        per_sector_bytes = (
            LANCZOS_VERTEX_FACTOR * vertex_pp_full + chi0_pp_full + (2 * config.eliashberg.n_eig + 20) * gap_bytes
        )
        sing_ranks, trip_ranks = get_ranks_for_lanczos(
            comm, len(parities), psutil.virtual_memory().available, per_sector_bytes, giwk_dga.mat.nbytes
        )
        bubble_rank = sing_ranks[0]
    if comm.rank != bubble_rank:
        giwk_dga.free()

    f_dens_pp = dispatch_full_vertex_calculation(SpinChannel.DENS, u_loc, v_nonloc, niv_pp, mpi_dist_irrk)
    f_magn_pp = dispatch_full_vertex_calculation(SpinChannel.MAGN, u_loc, v_nonloc, niv_pp, mpi_dist_irrk)

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

    if config.eliashberg.include_local_part:
        # the local diagrams are reduced from local vertices on the full asymptotic fermionic box; one rank per node
        # reads and reduces them and broadcasts the pp-box-sized results, so that transient exists once per node
        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) if comm.size > 1 else None
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

    def empty_sector_gaps() -> list[GapFunction]:
        return [GapFunction(np.empty(0)) for _ in range(config.eliashberg.n_eig)]

    results: dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]] = {}

    if not config.memory.save_memory_for_lanczos:
        results = _solve_sectors_in_memory(
            mpi_dist_irrk, gamma_sing_pp, gamma_trip_pp, giwk_dga, niv_pp, sing_ranks, trip_ranks, bubble_rank, parities
        )
    else:
        mpi_dist_v = MpiDistributor.create_distributor(
            ntasks=gamma_sing_pp.current_shape[-2], comm=comm, name="V", output_path=config.output.output_path
        )

        logger.info("Distributing Gamma_sing_pp along v equally to ranks/nodes.")
        gamma_sing_pp = mpi_utils.gather_full_ibz_for_vslice(
            gamma_sing_pp, mpi_dist_irrk, mpi_dist_v, config.lattice.k_grid
        )
        logger.info("Gamma_sing_pp distributed. Starting with singlet Lanczos solver.")

        active_ranks = [
            r
            for r in range(comm.size)
            if mpi_dist_v.slices[r] is not None and (mpi_dist_v.slices[r].stop - mpi_dist_v.slices[r].start) > 0
        ]

        root = active_ranks[0]
        # collective on the full comm (inactive ranks included): splits each active rank's affinity mask among the
        # active ranks of its node, so the matvec threading never oversubscribes shared cores
        v2_n_threads = _v2_solver_thread_budget(comm, active_ranks)
        # the matvec collectives must span exactly the ranks that enter the solve: with empty frequency slices
        # present, the active ranks run on a Split sub-communicator carrying only their slices
        if len(active_ranks) < comm.size:
            active_comm = comm.Split(0 if comm.rank in active_ranks else 1, comm.rank)
            solver_dist_v = mpi_dist_v.restricted_to(active_comm, active_ranks) if comm.rank in active_ranks else None
        else:
            solver_dist_v = mpi_dist_v
        if comm.rank == root:
            # calculating gchi0 in the full BZ only once
            # in the lanczos step only active_rank[0] will perform chi0 * delta due to memory reasons
            gchi0_q_pp = BubbleGenerator.create_generalized_chi0_q_pp_w0(
                giwk_dga, niv_pp, config.lattice.k_grid
            ).decompress_q_dimension()
        else:
            gchi0_q_pp = None

        sectors_sing = sectors_trip = None
        if comm.rank in active_ranks:
            sectors_sing = solve_eliashberg_lanczos_v2(
                gamma_sing_pp, gchi0_q_pp, solver_dist_v, active_ranks, v2_n_threads
            )
            gamma_sing_pp.free()

        logger.info("Distributing Gamma_trip_pp along v equally to ranks/nodes.")
        gamma_trip_pp = mpi_utils.gather_full_ibz_for_vslice(
            gamma_trip_pp, mpi_dist_irrk, mpi_dist_v, config.lattice.k_grid
        )
        logger.info("Gamma_trip_pp distributed. Starting with triplet Lanczos solver.")

        if comm.rank in active_ranks:
            sectors_trip = solve_eliashberg_lanczos_v2(
                gamma_trip_pp, gchi0_q_pp, solver_dist_v, active_ranks, v2_n_threads
            )
            gamma_trip_pp.free()

        for channel, sectors in ((SpinChannel.SING, sectors_sing), (SpinChannel.TRIP, sectors_trip)):
            for parity in parities:
                local = sectors[parity] if sectors is not None else None
                lambdas = comm.bcast(local[0] if local is not None else None, root=root)
                gaps = local[1] if local is not None else empty_sector_gaps()
                gaps = [mpi_dist_irrk.bcast_npoint(gap, root=root) for gap in gaps]
                results[(channel, parity)] = (lambdas, gaps)

    return results


# Fraction of a node's available host memory the sector packing may occupy (mirrors DGAmore.NODE_MEMORY_FRACTION).
NODE_MEMORY_FRACTION: float = 0.97


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

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
:math:`\Delta(k, \nu)`. Requires ``nq == nk``. Equation numbers refer to the author's master's thesis (Chapter 4).
"""

import os

import mpi4py.MPI as MPI
import numpy as np
import scipy as sp

import dgamore.config as config
from dgamore import nonlocal_sde, mpi_utils
from dgamore.bubble_gen import BubbleGenerator
from dgamore.four_point import FourPoint
from dgamore.gap_function import GapFunction
from dgamore.greens_function import GreensFunction
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.local_four_point import LocalFourPoint
from dgamore.matsubara_frequencies import MFHelper
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
    # only the |w| <= 2*niv_pp - 1 anti-diagonals (omega = v - v') are read below, so the bosonic axis is trimmed
    # to that window before to_full_niw_range doubles it (cut_niw cannot be used here - its no-op guard misjudges
    # half-range objects)
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
    return FourPoint(mat, f_q_r.channel, config.lattice.q_grid.nk, 0, 2, True, True, True, FrequencyNotation.PP)


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

    if config.eliashberg.construct_fq_cheap:
        gamma_r = gamma_r.cut_niv(niv_pp)
        gchi0_q_inv = gchi0_q_inv.cut_niv(niv_pp)

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

    if config.eliashberg.construct_fq_cheap:
        vrg_q_r_left = vrg_q_r_left.cut_niv(niv_pp)

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

    if config.eliashberg.construct_fq_cheap:
        vrg_q_r_right = vrg_q_r_right.cut_niv(niv_pp)

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

    if config.eliashberg.construct_fq_cheap:
        gamma_r = gamma_r.cut_niv(niv_pp)
        gchi0_q_inv = gchi0_q_inv.cut_niv(niv_pp)
        vrg_q_r_left = vrg_q_r_left.cut_niv(niv_pp)
        vrg_q_r_right = vrg_q_r_right.cut_niv(niv_pp)

    logger.info(f"Loaded vrg_q_{gamma_r.channel.value} and chi_phys_q_{gamma_r.channel.value} from files.")

    irrk_q_list = config.lattice.q_grid.get_irrq_list()
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
            f_q_r_mat, gamma_r.channel, config.lattice.q_grid.nk, 0, 2, False, True, True, FrequencyNotation.PP
        )
        logger.log_memory_usage(
            f"Full ladder-vertex ({f_q_r.channel.value})",
            f_q_r,
            mpi_dist_irrk.comm.size * 4 * (config.box.niw_core + 1),
        )
        return f_q_r

    f_q_r = FourPoint(
        f_q_r_mat, gamma_r.channel, config.lattice.q_grid.nk, 1, 2, False, True, True, FrequencyNotation.PP
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
    :param beta: Inverse temperature :math:`\beta`, see :attr:`~dgamore.config.SystemConfig.beta`.
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
    subtracted/added when :attr:`~dgamore.config.EliashbergConfig.include_local_part` is enabled, to avoid double
    counting the local pairing contribution (thesis Eqs. 4.49-4.52).

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


# --- Gap initialisation ---
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


# --- Eliashberg eigensolver (Lanczos / ARPACK) ---
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


def symmetrize_degenerate_gaps(
    lambdas: np.ndarray, gaps: np.ndarray, gap_shape: tuple, tol: float = 1e-4
) -> np.ndarray:
    r"""
    Orthonormalizes the eigenvectors returned by the Lanczos solver within clusters of (near-)degenerate
    eigenvalues and rotates every two-dimensional cluster to the mirror-adapted basis. The pairing kernel is only
    symmetrizable, not Hermitian in the plain inner product, so ARPACK may return oblique (mutually
    non-orthogonal) combinations inside a degenerate doublet: the doublet subspace is symmetry-covariant, but the
    two returned vectors then do not form 90-degree-rotated partners.

    Per cluster the following steps are applied: (i) Loewdin orthonormalization, i.e. :math:`S^{-1/2}` applied to
    the cluster overlap matrix :math:`S`, which yields the orthonormal basis closest to the input vectors; (ii)
    for doublets, the mirror operation

    .. math:: M_y: \Delta_{12}(k_x, k_y, k_z, \nu) \to \Delta_{12}(k_x, -k_y, k_z, \nu)

    is diagonalized within the cluster, ordering the even (:math:`+1`, :math:`p_x`-like) partner first and the
    odd (:math:`-1`, :math:`p_y`-like) partner second; (iii) the global phase of every vector is fixed such that
    its largest-magnitude element is real and positive. Eigenvalues are not modified; vectors of non-degenerate
    eigenvalues are only phase-fixed. Enabled via
    :attr:`~dgamore.config.EliashbergConfig.symmetrize_degenerate_gaps`.

    :param lambdas: Eigenvalues sorted in descending order.
    :param gaps: Eigenvector matrix ``[n, n_eig]`` with one flattened gap function per column.
    :param gap_shape: Full gap shape ``[kx, ky, kz, o1, o2, 2*niv_pp]``, used to locate the :math:`k_y` axis.
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

        for col in range(block.shape[1]):
            mags = np.abs(block[:, col])
            # tie-break on the first index among the maximal-modulus elements, stable against fp noise
            phase = block[np.flatnonzero(mags >= mags.max() * (1.0 - 1e-8))[0], col]
            block[:, col] *= phase.conjugate() / abs(phase)
        gaps[:, cluster] = block

    return gaps


def _apply_gchi0_pp(chi0_mm: np.ndarray, gap: np.ndarray, n_bands: int) -> np.ndarray:
    r"""
    Batched-matmul equivalent of ``np.einsum("xyzabcdv,xyzcdv->xyzabv", chi0, gap)`` (multiply the gap by the bare pp
    bubble per momentum and frequency). ``np.matmul`` is both faster than ``np.einsum`` and far leaner here: einsum
    materializes a vertex-sized internal temporary, the matmul allocates only the gap-sized output.

    :param chi0_mm: The bubble in matmul layout from :func:`_chi0_to_matmul_layout`, shape ``[x, y, z, v, o2, o2]``.
    :param gap: The gap vector, reshapeable to ``[x, y, z, o, o, v]``.
    :param n_bands: Number of orbitals ``o``.
    :return: ``chi0 @ gap`` in shape ``[x, y, z, o, o, v]``.
    """
    nqx, nqy, nqz, v = chi0_mm.shape[0], chi0_mm.shape[1], chi0_mm.shape[2], chi0_mm.shape[3]
    oo = n_bands * n_bands
    gap_r = np.moveaxis(gap.reshape(nqx, nqy, nqz, oo, v), -1, 3)[..., None]  # [x, y, z, v, o2, 1]
    out = np.matmul(chi0_mm, gap_r)[..., 0]  # [x, y, z, v, o2]
    return np.moveaxis(out, 3, -1).reshape(nqx, nqy, nqz, n_bands, n_bands, v)


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


def _apply_gamma_pp(gamma_mm: np.ndarray, gap_gg: np.ndarray, n_bands: int) -> np.ndarray:
    r"""
    Batched-matmul equivalent of ``np.einsum("xyzacbdvp,xyzcdp->xyzabv", gamma, gap_gg)`` (contract the pairing vertex
    with the gap over ``(c, d, p)``). Faster and leaner than ``np.einsum`` (see :func:`_apply_gchi0_pp`).

    :param gamma_mm: The vertex in matmul layout from :func:`_gamma_to_matmul_layout`, shape ``[x, y, z, o2*nv, o2*np]``.
    :param gap_gg: The transformed gap, shape ``[x, y, z, c, d, p]``.
    :param n_bands: Number of orbitals ``o``.
    :return: ``gamma @ gap_gg`` in shape ``[x, y, z, o, o, nv]``.
    """
    nqx, nqy, nqz = gamma_mm.shape[:3]
    oo = n_bands * n_bands
    npp = gap_gg.shape[-1]
    nv = gamma_mm.shape[3] // oo
    gg_r = gap_gg.reshape(nqx, nqy, nqz, oo * npp)[..., None]  # [x, y, z, o2*np, 1]
    out = np.matmul(gamma_mm, gg_r)[..., 0]  # [x, y, z, o2*nv]
    return out.reshape(nqx, nqy, nqz, n_bands, n_bands, nv)


def solve_eliashberg_lanczos(
    gamma_r_pp: FourPoint, gchi0_q0_pp: FourPoint, ranks: tuple[int, int]
) -> tuple[list[float], list[GapFunction]]:
    r"""
    Solves the linearized Eliashberg equation for the leading superconducting eigenvalue(s) and gap function(s) using
    an ARPACK/Lanczos eigensolver, with the pairing kernel applied matrix-free via FFTs over the BZ. This in-memory
    variant holds the full-BZ pairing vertex on the solving rank. The passed pairing vertex is **consumed** (mapped
    to the full BZ, Fourier transformed and sign-folded in place, then freed once its matmul-layout copy is built).

    :param gamma_r_pp: The pairing vertex :math:`\Gamma^{pp}_{r}` (irreducible BZ, pp notation) for one channel;
        consumed by the solve.
    :param gchi0_q0_pp: The bare pp bubble :math:`\chi_0^{pp}` at :math:`\omega = 0`.
    :param ranks: The ``(rank_sing, rank_trip)`` pair used for logging.
    :return: A tuple ``(lambdas, gaps)`` of the leading eigenvalues and the corresponding :class:`GapFunction` objects.
    """
    logger = config.logger

    logger.info(
        f"Starting to solve the Eliashberg equation for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=ranks,
    )

    gamma_r_pp = gamma_r_pp.map_to_full_bz(config.lattice.q_grid, config.lattice.q_grid.nk).decompress_q_dimension()
    logger.log_memory_usage(f"Gamma_pp_{gamma_r_pp.channel.value}", gamma_r_pp, 1, allowed_ranks=ranks)

    sign = 1 if gamma_r_pp.channel == SpinChannel.SING else -1

    # in-place sign fold: the former ``sign * fft`` deep-copied the full-BZ vertex and left the pre-copy original
    # alive (still referenced by the caller) for the entire solve
    gamma_r_pp = gamma_r_pp.fft(False)
    if sign != 1:
        gamma_r_pp.scale(sign)

    gap_shape = gamma_r_pp.nq + 2 * (gamma_r_pp.n_bands,) + (2 * gamma_r_pp.niv,)
    gchi0_q0_pp = gchi0_q0_pp.decompress_q_dimension()

    gap0 = get_initial_gap_function(gap_shape, gamma_r_pp.channel)
    symmetry_label = config.eliashberg.symmetry.lower() if config.eliashberg.symmetry else "random"
    logger.info(
        f"Initialized the gap function as {symmetry_label} for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=ranks,
    )

    n_bands = gamma_r_pp.n_bands
    norm = 0.5 / config.lattice.q_grid.nk_tot / config.sys.beta

    chi0_mm = _chi0_to_matmul_layout(gchi0_q0_pp.mat)
    # The pairing vertex arrives in the w2dynamics G2 leg order (c cdag c cdag), whereas the contraction in
    # _apply_gamma_pp expects the TRIQS order (cdag c cdag c), see
    # https://triqs.github.io/tprf/latest/theory/eliashberg.html
    gamma_mm = _gamma_to_matmul_layout(gamma_r_pp.permute_orbitals("abcd->badc", False).mat)
    gamma_r_pp.free()

    def mv(gap: np.ndarray):
        r"""
        Applies the pairing kernel to a flattened gap vector (the matrix-vector product for the eigensolver):
        multiplies by :math:`\chi_0^{pp}`, FFTs to real space, contracts with the pairing vertex (direct plus the
        crossed term, the latter reusing the direct vertex via gap-sized index shuffles), and transforms back. The
        orbital contractions are batched ``np.matmul`` products and the BZ transforms run in place through
        ``scipy.fft``.

        :param gap: The flattened gap vector.
        :return: The flattened result of applying the pairing kernel to ``gap``.
        """
        gap_gg = sp.fft.fftn(_apply_gchi0_pp(chi0_mm, gap, n_bands), axes=(0, 1, 2), overwrite_x=True)
        gap_new = _apply_gamma_pp(gamma_mm, gap_gg, n_bands)
        # crossed term: Gamma_flip[K] @ gap_flip[K] == sign * flip_K[swap_ab[Gamma @ flip_p(gap_gg)]]
        crossed = _apply_gamma_pp(gamma_mm, np.flip(gap_gg, axis=-1), n_bands)
        crossed = np.roll(np.flip(crossed.swapaxes(3, 4), axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
        if sign != 1:
            crossed *= sign
        gap_new += crossed
        gap_new = sp.fft.ifftn(gap_new, axes=(0, 1, 2), overwrite_x=True)
        gap_new *= norm
        return gap_new.flatten()

    mat = sp.sparse.linalg.LinearOperator(shape=(np.prod(gap_shape), np.prod(gap_shape)), matvec=mv)

    n_eig = config.eliashberg.n_eig
    eig_label = "" if n_eig > 1 else f" {n_eig}"
    plural = "" if n_eig == 1 else "s"
    logger.info(
        f"Starting Lanczos method to retrieve largest{eig_label} eigenvalue{plural} and eigenvector{plural} "
        f"for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=ranks,
    )

    lambdas, gaps = sp.sparse.linalg.eigsh(
        mat, k=n_eig, tol=config.eliashberg.epsilon, v0=gap0.flatten(), which="LA", maxiter=10000
    )

    logger.info(
        f"Finished Lanczos method for the largest{eig_label} eigenvalue{plural} and eigenvector{plural} "
        f"for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=ranks,
    )

    order = lambdas.argsort()[::-1]  # sort eigenvalues in descending order
    lambdas = lambdas[order]
    gaps = gaps[:, order]

    if config.eliashberg.symmetrize_degenerate_gaps:
        gaps = symmetrize_degenerate_gaps(lambdas, gaps, gap_shape)

    logger.info(
        f"Largest{eig_label} eigenvalue{plural} for the {gamma_r_pp.channel.value}let "
        f"channel {"is" if n_eig == 1 else "are"}: " + ", ".join(f"{lam:.6f}" for lam in lambdas),
        allowed_ranks=ranks,
    )

    gaps = [
        GapFunction(gaps[..., i].reshape(gap_shape), gamma_r_pp.channel, gamma_r_pp.nq)
        for i in range(config.eliashberg.n_eig)
    ]

    logger.info(
        f"Finished solving the Eliashberg equation for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=ranks,
    )

    return lambdas, gaps


# --- Eliashberg eigensolver (Lanczos / ARPACK) ---
def solve_eliashberg_lanczos_v2(
    gamma_r_pp: FourPoint, gchi0_q0_pp: FourPoint, mpi_dist_v: MpiDistributor, active_ranks: list
) -> tuple[list[float], list[GapFunction]]:
    r"""
    Solves the linearized Eliashberg equation for the leading superconducting eigenvalue(s) and gap function(s) using
    an ARPACK/Lanczos eigensolver. This variant distributes the gap function along the fermionic frequency axis across
    ranks (and performs the :math:`\chi_0^{pp}` multiplication only on the root rank), so it is more memory-efficient
    but slower than :func:`solve_eliashberg_lanczos`. The passed pairing vertex is **consumed** (Fourier transformed
    and sign-folded in place, then freed once its matmul-layout copy is built).

    :param gamma_r_pp: The pairing vertex :math:`\Gamma^{pp}_{r}` (frequency-distributed) for one channel; consumed
        by the solve.
    :param gchi0_q0_pp: The bare pp bubble :math:`\chi_0^{pp}` at :math:`\omega = 0` (held on the root rank).
    :param mpi_dist_v: MPI distributor over the fermionic frequency axis (see :class:`MpiDistributor`).
    :param active_ranks: The ranks participating in this solve; the first is used as root.
    :return: A tuple ``(lambdas, gaps)`` of the leading eigenvalues and the corresponding :class:`GapFunction` objects.
    """
    logger = config.logger
    root = active_ranks[0]

    logger.info(
        f"Starting to solve the Eliashberg equation for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=root,
    )
    logger.log_memory_usage(f"Gamma_pp_{gamma_r_pp.channel.value}", gamma_r_pp, len(active_ranks), allowed_ranks=root)

    sign = 1 if gamma_r_pp.channel == SpinChannel.SING else -1

    # in-place sign fold (see solve_eliashberg_lanczos): avoids the deep copy and the orphaned pre-copy vertex
    gamma_r_pp = gamma_r_pp.fft(False).decompress_q_dimension()
    if sign != 1:
        gamma_r_pp.scale(sign)

    gap_shape = gamma_r_pp.nq + 2 * (gamma_r_pp.n_bands,) + (gamma_r_pp.current_shape[-1],)

    gap0 = get_initial_gap_function(gap_shape, gamma_r_pp.channel)
    gap0 = mpi_dist_v.bcast_chunked(gap0, root=root)

    symmetry_label = config.eliashberg.symmetry.lower() if config.eliashberg.symmetry else "random"
    logger.info(
        f"Initialized the gap function as {symmetry_label} for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=root,
    )

    n_bands = gamma_r_pp.n_bands
    norm = 0.5 / config.lattice.q_grid.nk_tot / config.sys.beta

    # Batched-matmul layout (see :func:`solve_eliashberg_lanczos`): chi0 on the root rank (a view), the
    # frequency-sliced pairing vertex materialized once per rank. The flipped vertex is never stored - its
    # contraction reuses the direct array via gap-sized index shuffles in the matvec (the crossing map touches only
    # the row orbitals, the column frequency and the momentum, so it holds per v-slice).
    chi0_mm = _chi0_to_matmul_layout(gchi0_q0_pp.mat) if mpi_dist_v.comm.rank == root else None
    # The pairing vertex arrives in the w2dynamics G2 leg order (c cdag c cdag), whereas the contraction in
    # _apply_gamma_pp expects the TRIQS order (cdag c cdag c). The two differ by the "abcd->badc" swap
    # (o1<->o2, o3<->o4); applying it makes the gap's creation legs contract with the vertex's creation legs.
    # This is a no-op for a single band.
    gamma_mm = _gamma_to_matmul_layout(gamma_r_pp.permute_orbitals("abcd->badc", False).mat)
    gamma_r_pp.free()

    def mv(gap: np.ndarray):
        r"""
        Applies the pairing kernel to a flattened gap vector in the frequency-distributed scheme: the root rank
        multiplies by :math:`\chi_0^{pp}` and broadcasts, all ranks FFT and contract with their frequency slice of the
        pairing vertex, then the result is reassembled across the frequency axis via all-gather.

        :param gap: The flattened gap vector (full on root, sliced elsewhere).
        :return: The flattened result of applying the pairing kernel to ``gap``.
        """
        # 1. multiply chi0 * gap for the full BZ (only done by one rank, since memory would be an issue)
        gap_gg = _apply_gchi0_pp(chi0_mm, gap, n_bands) if mpi_dist_v.comm.rank == root else None
        gap_gg = mpi_dist_v.bcast_chunked(gap_gg, root=root)
        # 2. perform Fourier transform for the full chi0 * gap quantity
        gap_gg = sp.fft.fftn(gap_gg, axes=(0, 1, 2), overwrite_x=True)
        # 3. contract with the pairing vertex over (c, d, p) -> the gap for this rank's v slice; the crossed term
        #    reuses the direct vertex: Gamma_flip[K] @ gap_flip[K] == sign * flip_K[swap_ab[Gamma @ flip_p(gap_gg)]]
        gap_new = _apply_gamma_pp(gamma_mm, gap_gg, n_bands)
        crossed = _apply_gamma_pp(gamma_mm, np.flip(gap_gg, axis=-1), n_bands)
        crossed = np.roll(np.flip(crossed.swapaxes(3, 4), axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
        if sign != 1:
            crossed *= sign
        gap_new += crossed
        # 4. perform fourier transform on the local v slice
        gap_new = sp.fft.ifftn(gap_new, axes=(0, 1, 2), overwrite_x=True)
        gap_new *= norm
        # 5. assemble gap_new for the full v range through mpi_dist_v and allgather (remember we distributed v)
        gap_new = np.moveaxis(gap_new, -1, 0)  # (v_local, nq_tot, orb, orb)
        gap_new = mpi_dist_v.allgather(gap_new)  # (v_total, nq_tot, orb, orb)
        return np.moveaxis(gap_new, 0, -1).flatten()  # (nq_tot, orb, orb, v_total)

    mat = sp.sparse.linalg.LinearOperator(shape=(np.prod(gap_shape), np.prod(gap_shape)), matvec=mv)

    n_eig = config.eliashberg.n_eig
    eig_label = "" if n_eig > 1 else f" {n_eig}"
    plural = "" if n_eig == 1 else "s"
    logger.info(
        f"Starting Lanczos method to retrieve largest{eig_label} eigenvalue{plural} and eigenvector{plural} "
        f"for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=root,
    )

    lambdas, gaps = sp.sparse.linalg.eigsh(
        mat, k=n_eig, tol=config.eliashberg.epsilon, v0=gap0.flatten(), which="LA", maxiter=10000
    )

    logger.info(
        f"Finished Lanczos method for the largest{eig_label} eigenvalue{plural} and eigenvector{plural} "
        f"for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=root,
    )

    order = lambdas.argsort()[::-1]  # sort eigenvalues in descending order
    lambdas = lambdas[order]
    gaps = gaps[:, order]

    if config.eliashberg.symmetrize_degenerate_gaps:
        gaps = symmetrize_degenerate_gaps(lambdas, gaps, gap_shape)

    logger.info(
        f"Largest{eig_label} eigenvalue{plural} for the {gamma_r_pp.channel.value}let "
        f"channel {"is" if n_eig == 1 else "are"}: " + ", ".join(f"{lam:.6f}" for lam in lambdas),
        allowed_ranks=root,
    )

    gaps = [
        GapFunction(gaps[..., i].reshape(gap_shape), gamma_r_pp.channel, gamma_r_pp.nq)
        for i in range(config.eliashberg.n_eig)
    ]

    logger.info(
        f"Finished solving the Eliashberg equation for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=root,
    )

    return lambdas, gaps


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
    :return: A tuple ``(lambdas_sing, lambdas_trip, gaps_sing, gaps_trip)`` of the singlet/triplet eigenvalues and
        :class:`GapFunction` lists.
    """
    logger = config.logger

    mpi_dist_irrk = MpiDistributor.create_distributor(
        ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q", output_path=config.output.output_path
    )
    irrk_q_list = config.lattice.q_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    # giwk_dga is consumed only by the pp-bubble build on a single rank (the singlet solver rank of the in-memory
    # variant, or rank 0 = the root of the frequency-distributed one), so every other rank drops its replicated
    # full-grid copy for the whole vertex-construction and solver phase.
    if config.memory.save_memory_for_lanczos:
        rank_sing = rank_trip = bubble_rank = 0
    else:
        rank_sing, rank_trip = get_ranks_for_lanczos(comm)
        if comm.size == 1:
            rank_trip = rank_sing
        bubble_rank = rank_sing
    if comm.rank != bubble_rank:
        giwk_dga.free()

    niv_pp = min(config.box.niw_core // 2, config.box.niv_core // 2)

    f_dens_pp = dispatch_full_vertex_calculation(SpinChannel.DENS, u_loc, v_nonloc, niv_pp, mpi_dist_irrk)
    f_magn_pp = dispatch_full_vertex_calculation(SpinChannel.MAGN, u_loc, v_nonloc, niv_pp, mpi_dist_irrk)

    delete_files(config.output.eliashberg_path, f"gchi0_q_inv_rank_{comm.rank}.npy")
    delete_files(config.output.output_path, f"gchi0_q_rank_{comm.rank}.npy")

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
        f_ud_loc_pp_w0, gamma_ud_loc_pp_w0, phi_ud_loc_pp_w0 = create_local_ud_diagrams_pp_w0(g_dmft, niv_pp)

        if mpi_dist_irrk.my_rank == 0:
            f_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="f_ud_loc_pp_w0")
            phi_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="phi_ud_loc_pp_w0")
            gamma_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="gamma_ud_loc_pp_w0")
            logger.info("Saved local ud diagrams in pp notation to file.")

        del f_ud_loc_pp_w0, gamma_ud_loc_pp_w0

        # special treatment of local full vertex that is subtracted with a different frequency notation and is
        # different from the regular pp
        f_dens_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"f_dens_loc.npy"), SpinChannel.DENS)
        f_magn_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"f_magn_loc.npy"), SpinChannel.MAGN)
        f_ud_loc = (f_dens_loc - f_magn_loc).set_channel(SpinChannel.UD).scale(0.5)
        f_ud_loc_transf_w0 = transform_vertex_loc_frequencies_w0(f_ud_loc, niv_pp)
        del f_dens_loc, f_magn_loc, f_ud_loc

        # Eqs. (4.49)-(4.52): the assembled vertex holds the negative crossed slot, so the local full vertex enters
        # with a relative minus (the transform already carries one minus) and the local pp-reducible diagrams phi
        # enter with a plus, both in crossed-slot form (frequencies (v, -v'), orbitals 1432).
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

    gaps_sing = [GapFunction(np.empty(0)) for _ in range(config.eliashberg.n_eig)]
    gaps_trip = [GapFunction(np.empty(0)) for _ in range(config.eliashberg.n_eig)]

    if not config.memory.save_memory_for_lanczos:
        gamma_sing_pp.mat = mpi_dist_irrk.gather(gamma_sing_pp.mat, root=rank_sing)
        gamma_trip_pp.mat = mpi_dist_irrk.gather(gamma_trip_pp.mat, root=rank_trip)

        if mpi_dist_irrk.my_rank == rank_sing:
            gchi0_q_pp = BubbleGenerator.create_generalized_chi0_q_pp_w0(giwk_dga, niv_pp, config.lattice.q_grid)
            logger.info("Created the bare bubble susceptibility in pp notation.", allowed_ranks=(rank_sing,))

        if mpi_dist_irrk.my_rank == rank_sing and mpi_dist_irrk.mpi_size > 1:
            mpi_dist_irrk.send_to_rank(gchi0_q_pp, dest=rank_trip, base_tag=0)
        elif mpi_dist_irrk.my_rank == rank_trip and mpi_dist_irrk.mpi_size > 1:
            gchi0_q_pp = mpi_dist_irrk.recv_from_rank(source=rank_sing, base_tag=0)

        lambdas_sing, lambdas_trip = (None,) * 2
        if mpi_dist_irrk.my_rank == rank_sing:
            lambdas_sing, gaps_sing = solve_eliashberg_lanczos(gamma_sing_pp, gchi0_q_pp, (rank_sing, rank_trip))
        if mpi_dist_irrk.my_rank == rank_trip:
            lambdas_trip, gaps_trip = solve_eliashberg_lanczos(gamma_trip_pp, gchi0_q_pp, (rank_sing, rank_trip))

        mpi_dist_irrk.delete_file()

        lambdas_sing = mpi_dist_irrk.bcast(lambdas_sing, root=rank_sing)
        lambdas_trip = mpi_dist_irrk.bcast(lambdas_trip, root=rank_trip)

        for i in range(len(gaps_sing)):
            gaps_sing[i] = mpi_dist_irrk.bcast_npoint(gaps_sing[i], root=rank_sing)
            gaps_trip[i] = mpi_dist_irrk.bcast_npoint(gaps_trip[i], root=rank_trip)
    else:
        mpi_dist_v = MpiDistributor.create_distributor(
            ntasks=gamma_sing_pp.current_shape[-2], comm=comm, name="V", output_path=config.output.output_path
        )

        logger.info("Distributing Gamma_sing_pp along v equally to ranks/nodes.")
        gamma_sing_pp = mpi_utils.gather_full_ibz_for_vslice(
            gamma_sing_pp, mpi_dist_irrk, mpi_dist_v, config.lattice.q_grid
        )
        logger.info("Gamma_sing_pp distributed. Starting with singlet Lanczos solver.")

        active_ranks = [
            r
            for r in range(comm.size)
            if mpi_dist_v.slices[r] is not None and (mpi_dist_v.slices[r].stop - mpi_dist_v.slices[r].start) > 0
        ]

        root = active_ranks[0]
        if comm.rank == root:
            # calculating gchi0 in the full BZ only once
            # in the lanczos step only active_rank[0] will perform chi0 * delta due to memory reasons
            gchi0_q_pp = BubbleGenerator.create_generalized_chi0_q_pp_w0(
                giwk_dga, niv_pp, config.lattice.q_grid
            ).decompress_q_dimension()
        else:
            gchi0_q_pp = None

        if comm.rank in active_ranks:
            lambdas_sing, gaps_sing = solve_eliashberg_lanczos_v2(gamma_sing_pp, gchi0_q_pp, mpi_dist_v, active_ranks)
            gamma_sing_pp.free()
        else:
            lambdas_sing = None

        logger.info("Distributing Gamma_trip_pp along v equally to ranks/nodes.")
        gamma_trip_pp = mpi_utils.gather_full_ibz_for_vslice(
            gamma_trip_pp, mpi_dist_irrk, mpi_dist_v, config.lattice.q_grid
        )
        logger.info("Gamma_trip_pp distributed. Starting with triplet Lanczos solver.")

        if comm.rank in active_ranks:
            lambdas_trip, gaps_trip = solve_eliashberg_lanczos_v2(gamma_trip_pp, gchi0_q_pp, mpi_dist_v, active_ranks)
            gamma_trip_pp.free()
        else:
            lambdas_trip = None

        lambdas_sing = comm.bcast(lambdas_sing, root=root)
        lambdas_trip = comm.bcast(lambdas_trip, root=root)

        for i in range(len(gaps_sing)):
            gaps_sing[i] = mpi_dist_irrk.bcast_npoint(gaps_sing[i], root=root)
            gaps_trip[i] = mpi_dist_irrk.bcast_npoint(gaps_trip[i], root=root)

    return lambdas_sing, lambdas_trip, gaps_sing, gaps_trip


def get_ranks_for_lanczos(comm: MPI.Comm) -> tuple[int, int]:
    """
    Picks two MPI ranks on different cluster nodes (if available) so the singlet and triplet Lanczos solves can run
    concurrently on separate nodes; falls back to two ranks on the same node otherwise.

    :param comm: The MPI communicator.
    :return: The tuple ``(rank_for_singlet, rank_for_triplet)``.
    """
    import socket

    hostname = socket.gethostname()

    # Gather all hostnames so every rank knows the full layout
    all_hostnames = comm.allgather(hostname)

    # Build a mapping: node_name -> list of ranks on that node
    node_to_ranks = {}
    for r, h in enumerate(all_hostnames):
        node_to_ranks.setdefault(h, []).append(r)

    nodes = list(node_to_ranks.keys())

    if len(nodes) >= 2:
        rank_for_singlet = node_to_ranks[nodes[0]][0]
        rank_for_triplet = node_to_ranks[nodes[1]][0]
    else:
        # Fallback: both on the same node, pick any two ranks
        ranks_on_node = node_to_ranks[nodes[0]]
        rank_for_singlet = ranks_on_node[0]
        rank_for_triplet = ranks_on_node[1] if len(ranks_on_node) > 1 else ranks_on_node[0]

    return rank_for_singlet, rank_for_triplet

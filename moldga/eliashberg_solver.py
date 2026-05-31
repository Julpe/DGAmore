# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
from typing import Tuple

import mpi4py.MPI as MPI
import numpy as np
import scipy as sp

import moldga.config as config
from moldga import nonlocal_sde, mpi_utils
from moldga.bubble_gen import BubbleGenerator
from moldga.four_point import FourPoint
from moldga.gap_function import GapFunction
from moldga.greens_function import GreensFunction
from moldga.interaction import LocalInteraction, Interaction
from moldga.local_four_point import LocalFourPoint
from moldga.matsubara_frequencies import MFHelper
from moldga.mpi_distributor import MpiDistributor
from moldga.n_point_base import SpinChannel, FrequencyNotation


def delete_files(filepath: str, *args) -> None:
    """
    Deletes files in the given directory. If the file is not found, it will be ignored. The files that are deleted
    are usually temporary files that are not needed anymore after the calculation is done.
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
    """
    Transforms the vertex function from particle-hole notation to particle-particle notation based on Motoharu Kitatani's
    frequency convention. This flips the fermionic frequency and maps w = v - v'.
    """
    vn = MFHelper.vn(niv_pp)
    wn = MFHelper.wn(config.box.niw_core)

    omega = vn[:, None] - vn[None, :]
    vertex = vertex.cut_niv(niv_pp).to_full_niw_range().flip_frequency_axis(-1, False)
    f_q_r_pp_mat = np.zeros((*vertex.current_shape[:-3], 2 * niv_pp, 2 * niv_pp), dtype=vertex.mat.dtype)

    for idx, w in enumerate(wn):
        f_q_r_pp_mat[..., omega == w] = -vertex[..., idx, omega == w]
    return f_q_r_pp_mat


def transform_vertex_loc_frequencies_w0(f_r_loc: LocalFourPoint, niv_pp: int) -> LocalFourPoint:
    """
    Transforms the vertex function from particle-hole notation to a modified particle-particle notation.
    """
    mat = _transform_vertex_frequencies_w0(f_r_loc, niv_pp)
    return LocalFourPoint(mat, SpinChannel.UD, 0, 2, True, True, FrequencyNotation.PP)


def transform_vertex_q_frequencies_w0(f_q_r: FourPoint, niv_pp: int) -> FourPoint:
    """
    Transforms the vertex function from particle-hole notation to a modified particle-particle notation.
    """
    mat = _transform_vertex_frequencies_w0(f_q_r, niv_pp)
    return FourPoint(mat, f_q_r.channel, config.lattice.q_grid.nk, 0, 2, True, True, True, FrequencyNotation.PP)


# --- Full q-dependent vertex creation and transformation ---
def create_full_vertex_q_r(
    u_loc: LocalInteraction, v_nonloc: Interaction, gamma_r: LocalFourPoint, niv_pp: int, mpi_dist: MpiDistributor
) -> FourPoint:
    """
    Calculates the full vertex in the given channel (either density or magnetic).
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

    f_q_r = config.sys.beta**2 * (gchi0_q_inv - gchi0_q_inv @ f_q_r @ gchi0_q_inv)
    gchi0_q_inv.free()

    if not config.eliashberg.save_fq:
        f_q_r = transform_vertex_q_frequencies_w0(f_q_r, niv_pp)

    mpi_dist.barrier()

    logger.info(f"Calculated first part of full {gamma_r.channel.value} vertex.")

    vrg_q_r = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"vrg_q_{gamma_r.channel.value}_rank_{mpi_dist.my_rank}.npy"),
        channel=gamma_r.channel,
        num_vn_dimensions=1,
    )

    if config.eliashberg.construct_fq_cheap:
        vrg_q_r = vrg_q_r.cut_niv(niv_pp)

    gchi_aux_q_r_sum = FourPoint.load(
        os.path.join(
            config.output.eliashberg_path, f"gchi_aux_q_{gamma_r.channel.value}_sum_rank_{mpi_dist.my_rank}.npy"
        ),
        channel=gamma_r.channel,
        num_vn_dimensions=0,
    )
    logger.info(f"Loaded vrg_q_{gamma_r.channel.value} and gchi_aux_q_{gamma_r.channel.value}_sum from files.")

    u = u_loc.as_channel(gamma_r.channel) + v_nonloc.as_channel(gamma_r.channel)
    f_q_r_2 = u @ (vrg_q_r * vrg_q_r) - u @ gchi_aux_q_r_sum @ u @ (vrg_q_r * vrg_q_r)
    vrg_q_r.free()
    gchi_aux_q_r_sum.free()

    if not config.eliashberg.save_fq:
        f_q_r_2 = transform_vertex_q_frequencies_w0(f_q_r_2, niv_pp)
    f_q_r += f_q_r_2
    f_q_r_2.free()

    mpi_dist.barrier()

    logger.info(f"Calculated second part of full {f_q_r.channel.value} vertex.")

    delete_files(
        config.output.eliashberg_path,
        f"vrg_q_{gamma_r.channel.value}_rank_{mpi_dist.my_rank}.npy",
        f"gchi_aux_q_{gamma_r.channel.value}_sum_rank_{mpi_dist.my_rank}.npy",
    )

    return f_q_r


def create_full_vertex_q_r_pp_w0(
    u_loc: LocalInteraction, v_nonloc: Interaction, gamma_r: LocalFourPoint, niv_pp: int, mpi_dist_irrk: MpiDistributor
):
    """
    Calculates the full vertex in PH notation and transforms it to PP notation for the density or magnetic channel.
    """
    logger = config.logger

    f_q_r = create_full_vertex_q_r(u_loc, v_nonloc, gamma_r, niv_pp, mpi_dist_irrk)

    if config.eliashberg.save_fq:
        f_q_r.mat = mpi_dist_irrk.gather(f_q_r.mat)
        if mpi_dist_irrk.comm.rank == 0:
            f_q_r.save(output_dir=config.output.output_path, name=f"f_irrq_{f_q_r.channel.value}")
        f_q_r.mat = mpi_dist_irrk.scatter(f_q_r.mat)
        config.logger.info(f"Saved full ladder-vertex ({f_q_r.channel.value}) in the irreducible BZ to file.")

    logger.info(f"Full ladder-vertex ({f_q_r.channel.value}) calculated.")
    logger.log_memory_usage(f"Full ladder-vertex ({f_q_r.channel.value})", f_q_r, mpi_dist_irrk.comm.size)

    if config.eliashberg.save_fq:
        return transform_vertex_q_frequencies_w0(f_q_r, niv_pp)
    return f_q_r


def create_full_vertex_q_r_v2(
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    vrg_q_r: FourPoint,
    gchi_aux_q_r_sum: FourPoint,
    niv_pp: int,
    q_index: int,
) -> FourPoint:
    """
    Calculates the full vertex in the given channel (either density or magnetic).
    """
    gchi0_q_inv_idx = gchi0_q_inv.filter_q_index(q_index)
    vrg_q_r_idx = vrg_q_r.filter_q_index(q_index)
    gchi_aux_q_r_sum_idx = gchi_aux_q_r_sum.filter_q_index(q_index)
    v_nonloc_idx = v_nonloc.filter_q_index(q_index)

    u = u_loc.as_channel(gamma_r.channel) + v_nonloc_idx.as_channel(gamma_r.channel)

    f_q_r_idx = nonlocal_sde.create_auxiliary_chi_r_q(gamma_r, gchi0_q_inv_idx, u_loc, v_nonloc_idx)
    f_q_r_idx = (
        config.sys.beta**2 * (gchi0_q_inv_idx - gchi0_q_inv_idx @ f_q_r_idx @ gchi0_q_inv_idx)
        + u @ (vrg_q_r_idx * vrg_q_r_idx)
        - u @ gchi_aux_q_r_sum_idx @ u @ (vrg_q_r_idx * vrg_q_r_idx)
    )

    gchi0_q_inv_idx.free()
    vrg_q_r_idx.free()
    gchi_aux_q_r_sum_idx.free()

    if not config.eliashberg.save_fq:
        f_q_r_idx = transform_vertex_q_frequencies_w0(f_q_r_idx, niv_pp)

    return f_q_r_idx


def create_full_vertex_q_r_pp_w0_v2(
    u_loc: LocalInteraction, v_nonloc: Interaction, gamma_r: LocalFourPoint, niv_pp: int, mpi_dist_irrk: MpiDistributor
):
    """
    Calculates the full vertex in PH notation and transforms it to PP notation for the density or magnetic channel.
    """
    logger = config.logger

    gchi0_q_inv = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"gchi0_q_inv_rank_{mpi_dist_irrk.my_rank}.npy"),
        num_vn_dimensions=1,
    )
    logger.info(f"Loaded gchi0_q_inv from file.")

    vrg_q_r = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"vrg_q_{gamma_r.channel.value}_rank_{mpi_dist_irrk.my_rank}.npy"),
        channel=gamma_r.channel,
        num_vn_dimensions=1,
    )

    gchi_aux_q_r_sum = FourPoint.load(
        os.path.join(
            config.output.eliashberg_path, f"gchi_aux_q_{gamma_r.channel.value}_sum_rank_{mpi_dist_irrk.my_rank}.npy"
        ),
        channel=gamma_r.channel,
        num_vn_dimensions=0,
    )

    if config.eliashberg.construct_fq_cheap:
        gamma_r = gamma_r.cut_niv(niv_pp)
        gchi0_q_inv = gchi0_q_inv.cut_niv(niv_pp)
        vrg_q_r = vrg_q_r.cut_niv(niv_pp)

    logger.info(f"Loaded vrg_q_{gamma_r.channel.value} and gchi_aux_q_{gamma_r.channel.value}_sum from files.")

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
            u_loc, v_nonloc, gamma_r, gchi0_q_inv, vrg_q_r, gchi_aux_q_r_sum, niv_pp, idx
        ).mat

    logger.info(f"Full ladder-vertex ({gamma_r.channel.value}) calculated.")

    gchi0_q_inv.free()
    vrg_q_r.free()
    gchi_aux_q_r_sum.free()

    delete_files(
        config.output.eliashberg_path,
        f"vrg_q_{gamma_r.channel.value}_rank_{mpi_dist_irrk.my_rank}.npy",
        f"gchi_aux_q_{gamma_r.channel.value}_sum_rank_{mpi_dist_irrk.my_rank}.npy",
    )

    if not config.eliashberg.save_fq:
        return FourPoint(
            f_q_r_mat, gamma_r.channel, config.lattice.q_grid.nk, 0, 2, False, True, True, FrequencyNotation.PP
        )

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
def create_local_ud_diagrams_pp_w0(g_dmft: GreensFunction) -> Tuple[LocalFourPoint, LocalFourPoint, LocalFourPoint]:
    r"""
    Creates the local particle-particle reducible diagrams for :math:`\omega=0`.
    """
    gchi_dens_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"gchi_dens_loc.npy"), SpinChannel.DENS)
    gchi_magn_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"gchi_magn_loc.npy"), SpinChannel.MAGN)
    gchi_ud_loc = 0.5 * (gchi_dens_loc - gchi_magn_loc).set_channel(SpinChannel.UD)
    gchi_ud_loc_pp_w0 = gchi_ud_loc.permute_orbitals("abcd->acbd").change_frequency_notation_ph_to_pp_w0()
    del gchi_dens_loc, gchi_magn_loc, gchi_ud_loc

    gchi0_loc_pp_w0 = (
        BubbleGenerator.create_generalized_chi0_pp_w0(g_dmft, gchi_ud_loc_pp_w0.niv, config.sys.beta)
        .extend_vn_to_diagonal()
        .flip_frequency_axis(-1, False)
    )

    gamma_ud_loc_pp_w0 = config.sys.beta**2 * (
        (gchi_ud_loc_pp_w0 - gchi0_loc_pp_w0).invert() + gchi0_loc_pp_w0.invert()
    )

    gamma_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="gamma_ud_loc_pp_w0")

    f_dens_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"f_dens_loc.npy"), SpinChannel.DENS)
    f_magn_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"f_magn_loc.npy"), SpinChannel.MAGN)
    f_ud_loc = 0.5 * (f_dens_loc - f_magn_loc).set_channel(SpinChannel.UD)
    f_ud_loc_pp_w0 = f_ud_loc.change_frequency_notation_ph_to_pp_w0()

    del f_dens_loc, f_magn_loc, f_ud_loc

    phi_ud_loc_pp_w0 = f_ud_loc_pp_w0 - gamma_ud_loc_pp_w0
    phi_ud_loc_pp_w0 = phi_ud_loc_pp_w0.take_first_wn()
    f_ud_loc_pp_w0 = f_ud_loc_pp_w0.take_first_wn()

    return f_ud_loc_pp_w0, gamma_ud_loc_pp_w0, phi_ud_loc_pp_w0


# --- Gap initialisation ---
def get_initial_gap_function(shape: tuple, channel: SpinChannel) -> np.ndarray:
    """
    Generates the initial gap function based on the specified shape, spin channel and symmetry settings.
    """
    if channel not in {SpinChannel.SING, SpinChannel.TRIP}:
        raise ValueError("Channel must be either SING or TRIP.")

    gap0 = np.zeros(shape, dtype=np.complex64)
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
def solve_eliashberg_lanczos(
    gamma_r_pp: FourPoint, gchi0_q0_pp: FourPoint, ranks: tuple[int, int]
) -> tuple[list[float], list[GapFunction]]:
    """
    Solves the Eliashberg equation for the superconducting eigenvalue and gap function using ARPACK.
    Returns (lambdas, gaps).
    """
    logger = config.logger

    logger.info(
        f"Starting to solve the Eliashberg equation for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=ranks,
    )

    gamma_r_pp = gamma_r_pp.map_to_full_bz(config.lattice.q_grid, config.lattice.q_grid.nk).decompress_q_dimension()
    logger.log_memory_usage(f"Gamma_pp_{gamma_r_pp.channel.value}", gamma_r_pp, 1, allowed_ranks=ranks)

    sign = 1 if gamma_r_pp.channel == SpinChannel.SING else -1

    gamma_r_pp = sign * gamma_r_pp.fft(False)
    gamma_pp_flipped = (
        gamma_r_pp.flip_momentum_axis().flip_frequency_axis(-1, False).permute_orbitals("abcd->adcb", False)
    )

    gap_shape = gamma_r_pp.nq + 2 * (gamma_r_pp.n_bands,) + (2 * gamma_r_pp.niv,)
    gchi0_q0_pp = gchi0_q0_pp.decompress_q_dimension()

    gap0 = get_initial_gap_function(gap_shape, gamma_r_pp.channel)
    gap0 = gap0.reshape((np.prod(gamma_r_pp.nq),) + 2 * (gamma_r_pp.n_bands,) + (2 * gamma_r_pp.niv,))[
        config.lattice.q_grid.irrk_ind
    ][config.lattice.q_grid.irrk_inv].reshape(gap_shape)
    symmetry_label = config.eliashberg.symmetry.lower() if config.eliashberg.symmetry else "random"
    logger.info(
        f"Initialized the gap function as {symmetry_label} for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=ranks,
    )

    einsum_str1 = "xyzabcdv,xyzcdv->xyzabv"
    path1 = np.einsum_path(einsum_str1, gchi0_q0_pp.mat, gap0, optimize=True)[1]
    einsum_str2 = "xyzacbdvp,xyzcdp->xyzabv"
    path2 = np.einsum_path(einsum_str2, gamma_r_pp.mat, gap0, optimize=True)[1]

    norm = 0.5 / config.lattice.q_grid.nk_tot / config.sys.beta

    def mv(gap: np.ndarray):
        gap_gg = np.fft.fftn(
            np.einsum(einsum_str1, gchi0_q0_pp.mat, gap.reshape(gap_shape), optimize=path1), axes=(0, 1, 2)
        )
        gap_gg_flipped = np.roll(np.flip(gap_gg, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
        gap_new = np.einsum(einsum_str2, gamma_r_pp.mat, gap_gg, optimize=path2) + sign * np.einsum(
            einsum_str2, gamma_pp_flipped.mat, gap_gg_flipped, optimize=path2
        )
        return np.fft.ifftn(norm * gap_new, axes=(0, 1, 2)).flatten()

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
        mat, k=n_eig, tol=config.eliashberg.epsilon, v0=gap0, which="LA", maxiter=10000
    )

    logger.info(
        f"Finished Lanczos method for the largest{eig_label} eigenvalue{plural} and eigenvector{plural} "
        f"for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=ranks,
    )

    order = lambdas.argsort()[::-1]  # sort eigenvalues in descending order
    lambdas = lambdas[order]
    gaps = gaps[:, order]

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
    """
    Solves the Eliashberg equation for the superconducting eigenvalue and gap function using ARPACK.
    More memory-efficient but slower variant than the other method.
    Returns (lambdas, gaps).
    """
    logger = config.logger
    root = active_ranks[0]

    logger.info(
        f"Starting to solve the Eliashberg equation for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=root,
    )
    logger.log_memory_usage(f"Gamma_pp_{gamma_r_pp.channel.value}", gamma_r_pp, len(active_ranks), allowed_ranks=root)

    sign = 1 if gamma_r_pp.channel == SpinChannel.SING else -1

    gamma_r_pp = sign * gamma_r_pp.fft(False).decompress_q_dimension()
    gamma_r_pp_flipped = (
        gamma_r_pp.flip_momentum_axis().flip_frequency_axis(-1, False).permute_orbitals("abcd->adcb", False)
    )

    gap_shape = gamma_r_pp.nq + 2 * (gamma_r_pp.n_bands,) + (gamma_r_pp.current_shape[-1],)

    gap0 = get_initial_gap_function(gap_shape, gamma_r_pp.channel)
    gap0 = mpi_dist_v.bcast(gap0, root=root)

    symmetry_label = config.eliashberg.symmetry.lower() if config.eliashberg.symmetry else "random"
    logger.info(
        f"Initialized the gap function as {symmetry_label} for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=root,
    )

    if mpi_dist_v.comm.rank == root:
        einsum_str1 = "xyzabcdv,xyzcdv->xyzabv"
        path1 = np.einsum_path(einsum_str1, gchi0_q0_pp.mat, gap0, optimize=True)[1]
    einsum_str2 = "xyzacbdvp,xyzcdp->xyzabv"  # sliced v
    path2 = np.einsum_path(einsum_str2, gamma_r_pp.mat, gap0, optimize=True)[1]

    norm = 0.5 / config.lattice.q_grid.nk_tot / config.sys.beta

    def mv(gap: np.ndarray):
        # 1. multiply chi0 * gap for the full BZ (only done by one rank, since memory would be an issue)
        gap_gg = None
        if mpi_dist_v.comm.rank == root:
            gap = gap.reshape(gap_shape)
            gap_gg = np.einsum(einsum_str1, gchi0_q0_pp.mat, gap, optimize=path1)
        gap_gg = mpi_dist_v.bcast_chunked(gap_gg, root=root)
        # 2. perform Fourier transform for the full chi0 * gap quantity
        gap_gg = np.fft.fftn(gap_gg, axes=(0, 1, 2))
        # 3. create gap_gg_flipped
        gap_gg_flipped = np.roll(np.flip(gap_gg, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
        # 4. multiply with the pairing vertex along the vp frequency axis (yields the gap for the broadcasted v)
        gap_new = np.einsum(einsum_str2, gamma_r_pp.mat, gap_gg, optimize=path2) + sign * np.einsum(
            einsum_str2, gamma_r_pp_flipped.mat, gap_gg_flipped, optimize=path2
        )
        # 5. perform fourier transform on the local v slice
        gap_new = np.fft.ifftn(norm * gap_new, axes=(0, 1, 2))
        # 6. assemble gap_new for the full v range through mpi_dist_v and allgather (remember we distributed v)
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
        mat, k=n_eig, tol=config.eliashberg.epsilon, v0=gap0, which="LA", maxiter=10000
    )

    logger.info(
        f"Finished Lanczos method for the largest{eig_label} eigenvalue{plural} and eigenvector{plural} "
        f"for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=root,
    )

    order = lambdas.argsort()[::-1]  # sort eigenvalues in descending order
    lambdas = lambdas[order]
    gaps = gaps[:, order]

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


def solve(
    giwk_dga: GreensFunction, g_dmft: GreensFunction, u_loc: LocalInteraction, v_nonloc: Interaction, comm: MPI.Comm
):
    """
    Solves the Eliashberg equation for largest the superconducting eigenvalues and corresponding gap functions.
    """
    logger = config.logger

    mpi_dist_irrk = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q")
    irrk_q_list = config.lattice.q_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    niv_pp = min(config.box.niw_core // 2, config.box.niv_core // 2)

    def dispatch_full_vertex_calculation(channel, u, v, niv, mpi_dist) -> FourPoint:
        gamma_r = LocalFourPoint.load(
            os.path.join(config.output.output_path, f"gamma_{channel.value}_loc.npy"), channel
        )
        if config.memory.save_memory_for_fq:
            f_q_r = create_full_vertex_q_r_pp_w0_v2(u, v, gamma_r, niv, mpi_dist)
        else:
            f_q_r = create_full_vertex_q_r_pp_w0(u, v, gamma_r, niv, mpi_dist)
        gamma_r.free()
        mpi_dist.barrier()
        return f_q_r

    f_dens_pp = dispatch_full_vertex_calculation(SpinChannel.DENS, u_loc, v_nonloc, niv_pp, mpi_dist_irrk)
    f_magn_pp = dispatch_full_vertex_calculation(SpinChannel.MAGN, u_loc, v_nonloc, niv_pp, mpi_dist_irrk)

    delete_files(config.output.eliashberg_path, f"gchi0_q_inv_rank_{comm.rank}.npy")
    delete_files(config.output.output_path, f"gchi0_q_rank_{comm.rank}.npy")

    mpi_dist_irrk.delete_file()

    gamma_sing_pp = 0.5 * f_dens_pp - 1.5 * f_magn_pp
    gamma_sing_pp.channel = SpinChannel.SING
    logger.info("Calculated full ladder-vertex (singlet) in pp notation.")

    gamma_trip_pp = 0.5 * f_dens_pp + 0.5 * f_magn_pp
    gamma_trip_pp.channel = SpinChannel.TRIP
    f_dens_pp.free()
    f_magn_pp.free()
    logger.info("Calculated full ladder-vertex (triplet) in pp notation.")

    if config.eliashberg.include_local_part:
        f_ud_loc_pp_w0, gamma_ud_loc_pp_w0, phi_ud_loc_pp_w0 = create_local_ud_diagrams_pp_w0(g_dmft)

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
        f_ud_loc = 0.5 * (f_dens_loc - f_magn_loc).set_channel(SpinChannel.UD)
        f_ud_loc_transf_w0 = transform_vertex_loc_frequencies_w0(f_ud_loc, niv_pp)
        del f_dens_loc, f_magn_loc, f_ud_loc

        gamma_sing_pp += f_ud_loc_transf_w0 + phi_ud_loc_pp_w0
        gamma_trip_pp += f_ud_loc_transf_w0 + phi_ud_loc_pp_w0
        del phi_ud_loc_pp_w0, f_ud_loc_transf_w0

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
        rank_sing, rank_trip = get_ranks_for_lanczos(mpi_dist_irrk.comm)

        if comm.size == 1:
            rank_trip = rank_sing

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
            gaps_sing[i] = mpi_dist_irrk.bcast(gaps_sing[i], root=rank_sing)
            gaps_trip[i] = mpi_dist_irrk.bcast(gaps_trip[i], root=rank_trip)
    else:
        mpi_dist_v = MpiDistributor.create_distributor(ntasks=gamma_sing_pp.current_shape[-2], comm=comm, name="V")

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
            lambdas_sing, gaps_sing = None, None

        logger.info("Distributing Gamma_trip_pp along v equally to ranks/nodes.")
        gamma_trip_pp = mpi_utils.gather_full_ibz_for_vslice(
            gamma_trip_pp, mpi_dist_irrk, mpi_dist_v, config.lattice.q_grid
        )
        logger.info("Gamma_trip_pp distributed. Starting with triplet Lanczos solver.")

        if comm.rank in active_ranks:
            lambdas_trip, gaps_trip = solve_eliashberg_lanczos_v2(gamma_trip_pp, gchi0_q_pp, mpi_dist_v, active_ranks)
            gamma_trip_pp.free()
        else:
            lambdas_trip, gaps_trip = None, None

        lambdas_sing = comm.bcast(lambdas_sing, root=root)
        lambdas_trip = comm.bcast(lambdas_trip, root=root)

        for i in range(len(gaps_sing)):
            gaps_sing[i] = mpi_dist_irrk.bcast(gaps_sing[i], root=root)
            gaps_trip[i] = mpi_dist_irrk.bcast(gaps_trip[i], root=root)

    return lambdas_sing, lambdas_trip, gaps_sing, gaps_trip


def get_ranks_for_lanczos(comm: MPI.Comm) -> tuple[int, int]:
    """
    Returns two ranks that are on different SLURM-cluster-nodes if possible, to run the Lanczos eigensolver
    for the singlet and triplet channel separately.
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

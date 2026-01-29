# python
import os
from copy import deepcopy
from typing import Tuple

import mpi4py.MPI as MPI
import numpy as np
import scipy as sp

import moldga.config as config
from moldga import nonlocal_sde
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
    vertex = vertex.cut_niv(niv_pp).to_full_niw_range().flip_frequency_axis(-1)
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
    u_loc: LocalInteraction, v_nonloc: Interaction, gamma_r: LocalFourPoint, comm: MPI.Comm
) -> FourPoint:
    """
    Calculates the full vertex in the given channel (either density or magnetic).
    """
    logger = config.logger
    logger.info(f"Starting to calculate the full {gamma_r.channel.value} vertex.")

    gchi0_q_inv = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"gchi0_q_inv_rank_{comm.rank}.npy"), num_vn_dimensions=1
    )
    logger.info(f"Loaded gchi0_q_inv from file.")
    f_q_r = nonlocal_sde.create_auxiliary_chi_r_q(gamma_r, gchi0_q_inv, u_loc, v_nonloc)
    logger.info(f"Non-Local auxiliary susceptibility ({gamma_r.channel.value}) calculated.")

    f_q_r = config.sys.beta**2 * (gchi0_q_inv - gchi0_q_inv @ f_q_r @ gchi0_q_inv)
    del gchi0_q_inv
    logger.info(f"Calculated first part of full {gamma_r.channel.value} vertex.")

    vrg_q_r = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"vrg_q_{gamma_r.channel.value}_rank_{comm.rank}.npy"),
        channel=gamma_r.channel,
        num_vn_dimensions=1,
    )
    gchi_aux_q_r_sum = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"gchi_aux_q_{gamma_r.channel.value}_sum_rank_{comm.rank}.npy"),
        channel=gamma_r.channel,
        num_vn_dimensions=0,
    )
    logger.info(f"Loaded vrg_q_{gamma_r.channel.value} and gchi_aux_q_{gamma_r.channel.value}_sum from files.")

    u = u_loc.as_channel(gamma_r.channel) + v_nonloc.as_channel(gamma_r.channel)
    f_q_r += u @ (vrg_q_r * vrg_q_r) - u @ gchi_aux_q_r_sum @ u @ (vrg_q_r * vrg_q_r)
    del gchi_aux_q_r_sum, vrg_q_r
    logger.info(f"Calculated second part of full {f_q_r.channel.value} vertex.")

    delete_files(
        config.output.eliashberg_path,
        f"vrg_q_{gamma_r.channel.value}_rank_{comm.rank}.npy",
        f"gchi_aux_q_{gamma_r.channel.value}_sum_rank_{comm.rank}.npy",
    )

    return f_q_r


def create_full_vertex_q_r_pp_w0(
    u_loc: LocalInteraction, v_nonloc: Interaction, gamma_r: LocalFourPoint, niv_pp: int, mpi_dist_irrk: MpiDistributor
):
    """
    Calculates the full vertex in PH notation and transforms it to PP notation for the density or magnetic channel.
    """
    logger = config.logger

    f_q_r = create_full_vertex_q_r(u_loc, v_nonloc, gamma_r, mpi_dist_irrk.comm)

    if config.eliashberg.save_fq:
        f_q_r.mat = mpi_dist_irrk.gather(f_q_r.mat)
        if mpi_dist_irrk.comm.rank == 0:
            f_q_r.save(output_dir=config.output.output_path, name=f"f_irrq_{f_q_r.channel.value}")
        f_q_r.mat = mpi_dist_irrk.scatter(f_q_r.mat)
        config.logger.info(f"Saved full ladder-vertex ({f_q_r.channel.value}) in the irreducible BZ to file.")

    logger.info(f"Full ladder-vertex ({f_q_r.channel.value}) calculated.")
    logger.log_memory_usage(f"Full ladder-vertex ({f_q_r.channel.value})", f_q_r, mpi_dist_irrk.comm.size)

    return transform_vertex_q_frequencies_w0(f_q_r, niv_pp)


# --- Local particle-particle reducible diagrams (w=0) ---
def create_local_ud_diagrams_pp_w0(
    g_loc: GreensFunction, niv_pp: int
) -> Tuple[LocalFourPoint, LocalFourPoint, LocalFourPoint]:
    r"""
    Creates the local particle-particle reducible diagrams for :math:`\omega=0`.
    """
    gchi_dens_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"gchi_dens_loc.npy"), SpinChannel.DENS)
    gchi_magn_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"gchi_magn_loc.npy"), SpinChannel.MAGN)
    gchi_ud_loc = 0.5 * (gchi_dens_loc - gchi_magn_loc).set_channel(SpinChannel.UD)
    gchi_ud_loc_pp_w0 = gchi_ud_loc.change_frequency_notation_ph_to_pp_w0()
    del gchi_dens_loc, gchi_magn_loc, gchi_ud_loc

    gchi0_loc_pp_w0 = (
        BubbleGenerator.create_generalized_chi0_pp_w0(g_loc, gchi_ud_loc_pp_w0.niv)
        .extend_vn_to_diagonal()
        .flip_frequency_axis(-1)
    )

    gamma_ud_loc_pp_w0 = config.sys.beta**2 * (
        (gchi_ud_loc_pp_w0 - gchi0_loc_pp_w0).invert() + gchi0_loc_pp_w0.invert()
    )

    if config.output.save_quantities:
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
def solve_eliashberg_lanczos(gamma_q_r_pp: FourPoint, gchi0_q0_pp: FourPoint):
    """
    Solves the Eliashberg equation for the superconducting eigenvalue and gap function using ARPACK.
    Returns (lambdas, gaps).
    """
    logger = config.logger

    logger.info(
        f"Starting to solve the Eliashberg equation for the {gamma_q_r_pp.channel.value}let channel.",
        allowed_ranks=(0, 1),
    )

    gamma_q_r_pp = gamma_q_r_pp.map_to_full_bz(
        config.lattice.q_grid.irrk_inv, config.lattice.q_grid.nk
    ).decompress_q_dimension()
    logger.log_memory_usage(f"Gamma_pp_{gamma_q_r_pp.channel.value}", gamma_q_r_pp, 1, allowed_ranks=(0, 1))

    sign = 1 if gamma_q_r_pp.channel == SpinChannel.SING else -1

    gamma_x = sign * gamma_q_r_pp.fft()
    gamma_x_flipped = gamma_x.flip_momentum_axis().flip_frequency_axis(-1).permute_orbitals("abcd->adcb")

    gap_shape = gamma_q_r_pp.nq + 2 * (gamma_q_r_pp.n_bands,) + (2 * gamma_q_r_pp.niv,)
    gchi0_q0_pp = gchi0_q0_pp.decompress_q_dimension()

    gap0 = get_initial_gap_function(gap_shape, gamma_q_r_pp.channel)
    symmetry_label = config.eliashberg.symmetry if config.eliashberg.symmetry else "random"
    logger.info(
        f"Initialized the gap function as {symmetry_label} for the {gamma_q_r_pp.channel.value}let channel.",
        allowed_ranks=(0, 1),
    )

    norm = 0.5 / config.lattice.q_grid.nk_tot / config.sys.beta
    nx, ny, nz, o, _, v = gap_shape
    xyz = nx * ny * nz
    o2 = o * o
    m = o2 * v

    # reshape gamma to (xyz, m, m)
    gamma_x_mat_compound = gamma_x.mat.transpose(0, 1, 2, 3, 5, 7, 6, 4, 8).reshape(xyz, m, m)
    gamma_x_flipped_mat_compound = gamma_x_flipped.mat.transpose(0, 1, 2, 3, 5, 7, 6, 4, 8).reshape(xyz, m, m)
    gchi0_q0_pp_mat_compound_orbs = gchi0_q0_pp.mat.transpose(0, 1, 2, 7, 3, 5, 4, 6).reshape(xyz * v, o2, o2)

    def mv(gap: np.ndarray):
        gap = gap.reshape(gap_shape)
        gap_r = gap.transpose(0, 1, 2, 5, 4, 3).reshape(xyz * v, o2)
        # Perform matrix multiplication and reshape the result
        gap_gg = np.matmul(gchi0_q0_pp_mat_compound_orbs, gap_r[..., None])[..., 0]
        gap_gg = gap_gg.reshape(nx, ny, nz, v, o, o).transpose(0, 1, 2, 5, 4, 3)
        # Apply FFT and flip operations
        gap_gg_fft = np.fft.fftn(gap_gg, axes=(0, 1, 2))
        gap_gg_flipped = np.roll(np.flip(gap_gg_fft, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
        # Flatten and prepare for matrix multiplication
        gap_flat = gap_gg_fft.reshape(xyz, m)
        gap_flat_flip = gap_gg_flipped.reshape(xyz, m)
        # Perform matrix multiplications
        gap_new_flat = np.matmul(gamma_x_mat_compound, gap_flat[..., None])[..., 0]
        gap_new_flat += sign * np.matmul(gamma_x_flipped_mat_compound, gap_flat_flip[..., None])[..., 0]
        # Reshape and apply inverse FFT
        gap_new = gap_new_flat.reshape(nx, ny, nz, o, o, v).transpose(0, 1, 2, 4, 3, 5)
        return np.fft.ifftn(norm * gap_new, axes=(0, 1, 2)).flatten()

    mat = sp.sparse.linalg.LinearOperator(shape=(np.prod(gap_shape), np.prod(gap_shape)), matvec=mv)

    n_eig = config.eliashberg.n_eig
    eig_label = "" if n_eig > 1 else f" {n_eig}"
    plural = "" if n_eig == 1 else "s"
    logger.info(
        f"Starting Lanczos method to retrieve largest{eig_label} eigenvalue{plural} and eigenvector{plural} "
        f"for the {gamma_q_r_pp.channel.value}let channel.",
        allowed_ranks=(0, 1),
    )

    lambdas, gaps = sp.sparse.linalg.eigsh(
        mat, k=n_eig, tol=config.eliashberg.epsilon, v0=gap0, which="LA", maxiter=10000
    )

    logger.info(
        f"Finished Lanczos method for the largest{eig_label} eigenvalue{plural} and eigenvector{plural} "
        f"for the {gamma_q_r_pp.channel.value}let channel.",
        allowed_ranks=(0, 1),
    )

    order = lambdas.argsort()[::-1]  # sort eigenvalues in descending order
    lambdas = lambdas[order]
    gaps = gaps[:, order]

    logger.info(
        f"Largest{eig_label} eigenvalue{plural} for the {gamma_q_r_pp.channel.value}let "
        f"channel {"is" if n_eig == 1 else "are"}: " + ", ".join(f"{lam:.6f}" for lam in lambdas),
        allowed_ranks=(0, 1),
    )

    gaps = [
        GapFunction(gaps[..., i].reshape(gap_shape), gamma_q_r_pp.channel, gamma_q_r_pp.nq)
        for i in range(config.eliashberg.n_eig)
    ]

    logger.info(
        f"Finished solving the Eliashberg equation for the {gamma_q_r_pp.channel.value}let channel.",
        allowed_ranks=(0, 1),
    )

    return lambdas, gaps


# --- Main solve orchestration ---
def solve(
    giwk_dga: GreensFunction,
    g_loc: GreensFunction,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    gamma_dens: LocalFourPoint,
    gamma_magn: LocalFourPoint,
    comm: MPI.Comm,
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

    f_dens_pp = create_full_vertex_q_r_pp_w0(u_loc, v_nonloc, gamma_dens, niv_pp, mpi_dist_irrk)
    f_magn_pp = create_full_vertex_q_r_pp_w0(u_loc, v_nonloc, gamma_magn, niv_pp, mpi_dist_irrk)

    delete_files(config.output.eliashberg_path, f"gchi0_q_inv_rank_{comm.rank}.npy")
    delete_files(config.output.output_path, f"gchi0_q_rank_{comm.rank}.npy")

    mpi_dist_irrk.delete_file()

    gamma_sing_pp = 0.5 * f_dens_pp - 1.5 * f_magn_pp
    gamma_sing_pp.channel = SpinChannel.SING
    logger.info("Calculated full ladder-vertex (singlet) in pp notation.")

    gamma_trip_pp = 0.5 * f_dens_pp + 0.5 * f_magn_pp
    gamma_trip_pp.channel = SpinChannel.TRIP
    del f_dens_pp, f_magn_pp
    logger.info("Calculated full ladder-vertex (triplet) in pp notation.")

    gamma_sing_pp.mat = mpi_dist_irrk.gather(gamma_sing_pp.mat)
    gamma_trip_pp.mat = mpi_dist_irrk.gather(gamma_trip_pp.mat)

    gchi0_q0_pp = None
    if mpi_dist_irrk.my_rank == 0:
        gchi0_q0_pp = BubbleGenerator.create_generalized_chi0_q_pp_w0(giwk_dga, niv_pp)
        logger.info("Created the bare bubble susceptibility in pp notation.")

        if config.eliashberg.include_local_part:
            f_ud_loc_pp_w0, gamma_ud_loc_pp_w0, phi_ud_loc_pp_w0 = create_local_ud_diagrams_pp_w0(g_loc, niv_pp)

            if config.output.save_quantities:
                f_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="f_ud_loc_pp_w0")
                phi_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="phi_ud_loc_pp_w0")
                gamma_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="gamma_ud_loc_pp_w0")
                logger.info("Saved local ud diagrams in pp notation to file.")

            del f_ud_loc_pp_w0, gamma_ud_loc_pp_w0

            # special treatment of local full vertex that is subtracted with a different frequency notation and is
            # different from the regular pp
            f_dens_loc = LocalFourPoint.load(
                os.path.join(config.output.output_path, f"f_dens_loc.npy"), SpinChannel.DENS
            )
            f_magn_loc = LocalFourPoint.load(
                os.path.join(config.output.output_path, f"f_magn_loc.npy"), SpinChannel.MAGN
            )
            f_ud_loc = 0.5 * (f_dens_loc - f_magn_loc).set_channel(SpinChannel.UD)
            f_ud_loc_transf_w0 = transform_vertex_loc_frequencies_w0(f_ud_loc, niv_pp)
            del f_dens_loc, f_magn_loc, f_ud_loc

            gamma_sing_pp -= f_ud_loc_transf_w0 + phi_ud_loc_pp_w0
            gamma_trip_pp -= f_ud_loc_transf_w0 + phi_ud_loc_pp_w0
            del phi_ud_loc_pp_w0, f_ud_loc_transf_w0

        if config.eliashberg.save_pairing_vertex:
            gamma_sing_pp.save(
                output_dir=config.output.eliashberg_path, name=f"gamma_irrq_{gamma_sing_pp.channel.value}_pp"
            )
            gamma_trip_pp.save(
                output_dir=config.output.eliashberg_path, name=f"gamma_irrq_{gamma_trip_pp.channel.value}_pp"
            )
            config.logger.info(
                f"Saved singlet and triplet pairing vertices in pp notation in the irreducible BZ to file."
            )

    gchi0_q0_pp = mpi_dist_irrk.bcast(gchi0_q0_pp, root=0)
    gamma_trip_pp = mpi_dist_irrk.bcast(gamma_trip_pp, root=0)

    lambdas_sing, lambdas_trip, gaps_sing, gaps_trip = (None,) * 4
    if mpi_dist_irrk.my_rank == 0:
        lambdas_sing, gaps_sing = solve_eliashberg_lanczos(gamma_sing_pp, gchi0_q0_pp)
    if mpi_dist_irrk.mpi_size == 1 or mpi_dist_irrk.my_rank == 1:
        lambdas_trip, gaps_trip = solve_eliashberg_lanczos(gamma_trip_pp, gchi0_q0_pp)

    mpi_dist_irrk.delete_file()

    lambdas_sing = mpi_dist_irrk.bcast(lambdas_sing, root=0)
    lambdas_trip = mpi_dist_irrk.bcast(lambdas_trip, root=1 if mpi_dist_irrk.mpi_size > 1 else 0)

    gaps_sing = mpi_dist_irrk.bcast(gaps_sing, root=0)
    gaps_trip = mpi_dist_irrk.bcast(gaps_trip, root=1 if mpi_dist_irrk.mpi_size > 1 else 0)

    return lambdas_sing, lambdas_trip, gaps_sing, gaps_trip

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import glob
import os
import re
from copy import deepcopy

import mpi4py.MPI as MPI
import numpy as np
from scipy import optimize as opt

import dgamore.config as config
import dgamore.lambda_correction as lc
import dgamore.mpi_utils as mpi_utils
from dgamore.brillouin_zone import KGrid
from dgamore.bubble_gen import BubbleGenerator
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction, update_mu
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.local_four_point import LocalFourPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore.mpi_distributor import MpiDistributor
from dgamore.n_point_base import SpinChannel
from dgamore.self_energy import SelfEnergy


def get_hartree_fock(
    u_loc: LocalInteraction, v_nonloc: Interaction, q_list: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Returns the Hartree-Fock term separately for the local and non-local interaction. Since we are always SU(2)-symmetric,
    the sum over the spins of the first term in Eq. (4.55) in Anna Galler's thesis results in a simple factor of 2. This
    can be seen in my master's thesis, Eq. (3.56). The Hartree-Fock term is given by
    .. math:: \Sigma_{HF}^k = 2(U_{abcd} + V^{q=0}_{abcd}) n_{dc} - 1/N_q \sum_q (U_{adcb} + V^{q}_{adcb}) n^{k-q}_{dc}
    where the Hartree-term reads :math:`\Sigma_{H} = 2(U_{abcd} + V^{q=0}_{abcd}) n_{dc}` and the Fock-term reads
    :math:`\Sigma_{F}^k = - 1/N_q \sum_q (U_{adcb} + V^{q}_{adcb}) n^{k-q}_{dc}`.
    Processes the Fock-Term for each individual orbital to save memory, as for high momentum grids,
    the occ_qk property can become large.
    """
    v_q0 = v_nonloc.find_q((0, 0, 0))
    hartree = 2 * (u_loc + v_q0).times("qabcd,dc->ab", config.sys.occ)

    nb = config.sys.n_bands
    nk_tot = np.prod(config.lattice.nk)
    nq_tot = np.prod(config.lattice.nq)

    uq = (u_loc + v_nonloc.reduce_q(q_list)).permute_orbitals("abcd->adcb")  # (nq,a,d,c,b)

    fock = np.zeros((nk_tot, nb, nb), dtype=uq.mat.dtype)

    for d in range(nb):
        for c in range(nb):
            u_slice = uq[:, :, d, c, :]
            if not np.any(u_slice):
                continue

            occ_qk_dc = np.array(
                [np.roll(config.sys.occ_k[..., d, c], [-i for i in q], axis=(0, 1, 2)) for q in q_list]
            )
            occ_qk_dc = occ_qk_dc.reshape(len(q_list), nk_tot)
            contribution = u_slice[:, None, :, :] * occ_qk_dc[:, :, None, None]
            fock += contribution.sum(axis=0)

    fock *= -1.0 / nq_tot
    return hartree[None, ..., None], fock[..., None]  # [k,o1,o2,v]


def create_auxiliary_chi_r_q(
    gamma_r: LocalFourPoint, gchi0_q_inv: FourPoint, u_loc: LocalInteraction, v_nonloc: Interaction
) -> FourPoint:
    r"""
    Returns the auxiliary susceptibility, see Eq. (3.60) in my master's thesis.
    .. math:: \chi^{*;qvv'}_{r;abcd} = ((\chi_{0;abcd}^{qv})^{-1} + (\Gamma_{r;abcd}^{wvv'}-U_{r;abcd}-V_{r;abcd}^q)/\beta^2)^{-1}
    """
    return (
        (gchi0_q_inv + 1.0 / config.sys.beta**2 * gamma_r)
        - 1.0 / config.sys.beta**2 * (v_nonloc.as_channel(gamma_r.channel) + u_loc.as_channel(gamma_r.channel))
    ).invert(False)


def create_auxiliary_chi_r_q_sum_v1(
    gamma_r: LocalFourPoint, gchi0_q_inv: FourPoint, u_loc: LocalInteraction, v_nonloc: Interaction
) -> FourPoint:
    r"""
    Returns the auxiliary susceptibility, see Eq. (3.60) in my master's thesis.
    .. math:: \chi^{*;qvv'}_{r;abcd} = ((\chi_{0;abcd}^{qv})^{-1} +
    (\Gamma_{r;abcd}^{wvv'}-U_{r;abcd}-V_{r;abcd}^q)/\beta^2)^{-1}
    """
    return (
        (gchi0_q_inv + 1.0 / config.sys.beta**2 * gamma_r)
        - 1.0 / config.sys.beta**2 * (v_nonloc.as_channel(gamma_r.channel) + u_loc.as_channel(gamma_r.channel))
    ).invert_and_sum_over_last_vn(config.sys.beta)


def create_auxiliary_chi_r_q_sum_v2(
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    mpi_dist_irrq: MpiDistributor,
) -> FourPoint:
    r"""
    Returns the sum over the auxiliary susceptibility, see Eq. (3.60) in my master's thesis.
    .. math:: \sum_{v'}\chi^{*;qvv'}_{r;abcd} = \sum_{v'}((\chi_{0;abcd}^{qv})^{-1} +
    (\Gamma_{r;abcd}^{wvv'}-U_{r;abcd}-V_{r;abcd}^q)/\beta^2)^{-1}
    """
    irrk_q_list = config.lattice.q_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrq.my_slice]
    chi_r_q_sum_mat = np.zeros_like(gchi0_q_inv.mat)

    for idx in range(len(my_irr_q_list)):
        chi_r_q_sum_mat[idx] = (
            (
                (gchi0_q_inv.filter_q_index(idx) + 1.0 / config.sys.beta**2 * gamma_r)
                - 1.0
                / config.sys.beta**2
                * (v_nonloc.as_channel(gamma_r.channel).filter_q_index(idx) + u_loc.as_channel(gamma_r.channel))
            )
            .invert_and_sum_over_last_vn(config.sys.beta)
            .mat
        )
    return FourPoint(chi_r_q_sum_mat, gamma_r.channel, config.lattice.nq, 1, 1, False, has_compressed_q_dimension=True)


def create_auxiliary_chi_r_q_sum_v3(
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    mpi_dist_irrq: MpiDistributor,
) -> FourPoint:
    r"""
    Returns the sum over the auxiliary susceptibility, see Eq. (3.60) in my master's thesis.
    .. math:: \sum_{v'}\chi^{*;qvv'}_{r;abcd} = \sum_{v'}((\chi_{0;abcd}^{qv})^{-1} +
    (\Gamma_{r;abcd}^{wvv'}-U_{r;abcd}-V_{r;abcd}^q)/\beta^2)^{-1}
    """
    irrk_q_list = config.lattice.q_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrq.my_slice]
    chi_r_q_sum_mat = np.zeros_like(gchi0_q_inv.mat)

    for idx in range(len(my_irr_q_list)):
        chi_r_q_sum_mat[idx] = (
            (
                (gchi0_q_inv.filter_q_index(idx) + 1.0 / config.sys.beta**2 * gamma_r)
                - 1.0
                / config.sys.beta**2
                * (v_nonloc.as_channel(gamma_r.channel).filter_q_index(idx) + u_loc.as_channel(gamma_r.channel))
            )
            .invert_and_sum_over_last_vn_v2(config.sys.beta)
            .mat
        )
    return FourPoint(chi_r_q_sum_mat, gamma_r.channel, config.lattice.nq, 1, 1, False, has_compressed_q_dimension=True)


def create_vrg_r_q(gchi_aux_q_r_sum: FourPoint, gchi0_q_inv: FourPoint) -> FourPoint:
    r"""
    Returns the three-leg vertex, see Eq. (3.63) in my master's thesis.
    .. math:: \gamma_{r;abcd}^{qv} = \beta * (\chi^{qvv}_{0;ablm})^{-1} * (\sum_{v'} \chi^{*;qvv'}_{r;mlcd}).
    """
    return config.sys.beta * (gchi0_q_inv @ gchi_aux_q_r_sum)


def create_generalized_chi_q_with_shell_correction(
    gchi_aux_q_sum: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
) -> FourPoint:
    """
    Calculates the generalized susceptibility with the shell correction as described by
    Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005; DOI 10.1088/2515-7639/ac7e6d. Eq. A.15. See also Sec. 3.7.2
    in my master's thesis for details.
    """
    return (
        (gchi_aux_q_sum + gchi0_q_full_sum - gchi0_q_core_sum).invert()
        + (u_loc.as_channel(gchi_aux_q_sum.channel) + v_nonloc.as_channel(gchi_aux_q_sum.channel))
    ).invert()


def calculate_sigma_dc_kernel(f_dc_loc: LocalFourPoint, gchi0_q: FourPoint, u_loc: LocalInteraction) -> FourPoint:
    """
    Returns the double-counting kernel for the self-energy calculation. For details, see Eq. (4.28) in my
    master's thesis.
    """
    kernel = 1.0 / config.sys.beta**2 * u_loc.permute_orbitals("abcd->adcb") @ gchi0_q

    einsum_str = "abcdwv,dcefwvp->abefwp"
    path, _ = np.einsum_path(einsum_str, kernel.mat[0].copy(), f_dc_loc.mat, optimize="optimal")

    for q in range(kernel.current_shape[0]):
        kernel[q] = np.einsum(einsum_str, kernel[q].copy(), f_dc_loc.mat, optimize=path)

    return kernel.cut_niv(config.box.niv_core)


def calculate_kernel_r_q(
    vrg_q_r: FourPoint, gchi_aux_q_r_sum: FourPoint, v_nonloc: Interaction, u_loc: LocalInteraction
) -> FourPoint:
    r"""
    Returns the kernel for the self-energy calculation minus 2/3 times the identity if the channel is the magnetic
    channel (due to the extra factor of :math:`U_{ah21}` in Eq. (4.29) in my master's thesis).
    .. math:: K = \gamma_{r;abcd}^{qv} - \gamma_{r;abef}^{qv} U^{q}_{r;fehg} \chi_{r;ghcd}^{q}
    """
    u_r = v_nonloc.as_channel(vrg_q_r.channel) + u_loc.as_channel(vrg_q_r.channel)
    kernel = vrg_q_r - vrg_q_r @ u_r @ gchi_aux_q_r_sum

    if vrg_q_r.channel == SpinChannel.MAGN:
        kernel -= 2.0 / 3.0 * FourPoint.identity_like(kernel)

    return u_r @ kernel


def perform_ornstein_zernicke_fit(chi_phys_q_r: FourPoint) -> None:
    def oz_spin_w0(q_grid: KGrid, a: float, xi: float):
        qx = qy = np.pi
        qz = 0
        oz = a / (
            xi ** (-2)
            + (q_grid.kx[:, None, None] - qx) ** 2
            + (q_grid.ky[None, :, None] - qy) ** 2
            + (q_grid.kz[None, None, :] - qz) ** 2
        )
        return oz.flatten()

    def fit_oz_spin(q_grid: KGrid, mat: np.ndarray):
        initial_guess = (mat.max(), 2.0)
        return opt.curve_fit(oz_spin_w0, q_grid, mat, p0=initial_guess)[0]

    chi = deepcopy(chi_phys_q_r)
    chi_mat = chi.map_to_full_bz(config.lattice.q_grid).to_half_niw_range().take_first_wn().mat.real
    orb_shape = (config.sys.n_bands,) * 4
    oz_coeffs = np.zeros(orb_shape + (2,), dtype=float)

    for idx in np.ndindex(orb_shape):
        mat_slice = chi_mat[..., idx[0], idx[1], idx[2], idx[3]].flatten()
        try:
            coeffs = fit_oz_spin(config.lattice.q_grid, mat_slice) if not np.all(mat_slice == 0) else [0.0, 0.0]
        except (ValueError, RuntimeError, opt.OptimizeWarning):
            config.logger.warning(f"OZ fit did not converge for orbitals {idx}. Using [-1, -1].")
            coeffs = [-1.0, -1.0]
        oz_coeffs[idx] = coeffs

    rows = []
    for idx in np.ndindex(orb_shape):
        rows.append([*idx, *oz_coeffs[idx]])

    data_to_save = np.array(rows, dtype=float)
    path = os.path.join(config.output.output_path, f"oz_coeff.txt")
    np.savetxt(path, data_to_save, delimiter=",", fmt="%d %d %d %d %.9f %.9f", header="o1 o2 o3 o4 A xi")


def calculate_sigma_kernel_r_q(
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    mpi_dist_irrq: MpiDistributor,
) -> FourPoint:
    r"""
    Returns the kernel for the self-energy calculation in a specific spin channel. Calculates the auxiliary
    susceptibility, the three-leg vertex and the physical susceptibility with shell correction. Also performs a
    :math:`\lambda`-correction on the physical susceptibility if specified in the config for single-band input.
    """
    logger = config.logger

    if config.memory.save_memory_for_chiq_aux:
        gchi_aux_q_r_sum = create_auxiliary_chi_r_q_sum_v3(gamma_r, gchi0_q_inv, u_loc, v_nonloc, mpi_dist_irrq)
    else:
        gchi_aux_q_r_sum = create_auxiliary_chi_r_q_sum_v1(gamma_r, gchi0_q_inv, u_loc, v_nonloc)

    mpi_dist_irrq.barrier()

    logger.log_memory_usage(
        f"Gchi_aux ({gchi_aux_q_r_sum.channel.value})",
        gchi_aux_q_r_sum,
        mpi_dist_irrq.comm.size * 2 * config.box.niv_core,
    )
    logger.info(f"Non-Local auxiliary susceptibility ({gchi_aux_q_r_sum.channel.value}) calculated.")

    vrg_q_r = create_vrg_r_q(gchi_aux_q_r_sum, gchi0_q_inv)

    logger.info(f"Non-local three-leg vertex gamma^wv ({vrg_q_r.channel.value}) done.")
    logger.log_memory_usage(f"Three-leg vertex ({vrg_q_r.channel.value})", vrg_q_r, mpi_dist_irrq.comm.size)

    if config.eliashberg.perform_eliashberg:
        vrg_q_r.save(
            name=f"vrg_q_{vrg_q_r.channel.value}_rank_{mpi_dist_irrq.comm.rank}",
            output_dir=config.output.eliashberg_path,
        )

    chi_phys_q_r = gchi_aux_q_r_sum.sum_over_all_vn(config.sys.beta)
    gchi_aux_q_r_sum.free()

    chi_phys_q_r = create_generalized_chi_q_with_shell_correction(
        chi_phys_q_r, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
    )

    logger.info(f"Updated non-local susceptibility chi^q ({chi_phys_q_r.channel.value}) with asymptotic correction.")

    if config.self_consistency.restrict_chi_phys:
        logger.warning("Restricting physical susceptibility to positive values.")
        chi_phys_q_r = chi_phys_q_r.invert()
        chi_phys_q_r.mat[chi_phys_q_r.mat < 0] = 1e-4
        chi_phys_q_r = chi_phys_q_r.invert()

    logger.log_memory_usage(
        f"Physical susceptibility ({chi_phys_q_r.channel.value})", chi_phys_q_r, mpi_dist_irrq.comm.size
    )

    chi_phys_q_r.mat = mpi_dist_irrq.gather(chi_phys_q_r.mat)
    if mpi_dist_irrq.comm.rank == 0:
        if config.lambda_correction.perform_lambda_correction:
            chi_phys_q_r = perform_lambda_correction(chi_phys_q_r)
        chi_phys_q_r.save(name=f"chi_phys_q_{chi_phys_q_r.channel.value}", output_dir=config.output.output_path)

        # perform Ornstein-Zernicke fit
        if chi_phys_q_r.channel == SpinChannel.MAGN:
            perform_ornstein_zernicke_fit(chi_phys_q_r)

    chi_phys_q_r.mat = mpi_dist_irrq.scatter(chi_phys_q_r.mat)
    logger.info(f"Saved physical susceptibility ({chi_phys_q_r.channel.value}) to file.")

    if config.eliashberg.perform_eliashberg:
        chi_phys_q_r.save(
            name=f"gchi_aux_q_{chi_phys_q_r.channel.value}_sum_rank_{mpi_dist_irrq.comm.rank}",
            output_dir=config.output.eliashberg_path,
        )

    return calculate_kernel_r_q(vrg_q_r, chi_phys_q_r, v_nonloc, u_loc)


def perform_lambda_correction(chi_phys_q_r: FourPoint) -> FourPoint:
    r"""
    Performs the :math:`\lambda`-correction on the physical susceptibility. If 'spch' is specified, the lambda
    correction will be performed on both the density and magnetic channel whereas only the magnetic channel will be
    corrected if 'sp' is specified as :math:`\lambda`-correction type in the corresponding config.
    """
    logger = config.logger

    if config.lambda_correction.type.lower() not in ["spch", "sp"]:
        raise ValueError("Lambda correction type must be either 'spch' or 'sp'.")

    logger.info(f"Lambda correction type set to '{config.lambda_correction.type}'.")

    if config.lambda_correction.type.lower() == "spch":
        logger.info(f"Performing lambda correction for {chi_phys_q_r.channel.value} channel.")
        chi_r_loc = LocalFourPoint.load(
            os.path.join(config.output.output_path, f"chi_{chi_phys_q_r.channel.value}_loc.npy"),
            chi_phys_q_r.channel,
            num_vn_dimensions=0,
        ).to_full_niw_range()
        chi_phys_q_r, lambda_r = lc.perform_single_lambda_correction(
            chi_phys_q_r, chi_r_loc.mat.sum() / config.sys.beta
        )
        chi_r_loc.free()
        logger.info(
            f"Lambda correction for the {chi_phys_q_r.channel.value} channel applied with lambda = {lambda_r:.6f}."
        )

        with open(os.path.join(config.output.output_path, f"lambda_{config.lambda_correction.type}.txt"), "a") as f:
            f.write(f"lambda_{chi_phys_q_r.channel.value}: {lambda_r}\n")

        return chi_phys_q_r

    # else: "sp"
    if chi_phys_q_r.channel != SpinChannel.MAGN:
        return chi_phys_q_r

    logger.info(f"Performing lambda correction for magn channel.")
    chi_phys_q_dens = FourPoint.load(
        os.path.join(config.output.output_path, f"chi_phys_q_dens.npy"),
        SpinChannel.DENS,
        num_vn_dimensions=0,
    ).to_full_niw_range()

    chi_dens_loc, chi_magn_loc = [
        LocalFourPoint.load(
            os.path.join(config.output.output_path, f"chi_{channel.value}_loc.npy"),
            channel,
            num_vn_dimensions=0,
        ).to_full_niw_range()
        for channel in [SpinChannel.DENS, SpinChannel.MAGN]
    ]

    chi_magn_loc_sum = (chi_dens_loc.mat + chi_magn_loc.mat).sum() - 1 / config.lattice.q_grid.nk_tot * (
        config.lattice.q_grid.irrk_count[:, None, None, None, None, None] * chi_phys_q_dens.mat
    ).sum()
    chi_phys_q_r, lambda_r = lc.perform_single_lambda_correction(chi_phys_q_r, chi_magn_loc_sum / config.sys.beta)
    logger.info(f"Lambda correction 'sp' applied. Lambda for magn channel is: {lambda_r:.6f}.")

    with open(os.path.join(config.output.output_path, f"lambda_{config.lambda_correction.type}.txt"), "a") as f:
        f.write(f"lambda_{chi_phys_q_r.channel.value}: {lambda_r}\n")

    return chi_phys_q_r


def calculate_sigma_from_kernel(kernel: FourPoint, giwk: GreensFunction, my_full_q_list: np.ndarray) -> SelfEnergy:
    r"""
    Returns :math:`\Sigma_{ij}^{k} = -1/2 * 1/\beta * 1/N_q \sum_q [ U^q_{r;aibc} * K_{r;cbjd}^{qv} * G_{ad}^{w-v} ]`.
    For very large momentum grids, this function is the slowest part compared to the rest of the code due to the
    repeated loops. Potential speed-ups could be achieved by batching the q-points or using numba.
    """
    mat = np.zeros(
        (*config.lattice.k_grid.nk, config.sys.n_bands, config.sys.n_bands, config.box.niv_core),
        dtype=kernel.mat.dtype,
    )

    kernel = kernel.to_full_niw_range()
    wn = MFHelper.wn(config.box.niw_core)
    path = np.einsum_path("aijdv,xyzadv->xyzijv", kernel[0, ..., 0, :], mat, optimize=True)[1]

    for idx_q, q in enumerate(my_full_q_list):
        shifted_mat = np.roll(giwk.mat, [-i for i in q], axis=(0, 1, 2))
        for idx_w, wn_i in enumerate(wn):
            g_qk = shifted_mat[..., giwk.niv - wn_i : giwk.niv + config.box.niv_core - wn_i]
            k_slice = kernel[idx_q, ..., idx_w, config.box.niv_core :]
            mat += np.einsum("aijdv,xyzadv->xyzijv", k_slice, g_qk, optimize=path)

    mat *= -0.5 / config.sys.beta / config.lattice.q_grid.nk_tot
    return SelfEnergy(mat, config.lattice.nk, False).compress_q_dimension().to_full_niv_range()


def calculate_sigma_from_kernel_cpu(
    kernel: FourPoint,
    giwk: GreensFunction,
    my_full_q_list: np.ndarray,
) -> SelfEnergy:
    r"""
    Returns :math:`\Sigma_{ij}^{k} = -1/2 * 1/\beta * 1/N_q \sum_q [ U^q_{r;aibc} * K_{r;cbjd}^{qv} * G_{ad}^{w-v} ]`.
    For very large momentum grids, this function is the slowest part compared to the rest of the code due to the
    repeated loops. There is no real way to speed it up further without leveraging GPUs or other hardware accelerators.
    """
    nkx, nky, nkz = config.lattice.k_grid.nk
    nb = config.sys.n_bands
    niv_core = config.box.niv_core

    mat = np.zeros((nkx, nky, nkz, nb, nb, niv_core), dtype=kernel.mat.dtype)
    wn = MFHelper.wn(config.box.niw_core)

    giwk_mat = np.asfortranarray(giwk.mat)
    kernel = np.asfortranarray(kernel.to_full_niw_range().mat[..., niv_core:])

    kxs, kys, kzs = np.arange(nkx), np.arange(nky), np.arange(nkz)
    kx_indices = [((kxs + q[0]) % nkx) for q in my_full_q_list]
    ky_indices = [((kys + q[1]) % nky) for q in my_full_q_list]
    kz_indices = [((kzs + q[2]) % nkz) for q in my_full_q_list]

    acc = np.empty((nkx, nky, nkz, nb, nb, niv_core), dtype=mat.dtype)

    for iq in range(len(my_full_q_list)):
        g_q_view = giwk_mat[
            kx_indices[iq][:, None, None], ky_indices[iq][None, :, None], kz_indices[iq][None, None, :], ...
        ]

        for iw, w in enumerate(wn):
            g_slice = g_q_view[..., giwk.niv - w : giwk.niv + niv_core - w]
            k_slice = kernel[iq, ..., iw, :]
            np.einsum("xyzadv,aijdv->xyzijv", g_slice, k_slice, out=acc, optimize=True)
            np.add(mat, acc, out=mat)

    mat *= -0.5 / config.sys.beta / config.lattice.q_grid.nk_tot
    return SelfEnergy(np.ascontiguousarray(mat), config.lattice.nk, False).compress_q_dimension().to_full_niv_range()


def calculate_sigma_from_kernel_gpu(
    kernel: FourPoint,
    giwk: GreensFunction,
    my_full_q_list: np.ndarray,
) -> SelfEnergy:
    r"""
    Returns :math:`\Sigma_{ij}^{k} = -1/2 * 1/\beta * 1/N_q \sum_q [ U^q_{r;aibc} * K_{r;cbjd}^{qv} * G_{ad}^{w-v} ]`.
    For very large momentum grids, this function is the slowest part compared to the rest of the code due to the
    repeated loops. This function tries to execute it on the GPU using CuPy.
    """
    import cupy as cp

    nkx, nky, nkz = config.lattice.k_grid.nk
    nb = config.sys.n_bands
    niv_core = config.box.niv_core

    mat_gpu = cp.zeros((nkx, nky, nkz, nb, nb, niv_core), dtype=kernel.mat.dtype, order="F")
    wn = MFHelper.wn(config.box.niw_core)

    giwk_mat = cp.asarray(giwk.mat, order="F")
    kernel = cp.asarray(kernel.to_full_niw_range().mat, order="F")[..., niv_core:]

    kxs, kys, kzs = cp.arange(nkx), cp.arange(nky), cp.arange(nkz)
    kx_indices = [((kxs + q[0]) % nkx) for q in my_full_q_list]
    ky_indices = [((kys + q[1]) % nky) for q in my_full_q_list]
    kz_indices = [((kzs + q[2]) % nkz) for q in my_full_q_list]

    for iq in range(len(my_full_q_list)):
        g_q_view = giwk_mat[
            kx_indices[iq][:, None, None], ky_indices[iq][None, :, None], kz_indices[iq][None, None, :], ...
        ]

        for iw, w in enumerate(wn):
            g_slice = g_q_view[..., giwk.niv - w : giwk.niv + niv_core - w]
            k_slice = kernel[iq, ..., iw, :]
            mat_gpu += cp.einsum("xyzadv,aijdv->xyzijv", g_slice, k_slice, optimize=True)

    mat_gpu *= -0.5 / config.sys.beta / config.lattice.q_grid.nk_tot
    return (
        SelfEnergy(np.ascontiguousarray(cp.asnumpy(mat_gpu)), config.lattice.nk, False)
        .compress_q_dimension()
        .to_full_niv_range()
    )


def calculate_sigma_from_kernel_auto(
    mpi_distributor: MpiDistributor, kernel: FourPoint, giwk: GreensFunction, my_full_q_list: np.ndarray
) -> SelfEnergy:
    """
    Automatically tries to calculate the self-energy from the kernel on the GPU using CuPy. If CuPy is not installed
    or no GPU is available, it falls back to the CPU implementation.
    """
    logger = config.logger

    try:
        import cupy as cp

        n_gpus = cp.cuda.runtime.getDeviceCount()

        if cp.cuda.is_available() and n_gpus > 0:
            logger.info(f"CuPy detected {n_gpus} GPU(s). Using GPU acceleration for self-energy calculation.")

            gpu_id = mpi_distributor.my_rank % n_gpus
            cp.cuda.Device(gpu_id).use()
            return calculate_sigma_from_kernel_gpu(kernel, giwk, my_full_q_list)
    except:
        # CuPy not installed or device could not be found
        pass

    return calculate_sigma_from_kernel_cpu(kernel, giwk, my_full_q_list)


def calculate_sigma_from_kernel_fft_cpu(
    mpi_dist: MpiDistributor, kernel: FourPoint, giwk: GreensFunction
) -> SelfEnergy:
    """
    Optimized Sigma calculation using Distributed FFTs.
    Replaces the iq-loop with a real-space pointwise multiplication.
    Returns Sigma in R-space, positive-v half only; caller must ifft over (kx,ky,kz)
    and then call .to_full_niv_range() before using.
    """
    comm = mpi_dist.comm
    rank = comm.Get_rank()
    size = comm.Get_size()
    nkx, nky, nkz = config.lattice.k_grid.nk
    nk_tot = config.lattice.q_grid.nk_tot
    nb = config.sys.n_bands
    niv_core = config.box.niv_core
    niw = config.box.niw_core
    beta = config.sys.beta

    # G(k) -> F[G](R), forward FFT, replicated on every rank
    g_r_mat = giwk.fft().mat

    # K(q) -> F[K](-R) via the conjugate trick: conj, fft, conj.
    kernel = kernel.to_full_niw_range().to_half_niv_range()
    kernel.mat = np.conj(kernel.mat)
    kernel = mpi_utils.execute_distributed_fft(kernel, comm)
    kernel.mat = np.conj(kernel.mat)

    # Local R-space contraction; each rank owns a slice of R-points
    n_r_local = kernel.mat.shape[0]
    mat = np.zeros((n_r_local, nb, nb, niv_core), dtype=kernel.mat.dtype)
    acc = np.empty_like(mat)

    my_r_indices = mpi_utils.get_pencil_indices(rank, size, (nkx, nky, nkz), "flat")
    g_r_local = g_r_mat.reshape(nk_tot, nb, nb, -1)[my_r_indices]

    wn = MFHelper.wn(niw)
    for iw, w in enumerate(wn):
        g_slice = g_r_local[..., giwk.niv - w : giwk.niv + niv_core - w]
        k_slice = kernel.mat[..., iw, :]
        np.einsum("Radv,Raijdv->Rijv", g_slice, k_slice, out=acc, optimize=True)
        np.add(mat, acc, out=mat)

    mat *= -0.5 / beta / nk_tot
    return SelfEnergy(
        np.ascontiguousarray(mat),
        config.lattice.nk,
        full_niv_range=False,
        has_compressed_q_dimension=True,
        calc_smom=False,
    )


def calculate_sigma_from_kernel_fft_gpu(
    mpi_dist: MpiDistributor, kernel: FourPoint, giwk: GreensFunction
) -> SelfEnergy:
    """
    Optimized Sigma calculation using Distributed FFTs, running on GPUs.
    Replaces the iq-loop with a real-space pointwise multiplication.
    Returns Sigma in R-space, positive-v half only; caller must ifft over (kx,ky,kz)
    and then call .to_full_niv_range() before using.
    """
    import cupy as cp

    comm = mpi_dist.comm
    rank = comm.Get_rank()
    size = comm.Get_size()
    nkx, nky, nkz = config.lattice.k_grid.nk
    nk_tot = config.lattice.q_grid.nk_tot
    nb = config.sys.n_bands
    niv_core = config.box.niv_core
    niw = config.box.niw_core
    beta = config.sys.beta

    # G(k) -> F[G](R), forward FFT, replicated on every rank
    g_r_mat = cp.asarray(giwk.fft().mat)

    # K(q) -> F[K](-R) via the conjugate trick: conj, fft, conj.
    kernel = kernel.to_full_niw_range().to_half_niv_range()
    kernel.mat = np.conj(kernel.mat)
    kernel = mpi_utils.execute_distributed_fft(kernel, comm)
    kernel.mat = cp.conj(cp.asarray(kernel.mat))

    # Local R-space contraction; each rank owns a slice of R-points
    n_r_local = kernel.mat.shape[0]
    mat = cp.zeros((n_r_local, nb, nb, niv_core), dtype=kernel.mat.dtype)

    my_r_indices = mpi_utils.get_pencil_indices(rank, size, (nkx, nky, nkz), "flat")
    g_r_local = g_r_mat.reshape(nk_tot, nb, nb, -1)[cp.asarray(my_r_indices)]

    wn = MFHelper.wn(niw)
    for iw, w in enumerate(wn):
        g_slice = g_r_local[..., giwk.niv - w : giwk.niv + niv_core - w]
        k_slice = kernel.mat[..., iw, :]
        mat += cp.einsum("Radv,Raijdv->Rijv", g_slice, k_slice, optimize=True)

    mat *= -0.5 / beta / nk_tot
    return SelfEnergy(
        np.ascontiguousarray(cp.asnumpy(mat)),
        config.lattice.nk,
        full_niv_range=False,
        has_compressed_q_dimension=True,
        calc_smom=False,
    )


def calculate_sigma_from_kernel_fft_auto(
    mpi_distributor: MpiDistributor, kernel: FourPoint, giwk: GreensFunction
) -> SelfEnergy:
    """
    Automatically tries to calculate the self-energy from the kernel on the GPU using CuPy. If CuPy is not installed
    or no GPU is available, it falls back to the CPU implementation.
    """
    logger = config.logger

    try:
        import cupy as cp

        n_gpus = cp.cuda.runtime.getDeviceCount()

        if cp.cuda.is_available() and n_gpus > 0:
            logger.info(f"CuPy detected {n_gpus} GPU(s). Using GPU acceleration for self-energy calculation.")

            gpu_id = mpi_distributor.my_rank % n_gpus
            cp.cuda.Device(gpu_id).use()
            return calculate_sigma_from_kernel_fft_gpu(mpi_distributor, kernel, giwk)
    except:
        # CuPy not installed or device could not be found
        pass

    return calculate_sigma_from_kernel_fft_cpu(mpi_distributor, kernel, giwk)


def get_starting_sigma(output_path: str, default_sigma: SelfEnergy) -> tuple[SelfEnergy, int]:
    """
    If the output directory is specified to be the same directory as was used by a previous calculation, we try to
    retrieve the last calculated self-energy as a starting point for the next calculation. If no sigma_dga_N.npy file
    is found, we return the dmft self-energy as a starting point.
    """
    if output_path == "" or output_path is None or not os.path.exists(output_path):
        return default_sigma, 0

    files = glob.glob(os.path.join(output_path, "sigma_dga_iteration_*.npy"))
    if not files:
        return default_sigma, 0

    iterations = [int(match.group(1)) for f in files if (match := re.search(r"sigma_dga_iteration_(\d+)\.npy$", f))]
    if not iterations:
        return default_sigma, 0

    max_iter = max(iterations)
    mat = np.load(os.path.join(output_path, f"sigma_dga_iteration_{max_iter}.npy"))
    return SelfEnergy(mat, config.lattice.nk, True, True, False), max_iter


def read_last_n_sigmas_from_files(n: int, output_path: str = "./", previous_sc_path: str = "./") -> list[np.ndarray]:
    """
    Reads the last n total self-energies from the output directory and - if specified - the previous self-consistency
    path. This is used for the predictive Pulay-mixing scheme. If one has a history of self-energies from a previous
    calculation, these will be used as well.
    """
    files_output_dir = glob.glob(os.path.join(output_path, "sigma_dga_iteration_*.npy"))
    if previous_sc_path != "" and previous_sc_path is not None and os.path.exists(previous_sc_path):
        files_prev_sc_dir = glob.glob(os.path.join(previous_sc_path, "sigma_dga_iteration_*.npy"))
    else:
        files_prev_sc_dir = []
    files = files_output_dir + files_prev_sc_dir

    last_iterations = sorted(
        [(int(match.group(1)), f) for f in files if (match := re.search(r"sigma_dga_iteration_(\d+)\.npy$", f))],
        key=lambda x: x[0],
    )[-n:]
    return [np.load(file) for _, file in last_iterations]


def calculate_self_energy_q(
    comm: MPI.Comm, u_loc: LocalInteraction, v_nonloc: Interaction, sigma_dmft: SelfEnergy, sigma_local: SelfEnergy
) -> SelfEnergy:
    """
    Main routine for the non-local DGA self-energy calculation. Calculates the Hartree- and Fock-terms, the bubble,
    the double-counting correction and the kernel in the density and magnetic channel. Finally, calculates the
    non-local self-energy from the kernel and the Green's function. Also takes care of the self-consistency loop and
    the chemical potential adjustment as well as the self-energy mixing, etc.
    """
    logger = config.logger

    logger.info("Starting with non-local DGA routine.")
    logger.info("Initializing MPI distributor.")

    # MPI distributor for the irreducible BZ
    mpi_dist_irrk = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q")
    full_q_list = config.lattice.q_grid.get_q_list()
    irrk_q_list = config.lattice.q_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    mpi_dist_fullbz = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_tot, comm=comm, name="FBZ")
    my_full_q_list = full_q_list[mpi_dist_fullbz.my_slice]

    sigma_old, starting_iter = get_starting_sigma(config.self_consistency.previous_sc_path, sigma_dmft)
    if starting_iter > 0:
        logger.info(
            f"Using previous calculation and starting the self-consistency loop at iteration {starting_iter + 1}."
        )

    mu_history = (
        [config.sys.mu]
        if starting_iter == 0
        else [float(np.load(os.path.join(config.self_consistency.previous_sc_path, "mu_history.npy"))[-1])]
    )

    niv_cut = min(config.box.niw_core + config.box.niv_full + 10, config.box.niv_dmft)
    sigma_dmft_full = deepcopy(sigma_dmft)

    if comm.rank == 0:
        sigma_old = sigma_old.concatenate_self_energies(sigma_dmft_full)
        giwk_full = GreensFunction.get_g_full(sigma_old, mu_history[-1], config.lattice.hamiltonian.get_ek())
        config.sys.n, config.sys.occ, config.sys.occ_k = giwk_full.get_fill_nonlocal()
        giwk_full.cut_niv(niv_cut)

        if sigma_old is sigma_dmft:
            giwk_full.save(output_dir=config.output.output_path, name="g_latt_dmft")
    config.sys.n, config.sys.occ, config.sys.occ_k = comm.bcast(
        (config.sys.n, config.sys.occ, config.sys.occ_k), root=0
    )

    sigma_old = sigma_old.cut_niv(niv_cut)
    sigma_dmft = sigma_dmft.cut_niv(niv_cut)

    delta_sigma = sigma_dmft.cut_niv(config.box.niv_core) - sigma_local.cut_niv(config.box.niv_core)

    v_nonloc_full = deepcopy(v_nonloc)
    v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    for current_iter in range(starting_iter + 1, starting_iter + config.self_consistency.max_iter + 1):
        logger.info("----------------------------------------")
        logger.info(f"Starting iteration {current_iter}.")
        logger.info("----------------------------------------")

        hartree, fock = get_hartree_fock(u_loc, v_nonloc_full, my_full_q_list)
        fock = mpi_dist_fullbz.allreduce(fock)
        logger.info("Calculated Hartree and Fock terms.")

        giwk_full = GreensFunction.get_g_full(sigma_old, mu_history[-1], config.lattice.hamiltonian.get_ek())

        logger.log_memory_usage("giwk", giwk_full, comm.size)
        if config.memory.save_memory_for_chi0q:
            gchi0_q = BubbleGenerator.create_generalized_chi0_q_auto(
                mpi_dist_irrk,
                giwk_full,
                config.box.niw_core,
                config.box.niv_full,
                my_irr_q_list,
                config.lattice.q_grid,
                config.sys.beta,
                config.logger,
            )
        else:
            gchi0_q = BubbleGenerator.create_generalized_chi0_q_fft_auto(
                mpi_dist_irrk,
                giwk_full,
                config.box.niw_core,
                config.box.niv_full,
                config.lattice.k_grid,
                config.sys.beta,
                config.logger,
            )

        if config.eliashberg.perform_eliashberg:
            gchi0_q.save(name=f"gchi0_q_rank_{comm.rank}", output_dir=config.output.output_path)

        logger.log_memory_usage("Gchi0_q_full", gchi0_q, comm.size)
        giwk_full = giwk_full.cut_niv(niv_cut)

        f_dc_loc = 2 * LocalFourPoint.load(os.path.join(config.output.output_path, "f_magn_loc.npy")).permute_orbitals(
            "abcd->cbad"
        )
        kernel = -calculate_sigma_dc_kernel(f_dc_loc, gchi0_q, u_loc)
        f_dc_loc.free()
        logger.info("Calculated double-counting kernel.")

        gchi0_q_full_sum = 1.0 / config.sys.beta * gchi0_q.sum_over_all_vn(config.sys.beta)
        gchi0_q_core = gchi0_q.cut_niv(config.box.niv_core)
        gchi0_q.free()
        logger.log_memory_usage("Gchi0_q_core", gchi0_q_core, comm.size)

        gchi0_q_core_inv = deepcopy(gchi0_q_core).invert(False)
        logger.log_memory_usage("Gchi0_q_inv", gchi0_q_core_inv, comm.size)

        if config.eliashberg.perform_eliashberg:
            gchi0_q_core_inv.save(name=f"gchi0_q_inv_rank_{comm.rank}", output_dir=config.output.eliashberg_path)

        gchi0_q_core_sum = 1.0 / config.sys.beta * gchi0_q_core.sum_over_all_vn(config.sys.beta)
        gchi0_q_core.free()

        gamma_dens = LocalFourPoint.load(
            os.path.join(config.output.output_path, "gamma_dens_loc.npy"), SpinChannel.DENS
        )
        kernel += calculate_sigma_kernel_r_q(
            gamma_dens, gchi0_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc, mpi_dist_irrk
        )
        gamma_dens.free()
        mpi_dist_irrk.barrier()
        logger.info("Calculated kernel for density channel.")

        gamma_magn = LocalFourPoint.load(
            os.path.join(config.output.output_path, "gamma_magn_loc.npy"), SpinChannel.MAGN
        )
        kernel += 3 * calculate_sigma_kernel_r_q(
            gamma_magn, gchi0_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc, mpi_dist_irrk
        )
        gchi0_q_core_inv.free()
        gchi0_q_full_sum.free()
        gchi0_q_core_sum.free()
        gamma_magn.free()
        logger.info("Calculated kernel for magnetic channel.")

        if not config.memory.save_memory_for_chiq_aux:
            kernel = mpi_utils.map_irrbz_fullbz(kernel, mpi_dist_irrk, mpi_dist_fullbz)
            logger.info("Kernel mapped to full BZ and scattered across all MPI ranks.")
        else:
            kernel = mpi_utils.exchange_and_map_irrbz_fullbz(kernel, mpi_dist_irrk, mpi_dist_fullbz)
            logger.info("Kernel mapped to full BZ individually on each rank.")

        logger.info("Started calculation of DGA self-energy.")

        if config.memory.save_memory_for_sde:
            sigma_new = calculate_sigma_from_kernel_auto(mpi_dist_fullbz, kernel, giwk_full, my_full_q_list)
            kernel.free()
            sigma_new.mat = mpi_dist_fullbz.allreduce(sigma_new.mat)
        else:
            sigma_new = calculate_sigma_from_kernel_fft_auto(mpi_dist_irrk, kernel, giwk_full)
            kernel.free()
            sigma_new.mat = mpi_dist_fullbz.gather(sigma_new.mat)
            if comm.rank == 0:
                sigma_new = sigma_new.ifft().to_full_niv_range()
            sigma_new = mpi_dist_fullbz.bcast(sigma_new)

        logger.info("Self-energy calculated from kernel.")
        logger.log_memory_usage("Non-local sigma", sigma_new, comm.size)

        sigma_new = sigma_new + hartree + fock
        logger.info("Full non-local self-energy calculated.")

        # This is done to minimize noise. We remove some fluctuations from dmft that are included in the local self-energy
        # calculated in this code and add the smooth dmft self-energy
        sigma_new += delta_sigma
        sigma_new = sigma_new.concatenate_self_energies(sigma_dmft)

        logger.info("Applying mixing strategy to the self-energy.")
        sigma_new = apply_mixing_strategy(sigma_new, sigma_old, sigma_dmft, current_iter)

        old_mu = mu_history[-1]
        if comm.rank == 0:
            mu_finding_failed = False
            new_mu = update_mu(
                old_mu, config.sys.n, giwk_full.ek, sigma_new.mat, config.sys.beta, sigma_new.fit_smom()[0]
            )

            if new_mu is np.nan:
                mu_finding_failed = True

            # will not be changed if mu finding failed
            if not mu_finding_failed:
                config.sys.mu = config.self_consistency.mixing * new_mu + (1 - config.self_consistency.mixing) * old_mu

        config.sys.mu = comm.bcast(config.sys.mu)
        mu_history.append(config.sys.mu)
        logger.info(f"Updated mu from {old_mu} to {config.sys.mu}.")

        if comm.rank == 0:
            sigma_occ = deepcopy(sigma_new).concatenate_self_energies(sigma_dmft_full)
            giwk_occ = giwk_full.get_g_full(sigma_occ, config.sys.mu, config.lattice.hamiltonian.get_ek())
            # calculate new occupation matrix from new Green's function (outside asympt region it is the DMFT
            # lattice Green's function)
            _, config.sys.occ, config.sys.occ_k = giwk_occ.get_fill_nonlocal()  # n should not change
        config.sys.occ, config.sys.occ_k = comm.bcast((config.sys.occ, config.sys.occ_k), root=0)
        if config.self_consistency.max_iter > 1:
            logger.info("Updated occupation matrix with new Green's function.")

        if comm.rank == 0:
            sigma_new.save(name=f"sigma_dga_iteration_{current_iter}", output_dir=config.output.output_path)
            logger.info(f"Saved sigma for iteration {current_iter} as numpy array.")

            if config.self_energy_interpolation.do_interpolation:
                beta_target = config.self_energy_interpolation.beta_target
                niv_target = config.self_energy_interpolation.niv_target
                beta_source = config.sys.beta
                sigma_new.interpolate(beta_source, beta_target, niv_target).save(
                    name=f"sigma_dga_interpolated_beta{beta_target}_niv{niv_target}_iteration_{current_iter}",
                    output_dir=config.output.output_path,
                )

        logger.info("Checking self-consistency convergence.")
        if comm.rank == 0 and current_iter > starting_iter + 1:
            niv_start = sigma_new.niv
            niv_end = niv_start + int(np.ceil(config.box.niv_core / 5))

            sigma_converged = np.allclose(
                sigma_old.compress_q_dimension()[..., niv_start:niv_end],
                sigma_new.compress_q_dimension()[..., niv_start:niv_end],
                atol=config.self_consistency.epsilon,
            )
            logger.info(f"Self-energy convergence: {sigma_converged}.")

            mu_converged = (
                abs(mu_history[-1] - mu_history[-2]) < np.pi / (10 * config.sys.beta)
            ) and not mu_finding_failed
            logger.info(f"Chemical potential convergence: {mu_converged}.")

            converged = mu_converged and sigma_converged
        else:
            converged = False
        converged = comm.bcast(converged)

        sigma_old = sigma_new
        if converged:
            if config.self_consistency.restrict_chi_phys:
                logger.info(
                    "ATTENTION: Self-consistency with restricted susceptibility reached. Disabling restriction."
                )
                config.self_consistency.restrict_chi_phys = False
            else:
                logger.info(f"Self-consistency of sigma and mu reached at iteration {current_iter}.")
                break
        logger.info("Self-consistency not reached.")

    mpi_dist_irrk.delete_file()
    mpi_dist_fullbz.delete_file()

    np.save(os.path.join(config.output.output_path, "mu_history.npy"), mu_history)
    logger.info("Saved mu history as numpy array.")

    return sigma_old


def apply_mixing_strategy(
    sigma_new: SelfEnergy, sigma_old: SelfEnergy, sigma_dmft: SelfEnergy, current_iter: int
) -> SelfEnergy:
    """
    Applies the mixing strategy for the self-consistency loop. The mixing strategy is defined in the config file and
    is either 'linear' or 'pulay'.
    """
    logger = config.logger
    n_hist = config.self_consistency.mixing_history_length
    alpha = config.self_consistency.mixing

    if config.self_consistency.mixing_strategy.lower() == "pulay" and current_iter > n_hist:
        last_results = read_last_n_sigmas_from_files(
            n_hist, config.output.output_path, config.self_consistency.previous_sc_path
        )
        sigma_dmft_stacked = np.tile(sigma_dmft.mat, (config.lattice.k_grid.nk_tot, 1, 1, 1))
        last_proposals = [sigma_dmft_stacked] + last_results
        last_results = last_results + [sigma_new.mat]

        niv = sigma_new.current_shape[-1] // 2
        niv_core = config.box.niv_core
        last_proposals = [sigma[..., niv - niv_core : niv + niv_core] for sigma in last_proposals]
        last_results = [sigma[..., niv - niv_core : niv + niv_core] for sigma in last_results]
        logger.info(f"Loaded last {n_hist} self-energies from files.")

        shape = last_results[-1].shape
        n_total = int(np.prod(shape))
        r_matrix = np.zeros((2 * n_total, n_hist), dtype=np.float64)
        f_matrix = np.zeros_like(r_matrix)
        f_i = np.zeros((2 * n_total), dtype=np.float64)

        def get_proposal(idx: int):
            return last_proposals[idx].flatten()

        def get_result(idx: int):
            return last_results[idx].flatten()

        for i in range(n_hist):
            proposal_diff = get_proposal(-1 - i) - get_proposal(-2 - i)
            r_matrix[:n_total, i] = proposal_diff.real
            r_matrix[n_total:, i] = proposal_diff.imag

            result_diff = get_result(-1 - i) - get_result(-2 - i)
            f_matrix[:n_total, i] = result_diff.real
            f_matrix[n_total:, i] = result_diff.imag

            f_matrix[:, i] -= r_matrix[:, i]

        # Residual: F(x_n) - x_n, where x_n = last_proposals[-1] = sigma_old (core window)
        iter_diff = get_result(-1) - get_proposal(-1)
        f_i[:n_total] = iter_diff.real
        f_i[n_total:] = iter_diff.imag
        norm_f = np.linalg.norm(f_i)
        norm_x = np.linalg.norm(get_proposal(-1))
        rel_res = norm_f / norm_x if norm_x > 0 else np.inf

        # Solve min||F @ c - f_i|| via truncated-SVD pseudoinverse (drops collinear directions)
        u, s, vh = np.linalg.svd(f_matrix, full_matrices=False)
        cutoff = 1e-5 * (s[0] if s.size else 1.0)
        mask = s > cutoff
        if not np.any(mask):
            logger.warning("Pulay SVD ill-conditioned - falling back to linear mixing.")
            return alpha * sigma_new + (1 - alpha) * sigma_old
        coeffs = vh[mask].T @ ((u[:, mask].T @ f_i) / s[mask])

        # Pulay update: x_{n+1} = x_n + alpha*f_i - (R + alpha*F) @ c
        update = alpha * f_i - (r_matrix + alpha * f_matrix) @ coeffs
        norm_u = np.linalg.norm(update)
        if norm_f > 0 and norm_u > 10.0 * norm_f:
            update *= 10.0 * norm_f / norm_u
            logger.warning(f"Pulay step clamped (norm_u={norm_u:.3e}, norm_f={norm_f:.3e}).")
        update = update[:n_total] + 1j * update[n_total:]

        # Update the new self energy
        sigma_new.mat[..., niv - niv_core : niv + niv_core] = get_proposal(-1).reshape(shape) + update.reshape(shape)

        logger.info(
            f"Pulay mixing applied (m={n_hist}, alpha={alpha:.3f}, norm_f={norm_f:.3e}, rel_res={rel_res:.3e})."
        )

        return sigma_new
    if config.self_consistency.mixing_strategy.lower() == "anderson" and current_iter > n_hist:
        last_sigmas = read_last_n_sigmas_from_files(
            n_hist, config.output.output_path, config.self_consistency.previous_sc_path
        )

        niv = sigma_new.current_shape[-1] // 2
        niv_core = config.box.niv_core
        sl = slice(niv - niv_core, niv + niv_core)

        sigma_dmft_stacked = np.tile(sigma_dmft.mat, (config.lattice.k_grid.nk_tot, 1, 1, 1))

        last_proposals = [sigma_dmft_stacked] + last_sigmas  # [dmft, s1, ..., s_{n-1}]
        last_results = last_sigmas + [sigma_new.mat]  # [s1,  s2, ..., s_new]
        last_proposals = [s[..., sl] for s in last_proposals]
        last_results = [s[..., sl] for s in last_results]

        shape = last_results[-1].shape
        n_total = int(np.prod(shape))
        flat = lambda x: x.reshape(-1)

        # Current residual f_n = F(x_n) - x_n
        f_curr = flat(last_results[-1]) - flat(last_proposals[-1])
        f_vec = np.concatenate([f_curr.real, f_curr.imag])
        norm_f = np.linalg.norm(f_vec)
        x_curr = flat(last_proposals[-1])
        norm_x = np.linalg.norm(np.concatenate([x_curr.real, x_curr.imag]))
        rel_res = norm_f / norm_x if norm_x > 0 else np.inf

        # Build dX and dF matrices (n_hist columns)
        # dX[:,i] = x_{n-i} - x_{n-i-1}  (proposal differences)
        # dF[:,i] = f_{n-i} - f_{n-i-1}  (residual differences)
        dx_cols = []
        df_cols = []
        for i in range(n_hist):
            dx = flat(last_proposals[-1 - i]) - flat(last_proposals[-2 - i])
            dx_cols.append(np.concatenate([dx.real, dx.imag]))

            df_i = flat(last_results[-1 - i]) - flat(last_proposals[-1 - i])
            df_im1 = flat(last_results[-2 - i]) - flat(last_proposals[-2 - i])
            df = df_i - df_im1
            df_cols.append(np.concatenate([df.real, df.imag]))

        dx_matrix = np.column_stack(dx_cols)  # (2*n_total, n_hist)
        df_matrix = np.column_stack(df_cols)  # (2*n_total, n_hist)

        # Anderson: solve min ||f_curr - dF @ c||
        try:
            u, s, vh = np.linalg.svd(df_matrix, full_matrices=False)

            s_max = s[0] if len(s) > 0 else 1.0
            cutoff = 1e-5 * s_max
            mask = s > cutoff

            if not np.any(mask):
                raise np.linalg.LinAlgError("All singular values below threshold.")

            s_reg = s[mask] / (s[mask] ** 2 + cutoff**2)
            coeffs = vh[mask].T @ (s_reg * (u[:, mask].T @ f_vec))

        except np.linalg.LinAlgError:
            logger.warning("Anderson SVD failed - falling back to linear mixing.")
            return alpha * sigma_new + (1 - alpha) * sigma_old

        # Undamped Anderson proposal: x_n + f_n - (dX + dF) @ c
        x_n = flat(last_proposals[-1])
        x_anderson = np.concatenate([x_n.real, x_n.imag]) + f_vec - (dx_matrix + df_matrix) @ coeffs
        x_anderson = x_anderson[:n_total] + 1j * x_anderson[n_total:]

        # Damp between old proposal and Anderson proposal
        x_n_complex = x_n
        candidate = (1 - alpha) * x_n_complex + alpha * x_anderson.reshape(-1)

        # Safety clamp
        update = candidate - x_n_complex
        norm_u = np.linalg.norm(update)
        if norm_f > 0 and norm_u > 3.0 * norm_f:
            candidate = x_n_complex + update * (3.0 * norm_f / norm_u)
            logger.warning(f"Anderson step clamped (norm_u={norm_u:.3e}, norm_f={norm_f:.3e}).")

        sigma_new.mat[..., sl] = candidate.reshape(shape)

        logger.info(
            f"Anderson acceleration applied (m={n_hist}, alpha={alpha:.3f}, norm_f={norm_f:.3e}, rel_res={rel_res:.3e})."
        )

        return sigma_new

    sigma_new = config.self_consistency.mixing * sigma_new + (1 - config.self_consistency.mixing) * sigma_old
    logger.info(
        f"Sigma linearly mixed with previous iteration using a mixing parameter of {config.self_consistency.mixing}."
    )
    return sigma_new

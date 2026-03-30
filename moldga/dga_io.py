# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os

import moldga.brillouin_zone as bz
import moldga.config as config
import moldga.w2dyn_aux as w2dyn_aux
from moldga.greens_function import GreensFunction
from moldga.hamiltonian import Hamiltonian
from moldga.local_four_point import LocalFourPoint
from moldga.n_point_base import *
from moldga.self_energy import SelfEnergy


def uniquify_path(path: str = None):
    """
    :param path: Path to be checked for uniqueness
    :return: Updated unique path
    """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path


def load_from_w2dyn_file_and_update_config(
    combine_two_atoms_to_one_obj: bool = False,
) -> tuple[GreensFunction, SelfEnergy, LocalFourPoint, LocalFourPoint]:
    """
    Loads data from the w2dyn file and updates the config file.
    If combine_atoms_to_one_obj is True, we are doubling the orbital space and putting the data from the
    inequivalent atoms into the diagonal blocks after we average over them.
    """
    file = w2dyn_aux.W2dynFile(fname=str(os.path.join(config.dmft.input_path, config.dmft.fname_1p)))

    config.sys.beta = file.get_beta()

    config.lattice.interaction.udd = file.get_udd()
    config.lattice.interaction.udp = file.get_udp()
    config.lattice.interaction.upp = file.get_upp()
    config.lattice.interaction.uppod = file.get_uppod()
    config.lattice.interaction.jdd = file.get_jdd()
    config.lattice.interaction.jdp = file.get_jdp()
    config.lattice.interaction.jpp = file.get_jpp()
    config.lattice.interaction.jppod = file.get_jppod()
    config.lattice.interaction.vdd = file.get_vdd()
    config.lattice.interaction.vpp = file.get_vpp()

    config.sys.mu = file.get_mu()
    config.sys.nd_bands = file.get_nd()
    config.sys.np_bands = file.get_np()
    config.sys.n_bands = config.sys.nd_bands + config.sys.np_bands
    config.sys.n = file.get_totdens()
    config.sys.occ_dmft = 2 * np.mean(file.get_rho1(), axis=(1, 3))

    if combine_two_atoms_to_one_obj:
        config.sys.n_bands *= 2
        config.sys.nd_bands *= 2
        config.sys.np_bands *= 2
        config.sys.n *= 2

        config.sys.occ_dmft = np.zeros((config.sys.n_bands, config.sys.n_bands))
        rho_1_mean = 0.5 * (np.mean(file.get_rho1(), axis=(1, 3)) + np.mean(file.get_rho1(ineq=2), axis=(1, 3)))
        config.sys.occ_dmft[0:2, 0:2] = 2 * rho_1_mean
        config.sys.occ_dmft[2:4, 2:4] = 2 * rho_1_mean

    if config.sys.n == 0:
        config.sys.n = 2 * np.sum(config.sys.occ_dmft)

    file2 = w2dyn_aux.W2dynG4iwFile(fname=str(os.path.join(config.dmft.input_path, config.dmft.fname_2p)))
    g2_dens = LocalFourPoint(
        file2.read_g2_full_multiband(file.get_nd() + file.get_np(), name="dens"), channel=SpinChannel.DENS
    )
    g2_magn = LocalFourPoint(
        file2.read_g2_full_multiband(file.get_nd() + file.get_np(), name="magn"), channel=SpinChannel.MAGN
    )

    if combine_two_atoms_to_one_obj:
        g2_dens_2 = LocalFourPoint(
            file2.read_g2_full_multiband(file.get_nd() + file.get_np(), ineq=2, name="dens"), channel=SpinChannel.DENS
        )
        g2_magn_2 = LocalFourPoint(
            file2.read_g2_full_multiband(file.get_nd() + file.get_np(), ineq=2, name="magn"), channel=SpinChannel.MAGN
        )

        def construct_large_g2(g2_1: LocalFourPoint, g2_2: LocalFourPoint) -> LocalFourPoint:
            g2_mean = 0.5 * (g2_1.mat + g2_2.mat)
            del g2_2
            g2_1.mat = np.zeros((4, 4, 4, 4, *g2_mean.shape[4:]))
            g2_1.mat[0:2, 0:2, 0:2, 0:2] = g2_mean
            g2_1.mat[2:4, 2:4, 2:4, 2:4] = g2_mean
            del g2_mean
            return g2_1

        g2_dens = construct_large_g2(g2_dens, g2_dens_2)
        g2_magn = construct_large_g2(g2_magn, g2_magn_2)

    file2.close()

    update_frequency_boxes(g2_dens.niw, g2_dens.niv)

    def extend_orbital(arr: np.ndarray) -> np.ndarray:
        return np.einsum("i...,ij->ij...", arr, np.eye(arr.shape[0]))

    giw_spin_mean = np.mean(file.get_giw(), axis=1)  # [band,spin,niv]
    niv_dmft = giw_spin_mean.shape[-1] // 2
    niv_cut = config.box.niw_core + config.box.niv_full + 10
    giw_spin_mean = giw_spin_mean[..., niv_dmft - niv_cut : niv_dmft + niv_cut]
    g_dmft = GreensFunction(extend_orbital(giw_spin_mean))

    if combine_two_atoms_to_one_obj:
        giw_spin_mean_2 = np.mean(file.get_giw(ineq=2), axis=1)[..., niv_dmft - niv_cut : niv_dmft + niv_cut]
        giw_spin_mean = 0.5 * (giw_spin_mean + giw_spin_mean_2)
        del giw_spin_mean_2
        giw_spin_mean_large = np.zeros((2 * giw_spin_mean.shape[0], *giw_spin_mean.shape[1:]))
        giw_spin_mean_large[0:2] = giw_spin_mean
        giw_spin_mean_large[2:4] = giw_spin_mean
        g_dmft = GreensFunction(extend_orbital(giw_spin_mean_large))
        del giw_spin_mean_large

    siw_spin_mean = np.mean(file.get_siw(), axis=1)  # [band,spin,niv]
    siw_spin_mean = extend_orbital(siw_spin_mean)[None, None, None, ...]
    siw_dc_spin_mean = np.mean(file.get_dc(), axis=-1)  # [band,spin]
    siw_dc_spin_mean = extend_orbital(siw_dc_spin_mean)[None, None, None, ..., None]
    siw_spin_mean = siw_spin_mean[..., niv_dmft - niv_cut : niv_dmft + niv_cut]
    sigma_dmft = SelfEnergy(siw_spin_mean, estimate_niv_core=True) + siw_dc_spin_mean

    if combine_two_atoms_to_one_obj:
        siw_spin_mean_2 = np.mean(file.get_siw(ineq=2), axis=1)
        siw_spin_mean_2 = extend_orbital(siw_spin_mean_2)[None, None, None, ...]
        siw_dc_spin_mean_2 = np.mean(file.get_dc(ineq=2), axis=-1)
        siw_dc_spin_mean_2 = extend_orbital(siw_dc_spin_mean_2)[None, None, None, ..., None]
        siw_spin_mean_2 = siw_spin_mean_2[..., niv_dmft - niv_cut : niv_dmft + niv_cut]
        siw_spin_mean = 0.5 * (siw_spin_mean + siw_spin_mean_2)
        del siw_spin_mean_2
        siw_dc_spin_mean = 0.5 * (siw_dc_spin_mean + siw_dc_spin_mean_2)
        del siw_dc_spin_mean_2

        siw_spin_mean_large = np.zeros(
            (1, 1, 1, 2 * siw_spin_mean.shape[3], 2 * siw_spin_mean.shape[3], siw_spin_mean.shape[-1])
        )
        siw_spin_mean_large[:, :, :, 0:2, 0:2, ...] = siw_spin_mean
        siw_spin_mean_large[:, :, :, 2:4, 2:4, ...] = siw_spin_mean
        siw_dc_spin_mean_large = np.zeros_like(siw_spin_mean_large)
        siw_dc_spin_mean_large[:, :, :, 0:2, 0:2, ...] = siw_dc_spin_mean
        siw_dc_spin_mean_large[:, :, :, 2:4, 2:4, ...] = siw_dc_spin_mean
        sigma_dmft = SelfEnergy(siw_spin_mean_large, estimate_niv_core=True) + siw_dc_spin_mean_large
        del siw_spin_mean_large, siw_dc_spin_mean_large

    del giw_spin_mean, siw_spin_mean, siw_dc_spin_mean

    file.close()

    config.lattice.hamiltonian = set_hamiltonian(
        config.lattice.type, config.lattice.er_input, config.lattice.interaction_type, config.lattice.interaction_input
    )

    output_format = "LDGA_Nk{}_Nq{}_wc{}_vc{}_vs{}".format(
        config.lattice.k_grid.nk_tot,
        config.lattice.q_grid.nk_tot,
        config.box.niw_core,
        config.box.niv_core,
        config.box.niv_shell,
    )

    config.output.output_path = uniquify_path(os.path.join(config.output.output_path, output_format))
    config.output.plotting_path = os.path.join(config.output.output_path, config.output.plotting_subfolder_name)
    config.output.eliashberg_path = os.path.join(config.output.output_path, config.eliashberg.subfolder_name)

    if not os.path.exists(config.output.output_path) and config.output.save_quantities:
        os.makedirs(config.output.output_path)
    if not os.path.exists(config.output.plotting_path) and config.output.do_plotting:
        os.makedirs(config.output.plotting_path)
    if not os.path.exists(config.output.eliashberg_path) and config.eliashberg.perform_eliashberg:
        os.makedirs(config.output.eliashberg_path)

    g2_dens = g2_dens.cut_niw_and_niv(config.box.niw_core, config.box.niv_core)
    g2_magn = g2_magn.cut_niw_and_niv(config.box.niw_core, config.box.niv_core)

    if config.dmft.symmetrize_orbitals:
        g2_dens = g2_dens.symmetrize_orbitals(config.dmft.symmetrize_orbitals)
        g2_magn = g2_magn.symmetrize_orbitals(config.dmft.symmetrize_orbitals)
        g_dmft = g_dmft.symmetrize_orbitals(config.dmft.symmetrize_orbitals)
        sigma_dmft = sigma_dmft.symmetrize_orbitals(config.dmft.symmetrize_orbitals)
        config.logger.info(
            f"Symmetrized G2 with respect to orbitals {', '.join(str(o) for o in config.dmft.symmetrize_orbitals)}."
        )

    if config.dmft.do_sym_v_vp:
        g2_dens = g2_dens.symmetrize_v_vp()
        g2_magn = g2_magn.symmetrize_v_vp()
        config.logger.info(f"Symmetrized G2 with respect to v and v'.")

    config.lattice.k_grid.specify_orbital_basis(config.sys.n_bands, config.lattice.orbital_basis)
    config.lattice.q_grid.specify_orbital_basis(config.sys.n_bands, config.lattice.orbital_basis)

    return g_dmft, sigma_dmft, g2_dens, g2_magn


def update_frequency_boxes(niw: int, niv: int) -> None:
    """
    Updates the frequency boxes based on the available frequencies in the DMFT four-point object.
    """
    logger = config.logger

    if config.box.niw_core == -1:
        config.box.niw_core = niw
        logger.info(f"Number of bosonic Matsubara frequency is set to '-1'. Using niw = {niw}.")
    elif config.box.niw_core > niw:
        config.box.niw_core = niw
        logger.info(
            f"Number of bosonic Matsubara frequencies cannot exceed available "
            f"frequencies in the DMFT four-point object. Using niw = {niw}."
        )

    if config.box.niv_core == -1:
        config.box.niv_core = niv
        logger.info(f"Number of fermionic Matsubara frequency is set to '-1'. Using niv = {niv}.")
    elif config.box.niv_core > niv:
        config.box.niv_core = niv
        logger.info(
            f"Number of fermionic Matsubara frequencies cannot exceed available "
            f"frequencies in the DMFT four-point object. Using niv = {niv}."
        )

    config.box.niv_full = config.box.niv_core + config.box.niv_shell


def set_hamiltonian(er_type: str, er_input: str | list, int_type: str, int_input: str | list) -> Hamiltonian:
    """
    Sets the Hamiltonian based on the input from the config file. \n
    The kinetic part can be set in two ways: \n
    1. By providing the single-band hopping parameters t, tp, tpp. \n
    2. By providing the path + filename to the wannier_hr / wannier_hk file. \n
    The interaction can be set in three ways: \n
    1. By retrieving the data from the DMFT files. \n
    2. By providing the Kanamori interaction parameters [n_bands, U, J, (V)]. \n
    3. By providing the full path + filename to the U-matrix file. \n
    """
    ham = Hamiltonian()
    if er_type == "t_tp_tpp":
        if not isinstance(er_input, list):
            raise ValueError("Invalid input for t, tp, tpp.")
        ham = ham.kinetic_one_band_2d_t_tp_tpp(*er_input)
    elif er_type == "from_wannier90":
        if not isinstance(er_input, str):
            raise ValueError("Invalid input for wannier_hr.dat.")
        ham = ham.read_hr_w2k(er_input)
    elif er_type == "from_wannierHK":
        if not isinstance(er_input, str):
            raise ValueError("Invalid input for wannier.hk.")
        ham, k_points = ham.read_hk_w2k(er_input)
        if config.lattice.nk is None:
            # ATTENTION: This is currently only available for 2D square lattices.
            config.logger.info("Using q- and k-grid from wannier.hk file.")
            config.lattice.nk = (int(np.sqrt(k_points[:, 0].size)), int(np.sqrt(k_points[:, 0].size)), 1)
            config.lattice.nq = config.lattice.nk
            config.lattice.k_grid = bz.KGrid(config.lattice.nk, config.lattice.symmetries)
            config.lattice.q_grid = bz.KGrid(config.lattice.nq, config.lattice.symmetries)
        ham = ham.set_ek(ham.get_ek().reshape(*config.lattice.nk, config.sys.n_bands, config.sys.n_bands))
    else:
        raise NotImplementedError(f"Hamiltonian type {er_type} not supported.")

    if int_type == "one_band_from_dmft" or int_type == "" or int_type is None:
        return ham.single_band_interaction(config.lattice.interaction.udd)
    elif int_type == "kanamori_from_dmft":
        return ham.kanamori_interaction_dp(
            config.sys.nd_bands,
            config.sys.np_bands,
            config.lattice.interaction.udd,
            config.lattice.interaction.upp,
            config.lattice.interaction.udp,
            config.lattice.interaction.jdd,
            config.lattice.interaction.jpp,
            config.lattice.interaction.jdp,
            config.lattice.interaction.vdd,
            config.lattice.interaction.vpp,
        )
    elif int_type == "custom":
        if not isinstance(int_input, str):
            raise ValueError("Invalid input for umatrix file.")
        return ham.read_umatrix(int_input)
    else:
        raise NotImplementedError(f"Interaction type {int_type} not supported.")

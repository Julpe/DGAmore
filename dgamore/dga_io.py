# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os

import dgamore.brillouin_zone as bz
import dgamore.config as config
from dgamore.dmft_interface import W2dynInterface, TriqsInterface
from dgamore.greens_function import GreensFunction
from dgamore.hamiltonian import Hamiltonian
from dgamore.local_four_point import LocalFourPoint
from dgamore.n_point_base import *
from dgamore.self_energy import SelfEnergy


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


def load_from_dmft_file_and_update_config() -> (
    tuple[list[GreensFunction], list[SelfEnergy], list[LocalFourPoint], list[LocalFourPoint]]
):
    """
    Loads data from the w2dyn file and updates the config file.
    If combine_atoms_to_one_obj is True, we are doubling the orbital space and putting the data from the
    (in)equivalent atoms into the diagonal blocks after we average over them.
    """
    if config.dmft.type.lower() == "w2dyn":
        dmft_interface = W2dynInterface()
    elif config.dmft.type.lower() == "triqs":
        dmft_interface = TriqsInterface()
    else:
        raise ValueError("DMFT input not supported.")

    config.sys.beta = dmft_interface.get_beta()

    config.lattice.interaction.udd = dmft_interface.get_udd()
    config.lattice.interaction.jdd = dmft_interface.get_jdd()
    config.lattice.interaction.vdd = dmft_interface.get_vdd()

    config.sys.mu = dmft_interface.get_mu()
    config.sys.mu_dmft = config.sys.mu
    config.sys.n = dmft_interface.get_totdens()

    sigma_per_ineq, g_per_ineq, g2_dens_per_ineq, g2_magn_per_ineq = [], [], [], []
    for ineq in range(1, config.dmft.n_ineq + 1):
        config.dmft.n_bands_per_ineq.append(dmft_interface.get_nd(ineq))
        config.sys.occ_dmft_per_ineq.append(dmft_interface.get_occ(ineq))

        g2_dens = dmft_interface.get_g2iw(SpinChannel.DENS, ineq)
        g2_dens_per_ineq.append(g2_dens)
        g2_magn = dmft_interface.get_g2iw(SpinChannel.MAGN, ineq)
        g2_magn_per_ineq.append(g2_magn)

        update_frequency_boxes(g2_dens.niw, g2_dens.niv)

        g_dmft = dmft_interface.get_giw(ineq)
        g_per_ineq.append(g_dmft)

        config.box.niv_dmft = g_dmft.niv

        sigma_dmft = dmft_interface.get_siw(ineq)
        sigma_per_ineq.append(sigma_dmft)

    config.sys.n_bands = sum(config.dmft.n_bands_per_ineq[ineq - 1] for ineq in config.dmft.ineq_ordering)

    config.lattice.hamiltonian = set_hamiltonian(
        config.lattice.type, config.lattice.er_input, config.lattice.interaction_type, config.lattice.interaction_input
    )  # Hamiltonian always has config.sys.n_bands orbitals

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

    if not os.path.exists(config.output.output_path):
        os.makedirs(config.output.output_path)
    if not os.path.exists(config.output.plotting_path) and config.output.do_plotting:
        os.makedirs(config.output.plotting_path)
    if not os.path.exists(config.output.eliashberg_path) and config.eliashberg.perform_eliashberg:
        os.makedirs(config.output.eliashberg_path)

    for i in range(len(g2_dens_per_ineq)):
        g2_dens_per_ineq[i] = g2_dens_per_ineq[i].cut_niw_and_niv(config.box.niw_core, config.box.niv_core)
        g2_magn_per_ineq[i] = g2_magn_per_ineq[i].cut_niw_and_niv(config.box.niw_core, config.box.niv_core)

        if config.dmft.symmetrize_orbitals:
            g2_dens_per_ineq[i] = g2_dens_per_ineq[i].symmetrize_orbitals(config.dmft.symmetrize_orbitals)
            g2_magn_per_ineq[i] = g2_magn_per_ineq[i].symmetrize_orbitals(config.dmft.symmetrize_orbitals)
            g_per_ineq[i] = g_per_ineq[i].symmetrize_orbitals(config.dmft.symmetrize_orbitals)
            sigma_per_ineq[i] = sigma_per_ineq[i].symmetrize_orbitals(config.dmft.symmetrize_orbitals)
            config.logger.info(
                f"Symmetrized G2 with respect to orbitals {', '.join(str(o) for o in config.dmft.symmetrize_orbitals)} "
                f"for atom {i+1}."
            )

        if config.dmft.do_sym_v_vp:
            g2_dens_per_ineq[i] = g2_dens_per_ineq[i].symmetrize_v_vp()
            g2_magn_per_ineq[i] = g2_magn_per_ineq[i].symmetrize_v_vp()
            config.logger.info(f"Symmetrized G2 with respect to v and v' for atom {i+1}.")

    return g_per_ineq, sigma_per_ineq, g2_dens_per_ineq, g2_magn_per_ineq


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
    if er_type.lower() == "t_tp_tpp":
        if not isinstance(er_input, list):
            raise ValueError("Invalid input for t, tp, tpp.")
        ham = ham.kinetic_one_band_2d_t_tp_tpp(*er_input)
    elif er_type.lower() == "from_wannier90":
        if not isinstance(er_input, str):
            raise ValueError("Invalid input for wannier_hr.dat.")
        ham = ham.read_hr_w2k(er_input)
    elif er_type.lower() == "from_wannierhk":
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

    if int_type.lower() == "one_band_from_dmft" or int_type == "" or int_type is None:
        return ham.single_band_interaction(config.lattice.interaction.udd)
    elif int_type.lower() == "kanamori_from_dmft":
        return ham.kanamori_interaction_d(
            config.sys.n_bands,
            config.lattice.interaction.udd,
            config.lattice.interaction.jdd,
            config.lattice.interaction.vdd,
        )
    elif int_type.lower() == "custom":
        if not isinstance(int_input, str):
            raise ValueError("Invalid input for umatrix file.")
        return ham.read_umatrix(int_input)
    else:
        raise NotImplementedError(f"Interaction type {int_type} not supported.")

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


def load_from_w2dyn_file_and_update_config() -> tuple[GreensFunction, SelfEnergy, LocalFourPoint, LocalFourPoint]:
    """
    Loads data from the w2dyn file and updates the config file.
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
    config.sys.n_bands = file.get_nd() + file.get_np()
    config.sys.n = file.get_totdens()
    config.sys.occ_dmft = 2 * np.mean(file.get_rho1(), axis=(1, 3))

    if config.sys.n == 0:
        config.sys.n = 2 * np.sum(config.sys.occ_dmft)

    file2 = w2dyn_aux.W2dynG4iwFile(fname=str(os.path.join(config.dmft.input_path, config.dmft.fname_2p)))
    g2_dens = LocalFourPoint(file2.read_g2_full_multiband(config.sys.n_bands, name="dens"), channel=SpinChannel.DENS)
    g2_magn = LocalFourPoint(file2.read_g2_full_multiband(config.sys.n_bands, name="magn"), channel=SpinChannel.MAGN)
    file2.close()

    update_frequency_boxes(g2_dens.niw, g2_dens.niv)

    def extend_orbital(arr: np.ndarray) -> np.ndarray:
        return np.einsum("i...,ij->ij...", arr, np.eye(config.sys.n_bands))

    giw_spin_mean = np.mean(file.get_giw(), axis=1)  # [band,spin,niv]
    niv_dmft = giw_spin_mean.shape[-1] // 2
    niv_cut = config.box.niw_core + config.box.niv_full
    giw_spin_mean = giw_spin_mean[..., niv_dmft - niv_cut : niv_dmft + niv_cut]
    g_dmft = GreensFunction(extend_orbital(giw_spin_mean))

    siw_spin_mean = np.mean(file.get_siw(), axis=1)  # [band,spin,niv]
    siw_spin_mean = extend_orbital(siw_spin_mean)[None, None, None, ...]
    siw_dc_spin_mean = np.mean(file.get_dc(), axis=-1)  # [band,spin]
    siw_dc_spin_mean = extend_orbital(siw_dc_spin_mean)[None, None, None, ..., None]
    siw_spin_mean = siw_spin_mean[..., niv_dmft - niv_cut : niv_dmft + niv_cut]
    sigma_dmft = SelfEnergy(siw_spin_mean, estimate_niv_core=True) + siw_dc_spin_mean

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

    g2_dens = update_g2_from_dmft(g2_dens)
    g2_magn = update_g2_from_dmft(g2_magn)

    return g_dmft, sigma_dmft, g2_dens, g2_magn


def update_frequency_boxes(niw: int, niv: int) -> None:
    """
    Updates the frequency boxes based on the available frequencies in the DMFT four-point object.
    """
    logger = config.logger

    if config.box.niw_core == -1:
        config.box.niw_core = niw
        logger.log_info(f"Number of bosonic Matsubara frequency is set to '-1'. Using niw = {niw}.")
    elif config.box.niw_core > niw:
        config.box.niw_core = niw
        logger.log_info(
            f"Number of bosonic Matsubara frequencies cannot exceed available "
            f"frequencies in the DMFT four-point object. Using niw = {niw}."
        )

    if config.box.niv_core == -1:
        config.box.niv_core = niv
        logger.log_info(f"Number of fermionic Matsubara frequency is set to '-1'. Using niv = {niv}.")
    elif config.box.niv_core > niv:
        config.box.niv_core = niv
        logger.log_info(
            f"Number of fermionic Matsubara frequencies cannot exceed available "
            f"frequencies in the DMFT four-point object. Using niv = {niv}."
        )

    config.box.niv_full = config.box.niv_core + config.box.niv_shell


def update_g2_from_dmft(g2: LocalFourPoint) -> LocalFourPoint:
    """
    Updates the four-point object based on the available frequencies in the DMFT four-point object. Also symmetrizes
    with respect to v and v' if specified in the config file.
    """
    g2 = g2.cut_niw_and_niv(config.box.niw_core, config.box.niv_core)
    if config.dmft.do_sym_v_vp:
        g2 = g2.symmetrize_v_vp()
        config.logger.log_info(f"Symmetrized G2 ({g2.channel.value}) with respect to v and v'.")
    return g2


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
            config.logger.log_info("Using q- and k-grid from wannier.hk file.")
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
        return ham.kanamori_interaction(
            config.sys.n_bands,
            config.lattice.interaction.udd,
            config.lattice.interaction.jdd,
            config.lattice.interaction.vdd,
        )
    elif int_type == "custom":
        if not isinstance(int_input, str):
            raise ValueError("Invalid input for umatrix file.")
        return ham.read_umatrix(int_input)
    else:
        raise NotImplementedError(f"Interaction type {int_type} not supported.")

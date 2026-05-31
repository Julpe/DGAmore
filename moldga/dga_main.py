# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import itertools as it
import logging
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from mpi4py import MPI

import moldga.config as config
import moldga.dga_io as dga_io
import moldga.eliashberg_solver as eliashberg_solver
import moldga.local_sde as local_sde
import moldga.nonlocal_sde as nonlocal_sde
import moldga.plotting as plotting
from moldga import max_ent
from moldga.brillouin_zone import AUTO_SYMMETRIES_SENTINEL
from moldga.config_parser import ConfigParser
from moldga.greens_function import GreensFunction
from moldga.interaction import LocalInteraction
from moldga.local_four_point import LocalFourPoint
from moldga.self_energy import SelfEnergy

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def execute_dga_routine():
    configure_matplotlib()

    comm = MPI.COMM_WORLD

    config_parser = ConfigParser().parse_config(comm)
    logger = config.logger
    logger.info("Starting DGA routine.")
    logger.info(f"Running on {str(comm.size)} {"process" if comm.size == 1 else "processes"}.")

    if comm.rank == 0:
        g_dmft_per_ineq, sigma_dmft_per_ineq, g2_dens_per_ineq, g2_magn_per_ineq = (
            dga_io.load_from_dmft_file_and_update_config()
        )
    else:
        g_dmft_per_ineq, sigma_dmft_per_ineq, g2_dens_per_ineq, g2_magn_per_ineq = None, None, None, None

    (
        config.dmft,
        config.lattice,
        config.box,
        config.output,
        config.sys,
        config.self_consistency,
        config.eliashberg,
        config.lambda_correction,
        config.self_energy_interpolation,
        config.memory,
        config.ana_cont,
    ) = comm.bcast(
        (
            config.dmft,
            config.lattice,
            config.box,
            config.output,
            config.sys,
            config.self_consistency,
            config.eliashberg,
            config.lambda_correction,
            config.self_energy_interpolation,
            config.memory,
            config.ana_cont,
        ),
        root=0,
    )

    setup_lambda_correction_settings(comm)

    config_parser.save_config_file(path=config.output.output_path, name="dga_config.yaml")

    logger.info("Config init and folder setup done.")
    logger.info("Loaded data from w2dyn file.")

    g_dmft_per_ineq = comm.bcast(g_dmft_per_ineq, root=0)
    sigma_dmft_per_ineq = comm.bcast(sigma_dmft_per_ineq, root=0)

    if comm.rank == 0:
        logger.log_memory_usage("g_dmft & sigma_dmft", g_dmft_per_ineq[0] * len(g_dmft_per_ineq), 2 * comm.size)
        logger.log_memory_usage("g2_dens & g2_magn", g2_dens_per_ineq[0] * len(g2_dens_per_ineq), 2)

    logger.info("Preprocessing done.")

    ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)

    if isinstance(config.lattice.k_grid.symmetries, type(AUTO_SYMMETRIES_SENTINEL)):
        config.lattice.k_grid.specify_auto_symmetries(ek)
        logger.info(
            f"Automatically determined symmetries for the k-grid. The irreducible BZ has "
            f"{config.lattice.k_grid.nk_irr}/{config.lattice.k_grid.nk_tot} elements."
        )

        if config.lattice.k_grid.nk == config.lattice.q_grid.nk:
            config.lattice.q_grid = config.lattice.k_grid
        else:
            config.lattice.q_grid.specify_auto_symmetries(ek)

        logger.info(
            f"Automatically determined symmetries for the q-grid. The irreducible BZ has "
            f"{config.lattice.q_grid.nk_irr}/{config.lattice.q_grid.nk_tot} elements."
        )

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    if comm.rank == 0:
        (
            g2_dens_full,
            g2_magn_full,
            gamma_d_full,
            gamma_m_full,
            chi_d_full,
            chi_m_full,
            vrg_d_full,
            vrg_m_full,
            f_d_full,
            f_m_full,
            gchi_d_full,
            gchi_m_full,
            sigma_loc_full,
            sigma_dmft_full,
            g_dmft_full,
        ) = (None,) * 15
        offsets = []
        offset = 0

        for ineq in config.dmft.ineq_ordering:
            offsets.append(offset)
            offset += config.dmft.n_bands_per_ineq[ineq - 1]

        first_block = {}
        for k, ineq in enumerate(config.dmft.ineq_ordering):
            if ineq not in first_block:
                first_block[ineq] = k

        (
            gamma_d_per_ineq,
            gamma_m_per_ineq,
            chi_d_per_ineq,
            chi_m_per_ineq,
            vrg_d_per_ineq,
            vrg_m_per_ineq,
            f_d_per_ineq,
            f_m_per_ineq,
            gchi_d_per_ineq,
            gchi_m_per_ineq,
            sigma_loc_per_ineq,
        ) = ([], [], [], [], [], [], [], [], [], [], [])
        for ineq in range(1, config.dmft.n_ineq + 1):
            k = first_block[ineq]
            n_start = offsets[k]
            n_end = n_start + config.dmft.n_bands_per_ineq[ineq - 1]

            config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[ineq - 1]

            u_loc_ineq = LocalInteraction(u_loc.mat[n_start:n_end, n_start:n_end, n_start:n_end, n_start:n_end])

            logger.info(f"Starting local Schwinger-Dyson equation (SDE) for atom {ineq}.")

            if comm.rank == 0:
                (gamma_d, gamma_m, chi_d, chi_m, vrg_d, vrg_m, f_d, f_m, gchi_d, gchi_m, sigma_loc) = (
                    local_sde.perform_local_schwinger_dyson(
                        g_dmft_per_ineq[ineq - 1], g2_dens_per_ineq[ineq - 1], g2_magn_per_ineq[ineq - 1], u_loc_ineq
                    )
                )
            else:
                (gamma_d, gamma_m, chi_d, chi_m, vrg_d, vrg_m, f_d, f_m, gchi_d, gchi_m, sigma_loc) = (None,) * 11

            gamma_d_per_ineq.append(gamma_d)
            gamma_m_per_ineq.append(gamma_m)
            chi_d_per_ineq.append(chi_d)
            chi_m_per_ineq.append(chi_m)
            vrg_d_per_ineq.append(vrg_d)
            vrg_m_per_ineq.append(vrg_m)
            f_d_per_ineq.append(f_d)
            f_m_per_ineq.append(f_m)
            gchi_d_per_ineq.append(gchi_d)
            gchi_m_per_ineq.append(gchi_m)
            sigma_loc_per_ineq.append(sigma_loc)

            logger.info(f"Local Schwinger-Dyson equation (SDE) for atom {ineq} done.")

        def write_to_full_4pt_quantity(obj_full, obj_ineq: LocalFourPoint, sl: slice):
            if obj_full is None:
                obj_full = deepcopy(obj_ineq)
                obj_full.mat = np.zeros(
                    (config.sys.n_bands,) * 4 + obj_ineq.current_shape[4:], dtype=obj_ineq.mat.dtype
                )
                obj_full.update_original_shape()
            obj_full[sl, sl, sl, sl] = obj_ineq.mat
            return obj_full

        def write_to_full_2pt_quantity(
            obj_full, obj_ineq: SelfEnergy | GreensFunction, sl: slice, has_momentum: bool = True
        ):
            if obj_full is None:
                obj_full = deepcopy(obj_ineq)
                obj_full.mat = np.zeros(
                    ((1, 1, 1) + (config.sys.n_bands,) * 2 if has_momentum else (config.sys.n_bands,) * 2)
                    + (obj_ineq.current_shape[-1],),
                    dtype=obj_ineq.mat.dtype,
                )
                obj_full.update_original_shape()
            if has_momentum:
                obj_full[0, 0, 0, sl, sl, :] = obj_ineq.mat
                obj_full._smom0 = np.zeros((config.sys.n_bands,) * 2)
                obj_full._smom1 = np.zeros((config.sys.n_bands,) * 2)
            else:
                obj_full[sl, sl] = obj_ineq.mat
            return obj_full

        def write_smom(obj_full: SelfEnergy, obj_ineq: SelfEnergy, sl: slice):
            obj_full._smom0[sl, sl] = obj_ineq._smom0
            obj_full._smom1[sl, sl] = obj_ineq._smom1
            return obj_full

        for idx, ineq in enumerate(config.dmft.ineq_ordering):
            if comm.rank != 0:
                continue

            n_start = sum([config.dmft.n_bands_per_ineq[i - 1] for i in config.dmft.ineq_ordering[:idx]])
            n_end = n_start + config.dmft.n_bands_per_ineq[ineq - 1]
            s = slice(n_start, n_end)

            g2_dens_full = write_to_full_4pt_quantity(g2_dens_full, g2_dens_per_ineq[ineq - 1], s)
            g2_magn_full = write_to_full_4pt_quantity(g2_magn_full, g2_magn_per_ineq[ineq - 1], s)
            gamma_d_full = write_to_full_4pt_quantity(gamma_d_full, gamma_d_per_ineq[ineq - 1], s)
            gamma_m_full = write_to_full_4pt_quantity(gamma_m_full, gamma_m_per_ineq[ineq - 1], s)
            chi_d_full = write_to_full_4pt_quantity(chi_d_full, chi_d_per_ineq[ineq - 1], s)
            chi_m_full = write_to_full_4pt_quantity(chi_m_full, chi_m_per_ineq[ineq - 1], s)
            vrg_d_full = write_to_full_4pt_quantity(vrg_d_full, vrg_d_per_ineq[ineq - 1], s)
            vrg_m_full = write_to_full_4pt_quantity(vrg_m_full, vrg_m_per_ineq[ineq - 1], s)
            f_d_full = write_to_full_4pt_quantity(f_d_full, f_d_per_ineq[ineq - 1], s)
            f_m_full = write_to_full_4pt_quantity(f_m_full, f_m_per_ineq[ineq - 1], s)
            gchi_d_full = write_to_full_4pt_quantity(gchi_d_full, gchi_d_per_ineq[ineq - 1], s)
            gchi_m_full = write_to_full_4pt_quantity(gchi_m_full, gchi_m_per_ineq[ineq - 1], s)
            sigma_dmft_full = write_to_full_2pt_quantity(sigma_dmft_full, sigma_dmft_per_ineq[ineq - 1], s)
            g_dmft_full = write_to_full_2pt_quantity(g_dmft_full, g_dmft_per_ineq[ineq - 1], s, has_momentum=False)
            sigma_loc_full = write_to_full_2pt_quantity(sigma_loc_full, sigma_loc_per_ineq[ineq - 1], s)

            sigma_loc_full = write_smom(sigma_loc_full, sigma_loc_per_ineq[ineq - 1], s)
            sigma_dmft_full = write_smom(sigma_dmft_full, sigma_dmft_per_ineq[ineq - 1], s)

    if config.lambda_correction.perform_lambda_correction and comm.rank == 0:
        chi_d_full.save(name="chi_dens_loc", output_dir=config.output.output_path)
        chi_m_full.save(name="chi_magn_loc", output_dir=config.output.output_path)

    if comm.rank == 0:
        g2_dens_full.save(name="g2_dens_loc", output_dir=config.output.output_path)
        g2_magn_full.save(name="g2_magn_loc", output_dir=config.output.output_path)
        del g2_dens_per_ineq, g2_magn_per_ineq

        gamma_d_full.save(name="gamma_dens_loc", output_dir=config.output.output_path)
        gamma_m_full.save(name="gamma_magn_loc", output_dir=config.output.output_path)

        vrg_d_full.save(name="vrg_dens_loc", output_dir=config.output.output_path)
        vrg_m_full.save(name="vrg_magn_loc", output_dir=config.output.output_path)
        del vrg_d_full, vrg_m_full

        gchi_d_full.save(name="gchi_dens_loc", output_dir=config.output.output_path)
        gchi_m_full.save(name="gchi_magn_loc", output_dir=config.output.output_path)
        f_d_full.save(name="f_dens_loc", output_dir=config.output.output_path)
        f_m_full.save(name="f_magn_loc", output_dir=config.output.output_path)
        del f_d_full, f_m_full
        logger.info("Saved all relevant quantities as numpy files.")

    if config.output.do_plotting and comm.rank == 0:
        plotting.plot_nu_nup(gchi_d_full, omega=0, name=f"Gchi_dens", output_dir=config.output.plotting_path)
        plotting.plot_nu_nup(gchi_m_full, omega=0, name=f"Gchi_magn", output_dir=config.output.plotting_path)
        logger.info(f"Local generalized susceptibilities dens & magn plotted.")
        del gchi_m_full, gchi_d_full

        gamma_dens_plot = gamma_d_full.cut_niv(min(config.box.niv_core, 2 * int(config.sys.beta)))
        plotting.plot_nu_nup(gamma_dens_plot, omega=0, name="Gamma_dens", output_dir=config.output.plotting_path)
        plotting.plot_nu_nup(gamma_dens_plot, omega=10, name="Gamma_dens", output_dir=config.output.plotting_path)
        plotting.plot_nu_nup(gamma_dens_plot, omega=-10, name="Gamma_dens", output_dir=config.output.plotting_path)
        logger.info("Plotted gamma (dens).")
        del gamma_dens_plot, gamma_d_full

        gamma_magn_plot = gamma_m_full.cut_niv(min(config.box.niv_core, 2 * int(config.sys.beta)))
        plotting.plot_nu_nup(gamma_magn_plot, omega=0, name="Gamma_magn", output_dir=config.output.plotting_path)
        plotting.plot_nu_nup(gamma_magn_plot, omega=10, name="Gamma_magn", output_dir=config.output.plotting_path)
        plotting.plot_nu_nup(gamma_magn_plot, omega=-10, name="Gamma_magn", output_dir=config.output.plotting_path)
        logger.info("Plotted gamma (magn).")
        del gamma_magn_plot, gamma_m_full

        g_dmft_full._ek = ek
        plotting.chi_checks(
            [chi_d_full.mat],
            [chi_m_full.mat],
            config.sys.beta,
            ["Loc-tilde"],
            g_dmft_full.e_kin,
            name="loc",
            output_dir=config.output.plotting_path,
        )
        del chi_d, chi_m
        logger.info("Plotted checks of the susceptibility.")

        sigma_list = []
        sigma_names = []
        for i, j in it.product(range(config.sys.n_bands), repeat=2):
            try:
                sigma_list.append(sigma_loc_full[0, 0, 0, i, j])
                sigma_list.append(sigma_dmft_full[0, 0, 0, i, j])
                sigma_names.append(f"SDE{i}{j}")
                sigma_names.append(f"Input{i}{j}")
            except IndexError:
                break

        plotting.sigma_loc_checks(
            sigma_list,
            sigma_names,
            config.sys.beta,
            show=False,
            save=True,
            xmax=config.box.niv_core,
            name="DMFT",
            output_dir=config.output.plotting_path,
        )
        logger.info("Plotted local self-energies for comparison.")
        logger.info("Finished plotting.")

    logger.info("Local DGA routine finished.")

    if comm.rank == 0:
        sigma_dmft_full.save(name="sigma_dmft", output_dir=config.output.output_path)
        g_dmft_full.save(name="g_dmft", output_dir=config.output.output_path)
        sigma_loc_full.save(name="siw_dga_local", output_dir=config.output.output_path)

    if config.output.do_plotting and comm.rank == 0:
        for g2, name in [(g2_dens_full, f"G2_dens"), (g2_magn_full, f"G2_magn")]:
            for omega in ([0, -10, 10] if config.box.niw_core > 10 else [0]):
                plotting.plot_nu_nup(g2, omega=omega, name=name, output_dir=config.output.plotting_path)
        logger.info(f"Plotted g2 (dens) and g2 (magn).")
        del g2_dens_full, g2_magn_full

    if comm.rank != 0:
        sigma_loc_full, sigma_dmft_full, g_dmft_full = (None,) * 3

    # there is no need to broadcast the other quantities
    sigma_loc_full = comm.bcast(sigma_loc_full, root=0)
    sigma_dmft_full = comm.bcast(sigma_dmft_full, root=0)
    g_dmft_full = comm.bcast(g_dmft_full, root=0)

    logger.info("Starting non-local ladder-DGA routine.")
    sigma_dga = nonlocal_sde.calculate_self_energy_q(comm, u_loc, v_nonloc, sigma_dmft_full, sigma_loc_full)
    del sigma_dmft_full, sigma_loc_full
    logger.info("Non-local ladder-DGA routine finished.")

    giwk_dga = GreensFunction.get_g_full(sigma_dga, config.sys.mu, ek)

    if config.ana_cont.do_ana_cont_green_dga:
        spectrum = max_ent.perform_maxent_giwk(giwk_dga, "DGA", comm)

        if config.ana_cont.plot_spectrum and comm.rank == 0:
            plotting.plot_spectrum(
                spectrum,
                config.lattice.k_grid.kx,
                config.lattice.k_grid.ky,
                config.lattice.k_grid.kz,
                config.ana_cont.k_path,
                config.ana_cont.energy_window,
                config.sys.beta,
                r"$\mathrm{D}\Gamma\mathrm{A} Spectrum$",
                output_dir=config.output.output_path,
                name="dga",
            )
            logger.info("Plotted DGA spectrum.")
        del spectrum

    if config.ana_cont.do_ana_cont_green_dmft:
        g_latt = None
        if comm.rank == 0:
            g_latt = GreensFunction(np.load(os.path.join(config.output.output_path, "g_latt_dmft.npy"))).cut_niv(
                config.box.niv_core
            )
        g_latt = comm.bcast(g_latt, root=0)
        spectrum = max_ent.perform_maxent_giwk(g_latt, "DMFT", comm)

        if config.ana_cont.plot_spectrum and comm.rank == 0:
            plotting.plot_spectrum(
                spectrum,
                config.lattice.k_grid.kx,
                config.lattice.k_grid.ky,
                config.lattice.k_grid.kz,
                config.ana_cont.k_path,
                config.ana_cont.energy_window,
                config.sys.beta,
                r"$\mathrm{DMFT}$",
                output_dir=config.output.output_path,
                name="dmft",
            )
            logger.info("Plotted DMFT spectrum.")
        del spectrum

    if comm.rank == 0:
        sigma_dga.save(name=f"sigma_dga", output_dir=config.output.output_path)
        logger.info("Saved non-local self-energy as numpy file.")

        giwk_dga.save(name=f"giwk_dga", output_dir=config.output.output_path)
        logger.info("Saved non-local Green's function as numpy file.")

    if config.output.do_plotting and comm.rank == 0:
        kx, ky = config.lattice.k_grid.kx_shift_closed, config.lattice.k_grid.ky_shift_closed
        plotting.plot_two_point_kx_ky(
            sigma_dga,
            kx,
            ky,
            title=r"$\Sigma^{k_xk_y k_z=0;\nu=0}$",
            name="Sigma_dga_kz0",
            output_dir=config.output.plotting_path,
        )
        plotting.plot_two_point_kx_ky_real_and_imag(
            sigma_dga,
            kx,
            ky,
            title=r"\Sigma^{k_xk_y k_z=0;\nu=0}",
            name="Sigma_dga_kz0",
            output_dir=config.output.plotting_path,
        )
        logger.info("Plotted non-local self-energy as a function of kx and ky.")

        plotting.plot_two_point_kx_ky(
            giwk_dga,
            kx,
            ky,
            title=r"$G^{k_x k_y k_z=0;\nu=0}$",
            name="Giwk_dga_kz0",
            output_dir=config.output.plotting_path,
        )
        plotting.plot_two_point_kx_ky_real_and_imag(
            giwk_dga,
            kx,
            ky,
            title=r"G^{k_x k_y k_z=0;\nu=0}",
            name="Giwk_dga_kz0",
            output_dir=config.output.plotting_path,
        )
        logger.info("Plotted non-local Green's function as a function of kx and ky.")

    logger.info("DGA routine finished.")

    if config.eliashberg.perform_eliashberg:
        if not np.allclose(config.lattice.q_grid.nk, config.lattice.k_grid.nk):
            raise ValueError("Eliashberg equation can only be solved when nq = nk.")
        logger.info("Starting with Eliashberg equation.")
        lambdas_sing, lambdas_trip, gaps_sing, gaps_trip = eliashberg_solver.solve(
            giwk_dga, g_dmft_full, u_loc, v_nonloc, comm
        )

        if comm.rank == 0:
            np.savetxt(
                os.path.join(config.output.eliashberg_path, "eigenvalues.txt"),
                [lambdas_sing.real, lambdas_trip.real],
                delimiter=",",
                fmt="%.9f",
            )

            for i in range(len(gaps_sing)):
                gaps_sing[i].save(name=f"gap_sing_{i+1}", output_dir=config.output.eliashberg_path)
                gaps_trip[i].save(name=f"gap_trip_{i+1}", output_dir=config.output.eliashberg_path)
            logger.info("Saved singlet and triplet gap functions to files.")

        if config.output.do_plotting and comm.rank == 0:
            kx, ky = config.lattice.k_grid.kx_shift_closed, config.lattice.k_grid.ky_shift_closed
            for i in range(len(gaps_sing)):
                plotting.plot_gap_function(
                    gaps_sing[i], kx, ky, name=f"gap_sing_{i+1}", output_dir=config.output.eliashberg_path
                )
                plotting.plot_gap_function(
                    gaps_trip[i], kx, ky, name=f"gap_trip_{i+1}", output_dir=config.output.eliashberg_path
                )
            logger.info("Plotted singlet and triplet gap functions.")

    logger.info("Exiting ...")
    MPI.Finalize()


def setup_lambda_correction_settings(comm: MPI.Comm) -> None:
    """
    Sets up the lambda correction settings based on the configuration provided by the user. If the user has enabled
    the lambda correction in the self-consistency settings, it will be enabled in the lambda correction settings as well.
    If the user has enabled the lambda correction in the lambda correction settings, but not in the self-consistency settings,
    the self-consistency will be set to a single iteration with full mixing. Will raise an error if the user tries to enable
    the lambda correction for multi-band systems.
    """
    if (
        comm.rank == 0
        and config.sys.n_bands != 1
        and (config.lambda_correction.perform_lambda_correction or config.self_consistency.use_lambda_correction)
    ):
        raise ValueError(
            "Lambda correction is not available for multi-band systems. Please disable it in the config file."
        )

    if config.self_consistency.max_iter > 1 and not config.self_consistency.use_lambda_correction:
        config.lambda_correction.perform_lambda_correction = False
        config.logger.info("Calculating self-consistency without lambda correction.")
        return

    if config.self_consistency.max_iter > 1 and config.self_consistency.use_lambda_correction:
        config.lambda_correction.perform_lambda_correction = True
        config.logger.info("Calculating self-consistency with lambda correction.")
        return

    if config.lambda_correction.perform_lambda_correction:
        config.self_consistency.max_iter = 1
        config.self_consistency.mixing = 1.0
        config.logger.info("Performing one-shot DGA with lambda correction.")
        return
    elif not config.lambda_correction.perform_lambda_correction:
        config.self_consistency.max_iter = 1
        config.self_consistency.mixing = 1.0
        config.logger.info("Performing one-shot DGA without lambda correction.")
        return

    raise ValueError("Invalid configuration for lambda correction and self-consistency. Please check the config file.")


def configure_matplotlib():
    """
    Configures matplotlib to use the Euler font for mathematical expressions if it is available on the system. This is
    done because The Euler font is the default math font in my thesis.
    """
    euler_font = [s for s in font_manager.findSystemFonts() if "euler" in s.lower()]
    if len(euler_font) == 0:
        return
    euler_font_path = euler_font[0]
    font_manager.fontManager.addfont(euler_font_path)
    prop_euler = font_manager.FontProperties(fname=euler_font_path)
    plt.rc("axes", unicode_minus=False)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = prop_euler.get_name()
    plt.rcParams["font.size"] = 12
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.rm"] = prop_euler.get_name()
    plt.rcParams["mathtext.it"] = prop_euler.get_name()
    plt.rcParams["mathtext.bf"] = prop_euler.get_name()


if __name__ == "__main__":
    execute_dga_routine()

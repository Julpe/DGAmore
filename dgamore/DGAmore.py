#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Main entry point and top-level driver of a DGAmore run (installed on the PATH as ``DGAmore``). The
:func:`main` orchestrates the full pipeline: parse the config, load the DMFT input, run the local
Schwinger-Dyson step per inequivalent atom and assemble the full multi-band quantities, run the non-local ladder
DGA self-energy, optionally analytically continue to real frequencies, and optionally solve the Eliashberg
equation -- saving and plotting results along the way. Rank 0 owns the file I/O, local assembly and plotting; the
configuration and the assembled local quantities are broadcast to the other MPI ranks.
"""

import itertools as it
import logging
import os

# OpenMPI: exclude the UCX one-sided (RMA) component before MPI is initialised. On some OpenMPI 5.x builds it fails its
# own component-query and prints a benign "OSC UCX component priority set inside component query failed" warning when
# the per-node shared-memory giwk window is created.
os.environ.setdefault("OMPI_MCA_osc", "^ucx")

import matplotlib.pyplot as plt
import numpy as np
import psutil
from matplotlib import font_manager
from mpi4py import MPI

import dgamore.config as config
import dgamore.dga_io as dga_io
import dgamore.eliashberg_solver as eliashberg_solver
import dgamore.local_sde as local_sde
import dgamore.memory_estimator as memory_estimator
import dgamore.nonlocal_sde as nonlocal_sde
import dgamore.plotting as plotting
from dgamore import max_ent
from dgamore.brillouin_zone import AUTO_SYMMETRIES_SENTINEL
from dgamore.config_parser import ConfigParser
from dgamore.greens_function import GreensFunction
from dgamore.interaction import LocalInteraction
from dgamore.local_four_point import LocalFourPoint
from dgamore.self_energy import SelfEnergy

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main():
    """
    Runs the complete DGA pipeline end to end: config parsing and folder setup, DMFT input loading, the local
    Schwinger-Dyson step (per inequivalent atom, assembled into full multi-band quantities), the non-local
    ladder-DGA self-energy and Green's function, optional analytic continuation, and the optional Eliashberg
    solution -- saving and plotting results throughout. This is the console-script entry point.

    :return: None.
    """
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

    autodetect_memory_settings(comm)

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

            u_loc_ineq = LocalInteraction(u_loc.mat[n_start:n_end, n_start:n_end, n_start:n_end, n_start:n_end].copy())

            logger.info(f"Starting local Schwinger-Dyson equation (SDE) for atom {ineq}.")

            if comm.rank == 0:
                gamma_d, gamma_m, chi_d, chi_m, vrg_d, vrg_m, f_d, f_m, gchi_d, gchi_m, sigma_loc = (
                    local_sde.perform_local_schwinger_dyson(
                        g_dmft_per_ineq[ineq - 1], g2_dens_per_ineq[ineq - 1], g2_magn_per_ineq[ineq - 1], u_loc_ineq
                    )
                )
            else:
                gamma_d, gamma_m, chi_d, chi_m, vrg_d, vrg_m, f_d, f_m, gchi_d, gchi_m, sigma_loc = (None,) * 11

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
            """
            Writes a single inequivalent atom's four-point quantity into the orbital-diagonal block of the assembled
            full multi-band quantity (allocating the full object on the first call).

            :param obj_full: The full multi-band object, or None to allocate it from ``obj_ineq``.
            :param obj_ineq: The per-atom :class:`LocalFourPoint` to insert.
            :param sl: The orbital slice (block) of this atom in the full object.
            :return: The full object with this atom's block filled in.
            """
            if obj_full is None:
                obj_full = obj_ineq.copy()
                obj_full.mat = np.zeros(
                    (config.sys.n_bands,) * 4 + obj_ineq.current_shape[4:], dtype=obj_ineq.mat.dtype
                )
                obj_full.update_original_shape()
            obj_full[sl, sl, sl, sl] = obj_ineq.mat
            return obj_full

        def write_to_full_2pt_quantity(
            obj_full, obj_ineq: SelfEnergy | GreensFunction, sl: slice, has_momentum: bool = True
        ):
            """
            Writes a single inequivalent atom's two-point quantity into the orbital-diagonal block of the assembled
            full multi-band quantity (allocating the full object on the first call), resetting the self-energy moments.

            :param obj_full: The full multi-band object, or None to allocate it from ``obj_ineq``.
            :param obj_ineq: The per-atom :class:`SelfEnergy` or :class:`GreensFunction` to insert.
            :param sl: The orbital slice (block) of this atom in the full object.
            :param has_momentum: Whether the object carries leading momentum axes ``[1, 1, 1, ...]``.
            :return: The full object with this atom's block filled in.
            """
            if obj_full is None:
                obj_full = obj_ineq.copy()
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
            """
            Writes a single inequivalent atom's self-energy high-frequency moments into the orbital-diagonal block of
            the assembled full self-energy.

            :param obj_full: The full multi-band :class:`SelfEnergy`.
            :param obj_ineq: The per-atom :class:`SelfEnergy` whose moments are copied.
            :param sl: The orbital slice (block) of this atom in the full object.
            :return: The full self-energy with this atom's moment block filled in.
            """
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
        del chi_d, chi_m

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

    giwk_dga = GreensFunction.get_g_full(sigma_dga, config.sys.mu, ek, config.sys.beta)

    if config.ana_cont.do_spectrum_dga:
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
                r"$\mathrm{D}\Gamma\mathrm{A Spectrum}",
                output_dir=config.output.output_path,
                name="dga",
            )
            logger.info("Plotted DGA spectrum.")
        del spectrum

    if config.ana_cont.do_spectrum_dmft:
        g_latt = None
        if comm.rank == 0:
            g_latt = GreensFunction(
                np.load(os.path.join(config.output.output_path, "g_latt_dmft.npy")),
                nk=config.lattice.nk,
                beta=config.sys.beta,
            ).cut_niv(config.box.niv_core)
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
                r"$\mathrm{DMFT Spectrum}$",
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
        # sigma_dga is already saved to disk and never consumed by the Eliashberg step - drop the replicated
        # full-grid copy on every rank before the memory-heavy vertex construction
        sigma_dga.free()
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


def autodetect_memory_settings(comm: MPI.Comm) -> None:
    """
    Sets the four ``config.memory.save_memory_*`` switches automatically from the host memory available on every node
    the job runs on and an analytic estimate of the peak memory each affected operation consumes; the flag-less
    Schwinger-Dyson contraction (always the two-pass FFT path) is verified to fit as well. Must be called only after
    the irreducible BZ is known (i.e. after auto-symmetry discovery), as the estimate depends on ``q_grid.nk_irr``.

    The budget is a **node total**: on a node with ``r`` ranks the memory held by all of them at a branch's peak is
    ``r * (baseline + distributed) + single`` (every rank holds the branch's persistent baseline; a *distributed*
    transient is held by every rank at once, a *single-rank* transient by one rank while the others idle), minus
    ``(r - 1) * giwk_shareable`` when ``config.memory.use_shared_memory_common_obj`` deduplicates the branch's ``giwk_full``
    to one copy per node, and this must not exceed ``psutil.virtual_memory().available * 0.9`` for that node. Each
    node's rank count and available memory are collected with a single ``allgather`` of
    ``(hostname, available_bytes)``; a branch's path is judged to "fit" only if it fits on **every** node (the flags
    are process-wide, so the tightest node governs, and a single-rank transient may land on any node). The
    ``lanczos`` fast-path single-rank peak is doubled on a single-node multi-rank job because the singlet and triplet
    solves then run concurrently on the same node. For each branch the fast path is checked and the flag is switched
    on if it would not fit -- but an explicit ``True`` from the config is always kept (floor semantics:
    ``final = user_flag or autodetect_on``). A :class:`MemoryError` is raised only if the path that would actually
    run does not fit.

    :param comm: The MPI communicator (used to group ranks by node).
    :return: None.
    :raises MemoryError: If the code path selected for some branch overflows some node's budget.
    """
    logger = config.logger

    # Gather (hostname, available bytes) from every rank in a single collective and reduce to one entry per node:
    # the rank count and the (minimum, conservative) available memory on that node.
    node_available = psutil.virtual_memory().available
    hostname = str(MPI.Get_processor_name()).strip()
    nodes: dict[str, list] = {}
    for host, avail in comm.allgather((hostname, node_available)):
        if host not in nodes:
            nodes[host] = [0, avail]
        nodes[host][0] += 1
        nodes[host][1] = min(nodes[host][1], avail)

    niv_pp = min(config.box.niw_core // 2, config.box.niv_core // 2)
    # Must mirror the giwk_full window the SDE section starts from in nonlocal_sde.calculate_self_energy_q.
    niv_cut = min(config.box.niw_core + config.box.niv_full + 10, config.box.niv_dmft)
    peaks = memory_estimator.estimate_peaks(
        n_bands=config.sys.n_bands,
        nk_tot=config.lattice.q_grid.nk_tot,
        nk_irr=config.lattice.q_grid.nk_irr,
        niw_core=config.box.niw_core,
        niv_core=config.box.niv_core,
        niv_full=config.box.niv_full,
        niv_cut=niv_cut,
        niv_pp=niv_pp,
        n_ranks=comm.size,
        with_eliashberg=config.eliashberg.perform_eliashberg,
        save_fq=config.eliashberg.save_fq,
        construct_fq_cheap=config.eliashberg.construct_fq_cheap,
        save_pairing_vertex=config.eliashberg.save_pairing_vertex,
        n_eig=config.eliashberg.n_eig,
    )

    # The singlet and triplet in-memory Eliashberg solves run concurrently on two ranks; on a single-node multi-rank
    # job both land on the same node, so its lanczos fast-path single-rank peak is doubled.
    single_node_multi_rank = len(nodes) == 1 and comm.size >= 2

    def node_total(bp: memory_estimator.BranchPeak, distributed: float, single: float, n_ranks: int) -> float:
        """Memory held on a node with ``n_ranks`` ranks at a branch's peak (see :func:`autodetect_memory_settings`).
        When ``config.memory.use_shared_memory_common_obj`` is on, the branch's shareable ``giwk_full`` is counted once per
        node instead of once per rank."""
        total = n_ranks * (bp.baseline + distributed) + single
        if config.memory.use_shared_memory_common_obj:
            total -= (n_ranks - 1) * bp.giwk_shareable
        return total

    def fits_everywhere(bp: memory_estimator.BranchPeak, distributed: float, single: float) -> bool:
        """Whether a transient (per-rank ``distributed`` + one-off ``single``) fits the 90% budget on every node."""
        return all(node_total(bp, distributed, single, r) <= avail * 0.9 for r, avail in nodes.values())

    flag_to_key = {
        "save_memory_for_chi0q": "chi0q",
        "save_memory_for_chiq_aux": "chiq_aux",
        "save_memory_for_fq": "fq",
        "save_memory_for_lanczos": "lanczos",
    }
    key_to_label = {
        "chi0q": "Bare bubble",
        "chiq_aux": "Auxiliary susceptibility",
        "fq": "Full vertex",
        "lanczos": "Eliashberg solver",
    }

    logger.info(f"Auto memory detection (node-total budget): {len(nodes)} node(s).")

    # The Schwinger-Dyson contraction has no save_memory switch (the q-loop variant is unused - it peaked HIGHER
    # than the two-pass FFT path); its single path is still checked so an oversized box fails fast, not mid-run.
    if "sde" in peaks:
        bp_sde = peaks["sde"]
        if not fits_everywhere(bp_sde, bp_sde.off_distributed, bp_sde.off_single):
            worst = max(node_total(bp_sde, bp_sde.off_distributed, bp_sde.off_single, r) for r, _ in nodes.values())
            raise MemoryError(
                f"The Schwinger-Dyson equation needs {worst / 1024**3:.3f} GB on a node, which exceeds 90% of that "
                f"node's available memory. Use more nodes, fewer ranks per node, a smaller frequency box or k-grid."
            )
        worst_sde = max(node_total(bp_sde, bp_sde.off_distributed, bp_sde.off_single, r) for r, _ in nodes.values())
        logger.info(
            f"Schwinger-Dyson equation: per-rank baseline {bp_sde.baseline / 1024**3:.3f} GB, "
            f"node total {worst_sde / 1024**3:.3f} GB (single FFT path, no memory-saving switch)."
        )
    for attr, key in flag_to_key.items():
        if key not in peaks:
            continue
        bp = peaks[key]
        label = key_to_label[key]
        off_single = bp.off_single * (2 if key == "lanczos" and single_node_multi_rank else 1)
        fits_off = fits_everywhere(bp, bp.off_distributed, off_single)
        fits_on = fits_everywhere(bp, bp.on_distributed, bp.on_single)
        autodetect_on = not fits_off
        final = bool(getattr(config.memory, attr)) or autodetect_on
        if final and not fits_on:
            worst = max(node_total(bp, bp.on_distributed, bp.on_single, r) for r, _ in nodes.values())
            raise MemoryError(
                f"The memory-saving path for '{label}' needs {worst / 1024**3:.3f} GB on a node, which exceeds 90% "
                f"of that node's available memory"
                + (
                    " (and its fast path does not fit either)"
                    if autodetect_on
                    else " (its fast path would fit; unset the save_memory flag)"
                )
                + ". Use more nodes, fewer ranks per node, a smaller frequency box or k-grid."
            )
        setattr(config.memory, attr, final)
        worst_off = max(node_total(bp, bp.off_distributed, off_single, r) for r, _ in nodes.values())
        logger.info(
            f"{label}: per-rank baseline {bp.baseline / 1024**3:.3f} GB, fast-path node total "
            f"{worst_off / 1024**3:.3f} GB -> memory saving {'enabled' if final else 'disabled'}."
        )


def _disable_restrict_chi_phys_with_lambda_correction() -> None:
    """
    Disables the susceptibility restriction when the lambda correction is active: the lambda correction calibrates
    its sum rule on the physical susceptibility, which must not contain eigenvalue-floored blocks. The combination
    is not recommended and the lambda correction takes precedence.

    :return: None.
    """
    if config.self_consistency.restrict_chi_phys:
        config.self_consistency.restrict_chi_phys = False
        config.logger.warning(
            "Both the lambda correction and restrict_chi_phys were enabled - this combination is not recommended "
            "since the lambda correction would calibrate its sum rule on eigenvalue-floored susceptibilities. "
            "Keeping the lambda correction and disabling restrict_chi_phys."
        )


def setup_lambda_correction_settings(comm: MPI.Comm) -> None:
    """
    Sets up the lambda correction settings based on the configuration provided by the user. If the user has enabled
    the lambda correction in the self-consistency settings, it will be enabled in the lambda correction settings
    as well. If the user has enabled the lambda correction in the lambda correction settings, but not in the
    self-consistency settings, the self-consistency will be set to a single iteration with full mixing. Will raise
    an error if the user tries to enable the lambda correction for multi-band systems.

    :param comm: The MPI communicator (only rank 0 validates the multi-band restriction).
    :return: None.
    :raises ValueError: If lambda correction is requested for a multi-band system, or the lambda/self-consistency
        settings are inconsistent.
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
        _disable_restrict_chi_phys_with_lambda_correction()
        return

    if config.lambda_correction.perform_lambda_correction:
        config.self_consistency.max_iter = 1
        config.self_consistency.mixing = 1.0
        config.logger.info("Performing one-shot DGA with lambda correction.")
        _disable_restrict_chi_phys_with_lambda_correction()
        return
    elif not config.lambda_correction.perform_lambda_correction:
        config.self_consistency.max_iter = 1
        config.self_consistency.mixing = 1.0
        config.logger.info("Performing one-shot DGA without lambda correction.")
        return

    raise ValueError("Invalid configuration for lambda correction and self-consistency. Please review the config file.")


def configure_matplotlib():
    """
    Configures matplotlib to use the Euler font for mathematical expressions if it is available on the system. This is
    done because the Euler font is the default math font in my thesis.

    :return: None.
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
    main()

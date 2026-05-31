# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

from dgamore.ana_cont import AnalyticContinuationProblem, RealFreqTwoPoint
from dgamore.greens_function import *
from dgamore.mpi_distributor import *


def orbital_to_band_basis(hk: np.ndarray, data: np.ndarray) -> np.ndarray:
    assert hk.shape == data.shape, "Shape mismatch!"

    nkx, nky, nkz, n_orb, _ = hk.shape
    for ix in range(nkx):
        for iy in range(nky):
            for iz in range(nkz):
                w, v = np.linalg.eigh(hk[ix, iy, iz])
                data[ix, iy, iz] = v.conj().T @ data[ix, iy, iz] @ v
    return data


def perform_maxent_giwk(giwk: GreensFunction, name: str, comm: MPI.Comm):
    logger = config.logger

    logger.info(f"Starting analytic continuation of the {name} Green's function using the maximum entropy method.")
    giwk_maxent = giwk.cut_niv(config.box.niv_core).to_half_niv_range()

    irrq_list = config.lattice.k_grid.get_irrq_list()

    mpi_dist = MpiDistributor(ntasks=len(irrq_list), comm=comm, name="Maxent_G")

    giwk_maxent = giwk_maxent.reduce_q(irrq_list)
    logger.info("Scattering Green's function in the IBZ to all ranks.")
    giwk_maxent.mat = mpi_dist.scatter(giwk_maxent.mat)  # each rank now has a slice of the irr BZ

    wn = np.pi / config.sys.beta * (2 * np.arange(config.box.niv_core) + 1)
    w = (
        15
        * np.tan(np.linspace(-np.pi / 2.1, np.pi / 2.1, num=config.ana_cont.w_count, endpoint=True))
        / np.tan(np.pi / 2.1)
    )
    model = np.ones_like(w)
    model /= np.trapezoid(model)
    stdev = np.array([0.0001 for _ in range(config.box.niv_core)])

    spectral_function = np.zeros((len(mpi_dist.my_tasks), config.sys.n_bands, len(w)), dtype=np.float32)

    for band in range(config.sys.n_bands):
        logger.info(f"Processing analytic continuation of band {band+1}.")
        for k in range(giwk_maxent.mat.shape[0]):
            try:
                probl_maxent = AnalyticContinuationProblem(
                    im_axis=wn, re_axis=w, im_data=giwk_maxent[k, band, band], beta=config.sys.beta
                )
                result = probl_maxent.solve(model=model, stdev=stdev)[0]
                spectral_function[k, band] = result.A_opt.astype(np.float32)

                del probl_maxent, result
                gc.collect()
            except Exception:
                spectral_function[k, band] = 0.0
        mpi_dist.comm.barrier()
        logger.info(f"Completed analytic continuation of band {band+1}.")
    spectral_function = mpi_dist.gather(spectral_function)
    logger.info("Analytic continuation of Green's function finished.")

    if mpi_dist.comm.rank == 0:
        spectral_function = spectral_function[config.lattice.k_grid.irrk_inv]  # map the spectral function to the FBZ

        np.save(os.path.join(config.output.output_path, "spectral_function.npy"), spectral_function)
        logger.info(f"Saved {name} spectral function for the full BZ to file.")

    mpi_dist.delete_file()
    return spectral_function


def perform_maxent_dmft(sigma_dmft: SelfEnergy, hk: np.ndarray) -> np.ndarray:
    logger = config.logger

    logger.info(f"Starting analytic continuation of the DMFT Green's function using the maximum entropy method.")
    sigma_maxent = sigma_dmft.to_half_niv_range().mat[0, 0, 0]
    hartree = np.array([np.max(sigma_maxent[i, i].real) for i in range(config.sys.n_bands)]) - 1e-3

    wn = np.pi / config.sys.beta * (2 * np.arange(sigma_maxent.shape[-1]) + 1)
    w = (
        15
        * np.tan(np.linspace(-np.pi / 2.1, np.pi / 2.1, num=config.ana_cont.w_count, endpoint=True))
        / np.tan(np.pi / 2.1)
    )
    model = np.ones_like(w)
    model /= np.trapezoid(model)
    stdev = np.array([0.0001 for _ in range(sigma_maxent.shape[-1])])

    siw_cont = np.zeros((config.sys.n_bands, config.sys.n_bands, len(w)), dtype=np.complex64)

    for band in range(config.sys.n_bands):
        logger.info(f"Processing analytic continuation of band {band+1}.")
        try:
            probl_maxent = AnalyticContinuationProblem(
                im_axis=wn, re_axis=w, im_data=sigma_maxent[band, band] - hartree[band], beta=config.sys.beta
            )
            result = probl_maxent.solve(model=model, stdev=stdev)[0]
            a_opt = result.A_opt

            del probl_maxent, result
            gc.collect()
        except Exception:
            continue
        logger.info(f"Completed analytic continuation of band {band+1}.")
        siw_cont[band, band] = RealFreqTwoPoint(spectrum=a_opt, wgrid=w, kind="fermionic").kkt() + hartree[band]

    eye_bands = np.eye(config.sys.n_bands)
    g_cont = (
        w * eye_bands[None, None, None, ..., None]
        - hk[..., None]
        + config.sys.mu * eye_bands[None, None, None, ..., None]
        - siw_cont[None, None, None, ...]
    )
    g_cont = np.linalg.inv(g_cont.transpose(0, 1, 2, 5, 3, 4)).transpose(0, 1, 2, 4, 5, 3)
    logger.info("Analytic continuation of Green's function finished.")

    spectral_function = np.moveaxis(np.diagonal(-1 / np.pi * g_cont.imag, axis1=-2, axis2=-3), -2, -1)
    np.save(os.path.join(config.output.output_path, "spectral_function_dmft.npy"), spectral_function)
    logger.info("Saved DMFT spectral function for the full BZ to file.")

    return spectral_function

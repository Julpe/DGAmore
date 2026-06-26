# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Analytic continuation of imaginary-frequency quantities to the real axis via the maximum-entropy method. This
module wraps the (vendored) :mod:`dgamore.ana_cont` solver to continue the momentum-dependent DGA Green's function
and the local DMFT Green's function to real frequencies, yielding the spectral function
:math:`A(\mathbf{k}, \omega)`. The momentum-resolved continuation is distributed over MPI ranks across the
irreducible BZ.
"""

import gc
import os

import numpy as np
from mpi4py import MPI

import dgamore.config as config
from dgamore.ana_cont import AnalyticContinuationProblem, RealFreqTwoPoint
from dgamore.greens_function import GreensFunction
from dgamore.mpi_distributor import MpiDistributor
from dgamore.n_point_base import DTYPE
from dgamore.self_energy import SelfEnergy


def orbital_to_band_basis(hk: np.ndarray, data: np.ndarray) -> np.ndarray:
    r"""
    Rotates a momentum-dependent quantity from the orbital basis into the band (eigen) basis of the Hamiltonian,
    per k-point: :math:`O^{\mathrm{band}}(k) = U(k)^\dagger O(k) U(k)`, where the columns of :math:`U(k)` are the
    (energy-ascending) eigenvectors of :math:`H(k)`. The Hamiltonian is diagonalized only once per k-point; if a
    trailing fermionic-frequency axis is present, the same :math:`U(k)` is reused for every frequency.

    Note that the band basis is defined by :math:`H(k)` alone, so for an interacting quantity whose self-energy is
    not simultaneously diagonal with :math:`H(k)` the rotated object is not exactly diagonal -- the off-diagonal
    band components are kept here and only discarded later when the band-diagonal is taken. Within a degenerate
    eigenspace of :math:`H(k)` the individual band assignment is basis-dependent (the subspace trace is not).

    :param hk: The Hamiltonian :math:`H(k)` of shape ``[kx, ky, kz, n_orb, n_orb]``.
    :param data: The quantity to rotate, of shape ``[kx, ky, kz, n_orb, n_orb]`` or, with a trailing fermionic
        frequency axis, ``[kx, ky, kz, n_orb, n_orb, n_v]`` (rotated in place and returned).
    :return: The quantity in the band basis (same shape as ``data``).
    :raises AssertionError: If the momentum/orbital axes of ``hk`` and ``data`` do not match.
    """
    has_frequency_axis = data.ndim == hk.ndim + 1
    assert data.shape[: hk.ndim] == hk.shape and (has_frequency_axis or data.shape == hk.shape), "Shape mismatch!"

    nkx, nky, nkz, n_orb, _ = hk.shape
    for ix in range(nkx):
        for iy in range(nky):
            for iz in range(nkz):
                w, v = np.linalg.eigh(hk[ix, iy, iz])
                if has_frequency_axis:
                    data[ix, iy, iz] = np.einsum("ai,abv,bj->ijv", v.conj(), data[ix, iy, iz], v, optimize=True)
                else:
                    data[ix, iy, iz] = v.conj().T @ data[ix, iy, iz] @ v
    return data


def perform_maxent_giwk(giwk: GreensFunction, name: str, comm: MPI.Comm):
    r"""
    Analytically continues the momentum-dependent Green's function to the real axis via maximum entropy, per
    band and per irreducible-BZ k-point, and assembles the spectral function over the full BZ on rank 0. The
    k-points are distributed across MPI ranks; failed continuations are set to zero.

    :param giwk: The momentum-dependent :class:`GreensFunction` to continue.
    :param name: Label used in log messages (e.g. ``"DGA"``).
    :param comm: The MPI communicator.
    :return: The spectral function :math:`A(\mathbf{k}, \omega)` of shape ``[k, n_bands, w]`` (full BZ on rank 0).
    """
    logger = config.logger

    logger.info(f"Starting analytic continuation of the {name} Green's function using the maximum entropy method.")
    giwk_maxent = giwk.cut_niv(config.box.niv_core).to_half_niv_range()

    # Rotate the Green's function into the band (H(k)-eigen) basis before the continuation, so that the
    # band-diagonal taken below is the band-resolved spectral function rather than the orbital-diagonal one. This
    # is also what makes the naive irrk_inv unfold to the full BZ correct: the band-diagonal spectrum is invariant
    # under the lattice symmetries (eigenvalues of H(k) and H(k_rep) coincide). Done on the full BZ here so it
    # matches the shape of H(k) from get_ek; the diagonalization is reused across all frequencies.
    hk = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
    giwk_maxent = giwk_maxent.decompress_q_dimension()
    orbital_to_band_basis(hk, giwk_maxent.mat)

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
    r"""
    Analytically continues the local DMFT self-energy to the real axis via maximum entropy (per band, with the
    Hartree shift removed and restored through a Kramers-Kronig transform), then builds the real-frequency lattice
    Green's function and its spectral function.

    :param sigma_dmft: The local DMFT :class:`SelfEnergy`.
    :param hk: The Hamiltonian :math:`H(k)` of shape ``[kx, ky, kz, n_orb, n_orb]``.
    :return: The spectral function :math:`A(\mathbf{k}, \omega)` of shape ``[kx, ky, kz, n_bands, w]``.
    """
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

    siw_cont = np.zeros((config.sys.n_bands, config.sys.n_bands, len(w)), dtype=DTYPE)

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

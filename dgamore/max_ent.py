# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Analytic continuation of imaginary-frequency quantities to the real axis via the maximum-entropy method. This module
wraps the (vendored) :mod:`dgamore.ana_cont` solver to continue the momentum-dependent DGA Green's function and the
local DMFT Green's function to real frequencies, yielding the spectral function :math:`A(\mathbf{k}, \omega)`. The
momentum-resolved continuation is distributed over MPI ranks across the irreducible BZ.
"""

import contextlib
import gc
import io
import os
import warnings

import numpy as np
from mpi4py import MPI
from scipy.optimize import OptimizeWarning

import dgamore.config as config
from dgamore.ana_cont import AnalyticContinuationProblem, RealFreqTwoPoint
from dgamore.greens_function import GreensFunction
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import DTYPE
from dgamore.self_energy import SelfEnergy


def orbital_to_band_basis(hk: np.ndarray, data: np.ndarray) -> np.ndarray:
    r"""
    Rotates a momentum-dependent quantity from the orbital basis into the band (eigen) basis of the Hamiltonian, per
    k-point: :math:`O^{\mathrm{band}}(\mathbf{k}) = U(\mathbf{k})^\dagger O(\mathbf{k}) U(\mathbf{k})`, where the
    columns of :math:`U(\mathbf{k})` are the (energy-ascending) eigenvectors of :math:`H(\mathbf{k})`. The Hamiltonian
    is diagonalized only once per k-point; if a trailing fermionic-frequency axis is present, the same
    :math:`U(\mathbf{k})` is reused for every frequency.

    Note that the band basis is defined by :math:`H(\mathbf{k})` alone, so for an interacting quantity whose self-energy
    is not simultaneously diagonal with :math:`H(\mathbf{k})` the rotated object is not exactly diagonal -- the
    off-diagonal band components are kept here and only discarded later when the band-diagonal is taken. Within a
    degenerate eigenspace of :math:`H(\mathbf{k})` the individual band assignment is basis-dependent (the subspace trace
    is not).

    :param hk: The Hamiltonian :math:`H(\mathbf{k})` of shape ``[kx, ky, kz, n_orb, n_orb]``.
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
    :param name: Label of the continued quantity (e.g. ``"DGA"``), used in the log messages and, lowercased, in the
        output file name ``spectral_function_<name>.npy``.
    :param comm: The MPI communicator.
    :return: The spectral function :math:`A(\mathbf{k}, \omega)` of shape ``[kx, ky, kz, n_bands, w]`` (full BZ on
        rank 0), i.e. with a decompressed momentum dimension, like :func:`perform_maxent_dmft`.
    """
    logger = config.logger

    logger.info(f"Starting analytic continuation of the {name} Green's function using the maximum entropy method.")
    irrq_list = config.lattice.k_grid.get_irrq_list()
    mpi_dist = MpiDistributor(ntasks=len(irrq_list), comm=comm, name="Maxent_G", output_path=config.output.output_path)

    # The full-BZ preparation runs on rank 0 only: every rank used to build the identical core-cut, band-rotated,
    # irr-reduced Green's function just to keep its own scatter slice - multi-GB transients times the rank count.
    g_irr_mat = None
    if comm.rank == 0:
        giwk_maxent = giwk.cut_niv(config.box.niv_core).to_half_niv_range()

        # Rotate G into the band (H(k)-eigen) basis first, so the band-diagonal below is the band-resolved spectral
        # function (invariant under lattice symmetries, so the naive irrk_inv unfold is correct); on the full BZ.
        hk = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
        giwk_maxent = giwk_maxent.decompress_q_dimension()
        orbital_to_band_basis(hk, giwk_maxent.mat)
        g_irr_mat = giwk_maxent.reduce_q(irrq_list).mat

    logger.info("Scattering Green's function in the IBZ to all ranks.")
    g_irr_slice = mpi_dist.scatter(g_irr_mat)  # each rank now has a slice of the irr BZ

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
        for k in range(g_irr_slice.shape[0]):
            # Capture the vendored solver's stdout so its print() diagnostics go through the logger instead of
            # leaking to the output; re-logged (prefixed) below whether the continuation succeeds or fails.
            captured_output = io.StringIO()
            try:
                with warnings.catch_warnings(), contextlib.redirect_stdout(captured_output):
                    # Escalate numpy/scipy RuntimeWarnings (divide/invalid/overflow) to exceptions so a numerically
                    # broken continuation falls through to the A(k, w) = 0 fallback below (its own errors are caught too).
                    warnings.simplefilter("error", RuntimeWarning)
                    # The alpha-fit curve_fit inside the solver harmlessly fails to estimate its covariance; mute it.
                    warnings.simplefilter("ignore", OptimizeWarning)
                    probl_maxent = AnalyticContinuationProblem(
                        im_axis=wn, re_axis=w, im_data=g_irr_slice[k, band, band], beta=config.sys.beta
                    )
                    result = probl_maxent.solve(model=model, stdev=stdev)[0]
                    spectral_function[k, band] = result.A_opt.astype(np.float32)

                del probl_maxent, result
                gc.collect()
            except Exception:
                kpt = tuple(int(c) for c in irrq_list[mpi_dist.my_tasks[k]])
                logger.info(
                    f"Failed to determine analytic continuation of k={kpt} (band {band + 1}), "
                    f"setting A(k={kpt}, w) = 0.0."
                )
                spectral_function[k, band] = 0.0
            finally:
                for message in captured_output.getvalue().splitlines():
                    if message.strip():
                        logger.info(f"ana_cont: {message}")
        mpi_dist.comm.barrier()
        logger.info(f"Completed analytic continuation of band {band+1}.")
    spectral_function = mpi_dist.gather(spectral_function)
    logger.info("Analytic continuation of Green's function finished.")

    if mpi_dist.comm.rank == 0:
        # map to the FBZ through a flattened index map, then decompress explicitly: numpy's ``return_inverse`` shape
        # changed across 2.x, so indexing with ``irrk_inv`` alone would leave a version-dependent momentum layout
        spectral_function = spectral_function[np.reshape(config.lattice.k_grid.irrk_inv, -1)]
        spectral_function = spectral_function.reshape(*config.lattice.k_grid.nk, *spectral_function.shape[1:])

        # the file carries the label, so continuing several quantities in one run does not overwrite the earlier ones
        np.save(os.path.join(config.output.output_path, f"spectral_function_{name.lower()}.npy"), spectral_function)
        logger.info(f"Saved {name} spectral function for the full BZ to file.")

    mpi_dist.delete_file()
    return spectral_function


def perform_maxent_dmft(sigma_dmft: SelfEnergy, hk: np.ndarray) -> np.ndarray:
    r"""
    Analytically continues the local DMFT self-energy to the real axis via maximum entropy (per band, with the
    Hartree shift removed and restored through a Kramers-Kronig transform), then builds the real-frequency lattice
    Green's function and its spectral function.

    :param sigma_dmft: The local DMFT :class:`SelfEnergy`.
    :param hk: The Hamiltonian :math:`H(\mathbf{k})` of shape ``[kx, ky, kz, n_orb, n_orb]``.
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
            with warnings.catch_warnings():
                # The alpha-fit curve_fit inside the solver harmlessly fails to estimate its covariance; mute it.
                warnings.simplefilter("ignore", OptimizeWarning)
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

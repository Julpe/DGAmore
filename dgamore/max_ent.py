# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Analytic continuation of imaginary-frequency quantities to the real axis via the maximum-entropy method. This module
wraps the (vendored) :mod:`dgamore.ana_cont` solver to continue the momentum-dependent DGA Green's function and the
local DMFT Green's function to real frequencies, yielding the spectral function :math:`A(\mathbf{k}, \omega)`, band
resolved and as the full orbital matrix. The momentum-resolved continuation is distributed over MPI ranks across the
irreducible BZ.
"""

import contextlib
import gc
import io
import itertools
import os
import warnings

import numpy as np
from mpi4py import MPI
from scipy.optimize import OptimizeWarning
from threadpoolctl import threadpool_limits

import dgamore.config as config
from dgamore.ana_cont import AnalyticContinuationProblem, RealFreqTwoPoint
from dgamore.greens_function import GreensFunction, orbital_block
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import DTYPE
from dgamore.self_energy import SelfEnergy


def orbital_to_band_basis(hk: np.ndarray, data: np.ndarray, inverse: bool = False) -> np.ndarray:
    r"""
    Rotates a momentum-dependent quantity from the orbital basis into the band (eigen) basis of the Hamiltonian, per
    k-point: :math:`O^{\mathrm{band}}(\mathbf{k}) = U(\mathbf{k})^\dagger O(\mathbf{k}) U(\mathbf{k})`, where the
    columns of :math:`U(\mathbf{k})` are the (energy-ascending) eigenvectors of :math:`H(\mathbf{k})`, or with
    ``inverse`` back into the orbital basis, :math:`O(\mathbf{k}) = U(\mathbf{k}) O^{\mathrm{band}}(\mathbf{k})
    U(\mathbf{k})^\dagger`. The Hamiltonian is diagonalized only once per k-point; if a trailing frequency axis is
    present, the same :math:`U(\mathbf{k})` is reused for every frequency. The same :math:`H(\mathbf{k})` yields the
    same :math:`U(\mathbf{k})` in both directions, so the inverse undoes the forward rotation exactly, off-diagonal
    band components included.

    Note that the band basis is defined by :math:`H(\mathbf{k})` alone, so for an interacting quantity whose self-energy
    is not simultaneously diagonal with :math:`H(\mathbf{k})` the rotated object is not exactly diagonal -- the
    off-diagonal band components are kept here. Within a degenerate eigenspace of :math:`H(\mathbf{k})` the individual
    band assignment is basis-dependent (the subspace trace is not).

    :param hk: The Hamiltonian :math:`H(\mathbf{k})` of shape ``[..., n_orb, n_orb]``, with any momentum layout in front
        of the orbital axes (e.g. ``[kx, ky, kz, n_orb, n_orb]`` or ``[k, n_orb, n_orb]``).
    :param data: The quantity to rotate, of the shape of ``hk`` or, with a trailing frequency axis,
        ``[..., n_orb, n_orb, n_v]`` (rotated in place and returned).
    :param inverse: If True, rotates from the band basis back into the orbital basis.
    :return: The rotated quantity (same shape as ``data``).
    :raises AssertionError: If the momentum/orbital axes of ``hk`` and ``data`` do not match.
    """
    has_frequency_axis = data.ndim == hk.ndim + 1
    assert data.shape[: hk.ndim] == hk.shape and (has_frequency_axis or data.shape == hk.shape), "Shape mismatch!"

    for k in np.ndindex(hk.shape[:-2]):
        v = np.linalg.eigh(hk[k])[1]
        if inverse:
            v = v.conj().T
        if has_frequency_axis:
            data[k] = np.einsum("ai,abv,bj->ijv", v.conj(), data[k], v, optimize=True)
        else:
            data[k] = v.conj().T @ data[k] @ v
    return data


def _continue(
    problem: AnalyticContinuationProblem | None,
    data: np.ndarray,
    wn: np.ndarray,
    w: np.ndarray,
    model: np.ndarray,
    stdev: np.ndarray,
    offdiag: bool = False,
) -> tuple[AnalyticContinuationProblem | None, np.ndarray | None]:
    r"""
    Runs one maximum-entropy continuation of ``data``, given on the positive Matsubara frequencies ``wn``, onto the
    real frequencies ``w``. The continuation problem is built on the first call and reused afterwards, where only the
    data and, for the off-diagonal solver, the default model change. The solver's printed diagnostics are re-logged
    with the prefix ``ana_cont:``, and a numpy/scipy RuntimeWarning counts as a failure.

    :param problem: The :class:`AnalyticContinuationProblem` to reuse, or None to build one.
    :param data: The Green's function data on ``wn``.
    :param wn: The positive Matsubara frequencies.
    :param w: The real-frequency grid.
    :param model: The default model on ``w``.
    :param stdev: The assumed standard deviation of ``data`` per frequency.
    :param offdiag: Whether to continue with the positive-negative entropy (an off-diagonal element) instead of the
        positive one.
    :return: The problem to reuse and the spectrum on ``w`` as ``float32``, or None in its place if the continuation
        failed.
    """
    captured_output = io.StringIO()
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(captured_output):
            # Escalate numpy/scipy RuntimeWarnings (divide/invalid/overflow) to exceptions so a numerically broken
            # continuation counts as failed; the alpha-fit curve_fit harmlessly fails to estimate its covariance.
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("ignore", OptimizeWarning)
            if problem is None:
                problem = AnalyticContinuationProblem(im_axis=wn, re_axis=w, im_data=data, beta=config.sys.beta)
            elif offdiag:
                problem.update_data(data, model=model)
            else:
                problem.update_data(data)
            spectrum = problem.solve(model=model, stdev=stdev, offdiag=offdiag)[0].A_opt.astype(np.float32)
        gc.collect()
    except Exception:
        spectrum = None
    finally:
        for message in captured_output.getvalue().splitlines():
            if message.strip():
                config.logger.info(f"ana_cont: {message}")
    return problem, spectrum


def perform_maxent_giwk(giwk: GreensFunction | None, name: str, comm: MPI.Comm):
    r"""
    Analytically continues the momentum-dependent Green's function to the real axis via maximum entropy, per
    irreducible-BZ k-point in the band (:math:`H(\mathbf{k})`-eigen) basis, with uncorrelated orbitals in the eigenbasis
    of the correlated block :math:`H_{dd}(\mathbf{k})` the Green's function holds. Every band-diagonal element
    is continued with the positive entropy. Every band pair continues the Hermitian part :math:`(G_{12} + G_{21})/2`
    and the anti-Hermitian part :math:`(G_{12} - G_{21})/(2i)` with the positive-negative entropy and the default model
    :math:`\sqrt{A_{11} A_{22}}` (floored at :math:`10^{-6}` of its maximum): their real spectra are
    :math:`\mathrm{Re}\, A_{12}` and :math:`\mathrm{Im}\, A_{12}`. A part whose data stay within the assumed noise at
    every frequency (zero by a symmetry of the little group of :math:`\mathbf{k}`, or the anti-Hermitian part of a
    symmetric :math:`G`) is zero without a fit, as is every pair of a band whose continuation failed. The k-points are
    distributed across MPI ranks; failed continuations are set to zero. The continuation problems (kernel, its SVD and
    the preblur convolution) are set up once per rank, one for the diagonal and one for the off-diagonal elements, and
    reused for every k-point and band.

    Rank 0 writes the band-resolved spectral function over the full BZ to ``spectral_function_<name>.npy`` and the
    Hermitian spectral matrix, rotated back into the orbital basis, on the irreducible BZ to
    ``spectral_function_matrix_<name>.npy`` (rows in the order of ``KGrid.irrk_ind``;
    :meth:`TwoPoint.map_to_full_bz` unfolds it with the orbital rotations of the lattice symmetries). The
    element-wise continuation does not enforce positive semidefiniteness, so the saved matrix keeps only the
    non-negative eigenvalues per k-point and frequency, rescaled to the trace of the continued matrix (the
    band-resolved sum), so the cut adds no spectral weight; the eigenvalue range before the cut is logged.

    :param giwk: The momentum-dependent :class:`GreensFunction` to continue, read on rank 0 only (``None`` elsewhere).
    :param name: Label of the continued quantity (e.g. ``"DGA"``), used in the log messages and, lowercased, in the
        output file names ``spectral_function_<name>.npy`` and ``spectral_function_matrix_<name>.npy``.
    :param comm: The MPI communicator.
    :return: The band-resolved spectral function :math:`A(\mathbf{k}, \omega)` of shape ``[kx, ky, kz, n_bands, w]``
        (full BZ on rank 0), i.e. with a decompressed momentum dimension, like :func:`perform_maxent_dmft`.
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

        # Rotate G into the band basis of the H(k) block it holds first, so the band-diagonal below is the band-resolved
        # spectral function (invariant under lattice symmetries, so the naive irrk_inv unfold is correct); full BZ.
        hk = orbital_block(config.lattice.hamiltonian.get_ek(config.lattice.k_grid), config.sys.correlated_orbitals)
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

    n_bands, n_tasks = config.sys.n_bands, len(mpi_dist.my_tasks)
    spectral_function = np.zeros((n_tasks, n_bands, len(w)), dtype=np.float32)
    spectral_matrix = np.zeros((n_tasks, n_bands, n_bands, len(w)), dtype=np.complex64)
    n_continued = np.zeros(n_tasks, dtype=int)

    # One continuation problem per rank and entropy: kernel, SVD and preblur convolution are data independent, so
    # only the data (and the off-diagonal default model) change per fit. BLAS stays single threaded: small matrices.
    probl_maxent = probl_offdiag = None
    with threadpool_limits(limits=1):
        for band in range(n_bands):
            logger.info(f"Processing analytic continuation of band {band+1}.")
            for k in range(n_tasks):
                probl_maxent, spectrum = _continue(probl_maxent, g_irr_slice[k, band, band], wn, w, model, stdev)
                if spectrum is None:
                    kpt = tuple(int(c) for c in irrq_list[mpi_dist.my_tasks[k]])
                    logger.info(
                        f"Failed to determine analytic continuation of k={kpt} (band {band + 1}), "
                        f"setting A(k={kpt}, w) = 0.0."
                    )
                else:
                    spectral_function[k, band] = spectrum
            mpi_dist.comm.barrier()
            logger.info(f"Completed analytic continuation of band {band+1}.")

        bands = np.arange(n_bands)
        spectral_matrix[:, bands, bands] = spectral_function
        for b1, b2 in itertools.combinations(range(n_bands), 2):
            logger.info(f"Processing analytic continuation of bands ({b1 + 1}, {b2 + 1}).")
            for k in range(n_tasks):
                a1, a2 = (np.clip(spectral_function[k, b], 0.0, None).astype(np.float64) for b in (b1, b2))
                default_model = np.sqrt(a1 * a2)
                if not default_model.any():
                    continue
                # the spline-blurred spectra touch zero (or dip a hair below) in their tails, where the
                # positive-negative entropy still needs a positive model
                default_model = np.maximum(default_model, 1e-6 * default_model.max())
                g12, g21 = g_irr_slice[k, b1, b2], g_irr_slice[k, b2, b1]
                for part, unit, data in (("real", 1.0, (g12 + g21) / 2), ("imaginary", 1j, (g12 - g21) / 2j)):
                    if np.all(np.abs(data) <= stdev):
                        continue
                    probl_offdiag, spectrum = _continue(probl_offdiag, data, wn, w, default_model, stdev, offdiag=True)
                    if spectrum is None:
                        kpt = tuple(int(c) for c in irrq_list[mpi_dist.my_tasks[k]])
                        logger.info(
                            f"Failed to determine analytic continuation of the {part} part of bands "
                            f"({b1 + 1}, {b2 + 1}) at k={kpt}, setting it to 0.0."
                        )
                    else:
                        spectral_matrix[k, b1, b2] += unit * spectrum
                        n_continued[k] += 1
            spectral_matrix[:, b2, b1] = spectral_matrix[:, b1, b2].conj()
            mpi_dist.comm.barrier()
            logger.info(f"Completed analytic continuation of bands ({b1 + 1}, {b2 + 1}).")
    spectral_function = mpi_dist.gather(spectral_function)
    spectral_matrix = mpi_dist.gather(spectral_matrix)
    n_continued = mpi_dist.gather(n_continued)
    logger.info("Analytic continuation of Green's function finished.")

    if mpi_dist.comm.rank == 0:
        # map to the FBZ through a flattened index map, then decompress explicitly: numpy's ``return_inverse`` shape
        # changed across 2.x, so indexing with ``irrk_inv`` alone would leave a version-dependent momentum layout
        spectral_function = spectral_function[np.reshape(config.lattice.k_grid.irrk_inv, -1)]
        spectral_function = spectral_function.reshape(*config.lattice.k_grid.nk, *spectral_function.shape[1:])

        # the file carries the label, so continuing several quantities in one run does not overwrite the earlier ones
        np.save(os.path.join(config.output.output_path, f"spectral_function_{name.lower()}.npy"), spectral_function)
        logger.info(f"Saved {name} spectral function for the full BZ to file.")

        # back into the orbital basis with the U(k) of the forward rotation (the same H(k) rows, irrk_ind order)
        hk_irr = hk.reshape(-1, n_bands, n_bands)[config.lattice.k_grid.irrk_ind]
        orbital_to_band_basis(hk_irr, spectral_matrix, inverse=True)

        # the element-wise continuation does not keep A(k, w) positive semidefinite: cut the negative eigenvalues and
        # rescale the rest to the trace (the band-resolved sum), so no spectral weight is added
        eigenvalues, vectors = np.linalg.eigh(np.moveaxis(spectral_matrix, -1, 1))
        del spectral_matrix
        kept = np.clip(eigenvalues, 0.0, None)
        trace = np.clip(eigenvalues.sum(axis=-1), 0.0, None)
        kept *= (trace / np.maximum(kept.sum(axis=-1), np.finfo(kept.dtype).tiny))[..., None]
        spectral_matrix = np.einsum("kwai,kwi,kwbi->kabw", vectors, kept, vectors.conj())
        np.save(
            os.path.join(config.output.output_path, f"spectral_function_matrix_{name.lower()}.npy"), spectral_matrix
        )
        logger.info(
            f"Saved {name} orbital spectral matrix for the irreducible BZ to file: {n_continued.sum()} of "
            f"{n_continued.size * n_bands * (n_bands - 1)} off-diagonal band parts continued (the rest lie within the "
            f"noise or failed), eigenvalues from {eigenvalues.min():.2e} to {eigenvalues.max():.2e} before the "
            f"negative ones were cut and the rest rescaled to the trace."
        )

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

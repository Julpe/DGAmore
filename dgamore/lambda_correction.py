# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Moriya :math:`\lambda`-correction of the physical susceptibility. The non-local susceptibility is shifted by a
single constant :math:`\lambda` (per channel) so that its momentum sum matches the corresponding local
sum-rule. Single-band only, since a multi-orbital correction would be non-unique.
"""

import numpy as np

from dgamore import config
from dgamore.four_point import FourPoint


def get_lambda_start(chi_r: np.ndarray) -> float:
    r"""
    Returns the lower bound for :math:`\lambda`, i.e. the value at which the corrected susceptibility at
    :math:`\omega = 0` would first diverge (:math:`-\min_q 1/\chi_r^{q,\omega=0}`). The search starts just above it.

    :param chi_r: Physical susceptibility in the irreducible BZ with a trailing bosonic frequency axis,
        shape ``[q, w]``.
    :return: The starting (lower-bound) value of :math:`\lambda`.
    """
    w0 = chi_r.shape[-1] // 2
    return -np.min(1.0 / chi_r[..., w0].real)


def apply_lambda(chi_r: np.ndarray, lambda_: float) -> np.ndarray:
    r"""
    Applies the :math:`\lambda`-correction :math:`\chi_r \to (1/\chi_r + \lambda)^{-1}` to the susceptibility.

    :param chi_r: Physical susceptibility array.
    :param lambda_: The correction shift :math:`\lambda`.
    :return: The corrected susceptibility, same shape as ``chi_r``.
    """
    return 1.0 / (1.0 / chi_r + lambda_)


def find_lambda(
    chi_r_mat: np.ndarray,
    chi_r_loc_sum: complex,
    delta: float = 0.1,
    eps: float = 1e-7,
    maxiter: int = 1000,
) -> float:
    r"""
    Finds :math:`\lambda` such that the momentum-summed corrected susceptibility matches the local sum
    ``chi_r_loc_sum`` via a Newton-like iteration. The momentum sum is evaluated over the irreducible BZ using
    the per-point multiplicities. When a Newton step would lower :math:`\lambda` below the current value the
    step size ``delta`` is halved and the search is reset just above the divergence bound.

    :param chi_r_mat: Physical susceptibility in the irreducible BZ, shape ``[q, w]``.
    :param chi_r_loc_sum: Target value: the local susceptibility sum (already divided by :math:`\beta`).
    :param delta: Initial offset above the divergence bound and step size for the bisection-style reset.
    :param eps: Convergence tolerance on the real part of the sum-rule residual.
    :param maxiter: Maximum number of iterations before giving up (logs a warning and returns the last value).
    :return: The converged :math:`\lambda` (or the last value reached if not converged).
    """
    lambda_start = get_lambda_start(chi_r_mat)
    lambda_: float = lambda_start + delta
    factor = 1 / config.sys.beta / config.lattice.q_grid.nk_tot

    for _ in range(maxiter):
        chi_lam = apply_lambda(chi_r_mat, lambda_)
        chir_sum = (config.lattice.q_grid.irrk_count[:, None] * chi_lam).sum() * factor
        f_lam = chir_sum - chi_r_loc_sum
        fp_lam = -(config.lattice.q_grid.irrk_count[:, None] * chi_lam**2).sum() * factor
        # NB: a vanishing fp_lam is intentional here -- the resulting (inf) Newton step is what triggers the
        # delta-halving reset branch below, so it must NOT be guarded against.
        lambda_new = lambda_ - (f_lam / fp_lam).real

        if abs(f_lam.real) < eps:
            return lambda_new

        if lambda_new < lambda_:
            delta /= 2
            lambda_ = lambda_start + delta
        else:
            lambda_ = lambda_new

    logger = getattr(config, "logger", None)
    if logger is not None:
        logger.warning(f"Lambda correction did not converge within {maxiter} iterations.")
    return lambda_


def perform_single_lambda_correction(chi_r: FourPoint, chi_r_loc_sum: complex) -> tuple[FourPoint, float]:
    r"""
    Performs the :math:`\lambda`-correction on the physical susceptibility for a single spin channel. Only works
    for single-band systems, since a multi-orbital correction would be a non-unique multidimensional problem.

    :param chi_r: Physical (single-band) susceptibility in the irreducible BZ.
    :param chi_r_loc_sum: Target local susceptibility sum (already divided by :math:`\beta`).
    :return: A tuple of (i) the corrected susceptibility in the irreducible BZ and half bosonic frequency range,
        and (ii) the determined :math:`\lambda`.
    """
    chi_r = chi_r.to_full_niw_range()
    chi_r_mat = chi_r.compress_q_dimension().mat.squeeze()
    lambda_r = find_lambda(chi_r_mat, chi_r_loc_sum)
    chi_r.mat = apply_lambda(chi_r_mat, lambda_r)[:, None, None, None, None, :]
    return chi_r.to_half_niw_range(), lambda_r

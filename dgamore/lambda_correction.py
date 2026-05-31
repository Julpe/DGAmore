# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import numpy as np

from dgamore import config
from dgamore.four_point import FourPoint


def get_lambda_start(chi_r: np.ndarray) -> float:
    r"""
    Returns the starting value of :math:`\lambda` for the :math:`\lambda`-correction.
    """
    w0 = chi_r.shape[-1] // 2
    return -np.min(1.0 / chi_r[..., w0].real)


def apply_lambda(chi_r: np.ndarray, lambda_: float) -> np.ndarray:
    r"""
    Applies the :math:`\lambda`-correction to the physical susceptibility.
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
    Finds :math:`\lambda` for the correction of the physical susceptibility by an iterative scheme (similar to newton).
    Instead of processing the items in the full Brillouin zone, we make use of the weights of the individual k-points
    in the irreducible Brillouin zone.
    """
    lambda_start = get_lambda_start(chi_r_mat)
    lambda_: float = lambda_start + delta
    factor = 1 / config.sys.beta / config.lattice.q_grid.nk_tot

    for _ in range(maxiter):
        chi_lam = apply_lambda(chi_r_mat, lambda_)
        chir_sum = (config.lattice.q_grid.irrk_count[:, None] * chi_lam).sum() * factor
        f_lam = chir_sum - chi_r_loc_sum
        fp_lam = -(config.lattice.q_grid.irrk_count[:, None] * chi_lam**2).sum() * factor
        lambda_new = lambda_ - (f_lam / fp_lam).real

        if abs(f_lam.real) < eps:
            return lambda_new

        if lambda_new < lambda_:
            delta /= 2
            lambda_ = lambda_start + delta
        else:
            lambda_ = lambda_new

    return lambda_


def perform_single_lambda_correction(chi_r: FourPoint, chi_r_loc_sum: complex) -> tuple[FourPoint, float]:
    """
    Performs the lambda correction on the physical susceptibility for a single spin-channel. Returns (i) the corrected
    susceptibility in the irreducible BZ and half niw range and (ii) lambda as a tuple. Note that this only works
    for single-band systems, since it would yield a non-unique multidimensional problem for multiple orbitals.
    """
    chi_r = chi_r.to_full_niw_range()
    chi_r_mat = chi_r.compress_q_dimension().mat.squeeze()
    lambda_r = find_lambda(chi_r_mat, chi_r_loc_sum)
    chi_r.mat = apply_lambda(chi_r_mat, lambda_r)[:, None, None, None, None, :]
    return chi_r.to_half_niw_range(), lambda_r

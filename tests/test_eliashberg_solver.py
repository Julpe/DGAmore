# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import numpy as np
import pytest

from dgamore.eliashberg_solver import (
    _apply_gamma_pp,
    _apply_gchi0_pp,
    _chi0_to_matmul_layout,
    _gamma_to_matmul_layout,
)


def test_apply_gchi0_pp_matches_einsum():
    """_apply_gchi0_pp reproduces einsum('xyzabcdv,xyzcdv->xyzabv') (the chi0*gap multiplication in the matvec)."""
    rng = np.random.default_rng(0)
    nqx, nqy, nqz, o, v = 3, 4, 2, 3, 6
    chi0 = rng.standard_normal((nqx, nqy, nqz, o, o, o, o, v)) + 1j * rng.standard_normal(
        (nqx, nqy, nqz, o, o, o, o, v)
    )
    gap = rng.standard_normal((nqx, nqy, nqz, o, o, v)) + 1j * rng.standard_normal((nqx, nqy, nqz, o, o, v))
    ref = np.einsum("xyzabcdv,xyzcdv->xyzabv", chi0, gap, optimize=True)
    got = _apply_gchi0_pp(_chi0_to_matmul_layout(chi0), gap.ravel(), o)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=1e-10)


@pytest.mark.parametrize("nv", [6, 3])
def test_apply_gamma_pp_matches_einsum(nv):
    """_apply_gamma_pp reproduces einsum('xyzacbdvp,xyzcdp->xyzabv') for full (nv==np) and frequency-sliced (nv<np) v."""
    rng = np.random.default_rng(1)
    nqx, nqy, nqz, o, npp = 2, 3, 2, 3, 6
    shape = (nqx, nqy, nqz, o, o, o, o, nv, npp)
    gamma = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    gap_gg = rng.standard_normal((nqx, nqy, nqz, o, o, npp)) + 1j * rng.standard_normal((nqx, nqy, nqz, o, o, npp))
    ref = np.einsum("xyzacbdvp,xyzcdp->xyzabv", gamma, gap_gg, optimize=True)
    got = _apply_gamma_pp(_gamma_to_matmul_layout(gamma), gap_gg, o)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=1e-10)

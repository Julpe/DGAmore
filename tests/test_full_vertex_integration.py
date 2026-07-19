# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from dgamore import config, eliashberg_solver, nonlocal_sde
from dgamore.dga_logger import DgaLogger
from dgamore.four_point import FourPoint
from dgamore.interaction import Interaction, LocalInteraction
from dgamore.local_four_point import LocalFourPoint
from dgamore.n_point_base import SpinChannel
from tests import conftest

# captured at import time, before the autouse mock_numpy_save fixture patches np.save; the integration test needs
# real file round trips because create_full_vertex_q_r loads its intermediates from disk
_real_np_save = np.save


def _make_single_rank_distributor():
    """Builds a minimal MpiDistributor stand-in on one rank: identity gather/scatter, no-op barrier."""
    dist = MagicMock(comm=SimpleNamespace(rank=0, size=1), my_rank=0, my_slice=slice(None))
    dist.gather.side_effect = lambda mat, root=0: mat
    dist.scatter.side_effect = lambda mat, root=0: mat
    return dist


def _to_mat(x: np.ndarray, nv: int, no: int) -> np.ndarray:
    """Maps a two-frequency four-orbital tensor to its compound matrix (product and inversion become matrix ops)."""
    d = nv * no * no
    return x.transpose(4, 0, 1, 5, 3, 2).reshape(d, d)


def _from_mat(m: np.ndarray, nv: int, no: int) -> np.ndarray:
    """Inverse of :func:`_to_mat`."""
    return m.reshape(nv, no, no, nv, no, no).transpose(1, 2, 5, 4, 0, 3)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(np, "save", _real_np_save)  # restore real saving over the autouse mock

    comm_mock = conftest.create_comm_mock()
    config.logger = DgaLogger(comm_mock, "./")

    config.sys.beta = 7.3
    config.box.niw_core = 2
    config.box.niv_core = 3
    config.output.output_path = str(tmp_path)
    config.output.eliashberg_path = str(tmp_path)
    config.eliashberg.perform_eliashberg = True
    config.eliashberg.save_fq = True  # keep the result in ph notation for the comparison
    config.memory.save_memory_for_chiq_aux = False
    config.lambda_correction.perform_lambda_correction = False
    config.stabilization.use_chi_phys_restriction = False

    yield tmp_path


@pytest.mark.parametrize("no", [2, 3, 4, 5])
def test_create_full_vertex_q_r_matches_exact_bse_inversion(setup, no):
    """The production F^q chain reproduces the exact BSE inversion on reversal-symmetric bubbles with zero shell."""
    rng = np.random.default_rng(0)
    beta = config.sys.beta
    n_q, n_w, niv = 2, 3, 3

    chi0_mat = np.zeros((n_q, no, no, no, no, n_w, 2 * niv), complex)
    for iq in range(n_q):
        for iw in range(n_w):
            for iv in range(2 * niv):
                a = rng.standard_normal((no, no)) + 1j * rng.standard_normal((no, no))
                b = rng.standard_normal((no, no)) + 1j * rng.standard_normal((no, no))
                chi0_mat[iq, ..., iw, iv] = -beta * (
                    np.einsum("il,kj->ijkl", a, b) + np.einsum("il,kj->ijkl", a.T, b.T)
                )

    gchi0_q = FourPoint(chi0_mat, SpinChannel.NONE, (n_q, 1, 1), 1, 1, False, True, True)
    gchi0_q_inv = gchi0_q.invert()
    gchi0_q_inv.save(name="gchi0_q_inv_rank_0", output_dir=config.output.eliashberg_path)

    gamma_mat = 0.3 * (
        rng.standard_normal((no, no, no, no, n_w, 2 * niv, 2 * niv))
        + 1j * rng.standard_normal((no, no, no, no, n_w, 2 * niv, 2 * niv))
    )
    gamma_r = LocalFourPoint(gamma_mat, SpinChannel.DENS, 1, 2, full_niw_range=False).symmetrize_v_vp()

    u_mat = rng.standard_normal((no, no, no, no))
    u_mat = 0.5 * (u_mat + u_mat.transpose(1, 0, 3, 2))
    u_mat = 0.5 * (u_mat + u_mat.transpose(3, 2, 1, 0))
    u_loc = LocalInteraction(0.5 * u_mat)
    v_nonloc = Interaction(np.zeros((n_q, no, no, no, no)), SpinChannel.NONE, (n_q, 1, 1), True)

    u_r = u_loc.as_channel(SpinChannel.DENS).mat
    assert np.allclose(u_r, u_r.transpose(3, 2, 1, 0))

    zero_sum = FourPoint(np.zeros((n_q, no, no, no, no, n_w)), SpinChannel.NONE, (n_q, 1, 1), 1, 0, False, True, True)

    dist = _make_single_rank_distributor()
    nonlocal_sde.calculate_sigma_kernel_r_q(gamma_r, gchi0_q_inv, zero_sum, zero_sum, u_loc, v_nonloc, dist)

    f_q = eliashberg_solver.create_full_vertex_q_r(u_loc, v_nonloc, gamma_r, niv, dist)

    # exact reference per (q, omega) slice from the raw toy tensors: F depends only on chi0, Gamma and beta
    for iq in range(n_q):
        for iw in range(n_w):
            m_chi0 = np.zeros((2 * niv * no * no, 2 * niv * no * no), complex)
            blk = no * no
            for iv in range(2 * niv):
                m_chi0[iv * blk : (iv + 1) * blk, iv * blk : (iv + 1) * blk] = (
                    chi0_mat[iq, ..., iw, iv].transpose(0, 1, 3, 2).reshape(blk, blk)
                )
            m_chi0_inv = np.linalg.inv(m_chi0)
            m_gamma = _to_mat(gamma_r.mat[..., iw, :, :], 2 * niv, no)
            m_chi = np.linalg.inv(m_chi0_inv + m_gamma / beta**2)
            f_exact = _from_mat(beta**2 * (m_chi0_inv - m_chi0_inv @ m_chi @ m_chi0_inv), 2 * niv, no)

            f_code = f_q.mat[iq, :, :, :, :, iw, :, :]
            scale = np.max(np.abs(f_exact))
            assert np.max(np.abs(f_code - f_exact)) / scale < 1e-4, f"F^q mismatch at q={iq}, w={iw}"

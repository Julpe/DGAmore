# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os

import numpy as np

from dgamore import local_sde
from dgamore.hamiltonian import Hamiltonian
from dgamore.n_point_base import SpinChannel

TEST_DATA = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/local_sde"


def test_local_hartree_fock_matches_reference():
    """
    Verifies the local Hartree-Fock (static) self-energy for a genuine multi-orbital interaction. The two-orbital
    La3Ni2O7 ``u_matrix.dat`` carries off-diagonal Kanamori terms (inter-orbital density ``U' = 2.39`` at
    ``U_{abab}``, Hund ``J = 0.61``); the reference ``sigma_HF.npy`` was generated to match the DMFT
    high-frequency self-energy moment of that system. This guards the orbital-index convention of the Hartree
    term (see :func:`dgamore.local_sde.get_local_hartree_fock`): reverting the ``abcd->acbd`` swap makes the
    Hartree pick ``U_{aabb} = J`` instead of ``U'`` and the test fails.
    """
    u_loc = Hamiltonian().read_umatrix(f"{TEST_DATA}/u_matrix.dat").get_local_u()
    occ = np.load(f"{TEST_DATA}/occ.npy", allow_pickle=False)
    sigma_hf_ref = np.load(f"{TEST_DATA}/sigma_HF.npy", allow_pickle=False)

    sigma_hf = local_sde.get_local_hartree_fock(u_loc, occ)

    assert sigma_hf.shape == sigma_hf_ref.shape
    assert np.allclose(sigma_hf, sigma_hf_ref)


def test_local_hartree_fock_uses_inter_orbital_u_prime():
    """
    Guards against the convention bug directly: the inter-orbital Hartree must use ``U' = U_{abab}``, not
    ``J = U_{aabb}``. The pre-fix contraction (without the ``abcd->acbd`` swap) gives a clearly different,
    too-small static self-energy, so it must NOT match the reference.
    """
    u_loc = Hamiltonian().read_umatrix(f"{TEST_DATA}/u_matrix.dat").get_local_u()
    occ = np.load(f"{TEST_DATA}/occ.npy", allow_pickle=False)
    sigma_hf_ref = np.load(f"{TEST_DATA}/sigma_HF.npy", allow_pickle=False)

    sigma_hf_buggy = u_loc.as_channel(SpinChannel.DENS).times("abcd,dc->ab", occ)

    assert not np.allclose(sigma_hf_buggy, sigma_hf_ref)

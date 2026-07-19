# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os

import numpy as np

from dgamore import local_sde
from dgamore.hamiltonian import Hamiltonian
from dgamore.n_point_base import SpinChannel

TEST_DATA = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/local_sde"


def test_local_hartree_fock_matches_reference():
    """The local Hartree-Fock matches the multi-orbital reference; the abcd->acbd swap picks U' not J."""
    u_loc = Hamiltonian().read_umatrix(f"{TEST_DATA}/u_matrix.dat").get_local_u()
    occ = np.load(f"{TEST_DATA}/occ.npy", allow_pickle=False)
    sigma_hf_ref = np.load(f"{TEST_DATA}/sigma_HF.npy", allow_pickle=False)

    sigma_hf = local_sde.get_local_hartree_fock(u_loc, occ)

    assert sigma_hf.shape == sigma_hf_ref.shape
    assert np.allclose(sigma_hf, sigma_hf_ref)


def test_local_hartree_fock_uses_inter_orbital_u_prime():
    """The inter-orbital Hartree must use U'=U_{abab}; the pre-fix contraction (no abcd->acbd swap) must not match."""
    u_loc = Hamiltonian().read_umatrix(f"{TEST_DATA}/u_matrix.dat").get_local_u()
    occ = np.load(f"{TEST_DATA}/occ.npy", allow_pickle=False)
    sigma_hf_ref = np.load(f"{TEST_DATA}/sigma_HF.npy", allow_pickle=False)

    sigma_hf_buggy = u_loc.as_channel(SpinChannel.DENS).times("abcd,dc->ab", occ)

    assert not np.allclose(sigma_hf_buggy, sigma_hf_ref)


def _local_chain_inputs(o=2, nw=3, niv=3, beta=12.5, seed=51):
    """Builds (gchi_r [half niw, 2 vn], gchi0_inv [half niw, 1 vn], u_loc) for the local assembly parity tests."""
    import dgamore.config as config
    from dgamore.interaction import LocalInteraction
    from dgamore.local_four_point import LocalFourPoint

    config.sys.beta = beta
    config.box.niv_core = niv
    rng = np.random.default_rng(seed)
    chi_shape = (o, o, o, o, nw, 2 * niv, 2 * niv)
    gchi_r = LocalFourPoint(
        rng.standard_normal(chi_shape) + 1j * rng.standard_normal(chi_shape), SpinChannel.DENS, 1, 2, False, True
    )
    inv_shape = (o, o, o, o, nw, 2 * niv)
    gchi0_inv = LocalFourPoint(
        rng.standard_normal(inv_shape) + 1j * rng.standard_normal(inv_shape), SpinChannel.NONE, 1, 1, False, True
    )
    u_loc = LocalInteraction(rng.standard_normal((o,) * 4), SpinChannel.NONE)
    return gchi_r, gchi0_inv, u_loc


def test_create_auxiliary_chi_matches_two_block_expression():
    """The fused local auxiliary susceptibility matches (chi0^-1 + (Gamma-U)/beta^2)^-1 and leaves gamma untouched."""
    import dgamore.config as config

    gamma_r, gchi0_inv, u_loc = _local_chain_inputs()
    beta = config.sys.beta
    gamma_before = gamma_r.mat.copy()
    ref = (gchi0_inv + (gamma_r.sub(u_loc.as_channel(gamma_r.channel))).scale(1.0 / beta**2)).invert()
    out = local_sde.create_auxiliary_chi(gamma_r, gchi0_inv, u_loc)
    assert np.allclose(out.mat, ref.mat, atol=1e-8)
    assert np.array_equal(gamma_r.mat, gamma_before)


def test_create_gamma_r_with_shell_correction_matches_two_block_expression():
    """The fused shell-corrected irreducible vertex matches the former subtract/scale/add chain, chi untouched."""
    import dgamore.config as config

    gchi_r, _, u_loc = _local_chain_inputs(seed=52)
    from dgamore.bubble_gen import BubbleGenerator
    from dgamore.greens_function import GreensFunction

    rng = np.random.default_rng(53)
    g_mat = rng.standard_normal((2, 2, 24)) + 1j * rng.standard_normal((2, 2, 24))
    gchi0 = BubbleGenerator.create_generalized_chi0(GreensFunction(g_mat), 2, 6, config.sys.beta)
    gchi_before = gchi_r.mat.copy()

    beta = config.sys.beta
    # isolate the outer fused subtract/scale/add: the inner shell inverse uses the same Woodbury path as production
    # (its equivalence to the dense chain is locked separately by test_shell_inverse_core_matches_dense_chain).
    chi_tilde_core_inv = gchi0.shell_inverse_core(u_loc.as_channel(gchi_r.channel), beta, config.box.niv_core).invert()
    ref = (gchi_r.invert() - chi_tilde_core_inv).scale(beta**2) + u_loc.as_channel(gchi_r.channel)

    out = local_sde.create_gamma_r_with_shell_correction(gchi_r, gchi0, u_loc)
    assert np.allclose(out.mat, ref.mat, atol=1e-6)
    assert np.array_equal(gchi_r.mat, gchi_before)


def test_create_full_vertex_from_gamma_matches_batched_expression():
    """The per-w full-vertex solve matches the former batched identity/matmul/invert expression, gamma untouched."""
    import dgamore.config as config
    from dgamore.bubble_gen import BubbleGenerator
    from dgamore.greens_function import GreensFunction
    from dgamore.interaction import LocalInteraction
    from dgamore.local_four_point import LocalFourPoint

    o, nw, niv_core, niv_full, beta = 2, 3, 2, 3, 12.5
    config.sys.beta = beta
    config.box.niv_core = niv_core
    config.box.niv_full = niv_full
    rng = np.random.default_rng(71)
    gamma_shape = (o, o, o, o, nw, 2 * niv_core, 2 * niv_core)
    gamma_r = LocalFourPoint(
        rng.standard_normal(gamma_shape) + 1j * rng.standard_normal(gamma_shape), SpinChannel.DENS, 1, 2, False, True
    )
    g_mat = rng.standard_normal((o, o, 2 * (niv_full + nw))) + 1j * rng.standard_normal((o, o, 2 * (niv_full + nw)))
    gchi0 = BubbleGenerator.create_generalized_chi0(GreensFunction(g_mat), nw - 1, niv_full, beta).take_vn_diagonal()
    u_loc = LocalInteraction(rng.standard_normal((o,) * 4), SpinChannel.NONE)
    gamma_before = gamma_r.mat.copy()

    gamma_urange = gamma_r.pad_with_u(u_loc.as_channel(gamma_r.channel), niv_full)
    ref = gamma_urange @ (LocalFourPoint.identity_like(gamma_urange) + 1.0 / beta**2 * gchi0 @ gamma_urange).invert(
        False
    )

    out = local_sde.create_full_vertex_from_gamma(gamma_r, gchi0, u_loc)
    assert out.mat.shape == ref.mat.shape
    assert np.allclose(out.mat, ref.mat, atol=5e-3)
    assert np.array_equal(gamma_r.mat, gamma_before)

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os

import numpy as np
import pytest

import dgamore.config as config
from dgamore import local_sde
from dgamore.hamiltonian import Hamiltonian
from dgamore.n_point_base import FrequencyNotation, SpinChannel

TEST_DATA = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/local_sde"


def test_local_hartree_fock_matches_reference():
    """The local Hartree-Fock matches the multi-orbital reference (U' stored on aabb, no slot swap needed)."""
    u_loc = Hamiltonian().read_umatrix(f"{TEST_DATA}/u_matrix.dat").get_local_u()
    occ = np.load(f"{TEST_DATA}/occ.npy", allow_pickle=False)
    sigma_hf_ref = np.load(f"{TEST_DATA}/sigma_HF.npy", allow_pickle=False)

    sigma_hf = local_sde.get_local_hartree_fock(u_loc, occ)

    assert sigma_hf.shape == sigma_hf_ref.shape
    assert np.allclose(sigma_hf, sigma_hf_ref)


def test_local_hartree_fock_needs_no_slot_swap_in_the_equation_layout():
    """With U' stored on aabb the plain density projection reproduces the reference; a middle swap would not."""
    u_loc = Hamiltonian().read_umatrix(f"{TEST_DATA}/u_matrix.dat").get_local_u()
    occ = np.load(f"{TEST_DATA}/occ.npy", allow_pickle=False)
    sigma_hf_ref = np.load(f"{TEST_DATA}/sigma_HF.npy", allow_pickle=False)

    plain = u_loc.as_channel(SpinChannel.DENS).times("abcd,dc->ab", occ)
    swapped = u_loc.permute_orbitals("abcd->acbd").as_channel(SpinChannel.DENS).times("abcd,dc->ab", occ)

    assert np.allclose(plain, sigma_hf_ref)
    assert not np.allclose(swapped, sigma_hf_ref)


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


@pytest.mark.parametrize("builder", ["vrg", "gamma"])
def test_local_self_energies_carry_a_single_momentum_dimension(builder):
    """The locally computed self-energies descend from TwoPoint, so they carry ``[1, 1, 1]`` momentum axes."""
    import dgamore.config as config
    from dgamore.greens_function import GreensFunction
    from dgamore.interaction import LocalInteraction
    from dgamore.local_four_point import LocalFourPoint

    o, niw, niv = 2, 3, 4
    config.sys.beta = 10.0
    config.box.niw_core, config.box.niv_core = niw, niv
    config.sys.occ_dmft = np.eye(o) * 0.5
    rng = np.random.default_rng(17)

    def four_point(num_vn):
        shape = (o, o, o, o, niw + 1) + (2 * niv,) * num_vn
        return LocalFourPoint(
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape), SpinChannel.DENS, 1, num_vn, False, True
        )

    g_mat = rng.standard_normal((o, o, 4 * (niw + niv))) + 1j * rng.standard_normal((o, o, 4 * (niw + niv)))
    g_dmft = GreensFunction(g_mat[None, None, None, ...], beta=config.sys.beta)
    u_loc = LocalInteraction(rng.standard_normal((o,) * 4), SpinChannel.NONE)

    if builder == "vrg":
        sigma = local_sde.get_loc_self_energy_vrg(four_point(1), four_point(0), g_dmft, u_loc)
    else:
        sigma = local_sde.get_loc_self_energy_gamma_abinitio_dga(four_point(1), u_loc, g_dmft)

    assert sigma.current_shape[:3] == (1, 1, 1)
    assert sigma.current_shape[3:5] == (o, o)


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
    gchi0 = BubbleGenerator.create_generalized_chi0(GreensFunction(g_mat[None, None, None, ...]), 2, 6, config.sys.beta)
    gchi_before = gchi_r.mat.copy()

    beta = config.sys.beta
    # isolates the outer fused subtract/scale/add; the inner shell inverse is locked by its own parity test
    chi_tilde_core_inv = gchi0.get_core_from_shell_inversion(
        1.0 / beta**2 * u_loc.as_channel(gchi_r.channel), config.box.niv_core
    ).invert()
    ref = (gchi_r.invert() - chi_tilde_core_inv).scale(beta**2) + u_loc.as_channel(gchi_r.channel)

    out = local_sde.create_gamma_r_with_shell_correction(gchi_r, gchi0, u_loc)
    assert np.allclose(out.mat, ref.mat, atol=1e-6)
    assert np.array_equal(gchi_r.mat, gchi_before)


def test_create_full_vertex_from_gamma_matches_batched_expression():
    """The full-vertex build matches the batched expression on the core nu' window it now returns, gamma untouched."""
    import dgamore.config as config
    from dgamore.bubble_gen import BubbleGenerator
    from dgamore.greens_function import GreensFunction
    from dgamore.interaction import LocalInteraction
    from dgamore.local_four_point import LocalFourPoint

    o, nw, niv_core, niv_full, beta = 2, 3, 6, 18, 12.5
    config.sys.beta = beta
    config.box.niv_core = niv_core
    config.box.niv_full = niv_full
    rng = np.random.default_rng(71)
    gamma_shape = (o, o, o, o, nw, 2 * niv_core, 2 * niv_core)
    gamma_r = LocalFourPoint(
        rng.standard_normal(gamma_shape) + 1j * rng.standard_normal(gamma_shape), SpinChannel.DENS, 1, 2, False, True
    )
    g_mat = rng.standard_normal((o, o, 2 * (niv_full + nw))) + 1j * rng.standard_normal((o, o, 2 * (niv_full + nw)))
    gchi0 = BubbleGenerator.create_generalized_chi0(
        GreensFunction(g_mat[None, None, None, ...]), nw - 1, niv_full, beta
    ).take_vn_diagonal()
    u_loc = LocalInteraction(rng.standard_normal((o,) * 4), SpinChannel.NONE)
    gamma_before = gamma_r.mat.copy()

    gamma_urange = gamma_r.pad_with_u(u_loc.as_channel(gamma_r.channel), niv_full)
    ref = gamma_urange @ (LocalFourPoint.identity_like(gamma_urange) + 1.0 / beta**2 * gchi0 @ gamma_urange).invert(
        False
    )

    out = local_sde.create_full_vertex_from_gamma(gamma_r, gchi0, u_loc)
    window = slice(niv_full - niv_core, niv_full + niv_core)
    assert out.niv == niv_core and out.niv_first == niv_full
    assert np.allclose(out.mat, ref.mat[..., window], atol=5e-3)
    assert np.array_equal(gamma_r.mat, gamma_before)


def _full_vertex_inputs(o, niw, niv_core, niv_full, beta, full_niw, seed=11):
    """Builds (gamma, chi_0, U) for the local full-vertex tests and points the config box at that box."""
    import dgamore.config as config
    from dgamore.interaction import LocalInteraction
    from dgamore.local_four_point import LocalFourPoint

    config.sys.beta = beta
    config.box.niv_core, config.box.niv_full = niv_core, niv_full
    rng = np.random.default_rng(seed)
    gamma_shape = (o, o, o, o, niw + 1, 2 * niv_core, 2 * niv_core)
    gamma = (rng.standard_normal(gamma_shape) + 1j * rng.standard_normal(gamma_shape)).astype(np.complex64)
    gamma_r = LocalFourPoint(gamma, SpinChannel.DENS, 1, 2, False, True)
    chi0_shape = (o, o, o, o, 2 * niw + 1 if full_niw else niw + 1, 2 * niv_full)
    chi0 = (rng.standard_normal(chi0_shape) + 1j * rng.standard_normal(chi0_shape)).astype(np.complex64) * 0.1
    for a in range(o):
        for b in range(o):
            chi0[a, b, b, a] += 3.0  # compound-diagonal boost keeps the bubble blocks invertible
    gchi0 = LocalFourPoint(chi0, SpinChannel.NONE, num_vn_dimensions=1, full_niw_range=full_niw)
    return gamma_r, gchi0, LocalInteraction(rng.standard_normal((o,) * 4).astype(np.complex64), SpinChannel.NONE)


def _dense_padded_full_vertex(gamma_r, gchi0, u_channel, beta, niv_full):
    """Dense per-omega reference F = Gamma_U (1 + chi_0 Gamma_U / beta^2)^-1 with Gamma_U padded explicitly."""
    o, niv_core = gamma_r.n_bands, gamma_r.niv
    o2, off, w_dim = o * o, niv_full - gamma_r.niv, gamma_r.current_shape[-3]
    chi0 = gchi0.mat[..., gchi0.mat.shape[-2] // 2 :, :] if gchi0.full_niw_range else gchi0.mat
    u_c = u_channel.mat.transpose(0, 1, 3, 2).reshape(o2, o2)
    n = o2 * 2 * niv_full
    out = np.empty((o, o, o, o, w_dim, 2 * niv_full, 2 * niv_full), dtype=np.complex64)
    for iw in range(w_dim):
        padded = np.empty((o2, 2 * niv_full, o2, 2 * niv_full), dtype=np.complex64)
        padded[...] = u_c[:, None, :, None]
        padded[:, off : off + 2 * niv_core, :, off : off + 2 * niv_core] = (
            gamma_r.mat[..., iw, :, :].transpose(0, 1, 4, 3, 2, 5).reshape(o2, 2 * niv_core, o2, 2 * niv_core)
        )
        c = chi0[..., iw, :].transpose(4, 0, 1, 3, 2).reshape(2 * niv_full, o2, o2) / beta**2
        a = np.einsum("vij,jvkl->ivkl", c, padded, optimize=True).reshape(n, n)
        a[np.arange(n), np.arange(n)] += 1.0
        f = np.linalg.solve(a.T, padded.reshape(n, n).T).T
        out[..., iw, :, :] = f.reshape(o, o, 2 * niv_full, o, o, 2 * niv_full).transpose(0, 1, 4, 3, 2, 5)
    return out


@pytest.mark.parametrize("o", [1, 2, 3])
@pytest.mark.parametrize("full_niw", [True, False])
@pytest.mark.parametrize("niv_core, niv_full", [(6, 18), (8, 24), (6, 12), (8, 8)])
def test_create_full_vertex_from_gamma_matches_dense_padded_solve(o, full_niw, niv_core, niv_full):
    """The full-vertex build equals the nu' window of the dense padded solve across bands, niw ranges and shells."""
    niw, beta = 2, 3.0
    gamma_r, gchi0, u_loc = _full_vertex_inputs(o, niw, niv_core, niv_full, beta, full_niw)
    gamma_before = gamma_r.mat.copy()

    ref = _dense_padded_full_vertex(gamma_r, gchi0, u_loc.as_channel(gamma_r.channel), beta, niv_full)
    out = local_sde.create_full_vertex_from_gamma(gamma_r, gchi0, u_loc)

    window = slice(niv_full - niv_core, niv_full + niv_core)
    assert out.niv_first == niv_full and out.niv == niv_core and out.channel == SpinChannel.DENS
    assert np.allclose(out.mat, ref[..., window], atol=1e-3)
    assert np.array_equal(gamma_r.mat, gamma_before)


def test_create_full_vertex_from_gamma_rejects_a_full_box_narrower_than_the_core_box():
    """A full fermionic box narrower than the irreducible vertex's own core box is rejected."""
    import dgamore.config as config

    gamma_r, gchi0, u_loc = _full_vertex_inputs(2, 1, 8, 8, 3.0, False)
    config.box.niv_full = 6

    with pytest.raises(ValueError):
        local_sde.create_full_vertex_from_gamma(gamma_r, gchi0, u_loc)


def _random_local_sde_inputs(o, niw=3, niv=4, beta=10.0, seed=17):
    """Builds random (vrg_dens, gchi_dens_sum, g_dmft, u_loc) for the local SDE tests."""
    import dgamore.config as config
    from dgamore.greens_function import GreensFunction
    from dgamore.interaction import LocalInteraction
    from dgamore.local_four_point import LocalFourPoint

    config.sys.beta = beta
    config.box.niw_core, config.box.niv_core = niw, niv
    config.sys.occ_dmft = np.zeros((o, o))
    rng = np.random.default_rng(seed)

    def four_point(channel, num_vn):
        shape = (o, o, o, o, niw + 1) + (2 * niv,) * num_vn
        mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        return LocalFourPoint(mat, channel, 1, num_vn, False, True)

    g_shape = (1, 1, 1, o, o, 2 * (niw + niv))
    g_dmft = GreensFunction(rng.standard_normal(g_shape) + 1j * rng.standard_normal(g_shape), beta=beta)
    u_mat = rng.standard_normal((o,) * 4)
    u_loc = LocalInteraction(0.5 * (u_mat + u_mat.transpose(2, 3, 0, 1)))
    return four_point(SpinChannel.DENS, 1), four_point(SpinChannel.DENS, 0), g_dmft, u_loc


def _eta(vrg, u_r, chi_sum):
    """Returns gamma - gamma U_r chi_r in the standard compound product, (AB)_{1234} = sum_{ab} A_{12ab} B_{ba34}."""
    vrg_u = np.einsum("abyxwv,xyef->abefwv", vrg.mat, u_r, optimize=True)
    return vrg.mat - np.einsum("abyxwv,xyefw->abefwv", vrg_u, chi_sum.mat, optimize=True)


def _local_sde_reference(inner_half, g_dmft, u_loc):
    """Contracts a half-range inner [o, o, o, o, w, v] into sigma_ij = -1/beta sum U_kjop inner_ilpo G_lk(v - w)."""
    import dgamore.config as config
    from dgamore.local_four_point import LocalFourPoint
    from dgamore.matsubara_frequencies import MFHelper

    inner = LocalFourPoint(inner_half, SpinChannel.NONE, 1, 1, False, True).to_full_niw_range().mat
    g_wv = g_dmft.get_g_wv(MFHelper.wn(config.box.niw_core), config.box.niv_core)
    return -1.0 / config.sys.beta * np.einsum("kjop,ilpowv,lkwv->ijv", u_loc.mat, inner, g_wv, optimize=True)


def _bare_local_sde_inputs(o, u_mat, niw=3, niv=4, beta=5.0, eps=0.3):
    """Builds second-order local SDE inputs (bare three-leg vertex and bubble sum, diagonal G, U) for o bands."""
    import dgamore.config as config
    from dgamore.bubble_gen import BubbleGenerator
    from dgamore.greens_function import GreensFunction
    from dgamore.interaction import LocalInteraction
    from dgamore.local_four_point import LocalFourPoint

    config.sys.beta = beta
    config.box.niw_core, config.box.niv_core = niw, niv
    config.sys.occ_dmft = np.zeros((o, o))
    niv_g = niw + niv
    vn = np.pi / beta * (2 * np.arange(-niv_g, niv_g) + 1)
    g_mat = np.zeros((1, 1, 1, o, o, 2 * niv_g), dtype=complex)
    g_mat[0, 0, 0, np.arange(o), np.arange(o)] = 1.0 / (1j * vn - eps)
    g_dmft = GreensFunction(g_mat, beta=beta)
    chi_sum = BubbleGenerator.create_generalized_chi0(g_dmft, niw, niv, beta).sum_over_all_vn(beta).scale(1.0 / beta)
    unit = np.zeros((o,) * 4 + (niw + 1, 2 * niv), dtype=np.complex64)
    orb = np.arange(o)
    unit[orb[:, None], orb[None, :], orb[None, :], orb[:, None]] = 1.0
    return LocalFourPoint(unit, SpinChannel.DENS, 1, 1, False, True), chi_sum, g_dmft, LocalInteraction(u_mat)


@pytest.mark.parametrize("o", [1, 2])
def test_local_self_energy_is_the_density_form_with_the_direct_attachment(o):
    """The local SDE is -1/beta U_{a2bc} (eta_d - 1)_{1dcb} G_{da} with eta_d = gamma_d (1 - U_d chi_d)."""
    vrg_d, chi_d, g_dmft, u_loc = _random_local_sde_inputs(o)
    u = u_loc.mat
    orb = np.arange(o)
    unit = np.zeros((o,) * 4)
    unit[orb[:, None], orb[None, :], orb[None, :], orb[:, None]] = 1.0
    inner = _eta(vrg_d, 2 * u - u.transpose(0, 3, 2, 1), chi_d) - unit[..., None, None]
    ref = _local_sde_reference(inner, g_dmft, u_loc)

    sigma = local_sde.get_loc_self_energy_vrg(vrg_d, chi_d, g_dmft, u_loc)

    assert np.allclose(sigma.mat[0, 0, 0], ref, atol=1e-4)


def test_double_counting_vertex_is_the_transversal_local_part():
    """f_dc = 1/2 (F_d + 3 F_m) with the magnetic metadata, built from a copy: both inputs stay untouched."""
    from dgamore.local_four_point import LocalFourPoint

    rng = np.random.default_rng(5)
    shape = (1, 1, 1, 1, 3, 8, 6)
    f_d = LocalFourPoint(rng.standard_normal(shape).astype(np.complex64), SpinChannel.DENS, 1, 2, False, True)
    f_m = LocalFourPoint(
        rng.standard_normal(shape).astype(np.complex64), SpinChannel.MAGN, 1, 2, False, True, FrequencyNotation.PH_BAR
    )
    f_d_mat, f_m_mat = f_d.mat.copy(), f_m.mat.copy()

    f_dc = local_sde.double_counting_vertex(f_d, f_m)
    assert f_dc.channel == SpinChannel.MAGN
    assert f_dc.frequency_notation == FrequencyNotation.PH_BAR
    assert f_dc.mat.shape == shape
    assert not f_dc.full_niw_range and f_dc.num_wn_dimensions == 1 and f_dc.num_vn_dimensions == 2
    assert np.allclose(f_dc.mat, 0.5 * (f_d_mat + 3.0 * f_m_mat), atol=1e-6)
    assert np.array_equal(f_d.mat, f_d_mat) and np.array_equal(f_m.mat, f_m_mat)


def test_second_order_self_energy_from_the_kanamori_builder_weighs_u_prime_twice():
    """Density form: both spin loops of the other orbital enter, U^2 + 2 U'^2, only with U' stored on aabb."""
    o, u, u_prime = 2, 2.0, 1.3
    with_u_prime = Hamiltonian().kanamori_interaction_d(n_bands=o, udd=u, jdd=0.0, vdd=u_prime).get_local_u().mat.real
    without = Hamiltonian().kanamori_interaction_d(n_bands=o, udd=u, jdd=0.0, vdd=0.0).get_local_u().mat.real

    sigma_with = local_sde.get_loc_self_energy_vrg(*_bare_local_sde_inputs(o, with_u_prime))
    sigma_without = local_sde.get_loc_self_energy_vrg(*_bare_local_sde_inputs(o, without))

    ratio = sigma_with.mat[0, 0, 0, 0, 0] / sigma_without.mat[0, 0, 0, 0, 0]
    assert np.allclose(ratio, (u**2 + 2 * u_prime**2) / u**2, atol=1e-4)

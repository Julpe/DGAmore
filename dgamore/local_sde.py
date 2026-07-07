# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Local Schwinger-Dyson step. Given the two-particle DMFT Green's functions and the bare interaction, the functions
here build the local vertex hierarchy per spin channel - the generalized susceptibility :math:`\chi_{r}`, the
irreducible vertex :math:`\Gamma_{r}` (with the Kitatani shell asymptotics), the auxiliary susceptibility
:math:`\chi^{*}_{r}`, the three-leg vertex :math:`\gamma_{r}` (``vrg``), the full vertex :math:`F_{r}`, and the
physical susceptibility - and recompute the local self-energy via the Schwinger-Dyson equation as a sanity check
against the DMFT input. Equation numbers refer to the author's master's thesis (Chapter 3). A second set of
functions implements the alternative ab-initio DGA formulation.
"""

import numpy as np

import dgamore.config as config
from dgamore.bubble_gen import BubbleGenerator
from dgamore.greens_function import GreensFunction
from dgamore.interaction import LocalInteraction
from dgamore.local_four_point import LocalFourPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore.n_point_base import SpinChannel, FrequencyNotation
from dgamore.self_energy import SelfEnergy


def create_generalized_chi(g2: LocalFourPoint, g_dmft: GreensFunction) -> LocalFourPoint:
    r"""
    Returns the generalized susceptibility, see also Eq. (3.41) in my master's thesis,
    :math:`\chi_{r;1234}^{\omega\nu\nu'} = \beta (G_{r;1234}^{(2);\omega\nu\nu'} - 2 \delta_{r,\mathrm{dens}}
    \delta_{\omega 0} G_{12}^{\nu} G_{34}^{\nu'})`. The disconnected term is subtracted only in the density
    (ph) channel at :math:`\omega = 0`.

    :param g2: Two-particle (DMFT) Green's function :math:`G^{(2)}_{r}` as a :class:`LocalFourPoint`.
    :param g_dmft: The local (DMFT) :class:`GreensFunction`.
    :return: The generalized susceptibility :math:`\chi_{r}` as a :class:`LocalFourPoint` (half niw range).
    """
    chi = config.sys.beta * g2.to_half_niw_range()

    if g2.channel == SpinChannel.DENS and g2.frequency_notation == FrequencyNotation.PH:
        g_loc_slice_mat = g_dmft.mat[..., g_dmft.niv - config.box.niv_core : g_dmft.niv + config.box.niv_core]
        ggv_mat = g_loc_slice_mat[:, :, None, None, :, None] * g_loc_slice_mat[None, None, :, :, None, :]
        chi[:, :, :, :, 0, ...] -= 2.0 * config.sys.beta * ggv_mat

    return chi


def create_gamma_r(gchi_r: LocalFourPoint, gchi0_inv: LocalFourPoint, beta: float) -> LocalFourPoint:
    r"""
    Returns the ph-irreducible vertex
    :math:`\Gamma_{r;1234}^{\omega\nu\nu'} = \beta^2 [(\chi_{r;1234}^{\omega\nu\nu'})^{-1} -
    (\delta_{\nu\nu'}\chi_{0;1234}^{\omega\nu})^{-1}]`.

    :param gchi_r: The generalized susceptibility :math:`\chi_{r}`.
    :param gchi0_inv: The inverse bare bubble :math:`\chi_0^{-1}` (diagonal in :math:`\nu\nu'`).
    :param beta: Inverse temperature :math:`\beta`.
    :return: The irreducible vertex :math:`\Gamma_{r}` as a :class:`LocalFourPoint`.
    """
    return (gchi_r.invert() - gchi0_inv).scale(beta**2)


def create_gamma_r_with_shell_correction(
    gchi_r: LocalFourPoint, gchi0: LocalFourPoint, u_loc: LocalInteraction
) -> LocalFourPoint:
    r"""
    Calculates the irreducible vertex with the shell correction as described by Motoharu Kitatani
    et al. 2022 J. Phys. Mater. 5 034005; DOI 10.1088/2515-7639/ac7e6d. More specifically, see equations A.4 and A.8.
    The irreducible vertex has an additional factor of :math:`1/\beta^2` compared to DGApy. This is also described in
    my master's thesis, Sec. 3.7.2.

    :param gchi_r: The generalized susceptibility :math:`\chi_{r}` (core frequency box).
    :param gchi0: The bare bubble :math:`\chi_0` over the full frequency box.
    :param u_loc: The bare local interaction :math:`U`.
    :return: The shell-corrected irreducible vertex :math:`\Gamma_{r}` as a :class:`LocalFourPoint`.
    """
    # the +U below must couple ALL fermionic frequencies (the Kitatani shell ladder with the constant-U vertex),
    # so the block-diagonally inverted bubble is explicitly extended to the two-fermion layout first - keeping it
    # 1-vn would put U on the nu-diagonal only (wrong physics). The first inversion is still per (w, v) block.
    chi_tilde_shell = (
        gchi0.invert().extend_vn_to_diagonal() + 1.0 / config.sys.beta**2 * u_loc.as_channel(gchi_r.channel)
    ).invert()
    chi_tilde_core_inv = chi_tilde_shell.cut_niv(config.box.niv_core).invert()
    # subtract/scale/accumulate in place on the freshly inverted block, so only that one two-fermion block is
    # allocated (the former chain held up to three such blocks at its peak)
    return (
        gchi_r.invert()
        .sub(chi_tilde_core_inv, copy=False)
        .scale(config.sys.beta**2)
        .add(u_loc.as_channel(gchi_r.channel), copy=False)
    )


def create_auxiliary_chi(gamma_r: LocalFourPoint, gchi0_inv: LocalFourPoint, u_loc: LocalInteraction) -> LocalFourPoint:
    r"""
    Returns the auxiliary susceptibility, see Eq. (3.60) in my master's thesis,
    :math:`\chi^{*;\omega\nu\nu'}_{r;1234} = ((\chi_{0;1234}^{\omega\nu})^{-1} +
    (\Gamma_{r;1234}^{\omega\nu\nu'} - U_{r;1234})/\beta^2)^{-1}`.

    :param gamma_r: The irreducible vertex :math:`\Gamma_{r}`.
    :param gchi0_inv: The inverse bare bubble :math:`\chi_0^{-1}` (core box).
    :param u_loc: The bare local interaction :math:`U`.
    :return: The auxiliary susceptibility :math:`\chi^{*}_{r}` as a :class:`LocalFourPoint`.
    """
    # single-block assembly: (gamma - U) is the only allocation, the diagonal bubble adds in place and the
    # inversion consumes the block (the former expression held the extended bubble and the sum alongside)
    return (
        gamma_r.sub(u_loc.as_channel(gamma_r.channel))
        .scale(1.0 / config.sys.beta**2)
        .add_on_vn_diagonal(gchi0_inv)
        .invert(copy=False)
    )


def create_generalized_chi_with_shell_correction(
    gchi_aux_sum: LocalFourPoint, gchi0: LocalFourPoint, u_loc: LocalInteraction
) -> LocalFourPoint:
    r"""
    Calculates the generalized susceptibility with the shell correction as described by
    Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005; DOI 10.1088/2515-7639/ac7e6d. Eq. A.15. This is also
    described in my master's thesis, Sec. 3.7.2.

    :param gchi_aux_sum: The frequency-summed auxiliary susceptibility :math:`\sum_{\nu\nu'} \chi^{*}_{r}`.
    :param gchi0: The bare bubble :math:`\chi_0` over the full frequency box.
    :param u_loc: The bare local interaction :math:`U`.
    :return: The shell-corrected physical susceptibility :math:`\chi_{r}^{\omega}` as a :class:`LocalFourPoint`.
    """
    gchi0_full_sum = gchi0.sum_over_all_vn(config.sys.beta).scale(1.0 / config.sys.beta)
    gchi0_core_sum = gchi0.cut_niv(config.box.niv_core).sum_over_all_vn(config.sys.beta).scale(1.0 / config.sys.beta)
    return ((gchi_aux_sum + gchi0_full_sum - gchi0_core_sum).invert() + u_loc.as_channel(gchi_aux_sum.channel)).invert()


def create_full_vertex_from_gamma(gamma_r, gchi0, u_loc):
    r"""
    Returns the local full vertex in the ``niv_full`` region from the irreducible vertex,
    :math:`F = \Gamma [1 + \frac{1}{\beta^2} \chi_0 \Gamma]^{-1}` (with :math:`\Gamma` padded with :math:`U` beyond the core box).

    Every bosonic slice is independent, so the solve runs one :math:`\omega` at a time in compound space and writes
    the result back into the padded vertex: only compound ``[x1, x2]``-sized workspaces accompany the single
    ``niv_full``-sized block, where the former batched identity/matmul/invert chain held ~4 such blocks (plus the
    fully materialized compound identity) at its peak. The linear system is solved directly
    (``F^T = (A^T)^{-1} \Gamma^T``) instead of inverting and multiplying.

    :param gamma_r: The irreducible vertex :math:`\Gamma_{r}` (core box).
    :param gchi0: The bare bubble :math:`\chi_0` (with its fermionic axis taken on the diagonal).
    :param u_loc: The bare local interaction :math:`U`, used to pad the shell.
    :return: The full vertex :math:`F_{r}` as a :class:`LocalFourPoint`.
    """
    gamma_urange = gamma_r.pad_with_u(u_loc.as_channel(gamma_r.channel), config.box.niv_full)

    n = gamma_urange.n_bands
    niv2 = gamma_urange.current_shape[-1]
    size = n * n * niv2
    w_dim = gamma_urange.current_shape[-3]
    diag = np.arange(size)
    # the bubble arrives in the full bosonic range while gamma is half-range: align via a half-range view (the
    # w >= 0 block), exactly what the former matmul's internal to_half_niw_range did
    chi0_mat = gchi0.mat
    if gchi0.full_niw_range:
        chi0_mat = chi0_mat[..., chi0_mat.shape[-2] // 2 :, :]
    chi0_scaled = chi0_mat * (1.0 / config.sys.beta**2)

    path = None
    for iw in range(w_dim):
        gamma_w = gamma_urange.mat[..., iw, :, :]
        if path is None:  # identical shapes across the loop -> compute the contraction path once
            path = np.einsum_path("abcdv,dcefvp->abefvp", chi0_scaled[..., 0, :], gamma_w, optimize="optimal")[0]
        a_w = np.einsum("abcdv,dcefvp->abefvp", chi0_scaled[..., iw, :], gamma_w, optimize=path)
        # compound pairing (rows {1, 2, v}, cols {4, 3, v'}, as in to_compound_indices) + identity on the diagonal
        a_w = a_w.transpose(0, 1, 4, 3, 2, 5).reshape(size, size)
        a_w[diag, diag] += 1.0
        gamma_w = gamma_w.transpose(0, 1, 4, 3, 2, 5).reshape(size, size)
        f_w = np.linalg.solve(a_w.T, gamma_w.T).T
        gamma_urange.mat[..., iw, :, :] = f_w.reshape(n, n, niv2, n, n, niv2).transpose(0, 1, 4, 3, 2, 5)

    return gamma_urange


def create_full_vertex(gchi_r: LocalFourPoint, gchi0_inv: LocalFourPoint) -> LocalFourPoint:
    r"""
    Returns the local full vertex in the ``niv_core`` region, see Eq. (3.58) in my master's thesis,
    :math:`F_{r;1234}^{\omega\nu\nu'} = \beta^2 ((\chi_{0;1234}^{\omega\nu\nu'})^{-1} - \sum_{abcd} (\chi_{0;12ab}^{\omega\nu})^{-1}
    \chi_{r;bacd}^{\omega\nu\nu'} (\chi_{0;dc34}^{\omega\nu'})^{-1})`.

    :param gchi_r: The generalized susceptibility :math:`\chi_{r}`.
    :param gchi0_inv: The inverse bare bubble :math:`\chi_0^{-1}`.
    :return: The full vertex :math:`F_{r}` as a :class:`LocalFourPoint`.
    """
    return (gchi0_inv - gchi0_inv @ gchi_r @ gchi0_inv).scale(config.sys.beta**2)


def create_vrg(gchi_aux: LocalFourPoint, gchi0_inv: LocalFourPoint) -> LocalFourPoint:
    r"""
    Returns the three-leg vertex, see Eq. (3.63) in my master's thesis,
    :math:`\gamma_{r;1234}^{\omega\nu} = \beta \sum_{ab} \sum_{\nu'} (\chi^{\omega\nu}_{0;12ab})^{-1}
    \chi^{*;\omega\nu\nu'}_{r;ba34}`.

    :param gchi_aux: The auxiliary susceptibility :math:`\chi^{*}_{r}`.
    :param gchi0_inv: The inverse bare bubble :math:`\chi_0^{-1}` (core box).
    :return: The three-leg vertex :math:`\gamma_{r}` (``vrg``) as a :class:`LocalFourPoint`.
    """
    gchi_aux_sum = gchi_aux.sum_over_vn(config.sys.beta, axis=(-1,))
    return (gchi0_inv @ gchi_aux_sum).scale(config.sys.beta)


def create_vertex_functions(
    g2_r: LocalFourPoint,
    gchi0: LocalFourPoint,
    gchi0_inv_core: LocalFourPoint,
    g_dmft: GreensFunction,
    u_loc: LocalInteraction,
) -> tuple[LocalFourPoint, LocalFourPoint, LocalFourPoint, LocalFourPoint, LocalFourPoint]:
    r"""
    Builds the full local vertex hierarchy for a single spin channel: the irreducible vertex :math:`\Gamma_{r}`
    (with shell correction), the frequency-summed (shell-corrected) physical susceptibility, the three-leg vertex
    :math:`\gamma_{r}`, the full vertex :math:`F_{r}`, and the generalized susceptibility :math:`\chi_{r}`. Employs
    explicit asymptotics as proposed by Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005;
    DOI 10.1088/2515-7639/ac7e6d for the local irreducible vertex.

    :param g2_r: The two-particle (DMFT) Green's function :math:`G^{(2)}_{r}` for this channel.
    :param gchi0: The bare bubble :math:`\chi_0` over the full frequency box.
    :param gchi0_inv_core: The inverse bare bubble :math:`\chi_0^{-1}` over the core box (diagonal in :math:`\nu`).
    :param g_dmft: The local (DMFT) :class:`GreensFunction`.
    :param u_loc: The bare local interaction :math:`U`.
    :return: The tuple ``(gamma_r, gchi_r_sum, vrg_r, f_r, gchi_r)`` of local vertex functions.
    """
    logger = config.logger

    gchi_r = create_generalized_chi(g2_r, g_dmft)
    logger.info(f"Local generalized susceptibility chi^wvv' ({gchi_r.channel.value}) calculated.")

    gamma_r = create_gamma_r_with_shell_correction(gchi_r, gchi0, u_loc)

    gchi0 = gchi0.take_vn_diagonal()
    logger.info(
        f"Local irreducible vertex Gamma^wvv' ({gamma_r.channel.value})"
        f"{" with asymptotic correction" if config.box.niv_shell > 0 else ""} calculated."
    )

    f_r = create_full_vertex_from_gamma(gamma_r, gchi0, u_loc)
    logger.info(f"Local full vertex F^wvv' ({f_r.channel.value}) calculated.")

    gchi_r_aux = create_auxiliary_chi(gamma_r, gchi0_inv_core, u_loc)
    logger.info(f"Local auxiliary susceptibility chi^*wvv' ({gchi_r_aux.channel.value}) calculated.")

    vrg_r = create_vrg(gchi_r_aux, gchi0_inv_core)
    logger.info(f"Local three-leg vertex gamma^wv ({vrg_r.channel.value}) calculated.")

    gchi_r_aux_sum = gchi_r_aux.sum_over_all_vn(config.sys.beta)
    del gchi_r_aux

    gchi_r_aux_sum = create_generalized_chi_with_shell_correction(gchi_r_aux_sum, gchi0, u_loc)
    logger.info(
        f"Updated local susceptibility chi^w ({gchi_r_aux_sum.channel.value})"
        f"{" with asymptotic correction" if config.box.niv_shell > 0 else ""}."
    )

    return gamma_r, gchi_r_aux_sum, vrg_r, f_r, gchi_r


def get_local_hartree_fock(u_loc: LocalInteraction, occ: np.ndarray) -> np.ndarray:
    r"""
    Returns the local Hartree-Fock (static, frequency-independent) self-energy
    :math:`\Sigma^{HF}_{12}` from the bare interaction and the local occupation, i.e. the density-channel
    interaction contracted with the occupation, see Eq. (3.55) in my master's thesis.

    The interaction tensor is stored with the inter-orbital density :math:`U'` at :math:`U_{1212}` (the convention
    of :meth:`Hamiltonian.kanamori_interaction_dp` and the w2dynamics ``umatrix`` files), whereas the density-channel
    projection contracted as ``"abcd,dc->ab"`` picks up :math:`U_{1122}`. The middle two orbital indices are
    therefore swapped (``"abcd->acbd"``) before the projection, so that the Hartree term uses :math:`U'` while the
    Fock term still uses :math:`U_{1432}`. This only affects multi-orbital systems with off-diagonal interactions;
    single-orbital and purely orbital-diagonal interactions are unchanged.

    :param u_loc: The bare local interaction :math:`U`.
    :param occ: The local occupation matrix :math:`n_{12}`, shape ``[n_bands, n_bands]``.
    :return: The Hartree-Fock self-energy, shape ``[n_bands, n_bands]``.
    """
    return u_loc.permute_orbitals("abcd->acbd").as_channel(SpinChannel.DENS).times("abcd,dc->ab", occ)


def get_loc_self_energy_vrg(
    vrg_dens: LocalFourPoint,
    vrg_magn: LocalFourPoint,
    gchi_dens_sum: LocalFourPoint,
    gchi_magn_sum: LocalFourPoint,
    g_dmft: GreensFunction,
    u_loc: LocalInteraction,
) -> SelfEnergy:
    r"""
    Performs the local self-energy calculation using the Schwinger-Dyson equation, i.e. the local variant of Eq. (3.64)
    in my master's thesis. This is done to verify the implementation of the Schwinger-Dyson equation with the three-leg
    vertex and the local susceptibility against the DMFT self-energy. Note that there will never be a perfect match due
    to the sampling method of w2dynamics and the stochastic nature of the CTQMC solver. Nevertheless, the results should
    be very close. For more details, see also Paul Worm's PhD thesis, Eq. (3.70) and Anna Galler's PhD Thesis, P. 76 ff.

    :param vrg_dens: The density three-leg vertex :math:`\gamma_{\mathrm{dens}}`.
    :param vrg_magn: The magnetic three-leg vertex :math:`\gamma_{\mathrm{magn}}`.
    :param gchi_dens_sum: The frequency-summed density susceptibility :math:`\chi_{\mathrm{dens}}^{\omega}`.
    :param gchi_magn_sum: The frequency-summed magnetic susceptibility :math:`\chi_{\mathrm{magn}}^{\omega}`.
    :param g_dmft: The local (DMFT) :class:`GreensFunction`.
    :param u_loc: The bare local interaction :math:`U`.
    :return: The local :class:`SelfEnergy` (including the Hartree-Fock term).
    """
    # 1=i, 2=j, 3=k, 4=l, 7=o, 8=p
    g_wv = g_dmft.get_g_wv(MFHelper.wn(config.box.niw_core), config.box.niv_core)
    inner = vrg_dens - vrg_dens @ u_loc.as_channel(SpinChannel.DENS) @ gchi_dens_sum
    inner -= vrg_magn - vrg_magn @ u_loc.as_channel(SpinChannel.MAGN) @ gchi_magn_sum
    inner = 0.5 * inner.to_full_niw_range()
    sigma_sum = -1.0 / config.sys.beta * u_loc.times("kjop,ilpowv,lkwv->ijv", inner, g_wv)
    hartree_fock = get_local_hartree_fock(u_loc, config.sys.occ_dmft)[..., None]
    return SelfEnergy((hartree_fock + sigma_sum)[None, None, None, ...], beta=config.sys.beta)


def perform_local_schwinger_dyson(
    g_dmft: GreensFunction, g2_dens: LocalFourPoint, g2_magn: LocalFourPoint, u_loc: LocalInteraction
):
    r"""
    Performs the local Schwinger-Dyson equation calculation for the local self-energy for sanity checks against the
    DMFT self-energy. Includes the calculation of the local three-leg and full vertices, (auxiliary/bare/physical)
    susceptibilities and the irreducible vertices for both the density and magnetic channel. Employs explicit
    asymptotics as proposed by Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005; DOI 10.1088/2515-7639/ac7e6d.

    :param g_dmft: The local (DMFT) :class:`GreensFunction`.
    :param g2_dens: The two-particle (DMFT) Green's function in the density channel.
    :param g2_magn: The two-particle (DMFT) Green's function in the magnetic channel.
    :param u_loc: The bare local interaction :math:`U`.
    :return: The tuple ``(gamma_d, gamma_m, gchi_d_sum, gchi_m_sum, vrg_d, vrg_m, f_d, f_m, gchi_d, gchi_m,
        sigma_loc)`` of local vertex functions and the local self-energy.
    """
    gchi0 = BubbleGenerator.create_generalized_chi0(g_dmft, config.box.niw_core, config.box.niv_full, config.sys.beta)

    if config.dmft.symmetrize_orbitals:
        gchi0 = gchi0.symmetrize_orbitals(config.dmft.symmetrize_orbitals)
        config.logger.info(
            f"Symmetrized gchi0 with respect to orbitals {', '.join(str(o) for o in config.dmft.symmetrize_orbitals)}."
        )

    if config.eliashberg.perform_eliashberg:
        gchi0.save(name="gchi0_loc", output_dir=config.output.output_path)

    gchi0_inv_core = gchi0.cut_niv(config.box.niv_core).invert().take_vn_diagonal()

    gamma_d, gchi_d_sum, vrg_d, f_d, gchi_d = create_vertex_functions(g2_dens, gchi0, gchi0_inv_core, g_dmft, u_loc)
    gamma_m, gchi_m_sum, vrg_m, f_m, gchi_m = create_vertex_functions(g2_magn, gchi0, gchi0_inv_core, g_dmft, u_loc)

    if config.dmft.symmetrize_orbitals:
        for vertex in [gamma_d, gamma_m, gchi_d_sum, gchi_m_sum, vrg_d, vrg_m, f_d, f_m, gchi_d, gchi_m]:
            vertex.symmetrize_orbitals(config.dmft.symmetrize_orbitals)
        config.logger.info(
            f"Symmetrized local vertex functions with respect to orbitals "
            f"{', '.join(str(o) for o in config.dmft.symmetrize_orbitals)}."
        )

    sigma_loc = get_loc_self_energy_vrg(vrg_d, vrg_m, gchi_d_sum, gchi_m_sum, g_dmft, u_loc)

    if config.dmft.symmetrize_orbitals:
        sigma_loc = sigma_loc.symmetrize_orbitals(config.dmft.symmetrize_orbitals)
        config.logger.info(
            f"Symmetrized local self-energy with respect to orbitals "
            f"{', '.join(str(o) for o in config.dmft.symmetrize_orbitals)}."
        )

    return gamma_d, gamma_m, gchi_d_sum, gchi_m_sum, vrg_d, vrg_m, f_d, f_m, gchi_d, gchi_m, sigma_loc


# ----------------------------------------------- AbinitioDGA algorithms -----------------------------------------------
#
# DEVELOPMENT / TESTING ONLY. The functions below are an alternative ("ab-initio DGA") rewriting of the local
# Schwinger-Dyson cross-check: they build the local self-energy from the full vertex F and the density three-leg
# vertex derived from it (density channel only, by design), as opposed to the auxiliary-susceptibility (Hedin)
# form used by the production routine above. They are NOT part of the production routine (``DGAmore.py`` calls
# :func:`perform_local_schwinger_dyson`) and exist only as an internal consistency check, hence are left untested.


def get_loc_self_energy_gamma_abinitio_dga(
    gamma_dens: LocalFourPoint, u_loc: LocalInteraction, g_loc: GreensFunction
) -> SelfEnergy:
    r"""
    DEVELOPMENT / TESTING ONLY. Returns the local self-energy with the density three-leg :math:`\gamma` from
    ab-initio DGA (density channel only, by design),

    .. math:: \Sigma_{ij}^{\nu} = -\frac{1}{\beta} \sum_\omega U_{iabc}\, \gamma_{cbdj}^{\omega\nu}\, G_{ad}^{\omega-\nu}.

    :param gamma_dens: The density three-leg vertex :math:`\gamma_{\mathrm{dens}}` (ab-initio convention).
    :param u_loc: The bare local interaction :math:`U`.
    :param g_loc: The local :class:`GreensFunction`.
    :return: The local :class:`SelfEnergy` (including the Hartree term).
    """
    g_wv = g_loc.get_g_wv(MFHelper.wn(config.box.niw_core), config.box.niv_core)
    sigma = -1.0 / config.sys.beta * u_loc.times("iabc,cbdjwv,adwv->ijv", gamma_dens.to_full_niw_range(), g_wv)
    hartree = u_loc.as_channel(SpinChannel.DENS).times("abcd,dc->ab", config.sys.occ_dmft)[..., None]
    return SelfEnergy((sigma + hartree)[None, None, None, ...], beta=config.sys.beta)


def perform_local_schwinger_dyson_abinitio_dga(
    g_loc: GreensFunction,
    g2_dens: LocalFourPoint,
    g2_magn: LocalFourPoint,
    u_loc: LocalInteraction,
):
    r"""
    DEVELOPMENT / TESTING ONLY -- this is the ab-initio DGA cross-check, NOT the production routine
    (:func:`perform_local_schwinger_dyson`). Performs the local Schwinger-Dyson equation for the local (DMFT)
    self-energy as an internal sanity check, building the local self-energy from the full vertices.

    :param g_loc: The local :class:`GreensFunction`.
    :param g2_dens: The two-particle (DMFT) Green's function in the density channel.
    :param g2_magn: The two-particle (DMFT) Green's function in the magnetic channel.
    :param u_loc: The bare local interaction :math:`U`.
    :return: The tuple ``(gchi_dens_loc, gchi_magn_loc, gchi0_loc_full, one_plus_gamma_dens_loc,
        one_plus_gamma_magn_loc, f_dens_loc, f_magn_loc, sigma_loc)``.
    """
    logger = config.logger

    gchi_dens_loc = create_generalized_chi(g2_dens, g_loc)
    logger.info("Generalized susceptibility chi^wvv' (dens) calculated.")
    del g2_dens
    gchi_magn_loc = create_generalized_chi(g2_magn, g_loc)
    logger.info("Generalized susceptibility chi^wvv' (magn) calculated.")
    del g2_magn

    gchi0_loc_full = BubbleGenerator.create_generalized_chi0(
        g_loc, config.box.niw_core, config.box.niv_full, config.sys.beta
    )
    logger.info("Local bare susceptibility chi_0^wv calculated.")
    gchi0_core = gchi0_loc_full.cut_niv(config.box.niv_core)

    # 1 + chi0 * F_r = gchi_r * (chi0)^(-1) = 1 + gamma_r or
    # F_r = -beta^2 * [chi0^(-1) - chi0^(-1) chi_r chi0^(-1)]
    # gamma_r is NOT the irreducible vertex in channel r but rather the three-point vertex from AbinitioDGA
    gchi0_inv_core = gchi0_core.invert()
    f_dens_loc = (gchi0_inv_core - gchi0_inv_core @ gchi_dens_loc @ gchi0_inv_core).scale(-config.sys.beta**2)
    logger.info("Local full vertex F^wvv' (dens) calculated.")
    f_magn_loc = (gchi0_inv_core - gchi0_inv_core @ gchi_magn_loc @ gchi0_inv_core).scale(-config.sys.beta**2)
    logger.info("Local full vertex F^wvv' (magn) calculated.")
    del gchi0_inv_core

    # f_dens_loc_with_asympt = create_asympt_f(gchi_dens_loc, gchi_magn_loc, gchi_ud_pp_loc_sum, u_loc)

    # in most equations we need 1 + gamma_r so we add it here
    gamma_dens_loc = (gchi0_core @ f_dens_loc).sum_over_vn(config.sys.beta, axis=(-2,)).scale(1.0 / config.sys.beta)
    one_plus_gamma_dens_loc = LocalFourPoint.identity_like(gamma_dens_loc) + gamma_dens_loc
    logger.info("Local three-leg vertex gamma^wv (dens) calculated.")

    gamma_magn_loc = (gchi0_core @ f_magn_loc).sum_over_vn(config.sys.beta, axis=(-2,)).scale(1.0 / config.sys.beta)
    one_plus_gamma_magn_loc = LocalFourPoint.identity_like(gamma_magn_loc) + gamma_magn_loc
    logger.info("Local three-leg vertex gamma^wv (magn) calculated.")
    del gchi0_core, gamma_magn_loc

    sigma_loc = get_loc_self_energy_gamma_abinitio_dga(gamma_dens_loc, u_loc, g_loc)
    logger.info("Local self-energy calculated.")
    del gamma_dens_loc

    return (
        gchi_dens_loc,
        gchi_magn_loc,
        gchi0_loc_full,
        one_plus_gamma_dens_loc,
        one_plus_gamma_magn_loc,
        f_dens_loc,
        f_magn_loc,
        sigma_loc,
    )

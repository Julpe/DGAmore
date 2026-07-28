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
from dgamore.n_point_base import SpinChannel, FrequencyNotation, DTYPE
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
        g_loc_slice_mat = g_dmft.mat[0, 0, 0][..., g_dmft.niv - config.box.niv_core : g_dmft.niv + config.box.niv_core]
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
    # the +U couples ALL fermionic frequencies (a rank-orbital^2 update on the block-diagonal inverse bubble), so the
    # core box of (extend(chi0^-1) + U/beta^2)^-1 is built by the Woodbury identity - no full-box dense inversion.
    chi_tilde_core_inv = gchi0.get_core_from_shell_inversion(
        1.0 / config.sys.beta**2 * u_loc.as_channel(gchi_r.channel), config.box.niv_core
    ).invert()
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


def create_full_vertex_from_gamma(gamma_r: LocalFourPoint, gchi0: LocalFourPoint, u_loc: LocalInteraction):
    r"""
    Returns the local full vertex from the irreducible vertex,
    :math:`F = \Gamma [1 + \frac{1}{\beta^2} \chi_0 \Gamma]^{-1}` (with :math:`\Gamma` padded with :math:`U` beyond
    the core box), on the asymmetric :math:`2 n_{\nu,\mathrm{full}} \times 2 n_{\nu,\mathrm{core}}` box.

    The vertex is only ever read with its second (free) fermionic index on the core box: the double-counting kernel
    of the non-local self-energy sums the first index over the full asymptotic range but keeps the second one only on
    the core box, and the Eliashberg equation cuts both indices to ``niv_pp``, which is at most half the core box. The
    free index is therefore restricted right away instead of being computed and discarded. Only the double-counting
    kernel needs the summed index on the full box, so this asymmetric object is written to its own ``f_dc_loc`` file
    while the per-channel ``f_dens_loc``/``f_magn_loc`` files are cut square to the core box.

    :param gamma_r: The irreducible vertex :math:`\Gamma_{r}` (core box).
    :param gchi0: The bare bubble :math:`\chi_0` (with its fermionic axis taken on the diagonal).
    :param u_loc: The bare local interaction :math:`U`, used to pad the shell.
    :return: The full vertex :math:`F_{r}` as a :class:`LocalFourPoint`.
    :raises ValueError: If the full fermionic box is narrower than the vertex's own core box.
    """
    niv_full = config.box.niv_full
    if niv_full < gamma_r.niv:
        raise ValueError("The full fermionic box cannot be narrower than the core box of the irreducible vertex.")
    if niv_full == gamma_r.niv:  # without a shell the low-rank split is not smaller, so solve densely
        return _solve_full_vertex_dense(gamma_r, gchi0, config.sys.beta)
    return _solve_full_vertex_push_through(gamma_r, gchi0, u_loc.as_channel(gamma_r.channel), config.sys.beta, niv_full)


def _half_niw_bubble_and_w_dim(gamma_r: LocalFourPoint, gchi0: LocalFourPoint) -> tuple[np.ndarray, int]:
    r"""
    Returns the bubble's array aligned to the vertex's bosonic range together with the number of bosonic slices.

    The bubble arrives over the full bosonic range while the irreducible vertex is stored half-range, so the
    :math:`\omega \geq 0` block is taken as a read-only view (``gchi0`` is not modified).

    :param gamma_r: The irreducible vertex, which defines the bosonic range to align to.
    :param gchi0: The bare bubble with a single fermionic axis.
    :return: The tuple ``(chi0_mat, w_dim)``.
    :raises ValueError: If the aligned bubble does not carry one bosonic slice per slice of the vertex.
    """
    chi0_mat = gchi0.mat
    if gchi0.full_niw_range:
        chi0_mat = chi0_mat[..., chi0_mat.shape[-2] // 2 :, :]
    w_dim = gamma_r.current_shape[-3]
    if chi0_mat.shape[-2] != w_dim:
        raise ValueError(f"Bubble carries {chi0_mat.shape[-2]} bosonic slices, but the vertex has {w_dim}.")
    return chi0_mat, w_dim


def _solve_full_vertex_push_through(
    gamma_r: LocalFourPoint, gchi0: LocalFourPoint, u_channel: LocalInteraction, beta: float, niv_full: int
) -> LocalFourPoint:
    r"""
    Returns :math:`F = \Gamma_U [\mathbb{1} + \frac{1}{\beta^2}\chi_0 \Gamma_U]^{-1}` on the
    :math:`2 n_{\nu,\mathrm{full}} \times 2 n_{\nu,\mathrm{core}}` box without ever building the padded vertex
    :math:`\Gamma_U`.

    Padding fills :math:`\mathcal{U}` over the whole box and overwrites only the core block, so the padded vertex
    splits exactly as :math:`\Gamma_U = \mathcal{U}\otimes J + \Delta`, with :math:`J` the all-ones matrix in
    :math:`(\nu, \nu')` and :math:`\Delta = \Gamma - \mathcal{U}` supported on the core block alone. Both terms are
    of low rank in the compound index, i.e. :math:`\Gamma_U = LR` with
    :math:`r = \mathrm{orbital}^2 (1 + 2 n_{\nu,\mathrm{core}})` against the compound dimension
    :math:`N = 2 n_{\nu,\mathrm{full}}\,\mathrm{orbital}^2`. With :math:`C` the bubble (block-diagonal in
    :math:`\nu`), the push-through identity :math:`R(\mathbb{1} + CLR)^{-1} = (\mathbb{1}_r + RCL)^{-1}R` gives

    .. math:: F = L\,(\mathbb{1}_r + R\,C\,L)^{-1}\,R ,

    replacing the :math:`O(N^3)` dense solve per :math:`\omega` by an :math:`O(r^3)` one. Only the core columns of
    :math:`R` are needed, since the free index :math:`\nu'` is kept on the core box; applying :math:`L` is a
    :math:`\nu`-independent broadcast plus a scatter onto the core rows of :math:`\nu`.

    :param gamma_r: The irreducible vertex :math:`\Gamma_{r}` on the core box.
    :param gchi0: The bare bubble :math:`\chi_0` on the ``niv_full`` box, fermionic axis taken on the diagonal.
    :param u_channel: The channel-projected local interaction :math:`\mathcal{U}` the shell is padded with.
    :param beta: Inverse temperature :math:`\beta`.
    :param niv_full: Number of positive fermionic frequencies of the summed index :math:`\nu`.
    :return: The full vertex as a two-fermion :class:`LocalFourPoint`. ``gamma_r`` is not modified.
    """
    o = gamma_r.n_bands
    o2 = o * o
    nf, nc = 2 * niv_full, 2 * gamma_r.niv
    core = slice(niv_full - gamma_r.niv, niv_full + gamma_r.niv)

    chi0_mat, w_dim = _half_niw_bubble_and_w_dim(gamma_r, gchi0)
    u_c = u_channel.mat.transpose(0, 1, 3, 2).reshape(o2, o2)

    r = o2 + o2 * nc
    diag_r = np.arange(r)
    out = np.empty((o, o, o, o, w_dim, nf, nc), dtype=DTYPE)

    for iw in range(w_dim):
        chi0_w = chi0_mat[..., iw, :].transpose(4, 0, 1, 3, 2).reshape(nf, o2, o2) / beta**2
        chi0_core = chi0_w[core]
        gamma_w = gamma_r.mat[..., iw, :, :].transpose(0, 1, 4, 3, 2, 5).reshape(o2, nc, o2, nc)
        delta = gamma_w - u_c[:, None, :, None]

        m = np.empty((r, r), dtype=DTYPE)
        m[:o2, :o2] = u_c @ chi0_w.sum(axis=0)  # the shell enters only through this full-box bubble sum
        m[:o2, o2:] = np.einsum("ja,vak->jkv", u_c, chi0_core, optimize=True).reshape(o2, o2 * nc)
        d2 = np.einsum("jwav,vak->jwkv", delta, chi0_core, optimize=True)
        m[o2:, :o2] = d2.sum(axis=-1).reshape(o2 * nc, o2)
        m[o2:, o2:] = d2.reshape(o2 * nc, o2 * nc)
        m[diag_r, diag_r] += 1.0

        rhs = np.empty((r, o2 * nc), dtype=DTYPE)
        rhs[:o2] = np.repeat(u_c, nc, axis=1)
        rhs[o2:] = delta.reshape(o2 * nc, o2 * nc)
        y = np.linalg.solve(m, rhs)

        # expand L: the first block is nu-independent, the second one only corrects the core rows of nu
        block = np.empty((o2, nf, o2, nc), dtype=DTYPE)
        block[...] = y[:o2].reshape(o2, o2, nc)[:, None]
        block[:, core] += y[o2:].reshape(o2, nc, o2, nc)
        out[..., iw, :, :] = block.reshape(o, o, nf, o, o, nc).transpose(0, 1, 4, 3, 2, 5)

    return LocalFourPoint(
        out, gamma_r.channel, 1, 2, gamma_r.full_niw_range, gamma_r.full_niv_range, gamma_r.frequency_notation
    )


def _solve_full_vertex_dense(gamma_r: LocalFourPoint, gchi0: LocalFourPoint, beta: float) -> LocalFourPoint:
    r"""
    Returns :math:`F = \Gamma [\mathbb{1} + \frac{1}{\beta^2}\chi_0 \Gamma]^{-1}` by a dense compound solve per
    :math:`\omega`, for the shell-free case where the vertex already spans the full fermionic box.

    Used instead of :func:`_solve_full_vertex_push_through` when there is nothing to pad: the low-rank split of the
    padded vertex is then no smaller than the compound dimension itself, so it would only add rounding.

    :param gamma_r: The irreducible vertex :math:`\Gamma_{r}`, spanning the full fermionic box.
    :param gchi0: The bare bubble :math:`\chi_0`, fermionic axis taken on the diagonal.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The full vertex as a two-fermion :class:`LocalFourPoint`. ``gamma_r`` is not modified.
    """
    o = gamma_r.n_bands
    o2, nv = o * o, 2 * gamma_r.niv
    n = o2 * nv

    chi0_mat, w_dim = _half_niw_bubble_and_w_dim(gamma_r, gchi0)
    diag = np.arange(n)
    out = np.empty((o, o, o, o, w_dim, nv, nv), dtype=DTYPE)

    for iw in range(w_dim):
        gamma_w = gamma_r.mat[..., iw, :, :].transpose(0, 1, 4, 3, 2, 5).reshape(o2, nv, o2, nv)
        chi0_w = chi0_mat[..., iw, :].transpose(4, 0, 1, 3, 2).reshape(nv, o2, o2) / beta**2
        a = np.einsum("vij,jvkl->ivkl", chi0_w, gamma_w, optimize=True).reshape(n, n)
        a[diag, diag] += 1.0
        f = np.linalg.solve(a.T, gamma_w.reshape(n, n).T).T
        out[..., iw, :, :] = f.reshape(o, o, nv, o, o, nv).transpose(0, 1, 4, 3, 2, 5)

    return LocalFourPoint(
        out, gamma_r.channel, 1, 2, gamma_r.full_niw_range, gamma_r.full_niv_range, gamma_r.frequency_notation
    )


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
    Returns the local Hartree-Fock (static, frequency-independent) self-energy :math:`\Sigma^{\mathrm{HF}}_{12}` from
    the bare interaction and the local occupation, i.e. the density-channel interaction contracted with the occupation,
    see Eq. (3.55) in my master's thesis.

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

# DEVELOPMENT / TESTING ONLY: an alternative ("ab-initio DGA") local Schwinger-Dyson cross-check building the local
# self-energy from F and its density three-leg vertex (density only), not the Hedin form used in production. Untested.


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

    # 1 + chi0 * F_r = gchi_r * chi0^(-1) = 1 + gamma_r, i.e. F_r = -beta^2 [chi0^(-1) - chi0^(-1) chi_r chi0^(-1)];
    # gamma_r is NOT the irreducible vertex in channel r but the three-point vertex from AbinitioDGA.
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

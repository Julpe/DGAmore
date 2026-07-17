# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Numerical kernels of the non-local ladder DGA step. Starting from the local irreducible vertex :math:`\Gamma_{r}`,
the inverse bare bubble and the interactions, the functions here build, per momentum :math:`q` and spin channel,
the Hartree-Fock term, the auxiliary susceptibility :math:`\chi^{*;q}_{r}`, the three-leg vertices
:math:`\gamma^q_{r}` / :math:`\tilde{\gamma}^q_{r}`, the shell-corrected physical susceptibility and the
self-energy kernels, and contract the kernel with the Green's function to get the momentum-dependent self-energy
:math:`\Sigma(k, \nu)` (several CPU/GPU/FFT variants, distributed over MPI ranks). The self-consistency driver
that orchestrates these lives in :mod:`dgamore.nonlocal_sde`. Equation numbers refer to the author's master's
thesis (Chapters 3 & 4).
"""

import mpi4py.MPI as MPI
import numpy as np

import dgamore.config as config
import dgamore.mpi_utils as mpi_utils
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.local_four_point import LocalFourPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel
from dgamore.self_energy import SelfEnergy


def get_hartree_fock(
    u_loc: LocalInteraction, v_nonloc: Interaction, q_list: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Returns the Hartree-Fock term separately for the local and non-local interaction. Since we are always SU(2)-symmetric,
    the sum over the spins of the first term in Eq. (4.55) in Anna Galler's thesis results in a simple factor of 2. This
    can be seen in my master's thesis, Eq. (3.55). The Hartree-Fock term is given by

    .. math:: \Sigma_{HF}^k = 2(U_{acbd} + V^{q=0}_{acbd}) n_{dc} - 1/N_q \sum_q (U_{adcb} + V^{q}_{adcb}) n^{k-q}_{dc}

    where the Hartree term reads :math:`\Sigma_{H} = 2(U_{acbd} + V^{q=0}_{acbd}) n_{dc}` and the Fock term reads
    :math:`\Sigma_{F}^k = - 1/N_q \sum_q (U_{adcb} + V^{q}_{adcb}) n^{k-q}_{dc}`. The Hartree contraction uses the
    middle-index-swapped ``U_{acbd}`` so it picks up the inter-orbital density :math:`U'` (stored at :math:`U_{abab}`);
    see :func:`dgamore.local_sde.get_local_hartree_fock`.
    Processes the Fock term for each individual orbital to save memory, as for high momentum grids,
    the ``occ_qk`` property can become large.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}` (see :class:`Interaction`).
    :param q_list: Array of integer q-point index triplets handled by this rank.
    :return: The tuple ``(hartree, fock)`` of self-energy contributions, broadcastable to ``[k, o1, o2, v]``.
    """
    v_q0 = v_nonloc.find_q((0, 0, 0))
    # The inter-orbital density U' is stored at U_{abab}, so the Hartree term contracts "qacbd" (swapping the middle
    # orbital indices) to pick it up; the Fock term below uses U_{adcb}. See local_sde.get_local_hartree_fock.
    hartree = 2 * (u_loc + v_q0).times("qacbd,dc->ab", config.sys.occ)

    nb = config.sys.n_bands
    nk_tot = np.prod(config.lattice.nk)
    nq_tot = np.prod(config.lattice.nk)

    uq = (u_loc + v_nonloc.reduce_q(q_list)).permute_orbitals("abcd->adcb")  # (nq,a,d,c,b)

    fock = np.zeros((nk_tot, nb, nb), dtype=uq.mat.dtype)

    for d in range(nb):
        for c in range(nb):
            u_slice = uq[:, :, d, c, :]
            if not np.any(u_slice):
                continue

            occ_qk_dc = np.array(
                [np.roll(config.sys.occ_k[..., d, c], [-i for i in q], axis=(0, 1, 2)) for q in q_list]
            )
            occ_qk_dc = occ_qk_dc.reshape(len(q_list), nk_tot)
            # contract over q directly: the broadcast product materialized a full [q, k, o1, o2] block
            fock += np.einsum("qab,qk->kab", u_slice, occ_qk_dc, optimize=True)

    fock *= -1.0 / nq_tot
    return hartree[None, ..., None], fock[..., None]  # [k,o1,o2,v]


def create_inverse_auxiliary_chi_r_q(gamma_r: LocalFourPoint, gchi0_q_inv: FourPoint, u_r: Interaction) -> FourPoint:
    r"""
    Assembles the Bethe-Salpeter matrix whose inversion yields the auxiliary susceptibility (Eq. (3.60) in my
    master's thesis),

    .. math:: M^{q\nu\nu'}_{1234} = (\chi_{0;1234}^{q\nu})^{-1}\delta_{\nu\nu'} + (\Gamma_{r;1234}^{\omega\nu\nu'}-\mathcal{U}_{r;1234}^{q})/\beta^2,

    in a **single** two-fermion block: the result is broadcast-filled with the scaled local vertex over the momentum
    axis, the inverse bubble is added on the fermionic frequency diagonal in place (see
    :meth:`~dgamore.local_four_point.LocalFourPoint.add_on_vn_diagonal`) and the channel interaction is subtracted
    in place. The former add/extend/subtract chain held two of these blocks alive at its peak (the diagonally
    extended bubble plus the out-of-place result). Neither input is mutated.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` (full or half bosonic range; read via a
        half-range view).
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi_0^q)^{-1}` (half bosonic range, one fermionic dimension).
    :param u_r: The channel-projected total interaction :math:`\mathcal{U}_{r}^{q} = U_{r} + V_{r}^{q}`.
    :return: The assembled matrix :math:`M^{q}` as a :class:`FourPoint` (half niw range, two fermionic dimensions).
    """
    beta = config.sys.beta
    gamma_mat = gamma_r.mat
    if gamma_r.full_niw_range:
        gamma_mat = gamma_mat[..., gamma_mat.shape[-3] // 2 :, :, :]

    out = np.empty((gchi0_q_inv.current_shape[0],) + gamma_mat.shape, dtype=gchi0_q_inv.mat.dtype)
    np.multiply(gamma_mat, 1.0 / beta**2, out=out)

    return (
        FourPoint(out, gamma_r.channel, gchi0_q_inv.nq, 1, 2, False, gchi0_q_inv.full_niv_range, True)
        .add_on_vn_diagonal(gchi0_q_inv)
        .sub(u_r.scale(1.0 / beta**2, copy=True), copy=False)
    )


def create_auxiliary_chi_r_q(
    gamma_r: LocalFourPoint, gchi0_q_inv: FourPoint, u_loc: LocalInteraction, v_nonloc: Interaction
) -> FourPoint:
    r"""
    Returns the auxiliary susceptibility, see Eq. (3.60) in my master's thesis,

    .. math:: \chi^{*;q\nu\nu'}_{r;abcd} = ((\chi_{0;abcd}^{q\nu})^{-1} + (\Gamma_{r;abcd}^{\omega\nu\nu'}-U_{r;abcd}-V_{r;abcd}^q)/\beta^2)^{-1},

    with the matrix to invert assembled in a single block by :func:`create_inverse_auxiliary_chi_r_q`.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}`.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi_0^q)^{-1}` (core box).
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :return: The momentum-dependent auxiliary susceptibility :math:`\chi^{*;q}_{r}` as a :class:`FourPoint`.
    """
    u_r = v_nonloc.as_channel(gamma_r.channel) + u_loc.as_channel(gamma_r.channel)
    return create_inverse_auxiliary_chi_r_q(gamma_r, gchi0_q_inv, u_r).invert(False)


def create_auxiliary_chi_r_q_sum_v1(
    gamma_r: LocalFourPoint, gchi0_q_inv: FourPoint, u_loc: LocalInteraction, v_nonloc: Interaction
) -> FourPoint:
    r"""
    Returns the sum over the auxiliary susceptibility, see Eq. (3.60) in my master's thesis. This variant inverts
    and sums over the last fermionic frequency in one fused step (see
    :meth:`FourPoint.invert_and_sum_over_last_vn`),

    .. math:: \sum_{\nu'}\chi^{*;q\nu\nu'}_{r;abcd} = \sum_{\nu'}((\chi_{0;abcd}^{q\nu})^{-1} + (\Gamma_{r;abcd}^{\omega\nu\nu'}-U_{r;abcd}-V_{r;abcd}^q)/\beta^2)^{-1}.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}`.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi_0^q)^{-1}` (core box).
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :return: The frequency-summed auxiliary susceptibility :math:`\sum_{\nu'}\chi^{*;q}_{r}` as a :class:`FourPoint`.
    """
    u_r = v_nonloc.as_channel(gamma_r.channel) + u_loc.as_channel(gamma_r.channel)
    return create_inverse_auxiliary_chi_r_q(gamma_r, gchi0_q_inv, u_r).invert_and_sum_over_last_vn(config.sys.beta)


def create_auxiliary_chi_r_q_sum_v2(
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    mpi_dist_irrq: MpiDistributor,
) -> FourPoint:
    r"""
    Returns the sum over the auxiliary susceptibility, see Eq. (3.60) in my master's thesis. This variant loops over
    the rank-local q-points (capping peak memory to one q at a time) and uses the standard fused invert-and-sum per
    q (see :meth:`FourPoint.invert_and_sum_over_last_vn`),

    .. math:: \sum_{\nu'}\chi^{*;q\nu\nu'}_{r;abcd} = \sum_{\nu'}((\chi_{0;abcd}^{q\nu})^{-1} + (\Gamma_{r;abcd}^{\omega\nu\nu'}-U_{r;abcd}-V_{r;abcd}^q)/\beta^2)^{-1}.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}`.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi_0^q)^{-1}` (core box).
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :param mpi_dist_irrq: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :return: The frequency-summed auxiliary susceptibility :math:`\sum_{\nu'}\chi^{*;q}_{r}` as a :class:`FourPoint`.
    """
    irrk_q_list = config.lattice.k_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrq.my_slice]
    chi_r_q_sum_mat = np.zeros_like(gchi0_q_inv.mat)

    u_r = v_nonloc.as_channel(gamma_r.channel) + u_loc.as_channel(gamma_r.channel)
    for idx in range(len(my_irr_q_list)):
        chi_r_q_sum_mat[idx] = (
            create_inverse_auxiliary_chi_r_q(gamma_r, gchi0_q_inv.filter_q_index(idx), u_r.filter_q_index(idx))
            .invert_and_sum_over_last_vn(config.sys.beta)
            .mat
        )
    return FourPoint(chi_r_q_sum_mat, gamma_r.channel, config.lattice.nk, 1, 1, False, has_compressed_q_dimension=True)


def create_auxiliary_chi_r_q_sum_v3(
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    mpi_dist_irrq: MpiDistributor,
) -> FourPoint:
    r"""
    Returns the sum over the auxiliary susceptibility, see Eq. (3.60) in my master's thesis. This is the most
    memory-lean variant: it loops over the rank-local q-points and uses the highly memory-efficient
    linear-solver-based fused invert-and-sum per q (see :meth:`FourPoint.invert_and_sum_over_last_vn_v2`),

    .. math:: \sum_{\nu'}\chi^{*;q\nu\nu'}_{r;abcd} = \sum_{\nu'}((\chi_{0;abcd}^{q\nu})^{-1} + (\Gamma_{r;abcd}^{\omega\nu\nu'}-U_{r;abcd}-V_{r;abcd}^q)/\beta^2)^{-1}.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}`.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi_0^q)^{-1}` (core box).
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :param mpi_dist_irrq: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :return: The frequency-summed auxiliary susceptibility :math:`\sum_{\nu'}\chi^{*;q}_{r}` as a :class:`FourPoint`.
    """
    irrk_q_list = config.lattice.k_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrq.my_slice]
    chi_r_q_sum_mat = np.zeros_like(gchi0_q_inv.mat)

    u_r = v_nonloc.as_channel(gamma_r.channel) + u_loc.as_channel(gamma_r.channel)
    for idx in range(len(my_irr_q_list)):
        chi_r_q_sum_mat[idx] = (
            create_inverse_auxiliary_chi_r_q(gamma_r, gchi0_q_inv.filter_q_index(idx), u_r.filter_q_index(idx))
            .invert_and_sum_over_last_vn_v2(config.sys.beta)
            .mat
        )
    return FourPoint(chi_r_q_sum_mat, gamma_r.channel, config.lattice.nk, 1, 1, False, has_compressed_q_dimension=True)


def create_vrg_r_q(gchi_aux_q_r_sum: FourPoint, gchi0_q_inv: FourPoint) -> FourPoint:
    r"""
    Returns the momentum-dependent three-leg vertex, see Eq. (3.63) in my master's thesis,
    :math:`\gamma_{r;1234}^{q\nu} = \beta \sum_{ab} \sum_{\nu'} (\chi^{q\nu}_{0;12ab})^{-1} \chi^{*;q\nu\nu'}_{r;ba34}`.

    :param gchi_aux_q_r_sum: The frequency-summed auxiliary susceptibility :math:`\sum_{\nu'}\chi^{*;q\nu\nu'}_{r}`.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi_0^q)^{-1}` (core box).
    :return: The three-leg vertex :math:`\gamma^q_{r}` (``vrg``) as a :class:`FourPoint`.
    """
    return (gchi0_q_inv @ gchi_aux_q_r_sum).scale(config.sys.beta)


def create_vrg_r_q_right(gchi_aux_q_r_sum: FourPoint, gchi0_q_inv: FourPoint) -> FourPoint:
    r"""
    Returns the momentum-dependent right-sided three-leg vertex, i.e. the counterpart of :meth:`create_vrg_r_q`
    (Eq. (3.63) in my master's thesis) with the summed frequency argument of :math:`\chi^{*}` and the position of
    :math:`(\chi_0)^{-1}` swapped. It thus reads
    :math:`\tilde{\gamma}_{r;1234}^{q\nu} = \beta \sum_{ab} \sum_{\nu'} \chi^{*;q\nu'\nu}_{r;12ab} (\chi^{q\nu}_{0;ba34})^{-1}`.
    Notice that the sum runs over the *first* frequency argument, whereas only the sum over the last frequency is
    available (see :meth:`FourPoint.invert_and_sum_over_last_vn`). The two are related by the time-reversal symmetry
    :math:`\chi^{*;q\nu\nu'}_{r;1234} = \chi^{*;q\nu'\nu}_{r;4321}` (enforced on the DMFT two-particle Green's
    function via :meth:`LocalFourPoint.symmetrize_v_vp` and inherited by all vertices built from it), which carries
    an orbital reversal along with the frequency swap:
    :math:`\sum_{\nu'} \chi^{*;q\nu'\nu}_{r;12ab} = \sum_{\nu'} \chi^{*;q\nu\nu'}_{r;ba21}`. Hence the last-frequency
    sum enters with the orbital permutation ``"abcd->dcba"`` applied.

    :param gchi_aux_q_r_sum: The frequency-summed auxiliary susceptibility :math:`\sum_{\nu'}\chi^{*;q\nu\nu'}_{r}`.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi_0^q)^{-1}` (core box).
    :return: The right-sided three-leg vertex :math:`\tilde{\gamma}^q_{r}` (``vrg_right``) as a :class:`FourPoint`.
    """
    return (gchi_aux_q_r_sum.permute_orbitals("abcd->dcba") @ gchi0_q_inv).scale(config.sys.beta)


def create_generalized_chi_q_with_shell_correction(
    chi_phys_q_r: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
) -> FourPoint:
    r"""
    Calculates the generalized susceptibility with the shell correction as described by
    Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005; DOI 10.1088/2515-7639/ac7e6d. Eq. A.15. See also Sec. 3.7.2
    in my master's thesis for details.

    :param chi_phys_q_r: The physical susceptibility :math:`\chi^{phys;q}_{r}`.
    :param gchi0_q_full_sum: The frequency-summed bare bubble over the full box.
    :param gchi0_q_core_sum: The frequency-summed bare bubble over the core box.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :return: The shell-corrected physical susceptibility :math:`\chi^{q}_{r}` as a :class:`FourPoint`.
    """
    return (
        (chi_phys_q_r + gchi0_q_full_sum - gchi0_q_core_sum).invert()
        + (u_loc.as_channel(chi_phys_q_r.channel) + v_nonloc.as_channel(chi_phys_q_r.channel))
    ).invert()


def calculate_sigma_dc_kernel(f_dc_loc: LocalFourPoint, gchi0_q: FourPoint, u_loc: LocalInteraction) -> FourPoint:
    r"""
    Returns the double-counting kernel for the self-energy calculation, contracting the local full vertex with the
    momentum-dependent bubble per q-point. For details, see Eq. (4.28) in my master's thesis.

    :param f_dc_loc: The local full vertex :math:`F` used for the double-counting correction.
    :param gchi0_q: The momentum-dependent bare bubble :math:`\chi_0^q`.
    :param u_loc: The bare local interaction :math:`U`.
    :return: The double-counting kernel as a :class:`FourPoint`, cut to the core fermionic box.
    """
    kernel = 1.0 / config.sys.beta**2 * u_loc.permute_orbitals("abcd->adcb") @ gchi0_q

    einsum_str = "abcdwv,dcefwvp->abefwp"
    path, _ = np.einsum_path(einsum_str, kernel.mat[0].copy(), f_dc_loc.mat, optimize="optimal")

    for q in range(kernel.current_shape[0]):
        kernel[q] = np.einsum(einsum_str, kernel[q].copy(), f_dc_loc.mat, optimize=path)

    return kernel.cut_niv(config.box.niv_core)


def calculate_kernel_r_q(
    vrg_q_r: FourPoint, chi_phys_q_r: FourPoint, v_nonloc: Interaction, u_loc: LocalInteraction
) -> FourPoint:
    r"""
    Returns the kernel for the self-energy calculation minus 2/3 times the identity if the channel is the magnetic
    channel (due to the extra factor of :math:`U_{ah21}` in Eq. (4.29) in my master's thesis),

    .. math:: K = \gamma_{r;abcd}^{q\nu} - \gamma_{r;abef}^{q\nu} U^{q}_{r;fehg} \chi_{r;ghcd}^{q}.

    :param vrg_q_r: The momentum-dependent three-leg vertex :math:`\gamma^q_{r}`.
    :param chi_phys_q_r: The (shell-corrected) physical susceptibility :math:`\chi^{phys;q}_{r}`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :param u_loc: The bare local interaction :math:`U`.
    :return: The self-energy kernel :math:`U_r K` as a :class:`FourPoint`.
    """
    u_r = v_nonloc.as_channel(vrg_q_r.channel) + u_loc.as_channel(vrg_q_r.channel)
    kernel = vrg_q_r - vrg_q_r @ u_r @ chi_phys_q_r

    if vrg_q_r.channel == SpinChannel.MAGN:
        # subtract 2/3 directly on the compound-identity positions (o1 == o4, o2 == o3, every frequency) instead of
        # materializing a full kernel-sized identity block via FourPoint.identity_like
        orb = np.arange(kernel.n_bands)
        kernel.mat[:, orb[:, None], orb[None, :], orb[None, :], orb[:, None], ...] -= 2.0 / 3.0

    return u_r @ kernel


def calculate_sigma_from_kernel(kernel: FourPoint, giwk: GreensFunction, my_full_q_list: np.ndarray) -> SelfEnergy:
    r"""
    Returns :math:`\Sigma_{12}^{k\nu} = -\frac{1}{2\beta N_q} \sum_{q\omega} \sum_{abcd} U^q_{r;a1bc} K_{r;cb2d}^{q\omega\nu} G_{ad}^{\nu-\omega}`.
    For very large momentum grids, this function is the slowest part compared to the rest of the code due to the
    repeated loops. Potential speed-ups could be achieved by batching the q-points or using numba.

    Currently unused: the pipeline always runs the two-pass FFT contraction (see
    :func:`calculate_sigma_from_kernel_fft_cpu`), since this q-loop variant restores the full bosonic range on the
    kernel and therefore peaks *higher* in memory. Kept (with its cpu/gpu/auto siblings) for reference; the
    variants' mutual parity is unit-tested.

    :param kernel: The self-energy kernel :math:`K` (full BZ, scattered across ranks).
    :param giwk: The momentum-dependent :class:`GreensFunction`.
    :param my_full_q_list: Array of integer q-point index triplets handled by this rank.
    :return: The rank-local contribution to the non-local :class:`SelfEnergy` (compressed q, full niv range).
    """
    mat = np.zeros(
        (*config.lattice.k_grid.nk, config.sys.n_bands, config.sys.n_bands, config.box.niv_core),
        dtype=kernel.mat.dtype,
    )

    kernel = kernel.to_full_niw_range()
    wn = MFHelper.wn(config.box.niw_core)
    path = np.einsum_path("aijdv,xyzadv->xyzijv", kernel[0, ..., 0, config.box.niv_core :], mat, optimize=True)[0]

    for idx_q, q in enumerate(my_full_q_list):
        shifted_mat = np.roll(giwk.mat, [-i for i in q], axis=(0, 1, 2))
        for idx_w, wn_i in enumerate(wn):
            g_qk = shifted_mat[..., giwk.niv - wn_i : giwk.niv + config.box.niv_core - wn_i]
            k_slice = kernel[idx_q, ..., idx_w, config.box.niv_core :]
            mat += np.einsum("aijdv,xyzadv->xyzijv", k_slice, g_qk, optimize=path)

    mat *= -0.5 / config.sys.beta / config.lattice.k_grid.nk_tot
    return SelfEnergy(mat, config.lattice.nk, False, beta=config.sys.beta).compress_q_dimension().to_full_niv_range()


def calculate_sigma_from_kernel_cpu(
    kernel: FourPoint,
    giwk: GreensFunction,
    my_full_q_list: np.ndarray,
) -> SelfEnergy:
    r"""
    Returns :math:`\Sigma_{12}^{k\nu} = -\frac{1}{2\beta N_q} \sum_{q\omega} \sum_{abcd} U^q_{r;a1bc} K_{r;cb2d}^{q\omega\nu} G_{ad}^{\nu-\omega}`.
    For very large momentum grids, this function is the slowest part compared to the rest of the code due to the
    repeated loops. There is no real way to speed it up further without leveraging GPUs or other hardware accelerators.
    This is the CPU implementation (Fortran-ordered buffers, preallocated accumulator). Currently unused, see
    :func:`calculate_sigma_from_kernel`.

    :param kernel: The self-energy kernel :math:`K` (full BZ, scattered across ranks).
    :param giwk: The momentum-dependent :class:`GreensFunction`.
    :param my_full_q_list: Array of integer q-point index triplets handled by this rank.
    :return: The rank-local contribution to the non-local :class:`SelfEnergy` (compressed q, full niv range).
    """
    nkx, nky, nkz = config.lattice.k_grid.nk
    nb = config.sys.n_bands
    niv_core = config.box.niv_core

    mat = np.zeros((nkx, nky, nkz, nb, nb, niv_core), dtype=kernel.mat.dtype)
    wn = MFHelper.wn(config.box.niw_core)

    giwk_mat = np.asfortranarray(giwk.mat)
    kernel = np.asfortranarray(kernel.to_full_niw_range().mat[..., niv_core:])

    kxs, kys, kzs = np.arange(nkx), np.arange(nky), np.arange(nkz)
    kx_indices = [((kxs + q[0]) % nkx) for q in my_full_q_list]
    ky_indices = [((kys + q[1]) % nky) for q in my_full_q_list]
    kz_indices = [((kzs + q[2]) % nkz) for q in my_full_q_list]

    acc = np.empty((nkx, nky, nkz, nb, nb, niv_core), dtype=mat.dtype)

    for iq in range(len(my_full_q_list)):
        g_q_view = giwk_mat[
            kx_indices[iq][:, None, None], ky_indices[iq][None, :, None], kz_indices[iq][None, None, :], ...
        ]

        for iw, w in enumerate(wn):
            g_slice = g_q_view[..., giwk.niv - w : giwk.niv + niv_core - w]
            k_slice = kernel[iq, ..., iw, :]
            np.einsum("xyzadv,aijdv->xyzijv", g_slice, k_slice, out=acc, optimize=True)
            np.add(mat, acc, out=mat)

    mat *= -0.5 / config.sys.beta / config.lattice.k_grid.nk_tot
    return (
        SelfEnergy(np.ascontiguousarray(mat), config.lattice.nk, False, beta=config.sys.beta)
        .compress_q_dimension()
        .to_full_niv_range()
    )


def calculate_sigma_from_kernel_gpu(
    kernel: FourPoint,
    giwk: GreensFunction,
    my_full_q_list: np.ndarray,
) -> SelfEnergy:
    r"""
    Returns :math:`\Sigma_{12}^{k\nu} = -\frac{1}{2\beta N_q} \sum_{q\omega} \sum_{abcd} U^q_{r;a1bc} K_{r;cb2d}^{q\omega\nu} G_{ad}^{\nu-\omega}`.
    For very large momentum grids, this function is the slowest part compared to the rest of the code due to the
    repeated loops. This is the GPU implementation using CuPy. Currently unused, see
    :func:`calculate_sigma_from_kernel`.

    :param kernel: The self-energy kernel :math:`K` (full BZ, scattered across ranks).
    :param giwk: The momentum-dependent :class:`GreensFunction`.
    :param my_full_q_list: Array of integer q-point index triplets handled by this rank.
    :return: The rank-local contribution to the non-local :class:`SelfEnergy` (compressed q, full niv range).
    """
    import cupy as cp

    nkx, nky, nkz = config.lattice.k_grid.nk
    nb = config.sys.n_bands
    niv_core = config.box.niv_core

    mat_gpu = cp.zeros((nkx, nky, nkz, nb, nb, niv_core), dtype=kernel.mat.dtype, order="F")
    wn = MFHelper.wn(config.box.niw_core)

    giwk_mat = cp.asarray(giwk.mat, order="F")
    kernel = cp.asarray(kernel.to_full_niw_range().mat, order="F")[..., niv_core:]

    kxs, kys, kzs = cp.arange(nkx), cp.arange(nky), cp.arange(nkz)
    kx_indices = [((kxs + q[0]) % nkx) for q in my_full_q_list]
    ky_indices = [((kys + q[1]) % nky) for q in my_full_q_list]
    kz_indices = [((kzs + q[2]) % nkz) for q in my_full_q_list]

    for iq in range(len(my_full_q_list)):
        g_q_view = giwk_mat[
            kx_indices[iq][:, None, None], ky_indices[iq][None, :, None], kz_indices[iq][None, None, :], ...
        ]

        for iw, w in enumerate(wn):
            g_slice = g_q_view[..., giwk.niv - w : giwk.niv + niv_core - w]
            k_slice = kernel[iq, ..., iw, :]
            mat_gpu += cp.einsum("xyzadv,aijdv->xyzijv", g_slice, k_slice, optimize=True)

    mat_gpu *= -0.5 / config.sys.beta / config.lattice.k_grid.nk_tot
    return (
        SelfEnergy(np.ascontiguousarray(cp.asnumpy(mat_gpu)), config.lattice.nk, False, beta=config.sys.beta)
        .compress_q_dimension()
        .to_full_niv_range()
    )


def calculate_sigma_from_kernel_auto(
    mpi_distributor: MpiDistributor, kernel: FourPoint, giwk: GreensFunction, my_full_q_list: np.ndarray
) -> SelfEnergy:
    r"""
    Dispatches the q-loop self-energy contraction to the GPU (:func:`calculate_sigma_from_kernel_gpu`) when CuPy and
    a usable CUDA device are available (one GPU per MPI rank, round-robin), otherwise falls back to the CPU
    implementation (:func:`calculate_sigma_from_kernel_cpu`). Currently unused, see
    :func:`calculate_sigma_from_kernel`.

    :param mpi_distributor: MPI distributor used to choose the per-rank GPU (see :class:`MpiDistributor`).
    :param kernel: The self-energy kernel :math:`K` (full BZ, scattered across ranks).
    :param giwk: The momentum-dependent :class:`GreensFunction`.
    :param my_full_q_list: Array of integer q-point index triplets handled by this rank.
    :return: The rank-local contribution to the non-local :class:`SelfEnergy`.
    """
    logger = config.logger

    cp = None
    try:
        import cupy as cp
    except ImportError:
        pass  # CuPy not installed -> CPU

    n_gpus = 0
    if cp is not None:
        try:
            n_gpus = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError:
            n_gpus = 0  # no usable CUDA driver/device -> CPU

    if n_gpus > 0 and cp.cuda.is_available():
        logger.info(f"CuPy detected {n_gpus} GPU(s). Using GPU acceleration for self-energy calculation.")

        gpu_id = mpi_distributor.my_rank % n_gpus
        cp.cuda.Device(gpu_id).use()
        return calculate_sigma_from_kernel_gpu(kernel, giwk, my_full_q_list)

    return calculate_sigma_from_kernel_cpu(kernel, giwk, my_full_q_list)


def calculate_sigma_from_kernel_fft_cpu(
    mpi_dist: MpiDistributor,
    kernel: FourPoint,
    giwk: GreensFunction,
    niw_index_w_pairs: list[tuple[int, int]],
    node_comm: "MPI.Comm | None" = None,
) -> SelfEnergy:
    r"""
    Computes the self-energy using distributed FFTs (CPU). Replaces the q-loop with a real-space pointwise
    multiplication: both the Green's function and the kernel are FFT'd to real space (the kernel to :math:`-R` via the
    conjugate trick), contracted pointwise per rank-local R-slice, and accumulated. Returns :math:`\Sigma` in R-space,
    positive-:math:`\nu` half only; the caller must ifft over :math:`(k_x, k_y, k_z)` and then call
    :meth:`SelfEnergy.to_full_niv_range` before use.

    This contracts a **single** bosonic-frequency half (positive-w *or* negative-w) so the full-BZ kernel is never
    materialized over the full niw range (see :meth:`LocalNPoint.to_negative_niw_range`): the kernel is consumed in
    whatever niw orientation it is handed (no internal ``to_full_niw_range``), and ``niw_index_w_pairs`` selects which
    kernel w-slices to contract and how to shift the Green's function.

    :param mpi_dist: MPI distributor providing the communicator and R-space pencil decomposition.
    :param kernel: The self-energy kernel :math:`K` for one niw half (full BZ): the positive half (``w >= 0``) or the
        negative block from :meth:`LocalNPoint.to_negative_niw_range` (``w = 0, -1, ..., -niw``).
    :param giwk: The momentum-dependent :class:`GreensFunction`.
    :param niw_index_w_pairs: The ``(kernel_w_index, w)`` pairs to contract. The positive pass passes
        ``[(i, i) for i in range(niw + 1)]`` (``w = 0..+niw``), the negative pass
        ``[(i, -i) for i in range(1, niw + 1)]`` (``w = -1..-niw``, skipping the ``w = 0`` duplicate).
    :param node_comm: Optional node-local communicator; when given (and the shared-giwk window is enabled),
        the R-space Green's function is built once per node in a shared-memory window instead of once per rank.
    :return: The rank-local R-space :class:`SelfEnergy` (compressed q, half niv range, moments not fitted).
    """
    comm = mpi_dist.comm
    rank = comm.Get_rank()
    size = comm.Get_size()
    nkx, nky, nkz = config.lattice.k_grid.nk
    nk_tot = config.lattice.k_grid.nk_tot
    nb = config.sys.n_bands
    niv_core = config.box.niv_core
    beta = config.sys.beta

    # G(k) -> F[G](R), forward FFT: built once per node in a shared window when available, else privately per rank.
    # Each rank keeps only its R-pencil slice (a copy), so the R-space object frees before the kernel FFT.
    if node_comm is not None and config.memory.use_shared_memory_common_obj:
        g_r_mat, g_r_win = mpi_utils.build_node_shared_array(node_comm, lambda: giwk.fft().mat)
    else:
        g_r_mat, g_r_win = giwk.fft().mat, None
    my_r_indices = mpi_utils.get_pencil_indices(rank, size, (nkx, nky, nkz), "flat")
    g_r_local = g_r_mat.reshape(nk_tot, nb, nb, -1)[my_r_indices]
    del g_r_mat
    mpi_utils.free_shared_window(g_r_win, node_comm)

    # K(q) -> F[K](-R) via the conjugate trick: conj, fft, conj. The kernel is already a single niw half (positive
    # half, or the negative block), so there is no to_full_niw_range here -- the full-niw kernel is never built.
    kernel = kernel.to_half_niv_range()
    kernel.mat = np.conj(kernel.mat)
    kernel = mpi_utils.execute_distributed_fft(kernel, comm)
    kernel.mat = np.conj(kernel.mat)

    # Local R-space contraction; each rank owns a slice of R-points
    n_r_local = kernel.mat.shape[0]
    mat = np.zeros((n_r_local, nb, nb, niv_core), dtype=kernel.mat.dtype)
    acc = np.empty_like(mat)

    path = None
    for idx, w in niw_index_w_pairs:
        g_slice = g_r_local[..., giwk.niv - w : giwk.niv + niv_core - w]
        k_slice = kernel.mat[..., idx, :]
        if path is None:  # slice shapes are identical across the loop -> compute the contraction path once
            path = np.einsum_path("Radv,Raijdv->Rijv", g_slice, k_slice, optimize="optimal")[0]
        np.einsum("Radv,Raijdv->Rijv", g_slice, k_slice, out=acc, optimize=path)
        np.add(mat, acc, out=mat)

    mat *= -0.5 / beta / nk_tot
    return SelfEnergy(
        np.ascontiguousarray(mat),
        config.lattice.nk,
        full_niv_range=False,
        has_compressed_q_dimension=True,
        calc_smom=False,
        beta=config.sys.beta,
    )


def calculate_sigma_from_kernel_fft_gpu(
    mpi_dist: MpiDistributor,
    kernel: FourPoint,
    giwk: GreensFunction,
    niw_index_w_pairs: list[tuple[int, int]],
    node_comm: "MPI.Comm | None" = None,
) -> SelfEnergy:
    r"""
    Computes the self-energy using distributed FFTs, running on the GPU (CuPy). Same algorithm as
    :func:`calculate_sigma_from_kernel_fft_cpu` (including the single-niw-half / ``niw_index_w_pairs`` contraction).
    Returns :math:`\Sigma` in R-space, positive-:math:`\nu` half only; the caller must ifft over
    :math:`(k_x, k_y, k_z)` and then call :meth:`SelfEnergy.to_full_niv_range` before use.

    :param mpi_dist: MPI distributor providing the communicator and R-space pencil decomposition.
    :param kernel: The self-energy kernel :math:`K` for one niw half (full BZ): the positive half or the negative
        block from :meth:`LocalNPoint.to_negative_niw_range`.
    :param giwk: The momentum-dependent :class:`GreensFunction`.
    :param niw_index_w_pairs: The ``(kernel_w_index, w)`` pairs to contract (see
        :func:`calculate_sigma_from_kernel_fft_cpu`).
    :param node_comm: Optional node-local communicator; when given (and the shared-giwk window is enabled),
        the R-space Green's function is built once per node in a shared-memory window instead of once per rank.
    :return: The rank-local R-space :class:`SelfEnergy` (compressed q, half niv range, moments not fitted).
    """
    import cupy as cp

    comm = mpi_dist.comm
    rank = comm.Get_rank()
    size = comm.Get_size()
    nkx, nky, nkz = config.lattice.k_grid.nk
    nk_tot = config.lattice.k_grid.nk_tot
    nb = config.sys.n_bands
    niv_core = config.box.niv_core
    beta = config.sys.beta

    # G(k) -> F[G](R), forward FFT: built once per node in a (host-side) shared window when available, else
    # privately per rank; only the rank's R-pencil slice is uploaded to the device, then the host copy is released.
    if node_comm is not None and config.memory.use_shared_memory_common_obj:
        g_r_mat, g_r_win = mpi_utils.build_node_shared_array(node_comm, lambda: giwk.fft().mat)
    else:
        g_r_mat, g_r_win = giwk.fft().mat, None
    my_r_indices = mpi_utils.get_pencil_indices(rank, size, (nkx, nky, nkz), "flat")
    g_r_local = cp.asarray(g_r_mat.reshape(nk_tot, nb, nb, -1)[my_r_indices])
    del g_r_mat
    mpi_utils.free_shared_window(g_r_win, node_comm)

    # K(q) -> F[K](-R) via the conjugate trick: conj, fft, conj. The kernel is already a single niw half, so there is
    # no to_full_niw_range here -- the full-niw kernel is never built.
    kernel = kernel.to_half_niv_range()
    kernel.mat = np.conj(kernel.mat)
    kernel = mpi_utils.execute_distributed_fft(kernel, comm)
    kernel.mat = cp.conj(cp.asarray(kernel.mat))

    # Local R-space contraction; each rank owns a slice of R-points
    n_r_local = kernel.mat.shape[0]
    mat = cp.zeros((n_r_local, nb, nb, niv_core), dtype=kernel.mat.dtype)

    for idx, w in niw_index_w_pairs:
        g_slice = g_r_local[..., giwk.niv - w : giwk.niv + niv_core - w]
        k_slice = kernel.mat[..., idx, :]
        mat += cp.einsum("Radv,Raijdv->Rijv", g_slice, k_slice, optimize=True)

    mat *= -0.5 / beta / nk_tot
    return SelfEnergy(
        np.ascontiguousarray(cp.asnumpy(mat)),
        config.lattice.nk,
        full_niv_range=False,
        has_compressed_q_dimension=True,
        calc_smom=False,
        beta=config.sys.beta,
    )


def select_sigma_fft_device(mpi_distributor: MpiDistributor) -> bool:
    """
    Detects whether CuPy and a usable CUDA device are available (one GPU per MPI rank, round-robin), selects this
    rank's GPU and logs the decision **once**. Intended to be called a single time before the positive- and
    negative-w FFT passes so the GPU-detection message is not emitted per pass.

    :param mpi_distributor: MPI distributor providing the per-rank GPU choice.
    :return: True if the GPU implementation should be used, False to fall back to the CPU implementation.
    """
    try:
        import cupy as cp
    except ImportError:
        return False  # CuPy not installed -> CPU

    try:
        n_gpus = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        n_gpus = 0  # no usable CUDA driver/device -> CPU

    if n_gpus > 0 and cp.cuda.is_available():
        config.logger.info(f"CuPy detected {n_gpus} GPU(s). Using GPU acceleration for self-energy calculation.")
        cp.cuda.Device(mpi_distributor.my_rank % n_gpus).use()
        return True

    return False


def calculate_sigma_from_kernel_fft(
    mpi_dist: MpiDistributor,
    kernel: FourPoint,
    giwk: GreensFunction,
    niw_index_w_pairs: list[tuple[int, int]],
    use_gpu: bool,
    node_comm: "MPI.Comm | None" = None,
) -> SelfEnergy:
    r"""
    Dispatches one bosonic-frequency FFT pass to the GPU (:func:`calculate_sigma_from_kernel_fft_gpu`) or CPU
    (:func:`calculate_sigma_from_kernel_fft_cpu`) implementation according to ``use_gpu`` (decided once by
    :func:`select_sigma_fft_device`, so no per-pass GPU-detection logging).

    :param mpi_dist: MPI distributor providing the communicator and R-space pencil decomposition.
    :param kernel: The self-energy kernel :math:`K` for one niw half (full BZ).
    :param giwk: The momentum-dependent :class:`GreensFunction`.
    :param niw_index_w_pairs: The ``(kernel_w_index, w)`` pairs to contract (see
        :func:`calculate_sigma_from_kernel_fft_cpu`).
    :param use_gpu: Whether to run the GPU implementation (as decided by :func:`select_sigma_fft_device`).
    :param node_comm: Optional node-local communicator; when given (and the shared-giwk window is enabled),
        the R-space Green's function is built once per node in a shared-memory window instead of once per rank.
    :return: The rank-local R-space :class:`SelfEnergy`.
    """
    if use_gpu:
        return calculate_sigma_from_kernel_fft_gpu(mpi_dist, kernel, giwk, niw_index_w_pairs, node_comm)
    return calculate_sigma_from_kernel_fft_cpu(mpi_dist, kernel, giwk, niw_index_w_pairs, node_comm)


def _run_fft_sde_pass(
    kernel_src: FourPoint,
    mpi_dist_irrk: MpiDistributor,
    mpi_dist_fullbz: MpiDistributor,
    giwk_full: GreensFunction,
    niw_index_w_pairs: list[tuple[int, int]],
    use_gpu: bool,
    negative_w: bool,
    node_comm: "MPI.Comm | None" = None,
) -> SelfEnergy:
    r"""
    Runs one bosonic-frequency FFT self-energy pass: maps the (small) irreducible-BZ kernel to the full BZ
    (consuming ``kernel_src``), optionally builds its time-reversed negative-:math:`\omega` block, contracts the
    requested ``niw_index_w_pairs`` via :func:`calculate_sigma_from_kernel_fft`, and frees the full-BZ kernel. Both
    passes of :func:`~dgamore.nonlocal_sde.calculate_self_energy_q` go through this helper: the caller hands the positive pass a
    :meth:`~dgamore.n_point_base.IHaveMat.copy` of the irreducible kernel (so it survives for the negative pass) and
    the negative pass the original (which this consumes), so only a single full-BZ niw half is ever resident.

    :param kernel_src: The irreducible-BZ kernel for this pass; consumed by the full-BZ map (mutated or replaced).
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points.
    :param mpi_dist_fullbz: MPI distributor over the full BZ q-points.
    :param giwk_full: The momentum-dependent :class:`GreensFunction`.
    :param niw_index_w_pairs: The ``(kernel_w_index, w)`` pairs to contract (see
        :func:`calculate_sigma_from_kernel_fft_cpu`).
    :param use_gpu: Whether to run the GPU implementation (as decided by :func:`select_sigma_fft_device`).
    :param negative_w: If True, build the negative-:math:`\omega` block via
        :meth:`LocalNPoint.to_negative_niw_range` before contracting (the negative pass) and trim the kernel peak
        back to the OS on free; if False, contract the mapped kernel directly (the positive pass).
    :param node_comm: Optional node-local communicator; when given (and the shared-giwk window is enabled),
        the R-space Green's function is built once per node in a shared-memory window instead of once per rank.
    :return: The rank-local R-space :class:`SelfEnergy` of this pass.
    """
    # the distributed p2p exchange always returns a fresh full-BZ object, so the irreducible source is freed
    kernel_full = mpi_utils.exchange_and_map_irrbz_fullbz(kernel_src, mpi_dist_irrk, mpi_dist_fullbz)
    kernel_src.free()

    if negative_w:
        kernel_neg = kernel_full.to_negative_niw_range()
        kernel_full.free()  # release the full-BZ positive copy as soon as the negative block is built
        kernel_full = kernel_neg

    sigma = calculate_sigma_from_kernel_fft(
        mpi_dist_irrk, kernel_full, giwk_full, niw_index_w_pairs, use_gpu, node_comm
    )
    kernel_full.free(trim=negative_w)  # coarse per-iteration trim on the last (negative) pass
    return sigma

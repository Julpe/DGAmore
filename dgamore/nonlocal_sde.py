# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Non-local ladder DGA step - the parallel-heavy core of the code. Starting from the local irreducible vertex
:math:`\Gamma_{r}` and the bare interaction, the functions here build, per momentum :math:`q` and spin channel,
the bubble :math:`\chi_0^q`, the auxiliary susceptibility :math:`\chi^{*;q}_{r}`, the three-leg vertex
:math:`\gamma^q_{r}`, the physical susceptibility :math:`\chi^q_{r}` (with shell and optional
:math:`\lambda`-correction) and the self-energy kernel, then contract the kernel with the Green's function to get
the momentum-dependent self-energy :math:`\Sigma(k, \nu)`. Several CPU/GPU/FFT variants of the heavy contractions
are provided, distributed over MPI ranks. The whole thing is wrapped in a self-consistency loop with chemical-
potential adjustment and self-energy mixing (linear / Pulay / Anderson). Equation numbers refer to the author's
master's thesis (Chapters 3 & 4).
"""

import glob
import os
import re
from contextlib import contextmanager

import mpi4py.MPI as MPI
import numpy as np
from scipy import optimize as opt

import dgamore.config as config
import dgamore.jacobian_stabilization as jstab
import dgamore.mpi_utils as mpi_utils
from dgamore.brillouin_zone import KGrid
from dgamore.bubble_gen import BubbleGenerator
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction, update_mu
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.local_four_point import LocalFourPoint
from dgamore.lambda_ops import LambdaAnnealer, LambdaCorrection, MultiOrbitalLambdaCorrection
from dgamore.matsubara_frequencies import MFHelper
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel
from dgamore.self_energy import SelfEnergy

# Trigger/watchdog tuning of the physical-solution stabilizer (see _update_stabilizer_probe/_watchdog). Deliberately
# NOT user-configurable: the stabilizer must only engage when plain iteration cannot reach the physical solution.
_STAB_GROWTH_FACTOR = 3.0  # residual this far above its best counts as divergence
_STAB_FAR_RESIDUAL_FACTOR = 100.0  # a plateau only counts when still this far above epsilon
_STAB_PLATEAU_WINDOW_FACTOR = 3  # plateau window, in units of the probe window
_STAB_WATCH_WINDOW_FACTOR = 3  # post-arming do-no-harm window, in units of the probe window


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
    nq_tot = np.prod(config.lattice.nq)

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

    with the matrix to invert assembled in a single block by :func:`_assemble_bse_matrix`.

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
    irrk_q_list = config.lattice.q_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrq.my_slice]
    chi_r_q_sum_mat = np.zeros_like(gchi0_q_inv.mat)

    u_r = v_nonloc.as_channel(gamma_r.channel) + u_loc.as_channel(gamma_r.channel)
    for idx in range(len(my_irr_q_list)):
        chi_r_q_sum_mat[idx] = (
            create_inverse_auxiliary_chi_r_q(gamma_r, gchi0_q_inv.filter_q_index(idx), u_r.filter_q_index(idx))
            .invert_and_sum_over_last_vn(config.sys.beta)
            .mat
        )
    return FourPoint(chi_r_q_sum_mat, gamma_r.channel, config.lattice.nq, 1, 1, False, has_compressed_q_dimension=True)


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
    irrk_q_list = config.lattice.q_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrq.my_slice]
    chi_r_q_sum_mat = np.zeros_like(gchi0_q_inv.mat)

    u_r = v_nonloc.as_channel(gamma_r.channel) + u_loc.as_channel(gamma_r.channel)
    for idx in range(len(my_irr_q_list)):
        chi_r_q_sum_mat[idx] = (
            create_inverse_auxiliary_chi_r_q(gamma_r, gchi0_q_inv.filter_q_index(idx), u_r.filter_q_index(idx))
            .invert_and_sum_over_last_vn_v2(config.sys.beta)
            .mat
        )
    return FourPoint(chi_r_q_sum_mat, gamma_r.channel, config.lattice.nq, 1, 1, False, has_compressed_q_dimension=True)


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


def restrict_chi_phys_to_positive_eigenvalues(chi_phys_q_r: FourPoint, floor: float = 1e-4) -> tuple[FourPoint, int]:
    r"""
    Regularizes the physical susceptibility: for every momentum and bosonic frequency
    the eigenvalues of the Hermitian part of the inverse compound matrix :math:`(\chi^{q\omega}_{r;1234})^{-1}`
    are floored at :math:`+\text{floor}` (the skew-Hermitian part is kept), and the result is inverted back. A
    negative eigenvalue of the inverse marks a crossed pole of the Bethe-Salpeter equation (an unphysical branch
    of the ladder, e.g. the high-temperature charge-channel instability); flooring it pins the corresponding
    susceptibility eigenvalue at :math:`1/\text{floor}` while all healthy eigenpairs - including legitimately
    negative off-diagonal matrix elements - pass through unchanged. For a single band the compound block is a
    scalar and this reduces to the plain clamp of negative inverse values.

    :param chi_phys_q_r: The physical susceptibility :math:`\chi^{q\omega}_{r;1234}` (no fermionic frequency
        dimensions).
    :param floor: Lower bound imposed on the eigenvalues of the inverse susceptibility.
    :return: The tuple ``(chi_restricted, n_floored)`` of the restricted susceptibility as a :class:`FourPoint`
        and the number of floored eigenvalues (a per-iteration diagnostic: if it decays to zero during the
        restricted phase of the self-consistency, releasing the restriction is safe).
    """
    chi_inv = chi_phys_q_r.invert().to_compound_indices()
    herm = 0.5 * (chi_inv.mat + np.conj(np.swapaxes(chi_inv.mat, -1, -2)))
    chi_inv.mat -= herm
    eigs, vecs = np.linalg.eigh(herm)
    n_floored = int((eigs < floor).sum())
    chi_inv.mat += np.einsum(
        "...ab,...b,...cb->...ac", vecs, np.maximum(eigs, floor).astype(eigs.dtype), np.conj(vecs), optimize=True
    )
    return chi_inv.invert(copy=False), n_floored


def _effective_epsilon(annealer: "LambdaAnnealer | None" = None) -> float:
    """
    Returns the effective self-energy convergence threshold of the self-consistency loop: ten times the configured
    epsilon while the susceptibility restriction, the per-iteration lambda correction, or the lambda-annealing
    scaffold is active (those phases are only scaffolds for the subsequent pure phase, so full precision there is
    wasted iterations), and the plain epsilon otherwise. The one-shot lambda correction never relaxes the threshold
    (it runs a single iteration).

    :param annealer: The active :class:`LambdaAnnealer`, or ``None`` when annealing is off.
    :return: The effective convergence threshold.
    """
    relaxed = (
        config.stabilization.use_chi_phys_restriction
        or config.stabilization.use_lambda_correction
        or (annealer is not None and annealer.active)
    )
    return (10.0 if relaxed else 1.0) * config.self_consistency.epsilon


def min_static_compound_eigenvalue(chi_phys_q_r: FourPoint) -> float:
    r"""
    Returns the smallest eigenvalue of the Hermitian part of the static compound blocks
    :math:`\chi^{q(\omega=0)}_{r;1234}` over all rank-local momenta. A physical static susceptibility is positive
    semi-definite per momentum, so a significantly negative value flags that the ladder sits on an unphysical
    (past-pole) branch. Expects the object with a compressed momentum dimension and no fermionic frequency
    dimensions.

    :param chi_phys_q_r: The physical susceptibility :math:`\chi^{q\omega}_{r;1234}`.
    :return: The minimum eigenvalue as a float.
    """
    w0 = chi_phys_q_r.niw if chi_phys_q_r.full_niw_range else 0
    n = chi_phys_q_r.n_bands**2
    static = chi_phys_q_r.mat[..., w0].transpose(0, 1, 2, 4, 3).reshape(-1, n, n)
    return float(np.linalg.eigvalsh(0.5 * (static + np.conj(np.swapaxes(static, -1, -2)))).min())


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


def perform_ornstein_zernike_fit(chi_phys_q_r: FourPoint) -> None:
    r"""
    Fits the static (:math:`\omega = 0`) physical susceptibility to an Ornstein-Zernike form
    :math:`\chi(q) = A / (\xi^{-2} + (q - q_0)^2)` around the antiferromagnetic wave vector
    :math:`q_0 = (\pi, \pi, 0)`, per orbital combination, and writes the amplitude :math:`A` and correlation length
    :math:`\xi` to ``oz_coeff.txt``. Non-converging fits are flagged with ``[-1, -1]``.

    :param chi_phys_q_r: The momentum-dependent physical susceptibility :math:`\chi^{q}_{r}` (irreducible BZ).
    :return: None.
    """

    def oz_spin_w0(q_grid: KGrid, a: float, xi: float):
        r"""
        Evaluates the Ornstein-Zernike model on the full BZ grid, flattened to match the fit data.

        :param q_grid: The :class:`KGrid` providing the momentum coordinates.
        :param a: The amplitude :math:`A`.
        :param xi: The correlation length :math:`\xi`.
        :return: The flattened model susceptibility over the BZ grid.
        """
        qx = qy = np.pi
        qz = 0
        oz = a / (
            xi ** (-2)
            + (q_grid.kx[:, None, None] - qx) ** 2
            + (q_grid.ky[None, :, None] - qy) ** 2
            + (q_grid.kz[None, None, :] - qz) ** 2
        )
        return oz.flatten()

    def fit_oz_spin(q_grid: KGrid, mat: np.ndarray):
        """
        Least-squares fits the Ornstein-Zernike model to one orbital slice of the susceptibility.

        :param q_grid: The :class:`KGrid` providing the momentum coordinates.
        :param mat: The flattened susceptibility slice to fit.
        :return: The fitted ``(A, xi)`` coefficients.
        """
        initial_guess = (mat.max(), 2.0)
        return opt.curve_fit(oz_spin_w0, q_grid, mat, p0=initial_guess)[0]

    chi = chi_phys_q_r.copy()
    chi_mat = chi.map_to_full_bz(config.lattice.q_grid).to_half_niw_range().take_first_wn().mat.real
    orb_shape = (config.sys.n_bands,) * 4
    oz_coeffs = np.zeros(orb_shape + (2,), dtype=float)
    failed_orbitals = []

    for idx in np.ndindex(orb_shape):
        mat_slice = chi_mat[..., idx[0], idx[1], idx[2], idx[3]].flatten()
        try:
            coeffs = fit_oz_spin(config.lattice.q_grid, mat_slice) if not np.all(mat_slice == 0) else [0.0, 0.0]
        except (ValueError, RuntimeError, opt.OptimizeWarning):
            failed_orbitals.append(idx)
            coeffs = [-1.0, -1.0]
        oz_coeffs[idx] = coeffs

    if failed_orbitals:
        one_based = [tuple(o + 1 for o in idx) for idx in failed_orbitals]
        config.logger.warning(
            f"OZ fit did not converge for {len(failed_orbitals)} orbital combination(s): "
            f"{one_based}. Using [-1, -1]."
        )

    rows = []
    for idx in np.ndindex(orb_shape):
        rows.append([*idx, *oz_coeffs[idx]])

    data_to_save = np.array(rows, dtype=float)
    path = os.path.join(config.output.output_path, f"oz_coeff.txt")
    np.savetxt(path, data_to_save, delimiter=",", fmt="%d %d %d %d %.9f %.9f", header="o1 o2 o3 o4 A xi")


def calculate_and_save_chi_q_r_rpa(
    gchi0_q_core_inv: FourPoint, u_loc: LocalInteraction, v_nonloc: Interaction, mpi_dist_irrk: MpiDistributor
):
    r"""
    Calculates and saves the RPA susceptibility (for both density and magnetic channels) from the DMFT Green's
    functions, :math:`\chi_{d/m;\mathrm{RPA}} = \chi_0 (1 + U_{d/m}\chi_0)^{-1} = (\chi_0^{-1} + U_{d/m})^{-1}`. The
    result is gathered to rank 0 and written to file.

    :param gchi0_q_core_inv: The inverse bare bubble :math:`(\chi_0^q)^{-1}` (core box).
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :return: None.
    """
    for channel in [SpinChannel.DENS, SpinChannel.MAGN]:
        u_r = u_loc.as_channel(channel) + v_nonloc.as_channel(channel)
        chi_rpa_q_r = (gchi0_q_core_inv + u_r).invert(False).sum_over_all_vn(config.sys.beta)
        chi_rpa_q_r.mat = mpi_dist_irrk.gather(chi_rpa_q_r.mat)

        if mpi_dist_irrk.my_rank == 0:
            chi_rpa_q_r.save(name=f"chi_rpa_q_{channel.value}", output_dir=config.output.output_path)

        chi_rpa_q_r.free()
        config.logger.info(f"Calculated RPA susceptibility ({channel.value}).")


def _select_and_apply_lambda_correction(chi_phys_q_r: FourPoint, quiet: bool) -> FourPoint:
    r"""
    Applies the configured lambda correction to the (rank-0 gathered) physical susceptibility and returns it. The
    correction runs when either the one-shot ``config.lambda_correction.perform_lambda_correction`` or the
    per-iteration ``config.stabilization.use_lambda_correction`` is enabled and is dispatched by the band count:
    single-band input uses the scalar Moriya correction, multi-band input the multi-orbital matrix correction (the
    dispatch is logged once at setup). If neither flag is enabled the susceptibility is returned unchanged.

    :param chi_phys_q_r: The rank-0 gathered physical susceptibility :math:`\chi^{q}_{r}` in the irreducible BZ.
    :param quiet: Forwarded to the correction (suppresses the lambda text-file write during stabilizer probes).
    :return: The (possibly corrected) physical susceptibility.
    """
    if config.lambda_correction.perform_lambda_correction or config.stabilization.use_lambda_correction:
        if config.sys.n_bands == 1:
            return LambdaCorrection.perform(chi_phys_q_r, quiet=quiet)
        return MultiOrbitalLambdaCorrection.perform(chi_phys_q_r, quiet=quiet)
    return chi_phys_q_r


def calculate_sigma_kernel_r_q(
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    mpi_dist_irrq: MpiDistributor,
    quiet: bool = False,
    annealer: "LambdaAnnealer | None" = None,
) -> FourPoint:
    r"""
    Returns the kernel for the self-energy calculation in a specific spin channel. Calculates the auxiliary
    susceptibility, the three-leg vertex and the physical susceptibility with shell correction. Also performs a
    :math:`\lambda`-correction on the physical susceptibility if specified in the config (dispatched by the band
    count). Saves the physical susceptibility (and, if Eliashberg is enabled, the intermediate vertices) to file.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}`.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi_0^q)^{-1}` (core box).
    :param gchi0_q_full_sum: The frequency-summed bare bubble over the full box.
    :param gchi0_q_core_sum: The frequency-summed bare bubble over the core box.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :param mpi_dist_irrq: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param quiet: If ``True``, suppresses every file write (Eliashberg vertex dumps, susceptibility saves, the
        Ornstein-Zernike fit and the lambda text file) - used by the stabilizer's Jacobian probes, which must not
        pollute the run directory. The physics (including the lambda correction itself) is evaluated unchanged.
    :param annealer: The active :class:`LambdaAnnealer` (its boson mass is applied to the physical susceptibility),
        or ``None`` when annealing is off. Quiet probes apply the current mass but skip the gap measurement.
    :return: The self-energy kernel for this channel as a :class:`FourPoint`.
    """
    logger = config.logger

    if config.memory.save_memory_for_chiq_aux:
        gchi_aux_q_r_sum = create_auxiliary_chi_r_q_sum_v3(gamma_r, gchi0_q_inv, u_loc, v_nonloc, mpi_dist_irrq)
    else:
        gchi_aux_q_r_sum = create_auxiliary_chi_r_q_sum_v1(gamma_r, gchi0_q_inv, u_loc, v_nonloc)

    mpi_dist_irrq.barrier()

    logger.log_memory_usage(
        f"Auxiliary susceptibility ({gchi_aux_q_r_sum.channel.value})",
        gchi_aux_q_r_sum,
        mpi_dist_irrq.comm.size * 2 * config.box.niv_core,
    )
    logger.info(f"Non-Local auxiliary susceptibility ({gchi_aux_q_r_sum.channel.value}) calculated.")

    if config.eliashberg.perform_eliashberg and not quiet:
        vrg_q_r_right = create_vrg_r_q_right(gchi_aux_q_r_sum, gchi0_q_inv)
        vrg_q_r_right.save(
            name=f"vrg_q_{vrg_q_r_right.channel.value}_right_rank_{mpi_dist_irrq.comm.rank}",
            output_dir=config.output.eliashberg_path,
        )
        vrg_q_r_right.free()

    vrg_q_r = create_vrg_r_q(gchi_aux_q_r_sum, gchi0_q_inv)

    logger.info(f"Non-local three-leg vertex gamma^wv ({vrg_q_r.channel.value}) done.")
    logger.log_memory_usage(f"Three-leg vertex ({vrg_q_r.channel.value})", vrg_q_r, mpi_dist_irrq.comm.size)

    if config.eliashberg.perform_eliashberg and not quiet:
        vrg_q_r.save(
            name=f"vrg_q_{vrg_q_r.channel.value}_rank_{mpi_dist_irrq.comm.rank}",
            output_dir=config.output.eliashberg_path,
        )

    chi_phys_q_r = gchi_aux_q_r_sum.sum_over_all_vn(config.sys.beta)
    gchi_aux_q_r_sum.free()

    chi_phys_q_r = create_generalized_chi_q_with_shell_correction(
        chi_phys_q_r, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
    )

    logger.info(f"Updated non-local susceptibility chi^q ({chi_phys_q_r.channel.value}) with asymptotic correction.")

    if annealer is not None:
        chi_phys_q_r = annealer.apply(chi_phys_q_r, mpi_dist_irrq, measure=not quiet)

    if config.stabilization.use_chi_phys_restriction:
        chi_phys_q_r, n_floored = restrict_chi_phys_to_positive_eigenvalues(chi_phys_q_r)
        if mpi_dist_irrq.comm.size > 1:
            n_floored = mpi_dist_irrq.comm.allreduce(n_floored)
        logger.warning(
            f"Restricted physical susceptibility ({chi_phys_q_r.channel.value}): floored {n_floored} eigenvalues "
            "of the inverse. Releasing the restriction is only safe once this count decays to zero."
        )

    logger.log_memory_usage(
        f"Physical susceptibility ({chi_phys_q_r.channel.value})", chi_phys_q_r, mpi_dist_irrq.comm.size
    )

    chi_phys_q_r.mat = mpi_dist_irrq.gather(chi_phys_q_r.mat)
    if mpi_dist_irrq.comm.rank == 0:
        chi_phys_q_r = _select_and_apply_lambda_correction(chi_phys_q_r, quiet)
        if not quiet:
            chi_phys_q_r.save(name=f"chi_phys_q_{chi_phys_q_r.channel.value}", output_dir=config.output.output_path)

            # perform Ornstein-Zernike fit
            if chi_phys_q_r.channel == SpinChannel.MAGN:
                perform_ornstein_zernike_fit(chi_phys_q_r)

    chi_phys_q_r.mat = mpi_dist_irrq.scatter(chi_phys_q_r.mat)
    logger.info(f"Saved physical susceptibility ({chi_phys_q_r.channel.value}) to file.")

    min_eig = min_static_compound_eigenvalue(chi_phys_q_r)
    if mpi_dist_irrq.comm.size > 1:
        min_eig = mpi_dist_irrq.comm.allreduce(min_eig, op=MPI.MIN)
    logger.info(f"Minimum static compound eigenvalue of chi_phys ({chi_phys_q_r.channel.value}): {min_eig:.6f}.")
    if min_eig < -5e-2:
        logger.warning(
            f"The static physical susceptibility ({chi_phys_q_r.channel.value}) is not positive semi-definite "
            f"(minimum eigenvalue {min_eig:.3f}): the ladder sits on an unphysical (past-pole) branch and derived "
            "quantities (self-energy, Eliashberg eigenvalues) might be unreliable."
        )

    if config.eliashberg.perform_eliashberg and not quiet:
        chi_phys_q_r.save(
            name=f"chi_phys_q_{chi_phys_q_r.channel.value}_rank_{mpi_dist_irrq.comm.rank}",
            output_dir=config.output.eliashberg_path,
        )

    return calculate_kernel_r_q(vrg_q_r, chi_phys_q_r, v_nonloc, u_loc)


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

    mat *= -0.5 / config.sys.beta / config.lattice.q_grid.nk_tot
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

    mat *= -0.5 / config.sys.beta / config.lattice.q_grid.nk_tot
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

    mat_gpu *= -0.5 / config.sys.beta / config.lattice.q_grid.nk_tot
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
    node_comm=None,
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
    nk_tot = config.lattice.q_grid.nk_tot
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
    _free_shared_window(g_r_win, node_comm)

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
    node_comm=None,
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
    nk_tot = config.lattice.q_grid.nk_tot
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
    _free_shared_window(g_r_win, node_comm)

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
    node_comm=None,
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
    node_comm=None,
) -> SelfEnergy:
    r"""
    Runs one bosonic-frequency FFT self-energy pass: maps the (small) irreducible-BZ kernel to the full BZ
    (consuming ``kernel_src``), optionally builds its time-reversed negative-:math:`\omega` block, contracts the
    requested ``niw_index_w_pairs`` via :func:`calculate_sigma_from_kernel_fft`, and frees the full-BZ kernel. Both
    passes of :func:`calculate_self_energy_q` go through this helper: the caller hands the positive pass a
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


def get_starting_sigma(default_sigma: SelfEnergy) -> tuple[SelfEnergy, int]:
    """
    Tries to retrieve the last calculated self-energy from a previous self-consistency calculation as a starting point
    for the next calculation. Whether the normal or interpolated sigma is chosen depends on the setting. If no
    ``sigma_dga_*_N.npy`` file is found, we use the DMFT self-energy as a starting point.

    :param default_sigma: The fallback (DMFT) :class:`SelfEnergy` used when no previous result is found.
    :return: A tuple of the starting :class:`SelfEnergy` (cut to the core box and interpolated onto the k-grid) and
        the iteration number it was taken from (0 if none found).
    """
    previous_sc_path = config.self_consistency.previous_sc_path

    if previous_sc_path is None or previous_sc_path == "" or not os.path.exists(previous_sc_path):
        return default_sigma, 0

    if config.self_consistency.use_interpolated_sigma:
        glob_pattern = "sigma_dga_interpolated_*_iteration_*.npy"
        iteration_regex = re.compile(r"sigma_dga_interpolated_.+_iteration_(\d+)\.npy$")
    else:
        glob_pattern = "sigma_dga_iteration_*.npy"
        iteration_regex = re.compile(r"sigma_dga_iteration_(\d+)\.npy$")

    files = glob.glob(os.path.join(previous_sc_path, glob_pattern))

    if not files or len(files) == 0:
        return default_sigma, 0
    iterations = [(int(match.group(1)), f) for f in files if (match := iteration_regex.search(f))]

    if not iterations or len(iterations) == 0:
        return default_sigma, 0
    max_iter, max_file = max(iterations, key=lambda x: x[0])

    mat = np.load(max_file)
    return (
        SelfEnergy(mat, mat.shape[:3], True, False, beta=config.sys.beta)
        .cut_niv(config.box.niv_core)
        .interpolate_q_grid(config.lattice.k_grid.nk, False),
        max_iter,
    )


def _init_mu_history(starting_iter: int) -> list[float]:
    r"""
    Seeds the chemical-potential history for the self-consistency loop. For a fresh run (``starting_iter == 0``) the
    history starts at the current (DMFT) chemical potential :math:`\mu`. When resuming from a previous self-consistency
    calculation it is seeded with that run's last :math:`\mu` (from ``mu_history.npy``) and the global ``config.sys.mu``
    is synced to it: otherwise ``config.sys.mu`` would stay at the DMFT value while ``giwk_full`` is built with the
    previous run's :math:`\mu`, and any quantity computed from the global (e.g. the lattice filling in
    :meth:`GreensFunction.get_fill_nonlocal`, which now reads ``self._mu``) would use an inconsistent chemical potential.

    :param starting_iter: The iteration the previous calculation stopped at (0 for a fresh run).
    :return: The single-element chemical-potential history list.
    """
    if starting_iter == 0:
        return [config.sys.mu]

    previous_mu = float(np.load(os.path.join(config.self_consistency.previous_sc_path, "mu_history.npy"))[-1])
    config.sys.mu = previous_mu
    return [previous_mu]


def read_last_n_sigmas_from_files(n: int, output_path: str = "./", previous_sc_path: str = "./") -> list[np.ndarray]:
    """
    Reads the last ``n`` total self-energies from the output directory and - if specified - the previous
    self-consistency path. This is used for the Pulay/Anderson mixing schemes. If one has a history of self-energies
    from a previous calculation, these will be used as well.

    :param n: Number of most recent self-energies to read.
    :param output_path: Directory holding the current run's ``sigma_dga_iteration_*.npy`` files.
    :param previous_sc_path: Directory of a previous self-consistency run to prepend to the history (if set).
    :return: A list of self-energy arrays (cut to the core box and interpolated onto the k-grid), oldest first.
    """

    def _get_top_n_files(path: str, pattern: str, regex: re.Pattern) -> list[tuple[int, str]]:
        """
        Finds the ``n`` highest-iteration files in ``path`` matching ``pattern``/``regex``.

        :param path: Directory to search.
        :param pattern: Glob pattern selecting candidate files.
        :param regex: Regex whose first group captures the iteration number.
        :return: A list of ``(iteration, filepath)`` tuples, sorted ascending, truncated to the last ``n``.
        """
        files = glob.glob(os.path.join(path, pattern))
        matched = [(int(match.group(1)), f) for f in files if (match := regex.search(f))]
        return sorted(matched, key=lambda x: x[0])[-n:]

    interp_pattern = "sigma_dga_interpolated_*_iteration_*.npy"
    interp_regex = re.compile(r"sigma_dga_interpolated_.+_iteration_(\d+)\.npy$")

    normal_pattern = "sigma_dga_iteration_*.npy"
    normal_regex = re.compile(r"sigma_dga_iteration_(\d+)\.npy$")

    last_iterations_previous_dir = []
    if previous_sc_path and previous_sc_path.strip():
        if config.self_consistency.use_interpolated_sigma:
            last_iterations_previous_dir = _get_top_n_files(previous_sc_path, interp_pattern, interp_regex)
        else:
            last_iterations_previous_dir = _get_top_n_files(previous_sc_path, normal_pattern, normal_regex)

    last_iterations_current_dir = _get_top_n_files(output_path, normal_pattern, normal_regex)
    last_iterations = (last_iterations_previous_dir + last_iterations_current_dir)[-n:]

    sigmas = []
    for _, file in last_iterations:
        sigma_mat = np.load(file)
        sigmas.append(
            SelfEnergy(sigma_mat, sigma_mat.shape[:3], True, False, False, False, beta=config.sys.beta)
            .cut_niv(config.box.niv_core)
            .interpolate_q_grid(config.lattice.k_grid.nk, False)
            .mat
        )
    return sigmas


def _load_node_shared_local_vertex(node_comm, path: str, channel: SpinChannel, transform=None) -> tuple:
    r"""
    Loads a local four-point vertex from file **once per node** into an MPI shared-memory window (see
    :func:`dgamore.mpi_utils.build_node_shared_array`): the node root reads the file (and applies ``transform``),
    every other rank maps the same physical buffer read-only. Without a node communicator each rank loads
    privately (the previous behavior). At production box sizes these local vertices are multi-GB and were held
    once **per rank** before - the largest replicated objects of the kernel section after ``giwk_full``.

    :param node_comm: The node-local communicator (or ``None`` for a private per-rank load).
    :param path: Path to the ``.npy`` file.
    :param channel: Spin channel of the loaded vertex.
    :param transform: Optional callable applied to the loaded :class:`LocalFourPoint` on the node root before the
        array is placed in the window (e.g. an orbital permute + scale).
    :return: The tuple ``(vertex, win)``; free the window via :func:`_free_shared_window` once every rank is done
        reading (``win`` is ``None`` on the private path, then ``vertex.free()`` applies as before).
    """

    def _load() -> np.ndarray:
        obj = LocalFourPoint.load(path, channel)
        if transform is not None:
            obj = transform(obj)
        return obj.mat

    if node_comm is None or not config.memory.use_shared_memory_common_obj:
        # ascontiguousarray since a pure orbital-permute transform returns a strided view of the loaded array
        return LocalFourPoint(np.ascontiguousarray(_load()), channel, 1, 2, False, True), None

    mat, win = mpi_utils.build_node_shared_array(node_comm, _load)
    return LocalFourPoint(mat, channel, 1, 2, False, True), win


def _build_giwk_full(comm: MPI.Comm, sigma: SelfEnergy, mu: float, ek: np.ndarray, beta: float) -> tuple:
    r"""
    Builds the full-grid Green's function :math:`G(k, \nu)`, optionally deduplicated across the MPI ranks that share
    a physical node. With ``config.memory.use_shared_memory_common_obj`` set (the default), the Dyson inversion runs only on
    each node's root rank and the result is placed in one MPI shared-memory window per node, so ``giwk_full`` occupies
    a single physical buffer per node instead of one private copy per rank (see
    :func:`dgamore.mpi_utils.build_node_shared_array`). Otherwise every rank builds its own copy. The node topology is
    discovered at runtime via ``comm.Split_type(MPI.COMM_TYPE_SHARED)`` (nothing about the cluster is hard-coded).

    :param comm: The (world) MPI communicator.
    :param sigma: The self-energy :math:`\Sigma` entering the Dyson equation.
    :param mu: Chemical potential :math:`\mu`.
    :param ek: Band dispersion :math:`\varepsilon(k)`.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The tuple ``(giwk_full, win, node_comm)``; ``win`` and ``node_comm`` are ``None`` on the non-shared path
        and must otherwise be released with :func:`_release_shared_giwk` once ``giwk_full`` has been cut to its private
        core box (the shared buffer is read-only and must not be freed while any rank still reads it).
    """
    if not config.memory.use_shared_memory_common_obj:
        return GreensFunction.get_g_full(sigma, mu, ek, beta), None, None

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    giwk_mat, win = mpi_utils.build_node_shared_array(
        node_comm, lambda: GreensFunction.get_g_full(sigma, mu, ek, beta).mat
    )
    giwk_full = GreensFunction(
        giwk_mat, sigma, ek, sigma.full_niv_range, False, False, nk=ek.shape[:3], beta=beta, mu=mu
    )
    return giwk_full, win, node_comm


def _release_shared_giwk(win, node_comm) -> None:
    r"""
    Releases the shared-memory window and node communicator allocated by :func:`_build_giwk_full`, once all node ranks
    have finished reading ``giwk_full`` (i.e. after it has been cut to a private copy). The barrier guarantees no rank
    is still reading the shared buffer when it is freed. A no-op when node-sharing was not used.

    :param win: The MPI shared-memory window (or ``None``).
    :param node_comm: The node-local communicator (or ``None``).
    :return: None.
    """
    if node_comm is None:
        return
    node_comm.Barrier()
    if win is not None:
        win.Free()
    node_comm.Free()


def _cut_and_reshare_giwk(giwk_full: GreensFunction, win, node_comm, niv: int) -> tuple:
    r"""
    Cuts ``giwk_full`` to the :math:`[-niv, niv)` core box. When ``giwk_full`` is node-shared (``node_comm`` is not
    ``None``), the node root cuts the shared full-niv Green's function into a **new, smaller per-node shared window**
    and every rank maps that; the caller then frees the old (large) full-niv window via :func:`_free_shared_window`.
    This keeps the deduplicated ``giwk_full`` at one copy per node through the whole self-energy step, not just the
    bubble. Without sharing it is a plain per-rank cut.

    :param giwk_full: The full-niv Green's function (possibly backed by a shared window).
    :param win: The shared-memory window backing ``giwk_full`` (unused here; freed by the caller afterwards).
    :param node_comm: The node-local communicator (or ``None`` on the non-shared path).
    :param niv: Half width of the target fermionic core box.
    :return: The tuple ``(giwk_cut, cut_win)``; ``cut_win`` is ``None`` on the non-shared or single-rank-node path.
    """
    if node_comm is None:
        return giwk_full.cut_niv(niv), None

    node_comm.Barrier()  # every rank has finished reading the full-niv window (the bubble)
    cut_mat, cut_win = mpi_utils.build_node_shared_array(node_comm, lambda: giwk_full.cut_niv(niv).mat)
    giwk_cut = GreensFunction(
        cut_mat,
        giwk_full._sigma,
        giwk_full._ek,
        giwk_full.full_niv_range,
        False,
        False,
        nk=giwk_full._ek.shape[:3],
        beta=giwk_full._beta,
        mu=giwk_full._mu,
    )
    return giwk_cut, cut_win


def _free_shared_window(win, node_comm) -> None:
    r"""
    Frees a shared-memory window while keeping its node communicator alive - the communicator is reused for the cut
    ``giwk_full`` window and released later by :func:`_release_shared_giwk`. A barrier guarantees no rank is still
    reading the window's buffer. A no-op when there is no window.

    :param win: The MPI shared-memory window (or ``None``).
    :param node_comm: The node-local communicator (or ``None``).
    :return: None.
    """
    if node_comm is None or win is None:
        return
    node_comm.Barrier()
    win.Free()


def calculate_sigma_proposal(
    sigma_in: SelfEnergy,
    mu: float,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    v_nonloc_full: Interaction,
    sigma_dmft: SelfEnergy,
    delta_sigma: SelfEnergy,
    my_irr_q_list: np.ndarray,
    my_full_q_list: np.ndarray,
    mpi_dist_irrk: MpiDistributor,
    mpi_dist_fullbz: MpiDistributor,
    comm: MPI.Comm,
    current_iter: int,
    quiet: bool = False,
    annealer: "LambdaAnnealer | None" = None,
) -> SelfEnergy:
    r"""
    Returns the raw (un-mixed) DGA self-energy proposal :math:`S(\Sigma_{\mathrm{in}})` at chemical potential
    :math:`\mu`: Hartree/Fock, the Dyson Green's function, the bubble, the double-counting, density and magnetic
    kernels, and the FFT Schwinger-Dyson contraction, finished with the noise-removal term and the DMFT tail.

    Single source of truth for the proposal map: it is called once per self-consistency iteration by
    :func:`calculate_self_energy_q` and repeatedly by the matrix-free Jacobian probes of the physical-fixed-point
    stabilizer (see :func:`build_stabilization_projector`). The local irreducible vertex is frozen, so every
    evaluation rebuilds the bubble, the ladder susceptibilities and the SDE self-energy. The Hartree/Fock term reads
    ``config.sys.occ`` / ``occ_k``, which the caller sets consistently with :math:`\Sigma_{\mathrm{in}}`.

    :param sigma_in: The input self-energy (full-BZ or local first-iteration, DMFT tail attached).
    :param mu: The chemical potential the Green's function is built with.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`, reduced to this rank's irreducible q-points.
    :param v_nonloc_full: The non-local interaction on the full q-grid (for the Hartree/Fock term).
    :param sigma_dmft: The DMFT self-energy (cut to the loop's niv), providing the high-frequency tail.
    :param delta_sigma: The DMFT-minus-local noise-removal term on the core box.
    :param my_irr_q_list: This rank's irreducible q-point list.
    :param my_full_q_list: This rank's full-BZ q-point list.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points.
    :param mpi_dist_fullbz: MPI distributor over the full BZ.
    :param comm: The MPI communicator.
    :param current_iter: The current iteration number (the RPA susceptibility is saved only on iteration 1).
    :param quiet: If ``True``, every file write and one-shot side effect (Eliashberg dumps, RPA/susceptibility
        saves, lambda text file, Ornstein-Zernike fit) is suppressed - used by the stabilizer's Jacobian probes.
        Pipeline logging is not gated here; the probes silence it wholesale via :func:`_suppressed_logging`.
    :param annealer: The active :class:`LambdaAnnealer` threaded into the kernel step, or ``None`` when annealing
        is off. Quiet probes apply the current boson mass but skip the gap measurement.
    :return: The raw full-BZ proposal :class:`SelfEnergy` (replicated on every rank, DMFT tail attached).
    """
    logger = config.logger

    hartree, fock = get_hartree_fock(u_loc, v_nonloc_full, my_full_q_list)
    fock = mpi_dist_fullbz.allreduce(fock)
    logger.info("Calculated Hartree and Fock terms.")

    giwk_full, giwk_win, shared_node_comm = _build_giwk_full(
        comm, sigma_in, mu, config.lattice.hamiltonian.get_ek(), config.sys.beta
    )

    logger.log_memory_usage("giwk", giwk_full, comm.size)

    if config.memory.save_memory_for_chi0q:
        gchi0_q = BubbleGenerator.create_generalized_chi0_q_auto(
            mpi_dist_irrk,
            giwk_full,
            config.box.niw_core,
            config.box.niv_full,
            my_irr_q_list,
            config.lattice.q_grid,
            config.sys.beta,
            config.logger,
        )
    else:
        gchi0_q = BubbleGenerator.create_generalized_chi0_q_fft_auto(
            mpi_dist_irrk,
            giwk_full,
            config.box.niw_core,
            config.box.niv_full,
            config.lattice.k_grid,
            config.sys.beta,
            config.logger,
            node_comm=shared_node_comm,
        )

    if config.eliashberg.perform_eliashberg and not quiet:
        gchi0_q.save(name=f"gchi0_q_rank_{comm.rank}", output_dir=config.output.output_path)

    logger.log_memory_usage("Gchi0_q_full", gchi0_q, comm.size)
    # Cut giwk to the core box for the self-energy step. When node-shared, the node root cuts into a new, smaller
    # per-node window and the large full-niv window is freed; the cut giwk stays one copy per node through the SDE.
    old_giwk_win = giwk_win
    giwk_full, giwk_win = _cut_and_reshare_giwk(
        giwk_full, giwk_win, shared_node_comm, config.box.niv_core + config.box.niw_core
    )
    _free_shared_window(old_giwk_win, shared_node_comm)

    # the local vertices are identical on every rank, so they are loaded once per node into shared windows
    f_dc_loc, f_dc_win = _load_node_shared_local_vertex(
        shared_node_comm,
        os.path.join(config.output.output_path, "f_magn_loc.npy"),
        SpinChannel.NONE,
        transform=lambda obj: obj.permute_orbitals("abcd->cbad", copy=False).scale(2.0),
    )
    kernel = calculate_sigma_dc_kernel(f_dc_loc, gchi0_q, u_loc).scale(-1.0)
    f_dc_loc.mat = None
    if f_dc_win is None:
        f_dc_loc.free()
    _free_shared_window(f_dc_win, shared_node_comm)
    logger.info("Calculated double-counting kernel.")

    gchi0_q_full_sum = gchi0_q.sum_over_all_vn(config.sys.beta).scale(1.0 / config.sys.beta)
    gchi0_q_core = gchi0_q.cut_niv(config.box.niv_core)
    gchi0_q.free()
    logger.log_memory_usage("Gchi0_q_core", gchi0_q_core, comm.size)

    gchi0_q_core_sum = gchi0_q_core.sum_over_all_vn(config.sys.beta).scale(1.0 / config.sys.beta)
    gchi0_q_core_inv = gchi0_q_core.invert(copy=False)
    del gchi0_q_core
    logger.log_memory_usage("Gchi0_q_inv", gchi0_q_core_inv, comm.size)

    if current_iter == 1 and not quiet:
        calculate_and_save_chi_q_r_rpa(gchi0_q_core_inv, u_loc, v_nonloc, mpi_dist_irrk)

    if config.eliashberg.perform_eliashberg and not quiet:
        gchi0_q_core_inv.save(name=f"gchi0_q_inv_rank_{comm.rank}", output_dir=config.output.eliashberg_path)

    gamma_dens, gamma_dens_win = _load_node_shared_local_vertex(
        shared_node_comm, os.path.join(config.output.output_path, "gamma_dens_loc.npy"), SpinChannel.DENS
    )
    kernel.add(
        calculate_sigma_kernel_r_q(
            gamma_dens,
            gchi0_q_core_inv,
            gchi0_q_full_sum,
            gchi0_q_core_sum,
            u_loc,
            v_nonloc,
            mpi_dist_irrk,
            quiet,
            annealer,
        ),
        copy=False,
    )
    gamma_dens.mat = None
    if gamma_dens_win is None:
        gamma_dens.free()
    _free_shared_window(gamma_dens_win, shared_node_comm)
    mpi_dist_irrk.barrier()
    logger.info("Calculated kernel for density channel.")

    gamma_magn, gamma_magn_win = _load_node_shared_local_vertex(
        shared_node_comm, os.path.join(config.output.output_path, "gamma_magn_loc.npy"), SpinChannel.MAGN
    )
    kernel.add(
        calculate_sigma_kernel_r_q(
            gamma_magn,
            gchi0_q_core_inv,
            gchi0_q_full_sum,
            gchi0_q_core_sum,
            u_loc,
            v_nonloc,
            mpi_dist_irrk,
            quiet,
            annealer,
        ).scale(3.0),
        copy=False,
    )
    gchi0_q_core_inv.free()
    gchi0_q_full_sum.free()
    gchi0_q_core_sum.free()
    gamma_magn.mat = None
    if gamma_magn_win is None:
        gamma_magn.free()
    _free_shared_window(gamma_magn_win, shared_node_comm)
    logger.info("Calculated kernel for magnetic channel.")

    logger.info("Starting calculation of DGA self-energy.")

    # FFT contraction (the only production path - the q-loop variant peaks HIGHER, see calculate_sigma_from_kernel):
    # split the bosonic sum into positive- and negative-w passes, so only one half-niw full-BZ kernel exists at a time.
    niw = config.box.niw_core
    kernel_irr = kernel  # the (small) irreducible-BZ positive-w kernel, mapped to the full BZ once per pass
    # Decide CPU/GPU (and select the GPU) once
    use_gpu = select_sigma_fft_device(mpi_dist_fullbz)

    sigma_prop = _run_fft_sde_pass(
        kernel_irr.copy(),
        mpi_dist_irrk,
        mpi_dist_fullbz,
        giwk_full,
        [(i, i) for i in range(niw + 1)],
        use_gpu,
        negative_w=False,
        node_comm=shared_node_comm,
    )
    sigma_neg = _run_fft_sde_pass(
        kernel_irr,
        mpi_dist_irrk,
        mpi_dist_fullbz,
        giwk_full,
        [(i, -i) for i in range(1, niw + 1)],
        use_gpu,
        negative_w=True,
        node_comm=shared_node_comm,
    )

    sigma_prop.mat += sigma_neg.mat  # accumulate the rank-local R-space partial self-energies (in place)
    sigma_neg.free()

    sigma_prop.mat = mpi_dist_fullbz.gather(sigma_prop.mat)
    if comm.rank == 0:
        sigma_prop = sigma_prop.ifft().to_full_niv_range()
    sigma_prop = mpi_dist_fullbz.bcast_npoint(sigma_prop)

    logger.info("Self-energy calculated from kernel.")
    logger.log_memory_usage("Non-local sigma", sigma_prop, comm.size)

    # giwk's momentum-space data is no longer needed (only its dispersion ek is used below); drop the shared view
    # on every rank, then release the per-node cut-giwk window and its node communicator.
    if giwk_win is not None:
        giwk_full.mat = None
    _release_shared_giwk(giwk_win, shared_node_comm)

    sigma_prop = sigma_prop + hartree + fock
    logger.info("Full non-local self-energy calculated.")

    # This is done to minimize noise. We remove some fluctuations from dmft that are included in the local self-energy
    # calculated in this code and add the smooth dmft self-energy
    sigma_prop += delta_sigma
    sigma_prop = sigma_prop.concatenate_self_energies(sigma_dmft)
    return sigma_prop


@contextmanager
def _suppressed_logging():
    """
    Temporarily silences info/debug/memory logging on ``config.logger`` for the duration of a block. Used around
    the proposal evaluations of the Jacobian build so the (otherwise per-step) pipeline logging from the bubble,
    kernel and SDE helpers is not emitted dozens of times. Saves and restores the original methods, so it is
    robust to the logger internals; warnings stay audible.
    """
    logger = config.logger
    saved = (logger.info, logger.log_memory_usage, logger.debug)

    def _mute(*a, **k):
        return None

    logger.info = logger.log_memory_usage = logger.debug = _mute
    try:
        yield
    finally:
        logger.info, logger.log_memory_usage, logger.debug = saved


def build_stabilization_projector(
    sigma_star: SelfEnergy,
    mu_star: float,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    v_nonloc_full: Interaction,
    sigma_dmft: SelfEnergy,
    sigma_dmft_full: SelfEnergy,
    delta_sigma: SelfEnergy,
    my_irr_q_list: np.ndarray,
    my_full_q_list: np.ndarray,
    mpi_dist_irrk: MpiDistributor,
    mpi_dist_fullbz: MpiDistributor,
    comm: MPI.Comm,
) -> "jstab.PhysicalSolutionStabilizer | None":
    r"""
    Builds the :class:`~dgamore.jacobian_stabilization.PhysicalSolutionStabilizer` for the modified iterative
    scheme (arXiv:2502.01420; the :math:`\Sigma`-mixing analog of :math:`\mathrm{Eq.~(9)}`, with the sign rule of
    :math:`\mathrm{Eq.~(6)}` and the Arnoldi projector of :math:`\mathrm{SM~Sec.~VI}`) by linearizing the proposal
    map at ``sigma_star``.

    ``sigma_star`` is taken as the (assumed) physical solution - in practice the warm-start self-energy from
    ``previous_sc_path``. A purely local starting self-energy (:math:`n_k = 1`) is broadcast to the full Brillouin
    zone so it matches the proposal output. The Jacobian is restricted to the inner
    :math:`n_{\nu,\mathrm{jac}} \sim n_{\nu,\mathrm{core}}/2` (:math:`\geq 15`) Matsubara window (the unstable
    eigenvectors are low-frequency localized), and each finite-difference probe re-solves :math:`\mu` and the
    occupation for the perturbed self-energy, so the rank-one :math:`\mu` feedback and the Hartree-Fock occupation
    feedback enter the Jacobian exactly. The constraint state (:math:`\mu`, filling, occupation) is snapshotted and
    restored around the build. Returns ``None`` if no reflection-curable unstable direction is found. If the build
    reduces the mixing (reflection-uncurable overshoot mode), the reduced value is written back to
    ``config.self_consistency.mixing`` so the loop damps consistently with the projector.

    Cost: one proposal evaluation for the base point plus one per Arnoldi step (adaptive, capped; see the
    stabilizer class), all paid once here; the loop never re-evaluates the proposal for stabilization.

    :param sigma_star: The warm-start self-energy to linearize at (assumed close to the physical solution).
    :param mu_star: The chemical potential belonging to ``sigma_star``.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`, reduced to this rank's irreducible q-points.
    :param v_nonloc_full: The non-local interaction on the full q-grid.
    :param sigma_dmft: The DMFT self-energy cut to the loop's niv (tail for the proposal).
    :param sigma_dmft_full: The uncut DMFT self-energy (tail for the occupation Green's function).
    :param delta_sigma: The DMFT-minus-local noise-removal term on the core box.
    :param my_irr_q_list: This rank's irreducible q-point list.
    :param my_full_q_list: This rank's full-BZ q-point list.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points.
    :param mpi_dist_fullbz: MPI distributor over the full BZ.
    :param comm: The MPI communicator (the probes are collective; the recurrence is replicated on every rank).
    :return: The built stabilizer, or ``None`` when the captured spectrum needs no reflection.
    :raises ValueError: If ``sigma_star`` lives on a different (non-local) k-grid than the current one.
    """
    logger = config.logger

    p = float(config.self_consistency.mixing)
    niv_core = config.box.niv_core

    # Inner Matsubara window for the Jacobian: about half the DGA core, but never fewer than 15 frequencies
    # (or the whole core if it is smaller).
    niv_jac = max(niv_core // 2, min(15, niv_core))

    mu_tol = 1e-10  # tight mu Newton solve so the finite difference is not contaminated

    ek = config.lattice.hamiltonian.get_ek()

    # The iterated quantity is the full-BZ self-energy. Broadcast a purely local starting self-energy to the
    # full BZ so the base matches the (full-BZ) proposal output.
    sigma_star = sigma_star.compress_q_dimension()
    nk_tot = config.lattice.k_grid.nk_tot
    if sigma_star.nq_tot != nk_tot:
        if sigma_star.nq_tot == 1:
            tiled = np.tile(sigma_star.mat, (nk_tot, 1, 1, 1))
            sigma_star = SelfEnergy(
                tiled, config.lattice.k_grid.nk, sigma_star.full_niv_range, True, False, True, beta=config.sys.beta
            )
        else:
            raise ValueError(
                f"Starting self-energy has {sigma_star.nq_tot} k-points but the current grid has "
                f"{nk_tot}; interpolate it to the current k-grid before enabling stabilization."
            )
    niv = sigma_star.niv
    sl = slice(niv - niv_jac, niv + niv_jac)
    base_inner = np.ascontiguousarray(sigma_star.mat[..., sl])

    # Snapshot the constraint state so the loop resumes from the correct physical values.
    mu0, n0 = mu_star, config.sys.n
    occ0, occ_k0 = config.sys.occ, config.sys.occ_k

    def proposal_fn(inner_mat: np.ndarray) -> np.ndarray:
        with _suppressed_logging():
            sig = sigma_star.copy().compress_q_dimension()
            sig.mat[..., sl] = inner_mat.astype(sig.mat.dtype, copy=False)

            if comm.rank == 0:
                mu = update_mu(mu0, n0, ek, sig.mat, config.sys.beta, sig.fit_smom()[0], logger=logger, tol=mu_tol)
                config.sys.mu = mu
                sig_occ = sig.copy().concatenate_self_energies(sigma_dmft_full)
                giwk_occ = GreensFunction.get_g_full(sig_occ, mu, ek, config.sys.beta)
                _, occ, occ_k = giwk_occ.get_fill_nonlocal()
            else:
                occ, occ_k = None, None
            config.sys.mu = comm.bcast(config.sys.mu, root=0)
            config.sys.occ, config.sys.occ_k = comm.bcast((occ, occ_k), root=0)

            # No annealer is threaded here: probes only run once the scaffold released (mass exactly zero) and quiet
            # probes skip the gap measurement anyway, so passing it would be a no-op on the probed map.
            prop = calculate_sigma_proposal(
                sig,
                config.sys.mu,
                u_loc,
                v_nonloc,
                v_nonloc_full,
                sigma_dmft,
                delta_sigma,
                my_irr_q_list,
                my_full_q_list,
                mpi_dist_irrk,
                mpi_dist_fullbz,
                comm,
                current_iter=0,
                quiet=True,
            )
        return np.ascontiguousarray(prop.compress_q_dimension().mat[..., sl])

    try:
        logger.info(
            f"Building stabilizer at the starting self-energy (inner window niv_jac={niv_jac} of core={niv_core})."
        )
        logger.info(
            "This requires evaluating the DGA self-energy several times to construct the Jacobian; "
            "this step will take a while."
        )
        stabilizer = jstab.PhysicalSolutionStabilizer(
            proposal_fn,
            base_inner,
            p,
            niv_jac,
            n_modes=config.stabilization.stabilizer_n_modes,
            max_residual=config.stabilization.max_stabilizer_base_residual,
            logger=logger,
        )
        if stabilizer.mixing_reduced:
            # A reflection-uncurable instability forced the stabilizer to a smaller, contractive mixing. Adopt it for
            # the loop (apply_mixing_strategy reads config.self_consistency.mixing) so damping matches the projector.
            logger.warning(
                f"Mixing parameter changed: config.self_consistency.mixing "
                f"{config.self_consistency.mixing:.3f} -> {stabilizer.p:.2f} "
                f"(stabilizer required stronger damping to make the physical fixed point attractive)."
            )
            config.self_consistency.mixing = stabilizer.p
        if stabilizer.n_unstable == 0:
            logger.info(
                "No reflection-curable unstable directions remaining at the starting self-energy; "
                "using conventional mixing (at the possibly reduced mixing parameter)."
            )
            return None
        logger.info(f"Stabilizer built with {stabilizer.n_unstable} unstable direction(s).")
    finally:
        config.sys.mu, config.sys.n = mu0, n0
        config.sys.occ, config.sys.occ_k = occ0, occ_k0

    return stabilizer


def _relative_sigma_residual(sigma_new: SelfEnergy, sigma_old: SelfEnergy) -> float:
    r"""
    Returns the relative L2 residual :math:`\lVert\Sigma_{\mathrm{new}} - \Sigma_{\mathrm{old}}\rVert /
    \lVert\Sigma_{\mathrm{old}}\rVert` over the positive fermionic core frequencies (all momenta and orbitals,
    real and imaginary parts stacked). Evaluated on the raw proposal it measures the mixing-independent distance
    to the fixed point, :math:`\lVert S(\Sigma)-\Sigma\rVert/\lVert\Sigma\rVert`; evaluated on the mixed iterate
    it measures the per-iteration step size (which shrinks with the mixing parameter). A local (single-k)
    self-energy is broadcast against a full-BZ one. Layout-safe: the two iterates may arrive with different
    momentum layouts (compressed vs decompressed) and are normalized before comparing.

    :param sigma_new: The new self-energy (raw proposal or mixed iterate).
    :param sigma_old: The previous iterate the residual is measured against.
    :return: The relative residual as a float.
    """
    new_core = sigma_new.mat[..., sigma_new.niv : sigma_new.niv + config.box.niv_core]
    old_core = sigma_old.mat[..., sigma_old.niv : sigma_old.niv + config.box.niv_core]
    # Normalize both to the compressed [q, o1, o2, v] layout before comparing: mismatched layouts (rank 0's iterate is
    # left decompressed by the save path) would pair wrong momenta; a local iterate is broadcast to the full BZ.
    new_core = new_core.reshape(-1, *new_core.shape[-3:])
    old_core = old_core.reshape(-1, *old_core.shape[-3:])
    if old_core.shape[0] != new_core.shape[0]:
        old_core = np.broadcast_to(old_core, new_core.shape)
    return float(np.linalg.norm(new_core - old_core) / np.linalg.norm(old_core))


def _mixing_history_cap(
    current_iter: int, release_iter: int | None, stab_arm_iter: int | None, anneal_reset_iter: int | None = None
) -> int | None:
    """
    Returns the accelerated-mixing history cap for this iteration: the number of iterations since the most recent
    map-switching event - the susceptibility-restriction release, the arming of the stabilizer's reflection, or a
    change of the lambda-annealing mass. Anderson/Pulay must not extrapolate across any of these discontinuities,
    so their usable history is capped to the post-event iterations (``None`` when no event has occurred).

    :param current_iter: The current self-consistency iteration number.
    :param release_iter: The iteration the susceptibility restriction was released on (``None`` if never).
    :param stab_arm_iter: The iteration the stabilizer's reflection was armed on (``None`` if never).
    :param anneal_reset_iter: The iteration the annealing mass last changed on (``None`` if never).
    :return: The history cap, or ``None`` for no cap.
    """
    events = (release_iter, stab_arm_iter, anneal_reset_iter)
    last_reset_iter = max((it for it in events if it is not None), default=None)
    return None if last_reset_iter is None else max(0, current_iter - last_reset_iter - 1)


def _stabilizer_probe_active(
    stab_armed: bool, stab_projector, converged: bool, annealer: "LambdaAnnealer | None"
) -> bool:
    """
    Returns whether the stall detector of the armed stabilizer may run this iteration. It is paused when the
    stabilizer is disarmed or already deployed, when the cycle just converged, and - crucially - while
    ``use_chi_phys_restriction`` or the lambda-annealing scaffold is active: the scaffolded map is a convergence aid,
    and linearizing it would target the wrong Jacobian, so the projector may only be built in the pure phase.

    :param stab_armed: Whether the stabilizer is armed (config flag, not yet deployed or given up).
    :param stab_projector: The deployed stabilizer (``None`` while not built).
    :param converged: Whether this iteration satisfied the convergence criterion.
    :param annealer: The active :class:`LambdaAnnealer`, or ``None`` when annealing is off.
    :return: Whether the stall detector should be updated this iteration.
    """
    return (
        stab_armed
        and stab_projector is None
        and not converged
        and not config.stabilization.use_chi_phys_restriction
        and not (annealer is not None and annealer.active)
    )


def _update_stabilizer_probe(
    relative_residual: float, best_residual: float, growth_count: int, stall_count: int, probe_iters: int
) -> tuple[float, int, int, bool]:
    r"""
    Stall detector for the *armed* physical-solution stabilizer. Deliberately conservative: the stabilizer must
    only engage when plain iteration demonstrably *cannot* reach the physical solution - applying the reflection
    to a run that would have converged on its own steers it onto a different (typically unphysical) branch. Two
    trigger paths, both requiring sustained evidence:

    1. **Divergence**: the residual sits a factor :data:`_STAB_GROWTH_FACTOR` above the best seen for
       ``probe_iters`` consecutive iterations - the iterate is escaping a repelling fixed point.
    2. **Far plateau**: no meaningful improvement (a relative drop of at least :math:`10^{-3}`) for
       :data:`_STAB_PLATEAU_WINDOW_FACTOR` ``* probe_iters`` consecutive iterations while the residual is still
       :data:`_STAB_FAR_RESIDUAL_FACTOR` times above the convergence epsilon - a limit cycle around a repelling
       fixed point. A plateau *near* epsilon (slow tail of a converging run) never triggers.

    The detector cannot fire when plain iteration converges (even to an unphysical branch): a decreasing residual
    resets both counters. Distinguishing physical from unphysical *convergence* is not possible from the residual
    trend alone; the remedy for that regime is a warm start inside the physical basin (temperature continuation).

    :param relative_residual: The current iteration's relative self-energy residual.
    :param best_residual: The lowest residual seen so far.
    :param growth_count: Consecutive iterations spent a factor :data:`_STAB_GROWTH_FACTOR` above the best.
    :param stall_count: Consecutive iterations without meaningful improvement.
    :param probe_iters: The base probe window (the divergence path fires after this many growth iterations).
    :return: The updated ``(best_residual, growth_count, stall_count, trigger)`` tuple.
    """
    if not np.isfinite(relative_residual):
        # an overflowed/NaN residual is the strongest divergence evidence there is; NaN comparisons below would
        # otherwise all evaluate False and silently reset the growth counter
        return best_residual, growth_count + 1, stall_count + 1, growth_count + 1 >= probe_iters
    if relative_residual < best_residual * (1.0 - 1e-3):
        return relative_residual, 0, 0, False
    stall_count += 1
    growth_count = growth_count + 1 if relative_residual >= _STAB_GROWTH_FACTOR * best_residual else 0
    diverging = growth_count >= probe_iters
    far_plateau = (
        stall_count >= _STAB_PLATEAU_WINDOW_FACTOR * probe_iters
        and relative_residual > _STAB_FAR_RESIDUAL_FACTOR * config.self_consistency.epsilon
    )
    return best_residual, growth_count, stall_count, diverging or far_plateau


def _update_stabilizer_watchdog(
    relative_residual: float, arming_residual: float, watch_count: int, probe_iters: int
) -> tuple[int, bool, bool]:
    r"""
    Do-no-harm watchdog for the *deployed* reflection: if the modified scheme has not meaningfully improved on the
    residual level it was armed at within :data:`_STAB_WATCH_WINDOW_FACTOR` ``* probe_iters`` iterations, the
    reflection is to be reverted (plain mixing resumes) - the projector is evidently not curing this run and must
    not be allowed to steer it onto a different branch. A genuinely repelling-but-flipped fixed point improves the
    residual within the window even at slow post-flip rates.

    :param relative_residual: The current iteration's relative self-energy residual.
    :param arming_residual: The residual level at which the reflection was armed.
    :param watch_count: Consecutive reflected iterations observed so far.
    :param probe_iters: The base probe window (the watchdog allows ``_STAB_WATCH_WINDOW_FACTOR`` times this).
    :return: The updated ``(watch_count, passed, revert)`` tuple; ``passed`` ends the watch, ``revert`` disables
        the reflection.
    """
    if relative_residual < arming_residual * (1.0 - 1e-3):
        return watch_count, True, False
    watch_count += 1
    return watch_count, False, watch_count >= _STAB_WATCH_WINDOW_FACTOR * probe_iters


def apply_modified_preconditioner(
    sigma_new: SelfEnergy, sigma_old: SelfEnergy, stabilizer: "jstab.PhysicalSolutionStabilizer"
) -> SelfEnergy:
    r"""
    Reflects the unstable component of the inner-window residual *before* the normal mixing (arXiv:2502.01420).
    This realizes the modified iterative scheme while preserving the configured mixing: the subsequent
    :func:`apply_mixing_strategy` then linearly mixes or Anderson/Pulay-accelerates the *modified* fixed-point map
    :math:`\Sigma_{n+1} = \Sigma_n + \mathcal{P}\,(S(\Sigma_n) - \Sigma_n)`. Only the inner Jacobian window is
    touched; outside it the proposal is unchanged.

    :param sigma_new: The raw self-energy proposal :math:`S(\Sigma_n)`.
    :param sigma_old: The previous iterate :math:`\Sigma_n`.
    :param stabilizer: The built :class:`~dgamore.jacobian_stabilization.PhysicalSolutionStabilizer`.
    :return: The proposal with the reflected inner window (same object, modified in place on the window).
    """
    sigma_new = sigma_new.compress_q_dimension()
    sigma_old = sigma_old.compress_q_dimension()

    niv = sigma_new.niv
    niv_jac = stabilizer.niv_jac
    sl = slice(niv - niv_jac, niv + niv_jac)
    new_inner = sigma_new.mat[..., sl]
    old_inner = sigma_old.mat[..., sl]
    if old_inner.shape != new_inner.shape:
        # First iteration without a warm start: ``sigma_old`` is the local (n_k = 1) DMFT self-energy, the proposal
        # full-BZ. It is k-independent, so broadcast it before reflecting (the 1-D residual needs matching shapes).
        old_inner = np.broadcast_to(old_inner, new_inner.shape)
    reflected = stabilizer.reflect_proposal(new_inner, old_inner)
    sigma_new.mat[..., sl] = reflected.astype(sigma_new.mat.dtype, copy=False)

    config.logger.info(f"Modified-scheme residual reflection applied (n_unstable={stabilizer.n_unstable}).")
    return sigma_new


def calculate_self_energy_q(
    comm: MPI.Comm, u_loc: LocalInteraction, v_nonloc: Interaction, sigma_dmft: SelfEnergy, sigma_local: SelfEnergy
) -> SelfEnergy:
    r"""
    Runs the non-local DGA self-energy calculation. Calculates the Hartree- and Fock terms, the bubble,
    the double-counting correction and the kernel in the density and magnetic channel. Finally, calculates the
    non-local self-energy from the kernel and the Green's function. Also takes care of the self-consistency loop and
    the chemical potential adjustment as well as the self-energy mixing, etc.

    :param comm: The MPI communicator.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :param sigma_dmft: The DMFT self-energy (used as the starting point and for the shell/tail correction).
    :param sigma_local: The locally recomputed self-energy (used for smoothing out the DGA :class:`SelfEnergy`).
    :return: The converged (or last-iteration) momentum-dependent DGA :class:`SelfEnergy`.
    """
    logger = config.logger

    logger.info("Starting with non-local DGA routine.")
    logger.info("Initializing MPI distributor.")

    # MPI distributor for the irreducible BZ
    mpi_dist_irrk = MpiDistributor.create_distributor(
        ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q", output_path=config.output.output_path
    )
    full_q_list = config.lattice.q_grid.get_q_list()
    irrk_q_list = config.lattice.q_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    mpi_dist_fullbz = MpiDistributor.create_distributor(
        ntasks=config.lattice.q_grid.nk_tot, comm=comm, name="FBZ", output_path=config.output.output_path
    )
    my_full_q_list = full_q_list[mpi_dist_fullbz.my_slice]

    sigma_old, starting_iter = get_starting_sigma(sigma_dmft)
    if starting_iter > 0:
        logger.info(
            f"Using previous calculation and starting the self-consistency loop at iteration {starting_iter + 1}."
        )

    mu_history = _init_mu_history(starting_iter)

    # rank 0 keeps the accelerated-mixing self-energy history in memory (seeded once from files for resumed runs):
    # every rank used to re-read/re-interpolate the last n sigma files each iteration - identical data, redundant IO.
    sigma_history = None
    if comm.rank == 0 and config.self_consistency.mixing_strategy.lower() in ("pulay", "anderson"):
        sigma_history = read_last_n_sigmas_from_files(
            config.self_consistency.mixing_history_length,
            config.output.output_path,
            config.self_consistency.previous_sc_path,
        )

    niv_cut = min(config.box.niw_core + config.box.niv_full + 10, config.box.niv_dmft)
    sigma_dmft_full = sigma_dmft.copy()

    if comm.rank == 0:
        giwk_full_dmft = GreensFunction.get_g_full(
            sigma_dmft_full, config.sys.mu_dmft, config.lattice.hamiltonian.get_ek(), config.sys.beta
        )
        giwk_full_dmft.save(output_dir=config.output.output_path, name="g_latt_dmft")
        giwk_full_dmft.free()

        sigma_old = sigma_old.concatenate_self_energies(sigma_dmft_full)

        giwk_full = GreensFunction.get_g_full(
            sigma_old, mu_history[-1], config.lattice.hamiltonian.get_ek(), config.sys.beta
        )
        config.sys.n, config.sys.occ, config.sys.occ_k = giwk_full.get_fill_nonlocal()
        giwk_full.free()

    config.sys.n, config.sys.occ, config.sys.occ_k = comm.bcast(
        (config.sys.n, config.sys.occ, config.sys.occ_k), root=0
    )

    sigma_old = sigma_old.cut_niv(niv_cut)
    sigma_dmft = sigma_dmft.cut_niv(niv_cut)

    if sigma_old.niv < niv_cut:
        sigma_old = sigma_old.concatenate_self_energies(sigma_dmft)

    delta_sigma = sigma_dmft.cut_niv(config.box.niv_core) - sigma_local.cut_niv(config.box.niv_core)

    v_nonloc_full = v_nonloc.copy()
    v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    # The modified iterative scheme (arXiv:2502.01420) is only needed once the physical fixed point turns *repelling*
    # (plain iteration stops contracting); so the stabilizer is *armed*, engaging only after the residual stalls.
    stab_armed = config.stabilization.use_jacobian_stabilization
    stab_projector = None
    sigma_warm = sigma_old.copy() if stab_armed else None
    stab_probe_iters = max(int(config.stabilization.stabilizer_probe_iters), 1)
    stab_best_residual = float("inf")
    stab_growth_count = 0
    stab_stall_count = 0
    stab_arm_iter = None
    stab_arm_residual = None
    stab_watch_count = 0
    stab_watch_passed = False

    annealer = LambdaAnnealer() if config.stabilization.use_lambda_annealing else None
    anneal_reset_iter = None
    release_iter = None
    for current_iter in range(starting_iter + 1, starting_iter + config.self_consistency.max_iter + 1):
        logger.info("----------------------------------------")
        logger.info(f"Starting iteration {current_iter}.")
        logger.info("----------------------------------------")

        sigma_new = calculate_sigma_proposal(
            sigma_old,
            mu_history[-1],
            u_loc,
            v_nonloc,
            v_nonloc_full,
            sigma_dmft,
            delta_sigma,
            my_irr_q_list,
            my_full_q_list,
            mpi_dist_irrk,
            mpi_dist_fullbz,
            comm,
            current_iter,
            annealer=annealer,
        )
        # delta_sigma = sigma_dmft.cut_niv(config.box.niv_core) - sigma_new.q_mean().cut_niv(config.box.niv_core)

        # Modified iterative scheme: reflect the proposal residual on the unstable subspace so the physical fixed point
        # becomes attractive. Applied before the usual mixing, which still sees a consistent (preconditioned) proposal.
        if stab_projector is not None:
            sigma_new = apply_modified_preconditioner(sigma_new, sigma_old, stab_projector)

        sigma_old = sigma_old.cut_niv(config.box.niv_core)

        logger.info("Applying mixing strategy to the self-energy.")
        sigma_old = sigma_old.concatenate_self_energies(sigma_dmft)
        history_cap = _mixing_history_cap(current_iter, release_iter, stab_arm_iter, anneal_reset_iter)
        # mixing runs on rank 0 only (all ranks computed identical results before) and the mixed sigma is broadcast
        if comm.rank == 0:
            sigma_new = apply_mixing_strategy(
                sigma_new, sigma_old, sigma_dmft, current_iter, history_cap, sigma_history
            )
        sigma_new = mpi_dist_fullbz.bcast_npoint(sigma_new)
        if sigma_history is not None:
            # the in-memory analog of what read_last_n_sigmas_from_files reproduced from the file just saved below
            sigma_history.append(sigma_new.decompress_q_dimension().cut_niv(config.box.niv_core).mat)
            del sigma_history[: -config.self_consistency.mixing_history_length]

        sigma_new = sigma_new.compress_q_dimension()
        sigma_old = sigma_old.compress_q_dimension()

        # Post-mixing step residual (the historical convergence measure; shrinks with the mixing parameter)
        relative_residual = _relative_sigma_residual(sigma_new, sigma_old)

        old_mu = mu_history[-1]
        if comm.rank == 0:
            config.sys.mu = update_mu(
                old_mu,
                config.sys.n,
                config.lattice.hamiltonian.get_ek(),
                sigma_new.mat,
                config.sys.beta,
                sigma_new.fit_smom()[0],
                logger=logger,
            )

        config.sys.mu = comm.bcast(config.sys.mu)
        mu_history.append(config.sys.mu)
        logger.info(f"Updated mu from {old_mu} to {config.sys.mu}.")

        if comm.rank == 0:
            sigma_occ = sigma_new.copy().concatenate_self_energies(sigma_dmft_full)
            giwk_occ = GreensFunction.get_g_full(
                sigma_occ, config.sys.mu, config.lattice.hamiltonian.get_ek(), config.sys.beta
            )
            # calculate new occupation matrix from new Green's function (outside asympt region it is the DMFT
            # lattice Green's function)
            _, config.sys.occ, config.sys.occ_k = giwk_occ.get_fill_nonlocal()  # n should not change

            ekin = giwk_occ.get_ekin()
            logger.info(f"Kinetic energy: {ekin:.4f} [t or eV].")

            epot = giwk_occ.get_epot()
            logger.info(f"Potential energy: {epot:.4f} [t or eV].")
            logger.info(f"Total energy: {(ekin + epot):.4f} [t or eV].")
        config.sys.occ, config.sys.occ_k = comm.bcast((config.sys.occ, config.sys.occ_k), root=0)

        if config.self_consistency.max_iter > 1:
            logger.info("Updated occupation matrix from new Green's function.")

        if comm.rank == 0:
            sigma_new.decompress_q_dimension().save(
                name=f"sigma_dga_iteration_{current_iter}", output_dir=config.output.output_path
            )
            logger.info(f"Saved sigma for iteration {current_iter}.")

            if config.self_energy_interpolation.do_interpolation:
                beta_target = config.self_energy_interpolation.beta_target
                niv_target = config.self_energy_interpolation.niv_target
                sigma_new.decompress_q_dimension().interpolate(beta_target, niv_target).save(
                    name=f"sigma_dga_interpolated_beta{beta_target}_niv{niv_target}_iteration_{current_iter}",
                    output_dir=config.output.output_path,
                )
                logger.info(
                    f"Interpolated sigma for iteration {current_iter} to beta={beta_target} and niv={niv_target}."
                )

        logger.info("Checking self-consistency convergence.")
        if comm.rank == 0 and current_iter > starting_iter + 1:
            # Convergence is declared on the post-mixing step residual (the returned iterate). The un-mixed proposal
            # residual is deliberately not used: it can plateau above epsilon and would block convergence forever.
            eps = _effective_epsilon(annealer)
            sigma_converged = abs(relative_residual) < eps
            logger.info(
                f"Self-energy convergence: {sigma_converged} "
                f"(relative step residual={relative_residual:.3e}, epsilon={eps:.3e})."
            )

            mu_converged = abs(mu_history[-1] - mu_history[-2]) < np.pi / (10 * config.sys.beta)
            logger.info(f"Chemical potential convergence: {mu_converged}.")

            converged = mu_converged and sigma_converged
        else:
            converged = False
        converged = comm.bcast(converged)

        # Lambda-annealing schedule (single owner): init/bump/halve the shared mass once per iteration, resetting the
        # mixing history on any change; a change means the converged verdict belongs to the OLD (scaffolded) map.
        anneal_blocks_break = False
        if annealer is not None:
            anneal_mass_changed = annealer.update(converged)
            if anneal_mass_changed:
                anneal_reset_iter = current_iter
            anneal_blocks_break = anneal_mass_changed or annealer.mass_present

        # Arm-and-trigger: deploy the modified scheme only once plain iteration stops contracting (the residual stalls);
        # decided on rank 0 (authoritative residual) and broadcast so the collective projector build runs in lock-step.
        if _stabilizer_probe_active(stab_armed, stab_projector, converged, annealer):
            # the detectors watch the post-mixing step residual (growth is ratio-based and mixing-invariant; the
            # plateau threshold is measured against epsilon on the same step residual the convergence check uses)
            if comm.rank == 0:
                stab_best_residual, stab_growth_count, stab_stall_count, trigger = _update_stabilizer_probe(
                    relative_residual, stab_best_residual, stab_growth_count, stab_stall_count, stab_probe_iters
                )
            else:
                trigger = False
            trigger = comm.bcast(trigger)
            if trigger:
                logger.info(
                    "Plain iteration is demonstrably not reaching the physical fixed point (sustained residual "
                    "growth or a long plateau far above epsilon): it appears repelling. Arming the modified "
                    "iterative scheme (building the projector at the warm-start self-energy)."
                )
                mixing_before_build = config.self_consistency.mixing
                try:
                    stab_projector = build_stabilization_projector(
                        sigma_warm,
                        mu_history[-1],
                        u_loc,
                        v_nonloc,
                        v_nonloc_full,
                        sigma_dmft,
                        sigma_dmft_full,
                        delta_sigma,
                        my_irr_q_list,
                        my_full_q_list,
                        mpi_dist_irrk,
                        mpi_dist_fullbz,
                        comm,
                    )
                except jstab.PhysicalSolutionStabilizerError as exc:
                    # deliberately non-fatal inside the loop: the guard means the starting point (typically a cold
                    # DMFT start) is unusable for the linearization, not that the run itself is lost
                    stab_projector = None
                    logger.warning(
                        f"Stabilizer build aborted; continuing with plain mixing. Reason: {exc} "
                        f"If this run converges, verify the result is physical (minimum static compound "
                        f"eigenvalue lines)."
                    )
                stab_armed = False
                sigma_warm = None  # the linearization point is no longer needed; free the full-BZ copy
                if stab_projector is not None or config.self_consistency.mixing != mixing_before_build:
                    # the reflection switches the iterated map (and a mixing-only build changes it too); the accelerated
                    # mixing history must not extrapolate across either switch (as at the restriction release)
                    stab_arm_iter = current_iter
                if stab_projector is not None:
                    stab_arm_residual = relative_residual
        elif stab_projector is not None and not stab_watch_passed and not converged:
            # Do-no-harm watchdog on the deployed reflection: if it has not improved on its arming residual within the
            # window, revert to plain mixing (another history reset) rather than let a bad projector steer the run.
            if comm.rank == 0:
                stab_watch_count, stab_watch_passed, revert = _update_stabilizer_watchdog(
                    relative_residual, stab_arm_residual, stab_watch_count, stab_probe_iters
                )
            else:
                revert = False
            stab_watch_passed, revert = comm.bcast((stab_watch_passed, revert))
            if revert:
                stab_projector = None
                stab_arm_iter = current_iter
                logger.warning(
                    "The modified scheme did not improve its arming residual within its watch window; reverted. "
                    "Verify the converged result is physical (min static compound eigenvalue), else warm-start."
                )

        sigma_old = sigma_new
        if converged:
            if config.stabilization.use_chi_phys_restriction:
                config.stabilization.use_chi_phys_restriction = False
                release_iter = current_iter
                logger.info(
                    "ATTENTION: Self-consistency with restricted susceptibility reached (at 10x epsilon). "
                    "Disabling the restriction and continuing to full precision with a reset mixing history."
                )
                if current_iter == starting_iter + config.self_consistency.max_iter:
                    logger.warning(
                        "The restriction was released on the final iteration - no unrestricted iterations remain, "
                        "so the returned self-energy is the restricted-phase result."
                    )
            elif config.stabilization.use_lambda_correction:
                config.stabilization.use_lambda_correction = False
                release_iter = current_iter
                logger.info(
                    "ATTENTION: Self-consistency with the lambda correction reached (at 10x epsilon). "
                    "Disabling the correction and continuing to the pure fixed point with a reset mixing history."
                )
                if current_iter == starting_iter + config.self_consistency.max_iter:
                    logger.warning(
                        "The lambda correction was released on the final iteration - no uncorrected iterations "
                        "remain, so the returned self-energy is the lambda-corrected result, NOT pure self-consistency."
                    )
            elif anneal_blocks_break:
                # an annealing phase converged at the relaxed epsilon; the schedule above already advanced (or
                # bumped) the masses - only a converged phase with all masses at exactly zero counts as final
                if current_iter == starting_iter + config.self_consistency.max_iter:
                    logger.warning(
                        "The annealing mass is still nonzero on the final iteration - no further iterations "
                        "remain, so the returned self-energy is a scaffolded-phase result, NOT pure "
                        "self-consistency."
                    )
            else:
                logger.info(f"Self-consistency of sigma and mu reached at iteration {current_iter}.")
                break
        else:
            logger.info("Self-consistency not reached.")

    mpi_dist_irrk.delete_file()
    mpi_dist_fullbz.delete_file()

    np.save(os.path.join(config.output.output_path, "mu_history.npy"), mu_history)
    logger.info("Saved mu history as numpy array.")

    return sigma_old


def apply_mixing_strategy(
    sigma_new: SelfEnergy,
    sigma_old: SelfEnergy,
    sigma_dmft: SelfEnergy,
    current_iter: int,
    history_cap: int | None = None,
    sigma_history: list | None = None,
) -> SelfEnergy:
    """
    Applies the self-energy mixing strategy for the self-consistency loop. Supports linear mixing as well as the
    accelerated Pulay (DIIS) and Anderson schemes (which use the self-energy history read from file); the accelerated
    schemes fall back to linear mixing when their least-squares problem is ill-conditioned or the history is too short.
    The mixing strategy and parameters are taken from the config.

    :param sigma_new: The freshly computed self-energy proposal.
    :param sigma_old: The previous iteration's self-energy.
    :param sigma_dmft: The DMFT self-energy (used to seed the proposal history for the accelerated schemes).
    :param current_iter: The current self-consistency iteration number.
    :param history_cap: Optional upper bound on the number of history entries used by the accelerated schemes
        (``None`` for no bound). Used to reset the mixing history after the susceptibility-restriction release, so
        the accelerated schemes never extrapolate across the restricted-to-unrestricted discontinuity.
    :param sigma_history: Optional in-memory self-energy history (core-cut arrays, oldest first, the layout of
        :func:`read_last_n_sigmas_from_files`). When given, the accelerated schemes read it instead of re-loading
        and re-interpolating the saved sigma files - the caller (rank 0 of the self-consistency loop) maintains it
        across iterations, so the per-iteration per-rank file reads disappear.
    :return: The mixed :class:`SelfEnergy` for the next iteration.
    """
    logger = config.logger
    n_hist = config.self_consistency.mixing_history_length
    if history_cap is not None:
        n_hist = min(n_hist, history_cap)
    alpha = config.self_consistency.mixing

    last_results, last_proposals = [], []
    if config.self_consistency.mixing_strategy.lower() in ("pulay", "anderson") and n_hist > 0:
        if sigma_history is not None:
            last_results = list(sigma_history[-n_hist:])
        else:
            last_results = read_last_n_sigmas_from_files(
                n_hist, config.output.output_path, config.self_consistency.previous_sc_path
            )
        sigma_dmft_stacked = np.tile(
            sigma_dmft.cut_niv(config.box.niv_core).mat, (config.lattice.k_grid.nk_tot, 1, 1, 1)
        )

        last_proposals = [sigma_dmft_stacked] + last_results  # [dmft, s1, ..., s_{n-1}]
        last_results = last_results + [sigma_new.cut_niv(config.box.niv_core).mat]  # [s1,  s2, ..., s_n]

        logger.info(f"Using the last {min(n_hist, len(last_results))} self-energies of the mixing history.")

    accelerated_mixing_condition = current_iter > n_hist and len(last_results) > n_hist and len(last_proposals) > n_hist

    if config.self_consistency.mixing_strategy.lower() == "pulay" and accelerated_mixing_condition:
        shape = last_results[-1].shape
        n_total = int(np.prod(shape))
        r_matrix = np.zeros((2 * n_total, n_hist), dtype=np.float64)
        f_matrix = np.zeros_like(r_matrix)
        f_i = np.zeros((2 * n_total), dtype=np.float64)

        def get_proposal(idx: int):
            """
            Fetches a flattened proposal self-energy from the history.

            :param idx: Index into the proposal history.
            :return: The flattened proposal self-energy at ``idx``.
            """
            return last_proposals[idx].flatten()

        def get_result(idx: int):
            """
            Fetches a flattened result self-energy from the history.

            :param idx: Index into the result history.
            :return: The flattened result self-energy at ``idx``.
            """
            return last_results[idx].flatten()

        for i in range(n_hist):
            proposal_diff = get_proposal(-1 - i) - get_proposal(-2 - i)
            r_matrix[:n_total, i] = proposal_diff.real
            r_matrix[n_total:, i] = proposal_diff.imag

            result_diff = get_result(-1 - i) - get_result(-2 - i)
            f_matrix[:n_total, i] = result_diff.real
            f_matrix[n_total:, i] = result_diff.imag

            f_matrix[:, i] -= r_matrix[:, i]

        # Residual: F(x_n) - x_n, where x_n = last_proposals[-1] = sigma_old (core window)
        iter_diff = get_result(-1) - get_proposal(-1)
        f_i[:n_total] = iter_diff.real
        f_i[n_total:] = iter_diff.imag
        norm_f = np.linalg.norm(f_i)

        # Solve min||F @ c - f_i|| via truncated-SVD pseudoinverse (drops collinear directions)
        u, s, vh = np.linalg.svd(f_matrix, full_matrices=False)
        cutoff = 1e-5 * (s[0] if s.size else 1.0)
        mask = s > cutoff
        if not np.any(mask):
            logger.warning("Pulay SVD ill-conditioned - falling back to linear mixing.")
            return alpha * sigma_new + (1 - alpha) * sigma_old
        coeffs = vh[mask].T @ ((u[:, mask].T @ f_i) / s[mask])

        # Pulay update: x_{n+1} = x_n + alpha*f_i - (R + alpha*F) @ c
        update = alpha * f_i - (r_matrix + alpha * f_matrix) @ coeffs
        norm_u = np.linalg.norm(update)
        if norm_f > 0 and norm_u > 10.0 * norm_f:
            update *= 10.0 * norm_f / norm_u
            logger.warning(f"Pulay step clamped (norm_u={norm_u:.3e}, norm_f={norm_f:.3e}).")
        update = update[:n_total] + 1j * update[n_total:]

        # Update the new self energy
        niv = sigma_new.niv
        niv_core = config.box.niv_core
        sigma_new.mat[..., niv - niv_core : niv + niv_core] = get_proposal(-1).reshape(shape) + update.reshape(shape)

        logger.info(f"Pulay mixing applied (m={n_hist}, alpha={alpha:.3f}, norm_f={norm_f:.3e}).")

        return sigma_new
    if config.self_consistency.mixing_strategy.lower() == "anderson" and accelerated_mixing_condition:
        shape = last_results[-1].shape
        n_total = int(np.prod(shape))
        flat = lambda x: x.reshape(-1)

        # Current residual f_n = F(x_n) - x_n
        f_curr = flat(last_results[-1]) - flat(last_proposals[-1])
        f_vec = np.concatenate([f_curr.real, f_curr.imag])
        norm_f = np.linalg.norm(f_vec)

        # Build dX and dF matrices (n_hist columns): dX[:,i] = x_{n-i} - x_{n-i-1} (proposal differences),
        # dF[:,i] = f_{n-i} - f_{n-i-1} (residual differences).
        dx_cols = []
        df_cols = []
        for i in range(n_hist):
            dx = flat(last_proposals[-1 - i]) - flat(last_proposals[-2 - i])
            dx_cols.append(np.concatenate([dx.real, dx.imag]))

            df_i = flat(last_results[-1 - i]) - flat(last_proposals[-1 - i])
            df_im1 = flat(last_results[-2 - i]) - flat(last_proposals[-2 - i])
            df = df_i - df_im1
            df_cols.append(np.concatenate([df.real, df.imag]))

        dx_matrix = np.column_stack(dx_cols)  # (2*n_total, n_hist)
        df_matrix = np.column_stack(df_cols)  # (2*n_total, n_hist)

        # Anderson: solve min ||f_curr - dF @ c||
        try:
            u, s, vh = np.linalg.svd(df_matrix, full_matrices=False)

            s_max = s[0] if len(s) > 0 else 1.0
            cutoff = 1e-5 * s_max
            mask = s > cutoff

            if not np.any(mask):
                raise np.linalg.LinAlgError("All singular values below threshold.")

            s_reg = s[mask] / (s[mask] ** 2 + cutoff**2)
            coeffs = vh[mask].T @ (s_reg * (u[:, mask].T @ f_vec))

        except np.linalg.LinAlgError:
            logger.warning("Anderson SVD failed - falling back to linear mixing.")
            return alpha * sigma_new + (1 - alpha) * sigma_old

        # Undamped Anderson proposal: x_n + f_n - (dX + dF) @ c
        x_n = flat(last_proposals[-1])
        x_anderson = np.concatenate([x_n.real, x_n.imag]) + f_vec - (dx_matrix + df_matrix) @ coeffs
        x_anderson = x_anderson[:n_total] + 1j * x_anderson[n_total:]

        # Damp between old proposal and Anderson proposal
        x_n_complex = x_n
        candidate = (1 - alpha) * x_n_complex + alpha * x_anderson.reshape(-1)

        # Safety clamp
        update = candidate - x_n_complex
        norm_u = np.linalg.norm(update)
        if norm_f > 0 and norm_u > 3.0 * norm_f:
            candidate = x_n_complex + update * (3.0 * norm_f / norm_u)
            logger.warning(f"Anderson step clamped (norm_u={norm_u:.3e}, norm_f={norm_f:.3e}).")

        # Update the new self energy
        niv = sigma_new.niv
        niv_core = config.box.niv_core
        sigma_new.mat[..., niv - niv_core : niv + niv_core] = candidate.reshape(shape)

        logger.info(f"Anderson acceleration applied (m={n_hist}, alpha={alpha:.3f}, norm_f={norm_f:.3e}).")

        return sigma_new

    sigma_new = alpha * sigma_new + (1 - alpha) * sigma_old
    logger.info(f"Sigma linearly mixed (m=1, alpha={alpha}).")
    return sigma_new

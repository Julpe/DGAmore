# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Non-local ladder DGA step - the self-consistency driver of the parallel-heavy core of the code. Starting from the local
irreducible vertex :math:`\Gamma_{r}` and the bare interaction, one proposal builds, per momentum :math:`\mathbf{q}` and
spin channel, the bubble :math:`\chi^{\mathrm{q}\nu}_{0}`, the auxiliary susceptibility :math:`\chi^{*;\mathrm{q}}_{r}`,
the three-leg vertex :math:`\gamma^{\mathrm{q}\nu}_{r}`, the physical susceptibility :math:`\chi^{\mathrm{q}}_{r}` (with
shell and optional :math:`\lambda`-correction) and the self-energy kernel, then contracts the kernel with the Green's
function to get the momentum-dependent self-energy :math:`\Sigma^{\mathrm{k}}_{12}`, distributed over MPI ranks. The
numerics of these steps live in :mod:`~dgamore.sde_kernels`; this module runs them per proposal, dispatches the
:math:`\lambda`-correction, monitors and restricts the physical susceptibility, and wraps the proposals in a
self-consistency loop with chemical-potential adjustment, starting and resumed self-energies and Jacobian tracking. The
self-energy mixing (linear / Pulay / Anderson) lives in :mod:`~dgamore.mixing`. Equation numbers refer to the author's
master's thesis (Chapters 3 & 4).
"""

import glob
import os
import pickle
import re
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass

import mpi4py.MPI as MPI
import numpy as np
from scipy import optimize as opt

import dgamore.config as config
import dgamore.mpi_utils as mpi_utils
from dgamore.brillouin_zone import KGrid
from dgamore.bubble_gen import BubbleGenerator
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction, non_interacting_mu, orbital_block, update_mu
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.jacobian_stabilization import (
    JACOBIAN_FILE,
    JacobianTracker,
    TRACKER_PAIRS,
    load_spectrum,
    to_mat,
    to_vec,
)
from dgamore.local_four_point import LocalFourPoint
from dgamore.local_n_point import LocalNPoint
from dgamore.lambda_ops import LambdaAnnealer, LambdaCorrection, MultiOrbitalLambdaCorrection
from dgamore.matsubara_frequencies import MFHelper
from dgamore import memory_estimator, mixing, output_files, sde_kernels, sigma_jacobian
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import DTYPE, SpinChannel, deferred_collection
from dgamore.self_energy import SelfEnergy
from dgamore.two_point import TwoPoint

_JACOBIAN_ROW_WIDTH = TRACKER_PAIRS - 1  # Ritz values per row of the traces: every value an estimate can have
_FIRST_FREQUENCY_RATIO = 2.0  # w_1 over static compound norm of chi_phys above which the loop warns of a pole
_RESTRICTION_PIN_FACTOR = 2.0  # pin of a crossed mode, in units of the largest healthy static eigenvalue of the channel
_RESTRICTION_FALLBACK_FLOOR = 1e-2  # floor of the inverse static eigenvalues when no static block is healthy
_SHELL_ROWS = 64  # momenta per chunk of a damped step's padded rows: 2.5 MB of complex64 at four bands, 300 frequencies
_PROPOSAL_ROWS_TAG = 1500  # base MPI tag of the proposal's core rows sent to the other nodes


def save_rank_file(obj: LocalNPoint, name: str, niv_cut: int | None = None) -> None:
    """
    Saves one per-rank intermediate of the Eliashberg step as ``<name>.npy`` in the run's Eliashberg folder. Once a
    failed rank has placed :data:`~dgamore.output_files.ABORT_MARKER` there (see
    :func:`~dgamore.eliashberg_solver.delete_rank_files`), nothing is written, and a file whose save straddles the
    marker's creation is deleted again, so no rank adds a rank file after the cleanup began.

    :param obj: The object to save.
    :param name: File name (without extension).
    :param niv_cut: Number of fermionic frequencies (per sign) to write, or ``None`` for all (see
        :meth:`~dgamore.local_n_point.LocalNPoint.save`).
    :return: None.
    """
    path = config.output.eliashberg_path
    marker = os.path.join(path, output_files.ABORT_MARKER)
    if os.path.exists(marker):
        return
    obj.save(name=name, output_dir=path, niv_cut=niv_cut)
    if os.path.exists(marker):
        with suppress(OSError):
            os.remove(os.path.join(path, f"{name}.npy"))


def restrict_chi_phys_to_positive_eigenvalues(
    chi_phys_q_r: FourPoint, comm: MPI.Comm | None = None, floor: float | None = None
) -> tuple[FourPoint, int]:
    r"""
    Regularizes the physical susceptibility per momentum with the two bounds of a bosonic susceptibility, positive
    semi-definite and decreasing in :math:`|\omega|`, touching only the compound blocks that violate them.

    - Static slice: the eigenvalues of the Hermitian part of :math:`(\chi^{(\mathbf{q},\omega=0)}_{r;1234})^{-1}`
      are floored at :math:`+\text{floor}` (the skew-Hermitian part is kept) and the block is inverted back; a
      negative one marks a crossed pole of the Bethe-Salpeter equation. Without an explicit floor it is the smallest
      inverse eigenvalue of the healthy static blocks over all ranks divided by ``_RESTRICTION_PIN_FACTOR``, or
      ``_RESTRICTION_FALLBACK_FLOOR`` when no block is healthy.
    - Finite frequencies: the eigenvalues of the Hermitian part of :math:`\chi^{(\mathbf{q},\omega_n)}_{r;1234}` and
      of its skew-Hermitian part (:math:`i` times a Hermitian matrix) are clipped into
      :math:`[-c_{\mathbf{q}}, c_{\mathbf{q}}]`, :math:`c_{\mathbf{q}}` the largest eigenvalue of the (restricted)
      static block.

    Blocks within both bounds are returned bit for bit, so a healthy susceptibility gives a zero count.

    :param chi_phys_q_r: The physical susceptibility :math:`\chi^{\mathrm{q}}_{r;1234}` (no fermionic frequency
        dimensions).
    :param comm: The MPI communicator (reduces the healthy static maximum over the ranks); ``None`` on one rank.
    :param floor: Lower bound imposed on the eigenvalues of the inverse static susceptibility; ``None`` derives it
        from the healthy static blocks as described above.
    :return: The tuple ``(chi_restricted, n_restricted)`` of the restricted susceptibility as a :class:`FourPoint` in
        the half bosonic frequency range and the number of floored or clipped eigenvalues (a per-iteration
        diagnostic: if it decays to zero during the restricted phase of the self-consistency, releasing the
        restriction is safe).
    """
    chi = chi_phys_q_r.copy().to_half_niw_range().to_compound_indices()
    mat = chi.mat  # [q, w, x1, x2], w = 0 the static slice

    def hermitian_split(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns the Hermitian part of a stack of square blocks and the rest."""
        herm = 0.5 * (blocks + np.conj(np.swapaxes(blocks, -1, -2)))
        return herm, blocks - herm

    herm, skew = hermitian_split(np.linalg.inv(mat[:, 0]))
    eigs, vecs = np.linalg.eigh(herm)
    if floor is None:
        healthy = (eigs > 0.0).all(axis=-1)
        smallest = float(eigs[healthy].min()) if healthy.any() else np.inf
        if comm is not None and comm.size > 1:
            smallest = comm.allreduce(smallest, op=MPI.MIN)
        floor = smallest / _RESTRICTION_PIN_FACTOR if np.isfinite(smallest) else _RESTRICTION_FALLBACK_FLOOR
    low = (eigs < floor).any(axis=-1)
    n_restricted = int((eigs < floor).sum())
    floored = np.einsum("qab,qb,qcb->qac", vecs[low], np.maximum(eigs[low], floor), np.conj(vecs[low]))
    mat[low, 0] = np.linalg.inv(skew[low] + floored)

    bound = np.linalg.eigvalsh(hermitian_split(mat[:, 0])[0])[:, -1][:, None, None]
    herm, skew = hermitian_split(mat[:, 1:])
    # the skew-Hermitian part is i times a Hermitian matrix, bounded the same way
    parts = [np.linalg.eigh(herm), np.linalg.eigh(-1j * skew)]
    clipped = [np.clip(eigs, -bound, bound) for eigs, _ in parts]
    changed = [clip != eigs for clip, (eigs, _) in zip(clipped, parts)]
    outside = changed[0].any(axis=-1) | changed[1].any(axis=-1)
    n_restricted += int(changed[0].sum() + changed[1].sum())
    herm_new, skew_new = (
        np.einsum("kab,kb,kcb->kac", vecs[outside], clip[outside], np.conj(vecs[outside]))
        for clip, (_, vecs) in zip(clipped, parts)
    )
    finite = mat[:, 1:]
    finite[outside] = herm_new + 1j * skew_new
    return chi.to_full_indices(), n_restricted


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
    :math:`\chi^{(\mathbf{q},\omega=0)}_{r;1234}` over all rank-local momenta. It is logged without a verdict: a
    negative value can be physical, e.g. for the charge susceptibility of a genuinely charge-ordered material. Expects
    the object with a compressed momentum dimension and no fermionic frequency dimensions.

    :param chi_phys_q_r: The physical susceptibility :math:`\chi^{\mathrm{q}}_{r;1234}`.
    :return: The minimum eigenvalue as a float, ``inf`` when the object stores no momentum.
    """
    w0 = chi_phys_q_r.niw if chi_phys_q_r.full_niw_range else 0
    n = chi_phys_q_r.n_bands**2
    static = chi_phys_q_r.mat[..., w0].transpose(0, 1, 2, 4, 3).reshape(-1, n, n)
    if static.shape[0] == 0:
        return np.inf
    return float(np.linalg.eigvalsh(0.5 * (static + np.conj(np.swapaxes(static, -1, -2)))).min())


def max_compound_norm(chi_phys_q_r: FourPoint, w_offset: int) -> float:
    r"""
    Returns the largest spectral norm of the compound blocks :math:`\chi^{(\mathbf{q},\omega_n)}_{r;1234}` at the
    bosonic frequency :math:`\omega_n` with :math:`n` = ``w_offset`` (0 is the static slice) over all rank-local
    momenta. Expects the object with a compressed momentum dimension and no fermionic frequency dimensions.

    :param chi_phys_q_r: The physical susceptibility :math:`\chi^{\mathrm{q}}_{r;1234}`.
    :param w_offset: The non-negative bosonic frequency index.
    :return: The largest norm, or 0.0 when the object stores no such frequency or no momentum.
    """
    w = (chi_phys_q_r.niw if chi_phys_q_r.full_niw_range else 0) + w_offset
    if w >= chi_phys_q_r.mat.shape[-1] or chi_phys_q_r.mat.shape[0] == 0:
        return 0.0
    n = chi_phys_q_r.n_bands**2
    blocks = chi_phys_q_r.mat[..., w].transpose(0, 1, 2, 4, 3).reshape(-1, n, n)
    return float(np.linalg.norm(blocks, ord=2, axis=(-2, -1)).max())


def _monitor_chi_phys(chi_phys_q_r: FourPoint, mpi_dist_irrq: MpiDistributor) -> None:
    r"""
    Logs the lowest eigenvalue of the static susceptibility (:func:`min_static_compound_eigenvalue`) and of the
    Hermitian part of its inverse (:meth:`~dgamore.lambda_ops.LambdaAnnealer._static_gap`) as information only, and
    the largest compound norm at the first bosonic frequency against the static one (:func:`max_compound_norm`),
    the only monitor that warns, above ``_FIRST_FREQUENCY_RATIO`` times it. All are reduced over the ranks.

    :param chi_phys_q_r: The physical susceptibility :math:`\chi^{\mathrm{q}}_{r;1234}` (rank-local momenta).
    :param mpi_dist_irrq: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :return: None.
    """
    channel = chi_phys_q_r.channel.value
    comm = mpi_dist_irrq.comm
    min_eig = min_static_compound_eigenvalue(chi_phys_q_r)
    min_inverse_eig = LambdaAnnealer._static_gap(chi_phys_q_r, mpi_dist_irrq)
    norm_w0, norm_w1 = max_compound_norm(chi_phys_q_r, 0), max_compound_norm(chi_phys_q_r, 1)
    if comm.size > 1:
        min_eig = comm.allreduce(min_eig, op=MPI.MIN)
        norm_w0, norm_w1 = comm.allreduce(norm_w0, op=MPI.MAX), comm.allreduce(norm_w1, op=MPI.MAX)
    config.logger.info(
        f"Lowest eigenvalue of the static susceptibility ({channel}): {min_eig:.6f}, and of its inverse: "
        f"{min_inverse_eig:.6f}."
    )
    config.logger.info(f"Largest compound norm of chi_phys ({channel}) at w_1: {norm_w1:.6f} (static {norm_w0:.6f}).")
    if norm_w1 > _FIRST_FREQUENCY_RATIO * norm_w0:
        config.logger.warning(
            f"The physical susceptibility ({channel}) at the first bosonic frequency exceeds its static value "
            f"(largest compound norm {norm_w1:.3f} against {norm_w0:.3f}): a bosonic susceptibility decreases with "
            "|w|, so it has crossed a pole, the static one where its lowest static eigenvalue is negative, else one at "
            "w_1 the static monitor cannot see, and the iteration may be heading to an unphysical fixed point."
        )


def static_stoner_factor(
    gchi0_q_sum: FourPoint, u_loc: LocalInteraction, v_nonloc: Interaction, channel: SpinChannel
) -> float:
    r"""
    Returns the static Stoner factor of a channel, the largest eigenvalue of :math:`-(U_{r} + V^{\mathbf{q}}_{r})
    \chi^{(\mathbf{q},\omega=0)}_{0}` over the rank-local momenta. The RPA susceptibility (see
    :func:`~dgamore.sde_kernels.create_rpa_susceptibility`) has its static pole where the factor reaches 1.

    :param gchi0_q_sum: The frequency-summed bare bubble in the half bosonic frequency range.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param channel: Spin channel, i.e. density or magnetic.
    :return: The largest real part of the eigenvalues, ``-inf`` when the object stores no momentum.
    """
    product = (v_nonloc.as_channel(channel) + u_loc.as_channel(channel)) @ gchi0_q_sum.take_wn_slice(0, 1)
    n = gchi0_q_sum.n_bands**2
    blocks = product.mat[..., 0].transpose(0, 1, 2, 4, 3).reshape(-1, n, n)
    if blocks.shape[0] == 0:
        return -np.inf
    return float(np.linalg.eigvals(-blocks).real.max())


def _monitor_stoner_factor(
    gchi0_q_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    channel: SpinChannel,
    mpi_dist_irrq: MpiDistributor,
    annealer: "LambdaAnnealer | None" = None,
) -> None:
    r"""
    Logs the static Stoner factor of a channel (:func:`static_stoner_factor`), reduced over the ranks, and warns when
    it reaches 1 while no lambda correction, restriction or annealer moves the pole: the RPA susceptibility is then
    past its pole, and the self-energy and pairing vertex built from it are unphysical.

    :param gchi0_q_sum: The frequency-summed bare bubble in the half bosonic frequency range (rank-local momenta).
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param channel: Spin channel, i.e. density or magnetic.
    :param mpi_dist_irrq: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param annealer: The active :class:`LambdaAnnealer`, or ``None`` when annealing is off.
    :return: None.
    """
    factor = static_stoner_factor(gchi0_q_sum, u_loc, v_nonloc, channel)
    if mpi_dist_irrq.comm.size > 1:
        factor = mpi_dist_irrq.comm.allreduce(factor, op=MPI.MAX)
    config.logger.info(f"Static Stoner factor ({channel.value}): {factor:.6f}.")
    moved = (
        annealer is not None
        or config.stabilization.use_chi_phys_restriction
        or config.stabilization.use_lambda_correction
        or config.lambda_correction.perform_lambda_correction
    )
    if factor >= 1 and not moved:
        config.logger.warning(
            f"The static Stoner factor ({channel.value}), the largest eigenvalue of -(U_r + V_r(q)) chi_0(q, w=0), "
            f"is {factor:.3f}: the RPA susceptibility is past its pole, and the self-energy and pairing vertex built "
            "from it are unphysical."
        )


def first_frequency_local_vertex() -> float | None:
    r"""
    Returns the local density vertex that sets the first-frequency pole of the density ladder,

    .. math:: \gamma_{\mathrm{loc}} = 1/\chi^{\omega_1\nu_0\nu_0}_{\mathrm{d,loc}}
        - 1/\chi^{\omega_1\nu_0}_{0,\mathrm{loc}}, \qquad
        \chi^{\omega_1\nu_0}_{0,\mathrm{loc}} = -\beta G(\nu_0) G(\nu_0 - \omega_1),

    at :math:`\nu_0 = \pi T`, from the impurity generalized susceptibility and Green's function the local part saves
    before the loop (``gchi_dens_loc.npy`` on the half bosonic range, ``g_dmft.npy``). See
    :func:`first_frequency_pole_ratio` for its use.

    :return: :math:`\mathrm{Re}\,\gamma_{\mathrm{loc}}`, or ``None`` for more than one band or without the local
        files (the monitor is a diagnostic and never stops a run).
    """
    gchi_path = output_files.npy_path(config.output.output_path, output_files.local_name("gchi", SpinChannel.DENS))
    g_path = output_files.npy_path(config.output.output_path, output_files.G_DMFT)
    if not (os.path.exists(gchi_path) and os.path.exists(g_path)):
        return None
    # one band only; several bands need the orbital-compound block at nu0 and its leading eigenvalue
    gchi = np.load(gchi_path, mmap_mode="r")
    if gchi.shape[0] > 1:
        return None
    g = np.load(g_path, mmap_mode="r").reshape(-1)
    beta, niv, niv_g = config.sys.beta, gchi.shape[-1] // 2, g.shape[-1] // 2
    w1 = 1 if gchi.shape[-3] == config.box.niw_core + 1 else config.box.niw_core + 1
    chi0_loc = -beta * g[niv_g] * g[niv_g - 1]
    return float((1.0 / gchi[0, 0, 0, 0, w1, niv, niv] - 1.0 / chi0_loc).real)


def first_frequency_pole_ratio(sigma: SelfEnergy, mu: float, gamma_loc: float) -> float:
    r"""
    Returns the first-frequency pole ratio of the density ladder,
    :math:`R = \gamma_{\mathrm{loc}}\,\beta\,\mathrm{Re}\langle G^{(\mathbf{k},\nu_0)} G^{(\mathbf{k},\nu_0 -
    \omega_1)}\rangle_{\mathbf{k}}` with :math:`\nu_0 = \pi T`. At :math:`\omega_1` the density Bethe-Salpeter matrix is
    dominated by the bubble element of the pair :math:`(\nu_0, \nu_0 - \omega_1) = (\pi T, -\pi T)`; its 1x1 Schur
    complement :math:`\gamma_{\mathrm{loc}} - 1/(\beta C(\mathbf{q}))`, :math:`C(\mathbf{q}) = \mathrm{Re}\langle
    G^{(\mathbf{k},\nu_0)} G^{(\mathbf{k}+\mathbf{q},\nu_0 - \omega_1)}\rangle_{\mathbf{k}}`, is largest at
    :math:`\mathbf{q} = 0`, so a pole ring lies inside the Brillouin zone exactly when :math:`R \geq 1`. A lattice
    Green's function that has lost local scattering at :math:`\pi T` pushes :math:`R` across it. With uncorrelated
    orbitals :math:`G` is the correlated orbital's element of the inverse Dyson matrix over every orbital.

    :param sigma: The iterate :class:`SelfEnergy` (one correlated band; full-BZ or momentum-local, either momentum
        layout).
    :param mu: Chemical potential :math:`\mu` of ``sigma``.
    :param gamma_loc: The local vertex of :func:`first_frequency_local_vertex`.
    :return: The ratio :math:`R`.
    """
    beta, niv, orbitals = config.sys.beta, sigma.niv, config.sys.correlated_orbitals
    ek = config.lattice.hamiltonian.get_ek()
    s = sigma.mat.reshape(-1, 2 * niv)
    nu0 = np.pi / beta
    if orbitals is None:
        ek = ek.reshape(-1)
        g_nu0 = 1.0 / (1j * nu0 + mu - ek - s[:, niv])
        g_minus_nu0 = 1.0 / (-1j * nu0 + mu - ek - s[:, niv - 1])
    else:
        n, d = ek.shape[-1], orbitals[0]
        g = []
        for iv, s_v in ((1j * nu0, s[:, niv]), (-1j * nu0, s[:, niv - 1])):
            dyson = (iv + mu) * np.eye(n) - ek.reshape(-1, n, n)
            dyson[:, d, d] -= s_v
            g.append(np.linalg.inv(dyson)[:, d, d])
        g_nu0, g_minus_nu0 = g
    return float(gamma_loc * beta * np.mean(g_nu0 * g_minus_nu0).real)


def perform_ornstein_zernike_fit(chi_phys_q_r: FourPoint) -> None:
    r"""
    Fits the static (:math:`\omega = 0`) physical susceptibility to an Ornstein-Zernike form :math:`\chi(\mathbf{q}) = A
    / (\xi^{-2} + (\mathbf{q} - \mathbf{q}_0)^2)` around the antiferromagnetic wave vector :math:`\mathbf{q}_0 = (\pi,
    \pi, 0)`, per orbital combination, and writes the amplitude :math:`A` and the correlation length :math:`\xi` to
    ``oz_coeff.txt``. Non-converging fits are flagged with ``[-1, -1]``.

    :param chi_phys_q_r: The momentum-dependent physical susceptibility :math:`\chi^{\mathrm{q}}_{r}` (irreducible BZ).
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

    # only the static slice is fitted, so it is selected before the unfold
    chi = chi_phys_q_r.copy().to_half_niw_range().take_first_wn()
    chi_mat = chi.map_to_full_bz(config.lattice.k_grid).mat.real
    orb_shape = (config.sys.n_bands,) * 4
    oz_coeffs = np.zeros(orb_shape + (2,), dtype=float)
    failed_orbitals = []

    for idx in np.ndindex(orb_shape):
        mat_slice = chi_mat[..., idx[0], idx[1], idx[2], idx[3]].flatten()
        try:
            coeffs = fit_oz_spin(config.lattice.k_grid, mat_slice) if not np.all(mat_slice == 0) else [0.0, 0.0]
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
    gchi0_q_full_sum: FourPoint, u_loc: LocalInteraction, v_nonloc: Interaction, mpi_dist_irrk: MpiDistributor
):
    r"""
    Calculates and saves the RPA susceptibility (for both density and magnetic channels) of the frequency-summed bubble
    :math:`\chi^{\mathrm{q}}_{0} = \frac{1}{\beta^2} \sum_{\nu} \chi^{\mathrm{q}\nu}_{0}`,
    :math:`\chi^{\mathrm{q}}_{r;\mathrm{RPA}} = \chi^{\mathrm{q}}_{0} (1 + (U_{r} + V^{\mathbf{q}}_{r})
    \chi^{\mathrm{q}}_{0})^{-1}` (see :func:`~dgamore.sde_kernels.create_rpa_susceptibility`). The result is gathered
    to rank 0 and written to file. The file is a diagnostic only, so a channel whose :math:`1 + (U_{r} +
    V^{\mathbf{q}}_{r}) \chi^{\mathrm{q}}_{0}` is singular at any momentum on any rank (an exact Stoner pole) is skipped
    with a warning instead of aborting the run.

    :param gchi0_q_full_sum: The frequency-summed bare bubble over the full box.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :return: None.
    """
    for channel in [SpinChannel.DENS, SpinChannel.MAGN]:
        try:
            chi_rpa_q_r = sde_kernels.create_rpa_susceptibility(gchi0_q_full_sum, u_loc, v_nonloc, channel)
        except np.linalg.LinAlgError:
            chi_rpa_q_r = None
        # every rank has to agree on the skip, else the others would block in the gather
        singular = chi_rpa_q_r is None
        if mpi_dist_irrk.comm.size > 1:
            singular = mpi_dist_irrk.comm.allreduce(singular, op=MPI.MAX)
        if singular:
            config.logger.warning(
                f"The RPA susceptibility ({channel.value}) is singular at a momentum, an exact pole of "
                f"chi_0 (1 + (U_r + V_r(q)) chi_0)^-1; chi_rpa_q_{channel.value} is not written."
            )
            continue
        chi_rpa_q_r.mat = mpi_dist_irrk.gather(chi_rpa_q_r.mat)

        if mpi_dist_irrk.my_rank == 0:
            chi_rpa_q_r.save(name=f"chi_rpa_q_{channel.value}", output_dir=config.output.output_path)

        chi_rpa_q_r.free()
        config.logger.info(f"Calculated RPA susceptibility ({channel.value}).")


def _select_and_apply_lambda_correction(chi_phys_q_r: FourPoint, lambda_previous: dict | None = None) -> FourPoint:
    r"""
    Applies the configured lambda correction to the (rank-0 gathered) physical susceptibility and returns it. The
    correction runs when either the one-shot ``config.lambda_correction.perform_lambda_correction`` or the
    per-iteration ``config.stabilization.use_lambda_correction`` is enabled and is dispatched by the band count:
    single-band input uses the scalar Moriya correction, multi-band input the multi-orbital matrix correction (the
    dispatch is logged once at setup). If neither flag is enabled the susceptibility is returned unchanged.

    :param chi_phys_q_r: The rank-0 gathered physical susceptibility :math:`\chi^{\mathrm{q}}_{r}` in the irreducible
        BZ.
    :param lambda_previous: The per-iteration :math:`\lambda` of each corrected channel, kept by the
        self-consistency loop across its iterations; ``None`` for the one-shot correction (see
        :meth:`~dgamore.lambda_ops.LambdaCorrection.perform`).
    :return: The (possibly corrected) physical susceptibility.
    """
    if config.lambda_correction.perform_lambda_correction or config.stabilization.use_lambda_correction:
        if config.sys.n_bands == 1:
            return LambdaCorrection.perform(chi_phys_q_r, lambda_previous=lambda_previous)
        return MultiOrbitalLambdaCorrection.perform(chi_phys_q_r)
    return chi_phys_q_r


def _write_physical_susceptibility(
    chi_phys_q_r: FourPoint,
    mpi_dist_irrq: MpiDistributor,
    lambda_previous: dict | None = None,
    static_magn: list | None = None,
) -> FourPoint:
    r"""
    Gathers the physical susceptibility on rank 0, which applies the configured lambda correction, writes it to file
    and keeps the static slice of the magnetic channel in ``static_magn``. Collective over the communicator of
    ``mpi_dist_irrq``.

    :param chi_phys_q_r: The physical susceptibility :math:`\chi^{\mathrm{q}}_{r}` (rank-local momenta); its array is
        replaced by the gathered one on rank 0 and by ``None`` elsewhere.
    :param mpi_dist_irrq: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param lambda_previous: The per-iteration :math:`\lambda` of each corrected channel, kept by the
        self-consistency loop across its iterations; ``None`` for the one-shot correction.
    :param static_magn: Holder of the static magnetic susceptibility for the loop's Ornstein-Zernike fit (see
        :func:`perform_ornstein_zernike_fit`): on rank 0 the magnetic channel replaces its content with the
        irreducible-BZ :math:`\omega = 0` slice; ``None`` keeps nothing.
    :return: The corrected, gathered susceptibility on rank 0; ``chi_phys_q_r`` without its array elsewhere.
    """
    chi_phys_q_r.mat = mpi_dist_irrq.gather(chi_phys_q_r.mat)
    if mpi_dist_irrq.comm.rank == 0:
        chi_phys_q_r = _select_and_apply_lambda_correction(chi_phys_q_r, lambda_previous)
        chi_phys_q_r.save(
            name=output_files.channel_name(output_files.CHI_PHYS_Q, chi_phys_q_r.channel),
            output_dir=config.output.output_path,
        )

        # the loop fits the Ornstein-Zernike form once after its last proposal, so only the static slice is kept
        if chi_phys_q_r.channel == SpinChannel.MAGN and static_magn is not None:
            static_magn[:] = [chi_phys_q_r.copy().to_half_niw_range().take_wn_slice(0, 1)]
    config.logger.info(f"Saved physical susceptibility ({chi_phys_q_r.channel.value}) to file.")
    return chi_phys_q_r


def _finish_physical_susceptibility(
    chi_phys_q_r: FourPoint,
    mpi_dist_irrq: MpiDistributor,
    annealer: "LambdaAnnealer | None" = None,
    save_eliashberg: bool = False,
    lambda_previous: dict | None = None,
    static_magn: list | None = None,
    last_chi: dict | None = None,
) -> FourPoint:
    r"""
    Returns the physical susceptibility the self-energy kernel reads: the annealer's boson mass and the restriction are
    applied on every rank, the rank-0 gathered susceptibility gets the configured lambda correction, is written to file
    (rank 0 keeps the static slice of the magnetic channel in ``static_magn``) and scattered back, the pole monitor
    reads it and, with ``save_eliashberg``, every rank writes its momenta for the Eliashberg step. Without a lambda
    correction a given ``last_chi`` takes every rank's copy of its momenta instead of the gather, the write and the
    scatter, since the susceptibility only changes through the correction (see :func:`_write_physical_susceptibility`).

    :param chi_phys_q_r: The physical susceptibility :math:`\chi^{\mathrm{q}}_{r}` (rank-local momenta).
    :param mpi_dist_irrq: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param annealer: The active :class:`LambdaAnnealer` (its boson mass is applied to the physical susceptibility),
        or ``None`` when annealing is off.
    :param save_eliashberg: Whether to write this rank's physical susceptibility for the Eliashberg step.
    :param lambda_previous: The per-iteration :math:`\lambda` of each corrected channel, kept by the
        self-consistency loop across its iterations; ``None`` for the one-shot correction.
    :param static_magn: Holder of the static magnetic susceptibility for the loop's Ornstein-Zernike fit (see
        :func:`perform_ornstein_zernike_fit`): on rank 0 the magnetic channel replaces its content with the
        irreducible-BZ :math:`\omega = 0` slice; ``None`` keeps nothing.
    :param last_chi: Holder of the loop's latest physical susceptibility per channel, which the loop writes once
        after its last proposal, or ``None`` to write it in every call.
    :return: The treated physical susceptibility (rank-local momenta).
    """
    logger = config.logger

    if annealer is not None:
        chi_phys_q_r = annealer.apply(chi_phys_q_r, mpi_dist_irrq)

    if config.stabilization.use_chi_phys_restriction:
        chi_phys_q_r, n_restricted = restrict_chi_phys_to_positive_eigenvalues(chi_phys_q_r, mpi_dist_irrq.comm)
        if mpi_dist_irrq.comm.size > 1:
            n_restricted = mpi_dist_irrq.comm.allreduce(n_restricted)
        logger.warning(
            f"Restricted physical susceptibility ({chi_phys_q_r.channel.value}): {n_restricted} eigenvalues "
            "restricted (static inverse floored, finite-frequency values clipped to the static maximum). Releasing "
            "the restriction is only safe once this count decays to zero."
        )

    logger.log_memory_usage(
        f"Physical susceptibility ({chi_phys_q_r.channel.value})", chi_phys_q_r, mpi_dist_irrq.comm.size
    )

    lambda_on = config.lambda_correction.perform_lambda_correction or config.stabilization.use_lambda_correction
    if last_chi is not None and not lambda_on:
        last_chi[chi_phys_q_r.channel] = chi_phys_q_r.copy()
    else:
        chi_phys_q_r = _write_physical_susceptibility(chi_phys_q_r, mpi_dist_irrq, lambda_previous, static_magn)
        mpi_utils.idle_barrier(mpi_dist_irrq.comm)  # the other ranks sleep while rank 0 corrects and writes
        chi_phys_q_r.mat = mpi_dist_irrq.scatter(chi_phys_q_r.mat)

    _monitor_chi_phys(chi_phys_q_r, mpi_dist_irrq)

    if save_eliashberg:
        save_rank_file(
            chi_phys_q_r, output_files.rank_name(output_files.CHI_PHYS_Q, mpi_dist_irrq.comm.rank, chi_phys_q_r.channel)
        )
    return chi_phys_q_r


def calculate_sigma_kernel_r_q(
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    mpi_dist_irrq: MpiDistributor,
    annealer: "LambdaAnnealer | None" = None,
    save_eliashberg: bool = False,
    lambda_previous: dict | None = None,
    kernel: FourPoint | None = None,
    weight: float = 1.0,
    involution: np.ndarray | None = None,
    static_magn: list | None = None,
    last_chi: dict | None = None,
    node_comm: MPI.Comm | None = None,
) -> FourPoint | None:
    r"""
    Adds the kernel for the self-energy calculation in a specific spin channel, times ``weight``, to ``kernel`` in
    place. Solves the ladder of :math:`\Gamma_{r}` without the :math:`U_{r}` shift (see
    :func:`~dgamore.sde_kernels.create_auxiliary_chi_r_q_sum` with ``subtract_u`` False) and builds from it the
    three-leg vertex, the shell-corrected physical susceptibility (see
    :func:`~dgamore.sde_kernels.create_shell_corrected_chi_from_gamma_ladder`) and the kernel (see
    :func:`~dgamore.sde_kernels.create_kernel_middle` and :func:`~dgamore.sde_kernels.calculate_pole_free_kernel_r_q`),
    so no inversion meets the zero eigenvalues the auxiliary Bethe-Salpeter matrix can have where the physical
    susceptibility is regular. Also performs a :math:`\lambda`-correction on the physical susceptibility if specified in
    the config (dispatched by the band count). Saves the physical susceptibility to file and, with ``save_eliashberg``,
    the per-rank Eliashberg intermediates: the three-leg vertex as ``vrg_q_{channel}_rank_{rank}`` (on the pp box the
    Eliashberg step reads, on the whole core box with ``eliashberg.save_fq``) and the matrix between the three-leg
    vertices of the ladder vertex (see :func:`~dgamore.sde_kernels.create_ladder_vertex_middle`) as
    ``vrg_middle_q_{channel}_rank_{rank}``.

    The three-leg vertex and the channel kernel are built in momentum groups of
    :func:`~dgamore.memory_estimator.kernel_momentum_group` on views of the rank's arrays: each group of the summed
    ladder susceptibility is overwritten with its three-leg vertex, and each group's kernel is added to its window
    of ``kernel``. Beside ``kernel``, the inverse bubble and the bosonic matrices the step holds a single one-fermion
    block (the summed susceptibility, then the vertex) and one group's temporaries, and every momentum gets the bits of
    the ungrouped products.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}`.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` (core box).
    :param gchi0_q_full_sum: The frequency-summed bare bubble over the full box.
    :param gchi0_q_core_sum: The frequency-summed bare bubble over the core box.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param mpi_dist_irrq: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param annealer: The active :class:`LambdaAnnealer` (its boson mass is applied to the physical susceptibility),
        or ``None`` when annealing is off.
    :param save_eliashberg: Whether to write this rank's Gamma-ladder three-leg vertex and vertex middle for the
        Eliashberg step (the self-consistency loop asks for it once per run).
    :param lambda_previous: The per-iteration :math:`\lambda` of each corrected channel, kept by the
        self-consistency loop across its iterations; ``None`` for the one-shot correction.
    :param kernel: The self-energy kernel accumulated over the channels (contraction layout), or ``None`` to skip the
        channel kernel (the Eliashberg-only pass).
    :param weight: The channel's weight in the sum of the kernels.
    :param involution: The orbital permutation that transposes the dispersion (see
        :func:`~dgamore.sde_kernels.orbital_involution`), or None; passed to
        :func:`~dgamore.sde_kernels.create_auxiliary_chi_r_q_sum`.
    :param static_magn: Holder of the static magnetic susceptibility for the loop's Ornstein-Zernike fit (see
        :func:`_finish_physical_susceptibility`), or ``None``.
    :param last_chi: Holder of the loop's latest physical susceptibility per channel (see
        :func:`_finish_physical_susceptibility`), or ``None``.
    :param node_comm: The node-local communicator whose ranks may share the Bethe-Salpeter slices (see
        :func:`~dgamore.sde_kernels.create_auxiliary_chi_r_q_sum`), or None.
    :return: ``kernel``.
    """
    logger = config.logger

    gchi_ladder_q_r_sum = sde_kernels.create_auxiliary_chi_r_q_sum(
        gamma_r, gchi0_q_inv, u_loc, involution=involution, subtract_u=False, node_comm=node_comm
    )

    mpi_dist_irrq.barrier()

    # reported as the full two-fermion auxiliary susceptibility, which the summed object stands in for
    logger.log_memory_usage(
        f"Auxiliary susceptibility ({gchi_ladder_q_r_sum.channel.value})",
        gchi_ladder_q_r_sum,
        mpi_dist_irrq.comm.size,
        scale=2 * config.box.niv_core,
    )
    logger.info(f"Non-Local auxiliary susceptibility ({gchi_ladder_q_r_sum.channel.value}) calculated.")

    x_g = gchi_ladder_q_r_sum.sum_over_all_vn(config.sys.beta)

    n_q = gchi0_q_inv.current_shape[0]
    step = memory_estimator.kernel_momentum_group(n_q)
    windows = [(start, min(n_q, start + step)) for start in range(0, n_q, step)]
    vrg_q_r = _three_leg_vertex_in_place(gchi_ladder_q_r_sum, gchi0_q_inv, windows)
    del gchi_ladder_q_r_sum

    logger.info(f"Non-local three-leg vertex gamma^wv ({vrg_q_r.channel.value}) done.")
    logger.log_memory_usage(f"Three-leg vertex ({vrg_q_r.channel.value})", vrg_q_r, mpi_dist_irrq.comm.size)

    chi_phys_q_r = sde_kernels.create_shell_corrected_chi_from_gamma_ladder(
        x_g, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
    )

    logger.info(f"Updated non-local susceptibility chi^q ({chi_phys_q_r.channel.value}) with asymptotic correction.")

    # the finishing step overwrites its input, and the middles read chi before it; the Eliashberg step reads the
    # middles, so the per-rank chi is not written
    chi_shell = chi_phys_q_r.copy()
    chi_phys_q_r = _finish_physical_susceptibility(
        chi_phys_q_r,
        mpi_dist_irrq,
        annealer,
        lambda_previous=lambda_previous,
        static_magn=static_magn,
        last_chi=last_chi,
    )
    parts = (gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc, chi_shell, chi_phys_q_r)

    if kernel is not None:
        middle_q_r = sde_kernels.create_kernel_middle(x_g, *parts)
        # one statement per group, so its objects are released inside the deferred collection
        with deferred_collection():
            for start, stop in windows:
                kernel.take_q_index_slice(start, stop, copy=False).add(
                    sde_kernels.calculate_pole_free_kernel_r_q(
                        vrg_q_r.take_q_index_slice(start, stop, copy=False),
                        middle_q_r.take_q_index_slice(start, stop, copy=False),
                        v_nonloc.take_q_index_slice(start, stop, copy=False),
                        u_loc,
                    ).scale(weight),
                    copy=False,
                )
    if save_eliashberg:
        channel, rank = vrg_q_r.channel.value, mpi_dist_irrq.comm.rank
        # the pp band reads the three-leg vertex on the pp box only (the box of eliashberg_solver.solve); the streamed
        # full vertex reads the whole core box
        niv_pp = None if config.eliashberg.save_fq else min(config.box.niw_core // 2, config.box.niv_core // 2)
        save_rank_file(vrg_q_r, output_files.rank_name(output_files.VRG_Q, rank, vrg_q_r.channel), niv_pp)
        middle_name = output_files.rank_name(output_files.VRG_MIDDLE_Q, rank, vrg_q_r.channel)
        save_rank_file(sde_kernels.create_ladder_vertex_middle(x_g, *parts), middle_name)
        logger.info(f"Saved the Gamma-ladder three-leg vertex and vertex middle ({channel}) for the Eliashberg step.")
    return kernel


def _three_leg_vertex_in_place(gchi_q_r_sum: FourPoint, gchi0_q_inv: FourPoint, windows: list) -> FourPoint:
    r"""
    Overwrites a frequency-summed susceptibility with its three-leg vertex (see
    :func:`~dgamore.sde_kernels.create_vrg_r_q`) one momentum window at a time, on views, so every momentum gets the
    bits of the ungrouped product.

    :param gchi_q_r_sum: The frequency-summed susceptibility; overwritten.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` (core box).
    :param windows: The momentum windows ``(start, stop)`` of the rank.
    :return: ``gchi_q_r_sum``, now the three-leg vertex.
    """
    with deferred_collection():
        for start, stop in windows:
            sde_kernels.create_vrg_r_q(
                gchi_q_r_sum.take_q_index_slice(start, stop, copy=False),
                gchi0_q_inv.take_q_index_slice(start, stop, copy=False),
                copy=False,
            )
    return gchi_q_r_sum


def _has_point_group_covariance(mat: np.ndarray, comm: MPI.Comm | None = None, bound: float | None = None) -> bool:
    r"""
    Returns whether a momentum-resolved orbital matrix (a dispersion, Green's function or self-energy) obeys the grid's
    point group, :math:`G^{(g\mathbf{k},\nu)} = U_g G^{(\mathbf{k},\nu)} U_g^\dagger` for every element, to ``bound``
    relative to its largest entry; False on a grid whose contraction does not average its results over a group (see
    :class:`~dgamore.sde_kernels.ColumnSdeStream`). A momentum-local matrix (a single momentum) obeys it when every
    rotation leaves it unchanged. Collective over ``comm``: every rank checks its share of the momenta in blocks of rows
    against the rows of their images, so no copy of its share is held, and all ranks return the same answer.

    :param mat: The matrix ``[k..., o1, o2, v]`` on the whole grid or on a single momentum, whole on every rank; only
        read.
    :param comm: The MPI communicator, or ``None`` for a single rank.
    :param bound: The relative bound; ``None`` takes ``1e-6`` for single and ``1e-12`` for double precision, above
        the rounding of the storage and of the rotations, which are known to about ``1e-14``.
    :return: Whether the matrix obeys every element on every momentum and frequency.
    """
    actions = config.lattice.k_grid.auto_point_group()
    if actions is None:
        return False
    nb = mat.shape[-2]
    mat = np.reshape(mat, (-1, nb, nb, mat.shape[-1]), copy=False)
    if bound is None:
        bound = 1e-6 if mat.dtype == np.complex64 else 1e-12
    maps, us = actions[0][1:], actions[1][1:].astype(mat.real.dtype)
    size, rank = (comm.size, comm.rank) if comm is not None else (1, 0)
    first, last = rank * len(mat) // size, (rank + 1) * len(mat) // size
    diff = scale = 0.0
    for r0 in range(first, last, 64):
        rows = mat[r0 : min(r0 + 64, last)]
        scale = max(scale, float(np.abs(rows).max()))
        for idx, u in zip(maps, us):
            rotated = np.einsum("ab,kbcv,dc->kadv", u, rows, u, optimize=True)
            diff = max(diff, float(np.abs(mat[idx[r0 : r0 + len(rows)]] - rotated).max()))
    if size > 1:
        diff, scale = (max(values) for values in zip(*comm.allgather((diff, scale))))
    return diff <= bound * scale


def _has_time_reversal_covariance(mat: np.ndarray, bound: float) -> bool:
    r"""
    Returns whether a self-energy on the whole grid obeys the time reversal of real hoppings,
    :math:`\Sigma^{(\mathbf{k},\nu)}_{12} = \Sigma^{(-\mathbf{k},\nu)}_{21}`, to ``bound`` relative to its largest
    entry. It checks blocks of rows against the rows of their time-reversed partners, so no copy of it is held.

    :param mat: The self-energy ``[k, o1, o2, v]`` (compressed momentum axis) on the configured grid; only read.
    :param bound: The relative bound.
    :return: Whether the self-energy obeys the relation on every momentum and frequency.
    """
    partners = BubbleGenerator._reversed_momenta(config.lattice.k_grid.nk)
    diff = scale = 0.0
    for r0 in range(0, len(mat), 64):
        rows = mat[r0 : r0 + 64]
        scale = max(scale, float(np.abs(rows).max()))
        diff = max(diff, float(np.abs(rows - mat[partners[r0 : r0 + 64]].swapaxes(1, 2)).max()))
    return diff <= bound * scale


def _lattice_sum_momenta(
    sigma: SelfEnergy,
    donor: SelfEnergy,
    mpi_dist_irrk: MpiDistributor,
    mpi_dist_fullbz: MpiDistributor,
    comm: MPI.Comm,
) -> tuple[np.ndarray, MpiDistributor, tuple[np.ndarray, np.ndarray] | None]:
    r"""
    Returns the momenta the lattice sums of the chemical-potential search and of the occupation run over, as the
    ``sigma_mat``, ``mpi_dist`` and ``wedge`` of :func:`~dgamore.greens_function.update_mu`: the irreducible momenta
    with the sizes of their stars when the self-energy acts on every orbital of the dispersion and the dispersion,
    the self-energy and its momentum-local shell donor obey the grid's point group (see
    :func:`_has_point_group_covariance`; the dispersion to the bound of the self-energy's precision), every momentum
    of the grid otherwise. Collective over ``comm``.

    :param sigma: The self-energy ``[k, o1, o2, v]`` (compressed momentum axis), whole on every rank; only read.
    :param donor: The momentum-local self-energy of the frequencies beyond those of ``sigma``; only read.
    :param mpi_dist_irrk: MPI distributor over the irreducible momenta (see :class:`MpiDistributor`).
    :param mpi_dist_fullbz: MPI distributor over the momenta of the whole grid.
    :param comm: The MPI communicator.
    :return: This rank's rows of the self-energy (a copy of its irreducible rows, or a view of its share of the
        grid), the distributor of those rows and the irreducible indices with their star sizes (``None`` for the
        whole grid).
    """
    k_grid = config.lattice.k_grid
    if config.sys.correlated_orbitals is None and k_grid.auto_point_group() is not None:
        bound = 1e-6 if sigma.mat.dtype == np.complex64 else 1e-12
        inputs = ((config.lattice.hamiltonian.get_ek()[..., None], bound), (donor.mat, None), (sigma.mat, None))
        if all(_has_point_group_covariance(mat, comm, bound) for mat, bound in inputs):
            wedge = (k_grid.irrk_ind, k_grid.irrk_count)
            return sigma.mat[k_grid.irrk_ind[mpi_dist_irrk.my_slice]], mpi_dist_irrk, wedge
    return sigma.mat[mpi_dist_fullbz.my_slice], mpi_dist_fullbz, None


def _sde_tail_moment(kernel: FourPoint, mpi_dist_irrk: MpiDistributor) -> np.ndarray | None:
    r"""
    Returns the coefficient :math:`\Sigma_{1}` of :math:`-1/\imath\nu` that the contraction of
    :func:`~dgamore.sde_kernels.run_column_sde` approaches for a kernel without fermionic frequency where the Green's
    function falls off as :math:`\delta_{ab}/\imath\nu`,

    .. math:: \Sigma_{1;12} = \frac{1}{2\beta n_{\mathbf{q}}} \sum_{\mathrm{q}} \sum_{a} K^{\mathrm{q}}_{a12a},

    with the momentum sum over the full BZ, unfolded as in the contraction one bosonic frequency at a time on rank 0,
    and the bosonic sum over the contraction's box, its negative half the conjugate of the positive one at
    :math:`\mathbf{R} = 0` (time reversal, see :class:`~dgamore.sde_kernels.ColumnSdeStream`). The partial trace is
    taken before the unfold: under it :math:`T_{12} = \sum_{a} K_{a12a}` goes to :math:`U^{*} T U^{T}` (the unfold keeps
    no anti-symmetry), so :math:`T^{T}` unfolds as a two-index object, :math:`n_{\mathrm{o}}^{2}` numbers per momentum
    instead of :math:`n_{\mathrm{o}}^{4}`.

    :param kernel: The rank-local irreducible-BZ kernel ``[q, o1, o2, o3, o4, w]`` (half bosonic range, no fermionic
        axis, compressed momentum axis); only read.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :return: :math:`\Sigma_{1}`, shape ``[o1, o2]``, on rank 0; ``None`` elsewhere.
    """
    k_grid, moment = config.lattice.k_grid, None
    for w in range(kernel.current_shape[-1]):
        trace = sum(kernel.mat[:, a, :, :, a, w] for a in range(kernel.n_bands))
        column = mpi_dist_irrk.gather(trace.swapaxes(1, 2)[..., None])
        if column is None:
            continue
        full = TwoPoint(column, config.lattice.nk, False, True).map_to_full_bz(k_grid)
        local = full.mat[..., 0].sum(axis=0).T / k_grid.nk_tot
        contribution = local if w == 0 else local + local.conj()
        moment = contribution if moment is None else moment + contribution
    return None if moment is None else moment / (2.0 * config.sys.beta)


def _rpa_shell_donor(moment: np.ndarray, niv: int) -> SelfEnergy:
    r"""
    Returns the momentum-local shell donor of RPA mode on ``niv`` positive frequencies, the tail
    :math:`-\Sigma_{1}/\imath\nu` of its self-energy; the tail's constant, the Hartree-Fock term, enters as the shell
    offset (see :meth:`SelfEnergy.concatenate_self_energies`).

    :param moment: The coefficient :math:`\Sigma_{1}`, shape ``[o1, o2]`` (see :func:`_sde_tail_moment`).
    :param niv: Number of positive fermionic frequencies.
    :return: The donor :class:`SelfEnergy` (one momentum, full fermionic range, moments not fitted).
    """
    iv = 1j * MFHelper.vn(niv, config.sys.beta)
    return SelfEnergy((-moment[..., None] / iv)[None, None, None], calc_smom=False, beta=config.sys.beta)


def get_starting_sigma(default_sigma: SelfEnergy) -> tuple[SelfEnergy, int, bool]:
    r"""
    Retrieves the starting self-energy of the self-consistency cycle from a previous run. With
    ``use_interpolated_sigma`` this is the predecessor's final self-energy re-gridded to this temperature,
    ``sigma_dga_interpolated_beta<b>_niv<n>.npy`` (the file with ``<b>`` closest to the current :math:`\beta` if
    several exist); otherwise it is the raw iterate ``sigma_dga_iteration_<i>.npy`` with the highest ``<i>``. The
    iteration count comes from the raw iterates in both cases, read from the run's ``Sigma_Iterates`` subfolder or,
    for a run written before that subfolder existed, from the run folder itself. A finished run without raw iterates
    (one that wrote ``sigma_dga.npy``, the same array as its last iterate) is resumed from that file, or from its
    interpolated file with ``use_interpolated_sigma``, with a warning: its iteration count is unknown, so the loop
    starts at iteration 1. Without a usable file the DMFT self-energy is returned.

    Where the DMFT self-energy obeys the grid's point group (see :func:`_has_point_group_covariance`), a starting
    self-energy that does not obey it to the accuracy of the rotations, ``1e-12`` of its largest entry in either
    precision, is unfolded from its irreducible momenta (see :meth:`SelfEnergy.unfold_irreducible_momenta`), so the
    self-energy contraction and the lattice sums run over the irreducible momenta from the first iteration on (see
    :func:`~dgamore.sde_kernels.run_column_sde` and :func:`_lattice_sum_momenta`). Such a start is, for example, one
    interpolated onto a grid whose point group shears the grid axes: the Fourier interpolation along each axis does not
    commute with those operations. A start that obeys the group to that accuracy keeps its bits.

    Where the cycle averages its proposals with their time-reversed partners (several orbitals and a dispersion with
    real hoppings, see :func:`~dgamore.sde_kernels.has_time_reversal` and :func:`calculate_sigma_proposal`), a starting
    self-energy that does not obey time reversal to ``1e-12`` of its largest entry (see
    :func:`_has_time_reversal_covariance`) is then averaged with its time-reversed partner as well (see
    :meth:`SelfEnergy.symmetrize_time_reversal`), so the bubble takes its orbital-transpose and crossing routes from the
    first iteration on (see :meth:`~dgamore.bubble_gen.BubbleGenerator.create_generalized_chi0_q_fft`). Such a start is,
    for example, one in which an inversion that permutes the orbitals, such as the layer swap of a bilayer, holds only
    to rounding: the unfold onto the point group carries that rounding into time reversal. A start that obeys time
    reversal to that accuracy keeps its bits.

    :param default_sigma: The fallback (DMFT) :class:`SelfEnergy` used when no previous result is found; whether it
        obeys the grid's point group decides the unfold of the starting self-energy.
    :return: A tuple of the starting :class:`SelfEnergy` (cut to the core box and interpolated onto the k-grid), the
        iteration number it was taken from (0 if none found) and whether it comes from the previous run (a warm
        start).
    """
    previous_sc_path = config.self_consistency.previous_sc_path

    if previous_sc_path is None or previous_sc_path == "" or not os.path.exists(previous_sc_path):
        if config.self_consistency.use_interpolated_sigma:
            config.logger.warning(
                "use_interpolated_sigma is set but previous_sc_path is empty or missing; starting from DMFT."
            )
        return default_sigma, 0, False

    pattern = "sigma_dga_iteration_*.npy"
    iterates = glob.glob(
        os.path.join(previous_sc_path, config.self_consistency.sigma_iterates_subfolder_name, pattern)
    ) or glob.glob(os.path.join(previous_sc_path, pattern))
    files = {int(match.group(1)): f for f in iterates if (match := re.search(r"iteration_(\d+)\.npy$", f))}
    max_iter = max(files, default=0)
    max_file = files.get(max_iter, output_files.npy_path(previous_sc_path, output_files.SIGMA_DGA))
    if not files:
        if not os.path.isfile(max_file):
            return default_sigma, 0, False
        config.logger.warning(
            f"No {pattern} in {previous_sc_path}: its iteration count is unknown, so the loop starts at iteration 1."
        )

    if config.self_consistency.use_interpolated_sigma:
        beta_regex = re.compile(r"sigma_dga_interpolated_beta([0-9.eE+-]+)_niv\d+\.npy$")
        candidates = glob.glob(os.path.join(previous_sc_path, "sigma_dga_interpolated_beta*_niv*.npy"))
        betas = [
            (abs(float(match.group(1)) - config.sys.beta), f) for f in candidates if (match := beta_regex.search(f))
        ]
        if not betas:
            config.logger.warning(f"No interpolated self-energy found in {previous_sc_path}; starting from DMFT.")
            return default_sigma, 0, False
        max_file = min(betas, key=lambda x: x[0])[1]
    config.logger.info(f"Starting the self-consistency from {max_file} (iteration {max_iter}).")
    mat = np.load(max_file)
    sigma = SelfEnergy(mat, mat.shape[:3], True, False, beta=config.sys.beta).cut_niv(config.box.niv_core)
    sigma = sigma.interpolate_q_grid(config.lattice.k_grid.nk, False)
    if _has_point_group_covariance(default_sigma.mat):
        if not _has_point_group_covariance(sigma.compress_q_dimension().mat, bound=1e-12):
            config.logger.info("The starting self-energy obeys the point group only approximately: it is unfolded.")
            sigma.unfold_irreducible_momenta(config.lattice.k_grid)
        sigma.decompress_q_dimension()
    if sigma.n_bands > 1 and sde_kernels.has_time_reversal(config.lattice.hamiltonian.get_ek()):
        if not _has_time_reversal_covariance(sigma.compress_q_dimension().mat, bound=1e-12):
            config.logger.info(
                "The starting self-energy obeys time reversal only approximately: it is averaged with its "
                "time-reversed partner."
            )
            sigma.symmetrize_time_reversal()
        sigma.decompress_q_dimension()
    return sigma, max_iter, True


def _init_mu_history(warm_start: bool) -> list[float]:
    r"""
    Seeds the chemical-potential history for the self-consistency loop. For a fresh run (no warm start) the history
    starts at the current (DMFT) chemical potential :math:`\mu`. When resuming from a previous self-consistency
    calculation it is seeded with that run's last :math:`\mu` (from ``mu_history.npy``) and the global ``config.sys.mu``
    is synced to it: otherwise ``config.sys.mu`` would stay at the DMFT value while ``giwk_full`` is built with the
    previous run's :math:`\mu`, and any quantity computed from the global (e.g. the lattice filling in
    :meth:`GreensFunction.get_fill_nonlocal`, which now reads ``self._mu``) would use an inconsistent chemical potential.
    That value is only the starting guess: :func:`calculate_self_energy_q` re-solves it so the starting self-energy has
    the DMFT lattice filling. A previous run without ``mu_history.npy`` seeds the history with the current
    :math:`\mu`, with a warning.

    :param warm_start: Whether the loop starts from a previous run's self-energy (see :func:`get_starting_sigma`).
    :return: The single-element chemical-potential history list.
    """
    if not warm_start:
        return [config.sys.mu]

    path = os.path.join(config.self_consistency.previous_sc_path, "mu_history.npy")
    if not os.path.isfile(path):
        config.logger.warning(f"No {path}; the chemical potential search starts from mu = {config.sys.mu}.")
        return [config.sys.mu]
    previous_mu = float(np.load(path)[-1])
    config.sys.mu = previous_mu
    return [previous_mu]


def _save_sigma_iteration(sigma: SelfEnergy, base_name: str, current_iter: int) -> None:
    """
    Saves a self-energy of the current iteration into the run's ``Sigma_Iterates`` subfolder as
    ``<base_name>_iteration_<i>``.

    :param sigma: The :class:`SelfEnergy` to save (the mixed iterate or the raw proposal).
    :param base_name: File-name prefix.
    :param current_iter: The current self-consistency iteration number.
    :return: None.
    """
    sigma.decompress_q_dimension().save(
        name=f"{base_name}_iteration_{current_iter}", output_dir=config.output.sigma_iterates_path
    )
    config.logger.info(f"Saved {base_name} for iteration {current_iter}.")


def _prune_sigma_iterates(keep_from: int) -> None:
    """
    Deletes the files ``<base_name>_iteration_<i>.npy`` with ``i < keep_from`` from the run's ``Sigma_Iterates``
    subfolder (the mixed iterates and, with Jacobian stabilization, the proposals).

    :param keep_from: The first iteration whose files are kept.
    :return: None.
    """
    for f in glob.glob(os.path.join(config.output.sigma_iterates_path, "*_iteration_*.npy")):
        if (match := re.search(r"_iteration_(\d+)\.npy$", f)) and int(match.group(1)) < keep_from:
            os.remove(f)


def _share_array_per_node(
    arr: np.ndarray | None, comm: MPI.Comm, node_comm, roots_comm
) -> tuple[np.ndarray, MPI.Win | None]:
    r"""
    Hands rank 0's array to every rank as a view of one MPI shared-memory window per node: every node allocates its
    window (see :func:`dgamore.mpi_utils.allocate_node_shared_array`), the other node roots receive the array straight
    into theirs through a chunked broadcast over ``roots_comm`` while rank 0 sends from its own array, which it copies
    into its window afterwards, and the rest of each node sleeps (see :func:`dgamore.mpi_utils.idle_barrier`). Every
    other rank must treat the buffer as read-only.

    :param arr: The array to share; only read on rank 0.
    :param comm: The MPI communicator.
    :param node_comm: The node-local (shared-memory) communicator, or ``None`` on a single rank.
    :param roots_comm: Communicator over exactly the node-root ranks (global rank 0 first), or ``None`` on a single
        rank.
    :return: The tuple ``(array, win)`` of this rank's view of its node's buffer and the MPI shared-memory window
        (``None`` for a single-rank node); free the window via :func:`~dgamore.mpi_utils.free_shared_window` once no
        rank reads it.
    """
    shape, dtype = comm.bcast((arr.shape, arr.dtype) if comm.rank == 0 else None, root=0)
    shared, win = mpi_utils.allocate_node_shared_array(node_comm, shape, dtype)
    arr = np.ascontiguousarray(arr) if comm.rank == 0 else None
    if node_comm is not None and node_comm.rank == 0:
        mpi_utils.bcast_rows_into(roots_comm, arr if comm.rank == 0 else shared, root=0)
    # rank 0 sends from its own array first, so the other roots never wait for its copy into the window
    if comm.rank == 0 and win is None:
        shared = arr
    elif comm.rank == 0:
        shared[...] = arr
    if node_comm is not None:
        mpi_utils.idle_barrier(node_comm)
    return shared, win


def _share_sigma_per_node(
    sigma: SelfEnergy | None, comm: MPI.Comm, node_comm, roots_comm
) -> tuple[SelfEnergy, MPI.Win | None]:
    r"""
    Hands rank 0's full-BZ self-energy to every rank as a view of one per-node MPI shared-memory window: rank 0
    compresses the momenta and broadcasts the object without its array (a small pickled blob), and the array reaches
    every node's window once (see :func:`_share_array_per_node`), so only the node roots ever receive it. Every rank
    gets its own :class:`SelfEnergy` object; only the buffer is shared and must be treated as read-only (every consumer
    copies via ``cut_niv``/``copy``/``concatenate`` before writing).

    :param sigma: The :class:`SelfEnergy` to share; only read on rank 0.
    :param comm: The MPI communicator.
    :param node_comm: The node-local (shared-memory) communicator.
    :param roots_comm: Communicator over exactly the node-root ranks (global rank 0 first).
    :return: The tuple ``(sigma, win)`` of this rank's :class:`SelfEnergy` viewing its node's buffer and the MPI
        shared-memory window (``None`` for a single-rank node); free the window via
        :func:`~dgamore.mpi_utils.free_shared_window` once no rank reads the buffer anymore.
    """
    mat = None
    if comm.rank == 0:
        sigma = sigma.compress_q_dimension()
        mat, sigma.mat = sigma.mat, None
    sigma = pickle.loads(comm.bcast(pickle.dumps(sigma) if comm.rank == 0 else None, root=0))
    sigma.mat, win = _share_array_per_node(mat, comm, node_comm, roots_comm)
    return sigma, win


@dataclass
class _RowLayout:
    r"""
    The momentum rows of the loop's mixing: every rank owns the contiguous block ``rows`` of the flat momenta for the
    whole run (its rows of the mixing history), and every node the contiguous block ``bounds[n]:bounds[n + 1]`` of
    the rows of its ranks, nodes numbered as their roots in ``roots_comm`` (rank 0's node first).

    :ivar rows: This rank's rows.
    :ivar owners: The communicator over the ranks that own rows, in the order of their rows (``None`` on a rank
        without rows).
    :ivar bounds: The first row of every node and the number of rows, shape ``[n_nodes + 1]``.
    :ivar node: The number of this rank's node.
    :ivar node_comm: The node-local communicator, or ``None`` on a single rank.
    :ivar roots_comm: The communicator over the node roots, or ``None`` on a single rank.
    """

    rows: slice
    owners: MPI.Comm | None
    bounds: np.ndarray
    node: int
    node_comm: MPI.Comm | None
    roots_comm: MPI.Comm | None


def _row_layout(comm: MPI.Comm, node_comm, roots_comm, n_rows: int, one_owner: bool) -> _RowLayout:
    r"""
    Lays the momentum rows of the loop's mixing out over the ranks: ordered node by node, every rank a contiguous
    block, the larger blocks first so rank 0 owns rows whenever any rank does, or every row on rank 0 with
    ``one_owner``. Collective over ``comm``.

    :param comm: The MPI communicator.
    :param node_comm: The node-local communicator, or ``None`` on a single rank.
    :param roots_comm: The communicator over the node roots, or ``None`` on a single rank.
    :param n_rows: Number of momentum rows.
    :param one_owner: Whether rank 0 owns every row (the Jacobian tracker reads the whole core).
    :return: The :class:`_RowLayout` of this rank.
    """
    if node_comm is None:
        return _RowLayout(slice(0, n_rows), comm, np.array([0, n_rows]), 0, None, None)
    node = node_comm.bcast(roots_comm.rank if node_comm.rank == 0 else None, root=0)
    ordered = comm.Split(0, node * comm.size + node_comm.rank)
    sizes = np.zeros(comm.size, dtype=int)
    if one_owner:
        sizes[0] = n_rows
    else:
        sizes += n_rows // comm.size
        sizes[: n_rows % comm.size] += 1
    first, position = np.cumsum(sizes) - sizes, ordered.rank
    rows = slice(int(first[position]), int(first[position] + sizes[position]))
    nodes = ordered.allgather(node)
    bounds = np.array([first[nodes.index(n)] for n in range(max(nodes) + 1)] + [n_rows])
    owners = ordered.Split(0 if sizes[position] else 1)
    ordered.Free()
    if not sizes[position]:
        owners.Free()
        owners = None
    return _RowLayout(rows, owners, bounds, node, node_comm, roots_comm)


def _scatter_proposal_rows(core: np.ndarray, layout: _RowLayout) -> None:
    r"""
    Hands every node its block of the proposal's core rows: rank 0, which holds every row in the shared window of its
    node, sends each other node root its node's block (one chunked message per node) into that node's shared window;
    the other ranks of each node sleep meanwhile (see :func:`dgamore.mpi_utils.idle_barrier`). Collective over the
    ranks of ``layout``.

    :param core: This node's shared core window ``[k, o1, o2, v]`` (on rank 0's node the proposal's).
    :param layout: The row layout of the mixing (see :func:`_row_layout`).
    :return: None.
    """
    if layout.node_comm is None:
        return
    bounds = layout.bounds
    if layout.node_comm.rank == 0 and layout.node == 0:
        for n in range(1, len(bounds) - 1):
            mpi_utils.send_rows(layout.roots_comm, core[bounds[n] : bounds[n + 1]], n, _PROPOSAL_ROWS_TAG)
    elif layout.node_comm.rank == 0:
        block = core[bounds[layout.node] : bounds[layout.node + 1]]
        mpi_utils.recv_rows_into(layout.roots_comm, block, 0, _PROPOSAL_ROWS_TAG)
    mpi_utils.idle_barrier(layout.node_comm)


def _gather_mixed_rows(core: np.ndarray, layout: _RowLayout) -> None:
    r"""
    Completes every node's shared core window with the mixed rows of the other nodes: once the owners of a node have
    written their rows, its root broadcasts the node's block to the other roots, node by node, straight into their
    windows, while the other ranks of each node sleep (see :func:`dgamore.mpi_utils.idle_barrier`). Collective over
    the ranks of ``layout``.

    :param core: This node's shared core window ``[k, o1, o2, v]``, holding the mixed rows of its node's owners.
    :param layout: The row layout of the mixing (see :func:`_row_layout`).
    :return: None.
    """
    if layout.node_comm is None:
        return
    mpi_utils.idle_barrier(layout.node_comm)
    if layout.node_comm.rank == 0:
        for n in range(len(layout.bounds) - 1):
            mpi_utils.bcast_rows_into(layout.roots_comm, core[layout.bounds[n] : layout.bounds[n + 1]], root=n)
    mpi_utils.idle_barrier(layout.node_comm)


def _build_iterate_window(
    core: np.ndarray,
    sigma_old: SelfEnergy,
    shell: tuple,
    node_comm,
    beta: float,
) -> tuple[SelfEnergy, MPI.Win | None, float]:
    r"""
    Builds the next iterate of the loop in a new shared window of every node, every rank of the node a contiguous
    share of the momenta: the mixed core of the node's core window padded with the proposal's shell (see
    :meth:`SelfEnergy.concatenate_self_energies`), whose frequencies are averaged with their time-reversed partners
    when the proposal's core was (see :meth:`SelfEnergy.symmetrize_time_reversal`). After a damped step every row is
    mixed over every frequency with the previous iterate, cut to the core and padded again from its outermost
    frequency (see :meth:`SelfEnergy.shell_offset_from`), and takes the mixed core again. Every element holds the bits
    of the mixing of the whole concatenated self-energies. Collective over ``node_comm``.

    :param core: This node's shared core window ``[k, o1, o2, 2 niv_core]`` of the mixed rows.
    :param sigma_old: The previous iterate (compressed momentum axis), on the whole grid or momentum-local.
    :param shell: The tuple ``(donor, offset, average, donor_old, damped, alpha)`` of the proposal's shell donor, its
        shell offset ``[k, o1, o2]`` (every momentum), whether its shell is averaged with its time-reversed partner,
        the shell donor of the previous iterate, whether the step was damped and its mixing parameter.
    :param node_comm: The node-local communicator, or ``None`` on a single rank.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The tuple ``(sigma, win, removed)`` of this rank's :class:`SelfEnergy` viewing its node's new window
        (moments not fitted), the window (``None`` for a single-rank node) and the largest change of the shell's
        time-reversal average (zero without it).
    """
    donor, offset, average, donor_old, damped, alpha = shell
    n_q, n_core, n_v = core.shape[0], core.shape[-1], 2 * donor.niv
    window, win = mpi_utils.allocate_node_shared_array(node_comm, (*core.shape[:-1], n_v), core.dtype)
    size, rank = (node_comm.Get_size(), node_comm.Get_rank()) if win is not None else (1, 0)
    mine, every = slice(rank * n_q // size, (rank + 1) * n_q // size), slice(0, n_v)
    view = lambda mat: SelfEnergy(mat, (len(mat), 1, 1), True, True, calc_smom=False, beta=beta)
    padding = offset[mine] if len(offset) > 1 else offset
    view(core[mine]).concatenate_self_energies(donor, shell_offset=padding, out=window[mine], frequencies=every)
    sigma = SelfEnergy(window, config.lattice.k_grid.nk, True, True, calc_smom=False, beta=beta)
    removed = 0.0
    if average:
        # the partners' rows are other ranks' padding
        if win is not None:
            node_comm.Barrier()
        lo, hi = (n_v - n_core) // 2, (n_v + n_core) // 2
        removed = max(
            sigma.symmetrize_time_reversal(frequencies=f, momenta=mine) for f in (slice(0, lo), slice(hi, n_v))
        )
        if win is not None:
            node_comm.Barrier()
    if damped:
        old_mat = sigma_old.compress_q_dimension().mat
        for start in range(mine.start, mine.stop, _SHELL_ROWS):
            chunk = slice(start, min(start + _SHELL_ROWS, mine.stop))
            old = view(old_mat[chunk] if len(old_mat) > 1 else old_mat)
            old_offset = old.shell_offset_from(donor_old)
            old = old.cut_niv(n_core // 2).concatenate_self_energies(donor_old, shell_offset=old_offset)
            mixed = alpha * view(window[chunk]) + (1 - alpha) * old
            view(core[chunk]).concatenate_self_energies(mixed, out=window[chunk], frequencies=every)
    if win is not None:
        node_comm.Barrier()
        removed = max(node_comm.allgather(removed))
    return sigma, win, removed


@dataclass
class _Proposal:
    r"""
    The raw self-energy proposal of one iteration as a rank of rank 0's node holds it: the finished core window
    ``[k, o1, o2, 2 niv_core]`` (compressed momentum axis) in a shared window of the node, and on rank 0 the
    momentum-local donor of the shell frequencies, the shell offset ``[k, o1, o2]``, whether the shell is averaged
    with its time-reversed partner and the largest change of the core's average. The shell frequencies are never
    held: every node builds them for the next iterate (see :func:`_build_iterate_window`).

    :ivar core: The finished core of the proposal, viewing the node's shared window.
    :ivar win: The shared window of the core (``None`` on a single-rank node).
    :ivar donor: The momentum-local self-energy of the shell frequencies (rank 0 only).
    :ivar shell_offset: The momentum-dependent constant added to the shell frequencies (rank 0 only).
    :ivar average: Whether the shell is averaged with its time-reversed partner (rank 0 only).
    :ivar removed: The largest change of the core's time-reversal average (rank 0 only, ``None`` without it).
    """

    core: SelfEnergy
    win: MPI.Win | None
    donor: SelfEnergy | None = None
    shell_offset: np.ndarray | None = None
    average: bool = False
    removed: float | None = None


def _finish_proposal_on_node(
    core: SelfEnergy | None,
    terms: list[np.ndarray] | None,
    average: bool,
    node_comm,
    unfold: bool = False,
) -> tuple[SelfEnergy, MPI.Win | None, float | None]:
    r"""
    Finishes the core of rank 0's proposal on rank 0's node: rank 0 copies the core into a shared window of the node,
    and every rank of the node adds ``terms`` to its contiguous share of the momenta, in their order (each term
    broadcast over the core or holding every momentum, see :meth:`SelfEnergy.add`); with ``unfold`` overwrites a core
    that obeys the grid's point group to rounding (see :func:`_has_point_group_covariance`) with the unfold of its
    irreducible momenta, every rank the stars of its share of them (see :meth:`SelfEnergy.unfold_irreducible_momenta`);
    and with ``average`` averages the core with its time-reversed partner, every rank the pairs whose smaller momentum
    is its own (see :meth:`SelfEnergy.symmetrize_time_reversal`). The core holds the bits of rank 0 finishing the
    concatenated proposal alone; its shell is every node's to build (see :func:`_build_iterate_window`). The other
    ranks sleep while rank 0 arrives (see :func:`dgamore.mpi_utils.idle_barrier`). Collective over ``node_comm``,
    whose root is rank 0.

    :param core: The core proposal on rank 0 (full fermionic range); ignored elsewhere.
    :param terms: The arrays added to the core on rank 0, each ``[1, o1, o2, ...]`` or ``[k, o1, o2, ...]`` in the
        compressed momentum layout; ignored elsewhere.
    :param average: Whether the proposal is averaged with its time-reversed partner; only read on rank 0.
    :param node_comm: The node-local communicator of rank 0's node.
    :param unfold: Whether a core that obeys the grid's point group to rounding is unfolded from its irreducible
        momenta before the average, so it obeys the group without rounding on every momentum; a core that does not
        obey the group, e.g. through a local term that breaks it, is kept. Only read on rank 0.
    :return: The tuple ``(core, win, removed)`` on every rank of the node: the finished core (compressed momentum
        axis) viewing the node's shared window, the window (``None`` on a single-rank node; free it with
        :func:`~dgamore.mpi_utils.free_shared_window` once no rank reads the core) and the largest change of the core's
        average (``None`` without it).
    """
    mpi_utils.idle_barrier(node_comm)
    root = node_comm.Get_rank() == 0
    mat = None
    if root:
        core = core.compress_q_dimension()
        mat, core.mat = core.mat, None
    blob = pickle.dumps((core, average, unfold, len(terms), mat.shape, mat.dtype)) if root else None
    core, average, unfold, n_terms, shape, dtype = pickle.loads(node_comm.bcast(blob, root=0))
    core_mat, core_win = mpi_utils.allocate_node_shared_array(node_comm, shape, dtype)
    if root and core_win is None:
        core_mat = mat
    elif root:
        core_mat[...] = mat
    mat = None
    shared = [mpi_utils.build_node_shared_array(node_comm, lambda i=i: terms[i]) for i in range(n_terms)]
    core.mat = core_mat

    size, rank = (node_comm.Get_size(), node_comm.Get_rank()) if core_win is not None else (1, 0)
    n_q, n_irr = shape[0], config.lattice.k_grid.nk_irr
    mine = slice(rank * n_q // size, (rank + 1) * n_q // size)
    rows = SelfEnergy(core_mat[mine], (mine.stop - mine.start, 1, 1), True, True, calc_smom=False, beta=config.sys.beta)
    for term, _ in shared:
        rows.add(term[mine] if len(term) > 1 else term, copy=False)
    if core_win is not None:
        node_comm.Barrier()
    if unfold and _has_point_group_covariance(core_mat, node_comm if core_win is not None else None):
        irreducible = slice(rank * n_irr // size, (rank + 1) * n_irr // size)
        core.unfold_irreducible_momenta(config.lattice.k_grid, irreducible=irreducible)
        if core_win is not None:
            node_comm.Barrier()
    removed = None
    if average:
        removed = max(node_comm.allgather(core.symmetrize_time_reversal(momenta=mine)))

    wins = [win for _, win in shared]
    shared = term = rows = None
    for win in wins:
        mpi_utils.free_shared_window(win, node_comm)
    return core, core_win, removed


def calculate_sigma_proposal(
    sigma_in: SelfEnergy,
    mu: float,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    v_nonloc_full: Interaction,
    sigma_dmft: SelfEnergy,
    delta_sigma: SelfEnergy,
    my_irr_q_list: np.ndarray,
    mpi_dist_irrk: MpiDistributor,
    comm: MPI.Comm,
    current_iter: int,
    annealer: "LambdaAnnealer | None" = None,
    chunk_budgets: memory_estimator.ChunkBudgets | None = None,
    save_eliashberg: bool = False,
    eliashberg_only: bool = False,
    lambda_previous: dict | None = None,
    static_magn: list | None = None,
    last_chi: dict | None = None,
    w_r_full: list | None = None,
) -> _Proposal | None:
    r"""
    Returns the raw (un-mixed) DGA self-energy proposal :math:`S(\Sigma_{\mathrm{in}})` at chemical potential
    :math:`\mu`: Hartree/Fock, the Dyson Green's function, the bubble, the double-counting, density and magnetic
    kernels, and the FFT Schwinger-Dyson contraction, finished with the noise-removal term and the DMFT tail. The tail
    carries the momentum-dependent part of the Hartree-Fock term, i.e. the contribution of :math:`V^{\mathbf{q}}`, which
    the impurity self-energy does not contain. The core frequencies are finished on rank 0's node by momentum rows (see
    :func:`_finish_proposal_on_node`) and returned in its shared window; the tail is returned as its donor and its
    offset, from which every node builds the shell of the next iterate. For several orbitals on a lattice with real
    hoppings the proposal is finally averaged with its time-reversed partner (see
    :meth:`~dgamore.self_energy.SelfEnergy.symmetrize_time_reversal`): the Schwinger-Dyson equation is one-sided, with
    the three-leg vertex on the first external orbital and the interaction on the second, time reversal maps it onto the
    mirrored form, and the average is the two-sided equation. One band needs no average, its one-sided equation already
    obeys time reversal. On a grid with a discovered point group of real orbital rotations the contraction averages its
    result over the group and, where the Green's function obeys the group, scatters its kernel columns instead of
    unfolding them (see :class:`~dgamore.sde_kernels.ColumnSdeStream`). An orbital permutation that transposes the
    dispersion (see :func:`~dgamore.sde_kernels.orbital_involution`) reaches the bubble and the Bethe-Salpeter slices of
    both channels, which the ranks of a node may share (see :func:`~dgamore.sde_kernels.create_auxiliary_chi_r_q_sum`).
    In the bare-vertex modes (``config.misc.do_flex``, ``config.misc.do_rpa``) no local vertex is loaded and no
    Bethe-Salpeter slice is solved: the double-counting kernel takes the bare vertex (see
    :func:`~dgamore.sde_kernels.calculate_bare_dc_kernel`) and each channel the RPA kernel of its RPA susceptibility
    (see :func:`~dgamore.sde_kernels.calculate_rpa_kernel_r_q`), treated like the ladder's (see
    :func:`_finish_physical_susceptibility`); both carry no fermionic frequency until the contraction.

    Single source of truth for the proposal map: it is called once per self-consistency iteration by
    :func:`calculate_self_energy_q`, plus once with ``eliashberg_only`` after a loop that converged before its last
    scheduled iteration. The local irreducible vertex is frozen, so every
    evaluation rebuilds the bubble, the ladder susceptibilities and the SDE self-energy. The Hartree/Fock term reads
    ``config.sys.occ`` / ``occ_k``, which the caller sets consistently with :math:`\Sigma_{\mathrm{in}}`.

    :param sigma_in: The input self-energy (full-BZ or local first-iteration, DMFT tail attached).
    :param mu: Chemical potential :math:`\mu`.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`, reduced to this rank's irreducible q-points.
    :param v_nonloc_full: The non-local interaction on the full q-grid (for the Hartree/Fock term); only read on
        rank 0.
    :param sigma_dmft: The DMFT self-energy (cut to the loop's niv), providing the high-frequency tail.
    :param delta_sigma: The DMFT-minus-local noise-removal term on the core box (zero in the bare-vertex modes).
    :param my_irr_q_list: This rank's irreducible q-point list.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param comm: The MPI communicator.
    :param current_iter: The current iteration number (the RPA susceptibility is saved and the orbital involution
        logged only on iteration 1).
    :param annealer: The active :class:`LambdaAnnealer` threaded into the kernel step, or ``None`` when annealing
        is off.
    :param chunk_budgets: Chunk byte budgets, of which the self-energy contraction reads its own (sized by the driver
        from the memory estimate); ``None`` gives it the job-wide fair-share budget of
        :func:`~dgamore.sde_kernels.sde_chunk_budget`.
    :param save_eliashberg: Whether to write this rank's Eliashberg intermediates (three-leg vertices and vertex
        middles, and on rank 0 the Green's function the Eliashberg step rebuilds the inverse core bubble from; in the
        bare-vertex modes the physical susceptibilities only); the loop asks for it in its last scheduled iteration.
    :param eliashberg_only: Whether to stop after the two channel kernels: the double-counting kernel and the SDE are
        skipped and ``None`` is returned on every rank (the pass after a loop that converged before its last
        iteration).
    :param lambda_previous: The per-iteration :math:`\lambda` of each corrected channel, kept by the
        self-consistency loop across its iterations; ``None`` for the one-shot correction.
    :param static_magn: Holder of the static magnetic susceptibility for the loop's Ornstein-Zernike fit (see
        :func:`_finish_physical_susceptibility`), or ``None``.
    :param last_chi: Holder of the loop's latest physical susceptibility per channel (see
        :func:`_finish_physical_susceptibility`), or ``None``.
    :param w_r_full: Holder of the real-space interactions of the two Hartree-Fock terms, :math:`U + V^{\mathbf{q}}` and
        :math:`V^{\mathbf{q}}` alone (see :func:`~dgamore.sde_kernels.real_space_interaction`), which rank 0 transforms
        into an empty holder and reads from a filled one, or ``None`` to transform them every time; only read on rank 0.
    :return: The proposal (see :class:`_Proposal`): its core on every rank of rank 0's node, with the donor and the
        offset of its tail on rank 0; ``None`` on the other nodes, and on every rank with ``eliashberg_only``.
    """
    logger = config.logger

    giwk_full, giwk_win, shared_node_comm = sde_kernels.build_giwk_full(
        comm, sigma_in, mu, config.lattice.hamiltonian.get_ek(), config.sys.beta
    )

    # giwk lives in one shared window per node, so the job holds one copy per node - not one per rank
    logger.log_memory_usage("giwk", giwk_full, mpi_utils.count_nodes(comm, shared_node_comm), per="node")

    # an orbital permutation that transposes the dispersion halves the bubble's orbital components and reaches the
    # Bethe-Salpeter slices of both channels
    involution = sde_kernels.orbital_involution(config.lattice.hamiltonian.get_ek(), config.sys.correlated_orbitals)
    niv_sde, cut = config.box.niv_core + config.box.niw_core, []

    def release() -> None:
        """Cuts giwk to the box of the kernels and the SDE into a smaller node window and frees the full one, as
        soon as the bubble has read giwk for the last time (its columns read only the real-space window)."""
        cut.extend(sde_kernels.cut_and_reshare_giwk(giwk_full, giwk_win, shared_node_comm, niv_sde))
        if giwk_win is not None:
            giwk_full.mat = None
        mpi_utils.free_shared_window(giwk_win, shared_node_comm)

    blocks = BubbleGenerator.create_generalized_chi0_q_fft(
        mpi_dist_irrk,
        giwk_full,
        config.box.niw_core,
        config.box.niv_full,
        config.lattice.k_grid,
        config.sys.beta,
        node_comm=shared_node_comm,
        involution=involution,
        blocks=True,
        release=release,
    )
    # the cut giwk stays one copy per node through the SDE
    giwk_full, giwk_win = cut

    logger.log_memory_usage("Gchi0_q_full", blocks, comm.size)

    # the local vertices are identical on every rank, so they are loaded once per node into shared windows; the
    # double-counting kernel only enters the SDE, which the Eliashberg-only pass skips
    bare_vertex = config.misc.bare_vertex
    tail_moment = f_dc_loc = None
    if not eliashberg_only and not bare_vertex:
        f_dc_loc, f_dc_win = sde_kernels.load_node_shared_local_vertex(
            shared_node_comm,
            output_files.npy_path(config.output.output_path, output_files.F_DC_LOC),
            SpinChannel.MAGN,
            axes=(4, 0, 1, 5, 2, 3, 6),  # the order contract_first_pair_with_local_vertex reads as a view
        )
    # the full-box bubble is consumed bosonic window by bosonic window, so it never sits beside the whole core box
    gchi0_q_full_sum, dc_contraction, gchi0_q_core = sde_kernels.bubble_box_parts(blocks, f_dc_loc, not bare_vertex)
    if f_dc_loc is not None:
        f_dc_loc.mat = None
        if f_dc_win is None:
            f_dc_loc.free()
        mpi_utils.free_shared_window(f_dc_win, shared_node_comm)

    chunk_bytes = sde_kernels.sde_chunk_budget(comm, shared_node_comm) if chunk_budgets is None else chunk_budgets.sde
    if bare_vertex:
        # the RPA susceptibility needs the whole frequency sum of the bubble, so the box sum gets its tail
        tail = sde_kernels.create_bubble_tail(
            gchi0_q_full_sum.n_bands, config.box.niw_core, config.box.niv_full, config.sys.beta
        )
        gchi0_q_full_sum.add(tail, copy=False)
        # the bare-vertex kernels do not depend on the fermionic frequency, so they get its axis only for the SDE
        kernel = None if eliashberg_only else sde_kernels.calculate_bare_dc_kernel(gchi0_q_full_sum, u_loc)
        if current_iter == 1:
            calculate_and_save_chi_q_r_rpa(gchi0_q_full_sum, u_loc, v_nonloc, mpi_dist_irrk)
        for channel, weight in ((SpinChannel.DENS, 1.0), (SpinChannel.MAGN, 3.0)):
            _monitor_stoner_factor(gchi0_q_full_sum, u_loc, v_nonloc, channel, mpi_dist_irrk, annealer)
            chi_q_r = sde_kernels.create_rpa_susceptibility(gchi0_q_full_sum, u_loc, v_nonloc, channel)
            chi_q_r = _finish_physical_susceptibility(
                chi_q_r,
                mpi_dist_irrk,
                annealer,
                save_eliashberg,
                lambda_previous,
                static_magn=static_magn,
                last_chi=last_chi,
            )
            if kernel is not None:
                kernel.add(sde_kernels.calculate_rpa_kernel_r_q(chi_q_r, v_nonloc, u_loc).scale(weight), copy=False)
            logger.info(f"Calculated RPA kernel for {channel.value} channel.")
        gchi0_q_full_sum.free()
        # RPA mode pads its proposal with its own tail, whose 1/iv coefficient comes from the kernel before its
        # fermionic axis
        if kernel is not None and config.misc.do_rpa:
            tail_moment = _sde_tail_moment(kernel, mpi_dist_irrk)
        if kernel is not None:
            kernel = kernel.repeat_over_vn(config.box.niv_core, 1)
    else:
        logger.log_memory_usage("Gchi0_q_core", gchi0_q_core, comm.size)

        # the interaction is attached once the full-box bubble is freed, so its operand copy never sits beside the
        # bubble
        kernel = None
        if dc_contraction is not None:
            kernel = sde_kernels.attach_dc_interaction(dc_contraction, u_loc)
            dc_contraction.free()
            logger.info("Calculated double-counting kernel.")

        gchi0_q_core_sum = gchi0_q_core.sum_over_all_vn(config.sys.beta).scale(1.0 / config.sys.beta)
        gchi0_q_core_inv = gchi0_q_core.invert(copy=False)
        del gchi0_q_core
        logger.log_memory_usage("Gchi0_q_inv", gchi0_q_core_inv, comm.size)

        if current_iter == 1:
            calculate_and_save_chi_q_r_rpa(gchi0_q_full_sum, u_loc, v_nonloc, mpi_dist_irrk)

        if save_eliashberg and comm.rank == 0:
            # the Eliashberg step rebuilds the inverse core bubble from this Green's function (see
            # eliashberg_solver.inverse_core_bubble): one file instead of every rank's inverse bubble
            save_rank_file(giwk_full, output_files.GIWK_FILE_STEM)

        if involution is not None and current_iter == 1:
            logger.info(
                f"The orbital permutation {involution.tolist()} transposes the dispersion: Bethe-Salpeter slices that "
                f"are complex-symmetric only with their orbital pairs permuted take the permuted factorization."
            )

        gamma_dens, gamma_dens_win = sde_kernels.load_node_shared_local_vertex(
            shared_node_comm,
            output_files.npy_path(config.output.output_path, output_files.local_name("gamma", SpinChannel.DENS)),
            SpinChannel.DENS,
        )
        calculate_sigma_kernel_r_q(
            gamma_dens,
            gchi0_q_core_inv,
            gchi0_q_full_sum,
            gchi0_q_core_sum,
            u_loc,
            v_nonloc,
            mpi_dist_irrk,
            annealer,
            save_eliashberg=save_eliashberg,
            lambda_previous=lambda_previous,
            kernel=kernel,
            involution=involution,
            last_chi=last_chi,
            node_comm=shared_node_comm,
        )
        gamma_dens.mat = None
        if gamma_dens_win is None:
            gamma_dens.free()
        mpi_utils.free_shared_window(gamma_dens_win, shared_node_comm)
        mpi_dist_irrk.barrier()
        logger.info("Calculated kernel for density channel.")

        gamma_magn, gamma_magn_win = sde_kernels.load_node_shared_local_vertex(
            shared_node_comm,
            output_files.npy_path(config.output.output_path, output_files.local_name("gamma", SpinChannel.MAGN)),
            SpinChannel.MAGN,
        )
        calculate_sigma_kernel_r_q(
            gamma_magn,
            gchi0_q_core_inv,
            gchi0_q_full_sum,
            gchi0_q_core_sum,
            u_loc,
            v_nonloc,
            mpi_dist_irrk,
            annealer,
            save_eliashberg=save_eliashberg,
            lambda_previous=lambda_previous,
            kernel=kernel,
            weight=3.0,
            involution=involution,
            static_magn=static_magn,
            last_chi=last_chi,
            node_comm=shared_node_comm,
        )
        gchi0_q_core_inv.free()
        gchi0_q_full_sum.free()
        gchi0_q_core_sum.free()
        gamma_magn.mat = None
        if gamma_magn_win is None:
            gamma_magn.free()
        mpi_utils.free_shared_window(gamma_magn_win, shared_node_comm)
        logger.info("Calculated kernel for magnetic channel.")

    if eliashberg_only:
        if giwk_win is not None:
            giwk_full.mat = None
        sde_kernels.release_shared_giwk(giwk_win, shared_node_comm)
        return None

    logger.info("Starting calculation of DGA self-energy.")

    # the kernel columns are scattered instead of unfolded where the Green's function obeys the point group
    symmetric = _has_point_group_covariance(giwk_full.mat, comm)
    if not symmetric and config.lattice.k_grid.auto_point_group() is not None:
        logger.info("The Green's function obeys the point group only approximately: the kernel columns are unfolded.")

    # Column-distributed real-space contraction (the q-loop variant peaks HIGHER, see calculate_sigma_from_kernel);
    # the k-space giwk window is released once the R-space window is built, only giwk's dispersion is read later.
    g_r, g_r_win = sde_kernels.build_rspace_giwk_window(giwk_full, shared_node_comm)
    giwk_niv = giwk_full.niv
    if giwk_win is not None:
        giwk_full.mat = None
    mpi_utils.free_shared_window(giwk_win, shared_node_comm)

    # the node's ranks share the column work only where the budgets say so; each rank alone is faster where it fits
    sde_node = shared_node_comm if chunk_budgets is None or chunk_budgets.sde_node else None
    sigma_prop = sde_kernels.run_column_sde(kernel, mpi_dist_irrk, g_r, giwk_niv, chunk_bytes, symmetric, sde_node)
    kernel.free()
    g_r = None
    mpi_utils.free_shared_window(g_r_win, shared_node_comm)

    # rank 0's node finishes the core of the proposal; every node builds the shell of the next iterate on its own
    if not shared_node_comm.bcast(comm.rank == 0, root=0):
        shared_node_comm.Free()
        return None

    terms = donor = shell_offset = None
    average = False
    if comm.rank == 0:
        sigma_prop = sigma_prop.ifft().to_full_niv_range()
        logger.info("Self-energy calculated from kernel.")
        logger.log_memory_usage("Non-local sigma", sigma_prop)

        zero_u = LocalInteraction(np.zeros_like(u_loc.mat))
        if w_r_full is not None and not w_r_full:
            w_r_full[:] = sde_kernels.real_space_interaction(u_loc, v_nonloc_full), sde_kernels.real_space_interaction(
                zero_u, v_nonloc_full
            )
        w_r, w_r_v = w_r_full if w_r_full else (None, None)
        hartree, fock = sde_kernels.get_hartree_fock(u_loc, v_nonloc_full, w_r_mat=w_r)
        # the V^q part of Hartree-Fock is absent from the impurity self-energy padding the shell, so it is re-added
        hartree_v, fock_v = sde_kernels.get_hartree_fock(zero_u, v_nonloc_full, w_r_mat=w_r_v)
        logger.info("Calculated Hartree and Fock terms.")

        # the noise-removal term takes the fluctuations of the local self-energy out and the smooth DMFT one in
        terms = [hartree, fock, delta_sigma.mat.reshape(1, *delta_sigma.mat.shape[-3:])]
        donor, shell_offset = sigma_dmft, (hartree_v + fock_v)[..., 0]
        if config.misc.do_rpa:
            # RPA mode carries no DMFT self-energy: its shell is the whole Hartree-Fock term and the kernels' 1/iv tail
            donor, shell_offset = _rpa_shell_donor(tail_moment, sigma_dmft.niv), (hartree + fock)[..., 0]
        # the one-sided SDE breaks Sigma(-k) = Sigma(k)^T for several orbitals; the average is the two-sided SDE
        average = sigma_prop.n_bands > 1 and sde_kernels.has_time_reversal(config.lattice.hamiltonian.get_ek())
    # a grid whose contraction averages over a point group gets the core of a proposal that obeys the group unfolded
    # from its irreducible momenta, which removes the rounding of the transforms and of the terms added on rank 0
    unfold = comm.rank == 0 and config.lattice.k_grid.auto_point_group() is not None
    core, win, removed = _finish_proposal_on_node(sigma_prop, terms, average, shared_node_comm, unfold)
    shared_node_comm.Free()
    if comm.rank != 0:
        return _Proposal(core, win)
    logger.info("Full non-local self-energy calculated.")
    if not average and core.n_bands > 1:
        logger.info("The dispersion breaks time reversal (complex hoppings); the self-energy proposal is not averaged.")
    return _Proposal(core, win, donor, shell_offset, average, removed)


def interpolate_sigma(sigma: SelfEnergy, beta_target: float, niv_target: int, comm: MPI.Comm) -> SelfEnergy | None:
    r"""
    Re-grids the self-energy to ``beta_target``. The plain interpolation (:meth:`SelfEnergy.interpolate`) runs on the
    irreducible Brillouin zone on rank 0 alone; at the momenta flagged by :meth:`SelfEnergy.pole_fit_mask` the target
    frequencies below the innermost source frequency are replaced by the MiniPole extrapolation of
    :meth:`SelfEnergy.pole_extrapolate`, with the fits distributed over the ranks (each rank re-grids only the momenta
    it fits, which equals the corresponding rows of the whole-zone interpolation) and rejected fits keeping the plain
    values. The result is unfolded to the full Brillouin zone with :meth:`SelfEnergy.map_to_full_bz`, orbital
    rotations included, on rank 0 only (the only consumer); every other rank returns ``None``. Collective over
    ``comm``.

    :param sigma: The momentum-dependent :class:`SelfEnergy` on the full Brillouin zone.
    :param beta_target: Inverse temperature :math:`\beta` of the target grid.
    :param niv_target: Number of positive fermionic frequencies of the target grid.
    :param comm: The MPI communicator.
    :return: The re-gridded :class:`SelfEnergy` on the full Brillouin zone, momentum layout ``[kx, ky, kz, ...]``,
        on rank 0; ``None`` elsewhere.
    """
    k_grid = config.lattice.k_grid
    irrq_list = k_grid.get_irrq_list()
    # only rank 0 re-grids the whole irreducible zone, since only rank 0 unfolds it
    flagged = None
    if comm.rank == 0:
        sigma_irr = sigma.reduce_q(irrq_list)
        flagged = np.flatnonzero(sigma_irr.pole_fit_mask())
        interpolated = sigma_irr.interpolate(beta_target, niv_target)
    flagged = comm.bcast(flagged, root=0)
    if flagged.size > 0:
        dist = MpiDistributor.create_distributor(ntasks=flagged.size, comm=comm)
        # every rank re-grids only the momenta it fits; reduce_q keeps the order of the irreducible list
        mine = sigma.reduce_q(irrq_list[flagged[dist.my_slice]])
        reference = mine.interpolate(beta_target, niv_target) if dist.my_size else mine
        values, accepted = mine.pole_extrapolate(beta_target, niv_target, np.arange(dist.my_size), reference)
        # the flags travel as uint8: a bool buffer has no fixed MPI datatype in the Allgatherv
        values, accepted = dist.allgather(values), dist.allgather(accepted.astype(np.uint8)).astype(bool)
        if comm.rank == 0:
            interpolated.replace_innermost(flagged[accepted], values[accepted])
        config.logger.info(
            f"MiniPole extrapolation below the innermost frequency accepted at {accepted.sum()} of {flagged.size} "
            f"flagged irreducible k-points."
        )
    if comm.rank != 0:
        return None
    return interpolated.map_to_full_bz(k_grid).decompress_q_dimension()


def _jacobian_expander(state: dict) -> Callable[[np.ndarray], np.ndarray] | None:
    r"""
    Builds the mapping of one stored column of a predecessor's spectrum file to the real vector in this run's tracker
    coordinates: the column, a real vector on the predecessor's weighted symmetry sector (see
    :func:`~dgamore.sigma_jacobian.tracker_sector_maps`), is taken back to the complex self-energy on the irreducible
    momenta, completed to the negative frequencies by the Matsubara Hermiticity, re-gridded on the frequency axis by
    :meth:`SelfEnergy.interpolate` (the hand-over's PCHIP, row by row, so the irreducible rows re-grid exactly as on
    the whole grid), zeroed on every target frequency beyond the predecessor's highest source frequency (an eigenvector
    is not extrapolated) and weighted back into this run's sector. Identical frequency grids skip the re-gridding.
    Refused, with a warning, for another orbital count. For another momentum grid or symmetry reduction, whose sector
    the columns do not live in, every column maps to an empty vector, so the eigenvalues and their damping bound are
    carried without vectors, with a warning. Re-gridding a Ritz vector this way is an approximation.

    :param state: The carried spectrum (``shape``, ``beta``, ``irrk_ind`` and ``irrk_count`` are read).
    :return: The callable, or ``None`` when the carry-over is refused.
    """
    shape_src = tuple(int(n) for n in state["shape"])
    beta_src = float(state["beta"])
    nb, k_grid = config.sys.n_bands, config.lattice.k_grid
    if shape_src[3:5] != (nb, nb):
        config.logger.warning("The carried Jacobian spectrum belongs to another orbital count; nothing carried.")
        return None
    same_sector = (
        shape_src[:3] == tuple(int(n) for n in k_grid.nk)
        and np.array_equal(state["irrk_ind"], k_grid.irrk_ind)
        and np.array_equal(state["irrk_count"], k_grid.irrk_count)
    )
    if not same_sector:
        config.logger.warning(
            "The carried Jacobian spectrum belongs to another momentum grid or symmetry reduction; its eigenvalues "
            "are carried without vectors."
        )
        return lambda column: np.zeros(0)
    niv_tgt, beta_tgt, niv_src = config.box.niv_core, config.sys.beta, shape_src[-1] // 2
    if niv_src == niv_tgt and beta_src == beta_tgt:
        return lambda column: np.asarray(column, dtype=np.float64)
    weight = np.sqrt(2.0 * k_grid.irrk_count)[:, None, None, None]
    inside = np.abs(MFHelper.vn(niv_tgt, beta_tgt)) <= np.abs(MFHelper.vn(niv_src, beta_src)).max()

    def expand(column: np.ndarray) -> np.ndarray:
        half = to_mat(np.asarray(column, dtype=np.float64), (k_grid.nk_irr, nb, nb, niv_src)) / weight
        sigma = SelfEnergy(half, k_grid.nk, False, True, calc_smom=False, beta=beta_src).to_full_niv_range()
        window = sigma.interpolate(beta_tgt, niv_tgt).mat * inside
        return to_vec(window[..., niv_tgt:] * weight)

    return expand


def _carry_jacobian_spectrum(tracker: JacobianTracker, annealer: LambdaAnnealer | None) -> None:
    r"""
    Hands the predecessor's certified Jacobian spectrum in ``jacobian.npz`` of ``previous_sc_path`` (see
    :meth:`JacobianTracker.save_spectrum`) to the tracker before the first iteration of a resumed run, with this run's
    :math:`\beta` for the cross-rung prediction (see :meth:`JacobianTracker.carry_in`). A predecessor short of the pure
    fixed point is carried with a warning; nothing is carried when the expander refuses (see
    :func:`_jacobian_expander`), when the file holds the traces alone, or, with a warning, when it cannot be read or
    lacks a key. Flips stay pending while a susceptibility-reshaping scaffold is on.

    :param tracker: The rank-0 tracker of this run.
    :param annealer: The :class:`~dgamore.lambda_ops.LambdaAnnealer`, or ``None`` while annealing is off.
    :return: None.
    """
    folder = config.self_consistency.previous_sc_path
    path = os.path.join(folder, JACOBIAN_FILE)
    if not os.path.isfile(path):
        config.logger.info(f"No {JACOBIAN_FILE} in {folder}; nothing carried.")
        return
    try:
        state = load_spectrum(path)
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile) as error:
        config.logger.warning(f"{path} could not be read ({error}); nothing carried.")
        return
    if "lam_pi" not in state:
        config.logger.info(f"{path} holds no certified spectrum (the run did not end its loop); nothing carried.")
        return
    allow_flip = not (
        config.stabilization.use_chi_phys_restriction
        or config.stabilization.use_lambda_correction
        or (annealer is not None and annealer.mass_present)
    )
    try:
        if not bool(state["converged"]):
            config.logger.warning(
                f"The predecessor's Jacobian spectrum in {path} belongs to a run that did not reach the pure fixed "
                "point."
            )
        expand = _jacobian_expander(state)
        if expand is not None:
            tracker.carry_in(state, expand, allow_flip=allow_flip, beta=config.sys.beta)
    except KeyError as error:
        config.logger.warning(f"{path} lacks the key {error} of the spectrum layout; nothing carried.")


def _jacobian_traces(rows_lam: list, rows_res: list, rows_damping: list) -> dict[str, np.ndarray]:
    """
    Stacks the per-iteration rows the loop collected into the trace arrays of ``jacobian.npz``.

    :param rows_lam: The leading Ritz value rows, oldest first.
    :param rows_res: The matching Ritz residual rows, oldest first.
    :param rows_damping: The matching effective dampings, oldest first.
    :return: The arrays keyed ``eigenvalues`` (``[n, TRACKER_PAIRS - 1]`` complex), ``eigenvalue_residuals``
        (``[n, TRACKER_PAIRS - 1]``) and ``damping`` (``[n]``).
    """
    return {
        "eigenvalues": np.array(rows_lam, dtype=np.complex128).reshape(len(rows_lam), _JACOBIAN_ROW_WIDTH),
        "eigenvalue_residuals": np.array(rows_res, dtype=np.float64).reshape(len(rows_res), _JACOBIAN_ROW_WIDTH),
        "damping": np.array(rows_damping, dtype=np.float64),
    }


def _save_jacobian_spectrum(
    tracker: JacobianTracker, converged: bool, rows_lam: list, rows_res: list, rows_damping: list
) -> None:
    """
    Writes the tracker's certified spectrum with the per-iteration traces and the grid's ``irrk_ind`` and
    ``irrk_count``, which define its sector coordinates, to ``jacobian.npz`` in the output folder for a successor run
    (see :meth:`JacobianTracker.save_spectrum`), replacing the trace-only file of :func:`_append_jacobian_eigenvalues`.

    :param tracker: The rank-0 tracker of this run.
    :param converged: Whether the run reached the pure fixed point (a scaffolded phase that ended on the last
        iteration does not count).
    :param rows_lam: The leading Ritz value rows collected over the run, oldest first.
    :param rows_res: The matching Ritz residual rows, oldest first.
    :param rows_damping: The matching effective dampings, oldest first.
    :return: None.
    """
    nb, k_grid = config.sys.n_bands, config.lattice.k_grid
    shape = (*(int(n) for n in k_grid.nk), nb, nb, 2 * config.box.niv_core)
    grid = {"irrk_ind": np.asarray(k_grid.irrk_ind), "irrk_count": np.asarray(k_grid.irrk_count)}
    tracker.save_spectrum(
        os.path.join(config.output.output_path, JACOBIAN_FILE),
        shape,
        config.sys.beta,
        converged,
        DTYPE,
        traces={**_jacobian_traces(rows_lam, rows_res, rows_damping), **grid},
    )


def _append_jacobian_eigenvalues(tracker: JacobianTracker, rows_lam: list, rows_res: list, rows_damping: list) -> None:
    """
    Appends this iteration's leading Jacobian Ritz values, residuals and damping to the running rows and rewrites
    ``jacobian.npz`` in the output folder with the three traces (``eigenvalues``, ``eigenvalue_residuals``,
    ``damping``) through a temporary file renamed onto it, so a killed run keeps the previous file; the certified
    spectrum joins the file when the loop ends (see :func:`_save_jacobian_spectrum`). An iteration without an estimate
    (see :attr:`JacobianTracker.estimated`) appends ``nan`` rows to the first two traces.

    :param tracker: The rank-0 tracker owning the last update.
    :param rows_lam: The Ritz value rows collected so far, oldest first, extended in place.
    :param rows_res: The matching Ritz residual rows collected so far, oldest first, extended in place.
    :param rows_damping: The matching effective dampings collected so far, oldest first, extended in place.
    :return: None.
    """
    lam_pi, res = (
        tracker.leading(_JACOBIAN_ROW_WIDTH) if tracker.estimated else (np.full(_JACOBIAN_ROW_WIDTH, np.nan),) * 2
    )
    rows_lam.append(lam_pi)
    rows_res.append(res)
    rows_damping.append(tracker.p_eff)
    path = os.path.join(config.output.output_path, JACOBIAN_FILE)
    partial = f"{path}.tmp"
    with open(partial, "wb") as handle:
        np.savez(handle, **_jacobian_traces(rows_lam, rows_res, rows_damping))
    os.replace(partial, path)


def _update_jacobian_tracker(
    tracker: JacobianTracker,
    window: list,
    current_iter: int,
    release_iter: int | None,
    anneal_reset_iter: int | None,
    annealer: LambdaAnnealer | None,
) -> bool:
    r"""
    Runs one monitoring step of the Jacobian tracker on the raw :math:`(x, S(x))` pairs recorded up to this
    iteration's. The window restarts only on a scaffold event (the restriction or lambda-correction release, a change
    of the annealing mass), dropping the pair recorded at it, never on the tracker's own events; flips are paused
    while a scaffold is active.

    :param tracker: The tracker owning the flipped subspace and the effective damping.
    :param window: The pairs :meth:`JacobianTracker.record` recorded, oldest first, this iteration's last.
    :param current_iter: The current self-consistency iteration number.
    :param release_iter: The iteration a susceptibility-reshaping scaffold was released on (``None`` if never).
    :param anneal_reset_iter: The iteration the annealing mass last changed on (``None`` if never).
    :param annealer: The :class:`~dgamore.lambda_ops.LambdaAnnealer`, or ``None`` while annealing is off.
    :return: Whether the tracked subspace changed, i.e. the reflected map the accelerated mixing sees switched.
    """
    raw_cap = mixing.mixing_history_cap(current_iter, release_iter, anneal_reset_iter)
    n_entries = len(window) if raw_cap is None else min(len(window), raw_cap + 1)
    allow_flip = not (
        config.stabilization.use_chi_phys_restriction
        or config.stabilization.use_lambda_correction
        or (annealer is not None and annealer.mass_present)
    )
    return tracker.update_recorded(window[len(window) - n_entries :], allow_flip=allow_flip)


def _install_exact_jacobian(
    tracker: JacobianTracker,
    sigma: SelfEnergy,
    mu: float,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    v_nonloc_full: Interaction | None,
    sigma_dmft_full: SelfEnergy,
    mpi_dist_irrk: MpiDistributor,
    mpi_dist_fullbz: MpiDistributor,
    comm: MPI.Comm,
    chunk_budgets: memory_estimator.ChunkBudgets | None = None,
) -> None:
    r"""
    Evaluates the leading eigenpairs of the exact Jacobian of the self-energy map at the converged self-energy
    (see :class:`~dgamore.sigma_jacobian.ExactJacobian` and :func:`~dgamore.sigma_jacobian.leading_eigenpairs`) on
    blocks of the operator's block width, started from the least stable and the stiffest certified direction of rank
    0's tracker (see
    :meth:`JacobianTracker.start_directions`), and makes them the spectrum the tracker writes to ``jacobian.npz``
    (see :meth:`JacobianTracker.install_exact_spectrum`). Collective over ``comm``.

    :param tracker: The rank-0 tracker of this run.
    :param sigma: The converged :class:`SelfEnergy` on the loop's box (full BZ, identical on every rank).
    :param mu: Chemical potential :math:`\mu` of ``sigma``.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`, reduced to this rank's irreducible q-points.
    :param v_nonloc_full: The non-local interaction on the full q-grid (for the Hartree/Fock term); only read on
        rank 0.
    :param sigma_dmft_full: The DMFT :class:`SelfEnergy` supplying the shell frequencies (momentum-local).
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param mpi_dist_fullbz: MPI distributor over the full BZ q-points.
    :param comm: The MPI communicator.
    :param chunk_budgets: Chunk byte budget, block width and vertex residency of the products (see
        :class:`~dgamore.memory_estimator.ChunkBudgets`); ``None`` gives the job-wide fair-share budget of
        :func:`~dgamore.sde_kernels.sde_chunk_budget`, one-column products and held vertices.
    :return: None.
    """
    config.logger.info("Evaluating the leading eigenpairs of the exact Jacobian at the converged self-energy.")
    jac = sigma_jacobian.ExactJacobian(
        sigma, mu, u_loc, v_nonloc, v_nonloc_full, sigma_dmft_full, mpi_dist_irrk, mpi_dist_fullbz, comm, chunk_budgets
    )
    starts = tracker.start_directions() if comm.rank == 0 else (None, None)
    spectrum = sigma_jacobian.leading_eigenpairs(jac, comm, *starts, block=jac.block)
    jac.free()  # on the normal path only: an exception aborts the job instead of meeting another collective
    if comm.rank == 0:
        tracker.install_exact_spectrum(*spectrum)


def _run_exact_check(
    tracker: JacobianTracker,
    start: np.ndarray | None,
    sigma: SelfEnergy,
    mu: float,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    v_nonloc_full: Interaction | None,
    sigma_dmft_full: SelfEnergy,
    mpi_dist_irrk: MpiDistributor,
    mpi_dist_fullbz: MpiDistributor,
    comm: MPI.Comm,
    chunk_budgets: memory_estimator.ChunkBudgets | None = None,
) -> bool:
    r"""
    Runs the exact check the tracker requested before this iteration's step: builds the exact Jacobian at the iterate
    (see :class:`~dgamore.sigma_jacobian.ExactJacobian`), certifies the Ritz pairs next to the requested directions
    (see :func:`~dgamore.sigma_jacobian.certify_subspace`), frees the operator, and hands the pairs to rank 0's
    tracker (see :meth:`JacobianTracker.install_exact_check`). Collective over ``comm``.

    :param tracker: The tracker of this run; only rank 0's installs the pairs.
    :param start: The requested real start columns on rank 0 (see :meth:`JacobianTracker.exact_check_request`);
        ``None`` elsewhere.
    :param sigma: The iterate of this iteration's proposal on the loop's box (full BZ, identical on every rank).
    :param mu: Chemical potential :math:`\mu` of ``sigma``.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`, reduced to this rank's irreducible q-points.
    :param v_nonloc_full: The non-local interaction on the full q-grid (for the Hartree/Fock term); only read on
        rank 0.
    :param sigma_dmft_full: The DMFT :class:`SelfEnergy` supplying the shell frequencies (momentum-local).
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param mpi_dist_fullbz: MPI distributor over the full BZ q-points.
    :param comm: The MPI communicator.
    :param chunk_budgets: Chunk byte budget, block width and vertex residency of the products (see
        :class:`~dgamore.memory_estimator.ChunkBudgets`); ``None`` gives the job-wide fair-share budget of
        :func:`~dgamore.sde_kernels.sde_chunk_budget`, one-column products and held vertices.
    :return: On rank 0 whether the reflected map switched; ``False`` elsewhere.
    """
    jac = sigma_jacobian.ExactJacobian(
        sigma, mu, u_loc, v_nonloc, v_nonloc_full, sigma_dmft_full, mpi_dist_irrk, mpi_dist_fullbz, comm, chunk_budgets
    )
    pairs = sigma_jacobian.certify_subspace(jac, start, comm)
    jac.free()  # on the normal path only: an exception aborts the job instead of meeting another collective
    if comm.rank != 0:
        return False
    lam_pi, res, u, _, kappa = pairs
    return tracker.install_exact_check(lam_pi, res, u, kappa)


def calculate_self_energy_q(
    comm: MPI.Comm,
    u_loc: LocalInteraction,
    v_nonloc: Interaction | None,
    sigma_dmft: SelfEnergy,
    sigma_local: SelfEnergy,
    chunk_budgets: memory_estimator.ChunkBudgets | None = None,
) -> SelfEnergy | None:
    r"""
    Runs the non-local DGA self-energy calculation. Calculates the Hartree- and Fock terms, the bubble,
    the double-counting correction and the kernel in the density and magnetic channel. Finally, calculates the
    non-local self-energy from the kernel and the Green's function. Also takes care of the self-consistency loop and
    the chemical potential adjustment as well as the self-energy mixing, etc.

    With ``config.stabilization.use_jacobian_stabilization`` a :class:`~dgamore.jacobian_stabilization.JacobianTracker`
    updates on each iteration's raw (iterate, proposal) pair before that iteration's step; the mixing reflects the
    proposal residual on its certified unstable directions, a damped step moves at its bound
    :attr:`~dgamore.jacobian_stabilization.JacobianTracker.p_eff`, and a change of the reflected map restarts the
    accelerated-mixing history. With ``config.stabilization.use_exact_jacobian`` a requested exact check
    (:func:`_run_exact_check`) runs before the step, and a pure fixed point gets the exact leading eigenpairs
    (:func:`_install_exact_jacobian`). A non-zero :math:`V^{\mathbf{q}}` is warned about once, since its Hartree-Fock
    shell offset lies outside the tracked window.

    Convergence is declared once three residuals lie below epsilon, together with a settled chemical potential: the
    post-mixing step residual (scaled by :math:`p / p_{\mathrm{eff}}` for a damped step of a tracked run), the
    proposal step residual between the proposals of two consecutive iterations, and the map residual
    :math:`\lVert S(\Sigma) - \Sigma\rVert / \lVert\Sigma\rVert` of the proposal against the iterate it was computed
    from. With the tracker the pure fixed point waits for
    :meth:`~dgamore.jacobian_stabilization.JacobianTracker.minimum_iterations` (capped by ``max_iter``), and a
    convergence below the count of the whole certified spectrum (in-band modes counted with the band) is warned about.

    With ``config.misc.do_rpa`` the run starts from the non-interacting lattice instead of the DMFT self-energy: the
    filling of the DMFT lattice is the target, the chemical potential at which a zero self-energy holds it is stored in
    ``config.sys.mu_non_interacting``, and the proposals are evaluated from there.

    Rank 0 fits the Ornstein-Zernike form (:func:`perform_ornstein_zernike_fit`) once at the end, to the static
    magnetic susceptibility of the last proposal (or of the Eliashberg-only pass after an early convergence).
    Without a lambda correction the physical susceptibilities of the last proposal are written once after the loop,
    since no iteration reads them from file.

    :param comm: The MPI communicator.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}` on the full q-grid; only read on rank 0.
    :param sigma_dmft: The DMFT self-energy (used as the starting point and for the shell/tail correction).
    :param sigma_local: The locally recomputed self-energy (used for smoothing out the DGA :class:`SelfEnergy`; not
        read in the bare-vertex modes).
    :param chunk_budgets: Chunk byte budgets, of which the self-energy contraction reads its own (sized by the driver
        from the memory estimate); ``None`` gives it the job-wide fair-share budget of
        :func:`~dgamore.sde_kernels.sde_chunk_budget`.
    :return: The converged (or last-iteration) momentum-dependent DGA :class:`SelfEnergy` on rank 0; ``None`` on every
        other rank.
    """
    logger = config.logger

    logger.info("Starting with non-local DGA routine.")
    logger.info("Initializing MPI distributor.")

    # MPI distributor for the irreducible BZ
    mpi_dist_irrk = sde_kernels.irreducible_momentum_distributor(comm)
    irrk_q_list = config.lattice.k_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    mpi_dist_fullbz = MpiDistributor.create_distributor(ntasks=config.lattice.k_grid.nk_tot, comm=comm)

    # the loop's full-BZ self-energy is held once per NODE, window-shared read-only; a single-rank run keeps rank 0's
    # own array. Every rank mixes a fixed block of momentum rows, rank 0 every row when the tracker reads the core
    sc_node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) if comm.size > 1 else None
    sc_roots_comm = comm.Split(0 if sc_node_comm.rank == 0 else 1) if sc_node_comm is not None else None
    sigma_win = None
    layout = _row_layout(
        comm,
        sc_node_comm,
        sc_roots_comm,
        config.lattice.k_grid.nk_tot,
        one_owner=config.stabilization.use_jacobian_stabilization,
    )

    # only rank 0 reads a previous run's iterate; the other ranks receive it below, once per node
    sigma_old, starting_iter, warm_start = get_starting_sigma(sigma_dmft) if comm.rank == 0 else (sigma_dmft, 0, False)
    starting_iter, warm_start = comm.bcast((starting_iter, warm_start), root=0)
    if warm_start:
        logger.info(
            f"Using previous calculation and starting the self-consistency loop at iteration {starting_iter + 1}."
        )

    mu_history = _init_mu_history(warm_start)

    # every owner of momentum rows keeps the accelerated-mixing (iterate, proposal) pairs of its rows in memory; the
    # saved sigma files hold only post-mixing iterates (no proposals), so resumed runs start empty and re-arm
    mixing_history = None
    if layout.owners is not None and config.self_consistency.mixing_strategy.lower() in ("pulay", "anderson"):
        mixing_history = []

    # the tracker is built on every rank (a few floats) but only rank 0, which owns every row then, ever feeds it; it
    # keeps its vectors, the recorded pair window included, on the weighted symmetry sector of the core window
    tracker = tracker_window = rows_lam = rows_res = rows_damping = stab_event_iter = None
    # every owner's rows of the positive core window of the previous proposal, refilled in place every iteration
    previous_proposal = None
    if config.stabilization.use_jacobian_stabilization:
        maps = sigma_jacobian.tracker_sector_maps(
            config.lattice.k_grid, config.box.niv_core, config.sys.n_bands, config.sys.beta
        )
        tracker = JacobianTracker(
            config.self_consistency.mixing, logger, *maps, exact_checks=config.stabilization.use_exact_jacobian
        )
        rows_lam, rows_res, rows_damping = [], [], []
        tracker_window = [] if comm.rank == 0 else None

    niv_cut = min(config.box.niw_core + config.box.niv_full + 10, config.box.niv_dmft)
    sigma_dmft_full = sigma_dmft.copy()

    ek = config.lattice.hamiltonian.get_ek()
    orbitals = config.sys.correlated_orbitals
    # the DMFT lattice Green's function on the whole DMFT box is built only for the DMFT spectrum continuation; a warm
    # start holds its filling, which is summed without building it
    if comm.rank == 0 and config.ana_cont.do_spectrum_dmft:
        logger.info("Calculating lattice Green's function.")
        giwk_full_dmft = GreensFunction.get_g_full(
            sigma_dmft_full, config.sys.mu_dmft, ek, config.sys.beta, orbitals=orbitals
        )
        giwk_full_dmft.save(output_dir=config.output.output_path, name=output_files.G_LATT_DMFT)
        giwk_full_dmft.free()
        logger.info("Lattice Green's function calculated.")
    if comm.rank == 0 and warm_start:
        n_dmft = GreensFunction.get_occupation(
            sigma_dmft_full, config.sys.mu_dmft, ek, config.sys.beta, orbitals=orbitals
        )[0]

    if comm.rank == 0 or not warm_start:
        sigma_old = sigma_old.concatenate_self_energies(sigma_dmft_full)

    if not warm_start:
        # a fresh start holds the momentum-local DMFT sigma on every rank, so each rank evaluates the filling of the
        # full DMFT box on its own momentum slice (arithmetic decided on all momenta) and assembles it exactly
        n_my, nb = mpi_dist_fullbz.my_size, ek.shape[-1]
        ek_my = ek.reshape(-1, nb, nb)[mpi_dist_fullbz.my_slice].reshape(n_my, 1, 1, nb, nb)
        real_ek = np.isrealobj(np.real_if_close(ek))
        occ_k_my = (
            GreensFunction.get_occupation(
                sigma_old, mu_history[-1], ek_my, config.sys.beta, real_ek, orbitals=orbitals
            )[2]
            if n_my
            else np.zeros((0, 1, 1, nb, nb))
        )
        config.sys.n, config.sys.occ, config.sys.occ_k = sde_kernels.assemble_occupation(
            occ_k_my, mpi_dist_fullbz, orbitals
        )
        if config.misc.do_rpa:
            # RPA mode evaluates its one proposal on the non-interacting lattice: a zero self-energy (new, so its
            # fitted moments vanish too) at the chemical potential at which it holds the filling just assembled
            sigma_old = SelfEnergy(
                np.zeros_like(sigma_old.mat),
                sigma_old.nq,
                sigma_old.full_niv_range,
                sigma_old.has_compressed_q_dimension,
                beta=config.sys.beta,
            )
            mu_history[-1] = comm.bcast(non_interacting_mu(ek, config.sys.n, config.sys.beta), root=0)
            config.sys.mu = config.sys.mu_non_interacting = mu_history[-1]
            occ_k_my = (
                GreensFunction.get_occupation(
                    sigma_old, mu_history[-1], ek_my, config.sys.beta, real_ek, orbitals=orbitals
                )[2]
                if n_my
                else np.zeros((0, 1, 1, nb, nb))
            )
            n_bare, config.sys.occ, config.sys.occ_k = sde_kernels.assemble_occupation(
                occ_k_my, mpi_dist_fullbz, orbitals
            )
            logger.info(
                f"RPA mode starts from the non-interacting lattice: mu = {config.sys.mu} gives the filling "
                f"{n_bare:.6f} (target {config.sys.n:.6f})."
            )
    else:
        # a warm start's own filling drifts along a chain of rungs, so mu is re-solved for the DMFT lattice filling on
        # the loop's frequency box, every rank on its momenta of rank 0's starting iterate
        sigma_cut = sigma_old.cut_niv(niv_cut).compress_q_dimension() if comm.rank == 0 else None
        n_dmft, smom0 = comm.bcast((n_dmft, sigma_cut.fit_smom()[0]) if comm.rank == 0 else None, root=0)
        sigma_my = mpi_dist_fullbz.scatter(sigma_cut.mat if comm.rank == 0 else None)
        del sigma_cut
        mu_previous = mu_history[-1]
        mu_history[-1] = update_mu(
            mu_previous,
            n_dmft,
            ek,
            sigma_my,
            config.sys.beta,
            smom0,
            logger=logger,
            mpi_dist=mpi_dist_fullbz,
            orbitals=orbitals,
        )
        del sigma_my
        if comm.rank == 0:
            logger.info(
                f"Warm start holds the DMFT lattice filling {n_dmft:.6f}: mu re-solved from {mu_previous} to "
                f"{mu_history[-1]}."
            )
            # only the occupations are needed here: sum the Dyson chunks directly instead of holding a second full-k
            # Green's function on the full DMFT frequency box (tens of GB at production scale)
            _, occ, occ_k = GreensFunction.get_occupation(
                sigma_old, mu_history[-1], ek, config.sys.beta, orbitals=orbitals
            )
            config.sys.occ, config.sys.occ_k = orbital_block(occ, orbitals), orbital_block(occ_k, orbitals)
            config.sys.n = n_dmft
        config.sys.n, config.sys.occ, config.sys.occ_k = comm.bcast(
            (config.sys.n, config.sys.occ, config.sys.occ_k), root=0
        )
        config.sys.mu = mu_history[-1]

    sigma_old = sigma_old.cut_niv(niv_cut)
    sigma_dmft = sigma_dmft.cut_niv(niv_cut)

    # the starting iterate keeps the plain DMFT tail; the first proposal attaches the V^q Hartree-Fock offset
    if sigma_old.niv < niv_cut:
        sigma_old = sigma_old.concatenate_self_energies(sigma_dmft)

    # rank 0 holds a resumed run's full-BZ starting iterate: spread it like the loop's mixed iterate, one window
    # per node (the loop frees this window after the first proposal)
    if warm_start and sc_node_comm is not None:
        sigma_old, sigma_win = _share_sigma_per_node(sigma_old, comm, sc_node_comm, sc_roots_comm)

    # the bare-vertex modes add no noise-removal term: their self-energy is the one of their RPA kernels alone
    reference = sigma_dmft if config.misc.bare_vertex else sigma_local
    delta_sigma = sigma_dmft.cut_niv(config.box.niv_core) - reference.cut_niv(config.box.niv_core)

    # only rank 0 finishes the proposal with the Hartree-Fock term, which reads the full q-grid interaction; the first
    # proposal transforms the interactions of both terms for the whole loop
    v_nonloc_full = v_nonloc if comm.rank == 0 else None
    w_r_full = []
    if config.stabilization.use_jacobian_stabilization and comm.rank == 0 and np.any(v_nonloc.mat):
        held = " and the exact Jacobian holds the offset fixed" if config.stabilization.use_exact_jacobian else ""
        logger.warning(
            "Jacobian stabilization with a non-zero V^q: the loop recomputes the V^q Hartree-Fock offset of the shell "
            f"from every iterate's occupations, while the tracker records the core window alone{held}, so the "
            "measured spectrum and the one in jacobian.npz leave out its feedback."
        )
    v_nonloc = sde_kernels.scatter_irreducible_interaction(v_nonloc, mpi_dist_irrk)
    # the operator arguments of both exact-Jacobian helpers after the iterate and its mu
    exact_args = (u_loc, v_nonloc, v_nonloc_full, sigma_dmft_full, mpi_dist_irrk, mpi_dist_fullbz, comm, chunk_budgets)

    annealer = LambdaAnnealer() if config.stabilization.use_lambda_annealing else None
    # the pole ratio watches the density Bethe-Salpeter slices, which the bare-vertex modes do not solve
    pole_vertex = first_frequency_local_vertex() if comm.rank == 0 and not config.misc.bare_vertex else None
    # one dict for the whole run warm-starts every channel's search from the previous iteration's lambda
    lambda_previous = None
    if config.stabilization.use_lambda_correction and not config.lambda_correction.perform_lambda_correction:
        lambda_previous = {}
    if tracker is not None and comm.rank == 0 and warm_start:
        _carry_jacobian_spectrum(tracker, annealer)
    anneal_reset_iter = None
    release_iter = None
    pure_converged = False
    static_magn = []  # rank 0: the static magnetic susceptibility of the latest proposal, fitted after the loop
    last_chi = {}  # every rank: its momenta of the latest physical susceptibilities, written after the loop
    # rank 0 writes each iterate in the background while the next proposal runs; the write reads the shared window
    # of the iterate, so it is awaited before that window is freed
    saver = ThreadPoolExecutor(max_workers=1) if comm.rank == 0 else None
    saving = None
    last_iter = starting_iter + config.self_consistency.max_iter
    for current_iter in range(starting_iter + 1, last_iter + 1):
        logger.info("----------------------------------------")
        logger.info(f"Starting iteration {current_iter}.")
        logger.info("----------------------------------------")

        if pole_vertex is not None:
            ratio = first_frequency_pole_ratio(sigma_old, mu_history[-1], pole_vertex)
            config.logger.info(
                f"First-frequency density pole ratio R = {ratio:.4f} (a pole ring of the density ladder at w_1 lies "
                "inside the Brillouin zone once R >= 1)."
            )
            if ratio >= 1.0:
                config.logger.warning(
                    f"First-frequency density pole ratio R = {ratio:.4f} >= 1: the density ladder at the first bosonic "
                    "frequency has a pole ring inside the Brillouin zone, so chi_dens(q, w_1) is unphysical near it. "
                    "The iterate has lost too much local scattering at pi T for the DMFT vertex; mixing and damping "
                    "do not move this threshold."
                )

        proposal = calculate_sigma_proposal(
            sigma_old,
            mu_history[-1],
            u_loc,
            v_nonloc,
            v_nonloc_full,
            sigma_dmft,
            delta_sigma,
            my_irr_q_list,
            mpi_dist_irrk,
            comm,
            current_iter,
            annealer=annealer,
            chunk_budgets=chunk_budgets,
            save_eliashberg=config.eliashberg.perform_eliashberg and current_iter == last_iter,
            lambda_previous=lambda_previous,
            static_magn=static_magn,
            last_chi=last_chi,
            w_r_full=w_r_full,
        )
        mpi_utils.idle_barrier(comm)  # the other nodes sleep while rank 0's node finishes the proposal

        # the tracker decides on this iteration's pair before its step, while every rank still holds the iterate and
        # before the core rows leave rank 0's node; it records sector vectors, so the core windows are read in place
        if tracker is not None:
            request = None
            if comm.rank == 0:
                tracker_window.append(tracker.record(*mixing.core_pair(proposal.core, sigma_old, copy=False)))
                del tracker_window[:-TRACKER_PAIRS]
                if _update_jacobian_tracker(
                    tracker, tracker_window, current_iter, release_iter, anneal_reset_iter, annealer
                ):
                    stab_event_iter = current_iter - 1
                request = tracker.exact_check_request() if config.stabilization.use_exact_jacobian else None
            if comm.bcast(request is not None, root=0) and _run_exact_check(
                tracker, request, sigma_old, mu_history[-1], *exact_args
            ):
                stab_event_iter = current_iter - 1
            request = None  # the start columns are not kept through the next proposal
            if comm.rank == 0:
                _append_jacobian_eigenvalues(tracker, rows_lam, rows_res, rows_damping)

        # every node holds the proposal's shell inputs and its block of the core rows: rank 0's node every row in the
        # proposal's window, every other node its block in a window of its own (the mixing overwrites both)
        meta = None
        if comm.rank == 0:
            meta = proposal.core.mat.shape, proposal.core.mat.dtype, proposal.donor, proposal.average
        core_shape, core_dtype, donor_new, average = comm.bcast(meta, root=0)
        offset = proposal.shell_offset if comm.rank == 0 else None
        offset, offset_win = _share_array_per_node(offset, comm, sc_node_comm, sc_roots_comm)
        if proposal is not None:
            core, core_win = proposal.core.mat, proposal.win
        else:
            core, core_win = mpi_utils.allocate_node_shared_array(sc_node_comm, core_shape, core_dtype)
        core_removed = proposal.removed if comm.rank == 0 else None
        proposal = meta = None
        _scatter_proposal_rows(core, layout)

        # this rank's rows of the proposal's core and of the iterate's core, read in place
        rows, owners, beta = layout.rows, layout.owners, config.sys.beta
        old_mat = sigma_old.compress_q_dimension().mat
        n_old = old_mat.shape[-1] // 2
        old_core = old_mat[..., n_old - config.box.niv_core : n_old + config.box.niv_core]
        new_rows = old_rows = None
        if owners is not None:
            new_rows = SelfEnergy(core[rows], (rows.stop - rows.start, 1, 1), True, True, calc_smom=False, beta=beta)
            old_rows = old_core[rows] if len(old_core) > 1 else old_core
            old_rows = SelfEnergy(old_rows, (len(old_rows), 1, 1), True, True, calc_smom=False, beta=beta)

        # the map's own distance from a fixed point and the change of the proposal, read before the mixing overwrites it
        map_residual = proposal_residual = relative_residual = None
        if owners is not None:
            map_residual = mixing.relative_sigma_residual(new_rows, old_rows, owners)
            if previous_proposal is not None:
                proposal_residual = mixing.relative_core_residual(
                    mixing.positive_core(new_rows), previous_proposal, owners
                )
                previous_proposal[...] = mixing.positive_core(new_rows)
            elif current_iter < last_iter:
                previous_proposal = mixing.positive_core(new_rows).copy()

        # the shell donor of the iterate, whose shell the next one mixes; RPA mode pads with its own tail, whose
        # moments the outermost frequency holds exactly
        donor_old = sigma_dmft
        if config.misc.do_rpa:
            donor_old = _rpa_shell_donor(sigma_old.fit_smom(n_freq=1)[1], sigma_dmft.niv) if comm.rank == 0 else None
            donor_old = comm.bcast(donor_old, root=0)

        logger.info("Applying mixing strategy to the self-energy.")
        history_cap = mixing.mixing_history_cap(current_iter, release_iter, anneal_reset_iter, stab_event_iter)
        # every owner mixes its rows into its node's core window: the accelerated step in place, a damped step copied
        damped_step = False
        if owners is not None:
            if config.stabilization.use_jacobian_stabilization:
                # the record of the pairs the tracker reads: the core window, the rest is the shell tail and its offset
                view = SelfEnergy(core, config.lattice.k_grid.nk, True, True, calc_smom=False, beta=beta)
                _save_sigma_iteration(view, "sigma_dga_proposal", current_iter)
            mixed = mixing.apply_mixing_strategy(new_rows, old_rows, history_cap, mixing_history, tracker, owners)
            damped_step = mixed is not new_rows
            if damped_step:
                core[rows] = mixed.compress_q_dimension().mat
            # Post-mixing step residual (the historical convergence measure; shrinks with the mixing parameter)
            positive = core[rows][..., config.box.niv_core :]
            relative_residual = mixing.relative_core_residual(positive, mixing.positive_core(old_rows), owners)
            mixed = new_rows = old_rows = None
        alpha = tracker.p_eff if tracker is not None else config.self_consistency.mixing
        mpi_utils.idle_barrier(comm)  # the other ranks sleep while the owners mix (rank 0 alone with the tracker)
        damped_step, alpha = comm.bcast((damped_step, alpha) if comm.rank == 0 else None, root=0)

        # every node completes its core window with the other nodes' rows and builds the next iterate in a new window,
        # the shell of every row its own; the background write of the previous iterate reads the old window
        _gather_mixed_rows(core, layout)
        if saving is not None:
            saving.result()
        shell = donor_new, offset, average, donor_old, damped_step, alpha
        sigma_new, new_win, shell_removed = _build_iterate_window(core, sigma_old, shell, sc_node_comm, beta)
        if comm.rank == 0 and average:
            removed = max(core_removed, shell_removed)
            logger.info(f"Time-reversal average of the self-energy proposal: largest change {removed:.3e}.")
        # no view into the freed windows survives them (the tracker's proposal record, the previous iterate's write)
        core = offset = shell = sigma_old = old_mat = old_core = positive = view = None
        for win in (core_win, offset_win, sigma_win):
            mpi_utils.free_shared_window(win, sc_node_comm)
        sigma_win = new_win

        # every rank solves mu on its momenta of the shared sigma; the moments are fitted on rank 0 alone, in RPA mode
        # from the outermost frequency, where its shell holds them exactly however short the shell is
        old_mu = mu_history[-1]
        n_freq = 1 if config.misc.do_rpa else None
        smom0, smom1 = comm.bcast(sigma_new.fit_smom(n_freq) if comm.rank == 0 else None, root=0)
        sigma_new._smom0, sigma_new._smom1 = smom0, smom1
        donor = _rpa_shell_donor(smom1, sigma_dmft_full.niv) if config.misc.do_rpa else sigma_dmft_full
        sigma_rows, mu_dist, wedge = _lattice_sum_momenta(sigma_new, donor, mpi_dist_irrk, mpi_dist_fullbz, comm)
        config.sys.mu = update_mu(
            old_mu,
            config.sys.n,
            config.lattice.hamiltonian.get_ek(),
            sigma_rows,
            config.sys.beta,
            smom0,
            logger=logger,
            mpi_dist=mu_dist,
            orbitals=config.sys.correlated_orbitals,
            wedge=wedge,
        )
        del sigma_rows
        mu_history.append(config.sys.mu)
        logger.info(f"Updated mu from {old_mu} to {config.sys.mu}.")
        if comm.rank == 0:
            # rewritten every iteration, so a run killed while iterating can still be resumed from its last mu
            np.save(os.path.join(config.output.output_path, "mu_history.npy"), mu_history)

        # new occupation matrix and energies from the new Green's function (outside the asympt region its shell is the
        # donor's: DMFT, or RPA mode's own tail); k-distributed, so no rank builds the whole DMFT-box Green's function
        n_current, config.sys.occ, config.sys.occ_k, ekin, epot = sde_kernels.update_occ_and_energies_distributed(
            sigma_new, donor, mpi_dist_fullbz, config.sys.mu, mpi_dist_irrk if wedge is not None else None
        )
        logger.info(f"Filling of the updated Green's function: {n_current:.6f} (target {config.sys.n:.6f}).")
        logger.info(f"Kinetic energy: {ekin:.4f} [t or eV].")
        logger.info(f"Potential energy: {epot:.4f} [t or eV].")
        logger.info(f"Total energy: {(ekin + epot):.4f} [t or eV].")

        if config.self_consistency.max_iter > 1:
            logger.info("Updated occupation matrix from new Green's function.")

        if comm.rank == 0:
            # the write gets an object of its own: saving decompresses it, while the loop goes on with sigma_new
            compressed = sigma_new.has_compressed_q_dimension
            view = SelfEnergy(sigma_new.mat, sigma_new.nq, sigma_new.full_niv_range, compressed, calc_smom=False)
            saving = saver.submit(_save_sigma_iteration, view, output_files.SIGMA_DGA, current_iter)
            if not config.self_consistency.save_sigma_iterates:
                # the earlier writes have finished; only this iteration's file is still being written
                _prune_sigma_iterates(current_iter - config.self_consistency.mixing_history_length)

        logger.info("Checking self-consistency convergence.")
        if comm.rank == 0 and current_iter > starting_iter + 1:
            # the returned iterate, the proposal and the proposal's distance from its iterate must all have settled
            eps = _effective_epsilon(annealer)
            # a damped step moves at p_eff, so its residual is read at the configured mixing, as without the tracker
            step_scale = tracker.p_config / tracker.p_eff if damped_step and tracker is not None else 1.0
            step_residual = abs(relative_residual) * step_scale
            sigma_converged = max(step_residual, proposal_residual, map_residual) < eps
            scaled = f", {step_residual:.3e} at the configured mixing" if step_scale != 1.0 else ""
            logger.info(
                f"Self-energy convergence: {sigma_converged} (relative step residual={relative_residual:.3e}{scaled}, "
                f"proposal step residual={proposal_residual:.3e}, map residual={map_residual:.3e}, "
                f"epsilon={eps:.3e})."
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
                # the iterate is driven the minimum count of the modes outside their band away from the start (capped)
                n_min, n_resolved = comm.bcast(
                    (tracker.minimum_iterations(), tracker.minimum_iterations(outside_band=True))
                    if tracker is not None and comm.rank == 0
                    else (0, 0)
                )
                done = current_iter - starting_iter
                if done < min(n_resolved, config.self_consistency.max_iter):
                    logger.info(
                        f"Self-consistency of sigma and mu reached at iteration {current_iter}, {done} iterations "
                        f"after the start, below the minimum {n_resolved} the certified modes outside their "
                        f"undecidable band ask for (p_eff = {tracker.p_eff:.4f}): continuing, so that a flat "
                        "direction can settle."
                    )
                else:
                    pure_converged = True
                    logger.info(f"Self-consistency of sigma and mu reached at iteration {current_iter}.")
                    if tracker is not None and comm.rank == 0:
                        logger.info(
                            f"Jacobian tracker at convergence: {0 if tracker.q is None else tracker.q.shape[1]} "
                            f"flipped directions, p_eff={tracker.p_eff:.4f}, rho={tracker.predicted_rate():.4f}."
                        )
                    if done < n_min:
                        logger.warning(
                            f"Jacobian tracker: converged after {done} iterations, below the minimum {n_min} the "
                            f"certified spectrum suggests at the damping each certified direction takes (eps = "
                            f"{tracker.certified_margin:.3e}, p_eff = {tracker.p_eff:.4f} on the directions the "
                            "installed map leaves to the scalar step); a flat direction may not have settled."
                        )
                    break
        else:
            logger.info("Self-consistency not reached.")

    if saver is not None:
        saver.shutdown()
        if saving is not None:
            saving.result()
    mixing_history = tracker_window = previous_proposal = w_r_full = None  # nothing after the loop reads them
    if comm.rank == 0 and not config.self_consistency.save_sigma_iterates:
        _prune_sigma_iterates(current_iter)
    if tracker is not None and comm.rank == 0:
        tracker.release_after_loop()

    # the Eliashberg step reads the intermediates of the final iterate, which the loop writes only in its last scheduled
    # iteration; a loop that converged earlier builds them once from the converged sigma, still shared per node here
    if config.eliashberg.perform_eliashberg and current_iter < last_iter:
        logger.info("Building the Eliashberg intermediates from the converged self-energy.")
        calculate_sigma_proposal(
            sigma_old,
            mu_history[-1],
            u_loc,
            v_nonloc,
            v_nonloc_full,
            sigma_dmft,
            delta_sigma,
            my_irr_q_list,
            mpi_dist_irrk,
            comm,
            current_iter,
            annealer=annealer,
            chunk_budgets=chunk_budgets,
            save_eliashberg=True,
            eliashberg_only=True,
            lambda_previous=lambda_previous,
            static_magn=static_magn,
            last_chi=last_chi,
        )

    # without a lambda correction the proposals keep their susceptibilities on the ranks, so only the last is written
    for chi in last_chi.values():
        _write_physical_susceptibility(chi, mpi_dist_irrk, static_magn=static_magn)
    last_chi = None

    # the exact Jacobian and the interpolation read the final sigma on every rank, so they run while the shared window
    # still holds it
    if pure_converged and tracker is not None and config.stabilization.use_exact_jacobian:
        _install_exact_jacobian(tracker, sigma_old, mu_history[-1], *exact_args)

    if config.self_energy_interpolation.do_interpolation:
        beta_target = config.self_energy_interpolation.beta_target
        niv_target = config.self_energy_interpolation.niv_target
        sigma_interpolated = interpolate_sigma(sigma_old, beta_target, niv_target, comm)
        if comm.rank == 0:
            sigma_interpolated.save(
                name=f"sigma_dga_interpolated_beta{beta_target}_niv{niv_target}", output_dir=config.output.output_path
            )
            logger.info(f"Interpolated the final sigma to beta={beta_target} and niv={niv_target}.")
        del sigma_interpolated

    # only rank 0 reads the result after the loop: it keeps a private copy, every other rank drops its window view
    if sigma_win is not None:
        sigma_old.mat = sigma_old.mat.copy() if comm.rank == 0 else None
        view = None  # rank 0's view of the last iterate, whose write has finished
        mpi_utils.free_shared_window(sigma_win, sc_node_comm)
    if sc_node_comm is not None:
        if layout.owners is not None:
            layout.owners.Free()
        sc_roots_comm.Free()
        sc_node_comm.Free()

    if comm.rank == 0:
        np.save(os.path.join(config.output.output_path, "mu_history.npy"), mu_history)
        logger.info("Saved mu history as numpy array.")

    if tracker is not None and comm.rank == 0:
        _save_jacobian_spectrum(tracker, pure_converged, rows_lam, rows_res, rows_damping)

    # oz_coeff.txt reports the final susceptibility only, so the fit runs once instead of in every iteration
    if static_magn:
        perform_ornstein_zernike_fit(static_magn.pop())
    mpi_utils.idle_barrier(comm)  # the other ranks sleep while rank 0 fits

    return sigma_old if comm.rank == 0 else None

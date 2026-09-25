# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Non-local ladder DGA step - the parallel-heavy core of the code. Starting from the local irreducible vertex
:math:`\Gamma_{r}` and the bare interaction, the functions here build, per momentum :math:`\mathbf{q}` and spin channel,
the bubble :math:`\chi^{\mathrm{q}\nu}_{0}`, the auxiliary susceptibility :math:`\chi^{*;\mathrm{q}}_{r}`, the three-leg
vertex :math:`\gamma^{\mathrm{q}\nu}_{r}`, the physical susceptibility :math:`\chi^{\mathrm{q}}_{r}` (with shell and
optional :math:`\lambda`-correction) and the self-energy kernel, then contract the kernel with the Green's function to
get the momentum-dependent self-energy :math:`\Sigma^{\mathrm{k}}_{12}`. Both a q-loop and an FFT variant of the heavy
contractions are provided, distributed over MPI ranks. The whole thing is wrapped in a self-consistency loop with
chemical-potential adjustment and self-energy mixing (linear / Pulay / Anderson). Equation numbers refer to the author's
master's thesis (Chapters 3 & 4).
"""

import glob
import os
import pickle
import re
from collections.abc import Callable

import mpi4py.MPI as MPI
import numpy as np
import scipy as sp
from scipy import optimize as opt

import dgamore.config as config
import dgamore.mpi_utils as mpi_utils
from dgamore.brillouin_zone import KGrid
from dgamore.bubble_gen import BubbleGenerator
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction, update_mu
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.jacobian_stabilization import (
    JACOBIAN_FILE,
    JacobianTracker,
    TRACKER_PAIRS,
    load_spectrum,
    to_vec,
)
from dgamore.local_four_point import LocalFourPoint
from dgamore.lambda_ops import LambdaAnnealer, LambdaCorrection, MultiOrbitalLambdaCorrection
from dgamore.matsubara_frequencies import MFHelper
from dgamore import memory_estimator
from dgamore.memory_estimator import SLICE_CHUNK_BYTES
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import DTYPE, SpinChannel, deferred_collection
from dgamore.self_energy import SelfEnergy

_JACOBIAN_ROW_WIDTH = TRACKER_PAIRS - 1  # Ritz values per row of the per-iteration eigenvalue traces: every value an
# estimate can have, so a near-critical mode of small modulus is never crowded out by stiffer ones
_FIRST_FREQUENCY_RATIO = 2.0  # largest compound norm of chi_phys at the first bosonic frequency, in units of the
# static one, above which the loop warns of a finite-frequency pole
_RESTRICTION_PIN_FACTOR = 2.0  # a crossed static mode of the restriction is pinned at this multiple of the largest
# healthy static susceptibility eigenvalue of the channel
_RESTRICTION_FALLBACK_FLOOR = 1e-2  # floor of the inverse static eigenvalues when no static block is healthy


def get_hartree_fock(
    u_loc: LocalInteraction, v_nonloc: Interaction, occ: np.ndarray | None = None, occ_k: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Returns the Hartree-Fock term separately for the local and non-local interaction. Since we are always SU(2)-symmetric,
    the sum over the spins of the first term in Eq. (4.55) in Anna Galler's thesis results in a simple factor of 2. This
    can be seen in my master's thesis, Eq. (3.55). The Hartree-Fock term is given by

    .. math:: \Sigma^{\mathbf{k}}_{\mathrm{HF};12} = 2\sum_{ab}(U_{12ab} + V^{\mathbf{q}=0}_{12ab}) n_{ba}
        - \frac{1}{n_{\mathbf{q}}} \sum_{\mathbf{q}ab} (U_{1ba2} + V^{\mathbf{q}}_{1ba2}) n^{\mathbf{k}-\mathbf{q}}_{ba}

    where the first sum is the Hartree and the second the Fock term, both in the stored equation layout
    (:math:`U'` at ``aabb``, see :meth:`~dgamore.hamiltonian.Hamiltonian.read_umatrix`): the Hartree contraction
    places the external orbitals on the first two slots and the Fock contraction on the outer slots, exactly as
    :func:`dgamore.local_sde.get_local_hartree_fock` does.

    The Fock momentum sum is a circular convolution over the periodic Brillouin zone, so it is evaluated with the
    convolution theorem instead of an explicit q-loop: :math:`\Sigma^{\mathbf{k}}_{\mathrm{F}} =
    -\tfrac{1}{n_{\mathbf{q}}}\,\mathcal{F}^{-1}
    \big[\mathcal{F}[(U+V)^{\mathbf{q}}_{1ba2}]\,\mathcal{F}[n_{ba}]\big]` (a plain convolution, since the shift
    is :math:`n^{\mathbf{k}-\mathbf{q}}`). This is
    :math:`O(n_{\mathbf{k}} \log n_{\mathbf{k}})` and materializes only R-space ``[k, o^4]``/``[k, o^2]`` arrays, never
    a ``[q, k]`` occupation block, so the whole full-BZ term is computed on one rank without a q-distribution.

    The term is linear in the occupations, so a change of the occupations gives the change of the term.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}` on the full q-grid (see :class:`Interaction`).
    :param occ: The k-averaged occupation ``[o1, o2]``; None reads ``config.sys.occ``.
    :param occ_k: The k-resolved occupation ``[kx, ky, kz, o1, o2]``; None reads ``config.sys.occ_k``.
    :return: The tuple ``(hartree, fock)`` of self-energy contributions, broadcastable to ``[k, o1, o2, v]``.
    """
    occ = config.sys.occ if occ is None else occ
    occ_k = config.sys.occ_k if occ_k is None else occ_k
    v_q0 = v_nonloc.find_q((0, 0, 0))
    # equation layout: Hartree 2 U_{12ab} n_{ba} with the external orbitals on the first two slots; the Fock term
    # below is -U_{1ba2} n_{ba} with them on the outer slots, like local_sde.get_local_hartree_fock
    hartree = 2 * (u_loc + v_q0).times("qabcd,dc->ab", occ)

    nb = config.sys.n_bands
    nk_tot = np.prod(config.lattice.nk)

    # Fourier the interaction and the occupation to R-space, contract the summed orbitals pointwise per R and
    # transform back: convolution theorem for the n^{k-q} sum.
    w_r = (u_loc + v_nonloc).fft(copy=False)
    w_r_mat = w_r.decompress_q_dimension().mat  # [kx, ky, kz, a, b, c, d]
    occ_r = sp.fft.fftn(occ_k.astype(w_r_mat.dtype, copy=False), axes=(0, 1, 2))  # [kx, ky, kz, d, c]

    fock_r = np.einsum("xyzadcb,xyzdc->xyzab", w_r_mat, occ_r, optimize=True)
    fock = sp.fft.ifftn(fock_r, axes=(0, 1, 2), overwrite_x=True).reshape(nk_tot, nb, nb)
    fock *= -1.0 / nk_tot
    return hartree[None, ..., None], fock[..., None]  # [k,o1,o2,v]


def create_inverse_auxiliary_chi_r_q(
    gamma_r: LocalFourPoint, gchi0_q_inv: FourPoint, u_r: LocalInteraction
) -> FourPoint:
    r"""
    Assembles the Bethe-Salpeter matrix whose inversion yields the auxiliary susceptibility (Eq. (3.60) in my
    master's thesis),

    .. math:: M^{\mathrm{q}\nu\nu'}_{1234} = (\chi^{\mathrm{q}\nu}_{0;1234})^{-1}\delta_{\nu\nu'} + (\Gamma^{\omega\nu\nu'}_{r;1234}-U_{r;1234})/\beta^2,

    in a **single** two-fermion block: the result is broadcast-filled with the scaled local vertex over the momentum
    axis, the inverse bubble is added on the fermionic frequency diagonal in place (see
    :meth:`~dgamore.local_four_point.LocalFourPoint.add_on_vn_diagonal`) and the channel interaction is subtracted
    in place. The former add/extend/subtract chain held two of these blocks alive at its peak (the diagonally
    extended bubble plus the out-of-place result). Neither input is mutated.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` (full or half bosonic range; read via a
        half-range view).
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` (half bosonic range, one
        fermionic dimension).
    :param u_r: The channel-projected local interaction :math:`U_{r}`. The non-local :math:`V^{\mathbf{q}}_{r}` does
        not enter: the ladder vertex :math:`\Gamma^{\mathrm{q}}_{r} = \Gamma^{\omega}_{r} + V^{\mathbf{q}}_{r}` minus the
        crossing-symmetric :math:`\mathcal{U}^{\mathbf{q}}_{r} = U_{r} + V^{\mathbf{q}}_{r}` leaves
        :math:`\Gamma^{\omega}_{r} - U_{r}`, so :math:`V^{\mathbf{q}}` reaches the ladder only through the physical
        susceptibility and the self-energy kernel.
    :return: The assembled matrix :math:`M^{\mathrm{q}}` as a :class:`FourPoint` (half niw range, two fermionic
        dimensions).
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


def create_auxiliary_chi_r_q(gamma_r: LocalFourPoint, gchi0_q_inv: FourPoint, u_loc: LocalInteraction) -> FourPoint:
    r"""
    Returns the auxiliary susceptibility, see Eq. (3.60) in my master's thesis,

    .. math:: \chi^{*;\mathrm{q}\nu\nu'}_{r;abcd} = ((\chi^{\mathrm{q}\nu}_{0;abcd})^{-1} + (\Gamma^{\omega\nu\nu'}_{r;abcd}-U_{r;abcd}-V^{\mathbf{q}}_{r;abcd})/\beta^2)^{-1},

    with the matrix to invert assembled in a single block by :func:`create_inverse_auxiliary_chi_r_q`.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}`.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` (core box).
    :param u_loc: The bare local interaction :math:`U`.
    :return: The momentum-dependent auxiliary susceptibility :math:`\chi^{*;\mathrm{q}}_{r}` as a :class:`FourPoint`.
    """
    u_r = u_loc.as_channel(gamma_r.channel)
    return create_inverse_auxiliary_chi_r_q(gamma_r, gchi0_q_inv, u_r).invert(False)


def create_auxiliary_chi_r_q_sum(
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    u_loc: LocalInteraction,
    chunk_bytes: int | None = None,
    rhs: FourPoint | None = None,
) -> FourPoint:
    r"""
    Returns the sum over the auxiliary susceptibility, see Eq. (3.60) in my master's thesis,

    .. math:: \sum_{\nu'}\chi^{*;\mathrm{q}\nu\nu'}_{r;1234} = \sum_{\nu'}((\chi^{\mathrm{q}\nu}_{0;1234})^{-1} + (\Gamma^{\omega\nu\nu'}_{r;1234}-U_{r;1234})/\beta^2)^{-1},

    walking the rank-local momenta and their bosonic axis in byte-bounded chunks: each chunk assembles its window of
    the Bethe-Salpeter matrix (see :func:`create_inverse_auxiliary_chi_r_q`) and back-substitutes only the
    :math:`\nu'`-summed columns through the per-slice factorization of
    :meth:`~dgamore.four_point.FourPoint.invert_and_sum_over_last_vn_v2`, so the transient never exceeds a few
    chunk budgets regardless of the box size while small problems degenerate into a single batched evaluation. The
    orbital pairs without vertex (see :meth:`~dgamore.local_four_point.LocalFourPoint.orbital_pairs_without_vertex`),
    found once per channel, are eliminated frequency by frequency inside every slice.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` (full or half bosonic range).
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` (core box).
    :param u_loc: The bare local interaction :math:`U`.
    :param chunk_bytes: Chunk byte budget of the build; defaults to the :data:`SLICE_CHUNK_BYTES` floor (the
        pipeline passes the budget the driver sizes from the memory estimate, see
        :func:`~dgamore.memory_estimator.max_chunk_budget`). Every budget yields the same bits.
    :param rhs: Right-hand sides in the layout of the result (half niw range, one fermionic dimension); the chunks
        then solve the Bethe-Salpeter system for them instead of summing its inverse (see
        :meth:`~dgamore.four_point.FourPoint.invert_and_sum_over_last_vn_v2`), and the result is that solution.
        None sums, as the ladder does.
    :return: The frequency-summed auxiliary susceptibility :math:`\sum_{\nu'}\chi^{*;\mathrm{q}}_{r}` as a
        :class:`FourPoint` (half niw range, one fermionic dimension), or the solution for ``rhs``.
    """
    u_r = u_loc.as_channel(gamma_r.channel)
    gamma_half = gamma_r.copy().to_half_niw_range() if gamma_r.full_niw_range else gamma_r
    inactive = gamma_half.orbital_pairs_without_vertex(u_r)
    if 0 < inactive.size < gamma_half.n_bands**2:
        config.logger.info(
            f"Auxiliary susceptibility ({gamma_r.channel.value}): {inactive.size} of {gamma_half.n_bands**2} orbital "
            f"pairs carry no vertex and are eliminated frequency by frequency."
        )

    budget = SLICE_CHUNK_BYTES if chunk_bytes is None else chunk_bytes
    n_q, n_w = gchi0_q_inv.current_shape[0], gchi0_q_inv.current_shape[-2]
    one_wn_bytes = gamma_half.current_shape[-1] ** 2 * gamma_half.n_bands**4 * np.dtype(gamma_half.mat.dtype).itemsize
    w_chunk = max(1, int(budget // one_wn_bytes))
    q_group = max(1, int(budget // max(one_wn_bytes * n_w, 1)))

    chi_r_q_sum_mat = np.empty_like(gchi0_q_inv.mat)
    # deferred_collection batches the gc pass of the per-chunk temporaries into one collection at the end
    # (a full gc walk per released slice dominates the wall time of the chunk loop otherwise).
    with deferred_collection():
        for q_start in range(0, n_q, q_group):
            q_stop = min(n_q, q_start + q_group)
            gchi0_q = gchi0_q_inv.take_q_index_slice(q_start, q_stop)
            rhs_q = None if rhs is None else rhs.take_q_index_slice(q_start, q_stop)
            for w_start in range(0, n_w, w_chunk):
                w_stop = min(n_w, w_start + w_chunk)
                chunk = create_inverse_auxiliary_chi_r_q(
                    gamma_half.take_wn_slice(w_start, w_stop), gchi0_q.take_wn_slice(w_start, w_stop), u_r
                )
                chunk_rhs = None if rhs is None else rhs_q.take_wn_slice(w_start, w_stop)
                chi_r_q_sum_mat[q_start:q_stop, ..., w_start:w_stop, :] = chunk.invert_and_sum_over_last_vn_v2(
                    config.sys.beta, inactive, chunk_rhs
                ).mat
    return FourPoint(chi_r_q_sum_mat, gamma_r.channel, config.lattice.nk, 1, 1, False, has_compressed_q_dimension=True)


def create_vrg_r_q(gchi_aux_q_r_sum: FourPoint, gchi0_q_inv: FourPoint) -> FourPoint:
    r"""
    Returns the momentum-dependent three-leg vertex, see Eq. (3.63) in my master's thesis,
    :math:`\gamma^{\mathrm{q}\nu}_{r;1234} = \beta \sum_{ab} \sum_{\nu'} (\chi^{\mathrm{q}\nu}_{0;12ab})^{-1}
    \chi^{*;\mathrm{q}\nu\nu'}_{r;ba34}`.

    :param gchi_aux_q_r_sum: The frequency-summed auxiliary susceptibility
        :math:`\sum_{\nu'}\chi^{*;\mathrm{q}\nu\nu'}_{r}`.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` (core box).
    :return: The three-leg vertex :math:`\gamma^{\mathrm{q}\nu}_{r}` (``vrg``) as a :class:`FourPoint`.
    """
    return (gchi0_q_inv @ gchi_aux_q_r_sum).scale(config.sys.beta)


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

    :param chi_phys_q_r: The physical susceptibility :math:`\chi^{\mathrm{phys};\mathrm{q}}_{r}`.
    :param gchi0_q_full_sum: The frequency-summed bare bubble over the full box.
    :param gchi0_q_core_sum: The frequency-summed bare bubble over the core box.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :return: The shell-corrected physical susceptibility :math:`\chi^{\mathrm{q}}_{r}` as a :class:`FourPoint`.
    """
    return (
        (chi_phys_q_r + gchi0_q_full_sum - gchi0_q_core_sum).invert()
        + (u_loc.as_channel(chi_phys_q_r.channel) + v_nonloc.as_channel(chi_phys_q_r.channel))
    ).invert()


def restrict_chi_phys_to_positive_eigenvalues(
    chi_phys_q_r: FourPoint, comm: MPI.Comm | None = None, floor: float | None = None
) -> tuple[FourPoint, int]:
    r"""
    Regularizes the physical susceptibility per momentum with the two bounds a bosonic susceptibility obeys, positive
    semi-definite and decreasing in :math:`|\omega|`, touching only the compound blocks that violate them.

    - Static slice: the eigenvalues of the Hermitian part of the inverse compound matrix
      :math:`(\chi^{(\mathbf{q},\omega=0)}_{r;1234})^{-1}` are floored at :math:`+\text{floor}` (the skew-Hermitian
      part is kept) and the block is inverted back. A negative eigenvalue of the inverse marks a crossed pole of the
      Bethe-Salpeter equation (an unphysical branch of the ladder, e.g. the high-temperature charge-channel
      instability); flooring it pins the corresponding susceptibility eigenvalue at :math:`1/\text{floor}`. Without
      an explicit floor it is the smallest inverse eigenvalue of the healthy static blocks (those without a
      negative one) over all ranks divided by ``_RESTRICTION_PIN_FACTOR``, so a crossed mode is pinned at that
      multiple of the channel's largest healthy static susceptibility; ``_RESTRICTION_FALLBACK_FLOOR`` when no
      block is healthy.
    - Finite frequencies: the eigenvalues of the Hermitian part of :math:`\chi^{(\mathbf{q},\omega_n)}_{r;1234}` are
      clipped into :math:`[-c_{\mathbf{q}}, c_{\mathbf{q}}]`, with :math:`c_{\mathbf{q}}` the largest eigenvalue of the
      (restricted) static block, on their own eigenvectors, and so are those of the skew-Hermitian part (:math:`i` times
      a Hermitian matrix), which vanishes for a physical susceptibility and spikes with the pole. A physical
      :math:`\chi(\mathbf{q}, i\omega_n)` lies in :math:`[0, \chi(\mathbf{q}, 0)]`, so a value beyond the static one
      marks a pole of the ladder at that frequency, which the static bound cannot see; the symmetric window leaves the
      small negative values the frequency-box truncation produces untouched.

    Blocks that satisfy both bounds are returned bit for bit, so on a healthy susceptibility the count is zero and the
    restriction changes nothing. Healthy eigenpairs of a restricted block, including legitimately negative
    off-diagonal matrix elements, pass through unchanged. For a single band the compound block is a scalar and the
    two bounds reduce to pinning a negative static value and clipping the real and imaginary parts of the
    finite-frequency values.

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
    :math:`\chi^{(\mathbf{q},\omega=0)}_{r;1234}` over all rank-local momenta. A physical static susceptibility is
    positive semi-definite per momentum, so a significantly negative value flags that the ladder sits on an unphysical
    (past-pole) branch. Expects the object with a compressed momentum dimension and no fermionic frequency dimensions.

    :param chi_phys_q_r: The physical susceptibility :math:`\chi^{\mathrm{q}}_{r;1234}`.
    :return: The minimum eigenvalue as a float.
    """
    w0 = chi_phys_q_r.niw if chi_phys_q_r.full_niw_range else 0
    n = chi_phys_q_r.n_bands**2
    static = chi_phys_q_r.mat[..., w0].transpose(0, 1, 2, 4, 3).reshape(-1, n, n)
    return float(np.linalg.eigvalsh(0.5 * (static + np.conj(np.swapaxes(static, -1, -2)))).min())


def max_compound_norm(chi_phys_q_r: FourPoint, w_offset: int) -> float:
    r"""
    Returns the largest spectral norm of the compound blocks :math:`\chi^{(\mathbf{q},\omega_n)}_{r;1234}` at the
    bosonic frequency :math:`\omega_n` with :math:`n` = ``w_offset`` (0 is the static slice) over all rank-local
    momenta. A bosonic susceptibility is positive and decreasing in :math:`|\omega_n|`, so the first-frequency norm
    exceeding the static one flags a pole at that frequency, which the static monitor
    (:func:`min_static_compound_eigenvalue`) cannot see. Expects the object with a compressed momentum dimension and
    no fermionic frequency dimensions.

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
    Logs the pole monitors of the physical susceptibility and warns when one fires: the smallest static compound
    eigenvalue (:func:`min_static_compound_eigenvalue`), significantly negative on a past-pole branch; the smallest
    eigenvalue of the Hermitian part of the static inverse compound susceptibility, the distance of the static
    Bethe-Salpeter equation from its pole (see :meth:`~dgamore.lambda_ops.LambdaAnnealer._static_gap`); and the largest
    compound norm at the first bosonic frequency against the static one (:func:`max_compound_norm`), above
    ``_FIRST_FREQUENCY_RATIO`` times it at a pole of the ladder at that frequency. All are reduced over the ranks.

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
        f"Minimum static compound eigenvalue of chi_phys ({channel}): {min_eig:.6f}, of 1/chi_phys: "
        f"{min_inverse_eig:.6f}."
    )
    config.logger.info(f"Largest compound norm of chi_phys ({channel}) at w_1: {norm_w1:.6f} (static {norm_w0:.6f}).")
    if min_eig < -5e-2:
        config.logger.warning(
            f"The static physical susceptibility ({channel}) is not positive semi-definite "
            f"(minimum eigenvalue {min_eig:.3f}): the ladder sits on an unphysical (past-pole) branch and derived "
            "quantities (self-energy, Eliashberg eigenvalues) might be unreliable."
        )
    if norm_w1 > _FIRST_FREQUENCY_RATIO * norm_w0:
        config.logger.warning(
            f"The physical susceptibility ({channel}) at the first bosonic frequency exceeds its static value "
            f"(largest compound norm {norm_w1:.3f} against {norm_w0:.3f}): a bosonic susceptibility decreases with "
            "|w|, so the ladder has a pole at w_1 the static monitor cannot see, and the iteration may be heading "
            "to an unphysical fixed point."
        )


def first_frequency_local_vertex() -> float | None:
    r"""
    Returns the local density vertex that sets the first-frequency pole of the density ladder,

    .. math:: \gamma_{\mathrm{loc}} = 1/\chi^{\omega_1\nu_0\nu_0}_{\mathrm{d,loc}} - 1/\chi^{\omega_1\nu_0}_{0,\mathrm{loc}},
        \qquad \chi^{\omega_1\nu_0}_{0,\mathrm{loc}} = -\beta G(\nu_0) G(\nu_0 - \omega_1),

    at :math:`\nu_0 = \pi T`, from the impurity generalized susceptibility and Green's function the local part saves
    before the loop (``gchi_dens_loc.npy`` on the half bosonic range, ``g_dmft.npy``). See
    :func:`first_frequency_pole_ratio` for its use.

    :return: :math:`\mathrm{Re}\,\gamma_{\mathrm{loc}}`, or ``None`` for more than one band or without the local
        files (the monitor is a diagnostic and never stops a run).
    """
    gchi_path = os.path.join(config.output.output_path, "gchi_dens_loc.npy")
    g_path = os.path.join(config.output.output_path, "g_dmft.npy")
    if not (os.path.exists(gchi_path) and os.path.exists(g_path)):
        return None
    # ponytail: one band only; several bands need the orbital-compound block at nu0 and its leading eigenvalue
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
    :math:`R = \gamma_{\mathrm{loc}}\,\beta\,\mathrm{Re}\langle G^{\mathbf{k}}(\nu_0) G^{\mathbf{k}}(\nu_0 -
    \omega_1)\rangle_{\mathbf{k}}` with :math:`\nu_0 = \pi T`. At :math:`\omega_1` the density Bethe-Salpeter matrix is
    dominated by the bubble element of the pair :math:`(\nu_0, \nu_0 - \omega_1) = (\pi T, -\pi T)`; its 1x1 Schur
    complement :math:`\gamma_{\mathrm{loc}} - 1/(\beta C(\mathbf{q}))`, :math:`C(\mathbf{q}) = \mathrm{Re}\langle
    G^{\mathbf{k}}(\nu_0) G^{\mathbf{k}+\mathbf{q}}(\nu_0 - \omega_1)\rangle_{\mathbf{k}}`, is largest at
    :math:`\mathbf{q} = 0`, so a pole ring lies inside the Brillouin zone exactly when :math:`R \geq 1`. The DMFT
    Green's function stays below 1 (0.87 at :math:`\beta = 12.5`); a lattice Green's function that has lost local
    scattering at :math:`\pi T` pushes :math:`R` across it.

    :param sigma: The iterate :class:`SelfEnergy` (one band; full-BZ or momentum-local, either momentum layout).
    :param mu: Its chemical potential :math:`\mu`.
    :param gamma_loc: The local vertex of :func:`first_frequency_local_vertex`.
    :return: The ratio :math:`R`.
    """
    beta, niv = config.sys.beta, sigma.niv
    ek = config.lattice.hamiltonian.get_ek().reshape(-1)
    s = sigma.mat.reshape(-1, 2 * niv)
    nu0 = np.pi / beta
    g_nu0 = 1.0 / (1j * nu0 + mu - ek - s[:, niv])
    g_minus_nu0 = 1.0 / (-1j * nu0 + mu - ek - s[:, niv - 1])
    return float(gamma_loc * beta * np.mean(g_nu0 * g_minus_nu0).real)


def _log_first_frequency_pole_ratio(ratio: float) -> None:
    r"""
    Logs the first-frequency pole ratio of the iterate a proposal is about to evaluate (see
    :func:`first_frequency_pole_ratio`) and warns once it reaches 1.

    :param ratio: The ratio :math:`R`.
    :return: None.
    """
    config.logger.info(
        f"First-frequency density pole ratio R = {ratio:.4f} (a pole ring of the density ladder at w_1 lies inside "
        "the Brillouin zone once R >= 1)."
    )
    if ratio >= 1.0:
        config.logger.warning(
            f"First-frequency density pole ratio R = {ratio:.4f} >= 1: the density ladder at the first bosonic "
            "frequency has a pole ring inside the Brillouin zone, so chi_dens(q, w_1) is unphysical near it. The "
            "iterate has lost too much local scattering at pi T for the DMFT vertex; mixing and damping do not move "
            "this threshold."
        )


def calculate_sigma_dc_kernel(f_dc_loc: LocalFourPoint, gchi0_q: FourPoint, u_loc: LocalInteraction) -> FourPoint:
    r"""
    Returns the double-counting kernel for the self-energy calculation - the local contribution that the two
    ladder terms of the Schwinger-Dyson equation count twice, closed with the exchange attachment,

    .. math:: -\Sigma^{\mathrm{dc}}_{12} = +\frac{1}{\beta}\sum_{\mathrm{q}\nu'} U_{acb2}\,
        F^{\omega\nu\nu'}_{\mathrm{dc};1dfe}\,\chi^{\mathrm{q}\nu'}_{0;efcb}\,G^{\mathrm{k}-\mathrm{q}}_{da},

    assembled in the contraction layout of :func:`calculate_kernel_r_q` (the factor :math:`-2` relative to the
    global :math:`-\tfrac12` prefactor of the sigma contraction realizes the sign above). Keeping the full ladder
    vertices in both bracket terms and adding this correction leaves exactly one local copy, so the local limit of
    the total reproduces the local Schwinger-Dyson equation. The closed vertex :math:`F_{\mathrm{dc}}` is the local
    part of the transversal term, :math:`\tfrac12(F_{\mathrm{d}} + 3F_{\mathrm{m}})` (see
    :func:`~dgamore.local_sde.double_counting_vertex`), whose local content cancels term by term against the
    density form of the local equation for any band count. The frequency sum acts on the
    *second* fermionic argument of the local double-counting vertex; since ``f_dc_loc`` stores the asymmetric
    :math:`2 n_{\nu,\mathrm{full}} \times 2 n_{\nu,\mathrm{core}}` box with the **first** index on the full box
    (see :func:`~dgamore.local_sde.create_full_vertex_from_gamma`), the compound symmetry of the symmetrized local
    vertex, :math:`F^{\omega\nu\nu'}_{1234} = F^{\omega\nu'\nu}_{4321}`, is used to read the stored first index as
    the summed :math:`\nu'`.

    :param f_dc_loc: The local double-counting vertex :math:`F_{\mathrm{dc}}` (see
        :func:`~dgamore.local_sde.double_counting_vertex`).
    :param gchi0_q: The momentum-dependent bare bubble :math:`\chi^{\mathrm{q}\nu}_{0}` (full fermionic box).
    :param u_loc: The bare local interaction :math:`U`.
    :return: The double-counting kernel (contraction layout) as a :class:`FourPoint`, cut to the core fermionic box.
    """
    # the stored full-box first index supplies the summed v' via F_{1dfe}(v,v') = F_{efd1}(v',v): the stored slots
    # (e,f,d,1) then pair directly with chi0's row pair (e,f) - the standard-product reversal F_{..fe} chi0_{ef..}
    t = gchi0_q.contract_first_pair_with_local_vertex(f_dc_loc, min(f_dc_loc.niv_second, config.box.niv_core))
    # transversal attachment U_{acb2} = -U_magn;a2bc: [T @ U_magn]_{1da2}, then into the contraction layout
    kernel = (t @ u_loc.as_channel(SpinChannel.MAGN)).permute_orbitals("abcd->badc", copy=False)
    return kernel.scale(2.0 / config.sys.beta**2, copy=False)


def calculate_kernel_r_q(
    vrg_q_r: FourPoint, chi_phys_q_r: FourPoint, v_nonloc: Interaction, u_loc: LocalInteraction
) -> FourPoint:
    r"""
    Returns the kernel for the self-energy calculation minus the identity (which carries the constant
    :math:`-\mathcal{U}^{\mathbf{q}}_{r}` term of the three-leg Schwinger-Dyson bracket),

    .. math:: K_{1234} = \gamma^{\mathrm{q}\nu}_{r;1234} - \sum_{abcd} \gamma^{\mathrm{q}\nu}_{r;12ab}\,
        \mathcal{U}^{\mathbf{q}}_{r;badc}\, \chi^{\mathrm{q}}_{r;dc34} - \mathbb{1}_{1234},

    right-multiplied by the channel interaction and stored in the layout of the sigma contraction. In the
    multi-orbital Schwinger-Dyson equation the interaction attaches to the :math:`\chi^{*}`-side index pair of the
    three-leg vertex, while its amputated pair carries the external orbital index and the Green's-function leg
    (P. Worm's PhD thesis, Eq. (3.70)). The result :math:`M_{1ab2} = [K\,\mathcal{U}^{\mathbf{q}}_{r}]_{1ab2}`
    enters the self-energy as :math:`\Sigma_{12} \propto \sum_{ab} M_{1ab2} G_{ab}` and is therefore stored with
    the orbital permutation ``"abcd->badc"``, so that the existing contraction
    :math:`\Sigma_{12} \propto \sum_{ab} K'_{a12b} G_{ab}` of :func:`_run_column_sde` applies unchanged.

    :param vrg_q_r: The momentum-dependent three-leg vertex :math:`\gamma^{\mathrm{q}\nu}_{r}`.
    :param chi_phys_q_r: The (shell-corrected) physical susceptibility :math:`\chi^{\mathrm{phys};\mathrm{q}}_{r}`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param u_loc: The bare local interaction :math:`U`.
    :return: The self-energy kernel :math:`[(K-\mathbb{1})\,\mathcal{U}_r]` (contraction layout) as a
        :class:`FourPoint`.
    """
    u_r = v_nonloc.as_channel(vrg_q_r.channel) + u_loc.as_channel(vrg_q_r.channel)
    kernel = vrg_q_r - vrg_q_r @ u_r @ chi_phys_q_r

    # subtract the identity directly on the compound-identity positions (o1 == o4, o2 == o3, every frequency) instead
    # of materializing an identity block; the channel weights (1x dens, 3x magn) turn this into -U_d - 3U_m
    orb = np.arange(kernel.n_bands)
    kernel.mat[:, orb[:, None], orb[None, :], orb[None, :], orb[:, None], ...] -= 1.0

    return (kernel @ u_r).permute_orbitals("abcd->badc", copy=False)


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
    Calculates and saves the RPA susceptibility (for both density and magnetic channels) from the DMFT Green's
    functions, :math:`\chi^{\mathrm{q}}_{r;\mathrm{RPA}} = ((\chi^{\mathrm{q}}_{0})^{-1} + U_{r} +
    V^{\mathbf{q}}_{r})^{-1}` with the frequency-summed bubble :math:`\chi^{\mathrm{q}}_{0} = \frac{1}{\beta^2}
    \sum_{\nu} \chi^{\mathrm{q}\nu}_{0}`, the form the shell-corrected physical susceptibility takes (see
    :func:`create_generalized_chi_q_with_shell_correction`). The result is gathered to rank 0 and written to file.

    :param gchi0_q_full_sum: The frequency-summed bare bubble over the full box.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :return: None.
    """
    gchi0_q_sum_inv = gchi0_q_full_sum.invert()
    for channel in [SpinChannel.DENS, SpinChannel.MAGN]:
        u_r = u_loc.as_channel(channel) + v_nonloc.as_channel(channel)
        chi_rpa_q_r = (gchi0_q_sum_inv + u_r).invert(False)
        chi_rpa_q_r.mat = mpi_dist_irrk.gather(chi_rpa_q_r.mat)

        if mpi_dist_irrk.my_rank == 0:
            chi_rpa_q_r.save(name=f"chi_rpa_q_{channel.value}", output_dir=config.output.output_path)

        chi_rpa_q_r.free()
        config.logger.info(f"Calculated RPA susceptibility ({channel.value}).")
    gchi0_q_sum_inv.free()


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


def calculate_sigma_kernel_r_q(
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    mpi_dist_irrq: MpiDistributor,
    annealer: "LambdaAnnealer | None" = None,
    chunk_bytes: int | None = None,
    lambda_previous: dict | None = None,
) -> FourPoint:
    r"""
    Returns the kernel for the self-energy calculation in a specific spin channel. Calculates the auxiliary
    susceptibility, the three-leg vertex and the physical susceptibility with shell correction. Also performs a
    :math:`\lambda`-correction on the physical susceptibility if specified in the config (dispatched by the band
    count). Saves the physical susceptibility (and, if Eliashberg is enabled, the intermediate vertices) to file.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}`.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` (core box).
    :param gchi0_q_full_sum: The frequency-summed bare bubble over the full box.
    :param gchi0_q_core_sum: The frequency-summed bare bubble over the core box.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param mpi_dist_irrq: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param annealer: The active :class:`LambdaAnnealer` (its boson mass is applied to the physical susceptibility),
        or ``None`` when annealing is off.
    :param chunk_bytes: Chunk byte budget of the auxiliary-susceptibility build (``None`` uses the floor).
    :param lambda_previous: The per-iteration :math:`\lambda` of each corrected channel, kept by the
        self-consistency loop across its iterations; ``None`` for the one-shot correction.
    :return: The self-energy kernel for this channel as a :class:`FourPoint`.
    """
    logger = config.logger

    gchi_aux_q_r_sum = create_auxiliary_chi_r_q_sum(gamma_r, gchi0_q_inv, u_loc, chunk_bytes)

    mpi_dist_irrq.barrier()

    # reported as the full two-fermion auxiliary susceptibility, which the summed object stands in for
    logger.log_memory_usage(
        f"Auxiliary susceptibility ({gchi_aux_q_r_sum.channel.value})",
        gchi_aux_q_r_sum,
        mpi_dist_irrq.comm.size,
        scale=2 * config.box.niv_core,
    )
    logger.info(f"Non-Local auxiliary susceptibility ({gchi_aux_q_r_sum.channel.value}) calculated.")

    vrg_q_r = create_vrg_r_q(gchi_aux_q_r_sum, gchi0_q_inv)

    logger.info(f"Non-local three-leg vertex gamma^wv ({vrg_q_r.channel.value}) done.")
    logger.log_memory_usage(f"Three-leg vertex ({vrg_q_r.channel.value})", vrg_q_r, mpi_dist_irrq.comm.size)

    if config.eliashberg.perform_eliashberg:
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

    chi_phys_q_r.mat = mpi_dist_irrq.gather(chi_phys_q_r.mat)
    if mpi_dist_irrq.comm.rank == 0:
        chi_phys_q_r = _select_and_apply_lambda_correction(chi_phys_q_r, lambda_previous)
        chi_phys_q_r.save(name=f"chi_phys_q_{chi_phys_q_r.channel.value}", output_dir=config.output.output_path)

        # perform Ornstein-Zernike fit
        if chi_phys_q_r.channel == SpinChannel.MAGN:
            perform_ornstein_zernike_fit(chi_phys_q_r)

    chi_phys_q_r.mat = mpi_dist_irrq.scatter(chi_phys_q_r.mat)
    logger.info(f"Saved physical susceptibility ({chi_phys_q_r.channel.value}) to file.")

    _monitor_chi_phys(chi_phys_q_r, mpi_dist_irrq)

    if config.eliashberg.perform_eliashberg:
        chi_phys_q_r.save(
            name=f"chi_phys_q_{chi_phys_q_r.channel.value}_rank_{mpi_dist_irrq.comm.rank}",
            output_dir=config.output.eliashberg_path,
        )

    return calculate_kernel_r_q(vrg_q_r, chi_phys_q_r, v_nonloc, u_loc)


def calculate_sigma_from_kernel(kernel: FourPoint, giwk: GreensFunction, my_full_q_list: np.ndarray) -> SelfEnergy:
    r"""
    Returns :math:`\Sigma^{\mathrm{k}}_{12} = -\frac{1}{2\beta n_{\mathbf{q}}} \sum_{\mathrm{q}} \sum_{ab}
    K^{\mathrm{q}\nu}_{a12b} G^{\mathrm{k}-\mathrm{q}}_{ab}`. For very large momentum grids,
    this function is the slowest part of the code because of its repeated loops. Batching the
    q-points or using numba could speed it up further.

    Currently unused: the pipeline always runs the column-distributed real-space contraction (see
    :func:`_run_column_sde`), since this q-loop variant restores the full bosonic range on the kernel and therefore
    peaks *higher* in memory. Kept (with its buffered sibling
    :func:`calculate_sigma_from_kernel_loop`) for reference; the two variants' mutual parity is unit-tested.

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
        shifted_mat = np.roll(giwk.mat, tuple(q), axis=(0, 1, 2))  # roll by +q -> G^{k-q}
        for idx_w, wn_i in enumerate(wn):
            g_qk = shifted_mat[..., giwk.niv - wn_i : giwk.niv + config.box.niv_core - wn_i]
            k_slice = kernel[idx_q, ..., idx_w, config.box.niv_core :]
            mat += np.einsum("aijdv,xyzadv->xyzijv", k_slice, g_qk, optimize=path)

    mat *= -0.5 / config.sys.beta / config.lattice.k_grid.nk_tot
    return SelfEnergy(mat, config.lattice.nk, False, beta=config.sys.beta).compress_q_dimension().to_full_niv_range()


def calculate_sigma_from_kernel_loop(
    kernel: FourPoint,
    giwk: GreensFunction,
    my_full_q_list: np.ndarray,
) -> SelfEnergy:
    r"""
    Returns :math:`\Sigma^{\mathrm{k}}_{12} = -\frac{1}{2\beta n_{\mathbf{q}}} \sum_{\mathrm{q}} \sum_{ab}
    K^{\mathrm{q}\nu}_{a12b} G^{\mathrm{k}-\mathrm{q}}_{ab}`. For very large momentum grids,
    this function is the slowest part of the code because of its repeated loops. This is the buffered q-loop
    implementation (Fortran-ordered inputs, preallocated accumulator, precomputed shift indices). Currently unused,
    see :func:`calculate_sigma_from_kernel`.

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
    kx_indices = [((kxs - q[0]) % nkx) for q in my_full_q_list]
    ky_indices = [((kys - q[1]) % nky) for q in my_full_q_list]
    kz_indices = [((kzs - q[2]) % nkz) for q in my_full_q_list]

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


def _build_rspace_giwk_window(giwk: GreensFunction, node_comm) -> tuple[np.ndarray, MPI.Win | None]:
    r"""
    Builds the real-space Green's function :math:`F[G](R)` of the self-energy contraction once per node, in one
    shared-memory window in frequency-first layout ``[2 niv, n_R, o1, o2]``, so every frequency slab the
    contraction reads is contiguous. The node root transforms :math:`G(\mathbf{k}) \to F[G](R)` (see
    :meth:`GreensFunction.fft`) into the window, every other rank maps it read-only; a single-rank node holds a
    private array.

    :param giwk: The momentum-dependent :class:`GreensFunction` (cut to the self-energy box).
    :param node_comm: The node-local communicator.
    :return: The tuple ``(g_r, win)``; release the window once no rank reads it anymore (``win`` is ``None`` on a
        single-rank node).
    """
    nb, n_r = giwk.n_bands, config.lattice.k_grid.nk_tot
    g_r, win = mpi_utils.allocate_node_shared_array(node_comm, (2 * giwk.niv, n_r, nb, nb), giwk.mat.dtype)
    if node_comm.Get_rank() == 0:
        g_r[...] = np.moveaxis(giwk.fft().mat.reshape(n_r, nb, nb, -1), -1, 0)
    node_comm.Barrier()
    return g_r, win


def _run_column_sde(
    kernel_irr: FourPoint,
    mpi_dist_irrk: MpiDistributor,
    g_r: np.ndarray,
    giwk_niv: int,
    chunk_bytes: int | None = None,
) -> SelfEnergy | None:
    r"""
    Contracts the self-energy kernel with the Green's function,

    .. math:: \Sigma^{\mathrm{k}}_{12} = -\frac{1}{2\beta n_{\mathbf{q}}} \sum_{\mathrm{q}} \sum_{ab}
        K^{\mathrm{q}\nu}_{a12b} G^{\mathrm{k}-\mathrm{q}}_{ab},

    for :math:`\nu \geq 0` as a real-space product (convolution theorem), distributing the kernel's frequency columns
    over the ranks instead of its momenta. A column is one bosonic frequency :math:`\omega` of one fermionic frequency
    :math:`\nu`; a negative :math:`\omega` is read from the stored positive one by time reversal,
    :math:`K^{-\omega,\nu} = (K^{\omega,-\nu})^*`. The tasks - runs of
    :data:`~dgamore.memory_estimator.SDE_W_BLOCK` consecutive columns of one :math:`\nu` - are laid out by
    :func:`~dgamore.memory_estimator.column_sde_schedule`, and every rank:

    1. receives the irreducible-BZ rows of its tasks' columns in rounds of at most ``chunk_bytes``
       (:func:`~dgamore.mpi_utils.transpose_columns`),
    2. expands each column to the full BZ with :meth:`FourPoint.map_to_full_bz` (a time-reversed column with the
       conjugate orbital rotation), Fourier transforms it over the momentum axes in the order z, y, x and contracts
       it with the frequency slab of ``g_r`` at :math:`\nu - \omega` in bounded real-space row chunks,
    3. folds its tasks' partial sums of each :math:`\nu` in block order; the owner of :math:`\nu` (the rank holding
       its first block) folds the other ranks' sums in rank order, and rank 0 gathers the result.

    The bosonic sum is associated in the fixed blocks and in the fixed rank order of the schedule, so the result
    depends on the box and the rank count only, never on ``chunk_bytes``. Beyond the read-only kernel and window a
    rank holds one round of irreducible columns, under two full-BZ columns and a few real-space slabs.

    :param kernel_irr: The rank-local irreducible-BZ kernel ``[q, o1, o2, o3, o4, w, v]`` (half bosonic, full
        fermionic range, compressed momentum axis); only read.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param g_r: The frequency-first real-space Green's function from :func:`_build_rspace_giwk_window`.
    :param giwk_niv: The central fermionic-frequency index of ``g_r`` (``giwk.niv``).
    :param chunk_bytes: Byte budget of one round of transposed irreducible columns (``None`` uses the floor).
    :return: The real-space self-energy :class:`SelfEnergy` for :math:`\nu \geq 0` (compressed momentum axis, moments
        not fitted) on rank 0; ``None`` elsewhere. Transform it back with :meth:`SelfEnergy.ifft` and complete it
        with :meth:`SelfEnergy.to_full_niv_range`.
    """
    comm, rank, size = mpi_dist_irrk.comm, mpi_dist_irrk.my_rank, mpi_dist_irrk.mpi_size
    k_grid = config.lattice.k_grid
    n_r, nb = k_grid.nk_tot, config.sys.n_bands
    niw, niv = config.box.niw_core, config.box.niv_core
    dtype = kernel_irr.mat.dtype
    scale = -0.5 / config.sys.beta / n_r

    blocks, bounds, owner = memory_estimator.column_sde_schedule(niw, niv, size)
    n_b = len(blocks)

    def columns(tasks: range) -> np.ndarray:
        """The (bosonic, fermionic) kernel index rows ``[2, n_cols]`` of consecutive tasks, in task and block order."""
        pairs = []
        for t in tasks:
            v, (negative, ws) = t // n_b, blocks[t % n_b]
            pairs += [(w, niv - 1 - v if negative else niv + v) for w in ws]
        return np.array(pairs, dtype=int).reshape(-1, 2).T

    task_bytes = k_grid.nk_irr * nb**4 * memory_estimator.SDE_W_BLOCK * np.dtype(dtype).itemsize
    per_round = max(1, int((SLICE_CHUNK_BYTES if chunk_bytes is None else chunk_bytes) // task_bytes))
    n_rounds = -(-int(np.max(np.diff(bounds))) // per_round)
    rows = max(1, memory_estimator.SDE_CONTRACTION_CHUNK_BYTES // (nb**4 * np.dtype(dtype).itemsize))
    path = ["einsum_path", (0, 1)]
    transpose_tag, partial_tag = 1200, 1300  # base MPI tags of the column transpose and the partial sums

    my_v = np.flatnonzero(owner == rank)
    result = np.empty((my_v.size, n_r, nb, nb), dtype=dtype)
    sums = {}
    partial = np.empty((n_r, nb, nb), dtype=dtype)
    acc = np.empty_like(partial)
    report_at = {-(-n_rounds // 2), n_rounds}
    # deferred_collection batches the gc pass of the per-column object releases into one collection at the end
    with deferred_collection():
        for k in range(n_rounds):
            round_of = [
                range(bounds[r] + k * per_round, min(bounds[r + 1], bounds[r] + (k + 1) * per_round))
                for r in range(size)
            ]
            block = mpi_utils.transpose_columns(
                kernel_irr.mat, mpi_dist_irrk, [columns(tasks) for tasks in round_of], transpose_tag
            )
            c = 0
            for t in round_of[rank]:
                v, (negative, ws) = t // n_b, blocks[t % n_b]
                partial[...] = 0
                for w in ws:
                    col = FourPoint(block[..., c : c + 1], SpinChannel.NONE, config.lattice.nk, 1, 0, False, True, True)
                    c += 1
                    if negative:
                        col = col.to_negative_niw_range()
                    col = col.map_to_full_bz(k_grid, conjugate=negative).fft(copy=False, axes=(2, 1, 0))
                    g_slab = g_r[giwk_niv + (w if negative else -w) + v]
                    k_mat = col.mat[..., 0]
                    for r0 in range(0, n_r, rows):
                        np.einsum(
                            "Rad,Raijd->Rij",
                            g_slab[r0 : r0 + rows],
                            k_mat[r0 : r0 + rows],
                            out=acc[r0 : r0 + rows],
                            optimize=path,
                        )
                    np.add(partial, acc, out=partial)
                    col.free()
                partial *= scale
                if v not in sums:
                    sums[v] = result[v - my_v[0]] if owner[v] == rank else np.empty_like(partial)
                    np.copyto(sums[v], partial)
                else:
                    sums[v] += partial
            del block
            if k + 1 in report_at:
                config.logger.info(f"Self-energy contraction: {k + 1} of {n_rounds} column rounds done.")

    # every rank's run starts inside at most one foreign frequency; its owner folds the sums in rank order
    def ranks_of(v: int) -> list[int]:
        """The ranks holding blocks of frequency ``v`` (a nonempty overlap of task runs), in rank order."""
        return [r for r in range(size) if max(bounds[r], v * n_b) < min(bounds[r + 1], (v + 1) * n_b)]

    sends = [(int(owner[v]), sums[v]) for v in sums if owner[v] != rank]
    sources = [r for v in my_v for r in ranks_of(v) if r != rank]
    received = iter(mpi_utils.exchange_blocks(comm, sends, sources, (n_r, nb, nb), dtype, partial_tag))
    for i, v in enumerate(my_v):
        for r in ranks_of(v):
            if r != rank:
                result[i] += next(received)
    del sends, sums

    full = MpiDistributor(ntasks=niv, comm=comm, sizes=np.bincount(owner, minlength=size)).gather(result)
    if rank != 0:
        return None
    return SelfEnergy(
        np.ascontiguousarray(np.moveaxis(full, 0, -1)),
        config.lattice.nk,
        full_niv_range=False,
        has_compressed_q_dimension=True,
        calc_smom=False,
        beta=config.sys.beta,
    )


def _sde_chunk_budget(comm: MPI.Comm, shared_node_comm) -> int:
    r"""
    Returns the chunk byte budget of the self-energy section, identical on every rank: each rank derives its node's
    budget (see :func:`dgamore.memory_estimator.dynamic_chunk_budget`) and the job-wide minimum is taken, so the
    node with the least memory per rank bounds the chunking everywhere. The round count of :func:`_run_column_sde`
    follows from this budget and its column transpose is collective, so ranks with different budgets would walk
    different round counts and deadlock.

    :param comm: The MPI communicator.
    :param shared_node_comm: The node-local communicator (``None`` counts as one rank per node).
    :return: The chunk budget in bytes.
    """
    node_ranks = shared_node_comm.size if shared_node_comm is not None else 1
    chunk_bytes = memory_estimator.dynamic_chunk_budget(mpi_utils.job_memory_total(), node_ranks)
    return comm.allreduce(chunk_bytes, op=MPI.MIN) if comm.size > 1 else chunk_bytes


def get_starting_sigma(default_sigma: SelfEnergy) -> tuple[SelfEnergy, int]:
    r"""
    Retrieves the starting self-energy of the self-consistency cycle from a previous run. With
    ``use_interpolated_sigma`` this is the predecessor's final self-energy re-gridded to this temperature,
    ``sigma_dga_interpolated_beta<b>_niv<n>.npy`` (the file with ``<b>`` closest to the current :math:`\beta` if
    several exist); otherwise it is the raw iterate ``sigma_dga_iteration_<i>.npy`` with the highest ``<i>``. The
    iteration count comes from the raw iterates in both cases, read from the run's ``Sigma_Iterates`` subfolder or,
    for a run written before that subfolder existed, from the run folder itself. Without a usable file the DMFT
    self-energy is returned.

    :param default_sigma: The fallback (DMFT) :class:`SelfEnergy` used when no previous result is found.
    :return: A tuple of the starting :class:`SelfEnergy` (cut to the core box and interpolated onto the k-grid) and
        the iteration number it was taken from (0 if none found).
    """
    previous_sc_path = config.self_consistency.previous_sc_path

    if previous_sc_path is None or previous_sc_path == "" or not os.path.exists(previous_sc_path):
        if config.self_consistency.use_interpolated_sigma:
            config.logger.warning(
                "use_interpolated_sigma is set but previous_sc_path is empty or missing; starting from DMFT."
            )
        return default_sigma, 0

    pattern = "sigma_dga_iteration_*.npy"
    iterates = glob.glob(
        os.path.join(previous_sc_path, config.self_consistency.sigma_iterates_subfolder_name, pattern)
    ) or glob.glob(os.path.join(previous_sc_path, pattern))
    files = {int(match.group(1)): f for f in iterates if (match := re.search(r"iteration_(\d+)\.npy$", f))}
    if not files:
        return default_sigma, 0
    max_iter = max(files)
    max_file = files[max_iter]

    if config.self_consistency.use_interpolated_sigma:
        beta_regex = re.compile(r"sigma_dga_interpolated_beta([0-9.eE+-]+)_niv\d+\.npy$")
        candidates = glob.glob(os.path.join(previous_sc_path, "sigma_dga_interpolated_beta*_niv*.npy"))
        betas = [
            (abs(float(match.group(1)) - config.sys.beta), f) for f in candidates if (match := beta_regex.search(f))
        ]
        if not betas:
            config.logger.warning(f"No interpolated self-energy found in {previous_sc_path}; starting from DMFT.")
            return default_sigma, 0
        max_file = min(betas, key=lambda x: x[0])[1]
    config.logger.info(f"Starting the self-consistency from {max_file} (iteration {max_iter}).")
    mat = np.load(max_file)
    sigma = SelfEnergy(mat, mat.shape[:3], True, False, beta=config.sys.beta).cut_niv(config.box.niv_core)
    return sigma.interpolate_q_grid(config.lattice.k_grid.nk, False), max_iter


def _init_mu_history(starting_iter: int) -> list[float]:
    r"""
    Seeds the chemical-potential history for the self-consistency loop. For a fresh run (``starting_iter == 0``) the
    history starts at the current (DMFT) chemical potential :math:`\mu`. When resuming from a previous self-consistency
    calculation it is seeded with that run's last :math:`\mu` (from ``mu_history.npy``) and the global ``config.sys.mu``
    is synced to it: otherwise ``config.sys.mu`` would stay at the DMFT value while ``giwk_full`` is built with the
    previous run's :math:`\mu`, and any quantity computed from the global (e.g. the lattice filling in
    :meth:`GreensFunction.get_fill_nonlocal`, which now reads ``self._mu``) would use an inconsistent chemical potential.
    That value is only the starting guess: :func:`calculate_self_energy_q` re-solves it so the starting self-energy has
    the DMFT lattice filling.

    :param starting_iter: The iteration the previous calculation stopped at (0 for a fresh run).
    :return: The single-element chemical-potential history list.
    """
    if starting_iter == 0:
        return [config.sys.mu]

    previous_mu = float(np.load(os.path.join(config.self_consistency.previous_sc_path, "mu_history.npy"))[-1])
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


def _load_node_shared_local_vertex(
    node_comm, path: str, channel: SpinChannel, transform=None, axes: tuple[int, ...] | None = None
) -> tuple:
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
    :param axes: Optional axis order in which the array is stored (window or private copy); the returned vertex
        exposes the unchanged logical layout as a strided view of that storage, so a consumer whose contraction
        reads the stored order (e.g. :meth:`FourPoint.contract_first_pair_with_local_vertex`) works without a
        copy.
    :return: The tuple ``(vertex, win)``; free the window via :func:`_free_shared_window` once every rank is done
        reading (``win`` is ``None`` on the private path, then ``vertex.free()`` applies as before).
    """

    def _load() -> np.ndarray:
        obj = LocalFourPoint.load(path, channel)
        if transform is not None:
            obj = transform(obj)
        # ascontiguousarray since a pure orbital-permute transform returns a strided view of the loaded array
        return np.ascontiguousarray(obj.mat if axes is None else obj.mat.transpose(axes))

    if node_comm is None:
        mat, win = _load(), None
    else:
        mat, win = mpi_utils.build_node_shared_array(node_comm, _load)
    if axes is not None:
        mat = mat.transpose(np.argsort(axes))
    return LocalFourPoint(mat, channel, 1, 2, False, True), win


def _build_giwk_full(comm: MPI.Comm, sigma: SelfEnergy, mu: float, ek: np.ndarray, beta: float) -> tuple:
    r"""
    Builds the full-grid Green's function :math:`G^{\mathrm{k}}_{12}`, optionally deduplicated across the MPI ranks that
    share a physical node. The Dyson inversion
    runs only on each node's root rank and the result is placed in one MPI shared-memory window per node, so
    ``giwk_full`` occupies a single physical buffer per node instead of one private copy per rank (see
    :func:`dgamore.mpi_utils.build_node_shared_array`). Otherwise every rank builds its own copy. The node topology is
    discovered at runtime via ``comm.Split_type(MPI.COMM_TYPE_SHARED)`` (nothing about the cluster is hard-coded).
    The result keeps the dispersion but no reference to :math:`\Sigma`, so none of its copies (the frequency cut, the
    real-space transforms) duplicates the full-BZ self-energy.

    :param comm: The MPI communicator.
    :param sigma: The self-energy :math:`\Sigma` entering the Dyson equation.
    :param mu: Chemical potential :math:`\mu`.
    :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})`.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The tuple ``(giwk_full, win, node_comm)``; ``win`` is ``None`` on a single-rank node
        and must otherwise be released with :func:`_release_shared_giwk` once ``giwk_full`` has been cut to its private
        core box (the shared buffer is read-only and must not be freed while any rank still reads it).
    """
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    giwk_mat, win = mpi_utils.build_node_shared_array(
        node_comm, lambda: GreensFunction.get_g_full(sigma, mu, ek, beta).mat
    )
    giwk_full = GreensFunction(
        giwk_mat, None, ek, sigma.full_niv_range, False, False, nk=ek.shape[:3], beta=beta, mu=mu
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
    :param node_comm: The node-local communicator (or ``None`` when never created).
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


def _share_sigma_per_node(
    sigma: SelfEnergy | None, comm: MPI.Comm, node_comm, roots_comm
) -> tuple[SelfEnergy, MPI.Win | None]:
    r"""
    Hands rank 0's full-BZ self-energy to every rank as a view of one per-node MPI shared-memory window: rank 0
    compresses the momenta and broadcasts the object without its array (a small pickled blob), the node roots receive
    the array through a chunked broadcast over ``roots_comm`` and expose it to their node's other ranks (see
    :func:`dgamore.mpi_utils.build_node_shared_array`). The self-consistency loop thereby holds the full-BZ
    self-energy **once per node instead of once per rank**, and only the node roots ever receive it. Every rank gets
    its own :class:`SelfEnergy` object; only the buffer is shared and must be treated as read-only (every consumer
    copies via ``cut_niv``/``copy``/``concatenate`` before writing).

    :param sigma: The :class:`SelfEnergy` to share; only read on rank 0.
    :param comm: The MPI communicator.
    :param node_comm: The node-local (shared-memory) communicator.
    :param roots_comm: Communicator over exactly the node-root ranks (global rank 0 first).
    :return: The tuple ``(sigma, win)`` of this rank's :class:`SelfEnergy` viewing its node's buffer and the MPI
        shared-memory window (``None`` for a single-rank node); free the window via :func:`_free_shared_window` once
        no rank reads the buffer anymore.
    """
    mat = None
    if comm.rank == 0:
        sigma = sigma.compress_q_dimension()
        mat, sigma.mat = sigma.mat, None
    sigma = pickle.loads(comm.bcast(pickle.dumps(sigma) if comm.rank == 0 else None, root=0))
    if node_comm.rank == 0:
        mat = mpi_utils.bcast_rows(roots_comm, mat, root=0)
    shared, win = mpi_utils.build_node_shared_array(node_comm, lambda: mat)
    sigma.mat = shared
    return sigma, win


def _occupation_self_energy(
    sigma_new: SelfEnergy, sigma_dmft_full: SelfEnergy, mpi_dist_fullbz: MpiDistributor
) -> SelfEnergy:
    r"""
    Returns this rank's momentum slice of the self-energy the occupation is evaluated with: ``sigma_new`` on its own
    frequency box and the DMFT self-energy beyond it, shifted by the momentum-dependent shell offset ``sigma_new``
    carries (see :meth:`SelfEnergy.shell_offset_from`), with the high-frequency moments of the whole concatenation
    fitted on rank 0 and broadcast.

    :param sigma_new: The mixed :class:`SelfEnergy` (full BZ, identical on every rank; compressed in place).
    :param sigma_dmft_full: The DMFT :class:`SelfEnergy` supplying the shell frequencies (momentum-local).
    :param mpi_dist_fullbz: MPI distributor over the full BZ q-points.
    :return: The concatenated :class:`SelfEnergy` of this rank's momenta (compressed momentum axis, moments set).
    """
    sigma_slice = SelfEnergy(
        sigma_new.compress_q_dimension().mat[mpi_dist_fullbz.my_slice],
        (mpi_dist_fullbz.my_size, 1, 1),
        has_compressed_q_dimension=True,
        calc_smom=False,
        beta=config.sys.beta,
    )
    shell_offset = sigma_new.shell_offset_from(sigma_dmft_full)
    sigma_occ = sigma_slice.concatenate_self_energies(
        sigma_dmft_full, shell_offset=shell_offset[mpi_dist_fullbz.my_slice]
    )

    # the full-box moments come from the k-mean fit window of the replicated sigma_new, so rank 0 fits them once
    # (the fit window is a full-BZ array) and broadcasts; the fit equals the momentum-resolved concatenation's fit
    moments = (
        sigma_new.fit_smom_concatenated(sigma_dmft_full, shell_offset=shell_offset)
        if mpi_dist_fullbz.my_rank == 0
        else None
    )
    sigma_occ._smom0, sigma_occ._smom1 = mpi_dist_fullbz.bcast(moments)
    return sigma_occ


def _update_occ_and_energies_distributed(
    sigma_new: SelfEnergy, sigma_dmft_full: SelfEnergy, mpi_dist_fullbz: MpiDistributor, mu: float
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    r"""
    Computes the occupation and the kinetic and potential energies of the mixed self-energy on the DMFT frequency
    box, distributed over the full-BZ momenta: every rank concatenates and Dyson-inverts only its own momentum
    slice, evaluates the occupation and energy sums there, and the results are recombined (the former evaluation
    built the whole DMFT-box Green's function and its asymptotic tail sums on rank 0 while every other rank idled).
    The self-energy moments are fitted from the momentum-averaged concatenated self-energy, allreduced first so
    they match the full-box fit on every rank; the k-resolved occupation is allgathered and the k-summed scalars
    are recombined with each rank's momentum count as weight. The momentum-dependent constant the mixed
    self-energy carries in its shell (see :meth:`SelfEnergy.shell_offset_from`) is carried into the DMFT-box
    extension and its moment fit.

    :param sigma_new: The mixed :class:`SelfEnergy` (full BZ, compressed momenta, identical on every rank).
    :param sigma_dmft_full: The DMFT :class:`SelfEnergy` supplying the shell frequencies (momentum-local).
    :param mpi_dist_fullbz: MPI distributor over the full BZ q-points.
    :param mu: Chemical potential :math:`\mu`.
    :return: The tuple ``(n, occ, occ_k, ekin, epot)``: total filling, k-averaged occupation ``[o1, o2]``,
        k-resolved occupation ``[kx, ky, kz, o1, o2]``, and the kinetic and potential energies per site.
    """
    nk_tot = config.lattice.k_grid.nk_tot
    n_bands = config.sys.n_bands
    n_my = mpi_dist_fullbz.my_size

    sigma_occ = _occupation_self_energy(sigma_new, sigma_dmft_full, mpi_dist_fullbz)

    ek = config.lattice.hamiltonian.get_ek()
    ek_slice = ek.reshape(nk_tot, n_bands, n_bands)[mpi_dist_fullbz.my_slice].reshape(n_my, 1, 1, n_bands, n_bands)
    giwk_occ = GreensFunction.get_g_full(sigma_occ, mu, ek_slice, config.sys.beta)
    _, _, occ_k_slice = giwk_occ.get_fill_nonlocal()
    ekin, epot = giwk_occ.get_ekin(), giwk_occ.get_epot()
    giwk_occ.free()
    sigma_occ.free()

    n_el, occ, occ_k = _assemble_occupation(occ_k_slice, mpi_dist_fullbz)
    scalars = np.array([ekin, epot]) * n_my
    if mpi_dist_fullbz.mpi_size > 1:
        scalars = mpi_dist_fullbz.comm.allreduce(scalars)
    ekin, epot = scalars / nk_tot
    return n_el, occ, occ_k, float(ekin), float(epot)


def _assemble_occupation(
    occ_k_slice: np.ndarray, mpi_dist_fullbz: MpiDistributor
) -> tuple[float, np.ndarray, np.ndarray]:
    r"""
    Assembles the k-resolved occupation from the momentum slices the ranks evaluated and derives the k-averaged
    occupation and the filling from the assembled array, so both equal a full-grid evaluation bit for bit. The
    assembly is the chunked Allreduce of zero-padded slices: each momentum is contributed by exactly one rank, so
    the sum is exact and no point-to-point matching is involved. The dtype is pinned to complex128 because the
    fill's dtype is value-dependent (a real-matrix eigendecomposition returns float when a slice's eigenvalues are
    all real) and ranks entering a collective with different element types abort with truncated messages.

    :param occ_k_slice: This rank's k-resolved occupation ``[kx, ky, kz, o1, o2]`` on its momentum slice.
    :param mpi_dist_fullbz: MPI distributor over the full BZ q-points.
    :return: A tuple of the total filling :math:`n`, the k-averaged occupation ``[o1, o2]`` and the k-resolved
        occupation ``[kx, ky, kz, o1, o2]`` on the full grid.
    """
    nk_tot, n_bands, n_my = config.lattice.k_grid.nk_tot, occ_k_slice.shape[-1], mpi_dist_fullbz.my_size
    occ_k = np.zeros((nk_tot, n_bands, n_bands), dtype=np.complex128)
    occ_k[mpi_dist_fullbz.my_slice] = occ_k_slice.reshape(n_my, n_bands, n_bands)
    if mpi_dist_fullbz.mpi_size > 1:
        occ_k = mpi_dist_fullbz.allreduce(occ_k)
    occ_k = occ_k.reshape(*config.lattice.k_grid.nk, n_bands, n_bands)
    occ = np.mean(occ_k, axis=(0, 1, 2))
    occ.real[np.abs(occ) < 1e-12] = 0.0
    return 2.0 * np.trace(occ).real, occ, occ_k


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
    lambda_previous: dict | None = None,
) -> SelfEnergy | None:
    r"""
    Returns the raw (un-mixed) DGA self-energy proposal :math:`S(\Sigma_{\mathrm{in}})` at chemical potential
    :math:`\mu`: Hartree/Fock, the Dyson Green's function, the bubble, the double-counting, density and magnetic
    kernels, and the FFT Schwinger-Dyson contraction, finished with the noise-removal term and the DMFT tail. The
    tail carries the momentum-dependent part of the Hartree-Fock term, i.e. the contribution of
    :math:`V^{\mathbf{q}}`, which the impurity self-energy does not contain.

    Single source of truth for the proposal map: it is called once per self-consistency iteration by
    :func:`calculate_self_energy_q`. The local irreducible vertex is frozen, so every
    evaluation rebuilds the bubble, the ladder susceptibilities and the SDE self-energy. The Hartree/Fock term reads
    ``config.sys.occ`` / ``occ_k``, which the caller sets consistently with :math:`\Sigma_{\mathrm{in}}`.

    :param sigma_in: The input self-energy (full-BZ or local first-iteration, DMFT tail attached).
    :param mu: Chemical potential :math:`\mu`.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`, reduced to this rank's irreducible q-points.
    :param v_nonloc_full: The non-local interaction on the full q-grid (for the Hartree/Fock term); only read on
        rank 0.
    :param sigma_dmft: The DMFT self-energy (cut to the loop's niv), providing the high-frequency tail.
    :param delta_sigma: The DMFT-minus-local noise-removal term on the core box.
    :param my_irr_q_list: This rank's irreducible q-point list.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param comm: The MPI communicator.
    :param current_iter: The current iteration number (the RPA susceptibility is saved only on iteration 1).
    :param annealer: The active :class:`LambdaAnnealer` threaded into the kernel step, or ``None`` when annealing
        is off.
    :param chunk_budgets: Chunk byte budgets of the auxiliary-susceptibility build and the self-energy
        contraction (sized by the driver from the memory estimate); ``None`` gives both the job-wide fair-share
        budget of :func:`_sde_chunk_budget`.
    :param lambda_previous: The per-iteration :math:`\lambda` of each corrected channel, kept by the
        self-consistency loop across its iterations; ``None`` for the one-shot correction.
    :return: The raw full-BZ proposal :class:`SelfEnergy` (DMFT tail attached) on rank 0; ``None`` on every other rank,
        since only rank 0 mixes it.
    """
    logger = config.logger

    giwk_full, giwk_win, shared_node_comm = _build_giwk_full(
        comm, sigma_in, mu, config.lattice.hamiltonian.get_ek(), config.sys.beta
    )

    # giwk lives in one shared window per node, so the job holds one copy per node - not one per rank
    logger.log_memory_usage("giwk", giwk_full, mpi_utils.count_nodes(comm, shared_node_comm), per="node")

    gchi0_q = BubbleGenerator.create_generalized_chi0_q_fft(
        mpi_dist_irrk,
        giwk_full,
        config.box.niw_core,
        config.box.niv_full,
        config.lattice.k_grid,
        config.sys.beta,
        node_comm=shared_node_comm,
    )

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
        os.path.join(config.output.output_path, "f_dc_loc.npy"),
        SpinChannel.MAGN,
        axes=(4, 0, 1, 5, 2, 3, 6),  # the order contract_first_pair_with_local_vertex reads as a view
    )
    kernel = calculate_sigma_dc_kernel(f_dc_loc, gchi0_q, u_loc)
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

    if current_iter == 1:
        calculate_and_save_chi_q_r_rpa(gchi0_q_full_sum, u_loc, v_nonloc, mpi_dist_irrk)

    if config.eliashberg.perform_eliashberg:
        gchi0_q_core_inv.save(name=f"gchi0_q_inv_rank_{comm.rank}", output_dir=config.output.eliashberg_path)

    chunk_bytes = _sde_chunk_budget(comm, shared_node_comm) if chunk_budgets is None else chunk_budgets.sde
    aux_chunk_bytes = chunk_bytes if chunk_budgets is None else chunk_budgets.chiq_aux

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
            annealer,
            aux_chunk_bytes,
            lambda_previous,
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
            annealer,
            aux_chunk_bytes,
            lambda_previous,
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

    # Column-distributed real-space contraction (the q-loop variant peaks HIGHER, see calculate_sigma_from_kernel);
    # the k-space giwk window is released once the R-space window is built, only giwk's dispersion is read later.
    g_r, g_r_win = _build_rspace_giwk_window(giwk_full, shared_node_comm)
    giwk_niv = giwk_full.niv
    if giwk_win is not None:
        giwk_full.mat = None
    _free_shared_window(giwk_win, shared_node_comm)

    sigma_prop = _run_column_sde(kernel, mpi_dist_irrk, g_r, giwk_niv, chunk_bytes)
    kernel.free()
    g_r = None
    _release_shared_giwk(g_r_win, shared_node_comm)

    # only rank 0 mixes, so only rank 0 finishes the proposal; the others receive the mixed iterate once per node
    if comm.rank != 0:
        return None

    sigma_prop = sigma_prop.ifft().to_full_niv_range()
    logger.info("Self-energy calculated from kernel.")
    logger.log_memory_usage("Non-local sigma", sigma_prop)

    hartree, fock = get_hartree_fock(u_loc, v_nonloc_full)
    # the V^q part of Hartree-Fock is absent from the impurity self-energy padding the shell, so it is re-added there
    hartree_v, fock_v = get_hartree_fock(LocalInteraction(np.zeros_like(u_loc.mat)), v_nonloc_full)
    hf_v = (hartree_v + fock_v)[..., 0]
    logger.info("Calculated Hartree and Fock terms.")

    sigma_prop = sigma_prop + hartree + fock
    logger.info("Full non-local self-energy calculated.")

    # This is done to minimize noise. We remove some fluctuations from dmft that are included in the local self-energy
    # calculated in this code and add the smooth dmft self-energy
    sigma_prop += delta_sigma
    sigma_prop = sigma_prop.concatenate_self_energies(sigma_dmft, shell_offset=hf_v)
    return sigma_prop


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
        dist = MpiDistributor.create_distributor(ntasks=flagged.size, comm=comm, name="PoleFit")
        # every rank re-grids only the momenta it fits; reduce_q keeps the ascending irreducible order
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


def _jacobian_expander(state: dict) -> Callable[[np.ndarray], np.ndarray] | None:
    r"""
    Builds the mapping of one stored column of a predecessor's spectrum file to the real vector on this run's window:
    the column, a complex :math:`(k_x, k_y, k_z, n_b, n_b, 2\nu)` array on the predecessor's grid, is re-gridded on
    the frequency axis by :meth:`SelfEnergy.interpolate` (the hand-over's PCHIP, without the MiniPole path, which is
    for self-energies), re-sampled onto this run's momentum grid by
    :meth:`~dgamore.n_point_base.IAmNonLocal.interpolate_q_grid`, zeroed on every target frequency beyond the
    predecessor's highest source frequency (an eigenvector is not extrapolated) and flattened with
    :func:`~dgamore.jacobian_stabilization.to_vec`. Identical windows skip the re-gridding. Refused, with a warning,
    for another orbital count.

    The chain is an approximation for an eigenvector, not an exact mapping of one. PCHIP derivatives depend
    nonlinearly on the data they are built from, so the real and the imaginary part of a column are re-gridded
    independently and the relative weighting inside the column pair of a complex mode is not preserved exactly, and
    the same-sign-branch extrapolation below the innermost source frequency was designed for the causal shape of a
    self-energy rather than for a Ritz vector. The :class:`~dgamore.self_energy.SelfEnergy` the column travels in
    also fits high-frequency moments and estimates a core box on it, which is wasted work on a vector.

    A column that comes out on another window than this run's is dropped, empty, where the re-gridding produced
    it rather than at the first step that multiplies by it.

    :param state: The carried spectrum (``shape`` and ``beta`` are read).
    :return: The callable, or ``None`` when the carry-over is refused.
    """
    shape_src = tuple(int(n) for n in state["shape"])
    beta_src = float(state["beta"])
    nb = config.sys.n_bands
    if shape_src[3:5] != (nb, nb):
        config.logger.warning("The carried Jacobian spectrum belongs to another orbital count; nothing carried.")
        return None
    nk_tgt, niv_tgt, beta_tgt = tuple(int(n) for n in config.lattice.k_grid.nk), config.box.niv_core, config.sys.beta
    nk_src, niv_src = shape_src[:3], shape_src[-1] // 2
    n_real = 4 * int(np.prod(nk_tgt)) * nb**2 * niv_tgt
    if nk_src == nk_tgt and niv_src == niv_tgt and beta_src == beta_tgt:
        return lambda column: to_vec(np.asarray(column, dtype=np.complex128).reshape(shape_src))
    inside = np.abs(MFHelper.vn(niv_tgt, beta_tgt)) <= np.abs(MFHelper.vn(niv_src, beta_src)).max()

    def expand(column: np.ndarray) -> np.ndarray:
        window = np.asarray(column, dtype=np.complex128).reshape(shape_src)
        sigma = SelfEnergy(window, nk_src, True, False, calc_smom=False, beta=beta_src)
        sigma = sigma.interpolate(beta_tgt, niv_tgt).interpolate_q_grid(nk_tgt, False)
        vector = to_vec(sigma.mat * inside)
        if vector.size != n_real:
            config.logger.warning(
                f"A carried Jacobian column re-grids to {vector.size} entries instead of the {n_real} of this run's "
                "window; it is dropped."
            )
            return np.zeros(0)
        return vector

    return expand


def _carry_jacobian_spectrum(tracker: JacobianTracker, annealer: LambdaAnnealer | None) -> None:
    r"""
    Hands the predecessor's certified Jacobian spectrum to the tracker before the first iteration of a resumed run,
    read from ``jacobian.npz`` in ``previous_sc_path`` (see :meth:`JacobianTracker.save_spectrum`), together with
    this run's inverse temperature, so that a mode the file follows over two rungs can be flipped ahead of the
    crossing its extrapolation in :math:`\beta` predicts (see :meth:`JacobianTracker.carry_in`); the
    tracker keeps the re-gridded columns for the match :func:`_save_jacobian_spectrum` writes at the end. A
    predecessor that did not reach the pure fixed point is carried with a warning; a predecessor the expander
    refuses (see :func:`_jacobian_expander`) is not carried, and neither is a file that holds the per-iteration
    traces alone, which a run that never reached the end of its loop leaves behind. Flips are kept pending while a
    susceptibility-reshaping scaffold is on, as in :func:`_update_jacobian_tracker`.

    :param tracker: The rank-0 tracker of this run.
    :param annealer: The :class:`~dgamore.lambda_ops.LambdaAnnealer`, or ``None`` while annealing is off.
    :return: None.
    """
    folder = config.self_consistency.previous_sc_path
    path = os.path.join(folder, JACOBIAN_FILE)
    if not os.path.isfile(path):
        config.logger.info(f"No {JACOBIAN_FILE} in {folder}; nothing carried.")
        return
    state = load_spectrum(path)
    if "lam_pi" not in state:
        config.logger.info(f"{path} holds no certified spectrum (the run did not end its loop); nothing carried.")
        return
    if not bool(state["converged"]):
        config.logger.warning(
            f"The predecessor's Jacobian spectrum in {path} belongs to a run that did not reach the pure fixed point."
        )
    expand = _jacobian_expander(state)
    if expand is None:
        return
    allow_flip = not (
        config.stabilization.use_chi_phys_restriction
        or config.stabilization.use_lambda_correction
        or (annealer is not None and annealer.mass_present)
    )
    tracker.carry_in(state, expand, allow_flip=allow_flip, beta=config.sys.beta)


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
    Writes the tracker's certified spectrum, together with the per-iteration traces, to ``jacobian.npz`` in the
    output folder for a successor run (see :meth:`JacobianTracker.save_spectrum`), on the full momentum grid, every
    mode with the value it matched in the predecessor's carried columns. This final write replaces the trace-only
    file the loop rewrote every iteration (see :func:`_append_jacobian_eigenvalues`).

    :param tracker: The rank-0 tracker of this run.
    :param converged: Whether the run reached the pure fixed point (a scaffolded phase that ended on the last
        iteration does not count).
    :param rows_lam: The leading Ritz value rows collected over the run, oldest first.
    :param rows_res: The matching Ritz residual rows, oldest first.
    :param rows_damping: The matching effective dampings, oldest first.
    :return: None.
    """
    nb = config.sys.n_bands
    shape = (*(int(n) for n in config.lattice.k_grid.nk), nb, nb, 2 * config.box.niv_core)
    tracker.save_spectrum(
        os.path.join(config.output.output_path, JACOBIAN_FILE),
        shape,
        config.sys.beta,
        converged,
        DTYPE,
        traces=_jacobian_traces(rows_lam, rows_res, rows_damping),
    )


def _append_jacobian_eigenvalues(tracker: JacobianTracker, rows_lam: list, rows_res: list, rows_damping: list) -> None:
    """
    Appends this iteration's leading Jacobian Ritz values, residuals and damping to the running rows and rewrites
    ``jacobian.npz`` in the output folder with the three traces (``eigenvalues``, ``eigenvalue_residuals``,
    ``damping``); the certified spectrum joins the same file when the loop ends (see :func:`_save_jacobian_spectrum`).

    An iteration whose tracker update produced no estimate (see :attr:`JacobianTracker.estimated`) appends a row
    of ``nan`` to the first two, so the traces end up with exactly one row per iteration of this run, readable
    as the leading eigenvalue's approach to the instability and the damping the tracker ran at. The damping entry
    always carries a number, since the effective damping holds its last value through an iteration without an
    estimate.

    :param tracker: The rank-0 tracker owning the last update.
    :param rows_lam: The Ritz value rows collected so far, oldest first, extended in place.
    :param rows_res: The matching Ritz residual rows collected so far, oldest first, extended in place.
    :param rows_damping: The matching effective dampings collected so far, oldest first, extended in place.
    :return: None.
    """
    if tracker.estimated:
        lam_pi, res = tracker.leading(_JACOBIAN_ROW_WIDTH)
    else:
        lam_pi, res = (
            np.full(_JACOBIAN_ROW_WIDTH, np.nan, dtype=np.complex128),
            np.full(_JACOBIAN_ROW_WIDTH, np.nan, dtype=np.float64),
        )
    rows_lam.append(lam_pi)
    rows_res.append(res)
    rows_damping.append(tracker.p_eff)
    path = os.path.join(config.output.output_path, JACOBIAN_FILE)
    np.savez_compressed(path, **_jacobian_traces(rows_lam, rows_res, rows_damping))


def _mixing_history_cap(
    current_iter: int,
    release_iter: int | None,
    anneal_reset_iter: int | None = None,
    extra_event_iter: int | None = None,
) -> int | None:
    """
    Returns the history cap for this iteration: the number of iterations since the most recent map-switching event
    - the susceptibility-restriction release, a change of the lambda-annealing mass, or the extra event the caller
    tracks (the accelerated mixing passes the iteration the Jacobian tracker last changed the reflected map on; the
    tracker's own raw window passes none). Nothing may extrapolate across such a discontinuity, so the usable
    history is capped to the post-event iterations (``None`` when no event has occurred).

    :param current_iter: The current self-consistency iteration number.
    :param release_iter: The iteration the susceptibility restriction was released on (``None`` if never).
    :param anneal_reset_iter: The iteration the annealing mass last changed on (``None`` if never).
    :param extra_event_iter: The iteration of the caller's own map-switching event (``None`` if never).
    :return: The history cap, or ``None`` for no cap.
    """
    events = (release_iter, anneal_reset_iter, extra_event_iter)
    last_reset_iter = max((it for it in events if it is not None), default=None)
    return None if last_reset_iter is None else max(0, current_iter - last_reset_iter - 1)


def _update_jacobian_tracker(
    tracker: JacobianTracker,
    mixing_history: list,
    current_iter: int,
    release_iter: int | None,
    anneal_reset_iter: int | None,
    annealer: LambdaAnnealer | None,
) -> bool:
    r"""
    Runs one monitoring step of the Jacobian tracker on the raw :math:`(x, S(x))` pairs the mixing has just
    recorded.

    The tracker's window restarts on a scaffold event only - the susceptibility-restriction or lambda-correction
    release and a change of the annealing mass - because a pair recorded under a scaffolded map samples a different
    map; the pair recorded AT such an event was still computed with the scaffolded map and is dropped with it,
    leaving the pairs of the iterations after it including the current one. It deliberately does not restart on the
    tracker's own events: the raw pairs do not depend on which directions were reflected afterwards, so the whole
    window stays a sample of one and the same map. Flips are paused while any scaffold is active, which leaves the
    tracker measuring the spectrum and releasing a subspace installed from the physical map.

    :param tracker: The tracker owning the flipped subspace and the effective damping.
    :param mixing_history: The (iterate, raw proposal, used proposal) triples recorded by the mixing, oldest first.
    :param current_iter: The current self-consistency iteration number.
    :param release_iter: The iteration a susceptibility-reshaping scaffold was released on (``None`` if never).
    :param anneal_reset_iter: The iteration the annealing mass last changed on (``None`` if never).
    :param annealer: The :class:`~dgamore.lambda_ops.LambdaAnnealer`, or ``None`` while annealing is off.
    :return: Whether the tracked subspace changed, i.e. the reflected map the accelerated mixing sees switched.
    """
    raw_cap = _mixing_history_cap(current_iter, release_iter, anneal_reset_iter)
    n_entries = len(mixing_history) if raw_cap is None else min(len(mixing_history), raw_cap + 1)
    entries = mixing_history[len(mixing_history) - n_entries :]
    allow_flip = not (
        config.stabilization.use_chi_phys_restriction
        or config.stabilization.use_lambda_correction
        or (annealer is not None and annealer.mass_present)
    )
    iterates = [entry[0] for entry in entries]
    proposals = [entry[1] for entry in entries]
    return tracker.update(iterates, proposals, allow_flip=allow_flip)


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
    (see :class:`~dgamore.sigma_jacobian.ExactJacobian` and :func:`~dgamore.sigma_jacobian.leading_eigenpairs`) and
    makes them the spectrum rank 0's tracker writes to ``jacobian.npz`` (see
    :meth:`JacobianTracker.install_exact_spectrum`). Collective over ``comm``.

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
    :param chunk_budgets: Chunk byte budgets of the auxiliary-susceptibility build and the self-energy
        contraction (sized by the driver from the memory estimate); ``None`` gives both the job-wide fair-share
        budget of :func:`_sde_chunk_budget`.
    :return: None.
    """
    from dgamore.sigma_jacobian import ExactJacobian, leading_eigenpairs

    config.logger.info("Evaluating the leading eigenpairs of the exact Jacobian at the converged self-energy.")
    jac = ExactJacobian(
        sigma, mu, u_loc, v_nonloc, v_nonloc_full, sigma_dmft_full, mpi_dist_irrk, mpi_dist_fullbz, comm, chunk_budgets
    )
    try:
        spectrum = leading_eigenpairs(jac, comm)
    finally:
        jac.free()
    if comm.rank == 0:
        tracker.install_exact_spectrum(*spectrum)


def calculate_self_energy_q(
    comm: MPI.Comm,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    sigma_dmft: SelfEnergy,
    sigma_local: SelfEnergy,
    chunk_budgets: memory_estimator.ChunkBudgets | None = None,
) -> SelfEnergy | None:
    r"""
    Runs the non-local DGA self-energy calculation. Calculates the Hartree- and Fock terms, the bubble,
    the double-counting correction and the kernel in the density and magnetic channel. Finally, calculates the
    non-local self-energy from the kernel and the Green's function. Also takes care of the self-consistency loop and
    the chemical potential adjustment as well as the self-energy mixing, etc.

    With ``config.stabilization.use_jacobian_stabilization`` enabled, a
    :class:`~dgamore.jacobian_stabilization.JacobianTracker` measures the leading eigenvalues of the self-energy
    map from the raw (iterate, proposal) pairs the mixing records on rank 0, so that the mixing reflects the
    proposal residual on the certified unstable directions and a change of that reflected map restarts the
    accelerated-mixing history like the scaffold releases do. The tracker also measures the largest damping the
    certified spectrum allows, which the mixing applies to every damped Picard step, while an accelerated step
    keeps the configured parameter with or without a reflection in force. With
    ``config.stabilization.use_exact_jacobian`` a run that reaches the pure fixed point replaces the tracker's
    estimate by the leading eigenpairs of the exact Jacobian there before the spectrum is written (see
    :func:`_install_exact_jacobian`).

    :param comm: The MPI communicator.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param sigma_dmft: The DMFT self-energy (used as the starting point and for the shell/tail correction).
    :param sigma_local: The locally recomputed self-energy (used for smoothing out the DGA :class:`SelfEnergy`).
    :param chunk_budgets: Chunk byte budgets of the auxiliary-susceptibility build and the self-energy
        contraction (sized by the driver from the memory estimate); ``None`` gives both the job-wide fair-share
        budget of :func:`_sde_chunk_budget`.
    :return: The converged (or last-iteration) momentum-dependent DGA :class:`SelfEnergy` on rank 0; ``None`` on every
        other rank.
    """
    logger = config.logger

    logger.info("Starting with non-local DGA routine.")
    logger.info("Initializing MPI distributor.")

    # MPI distributor for the irreducible BZ
    mpi_dist_irrk = MpiDistributor.create_distributor(
        ntasks=config.lattice.k_grid.nk_irr, comm=comm, name="Q", output_path=config.output.output_path
    )
    irrk_q_list = config.lattice.k_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    mpi_dist_fullbz = MpiDistributor.create_distributor(
        ntasks=config.lattice.k_grid.nk_tot, comm=comm, name="FBZ", output_path=config.output.output_path
    )

    # the loop's full-BZ self-energy is held once per NODE (mixed on rank 0, broadcast to the node roots, then
    # window-shared read-only); a single-rank run keeps rank 0's own array
    sc_node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) if comm.size > 1 else None
    sc_roots_comm = comm.Split(0 if sc_node_comm.rank == 0 else 1) if sc_node_comm is not None else None
    sigma_win = None

    # only rank 0 reads a previous run's iterate; the other ranks receive it below, once per node
    sigma_old, starting_iter = get_starting_sigma(sigma_dmft) if comm.rank == 0 else (sigma_dmft, 0)
    starting_iter = comm.bcast(starting_iter, root=0)
    if starting_iter > 0:
        logger.info(
            f"Using previous calculation and starting the self-consistency loop at iteration {starting_iter + 1}."
        )

    mu_history = _init_mu_history(starting_iter)

    # rank 0 keeps the accelerated-mixing (iterate, proposal) pair history in memory; the saved sigma files hold
    # only post-mixing iterates (no proposals), so resumed runs start empty and re-arm as pairs accumulate
    mixing_history = None
    if comm.rank == 0 and config.self_consistency.mixing_strategy.lower() in ("pulay", "anderson"):
        mixing_history = []

    # the tracker is built on every rank (a few floats) but only rank 0, which owns the pair history, ever feeds it
    tracker = None
    rows_lam, rows_res, rows_damping = None, None, None
    if config.stabilization.use_jacobian_stabilization:
        tracker = JacobianTracker(config.self_consistency.mixing, logger=logger)
        rows_lam, rows_res, rows_damping = [], [], []
        if comm.rank == 0 and mixing_history is None:
            mixing_history = []
    stab_event_iter = None

    niv_cut = min(config.box.niw_core + config.box.niv_full + 10, config.box.niv_dmft)
    sigma_dmft_full = sigma_dmft.copy()

    ek = config.lattice.hamiltonian.get_ek()
    # the DMFT lattice Green's function on the whole DMFT box is read back only by the DMFT spectrum continuation and
    # by a warm start, which holds its filling
    if comm.rank == 0 and (config.ana_cont.do_spectrum_dmft or starting_iter > 0):
        giwk_full_dmft = GreensFunction.get_g_full(sigma_dmft_full, config.sys.mu_dmft, ek, config.sys.beta)
        if config.ana_cont.do_spectrum_dmft:
            giwk_full_dmft.save(output_dir=config.output.output_path, name="g_latt_dmft")
        if starting_iter > 0:
            n_dmft = giwk_full_dmft.get_fill_nonlocal()[0]
        giwk_full_dmft.free()

    if comm.rank == 0 or starting_iter == 0:
        sigma_old = sigma_old.concatenate_self_energies(sigma_dmft_full)

    if starting_iter == 0:
        # a fresh start holds the momentum-local DMFT sigma on every rank, so each rank evaluates the filling of the
        # full DMFT box on its own momentum slice and the slices are assembled exactly (zero-padded reduction)
        n_my, nb = mpi_dist_fullbz.my_size, ek.shape[-1]
        ek_my = ek.reshape(-1, nb, nb)[mpi_dist_fullbz.my_slice].reshape(n_my, 1, 1, nb, nb)
        occ_k_my = (
            GreensFunction.get_g_full(sigma_old, mu_history[-1], ek_my, config.sys.beta).get_fill_nonlocal()[2]
            if n_my
            else np.zeros((0, 1, 1, nb, nb))
        )
        config.sys.n, config.sys.occ, config.sys.occ_k = _assemble_occupation(occ_k_my, mpi_dist_fullbz)
    else:
        if comm.rank == 0:
            # the loop holds the filling of the DMFT lattice Green's function the local vertex belongs to; a warm
            # start's own filling (its self-energy at the predecessor's mu) drifts along a chain of rungs, so its mu
            # is re-solved for the DMFT filling on the loop's frequency box
            sigma_cut = sigma_old.cut_niv(niv_cut).compress_q_dimension()
            mu_previous = mu_history[-1]
            mu_history[-1] = update_mu(
                mu_previous, n_dmft, ek, sigma_cut.mat, config.sys.beta, sigma_cut.fit_smom()[0], logger=logger
            )
            logger.info(
                f"Warm start holds the DMFT lattice filling {n_dmft:.6f}: mu re-solved from {mu_previous} to "
                f"{mu_history[-1]}."
            )
            # only the occupations are needed here: sum the Dyson chunks directly instead of holding a second full-k
            # Green's function on the full DMFT frequency box (tens of GB at production scale)
            _, config.sys.occ, config.sys.occ_k = GreensFunction.get_fill_nonlocal_from_sigma(
                sigma_old, mu_history[-1], ek, config.sys.beta
            )
            config.sys.n = n_dmft
        config.sys.n, config.sys.occ, config.sys.occ_k, mu_history[-1] = comm.bcast(
            (config.sys.n, config.sys.occ, config.sys.occ_k, mu_history[-1]), root=0
        )
        config.sys.mu = mu_history[-1]

    sigma_old = sigma_old.cut_niv(niv_cut)
    sigma_dmft = sigma_dmft.cut_niv(niv_cut)

    # the starting iterate keeps the plain DMFT tail; the first proposal attaches the V^q Hartree-Fock offset
    if sigma_old.niv < niv_cut:
        sigma_old = sigma_old.concatenate_self_energies(sigma_dmft)

    # rank 0 holds a resumed run's full-BZ starting iterate: spread it like the loop's mixed iterate, one window
    # per node (the loop frees this window after the first proposal)
    if starting_iter > 0 and sc_node_comm is not None:
        sigma_old, sigma_win = _share_sigma_per_node(sigma_old, comm, sc_node_comm, sc_roots_comm)

    delta_sigma = sigma_dmft.cut_niv(config.box.niv_core) - sigma_local.cut_niv(config.box.niv_core)

    # only rank 0 finishes the proposal with the Hartree-Fock term, which reads the full q-grid interaction
    v_nonloc_full = v_nonloc if comm.rank == 0 else None
    v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    annealer = LambdaAnnealer() if config.stabilization.use_lambda_annealing else None
    pole_vertex = first_frequency_local_vertex() if comm.rank == 0 else None
    # the per-iteration correction keeps one dict for the whole run, so every channel's search warm-starts from the
    # lambda of the previous iteration; the one-shot correction takes precedence and searches without a warm start
    lambda_previous = (
        {}
        if config.stabilization.use_lambda_correction and not config.lambda_correction.perform_lambda_correction
        else None
    )
    if tracker is not None and comm.rank == 0 and starting_iter > 0:
        _carry_jacobian_spectrum(tracker, annealer)
    anneal_reset_iter = None
    release_iter = None
    pure_converged = False
    for current_iter in range(starting_iter + 1, starting_iter + config.self_consistency.max_iter + 1):
        logger.info("----------------------------------------")
        logger.info(f"Starting iteration {current_iter}.")
        logger.info("----------------------------------------")

        if pole_vertex is not None:
            _log_first_frequency_pole_ratio(first_frequency_pole_ratio(sigma_old, mu_history[-1], pole_vertex))

        sigma_new = calculate_sigma_proposal(
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
            lambda_previous=lambda_previous,
        )
        # delta_sigma = sigma_dmft.cut_niv(config.box.niv_core) - sigma_new.q_mean().cut_niv(config.box.niv_core)

        # only rank 0 mixes and measures the residual, so only rank 0 keeps a private copy of the previous iterate
        # (the cut copies everything still needed from the previous iteration's shared sigma buffer)
        shell_offset = sigma_old.shell_offset_from(sigma_dmft) if comm.rank == 0 else None
        sigma_old = sigma_old.cut_niv(config.box.niv_core) if comm.rank == 0 else None
        _free_shared_window(sigma_win, sc_node_comm)
        sigma_win = None

        logger.info("Applying mixing strategy to the self-energy.")
        history_cap = _mixing_history_cap(current_iter, release_iter, anneal_reset_iter, stab_event_iter)
        # mixing runs on rank 0 only, the only rank holding the proposal; the mixed Sigma then reaches the other
        # ranks once per node through a shared window instead of once per rank
        if comm.rank == 0:
            sigma_old = sigma_old.concatenate_self_energies(sigma_dmft, shell_offset=shell_offset)
            if config.stabilization.use_jacobian_stabilization:
                _save_sigma_iteration(sigma_new, "sigma_dga_proposal", current_iter)
            sigma_new = apply_mixing_strategy(sigma_new, sigma_old, history_cap, mixing_history, tracker)
            if tracker is not None:
                if _update_jacobian_tracker(
                    tracker, mixing_history, current_iter, release_iter, anneal_reset_iter, annealer
                ):
                    stab_event_iter = current_iter
                _append_jacobian_eigenvalues(tracker, rows_lam, rows_res, rows_damping)
        if sc_node_comm is not None:
            sigma_new, sigma_win = _share_sigma_per_node(sigma_new, comm, sc_node_comm, sc_roots_comm)

        sigma_new = sigma_new.compress_q_dimension()

        # Post-mixing step residual (the historical convergence measure; shrinks with the mixing parameter)
        relative_residual = (
            _relative_sigma_residual(sigma_new, sigma_old.compress_q_dimension()) if comm.rank == 0 else None
        )

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

        # new occupation matrix and energies from the new Green's function (outside the asympt region it is the
        # DMFT lattice Green's function); k-distributed, so no rank builds the whole DMFT-box Green's function
        n_current, config.sys.occ, config.sys.occ_k, ekin, epot = _update_occ_and_energies_distributed(
            sigma_new, sigma_dmft_full, mpi_dist_fullbz, config.sys.mu
        )
        logger.info(f"Filling of the updated Green's function: {n_current:.6f} (target {config.sys.n:.6f}).")
        logger.info(f"Kinetic energy: {ekin:.4f} [t or eV].")
        logger.info(f"Potential energy: {epot:.4f} [t or eV].")
        logger.info(f"Total energy: {(ekin + epot):.4f} [t or eV].")

        if config.self_consistency.max_iter > 1:
            logger.info("Updated occupation matrix from new Green's function.")

        if comm.rank == 0:
            _save_sigma_iteration(sigma_new, "sigma_dga", current_iter)

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
                pure_converged = True
                logger.info(f"Self-consistency of sigma and mu reached at iteration {current_iter}.")
                if tracker is not None and comm.rank == 0:
                    logger.info(
                        f"Jacobian tracker at convergence: {0 if tracker.q is None else tracker.q.shape[1]} flipped "
                        f"directions, p_eff={tracker.p_eff:.4f}, rho={tracker.predicted_rate():.4f}."
                    )
                    n_min = tracker.minimum_iterations()
                    if current_iter - starting_iter < n_min:
                        logger.warning(
                            f"Jacobian tracker: converged after {current_iter - starting_iter} iterations, below "
                            f"the minimum {n_min} the certified spectrum suggests at the damping each certified "
                            f"direction takes (eps = {tracker.certified_margin:.3e}, p_eff = {tracker.p_eff:.4f} "
                            f"on the directions the installed map leaves to the scalar step); a flat direction may "
                            f"not have settled."
                        )
                break
        else:
            logger.info("Self-consistency not reached.")

    # the exact Jacobian and the interpolation read the final sigma on every rank, so they run while the shared window
    # still holds it
    if pure_converged and tracker is not None and config.stabilization.use_exact_jacobian:
        _install_exact_jacobian(
            tracker,
            sigma_old,
            mu_history[-1],
            u_loc,
            v_nonloc,
            v_nonloc_full,
            sigma_dmft_full,
            mpi_dist_irrk,
            mpi_dist_fullbz,
            comm,
            chunk_budgets,
        )

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
        _free_shared_window(sigma_win, sc_node_comm)
    if sc_node_comm is not None:
        sc_roots_comm.Free()
        sc_node_comm.Free()

    mpi_dist_irrk.delete_file()
    mpi_dist_fullbz.delete_file()

    np.save(os.path.join(config.output.output_path, "mu_history.npy"), mu_history)
    logger.info("Saved mu history as numpy array.")

    if tracker is not None and comm.rank == 0:
        _save_jacobian_spectrum(tracker, pure_converged, rows_lam, rows_res, rows_damping)

    return sigma_old if comm.rank == 0 else None


def apply_mixing_strategy(
    sigma_new: SelfEnergy,
    sigma_old: SelfEnergy,
    history_cap: int | None = None,
    mixing_history: list | None = None,
    tracker: JacobianTracker | None = None,
) -> SelfEnergy:
    """
    Applies the self-energy mixing strategy for the self-consistency loop. Supports linear mixing as well as the
    accelerated Pulay (DIIS) and Anderson schemes; the accelerated schemes fall back to linear mixing when their
    least-squares problem is ill-conditioned or the history is too short. The mixing strategy is taken from the
    config. So is the mixing parameter, unless ``tracker`` is given: its bound :attr:`JacobianTracker.p_eff` then
    damps every damped Picard step below (linear mixing, the warm-up and the fallbacks of the accelerated schemes),
    while an accelerated step keeps the configured parameter, since a whole-spectrum bound describes the damped
    iteration and not a quasi-Newton step.

    The accelerated schemes build their secant history from genuine (iterate, proposal) pairs: each history entry
    holds an iterate :math:`x` together with the un-mixed proposal :math:`S(x)` the self-energy map produced from
    it. The post-mixing iterates alone (the per-iteration sigma files) cannot serve as proposals - at mixing
    parameters below one, ``mix(S(x), x, history) != S(x)``, so pairing consecutive iterates would feed the secant
    model inconsistent data. This function therefore records the current pair (cut to the core window, before
    mixing overwrites ``sigma_new`` in place) into ``mixing_history`` itself, under every strategy, since the
    Jacobian tracker measures the spectrum of the map from the same pairs. A momentum-local iterate (the fresh-run
    starting self-energy) is broadcast over the proposal's momentum grid.

    Whenever the tracker has a flipped subspace installed, the proposal residual is reflected on it
    (:meth:`JacobianTracker.reflect`) before anything else touches it: the reflected proposal enters the core
    window of ``sigma_new`` and is what every mixing strategy below acts on, regardless of whether
    ``mixing_history`` is given. When it is, the raw proposal is what gets recorded, since the tracker estimates
    the spectrum from it, and the reflected one is what the accelerated schemes take as their map result. The
    reflector carries no damping, so that map stays the same map for as long as the subspace does. The iterate a
    DAMPED step returns - linear mixing, and the warm-up and the fallbacks of the accelerated schemes - is then
    corrected on the certified span by :meth:`JacobianTracker.stabilize_step`: those directions take the step their
    own eigenvalues allow whatever the damping did to them, while a direction the estimate does not certify keeps
    the configured mixing, since the projector that takes the certified span out of the step is the oblique one
    along the uncertified subspace (the orthogonal projector where the two are too close to separate, which the
    tracker logs). An accelerated step keeps the step it computed on the reflected map: the per-direction damping
    is the damping of a Picard step, and overwriting a quasi-Newton step with it throws away the secant model
    exactly where the same secants certified it.

    :param sigma_new: The freshly computed self-energy proposal.
    :param sigma_old: The previous iteration's self-energy.
    :param history_cap: Optional upper bound on the number of secant pairs used by the accelerated schemes
        (``None`` for no bound). Used to reset the mixing history after the susceptibility-restriction release, so
        the accelerated schemes never extrapolate across the restricted-to-unrestricted discontinuity. Pairs keep
        being recorded while capped, so the history re-arms from genuine post-release pairs.
    :param mixing_history: Mutable list of (iterate, raw proposal, used proposal) triples of core-cut decompressed
        arrays, oldest first, maintained across iterations by the caller (rank 0 of the self-consistency loop).
        This function appends the current triple and trims the list to the longer of the configured history length
        and the tracker's pair window, plus one. ``None`` disables both the recording and the accelerated schemes.
    :param tracker: The :class:`JacobianTracker` owning the flipped subspace and the damping bound, or ``None`` to
        mix the raw proposal with the configured mixing parameter.
    :return: The mixed :class:`SelfEnergy` for the next iteration.
    """
    logger = config.logger
    n_hist = config.self_consistency.mixing_history_length
    if history_cap is not None:
        n_hist = min(n_hist, history_cap)
    p_config = config.self_consistency.mixing
    alpha = tracker.p_eff if tracker is not None else p_config
    tracker_active = tracker is not None and tracker.active
    accelerated_mixing_condition = (
        mixing_history is not None
        and n_hist > 0
        and len(mixing_history) >= n_hist
        and config.self_consistency.mixing_strategy.lower() in ("pulay", "anderson")
    )

    def stabilize(mixed: SelfEnergy) -> SelfEnergy:
        """
        Overwrites the certified span of a damped step with the tracker's stabilized step.

        :param mixed: The self-energy of whichever damped step produced it.
        :return: The same object, its core window corrected where a per-direction damping is installed.
        """
        if not tracker_active:
            return mixed
        niv_mixed = mixed.niv
        mixed_window = slice(niv_mixed - config.box.niv_core, niv_mixed + config.box.niv_core)
        mixed.mat[..., mixed_window] = tracker.stabilize_step(
            mixed.mat[..., mixed_window], iterate, proposal_raw - iterate
        )
        return mixed

    last_results, last_proposals = [], []
    if tracker_active or mixing_history is not None:
        niv_core = config.box.niv_core
        sigma_old.decompress_q_dimension()
        sigma_new.decompress_q_dimension()
        niv_o, niv_n = sigma_old.niv, sigma_new.niv
        core_window = slice(niv_n - niv_core, niv_n + niv_core)
        proposal_raw = sigma_new.mat[..., core_window].copy()
        # broadcast_to expands a momentum-local iterate (the fresh-run starting sigma) over the proposal's
        # k-grid; the copies must survive the in-place core-window update of sigma_new below
        iterate = np.broadcast_to(sigma_old.mat[..., niv_o - niv_core : niv_o + niv_core], proposal_raw.shape).copy()
        proposal_used = proposal_raw
        if tracker_active:
            proposal_used = tracker.reflect(proposal_raw, iterate)
            sigma_new.mat[..., core_window] = proposal_used

        if mixing_history is not None:
            mixing_history.append((iterate, proposal_raw, proposal_used))
            n_keep = max(config.self_consistency.mixing_history_length, TRACKER_PAIRS if tracker is not None else 0)
            del mixing_history[: -(n_keep + 1)]

            if config.self_consistency.mixing_strategy.lower() in ("pulay", "anderson") and n_hist > 0:
                pairs = mixing_history[-(n_hist + 1) :]
                last_proposals = [entry[0] for entry in pairs]  # [x_{n-m}, ..., x_n]
                last_results = [entry[2] for entry in pairs]  # [S(x_{n-m}), ..., S(x_n)]
                logger.info(f"Using the last {len(pairs)} (iterate, proposal) pairs of the mixing history.")

    if config.self_consistency.mixing_strategy.lower() == "pulay" and accelerated_mixing_condition:
        shape = last_results[-1].shape
        n_total = int(np.prod(shape))
        f_matrix = np.zeros((2 * n_total, n_hist), dtype=np.float64)
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

        # F[:,i] = (S(x_{n-i}) - S(x_{n-i-1})) - R[:,i] with the proposal differences R[:,i] = x_{n-i} - x_{n-i-1};
        # R is not stored, it is added back into F once the solve no longer needs F
        for i in range(n_hist):
            result_diff = get_result(-1 - i) - get_result(-2 - i)
            f_matrix[:n_total, i] = result_diff.real
            f_matrix[n_total:, i] = result_diff.imag

            proposal_diff = get_proposal(-1 - i) - get_proposal(-2 - i)
            f_matrix[:n_total, i] -= proposal_diff.real
            f_matrix[n_total:, i] -= proposal_diff.imag
        del result_diff, proposal_diff

        # Residual: F(x_n) - x_n, where x_n = last_proposals[-1] = sigma_old (core window)
        iter_diff = get_result(-1) - get_proposal(-1)
        f_i[:n_total] = iter_diff.real
        f_i[n_total:] = iter_diff.imag
        del iter_diff
        norm_f = np.linalg.norm(f_i)

        # Solve min||F @ c - f_i|| via truncated-SVD pseudoinverse (drops collinear directions)
        u, s, vh = np.linalg.svd(f_matrix, full_matrices=False)
        cutoff = 1e-5 * (s[0] if s.size else 1.0)
        mask = s > cutoff
        if not np.any(mask):
            logger.warning("Pulay SVD ill-conditioned - falling back to linear mixing.")
            return stabilize(alpha * sigma_new + (1 - alpha) * sigma_old)
        coeffs = vh[mask].T @ ((u[:, mask].T @ f_i) / s[mask])

        # Pulay update at the configured parameter p: x_{n+1} = x_n + p*f_i - (R + p*F) @ c, with R + p*F formed in
        # place of F
        del u
        f_matrix *= p_config
        for i in range(n_hist):
            proposal_diff = get_proposal(-1 - i) - get_proposal(-2 - i)
            f_matrix[:n_total, i] += proposal_diff.real
            f_matrix[n_total:, i] += proposal_diff.imag
        del proposal_diff
        update = p_config * f_i - f_matrix @ coeffs
        norm_u = np.linalg.norm(update)
        if norm_f > 0 and norm_u > 10.0 * norm_f:
            update *= 10.0 * norm_f / norm_u
            logger.warning(f"Pulay step clamped (norm_u={norm_u:.3e}, norm_f={norm_f:.3e}).")
        update = update[:n_total] + 1j * update[n_total:]

        # Update the new self energy
        niv = sigma_new.niv
        niv_core = config.box.niv_core
        sigma_new.mat[..., niv - niv_core : niv + niv_core] = get_proposal(-1).reshape(shape) + update.reshape(shape)

        logger.info(f"Pulay mixing applied (m={n_hist}, alpha={p_config:.3f}, norm_f={norm_f:.3e}).")

        return sigma_new
    if config.self_consistency.mixing_strategy.lower() == "anderson" and accelerated_mixing_condition:
        shape = last_results[-1].shape
        n_total = int(np.prod(shape))
        flat = lambda x: x.reshape(-1)

        # Current residual f_n = F(x_n) - x_n
        f_curr = flat(last_results[-1]) - flat(last_proposals[-1])
        f_vec = np.concatenate([f_curr.real, f_curr.imag])
        del f_curr
        norm_f = np.linalg.norm(f_vec)

        # dF (n_hist columns, real parts over imaginary parts): dF[:,i] = f_{n-i} - f_{n-i-1} (residual differences)
        df_matrix = np.empty((2 * n_total, n_hist), dtype=f_vec.dtype)
        for i in range(n_hist):
            df = (flat(last_results[-1 - i]) - flat(last_proposals[-1 - i])) - (
                flat(last_results[-2 - i]) - flat(last_proposals[-2 - i])
            )
            df_matrix[:n_total, i], df_matrix[n_total:, i] = df.real, df.imag
        del df

        # Anderson: solve min ||f_n - dF @ c||
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
            return stabilize(alpha * sigma_new + (1 - alpha) * sigma_old)

        # Undamped Anderson proposal: x_n + f_n - (dX + dF) @ c; the proposal differences
        # dX[:,i] = x_{n-i} - x_{n-i-1} are added into dF's columns once the solve no longer needs dF
        del u
        for i in range(n_hist):
            dx = flat(last_proposals[-1 - i]) - flat(last_proposals[-2 - i])
            df_matrix[:n_total, i] += dx.real
            df_matrix[n_total:, i] += dx.imag
        del dx
        x_n = flat(last_proposals[-1])
        x_anderson = np.concatenate([x_n.real, x_n.imag]) + f_vec - df_matrix @ coeffs
        x_anderson = x_anderson[:n_total] + 1j * x_anderson[n_total:]

        # Damp between old proposal and Anderson proposal
        x_n_complex = x_n
        candidate = (1 - p_config) * x_n_complex + p_config * x_anderson.reshape(-1)

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

        logger.info(f"Anderson acceleration applied (m={n_hist}, alpha={p_config:.3f}, norm_f={norm_f:.3e}).")

        return sigma_new

    sigma_new = alpha * sigma_new + (1 - alpha) * sigma_old
    logger.info(f"Sigma linearly mixed (m=1, alpha={alpha}).")
    return stabilize(sigma_new)

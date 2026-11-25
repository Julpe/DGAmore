# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Numerics of the non-local ladder DGA step, shared by the self-consistency driver (:mod:`~dgamore.nonlocal_sde`), the
exact Jacobian of its map (:mod:`~dgamore.sigma_jacobian`) and the Eliashberg step: the Hartree and Fock terms, the
auxiliary susceptibility :math:`\chi^{*;\mathrm{q}}_{r}`, the three-leg vertex :math:`\gamma^{\mathrm{q}\nu}_{r}`, the
shell-corrected physical susceptibility, the self-energy kernels (ladder, pole-free, double counting, RPA), their
contraction with the Green's function, the node-shared Green's function and local-vertex windows, the momentum
distribution of the irreducible zone and the occupation of the lattice Green's function.
"""

from collections.abc import Callable

import mpi4py.MPI as MPI
import numpy as np
import scipy as sp
from scipy.sparse.csgraph import connected_components

import dgamore.config as config
import dgamore.mpi_utils as mpi_utils
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction, orbital_block
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.local_four_point import LocalFourPoint
from dgamore.local_n_point import LocalNPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore import memory_estimator, symmetry_reduction
from dgamore.memory_estimator import SLICE_CHUNK_BYTES
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel, deferred_collection
from dgamore.self_energy import SelfEnergy

NODE_SHARED_SLICE_GAIN = 0.03  # least cut of the busiest rank's Bethe-Salpeter slices for which a node shares them


def real_space_interaction(u_loc: LocalInteraction, v_nonloc: Interaction) -> np.ndarray:
    r"""
    Returns the real-space interaction :math:`\mathcal{F}[U + V^{\mathbf{q}}]` the Fock term of
    :func:`get_hartree_fock` contracts with the transformed occupation.

    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}` on the full q-grid.
    :return: The array ``[kx, ky, kz, o1, o2, o3, o4]``.
    """
    return (u_loc + v_nonloc).fft(copy=False).decompress_q_dimension().mat


def get_hartree_fock(
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    occ: np.ndarray | None = None,
    occ_k: np.ndarray | None = None,
    w_r_mat: np.ndarray | None = None,
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
    :param w_r_mat: The real-space interaction ``(u_loc + v_nonloc).fft()`` as ``[kx, ky, kz, o1, o2, o3, o4]`` when
        the caller holds it across calls; ``None`` transforms it here.
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
    if w_r_mat is None:
        w_r_mat = real_space_interaction(u_loc, v_nonloc)  # [kx, ky, kz, a, b, c, d]
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


def _inactive_orbital_pairs(gamma_half: LocalFourPoint, u_r: LocalInteraction) -> np.ndarray:
    r"""
    Returns the orbital pairs without vertex of a channel (see
    :meth:`~dgamore.local_four_point.LocalFourPoint.orbital_pairs_without_vertex`) and logs their count when some,
    but not all, pairs are inactive.

    :param gamma_half: The local irreducible vertex :math:`\Gamma_{r}` on the half bosonic range.
    :param u_r: The channel-projected local interaction :math:`U_{r}`.
    :return: The sorted flat indices of the pairs without vertex.
    """
    inactive = gamma_half.orbital_pairs_without_vertex(u_r)
    if 0 < inactive.size < gamma_half.n_bands**2:
        config.logger.info(
            f"Auxiliary susceptibility ({gamma_half.channel.value}): {inactive.size} of {gamma_half.n_bands**2} "
            f"orbital pairs carry no vertex and are eliminated frequency by frequency."
        )
    return inactive


def create_auxiliary_chi_r_q_sum(
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    u_loc: LocalInteraction,
    chunk_bytes: int | None = None,
    rhs: list[FourPoint] | None = None,
    inactive: np.ndarray | None = None,
    slice_rhs: Callable[[int, int, FourPoint], list[FourPoint]] | None = None,
    involution: np.ndarray | None = None,
    subtract_u: bool = True,
    node_comm: MPI.Comm | None = None,
) -> FourPoint | list[FourPoint] | tuple[FourPoint, list[FourPoint]]:
    r"""
    Returns the sum over the auxiliary susceptibility, see Eq. (3.60) in my master's thesis,

    .. math:: \sum_{\nu'}\chi^{*;\mathrm{q}\nu\nu'}_{r;1234} = \sum_{\nu'}((\chi^{\mathrm{q}\nu}_{0;1234})^{-1} + (\Gamma^{\omega\nu\nu'}_{r;1234}-U_{r;1234})/\beta^2)^{-1},

    solved slice by slice by
    :meth:`~dgamore.four_point.FourPoint.invert_with_local_kernel_and_sum_over_last_vn` on the inverse bubble as the
    frequency-diagonal part, with the kernel :math:`\Gamma_{r}`, the shift :math:`U_{r}` and the scale
    :math:`1/\beta^2`: every rank-local momentum and bosonic frequency assembles its Bethe-Salpeter slice from a
    momentum-independent buffer per bosonic frequency and back-substitutes only the :math:`\nu'`-summed columns, so
    the transient is two compound slices regardless of the box size. The orbital pairs without vertex (see
    :meth:`~dgamore.local_four_point.LocalFourPoint.orbital_pairs_without_vertex`), found once per call unless the
    caller passes them, are eliminated frequency by frequency inside every slice.

    With ``subtract_u`` False the shift is zero and the sum is that of the ladder of :math:`\Gamma_{r}` alone,
    :math:`\sum_{\nu'}((\chi^{\mathrm{q}\nu}_{0;1234})^{-1} + \Gamma^{\omega\nu\nu'}_{r;1234}/\beta^2)^{-1}`, whose
    matrix stays regular where the shifted one has a near-zero eigenvalue the physical susceptibility does not have.

    With a ``node_comm`` and neither ``rhs`` nor ``slice_rhs``, the ranks of the node gather their momentum counts and,
    where sharing cuts the slice count of the node's busiest rank enough (see :func:`_share_slices_on_node`), share
    the slices of their momenta (see :func:`_solve_slices_on_node`), which rank 0's node logs. Every slice keeps its
    bits either way.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}` (full or half bosonic range).
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` (core box).
    :param u_loc: The bare local interaction :math:`U`.
    :param chunk_bytes: Not read: the build holds two compound slices at any budget.
    :param rhs: Right-hand sides in the layout of the result (half niw range, one fermionic dimension), one object
        per system; every slice then solves the Bethe-Salpeter system for all of them against its one factorization
        instead of summing its inverse (see
        :meth:`~dgamore.four_point.FourPoint.invert_with_local_kernel_and_sum_over_last_vn`), and the result is the
        list of solutions. None sums, as the ladder does.
    :param inactive: The orbital pairs without vertex of ``gamma_r`` (see
        :meth:`~dgamore.local_four_point.LocalFourPoint.orbital_pairs_without_vertex`) when the caller holds them;
        ``None`` finds them here and logs their count.
    :param slice_rhs: The callable ``(i, w, total) -> list[FourPoint]`` that returns the right-hand sides of the
        slice of momentum ``i`` and bosonic index ``w`` of ``gchi0_q_inv`` from that slice's summed auxiliary
        susceptibility ``total``; every slice then also solves them against the factorization of its sum, and the
        result is the tuple of the sum and the solutions (``rhs`` None). None sums only.
    :param involution: The orbital permutation that transposes the dispersion (see :func:`orbital_involution`),
        with which slices that are complex-symmetric only with their orbital pairs permuted take the permuted
        factorization (see
        :meth:`~dgamore.four_point.FourPoint.invert_with_local_kernel_and_sum_over_last_vn`), or None.
    :param subtract_u: Whether the shift is :math:`U_{r}` (the auxiliary susceptibility) or zero.
    :param node_comm: The node-local communicator, over which the call is then collective and whose ranks may share
        their slices, or None to solve rank by rank.
    :return: The frequency-summed auxiliary susceptibility :math:`\sum_{\nu'}\chi^{*;\mathrm{q}}_{r}` (with
        ``subtract_u`` False that of the ladder of :math:`\Gamma_{r}` alone) as a :class:`FourPoint` (half niw range,
        one fermionic dimension), the solutions for ``rhs`` as a list, or with ``slice_rhs`` the tuple
        ``(sum, solutions)``.
    """
    u_r = u_loc.as_channel(gamma_r.channel)
    gamma_half = gamma_r.copy().to_half_niw_range() if gamma_r.full_niw_range else gamma_r
    if inactive is None:
        inactive = _inactive_orbital_pairs(gamma_half, u_r)
    beta = config.sys.beta
    shift = u_r if subtract_u else u_r.scale(0.0, copy=True)
    if node_comm is not None and rhs is None and slice_rhs is None:
        rows = gchi0_q_inv.to_half_niw_range().compress_q_dimension().mat  # [q, o1, o2, o3, o4, w, v]
        counts, n_w = np.array(node_comm.allgather(rows.shape[0])), rows.shape[-2]
        if _share_slices_on_node(counts, n_w):
            config.logger.info(
                f"Bethe-Salpeter slices ({gamma_r.channel.value}) shared by the {counts.size} ranks of each node: at "
                f"most {-(-counts.sum() * n_w // counts.size)} instead of {counts.max() * n_w} per rank."
            )
            return _solve_slices_on_node(gchi0_q_inv, counts, node_comm, gamma_half, shift, inactive, involution)
    return gchi0_q_inv.invert_with_local_kernel_and_sum_over_last_vn(
        gamma_half, shift, 1.0 / beta**2, beta, inactive, rhs, slice_rhs, involution
    )


def _share_slices_on_node(counts: np.ndarray, n_w: int) -> bool:
    """
    Returns whether the ranks of a node share the Bethe-Salpeter slices of their momenta (see
    :func:`_solve_slices_on_node`): when equal parts of the node's slices cut the slice count of its busiest rank by
    at least the fraction ``NODE_SHARED_SLICE_GAIN``. A single rank and a node without momenta never share.

    :param counts: The momentum counts of the node's ranks.
    :param n_w: Number of bosonic frequencies, i.e. slices, of a momentum.
    :return: Whether the node shares its slices.
    """
    slices = int(counts.sum()) * n_w
    if counts.size < 2 or slices == 0:
        return False
    return -(-slices // counts.size) <= (1 - NODE_SHARED_SLICE_GAIN) * int(counts.max()) * n_w


def _solve_slices_on_node(
    gchi0_q_inv: FourPoint,
    counts: np.ndarray,
    node_comm: MPI.Comm,
    gamma_half: LocalFourPoint,
    shift: LocalInteraction,
    inactive: np.ndarray,
    involution: np.ndarray | None,
) -> FourPoint:
    r"""
    Returns this rank's sum of :func:`create_auxiliary_chi_r_q_sum` with the Bethe-Salpeter slices of the node's
    momenta shared by its ranks. Every rank copies its rows of the inverse bubble into a node window
    ``[q_node, o1, o2, o3, o4, w, v]`` (the ranks' rows in their node order) and solves its part of the node's slices
    in frequency-major order (see :func:`~dgamore.mpi_utils.node_slice_runs`) into a node result window, building
    :math:`A^{\omega\nu\nu'}` once per frequency of its part. Once the input window is freed, it copies its own rows
    of the result into an array of its inverse bubble's memory layout, which fixes the bits of later frequency sums.
    Every slice is solved as without the node, so the sum has the bits of the per-rank call. The two windows, and
    then the result window beside the rank's own result, hold one block per rank more than that call's result.
    Collective over ``node_comm`` only.

    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` (core box, half niw range,
        compressed momenta).
    :param counts: The momentum counts of the node's ranks, in their node order.
    :param node_comm: The node-local communicator.
    :param gamma_half: The local irreducible vertex :math:`\Gamma_{r}` on the half bosonic range.
    :param shift: The frequency-independent shift, :math:`U_{r}` or zero.
    :param inactive: The orbital pairs without vertex of ``gamma_half``.
    :param involution: The orbital permutation that transposes the dispersion (see :func:`orbital_involution`), or
        None.
    :return: This rank's frequency-summed auxiliary susceptibility as a :class:`FourPoint` (half niw range, one
        fermionic dimension).
    """
    beta, diagonal = config.sys.beta, gchi0_q_inv.mat
    start = int(counts[: node_comm.rank].sum())
    own, shape = slice(start, start + diagonal.shape[0]), (int(counts.sum()), *diagonal.shape[1:])
    rows, rows_win = mpi_utils.allocate_node_shared_array(node_comm, shape, diagonal.dtype)
    sums, sums_win = mpi_utils.allocate_node_shared_array(node_comm, shape, diagonal.dtype)
    rows[own] = diagonal
    node_comm.Barrier()
    window = FourPoint(rows, gchi0_q_inv.channel, (shape[0], 1, 1), 1, 1, False, gchi0_q_inv.full_niv_range, True)
    window.frequency_notation = gchi0_q_inv.frequency_notation
    runs = mpi_utils.node_slice_runs(shape[0], shape[-2], node_comm.rank, node_comm.size)
    window.invert_with_local_kernel_and_sum_over_last_vn(
        gamma_half, shift, 1.0 / beta**2, beta, inactive, orbital_involution=involution, runs=runs, out=sums
    )
    window = rows = None
    mpi_utils.free_shared_window(rows_win, node_comm)  # its barrier: every rank has solved its part
    mine = np.empty_like(diagonal)
    mine[...] = sums[own]
    sums = None
    mpi_utils.free_shared_window(sums_win, node_comm)
    return FourPoint(mine, gamma_half.channel, gchi0_q_inv.nq, 1, 1, False, has_compressed_q_dimension=True)


def create_vrg_r_q(gchi_aux_q_r_sum: FourPoint, gchi0_q_inv: FourPoint, copy: bool = True) -> FourPoint:
    r"""
    Returns the momentum-dependent three-leg vertex, see Eq. (3.63) in my master's thesis,
    :math:`\gamma^{\mathrm{q}\nu}_{r;1234} = \beta \sum_{ab} \sum_{\nu'} (\chi^{\mathrm{q}\nu}_{0;12ab})^{-1}
    \chi^{*;\mathrm{q}\nu\nu'}_{r;ba34}`.

    :param gchi_aux_q_r_sum: The frequency-summed auxiliary susceptibility
        :math:`\sum_{\nu'}\chi^{*;\mathrm{q}\nu\nu'}_{r}`, or that of the ladder of :math:`\Gamma_{r}` alone (see
        :func:`create_auxiliary_chi_r_q_sum` with ``subtract_u`` False), which gives its three-leg vertex.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` (core box).
    :param copy: If True, return the vertex as a new object; if False, write it over ``gchi_aux_q_r_sum`` in place and
        return that object.
    :return: The three-leg vertex :math:`\gamma^{\mathrm{q}\nu}_{r}` (``vrg``) as a :class:`FourPoint`.
    """
    return gchi_aux_q_r_sum.matmul(gchi0_q_inv, left_hand_side=False, copy=copy).scale(config.sys.beta)


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


def create_ladder_vertex_middle(
    x_g: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    chi: FourPoint,
    chi_hat: FourPoint,
) -> FourPoint:
    r"""
    Returns the matrix between the three-leg vertices of the ladder vertex built from the ladder of :math:`\Gamma_{r}`
    alone (see :func:`create_auxiliary_chi_r_q_sum` with ``subtract_u`` False), in compound products,

    .. math:: Y^{\mathrm{q}}_{r} = D^{-1} (V^{\mathbf{q}}_{r} - \mathcal{U}^{\mathbf{q}}_{r} S U_{r})
        - (\mathbb{1} - U_{r} X)^{-1} \mathcal{U}^{\mathbf{q}}_{r} (\hat\chi - \chi) \mathcal{U}^{\mathbf{q}}_{r}
        (\mathbb{1} - X U_{r})^{-1}, \qquad
        D = \mathbb{1} + V^{\mathbf{q}}_{r} X + \mathcal{U}^{\mathbf{q}}_{r} S (\mathbb{1} - U_{r} X),

    with :math:`X` the summed ladder ``x_g``, :math:`S` the shell bubble (full-box minus core-box sum),
    :math:`\mathcal{U}^{\mathbf{q}}_{r} = U_{r} + V^{\mathbf{q}}_{r}`, :math:`\chi` the shell-corrected physical
    susceptibility and :math:`\hat\chi` the one the self-energy kernel reads. The ladder vertex
    :math:`F_{r} = F^{\Gamma}_{r} + (\gamma^{\Gamma}_{r} Y_{r}) \tilde\gamma^{\Gamma}_{r}`, with the full vertex and
    the left and right three-leg vertices of the ladder of :math:`\Gamma_{r}`, is then exactly the vertex of the
    auxiliary ladder (shift :math:`U_{r}`) with its three-leg vertices and :math:`\hat\chi` (Woodbury identity on the
    frequency-constant shift). The first term never inverts a matrix that is singular where the auxiliary ladder is,
    so it stays exact wherever the physical susceptibility is regular. The second term carries
    :math:`(\mathbb{1} - U_{r} X)^{-1}`, singular there, and is formed only at the momenta where :math:`\hat\chi
    \neq \chi` (an annealing, restriction or lambda correction of the channel).

    :param x_g: The summed ladder of :math:`\Gamma_{r}`, :math:`\frac{1}{\beta^2}\sum_{\nu\nu'}
        ((\chi^{\mathrm{q}\nu}_{0})^{-1} + \Gamma^{\omega\nu\nu'}_{r}/\beta^2)^{-1}`.
    :param gchi0_q_full_sum: The frequency-summed bare bubble over the full box.
    :param gchi0_q_core_sum: The frequency-summed bare bubble over the core box.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param chi: The shell-corrected physical susceptibility :math:`\chi^{\mathrm{q}}_{r}`.
    :param chi_hat: The physical susceptibility the self-energy kernel reads.
    :return: The matrix :math:`Y^{\mathrm{q}}_{r}` as a :class:`FourPoint` in the channel of ``x_g`` (half niw range,
        no fermionic dimension).
    """
    u_r, v_r, u_tot, one, shell, d_inv = _gamma_ladder_shift_parts(
        x_g, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
    )
    middle = d_inv @ ((u_tot @ shell @ u_r).scale(-1.0) + v_r)
    delta = chi_hat - chi
    for start, stop in _changed_momentum_windows(delta):
        x_w, u_w = x_g.take_q_index_slice(start, stop, copy=False), u_tot.take_q_index_slice(start, stop, copy=False)
        left = ((u_r @ x_w).scale(-1.0) + one).invert()
        right = ((x_w @ u_r).scale(-1.0) + one).invert()
        middle.take_q_index_slice(start, stop, copy=False).sub(
            left @ (u_w @ delta.take_q_index_slice(start, stop, copy=False) @ u_w) @ right, copy=False
        )
    return middle


def _changed_momentum_windows(delta: FourPoint) -> list[tuple[int, int]]:
    r"""
    Returns the windows of consecutive momenta at which :math:`\hat\chi - \chi` has a non-zero entry, the only
    momenta whose vertex and kernel middles get the term carrying :math:`(\mathbb{1} - U_{r} X)^{-1}`.

    :param delta: The difference :math:`\hat\chi - \chi` (compressed momenta).
    :return: The windows ``(start, stop)`` in momentum order.
    """
    changed = np.any(delta.mat.reshape(delta.mat.shape[0], -1) != 0, axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], changed.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _gamma_ladder_shift_parts(
    x_g: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
) -> tuple:
    r"""
    Returns what the pole-free forms of the shell-corrected susceptibility, the kernel and the ladder vertex share: the
    channel interactions :math:`U_{r}`, :math:`V^{\mathbf{q}}_{r}` and :math:`\mathcal{U}^{\mathbf{q}}_{r} = U_{r} +
    V^{\mathbf{q}}_{r}`, the compound identity, the shell bubble :math:`S` (full-box minus core-box sum) and

    .. math:: D^{-1} = [\mathbb{1} + V^{\mathbf{q}}_{r} X + \mathcal{U}^{\mathbf{q}}_{r} S (\mathbb{1} - U_{r} X)]^{-1},

    in compound products, with :math:`X` the summed ladder of :math:`\Gamma_{r}`. :math:`D` is regular wherever the
    physical susceptibility is, also where :math:`\mathbb{1} - U_{r} X` (the auxiliary ladder) is singular.

    :param x_g: The summed ladder of :math:`\Gamma_{r}`, :math:`\frac{1}{\beta^2}\sum_{\nu\nu'}
        ((\chi^{\mathrm{q}\nu}_{0})^{-1} + \Gamma^{\omega\nu\nu'}_{r}/\beta^2)^{-1}`.
    :param gchi0_q_full_sum: The frequency-summed bare bubble over the full box.
    :param gchi0_q_core_sum: The frequency-summed bare bubble over the core box.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :return: ``(u_r, v_r, u_tot, one, shell, d_inv)``.
    """
    channel = x_g.channel
    u_r, v_r = u_loc.as_channel(channel), v_nonloc.as_channel(channel)
    u_tot = u_r + v_r
    eye = np.eye(u_loc.n_bands)
    one = LocalInteraction(np.einsum("ad,bc->abcd", eye, eye), channel)  # the compound identity
    shell = gchi0_q_full_sum - gchi0_q_core_sum
    d_inv = (x_g.matmul(v_r, left_hand_side=False) + u_tot @ (shell - shell @ u_r @ x_g) + one).invert()
    return u_r, v_r, u_tot, one, shell, d_inv


def create_shell_corrected_chi_from_gamma_ladder(
    x_g: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
) -> FourPoint:
    r"""
    Returns the shell-corrected physical susceptibility of :func:`create_generalized_chi_q_with_shell_correction` from
    the summed ladder of :math:`\Gamma_{r}` alone (see :func:`create_auxiliary_chi_r_q_sum` with ``subtract_u``
    False), in compound products,

    .. math:: \chi^{\mathrm{q}}_{r} = [X + S (\mathbb{1} - U_{r} X)]\, D^{-1}

    with :math:`D` of :func:`_gamma_ladder_shift_parts` (Woodbury identity on the frequency-constant shift, the
    auxiliary sum being :math:`X (\mathbb{1} - U_{r} X)^{-1}`). Neither :math:`\mathbb{1} - U_{r} X` nor :math:`X`
    or :math:`U_{r}` is inverted, so the result stays exact where the auxiliary ladder is singular.

    :param x_g: The summed ladder of :math:`\Gamma_{r}`, :math:`\frac{1}{\beta^2}\sum_{\nu\nu'}
        ((\chi^{\mathrm{q}\nu}_{0})^{-1} + \Gamma^{\omega\nu\nu'}_{r}/\beta^2)^{-1}`.
    :param gchi0_q_full_sum: The frequency-summed bare bubble over the full box.
    :param gchi0_q_core_sum: The frequency-summed bare bubble over the core box.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :return: The shell-corrected physical susceptibility :math:`\chi^{\mathrm{q}}_{r}` as a :class:`FourPoint`.
    """
    u_r, _, _, _, shell, d_inv = _gamma_ladder_shift_parts(x_g, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc)
    return (x_g + shell - shell @ u_r @ x_g) @ d_inv


def create_kernel_middle(
    x_g: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    chi: FourPoint,
    chi_hat: FourPoint,
) -> FourPoint:
    r"""
    Returns the matrix that right-multiplies the three-leg vertex of the ladder of :math:`\Gamma_{r}` in the
    Schwinger-Dyson bracket, in compound products,

    .. math:: Y^{\mathrm{q}}_{r} = D^{-1} - (\mathbb{1} - U_{r} X)^{-1} \mathcal{U}^{\mathbf{q}}_{r}
        (\hat\chi - \chi),

    with :math:`D` of :func:`_gamma_ladder_shift_parts`, :math:`\chi` the shell-corrected physical susceptibility and
    :math:`\hat\chi` the one the kernel reads. Then :math:`\gamma^{\Gamma}_{r} Y_{r} = \gamma^{*}_{r} -
    \gamma^{*}_{r} \mathcal{U}^{\mathbf{q}}_{r} \hat\chi`, the bracket of :func:`calculate_kernel_r_q` with the
    auxiliary three-leg vertex :math:`\gamma^{*}_{r} = \gamma^{\Gamma}_{r} (\mathbb{1} - U_{r} X)^{-1}`. The first
    term stays exact where the auxiliary ladder is singular; the second carries :math:`(\mathbb{1} - U_{r} X)^{-1}` and
    is formed only at the momenta where :math:`\hat\chi \neq \chi` (an annealing, restriction or lambda correction of
    the channel).

    :param x_g: The summed ladder of :math:`\Gamma_{r}`, :math:`\frac{1}{\beta^2}\sum_{\nu\nu'}
        ((\chi^{\mathrm{q}\nu}_{0})^{-1} + \Gamma^{\omega\nu\nu'}_{r}/\beta^2)^{-1}`.
    :param gchi0_q_full_sum: The frequency-summed bare bubble over the full box.
    :param gchi0_q_core_sum: The frequency-summed bare bubble over the core box.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param chi: The shell-corrected physical susceptibility :math:`\chi^{\mathrm{q}}_{r}`.
    :param chi_hat: The physical susceptibility the self-energy kernel reads.
    :return: The matrix :math:`Y^{\mathrm{q}}_{r}` as a :class:`FourPoint` (half niw range, no fermionic dimension).
    """
    u_r, _, u_tot, one, _, d_inv = _gamma_ladder_shift_parts(x_g, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc)
    delta = chi_hat - chi
    for start, stop in _changed_momentum_windows(delta):
        x_w = x_g.take_q_index_slice(start, stop, copy=False)
        d_inv.take_q_index_slice(start, stop, copy=False).sub(
            ((u_r @ x_w).scale(-1.0) + one).invert()
            @ (u_tot.take_q_index_slice(start, stop, copy=False) @ delta.take_q_index_slice(start, stop, copy=False)),
            copy=False,
        )
    return d_inv


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
    return attach_dc_interaction(contract_dc_vertex(f_dc_loc, gchi0_q), u_loc)


def contract_dc_vertex(
    f_dc_loc: LocalFourPoint,
    gchi0_q: FourPoint,
    components: np.ndarray | None = None,
    outer_sum: FourPoint | None = None,
) -> FourPoint:
    r"""
    Returns the vertex contraction of the double-counting kernel (see :func:`calculate_sigma_dc_kernel`): the bubble
    summed with the local vertex over its stored full-box first frequency (see
    :meth:`~dgamore.four_point.FourPoint.contract_first_pair_with_local_vertex`), kept on the core fermionic box;
    the bubble is only read.

    :param f_dc_loc: The local double-counting vertex :math:`F_{\mathrm{dc}}` (see
        :func:`~dgamore.local_sde.double_counting_vertex`), or its window of the bubble's bosonic frequencies.
    :param gchi0_q: The momentum-dependent bare bubble :math:`\chi^{\mathrm{q}\nu}_{0}`: on the full fermionic box, or
        on the core box with ``outer_sum``.
    :param components: The non-zero orbital components of the whole vertex, or ``None`` to find them in ``f_dc_loc``.
    :param outer_sum: The bubble summed over the fermionic frequencies outside the core box (no prefactor), for a
        vertex constant in its first frequency there, or ``None``.
    :return: The contraction as a :class:`FourPoint` on the core fermionic box.
    """
    # the stored full-box first index supplies the summed v' via F_{1dfe}(v,v') = F_{efd1}(v',v): the stored slots
    # (e,f,d,1) then pair directly with chi0's row pair (e,f) - the standard-product reversal F_{..fe} chi0_{ef..}
    niv_out = min(f_dc_loc.niv_second, config.box.niv_core)
    return gchi0_q.contract_first_pair_with_local_vertex(f_dc_loc, niv_out, components, outer_sum)


def bubble_box_parts(
    blocks: list[FourPoint],
    f_dc_loc: LocalFourPoint | None,
    with_core: bool,
    outer_sums: list[FourPoint] | None = None,
) -> tuple[FourPoint, FourPoint | None, FourPoint | None]:
    r"""
    Returns what the kernel section reads of the full-box bubble, from the bubble given as consecutive bosonic
    windows: its frequency sum :math:`\frac{1}{\beta^2} \sum_{\nu} \chi^{\mathrm{q}\nu}_{0}` over the full box, the
    vertex contraction of the double-counting kernel (see :func:`contract_dc_vertex`) and the core-box cut. The
    windows are consumed in turn: each is freed once its parts are taken, so the whole full-box bubble never sits
    beside the whole core box; every part is the window of the whole bubble's result, bit for bit.

    With ``outer_sums`` the windows hold the core box alone and come with the bubble's sum over the fermionic
    frequencies outside it: all three parts read those frequencies only through that sum (the double-counting vertex
    is constant in its summed frequency there, see :meth:`LocalFourPoint.first_frequency_is_constant_outside`), so
    the parts equal those of the full-box bubble to rounding and the core windows are handed on as they are.

    :param blocks: The bubble's bosonic windows in order (see
        :meth:`~dgamore.bubble_gen.BubbleGenerator.create_generalized_chi0_q_fft`); emptied.
    :param f_dc_loc: The local double-counting vertex, or ``None`` for no contraction.
    :param with_core: Whether to cut the core box.
    :param outer_sums: Per window of ``blocks`` on the core box, the bubble's sum over the frequencies outside it
        (no prefactor); emptied. ``None`` for windows on the full box.
    :return: The tuple ``(full_sum, dc_contraction, core)``; ``None`` for the parts not asked for.
    """
    beta = config.sys.beta
    components = None if f_dc_loc is None else f_dc_loc.nonzero_orbital_components()
    sums, contractions, cores = [], [], []
    w0 = 0
    with deferred_collection():
        while blocks:
            block = blocks.pop(0)
            outer = outer_sums.pop(0) if outer_sums else None
            w1 = w0 + block.current_shape[-2]
            sums.append(block.sum_over_all_vn(beta).scale(1.0 / beta))
            if outer is not None:
                sums[-1].add(outer.scale(1.0 / beta**2, copy=True), copy=False)
            if f_dc_loc is not None:
                window = f_dc_loc.take_wn_slice(w0, w1, copy=False)
                contractions.append(contract_dc_vertex(window, block, components, outer))
            if with_core:
                cores.append(block if outer is not None else block.cut_niv(config.box.niv_core))
            if outer is None or not with_core:
                block.free()
            if outer is not None:
                outer.free()
            w0 = w1
        parts = [LocalNPoint.join_wn(part) if part else None for part in (sums, contractions, cores)]
    return tuple(parts)


def attach_dc_interaction(contraction: FourPoint, u_loc: LocalInteraction, per_wn: bool = False) -> FourPoint:
    r"""
    Returns the double-counting kernel from its vertex contraction (see :func:`contract_dc_vertex`): the transversal
    attachment of the interaction, the contraction layout and the prefactor of :func:`calculate_sigma_dc_kernel`.

    :param contraction: The vertex contraction on the core fermionic box (only read).
    :param u_loc: The bare local interaction :math:`U`.
    :param per_wn: Whether the attachment takes the bosonic frequency as a batch index (see
        :meth:`~dgamore.four_point.FourPoint.matmul`), so a bosonic window's kernel does not depend on the frequencies
        outside it.
    :return: The double-counting kernel (contraction layout) as a :class:`FourPoint`.
    """
    # transversal attachment U_{acb2} = -U_magn;a2bc: [T @ U_magn]_{1da2}, then into the contraction layout
    kernel = contraction.matmul(u_loc.as_channel(SpinChannel.MAGN), per_wn=per_wn).permute_orbitals(
        "abcd->badc", copy=False
    )
    return kernel.scale(2.0 / config.sys.beta**2, copy=False)


def calculate_bare_dc_kernel(gchi0_q_full_sum: FourPoint, u_loc: LocalInteraction) -> FourPoint:
    r"""
    Returns the double-counting kernel of :func:`calculate_sigma_dc_kernel` for the bare transversal vertex
    :math:`F_{\mathrm{dc}} = \tfrac12(U_{\mathrm{d}} + 3U_{\mathrm{m}})`, the lowest order of the local vertex that
    kernel subtracts: ladders built on the bare interaction start with it in both bracket terms, and the kernel leaves
    it counted once. The vertex does not depend on frequency, so the sum over :math:`\nu'` acts on the bubble alone and
    the kernel carries no fermionic frequency. The exchange of the stored and the summed frequency in
    :func:`calculate_sigma_dc_kernel` holds for it, since :math:`F_{\mathrm{dc};1234} = F_{\mathrm{dc};4321}` for an
    interaction with the reality and pair-exchange symmetry.

    :param gchi0_q_full_sum: The frequency-summed bare bubble over the full box,
        :math:`\frac{1}{\beta^2}\sum_{\nu}\chi^{\mathrm{q}\nu}_{0}`.
    :param u_loc: The bare local interaction :math:`U`.
    :return: The double-counting kernel (contraction layout) as a :class:`FourPoint` without fermionic frequency.
    """
    # beta^2 turns the normalized bubble sum into the plain frequency sum the attachment expects
    f_dc = (u_loc.as_channel(SpinChannel.MAGN).scale(3.0) + u_loc.as_channel(SpinChannel.DENS)).scale(
        0.5 * config.sys.beta**2
    )
    return attach_dc_interaction(f_dc @ gchi0_q_full_sum, u_loc)


def calculate_kernel_r_q(
    vrg_q_r: FourPoint, chi_phys_q_r: FourPoint, v_nonloc: Interaction, u_loc: LocalInteraction, per_wn: bool = False
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
    :math:`\Sigma_{12} \propto \sum_{ab} K'_{a12b} G_{ab}` of :func:`run_column_sde` applies unchanged.

    :param vrg_q_r: The momentum-dependent three-leg vertex :math:`\gamma^{\mathrm{q}\nu}_{r}`.
    :param chi_phys_q_r: The (shell-corrected) physical susceptibility :math:`\chi^{\mathrm{phys};\mathrm{q}}_{r}`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param u_loc: The bare local interaction :math:`U`.
    :param per_wn: Whether the products with the channel interaction take the bosonic frequency as a batch index
        (see :meth:`~dgamore.four_point.FourPoint.matmul`), so a bosonic window's kernel does not depend on the
        frequencies outside it.
    :return: The self-energy kernel :math:`[(K-\mathbb{1})\,\mathcal{U}_r]` (contraction layout) as a
        :class:`FourPoint`.
    """
    u_r = v_nonloc.as_channel(vrg_q_r.channel) + u_loc.as_channel(vrg_q_r.channel)
    return _kernel_from_bracket(vrg_q_r - vrg_q_r.matmul(u_r, per_wn=per_wn) @ chi_phys_q_r, u_r, per_wn)


def calculate_pole_free_kernel_r_q(
    vrg_q_r: FourPoint, middle_q_r: FourPoint, v_nonloc: Interaction, u_loc: LocalInteraction
) -> FourPoint:
    r"""
    Returns the self-energy kernel of :func:`calculate_kernel_r_q` from the three-leg vertex of the ladder of
    :math:`\Gamma_{r}` alone, whose bracket :math:`\gamma^{\Gamma}_{r} Y_{r}` equals
    :math:`\gamma^{*}_{r} - \gamma^{*}_{r} \mathcal{U}^{\mathbf{q}}_{r} \hat\chi` (see
    :func:`create_kernel_middle`) without the zero eigenvalues of the auxiliary ladder.

    :param vrg_q_r: The three-leg vertex :math:`\gamma^{\Gamma;\mathrm{q}\nu}_{r}` of the ladder of
        :math:`\Gamma_{r}`.
    :param middle_q_r: The matrix :math:`Y^{\mathrm{q}}_{r}` of :func:`create_kernel_middle`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param u_loc: The bare local interaction :math:`U`.
    :return: The self-energy kernel :math:`[(K-\mathbb{1})\,\mathcal{U}_r]` (contraction layout) as a
        :class:`FourPoint`.
    """
    u_r = v_nonloc.as_channel(vrg_q_r.channel) + u_loc.as_channel(vrg_q_r.channel)
    return _kernel_from_bracket(vrg_q_r @ middle_q_r, u_r)


def _kernel_from_bracket(kernel: FourPoint, u_r: Interaction, per_wn: bool = False) -> FourPoint:
    r"""
    Subtracts the identity from the Schwinger-Dyson bracket :math:`K` in place and returns
    :math:`(K - \mathbb{1})\,\mathcal{U}^{\mathbf{q}}_{r}` in the layout of the sigma contraction (see
    :func:`calculate_kernel_r_q`).

    :param kernel: The bracket :math:`K`; overwritten.
    :param u_r: The channel-projected total interaction :math:`\mathcal{U}^{\mathbf{q}}_{r}`.
    :param per_wn: Whether the product with the channel interaction takes the bosonic frequency as a batch index
        (see :meth:`~dgamore.four_point.FourPoint.matmul`).
    :return: The self-energy kernel (contraction layout) as a :class:`FourPoint`.
    """
    # subtract the identity directly on the compound-identity positions (o1 == o4, o2 == o3, every frequency) instead
    # of materializing an identity block; the channel weights (1x dens, 3x magn) turn this into -U_d - 3U_m
    orb = np.arange(kernel.n_bands)
    kernel.mat[:, orb[:, None], orb[None, :], orb[None, :], orb[:, None], ...] -= 1.0

    return kernel.matmul(u_r, per_wn=per_wn).permute_orbitals("abcd->badc", copy=False)


def calculate_rpa_kernel_r_q(chi_q_r: FourPoint, v_nonloc: Interaction, u_loc: LocalInteraction) -> FourPoint:
    r"""
    Returns the self-energy kernel of :func:`calculate_kernel_r_q` at a unit three-leg vertex, the value it takes when
    the local irreducible vertex is the bare interaction, :math:`\Gamma_{r} = U_{r}`,

    .. math:: [(K - \mathbb{1})\,\mathcal{U}^{\mathbf{q}}_{r}]_{1234} = -\sum_{abcd} \mathcal{U}^{\mathbf{q}}_{r;12ab}\,
        \chi^{\mathrm{q}}_{r;bacd}\, \mathcal{U}^{\mathbf{q}}_{r;dc34},

    stored in the layout of the sigma contraction. It does not depend on the fermionic frequency and carries no
    fermionic axis.

    :param chi_q_r: The physical susceptibility :math:`\chi^{\mathrm{q}}_{r}` (see :func:`create_rpa_susceptibility`).
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param u_loc: The bare local interaction :math:`U`.
    :return: The self-energy kernel (contraction layout) as a :class:`FourPoint` without fermionic frequency.
    """
    u_r = v_nonloc.as_channel(chi_q_r.channel) + u_loc.as_channel(chi_q_r.channel)
    return (u_r @ chi_q_r @ u_r).scale(-1.0).permute_orbitals("abcd->badc", copy=False)


def create_bubble_tail(n_bands: int, niw: int, niv: int, beta: float) -> LocalFourPoint:
    r"""
    Returns the part of the frequency-summed bare bubble :math:`\chi^{\mathrm{q}}_{0;1234} = -\frac{1}{\beta}
    \frac{1}{n_{\mathbf{k}}} \sum_{\mathrm{k}} G^{\mathrm{k}}_{14} G^{\mathrm{k}-\mathrm{q}}_{32}` that the box
    :math:`\nu \in [-\mathrm{niv}, \mathrm{niv})` of
    :meth:`~dgamore.bubble_gen.BubbleGenerator.create_generalized_chi0_q_fft` leaves out, to the leading order
    :math:`G^{\mathrm{k}}_{12} \simeq \delta_{12} / i\nu` of the Green's function at large frequency,

    .. math:: \delta_{14} \delta_{23}\, T(\omega), \qquad T(\omega) = \frac{1}{\beta} \sum_{\nu \notin \mathrm{box}}
        \frac{1}{\nu (\nu - \omega)} = \frac{\beta}{4} \delta_{\omega 0} - \frac{1}{\beta} \sum_{\nu \in \mathrm{box}}
        \frac{1}{\nu (\nu - \omega)},

    since the sum over all :math:`\nu` is :math:`\beta/4` at :math:`\omega = 0` and vanishes at every other bosonic
    frequency. This order is momentum independent and the same for every Green's function; for :math:`\omega` well
    inside the box the next one carries the dispersion and the self-energy and falls off as
    :math:`(\beta/\mathrm{niv})^3`, while towards the box edge the left-out :math:`\nu` reach
    :math:`\nu \approx \omega`, where :math:`G^{\mathrm{k}-\mathrm{q}}` is far from its asymptote, and the error stays
    of first order. Without the tail the box sum is short by about :math:`\beta / (2 \pi^2 \mathrm{niv})` on these
    elements, which keeps an RPA susceptibility away from its Stoner pole.

    :param n_bands: Number of orbitals of the bubble.
    :param niw: Number of positive bosonic frequencies.
    :param niv: Number of positive fermionic frequencies.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The tail as a :class:`LocalFourPoint` without fermionic frequency in the half bosonic range, to be added
        to the box-summed bubble.
    """
    nu = MFHelper.vn(niv, float(beta))
    w = MFHelper.wn(niw, float(beta), return_only_positive=True)
    tail = np.where(w == 0, beta / 4, 0.0) - np.sum(1.0 / (nu * (nu - w[:, None])), axis=-1) / beta
    eye = np.eye(n_bands)
    return LocalFourPoint(np.einsum("ad,bc,w->abcdw", eye, eye, tail), SpinChannel.NONE, 1, 0, False)


def create_rpa_susceptibility(
    gchi0_q_sum: FourPoint, u_loc: LocalInteraction, v_nonloc: Interaction, channel: SpinChannel
) -> FourPoint:
    r"""
    Returns the RPA susceptibility :math:`\chi^{\mathrm{q}}_{r;\mathrm{RPA}} = \chi^{\mathrm{q}}_{0} (1 + (U_{r} +
    V^{\mathbf{q}}_{r}) \chi^{\mathrm{q}}_{0})^{-1}` of a channel from the frequency-summed bubble
    :math:`\chi^{\mathrm{q}}_{0} = \frac{1}{\beta^2} \sum_{\nu} \chi^{\mathrm{q}\nu}_{0}`. Where the bubble is
    invertible this is :math:`((\chi^{\mathrm{q}}_{0})^{-1} + U_{r} + V^{\mathbf{q}}_{r})^{-1}`, the form the
    shell-corrected physical susceptibility takes (see :func:`create_generalized_chi_q_with_shell_correction`). The
    bubble itself is never inverted: summed over all frequencies it is singular where a density is conserved, at
    :math:`\mathbf{q} = 0` and :math:`\omega \neq 0`.

    :param gchi0_q_sum: The frequency-summed bare bubble.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`.
    :param channel: Spin channel of the susceptibility, i.e. density or magnetic.
    :return: The RPA susceptibility as a :class:`FourPoint` in ``channel``.
    """
    u_r = v_nonloc.as_channel(channel) + u_loc.as_channel(channel)
    eye = np.eye(gchi0_q_sum.n_bands)
    dressing = (u_r @ gchi0_q_sum).add(LocalInteraction(np.einsum("ad,bc->abcd", eye, eye)), copy=False)
    return (gchi0_q_sum @ dressing.invert(False)).set_channel(channel)


def calculate_sigma_from_kernel(kernel: FourPoint, giwk: GreensFunction, my_full_q_list: np.ndarray) -> SelfEnergy:
    r"""
    Returns :math:`\Sigma^{\mathrm{k}}_{12} = -\frac{1}{2\beta n_{\mathbf{q}}} \sum_{\mathrm{q}} \sum_{ab}
    K^{\mathrm{q}\nu}_{a12b} G^{\mathrm{k}-\mathrm{q}}_{ab}`. For very large momentum grids,
    this function is the slowest part of the code because of its repeated loops. Batching the
    q-points or using numba could speed it up further.

    Currently unused: the pipeline always runs the column-distributed real-space contraction (see
    :func:`run_column_sde`), since this q-loop variant restores the full bosonic range on the kernel and therefore
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


def build_rspace_giwk_window(
    giwk: GreensFunction, node_comm, niv_window: int | None = None
) -> tuple[np.ndarray, MPI.Win | None]:
    r"""
    Builds the real-space Green's function :math:`F[G](R)` of the self-energy contraction once per node, in one
    shared-memory window in frequency-first layout ``[2 niv, n_R, o1, o2]``, so every frequency slab the
    contraction reads is contiguous. Every rank of the node copies its own contiguous share of the frequencies into
    the window and transforms :math:`G(\mathbf{k}) \to F[G](R)` there in place, so neither a cut nor a transformed
    copy of ``giwk`` is allocated; a single-rank node holds a private array. Each frequency transforms on its own, so
    every frequency slab holds the bits of :meth:`GreensFunction.fft` at any rank count. With ``niv_window`` the
    window holds the fermionic frequencies ``[-niv_window, niv_window)`` of ``giwk``, the bits of the window of the cut
    Green's function.

    :param giwk: The momentum-dependent :class:`GreensFunction` (cut to the self-energy box unless ``niv_window`` is
        given).
    :param node_comm: The node-local communicator.
    :param niv_window: Number of positive fermionic frequencies of the window, at most ``giwk.niv``, or ``None`` for
        all of them.
    :return: The tuple ``(g_r, win)``; release the window once no rank reads it anymore (``win`` is ``None`` on a
        single-rank node).
    :raises ValueError: If ``niv_window`` exceeds ``giwk.niv``.
    """
    niv_window = giwk.niv if niv_window is None else niv_window
    if niv_window > giwk.niv:
        raise ValueError(f"A window of {niv_window} positive frequencies exceeds the {giwk.niv} of the function.")
    nk, nb = giwk.nq, giwk.n_bands
    n_r, n_v = int(np.prod(nk)), 2 * niv_window
    g_r, win = mpi_utils.allocate_node_shared_array(node_comm, (n_v, n_r, nb, nb), giwk.mat.dtype)
    size, rank = (node_comm.Get_size(), node_comm.Get_rank()) if win is not None else (1, 0)
    first, last = rank * n_v // size, (rank + 1) * n_v // size
    if last > first:
        start = giwk.niv - niv_window
        g_r[first:last] = np.moveaxis(giwk.mat.reshape(n_r, nb, nb, -1)[..., start + first : start + last], -1, 0)
        # scipy.fft transforms the slab in place; numpy.fft would return a complex128 copy
        view = g_r[first:last].reshape(last - first, *nk, nb, nb)
        transformed = sp.fft.fftn(view, axes=(1, 2, 3), overwrite_x=True)
        if not np.shares_memory(transformed, view):
            view[...] = transformed
    node_comm.Barrier()
    return g_r, win


class ColumnSdeStream:
    r"""
    The column-distributed contraction of self-energy kernels with Green's functions,

    .. math:: \Sigma^{\mathrm{k}}_{12} = -\frac{1}{2\beta n_{\mathbf{q}}} \sum_{\mathrm{q}} \sum_{ab}
        K^{\mathrm{q}\nu}_{a12b} G^{\mathrm{k}-\mathrm{q}}_{ab},

    for :math:`\nu \geq 0` as a real-space product (convolution theorem), distributing the kernels' frequency columns
    over the ranks instead of their momenta. Output :math:`j` sums the products of its terms, each term pairing one
    kernel with one Green's function. A column is one bosonic frequency :math:`\omega` of one fermionic frequency
    :math:`\nu`; a negative :math:`\omega` is read from the stored positive one by time reversal, which for real
    hoppings is a plain conjugation in real space, :math:`K^{-\omega,\nu}(\mathbf{R}) = (K^{\omega,-\nu}(\mathbf{R}))^*`
    (in momentum space it maps :math:`\mathbf{q} \to -\mathbf{q}`). The tasks - blocks of consecutive columns of one
    :math:`\nu` - are laid out by :func:`~dgamore.memory_estimator.column_sde_schedule`. The kernels arrive in groups
    of stored bosonic frequencies (:meth:`add_group`), and per group every rank:

    1. receives the irreducible-BZ rows of the columns of its tasks in the group, of every kernel, in rounds of whole
       tasks of at most ``chunk_bytes`` (:func:`~dgamore.mpi_utils.transpose_columns`),
    2. expands its columns of each kernel to the full BZ with :meth:`FourPoint.map_to_full_bz`, in batches of
       consecutive columns of at most :data:`~dgamore.memory_estimator.SDE_UNFOLD_BATCH_BYTES` with their tasks' partial
       sums (one column when a column is larger, see :func:`~dgamore.memory_estimator.sde_unfold_batch`) into one
       buffer reused for every batch, Fourier transforms them in place over the momentum axes in the order z, y, x (a
       time-reversed column is conjugated afterwards) and contracts each column with the frequency slab at
       :math:`\nu - \omega` of the Green's function of every term of the kernel in bounded real-space row chunks,
       adding the product to the task's partial sum of the term's output (per batch the kernels in the given order,
       per kernel the columns in block order and their terms in the given order),
    3. scales a finished task's partial sum of every output and folds it into the output's running sum of
       :math:`\nu`, which the rank keeps across the groups.

    :meth:`finish` then reduces one output: the owner of :math:`\nu` (the rank holding its first block) folds the other
    ranks' sums in rank order, and rank 0 gathers the result. The bosonic sum is associated in the fixed blocks and in
    the fixed rank order of the schedule. With ``by_frequency`` every block holds one stored frequency and the groups
    pass the blocks in the schedule's order whatever their sizes, so an output depends on the box, the rank count and
    its own terms only, never on the groups, ``chunk_bytes`` or the other outputs beyond the rounding of the batched
    unfold, whose batch widths they set; without it the only group is the whole bosonic range.

    On a grid with a discovered point group of real orbital rotations
    (:meth:`~dgamore.brillouin_zone.KGrid.auto_point_group`) :meth:`finish` averages every result over the group
    before it is gathered (:func:`~dgamore.symmetry_reduction.project_onto_point_group`), so an output is the part of
    the contraction that obeys the group, to which the contraction of an invariant kernel with an invariant Green's
    function is equal. With invariant Green's functions that average is all the unfold contributes: a group passed
    as ``symmetric`` scatters its columns instead (:meth:`FourPoint.scatter_to_full_bz`: the irreducible rows
    weighted by their stars, no rotation), and a time-reversed column stays the conjugate at fixed
    :math:`\mathbf{R}`, which then sums the mirrored wedge. Both routes agree to rounding.

    With a ``node_comm`` the ranks of a node share the column work of all their tasks: every rank receives its rounds
    into its block of a node-shared window, and for every batch of every rank of the node, in node order, each rank
    of the node unfolds its own contiguous run of real-space rows into one node-shared buffer, the node transforms
    the buffer in two passes (z and y over slabs of x planes, then x over slabs of y planes, the lines split among the
    ranks) and each rank contracts and folds its rows. Partial and running sums are held per rank of the node and
    per frequency, split by rows; :meth:`finish` assembles them in a node-shared window before the reduction. Every
    momentum unfolds, every line transforms and every row contracts on its own, so the node shares the work with
    the bits of each rank doing its own (a single-rank node, or no ``node_comm``). Beyond the read-only kernels and
    windows a node holds one round of irreducible columns of its ranks' tasks of every kernel and one batch of
    full-BZ columns, and per output the running sums and results of its ranks; a rank holds its rows of the partial
    sums of a batch.

    A kernel whose last orbital pair never leaves an orbital block (an atom-diagonal interaction gives
    :math:`K_{a12b} = 0` unless :math:`2` and :math:`b` lie on one atom) is transposed, transformed and contracted on
    the components of those blocks alone (:meth:`_last_pair_layout`), wherever the columns reach the full BZ without
    an orbital rotation: on a grid without orbital rotations, and for scattered columns. The other components are zero
    everywhere, so only the order of the contraction's sum, and with it the rounding, differs from all components.
    """

    _TRANSPOSE_TAG, _PARTIAL_TAG = 1200, 1300  # base MPI tags of the column transpose and the partial sums

    def __init__(
        self,
        mpi_dist_irrk: MpiDistributor,
        n_out: int,
        dtype,
        chunk_bytes: int | None = None,
        by_frequency: bool = False,
        node_comm=None,
    ):
        r"""
        Lays out the tasks and the real-space rows this rank contracts of every column of its node.

        :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
        :param n_out: Number of outputs.
        :param dtype: Element type of the kernels, the Green's functions and the outputs.
        :param chunk_bytes: Byte budget of one round of transposed irreducible columns of all kernels (``None`` uses
            the floor).
        :param by_frequency: Whether every block holds one stored frequency, so the kernels can arrive in groups of
            any size (see :func:`~dgamore.memory_estimator.column_sde_schedule`).
        :param node_comm: The node-local communicator whose ranks share the column work of their tasks, or ``None``
            for a rank working alone.
        """
        niw, niv = config.box.niw_core, config.box.niv_core
        n_r = config.lattice.k_grid.nk_tot
        self._dist, self._dtype, self._by_frequency = mpi_dist_irrk, np.dtype(dtype), by_frequency
        self._chunk = SLICE_CHUNK_BYTES if chunk_bytes is None else chunk_bytes
        self._blocks, self._bounds, self._owner = memory_estimator.column_sde_schedule(
            niw, niv, mpi_dist_irrk.mpi_size, by_frequency
        )
        self._my_v = np.flatnonzero(self._owner == mpi_dist_irrk.my_rank)
        shared = node_comm is not None and node_comm.Get_size() > 1
        self._node = node_comm if shared else None
        # the ranks of the node in node order, and this rank's run of real-space rows of every column
        self._members = node_comm.allgather(mpi_dist_irrk.my_rank) if shared else [mpi_dist_irrk.my_rank]
        share, n_share = (node_comm.Get_rank(), node_comm.Get_size()) if shared else (0, 1)
        self._rows = slice(share * n_r // n_share, (share + 1) * n_r // n_share)
        # per output this rank's rows of every running sum of the node, keyed by (rank, fermionic frequency)
        self._sums = [{} for _ in range(n_out)]
        self._w_next = 0
        self._actions = config.lattice.k_grid.auto_point_group(real_space=True)

    def _barrier(self) -> None:
        """
        Waits for every rank of the node (a no-op for a rank working alone).

        :return: None.
        """
        if self._node is not None:
            self._node.Barrier()

    def _transform(self, out_buf: np.ndarray) -> None:
        r"""
        Fourier transforms the node-shared full-BZ columns over the momentum axes in the order z, y, x, in place: a rank
        working alone in one call, the ranks of a node in two passes (z and y over slabs of x planes, then x over slabs
        of y planes), each rank its share of the planes and of the trailing components, every line on its own.

        :param out_buf: The columns ``[n_R, ...]`` in the momentum order of the grid; transformed in place.
        :return: None.
        """
        nk = config.lattice.k_grid.nk
        grid = out_buf.reshape(*nk, -1)
        if self._node is None:
            transformed = sp.fft.fftn(grid, axes=(2, 1, 0), overwrite_x=True)
            if not np.shares_memory(transformed, grid):
                grid[...] = transformed
            return
        share, n_share = self._node.Get_rank(), self._node.Get_size()
        n_c = grid.shape[-1]
        for axis, axes in ((0, (2, 1)), (1, (0,))):
            parts = -(-n_share // nk[axis])
            units = [(p, c) for p in range(nk[axis]) for c in range(parts)]
            for p, c in units[share * len(units) // n_share : (share + 1) * len(units) // n_share]:
                index = [slice(None)] * 4
                index[axis], index[3] = slice(p, p + 1), slice(c * n_c // parts, (c + 1) * n_c // parts)
                view = grid[tuple(index)]
                transformed = sp.fft.fftn(view, axes=axes, overwrite_x=True)
                if not np.shares_memory(transformed, view):
                    view[...] = transformed
            self._barrier()

    def add_group(
        self,
        w_window: tuple[int, int],
        kernels: list[FourPoint],
        terms: list[tuple[int, int, np.ndarray, int]],
        symmetric: bool = False,
    ) -> None:
        r"""
        Contracts the columns of the group's tasks into the running sums (collective; every rank passes the same
        window, the same number of kernels and the same terms).

        :param w_window: The stored bosonic frequency indices ``(w0, w1)`` of the group; ``w0`` is the previous
            group's ``w1`` (0 for the first group).
        :param kernels: The rank-local irreducible-BZ kernels of the group ``[q, o1, o2, o3, o4, w1 - w0, v]`` (half
            bosonic, full fermionic range, compressed momentum axis); only read.
        :param terms: The terms as ``(kernel index, output index, g_r, giwk_niv)``: the frequency-first real-space
            Green's function (see :func:`build_rspace_giwk_window`) and its central fermionic-frequency index; only
            read.
        :param symmetric: Whether every kernel and Green's function of the group obeys the grid's point group, so
            the columns are scattered instead of unfolded; read only where the results are averaged over the group.
        :return: None.
        :raises ValueError: If the window does not continue the previous group inside the bosonic range (without
            ``by_frequency``: is not the whole range), or a kernel does not hold the window's frequencies.
        """
        rank, size = self._dist.my_rank, self._dist.mpi_size
        k_grid = config.lattice.k_grid
        n_r, nb = k_grid.nk_tot, config.sys.n_bands
        niw, niv = config.box.niw_core, config.box.niv_core
        w0, w1 = w_window
        whole = w0 == 0 and w1 == niw + 1
        if w0 != self._w_next or not w0 < w1 <= niw + 1 or not (self._by_frequency or whole):
            raise ValueError(f"The bosonic group ({w0}, {w1}) does not continue the groups at {self._w_next}.")
        if any(kernel.mat.shape[-2] != w1 - w0 for kernel in kernels):
            raise ValueError(f"A kernel does not hold the {w1 - w0} bosonic frequencies of the group ({w0}, {w1}).")
        self._w_next = w1
        # the unfold of an auto grid rotates the orbitals and needs every component of a column
        scattered = symmetric and self._actions is not None
        layouts = [self._last_pair_layout(kernel) if scattered or not k_grid.is_auto else None for kernel in kernels]
        shapes = [(nb,) * 4 if layout is None else (len(layout[1][0]), 1, 1, 1) for layout in layouts]
        blocks, bounds, n_b = self._blocks, self._bounds, len(self._blocks)
        in_group = np.array([w0 <= ws[0] < w1 for _, ws in blocks])
        tasks_of = [np.flatnonzero(in_group[np.arange(lo, hi) % n_b]) + lo for lo, hi in zip(bounds[:-1], bounds[1:])]

        def columns(tasks: np.ndarray) -> np.ndarray:
            """The (bosonic, fermionic) kernel index rows ``[2, n_cols]`` of tasks, in task and block order."""
            pairs = []
            for t in tasks:
                v, (negative, ws) = t // n_b, blocks[t % n_b]
                pairs += [(w - w0, niv - 1 - v if negative else niv + v) for w in ws]
            return np.array(pairs, dtype=int).reshape(-1, 2).T

        width = 1 if self._by_frequency else memory_estimator.SDE_W_BLOCK
        task_bytes = len(kernels) * k_grid.nk_irr * nb**4 * width * self._dtype.itemsize
        per_round = max(1, int(self._chunk // task_bytes))
        n_rounds = -(-max(tasks.size for tasks in tasks_of) // per_round)
        rows = max(1, memory_estimator.SDE_CONTRACTION_CHUNK_BYTES // (nb**4 * self._dtype.itemsize))
        path = ["einsum_path", (0, 1)]
        scale = -0.5 / config.sys.beta / n_r

        n_column = n_r * nb**4

        def batch_of(r: int) -> int:
            """The unfold batch of rank ``r``: as many of its columns of a round as fit with their partial sums."""
            n_mine = [
                sum(len(blocks[t % n_b][1]) for t in tasks_of[r][k * per_round : (k + 1) * per_round])
                for k in range(n_rounds)
            ]
            partial_bytes = len(self._sums) * n_r * nb**2 * self._dtype.itemsize
            return memory_estimator.sde_unfold_batch(
                n_column * self._dtype.itemsize, partial_bytes, max(n_mine, default=0)
            )

        batches = {r: batch_of(r) for r in self._members}
        mine_rows = self._rows
        n_rows = mine_rows.stop - mine_rows.start
        acc = np.empty((n_rows, nb, nb), dtype=self._dtype)
        # every batch of columns of the node is unfolded into this one buffer
        buffer, buffer_win = mpi_utils.allocate_node_shared_array(
            self._node, (max(batches.values()) * n_column,), self._dtype
        )

        def fold(r: int, t: int, partials: list[np.ndarray]) -> None:
            """Scales a finished task's partial sums (this rank's rows) and folds them into the running sums of its
            rank and frequency."""
            v = t // n_b
            for out, partial in enumerate(partials):
                partial *= scale
                sums = self._sums[out]
                if (r, v) not in sums:
                    sums[(r, v)] = partial
                else:
                    sums[(r, v)] += partial

        # deferred_collection batches the gc pass of the per-batch object releases into one collection at the end
        with deferred_collection():
            for k in range(n_rounds):
                round_of = [tasks[k * per_round : (k + 1) * per_round] for tasks in tasks_of]
                index_of = [columns(tasks) for tasks in round_of]
                # every rank receives its columns into its block of a node-shared window, read by the whole node
                received, windows = [], []
                for kernel, layout in zip(kernels, layouts):
                    middle = kernel.mat.shape[1:-2] if layout is None else (len(layout[1][0]),)
                    sizes = [k_grid.nk_irr * int(np.prod(middle)) * len(index_of[r][0]) for r in self._members]
                    flat, win = mpi_utils.allocate_node_shared_array(self._node, (sum(sizes),), self._dtype)
                    starts = np.concatenate(([0], np.cumsum(sizes)))
                    views = {
                        r: flat[starts[m] : starts[m + 1]].reshape(k_grid.nk_irr, *middle, len(index_of[r][0]))
                        for m, r in enumerate(self._members)
                    }
                    mpi_utils.transpose_columns(
                        kernel.mat,
                        self._dist,
                        index_of,
                        self._TRANSPOSE_TAG,
                        None if layout is None else layout[1],
                        out=views[rank],
                    )
                    received.append(views)
                    windows.append(win)
                self._barrier()
                for r in self._members:
                    # rank r's columns as (task, bosonic frequency), in the order of its received blocks
                    mine = [(t, w) for t in round_of[r] for w in blocks[t % n_b][1]]
                    partials = {}
                    for c0 in range(0, len(mine), batches[r]):
                        chunk = mine[c0 : c0 + batches[r]]
                        for t, _ in chunk:
                            if t not in partials:
                                partials[t] = [np.zeros((n_rows, nb, nb), dtype=self._dtype) for _ in self._sums]
                        for i, views in enumerate(received):
                            n_values = len(chunk) * n_r * int(np.prod(shapes[i]))
                            out_buf = buffer[:n_values].reshape(n_r, *shapes[i], len(chunk))
                            src = views[r][..., c0 : c0 + len(chunk)].reshape(-1, *shapes[i], len(chunk))
                            cols = FourPoint(src, SpinChannel.NONE, config.lattice.nk, 1, 0, False, True, True)
                            if scattered:
                                cols.scatter_to_full_bz(k_grid, out=out_buf, rows=mine_rows)
                            else:
                                cols.map_to_full_bz(k_grid, out=out_buf, rows=mine_rows)
                            self._barrier()
                            self._transform(out_buf)
                            for j, (t, w) in enumerate(chunk):
                                v, negative = t // n_b, blocks[t % n_b][0]
                                k_mat = out_buf[mine_rows, ..., j]
                                # time reversal of real hoppings conjugates at fixed R: K^{-w,v}(R) = conj(K^{w,-v}(R))
                                if negative:
                                    np.conjugate(k_mat, out=k_mat)
                                for _, out, g_r, giwk_niv in (term for term in terms if term[0] == i):
                                    g_slab = g_r[giwk_niv + (w if negative else -w) + v, mine_rows]
                                    if layouts[i] is not None:
                                        self._contract_blocks(g_slab, k_mat, layouts[i][0], acc, rows)
                                    else:
                                        for r0 in range(0, n_rows, rows):
                                            np.einsum(
                                                "Rad,Raijd->Rij",
                                                g_slab[r0 : r0 + rows],
                                                k_mat[r0 : r0 + rows],
                                                out=acc[r0 : r0 + rows],
                                                optimize=path,
                                            )
                                    np.add(partials[t][out], acc, out=partials[t][out])
                            # every rank of the node is done with the buffer before the next batch is unfolded into it
                            self._barrier()
                            cols.free()
                        # every task except one continuing into the next batch is complete
                        following = mine[c0 + len(chunk)][0] if c0 + len(chunk) < len(mine) else None
                        for t in [t for t in partials if t != following]:
                            fold(r, t, partials.pop(t))
                del received
                self._barrier()
                for win in windows:
                    if win is not None:
                        win.Free()
        buffer = None
        if buffer_win is not None:
            buffer_win.Free()

    def _last_pair_layout(self, kernel: FourPoint) -> tuple[list[np.ndarray], tuple[np.ndarray, ...]] | None:
        r"""
        Returns the orbital blocks of a kernel's last orbital pair (collective): the finest partition of the orbitals
        such that every component :math:`K_{a12b}` that is non-zero on any rank has :math:`2` and :math:`b` in one
        block. An atom-diagonal interaction leaves the atoms as blocks.

        :param kernel: The rank-local kernel ``[q, o1, o2, o3, o4, w, v]``; only read.
        :return: The tuple of the blocks (orbital index arrays) and the kernel indices ``(o1, o2, o3, o4)`` of their
            components, block after block and within a block in the order :math:`(1, 2, a, b)`; None when the
            orbitals form a single block.
        """
        pairs = kernel.nonzero_orbital_components().any(axis=(0, 1))
        if self._dist.mpi_size > 1:
            pairs = np.logical_or.reduce(self._dist.comm.allgather(pairs))
        n_blocks, labels = connected_components(pairs, directed=False)
        if n_blocks == 1:
            return None
        everyone = np.arange(len(pairs))
        orbital_blocks = [np.flatnonzero(labels == b) for b in range(n_blocks)]
        grids = [np.meshgrid(everyone, block, everyone, block, indexing="ij") for block in orbital_blocks]
        o2, o3, o1, o4 = (np.concatenate([grid[axis].ravel() for grid in grids]) for axis in range(4))
        return orbital_blocks, (o1, o2, o3, o4)

    @staticmethod
    def _contract_blocks(
        g_slab: np.ndarray, k_mat: np.ndarray, orbital_blocks: list[np.ndarray], acc: np.ndarray, rows: int
    ) -> None:
        r"""
        Writes the contraction :math:`\sum_{ab} K_{a12b} G_{ab}` of one column stored on the orbital blocks of its
        last pair (see :meth:`_last_pair_layout`) into ``acc``: per block and real-space point one matrix-vector
        product over :math:`(a, b)`, in chunks of real-space rows.

        :param g_slab: The Green's function slab ``[R, a, b]``; only read.
        :param k_mat: The transformed column, its components along the first axis after ``R``; only read.
        :param orbital_blocks: The orbital blocks of the layout.
        :param acc: The contraction ``[R, 1, 2]``; every entry is written.
        :param rows: Number of real-space rows of a chunk.
        :return: None.
        """
        n_r, nb = acc.shape[:2]
        k_mat = k_mat.reshape(n_r, -1)
        offset = 0
        for block in orbital_blocks:
            m = nb * len(block)
            for r0 in range(0, n_r, rows):
                kernel_rows = k_mat[r0 : r0 + rows, offset : offset + m * m].reshape(-1, m, m)  # [R, (1 2), (a b)]
                g_rows = g_slab[r0 : r0 + rows][:, :, block].reshape(-1, m)
                product = np.einsum("Rpc,Rc->Rp", kernel_rows, g_rows)
                acc[r0 : r0 + rows, :, block] = product.reshape(-1, nb, len(block))
            offset += m * m

    def finish(self, out: int) -> SelfEnergy | None:
        r"""
        Reduces output ``out`` after the last group (collective): the running sums of the node's ranks are assembled
        from the rows every rank of the node holds of them, the owner of :math:`\nu` folds the other ranks' sums in
        rank order and, on a grid with a point group, averages the result over it; rank 0 gathers the result, and the
        output's slabs are released.

        :param out: The output index.
        :return: The real-space self-energy :class:`SelfEnergy` of the output for :math:`\nu \geq 0` (compressed
            momentum axis, moments not fitted) on rank 0; ``None`` elsewhere. Transform it back with
            :meth:`SelfEnergy.ifft` and complete it with :meth:`SelfEnergy.to_full_niv_range`.
        :raises ValueError: If the groups have not reached the last bosonic frequency.
        """
        if self._w_next != config.box.niw_core + 1:
            raise ValueError(f"The bosonic groups end at {self._w_next}, before {config.box.niw_core + 1}.")
        comm, rank, size = self._dist.comm, self._dist.my_rank, self._dist.mpi_size
        n_r, nb = config.lattice.k_grid.nk_tot, config.sys.n_bands
        n_b, bounds, owner, my_v = len(self._blocks), self._bounds, self._owner, self._my_v
        sums = self._sums[out]
        self._sums[out] = None

        # every rank of the node holds the same keys, its rows of every sum; a rank's own sums are whole after this
        keys = sorted(sums)
        slabs, win = mpi_utils.allocate_node_shared_array(self._node, (len(keys), n_r, nb, nb), self._dtype)
        for i, key in enumerate(keys):
            slabs[i, self._rows] = sums.pop(key)
        self._barrier()
        own = {v: i for i, (r, v) in enumerate(keys) if r == rank}

        # every rank's run starts inside at most one foreign frequency; its owner folds the sums in rank order
        def ranks_of(v: int) -> list[int]:
            """The ranks holding blocks of frequency ``v`` (a nonempty overlap of task runs), in rank order."""
            return [r for r in range(size) if max(bounds[r], v * n_b) < min(bounds[r + 1], (v + 1) * n_b)]

        sends = [(int(owner[v]), slabs[i]) for v, i in own.items() if owner[v] != rank]
        sources = [r for v in my_v for r in ranks_of(v) if r != rank]
        received = iter(mpi_utils.exchange_blocks(comm, sends, sources, (n_r, nb, nb), self._dtype, self._PARTIAL_TAG))
        # the owned frequencies are consecutive keys of this rank, so their slabs are one block of the window
        result = slabs[own[my_v[0]] : own[my_v[0]] + my_v.size] if my_v.size else slabs[:0]
        for i, v in enumerate(my_v):
            for r in ranks_of(v):
                if r != rank:
                    result[i] += next(received)
            if self._actions is not None:
                result[i] = symmetry_reduction.project_onto_point_group(result[i], *self._actions)
        del sends

        full = MpiDistributor(ntasks=config.box.niv_core, comm=comm, sizes=np.bincount(owner, minlength=size)).gather(
            result
        )
        # the node's other ranks may still read their slabs of the window
        result = slabs = None
        self._barrier()
        if win is not None:
            win.Free()
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


def run_column_sde(
    kernel_irr: FourPoint,
    mpi_dist_irrk: MpiDistributor,
    g_r: np.ndarray,
    giwk_niv: int,
    chunk_bytes: int | None = None,
    symmetric: bool = False,
    node_comm=None,
) -> SelfEnergy | None:
    r"""
    Contracts the self-energy kernel with the Green's function,

    .. math:: \Sigma^{\mathrm{k}}_{12} = -\frac{1}{2\beta n_{\mathbf{q}}} \sum_{\mathrm{q}} \sum_{ab}
        K^{\mathrm{q}\nu}_{a12b} G^{\mathrm{k}-\mathrm{q}}_{ab},

    for :math:`\nu \geq 0`: one :class:`ColumnSdeStream` of one output, fed the whole bosonic range at once, its
    blocks the runs of :data:`~dgamore.memory_estimator.SDE_W_BLOCK` consecutive frequencies of
    :func:`~dgamore.memory_estimator.column_sde_schedule` (all positive, then all negative frequencies). The result
    depends on the box and the rank count only, never on ``chunk_bytes`` beyond the rounding of the batched unfold (see
    :class:`ColumnSdeStream`).

    :param kernel_irr: The rank-local irreducible-BZ kernel ``[q, o1, o2, o3, o4, w, v]`` (half bosonic, full
        fermionic range, compressed momentum axis); only read.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param g_r: The frequency-first real-space Green's function from :func:`build_rspace_giwk_window`.
    :param giwk_niv: The central fermionic-frequency index of ``g_r`` (``giwk.niv``).
    :param chunk_bytes: Byte budget of one round of transposed irreducible columns (``None`` uses the floor).
    :param symmetric: Whether the kernel and the Green's function obey the grid's point group, so the columns are
        scattered instead of unfolded (see :class:`ColumnSdeStream`).
    :param node_comm: The node-local communicator whose ranks share the column work of their tasks (see
        :class:`ColumnSdeStream`), or ``None`` for every rank working alone.
    :return: The real-space self-energy :class:`SelfEnergy` for :math:`\nu \geq 0` (compressed momentum axis, moments
        not fitted) on rank 0; ``None`` elsewhere. Transform it back with :meth:`SelfEnergy.ifft` and complete it
        with :meth:`SelfEnergy.to_full_niv_range`.
    """
    stream = ColumnSdeStream(mpi_dist_irrk, 1, kernel_irr.mat.dtype, chunk_bytes, node_comm=node_comm)
    stream.add_group((0, config.box.niw_core + 1), [kernel_irr], [(0, 0, g_r, giwk_niv)], symmetric)
    return stream.finish(0)


def sde_chunk_budget(comm: MPI.Comm, shared_node_comm) -> int:
    r"""
    Returns the chunk byte budget of the self-energy section, identical on every rank: each rank derives its node's
    budget (see :func:`dgamore.memory_estimator.dynamic_chunk_budget`) and the job-wide minimum is taken, so the
    node with the least memory per rank bounds the chunking everywhere. The round count of :func:`run_column_sde`
    follows from this budget and its column transpose is collective, so ranks with different budgets would walk
    different round counts and deadlock.

    :param comm: The MPI communicator.
    :param shared_node_comm: The node-local communicator (``None`` counts as one rank per node).
    :return: The chunk budget in bytes.
    """
    node_ranks = shared_node_comm.size if shared_node_comm is not None else 1
    chunk_bytes = memory_estimator.dynamic_chunk_budget(mpi_utils.job_memory_total(), node_ranks)
    return comm.allreduce(chunk_bytes, op=MPI.MIN) if comm.size > 1 else chunk_bytes


def load_node_shared_local_vertex(
    node_comm,
    path: str,
    channel: SpinChannel,
    transform=None,
    axes: tuple[int, ...] | None = None,
    w_window: tuple[int, int] | None = None,
) -> tuple:
    r"""
    Loads a local four-point vertex from file **once per node** into an MPI shared-memory window (see
    :func:`dgamore.mpi_utils.build_node_shared_array`): the node root reads the file (and applies ``transform``),
    every other rank maps the same physical buffer read-only. Without a node communicator each rank loads
    privately (the previous behavior). At production box sizes these local vertices are multi-GB and were held
    once **per rank** before - the largest replicated objects of the kernel section after ``giwk_full``.
    Without a transform the root streams the file straight into the window (see :func:`_stream_file_into_node_window`).
    With ``w_window`` only that window of the stored bosonic indices is loaded; without a transform it is streamed on a
    single-rank node too, so only the window is read from the file.

    :param node_comm: The node-local communicator (or ``None`` for a private per-rank load).
    :param path: Path to the ``.npy`` file.
    :param channel: Spin channel of the loaded vertex.
    :param transform: Optional callable applied to the loaded :class:`LocalFourPoint` on the node root before the
        array is placed in the window (e.g. an orbital permute + scale).
    :param axes: Optional axis order in which the array is stored (window or private copy); the returned vertex
        exposes the unchanged logical layout as a strided view of that storage, so a consumer whose contraction
        reads the stored order (e.g. :meth:`FourPoint.contract_first_pair_with_local_vertex`) works without a
        copy.
    :param w_window: The window ``(start, stop)`` of stored bosonic indices (half niw range) to load, or ``None`` for
        the whole vertex.
    :return: The tuple ``(vertex, win)``; free the window via :func:`~dgamore.mpi_utils.free_shared_window` once every
        rank is done reading (``win`` is ``None`` on the private path, then ``vertex.free()`` applies as before).
    """

    def _load() -> np.ndarray:
        obj = LocalFourPoint.load(path, channel)
        if w_window is not None:
            obj = obj.take_wn_slice(*w_window)
        if transform is not None:
            obj = transform(obj)
        # ascontiguousarray since a pure orbital-permute transform returns a strided view of the loaded array
        return np.ascontiguousarray(obj.mat if axes is None else obj.mat.transpose(axes))

    if node_comm is None:
        mat, win = _load(), None
    elif transform is None and (node_comm.Get_size() > 1 or w_window is not None):
        mat, win = _stream_file_into_node_window(node_comm, path, axes, w_window)
    else:
        mat, win = mpi_utils.build_node_shared_array(node_comm, _load)
    if axes is not None:
        mat = mat.transpose(np.argsort(axes))
    return LocalFourPoint(mat, channel, 1, 2, False, True), win


def _stream_file_into_node_window(
    node_comm, path: str, axes: tuple[int, ...] | None, w_window: tuple[int, int] | None = None
) -> tuple:
    """
    Copies the array of a ``.npy`` file into a node's shared-memory window without a private copy: the node root maps
    the file read-only and writes it, in the storage order ``axes``, straight into the window, one slab of the file's
    last three axes at a time; only the scatter into the reordered window is strided, since a copy in window order
    would read the file in runs of one fermionic axis. With ``w_window`` only that window of the file's bosonic axis,
    the third from the end, is copied, and only it is read. Collective over ``node_comm``.

    :param node_comm: The node-local communicator.
    :param path: Path to the ``.npy`` file.
    :param axes: The axis order the window stores, ``None`` for the file's own.
    :param w_window: The window ``(start, stop)`` of the file's third axis from the end, or ``None`` for all of it.
    :return: The tuple ``(array, win)`` of the window in the storage order.
    """
    root = node_comm.Get_rank() == 0
    source = np.load(path, mmap_mode="r") if root else None
    if root and w_window is not None:
        source = source[..., w_window[0] : w_window[1], :, :]
    shape, dtype = node_comm.bcast((source.transpose(axes or range(source.ndim)).shape, source.dtype) if root else None)
    shared, win = mpi_utils.allocate_node_shared_array(node_comm, shape, dtype)
    if root:
        logical = shared if axes is None else shared.transpose(np.argsort(axes))
        for index in np.ndindex(source.shape[:-3]):
            logical[index] = source[index]
    del source
    node_comm.Barrier()
    return shared, win


def build_giwk_full(comm: MPI.Comm, sigma: SelfEnergy, mu: float, ek: np.ndarray, beta: float) -> tuple:
    r"""
    Builds the full-grid Green's function :math:`G^{\mathrm{k}}_{12}`, deduplicated across the MPI ranks that share a
    physical node: ``giwk_full`` lives in one MPI shared-memory window per node, a single physical buffer per node
    instead of one private copy per rank (see :func:`dgamore.mpi_utils.allocate_node_shared_array`), and every rank of
    the node inverts the Dyson matrix of its own contiguous share of the momenta straight into the window. The inversion
    is per momentum and frequency, so the window holds the bits of a single-rank build at any rank count. A single-rank
    node builds a private array. The node topology is discovered at runtime via
    ``comm.Split_type(MPI.COMM_TYPE_SHARED)`` (nothing about the cluster is hard-coded). The result keeps the dispersion
    but no reference to :math:`\Sigma`, so none of its copies (the frequency cut, the real-space transforms) duplicates
    the full-BZ self-energy.

    :param comm: The MPI communicator.
    :param sigma: The self-energy :math:`\Sigma` entering the Dyson equation.
    :param mu: Chemical potential :math:`\mu`.
    :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})`.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The tuple ``(giwk_full, win, node_comm)``; ``win`` is ``None`` on a single-rank node
        and must otherwise be released with :func:`release_shared_giwk` once ``giwk_full`` has been cut to its private
        core box (the shared buffer is read-only and must not be freed while any rank still reads it).
    """
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    orbitals = config.sys.correlated_orbitals
    nk_tot, nb, n_orb = int(np.prod(ek.shape[:3])), sigma.n_bands, ek.shape[-1]
    shape = (*ek.shape[:3], nb, nb, 2 * sigma.niv)
    giwk_mat, win = mpi_utils.allocate_node_shared_array(node_comm, shape, sigma.mat.dtype)
    size, rank = (node_comm.Get_size(), node_comm.Get_rank()) if win is not None else (1, 0)
    mine = slice(rank * nk_tot // size, (rank + 1) * nk_tot // size)
    n_my = mine.stop - mine.start
    if n_my:
        # a momentum-local sigma (one row) enters every rank's share whole and broadcasts against its dispersion
        rows = sigma.mat.reshape(-1, *sigma.mat.shape[-3:])
        rows = rows if rows.shape[0] == 1 else rows[mine]
        sigma_my = SelfEnergy(rows, (rows.shape[0], 1, 1), sigma.full_niv_range, True, calc_smom=False, beta=beta)
        GreensFunction.get_g_full(
            sigma_my,
            mu,
            ek.reshape(nk_tot, n_orb, n_orb)[mine].reshape(n_my, 1, 1, n_orb, n_orb),
            beta,
            orbitals=orbitals,
            out=giwk_mat.reshape(nk_tot, nb, nb, -1)[mine].reshape(n_my, 1, 1, nb, nb, -1),
        )
    if win is not None:
        node_comm.Barrier()
    giwk_full = GreensFunction(
        giwk_mat, None, ek, sigma.full_niv_range, False, False, nk=ek.shape[:3], beta=beta, mu=mu, orbitals=orbitals
    )
    return giwk_full, win, node_comm


def release_shared_giwk(win, node_comm) -> None:
    r"""
    Releases the shared-memory window and node communicator allocated by :func:`build_giwk_full`, once all node ranks
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


def cut_and_reshare_giwk(giwk_full: GreensFunction, win, node_comm, niv: int) -> tuple:
    r"""
    Cuts ``giwk_full`` to the :math:`[-niv, niv)` core box. When ``giwk_full`` is node-shared (``node_comm`` is not
    ``None``), the node root cuts the shared full-niv Green's function into a **new, smaller per-node shared window**
    and every rank maps that; the caller then frees the old (large) full-niv window via
    :func:`~dgamore.mpi_utils.free_shared_window`. This keeps the deduplicated ``giwk_full`` at one copy per node
    through the whole self-energy step, not just the bubble. Without sharing it is a plain per-rank cut.

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
        orbitals=giwk_full._orbitals,
    )
    return giwk_cut, cut_win


def irreducible_momentum_distributor(comm: MPI.Comm) -> MpiDistributor:
    """
    Returns the distributor of the irreducible momenta, its larger counts spread evenly over the ranks (see
    :func:`~dgamore.mpi_utils.even_bounds`). The self-consistency loop and the Eliashberg step both build it here, so
    the Eliashberg step reads the per-rank files of the loop with the split that wrote them.

    :param comm: The MPI communicator.
    :return: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    """
    return MpiDistributor.create_distributor(config.lattice.k_grid.nk_irr, comm, spread=True)


def scatter_irreducible_interaction(v_nonloc: Interaction | None, mpi_dist_irrk: MpiDistributor) -> Interaction:
    r"""
    Returns this rank's irreducible momenta of the non-local interaction :math:`V^{\mathbf{q}}`, which only rank 0
    holds on the full q-grid: rank 0 reduces it to the irreducible q-points in the order of their index list
    (:meth:`~dgamore.n_point_base.IAmNonLocal.reduce_q`) and scatters the rows, so every rank receives its momenta in
    the distributor's order, the order of every other irreducible object, and no other rank ever holds the full grid.
    Collective over the communicator of ``mpi_dist_irrk``.

    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}` on the full q-grid; only read on rank 0.
    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :return: The interaction on this rank's irreducible q-points (compressed momentum axis).
    """
    rows = channel = None
    if mpi_dist_irrk.my_rank == 0:
        rows = v_nonloc.reduce_q(config.lattice.k_grid.get_irrq_list()).mat
        channel = v_nonloc.channel
    channel = mpi_dist_irrk.bcast(channel)
    return Interaction(mpi_dist_irrk.scatter(rows), channel, config.lattice.k_grid.nk, has_compressed_q_dimension=True)


def occupation_self_energy(
    sigma_new: SelfEnergy,
    sigma_dmft_full: SelfEnergy,
    mpi_dist_fullbz: MpiDistributor,
    rows: np.ndarray | None = None,
) -> SelfEnergy:
    r"""
    Returns this rank's momentum slice of the self-energy the occupation is evaluated with: ``sigma_new`` on its own
    frequency box and the DMFT self-energy beyond it, shifted by the momentum-dependent shell offset ``sigma_new``
    carries (see :meth:`SelfEnergy.shell_offset_from`), with the high-frequency moments of the whole concatenation
    fitted on rank 0 and broadcast.

    :param sigma_new: The mixed :class:`SelfEnergy` (full BZ, identical on every rank; compressed in place).
    :param sigma_dmft_full: The DMFT :class:`SelfEnergy` supplying the shell frequencies (momentum-local).
    :param mpi_dist_fullbz: MPI distributor over the full BZ q-points.
    :param rows: The flat grid indices of this rank's momenta, or ``None`` for its slice of ``mpi_dist_fullbz``.
    :return: The concatenated :class:`SelfEnergy` of this rank's momenta (compressed momentum axis, moments set).
    """
    rows = mpi_dist_fullbz.my_slice if rows is None else rows
    mat = sigma_new.compress_q_dimension().mat[rows]
    sigma_slice = SelfEnergy(
        mat, (len(mat), 1, 1), has_compressed_q_dimension=True, calc_smom=False, beta=config.sys.beta
    )
    shell_offset = sigma_new.shell_offset_from(sigma_dmft_full)
    sigma_occ = sigma_slice.concatenate_self_energies(sigma_dmft_full, shell_offset=shell_offset[rows])

    # the full-box moments come from the k-mean fit window of the replicated sigma_new, so rank 0 fits them once
    # (the fit window is a full-BZ array) and broadcasts; the fit equals the momentum-resolved concatenation's fit
    moments = (
        sigma_new.fit_smom_concatenated(sigma_dmft_full, shell_offset=shell_offset)
        if mpi_dist_fullbz.my_rank == 0
        else None
    )
    sigma_occ._smom0, sigma_occ._smom1 = mpi_dist_fullbz.bcast(moments)
    return sigma_occ


def _slice_occupation_and_energies(
    sigma_occ: SelfEnergy, mu: float, rows: slice | np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, float, float]:
    r"""
    Returns the k-resolved occupation and the kinetic and potential energies of this rank's momentum slice, which
    must hold at least one momentum. The real-or-complex arithmetic of the occupation is decided on the whole
    dispersion, so every rank's slice takes the same.

    :param sigma_occ: This rank's concatenated self-energy slice (see :func:`occupation_self_energy`).
    :param mu: Chemical potential :math:`\mu`.
    :param rows: The flat grid indices of this rank's momenta.
    :param weights: A weight per momentum of the slice in its energy sums (the size of the star of an irreducible
        momentum), or ``None`` for the plain sums.
    :return: The tuple ``(occ_k_slice, ekin, epot)`` of this rank's k-resolved occupation ``[k, 1, 1, o1, o2]`` on
        every orbital of H(k) and its kinetic and potential energy sums divided by its number of momenta.
    """
    ek = config.lattice.hamiltonian.get_ek()
    n_orbitals = ek.shape[-1]
    ek_slice = ek.reshape(-1, n_orbitals, n_orbitals)[rows].reshape(-1, 1, 1, n_orbitals, n_orbitals)
    giwk_occ = GreensFunction.get_g_full(
        sigma_occ, mu, ek_slice, config.sys.beta, orbitals=config.sys.correlated_orbitals
    )
    _, _, occ_k_slice = giwk_occ.get_fill_nonlocal(real_dispersion=np.isrealobj(np.real_if_close(ek)))
    ekin, epot = giwk_occ.get_ekin(weights), giwk_occ.get_epot(weights=weights)
    giwk_occ.free()
    return occ_k_slice, ekin, epot


def update_occ_and_energies_distributed(
    sigma_new: SelfEnergy,
    sigma_dmft_full: SelfEnergy,
    mpi_dist_fullbz: MpiDistributor,
    mu: float,
    mpi_dist_irrk: MpiDistributor | None = None,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    r"""
    Computes the occupation and the kinetic and potential energies of the mixed self-energy on the DMFT frequency
    box, distributed over the full-BZ momenta: every rank concatenates and Dyson-inverts only its own momentum
    slice, evaluates the occupation and energy sums there, and the results are recombined (the former evaluation
    built the whole DMFT-box Green's function and its asymptotic tail sums on rank 0 while every other rank idled).
    The self-energy moments are fitted from the momentum-averaged concatenated self-energy, allreduced first so
    they match the full-box fit on every rank; the k-resolved occupation is allgathered and the k-summed scalars
    are recombined with each rank's momentum count as weight. The real-or-complex arithmetic of the occupation is
    decided on the whole dispersion, so a rank whose momenta all have a real :math:`H(\mathbf{k})` keeps the
    complex arithmetic of a complex dispersion and the occupation does not depend on the rank count. The
    momentum-dependent constant the mixed self-energy carries in its shell (see
    :meth:`SelfEnergy.shell_offset_from`) is carried into the DMFT-box extension and its moment fit.

    With ``mpi_dist_irrk`` the ranks evaluate the irreducible momenta only, for a dispersion and a self-energy that obey
    the grid's point group (see :func:`~dgamore.nonlocal_sde._lattice_sum_momenta`): the energy sums weight every
    irreducible momentum with the size of its star, and the k-resolved occupation is unfolded from the irreducible
    momenta (see :func:`assemble_occupation`).

    :param sigma_new: The mixed :class:`SelfEnergy` (full BZ, compressed momenta, identical on every rank).
    :param sigma_dmft_full: The DMFT :class:`SelfEnergy` supplying the shell frequencies (momentum-local).
    :param mpi_dist_fullbz: MPI distributor over the full BZ q-points.
    :param mu: Chemical potential :math:`\mu`.
    :param mpi_dist_irrk: MPI distributor over the irreducible momenta, or ``None`` to evaluate every momentum.
    :return: The tuple ``(n, occ, occ_k, ekin, epot)``: total filling over every orbital, k-averaged ``[o1, o2]``
        and k-resolved ``[kx, ky, kz, o1, o2]`` occupation of the correlated orbitals, and the kinetic and potential
        energies per site.
    """
    k_grid = config.lattice.k_grid
    n_orbitals = config.lattice.hamiltonian.get_ek().shape[-1]
    dist, rows, weights = mpi_dist_fullbz, mpi_dist_fullbz.my_slice, None
    if mpi_dist_irrk is not None:
        dist = mpi_dist_irrk
        rows, weights = k_grid.irrk_ind[dist.my_slice], k_grid.irrk_count[dist.my_slice]
    n_my = dist.my_size

    sigma_occ = occupation_self_energy(sigma_new, sigma_dmft_full, mpi_dist_fullbz, rows)

    # a rank without momenta (more ranks than momenta) adds nothing to the reductions below
    occ_k_slice, ekin, epot = np.zeros((0, 1, 1, n_orbitals, n_orbitals)), 0.0, 0.0
    if n_my:
        occ_k_slice, ekin, epot = _slice_occupation_and_energies(sigma_occ, mu, rows, weights)
    sigma_occ.free()

    unfold = mpi_dist_irrk is not None
    n_el, occ, occ_k = assemble_occupation(occ_k_slice, dist, config.sys.correlated_orbitals, unfold)
    scalars = np.array([ekin, epot]) * n_my
    if dist.mpi_size > 1:
        scalars = dist.comm.allreduce(scalars)
    ekin, epot = scalars / k_grid.nk_tot
    return n_el, occ, occ_k, float(ekin), float(epot)


def assemble_occupation(
    occ_k_slice: np.ndarray, mpi_dist: MpiDistributor, orbitals: np.ndarray | None = None, unfold: bool = False
) -> tuple[float, np.ndarray, np.ndarray]:
    r"""
    Assembles the k-resolved occupation from the momentum slices the ranks evaluated and derives the k-averaged
    occupation and the filling from the assembled array, so both equal a full-grid evaluation bit for bit. The
    assembly is the chunked Allreduce of zero-padded slices: each momentum is contributed by exactly one rank, so
    the sum is exact and no point-to-point matching is involved. The dtype is pinned to complex128 because the
    fill's dtype is value-dependent (a real-matrix eigendecomposition returns float when a slice's eigenvalues are
    all real) and ranks entering a collective with different element types abort with truncated messages.

    With ``unfold`` the slices are those of the irreducible momenta, and the occupation of the whole grid is their
    unfold under the grid's point group, every momentum the image of its irreducible one (see
    :func:`~dgamore.symmetry_reduction.project_onto_point_group`).

    :param occ_k_slice: This rank's k-resolved occupation ``[kx, ky, kz, o1, o2]`` on its momentum slice.
    :param mpi_dist: MPI distributor over the full BZ q-points, with ``unfold`` over the irreducible ones.
    :param orbitals: Indices of the orbitals of the dispersion the self-energy acts on (the correlated orbitals); None
        when it acts on every orbital.
    :param unfold: Whether the slices hold the irreducible momenta.
    :return: A tuple of the total filling :math:`n` over every orbital, and the k-averaged ``[o1, o2]`` and the
        k-resolved ``[kx, ky, kz, o1, o2]`` occupation of the correlated orbitals on the full grid.
    """
    k_grid, n_bands, n_my = config.lattice.k_grid, occ_k_slice.shape[-1], mpi_dist.my_size
    occ_k = np.zeros((k_grid.nk_irr if unfold else k_grid.nk_tot, n_bands, n_bands), dtype=np.complex128)
    occ_k[mpi_dist.my_slice] = occ_k_slice.reshape(n_my, n_bands, n_bands)
    if mpi_dist.mpi_size > 1:
        occ_k = mpi_dist.allreduce(occ_k)
    if unfold:
        occ_k *= k_grid.irrk_count[:, None, None]
        actions = k_grid.auto_point_group()
        occ_k = symmetry_reduction.project_onto_point_group(occ_k, *actions, rows=k_grid.irrk_ind)
    occ_k = occ_k.reshape(*k_grid.nk, n_bands, n_bands)
    occ = np.mean(occ_k, axis=(0, 1, 2))
    occ.real[np.abs(occ) < 1e-12] = 0.0
    return 2.0 * np.trace(occ).real, orbital_block(occ, orbitals), orbital_block(occ_k, orbitals)


def has_time_reversal(ek: np.ndarray) -> bool:
    r"""
    Returns whether the dispersion obeys the time reversal of real hoppings,
    :math:`\varepsilon_{12}(-\mathbf{k}) = \varepsilon_{21}(\mathbf{k})`, to 1e-4 of its largest element: above the
    six-digit precision of wannier90 hopping files, below any physical complex hopping.

    :param ek: The band dispersion ``[kx, ky, kz, o1, o2]``.
    :return: Whether the relation holds.
    """
    flipped = np.roll(np.flip(ek, axis=(0, 1, 2)), 1, axis=(0, 1, 2))
    return bool(np.allclose(flipped, np.swapaxes(ek, -1, -2), rtol=0.0, atol=1e-4 * np.abs(ek).max()))


def orbital_involution(ek: np.ndarray, orbitals: np.ndarray | None = None) -> np.ndarray | None:
    r"""
    Returns the orbital permutation :math:`\pi`, its own inverse and not the identity, under which the dispersion is
    transposed, :math:`\varepsilon_{21}(\mathbf{k}) = \varepsilon_{\pi(1)\pi(2)}(\mathbf{k})` at every momentum
    to 1e-4 of its largest element (the tolerance of :func:`has_time_reversal`), or None. Real hoppings and an
    inversion that permutes the orbitals, such as the layer swap of a bilayer, give it, and the ladder's
    Bethe-Salpeter slices inherit it on their orbital pairs to the precision of the self-energy; whether a slice
    uses it is decided per slice (see
    :meth:`~dgamore.four_point.FourPoint.invert_with_local_kernel_and_sum_over_last_vn`). The candidate pairs every
    orbital with the one other orbital whose diagonal dispersion matches its own; an orbital with several such
    partners gives None. With ``orbitals`` the correlated Green's function inherits the permutation only if it maps
    the correlated orbitals onto themselves, and it is returned in their own numbering.

    :param ek: The band dispersion ``[kx, ky, kz, o1, o2]``.
    :param orbitals: Indices of the orbitals of the dispersion the self-energy acts on (the correlated orbitals); None
        when it acts on every orbital.
    :return: The permutation as an index array, or None.
    """
    n_bands = ek.shape[-1]
    atol = 1e-4 * np.abs(ek).max()
    diagonal = np.diagonal(ek, axis1=-2, axis2=-1).reshape(-1, n_bands)
    pi = np.arange(n_bands)
    for i in range(n_bands):
        matches = [np.allclose(diagonal[:, j], diagonal[:, i], rtol=0.0, atol=atol) for j in range(n_bands)]
        partners = [j for j in range(n_bands) if j != i and matches[j]]
        if len(partners) > 1:
            return None
        pi[i] = partners[0] if partners else i
    transposed = np.allclose(np.swapaxes(ek, -1, -2), ek[..., pi, :][..., pi], rtol=0.0, atol=atol)
    if not transposed or not np.any(pi != np.arange(n_bands)):
        return None
    if orbitals is None:
        return pi
    # ponytail: pairs on the full H(k), so degenerate p levels hide an involution of the d block (a speedup lost, no
    # error); pair on the correlated moments of symmetry_reduction if a d-p cell needs it
    mapped = pi[orbitals]
    if not np.array_equal(np.sort(mapped), orbitals):
        return None
    restricted = np.searchsorted(orbitals, mapped)
    return restricted if np.any(restricted != np.arange(len(orbitals))) else None

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Non-local ladder DGA step - the self-consistency driver. Wraps the numerical kernels of :mod:`dgamore.sde_kernels`
(bubble, auxiliary susceptibility, three-leg vertices, physical susceptibility, Schwinger-Dyson equation) in the
self-consistency loop with chemical-potential adjustment, self-energy mixing (see :mod:`dgamore.mixing`) and the
convergence-stabilization scaffolds (susceptibility restriction, per-iteration lambda correction, lambda
annealing). Equation numbers refer to the author's master's thesis (Chapters 3 & 4).
"""

import glob
import os
from collections.abc import Callable

import mpi4py.MPI as MPI
import numpy as np
from scipy import optimize as opt

import dgamore.config as config
import dgamore.mpi_utils as mpi_utils
import dgamore.output_files as output_files
from dgamore.brillouin_zone import KGrid
from dgamore.bubble_gen import BubbleGenerator
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction, update_mu
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.local_four_point import LocalFourPoint
from dgamore.lambda_ops import StabilizationState, select_and_apply_lambda_correction
from dgamore.mixing import _mixing_history_cap, apply_mixing_strategy, read_last_n_sigmas_from_files
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel
from dgamore.sde_kernels import (
    _run_fft_sde_pass,
    calculate_kernel_r_q,
    calculate_sigma_dc_kernel,
    create_auxiliary_chi_r_q_sum_v1,
    create_auxiliary_chi_r_q_sum_v3,
    create_generalized_chi_q_with_shell_correction,
    create_vrg_r_q,
    create_vrg_r_q_right,
    get_hartree_fock,
    select_sigma_fft_device,
)
from dgamore.self_energy import SelfEnergy


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


def perform_ornstein_zernike_fit(chi_phys_q_r: FourPoint) -> None:
    r"""
    Fits the static (:math:`\omega = 0`) physical susceptibility to an Ornstein-Zernike form
    :math:`\chi(q) = A / (\xi^{-2} + (q - q_0)^2)` around the antiferromagnetic wave vector
    :math:`q_0 = (\pi, \pi, 0)`, per orbital combination, and writes the amplitude :math:`A` and correlation length
    :math:`\xi` to ``oz_coeff.txt``. Non-converging fits are flagged with ``[-1, -1]``.

    :param chi_phys_q_r: The momentum-dependent physical susceptibility :math:`\chi^{q}_{r}` (irreducible BZ).
    :return: None.
    """

    def oz_spin_w0(q_grid: KGrid, a: float, xi: float) -> np.ndarray:
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

    def fit_oz_spin(q_grid: KGrid, mat: np.ndarray) -> np.ndarray:
        """
        Least-squares fits the Ornstein-Zernike model to one orbital slice of the susceptibility.

        :param q_grid: The :class:`KGrid` providing the momentum coordinates.
        :param mat: The flattened susceptibility slice to fit.
        :return: The fitted ``(A, xi)`` coefficients.
        """
        initial_guess = (mat.max(), 2.0)
        return opt.curve_fit(oz_spin_w0, q_grid, mat, p0=initial_guess)[0]

    chi = chi_phys_q_r.copy()
    chi_mat = chi.map_to_full_bz(config.lattice.k_grid).to_half_niw_range().take_first_wn().mat.real
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
    gchi0_q_core_inv: FourPoint, u_loc: LocalInteraction, v_nonloc: Interaction, mpi_dist_irrk: MpiDistributor
) -> None:
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
            chi_rpa_q_r.save(name=output_files.chi_rpa_q_name(channel), output_dir=config.output.output_path)

        chi_rpa_q_r.free()
        config.logger.info(f"Calculated RPA susceptibility ({channel.value}).")


def calculate_sigma_kernel_r_q(
    gamma_r: LocalFourPoint,
    gchi0_q_inv: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    mpi_dist_irrq: MpiDistributor,
    stab: "StabilizationState | None" = None,
) -> FourPoint:
    r"""
    Returns the kernel for the self-energy calculation in a specific spin channel. Calculates the auxiliary
    susceptibility, the three-leg vertex and the physical susceptibility with shell correction. Also performs a
    :math:`\lambda`-correction on the physical susceptibility if specified (dispatched by the band
    count). Saves the physical susceptibility (and, if Eliashberg is enabled, the intermediate vertices) to file.

    :param gamma_r: The local irreducible vertex :math:`\Gamma_{r}`.
    :param gchi0_q_inv: The inverse bare bubble :math:`(\chi_0^q)^{-1}` (core box).
    :param gchi0_q_full_sum: The frequency-summed bare bubble over the full box.
    :param gchi0_q_core_sum: The frequency-summed bare bubble over the core box.
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{q}`.
    :param mpi_dist_irrq: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param stab: The loop-owned :class:`~dgamore.lambda_ops.StabilizationState` (its annealer mass, restriction
        and lambda-correction flags are applied to the physical susceptibility), or ``None`` when no
        stabilization scaffold is active.
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

    if config.eliashberg.perform_eliashberg:
        vrg_q_r_right = create_vrg_r_q_right(gchi_aux_q_r_sum, gchi0_q_inv)
        vrg_q_r_right.save(
            name=output_files.vrg_q_right_rank_name(vrg_q_r_right.channel, mpi_dist_irrq.comm.rank),
            output_dir=config.output.eliashberg_path,
        )
        vrg_q_r_right.free()

    vrg_q_r = create_vrg_r_q(gchi_aux_q_r_sum, gchi0_q_inv)

    logger.info(f"Non-local three-leg vertex gamma^wv ({vrg_q_r.channel.value}) done.")
    logger.log_memory_usage(f"Three-leg vertex ({vrg_q_r.channel.value})", vrg_q_r, mpi_dist_irrq.comm.size)

    if config.eliashberg.perform_eliashberg:
        vrg_q_r.save(
            name=output_files.vrg_q_rank_name(vrg_q_r.channel, mpi_dist_irrq.comm.rank),
            output_dir=config.output.eliashberg_path,
        )

    chi_phys_q_r = gchi_aux_q_r_sum.sum_over_all_vn(config.sys.beta)
    gchi_aux_q_r_sum.free()

    chi_phys_q_r = create_generalized_chi_q_with_shell_correction(
        chi_phys_q_r, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
    )

    logger.info(f"Updated non-local susceptibility chi^q ({chi_phys_q_r.channel.value}) with asymptotic correction.")

    if stab is not None and stab.annealer is not None:
        chi_phys_q_r = stab.annealer.apply(chi_phys_q_r, mpi_dist_irrq)

    if stab is not None and stab.chi_phys_restriction_active:
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
        chi_phys_q_r = select_and_apply_lambda_correction(
            chi_phys_q_r, stab is not None and stab.lambda_correction_active
        )
        chi_phys_q_r.save(name=output_files.chi_phys_q_name(chi_phys_q_r.channel), output_dir=config.output.output_path)

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

    if config.eliashberg.perform_eliashberg:
        chi_phys_q_r.save(
            name=output_files.chi_phys_q_rank_name(chi_phys_q_r.channel, mpi_dist_irrq.comm.rank),
            output_dir=config.output.eliashberg_path,
        )

    return calculate_kernel_r_q(vrg_q_r, chi_phys_q_r, v_nonloc, u_loc)


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
        glob_pattern = output_files.SIGMA_INTERPOLATED_GLOB
        iteration_regex = output_files.SIGMA_INTERPOLATED_REGEX
    else:
        glob_pattern = output_files.SIGMA_ITERATION_GLOB
        iteration_regex = output_files.SIGMA_ITERATION_REGEX

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

    previous_mu = float(
        np.load(os.path.join(config.self_consistency.previous_sc_path, output_files.MU_HISTORY_FILENAME))[-1]
    )
    config.sys.mu = previous_mu
    return [previous_mu]


def _load_node_shared_local_vertex(
    node_comm: "MPI.Comm | None",
    path: str,
    channel: SpinChannel,
    transform: "Callable[[LocalFourPoint], LocalFourPoint] | None" = None,
) -> "tuple[LocalFourPoint, MPI.Win | None]":
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
    :return: The tuple ``(vertex, win)``; free the window via :func:`~dgamore.mpi_utils.free_shared_window` once every rank is done
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


def _build_giwk_full(
    comm: MPI.Comm, sigma: SelfEnergy, mu: float, ek: np.ndarray, beta: float
) -> "tuple[GreensFunction, MPI.Win | None, MPI.Comm | None]":
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


def _release_shared_giwk(win: "MPI.Win | None", node_comm: "MPI.Comm | None") -> None:
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


def _cut_and_reshare_giwk(
    giwk_full: GreensFunction, win: "MPI.Win | None", node_comm: "MPI.Comm | None", niv: int
) -> "tuple[GreensFunction, MPI.Win | None]":
    r"""
    Cuts ``giwk_full`` to the :math:`[-niv, niv)` core box. When ``giwk_full`` is node-shared (``node_comm`` is not
    ``None``), the node root cuts the shared full-niv Green's function into a **new, smaller per-node shared window**
    and every rank maps that; the caller then frees the old (large) full-niv window via :func:`~dgamore.mpi_utils.free_shared_window`.
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
    stab: "StabilizationState | None" = None,
) -> SelfEnergy:
    r"""
    Returns the raw (un-mixed) DGA self-energy proposal :math:`S(\Sigma_{\mathrm{in}})` at chemical potential
    :math:`\mu`: Hartree/Fock, the Dyson Green's function, the bubble, the double-counting, density and magnetic
    kernels, and the FFT Schwinger-Dyson contraction, finished with the noise-removal term and the DMFT tail.

    Single source of truth for the proposal map: it is called once per self-consistency iteration by
    :func:`calculate_self_energy_q`. The local irreducible vertex is frozen, so every
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
    :param stab: The loop-owned :class:`~dgamore.lambda_ops.StabilizationState` threaded into the kernel step, or
        ``None`` when no stabilization scaffold is active.
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
            config.lattice.k_grid,
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

    logger.log_memory_usage("Gchi0_q_full", gchi0_q, comm.size)
    # Cut giwk to the core box for the self-energy step. When node-shared, the node root cuts into a new, smaller
    # per-node window and the large full-niv window is freed; the cut giwk stays one copy per node through the SDE.
    old_giwk_win = giwk_win
    giwk_full, giwk_win = _cut_and_reshare_giwk(
        giwk_full, giwk_win, shared_node_comm, config.box.niv_core + config.box.niw_core
    )
    mpi_utils.free_shared_window(old_giwk_win, shared_node_comm)

    # the local vertices are identical on every rank, so they are loaded once per node into shared windows
    f_dc_loc, f_dc_win = _load_node_shared_local_vertex(
        shared_node_comm,
        output_files.npy_path(config.output.output_path, output_files.local_vertex_name("f", SpinChannel.MAGN)),
        SpinChannel.NONE,
        transform=lambda obj: obj.permute_orbitals("abcd->cbad", copy=False).scale(2.0),
    )
    kernel = calculate_sigma_dc_kernel(f_dc_loc, gchi0_q, u_loc).scale(-1.0)
    f_dc_loc.mat = None
    if f_dc_win is None:
        f_dc_loc.free()
    mpi_utils.free_shared_window(f_dc_win, shared_node_comm)
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
        calculate_and_save_chi_q_r_rpa(gchi0_q_core_inv, u_loc, v_nonloc, mpi_dist_irrk)

    if config.eliashberg.perform_eliashberg:
        gchi0_q_core_inv.save(
            name=output_files.gchi0_q_inv_rank_name(comm.rank), output_dir=config.output.eliashberg_path
        )

    gamma_dens, gamma_dens_win = _load_node_shared_local_vertex(
        shared_node_comm,
        output_files.npy_path(config.output.output_path, output_files.local_vertex_name("gamma", SpinChannel.DENS)),
        SpinChannel.DENS,
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
            stab,
        ),
        copy=False,
    )
    gamma_dens.mat = None
    if gamma_dens_win is None:
        gamma_dens.free()
    mpi_utils.free_shared_window(gamma_dens_win, shared_node_comm)
    mpi_dist_irrk.barrier()
    logger.info("Calculated kernel for density channel.")

    gamma_magn, gamma_magn_win = _load_node_shared_local_vertex(
        shared_node_comm,
        output_files.npy_path(config.output.output_path, output_files.local_vertex_name("gamma", SpinChannel.MAGN)),
        SpinChannel.MAGN,
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
            stab,
        ).scale(3.0),
        copy=False,
    )
    gchi0_q_core_inv.free()
    gchi0_q_full_sum.free()
    gchi0_q_core_sum.free()
    gamma_magn.mat = None
    if gamma_magn_win is None:
        gamma_magn.free()
    mpi_utils.free_shared_window(gamma_magn_win, shared_node_comm)
    logger.info("Calculated kernel for magnetic channel.")

    logger.info("Starting calculation of DGA self-energy.")

    # FFT contraction (the only production path - the q-loop variant peaks HIGHER, see sde_kernels.calculate_sigma_from_kernel):
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
        ntasks=config.lattice.k_grid.nk_irr, comm=comm, name="Q", output_path=config.output.output_path
    )
    full_q_list = config.lattice.k_grid.get_q_list()
    irrk_q_list = config.lattice.k_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    mpi_dist_fullbz = MpiDistributor.create_distributor(
        ntasks=config.lattice.k_grid.nk_tot, comm=comm, name="FBZ", output_path=config.output.output_path
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
        giwk_full_dmft.save(output_dir=config.output.output_path, name=output_files.G_LATT_DMFT_NAME)
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

    # the loop owns the mutable stabilization state; the (frozen) config.stabilization flags are only read once here
    stab = StabilizationState.from_config()
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
            stab=stab,
        )
        # delta_sigma = sigma_dmft.cut_niv(config.box.niv_core) - sigma_new.q_mean().cut_niv(config.box.niv_core)

        sigma_old = sigma_old.cut_niv(config.box.niv_core)

        logger.info("Applying mixing strategy to the self-energy.")
        sigma_old = sigma_old.concatenate_self_energies(sigma_dmft)
        history_cap = _mixing_history_cap(current_iter, release_iter, anneal_reset_iter)
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
                name=output_files.sigma_iteration_name(current_iter), output_dir=config.output.output_path
            )
            logger.info(f"Saved sigma for iteration {current_iter}.")

            if config.self_energy_interpolation.do_interpolation:
                beta_target = config.self_energy_interpolation.beta_target
                niv_target = config.self_energy_interpolation.niv_target
                sigma_new.decompress_q_dimension().interpolate(beta_target, niv_target).save(
                    name=output_files.sigma_interpolated_iteration_name(beta_target, niv_target, current_iter),
                    output_dir=config.output.output_path,
                )
                logger.info(
                    f"Interpolated sigma for iteration {current_iter} to beta={beta_target} and niv={niv_target}."
                )

        logger.info("Checking self-consistency convergence.")
        if comm.rank == 0 and current_iter > starting_iter + 1:
            # Convergence is declared on the post-mixing step residual (the returned iterate). The un-mixed proposal
            # residual is deliberately not used: it can plateau above epsilon and would block convergence forever.
            eps = stab.effective_epsilon()
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
        if stab.annealer is not None:
            anneal_mass_changed = stab.annealer.update(converged)
            if anneal_mass_changed:
                anneal_reset_iter = current_iter
            anneal_blocks_break = anneal_mass_changed or stab.annealer.mass_present

        sigma_old = sigma_new
        if converged:
            if stab.chi_phys_restriction_active:
                stab.chi_phys_restriction_active = False
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
            elif stab.lambda_correction_active:
                stab.lambda_correction_active = False
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

    np.save(os.path.join(config.output.output_path, output_files.MU_HISTORY_FILENAME), mu_history)
    logger.info("Saved mu history as numpy array.")

    return sigma_old

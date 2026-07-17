# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Self-energy mixing for the DGA self-consistency loop: linear mixing plus the accelerated Pulay (DIIS) and Anderson
schemes, the file-based self-energy history they read, and the history cap that keeps the accelerated schemes from
extrapolating across map-switching events of the loop (scaffold releases, annealing-mass changes).
"""

import glob
import os
import re

import numpy as np

import dgamore.config as config
import dgamore.output_files as output_files
from dgamore.self_energy import SelfEnergy


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

    interp_pattern = output_files.SIGMA_INTERPOLATED_GLOB
    interp_regex = output_files.SIGMA_INTERPOLATED_REGEX

    normal_pattern = output_files.SIGMA_ITERATION_GLOB
    normal_regex = output_files.SIGMA_ITERATION_REGEX

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


def _mixing_history_cap(
    current_iter: int, release_iter: int | None, anneal_reset_iter: int | None = None
) -> int | None:
    """
    Returns the accelerated-mixing history cap for this iteration: the number of iterations since the most recent
    map-switching event - the susceptibility-restriction release or a change of the lambda-annealing mass.
    Anderson/Pulay must not extrapolate across any of these discontinuities, so their usable history is capped to
    the post-event iterations (``None`` when no event has occurred).

    :param current_iter: The current self-consistency iteration number.
    :param release_iter: The iteration the susceptibility restriction was released on (``None`` if never).
    :param anneal_reset_iter: The iteration the annealing mass last changed on (``None`` if never).
    :return: The history cap, or ``None`` for no cap.
    """
    events = (release_iter, anneal_reset_iter)
    last_reset_iter = max((it for it in events if it is not None), default=None)
    return None if last_reset_iter is None else max(0, current_iter - last_reset_iter - 1)


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

        def get_proposal(idx: int) -> np.ndarray:
            """
            Fetches a flattened proposal self-energy from the history.

            :param idx: Index into the proposal history.
            :return: The flattened proposal self-energy at ``idx``.
            """
            return last_proposals[idx].flatten()

        def get_result(idx: int) -> np.ndarray:
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

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Self-energy mixing of the self-consistency loop (:mod:`~dgamore.nonlocal_sde`): the linear, Pulay and Anderson
strategies on the core window of the distributed iterates, the bound of their history, and the relative residuals the
loop's convergence check reads.
"""

import functools

import mpi4py.MPI as MPI
import numpy as np

import dgamore.config as config
from dgamore.jacobian_stabilization import JacobianTracker
from dgamore.memory_estimator import GRAM_ROWS
from dgamore.self_energy import SelfEnergy


def positive_core(sigma: SelfEnergy) -> np.ndarray:
    """
    Returns the positive fermionic core frequencies of a self-energy in the compressed ``[q, o1, o2, v]`` layout,
    whichever momentum layout the self-energy holds (a view of its array wherever the layout allows one).

    :param sigma: The self-energy.
    :return: Its positive core window.
    """
    core = sigma.mat[..., sigma.niv : sigma.niv + config.box.niv_core]
    return core.reshape(-1, *core.shape[-3:])


def relative_core_residual(new_core: np.ndarray, old_core: np.ndarray, comm: MPI.Comm | None = None) -> float:
    r"""
    Returns the larger of the relative L2 residuals of the real and the imaginary part of two positive core windows
    (:func:`positive_core`), :math:`\max\big(\lVert\mathrm{Re}\,(\Sigma_{\mathrm{new}} - \Sigma_{\mathrm{old}})\rVert
    / \lVert\mathrm{Re}\,\Sigma_{\mathrm{old}}\rVert,\ \lVert\mathrm{Im}\,(\Sigma_{\mathrm{new}} -
    \Sigma_{\mathrm{old}})\rVert / \lVert\mathrm{Im}\,\Sigma_{\mathrm{old}}\rVert\big)` over all momenta, orbitals and
    frequencies of the window. Each part is measured against its own norm, so a change in the smaller part is not
    hidden by the larger one; a part whose norm vanishes is measured against :math:`\lVert\Sigma_{\mathrm{old}}\rVert`
    instead, and a vanishing old window has no relative residual (infinite). A local (single-k) old window is
    broadcast against a full-BZ new one. The norms are accumulated in double precision (see :func:`_squared_norm`);
    with ``comm`` the windows hold this rank's momentum rows and the norms cover the rows of every rank in it.

    :param new_core: The positive core window of the new self-energy.
    :param old_core: The positive core window the residual is measured against.
    :param comm: The communicator over the owners of the momentum rows, or ``None`` when the windows hold every row.
    :return: The larger of the two relative residuals as a float.
    """
    if old_core.shape[0] != new_core.shape[0]:
        old_core = np.broadcast_to(old_core, new_core.shape)
    diff = new_core - old_core
    diff_re, diff_im, old_re, old_im = (
        np.sqrt(_squared_norm(part, comm)) for part in (diff.real, diff.imag, old_core.real, old_core.imag)
    )
    old_full = np.hypot(old_re, old_im)
    if not old_full:
        return float("inf")
    return float(max(diff_re / (old_re or old_full), diff_im / (old_im or old_full)))


def relative_sigma_residual(sigma_new: SelfEnergy, sigma_old: SelfEnergy, comm: MPI.Comm | None = None) -> float:
    r"""
    Returns the relative residual of :func:`relative_core_residual` between the positive fermionic core frequencies
    of two self-energies. Evaluated on the raw proposal it measures the mixing-independent distance to the fixed
    point, :math:`\lVert S(\Sigma)-\Sigma\rVert/\lVert\Sigma\rVert` per part; evaluated on the mixed iterate it
    measures the per-iteration step size (which shrinks with the mixing parameter). Layout-safe: the two iterates may
    arrive with different momentum layouts (compressed vs decompressed) and are normalized before comparing, and a
    local (single-k) self-energy is broadcast against a full-BZ one.

    :param sigma_new: The new self-energy (raw proposal or mixed iterate).
    :param sigma_old: The previous iterate the residual is measured against.
    :param comm: The communicator over the owners of the momentum rows, or ``None`` when the self-energies hold every
        row (see :func:`relative_core_residual`).
    :return: The larger of the real-part and the imaginary-part relative residual as a float.
    """
    # both windows come in the compressed layout: mismatched layouts (rank 0's iterate is left decompressed by the
    # save path) would pair wrong momenta
    return relative_core_residual(positive_core(sigma_new), positive_core(sigma_old), comm)


def mixing_history_cap(
    current_iter: int,
    release_iter: int | None,
    anneal_reset_iter: int | None = None,
    extra_event_iter: int | None = None,
) -> int | None:
    """
    Returns the history cap for this iteration: the number of iterations since the most recent map-switching event
    (the susceptibility-restriction release, a change of the lambda-annealing mass, or the caller's own event, e.g. a
    switch of the Jacobian tracker's reflected map). Nothing may extrapolate across such a discontinuity, so the
    usable history is capped to the post-event iterations (``None`` when no event has occurred).

    :param current_iter: The current self-consistency iteration number.
    :param release_iter: The iteration the susceptibility restriction was released on (``None`` if never).
    :param anneal_reset_iter: The iteration the annealing mass last changed on (``None`` if never).
    :param extra_event_iter: The iteration of the caller's own map-switching event (``None`` if never).
    :return: The history cap, or ``None`` for no cap.
    """
    events = (release_iter, anneal_reset_iter, extra_event_iter)
    last_reset_iter = max((it for it in events if it is not None), default=None)
    return None if last_reset_iter is None else max(0, current_iter - last_reset_iter - 1)


def core_pair(sigma_new: SelfEnergy, sigma_old: SelfEnergy, copy: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns this iteration's (iterate, raw proposal) pair: both core windows in the decompressed momentum layout, a
    momentum-local iterate (the fresh-run starting self-energy) broadcast over the proposal's momentum grid. Both
    self-energies keep the momentum layout they came with.

    :param sigma_new: The freshly computed self-energy proposal.
    :param sigma_old: The previous iteration's self-energy.
    :param copy: Whether the pair is a copy, which outlives both self-energies and the in-place core-window update of
        ``sigma_new`` the mixing makes (the mixing history's entries), or views of the two windows, for a caller that
        reads them while both self-energies stay unchanged.
    :return: The iterate and the raw proposal, cut to the core window.
    """
    niv_core = config.box.niv_core
    compressed = [sigma.has_compressed_q_dimension for sigma in (sigma_new, sigma_old)]
    sigma_old.decompress_q_dimension()
    sigma_new.decompress_q_dimension()
    proposal = sigma_new.mat[..., sigma_new.niv - niv_core : sigma_new.niv + niv_core]
    iterate = np.broadcast_to(sigma_old.mat[..., sigma_old.niv - niv_core : sigma_old.niv + niv_core], proposal.shape)
    if copy:
        iterate, proposal = iterate.copy(), proposal.copy()
    for sigma, was_compressed in zip((sigma_new, sigma_old), compressed):
        if was_compressed:
            sigma.compress_q_dimension()
    return iterate, proposal


def _real_rows(values: np.ndarray) -> list[tuple[slice, np.ndarray]]:
    """
    Splits a flat history vector into the row blocks of the real least-squares problem of the accelerated mixing: the
    real part over the imaginary part of a complex core window, a real vector in the tracker's coordinates as it is.

    :param values: The flat vector.
    :return: The ``(rows, part)`` blocks.
    """
    if np.iscomplexobj(values):
        return [(slice(0, values.size), values.real), (slice(values.size, 2 * values.size), values.imag)]
    return [(slice(0, values.size), values)]


def _row_sum(partial: np.ndarray, comm: MPI.Comm | None) -> np.ndarray:
    """
    Sums the partial sums of the ranks that own momentum rows in rank order, so every rank gets the same bits.

    :param partial: This rank's partial sum.
    :param comm: The communicator over the owners of the rows, or ``None`` when this rank owns every row.
    :return: The sum over the ranks.
    """
    return partial if comm is None else functools.reduce(np.add, comm.allgather(partial))


def _squared_norm(values: np.ndarray, comm: MPI.Comm | None) -> float:
    """
    Returns the squared norm of a row-distributed vector, accumulated in double precision (a float64 copy of real
    entries, a complex128 one of complex entries) in chunks of :data:`~dgamore.memory_estimator.GRAM_ROWS` entries
    and summed over the owners of the rows (see :func:`_row_sum`).

    :param values: This rank's rows of the vector (real or complex).
    :param comm: The communicator over the owners of the rows, or ``None`` when this rank owns every row.
    :return: The squared norm of the whole vector.
    """
    flat, total = values.reshape(-1), 0.0
    wide = np.complex128 if np.iscomplexobj(values) else np.float64
    for start in range(0, flat.size, GRAM_ROWS):
        chunk = flat[start : start + GRAM_ROWS].astype(wide).view(np.float64)
        total += np.dot(chunk, chunk)
    return float(_row_sum(np.array(total), comm))


def _least_squares_coefficients(
    columns: np.ndarray, rhs: np.ndarray, tikhonov: bool, comm: MPI.Comm | None
) -> tuple[np.ndarray | None, float]:
    r"""
    Solves the least-squares problem :math:`\min_c \lVert A c - b \rVert` of row-distributed real columns from the
    augmented Gram matrix :math:`[A\,b]^T [A\,b]`, accumulated in double precision in chunks of
    :data:`~dgamore.memory_estimator.GRAM_ROWS` rows (one float64 block, which every chunk overwrites) and summed over
    the owners of the rows (see :func:`_row_sum`): with the eigendecomposition
    :math:`A^T A = V \mathrm{diag}(\lambda) V^T` and the singular values :math:`s = \sqrt{\lambda}` of :math:`A`, the
    directions with :math:`s \leq 10^{-5} s_{\max}` are dropped and :math:`c = V \mathrm{diag}(1/\lambda) V^T A^T b`
    over the others, or :math:`c = V \mathrm{diag}(1/(\lambda + s_{\mathrm{cut}}^2)) V^T A^T b` with ``tikhonov``,
    which are the truncated and the regularized pseudoinverse of the thin SVD of :math:`A` written with its Gram
    matrix. The first owner solves and broadcasts the coefficients, so every owner takes the same step.

    :param columns: This rank's rows of :math:`A`, shape ``[rows, m]``.
    :param rhs: This rank's rows of :math:`b`.
    :param tikhonov: Whether the kept directions are regularized by the cutoff instead of inverted.
    :param comm: The communicator over the owners of the rows, or ``None`` when this rank owns every row.
    :return: The tuple ``(c, norm_b)`` of the coefficients (``None`` when no direction survives the cutoff or the
        eigendecomposition fails) and :math:`\lVert b \rVert`.
    """
    gram = np.zeros((columns.shape[1] + 1,) * 2)
    block = np.empty((min(len(rhs), GRAM_ROWS), columns.shape[1] + 1))
    for start in range(0, len(rhs), GRAM_ROWS):
        rows = block[: min(GRAM_ROWS, len(rhs) - start)]
        rows[:, :-1], rows[:, -1] = columns[start : start + len(rows)], rhs[start : start + len(rows)]
        gram += rows.T @ rows
    gram = _row_sum(gram, comm)
    coeffs = _gram_solution(gram, tikhonov) if comm is None or comm.rank == 0 else None
    return (coeffs if comm is None else comm.bcast(coeffs, root=0)), float(np.sqrt(gram[-1, -1]))


def _gram_solution(gram: np.ndarray, tikhonov: bool) -> np.ndarray | None:
    r"""
    Returns the least-squares coefficients of :func:`_least_squares_coefficients` from its augmented Gram matrix.

    :param gram: The augmented Gram matrix :math:`[A\,b]^T [A\,b]`.
    :param tikhonov: Whether the kept directions are regularized by the cutoff instead of inverted.
    :return: The coefficients, or ``None`` when no direction survives the cutoff or the eigendecomposition fails.
    """
    try:
        lam, v = np.linalg.eigh(gram[:-1, :-1])
    except np.linalg.LinAlgError:
        return None
    s = np.sqrt(np.clip(lam, 0.0, None))
    cutoff = 1e-5 * (s.max() if s.size else 1.0)
    mask = s > cutoff
    if not np.any(mask):
        return None
    denominator = lam[mask] + cutoff**2 if tikhonov else lam[mask]
    return v[:, mask] @ ((v[:, mask].T @ gram[:-1, -1]) / denominator)


def _step_core_window(core: np.ndarray, iterate: np.ndarray, p_config: float, correction: np.ndarray) -> None:
    r"""
    Overwrites a core window that holds the proposal :math:`S(x)` with :math:`x + p\,(S(x) - x) + c` in place.

    :param core: The core window, holding the proposal on entry.
    :param iterate: The iterate :math:`x` on the same window.
    :param p_config: The mixing parameter :math:`p`.
    :param correction: The correction :math:`c` on the same window.
    :return: None.
    """
    core -= iterate
    core *= p_config
    core += iterate
    core += correction


def apply_mixing_strategy(
    sigma_new: SelfEnergy,
    sigma_old: SelfEnergy,
    history_cap: int | None = None,
    mixing_history: list | None = None,
    tracker: JacobianTracker | None = None,
    comm: MPI.Comm | None = None,
) -> SelfEnergy:
    r"""
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
    mixing overwrites ``sigma_new`` in place) into ``mixing_history`` itself. A momentum-local iterate (the fresh-run
    starting self-energy) is broadcast over the proposal's momentum grid. Without a tracker the history copies the
    pair's core windows; with one it holds their real sector vectors (:meth:`JacobianTracker.record`), and an
    accelerated step is the linear step :math:`p\,(S(x) - x)` on the whole core window plus the history's
    correction unfolded from the sector (:meth:`JacobianTracker.unfold`).

    While the tracker has a flipped subspace installed, the proposal residual is reflected on it
    (:meth:`JacobianTracker.reflect`) before any strategy acts, and the reflected proposal is what the history
    records. A damped step (linear mixing, the warm-up and the fallbacks of the accelerated schemes) is then corrected
    on the certified span by :meth:`JacobianTracker.stabilize_step`. An accelerated step keeps its own step and is
    replaced by the stabilized damped step, with a warning, when :meth:`JacobianTracker.opposes_flip` reports that it
    opposes the flipped damped step; once the estimates have certified the flipped direction itself over two secant
    windows (:meth:`JacobianTracker.reflection_is_settled`), only the step's component on the reflected subspace is
    replaced (:meth:`JacobianTracker.replace_flipped_component`), with a warning, and the rest of the step stands.

    The accelerated schemes solve their least-squares problem from the double-precision Gram matrix of their secant
    columns (see :func:`_least_squares_coefficients`). The two self-energies may hold a block of the momentum rows of
    the whole self-energies (compressed momentum axis), as every rank of the self-consistency loop holds its own:
    every step acts on each row on its own, except the Gram matrix and the norms of the clamp, which are summed over
    the owners of the rows in ``comm`` (the first owner solving the Gram matrix for all of them), so every rank takes
    the same decisions and its rows hold the whole step to rounding: the grouping of those sums and of the products
    over blocks of rows. A tracker needs every row on one rank.

    :param sigma_new: The freshly computed self-energy proposal (this rank's momentum rows).
    :param sigma_old: The previous iteration's self-energy (the same rows, or momentum-local).
    :param history_cap: Optional upper bound on the number of secant pairs used by the accelerated schemes
        (``None`` for no bound). Used to reset the mixing history after the susceptibility-restriction release, so
        the accelerated schemes never extrapolate across the restricted-to-unrestricted discontinuity. Pairs keep
        being recorded while capped, so the history re-arms from genuine post-release pairs.
    :param mixing_history: Mutable list of (iterate, proposal) pairs of core-cut decompressed arrays, or with a
        tracker of their real vectors in its coordinates, the proposal the reflected one while a reflector is
        installed, oldest first, maintained across iterations by the caller (every owner of momentum rows of the
        self-consistency loop, its own rows). This function appends the current pair and trims the list to the
        configured history length plus one. ``None`` disables both the recording and the accelerated schemes.
    :param tracker: The :class:`JacobianTracker` owning the flipped subspace and the damping bound, or ``None`` to
        mix the raw proposal with the configured mixing parameter.
    :param comm: The communicator over the owners of the momentum rows, whose Gram matrices and norms are summed in
        rank order (see :func:`_row_sum`); ``None`` when the self-energies hold every row.
    :return: The mixed :class:`SelfEnergy` for the next iteration: ``sigma_new`` itself, its core window
        overwritten, after an accelerated step, and a new object after a damped step.
    """
    logger = config.logger
    n_hist = config.self_consistency.mixing_history_length
    if history_cap is not None:
        n_hist = min(n_hist, history_cap)
    p_config = config.self_consistency.mixing

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
        mixed.mat[..., mixed_window] = tracker.stabilize_step(mixed.mat[..., mixed_window], iterate, residual)
        return mixed

    last_results, last_proposals, residual = [], [], None
    if tracker is not None or mixing_history is not None:
        niv_core = config.box.niv_core
        # only a history of windows copies the pair; a tracker's history holds its sector vectors
        iterate, proposal_raw = core_pair(sigma_new, sigma_old, copy=mixing_history is not None and tracker is None)
        sigma_old.decompress_q_dimension()
        sigma_new.decompress_q_dimension()
        core_window = slice(sigma_new.niv - niv_core, sigma_new.niv + niv_core)
        proposal_used = proposal_raw
        if tracker is not None and tracker.active:
            # the raw proposal is a view of the core window the reflected one overwrites
            residual = proposal_raw - iterate
            proposal_used = tracker.reflect(proposal_raw, iterate)
            sigma_new.mat[..., core_window] = proposal_used
        if mixing_history is not None:
            mixing_history.append(
                (iterate, proposal_used) if tracker is None else tracker.record(iterate, proposal_used)[:2]
            )
            del mixing_history[: -(config.self_consistency.mixing_history_length + 1)]

        if (
            mixing_history is not None
            and config.self_consistency.mixing_strategy.lower() in ("pulay", "anderson")
            and n_hist > 0
        ):
            pairs = mixing_history[-(n_hist + 1) :]
            last_proposals = [iterate for iterate, _ in pairs]  # [x_{n-m}, ..., x_n]
            last_results = [proposal for _, proposal in pairs]  # [S(x_{n-m}), ..., S(x_n)]
            logger.info(f"Using the last {len(pairs)} (iterate, proposal) pairs of the mixing history.")

    alpha = tracker.p_eff if tracker is not None else p_config
    tracker_active = tracker is not None and tracker.active
    accelerated_mixing_condition = len(last_results) > n_hist

    if config.self_consistency.mixing_strategy.lower() == "pulay" and accelerated_mixing_condition:
        shape = last_results[-1].shape
        n_total = int(np.prod(shape))
        n_rows = 2 * n_total if tracker is None else n_total
        f_matrix = np.zeros((n_rows, n_hist), dtype=np.float64)
        f_i = np.zeros(n_rows, dtype=np.float64)

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
            for rows, part in _real_rows(get_result(-1 - i) - get_result(-2 - i)):
                f_matrix[rows, i] = part
            for rows, part in _real_rows(get_proposal(-1 - i) - get_proposal(-2 - i)):
                f_matrix[rows, i] -= part
        del part

        # Residual: F(x_n) - x_n, where x_n = last_proposals[-1] is the history's newest iterate
        for rows, part in _real_rows(get_result(-1) - get_proposal(-1)):
            f_i[rows] = part
        del part

        # Solve min||F @ c - f_i|| with the truncated pseudoinverse (drops collinear directions)
        coeffs, norm_f = _least_squares_coefficients(f_matrix, f_i, False, comm)
        if coeffs is None:
            logger.warning("Pulay least-squares problem ill-conditioned - falling back to linear mixing.")
            return stabilize(alpha * sigma_new + (1 - alpha) * sigma_old)

        # Pulay update at the configured parameter p: x_{n+1} = x_n + p*f_i - (R + p*F) @ c, with R + p*F formed in
        # place of F
        f_matrix *= p_config
        for i in range(n_hist):
            for rows, part in _real_rows(get_proposal(-1 - i) - get_proposal(-2 - i)):
                f_matrix[rows, i] += part
        del part
        update = p_config * f_i - f_matrix @ coeffs
        norm_u = np.sqrt(_squared_norm(update, comm))
        if norm_f > 0 and norm_u > 10.0 * norm_f:
            update *= 10.0 * norm_f / norm_u
            logger.warning(f"Pulay step clamped (norm_u={norm_u:.3e}, norm_f={norm_f:.3e}).")
        if tracker is None:
            update = update[:n_total] + 1j * update[n_total:]
        elif tracker.opposes_flip(update, residual):
            if not tracker.reflection_is_settled():
                del f_matrix, update
                logger.warning(
                    "Pulay step opposes the flipped damped step on the reflected subspace - taking the stabilized "
                    "damped step instead."
                )
                return stabilize(alpha * sigma_new + (1 - alpha) * sigma_old)
            update = tracker.replace_flipped_component(update, residual, alpha)
            logger.warning(
                "Pulay step opposes the flipped damped step on the settled reflected subspace - taking the stabilized "
                "damped step there."
            )

        # Update the new self energy; with a tracker the core window takes the linear step and the history's
        # correction to it, unfolded once from the tracker's coordinates
        niv = sigma_new.niv
        niv_core = config.box.niv_core
        if tracker is None:
            sigma_new.mat[..., niv - niv_core : niv + niv_core] = iterate + update.reshape(shape)
        else:
            update -= p_config * f_i
            correction = tracker.unfold(update, iterate.shape)
            _step_core_window(sigma_new.mat[..., niv - niv_core : niv + niv_core], iterate, p_config, correction)

        logger.info(f"Pulay mixing applied (m={n_hist}, alpha={p_config:.3f}, norm_f={norm_f:.3e}).")

        return sigma_new
    if config.self_consistency.mixing_strategy.lower() == "anderson" and accelerated_mixing_condition:
        shape = last_results[-1].shape
        n_total = int(np.prod(shape))
        flat = lambda x: x.reshape(-1)

        # Current residual f_n = F(x_n) - x_n
        f_curr = flat(last_results[-1]) - flat(last_proposals[-1])
        f_vec = np.concatenate([part for _, part in _real_rows(f_curr)])
        del f_curr

        # dF (n_hist columns, the row blocks of _real_rows): dF[:,i] = f_{n-i} - f_{n-i-1} (residual differences)
        df_matrix = np.empty((f_vec.size, n_hist), dtype=f_vec.dtype)
        for i in range(n_hist):
            df = (flat(last_results[-1 - i]) - flat(last_proposals[-1 - i])) - (
                flat(last_results[-2 - i]) - flat(last_proposals[-2 - i])
            )
            for rows, part in _real_rows(df):
                df_matrix[rows, i] = part
        del df, part

        # Anderson: solve min ||f_n - dF @ c|| with the regularized pseudoinverse; the update keeps the row precision
        coeffs, norm_f = _least_squares_coefficients(df_matrix, f_vec, True, comm)
        if coeffs is None:
            logger.warning("Anderson least-squares problem ill-conditioned - falling back to linear mixing.")
            return stabilize(alpha * sigma_new + (1 - alpha) * sigma_old)
        coeffs = coeffs.astype(df_matrix.dtype)

        # Undamped Anderson proposal: x_n + f_n - (dX + dF) @ c; the proposal differences
        # dX[:,i] = x_{n-i} - x_{n-i-1} are added into dF's columns once the solve no longer needs dF
        for i in range(n_hist):
            dx = flat(last_proposals[-1 - i]) - flat(last_proposals[-2 - i])
            for rows, part in _real_rows(dx):
                df_matrix[rows, i] += part
        del dx, part
        x_n = flat(last_proposals[-1])
        x_anderson = np.concatenate([part for _, part in _real_rows(x_n)]) + f_vec - df_matrix @ coeffs
        if tracker is None:
            x_anderson = x_anderson[:n_total] + 1j * x_anderson[n_total:]

        # Damp between old proposal and Anderson proposal
        candidate = (1 - p_config) * x_n + p_config * x_anderson.reshape(-1)

        # Safety clamp
        update = candidate - x_n
        norm_u = np.sqrt(_squared_norm(update, comm))
        if norm_f > 0 and norm_u > 3.0 * norm_f:
            update = update * (3.0 * norm_f / norm_u)
            candidate = x_n + update
            logger.warning(f"Anderson step clamped (norm_u={norm_u:.3e}, norm_f={norm_f:.3e}).")
        if tracker is not None and tracker.opposes_flip(update, residual):
            if not tracker.reflection_is_settled():
                del df_matrix, x_anderson, candidate, update
                logger.warning(
                    "Anderson step opposes the flipped damped step on the reflected subspace - taking the stabilized "
                    "damped step instead."
                )
                return stabilize(alpha * sigma_new + (1 - alpha) * sigma_old)
            update = tracker.replace_flipped_component(update, residual, alpha)
            logger.warning(
                "Anderson step opposes the flipped damped step on the settled reflected subspace - taking the "
                "stabilized damped step there."
            )

        # Update the new self energy; with a tracker the core window takes the linear step and the history's
        # correction to it, unfolded once from the tracker's coordinates
        niv = sigma_new.niv
        niv_core = config.box.niv_core
        if tracker is None:
            sigma_new.mat[..., niv - niv_core : niv + niv_core] = candidate.reshape(shape)
        else:
            update -= p_config * f_vec
            correction = tracker.unfold(update, iterate.shape)
            _step_core_window(sigma_new.mat[..., niv - niv_core : niv + niv_core], iterate, p_config, correction)

        logger.info(f"Anderson acceleration applied (m={n_hist}, alpha={p_config:.3f}, norm_f={norm_f:.3e}).")

        return sigma_new

    sigma_new = alpha * sigma_new + (1 - alpha) * sigma_old
    logger.info(f"Sigma linearly mixed (m=1, alpha={alpha}).")
    return stabilize(sigma_new)

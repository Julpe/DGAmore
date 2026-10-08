# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Stabilization of the physical fixed point of the self-energy self-consistency by flipping the sign of the damping on
the unstable directions of the proposal map.

The damped iteration :math:`x_{n+1} = x_n + p\,(S(x_n) - x_n)` over the vectorized core window :math:`x` of
:math:`\Sigma` is attractive only where the eigenvalues :math:`\lambda_\Pi` of :math:`\Pi = \mathbb{1} - J`, with
:math:`J = \mathrm{d}S/\mathrm{d}x` the Jacobian of the proposal map, have a positive real part. Directions with
:math:`\mathrm{Re}\,\lambda_\Pi \leq 0` cannot be damped into the unit disk at any :math:`p > 0` and are cured
instead by reversing the sign of the damping on them.

This module reads the leading eigenpairs of :math:`J` with residual certificates off the (iterate, proposal) pairs the
self-consistency already records (a Rayleigh-Ritz on the secant samples :math:`\delta F \simeq J\,\delta X`), decides
which directions to flip, also ahead of an extrapolated crossing or right after a pole, and builds the reflector and
the per-direction damping map that :class:`JacobianTracker` installs and releases. The exact checks the tracker
requests (:meth:`JacobianTracker.exact_check_request`) run outside this module, like every evaluation of the
proposal map, and the module imports neither MPI nor the configuration.

Follows H. Essl, S. Rohshap, M. Gievers, M. Wallerberger, A. Toschi and A. Kauch, arXiv:2606.04936, and H. Essl, M.
Reitner, E. Kozik and A. Toschi, Phys. Rev. Lett., doi:10.1103/zjy7-4jqd. The reflector follows the Schur construction
of arXiv:2609.11405, App. G3, the boson-exchange follow-up of arXiv:2606.04936, and the per-direction damping Eq. (27)
of arXiv:2606.04936 and Eq. (21) of arXiv:2609.11405 with the stabilization matrix of its App. G3."""

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr, schur, solve_sylvester, solve_triangular
from scipy.linalg.lapack import dtrsen

from dgamore.local_n_point import _flush_and_drop

_MARGIN = 1e-2  # |Re lambda_Pi| below this is undecidable: not flipped, binds the damping with the margin
_DAMPING_C = 0.5  # safety factor c in (0, 1) of the damping bound, at the vertex of the stability parabola
_MIXING_FLOOR = 0.01  # the effective damping is never lowered below this
_RITZ_GATE = 3e-2  # Ritz residual gate, relative to max(1, |theta|)
_FLIP_PERSIST = 1  # updates carrying a certified unstable mode before the flip is acted on
_PERSIST = 3  # updates of the calm and growth releases, and of the hold a growth release leaves behind
_MIN_PAIRS = 3  # fewer pairs than this carry no estimate
_SUBSPACE_ANGLE = 0.1  # rad; largest principal angle above which the flipped basis is replaced
_COLLINEAR_CUTOFF = 1e-8  # relative cutoff below which a secant column counts as collinear
_NOISE_FACTOR = 1000.0  # noise floor and step gate in machine epsilons of the iterate norm
_COND_CAP = 1e3  # Sylvester-solution norm and carried-column condition above which no reflector is built
_MAP_NORM_CAP = 2.0  # largest spectral norm of a damping map in units of the largest damping it encodes
_SIGNS_UNBOUNDED: tuple = ()  # _carried_nonuniform_map result whose signs carry no bounded weighting
_OBLIQUE_CAP = 10.0  # spectral norm above which an oblique projector falls back to the orthogonal one
_REAL_CUTOFF = 1e-8  # relative cutoff below which a Ritz value counts as real
_SCHUR_TOL = 1e-10  # relative tolerance locating a Ritz value on its Schur form and tying two blocks
_FLIP_OVERLAP = 0.5  # |Q^T u| above which a Ritz vector lies in a flipped subspace or matches a mode
_PREDICT_IN_LOOP = True  # flip a direction as soon as its extrapolated real part crosses, before it is measured
_PREDICT_ACROSS_RUNGS = True  # flip a carried direction whose two-rung extrapolation in beta crosses
_NONUNIFORM_IN_LOOP = True  # build the per-direction damping map from the running estimate; False leaves the damped
# steps on the scalar damping alone (the map of a carried set is unaffected)
_STORE_BAND = 0.1  # Re lambda_Pi below which a certified mode's Ritz vector is written for a successor run; this is
# the storage band, not the flip band -max(_MARGIN, res) of classify
_CHECK_BAND = 1.0  # Re lambda_Pi below which an exact snapshot stores a mode's vector for the successor's exact check
_MAX_EXACT_CHECKS = 5  # mid-rung exact checks one run may request, plus up to as many that certified their lead and
# found every certified pair above its undecidable band (the rung-start check does not count)
_CHECK_START = 6  # real start columns of the rung-start exact check, least stable modes first
_CHAIN_LENGTH = 3  # informative updates the values of a mode are chained over for the extrapolation
_POLE_MODULUS = 5.0  # |lambda_Pi| beyond which a mode counts as approaching or crossing a pole
_POLE_CONTINUITY = 0.5  # relative growth of the 1/lambda_Pi step still read as one drift through a pole
_MAX_BETA_STEP_RATIO = 2.0  # largest (beta_3 - beta_2) / (beta_2 - beta_1) a two-rung extrapolation may bridge
TRACKER_PAIRS = 7  # (iterate, proposal) pairs the tracker keeps, i.e. 6 secant differences per estimate
_MIN_ITERATION_CAP = 15000  # upper limit of the minimum-iteration guard
JACOBIAN_FILE = "jacobian.npz"  # per-iteration traces, rewritten every iteration, plus the final spectrum


@dataclass(frozen=True, eq=False)
class SchurForm:
    """
    Real Schur form :math:`B = Z T Z^{T}` of a projected map together with the diagonal position of every Ritz
    value read from it.

    :ivar t: The quasi-triangular factor :math:`T`, shape ``[k, k]``.
    :ivar z: The orthogonal factor :math:`Z`, shape ``[k, k]``.
    :ivar positions: The diagonal position of each Ritz value on :math:`T`, in the order of the Ritz values, or
        ``None`` when a Ritz value could not be located.
    """

    t: np.ndarray
    z: np.ndarray
    positions: np.ndarray | None


def to_vec(mat: np.ndarray) -> np.ndarray:
    r"""
    Flattens a complex array into its real representation :math:`[\mathrm{Re};\,\mathrm{Im}]`. The proposal map is
    real-linear but not complex-analytic, so its Jacobian lives on that real space.

    :param mat: Complex array of any shape.
    :return: Real ``float64`` vector of length ``2 * mat.size``.
    """
    flat = np.asarray(mat).reshape(-1)
    return np.concatenate((flat.real, flat.imag)).astype(np.float64, copy=False)


def to_mat(vec: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """
    Inverse of :func:`to_vec`: rebuilds the complex array from its real representation.

    :param vec: Real vector of length ``2 * prod(shape)``.
    :param shape: Shape of the complex array to rebuild.
    :return: The ``complex128`` array of the given shape.
    """
    half = vec.size // 2
    return (vec[:half] + 1j * vec[half:]).reshape(shape)


def noise_floor(x: np.ndarray, dtype: np.dtype | None = None) -> float:
    r"""
    Returns the absolute norm below which an increment of the iterate :math:`x` carries no information beyond the
    rounding noise of its storage precision, :math:`10^{3}\,\epsilon_{\mathrm{mach}}\,\lVert x \rVert` with
    :math:`\epsilon_{\mathrm{mach}}` of the storage dtype.

    :param x: The iterate (real vector or complex array).
    :param dtype: The dtype the iterate is stored in; ``None`` takes the array's own.
    :return: The noise floor.
    """
    single = np.dtype(np.asarray(x).dtype if dtype is None else dtype) in (np.complex64, np.float32)
    return _NOISE_FACTOR * np.finfo(np.float32 if single else np.float64).eps * float(np.linalg.norm(x))


def _independent_columns(r: np.ndarray, noise_floor: float) -> np.ndarray:
    r"""
    Selects the columns of the thin-QR factor :math:`R` of the secant increments that carry an independent
    direction, newest first, so a stalled or repeated step cannot make the second QR factorization singular.
    Because :math:`\delta X = QR` with :math:`Q` orthonormal, dropping a column of :math:`R` drops the matching
    increment, so the sweep runs on the small :math:`R`. A column is kept when its part orthogonal to the already kept
    (more recent) ones exceeds :math:`\max(\epsilon_{\mathrm{col}}\,\sigma_{\mathrm{max}}(R),\, \eta)`: the relative
    cutoff catches a repeated step, the absolute floor :math:`\eta` a column of pure rounding noise.

    :param r: Upper-triangular factor of the thin QR of the secant increments, shape ``[min(n_real, m), m]``.
    :param noise_floor: Absolute norm :math:`\eta` below which an increment carries no information beyond
        rounding noise.
    :return: Sorted indices of the kept columns.
    """
    tol = max(_COLLINEAR_CUTOFF * float(np.linalg.svd(r, compute_uv=False)[0]), noise_floor)
    basis, keep = [], []
    for col in range(r.shape[1] - 1, -1, -1):
        residual = r[:, col].astype(np.float64, copy=True)
        for _ in range(2):
            for vec in basis:
                residual -= (vec @ residual) * vec
        norm = float(np.linalg.norm(residual))
        if norm > tol:
            basis.append(residual / norm)
            keep.append(col)
    return np.array(sorted(keep), dtype=int)


def secant_ritz(xs: np.ndarray, fs: np.ndarray, noise_floor: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Rayleigh-Ritz on the secant samples of the proposal map. Consecutive differences of the iterates and of their
    proposals are first-order samples of the Jacobian, :math:`\delta F \simeq J\,\delta X`, so projecting onto the
    space they span gives Ritz pairs of :math:`J` together with a residual certificate:

    .. math::

        \delta X = QR, \qquad B = Q^{T}\,\delta F\,R^{-1} \simeq Q^{T} J Q, \qquad B y = \theta y,
        \qquad u = Q y, \qquad \mathrm{res} = \frac{\lVert \delta F R^{-1} y - \theta u \rVert}{\lVert u \rVert}.

    Collinear increments and increments below the noise floor are dropped first (see :func:`_independent_columns`)
    and the kept columns of :math:`R` are re-factored, :math:`R_{\mathrm{kept}} = Q_2 R_2`, so that :math:`R_2^{-1}`
    cannot amplify the rounding noise of the stored iterates into certified but meaningless Ritz values. The Ritz
    values estimate :math:`\lambda_J`; the caller forms :math:`\lambda_\Pi = 1 - \lambda_J`. This is the reference
    form of :func:`_ritz_from_increments`, returning the three Ritz arrays alone.

    :param xs: Real column stack of the iterates, oldest first, shape ``[n_real, n]`` with ``n >= 3``.
    :param fs: Real column stack of the matching proposals, oldest first, shape ``[n_real, n]``.
    :param noise_floor: Absolute norm below which an increment carries no information beyond rounding noise.
    :return: The Ritz values :math:`\theta`, the Ritz vectors :math:`u` as columns, and the Ritz residuals; all
        empty when no increment survives the noise floor.
    """
    return _ritz_from_increments(np.diff(xs, axis=1), np.diff(fs, axis=1), noise_floor)[:3]


def _increments(window: list[np.ndarray]) -> np.ndarray:
    r"""
    Builds the secant increments :math:`\delta X_{:,j} = x_{j+1} - x_j` of a window of real vectors in the tracker's
    coordinates as the columns of one preallocated ``float64`` array, every difference taken in ``float64`` whatever
    precision the vectors are stored in.

    :param window: The recorded real vectors, oldest first.
    :return: The increments, shape ``[n_real, len(window) - 1]``, oldest first.
    """
    increments = np.empty((window[0].size, len(window) - 1), dtype=np.float64)
    for col in range(increments.shape[1]):
        np.subtract(window[col + 1], window[col], out=increments[:, col], dtype=np.float64)
    return increments


def _ritz_from_increments(
    dx: np.ndarray, df: np.ndarray, noise_floor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SchurForm, np.ndarray]:
    r"""
    Runs the Rayleigh-Ritz of :func:`secant_ritz` on the secant increments themselves. The tall factor :math:`Q` is
    released once the kept basis :math:`Q_K = Q Q_2` exists, :math:`R_2^{-1}` is applied as a solve, the eigenpairs
    are read from the real Schur form of :math:`B` (:func:`_schur_ritz`), and :math:`u` and the residual are
    assembled from the real and imaginary part of :math:`y` without a complex copy of a tall factor. Each Ritz value
    also gets its eigenvalue condition number :math:`\kappa_\alpha = \lVert x_\alpha \rVert \lVert y_\alpha \rVert /
    |x_\alpha^{H} y_\alpha|` in :math:`B` (:math:`x_\alpha` the rows of :math:`y^{-1}`), which turns the residual
    into an error bound of the value.

    :param dx: Secant increments of the iterates, shape ``[n_real, m]``, oldest first.
    :param df: Matching increments of the proposals, shape ``[n_real, m]``.
    :param noise_floor: Absolute norm below which an increment carries no information beyond rounding noise.
    :return: The Ritz values, the Ritz vectors as columns, the Ritz residuals, the orthonormal basis of the kept
        increments, the Schur form of the projected matrix :math:`B` the Ritz pairs are read from and the condition
        numbers of the Ritz values; all empty when no increment survives the noise floor.
    """
    q, r = np.linalg.qr(dx)
    keep = _independent_columns(r, noise_floor)
    if keep.size == 0:
        cols, form = np.zeros((dx.shape[0], 0)), SchurForm(np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0, dtype=int))
        return np.zeros(0, dtype=np.complex128), cols.astype(np.complex128), np.zeros(0), cols, form, np.zeros(0)
    q2, r2 = np.linalg.qr(r[:, keep])
    q_kept = q @ q2
    del q
    jq = np.linalg.solve(r2.T, df[:, keep].T).T
    theta, y, form = _schur_ritz(q_kept.T @ jq)
    try:
        kappa = np.linalg.norm(np.linalg.inv(y), axis=1) * np.linalg.norm(y, axis=0)
    except np.linalg.LinAlgError:
        kappa = np.full(theta.size, 1.0 / np.finfo(np.float64).eps)
    u = np.empty((q_kept.shape[0], theta.size), dtype=np.complex128)
    u.real = q_kept @ y.real
    u.imag = q_kept @ y.imag
    res = np.empty(theta.size, dtype=np.float64)
    for col in range(theta.size):
        re_t, im_t = theta[col].real, theta[col].imag
        part_re = np.linalg.norm(jq @ y.real[:, col] - re_t * u.real[:, col] + im_t * u.imag[:, col])
        part_im = np.linalg.norm(jq @ y.imag[:, col] - re_t * u.imag[:, col] - im_t * u.real[:, col])
        res[col] = np.hypot(part_re, part_im) / np.linalg.norm(u[:, col])
    return theta, u, res, q_kept, form, kappa


def classify(
    lam_pi: np.ndarray, gated: np.ndarray, p_config: float, res: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    r"""
    Decides which certified modes to flip and how much damping the measured spectrum allows. A mode is flipped when
    no positive damping can pull it into the unit disk, and every certified mode binds the damping through

    .. math::

        p_{\mathrm{eff}} = \min\left( p_{\mathrm{config}},\; \max\left(
        c \min_\alpha \frac{2\,r_\alpha}{|\lambda_{\Pi,\alpha}|^{2}}, \; p_{\mathrm{floor}} \right) \right),
        \qquad r_\alpha = \begin{cases} \mathrm{Re}\,\lambda_{\Pi,\alpha} & \mathrm{Re}\,\lambda_{\Pi,\alpha} > 0,
        \\ \max(|\mathrm{Re}\,\lambda_{\Pi,\alpha}|,\, m_\alpha) & \text{otherwise}, \end{cases}

    evaluated with the real part of a flipped mode already reversed, so the configured damping is only ever lowered.
    A mode inside its undecidable band :math:`m_\alpha = \max(m, \mathrm{res}_\alpha)` is not flipped; measured on
    the stable side it binds with its real part (arXiv:2606.04936 Eq. (13)), measured on the unstable side with the
    band, so a pseudo-divergence with a large imaginary part still lowers the damping, a near-neutral mode does not.

    :param lam_pi: Measured eigenvalues :math:`\lambda_\Pi` of :math:`\Pi = \mathbb{1} - J`.
    :param gated: Whether each mode passed the Ritz residual gate.
    :param p_config: The configured damping, i.e. the ceiling of the returned value.
    :param res: The error bounds of the values, widening each mode's undecidable band; ``None`` keeps the fixed
        margin.
    :return: The boolean flip mask and the effective damping.
    """
    margin = np.full(lam_pi.shape, _MARGIN) if res is None else np.maximum(_MARGIN, res)
    flip = gated & (lam_pi.real < -margin)
    p_max = np.inf
    if gated.any():
        re_part = lam_pi.real[gated]
        real_part = np.where(re_part > 0.0, re_part, np.maximum(-re_part, margin[gated]))
        modulus_sq = np.maximum(np.abs(lam_pi[gated]) ** 2, np.finfo(np.float64).tiny)
        p_max = float(np.min(2.0 * real_part / modulus_sq))
    return flip, min(p_config, max(_DAMPING_C * p_max, _MIXING_FLOOR))


def _direction_damping(lam_pi: np.ndarray, flip: np.ndarray, band_damping: float | None = None) -> np.ndarray:
    r"""
    Returns the damping each direction allows on its own,

    .. math::

        s_\alpha p_\alpha, \qquad p_\alpha = \min\left(1,\; c\,
        \frac{2\,|\mathrm{Re}\,\lambda_{\Pi,\alpha}|}{|\lambda_{\Pi,\alpha}|^{2}}\right), \qquad
        s_\alpha = -1 \text{ on a flipped direction},

    so that a direction damped with it iterates with :math:`1 - s_\alpha p_\alpha \lambda_{\Pi,\alpha}`; both
    members of a conjugate pair receive the same damping. Unlike :func:`classify` it reads no undecidable band.

    :param lam_pi: The eigenvalues :math:`\lambda_\Pi` of the directions.
    :param flip: Which of them are flipped.
    :param band_damping: The damping of an unflipped direction with a negative real part, i.e. of undecidable sign;
        ``None`` gives it its own :math:`p_\alpha`.
    :return: The signed dampings.
    """
    modulus_sq = np.maximum(np.abs(lam_pi) ** 2, np.finfo(np.float64).tiny)
    damping = np.minimum(1.0, _DAMPING_C * 2.0 * np.abs(lam_pi.real) / modulus_sq)
    if band_damping is not None:
        damping = np.where(~flip & (lam_pi.real < 0.0), band_damping, damping)
    return np.where(flip, -damping, damping)


def _real_columns(theta: np.ndarray, u: np.ndarray, take: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Collects the real columns a set of Ritz vectors spans, one per real mode and two (the real and the imaginary
    part) per complex-conjugate pair, of which only the member with the positive imaginary part contributes since
    its partner spans the same real plane.

    :param theta: The Ritz values the vectors belong to.
    :param u: The Ritz vectors as columns.
    :param take: Which modes contribute.
    :return: The columns as one real ``[n_real, k]`` array and the mode index each of them belongs to; both empty
        when no column was collected (e.g. a lone conjugate-pair member with a negative imaginary part).
    """
    columns, owners = [], []
    for idx in np.flatnonzero(take):
        cutoff = _REAL_CUTOFF * (1.0 + abs(theta[idx]))
        if theta[idx].imag < -cutoff:
            continue
        columns.append(u[:, idx].real)
        owners.append(idx)
        if theta[idx].imag > cutoff:
            columns.append(u[:, idx].imag)
            owners.append(idx)
    if not columns:
        return np.zeros((u.shape[0], 0)), np.zeros(0, dtype=int)
    return np.column_stack(columns).astype(np.float64, copy=False), np.array(owners, dtype=int)


def _schur_blocks(t_mat: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    Returns the half-open index range of every diagonal block of a real Schur factor, ``2 x 2`` where the
    subdiagonal entry below the block is nonzero (a complex-conjugate pair) and ``1 x 1`` otherwise, and the
    eigenvalue every diagonal position carries (the conjugate pair of a ``2 x 2`` block on its two positions).

    :param t_mat: The quasi-triangular factor, shape ``[k, k]``.
    :return: The blocks as ``(start, stop)`` pairs, in order, and the complex values, one per diagonal position.
    """
    bounds, low, values = [], 0, np.empty(t_mat.shape[0], dtype=np.complex128)
    while low < t_mat.shape[0]:
        high = low + (2 if low + 1 < t_mat.shape[0] and t_mat[low + 1, low] != 0.0 else 1)
        values[low:high] = t_mat[low, low] if high - low == 1 else np.linalg.eigvals(t_mat[low:high, low:high])
        bounds.append((low, high))
        low = high
    return bounds, values


def _diagonal_positions(theta: np.ndarray, t_mat: np.ndarray) -> np.ndarray | None:
    """
    Locates every Ritz value on the diagonal of the real Schur factor it was read from: each value takes the first
    unclaimed diagonal position whose eigenvalue lies within ``_SCHUR_TOL`` (relative) of it, so the two members of
    a conjugate pair claim the two positions of their ``2 x 2`` block. Both readings come from the same
    quasi-triangular matrix, so the tolerance is a bookkeeping check and not a conditioning question.

    :param theta: The Ritz values, i.e. the eigenvalues of ``t_mat``.
    :param t_mat: The quasi-triangular factor, shape ``[k, k]``.
    :return: The diagonal position of each Ritz value, or ``None`` when one of them finds no unclaimed position.
    """
    values = _schur_blocks(t_mat)[1]
    positions = np.empty(theta.size, dtype=int)
    claimed = np.zeros(values.size, dtype=bool)
    for idx, value in enumerate(theta):
        free = np.flatnonzero(~claimed & (np.abs(values - value) <= _SCHUR_TOL * max(1.0, abs(value))))
        if free.size == 0:
            return None
        positions[idx] = free[0]
        claimed[free[0]] = True
    return positions


def _schur_ritz(b_proj: np.ndarray) -> tuple[np.ndarray, np.ndarray, SchurForm]:
    r"""
    Decomposes a projected map once, :math:`B = Z T Z^{T}`, and reads its eigenpairs off the quasi-triangular
    factor, :math:`T y = \theta y`, so that the Ritz values, the Schur blocks and the positions locating each
    value on the diagonal all come from the same matrix.

    :param b_proj: The projected map :math:`B`, shape ``[k, k]``.
    :return: The eigenvalues, the eigenvectors :math:`Z y` of :math:`B` as unit columns, and the Schur form with
        its positions.
    """
    t_mat, z_mat = schur(b_proj, output="real")
    theta, y = np.linalg.eig(t_mat)
    return theta, z_mat @ y, SchurForm(t_mat, z_mat, _diagonal_positions(theta, t_mat))


def _commuting_upper(t_mat: np.ndarray, bounds: list[tuple[int, int]], values: np.ndarray) -> np.ndarray:
    r"""
    Returns the block upper-triangular :math:`M` that commutes with a real Schur factor :math:`T` and carries one
    value per diagonal block, :math:`M_{aa} = m_a \mathbb{1}`. Reading :math:`[M, T] = 0` block by block leaves the
    Sylvester equation

    .. math::

        T_{aa} M_{ab} - M_{ab} T_{bb} = (m_a - m_b)\,T_{ab}
        + \sum_{a < c < b} \left( M_{ac} T_{cb} - T_{ac} M_{cb} \right),

    solved for increasing block distance, so the sum only reads blocks that are already known. The result is the
    weighted sum :math:`\sum_a m_a \Pi_a` of the spectral projectors of the blocks, formed without ever inverting
    the eigenvector matrix of a nearly defective map: two blocks that nearly coincide carry nearly equal values, so
    the right-hand side vanishes together with the separation that makes the equation ill-conditioned.

    :param t_mat: The quasi-triangular factor, shape ``[k, k]``.
    :param bounds: The diagonal blocks as ``(start, stop)`` pairs (see :func:`_schur_blocks`).
    :param values: One value per block.
    :return: The map :math:`M`, shape ``[k, k]``.
    """
    m_mat = np.zeros_like(t_mat)
    for (low, high), value in zip(bounds, values):
        m_mat[low:high, low:high] = value * np.eye(high - low)
    for distance in range(1, len(bounds)):
        for first in range(len(bounds) - distance):
            (la, ha), (lb, hb) = bounds[first], bounds[first + distance]
            rhs = (values[first] - values[first + distance]) * t_mat[la:ha, lb:hb]
            for lc, hc in bounds[first + 1 : first + distance]:
                rhs = rhs + m_mat[la:ha, lc:hc] @ t_mat[lc:hc, lb:hb] - t_mat[la:ha, lc:hc] @ m_mat[lc:hc, lb:hb]
            m_mat[la:ha, lb:hb] = solve_sylvester(t_mat[la:ha, la:ha], -t_mat[lb:hb, lb:hb], rhs)
    return m_mat


def _bounded_map(
    build: Callable[[np.ndarray], np.ndarray],
    damping: np.ndarray,
    bounds: list[tuple[int, int]],
    tied: list[tuple[int, int]],
    log: Callable[[str], None] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    r"""
    Assembles a damping map and caps it against the damping it encodes: a map whose spectral norm exceeds
    ``_MAP_NORM_CAP`` times the largest :math:`|m_a|` it carries would amplify the residual instead of damping it.
    The blocks are grouped, starting from the ``tied`` pairs and joining the two groups the largest off-diagonal block
    of the map couples, until the map lies inside the cap; every block of a group takes the smallest damping of the
    group with its own sign, :math:`\min_{\alpha \in G} p_\alpha \sum_{\alpha \in G} s_\alpha \Pi_\alpha`. A map
    still above the cap with every block in one group (a flipped and a stable eigenvector nearly parallel) is refused.

    :param build: Builds the map in the coordinates of its own basis from one signed value per block, the weighted
        sum of the blocks' spectral projectors.
    :param damping: The signed damping of every block.
    :param bounds: The index range of every block in those coordinates, as ``(start, stop)`` pairs.
    :param tied: Pairs of blocks that share one value whatever the map's norm.
    :param log: Callable receiving the fallback and refusal messages, or ``None`` to stay silent.
    :return: The map together with the signed damping each block takes in it, its own one or that of its group;
        ``None`` when even the uniform map lies above the cap.
    """
    group = np.arange(damping.size)
    for first, second in tied:
        group[group == group[second]] = group[first]
    while True:
        values = np.sign(damping) * np.array([np.min(np.abs(damping[group == label])) for label in group])
        m_mat = build(values)
        if float(np.linalg.norm(m_mat, 2)) <= _MAP_NORM_CAP * float(np.max(np.abs(values))):
            break
        if np.all(group == group[0]):
            if log is not None:
                log(
                    "Jacobian tracker: the signs of the certified damping cannot be carried on the directions they "
                    f"belong to within {_MAP_NORM_CAP:g} times the damping they encode, no per-direction damping used."
                )
            return None
        coupling = [
            (float(np.linalg.norm(m_mat[la:ha, lb:hb])), a, b)
            for a, (la, ha) in enumerate(bounds)
            for b, (lb, hb) in enumerate(bounds)
            if group[a] != group[b]
        ]
        _, first, second = max(coupling)
        group[group == group[second]] = group[first]
    if log is not None and np.any(values != damping):
        shared = np.flatnonzero(np.bincount(group, minlength=group.size)[group] > 1)
        log(
            f"Jacobian tracker: nonuniform damping not separable on {shared.size} coupled certified blocks of the "
            f"span (spectral norm above {_MAP_NORM_CAP:g} times the damping it encodes), uniform damping "
            f"{', '.join(f'{p:.4f}' for p in np.unique(np.abs(values[shared])))} used on each coupled group of them."
        )
    return m_mat, values


def _invariant_columns(
    form: SchurForm, selected: np.ndarray, log: Callable[[str], None] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    r"""
    Separates the invariant subspace of a set of Ritz values from the complementary one by reordering the real
    Schur form they were read from. The diagonal blocks of the complementary values are moved into the leading
    block :math:`T_{11}` and those of the selected values into the trailing block :math:`T_{22}`, which splits the
    orthogonal factor into the columns :math:`Z_1` of the leading block and :math:`Z_2` of the trailing one,
    coupled by the off-diagonal block :math:`T_{12}`:

    .. math::

        B = Z T Z^{T}, \qquad T_{11} X - X T_{22} = -T_{12}, \qquad \Pi = (Z_2 + Z_1 X) Z_2^{T},

    with :math:`\Pi` the spectral projector onto the invariant subspace of the selected values along the
    complementary one. Blocks are selected by their diagonal position, never by their value.

    :param form: The Schur form the Ritz values were read from, with their diagonal positions.
    :param selected: Which Ritz values are selected, in the order of the form's positions.
    :param log: Callable receiving the refusal message, or ``None`` to stay silent.
    :return: The columns :math:`Z_2 + Z_1 X` of the invariant subspace, the columns :math:`Z_2` and the trailing
        block :math:`T_{22}` (:math:`B C = C T_{22}`); ``None`` when the subspace cannot be split off, each refusal
        logged with its reason when ``log`` is given.
    """

    def refuse(reason: str) -> None:
        """
        Logs why the selected subspace cannot be split off.

        :param reason: The refusal, as a clause.
        :return: ``None``, the refused result.
        """
        if log is not None:
            log(f"Jacobian tracker: {reason}, the selected invariant subspace cannot be split off.")
        return None

    if form.positions is None:
        return refuse("a Ritz value could not be located on the Schur form")
    select = np.ones(form.t.shape[0], dtype=np.int32)
    select[form.positions[selected]] = 0
    t_mat, z_mat, _, _, m, _, _, info = dtrsen(select, form.t, form.z, job="N", wantq=1)
    if info != 0:
        return refuse("two adjacent Schur blocks are too close to swap")
    if m != int(select.sum()):
        return refuse("a conjugate pair is selected on one side only")
    t22 = t_mat[m:, m:]
    try:
        x = np.zeros((0, t22.shape[0])) if m == 0 else solve_sylvester(t_mat[:m, :m], -t22, -t_mat[:m, m:])
    except np.linalg.LinAlgError:
        return refuse("the Sylvester equation could not be solved")
    if float(np.linalg.norm(x)) > _COND_CAP:
        return refuse(f"the Sylvester solution norm exceeds {_COND_CAP:g}")
    z_selected = z_mat[:, m:]
    return z_selected + z_mat[:, :m] @ x, z_selected, t22


def _nonuniform_map(
    q_kept: np.ndarray,
    form: SchurForm,
    theta: np.ndarray,
    lam_pi: np.ndarray,
    flip: np.ndarray,
    gated: np.ndarray,
    log: Callable[[str], None] | None = None,
    band_damping: float | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None:
    r"""
    Builds the per-direction damping of an estimate on the invariant subspace of its certified directions, separated
    from the uncertified one by :func:`_invariant_columns`. With the thin QR :math:`Q_K C = Q_c R_c` of its columns
    :math:`C`, :math:`\Pi_{\mathrm{cert}} = Q_c\,W_c` with :math:`W_c = R_c Z_c^{T} Q_K^{T}` is the OBLIQUE projector
    onto it, and the map is :math:`N = \sum_\alpha s_\alpha p_\alpha \Pi_\alpha`, the commuting upper-triangular map
    of :func:`_commuting_upper` on :math:`B_c = R_c T_{22} R_c^{-1}`; both annihilate an uncertified Ritz direction,
    which keeps the step the mixing gave it. Blocks closer than ``_SCHUR_TOL`` share one damping, coupled blocks are
    capped by :func:`_bounded_map`, and a projector above ``_OBLIQUE_CAP`` falls back to the orthogonal
    :math:`Q_c Q_c^{T}`.

    :param q_kept: The orthonormal basis of the kept secant increments, shape ``[n_real, k]``.
    :param form: The Schur form of the projected Jacobian :math:`B` on that basis, with the Ritz positions.
    :param theta: Its Ritz values, in the order ``lam_pi``, ``flip`` and ``gated`` are given in.
    :param lam_pi: The eigenvalues :math:`\lambda_\Pi` of the Ritz directions.
    :param flip: Which of them are flipped.
    :param gated: Which of them passed the Ritz residual gate.
    :param log: Callable receiving the fallback and refusal messages, or ``None`` to stay silent.
    :param band_damping: The damping of a certified block of undecidable negative sign, the scalar damping of the
        damped step (see :func:`_direction_damping`); ``None`` gives the block its own :math:`p_\alpha`.
    :return: The triple of the orthonormal span of the certified subspace, the dual rows of the projector onto it
        and the damping map in the span's coordinates, together with the signed damping the map applies to each
        mode (``nan`` on an uncertified one); ``None`` when no direction is certified or when the certified
        subspace cannot be split off the uncertified one.
    """
    if not gated.any():
        return None
    split = _invariant_columns(form, gated, log)
    if split is None:
        if log is not None:
            log(
                "Jacobian tracker: the certified invariant subspace is not separable from the uncertified one, "
                "no per-direction damping installed."
            )
        return None
    columns, z_c, t22 = split
    coords, upper = np.linalg.qr(columns)
    bounds, diagonal = _schur_blocks(t22)
    starts = [low for low, _ in bounds]
    ordered = np.flatnonzero(gated)[np.argsort(form.positions[gated])]
    damping = _direction_damping(lam_pi[ordered[starts]], flip[ordered[starts]], band_damping)
    eigen = diagonal[starts]
    tol = _SCHUR_TOL * max(1.0, float(np.max(np.abs(theta))))
    tied = [(j, i) for i in range(len(eigen)) for j in range(i) if abs(eigen[i] - eigen[j]) <= tol]
    try:
        b_c = solve_triangular(upper.T, (upper @ t22).T, lower=True).T
        bounded = _bounded_map(lambda values: _commuting_upper(b_c, bounds, values), damping, bounds, tied, log)
    except np.linalg.LinAlgError:
        if log is not None:
            log("Jacobian tracker: the restricted Jacobian of the certified span is unusable, no per-direction damping")
        return None
    if bounded is None:
        return None
    m_mat, block_damping = bounded
    applied = np.full(theta.size, np.nan)
    for (low, high), value in zip(bounds, block_damping):
        applied[ordered[low:high]] = value
    span, dual = q_kept @ coords, upper @ z_c.T
    if float(np.linalg.norm(dual, 2)) > _OBLIQUE_CAP:
        if log is not None:
            log(
                "Jacobian tracker: the certified and the uncertified subspace are closer than the obliquity cap "
                f"{_OBLIQUE_CAP:g}, the orthogonal projector is used on the certified span."
            )
        return (span, span.T, m_mat), applied
    return (span, dual @ q_kept.T, m_mat), applied


def _carried_nonuniform_map(
    lam_pi: np.ndarray,
    u: np.ndarray,
    flip: np.ndarray,
    log: Callable[[str], None] | None = None,
    band_damping: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[()] | None:
    r"""
    Builds the per-direction damping of a carried set, which has Ritz vectors but no projected matrix: the real
    columns of every carried mode (:func:`_real_columns`) are orthonormalized by one column-pivoted QR and the whole
    carried span counts as certified, with the orthogonal projector onto it. A column whose diagonal entry falls more
    than ``_COND_CAP`` below the leading one is dropped, which refuses the set when that column is flipped. Each kept
    column takes the damping of its mode, capped by :func:`_bounded_map` with the two columns of a conjugate pair
    tied; a mixed-sign set above the cap returns ``_SIGNS_UNBOUNDED``, a refusal of the map alone, since the set may
    still act through its reflector.

    :param lam_pi: The carried eigenvalues :math:`\lambda_\Pi`.
    :param u: Their normalized Ritz vectors as columns.
    :param flip: Which of them are flipped.
    :param log: Callable receiving the dropped-column, fallback and refusal messages, or ``None`` to stay silent.
    :param band_damping: The damping of a mode of undecidable negative sign, the scalar damping of the damped step;
        ``None`` gives it its own :math:`p_\alpha` like every other mode.
    :return: The span, the dual rows of the orthogonal projector onto it and the damping map in its coordinates;
        ``_SIGNS_UNBOUNDED`` when the signs of the kept columns carry no bounded weighting; ``None`` when no
        column was collected or a flipped column is dependent on the kept ones beyond ``_COND_CAP``.
    """
    stacked, owners = _real_columns(1.0 - lam_pi, u, np.ones(lam_pi.size, dtype=bool))
    if stacked.shape[1] == 0:
        return None
    span, upper, pivots = qr(stacked, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(upper))
    kept = int(np.count_nonzero(diagonal > diagonal[0] / _COND_CAP))
    dropped = pivots[kept:]
    if kept == 0 or bool(flip[owners[dropped]].any()):
        return None
    if dropped.size and log is not None:
        log(
            f"Jacobian tracker: {dropped.size} stable carried column(s) of lambda_Pi "
            f"{', '.join(f'{lam_pi[owner].real:+.4f}' for owner in owners[dropped])} lie in the span of the "
            f"others beyond the condition cap {_COND_CAP:g} and are dropped."
        )
    damping = _direction_damping(lam_pi, flip, band_damping)
    columns = owners[pivots[:kept]]
    coords = upper[:kept, :kept]
    tied = [(j, i) for i in range(kept) for j in range(i) if columns[i] == columns[j]]
    bounds = [(col, col + 1) for col in range(kept)]
    bounded = _bounded_map(
        lambda values: np.linalg.solve(coords.T, (coords * values).T).T.real, damping[columns], bounds, tied, log
    )
    if bounded is None:
        return _SIGNS_UNBOUNDED
    span = np.ascontiguousarray(span[:, :kept])
    return span, span.T, bounded[0]


def _conjugate_partners(lam: np.ndarray) -> np.ndarray:
    """
    Returns the index of the complex-conjugate partner of every value of an array, ``-1`` for a real value (imaginary
    part within ``_REAL_CUTOFF``) and for a complex value whose partner is missing.

    :param lam: The values.
    :return: The partner index of each value.
    """
    partners = np.full(lam.size, -1, dtype=int)
    cutoff = _REAL_CUTOFF * (1.0 + np.abs(lam))
    for idx in np.flatnonzero(lam.imag > cutoff):
        free = np.flatnonzero((lam.imag < -cutoff) & (partners < 0))
        if free.size:
            match = free[np.argmin(np.abs(lam[free] - np.conj(lam[idx])))]
            if abs(lam[match] - np.conj(lam[idx])) <= cutoff[idx]:
                partners[idx], partners[match] = match, idx
    return partners


def _mode_planes(lam: np.ndarray, u: np.ndarray) -> tuple[list[tuple[int, int]], list[np.ndarray]]:
    r"""
    Groups Ritz modes into the real subspaces their identity is followed on: the real line of a real mode and the
    real plane :math:`\{\mathrm{Re}\,u, \mathrm{Im}\,u\}` of a complex-conjugate pair, owned by the member with the
    positive imaginary part. A lone conjugate member and a mode whose columns span nothing belong to no group.
    Every basis is computed in double precision, whatever precision the vectors are stored in.

    :param lam: The Ritz values.
    :param u: Their Ritz vectors as columns.
    :return: The members of each group as ``(mode, partner)`` with ``partner = -1`` for a real mode, and the
        orthonormal ``float64`` basis of each group's subspace.
    """
    partners = _conjugate_partners(lam)
    cutoff = _REAL_CUTOFF * (1.0 + np.abs(lam))
    groups, bases = [], []
    for idx in range(lam.size):
        if abs(lam[idx].imag) <= cutoff[idx]:
            columns = u[:, idx].real[:, None]
        elif partners[idx] >= 0 and lam[idx].imag > 0.0:
            columns = np.column_stack([u[:, idx].real, u[:, idx].imag])
        else:
            continue
        basis, upper = np.linalg.qr(columns.astype(np.float64, copy=False))
        diagonal = np.abs(np.diag(upper))
        if diagonal.max() > 0.0 and diagonal.min() > _COLLINEAR_CUTOFF * diagonal.max():
            groups.append((idx, int(partners[idx])))
            bases.append(basis)
    return groups, bases


def _match_modes(lam_new: np.ndarray, u_new: np.ndarray, lam_old: np.ndarray, u_old: np.ndarray) -> np.ndarray:
    r"""
    Matches the modes of one estimate to those of an earlier one by the overlap of their Ritz vectors, one-to-one and
    largest overlap first, a match needing an overlap above ``_FLIP_OVERLAP``. A real mode is matched to a real mode
    by :math:`|u^{T} u'|`; a complex-conjugate pair is matched to a pair by the cosine of the smallest principal angle
    between their real planes :math:`\{\mathrm{Re}\,u, \mathrm{Im}\,u\}` (see :func:`_mode_planes`), and its two
    members are matched together, the member with the positive imaginary part to the one with the positive imaginary
    part. A real mode never matches a pair.

    :param lam_new: The Ritz values of the later estimate.
    :param u_new: Their Ritz vectors as columns.
    :param lam_old: The Ritz values of the earlier estimate.
    :param u_old: Their Ritz vectors as columns.
    :return: For each later mode the index of its earlier match, or ``-1``.
    """
    prev = np.full(lam_new.size, -1, dtype=int)
    groups_new, bases_new = _mode_planes(lam_new, u_new)
    groups_old, bases_old = _mode_planes(lam_old, u_old)
    if not groups_new or not groups_old:
        return prev
    overlap = np.zeros((len(groups_new), len(groups_old)))
    for row, basis_new in enumerate(bases_new):
        for col, basis_old in enumerate(bases_old):
            if basis_new.shape[1] == basis_old.shape[1]:
                overlap[row, col] = np.linalg.svd(basis_new.T @ basis_old, compute_uv=False)[0]
    while True:
        row, col = np.unravel_index(np.argmax(overlap), overlap.shape)
        if overlap[row, col] <= _FLIP_OVERLAP:
            return prev
        for mode_new, mode_old in zip(groups_new[row], groups_old[col]):
            if mode_new >= 0:
                prev[mode_new] = mode_old
        overlap[row, :] = 0.0
        overlap[:, col] = 0.0


def _extended_set(
    mode_set: tuple[np.ndarray, np.ndarray, np.ndarray],
    lam_est: np.ndarray | None,
    u_est: np.ndarray | None,
    take: np.ndarray,
    replace: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a set of modes extended, as flipped modes, by the taken modes of an estimate that match none of its own
    (see :func:`_match_modes`). With ``replace`` a taken mode that matches a member the set does not flip replaces
    that member, value and vector, as a flipped one, so a later estimate overrules a member an earlier one found
    stable; a match to a flipped member changes nothing.

    :param mode_set: The Ritz values, the Ritz vectors as columns and the flip decisions of the set.
    :param lam_est: The Ritz values of the estimate (``None`` without one).
    :param u_est: Their Ritz vectors as columns.
    :param take: Which modes of the estimate may join.
    :param replace: Whether a taken mode replaces the unflipped member it matches.
    :return: The Ritz values, vectors and flip decisions of the extended set, or ``mode_set`` itself when nothing
        joins or is replaced.
    """
    lam_pi, u, flip = mode_set
    if not np.any(take):
        return mode_set
    match = _match_modes(lam_est, u_est, lam_pi, u)
    add = take & (match < 0)
    swap = take & (match >= 0)
    swap[swap] = replace & ~flip[match[swap]]
    if not (add.any() or swap.any()):
        return mode_set
    lam_pi = np.concatenate([lam_pi, lam_est[add]])
    u = np.column_stack([u, u_est[:, add]])
    flip = np.concatenate([flip, np.ones(int(add.sum()), dtype=bool)])
    lam_pi[match[swap]], u[:, match[swap]], flip[match[swap]] = lam_est[swap], u_est[:, swap], True
    return lam_pi, u, flip


def predict_crossing(
    lam_now: np.ndarray, lam_prev: np.ndarray, lam_prev2: np.ndarray, res_now: np.ndarray, res_prev: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Extrapolates the real part of every chained mode one update ahead and decides whether that crosses the stability
    boundary. From three consecutive matched certified values :math:`\mathrm{Re}\,\lambda^{(n-2)}`,
    :math:`\mathrm{Re}\,\lambda^{(n-1)}`, :math:`\mathrm{Re}\,\lambda^{(n)}` of one mode,

    .. math::

        \mathrm{Re}\,\lambda^{\mathrm{pred}} = 2\,\mathrm{Re}\,\lambda^{(n)} - \mathrm{Re}\,\lambda^{(n-1)}, \qquad
        \epsilon_{\mathrm{fit}} = \Big|\mathrm{Re}\,\lambda^{(n-1)} - \tfrac12\big(\mathrm{Re}\,\lambda^{(n)}
        + \mathrm{Re}\,\lambda^{(n-2)}\big)\Big|, \qquad
        \delta = \max\big(\mathrm{res}^{(n)}, \mathrm{res}^{(n-1)}\big) + 2\,\epsilon_{\mathrm{fit}},

    and the mode is a predicted crossing when :math:`\mathrm{Re}\,\lambda^{(n)} > -m`, the three values decrease
    monotonically and :math:`\mathrm{Re}\,\lambda^{\mathrm{pred}} < -(m + \delta)`, with :math:`m = \max(m_0,
    \mathrm{res}^{(n)})` the undecidable band of :func:`classify`. A mode whose modulus exceeds ``_POLE_MODULUS``
    and grew on both matched updates is approaching a pole; it is reported instead of predicted (see
    :func:`detect_pole_jump`).

    :param lam_now: The certified eigenvalues :math:`\lambda_\Pi` of this update.
    :param lam_prev: The matched values of the previous informative update, ``nan`` where a mode has no match.
    :param lam_prev2: The matched values two informative updates back, ``nan`` where the chain is shorter.
    :param res_now: The Ritz residuals of this update.
    :param res_prev: The matched residuals of the previous update, ``nan`` where a mode has no match.
    :return: The predicted-crossing mask, the extrapolated real parts, the error bars :math:`\delta` and the mask of
        the pole-type approaches.
    """
    band = np.maximum(_MARGIN, res_now)
    re_now, re_prev, re_prev2 = lam_now.real, lam_prev.real, lam_prev2.real
    chained = np.isfinite(re_prev) & np.isfinite(re_prev2)
    pred = 2.0 * re_now - re_prev
    fit = np.abs(re_prev - 0.5 * (re_now + re_prev2))
    delta = np.maximum(res_now, res_prev) + 2.0 * fit
    growing = (np.abs(lam_now) > np.abs(lam_prev)) & (np.abs(lam_prev) > np.abs(lam_prev2))
    approach = chained & (np.abs(lam_now) > _POLE_MODULUS) & growing
    monotone = (re_now < re_prev) & (re_prev < re_prev2)
    predicted = chained & ~approach & (re_now > -band) & monotone & (pred < -(band + delta))
    return predicted, pred, delta, approach


def detect_pole_jump(
    lam_now: np.ndarray, lam_prev: np.ndarray, lam_prev2: np.ndarray, res_now: np.ndarray, res_prev: np.ndarray
) -> np.ndarray:
    r"""
    Detects a mode that has just passed through a pole: its real part changes from certified positive to certified
    negative between two consecutive matched updates, both values lie beyond ``_POLE_MODULUS``, and the step of
    :math:`1/\lambda_\Pi`, which is continuous through the pole while the eigenvalue is not, continues the step
    before it,

    .. math::

        \big|1/\lambda^{(n)} - 1/\lambda^{(n-1)}\big| \leq (1 + c)\,\big|1/\lambda^{(n-1)} - 1/\lambda^{(n-2)}\big|,

    with :math:`c` the constant ``_POLE_CONTINUITY``; the identity across the pole rests on the eigenvector overlap
    of the chain (see :func:`_match_modes`). A mode before the jump cannot be flipped ahead of it.

    :param lam_now: The certified eigenvalues :math:`\lambda_\Pi` of this update.
    :param lam_prev: The matched values of the previous informative update, ``nan`` where a mode has no match.
    :param lam_prev2: The matched values two informative updates back, ``nan`` where the chain is shorter.
    :param res_now: The Ritz residuals of this update.
    :param res_prev: The matched residuals of the previous update, ``nan`` where a mode has no match.
    :return: The mask of the modes that crossed a pole on this update.
    """
    band_now, band_prev = np.maximum(_MARGIN, res_now), np.maximum(_MARGIN, res_prev)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_now, inv_prev, inv_prev2 = 1.0 / lam_now, 1.0 / lam_prev, 1.0 / lam_prev2
        step, step_prev = np.abs(inv_now - inv_prev), np.abs(inv_prev - inv_prev2)
    beyond = (np.abs(lam_now) > _POLE_MODULUS) & (np.abs(lam_prev) > _POLE_MODULUS)
    sign_change = (lam_prev.real > band_prev) & (lam_now.real < -band_now)
    continuous = np.isfinite(step_prev) & (step <= (1.0 + _POLE_CONTINUITY) * step_prev)
    return beyond & sign_change & continuous


def predict_rung_crossing(
    lam: np.ndarray, lam_prev: np.ndarray, res: np.ndarray, res_prev: np.ndarray, ratio: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Extrapolates a carried mode linearly in the inverse temperature from its values at the two preceding rungs
    :math:`\beta_1 < \beta_2` to the new rung :math:`\beta_3`, with :math:`r = (\beta_3 - \beta_2) / (\beta_2 -
    \beta_1)` the step ratio,

    .. math::

        \mathrm{Re}\,\lambda^{\mathrm{pred}} = \mathrm{Re}\,\lambda^{(2)} + \big(\mathrm{Re}\,\lambda^{(2)}
        - \mathrm{Re}\,\lambda^{(1)}\big)\,r, \qquad
        \delta = \max\big(\mathrm{res}^{(1)}, \mathrm{res}^{(2)}\big)\,(1 + |r|),

    and marks the mode for a flip ahead of its crossing when :math:`\mathrm{Re}\,\lambda^{(2)} > -m`,
    :math:`\mathrm{Re}\,\lambda^{(2)} < \mathrm{Re}\,\lambda^{(1)}` and :math:`\mathrm{Re}\,\lambda^{\mathrm{pred}}
    < -(m + \delta)`, with :math:`m = \max(m_0, \mathrm{res}^{(2)})` the undecidable band of :func:`classify`. A
    mode heading for a pole is marked as well when :math:`\mathrm{Re}(1/\lambda)`, extrapolated linearly in the
    same way, changes sign before :math:`\beta_3` while :math:`|\lambda^{(2)}| > |\lambda^{(1)}| >` ``_POLE_MODULUS``:
    the new rung then sits past the pole. For such a mode the returned prediction is the inverse of the extrapolated
    :math:`\mathrm{Re}(1/\lambda)`.

    :param lam: The carried eigenvalues :math:`\lambda_\Pi` at :math:`\beta_2`.
    :param lam_prev: Their matched values at :math:`\beta_1`, ``nan`` where a mode has no match.
    :param res: The Ritz residuals at :math:`\beta_2`.
    :param res_prev: The matched residuals at :math:`\beta_1`, ``nan`` where a mode has no match.
    :param ratio: The step ratio :math:`r`, one value or one per mode.
    :return: The mask of the modes to flip ahead of their crossing, the extrapolated real parts and the mask of the
        modes flipped for lying past a pole.
    """
    band = np.maximum(_MARGIN, res)
    matched = np.isfinite(lam_prev.real)
    re_now, re_prev = lam.real, lam_prev.real
    pred = re_now + (re_now - re_prev) * ratio
    delta = np.maximum(res, res_prev) * (1.0 + np.abs(ratio))
    stable_side = re_now > -band
    predicted = matched & stable_side & (re_now < re_prev) & (pred < -(band + delta))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_now, inv_prev = (1.0 / lam).real, (1.0 / lam_prev).real
        inv_pred = inv_now + (inv_now - inv_prev) * ratio
        past_pole = 1.0 / inv_pred
    pole = matched & stable_side & (np.abs(lam) > np.abs(lam_prev)) & (np.abs(lam_prev) > _POLE_MODULUS)
    pole = pole & (inv_now * inv_pred < 0.0)
    return predicted | pole, np.where(pole, past_pole, pred), pole


def load_spectrum(path: str) -> dict[str, np.ndarray]:
    """
    Loads the Jacobian file ``jacobian.npz`` a previous run wrote (see :meth:`JacobianTracker.save_spectrum`), the
    certified spectrum with the per-iteration traces. A file a run rewrote every iteration but never finished holds
    the traces alone, without the spectrum keys.

    :param path: Path of the ``.npz`` file.
    :return: Dictionary of its arrays.
    """
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


class JacobianTracker:
    r"""
    Tracks the leading Jacobian eigenpairs of the proposal map inside the running self-consistency and owns the
    resulting flip state.

    Every monitoring step re-derives the spectrum from the raw (iterate, proposal) pairs, which do not depend on
    whether a reflection was applied afterwards, so the flip set is re-decided from scratch. While a reflector is
    installed the proposal residual is reflected on the invariant subspace of the flipped directions
    (:meth:`reflect`), and a DAMPED step is corrected on the certified span afterwards (:meth:`stabilize_step`). The
    install, prediction and release rules are those of :meth:`update_recorded`.

    :attr:`p_eff` is the largest damping the certified spectrum allows, never above the configured value and never
    raised again within a run. :attr:`estimated` tells whether the last :meth:`update_recorded` call produced a fresh
    spectrum estimate.
    """

    def __init__(
        self,
        p_config: float,
        logger: "DgaLogger | None" = None,
        to_sector: Callable[[np.ndarray], np.ndarray] | None = None,
        from_sector: Callable[[np.ndarray], np.ndarray] | None = None,
        exact_checks: bool = True,
    ):
        """
        Initializes an inactive tracker: nothing flipped, no spectrum measured yet, damping at its configured value.
        Every vector the tracker keeps lives in the real coordinates of ``to_sector`` (an isometry such as the
        symmetry sector of :func:`~dgamore.sigma_jacobian.tracker_sector_maps`), its Ritz vectors in the complex
        precision of the recorded window (see :meth:`record`).

        :param p_config: The configured damping, i.e. the ceiling of :attr:`p_eff`.
        :param logger: Logger receiving the measured spectrum, or ``None`` to stay silent.
        :param to_sector: Map of a complex window array to its real vector in the tracker's coordinates, ``None``
            for :func:`to_vec`.
        :param from_sector: Map of a real vector in the tracker's coordinates back to a complex array holding the
            window's entries in their order, ``None`` for the inverse of :func:`to_vec` onto a flat array.
        :param exact_checks: Whether the caller runs the exact checks the tracker requests (see
            :meth:`exact_check_request`); without them no request is formed and no start column is kept.
        """
        self._to_sector = to_vec if to_sector is None else to_sector
        self._from_sector = (lambda vector: to_mat(vector, (-1,))) if from_sector is None else from_sector
        self.p_config = self.p_eff = float(p_config)
        self.q = self.w = self.last_lam_pi = self.last_res = self.last_gated = None
        self.n_updates, self.estimated = 0, False
        self._logger = logger
        self._flip = self._last_u = None
        self._persist = self._calm = self._grow = 0
        self._hold = 0  # updates left before the estimate may install its map again after a growth release
        self._last_residual_norm = np.inf
        self._floor_warned = self._saturation_warned = False
        self._exact_checks = exact_checks
        self._single = False  # whether the recorded window is single precision, which the Ritz vectors then follow
        self._nonuniform = None  # (orthonormal span, dual rows, damping map) of the certified directions
        self._applied = None  # signed damping the map built from the estimate applies per mode; None on a carried map
        self._pending = None  # carried (lam_pi, u, flip, label) waiting for flips to be allowed
        self._carried_set = None  # carried (lam_pi, u, flip) in force from a predecessor run or an exact check
        self._set_label = "carried"  # what installed the set in force, "carried" or "exact-check", for the log lines
        self._check_candidate = None  # (lam_pi, u) of the last informative update's least-stable mode below the band
        self._check_request = None  # (start columns, description, counted) of the exact check due before this step
        self._start_check = None  # carried columns whose exact check waits for the first update allowing flips
        self._checks_done = 0  # mid-rung exact checks requested so far, less the ones given back as clearly stable
        self._rearmed = 0  # mid-rung checks given back so far, at most _MAX_EXACT_CHECKS
        self._counted_check = False  # whether the last request handed out is a counted mid-rung check
        self._since_check = _PERSIST  # informative updates since the last requested exact check
        self._last_certified = None  # (lam_pi, res, gated, flip, u, predicted) of the last update that certified a mode
        self._snapshot_kind = None  # what made that snapshot: "estimate", "carried" or "exact"
        self._held_flips = None  # (lam_pi, res, u, predicted) of reflected flips the current snapshot lacks
        self._history = []  # the last _CHAIN_LENGTH informative estimates, matched mode by mode (see _predict)
        self._predicted = None  # the predicted-crossing mask of the last estimate, aligned with last_lam_pi
        self._predecessor = None  # (lam_pi, res, beta, u) of the predecessor's carried columns on this window

    @property
    def active(self) -> bool:
        """Whether a flipped subspace or a per-direction damping map is currently installed."""
        return self.q is not None or self._nonuniform is not None

    def _stored(self, u: np.ndarray) -> np.ndarray:
        """
        Returns Ritz vectors in the precision the tracker keeps them in: complex64 when the recorded window is single
        precision, the given array otherwise.

        :param u: The Ritz vectors as columns.
        :return: The vectors to keep.
        """
        return u.astype(np.complex64, copy=False) if self._single else u

    def spectrum_state(self, shape: tuple[int, ...], beta: float, converged: bool, dtype: np.dtype) -> dict:
        r"""
        Collects the certified part of the latest snapshot for a successor run: the last update that certified a mode
        with flips allowed, the set :meth:`carry_in` received, the last :meth:`install_exact_check` or the spectrum
        of :meth:`install_exact_spectrum`, whichever came last, plus the flips the installed reflector still holds
        from an earlier snapshot (see :meth:`_replace_snapshot`). Per mode it stores the Ritz value, residual, flip
        decision and prediction flag, the values of the predecessor mode it matches (``lam_prev``, ``res_prev``,
        ``beta_prev``, matched by :func:`_match_modes` to the columns :meth:`carry_in` kept, ``nan`` without a match)
        and, below ``_STORE_BAND`` (``_CHECK_BAND`` for an exact snapshot), the Ritz vector as a real and an imaginary
        column in the tracker's coordinates. ``flip`` and ``p_eff`` are recorded for inspection only; ``exact`` tells
        whether the snapshot comes from the exact Jacobian.

        :param shape: The complex window shape ``(kx, ky, kz, nb, nb, 2 niv_core)`` of the iterated array.
        :param beta: Inverse temperature :math:`\beta` of the run.
        :param converged: Whether the run reached the pure fixed point.
        :param dtype: Complex precision the loop keeps its self-energies in; the columns are stored in its real
            precision, so the file never holds more or less of a vector than the run itself does.
        :return: The state.
        """
        if self._last_certified is None:
            lam_pi, res = np.zeros(0, dtype=np.complex128), np.zeros(0, dtype=np.float64)
            flip, u = np.zeros(0, dtype=bool), np.zeros((0, 0), dtype=np.complex128)
            predicted = np.zeros(0, dtype=bool)
        else:
            lam_pi, res, gated, flip, u, predicted = self._last_certified
            lam_pi, res = lam_pi[gated].astype(np.complex128), res[gated].astype(np.float64)
            flip, u, predicted = flip[gated].astype(bool), u[:, gated], predicted[gated].astype(bool)
            if self._held_flips is not None:
                held = self._held_flipped(self._held_flips[2])
                lam_h, res_h, u_h, predicted_h = (part[..., held] for part in self._held_flips)
                lam_pi, res, u = np.concatenate([lam_pi, lam_h]), np.concatenate([res, res_h]), np.hstack([u, u_h])
                flip = np.concatenate([flip, np.ones(lam_h.size, dtype=bool)])
                predicted = np.concatenate([predicted, predicted_h])
        exact = self._snapshot_kind == "exact"
        stored = lam_pi.real < (_CHECK_BAND if exact else _STORE_BAND)
        lam_prev = np.full(lam_pi.size, np.nan, dtype=np.complex128)
        res_prev, beta_prev = np.full(lam_pi.size, np.nan), np.full(lam_pi.size, np.nan)
        # the carried set itself is not matched against the columns it was built from
        if self._predecessor is not None and lam_pi.size and self._snapshot_kind != "carried":
            lam_p, res_p, beta_p, u_p = self._predecessor
            match = _match_modes(lam_pi, u, lam_p, u_p)
            hit = match >= 0
            lam_prev[hit], res_prev[hit], beta_prev[hit] = lam_p[match[hit]], res_p[match[hit]], beta_p
        real = np.zeros(0, dtype=dtype).real.dtype
        return {
            "lam_pi": lam_pi,
            "res": res,
            "flip": flip,
            "predicted": predicted,
            "stored": stored,
            "lam_prev": lam_prev,
            "res_prev": res_prev,
            "beta_prev": beta_prev,
            "u_re": u[:, stored].real.astype(real),
            "u_im": u[:, stored].imag.astype(real),
            "shape": np.asarray(shape, dtype=np.int64),
            "beta": np.float64(beta),
            "p_eff": np.float64(self.p_eff),
            "converged": np.bool_(converged),
            "exact": np.bool_(exact),
        }

    def save_spectrum(
        self,
        path: str,
        shape: tuple[int, ...],
        beta: float,
        converged: bool,
        dtype: np.dtype,
        traces: dict[str, np.ndarray] | None = None,
    ) -> None:
        r"""
        Writes :meth:`spectrum_state` and the per-iteration ``traces`` to ``path`` as one uncompressed ``.npz``, also
        for a run that certified nothing. The file is written to a temporary name, forced to disk and renamed onto
        ``path``, so a run killed while writing leaves the previous file intact.

        :param path: Destination file.
        :param shape: The complex window shape of the iterated array.
        :param beta: Inverse temperature :math:`\beta` of the run.
        :param converged: Whether the run reached the pure fixed point.
        :param dtype: Complex precision the vector columns are stored in (see :meth:`spectrum_state`).
        :param traces: Further arrays written beside the spectrum, keyed by name; ``None`` writes the spectrum alone.
        :return: None.
        """
        state = self.spectrum_state(shape, beta, converged, dtype)
        partial = f"{path}.tmp"
        with open(partial, "wb") as handle:
            np.savez(handle, **state, **(traces or {}))
        _flush_and_drop(partial)
        os.replace(partial, path)
        self._log_info(
            f"Jacobian tracker: saved {state['lam_pi'].size} certified modes ({int(state['stored'].sum())} with "
            f"vectors, {int(np.isfinite(state['lam_prev'].real).sum())} matched to the predecessor's, "
            f"{int(state['predicted'].sum())} flipped by prediction) to {path}."
        )

    def install_exact_spectrum(self, lam_pi: np.ndarray, res: np.ndarray, u: np.ndarray) -> None:
        r"""
        Makes an exactly computed spectrum the snapshot :meth:`spectrum_state` writes for a successor run, in place of
        the secant estimate: the eigenvalues :math:`\lambda_\Pi` of the exact Jacobian at the converged point (see
        :func:`~dgamore.sigma_jacobian.leading_eigenpairs`) with their residual bounds and eigenvectors, every mode
        certified, the flips decided by :func:`classify` and none predicted. The reflector, the damping map and
        :attr:`p_eff` in force are left as they are. An exact mode past the flip margin, :math:`\mathrm{Re}\,\lambda_\Pi
        < -\max(m, \mathrm{res})`, that the installed reflector does not hold (see :meth:`_held_flipped`) is unstable
        for the damped iteration at the converged point: it is warned about, and the successor run carries it as a flip.

        :param lam_pi: The eigenvalues :math:`\lambda_\Pi = 1 - \theta` of the modes.
        :param res: Their residual bounds.
        :param u: Their normalized complex eigenvectors as columns, in the tracker's coordinates (those of
            :class:`~dgamore.sigma_jacobian.ExactJacobian`); kept in the precision of the recorded window.
        :return: None.
        """
        lam_pi, res = np.asarray(lam_pi, dtype=np.complex128), np.asarray(res, dtype=np.float64)
        gated = np.ones(lam_pi.size, dtype=bool)
        flip, _ = classify(lam_pi, gated, self.p_config, res)
        u = self._stored(np.asarray(u, dtype=np.complex128))
        self._replace_snapshot((lam_pi, res, gated, flip, u, np.zeros(lam_pi.size, dtype=bool)), "exact")
        kept = 0 if self._held_flips is None else self._held_flips[0].size
        self._log_info(
            f"Jacobian tracker: installed the exact spectrum of {lam_pi.size} modes, {int(flip.sum())} of them past "
            f"the flip margin{f', beside {kept} reflected flip(s) no exact mode matches' if kept else ''}."
        )
        unheld = flip & ~self._held_flipped(u)
        if unheld.any() and self._logger is not None:
            values = ", ".join(f"{lam:+.4f}" for lam in lam_pi[unheld])
            self._logger.warning(
                "Jacobian tracker: the exact spectrum at the converged point has unstable modes the installed "
                f"reflector does not hold, lambda_Pi {values}: the damped iteration is unstable there, so the run "
                "converged by another route (a stall, an accelerated step or a slowly contracting component)."
            )

    def install_exact_check(self, lam_pi: np.ndarray, res: np.ndarray, u: np.ndarray) -> bool:
        r"""
        Acts at once on the Ritz pairs of an exact check (:func:`~dgamore.sigma_jacobian.certify_subspace`): the
        pairs whose exact residual passes the Ritz gate :math:`\mathrm{res} \leq g \max(1, |1 - \lambda_\Pi|)` are
        flipped by :func:`classify`, bind :attr:`p_eff`, become the exact snapshot (see :meth:`_replace_snapshot`)
        and are installed as an exact-check set (:meth:`_install_carried`) together with the reflected flips no pair
        matches (:meth:`_reflected_flips`). Pairs that fail the gate change nothing. A counted mid-rung check (see
        :meth:`_note_check_candidate`) that certifies its lead, the pair of the smallest real part, and finds every
        certified pair stable, :math:`\mathrm{Re}\,\lambda_\Pi > \max(m, \mathrm{res})`, gives its slot of the
        ``_MAX_EXACT_CHECKS`` budget back, at most ``_MAX_EXACT_CHECKS`` times per run.

        :param lam_pi: The eigenvalues :math:`\lambda_\Pi` of the check's Ritz pairs.
        :param res: Their exact residuals.
        :param u: Their normalized complex eigenvectors as columns, in the tracker's coordinates (those of
            :class:`~dgamore.sigma_jacobian.ExactJacobian`); the certified ones are kept in the precision of the
            recorded window.
        :return: Whether the reflected map switched, i.e. whether the reflected subspace changed.
        """
        lam_pi, res = np.asarray(lam_pi, dtype=np.complex128), np.asarray(res, dtype=np.float64)
        gated = res <= _RITZ_GATE * np.maximum(1.0, np.abs(1.0 - lam_pi))
        counted, self._counted_check = self._counted_check, False
        if not gated.any():
            self._log_info("Jacobian tracker: the exact check certified no pair, nothing installed.")
            return False
        lead_certified = bool(gated[np.argmin(lam_pi.real)])
        lam_pi, res, u = lam_pi[gated], res[gated], self._stored(np.asarray(u, dtype=np.complex128)[:, gated])
        stable = lead_certified and np.all(lam_pi.real > np.maximum(_MARGIN, res))
        if counted and stable and self._rearmed < _MAX_EXACT_CHECKS:
            self._checks_done, self._rearmed = self._checks_done - 1, self._rearmed + 1
            self._log_info("Jacobian tracker: the exact check found every certified pair stable, its slot re-armed.")
        certified = np.ones(lam_pi.size, dtype=bool)
        flip, p_new = classify(lam_pi, certified, self.p_config, res)
        self._lower_p_eff(p_new)
        held = self._reflected_flips()
        q_before = self.q
        merged = (lam_pi, u, flip)
        if held is not None:
            merged = _extended_set(merged, held[0], held[2], np.ones(held[0].size, dtype=bool))
        switched = self._install_carried(*merged, label="exact-check") and self._switched(q_before)
        self._replace_snapshot((lam_pi, res, certified, flip, u, np.zeros(lam_pi.size, dtype=bool)), "exact")
        return switched

    def _reflected_flips(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        r"""
        Collects the certified flipped modes of the snapshot together with the ones kept beside it (see
        :meth:`_replace_snapshot`) that the installed reflector holds (see :meth:`_held_flipped`).

        :return: Their eigenvalues :math:`\lambda_\Pi`, error bounds, Ritz vectors as columns (in the precision of
            :meth:`_stored`, also for a snapshot carried from a file of higher precision) and prediction flags;
            ``None`` without a reflector or without such a mode.
        """
        if self.q is None:
            return None
        parts = [] if self._held_flips is None else [self._held_flips]
        if self._last_certified is not None:
            lam_pi, res, gated, flip, u, predicted = self._last_certified
            take = gated & flip
            parts.append((lam_pi[take], res[take], u[:, take], predicted[take]))
        if not parts:
            return None
        lam_pi, res, u, predicted = (np.concatenate(group, axis=-1) for group in zip(*parts))
        held = self._held_flipped(u)
        return (lam_pi[held], res[held], self._stored(u[:, held]), predicted[held]) if held.any() else None

    def _replace_snapshot(self, snapshot: tuple, kind: str) -> None:
        """
        Makes ``snapshot`` the one :meth:`spectrum_state` writes, and keeps beside it the certified flips the
        installed reflector holds (:meth:`_reflected_flips`, read before the replacement) that the new snapshot does
        not match (:func:`_match_modes`), with the value, error bound, vector and prediction flag of the snapshot that
        certified them, since a flipped direction contracts out of the secant window.

        :param snapshot: The ``(lam_pi, res, gated, flip, u, predicted)`` of the new snapshot.
        :param kind: What made it, ``"estimate"``, ``"carried"`` or ``"exact"``.
        :return: None.
        """
        held = self._reflected_flips()
        self._last_certified, self._snapshot_kind, self._held_flips = snapshot, kind, None
        if held is None:
            return
        lam_pi, _, gated, _, u, _ = snapshot
        new = _match_modes(held[0], held[2], lam_pi[gated], u[:, gated]) < 0
        if new.any():
            self._held_flips = (held[0][new], held[1][new], held[2][:, new], held[3][new])

    def _switched(self, q_before: np.ndarray | None) -> bool:
        """
        Whether the reflected subspace differs from another one: installed or released, of another dimension, or
        turned by more than ``_SUBSPACE_ANGLE``, the largest principal angle between the two.

        :param q_before: The other orthonormal basis, ``None`` for no reflector.
        :return: Whether the reflected map switched.
        """
        if q_before is None or self.q is None:
            return (q_before is None) != (self.q is None)
        if q_before.shape[1] != self.q.shape[1]:
            return True
        overlap = np.linalg.svd(self.q.T @ q_before, compute_uv=False)
        return float(np.arccos(np.clip(overlap.min(), -1.0, 1.0))) > _SUBSPACE_ANGLE

    def carry_in(
        self,
        state: dict,
        expand: Callable[[np.ndarray], np.ndarray],
        allow_flip: bool = True,
        beta: float | None = None,
    ) -> bool:
        r"""
        Installs the certified spectrum a predecessor run ended with, before the first step of this run.

        Every stored Ritz vector is rebuilt from its two columns through ``expand``, as a real vector plus ``1j``
        times another, normalized and held in the complex precision of the stored columns; columns that come back
        empty are dropped. A carried mode is flipped by the flip rule of :func:`classify`, or ahead of its crossing by
        the two-rung extrapolation in :math:`\beta` of :meth:`_rung_prediction`, the preemptive flip arXiv:2606.04936
        recommends for a carried Jacobian. A matched mode of a converged predecessor whose real part lay within
        ``_STORE_BAND`` of zero at :math:`\beta_1` and rose again at :math:`\beta_2` by more than its error bar, the
        cusp of Phys. Rev. Lett. doi:10.1103/zjy7-4jqd Sec. VI, is warned about and never flipped. :attr:`p_eff` is
        lowered to the bound of the carried spectrum at once, also when the install (:meth:`_install_carried`) waits
        for a scaffold's release, and the damping the predecessor ended with is not read back; a carried spectrum
        without a certified mode leaves the damping where it is. A converged predecessor's columns are kept for
        :meth:`spectrum_state` to match against. The real columns of an exact file's least stable modes, at most
        ``_CHECK_START`` and never half a conjugate pair, wait for the first update that allows flips as an exact
        check request (see :meth:`exact_check_request`), unless the tracker runs without exact checks.

        :param state: The dictionary :func:`load_spectrum` returns.
        :param expand: Callable mapping one stored column to the real vector in this run's tracker coordinates.
        :param allow_flip: Whether the carried set may be installed now; otherwise it is kept pending and installed
            on the first update that allows flips while nothing else is installed.
        :param beta: Inverse temperature :math:`\beta` of this run, i.e. the :math:`\beta_3` of the extrapolation;
            ``None`` skips the extrapolation.
        :return: Whether the carried set is installed.
        """
        lam_pi = np.asarray(state["lam_pi"], dtype=np.complex128)
        res = np.asarray(state["res"], dtype=np.float64)
        converged, beta_last = bool(state["converged"]), float(state["beta"])
        # every key is read here, before the tracker changes, so a file lacking one raises with the tracker untouched
        exact = bool(state["exact"])
        stored, u_re, u_im = np.flatnonzero(np.asarray(state["stored"], dtype=bool)), state["u_re"], state["u_im"]
        previous = (
            np.asarray(state["lam_prev"], dtype=np.complex128),
            np.asarray(state["res_prev"], dtype=np.float64),
            np.asarray(state["beta_prev"], dtype=np.float64),
        )
        if lam_pi.size == 0:
            self._log_info(
                "Jacobian tracker: the carried spectrum holds no certified mode, nothing carried; "
                f"p_eff={self.p_eff:.4f}."
            )
            return False
        _, p_new = classify(lam_pi, np.ones(lam_pi.size, dtype=bool), self.p_config, res)
        self._lower_p_eff(p_new)
        if converged and self._logger is not None:
            lam_prev, res_prev, beta_prev = previous
            near = np.abs(lam_prev.real) < _STORE_BAND
            for idx in np.flatnonzero(near & (lam_pi.real - lam_prev.real > np.maximum(res, res_prev))):
                self._logger.warning(
                    f"Jacobian tracker: the carried lambda_Pi={lam_pi[idx]:+.4f} at "
                    f"beta={beta_last:g} rose again from {lam_prev[idx].real:+.4f} at beta={beta_prev[idx]:g}, "
                    "close to the boundary: a mode that touched the boundary between the rungs and receded, which can "
                    "mean the predecessor converged beyond a crossing; it is carried as measured, not flipped."
                )
        u, norms = None, np.zeros(stored.size)
        for col in range(stored.size):
            column = expand(u_re[:, col]) + 1j * expand(u_im[:, col])
            if u is None:
                u = np.empty((column.size, stored.size), dtype=np.result_type(u_re.dtype, np.complex64))
            u[:, col] = column
            norms[col] = np.linalg.norm(u[:, col])
        keep = norms > 0.0
        self._log_info(
            f"Jacobian tracker: carried {lam_pi.size} certified modes "
            f"(lambda_Pi {', '.join(f'{lam:+.4f}' for lam in lam_pi)}), "
            f"{int(keep.sum())} with a usable vector, p_eff={self.p_eff:.4f}."
        )
        if not keep.any():
            return False
        u = u if keep.all() else u[:, keep]
        u /= norms[keep]
        indices = stored[keep]
        lam_c, res_c = lam_pi[indices], res[indices]
        flip = lam_c.real < -np.maximum(_MARGIN, res_c)
        predicted, lam_map = np.zeros(lam_c.size, dtype=bool), lam_c
        if _PREDICT_ACROSS_RUNGS and beta is not None:
            predicted, lam_map = self._rung_prediction(
                previous, beta_last, converged, indices, lam_pi, res, float(beta)
            )
            flip = flip | predicted
        if converged:
            self._predecessor = (lam_c, res_c, beta_last, u)
        if exact and self._exact_checks:
            columns, owners = _real_columns(1.0 - lam_c, u, np.ones(lam_c.size, dtype=bool))
            cut = min(_CHECK_START, owners.size)
            # a conjugate pair is checked from both of its columns or not at all
            cut -= int(0 < cut < owners.size and owners[cut] == owners[cut - 1])
            self._start_check = np.ascontiguousarray(columns[:, :cut]) if cut else None
        self.last_lam_pi, self.last_res, self._last_u = lam_c, res_c, u
        self._flip, self.last_gated = flip, np.ones(lam_c.size, dtype=bool)
        self._replace_snapshot((lam_c, res_c, self.last_gated, flip, u, predicted), "carried")
        if not allow_flip:
            self._pending = (lam_map, u, flip, "carried")
            self._log_info("Jacobian tracker: flips are paused by a scaffold, the carried flip waits for its release.")
            return False
        return self._install_carried(lam_map, u, flip)

    def _rung_prediction(
        self,
        previous: tuple[np.ndarray, np.ndarray, np.ndarray],
        beta_last: float,
        converged: bool,
        indices: np.ndarray,
        lam_pi: np.ndarray,
        res: np.ndarray,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Decides which carried modes with vectors are flipped ahead of their crossing by the two-rung extrapolation of
        :func:`predict_rung_crossing`, and lowers the damping to the bound of the carried spectrum with the
        extrapolated real part on the predicted modes. Refused, with a log line, for a predecessor that did not reach
        the pure fixed point and for a mode whose step ratio :math:`(\beta_3 - \beta_2) / (\beta_2 - \beta_1)`
        exceeds ``_MAX_BETA_STEP_RATIO`` in modulus.

        :param previous: The values, error bounds and inverse temperatures of the modes the file's modes match at
            the predecessor's own predecessor (``lam_prev``, ``res_prev``, ``beta_prev``).
        :param beta_last: The predecessor's inverse temperature.
        :param converged: Whether the predecessor reached the pure fixed point.
        :param indices: The positions of the carried modes with vectors among the file's certified modes.
        :param lam_pi: All certified eigenvalues of the file.
        :param res: Their Ritz residuals.
        :param beta: Inverse temperature :math:`\beta` of this run.
        :return: The mask of the carried modes with vectors that are flipped ahead of their crossing, and the
            eigenvalues the carried map is built from (the predicted ones on those modes, the carried ones else).
        """
        none = np.zeros(indices.size, dtype=bool)
        lam_c, res_c = lam_pi[indices], res[indices]
        lam_prev, res_prev, beta_prev = (values[indices] for values in previous)
        matched = np.isfinite(lam_prev.real) & np.isfinite(beta_prev)
        if not matched.any():
            return none, lam_c
        if not converged:
            self._log_info(
                "Jacobian tracker: the predecessor did not reach the pure fixed point, no cross-rung prediction."
            )
            return none, lam_c
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(matched, (beta - beta_last) / (beta_last - beta_prev), np.nan)
        over = matched & ~(np.abs(ratio) <= _MAX_BETA_STEP_RATIO)
        for idx in np.flatnonzero(over):
            self._log_info(
                f"Jacobian tracker: lambda_Pi={lam_c[idx]:+.4f} has a step ratio "
                f"{ratio[idx]:.2f} in beta above the cap {_MAX_BETA_STEP_RATIO:g}, no cross-rung prediction."
            )
        predicted, pred, pole = predict_rung_crossing(lam_c, lam_prev, res_c, res_prev, np.where(over, np.nan, ratio))
        if not predicted.any():
            return predicted, lam_c
        lam_map = lam_c.copy()
        lam_map[predicted] = pred[predicted] + 1j * lam_c[predicted].imag
        lam_pred = lam_pi.copy()
        lam_pred[indices] = lam_map
        _, p_pred = classify(lam_pred, np.ones(lam_pi.size, dtype=bool), self.p_config, res)
        self._lower_p_eff(p_pred, predicted=True)
        for idx in np.flatnonzero(predicted):
            self._log_info(
                f"Jacobian tracker: cross-rung prediction: lambda_Pi={lam_c[idx]:+.4f} at beta={beta_last:g} from "
                f"{lam_prev[idx]:+.4f} at beta={beta_prev[idx]:g} extrapolates to {pred[idx]:+.4f} at beta={beta:g}"
                f"{' (past the pole)' if pole[idx] else ''}, flipped ahead of its crossing."
            )
        return predicted, lam_map

    def _install_carried(
        self, lam_pi: np.ndarray, u: np.ndarray, flip: np.ndarray | None = None, label: str = "carried"
    ) -> bool:
        r"""
        Builds and installs the per-direction damping of the carried set (:func:`_carried_nonuniform_map`) beside
        the orthogonal reflector :math:`\mathbb{1} - 2 Q Q^{T}` of the real columns the flipped carried Ritz vectors
        span (see :func:`_real_columns`); a set without a flipped mode installs its map alone, one whose signs carry
        no bounded weighting its reflector and the scalar damping bound alone. A mode of undecidable negative sign
        takes the current :attr:`p_eff` in the map, as it does in the estimate's own map. Resets the persistence,
        calm and residual-growth counters and clears the hold a growth release left behind. A set that collects no
        column at all, or whose flipped columns are mutually dependent beyond ``_COND_CAP``, installs nothing.

        :param lam_pi: The carried Ritz values of the modes with vectors.
        :param u: Their normalized Ritz vectors as columns.
        :param flip: Which of them are flipped; ``None`` flips every one.
        :param label: What the set comes from, ``"carried"`` for a predecessor's set or ``"exact-check"`` for the
            result of an exact check, as the log lines of its install and of its release name it.
        :return: Whether the carried set was installed.
        """
        flip = np.ones(lam_pi.size, dtype=bool) if flip is None else np.asarray(flip, dtype=bool)
        nonuniform = _carried_nonuniform_map(lam_pi, u, flip, self._log_info, self.p_eff)
        stacked = _real_columns(1.0 - lam_pi, u, flip)[0]
        self._pending = None
        reflector = None
        if stacked.shape[1]:
            singular = np.linalg.svd(stacked, compute_uv=False)
            if singular[-1] > 0.0 and singular[0] / singular[-1] <= _COND_CAP:
                q_mat = np.ascontiguousarray(np.linalg.qr(stacked)[0])
                reflector = (q_mat, q_mat.T)
        unbounded = nonuniform is _SIGNS_UNBOUNDED
        if nonuniform is None or (flip.any() and reflector is None) or (unbounded and reflector is None):
            self._log_info(
                f"Jacobian tracker: no column was collected from the {label} modes, or a flipped one lies in the "
                f"span of the others beyond the condition cap {_COND_CAP:g}, nothing installed."
            )
            return False
        self.q, self.w = reflector if reflector is not None else (None, None)
        self._nonuniform, self._applied = (None if unbounded else nonuniform), None
        self._carried_set, self._set_label = (lam_pi, u, flip), label
        self._persist = self._calm = self._grow = self._hold = 0
        reflected = 0 if self.q is None else self.q.shape[1]
        if unbounded:
            self._log_info(
                f"Jacobian tracker: {label} set installed on its reflector alone over {reflected} "
                f"direction{'s' if reflected != 1 else ''}, its signs carrying no per-direction damping."
            )
        else:
            self._log_info(
                f"Jacobian tracker: {label} set installed on {nonuniform[0].shape[1]} directions, "
                f"{reflected} of them reflected."
            )
        return True

    def record(self, iterate: np.ndarray, proposal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Returns one (iterate, proposal) pair as the entry :meth:`update_recorded` reads: both arrays in the
        tracker's coordinates, stored in the real precision of the iterate, and the noise floor of the iterate in its
        storage precision (see :func:`noise_floor`). The Ritz vectors the tracker computes from then on are kept in
        that precision too.

        :param iterate: The iterate as a complex core-window array.
        :param proposal: Its proposal: the raw one for the tracker's own window, the reflected one for the
            accelerated-mixing history.
        :return: The iterate's and the proposal's real vectors and the iterate's noise floor.
        """
        real = np.asarray(iterate).real.dtype
        self._single = real == np.float32
        vector = self._to_sector(iterate)
        floor = noise_floor(vector, real)
        return vector.astype(real, copy=False), self._to_sector(proposal).astype(real, copy=False), floor

    def update(self, iterates: list[np.ndarray], proposals: list[np.ndarray], allow_flip: bool = True) -> bool:
        """
        Runs :meth:`update_recorded` on the window of the given pairs (see :meth:`record`).

        :param iterates: The iterates as complex core-window arrays of one shape, oldest first.
        :param proposals: The matching raw proposals, never reflected ones, oldest first.
        :param allow_flip: See :meth:`update_recorded`.
        :return: See :meth:`update_recorded`.
        """
        pairs = zip(iterates[-TRACKER_PAIRS:], proposals[-TRACKER_PAIRS:])
        return self.update_recorded([self.record(x, f) for x, f in pairs], allow_flip=allow_flip)

    def update_recorded(self, entries: list[tuple[np.ndarray, np.ndarray, float]], allow_flip: bool = True) -> bool:
        r"""
        Runs one monitoring step on the recorded pairs and updates the flip state.

        The tracked subspace and the damping are frozen while the last iterate has zero norm, the last step lies below
        its noise floor (see :func:`noise_floor`) or the secant window keeps no informative increment, and a window of
        fewer than ``_MIN_PAIRS`` pairs also clears the chained history; a single surviving increment still certifies
        the dominant mode. Each Ritz value is certified by its residual and read with its error bound, the residual
        times its condition number (see :func:`_ritz_from_increments`). A certified unstable mode on
        ``_FLIP_PERSIST`` consecutive updates, or one whose chained estimates predict a crossing or show a pole jump
        (see :meth:`_predict`), installs the reflector of :meth:`_flipped_reflector`; it replaces a carried set and
        extends an exact-check set (see :func:`_extended_set`). The per-direction map is rebuilt on every certifying
        update and left in force by one that certifies nothing; a flipped mode enters it with the reversed sign only
        once a reflector holds it and stays out of it until then, and a mode of undecidable negative sign takes the
        damping of the damped step. A reflector, the tracker's own or a carried one, is released with its map once a
        certified STABLE mode lies inside it on ``_PERSIST`` consecutive updates without a flip candidate. Everything
        installed is released once the raw residual grows on ``_PERSIST`` consecutive updates, which outranks an
        install on that update, holds the estimate's map and new reflectors off for ``_PERSIST`` informative updates
        and, where a reflector was released, clears the persistence streak. A pending carried set is installed on the
        first update that allows flips while nothing is installed, and the columns of an exact carried set become a
        check request there (see :meth:`carry_in`). A certifying update with flips allowed becomes the snapshot of
        :meth:`_replace_snapshot`, and every informative update notes its least-stable mode for an exact check (see
        :meth:`_note_check_candidate`).

        :param entries: The recorded pairs (see :meth:`record`), oldest first; the newest ``TRACKER_PAIRS`` are read.
        :param allow_flip: Whether the measured spectrum may be acted on. Without it the spectrum is still measured
            but no flip is acted on, every streak is reset and whatever is installed is released, a carried or
            exact-check set going back to pending; the snapshot is left as it is.
        :return: Whether the reflected map the mixing sees switched; a lowered damping or a change of the
            per-direction map alone is no switch.
        """
        self.estimated = False
        if allow_flip and self._start_check is not None:
            count = self._start_check.shape[1]
            what = f"the {count} carried direction{'s' if count != 1 else ''}"
            self._check_request, self._start_check = (self._start_check, what, False), None
        event, released = False, ""
        if not allow_flip:
            self._persist = self._calm = self._grow = 0
            if self.active:
                event = self.q is not None
                if self._carried_set is not None:
                    self._pending = (*self._carried_set, self._set_label)
                    released = f"{self._set_label} set pending until flips are allowed again, tracked basis released"
                else:
                    what = "tracked basis" if event else "per-direction damping"
                    released = f"flips suppressed (allow_flip=False), {what} released"
                self.q = self.w = self._nonuniform = self._applied = self._carried_set = None

        def frozen(message: str) -> bool:
            """
            Logs why this update carries no estimate, after the release it made, if any.

            :param message: What stopped the estimate.
            :return: Whether the reflected map switched on this update.
            """
            self._log_info(f"Jacobian tracker: {released + '; ' if released else ''}{message}")
            return event

        if len(entries) < _MIN_PAIRS:
            self._history, self._check_candidate = [], None
            return frozen(f"insufficient history ({len(entries)} pairs), no estimate.")

        entries = entries[-TRACKER_PAIRS:]
        x_last, x_prev = entries[-1][0], entries[-2][0]
        x_last_norm = float(np.linalg.norm(x_last.astype(np.float64, copy=False)))
        if x_last_norm == 0.0:
            return frozen("last iterate has zero norm, below the roundoff floor, tracker frozen.")
        floor = entries[-1][2]
        step = float(np.linalg.norm(np.subtract(x_last, x_prev, dtype=np.float64)))
        if step < floor:
            return frozen(f"step {step / x_last_norm:.2e} below the roundoff floor, tracker frozen.")

        residual_norm = float(np.linalg.norm(np.subtract(entries[-1][1], x_last, dtype=np.float64)))
        dx, df = (_increments([entry[part] for entry in entries]) for part in (0, 1))
        theta, u, res, q_kept, form, kappa = _ritz_from_increments(dx, df, floor)
        del dx, df
        u = self._stored(u)
        if theta.size == 0:
            return frozen("secant window has no informative increments, tracker frozen.")

        self.last_lam_pi, self._last_u = 1.0 - theta, u
        gated = self.last_gated = res <= _RITZ_GATE * np.maximum(1.0, np.abs(theta))
        # past the gate a Ritz value is read with its error bound, the backward error times its condition number
        res = self.last_res = res * kappa

        flip, p_new = classify(self.last_lam_pi, self.last_gated, self.p_config, res)
        predicted, jump = self._predict(self.last_lam_pi, res, u, gated, allow_flip)
        flip = flip | predicted
        self._predicted = predicted
        forced = bool(predicted.any() or jump.any())
        reflected = False  # whether a reflector of the persisted flip was built on this update
        # the hold a growth release of the map leaves behind counts down once per update
        hold_free = may_install = self._hold == 0
        self._hold = max(self._hold - 1, 0)
        if not allow_flip:
            flip[:] = False
            p_new = self.p_eff
        else:
            self._warn_saturation(theta.size, gated, flip)
            installed = False
            if self._pending is not None and not self.active:
                # a carried install brings its map with it, whatever hold a growth release left behind; only a
                # reflector switches the map the accelerated mixing sees
                installed = self._install_carried(*self._pending)
                event = installed and self.q is not None
                may_install = may_install or installed
            self._persist = self._persist + 1 if flip.any() else 0
            if forced and hold_free:
                # a predicted crossing or a pole jump stands in for the streak, unless a growth release holds
                self._persist = max(self._persist, _FLIP_PERSIST)
            # a reflector holds until a certified STABLE mode is seen inside it (or the residual grows)
            calm_now = self.q is not None and not flip.any()
            if calm_now:
                band = np.maximum(_MARGIN, res[gated])
                calm_now = bool((self._held_flipped(u[:, gated]) & (self.last_lam_pi[gated].real > band)).any())
            self._calm = self._calm + 1 if calm_now else 0
            # a set installed on this update restarts the growth count, as the tracker's own install below does
            grew = self.active and not installed and residual_norm > self._last_residual_norm
            self._grow = self._grow + 1 if grew else 0
            if self.active and (self._grow >= _PERSIST or self._calm >= _PERSIST):
                event = event or self.q is not None
                if self._grow >= _PERSIST:
                    what = "tracked basis" if event else "per-direction damping"
                    released = f"residual grew on consecutive updates, {what} released"
                    # a growth release outranks a persisting flip, so a re-install needs _PERSIST fresh certifications
                    self._persist, self._hold = (0 if event else self._persist), _PERSIST
                else:
                    basis = "tracked basis" if self._carried_set is None else f"{self._set_label} basis"
                    released = f"a certified stable mode lies in the {basis}, {basis} released"
                self.q = self.w = self._nonuniform = self._applied = self._carried_set = None
                self._grow = 0
                may_install = False
            elif flip.any() and self._persist >= _FLIP_PERSIST and hold_free:
                extend_exact = self._carried_set is not None and self._set_label == "exact-check"
                reflector = None if extend_exact else self._flipped_reflector(flip, q_kept, form)
                if extend_exact:
                    # an exact-check set is extended by the flips the estimate adds, never replaced by them
                    if (flip & ~self._held_flipped(u)).any():
                        q_before = self.q
                        merged = _extended_set(self._carried_set, self.last_lam_pi, u, flip & gated, replace=True)
                        if merged is not self._carried_set and self._install_carried(*merged, label="exact-check"):
                            may_install, event = True, self._switched(q_before)
                elif reflector is None:
                    self._log_info(
                        "Jacobian tracker: the flipped directions could not be assembled into a reflector, none "
                        "installed."
                    )
                else:
                    reflected = True
                    if self._carried_set is not None or self._switched(reflector[0]):
                        self.q, self.w = reflector
                        self._carried_set = None
                        # a reflector brings its map with it, whatever hold a growth release left behind, and the
                        # growth interlock judges it from its install on
                        may_install, self._hold, self._grow = True, 0, 0
                        event = True
        self._last_residual_norm = residual_norm

        self._flip = flip
        if allow_flip and may_install and gated.any():
            # the reversed sign enters the map with the reflector that holds it: one built now, or the installed one
            signs = flip if reflected else flip & self._held_flipped(u)
            # a flip no reflector holds is no part of the map either: it keeps the step the mixing gave it
            mapped = gated & ~(flip & ~signs)
            refreshed = None
            if _NONUNIFORM_IN_LOOP:
                # a mode of undecidable negative sign takes the damped step's own damping, which splits nothing off
                band = min(self.p_eff, p_new)
                refreshed = _nonuniform_map(q_kept, form, theta, self.last_lam_pi, signs, mapped, self._log_info, band)
            if self.q is None:
                # without a reflector the map is the only installed object, so nothing carried is left
                self._carried_set = None
                if refreshed is None and self._nonuniform is not None:
                    released = "the certified span carries no per-direction damping, tracked map released"
                elif refreshed is not None and self._nonuniform is None:
                    count = refreshed[0][0].shape[1]
                    self._log_info(
                        f"Jacobian tracker: per-direction damping installed on {count} certified "
                        f"direction{'s' if count != 1 else ''}."
                    )
            self._nonuniform, self._applied = (None, None) if refreshed is None else refreshed
        if allow_flip and gated.any():
            self._replace_snapshot((self.last_lam_pi, res, gated, flip, u, predicted), "estimate")
        self._lower_p_eff(p_new)
        self._note_check_candidate(self.last_lam_pi, gated, u, allow_flip)

        self.n_updates += 1
        self.estimated = True
        self._log_spectrum(flip, released)
        return event

    def _warn_saturation(self, size: int, gated: np.ndarray, flip: np.ndarray) -> None:
        """
        Warns once per run when every Ritz value of a full secant window (``TRACKER_PAIRS - 1`` of them) is certified
        and flipped: the window is then saturated with unstable directions, and more of them than it holds may exist
        outside it.

        :param size: The number of Ritz values of the estimate.
        :param gated: Which of them are certified.
        :param flip: Which of them are flipped.
        :return: None.
        """
        if self._saturation_warned or self._logger is None or size < TRACKER_PAIRS - 1:
            return
        if gated.all() and flip.all():
            self._saturation_warned = True
            self._logger.warning(
                f"Jacobian tracker: every Ritz value of a full window of {size} is certified and flipped; the window "
                "may hold fewer unstable directions than the map has, and the ones outside it are not reflected."
            )

    def _predict(
        self, lam_pi: np.ndarray, res: np.ndarray, u: np.ndarray, gated: np.ndarray, allow_flip: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Chains the certified modes of this estimate to those of the previous informative updates by the overlap of
        their Ritz vectors (:func:`_match_modes`), keeps the last ``_CHAIN_LENGTH`` estimates (cleared by the window
        restart of :meth:`update_recorded`), and reads the predicted crossings (:func:`predict_crossing`) and the pole
        jumps (:func:`detect_pole_jump`) off the chains. Every prediction, pole jump and pole-type approach is logged,
        also while flips are paused.

        :param lam_pi: The eigenvalues :math:`\lambda_\Pi` of this estimate.
        :param res: Their Ritz residuals.
        :param u: Their Ritz vectors as columns.
        :param gated: Which of them are certified.
        :param allow_flip: Whether the measured spectrum may be acted on.
        :return: The predicted-crossing mask and the pole-jump mask, both over all modes of the estimate.
        """
        cert = np.flatnonzero(gated)
        lam_c, res_c = lam_pi[cert], res[cert]
        prev = np.full(cert.size, -1, dtype=int)
        if self._history:
            last = self._history[-1]
            prev = _match_modes(lam_c, u[:, cert], last["lam"], last["u"][:, last["cert"]])
            last["u"] = None
        entry = {"lam": lam_c, "res": res_c, "u": u, "cert": cert, "prev": prev}
        self._history = (self._history + [entry])[-_CHAIN_LENGTH:]
        predicted, jump = np.zeros(lam_pi.size, dtype=bool), np.zeros(lam_pi.size, dtype=bool)
        if not _PREDICT_IN_LOOP or cert.size == 0 or len(self._history) < 2:
            return predicted, jump
        lam_1 = np.full(cert.size, np.nan, dtype=np.complex128)
        lam_2, res_1 = lam_1.copy(), np.full(cert.size, np.nan)
        last = self._history[-2]
        older = self._history[-3] if len(self._history) >= 3 else None
        for idx, match in enumerate(prev):
            if match >= 0:
                lam_1[idx], res_1[idx] = last["lam"][match], last["res"][match]
                if older is not None and last["prev"][match] >= 0:
                    lam_2[idx] = older["lam"][last["prev"][match]]
        crossing, pred, delta, approach = predict_crossing(lam_c, lam_1, lam_2, res_c, res_1)
        jumped = detect_pole_jump(lam_c, lam_1, lam_2, res_c, res_1)
        predicted[cert], jump[cert] = crossing, jumped
        acted = ", flips paused" if not allow_flip else ", flipped ahead of its crossing"
        for idx in np.flatnonzero(crossing):
            self._log_info(
                f"Jacobian tracker: predicted crossing of lambda_Pi={lam_c[idx]:+.4f} (extrapolated {pred[idx]:+.4f}, "
                f"band {max(_MARGIN, res_c[idx]):.4f}, error bar {delta[idx]:.4f}){acted}."
            )
        for idx in np.flatnonzero(jumped):
            self._log_info(
                f"Jacobian tracker: pole crossing of lambda_Pi from {lam_1[idx].real:+.4f} to {lam_c[idx].real:+.4f}"
                f"{', flips paused' if not allow_flip else ', flipped at once'}."
            )
        for idx in np.flatnonzero(approach):
            self._log_info(
                f"Jacobian tracker: pole-type approach of lambda_Pi={lam_c[idx]:+.4f} "
                f"(modulus {abs(lam_c[idx]):.2f} growing), no prediction."
            )
        return predicted, jump

    def _note_check_candidate(self, lam_pi: np.ndarray, gated: np.ndarray, u: np.ndarray, allow_flip: bool) -> None:
        r"""
        Requests an exact check (see :meth:`exact_check_request`) of the least-stable mode of an estimate, the smallest
        real part of :math:`\lambda_\Pi` (one value or both members of a complex pair), when the mode fails the Ritz
        gate, its real part lies below ``_STORE_BAND`` and it matches (:func:`_match_modes`) the least-stable mode of
        the previous informative update, which lay below the band as well. Nothing is requested while flips are paused
        or another request is pending, beyond the ``_MAX_EXACT_CHECKS`` budget (see :meth:`install_exact_check`),
        within ``_PERSIST`` informative updates of the last request, while an exact-check set is in force, for a mode
        the installed reflector already carries, or without exact checks. The start columns are the real plane of the
        mode (:func:`_real_columns`).

        :param lam_pi: The eigenvalues :math:`\lambda_\Pi` of the estimate.
        :param gated: Which of them are certified.
        :param u: Their Ritz vectors as columns.
        :param allow_flip: Whether the measured spectrum may be acted on.
        :return: None.
        """
        self._since_check += 1
        if not self._exact_checks:
            return
        lead = int(np.argmin(lam_pi.real))
        mode = np.flatnonzero(np.isclose(lam_pi, lam_pi[lead]) | np.isclose(lam_pi, np.conj(lam_pi[lead])))
        below = allow_flip and lam_pi[lead].real < _STORE_BAND
        previous, self._check_candidate = self._check_candidate, (lam_pi[mode], u[:, mode]) if below else None
        if not below or previous is None or gated[mode].all() or self._check_request is not None:
            return
        if self._checks_done >= _MAX_EXACT_CHECKS or self._since_check < _PERSIST:
            return
        if self._carried_set is not None and self._set_label == "exact-check":
            return
        lead_u = np.asarray(u[:, mode], dtype=np.complex128)
        norms = np.linalg.norm(lead_u, axis=0)
        if self.q is not None and np.all(np.linalg.norm(self.w @ lead_u, axis=0) > _FLIP_OVERLAP * norms):
            return
        if not np.all(_match_modes(lam_pi[mode], u[:, mode], *previous) >= 0):
            return
        columns = _real_columns(1.0 - lam_pi, u, np.isin(np.arange(lam_pi.size), mode))[0]
        if not columns.shape[1]:
            return
        self._checks_done, self._since_check = self._checks_done + 1, 0
        values = ", ".join(f"{lam:+.4f}" for lam in lam_pi[mode])
        self._check_request = (columns, f"the uncertified lambda_Pi={values} matched over two updates", True)

    def exact_check_request(self) -> np.ndarray | None:
        """
        Hands out the directions an exact check should certify before this iteration's step and clears the request:
        the carried columns of an exact predecessor spectrum (see :meth:`carry_in`) or the real plane of a
        least-stable mode the estimate sees without certifying it (see :meth:`_note_check_candidate`). The caller
        runs the check and hands its pairs to :meth:`install_exact_check`.

        :return: The real start columns in the tracker's coordinates, or ``None`` when no check is due.
        """
        if self._check_request is None:
            return None
        (columns, what, self._counted_check), self._check_request = self._check_request, None
        self._log_info(f"Jacobian tracker: exact check requested for {what}.")
        return columns

    def release_after_loop(self) -> None:
        """
        Releases what only the self-consistency loop reads, once it has ended, and keeps what the end of a run reads:
        the reflector's basis, the snapshot with the flips kept beside it, the predecessor's columns and
        :attr:`p_eff`. The tracker reflects, stabilizes and updates no more afterwards.

        :return: None.
        """
        self.w = self._nonuniform = self._applied = self._carried_set = self._pending = None
        self._start_check = self._check_request = self._check_candidate = None
        self.last_lam_pi = self.last_res = self.last_gated = self._last_u = self._flip = self._predicted = None
        self._history = []

    def start_directions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        r"""
        Returns two real start directions for an exact eigen-solve (see
        :func:`~dgamore.sigma_jacobian.leading_eigenpairs`), read from the certified modes of the snapshot
        :meth:`spectrum_state` writes: the real part of the Ritz vector of the least stable mode, the smallest
        :math:`\mathrm{Re}\,\lambda_\Pi`, and of the stiffest one, the largest :math:`|1 - \lambda_\Pi|`.

        :return: The two directions as ``float64`` vectors in the tracker's coordinates; ``None`` for a direction
            without a certified mode or whose real part vanishes.
        """
        if self._last_certified is None or not self._last_certified[2].any():
            return None, None
        lam_pi, _, gated, _, u, _ = self._last_certified
        certified = np.flatnonzero(gated)
        picks = (np.argmin(lam_pi[certified].real), np.argmax(np.abs(1.0 - lam_pi[certified])))
        directions = [u[:, certified[pick]].real.astype(np.float64) for pick in picks]
        return tuple(vector if np.any(vector) else None for vector in directions)

    def reflect(self, proposal: np.ndarray, iterate: np.ndarray) -> np.ndarray:
        r"""
        Reflects the residual :math:`r = S(x) - x` of the proposal on the tracked invariant subspace, returning

        .. math:: \tilde{S}(x) = x + r - 2\,Q\,(W r),

        which reverses the residual of every flipped direction and leaves every other one exactly where it was. The
        reflector carries no damping (see :meth:`stabilize_step`), and the map shares its fixed points with the
        plain map.

        :param proposal: The raw proposal :math:`S(x)` on the core window.
        :param iterate: The current iterate :math:`x` on the same window.
        :return: The reflected proposal in the dtype of ``proposal``, or ``proposal`` itself while no subspace is
            tracked.
        """
        if self.q is None:
            return proposal
        residual = self._to_sector(proposal) - self._to_sector(iterate)
        correction = np.reshape(self._from_sector(-2.0 * (self.q @ (self.w @ residual))), proposal.shape)
        return np.add(proposal, correction, out=correction).astype(proposal.dtype, copy=False)

    def opposes_flip(self, step: np.ndarray, residual: np.ndarray) -> bool:
        r"""
        Returns whether a step moves against the flipped damped step on the reflected subspace: since :math:`W Q =
        \mathbb{1}` that step has the coordinates :math:`-p\,W r` there, :math:`r = S(x) - x` the raw residual, so a
        step :math:`\delta x` opposes it when :math:`(W \delta x) \cdot (W r) > 0`. A damped iteration on the
        reflected map converges only to the fixed points the flips make stable, while a quasi-Newton step can be
        drawn to any root of :math:`S(x) - x`.

        :param step: The step :math:`\delta x` as a real vector in the tracker's coordinates.
        :param residual: The raw residual :math:`S(x) - x` on the core window (any array holding the window's
            entries).
        :return: Whether the step opposes the flipped damped step; ``False`` while no reflector is installed.
        """
        if self.q is None:
            return False
        return float((self.w @ step) @ (self.w @ self._to_sector(residual))) > 0.0

    def unfold(self, vector: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """
        Returns a real vector in the tracker's coordinates as the complex window it stands for.

        :param vector: The real vector.
        :param shape: The shape of the window array to return.
        :return: The complex window array.
        """
        return np.reshape(self._from_sector(vector), shape)

    def stabilize_step(self, x_new: np.ndarray, iterate: np.ndarray, residual: np.ndarray) -> np.ndarray:
        r"""
        Overwrites the certified part of a DAMPED iterate with the step the per-direction damping prescribes,

        .. math::

            x_{n+1} = x_{\mathrm{mix}} - \Pi_{\mathrm{cert}}\,(x_{\mathrm{mix}} - x_n) + N\,r, \qquad
            N = \sum_\alpha s_\alpha p_\alpha \Pi_\alpha,

        with :math:`r = S(x_n) - x_n` the raw residual, :math:`\Pi_{\mathrm{cert}} = Q_c W_c` the oblique
        projector onto the certified invariant subspace along the uncertified one (orthogonal where the two are
        too close to separate, and on a carried set) and :math:`\Pi_\alpha` the spectral projector of the
        certified direction :math:`\alpha` inside it. A certified direction therefore iterates with
        :math:`1 - s_\alpha p_\alpha \lambda_{\Pi,\alpha}`, while an uncertified one keeps the step the mixing took.

        :param x_new: The damped iterate on the core window.
        :param iterate: The iterate :math:`x_n` the step started from, on the same window.
        :param residual: The raw residual :math:`S(x_n) - x_n` on the same window.
        :return: The stabilized iterate in the dtype of ``x_new``, or ``x_new`` itself while no map is installed.
        """
        if self._nonuniform is None:
            return x_new
        span, dual, damping = self._nonuniform
        step = self._to_sector(x_new) - self._to_sector(iterate)
        correction = span @ (damping @ (dual @ self._to_sector(residual))) - span @ (dual @ step)
        correction = np.reshape(self._from_sector(correction), x_new.shape)
        return np.add(x_new, correction, out=correction).astype(x_new.dtype, copy=False)

    def _held_flipped(self, u: np.ndarray) -> np.ndarray:
        """
        Whether each Ritz direction already lies in the installed flipped subspace, i.e. whether its cosine with
        the orthonormal basis of the installed reflector exceeds ``_FLIP_OVERLAP``. Nothing is held while no
        reflector is installed.

        :param u: The Ritz vectors as columns.
        :return: The mask of the directions the installed flipped subspace already carries.
        """
        if self.q is None:
            return np.zeros(u.shape[1], dtype=bool)
        u = np.asarray(u, dtype=np.complex128)
        return np.linalg.norm(self.q.T @ u, axis=0) / np.linalg.norm(u, axis=0) > _FLIP_OVERLAP

    def _certified_damping(self) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Returns, for every certified direction of the last estimate, its eigenvalue :math:`\lambda_\Pi` and the
        signed damping it takes on the next step: the damping recorded for its block where the map built from the
        estimate acts on it, :math:`-p_\alpha` where that map carries the reversed sign; on a carried map, which
        records none, the map's Rayleigh quotient where the Ritz vector overlaps the installed span by more than
        ``_FLIP_OVERLAP``; :attr:`p_eff` elsewhere, also on a certified unstable direction the map leaves out (see
        :meth:`update_recorded`) and while no map is installed.

        :return: The certified eigenvalues and their signed dampings, both empty while there is no estimate or no
            certified mode.
        """
        if self.last_lam_pi is None:
            return np.zeros(0, dtype=np.complex128), np.zeros(0)
        gated = self.last_gated
        lam_pi = self.last_lam_pi[gated]
        damping = np.full(lam_pi.size, self.p_eff)
        if self._nonuniform is None or lam_pi.size == 0:
            return lam_pi, damping
        if self._applied is not None:
            applied = self._applied[gated]
            return lam_pi, np.where(np.isnan(applied), damping, applied)
        span, dual, weighting = self._nonuniform
        u = np.asarray(self._last_u[:, gated], dtype=np.complex128)
        coords = span.T @ u
        mapped = np.linalg.norm(coords, axis=0) > _FLIP_OVERLAP
        gain = np.sum(np.conj(coords) * (weighting @ (dual @ u)), axis=0).real / np.linalg.norm(u, axis=0) ** 2
        damping[mapped] = gain[mapped]
        return lam_pi, damping

    def predicted_rate(self) -> float:
        r"""
        Returns the largest contraction factor :math:`|1 - s_\alpha p_\alpha \lambda_{\Pi,\alpha}|` the last
        certified spectrum predicts, with the signed damping of :meth:`_certified_damping`.

        :return: The predicted rate over the certified modes, or ``0.0`` while there is no estimate.
        """
        lam_pi, damping = self._certified_damping()
        return float(np.max(np.abs(1.0 - damping * lam_pi))) if lam_pi.size else 0.0

    @property
    def certified_margin(self) -> float:
        r"""
        The distance of the certified spectrum from the stability boundary,
        :math:`\epsilon = \min_\alpha \max(|\mathrm{Re}\,\lambda_{\Pi,\alpha}|,\, m_\alpha)` over the certified
        modes of the last estimate held, a mode inside its undecidable band :math:`m_\alpha = \max(m,
        \mathrm{res}_\alpha)` of :func:`classify` counting with the band; ``0.0`` while no mode is certified.
        """
        if self.last_lam_pi is None or not self.last_gated.any():
            return 0.0
        real_part = np.abs(self.last_lam_pi[self.last_gated].real)
        return float(np.min(np.maximum(real_part, np.maximum(_MARGIN, self.last_res[self.last_gated]))))

    def minimum_iterations(self, outside_band: bool = False) -> int:
        r"""
        Returns the number of iterations
        :math:`n_{\min} = \min(n_{\mathrm{cap}}, \max_\alpha \lceil 1 / (|\mathrm{Re}\,\lambda_{\Pi,\alpha}|\,
        \min(1, p_\alpha / c)) \rceil)` the slowest certified direction needs to be driven away from the starting
        point, a direction inside its undecidable band counting with the band as in :attr:`certified_margin`, each at
        the modulus of the damping :meth:`_certified_damping` gives it, divided by the constant :math:`c` of the
        damping bound and clipped at one, so a flat direction stepped at :math:`p_\alpha = 1` counts the
        :math:`\lceil 1 / \epsilon \rceil` of App. G4. The division recovers the constant only where the vertex
        formula set the damping; where it is the configured ceiling or the floor instead, the count is off by the
        ratio of the two. The count is capped at ``_MIN_ITERATION_CAP``; zero while no mode is certified, or with
        ``outside_band`` while none lies outside its band.

        :param outside_band: Whether only the certified directions outside their undecidable band count, the ones
            whose distance from the stability boundary the estimate resolves.
        :return: The minimum iteration count.
        """
        lam_pi, damping = self._certified_damping()
        if lam_pi.size == 0:
            return 0
        band = np.maximum(_MARGIN, self.last_res[self.last_gated])
        keep = np.abs(lam_pi.real) > band if outside_band else np.ones(lam_pi.size, dtype=bool)
        if not keep.any():
            return 0
        real_part = np.maximum(np.abs(lam_pi.real[keep]), band[keep])
        rate = real_part * np.minimum(1.0, np.abs(damping[keep]) / _DAMPING_C)
        # taken a relative 1e-9 short, so the last bits of an eigenvalue cannot raise a count that lands on an integer
        counts = np.ceil((1.0 - 1e-9) / np.maximum(rate, np.finfo(np.float64).tiny))
        return min(_MIN_ITERATION_CAP, int(np.max(counts)))

    def leading(self, count: int = 3) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Returns the leading Ritz values :math:`\lambda_\Pi` of the last estimate held (a successful update, or
        the carried set right after :meth:`carry_in`) and their residuals, sorted by descending modulus and
        padded with ``nan`` to length ``count``.

        :param count: The number of leading modes to return.
        :return: The Ritz values (complex, ``nan``-padded) and their residuals (real, ``nan``-padded), each of
            length ``count``, or two all-``nan`` arrays of that length while no estimate has ever been produced.
        """
        lam_pi = np.full(count, np.nan, dtype=np.complex128)
        res = np.full(count, np.nan, dtype=np.float64)
        if self.last_lam_pi is None:
            return lam_pi, res
        order = np.argsort(-np.abs(self.last_lam_pi))[:count]
        lam_pi[: order.size] = self.last_lam_pi[order]
        res[: order.size] = self.last_res[order]
        return lam_pi, res

    def _flipped_reflector(
        self, flip: np.ndarray, q_kept: np.ndarray, form: SchurForm
    ) -> tuple[np.ndarray, np.ndarray] | None:
        r"""
        Builds the reflector of the flipped directions from the invariant subspace they span inside the projected
        Jacobian :math:`B`, the columns :math:`Z_2 + Z_1 X` and :math:`Z_2` of :func:`_invariant_columns`:

        .. math::

            \Pi = (Z_2 + Z_1 X) Z_2^{T}, \qquad R = \mathbb{1} - 2 Q_K \Pi Q_K^{T}.

        :math:`\Pi` is the spectral projector onto the invariant subspace of the flipped eigenvalues along the
        complementary one, so :math:`R` reverses every Ritz direction of a flipped eigenvalue and fixes every other
        one. It is applied as :math:`r - 2 Q (W r)` by :meth:`reflect`, with :math:`Q_K (Z_2 + Z_1 X) = Q R_f` the
        thin QR and :math:`W = R_f Z_2^{T} Q_K^{T}`. A projector whose spectral norm exceeds ``_OBLIQUE_CAP`` is
        replaced by the orthogonal reflection :math:`W = Q^{T}` of the same flipped span, which reverses it just as
        exactly but no longer fixes an overlapping complementary direction.

        :param flip: Which modes are flipped, in the order of the Ritz values.
        :param q_kept: The orthonormal basis of the kept secant increments, shape ``[n_real, k]``.
        :param form: The Schur form of the projected Jacobian :math:`B` on that basis, with the Ritz positions.
        :return: The orthonormal basis :math:`Q` (``[n_real, k_f]``) and the dual rows :math:`W`
            (``[k_f, n_real]``), or ``None`` when nothing is flipped, when the Ritz values could not be located on
            the form, or when the flipped subspace is not separable from its complement.
        """
        if not flip.any():
            return None
        split = _invariant_columns(form, flip, self._log_info)
        if split is None:
            self._log_info(
                "Jacobian tracker: the flipped invariant subspace is not separable from its complement, reflector "
                "not installed."
            )
            return None
        columns, z_flip, _ = split
        q, r_flip = np.linalg.qr(q_kept @ columns)
        q = np.ascontiguousarray(q)
        if float(np.linalg.norm(r_flip @ z_flip.T, 2)) > _OBLIQUE_CAP:
            self._log_info(
                "Jacobian tracker: the flipped and the complementary subspace are closer than the obliquity cap "
                f"{_OBLIQUE_CAP:g}, the orthogonal reflection of the flipped span is used."
            )
            return q, q.T
        return q, np.ascontiguousarray(r_flip @ z_flip.T @ q_kept.T)

    def _lower_p_eff(self, p_new: float, predicted: bool = False) -> None:
        """
        Lowers the effective damping to ``p_new`` when that is smaller than the current value, and never raises it.
        Warns once when the bound of the measured spectrum reaches the floor below the configured damping, also when
        the damping already sits there; a bound set by the predicted spectrum of a carried set warns separately when
        it lowers the damping to the floor, and leaves that one warning to the measured spectrum.

        :param p_new: The candidate damping.
        :param predicted: Whether the bound comes from the predicted spectrum of the carried modes.
        :return: None.
        """
        previous = self.p_eff
        lowered = p_new < previous - 1e-12
        if lowered:
            self.p_eff = p_new
        if self._logger is None or p_new > _MIXING_FLOOR or p_new >= self.p_config:
            return
        if predicted and lowered:
            self._logger.warning(
                f"Jacobian tracker: the effective mixing is at the floor ({_MIXING_FLOOR:g}) from the predicted "
                f"spectrum of the carried modes at this rung's temperature; the carried spectrum itself allowed "
                f"p_eff={previous:.4f}."
            )
        elif not predicted and not self._floor_warned:
            self._floor_warned = True
            pending = self._predicted is not None and bool(np.any(self._predicted))
            self._logger.warning(
                f"Jacobian tracker: the damping bound of the measured spectrum is at the floor "
                f"({_MIXING_FLOOR:g}), so every damped step runs at that damping; a certified direction the floor "
                f"cannot contract (2|Re lambda_Pi|/|lambda_Pi|^2 <= {_MIXING_FLOOR:g}) contracts only where an "
                f"installed per-direction map carries it. Consider continuing from a closer converged state."
                f"{' A predicted crossing is pending on this update.' if pending else ''}"
            )

    def _log_info(self, message: str) -> None:
        """
        Forwards a message to the logger if one was given.

        :param message: The message to log.
        """
        if self._logger is not None:
            self._logger.info(message)

    def _log_spectrum(self, flip: np.ndarray, released: str = "") -> None:
        """
        Logs the measured modes that matter for the stability, followed by the flip count, :attr:`p_eff` and the
        predicted rate: every flipped mode and the least-stable certified mode always, the remaining places up to four
        by descending modulus. An uncertified mode is tagged as such, a certified one with its decision (flip,
        predicted flip, stable, marginal inside its undecidable band, or unstable but suppressed while flips are
        paused). The release message of the update, if any, is folded into the same line.

        :param flip: Which modes are flipped.
        :param released: The release message of this update, or an empty string.
        """
        if self._logger is None:
            return
        shown = [int(idx) for idx in np.flatnonzero(flip)]
        certified = np.flatnonzero(self.last_gated)
        if certified.size:
            shown.append(int(certified[np.argmin(self.last_lam_pi[certified].real)]))
        required = len(set(shown))
        shown += [int(idx) for idx in np.argsort(-np.abs(self.last_lam_pi))]
        shown = list(dict.fromkeys(shown))[: max(4, required)]
        modes = []
        for idx in shown:
            lam = self.last_lam_pi[idx]
            if self.last_gated[idx]:
                if flip[idx]:
                    decision = "predicted flip" if self._predicted is not None and self._predicted[idx] else "flip"
                else:
                    band = max(_MARGIN, self.last_res[idx])
                    decision = (
                        "marginal" if abs(lam.real) <= band else "stable" if lam.real > 0.0 else "unstable, suppressed"
                    )
                tag = f"certified, {decision}"
            else:
                tag = "uncertified"
            modes.append(f"lambda_Pi={lam:+.4f} (res {self.last_res[idx]:.1e}, {tag})")
        self._logger.info(
            f"Jacobian tracker: {released + '; ' if released else ''}{', '.join(modes)}; "
            f"{int(np.count_nonzero(flip))} flipped, p_eff={self.p_eff:.4f}, rho={self.predicted_rate():.4f}."
        )

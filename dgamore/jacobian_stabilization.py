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

This module turns the (iterate, proposal) pairs that the self-consistency already records into the leading
eigenpairs of :math:`J` - consecutive differences are secant samples :math:`\delta F \simeq J\,\delta X`, and a
Rayleigh-Ritz on them returns Ritz pairs with residual certificates - decides from those which directions to flip,
and damps each certified direction of a DAMPED step by the amount its own eigenvalue allows, with the sign
reversed on the flipped ones, through a map rebuilt on every update that certifies a mode and otherwise kept in
force until a growth release, the calm release of the reflector beside it or a scaffold. A certified unstable
direction whose flip is not acted on yet, below its persistence streak or with its reflector refused, keeps the
mixing's step, and the directions the estimate does not certify take the damped step at the bound of the whole
certified spectrum. An accelerated step keeps the step its own secant model produced on the reflected map. No
evaluation of the proposal map happens here, and the module imports neither MPI nor the configuration. A certified
direction is also flipped ahead of its crossing when the real parts of three consecutive matched estimates extrapolate
across the boundary beyond their error bar, or when its eigenvalue has just passed through a pole, and a direction
carried over from a predecessor run is flipped ahead of the crossing that its values at the two preceding temperatures
extrapolate to.

Follows H. Essl, S. Rohshap, M. Gievers, M. Wallerberger, A. Toschi and A. Kauch, arXiv:2606.04936, and H. Essl, M.
Reitner, E. Kozik and A. Toschi, Phys. Rev. Lett., doi:10.1103/zjy7-4jqd. The reflector follows the Schur construction
of arXiv:2609.11405, App. G3, the boson-exchange follow-up of arXiv:2606.04936, and the per-direction damping Eq. (27)
of arXiv:2606.04936 and Eq. (21) of arXiv:2609.11405 with the stabilization matrix of its App. G3. The follow-up finds
the per-direction damping helps TRILEX and SBE, whose iterated object holds the self-energy and the screened interaction
(the SBE couplings on top of them), and that uniform mixing is more stable for MBE, which it attributes to MBE being
affected by vertex divergences the way the parquet iteration is; on that iteration the earlier paper measured no
speed-up and less stability than uniform damping. Ladder DGA iterates the self-energy alone, yet sits close to those
divergences on the beta ladder, so the benefit here is a hypothesis for the ladder to confirm."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr, schur, solve_sylvester, solve_triangular
from scipy.linalg.lapack import dtrsen

_MARGIN = 1e-2  # a real part of lambda_Pi below this in modulus is undecidable: not flipped, and it enters the
# damping bound with the margin in place of its real part
_DAMPING_C = 0.5  # the safety factor c in (0, 1) of the damping bound (a smaller c widens the basin), at the
# vertex of the stability parabola, the value that contracts a binding mode fastest
_MIXING_FLOOR = 0.01  # the effective damping is never lowered below this
_RITZ_GATE = 3e-2  # Ritz residual gate, relative to max(1, |theta|)
_PERSIST = 3  # updates carrying a certified unstable mode before the first flip
_MIN_PAIRS = 3  # fewer pairs than this carry no estimate
_SUBSPACE_ANGLE = 0.1  # rad; largest principal angle above which the flipped basis is replaced
_COLLINEAR_CUTOFF = 1e-8  # relative cutoff below which a secant column counts as collinear
_NOISE_FACTOR = (
    1000.0  # noise floor in machine epsilons of the iterate norm (also the step gate); noise gain of R^-1 <= 1e-3
)
_COND_CAP = 1e3  # norm of the Sylvester solution, and condition number of the carried flipped columns, above
# which no reflector is built
_MAP_NORM_CAP = 2.0  # spectral norm an assembled damping map may reach in units of the largest damping it encodes;
# above it the span's own signed projectors replace it, and the map is refused when they exceed the cap as well
_SIGNS_UNBOUNDED: tuple = ()  # result of _carried_nonuniform_map where the signs of its kept columns carry no
# weighting inside that cap; the reflector alone may still carry the install
_OBLIQUE_CAP = 10.0  # spectral norm above which an oblique spectral projector counts as inseparable from its
# complement, and the orthogonal projector of the same subspace is used instead
_REAL_CUTOFF = 1e-8  # relative cutoff below which a Ritz value counts as real
_SCHUR_TOL = 1e-10  # relative tolerance locating a Ritz value on the diagonal of the Schur form it was read from,
# and separating two blocks of that form for the commuting map
_FLIP_OVERLAP = 0.5  # |Q^T u| above which a Ritz vector counts as lying in the flipped subspace, used by the
# map's sign rule, the calm release of a carried basis and the identity of a mode across updates and runs
_PREDICT_IN_LOOP = True  # flip a direction as soon as its extrapolated real part crosses, instead of after the streak
_PREDICT_ACROSS_RUNGS = True  # flip a carried direction whose two-rung extrapolation in beta crosses
_NONUNIFORM_IN_LOOP = True  # build the per-direction damping map from the running estimate; False leaves the damped
# steps on the scalar damping alone (the map of a carried set is unaffected)
_STORE_BAND = 0.1  # Re lambda_Pi below which a certified mode's Ritz vector is written for a successor run; this is
# the storage band, not the flip band -max(_MARGIN, res) of classify
_CHAIN_LENGTH = 3  # informative updates the values of a mode are chained over for the extrapolation
_POLE_MODULUS = 5.0  # |lambda_Pi| beyond which a mode counts as approaching or crossing a pole
_POLE_CONTINUITY = 0.5  # relative growth of the 1/lambda_Pi step across a sign change still read as one continuous
# drift through the pole
_MAX_BETA_STEP_RATIO = 2.0  # largest (beta_3 - beta_2) / (beta_2 - beta_1) a two-rung extrapolation may bridge
TRACKER_PAIRS = 7  # (iterate, proposal) pairs the tracker keeps, i.e. 6 secant differences per estimate
_MIN_ITERATION_CAP = 15000  # upper limit of the minimum-iteration guard
JACOBIAN_FILE = "jacobian.npz"  # the one file a run keeps its Jacobian record in: the per-iteration traces (leading
# lambda_Pi, their residuals, the damping), rewritten every iteration, and the certified spectrum added at the end


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


def noise_floor(x: np.ndarray) -> float:
    r"""
    Returns the absolute norm below which an increment of the iterate :math:`x` carries no information beyond the
    rounding noise of its storage precision, :math:`10^{3}\,\epsilon_{\mathrm{mach}}\,\lVert x \rVert` with
    :math:`\epsilon_{\mathrm{mach}}` of the array's real dtype.

    :param x: The iterate (real vector or complex array).
    :return: The noise floor.
    """
    single = np.asarray(x).dtype in (np.complex64, np.float32)
    return _NOISE_FACTOR * np.finfo(np.float32 if single else np.float64).eps * float(np.linalg.norm(x))


def _independent_columns(r: np.ndarray, noise_floor: float) -> np.ndarray:
    r"""
    Selects the columns of the thin-QR factor :math:`R` of the secant increments that carry an independent
    direction, newest first, so a stalled or repeated step cannot make the second QR factorization singular.
    Because :math:`\delta X = QR` with :math:`Q` orthonormal, the singular values of :math:`R` equal those of
    :math:`\delta X` and dropping a column of :math:`R` is the same as dropping the matching column of
    :math:`\delta X`, so the sweep runs on the small :math:`R` (``[m, m]``) instead of the tall :math:`\delta X`
    (``[n_real, m]``). A column is kept when the part of it orthogonal to the already kept (more recent) ones
    exceeds :math:`\max(\epsilon_{\mathrm{col}}\,\sigma_{\mathrm{max}}(R),\, \eta)`: the relative cutoff
    :math:`\epsilon_{\mathrm{col}}` catches a repeated step, the absolute floor :math:`\eta` a column whose
    orthogonal part is pure rounding noise of the stored iterates, which the relative cutoff misses once the
    iteration has converged in the stable directions.

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

    The thin QR of :math:`\delta X` is computed once; columns of its factor :math:`R` that are collinear with more
    recent ones or whose orthogonal part is below the noise floor are dropped (see :func:`_independent_columns`),
    together with the matching columns of :math:`\delta F`, and the kept columns of :math:`R` are re-factored with
    a second, small QR, :math:`R_{\mathrm{kept}} = Q_2 R_2`, so that :math:`\delta X_{\mathrm{kept}} = (Q Q_2) R_2`
    is the thin QR of the kept increments without ever factoring the tall :math:`\delta X` a second time or taking
    an SVD of it. Dropping the noise columns is what keeps :math:`R_2^{-1}` from amplifying the rounding noise of
    the stored iterates into Ritz values that are self-consistent, and therefore certified, but meaningless. The
    Ritz values estimate the eigenvalues :math:`\lambda_J` of the Jacobian; the caller forms
    :math:`\lambda_\Pi = 1 - \lambda_J`. The residual certificate is informative only when the real ambient
    dimension exceeds one - for a single real component any nonzero pair is trivially collinear. The production
    path (:meth:`JacobianTracker.update`) builds the increments directly and calls :func:`_ritz_from_increments`,
    whose two further return values (the kept orthonormal basis and the Schur form of the projected matrix
    :math:`B`) the reflector needs; this function is the reference form of the same computation, kept for the
    derivation and the tests, and returns the three Ritz arrays alone.

    :param xs: Real column stack of the iterates, oldest first, shape ``[n_real, n]`` with ``n >= 3``.
    :param fs: Real column stack of the matching proposals, oldest first, shape ``[n_real, n]``.
    :param noise_floor: Absolute norm below which an increment carries no information beyond rounding noise. An
        increment is dropped when the part of it orthogonal to the more recent kept ones stays below the larger of
        this floor and the relative collinearity cutoff.
    :return: The Ritz values :math:`\theta`, the Ritz vectors :math:`u` as columns, and the Ritz residuals; all
        empty when no increment survives the noise floor.
    """
    return _ritz_from_increments(np.diff(xs, axis=1), np.diff(fs, axis=1), noise_floor)[:3]


def _increments(window: list[np.ndarray]) -> np.ndarray:
    r"""
    Builds the secant increments :math:`\delta X_{:,j} = \mathrm{vec}(x_{j+1}) - \mathrm{vec}(x_j)` of a window of
    complex arrays as the columns of one preallocated real array, so that only one real vector besides the result
    is alive while the window is converted.

    :param window: The recorded arrays of one shape, oldest first.
    :return: The increments, shape ``[n_real, len(window) - 1]``, oldest first.
    """
    prev = to_vec(window[0])
    increments = np.empty((prev.size, len(window) - 1), dtype=np.float64)
    for col in range(increments.shape[1]):
        cur = to_vec(window[col + 1])
        increments[:, col] = cur - prev
        prev = cur
    return increments


def _ritz_from_increments(
    dx: np.ndarray, df: np.ndarray, noise_floor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SchurForm]:
    r"""
    Runs the Rayleigh-Ritz of :func:`secant_ritz` on the secant increments themselves, so that a caller building
    the increments column by column never holds the iterate and proposal stacks as well. The tall factor
    :math:`Q` of the first QR is released as soon as the kept basis :math:`Q_K = Q Q_2` exists, and
    :math:`R_2^{-1}` is applied as a triangular solve rather than formed. The eigenpairs of the projected matrix
    are read from its real Schur form (:func:`_schur_ritz`), which the reflector and the damping map reorder.

    :param dx: Secant increments of the iterates, shape ``[n_real, m]``, oldest first.
    :param df: Matching increments of the proposals, shape ``[n_real, m]``.
    :param noise_floor: Absolute norm below which an increment carries no information beyond rounding noise.
    :return: The Ritz values, the Ritz vectors as columns, the Ritz residuals, the orthonormal basis of the kept
        increments and the Schur form of the projected matrix :math:`B` the Ritz pairs are read from; all empty
        when no increment survives the noise floor.
    """
    q, r = np.linalg.qr(dx)
    keep = _independent_columns(r, noise_floor)
    if keep.size == 0:
        empty = np.zeros(0, dtype=np.complex128)
        return (
            empty,
            np.zeros((dx.shape[0], 0), dtype=np.complex128),
            np.zeros(0, dtype=np.float64),
            np.zeros((dx.shape[0], 0), dtype=np.float64),
            SchurForm(np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0, dtype=int)),
        )
    q2, r2 = np.linalg.qr(r[:, keep])
    q_kept = q @ q2
    del q
    jq = np.linalg.solve(r2.T, df[:, keep].T).T
    theta, y, form = _schur_ritz(q_kept.T @ jq)
    u = q_kept @ y
    res = np.empty(theta.size, dtype=np.float64)
    for col in range(theta.size):
        res[col] = np.linalg.norm(jq @ y[:, col] - theta[col] * u[:, col]) / np.linalg.norm(u[:, col])
    return theta, u, res, q_kept, form


def classify(
    lam_pi: np.ndarray, gated: np.ndarray, p_config: float, res: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    r"""
    Decides which certified modes to flip and how much damping the measured spectrum allows. A mode is flipped when
    no positive damping can pull it into the unit disk, and every certified mode binds the damping through

    .. math::

        p_{\mathrm{eff}} = \min\left( p_{\mathrm{config}},\; \max\left(
        c \min_\alpha \frac{2\,\max(|\mathrm{Re}\,\lambda_{\Pi,\alpha}|,\, m_\alpha)}{|\lambda_{\Pi,\alpha}|^{2}},
        \; p_{\mathrm{floor}} \right) \right),

    evaluated with the real part of a flipped mode already reversed, so the configured damping is a ceiling that is
    only ever lowered. The band :math:`m_\alpha = \max(m, \mathrm{res}_\alpha)` of undecidable sign widens with
    the Ritz residual, since the residual certifies a backward error that the eigenvalue condition number of a
    non-normal map can amplify: a mode whose real part lies inside its band is not flipped, and it enters the bound
    with the band in place of its real part, the loosest bound compatible with any sign inside it, so that an
    undecidable mode with a large imaginary part (a pseudo-divergence) still lowers the damping it is unstable at,
    while a near-neutral one does not bind.

    :param lam_pi: Measured eigenvalues :math:`\lambda_\Pi` of :math:`\Pi = \mathbb{1} - J`.
    :param gated: Whether each mode passed the Ritz residual gate.
    :param p_config: The configured damping, i.e. the ceiling of the returned value.
    :param res: The Ritz residuals, widening each mode's undecidable band; ``None`` keeps the fixed margin.
    :return: The boolean flip mask and the effective damping.
    """
    margin = np.full(lam_pi.shape, _MARGIN) if res is None else np.maximum(_MARGIN, res)
    flip = gated & (lam_pi.real < -margin)
    p_max = np.inf
    if gated.any():
        real_part = np.maximum(np.abs(lam_pi.real[gated]), margin[gated])
        modulus_sq = np.maximum(np.abs(lam_pi[gated]) ** 2, np.finfo(np.float64).tiny)
        p_max = float(np.min(2.0 * real_part / modulus_sq))
    return flip, min(p_config, max(_DAMPING_C * p_max, _MIXING_FLOOR))


def _direction_damping(lam_pi: np.ndarray, flip: np.ndarray) -> np.ndarray:
    r"""
    Returns the damping each direction allows on its own,

    .. math::

        s_\alpha p_\alpha, \qquad p_\alpha = \min\left(1,\; c\,
        \frac{2\,|\mathrm{Re}\,\lambda_{\Pi,\alpha}|}{|\lambda_{\Pi,\alpha}|^{2}}\right), \qquad
        s_\alpha = -1 \text{ on a flipped direction},

    so that a direction damped with it iterates with :math:`1 - s_\alpha p_\alpha \lambda_{\Pi,\alpha}`: the
    bound of :func:`classify` evaluated per direction instead of over the whole spectrum, with the same constant
    and the upper limit of one. The value is invariant under conjugation, so both members of a complex-conjugate
    pair receive the same damping.

    :param lam_pi: The eigenvalues :math:`\lambda_\Pi` of the directions.
    :param flip: Which of them are flipped.
    :return: The signed dampings.
    """
    modulus_sq = np.maximum(np.abs(lam_pi) ** 2, np.finfo(np.float64).tiny)
    damping = np.minimum(1.0, _DAMPING_C * 2.0 * np.abs(lam_pi.real) / modulus_sq)
    return np.where(flip, -damping, damping)


def _spectral_weighting(basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    r"""
    Returns :math:`B\,\mathrm{diag}(w)\,B^{-1}`, i.e. the weighted sum :math:`\sum_\alpha w_\alpha \Pi_\alpha`
    of the spectral projectors :math:`\Pi_\alpha` of the columns of :math:`B`, formed as a linear solve rather
    than an inverse. Any such sum commutes with every map the columns of :math:`B` are eigenvectors of. A weight
    shared by the members of a complex-conjugate pair makes the result real, and the real part is what is returned.

    :param basis: The coordinates of the directions as columns, shape ``[k, k]``.
    :param weights: One real weight per column.
    :return: The real ``[k, k]`` map.
    """
    return np.linalg.solve(basis.T, (basis * weights).T).T.real


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
    return np.column_stack(columns).astype(np.float64), np.array(owners, dtype=int)


def _schur_blocks(t_mat: np.ndarray) -> list[tuple[int, int]]:
    """
    Returns the half-open index range of every diagonal block of a real Schur factor, ``2 x 2`` where the
    subdiagonal entry below the block is nonzero (a complex-conjugate pair) and ``1 x 1`` otherwise.

    :param t_mat: The quasi-triangular factor, shape ``[k, k]``.
    :return: The blocks as ``(start, stop)`` pairs, in order.
    """
    bounds, start = [], 0
    while start < t_mat.shape[0]:
        stop = start + (2 if start + 1 < t_mat.shape[0] and t_mat[start + 1, start] != 0.0 else 1)
        bounds.append((start, stop))
        start = stop
    return bounds


def _diagonal_values(t_mat: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    """
    Returns the eigenvalue every diagonal position of a real Schur factor carries: the entry itself on a ``1 x 1``
    block, and the conjugate pair of a ``2 x 2`` block on its two positions.

    :param t_mat: The quasi-triangular factor, shape ``[k, k]``.
    :param bounds: Its diagonal blocks as ``(start, stop)`` pairs (see :func:`_schur_blocks`).
    :return: The complex values, one per diagonal position.
    """
    values = np.empty(t_mat.shape[0], dtype=np.complex128)
    for low, high in bounds:
        values[low:high] = t_mat[low, low] if high - low == 1 else np.linalg.eigvals(t_mat[low:high, low:high])
    return values


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
    values = _diagonal_values(t_mat, _schur_blocks(t_mat))
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
    m_mat: np.ndarray | None,
    damping: np.ndarray,
    reflection: Callable[[], np.ndarray],
    log: Callable[[str], None] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    r"""
    Caps an assembled damping map against the damping it encodes. A map whose spectral norm exceeds
    ``_MAP_NORM_CAP`` times the largest :math:`|m_a|` it carries would amplify the residual instead of damping it,
    and is replaced by the uniform signed map :math:`\min_\alpha p_\alpha \sum_\alpha s_\alpha \Pi_\alpha`, the same
    weighted sum of the same spectral projectors with every damping replaced by its own sign, so that every block
    keeps the sign it was given on its own eigenvector rather than on the coordinate axis its basis happens to use.
    A map that could not be assembled at all takes the same replacement.

    Without a flipped block that sum is the identity and the replacement the scalar :math:`\min_\alpha p_\alpha`.
    With one it is the reflection :math:`\mathbb{1} - 2\,\Pi_{\mathrm{flip}}` of the flipped blocks along the
    others, whose norm grows without bound as a flipped and a stable eigenvector approach each other, since no
    bounded map can reverse one of two nearly parallel directions and fix the other. It is therefore capped the
    same way, and the whole map is refused above the cap: those directions keep the step the mixing gave them.

    :param m_mat: The assembled map in the coordinates of its own basis, or ``None`` when it could not be assembled.
    :param damping: The signed damping of every block.
    :param reflection: Builds the same weighted sum of spectral projectors with every damping replaced by its sign,
        in the coordinates of the same basis; called only where the assembled map lies above the cap.
    :param log: Callable receiving the fallback and refusal messages, or ``None`` to stay silent.
    :return: The map itself with the damping it was given, or the uniform signed map on the same basis with the
        signed damping each block takes in it; ``None`` when that replacement lies above the cap as well.
    """
    if m_mat is not None and float(np.linalg.norm(m_mat, 2)) <= _MAP_NORM_CAP * float(np.max(np.abs(damping))):
        return m_mat, damping
    uniform = float(np.min(np.abs(damping)))
    signed = reflection()
    if float(np.linalg.norm(signed, 2)) > _MAP_NORM_CAP:
        if log is not None:
            log(
                "Jacobian tracker: the signs of the certified damping cannot be carried on the directions they "
                f"belong to within {_MAP_NORM_CAP:g} times the damping they encode, no per-direction damping used."
            )
        return None
    if log is not None:
        log(
            "Jacobian tracker: nonuniform damping not separable on the certified span (spectral norm above "
            f"{_MAP_NORM_CAP:g} times the damping it encodes), uniform damping {uniform:.4f} used on every "
            "certified direction of it."
        )
    return uniform * signed, np.sign(damping) * uniform


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
    complementary one (:math:`\Pi^{2} = \Pi`, :math:`\Pi Z_1 = 0`, :math:`B \Pi = \Pi B`). Blocks are selected by
    their diagonal position, never by their value. Two values on opposite sides of the selection that nearly
    coincide make the Sylvester equation ill-conditioned, which the norm of its solution reports; two adjacent
    blocks the reordering cannot swap, or a conjugate pair selected on one side only, cannot be separated at all.

    :param form: The Schur form the Ritz values were read from, with their diagonal positions.
    :param selected: Which Ritz values are selected, in the order of the form's positions.
    :param log: Callable receiving the refusal message, or ``None`` to stay silent.
    :return: The columns :math:`Z_2 + Z_1 X` of the invariant subspace, the columns :math:`Z_2` and the trailing
        block :math:`T_{22}`, the representation of :math:`B` on that subspace in the basis of the columns
        (:math:`B C = C T_{22}`); ``None`` when the Ritz values have no positions, when the reordering fails, when
        a conjugate pair is selected on one side only, when the Sylvester equation cannot be solved, or when its
        solution norm exceeds ``_COND_CAP``.
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
    Builds the per-direction damping of an estimate on the invariant subspace of its certified directions. That
    subspace is separated from the uncertified one by :func:`_invariant_columns`, whose columns :math:`C` give
    both its orthonormal span :math:`Q_c` and the OBLIQUE projector onto it along the uncertified subspace,
    :math:`\Pi_{\mathrm{cert}} = Q_c\,W_c` with the dual rows :math:`W_c = R_c Z_c^{T} Q_K^{T}` of the thin QR
    :math:`Q_K C = Q_c R_c`, the same construction as the reflector's. Both the projector and the map annihilate
    an uncertified Ritz direction, which therefore keeps exactly the step the mixing gave it. On the span the map
    is :math:`N = \sum_\alpha s_\alpha p_\alpha \Pi_\alpha` with :math:`\Pi_\alpha` the spectral projector of the
    Schur block :math:`\alpha` of the restricted Jacobian along the others: the trailing block :math:`T_{22}` of
    the reordered form represents the Jacobian on the columns, so in the orthonormal coordinates it reads
    :math:`B_c = R_c T_{22} R_c^{-1}`, block upper-triangular with the blocks of :math:`T_{22}`, and the map is
    the commuting upper-triangular :math:`M` of :func:`_commuting_upper` on it, with no second decomposition. The
    reordering leaves the certified modes in the trailing block in the order of their diagonal positions, so each
    block takes the damping and the flip decision of the modes at its positions (both members of a conjugate
    pair, which carry the same damping).

    Two blocks closer than ``_SCHUR_TOL`` (relative), or an assembled map above the magnitude cap of
    :func:`_bounded_map`, fall back to the uniform damping :math:`\min_\alpha p_\alpha` on the same span with the
    per-block signs kept, built from the same spectral projectors so that every certified eigenvector keeps the
    sign of its own block; where that reflection is above the cap as well, no map is built at all. A projector
    whose spectral norm exceeds ``_OBLIQUE_CAP`` falls back to the orthogonal projector :math:`Q_c Q_c^{T}` onto
    the same span, since the two subspaces are then too close for the oblique split to be applied to a residual.

    :param q_kept: The orthonormal basis of the kept secant increments, shape ``[n_real, k]``.
    :param form: The Schur form of the projected Jacobian :math:`B` on that basis, with the Ritz positions.
    :param theta: Its Ritz values, in the order ``lam_pi``, ``flip`` and ``gated`` are given in.
    :param lam_pi: The eigenvalues :math:`\lambda_\Pi` of the Ritz directions.
    :param flip: Which of them are flipped.
    :param gated: Which of them passed the Ritz residual gate.
    :param log: Callable receiving the fallback and refusal messages, or ``None`` to stay silent.
    :param band_damping: The damping of a certified block that is not flipped although its real part is negative,
        i.e. whose sign lies inside the undecidable band of :func:`classify`: the scalar damping of the damped step,
        so that such a block keeps the step the mixing gives it instead of a positive :math:`p_\alpha` that would
        push it outward; ``None`` gives it its own :math:`p_\alpha` like every other block.
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
    bounds = _schur_blocks(t22)
    starts = [low for low, _ in bounds]
    ordered = np.flatnonzero(gated)[np.argsort(form.positions[gated])]
    damping = _direction_damping(lam_pi[ordered[starts]], flip[ordered[starts]])
    if band_damping is not None:
        undecided = ~flip[ordered[starts]] & (lam_pi[ordered[starts]].real < 0.0)
        damping = np.where(undecided, band_damping, damping)
    values = _diagonal_values(t22, bounds)[starts]
    tol = _SCHUR_TOL * max(1.0, float(np.max(np.abs(theta))))
    separated = all(abs(values[i] - values[j]) > tol for i in range(len(values)) for j in range(i))
    try:
        b_c = solve_triangular(upper.T, (upper @ t22).T, lower=True).T
        assembled = _commuting_upper(b_c, bounds, damping) if separated else None
        bounded = _bounded_map(assembled, damping, lambda: _commuting_upper(b_c, bounds, np.sign(damping)), log)
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
    dual = upper @ z_c.T
    if float(np.linalg.norm(dual, 2)) > _OBLIQUE_CAP:
        dual = coords.T
        if log is not None:
            log(
                "Jacobian tracker: the certified and the uncertified subspace are closer than the obliquity cap "
                f"{_OBLIQUE_CAP:g}, the orthogonal projector is used on the certified span."
            )
    return (q_kept @ coords, dual @ q_kept.T, m_mat), applied


def _carried_nonuniform_map(
    lam_pi: np.ndarray, u: np.ndarray, flip: np.ndarray, log: Callable[[str], None] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[()] | None:
    r"""
    Builds the per-direction damping of a carried set, which has Ritz vectors but no projected matrix: the real
    columns of every carried mode (see :func:`_real_columns`) are orthonormalized by one column-pivoted QR and each
    column takes the damping of the mode it belongs to, the whole carried span counting as certified. Pivoting
    orders the columns by decreasing weight, so a column whose diagonal entry has fallen below the leading one by
    more than ``_COND_CAP`` is dropped rather than allowed to spoil the inverse the weighting solves for. Only the
    flipped columns are indispensable: dropping a stable one costs its own damping, while dropping a flipped one
    would lose the reversed sign, so that refuses the set; a run stores a vector for every certified mode below the
    storage band ``_STORE_BAND`` (see :meth:`JacobianTracker.spectrum_state`), so a carried set holds stable columns as
    well, and dropping one of them costs that mode's own damping only. There is no projected matrix to split the span
    off an uncertified subspace with, so the projector onto it is the orthogonal one, and the weighting of two nearly
    parallel kept columns is capped against the damping it encodes exactly as the estimate's map is
    (:func:`_bounded_map`), the uniform fallback built from the same weighting with every damping replaced by its sign.
    Two nearly parallel columns of opposite sign have no bounded weighting at all, so a mixed-sign set above the cap
    returns ``_SIGNS_UNBOUNDED``, which is a refusal of the map alone: the carried set may still act through its
    orthogonal reflector and the scalar damping bound.
    The two refusals that return ``None`` instead leave nothing of the set to install.

    :param lam_pi: The carried eigenvalues :math:`\lambda_\Pi`.
    :param u: Their normalized Ritz vectors as columns.
    :param flip: Which of them are flipped.
    :param log: Callable receiving the dropped-column, fallback and refusal messages, or ``None`` to stay silent.
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
    weights = _direction_damping(lam_pi, flip)[owners[pivots[:kept]]]
    coords = upper[:kept, :kept]
    bounded = _bounded_map(
        _spectral_weighting(coords, weights), weights, lambda: _spectral_weighting(coords, np.sign(weights)), log
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

    :param lam: The Ritz values.
    :param u: Their Ritz vectors as columns.
    :return: The members of each group as ``(mode, partner)`` with ``partner = -1`` for a real mode, and the
        orthonormal basis of each group's subspace.
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
        basis, upper = np.linalg.qr(columns)
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
    \mathrm{res}^{(n)})` the undecidable band of :func:`classify`. Only the real part is extrapolated: a complex pair
    drives :math:`|1 - p\lambda|` past one while its real part is still positive, a case the damping bound cures and
    a flip would not. The third point supplies the fit scatter, so two points never predict, and the monotonicity
    keeps a mode oscillating around the axis from ever predicting. A mode whose modulus exceeds ``_POLE_MODULUS`` and
    grew on both matched updates is approaching a pole, where the eigenvalue jumps rather than drifts; it is reported
    instead of predicted (see :func:`detect_pole_jump`).

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

    with :math:`c` the constant ``_POLE_CONTINUITY``. The identity across the pole rests on the eigenvector overlap
    of the chain (see :func:`_match_modes`), which is continuous through the pole as well. The jump itself is the
    evidence for the flip: after a pole the growth is :math:`|1 + p\,|\mathrm{Re}\,\lambda_\Pi||` with a large real
    part, the case where waiting for the persistence streak is expensive. A mode before the jump cannot be flipped
    ahead of it, since its flipped factor :math:`1 + p_\alpha \lambda_\Pi \simeq 1 + 2c` diverges until the pole is
    actually crossed.

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
    the new rung then sits past the pole and the mode is unstable there, which is safe to act on because the whole
    run is at :math:`\beta_3`. For such a mode the returned prediction is the inverse of the extrapolated
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
    whether a reflection was applied afterwards, so the flip set is re-decided from scratch and a mode that a noisy
    estimate flipped is released again by the same rule. While a reflector is installed the proposal residual
    :math:`r = S(x_n) - x_n` is reflected on the invariant subspace of the flipped directions
    (:meth:`reflect`), and the iterate a DAMPED step returns is corrected on the certified span afterwards
    (:meth:`stabilize_step`; an accelerated step keeps the step its own secant model produced),

    .. math::

        x_{n+1} = x_{\mathrm{mix}} - \Pi_{\mathrm{cert}}\,(x_{\mathrm{mix}} - x_n) + N\,r, \qquad
        N = \sum_\alpha s_\alpha p_\alpha \Pi_\alpha,

    with :math:`\Pi_\alpha` the spectral projector of the certified Ritz direction :math:`\alpha`,
    :math:`\Pi_{\mathrm{cert}}` the OBLIQUE projector onto the subspace they span along the uncertified one and
    :math:`s_\alpha p_\alpha` the damping of :func:`_direction_damping`, so a certified direction iterates with
    :math:`1 - s_\alpha p_\alpha \lambda_{\Pi,\alpha}` whatever the damping did to it, while a direction the
    estimate does not certify keeps exactly the step the mixing gave it; where the two subspaces are too close to
    separate obliquely the orthogonal projector is used instead and the log says so, and on a carried set, which
    carries no projected matrix, the projector is orthogonal throughout. :math:`Q W`, the spectral
    projector onto the invariant subspace of the flipped Ritz values along the subspace of the others (see
    :meth:`_flipped_reflector`), is what the reflection applies and what the release rules read; where it is
    orthogonal, i.e. for a normal projected map and for a carried set, :math:`W = Q^{T}`.

    The map is built from the current estimate on every update that certifies a mode, with or without a reflector,
    and stays in force until the next certifying rebuild, a growth release, the calm release of the reflector
    beside it or a scaffold (an update that certifies nothing leaves it in place): a direction certified stable
    takes its own positive damping even where it was flipped before, and a direction certified unstable carries the
    reversed sign only while the installed flipped subspace already holds it or the persistence streak of
    :meth:`update` admits a new flip whose reflector could be built; until then it is no part of the map and keeps
    the step the mixing gave it. The calm release of :meth:`update` undoes a flip and therefore needs a reflector,
    whose map it drops with it until the next certifying update, while the growth release applies to everything
    installed and holds the map off for ``_PERSIST`` informative updates (a frozen update does not count).

    :attr:`p_eff` is the largest damping the certified spectrum allows, never above the configured value and never
    raised again within a run; the mixing damps every damped Picard step with it, and the certified span of such a
    step is overwritten afterwards. :attr:`estimated` tells whether the last :meth:`update` call produced a fresh
    spectrum estimate.

    The certified modes of consecutive informative updates are chained by the overlap of their Ritz vectors
    (:func:`_match_modes`), and a mode whose chained real parts extrapolate across the boundary beyond their error bar
    (:func:`predict_crossing`), or whose eigenvalue has just passed through a pole (:func:`detect_pole_jump`), is a
    flip candidate with its persistence streak treated as satisfied on that update, so the flip lands ahead of the
    crossing instead of ``_PERSIST`` certified updates after it; the rest of the install path, the refusals and the
    releases are the same. A predicted flip that does not materialize ends as a candidate the moment the real part
    turns back up and is undone by the calm release.
    """

    def __init__(self, p_config: float, logger=None):
        """
        Initializes an inactive tracker: nothing flipped, no spectrum measured yet, damping at its configured value.

        :param p_config: The configured damping, i.e. the ceiling of :attr:`p_eff`.
        :param logger: Logger receiving the measured spectrum, or ``None`` to stay silent.
        """
        self.p_config = float(p_config)
        self.p_eff = float(p_config)
        self.q = None
        self.w = None
        self.last_lam_pi = None
        self.last_res = None
        self.last_gated = None
        self.n_updates = 0
        self.estimated = False
        self._logger = logger
        self._flip = None
        self._last_u = None
        self._persist = 0
        self._calm = 0
        self._grow = 0
        self._hold = 0  # updates left before the estimate may install its map again after a growth release
        self._last_residual_norm = np.inf
        self._floor_warned = False
        self._nonuniform = None  # (orthonormal span, dual rows, damping map) of the certified directions
        self._applied = None  # signed damping the map built from the estimate applies per mode; None on a carried map
        self._pending = None  # carried (lam_pi, u, flip) waiting for flips to be allowed
        # installed carried (lam_pi, u, flip), re-pended on a scaffold-forced release; set exactly while the
        # reflector or the map in force came from a predecessor run
        self._carried_set = None
        self._last_certified = None  # (lam_pi, res, gated, flip, u, predicted) of the last update that certified a mode
        self._exact_certified = None  # the snapshot install_exact_spectrum made, while it is the last certified one
        self._history = []  # the last _CHAIN_LENGTH informative estimates, matched mode by mode (see _predict)
        self._predicted = None  # the predicted-crossing mask of the last estimate, aligned with last_lam_pi
        # (lam_pi, res, beta, u) of the predecessor's carried columns on this run's window, matched at save time
        self._predecessor = None

    @property
    def active(self) -> bool:
        """Whether a flipped subspace or a per-direction damping map is currently installed."""
        return self.q is not None or self._nonuniform is not None

    def spectrum_state(self, shape: tuple[int, ...], beta: float, converged: bool, dtype: np.dtype) -> dict:
        r"""
        Collects the certified part of the last update that certified a mode with flips allowed, or the set
        :meth:`carry_in` installed, whatever ``allow_flip`` was then, for a successor run: every certified Ritz value
        with its residual, its flip decision and whether that flip was predicted, and, for every mode below the
        storage band ``_STORE_BAND`` (a band of the real part distinct from the flip band of :func:`classify`, so
        that a successor can follow a mode still on the stable side), the Ritz vector in the given precision. The
        real and the imaginary part of a Ritz vector are each rebuilt into the complex window array they flatten
        from and stored as one column each; a real mode's imaginary column is zero. Every mode also carries the
        value, residual and inverse temperature of the predecessor mode it matches (``lam_prev``, ``res_prev``,
        ``beta_prev``): the columns :meth:`carry_in` re-gridded onto this run's window are matched against the
        snapshot's vectors by :func:`_match_modes`, ``nan`` where a mode has no match or the run had no predecessor,
        so a successor reads one file and holds the values of two rungs. The last update of a run is often one whose
        estimate is frozen, certifies nothing, or ran with flips paused (a scaffolded map never replaces the snapshot
        even when it does certify something, since that map never ran physically), so the modes come from the stored
        snapshot rather than from the last estimate. The mode arrays are empty when nothing was ever certified. The
        flip decisions and :attr:`p_eff` are recorded for inspection and are not read back by :meth:`carry_in`, which
        re-decides the flips with the margin of :func:`classify`, the two-rung extrapolation of
        :func:`predict_rung_crossing`, and re-derives the damping bound from the carried spectrum itself. ``exact``
        tells whether the snapshot is the exact spectrum of :meth:`install_exact_spectrum` rather than an estimate.

        :param shape: The complex window shape ``(kx, ky, kz, nb, nb, 2 niv_core)`` of the iterated array.
        :param beta: Inverse temperature :math:`\beta` of the run.
        :param converged: Whether the run reached the pure fixed point.
        :param dtype: Complex precision the vector columns are stored in, i.e. the precision the loop keeps
            its self-energies in, so the file never holds more or less of a vector than the run itself does.
        :return: The state.
        """
        n_complex = int(np.prod(shape))
        if self._last_certified is None:
            lam_pi, res = np.zeros(0, dtype=np.complex128), np.zeros(0, dtype=np.float64)
            flip, u = np.zeros(0, dtype=bool), np.zeros((2 * n_complex, 0), dtype=np.complex128)
            predicted = np.zeros(0, dtype=bool)
        else:
            lam_pi, res, gated, flip, u, predicted = self._last_certified
            lam_pi, res = lam_pi[gated].astype(np.complex128), res[gated].astype(np.float64)
            flip, u, predicted = flip[gated].astype(bool), u[:, gated], predicted[gated].astype(bool)
        stored = lam_pi.real < _STORE_BAND
        # the carried set itself is not matched against the columns it was built from
        own = self._last_certified is not None and self._predecessor is not None
        own = own and self._last_certified[4] is self._predecessor[3]
        lam_prev, res_prev, beta_prev = self._predecessor_values(lam_pi, u, skip=own)
        re_cols = [to_mat(u[:, j].real, shape).reshape(-1).astype(dtype) for j in np.flatnonzero(stored)]
        im_cols = [to_mat(u[:, j].imag, shape).reshape(-1).astype(dtype) for j in np.flatnonzero(stored)]
        return {
            "lam_pi": lam_pi,
            "res": res,
            "flip": flip,
            "predicted": predicted,
            "stored": stored,
            "lam_prev": lam_prev,
            "res_prev": res_prev,
            "beta_prev": beta_prev,
            "u_re": np.column_stack(re_cols) if re_cols else np.zeros((n_complex, 0), dtype=dtype),
            "u_im": np.column_stack(im_cols) if im_cols else np.zeros((n_complex, 0), dtype=dtype),
            "shape": np.asarray(shape, dtype=np.int64),
            "beta": np.float64(beta),
            "p_eff": np.float64(self.p_eff),
            "converged": np.bool_(converged),
            "exact": np.bool_(self._last_certified is not None and self._last_certified is self._exact_certified),
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
        Writes :meth:`spectrum_state` to ``path`` as a compressed ``.npz``, always: a run that certified nothing
        still records the spectrum and the damping it ended with. The per-iteration traces the loop collected (the
        leading Ritz values, their residuals and the damping, one row per iteration) go into the same file under
        their own keys, so a run leaves one Jacobian file.

        :param path: Destination file.
        :param shape: The complex window shape of the iterated array.
        :param beta: Inverse temperature :math:`\beta` of the run.
        :param converged: Whether the run reached the pure fixed point.
        :param dtype: Complex precision the vector columns are stored in (see :meth:`spectrum_state`).
        :param traces: Further arrays written beside the spectrum, keyed by name; ``None`` writes the spectrum alone.
        :return: None.
        """
        state = self.spectrum_state(shape, beta, converged, dtype)
        np.savez_compressed(path, **state, **(traces or {}))
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
        :attr:`p_eff` in force are left as they are.

        :param lam_pi: The eigenvalues :math:`\lambda_\Pi = 1 - \theta` of the modes.
        :param res: Their residual bounds.
        :param u: Their normalized complex eigenvectors as columns, in the vector layout of the iterated window.
        :return: None.
        """
        lam_pi, res = np.asarray(lam_pi, dtype=np.complex128), np.asarray(res, dtype=np.float64)
        gated = np.ones(lam_pi.size, dtype=bool)
        flip, _ = classify(lam_pi, gated, self.p_config, res)
        u = np.asarray(u, dtype=np.complex128)
        self._last_certified = (lam_pi, res, gated, flip, u, np.zeros(lam_pi.size, dtype=bool))
        self._exact_certified = self._last_certified
        self._log_info(
            f"Jacobian tracker: installed the exact spectrum of {lam_pi.size} modes, {int(flip.sum())} of them past "
            "the flip margin."
        )

    def _predecessor_values(
        self, lam_pi: np.ndarray, u: np.ndarray, skip: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Matches the modes of a spectrum snapshot to the predecessor's carried columns on this run's window (see
        :func:`_match_modes`) and returns, per mode, the predecessor's eigenvalue, residual and inverse temperature,
        ``nan`` where a mode has no match or nothing was carried. A snapshot that still is the carried set itself
        (a run that never certified a mode of its own) is not matched against the columns it was built from, since
        that would record the predecessor's values as a second rung of the chain.

        :param lam_pi: The certified eigenvalues of the snapshot.
        :param u: Their Ritz vectors as columns.
        :param skip: Whether the snapshot is the carried set itself, in which case every value is ``nan``.
        :return: The matched predecessor eigenvalues, residuals and inverse temperatures.
        """
        lam_prev = np.full(lam_pi.size, np.nan, dtype=np.complex128)
        res_prev, beta_prev = np.full(lam_pi.size, np.nan), np.full(lam_pi.size, np.nan)
        if self._predecessor is not None and lam_pi.size and not skip:
            lam_p, res_p, beta_p, u_p = self._predecessor
            match = _match_modes(lam_pi, u, lam_p, u_p)
            hit = match >= 0
            lam_prev[hit], res_prev[hit], beta_prev[hit] = lam_p[match[hit]], res_p[match[hit]], beta_p
        return lam_prev, res_prev, beta_prev

    def carry_in(
        self,
        state: dict,
        expand: Callable[[np.ndarray], np.ndarray],
        allow_flip: bool = True,
        beta: float | None = None,
    ) -> bool:
        r"""
        Installs the certified spectrum a predecessor run ended with, before the first step of this run.

        Every stored Ritz vector is rebuilt from its two columns through ``expand`` (the caller's re-gridding onto this
        run's window), as a real vector plus ``1j`` times another, and normalized; columns that come back empty are
        dropped. A vector is stored for every certified mode below the storage band of :meth:`spectrum_state`, so
        the carried columns hold the flipped modes together with the stable ones close to the boundary. A carried
        mode is flipped by the flip rule of :func:`classify` on its own value, or ahead of its crossing by the
        two-rung extrapolation of :func:`predict_rung_crossing` when the file carries the value the mode had at the
        predecessor's own predecessor (``lam_prev``): the values at :math:`\beta_1 < \beta_2` are extrapolated
        linearly in :math:`\beta` to this run's ``beta``, refused when the step ratio :math:`(\beta_3 - \beta_2) /
        (\beta_2 - \beta_1)` exceeds ``_MAX_BETA_STEP_RATIO`` or when the predecessor did not reach the pure fixed
        point (either case carries the set as without the extrapolation); this is the preemptive flip
        arXiv:2606.04936 recommends for a carried Jacobian, which flips the modes with a very small positive real
        part or approaching a pole (the supplement of Phys. Rev. Lett. doi:10.1103/zjy7-4jqd Sec. VI instead
        identifies the mode by a cusp of the eigenvalue at the converged points and flips it beyond that point). A
        file written without ``lam_prev`` carries as if no predecessor value existed. The set acts through the
        orthogonal reflector of :meth:`_install_carried` on the flipped columns together with the per-direction
        damping :func:`_carried_nonuniform_map` builds on all of them, a stable column with its own positive
        damping. The effective damping becomes the bound of the whole carried spectrum under this run's ceiling,
        lowered further to the bound of the predicted spectrum (the extrapolated real part in place of the carried
        one on the predicted modes only), applied at once even when the installation has to wait for a scaffold's
        release; a carried spectrum without a certified mode carries no bound and leaves the damping where it is.
        The damping the predecessor ended with is not read back, so that one transient early in a ladder cannot pin
        every later rung to the value it forced. The reflector of a carried set is orthogonal, and so is the
        projector its per-direction damping acts through, since a carried set brings no projected matrix to separate
        its span from an uncertified one. The re-gridded columns of a converged predecessor are kept until this
        run's spectrum is saved, where :meth:`spectrum_state` matches the run's own final vectors against them.

        :param state: The dictionary :func:`load_spectrum` returns.
        :param expand: Callable mapping one stored column to the real vector on this run's window.
        :param allow_flip: Whether the carried set may be installed now; otherwise it is kept pending and installed
            on the first update that allows flips while nothing else is installed.
        :param beta: Inverse temperature :math:`\beta` of this run, i.e. the :math:`\beta_3` of the extrapolation;
            ``None`` skips the extrapolation.
        :return: Whether the carried set is installed.
        """
        lam_pi = np.asarray(state["lam_pi"], dtype=np.complex128)
        res = np.asarray(state["res"], dtype=np.float64)
        if lam_pi.size == 0:
            self._log_info(
                "Jacobian tracker: the carried spectrum holds no certified mode, nothing carried; "
                f"p_eff={self.p_eff:.4f}."
            )
            return False
        _, p_new = classify(lam_pi, np.ones(lam_pi.size, dtype=bool), self.p_config, res)
        self._lower_p_eff(p_new)
        stored = np.flatnonzero(np.asarray(state["stored"], dtype=bool))
        columns = [
            np.asarray(expand(state["u_re"][:, i]), dtype=np.float64)
            + 1j * np.asarray(expand(state["u_im"][:, i]), dtype=np.float64)
            for i in range(stored.size)
        ]
        norms = np.array([np.linalg.norm(c) for c in columns])
        keep = norms > 0.0
        self._log_info(
            f"Jacobian tracker: carried {lam_pi.size} certified modes "
            f"(lambda_Pi {', '.join(f'{lam.real:+.4f}{lam.imag:+.4f}j' for lam in lam_pi)}), "
            f"{int(keep.sum())} with a usable vector, p_eff={self.p_eff:.4f}."
        )
        if not keep.any():
            return False
        u = np.column_stack([c / n for c, n, k in zip(columns, norms, keep) if k])
        indices = stored[keep]
        lam_c, res_c = lam_pi[indices], res[indices]
        flip = lam_c.real < -np.maximum(_MARGIN, res_c)
        predicted, lam_map = np.zeros(lam_c.size, dtype=bool), lam_c
        if _PREDICT_ACROSS_RUNGS and beta is not None and "lam_prev" in state:
            predicted, lam_map = self._rung_prediction(state, indices, lam_pi, res, float(beta))
            flip = flip | predicted
        if bool(state["converged"]):
            self._predecessor = (lam_c, res_c, float(state["beta"]), u)
        self.last_lam_pi, self.last_res, self._last_u = lam_c, res_c, u
        self._flip, self.last_gated = flip, np.ones(lam_c.size, dtype=bool)
        self._last_certified = (lam_c, res_c, self.last_gated, flip, u, predicted)
        if not allow_flip:
            self._pending = (lam_map, u, flip)
            self._log_info("Jacobian tracker: flips are paused by a scaffold, the carried flip waits for its release.")
            return False
        return self._install_carried(lam_map, u, flip)

    def _rung_prediction(
        self, state: dict, indices: np.ndarray, lam_pi: np.ndarray, res: np.ndarray, beta: float
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Decides which carried modes with vectors are flipped ahead of their crossing by the two-rung extrapolation of
        :func:`predict_rung_crossing`, and lowers the damping to the bound of the predicted spectrum, the carried
        spectrum with the extrapolated real part in place of the carried one on the predicted modes. The same
        predicted values are returned for the carried map, so a pre-flipped direction is damped by the eigenvalue
        it is expected to have at this rung, where its carried value would give the reversed damping of a mode
        still on the stable side. Refused, with a log line and no prediction, for a predecessor that did not reach
        the pure fixed point and for a mode whose step ratio :math:`(\beta_3 - \beta_2) / (\beta_2 - \beta_1)`
        exceeds ``_MAX_BETA_STEP_RATIO`` in modulus.

        :param state: The dictionary :func:`load_spectrum` returns, with the predecessor keys.
        :param indices: The positions of the carried modes with vectors among the file's certified modes.
        :param lam_pi: All certified eigenvalues of the file.
        :param res: Their Ritz residuals.
        :param beta: Inverse temperature :math:`\beta` of this run.
        :return: The mask of the carried modes with vectors that are flipped ahead of their crossing, and the
            eigenvalues the carried map is built from (the predicted ones on those modes, the carried ones else).
        """
        none = np.zeros(indices.size, dtype=bool)
        lam_c, res_c = lam_pi[indices], res[indices]
        lam_prev = np.asarray(state["lam_prev"], dtype=np.complex128)[indices]
        res_prev = np.asarray(state["res_prev"], dtype=np.float64)[indices]
        beta_prev = np.asarray(state["beta_prev"], dtype=np.float64)[indices]
        beta_last = float(state["beta"])
        matched = np.isfinite(lam_prev.real) & np.isfinite(beta_prev)
        if not matched.any():
            return none, lam_c
        if not bool(state["converged"]):
            self._log_info(
                "Jacobian tracker: the predecessor did not reach the pure fixed point, no cross-rung prediction."
            )
            return none, lam_c
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (beta - beta_last) / (beta_last - beta_prev)
        over = matched & ~(np.abs(ratio) <= _MAX_BETA_STEP_RATIO)
        for idx in np.flatnonzero(over):
            self._log_info(
                f"Jacobian tracker: lambda_Pi={lam_c[idx].real:+.4f}{lam_c[idx].imag:+.4f}j has a step ratio "
                f"{ratio[idx]:.2f} in beta above the cap {_MAX_BETA_STEP_RATIO:g}, no cross-rung prediction."
            )
        usable = matched & ~over
        if not usable.any():
            return none, lam_c
        crossing, pred, pole = predict_rung_crossing(
            lam_c[usable], lam_prev[usable], res_c[usable], res_prev[usable], ratio[usable]
        )
        predicted = none.copy()
        predicted[usable] = crossing
        if not predicted.any():
            return predicted, lam_c
        lam_map = lam_c.copy()
        lam_map[predicted] = pred[crossing] + 1j * lam_c[predicted].imag
        lam_pred = lam_pi.copy()
        lam_pred[indices] = lam_map
        _, p_pred = classify(lam_pred, np.ones(lam_pi.size, dtype=bool), self.p_config, res)
        self._lower_p_eff(p_pred, predicted=True)
        for where, idx in enumerate(np.flatnonzero(usable)[crossing]):
            self._log_info(
                f"Jacobian tracker: cross-rung prediction: lambda_Pi={lam_c[idx].real:+.4f}{lam_c[idx].imag:+.4f}j "
                f"at beta={beta_last:g} from {lam_prev[idx].real:+.4f}{lam_prev[idx].imag:+.4f}j at "
                f"beta={beta_prev[idx]:g} extrapolates to {pred[crossing][where]:+.4f} at beta={beta:g}"
                f"{' (past the pole)' if pole[crossing][where] else ''}, flipped ahead of its crossing."
            )
        return predicted, lam_map

    def _install_carried(self, lam_pi: np.ndarray, u: np.ndarray, flip: np.ndarray | None = None) -> bool:
        r"""
        Builds and installs the per-direction damping of the carried set (:func:`_carried_nonuniform_map`) beside
        the orthogonal reflector :math:`\mathbb{1} - 2 Q Q^{T}` of the real columns the flipped carried Ritz vectors
        span (see :func:`_real_columns`); a set without a flipped mode installs its map alone.

        Resets the persistence, calm and residual-growth counters and clears the hold a growth release left behind,
        as the tracker's own reflector install does, so the release rules of :meth:`update` start counting from this
        installation and the estimate's map may replace the carried one at once. A carried set whose signs carry no
        bounded weighting is installed on the reflector and the scalar damping bound alone. A set that collects no
        column at all, or whose flipped columns are mutually dependent beyond ``_COND_CAP``, is refused as a whole
        and installs nothing, reflector or not.

        :param lam_pi: The carried Ritz values of the modes with vectors.
        :param u: Their normalized Ritz vectors as columns.
        :param flip: Which of them are flipped; ``None`` flips every one.
        :return: Whether the carried set was installed.
        """
        flip = np.ones(lam_pi.size, dtype=bool) if flip is None else np.asarray(flip, dtype=bool)
        nonuniform = _carried_nonuniform_map(lam_pi, u, flip, self._log_info)
        stacked = _real_columns(1.0 - lam_pi, u, flip)[0]
        self._pending = None
        reflector = None
        if stacked.shape[1]:
            singular = np.linalg.svd(stacked, compute_uv=False)
            if singular[-1] > 0.0 and singular[0] / singular[-1] <= _COND_CAP:
                q_mat = np.ascontiguousarray(np.linalg.qr(stacked)[0])
                reflector = (q_mat, np.ascontiguousarray(q_mat.T))
        unbounded = nonuniform is _SIGNS_UNBOUNDED
        if nonuniform is None or (flip.any() and reflector is None) or (unbounded and reflector is None):
            self._log_info(
                "Jacobian tracker: no column was collected from the carried modes, or a flipped one lies in the "
                f"span of the others beyond the condition cap {_COND_CAP:g}, nothing installed."
            )
            return False
        self.q, self.w = reflector if reflector is not None else (None, None)
        self._nonuniform = None if unbounded else nonuniform
        self._applied = None
        self._carried_set = (lam_pi, u, flip)
        self._persist = self._calm = self._grow = self._hold = 0
        reflected = 0 if self.q is None else self.q.shape[1]
        if unbounded:
            self._log_info(
                f"Jacobian tracker: carried set installed on its reflector alone over {reflected} "
                f"direction{'s' if reflected != 1 else ''}, its signs carrying no per-direction damping."
            )
        else:
            self._log_info(
                f"Jacobian tracker: carried set installed on {nonuniform[0].shape[1]} directions, "
                f"{reflected} of them reflected."
            )
        return True

    def update(self, iterates: list[np.ndarray], proposals: list[np.ndarray], allow_flip: bool = True) -> bool:
        r"""
        Runs one monitoring step on the recorded pairs and updates the flip state.

        The estimate is trusted only while the iteration still resolves the Jacobian: the secant step is a finite
        difference of size :math:`\lVert \delta x \rVert` whose roundoff error grows with its inverse, so below a
        relative step of :math:`10^{3}\,\epsilon_{\mathrm{mach}}` (including a last iterate of exactly zero norm)
        the tracked subspace and the damping are frozen. The same absolute noise floor
        :math:`10^{3}\,\epsilon_{\mathrm{mach}}\,\lVert x \rVert` of the iterate - the rounding noise the stored
        iterates carry in the directions the iteration has already converged in - is used to read the secant
        window, so the newest increment survives the absolute floor (only the relative collinearity cutoff against
        a far larger older increment could still drop it) while older ones are dropped once they are noise; a single
        surviving increment still certifies the dominant mode (a rank-one window is what one growing direction
        produces), while a window without any is frozen too. Each Ritz value is certified separately by its
        residual, and a flip is only acted on once a certified unstable mode has been seen on ``_PERSIST``
        *consecutive* updates, which gates the reversed sign of the per-direction map exactly as it gates the
        install: the sign reaches the map together with the reflector that holds it, one built on this update or
        the installed one the direction already lies in, so until the streak is reached, and where the reflector of
        a persisted flip could not be built, every certified unstable direction no installed reflector carries is
        left out of the map and keeps the step the mixing gave it, while a certified direction whose negative real
        part lies inside the undecidable band of :func:`classify`, whose sign is not resolved, takes the damping of
        the damped step (the new :attr:`p_eff`) in the map. The map itself is built from the current estimate on
        every update that certifies a mode, whether or not a reflector is installed, and an update that
        certifies nothing leaves the installed one in force, so a direction certified stable again takes its own
        positive damping and an uncertified one is no part of the map at all; a reflector installed by the tracker
        is released once no certified flip candidate has been seen on ``_PERSIST`` consecutive updates (a map
        without one has no flip to undo and is never released that way, while the map beside a released reflector
        goes with it until the next certifying update), and everything installed is released once the raw residual
        :math:`\lVert S(x) - x \rVert` grows on ``_PERSIST`` consecutive updates, a safety interlock against a step
        that makes the iteration worse. The growth release is tested before the install and independently of a
        flip that still certifies, so a reflection the iteration keeps losing ground under is undone even while the
        raw pairs go on certifying the mode, and nothing is installed in the update that releases one; it holds the
        map off for ``_PERSIST`` further informative updates (a frozen update does not count) and, where a
        reflector was released, clears the persistence streak as well, so a re-install needs ``_PERSIST`` fresh
        certifications and at most one install falls on ``_PERSIST`` updates, while a reflector or a pending carried
        set installed after that brings its map with it, clears the hold and restarts the growth count. A carried
        subspace (see :meth:`carry_in`) is held through calm updates and is released only
        once a certified stable mode overlaps it on ``_PERSIST`` consecutive updates or the residual grows the same
        way, and a certified unstable mode that persists replaces it outright; a carried set kept pending while
        flips were paused is installed on the first update that allows flips while nothing is installed, which
        counts as a switch of the reflected map. Every streak restarts from zero the moment it is interrupted, so a
        mode that flickers in and out of certification never accumulates progress towards a flip or a release
        across the interruption. The persistence streak is treated as satisfied on an update whose chained
        estimates predict a crossing or show a pole jump (see :meth:`_predict`), so such a mode is a flip candidate
        and may install its reflector at once; the growth release still outranks that install, and the calm release
        undoes the flip once the mode stops being a candidate. A window that restarts (fewer than ``_MIN_PAIRS``
        pairs) clears every chain, a frozen update keeps them.

        :param iterates: The recorded iterates as complex core-window arrays of one shape, oldest first.
        :param proposals: The matching raw proposals, never reflected ones, oldest first.
        :param allow_flip: Whether the measured spectrum may be acted on. A scaffolded map is not the physical map,
            so the spectrum is still measured and monitoring continues, but no flip is acted on, the persistence
            and calm streaks are reset, a reflector or map installed from an earlier, physical map is released
            immediately since it does not apply here, and the measured spectrum never replaces the snapshot the
            spectrum file is written from (see :meth:`spectrum_state`); a carried subspace released that way goes
            back to pending and returns once flips are allowed again.
        :return: Whether the tracked subspace changed, i.e. whether the reflected map the mixing sees switched.
            Neither a lowered damping nor the install, refresh or release of the per-direction map alone is such
            a switch: they reach the step outside the map the accelerated history models.
        """
        self.estimated = False
        if len(proposals) < _MIN_PAIRS:
            self._history = []
            self._log_info(f"Jacobian tracker: insufficient history ({len(proposals)} pairs), no estimate.")
            return False

        window_x, window_f = iterates[-TRACKER_PAIRS:], proposals[-TRACKER_PAIRS:]
        x_last, x_prev = to_vec(window_x[-1]), to_vec(window_x[-2])
        x_last_norm = float(np.linalg.norm(x_last))
        if x_last_norm == 0.0:
            self._log_info("Jacobian tracker: last iterate has zero norm, below the roundoff floor, tracker frozen.")
            return False
        # the stored array sets the precision of the floor; the float64 vector would select the float64 epsilon
        floor = noise_floor(np.asarray(window_x[-1]))
        step = float(np.linalg.norm(x_last - x_prev))
        if step < floor:
            self._log_info(f"Jacobian tracker: step {step / x_last_norm:.2e} below the roundoff floor, tracker frozen.")
            return False

        residual_norm = float(np.linalg.norm(to_vec(window_f[-1]) - x_last))
        dx = _increments(window_x)
        df = _increments(window_f)
        theta, u, res, q_kept, form = _ritz_from_increments(dx, df, floor)
        del dx, df
        if theta.size == 0:
            self._log_info("Jacobian tracker: secant window has no informative increments, tracker frozen.")
            return False

        self.last_lam_pi = 1.0 - theta
        self.last_res = res
        self.last_gated = res <= _RITZ_GATE * np.maximum(1.0, np.abs(theta))
        gated = self.last_gated
        self._last_u = u

        flip, p_new = classify(self.last_lam_pi, self.last_gated, self.p_config, res)
        predicted, jump = self._predict(self.last_lam_pi, res, u, gated, allow_flip)
        flip = flip | predicted
        self._predicted = predicted
        forced = bool(predicted.any() or jump.any())
        event = False
        released = ""
        reflected = False  # whether a reflector of the persisted flip was built on this update
        # the hold a growth release of the map leaves behind counts down once per update
        hold_free = may_install = self._hold == 0
        self._hold = max(self._hold - 1, 0)
        if not allow_flip:
            flip[:] = False
            p_new = self.p_eff
            self._persist = self._calm = self._grow = 0
            if self.active:
                event = self.q is not None
                if self._carried_set is not None:
                    self._pending = self._carried_set
                    released = "carried set pending until flips are allowed again, tracked basis released"
                elif event:
                    released = "flips suppressed (allow_flip=False), tracked basis released"
                else:
                    released = "flips suppressed (allow_flip=False), per-direction damping released"
                self.q = self.w = self._nonuniform = self._applied = self._carried_set = None
        else:
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
                self._persist = max(self._persist, _PERSIST)
            # only a reflector holds a flip the calm release can undo; a map without one lives while it certifies
            calm_now = self.q is not None and not flip.any()
            if calm_now and self._carried_set is not None:
                # a carried basis holds until a certified STABLE mode is seen inside it (or the residual grows)
                overlap = np.linalg.norm(self.w @ u[:, gated], axis=0) > _FLIP_OVERLAP
                band = np.maximum(_MARGIN, res[gated])
                calm_now = bool((overlap & (self.last_lam_pi[gated].real > band)).any())
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
                elif self._carried_set is not None:
                    released = "a certified stable mode lies in the carried basis, carried basis released"
                self.q = self.w = self._nonuniform = self._applied = self._carried_set = None
                self._grow = 0
                may_install = False
            elif flip.any() and self._persist >= _PERSIST:
                reflector = self._flipped_reflector(flip, q_kept, form)
                if reflector is None:
                    self._log_info(
                        "Jacobian tracker: the flipped directions could not be assembled into a reflector, none "
                        "installed."
                    )
                else:
                    reflected = True
                    if (
                        self.q is None
                        or self._carried_set is not None
                        or self.q.shape[1] != reflector[0].shape[1]
                        or self._principal_angle(reflector[0]) > _SUBSPACE_ANGLE
                    ):
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
        if allow_flip and self.last_gated.any():
            self._last_certified = (
                self.last_lam_pi,
                self.last_res,
                self.last_gated,
                self._flip,
                self._last_u,
                self._predicted,
            )
        self._lower_p_eff(p_new)

        self.n_updates += 1
        self.estimated = True
        self._log_spectrum(flip, released)
        return event

    def _predict(
        self, lam_pi: np.ndarray, res: np.ndarray, u: np.ndarray, gated: np.ndarray, allow_flip: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Chains the certified modes of this estimate to those of the previous informative updates by the overlap of
        their Ritz vectors (:func:`_match_modes`), keeps the last ``_CHAIN_LENGTH`` estimates (values and residuals
        of every one, the Ritz vectors of the newest only, since the next match reads nothing else), and reads the
        predicted crossings (:func:`predict_crossing`) and the pole jumps (:func:`detect_pole_jump`) off the chains.
        A mode that fails the certificate or the overlap on one update starts a new chain; the history itself is
        cleared by the window restart of :meth:`update`. Every prediction, pole jump and pole-type approach is
        logged, with flips paused as well, where nothing is acted on.

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
                f"Jacobian tracker: predicted crossing of lambda_Pi={lam_c[idx].real:+.4f}{lam_c[idx].imag:+.4f}j "
                f"(extrapolated {pred[idx]:+.4f}, band {max(_MARGIN, res_c[idx]):.4f}, error bar {delta[idx]:.4f})"
                f"{acted}."
            )
        for idx in np.flatnonzero(jumped):
            self._log_info(
                f"Jacobian tracker: pole crossing of lambda_Pi from {lam_1[idx].real:+.4f} to {lam_c[idx].real:+.4f}"
                f"{', flips paused' if not allow_flip else ', flipped at once'}."
            )
        for idx in np.flatnonzero(approach):
            self._log_info(
                f"Jacobian tracker: pole-type approach of lambda_Pi={lam_c[idx].real:+.4f}{lam_c[idx].imag:+.4f}j "
                f"(modulus {abs(lam_c[idx]):.2f} growing), no prediction."
            )
        return predicted, jump

    def reflect(self, proposal: np.ndarray, iterate: np.ndarray) -> np.ndarray:
        r"""
        Reflects the residual :math:`r = S(x) - x` of the proposal on the tracked invariant subspace, returning

        .. math:: \tilde{S}(x) = x + r - 2\,Q\,(W r),

        which reverses the residual of every flipped direction and leaves every other one exactly where it was.
        The reflector carries no damping, so the preconditioned map an accelerated scheme models stays the same
        map for as long as the subspace does; the per-direction damping of the certified span is applied to the
        mixed iterate afterwards (:meth:`stabilize_step`).

        The map is a preconditioner of the proposal, not a mixer: it shares its fixed points with the plain map, so
        an accelerated mixing keeps them.

        :param proposal: The raw proposal :math:`S(x)` on the core window.
        :param iterate: The current iterate :math:`x` on the same window.
        :return: The reflected proposal in the dtype of ``proposal``, or ``proposal`` itself while no subspace is
            tracked.
        """
        if self.q is None:
            return proposal
        x = to_vec(iterate)
        residual = to_vec(proposal) - x
        residual = residual - 2.0 * (self.q @ (self.w @ residual))
        return to_mat(x + residual, proposal.shape).astype(proposal.dtype, copy=False)

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
        :math:`1 - s_\alpha p_\alpha \lambda_{\Pi,\alpha}` whatever the damping did to it, while a direction the
        estimate does not certify keeps the step the mixing took, since both the projector and the map annihilate
        it. The caller applies this to its damped steps only: an accelerated step models the map with the same
        secants that certified the span, so its step there is the more informed one.

        :param x_new: The damped iterate on the core window.
        :param iterate: The iterate :math:`x_n` the step started from, on the same window.
        :param residual: The raw residual :math:`S(x_n) - x_n` on the same window.
        :return: The stabilized iterate in the dtype of ``x_new``, or ``x_new`` itself while no map is installed.
        """
        if self._nonuniform is None:
            return x_new
        span, dual, damping = self._nonuniform
        x = to_vec(iterate)
        step = to_vec(x_new) - x
        step = step - span @ (dual @ step) + span @ (damping @ (dual @ to_vec(residual)))
        return to_mat(x + step, x_new.shape).astype(x_new.dtype, copy=False)

    def _held_flipped(self, u: np.ndarray) -> np.ndarray:
        """
        Whether each Ritz direction already lies in the installed flipped subspace, i.e. whether its cosine with
        the orthonormal basis of the installed reflector exceeds ``_FLIP_OVERLAP``. Nothing is held while no
        reflector is installed, so a flip no installed subspace carries yet has to earn its persistence streak
        first.

        :param u: The Ritz vectors as columns.
        :return: The mask of the directions the installed flipped subspace already carries.
        """
        if self.q is None:
            return np.zeros(u.shape[1], dtype=bool)
        return np.linalg.norm(self.q.T @ u, axis=0) / np.linalg.norm(u, axis=0) > _FLIP_OVERLAP

    def _certified_damping(self) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Returns, for every certified direction of the last estimate, its eigenvalue :math:`\lambda_\Pi` and the
        signed damping it takes on the next step. While the map built from the estimate is installed, a direction it
        acts on takes the signed damping recorded for its block, exact for the assembled map and for the uniform
        fallback alike, i.e. :math:`-p_\alpha` where the map carries the reversed sign; a certified unstable
        direction the map leaves out (see :meth:`update`) takes the scalar :attr:`p_eff` of the damped step it
        keeps. While a carried map is installed, which records no such damping, a direction counts as mapped when
        its Ritz vector overlaps the installed span by more than ``_FLIP_OVERLAP`` (a genuine cosine, since that
        span is orthonormal) and takes the Rayleigh quotient of the installed map, exact there because the carried
        weighting commutes with its own vectors. Every other certified direction takes :attr:`p_eff`, and so does
        every one of them while no map is installed.

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
            mapped = ~np.isnan(applied)
            damping[mapped] = applied[mapped]
            return lam_pi, damping
        span, dual, weighting = self._nonuniform
        u = self._last_u[:, gated]
        coords = span.T @ u
        mapped = np.linalg.norm(coords, axis=0) > _FLIP_OVERLAP
        gain = np.sum(np.conj(coords) * (weighting @ (dual @ u)), axis=0).real / np.linalg.norm(u, axis=0) ** 2
        damping[mapped] = gain[mapped]
        return lam_pi, damping

    def predicted_rate(self) -> float:
        r"""
        Returns the largest contraction factor the last certified spectrum predicts,
        :math:`|1 - s_\alpha p_\alpha \lambda_{\Pi,\alpha}|` over the certified directions with the signed
        damping each of them takes on the next step (see :meth:`_certified_damping`): the damping of the installed
        map where it acts on the direction, i.e. :math:`|1 + p_\alpha \lambda_{\Pi,\alpha}|` where the map
        carries the reversed sign, and the bound factor :math:`|1 - p_{\mathrm{eff}} \lambda_{\Pi,\alpha}|` of
        the damped step elsewhere.

        :return: The predicted rate over the certified modes, or ``0.0`` while there is no estimate.
        """
        lam_pi, damping = self._certified_damping()
        return float(np.max(np.abs(1.0 - damping * lam_pi))) if lam_pi.size else 0.0

    @property
    def certified_margin(self) -> float:
        r"""
        The distance of the certified spectrum from the stability boundary,
        :math:`\epsilon = \min_\alpha |\mathrm{Re}\,\lambda_{\Pi,\alpha}|` over the certified modes of the last
        estimate held whose real part lies outside their undecidable band :math:`\max(m, \mathrm{res}_\alpha)`,
        the band :func:`classify` refuses to sign, or ``0.0`` when no certified mode lies outside it.
        """
        if self.last_lam_pi is None or not self.last_gated.any():
            return 0.0
        real_part = np.abs(self.last_lam_pi[self.last_gated].real)
        outside = real_part > np.maximum(_MARGIN, self.last_res[self.last_gated])
        return float(np.min(real_part[outside])) if outside.any() else 0.0

    def minimum_iterations(self) -> int:
        r"""
        Returns the number of iterations
        :math:`n_{\min} = \min(n_{\mathrm{cap}}, \max_\alpha \lceil 1 / (|\mathrm{Re}\,\lambda_{\Pi,\alpha}|\,
        \min(1, p_\alpha / c)) \rceil)` the slowest certified direction needs to be driven away from the starting
        point, over the certified directions outside their undecidable band (the ones the :attr:`certified_margin` is
        read from), each at the damping it actually gets (see :meth:`_certified_damping`): the modulus of its applied
        damping where the installed map acts on it, the scalar :attr:`p_eff` of the damped step it keeps elsewhere.
        The count belongs to the undamped constant of the damping bound, so every damping is divided by the constant
        :math:`c` that placed it at the vertex of the stability parabola and clipped at one, which no damping exceeds:
        a flat direction the map steps at :math:`p_\alpha = 1` counts the :math:`\lceil 1 / \epsilon \rceil` of App.
        G4. The division recovers the constant only where the vertex formula set the damping; where it is the
        configured ceiling or the floor instead, the count is off by the ratio of the two. The count is capped at
        ``_MIN_ITERATION_CAP``, and the quotient is taken a relative :math:`10^{-9}` short, so a count that falls on
        an integer is decided there rather than by the last bits of the eigenvalue it came from. Zero while no
        certified mode lies outside its undecidable band, where the estimate carries no finite count.

        :return: The minimum iteration count.
        """
        lam_pi, damping = self._certified_damping()
        if lam_pi.size == 0:
            return 0
        real_part = np.abs(lam_pi.real)
        outside = real_part > np.maximum(_MARGIN, self.last_res[self.last_gated])
        if not outside.any():
            return 0
        rate = real_part * np.minimum(1.0, np.abs(damping) / _DAMPING_C)
        # taken a relative 1e-9 short, so the last bits of an eigenvalue cannot raise a count that lands on an integer
        counts = np.ceil((1.0 - 1e-9) / np.maximum(rate[outside], np.finfo(np.float64).tiny))
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
        complementary one, so :math:`R` reverses every Ritz direction of a flipped eigenvalue, fixes every other
        one exactly whatever their mutual overlaps, and leaves the directions outside the Ritz span alone. It is
        applied as :math:`r - 2 Q (W r)` by :meth:`reflect`, with :math:`Q_K (Z_2 + Z_1 X) = Q R_f` the thin QR and
        :math:`W = R_f Z_2^{T} Q_K^{T}`, so that :math:`Q W = Q_K \Pi Q_K^{T}` and :math:`R^{2} = \mathbb{1}`. A
        flipped and a complementary eigenvalue that nearly coincide make the Sylvester equation ill-conditioned:
        a solution norm above ``_COND_CAP``, a reordering the decomposition itself refuses, or a conjugate pair
        flipped on one side only means the two subspaces are not separable and no reflector is built, and a
        projector whose spectral norm exceeds ``_OBLIQUE_CAP`` is replaced by the orthogonal reflection
        :math:`W = Q^{T}` of the same flipped span, which reverses it just as exactly but no longer fixes an
        overlapping complementary direction.

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
        if float(np.linalg.norm(r_flip @ z_flip.T, 2)) > _OBLIQUE_CAP:
            self._log_info(
                "Jacobian tracker: the flipped and the complementary subspace are closer than the obliquity cap "
                f"{_OBLIQUE_CAP:g}, the orthogonal reflection of the flipped span is used."
            )
            return np.ascontiguousarray(q), np.ascontiguousarray(q.T)
        return np.ascontiguousarray(q), np.ascontiguousarray(r_flip @ z_flip.T @ q_kept.T)

    def _principal_angle(self, q_new: np.ndarray) -> float:
        """
        Returns the largest principal angle between the tracked subspace and a newly assembled one of equal
        dimension, i.e. the arc cosine of the smallest singular value of their overlap.

        :param q_new: The newly assembled orthonormal basis.
        :return: The largest principal angle in radians.
        """
        overlap = np.linalg.svd(self.q.T @ q_new, compute_uv=False)
        return float(np.arccos(np.clip(overlap.min(), -1.0, 1.0)))

    def _lower_p_eff(self, p_new: float, predicted: bool = False) -> None:
        """
        Lowers the effective damping to ``p_new`` when that is smaller than the current value, and never raises it.
        Warns once when the measured spectrum has pushed it down to its floor; a bound set by the predicted spectrum
        of a carried set warns separately, and leaves that one warning to the measured spectrum.

        :param p_new: The candidate damping.
        :param predicted: Whether the bound comes from the predicted spectrum of the carried modes.
        :return: None.
        """
        if p_new >= self.p_eff - 1e-12:
            return
        previous, self.p_eff = self.p_eff, p_new
        if self._logger is None or p_new > _MIXING_FLOOR:
            return
        if predicted:
            self._logger.warning(
                f"Jacobian tracker: the effective mixing is at the floor ({_MIXING_FLOOR:g}) from the predicted "
                f"spectrum of the carried modes at this rung's temperature; the carried spectrum itself allowed "
                f"p_eff={previous:.4f}."
            )
        elif not self._floor_warned:
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
        Logs the dominant measured modes with their residual certificate and flip decision, followed by the flip
        count, the effective damping and the predicted convergence rate. An uncertified mode carries only its
        "uncertified" tag, since the residual gate failed and there is nothing else to report about it; a
        certified mode carries "certified" plus exactly one of "flip", "predicted flip" (a flip ahead of the crossing,
        see :meth:`_predict`), "stable", "marginal" for a real part inside its undecidable band (see
        :func:`classify`) or "unstable, suppressed" for a clearly negative real part that is not flipped because
        flips are paused. When the update released an installed basis or
        map with a message (the release forced while flips are paused, the residual-growth release, the calm
        release of a carried basis, or a map the estimate could not rebuild), that message is folded into the same
        line.

        :param flip: Which modes are flipped.
        :param released: The release message of this update, or an empty string.
        """
        if self._logger is None:
            return
        modes = []
        for idx in np.argsort(-np.abs(self.last_lam_pi))[:4]:
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
            modes.append(f"lambda_Pi={lam.real:+.4f}{lam.imag:+.4f}j (res {self.last_res[idx]:.1e}, {tag})")
        prefix = f"{released}; " if released else ""
        self._logger.info(
            f"Jacobian tracker: {prefix}{', '.join(modes)}; {int(np.count_nonzero(flip))} flipped, "
            f"p_eff={self.p_eff:.4f}, rho={self.predicted_rate():.4f}."
        )

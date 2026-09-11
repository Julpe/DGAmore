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
and applies the flip as an orthogonal reflection of the residual. No evaluation of the proposal map happens here,
and the module is pure numpy: it imports neither MPI nor the configuration.

Follows H. Essl, S. Rohshap, M. Gievers, M. Wallerberger, A. Toschi and A. Kauch, arXiv:2606.04936, and
H. Essl, M. Reitner, E. Kozik and A. Toschi, Phys. Rev. Lett., doi:10.1103/zjy7-4jqd.
"""

from collections.abc import Callable

import numpy as np

_MARGIN = 1e-2  # a real part of lambda_Pi below this in modulus is undecidable: not flipped, and it enters the
# damping bound with the margin in place of its real part
_DAMPING_C = 0.5  # safety factor of the damping bound
_MIXING_FLOOR = 0.01  # the effective damping is never lowered below this
_RITZ_GATE = 3e-2  # Ritz residual gate, relative to max(1, |theta|)
_PERSIST = 3  # updates carrying a certified unstable mode before the first flip
_MONITOR_WINDOW = 6  # number of secant differences the estimate looks at (the tracker uses _MONITOR_WINDOW + 1 pairs)
_MIN_PAIRS = 3  # fewer pairs than this carry no estimate
_SUBSPACE_ANGLE = 0.1  # rad; largest principal angle above which the flipped basis is replaced
_COLLINEAR_CUTOFF = 1e-8  # relative cutoff below which a secant column counts as collinear
_NOISE_FACTOR = (
    1000.0  # noise floor in machine epsilons of the iterate norm (also the step gate); noise gain of R^-1 <= 1e-3
)
_COND_CAP = 1e3  # condition number of the certified Ritz columns above which no reflector is built
_REAL_CUTOFF = 1e-8  # relative cutoff below which a Ritz value counts as real
_FLIP_OVERLAP = 0.5  # |Q^T u| above which a Ritz vector counts as lying in the flipped subspace, used by
# the predicted rate and the calm release of a carried basis
TRACKER_PAIRS = _MONITOR_WINDOW + 1  # (iterate, proposal) pairs the tracker keeps; the mixing history is trimmed to it
_PREEMPT_BAND = (
    0.1  # a carried certified mode with a real part of lambda_Pi below this is flipped ahead of its crossing
)
SPECTRUM_FILE = "jacobian_spectrum.npz"  # file a run writes its final certified spectrum to, read by its successor
EIGENVALUE_FILE = "jacobian_eigenvalues.npy"  # per-iteration leading lambda_Pi Ritz values of the running tracker
EIGENVALUE_RESIDUAL_FILE = "jacobian_eigenvalue_residuals.npy"  # the matching per-iteration Ritz residuals
DAMPING_FILE = "jacobian_damping.npy"  # per-iteration effective damping and flipped-direction bound


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
    path (:meth:`JacobianTracker.update`) builds the increments directly and calls :func:`_ritz_from_increments`;
    this function is the reference form of the same computation, kept for the derivation and the tests.

    :param xs: Real column stack of the iterates, oldest first, shape ``[n_real, n]`` with ``n >= 3``.
    :param fs: Real column stack of the matching proposals, oldest first, shape ``[n_real, n]``.
    :param noise_floor: Absolute norm below which an increment carries no information beyond rounding noise. An
        increment is dropped when the part of it orthogonal to the more recent kept ones stays below the larger of
        this floor and the relative collinearity cutoff.
    :return: The Ritz values :math:`\theta`, the Ritz vectors :math:`u` as columns, and the Ritz residuals; all
        empty when no increment survives the noise floor.
    """
    return _ritz_from_increments(np.diff(xs, axis=1), np.diff(fs, axis=1), noise_floor)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Runs the Rayleigh-Ritz of :func:`secant_ritz` on the secant increments themselves, so that a caller building
    the increments column by column never holds the iterate and proposal stacks as well. The tall factor
    :math:`Q` of the first QR is released as soon as the kept basis :math:`Q_K = Q Q_2` exists, and
    :math:`R_2^{-1}` is applied as a triangular solve rather than formed.

    :param dx: Secant increments of the iterates, shape ``[n_real, m]``, oldest first.
    :param df: Matching increments of the proposals, shape ``[n_real, m]``.
    :param noise_floor: Absolute norm below which an increment carries no information beyond rounding noise.
    :return: The Ritz values, the Ritz vectors as columns, and the Ritz residuals; all empty when no increment
        survives the noise floor.
    """
    q, r = np.linalg.qr(dx)
    keep = _independent_columns(r, noise_floor)
    if keep.size == 0:
        empty = np.zeros(0, dtype=np.complex128)
        return empty, np.zeros((dx.shape[0], 0), dtype=np.complex128), np.zeros(0, dtype=np.float64)
    q2, r2 = np.linalg.qr(r[:, keep])
    q_kept = q @ q2
    del q
    jq = np.linalg.solve(r2.T, df[:, keep].T).T
    theta, y = np.linalg.eig(q_kept.T @ jq)
    u = q_kept @ y
    res = np.empty(theta.size, dtype=np.float64)
    for col in range(theta.size):
        res[col] = np.linalg.norm(jq @ y[:, col] - theta[col] * u[:, col]) / np.linalg.norm(u[:, col])
    return theta, u, res


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


def _well_conditioned(columns: np.ndarray) -> bool:
    """
    Whether the condition number of a column set stays at or below ``_COND_CAP``.

    :param columns: The column set, shape ``[n_real, k]``.
    :return: ``True`` when the smallest singular value is positive and the condition number is within the cap.
    """
    sv = np.linalg.svd(columns, compute_uv=False)
    return sv[-1] > 0.0 and sv[0] / sv[-1] <= _COND_CAP


def load_spectrum(path: str) -> dict[str, np.ndarray]:
    """
    Loads the spectrum file a previous run wrote (see :meth:`JacobianTracker.save_spectrum`).

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
    estimate flipped is released again by the same rule. The flip itself is an orthogonal reflection of the residual
    on the tracked subspace,

    .. math:: x_{n+1} = x_n + p_{\mathrm{eff}}\,(\mathbb{1} - 2 Q W)\,(S(x_n) - x_n),

    with :math:`Q` an orthonormal basis of the flipped Ritz directions and :math:`W` their dual rows on the whole
    certified Ritz set (see :meth:`_flipped_reflector`), so that :math:`W u = 0` for every certified stable Ritz
    vector :math:`u`: the reflection flips exactly the flipped directions and leaves the other certified ones fixed
    even when the Ritz vectors of a non-normal :math:`J` overlap. On an invariant subspace the Jacobian of the
    stabilized iteration then has the spectrum of the damped one with the flipped eigenvalues :math:`\lambda_M`
    replaced by :math:`2 - \lambda_M`. With :math:`W = Q^{T}`, the case of no certified stable mode alongside, the
    reflection is orthogonal.

    :attr:`p_eff` is the largest damping the certified spectrum allows, never above the configured value and never
    raised again within a run. :attr:`p_flip` is the bound of the flipped directions alone, the damping an accelerated
    step takes while a reflection is in force; it is refreshed on every update that certifies a flipped mode while a
    basis is installed (a held carried basis, never angle-checked against the estimate, only ever tightens it), and
    reset to the configured value on release, so unlike :attr:`p_eff` it is not monotone.
    :attr:`estimated` tells whether the last :meth:`update` call produced a fresh spectrum estimate. Which iterations
    apply which bound is the mixing's decision: every damped Picard step (linear mixing, the warm-up and the fallbacks
    of the accelerated schemes) takes :attr:`p_eff`, and a quasi-Newton step takes :attr:`p_flip` while a basis is
    installed, since a flipped mode iterates with :math:`1 + p\lambda_\Pi` and is stable only under that bound, and the
    configured parameter without one, because the whole-spectrum bound describes the damped iteration and not the
    damping of such a step.
    """

    def __init__(self, p_config: float, logger=None):
        """
        Initializes an inactive tracker: nothing flipped, no spectrum measured yet, damping at its configured value.

        :param p_config: The configured damping, i.e. the ceiling of :attr:`p_eff` and the reset value of
            :attr:`p_flip`.
        :param logger: Logger receiving the measured spectrum, or ``None`` to stay silent.
        """
        self.p_config = float(p_config)
        self.p_eff = float(p_config)
        self.p_flip = float(p_config)
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
        self._last_residual_norm = np.inf
        self._floor_warned = False
        self._carried = False  # whether the installed basis came from a predecessor run (carry_in)
        self._pending = None  # carried (lam_pi, u, flip, res) waiting for flips to be allowed
        self._carried_set = None  # installed carried (lam_pi, u, flip, res), re-pended on a scaffold-forced release
        self._last_certified = None  # (lam_pi, res, gated, flip, u) of the last update that certified a mode

    @property
    def active(self) -> bool:
        """Whether a flipped subspace is currently tracked."""
        return self.q is not None

    @property
    def n_pairs(self) -> int:
        """The number of trailing (iterate, proposal) pairs that :meth:`update` looks at."""
        return TRACKER_PAIRS

    def spectrum_state(self, shape: tuple[int, ...], beta: float, converged: bool) -> dict:
        r"""
        Collects the certified part of the last update that certified a mode with flips allowed, or the set
        :meth:`carry_in` installed, whatever ``allow_flip`` was then, for a successor run: every certified Ritz value
        with its residual and flip decision, and, for the modes a successor could flip (real part of :math:`\lambda_\Pi`
        below ``_PREEMPT_BAND``), the Ritz vector in single precision. The real and the imaginary part of a Ritz vector
        are each rebuilt into the complex window array they flatten from and stored as one column each; a real mode's
        imaginary column is zero. The last update of a run is often one whose estimate is frozen, certifies nothing, or
        ran with flips paused (a scaffolded map never replaces the snapshot even when it does certify something, since
        that map never ran physically), so the modes come from the stored snapshot rather than from the last estimate.
        The mode arrays are empty when nothing was ever certified. The flip decisions and :attr:`p_eff` are recorded for
        inspection and are not read back by :meth:`carry_in`, which re-decides the flips from ``_PREEMPT_BAND`` and
        re-derives the damping bound from the carried spectrum itself.

        :param shape: The complex window shape ``(kx, ky, kz, nb, nb, 2 niv_core)`` of the iterated array.
        :param beta: The inverse temperature of the run.
        :param converged: Whether the run reached the pure fixed point.
        :return: The state.
        """
        n_complex = int(np.prod(shape))
        if self._last_certified is None:
            lam_pi, res = np.zeros(0, dtype=np.complex128), np.zeros(0, dtype=np.float64)
            flip, u = np.zeros(0, dtype=bool), np.zeros((2 * n_complex, 0), dtype=np.complex128)
        else:
            lam_pi, res, gated, flip, u = self._last_certified
            lam_pi, res = lam_pi[gated].astype(np.complex128), res[gated].astype(np.float64)
            flip, u = flip[gated].astype(bool), u[:, gated]
        stored = lam_pi.real < _PREEMPT_BAND
        re_cols = [to_mat(u[:, j].real, shape).reshape(-1).astype(np.complex64) for j in np.flatnonzero(stored)]
        im_cols = [to_mat(u[:, j].imag, shape).reshape(-1).astype(np.complex64) for j in np.flatnonzero(stored)]
        return {
            "lam_pi": lam_pi,
            "res": res,
            "flip": flip,
            "stored": stored,
            "u_re": np.column_stack(re_cols) if re_cols else np.zeros((n_complex, 0), dtype=np.complex64),
            "u_im": np.column_stack(im_cols) if im_cols else np.zeros((n_complex, 0), dtype=np.complex64),
            "shape": np.asarray(shape, dtype=np.int64),
            "beta": np.float64(beta),
            "p_eff": np.float64(self.p_eff),
            "converged": np.bool_(converged),
        }

    def save_spectrum(self, path: str, shape: tuple[int, ...], beta: float, converged: bool) -> None:
        """
        Writes :meth:`spectrum_state` to ``path`` as a compressed ``.npz``, always: a run that certified nothing
        still records the spectrum and the damping it ended with.

        :param path: Destination file.
        :param shape: The complex window shape of the iterated array.
        :param beta: The inverse temperature of the run.
        :param converged: Whether the run reached the pure fixed point.
        :return: None.
        """
        state = self.spectrum_state(shape, beta, converged)
        np.savez_compressed(path, **state)
        self._log_info(
            f"Jacobian tracker: saved {state['lam_pi'].size} certified modes ({int(state['stored'].sum())} with "
            f"vectors) to {path}."
        )

    def carry_in(self, state: dict, expand: Callable[[np.ndarray], np.ndarray], allow_flip: bool = True) -> bool:
        r"""
        Installs the certified spectrum a predecessor run ended with, before the first step of this run.

        Every stored Ritz vector is rebuilt from its two columns through ``expand`` (the caller's re-gridding onto this
        run's window), as a real vector plus ``1j`` times another, and normalized; columns that come back empty are
        dropped. A carried mode is flipped when its real part lies below ``_PREEMPT_BAND``: the modes the predecessor
        had flipped, the marginal ones and those about to cross, following the smooth evolution of the Jacobian with the
        parameter (supplemental material of Phys. Rev. Lett. doi:10.1103/zjy7-4jqd, arXiv:2502.01420). Flipping a
        still-stable mode ahead of its crossing is an extension of the papers and is not free: a mode with a real part
        between zero and ``_PREEMPT_BAND`` iterates flipped with :math:`|1 + p\,\lambda_\Pi| > 1` at every
        :math:`p > 0`, which no damping bound can cure, so a wrong preemptive flip grows the error along it by the
        factor :math:`|1 + p\,\lambda_\Pi|` per update (at most ``1 + p_eff * _PREEMPT_BAND`` for a real mode, more for
        a complex one) until one of the release rules of :meth:`update` undoes it, which needs ``_PERSIST`` consecutive
        updates. The effective damping becomes the bound of the whole carried spectrum under this run's ceiling, applied
        at once even when the flip has to wait for a scaffold's release; a carried spectrum without a certified mode
        carries no bound and leaves the damping where it is. An accelerated step under the carried flip takes
        :attr:`p_flip`, the bound of the flipped directions alone. The damping the predecessor ended with is not read
        back, so that one transient early in a ladder cannot pin every later rung to the value it forced. The
        reflector of a carried set is orthogonal (no stable columns are stored).

        :param state: The dictionary :func:`load_spectrum` returns.
        :param expand: Callable mapping one stored column to the real vector on this run's window.
        :param allow_flip: Whether a reflector may be installed now; otherwise the flip set is kept pending and
            installed on the first update that allows flips while nothing else is installed.
        :return: Whether a reflector is installed.
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
        lam_c, res_c = lam_pi[stored][keep], res[stored][keep]
        # the same predicate that spectrum_state used for stored, so every mode flips unless saved with another band
        flip = lam_c.real < _PREEMPT_BAND
        self.last_lam_pi, self.last_res, self._last_u = lam_c, res_c, u
        self.last_gated = np.ones(lam_c.size, dtype=bool)
        self._flip = flip
        self._last_certified = (lam_c, res_c, self.last_gated, flip, u)
        if not flip.any():
            # only reachable from a file written with a different _PREEMPT_BAND
            self._log_info("Jacobian tracker: no carried mode lies below the flip band, nothing to flip.")
            return False
        if not allow_flip:
            self._pending = (lam_c, u, flip, res_c)
            self._log_info("Jacobian tracker: flips are paused by a scaffold, the carried flip waits for its release.")
            return False
        return self._install_carried(lam_c, u, flip, res_c)

    def _install_carried(self, lam_pi: np.ndarray, u: np.ndarray, flip: np.ndarray, res: np.ndarray) -> bool:
        """
        Builds and installs the reflector of a carried flip set.

        Sets ``last_gated`` itself (every carried mode is gated) since it is also called on a pending set from
        :meth:`update`, not only from :meth:`carry_in`. Also resets the persistence, calm and residual-growth
        counters, so the release rules of :meth:`update` start counting from this installation.

        :param lam_pi: The carried Ritz values of the modes with vectors.
        :param u: Their normalized Ritz vectors as columns.
        :param flip: Which of them are flipped.
        :param res: Their Ritz residuals.
        :return: Whether the reflector was installed.
        """
        self.last_gated = np.ones(lam_pi.size, dtype=bool)
        reflector = self._flipped_reflector(1.0 - lam_pi, u, flip)
        self._pending = None
        if reflector is None:
            self._log_info(
                "Jacobian tracker: the carried flipped directions could not be assembled into a reflector (none "
                f"collected, or mutually dependent beyond the condition cap {_COND_CAP:g}), reflector not "
                "installed."
            )
            return False
        self.q, self.w = reflector
        self.p_flip = self._flip_bound(lam_pi, flip, res)
        self._carried = True
        self._carried_set = (lam_pi, u, flip, res)
        self._persist = self._calm = self._grow = 0
        self._log_info(f"Jacobian tracker: carried reflector installed on {self.q.shape[1]} directions.")
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
        *consecutive* updates; a subspace installed by the tracker is released once no certified flip candidate has
        been seen on ``_PERSIST`` consecutive updates, or once the raw residual :math:`\lVert S(x) - x \rVert` grows
        on ``_PERSIST`` consecutive updates, a safety interlock against a flip that makes the iteration worse. The
        growth release is tested before the install and independently of a flip that still certifies, so a
        reflection the iteration keeps losing ground under is undone even while the raw pairs go on certifying the
        mode, and no basis is installed in the update that releases one; it also clears the persistence streak, so a
        re-install needs ``_PERSIST`` fresh certifications and at most one install falls on ``_PERSIST`` updates. A
        carried subspace (see :meth:`carry_in`) is held through calm updates and is released only once a certified
        stable mode overlaps it on ``_PERSIST`` consecutive updates or the residual grows the same way, and a
        certified unstable mode that persists replaces it outright; a carried set kept pending while flips were
        paused is installed on the first update that allows flips while nothing is installed, which counts as a
        switch of the reflected map. Every streak restarts from zero the moment it is interrupted, so a mode that
        flickers in and out of certification never accumulates progress towards a flip or a release across the
        interruption.

        :param iterates: The recorded iterates as complex core-window arrays of one shape, oldest first.
        :param proposals: The matching raw proposals, never reflected ones, oldest first.
        :param allow_flip: Whether the measured spectrum may be acted on. A scaffolded map is not the physical map,
            so the spectrum is still measured and monitoring continues, but no flip is acted on, the persistence
            and calm streaks are reset, a subspace already tracked from an earlier, physical map is released
            immediately since it does not apply here, and the measured spectrum never replaces the snapshot the
            spectrum file is written from (see :meth:`spectrum_state`); a carried subspace released that way goes
            back to pending and returns once flips are allowed again.
        :return: Whether the tracked subspace changed, i.e. whether the reflected map the mixing sees switched. A
            lowered damping is not such a switch: it reaches the mixing through :attr:`p_eff` alone.
        """
        self.estimated = False
        if len(proposals) < _MIN_PAIRS:
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
        theta, u, res = _ritz_from_increments(dx, df, floor)
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
        event = False
        released = ""
        if not allow_flip:
            flip[:] = False
            p_new = self.p_eff
            self._persist = self._calm = self._grow = 0
            if self.q is not None:
                if self._carried:
                    self._pending = self._carried_set
                    released = "carried set pending until flips are allowed again, tracked basis released"
                else:
                    released = "flips suppressed (allow_flip=False), tracked basis released"
                self.q = self.w = None
                self.p_flip = self.p_config
                self._carried = False
                event = True
        else:
            if self._pending is not None and self.q is None:
                event = self._install_carried(*self._pending)
                self.last_gated = gated
            self._persist = self._persist + 1 if flip.any() else 0
            calm_now = self.q is not None and not flip.any()
            if calm_now and self._carried:
                # a carried basis holds until a certified STABLE mode is seen inside it (or the residual grows)
                overlap = np.linalg.norm(self._dual @ u[:, gated], axis=0) > _FLIP_OVERLAP
                band = np.maximum(_MARGIN, res[gated])
                calm_now = bool((overlap & (self.last_lam_pi[gated].real > band)).any())
            self._calm = self._calm + 1 if calm_now else 0
            self._grow = self._grow + 1 if (self.q is not None and residual_norm > self._last_residual_norm) else 0
            if self.q is not None and (self._grow >= _PERSIST or self._calm >= _PERSIST):
                if self._grow >= _PERSIST:
                    released = "residual grew on consecutive updates, tracked basis released"
                    # a growth release outranks a persisting flip, so a re-install needs _PERSIST fresh certifications
                    self._persist = 0
                elif self._carried:
                    released = "a certified stable mode lies in the carried basis, carried basis released"
                self.q = self.w = None
                self.p_flip = self.p_config
                self._carried = False
                self._carried_set = None
                self._grow = 0
                event = True
            elif flip.any() and self._persist >= _PERSIST:
                reflector = self._flipped_reflector(theta, u, flip)
                if reflector is None:
                    self._log_info(
                        "Jacobian tracker: the flipped directions could not be assembled into a reflector (none "
                        f"collected, or mutually dependent beyond the condition cap {_COND_CAP:g}), reflector "
                        "not installed."
                    )
                else:
                    if (
                        self.q is None
                        or self._carried
                        or self.q.shape[1] != reflector[0].shape[1]
                        or self._principal_angle(reflector[0]) > _SUBSPACE_ANGLE
                    ):
                        self.q, self.w = reflector
                        self._carried = False
                        self._carried_set = None
                        event = True
        self._last_residual_norm = residual_norm

        self._flip = flip
        if self.q is not None and flip.any():
            # a carried basis never had the angle check against this estimate, so its own bound can only tighten
            p_new_flip = self._flip_bound(self.last_lam_pi, flip, res)
            self.p_flip = min(p_new_flip, self.p_flip) if self._carried else p_new_flip
        if allow_flip and self.last_gated.any():
            self._last_certified = (self.last_lam_pi, self.last_res, self.last_gated, self._flip, self._last_u)
        self._lower_p_eff(p_new)

        self.n_updates += 1
        self.estimated = True
        self._log_spectrum(flip, released)
        return event

    def reflect(self, proposal: np.ndarray, iterate: np.ndarray) -> np.ndarray:
        r"""
        Reflects the residual on the tracked directions, returning :math:`x + (\mathbb{1} - 2 Q W)\,(S(x) - x)`
        with :math:`W = Q^{T}` while no dual rows are installed.

        The reflection is a preconditioner of the proposal, not a mixer: fed to linear mixing it reproduces the
        stabilized iteration exactly, and since it is an involution that shares its fixed points with the plain
        map, an accelerated mixing keeps its fixed points (and, for the orthogonal case, its least-squares norms).

        :param proposal: The raw proposal :math:`S(x)` on the core window.
        :param iterate: The current iterate :math:`x` on the same window.
        :return: The reflected proposal in the dtype of ``proposal``, or ``proposal`` itself while nothing is
            tracked.
        """
        if self.q is None:
            return proposal
        x = to_vec(iterate)
        residual = to_vec(proposal) - x
        residual = residual - 2.0 * (self.q @ (self._dual @ residual))
        return to_mat(x + residual, proposal.shape).astype(proposal.dtype, copy=False)

    @property
    def _dual(self) -> np.ndarray:
        """The dual rows of the tracked directions: the installed ones, or the transpose of the orthonormal basis."""
        return self.q.T if self.w is None else self.w

    def predicted_rate(self) -> float:
        r"""
        Returns the linear convergence rate the last certified spectrum predicts for the damped Picard iteration
        under the current reflection, :math:`\rho = \max_\alpha |\lambda_{M,\alpha}|` with
        :math:`\lambda_M = 1 - p_{\mathrm{eff}}\lambda_\Pi`. While a basis is installed (:attr:`active`), a mode
        whose Ritz vector the reflection acts on (:math:`\lVert W u \rVert` above ``_FLIP_OVERLAP``) contributes
        :math:`|2 - \lambda_{M,\alpha}|` instead; with nothing installed every mode contributes
        :math:`|\lambda_{M,\alpha}|`. A Pulay or Anderson step runs at :attr:`p_flip` while a basis is installed,
        so this rate does not describe that step.

        :return: The predicted rate over the captured modes, or ``0.0`` while there is no estimate.
        """
        if self.last_lam_pi is None:
            return 0.0
        lam_pi = self.last_lam_pi[self.last_gated]
        if lam_pi.size == 0:
            return 0.0
        lam_m = 1.0 - self.p_eff * lam_pi
        if self.active:
            flipped = np.linalg.norm(self._dual @ self._last_u[:, self.last_gated], axis=0) > _FLIP_OVERLAP
            return float(np.max(np.where(flipped, np.abs(2.0 - lam_m), np.abs(lam_m))))
        return float(np.max(np.abs(lam_m)))

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
        self, theta: np.ndarray, u: np.ndarray, flip: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        r"""
        Builds the reflector of the flipped directions on the certified Ritz set. With :math:`U` the real columns of
        the flipped Ritz vectors followed by those of the certified stable ones (one member per conjugate pair; its
        real part and, for a complex Ritz value, its imaginary part), and :math:`D = -1` on the flipped columns and
        :math:`+1` elsewhere,

        .. math:: R = U D U^{+} + (\mathbb{1} - U U^{+})

        flips exactly the flipped directions and leaves every other column of :math:`U` fixed, whatever their mutual
        overlaps. It is applied as :math:`r - 2 Q (W r)` with :math:`U_f = Q R_f` the thin QR of the flipped columns
        and :math:`W = R_f (U^{+})_f` their dual rows, so that :math:`R^{2} = \mathbb{1}` and :math:`W u = 0` for
        every stable column kept. The stable columns are added one by one in the order of their overlap with the
        flipped span, and a column that would push the condition number of :math:`U` above ``_COND_CAP`` is left
        out (logged): the eigenbasis of a non-normal map is nearly defective, and the reflector fixes what it can
        separate. With no stable column kept, :math:`W = Q^{T}` and the reflection is orthogonal.

        :param theta: The measured Ritz values.
        :param u: The Ritz vectors as columns.
        :param flip: Which modes are flipped.
        :return: The orthonormal basis :math:`Q` (``[n_real, k]``) and the dual rows :math:`W` (``[k, n_real]``),
            or ``None`` when no flipped column was collected (e.g. a lone conjugate-pair member with a negative
            imaginary part) or the flipped columns themselves are dependent beyond ``_COND_CAP``.
        """
        flipped_cols, stable_cols = [], []
        for idx in np.flatnonzero(self.last_gated):
            imag = theta[idx].imag
            if imag < -_REAL_CUTOFF * (1.0 + abs(theta[idx])):
                continue  # the conjugate partner spans the same real plane
            block = [u[:, idx].real] + ([u[:, idx].imag] if imag > _REAL_CUTOFF * (1.0 + abs(theta[idx])) else [])
            (flipped_cols if flip[idx] else stable_cols).extend(block)
        if not flipped_cols:
            return None
        certified = np.column_stack(flipped_cols).astype(np.float64)
        if not _well_conditioned(certified):
            return None
        n_flip = certified.shape[1]
        q, r_flip = np.linalg.qr(certified)
        skipped = 0
        if stable_cols:
            stable = np.column_stack(stable_cols).astype(np.float64)
            for j in np.argsort(-np.linalg.norm(q.T @ stable, axis=0)):
                trial = np.column_stack([certified, stable[:, j]])
                if _well_conditioned(trial):
                    certified = trial
                else:
                    skipped += 1
        if skipped:
            self._log_info(
                f"Jacobian tracker: {skipped} of {len(stable_cols)} certified stable directions are not separable "
                f"from the flipped ones (condition number above {_COND_CAP:g}) and are left out of the reflector."
            )
        w = r_flip @ np.linalg.pinv(certified)[:n_flip]
        return np.ascontiguousarray(q), np.ascontiguousarray(w)

    def _principal_angle(self, q_new: np.ndarray) -> float:
        """
        Returns the largest principal angle between the tracked subspace and a newly assembled one of equal
        dimension, i.e. the arc cosine of the smallest singular value of their overlap.

        :param q_new: The newly assembled orthonormal basis.
        :return: The largest principal angle in radians.
        """
        overlap = np.linalg.svd(self.q.T @ q_new, compute_uv=False)
        return float(np.arccos(np.clip(overlap.min(), -1.0, 1.0)))

    def _flip_bound(self, lam_pi: np.ndarray, flip: np.ndarray, res: np.ndarray) -> float:
        r"""
        Returns the damping bound of the flipped directions alone, the formula :func:`classify` applies to the whole
        certified spectrum evaluated on the flipped subset, or the configured damping when nothing is flipped.

        :param lam_pi: The measured eigenvalues :math:`\lambda_\Pi` the reflector was built from.
        :param flip: Which of them are flipped.
        :param res: Their Ritz residuals.
        :return: The bound of the flipped directions, never above the configured damping.
        """
        if not flip.any():
            return self.p_config
        return classify(lam_pi[flip], np.ones(int(flip.sum()), dtype=bool), self.p_config, res[flip])[1]

    def _lower_p_eff(self, p_new: float) -> None:
        """
        Lowers the effective damping to ``p_new`` when that is smaller than the current value, and never raises it.

        :param p_new: The candidate damping.
        :return: None.
        """
        if p_new < self.p_eff - 1e-12:
            self.p_eff = p_new
            self._warn_at_floor()

    def _warn_at_floor(self) -> None:
        """Warns once when the measured spectrum has pushed the effective damping down to its floor."""
        if self._logger is not None and self.p_eff <= _MIXING_FLOOR and not self._floor_warned:
            self._floor_warned = True
            self._logger.warning(
                f"Jacobian tracker: the effective mixing is at the floor ({_MIXING_FLOOR:g}). The measured "
                f"spectrum cannot be damped into the unit disk from here, i.e. the branch is not reachable "
                f"from this starting point. Continue from a closer converged state instead."
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
        count, the effective damping, the flip bound and the predicted convergence rate. An uncertified mode carries
        only its "uncertified" tag, since the residual gate failed and there is nothing else to report about it; a
        certified mode carries "certified" plus exactly one of "flip", "stable" or "marginal", the last one for a real
        part inside its undecidable band (see :func:`classify`). When the update released an installed basis
        with a message (the release forced while flips are paused, the residual-growth release, or the calm
        release of a carried basis), that message is folded into the same line.

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
                    decision = "flip"
                else:
                    decision = "marginal" if abs(lam.real) <= max(_MARGIN, self.last_res[idx]) else "stable"
                tag = f"certified, {decision}"
            else:
                tag = "uncertified"
            modes.append(f"lambda_Pi={lam.real:+.4f}{lam.imag:+.4f}j (res {self.last_res[idx]:.1e}, {tag})")
        prefix = f"{released}; " if released else ""
        self._logger.info(
            f"Jacobian tracker: {prefix}{', '.join(modes)}; {int(np.count_nonzero(flip))} flipped, "
            f"p_eff={self.p_eff:.4f}, p_flip={self.p_flip:.4f}, rho={self.predicted_rate():.4f}."
        )

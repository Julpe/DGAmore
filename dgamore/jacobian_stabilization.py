# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Stabilization of the physical fixed point of the self-energy self-consistency via the modified iterative scheme of
Essl, Reitner, Kozik and Toschi, arXiv:2502.01420. This module is deliberately self-contained (numpy only, no MPI or
config imports): it linearizes an arbitrary proposal map :math:`S(\Sigma)` matrix-free by finite differences, runs an
adaptive-length Arnoldi factorization of the damped-map Jacobian to find the unstable directions, and exposes the
orthogonal residual reflection that realizes the modified scheme. The DGAmore-specific wiring (building the proposal
map from the non-local SDE pipeline and applying the reflection inside the self-consistency loop) lives in
:mod:`dgamore.nonlocal_sde`.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Callable

import numpy as np

# When a reflection-uncurable instability forces the mixing down, back off below the exact stability bound p_max by
# this factor (round down to two decimals) so the binding mode lands inside the unit disk (e.g. 0.146 -> 0.12).
_MIXING_SAFETY = 0.85
_MIXING_FLOOR = 0.01  # never reduce the mixing parameter below this
_MIN_ARNOLDI_STEPS = 4  # never trust (and exit on) a Ritz spectrum from fewer steps


class PhysicalSolutionStabilizerError(RuntimeError):
    """Raised when the modified iterative scheme cannot be applied meaningfully - currently when
    the base self-energy is too far from a fixed point for the linearization to be trustworthy."""


class PhysicalSolutionStabilizer:
    r"""
    Stabilize the physical fixed point of the DGA self-energy self-consistency using the
    modified iterative scheme of Essl, Reitner, Kozik and Toschi, arXiv:2502.01420.

    The damped self-energy iteration :math:`\Sigma_{n+1} = p\,S(\Sigma_n) + (1-p)\,\Sigma_n`
    converges to an unphysical solution once the physical fixed point becomes unstable, i.e.
    once the damped map has a Jacobian eigenvalue of modulus :math:`\geq 1`. Since
    :math:`\Sigma`-mixing is the affine conjugate of the Green's-function iteration (scheme II
    of the paper), the stabilization is the analog of :math:`\mathrm{Eq.~(9)}`: the proposal
    map :math:`S` is linearized at the (assumed) physical solution and the damping sign is
    flipped on the unstable subspace via the projector :math:`\mathcal{P}`.

    With :math:`s_\alpha` the eigenvalues of the Jacobian
    :math:`J = \mathrm{d}S/\mathrm{d}\Sigma`, the damped map has eigenvalues
    :math:`(1-p) + p\,s_\alpha`; a direction is unstable when that modulus reaches one. The
    projector uses the sign rule of :math:`\mathrm{Eq.~(6)}` (:math:`+p` on stable,
    :math:`-p` on unstable directions) realized, as in :math:`\mathrm{SM~Sec.~VI}`, from the
    leading Jacobian eigenvectors. With an orthonormal real basis :math:`U` of the unstable
    subspace this is :math:`\mathcal{P} = p\,(\mathbb{1} - 2\,U U^{T})`, the Moore-Penrose
    form specialized to a uniform sign flip, reducing to :math:`p\,\mathbb{1}` where
    :math:`U` has no support.

    All work is done on the real representation :math:`[\mathrm{Re};\,\mathrm{Im}]` of the
    inner Matsubara window of the (replicated, full-BZ) core self-energy. The map :math:`S` is
    real-linear but not complex-analytic (conjugations, real :math:`\mu` solve), so :math:`J`
    is a real :math:`2N\times 2N` operator whose unstable eigenvalues generically come in
    complex-conjugate pairs out of particle-hole symmetry. Construction evaluates :math:`S`
    once for the base point and once per Arnoldi step; :meth:`reflect_proposal` is then
    cheap and triggers no further evaluations of :math:`S`.
    """

    def __init__(
        self,
        proposal_fn: Callable[[np.ndarray], np.ndarray],
        base_mat: np.ndarray,
        p: float,
        niv_jac: int,
        *,
        n_modes: int = 4,
        eps_rel: float | None = None,
        margin: float = 1e-2,
        max_residual: float | None = None,
        logger=None,
    ):
        r"""
        Build the projector by linearizing ``proposal_fn`` at ``base_mat``.

        ``base_mat`` is the assumed physical solution restricted to the inner Matsubara
        window (shape :math:`(n_k, n_b, n_b, 2\,n_{\nu,\mathrm{jac}})`, replicated on every
        rank). ``proposal_fn`` must map such an array to the raw proposal :math:`S` on the
        same window, re-solving :math:`\mu` and the occupation internally, and must be
        collective with a replicated result so an identical Arnoldi recurrence runs on all
        ranks.

        The finite-difference step ``eps_rel`` defaults to :math:`\sqrt{\epsilon_{\mathrm{mach}}}`
        of the *map's* working precision (detected from the proposal output dtype): for a
        single-precision (``complex64``) map this is :math:`\approx 3.5\times10^{-4}`, which
        minimizes the combined roundoff/truncation error of the directional derivative.
        Using a much smaller step (e.g. :math:`10^{-6}`) on a single-precision map makes the
        Jacobian-vector products roundoff-dominated and produces spurious unstable modes.

        The construction evaluates :math:`S` once for the base point plus once per Arnoldi
        step. The step count is adaptive: it stops early once the resolution criterion (every
        clearly-unstable Ritz value converged AND at least one converged stable Ritz value,
        the resolved stable tail) holds on two consecutive steps with an unchanged
        clearly-unstable count, it never runs fewer than 4 steps, and its budget is capped at
        :math:`m_{\mathrm{cap}} = \min(\max(2\,n_{\mathrm{modes}} + 2,\, 8),\, 2N-1)` steps.
        If the subspace is still unresolved at the cap, the factorization continues (each
        extra step costs exactly one more evaluation, nothing is recomputed) up to a hard cap
        of :math:`2\,m_{\mathrm{cap}}` steps before giving up with the saturation warning.
        """
        self.p = float(p)
        self.niv_jac = int(niv_jac)
        self._logger = logger
        self._shape = tuple(int(s) for s in base_mat.shape)
        self._n_complex = int(np.prod(self._shape))
        self.n_real = 2 * self._n_complex

        base_mat = np.ascontiguousarray(base_mat, dtype=np.complex128)
        self._base_vec = self._to_vec(base_mat)
        self._base_norm = float(np.linalg.norm(self._base_vec)) or 1.0

        f0_raw = proposal_fn(base_mat)
        if eps_rel is None:
            single = np.asarray(f0_raw).dtype in (np.complex64, np.float32)
            eps_rel = float(np.sqrt(np.finfo(np.float32 if single else np.float64).eps))
            if logger is not None:
                logger.info(
                    f"Finite-difference step eps_rel={eps_rel:.2e} "
                    f"({'single' if single else 'double'}-precision proposal map)."
                )
        self.eps_rel = float(eps_rel)

        f0_vec = self._to_vec(f0_raw)
        self.residual = float(np.linalg.norm(f0_vec - self._base_vec) / self._base_norm)
        if logger is not None:
            logger.info(f"Stabilizer base relative residual ||S(sigma)-sigma||: {self.residual:.3e}.")

        if max_residual is not None and self.residual > max_residual:
            raise PhysicalSolutionStabilizerError(
                f"Stabilizer base residual {self.residual:.3e} exceeds max_residual={max_residual:.3g}: "
                f"the starting self-energy is too far from a fixed point for the modified iterative "
                f"scheme to be meaningful. The Jacobian would be linearized at the wrong point, its "
                f"unstable subspace would be unrelated to the physical fixed point, and the iteration "
                f"would be steered toward an unphysical solution. Start from a warm self-energy near "
                f"the physical solution (a converged higher-T run interpolated down in temperature), "
                f"not from the DMFT/local self-energy. Raise max_residual only to override deliberately."
            )

        self.predicted_rate = 0.0
        self.slowest_flipped_rate = 0.0
        self.slowest_stable_rate = 0.0
        self.n_uncurable = 0
        self.saturated = False
        self.mixing_reduced = False
        self.p_initial = float(p)
        self.arnoldi_steps = 0
        self.n_evaluations = 1
        self.u_real = self._unstable_subspace(proposal_fn, base_mat, f0_vec, n_modes, self.eps_rel, margin)

    @property
    def n_unstable(self) -> int:
        r"""Dimension of the captured unstable subspace (``0`` if the scheme is already stable)."""
        return 0 if self.u_real is None else self.u_real.shape[1]

    def reflect_proposal(self, proposal_inner: np.ndarray, old_inner: np.ndarray) -> np.ndarray:
        r"""
        Reflect the unstable component of the residual on the inner window, returning
        :math:`\Sigma_n + (\mathbb{1} - 2\,U U^{T})\,(S(\Sigma_n) - \Sigma_n)`.

        This is a *preconditioner*, not a mixer: feeding the reflected proposal to the
        configured mixing (linear / Anderson / Pulay) reproduces the modified scheme
        :math:`\Sigma_{n+1} = \Sigma_n + \mathcal{P}\,(S - \Sigma_n)` for linear mixing
        (since the mixing factor equals :math:`p`), and accelerates the *modified*
        fixed-point map for Anderson/Pulay - so the history-based acceleration is preserved.
        The reflection :math:`\mathbb{1} - 2\,U U^{T}` is orthogonal, so it leaves the
        Anderson/Pulay least-squares residual norm unchanged. Returns the proposal unchanged
        where there is no unstable direction.

        :param proposal_inner: The raw proposal :math:`S(\Sigma_n)` on the inner window.
        :param old_inner: The previous iterate :math:`\Sigma_n` on the inner window.
        :return: The reflected proposal on the inner window (complex, same shape).
        """
        x_old = self._to_vec(old_inner)
        r = self._to_vec(proposal_inner) - x_old
        if self.u_real is not None:
            r = r - 2.0 * (self.u_real @ (self.u_real.T @ r))
        return self._to_mat(x_old + r)

    # ------------------------------------------------------------------ internals
    def _to_vec(self, mat: np.ndarray) -> np.ndarray:
        """Flattens a complex window tensor into the real [Re; Im] vector representation."""
        flat = mat.reshape(-1)
        return np.concatenate((flat.real, flat.imag)).astype(np.float64, copy=False)

    def _to_mat(self, vec: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`_to_vec`: rebuilds the complex window tensor from the real vector."""
        re = vec[: self._n_complex]
        im = vec[self._n_complex :]
        return (re + 1j * im).reshape(self._shape)

    def _subspace_resolved(self, h_mat: np.ndarray, n_steps: int, margin: float) -> tuple[bool, int]:
        r"""
        Early-exit criterion for the adaptive Arnoldi length, evaluated on the small Hessenberg
        matrix after ``n_steps`` steps: the subspace is resolved once every Ritz value with
        :math:`|\theta| > 1 + \mathrm{margin}` has passed the convergence gate AND at least one
        *stable* Ritz value has converged - the resolved stable tail proving that Arnoldi has
        reached past the unstable spectrum into the stable bulk. (Vacuously, a stable map
        resolves as soon as its dominant stable mode converges.) Also returns the number of
        clearly-unstable Ritz values, so the caller can demand the criterion to hold on two
        consecutive steps with an unchanged unstable count - a lone check could exit while a
        subdominant unstable mode is still hidden inside the unresolved bulk.
        """
        ritz_vals, small_vecs = np.linalg.eig(h_mat[:n_steps, :n_steps])
        beta = float(h_mat[n_steps, n_steps - 1])
        ritz_resid = abs(beta) * np.abs(small_vecs[n_steps - 1, :])
        converged = ritz_resid <= 3e-2 * np.maximum(1.0, np.abs(ritz_vals))
        clearly_unstable = np.abs(ritz_vals) > (1.0 + margin)
        resolved = bool(np.all(converged[clearly_unstable]) and np.any(converged & ~clearly_unstable))
        return resolved, int(np.count_nonzero(clearly_unstable))

    def _unstable_subspace(self, proposal_fn, base_mat, f0_vec, n_modes, eps_rel, margin):
        r"""
        Adaptive-length Arnoldi factorization of the damped-map Jacobian
        :math:`M = (1-p)\,\mathbb{1} + p\,J`, whose eigenvalues are the eigenvalues of the
        damped iteration. The dangerous directions are exactly those with
        :math:`|\mathrm{eig}(M)| \geq 1`, i.e. the dominant-magnitude eigenpairs, which an
        Arnoldi factorization resolves first. Returns an orthonormal real basis of the
        unstable subspace or ``None``. Performs one evaluation of :math:`S` per step; the
        step count adapts via :meth:`_subspace_resolved` between the 4-step minimum, the
        budget cap :math:`m_{\mathrm{cap}}` and the continuation hard cap
        :math:`2\,m_{\mathrm{cap}}` (see ``__init__``).

        The :math:`p`-independent proposal-Jacobian eigenvalues are cached, so if a
        reflection-uncurable instability is present the mixing :math:`p` is reduced to the
        largest contractive value (:meth:`_max_stable_mixing`) and the classification rebuilt
        at the reduced :math:`p` - with no further evaluations of :math:`S`. The reduced
        :math:`p` is exposed via :attr:`p` and the :attr:`mixing_reduced` flag.
        """
        n = self.n_real
        m_cap = int(min(max(2 * n_modes + 2, 8), n - 1))
        m_hard = int(min(2 * m_cap, n - 1))
        min_steps = min(_MIN_ARNOLDI_STEPS, m_cap)

        def matvec_damped(v: np.ndarray) -> np.ndarray:
            # M v = (1-p) v + p J v, with J v the finite-difference directional derivative.
            vnorm = float(np.linalg.norm(v))
            if vnorm == 0.0:  # pragma: no cover - defensive; Arnoldi only feeds unit vectors
                return np.zeros_like(v)
            h = eps_rel * self._base_norm / vnorm
            jv = (self._to_vec(proposal_fn(base_mat + self._to_mat(h * v))) - f0_vec) / h
            return (1.0 - self.p) * v + self.p * jv

        # Adaptive-length Arnoldi (modified Gram-Schmidt); one S evaluation per step. The (large)
        # basis Q is allocated for the budget cap and only grown if the continuation is entered.
        rng = np.random.default_rng(0)  # fixed seed -> identical recurrence on every MPI rank
        Q = np.zeros((n, m_cap + 1), dtype=np.float64)
        H = np.zeros((m_hard + 1, m_hard), dtype=np.float64)
        q = rng.standard_normal(n)
        Q[:, 0] = q / np.linalg.norm(q)
        m_eff = 0
        j = 0
        prev_state: tuple[bool, int] | None = None
        while j < m_hard:
            if j == Q.shape[1] - 1:
                Q = np.concatenate((Q, np.zeros((n, m_hard - j), dtype=np.float64)), axis=1)
                if self._logger is not None:
                    self._logger.warning(
                        f"Arnoldi subspace unresolved at the budget cap ({m_cap} steps): continuing "
                        f"the factorization up to {m_hard} steps (one proposal evaluation per step)."
                    )
            w = matvec_damped(Q[:, j])
            for i in range(j + 1):
                H[i, j] = Q[:, i] @ w
                w = w - H[i, j] * Q[:, i]
            hj = float(np.linalg.norm(w))
            H[j + 1, j] = hj
            j += 1
            m_eff = j
            if hj < 1e-12:  # invariant subspace reached
                break
            Q[:, j] = w / hj
            if j >= min_steps:
                state = self._subspace_resolved(H, j, margin)
                if state[0] and prev_state is not None and prev_state[0] and state[1] == prev_state[1]:
                    break
                prev_state = state
        self.arnoldi_steps = m_eff
        self.n_evaluations = 1 + m_eff

        ritz_vals, small_vecs = np.linalg.eig(H[:m_eff, :m_eff])
        ritz_vecs = Q[:, :m_eff] @ small_vecs  # eigenvectors of M (= of J), complex

        # Arnoldi Ritz-residual estimate ||M v - lambda v|| = |beta| * |last component of the small eigenvector| (beta
        # the next subdiagonal). Only trustworthy if small - the gate that keeps unresolved modes out of the signal.
        beta = float(H[m_eff, m_eff - 1]) if m_eff >= 1 else 0.0
        ritz_resid = abs(beta) * np.abs(small_vecs[m_eff - 1, :])
        converged = ritz_resid <= 3e-2 * np.maximum(1.0, np.abs(ritz_vals))

        # The proposal-Jacobian J is p-independent (finite differences compute J v; p enters only via
        # M = (1-p) I + p J). M and J share eigenvectors, so lambda_J = (lambda_M - (1-p)) / p re-derives everything.
        self._lam_J = (ritz_vals - (1.0 - self.p)) / self.p
        self._ritz_vecs = ritz_vecs
        self._converged = converged
        self._arnoldi_full = m_eff == m_hard

        res = self._classify(self.p, margin)

        # A reflection-uncurable instability (type-(i) overshoot lambda_J < 1, |lambda_M| > 1, or a positive mode past
        # the flippable window) is cured by a smaller mixing p: rebuild the classification at the largest curable p.
        if res.n_uncurable > 0:
            p_max = self._max_stable_mixing()
            if np.isfinite(p_max):
                p_new = max(math.floor(_MIXING_SAFETY * p_max * 100.0) / 100.0, _MIXING_FLOOR)
            else:  # pragma: no cover - defensive; an uncurable mode always binds
                p_new = self.p
            if p_new < self.p:
                if self._logger is not None:
                    self._logger.warning(
                        f"Detected {res.n_uncurable} reflection-uncurable instability(ies) at mixing "
                        f"p={self.p:.3f}: the conventional iteration diverges along them. Maximum "
                        f"stable mixing p_max={p_max:.4f}; reducing the mixing parameter "
                        f"p: {self.p:.3f} -> {p_new:.2f} (safety factor {_MIXING_SAFETY:g}, rounded "
                        f"down) so the modified scheme is contractive."
                    )
                self.p = p_new
                self.mixing_reduced = True
                res = self._classify(self.p, margin)

        self.saturated = res.saturated
        self.n_uncurable = res.n_uncurable
        self.predicted_rate = res.predicted_rate
        self.slowest_flipped_rate = res.slowest_flipped_rate
        self.slowest_stable_rate = res.slowest_stable_rate

        if self._logger is not None:
            self._log_spectrum(res, m_eff)

        # The cached spectrum only serves the build-time classification and mixing re-derivation; the Ritz-vector
        # matrix (n_real x m_eff complex128, multiple GB/rank at scale) must not stay resident for the rest of the run.
        self._ritz_vecs = None
        self._lam_J = None
        self._converged = None

        return res.u_real

    def _classify(self, p: float, margin: float) -> SimpleNamespace:
        r"""
        Classify the cached Ritz spectrum at mixing ``p`` and build the projector - without any
        proposal evaluation. The damped-map eigenvalues are
        :math:`\lambda_M = (1-p) + p\,\lambda_J` (``M`` and ``J`` share eigenvectors), so the
        flip decision and the orthonormal unstable basis ``U`` follow from the cached
        ``self._lam_J`` and ``self._ritz_vecs`` at any ``p``.
        """
        lam_M = (1.0 - p) + p * self._lam_J
        conv = self._converged
        clearly_unstable = np.abs(lam_M) > (1.0 + margin)
        flip_stabilizes = np.abs(2.0 - lam_M) < 1.0
        unstable = clearly_unstable & flip_stabilizes & conv

        post = np.where(unstable, np.abs(2.0 - lam_M), np.abs(lam_M))
        flipped = post[unstable]
        other = post[~unstable]

        # Resolved = Arnoldi reached past the unstable spectrum from above: every clearly-unstable Ritz value converged
        # AND a converged stable value exists (a lone converged stable value is not enough - it can be an edge).
        resolved = bool(np.all(conv[clearly_unstable]) and np.any(conv & ~clearly_unstable))

        cols = []
        for idx in np.where(unstable)[0]:
            cols.append(self._ritz_vecs[:, idx].real)
            if abs(lam_M[idx].imag) > 1e-8 * (1 + abs(lam_M[idx])):
                cols.append(self._ritz_vecs[:, idx].imag)
        if cols:
            u, sv, _ = np.linalg.svd(np.column_stack(cols).astype(np.float64), full_matrices=False)
            keep = sv > 1e-10 * (sv[0] if sv.size else 1.0)
            u_real = np.ascontiguousarray(u[:, keep])
        else:
            u_real = None

        return SimpleNamespace(
            lam_M=lam_M,
            clearly_unstable=clearly_unstable,
            unstable=unstable,
            n_uncurable=int(np.count_nonzero(clearly_unstable & ~flip_stabilizes & conv)),
            saturated=bool(self._arnoldi_full and not resolved),
            predicted_rate=float(post.max()) if post.size else 0.0,
            slowest_flipped_rate=float(flipped.max()) if flipped.size else 0.0,
            slowest_stable_rate=float(other.max()) if other.size else 0.0,
            u_real=u_real,
        )

    def _max_stable_mixing(self) -> float:
        r"""
        Largest mixing :math:`p` for which every converged captured mode is curable - stable, or
        reflection-flippable into the unit disk. For a proposal-Jacobian eigenvalue
        :math:`\lambda` the damped eigenvalue is :math:`1 + p(\lambda-1)`; this is stable
        (:math:`\mathrm{Re}\,\lambda < 1`) or flips inside (:math:`\mathrm{Re}\,\lambda > 1`) iff

        .. math:: p < \frac{2\,|\mathrm{Re}\,\lambda - 1|}{|\lambda - 1|^{2}},

        which for real :math:`\lambda` is :math:`2/|\lambda-1|`. The binding bound is the minimum
        over converged modes; near-unit modes (:math:`|\lambda-1|\approx 0`, the slow stable
        modes) do not bind and are excluded. Returns ``inf`` if nothing binds.
        """
        lam = self._lam_J[self._converged]
        d = lam - 1.0
        denom = np.abs(d) ** 2
        binding = denom > 1e-12
        if not np.any(binding):  # pragma: no cover - defensive; an uncurable mode always binds
            return float("inf")
        pmax = 2.0 * np.abs(d.real[binding]) / denom[binding]
        return float(np.min(pmax))

    def _log_spectrum(self, res: SimpleNamespace, m_eff: int) -> None:
        """Log the captured spectrum, saturation, and the predicted-rate diagnostic at the final p."""
        lam_M = res.lam_M
        order = np.argsort(-np.abs(lam_M))
        for idx in order:
            lam = lam_M[idx]
            if res.unstable[idx]:
                tag = "unstable -> flip"
            elif res.clearly_unstable[idx]:
                tag = "unstable, flip ineffective (|2-lambda|>=1) -> skip; reduce p / warm-start"
            else:
                tag = "stable"
            self._logger.info(f"  |damped eig|={abs(lam):.4f} ({lam.real:+.4f}{lam.imag:+.4f}j)  {tag}")
        self._logger.info(f"Arnoldi factorization used {m_eff} step(s) ({self.n_evaluations} proposal evaluations).")
        if res.saturated:
            self._logger.warning(
                f"The Arnoldi budget is exhausted ({m_eff} steps) without resolving the unstable "
                f"spectrum: the unstable subspace is likely under-resolved - more unstable "
                f"directions may lie beyond the Arnoldi window. This regime has many competing "
                f"instabilities; continue from a closer warm start (higher-T continuation) instead. "
                f"A fully resolved spectrum has every unstable Ritz value converged plus a converged "
                f"stable tail."
            )
        self._logger.info(
            f"Predicted linear rate over captured modes rho={res.predicted_rate:.4f}/iter "
            f"(slowest flipped |2-lambda|={res.slowest_flipped_rate:.4f}, slowest other "
            f"captured |lambda|={res.slowest_stable_rate:.4f}). Un-accelerated and "
            f"captured-spectrum only: Anderson/Pulay are faster, modes outside the Arnoldi "
            f"window may be slower; not an iteration count."
        )
        if res.n_uncurable > 0:
            at_floor = (
                f" The mixing is at the floor ({_MIXING_FLOOR:g}) and cannot be lowered further to " f"damp them."
                if self.p <= _MIXING_FLOOR
                else ""
            )
            self._logger.warning(
                f"{res.n_uncurable} captured mode(s) unstable but not reflection-curable "
                f"(|lambda|>1, |2-lambda|>=1) even at p={self.p:.2f}.{at_floor} The modified "
                f"iteration is not contractive along them; the warm start is too far from the "
                f"physical branch (continue from higher T)."
            )
        elif res.predicted_rate >= 1.0:
            self._logger.warning(
                f"Captured linear rate rho={res.predicted_rate:.4f} >= 1: linear convergence is "
                f"not guaranteed; rely on Anderson/Pulay or reduce p."
            )

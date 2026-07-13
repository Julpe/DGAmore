# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Unit tests for dgamore.jacobian_stabilization.PhysicalSolutionStabilizer.

These exercise the modified iterative scheme (arXiv:2502.01420) in isolation, on synthetic affine maps with a
prescribed Jacobian, so every quantity is known analytically: the finite-difference Jacobian and damped-map Arnoldi
recover the known spectrum, the finite-difference step is auto-selected from the map precision (the fp32 fix), the
flip criterion |lambda| > 1 + margin flips genuinely unstable modes only, the projector reflection is orthogonal and
reduces to identity when stable (the Anderson/Pulay-preserving composition), the stabilization converges where
conventional damping diverges (type-i and type-iii), and the adaptive Arnoldi length exits early on resolved spectra,
continues past the budget cap on unresolved ones and stays bounded by the hard cap.
"""

import math

import numpy as np
import pytest

from dgamore.jacobian_stabilization import _MIXING_FLOOR, _MIXING_SAFETY
from dgamore.jacobian_stabilization import PhysicalSolutionStabilizer as Stab
from dgamore.jacobian_stabilization import PhysicalSolutionStabilizerError

FP32_EPS = float(np.sqrt(np.finfo(np.float32).eps))
FP64_EPS = float(np.sqrt(np.finfo(np.float64).eps))


def _vec(m):
    """Flattens a complex tensor into the real [Re; Im] vector used by the stabilizer."""
    f = m.reshape(-1)
    return np.concatenate((f.real, f.imag))


def _mat_factory(shape):
    """Returns the inverse of _vec for the given tensor shape."""
    nc = int(np.prod(shape))
    return lambda v: (v[:nc] + 1j * v[nc:]).reshape(shape)


def affine_map(shape, jac_real, b_real, *, dtype=np.complex128, counter=None):
    """Real-linear map S(x) = to_mat(jac_real @ to_vec(x) + b_real) with known real Jacobian and fixed point."""
    to_mat = _mat_factory(shape)

    def s_fn(mat):
        if counter is not None:
            counter["n"] += 1
        out = to_mat(jac_real @ _vec(mat) + b_real)
        return out.astype(dtype, copy=False)

    xstar = np.linalg.solve(np.eye(jac_real.shape[0]) - jac_real, b_real)
    return s_fn, to_mat(xstar), xstar


def jacobian_from_damped(n, p, damped_eigs, rng, non_normal=False):
    """
    Builds a real n x n Jacobian whose damped map M = (1-p) 1 + p J has the requested (real or
    complex-conjugate-pair) eigenvalues; the remaining modes are placed safely stable, and an optional
    strictly-upper coupling makes M non-normal without changing the eigenvalues.
    """
    blocks = []
    used = 0
    for lam in damped_eigs:
        s = (lam - (1.0 - p)) / p
        if abs(np.imag(s)) < 1e-12:
            blocks.append(np.array([[float(np.real(s))]]))
            used += 1
        else:
            a, b = float(np.real(s)), float(np.imag(s))
            blocks.append(np.array([[a, -b], [b, a]]))
            used += 2
    d_mat = np.zeros((n, n))
    i = 0
    for blk in blocks:
        k = blk.shape[0]
        d_mat[i : i + k, i : i + k] = blk
        i += k
    for j, s in enumerate(rng.uniform(-0.5, 0.5, n - i)):
        d_mat[i + j, i + j] = s
    if non_normal:
        d_mat = d_mat + np.triu(rng.standard_normal((n, n)), k=1) * 0.3
    q_mat, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q_mat @ d_mat @ q_mat.T


def conventional_diverges(s_fn, xstar, to_mat, p, n_iter=200, kick=1e-3, kdir=None):
    """Runs the plain damped iteration from a kicked start and reports whether it left the fixed point."""
    n = xstar.size
    v = kdir if kdir is not None else np.eye(n)[:, 0]
    x = to_mat(xstar + kick * v)
    for _ in range(n_iter):
        x = to_mat((1 - p) * _vec(x) + p * _vec(s_fn(x)))
    err = np.linalg.norm(_vec(x) - xstar)
    return (not np.isfinite(err)) or err > 1.0


def modified_run(stab, s_fn, xstar, to_mat, p, n_iter=400, kick=1e-3):
    """Runs the modified scheme (reflected residual + linear mixing) and returns the final error."""
    n = xstar.size
    x = to_mat(xstar + kick * np.eye(n)[:, 0])
    for _ in range(n_iter):
        refl = stab.reflect_proposal(s_fn(x), x)
        x = to_mat(p * _vec(refl) + (1 - p) * _vec(x))
    return np.linalg.norm(_vec(x) - xstar)


def _affine_with_eigs(damped_eigs, p, seed, nc=20, dtype=np.complex128, non_normal=False):
    """Builds an affine map on the (1, 1, 1, nc) shape with the requested damped eigenvalues."""
    shape = (1, 1, 1, nc)
    n = 2 * nc
    rng = np.random.default_rng(seed)
    jac = jacobian_from_damped(n, p, damped_eigs, rng, non_normal=non_normal)
    b = rng.standard_normal(n)
    return affine_map(shape, jac, b, dtype=dtype), shape, n


def _anderson(g, x0, depth=3, n_iter=60):
    """Minimal Anderson type-II acceleration of x -> g(x); returns the final iterate."""
    xs, gs = [x0], [g(x0)]
    x = gs[0]
    for _ in range(n_iter):
        gx = g(x)
        f = gx - x
        xs.append(x)
        gs.append(gx)
        m = min(depth, len(xs) - 1)
        if m >= 1:
            df = np.column_stack([(gs[-i] - xs[-i]) - (gs[-i - 1] - xs[-i - 1]) for i in range(1, m + 1)])
            dg = np.column_stack([gs[-i] - gs[-i - 1] for i in range(1, m + 1)])
            try:
                gamma, *_ = np.linalg.lstsq(df, f, rcond=None)
                x = gx - dg @ gamma
            except np.linalg.LinAlgError:
                x = gx
        else:
            x = gx
    return x


class RecordingLogger:
    """Minimal logger stub that records info and warning messages for assertions."""

    def __init__(self):
        self.infos, self.warnings = [], []

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)


@pytest.mark.parametrize("shape", [(1, 1, 1, 6), (4, 2, 2, 10), (16, 1, 1, 8)])
def test_vectorizer_roundtrip(shape):
    """The [Re; Im] vectorizer round-trips arbitrary complex tensors of the window shape."""
    rng = np.random.default_rng(0)
    m = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    n = 2 * int(np.prod(shape))
    s_fn, base, _ = affine_map(shape, np.zeros((n, n)), np.zeros(n))
    st = Stab(s_fn, base, p=0.5, niv_jac=shape[-1] // 2, n_modes=2)
    assert np.allclose(st._to_mat(st._to_vec(m)), m)
    assert st._to_vec(m).size == n
    assert st._to_vec(m).dtype == np.float64


def test_eps_rel_autodetect_single_precision():
    """A complex64 proposal map auto-selects the fp32 finite-difference step sqrt(eps_mach)."""
    s_fn, base, _ = affine_map((1, 1, 1, 10), np.zeros((20, 20)), np.zeros(20), dtype=np.complex64)
    st = Stab(s_fn, base, p=0.5, niv_jac=5, n_modes=2)
    assert st.eps_rel == pytest.approx(FP32_EPS, rel=1e-6)


def test_eps_rel_autodetect_double_precision():
    """A complex128 proposal map auto-selects the fp64 finite-difference step."""
    s_fn, base, _ = affine_map((1, 1, 1, 10), np.zeros((20, 20)), np.zeros(20), dtype=np.complex128)
    st = Stab(s_fn, base, p=0.5, niv_jac=5, n_modes=2)
    assert st.eps_rel == pytest.approx(FP64_EPS, rel=1e-6)


def test_eps_rel_explicit_override():
    """An explicitly passed eps_rel wins over the dtype auto-detection."""
    s_fn, base, _ = affine_map((1, 1, 1, 10), np.zeros((20, 20)), np.zeros(20), dtype=np.complex64)
    st = Stab(s_fn, base, p=0.5, niv_jac=5, n_modes=2, eps_rel=1e-5)
    assert st.eps_rel == 1e-5


def test_base_residual_zero_at_fixed_point():
    """Building exactly at the fixed point reports a (numerically) zero base residual."""
    (s_fn, base, _), _, _ = _affine_with_eigs([0.5], 0.3, 1, nc=12)
    st = Stab(s_fn, base, p=0.3, niv_jac=6, n_modes=4)
    assert st.residual < 1e-6


def test_base_residual_nonzero_off_fixed_point():
    """Building off the fixed point reports a finite base residual without crashing."""
    (s_fn, base, xstar), shape, n = _affine_with_eigs([0.5], 0.3, 2, nc=12)
    off = _mat_factory(shape)(xstar + 0.05 * np.linalg.norm(xstar) * np.eye(n)[:, 0])
    st = Stab(s_fn, off, p=0.3, niv_jac=6, n_modes=4)
    assert st.residual > 1e-3


def test_aborts_when_base_residual_exceeds_max():
    """Cold-start guard: the build refuses to linearize far from any fixed point."""
    (s_fn, base, xstar), shape, n = _affine_with_eigs([1.3], 0.3, 201)
    far = _mat_factory(shape)(xstar + 5.0 * np.random.default_rng(201).standard_normal(n))
    with pytest.raises(PhysicalSolutionStabilizerError):
        Stab(s_fn, far, p=0.3, niv_jac=10, n_modes=6, max_residual=0.5)


def test_no_abort_at_fixed_point_with_max_residual():
    """The guard passes silently when the base residual is below the threshold."""
    (s_fn, base, _), _, _ = _affine_with_eigs([1.3], 0.3, 202)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6, max_residual=0.5)
    assert st.residual < 0.5 and st.n_unstable == 1


def test_finds_single_real_instability():
    """One unstable real damped eigenvalue yields a one-dimensional unstable subspace."""
    (s_fn, base, _), _, _ = _affine_with_eigs([1.30], 0.3, 3)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert st.n_unstable == 1


def test_finds_complex_conjugate_pair():
    """One unstable complex-conjugate pair yields a two-dimensional real unstable subspace."""
    (s_fn, base, _), _, _ = _affine_with_eigs([1.2 + 0.3j], 0.3, 4)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert st.n_unstable == 2


def test_captured_subspace_matches_true_eigenvectors():
    """The captured span contains the true unstable eigenvector of the damped map."""
    (s_fn, base, _), _, n = _affine_with_eigs([1.4], 0.3, 5)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    jac = np.column_stack(
        [_vec(s_fn(st._to_mat(np.eye(n)[:, i]))) - _vec(s_fn(st._to_mat(np.zeros(n)))) for i in range(n)]
    )
    w, v = np.linalg.eig((1 - 0.3) * np.eye(n) + 0.3 * jac)
    vstar = np.real(v[:, np.argmax(np.abs(w))])
    vstar /= np.linalg.norm(vstar)
    proj = st.u_real @ (st.u_real.T @ vstar)
    assert np.linalg.norm(proj - vstar) < 1e-3


def test_stable_map_yields_empty_projector():
    """A fully stable map produces no unstable directions and no projector."""
    (s_fn, base, _), _, _ = _affine_with_eigs([0.6], 0.5, 6, nc=15)
    st = Stab(s_fn, base, p=0.5, niv_jac=7, n_modes=6)
    assert st.n_unstable == 0
    assert st.u_real is None


def test_near_boundary_stable_mode_not_flipped():
    """Regression: a stable mode just inside the unit circle must not be flagged unstable."""
    (s_fn, base, _), _, _ = _affine_with_eigs([0.995], 0.3, 7)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6, margin=1e-2)
    assert st.n_unstable == 0


def test_clearly_unstable_mode_is_flipped():
    """A mode beyond 1 + margin is selected for the flip."""
    (s_fn, base, _), _, _ = _affine_with_eigs([1.05], 0.3, 8)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6, margin=1e-2)
    assert st.n_unstable == 1


@pytest.mark.parametrize("damped_eig", [3.5, -2.0, -0.5 + 2.66j, 8.0])
def test_uncurable_instability_triggers_mixing_reduction(damped_eig):
    """
    A mode the flip cannot stabilize at the build p (|2-lambda| >= 1) forces an automatic reduction of the
    mixing to the largest contractive value, which cures it (it becomes either stable or, once moved into the
    flippable window, reflection-curable).
    """
    (s_fn, base, _), _, _ = _affine_with_eigs([damped_eig], 0.3, 108)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6, margin=1e-2)
    assert st.mixing_reduced is True
    assert _MIXING_FLOOR <= st.p < 0.3
    assert st.n_uncurable == 0
    assert st.predicted_rate < 1.0


def test_predicted_rate_matches_flipped_mode():
    """rho equals the slowest captured post-flip rate; for a lone near-+1 mode that is |2-lambda|."""
    (s_fn, base, _), _, _ = _affine_with_eigs([1.05], 0.3, 109)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert st.n_unstable == 1 and st.n_uncurable == 0
    assert st.slowest_flipped_rate == pytest.approx(0.95, abs=2e-2)
    assert st.predicted_rate == pytest.approx(0.95, abs=2e-2)


def test_margin_excludes_marginal_modes():
    """The margin decides whether a barely-unstable mode is acted on."""
    (s_fn, base, _), _, _ = _affine_with_eigs([1.005], 0.3, 9)
    st_big = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6, margin=5e-2)
    st_small = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6, margin=1e-3)
    assert st_big.n_unstable == 0
    assert st_small.n_unstable == 1


def test_reflection_is_orthogonal():
    """(1 - 2 U U^T) is a reflection: it preserves the residual norm (Anderson/Pulay-safe)."""
    (s_fn, base, _), shape, n = _affine_with_eigs([1.3, 1.1 + 0.2j], 0.3, 10)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=8)
    to_mat = _mat_factory(shape)
    rng = np.random.default_rng(10)
    prop = to_mat(rng.standard_normal(n))
    old = to_mat(rng.standard_normal(n))
    refl = st.reflect_proposal(prop, old)
    r_in = _vec(prop) - _vec(old)
    r_out = _vec(refl) - _vec(old)
    assert np.linalg.norm(r_out) == pytest.approx(np.linalg.norm(r_in), rel=1e-10)


def test_reflection_is_identity_when_stable():
    """With no unstable direction the reflection returns the proposal unchanged."""
    (s_fn, base, _), shape, n = _affine_with_eigs([0.6], 0.5, 11, nc=15)
    st = Stab(s_fn, base, p=0.5, niv_jac=7, n_modes=6)
    to_mat = _mat_factory(shape)
    rng = np.random.default_rng(11)
    prop = to_mat(rng.standard_normal(n))
    old = to_mat(rng.standard_normal(n))
    assert np.allclose(st.reflect_proposal(prop, old), prop)


def test_composition_equals_direct_modified_update():
    """reflect_proposal followed by linear mixing (alpha = p) equals x + P (S(x)-x) with P = p (1 - 2 U U^T)."""
    (s_fn, base, _), shape, n = _affine_with_eigs([1.4], 0.3, 12)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    to_mat = _mat_factory(shape)
    rng = np.random.default_rng(12)
    x = to_mat(rng.standard_normal(n))
    r = _vec(s_fn(x)) - _vec(x)
    u = st.u_real
    direct = _vec(x) + 0.3 * (r - 2.0 * (u @ (u.T @ r)))
    composed = 0.3 * _vec(st.reflect_proposal(s_fn(x), x)) + 0.7 * _vec(x)
    assert np.allclose(direct, composed, atol=1e-10)


def test_type_i_real_instability_stabilized():
    """Two real instabilities near +1: conventional damping diverges, the modified scheme converges."""
    (s_fn, base, xstar), shape, _ = _affine_with_eigs([1.40, 1.25], 0.2, 13)
    to_mat = _mat_factory(shape)
    assert conventional_diverges(s_fn, xstar, to_mat, p=0.2)
    st = Stab(s_fn, base, p=0.2, niv_jac=10, n_modes=6)
    assert st.n_unstable == 2
    assert modified_run(st, s_fn, xstar, to_mat, p=0.2, n_iter=300) < 1e-6


def test_type_iii_complex_pair_stabilized():
    """A type-(iii) complex pair outside the circle is flipped inside and the iteration converges."""
    (s_fn, base, xstar), shape, _ = _affine_with_eigs([1.18 + 0.30j], 0.3, 14)
    to_mat = _mat_factory(shape)
    assert conventional_diverges(s_fn, xstar, to_mat, p=0.3)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert st.n_unstable == 2
    assert modified_run(st, s_fn, xstar, to_mat, p=0.3, n_iter=500) < 1e-6


def test_non_normal_jacobian_stabilized():
    """Block-triangular argument: the orthogonal projector stabilizes even when M is non-normal."""
    (s_fn, base, xstar), shape, _ = _affine_with_eigs([1.5], 0.25, 15, non_normal=True)
    to_mat = _mat_factory(shape)
    assert conventional_diverges(s_fn, xstar, to_mat, p=0.25)
    st = Stab(s_fn, base, p=0.25, niv_jac=10, n_modes=8)
    assert st.n_unstable >= 1
    assert modified_run(st, s_fn, xstar, to_mat, p=0.25, n_iter=400) < 1e-6


def test_fp32_stable_map_no_spurious_modes():
    """With the auto fp32 step, a single-precision stable map must not invent unstable modes."""
    (s_fn, base, _), _, _ = _affine_with_eigs([0.6], 0.3, 16, dtype=np.complex64)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert st.eps_rel == pytest.approx(FP32_EPS, rel=1e-6)
    assert st.n_unstable == 0


def test_fp32_clear_instability_detected():
    """A clear instability is detected through the fp32 finite-difference noise."""
    (s_fn, base, _), _, _ = _affine_with_eigs([1.4], 0.3, 17, dtype=np.complex64)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert st.n_unstable == 1


@pytest.mark.parametrize("n_modes,nc", [(2, 20), (4, 20), (6, 20), (8, 30)])
def test_solve_count_bounded_and_consistent(n_modes, nc):
    """The evaluation count is 1 + arnoldi_steps, at least the 4-step floor and at most 1 + the hard cap."""
    shape = (1, 1, 1, nc)
    n = 2 * nc
    rng = np.random.default_rng(18)
    jac = jacobian_from_damped(n, 0.3, [1.3], rng)
    counter = {"n": 0}
    s_fn, base, _ = affine_map(shape, jac, rng.standard_normal(n), counter=counter)
    st = Stab(s_fn, base, p=0.3, niv_jac=nc // 2, n_modes=n_modes)
    m_cap = min(max(2 * n_modes + 2, 8), n - 1)
    assert counter["n"] == st.n_evaluations == 1 + st.arnoldi_steps
    assert 1 + 4 <= counter["n"] <= 1 + min(2 * m_cap, n - 1)


def test_early_exit_stops_before_cap():
    """A single, well-separated instability resolves early: far fewer evaluations than the budget cap."""
    shape = (1, 1, 1, 20)
    n = 40
    rng = np.random.default_rng(180)
    jac = jacobian_from_damped(n, 0.3, [1.3], rng)
    counter = {"n": 0}
    s_fn, base, _ = affine_map(shape, jac, rng.standard_normal(n), counter=counter)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=8)
    assert st.n_unstable == 1
    assert 5 <= st.arnoldi_steps < 18
    assert counter["n"] == 1 + st.arnoldi_steps


def test_early_exit_respects_minimum_steps():
    """Even a trivially stable map runs at least the minimum number of Arnoldi steps."""
    (s_fn, base, _), _, _ = _affine_with_eigs([0.5], 0.3, 181)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=8)
    assert st.arnoldi_steps >= 5
    assert st.n_unstable == 0


def test_continuation_past_cap_resolves_unresolved_subspace():
    """
    Eight well-separated unstable modes cannot fit into the n_modes=1 budget cap of 8 steps together with a
    stable tail, so the factorization continues past the cap (with a warning) and resolves all of them.
    """
    shape = (1, 1, 1, 30)
    n = 60
    rng = np.random.default_rng(182)
    many = [1.1 + 0.25 * k for k in range(8)]
    jac = jacobian_from_damped(n, 0.3, many, rng)
    counter = {"n": 0}
    s_fn, base, _ = affine_map(shape, jac, rng.standard_normal(n), counter=counter)
    log = RecordingLogger()
    st = Stab(s_fn, base, p=0.3, niv_jac=15, n_modes=1, logger=log)
    assert st.arnoldi_steps > 8
    assert st.saturated is False
    assert st.n_unstable == 8
    assert any("continuing" in w for w in log.warnings)
    assert counter["n"] == 1 + st.arnoldi_steps


@pytest.mark.parametrize("shape", [(16, 2, 2, 10), (32, 4, 4, 8), (8, 1, 1, 12)])
def test_multi_k_multi_orbital_shapes(shape):
    """The stabilizer accepts arbitrary (k, o1, o2, v) window shapes and returns matching reflections."""
    n = 2 * int(np.prod(shape))
    rng = np.random.default_rng(19)
    jac = 0.1 * rng.standard_normal((n, n))
    s_fn, base, _ = affine_map(shape, jac, rng.standard_normal(n))
    st = Stab(s_fn, base, p=0.5, niv_jac=shape[-1] // 2, n_modes=4)
    assert st.residual < 1e-6
    assert st.n_real == n
    out = st.reflect_proposal(_mat_factory(shape)(np.zeros(n)), _mat_factory(shape)(np.zeros(n)))
    assert out.shape == shape


def test_modified_scheme_composes_with_anderson():
    """The reflected (modified) map is Anderson-acceleratable and converges to the physical solution."""
    (s_fn, base, xstar), shape, n = _affine_with_eigs([1.6], 0.3, 20)
    to_mat = _mat_factory(shape)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert st.n_unstable == 1

    def g_mod(xv):
        return _vec(st.reflect_proposal(s_fn(to_mat(xv)), to_mat(xv)))

    x_final = _anderson(g_mod, xstar + 1e-3 * np.eye(n)[:, 0], depth=3, n_iter=80)
    assert np.linalg.norm(x_final - xstar) < 1e-6


def test_logger_emits_eigs_residual_and_rate():
    """The build logs the base residual, the eps auto-detection, the spectrum tags and the rate."""
    (s_fn, base, _), _, _ = _affine_with_eigs([1.05], 0.3, 40)
    log = RecordingLogger()
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6, logger=log)
    text = "\n".join(log.infos)
    assert "Stabilizer base relative residual" in text
    assert "Finite-difference step eps_rel" in text
    assert "damped eig" in text and "unstable -> flip" in text
    assert "Predicted linear rate over captured modes" in text
    assert st.n_unstable == 1 and not log.warnings


def test_logger_warns_on_mixing_reduction():
    """The automatic mixing reduction is announced with the computed p_max."""
    (s_fn, base, _), _, _ = _affine_with_eigs([3.5], 0.3, 41)
    log = RecordingLogger()
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6, logger=log)
    assert st.mixing_reduced is True and st.n_uncurable == 0
    assert any("reducing the mixing parameter" in w for w in log.warnings)
    assert any("Maximum stable mixing p_max" in w for w in log.warnings)


def test_logger_warns_on_marginal_rate_ge_one():
    """A marginal mode (inside the margin band) is not flipped but flags rho >= 1."""
    (s_fn, base, _), _, _ = _affine_with_eigs([1.005], 0.3, 42)
    log = RecordingLogger()
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6, margin=1e-2, logger=log)
    assert st.n_unstable == 0 and st.n_uncurable == 0
    assert st.predicted_rate >= 1.0
    assert any(">= 1" in w for w in log.warnings)


def test_to_mat_inverse_of_to_vec_both_directions():
    """_to_mat and _to_vec are mutual inverses in both directions."""
    (s_fn, base, _), shape, n = _affine_with_eigs([0.5], 0.3, 1)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=4)
    rng = np.random.default_rng(0)
    m = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    v = rng.standard_normal(n)
    assert np.allclose(st._to_mat(st._to_vec(m)), m)
    assert np.allclose(st._to_vec(st._to_mat(v)), v)


def test_zero_base_does_not_divide_by_zero():
    """An all-zero base self-energy floors the base norm instead of dividing by zero."""
    shape, n = (1, 1, 1, 10), 20
    rng = np.random.default_rng(2)
    jac = jacobian_from_damped(n, 0.3, [0.6], rng)
    s_fn, _, _ = affine_map(shape, jac, rng.standard_normal(n))
    st = Stab(s_fn, _mat_factory(shape)(np.zeros(n)), p=0.3, niv_jac=5, n_modes=4)
    assert np.isfinite(st.residual)


def test_determinism_same_inputs_same_output():
    """Two builds on identical inputs give identical projectors (fixed Arnoldi seed)."""
    (s_fn, base, _), _, _ = _affine_with_eigs([1.08 + 0.05j], 0.3, 7)
    a = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    b = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert a.n_unstable == b.n_unstable
    assert a.predicted_rate == pytest.approx(b.predicted_rate, rel=1e-12)
    assert np.allclose(a.u_real, b.u_real)


def test_u_columns_orthonormal():
    """The unstable basis U has orthonormal columns of the captured dimension."""
    (s_fn, base, _), _, n = _affine_with_eigs([1.3, 1.1 + 0.2j], 0.3, 8)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=8)
    u = st.u_real
    assert u is not None and u.shape[0] == n and u.shape[1] == st.n_unstable
    assert np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-9)


def test_double_reflection_recovers_residual():
    """R = 1 - 2 U U^T is an involution: reflecting the residual twice recovers the proposal."""
    (s_fn, base, _), shape, n = _affine_with_eigs([1.3], 0.3, 9)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    to_mat = _mat_factory(shape)
    rng = np.random.default_rng(3)
    prop = to_mat(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    old = to_mat(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    twice = st.reflect_proposal(st.reflect_proposal(prop, old), old)
    assert np.allclose(twice, prop, atol=1e-10)


def test_reflection_leaves_orthogonal_complement_untouched():
    """A residual orthogonal to the unstable subspace passes through the reflection unchanged."""
    (s_fn, base, _), shape, n = _affine_with_eigs([1.3], 0.3, 10)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    u = st.u_real
    rng = np.random.default_rng(4)
    r = rng.standard_normal(n)
    r -= u @ (u.T @ r)
    to_mat, old = _mat_factory(shape), _mat_factory(shape)(rng.standard_normal(n))
    prop = to_mat(_vec(old) + r)
    assert np.allclose(st.reflect_proposal(prop, old), prop, atol=1e-10)


def test_eps_rel_from_float32_real_output():
    """A map returning complex64 output is treated as single precision for the step selection."""
    shape, n = (1, 1, 1, 10), 20
    rng = np.random.default_rng(5)
    jac = jacobian_from_damped(n, 0.3, [0.5], rng)
    to_mat = _mat_factory(shape)

    def s_real32(mat):
        return to_mat(jac @ _vec(mat)).astype(np.complex64)

    st = Stab(s_real32, to_mat(np.zeros(n)), p=0.3, niv_jac=5, n_modes=4)
    assert st.eps_rel == pytest.approx(FP32_EPS, rel=1e-6)


def test_max_residual_none_never_aborts():
    """Without a max_residual guard the build proceeds even far off any fixed point."""
    (s_fn, base, xstar), shape, n = _affine_with_eigs([1.3], 0.3, 11)
    far = _mat_factory(shape)(xstar + 5.0 * np.random.default_rng(6).standard_normal(n))
    st = Stab(s_fn, far, p=0.3, niv_jac=10, n_modes=6, max_residual=None)
    assert st.residual > 0.5


def test_predicted_rate_stable_only():
    """A fully stable spectrum reports rho < 1 and no flipped rate."""
    (s_fn, base, _), _, _ = _affine_with_eigs([0.7], 0.3, 12)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert st.n_unstable == 0 and st.n_uncurable == 0
    assert st.predicted_rate < 1.0 and st.slowest_flipped_rate == 0.0


def test_mixing_reduced_to_expected_value_real_overshoot():
    """
    Reproduces the production case: a real overshoot at damped eigenvalue -3.1179 (p=0.3) implies
    lambda_J = -12.73 and p_max = 2/13.73 = 0.1457, so p -> floor(0.85 * 0.1457, 2) = 0.12 and the mode
    becomes stable (no longer uncurable).
    """
    (s_fn, base, _), _, _ = _affine_with_eigs([-3.1179], 0.3, 13)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert st.mixing_reduced is True
    assert st.p == pytest.approx(0.12, abs=1e-9)
    assert st.n_uncurable == 0 and st.predicted_rate < 1.0


def test_mixing_reduction_makes_positive_mode_flippable():
    """
    A far-positive overshoot (damped 3.5 -> lambda_J = 9.33, p_max = 2/8.33 = 0.24) drops to
    p = floor(0.85 * 0.24, 2) = 0.20, where the mode lands in the flippable window and is reflected.
    """
    (s_fn, base, _), _, _ = _affine_with_eigs([3.5], 0.3, 13)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert st.mixing_reduced is True
    assert st.p == pytest.approx(0.20, abs=1e-9)
    assert st.n_unstable == 1 and st.n_uncurable == 0


def test_no_mixing_reduction_when_all_curable():
    """A purely type-(iii)-like flippable instability needs no damping change."""
    (s_fn, base, _), _, _ = _affine_with_eigs([1.05], 0.3, 77)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert st.mixing_reduced is False
    assert st.p == pytest.approx(0.3) and st.n_unstable == 1 and st.n_uncurable == 0


def test_no_mixing_reduction_when_fully_stable():
    """No instability at all: no projector, no reduction, p untouched."""
    (s_fn, base, _), _, _ = _affine_with_eigs([0.7], 0.3, 78)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert st.mixing_reduced is False and st.p == pytest.approx(0.3)
    assert st.n_unstable == 0 and st.n_uncurable == 0


@pytest.mark.parametrize("damped,p0", [(-3.1179, 0.3), (3.5, 0.3), (-2.05, 0.3), (-0.5 + 2.66j, 0.3), (8.0, 0.25)])
def test_reduced_mixing_matches_pmax_formula(damped, p0):
    """
    The reduced p equals floor(safety * p_max, 2) with p_max = 2 |Re(lambda_J) - 1| / |lambda_J - 1|^2 from
    the recovered (p-independent) proposal-Jacobian eigenvalue lambda_J = (lambda_M - (1-p)) / p.
    """
    (s_fn, base, _), _, _ = _affine_with_eigs([damped], p0, 211)
    st = Stab(s_fn, base, p=p0, niv_jac=10, n_modes=6)
    lam_j = (damped - (1 - p0)) / p0
    p_max = 2 * abs(lam_j.real - 1) / abs(lam_j - 1) ** 2
    expected = max(math.floor(_MIXING_SAFETY * p_max * 100) / 100, _MIXING_FLOOR)
    assert st.mixing_reduced is True
    assert st.p == pytest.approx(expected, abs=1e-9)
    assert st.n_uncurable == 0


def test_reflection_orthonormal_after_mixing_reduction():
    """After the build lowers p and rebuilds U, the projector is still an orthonormal basis."""
    (s_fn, base, _), _, _ = _affine_with_eigs([3.5], 0.3, 99)
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6)
    assert st.mixing_reduced and st.u_real is not None
    assert np.allclose(st.u_real.T @ st.u_real, np.eye(st.u_real.shape[1]), atol=1e-9)


def test_mixing_floor_clamps_and_warns_when_unfixable():
    """
    A mode so stiff that even the mixing floor cannot stabilize it: p is clamped to the floor, mixing_reduced
    is set, and the residual uncurable instability is still reported.
    """
    (s_fn, base, _), _, _ = _affine_with_eigs([-89.3], 0.3, 111)
    log = RecordingLogger()
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6, logger=log)
    assert st.mixing_reduced is True
    assert st.p == pytest.approx(_MIXING_FLOOR, abs=1e-9)
    assert st.n_uncurable >= 1
    assert any("not reflection-curable" in w and "even at p=" in w for w in log.warnings)


def test_mixing_already_at_floor_does_not_reduce_further():
    """
    If the build mixing is already at the floor and a mode is still uncurable, the stabilizer must not lower
    p further (no reduction, no crash, no loop) and must say so explicitly.
    """
    (s_fn, base, _), _, _ = _affine_with_eigs([-2.0], _MIXING_FLOOR, 112)
    log = RecordingLogger()
    st = Stab(s_fn, base, p=_MIXING_FLOOR, niv_jac=10, n_modes=6, logger=log)
    assert st.mixing_reduced is False
    assert st.p == pytest.approx(_MIXING_FLOOR)
    assert st.n_uncurable >= 1
    assert not any("reducing the mixing parameter" in w for w in log.warnings)
    assert any("floor" in w for w in log.warnings)


def test_mixing_below_floor_is_not_raised():
    """A user mixing below the floor with an uncurable mode is left untouched, never raised to the floor."""
    (s_fn, base, _), _, _ = _affine_with_eigs([-2.0], 0.005, 113)
    st = Stab(s_fn, base, p=0.005, niv_jac=10, n_modes=6)
    assert st.mixing_reduced is False and st.p == pytest.approx(0.005)
    assert st.n_uncurable >= 1


def test_stored_scalar_attributes():
    """The build stores the mixing, window size and real dimension it was given."""
    (s_fn, base, _), _, _ = _affine_with_eigs([1.2], 0.25, 14, nc=16)
    st = Stab(s_fn, base, p=0.25, niv_jac=8, n_modes=4)
    assert st.p == 0.25 and st.niv_jac == 8 and st.n_real == 2 * 16


@pytest.mark.parametrize("n_orb", [2, 3, 4])
def test_unstable_subspace_carries_orbital_structure(n_orb):
    """The unstable eigenvector, reshaped to the window tensor, has support on orbital off-diagonals."""
    nk, niv_jac = 4, 4
    shape = (nk, n_orb, n_orb, 2 * niv_jac)
    n = 2 * int(np.prod(shape))
    rng = np.random.default_rng(20 + n_orb)
    jac = jacobian_from_damped(n, 0.3, [1.1 + 0.1j], rng)
    s_fn, base, _ = affine_map(shape, jac, rng.standard_normal(n))
    st = Stab(s_fn, base, p=0.3, niv_jac=niv_jac, n_modes=6)
    assert st.n_unstable == 2
    mat = st._to_mat(st.u_real[:, 0])
    assert np.linalg.norm(mat[:, 0, 1, :]) > 0.0


@pytest.mark.parametrize("shape", [(2, 2, 2, 6), (4, 3, 3, 8)])
def test_reflect_proposal_shape_and_dtype_multiorbital(shape):
    """reflect_proposal preserves the multi-orbital window shape and returns a complex tensor."""
    n = 2 * int(np.prod(shape))
    rng = np.random.default_rng(30)
    jac = jacobian_from_damped(n, 0.3, [1.2], rng)
    s_fn, base, _ = affine_map(shape, jac, rng.standard_normal(n))
    st = Stab(s_fn, base, p=0.3, niv_jac=shape[-1] // 2, n_modes=6)
    to_mat = _mat_factory(shape)
    out = st.reflect_proposal(to_mat(rng.standard_normal(n)), to_mat(rng.standard_normal(n)))
    assert out.shape == shape and np.iscomplexobj(out)


def _run_modified_loop(s_fn, x0, to_mat, p, stab, *, anderson=False, depth=3, n_iter=400):
    """End-to-end modified iteration: proposal, residual reflection, then linear or Anderson mixing."""

    def g_mod(xv):
        return _vec(stab.reflect_proposal(s_fn(to_mat(xv)), to_mat(xv)))

    if anderson:
        return _anderson(g_mod, x0, depth=depth, n_iter=n_iter)
    x = x0
    for _ in range(n_iter):
        x = p * g_mod(x) + (1 - p) * x
    return x


def test_full_stabilization_run_multiorbital_linear():
    """
    A multi-orbital sigma on a tiny grid: conventional iteration diverges, the modified scheme converges to
    the physical fixed point and recovers the full orbital structure.
    """
    nk, n_orb, niv_jac = 4, 2, 4
    shape = (nk, n_orb, n_orb, 2 * niv_jac)
    n = 2 * int(np.prod(shape))
    p = 0.2
    rng = np.random.default_rng(2024)
    jac = jacobian_from_damped(n, p, [1.06 + 0.05j, 1.10], rng)
    s_fn, base, xstar = affine_map(shape, jac, rng.standard_normal(n))
    to_mat = _mat_factory(shape)

    assert conventional_diverges(s_fn, xstar, to_mat, p=p)
    stab = Stab(s_fn, base, p=p, niv_jac=niv_jac, n_modes=8, max_residual=0.5)
    assert stab.n_unstable == 3
    assert stab.n_uncurable == 0 and stab.predicted_rate < 1.0

    x_final = _run_modified_loop(s_fn, xstar + 1e-3 * np.eye(n)[:, 0], to_mat, p, stab, n_iter=600)
    assert np.linalg.norm(x_final - xstar) < 1e-6
    assert np.allclose(to_mat(x_final), to_mat(xstar), atol=1e-6)


def test_full_stabilization_run_multiorbital_anderson():
    """The same end-to-end run with Anderson acceleration on top of the reflected map."""
    nk, n_orb, niv_jac = 4, 2, 6
    shape = (nk, n_orb, n_orb, 2 * niv_jac)
    n = 2 * int(np.prod(shape))
    p = 0.2
    rng = np.random.default_rng(2025)
    jac = jacobian_from_damped(n, p, [1.08 + 0.04j], rng)
    s_fn, base, xstar = affine_map(shape, jac, rng.standard_normal(n))
    to_mat = _mat_factory(shape)

    assert conventional_diverges(s_fn, xstar, to_mat, p=p)
    stab = Stab(s_fn, base, p=p, niv_jac=niv_jac, n_modes=6, max_residual=0.5)
    assert stab.n_unstable == 2

    x_final = _run_modified_loop(
        s_fn, xstar + 1e-3 * np.eye(n)[:, 0], to_mat, p, stab, anderson=True, depth=4, n_iter=120
    )
    assert np.linalg.norm(x_final - xstar) < 1e-6


def test_full_run_built_off_fixed_point_still_converges():
    """
    A warm start a couple of percent off the fixed point (the realistic case): the build still resolves the
    unstable pair and the modified loop converges.
    """
    nk, n_orb, niv_jac = 4, 2, 4
    shape = (nk, n_orb, n_orb, 2 * niv_jac)
    n = 2 * int(np.prod(shape))
    p = 0.2
    rng = np.random.default_rng(2026)
    jac = jacobian_from_damped(n, p, [1.07 + 0.05j], rng)
    s_fn, base, xstar = affine_map(shape, jac, rng.standard_normal(n))
    to_mat = _mat_factory(shape)

    warm = to_mat(xstar + 0.02 * np.linalg.norm(xstar) / np.sqrt(n) * rng.standard_normal(n))
    stab = Stab(s_fn, warm, p=p, niv_jac=niv_jac, n_modes=6, max_residual=0.5)
    assert 0 < stab.residual < 0.5 and stab.n_unstable == 2
    x_final = _run_modified_loop(s_fn, xstar + 1e-3 * np.eye(n)[:, 0], to_mat, p, stab, n_iter=600)
    assert np.linalg.norm(x_final - xstar) < 1e-6


def test_saturation_flagged_when_unstable_exceed_subspace():
    """More unstable directions than even the continued Krylov budget: saturated, warn to raise n_modes."""
    shape, n = (1, 1, 1, 30), 60
    rng = np.random.default_rng(303)
    many = [1.1 + 0.1 * k for k in range(18)]
    jac = jacobian_from_damped(n, 0.3, many, rng)
    s_fn, base, _ = affine_map(shape, jac, rng.standard_normal(n))
    log = RecordingLogger()
    st = Stab(s_fn, base, p=0.3, niv_jac=15, n_modes=1, logger=log)
    assert st.saturated is True
    assert any("under-resolved" in w for w in log.warnings)


def test_no_saturation_when_stable_tail_present():
    """A fully resolved spectrum (converged stable Ritz values captured) is not flagged saturated."""
    (s_fn, base, _), _, _ = _affine_with_eigs([1.1 + 0.1j], 0.3, 304)
    log = RecordingLogger()
    st = Stab(s_fn, base, p=0.3, niv_jac=10, n_modes=6, logger=log)
    assert st.saturated is False
    assert not any("under-resolved" in w for w in log.warnings)


def test_increasing_n_modes_resolves_more_directions():
    """Raising n_modes turns an under-resolved build into a resolved one that flips every unstable mode."""
    shape, n = (1, 1, 1, 30), 60
    rng = np.random.default_rng(305)
    many = [1.1 + 0.2 * k for k in range(6)]
    jac = jacobian_from_damped(n, 0.3, many, rng)
    s_fn, base, _ = affine_map(shape, jac, rng.standard_normal(n))
    big = Stab(s_fn, base, p=0.3, niv_jac=15, n_modes=8)
    assert big.saturated is False
    assert big.n_unstable == 6

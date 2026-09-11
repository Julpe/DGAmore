# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Unit tests for dgamore.jacobian_stabilization.

Everything runs on small dense maps built here, so every quantity is known analytically or from np.linalg.eig: the
secant Rayleigh-Ritz recovers planted Jacobian eigenvalues from a damped iteration history and reproduces a dense
eigendecomposition once the history spans the whole space, it drops the increments that carry only the rounding
noise of a single-precision history instead of certifying the spectrum that noise produces, the classification
table implements the sign rule and the damping bound, the orthogonal reflection flips exactly the spectrum of the
tracked invariant subspace, and the tracker converges the two toy maps of arXiv:2606.04936 that plain damping sends
to the unphysical fixed point. The carry-over tests plant the spectrum state a run writes, round-trip it through
save_spectrum and load_spectrum, and drive carry_in and the release rules that hold, re-pend or drop a carried basis.
"""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest

from dgamore.jacobian_stabilization import (
    _FLIP_OVERLAP,
    _MARGIN,
    _MIXING_FLOOR,
    _MONITOR_WINDOW,
    _NOISE_FACTOR,
    _PERSIST,
    _PREEMPT_BAND,
    _RITZ_GATE,
    _SUBSPACE_ANGLE,
    JacobianTracker,
    SPECTRUM_FILE,
    TRACKER_PAIRS,
    classify,
    load_spectrum,
    secant_ritz,
    to_mat,
    to_vec,
)

MU = 1.0
A_2D = np.array([[2.0, 0.1], [0.1, -2.0]])
B_2D = np.array([1.0, 1.0])


def scalar_map(x: np.ndarray) -> np.ndarray:
    """Proposal map f(x) = x + mu x - x^2 of arXiv:2606.04936, with fixed points 0 and mu."""
    return x + MU * x - x * x


def two_d_map(x: np.ndarray) -> np.ndarray:
    """Proposal map f_x = 1 + 2x + 0.1y, f_y = 1 - 2y + 0.1x of arXiv:2606.04936."""
    return A_2D @ x + B_2D


def planted_jacobian(
    n: int,
    real_eigs: list[float],
    pair_eigs: list[complex],
    rng: np.random.Generator,
    coupling: float = 0.05,
    stable: float = 0.8,
) -> np.ndarray:
    """Builds a real non-normal n x n matrix with the planted eigenvalues and a bulk inside |lambda| < stable."""
    blocks = [np.array([[lam]]) for lam in real_eigs]
    blocks += [np.array([[lam.real, -lam.imag], [lam.imag, lam.real]]) for lam in pair_eigs]
    blocks += [np.array([[s]]) for s in rng.uniform(-stable, stable, n - sum(b.shape[0] for b in blocks))]
    mat, spans, start = np.zeros((n, n)), [], 0
    for block in blocks:
        size = block.shape[0]
        mat[start : start + size, start : start + size] = block
        spans.append((start, start + size))
        start += size
    upper = np.triu(rng.standard_normal((n, n)), k=1) * coupling
    for low, high in spans:
        upper[low:high, low:high] = 0.0
    orth, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return orth @ (mat + upper) @ orth.T


def damped_history(
    mat: np.ndarray, shift: np.ndarray, p: float, n_iter: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Runs x -> x + p (A x + b - x) from a small random start and returns the iterate and proposal column stacks."""
    x = rng.standard_normal(mat.shape[0]) * 1e-2
    iterates, proposals = [], []
    for _ in range(n_iter):
        proposal = mat @ x + shift
        iterates.append(x.copy())
        proposals.append(proposal.copy())
        x = x + p * (proposal - x)
    return np.column_stack(iterates), np.column_stack(proposals)


def single_precision_history(n_iter: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Runs a damped map with lambda_J = 1.3 and a fivefold stable rate 0.5, storing every pair in complex64."""
    rng = np.random.default_rng(0)
    unstable_dir = rng.standard_normal(6)
    unstable_dir /= np.linalg.norm(unstable_dir)
    mat = 0.5 * np.eye(6) + 0.8 * np.outer(unstable_dir, unstable_dir)
    x = rng.standard_normal(6)
    x = x - (unstable_dir @ x) * unstable_dir + 1e-6 * unstable_dir
    iterates, proposals = [], []
    for _ in range(n_iter):
        iterate = x.astype(np.complex64)
        rounded = iterate.real.astype(np.float64)
        proposal = (mat @ rounded).astype(np.complex64)
        iterates.append(iterate)
        proposals.append(proposal)
        x = rounded + 0.5 * (proposal.real.astype(np.float64) - rounded)
    return iterates, proposals


def duplicate_secant(xs: np.ndarray, fs: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns the history with the secant increment at ``index`` inserted a second time."""
    dx = np.insert(np.diff(xs, axis=1), index, np.diff(xs, axis=1)[:, index], axis=1)
    df = np.insert(np.diff(fs, axis=1), index, np.diff(fs, axis=1)[:, index], axis=1)
    xs_dup = np.column_stack([xs[:, 0], xs[:, :1] + np.cumsum(dx, axis=1)])
    fs_dup = np.column_stack([fs[:, 0], fs[:, :1] + np.cumsum(df, axis=1)])
    return xs_dup, fs_dup


def linear_pairs(
    mat: np.ndarray, shift: np.ndarray, p: float, n_iter: int, x0: list[float]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns the (iterate, proposal) lists of a damped linear iteration as complex core-window arrays."""
    x = np.asarray(x0, dtype=np.complex128)
    iterates, proposals = [], []
    for _ in range(n_iter):
        proposal = (mat @ x.real + shift).astype(np.complex128)
        iterates.append(x.copy())
        proposals.append(proposal.copy())
        x = x + p * (proposal - x)
    return iterates, proposals


def tracked_run(
    step_fn: Callable[[np.ndarray], np.ndarray],
    x0: list[float],
    p: float,
    n_iter: int,
    tracker: JacobianTracker | None,
    bound_only_with_basis: bool = False,
) -> np.ndarray:
    """Runs the tracker protocol proposal -> record pair -> reflect -> mix and returns the final iterate."""
    x = np.asarray(x0, dtype=np.complex128)
    iterates, proposals = [], []
    for _ in range(n_iter):
        proposal = step_fn(x)
        iterates.append(x.copy())
        proposals.append(proposal.copy())
        used = proposal if tracker is None else tracker.reflect(proposal, x)
        alpha = p if tracker is None or (bound_only_with_basis and not tracker.active) else tracker.p_eff
        x = x + alpha * (used - x)
        if tracker is not None:
            tracker.update(iterates, proposals)
    return x.real


def rotated_map(alpha: float) -> np.ndarray:
    """Returns a real 2x2 map with eigenvalues 2.0 and 0.5 whose unstable eigenvector is rotated by ``alpha``."""
    rot = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return rot @ np.diag([2.0, 0.5]) @ rot.T


def _carried_state(lam_pi, u, shape=(1, 1, 1, 1, 1, 2), res=None, converged=True, p_eff=0.4):
    """Builds a spectrum state from modes given as real-representation columns (all below the band get vectors)."""
    lam_pi = np.asarray(lam_pi, dtype=np.complex128)
    u = np.asarray(u, dtype=np.complex128)
    stored = lam_pi.real < 0.1
    cols = list(np.flatnonzero(stored))
    re_cols = [to_mat(u[:, j].real, shape).reshape(-1).astype(np.complex64) for j in cols]
    im_cols = [to_mat(u[:, j].imag, shape).reshape(-1).astype(np.complex64) for j in cols]
    n_complex = int(np.prod(shape))
    return {
        "lam_pi": lam_pi,
        "res": np.zeros(lam_pi.size) if res is None else np.asarray(res, dtype=np.float64),
        "flip": lam_pi.real < 0.0,
        "stored": stored,
        "u_re": np.column_stack(re_cols) if re_cols else np.zeros((n_complex, 0), dtype=np.complex64),
        "u_im": np.column_stack(im_cols) if im_cols else np.zeros((n_complex, 0), dtype=np.complex64),
        "shape": np.asarray(shape, dtype=np.int64),
        "beta": np.float64(10.0),
        "p_eff": np.float64(p_eff),
        "converged": np.bool_(converged),
    }


def _expand_identity(shape):
    """Expansion for equal grids: the stored column is the window itself, flattened back to the real vector."""
    return lambda column: to_vec(column.reshape(shape))


def _real_column(index: int, n: int) -> np.ndarray:
    """Unit column on the real part of coordinate ``index`` of an n-dimensional complex iterate ([Re; Im] layout)."""
    column = np.zeros((2 * n, 1))
    column[index, 0] = 1.0
    return column


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_to_vec_to_mat_round_trip(dtype):
    """to_vec and to_mat are inverse for both storage precisions, and to_mat returns complex128."""
    rng = np.random.default_rng(0)
    mat = (rng.standard_normal((3, 2, 4)) + 1j * rng.standard_normal((3, 2, 4))).astype(dtype)
    vec = to_vec(mat)
    back = to_mat(vec, mat.shape)
    assert vec.shape == (2 * mat.size,) and vec.dtype == np.float64
    assert back.dtype == np.complex128
    assert np.allclose(back, mat, atol=1e-6)


def test_secant_ritz_recovers_dominant_real_eigenvalue():
    """A planted real lambda_J = 1.3 is a certified Ritz value of the damped iteration's own secants."""
    rng = np.random.default_rng(0)
    mat = planted_jacobian(48, [1.3], [], rng)
    xs, fs = damped_history(mat, rng.standard_normal(48), 0.5, 14, np.random.default_rng(90))
    theta, _, res = secant_ritz(xs[:, -7:], fs[:, -7:])
    gated = res <= _RITZ_GATE * np.maximum(1.0, np.abs(theta))
    assert np.min(np.abs(theta[gated] - 1.3)) < 1e-3
    assert np.min(res[np.abs(theta - 1.3) < 1e-3]) < _RITZ_GATE


def test_secant_ritz_recovers_dominant_complex_pair():
    """A planted pair lambda_J = 1.05 +- 0.4j appears as two certified Ritz values."""
    rng = np.random.default_rng(0)
    mat = planted_jacobian(52, [], [1.05 + 0.4j], rng)
    xs, fs = damped_history(mat, rng.standard_normal(52), 0.5, 14, np.random.default_rng(90))
    theta, _, res = secant_ritz(xs[:, -7:], fs[:, -7:])
    gated = res <= _RITZ_GATE * np.maximum(1.0, np.abs(theta))
    assert np.min(np.abs(theta[gated] - (1.05 + 0.4j))) < 1e-3
    assert np.min(np.abs(theta[gated] - (1.05 - 0.4j))) < 1e-3


def test_secant_ritz_matches_dense_eig_on_tiny_map():
    """Seven pairs of a six-dimensional map span the whole space, so the Ritz values are the dense eigenvalues."""
    rng = np.random.default_rng(0)
    mat = planted_jacobian(6, [], [], rng, coupling=0.4, stable=1.4)
    xs, fs = damped_history(mat, rng.standard_normal(6), 1.0, 7, np.random.default_rng(100))
    theta, _, res = secant_ritz(xs, fs)
    assert theta.size == 6
    assert np.allclose(np.sort_complex(theta), np.sort_complex(np.linalg.eigvals(mat)), atol=1e-8)
    assert np.max(res) < 1e-8


def test_secant_ritz_survives_collinear_history():
    """A repeated secant increment is dropped, leaving the spectrum of the independent columns intact."""
    rng = np.random.default_rng(0)
    mat = planted_jacobian(20, [1.3], [], rng)
    xs, fs = damped_history(mat, rng.standard_normal(20), 0.5, 8, np.random.default_rng(90))
    theta, _, _ = secant_ritz(xs, fs)
    theta_dup, _, res_dup = secant_ritz(*duplicate_secant(xs, fs, 2))
    assert np.all(np.isfinite(theta_dup)) and np.all(np.isfinite(res_dup))
    assert theta_dup.size == theta.size
    assert np.allclose(np.sort_complex(theta_dup), np.sort_complex(theta), atol=1e-8)


def test_secant_ritz_drops_noise_columns_at_single_precision():
    """Without the floor a complex64 window certifies a garbage Ritz value; with it only the planted ones survive."""
    iterates, proposals = single_precision_history(52)
    xs = np.column_stack([to_vec(mat) for mat in iterates[-7:]])
    fs = np.column_stack([to_vec(mat) for mat in proposals[-7:]])
    floor = _NOISE_FACTOR * np.finfo(np.float32).eps * float(np.linalg.norm(xs[:, -1]))
    theta, _, res = secant_ritz(xs, fs)
    lam_pi = 1.0 - theta[res <= _RITZ_GATE * np.maximum(1.0, np.abs(theta))]
    theta_kept, _, res_kept = secant_ritz(xs, fs, floor)
    gated_kept = res_kept <= _RITZ_GATE * np.maximum(1.0, np.abs(theta_kept))
    assert np.max(np.abs(lam_pi)) > 3.0
    assert theta_kept.size == 2
    assert np.max(np.min(np.abs(theta_kept[gated_kept, None] - np.array([1.3, 0.5])), axis=1)) < 1e-2
    assert np.min(np.abs(theta_kept[gated_kept] - 1.3)) < 1e-2


@pytest.mark.parametrize(
    "lam_pi, gated, p_config, flip_expected, p_expected",
    [
        ([0.5], [True], 0.4, [False], 0.4),
        ([-0.3], [True], 4.0, [True], 0.5 * 2.0 * 0.3 / 0.09),
        ([-0.1 + 0.5j, -0.1 - 0.5j], [True, True], 4.0, [True, True], 0.5 * 2.0 * 0.1 / 0.26),
        ([0.005], [True], 0.4, [False], 0.4),
        ([0.003 + 0.5j, 0.003 - 0.5j], [True, True], 0.4, [False, False], 0.5 * 2.0 * _MARGIN / (0.003**2 + 0.25)),
        ([2.0j, -2.0j], [True, True], 0.4, [False, False], _MIXING_FLOOR),
        ([-0.3], [False], 0.4, [False], 0.4),
        ([-200.0], [True], 0.4, [True], _MIXING_FLOOR),
        ([-200.0], [True], 0.005, [True], 0.005),
    ],
)
def test_classify_table(lam_pi, gated, p_config, flip_expected, p_expected):
    """The sign rule flips clearly negative certified modes; marginal modes bind with the margin, never flip."""
    flip, p_eff = classify(np.array(lam_pi, dtype=np.complex128), np.array(gated), p_config)
    assert np.array_equal(flip, np.array(flip_expected))
    assert np.allclose(p_eff, p_expected, atol=1e-12)


def test_classify_widens_the_band_with_the_residual():
    """A residual larger than the margin widens the undecidable band: no flip inside it, and it enters the bound."""
    lam_pi, gated = np.array([-0.04 + 0.0j, -0.04 + 0.0j]), np.array([True, True])
    flip, p_eff = classify(lam_pi, gated, 0.4, res=np.array([0.001, 0.05]))
    assert np.array_equal(flip, np.array([True, False]))
    assert np.allclose(p_eff, 0.4, atol=1e-12)
    _, p_wide = classify(np.array([2.0j]), np.array([True]), 0.4, res=np.array([0.5]))
    assert np.allclose(p_wide, min(0.4, max(0.5 * 2.0 * 0.5 / 4.0, _MIXING_FLOOR)), atol=1e-12)


def test_stabilized_spectrum_is_exact_for_invariant_subspace():
    """Reflecting an invariant subspace maps its eigenvalues to 2 - lambda and leaves the rest untouched."""
    rng = np.random.default_rng(13)
    orth, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    block = rng.standard_normal((12, 12))
    block[2:, :2] = 0.0
    jac = orth @ block @ orth.T
    q = orth[:, :2]
    stabilized = np.eye(12) + (np.eye(12) - 2.0 * q @ q.T) @ (jac - np.eye(12))
    expected = np.concatenate([2.0 - np.linalg.eigvals(block[:2, :2]), np.linalg.eigvals(block[2:, 2:])])
    assert np.allclose(np.sort_complex(np.linalg.eigvals(stabilized)), np.sort_complex(expected), atol=1e-10)


def test_reflect_is_involution_and_identity_without_q():
    """Applying the reflection twice returns the proposal, and without a tracked subspace it is a no-op."""
    rng = np.random.default_rng(1)
    tracker = JacobianTracker(0.4)
    proposal = (rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))).astype(np.complex128)
    iterate = (rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))).astype(np.complex128)
    assert tracker.reflect(proposal, iterate) is proposal
    tracker.q = np.linalg.qr(rng.standard_normal((12, 2)))[0]
    reflected = tracker.reflect(proposal, iterate)
    assert not np.allclose(reflected, proposal, atol=1e-8)
    assert np.allclose(tracker.reflect(reflected, iterate), proposal, atol=1e-12)


def test_tracker_converges_paper_scalar_map():
    """The tracked scalar map settles on the physical fixed point 0 while plain damping runs into mu."""
    tracker = JacobianTracker(0.2)
    tracked = tracked_run(scalar_map, [0.3, 0.15], 0.2, 200, tracker)
    plain = tracked_run(scalar_map, [0.3, 0.15], 0.2, 200, None)
    assert tracker.active
    assert np.allclose(tracked, 0.0, atol=1e-8)
    assert np.allclose(plain, MU, atol=1e-8)


def test_tracker_converges_paper_2d_map():
    """The tracked two-dimensional map converges to the fixed point that plain damping diverges away from."""
    tracker = JacobianTracker(0.2)
    tracked = tracked_run(two_d_map, [0.0, 0.0], 0.2, 400, tracker)
    plain = tracked_run(two_d_map, [0.0, 0.0], 0.2, 200, None)
    assert np.allclose(tracked, np.linalg.solve(np.eye(2) - A_2D, B_2D), atol=1e-2)
    assert np.linalg.norm(plain) > 1e3
    assert tracker.q.shape[1] == 1 and tracker.p_eff <= 0.2
    assert tracker.predicted_rate() < 1.0


def test_tracker_persistence_delays_first_flip():
    """The first flip lands on the update where the certified unstable mode has been seen _PERSIST times."""
    tracker = JacobianTracker(0.2)
    iterates, proposals = linear_pairs(A_2D, B_2D, 0.2, 9, [0.0, 0.0])
    events = []
    for end in range(3, len(iterates) + 1):
        events.append(tracker.update(iterates[:end], proposals[:end]))
        if tracker.active:
            break
    assert tracker.n_updates == _PERSIST
    assert events == [False] * (_PERSIST - 1) + [True]


def test_tracker_counters_reset_on_interruption():
    """An interrupted persistence or calm streak restarts from zero instead of carrying over stale progress."""
    tracker = JacobianTracker(0.4)
    unstable, stable, shift = np.diag([5.0, 0.7]), np.diag([0.2, 0.45]), np.array([1.0, 1.0])
    x0 = iter(np.arange(0.1, 4.0, 0.1))
    burst = [unstable] * (_PERSIST - 1) + [stable] + [unstable] * (_PERSIST - 1)
    for mat in burst:
        tracker.update(*linear_pairs(mat, shift, 0.3, 3, [next(x0), next(x0)]))
    assert not tracker.active
    for mat in [unstable] * _PERSIST:
        tracker.update(*linear_pairs(mat, shift, 0.3, 3, [next(x0), next(x0)]))
    assert tracker.active
    for mat in [stable] * _PERSIST:
        tracker.update(*linear_pairs(mat, shift, 0.3, 3, [next(x0), next(x0)]))
    assert not tracker.active
    for mat in [unstable] * _PERSIST:
        tracker.update(*linear_pairs(mat, shift, 0.3, 3, [next(x0), next(x0)]))
    assert tracker.active
    tracker.update(*linear_pairs(stable, shift, 0.3, 3, [next(x0), next(x0)]))
    assert tracker.active


def test_tracker_step_floor_freezes_state():
    """A relative step below the single-precision roundoff floor leaves the tracked subspace and damping untouched."""
    tracker = JacobianTracker(0.4)
    tracker.q = np.eye(8)[:, :1]
    base = np.ones((2, 2), dtype=np.complex64)
    iterates = [(base * (1.0 + 1e-5 * k)).astype(np.complex64) for k in range(4)]
    proposals = [(base * (1.0 + 1e-5 * (k + 1))).astype(np.complex64) for k in range(4)]
    q_before = tracker.q
    assert tracker.update(iterates, proposals) is False
    assert tracker.q is q_before and np.allclose(tracker.p_eff, 0.4, atol=1e-12) and tracker.n_updates == 0


def test_tracker_zero_norm_iterate_freezes_state():
    """A last iterate of exactly zero norm is frozen like a step below the floor instead of reaching secant_ritz."""
    tracker = JacobianTracker(0.4)
    tracker.q = np.eye(4)[:, :1]
    iterates = [np.zeros((1, 2), dtype=np.complex128) for _ in range(4)]
    proposals = [np.full((1, 2), 0.1 * k, dtype=np.complex128) for k in range(4)]
    q_before = tracker.q
    assert tracker.update(iterates, proposals) is False
    assert tracker.q is q_before and np.allclose(tracker.p_eff, 0.4, atol=1e-12) and tracker.n_updates == 0


def test_tracker_rank_one_window_certifies_the_dominant_mode():
    """A window whose stable directions decayed into single-precision noise still certifies the growing mode."""
    tracker = JacobianTracker(0.4)
    iterates, proposals = single_precision_history(100)
    # by iteration 100 the fivefold stable rate 0.5 has decayed far below the noise floor, leaving only lambda_J = 1.3
    assert tracker.update(iterates[-7:], proposals[-7:]) is False
    assert tracker.n_updates == 1 and tracker.last_lam_pi.size == 1 and bool(tracker.last_gated[0])
    assert np.allclose(tracker.last_lam_pi, -0.3, atol=1e-2)


def test_secant_ritz_returns_empty_spectrum_when_every_increment_is_noise():
    """A noise floor above every increment leaves no Ritz pair, and the tracker treats that window as frozen."""
    iterates, proposals = single_precision_history(12)
    xs = np.column_stack([to_vec(x) for x in iterates])
    fs = np.column_stack([to_vec(f) for f in proposals])
    theta, u, res = secant_ritz(xs, fs, noise_floor=1e9)
    assert theta.size == 0 and u.shape == (xs.shape[0], 0) and res.size == 0


def test_tracker_unflips_a_stable_mode():
    """A tracked subspace of a contracting map is dropped after _PERSIST updates without a certified flip."""
    tracker = JacobianTracker(0.4)
    tracker.q = np.eye(4)[:, :1]
    iterates, proposals = linear_pairs(np.diag([0.2, -0.3]), np.array([1.0, 1.0]), 0.4, 8, [0.7, -0.4])
    events = [tracker.update(iterates[:end], proposals[:end]) for end in range(3, 3 + _PERSIST)]
    assert tracker.q is None
    assert events == [False] * (_PERSIST - 1) + [True]


def test_tracker_allow_flip_false_only_monitors():
    """With allow_flip disabled the spectrum is still measured but nothing is flipped and the damping is held."""
    tracker = JacobianTracker(0.2)
    iterates, proposals = linear_pairs(A_2D, B_2D, 0.2, 9, [0.0, 0.0])
    for end in range(3, len(iterates) + 1):
        tracker.update(iterates[:end], proposals[:end], allow_flip=False)
    assert tracker.last_lam_pi is not None and np.any(tracker.last_lam_pi.real < 0.0)
    assert tracker.q is None and np.allclose(tracker.p_eff, 0.2, atol=1e-12) and not tracker.active


def test_tracker_allow_flip_false_releases_installed_basis():
    """Suppressing flips releases a previously installed basis at once, and a further call finds nothing to do."""
    tracker = JacobianTracker(0.2)
    tracker.q = np.linalg.qr(np.random.default_rng(2).standard_normal((4, 1)))[0]
    iterates, proposals = linear_pairs(A_2D, B_2D, 0.2, 3, [0.0, 0.0])
    assert tracker.update(iterates, proposals, allow_flip=False) is True
    assert tracker.q is None
    assert tracker.update(iterates, proposals, allow_flip=False) is False


def test_tracker_p_eff_only_decreases():
    """A binding mode lowers the damping, and a later harmless spectrum never raises it again."""
    tracker = JacobianTracker(0.5)
    iterates, proposals = linear_pairs(np.diag([4.0, 0.3]), np.array([1.0, 1.0]), 0.2, 8, [0.1, 0.1])
    for end in range(3, len(iterates) + 1):
        tracker.update(iterates[:end], proposals[:end])
    lowered = tracker.p_eff
    iterates, proposals = linear_pairs(np.diag([0.5, 0.4]), np.array([1.0, 1.0]), 0.5, 8, [0.9, -0.6])
    for end in range(3, len(iterates) + 1):
        tracker.update(iterates[:end], proposals[:end])
    assert np.allclose(lowered, 0.5 * 2.0 * 3.0 / 9.0, atol=1e-12)
    assert np.allclose(tracker.p_eff, lowered, atol=1e-12)


def test_update_sets_the_flip_bound_from_the_flipped_modes_only():
    """The installed reflector binds p_flip with the flipped mode alone, while the stiff stable one binds p_eff."""
    tracker = JacobianTracker(0.4)
    iterates, proposals = linear_pairs(np.diag([5.0, -39.0]), np.zeros(2), 0.01, 9, [1.0, 1.0])
    for end in range(3, len(iterates) + 1):
        tracker.update(iterates[:end], proposals[:end])
    assert tracker.active
    assert np.isclose(tracker.p_flip, 0.25, atol=1e-6) and np.isclose(tracker.p_eff, 0.025, atol=1e-6)
    assert tracker.p_flip > tracker.p_eff


def test_flip_bound_is_refreshed_while_the_basis_is_held():
    """A held basis still refreshes p_flip when the flipped mode's own bound tightens."""
    tracker = JacobianTracker(0.4)
    iterates, proposals = linear_pairs(np.diag([5.0, -39.0]), np.zeros(2), 0.01, 9, [1.0, 1.0])
    for end in range(3, len(iterates) + 1):
        tracker.update(iterates[:end], proposals[:end])
    assert tracker.active and np.isclose(tracker.p_flip, 0.25, atol=1e-6)
    q_before = tracker.q
    grown_pairs = linear_pairs(np.diag([9.0, -39.0]), np.zeros(2), 0.01, 9, [1.0, 1.0])
    assert tracker.update(*grown_pairs) is False
    assert tracker.q is q_before
    assert np.isclose(tracker.p_flip, 0.125, atol=1e-6)


def test_flip_bound_is_kept_when_a_held_basis_certifies_no_flip():
    """A held basis keeps its p_flip when the tracked direction no longer certifies as flipped."""
    tracker = JacobianTracker(0.4)
    iterates, proposals = linear_pairs(np.diag([5.0, -39.0]), np.zeros(2), 0.01, 9, [1.0, 1.0])
    for end in range(3, len(iterates) + 1):
        tracker.update(iterates[:end], proposals[:end])
    q_before, p_flip_before = tracker.q, tracker.p_flip
    calm_pairs = linear_pairs(np.diag([0.995, -39.0]), np.zeros(2), 0.01, 9, [1.0, 1.0])
    assert tracker.update(*calm_pairs) is False
    assert not tracker._flip.any() and tracker.q is q_before
    assert np.isclose(tracker.p_flip, p_flip_before, atol=1e-6)


def test_flip_bound_is_refreshed_right_after_a_flip_free_update():
    """A flip certified again after one flip-free update refreshes p_flip before persistence is reached."""
    tracker = JacobianTracker(0.4)
    iterates, proposals = linear_pairs(np.diag([5.0, -39.0]), np.zeros(2), 0.01, 9, [1.0, 1.0])
    for end in range(3, len(iterates) + 1):
        tracker.update(iterates[:end], proposals[:end])
    q_before = tracker.q
    tracker.update(*linear_pairs(np.diag([0.995, -39.0]), np.zeros(2), 0.01, 9, [1.0, 1.0]))
    assert tracker._persist == 0 and tracker.q is q_before
    tracker.update(*linear_pairs(np.diag([9.0, -39.0]), np.zeros(2), 0.01, 9, [1.0, 1.0]))
    assert tracker._persist == 1 and tracker.q is q_before
    assert np.isclose(tracker.p_flip, 0.125, atol=1e-6)


def test_flip_bound_returns_to_the_configured_value_on_release():
    """Releasing the tracked basis puts the flip bound back at the configured damping."""
    tracker = JacobianTracker(0.4)
    tracker.q = np.eye(4)[:, :1]
    tracker.p_flip = 0.05
    iterates, proposals = linear_pairs(np.diag([0.2, -0.3]), np.array([1.0, 1.0]), 0.4, 8, [0.7, -0.4])
    for end in range(3, 3 + _PERSIST):
        tracker.update(iterates[:end], proposals[:end])
    assert tracker.q is None and tracker.p_flip == 0.4


def test_flip_bound_is_the_configured_value_without_a_reflection():
    """A spectrum of certified stable modes lowers p_eff and leaves the flip bound at the configured damping."""
    tracker = JacobianTracker(0.4)
    iterates, proposals = linear_pairs(np.diag([-3.0, 0.3]), np.zeros(2), 0.2, 7, [1.0, 1.0])
    for end in range(3, len(iterates) + 1):
        tracker.update(iterates[:end], proposals[:end])
    assert not tracker.active and np.isclose(tracker.p_eff, 0.25, atol=1e-12) and tracker.p_flip == 0.4


def test_flip_needs_the_bound_when_the_bound_is_applied_only_with_a_basis():
    """Damping the configured 0.4 until a basis exists and the bound from then on converges lambda_Pi = -6."""
    mat, shift = np.diag([7.0, 0.5]), np.array([1.0, 1.0])
    step = lambda x: (mat @ x.real + shift).astype(np.complex128)
    tracker = JacobianTracker(0.4)
    x_final = tracked_run(step, [0.01, 0.01], 0.4, 250, tracker, bound_only_with_basis=True)
    assert tracker.active and np.allclose(tracker.p_eff, 0.5 * 2.0 * 6.0 / 36.0, atol=1e-3)
    assert np.allclose(x_final, np.linalg.solve(np.eye(2) - mat, shift), atol=1e-6)
    x_plain = tracked_run(step, [0.01, 0.01], 0.4, 40, None)
    assert np.linalg.norm(x_plain) > 1e3


def test_tracker_releases_basis_when_the_residual_keeps_growing():
    """A basis is dropped once the raw residual has grown on _PERSIST consecutive updates while it was installed."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    tracker.q = np.eye(4)[:, :1]
    tracker._last_residual_norm = 0.0
    mat, shift = np.diag([0.6, 0.5]), np.array([1.0, 1.0])
    for k in range(_PERSIST):
        iterates, proposals = linear_pairs(mat, shift, 0.3, 3, [10.0 * (k + 1), 10.0 * (k + 1)])
        released = tracker.update(iterates, proposals)
    assert released and tracker.q is None
    assert "residual grew" in logger.info.call_args[0][0]


def _grow_with_a_certified_flip(tracker: JacobianTracker, n_updates: int) -> bool:
    """Feeds n_updates windows of a map with one flipped mode, each restarted further out so the residual grows."""
    mat, shift = np.diag([2.0, 0.5]), np.array([1.0, 1.0])
    for k in range(n_updates):
        event = tracker.update(*linear_pairs(mat, shift, 0.3, 3, [10.0 * (k + 1), 10.0 * (k + 1)]))
    return event


def test_growing_residual_releases_a_basis_while_the_flip_still_certifies():
    """A residual growing on _PERSIST consecutive updates releases the basis while a certified flip persists."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    tracker.q = np.eye(4)[:, :1]
    tracker._last_residual_norm = 0.0
    event = _grow_with_a_certified_flip(tracker, _PERSIST)
    assert tracker._flip.any() and event and tracker.q is None and tracker._persist == 0
    assert "residual grew" in logger.info.call_args[0][0]


def test_release_by_growth_needs_fresh_certification_to_reinstall():
    """The certifying update right after a growth release only restarts the persistence streak."""
    tracker = JacobianTracker(0.4)
    tracker.q = np.eye(4)[:, :1]
    tracker._last_residual_norm = 0.0
    _grow_with_a_certified_flip(tracker, _PERSIST + 1)
    assert tracker.q is None and tracker._flip.any() and tracker._persist == 1


def test_tracker_warns_once_at_floor():
    """The floor warning fires exactly once even though later updates keep the damping pinned at the floor."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    iterates, proposals = linear_pairs(np.diag([201.0, 0.5]), np.array([1.0, 1.0]), 0.1, 9, [0.05, 0.05])
    for end in range(3, len(iterates) + 1):
        tracker.update(iterates[:end], proposals[:end])
    assert logger.warning.call_count == 1
    assert "floor" in logger.warning.call_args[0][0]


def test_tracker_update_applies_the_noise_floor(monkeypatch):
    """Through update a complex64 window keeps only the two planted modes; without the floor it certifies garbage."""
    iterates, proposals = single_precision_history(52)
    tracker = JacobianTracker(0.4)
    tracker.update(iterates, proposals)
    certified = np.sort(tracker.last_lam_pi[tracker.last_gated].real)
    assert tracker.last_lam_pi.size == 2 and np.allclose(certified, [-0.3, 0.5], atol=1e-2)
    monkeypatch.setattr("dgamore.jacobian_stabilization._NOISE_FACTOR", 0.0)
    unfloored = JacobianTracker(0.4)
    unfloored.update(iterates, proposals)
    assert np.max(np.abs(unfloored.last_lam_pi[unfloored.last_gated])) > 3.0


def test_tracker_n_pairs():
    """n_pairs exposes one more than _MONITOR_WINDOW, the number of history pairs update looks at."""
    tracker = JacobianTracker(0.3)
    assert tracker.n_pairs == _MONITOR_WINDOW + 1 == TRACKER_PAIRS


def test_leading_returns_the_largest_modes_padded_with_nan():
    """leading sorts a planted spectrum by descending modulus and pads it with nan; a fresh tracker returns all nan."""
    tracker = JacobianTracker(0.3)
    tracker.last_lam_pi = np.array([0.5 + 0j, 2.0 + 1j])
    tracker.last_res = np.array([1e-3, 2e-3])
    lam_pi, res = tracker.leading(3)
    assert np.array_equal(lam_pi[:2], np.array([2.0 + 1j, 0.5 + 0j])) and np.isnan(lam_pi[2])
    assert np.allclose(res[:2], [2e-3, 1e-3], atol=1e-12) and np.isnan(res[2])
    fresh_lam_pi, fresh_res = JacobianTracker(0.3).leading(3)
    assert np.isnan(fresh_lam_pi).all() and np.isnan(fresh_res).all()


def test_update_marks_whether_an_estimate_was_produced():
    """estimated stays False on an insufficient-history update and turns True once one succeeds."""
    tracker = JacobianTracker(0.2)
    tracker.update([np.zeros((1, 2), dtype=np.complex128)], [np.zeros((1, 2), dtype=np.complex128)])
    assert tracker.estimated is False
    iterates, proposals = linear_pairs(A_2D, B_2D, 0.2, 3, [0.0, 0.0])
    tracker.update(iterates, proposals)
    assert tracker.estimated is True


def test_tracker_logs_spectrum_line():
    """One full update emits exactly one info line carrying the measured lambda_Pi spectrum."""
    logger = MagicMock()
    tracker = JacobianTracker(0.2, logger=logger)
    iterates, proposals = linear_pairs(A_2D, B_2D, 0.2, 3, [0.0, 0.0])
    tracker.update(iterates, proposals)
    assert logger.info.call_count == 1
    assert "lambda_Pi" in logger.info.call_args[0][0]


@pytest.mark.parametrize("alpha, expected", [(0.3, True), (0.02, False)])
def test_tracker_replaces_basis_on_subspace_rotation(alpha, expected):
    """The tracked basis is replaced only when the new unstable direction is further away than _SUBSPACE_ANGLE."""
    tracker = JacobianTracker(0.2)
    iterates, proposals = linear_pairs(rotated_map(0.0), np.array([1.0, 1.0]), 0.2, 9, [0.1, -0.2])
    for end in range(3, 3 + _PERSIST):
        tracker.update(iterates[:end], proposals[:end])
    q_before = tracker.q
    assert tracker.active and (alpha > _SUBSPACE_ANGLE) is expected
    rotated_pairs = linear_pairs(rotated_map(alpha), np.array([1.0, 1.0]), 0.2, 9, [0.1, -0.2])
    assert tracker.update(*rotated_pairs) is expected
    assert (tracker.q is q_before) is not expected
    overlap = np.clip(abs(q_before[:, 0] @ tracker.q[:, 0]), -1.0, 1.0)
    assert np.allclose(np.arccos(overlap), alpha if expected else 0.0, atol=1e-6)


def test_reflector_fixes_certified_stable_directions_of_a_non_normal_map():
    """The reflector flips the certified unstable direction and leaves an overlapping certified stable one fixed."""
    mat, shift = np.array([[4.0, 5.0], [0.0, -3.6]]), np.array([1.0, 1.0])
    u_f, u_s = np.array([1.0, 0.0, 0.0, 0.0]), np.array([5.0, -7.6, 0.0, 0.0]) / np.hypot(5.0, 7.6)
    assert abs(u_f @ u_s) > _FLIP_OVERLAP
    tracker = JacobianTracker(0.2)
    iterates, proposals = linear_pairs(mat, shift, 0.2, 9, [0.01, 0.01])
    for end in range(3, 3 + _PERSIST):
        tracker.update(iterates[:end], proposals[:end])
    assert tracker.active and tracker.w is not None and tracker.q.shape[1] == 1
    reflect = lambda v: v - 2.0 * (tracker.q @ (tracker.w @ v))
    assert np.allclose(reflect(u_f), -u_f, atol=1e-6)
    assert np.allclose(reflect(u_s), u_s, atol=1e-6)
    assert np.allclose(reflect(reflect(u_s + 0.3 * u_f)), u_s + 0.3 * u_f, atol=1e-8)
    assert np.allclose(tracker.w @ u_s, 0.0, atol=1e-6)
    step = lambda x: (mat @ x.real + shift).astype(np.complex128)
    converged = tracked_run(step, [0.01, 0.01], 0.2, 80, JacobianTracker(0.2))
    assert np.allclose(converged, np.linalg.solve(np.eye(2) - mat, shift), atol=1e-6)


def test_reflector_leaves_out_stable_directions_it_cannot_separate():
    """A stable Ritz vector nearly parallel to the flipped one is left out (logged); dependent flipped ones: none."""
    logger = MagicMock()
    tracker = JacobianTracker(0.2, logger=logger)
    theta = np.array([4.0 + 0.0j, -3.6 + 0.0j])
    u = np.column_stack([[1.0, 0.0, 0.0, 0.0], [1.0, 1e-4, 0.0, 0.0] / np.hypot(1.0, 1e-4)]).astype(np.complex128)
    tracker.last_gated = np.array([True, True])
    q, w = tracker._flipped_reflector(theta, u, np.array([True, False]))
    assert np.allclose(w, q.T, atol=1e-12) and "not separable" in logger.info.call_args[0][0]
    assert tracker._flipped_reflector(theta, u, np.array([True, True])) is None


def test_install_carried_returns_false_for_a_lone_conjugate_partner():
    """A single Ritz value with a negative imaginary part collects no flipped column and installs nothing."""
    tracker = JacobianTracker(0.4)
    lam_pi, flip = np.array([-0.2 + 0.3j]), np.array([True])
    assert tracker._install_carried(lam_pi, _real_column(0, 2), flip, np.zeros(1)) is False
    assert tracker.q is None and tracker.w is None


def test_tracker_flips_a_complex_pair_end_to_end():
    """A planted unstable complex pair in a stable map is flipped and converges, while plain damping diverges."""
    gain, angle, stable = 1.3, 0.05, 0.3
    pair = gain * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    block = np.zeros((6, 6))
    block[:2, :2] = pair
    block[2, 2] = stable

    step = lambda x: to_mat(block @ to_vec(x), x.shape)

    tracker = JacobianTracker(2.0)
    x0 = [0.1 + 0.0j, -0.05 + 0.0j, 0.2 + 0.0j]
    tracked = tracked_run(step, x0, 0.5, 150, tracker)
    plain = tracked_run(step, x0, 0.5, 80, None)

    assert tracker.active and tracker.q.shape[1] == 2
    assert np.allclose(tracked, 0.0, atol=1e-6)
    assert np.linalg.norm(plain) > 1e3


def test_spectrum_state_masks_uncertified_modes_and_round_trips(tmp_path):
    """The uncertified mode is dropped everywhere; only sub-band modes get a vector; the state round-trips exactly."""
    shape = (1, 1, 1, 1, 1, 2)  # a window of 2 complex entries flattens to a real representation of length 4
    tracker = JacobianTracker(0.3)
    lam_pi = np.array([-0.3, 0.05, 8.0, -0.7], dtype=np.complex128)
    res = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    flip = np.array([True, False, False, True])
    u = np.array(
        [
            [1.0 + 0.5j, 0.2 + 0.0j, 9.0 + 0.0j, 3.0 + 0.0j],
            [2.0 + 1.5j, -0.2 + 0.0j, -9.0 + 0.0j, 3.0 + 0.0j],
            [3.0 + 2.5j, 0.4 + 0.0j, 8.0 + 0.0j, 3.0 + 0.0j],
            [4.0 + 3.5j, -0.4 + 0.0j, -8.0 + 0.0j, 3.0 + 0.0j],
        ]
    )
    tracker._last_certified = (lam_pi, res, np.array([True, True, True, False]), flip, u)

    state = tracker.spectrum_state(shape, 10.0, True)

    assert np.array_equal(state["lam_pi"], np.array([-0.3, 0.05, 8.0], dtype=np.complex128))
    assert (state["lam_pi"].real < _PREEMPT_BAND).tolist() == state["stored"].tolist() == [True, True, False]
    assert np.array_equal(state["flip"], flip[:3]) and np.array_equal(state["res"], res[:3])
    assert state["u_re"].shape == (2, 2) and state["u_re"].dtype == np.complex64
    assert np.allclose(state["u_re"][:, 0], [1.0 + 3.0j, 2.0 + 4.0j], atol=1e-6)
    assert np.allclose(state["u_im"][:, 0], [0.5 + 2.5j, 1.5 + 3.5j], atol=1e-6)
    assert float(state["beta"]) == 10.0 and float(state["p_eff"]) == tracker.p_eff

    path = str(tmp_path / SPECTRUM_FILE)
    tracker.save_spectrum(path, shape, 10.0, True)
    loaded = load_spectrum(path)
    assert np.array_equal(loaded["lam_pi"], state["lam_pi"])
    assert np.array_equal(loaded["u_re"], state["u_re"]) and np.array_equal(loaded["u_im"], state["u_im"])
    assert tuple(loaded["shape"]) == shape and bool(loaded["converged"]) is True


def test_spectrum_state_without_an_estimate_writes_empty_modes(tmp_path):
    """A tracker that never estimated anything still writes a file, with no mode and the configured damping."""
    shape = (1, 1, 1, 1, 1, 2)
    tracker = JacobianTracker(0.3)
    path = str(tmp_path / SPECTRUM_FILE)
    tracker.save_spectrum(path, shape, 10.0, False)
    loaded = load_spectrum(path)
    assert loaded["lam_pi"].size == 0 and loaded["res"].size == 0 and loaded["stored"].size == 0
    assert loaded["u_re"].shape == (2, 0) and loaded["u_im"].shape == (2, 0)
    assert loaded["u_re"].dtype == np.complex64 and float(loaded["p_eff"]) == 0.3


def test_spectrum_state_keeps_the_last_update_that_certified_a_mode(tmp_path):
    """An update that certifies nothing leaves the modes of the last certified update in the written state."""
    rng = np.random.default_rng(4)
    mat = planted_jacobian(8, [1.3, 0.9], [], rng)
    xs, fs = damped_history(mat, np.zeros(8), 0.3, 9, rng)
    tracker = JacobianTracker(0.3)
    iterates, proposals = list(xs.T.astype(np.complex128)), list(fs.T.astype(np.complex128))
    tracker.update(iterates, proposals)
    certified = tracker.last_lam_pi[tracker.last_gated].astype(np.complex128)
    for _ in range(TRACKER_PAIRS):
        iterates.append(rng.standard_normal(8) + 0j)
        proposals.append(rng.standard_normal(8) + 0j)
        tracker.update(iterates, proposals)
    assert certified.size > 0 and not tracker.last_gated.any()
    assert np.array_equal(tracker.spectrum_state((1, 1, 1, 1, 1, 8), 10.0, False)["lam_pi"], certified)


def test_spectrum_state_ignores_a_scaffolded_update_with_flips_paused():
    """A certified update with allow_flip=False never replaces the snapshot of an earlier physical certification."""
    tracker = JacobianTracker(0.4)
    iterates, proposals = linear_pairs(np.diag([0.4, 1.5]), np.zeros(2), 0.4, TRACKER_PAIRS, [1.0, 1.0])
    tracker.update(iterates, proposals, allow_flip=True)
    certified = tracker.last_lam_pi[tracker.last_gated].astype(np.complex128)

    scaffold = linear_pairs(np.diag([0.1, 0.8]), np.zeros(2), 0.4, TRACKER_PAIRS, [2.0, -3.0])
    tracker.update(*scaffold, allow_flip=False)

    assert tracker.last_gated.any() and not np.array_equal(tracker.last_lam_pi, certified)
    assert np.array_equal(tracker._last_certified[0], certified)
    assert np.array_equal(tracker.spectrum_state((1, 1, 1, 1, 1, 2), 10.0, False)["lam_pi"], certified)


def test_carry_in_flips_modes_below_the_band_and_sets_the_bound():
    """Carried modes with a real part below the band are flipped, stiff ones only bind the damping."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([-0.5, 0.05, 0.3, 8.0], np.eye(4))
    installed = tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)))
    assert installed and tracker.active and tracker.q.shape == (4, 2)
    assert tracker._flip.tolist() == [True, True]  # only the two modes with vectors are held
    assert np.isclose(tracker.p_eff, min(0.4, 0.5 * 2 * 8.0 / 64.0), atol=1e-12)
    proposal = np.array([1.0 + 1.0j, 1.0 + 1.0j]).reshape(1, 1, 1, 1, 1, 2)
    reflected = tracker.reflect(proposal, np.zeros_like(proposal)).reshape(-1)
    assert np.allclose(reflected, [-1.0 + 1.0j, -1.0 + 1.0j], atol=1e-12)


def test_carry_in_without_a_candidate_only_sets_the_bound():
    """Carried modes all above the band install nothing but still lower the damping."""
    tracker = JacobianTracker(0.4)
    assert tracker.carry_in(_carried_state([0.5, 8.0], np.eye(4)), _expand_identity((1, 1, 1, 1, 1, 2))) is False
    assert not tracker.active and np.isclose(tracker.p_eff, 0.125, atol=1e-12)


def test_carry_in_applies_the_expansion_and_renormalizes():
    """Every stored column passes through the expansion and is normalized before the reflector is built."""
    tracker = JacobianTracker(0.4)
    expand = MagicMock(
        side_effect=lambda column: np.concatenate([column.real, column.imag, column.real, column.imag]) * 3.0
    )
    assert tracker.carry_in(_carried_state([-0.2], _real_column(0, 2)), expand)
    assert expand.call_count == 2 and tracker.q.shape == (8, 1)
    assert np.isclose(np.linalg.norm(tracker.q[:, 0]), 1.0, atol=1e-12)


def test_carry_in_drops_columns_that_expand_to_zero():
    """A column that expands to nothing is dropped, and with no flipped column left nothing is installed."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([-0.2, 8.0], np.eye(4))
    assert tracker.carry_in(state, lambda column: np.zeros(4)) is False and not tracker.active


def test_carry_in_refuses_dependent_flipped_columns():
    """Two carried flipped columns that coincide leave no reflector installed."""
    tracker = JacobianTracker(0.4)
    u = np.array([[1.0, 1.0], [1e-9, 0.0], [0.0, 0.0], [0.0, 0.0]])
    state = _carried_state([-0.2, -0.3], u)
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2))) is False and not tracker.active


def test_carry_in_defers_the_flip_but_snapshots_the_carried_set_while_flips_are_paused():
    """With flips paused the damping applies at once, the reflector waits, and the carried set is still snapshotted."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([-0.5, 8.0], np.eye(4))
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)), allow_flip=False) is False
    assert not tracker.active and np.isclose(tracker.p_eff, 0.125, atol=1e-12) and tracker._pending is not None
    assert tracker._last_certified is not None
    assert np.array_equal(tracker._last_certified[0], np.array([-0.5 + 0.0j]))


def test_carry_in_returns_false_when_no_carried_mode_lies_below_the_flip_band():
    """Only a file written with a wider flip band reaches this branch; nothing is flipped and no error is raised."""
    tracker = JacobianTracker(0.4)
    shape = (1, 1, 1, 1, 1, 2)
    u_re = to_mat(np.array([1.0, 0.0, 0.0, 0.0]), shape).reshape(-1, 1).astype(np.complex64)
    state = {
        "lam_pi": np.array([0.5 + 0.0j]),
        "res": np.array([0.0]),
        "stored": np.array([True]),
        "u_re": u_re,
        "u_im": np.zeros_like(u_re),
    }
    assert tracker.carry_in(state, _expand_identity(shape)) is False
    assert not tracker.active


def test_carry_in_installs_the_orthogonal_reflector_of_a_conjugate_pair():
    """A carried complex-conjugate pair builds an orthogonal reflector on its shared real plane."""
    tracker = JacobianTracker(0.4)
    shape = (1, 1, 1, 1, 1, 2)
    w = np.array([1.0 + 0.5j, 0.3 - 0.2j, -0.4 + 0.1j, 0.2 + 0.3j])
    state = _carried_state([-0.2 + 0.3j, -0.2 - 0.3j], np.column_stack([w, np.conj(w)]), shape)
    assert tracker.carry_in(state, _expand_identity(shape))
    assert tracker.active and tracker.q.shape == (4, 2)
    assert np.allclose(tracker._dual, tracker.q.T, atol=1e-12)
    proposal = np.array([1.0 + 1.0j, 1.0 + 1.0j]).reshape(shape)
    reflected = tracker.reflect(proposal, np.zeros_like(proposal))
    r = to_vec(proposal)
    expected = to_mat(r - 2.0 * tracker.q @ (tracker.q.T @ r), shape)
    assert np.allclose(reflected, expected, atol=1e-12)


def test_carry_in_never_raises_p_eff():
    """A carried bound above the current damping does not raise p_eff back up."""
    tracker = JacobianTracker(0.4)
    tracker.p_eff = 0.02
    tracker.carry_in(_carried_state([0.5, 8.0], np.eye(4)), _expand_identity((1, 1, 1, 1, 1, 2)))
    assert np.isclose(tracker.p_eff, 0.02, atol=1e-12)


def test_carry_in_with_an_empty_spectrum_leaves_the_damping_at_the_configured_value():
    """An empty carried spectrum installs nothing and carries no bound, so the damping stays at its configured value."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([], np.zeros((4, 0)), p_eff=0.055)
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2))) is False
    assert not tracker.active and np.isclose(tracker.p_eff, 0.4, atol=1e-12)


def test_carry_in_ignores_the_carried_damping_in_favor_of_the_bound_of_the_spectrum():
    """The bound of the carried spectrum alone sets the damping, even where the carried p_eff lies below it."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([0.5, 8.0], np.eye(4), p_eff=0.02)
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2))) is False
    assert np.isclose(tracker.p_eff, 0.125, atol=1e-12)


def test_carry_in_sets_the_flip_bound_on_install():
    """A carried set binds the damping with its stiff pair and the flip bound with the flipped mode alone."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([-9.4429, 34.4004 - 25.595j, 34.4004 + 25.595j], np.eye(4))
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)))
    assert np.isclose(tracker.p_eff, 0.5 * 2.0 * 34.4004 / (34.4004**2 + 25.595**2), atol=1e-4)
    assert np.isclose(tracker.p_flip, 0.5 * 2.0 * 9.4429 / 9.4429**2, atol=1e-4)


def test_flip_bound_of_a_held_carried_basis_never_loosens():
    """A held carried basis keeps its own flip bound when the current estimate's flipped mode allows more."""
    tracker = JacobianTracker(0.4)
    assert tracker.carry_in(_carried_state([-40.0], np.eye(4)[:, :1]), _expand_identity((1, 1, 1, 1, 1, 2)))
    assert np.isclose(tracker.p_flip, 0.025, atol=1e-6)
    tracker.update(*linear_pairs(np.diag([5.0, -39.0]), np.zeros(2), 0.01, 9, [1.0, 1.0]))
    assert tracker._carried and tracker._flip.any()
    assert np.isclose(tracker.p_flip, 0.025, atol=1e-6)


def test_flip_bound_of_a_held_carried_basis_tightens():
    """A held carried basis takes the current estimate's flip bound when that one is tighter."""
    tracker = JacobianTracker(0.4)
    assert tracker.carry_in(_carried_state([-4.0], np.eye(4)[:, :1]), _expand_identity((1, 1, 1, 1, 1, 2)))
    assert np.isclose(tracker.p_flip, 0.25, atol=1e-6)
    tracker.update(*linear_pairs(np.diag([9.0, -39.0]), np.zeros(2), 0.01, 9, [1.0, 1.0]))
    assert tracker._carried and tracker._flip.any()
    assert np.isclose(tracker.p_flip, 0.125, atol=1e-6)


def test_carry_in_keeps_the_flip_bound_pending_until_the_install():
    """A carried flip kept pending leaves the flip bound alone until the deferred install sets it."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([-9.4429, 34.4004 - 25.595j, 34.4004 + 25.595j], np.eye(4))
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)), allow_flip=False) is False
    assert tracker.p_flip == 0.4
    iterates, proposals = _stable_pairs(4)
    tracker.update(iterates[:3], proposals[:3], allow_flip=False)
    tracker.update(iterates[:4], proposals[:4], allow_flip=True)
    assert tracker.active and np.isclose(tracker.p_flip, 0.5 * 2.0 * 9.4429 / 9.4429**2, atol=1e-4)


def _stable_pairs(n_iter: int, p: float = 0.4) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Damped pairs of a diagonal map whose two modes are stable (lambda_Pi 0.6 and 0.3)."""
    return linear_pairs(np.diag([0.4, 0.7]), np.zeros(2), p, n_iter, [1.0, 1.0])


def _carry(tracker, lam, column, n, allow_flip=True):
    """Carries one mode with the given real-representation column on an n-dimensional window."""
    shape = (1, 1, 1, 1, 1, n)
    return tracker.carry_in(_carried_state([lam], column, shape=shape), _expand_identity(shape), allow_flip=allow_flip)


def test_carried_basis_survives_calm_updates_without_a_certified_overlap():
    """Calm updates alone do not release a carried basis while the tracker certifies nothing that overlaps it."""
    tracker = JacobianTracker(0.4)
    _carry(tracker, -0.2, _real_column(2, 3), 3)
    iterates, proposals = linear_pairs(np.diag([0.4, 0.7, 1.0]), np.zeros(3), 0.4, 8, [1.0, 1.0, 0.0])
    for n in range(3, 9):
        tracker.update(iterates[:n], proposals[:n])
    assert tracker.active and tracker._carried


def test_carried_basis_is_released_by_a_certified_stable_overlap():
    """A certified stable mode lying in the carried basis releases it after three such updates."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    _carry(tracker, -0.2, _real_column(0, 2), 2)
    iterates, proposals = _stable_pairs(9)
    for n in range(3, 10):
        tracker.update(iterates[:n], proposals[:n])
    assert not tracker.active and not tracker._carried
    assert any("lies in the carried basis" in str(call) for call in logger.info.call_args_list)


def test_carried_basis_is_released_by_a_growing_residual():
    """A preemptive flip carried onto a direction that is in fact stable makes the residual grow and is released."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    _carry(tracker, 0.05, _real_column(0, 2), 2)
    assert tracker.predicted_rate() > 1.0  # the flipped mode expands at every damping until it is released
    tracker._last_residual_norm = 0.0
    mat = np.diag([0.4, 0.7])
    tracked_run(lambda x: mat @ x.real + 0j, [1.0, 1.0], 0.4, 6, tracker)
    assert not tracker.active
    assert any("residual grew" in str(call) for call in logger.info.call_args_list)


def test_carried_basis_is_replaced_by_a_certified_flip():
    """A certified unstable mode that persists installs the tracker's own reflector in place of the carried one."""
    tracker = JacobianTracker(0.4)
    _carry(tracker, -0.2, _real_column(0, 2), 2)
    iterates, proposals = linear_pairs(np.diag([0.4, 1.5]), np.zeros(2), 0.4, 9, [1.0, 1.0])
    for n in range(3, 10):
        tracker.update(iterates[:n], proposals[:n])
    assert tracker.active and not tracker._carried
    assert np.isclose(abs(tracker.q[1, 0]), 1.0, atol=1e-6)  # row 1 is the real part of the second coordinate


def test_carried_flip_on_the_certified_direction_is_still_replaced():
    """A certified flip landing on the exact carried direction still replaces it instead of being mistaken for it."""
    tracker = JacobianTracker(0.4)
    _carry(tracker, -0.2, _real_column(1, 2), 2)
    iterates, proposals = linear_pairs(np.diag([0.4, 1.5]), np.zeros(2), 0.4, 9, [1.0, 1.0])
    for n in range(3, 10):
        tracker.update(iterates[:n], proposals[:n])
    assert tracker.active and not tracker._carried


def test_pending_carried_flip_installs_on_the_first_free_update():
    """A carried flip kept pending under a scaffold is installed on the first update that allows flips."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    _carry(tracker, -0.5, _real_column(0, 2), 2, allow_flip=False)
    iterates, proposals = _stable_pairs(4)
    tracker.update(iterates[:3], proposals[:3], allow_flip=False)
    assert not tracker.active
    tracker.update(iterates[:4], proposals[:4], allow_flip=True)
    assert tracker.active and tracker._carried and tracker._pending is None
    assert any("carried reflector installed" in str(call) for call in logger.info.call_args_list)


def test_pending_carried_flip_reports_a_switch_on_install():
    """A pending carried flip installed on the first free update is reported as a switch of the reflected map."""
    tracker = JacobianTracker(0.4)
    _carry(tracker, -0.5, _real_column(0, 2), 2, allow_flip=False)
    iterates, proposals = _stable_pairs(4)
    tracker.update(iterates[:3], proposals[:3], allow_flip=False)
    assert tracker.update(iterates[:4], proposals[:4], allow_flip=True) is True


def test_carried_basis_is_released_when_flips_are_suppressed():
    """An allow_flip=False update releases an installed carried basis and clears the carried flag."""
    tracker = JacobianTracker(0.4)
    _carry(tracker, -0.2, _real_column(0, 2), 2)
    iterates, proposals = _stable_pairs(3)
    tracker.update(iterates, proposals, allow_flip=False)
    assert not tracker.active and not tracker._carried


def test_carried_basis_released_by_a_scaffold_is_re_pended_and_re_installed():
    """A carried basis released while flips are paused waits as pending and returns on the first free update."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    _carry(tracker, -0.5, _real_column(0, 2), 2)
    iterates, proposals = _stable_pairs(4)
    tracker.update(iterates[:3], proposals[:3], allow_flip=False)
    assert not tracker.active and tracker._pending is not None
    assert any("carried set pending until flips are allowed again" in str(call) for call in logger.info.call_args_list)
    tracker.update(iterates[:4], proposals[:4], allow_flip=True)
    assert tracker.active and tracker._carried and tracker._pending is None

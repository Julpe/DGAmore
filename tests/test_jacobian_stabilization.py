# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Unit tests for dgamore.jacobian_stabilization on small dense maps with spectra known analytically or from np.linalg.eig:
secant Rayleigh-Ritz and its noise floor, the classification table, the Schur reflector and per-direction map, the
tracker on the toy maps of arXiv:2606.04936, and the spectrum state a run writes, carries over and releases.
"""

import weakref
import zipfile
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.linalg import block_diag

import dgamore.jacobian_stabilization as jacobian_stabilization
from dgamore.jacobian_stabilization import _CHECK_BAND, _CHECK_START, _FLIP_OVERLAP, _FLIP_PERSIST, _MAP_NORM_CAP
from dgamore.jacobian_stabilization import _MARGIN, _MAX_BETA_STEP_RATIO, _MAX_EXACT_CHECKS, _MIN_ITERATION_CAP
from dgamore.jacobian_stabilization import _MIXING_FLOOR, _NOISE_FACTOR, _PERSIST, _RITZ_GATE, _SCHUR_TOL
from dgamore.jacobian_stabilization import _SIGNS_UNBOUNDED, _STORE_BAND, _SUBSPACE_ANGLE, JACOBIAN_FILE
from dgamore.jacobian_stabilization import JacobianTracker, SchurForm, TRACKER_PAIRS, _carried_nonuniform_map
from dgamore.jacobian_stabilization import _conjugate_partners, _diagonal_positions, _direction_damping, _match_modes
from dgamore.jacobian_stabilization import _nonuniform_map, _ritz_from_increments, _schur_ritz, classify
from dgamore.jacobian_stabilization import detect_pole_jump, load_spectrum, predict_crossing, predict_rung_crossing
from dgamore.jacobian_stabilization import secant_ritz, to_mat, to_vec
from tests.conftest import linear_pairs

MU = 1.0
A_2D = np.array([[2.0, 0.1], [0.1, -2.0]])
B_2D = np.array([1.0, 1.0])
_SHAPE = (1, 1, 1, 1, 1, 2)  # a window of two complex entries, four real coordinates


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
    in_block = block_diag(*[np.ones(b.shape) for b in blocks]) > 0.0
    upper = np.where(in_block, 0.0, np.triu(rng.standard_normal((n, n)), k=1) * coupling)
    orth, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return orth @ (block_diag(*blocks) + upper) @ orth.T


def damped_history(
    mat: np.ndarray, shift: np.ndarray, p: float, n_iter: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Runs x -> x + p (A x + b - x) from a small random start and returns the iterate and proposal column stacks."""
    pairs = linear_pairs(mat, shift, p, n_iter, rng.standard_normal(mat.shape[0]) * 1e-2)
    return tuple(np.column_stack(stack).real.copy() for stack in pairs)


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
    steps = [np.insert(np.diff(a, axis=1), index, np.diff(a, axis=1)[:, index], axis=1) for a in (xs, fs)]
    return tuple(np.column_stack([a[:, 0], a[:, :1] + np.cumsum(d, axis=1)]) for a, d in zip((xs, fs), steps))


def feed_windows(tracker: JacobianTracker, pairs: tuple, n_updates: int | None = None, **kwargs) -> list[bool]:
    """Updates the tracker on the growing windows pairs[:3], pairs[:4], ... and returns its events."""
    stop = len(pairs[0]) + 1 if n_updates is None else 3 + n_updates
    return [tracker.update(pairs[0][:end], pairs[1][:end], **kwargs) for end in range(3, stop)]


def tracked_run(
    step_fn: Callable[[np.ndarray], np.ndarray],
    x0: list[float],
    p: float,
    n_iter: int,
    tracker: JacobianTracker | None,
    bound_only_with_basis: bool = False,
) -> np.ndarray:
    """Runs the loop's protocol: proposal, record pair, update, reflect, mix, stabilize; returns the iterate."""
    x = np.asarray(x0, dtype=np.complex128)
    iterates, proposals = [], []
    for _ in range(n_iter):
        proposal = step_fn(x)
        iterates.append(x.copy())
        proposals.append(proposal.copy())
        if tracker is not None:
            tracker.update(iterates, proposals)
        alpha = p if tracker is None or (bound_only_with_basis and not tracker.active) else tracker.p_eff
        used = proposal if tracker is None else tracker.reflect(proposal, x)
        mixed = x + alpha * (used - x)
        if tracker is not None:
            mixed = tracker.stabilize_step(mixed, x, proposal - x)
        x = mixed
    return x.real


def rotated_map(alpha: float) -> np.ndarray:
    """Returns a real 2x2 map with eigenvalues 2.0 and 0.5 whose unstable eigenvector is rotated by ``alpha``."""
    rot = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return rot @ np.diag([2.0, 0.5]) @ rot.T


def _logged_tracker(p_config: float = 0.4) -> tuple[JacobianTracker, MagicMock]:
    """A tracker at the configured damping p_config whose logger is a MagicMock, returned beside it."""
    logger = MagicMock()
    return JacobianTracker(p_config, logger=logger), logger


def _carried_state(
    lam_pi: list[complex],
    u: np.ndarray,
    shape: tuple[int, ...] = (1, 1, 1, 1, 1, 2),
    res: list[float] | None = None,
    converged: bool = True,
    p_eff: float = 0.4,
    beta: float = 10.0,
    lam_prev: list[complex] | None = None,
    res_prev: list[float] | None = None,
    beta_prev: float | None = None,
    exact: bool = False,
) -> dict[str, np.ndarray]:
    """Builds the spectrum state a run writes from real-representation columns, vectors below the storage band."""
    lam_pi = np.asarray(lam_pi, dtype=np.complex128)
    u = np.asarray(u, dtype=np.complex128)
    stored = lam_pi.real < _STORE_BAND
    lam_prev = np.full(lam_pi.size, np.nan, dtype=np.complex128) if lam_prev is None else np.asarray(lam_prev)
    lam_prev = lam_prev.astype(np.complex128)
    return {
        "lam_pi": lam_pi,
        "res": np.zeros(lam_pi.size) if res is None else np.asarray(res, dtype=np.float64),
        "flip": lam_pi.real < 0.0,
        "predicted": np.zeros(lam_pi.size, dtype=bool),
        "stored": stored,
        "lam_prev": lam_prev,
        "res_prev": np.where(np.isnan(lam_prev), np.nan, 0.0) if res_prev is None else np.asarray(res_prev),
        "beta_prev": np.where(np.isnan(lam_prev), np.nan, np.nan if beta_prev is None else beta_prev),
        "u_re": u[:, np.flatnonzero(stored)].real.astype(np.float32),
        "u_im": u[:, np.flatnonzero(stored)].imag.astype(np.float32),
        "shape": np.asarray(shape, dtype=np.int64),
        "beta": np.float64(beta),
        "p_eff": np.float64(p_eff),
        "converged": np.bool_(converged),
        "exact": np.bool_(exact),
    }


def _expand_identity():
    """Expansion for equal grids: the stored column is the real vector itself."""
    return lambda column: np.asarray(column, dtype=np.float64)


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
    assert back.dtype == np.complex128 and np.allclose(back, mat, atol=1e-6)


@pytest.mark.parametrize("n, real_eigs, pair_eigs", [(48, [1.3], []), (52, [], [1.05 + 0.4j])], ids=["real", "pair"])
def test_secant_ritz_recovers_the_dominant_planted_mode(n, real_eigs, pair_eigs):
    """A planted real lambda_J = 1.3 or pair 1.05 +- 0.4j is certified among the damped iteration's own Ritz values."""
    rng = np.random.default_rng(0)
    mat = planted_jacobian(n, real_eigs, pair_eigs, rng)
    xs, fs = damped_history(mat, rng.standard_normal(n), 0.5, 14, np.random.default_rng(90))
    theta, _, res = secant_ritz(xs[:, -7:], fs[:, -7:])
    gated = res <= _RITZ_GATE * np.maximum(1.0, np.abs(theta))
    for target in real_eigs + pair_eigs + [lam.conjugate() for lam in pair_eigs]:
        assert np.min(np.abs(theta[gated] - target)) < 1e-3
    if real_eigs:
        assert np.min(res[np.abs(theta - 1.3) < 1e-3]) < _RITZ_GATE


def test_secant_ritz_matches_dense_eig_on_tiny_map():
    """Seven pairs of a six-dimensional map span the whole space, so the Ritz values are the dense eigenvalues."""
    rng = np.random.default_rng(0)
    mat = planted_jacobian(6, [], [], rng, coupling=0.4, stable=1.4)
    xs, fs = damped_history(mat, rng.standard_normal(6), 1.0, 7, np.random.default_rng(100))
    theta, _, res = secant_ritz(xs, fs)
    assert theta.size == 6 and np.allclose(np.sort_complex(theta), np.sort_complex(np.linalg.eigvals(mat)), atol=1e-8)
    assert np.max(res) < 1e-8


def test_secant_ritz_survives_collinear_history():
    """A repeated secant increment is dropped, leaving the spectrum of the independent columns intact."""
    rng = np.random.default_rng(0)
    mat = planted_jacobian(20, [1.3], [], rng)
    xs, fs = damped_history(mat, rng.standard_normal(20), 0.5, 8, np.random.default_rng(90))
    theta, _, _ = secant_ritz(xs, fs)
    theta_dup, _, res_dup = secant_ritz(*duplicate_secant(xs, fs, 2))
    assert np.all(np.isfinite(theta_dup)) and np.all(np.isfinite(res_dup))
    assert theta_dup.size == theta.size and np.allclose(np.sort_complex(theta_dup), np.sort_complex(theta), atol=1e-8)


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
        ([0.003 + 0.5j, 0.003 - 0.5j], [True, True], 0.4, [False, False], 0.5 * 2.0 * 0.003 / (0.003**2 + 0.25)),
        ([-0.003 + 0.5j, -0.003 - 0.5j], [True, True], 0.4, [False, False], 0.5 * 2.0 * _MARGIN / (0.003**2 + 0.25)),
        ([0.001 + 0.5j, 0.001 - 0.5j], [True, True], 0.4, [False, False], _MIXING_FLOOR),
        ([2.0j, -2.0j], [True, True], 0.4, [False, False], _MIXING_FLOOR),
        ([-0.3], [False], 0.4, [False], 0.4),
        ([-0.005], [True], 0.4, [False], 0.4),
        ([-0.02], [True], 0.4, [True], 0.4),
        ([-200.0], [True], 0.4, [True], _MIXING_FLOOR),
        ([-200.0], [True], 0.005, [True], 0.005),
    ],
)
def test_classify_table(lam_pi, gated, p_config, flip_expected, p_expected):
    """Clearly negative modes flip; a marginal mode never flips and binds with its stable real part or the margin."""
    flip, p_eff = classify(np.array(lam_pi, dtype=np.complex128), np.array(gated), p_config)
    assert np.array_equal(flip, np.array(flip_expected)) and np.allclose(p_eff, p_expected, atol=1e-12)


def test_classify_widens_the_band_with_the_residual():
    """A residual larger than the margin widens the undecidable band: no flip inside it, and it enters the bound."""
    lam_pi, gated = np.array([-0.04 + 0.0j, -0.04 + 0.0j]), np.array([True, True])
    flip, p_eff = classify(lam_pi, gated, 0.4, res=np.array([0.001, 0.05]))
    assert np.array_equal(flip, np.array([True, False])) and np.allclose(p_eff, 0.4, atol=1e-12)
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


def test_reflect_and_stabilize_step_hand_back_their_input_without_an_install():
    """Without a tracked subspace the reflection and without a map the stabilized step return their input object."""
    rng = np.random.default_rng(1)
    tracker = JacobianTracker(0.4)
    proposal, iterate = (rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3)) for _ in range(2))
    assert tracker.reflect(proposal, iterate) is proposal
    assert tracker.stabilize_step(proposal, iterate, proposal - iterate) is proposal


_NONUNIFORM_GAINS = np.diag([-39.0, 0.5, 5.0, 0.2])  # lambda_Pi = +40, +0.5 and -4 on the span, +0.8 outside it


def _installed_nonuniform_tracker() -> JacobianTracker:
    """Tracks a damped history of _NONUNIFORM_GAINS that never excites the fourth direction, up to the install."""
    tracker = JacobianTracker(0.4)
    iterates, proposals = linear_pairs(_NONUNIFORM_GAINS, np.zeros(4), 0.01, 9, [1.0, 1.0, 1.0, 0.0])
    for end in range(3, 10):
        tracker.update(iterates[:end], proposals[:end])
        if tracker.q is not None:
            break
    return tracker


def _step_factor(tracker: JacobianTracker, gains: np.ndarray, index: int, p_step: float = 0.3) -> float:
    """Composite reflect -> mix -> stabilize factor a tracker gives one coordinate of a diagonal map."""
    iterate = np.zeros(gains.shape[0], dtype=np.complex128)
    iterate[index] = 1e-3
    proposal = (gains @ iterate.real).astype(np.complex128)
    mixed = iterate + p_step * (tracker.reflect(proposal, iterate) - iterate)
    return float(tracker.stabilize_step(mixed, iterate, proposal - iterate).real[index]) / 1e-3


def _map_gain(tracker: JacobianTracker, residual: np.ndarray) -> float:
    """Norm the installed map adds to a mixed step, i.e. ||x_stab - x_mix + Pi_cert (x_mix - x)|| / ||r||."""
    span, dual = tracker._nonuniform[0], tracker._nonuniform[1]
    mixed = 0.3 * residual
    iterate = np.zeros_like(residual)
    added = to_vec(tracker.stabilize_step(mixed, iterate, residual)) - to_vec(mixed) + span @ (dual @ to_vec(mixed))
    return float(np.linalg.norm(added) / np.linalg.norm(to_vec(residual)))


def _schur_map(b_proj: np.ndarray, q_kept: np.ndarray, gated: list[bool], log: MagicMock | None = None) -> tuple:
    """Builds the per-direction map of b_proj on q_kept with the flip below -_MARGIN; returns lambda_Pi, flip, map."""
    theta, _, form = _schur_ritz(b_proj)
    lam_pi = 1.0 - theta
    flip = lam_pi.real < -_MARGIN
    return lam_pi, flip, _nonuniform_map(q_kept, form, theta, lam_pi, flip, np.array(gated), log)


def _mapped_columns(nonuniform: tuple, columns: np.ndarray) -> np.ndarray:
    """Applies an installed per-direction map to a set of columns, i.e. the certified part of the stabilized step."""
    span, dual, damping = nonuniform
    return span @ (damping @ (dual @ columns))


def test_certified_directions_contract_by_their_own_damping():
    """Every certified direction contracts by its own capped Eq. 27 damping and the untracked one by the step scalar."""
    tracker, p_step = _installed_nonuniform_tracker(), 0.3
    factors = [_step_factor(tracker, _NONUNIFORM_GAINS, index, p_step) for index in range(4)]
    assert tracker.active and np.allclose(factors, [0.0, 0.5, 0.0, 1.0 - p_step * 0.8], atol=1e-8)


def test_the_installed_map_follows_a_stiffening_estimate():
    """A basis kept while its flipped eigenvalue stiffens takes the new damping, not the one it was installed with."""
    gains = np.diag([-39.0, 0.5, 41.0, 0.2])  # the flipped direction's lambda_Pi moves from -4 to -40
    tracker = _installed_nonuniform_tracker()
    q_before = tracker.q
    tracker.update(*linear_pairs(gains, np.zeros(4), 0.01, 9, [1.0, 1.0, 1.0, 0.0]))
    iterate = 1e-3 * np.eye(4, dtype=np.complex128)[2]
    residual = (gains @ iterate.real).astype(np.complex128) - iterate
    assert tracker.q is q_before
    assert np.allclose(tracker.stabilize_step(iterate, iterate, residual).real[2], 0.0, atol=1e-8)


def test_a_refreshed_map_unflips_a_direction_the_estimate_certifies_as_stable():
    """A flipped direction certified stable again takes its own positive damping on the very next update."""
    gains = np.diag([-39.0, 0.5, 0.6, 0.2])  # the flipped direction's lambda_Pi moves from -4 to +0.4
    tracker = _installed_nonuniform_tracker()
    tracker.update(*linear_pairs(gains, np.zeros(4), 0.01, 9, [1.0, 1.0, 1.0, 0.0]))
    assert np.isclose(_step_factor(tracker, gains, 2), 0.6, atol=1e-6) and tracker.predicted_rate() < 1.0


def test_a_new_flip_reaches_the_step_of_the_update_that_certifies_it():
    """A direction certified unstable beside a certified stable one takes its reversed damping on that very update."""
    tracker = JacobianTracker(0.4)
    gains = np.diag([0.4, 5.0])
    iterates, proposals = linear_pairs(gains, np.zeros(2), 0.01, 9, [1.0, 1.0])
    factors = []
    # a fourth update of this synthetic, never-reflected history would release the basis by the growth interlock
    for end in range(3, 3 + _PERSIST):
        tracker.update(iterates[:end], proposals[:end])
        factors.append(_step_factor(tracker, gains, 1))
    assert np.allclose(factors, [0.0, 0.0, 0.0], atol=1e-6)


def test_a_persisted_flip_whose_reflector_is_refused_stays_out_of_the_map(monkeypatch):
    """A persisted flip whose reflector could not be built keeps the mixing's step and stays out of the map."""
    tracker, logger = _logged_tracker()
    monkeypatch.setattr(JacobianTracker, "_flipped_reflector", lambda self, flip, q_kept, form: None)
    gains = np.diag([0.4, 5.0])
    iterates, proposals = linear_pairs(gains, np.zeros(2), 0.01, 9, [1.0, 1.0])
    factors = []
    for end in range(3, 3 + _PERSIST):
        tracker.update(iterates[:end], proposals[:end])
        factors.append(_step_factor(tracker, gains, 1))
    assert tracker._persist == _PERSIST and tracker.q is None and tracker.active
    assert np.allclose(factors, [2.2, 2.2, 2.2], atol=1e-6) and np.isnan(tracker._applied[tracker._flip]).all()
    assert any("could not be assembled into a reflector" in call[0][0] for call in logger.info.call_args_list)


_STABLE_ONLY_GAINS = np.diag([-39.0, 0.5, 0.2])  # lambda_Pi = +40 (stiff), +0.5 and +0.8 (flat), nothing to flip


def _stable_only_tracker(n_updates: int) -> tuple[JacobianTracker, list[bool]]:
    """Feeds n_updates windows of a decaying stable-only history and returns the tracker with the events."""
    tracker = JacobianTracker(0.4)
    iterates, proposals = linear_pairs(_STABLE_ONLY_GAINS, np.zeros(3), 0.01, 3 + n_updates, [1.0, 1.0, 1.0])
    events = [tracker.update(iterates[:end], proposals[:end]) for end in range(4, 4 + n_updates)]
    return tracker, events


def test_a_stable_only_spectrum_installs_the_map_without_a_reflector():
    """A stable-only certified spectrum installs the map alone, no reflector, and reports no event."""
    tracker, events = _stable_only_tracker(1)
    factors = [_step_factor(tracker, _STABLE_ONLY_GAINS, index) for index in range(3)]
    assert tracker.active and tracker.q is None and events == [False]
    assert np.isclose(tracker.p_eff, 0.025, atol=1e-12) and np.allclose(factors, [0.0, 0.5, 0.2], atol=1e-8)


def test_the_stable_only_map_survives_calm_updates():
    """A map with no flip to undo is not calm-released: six calm certifying updates keep it, none is an event."""
    tracker, events = _stable_only_tracker(6)
    assert tracker.active and tracker.q is None and events == [False] * 6
    assert np.isclose(_step_factor(tracker, _STABLE_ONLY_GAINS, 1), 0.5, atol=1e-8)


def _grow_residual(tracker: JacobianTracker, n_updates: int, gains: tuple = (0.5, 0.2), shift: float = 0.0) -> list:
    """Feeds n_updates windows of diag(gains), each restarted further out so the residual grows; returns the events."""
    windows = (linear_pairs(np.diag(gains), np.full(2, shift), 0.3, 3, [10.0 * (k + 1)] * 2) for k in range(n_updates))
    return [tracker.update(*pairs) for pairs in windows]


def test_a_growing_residual_releases_the_map_and_holds_it_off_for_persist_updates():
    """Three consecutive residual growths release a map-only state, which stays off for _PERSIST more updates."""
    tracker, logger = _logged_tracker()
    assert not any(_grow_residual(tracker, 1 + _PERSIST))
    assert not tracker.active and "per-direction damping released" in logger.info.call_args[0][0]
    assert not any(_grow_residual(tracker, _PERSIST)) and not tracker.active
    assert not any(_grow_residual(tracker, 1)) and tracker.active and tracker.q is None


def test_nonuniform_map_damps_each_certified_direction_of_a_non_normal_map():
    """Every certified Ritz direction is an eigendirection of the map at its own signed damping, which is reported."""
    q_kept, b_proj = _projected_map(7, [4.0, -3.6, 0.7, 0.4], 0.5)
    theta, y, form = _schur_ritz(b_proj)
    lam_pi, gated = 1.0 - theta, np.array([True, True, True, False])
    (span, dual, damping), applied = _nonuniform_map(q_kept, form, theta, lam_pi, lam_pi.real < -_MARGIN, gated)
    expected = np.where(lam_pi.real < -_MARGIN, -1.0, 1.0) * np.minimum(1.0, np.abs(lam_pi.real) / np.abs(lam_pi) ** 2)
    coords = q_kept.T @ span
    assert span.shape == (10, 3) and np.allclose(b_proj @ coords, coords @ (coords.T @ b_proj @ coords), atol=1e-10)
    assert np.allclose(applied[gated], expected[gated], atol=1e-12) and np.isnan(applied[3])
    for index in np.flatnonzero(gated):
        column = q_kept @ y[:, index]
        assert np.allclose(span @ (damping @ (dual @ column)), expected[index] * column, atol=1e-10)


def test_the_oblique_projector_leaves_an_uncertified_direction_of_a_non_normal_map_at_the_mixings_step():
    """An uncertified Ritz direction of a non-normal map keeps exactly the step the mixing gave it."""
    q_kept, b_proj = _projected_map(7, [4.0, -3.6, 0.7, 0.4], 0.5)
    theta, y, form = _schur_ritz(b_proj)
    lam_pi, gated, p_step = 1.0 - theta, np.array([True, True, True, False]), 0.2
    tracker = JacobianTracker(0.4)
    tracker._nonuniform = _nonuniform_map(q_kept, form, theta, lam_pi, lam_pi.real < -_MARGIN, gated)[0]
    column = (q_kept @ y[:, 3]).real
    iterate, residual = to_mat(1e-3 * column, (5,)), to_mat(-1e-3 * lam_pi[3].real * column, (5,))
    stabilized = tracker.stabilize_step(iterate + p_step * residual, iterate, residual)
    assert np.isclose(to_vec(stabilized) @ column / 1e-3, 1.0 - p_step * lam_pi[3].real, atol=1e-8)


def test_the_certified_projector_falls_back_to_the_orthogonal_one_on_a_nearly_parallel_complement():
    """A certified subspace closer to the uncertified one than the obliquity cap is projected orthogonally."""
    log = MagicMock()
    (span, dual, _), _ = _schur_map(np.array([[4.0, 100.0], [0.0, 0.7]]), np.eye(8, 2), [True, False], log)[2]
    assert np.allclose(dual, span.T, atol=1e-12)
    assert any("orthogonal projector" in call[0][0] for call in log.call_args_list)


def test_nonuniform_map_leaves_an_uncertified_direction_outside_the_certified_span():
    """An uncertified Ritz direction of a normal map is orthogonal to the span the damping is built on."""
    q_kept, b_proj = _projected_map(7, [4.0, -3.6, 0.7, 0.4], 0.0)
    theta, y, form = _schur_ritz(b_proj)
    lam_pi, gated = 1.0 - theta, np.array([True, True, True, False])
    span = _nonuniform_map(q_kept, form, theta, lam_pi, lam_pi.real < -_MARGIN, gated)[0][0]
    assert np.allclose(span.T @ (q_kept @ y[:, 3]), 0.0, atol=1e-10)


_FALLBACK_FORM = np.array([[5.0, 0.2, 0.1], [0.0, 0.3, 0.5], [0.0, 0.0, 0.3]])  # one flipped and two stable blocks
_COUPLED_FORM = np.array(  # a flip at -0.5 beside a near-neutral pair 0.002 -+ 0.5i coupled by 10 to a stiff +1.6
    [[1.5, 0.0, 0.0, 0.0], [0.0, 0.998, 0.5, 10.0], [0.0, -0.5, 0.998, 10.0], [0.0, 0.0, 0.0, -0.6]]
)


def _coupled_fallback() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple, np.ndarray]:
    """Builds the map of _COUPLED_FORM; returns lambda_Pi, the flip mask, the Ritz vectors, the map and its damping."""
    theta, y, form = _schur_ritz(_COUPLED_FORM)
    lam_pi, q_kept = 1.0 - theta, np.eye(8, 4)
    flip = lam_pi.real < -_MARGIN
    built, applied = _nonuniform_map(q_kept, form, theta, lam_pi, flip, np.ones(4, dtype=bool))
    return lam_pi, flip, q_kept @ y, built, applied


def test_the_fallback_damps_only_the_coupled_blocks_uniformly_and_signs_every_eigenvector():
    """The pair and the stiff mode share the pair's damping while the uncoupled flip keeps its own -1, exactly."""
    lam_pi, flip, u, built, applied = _coupled_fallback()
    damping = _direction_damping(lam_pi, flip)
    expected = np.where(flip, damping, np.min(damping[~flip]))
    assert np.isclose(damping[flip][0], -1.0, atol=1e-12) and np.min(damping[~flip]) < 0.01
    assert np.allclose(applied, expected, atol=1e-12)
    assert np.allclose(_mapped_columns(built, u), expected * u, atol=1e-10)


def test_two_coincident_certified_blocks_share_one_damping_and_the_flipped_block_keeps_its_own():
    """Two stable blocks of one eigenvalue build without a uniform fallback, the flipped eigenvector at its own -p."""
    theta, y, form = _schur_ritz(_FALLBACK_FORM)
    lam_pi, flip, q_kept = 1.0 - theta, (1.0 - theta).real < -_MARGIN, np.eye(8, 3)
    built, applied = _nonuniform_map(q_kept, form, theta, lam_pi, flip, np.ones(3, dtype=bool))
    expected = _direction_damping(lam_pi, flip)
    assert np.allclose(_mapped_columns(built, q_kept @ y), expected * (q_kept @ y), atol=1e-10)
    assert np.allclose(applied, expected, atol=1e-12)


def test_nonuniform_map_refuses_near_degenerate_blocks_of_opposite_sign():
    """Two certified blocks 2e-4 apart with opposite signs carry no bounded signed map and are refused."""
    log = MagicMock()
    q_kept = np.linalg.qr(np.random.default_rng(3).standard_normal((10, 2)))[0]
    assert _schur_map(np.array([[1.0101, 0.5], [0.0, 1.0099]]), q_kept, [True, True], log)[2] is None
    assert any("no per-direction damping used" in call[0][0] for call in log.call_args_list)


def test_the_certified_map_never_amplifies_a_residual_beyond_the_damping_it_encodes():
    """The grouped fallback of a coupled span stays inside the magnitude cap it was chosen for, and is logged."""
    log = MagicMock()
    tracker = JacobianTracker(0.4)
    tracker._nonuniform, applied = _schur_map(_COUPLED_FORM, np.eye(8, 4), [True] * 4, log)[2]
    rng = np.random.default_rng(4)
    residual = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    assert any("uniform damping" in call[0][0] for call in log.call_args_list)
    assert _map_gain(tracker, residual) <= _MAP_NORM_CAP * np.max(np.abs(applied))


def test_the_certified_map_refuses_a_flipped_mode_a_uniform_damping_would_leave_growing():
    """A flipped mode at -0.5 leaning on two stable ones under a coupling of three is left to the mixing's step."""
    log = MagicMock()
    upper = np.array([[0.7, 3.0, 3.0], [0.0, 0.6, 3.0], [0.0, 0.0, 1.5]])
    assert _schur_map(upper, np.eye(12, 3), [True] * 3, log)[2] is None
    assert any("no per-direction damping used" in call[0][0] for call in log.call_args_list)


def test_predicted_rate_reports_the_damping_of_each_coupled_group():
    """On a grouped fallback every certified direction is reported at the signed damping the map applies to it."""
    tracker = JacobianTracker(0.4)
    lam_pi, flip, _, tracker._nonuniform, tracker._applied = _coupled_fallback()
    tracker.last_lam_pi, tracker.last_gated = lam_pi, np.ones(4, dtype=bool)
    damping = _direction_damping(lam_pi, flip)
    expected = np.max(np.abs(1.0 - np.where(flip, damping, np.min(damping[~flip])) * lam_pi))
    assert np.isclose(tracker.predicted_rate(), expected, atol=1e-12)


@pytest.mark.parametrize("angle", [1e-1, 1e-2, 3e-3, 1.1e-3])
def test_the_carried_map_refuses_two_nearly_parallel_columns_of_opposite_sign(angle):
    """Two nearly parallel carried columns of opposite sign carry no bounded signed map and are refused."""
    log = MagicMock()
    lam_pi, flip = np.array([-4.0 + 0.0j, 0.5 + 0.0j]), np.array([True, False])
    u = np.zeros((8, 2), dtype=np.complex128)
    u[0, 0], u[0, 1], u[1, 1] = 1.0, np.cos(angle), np.sin(angle)
    assert _carried_nonuniform_map(lam_pi, u, flip, log) is _SIGNS_UNBOUNDED
    assert any("no per-direction damping used" in call[0][0] for call in log.call_args_list)


@pytest.mark.parametrize("angle", [1e-1, 1e-2])
def test_the_carried_map_damps_two_nearly_parallel_stable_columns_uniformly(angle):
    """Two nearly parallel carried columns of equal sign take one uniform damping on each of their own directions."""
    log = MagicMock()
    lam_pi, flip = np.array([4.0 + 0.0j, 0.5 + 0.0j]), np.zeros(2, dtype=bool)
    u = np.zeros((8, 2), dtype=np.complex128)
    u[0, 0], u[0, 1], u[1, 1] = 1.0, np.cos(angle), np.sin(angle)
    tracker = JacobianTracker(0.4)
    tracker._nonuniform = _carried_nonuniform_map(lam_pi, u, flip, log)
    p_min = np.min(np.abs(_direction_damping(lam_pi, flip)))
    assert any("uniform damping" in call[0][0] for call in log.call_args_list)
    assert np.allclose(_mapped_columns(tracker._nonuniform, u), p_min * u, atol=1e-8)
    assert _map_gain(tracker, np.arange(4) + 1.0j) <= _MAP_NORM_CAP * p_min


def test_a_carried_set_refused_a_per_direction_map_still_installs_its_reflector(monkeypatch):
    """A carried set whose signs are not separable keeps its orthogonal reflector and the scalar damping bound."""
    tracker, logger = _logged_tracker()
    monkeypatch.setattr(jacobian_stabilization, "_carried_nonuniform_map", lambda *args, **kwargs: _SIGNS_UNBOUNDED)
    assert tracker._install_carried(np.array([-4.0 + 0.0j]), _real_column(0, 2))
    assert tracker.active and tracker._nonuniform is None and tracker.q.shape[1] == 1
    assert any("on its reflector alone" in call[0][0] for call in logger.info.call_args_list)


def test_a_carried_set_the_reflector_refuses_installs_nothing_although_the_map_builds():
    """Columns the reflector's condition measure refuses take the whole carried set down, map or no map."""
    tracker = JacobianTracker(0.4)
    u = np.zeros((6, 2), dtype=np.complex128)
    u[0, 0], u[0, 1], u[1, 1] = 1.0, np.cos(1.4e-3), np.sin(1.4e-3)
    assert _carried_nonuniform_map(np.array([-0.5 + 0.0j, -0.4 + 0.0j]), u, np.ones(2, dtype=bool)) is not None
    assert not tracker._install_carried(np.array([-0.5 + 0.0j, -0.4 + 0.0j]), u) and not tracker.active


def test_a_carried_set_dropping_a_column_installs_nothing_at_all():
    """Every carried column is flipped, so one beyond the condition cap refuses the whole set, reflector included."""
    tracker, logger = _logged_tracker()
    lam_pi = np.array([-0.5 + 0.0j, -0.6 + 0.0j])
    u = np.zeros((8, 2), dtype=np.complex128)
    u[0, 0], u[0, 1], u[1, 1] = 1.0, 1.0, 1e-9
    assert tracker._install_carried(lam_pi, u) is False
    assert not tracker.active and tracker.q is None and tracker._nonuniform is None
    assert any("nothing installed" in call[0][0] for call in logger.info.call_args_list)


def test_a_carried_map_drops_a_dependent_stable_column_and_logs_it():
    """A stable carried column beyond the condition cap is dropped, logged, and the flipped one still installs."""
    log = MagicMock()
    lam_pi = np.array([-0.5 + 0.0j, 0.05 + 0.0j])
    u = np.zeros((8, 2), dtype=np.complex128)
    u[0, 0], u[0, 1], u[1, 1] = 1.0, 1.0, 1e-9
    span, dual, weighting = _carried_nonuniform_map(lam_pi, u, np.array([True, False]), log)
    assert span.shape == (8, 1) and np.allclose(dual @ span, np.eye(1), atol=1e-12)
    assert np.allclose(weighting, [[-1.0]], atol=1e-12)
    assert "1 stable carried column(s) of lambda_Pi +0.0500" in log.call_args[0][0]
    assert log.call_args[0][0].endswith("are dropped.")


def test_nonuniform_map_refuses_a_certified_block_it_cannot_separate_from_an_uncertified_one():
    """A gated and an ungated eigenvalue 1e-9 apart leave the certified span unseparable, so no map is installed."""
    log = MagicMock()
    assert _schur_map(np.array([[4.0, 0.5], [0.0, 4.0 + 1e-9]]), np.eye(6, 2), [True, False], log)[2] is None
    assert "not separable from the uncertified one" in log.call_args[0][0]


@pytest.mark.parametrize("separation", [0.0, 1e-13, 1e-11, 5e-10])
def test_nonuniform_map_ties_two_certified_blocks_closer_than_the_tolerance(separation):
    """Two certified blocks closer than the block separation tolerance share one damping in a finite map."""
    upper = np.array([[5.0, 0.2, 0.1], [0.0, 0.3, 0.5], [0.0, 0.0, 0.3 + separation]])
    lam_pi, flip, ((span, _, damping_map), applied) = _schur_map(upper, np.eye(8, 3), [True] * 3)
    assert separation <= _SCHUR_TOL * 5.0 and span.shape == (8, 3) and np.all(np.isfinite(damping_map))
    assert np.allclose(applied[~flip], np.min(_direction_damping(lam_pi, flip)[~flip]), atol=1e-12)
    assert np.allclose(applied[flip], _direction_damping(lam_pi, flip)[flip], atol=1e-12)


@pytest.mark.parametrize("separation", [0.0, 1e-13, 1e-11, 5e-10])
def test_nonuniform_map_refuses_a_certified_block_an_uncertified_one_coincides_with(separation):
    """A certified and an uncertified eigenvalue that coincide refuse the map without raising."""
    log = MagicMock()
    upper = np.array([[5.0, 0.2, 0.1], [0.0, 0.3, 0.5], [0.0, 0.0, 0.3 + separation]])
    assert _schur_map(upper, np.eye(8, 3), [True, True, False], log)[2] is None
    assert "not separable from the uncertified one" in log.call_args[0][0]


def test_reflector_and_map_survive_a_nearly_defective_flipped_pair():
    """Two stiff modes 1e-2 apart coupled by 1e3 are reflected exactly and the map spans all four certified ones."""
    rng = np.random.default_rng(8)
    orth = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    upper = np.diag([4.0, 4.0 - 1e-2, 0.7, 0.4])
    upper[0, 1] = 1e3
    q_kept = np.linalg.qr(rng.standard_normal((10, 4)))[0]
    theta, y, form = _schur_ritz(orth @ upper @ orth.T)
    flip, u = theta.real > 2.0, q_kept @ y
    q, w = JacobianTracker(0.2)._flipped_reflector(flip, q_kept, form)
    reflect = lambda v: v - 2.0 * (q @ (w @ v))
    assert np.allclose(reflect(u[:, flip]), -u[:, flip], atol=1e-6)
    assert np.allclose(reflect(u[:, ~flip]), u[:, ~flip], atol=1e-6)
    (span, _, _), applied = _nonuniform_map(q_kept, form, theta, 1.0 - theta, flip, np.ones(4, dtype=bool))
    assert np.linalg.matrix_rank(span) == 4 and np.all(np.isfinite(applied))


def test_a_conjugate_pair_is_located_on_its_schur_block_and_flipped_as_one():
    """Both members of a complex pair take the two positions of one 2 x 2 block and the map flips both of them."""
    rng = np.random.default_rng(17)
    q_kept = np.linalg.qr(rng.standard_normal((10, 4)))[0]
    orth = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    block = np.array([[0.9, -1.2, 0.3, 0.1], [1.2, 0.9, 0.0, 0.2], [0.0, 0.0, 0.3, 0.05], [0.0, 0.0, 0.0, 0.5]])
    theta, y, form = _schur_ritz(orth @ block @ orth.T)
    pair = np.abs(theta.imag) > 1.0
    low = int(form.positions[pair].min())
    assert sorted(form.positions[pair]) == [low, low + 1] and form.t[low + 1, low] != 0.0
    (span, dual, damping), applied = _nonuniform_map(q_kept, form, theta, 1.0 - theta, pair, np.ones(4, dtype=bool))
    u = q_kept @ y[:, pair]
    assert applied[pair][0] < 0.0 and np.allclose(applied[pair], applied[pair][0], atol=1e-12)
    assert np.allclose(span @ (damping @ (dual @ u)), applied[pair][0] * u, atol=1e-10)


def test_predicted_rate_takes_the_mapped_directions_own_damping():
    """A mapped certified direction contributes its own contraction factor and an unmapped one the step's bound."""
    tracker = JacobianTracker(0.4)
    tracker.p_eff, tracker.last_lam_pi = 0.025, np.array([0.5 + 0.0j, -4.0 + 0.0j])
    tracker.last_gated, tracker._last_u = np.ones(2, dtype=bool), np.eye(4, 2).astype(np.complex128)
    tracker._nonuniform = (np.eye(4, 2), np.eye(4, 2).T, np.diag([1.0, -0.25]))
    # a carried map records no applied damping, so both directions are read off its Rayleigh quotient
    assert np.isclose(tracker.predicted_rate(), 0.5, atol=1e-12)
    tracker._applied = np.array([0.5, -0.25])
    assert np.isclose(tracker.predicted_rate(), 0.75, atol=1e-12)
    tracker._applied = np.array([0.5, np.nan])
    assert np.isclose(tracker.predicted_rate(), 1.1, atol=1e-12)
    tracker._applied, tracker._nonuniform = None, None
    assert np.isclose(tracker.predicted_rate(), 1.1, atol=1e-12)


def test_held_flipped_uses_the_cosine_with_the_flipped_basis_not_the_oblique_dual_rows():
    """A direction with 3 degrees of overlap with the flipped subspace is not held even along steep oblique rows."""
    tracker = JacobianTracker(0.4)
    tracker.q, tracker.w = np.eye(4, 1), np.array([[1.0, 5.0, 0.0, 0.0]])
    inside = np.eye(4, 1).astype(np.complex128)
    outside = np.array([[np.sin(np.radians(3.0))], [np.cos(np.radians(3.0))], [0.0], [0.0]], dtype=np.complex128)
    assert tracker._held_flipped(inside)[0] and not tracker._held_flipped(outside)[0]


def test_minimum_iterations_follows_the_certified_margin():
    """The minimum count is the capped ceil(c / (eps p)) over certified real parts or their bands, zero with none."""
    tracker = JacobianTracker(0.4)
    assert tracker.minimum_iterations() == 0 and tracker.certified_margin == 0.0
    tracker.p_eff, tracker.last_lam_pi = 0.2, np.array([0.25 + 0.0j, 4.0 + 0.0j, -0.05 + 0.0j])
    tracker.last_res, tracker.last_gated = np.zeros(3), np.array([True, True, False])
    assert np.isclose(tracker.certified_margin, 0.25, atol=1e-12) and tracker.minimum_iterations() == 10
    tracker.last_lam_pi = np.array([0.011 + 0.0j, 4.0 + 0.0j, -0.05 + 0.0j])
    assert tracker.minimum_iterations() == 228
    tracker.last_lam_pi = np.array([1e-9 + 0.0j, 4.0 + 0.0j, -0.05 + 0.0j])
    assert tracker.minimum_iterations() == 250
    tracker.last_gated = np.zeros(3, dtype=bool)
    assert tracker.minimum_iterations() == 0
    tracker.last_lam_pi, tracker.last_gated = np.array([4.0j]), np.ones(1, dtype=bool)
    tracker.last_res = np.zeros(1)
    assert tracker.minimum_iterations() == 250


def test_minimum_iterations_does_not_round_a_boundary_count_up():
    """A count that lands on an integer is not raised by the last bits of the eigenvalue it came from."""
    tracker = JacobianTracker(0.4)
    tracker.p_eff, tracker.last_lam_pi = 0.2, np.array([0.25 - 1e-13 + 0.0j])
    tracker.last_res, tracker.last_gated = np.zeros(1), np.ones(1, dtype=bool)
    assert tracker.minimum_iterations() == 10


def test_minimum_iterations_is_capped():
    """A mapped pseudo-divergence pair just outside the band drives the count to the cap, never above it."""
    tracker = JacobianTracker(0.4)
    pair = np.array([[1.0 - 1.01e-2, 3.0], [-3.0, 1.0 - 1.01e-2]])
    lam_pi, flip, (tracker._nonuniform, tracker._applied) = _schur_map(pair, np.eye(6, 2), [True, True])
    tracker.last_lam_pi, tracker.last_res, tracker.last_gated = lam_pi, np.zeros(2), np.ones(2, dtype=bool)
    assert np.allclose(tracker._applied, _direction_damping(lam_pi, flip), atol=1e-12)
    assert tracker.minimum_iterations() == _MIN_ITERATION_CAP


def test_minimum_iterations_counts_a_mapped_direction_at_its_own_damping():
    """With the map installed a flat direction counts at its own undamped damping, without it at the scalar step."""
    tracker, _ = _stable_only_tracker(1)
    assert np.isclose(tracker.p_eff, 0.025, atol=1e-12) and tracker.minimum_iterations() == 2
    tracker._nonuniform = tracker._applied = None
    assert tracker.minimum_iterations() == 40


def test_minimum_iterations_reads_the_rayleigh_quotient_of_a_carried_map():
    """A direction a carried map acts on counts at the damping the map gives it, not at the scalar step."""
    tracker = JacobianTracker(0.4)
    _carry(tracker, -0.05, _real_column(0, 2), 2)
    assert tracker.active and tracker._applied is None and tracker.minimum_iterations() == 20
    tracker._nonuniform = None
    assert tracker.minimum_iterations() == 25


def test_certified_margin_counts_a_marginal_mode_at_its_undecidable_band():
    """A mode inside max(_MARGIN, res) sets the margin and the whole count with that band, not the resolved count."""
    tracker = JacobianTracker(0.4)
    tracker.last_lam_pi = np.array([4e-3 + 0.0j, 0.25 + 0.0j, 0.02 + 0.0j])
    tracker.last_res, tracker.last_gated = np.array([1e-3, 1e-3, 5e-2]), np.ones(3, dtype=bool)
    assert np.isclose(tracker.certified_margin, _MARGIN, atol=1e-12)
    assert tracker.minimum_iterations() == 125 and tracker.minimum_iterations(outside_band=True) == 5
    tracker.last_lam_pi = np.array([4e-3 + 0.0j])
    tracker.last_res, tracker.last_gated = np.array([1e-3]), np.ones(1, dtype=bool)
    assert np.isclose(tracker.certified_margin, _MARGIN, atol=1e-12) and tracker.minimum_iterations() == 125
    assert tracker.minimum_iterations(outside_band=True) == 0


def test_tracker_converges_paper_scalar_map():
    """The tracked scalar map settles on the physical fixed point 0 while plain damping runs into mu."""
    tracker = JacobianTracker(0.2)
    tracked = tracked_run(scalar_map, [0.3, 0.15], 0.2, 200, tracker)
    plain = tracked_run(scalar_map, [0.3, 0.15], 0.2, 200, None)
    assert tracker.active and np.allclose(tracked, 0.0, atol=1e-8)
    assert np.allclose(plain, MU, atol=1e-8)


def test_tracker_converges_paper_2d_map():
    """The tracked two-dimensional map converges to the fixed point that plain damping diverges away from."""
    tracker = JacobianTracker(0.2)
    tracked = tracked_run(two_d_map, [0.0, 0.0], 0.2, 400, tracker)
    plain = tracked_run(two_d_map, [0.0, 0.0], 0.2, 200, None)
    assert np.allclose(tracked, np.linalg.solve(np.eye(2) - A_2D, B_2D), atol=1e-2)
    assert np.linalg.norm(plain) > 1e3
    assert tracker.q.shape[1] == 1 and tracker.p_eff <= 0.2 and tracker.predicted_rate() < 1.0


def test_the_first_certified_unstable_estimate_installs_the_flip():
    """The first flip lands on the first update that certifies the unstable mode, with no streak to wait out."""
    tracker = JacobianTracker(0.2)
    iterates, proposals = linear_pairs(A_2D, B_2D, 0.2, 9, [0.0, 0.0])
    assert tracker.update(iterates[:3], proposals[:3]) is True
    assert tracker.n_updates == 1 and tracker.q is not None and tracker._flip.sum() == 1


def test_tracker_counters_reset_on_interruption():
    """An interrupted calm streak restarts from zero instead of carrying over stale progress."""
    tracker = JacobianTracker(0.4)
    unstable, stable, shift = np.diag([5.0, 0.7]), np.diag([0.2, 0.45]), np.array([1.0, 1.0])
    # the windows start ever closer to the fixed point, so the growth interlock never enters
    x0 = iter(np.arange(4.0, 0.1, -0.1))
    tracker.update(*linear_pairs(unstable, shift, 0.3, 3, [next(x0), next(x0)]))
    assert tracker.q is not None
    for mat in [stable] * (_PERSIST - 1) + [unstable] + [stable] * (_PERSIST - 1):
        tracker.update(*linear_pairs(mat, shift, 0.3, 3, [next(x0), next(x0)]))
    assert tracker.q is not None
    tracker.update(*linear_pairs(stable, shift, 0.3, 3, [next(x0), next(x0)]))
    assert tracker.q is None
    tracker.update(*linear_pairs(unstable, shift, 0.3, 3, [next(x0), next(x0)]))
    assert tracker.q is not None


def test_a_raised_flip_streak_delays_the_flip_and_restarts_when_interrupted(monkeypatch):
    """At _FLIP_PERSIST = 3 a flip waits for three consecutive certifications, counted afresh after an interruption."""
    monkeypatch.setattr(jacobian_stabilization, "_FLIP_PERSIST", 3)
    tracker = JacobianTracker(0.4)
    unstable, stable, shift = np.diag([5.0, 0.7]), np.diag([0.2, 0.45]), np.array([1.0, 1.0])
    # the windows start ever closer to the fixed point, so the growth interlock never enters
    x0 = iter(np.arange(4.0, 0.1, -0.1))
    installed = []
    for mat in [unstable] * 2 + [stable] + [unstable] * 3:
        tracker.update(*linear_pairs(mat, shift, 0.3, 3, [next(x0), next(x0)]))
        installed.append(tracker.q is not None)
    assert installed == [False] * 5 + [True]


@pytest.mark.parametrize("zero_norm", [False, True], ids=["step_floor", "zero_norm_iterate"])
def test_tracker_freezes_state_below_the_floor(zero_norm):
    """A relative step below the single-precision floor, or a zero-norm last iterate, leaves the tracker untouched."""
    tracker = JacobianTracker(0.4)
    tracker.q = q_before = np.eye(4 if zero_norm else 8)[:, :1]
    if zero_norm:
        iterates = [np.zeros((1, 2), dtype=np.complex128) for _ in range(4)]
        proposals = [np.full((1, 2), 0.1 * k, dtype=np.complex128) for k in range(4)]
    else:
        base = np.ones((2, 2), dtype=np.complex64)
        iterates = [(base * (1.0 + 1e-5 * k)).astype(np.complex64) for k in range(4)]
        proposals = [(base * (1.0 + 1e-5 * (k + 1))).astype(np.complex64) for k in range(4)]
    assert tracker.update(iterates, proposals) is False
    assert tracker.q is q_before and np.allclose(tracker.p_eff, 0.4, atol=1e-12) and tracker.n_updates == 0


def test_tracker_rank_one_window_certifies_the_dominant_mode():
    """A window whose stable directions decayed into single-precision noise still certifies the growing mode."""
    tracker = JacobianTracker(0.4)
    iterates, proposals = single_precision_history(100)
    # by iteration 100 the fivefold stable rate 0.5 has decayed far below the noise floor, leaving only lambda_J = 1.3
    assert tracker.update(iterates[-7:], proposals[-7:]) is True
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
    assert tracker.q is None and events == [False] * (_PERSIST - 1) + [True]


def test_tracker_allow_flip_false_only_monitors():
    """With allow_flip disabled the spectrum is still measured but nothing is flipped and the damping is held."""
    tracker = JacobianTracker(0.2)
    feed_windows(tracker, linear_pairs(A_2D, B_2D, 0.2, 9, [0.0, 0.0]), allow_flip=False)
    assert tracker.last_lam_pi is not None and np.any(tracker.last_lam_pi.real < 0.0)
    assert tracker.q is None and np.allclose(tracker.p_eff, 0.2, atol=1e-12) and not tracker.active


def test_tracker_allow_flip_false_releases_installed_basis():
    """Suppressing flips releases a previously installed basis at once, and a further call finds nothing to do."""
    tracker = JacobianTracker(0.2)
    tracker.q = np.linalg.qr(np.random.default_rng(2).standard_normal((4, 1)))[0]
    iterates, proposals = linear_pairs(A_2D, B_2D, 0.2, 3, [0.0, 0.0])
    assert tracker.update(iterates, proposals, allow_flip=False) is True and tracker.q is None
    assert tracker.update(iterates, proposals, allow_flip=False) is False


def test_tracker_allow_flip_false_releases_a_map_installed_without_a_reflector():
    """Suppressing flips releases an installed per-direction map at once, which switches no reflected map."""
    tracker, logger = _logged_tracker()
    iterates, proposals = linear_pairs(_STABLE_ONLY_GAINS, np.zeros(3), 0.01, 3, [1.0, 1.0, 1.0])
    tracker.update(iterates, proposals)
    assert tracker.q is None and tracker._nonuniform is not None and tracker.active
    assert tracker.update(iterates, proposals, allow_flip=False) is False
    assert not tracker.active and tracker._nonuniform is None and tracker._applied is None
    assert "per-direction damping released" in logger.info.call_args[0][0]


def test_tracker_p_eff_only_decreases():
    """A binding mode lowers the damping, and a later harmless spectrum never raises it again."""
    tracker = JacobianTracker(0.5)
    feed_windows(tracker, linear_pairs(np.diag([4.0, 0.3]), np.array([1.0, 1.0]), 0.2, 8, [0.1, 0.1]))
    lowered = tracker.p_eff
    feed_windows(tracker, linear_pairs(np.diag([0.5, 0.4]), np.array([1.0, 1.0]), 0.5, 8, [0.9, -0.6]))
    assert np.allclose(lowered, 0.5 * 2.0 * 3.0 / 9.0, atol=1e-12) and np.allclose(tracker.p_eff, lowered, atol=1e-12)


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
    tracker, logger = _logged_tracker()
    tracker.q, tracker._last_residual_norm = np.eye(4)[:, :1], 0.0
    assert _grow_residual(tracker, _PERSIST, (0.6, 0.5), 1.0)[-1] and tracker.q is None
    assert "residual grew" in logger.info.call_args[0][0]


def test_growing_residual_releases_a_basis_while_the_flip_still_certifies():
    """A residual growing on _PERSIST consecutive updates releases the basis while a certified flip persists."""
    tracker, logger = _logged_tracker()
    tracker.q, tracker._last_residual_norm = np.eye(4)[:, :1], 0.0
    event = _grow_residual(tracker, _PERSIST, (2.0, 0.5), 1.0)[-1]
    assert tracker._flip.any() and event and tracker.q is None and tracker._persist == 0
    assert "residual grew" in logger.info.call_args[0][0]


def test_release_by_growth_needs_fresh_certification_to_reinstall():
    """The certifying update right after a growth release only restarts the persistence streak."""
    tracker = JacobianTracker(0.4)
    tracker.q, tracker._last_residual_norm = np.eye(4)[:, :1], 0.0
    _grow_residual(tracker, _PERSIST + 1, (2.0, 0.5), 1.0)
    assert tracker.q is None and tracker._flip.any() and tracker._persist == 1


def test_a_certified_flip_waits_out_the_hold_a_growth_release_left_behind():
    """After a growth release a certified flip is installed again only once the hold of _PERSIST updates is over."""
    tracker = JacobianTracker(0.4)
    tracker.q, tracker._last_residual_norm = np.eye(4)[:, :1], 0.0
    _grow_residual(tracker, _PERSIST, (2.0, 0.5), 1.0)
    assert tracker.q is None and tracker._hold == _PERSIST
    installs = []
    for k in range(_PERSIST + 1):
        # restarted ever closer to the origin, so the residual shrinks and no second growth release enters
        pairs = linear_pairs(np.diag([2.0, 0.5]), np.array([1.0, 1.0]), 0.3, 3, [0.5 / (k + 1), 0.5 / (k + 1)])
        tracker.update(*pairs)
        installs.append(tracker.q is not None)
    assert tracker._flip.any() and installs == [False] * _PERSIST + [True]


def test_tracker_warns_once_at_floor():
    """The floor warning fires once although the damping stays pinned, naming the bound and the per-direction map."""
    tracker, logger = _logged_tracker()
    feed_windows(tracker, linear_pairs(np.diag([201.0, 0.5]), np.array([1.0, 1.0]), 0.1, 9, [0.05, 0.05]))
    assert logger.warning.call_count == 1 and "floor" in logger.warning.call_args[0][0]
    message = logger.warning.call_args[0][0]
    assert "every damped step runs at" in message and "installed per-direction map carries it" in message


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


def test_leading_returns_the_largest_modes_padded_with_nan():
    """leading sorts a planted spectrum by descending modulus and pads it with nan; a fresh tracker returns all nan."""
    tracker = JacobianTracker(0.3)
    tracker.last_lam_pi, tracker.last_res = np.array([0.5 + 0j, 2.0 + 1j]), np.array([1e-3, 2e-3])
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
    """One full update emits one lambda_Pi line, preceded by the map's install line only where no flip lands."""
    tracker, logger = _logged_tracker(0.2)
    tracker.update(*linear_pairs(A_2D, B_2D, 0.2, 3, [0.0, 0.0]))
    lines = [call[0][0] for call in logger.info.call_args_list]
    assert ["lambda_Pi" in line for line in lines] == [True] and "; 1 flipped" in lines[0]
    logger.reset_mock()
    JacobianTracker(0.4, logger=logger).update(*linear_pairs(_STABLE_ONLY_GAINS, np.zeros(3), 0.01, 4, [1.0] * 3))
    lines = [call[0][0] for call in logger.info.call_args_list]
    assert ["lambda_Pi" in line for line in lines] == [False, True]
    assert "per-direction damping installed on" in lines[0] and "; 0 flipped" in lines[1]


def test_spectrum_line_tags_every_certificate_and_decision():
    """A mode is tagged certified stable, marginal, flip or unstable-suppressed, or uncertified, by its state."""
    tracker, logger = _logged_tracker()
    tracker.last_lam_pi = np.array([0.5, 0.005, -0.3, -0.2], dtype=np.complex128)
    tracker.last_res, tracker.last_gated = np.full(4, 1e-3), np.array([True, True, True, True])
    tracker._log_spectrum(np.array([False, False, True, False]))
    tracker.last_gated[0] = False
    tracker._log_spectrum(np.zeros(4, dtype=bool))
    first, second = (call[0][0] for call in logger.info.call_args_list)
    assert "lambda_Pi=+0.5000+0.0000j (res 1.0e-03, certified, stable)" in first
    assert "lambda_Pi=+0.0050+0.0000j (res 1.0e-03, certified, marginal)" in first
    assert "lambda_Pi=-0.3000+0.0000j (res 1.0e-03, certified, flip)" in first
    assert "lambda_Pi=-0.2000+0.0000j (res 1.0e-03, certified, unstable, suppressed)" in first
    assert "; 1 flipped, p_eff=0.4000" in first
    assert "lambda_Pi=+0.5000+0.0000j (res 1.0e-03, uncertified)" in second and "; 0 flipped" in second


@pytest.mark.parametrize("alpha, expected", [(0.3, True), (0.02, False)])
def test_tracker_replaces_basis_on_subspace_rotation(alpha, expected):
    """The tracked basis is replaced only when the new unstable direction is further away than _SUBSPACE_ANGLE."""
    tracker = JacobianTracker(0.2)
    feed_windows(tracker, linear_pairs(rotated_map(0.0), np.array([1.0, 1.0]), 0.2, 9, [0.1, -0.2]), _FLIP_PERSIST)
    q_before = tracker.q
    assert tracker.active and (alpha > _SUBSPACE_ANGLE) is expected
    rotated_pairs = linear_pairs(rotated_map(alpha), np.array([1.0, 1.0]), 0.2, 9, [0.1, -0.2])
    assert tracker.update(*rotated_pairs) is expected and (tracker.q is q_before) is not expected
    overlap = np.clip(abs(q_before[:, 0] @ tracker.q[:, 0]), -1.0, 1.0)
    assert np.allclose(np.arccos(overlap), alpha if expected else 0.0, atol=1e-6)


def test_reflector_fixes_certified_stable_directions_of_a_non_normal_map():
    """The reflector flips the certified unstable direction and leaves an overlapping certified stable one fixed."""
    mat, shift = np.array([[4.0, 5.0], [0.0, -3.6]]), np.array([1.0, 1.0])
    u_f, u_s = np.array([1.0, 0.0, 0.0, 0.0]), np.array([5.0, -7.6, 0.0, 0.0]) / np.hypot(5.0, 7.6)
    assert abs(u_f @ u_s) > _FLIP_OVERLAP
    tracker = JacobianTracker(0.2)
    feed_windows(tracker, linear_pairs(mat, shift, 0.2, 9, [0.01, 0.01]), _PERSIST)
    assert tracker.active and tracker.w is not None and tracker.q.shape[1] == 1
    reflect = lambda v: v - 2.0 * (tracker.q @ (tracker.w @ v))
    assert np.allclose(reflect(u_f), -u_f, atol=1e-8) and np.allclose(reflect(u_s), u_s, atol=1e-8)
    assert np.allclose(reflect(reflect(u_s + 0.3 * u_f)), u_s + 0.3 * u_f, atol=1e-8)
    assert np.allclose(tracker.w @ u_s, 0.0, atol=1e-8)
    step = lambda x: (mat @ x.real + shift).astype(np.complex128)
    converged = tracked_run(step, [0.01, 0.01], 0.2, 80, JacobianTracker(0.2))
    assert np.allclose(converged, np.linalg.solve(np.eye(2) - mat, shift), atol=1e-6)


def _projected_map(seed: int, diagonal: list[float], coupling: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns an orthonormal Ritz basis on a ten-dimensional space and a projected map with the given eigenvalues."""
    rng = np.random.default_rng(seed)
    q_kept = np.linalg.qr(rng.standard_normal((10, len(diagonal))))[0]
    orth = np.linalg.qr(rng.standard_normal((len(diagonal), len(diagonal))))[0]
    upper = np.triu(rng.standard_normal((len(diagonal), len(diagonal))), 1) * coupling
    return q_kept, orth @ (np.diag(diagonal) + upper) @ orth.T


def test_reflector_of_a_normal_map_is_the_orthogonal_reflection():
    """On a symmetric projected map the Schur reflector is the orthogonal reflection of the flipped eigendirections."""
    q_kept, b_proj = _projected_map(5, [4.0, -3.6, 0.7, 0.4], 0.0)
    theta, y, form = _schur_ritz(b_proj)
    flip = theta.real > 2.0
    u = q_kept @ y
    q, w = JacobianTracker(0.2)._flipped_reflector(flip, q_kept, form)
    reflect = lambda v: v - 2.0 * (q @ (w @ v))
    assert np.allclose(w, q.T, atol=1e-10) and np.allclose(reflect(u[:, flip]), -u[:, flip], atol=1e-10)
    assert np.allclose(reflect(u[:, ~flip]), u[:, ~flip], atol=1e-10)
    assert np.allclose(reflect(reflect(u)), u, atol=1e-10)


def test_reflector_projector_commutes_with_the_projected_jacobian():
    """The projector the reflector carries is idempotent and commutes with the projected Jacobian it was built from."""
    q_kept, b_proj = _projected_map(11, [4.0, -3.6, 0.7, 0.4], 0.5)
    theta, _, form = _schur_ritz(b_proj)
    q, w = JacobianTracker(0.2)._flipped_reflector(theta.real > 2.0, q_kept, form)
    projector = q_kept.T @ q @ w @ q_kept
    assert np.allclose(projector @ projector, projector, atol=1e-10)
    assert np.allclose(projector @ b_proj, b_proj @ projector, atol=1e-10)


def test_reflector_flips_a_conjugate_pair_as_one_real_block():
    """A flipped complex-conjugate pair gives a real two-column basis that reverses both members and squares to one."""
    rng = np.random.default_rng(17)
    q_kept = np.linalg.qr(rng.standard_normal((10, 4)))[0]
    orth = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    block = np.array([[0.9, -1.2, 0.3, 0.1], [1.2, 0.9, 0.0, 0.2], [0.0, 0.0, 0.3, 0.05], [0.0, 0.0, 0.0, 0.5]])
    theta, y, form = _schur_ritz(orth @ block @ orth.T)
    flip = np.abs(theta) > 1.0
    q, w = JacobianTracker(0.2)._flipped_reflector(flip, q_kept, form)
    u = q_kept @ y
    reflect = lambda v: v - 2.0 * (q @ (w @ v))
    assert q.shape == (10, 2) and np.isrealobj(q) and np.isrealobj(w)
    assert np.allclose(reflect(u[:, flip]), -u[:, flip], atol=1e-10) and np.allclose(reflect(reflect(u)), u, atol=1e-10)


def test_reflector_falls_back_to_the_orthogonal_reflection_on_a_nearly_parallel_complement():
    """A flipped subspace closer to its complement than the obliquity cap is reflected orthogonally."""
    logger = MagicMock()
    theta, _, form = _schur_ritz(np.array([[4.0, 100.0], [0.0, 0.7]]))
    q, w = JacobianTracker(0.2, logger=logger)._flipped_reflector(theta.real > 2.0, np.eye(8, 2), form)
    assert np.allclose(w, q.T, atol=1e-12)
    assert any("orthogonal reflection" in call[0][0] for call in logger.info.call_args_list)


def test_reflector_refuses_a_flipped_subspace_it_cannot_separate():
    """Two eigenvalues 1e-9 apart with one of them flipped blow up the Sylvester solution, so no reflector is built."""
    tracker, logger = _logged_tracker(0.2)
    theta, _, form = _schur_ritz(np.array([[4.0, 0.5], [0.0, 4.0 + 1e-9]]))
    assert tracker._flipped_reflector(np.array([False, True]), np.eye(6, 2), form) is None
    assert "Sylvester solution norm" in logger.info.call_args_list[-2][0][0]
    assert "not separable from its complement" in logger.info.call_args[0][0]


def test_reflector_refuses_a_schur_form_lapack_cannot_reorder(monkeypatch):
    """A reordering LAPACK itself refuses (two blocks too close to swap) leaves the flipped subspace inseparable."""
    tracker, logger = _logged_tracker(0.2)
    q_kept, b_proj = _projected_map(29, [4.0, -3.6, 0.7, 0.4], 0.3)
    theta, _, form = _schur_ritz(b_proj)
    refused = (form.t, form.z, np.diag(form.t), np.zeros(4), 2, 0.0, 0.0, 1)
    monkeypatch.setattr("dgamore.jacobian_stabilization.dtrsen", MagicMock(return_value=refused))
    assert tracker._flipped_reflector(theta.real > 2.0, q_kept, form) is None
    assert "too close to swap" in logger.info.call_args_list[-2][0][0]
    assert "not separable from its complement" in logger.info.call_args[0][0]


def test_reflector_refuses_a_conjugate_pair_flipped_on_one_side_only():
    """A flip set holding one member of a complex pair cannot be reordered as a block and installs no reflector."""
    logger = MagicMock()
    rng = np.random.default_rng(17)
    orth = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    block = np.array([[0.9, -1.2, 0.3, 0.1], [1.2, 0.9, 0.0, 0.2], [0.0, 0.0, 0.3, 0.05], [0.0, 0.0, 0.0, 0.5]])
    theta, _, form = _schur_ritz(orth @ block @ orth.T)
    flip = theta.imag > 1.0
    assert JacobianTracker(0.2, logger=logger)._flipped_reflector(flip, np.eye(8, 4), form) is None
    assert "conjugate pair" in logger.info.call_args_list[-2][0][0]


def test_reflector_builds_nothing_without_a_resolved_flipped_eigenvalue():
    """Neither an empty flip set nor a Ritz value the Schur form does not locate produces a reflector."""
    logger = MagicMock()
    q_kept, b_proj = _projected_map(23, [4.0, -3.6, 0.7, 0.4], 0.3)
    theta, _, form = _schur_ritz(b_proj)
    tracker = JacobianTracker(0.2, logger=logger)
    assert tracker._flipped_reflector(np.zeros(4, dtype=bool), q_kept, form) is None
    unlocated = SchurForm(form.t, form.z, None)
    assert tracker._flipped_reflector(theta.real > 2.0, q_kept, unlocated) is None
    assert "could not be located" in logger.info.call_args_list[-2][0][0]


def test_install_carried_returns_false_for_a_lone_conjugate_partner():
    """A single Ritz value with a negative imaginary part collects no flipped column and installs nothing."""
    tracker = JacobianTracker(0.4)
    assert tracker._install_carried(np.array([-0.2 + 0.3j]), _real_column(0, 2)) is False
    assert tracker.q is None and tracker.w is None


def test_tracker_flips_a_complex_pair_end_to_end():
    """A planted unstable complex pair in a stable map is flipped and converges, while plain damping diverges."""
    block = np.zeros((6, 6))
    block[:2, :2] = 1.3 * np.array([[np.cos(0.05), -np.sin(0.05)], [np.sin(0.05), np.cos(0.05)]])
    block[2, 2] = 0.3
    step = lambda x: to_mat(block @ to_vec(x), x.shape)
    tracker = JacobianTracker(2.0)
    x0 = [0.1 + 0.0j, -0.05 + 0.0j, 0.2 + 0.0j]
    tracked = tracked_run(step, x0, 0.5, 150, tracker)
    plain = tracked_run(step, x0, 0.5, 80, None)
    assert tracker.active and tracker.q.shape[1] == 2 and np.allclose(tracked, 0.0, atol=1e-6)
    assert np.linalg.norm(plain) > 1e3


def test_spectrum_state_stores_every_vector_below_the_storage_band_but_flips_by_the_flip_band():
    """A vector is stored for every certified mode below +0.1; the flip decision is the separate -max(_MARGIN, res)."""
    tracker = JacobianTracker(0.3)
    lam_pi, res = np.array([-0.3 + 0.0j, -0.3 + 0.0j, 0.05 + 0.0j, 0.3 + 0.0j]), np.array([1e-6, 0.5, 1e-6, 1e-6])
    u = np.ones((4, 4), dtype=np.complex128)
    flip = lam_pi.real < -np.maximum(_MARGIN, res)
    tracker._last_certified = (lam_pi, res, np.ones(4, dtype=bool), flip, u, np.zeros(4, dtype=bool))
    state = tracker.spectrum_state((1, 1, 1, 1, 1, 2), 10.0, True, np.complex64)
    assert state["stored"].tolist() == [True, True, True, False] and state["flip"].tolist() == [True] + [False] * 3
    assert state["u_re"].shape == (4, 3) and state["predicted"].tolist() == [False] * 4
    assert all(np.isnan(state[key]).all() for key in ("lam_prev", "res_prev", "beta_prev"))


def test_spectrum_state_stores_the_vectors_in_the_precision_it_is_given():
    """The stored columns carry the real precision of the loop's self-energies, not a fixed one."""
    tracker = JacobianTracker(0.3)
    lam_pi, u = np.array([-0.3 + 0.0j]), np.ones((4, 1), dtype=np.complex128)
    tracker._last_certified = (lam_pi, np.zeros(1), np.ones(1, bool), np.ones(1, bool), u, np.zeros(1, bool))
    for dtype in (np.complex64, np.complex128):
        state = tracker.spectrum_state((1, 1, 1, 1, 1, 2), 10.0, True, dtype)
        real = np.zeros(0, dtype=dtype).real.dtype
        assert state["u_re"].dtype == real and state["u_im"].dtype == real


def test_spectrum_state_masks_uncertified_modes_and_round_trips(tmp_path):
    """The uncertified mode is dropped everywhere; only sub-band modes get a vector; the state round-trips exactly."""
    shape = (1, 1, 1, 1, 1, 2)  # a window of 2 complex entries, whose real representation has length 4
    tracker = JacobianTracker(0.3)
    lam_pi = np.array([-0.3, 0.05, 8.0, -0.7], dtype=np.complex128)
    res, flip = np.array([1e-6, 2e-6, 3e-6, 4e-6]), np.array([True, False, False, True])
    u = np.array([[1, 0.2, 9, 3], [2, -0.2, -9, 3], [3, 0.4, 8, 3], [4, -0.4, -8, 3]], dtype=np.complex128)
    u[:, 0] += [0.5j, 1.5j, 2.5j, 3.5j]
    tracker._last_certified = (lam_pi, res, np.array([True, True, True, False]), flip, u, np.zeros(4, dtype=bool))
    state = tracker.spectrum_state(shape, 10.0, True, np.complex64)
    assert np.array_equal(state["lam_pi"], np.array([-0.3, 0.05, 8.0], dtype=np.complex128))
    assert (state["lam_pi"].real < _STORE_BAND).tolist() == state["stored"].tolist() == [True, True, False]
    assert np.array_equal(state["flip"], flip[:3]) and np.array_equal(state["res"], res[:3])
    assert state["u_re"].shape == (4, 2) and state["u_re"].dtype == np.float32
    assert np.allclose(state["u_re"][:, 0], [1.0, 2.0, 3.0, 4.0], atol=1e-6)
    assert np.allclose(state["u_im"][:, 0], [0.5, 1.5, 2.5, 3.5], atol=1e-6)
    assert float(state["beta"]) == 10.0 and float(state["p_eff"]) == tracker.p_eff
    path = str(tmp_path / JACOBIAN_FILE)
    tracker.save_spectrum(path, shape, 10.0, True, np.complex64)
    loaded = load_spectrum(path)
    assert np.array_equal(loaded["lam_pi"], state["lam_pi"])
    assert np.array_equal(loaded["u_re"], state["u_re"]) and np.array_equal(loaded["u_im"], state["u_im"])
    assert tuple(loaded["shape"]) == shape and bool(loaded["converged"]) is True
    assert np.isnan(loaded["lam_prev"]).all() and loaded["predicted"].shape == (3,)


def test_spectrum_state_without_an_estimate_writes_empty_modes(tmp_path):
    """A tracker that never estimated anything still writes a file, with no mode and the configured damping."""
    shape = (1, 1, 1, 1, 1, 2)
    tracker = JacobianTracker(0.3)
    path = str(tmp_path / JACOBIAN_FILE)
    tracker.save_spectrum(path, shape, 10.0, False, np.complex64)
    loaded = load_spectrum(path)
    assert loaded["lam_pi"].size == 0 and loaded["res"].size == 0 and loaded["stored"].size == 0
    assert loaded["predicted"].size == 0 and loaded["lam_prev"].size == 0 and loaded["beta_prev"].size == 0
    assert loaded["u_re"].shape == (0, 0) and loaded["u_im"].shape == (0, 0)
    assert loaded["u_re"].dtype == np.float32 and float(loaded["p_eff"]) == 0.3


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
    assert np.array_equal(tracker.spectrum_state((1, 1, 1, 1, 1, 8), 10.0, False, np.complex64)["lam_pi"], certified)


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
    assert np.array_equal(tracker.spectrum_state(_SHAPE, 10.0, False, np.complex64)["lam_pi"], certified)


def _exact_modes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An exact spectrum on a two-entry window: one flip, a stable complex pair below the storage band, a stiff mode."""
    lam_pi = np.array([-0.3 + 0.0j, 0.05 + 0.2j, 0.05 - 0.2j, 4.0 + 0.0j])
    pair = np.array([0.0, 1.0, 1.0j, 0.0]) / np.sqrt(2.0)
    u = np.column_stack([[1.0, 0.0, 0.0, 0.0], pair, pair.conj(), [0.0, 0.0, 0.0, 1.0]]).astype(np.complex128)
    return lam_pi, np.full(4, 1e-10), u


def test_install_exact_spectrum_is_the_saved_snapshot_and_carries_into_a_successor(tmp_path):
    """The exact set is what the file holds (flips by classify, marked exact) and what a successor carries in."""
    tracker = JacobianTracker(0.4)
    lam_pi, res, u = _exact_modes()
    tracker.install_exact_spectrum(lam_pi, res, u)

    state = tracker.spectrum_state(_SHAPE, 10.0, True, np.complex128)
    assert np.array_equal(state["lam_pi"], lam_pi) and np.array_equal(state["res"], res)
    assert state["flip"].tolist() == classify(lam_pi, np.ones(4, dtype=bool), 0.4, res)[0].tolist()
    assert state["flip"].tolist() == [True, False, False, False] and not state["predicted"].any()
    assert state["stored"].tolist() == [True, True, True, False] and bool(state["exact"])

    path = str(tmp_path / JACOBIAN_FILE)
    tracker.save_spectrum(path, _SHAPE, 10.0, True, np.complex128)
    loaded = load_spectrum(path)
    assert bool(loaded["exact"])
    successor = JacobianTracker(0.4)
    assert successor.carry_in(loaded, _expand_identity())
    assert successor.q.shape[1] == 1 and np.allclose(np.abs(successor.q[:, 0]), [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    assert not bool(successor.spectrum_state(_SHAPE, 12.0, True, np.complex128)["exact"])


def test_install_exact_check_flips_a_certified_unstable_pair_through_the_carried_path():
    """A certified negative pair installs a reflector, reports a switch, logs its label and is saved as exact."""
    tracker, logger = _logged_tracker()
    column = _real_column(0, 2).astype(np.complex128)
    assert tracker.install_exact_check(np.array([-0.7 + 0.0j]), np.array([1e-6]), column) is True
    assert tracker.q.shape[1] == 1 and tracker._carried_set is not None
    assert any("exact-check set installed" in call[0][0] for call in logger.info.call_args_list)
    state = tracker.spectrum_state(_SHAPE, 12.5, False, np.complex128)
    assert bool(state["exact"]) and state["flip"].tolist() == [True]


@pytest.mark.parametrize("value, res, installed", [(0.5, 1e-6, True), (-0.7, 0.5, False)])
def test_install_exact_check_without_a_certified_flip_reports_no_switch(value, res, installed):
    """A certified stable pair installs the per-direction damping alone; pairs failing the Ritz gate install nothing."""
    tracker = JacobianTracker(0.4)
    column = _real_column(0, 2).astype(np.complex128)
    assert tracker.install_exact_check(np.array([value + 0.0j]), np.array([res]), column) is False
    assert tracker.q is None and tracker.active is installed and (tracker._carried_set is not None) is installed
    assert installed or tracker._last_certified is None


def _own_flip_of_the_first_axis() -> tuple[JacobianTracker, tuple[list[np.ndarray], list[np.ndarray]]]:
    """A tracker whose own reflector holds the certified flip of axis 0 (lambda_Pi = -0.5), and the run it came from."""
    tracker = JacobianTracker(0.4)
    pairs = linear_pairs(np.diag([1.5, 0.5, 0.2, -0.3]), np.zeros(4), 0.3, 7, [1.0, 1.0, 1.0, 1.0])
    tracker.update(pairs[0][:5], pairs[1][:5])
    assert tracker.q is not None and tracker._carried_set is None
    assert np.linalg.norm(tracker.q.T @ _real_column(0, 4)) > 0.99
    return tracker, pairs


@pytest.mark.parametrize("value", [-0.3, 0.4])
def test_an_exact_check_keeps_the_trackers_own_certified_flip(value):
    """An exact check of another mode, flipped or stable, leaves the reflection of the tracker's own flip in place."""
    tracker, _ = _own_flip_of_the_first_axis()
    tracker.install_exact_check(np.array([value + 0.0j]), np.array([1e-9]), _real_column(1, 4).astype(np.complex128))
    assert np.linalg.norm(tracker.q.T @ _real_column(0, 4)) > 0.99
    assert bool(np.linalg.norm(tracker.q.T @ _real_column(1, 4)) > 0.99) == (value < 0)


def test_the_estimate_keeps_an_exact_check_set_that_holds_its_flip():
    """The next update, whose estimate still flips the held axis, keeps the exact-check set and its extra flip."""
    tracker, pairs = _own_flip_of_the_first_axis()
    tracker.install_exact_check(np.array([-0.3 + 0.0j]), np.array([1e-9]), _real_column(1, 4).astype(np.complex128))
    assert tracker.update(pairs[0][:6], pairs[1][:6]) is False
    assert tracker._set_label == "exact-check" and tracker.q.shape[1] == 2


def test_an_exact_check_set_survives_calm_updates_and_goes_on_residual_growth():
    """Calm updates that certify nothing inside the exact-check set keep it, and a growing residual releases it."""
    tracker = JacobianTracker(0.4)
    tracker.install_exact_check(np.array([-0.2 + 0.0j]), np.array([1e-6]), _real_column(2, 3).astype(np.complex128))
    for scale in [1.0 / (k + 1) for k in range(2 * _PERSIST)] + [10.0 * (k + 1) for k in range(_PERSIST)]:
        if scale == 10.0:
            assert tracker.active and tracker._carried_set is not None
        tracker.update(*linear_pairs(np.diag([0.5, 0.2, 1.0]), np.zeros(3), 0.3, 3, [scale, scale, 0.0]))
    assert not tracker.active and tracker._carried_set is None


def _one_mode(value: float, column: np.ndarray, certified: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A one-mode estimate for _note_check_candidate: its lambda_Pi, its certificate and its unit Ritz vector."""
    return np.array([value + 0.0j]), np.array([certified]), column.astype(np.complex128)


def test_two_matched_uncertified_updates_below_the_band_request_an_exact_check_of_the_mode():
    """The second matched uncertified estimate below the band requests the mode's real plane, handed out once."""
    tracker = JacobianTracker(0.4)
    column = _real_column(0, 2)
    for value in (-0.2, -0.05):
        tracker._note_check_candidate(*_one_mode(value, column), True)
    assert np.allclose(tracker.exact_check_request(), column, atol=1e-12)
    assert tracker.exact_check_request() is None


@pytest.mark.parametrize(
    "second, allow_flip",
    [
        (None, True),
        ((0.2, _real_column(0, 2), False), True),
        ((-0.05, _real_column(0, 2), True), True),
        ((-0.05, _real_column(1, 2), False), True),
        ((-0.05, _real_column(0, 2), False), False),
    ],
)
def test_no_exact_check_is_requested_without_a_persistent_uncertified_candidate(second, allow_flip):
    """One estimate, one above the band, a certified one, a mismatched one, or flips paused request no check."""
    tracker = JacobianTracker(0.4)
    tracker._note_check_candidate(*_one_mode(-0.2, _real_column(0, 2)), True)
    if second is not None:
        tracker._note_check_candidate(*_one_mode(*second), allow_flip)
    assert tracker.exact_check_request() is None


def test_exact_check_requests_wait_out_the_cooldown_and_stop_at_the_cap():
    """After a request the next ones wait _PERSIST informative updates, and no more than _MAX_EXACT_CHECKS come."""
    tracker = JacobianTracker(0.4)
    requested = []
    for _ in range(2 + (_MAX_EXACT_CHECKS + 1) * _PERSIST):
        tracker._note_check_candidate(*_one_mode(-0.2, _real_column(0, 2)), True)
        requested.append(tracker.exact_check_request() is not None)
    assert sum(requested) == _MAX_EXACT_CHECKS == 5
    assert np.all(np.diff(np.flatnonzero(requested)) == _PERSIST)


@pytest.mark.parametrize(
    "values, residuals, rearmed",
    [
        ([0.5], [1e-9], True),
        ([-0.7], [1e-9], False),
        ([0.005], [1e-9], False),
        ([0.5, -0.7], [1e-9, 1e-9], False),
        ([-0.2, 0.5], [0.5, 1e-9], False),
    ],
)
def test_a_check_that_certifies_its_lead_and_finds_every_pair_stable_gives_its_slot_back(values, residuals, rearmed):
    """A check certifying its lead and only clearly stable pairs re-arms its slot, at most _MAX_EXACT_CHECKS times."""
    tracker = JacobianTracker(0.4)
    columns = np.column_stack([_real_column(k + 1, 3) for k in range(len(values))]).astype(np.complex128)
    requested = 0
    for _ in range(2 + (2 * _MAX_EXACT_CHECKS + 2) * _PERSIST):
        tracker._note_check_candidate(*_one_mode(-0.2, _real_column(0, 3)), True)
        if tracker.exact_check_request() is not None:
            requested += 1
            tracker.install_exact_check(np.array(values, dtype=np.complex128), np.array(residuals), columns)
            # the installed set is released before the next candidate, as residual growth or a stable mode does
            tracker.q = tracker.w = tracker._nonuniform = tracker._applied = tracker._carried_set = None
    assert requested == (2 * _MAX_EXACT_CHECKS if rearmed else _MAX_EXACT_CHECKS)


def test_a_rung_start_check_leaves_the_mid_rung_budget_alone():
    """A stable rung-start check gives back no slot of the mid-rung budget, since it never took one."""
    tracker = JacobianTracker(0.4)
    tracker.carry_in(_exact_file_with_three_pairs(), _expand_identity())
    tracker.update_recorded([])
    columns = tracker.exact_check_request()
    assert columns is not None
    tracker.install_exact_check(np.array([0.5 + 0.0j]), np.array([1e-9]), columns[:, :1].astype(np.complex128))
    assert tracker._checks_done == 0 and tracker._rearmed == 0


def test_no_exact_check_is_requested_inside_the_reflector_or_while_an_exact_check_set_holds():
    """A candidate inside the installed reflector, or any candidate beside an exact-check set, is left alone."""
    reflected = JacobianTracker(0.4)
    reflected.q = _real_column(0, 2)
    reflected.w = reflected.q.T
    held = JacobianTracker(0.4)
    held.install_exact_check(np.array([0.5 + 0.0j]), np.array([1e-6]), _real_column(1, 2).astype(np.complex128))
    for tracker in (reflected, held):
        for value in (-0.2, -0.05):
            tracker._note_check_candidate(*_one_mode(value, _real_column(0, 2)), True)
        assert tracker.exact_check_request() is None


@pytest.mark.parametrize("exact", [True, False])
def test_carry_in_of_an_exact_file_requests_its_columns_on_the_first_update_allowing_flips(exact):
    """An exact predecessor's stored columns are requested once flips are allowed; an estimated file never asks."""
    predecessor = JacobianTracker(0.4)
    predecessor.install_exact_spectrum(np.array([-0.3, 0.5], dtype=complex), np.full(2, 1e-10), np.eye(4)[:, :2] + 0j)
    state = predecessor.spectrum_state(_SHAPE, 10.0, True, np.complex128)
    state["exact"] = np.bool_(exact)
    tracker = JacobianTracker(0.4)
    tracker.carry_in(state, _expand_identity())
    one_pair = ([np.zeros(2, dtype=np.complex128)], [np.ones(2, dtype=np.complex128)])
    tracker.update(*one_pair, allow_flip=False)
    assert tracker.exact_check_request() is None
    tracker.update(*one_pair)
    request = tracker.exact_check_request()
    assert (request is not None) is exact
    if exact:
        assert np.allclose(np.abs(request.T @ np.eye(4)[:, :2]), np.eye(2), atol=1e-12)


def test_an_exact_snapshot_stores_the_vectors_of_its_modes_below_the_check_band():
    """A +0.5 mode keeps its vector in an exact snapshot, which the storage band of an estimated one leaves out."""
    shape = (1, 1, 1, 1, 1, 2)
    exact = JacobianTracker(0.4)
    exact.install_exact_spectrum(np.array([0.5 + 0.0j]), np.array([1e-10]), _real_column(0, 2).astype(np.complex128))
    estimated = JacobianTracker(0.4)
    estimated.update(*linear_pairs(np.diag([0.5, 0.2]), np.zeros(2), 0.4, TRACKER_PAIRS, [1.0, 1.0]))
    assert 0.5 < _CHECK_BAND and exact.spectrum_state(shape, 10.0, True, np.complex128)["stored"].tolist() == [True]
    assert estimated.last_gated.any() and not estimated.spectrum_state(shape, 10.0, True, np.complex128)["stored"].any()


@pytest.mark.parametrize("gains", [[0.4, 1.5], [0.4, 0.7]])
def test_a_later_certifying_update_replaces_the_exact_snapshot_and_releases_its_eigenvectors(gains):
    """Once the tracker's own estimate certifies a mode again the snapshot is not exact and nothing keeps u alive."""
    tracker = JacobianTracker(0.4)
    lam_pi, res, u = _exact_modes()
    tracker.install_exact_spectrum(lam_pi, res, u)
    alive = weakref.ref(u)
    del u
    tracker.update(*linear_pairs(np.diag(gains), np.zeros(2), 0.4, TRACKER_PAIRS, [1.0, 1.0]))
    assert tracker.last_gated.any() and alive() is None
    assert not bool(tracker.spectrum_state(_SHAPE, 10.0, True, np.complex128)["exact"])


def test_carry_in_reflects_the_carried_flip_and_binds_the_damping_with_the_whole_spectrum():
    """The modes below the storage band carry vectors, the flipped one is reflected; every carried mode binds p_eff."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([-0.5, 0.05, 0.3, 8.0], np.eye(4))
    installed = tracker.carry_in(state, _expand_identity())
    assert installed and tracker.active and tracker.q.shape == (4, 1)
    assert tracker._flip.tolist() == [True, False]
    assert np.isclose(tracker.p_eff, min(0.4, 0.5 * 2 * 8.0 / 64.0), atol=1e-12)
    proposal = np.array([1.0 + 1.0j, 1.0 + 1.0j]).reshape(1, 1, 1, 1, 1, 2)
    zero = np.zeros_like(proposal)
    assert np.allclose(tracker.reflect(proposal, zero).reshape(-1), [-1.0 + 1.0j, 1.0 + 1.0j], atol=1e-12)


def test_carry_in_installs_the_per_direction_damping_beside_the_reflector():
    """A carried flip takes its own reversed damping through the installed map, not the scalar bound of the step."""
    tracker, logger = _logged_tracker()
    state = _carried_state([-4.0, 34.4 - 25.6j, 34.4 + 25.6j], np.eye(4))
    assert tracker.carry_in(state, _expand_identity())
    assert tracker.q.shape[1] == 1 and tracker._nonuniform[0].shape[1] == 1 and tracker._applied is None
    # the stiff pair binds the scalar step far below the damping the flipped mode allows on its own
    assert np.isclose(tracker._certified_damping()[1][0], -0.25, atol=1e-12) and tracker.p_eff < 0.05
    assert np.allclose(_mapped_columns(tracker._nonuniform, np.eye(4, 1)), -0.25 * np.eye(4, 1), atol=1e-12)
    assert any("1 of them reflected" in call[0][0] for call in logger.info.call_args_list)


@pytest.mark.parametrize(
    "lam, u, carried_p, p_eff, expected",
    [
        pytest.param([0.5, 8.0], np.eye(4), 0.02, 0.4, 0.125, id="binds-ignoring-the-carried-damping"),
        pytest.param([0.5, 8.0], np.eye(4), 0.4, 0.02, 0.02, id="never-raises-p-eff"),
        pytest.param([], np.zeros((4, 0)), 0.055, 0.4, 0.4, id="empty-spectrum-carries-no-bound"),
    ],
)
def test_carry_in_without_a_candidate_only_lowers_the_damping_to_the_carried_bound(lam, u, carried_p, p_eff, expected):
    """Carried modes above the margin install nothing and bind p_eff, never raising it; an empty spectrum binds none."""
    tracker = JacobianTracker(0.4)
    tracker.p_eff = p_eff
    assert tracker.carry_in(_carried_state(lam, u, p_eff=carried_p), _expand_identity()) is False
    assert not tracker.active and np.isclose(tracker.p_eff, expected, atol=1e-12)


def test_carry_in_applies_the_expansion_and_renormalizes():
    """Every stored column passes through the expansion and is normalized before the reflector is built."""
    tracker = JacobianTracker(0.4)
    expand = MagicMock(side_effect=lambda column: np.concatenate([column, column]) * 3.0)
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
    assert tracker.carry_in(state, _expand_identity()) is False and not tracker.active


def test_carry_in_defers_the_flip_but_snapshots_the_carried_set_while_flips_are_paused():
    """With flips paused the damping applies at once, the reflector waits, and the carried set is still snapshotted."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([-0.5, 8.0], np.eye(4))
    assert tracker.carry_in(state, _expand_identity(), allow_flip=False) is False
    assert not tracker.active and np.isclose(tracker.p_eff, 0.125, atol=1e-12) and tracker._pending is not None
    assert tracker._last_certified is not None
    assert np.array_equal(tracker._last_certified[0], np.array([-0.5 + 0.0j]))


def test_carry_in_installs_the_orthogonal_reflector_of_a_conjugate_pair():
    """A carried complex-conjugate pair reflects its shared real plane, the plain reflection at a step scalar one."""
    tracker = JacobianTracker(0.4)
    w = np.array([1.0 + 0.5j, 0.3 - 0.2j, -0.4 + 0.1j, 0.2 + 0.3j])
    state = _carried_state([-0.2 + 0.3j, -0.2 - 0.3j], np.column_stack([w, np.conj(w)]))
    assert tracker.carry_in(state, _expand_identity())
    assert tracker.active and tracker.q.shape == (4, 2)
    assert np.allclose(tracker.w, tracker.q.T, atol=1e-12)
    proposal = np.array([1.0 + 1.0j, 1.0 + 1.0j]).reshape(_SHAPE)
    reflected = tracker.reflect(proposal, np.zeros_like(proposal))
    r = to_vec(proposal)
    expected = to_mat(r - 2.0 * tracker.q @ (tracker.q.T @ r), _SHAPE)
    assert np.allclose(reflected, expected, atol=1e-12)


def _stable_pairs(n_iter: int, p: float = 0.4) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Damped pairs of a diagonal map whose two modes are stable (lambda_Pi 0.6 and 0.3)."""
    return linear_pairs(np.diag([0.4, 0.7]), np.zeros(2), p, n_iter, [1.0, 1.0])


def _carry(tracker, lam, column, n, allow_flip=True):
    """Carries one mode with the given real-representation column on an n-dimensional window."""
    shape = (1, 1, 1, 1, 1, n)
    return tracker.carry_in(_carried_state([lam], column, shape=shape), _expand_identity(), allow_flip=allow_flip)


def test_a_flipped_carried_set_survives_calm_updates_without_a_certified_overlap():
    """Calm updates alone do not release a carried reflector while the tracker certifies nothing that overlaps it."""
    tracker = JacobianTracker(0.4)
    _carry(tracker, -0.2, _real_column(2, 3), 3)
    feed_windows(tracker, linear_pairs(np.diag([0.4, 0.7, 1.0]), np.zeros(3), 0.4, 8, [1.0, 1.0, 0.0]))
    assert tracker.active and tracker._carried_set is not None


def test_a_flipped_carried_set_is_released_by_a_certified_stable_overlap():
    """A certified stable mode lying in the carried reflector releases it after three such updates."""
    tracker, logger = _logged_tracker()
    _carry(tracker, -0.2, _real_column(0, 2), 2)
    feed_windows(tracker, _stable_pairs(9))
    # the estimate's own map on the two certified stable directions takes over after the release
    assert tracker.q is None and tracker._carried_set is None and tracker.active and tracker._applied is not None
    assert any("lies in the carried basis" in str(call) for call in logger.info.call_args_list)


def test_carried_basis_is_released_by_a_growing_residual():
    """A carried basis whose iteration keeps losing ground is released by the growth interlock."""
    tracker, logger = _logged_tracker()
    _carry(tracker, -0.2, _real_column(0, 2), 2)
    tracker._last_residual_norm = 0.0
    assert _grow_residual(tracker, _PERSIST)[-1]
    assert not tracker.active and tracker._carried_set is None
    assert "residual grew" in logger.info.call_args[0][0]


@pytest.mark.parametrize("carried", [0, 1], ids=["other-direction", "the-certified-direction"])
def test_carried_basis_is_replaced_by_a_certified_flip(carried):
    """A persisting certified flip replaces the carried reflector, even one landing on the carried direction itself."""
    tracker = JacobianTracker(0.4)
    _carry(tracker, -0.2, _real_column(carried, 2), 2)
    feed_windows(tracker, linear_pairs(np.diag([0.4, 1.5]), np.zeros(2), 0.4, 9, [1.0, 1.0]), _PERSIST)
    assert tracker.active and tracker._carried_set is None
    assert np.isclose(abs(tracker.q[1, 0]), 1.0, atol=1e-6)  # row 1 is the real part of the second coordinate


@pytest.mark.parametrize(
    "allow_flip, message",
    [
        pytest.param(False, "carried set installed", id="pended-at-carry-in"),
        pytest.param(True, "carried set pending until flips are allowed again", id="released-by-a-scaffold"),
    ],
)
def test_a_pending_carried_flip_installs_on_the_first_free_update(allow_flip, message):
    """A carried flip pended at carry-in or by a scaffold waits as pending and returns on the first free update."""
    tracker, logger = _logged_tracker()
    _carry(tracker, -0.5, _real_column(0, 2), 2, allow_flip=allow_flip)
    iterates, proposals = _stable_pairs(4)
    tracker.update(iterates[:3], proposals[:3], allow_flip=False)
    assert not tracker.active and tracker._pending is not None
    assert not allow_flip or any(message in str(call) for call in logger.info.call_args_list)
    assert tracker.update(iterates[:4], proposals[:4], allow_flip=True) is True
    assert tracker.active and tracker._carried_set is not None and tracker._pending is None
    assert any(message in str(call) for call in logger.info.call_args_list)


def test_a_pending_carried_install_clears_the_hold_of_a_growth_release():
    """A carried set installed from pending clears the hold, so the estimate's map may replace it at once."""
    tracker = JacobianTracker(0.4)
    _grow_residual(tracker, 1 + _PERSIST)
    assert not tracker.active and tracker._hold == _PERSIST
    _carry(tracker, -0.5, _real_column(0, 2), 2, allow_flip=False)
    assert tracker.update(*_stable_pairs(3)) is True
    assert tracker._hold == 0 and tracker.active and tracker._applied is not None


def test_a_pending_carried_install_restarts_the_growth_count():
    """A carried set installed from pending on an update whose residual grew ends that update with no growth counted."""
    tracker = JacobianTracker(0.4)
    _carry(tracker, -0.5, _real_column(0, 2), 2, allow_flip=False)
    tracker.update(*_stable_pairs(3), allow_flip=False)
    grown = linear_pairs(np.diag([0.4, 0.7]), np.zeros(2), 0.4, 3, [5.0, 5.0])
    assert tracker.update(*grown, allow_flip=True) is True
    assert tracker.active and tracker._grow == 0


def test_carried_basis_is_released_when_flips_are_suppressed():
    """An allow_flip=False update releases an installed carried basis and clears the carried flag."""
    tracker = JacobianTracker(0.4)
    _carry(tracker, -0.2, _real_column(0, 2), 2)
    iterates, proposals = _stable_pairs(3)
    tracker.update(iterates, proposals, allow_flip=False)
    assert not tracker.active and tracker._carried_set is None


def _drift_update(
    tracker: JacobianTracker, gains: np.ndarray, k: int, p: float = 0.07, allow_flip: bool = True
) -> bool:
    """Feeds one 3-pair window of a linear map started ever closer to the origin, so the residual never grows."""
    x0 = [4.0 * 0.7**k] * gains.shape[0]
    return tracker.update(*linear_pairs(gains, np.zeros(gains.shape[0]), p, 3, x0), allow_flip=allow_flip)


def _real_drift(value: float) -> np.ndarray:
    """Diagonal map whose first mode has lambda_Pi = value beside a stable mode at lambda_Pi = 0.5."""
    return np.diag([1.0 - value, 0.5])


def _pair_drift(value: float, imag: float = 0.3) -> np.ndarray:
    """Rotation block whose conjugate pair has lambda_Pi = value -+ i imag."""
    return np.array([[1.0 - value, imag], [-imag, 1.0 - value]])


def _first_install(tracker: JacobianTracker, gains: list[np.ndarray]) -> int | None:
    """Returns the 1-based update on which the tracker first installed a reflector over the given windows."""
    for k, mat in enumerate(gains):
        _drift_update(tracker, mat, k)
        if tracker.q is not None:
            return k + 1
    return None


_CROSSING = [0.19, 0.13, 0.07, 0.01, -0.05, -0.11, -0.17, -0.23]  # lambda_Pi drifting linearly through zero


def test_match_modes_pairs_new_and_old_modes_one_to_one_by_overlap():
    """Modes are matched by |u^T u'| > 0.5 one-to-one, a conjugate pair as one plane, and an unmatched one gets -1."""
    w = np.array([0.0, 0.0, 1.0 + 0.5j, 1.0 - 0.5j, 0.0, 0.0])
    u_old = np.column_stack([np.eye(6)[:, 0], w, np.conj(w), np.eye(6)[:, 4]]).astype(np.complex128)
    lam_old = np.array([0.3, 0.1 + 0.4j, 0.1 - 0.4j, 0.7], dtype=np.complex128)
    rotated = np.cos(0.4) * np.eye(6)[:, 0] + np.sin(0.4) * np.eye(6)[:, 5]
    u_new = np.column_stack([np.conj(w) * 1j, rotated, np.eye(6)[:, 1], w * 1j]).astype(np.complex128)
    lam_new = np.array([0.05 - 0.4j, 0.2, 0.9, 0.05 + 0.4j], dtype=np.complex128)
    assert _match_modes(lam_new, u_new, lam_old, u_old).tolist() == [2, 0, -1, 1]


def test_match_modes_never_pairs_a_real_mode_with_a_plane_and_is_empty_without_old_modes():
    """A real mode lying in an old pair's plane stays unmatched, and an empty old set matches nothing."""
    w = np.array([1.0 + 0.5j, 0.0, 0.0, 0.0])
    u_old = np.column_stack([w, np.conj(w)]).astype(np.complex128)
    lam_old = np.array([0.1 + 0.4j, 0.1 - 0.4j])
    u_new = np.eye(4)[:, :1].astype(np.complex128)
    assert _match_modes(np.array([0.2 + 0.0j]), u_new, lam_old, u_old).tolist() == [-1]
    assert _match_modes(np.array([0.2 + 0.0j]), u_new, lam_old[:0], np.zeros((4, 0))).tolist() == [-1]


@pytest.mark.parametrize(
    "now, prev, prev2, res_now, res_prev, expected",
    [
        pytest.param(0.02, 0.16, 0.30, 1e-6, 1e-6, True, id="linear-decrease-clears-the-band"),
        pytest.param(0.02, 0.28, 0.30, 1e-6, 1e-6, False, id="kink-scatter-widens-the-bar-past-the-prediction"),
        pytest.param(0.02, 0.16, 0.30, 0.07, 1e-6, False, id="residual-widens-band-and-bar-past-the-prediction"),
        pytest.param(0.02, 0.16, np.nan, 1e-6, 1e-6, False, id="two-points-are-no-chain"),
        pytest.param(0.02, 0.05, 0.02, 1e-6, 1e-6, False, id="not-monotone"),
        pytest.param(-0.05, 0.01, 0.07, 1e-6, 1e-6, False, id="already-certified-unstable"),
        pytest.param(0.16, 0.30, 0.44, 1e-6, 1e-6, False, id="prediction-stays-stable"),
    ],
)
def test_predict_crossing_rule_table(now, prev, prev2, res_now, res_prev, expected):
    """A crossing is predicted from three monotone points whose extrapolation clears the band plus its error bar."""
    lam = (np.array([value + 0j]) for value in (now, prev, prev2))
    predicted, pred, delta, approach = predict_crossing(*lam, np.array([res_now]), np.array([res_prev]))
    assert predicted.tolist() == [expected] and not approach.any()
    if np.isfinite(prev2):
        assert np.isclose(pred[0], 2 * now - prev, atol=1e-12)
        assert np.isclose(delta[0], max(res_now, res_prev) + 2 * abs(prev - 0.5 * (now + prev2)), atol=1e-12)


def test_predict_crossing_refuses_a_pole_type_approach():
    """A mode beyond the pole modulus whose modulus grew twice is not predicted, and is reported as an approach."""
    lam = np.array([0.01 + 6.0j]), np.array([0.2 + 5.0j]), np.array([0.5 + 4.0j])
    predicted, _, _, approach = predict_crossing(*lam, np.array([1e-6]), np.array([1e-6]))
    assert not predicted.any() and approach.tolist() == [True]
    small = np.array([0.01 + 3.0j]), np.array([0.2 + 2.5j]), np.array([0.5 + 2.0j])
    predicted, _, _, approach = predict_crossing(*small, np.array([1e-6]), np.array([1e-6]))
    assert predicted.tolist() == [True] and not approach.any()


@pytest.mark.parametrize(
    "now, prev, prev2, expected",
    [
        pytest.param(-10.0, 10.0, 10.0 / 3.0, True, id="equal-inverse-steps-through-zero"),
        pytest.param(-6.0, 10.0, 10.0 / 3.0, True, id="inverse-step-inside-the-continuity-constant"),
        pytest.param(-10.0, 10.0, 7.0, False, id="inverse-step-not-a-continuation"),
        pytest.param(-10.0, 10.0, np.nan, False, id="no-third-point"),
        pytest.param(-3.0, 10.0, 10.0 / 3.0, False, id="far-side-below-the-pole-modulus"),
        pytest.param(-10.0, 3.0, 2.0, False, id="near-side-below-the-pole-modulus"),
        pytest.param(10.0, -10.0, -10.0 / 3.0, False, id="flipped-mode-returning"),
    ],
)
def test_detect_pole_jump_table(now, prev, prev2, expected):
    """A pole jump is a sign change at large modulus whose 1/lambda step continues the previous one."""
    lam = (np.array([value + 0j]) for value in (now, prev, prev2))
    assert detect_pole_jump(*lam, np.array([1e-6]), np.array([1e-6])).tolist() == [expected]


def test_predicted_crossing_installs_the_flip_before_the_crossing_is_measured(monkeypatch):
    """A real mode drifting through zero is flipped where its extrapolation clears the band, before it is measured."""
    predicted, logger = _logged_tracker()
    assert _first_install(predicted, [_real_drift(v) for v in _CROSSING]) == 4
    assert np.isclose(abs(predicted.q[0, 0]), 1.0, atol=1e-8) and predicted._flip.sum() == 1
    assert np.isclose(predicted.last_lam_pi[predicted._flip][0], 0.01, atol=1e-8)
    assert np.isclose(predicted._certified_damping()[1][predicted.last_lam_pi.real < 0.1][0], -1.0, atol=1e-12)
    assert any("predicted crossing" in call[0][0] for call in logger.info.call_args_list)
    assert any("certified, predicted flip" in call[0][0] for call in logger.info.call_args_list)
    monkeypatch.setattr("dgamore.jacobian_stabilization._PREDICT_IN_LOOP", False)
    measured = JacobianTracker(0.4)
    assert _first_install(measured, [_real_drift(v) for v in _CROSSING]) == 4 + _FLIP_PERSIST


def test_predicted_crossing_flips_a_complex_pair_together_and_never_from_its_modulus():
    """A rotating pair is flipped as one plane on the predicted update, with no flip while its real part is positive."""
    tracker = JacobianTracker(0.4)
    for k, value in enumerate([0.14, 0.09, 0.04]):
        _drift_update(tracker, _pair_drift(value), k)
        assert not tracker._flip.any() and tracker.q is None
    _drift_update(tracker, _pair_drift(-0.003), 3)
    assert tracker._flip.tolist() == [True, True] and tracker.q.shape[1] == 2


def test_a_mode_oscillating_around_the_axis_is_never_predicted():
    """Values alternating around the boundary fail the monotonicity rule, so nothing is predicted or installed."""
    tracker = JacobianTracker(0.4)
    for k, value in enumerate([0.05, 0.02, 0.05, 0.02, 0.05, 0.02, 0.05]):
        _drift_update(tracker, _real_drift(value), k)
        assert not tracker._flip.any() and tracker.q is None


@pytest.mark.parametrize("values, expected", [([0.30, 0.28, 0.02], None), ([0.30, 0.16, 0.02], 3)])
def test_the_fit_scatter_blocks_a_prediction_the_drift_alone_would_make(values, expected):
    """A kinked chain widens the error bar past the extrapolation; the linear chain of the same endpoints predicts."""
    assert _first_install(JacobianTracker(0.4), [_real_drift(v) for v in values]) == expected


def test_a_pole_jump_is_flipped_on_the_update_of_the_jump(monkeypatch):
    """lambda_Pi = 1/mu with mu through zero: no pre-flip on the near side, the flip lands on the jump itself."""
    values = [2.0, 10.0 / 3.0, 10.0, -10.0, -10.0 / 3.0, -2.0]
    jumped, logger = _logged_tracker()
    assert _first_install(jumped, [_real_drift(v) for v in values]) == 4
    assert any("pole crossing" in call[0][0] for call in logger.info.call_args_list)
    assert not jumped._predicted.any() and not any("predicted flip" in c[0][0] for c in logger.info.call_args_list)
    assert "certified, flip)" in logger.info.call_args[0][0]
    monkeypatch.setattr("dgamore.jacobian_stabilization._PREDICT_IN_LOOP", False)
    assert _first_install(JacobianTracker(0.4), [_real_drift(v) for v in values]) == 3 + _FLIP_PERSIST


def test_a_pole_type_approach_is_logged_and_not_predicted():
    """A pair heading for a pole with a growing modulus is left alone although its real part falls monotonically."""
    tracker, logger = _logged_tracker()
    for k, (value, imag) in enumerate([(0.5, 4.0), (0.2, 5.0), (0.01, 6.0)]):
        _drift_update(tracker, _pair_drift(value, imag), k)
    assert tracker.q is None and not tracker._flip.any()
    assert any("pole-type approach" in call[0][0] for call in logger.info.call_args_list)


def test_a_frozen_update_keeps_the_chain_and_a_window_restart_clears_it():
    """A step below the floor adds no point and keeps the chain; an insufficient window starts every chain afresh."""
    kept = JacobianTracker(0.4)
    _drift_update(kept, _real_drift(0.19), 0)
    _drift_update(kept, _real_drift(0.13), 1)
    frozen = [np.ones((1, 2), dtype=np.complex128)] * 3
    assert kept.update(frozen, frozen) is False
    _drift_update(kept, _real_drift(0.07), 2)
    _drift_update(kept, _real_drift(0.01), 3)
    assert kept.q is not None
    cleared = JacobianTracker(0.4)
    for k, value in enumerate([0.19, 0.13, 0.07]):
        _drift_update(cleared, _real_drift(value), k)
    short = linear_pairs(_real_drift(0.07), np.zeros(2), 0.07, 2, [1.0, 1.0])
    assert cleared.update(*short) is False
    _drift_update(cleared, _real_drift(0.01), 3)
    assert cleared.q is None


def test_a_prediction_that_does_not_materialize_is_undone_by_the_calm_release():
    """A mode turning back up after a predicted flip is calm-released three updates later."""
    tracker = JacobianTracker(0.4)
    assert _first_install(tracker, [_real_drift(v) for v in _CROSSING[:4]]) == 4
    events = [_drift_update(tracker, _real_drift(value), 4 + k) for k, value in enumerate([0.04, 0.08, 0.12])]
    assert events == [False, False, True] and tracker.q is None and not tracker.active


def test_predictions_are_computed_but_not_acted_on_while_flips_are_paused():
    """Under a scaffold the prediction is logged and nothing is installed."""
    tracker, logger = _logged_tracker()
    for k, value in enumerate(_CROSSING[:4]):
        _drift_update(tracker, _real_drift(value), k, allow_flip=False)
    assert tracker.q is None and not tracker.active
    assert any("predicted crossing" in call[0][0] for call in logger.info.call_args_list)


def test_the_floor_warning_names_a_pending_predicted_crossing():
    """The floor warning says when a predicted crossing is pending on the update that pins the damping."""
    tracker, logger = _logged_tracker()
    tracker._predicted = np.array([True])
    tracker._lower_p_eff(_MIXING_FLOOR)
    assert "predicted crossing is pending" in logger.warning.call_args[0][0]


@pytest.mark.parametrize(
    "lam, lam_prev, res, res_prev, ratio, expected",
    [
        pytest.param(0.03, 0.09, 0.0, 0.0, 1.0, True, id="falling-mode-clears-the-band"),
        pytest.param(0.03, 0.05, 0.0, 0.0, 1.0, False, id="prediction-stays-stable"),
        pytest.param(0.03, 0.09, 0.02, 0.0, 1.0, False, id="residual-bar-past-the-prediction"),
        pytest.param(0.09, 0.03, 0.0, 0.0, 1.0, False, id="rising-mode"),
        pytest.param(0.05, -0.5, 0.0, 0.0, -1.0, False, id="rising-mode-on-a-warming-rung"),
        pytest.param(-0.05, 0.03, 0.0, 0.0, 1.0, False, id="already-unstable"),
        pytest.param(0.03, np.nan, 0.0, np.nan, 1.0, False, id="no-predecessor-value"),
    ],
)
def test_predict_rung_crossing_rule_table(lam, lam_prev, res, res_prev, ratio, expected):
    """A rung pre-flip needs a falling matched value whose linear extrapolation in beta clears the band and bar."""
    values = (np.array([value]) for value in (lam + 0j, lam_prev + 0j, res, res_prev))
    predicted, pred, pole = predict_rung_crossing(*values, ratio)
    assert predicted.tolist() == [expected] and not pole.any()
    if expected:
        assert np.isclose(pred[0], lam + (lam - lam_prev) * ratio, atol=1e-12)


def test_predict_rung_crossing_flags_a_mode_past_a_pole():
    """1/lambda extrapolated linearly in beta changing sign, with growing moduli beyond the cap, is a pole pre-flip."""
    predicted, pred, pole = predict_rung_crossing(
        np.array([15.0 + 0j]), np.array([6.0 + 0j]), np.zeros(1), np.zeros(1), 1.0
    )
    assert predicted.tolist() == [True] and pole.tolist() == [True]
    assert np.isclose(pred[0], 1.0 / (1.0 / 15.0 + (1.0 / 15.0 - 1.0 / 6.0)), atol=1e-12)
    _, _, no_pole = predict_rung_crossing(np.array([12.0 + 0j]), np.array([6.0 + 0j]), np.zeros(1), np.zeros(1), 1.0)
    assert not no_pole.any()
    _, _, small = predict_rung_crossing(np.array([4.0 + 0j]), np.array([1.5 + 0j]), np.zeros(1), np.zeros(1), 1.0)
    assert not small.any()


def _certify_direction(tracker: JacobianTracker, value: float, n_updates: int = 2) -> None:
    """Feeds windows whose first mode sits at lambda_Pi = value so the tracker certifies it on that direction."""
    for k in range(n_updates):
        _drift_update(tracker, _real_drift(value), k)


def test_carry_over_round_trip_over_three_rungs_pre_flips_the_extrapolated_crossing(tmp_path):
    """Rung 2 records the matched predecessor value; rung 3 extrapolates the two rungs in beta and pre-flips."""
    first = JacobianTracker(0.4)
    _certify_direction(first, 0.09)
    first.save_spectrum(str(tmp_path / "rung1.npz"), _SHAPE, 15.0, True, np.complex64)
    state_1 = load_spectrum(str(tmp_path / "rung1.npz"))
    assert np.allclose(state_1["lam_pi"][state_1["stored"]], [0.09], atol=1e-8) and np.isnan(state_1["lam_prev"]).all()

    second = JacobianTracker(0.4)
    assert second.carry_in(state_1, _expand_identity(), beta=20.0) is True
    assert second.q is None and second.active  # a stable carried column installs its map alone
    _certify_direction(second, 0.03)
    second.save_spectrum(str(tmp_path / "rung2.npz"), _SHAPE, 20.0, True, np.complex64)
    state_2 = load_spectrum(str(tmp_path / "rung2.npz"))
    stored = np.flatnonzero(state_2["stored"])
    assert stored.size == 1 and np.isclose(state_2["lam_pi"][stored][0], 0.03, atol=1e-8)
    assert np.isclose(state_2["lam_prev"][stored][0], 0.09, atol=1e-8) and state_2["beta_prev"][stored][0] == 15.0
    assert np.isnan(state_2["lam_prev"][~state_2["stored"]]).all() and not state_2["predicted"].any()

    second._logger = logger = MagicMock()
    second.save_spectrum(str(tmp_path / "rung2.npz"), _SHAPE, 20.0, True, np.complex64)
    assert any("1 matched to the predecessor's, 0 flipped by prediction" in c[0][0] for c in logger.info.call_args_list)
    third, logger = _logged_tracker()
    assert third.carry_in(state_2, _expand_identity(), beta=25.0) is True
    assert third.q.shape == (4, 1) and np.isclose(abs(third.q[0, 0]), 1.0, atol=1e-8)
    assert third._flip.tolist() == [True]
    assert any("cross-rung" in call[0][0] for call in logger.info.call_args_list)
    assert third.spectrum_state(_SHAPE, 25.0, False, np.complex64)["predicted"].tolist() == [True]


@pytest.mark.parametrize(
    "lam, beta, toggle, converged, message",
    [
        pytest.param(0.03, 20.0 + 5.0 * (_MAX_BETA_STEP_RATIO + 0.2), True, True, "step ratio", id="step-ratio-cap"),
        pytest.param(0.03, 25.0, True, False, "did not reach the pure fixed point", id="unconverged-predecessor"),
        pytest.param(0.03, 25.0, False, True, None, id="rung-toggle-off"),
        pytest.param(0.03, None, True, True, None, id="no-beta"),
        pytest.param(0.08, 25.0, True, True, None, id="matched-mode-predicting-no-crossing"),
    ],
)
def test_carry_in_carries_a_falling_stable_mode_unflipped_without_a_rung_prediction(
    monkeypatch, lam, beta, toggle, converged, message
):
    """A refused, switched-off, beta-less or non-crossing rung prediction carries the mode with positive damping."""
    monkeypatch.setattr("dgamore.jacobian_stabilization._PREDICT_ACROSS_RUNGS", toggle)
    tracker, logger = _logged_tracker()
    state = _carried_state([lam], _real_column(0, 2), beta=20.0, lam_prev=[0.09], beta_prev=15.0, converged=converged)
    assert tracker.carry_in(state, _expand_identity(), beta=beta) and tracker.q is None and tracker.active
    infos = [call[0][0] for call in logger.info.call_args_list]
    assert tracker._flip.tolist() == [False]
    assert any(message in line for line in infos) if message else not any("cross-rung" in line for line in infos)
    if not converged:
        assert np.isnan(tracker.spectrum_state(_SHAPE, 25.0, False, np.complex64)["lam_prev"]).all()


def test_a_stable_carried_column_takes_its_own_positive_damping_beside_a_pre_flipped_one():
    """The pre-flipped column is reflected with the reversed damping while the stable one keeps its positive damping."""
    tracker = JacobianTracker(0.4)
    state = _carried_state(
        [0.03, 0.05], np.eye(4)[:, :2], beta=20.0, lam_prev=[0.09, np.nan], beta_prev=15.0, res_prev=[0.0, np.nan]
    )
    assert tracker.carry_in(state, _expand_identity(), beta=25.0)
    assert tracker.q.shape == (4, 1) and tracker._flip.tolist() == [True, False] and tracker._applied is None
    lam_pi, damping = tracker._certified_damping()
    assert np.allclose(lam_pi, [0.03, 0.05], atol=1e-12)
    assert np.isclose(damping[0], -min(1.0, 0.5 * 2 * 0.03 / 0.03**2), atol=1e-12) and np.isclose(damping[1], 1.0)


def test_the_predicted_spectrum_bound_lowers_the_damping_only_for_predicted_modes():
    """The bound over the predicted spectrum binds where a mode is pre-flipped and leaves an unpredicted one alone."""
    predicted = JacobianTracker(0.4)
    state = _carried_state([0.05], _real_column(0, 2), beta=15.0, lam_prev=[3.1], beta_prev=10.0)
    assert predicted.carry_in(state, _expand_identity(), beta=20.0)
    assert predicted._flip.tolist() == [True] and np.isclose(predicted.p_eff, 0.5 * 2 * 3.0 / 9.0, atol=1e-12)
    # the carried map damps the pre-flipped direction by its predicted eigenvalue, not by the carried one
    assert np.isclose(predicted._certified_damping()[1][0], -0.5 * 2 * 3.0 / 9.0, atol=1e-12)
    assert np.allclose(predicted.last_lam_pi, [0.05], atol=1e-12)
    plain = JacobianTracker(0.4)
    state = _carried_state([0.05], _real_column(0, 2), beta=15.0, lam_prev=[np.nan], beta_prev=10.0)
    assert plain.carry_in(state, _expand_identity(), beta=20.0)
    assert plain._flip.tolist() == [False] and np.isclose(plain.p_eff, 0.4, atol=1e-12)


@pytest.mark.parametrize(
    "lam, lam_prev, beta, pole",
    [
        pytest.param(0.05 + 6.0j, 0.08 + 5.2j, 15.0, True, id="past-the-pole"),
        pytest.param(0.03 + 0.3j, 0.09 + 0.3j, 20.0, False, id="falling-real-part"),
    ],
)
def test_a_carried_pair_is_pre_flipped_as_one_plane_and_recorded_for_both_members(lam, lam_prev, beta, pole):
    """A carried pair predicted to cross is pre-flipped as one plane; past the pole it binds at the predicted value."""
    tracker, logger = _logged_tracker()
    w = np.array([1.0 + 0.5j, 0.3 - 0.2j, -0.4 + 0.1j, 0.2 + 0.3j])
    lams, prevs = [lam, np.conj(lam)], [lam_prev, np.conj(lam_prev)]
    state = _carried_state(lams, np.column_stack([w, np.conj(w)]), beta=beta, lam_prev=prevs, beta_prev=beta - 5.0)
    assert tracker.carry_in(state, _expand_identity(), beta=beta + 5.0)
    assert tracker._flip.tolist() == [True, True] and tracker.q.shape == (4, 2)
    if pole:
        inv_pred = 2 * (0.05 / abs(0.05 + 6.0j) ** 2) - 0.08 / abs(0.08 + 5.2j) ** 2
        predicted = 1.0 / inv_pred
        assert predicted < -1e3 and np.isclose(tracker.p_eff, _MIXING_FLOOR, atol=1e-12)
        lam_pi, damping = tracker._certified_damping()
        assert np.allclose(lam_pi, [0.05 + 6.0j, 0.05 - 6.0j], atol=1e-12) and (damping < 0.0).all()
        assert np.allclose(damping, -min(1.0, 0.5 * 2 * abs(predicted) / abs(predicted + 6.0j) ** 2), atol=1e-12)
        assert any("past the pole" in call[0][0] for call in logger.info.call_args_list)
        # the carried pair itself pins the damping to the floor, so the one measured-spectrum warning is the only one
        assert logger.warning.call_count == 1 and "measured spectrum" in logger.warning.call_args[0][0]
    assert tracker.spectrum_state(_SHAPE, beta + 5.0, False, np.complex64)["predicted"].tolist() == [True, True]


def test_a_floor_set_by_the_predicted_spectrum_warns_separately_and_keeps_the_measured_warning():
    """After a predicted-spectrum floor a measured bound at the floor still warns, once, though nothing is lowered."""
    tracker, logger = _logged_tracker()
    tracker._lower_p_eff(_MIXING_FLOOR, predicted=True)
    assert "predicted spectrum" in logger.warning.call_args[0][0] and "p_eff=0.4000" in logger.warning.call_args[0][0]
    assert np.isclose(tracker.p_eff, _MIXING_FLOOR, atol=1e-12)
    _, p_measured = classify(np.array([-200.0 + 0.0j]), np.ones(1, dtype=bool), 0.4)
    for _ in range(2):
        tracker._lower_p_eff(p_measured)
    assert p_measured == _MIXING_FLOOR and logger.warning.call_count == 2
    assert "measured spectrum" in logger.warning.call_args[0][0]


def test_a_predicted_flip_waits_out_the_hold_a_growth_release_left_behind():
    """A persisting prediction installs nothing during the hold of a growth release and installs once it is over."""
    tracker = JacobianTracker(0.4)
    assert _first_install(tracker, [_real_drift(v) for v in _CROSSING[:4]]) == 4
    tracker.q = tracker.w = tracker._nonuniform = tracker._applied = None
    tracker._persist, tracker._hold, tracker._grow = 0, _PERSIST, 0
    installs = []
    for k, value in enumerate([0.002, -0.0025, -0.007, -0.0095]):
        _drift_update(tracker, _real_drift(value), 4 + k)
        installs.append(tracker.q is not None)
    assert tracker._predicted.any() and installs == [False, False, False, True]


def test_a_carried_snapshot_records_no_predecessor_value_for_itself():
    """A run that never certified on its own writes nan predecessor values, not a self-match of the carried set."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([0.05], _real_column(0, 2), beta=15.0, lam_prev=[np.nan], beta_prev=10.0)
    assert tracker.carry_in(state, _expand_identity(), beta=20.0)
    assert np.isnan(tracker.spectrum_state(_SHAPE, 20.0, False, np.complex64)["lam_prev"]).all()
    _certify_direction(tracker, 0.03)
    saved = tracker.spectrum_state(_SHAPE, 20.0, True, np.complex64)
    assert np.isclose(saved["lam_prev"][saved["stored"]][0], 0.05, atol=1e-8)
    assert saved["beta_prev"][saved["stored"]][0] == 15.0


def test_a_pending_carried_set_without_a_flip_installs_its_map_without_an_event():
    """A pending carried set of stable columns installs its map alone on release, which switches no reflected map."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([0.05], _real_column(0, 2), beta=15.0, lam_prev=[np.nan], beta_prev=10.0)
    assert tracker.carry_in(state, _expand_identity(), allow_flip=False, beta=20.0) is False
    assert tracker._pending is not None and not tracker._pending[2].any()
    assert tracker.update(*_stable_pairs(3), allow_flip=True) is False
    assert tracker.q is None and tracker.active
    assert tracker._carried_set is None and tracker._applied is not None


def test_save_spectrum_writes_the_traces_beside_the_spectrum(tmp_path):
    """The per-iteration traces handed to save_spectrum land in the same file as the certified spectrum."""
    tracker = JacobianTracker(0.3)
    path = str(tmp_path / JACOBIAN_FILE)
    traces = {"eigenvalues": np.ones((2, 3), dtype=np.complex128), "eigenvalue_residuals": np.zeros((2, 3))}
    traces["damping"] = np.array([0.3, 0.2])
    tracker.save_spectrum(path, _SHAPE, 10.0, False, np.complex64, traces=traces)
    loaded = load_spectrum(path)
    assert np.array_equal(loaded["damping"], traces["damping"]) and loaded["eigenvalues"].shape == (2, 3)
    assert loaded["lam_pi"].size == 0 and loaded["eigenvalue_residuals"].shape == (2, 3)


def _band_mode_tracker() -> JacobianTracker:
    """Tracks a damped history of a diagonal map with lambda_Pi = 0.6, 0.3 and -0.005, all three certified."""
    tracker = JacobianTracker(0.4)
    feed_windows(tracker, linear_pairs(np.diag([0.4, 1.005, 0.7]), np.zeros(3), 0.4, 9, [1.0, 1.0, 1.0]))
    return tracker


def test_a_certified_mode_inside_the_undecidable_band_takes_the_scalar_damping():
    """A certified mode with a slightly negative real part takes the damped step's p_eff instead of a positive p_a."""
    tracker = _band_mode_tracker()
    band = int(np.argmin(np.abs(tracker.last_lam_pi + 0.005)))
    assert tracker.last_gated.all() and not tracker._flip.any()
    assert np.allclose(tracker._applied[band], tracker.p_eff, atol=1e-12)
    assert np.allclose(np.delete(tracker._applied, band), 1.0, atol=1e-12)


def test_a_band_mode_close_to_a_flat_one_keeps_the_whole_certified_span_mapped():
    """A band mode coupled to a nearby flat stable mode is not split off, so no certified direction loses its map."""
    tracker, logger = _logged_tracker()
    gains = np.diag([0.4, 1.005, 0.996])
    gains[1, 2] = 0.2
    feed_windows(tracker, linear_pairs(gains, np.zeros(3), 0.4, 9, [1.0, 1.0, 1.0]))
    messages = " ".join(str(call.args[0]) for call in logger.info.call_args_list)
    assert tracker.last_gated.all() and np.isfinite(tracker._applied).all()
    assert "obliquity cap" not in messages and "no per-direction damping" not in messages


def test_the_in_loop_map_can_be_switched_off(monkeypatch):
    """With the in-loop map switched off a certifying update installs no per-direction damping."""
    monkeypatch.setattr(jacobian_stabilization, "_NONUNIFORM_IN_LOOP", False)
    tracker = _band_mode_tracker()
    assert tracker.last_gated.all() and tracker._nonuniform is None and tracker._applied is None


def test_an_estimate_flip_on_a_member_the_exact_check_found_stable_replaces_it_and_keeps_the_growth_release():
    """The estimate flipping the checked-stable axis turns that member, is installed once, and growth still releases."""
    tracker, logger = _logged_tracker()
    columns = np.column_stack([_real_column(0, 2), _real_column(1, 2)]).astype(np.complex128)
    tracker.install_exact_check(np.array([0.3 + 0.0j, -0.4 + 0.0j]), np.full(2, 1e-9), columns)
    assert tracker.q.shape[1] == 1 and tracker._set_label == "exact-check"
    assert _drift_update(tracker, _real_drift(-0.3), 0) is True
    assert tracker.q.shape[1] == 2 and tracker._set_label == "exact-check"
    assert np.isclose(tracker._carried_set[0][0], -0.3, atol=1e-8) and tracker._carried_set[2].all()
    for k in range(1, 4):
        _drift_update(tracker, _real_drift(-0.3), k)
    installs = [c for c in logger.info.call_args_list if "exact-check set installed" in c[0][0]]
    assert len(installs) == 2 and tracker.active
    for k in range(_PERSIST):
        tracker.update(*linear_pairs(_real_drift(-0.3), np.zeros(2), 0.07, 3, [10.0 * (k + 1)] * 2))
    assert not tracker.active and "residual grew" in logger.info.call_args[0][0]


def test_the_estimate_extends_an_exact_check_set_by_a_new_certified_flip():
    """A certified flip outside the exact-check set joins it: both axes reflected, the label kept, a switch reported."""
    tracker = JacobianTracker(0.4)
    tracker.install_exact_check(np.array([-0.4 + 0.0j]), np.full(1, 1e-9), _real_column(1, 2).astype(np.complex128))
    assert _drift_update(tracker, _real_drift(-0.3), 0) is True
    assert tracker._set_label == "exact-check" and tracker.q.shape[1] == 2
    assert np.allclose(np.sort(tracker._carried_set[0].real), [-0.4, -0.3], atol=1e-8)


def test_an_exact_check_set_released_by_a_scaffold_comes_back_with_its_label():
    """A scaffold re-pends an exact-check set with its label, and the first free update re-installs it as one."""
    tracker = JacobianTracker(0.4)
    tracker.install_exact_check(np.array([-0.4 + 0.0j]), np.full(1, 1e-9), _real_column(0, 2).astype(np.complex128))
    tracker.update(*_stable_pairs(3), allow_flip=False)
    assert not tracker.active and tracker._pending[3] == "exact-check"
    tracker.update(*_stable_pairs(3))
    assert tracker.active and tracker._set_label == "exact-check"


def test_a_scaffold_releases_the_reflector_on_an_update_without_an_estimate():
    """A scaffolded update with too short a window still releases the installed reflector and reports the switch."""
    tracker, pairs = _own_flip_of_the_first_axis()
    tracker._logger = logger = MagicMock()
    assert tracker.update(pairs[0][:1], pairs[1][:1], allow_flip=False) is True
    assert not tracker.active and "tracked basis released; insufficient history" in logger.info.call_args[0][0]


def test_the_trackers_own_reflector_survives_calm_updates_without_a_stable_mode_inside_it():
    """Calm updates certifying stable modes away from the reflected axis keep the tracker's own reflector."""
    tracker, _ = _own_flip_of_the_first_axis()
    q_before = tracker.q
    for k in range(2 * _PERSIST):
        scale = 1.0 / (k + 1)
        tracker.update(*linear_pairs(np.diag([1.5, 0.5, 0.2, -0.3]), np.zeros(4), 0.3, 3, [0.0, scale, scale, 0.0]))
        assert tracker.last_gated.any() and not tracker._flip.any()
    assert tracker.q is q_before


def test_the_saved_spectrum_keeps_a_reflected_flip_the_last_estimate_no_longer_sees(tmp_path):
    """A flip that left the secant window while its reflector holds is written with the value it was flipped at."""
    tracker, _ = _own_flip_of_the_first_axis()
    tracker.update(*linear_pairs(np.diag([1.5, 0.5, 0.2, -0.3]), np.zeros(4), 0.3, 3, [0.0, 1.0, 1.0, 0.0]))
    assert tracker.q is not None and not np.any(tracker.last_lam_pi.real < 0.0)
    path = str(tmp_path / JACOBIAN_FILE)
    tracker.save_spectrum(path, (1, 1, 1, 1, 1, 4), 10.0, True, np.complex128)
    saved = load_spectrum(path)
    flipped = saved["lam_pi"][saved["flip"]]
    assert flipped.size == 1 and np.isclose(flipped[0], -0.5, atol=1e-8) and saved["stored"][saved["flip"]].all()


def test_the_exact_spectrum_keeps_a_reflected_flip_no_exact_mode_matches():
    """An exact spectrum missing the reflected flip is written with that flip beside it, still marked exact."""
    tracker, _ = _own_flip_of_the_first_axis()
    tracker.install_exact_spectrum(np.array([0.5 + 0.0j]), np.array([1e-10]), _real_column(1, 4).astype(complex))
    state = tracker.spectrum_state((1, 1, 1, 1, 1, 4), 10.0, True, np.complex128)
    assert bool(state["exact"]) and state["flip"].tolist() == [False, True]
    assert np.isclose(state["lam_pi"][1], -0.5, atol=1e-8)


def test_a_carried_mode_inside_the_band_takes_the_scalar_damping_on_the_carried_map():
    """A carried mode of undecidable negative sign takes p_eff on the carried map instead of a positive p_a of one."""
    tracker = JacobianTracker(0.4)
    assert _carry(tracker, -0.005, _real_column(0, 2), 2) and tracker.q is None
    assert not tracker._flip.any() and np.isclose(tracker._certified_damping()[1][0], tracker.p_eff, atol=1e-12)


def test_an_exact_check_mode_inside_the_band_takes_the_scalar_damping():
    """An exact-check pair of undecidable negative sign is not flipped and its map damps it at p_eff."""
    tracker = JacobianTracker(0.4)
    column = _real_column(0, 2).astype(np.complex128)
    tracker.install_exact_check(np.array([-0.005 + 0.0j]), np.array([1e-9]), column)
    assert tracker.q is None and tracker.active
    assert np.allclose(_mapped_columns(tracker._nonuniform, column.real), tracker.p_eff * column.real, atol=1e-12)


def test_ritz_values_carry_their_condition_number_in_the_projected_map():
    """The condition number of both Ritz values of [[1, 3], [0, 2]] is sqrt(1 + 3^2 / (2 - 1)^2)."""
    dx = np.eye(6, 2)
    df = dx @ np.array([[1.0, 3.0], [0.0, 2.0]])
    theta, _, res, _, _, kappa = _ritz_from_increments(dx, df, 0.0)
    assert np.allclose(np.sort(theta.real), [1.0, 2.0], atol=1e-12) and np.allclose(res, 0.0, atol=1e-12)
    assert np.allclose(kappa, np.sqrt(10.0), atol=1e-12)


def test_the_flip_band_widens_with_the_condition_number_of_the_ritz_value(monkeypatch):
    """A Ritz value at -0.05 with residual 0.02 is flipped when well conditioned and not at condition number five."""
    flips = []
    for kappa in (1.0, 5.0):
        theta, y, form = _schur_ritz(np.array([[1.05]]))
        planted = (theta, np.eye(4, 1) @ y, np.array([0.02]), np.eye(4, 1), form, np.array([kappa]))
        monkeypatch.setattr(jacobian_stabilization, "_ritz_from_increments", lambda dx, df, floor: planted)
        tracker = JacobianTracker(0.4)
        tracker.update(*linear_pairs(np.diag([1.05, 0.5]), np.zeros(2), 0.3, 3, [1.0, 0.0]))
        flips.append(bool(tracker._flip[0]))
        assert np.isclose(tracker.last_res[0], 0.02 * kappa, atol=1e-12)
    assert flips == [True, False]


def test_a_full_window_of_flipped_modes_warns_once_that_it_may_be_saturated():
    """Six certified flipped Ritz values warn once per run, a lone certified flip among uncertified values does not."""
    tracker, logger = _logged_tracker()
    lone = np.arange(TRACKER_PAIRS - 1) == 0
    tracker._warn_saturation(TRACKER_PAIRS - 1, lone, lone)
    assert not logger.warning.called
    gains = np.array([1.5, 2.0, 3.0, 4.0, 6.0, 9.0])
    x, entries = np.ones(6), []
    for _ in range(TRACKER_PAIRS + 1):
        entries.append((x.copy(), gains * x, 0.0))
        x = x + 0.1 * (gains * x - x)
    tracker.update_recorded(entries[:TRACKER_PAIRS])
    tracker.update_recorded(entries[1:])
    assert tracker.last_lam_pi.size == TRACKER_PAIRS - 1 and tracker._flip.all()
    saturated = [c for c in logger.warning.call_args_list if "full window" in c[0][0]]
    assert len(saturated) == 1


def test_the_spectrum_line_always_shows_the_flipped_and_the_least_stable_mode():
    """Stiff modes fill the line only after every flipped mode and the least-stable certified one."""
    tracker, logger = _logged_tracker()
    tracker.last_res, tracker.last_gated = np.full(6, 1e-3), np.ones(6, dtype=bool)
    tracker.last_lam_pi = np.array([34.0 + 26.0j, 34.0 - 26.0j, 12.0, 8.0, 0.02, -0.5])
    tracker._log_spectrum(np.array([False] * 5 + [True]))
    tracker.last_lam_pi = np.array([34.0 + 26.0j, 34.0 - 26.0j, 12.0, 8.0, 0.02, 0.5])
    tracker._log_spectrum(np.zeros(6, dtype=bool))
    flipped, soft = (call[0][0] for call in logger.info.call_args_list)
    assert "lambda_Pi=-0.5000+0.0000j (res 1.0e-03, certified, flip)" in flipped and flipped.count("lambda_Pi=") == 4
    assert "lambda_Pi=+0.0200+0.0000j (res 1.0e-03, certified, stable)" in soft and soft.count("lambda_Pi=") == 4


@pytest.mark.parametrize("lam, lam_prev, warned", [(0.08, 0.02, True), (0.25, 0.2, False), (0.08, 0.07, False)])
def test_carry_in_warns_about_a_mode_that_approached_the_boundary_and_receded(lam, lam_prev, warned):
    """A carried value rising beyond its error bar from near zero at the previous rung is warned about, not flipped."""
    tracker, logger = _logged_tracker()
    state = _carried_state([lam], _real_column(0, 2), beta=20.0, lam_prev=[lam_prev], beta_prev=15.0, res=[0.02])
    tracker.carry_in(state, _expand_identity(), beta=25.0)
    assert any("rose again" in c[0][0] for c in logger.warning.call_args_list) is warned
    assert tracker.q is None


def test_carry_in_keeps_the_columns_in_the_precision_of_the_file():
    """Single-precision stored columns are carried as complex64, double-precision ones as complex128."""
    for real, expected in ((np.float32, np.complex64), (np.float64, np.complex128)):
        state = _carried_state([-0.2], _real_column(0, 2))
        state["u_re"], state["u_im"] = state["u_re"].astype(real), state["u_im"].astype(real)
        tracker = JacobianTracker(0.4)
        assert tracker.carry_in(state, lambda column: np.asarray(column, dtype=np.float64))
        assert tracker._last_u.dtype == expected and tracker._predecessor[3].dtype == expected


def _exact_file_with_three_pairs() -> dict:
    """An exact spectrum file with one real mode and three complex pairs below the check band, seven real columns."""
    lam_pi = np.array([-0.3, 0.01 + 0.2j, 0.01 - 0.2j, 0.02 + 0.3j, 0.02 - 0.3j, 0.03 + 0.1j, 0.03 - 0.1j])
    u = np.zeros((14, 7), dtype=np.complex128)
    u[0, 0] = 1.0
    for k, col in enumerate((1, 3, 5)):
        pair = np.zeros(14, dtype=np.complex128)
        pair[2 * k + 1], pair[2 * k + 2] = 1.0, 1.0j
        u[:, col], u[:, col + 1] = pair / np.sqrt(2.0), np.conj(pair) / np.sqrt(2.0)
    return _carried_state(lam_pi, u, shape=(1, 1, 1, 1, 1, 7), exact=True)


def test_the_rung_start_check_never_splits_a_conjugate_pair_and_holds_its_own_columns():
    """Seven carried columns are cut at five, before the lone half of the last pair, into a contiguous own array."""
    tracker = JacobianTracker(0.4)
    tracker.carry_in(_exact_file_with_three_pairs(), _expand_identity())
    columns = tracker._start_check
    assert columns.shape == (14, _CHECK_START - 1) and columns.flags.owndata and columns.flags.c_contiguous


def test_a_tracker_without_exact_checks_keeps_no_start_columns_and_requests_nothing():
    """Without exact checks an exact file keeps no start columns and an uncertified candidate requests nothing."""
    tracker = JacobianTracker(0.4, exact_checks=False)
    tracker.carry_in(_exact_file_with_three_pairs(), _expand_identity())
    for value in (-0.2, -0.05):
        tracker._note_check_candidate(*_one_mode(value, _real_column(0, 2)), True)
    assert tracker._start_check is None and tracker.exact_check_request() is None


def test_record_stores_the_window_in_the_real_precision_of_the_iterate():
    """A complex64 pair is stored as float32 with a single-precision floor of the sector norm, complex128 as float64."""
    rng = np.random.default_rng(6)
    window = rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))
    for dtype, real in ((np.complex64, np.float32), (np.complex128, np.float64)):
        tracker = JacobianTracker(0.4, to_sector=lambda array: 2.0 * to_vec(array))
        x, f, floor = tracker.record(window.astype(dtype), window.astype(dtype))
        assert x.dtype == real and f.dtype == real
        expected = _NOISE_FACTOR * np.finfo(real).eps * np.linalg.norm(2.0 * to_vec(window.astype(dtype)))
        assert np.isclose(floor, expected, atol=1e-6 * expected)


def test_save_spectrum_writes_an_uncompressed_file_and_leaves_the_old_one_when_a_write_fails(tmp_path, monkeypatch):
    """The spectrum file is stored uncompressed, and a write that fails midway leaves the previous file whole."""
    path = str(tmp_path / JACOBIAN_FILE)
    first = JacobianTracker(0.3)
    first.save_spectrum(path, _SHAPE, 10.0, True, np.complex64)
    with zipfile.ZipFile(path) as archive:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())

    def truncated_write(handle, **arrays):
        handle.write(b"PK")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", truncated_write)
    with pytest.raises(OSError):
        JacobianTracker(0.2).save_spectrum(path, _SHAPE, 10.0, True, np.complex64)
    assert float(load_spectrum(path)["p_eff"]) == 0.3


def test_a_lone_refused_flip_releases_the_installed_map(monkeypatch):
    """An estimate certifying only a flip whose reflector is refused leaves nothing to map, so the map is released."""
    tracker, _ = _stable_only_tracker(1)
    tracker._logger = logger = MagicMock()
    assert tracker.q is None and tracker._nonuniform is not None
    monkeypatch.setattr(JacobianTracker, "_flipped_reflector", lambda self, flip, q_kept, form: None)
    tracker.update(*linear_pairs(np.diag([1.5, 0.5, 0.2]), np.zeros(3), 0.3, 3, [1.0, 0.0, 0.0]))
    assert tracker._flip.all() and not tracker.active
    assert "tracked map released" in logger.info.call_args[0][0]


def test_schur_helpers_refuse_what_they_cannot_place_or_solve(monkeypatch):
    """An unlocated Ritz value, a failed Sylvester solve and an unusable triangular solve each refuse cleanly."""
    log = MagicMock()
    assert _diagonal_positions(np.array([5.0 + 0.0j]), np.array([[1.0]])) is None
    assert _conjugate_partners(np.array([1.0 + 1.0j, 2.0 - 1.0j])).tolist() == [-1, -1]
    theta, _, form = _schur_ritz(np.array([[4.0, 0.5], [0.0, 0.7]]))
    lam_pi = 1.0 - theta
    raising = MagicMock(side_effect=np.linalg.LinAlgError("singular"))
    monkeypatch.setattr(jacobian_stabilization, "solve_triangular", raising)
    assert _nonuniform_map(np.eye(6, 2), form, theta, lam_pi, lam_pi.real < -_MARGIN, np.ones(2, bool), log) is None
    assert "restricted Jacobian of the certified span is unusable" in log.call_args[0][0]
    monkeypatch.setattr(jacobian_stabilization, "solve_sylvester", raising)
    tracker = JacobianTracker(0.4, logger=log)
    assert tracker._flipped_reflector(lam_pi.real < -_MARGIN, np.eye(6, 2), form) is None
    assert any("Sylvester equation could not be solved" in c[0][0] for c in log.info.call_args_list)


def test_opposes_flip_compares_a_step_with_the_flipped_damped_step_on_the_reflected_subspace():
    """On the reflected axis a sector step along the raw residual opposes the flipped damped step, the reverse not."""
    tracker = JacobianTracker(0.4)
    residual = np.array([1.0 + 0.0j, 0.3 + 0.0j])
    assert not tracker.opposes_flip(to_vec(residual), residual)
    tracker.q = _real_column(0, 2)
    tracker.w = tracker.q.T
    assert tracker.opposes_flip(np.array([0.3, -1.0, 0.0, 0.0]), residual)
    assert not tracker.opposes_flip(np.array([-0.3, 1.0, 0.0, 0.0]), residual)
    assert not tracker.opposes_flip(np.array([0.0, 1.0, 0.0, 0.0]), residual)


def test_start_directions_are_the_least_stable_and_the_stiffest_certified_mode():
    """The start directions are the real parts of the snapshot's least stable and stiffest certified vectors."""
    tracker = JacobianTracker(0.4)
    assert tracker.start_directions() == (None, None)
    tracker.install_exact_spectrum(*_exact_modes())
    least_stable, stiffest = tracker.start_directions()
    assert least_stable.dtype == np.float64 and np.array_equal(least_stable, [1.0, 0.0, 0.0, 0.0])
    assert np.array_equal(stiffest, [0.0, 0.0, 0.0, 1.0])


def test_the_exact_spectrum_warns_about_an_unstable_mode_the_reflector_does_not_hold():
    """An exact mode past the flip margin outside the installed reflector is warned about; a held one is not."""
    tracker, _ = _own_flip_of_the_first_axis()
    tracker._logger = MagicMock()
    u = np.column_stack([_real_column(0, 4), _real_column(1, 4), _real_column(2, 4)]).astype(np.complex128)
    tracker.install_exact_spectrum(np.array([-0.5, 0.5, -0.4], dtype=complex), np.full(3, 1e-10), u)
    warnings = [str(call.args[0]) for call in tracker._logger.warning.call_args_list]
    assert len(warnings) == 1 and "-0.4000" in warnings[0] and "-0.5000" not in warnings[0]
    tracker._logger.reset_mock()
    tracker.install_exact_spectrum(np.array([-0.5, 0.5], dtype=complex), np.full(2, 1e-10), u[:, :2])
    tracker._logger.warning.assert_not_called()


_EXACT_SPECTRUM = (np.array([-0.4 + 0.0j, 0.1 + 0.0j, 3.0 + 0.0j]), np.full(3, 1e-10), np.eye(8)[:, [2, 5, 6]] + 0j)


def _checked_tracker(dtype: type = np.complex128, file_dtype: type | None = None) -> JacobianTracker:
    """A tracker carried from an exact file, by default in the window's precision, updated twice and exact-checked."""
    n, real = 4, np.zeros(0, dtype=dtype if file_dtype is None else file_dtype).real.dtype
    tracker = JacobianTracker(0.4)
    state = _carried_state([-0.3, 0.2, 0.6], np.eye(2 * n)[:, :3], shape=(1, 1, 1, 1, 1, 8), exact=True)
    state["u_re"], state["u_im"] = state["u_re"].astype(real), state["u_im"].astype(real)
    tracker.carry_in(state, _expand_identity())
    iterates, proposals = linear_pairs(np.diag([1.5, 0.5, 0.2, -0.3]), np.zeros(n), 0.3, 4, [1.0] * n)
    for count in (3, 4):
        tracker.update([x.astype(dtype) for x in iterates[:count]], [f.astype(dtype) for f in proposals[:count]])
    tracker.install_exact_check(np.array([-0.5 + 0j, 0.4 + 0j]), np.full(2, 1e-10), np.eye(2 * n)[:, [3, 4]] + 0j)
    return tracker


def _written_spectra(tracker: JacobianTracker, folder: Path, name: str) -> list[dict[str, bytes]]:
    """The jacobian.npz members a tracker writes as it stands and after it installs the exact spectrum."""
    files = []
    for step in ("tracker", "exact"):
        if step == "exact":
            tracker.install_exact_spectrum(*_EXACT_SPECTRUM)
        path = str(folder / f"{name}_{step}.npz")
        tracker.save_spectrum(path, (1, 1, 1, 1, 1, 8), 10.0, True, np.complex64)
        with zipfile.ZipFile(path) as archive:
            files.append({member: archive.read(member) for member in archive.namelist()})
    return files


def test_the_post_loop_release_drops_the_loop_state_and_writes_the_same_jacobian_file(tmp_path):
    """Released after the loop, the tracker gives the same start directions and jacobian.npz bytes, minus loop state."""
    import copy

    tracker = _checked_tracker()
    released = copy.deepcopy(tracker)
    released.release_after_loop()
    loop_state = ("w", "_nonuniform", "_carried_set", "_check_request", "_check_candidate", "_last_u")
    assert all(getattr(tracker, name) is not None and getattr(released, name) is None for name in loop_state)
    written = [[state.start_directions()] for state in (tracker, released)]
    for files, (name, state) in zip(written, (("kept", tracker), ("released", released))):
        files += _written_spectra(state, tmp_path, name)
    assert all(np.array_equal(a, b) for a, b in zip(written[0][0], written[1][0])) and written[0][1:] == written[1][1:]
    assert load_spectrum(str(tmp_path / "kept_tracker.npz"))["lam_pi"].size == 3
    assert np.isfinite(load_spectrum(str(tmp_path / "kept_exact.npz"))["lam_prev"]).any()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_the_ritz_sets_take_the_complex_precision_of_the_recorded_window(dtype):
    """Estimate, snapshot, held flips, exact-check set, candidate and exact spectrum take the window's precision."""
    tracker = _checked_tracker(dtype)
    held, checked, candidate = tracker._held_flips[2], tracker._carried_set[1], tracker._check_candidate[1]
    assert all(u.dtype == dtype for u in (tracker._last_u, tracker._last_certified[4], held, checked, candidate))
    tracker.install_exact_spectrum(*_EXACT_SPECTRUM)
    assert tracker._last_certified[4].dtype == dtype
    first = JacobianTracker(0.4)
    first.update([np.ones(4, dtype=dtype)], [np.ones(4, dtype=dtype)])
    first.install_exact_check(np.array([-0.5 + 0j]), np.array([1e-10]), _real_column(0, 4).astype(np.complex128))
    assert first._carried_set[1].dtype == dtype


def test_a_complex64_run_carrying_a_complex128_file_keeps_the_sets_it_builds_in_complex64():
    """A complex64 run keeps a complex128 file's columns and stores its own held flips and check set in complex64."""
    tracker = _checked_tracker(np.complex64, np.complex128)
    held, checked = tracker._held_flips[2], tracker._carried_set[1]
    assert tracker._predecessor[3].dtype == np.complex128 and held.dtype == checked.dtype == np.complex64


def test_complex64_ritz_sets_write_the_same_jacobian_file_as_complex128_ones(tmp_path, monkeypatch):
    """A complex64 run writes the same jacobian.npz members with its Ritz sets in complex64 as in complex128."""
    written = []
    for name in ("single", "double"):
        with monkeypatch.context() as mp:
            if name == "double":
                mp.setattr(JacobianTracker, "_stored", lambda self, u: u)
            tracker = _checked_tracker(np.complex64)
            assert tracker._last_u.dtype == (np.complex64 if name == "single" else np.complex128)
            written.append(_written_spectra(tracker, tmp_path, name))
    assert written[0] == written[1]


def test_complex64_ritz_vectors_are_read_in_double_precision():
    """Complex64 Ritz vectors give float64 mode planes and the carried map's gains of their complex128 values."""
    rng = np.random.default_rng(12)
    u = np.linalg.qr(rng.standard_normal((8, 2)))[0] + 0j
    state = _carried_state([-0.2, 0.05], u, shape=(1, 1, 1, 1, 1, 8))
    single, double = JacobianTracker(0.4), JacobianTracker(0.4)
    for tracker, real in ((single, np.float32), (double, np.float64)):
        state["u_re"], state["u_im"] = state["u_re"].astype(real), state["u_im"].astype(real)
        assert tracker.carry_in(state, _expand_identity())
    assert single._last_u.dtype == np.complex64 and single._applied is None
    bases = jacobian_stabilization._mode_planes(single.last_lam_pi, single._last_u)[1]
    assert len(bases) == 2 and all(basis.dtype == np.float64 for basis in bases)
    assert np.allclose(single._certified_damping()[1], double._certified_damping()[1], rtol=0.0, atol=1e-12)


def test_complex64_ritz_vectors_are_held_by_their_double_precision_cosine(monkeypatch):
    """Between its float32 and float64 values, the cosine of a complex64 vector is held or not as the float64 one is."""
    rng = np.random.default_rng(0)
    q = np.linalg.qr(rng.standard_normal((4096, 1)))[0]
    u = (0.9 * q + 0.05 * (rng.standard_normal(q.shape) + 1j * rng.standard_normal(q.shape))).astype(np.complex64)
    cosine = np.linalg.norm(q.T @ u.astype(np.complex128)) / np.linalg.norm(u.astype(np.complex128))
    single = np.linalg.norm(q.T @ u) / np.linalg.norm(u)
    assert cosine != single
    monkeypatch.setattr(jacobian_stabilization, "_FLIP_OVERLAP", 0.5 * (cosine + single))
    tracker = JacobianTracker(0.4)
    tracker.q, tracker.w = q, q.T
    for value in (-0.2, -0.05):
        tracker._note_check_candidate(np.array([value + 0.0j]), np.array([False]), u, True)
    held = bool(cosine > single)
    assert tracker._held_flipped(u)[0] == held and (tracker.exact_check_request() is None) == held


def test_orthogonal_dual_rows_are_views_and_reflect_as_the_copied_rows(monkeypatch):
    """Orthogonal dual rows are views of the basis or span, and reflect and stabilize as copied rows do to 1e-12."""
    checked, _ = _own_flip_of_the_first_axis()
    checked.install_exact_check(np.array([-0.3 + 0.0j]), np.array([1e-9]), _real_column(1, 4).astype(np.complex128))
    monkeypatch.setattr(jacobian_stabilization, "_OBLIQUE_CAP", 0.0)
    fallback = JacobianTracker(0.4)
    fallback.update(*linear_pairs(np.diag([1.5, 1.3, 0.2, -0.3]), np.zeros(4), 0.3, 5, [1.0, 1.0, 1.0, 1.0]))
    assert checked.q.shape[1] == fallback.q.shape[1] == 2
    rng = np.random.default_rng(13)
    proposal, iterate = (rng.standard_normal(4) + 1j * rng.standard_normal(4) for _ in range(2))
    for tracker in (checked, fallback):
        pairs = [(tracker.q, tracker.w), tracker._nonuniform[:2]]
        assert all(np.shares_memory(basis, dual) and np.array_equal(dual, basis.T) for basis, dual in pairs)
        step = tracker.stabilize_step(0.5 * (proposal + iterate), iterate, proposal - iterate)
        reflected = tracker.reflect(proposal, iterate)
        tracker.w = np.ascontiguousarray(tracker.w)
        span, dual, damping = tracker._nonuniform
        tracker._nonuniform = (span, np.ascontiguousarray(dual), damping)
        assert np.allclose(tracker.reflect(proposal, iterate), reflected, rtol=0.0, atol=1e-12)
        stabilized = tracker.stabilize_step(0.5 * (proposal + iterate), iterate, proposal - iterate)
        assert np.allclose(stabilized, step, rtol=0.0, atol=1e-12)

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
table implements the sign rule and the damping bound, the Schur reflector flips exactly the spectrum of the
tracked invariant subspace and fixes its complement, and the tracker converges the two toy maps of arXiv:2606.04936
that plain damping sends to the unphysical fixed point. The carry-over tests plant the spectrum state a run writes,
round-trip it through save_spectrum and load_spectrum, and drive carry_in and the release rules that hold, re-pend
or drop a carried basis.
"""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest

from dgamore.jacobian_stabilization import (
    _FLIP_OVERLAP,
    _MAP_NORM_CAP,
    _MARGIN,
    _MAX_BETA_STEP_RATIO,
    _MIN_ITERATION_CAP,
    _MIXING_FLOOR,
    _NOISE_FACTOR,
    _PERSIST,
    _POLE_MODULUS,
    _RITZ_GATE,
    _SCHUR_TOL,
    _SIGNS_UNBOUNDED,
    _STORE_BAND,
    _SUBSPACE_ANGLE,
    JACOBIAN_FILE,
    JacobianTracker,
    SchurForm,
    TRACKER_PAIRS,
    _carried_nonuniform_map,
    _direction_damping,
    _match_modes,
    _nonuniform_map,
    _schur_ritz,
    classify,
    detect_pole_jump,
    load_spectrum,
    predict_crossing,
    predict_rung_crossing,
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
    """Runs the tracker protocol proposal -> record pair -> reflect -> mix -> stabilize and returns the iterate."""
    x = np.asarray(x0, dtype=np.complex128)
    iterates, proposals = [], []
    for _ in range(n_iter):
        proposal = step_fn(x)
        iterates.append(x.copy())
        proposals.append(proposal.copy())
        alpha = p if tracker is None or (bound_only_with_basis and not tracker.active) else tracker.p_eff
        used = proposal if tracker is None else tracker.reflect(proposal, x)
        mixed = x + alpha * (used - x)
        if tracker is not None:
            mixed = tracker.stabilize_step(mixed, x, proposal - x)
            tracker.update(iterates, proposals)
        x = mixed
    return x.real


def rotated_map(alpha: float) -> np.ndarray:
    """Returns a real 2x2 map with eigenvalues 2.0 and 0.5 whose unstable eigenvector is rotated by ``alpha``."""
    rot = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return rot @ np.diag([2.0, 0.5]) @ rot.T


def _carried_state(
    lam_pi,
    u,
    shape=(1, 1, 1, 1, 1, 2),
    res=None,
    converged=True,
    p_eff=0.4,
    beta=10.0,
    lam_prev=None,
    res_prev=None,
    beta_prev=None,
):
    """Builds a spectrum state from real-representation columns: the old format, or with ``lam_prev`` the new one."""
    lam_pi = np.asarray(lam_pi, dtype=np.complex128)
    u = np.asarray(u, dtype=np.complex128)
    stored = lam_pi.real < (-_MARGIN if lam_prev is None else _STORE_BAND)
    cols = list(np.flatnonzero(stored))
    re_cols = [to_mat(u[:, j].real, shape).reshape(-1).astype(np.complex64) for j in cols]
    im_cols = [to_mat(u[:, j].imag, shape).reshape(-1).astype(np.complex64) for j in cols]
    n_complex = int(np.prod(shape))
    state = {
        "lam_pi": lam_pi,
        "res": np.zeros(lam_pi.size) if res is None else np.asarray(res, dtype=np.float64),
        "flip": lam_pi.real < 0.0,
        "stored": stored,
        "u_re": np.column_stack(re_cols) if re_cols else np.zeros((n_complex, 0), dtype=np.complex64),
        "u_im": np.column_stack(im_cols) if im_cols else np.zeros((n_complex, 0), dtype=np.complex64),
        "shape": np.asarray(shape, dtype=np.int64),
        "beta": np.float64(beta),
        "p_eff": np.float64(p_eff),
        "converged": np.bool_(converged),
    }
    if lam_prev is not None:
        lam_prev = np.asarray(lam_prev, dtype=np.complex128)
        state["predicted"] = np.zeros(lam_pi.size, dtype=bool)
        state["lam_prev"] = lam_prev
        state["res_prev"] = np.where(np.isnan(lam_prev), np.nan, 0.0) if res_prev is None else np.asarray(res_prev)
        state["beta_prev"] = np.where(np.isnan(lam_prev), np.nan, beta_prev)
    return state


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
        ([-0.005], [True], 0.4, [False], 0.4),
        ([-0.02], [True], 0.4, [True], 0.4),
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


def test_reflect_and_stabilize_step_hand_back_their_input_without_an_install():
    """Without a tracked subspace the reflection and without a map the stabilized step return their input object."""
    rng = np.random.default_rng(1)
    tracker = JacobianTracker(0.4)
    proposal = (rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))).astype(np.complex128)
    iterate = (rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))).astype(np.complex128)
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


def _mapped_columns(nonuniform: tuple, columns: np.ndarray) -> np.ndarray:
    """Applies an installed per-direction map to a set of columns, i.e. the certified part of the stabilized step."""
    span, dual, damping = nonuniform
    return span @ (damping @ (dual @ columns))


def test_certified_directions_contract_by_their_own_damping():
    """Every certified direction contracts by its own Eq. 21 damping and the untracked one by the step scalar."""
    tracker, p_step = _installed_nonuniform_tracker(), 0.3
    factors = [_step_factor(tracker, _NONUNIFORM_GAINS, index, p_step) for index in range(4)]
    assert tracker.active
    assert np.allclose(factors, [0.0, 0.5, 0.0, 1.0 - p_step * 0.8], atol=1e-8)


_STIFFENED_GAINS = np.diag([-39.0, 0.5, 41.0, 0.2])  # the flipped direction's lambda_Pi moves from -4 to -40


def test_the_installed_map_follows_a_stiffening_estimate():
    """A basis kept while its flipped eigenvalue stiffens takes the new damping, not the one it was installed with."""
    tracker = _installed_nonuniform_tracker()
    q_before = tracker.q
    tracker.update(*linear_pairs(_STIFFENED_GAINS, np.zeros(4), 0.01, 9, [1.0, 1.0, 1.0, 0.0]))
    iterate = np.zeros(4, dtype=np.complex128)
    iterate[2] = 1e-3
    residual = (_STIFFENED_GAINS @ iterate.real).astype(np.complex128) - iterate
    assert tracker.q is q_before
    assert np.allclose(tracker.stabilize_step(iterate, iterate, residual).real[2], 0.0, atol=1e-8)


_RESTABILIZED_GAINS = np.diag([-39.0, 0.5, 0.6, 0.2])  # the flipped direction's lambda_Pi moves from -4 to +0.4


def test_a_refreshed_map_unflips_a_direction_the_estimate_certifies_as_stable():
    """A flipped direction certified stable again takes its own positive damping on the very next update."""
    tracker = _installed_nonuniform_tracker()
    tracker.update(*linear_pairs(_RESTABILIZED_GAINS, np.zeros(4), 0.01, 9, [1.0, 1.0, 1.0, 0.0]))
    assert np.isclose(_step_factor(tracker, _RESTABILIZED_GAINS, 2), 0.6, atol=1e-6)
    assert tracker.predicted_rate() < 1.0


def test_a_new_flip_reaches_the_step_only_after_the_persistence_streak():
    """A direction certified unstable under an installed map keeps the mixing's step until the third update."""
    tracker = JacobianTracker(0.4)
    gains = np.diag([0.4, 5.0])
    iterates, proposals = linear_pairs(gains, np.zeros(2), 0.01, 9, [1.0, 1.0])
    factors = []
    # a fourth update of this synthetic, never-reflected history would release the basis by the growth interlock
    for end in range(3, 3 + _PERSIST):
        tracker.update(iterates[:end], proposals[:end])
        factors.append(_step_factor(tracker, gains, 1))
    assert np.allclose(factors, [2.2, 2.2, 0.0], atol=1e-6)


def test_a_persisted_flip_whose_reflector_is_refused_stays_out_of_the_map(monkeypatch):
    """A persisted flip whose reflector could not be built keeps the mixing's step and stays out of the map."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    monkeypatch.setattr(JacobianTracker, "_flipped_reflector", lambda self, flip, q_kept, form: None)
    gains = np.diag([0.4, 5.0])
    iterates, proposals = linear_pairs(gains, np.zeros(2), 0.01, 9, [1.0, 1.0])
    factors = []
    for end in range(3, 3 + _PERSIST):
        tracker.update(iterates[:end], proposals[:end])
        factors.append(_step_factor(tracker, gains, 1))
    assert tracker._persist == _PERSIST and tracker.q is None and tracker.active
    assert np.allclose(factors, [2.2, 2.2, 2.2], atol=1e-6)
    assert np.isnan(tracker._applied[tracker._flip]).all()
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
    assert np.isclose(tracker.p_eff, 0.025, atol=1e-12)
    assert np.allclose(factors, [0.0, 0.5, 0.2], atol=1e-8)


def test_the_stable_only_map_survives_calm_updates():
    """A map with no flip to undo is not calm-released: six calm certifying updates keep it, none is an event."""
    tracker, events = _stable_only_tracker(6)
    assert tracker.active and tracker.q is None and events == [False] * 6
    assert np.isclose(_step_factor(tracker, _STABLE_ONLY_GAINS, 1), 0.5, atol=1e-8)


def _grow_with_a_stable_only_spectrum(tracker: JacobianTracker, n_updates: int) -> list[bool]:
    """Feeds n_updates windows of a stable-only map, each restarted further out so the raw residual grows."""
    events = []
    for k in range(n_updates):
        pairs = linear_pairs(np.diag([0.5, 0.2]), np.zeros(2), 0.3, 3, [10.0 * (k + 1), 10.0 * (k + 1)])
        events.append(tracker.update(*pairs))
    return events


def test_a_growing_residual_releases_the_map_and_holds_it_off_for_persist_updates():
    """Three consecutive residual growths release a map-only state, which stays off for _PERSIST more updates."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    assert not any(_grow_with_a_stable_only_spectrum(tracker, 1 + _PERSIST))
    assert not tracker.active and "per-direction damping released" in logger.info.call_args[0][0]
    assert not any(_grow_with_a_stable_only_spectrum(tracker, _PERSIST)) and not tracker.active
    assert not any(_grow_with_a_stable_only_spectrum(tracker, 1))
    assert tracker.active and tracker.q is None


def test_nonuniform_map_damps_each_certified_direction_of_a_non_normal_map():
    """Every certified Ritz direction is an eigendirection of the map at its own signed damping, which is reported."""
    q_kept, b_proj = _projected_map(7, [4.0, -3.6, 0.7, 0.4], 0.5)
    theta, y, form = _schur_ritz(b_proj)
    lam_pi, gated = 1.0 - theta, np.array([True, True, True, False])
    (span, dual, damping), applied = _nonuniform_map(q_kept, form, theta, lam_pi, lam_pi.real < -_MARGIN, gated)
    expected = np.where(lam_pi.real < -_MARGIN, -1.0, 1.0) * np.minimum(1.0, np.abs(lam_pi.real) / np.abs(lam_pi) ** 2)
    coords = q_kept.T @ span
    assert span.shape == (10, 3)
    assert np.allclose(b_proj @ coords, coords @ (coords.T @ b_proj @ coords), atol=1e-10)
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
    theta, _, form = _schur_ritz(np.array([[4.0, 100.0], [0.0, 0.7]]))
    lam_pi, gated = 1.0 - theta, np.array([True, False])
    (span, dual, _), _ = _nonuniform_map(np.eye(8, 2), form, theta, lam_pi, lam_pi.real < -_MARGIN, gated, log)
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


def test_the_uniform_fallback_signs_every_certified_eigenvector_and_not_the_basis_axis():
    """Under the fallback the flipped eigenvector takes -p_min and every stable one +p_min, exactly."""
    theta, y, form = _schur_ritz(_FALLBACK_FORM)
    lam_pi, flip, q_kept = 1.0 - theta, (1.0 - theta).real < -_MARGIN, np.eye(8, 3)
    built, applied = _nonuniform_map(q_kept, form, theta, lam_pi, flip, np.ones(3, dtype=bool))
    expected = np.where(flip, -1.0, 1.0) * np.min(np.abs(_direction_damping(lam_pi, flip)))
    assert np.allclose(_mapped_columns(built, q_kept @ y), expected * (q_kept @ y), atol=1e-10)
    assert np.allclose(applied, expected, atol=1e-12)


def test_nonuniform_map_refuses_near_degenerate_blocks_of_opposite_sign():
    """Two certified blocks 2e-4 apart with opposite signs carry no bounded signed map and are refused."""
    log = MagicMock()
    q_kept = np.linalg.qr(np.random.default_rng(3).standard_normal((10, 2)))[0]
    theta, _, form = _schur_ritz(np.array([[1.0101, 0.5], [0.0, 1.0099]]))
    lam_pi, gated = 1.0 - theta, np.ones(2, dtype=bool)
    assert _nonuniform_map(q_kept, form, theta, lam_pi, lam_pi.real < -_MARGIN, gated, log) is None
    assert any("no per-direction damping used" in call[0][0] for call in log.call_args_list)


def test_the_certified_map_never_amplifies_a_residual_beyond_the_damping_it_encodes():
    """The uniform fallback of one flipped and two stable blocks stays inside the magnitude cap it was chosen for."""
    log = MagicMock()
    theta, _, form = _schur_ritz(_FALLBACK_FORM)
    lam_pi, flip = 1.0 - theta, (1.0 - theta).real < -_MARGIN
    tracker = JacobianTracker(0.4)
    tracker._nonuniform = _nonuniform_map(np.eye(8, 3), form, theta, lam_pi, flip, np.ones(3, dtype=bool), log)[0]
    rng = np.random.default_rng(4)
    residual = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    assert any("uniform damping" in call[0][0] for call in log.call_args_list)
    assert _map_gain(tracker, residual) <= _MAP_NORM_CAP * np.min(np.abs(_direction_damping(lam_pi, flip)))


def test_the_certified_map_refuses_a_flipped_mode_a_uniform_damping_would_leave_growing():
    """A flipped mode at -0.5 leaning on two stable ones under a coupling of three is left to the mixing's step."""
    log = MagicMock()
    theta, _, form = _schur_ritz(np.array([[0.7, 3.0, 3.0], [0.0, 0.6, 3.0], [0.0, 0.0, 1.5]]))
    lam_pi, flip = 1.0 - theta, (1.0 - theta).real < -_MARGIN
    assert _nonuniform_map(np.eye(12, 3), form, theta, lam_pi, flip, np.ones(3, dtype=bool), log) is None
    assert any("no per-direction damping used" in call[0][0] for call in log.call_args_list)


def test_predicted_rate_reports_the_uniform_fallback_damping():
    """On a uniform fallback every certified direction is reported at the one signed damping the map applies."""
    tracker = JacobianTracker(0.4)
    theta, _, form = _schur_ritz(_FALLBACK_FORM)
    lam_pi, flip, gated = 1.0 - theta, (1.0 - theta).real < -_MARGIN, np.ones(3, dtype=bool)
    tracker.last_lam_pi, tracker.last_gated = lam_pi, gated
    tracker._nonuniform, tracker._applied = _nonuniform_map(np.eye(8, 3), form, theta, lam_pi, flip, gated)
    p_min = np.min(np.abs(_direction_damping(lam_pi, flip)))
    expected = np.max(np.abs(1.0 - np.where(flip, -p_min, p_min) * lam_pi))
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
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    monkeypatch.setattr(
        "dgamore.jacobian_stabilization._carried_nonuniform_map", lambda *args, **kwargs: _SIGNS_UNBOUNDED
    )
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
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
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
    theta, _, form = _schur_ritz(np.array([[4.0, 0.5], [0.0, 4.0 + 1e-9]]))
    lam_pi, gated = 1.0 - theta, np.array([True, False])
    assert _nonuniform_map(np.eye(6, 2), form, theta, lam_pi, lam_pi.real < -_MARGIN, gated, log) is None
    assert "not separable from the uncertified one" in log.call_args[0][0]


@pytest.mark.parametrize("separation", [0.0, 1e-13, 1e-11, 5e-10])
def test_nonuniform_map_falls_back_to_uniform_damping_on_two_certified_blocks_closer_than_the_tolerance(separation):
    """Two certified blocks closer than the block separation tolerance build with the uniform fallback."""
    log = MagicMock()
    theta, _, form = _schur_ritz(np.array([[5.0, 0.2, 0.1], [0.0, 0.3, 0.5], [0.0, 0.0, 0.3 + separation]]))
    lam_pi, gated = 1.0 - theta, np.ones(3, dtype=bool)
    (span, _, _), applied = _nonuniform_map(np.eye(8, 3), form, theta, lam_pi, lam_pi.real < -_MARGIN, gated, log)
    assert separation <= _SCHUR_TOL * 5.0 and span.shape == (8, 3)
    assert any("uniform damping" in call[0][0] for call in log.call_args_list)
    assert np.allclose(np.abs(applied), np.min(np.abs(_direction_damping(lam_pi, lam_pi.real < -_MARGIN))), atol=1e-12)


@pytest.mark.parametrize("separation", [0.0, 1e-13, 1e-11, 5e-10])
def test_nonuniform_map_refuses_a_certified_block_an_uncertified_one_coincides_with(separation):
    """A certified and an uncertified eigenvalue that coincide refuse the map without raising."""
    log = MagicMock()
    theta, _, form = _schur_ritz(np.array([[5.0, 0.2, 0.1], [0.0, 0.3, 0.5], [0.0, 0.0, 0.3 + separation]]))
    lam_pi, gated = 1.0 - theta, np.array([True, True, False])
    assert _nonuniform_map(np.eye(8, 3), form, theta, lam_pi, lam_pi.real < -_MARGIN, gated, log) is None
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
    tracker.p_eff = 0.025
    tracker.last_lam_pi = np.array([0.5 + 0.0j, -4.0 + 0.0j])
    tracker.last_gated = np.ones(2, dtype=bool)
    tracker._last_u = np.eye(4, 2).astype(np.complex128)
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
    """The minimum iteration count is the capped ceil(c / (eps p)) over the certified real parts, zero without one."""
    tracker = JacobianTracker(0.4)
    assert tracker.minimum_iterations() == 0 and tracker.certified_margin == 0.0
    tracker.p_eff = 0.2
    tracker.last_lam_pi = np.array([0.25 + 0.0j, 4.0 + 0.0j, -0.05 + 0.0j])
    tracker.last_res, tracker.last_gated = np.zeros(3), np.array([True, True, False])
    assert np.isclose(tracker.certified_margin, 0.25, atol=1e-12) and tracker.minimum_iterations() == 10
    tracker.last_lam_pi = np.array([0.011 + 0.0j, 4.0 + 0.0j, -0.05 + 0.0j])
    assert tracker.minimum_iterations() == 228
    tracker.last_lam_pi = np.array([1e-9 + 0.0j, 4.0 + 0.0j, -0.05 + 0.0j])
    assert tracker.minimum_iterations() == 1
    tracker.last_gated = np.zeros(3, dtype=bool)
    assert tracker.minimum_iterations() == 0
    tracker.last_lam_pi, tracker.last_gated = np.array([4.0j]), np.ones(1, dtype=bool)
    tracker.last_res = np.zeros(1)
    assert tracker.minimum_iterations() == 0


def test_minimum_iterations_does_not_round_a_boundary_count_up():
    """A count that lands on an integer is not raised by the last bits of the eigenvalue it came from."""
    tracker = JacobianTracker(0.4)
    tracker.p_eff = 0.2
    tracker.last_lam_pi = np.array([0.25 - 1e-13 + 0.0j])
    tracker.last_res, tracker.last_gated = np.zeros(1), np.ones(1, dtype=bool)
    assert tracker.minimum_iterations() == 10


def test_minimum_iterations_is_capped():
    """A mapped pseudo-divergence pair just outside the band drives the count to the cap, never above it."""
    tracker = JacobianTracker(0.4)
    theta, _, form = _schur_ritz(np.array([[1.0 - 1.01e-2, 3.0], [-3.0, 1.0 - 1.01e-2]]))
    lam_pi, gated = 1.0 - theta, np.ones(2, dtype=bool)
    tracker.last_lam_pi, tracker.last_res, tracker.last_gated = lam_pi, np.zeros(2), gated
    flip = lam_pi.real < -_MARGIN
    tracker._nonuniform, tracker._applied = _nonuniform_map(np.eye(6, 2), form, theta, lam_pi, flip, gated)
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


def test_certified_margin_ignores_a_marginal_mode_inside_its_undecidable_band():
    """A certified mode whose real part lies inside max(_MARGIN, res) does not set the margin."""
    tracker = JacobianTracker(0.4)
    tracker.last_lam_pi = np.array([4e-3 + 0.0j, 0.25 + 0.0j, 0.02 + 0.0j])
    tracker.last_res = np.array([1e-3, 1e-3, 5e-2])
    tracker.last_gated = np.ones(3, dtype=bool)
    assert np.isclose(tracker.certified_margin, 0.25, atol=1e-12)
    tracker.last_lam_pi = np.array([4e-3 + 0.0j])
    tracker.last_res, tracker.last_gated = np.array([1e-3]), np.ones(1, dtype=bool)
    assert tracker.certified_margin == 0.0 and tracker.minimum_iterations() == 0


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
        if tracker.q is not None:
            break
    assert tracker.n_updates == _PERSIST
    assert events == [False] * (_PERSIST - 1) + [True]


def test_tracker_counters_reset_on_interruption():
    """An interrupted persistence or calm streak restarts from zero instead of carrying over stale progress."""
    tracker = JacobianTracker(0.4)
    unstable, stable, shift = np.diag([5.0, 0.7]), np.diag([0.2, 0.45]), np.array([1.0, 1.0])
    # the windows start ever closer to the fixed point, so the growth interlock never enters
    x0 = iter(np.arange(4.0, 0.1, -0.1))
    burst = [unstable] * (_PERSIST - 1) + [stable] + [unstable] * (_PERSIST - 1)
    for mat in burst:
        tracker.update(*linear_pairs(mat, shift, 0.3, 3, [next(x0), next(x0)]))
    assert tracker.q is None
    for mat in [unstable] * _PERSIST:
        tracker.update(*linear_pairs(mat, shift, 0.3, 3, [next(x0), next(x0)]))
    assert tracker.q is not None
    for mat in [stable] * _PERSIST:
        tracker.update(*linear_pairs(mat, shift, 0.3, 3, [next(x0), next(x0)]))
    assert tracker.q is None
    for mat in [unstable] * _PERSIST:
        tracker.update(*linear_pairs(mat, shift, 0.3, 3, [next(x0), next(x0)]))
    assert tracker.q is not None
    tracker.update(*linear_pairs(stable, shift, 0.3, 3, [next(x0), next(x0)]))
    assert tracker.q is not None


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


def test_tracker_allow_flip_false_releases_a_map_installed_without_a_reflector():
    """Suppressing flips releases an installed per-direction map at once, which switches no reflected map."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    iterates, proposals = linear_pairs(_STABLE_ONLY_GAINS, np.zeros(3), 0.01, 3, [1.0, 1.0, 1.0])
    tracker.update(iterates, proposals)
    assert tracker.q is None and tracker._nonuniform is not None and tracker.active
    assert tracker.update(iterates, proposals, allow_flip=False) is False
    assert not tracker.active and tracker._nonuniform is None and tracker._applied is None
    assert "per-direction damping released" in logger.info.call_args[0][0]


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
    """One full update emits one info line carrying the measured lambda_Pi spectrum, after the map's install line."""
    logger = MagicMock()
    tracker = JacobianTracker(0.2, logger=logger)
    iterates, proposals = linear_pairs(A_2D, B_2D, 0.2, 3, [0.0, 0.0])
    tracker.update(iterates, proposals)
    lines = [call[0][0] for call in logger.info.call_args_list]
    assert ["lambda_Pi" in line for line in lines] == [False, True]
    assert "per-direction damping installed on 1 certified direction" in lines[0]
    assert lines[0].endswith("1 certified direction.")


def test_spectrum_line_tags_every_certificate_and_decision():
    """A mode is tagged certified stable, marginal, flip or unstable-suppressed, or uncertified, by its state."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    tracker.last_lam_pi = np.array([0.5, 0.005, -0.3, -0.2], dtype=np.complex128)
    tracker.last_res = np.full(4, 1e-3)
    tracker.last_gated = np.array([True, True, True, True])
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
    assert np.allclose(reflect(u_f), -u_f, atol=1e-8)
    assert np.allclose(reflect(u_s), u_s, atol=1e-8)
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
    assert np.allclose(w, q.T, atol=1e-10)
    assert np.allclose(reflect(u[:, flip]), -u[:, flip], atol=1e-10)
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
    assert np.allclose(reflect(u[:, flip]), -u[:, flip], atol=1e-10)
    assert np.allclose(reflect(reflect(u)), u, atol=1e-10)


def test_reflector_falls_back_to_the_orthogonal_reflection_on_a_nearly_parallel_complement():
    """A flipped subspace closer to its complement than the obliquity cap is reflected orthogonally."""
    logger = MagicMock()
    theta, _, form = _schur_ritz(np.array([[4.0, 100.0], [0.0, 0.7]]))
    q, w = JacobianTracker(0.2, logger=logger)._flipped_reflector(theta.real > 2.0, np.eye(8, 2), form)
    assert np.allclose(w, q.T, atol=1e-12)
    assert any("orthogonal reflection" in call[0][0] for call in logger.info.call_args_list)


def test_reflector_refuses_a_flipped_subspace_it_cannot_separate():
    """Two eigenvalues 1e-9 apart with one of them flipped blow up the Sylvester solution, so no reflector is built."""
    logger = MagicMock()
    tracker = JacobianTracker(0.2, logger=logger)
    theta, _, form = _schur_ritz(np.array([[4.0, 0.5], [0.0, 4.0 + 1e-9]]))
    assert tracker._flipped_reflector(np.array([False, True]), np.eye(6, 2), form) is None
    assert "Sylvester solution norm" in logger.info.call_args_list[-2][0][0]
    assert "not separable from its complement" in logger.info.call_args[0][0]


def test_reflector_refuses_a_schur_form_lapack_cannot_reorder(monkeypatch):
    """A reordering LAPACK itself refuses (two blocks too close to swap) leaves the flipped subspace inseparable."""
    logger = MagicMock()
    tracker = JacobianTracker(0.2, logger=logger)
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


def test_spectrum_state_stores_every_vector_below_the_storage_band_but_flips_by_the_flip_band():
    """A vector is stored for every certified mode below +0.1; the flip decision is the separate -max(_MARGIN, res)."""
    tracker = JacobianTracker(0.3)
    lam_pi, res = np.array([-0.3 + 0.0j, -0.3 + 0.0j, 0.05 + 0.0j, 0.3 + 0.0j]), np.array([1e-6, 0.5, 1e-6, 1e-6])
    u = np.ones((4, 4), dtype=np.complex128)
    flip = lam_pi.real < -np.maximum(_MARGIN, res)
    tracker._last_certified = (lam_pi, res, np.ones(4, dtype=bool), flip, u, np.zeros(4, dtype=bool))
    state = tracker.spectrum_state((1, 1, 1, 1, 1, 2), 10.0, True, np.complex64)
    assert state["stored"].tolist() == [True, True, True, False] and state["flip"].tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert state["u_re"].shape == (2, 3) and state["predicted"].tolist() == [False] * 4
    assert (
        np.isnan(state["lam_prev"]).all() and np.isnan(state["res_prev"]).all() and np.isnan(state["beta_prev"]).all()
    )


def test_spectrum_state_stores_the_vectors_in_the_precision_it_is_given():
    """The stored columns carry the precision the loop keeps its self-energies in, not a fixed one."""
    tracker = JacobianTracker(0.3)
    u = np.ones((4, 1), dtype=np.complex128)
    tracker._last_certified = (
        np.array([-0.3 + 0.0j]),
        np.zeros(1),
        np.ones(1, dtype=bool),
        np.ones(1, dtype=bool),
        u,
        np.zeros(1, dtype=bool),
    )
    for dtype in (np.complex64, np.complex128):
        state = tracker.spectrum_state((1, 1, 1, 1, 1, 2), 10.0, True, dtype)
        assert state["u_re"].dtype == dtype and state["u_im"].dtype == dtype


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
    tracker._last_certified = (lam_pi, res, np.array([True, True, True, False]), flip, u, np.zeros(4, dtype=bool))

    state = tracker.spectrum_state(shape, 10.0, True, np.complex64)

    assert np.array_equal(state["lam_pi"], np.array([-0.3, 0.05, 8.0], dtype=np.complex128))
    assert (state["lam_pi"].real < _STORE_BAND).tolist() == state["stored"].tolist() == [True, True, False]
    assert np.array_equal(state["flip"], flip[:3]) and np.array_equal(state["res"], res[:3])
    assert state["u_re"].shape == (2, 2) and state["u_re"].dtype == np.complex64
    assert np.allclose(state["u_re"][:, 0], [1.0 + 3.0j, 2.0 + 4.0j], atol=1e-6)
    assert np.allclose(state["u_im"][:, 0], [0.5 + 2.5j, 1.5 + 3.5j], atol=1e-6)
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
    assert np.array_equal(tracker.spectrum_state((1, 1, 1, 1, 1, 2), 10.0, False, np.complex64)["lam_pi"], certified)


def test_carry_in_reflects_the_carried_flip_and_binds_the_damping_with_the_whole_spectrum():
    """Only the mode below the flip margin carries a vector and is reflected; every carried mode binds p_eff."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([-0.5, 0.05, 0.3, 8.0], np.eye(4))
    installed = tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)))
    assert installed and tracker.active and tracker.q.shape == (4, 1)
    assert tracker._flip.tolist() == [True]
    assert np.isclose(tracker.p_eff, min(0.4, 0.5 * 2 * 8.0 / 64.0), atol=1e-12)
    proposal = np.array([1.0 + 1.0j, 1.0 + 1.0j]).reshape(1, 1, 1, 1, 1, 2)
    zero = np.zeros_like(proposal)
    assert np.allclose(tracker.reflect(proposal, zero).reshape(-1), [-1.0 + 1.0j, 1.0 + 1.0j], atol=1e-12)


def test_carry_in_installs_the_per_direction_damping_beside_the_reflector():
    """A carried flip takes its own reversed damping through the installed map, not the scalar bound of the step."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    state = _carried_state([-4.0, 34.4 - 25.6j, 34.4 + 25.6j], np.eye(4))
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)))
    assert tracker.q.shape[1] == 1 and tracker._nonuniform[0].shape[1] == 1 and tracker._applied is None
    # the stiff pair binds the scalar step far below the damping the flipped mode allows on its own
    assert np.isclose(tracker._certified_damping()[1][0], -0.25, atol=1e-12) and tracker.p_eff < 0.05
    assert np.allclose(_mapped_columns(tracker._nonuniform, np.eye(4, 1)), -0.25 * np.eye(4, 1), atol=1e-12)
    assert any("1 of them reflected" in call[0][0] for call in logger.info.call_args_list)


def test_carry_in_without_a_candidate_only_sets_the_bound():
    """Carried modes all above the margin install nothing but still bind p_eff, ignoring the carried damping."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([0.5, 8.0], np.eye(4), p_eff=0.02)
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2))) is False
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


def test_carry_in_installs_the_orthogonal_reflector_of_a_conjugate_pair():
    """A carried complex-conjugate pair reflects its shared real plane, the plain reflection at a step scalar one."""
    tracker = JacobianTracker(0.4)
    shape = (1, 1, 1, 1, 1, 2)
    w = np.array([1.0 + 0.5j, 0.3 - 0.2j, -0.4 + 0.1j, 0.2 + 0.3j])
    state = _carried_state([-0.2 + 0.3j, -0.2 - 0.3j], np.column_stack([w, np.conj(w)]), shape)
    assert tracker.carry_in(state, _expand_identity(shape))
    assert tracker.active and tracker.q.shape == (4, 2)
    assert np.allclose(tracker.w, tracker.q.T, atol=1e-12)
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


def _stable_pairs(n_iter: int, p: float = 0.4) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Damped pairs of a diagonal map whose two modes are stable (lambda_Pi 0.6 and 0.3)."""
    return linear_pairs(np.diag([0.4, 0.7]), np.zeros(2), p, n_iter, [1.0, 1.0])


def _carry(tracker, lam, column, n, allow_flip=True):
    """Carries one mode with the given real-representation column on an n-dimensional window."""
    shape = (1, 1, 1, 1, 1, n)
    return tracker.carry_in(_carried_state([lam], column, shape=shape), _expand_identity(shape), allow_flip=allow_flip)


def test_a_flipped_carried_set_survives_calm_updates_without_a_certified_overlap():
    """Calm updates alone do not release a carried reflector while the tracker certifies nothing that overlaps it."""
    tracker = JacobianTracker(0.4)
    _carry(tracker, -0.2, _real_column(2, 3), 3)
    iterates, proposals = linear_pairs(np.diag([0.4, 0.7, 1.0]), np.zeros(3), 0.4, 8, [1.0, 1.0, 0.0])
    for n in range(3, 9):
        tracker.update(iterates[:n], proposals[:n])
    assert tracker.active and tracker._carried_set is not None


def test_a_flipped_carried_set_is_released_by_a_certified_stable_overlap():
    """A certified stable mode lying in the carried reflector releases it after three such updates."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    _carry(tracker, -0.2, _real_column(0, 2), 2)
    iterates, proposals = _stable_pairs(9)
    for n in range(3, 10):
        tracker.update(iterates[:n], proposals[:n])
    # the estimate's own map on the two certified stable directions takes over after the release
    assert tracker.q is None and tracker._carried_set is None and tracker.active and tracker._applied is not None
    assert any("lies in the carried basis" in str(call) for call in logger.info.call_args_list)


def test_carried_basis_is_released_by_a_growing_residual():
    """A carried basis whose iteration keeps losing ground is released by the growth interlock."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    _carry(tracker, -0.2, _real_column(0, 2), 2)
    tracker._last_residual_norm = 0.0
    assert _grow_with_a_certified_flip(tracker, _PERSIST)
    assert not tracker.active and tracker._carried_set is None
    assert "residual grew" in logger.info.call_args[0][0]


def test_carried_basis_is_replaced_by_a_certified_flip():
    """A certified unstable mode that persists installs the tracker's own reflector in place of the carried one."""
    tracker = JacobianTracker(0.4)
    _carry(tracker, -0.2, _real_column(0, 2), 2)
    iterates, proposals = linear_pairs(np.diag([0.4, 1.5]), np.zeros(2), 0.4, 9, [1.0, 1.0])
    for n in range(3, 3 + _PERSIST):
        tracker.update(iterates[:n], proposals[:n])
    assert tracker.active and tracker._carried_set is None
    assert np.isclose(abs(tracker.q[1, 0]), 1.0, atol=1e-6)  # row 1 is the real part of the second coordinate


def test_carried_flip_on_the_certified_direction_is_still_replaced():
    """A certified flip landing on the exact carried direction still replaces it instead of being mistaken for it."""
    tracker = JacobianTracker(0.4)
    _carry(tracker, -0.2, _real_column(1, 2), 2)
    iterates, proposals = linear_pairs(np.diag([0.4, 1.5]), np.zeros(2), 0.4, 9, [1.0, 1.0])
    for n in range(3, 3 + _PERSIST):
        tracker.update(iterates[:n], proposals[:n])
    assert tracker.active and tracker._carried_set is None


def test_pending_carried_flip_installs_on_the_first_free_update():
    """A carried flip kept pending under a scaffold is installed on the first update that allows flips."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    _carry(tracker, -0.5, _real_column(0, 2), 2, allow_flip=False)
    iterates, proposals = _stable_pairs(4)
    tracker.update(iterates[:3], proposals[:3], allow_flip=False)
    assert not tracker.active
    assert tracker.update(iterates[:4], proposals[:4], allow_flip=True) is True
    assert tracker.active and tracker._carried_set is not None and tracker._pending is None
    assert any("carried set installed" in str(call) for call in logger.info.call_args_list)


def test_a_pending_carried_install_clears_the_hold_of_a_growth_release():
    """A carried set installed from pending clears the hold, so the estimate's map may replace it at once."""
    tracker = JacobianTracker(0.4)
    _grow_with_a_stable_only_spectrum(tracker, 1 + _PERSIST)
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
    assert tracker.active and tracker._carried_set is not None and tracker._pending is None


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
    assert _match_modes(np.array([0.2 + 0.0j]), u_new, np.zeros(0, dtype=np.complex128), np.zeros((4, 0))).tolist() == [
        -1
    ]


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
    predicted, pred, delta, approach = predict_crossing(
        np.array([now + 0j]), np.array([prev + 0j]), np.array([prev2 + 0j]), np.array([res_now]), np.array([res_prev])
    )
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
    jump = detect_pole_jump(
        np.array([now + 0j]), np.array([prev + 0j]), np.array([prev2 + 0j]), np.array([1e-6]), np.array([1e-6])
    )
    assert jump.tolist() == [expected]


def test_predicted_crossing_installs_the_flip_ahead_of_the_streak(monkeypatch):
    """A real mode drifting through zero is flipped on the update its extrapolation clears the band, not the streak."""
    logger = MagicMock()
    predicted = JacobianTracker(0.4, logger=logger)
    assert _first_install(predicted, [_real_drift(v) for v in _CROSSING]) == 4
    assert np.isclose(abs(predicted.q[0, 0]), 1.0, atol=1e-8) and predicted._flip.sum() == 1
    assert np.isclose(predicted.last_lam_pi[predicted._flip][0], 0.01, atol=1e-8)
    assert np.isclose(predicted._certified_damping()[1][predicted.last_lam_pi.real < 0.1][0], -1.0, atol=1e-12)
    assert any("predicted crossing" in call[0][0] for call in logger.info.call_args_list)
    assert any("certified, predicted flip" in call[0][0] for call in logger.info.call_args_list)
    monkeypatch.setattr("dgamore.jacobian_stabilization._PREDICT_IN_LOOP", False)
    streak = JacobianTracker(0.4)
    assert _first_install(streak, [_real_drift(v) for v in _CROSSING]) == 7


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
    logger = MagicMock()
    values = [2.0, 10.0 / 3.0, 10.0, -10.0, -10.0 / 3.0, -2.0]
    jumped = JacobianTracker(0.4, logger=logger)
    assert _first_install(jumped, [_real_drift(v) for v in values]) == 4
    assert any("pole crossing" in call[0][0] for call in logger.info.call_args_list)
    assert not jumped._predicted.any() and not any("predicted flip" in c[0][0] for c in logger.info.call_args_list)
    assert "certified, flip)" in logger.info.call_args[0][0]
    monkeypatch.setattr("dgamore.jacobian_stabilization._PREDICT_IN_LOOP", False)
    assert _first_install(JacobianTracker(0.4), [_real_drift(v) for v in values]) == 6


def test_a_pole_type_approach_is_logged_and_not_predicted():
    """A pair heading for a pole with a growing modulus is left alone although its real part falls monotonically."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
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
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    for k, value in enumerate(_CROSSING[:4]):
        _drift_update(tracker, _real_drift(value), k, allow_flip=False)
    assert tracker.q is None and not tracker.active
    assert any("predicted crossing" in call[0][0] for call in logger.info.call_args_list)


def test_the_floor_warning_names_a_pending_predicted_crossing():
    """The floor warning says when a predicted crossing is pending on the update that pins the damping."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
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
    predicted, pred, pole = predict_rung_crossing(
        np.array([lam + 0j]), np.array([lam_prev + 0j]), np.array([res]), np.array([res_prev]), ratio
    )
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
    shape = (1, 1, 1, 1, 1, 2)
    first = JacobianTracker(0.4)
    _certify_direction(first, 0.09)
    first.save_spectrum(str(tmp_path / "rung1.npz"), shape, 15.0, True, np.complex64)
    state_1 = load_spectrum(str(tmp_path / "rung1.npz"))
    assert np.allclose(state_1["lam_pi"][state_1["stored"]], [0.09], atol=1e-8) and np.isnan(state_1["lam_prev"]).all()

    second = JacobianTracker(0.4)
    assert second.carry_in(state_1, _expand_identity(shape), beta=20.0) is True
    assert second.q is None and second.active  # a stable carried column installs its map alone
    _certify_direction(second, 0.03)
    second.save_spectrum(str(tmp_path / "rung2.npz"), shape, 20.0, True, np.complex64)
    state_2 = load_spectrum(str(tmp_path / "rung2.npz"))
    stored = np.flatnonzero(state_2["stored"])
    assert stored.size == 1 and np.isclose(state_2["lam_pi"][stored][0], 0.03, atol=1e-8)
    assert np.isclose(state_2["lam_prev"][stored][0], 0.09, atol=1e-8) and state_2["beta_prev"][stored][0] == 15.0
    assert np.isnan(state_2["lam_prev"][~state_2["stored"]]).all() and not state_2["predicted"].any()

    logger = MagicMock()
    second._logger = logger
    second.save_spectrum(str(tmp_path / "rung2.npz"), shape, 20.0, True, np.complex64)
    assert any("1 matched to the predecessor's, 0 flipped by prediction" in c[0][0] for c in logger.info.call_args_list)
    logger = MagicMock()
    third = JacobianTracker(0.4, logger=logger)
    assert third.carry_in(state_2, _expand_identity(shape), beta=25.0) is True
    assert third.q.shape == (4, 1) and np.isclose(abs(third.q[0, 0]), 1.0, atol=1e-8)
    assert third._flip.tolist() == [True]
    assert any("cross-rung" in call[0][0] for call in logger.info.call_args_list)
    assert third.spectrum_state(shape, 25.0, False, np.complex64)["predicted"].tolist() == [True]


def test_carry_in_refuses_the_rung_prediction_above_the_step_ratio_cap():
    """A rung more than twice the previous step away is carried as today, with the refusal logged."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    state = _carried_state([0.03], _real_column(0, 2), beta=20.0, lam_prev=[0.09], beta_prev=15.0)
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)), beta=20.0 + 5.0 * (_MAX_BETA_STEP_RATIO + 0.2))
    assert tracker.q is None and tracker._flip.tolist() == [False]
    assert any("step ratio" in call[0][0] for call in logger.info.call_args_list)


def test_carry_in_refuses_the_rung_prediction_of_an_unconverged_predecessor():
    """A predecessor that did not reach the pure fixed point carries as today and records no predecessor values."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    state = _carried_state([0.03], _real_column(0, 2), beta=20.0, lam_prev=[0.09], beta_prev=15.0, converged=False)
    tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)), beta=25.0)
    assert tracker.q is None and tracker._flip.tolist() == [False]
    assert any("did not reach the pure fixed point" in call[0][0] for call in logger.info.call_args_list)
    assert np.isnan(tracker.spectrum_state((1, 1, 1, 1, 1, 2), 25.0, False, np.complex64)["lam_prev"]).all()


def test_carry_in_without_the_rung_toggle_carries_as_today(monkeypatch):
    """With the across-rung prediction off a falling stable mode is carried with positive damping and no flip."""
    monkeypatch.setattr("dgamore.jacobian_stabilization._PREDICT_ACROSS_RUNGS", False)
    tracker = JacobianTracker(0.4)
    state = _carried_state([0.03], _real_column(0, 2), beta=20.0, lam_prev=[0.09], beta_prev=15.0)
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)), beta=25.0)
    assert tracker.q is None and tracker.active and tracker._flip.tolist() == [False]


def test_an_old_format_spectrum_file_carries_as_before():
    """A file without the predecessor keys installs its flipped columns exactly as the old rule did."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([-0.5, 0.05, 0.3, 8.0], np.eye(4))
    assert "lam_prev" not in state
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)), beta=20.0)
    assert tracker.q.shape == (4, 1) and tracker._flip.tolist() == [True]
    assert np.isclose(tracker.p_eff, min(0.4, 0.5 * 2 * 8.0 / 64.0), atol=1e-12)


def test_a_stable_carried_column_takes_its_own_positive_damping_beside_a_pre_flipped_one():
    """The pre-flipped column is reflected with the reversed damping while the stable one keeps its positive damping."""
    tracker = JacobianTracker(0.4)
    state = _carried_state(
        [0.03, 0.05], np.eye(4)[:, :2], beta=20.0, lam_prev=[0.09, np.nan], beta_prev=15.0, res_prev=[0.0, np.nan]
    )
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)), beta=25.0)
    assert tracker.q.shape == (4, 1) and tracker._flip.tolist() == [True, False] and tracker._applied is None
    lam_pi, damping = tracker._certified_damping()
    assert np.allclose(lam_pi, [0.03, 0.05], atol=1e-12)
    assert np.isclose(damping[0], -min(1.0, 0.5 * 2 * 0.03 / 0.03**2), atol=1e-12) and np.isclose(damping[1], 1.0)


def test_the_predicted_spectrum_bound_lowers_the_damping_only_for_predicted_modes():
    """The bound over the predicted spectrum binds where a mode is pre-flipped and leaves an unpredicted one alone."""
    predicted = JacobianTracker(0.4)
    state = _carried_state([0.05], _real_column(0, 2), beta=15.0, lam_prev=[3.1], beta_prev=10.0)
    assert predicted.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)), beta=20.0)
    assert predicted._flip.tolist() == [True] and np.isclose(predicted.p_eff, 0.5 * 2 * 3.0 / 9.0, atol=1e-12)
    # the carried map damps the pre-flipped direction by its predicted eigenvalue, not by the carried one
    assert np.isclose(predicted._certified_damping()[1][0], -0.5 * 2 * 3.0 / 9.0, atol=1e-12)
    assert np.allclose(predicted.last_lam_pi, [0.05], atol=1e-12)
    plain = JacobianTracker(0.4)
    state = _carried_state([0.05], _real_column(0, 2), beta=15.0, lam_prev=[np.nan], beta_prev=10.0)
    assert plain.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)), beta=20.0)
    assert plain._flip.tolist() == [False] and np.isclose(plain.p_eff, 0.4, atol=1e-12)


def test_a_carried_pair_past_the_pole_is_pre_flipped_together_and_binds_with_its_predicted_value():
    """A stored pair whose extrapolated 1/lambda changes sign is pre-flipped as one plane at the predicted bound."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    shape = (1, 1, 1, 1, 1, 2)
    w = np.array([1.0 + 0.5j, 0.3 - 0.2j, -0.4 + 0.1j, 0.2 + 0.3j])
    state = _carried_state(
        [0.05 + 6.0j, 0.05 - 6.0j],
        np.column_stack([w, np.conj(w)]),
        shape,
        beta=15.0,
        lam_prev=[0.08 + 5.2j, 0.08 - 5.2j],
        beta_prev=10.0,
    )
    assert tracker.carry_in(state, _expand_identity(shape), beta=20.0)
    assert tracker._flip.tolist() == [True, True] and tracker.q.shape == (4, 2)
    inv_pred = 2 * (0.05 / abs(0.05 + 6.0j) ** 2) - 0.08 / abs(0.08 + 5.2j) ** 2
    predicted = 1.0 / inv_pred
    assert predicted < -1e3 and np.isclose(tracker.p_eff, _MIXING_FLOOR, atol=1e-12)
    lam_pi, damping = tracker._certified_damping()
    assert np.allclose(lam_pi, [0.05 + 6.0j, 0.05 - 6.0j], atol=1e-12) and (damping < 0.0).all()
    assert np.allclose(damping, -min(1.0, 0.5 * 2 * abs(predicted) / abs(predicted + 6.0j) ** 2), atol=1e-12)
    assert any("past the pole" in call[0][0] for call in logger.info.call_args_list)
    # the carried pair itself pins the damping to the floor, so the one measured-spectrum warning is the only one
    assert logger.warning.call_count == 1 and "measured spectrum" in logger.warning.call_args[0][0]
    assert tracker.spectrum_state(shape, 20.0, False, np.complex64)["predicted"].tolist() == [True, True]


def test_a_floor_set_by_the_predicted_spectrum_warns_separately_and_keeps_the_measured_warning():
    """The predicted-spectrum floor warning names its source and leaves the once-only measured warning available."""
    logger = MagicMock()
    tracker = JacobianTracker(0.4, logger=logger)
    tracker._lower_p_eff(_MIXING_FLOOR, predicted=True)
    assert "predicted spectrum" in logger.warning.call_args[0][0] and "p_eff=0.4000" in logger.warning.call_args[0][0]
    assert not tracker._floor_warned and np.isclose(tracker.p_eff, _MIXING_FLOOR, atol=1e-12)
    tracker._lower_p_eff(0.5 * _MIXING_FLOOR)
    assert logger.warning.call_count == 2 and "measured spectrum" in logger.warning.call_args[0][0]
    assert tracker._floor_warned


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
    shape = (1, 1, 1, 1, 1, 2)
    state = _carried_state([0.05], _real_column(0, 2), shape, beta=15.0, lam_prev=[np.nan], beta_prev=10.0)
    assert tracker.carry_in(state, _expand_identity(shape), beta=20.0)
    assert np.isnan(tracker.spectrum_state(shape, 20.0, False, np.complex64)["lam_prev"]).all()
    _certify_direction(tracker, 0.03)
    saved = tracker.spectrum_state(shape, 20.0, True, np.complex64)
    assert (
        np.isclose(saved["lam_prev"][saved["stored"]][0], 0.05, atol=1e-8)
        and saved["beta_prev"][saved["stored"]][0] == 15.0
    )


def test_a_pending_carried_set_without_a_flip_installs_its_map_without_an_event():
    """A pending carried set of stable columns installs its map alone on release, which switches no reflected map."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([0.05], _real_column(0, 2), beta=15.0, lam_prev=[np.nan], beta_prev=10.0)
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)), allow_flip=False, beta=20.0) is False
    assert tracker._pending is not None and not tracker._pending[2].any()
    assert tracker.update(*_stable_pairs(3), allow_flip=True) is False
    assert tracker.q is None and tracker.active
    assert tracker._carried_set is None and tracker._applied is not None


def test_carry_in_without_a_beta_skips_the_rung_extrapolation():
    """Without this run's beta the two-rung extrapolation is skipped and the mode is carried as a stable column."""
    tracker = JacobianTracker(0.4)
    state = _carried_state([0.03], _real_column(0, 2), beta=20.0, lam_prev=[0.09], beta_prev=15.0)
    assert tracker.carry_in(state, _expand_identity((1, 1, 1, 1, 1, 2)))
    assert tracker.q is None and tracker.active and tracker._flip.tolist() == [False]


def test_a_carried_pair_is_pre_flipped_as_one_plane_and_recorded_for_both_members():
    """A falling conjugate pair carried over two rungs is pre-flipped on both members with one two-column reflector."""
    tracker = JacobianTracker(0.4)
    shape = (1, 1, 1, 1, 1, 2)
    w = np.array([1.0 + 0.5j, 0.3 - 0.2j, -0.4 + 0.1j, 0.2 + 0.3j])
    state = _carried_state(
        [0.03 + 0.3j, 0.03 - 0.3j],
        np.column_stack([w, np.conj(w)]),
        shape,
        beta=20.0,
        lam_prev=[0.09 + 0.3j, 0.09 - 0.3j],
        beta_prev=15.0,
    )
    assert tracker.carry_in(state, _expand_identity(shape), beta=25.0)
    assert tracker._flip.tolist() == [True, True] and tracker.q.shape == (4, 2)
    assert tracker.spectrum_state(shape, 25.0, False, np.complex64)["predicted"].tolist() == [True, True]


def test_save_spectrum_writes_the_traces_beside_the_spectrum(tmp_path):
    """The per-iteration traces handed to save_spectrum land in the same file as the certified spectrum."""
    tracker = JacobianTracker(0.3)
    path = str(tmp_path / JACOBIAN_FILE)
    traces = {
        "eigenvalues": np.ones((2, 3), dtype=np.complex128),
        "eigenvalue_residuals": np.zeros((2, 3)),
        "damping": np.array([0.3, 0.2]),
    }
    tracker.save_spectrum(path, (1, 1, 1, 1, 1, 2), 10.0, False, np.complex64, traces=traces)
    loaded = load_spectrum(path)
    assert np.array_equal(loaded["damping"], traces["damping"]) and loaded["eigenvalues"].shape == (2, 3)
    assert loaded["lam_pi"].size == 0 and loaded["eigenvalue_residuals"].shape == (2, 3)

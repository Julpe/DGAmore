# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import contextlib
from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np
import pytest

from dgamore.self_energy import SelfEnergy
from dgamore.nonlocal_sde import apply_mixing_strategy

BETA = 10.0
NB = 1
NK = (1, 1, 1)
NIV = 8
NIV_CORE = 4


def make_sigma(value: complex, nk: tuple[int, int, int] = NK, nb: int = NB, niv: int = NIV_CORE) -> SelfEnergy:
    """Creates a SelfEnergy with constant complex fill value."""
    mat = np.full((*nk, nb, nb, 2 * niv), value, dtype=np.complex64)
    return SelfEnergy(mat, nk, beta=BETA)


def make_sigma_mat(value: complex, nk: tuple[int, int, int] = NK, nb: int = NB, niv: int = NIV_CORE) -> np.ndarray:
    """Returns a raw numpy array with the given fill value in the expected shape."""
    return np.full((*nk, nb, nb, 2 * niv), value, dtype=np.complex64)


def make_pairs(values: list[tuple[complex, complex]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Builds a mixing history of (iterate, proposal) pairs from constant fill-value tuples, oldest first."""
    return [(make_sigma_mat(x), make_sigma_mat(f)) for x, f in values]


def make_config_mock(strategy: str = "linear", mixing: float = 0.5, n_hist: int = 3, niv_core: int = NIV_CORE):
    """Builds a mock config object for patching dgamore.nonlocal_sde.config."""
    cfg = MagicMock()
    cfg.self_consistency.mixing_strategy = strategy
    cfg.self_consistency.mixing = mixing
    cfg.self_consistency.mixing_history_length = n_hist
    cfg.box.niv_core = niv_core
    cfg.logger = MagicMock()
    return cfg


@contextlib.contextmanager
def patch_config(**kwargs):
    """Installs a mock config in dgamore.nonlocal_sde for the duration of the block."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("dgamore.nonlocal_sde.config", make_config_mock(**kwargs))
        yield


def run_pulay(
    sigma_new: SelfEnergy,
    sigma_old: SelfEnergy,
    history_pairs: list,
    mixing: float = 0.5,
    n_hist: int = 3,
    niv_core: int = NIV_CORE,
) -> SelfEnergy:
    with patch_config(strategy="pulay", mixing=mixing, n_hist=n_hist, niv_core=niv_core):
        return apply_mixing_strategy(sigma_new, sigma_old, mixing_history=list(history_pairs))


def run_anderson(
    sigma_new: SelfEnergy,
    sigma_old: SelfEnergy,
    history_pairs: list,
    mixing: float = 0.5,
    n_hist: int = 3,
    niv_core: int = NIV_CORE,
) -> SelfEnergy:
    with patch_config(strategy="anderson", mixing=mixing, n_hist=n_hist, niv_core=niv_core):
        return apply_mixing_strategy(sigma_new, sigma_old, mixing_history=list(history_pairs))


def make_affine_history(j: float, fixed_point: complex, alpha: float, n_pairs: int):
    """Simulates linear-mixing iterations of the affine map S(x) = fp + j*(x - fp) and returns pairs, x_n, S(x_n)."""
    s = lambda x: fixed_point + j * (x - fixed_point)
    x = 0.0 + 0.0j
    pairs = []
    for _ in range(n_pairs):
        pairs.append((x, s(x)))
        x = alpha * s(x) + (1 - alpha) * x
    return make_pairs(pairs), x, s(x)


def test_linear_mixing_basic():
    """x_mixed = alpha * x_new + (1 - alpha) * x_old"""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(0.0)

    with patch_config(strategy="linear", mixing=0.5):
        result = apply_mixing_strategy(sigma_new, sigma_old)

    assert np.allclose(result.mat, 1.0, atol=1e-5)


def test_linear_mixing_alpha_zero():
    """alpha=0 should return sigma_old unchanged."""
    sigma_new = make_sigma(5.0)
    sigma_old = make_sigma(1.0)

    with patch_config(strategy="linear", mixing=0.0):
        result = apply_mixing_strategy(sigma_new, sigma_old)

    assert np.allclose(result.mat, 1.0, atol=1e-5)


def test_linear_mixing_alpha_one():
    """alpha=1 should return sigma_new unchanged."""
    sigma_new = make_sigma(5.0)
    sigma_old = make_sigma(1.0)

    with patch_config(strategy="linear", mixing=1.0):
        result = apply_mixing_strategy(sigma_new, sigma_old)

    assert np.allclose(result.mat, 5.0, atol=1e-5)


def test_linear_mixing_complex():
    """Linear mixing should work correctly for complex-valued self-energies."""
    sigma_new = make_sigma(2.0 + 2.0j)
    sigma_old = make_sigma(0.0 + 0.0j)

    with patch_config(strategy="linear", mixing=0.5):
        result = apply_mixing_strategy(sigma_new, sigma_old)

    assert np.allclose(result.mat, 1.0 + 1.0j, atol=1e-5)


def test_linear_mixing_returns_self_energy_instance():
    """Linear mixing must return a SelfEnergy instance."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(1.0)

    with patch_config(strategy="linear", mixing=0.5):
        result = apply_mixing_strategy(sigma_new, sigma_old)

    assert isinstance(result, SelfEnergy)


def test_pulay_falls_back_to_linear_when_history_too_short():
    """Pulay must fall back to linear while fewer than n_hist + 1 genuine pairs have accumulated."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(0.0)

    with patch_config(strategy="pulay", mixing=0.5, n_hist=5):
        result = apply_mixing_strategy(sigma_new, sigma_old, mixing_history=make_pairs([(0.5, 0.8), (0.8, 1.0)]))

    assert np.allclose(result.mat, 1.0, atol=1e-5)


def test_pulay_falls_back_to_linear_without_history():
    """Pulay must fall back to linear when no mixing history list is given at all."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(0.0)

    with patch_config(strategy="pulay", mixing=0.5, n_hist=3):
        result = apply_mixing_strategy(sigma_new, sigma_old, mixing_history=None)

    assert np.allclose(result.mat, 1.0, atol=1e-5)


def test_pulay_returns_self_energy_instance():
    """Pulay mixing must return a SelfEnergy instance."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(1.0)
    pairs = make_pairs([(0.3, 0.5), (0.5, 0.8), (0.8, 1.0)])

    result = run_pulay(sigma_new, sigma_old, pairs)

    assert isinstance(result, SelfEnergy)


def test_pulay_converged_fixed_point():
    """If all iterates and proposals are identical, Pulay must return the same sigma in the core window."""
    value = 3.0 + 1.0j
    sigma_new = make_sigma(value)
    sigma_old = make_sigma(value)
    pairs = make_pairs([(value, value)] * 3)

    result = run_pulay(sigma_new, sigma_old, pairs)

    niv_dmft = sigma_new.mat.shape[-1] // 2
    assert np.allclose(
        result.mat[..., niv_dmft - NIV_CORE : niv_dmft + NIV_CORE],
        np.full_like(result.mat[..., niv_dmft - NIV_CORE : niv_dmft + NIV_CORE], value),
        atol=1e-4,
    )


def test_pulay_returns_same_object_as_sigma_new():
    """Pulay mixing writes into sigma_new directly and returns it."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(1.0)
    pairs = make_pairs([(0.3, 0.5), (0.5, 0.8), (0.8, 1.0)])

    result = run_pulay(sigma_new, sigma_old, pairs)

    assert result is sigma_new


def test_pulay_does_not_mutate_sigma_old():
    """apply_mixing_strategy must not corrupt sigma_old.mat."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(1.0)
    pairs = make_pairs([(0.3, 0.5), (0.5, 0.8), (0.8, 1.0)])

    original_mat = sigma_old.mat.copy()
    run_pulay(sigma_new, sigma_old, pairs)

    assert np.array_equal(sigma_old.mat, original_mat)


def test_pulay_tails_come_from_sigma_new():
    """Frequencies outside the core window must be taken from sigma_new, not sigma_old."""
    sigma_new = make_sigma(2.0, niv=NIV)
    sigma_old = make_sigma(99.0, niv=NIV)
    pairs = make_pairs([(2.0, 2.0)] * 3)

    result = run_pulay(sigma_new, sigma_old, pairs)

    niv_dmft = sigma_new.mat.shape[-1] // 2
    assert np.allclose(result.mat[..., : niv_dmft - NIV_CORE], 2.0, atol=1e-5)
    assert np.allclose(result.mat[..., niv_dmft + NIV_CORE :], 2.0, atol=1e-5)


def test_pulay_result_shape_matches_sigma_new():
    """The result must have the same shape as sigma_new.mat."""
    sigma_new = make_sigma(1.0)
    sigma_old = make_sigma(0.5)
    pairs = make_pairs([(0.2, 0.3), (0.3, 0.4), (0.4, 0.5)])

    result = run_pulay(sigma_new, sigma_old, pairs)

    assert result.mat.shape == sigma_new.mat.shape


def test_pulay_core_is_finite():
    """The core window of the Pulay result must contain only finite values."""
    sigma_new = make_sigma(1.5 + 0.5j)
    sigma_old = make_sigma(1.0 + 0.3j)
    pairs = make_pairs([(complex(0.4 + 0.1j * i), complex(0.5 + 0.1j * i)) for i in range(3)])

    result = run_pulay(sigma_new, sigma_old, pairs)

    niv_dmft = result.mat.shape[-1] // 2
    core = result.mat[..., niv_dmft - NIV_CORE : niv_dmft + NIV_CORE]
    assert np.all(np.isfinite(core))


def test_pulay_genuine_pairs_solve_affine_map_exactly():
    """With genuine (iterate, proposal) pairs of an affine map, the Pulay step lands on the fixed point."""
    fixed_point = 2.0 + 1.0j
    pairs, x_n, s_x_n = make_affine_history(j=0.5, fixed_point=fixed_point, alpha=0.2, n_pairs=3)
    sigma_new = make_sigma(s_x_n)
    sigma_old = make_sigma(x_n)

    result = run_pulay(sigma_new, sigma_old, pairs, mixing=0.2)

    niv_dmft = result.mat.shape[-1] // 2
    core = result.mat[..., niv_dmft - NIV_CORE : niv_dmft + NIV_CORE]
    assert np.allclose(core, fixed_point, atol=1e-4)


def test_anderson_falls_back_to_linear_when_history_too_short():
    """Anderson must fall back to linear while fewer than n_hist + 1 genuine pairs have accumulated."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(0.0)

    with patch_config(strategy="anderson", mixing=0.5, n_hist=5):
        result = apply_mixing_strategy(sigma_new, sigma_old, mixing_history=make_pairs([(0.5, 0.8), (0.8, 1.0)]))

    assert np.allclose(result.mat, 1.0, atol=1e-5)


def test_anderson_returns_self_energy_instance():
    """Anderson mixing must return a SelfEnergy instance."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(1.0)
    pairs = make_pairs([(0.3, 0.5), (0.5, 0.8), (0.8, 1.0)])

    result = run_anderson(sigma_new, sigma_old, pairs)

    assert isinstance(result, SelfEnergy)


def test_anderson_converged_fixed_point():
    """If all iterates and proposals are identical, Anderson must return the same sigma in the core window."""
    value = 3.0 + 1.0j
    sigma_new = make_sigma(value)
    sigma_old = make_sigma(value)
    pairs = make_pairs([(value, value)] * 3)

    result = run_anderson(sigma_new, sigma_old, pairs)

    niv_dmft = sigma_new.mat.shape[-1] // 2
    assert np.allclose(
        result.mat[..., niv_dmft - NIV_CORE : niv_dmft + NIV_CORE],
        np.full_like(result.mat[..., niv_dmft - NIV_CORE : niv_dmft + NIV_CORE], value),
        atol=1e-4,
    )


def test_anderson_returns_same_object_as_sigma_new():
    """Anderson mixing writes into sigma_new directly and returns it."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(1.0)
    pairs = make_pairs([(0.3, 0.5), (0.5, 0.8), (0.8, 1.0)])

    result = run_anderson(sigma_new, sigma_old, pairs)

    assert result is sigma_new


def test_anderson_does_not_mutate_sigma_old():
    """Anderson must not corrupt sigma_old.mat."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(1.0)
    pairs = make_pairs([(0.3, 0.5), (0.5, 0.8), (0.8, 1.0)])

    original_mat = sigma_old.mat.copy()
    run_anderson(sigma_new, sigma_old, pairs)

    assert np.array_equal(sigma_old.mat, original_mat)


def test_anderson_tails_come_from_sigma_new():
    """Frequencies outside the core window must be taken from sigma_new, not sigma_old."""
    sigma_new = make_sigma(2.0, niv=NIV)
    sigma_old = make_sigma(99.0, niv=NIV)
    pairs = make_pairs([(2.0, 2.0)] * 3)

    result = run_anderson(sigma_new, sigma_old, pairs)

    niv_dmft = sigma_new.mat.shape[-1] // 2
    assert np.allclose(result.mat[..., : niv_dmft - NIV_CORE], 2.0, atol=1e-5)
    assert np.allclose(result.mat[..., niv_dmft + NIV_CORE :], 2.0, atol=1e-5)


def test_anderson_result_shape_matches_sigma_new():
    """The result must have the same shape as sigma_new.mat."""
    sigma_new = make_sigma(1.0)
    sigma_old = make_sigma(0.5)
    pairs = make_pairs([(0.2, 0.3), (0.3, 0.4), (0.4, 0.5)])

    result = run_anderson(sigma_new, sigma_old, pairs)

    assert result.mat.shape == sigma_new.mat.shape


def test_anderson_core_is_finite():
    """The core window of the Anderson result must contain only finite values."""
    sigma_new = make_sigma(1.5 + 0.5j)
    sigma_old = make_sigma(1.0 + 0.3j)
    pairs = make_pairs([(complex(0.4 + 0.1j * i), complex(0.5 + 0.1j * i)) for i in range(3)])

    result = run_anderson(sigma_new, sigma_old, pairs)

    niv_dmft = result.mat.shape[-1] // 2
    core = result.mat[..., niv_dmft - NIV_CORE : niv_dmft + NIV_CORE]
    assert np.all(np.isfinite(core))


def test_anderson_core_differs_from_linear_with_nontrivial_history():
    """With a nontrivial history, Anderson's core window must differ from plain linear mixing."""
    sigma_new = make_sigma(2.0 + 0.5j)
    sigma_old = make_sigma(1.0 + 0.2j)
    pairs = make_pairs([(complex(v + 0.3j * v), complex(1.2 * v + 0.4j * v)) for v in [0.5, 1.0, 1.5]])

    result_anderson = run_anderson(deepcopy(sigma_new), sigma_old, pairs)

    linear_result = 0.5 * sigma_new.mat + 0.5 * sigma_old.mat
    niv_dmft = sigma_new.mat.shape[-1] // 2
    sl = slice(niv_dmft - NIV_CORE, niv_dmft + NIV_CORE)

    assert not np.allclose(
        result_anderson.mat[..., sl], linear_result[..., sl], atol=1e-6
    ), "Anderson with nontrivial history should differ from linear mixing in the core window"


def test_anderson_history_ordering_matters():
    """Passing history oldest-first vs newest-first must produce different Anderson results (ordering matters)."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(1.0)
    pairs = make_pairs([(0.5, 0.9), (1.0, 1.3), (1.5, 1.6)])

    result_forward = run_anderson(deepcopy(sigma_new), sigma_old, pairs)
    result_reversed = run_anderson(deepcopy(sigma_new), sigma_old, list(reversed(pairs)))

    niv_dmft = sigma_new.mat.shape[-1] // 2
    sl = slice(niv_dmft - NIV_CORE, niv_dmft + NIV_CORE)

    assert not np.allclose(
        result_forward.mat[..., sl], result_reversed.mat[..., sl], atol=1e-6
    ), "reversed history should give a different Anderson result"


def test_anderson_genuine_pairs_step_toward_affine_fixed_point():
    """With genuine pairs of an affine map, the damped Anderson step is (1 - alpha)*x_n + alpha*fixed_point."""
    fixed_point = 2.0 + 1.0j
    alpha = 0.2
    pairs, x_n, s_x_n = make_affine_history(j=0.5, fixed_point=fixed_point, alpha=alpha, n_pairs=3)
    sigma_new = make_sigma(s_x_n)
    sigma_old = make_sigma(x_n)

    result = run_anderson(sigma_new, sigma_old, pairs, mixing=alpha)

    niv_dmft = result.mat.shape[-1] // 2
    core = result.mat[..., niv_dmft - NIV_CORE : niv_dmft + NIV_CORE]
    assert np.allclose(core, (1 - alpha) * x_n + alpha * fixed_point, atol=1e-4)


def test_accelerated_mixing_accepts_compressed_input():
    """Compressed sigma_new/sigma_old must give the same accelerated step as decompressed input."""
    fixed_point = 2.0 + 1.0j
    alpha = 0.2
    nk = (2, 2, 1)
    pairs, x_n, s_x_n = make_affine_history(j=0.5, fixed_point=fixed_point, alpha=alpha, n_pairs=3)
    pairs = [(np.tile(x, (*nk, 1, 1, 1)), np.tile(f, (*nk, 1, 1, 1))) for x, f in pairs]
    sigma_new = make_sigma(s_x_n, nk=nk).compress_q_dimension()
    sigma_old = make_sigma(x_n, nk=nk).compress_q_dimension()

    result = run_anderson(sigma_new, sigma_old, pairs, mixing=alpha)

    niv_dmft = result.mat.shape[-1] // 2
    core = result.mat[..., niv_dmft - NIV_CORE : niv_dmft + NIV_CORE]
    assert np.allclose(core, (1 - alpha) * x_n + alpha * fixed_point, atol=1e-4)


def test_current_pair_is_recorded_and_history_trimmed():
    """apply_mixing_strategy appends the (sigma_old, sigma_new) core pair and trims the list to n_hist + 1."""
    sigma_new = make_sigma(2.0, niv=NIV)
    sigma_old = make_sigma(1.0, niv=NIV)
    history = make_pairs([(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)])

    with patch_config(strategy="anderson", mixing=0.5, n_hist=3):
        apply_mixing_strategy(sigma_new, sigma_old, mixing_history=history)

    assert len(history) == 4
    assert np.allclose(history[-1][0], 1.0, atol=1e-6)
    assert np.allclose(history[-1][1], 2.0, atol=1e-6)


def test_recorded_proposal_is_unaffected_by_in_place_mixing():
    """The recorded proposal entry must be a copy, untouched by the in-place core update of sigma_new."""
    fixed_point = 2.0 + 1.0j
    pairs, x_n, s_x_n = make_affine_history(j=0.5, fixed_point=fixed_point, alpha=0.2, n_pairs=3)
    sigma_new = make_sigma(s_x_n)
    history = list(pairs)

    with patch_config(strategy="anderson", mixing=0.2, n_hist=3):
        apply_mixing_strategy(sigma_new, make_sigma(x_n), mixing_history=history)

    assert np.allclose(history[-1][1], s_x_n, atol=1e-6)


def test_fresh_run_local_first_iterate_is_broadcast_over_the_k_grid():
    """A momentum-local starting sigma must enter the history broadcast to the proposal's k-grid, not crash."""
    nk = (2, 2, 1)
    history = []
    sigma_old = make_sigma(1.0, nk=(1, 1, 1))
    with patch_config(strategy="anderson", mixing=0.5, n_hist=3):
        for value in (1.5, 1.8, 2.0, 2.1):
            sigma_old = apply_mixing_strategy(make_sigma(value, nk=nk), sigma_old, mixing_history=history)

    assert history[0][0].shape == history[-1][1].shape
    assert np.all(np.isfinite(sigma_old.mat))


def test_history_cap_partial_window_still_solves_affine_map():
    """With history_cap=2 Anderson runs on the last three pairs and still lands the damped affine step."""
    fixed_point = 2.0 + 1.0j
    alpha = 0.2
    pairs, x_n, s_x_n = make_affine_history(j=0.5, fixed_point=fixed_point, alpha=alpha, n_pairs=3)

    with patch_config(strategy="anderson", mixing=alpha, n_hist=3):
        result = apply_mixing_strategy(make_sigma(s_x_n), make_sigma(x_n), history_cap=2, mixing_history=list(pairs))

    niv_dmft = result.mat.shape[-1] // 2
    core = result.mat[..., niv_dmft - NIV_CORE : niv_dmft + NIV_CORE]
    assert np.allclose(core, (1 - alpha) * x_n + alpha * fixed_point, atol=1e-4)


def test_history_cap_zero_falls_back_to_linear():
    """With history_cap=0 (the reset after the restriction release) the accelerated schemes mix linearly."""
    pairs = make_pairs([(0.6, 0.7), (0.8, 0.9), (1.0, 1.1)])
    with patch_config(strategy="anderson", mixing=0.5, n_hist=3):
        capped = apply_mixing_strategy(
            make_sigma(2.0 + 1.0j), make_sigma(1.0), history_cap=0, mixing_history=list(pairs)
        )
    with patch_config(strategy="linear", mixing=0.5):
        expected = apply_mixing_strategy(make_sigma(2.0 + 1.0j), make_sigma(1.0))
    assert np.allclose(capped.mat, expected.mat, atol=1e-12)


def test_history_cap_zero_still_records_the_pair():
    """A capped iteration must still record its genuine pair, so the history re-arms after the reset."""
    history = make_pairs([(0.6, 0.7), (0.8, 0.9), (1.0, 1.1)])

    with patch_config(strategy="anderson", mixing=0.5, n_hist=3):
        apply_mixing_strategy(make_sigma(2.0), make_sigma(1.0), history_cap=0, mixing_history=history)

    assert len(history) == 4
    assert np.allclose(history[-1][0], 1.0, atol=1e-6)

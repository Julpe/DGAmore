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
from dgamore.mixing import apply_mixing_strategy

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


def make_config_mock(
    strategy: str = "linear",
    mixing: float = 0.5,
    n_hist: int = 3,
    niv_core: int = NIV_CORE,
    nk_tot: int = 1,
    output_path: str = "./",
    previous_sc_path: str = "./",
):
    """Builds a mock config object for patching dgamore.mixing.config."""
    cfg = MagicMock()
    cfg.self_consistency.mixing_strategy = strategy
    cfg.self_consistency.mixing = mixing
    cfg.self_consistency.mixing_history_length = n_hist
    cfg.self_consistency.previous_sc_path = previous_sc_path
    cfg.output.output_path = output_path
    cfg.box.niv_core = niv_core
    cfg.lattice.k_grid.nk_tot = nk_tot
    cfg.logger = MagicMock()
    return cfg


@contextlib.contextmanager
def patch_config(**kwargs):
    """Installs a mock config in dgamore.mixing for the duration of the block."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("dgamore.mixing.config", make_config_mock(**kwargs))
        yield


@contextlib.contextmanager
def patch_file_history(file_sigmas=None, side_effect=None):
    """Replaces the sigma-history file read in dgamore.mixing for the duration of the block."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "dgamore.mixing.read_last_n_sigmas_from_files",
            MagicMock(return_value=file_sigmas, side_effect=side_effect),
        )
        yield


def run_pulay(
    sigma_new: SelfEnergy,
    sigma_old: SelfEnergy,
    sigma_dmft: SelfEnergy,
    file_sigmas: list,
    mixing: float = 0.5,
    n_hist: int = 3,
    niv_core: int = NIV_CORE,
    current_iter: int = 10,
) -> SelfEnergy:
    nk_tot = int(np.prod(NK))
    with (
        patch_config(strategy="pulay", mixing=mixing, n_hist=n_hist, niv_core=niv_core, nk_tot=nk_tot),
        patch_file_history(file_sigmas),
    ):
        return apply_mixing_strategy(sigma_new, sigma_old, sigma_dmft, current_iter=current_iter)


def run_anderson(
    sigma_new: SelfEnergy,
    sigma_old: SelfEnergy,
    sigma_dmft: SelfEnergy,
    file_sigmas: list,
    mixing: float = 0.5,
    n_hist: int = 3,
    niv_core: int = NIV_CORE,
    current_iter: int = 10,
) -> SelfEnergy:
    nk_tot = int(np.prod(NK))
    with (
        patch_config(strategy="anderson", mixing=mixing, n_hist=n_hist, niv_core=niv_core, nk_tot=nk_tot),
        patch_file_history(file_sigmas),
    ):
        return apply_mixing_strategy(sigma_new, sigma_old, sigma_dmft, current_iter=current_iter)


def test_linear_mixing_basic():
    """x_mixed = alpha * x_new + (1 - alpha) * x_old"""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(0.0)
    sigma_dmft = make_sigma(0.0)

    with patch_config(strategy="linear", mixing=0.5):
        result = apply_mixing_strategy(sigma_new, sigma_old, sigma_dmft, current_iter=1)

    assert np.allclose(result.mat, 1.0, atol=1e-5)


def test_linear_mixing_alpha_zero():
    """alpha=0 should return sigma_old unchanged."""
    sigma_new = make_sigma(5.0)
    sigma_old = make_sigma(1.0)
    sigma_dmft = make_sigma(0.0)

    with patch_config(strategy="linear", mixing=0.0):
        result = apply_mixing_strategy(sigma_new, sigma_old, sigma_dmft, current_iter=1)

    assert np.allclose(result.mat, 1.0, atol=1e-5)


def test_linear_mixing_alpha_one():
    """alpha=1 should return sigma_new unchanged."""
    sigma_new = make_sigma(5.0)
    sigma_old = make_sigma(1.0)
    sigma_dmft = make_sigma(0.0)

    with patch_config(strategy="linear", mixing=1.0):
        result = apply_mixing_strategy(sigma_new, sigma_old, sigma_dmft, current_iter=1)

    assert np.allclose(result.mat, 5.0, atol=1e-5)


def test_linear_mixing_complex():
    """Linear mixing should work correctly for complex-valued self-energies."""
    sigma_new = make_sigma(2.0 + 2.0j)
    sigma_old = make_sigma(0.0 + 0.0j)
    sigma_dmft = make_sigma(0.0)

    with patch_config(strategy="linear", mixing=0.5):
        result = apply_mixing_strategy(sigma_new, sigma_old, sigma_dmft, current_iter=1)

    assert np.allclose(result.mat, 1.0 + 1.0j, atol=1e-5)


def test_linear_mixing_returns_self_energy_instance():
    """Linear mixing must return a SelfEnergy instance."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(1.0)
    sigma_dmft = make_sigma(0.0)

    with patch_config(strategy="linear", mixing=0.5):
        result = apply_mixing_strategy(sigma_new, sigma_old, sigma_dmft, current_iter=1)

    assert isinstance(result, SelfEnergy)


def test_pulay_falls_back_to_linear_when_iter_too_small():
    """Pulay mixing must fall back to linear if current_iter <= n_hist."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(0.0)
    sigma_dmft = make_sigma(0.0)

    with patch_config(strategy="pulay", mixing=0.5, n_hist=5):
        result = apply_mixing_strategy(sigma_new, sigma_old, sigma_dmft, current_iter=3)

    assert np.allclose(result.mat, 1.0, atol=1e-5)


def test_pulay_returns_self_energy_instance():
    """Pulay mixing must return a SelfEnergy instance."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(1.0)
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(float(v)) for v in [0.5, 0.8, 1.0]]

    result = run_pulay(sigma_new, sigma_old, sigma_dmft, file_sigmas)

    assert isinstance(result, SelfEnergy)


def test_pulay_converged_fixed_point():
    """If all sigmas are identical, Pulay mixing must return the same sigma in the core window."""
    value = 3.0 + 1.0j
    sigma_new = make_sigma(value)
    sigma_old = make_sigma(value)
    sigma_dmft = make_sigma(value)
    file_sigmas = [make_sigma_mat(value) for _ in range(3)]

    result = run_pulay(sigma_new, sigma_old, sigma_dmft, file_sigmas)

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
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(float(v)) for v in [0.5, 0.8, 1.0]]

    result = run_pulay(sigma_new, sigma_old, sigma_dmft, file_sigmas)

    assert result is sigma_new


def test_pulay_does_not_mutate_sigma_old():
    """apply_mixing_strategy must not corrupt sigma_old.mat."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(1.0)
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(float(v)) for v in [0.5, 0.8, 1.0]]

    original_mat = sigma_old.mat.copy()
    run_pulay(sigma_new, sigma_old, sigma_dmft, file_sigmas)

    assert np.array_equal(sigma_old.mat, original_mat)


def test_pulay_tails_come_from_sigma_new():
    """Frequencies outside the core window must be taken from sigma_new, not sigma_old."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(99.0)
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(2.0) for _ in range(3)]

    result = run_pulay(sigma_new, sigma_old, sigma_dmft, file_sigmas)

    niv_dmft = sigma_new.mat.shape[-1] // 2
    assert np.allclose(result.mat[..., : niv_dmft - NIV_CORE], 2.0, atol=1e-5)
    assert np.allclose(result.mat[..., niv_dmft + NIV_CORE :], 2.0, atol=1e-5)


def test_pulay_result_shape_matches_sigma_new():
    """The result must have the same shape as sigma_new.mat."""
    sigma_new = make_sigma(1.0)
    sigma_old = make_sigma(0.5)
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(float(v)) for v in [0.3, 0.4, 0.5]]

    result = run_pulay(sigma_new, sigma_old, sigma_dmft, file_sigmas)

    assert result.mat.shape == sigma_new.mat.shape


def test_pulay_core_is_finite():
    """The core window of the Pulay result must contain only finite values."""
    sigma_new = make_sigma(1.5 + 0.5j)
    sigma_old = make_sigma(1.0 + 0.3j)
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(complex(0.5 + 0.1j * i)) for i in range(3)]

    result = run_pulay(sigma_new, sigma_old, sigma_dmft, file_sigmas)

    niv_dmft = result.mat.shape[-1] // 2
    core = result.mat[..., niv_dmft - NIV_CORE : niv_dmft + NIV_CORE]
    assert np.all(np.isfinite(core))


def test_anderson_falls_back_to_linear_when_iter_too_small():
    """Anderson must fall back to linear if current_iter <= n_hist."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(0.0)
    sigma_dmft = make_sigma(0.0)

    with patch_config(strategy="anderson", mixing=0.5, n_hist=5):
        result = apply_mixing_strategy(sigma_new, sigma_old, sigma_dmft, current_iter=3)

    assert np.allclose(result.mat, 1.0, atol=1e-5)


def test_anderson_returns_self_energy_instance():
    """Anderson mixing must return a SelfEnergy instance."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(1.0)
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(float(v)) for v in [0.5, 0.8, 1.0]]

    result = run_anderson(sigma_new, sigma_old, sigma_dmft, file_sigmas)

    assert isinstance(result, SelfEnergy)


def test_anderson_converged_fixed_point():
    """If all sigmas are identical, Anderson must return the same sigma in the core window."""
    value = 3.0 + 1.0j
    sigma_new = make_sigma(value)
    sigma_old = make_sigma(value)
    sigma_dmft = make_sigma(value)
    file_sigmas = [make_sigma_mat(value) for _ in range(3)]

    result = run_anderson(sigma_new, sigma_old, sigma_dmft, file_sigmas)

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
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(float(v)) for v in [0.5, 0.8, 1.0]]

    result = run_anderson(sigma_new, sigma_old, sigma_dmft, file_sigmas)

    assert result is sigma_new


def test_anderson_does_not_mutate_sigma_old():
    """Anderson must not corrupt sigma_old.mat."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(1.0)
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(float(v)) for v in [0.5, 0.8, 1.0]]

    original_mat = sigma_old.mat.copy()
    run_anderson(sigma_new, sigma_old, sigma_dmft, file_sigmas)

    assert np.array_equal(sigma_old.mat, original_mat)


def test_anderson_tails_come_from_sigma_new():
    """Frequencies outside the core window must be taken from sigma_new, not sigma_old."""
    sigma_new = make_sigma(2.0)
    sigma_old = make_sigma(99.0)
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(2.0) for _ in range(3)]

    result = run_anderson(sigma_new, sigma_old, sigma_dmft, file_sigmas)

    niv_dmft = sigma_new.mat.shape[-1] // 2
    assert np.allclose(result.mat[..., : niv_dmft - NIV_CORE], 2.0, atol=1e-5)
    assert np.allclose(result.mat[..., niv_dmft + NIV_CORE :], 2.0, atol=1e-5)


def test_anderson_result_shape_matches_sigma_new():
    """The result must have the same shape as sigma_new.mat."""
    sigma_new = make_sigma(1.0)
    sigma_old = make_sigma(0.5)
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(float(v)) for v in [0.3, 0.4, 0.5]]

    result = run_anderson(sigma_new, sigma_old, sigma_dmft, file_sigmas)

    assert result.mat.shape == sigma_new.mat.shape


def test_anderson_core_is_finite():
    """The core window of the Anderson result must contain only finite values."""
    sigma_new = make_sigma(1.5 + 0.5j)
    sigma_old = make_sigma(1.0 + 0.3j)
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(complex(0.5 + 0.1j * i)) for i in range(3)]

    result = run_anderson(sigma_new, sigma_old, sigma_dmft, file_sigmas)

    niv_dmft = result.mat.shape[-1] // 2
    core = result.mat[..., niv_dmft - NIV_CORE : niv_dmft + NIV_CORE]
    assert np.all(np.isfinite(core))


def test_anderson_core_differs_from_linear_with_nontrivial_history():
    """With a nontrivial history, Anderson's core window must differ from plain linear mixing."""
    sigma_new = make_sigma(2.0 + 0.5j)
    sigma_old = make_sigma(1.0 + 0.2j)
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(complex(v + 0.3j * v)) for v in [0.5, 1.0, 1.5]]

    result_anderson = run_anderson(deepcopy(sigma_new), sigma_old, sigma_dmft, file_sigmas)

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
    sigma_dmft = make_sigma(0.0)
    file_sigmas = [make_sigma_mat(float(v)) for v in [0.5, 1.0, 1.5]]

    result_forward = run_anderson(deepcopy(sigma_new), sigma_old, sigma_dmft, file_sigmas)
    result_reversed = run_anderson(deepcopy(sigma_new), sigma_old, sigma_dmft, list(reversed(file_sigmas)))

    niv_dmft = sigma_new.mat.shape[-1] // 2
    sl = slice(niv_dmft - NIV_CORE, niv_dmft + NIV_CORE)

    assert not np.allclose(
        result_forward.mat[..., sl], result_reversed.mat[..., sl], atol=1e-6
    ), "reversed history should give a different Anderson result"


def test_history_cap_zero_falls_back_to_linear():
    """With history_cap=0 the accelerated schemes ignore the on-file history and reproduce plain linear mixing."""
    nk_tot = int(np.prod(NK))
    file_sigmas = [make_sigma_mat(v) for v in (0.7, 0.9, 1.1)]
    with (
        patch_config(strategy="anderson", mixing=0.5, n_hist=3, niv_core=NIV_CORE, nk_tot=nk_tot),
        patch_file_history(file_sigmas),
    ):
        capped = apply_mixing_strategy(
            make_sigma(2.0 + 1.0j), make_sigma(1.0), make_sigma(0.5), current_iter=10, history_cap=0
        )
    with patch_config(strategy="linear", mixing=0.5):
        expected = apply_mixing_strategy(make_sigma(2.0 + 1.0j), make_sigma(1.0), make_sigma(0.5), current_iter=10)
    assert np.allclose(capped.mat, expected.mat, atol=1e-12)


def test_in_memory_history_matches_file_history_for_pulay_and_anderson():
    """An in-memory sigma_history reproduces the file-read path bit-identically for both accelerated schemes."""
    rng = np.random.default_rng(91)
    nk_tot = int(np.prod(NK))
    shape = (nk_tot, 1, 1, 2 * NIV_CORE)
    file_sigmas = [rng.standard_normal(shape) + 1j * rng.standard_normal(shape) for _ in range(3)]
    make = lambda seed: SelfEnergy(
        (np.random.default_rng(seed).standard_normal((*NK, 1, 1, 2 * NIV_CORE + 4)) * (1 + 1j)).astype(np.complex64),
        nk=NK,
        calc_smom=False,
        beta=10.0,
    )
    for runner, strategy in ((run_pulay, "pulay"), (run_anderson, "anderson")):
        ref = runner(make(1), make(2), make(3), file_sigmas)
        with (
            patch_config(strategy=strategy, mixing=0.5, n_hist=3, niv_core=NIV_CORE, nk_tot=nk_tot),
            patch_file_history(side_effect=AssertionError("file read!")),
        ):
            out = apply_mixing_strategy(make(1), make(2), make(3), current_iter=10, sigma_history=list(file_sigmas))
        assert np.array_equal(out.mat, ref.mat)

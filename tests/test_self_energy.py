# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

from unittest.mock import MagicMock

import numpy as np
import pytest

from moldga import config
from moldga.matsubara_frequencies import MFHelper
from moldga.self_energy import SelfEnergy
from moldga.config import sys

sys.beta = 1.0
nk = (4, 4, 1)
niv = 5
mat_decompressed = np.random.rand(*nk, 2, 2, 2 * niv)
mat_compressed = np.random.rand(16, 2, 2, 2 * niv)


def test_initializes_correctly_with_estimated_niv_core():
    self_energy = SelfEnergy(mat_decompressed, estimate_niv_core=True)
    assert self_energy._niv_core >= self_energy._niv_core_min


@pytest.mark.parametrize("has_compressed_q_dimension", [True, False])
def test_n_bands_returns_correct_value(has_compressed_q_dimension):
    mat = mat_decompressed if not has_compressed_q_dimension else mat_compressed
    self_energy = SelfEnergy(mat, nk=nk, has_compressed_q_dimension=has_compressed_q_dimension)
    assert self_energy.n_bands == 2


@pytest.mark.parametrize("has_compressed_q_dimension", [True, False])
def test_fit_smom_returns_correct_shape(has_compressed_q_dimension):
    mat = mat_compressed if has_compressed_q_dimension else mat_decompressed
    self_energy = SelfEnergy(mat, nk=nk, has_compressed_q_dimension=has_compressed_q_dimension)
    smom0, smom1 = self_energy.fit_smom()
    assert smom0.shape == (2, 2)
    assert smom1.shape == (2, 2)


@pytest.mark.parametrize(
    "has_compressed_q_dimension,custom_niv",
    [(True, 100), (True, 200), (True, 300), (False, 100), (False, 200), (False, 300)],
)
def test_fits_smom_algorithm_correctly_with_dummy_data(has_compressed_q_dimension, custom_niv):
    mat = (
        np.random.rand(*nk, 2, 2, 2 * custom_niv) + 1j * np.random.rand(*nk, 2, 2, 2 * custom_niv)
        if not has_compressed_q_dimension
        else np.random.rand(int(np.prod(nk)), 2, 2, 2 * custom_niv)
        + 1j * np.random.rand(int(np.prod(nk)), 2, 2, 2 * custom_niv)
    )
    self_energy = SelfEnergy(mat, nk=nk, has_compressed_q_dimension=has_compressed_q_dimension, full_niv_range=True)
    dummy_smom0 = np.random.rand(2, 2)
    dummy_smom1 = np.random.rand(2, 2)
    vn = 1j * MFHelper.vn(custom_niv, sys.beta)
    dummy_data = (
        (dummy_smom0[..., None] - 1.0 / vn * dummy_smom1[..., None])[None, None, None, ...]
        * np.ones(nk)[..., None, None, None]
        if not has_compressed_q_dimension
        else (dummy_smom0[..., None] - 1.0 / vn * dummy_smom1[..., None])[None, ...]
        * np.ones((int(np.prod(nk)),))[..., None, None, None]
    )
    self_energy.mat = dummy_data  # Assign dummy data to the matrix
    smom0, smom1 = self_energy.fit_smom()
    assert np.allclose(smom0, dummy_smom0, rtol=1e-2)
    assert np.allclose(smom1, dummy_smom1, rtol=1e-2)


def test_fits_smom_correctly_with_edge_case_data():
    self_energy = SelfEnergy(np.zeros_like(mat_decompressed), nk=nk, has_compressed_q_dimension=False)
    smom0, smom1 = self_energy.fit_smom()
    assert np.allclose(smom0, 0)
    assert np.allclose(smom1, 0)


@pytest.mark.parametrize(
    "custom_niv,n_min",
    [
        (10, None),
        (30, None),
        (50, None),
        (10, 0),
        (30, 0),
        (50, 0),
        (10, 10),
        (30, 10),
        (50, 10),
        (10, 20),
        (30, 20),
        (50, 20),
        (10, 50),
        (30, 50),
        (50, 50),
    ],
)
def test_returns_correct_asymptotic_self_energy(custom_niv, n_min):
    self_energy = SelfEnergy(mat_decompressed + 1j * mat_decompressed, nk=nk, has_compressed_q_dimension=False)

    smom0, smom1 = self_energy.fit_smom()
    vn = 1j * MFHelper.vn(niv, sys.beta, shift=n_min if n_min is not None else niv)
    asympt_expected = (smom0[..., None] - 1.0 / vn * smom1[..., None])[None, None, None, ...] * np.ones(nk)[
        ..., None, None, None
    ]

    asympt = self_energy._get_asympt(niv=niv, n_min=n_min)
    assert np.allclose(asympt.mat, asympt_expected, rtol=1e-2)


def test_asympt_returns_self_energy_unchanged_when_core_equals_niv():
    self_energy = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    self_energy._niv_core = self_energy.niv
    result = self_energy.create_with_asympt_up_to_core()
    assert np.allclose(result.mat, self_energy.mat)


def test_asympt_returns_self_energy_unchanged_when_asympt_niv_is_zero():
    self_energy = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    self_energy._get_asympt = lambda niv: SelfEnergy(np.zeros_like(self_energy.mat), nk=nk)
    result = self_energy.create_with_asympt_up_to_core()
    assert np.allclose(result.mat, self_energy.mat)


def test_concatenates_core_and_asymptotic_tail_correctly():
    self_energy = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    self_energy._niv_core = 3
    asympt = self_energy._get_asympt(niv=self_energy.niv)
    result = self_energy.create_with_asympt_up_to_core()
    expected = np.concatenate(
        (
            asympt.mat[..., : asympt.niv - result.niv],
            self_energy.cut_niv(self_energy._niv_core).mat,
            asympt.mat[..., asympt.niv + result.niv :],
        ),
        axis=-1,
    )
    assert np.allclose(result.mat, expected)


def test_handles_tail_edge_case_with_zero_core_niv():
    self_energy = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    self_energy._niv_core = 0
    result = self_energy.create_with_asympt_up_to_core()
    assert result.mat.shape[-1] == self_energy.mat.shape[-1]


def test_appends_asymptotic_tail_correctly_when_niv_is_greater_than_current():
    self_energy = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    asympt = self_energy._get_asympt(niv=10)
    result = self_energy.append_asympt(niv=10)
    expected = np.concatenate(
        (
            asympt.mat[..., : asympt.niv - self_energy.niv],
            self_energy.mat,
            asympt.mat[..., asympt.niv + self_energy.niv :],
        ),
        axis=-1,
    )
    assert np.allclose(result.mat, expected)


def test_append_returns_self_energy_unchanged_when_niv_is_less_than_or_equal_to_current():
    self_energy = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    result = self_energy.append_asympt(niv=self_energy.niv)
    assert np.allclose(result.mat, self_energy.mat)


def test_appends_asymptotic_tail_correctly_with_large_niv():
    self_energy = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    asympt = self_energy._get_asympt(niv=100)
    result = self_energy.append_asympt(niv=100)
    expected = np.concatenate(
        (
            asympt.mat[..., : asympt.niv - self_energy.niv],
            self_energy.mat,
            asympt.mat[..., asympt.niv + self_energy.niv :],
        ),
        axis=-1,
    )
    assert np.allclose(result.mat, expected)


def test_adds_two_self_energy_objects_correctly():
    self_energy1 = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    self_energy2 = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    result = self_energy1 + self_energy2
    assert np.allclose(result.mat, self_energy1.mat + self_energy2.mat)


def test_adds_self_energy_and_numpy_array_correctly():
    self_energy = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    array = np.random.rand(*mat_decompressed.shape)
    result = self_energy + array
    assert np.allclose(result.mat, self_energy.mat + array)


def test_subtracts_two_self_energy_objects_correctly():
    self_energy1 = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    self_energy2 = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    result = self_energy1 - self_energy2
    assert np.allclose(result.mat, self_energy1.mat - self_energy2.mat)


def test_subtracts_self_energy_and_numpy_array_correctly():
    self_energy = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    array = np.random.rand(*mat_decompressed.shape)
    result = self_energy - array
    assert np.allclose(result.mat, self_energy.mat - array)


@pytest.mark.parametrize(
    "has_compressed_q_dimension_1,has_compressed_q_dimension_2",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_adds_and_subtracts_two_self_energy_objects_correctly_with_different_compression(
    has_compressed_q_dimension_1, has_compressed_q_dimension_2
):
    mat1 = mat_compressed if has_compressed_q_dimension_1 else mat_decompressed
    mat2 = mat_compressed if has_compressed_q_dimension_2 else mat_decompressed
    self_energy1 = SelfEnergy(mat1, nk=nk, has_compressed_q_dimension=has_compressed_q_dimension_1)
    self_energy2 = SelfEnergy(mat2, nk=nk, has_compressed_q_dimension=has_compressed_q_dimension_2)
    result1 = self_energy1 + self_energy2
    result2 = self_energy1 - self_energy2
    assert np.allclose(result1.mat, self_energy1.mat + self_energy2.mat)
    assert np.allclose(result2.mat, self_energy1.mat - self_energy2.mat)


def test_concatenates_self_energies_correctly_with_equal_niv():
    self_energy1 = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    self_energy2 = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    result = self_energy1.concatenate_self_energies(self_energy2)
    assert np.allclose(result.mat, self_energy1.mat)


def test_raises_error_when_concatenating_with_smaller_niv():
    self_energy1 = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    smaller_mat = np.random.rand(*nk, 2, 2, 2 * (niv - 1))
    self_energy2 = SelfEnergy(smaller_mat, nk=nk, has_compressed_q_dimension=False)
    with pytest.raises(ValueError, match="Can not concatenate with a self-energy that has less frequencies."):
        self_energy1.concatenate_self_energies(self_energy2)


def test_concatenates_self_energies_correctly_with_larger_niv():
    self_energy1 = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    larger_mat = np.random.rand(*nk, 2, 2, 2 * (niv + 2))
    self_energy2 = SelfEnergy(larger_mat, nk=nk, has_compressed_q_dimension=False)
    result = self_energy1.concatenate_self_energies(self_energy2)
    niv_diff = self_energy2.niv - self_energy1.niv
    expected = np.concatenate(
        (self_energy2.mat[..., :niv_diff], self_energy1.mat, self_energy2.mat[..., niv_diff + 2 * self_energy1.niv :]),
        axis=-1,
    )
    assert np.allclose(result.mat, expected)


@pytest.mark.parametrize("has_compressed_q_dimension", [True, False])
def test_concatenates_self_energies_correctly_with_compression(has_compressed_q_dimension):
    mat1 = mat_compressed if has_compressed_q_dimension else mat_decompressed
    mat2 = mat_compressed if has_compressed_q_dimension else mat_decompressed
    self_energy1 = SelfEnergy(mat1, nk=nk, has_compressed_q_dimension=has_compressed_q_dimension)
    self_energy2 = SelfEnergy(mat2, nk=nk, has_compressed_q_dimension=has_compressed_q_dimension)
    result = self_energy1.concatenate_self_energies(self_energy2)
    assert np.allclose(result.mat, self_energy1.mat)


@pytest.mark.parametrize("has_compressed_q_dimension", [True, False])
def test_fits_polynomial_correctly_with_compression(has_compressed_q_dimension):
    mat = np.random.rand(*nk, 2, 2, 100) if not has_compressed_q_dimension else np.random.rand(16, 2, 2, 100)
    self_energy = SelfEnergy(mat, nk=nk, has_compressed_q_dimension=has_compressed_q_dimension)
    result = self_energy.fit_polynomial(n_fit=5, degree=2)
    assert result.mat.shape == self_energy.compress_q_dimension().mat.shape


def test_fits_polynomial_coefficients_correctly_with_default_parameters():
    mat = np.random.rand(*nk, 2, 2, 100).astype(np.complex128)
    vn = MFHelper.vn(50, sys.beta)
    f_vn = np.random.rand() + np.random.rand() * vn + np.random.rand() * vn**2
    mat = np.full(mat.shape, f_vn + 1j * f_vn)  # Dummy data for testing
    self_energy = SelfEnergy(mat, nk=nk, has_compressed_q_dimension=False)
    result = self_energy.fit_polynomial(n_fit=25, degree=2)
    assert np.allclose(result.mat[0, 0, 0], f_vn + 1j * f_vn, rtol=1e-2, atol=1e6)


@pytest.mark.parametrize("error_margin", [1e-5, 1e-3, 1e-1])
def test_estimates_niv_core_correctly_with_varying_error_margins(error_margin):
    self_energy = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    niv_core = self_energy._estimate_niv_core(err=error_margin)
    assert niv_core >= self_energy._niv_core_min


def test_estimates_niv_core_correctly_with_minimum_core():
    self_energy = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    self_energy._niv_core_min = 10
    niv_core = self_energy._estimate_niv_core()
    assert niv_core == 10


def test_estimates_niv_core_correctly_with_large_asymptotic_difference():
    self_energy = SelfEnergy(mat_decompressed * 10, nk=nk, has_compressed_q_dimension=False)
    niv_core = self_energy._estimate_niv_core()
    assert niv_core == self_energy._niv_core_min


def test_handles_edge_case_with_zero_matrix():
    self_energy = SelfEnergy(np.zeros_like(mat_decompressed), nk=nk, has_compressed_q_dimension=False)
    niv_core = self_energy._estimate_niv_core()
    assert niv_core == self_energy._niv_core_min


def test_handles_edge_case_with_identical_asymptotic_and_matrix():
    self_energy = SelfEnergy(mat_decompressed, nk=nk, has_compressed_q_dimension=False)
    self_energy._get_asympt = lambda niv, n_min: SelfEnergy(self_energy.mat, nk=nk)
    niv_core = self_energy._estimate_niv_core()
    assert niv_core == 20


@pytest.fixture
def self_energy():
    mat = np.zeros((1, 1, 1, 2, 2, 20))
    config.sys.beta = 1.0
    self_energy = SelfEnergy(mat, nk=(1, 1, 1), has_compressed_q_dimension=False)

    self_energy._symmetrize_orbitals = MagicMock()
    self_energy._is_orbitally_symmetrized = MagicMock()
    self_energy.fit_smom = MagicMock()

    return self_energy


def test_symmetrize_orbitals_returns_self_if_already_symmetrized(self_energy):
    orbitals = [1, 2]
    self_energy._is_orbitally_symmetrized.return_value = True

    result = self_energy.symmetrize_orbitals(orbitals)

    assert result is self_energy
    self_energy._is_orbitally_symmetrized.assert_called_once_with(orbitals, (3, 4))
    self_energy._symmetrize_orbitals.assert_not_called()


@pytest.mark.parametrize(
    "shape, expected_axes, compressed",
    [
        ((2, 2, 10), (0, 1), False),  # [o1,o2,v]
        ((3, 2, 2, 10), (1, 2), True),  # [k,o1,o2,v]
        ((2, 2, 2, 2, 2, 10), (3, 4), False),  # [kx,ky,kz,o1,o2,v]
    ],
)
def test_executes_symmetrization_if_not_already_symmetrized(shape, expected_axes, compressed, self_energy):
    se = self_energy
    se.mat = np.zeros(shape)
    se._has_compressed_q_dimension = compressed

    orbitals = [1, 2]
    se._is_orbitally_symmetrized.return_value = False

    assert se._get_orbital_axes() == expected_axes
    _ = self_energy.symmetrize_orbitals(orbitals)

    se._is_orbitally_symmetrized.assert_called_once_with(orbitals, expected_axes)
    se._symmetrize_orbitals.assert_called_once_with(orbitals, expected_axes)


@pytest.mark.parametrize(
    "shape, expected_axes, compressed",
    [
        ((2, 2, 10), (0, 1), False),  # [o1,o2,v]
        ((3, 2, 2, 10), (1, 2), True),  # [k,o1,o2,v]
        ((2, 2, 2, 2, 2, 10), (3, 4), False),  # [kx,ky,kz,o1,o2,v]
    ],
)
def test_does_not_executes_symmetrization_if_already_symmetrized(shape, expected_axes, compressed, self_energy):
    se = self_energy
    se.mat = np.zeros(shape)
    se._has_compressed_q_dimension = compressed

    orbitals = [1, 2]
    se._is_orbitally_symmetrized.return_value = True

    _ = self_energy.symmetrize_orbitals(orbitals)
    assert se._get_orbital_axes() == expected_axes

    se._is_orbitally_symmetrized.assert_called_once_with(orbitals, expected_axes)
    se._symmetrize_orbitals.assert_not_called()

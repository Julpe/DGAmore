# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

from unittest.mock import MagicMock

import numpy as np
import pytest

from dgamore.greens_function import GreensFunction


def test_symmetrize_orbitals_already_symmetrized():
    obj = GreensFunction(np.zeros((2, 2, 10)))

    obj._get_orbital_axes = MagicMock(return_value=(0, 1))
    obj.is_orbitally_symmetrized = MagicMock(return_value=True)
    obj._symmetrize_orbitals = MagicMock()

    orbitals = [1, 2]

    result = obj.symmetrize_orbitals(orbitals)

    assert result is obj
    obj.is_orbitally_symmetrized.assert_called_once_with(orbitals)
    obj._symmetrize_orbitals.assert_not_called()


def test_symmetrize_orbitals_calls_private():
    obj = GreensFunction(np.zeros((2, 2, 10)))

    obj._get_orbital_axes = MagicMock(return_value=(1, 2))
    obj.is_orbitally_symmetrized = MagicMock(return_value=False)
    obj._symmetrize_orbitals = MagicMock(return_value="symmetrized_obj")

    orbitals = [1, 3]

    result = obj.symmetrize_orbitals(orbitals)

    obj.is_orbitally_symmetrized.assert_called_once_with(orbitals)
    obj._symmetrize_orbitals.assert_called_once_with(orbitals, (1, 2))
    assert result == "symmetrized_obj"


def test_is_orbitally_symmetrized_delegates():
    obj = GreensFunction(np.zeros((2, 2, 10)))

    obj._get_orbital_axes = MagicMock(return_value=(3, 4))
    obj._is_orbitally_symmetrized = MagicMock(return_value=True)

    orbitals = np.array([1, 2, 3])

    result = obj.is_orbitally_symmetrized(orbitals)

    obj._is_orbitally_symmetrized.assert_called_once_with(orbitals, (3, 4))
    assert result is True


def test_symmetrize_orbitals_empty_list():
    obj = GreensFunction(np.zeros((2, 2, 10)))

    obj._get_orbital_axes = MagicMock(return_value=(0, 1))
    obj.is_orbitally_symmetrized = MagicMock(return_value=True)
    obj._symmetrize_orbitals = MagicMock()

    orbitals = []

    result = obj.symmetrize_orbitals(orbitals)

    assert result is obj
    obj._symmetrize_orbitals.assert_not_called()


@pytest.fixture
def greens_function():
    mat = np.zeros((1, 1, 1, 2, 2, 20))
    greens_function = GreensFunction(mat)

    greens_function._symmetrize_orbitals = MagicMock()
    greens_function._is_orbitally_symmetrized = MagicMock()
    greens_function.fit_smom = MagicMock()

    return greens_function


@pytest.mark.parametrize(
    "shape, expected_axes, compressed",
    [
        ((2, 2, 10), (0, 1), False),  # [o1,o2,v]
        ((3, 2, 2, 10), (1, 2), True),  # [k,o1,o2,v]
        ((2, 2, 2, 2, 2, 10), (3, 4), False),  # [kx,ky,kz,o1,o2,v]
    ],
)
def test_executes_symmetrization_if_not_already_symmetrized(shape, expected_axes, compressed, greens_function):
    gf = greens_function
    gf.mat = np.zeros(shape)
    gf._has_compressed_q_dimension = compressed

    orbitals = [1, 2]
    gf._is_orbitally_symmetrized.return_value = False

    assert gf._get_orbital_axes() == expected_axes
    _ = greens_function.symmetrize_orbitals(orbitals)

    gf._is_orbitally_symmetrized.assert_called_once_with(orbitals, expected_axes)
    gf._symmetrize_orbitals.assert_called_once_with(orbitals, expected_axes)


@pytest.mark.parametrize(
    "shape, expected_axes, compressed",
    [
        ((2, 2, 10), (0, 1), False),  # [o1,o2,v]
        ((3, 2, 2, 10), (1, 2), True),  # [k,o1,o2,v]
        ((2, 2, 2, 2, 2, 10), (3, 4), False),  # [kx,ky,kz,o1,o2,v]
    ],
)
def test_does_not_executes_symmetrization_if_already_symmetrized(shape, expected_axes, compressed, greens_function):
    gf = greens_function
    gf.mat = np.zeros(shape)
    gf._has_compressed_q_dimension = compressed

    orbitals = [1, 2]
    gf._is_orbitally_symmetrized.return_value = True

    _ = greens_function.symmetrize_orbitals(orbitals)
    assert gf._get_orbital_axes() == expected_axes

    gf._is_orbitally_symmetrized.assert_called_once_with(orbitals, expected_axes)
    gf._symmetrize_orbitals.assert_not_called()

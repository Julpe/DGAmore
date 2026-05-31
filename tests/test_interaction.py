# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import pytest
import numpy as np
from dgamore.interaction import Interaction, LocalInteraction, SpinChannel


def test_localinteraction_adds_correctly():
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    interaction1 = LocalInteraction(mat1)
    interaction2 = LocalInteraction(mat2)
    result = interaction1 + interaction2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


def test_localinteraction_raises_error_on_invalid_permutation():
    mat = np.array([[1, 2], [3, 4]])
    interaction = LocalInteraction(mat)
    with pytest.raises(ValueError, match="Invalid permutation."):
        interaction.permute_orbitals("invalid->permutation")


def test_interaction_handles_channel_transformation():
    mat = np.array([[1, 2], [3, 4]])
    interaction = Interaction(mat, SpinChannel.NONE)
    result = interaction.as_channel(SpinChannel.DENS)
    assert np.allclose(result.mat, 2 * mat, rtol=1e-2)


def test_interaction_raises_error_on_invalid_channel_transformation():
    mat = np.array([[1, 2], [3, 4]])
    interaction = Interaction(mat, SpinChannel.DENS)
    with pytest.raises(ValueError, match="Cannot transform interaction from channel"):
        interaction.as_channel(SpinChannel.MAGN)


def test_interaction_exponentiates_correctly():
    mat = np.array(np.random.rand(2, 2, 2, 2))
    interaction = LocalInteraction(mat)
    result = interaction**2
    assert np.allclose(result.mat, np.einsum("abcd,dcef->abef", mat, mat, optimize=True), rtol=1e-2)


def test_interaction_raises_error_on_invalid_exponentiation():
    mat = np.array([[1, 0], [0, 1]])
    interaction = LocalInteraction(mat)
    with pytest.raises(ValueError, match="Exponentiation of Interaction objects only supports positive powers"):
        interaction**0


def test_localinteraction_handles_identity_permutation():
    mat = np.array([[1, 2], [3, 4]])
    interaction = LocalInteraction(mat)
    result = interaction.permute_orbitals("abcd->abcd")
    assert np.allclose(result.mat, mat, rtol=1e-2)


def test_localinteraction_raises_error_on_invalid_exponentiation_zero():
    mat = np.array([[1, 0], [0, 1]])
    interaction = LocalInteraction(mat)
    with pytest.raises(ValueError, match="Exponentiation of Interaction objects only supports positive powers"):
        interaction**0


def test_interaction_handles_compressed_q_dimension_exponentiation():
    mat = np.random.rand(2, 2, 2, 2, 2)
    interaction = Interaction(mat, has_compressed_q_dimension=True)
    result = interaction**2
    assert result.mat.shape == mat.shape
    assert np.allclose(result.mat, np.einsum("qabcd,qdcef->qabef", mat, mat, optimize=True), rtol=1e-2)


def test_interaction_raises_error_on_invalid_permutation():
    mat = np.random.rand(2, 2, 2, 2, 2)
    interaction = Interaction(mat)
    with pytest.raises(ValueError, match="Invalid permutation."):
        interaction.permute_orbitals("invalid->permutation")


@pytest.mark.parametrize("n", [1, 2, 3])
def test_n_bands_returns_correct_value(n):
    mat = np.random.rand(n, n, n, n)
    interaction = LocalInteraction(mat)
    assert interaction.n_bands == n


u_loc = np.random.rand(2, 2, 2, 2)


@pytest.mark.parametrize(
    "channel, expected_mat",
    [
        (SpinChannel.DENS, 2 * u_loc - np.einsum("abcd->adcb", u_loc, optimize=True)),
        (SpinChannel.MAGN, -np.einsum("abcd->adcb", u_loc, optimize=True)),
        (SpinChannel.SING, u_loc + np.einsum("abcd->adcb", u_loc, optimize=True)),
        (SpinChannel.TRIP, u_loc - np.einsum("abcd->adcb", u_loc, optimize=True)),
    ],
)
def test_transforms_to_correct_channel(channel, expected_mat):
    interaction = LocalInteraction(u_loc, SpinChannel.NONE)
    result = interaction.as_channel(channel)
    assert np.allclose(result.mat, expected_mat, rtol=1e-2)
    assert result.channel == channel


def test_raises_error_when_transforming_from_non_none_channel():
    mat = np.random.rand(2, 2, 2, 2)
    interaction = LocalInteraction(mat, SpinChannel.DENS)
    with pytest.raises(ValueError, match="Cannot transform interaction from channel"):
        interaction.as_channel(SpinChannel.MAGN)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_n_bands_returns_correct_value_with_compressed_q_dimension(n):
    mat = np.random.rand(16, n, n, n, n)
    interaction = Interaction(mat, has_compressed_q_dimension=True)
    assert interaction.n_bands == n


@pytest.mark.parametrize("n", [1, 2, 3])
def test_n_bands_returns_correct_value_without_compressed_q_dimension(n):
    mat = np.random.rand(4, 4, 1, n, n, n, n)
    interaction = Interaction(mat, has_compressed_q_dimension=False)
    assert interaction.n_bands == n


@pytest.mark.parametrize("n", [1, 2, 3])
def test_permute_orbitals_returns_same_object_for_identity_permutation(n):
    mat = np.random.rand(16, n, n, n, n)
    interaction = Interaction(mat)
    result = interaction.permute_orbitals("abcd->abcd")
    assert np.allclose(result.mat, interaction.mat, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_permute_orbitals_applies_correct_permutation_with_compressed_q_dimension(n):
    mat = np.random.rand(16, n, n, n, n)
    interaction = Interaction(mat, has_compressed_q_dimension=True)
    result = interaction.permute_orbitals("abcd->adcb")
    expected = np.einsum("...abcd->...adcb", mat, optimize=True)
    assert np.allclose(result.mat, expected, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_permute_orbitals_applies_correct_permutation_with_decompressed_q_dimension(n):
    mat = np.random.rand(4, 4, 1, n, n, n, n)
    interaction = Interaction(mat, has_compressed_q_dimension=False)
    result = interaction.permute_orbitals("abcd->adcb")
    expected = np.einsum("...abcd->...adcb", mat, optimize=True)
    assert np.allclose(result.mat, expected, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_permute_orbitals_raises_error_on_invalid_permutation(n):
    mat = np.random.rand(4, n, n, n, n)
    interaction = Interaction(mat)
    with pytest.raises(ValueError, match="Invalid permutation."):
        interaction.permute_orbitals("invalid->permutation")


@pytest.mark.parametrize("n", [1, 2, 3])
def test_transforms_to_dens_channel_correctly(n):
    mat = np.random.rand(4, n, n, n, n)
    interaction = Interaction(mat, SpinChannel.NONE)
    result = interaction.as_channel(SpinChannel.DENS)
    assert np.allclose(result.mat, 2 * interaction.mat, rtol=1e-2)
    assert result.channel == SpinChannel.DENS


@pytest.mark.parametrize("n", [1, 2, 3])
def test_transforms_to_magn_channel_correctly(n):
    mat = np.random.rand(4, n, n, n, n)
    interaction = Interaction(mat, SpinChannel.NONE)
    result = interaction.as_channel(SpinChannel.MAGN)
    assert np.allclose(result.mat, 0 * interaction.mat, rtol=1e-2)
    assert result.channel == SpinChannel.MAGN


@pytest.mark.parametrize("n", [1, 2, 3])
def test_transforms_to_sing_channel_correctly(n):
    mat = np.random.rand(4, n, n, n, n)
    interaction = Interaction(mat, SpinChannel.NONE)
    result = interaction.as_channel(SpinChannel.SING)
    assert np.allclose(result.mat, interaction.mat, rtol=1e-2)
    assert result.channel == SpinChannel.SING


@pytest.mark.parametrize("n", [1, 2, 3])
def test_transforms_to_trip_channel_correctly(n):
    mat = np.random.rand(4, n, n, n, n)
    interaction = Interaction(mat, SpinChannel.NONE)
    result = interaction.as_channel(SpinChannel.TRIP)
    assert np.allclose(result.mat, interaction.mat, rtol=1e-2)
    assert result.channel == SpinChannel.TRIP


@pytest.mark.parametrize("n", [1, 2, 3])
def test_raises_error_when_transforming_from_non_none_channel(n):
    mat = np.random.rand(4, n, n, n, n)
    interaction = Interaction(mat, SpinChannel.DENS)
    with pytest.raises(ValueError, match="Cannot transform interaction from channel"):
        interaction.as_channel(SpinChannel.MAGN)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_adds_interaction_with_numpy_array_correctly(n):
    mat1 = np.random.rand(16, n, n, n, n)
    mat2 = np.random.rand(16, n, n, n, n)
    interaction = Interaction(mat1, has_compressed_q_dimension=True)
    result = interaction + mat2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_adds_two_interactions_correctly_1(n):
    mat1 = np.random.rand(16, n, n, n, n)
    mat2 = np.random.rand(16, n, n, n, n)
    interaction1 = Interaction(mat1, has_compressed_q_dimension=True)
    interaction2 = Interaction(mat2, has_compressed_q_dimension=True)
    result = interaction1 + interaction2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_adds_two_interactions_correctly_2(n):
    mat1 = np.random.rand(16, n, n, n, n)
    mat2 = np.random.rand(4, 4, 1, n, n, n, n)
    interaction1 = Interaction(mat1, has_compressed_q_dimension=True)
    interaction2 = Interaction(mat2, has_compressed_q_dimension=False)
    result = interaction1 + interaction2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_adds_two_interactions_correctly_3(n):
    mat1 = np.random.rand(4, 4, 1, n, n, n, n)
    mat2 = np.random.rand(16, n, n, n, n)
    interaction1 = Interaction(mat1, has_compressed_q_dimension=False)
    interaction2 = Interaction(mat2, has_compressed_q_dimension=True)
    result = interaction1 + interaction2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_adds_interaction_with_localinteraction_correctly_if_decompressed(n):
    mat1 = np.random.rand(4, 4, 1, n, n, n, n)
    mat2 = np.random.rand(n, n, n, n)
    interaction = Interaction(mat1, has_compressed_q_dimension=False)
    local_interaction = LocalInteraction(mat2)
    result = interaction + local_interaction
    expected = mat1 + mat2[None, ...]
    assert np.allclose(result.mat, expected, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_adds_interaction_with_localinteraction_correctly_if_compressed(n):
    mat1 = np.random.rand(16, n, n, n, n)
    mat2 = np.random.rand(n, n, n, n)
    interaction = Interaction(mat1, has_compressed_q_dimension=True)
    local_interaction = LocalInteraction(mat2)
    result = interaction + local_interaction
    expected = mat1 + mat2[None, ...]
    assert np.allclose(result.mat, expected, rtol=1e-2)


def test_raises_error_when_adding_unsupported_type():
    mat = np.random.rand(4, 2, 2, 2, 2)
    interaction = Interaction(mat, has_compressed_q_dimension=True)
    with pytest.raises(ValueError, match="Operation .* not supported."):
        interaction + "invalid_type"


def test_adds_two_interactions_using_operator_correctly():
    mat1 = np.random.rand(4, 4, 2, 2)
    mat2 = np.random.rand(4, 4, 2, 2)
    interaction1 = Interaction(mat1)
    interaction2 = Interaction(mat2)
    result = interaction1 + interaction2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


def test_adds_interaction_and_numpy_array_using_operator_correctly():
    mat1 = np.random.rand(4, 4, 2, 2)
    mat2 = np.random.rand(4, 4, 2, 2)
    interaction = Interaction(mat1)
    result = interaction + mat2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


def test_subtracts_two_interactions_using_operator_correctly():
    mat1 = np.random.rand(4, 4, 2, 2)
    mat2 = np.random.rand(4, 4, 2, 2)
    interaction1 = Interaction(mat1)
    interaction2 = Interaction(mat2)
    result = interaction1 - interaction2
    assert np.allclose(result.mat, mat1 - mat2, rtol=1e-2)


def test_subtracts_interaction_and_numpy_array_using_operator_correctly():
    mat1 = np.random.rand(4, 4, 2, 2)
    mat2 = np.random.rand(4, 4, 2, 2)
    interaction = Interaction(mat1)
    result = interaction - mat2
    assert np.allclose(result.mat, mat1 - mat2, rtol=1e-2)


def test_raises_error_when_exponentiating_with_invalid_power():
    mat = np.random.rand(4, 4, 2, 2)
    interaction = Interaction(mat)
    with pytest.raises(ValueError, match="Exponentiation of Interaction objects only supports positive powers"):
        interaction**0

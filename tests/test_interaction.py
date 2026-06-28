# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import pytest
import numpy as np
from dgamore.interaction import Interaction, LocalInteraction, SpinChannel

u_loc = np.random.rand(2, 2, 2, 2)


def test_localinteraction_adds_correctly():
    """Adding two LocalInteractions adds their matrices."""
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    interaction1 = LocalInteraction(mat1)
    interaction2 = LocalInteraction(mat2)
    result = interaction1 + interaction2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


def test_localinteraction_handles_identity_permutation():
    """LocalInteraction.permute_orbitals returns the same matrix for the identity permutation."""
    mat = np.array([[1, 2], [3, 4]])
    interaction = LocalInteraction(mat)
    result = interaction.permute_orbitals("abcd->abcd")
    assert np.allclose(result.mat, mat, rtol=1e-2)


def test_localinteraction_raises_error_on_invalid_permutation():
    """LocalInteraction.permute_orbitals rejects an invalid permutation string."""
    mat = np.array([[1, 2], [3, 4]])
    interaction = LocalInteraction(mat)
    with pytest.raises(ValueError, match="Invalid permutation."):
        interaction.permute_orbitals("invalid->permutation")


@pytest.mark.parametrize("n", [1, 2, 3])
def test_n_bands_returns_correct_value(n):
    """LocalInteraction.n_bands returns the orbital count."""
    mat = np.random.rand(n, n, n, n)
    interaction = LocalInteraction(mat)
    assert interaction.n_bands == n


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
    """LocalInteraction.as_channel builds the dens/magn/sing/trip combinations."""
    interaction = LocalInteraction(u_loc, SpinChannel.NONE)
    result = interaction.as_channel(channel)
    assert np.allclose(result.mat, expected_mat, rtol=1e-2)
    assert result.channel == channel


def test_interaction_exponentiates_correctly():
    """LocalInteraction ** 2 contracts the matrix with itself."""
    mat = np.array(np.random.rand(2, 2, 2, 2))
    interaction = LocalInteraction(mat)
    result = interaction**2
    assert np.allclose(result.mat, np.einsum("abcd,dcef->abef", mat, mat, optimize=True), rtol=1e-2)


def test_interaction_raises_error_on_invalid_exponentiation():
    """LocalInteraction exponentiation rejects non-positive powers."""
    mat = np.array([[1, 0], [0, 1]])
    interaction = LocalInteraction(mat)
    with pytest.raises(ValueError, match="Exponentiation of Interaction objects only supports positive powers"):
        interaction**0


def test_localinteraction_raises_error_on_invalid_exponentiation_zero():
    """LocalInteraction ** 0 raises (only positive powers are supported)."""
    mat = np.array([[1, 0], [0, 1]])
    interaction = LocalInteraction(mat)
    with pytest.raises(ValueError, match="Exponentiation of Interaction objects only supports positive powers"):
        interaction**0


def test_localinteraction_rsub_has_correct_sign():
    """LocalInteraction.__rsub__ computes B - A with the correct sign."""
    mat = np.ones((2, 2, 2, 2)) * 2.0
    u = LocalInteraction(mat)
    other = np.zeros_like(mat)
    result = u.__rsub__(other)  # directly call __rsub__ to test B - A = C
    assert np.allclose(result.mat, -mat, rtol=1e-2)


def test_interaction_handles_channel_transformation():
    """Interaction.as_channel transforms NONE to the dens channel."""
    mat = np.array([[1, 2], [3, 4]])
    interaction = Interaction(mat, SpinChannel.NONE)
    result = interaction.as_channel(SpinChannel.DENS)
    assert np.allclose(result.mat, 2 * mat, rtol=1e-2)


def test_interaction_raises_error_on_invalid_channel_transformation():
    """Interaction.as_channel raises when transforming from a non-NONE channel."""
    mat = np.array([[1, 2], [3, 4]])
    interaction = Interaction(mat, SpinChannel.DENS)
    with pytest.raises(ValueError, match="Cannot transform interaction from channel"):
        interaction.as_channel(SpinChannel.MAGN)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_n_bands_returns_correct_value_with_compressed_q_dimension(n):
    """Interaction.n_bands returns the orbital count for a compressed q dimension."""
    mat = np.random.rand(16, n, n, n, n)
    interaction = Interaction(mat, has_compressed_q_dimension=True)
    assert interaction.n_bands == n


@pytest.mark.parametrize("n", [1, 2, 3])
def test_n_bands_returns_correct_value_without_compressed_q_dimension(n):
    """Interaction.n_bands returns the orbital count for a decompressed q dimension."""
    mat = np.random.rand(4, 4, 1, n, n, n, n)
    interaction = Interaction(mat, has_compressed_q_dimension=False)
    assert interaction.n_bands == n


@pytest.mark.parametrize("n", [1, 2, 3])
def test_permute_orbitals_returns_same_object_for_identity_permutation(n):
    """Interaction.permute_orbitals returns the same matrix for the identity permutation."""
    mat = np.random.rand(16, n, n, n, n)
    interaction = Interaction(mat)
    result = interaction.permute_orbitals("abcd->abcd")
    assert np.allclose(result.mat, interaction.mat, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_permute_orbitals_applies_correct_permutation_with_compressed_q_dimension(n):
    """Interaction.permute_orbitals applies the permutation with a compressed q dimension."""
    mat = np.random.rand(16, n, n, n, n)
    interaction = Interaction(mat, has_compressed_q_dimension=True)
    result = interaction.permute_orbitals("abcd->adcb")
    expected = np.einsum("...abcd->...adcb", mat, optimize=True)
    assert np.allclose(result.mat, expected, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_permute_orbitals_applies_correct_permutation_with_decompressed_q_dimension(n):
    """Interaction.permute_orbitals applies the permutation with a decompressed q dimension."""
    mat = np.random.rand(4, 4, 1, n, n, n, n)
    interaction = Interaction(mat, has_compressed_q_dimension=False)
    result = interaction.permute_orbitals("abcd->adcb")
    expected = np.einsum("...abcd->...adcb", mat, optimize=True)
    assert np.allclose(result.mat, expected, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_permute_orbitals_raises_error_on_invalid_permutation(n):
    """Interaction.permute_orbitals raises for an invalid permutation across band counts."""
    mat = np.random.rand(4, n, n, n, n)
    interaction = Interaction(mat)
    with pytest.raises(ValueError, match="Invalid permutation."):
        interaction.permute_orbitals("invalid->permutation")


def test_interaction_raises_error_on_invalid_permutation():
    """Interaction.permute_orbitals raises for an invalid permutation string."""
    mat = np.random.rand(2, 2, 2, 2, 2)
    interaction = Interaction(mat)
    with pytest.raises(ValueError, match="Invalid permutation."):
        interaction.permute_orbitals("invalid->permutation")


def test_interaction_handles_compressed_q_dimension_exponentiation():
    """Interaction ** 2 contracts per q-point with a compressed q dimension."""
    mat = np.random.rand(2, 2, 2, 2, 2)
    interaction = Interaction(mat, has_compressed_q_dimension=True)
    result = interaction**2
    assert result.mat.shape == mat.shape
    assert np.allclose(result.mat, np.einsum("qabcd,qdcef->qabef", mat, mat, optimize=True), rtol=1e-2)


def test_raises_error_when_exponentiating_with_invalid_power():
    """Interaction exponentiation rejects non-positive powers."""
    mat = np.random.rand(4, 4, 2, 2)
    interaction = Interaction(mat)
    with pytest.raises(ValueError, match="Exponentiation of Interaction objects only supports positive powers"):
        interaction**0


@pytest.mark.parametrize("n", [1, 2, 3])
def test_transforms_to_dens_channel_correctly(n):
    """Interaction.as_channel(DENS) doubles the matrix."""
    mat = np.random.rand(4, n, n, n, n)
    interaction = Interaction(mat, SpinChannel.NONE)
    result = interaction.as_channel(SpinChannel.DENS)
    assert np.allclose(result.mat, 2 * interaction.mat, rtol=1e-2)
    assert result.channel == SpinChannel.DENS


@pytest.mark.parametrize("n", [1, 2, 3])
def test_transforms_to_magn_channel_correctly(n):
    """Interaction.as_channel(MAGN) zeroes the matrix."""
    mat = np.random.rand(4, n, n, n, n)
    interaction = Interaction(mat, SpinChannel.NONE)
    result = interaction.as_channel(SpinChannel.MAGN)
    assert np.allclose(result.mat, 0 * interaction.mat, rtol=1e-2)
    assert result.channel == SpinChannel.MAGN


@pytest.mark.parametrize("n", [1, 2, 3])
def test_transforms_to_sing_channel_correctly(n):
    """Interaction.as_channel(SING) keeps the matrix."""
    mat = np.random.rand(4, n, n, n, n)
    interaction = Interaction(mat, SpinChannel.NONE)
    result = interaction.as_channel(SpinChannel.SING)
    assert np.allclose(result.mat, interaction.mat, rtol=1e-2)
    assert result.channel == SpinChannel.SING


@pytest.mark.parametrize("n", [1, 2, 3])
def test_transforms_to_trip_channel_correctly(n):
    """Interaction.as_channel(TRIP) keeps the matrix."""
    mat = np.random.rand(4, n, n, n, n)
    interaction = Interaction(mat, SpinChannel.NONE)
    result = interaction.as_channel(SpinChannel.TRIP)
    assert np.allclose(result.mat, interaction.mat, rtol=1e-2)
    assert result.channel == SpinChannel.TRIP


@pytest.mark.parametrize("n", [1, 2, 3])
def test_raises_error_when_transforming_from_non_none_channel(n):
    """Interaction.as_channel raises when transforming from a non-NONE channel."""
    mat = np.random.rand(4, n, n, n, n)
    interaction = Interaction(mat, SpinChannel.DENS)
    with pytest.raises(ValueError, match="Cannot transform interaction from channel"):
        interaction.as_channel(SpinChannel.MAGN)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_adds_interaction_with_numpy_array_correctly(n):
    """Adding a numpy array to an Interaction adds elementwise."""
    mat1 = np.random.rand(16, n, n, n, n)
    mat2 = np.random.rand(16, n, n, n, n)
    interaction = Interaction(mat1, has_compressed_q_dimension=True)
    result = interaction + mat2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_adds_two_interactions_correctly_1(n):
    """Adding two compressed Interactions adds their matrices."""
    mat1 = np.random.rand(16, n, n, n, n)
    mat2 = np.random.rand(16, n, n, n, n)
    interaction1 = Interaction(mat1, has_compressed_q_dimension=True)
    interaction2 = Interaction(mat2, has_compressed_q_dimension=True)
    result = interaction1 + interaction2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_adds_two_interactions_correctly_2(n):
    """Adding a compressed and a decompressed Interaction adds their matrices."""
    mat1 = np.random.rand(16, n, n, n, n)
    mat2 = np.random.rand(4, 4, 1, n, n, n, n)
    interaction1 = Interaction(mat1, has_compressed_q_dimension=True)
    interaction2 = Interaction(mat2, has_compressed_q_dimension=False)
    result = interaction1 + interaction2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_adds_two_interactions_correctly_3(n):
    """Adding a decompressed and a compressed Interaction adds their matrices."""
    mat1 = np.random.rand(4, 4, 1, n, n, n, n)
    mat2 = np.random.rand(16, n, n, n, n)
    interaction1 = Interaction(mat1, has_compressed_q_dimension=False)
    interaction2 = Interaction(mat2, has_compressed_q_dimension=True)
    result = interaction1 + interaction2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_adds_interaction_with_localinteraction_correctly_if_decompressed(n):
    """A decompressed Interaction adds a LocalInteraction broadcast over momentum."""
    mat1 = np.random.rand(4, 4, 1, n, n, n, n)
    mat2 = np.random.rand(n, n, n, n)
    interaction = Interaction(mat1, has_compressed_q_dimension=False)
    local_interaction = LocalInteraction(mat2)
    result = interaction + local_interaction
    expected = mat1 + mat2[None, ...]
    assert np.allclose(result.mat, expected, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_adds_interaction_with_localinteraction_correctly_if_compressed(n):
    """A compressed Interaction adds a LocalInteraction broadcast over momentum."""
    mat1 = np.random.rand(16, n, n, n, n)
    mat2 = np.random.rand(n, n, n, n)
    interaction = Interaction(mat1, has_compressed_q_dimension=True)
    local_interaction = LocalInteraction(mat2)
    result = interaction + local_interaction
    expected = mat1 + mat2[None, ...]
    assert np.allclose(result.mat, expected, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2])
def test_adds_localinteraction_on_left_with_interaction_returns_interaction(n):
    """LocalInteraction + Interaction dispatches to Interaction.__radd__ and returns a typed Interaction."""
    v_mat = np.random.rand(16, n, n, n, n)
    u_mat = np.random.rand(n, n, n, n)
    v = Interaction(v_mat, has_compressed_q_dimension=True)
    u = LocalInteraction(u_mat)

    result = u + v

    assert isinstance(result, Interaction)
    assert result.has_compressed_q_dimension
    assert result.current_shape == v_mat.shape
    assert np.allclose(result.mat, u_mat[None, ...] + v_mat, rtol=1e-2)


@pytest.mark.parametrize("n", [1, 2])
def test_subtracts_interaction_from_localinteraction_on_left_returns_interaction(n):
    """LocalInteraction - Interaction dispatches to Interaction.__rsub__ and returns a typed Interaction."""
    v_mat = np.random.rand(16, n, n, n, n)
    u_mat = np.random.rand(n, n, n, n)
    v = Interaction(v_mat, has_compressed_q_dimension=True)
    u = LocalInteraction(u_mat)

    result = u - v

    assert isinstance(result, Interaction)
    assert result.has_compressed_q_dimension
    assert result.current_shape == v_mat.shape
    assert np.allclose(result.mat, u_mat[None, ...] - v_mat, rtol=1e-2)


def test_adds_two_interactions_using_operator_correctly():
    """The + operator adds two Interactions."""
    mat1 = np.random.rand(4, 4, 2, 2)
    mat2 = np.random.rand(4, 4, 2, 2)
    interaction1 = Interaction(mat1)
    interaction2 = Interaction(mat2)
    result = interaction1 + interaction2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


def test_adds_interaction_and_numpy_array_using_operator_correctly():
    """The + operator adds a numpy array to an Interaction."""
    mat1 = np.random.rand(4, 4, 2, 2)
    mat2 = np.random.rand(4, 4, 2, 2)
    interaction = Interaction(mat1)
    result = interaction + mat2
    assert np.allclose(result.mat, mat1 + mat2, rtol=1e-2)


def test_subtracts_two_interactions_using_operator_correctly():
    """The - operator subtracts two Interactions."""
    mat1 = np.random.rand(4, 4, 2, 2)
    mat2 = np.random.rand(4, 4, 2, 2)
    interaction1 = Interaction(mat1)
    interaction2 = Interaction(mat2)
    result = interaction1 - interaction2
    assert np.allclose(result.mat, mat1 - mat2, rtol=1e-2)


def test_subtracts_interaction_and_numpy_array_using_operator_correctly():
    """The - operator subtracts a numpy array from an Interaction."""
    mat1 = np.random.rand(4, 4, 2, 2)
    mat2 = np.random.rand(4, 4, 2, 2)
    interaction = Interaction(mat1)
    result = interaction - mat2
    assert np.allclose(result.mat, mat1 - mat2, rtol=1e-2)


def test_raises_error_when_adding_unsupported_type():
    """Adding an unsupported type to an Interaction raises."""
    mat = np.random.rand(4, 2, 2, 2, 2)
    interaction = Interaction(mat, has_compressed_q_dimension=True)
    with pytest.raises(ValueError, match="Operation .* not supported."):
        interaction + "invalid_type"


def test_nonlocal_interaction_rsub_has_correct_sign():
    """Interaction.__rsub__ computes B - A with the correct sign."""
    mat = np.ones((1, 2, 2, 2, 2)) * 3.0
    v = Interaction(mat, SpinChannel.NONE, (1, 1, 1), has_compressed_q_dimension=True)
    other = np.zeros_like(mat)
    result = v.__rsub__(other)  # directly call __rsub__ to test B - A = C
    assert np.allclose(result.mat, -mat, rtol=1e-2)

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import pytest

from moldga.symmetrize_new import *


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_index2component_general_and_back(num_bands):
    for ind in range(1, 16 * num_bands**4 + 1):
        bandspin, band, spin = index2component_general_4(num_bands, 4, ind)
        ind_back = component2index_general_4(num_bands, list(band), list(spin))
        assert ind_back == ind


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_index2component_general_and_back_raises_if_index_too_large_or_too_small(num_bands):
    with pytest.raises(ValueError):
        bandspin, band, spin = index2component_general_4(num_bands, 4, 16 * num_bands**4 + 1)
        _ = component2index_general_4(num_bands, list(band), list(spin))

    with pytest.raises(ValueError):
        bandspin, band, spin = index2component_general_4(num_bands, 4, 0)
        _ = component2index_general_4(num_bands, list(band), list(spin))


def test_component2index_general_invalid_num_bands():
    with pytest.raises(AssertionError):
        component2index_general_4(0, [0], [0])


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_index2component_band_and_back(num_bands):
    orbs = list(it.product(range(num_bands), repeat=4))

    for orb in orbs:
        ind = component2index_band_4(num_bands, 4, list(orb))
        indices_back = index2component_band_4(num_bands, 4, ind)
        assert indices_back == list(orb)


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components(num_bands):
    result = get_worm_components_4(num_bands)
    if num_bands == 1:
        assert result == [1, 4, 7, 10, 13, 16]
    assert len(result) == 6 * num_bands**4


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_component2index_general_2_returns_int(num_bands):
    result = component2index_general_2(num_bands, [0, 0], [0, 0])
    assert isinstance(result, (int, np.integer))


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_component2index_general_2_index_within_range(num_bands):
    max_index = (2 * num_bands) ** 2
    for b0 in range(num_bands):
        for b1 in range(num_bands):
            for s0, s1 in it.product([0, 1], repeat=2):
                idx = component2index_general_2(num_bands, [b0, b1], [s0, s1])
                assert 1 <= idx <= max_index


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_component2index_general_2_all_indices_unique(num_bands):
    indices = [
        component2index_general_2(num_bands, [b0, b1], [s0, s1])
        for b0 in range(num_bands)
        for b1 in range(num_bands)
        for s0, s1 in it.product([0, 1], repeat=2)
    ]
    assert len(indices) == len(set(indices))


def test_component2index_general_2_invalid_num_bands_raises():
    with pytest.raises(AssertionError):
        component2index_general_2(0, [0, 0], [0, 0])


def test_component2index_general_2_single_band_covers_all_four_indices():
    results = {component2index_general_2(1, [0, 0], [s0, s1]) for s0, s1 in it.product([0, 1], repeat=2)}
    assert results == {1, 2, 3, 4}


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_2_is_sorted(num_bands):
    result = get_worm_components_2(num_bands)
    assert result == sorted(result)


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_2_length(num_bands):
    assert len(get_worm_components_2(num_bands)) == 2 * num_bands**2


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_2_no_duplicates(num_bands):
    result = get_worm_components_2(num_bands)
    assert len(result) == len(set(result))


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_2_all_indices_positive(num_bands):
    assert all(idx >= 1 for idx in get_worm_components_2(num_bands))


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_2_is_sorted(num_bands):
    result = get_worm_components_partial_2(num_bands)
    assert result == sorted(result)


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_2_length(num_bands):
    assert len(get_worm_components_partial_2(num_bands)) == 2 * num_bands


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_2_no_duplicates(num_bands):
    result = get_worm_components_partial_2(num_bands)
    assert len(result) == len(set(result))


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_2_is_subset_of_full(num_bands):
    assert set(get_worm_components_partial_2(num_bands)).issubset(set(get_worm_components_2(num_bands)))


def test_get_worm_components_partial_2_single_band_matches_full():
    assert get_worm_components_partial_2(1) == get_worm_components_2(1)


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_4_is_sorted(num_bands):
    result = get_worm_components_partial_4(num_bands)
    assert result == sorted(result)


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_4_no_duplicates(num_bands):
    result = get_worm_components_partial_4(num_bands)
    assert len(result) == len(set(result))


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_4_is_subset_of_full(num_bands):
    assert set(get_worm_components_partial_4(num_bands)).issubset(set(get_worm_components_4(num_bands)))


def test_get_worm_components_partial_4_single_band_matches_full():
    assert get_worm_components_partial_4(1) == get_worm_components_4(1)


@pytest.mark.parametrize("num_bands", [2, 3, 4])
def test_get_worm_components_partial_4_excludes_ijjj_type_orbitals(num_bands):
    partial_indices = set(get_worm_components_partial_4(num_bands))
    spins = [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]]
    for i, j in it.permutations(range(num_bands), 2):
        for excluded_orb in [[i, j, j, j], [j, i, j, j], [j, j, i, j], [j, j, j, i]]:
            for s in spins:
                idx = int(component2index_general_4(num_bands, excluded_orb, s))
                assert idx not in partial_indices


@pytest.mark.parametrize("num_bands", [2, 3, 4])
def test_get_worm_components_partial_4_strictly_smaller_than_full(num_bands):
    assert len(get_worm_components_partial_4(num_bands)) < len(get_worm_components_4(num_bands))


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_component2index_band_4_roundtrip(num_bands):
    for orb in it.product(range(num_bands), repeat=4):
        idx = component2index_band_4(num_bands, 4, list(orb))
        assert index2component_band_4(num_bands, 4, idx) == list(orb)


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_component2index_band_4_all_indices_unique(num_bands):
    indices = [component2index_band_4(num_bands, 4, list(orb)) for orb in it.product(range(num_bands), repeat=4)]
    assert len(indices) == len(set(indices))


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_component2index_band_4_index_range(num_bands):
    indices = [component2index_band_4(num_bands, 4, list(orb)) for orb in it.product(range(num_bands), repeat=4)]
    assert min(indices) == 1
    assert max(indices) == num_bands**4


def test_component2index_band_4_single_band():
    assert component2index_band_4(1, 4, [0, 0, 0, 0]) == 1
    assert index2component_band_4(1, 4, 1) == [0, 0, 0, 0]


@pytest.mark.parametrize("num_bands", [1, 2, 3])
def test_both_indexing_schemes_produce_correct_number_of_unique_indices(num_bands):
    band_indices = {component2index_band_4(num_bands, 4, list(orb)) for orb in it.product(range(num_bands), repeat=4)}
    spin_indices = {
        component2index_general_4(num_bands, list(orb), [0, 0, 0, 0]) for orb in it.product(range(num_bands), repeat=4)
    }
    assert len(band_indices) == num_bands**4
    assert len(spin_indices) == num_bands**4

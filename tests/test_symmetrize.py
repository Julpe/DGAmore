# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import pytest

from dgamore.symmetrize import *


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_index2component_general_and_back(num_bands):
    """index2component_general and component2index_general_4dims round-trip every valid index."""
    for ind in range(1, 16 * num_bands**4 + 1):
        bandspin, band, spin = index2component_general(num_bands, 4, ind)
        ind_back = component2index_general_4dims(num_bands, list(band), list(spin))
        assert ind_back == ind


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_index2component_general_and_back_raises_if_index_too_large_or_too_small(num_bands):
    """index2component_general raises ValueError for an out-of-range index."""
    with pytest.raises(ValueError):
        bandspin, band, spin = index2component_general(num_bands, 4, 16 * num_bands**4 + 1)
        _ = component2index_general_4dims(num_bands, list(band), list(spin))

    with pytest.raises(ValueError):
        bandspin, band, spin = index2component_general(num_bands, 4, 0)
        _ = component2index_general_4dims(num_bands, list(band), list(spin))


def test_component2index_general_invalid_num_bands():
    """component2index_general_4dims asserts a positive band count."""
    with pytest.raises(AssertionError):
        component2index_general_4dims(0, [0], [0])


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_component2index_general_2_returns_int(num_bands):
    """component2index_general_2dims returns an integer index."""
    result = component2index_general_2dims(num_bands, [0, 0], [0, 0])
    assert isinstance(result, (int, np.integer))


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_component2index_general_2_index_within_range(num_bands):
    """component2index_general_2dims indices stay within [1, (2*num_bands)**2]."""
    max_index = (2 * num_bands) ** 2
    for b0 in range(num_bands):
        for b1 in range(num_bands):
            for s0, s1 in it.product([0, 1], repeat=2):
                idx = component2index_general_2dims(num_bands, [b0, b1], [s0, s1])
                assert 1 <= idx <= max_index


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_component2index_general_2_all_indices_unique(num_bands):
    """component2index_general_2dims assigns a unique index to every band/spin combination."""
    indices = [
        component2index_general_2dims(num_bands, [b0, b1], [s0, s1])
        for b0 in range(num_bands)
        for b1 in range(num_bands)
        for s0, s1 in it.product([0, 1], repeat=2)
    ]
    assert len(indices) == len(set(indices))


def test_component2index_general_2_invalid_num_bands_raises():
    """component2index_general_2dims asserts a positive band count."""
    with pytest.raises(AssertionError):
        component2index_general_2dims(0, [0, 0], [0, 0])


def test_component2index_general_2_single_band_covers_all_four_indices():
    """component2index_general_2dims covers indices 1-4 for a single band."""
    results = {component2index_general_2dims(1, [0, 0], [s0, s1]) for s0, s1 in it.product([0, 1], repeat=2)}
    assert results == {1, 2, 3, 4}


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_index2component_band_and_back(num_bands):
    """component2index_band and index2component_band round-trip every orbital tuple."""
    orbs = list(it.product(range(num_bands), repeat=4))

    for orb in orbs:
        ind = component2index_band(num_bands, 4, list(orb))
        indices_back = index2component_band(num_bands, 4, ind)
        assert indices_back == list(orb)


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_component2index_band_4_roundtrip(num_bands):
    """component2index_band and index2component_band round-trip all 4-orbital tuples."""
    for orb in it.product(range(num_bands), repeat=4):
        idx = component2index_band(num_bands, 4, list(orb))
        assert index2component_band(num_bands, 4, idx) == list(orb)


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_component2index_band_4_all_indices_unique(num_bands):
    """component2index_band assigns a unique index to every 4-orbital tuple."""
    indices = [component2index_band(num_bands, 4, list(orb)) for orb in it.product(range(num_bands), repeat=4)]
    assert len(indices) == len(set(indices))


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_component2index_band_4_index_range(num_bands):
    """component2index_band indices span 1..num_bands**4."""
    indices = [component2index_band(num_bands, 4, list(orb)) for orb in it.product(range(num_bands), repeat=4)]
    assert min(indices) == 1
    assert max(indices) == num_bands**4


def test_component2index_band_4_single_band():
    """component2index_band and index2component_band map the single-band tuple to index 1."""
    assert component2index_band(1, 4, [0, 0, 0, 0]) == 1
    assert index2component_band(1, 4, 1) == [0, 0, 0, 0]


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components(num_bands):
    """get_worm_components_all_4dims returns 6*num_bands**4 components (the known set for one band)."""
    result = get_worm_components_all_4dims(num_bands)
    if num_bands == 1:
        assert result == [1, 4, 7, 10, 13, 16]
    assert len(result) == 6 * num_bands**4


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_2_is_sorted(num_bands):
    """get_worm_components_all_2dims returns a sorted list."""
    result = get_worm_components_all_2dims(num_bands)
    assert result == sorted(result)


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_2_length(num_bands):
    """get_worm_components_all_2dims has length 2*num_bands**2."""
    assert len(get_worm_components_all_2dims(num_bands)) == 2 * num_bands**2


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_2_no_duplicates(num_bands):
    """get_worm_components_all_2dims contains no duplicates."""
    result = get_worm_components_all_2dims(num_bands)
    assert len(result) == len(set(result))


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_2_all_indices_positive(num_bands):
    """get_worm_components_all_2dims indices are all >= 1."""
    assert all(idx >= 1 for idx in get_worm_components_all_2dims(num_bands))


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_2_is_sorted(num_bands):
    """get_worm_components_partial_2dims returns a sorted list."""
    result = get_worm_components_partial_2dims(num_bands)
    assert result == sorted(result)


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_2_length(num_bands):
    """get_worm_components_partial_2dims has length 2*num_bands."""
    assert len(get_worm_components_partial_2dims(num_bands)) == 2 * num_bands


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_2_no_duplicates(num_bands):
    """get_worm_components_partial_2dims contains no duplicates."""
    result = get_worm_components_partial_2dims(num_bands)
    assert len(result) == len(set(result))


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_2_is_subset_of_full(num_bands):
    """get_worm_components_partial_2dims is a subset of the full 2dims set."""
    assert set(get_worm_components_partial_2dims(num_bands)).issubset(set(get_worm_components_all_2dims(num_bands)))


def test_get_worm_components_partial_2_single_band_matches_full():
    """get_worm_components_partial_2dims equals the full set for a single band."""
    assert get_worm_components_partial_2dims(1) == get_worm_components_all_2dims(1)


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_4_is_sorted(num_bands):
    """get_worm_components_partial_4dims returns a sorted list."""
    result = get_worm_components_partial_4dims(num_bands)
    assert result == sorted(result)


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_4_no_duplicates(num_bands):
    """get_worm_components_partial_4dims contains no duplicates."""
    result = get_worm_components_partial_4dims(num_bands)
    assert len(result) == len(set(result))


@pytest.mark.parametrize("num_bands", [1, 2, 3, 4])
def test_get_worm_components_partial_4_is_subset_of_full(num_bands):
    """get_worm_components_partial_4dims is a subset of the full 4dims set."""
    assert set(get_worm_components_partial_4dims(num_bands)).issubset(set(get_worm_components_all_4dims(num_bands)))


def test_get_worm_components_partial_4_single_band_matches_full():
    """get_worm_components_partial_4dims equals the full set for a single band."""
    assert get_worm_components_partial_4dims(1) == get_worm_components_all_4dims(1)


@pytest.mark.parametrize("num_bands", [2, 3, 4])
def test_get_worm_components_partial_4_excludes_ijjj_type_orbitals(num_bands):
    """get_worm_components_partial_4dims excludes i-j-j-j type orbital components."""
    partial_indices = set(get_worm_components_partial_4dims(num_bands))
    spins = [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]]
    for i, j in it.permutations(range(num_bands), 2):
        for excluded_orb in [[i, j, j, j], [j, i, j, j], [j, j, i, j], [j, j, j, i]]:
            for s in spins:
                idx = int(component2index_general_4dims(num_bands, excluded_orb, s))
                assert idx not in partial_indices


@pytest.mark.parametrize("num_bands", [2, 3, 4])
def test_get_worm_components_partial_4_strictly_smaller_than_full(num_bands):
    """get_worm_components_partial_4dims is strictly smaller than the full 4dims set for multiple bands."""
    assert len(get_worm_components_partial_4dims(num_bands)) < len(get_worm_components_all_4dims(num_bands))


@pytest.mark.parametrize("num_bands", [1, 2, 3])
def test_both_indexing_schemes_produce_correct_number_of_unique_indices(num_bands):
    """The band and general indexing schemes both produce num_bands**4 unique indices."""
    band_indices = {component2index_band(num_bands, 4, list(orb)) for orb in it.product(range(num_bands), repeat=4)}
    spin_indices = {
        component2index_general_4dims(num_bands, list(orb), [0, 0, 0, 0])
        for orb in it.product(range(num_bands), repeat=4)
    }
    assert len(band_indices) == num_bands**4
    assert len(spin_indices) == num_bands**4

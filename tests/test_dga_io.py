# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from dgamore import config
from dgamore.dga_io import set_hamiltonian
from dgamore.hamiltonian import Hamiltonian

TEST_DATA = f"{os.path.dirname(os.path.abspath(__file__))}/test_data"
KINETIC = ("t_tp_tpp", [1.0, -0.25, 0.12])


def _set_dmft_interaction(*atoms: tuple[int, float, float, float]) -> None:
    config.logger = MagicMock()
    config.dmft.n_ineq = len(atoms)
    config.dmft.ineq_ordering = list(range(1, len(atoms) + 1))
    config.dmft.n_bands_per_ineq = [a[0] for a in atoms]
    config.sys.n_bands = sum(a[0] for a in atoms)
    config.lattice.interaction.udd_per_ineq = [a[1] for a in atoms]
    config.lattice.interaction.jdd_per_ineq = [a[2] for a in atoms]
    config.lattice.interaction.vdd_per_ineq = [a[3] for a in atoms]


@pytest.mark.parametrize("int_type", ["from_dmft", "FROM_DMFT", ""])
def test_from_dmft_reduces_to_the_plain_hubbard_u_for_one_band(int_type):
    """A single band leaves no orbital pair for J or V, so from_dmft yields the plain Hubbard U."""
    _set_dmft_interaction((1, 3.2, 0.75, 1.7))
    u_loc = set_hamiltonian(*KINETIC, int_type, "").get_local_u()
    assert u_loc.mat.shape == (1, 1, 1, 1)
    assert np.allclose(u_loc.mat, 3.2)
    assert not config.logger.warning.called


def test_from_dmft_builds_the_kanamori_tensor_for_three_bands():
    """from_dmft puts U on aaaa, the inter-orbital V on aabb and Hund's J on abba and abab."""
    _set_dmft_interaction((3, 5.0, 0.75, 3.5))
    u_loc = set_hamiltonian(*KINETIC, "from_dmft", "").get_local_u()
    assert u_loc.mat.shape == (3, 3, 3, 3)
    assert np.allclose(u_loc.mat[0, 0, 0, 0], 5.0)
    assert np.allclose(u_loc.mat[0, 0, 1, 1], 3.5)
    assert np.allclose(u_loc.mat[0, 1, 1, 0], 0.75)
    assert np.allclose(u_loc.mat[0, 1, 0, 1], 0.75)


def test_from_dmft_gives_every_inequivalent_atom_its_own_interaction():
    """The orbital-diagonal block the local SDE slices out carries that atom's own U, J and V, not atom 1's."""
    _set_dmft_interaction((1, 4.0, 0.6, 2.8), (2, 9.0, 1.5, 6.0))
    u_loc = set_hamiltonian(*KINETIC, "from_dmft", "").get_local_u().mat
    assert np.allclose(u_loc[0, 0, 0, 0], 4.0)
    assert np.allclose([u_loc[i, i, i, i] for i in (1, 2)], 9.0)
    assert np.allclose([u_loc[1, 1, 2, 2], u_loc[1, 2, 2, 1]], [6.0, 1.5])


def test_from_dmft_leaves_different_inequivalent_atoms_uncoupled():
    """Orbitals of different atoms share no V or Hund's J, so an atom's block is its own Kanamori tensor."""
    _set_dmft_interaction((1, 4.0, 0.6, 2.8), (2, 4.0, 0.6, 2.8))
    u_loc = set_hamiltonian(*KINETIC, "from_dmft", "").get_local_u().mat
    assert np.allclose([u_loc[0, 0, 1, 1], u_loc[0, 1, 1, 0], u_loc[0, 1, 0, 1]], 0.0)


def test_from_dmft_repeats_an_atom_block_for_every_entry_of_the_ineq_ordering():
    """An atom listed twice in the ordering gets its block twice, with no interaction between the two copies."""
    _set_dmft_interaction((2, 9.0, 1.5, 6.0))
    config.dmft.ineq_ordering = [1, 1]
    config.sys.n_bands = 4
    u_loc = set_hamiltonian(*KINETIC, "from_dmft", "").get_local_u().mat
    assert u_loc.shape == (4, 4, 4, 4)
    assert np.allclose([u_loc[i, i, i, i] for i in range(4)], 9.0)
    assert np.allclose([u_loc[0, 0, 2, 2], u_loc[1, 2, 2, 1]], 0.0)


def test_custom_reads_the_interaction_from_the_umatrix_file():
    """A custom interaction type reads the tensor from the file named in the interaction input."""
    _set_dmft_interaction((2, 8.0, 0.0, 0.0))
    u_loc = set_hamiltonian(*KINETIC, "custom", f"{TEST_DATA}/local_sde/u_matrix.dat").get_local_u()
    assert u_loc.mat.shape == (2, 2, 2, 2)
    assert not config.logger.warning.called


def test_custom_rejects_a_non_string_interaction_input():
    """A custom interaction type with a non-string input is rejected."""
    _set_dmft_interaction((2, 8.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        set_hamiltonian(*KINETIC, "custom", [1, 0, 0, 0])


@pytest.mark.parametrize("int_type", ["one_band_from_dmft", "kanamori_from_dmft", "nonsense"])
def test_unrecognized_interaction_types_fall_back_to_from_dmft(int_type):
    """An interaction type that is neither from_dmft nor custom falls back to from_dmft and says so."""
    _set_dmft_interaction((2, 8.0, 0.0, 0.0))
    u_loc = set_hamiltonian(*KINETIC, int_type, "").get_local_u()
    assert np.array_equal(u_loc.mat, Hamiltonian().kanamori_interaction_d(2, 8.0, 0.0, 0.0).get_local_u().mat)
    assert config.logger.warning.called

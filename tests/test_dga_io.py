# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
import shutil
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from dgamore import config
from dgamore.dga_io import (
    check_p_orbitals_are_uncorrelated,
    correlated_orbital_layout,
    create_output_folders,
    load_from_dmft_file_and_update_config,
    set_hamiltonian,
)
from dgamore.hamiltonian import Hamiltonian
from tests import conftest

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


def _recorded_folder_creations(monkeypatch, do_plotting: bool, do_spectrum: bool, plot_spectrum: bool) -> list:
    """Runs the output-folder setup with no folder present and returns the paths it asked to create."""
    created = []
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    monkeypatch.setattr(os, "makedirs", lambda path: created.append(path))
    config.output.output_path = "/run"
    config.output.sigma_iterates_path = "/run/Sigma_Iterates"
    config.output.plotting_path = "/run/Plots"
    config.output.eliashberg_path = "/run/Eliashberg"
    config.output.do_plotting = do_plotting
    config.ana_cont.do_spectrum_dga = do_spectrum
    config.ana_cont.plot_spectrum = plot_spectrum
    config.eliashberg.perform_eliashberg = False
    create_output_folders()
    return created


def test_create_output_folders_makes_the_plots_folder_for_a_spectrum_plot_without_general_plotting(monkeypatch):
    """A requested spectrum plot needs the plots folder even when the general plotting switch is off."""
    created = _recorded_folder_creations(monkeypatch, do_plotting=False, do_spectrum=True, plot_spectrum=True)

    assert created == ["/run", "/run/Sigma_Iterates", "/run/Plots"]


def test_create_output_folders_skips_the_plots_folder_when_nothing_is_plotted(monkeypatch):
    """Without general plotting and without a spectrum plot only the run folder and its iterate folder are created."""
    created = _recorded_folder_creations(monkeypatch, do_plotting=False, do_spectrum=True, plot_spectrum=False)

    assert created == ["/run", "/run/Sigma_Iterates"]


@pytest.mark.parametrize(
    "ordering, n_d, n_p, n_orbitals, expected",
    [
        ([1], [2], [0], 2, None),
        ([1], [1], [2], 3, [0]),
        ([1, 1], [1], [2], 6, [0, 3]),
        ([1, 2], [2, 1], [1, 0], 4, [0, 1, 3]),
    ],
)
def test_correlated_orbitals_follow_the_w2dynamics_atom_layout(ordering, n_d, n_p, n_orbitals, expected):
    """Every atom holds its d orbitals followed by its p orbitals; no p orbital anywhere gives None."""
    layout = correlated_orbital_layout(ordering, n_d, n_p, n_orbitals)
    assert layout is None if expected is None else layout.tolist() == expected


def test_correlated_orbital_layout_refuses_a_hamiltonian_of_another_size():
    """An H(k) whose orbital count differs from the d + p count of its atoms is refused."""
    with pytest.raises(ValueError, match="orbitals"):
        correlated_orbital_layout([1], [1], [2], 4)


@pytest.mark.parametrize("key", ["udp", "upp", "jdp", "jpp", "vpp", "uppod", "jppod"])
def test_interacting_p_orbitals_are_refused(key):
    """Any interaction on the p orbitals is refused by name."""
    interaction = dict.fromkeys(["udp", "upp", "jdp", "jpp", "vpp", "uppod", "jppod"], 0.0) | {key: 0.5}
    with pytest.raises(NotImplementedError, match=key):
        check_p_orbitals_are_uncorrelated(interaction, np.zeros((3, 3)), np.array([0]))


def test_a_double_counting_outside_the_correlated_block_is_refused():
    """A lattice double counting on a p orbital is refused; one on the d block is accepted."""
    no_interaction = dict.fromkeys(["udp", "upp"], 0.0)
    dc = np.zeros((3, 3))
    dc[0, 0] = -1.2
    check_p_orbitals_are_uncorrelated(no_interaction, dc, np.array([0]))
    dc[1, 1] = 0.3
    with pytest.raises(NotImplementedError, match="double counting"):
        check_p_orbitals_are_uncorrelated(no_interaction, dc, np.array([0]))


def test_a_hk_file_with_uncorrelated_orbitals_keeps_every_orbital(tmp_path):
    """A wannier.hk with more orbitals than the correlated ones is read with the orbital count it is given."""
    _set_dmft_interaction((2, 8.0, 0.0, 0.0))
    config.lattice.nk = (2, 2, 1)
    ek = np.random.default_rng(1).standard_normal((4, 4, 4))
    ek = ek + ek.swapaxes(-1, -2)
    conftest.write_hk(tmp_path / "wide.hk", ek, np.zeros((4, 3)))

    ham = set_hamiltonian("from_wannierHK", str(tmp_path / "wide.hk"), "from_dmft", "", n_orbitals=4)

    assert ham.get_ek().shape == (2, 2, 1, 4, 4)
    assert np.allclose(ham.get_ek().reshape(4, 4, 4), ek, rtol=0, atol=1e-11)


def _two_atom_w2dyn_files(path, n_p: tuple[int, int]) -> None:
    """Writes the two-band end_2_end DMFT input to path as two inequivalent copies of its atom with n_p uncorrelated
    orbitals each, a zero double counting on every orbital and no coupling between atoms or to the p orbitals."""
    folder = f"{TEST_DATA}/end_2_end"
    for name in ("1p-data.hdf5", "g4iw_sym.hdf5"):
        shutil.copyfile(f"{folder}/{name}", path / name)
    n_orbitals = 4 + sum(n_p)
    with h5py.File(path / "1p-data.hdf5", "r+") as file:
        attrs = file[".config"].attrs
        for key in [k for k in attrs if k.startswith("atoms.1.")]:
            attrs["atoms.2." + key[len("atoms.1.") :]] = attrs[key]
        attrs["atoms.1.np"], attrs["atoms.2.np"] = n_p
        file.copy(file["dmft-last/ineq-001"], file["dmft-last"], name="ineq-002")
        del file["dmft-last/dc-latt/value"]
        file["dmft-last/dc-latt/value"] = np.zeros((n_orbitals, 2, n_orbitals, 2))
    with h5py.File(path / "g4iw_sym.hdf5", "r+") as file:
        file.copy(file["ineq-001"], file, name="ineq-002")
    ham, kpoints = Hamiltonian().read_hk_w2k(f"{folder}/wannier.hk")
    ek = ham.get_ek().reshape(-1, 2, 2)
    hk = np.zeros((ek.shape[0], n_orbitals, n_orbitals), dtype=complex)
    second = slice(2 + n_p[0], 4 + n_p[0])
    hk[:, :2, :2] = hk[:, second, second] = ek
    p = [2 + i for i in range(n_p[0])] + [4 + n_p[0] + i for i in range(n_p[1])]
    hk[:, p, p] = -40.0
    conftest.write_hk(path / "wannier.hk", hk, kpoints.T)


def test_two_inequivalent_atoms_with_different_uncorrelated_orbitals_load_their_own_layout_and_interaction(tmp_path):
    """Atoms with one and two p orbitals read from a w2dynamics file give the d orbitals [0, 1, 3, 4] and one U each."""
    _two_atom_w2dyn_files(tmp_path, (1, 2))
    config.logger = MagicMock()
    conftest.create_default_config(config, str(tmp_path))
    config.dmft.n_ineq, config.dmft.ineq_ordering = 2, [1, 2]

    load_from_dmft_file_and_update_config()

    u = config.lattice.hamiltonian.get_local_u().mat
    assert config.dmft.n_p_bands_per_ineq == [1, 2]
    assert config.sys.correlated_orbitals.tolist() == [0, 1, 3, 4]
    assert np.allclose(np.einsum("iiii->i", u), 8.0)

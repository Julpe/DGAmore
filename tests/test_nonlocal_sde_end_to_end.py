# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os

import numpy as np
import pytest

import dgamore.brillouin_zone as bz
from dgamore import config, dga_io, local_sde
from dgamore import nonlocal_sde
from dgamore.brillouin_zone import KnownSymmetries
from dgamore.dga_logger import DgaLogger
from dgamore.dmft_interface import W2dynInterface
from dgamore.hamiltonian import Hamiltonian
from tests import conftest


@pytest.fixture
def setup(monkeypatch):
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/end_2_end"

    comm_mock = conftest.create_comm_mock()

    monkeypatch.setattr("mpi4py.MPI.COMM_WORLD", comm_mock)
    config.logger = DgaLogger(comm_mock, "./")
    conftest.create_default_config(config, folder)
    yield folder, comm_mock


def _create_srvo3_cubic_config(c, f: str):
    """Configures the three-band SrVO3 fixture of folder f on a 12x12x12 auto-symmetry grid."""
    c.box.niw_core = -1
    c.box.niv_core = -1
    c.box.niv_shell = 0
    c.output.do_plotting = False
    c.lattice.nk = (12, 12, 12)
    c.lattice.k_grid = bz.KGrid(c.lattice.nk, [KnownSymmetries.AUTO])
    c.lattice.symmetries = [KnownSymmetries.AUTO]
    c.lattice.type = "from_wannier90"
    c.lattice.interaction_type = "from_dmft"
    c.lattice.er_input = f"{f}/wan_hr.dat"
    c.dmft.input_path = f
    c.dmft.symmetrize_orbitals = [1, 2, 3]
    c.dmft.n_ineq = 1
    c.dmft.ineq_ordering = [1]
    c.dmft.n_bands_per_ineq = []
    c.eliashberg.perform_eliashberg = False
    c.self_consistency.mixing = 1
    c.self_consistency.max_iter = 1
    c.sys.occ_dmft_per_ineq = []


@pytest.fixture
def setup_srvo3_cubic(monkeypatch):
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/srvo3_end2end"

    comm_mock = conftest.create_comm_mock()

    monkeypatch.setattr("mpi4py.MPI.COMM_WORLD", comm_mock)
    config.logger = DgaLogger(comm_mock, "./")
    _create_srvo3_cubic_config(config, folder)
    yield folder, comm_mock


def _srvo3_on_hk(folder: str, path, nk: tuple, e_p: list[float] | None = None, t_pd: float = 0.0) -> np.ndarray:
    """Configures SrVO3 on nk with its t2g dispersion (alphabetical orbitals) written to the .hk file path, plus one p
    orbital per t2g orbital at the levels e_p coupled to it by t_pd when e_p is given; returns the t2g dispersion."""
    config.lattice.nk = nk
    config.lattice.k_grid = bz.KGrid(nk, [KnownSymmetries.AUTO])
    config.lattice.type = "from_wannierHK"
    config.lattice.er_input = str(path)
    ek = Hamiltonian().read_hr_w2k(f"{folder}/wan_hr.dat").get_ek(config.lattice.k_grid)
    ek = ek[..., [1, 0, 2], :][..., [1, 0, 2]].reshape(-1, 3, 3)
    ek.imag[np.abs(ek.imag) < 1e-9] = 0
    hk = ek
    if e_p is not None:
        hk = np.zeros((ek.shape[0], 6, 6), dtype=complex)
        hk[:, :3, :3] = ek
        hk[:, 3:, 3:] = np.diag(e_p)
        hk[:, :3, 3:] = hk[:, 3:, :3] = t_pd * np.eye(3)
    conftest.write_hk(path, hk, config.lattice.k_grid.kmesh.reshape(3, -1).T / (2 * np.pi))
    return ek.reshape(*nk, 3, 3)


def _uncorrelated_orbitals(monkeypatch, n_p: int, n_orbitals: int):
    """Lets the DMFT file report n_p uncorrelated orbitals without interaction or double counting."""
    monkeypatch.setattr(W2dynInterface, "get_np", lambda self, ineq=1: n_p)
    monkeypatch.setattr(W2dynInterface, "get_p_interaction", lambda self, ineq=1: {})
    monkeypatch.setattr(
        W2dynInterface, "get_dc_lattice", lambda self, dmft_iter="dmft-last": np.zeros((n_orbitals, n_orbitals))
    )


def _srvo3_self_energy(comm_mock, folder: str) -> np.ndarray:
    """Loads the configured SrVO3 run, discovers its symmetries and returns the one-shot DGA self-energy array."""
    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())
    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]
    config.output.output_path = folder
    ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
    config.lattice.k_grid.specify_auto_symmetries(ek, orbitals=config.sys.correlated_orbitals)
    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)
    *_, s_loc = local_sde.perform_local_schwinger_dyson(g_dmft, g2_dens, g2_magn, u_loc)
    sigma = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft, s_loc)
    return sigma.decompress_q_dimension().mat


def test_calculates_nonlocal_sde_correctly(setup):
    """The non-local SDE reproduces the reference self-energy."""
    folder, comm_mock = setup

    config.box.niw_core = 20
    config.box.niv_core = 20
    config.box.niv_shell = 10
    config.dmft.symmetrize_orbitals = []

    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]

    config.output.output_path = folder

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)

    *_, s_loc = local_sde.perform_local_schwinger_dyson(g_dmft, g2_dens, g2_magn, u_loc)

    sigma_dga = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft, s_loc)

    sigma_dga_mat = sigma_dga.decompress_q_dimension().cut_niv(50).mat
    sigma_dga_ref = np.load(f"{folder}/sigma_dga.npy")

    assert np.allclose(sigma_dga_mat, sigma_dga_ref, atol=3e-5)


@pytest.mark.parametrize("e_p", [40.0, -40.0])
def test_decoupled_uncorrelated_orbitals_leave_the_self_energy_unchanged(setup, monkeypatch, tmp_path, e_p):
    """Two decoupled orbitals, empty or filled, keep the two-band reference self-energy and add their electrons."""
    folder, comm_mock = setup
    ham, kpoints = Hamiltonian().read_hk_w2k(f"{folder}/wannier.hk")
    ek = ham.get_ek().reshape(-1, 2, 2)
    wide = np.zeros((ek.shape[0], 4, 4), dtype=complex)
    wide[:, :2, :2] = ek
    wide[:, 2, 2] = wide[:, 3, 3] = e_p
    conftest.write_hk(tmp_path / "wide.hk", wide, kpoints.T)
    config.lattice.er_input = str(tmp_path / "wide.hk")
    _uncorrelated_orbitals(monkeypatch, 2, 4)

    config.box.niw_core = 20
    config.box.niv_core = 20
    config.box.niv_shell = 10
    config.dmft.symmetrize_orbitals = []

    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]

    config.output.output_path = folder

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)

    *_, s_loc = local_sde.perform_local_schwinger_dyson(g_dmft, g2_dens, g2_magn, u_loc)

    sigma_dga = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft, s_loc)

    sigma_dga_mat = sigma_dga.decompress_q_dimension().cut_niv(50).mat
    sigma_dga_ref = np.load(f"{folder}/sigma_dga.npy")

    assert config.sys.correlated_orbitals.tolist() == [0, 1]
    assert np.isclose(config.sys.n - 2 * np.trace(config.sys.occ).real, 4.0 if e_p < 0 else 0.0, atol=1e-2)
    assert np.allclose(sigma_dga_mat, sigma_dga_ref, atol=3e-5)


def test_calculates_srvo3_correctly(setup_srvo3_cubic):
    """The SrVO3 cubic run reproduces the cubic point-group symmetry of the self-energy under the three axis swaps."""
    folder_cubic, comm_mock = setup_srvo3_cubic

    g_dmft, s_dmft, g2_dens, g2_magn = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())

    assert g_dmft.is_orbitally_symmetrized([1, 2, 3])
    assert s_dmft.is_orbitally_symmetrized([1, 2, 3])
    assert g2_dens.is_orbitally_symmetrized([1, 2, 3])
    assert g2_magn.is_orbitally_symmetrized([1, 2, 3])

    config.output.output_path = folder_cubic

    ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
    # swap orbitals 0 and 1 in rows and columns because wan_hr.dat orders dxz, dxy, dyz instead of alphabetical
    perm = [1, 0, 2]
    ek = ek[..., perm, :][..., perm]
    ek.imag[np.abs(ek.imag) < 1e-9] = 0
    config.lattice.hamiltonian._ek = ek
    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]

    config.lattice.k_grid.specify_auto_symmetries(ek, atol=1e-12)

    s_dmft_ref = np.load(f"{folder_cubic}/sigma_dmft.npy")
    assert np.allclose(s_dmft_ref, s_dmft_ref)
    g_dmft_ref = np.load(f"{folder_cubic}/g_dmft.npy")
    assert np.allclose(g_dmft_ref, g_dmft_ref)

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)

    *_, s_loc = local_sde.perform_local_schwinger_dyson(g_dmft, g2_dens, g2_magn, u_loc)

    sigma_dga_cubic = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft, s_loc)

    niv = sigma_dga_cubic.current_shape[-1] // 2
    s_cubic = sigma_dga_cubic.compress_q_dimension().mat.reshape(12, 12, 12, 3, 3, 2 * niv)

    s_xy_cub = np.swapaxes(s_cubic, 0, 1)
    s_xz_cub = np.swapaxes(s_cubic, 0, 2)
    s_yz_cub = np.swapaxes(s_cubic, 1, 2)

    atol = 1e-6

    assert np.allclose(s_cubic[..., 0, 0, :], s_xy_cub[..., 0, 0, :], atol=atol), "X_Y_SYM dxy failed"
    assert np.allclose(s_cubic[..., 1, 1, :], s_xy_cub[..., 2, 2, :], atol=atol), "X_Y_SYM dxz<->dyz failed"

    assert np.allclose(s_cubic[..., 1, 1, :], s_xz_cub[..., 1, 1, :], atol=atol), "X_Z_SYM dxz failed"
    assert np.allclose(s_cubic[..., 0, 0, :], s_xz_cub[..., 2, 2, :], atol=atol), "X_Z_SYM dxy<->dyz failed"

    assert np.allclose(s_cubic[..., 2, 2, :], s_yz_cub[..., 2, 2, :], atol=atol), "Y_Z_SYM dyz failed"
    assert np.allclose(s_cubic[..., 0, 0, :], s_yz_cub[..., 1, 1, :], atol=atol), "Y_Z_SYM dxy<->dxz failed"


def test_decoupled_uncorrelated_orbitals_leave_the_srvo3_self_energy_unchanged(
    setup_srvo3_cubic, monkeypatch, tmp_path
):
    """Two filled and one empty decoupled orbital keep the SrVO3 self-energy and zone and add four electrons."""
    folder, comm_mock = setup_srvo3_cubic
    runs = []
    for name, e_p in (("t2g", None), ("t2g_p", [-40.0, 40.0, -40.0])):
        conftest.reset_config()
        _create_srvo3_cubic_config(config, folder)
        _srvo3_on_hk(folder, tmp_path / f"{name}.hk", (4, 4, 4), e_p)
        if e_p is not None:
            _uncorrelated_orbitals(monkeypatch, 3, 6)
        sigma = _srvo3_self_energy(comm_mock, folder)
        runs.append((sigma, config.sys.n - 2 * np.trace(config.sys.occ).real, config.lattice.k_grid.nk_irr))
    (sigma_ref, n_p_ref, irr_ref), (sigma, n_p, irr) = runs

    assert config.sys.correlated_orbitals.tolist() == [0, 1, 2]
    assert irr == irr_ref
    assert np.isclose(n_p - n_p_ref, 4.0, atol=1e-2)
    assert np.allclose(sigma, sigma_ref, atol=1e-5)


def test_hybridized_uncorrelated_orbitals_keep_the_cubic_symmetry_of_srvo3(setup_srvo3_cubic, monkeypatch, tmp_path):
    """A p orbital hybridized with each t2g orbital keeps the cubic zone and the cubic relations of the self-energy."""
    folder, comm_mock = setup_srvo3_cubic
    t2g = _srvo3_on_hk(folder, tmp_path / "t2g_p.hk", (4, 4, 4), [-2.0, -2.0, -2.0], t_pd=0.5)
    _uncorrelated_orbitals(monkeypatch, 3, 6)
    plain = bz.KGrid((4, 4, 4), [KnownSymmetries.AUTO])
    plain.specify_auto_symmetries(t2g)

    s = _srvo3_self_energy(comm_mock, folder)

    assert config.sys.correlated_orbitals.tolist() == [0, 1, 2]
    assert config.lattice.k_grid.nk_irr == plain.nk_irr
    # each axis swap (xy, xz, yz) fixes one t2g orbital (dxy, dxz, dyz order) and exchanges the other two
    for axes, fixed, (a, b) in (((0, 1), 0, (1, 2)), ((0, 2), 1, (0, 2)), ((1, 2), 2, (0, 1))):
        s_swap = np.swapaxes(s, *axes)
        assert np.allclose(s[..., fixed, fixed, :], s_swap[..., fixed, fixed, :], atol=1e-6)
        assert np.allclose(s[..., a, a, :], s_swap[..., b, b, :], atol=1e-6)

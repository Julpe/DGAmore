# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np
import pytest

import dgamore.brillouin_zone as bz
from dgamore import config, dga_io, local_sde
from dgamore import nonlocal_sde
from dgamore.brillouin_zone import KnownSymmetries
from dgamore.dga_logger import DgaLogger
from dgamore.dmft_interface import W2dynInterface
from dgamore.hamiltonian import Hamiltonian
from dgamore.local_four_point import LocalFourPoint
from dgamore.matsubara_frequencies import MFHelper
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


def _flex_and_bare_vertex_ladder(comm_mock, monkeypatch, s_dmft) -> tuple[np.ndarray, np.ndarray]:
    """Returns the FLEX-mode self-energy and the ladder self-energy of Gamma_r = U_r and the dc vertex
    1/2 (U_d + 3 U_m) = U - 2 U_{1432} of the loaded run, both with Sigma_loc = Sigma_DMFT."""
    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)
    niw, niv, niv_full = config.box.niw_core, config.box.niv_core, config.box.niv_full

    def bare_vertex(node_comm, path, channel, transform=None, axes=None, w_window=None):
        if os.path.basename(path) == "f_dc_loc.npy":
            mat, niv_first = u_loc.mat - 2.0 * u_loc.mat.transpose(0, 3, 2, 1), niv_full
        else:
            mat, niv_first = u_loc.as_channel(channel).mat, niv
        box = mat.shape + (niw + 1, 2 * niv_first, 2 * niv)
        return LocalFourPoint(np.broadcast_to(mat[..., None, None, None], box).copy(), channel, 1, 2, False), None

    sys_state = deepcopy(config.sys)
    monkeypatch.setattr(nonlocal_sde, "_load_node_shared_local_vertex", bare_vertex)
    ladder = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft.copy(), s_dmft.copy())

    config.sys = sys_state
    config.misc.do_flex = True
    monkeypatch.setattr(nonlocal_sde, "_load_node_shared_local_vertex", MagicMock(side_effect=AssertionError))
    flex = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft.copy(), s_dmft.copy())
    return flex.mat, ladder.mat


def test_flex_mode_is_the_ladder_with_the_bare_interaction_as_local_vertex(setup, monkeypatch):
    """FLEX mode gives the ladder self-energy of Gamma_r = U_r and the dc vertex 1/2 (U_d + 3 U_m) = U - 2 U_{1432}."""
    folder, comm_mock = setup
    config.box.niw_core, config.box.niv_core, config.box.niv_shell = 6, 6, 4
    config.dmft.symmetrize_orbitals = []
    _, s_dmft, _, _ = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())
    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]
    config.output.output_path = folder

    flex, ladder = _flex_and_bare_vertex_ladder(comm_mock, monkeypatch, s_dmft)

    assert np.allclose(flex, ladder, atol=1e-4 * np.abs(ladder).max())


@pytest.mark.parametrize("mode", ["do_flex", "do_rpa"])
def test_bare_vertex_modes_add_no_noise_removal_term(setup, mode):
    """FLEX and RPA are the self-energy of their kernels alone: a shifted local self-energy changes nothing."""
    folder, comm_mock = setup
    config.box.niw_core, config.box.niv_core, config.box.niv_shell = 6, 6, 4
    config.dmft.symmetrize_orbitals = []
    _, s_dmft, _, _ = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())
    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]
    config.output.output_path = folder
    setattr(config.misc, mode, True)
    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)
    shifted = s_dmft.copy()
    shifted.mat = shifted.mat + 0.3j
    sys_state = deepcopy(config.sys)

    reference = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft.copy(), s_dmft.copy())
    config.sys = sys_state
    out = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft.copy(), shifted)

    assert np.allclose(out.mat, reference.mat, atol=1e-6 * np.abs(reference.mat).max())


def test_rpa_mode_starts_from_the_non_interacting_lattice_at_the_dmft_filling(setup, monkeypatch):
    """RPA mode's one proposal sees a zero self-energy at the mu where the bare lattice holds the DMFT filling."""
    folder, comm_mock = setup
    config.box.niw_core, config.box.niv_core, config.box.niv_shell = 6, 6, 4
    config.dmft.symmetrize_orbitals = []
    _, s_dmft, _, _ = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())
    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]
    config.output.output_path = folder
    config.misc.do_rpa = True
    config.self_consistency.max_iter = 1
    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)
    proposal, seen = nonlocal_sde.calculate_sigma_proposal, []
    monkeypatch.setattr(
        nonlocal_sde,
        "calculate_sigma_proposal",
        lambda sigma_in, mu, *a, **k: seen.append((sigma_in.mat.copy(), mu, config.sys.occ.copy()))
        or proposal(sigma_in, mu, *a, **k),
    )

    sigma = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft.copy(), s_dmft.copy())

    ((sigma_in, mu, occ),) = seen
    nb = config.sys.n_bands
    eps = np.linalg.eigvalsh(config.lattice.hamiltonian.get_ek().reshape(-1, nb, nb))
    n_fermi = 2 * np.mean(np.sum(0.5 * (1 - np.tanh(0.5 * config.sys.beta * (eps - mu))), axis=-1))
    assert not np.any(sigma_in)
    assert mu == config.sys.mu_non_interacting and np.isclose(n_fermi, config.sys.n, atol=1e-10)
    assert np.isclose(2 * np.trace(occ).real, config.sys.n, atol=1e-4)
    assert np.all(np.isfinite(sigma.mat)) and np.any(sigma.mat)


def _rpa_run(folder: str, comm_mock, monkeypatch, max_iter: int, mixing: float) -> tuple[list, object]:
    """Runs RPA mode on the two-band fixture (box 6/6/4) and returns, per proposal, the iterate it was handed and the
    occupations it saw, with the returned self-energy."""
    config.box.niw_core, config.box.niv_core, config.box.niv_shell = 6, 6, 4
    config.dmft.symmetrize_orbitals = []
    _, s_dmft, _, _ = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())
    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]
    config.output.output_path = folder
    config.misc.do_rpa = True
    config.self_consistency.max_iter, config.self_consistency.mixing = max_iter, mixing
    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)
    proposal, seen = nonlocal_sde.calculate_sigma_proposal, []
    monkeypatch.setattr(
        nonlocal_sde,
        "calculate_sigma_proposal",
        lambda sigma_in, *a, **k: seen.append(
            (sigma_in.compress_q_dimension().mat.copy(), config.sys.occ.copy(), config.sys.occ_k.copy())
        )
        or proposal(sigma_in, *a, **k),
    )
    sigma = nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft.copy(), s_dmft.copy())
    return seen, sigma


def _shell_moment(mat: np.ndarray, niv_core: int) -> np.ndarray:
    """The coefficient s_1 of a shell c(k) - s_1 / iv, read from every positive shell frequency against the last one,
    shape [k, o1, o2, v]: a shell of that form gives the same value at every k and v."""
    niv = mat.shape[-1] // 2
    iv = 1j * MFHelper.vn(niv, config.sys.beta)[niv + niv_core :]
    shell = mat[..., niv + niv_core :]
    return -(shell[..., :-1] - shell[..., -1:]) / (1.0 / iv[:-1] - 1.0 / iv[-1])


def test_rpa_mode_pads_its_proposal_with_the_hartree_fock_term_and_its_own_tail(setup, monkeypatch):
    """RPA mode carries no DMFT self-energy: beyond the core box its proposal is Sigma_HF(k) - Sigma_1 / iv, with the
    Hartree-Fock term of the occupations the proposal saw and one k-independent Sigma_1."""
    folder, comm_mock = setup
    ((_, occ, occ_k),), sigma = _rpa_run(folder, comm_mock, monkeypatch, 1, 1.0)

    hartree, fock = nonlocal_sde.get_hartree_fock(
        config.lattice.hamiltonian.get_local_u(), config.lattice.hamiltonian.get_vq(config.lattice.k_grid), occ, occ_k
    )
    mat, niv, niv_core = sigma.compress_q_dimension().mat, sigma.niv, config.box.niv_core
    iv = 1j * MFHelper.vn(niv, config.sys.beta)[niv + niv_core :]
    sigma_1 = ((hartree + fock)[..., 0][..., None] - mat[..., niv + niv_core :]) * iv
    assert np.abs(sigma_1).max() > 1e-3
    assert np.allclose(sigma_1, sigma_1[:1, ..., :1], atol=1e-4 * np.abs(sigma_1).max())


def test_self_consistent_rpa_mode_keeps_its_own_tail_through_the_mixing(setup, monkeypatch):
    """The second iteration of a self-consistent RPA run starts from the mixed iterate, whose shell is again
    c(k) - s_1 / iv with one nonzero s_1: no DMFT self-energy enters through the mixing."""
    folder, comm_mock = setup
    ((first, *_), (second, *_)), _ = _rpa_run(folder, comm_mock, monkeypatch, 2, 0.5)

    s_1 = _shell_moment(second, config.box.niv_core)
    assert not np.any(first)
    assert np.abs(s_1).max() > 1e-3
    assert np.allclose(s_1, s_1[:1, ..., :1], atol=1e-4 * np.abs(s_1).max())


def test_rpa_mode_is_the_flex_proposal_on_the_non_interacting_lattice(setup_srvo3_cubic, monkeypatch, tmp_path):
    """Three Kanamori orbitals: RPA mode's one proposal is the FLEX proposal of its own non-interacting start on the
    core box (beyond it FLEX keeps the DMFT self-energy and RPA its own tail)."""
    folder, comm_mock = setup_srvo3_cubic
    _srvo3_on_hk(folder, tmp_path / "t2g.hk", (4, 4, 4))
    config.box.niw_core, config.box.niv_core, config.box.niv_shell = 6, 6, 4
    _, s_dmft, _, _ = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())
    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]
    config.output.output_path = str(tmp_path)
    config.lattice.k_grid.specify_auto_symmetries(config.lattice.hamiltonian.get_ek(config.lattice.k_grid))
    config.misc.do_rpa = True
    config.self_consistency.max_iter = 1
    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.k_grid)
    proposal, seen = nonlocal_sde.calculate_sigma_proposal, []

    def flex_then_rpa(sigma_in, *args, **kwargs):
        config.misc.do_rpa, config.misc.do_flex = False, True
        flex = proposal(sigma_in.copy(), *args, **kwargs)
        config.misc.do_rpa, config.misc.do_flex = True, False
        rpa = proposal(sigma_in, *args, **kwargs)
        seen.append((flex.cut_niv(config.box.niv_core).mat, rpa.cut_niv(config.box.niv_core).mat))
        return rpa

    monkeypatch.setattr(nonlocal_sde, "calculate_sigma_proposal", flex_then_rpa)

    nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft.copy(), s_dmft.copy())

    ((flex, rpa),) = seen
    assert np.allclose(rpa, flex, atol=1e-6 * np.abs(flex).max())


def test_flex_mode_is_the_bare_vertex_ladder_for_kanamori_orbitals(setup_srvo3_cubic, monkeypatch, tmp_path):
    """Three Kanamori orbitals (U = 5, U' = 3.5, J = 0.75), whose inter-orbital terms the two-band fixture lacks: FLEX
    mode is still the ladder of Gamma_r = U_r and the dc vertex U - 2 U_{1432}."""
    folder, comm_mock = setup_srvo3_cubic
    _srvo3_on_hk(folder, tmp_path / "t2g.hk", (4, 4, 4))
    config.box.niw_core, config.box.niv_core, config.box.niv_shell = 6, 6, 4
    _, s_dmft, _, _ = tuple(x[0] for x in dga_io.load_from_dmft_file_and_update_config())
    config.sys.occ_dmft = config.sys.occ_dmft_per_ineq[0]
    config.output.output_path = str(tmp_path)
    config.lattice.k_grid.specify_auto_symmetries(config.lattice.hamiltonian.get_ek(config.lattice.k_grid))

    flex, ladder = _flex_and_bare_vertex_ladder(comm_mock, monkeypatch, s_dmft)

    u = config.lattice.hamiltonian.get_local_u().mat
    assert np.isclose(u[0, 0, 1, 1], 3.5) and np.isclose(u[0, 1, 1, 0], 0.75)
    assert np.allclose(flex, ladder, atol=1e-4 * np.abs(ladder).max())


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

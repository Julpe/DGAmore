import numpy as np
import pytest
import moldga.dga_io as dga_io


class DummyW2dynFile:
    def __init__(self, *args, **kwargs):
        pass

    def get_beta(self):
        return 10.0

    def get_mu(self):
        return 0.5

    def get_nd(self):
        return 1

    def get_np(self):
        return 1

    def get_totdens(self):
        return 1.0

    # interactions (not relevant but required)
    def get_udd(self):
        return 1.0

    def get_udp(self):
        return 1.0

    def get_upp(self):
        return 1.0

    def get_uppod(self):
        return 1.0

    def get_jdd(self):
        return 1.0

    def get_jdp(self):
        return 1.0

    def get_jpp(self):
        return 1.0

    def get_jppod(self):
        return 1.0

    def get_vdd(self):
        return 1.0

    def get_vpp(self):
        return 1.0

    # distinguish ineq=1 and ineq=2 clearly
    def get_rho1(self, ineq=1, **kwargs):
        val = 1.0 if ineq == 1 else 3.0
        return val * np.ones((2, 2, 2, 2))

    def get_giw(self, ineq=1, **kwargs):
        val = 2.0 if ineq == 1 else 6.0
        return val * np.ones((2, 2, 40))

    def get_siw(self, ineq=1, **kwargs):
        val = 5.0 if ineq == 1 else 7.0
        return val * np.ones((2, 2, 40))

    def get_dc(self, ineq=1, **kwargs):
        val = 1.0 if ineq == 1 else 3.0
        return val * np.ones((2, 2))

    def close(self):
        pass


class DummyW2dynG4iwFile:
    def __init__(self, *args, **kwargs):
        pass

    def read_g2_full_multiband(self, n_bands, ineq=1, **kwargs):
        val = 10.0 if ineq == 1 else 20.0
        return val * np.ones((n_bands, n_bands, n_bands, n_bands, 5, 6, 6))

    def close(self):
        pass


class DummyLocalFourPoint:
    def __init__(self, mat, channel=None):
        self.mat = mat
        self.niw = mat.shape[4]
        self.niv = mat.shape[5]

    def cut_niw_and_niv(self, *args):
        return self

    def symmetrize_orbitals(self, *args):
        return self

    def symmetrize_v_vp(self):
        return self


class DummyGreensFunction:
    def __init__(self, data):
        self.data = data

    def symmetrize_orbitals(self, *args):
        return self


class DummySelfEnergy:
    def __init__(self, data, estimate_niv_core=False):
        self.data = data

    def __add__(self, other):
        return self

    def symmetrize_orbitals(self, *args):
        return self


@pytest.fixture
def cfg(tmp_path):
    class Dummy:
        pass

    cfg = Dummy()

    cfg.sys = Dummy()
    cfg.sys.n = 1
    cfg.sys.nd_bands = 1
    cfg.sys.np_bands = 1
    cfg.sys.n_bands = 2
    cfg.sys.occ_dmft = None

    cfg.box = Dummy()
    cfg.box.niw_core = 2
    cfg.box.niv_core = 2
    cfg.box.niv_shell = 1
    cfg.box.niv_full = None

    cfg.dmft = Dummy()
    cfg.dmft.input_path = ""
    cfg.dmft.fname_1p = "f1"
    cfg.dmft.fname_2p = "f2"
    cfg.dmft.symmetrize_orbitals = None
    cfg.dmft.do_sym_v_vp = False

    cfg.lattice = Dummy()
    cfg.lattice.interaction = Dummy()
    cfg.lattice.type = "t_tp_tpp"
    cfg.lattice.er_input = [1, 0, 0]
    cfg.lattice.interaction_type = ""
    cfg.lattice.interaction_input = None
    cfg.lattice.orbital_basis = None

    cfg.lattice.k_grid = Dummy()
    cfg.lattice.k_grid.nk_tot = 1
    cfg.lattice.k_grid.specify_orbital_basis = lambda *a: None

    cfg.lattice.q_grid = Dummy()
    cfg.lattice.q_grid.nk_tot = 1
    cfg.lattice.q_grid.specify_orbital_basis = lambda *a: None

    cfg.output = Dummy()
    cfg.output.output_path = str(tmp_path)
    cfg.output.plotting_subfolder_name = "Plots"
    cfg.output.save_quantities = False
    cfg.output.do_plotting = False

    cfg.eliashberg = Dummy()
    cfg.eliashberg.subfolder_name = "Eliashberg"
    cfg.eliashberg.perform_eliashberg = False

    cfg.logger = Dummy()
    cfg.logger.info = lambda *a: None

    return cfg


def test_combine_two_atoms_branch(monkeypatch, cfg):
    monkeypatch.setattr(dga_io, "config", cfg)
    monkeypatch.setattr(dga_io.w2dyn_aux, "W2dynFile", DummyW2dynFile)
    monkeypatch.setattr(dga_io.w2dyn_aux, "W2dynG4iwFile", DummyW2dynG4iwFile)
    monkeypatch.setattr(dga_io, "LocalFourPoint", DummyLocalFourPoint)
    monkeypatch.setattr(dga_io, "GreensFunction", DummyGreensFunction)
    monkeypatch.setattr(dga_io, "SelfEnergy", DummySelfEnergy)
    monkeypatch.setattr(dga_io, "set_hamiltonian", lambda *a, **k: "ham")

    g, sigma, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config(combine_two_atoms_to_one_obj=True)

    assert cfg.sys.n_bands == 4

    mat = g2_dens.mat

    # diagonal blocks
    assert np.allclose(mat[0:2, 0:2, 0:2, 0:2], 15.0)
    assert np.allclose(mat[2:4, 2:4, 2:4, 2:4], 15.0)

    # off-diagonal blocks should be zero
    assert np.allclose(mat[0:2, 0:2, 2:4, 2:4], 0.0)

    g_data = g.data
    assert np.allclose(g_data[0, 0], g_data[2, 2])  # duplicated blocks

    sigma_data = sigma.data
    assert sigma_data.shape[3] == 4  # orbital doubled

    occ = cfg.sys.occ_dmft
    assert np.allclose(occ[0:2, 0:2], 4.0)
    assert np.allclose(occ[2:4, 2:4], 4.0)
    assert np.allclose(occ[0:2, 2:4], 0.0)
    assert np.allclose(occ[2:4, 0:2], 0.0)

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os

import numpy as np
import pytest

from dgamore import brillouin_zone
from dgamore.brillouin_zone import KGrid
from dgamore.hamiltonian import Hamiltonian, HoppingElement, InteractionElement


def test_hoppingelement_valid():
    """HoppingElement stores r_lat, orbitals and value for a valid input."""
    he = HoppingElement([1, 0, 0], [1, 2], 3.5)
    assert he.r_lat == (1, 0, 0)
    assert np.all(he.orbs == np.array([1, 2]))
    assert he.value == 3.5


def test_hoppingelement_invalid_inputs():
    """HoppingElement raises ValueError for a malformed r_lat, orbitals, or value."""
    with pytest.raises(ValueError):
        HoppingElement([1, 0], [1, 2], 1.0)
    with pytest.raises(ValueError):
        HoppingElement([1, 0, 0], [0, 2], 1.0)
    with pytest.raises(ValueError):
        HoppingElement([0, 1, 0], [1, 2], "abc")


def test_interactionelement_valid():
    """InteractionElement stores orbitals and value for a valid input."""
    ie = InteractionElement([0, 0, 0], [1, 1, 1, 1], 10)
    assert np.all(ie.orbs == np.array([1, 1, 1, 1]))
    assert ie.value == 10


def test_interactionelement_invalid_inputs():
    """InteractionElement raises ValueError for a malformed r_lat, orbitals, or value."""
    with pytest.raises(ValueError):
        InteractionElement([0, 0], [1, 1, 1, 1], 10)
    with pytest.raises(ValueError):
        InteractionElement([0, 0, 0], [1, 1, 1], 10)
    with pytest.raises(ValueError):
        InteractionElement([0, 0, 0], [1, 1, 1, 1], "bad")


def test_parse_elements_with_dicts():
    """_parse_elements converts dicts into the requested element dataclass."""
    h = Hamiltonian()
    dicts = [{"r_lat": [1, 0, 0], "orbs": [1, 1], "value": 1.0}]
    parsed = h._parse_elements(dicts, HoppingElement)
    assert isinstance(parsed[0], HoppingElement)


def test_prepare_lattice_indices_and_orbs():
    """_prepare_lattice_indices_and_orbs returns the R-vector mapping, count and orbital count."""
    h = Hamiltonian()
    elems = [HoppingElement([1, 0, 0], [1, 2], 1.0), HoppingElement([2, 0, 0], [2, 1], 1.5)]
    mapping, n_rp, n_orbs = h._prepare_lattice_indices_and_orbs(elems)
    assert isinstance(mapping, dict)
    assert n_rp == 2
    assert n_orbs == 2


def test_add_kinetic_term():
    """_add_kinetic_term records a non-local hopping into the real-space e(R)."""
    h = Hamiltonian()
    hops = [HoppingElement([1, 0, 0], [1, 1], 1.0)]
    h._add_kinetic_term(hops)
    assert np.allclose(h._er, [[[1.0]]])


def test_add_kinetic_term_rejects_local():
    """_add_kinetic_term rejects a purely local (R=0) hopping."""
    h = Hamiltonian()
    hops = [HoppingElement([0, 0, 0], [1, 1], 1.0)]
    with pytest.raises(ValueError):
        h._add_kinetic_term(hops)


def test_kinetic_one_band_2d_t_tp_tpp():
    """kinetic_one_band_2d_t_tp_tpp sets the t/t'/t'' hoppings with the correct signs."""
    h = Hamiltonian()
    h.kinetic_one_band_2d_t_tp_tpp(t=1.0, tp=0.5, tpp=0.25)
    # Check that nearest, next-nearest, and next-next-nearest hoppings are set
    values = h._er.flatten()
    assert np.isclose(values[0], -0.5)
    assert np.isclose(values[1], -1)
    assert np.isclose(values[2], -0.25)


def test_add_interaction_term_local_and_nonlocal():
    """_add_interaction_term splits local and non-local interaction elements."""
    h = Hamiltonian()
    inter = [
        InteractionElement([0, 0, 0], [1, 1, 1, 1], 5.0),
        InteractionElement([1, 0, 0], [1, 1, 1, 1], 2.0),
    ]
    h._add_interaction_term(inter)
    assert h._ur_local[0, 0, 0, 0] == 5.0
    assert np.any(h._ur_nonlocal != 0)


def test_single_band_interaction_sets_correct_u():
    """single_band_interaction sets the single local Hubbard U."""
    h = Hamiltonian().single_band_interaction(4.0)
    assert np.isclose(h._ur_local[0, 0, 0, 0], 4.0)


def test_kanamori_interaction_defaults_1_band():
    """kanamori_interaction_d for a single band sets the local U."""
    h = Hamiltonian().kanamori_interaction_d(n_bands=1, udd=5.0, jdd=1.0)
    assert np.isclose(h._ur_local[0, 0, 0, 0], 5.0)


def test_kanamori_interaction_with_vdd_1_band():
    """kanamori_interaction_d with vdd for a single band sets the local U."""
    h = Hamiltonian().kanamori_interaction_d(n_bands=1, udd=5.0, jdd=1.0, vdd=2.0)
    assert np.isclose(h._ur_local[0, 0, 0, 0], 5.0)


def test_kanamori_interaction_with_vdd_2_band():
    """kanamori_interaction_d for two bands sets the U/V/J Kanamori entries."""
    params = {
        "udd": np.random.rand(),
        "jdd": np.random.rand(),
        "vdd": np.random.rand(),
    }

    h = Hamiltonian().kanamori_interaction_d(n_bands=2, **params)

    assert np.isclose(h._ur_local[0, 0, 0, 0], params["udd"])
    assert np.isclose(h._ur_local[1, 1, 1, 1], params["udd"])

    for i, j in [(0, 1), (1, 0)]:
        assert np.isclose(h._ur_local[i, j, i, j], params["vdd"])
        assert np.isclose(h._ur_local[i, j, j, i], params["jdd"])

    assert np.isclose(h._ur_local[0, 0, 1, 1], params["jdd"])
    assert np.isclose(h._ur_local[1, 1, 0, 0], params["jdd"])


def test_kanamori_d_basic():
    """kanamori_interaction_d fills the full U-tensor with the expected U/J/V structure."""
    ham = Hamiltonian()
    n = 3
    u_val = 4.0
    j = 1.0

    ham.kanamori_interaction_d(n_bands=n, udd=u_val, jdd=j)
    u = ham.get_local_u()

    v = u_val - 2 * j

    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    if a == b == c == d:
                        assert np.isclose(u[a, b, c, d], u_val)
                    elif (a == d and b == c) or (a == b and c == d):
                        assert np.isclose(u[a, b, c, d], j)
                    elif a == c and b == d:
                        assert np.isclose(u[a, b, c, d], v)
                    else:
                        assert np.isclose(u[a, b, c, d], 0.0)


def test_kanamori_p_basic():
    """kanamori_interaction_p fills the full U-tensor with the expected U/J/V structure."""
    ham = Hamiltonian()
    n = 2
    u_val = 3.0
    j = 0.5

    ham.kanamori_interaction_p(n_bands=n, upp=u_val, jpp=j)
    u = ham.get_local_u()

    v = u_val - 2 * j

    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    if a == b == c == d:
                        assert np.isclose(u[a, b, c, d], u_val)
                    elif (a == d and b == c) or (a == b and c == d):
                        assert np.isclose(u[a, b, c, d], j)
                    elif a == c and b == d:
                        assert np.isclose(u[a, b, c, d], v)
                    else:
                        assert np.isclose(u[a, b, c, d], 0.0)


def test_kanamori_dp_block_structure():
    """kanamori_interaction_dp produces the expected d/p block U/J/V structure."""
    ham = Hamiltonian()

    nd, npb = 2, 2

    udd, jdd = 8.0, 1.0
    upp, jpp = 4.0, 0.5
    udp, jdp = 2.0, 0.2

    ham.kanamori_interaction_dp(nd_bands=nd, np_bands=npb, udd=udd, upp=upp, udp=udp, jdd=jdd, jpp=jpp, jdp=jdp)

    u = ham.get_local_u().mat
    vdd = udd - 2 * jdd
    vpp = upp - 2 * jpp

    def is_d(i):
        return i < nd

    for a in range(nd + npb):
        for b in range(nd + npb):
            for c in range(nd + npb):
                for d in range(nd + npb):

                    if is_d(a) and is_d(b):
                        uu, jj, vv = udd, jdd, vdd
                    elif (not is_d(a)) and (not is_d(b)):
                        uu, jj, vv = upp, jpp, vpp
                    else:
                        uu, jj, vv = 0, jdp, udp

                    if a == b == c == d:
                        assert np.isclose(u[a, b, c, d], uu)
                    elif (a == d and b == c) or (a == b and c == d):
                        assert np.isclose(u[a, b, c, d], jj)
                    elif a == c and b == d:
                        assert np.isclose(u[a, b, c, d], vv)
                    else:
                        assert np.isclose(u[a, b, c, d], 0.0)


def test_kanamori_dp_index_split():
    """kanamori_interaction_dp splits d and p orbital indices with the right U/V/J values."""
    ham = Hamiltonian()

    nd, npb = 1, 1

    ham.kanamori_interaction_dp(nd_bands=nd, np_bands=npb, udd=10.0, upp=5.0, udp=2.0, jdd=1.0, jpp=0.5, jdp=0.1)

    u = ham.get_local_u()

    assert np.isclose(u[0, 0, 0, 0], 10.0)
    assert np.isclose(u[1, 1, 1, 1], 5.0)

    assert np.isclose(u[0, 1, 0, 1], 2.0)
    assert np.isclose(u[0, 1, 1, 0], 0.1)


def test_kanamori_dp_no_unexpected_terms():
    """kanamori_interaction_dp leaves cross-block off-diagonal terms at zero."""
    ham = Hamiltonian()

    ham.kanamori_interaction_dp(nd_bands=2, np_bands=1, udd=6.0, upp=3.0, udp=1.0, jdd=1.0, jpp=0.5, jdp=0.2)

    u = ham.get_local_u()

    assert np.isclose(u[0, 1, 2, 0], 0.0)
    assert np.isclose(u[2, 0, 1, 2], 0.0)


def test_convham_2_orbs():
    """_convham_2_orbs Fourier-transforms a 2-orbital real-space hopping to k-space."""
    h = Hamiltonian()
    h._er_r_grid = np.zeros((1, 1, 1, 3))
    h._er_r_weights = np.ones((1, 1))
    h._er = np.ones((1, 1, 1))
    kmesh = np.zeros((3, 1))
    out = h._convham_2_orbs(kmesh)
    assert np.allclose(out, 1.0)


def test_convham_4_orbs():
    """_convham_4_orbs Fourier-transforms a 4-orbital real-space interaction to k-space."""
    h = Hamiltonian()
    h._ur_r_grid = np.zeros((1, 1, 1, 1, 1, 3))
    h._ur_r_weights = np.ones((1, 1))
    h._ur_nonlocal = np.ones((1, 1, 1, 1, 1))
    kmesh = np.zeros((3, 1))
    out = h._convham_4_orbs(kmesh)
    assert np.allclose(out, 1.0)


def test_set_and_get_ek():
    """set_ek / get_ek round-trip the dispersion array."""
    h = Hamiltonian()
    test_ek = np.array([[[[1.0]]]])
    h.set_ek(test_ek)
    assert np.allclose(h.get_ek(), test_ek)


def test_get_local_u_returns_localinteraction():
    """get_local_u returns a LocalInteraction wrapping the local U-tensor."""
    h = Hamiltonian()
    h._ur_local = np.ones((1, 1, 1, 1))
    local_u = h.get_local_u()
    assert hasattr(local_u, "mat")
    assert local_u.mat.shape == (1, 1, 1, 1)


def test_get_vq_returns_interaction():
    """get_vq returns a momentum-dependent Interaction."""
    h = Hamiltonian()
    h._ur_r_grid = np.zeros((1, 1, 1, 1, 1, 3))
    h._ur_r_weights = np.ones((1, 1))
    h._ur_nonlocal = np.ones((1, 1, 1, 1, 1))

    nk = (1, 1, 1)
    kg = brillouin_zone.KGrid(nk=nk, symmetries=[])
    vq = h.get_vq(kg)
    assert hasattr(vq, "mat")
    assert vq.mat.shape[-4:] == (1, 1, 1, 1)


def test_read_write_hr_hk_files():
    """read_hr_w2k and read_hk_w2k yield matching e(k) for one- and two-band Wannier inputs."""
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/hamiltonian"
    k_grid = KGrid(nk=(24, 24, 1), symmetries=brillouin_zone.two_dimensional_square_symmetries())

    wannier_hr_oneband = Hamiltonian().read_hr_w2k(f"{folder}/wannier_hr_oneband.dat")
    ek = wannier_hr_oneband.get_ek(k_grid)

    assert wannier_hr_oneband._er.shape[-1] == 1
    assert wannier_hr_oneband._er.shape[-2] == 1

    assert ek.shape == (24, 24, 1, 1, 1)

    wannier_hk_oneband, _ = Hamiltonian().read_hk_w2k(f"{folder}/wannier_oneband_24x24.hk")
    ek_ref = wannier_hk_oneband.get_ek(k_grid).reshape(ek.shape)
    assert np.allclose(ek, ek_ref)

    wannier_hr_twoband = Hamiltonian().read_hr_w2k(f"{folder}/wannier_hr_twoband.dat")
    ek = wannier_hr_twoband.get_ek(k_grid)

    assert wannier_hr_twoband._er.shape[-1] == 2
    assert wannier_hr_twoband._er.shape[-2] == 2

    assert ek.shape == (24, 24, 1, 2, 2)

    wannier_hk_twoband, _ = Hamiltonian().read_hk_w2k(f"{folder}/wannier_twoband_24x24.hk")
    ek_ref = wannier_hk_twoband.get_ek(k_grid).reshape(ek.shape)
    assert np.allclose(ek, ek_ref)

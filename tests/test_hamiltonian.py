# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
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
        InteractionElement([-1, 0, 0], [1, 1, 1, 1], 2.0),
    ]
    h._add_interaction_term(inter)
    assert h._ur_local[0, 0, 0, 0] == 5.0
    assert np.any(h._ur_nonlocal != 0)


def _two_band_local(extra):
    """Builds a symmetric two-band local tensor (U on the diagonal) plus the given extra (orbs, value) elements."""
    elements = [InteractionElement([0, 0, 0], [a, a, a, a], 3.0) for a in (1, 2)]
    return elements + [InteractionElement([0, 0, 0], list(orbs), value) for orbs, value in extra]


def test_add_interaction_term_raises_for_a_local_tensor_without_pair_exchange_symmetry():
    """A local element U_{1211} without its pair-exchange partner U_{2111} is rejected."""
    with pytest.raises(ValueError, match="pair-exchange"):
        Hamiltonian()._add_interaction_term(_two_band_local([((1, 2, 1, 1), 0.3)]))


def test_add_interaction_term_raises_for_a_local_tensor_without_reality_symmetry():
    """A pair-exchange symmetric pair U_{1112} = U_{1121} without its reality partners U_{1211} and U_{2111} raises."""
    with pytest.raises(ValueError, match="reality"):
        Hamiltonian()._add_interaction_term(_two_band_local([((1, 1, 1, 2), 0.3), ((1, 1, 2, 1), 0.3)]))


def test_add_interaction_term_raises_when_the_mirrored_lattice_vector_is_missing():
    """A non-local vector listed without its mirror image -R is rejected."""
    inter = [InteractionElement([0, 0, 0], [1, 1, 1, 1], 5.0), InteractionElement([1, 0, 0], [1, 1, 1, 1], 2.0)]
    with pytest.raises(ValueError, match=r"\(-1, 0, 0\)"):
        Hamiltonian()._add_interaction_term(inter)


def test_add_interaction_term_raises_for_a_nonlocal_tensor_without_pair_exchange_symmetry():
    """V(R) and V(-R) that are not pair-exchange images of each other are rejected."""
    inter = [
        InteractionElement([0, 0, 0], [1, 1, 1, 1], 5.0),
        InteractionElement([1, 0, 0], [1, 1, 1, 1], 2.0),
        InteractionElement([-1, 0, 0], [1, 1, 1, 1], 1.0),
    ]
    with pytest.raises(ValueError, match="pair-exchange"):
        Hamiltonian()._add_interaction_term(inter)


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


@pytest.mark.parametrize(
    "build",
    [
        lambda h: h.single_band_interaction(4.0),
        lambda h: h.interaction_orbital_diagonal(4.0, 3),
        lambda h: h.kanamori_interaction_d(3, udd=4.0, jdd=0.6),
        lambda h: h.kanamori_interaction_p(2, upp=3.0, jpp=0.5, vpp=1.5),
        lambda h: h.kanamori_interaction_dp(
            nd_bands=2, np_bands=1, udd=4.0, upp=3.0, udp=2.0, jdd=0.6, jpp=0.4, jdp=0.3
        ),
        lambda h: h.read_umatrix(f"{os.path.dirname(os.path.abspath(__file__))}/test_data/local_sde/u_matrix.dat"),
        lambda h: h.read_umatrix(f"{os.path.dirname(os.path.abspath(__file__))}/../docs/u_matrix.dat"),
    ],
    ids=["single_band", "orbital_diagonal", "kanamori_d", "kanamori_p", "kanamori_dp", "umatrix_file", "docs_example"],
)
def test_common_interaction_types_pass_the_symmetry_check(build):
    """Every builder and shipped interaction file satisfies the pair-exchange and reality symmetries without raising."""
    h = build(Hamiltonian())
    assert h.get_local_u().mat.shape[0] >= 1


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

                    # the element couples orbitals a and b, or a and c for the pair hopping U_{aabb}
                    other = b if a != b else c
                    if is_d(a) and is_d(other):
                        uu, jj, vv = udd, jdd, vdd
                    elif (not is_d(a)) and (not is_d(other)):
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


def test_read_umatrix_example_and_vq_match_the_documented_lattice_sum():
    """The documented u_matrix.dat parses into the stated local U, and get_vq equals sum_{R!=0} e^{iqR} V(R)."""
    path = f"{os.path.dirname(os.path.abspath(__file__))}/../docs/u_matrix.dat"
    h = Hamiltonian().read_umatrix(path)
    u = h.get_local_u().mat
    assert np.allclose([u[0, 0, 0, 0], u[0, 1, 0, 1], u[0, 0, 1, 1], u[0, 1, 1, 0]], [3.2, 2.4, 0.4, 0.4])

    rows = np.loadtxt(path, skiprows=3)
    rows = rows[np.any(rows[:, :3] != 0, axis=1)]
    kg = KGrid(nk=(4, 4, 1), symmetries=[])
    q = kg.kmesh.reshape(3, -1)
    ref = np.zeros((q.shape[1], 2, 2, 2, 2), dtype=complex)
    for rx, ry, rz, o1, o2, o3, o4, re, _ in rows:
        ref[:, int(o1) - 1, int(o2) - 1, int(o3) - 1, int(o4) - 1] += re * np.exp(
            1j * (rx * q[0] + ry * q[1] + rz * q[2])
        )
    vq = h.get_vq(kg).mat.reshape(-1, 2, 2, 2, 2)
    assert np.allclose(vq, ref, atol=1e-5)
    assert np.allclose(vq[0, 0, 1, 0, 1], 4 * 0.4 + 4 * 0.283, atol=1e-5)  # q = 0 is the plain lattice sum


def test_read_umatrix_applies_the_lattice_vector_weights_in_the_fourier_transform(tmp_path):
    """Each lattice vector's contribution to V^q is divided by its weight, matched to the vectors in file order."""
    path = tmp_path / "u_matrix.dat"
    path.write_text("1\n3\n1.0 2.0 1.0\n0 0 0 1 1 1 1 2.0 0.0\n1 0 0 1 1 1 1 1.0 0.0\n-1 0 0 1 1 1 1 1.0 0.0\n")
    h = Hamiltonian().read_umatrix(str(path))
    kg = KGrid(nk=(4, 1, 1), symmetries=[])
    q = kg.kmesh.reshape(3, -1)[0]
    ref = np.exp(1j * q) / 2.0 + np.exp(-1j * q)
    assert np.allclose(h.get_local_u().mat, 2.0)
    assert np.allclose(h.get_vq(kg).mat.reshape(-1), ref, atol=1e-6)


def test_read_umatrix_raises_when_the_header_count_disagrees_with_the_listed_vectors(tmp_path):
    """A header announcing more lattice vectors than the rows contain raises instead of misassigning the weights."""
    path = tmp_path / "u_matrix.dat"
    path.write_text("1\n3\n1.0 1.0 1.0\n0 0 0 1 1 1 1 2.0 0.0\n1 0 0 1 1 1 1 1.0 0.0\n")
    with pytest.raises(ValueError, match="lattice vectors"):
        Hamiltonian().read_umatrix(str(path))


def test_read_umatrix_raises_on_a_duplicate_tensor_element(tmp_path):
    """A repeated (lattice vector, orbital quadruple) row raises instead of silently overwriting the earlier value."""
    path = tmp_path / "u_matrix.dat"
    path.write_text("1\n2\n1.0 1.0\n0 0 0 1 1 1 1 2.0 0.0\n1 0 0 1 1 1 1 1.0 0.0\n1 0 0 1 1 1 1 0.5 0.0\n")
    with pytest.raises(ValueError, match="duplicate"):
        Hamiltonian().read_umatrix(str(path))


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

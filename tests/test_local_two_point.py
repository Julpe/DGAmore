# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Tests for the shared two-point base classes :class:`LocalTwoPoint` (momentum-independent) and :class:`TwoPoint`
(momentum-dependent) and for the orbital helpers that the Green's function, self-energy and gap function inherit
from them, plus the Dyson-inversion / Fermi-Dirac density helpers factored out of the Green's function.
"""

import numpy as np
import pytest

from dgamore.brillouin_zone import KGrid, two_dimensional_square_symmetries
from dgamore.greens_function import GreensFunction, _fermi_dirac_density
from dgamore.gap_function import GapFunction
from dgamore.local_n_point import LocalNPoint
from dgamore.local_two_point import LocalTwoPoint
from dgamore.n_point_base import IAmNonLocal
from dgamore.self_energy import SelfEnergy
from dgamore.two_point import TwoPoint


def test_local_two_point_is_not_nonlocal():
    """LocalTwoPoint subclasses LocalNPoint but not IAmNonLocal (momentum-independent)."""
    assert issubclass(LocalTwoPoint, LocalNPoint)
    assert not issubclass(LocalTwoPoint, IAmNonLocal)


def test_two_point_is_nonlocal_local_two_point():
    """TwoPoint subclasses both IAmNonLocal and LocalTwoPoint."""
    assert issubclass(TwoPoint, IAmNonLocal)
    assert issubclass(TwoPoint, LocalTwoPoint)


@pytest.mark.parametrize("cls", [GreensFunction, SelfEnergy, GapFunction])
def test_concrete_two_point_classes_inherit_two_point(cls):
    """GreensFunction, SelfEnergy and GapFunction inherit TwoPoint and IAmNonLocal."""
    assert issubclass(cls, TwoPoint)
    assert issubclass(cls, IAmNonLocal)


def test_local_two_point_has_no_bz_unfold():
    """LocalTwoPoint does not carry the IAmNonLocal momentum machinery."""
    obj = LocalTwoPoint(np.zeros((2, 2, 4)))
    assert not hasattr(obj, "map_to_full_bz")
    assert not hasattr(obj, "_map_to_full_bz")
    assert not hasattr(obj, "compress_q_dimension")


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((2, 2, 6), (0, 1)),  # [o1,o2,v]
        ((3, 2, 2, 6), (1, 2)),  # [k,o1,o2,v]
        ((2, 2, 1, 2, 2, 6), (3, 4)),  # [kx,ky,kz,o1,o2,v]
    ],
)
def test_get_orbital_axes(shape, expected):
    """_get_orbital_axes returns the orbital axes for each of the three layouts."""
    obj = LocalTwoPoint(np.zeros(shape))
    assert obj._get_orbital_axes() == expected


def test_get_orbital_axes_invalid_dim():
    """_get_orbital_axes raises ValueError for an unsupported number of dimensions."""
    obj = LocalTwoPoint(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        obj._get_orbital_axes()


def test_local_permute_and_transpose():
    """transpose_orbitals swaps orbitals on a copy and the identity permutation returns self."""
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((2, 2, 4)) + 1j * rng.standard_normal((2, 2, 4))
    obj = LocalTwoPoint(mat.copy())

    transposed = obj.transpose_orbitals()
    assert np.allclose(transposed.mat, np.swapaxes(mat, 0, 1))
    # copy semantics: source untouched
    assert np.allclose(obj.mat, mat)
    # identity permutation returns self
    assert obj.permute_orbitals("ab->ab") is obj


def test_local_permute_invalid_string():
    """permute_orbitals rejects a malformed permutation string."""
    obj = LocalTwoPoint(np.zeros((2, 2, 4)))
    with pytest.raises(ValueError):
        obj.permute_orbitals("abc->cba")


@pytest.mark.parametrize(
    "shape, nk, compressed",
    [
        ((2, 2, 4), (1, 1, 1), False),  # local layout [o1,o2,v]
        ((6, 2, 2, 4), (6, 1, 1), True),  # compressed momentum [q,o1,o2,v]
        ((2, 3, 1, 2, 2, 4), (2, 3, 1), False),  # decompressed momentum [kx,ky,kz,o1,o2,v]
    ],
)
def test_two_point_transpose_orbitals_all_layouts(shape, nk, compressed):
    """transpose_orbitals swaps the orbital axes on a copy across all momentum layouts."""
    rng = np.random.default_rng(1)
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    g = GreensFunction(mat.copy(), nk=nk, has_compressed_q_dimension=compressed)
    a1, a2 = g._get_orbital_axes()

    transposed = g.transpose_orbitals()

    assert np.allclose(transposed.mat, np.swapaxes(mat, a1, a2))
    assert np.allclose(g.mat, mat)  # copy semantics preserved


def test_two_point_permute_identity_and_invalid():
    """The q-aware permute_orbitals short-circuits the identity and rejects a malformed string."""
    g = GreensFunction(np.zeros((2, 2, 4)))
    assert g.permute_orbitals("ab->ab") is g
    with pytest.raises(ValueError):
        g.permute_orbitals("abc->cba")


def test_two_point_map_to_full_bz_expands_momenta():
    """map_to_full_bz expands the irreducible BZ to the full BZ, leaving orbitals untouched."""
    nk = (2, 2, 1)
    kg = KGrid(nk, two_dimensional_square_symmetries())
    nb, niv = 1, 3
    irr_vals = np.arange(kg.nk_irr, dtype=float)
    mat = irr_vals[:, None, None, None] * np.ones((kg.nk_irr, nb, nb, 2 * niv))
    g = GreensFunction(mat.astype(complex), nk=nk, has_compressed_q_dimension=True)

    g.map_to_full_bz(kg)  # no auto symmetries -> pure momentum expansion, orbitals untouched

    assert g.current_shape[0] == kg.nk_tot
    assert np.allclose(g.mat, mat.astype(complex)[kg.irrk_inv.ravel()])


def test_symmetrize_orbitals_averages_equivalent_orbitals():
    """symmetrize_orbitals averages equivalent orbital values."""
    mat = np.zeros((2, 2, 4), dtype=complex)
    mat[0, 0] = 1.0
    mat[1, 1] = 3.0
    obj = LocalTwoPoint(mat.copy())

    assert not obj.is_orbitally_symmetrized([1, 2])
    obj.symmetrize_orbitals([1, 2])
    assert obj.is_orbitally_symmetrized([1, 2])
    assert np.allclose(obj.mat[0, 0], 2.0)
    assert np.allclose(obj.mat[1, 1], 2.0)


def test_symmetrize_orbitals_noop_when_already_symmetric():
    """symmetrize_orbitals returns self when the object is already symmetric."""
    mat = np.full((2, 2, 4), 2.0, dtype=complex)
    obj = LocalTwoPoint(mat.copy())
    assert obj.symmetrize_orbitals([1, 2]) is obj


@pytest.mark.parametrize("shape", [(2, 2, 5), (4, 2, 2, 5), (2, 3, 1, 2, 2, 5)])
def test_invert_last_orbital_block_matches_manual_inverse(shape):
    """_invert_last_orbital_block matches a manual per-frequency matrix inverse across layouts."""
    rng = np.random.default_rng(2)
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    # make the orbital blocks well-conditioned
    nb = shape[-2]
    mat = mat + (np.eye(nb) * 5.0)[..., None]

    out = GreensFunction._invert_last_orbital_block(mat)

    # reference: move v in front, invert per-batch, move back
    ref = np.moveaxis(np.linalg.inv(np.moveaxis(mat, -1, -3)), -3, -1)
    assert np.allclose(out, ref)
    assert out.shape == mat.shape


def test_fermi_dirac_density_diagonal():
    """_fermi_dirac_density returns the Fermi function on the diagonal of a diagonal Hamiltonian."""
    beta = 2.0
    h = np.diag([0.5, -0.7]).astype(complex)
    rho = _fermi_dirac_density(h, beta)
    expected = np.diag(1.0 / (1.0 + np.exp(beta * np.array([0.5, -0.7]))))
    assert np.allclose(rho, expected)


def test_fermi_dirac_density_numerically_stable():
    """_fermi_dirac_density stays finite and saturates to 0/1 for large eigenvalues."""
    beta = 1.0
    h = np.diag([1e3, -1e3]).astype(complex)
    rho = _fermi_dirac_density(h, beta)
    assert np.all(np.isfinite(rho))
    assert np.isclose(rho[0, 0], 0.0, atol=1e-12)
    assert np.isclose(rho[1, 1], 1.0, atol=1e-12)


def test_fermi_dirac_density_batched_matches_per_element():
    """_fermi_dirac_density on a k-batch equals applying it to each k-block independently."""
    beta = 1.5
    rng = np.random.default_rng(3)
    h = rng.standard_normal((4, 2, 2))
    h = h + np.swapaxes(h, -1, -2)  # Hermitian (real symmetric) per k
    h = h.astype(complex)

    rho_batched = _fermi_dirac_density(h, beta)
    for k in range(h.shape[0]):
        assert np.allclose(rho_batched[k], _fermi_dirac_density(h[k], beta))

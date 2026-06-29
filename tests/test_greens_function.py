# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

from unittest.mock import MagicMock

import numpy as np
import pytest

import dgamore.config as config
from dgamore.greens_function import GreensFunction, update_mu
from dgamore.self_energy import SelfEnergy


def test_symmetrize_orbitals_already_symmetrized():
    """symmetrize_orbitals returns self and skips the private call when already symmetrized."""
    obj = GreensFunction(np.zeros((2, 2, 10)))

    obj._get_orbital_axes = MagicMock(return_value=(0, 1))
    obj.is_orbitally_symmetrized = MagicMock(return_value=True)
    obj._symmetrize_orbitals = MagicMock()

    orbitals = [1, 2]

    result = obj.symmetrize_orbitals(orbitals)

    assert result is obj
    obj.is_orbitally_symmetrized.assert_called_once_with(orbitals)
    obj._symmetrize_orbitals.assert_not_called()


def test_symmetrize_orbitals_calls_private():
    """symmetrize_orbitals delegates to _symmetrize_orbitals when not yet symmetrized."""
    obj = GreensFunction(np.zeros((2, 2, 10)))

    obj._get_orbital_axes = MagicMock(return_value=(1, 2))
    obj.is_orbitally_symmetrized = MagicMock(return_value=False)
    obj._symmetrize_orbitals = MagicMock(return_value="symmetrized_obj")

    orbitals = [1, 3]

    result = obj.symmetrize_orbitals(orbitals)

    obj.is_orbitally_symmetrized.assert_called_once_with(orbitals)
    obj._symmetrize_orbitals.assert_called_once_with(orbitals, (1, 2))
    assert result == "symmetrized_obj"


def test_is_orbitally_symmetrized_delegates():
    """is_orbitally_symmetrized delegates to _is_orbitally_symmetrized with the resolved orbital axes."""
    obj = GreensFunction(np.zeros((2, 2, 10)))

    obj._get_orbital_axes = MagicMock(return_value=(3, 4))
    obj._is_orbitally_symmetrized = MagicMock(return_value=True)

    orbitals = np.array([1, 2, 3])

    result = obj.is_orbitally_symmetrized(orbitals)

    obj._is_orbitally_symmetrized.assert_called_once_with(orbitals, (3, 4))
    assert result is True


def test_symmetrize_orbitals_empty_list():
    """symmetrize_orbitals on an empty orbital list returns self without symmetrizing."""
    obj = GreensFunction(np.zeros((2, 2, 10)))

    obj._get_orbital_axes = MagicMock(return_value=(0, 1))
    obj.is_orbitally_symmetrized = MagicMock(return_value=True)
    obj._symmetrize_orbitals = MagicMock()

    orbitals = []

    result = obj.symmetrize_orbitals(orbitals)

    assert result is obj
    obj._symmetrize_orbitals.assert_not_called()


@pytest.fixture
def greens_function():
    mat = np.zeros((1, 1, 1, 2, 2, 20))
    greens_function = GreensFunction(mat)

    greens_function._symmetrize_orbitals = MagicMock()
    greens_function._is_orbitally_symmetrized = MagicMock()
    greens_function.fit_smom = MagicMock()

    return greens_function


@pytest.mark.parametrize(
    "shape, expected_axes, compressed",
    [
        ((2, 2, 10), (0, 1), False),  # [o1,o2,v]
        ((3, 2, 2, 10), (1, 2), True),  # [k,o1,o2,v]
        ((2, 2, 2, 2, 2, 10), (3, 4), False),  # [kx,ky,kz,o1,o2,v]
    ],
)
def test_executes_symmetrization_if_not_already_symmetrized(shape, expected_axes, compressed, greens_function):
    """symmetrize_orbitals runs symmetrization with the layout-correct orbital axes when not yet symmetrized."""
    gf = greens_function
    gf.mat = np.zeros(shape)
    gf._has_compressed_q_dimension = compressed

    orbitals = [1, 2]
    gf._is_orbitally_symmetrized.return_value = False

    assert gf._get_orbital_axes() == expected_axes
    _ = greens_function.symmetrize_orbitals(orbitals)

    gf._is_orbitally_symmetrized.assert_called_once_with(orbitals, expected_axes)
    gf._symmetrize_orbitals.assert_called_once_with(orbitals, expected_axes)


@pytest.mark.parametrize(
    "shape, expected_axes, compressed",
    [
        ((2, 2, 10), (0, 1), False),  # [o1,o2,v]
        ((3, 2, 2, 10), (1, 2), True),  # [k,o1,o2,v]
        ((2, 2, 2, 2, 2, 10), (3, 4), False),  # [kx,ky,kz,o1,o2,v]
    ],
)
def test_does_not_executes_symmetrization_if_already_symmetrized(shape, expected_axes, compressed, greens_function):
    """symmetrize_orbitals skips symmetrization when the object is already symmetrized."""
    gf = greens_function
    gf.mat = np.zeros(shape)
    gf._has_compressed_q_dimension = compressed

    orbitals = [1, 2]
    gf._is_orbitally_symmetrized.return_value = True

    _ = greens_function.symmetrize_orbitals(orbitals)
    assert gf._get_orbital_axes() == expected_axes

    gf._is_orbitally_symmetrized.assert_called_once_with(orbitals, expected_axes)
    gf._symmetrize_orbitals.assert_not_called()


def _toy_inputs():
    nk = (2, 2, 1)
    nbands, niv = 1, 8
    beta, mu = 5.0, 0.3
    ek = np.zeros((*nk, nbands, nbands))
    # SelfEnergy stored with a single compressed q axis [k, o1, o2, v], as the pipeline produces it.
    sig = SelfEnergy(
        np.zeros((int(np.prod(nk)), nbands, nbands, 2 * niv)), nk=nk, has_compressed_q_dimension=True, beta=beta
    )
    return nk, ek, sig, beta, mu


def test_get_g_full_stores_beta_mu_and_does_not_read_config(monkeypatch):
    """get_g_full stores the injected beta/mu instead of reading config.sys."""
    nk, ek, sig, beta, mu = _toy_inputs()
    monkeypatch.setattr(config.sys, "beta", 999.0)
    monkeypatch.setattr(config.sys, "mu", 999.0)
    g = GreensFunction.get_g_full(sig, mu, ek, beta)
    assert g._beta == beta
    assert g._mu == mu


def test_get_fill_nonlocal_caches_and_does_not_write_config(monkeypatch):
    """get_fill_nonlocal caches filling on the object and leaves config.sys untouched."""
    nk, ek, sig, beta, mu = _toy_inputs()
    sentinel = object()
    monkeypatch.setattr(config.sys, "n", sentinel)
    monkeypatch.setattr(config.sys, "occ", sentinel)
    monkeypatch.setattr(config.sys, "occ_k", sentinel)

    g = GreensFunction.get_g_full(sig, mu, ek, beta)
    n, occ, occ_k = g.get_fill_nonlocal()

    # results are cached on and exposed by the object
    assert g.n is n
    assert g.occ is occ
    assert g.occ_k is occ_k

    # the global config must be untouched
    assert config.sys.n is sentinel
    assert config.sys.occ is sentinel
    assert config.sys.occ_k is sentinel


def test_energies_use_injected_state_not_config(monkeypatch):
    """Kinetic and potential energies are computed from injected state, not from config.sys."""
    nk, ek, sig, beta, mu = _toy_inputs()
    monkeypatch.setattr(config.sys, "beta", 999.0)
    monkeypatch.setattr(config.sys, "mu", 999.0)
    monkeypatch.setattr(config.sys, "occ_k", None)

    g = GreensFunction.get_g_full(sig, mu, ek, beta)
    g.get_fill_nonlocal()
    # zero dispersion -> zero kinetic energy; computed from self._occ_k, not config.sys.occ_k
    assert g.get_ekin() == 0.0
    # potential energy is finite and does not raise despite config being poisoned
    assert np.isfinite(g.get_epot())


def test_fermi_dirac_density_matches_diagonal_matmul_reference():
    """_fermi_dirac_density (column-scaling) equals the explicit V diag(f) V^-1 construction, bit-for-bit."""
    from dgamore.greens_function import _fermi_dirac_density

    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 3, 3))
    h = 0.5 * (a + a.swapaxes(-1, -2))
    beta = 7.0
    eigvals, eigvecs = np.linalg.eig(beta * h)
    d = np.empty_like(eigvals)
    m = eigvals > 0
    d[m] = np.exp(-eigvals[m]) / (1 + np.exp(-eigvals[m]))
    d[~m] = 1 / (1 + np.exp(eigvals[~m]))
    ref = eigvecs @ np.einsum("...i,ij->...ij", d, np.eye(3, dtype=d.dtype)) @ np.linalg.inv(eigvecs)
    assert np.array_equal(_fermi_dirac_density(h, beta), ref)


def test_get_epot_e_corr_einsum_matches_transposed_product_reference():
    """get_epot's correlation term (fused einsum) matches the explicit transposed-product form (multi-band)."""
    nk = (2, 2, 1)
    nb, niv = 2, 6
    beta, mu = 5.0, 0.3
    rng = np.random.default_rng(1)
    ek = rng.standard_normal((*nk, nb, nb))
    ek = 0.5 * (ek + ek.swapaxes(-1, -2))
    sig_mat = rng.standard_normal((int(np.prod(nk)), nb, nb, 2 * niv)) + 1j * rng.standard_normal(
        (int(np.prod(nk)), nb, nb, 2 * niv)
    )
    sig = SelfEnergy(sig_mat, nk=nk, has_compressed_q_dimension=True, beta=beta)
    g = GreensFunction.get_g_full(sig, mu, ek, beta)
    g.get_fill_nonlocal()

    smom0 = sig.smom[0]
    dsigma = g._sigma.decompress_q_dimension().mat - smom0[..., None]
    e_corr_ref = (dsigma * g.decompress_q_dimension().transpose_orbitals().mat).sum().real / beta
    e_corr_new = np.einsum("...abv,...bav->...", dsigma, g.decompress_q_dimension().mat).sum().real / beta
    assert np.allclose(e_corr_new, e_corr_ref, rtol=1e-5)
    assert np.isfinite(g.get_epot())


def test_update_mu_without_logger_is_silent_on_failure():
    """update_mu with logger=None returns the input mu silently when root-finding fails."""
    nk, ek, sig, beta, mu = _toy_inputs()
    # an impossible target filling forces the RuntimeError branch; logger=None must not raise
    out = update_mu(mu, 1e9, ek, sig.mat, beta, sig.smom[0], logger=None)
    assert out == mu


def test_update_mu_with_logger_logs_on_failure():
    """update_mu logs a debug message and returns the input mu when root-finding fails."""
    nk, ek, sig, beta, mu = _toy_inputs()
    logger = MagicMock()
    out = update_mu(mu, 1e9, ek, sig.mat, beta, sig.smom[0], logger=logger)
    assert out == mu
    logger.debug.assert_called_once()

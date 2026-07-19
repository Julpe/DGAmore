# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import dgamore.brillouin_zone as bz
import dgamore.config as config
import dgamore.max_ent as max_ent
import dgamore.mpi_utils as mpi_utils
from dgamore.greens_function import GreensFunction
from dgamore.max_ent import orbital_to_band_basis
from dgamore.self_energy import SelfEnergy
from tests.conftest import FAKE_MPI, run_parallel


def _hermitian_hk(eigvals: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Builds H = U diag(eigvals) U^dagger for a single k-point."""
    return u @ np.diag(eigvals).astype(complex) @ u.conj().T


def _random_unitary(n: int, seed: int) -> np.ndarray:
    """Returns a Haar-ish random unitary via QR of a complex Gaussian matrix."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(z)
    # Fix the phases so the decomposition is deterministic.
    return q @ np.diag(np.exp(-1j * np.angle(np.diag(r))))


def test_orbital_to_band_basis_no_frequency_recovers_band_diagonal():
    """orbital_to_band_basis with no frequency axis recovers the band-diagonal Green's function."""
    # H(k) with distinct, ascending eigenvalues so eigh ordering is unambiguous.
    eigvals = np.array([-1.0, 2.0])
    u = _random_unitary(2, seed=1)
    hk = _hermitian_hk(eigvals, u)[None, None, None]

    g_band = np.diag(np.array([0.3 - 0.7j, -0.4 + 0.2j]))
    g_orb = (u @ g_band @ u.conj().T)[None, None, None]

    result = orbital_to_band_basis(hk.copy(), g_orb.copy())

    assert np.allclose(result[0, 0, 0], g_band, atol=1e-10)


def test_orbital_to_band_basis_with_frequency_axis_rotates_every_frequency():
    """orbital_to_band_basis rotates every frequency slice into the band basis."""
    eigvals = np.array([-0.5, 1.5])
    u = _random_unitary(2, seed=2)
    hk = _hermitian_hk(eigvals, u)[None, None, None]

    niv = 4
    # A different band-diagonal Green's function for each frequency.
    g0 = np.linspace(0.1, 0.4, niv) - 1j * np.linspace(0.5, 0.8, niv)
    g1 = -np.linspace(0.2, 0.6, niv) + 1j * np.linspace(0.1, 0.3, niv)
    g_band = np.zeros((2, 2, niv), dtype=complex)
    g_band[0, 0] = g0
    g_band[1, 1] = g1

    g_orb = np.einsum("ai,ijv,bj->abv", u, g_band, u.conj())[None, None, None]

    result = orbital_to_band_basis(hk.copy(), g_orb.copy())

    assert result.shape == (1, 1, 1, 2, 2, niv)
    assert np.allclose(result[0, 0, 0], g_band, atol=1e-10)


def test_orbital_to_band_basis_single_band_is_identity():
    """orbital_to_band_basis is the identity for a single band."""
    hk = np.array([[1.7]], dtype=complex)[None, None, None]
    niv = 3
    g = (np.arange(niv) - 1j * np.arange(niv)).reshape(1, 1, niv)
    g_orb = g[None, None, None]

    result = orbital_to_band_basis(hk.copy(), g_orb.copy())

    assert np.allclose(result[0, 0, 0], g, atol=1e-12)


def test_orbital_to_band_basis_band_diagonal_is_symmetry_invariant():
    """Band-diagonal spectral content is invariant when a symmetry operation maps a k-point to an equivalent one."""
    eigvals = np.array([-0.8, 1.1])
    u_rep = _random_unitary(2, seed=3)
    u_g = _random_unitary(2, seed=4)

    hk_rep = _hermitian_hk(eigvals, u_rep)
    hk_img = u_g @ hk_rep @ u_g.conj().T

    niv = 5
    rng = np.random.default_rng(5)
    g_rep = rng.standard_normal((2, 2, niv)) + 1j * rng.standard_normal((2, 2, niv))
    g_img = np.einsum("ai,ijv,jb->abv", u_g, g_rep, u_g.conj().T)

    hk = np.stack([hk_rep, hk_img])[:, None, None]
    data = np.stack([g_rep, g_img])[:, None, None]

    result = orbital_to_band_basis(hk.copy(), data.copy())

    diag_rep = np.diagonal(result[0, 0, 0], axis1=0, axis2=1)  # [niv, band]
    diag_img = np.diagonal(result[1, 0, 0], axis1=0, axis2=1)
    assert np.allclose(diag_rep, diag_img, atol=1e-10)


def _fake_problem(im_axis, re_axis, im_data, beta):
    """AnalyticContinuationProblem stand-in: encodes the first im-axis data value into a constant real-freq spectrum."""
    a_opt = np.full(len(re_axis), float(np.imag(im_data[0])))
    return MagicMock(solve=MagicMock(side_effect=lambda *args, **kwargs: [SimpleNamespace(A_opt=a_opt.copy())]))


def _problem_from_solve(solve):
    """Builds an AnalyticContinuationProblem stand-in whose instances delegate solve() to the given callable."""
    return lambda *args, **kwargs: MagicMock(solve=MagicMock(side_effect=solve))


def _fake_real_freq_two_point(spectrum=None, wgrid=None, kind=""):
    """RealFreqTwoPoint stand-in whose kkt() returns zero real part, so the self-energy reduces to the Hartree shift."""
    return MagicMock(kkt=MagicMock(return_value=np.zeros(len(wgrid))))


def _build_hk(nk, n_bands, seed=0):
    """Random Hermitian H(k) per k-point with off-diagonal coupling, so the band basis differs from the orbital"""
    rng = np.random.default_rng(seed)
    hk = np.zeros((*nk, n_bands, n_bands), dtype=complex)
    for idx in np.ndindex(*nk):
        a = rng.standard_normal((n_bands, n_bands)) + 1j * rng.standard_normal((n_bands, n_bands))
        hk[idx] = a + a.conj().T
    return hk


def _build_giwk_mat(nk, n_bands, niv, seed=1):
    rng = np.random.default_rng(seed)
    shape = (*nk, n_bands, n_bands, 2 * niv)
    mat = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return mat.astype(np.complex64)


def _setup_maxent_config(tmp_path, nk, n_bands, niv_core=3, w_count=7, seed=0):
    config.lattice.nk = nk
    config.lattice.k_grid = bz.KGrid(nk, symmetries=bz.two_dimensional_square_symmetries())
    config.box.niv_core = niv_core
    config.sys.beta = 10.0
    config.sys.n_bands = n_bands
    config.ana_cont.w_count = w_count
    config.output.output_path = str(tmp_path)
    config.logger = MagicMock()
    hk = _build_hk(nk, n_bands, seed=seed)
    config.lattice.hamiltonian = SimpleNamespace(get_ek=lambda k_grid=None: hk)
    return hk


def _expected_band_spectrum(mat, hk, nk, n_bands, niv_core, k_grid):
    """Reproduces the full-BZ band-resolved spectrum by rotating to band basis and unfolding the IBZ via irrk_inv."""
    giwk = GreensFunction(mat.copy(), nk=nk).cut_niv(niv_core).to_half_niv_range()
    rotated = orbital_to_band_basis(hk.copy(), giwk.mat.copy())
    nk_tot = int(np.prod(nk))
    rotated = rotated.reshape(nk_tot, n_bands, n_bands, -1)[k_grid.irrk_ind]  # [nk_irr, o, o, v]
    band_diag = np.imag(np.einsum("knnv->knv", rotated)[..., 0])  # [nk_irr, n_bands]
    return band_diag[k_grid.irrk_inv].reshape(nk_tot, n_bands)  # [nk_tot, n_bands]


def _orbital_diag_spectrum(mat, nk, n_bands, niv_core, k_grid):
    """Same as above but WITHOUT the band rotation (the old, orbital-diagonal behavior)."""
    giwk = GreensFunction(mat.copy(), nk=nk).cut_niv(niv_core).to_half_niv_range()
    nk_tot = int(np.prod(nk))
    arr = giwk.mat.reshape(nk_tot, n_bands, n_bands, -1)[k_grid.irrk_ind]
    orb_diag = np.imag(np.einsum("knnv->knv", arr)[..., 0])
    return orb_diag[k_grid.irrk_inv].reshape(nk_tot, n_bands)


@pytest.fixture
def patch_maxent_mpi(monkeypatch):
    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    monkeypatch.setattr(max_ent, "AnalyticContinuationProblem", _fake_problem)


@pytest.mark.parametrize("size", [1, 2])
def test_perform_maxent_giwk_continues_band_diagonal_and_unfolds(tmp_path, patch_maxent_mpi, size):
    """perform_maxent_giwk continues the band-diagonal G and unfolds it, unlike orbital-diagonal continuation."""
    nk, n_bands, niv_core, w_count = (4, 4, 1), 2, 3, 7
    hk = _setup_maxent_config(tmp_path, nk, n_bands, niv_core, w_count, seed=7)
    mat = _build_giwk_mat(nk, n_bands, niv=4, seed=11)
    k_grid = config.lattice.k_grid

    def fn(comm, rank):
        return max_ent.perform_maxent_giwk(GreensFunction(mat.copy(), nk=config.lattice.nk), "TEST", comm)

    _, results = run_parallel(size, fn)
    nk_tot = int(np.prod(nk))
    spectrum = results[0].reshape(nk_tot, n_bands, w_count)

    expected = _expected_band_spectrum(mat, hk, nk, n_bands, niv_core, k_grid)
    # The mock spreads the band-diagonal value across all real frequencies.
    assert np.allclose(spectrum, expected[:, :, None], atol=1e-5)

    orbital = _orbital_diag_spectrum(mat, nk, n_bands, niv_core, k_grid)
    assert not np.allclose(spectrum, orbital[:, :, None], atol=1e-5)


def test_perform_maxent_giwk_unfolds_symmetry_equivalent_kpoints(tmp_path, patch_maxent_mpi):
    """perform_maxent_giwk gives every full-BZ k-point its IBZ representative's spectrum."""
    nk, n_bands = (4, 4, 1), 2
    _setup_maxent_config(tmp_path, nk, n_bands, seed=3)
    mat = _build_giwk_mat(nk, n_bands, niv=4, seed=13)
    k_grid = config.lattice.k_grid

    def fn(comm, rank):
        return max_ent.perform_maxent_giwk(GreensFunction(mat.copy(), nk=config.lattice.nk), "TEST", comm)

    _, results = run_parallel(1, fn)
    nk_tot = int(np.prod(nk))
    spectrum = results[0].reshape(nk_tot, n_bands, -1)

    flat_rep = k_grid.irrk_ind[k_grid.irrk_inv].reshape(-1)  # representative flat index per FBZ point
    for k in range(nk_tot):
        assert np.allclose(spectrum[k], spectrum[flat_rep[k]], atol=1e-6)


def test_perform_maxent_giwk_failed_continuation_logs_kpoint_and_yields_zeros(tmp_path, monkeypatch):
    """perform_maxent_giwk logs a per-k-point error (not a stack trace) and yields zeros when continuation raises."""
    nk, n_bands, w_count = (4, 4, 1), 2, 6
    _setup_maxent_config(tmp_path, nk, n_bands, w_count=w_count, seed=4)
    mat = _build_giwk_mat(nk, n_bands, niv=4, seed=19)

    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    monkeypatch.setattr(
        max_ent, "AnalyticContinuationProblem", MagicMock(side_effect=RuntimeError("continuation failed"))
    )

    def fn(comm, rank):
        return max_ent.perform_maxent_giwk(GreensFunction(mat.copy(), nk=config.lattice.nk), "TEST", comm)

    _, results = run_parallel(1, fn)
    assert np.array_equal(results[0], np.zeros_like(results[0]))

    # one log per failed (irreducible k-point, band), each naming the k-point and the A=0 fallback
    fail_msgs = [
        call.args[0]
        for call in config.logger.info.call_args_list
        if "Failed to determine analytic continuation of k=" in call.args[0]
    ]
    assert len(fail_msgs) == config.lattice.k_grid.nk_irr * n_bands
    assert all("setting A(k=" in m and "= 0.0" in m for m in fail_msgs)


def test_perform_maxent_giwk_reroutes_solver_prints_to_logger(tmp_path, monkeypatch):
    """The vendored solver's stdout is captured and re-logged as 'ana_cont: <message>', not leaked to stdout."""
    nk, n_bands = (4, 4, 1), 1
    _setup_maxent_config(tmp_path, nk, n_bands, seed=6)
    mat = _build_giwk_mat(nk, n_bands, niv=4, seed=23)

    def solve(*args, **kwargs):
        print("Fermi fit failed.")
        return (SimpleNamespace(A_opt=np.zeros(config.ana_cont.w_count)),)

    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    monkeypatch.setattr(max_ent, "AnalyticContinuationProblem", _problem_from_solve(solve))

    def fn(comm, rank):
        return max_ent.perform_maxent_giwk(GreensFunction(mat.copy(), nk=config.lattice.nk), "TEST", comm)

    run_parallel(1, fn)
    info_msgs = [call.args[0] for call in config.logger.info.call_args_list]
    assert "ana_cont: Fermi fit failed." in info_msgs


def test_perform_maxent_giwk_runtime_warning_is_treated_as_failure(tmp_path, monkeypatch):
    """A numpy/scipy RuntimeWarning during the continuation is escalated to a failure: A(k, w) = 0."""
    nk, n_bands = (4, 4, 1), 1
    _setup_maxent_config(tmp_path, nk, n_bands, seed=8)
    mat = _build_giwk_mat(nk, n_bands, niv=4, seed=29)

    def solve(*args, **kwargs):
        warnings.warn("overflow encountered", RuntimeWarning)
        return (SimpleNamespace(A_opt=np.ones(config.ana_cont.w_count)),)

    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    monkeypatch.setattr(max_ent, "AnalyticContinuationProblem", _problem_from_solve(solve))

    def fn(comm, rank):
        return max_ent.perform_maxent_giwk(GreensFunction(mat.copy(), nk=config.lattice.nk), "TEST", comm)

    _, results = run_parallel(1, fn)
    assert np.array_equal(results[0], np.zeros_like(results[0]))
    fail_msgs = [c.args[0] for c in config.logger.info.call_args_list if "Failed to determine" in c.args[0]]
    assert len(fail_msgs) == config.lattice.k_grid.nk_irr * n_bands


def test_perform_maxent_giwk_optimize_warning_is_suppressed(tmp_path, monkeypatch):
    """An OptimizeWarning from the solver's alpha fit is muted and does not fail the continuation."""
    from scipy.optimize import OptimizeWarning

    nk, n_bands = (4, 4, 1), 1
    _setup_maxent_config(tmp_path, nk, n_bands, seed=10)
    mat = _build_giwk_mat(nk, n_bands, niv=4, seed=31)

    def solve(*args, **kwargs):
        warnings.warn("Covariance of the parameters could not be estimated", OptimizeWarning)
        return (SimpleNamespace(A_opt=np.ones(config.ana_cont.w_count)),)

    monkeypatch.setattr(mpi_utils, "MPI", FAKE_MPI)
    monkeypatch.setattr(max_ent, "AnalyticContinuationProblem", _problem_from_solve(solve))

    def fn(comm, rank):
        return max_ent.perform_maxent_giwk(GreensFunction(mat.copy(), nk=config.lattice.nk), "TEST", comm)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _, results = run_parallel(1, fn)

    assert not any(issubclass(r.category, OptimizeWarning) for r in recorded)
    assert np.any(results[0] != 0.0)


def test_perform_maxent_dmft_optimize_warning_is_suppressed(tmp_path, monkeypatch):
    """perform_maxent_dmft mutes the solver's OptimizeWarning rather than leaking it."""
    from scipy.optimize import OptimizeWarning

    nk, n_bands, w_count, niv = (3, 3, 1), 1, 7, 4
    config.sys.beta = 12.0
    config.sys.mu = 0.4
    config.sys.n_bands = n_bands
    config.ana_cont.w_count = w_count
    config.output.output_path = str(tmp_path)
    config.logger = MagicMock()

    hk = _build_hk(nk, n_bands, seed=33)
    sig = np.zeros((1, 1, 1, n_bands, n_bands, 2 * niv), dtype=np.complex64)
    sig[0, 0, 0, 0, 0] = 0.2 - 0.1j * np.ones(2 * niv)
    sigma_dmft = SelfEnergy(sig, nk=(1, 1, 1), full_niv_range=True, calc_smom=False, beta=config.sys.beta)

    def solve(*args, **kwargs):
        warnings.warn("Covariance of the parameters could not be estimated", OptimizeWarning)
        return (SimpleNamespace(A_opt=np.ones(w_count)),)

    monkeypatch.setattr(max_ent, "AnalyticContinuationProblem", _problem_from_solve(solve))
    monkeypatch.setattr(max_ent, "RealFreqTwoPoint", _fake_real_freq_two_point)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        spectrum = max_ent.perform_maxent_dmft(sigma_dmft, hk)

    assert not any(issubclass(r.category, OptimizeWarning) for r in recorded)
    assert spectrum.shape == (*nk, n_bands, w_count)


def test_perform_maxent_giwk_single_band(tmp_path, patch_maxent_mpi):
    """perform_maxent_giwk reduces to the orbital-diagonal continuation for a single band."""
    nk, n_bands, w_count = (4, 4, 1), 1, 5
    hk = _setup_maxent_config(tmp_path, nk, n_bands, w_count=w_count, seed=2)
    mat = _build_giwk_mat(nk, n_bands, niv=4, seed=17)
    k_grid = config.lattice.k_grid

    def fn(comm, rank):
        return max_ent.perform_maxent_giwk(GreensFunction(mat.copy(), nk=config.lattice.nk), "TEST", comm)

    _, results = run_parallel(1, fn)
    nk_tot = int(np.prod(nk))
    spectrum = results[0].reshape(nk_tot, n_bands, w_count)

    # For a single band the rotation is trivial: band-diagonal == orbital-diagonal.
    expected = _expected_band_spectrum(mat, hk, nk, n_bands, config.box.niv_core, k_grid)
    assert np.allclose(spectrum, expected[:, :, None], atol=1e-5)


def test_perform_maxent_dmft_builds_full_bz_spectral_function(tmp_path, monkeypatch):
    """perform_maxent_dmft builds a real, non-negative full-BZ spectral function."""
    nk, n_bands, w_count, niv = (3, 3, 1), 2, 9, 4
    config.sys.beta = 12.0
    config.sys.mu = 0.4
    config.sys.n_bands = n_bands
    config.ana_cont.w_count = w_count
    config.output.output_path = str(tmp_path)
    config.logger = MagicMock()

    hk = _build_hk(nk, n_bands, seed=21)

    sig = np.zeros((1, 1, 1, n_bands, n_bands, 2 * niv), dtype=np.complex64)
    for b in range(n_bands):
        sig[0, 0, 0, b, b] = (0.2 * (b + 1)) - 0.1j * np.ones(2 * niv)
    sigma_dmft = SelfEnergy(sig, nk=(1, 1, 1), full_niv_range=True, calc_smom=False, beta=config.sys.beta)

    monkeypatch.setattr(max_ent, "AnalyticContinuationProblem", _fake_problem)
    monkeypatch.setattr(max_ent, "RealFreqTwoPoint", _fake_real_freq_two_point)

    spectrum = max_ent.perform_maxent_dmft(sigma_dmft, hk)

    assert spectrum.shape == (*nk, n_bands, w_count)
    # A spectral function is real and non-negative.
    assert np.all(spectrum >= -1e-6)
    assert np.isrealobj(spectrum)

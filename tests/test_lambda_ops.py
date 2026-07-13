# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
from unittest.mock import patch

import mpi4py
import numpy as np
import pytest

from dgamore.lambda_ops import LambdaCorrection, MultiOrbitalLambdaCorrection
from dgamore.dga_logger import DgaLogger
from dgamore.four_point import FourPoint
from dgamore.local_four_point import LocalFourPoint
from dgamore.n_point_base import SpinChannel
import dgamore.config as config
import dgamore.brillouin_zone as bz


def load_four_point(lc_type: str, filename: str, channel: SpinChannel) -> FourPoint:
    return FourPoint.load(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/lambda_correction/"
        + lc_type
        + "/"
        + filename
        + ".npy",
        channel,
        num_vn_dimensions=0,
        has_compressed_q_dimension=True,
        nq=(4, 4, 1),
    )


def load_local_four_point(lc_type: str, filename: str, channel: SpinChannel) -> LocalFourPoint:
    return LocalFourPoint.load(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/lambda_correction/"
        + lc_type
        + "/"
        + filename
        + ".npy",
        channel,
        num_vn_dimensions=0,
    )


def test_lambda_correction_spch():
    """spch-type single lambda correction reproduces reference chi and lambdas for dens and magn channels; the dens
    channel stalls at the divergence bound and warns, so the logger must be mocked for a standalone run."""
    from unittest.mock import MagicMock

    config.lattice.k_grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.sys.beta = 12.5
    config.logger = MagicMock()

    def perform_lc(channel: SpinChannel):
        chi_r_before_lambda = load_four_point("spch", f"chi_phys_q_{channel.value}_before_lambda", channel)
        chi_r_loc = load_local_four_point("spch", f"chi_{channel.value}_loc", channel).to_full_niw_range()
        chi_r_loc_sum = chi_r_loc.mat.sum() / config.sys.beta
        return LambdaCorrection.perform_single(chi_r_before_lambda, chi_r_loc_sum)

    corrected_chi_dens, lambda_dens = perform_lc(SpinChannel.DENS)
    corrected_chi_magn, lambda_magn = perform_lc(SpinChannel.MAGN)

    reference_chi_dens = load_four_point("spch", "chi_phys_q_dens", SpinChannel.DENS)
    reference_chi_magn = load_four_point("spch", "chi_phys_q_magn", SpinChannel.MAGN)

    assert np.allclose(corrected_chi_dens.mat, reference_chi_dens.mat)
    assert np.allclose(corrected_chi_magn.mat, reference_chi_magn.mat)

    assert np.allclose(lambda_dens, -37.450340)
    assert np.allclose(lambda_magn, 4.328781)


def test_lambda_correction_sp():
    """sp-type single lambda correction (magn only) reproduces reference chi and lambda."""
    config.lattice.k_grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.sys.beta = 12.5

    chi_dens_q_before_lambda = load_four_point(
        "sp", f"chi_phys_q_dens_before_lambda", SpinChannel.DENS
    ).to_full_niw_range()
    chi_phys_q_dens = load_four_point("sp", f"chi_phys_q_dens", SpinChannel.DENS).to_full_niw_range()
    assert np.allclose(chi_dens_q_before_lambda.mat, chi_phys_q_dens.mat)

    chi_dens_loc_sum = load_local_four_point("sp", f"chi_dens_loc", SpinChannel.DENS).to_full_niw_range().mat.sum()
    chi_magn_loc_sum = load_local_four_point("sp", f"chi_magn_loc", SpinChannel.MAGN).to_full_niw_range().mat.sum()

    chi_loc_sum = (
        chi_dens_loc_sum
        + chi_magn_loc_sum
        - 1.0 / 16 * (config.lattice.k_grid.irrk_count[:, None, None, None, None, None] * chi_phys_q_dens.mat).sum()
    )

    chi_magn_q_before_lambda = load_four_point("sp", f"chi_phys_q_magn_before_lambda", SpinChannel.MAGN)
    corrected_chi_magn, lambda_magn = LambdaCorrection.perform_single(
        chi_magn_q_before_lambda, chi_loc_sum / config.sys.beta
    )

    reference_chi_magn = load_four_point("sp", "chi_phys_q_magn", SpinChannel.MAGN)

    assert np.allclose(corrected_chi_magn.mat, reference_chi_magn.mat)

    assert np.allclose(lambda_magn, 4.281153)


@pytest.mark.parametrize("lc_type", ["sp", "spch"])
def test_lambda_correction_in_sde_sp(lc_type):
    """LambdaCorrection.perform reproduces reference chi for both sp and spch types."""
    config.lattice.k_grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.sys.beta = 12.5
    config.lambda_correction.type = lc_type
    config.output.output_path = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/lambda_correction/{lc_type}"

    with patch("mpi4py.MPI.COMM_WORLD", wraps=mpi4py.MPI.COMM_WORLD) as comm_mock:
        config.logger = DgaLogger(comm_mock, "./")

        chi_dens_q_before_lambda = load_four_point(lc_type, f"chi_phys_q_dens_before_lambda", SpinChannel.DENS)
        chi_magn_q_before_lambda = load_four_point(lc_type, f"chi_phys_q_magn_before_lambda", SpinChannel.MAGN)

        chi_magn_corrected = LambdaCorrection.perform(chi_magn_q_before_lambda)
        chi_dens_corrected = LambdaCorrection.perform(chi_dens_q_before_lambda)

        reference_chi_dens = load_four_point(lc_type, "chi_phys_q_dens", SpinChannel.DENS)
        reference_chi_magn = load_four_point(lc_type, "chi_phys_q_magn", SpinChannel.MAGN)

        assert np.allclose(chi_dens_corrected.mat, reference_chi_dens.mat)
        assert np.allclose(chi_magn_corrected.mat, reference_chi_magn.mat)

        if lc_type == "sp":
            assert np.allclose(chi_dens_corrected.mat, chi_dens_q_before_lambda.mat)


def test_find_lambda_warns_on_non_convergence():
    """find_lambda warns and returns a finite value when the target is unreachable within maxiter."""
    from unittest.mock import MagicMock

    config.lattice.k_grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.sys.beta = 12.5
    config.logger = MagicMock()

    n_irrk = config.lattice.k_grid.irrk_count.shape[0]
    chi = np.full((n_irrk, 5), 0.5, dtype=np.complex64)  # [irrk, w], finite -> lambda_start finite
    result = LambdaCorrection.find_lambda(chi, 1e6 + 0j, maxiter=1)  # unreachable target in one iteration

    assert np.isfinite(result)
    assert config.logger.warning.called


def _synthetic_chi_inv_qw(n_bands: int, nq: int, nw: int, seed: int = 0) -> np.ndarray:
    """Physically-structured inverse compound susceptibility ``[nq, nw, No2, No2]``: real-symmetric (P3), SPD, with
    an :math:`\\omega^2` growth so the domain binds at :math:`\\omega=0` and the sum-rule map is monotone (unique
    root), like the real ladder susceptibility."""
    rng = np.random.default_rng(seed)
    no2 = n_bands**2
    w0 = nw // 2
    out = np.empty((nq, nw, no2, no2), dtype=np.complex128)
    for q in range(nq):
        a = rng.standard_normal((no2, no2))
        base = 0.5 * (a + a.T) + (no2 + 3.0) * np.eye(no2)
        for w in range(nw):
            out[q, w] = base + (1.5 * (w - w0)) ** 2 * np.eye(no2)
    return out


def _synthetic_chi_qw(n_bands: int, nq: int, nw: int, seed: int = 0) -> np.ndarray:
    """The susceptibility belonging to :func:`_synthetic_chi_inv_qw` (same seed, per-block inverse): the input the
    resolvent-form calibration consumes directly."""
    return np.linalg.inv(_synthetic_chi_inv_qw(n_bands, nq, nw, seed))


def test_find_lambda_matrix_recovers_planted_lambda():
    """The matrix Newton solver recovers a planted real-symmetric mass from the sum-rule target it generates."""
    beta, nq, nw, n_bands = 5.0, 6, 5, 2
    chi_inv_qw = _synthetic_chi_inv_qw(n_bands, nq, nw)
    no2 = n_bands**2
    a = np.random.default_rng(1).standard_normal((no2, no2))
    lambda_star = 0.3 * 0.5 * (a + a.T)
    s_r = np.linalg.inv(chi_inv_qw + lambda_star).sum(axis=(0, 1)).real / (beta * nq)

    lambda_found = MultiOrbitalLambdaCorrection.find_lambda_matrix(_synthetic_chi_qw(n_bands, nq, nw), s_r, beta, nq)

    assert np.allclose(lambda_found, lambda_star, atol=1e-4)


def test_find_lambda_matrix_solution_is_symmetric_feasible_and_off_diagonal():
    """The converged mass is real symmetric, genuinely matrix-valued, feasible, and zeroes the sum-rule residual."""
    beta, nq, nw, n_bands = 8.0, 5, 5, 2
    chi_inv_qw = _synthetic_chi_inv_qw(n_bands, nq, nw, seed=3)
    no2 = n_bands**2
    a = np.random.default_rng(4).standard_normal((no2, no2))
    lambda_star = 0.2 * 0.5 * (a + a.T)
    s_r = np.linalg.inv(chi_inv_qw + lambda_star).sum(axis=(0, 1)).real / (beta * nq)
    chi_qw = _synthetic_chi_qw(n_bands, nq, nw, seed=3)

    lam = MultiOrbitalLambdaCorrection.find_lambda_matrix(chi_qw, s_r, beta, nq)

    assert np.allclose(lam, lam.T)
    assert not np.allclose(lam, np.diag(np.diag(lam)))
    assert np.linalg.norm(MultiOrbitalLambdaCorrection._residual(chi_qw, lam, s_r, beta, nq)) < 1e-6
    assert MultiOrbitalLambdaCorrection._static_gap(chi_inv_qw, lam) > 0.0


def test_matrix_jacobian_is_negative_definite_on_symmetric_subspace():
    """The reduced Newton Jacobian is negative definite: the sum-rule map is strictly monotone (unique root)."""
    beta, nq, nw, n_bands = 8.0, 5, 5, 2
    chi_qw = _synthetic_chi_qw(n_bands, nq, nw, seed=5)
    no2 = n_bands**2
    _, jac = MultiOrbitalLambdaCorrection.residual_and_jacobian(
        chi_qw, 0.5 * np.eye(no2), np.zeros((no2, no2)), beta, nq
    )
    basis = MultiOrbitalLambdaCorrection._symmetric_basis(no2)
    hess = basis.T @ jac.reshape(no2 * no2, no2 * no2) @ basis

    assert float(np.linalg.eigvalsh(hess).max()) < 0.0


def test_find_lambda_matrix_warns_on_non_convergence():
    """An unreachable target makes the Newton solver exhaust its steps, warn and return a finite symmetric mass."""
    from unittest.mock import MagicMock

    config.logger = MagicMock()
    beta, nq, nw, n_bands = 5.0, 6, 5, 2
    chi_qw = _synthetic_chi_qw(n_bands, nq, nw)
    no2 = n_bands**2
    s_r = 1e6 * np.eye(no2)  # far above any achievable susceptibility sum

    lam = MultiOrbitalLambdaCorrection.find_lambda_matrix(chi_qw, s_r, beta, nq, maxiter=5)

    assert np.all(np.isfinite(lam)) and np.allclose(lam, lam.T)
    assert config.logger.warning.called


def test_find_lambda_matrix_falls_back_to_lstsq_on_singular_hessian():
    """When the Newton solve reports a singular Hessian the step is recovered by least squares (same root)."""
    beta, nq, nw, n_bands = 5.0, 6, 5, 2
    chi_inv_qw = _synthetic_chi_inv_qw(n_bands, nq, nw)
    no2 = n_bands**2
    lambda_star = 0.1 * np.eye(no2)
    s_r = np.linalg.inv(chi_inv_qw + lambda_star).sum(axis=(0, 1)).real / (beta * nq)

    with patch("numpy.linalg.solve", side_effect=np.linalg.LinAlgError("singular")):
        lam = MultiOrbitalLambdaCorrection.find_lambda_matrix(_synthetic_chi_qw(n_bands, nq, nw), s_r, beta, nq)

    assert np.allclose(lam, lambda_star, atol=1e-4)


def test_multiorbital_perform_single_matches_single_band_for_magn():
    """For the cleanly-converging magnetic channel, the No=1 matrix correction coincides with the scalar
    LambdaCorrection (same lambda and corrected chi on the spch reference data)."""
    config.lattice.k_grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.sys.beta = 12.5
    config.sys.n_bands = 1

    chi_before = load_four_point("spch", "chi_phys_q_magn_before_lambda", SpinChannel.MAGN)
    chi_loc = load_local_four_point("spch", "chi_magn_loc", SpinChannel.MAGN).to_full_niw_range()
    s_r = np.array([[chi_loc.mat.sum() / config.sys.beta]], dtype=np.complex128)

    corrected, lambda_mat = MultiOrbitalLambdaCorrection.perform_single(chi_before.copy(), s_r)
    ref_chi, _ = LambdaCorrection.perform_single(chi_before.copy(), chi_loc.mat.sum() / config.sys.beta)

    assert np.allclose(lambda_mat[0, 0], 4.328781, atol=1e-4)
    assert np.allclose(corrected.mat, ref_chi.mat, atol=1e-4)


@pytest.mark.parametrize("channel", [SpinChannel.DENS, SpinChannel.MAGN])
def test_multiorbital_perform_single_satisfies_sum_rule(channel):
    """The corrected susceptibility's full-BZ frequency sum matches the local target for BOTH channels - including
    density, where the scalar single-band solver stalls at the divergence bound instead of the true root."""
    config.lattice.k_grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.sys.beta = 12.5
    config.sys.n_bands = 1

    chi_before = load_four_point("spch", f"chi_phys_q_{channel.value}_before_lambda", channel)
    chi_loc = load_local_four_point("spch", f"chi_{channel.value}_loc", channel).to_full_niw_range()
    s_r = np.array([[chi_loc.mat.sum() / config.sys.beta]], dtype=np.complex128)

    corrected, _ = MultiOrbitalLambdaCorrection.perform_single(chi_before.copy(), s_r)

    q_grid = config.lattice.k_grid
    lhs = corrected.to_full_niw_range().map_to_full_bz(q_grid).mat.sum() / (config.sys.beta * q_grid.nk_tot)
    assert np.allclose(lhs, s_r[0, 0], atol=1e-6)


def _multiorbital_chi_fourpoint(n_bands: int, nq_grid: tuple, nw_half: int, seed: int) -> FourPoint:
    """Physical two-index-per-leg chi^q FourPoint on the IBZ (no fermionic frequencies): its compound inverse is a
    real-symmetric SPD block with omega^2 growth, so the sum-rule root is unique."""
    config.lattice.k_grid = bz.KGrid(nk=nq_grid, symmetries=bz.two_dimensional_square_symmetries())
    rng = np.random.default_rng(seed)
    n_irr = config.lattice.k_grid.irrk_count.shape[0]
    no2, nw = n_bands**2, 2 * nw_half + 1
    mat = np.empty((n_irr, n_bands, n_bands, n_bands, n_bands, nw), dtype=np.complex128)
    for q in range(n_irr):
        a = rng.standard_normal((no2, no2))
        base = 0.5 * (a + a.T) + (no2 + 3.0) * np.eye(no2)
        for w in range(nw):
            chi_comp = np.linalg.inv(base + (1.5 * (w - nw_half)) ** 2 * np.eye(no2))  # chi = (chi^-1)^-1
            mat[q, ..., w] = chi_comp.reshape(n_bands, n_bands, n_bands, n_bands).transpose(0, 1, 3, 2)
    return FourPoint(
        mat, SpinChannel.DENS, nq=nq_grid, num_wn_dimensions=1, num_vn_dimensions=0, has_compressed_q_dimension=True
    )


def test_multiorbital_perform_single_recovers_matrix_mass_and_sum_rule():
    """Two-band object path: perform_single recovers a planted matrix mass and its corrected chi satisfies the
    full-BZ sum rule (exercises map_to_full_bz + compound inversion for No > 1)."""
    config.sys.beta = 8.0
    config.sys.n_bands = 2
    chi = _multiorbital_chi_fourpoint(2, (4, 4, 1), 3, seed=7)
    q_grid = config.lattice.k_grid
    no2 = 4
    a = np.random.default_rng(9).standard_normal((no2, no2))
    lambda_star = 0.2 * 0.5 * (a + a.T)
    chi_inv_qw = np.linalg.inv(chi.copy().map_to_full_bz(q_grid).to_compound_indices().mat.astype(np.complex128))
    s_r = np.linalg.inv(chi_inv_qw + lambda_star).sum(axis=(0, 1)).real / (config.sys.beta * q_grid.nk_tot)

    corrected, lambda_found = MultiOrbitalLambdaCorrection.perform_single(chi.copy(), s_r)

    assert np.allclose(lambda_found, lambda_star, atol=1e-4)
    lhs = corrected.to_full_niw_range().map_to_full_bz(q_grid).to_compound_indices().mat.astype(np.complex128).sum(
        axis=(0, 1)
    ) / (config.sys.beta * q_grid.nk_tot)
    assert np.allclose(lhs, s_r, atol=1e-5)


def test_multiorbital_perform_loads_target_writes_lambda_and_corrects(tmp_path):
    """perform loads the channel's local sum-rule target, corrects the susceptibility and appends ||Lambda|| to the
    lambda file (single-band spch reference data in an isolated run directory)."""
    import shutil
    from unittest.mock import MagicMock

    config.lattice.k_grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.sys.beta = 12.5
    config.sys.n_bands = 1
    config.logger = MagicMock()
    src = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/lambda_correction/spch"
    shutil.copy(f"{src}/chi_magn_loc.npy", tmp_path / "chi_magn_loc.npy")
    config.output.output_path = str(tmp_path)

    chi_before = load_four_point("spch", "chi_phys_q_magn_before_lambda", SpinChannel.MAGN)
    corrected = MultiOrbitalLambdaCorrection.perform(chi_before, quiet=False)

    assert corrected.channel == SpinChannel.MAGN
    lambda_file = tmp_path / "lambda_trial.txt"
    assert lambda_file.exists()
    assert "lambda_magn_fro" in lambda_file.read_text()


def test_density_diagonal_sum_reduces_to_frequency_sum_for_q_constant_chi():
    """C_{r;a} = (1/(beta N_q)) sum_qw chi_aaaa reduces to (1/beta) sum_w chi_aaaa when chi is momentum-constant."""
    config.lattice.k_grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.sys.beta = 5.0
    n_irr, nw, n_bands = config.lattice.k_grid.irrk_count.shape[0], 5, 2
    single_q = np.random.default_rng(3).standard_normal((n_bands, n_bands, n_bands, n_bands, nw)) + 0.5
    mat = np.broadcast_to(single_q, (n_irr, *single_q.shape)).astype(np.complex128).copy()
    chi = FourPoint(
        mat, SpinChannel.DENS, nq=(4, 4, 1), num_wn_dimensions=1, num_vn_dimensions=0, has_compressed_q_dimension=True
    )

    result = MultiOrbitalLambdaCorrection._density_diagonal_sum(chi)

    expected = np.array([single_q[a, a, a, a, :].sum() for a in range(n_bands)]) / config.sys.beta
    assert np.allclose(result, expected, atol=1e-5)


def test_log_pauli_diagnostic_logs_largest_diagonal_deviation():
    """The Pauli diagnostic logs only the largest-magnitude diagonal deviation 1/2(C_d+C_m)_1 - n/2(1-n/2), one line."""
    from unittest.mock import MagicMock

    config.sys.beta = 5.0
    config.sys.n_bands = 2
    config.dmft.ineq_ordering = [1]
    config.dmft.n_bands_per_ineq = [2]
    config.sys.occ_dmft_per_ineq = [np.diag([0.8, 1.2]).astype(np.complex128)]
    config.output.output_path = "/run/dir"
    config.logger = MagicMock()

    chi_magn = _multiorbital_chi_fourpoint(2, (4, 4, 1), 2, seed=1)
    chi_dens = _multiorbital_chi_fourpoint(2, (4, 4, 1), 2, seed=2)
    c_d = MultiOrbitalLambdaCorrection._density_diagonal_sum(chi_dens)
    c_m = MultiOrbitalLambdaCorrection._density_diagonal_sum(chi_magn)

    with patch("dgamore.lambda_ops.os.path.exists", return_value=True), patch.object(
        FourPoint, "load", return_value=chi_dens
    ):
        MultiOrbitalLambdaCorrection._log_pauli_diagnostic(chi_magn)

    deviations = [0.5 * (c_d[a] + c_m[a]) - 0.5 * occ * (1.0 - 0.5 * occ) for a, occ in enumerate((0.8, 1.2))]
    worst = max(deviations, key=abs)
    assert config.logger.info.call_count == 1
    assert f"{worst:+.6e}" in config.logger.info.call_args_list[0].args[0]


def test_log_pauli_diagnostic_skips_without_occupation_metadata():
    """The Pauli diagnostic is skipped (no logging) when the per-inequivalent occupation metadata is absent."""
    from unittest.mock import MagicMock

    config.sys.occ_dmft_per_ineq = []
    config.output.output_path = "/run/dir"
    config.logger = MagicMock()

    chi_magn = _multiorbital_chi_fourpoint(2, (4, 4, 1), 2, seed=1)
    MultiOrbitalLambdaCorrection._log_pauli_diagnostic(chi_magn)

    assert config.logger.info.call_count == 0


def test_lambda_correction_perform_raises_on_unknown_type():
    """LambdaCorrection.perform rejects a lambda-correction type that is neither 'spch' nor 'sp'."""
    from unittest.mock import MagicMock

    config.logger = MagicMock()
    config.lambda_correction.type = "bogus"
    with pytest.raises(ValueError):
        LambdaCorrection.perform(None)


def test_find_lambda_matrix_feasibility_binds_at_omega_zero_only():
    """A high-|w| ladder-tail slice with a deeply negative Hermitian eigenvalue must not inflate the feasibility
    bound: the planted mass violates the all-frequency bound but is still recovered because only the static
    (w=0) slice binds (regression: an all-w bound pushed lambda_start orders of magnitude above the true root)."""
    beta, nq, nw, n_bands = 5.0, 6, 5, 2
    chi_inv_qw = _synthetic_chi_inv_qw(n_bands, nq, nw)
    no2 = n_bands**2
    chi_inv_qw[:, -1] = -50.0 * np.eye(no2)
    a = np.random.default_rng(11).standard_normal((no2, no2))
    lambda_star = 0.3 * 0.5 * (a + a.T)
    s_r = np.linalg.inv(chi_inv_qw + lambda_star).sum(axis=(0, 1)).real / (beta * nq)

    lambda_found = MultiOrbitalLambdaCorrection.find_lambda_matrix(np.linalg.inv(chi_inv_qw), s_r, beta, nq)

    assert np.allclose(lambda_found, lambda_star, atol=1e-4)
    shifted = chi_inv_qw + lambda_star
    herm_all_w = 0.5 * (shifted + np.conj(np.swapaxes(shifted, -1, -2)))
    assert float(np.linalg.eigvalsh(herm_all_w).min()) < 0.0


def test_log_pauli_diagnostic_scans_multiple_inequivalent_atoms():
    """With two single-band inequivalent atoms the diagnostic reports the single worst deviation across atoms,
    labeled with the correct inequivalent-atom index and global orbital offset."""
    from unittest.mock import MagicMock

    config.sys.beta = 5.0
    config.sys.n_bands = 2
    config.dmft.ineq_ordering = [1, 2]
    config.dmft.n_bands_per_ineq = [1, 1]
    config.sys.occ_dmft_per_ineq = [np.array([[0.8 + 0j]]), np.array([[1.6 + 0j]])]
    config.output.output_path = "/run/dir"
    config.logger = MagicMock()

    chi_magn = _multiorbital_chi_fourpoint(2, (4, 4, 1), 2, seed=1)
    chi_dens = _multiorbital_chi_fourpoint(2, (4, 4, 1), 2, seed=2)
    c_d = MultiOrbitalLambdaCorrection._density_diagonal_sum(chi_dens)
    c_m = MultiOrbitalLambdaCorrection._density_diagonal_sum(chi_magn)

    with patch("dgamore.lambda_ops.os.path.exists", return_value=True), patch.object(
        FourPoint, "load", return_value=chi_dens
    ):
        MultiOrbitalLambdaCorrection._log_pauli_diagnostic(chi_magn)

    deviations = [0.5 * (c_d[a] + c_m[a]) - 0.5 * occ * (1.0 - 0.5 * occ) for a, occ in enumerate((0.8, 1.6))]
    worst_orbital = int(np.argmax(np.abs(deviations)))
    assert config.logger.info.call_count == 1
    message = config.logger.info.call_args_list[0].args[0]
    assert f"{deviations[worst_orbital]:+.6e}" in message
    assert f"ineq {worst_orbital + 1}, orbital {worst_orbital}" in message


def test_multiorbital_perform_quiet_writes_nothing_and_skips_pauli(tmp_path):
    """quiet=True (a stabilizer Jacobian probe) evaluates the correction unchanged but writes no lambda file and
    skips the Pauli diagnostic."""
    import shutil
    from unittest.mock import MagicMock

    config.lattice.k_grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.sys.beta = 12.5
    config.sys.n_bands = 1
    config.logger = MagicMock()
    src = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/lambda_correction/spch"
    shutil.copy(f"{src}/chi_magn_loc.npy", tmp_path / "chi_magn_loc.npy")
    config.output.output_path = str(tmp_path)

    chi_before = load_four_point("spch", "chi_phys_q_magn_before_lambda", SpinChannel.MAGN)
    with patch.object(MultiOrbitalLambdaCorrection, "_log_pauli_diagnostic") as pauli:
        corrected = MultiOrbitalLambdaCorrection.perform(chi_before, quiet=True)

    assert corrected.channel == SpinChannel.MAGN
    assert not (tmp_path / "lambda_trial.txt").exists()
    pauli.assert_not_called()


def _synthetic_auto_grid(n_bands: int = 2):
    """KGrid (2, 2, 1) with hand-built auto-symmetry data: two 2-point stars whose non-representative members carry
    a nontrivial orbital rotation (one of them antiunitary and one with sigma = -1), exercising every branch of the
    irreducible-to-full-BZ expansion."""
    grid = bz.KGrid(nk=(2, 2, 1), symmetries=[])
    fbz2irrk = np.array([0, 0, 2, 2])
    grid.fbz2irrk = fbz2irrk.reshape(2, 2, 1)
    _, grid.irrk_ind, grid.irrk_inv, grid.irrk_count = np.unique(
        fbz2irrk, return_index=True, return_inverse=True, return_counts=True
    )
    theta = 0.3
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.complex128)
    grid._auto_us = np.stack([np.eye(2), rot, np.eye(2), rot.conj().T]).reshape(2, 2, 1, n_bands, n_bands)
    grid._auto_sigmas = np.array([1.0, -1.0, 1.0, 1.0]).reshape(2, 2, 1)
    grid._auto_conjs = np.array([False, False, False, True]).reshape(2, 2, 1)
    grid._auto_mode = True
    return grid


def _chi_on_grid(grid, n_bands: int, nw_half: int, seed: int) -> FourPoint:
    """Physical chi^q FourPoint on the grid's irreducible wedge (0 fermionic frequencies): SPD real-symmetric
    compound inverse with omega^2 growth, invertible at every (q, w)."""
    rng = np.random.default_rng(seed)
    n_irr = grid.irrk_count.shape[0]
    no2, nw = n_bands**2, 2 * nw_half + 1
    mat = np.empty((n_irr, n_bands, n_bands, n_bands, n_bands, nw), dtype=np.complex128)
    for q in range(n_irr):
        a = rng.standard_normal((no2, no2))
        base = 0.5 * (a + a.T) + (no2 + 3.0) * np.eye(no2)
        for w in range(nw):
            chi_comp = np.linalg.inv(base + (1.5 * (w - nw_half)) ** 2 * np.eye(no2))
            mat[q, ..., w] = chi_comp.reshape(n_bands, n_bands, n_bands, n_bands).transpose(0, 1, 3, 2)
    return FourPoint(
        mat, SpinChannel.DENS, nq=grid.nk, num_wn_dimensions=1, num_vn_dimensions=0, has_compressed_q_dimension=True
    )


def test_expand_compound_slice_is_exact_gather_without_auto_data():
    """Without auto-symmetry data the FBZ expansion is the plain irrk_inv gather, bit-identical per bosonic
    frequency to what map_to_full_bz produces."""
    grid = bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.lattice.k_grid = grid
    chi = _chi_on_grid(grid, 2, 2, seed=5)
    expected = chi.copy().map_to_full_bz(grid).to_compound_indices().mat
    irr = chi.copy().to_compound_indices().mat
    for w in range(irr.shape[1]):
        out = MultiOrbitalLambdaCorrection._expand_compound_slice_to_full_bz(irr[:, w], grid)
        assert np.array_equal(out, expected[:, w])


def test_expand_compound_slice_matches_object_map_and_commutes_with_inversion():
    """In auto mode the raw-array FBZ expansion reproduces map_to_full_bz (gather + per-k orbital rotation incl.
    the antiunitary member) and commutes with the compound inversion to complex128 accuracy, so expanding the
    cached irreducible-wedge inverse is exact."""
    grid = _synthetic_auto_grid()
    chi = _chi_on_grid(grid, 2, 1, seed=6)
    expected = chi.copy().map_to_full_bz(grid).to_compound_indices().mat
    irr = chi.copy().to_compound_indices().mat
    for w in range(irr.shape[1]):
        out = MultiOrbitalLambdaCorrection._expand_compound_slice_to_full_bz(irr[:, w], grid)
        assert np.allclose(out, expected[:, w], atol=1e-6)
    slice128 = irr[:, 0].astype(np.complex128)
    expanded_inverse = MultiOrbitalLambdaCorrection._expand_compound_slice_to_full_bz(np.linalg.inv(slice128), grid)
    inverse_expanded = np.linalg.inv(MultiOrbitalLambdaCorrection._expand_compound_slice_to_full_bz(slice128, grid))
    assert np.allclose(expanded_inverse, inverse_expanded, atol=1e-12)


@pytest.mark.parametrize("auto", [False, True])
def test_find_lambda_matrix_ibz_resident_matches_full_bz_reference(auto):
    """The IBZ-resident calibration (q_grid passed, inverse cached on the wedge, per-w expansion) yields the same
    mass as the reference full-BZ-stack path: near-bit-exact on a plain-symmetry grid, and at the complex64
    transform-noise level on an auto grid with genuine orbital rotations."""
    beta, n_bands = 5.0, 2
    grid = _synthetic_auto_grid() if auto else bz.KGrid(nk=(4, 4, 1), symmetries=bz.two_dimensional_square_symmetries())
    config.lattice.k_grid = grid
    chi = _chi_on_grid(grid, n_bands, 2, seed=7)
    no2 = n_bands**2
    a = np.random.default_rng(8).standard_normal((no2, no2))
    lambda_star = 0.2 * 0.5 * (a + a.T)

    full_chi = chi.copy().map_to_full_bz(grid).to_compound_indices().mat.astype(np.complex128)
    s_r = np.linalg.inv(np.linalg.inv(full_chi) + lambda_star).sum(axis=(0, 1)).real / (beta * grid.nk_tot)
    lam_reference = MultiOrbitalLambdaCorrection.find_lambda_matrix(full_chi, s_r, beta, grid.nk_tot)

    irr_chi = chi.copy().to_compound_indices().mat.astype(np.complex128)
    lam_ibz = MultiOrbitalLambdaCorrection.find_lambda_matrix(irr_chi, s_r, beta, grid.nk_tot, q_grid=grid)

    assert np.allclose(lam_ibz, lam_reference, atol=1e-5 if auto else 1e-10)
    assert np.allclose(lam_ibz, lambda_star, atol=1e-4)

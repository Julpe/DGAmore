# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Lambda operations on the physical susceptibility - the two ways DGAmore shifts :math:`\chi^q_r` by a bosonic mass
:math:`\lambda`:

- :class:`LambdaCorrection` - the Moriya sum-rule :math:`\lambda`-correction: a single constant per channel chosen
  so the momentum sum of :math:`\chi^q_r` matches the local sum rule. Single-band only (a multi-orbital correction
  would be non-unique).
- :class:`MultiOrbitalLambdaCorrection` - the matrix-valued generalization of the sum-rule correction: a full
  real-symmetric :math:`N_o^2\times N_o^2` mass matrix :math:`\Lambda_r` per channel, calibrated so the momentum sum
  of :math:`\chi^q_r` matches the local sum rule component by component (a well-posed matrix root). Multi-orbital.
- :class:`LambdaAnnealer` - the lambda-annealing convergence scaffold: a bosonic mass measured from the
  susceptibility spectrum, held per convergence phase and annealed to zero, so the final result is pure
  self-consistency. Multi-orbital-safe.

All three add a bosonic mass to the inverse susceptibility (:math:`\chi_r \to (\chi_r^{-1} + \lambda)^{-1}`, with
:math:`\lambda` a scalar for the sum-rule and annealing schemes and a matrix :math:`\Lambda_r` for the multi-orbital
one); they differ only in how it is chosen and whether it survives into the final result.
"""

import os

import numpy as np
from mpi4py import MPI

from dgamore import config
from dgamore.four_point import FourPoint
from dgamore.local_four_point import LocalFourPoint
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel


class LambdaCorrection:
    r"""
    Moriya :math:`\lambda`-correction of the physical susceptibility. The non-local susceptibility is shifted by a
    single constant :math:`\lambda` (per channel) so its momentum sum matches the corresponding local sum rule.
    Single-band only, since a multi-orbital correction would be a non-unique multidimensional problem. All methods
    are stateless (the sum-rule targets and the determined :math:`\lambda` are read/written from files/config).
    """

    @staticmethod
    def get_lambda_start(chi_r: np.ndarray) -> float:
        r"""
        Returns the lower bound for :math:`\lambda`, i.e. the value at which the corrected susceptibility at
        :math:`\omega = 0` would first diverge (:math:`-\min_q 1/\chi_r^{q,\omega=0}`). The search starts just
        above it.

        :param chi_r: Physical susceptibility in the irreducible BZ with a trailing bosonic frequency axis,
            shape ``[q, w]``.
        :return: The starting (lower-bound) value of :math:`\lambda`.
        """
        w0 = chi_r.shape[-1] // 2
        return -np.min(1.0 / chi_r[..., w0].real)

    @staticmethod
    def apply_lambda(chi_r: np.ndarray, lambda_: float) -> np.ndarray:
        r"""
        Applies the :math:`\lambda`-correction :math:`\chi_r \to (1/\chi_r + \lambda)^{-1}` to the susceptibility.

        :param chi_r: Physical susceptibility array.
        :param lambda_: The correction shift :math:`\lambda`.
        :return: The corrected susceptibility, same shape as ``chi_r``.
        """
        return 1.0 / (1.0 / chi_r + lambda_)

    @staticmethod
    def find_lambda(
        chi_r_mat: np.ndarray,
        chi_r_loc_sum: complex,
        delta: float = 0.1,
        eps: float = 1e-7,
        maxiter: int = 1000,
    ) -> float:
        r"""
        Finds :math:`\lambda` such that the momentum-summed corrected susceptibility matches the local sum
        ``chi_r_loc_sum`` via a Newton-like iteration. The momentum sum is evaluated over the irreducible BZ using
        the per-point multiplicities. When a Newton step would lower :math:`\lambda` below the current value the
        step size ``delta`` is halved and the search is reset just above the divergence bound.

        :param chi_r_mat: Physical susceptibility in the irreducible BZ, shape ``[q, w]``.
        :param chi_r_loc_sum: Target value: the local susceptibility sum (already divided by :math:`\beta`).
        :param delta: Initial offset above the divergence bound and step size for the bisection-style reset.
        :param eps: Convergence tolerance on the real part of the sum-rule residual.
        :param maxiter: Maximum number of iterations before giving up (logs a warning and returns the last value).
        :return: The converged :math:`\lambda` (or the last value reached if not converged).
        """
        lambda_start = LambdaCorrection.get_lambda_start(chi_r_mat)
        lambda_: float = lambda_start + delta
        factor = 1 / config.sys.beta / config.lattice.q_grid.nk_tot

        for _ in range(maxiter):
            chi_lam = LambdaCorrection.apply_lambda(chi_r_mat, lambda_)
            chir_sum = (config.lattice.q_grid.irrk_count[:, None] * chi_lam).sum() * factor
            f_lam = chir_sum - chi_r_loc_sum
            fp_lam = -(config.lattice.q_grid.irrk_count[:, None] * chi_lam**2).sum() * factor
            # NB: a vanishing fp_lam is intentional here - the resulting (inf) Newton step is what triggers the
            # delta-halving reset branch below, so it must NOT be guarded against.
            lambda_new = lambda_ - (f_lam / fp_lam).real

            if abs(f_lam.real) < eps:
                return lambda_new

            if lambda_new < lambda_:
                delta /= 2
                lambda_ = lambda_start + delta
            else:
                lambda_ = lambda_new

        config.logger.warning(f"Lambda correction did not converge within {maxiter} iterations.")
        return lambda_

    @staticmethod
    def perform_single(chi_r: FourPoint, chi_r_loc_sum: complex) -> tuple[FourPoint, float]:
        r"""
        Performs the :math:`\lambda`-correction on the physical susceptibility for a single spin channel. Only
        works for single-band systems, since a multi-orbital correction would be a non-unique multidimensional
        problem.

        :param chi_r: Physical (single-band) susceptibility in the irreducible BZ.
        :param chi_r_loc_sum: Target local susceptibility sum (already divided by :math:`\beta`).
        :return: A tuple of (i) the corrected susceptibility in the irreducible BZ and half bosonic frequency
            range, and (ii) the determined :math:`\lambda`.
        """
        chi_r = chi_r.to_full_niw_range()
        chi_r_mat = chi_r.compress_q_dimension().mat.squeeze()
        lambda_r = LambdaCorrection.find_lambda(chi_r_mat, chi_r_loc_sum)
        chi_r.mat = LambdaCorrection.apply_lambda(chi_r_mat, lambda_r)[:, None, None, None, None, :]
        return chi_r.to_half_niw_range(), lambda_r

    @staticmethod
    def perform(chi_phys_q_r: FourPoint, quiet: bool = False) -> FourPoint:
        r"""
        Performs the :math:`\lambda`-correction on the physical susceptibility. If 'spch' is specified, the lambda
        correction is performed on both the density and magnetic channel, whereas only the magnetic channel is
        corrected if 'sp' is specified as :math:`\lambda`-correction type in the config. The local susceptibility
        sum-rule target is read from the saved local susceptibilities, and the determined :math:`\lambda` is
        appended to a text file.

        :param chi_phys_q_r: The momentum-dependent physical susceptibility :math:`\chi^{q}_{r}` to correct.
        :param quiet: If ``True``, the determined :math:`\lambda` is not appended to the lambda text file (used by
            the stabilizer's Jacobian probes). Note the 'sp' type reads the *saved* density susceptibility from
            file, so quiet probes evaluate it with the last really-saved one - the lambda background is frozen
            during a build.
        :return: The :math:`\lambda`-corrected physical susceptibility (unchanged for 'sp' in non-magnetic
            channels).
        :raises ValueError: If the configured lambda-correction type is neither 'spch' nor 'sp'.
        """
        logger = config.logger

        if config.lambda_correction.type.lower() not in ["spch", "sp"]:
            raise ValueError("Lambda correction type must be either 'spch' or 'sp'.")

        logger.info(f"Lambda correction type set to '{config.lambda_correction.type}'.")

        if config.lambda_correction.type.lower() == "spch":
            logger.info(f"Performing lambda correction for {chi_phys_q_r.channel.value} channel.")
            chi_r_loc = LocalFourPoint.load(
                os.path.join(config.output.output_path, f"chi_{chi_phys_q_r.channel.value}_loc.npy"),
                chi_phys_q_r.channel,
                num_vn_dimensions=0,
            ).to_full_niw_range()
            chi_phys_q_r, lambda_r = LambdaCorrection.perform_single(
                chi_phys_q_r, chi_r_loc.mat.sum() / config.sys.beta
            )
            chi_r_loc.free()
            logger.info(
                f"Lambda correction for the {chi_phys_q_r.channel.value} channel applied with lambda = "
                f"{lambda_r:.6f}."
            )

            if not quiet:
                with open(
                    os.path.join(config.output.output_path, f"lambda_{config.lambda_correction.type}.txt"), "a"
                ) as f:
                    f.write(f"lambda_{chi_phys_q_r.channel.value}: {lambda_r}\n")

            return chi_phys_q_r

        # else: "sp"
        if chi_phys_q_r.channel != SpinChannel.MAGN:
            return chi_phys_q_r

        logger.info(f"Performing lambda correction for magn channel.")
        chi_phys_q_dens = FourPoint.load(
            os.path.join(config.output.output_path, f"chi_phys_q_dens.npy"),
            SpinChannel.DENS,
            num_vn_dimensions=0,
        ).to_full_niw_range()

        chi_dens_loc, chi_magn_loc = [
            LocalFourPoint.load(
                os.path.join(config.output.output_path, f"chi_{channel.value}_loc.npy"),
                channel,
                num_vn_dimensions=0,
            ).to_full_niw_range()
            for channel in [SpinChannel.DENS, SpinChannel.MAGN]
        ]

        chi_magn_loc_sum = (chi_dens_loc.mat + chi_magn_loc.mat).sum() - 1 / config.lattice.q_grid.nk_tot * (
            config.lattice.q_grid.irrk_count[:, None, None, None, None, None] * chi_phys_q_dens.mat
        ).sum()
        chi_phys_q_r, lambda_r = LambdaCorrection.perform_single(chi_phys_q_r, chi_magn_loc_sum / config.sys.beta)
        logger.info(f"Lambda correction 'sp' applied. Lambda for magn channel is: {lambda_r:.6f}.")

        if not quiet:
            with open(os.path.join(config.output.output_path, f"lambda_{config.lambda_correction.type}.txt"), "a") as f:
                f.write(f"lambda_{chi_phys_q_r.channel.value}: {lambda_r}\n")

        return chi_phys_q_r


class MultiOrbitalLambdaCorrection:
    r"""
    Matrix-valued Moriya :math:`\lambda`-correction for multi-orbital ladder D\ :math:`\Gamma`\ A.
    The single-band :class:`LambdaCorrection` shifts the scalar inverse
    susceptibility by one number; here the mass is a full :math:`N_o^2\times N_o^2` matrix
    :math:`\Lambda_r` added to the compound inverse susceptibility
    :math:`(\chi^{q\omega}_{r})^{-1} \to (\chi^{q\omega}_{r})^{-1} + \Lambda_r`, calibrated so the momentum- and
    frequency-summed corrected susceptibility matches the local (AIM) sum rule component by component -
    :math:`N_o^2(N_o^2+1)/2` real conditions for the same number of real parameters (a well-posed matrix root
    problem, not a rugged optimization). With ``spch`` both channels are corrected with independent
    :math:`\Lambda_d, \Lambda_m`.

    The calibration is done in complex128 (the near-singular static inversions that dominate the sum rule are
    least trustworthy in the object's native complex64). Two deliberate simplifications relative to the design
    note: the mass is enforced real-symmetric (P3) but is *not* explicitly projected onto the point-group-invariant
    subspace - that covariance is automatic, since the residual and Jacobian are built from the symmetric full-BZ
    sum; and the converged :math:`\Lambda_r` (a single point-group-invariant matrix) is applied to the irreducible-BZ
    representatives, so the downstream :meth:`~dgamore.n_point_base.IAmNonLocal.map_to_full_bz` reconstructs the
    corrected full-BZ susceptibility consistently.

    Crucially the momentum sum is evaluated over the FULL Brillouin zone, not by weighting the irreducible-BZ value
    with its multiplicity: symmetry-related momenta carry orbitally-rotated susceptibility matrices
    :math:`D\chi D^\dagger`, so the off-diagonal orbital components of the sum only come out right after the star is
    summed explicitly. Only the irreducible wedge is kept resident, though: the cached compound susceptibility lives
    on the wedge and every sum-rule evaluation expands one bosonic-frequency slice at a time to the full BZ with
    exactly the transformation of ``map_to_full_bz`` (:meth:`_expand_compound_slice_to_full_bz`, dtype-preserving on
    the raw array). The corrected susceptibility is evaluated in resolvent form,
    :math:`(\chi^{-1} + \Lambda)^{-1} = (\mathbb{1} + \chi\Lambda)^{-1}\chi`, so :math:`\chi^{-1}` is never formed:
    the high-frequency ladder tails where :math:`\chi` crosses zero (and :math:`1/\chi` blows up) enter the sum rule
    harmlessly, and the only remaining susceptibility inversion is the single static slice the feasibility/gap
    machinery diagonalizes. For a single band (:math:`N_o=1`) the matrix collapses to a scalar and this reduces
    exactly to :class:`LambdaCorrection`.
    """

    _DELTA = 0.1  # initial offset of Lambda_0 above the static-feasibility bound
    _EPS = 1e-6  # convergence tol on the Frobenius norm of the sum-rule residual; at the complex64 input floor
    # (~1e-7), a tighter target would not converge on single-precision susceptibilities and only spuriously warn
    _MAXITER = 100  # maximum number of Newton iterations before giving up
    _MAX_LINE_SEARCH = 40  # maximum step halvings in the damped line search
    _GAP_TOL = 1e-6  # static-gap floor min_q lambda_min(Herm(chi^-1 + Lambda)); set above the complex64 input
    # noise (~1e-7): a smaller gap is unresolvable given the single-precision susceptibility feeding the calibration

    @staticmethod
    def get_lambda_start_matrix(chi_inv_static: np.ndarray) -> float:
        r"""
        Returns the scalar feasibility bound for :math:`\Lambda`: the smallest shift
        :math:`-\min_{q}\lambda_{\min}(\mathrm{Herm}((\chi^{q,\omega=0}_r)^{-1}))` at which
        :math:`\mathrm{Herm}((\chi^{q,\omega=0}_r)^{-1}) + \lambda\mathbb{1}` first becomes positive definite. The
        matrix generalization of :meth:`LambdaCorrection.get_lambda_start`.

        Only the STATIC (:math:`\omega=0`) blocks are passed: the domain constraint binds at :math:`\omega=0`
        (:math:`\mathrm{Herm}(\chi^{-1})` grows like :math:`\omega^2`), while the high-frequency ladder tails, where
        :math:`\chi` is small and its inverse blows up, are not genuine poles of the collective mode and must not set
        the bound.

        :param chi_inv_static: Inverse compound susceptibility at :math:`\omega=0`, shape ``[Nq, No^2, No^2]``.
        :return: The lower-bound shift (Newton starts just above it).
        """
        herm = 0.5 * (chi_inv_static + np.conj(np.swapaxes(chi_inv_static, -1, -2)))
        return float(-np.linalg.eigvalsh(herm).min())

    @staticmethod
    def _static_gap(chi_inv_static: np.ndarray, lambda_mat: np.ndarray) -> float:
        r"""
        Returns :math:`\min_{q}\lambda_{\min}(\mathrm{Herm}((\chi^{q,\omega=0}_r)^{-1} + \Lambda))` over the static
        blocks - strictly positive iff :math:`\Lambda` lies in the domain where the corrected susceptibility is well
        defined at the binding (:math:`\omega=0`) frequency.

        :param chi_inv_static: Inverse compound susceptibility at :math:`\omega=0`, shape ``[Nq, No^2, No^2]``.
        :param lambda_mat: The mass matrix :math:`\Lambda`, shape ``[No^2, No^2]``.
        :return: The smallest eigenvalue of the shifted Hermitian static inverse over all momenta.
        """
        shifted = chi_inv_static + lambda_mat
        herm = 0.5 * (shifted + np.conj(np.swapaxes(shifted, -1, -2)))
        return float(np.linalg.eigvalsh(herm).min())

    @staticmethod
    def _expand_compound_slice_to_full_bz(slice_irr: np.ndarray, q_grid) -> np.ndarray:
        r"""
        Expands one compound bosonic-frequency slice from the irreducible wedge to the full Brillouin zone on the
        raw array, applying exactly the transformation of :meth:`~dgamore.n_point_base.IAmNonLocal.map_to_full_bz`:
        the ``irrk_inv`` gather plus, in auto-symmetry mode, the per-k orbital rotation
        (:func:`~dgamore.symmetry_reduction.apply_auto_orbital_transform`). Works dtype-preserving on the raw array
        because the object route would coerce the calibration's complex128 back to the complex64 storage dtype.
        Both the gather and the (anti)unitary rotation commute with the compound inversion, so the slice may equally
        be the susceptibility or its cached inverse.

        :param slice_irr: Compound slice over the irreducible BZ, shape ``[Nq_irr, No^2, No^2]``.
        :param q_grid: The momentum grid carrying ``irrk_inv`` and the optional auto-mode orbital rotations.
        :return: The compound slice over the full BZ, shape ``[Nq, No^2, No^2]``, same dtype as the input.
        """
        full = np.take(slice_irr, q_grid.irrk_inv.ravel(), axis=0)
        if q_grid.is_auto:
            from dgamore import symmetry_reduction

            n_bands = int(round(np.sqrt(slice_irr.shape[-1])))
            nk_tot = full.shape[0]
            # compound rows pair (o1 o2), columns (o4 o3); the rotation acts on the [o1, o2, o3, o4] layout
            four = full.reshape(nk_tot, n_bands, n_bands, n_bands, n_bands).transpose(0, 1, 2, 4, 3)
            four = symmetry_reduction.apply_auto_orbital_transform(
                np.ascontiguousarray(four),
                us=q_grid._auto_us.reshape(nk_tot, *q_grid._auto_us.shape[3:]),
                sigmas=q_grid._auto_sigmas.reshape(-1),
                conjs=q_grid._auto_conjs.reshape(-1),
                num_orbital_dimensions=4,
            )
            full = np.ascontiguousarray(four.transpose(0, 1, 2, 4, 3)).reshape(nk_tot, *slice_irr.shape[1:])
        return full

    @staticmethod
    def _symmetric_basis(no2: int) -> np.ndarray:
        r"""
        Returns an orthonormal (Frobenius) basis of the real-symmetric :math:`N_o^2\times N_o^2` matrices, flattened
        to columns, shape ``[No^2 * No^2, No^2(No^2+1)/2]``. Newton runs in the span of this basis, so the solve is
        restricted to the symmetric (and thereby well-posed) subspace.
        """
        mats = []
        for i in range(no2):
            e = np.zeros((no2, no2))
            e[i, i] = 1.0
            mats.append(e.reshape(-1))
        for i in range(no2):
            for j in range(i + 1, no2):
                e = np.zeros((no2, no2))
                e[i, j] = e[j, i] = 1.0 / np.sqrt(2.0)
                mats.append(e.reshape(-1))
        return np.stack(mats, axis=1)

    @staticmethod
    def _residual(chi_qw: np.ndarray, lambda_mat: np.ndarray, s_r: np.ndarray, beta: float, nk_tot: int, q_grid=None):
        r"""
        Returns the real-symmetric sum-rule residual :math:`G(\Lambda) = (\beta N_q)^{-1}\sum_{q\omega}
        (\chi^{q\omega}_r{}^{-1} + \Lambda)^{-1} - S_r` (Hermitian part, real), with the corrected susceptibility
        evaluated in resolvent form :math:`(\mathbb{1} + \chi^{q\omega}_r\Lambda)^{-1}\chi^{q\omega}_r` so
        :math:`\chi^{-1}` is never formed. Bosonic frequencies are looped so the transients never exceed a single
        frequency slice; when ``q_grid`` is given the cached susceptibility lives on the irreducible wedge and each
        slice is expanded to the full BZ on the fly.

        :param chi_qw: Compound susceptibility, shape ``[Nq, Nw, No^2, No^2]`` over the full BZ, or over the
            irreducible wedge when ``q_grid`` is given.
        :param lambda_mat: Current mass matrix :math:`\Lambda`, shape ``[No^2, No^2]``.
        :param s_r: Local (AIM) sum-rule target :math:`S_r`, shape ``[No^2, No^2]``.
        :param beta: Inverse temperature :math:`\beta`.
        :param nk_tot: Number of full-BZ momenta :math:`N_q`.
        :param q_grid: The momentum grid for the per-slice full-BZ expansion (``None`` for a full-BZ input).
        :return: The residual matrix ``G`` (real, symmetric), shape ``[No^2, No^2]``.
        """
        no2 = lambda_mat.shape[0]
        identity = np.eye(no2)
        g_sum = np.zeros((no2, no2), dtype=np.complex128)
        for w in range(chi_qw.shape[1]):
            chi_w = chi_qw[:, w]
            if q_grid is not None:
                chi_w = MultiOrbitalLambdaCorrection._expand_compound_slice_to_full_bz(chi_w, q_grid)
            g_sum += (np.linalg.inv(identity + chi_w @ lambda_mat) @ chi_w).sum(axis=0)
        g = g_sum / (beta * nk_tot) - s_r
        return (0.5 * (g + np.conj(g.T))).real

    @staticmethod
    def residual_and_jacobian(
        chi_qw: np.ndarray, lambda_mat: np.ndarray, s_r: np.ndarray, beta: float, nk_tot: int, q_grid=None
    ):
        r"""
        Returns the sum-rule residual and its closed-form Newton Jacobian. With the compound
        :math:`N_o^2\times N_o^2` object :math:`\chi^{\Lambda,q\omega} = (\chi^{q\omega}{}^{-1} + \Lambda)^{-1}`,
        evaluated in resolvent form :math:`(\mathbb{1} + \chi^{q\omega}\Lambda)^{-1}\chi^{q\omega}` (so
        :math:`\chi^{-1}` is never formed), and
        :math:`\delta\chi^{\Lambda,q\omega} = -\chi^{\Lambda,q\omega}\,\delta\Lambda\,\chi^{\Lambda,q\omega}`,

        .. math::
            (J)_{1234} = \frac{\partial G_{12}}{\partial \Lambda_{34}}
                       = -\frac{1}{\beta N_q}\sum_{q\omega} \chi^{\Lambda,q\omega}_{13}\, \chi^{\Lambda,q\omega}_{42}.

        The bosonic-frequency loop bounds the transients to a single frequency slice; when ``q_grid`` is given the
        cached susceptibility lives on the irreducible wedge and each slice is expanded to the full BZ on the fly.

        :param chi_qw: Compound susceptibility, shape ``[Nq, Nw, No^2, No^2]`` over the full BZ, or over the
            irreducible wedge when ``q_grid`` is given.
        :param lambda_mat: Current mass matrix :math:`\Lambda`, shape ``[No^2, No^2]``.
        :param s_r: Local (AIM) sum-rule target :math:`S_r`, shape ``[No^2, No^2]``.
        :param beta: Inverse temperature :math:`\beta`.
        :param nk_tot: Number of full-BZ momenta :math:`N_q`.
        :param q_grid: The momentum grid for the per-slice full-BZ expansion (``None`` for a full-BZ input).
        :return: A tuple ``(G, J)`` of the real-symmetric residual (``[No^2, No^2]``) and the real Jacobian tensor
            (``[No^2, No^2, No^2, No^2]``).
        """
        no2 = lambda_mat.shape[0]
        identity = np.eye(no2)
        factor = 1.0 / (beta * nk_tot)
        g_sum = np.zeros((no2, no2), dtype=np.complex128)
        jac = np.zeros((no2, no2, no2, no2), dtype=np.float64)
        for w in range(chi_qw.shape[1]):
            chi_w = chi_qw[:, w]
            if q_grid is not None:
                chi_w = MultiOrbitalLambdaCorrection._expand_compound_slice_to_full_bz(chi_w, q_grid)
            chi_lam = np.linalg.inv(identity + chi_w @ lambda_mat) @ chi_w
            g_sum += chi_lam.sum(axis=0)
            jac += np.einsum("qac,qdb->abcd", chi_lam, chi_lam).real
        g = g_sum * factor - s_r
        g = (0.5 * (g + np.conj(g.T))).real
        jac *= -factor
        return g, jac

    @staticmethod
    def find_lambda_matrix(
        chi_qw: np.ndarray,
        s_r: np.ndarray,
        beta: float,
        nk_tot: int,
        delta: float = None,
        eps: float = None,
        maxiter: int = None,
        q_grid=None,
    ) -> np.ndarray:
        r"""
        Solves the matrix sum rule :math:`G(\Lambda_r) = 0` for the real-symmetric mass :math:`\Lambda_r` by a damped
        Newton iteration in the symmetric subspace. The Newton system :math:`H\,\delta\lambda = -g` is the Jacobian
        and residual projected onto the orthonormal symmetric basis (:meth:`_symmetric_basis`); each step is line-
        searched to keep the static gap positive (feasibility) and to decrease the residual. Starts from
        :math:`\Lambda_0 = (\lambda_{\mathrm{start}} + \delta)\mathbb{1}` at the feasibility bound. All sum-rule
        evaluations use the resolvent form :math:`(\mathbb{1} + \chi\Lambda)^{-1}\chi`, so the only inversion of the
        susceptibility itself is the single static slice the feasibility/gap machinery needs. When ``q_grid`` is
        given, ``chi_qw`` holds only the irreducible wedge and every sum-rule evaluation expands the slices to the
        full BZ on the fly (:meth:`_expand_compound_slice_to_full_bz`) - same result, ``Nq/Nq_irr`` times less
        resident memory.

        :param chi_qw: Compound susceptibility, shape ``[Nq, Nw, No^2, No^2]`` (complex128) over the full BZ, or
            over the irreducible wedge when ``q_grid`` is given.
        :param s_r: Local (AIM) sum-rule target :math:`S_r`, shape ``[No^2, No^2]``.
        :param beta: Inverse temperature :math:`\beta`.
        :param nk_tot: Number of full-BZ momenta :math:`N_q`.
        :param delta: Offset of the initial guess above the feasibility bound (default :data:`_DELTA`).
        :param eps: Convergence tolerance on the Frobenius norm of the residual (default :data:`_EPS`).
        :param maxiter: Maximum number of Newton iterations (default :data:`_MAXITER`).
        :param q_grid: The momentum grid for the per-slice full-BZ expansion (``None`` for a full-BZ input).
        :return: The converged mass matrix :math:`\Lambda_r`, shape ``[No^2, No^2]``, real symmetric.
        """
        delta = MultiOrbitalLambdaCorrection._DELTA if delta is None else delta
        eps = MultiOrbitalLambdaCorrection._EPS if eps is None else eps
        maxiter = MultiOrbitalLambdaCorrection._MAXITER if maxiter is None else maxiter

        no2 = s_r.shape[0]
        basis = MultiOrbitalLambdaCorrection._symmetric_basis(no2)
        chi_static = chi_qw[:, chi_qw.shape[1] // 2]  # omega=0 slice, the binding frequency
        if q_grid is not None:
            # the static feasibility/gap machinery keeps seeing the full BZ (one small slice, resident once)
            chi_static = MultiOrbitalLambdaCorrection._expand_compound_slice_to_full_bz(chi_static, q_grid)
        chi_inv_static = np.linalg.inv(chi_static)
        lambda_mat = (MultiOrbitalLambdaCorrection.get_lambda_start_matrix(chi_inv_static) + delta) * np.eye(no2)

        for _ in range(maxiter):
            g, jac = MultiOrbitalLambdaCorrection.residual_and_jacobian(chi_qw, lambda_mat, s_r, beta, nk_tot, q_grid)
            res = float(np.linalg.norm(g))
            if res < eps:
                return lambda_mat

            hess = basis.T @ jac.reshape(no2 * no2, no2 * no2) @ basis
            grad = basis.T @ g.reshape(-1)
            try:
                step = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, -grad, rcond=None)[0]
            d_lambda = (basis @ step).reshape(no2, no2)

            step_length = 1.0
            for _ in range(MultiOrbitalLambdaCorrection._MAX_LINE_SEARCH):
                candidate = lambda_mat + step_length * d_lambda
                gap = MultiOrbitalLambdaCorrection._static_gap(chi_inv_static, candidate)
                if (
                    gap > MultiOrbitalLambdaCorrection._GAP_TOL
                    and float(
                        np.linalg.norm(
                            MultiOrbitalLambdaCorrection._residual(chi_qw, candidate, s_r, beta, nk_tot, q_grid)
                        )
                    )
                    < res
                ):
                    break
                step_length *= 0.5
            else:
                break  # no admissible step (feasible and residual-decreasing) was found
            lambda_mat = lambda_mat + step_length * d_lambda

        config.logger.warning(f"Multi-orbital lambda correction did not converge within {maxiter} iterations.")
        return lambda_mat

    @staticmethod
    def apply_lambda_matrix(chi_compound: np.ndarray, lambda_mat: np.ndarray) -> np.ndarray:
        r"""
        Applies the matrix correction :math:`\chi_r \to (\chi_r^{-1} + \Lambda_r)^{-1}` to a stack of compound
        susceptibility blocks. The mass is broadcast over all leading (momentum/frequency) axes.

        :param chi_compound: Compound susceptibility blocks, shape ``[..., No^2, No^2]`` (complex).
        :param lambda_mat: The mass matrix :math:`\Lambda_r`, shape ``[No^2, No^2]``.
        :return: The corrected compound susceptibility, same shape as ``chi_compound``.
        """
        # resolvent form (1 + chi Lambda)^-1 chi: chi^-1 is never formed, so zero crossings of chi are harmless
        identity = np.eye(chi_compound.shape[-1])
        return np.linalg.inv(identity + chi_compound @ lambda_mat) @ chi_compound

    @staticmethod
    def perform_single(chi_r: FourPoint, s_r: np.ndarray) -> tuple[FourPoint, np.ndarray]:
        r"""
        Performs the matrix :math:`\lambda`-correction on the physical susceptibility for a single spin channel. The
        sum rule is evaluated over the FULL Brillouin zone (per-slice expansion carrying the orbital rotations of
        symmetry-related momenta, see :meth:`find_lambda_matrix`) from an irreducible-wedge-resident complex128
        cache; the converged (point-group-invariant) mass is then applied to the irreducible-BZ representatives,
        consistent with the downstream full-BZ mapping.

        :param chi_r: Physical susceptibility :math:`\chi^{q}_{r}` in the irreducible BZ (no fermionic frequencies,
            compressed momentum dimension). Consumed in place.
        :param s_r: Local (AIM) sum-rule target :math:`S_r = \beta^{-1}\sum_\omega \chi_{r,\mathrm{loc}}(i\omega)`,
            a compound :math:`N_o^2\times N_o^2` matrix.
        :return: A tuple of (i) the corrected susceptibility in the irreducible BZ and half bosonic frequency range,
            and (ii) the determined mass matrix :math:`\Lambda_r`.
        """
        beta = config.sys.beta
        q_grid = config.lattice.q_grid
        chi_r = chi_r.to_full_niw_range()

        # The complex128 susceptibility is cached on the irreducible wedge only (never inverted - the calibration
        # runs in resolvent form); the Newton expands the slices to the full BZ on the fly.
        chi_irr = chi_r.copy().to_compound_indices()
        chi_qw = chi_irr.mat.astype(np.complex128)
        chi_irr.free()

        lambda_r = MultiOrbitalLambdaCorrection.find_lambda_matrix(chi_qw, s_r, beta, q_grid.nk_tot, q_grid=q_grid)
        del chi_qw

        chi_r = chi_r.to_compound_indices()
        chi_r.mat = MultiOrbitalLambdaCorrection.apply_lambda_matrix(chi_r.mat.astype(np.complex128), lambda_r)
        return chi_r.to_full_indices().to_half_niw_range(), lambda_r

    @staticmethod
    def perform(chi_phys_q_r: FourPoint, quiet: bool = False) -> FourPoint:
        r"""
        Performs the multi-orbital ('spch') matrix :math:`\lambda`-correction on the physical susceptibility of the
        object's channel, calibrating against that channel's local susceptibility sum rule. The determined mass's
        Frobenius norm is appended to a text file (unless ``quiet``). Unlike the single-band
        :meth:`LambdaCorrection.perform` this works for any number of orbitals.

        :param chi_phys_q_r: The momentum-dependent physical susceptibility :math:`\chi^{q}_{r}` to correct.
        :param quiet: If ``True``, the determined mass is not appended to the lambda text file (used by the
            stabilizer's Jacobian probes, which must not pollute the run directory).
        :return: The :math:`\lambda`-corrected physical susceptibility.
        """
        logger = config.logger
        channel = chi_phys_q_r.channel
        logger.info(f"Performing multi-orbital lambda correction for {channel.value} channel.")

        chi_r_loc = LocalFourPoint.load(
            os.path.join(config.output.output_path, f"chi_{channel.value}_loc.npy"),
            channel,
            num_vn_dimensions=0,
        ).to_full_niw_range()
        s_r = chi_r_loc.to_compound_indices().mat.astype(np.complex128).sum(axis=0) / config.sys.beta
        chi_r_loc.free()

        chi_phys_q_r, lambda_r = MultiOrbitalLambdaCorrection.perform_single(chi_phys_q_r, s_r)
        lambda_norm = float(np.linalg.norm(lambda_r))
        logger.info(
            f"Multi-orbital lambda correction for the {channel.value} channel applied "
            f"(||Lambda|| = {lambda_norm:.6f})."
        )

        if not quiet:
            with open(os.path.join(config.output.output_path, "lambda_trial.txt"), "a") as f:
                f.write(f"lambda_{channel.value}_fro: {lambda_norm}\n")
            if channel == SpinChannel.MAGN:
                MultiOrbitalLambdaCorrection._log_pauli_diagnostic(chi_phys_q_r)

        return chi_phys_q_r

    @staticmethod
    def _density_diagonal_sum(chi_r: FourPoint) -> np.ndarray:
        r"""
        Returns the density-diagonal, momentum- and frequency-summed corrected susceptibility per orbital,
        :math:`C_{r;1} = (\beta N_q)^{-1} \sum_{q\omega} \chi^{q\omega}_{r;1111}` (the compound-diagonal entries
        :math:`(11)(11)`), evaluated over the FULL BZ. Operates on a copy, so ``chi_r`` is not mutated.

        :param chi_r: The corrected physical susceptibility :math:`\chi^{q}_{r}` in the irreducible BZ.
        :return: The real array :math:`C_{r;a}`, shape ``[n_bands]``.
        """
        n_bands = chi_r.n_bands
        q_grid = config.lattice.q_grid
        summed = chi_r.copy().to_full_niw_range().map_to_full_bz(q_grid).to_compound_indices().mat.sum(axis=(0, 1)) / (
            config.sys.beta * q_grid.nk_tot
        )
        return np.array([summed[a * n_bands + a, a * n_bands + a].real for a in range(n_bands)])

    @staticmethod
    def _log_pauli_diagnostic(chi_magn: FourPoint) -> None:
        r"""
        Logs the largest-magnitude diagonal deviation - over all inequivalent atoms and orbitals - of the corrected
        susceptibility's density-diagonal sum from the exact Pauli value,
        :math:`\tfrac12 (C_d + C_m)_{1} - \tfrac{n_1}{2}\big(1 - \tfrac{n_1}{2}\big)`, with :math:`n_1` the
        (spin-summed) DMFT orbital occupation (:data:`config.sys.occ_dmft_per_ineq`, per inequivalent atom - the
        assembled ``occ_dmft`` is only the last atom's). This is a lattice-vs-AIM density (chemical-potential)
        consistency cross-check - a drift flags a mu problem, not a solver failure (see the design note) - so it is
        a log only, never a constraint. Called once, on the magnetic channel, where the corrected density
        susceptibility has already been saved. Skipped when the occupation metadata or the saved density
        susceptibility is unavailable.

        :param chi_magn: The freshly corrected magnetic susceptibility :math:`\chi^{q}_{m}` (irreducible BZ).
        :return: None.
        """
        dens_path = os.path.join(config.output.output_path, "chi_phys_q_dens.npy")
        if not (config.sys.occ_dmft_per_ineq and config.dmft.ineq_ordering and os.path.exists(dens_path)):
            return

        chi_dens = FourPoint.load(
            dens_path, SpinChannel.DENS, num_vn_dimensions=0, has_compressed_q_dimension=True, nq=chi_magn.nq
        )
        c_d = MultiOrbitalLambdaCorrection._density_diagonal_sum(chi_dens)
        c_m = MultiOrbitalLambdaCorrection._density_diagonal_sum(chi_magn)

        worst_deviation, worst_ineq, worst_orbital = 0.0, None, None
        n_start = 0
        for ineq in config.dmft.ineq_ordering:
            occ = config.sys.occ_dmft_per_ineq[ineq - 1]
            for j in range(config.dmft.n_bands_per_ineq[ineq - 1]):
                a = n_start + j
                n_a = float(np.real(occ[j, j]))
                deviation = 0.5 * (c_d[a] + c_m[a]) - 0.5 * n_a * (1.0 - 0.5 * n_a)
                if worst_ineq is None or abs(deviation) > abs(worst_deviation):
                    worst_deviation, worst_ineq, worst_orbital = deviation, ineq, a
            n_start += config.dmft.n_bands_per_ineq[ineq - 1]

        if worst_ineq is not None:
            config.logger.info(
                f"Pauli check (largest diagonal deviation, ineq {worst_ineq}, orbital {worst_orbital}): "
                f"1/2(C_d+C_m) - n/2(1-n/2) = {worst_deviation:+.6e}"
            )


class LambdaAnnealer:
    r"""
    Lambda-annealing scaffold: a SINGLE shared bosonic mass :math:`\lambda` added to the inverse physical
    susceptibility of every channel to damp it and keep the Bethe-Salpeter pole at bay - the multi-orbital-safe
    counterpart of the single-band sum-rule :class:`LambdaCorrection` (the identity shift is basis independent
    and needs no calibration).

    One shared mass (rather than one per channel) is deliberate: the channels are coupled through the self-energy,
    so a large mass on one channel distorts :math:`\Sigma` and can drive another channel's gap negative; per-
    channel masses then chase each other to unphysically large values. A single mass sized from the worst channel
    protects all of them at once and cannot ratchet. It is a pure convergence scaffold - :meth:`update` anneals
    it to exactly zero between converged phases, so the final result is always pure self-consistency.

    This object owns the complete scaffold state and schedule, so the self-consistency loop never reaches into a
    global mailbox: it is created by :func:`dgamore.nonlocal_sde.calculate_self_energy_q`, threaded into the
    proposal/kernel where it measures each channel's gap and applies the shared mass (:meth:`apply`), and
    advanced once per iteration from the loop (:meth:`update`).

    The gap is measured on the STATIC (:math:`\omega=0`) compound blocks only: the physical positivity statement
    (and the pole danger) lives there, while the full-frequency spectrum of the inverse carries the large
    negative shell-truncation baseline that would inflate the mass by orders of magnitude. It is measured on
    every non-probe iteration (also when the mass is zero) so a pole opening mid-run re-arms the scaffold; the
    stabilizer's quiet Jacobian probes skip the measurement, since their perturbed states must not steer the
    schedule.

    The mass is raised toward its target with damping :data:`_BUMP_DAMPING` per iteration rather than jumped, so
    it changes on the self-energy's own relaxation timescale and each measured gap tracks a settled
    :math:`\Sigma`; it is clamped at :data:`_MAX_LAMBDA`, past which the pole is too deep for the scaffold and a
    warmer start (higher-T continuation) is the only remedy.
    """

    _START_FACTOR = 1.5  # bump target: this times the static pole violation
    _BUMP_DAMPING = 0.5  # fraction of the distance to the target the mass moves per bump iteration
    # Below this a halved mass snaps to exactly zero. Set at the shell-truncation scale of the static gap
    # (healthy runs sit near -1e-2), so smaller masses are noise-level and only stretch the annealing tail.
    _LAMBDA_FLOOR = 1e-2
    _MAX_LAMBDA = 1e3  # ceiling: past this the pole is too deep for the scaffold (warm-start instead)

    def __init__(self, channels: tuple = (SpinChannel.DENS, SpinChannel.MAGN)):
        """
        Initializes an inert scaffold: zero shared mass, no gap measured yet.

        :param channels: The spin channels the scaffold protects (density and magnetic by default).
        """
        self._gaps = {channel.value: None for channel in channels}
        self._mass = 0.0
        self._initialized = False
        self._capped = False

    @property
    def active(self) -> bool:
        """Whether the scaffold is currently shaping the map (still measuring on init, or a mass is present)."""
        return not self._initialized or self.mass_present

    @property
    def mass_present(self) -> bool:
        """Whether the shared boson mass is currently nonzero."""
        return self._mass > 0.0

    def apply(self, chi_phys_q_r: FourPoint, mpi_dist_irrq: MpiDistributor, measure: bool = True) -> FourPoint:
        r"""
        Measures the channel's static boson gap (unless ``measure`` is False, i.e. a quiet Jacobian probe) and,
        if the shared mass is nonzero, adds it to the compound diagonal of the inverse susceptibility at all
        bosonic frequencies.

        :param chi_phys_q_r: The physical susceptibility :math:`\chi^{q\omega}_{r;1234}` (no fermionic frequency
            dimensions).
        :param mpi_dist_irrq: The irreducible-BZ distributor (the measured gap is reduced over ranks).
        :param measure: If ``True``, (re-)measure and store the static gap; probes pass ``False``.
        :return: The (possibly mass-shifted) physical susceptibility as a :class:`FourPoint`.
        """
        channel = chi_phys_q_r.channel.value

        if measure:
            self._gaps[channel] = self._static_gap(chi_phys_q_r, mpi_dist_irrq)

        if self._mass == 0.0:
            return chi_phys_q_r

        chi_inv = chi_phys_q_r.invert(copy=False).to_compound_indices()
        idx = np.arange(chi_inv.mat.shape[-1])
        chi_inv.mat[..., idx, idx] += self._mass
        if measure:
            config.logger.info(
                f"Lambda annealing ({channel}): shared boson mass lambda={self._mass:.6f} applied "
                f"(static gap of 1/chi: {self._gaps[channel]:.6f})."
            )
        return chi_inv.invert(copy=False)

    @staticmethod
    def _static_gap(chi_phys_q_r: FourPoint, mpi_dist_irrq: MpiDistributor) -> float:
        r"""
        Returns the smallest eigenvalue of the Hermitian part of the static inverse compound susceptibility
        :math:`(\chi^{q(\omega=0)}_{r;1234})^{-1}` over all momenta, reduced with ``MPI.MIN`` across ranks.
        """
        w0 = chi_phys_q_r.niw if chi_phys_q_r.full_niw_range else 0
        n = chi_phys_q_r.n_bands**2
        static = chi_phys_q_r.mat[..., w0].transpose(0, 1, 2, 4, 3).reshape(-1, n, n)
        static_inv = np.linalg.inv(static.astype(np.complex64, copy=False))
        herm = 0.5 * (static_inv + np.conj(np.swapaxes(static_inv, -1, -2)))
        min_eig = float(np.linalg.eigvalsh(herm).min())
        if mpi_dist_irrq.comm.size > 1:
            min_eig = mpi_dist_irrq.comm.allreduce(min_eig, op=MPI.MIN)
        return min_eig

    def _bump(self, mass: float, worst_gap: float) -> float:
        r"""
        Raises the shared mass toward its target :math:`1.5\,|g|` (``g`` the worst static gap) by the damping
        fraction, clamped at :data:`_MAX_LAMBDA`. Warns once when the ceiling is hit (the pole is too deep for
        the scaffold).
        """
        target = self._START_FACTOR * (-worst_gap)
        new_mass = mass + self._BUMP_DAMPING * (target - mass)
        if new_mass >= self._MAX_LAMBDA:
            new_mass = self._MAX_LAMBDA
            if not self._capped:
                self._capped = True
                config.logger.warning(
                    f"Lambda annealing: the shared boson mass hit the ceiling ({self._MAX_LAMBDA:g}) but the "
                    f"static gap is still {worst_gap:.6f}. The pole is too deep for the scaffold to hold - "
                    f"continue from a warmer start (higher-T continuation); the result may be unphysical."
                )
        else:
            self._capped = False
        return new_mass

    def update(self, converged: bool) -> bool:
        r"""
        Advances the schedule once per iteration from the loop, after the convergence decision. The shared mass is
        sized from the worst (most negative) static gap across channels. Exactly one action applies: initialization
        (first measured iteration - a damped bump if any channel is poled, else stay inert at zero); a bump (the
        *shifted* worst gap :math:`g + \lambda` is still negative, so the mass is raised toward its target with
        damping; this takes precedence over a halving, and also re-arms a zeroed scaffold whose pole reopened); or
        a halving (phase converged with a healthy shifted gap; below :data:`_LAMBDA_FLOOR` the mass snaps to zero).
        Any change alters the iterated map, so the caller must reset the accelerated-mixing history.

        :param converged: Whether this iteration satisfied the (relaxed) convergence criterion.
        :return: Whether the shared mass changed.
        """
        gaps = [g for g in self._gaps.values() if g is not None]
        if not gaps:
            return False  # nothing measured yet (e.g. a rank with an empty q-slice never populated a gap)
        worst_gap = min(gaps)
        old_mass = self._mass

        if not self._initialized:
            self._initialized = True
            if worst_gap >= 0.0:
                config.logger.info("Lambda annealing: healthy boson gap - the scaffold stays inert.")
                return False
            new_mass, reason = self._bump(old_mass, worst_gap), "initialized"
        elif worst_gap + old_mass < 0.0:
            new_mass, reason = self._bump(old_mass, worst_gap), "bumped (shifted static gap still negative)"
        elif converged and old_mass > 0.0:
            new_mass = 0.5 * old_mass if 0.5 * old_mass >= self._LAMBDA_FLOOR else 0.0
            reason = "halved (phase converged)"
        else:
            return False  # steady state (healthy, or capped): no change to the map

        if new_mass == old_mass:
            return False  # e.g. the mass is pinned at the ceiling

        self._mass = new_mass
        if new_mass == 0.0:
            config.logger.info(
                "Lambda annealed to zero - continuing with pure self-consistency at full epsilon "
                "(only that result counts as converged)."
            )
        else:
            config.logger.info(
                f"Lambda annealing: shared boson mass {old_mass:.6f} -> {new_mass:.6f}, {reason} "
                f"(worst static gap of 1/chi: {worst_gap:.6f})."
            )
        return True

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Single-particle self-energy. :class:`SelfEnergy` wraps the (possibly momentum-dependent) self-energy
:math:`\Sigma_{ab}(k, \nu)` and provides its high-frequency moments (:math:`\Sigma_\infty` and the
:math:`1/\imath\nu` coefficient), tools to estimate the core frequency box, to append the analytic asymptotic
tail beyond the core, to polynomial-fit the tail, and to interpolate the self-energy between temperatures. The
moments are obtained by fitting the highest available Matsubara frequencies.
"""

import itertools as it
from copy import deepcopy

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d

from dgamore.matsubara_frequencies import MFHelper
from dgamore.two_point import TwoPoint


class SelfEnergy(TwoPoint):
    r"""
    The (possibly momentum-dependent) single-particle self-energy :math:`\Sigma_{ab}(k, \nu)`. On top of the
    two-point orbital bookkeeping inherited from :class:`LocalTwoPoint` it provides the high-frequency moments
    (:math:`\Sigma_\infty` and the :math:`1/\imath\nu` coefficient), an estimate of the core frequency box, the
    construction/append of the analytic asymptotic tail beyond the core, a polynomial tail fit, and the
    temperature/frequency interpolation between grids. The moments are obtained by fitting the highest available
    Matsubara frequencies.
    """

    def __init__(
        self,
        mat: np.ndarray,
        nk: tuple[int, int, int] = (1, 1, 1),
        full_niv_range: bool = True,
        has_compressed_q_dimension: bool = False,
        estimate_niv_core: bool = False,
        calc_smom: bool = True,
        beta: float = None,
    ):
        r"""
        Initializes the self-energy and, unless disabled, fits its high-frequency moments and estimates the core box.

        :param mat: Underlying self-energy array (two orbital axes and one fermionic frequency axis, optionally
            preceded by momentum axes).
        :param nk: Number of momenta per spatial direction.
        :param full_niv_range: Whether the object spans the full (signed) fermionic range or only :math:`\nu \geq 0`.
        :param has_compressed_q_dimension: Whether the momentum is stored as a single compressed axis (True) or as
            ``[kx, ky, kz, ...]`` (False).
        :param estimate_niv_core: If True, estimate the core frequency box from the deviation to the asymptotic tail;
            otherwise the core extends over the full stored box.
        :param calc_smom: If True, fit the high-frequency moments :math:`(\Sigma_\infty, \Sigma_1)` on construction.
        :param beta: Inverse temperature :math:`\beta` used for the Matsubara frequencies in the moment fit and the
            asymptotic tail.
        """
        TwoPoint.__init__(self, mat, nk, full_niv_range, has_compressed_q_dimension)
        # TODO: check if this is a reasonable value. I'd suggest it depends on the input data size.
        self._niv_core_min = 20
        self._beta = beta

        if calc_smom:
            self._smom0, self._smom1 = self.fit_smom()
        self._niv_core = self._estimate_niv_core() if estimate_niv_core else self.niv

    @property
    def smom(self) -> tuple[np.ndarray, np.ndarray]:
        r"""
        The fitted high-frequency moments of the self-energy.

        :return: The first two high-frequency moments :math:`(\Sigma_\infty, \Sigma_1)` of the self-energy, each of
            shape ``[o1, o2]``.
        """
        return self._smom0, self._smom1

    def fit_smom(self):
        r"""
        Fits the first two high-frequency moments of the (k-averaged) self-energy from the highest stored Matsubara
        frequencies: the constant Hartree shift :math:`\Sigma_\infty` (real part) and the :math:`1/\imath\nu`
        coefficient :math:`\Sigma_1` (from the imaginary part). At least four frequencies are used for the fit.

        :return: The tuple ``(mom0, mom1)`` of moments, each of shape ``[o1, o2]``.
        """
        compress = False
        if self.has_compressed_q_dimension:
            compress = True
            self.decompress_q_dimension()

        mat_half_v = np.mean(self.mat[..., self.niv :], axis=(0, 1, 2))
        iv = 1j * MFHelper.vn(self.niv, self._beta, return_only_positive=True)

        n_freq_fit = int(0.2 * self.niv)
        if n_freq_fit < 4:
            n_freq_fit = 4

        iwfit = iv[self.niv - n_freq_fit :][None, None, :]  # * np.eye(self.n_bands)[:, :, None]
        fitdata = mat_half_v[..., self.niv - n_freq_fit :]

        mom0 = np.mean(fitdata.real, axis=-1)
        mom1 = np.mean(fitdata.imag * iwfit.imag, axis=-1)

        if compress:
            self.compress_q_dimension()
        return mom0, mom1

    def create_with_asympt_up_to_core(self) -> "SelfEnergy":
        """
        Returns a copy whose data inside the estimated core box is kept and whose shell (up to the full stored box) is
        replaced by the analytic asymptotic tail. A no-op (returns a copy) if the core already spans the full box or
        the tail is empty.

        :return: A copy of the self-energy with the asymptotic tail outside the core region.
        """
        # ``_get_asympt`` reads only the moments/grid (not ``mat``), and ``cut_niv`` already returns an independent
        # copy, so source from ``self`` directly and drop the wasteful initial ``deepcopy`` of the full array.
        asympt = self._get_asympt(niv=self.niv)

        if self._niv_core == self.niv:
            return deepcopy(self)
        if asympt.niv == 0:
            return deepcopy(self)

        copy = self.cut_niv(self._niv_core)
        if copy is self:
            # ``cut_niv`` was a no-op (``_niv_core >= self.niv``); deep-copy so the operation stays non-destructive.
            copy = deepcopy(self)
        copy.mat = np.concatenate(
            (asympt.mat[..., : asympt.niv - copy.niv], copy.mat, asympt.mat[..., asympt.niv + copy.niv :]), axis=-1
        )
        return copy

    def append_asympt(self, niv: int):
        """
        Returns a copy with the analytic asymptotic tail appended on both ends so that the frequency box extends to
        ``niv``. A no-op (returns a copy) if ``niv <= self.niv``.

        :param niv: Target number of positive fermionic frequencies.
        :return: A copy of the self-energy extended to ``niv``.
        """
        asympt = self._get_asympt(niv)
        if niv <= self.niv:
            return deepcopy(self)
        # ``np.concatenate`` allocates the result and only reads ``self.mat``, so clone the metadata without
        # duplicating the array first.
        copy = self._clone_without_mat()
        copy.mat = np.concatenate(
            (asympt.mat[..., : asympt.niv - self.niv], self.mat, asympt.mat[..., asympt.niv + self.niv :]), axis=-1
        )
        return copy

    def to_full_niv_range(self):
        r"""
        Extends the object to the full (signed) fermionic frequency range in place, using
        :math:`\Sigma(-\nu) = \Sigma(\nu)^*`. A no-op if there is no fermionic axis or the object is already full.

        :return: ``self`` in the full fermionic frequency range.
        """
        if self.num_vn_dimensions == 0 or self.full_niv_range:
            return self

        self.mat = np.concatenate((np.conj(np.flip(self.mat, axis=-1)), self.mat), axis=-1)
        self.update_original_shape()
        self._full_niv_range = True
        return self

    def __add__(self, other):
        """
        Operator form of :meth:`add` (``A + B``). See :meth:`add`.
        """
        return self.add(other)

    def __sub__(self, other):
        """
        Operator form of :meth:`sub` (``A - B``). See :meth:`sub`.
        """
        return self.sub(other)

    def add(self, other) -> "SelfEnergy":
        """
        Adds another :class:`SelfEnergy` (momentum dimensions are aligned first) or a numpy array; see :meth:`_add`.

        :param other: A :class:`SelfEnergy` or numpy array.
        :return: A new :class:`SelfEnergy` holding the sum (moments not recomputed).
        """
        return self._add(other)

    def _add(self, other, subtract: bool = False) -> "SelfEnergy":
        """
        Adds another :class:`SelfEnergy` (momentum dimensions are aligned first) or a numpy array.

        :param other: A :class:`SelfEnergy` or numpy array.
        :param subtract: If True, subtract ``other`` instead of adding it (used by :meth:`sub` to avoid a negated copy).
        :return: A new :class:`SelfEnergy` holding the sum (moments not recomputed).
        :raises ValueError: If ``other`` is neither a :class:`SelfEnergy` nor a numpy array.
        """
        if not isinstance(other, (SelfEnergy, np.ndarray)):
            raise ValueError(f"Can not add {type(other)} to {type(self)}.")

        op = np.subtract if subtract else np.add

        if isinstance(other, np.ndarray):
            return SelfEnergy(
                op(self.mat, other),
                self.nq,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                False,
                beta=self._beta,
            )

        other = self._align_q_dimensions_for_operations(other)
        return SelfEnergy(
            op(self.mat, other.mat),
            self.nq,
            self.full_niv_range,
            self.has_compressed_q_dimension,
            False,
            beta=self._beta,
        )

    def sub(self, other) -> "SelfEnergy":
        """
        Subtracts another :class:`SelfEnergy` or numpy array, implemented as ``self._add(other, subtract=True)`` (see
        :meth:`_add`).

        :param other: A :class:`SelfEnergy` or numpy array.
        :return: A new :class:`SelfEnergy` holding the difference.
        """
        return self._add(other, subtract=True)

    def concatenate_self_energies(self, other: "SelfEnergy") -> "SelfEnergy":
        """
        Builds a self-energy that keeps ``self`` inside its core box and uses ``other`` for the shell, up to
        ``other.niv``.

        :param other: The self-energy supplying the shell frequencies; must have at least as many frequencies as ``self``.
        :return: A new :class:`SelfEnergy` spanning ``other``'s frequency box.
        :raises ValueError: If ``other`` has fewer frequencies than ``self``.
        """
        if self.niv > other.niv:
            raise ValueError("Can not concatenate with a self-energy that has less frequencies.")
        niv_diff = other.niv - self.niv

        self.compress_q_dimension()
        other = other.compress_q_dimension()

        other_mat = np.tile(other.mat, (self.nq_tot, 1, 1, 1)) if other.nq_tot == 1 else other.mat
        result_mat = np.concatenate(
            (other_mat[..., :niv_diff], self.mat, other_mat[..., niv_diff + 2 * self.niv :]), axis=-1
        )
        return SelfEnergy(
            result_mat, self.nq, self.full_niv_range, self.has_compressed_q_dimension, False, beta=self._beta
        )

    def fit_polynomial(self, n_fit: int = 4, degree: int = 3, niv_core: int = 0) -> "SelfEnergy":
        """
        Replaces the self-energy by a per-(momentum, orbital) polynomial fit of the positive-frequency data,
        evaluated on the full positive frequency grid, then mirrored to the full range. A no-op (returns a copy) if
        ``n_fit == 0``.

        :param n_fit: Number of positive frequencies used for the fit; if 0 the object is returned unchanged, and if
            larger than ``self.niv`` (or negative) it falls back to ``niv_core + 200``.
        :param degree: Degree of the fitted polynomial.
        :param niv_core: Core frequency count used for the fallback ``n_fit`` when ``n_fit`` is out of range.
        :return: A new :class:`SelfEnergy` holding the polynomial fit, in the full fermionic frequency range.
        """
        if n_fit == 0:
            return deepcopy(self)

        if n_fit > self.niv or n_fit < 0:
            n_fit = niv_core + 200

        # Share ``self.mat`` into the metadata clone instead of deep-copying it: ``compress_q_dimension`` is a view and
        # ``to_half_niv_range`` copies (breaking the share), and ``poly_mat``/``fit_mat`` below are independent, so
        # ``self.mat`` is never mutated.
        copy = self._clone_without_mat()
        copy.mat = self.mat
        copy = copy.compress_q_dimension().to_half_niv_range()
        vn_fit = MFHelper.vn(n_fit, return_only_positive=True)
        vn_full = MFHelper.vn(2 * copy.niv, return_only_positive=True)
        poly_mat = np.zeros_like(copy.mat)
        fit_mat = copy.cut_niv(n_fit).mat

        for k in range(copy.nq_tot):
            for o1 in range(copy.n_bands):
                for o2 in range(copy.n_bands):
                    poly = np.polyfit(vn_fit, fit_mat[k, o1, o2, ...], degree)
                    poly_mat[k, o1, o2, :] = np.polyval(poly, vn_full)

        return SelfEnergy(
            poly_mat, copy.nq, copy.full_niv_range, copy.has_compressed_q_dimension, False, beta=copy._beta
        ).to_full_niv_range()

    def interpolate(self, beta_target: float, niv_target: int, niv_linear: int = 4) -> "SelfEnergy":
        r"""
        Re-grid the self-energy from its own inverse temperature :math:`\beta` (``self._beta``) to ``beta_target``.
        Only the last (frequency) axis is interpolated.

        The innermost niv_linear source frequencies are interpolated linearly
        (no curvature assumption where the grid is sparsest); everything above is
        interpolated with shape-preserving (PCHIP) splines. Re/Im are handled
        separately on the full signed grid. All target frequencies are returned.

        :param beta_target: Inverse temperature :math:`\beta` of the target grid.
        :param niv_target: Number of positive fermionic frequencies of the target grid.
        :param niv_linear: Number of innermost positive source frequencies interpolated linearly (PCHIP above).
        :return: A new :class:`SelfEnergy` on the target frequency grid (half niv range flag set to full on output).
        """
        vn_in = MFHelper.vn(self.niv, float(self._beta))
        vn_out = MFHelper.vn(niv_target, float(beta_target))

        pchip_re = PchipInterpolator(vn_in, self.mat.real, axis=-1, extrapolate=True)
        pchip_im = PchipInterpolator(vn_in, self.mat.imag, axis=-1, extrapolate=True)
        lin_re = interp1d(
            vn_in,
            self.mat.real,
            kind="linear",
            axis=-1,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        lin_im = interp1d(
            vn_in,
            self.mat.imag,
            kind="linear",
            axis=-1,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )

        # linear for |nu| at or below the niv_linear-th positive source frequency, PCHIP above
        nu_cut = np.abs(vn_in[self.niv : self.niv + niv_linear]).max()
        use_lin = np.abs(vn_out) <= nu_cut

        re = np.where(use_lin, lin_re(vn_out), pchip_re(vn_out))
        im = np.where(use_lin, lin_im(vn_out), pchip_im(vn_out))
        sigma_out = re + 1j * im

        return SelfEnergy(sigma_out, self.nq, False, self.has_compressed_q_dimension, True, beta=beta_target)

    def _estimate_niv_core(self, err: float = 1e-5):
        """
        Estimates the core frequency box as the smallest frequency beyond which the k-averaged self-energy (real and
        imaginary parts, over all orbital pairs) lies within ``err`` of its analytic asymptotic tail. The result is
        clamped from below by ``self._niv_core_min``.

        :param err: Absolute tolerance for the deviation between the self-energy and its asymptotic tail.
        :return: The estimated number of positive core frequencies.
        """
        asympt = self._get_asympt(niv=self.niv, n_min=0)

        max_ind_real = 0
        max_ind_imag = 0

        for i, j in it.product(range(self.n_bands), repeat=2):
            k_mean = np.mean(self.mat[:, :, :, i, j, :], axis=(0, 1, 2))
            asympt_mean = np.mean(asympt.mat[:, :, :, i, j, :], axis=(0, 1, 2))
            ind_real = np.argmax(np.abs(k_mean.real - asympt_mean.real) < err)
            ind_imag = np.argmax(np.abs(k_mean.imag - asympt_mean.imag) < err)

            max_ind_real = max(max_ind_real, ind_real)
            max_ind_imag = max(max_ind_imag, ind_imag)

        niv_core = max(max_ind_real, max_ind_imag)
        if niv_core < self._niv_core_min:
            return self._niv_core_min
        return niv_core

    def _get_asympt(self, niv: int, n_min: int = None) -> "SelfEnergy":
        r"""
        Builds the analytic high-frequency tail :math:`\Sigma_\infty - \Sigma_1/(\imath\nu)` from the fitted moments
        over the requested frequency range. Intended to be padded onto the self-energy as a shell, not used on its own.

        :param niv: Number of positive fermionic frequencies in the returned tail.
        :param n_min: Frequency-index offset (shift) of the first returned frequency; defaults to ``self.niv``.
        :return: A :class:`SelfEnergy` containing only the asymptotic tail.
        """
        if n_min is None:
            n_min = self.niv
        iv_asympt = 1j * MFHelper.vn(niv, self._beta, shift=n_min)[None, None, ...]
        asympt = (self._smom0[..., None] - 1.0 / iv_asympt * self._smom1[..., None])[None, None, None, ...] * np.ones(
            self.nq
        )[..., None, None, None]
        return SelfEnergy(asympt, self.nq, beta=self._beta)

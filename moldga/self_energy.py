import itertools
import itertools as it
from copy import deepcopy

import numpy as np
from scipy.interpolate import interp1d

import moldga.config as config
from moldga.brillouin_zone import KGrid
from moldga.local_n_point import LocalNPoint
from moldga.matsubara_frequencies import MFHelper
from moldga.n_point_base import IAmNonLocal


class SelfEnergy(IAmNonLocal, LocalNPoint):
    """
    Represents the self-energy. Will automatically map to full niv range if full_niv_range is set to False. This class
    is a bit of a mess and should be rewritten.
    """

    def __init__(
        self,
        mat: np.ndarray,
        nk: tuple[int, int, int] = (1, 1, 1),
        full_niv_range: bool = True,
        has_compressed_q_dimension: bool = False,
        estimate_niv_core: bool = False,
    ):
        LocalNPoint.__init__(self, mat, 2, 0, 1, full_niv_range=full_niv_range)
        IAmNonLocal.__init__(self, mat, nk, has_compressed_q_dimension=has_compressed_q_dimension)
        # TODO: check if this is a reasonable value. I'd suggest it depends on the input data size.
        self._niv_core_min = 20

        if not full_niv_range:
            self.to_full_niv_range()

        self._smom0, self._smom1 = self.fit_smom()
        self._niv_core = self._estimate_niv_core() if estimate_niv_core else self.niv

    @property
    def smom(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the first two local momenta of the self-energy.
        """
        return self._smom0, self._smom1

    def fit_smom(self):
        """
        Fits the first two local momenta of the self-energy.
        """
        compress = False
        if self.has_compressed_q_dimension:
            compress = True
            self.decompress_q_dimension()

        mat_half_v = np.mean(self.mat[..., self.niv :], axis=(0, 1, 2))
        iv = 1j * MFHelper.vn(self.niv, config.sys.beta, return_only_positive=True)

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
        Concatenates the core and the asymptotic tail of the self-energy from the 'estimated' core region to the actual
        specified core region (in settings).
        """
        copy = deepcopy(self)
        asympt = copy._get_asympt(niv=copy.niv)

        if copy._niv_core == copy.niv:
            return copy
        if asympt.niv == 0:
            return copy

        copy = copy.cut_niv(copy._niv_core)
        copy.mat = np.concatenate(
            (asympt.mat[..., : asympt.niv - copy.niv], copy.mat, asympt.mat[..., asympt.niv + copy.niv :]), axis=-1
        )
        return copy

    def append_asympt(self, niv: int):
        """
        Adds the asymptotic tail to the self-energy up to niv.
        """
        copy = deepcopy(self)
        asympt = copy._get_asympt(niv)
        if niv <= copy.niv:
            return copy
        copy.mat = np.concatenate(
            (asympt.mat[..., : asympt.niv - copy.niv], copy.mat, asympt.mat[..., asympt.niv + copy.niv :]), axis=-1
        )
        return copy

    def to_full_niv_range(self):
        """
        Converts the object to the full fermionic frequency range in-place. Works only on objects
        with a single fermionic frequency dimension.
        """
        if self.num_vn_dimensions == 0 or self.full_niv_range:
            return self

        self.mat = np.concatenate((np.conj(np.flip(self.mat, axis=-1)), self.mat), axis=-1)
        self.update_original_shape()
        self._full_niv_range = True
        return self

    def to_half_niv_range(self):
        """
        Converts the object to the half fermionic frequency range in-place. Works only on objects
        with a single fermionic frequency dimension.
        """
        if self.num_vn_dimensions == 0 or not self.full_niv_range:
            return self

        ind = np.arange(self.current_shape[-1] // 2, self.current_shape[-1])
        self.mat = np.take(self.mat, ind, axis=-1)
        self.update_original_shape()
        self._full_niv_range = False
        return self

    def __add__(self, other):
        """
        Adds two SelfEnergy objects.
        """
        return self.add(other)

    def __sub__(self, other):
        """
        Subtracts two SelfEnergy objects.
        """
        return self.sub(other)

    def add(self, other) -> "SelfEnergy":
        """
        Adds two SelfEnergy objects.
        """
        if not isinstance(other, (SelfEnergy, np.ndarray)):
            raise ValueError(f"Can not add {type(other)} to {type(self)}.")

        if isinstance(other, np.ndarray):
            return SelfEnergy(self.mat + other, self.nq, self.full_niv_range, self.has_compressed_q_dimension, False)

        other = self._align_q_dimensions_for_operations(other)
        return SelfEnergy(self.mat + other.mat, self.nq, self.full_niv_range, self.has_compressed_q_dimension, False)

    def sub(self, other) -> "SelfEnergy":
        """
        Subtracts two SelfEnergy objects.
        """
        return self.add(-other)

    def concatenate_self_energies(self, other: "SelfEnergy") -> "SelfEnergy":
        """
        Concats the self-energy with the other self-energy up to other.niv.
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
        return SelfEnergy(result_mat, self.nq, self.full_niv_range, self.has_compressed_q_dimension, False)

    def fit_polynomial(self, n_fit: int = 4, degree: int = 3, niv_core: int = 0) -> "SelfEnergy":
        """
        Fits a polynomial of a given degree to the self-energy.
        """
        copy = deepcopy(self)

        if n_fit == 0:
            return copy

        if n_fit > copy.niv or n_fit < 0:
            n_fit = niv_core + 200

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

        return SelfEnergy(poly_mat, copy.nq, copy.full_niv_range, copy.has_compressed_q_dimension, False)

    def symmetrize_orbitals(self, orbitals: list | np.ndarray) -> "SelfEnergy":
        r"""
        Symmetrizes the LocalNPoint object with respect to the orbitals given in the list. The minimum value that
        should be entered inside "orbs_list" is 1. and the max is the number of bands. For example, if the object has
        3 bands and we want to symmetrize with respect to the first and third orbital, we can enter "orbitals=[1,3]".
        The symmetrization is done by permuting the orbitals in all possible ways and averaging over the results.
        """
        orbital_axes = self._get_orbital_axes()
        if self.is_orbitally_symmetrized(orbitals):
            return self
        return self._symmetrize_orbitals(orbitals, orbital_axes)

    def is_orbitally_symmetrized(self, orbitals: list | np.ndarray) -> bool:
        """
        Check whether the LocalFourPoint object is orbitally symmetrized with respect to the orbitals given.
        """
        orbital_axes = self._get_orbital_axes()
        return self._is_orbitally_symmetrized(orbitals, orbital_axes)

    def map_to_full_bz(self, k_grid: KGrid, nq: tuple = None):
        return self._map_to_full_bz(k_grid, 2, nq)

    def interpolate(self, beta_source: float, beta_target: float, niv_target: int) -> "SelfEnergy":
        """
        Linearly interpolate the self-energy from beta_source to beta_target,
        using positive Matsubara data and explicit v=0 anchoring.
        """
        vn_in = MFHelper.vn(self.niv, float(beta_source), return_only_positive=True)
        vn_out = MFHelper.vn(niv_target, float(beta_target), return_only_positive=True)

        fit_mat = self.mat[..., self.niv :]
        sigma_zero = 0.5 * (self.mat[..., self.niv - 1] + self.mat[..., self.niv])

        # Augment grid with v=0
        vn_aug = np.concatenate(([0.0], vn_in))
        sigma_aug = np.concatenate((sigma_zero[..., None], fit_mat), axis=-1)

        interp = interp1d(
            vn_aug,
            sigma_aug,
            kind="linear",
            axis=-1,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )

        return SelfEnergy(interp(vn_out), self.nq, False, self.has_compressed_q_dimension, False)

    def _estimate_niv_core(self, err: float = 1e-5):
        """
        Check when the real and the imaginary part are within an error margin of the asymptotic.
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
        """
        Returns purely the asymptotic behaviour of the self-energy for the given frequency range.
        Not intended to be used as its own but intended to be padded to the self-energy as an asymptotic tail.
        """
        if n_min is None:
            n_min = self.niv
        iv_asympt = 1j * MFHelper.vn(niv, config.sys.beta, shift=n_min)[None, None, ...]
        asympt = (self._smom0[..., None] - 1.0 / iv_asympt * self._smom1[..., None])[None, None, None, ...] * np.ones(
            self.nq
        )[..., None, None, None]
        return SelfEnergy(asympt, self.nq)

    def _get_orbital_axes(self) -> tuple[int, int]:
        """
        Get the axes corresponding to the orbitals.
        """
        if len(self.current_shape) == 3:  # [o1,o2,v]
            orbital_axes = (0, 1)
        elif len(self.current_shape) == 4:  # [k,o1,o2,v]
            orbital_axes = (1, 2)
        elif len(self.current_shape) == 6:  # [kx,ky,kz,o1,o2,v]
            orbital_axes = (3, 4)
        else:
            raise ValueError("The object has to have either 3, 4 or 6 dimensions.")
        return orbital_axes

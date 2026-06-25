# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Single-particle Green's function. :class:`GreensFunction` builds the momentum-dependent interacting Green's
function :math:`G_{ab}(k, \nu) = [(\imath\nu + \mu)\delta_{ab} - \varepsilon_{ab}(k) - \Sigma_{ab}(k, \nu)]^{-1}`
from a :class:`SelfEnergy`, the band dispersion :math:`\varepsilon(k)` and the chemical potential :math:`\mu`,
and derives the filling, occupation, kinetic and (Galitskii–Migdal) potential energies. The module-level helpers
adjust :math:`\mu` to a target filling via a Newton root search. Moment-corrected asymptotic sums are used so the
finite Matsubara box does not bias the energies/filling.
"""

from copy import deepcopy

import numpy as np
from scipy import optimize as opt

import dgamore.config as config
from dgamore.brillouin_zone import KGrid
from dgamore.local_n_point import LocalNPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore.n_point_base import IAmNonLocal
from dgamore.self_energy import SelfEnergy


def get_total_fill(mu: float, ek: np.ndarray, sigma_mat: np.ndarray, beta: float, smom0: np.ndarray) -> float:
    r"""
    Returns the total filling calculated from the self-energy, :math:`\mu`, kinetic Hamiltonian and more.
    Helper method for the root finding of :math:`\mu` via a Newton method. A local model Green's function built
    from the self-energy moment is subtracted to accelerate the Matsubara sum convergence.
    Side note: One could refactor some functions in this file because they are very redundant.

    :param mu: Chemical potential :math:`\mu`.
    :param ek: Band dispersion :math:`\varepsilon(k)`, shape ``[kx, ky, kz, o1, o2]``.
    :param sigma_mat: Self-energy array, shape ``[k, o1, o2, v]``.
    :param beta: Inverse temperature :math:`\beta`.
    :param smom0: Zeroth moment :math:`\Sigma_\infty` of the self-energy, shape ``[o1, o2]``.
    :return: The total filling (electron number) :math:`n`.
    """
    n_bands = sigma_mat.shape[-2]
    eye_bands = np.eye(n_bands, n_bands)
    iv = 1j * MFHelper.vn(sigma_mat.shape[-1] // 2, beta)
    iv_bands = iv[None, None, :] * eye_bands[..., None]
    mu_bands = mu * eye_bands
    hloc = np.mean(ek, axis=(0, 1, 2))

    mat = iv_bands + mu_bands[..., None] - hloc[..., None] - smom0[..., None]
    g_model_mat = np.linalg.inv(mat.transpose(2, 0, 1)).transpose(1, 2, 0)

    ek = ek.reshape(np.prod(ek.shape[:3]), n_bands, n_bands)  # sigma will always enter with shape (k,o1,o2,v)
    mat = iv_bands[None, ...] + mu_bands[None, ..., None] - ek[..., None] - sigma_mat
    g_full_mat = np.linalg.inv(mat.transpose(0, 3, 1, 2)).transpose(0, 2, 3, 1)
    g_loc_mat = np.mean(g_full_mat, axis=0)

    eigenvals, eigenvecs = np.linalg.eig(beta * (hloc.real + smom0 - mu_bands))

    rho_diag = np.empty_like(eigenvals)
    mask = eigenvals > 0
    rho_diag[mask] = np.exp(-eigenvals[mask]) / (1 + np.exp(-eigenvals[mask]))
    rho_diag[~mask] = 1 / (1 + np.exp(eigenvals[~mask]))
    rho_diag = np.einsum("...i,ij->...ij", rho_diag, np.eye(n_bands, dtype=rho_diag.dtype))

    rho_loc = eigenvecs @ rho_diag @ np.linalg.inv(eigenvecs)
    occ = rho_loc + np.sum(g_loc_mat.real - g_model_mat.real, axis=-1) / beta
    return 2.0 * np.trace(occ).real


def root_fun(
    mu: float, target_filling: float, ek: np.ndarray, sigma_mat: np.ndarray, beta: float, smom0: np.ndarray
) -> float:
    r"""
    Residual function used to find a new chemical potential :math:`\mu` via Newton's method: the difference between
    the current filling and the target filling.

    :param mu: Chemical potential :math:`\mu`.
    :param target_filling: Desired total filling.
    :param ek: Band dispersion :math:`\varepsilon(k)`.
    :param sigma_mat: Self-energy array, shape ``[k, o1, o2, v]``.
    :param beta: Inverse temperature :math:`\beta`.
    :param smom0: Zeroth moment :math:`\Sigma_\infty` of the self-energy.
    :return: The signed filling residual ``filling(mu) - target_filling``.
    """
    return get_total_fill(mu, ek, sigma_mat, beta, smom0) - target_filling


def update_mu(
    mu0: float, target_filling: float, ek: np.ndarray, sigma_mat: np.ndarray, beta: float, smom0: np.ndarray
) -> float:
    r"""
    Updates the chemical potential to match the target filling by using Newton's method to find the optimal
    :math:`\mu`. On failure to converge the starting value is returned unchanged.

    :param mu0: Initial guess for the chemical potential.
    :param target_filling: Desired total filling.
    :param ek: Band dispersion :math:`\varepsilon(k)`.
    :param sigma_mat: Self-energy array, shape ``[k, o1, o2, v]``.
    :param beta: Inverse temperature :math:`\beta`.
    :param smom0: Zeroth moment :math:`\Sigma_\infty` of the self-energy.
    :return: The updated (real) chemical potential, or ``mu0`` if the root search did not converge.
    :raises ValueError: If the converged chemical potential has a non-negligible imaginary part.
    """
    mu = mu0
    try:
        mu = opt.newton(root_fun, mu, args=(target_filling, ek, sigma_mat, beta, smom0), tol=1e-6)
    except RuntimeError:
        config.logger.debug("Root finding for chemical potential failed.")
        return mu0

    if np.abs(mu.imag) < 1e-8:
        mu = mu.real
    else:
        raise ValueError("Chemical Potential must be real.")
    return mu


class GreensFunction(IAmNonLocal, LocalNPoint):
    """
    Represents a Green's function object. The Green's function class sadly is a bit of a mess because parts of it were
    heavily inspired by DGApy.
    """

    def __init__(
        self,
        mat: np.ndarray,
        sigma: SelfEnergy = None,
        ek: np.ndarray = None,
        full_niv_range: bool = True,
        calc_filling: bool = True,
        has_compressed_q_dimension: bool = False,
    ):
        r"""
        Initializes the Green's function; if a self-energy and dispersion are given (and ``calc_filling`` is True),
        also computes the local Green's function and updates the global filling/occupation.

        :param mat: Underlying Green's function array (two orbital axes and one fermionic frequency axis, optionally
            preceded by momentum axes). Overwritten by the local Green's function when ``calc_filling`` is True.
        :param sigma: The :class:`SelfEnergy` used to construct the Green's function (optional).
        :param ek: Band dispersion :math:`\varepsilon(k)` (optional).
        :param full_niv_range: Whether the object spans the full (signed) fermionic range or only :math:`\nu \geq 0`.
        :param calc_filling: If True (and ``sigma``/``ek`` are given), compute the local Green's function and update
            the global filling/occupation in :mod:`dgamore.config`.
        :param has_compressed_q_dimension: Whether the momentum is stored as a single compressed axis (True) or as
            ``[kx, ky, kz, ...]`` (False).
        """
        LocalNPoint.__init__(self, mat, 2, 0, 1, full_niv_range=full_niv_range)
        IAmNonLocal.__init__(self, mat, config.lattice.nk, has_compressed_q_dimension)
        self._sigma = sigma
        self._ek = ek

        if sigma is not None and ek is not None and calc_filling:
            self.mat = self._get_gloc_mat()
            # config.sys.n, config.sys.occ = self._get_fill()
            config.sys.n, config.sys.occ, config.sys.occ_k = self.get_fill_nonlocal()

    @property
    def ek(self) -> np.ndarray:
        """
        The band dispersion stored on this object.

        :return: The band dispersion :math:`\\varepsilon(k)` as a numpy array.
        """
        return self._ek

    @staticmethod
    def get_g_full(siw: SelfEnergy, mu: float, ek: np.ndarray):
        r"""
        Builds the full momentum-dependent Green's function
        :math:`G(k, \nu) = [(\imath\nu + \mu) - \varepsilon(k) - \Sigma(k, \nu)]^{-1}`.

        :param siw: The :class:`SelfEnergy` :math:`\Sigma`.
        :param mu: Chemical potential :math:`\mu`.
        :param ek: Band dispersion :math:`\varepsilon(k)`.
        :return: The momentum-dependent :class:`GreensFunction` (filling not recomputed).
        """
        eye_bands = np.eye(siw.n_bands, siw.n_bands)
        iv = 1j * MFHelper.vn(siw.niv, config.sys.beta)
        iv_bands = iv[None, None, :] * eye_bands[..., None]
        mu_bands = mu * eye_bands[:, :, None]
        mat = (
            iv_bands[None, None, None, ...]
            + mu_bands[None, None, None, ...]
            - ek[..., None]
            - siw.decompress_q_dimension().mat
        )
        mat = np.linalg.inv(mat.transpose(0, 1, 2, 5, 3, 4)).transpose(0, 1, 2, 4, 5, 3)
        return GreensFunction(mat, siw, ek, siw.full_niv_range, False, False)

    @staticmethod
    def create_g_loc(siw: SelfEnergy, ek: np.ndarray, calc_filling: bool = True) -> "GreensFunction":
        """
        Builds a local (k-summed) Green's function from a self-energy and band dispersion.

        :param siw: The :class:`SelfEnergy` :math:`\\Sigma`.
        :param ek: Band dispersion :math:`\\varepsilon(k)`.
        :param calc_filling: If True, compute the filling/occupation and update :mod:`dgamore.config`.
        :return: The local :class:`GreensFunction`.
        """
        return GreensFunction(np.empty_like(siw.mat), siw, ek, siw.full_niv_range, calc_filling)

    def permute_orbitals(self, permutation: str = "ab->ab"):
        """
        Permutes the two orbital axes of the Green's function according to an einsum-style string, returning a copy
        (the identity permutation returns ``self``).

        :param permutation: A permutation of the form ``"ab->..."`` using exactly the two orbital labels.
        :return: The orbital-permuted :class:`GreensFunction`.
        :raises ValueError: If the permutation is malformed or does not list both orbitals on each side.
        """
        split = permutation.split("->")
        if len(split) != 2 or len(split[0]) != 2 or len(split[1]) != 2:
            raise ValueError("Invalid permutation.")

        if split[0] == split[1]:
            return self

        copy = deepcopy(self)

        permutation = (
            (
                f"i{split[0]}...->i{split[1]}..."
                if self.has_compressed_q_dimension
                else f"ijk{split[0]}...->ijk{split[1]}..."
            )
            if len(self.current_shape) != 3
            else f"{split[0]}v->{split[1]}v"
        )

        copy.mat = np.einsum(permutation, copy.mat, optimize=True)
        return copy

    def symmetrize_orbitals(self, orbitals: list | np.ndarray) -> "GreensFunction":
        """
        Symmetrizes the object with respect to the given (equivalent) orbitals by averaging over all permutations of
        those orbitals applied to the two orbital axes. The orbital labels are 1-based, ranging from 1 to the number
        of bands; e.g. for a 3-band object ``orbitals=[1, 3]`` symmetrizes the first and third orbital.

        :param orbitals: 1-based orbital indices to treat as equivalent.
        :return: The symmetrized :class:`GreensFunction` (``self`` unchanged if already symmetrized).
        """
        orbital_axes = self._get_orbital_axes()
        if self.is_orbitally_symmetrized(orbitals):
            return self
        return self._symmetrize_orbitals(orbitals, orbital_axes)

    def is_orbitally_symmetrized(self, orbitals: list | np.ndarray) -> bool:
        """
        Checks whether the object is already symmetric under all permutations of the given orbitals.

        :param orbitals: 1-based orbital indices to test for equivalence.
        :return: True if the object is invariant under permutations of those orbitals.
        """
        orbital_axes = self._get_orbital_axes()
        return self._is_orbitally_symmetrized(orbitals, orbital_axes)

    def map_to_full_bz(self, k_grid: KGrid, nq: tuple = None):
        """
        Unfolds the object from the irreducible BZ to the full BZ using the grid's symmetry index map (see
        :meth:`IAmNonLocal._map_to_full_bz`), with two orbital dimensions.

        :param k_grid: The :class:`KGrid` providing the irreducible-to-full BZ index mapping.
        :param nq: Optional number of momenta per direction for the unfolded grid; defaults to the object's ``nq``.
        :return: ``self`` defined on the full BZ.
        """
        return self._map_to_full_bz(k_grid, 2, nq)

    def transpose_orbitals(self):
        r"""
        Transposes the two orbital indices, :math:`G_{ab}^k \to G_{ba}^k` (see :meth:`permute_orbitals`).

        :return: The orbitally transposed :class:`GreensFunction`.
        """
        return self.permute_orbitals("ab->ba")

    def get_g_wv(self, wn: np.ndarray, niv_cut: int) -> np.ndarray:
        r"""
        Returns the frequency-shifted Green's function :math:`G_{ab}^{\nu - \omega}` on a fermionic window of half
        width ``niv_cut``, for the bosonic frequencies in ``wn``.

        :param wn: Array of bosonic Matsubara indices :math:`\omega`.
        :param niv_cut: Half width of the fermionic window :math:`\nu`.
        :return: Array of shape ``[o1, o2, w, v]``.
        """
        niv_cut_range = np.arange(-niv_cut, niv_cut)
        return self.mat[..., self.niv + niv_cut_range[None, :] - wn[:, None]]

    def get_fill_nonlocal(self) -> tuple[float, np.ndarray, np.ndarray]:
        r"""
        Computes the filling and occupation from the momentum-resolved Green's function, using the analytic
        density-matrix of the model (moment) Green's function plus the box correction to accelerate convergence.

        :return: A tuple of (i) the total filling :math:`n`, (ii) the k-averaged occupation (shape ``[o1, o2]``),
            and (iii) the k-resolved occupation (shape ``[kx, ky, kz, o1, o2]``).
        """
        mat = self._get_gfull_mat()
        g_model = self._get_g_model_k_mat()
        smom0 = self._sigma.smom[0][None, None, None, ...]

        mu_bands: np.ndarray = config.sys.mu * np.eye(self.n_bands)[None, None, None, ...]

        eigenvals, eigenvecs = np.linalg.eig(config.sys.beta * (self._ek.real + smom0 - mu_bands))
        eigenvals = eigenvals.reshape((self.nq_tot, self.n_bands))
        eigenvecs = eigenvecs.reshape((self.nq_tot, self.n_bands, self.n_bands))

        rho_diag_k = np.empty_like(eigenvals)
        mask = eigenvals > 0
        rho_diag_k[mask] = np.exp(-eigenvals[mask]) / (1 + np.exp(-eigenvals[mask]))
        rho_diag_k[~mask] = 1 / (1 + np.exp(eigenvals[~mask]))
        rho_diag_k = np.einsum("...i,ij->...ij", rho_diag_k, np.eye(self.n_bands, dtype=rho_diag_k.dtype))

        rho_k = (eigenvecs @ rho_diag_k @ np.linalg.inv(eigenvecs)).reshape((*self.nq, self.n_bands, self.n_bands))
        occ_k = rho_k + np.sum(mat.real - g_model.real, axis=-1) / config.sys.beta
        occ_k.real[np.abs(occ_k) < 1e-12] = 0.0

        occ_mean = np.mean(occ_k, axis=(0, 1, 2))
        occ_mean.real[np.abs(occ_mean) < 1e-12] = 0.0
        n_el = 2.0 * np.trace(occ_mean).real
        return n_el, occ_mean, occ_k

    def get_ekin(self) -> float:
        r"""
        Returns the kinetic energy from the band dispersion and the k-resolved occupation,
        :math:`E_{kin} = \sum_{\sigma \vec{k} ab} \varepsilon(\vec{k})_{ab}\, n(\vec{k})_{ba}`.

        :return: The kinetic energy per site.
        """
        return 2 * np.sum(self._ek * config.sys.occ_k.swapaxes(-1, -2)).real / config.lattice.k_grid.nk_tot

    def get_epot(self, niv_asympt: int = 50000) -> float:
        r"""
        Moment-corrected Galitskii–Migdal potential energy,

        .. math::

            E_{pot} = \sum_k \mathrm{Tr}[\Sigma_\infty \rho_k]
                    + \frac{1}{\beta} \sum_{k,\nu} \mathrm{Tr}[(\Sigma - \Sigma_\infty) G]
                    + \frac{1}{\beta} \big[\textstyle\sum_{\mathrm{big}} - \sum_{\mathrm{box}}\big]\,
                      \mathrm{Tr}[(\Sigma_{\mathrm{mod}} - \Sigma_\infty) G_{\mathrm{mod}}],

        i.e. the exact Hartree term, the in-box correlation part, and the analytic :math:`1/\nu^2` tail. Here
        :math:`\Sigma_{\mathrm{mod}} - \Sigma_\infty = -\Sigma_1/(\imath\nu)` and
        :math:`G_{\mathrm{mod}} = [\imath\nu + \mu - \varepsilon_k - \Sigma_\infty]^{-1}`. The model subtraction
        cancels the :math:`1/\nu^2` tail of the correlation sum (remainder :math:`\sim 1/\nu^4`), while the large
        sum supplies the part beyond the stored box.

        :param niv_asympt: Number of positive fermionic frequencies used for the asymptotic ("big") tail sum.
        :return: The potential energy per site.
        """
        smom0, smom1 = self._sigma.smom  # Sigma_inf, first tail coeff; both [o1, o2]

        # 1) Hartree: physical (tail-corrected) occupation, convergence factor exact.
        e_hartree = np.sum(smom0[None, None, None] * config.sys.occ_k.swapaxes(-1, -2)).real

        # 2) In-box correlation part: Tr[(Sigma - Sigma_inf) G], Sigma_inf already counted above.
        dsigma = self._sigma.decompress_q_dimension().mat - smom0[..., None]
        g_ba = self.decompress_q_dimension().transpose_orbitals().mat
        e_corr = (dsigma * g_ba).sum().real / config.sys.beta

        # 3) Analytic 1/v^2 model tail: replace the truncated box value by the large-box one.
        e_tail = self._model_epot(smom0, smom1, niv_asympt, config.sys.beta) - self._model_epot(
            smom0, smom1, self.niv, config.sys.beta
        )

        return (e_hartree + e_corr + e_tail) / config.lattice.k_grid.nk_tot

    def _model_epot(self, smom0, smom1, niv, beta):
        r"""
        Evaluates the analytic :math:`1/\nu^2` model potential-energy tail
        :math:`\frac{1}{\beta}\sum_{k,\nu} \mathrm{Tr}[(-\Sigma_1/\imath\nu) G_{\mathrm{mod}}]` over a frequency box
        of half width ``niv`` (used as the difference of a large and a small box in :meth:`get_epot`).

        :param smom0: Zeroth self-energy moment :math:`\Sigma_\infty`, shape ``[o1, o2]``.
        :param smom1: First self-energy tail coefficient :math:`\Sigma_1`, shape ``[o1, o2]``.
        :param niv: Number of positive fermionic frequencies in the box.
        :param beta: Inverse temperature :math:`\beta`.
        :return: The model tail contribution to the potential energy (real scalar, not yet divided by ``nk_tot``).
        """
        h = (self._ek + smom0[None, None, None]).reshape(self.nq_tot, self.n_bands, self.n_bands)
        lam, u = np.linalg.eig(h)  # once per k
        u_inv = np.linalg.inv(u)
        smom1_rot = u_inv @ smom1 @ u  # rotate tail coeff into eigenbasis

        iv = 1j * MFHelper.vn(niv, beta)
        g_diag = 1.0 / (iv[None, :] + config.sys.mu - lam[:, :, None])  # [k, band, v]
        # Tr[(-smom1/iv) G_mod] = -sum_i (smom1_rot)_ii * g_diag_i / iv
        integrand = -np.einsum("kii,kiv->kv", smom1_rot, g_diag) / iv[None, :]
        return integrand.sum().real / beta

    def _get_gfull_mat(self) -> np.ndarray:
        r"""
        Builds the full momentum-dependent Green's function array
        :math:`[(\imath\nu + \mu) - \varepsilon(k) - \Sigma(k, \nu)]^{-1}`.

        :return: The Green's function array, shape ``[kx, ky, kz, o1, o2, v]``.
        """
        iv_bands, mu_bands = self._get_g_params_local()
        iv_bands = iv_bands[None, None, None, ...]
        mu_bands = mu_bands[None, None, None, ...]

        sigma_mat = self._sigma.mat
        if len(self._sigma.mat.shape) == 3:  # (o1,o1,v)
            sigma_mat = sigma_mat[None, None, None, ...]
        mat = iv_bands + mu_bands - self._ek[..., None] - sigma_mat
        return np.linalg.inv(mat.transpose(0, 1, 2, 5, 3, 4)).transpose(0, 1, 2, 4, 5, 3)

    def _get_gloc_mat(self) -> np.ndarray:
        """
        Builds the local (k-averaged) Green's function array.

        :return: The local Green's function array, shape ``[o1, o2, v]``.
        """
        return np.mean(self._get_gfull_mat(), axis=(0, 1, 2))

    def _get_g_model_mat(self) -> np.ndarray:
        """
        Builds the local model Green's function from the zeroth self-energy moment and the k-averaged band
        dispersion. Subtracting it accelerates the Matsubara sum convergence when computing the filling.

        :return: The local model Green's function array, shape ``[o1, o2, v]``.
        """
        iv_bands, mu_bands = self._get_g_params_local()
        hloc: np.ndarray = np.mean(self._ek, axis=(0, 1, 2))
        smom0, _ = self._sigma.smom
        mat = iv_bands + mu_bands - hloc[..., None] - smom0[..., None]
        return np.linalg.inv(mat.transpose(2, 0, 1)).transpose(1, 2, 0)

    def _get_g_model_k_mat(self) -> np.ndarray:
        """
        Builds the k-resolved model Green's function from the zeroth self-energy moment and the band dispersion.
        Subtracting it accelerates the Matsubara sum convergence when computing the k-resolved occupation.

        :return: The k-resolved model Green's function array, shape ``[kx, ky, kz, o1, o2, v]``.
        """
        iv_bands, mu_bands = self._get_g_params_local()
        smom0 = self._sigma.smom[0][None, None, None, ...]
        mat = iv_bands[None, None, None] + mu_bands[None, None, None] - self._ek[..., None] - smom0[..., None]
        return np.linalg.inv(mat.transpose(0, 1, 2, 5, 3, 4)).transpose(0, 1, 2, 4, 5, 3)

    def _get_g_params_local(self):
        r"""
        Projects the fermionic frequencies :math:`\imath\nu` and the chemical potential :math:`\mu` onto the diagonal
        of the orbital/band space.

        :return: The tuple ``(iv_bands, mu_bands)`` of diagonal frequency and chemical-potential arrays.
        """
        eye_bands = np.eye(self.n_bands, self.n_bands)
        iv = 1j * MFHelper.vn(self.niv, config.sys.beta)
        iv_bands = iv[None, None, :] * eye_bands[..., None]
        mu_bands = config.sys.mu * eye_bands[:, :, None]
        return iv_bands, mu_bands

    def _get_orbital_axes(self) -> tuple[int, int]:
        """
        Determines the axes carrying the two orbital indices for the current layout (local, compressed-q, or
        decompressed-q).

        :return: The tuple of the two orbital axis indices.
        :raises ValueError: If the object does not have 3, 4, or 6 dimensions.
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

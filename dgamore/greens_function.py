# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Single-particle Green's function. :class:`GreensFunction` builds the momentum-dependent interacting Green's function
:math:`G^{\mathrm{k}}_{12} = [(\imath\nu + \mu)\delta_{12} - \varepsilon_{12}(\mathbf{k}) -
\Sigma^{\mathrm{k}}_{12}]^{-1}` from a :class:`SelfEnergy`, the band dispersion :math:`\varepsilon(\mathbf{k})` and the
chemical potential :math:`\mu`, and derives the filling, occupation, kinetic and (Galitskii-Migdal) potential energies.
The module-level helpers adjust :math:`\mu` to a target filling via a Newton root search. Moment-corrected asymptotic
sums are used so the finite Matsubara box does not bias the energies/filling.
"""

import numpy as np
from scipy import optimize as opt

from dgamore.matsubara_frequencies import MFHelper
from dgamore.mpi_utils import MpiDistributor
from dgamore.self_energy import SelfEnergy
from dgamore.two_point import TwoPoint

# Element budget for one [k, band, v-chunk] temporary of the analytic potential-energy tail (16 MiB complex128; the
# tail holds three) and for one frequency chunk of the Dyson build; the default niv_asympt = 50000 would otherwise
# materialize [nk_tot, n_bands, 2*niv_asympt] - several GB - per plain sum.
_MODEL_EPOT_CHUNK_ELEMENTS: int = 2**20

# Element budget of one momentum chunk of the frequency arrays in the occupation and potential-energy sums (64 MiB
# in complex128); each sum holds a few such arrays at once, which for many momenta on a wide frequency box would otherwise
# reach several GB.
_MOMENTUM_CHUNK_ELEMENTS: int = 2**22


def _fermi_dirac_density(h: np.ndarray, beta: float) -> np.ndarray:
    r"""
    Returns the (possibly k-resolved) single-particle density matrix :math:`\rho = f(\beta h)` of a static
    effective Hamiltonian :math:`h = \varepsilon + \Sigma_\infty - \mu` (shape ``[..., o, o]``), evaluated in the
    eigenbasis with a numerically stable Fermi function (the two branches avoid overflow of ``exp`` for large
    positive/negative eigenvalues). This is the orbital density-matrix block shared by the local filling
    (:func:`get_total_fill`) and the k-resolved occupation (:meth:`GreensFunction.get_fill_nonlocal`).

    :param h: The static effective Hamiltonian :math:`\varepsilon + \Sigma_\infty - \mu`, shape ``[..., o, o]``.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The density matrix :math:`\rho`, same shape as ``h``.
    """
    eigenvals, eigenvecs = np.linalg.eig(beta * h)

    rho_diag = np.empty_like(eigenvals)
    mask = eigenvals > 0
    rho_diag[mask] = np.exp(-eigenvals[mask]) / (1 + np.exp(-eigenvals[mask]))
    rho_diag[~mask] = 1 / (1 + np.exp(eigenvals[~mask]))

    # rho = V diag(rho_diag) V^-1. Scaling the columns of V by rho_diag is identical to the diagonal matmul (each
    # entry is a single product, no accumulation), but avoids materializing the dense [..., o, o] diagonal and one matmul.
    return (eigenvecs * rho_diag[..., None, :]) @ np.linalg.inv(eigenvecs)


def get_total_fill(
    mu: float,
    ek: np.ndarray,
    sigma_mat: np.ndarray,
    beta: float,
    smom0: np.ndarray,
    mpi_dist: MpiDistributor | None = None,
) -> float:
    r"""
    Returns the total filling for a given :math:`\mu`, self-energy and kinetic Hamiltonian. A local model Green's
    function built from the self-energy moment is subtracted to accelerate the Matsubara sum convergence. This is
    the cheap, purely local (k-summed) scalar variant used inside the :math:`\mu` root search (a Newton method);
    :meth:`GreensFunction.get_fill_nonlocal` is the k-resolved counterpart that additionally returns the
    occupation matrices. Both share the Fermi-Dirac density matrix via :func:`_fermi_dirac_density`.

    With ``mpi_dist`` every rank builds and inverts the Dyson matrix of its own momenta only, and the momentum sum
    continues from rank to rank in momentum order (:meth:`MpiDistributor.ordered_sum`); rank 0's filling is broadcast,
    so every rank returns the filling of the whole grid, bit-identical to the single-process evaluation.

    :param mu: Chemical potential :math:`\mu`.
    :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})` on the whole grid, shape ``[kx, ky, kz, o1, o2]``.
    :param sigma_mat: Self-energy array, shape ``[k, o1, o2, v]``; with ``mpi_dist`` only this rank's momenta.
    :param beta: Inverse temperature :math:`\beta`.
    :param smom0: Zeroth moment :math:`\Sigma_\infty` of the self-energy, shape ``[o1, o2]``.
    :param mpi_dist: MPI distributor over the full-BZ momenta (see :class:`MpiDistributor`), or ``None`` for the
        whole grid on this process.
    :return: The total filling (electron number) :math:`n`.
    """
    n_bands = sigma_mat.shape[-2]
    eye_bands = np.eye(n_bands, n_bands)
    iv = 1j * MFHelper.vn(sigma_mat.shape[-1] // 2, beta)
    iv_bands = iv[None, None, :] * eye_bands[..., None]
    mu_bands = mu * eye_bands
    hloc = np.mean(ek, axis=(0, 1, 2))

    mat = iv_bands + mu_bands[..., None] - hloc[..., None] - smom0[..., None]
    g_model_mat = GreensFunction._invert_last_orbital_block(mat)

    ek = ek.reshape(np.prod(ek.shape[:3]), n_bands, n_bands)  # sigma will always enter with shape (k,o1,o2,v)
    if mpi_dist is not None:
        ek = ek[mpi_dist.my_slice]
    mat = iv_bands[None, ...] + mu_bands[None, ..., None] - ek[..., None]
    g_full_mat = GreensFunction._invert_last_orbital_block(np.subtract(mat, sigma_mat, out=mat))
    g_loc_mat = np.mean(g_full_mat, axis=0) if mpi_dist is None else mpi_dist.ordered_sum(g_full_mat) / mpi_dist.ntasks

    rho_loc = _fermi_dirac_density(hloc.real + smom0 - mu_bands, beta)
    occ = rho_loc + np.sum(g_loc_mat.real - g_model_mat.real, axis=-1) / beta
    fill = 2.0 * np.trace(occ).real
    return fill if mpi_dist is None else mpi_dist.bcast(fill, root=0)


def root_fun(
    mu: float,
    target_filling: float,
    ek: np.ndarray,
    sigma_mat: np.ndarray,
    beta: float,
    smom0: np.ndarray,
    mpi_dist: MpiDistributor | None = None,
) -> float:
    r"""
    Residual function used to find a new chemical potential :math:`\mu` via Newton's method: the difference between
    the current filling and the target filling.

    :param mu: Chemical potential :math:`\mu`.
    :param target_filling: Desired total filling.
    :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})`.
    :param sigma_mat: Self-energy array, shape ``[k, o1, o2, v]``; with ``mpi_dist`` only this rank's momenta.
    :param beta: Inverse temperature :math:`\beta`.
    :param smom0: Zeroth moment :math:`\Sigma_\infty` of the self-energy.
    :param mpi_dist: MPI distributor over the full-BZ momenta (see :class:`MpiDistributor`), or ``None`` for the
        whole grid on this process.
    :return: The signed filling residual ``filling(mu) - target_filling``.
    """
    return get_total_fill(mu, ek, sigma_mat, beta, smom0, mpi_dist) - target_filling


# Largest filling residual accepted from a "converged" Newton result; genuine roots sit orders of magnitude below
# this, while a secant that stalled in a flat filling region leaves a residual of order the filling itself.
_FILL_RESIDUAL_TOL: float = 1e-3


def _find_mu_bracket(
    mu0: float, args: tuple, initial_width: float = 0.05, max_width: float = 64.0
) -> tuple[float, float] | None:
    r"""
    Expands a symmetric interval around ``mu0``, doubling its half-width each step, until the filling residual
    :func:`root_fun` changes sign across it. Starting narrow makes the search return an interval around the root
    closest to ``mu0``, which keeps a self-consistency trajectory in its current basin.

    :param mu0: Center of the search interval.
    :param args: The :func:`root_fun` arguments after ``mu`` (target filling, dispersion, self-energy, beta, moment,
        distributor).
    :param initial_width: Half-width of the first interval.
    :param max_width: Half-width beyond which the search gives up.
    :return: A bracketing interval ``(lo, hi)``, or ``None`` if no sign change was found.
    """
    width = initial_width
    while width <= max_width:
        lo, hi = mu0 - width, mu0 + width
        if root_fun(lo, *args) * root_fun(hi, *args) < 0:
            return lo, hi
        width *= 2
    return None


def update_mu(
    mu0: float,
    target_filling: float,
    ek: np.ndarray,
    sigma_mat: np.ndarray,
    beta: float,
    smom0: np.ndarray,
    logger=None,
    tol: float = 1e-6,
    mpi_dist: MpiDistributor | None = None,
) -> float:
    r"""
    Updates the chemical potential to match the target filling by using Newton's method to find the optimal
    :math:`\mu`. A Newton result is only accepted if its filling residual is small; when Newton fails or lands
    away from an actual root, the root nearest the starting value is found instead with a bracketed Brent search
    (see :func:`_find_mu_bracket`). The starting value is returned unchanged only if no bracket exists.

    With ``mpi_dist`` every rank of its communicator must call this with its own momenta: each filling evaluation
    is distributed (see :func:`get_total_fill`) and returns the same value on every rank, so all ranks run the
    same search in lockstep and return the same :math:`\mu`, bit-identical to the single-process search.

    :param mu0: Initial guess for the chemical potential.
    :param target_filling: Desired total filling.
    :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})`.
    :param sigma_mat: Self-energy array, shape ``[k, o1, o2, v]``; with ``mpi_dist`` only this rank's momenta.
    :param beta: Inverse temperature :math:`\beta`.
    :param smom0: Zeroth moment :math:`\Sigma_\infty` of the self-energy (the same on every rank).
    :param logger: Optional logger; if given, the bracketed fallback is logged at info level and a fully failed
        root search at warning level.
    :param tol: Root search tolerance for the chemical potential.
    :param mpi_dist: MPI distributor over the full-BZ momenta (see :class:`MpiDistributor`), or ``None`` for the
        whole grid on this process.
    :return: The updated (real) chemical potential, or ``mu0`` if no root was found.
    :raises ValueError: If the converged chemical potential has a non-negligible imaginary part.
    """
    mu = mu0
    args = (target_filling, ek, sigma_mat, beta, smom0, mpi_dist)
    try:
        mu = opt.newton(root_fun, mu, args=args, tol=tol)
        # the secant step criterion can also "converge" inside a flat filling region far from any root, so the
        # residual is verified before the value is accepted
        if np.abs(root_fun(mu, *args)) > _FILL_RESIDUAL_TOL:
            raise RuntimeError
    except RuntimeError:
        bracket = _find_mu_bracket(mu0, args)
        if bracket is None:
            if logger is not None:
                logger.warning("Root finding for chemical potential failed; keeping the previous value.")
            return mu0
        if logger is not None:
            logger.info("Newton did not find a chemical potential root; using a bracketed root search.")
        mu = opt.brentq(root_fun, *bracket, args=args, xtol=tol)

    if np.abs(mu.imag) < 1e-8:
        mu = mu.real
    else:
        raise ValueError("Chemical Potential must be real.")
    return mu


class GreensFunction(TwoPoint):
    r"""
    The single-particle Green's function :math:`G^{\mathrm{k}}_{12} = [(\imath\nu + \mu)\delta_{12} -
    \varepsilon_{12}(\mathbf{k}) - \Sigma^{\mathrm{k}}_{12}]^{-1}`. Built from a :class:`SelfEnergy`, the band
    dispersion :math:`\varepsilon(\mathbf{k})` and the chemical potential :math:`\mu`; on top of the two-point orbital
    bookkeeping inherited from :class:`LocalTwoPoint` it adds the Dyson construction (local and momentum-resolved) and
    the derived quantities - filling, occupation matrices, kinetic and (Galitskii-Migdal) potential energy - all using
    moment-corrected asymptotic Matsubara sums so the finite frequency box does not bias the result.
    """

    def __init__(
        self,
        mat: np.ndarray,
        sigma: SelfEnergy = None,
        ek: np.ndarray = None,
        full_niv_range: bool = True,
        calc_filling: bool = True,
        has_compressed_q_dimension: bool = False,
        nk: tuple = (1, 1, 1),
        beta: float = None,
        mu: float = None,
    ):
        r"""
        Initializes the Green's function; if a self-energy and dispersion are given (and ``calc_filling`` is True),
        also computes the local Green's function and the filling/occupation.

        :param mat: Underlying Green's function array (two orbital axes and one fermionic frequency axis, optionally
            preceded by momentum axes). Overwritten by the local Green's function when ``calc_filling`` is True.
        :param sigma: The :class:`SelfEnergy` used to construct the Green's function (optional).
        :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})` (optional).
        :param full_niv_range: Whether the object spans the full (signed) fermionic range or only :math:`\nu \geq 0`.
        :param calc_filling: If True (and ``sigma``/``ek`` are given), compute the local Green's function and the
            filling/occupation, exposed via the :attr:`n`, :attr:`occ` and :attr:`occ_k` properties.
        :param has_compressed_q_dimension: Whether the momentum is stored as a single compressed axis ``[q, ...]``
            (True) or as three separate axes ``[kx, ky, kz, ...]`` (False).
        :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
        :param beta: Inverse temperature :math:`\beta`.
        :param mu: Chemical potential :math:`\mu`.
        """
        TwoPoint.__init__(self, mat, nk, full_niv_range, has_compressed_q_dimension)
        self._sigma = sigma
        self._ek = ek
        self._beta = beta
        self._mu = mu
        self._n = None
        self._occ = None
        self._occ_k = None

        if sigma is not None and ek is not None and calc_filling:
            self.mat = self._get_gloc_mat()
            self._n, self._occ, self._occ_k = self.get_fill_nonlocal()

    @property
    def ek(self) -> np.ndarray:
        r"""
        The band dispersion stored on this object.

        :return: The band dispersion :math:`\varepsilon(\mathbf{k})` as a numpy array.
        """
        return self._ek

    @property
    def n(self) -> float:
        r"""
        The total filling computed for this Green's function.

        :return: The total filling :math:`n`, or None if the filling has not been computed.
        """
        return self._n

    @property
    def occ(self) -> np.ndarray:
        """
        The k-averaged occupation matrix.

        :return: The k-averaged occupation (shape ``[o1, o2]``), or None if it has not been computed.
        """
        return self._occ

    @property
    def occ_k(self) -> np.ndarray:
        """
        The k-resolved occupation matrix.

        :return: The k-resolved occupation (shape ``[kx, ky, kz, o1, o2]``), or None if it has not been computed.
        """
        return self._occ_k

    @staticmethod
    def get_g_full(siw: SelfEnergy, mu: float, ek: np.ndarray, beta: float):
        r"""
        Builds the full momentum-dependent Green's function :math:`G^{\mathrm{k}} = [(\imath\nu + \mu) -
        \varepsilon(\mathbf{k}) - \Sigma^{\mathrm{k}}]^{-1}`. The Dyson matrix is assembled and inverted in bounded
        fermionic-frequency chunks, so beyond the result only one chunk-sized transient is alive at a time (large
        boxes such as the full DMFT one would otherwise triple the peak).

        :param siw: The :class:`SelfEnergy` :math:`\Sigma`.
        :param mu: Chemical potential :math:`\mu`.
        :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})`.
        :param beta: Inverse temperature :math:`\beta`.
        :return: The momentum-dependent :class:`GreensFunction` (filling not recomputed).
        """
        eye_bands = np.eye(siw.n_bands, siw.n_bands)
        iv = 1j * MFHelper.vn(siw.niv, beta)
        static_bands = (mu * eye_bands)[None, None, None, :, :, None] - ek[..., None]
        sigma_mat = siw.decompress_q_dimension().mat

        # a momentum-local sigma ([1, 1, 1, ...]) broadcasts against the dispersion, so the result is always full-k
        mat = np.empty((*ek.shape[:3], siw.n_bands, siw.n_bands, 2 * siw.niv), dtype=sigma_mat.dtype)
        step = max(1, _MODEL_EPOT_CHUNK_ELEMENTS // (int(np.prod(ek.shape[:3])) * siw.n_bands**2))
        for start in range(0, 2 * siw.niv, step):
            stop = min(2 * siw.niv, start + step)
            dyson = (iv[start:stop][None, None, :] * eye_bands[..., None])[None, None, None, ...] + static_bands
            dyson -= sigma_mat[..., start:stop]
            mat[..., start:stop] = GreensFunction._invert_last_orbital_block(dyson)
        return GreensFunction(mat, siw, ek, siw.full_niv_range, False, False, nk=ek.shape[:3], beta=beta, mu=mu)

    @staticmethod
    def create_g_loc(
        siw: SelfEnergy, ek: np.ndarray, beta: float, mu: float, calc_filling: bool = True
    ) -> "GreensFunction":
        r"""
        Builds a local (k-summed) Green's function from a self-energy and band dispersion.

        :param siw: The :class:`SelfEnergy` :math:`\Sigma`.
        :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})`.
        :param beta: Inverse temperature :math:`\beta`.
        :param mu: Chemical potential :math:`\mu`.
        :param calc_filling: If True, compute the filling/occupation (exposed via the ``n``/``occ``/``occ_k``
            properties).
        :return: The local :class:`GreensFunction`.
        """
        return GreensFunction(
            np.empty_like(siw.mat), siw, ek, siw.full_niv_range, calc_filling, nk=siw.nq, beta=beta, mu=mu
        )

    @staticmethod
    def _invert_last_orbital_block(mat: np.ndarray) -> np.ndarray:
        r"""
        Inverts the trailing orbital block of a ``[..., o1, o2, v]`` array, i.e. computes the per-frequency
        (and per-momentum) matrix inverse over the two orbital axes. The fermionic axis is moved in front of the
        orbital pair so ``numpy.linalg.inv`` batches over ``[..., v]`` and is moved back afterwards. Both moves are
        views, so the only allocation is the inverse itself (the Dyson step is therefore memory-neutral).

        :param mat: Array with layout ``[..., o1, o2, v]`` (local ``[o1, o2, v]`` or momentum-resolved).
        :return: The inverted array in the same ``[..., o1, o2, v]`` layout.
        """
        return np.moveaxis(np.linalg.inv(np.moveaxis(mat, -1, -3)), -3, -1)

    def get_g_wv(self, wn: np.ndarray, niv_cut: int) -> np.ndarray:
        r"""
        Returns the frequency-shifted Green's function :math:`G_{12}^{\nu - \omega}` on a fermionic window of half
        width ``niv_cut``, for the bosonic frequencies in ``wn``.

        :param wn: Array of bosonic Matsubara indices :math:`\omega`.
        :param niv_cut: Half width of the fermionic window :math:`\nu`.
        :return: Array of shape ``[o1, o2, w, v]``.
        """
        niv_cut_range = np.arange(-niv_cut, niv_cut)
        # the local Green's function carries a single-momentum dimension; index it away for the orbital algebra
        return self.mat[0, 0, 0][..., self.niv + niv_cut_range[None, :] - wn[:, None]]

    def get_fill_nonlocal(self, real_dispersion: bool | None = None) -> tuple[float, np.ndarray, np.ndarray]:
        r"""
        Computes the filling and occupation from the momentum-resolved Green's function, using the analytic
        density-matrix of the model (moment) Green's function plus the box correction to accelerate convergence.
        The box correction is summed in momentum chunks (see :meth:`get_occupation`); this object's array is not
        read.

        :param real_dispersion: Whether the dispersion is real, so the occupation takes the real-part arithmetic
            (see :meth:`_occupation_k`); None decides on this object's dispersion.
        :return: A tuple of (i) the total filling :math:`n`, (ii) the k-averaged occupation (shape ``[o1, o2]``),
            and (iii) the k-resolved occupation (shape ``[kx, ky, kz, o1, o2]``).
        """
        occ_k = self._occupation_k(self._sigma, self._mu, self._ek, self._beta, self.niv, self.n_bands, real_dispersion)
        self._n, self._occ, self._occ_k = self._fill_from_occupation(occ_k)
        return self._n, self._occ, self._occ_k

    @staticmethod
    def get_occupation(
        siw: SelfEnergy, mu: float, ek: np.ndarray, beta: float, real_dispersion: bool | None = None
    ) -> tuple[float, np.ndarray, np.ndarray]:
        r"""
        Returns the filling and occupation of the Green's function :meth:`get_g_full` builds from the same arguments,
        bit for bit what :meth:`get_fill_nonlocal` gives on it, without building that Green's function. The Dyson
        matrix and its moment model are inverted and summed over the frequencies one momentum chunk at a time
        (:data:`_MOMENTUM_CHUNK_ELEMENTS` complex128 entries each), so beyond the result about three chunk-sized
        arrays are alive however many momenta and frequencies the self-energy carries; the density-matrix part and
        the momentum mean are evaluated on all momenta at once. Like :meth:`get_g_full`, the self-energy is
        brought to its uncompressed momentum layout in place.

        :param siw: The :class:`SelfEnergy` :math:`\Sigma` on the full fermionic range.
        :param mu: Chemical potential :math:`\mu`.
        :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})`.
        :param beta: Inverse temperature :math:`\beta`.
        :param real_dispersion: Whether the dispersion is real, so the occupation takes the real-part arithmetic
            (see :meth:`_occupation_k`); None decides on ``ek``.
        :return: A tuple of (i) the total filling :math:`n`, (ii) the k-averaged occupation (shape ``[o1, o2]``),
            and (iii) the k-resolved occupation (shape ``[kx, ky, kz, o1, o2]``).
        :raises ValueError: If the self-energy is neither momentum-local nor on the momenta of ``ek``.
        """
        siw.decompress_q_dimension()
        occ_k = GreensFunction._occupation_k(siw, mu, ek, beta, siw.niv, siw.n_bands, real_dispersion)
        return GreensFunction._fill_from_occupation(occ_k)

    @staticmethod
    def _occupation_k(
        siw: SelfEnergy,
        mu: float,
        ek: np.ndarray,
        beta: float,
        niv: int,
        n_bands: int,
        real_dispersion: bool | None = None,
    ) -> np.ndarray:
        r"""
        Returns the k-resolved occupation :math:`n_{12}(\mathbf{k}) = \rho_{12}(\mathbf{k}) + \frac{1}{\beta}
        \sum_{\nu} [G - G_{\mathrm{mod}}]_{12}^{\mathbf{k}\nu}` with the moment model :math:`G_{\mathrm{mod}}` and its
        analytic density matrix :math:`\rho` (see :func:`_fermi_dirac_density`). A real dispersion takes the real
        part of the dispersion and of the frequency sums, a complex (Hermitian) one keeps their imaginary parts. The
        decision is taken once for all momenta: from ``real_dispersion`` when given, which lets a caller holding only
        some momenta of a complex dispersion keep its complex arithmetic, otherwise from ``ek`` (real when its
        imaginary part vanishes to rounding). The frequency sums run in momentum chunks of
        :data:`_MOMENTUM_CHUNK_ELEMENTS` Green's-function entries. Every entry is the same floating-point expression
        as on the whole grid at once, so the result does not depend on the chunking.

        :param siw: The :class:`SelfEnergy` :math:`\Sigma`, momentum-local or on the momenta of ``ek`` (uncompressed).
        :param mu: Chemical potential :math:`\mu`.
        :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})`, shape ``[kx, ky, kz, o1, o2]``.
        :param beta: Inverse temperature :math:`\beta`.
        :param niv: Number of positive fermionic frequencies of the Green's function.
        :param n_bands: Number of bands.
        :param real_dispersion: Whether the dispersion is real; None decides on ``ek``.
        :return: The k-resolved occupation, shape ``[kx, ky, kz, o1, o2]``; entries of magnitude below 1e-12 get a
            zero real part.
        :raises ValueError: If the self-energy is neither momentum-local nor on the momenta of ``ek``.
        """
        eye_bands = np.eye(n_bands, n_bands)
        iv = 1j * MFHelper.vn(niv, beta)
        iv_bands = (iv[None, None, :] * eye_bands[..., None])[None]
        mu_v = (mu * eye_bands[:, :, None])[None]
        smom0 = siw.smom[0]

        if real_dispersion is None:
            real_dispersion = np.isrealobj(np.real_if_close(ek))
        ek_density = ek.real if real_dispersion else ek
        rho_k = _fermi_dirac_density(ek_density + smom0[None, None, None] - mu * eye_bands[None, None, None], beta)
        complex_box = not real_dispersion

        nk = int(np.prod(ek.shape[:3]))
        ek_flat = ek.reshape(nk, n_bands, n_bands)
        sigma_flat = siw.mat.reshape(-1, n_bands, n_bands, siw.mat.shape[-1])
        sigma_k = tuple(siw.mat.shape[:3]) if siw.mat.ndim == 6 else (sigma_flat.shape[0],)
        if sigma_flat.shape[0] != 1 and sigma_k not in (tuple(ek.shape[:3]), (nk,)):
            raise ValueError(f"The self-energy holds momenta {sigma_k}, the dispersion {tuple(ek.shape[:3])}.")
        box_sum = np.empty((nk, n_bands, n_bands), dtype=np.complex128 if complex_box else np.float64)
        step = max(1, _MOMENTUM_CHUNK_ELEMENTS // (n_bands**2 * iv.size))
        for start in range(0, nk, step):
            k = slice(start, min(nk, start + step))
            sigma = sigma_flat if sigma_flat.shape[0] == 1 else sigma_flat[k]
            g = GreensFunction._invert_last_orbital_block(iv_bands + mu_v - ek_flat[k, ..., None] - sigma)
            g_model = GreensFunction._invert_last_orbital_block(
                iv_bands + mu_v - ek_flat[k, ..., None] - smom0[None, ..., None]
            )
            box = g - g_model if complex_box else g.real - g_model.real
            box_sum[k] = np.sum(box, axis=-1)
            del g, g_model, box

        occ_k = rho_k + box_sum.reshape(rho_k.shape) / beta
        occ_k.real[np.abs(occ_k) < 1e-12] = 0.0
        return occ_k

    @staticmethod
    def _fill_from_occupation(occ_k: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Returns the total filling and the k-averaged occupation of a k-resolved occupation.

        :param occ_k: The k-resolved occupation, shape ``[kx, ky, kz, o1, o2]``.
        :return: The tuple ``(n, occ, occ_k)``; k-averaged entries of magnitude below 1e-12 get a zero real part.
        """
        occ_mean = np.mean(occ_k, axis=(0, 1, 2))
        occ_mean.real[np.abs(occ_mean) < 1e-12] = 0.0
        return 2.0 * np.trace(occ_mean).real, occ_mean, occ_k

    def get_ekin(self) -> float:
        r"""
        Returns the kinetic energy from the band dispersion and the k-resolved occupation, :math:`E_{\mathrm{kin}} =
        \sum_{\sigma \mathbf{k} ab} \varepsilon_{ab}(\mathbf{k})\, n_{ba}(\mathbf{k})`.

        :return: The kinetic energy per site.
        """
        return 2 * np.sum(self._ek * self._occ_k.swapaxes(-1, -2)).real / self.nq_tot

    def get_epot(self, niv_asympt: int = 50000) -> float:
        r"""
        Computes the moment-corrected Galitskii-Migdal potential energy,

        .. math::

            E_{pot} = \sum_k \mathrm{Tr}[\Sigma_\infty \rho_k]
                    + \frac{1}{\beta} \sum_{k,\nu} \mathrm{Tr}[(\Sigma - \Sigma_\infty) G]
                    + \frac{1}{\beta} \big[\textstyle\sum_{\mathrm{big}} - \sum_{\mathrm{box}}\big]\,
                      \mathrm{Tr}[(\Sigma_{\mathrm{mod}} - \Sigma_\infty) G_{\mathrm{mod}}],

        i.e. the exact Hartree term, the in-box correlation part, and the analytic :math:`1/\nu^2` tail. Here
        :math:`\Sigma_{\mathrm{mod}} - \Sigma_\infty = -\Sigma_1/(\imath\nu)` and :math:`G_{\mathrm{mod}} = [\imath\nu +
        \mu - \varepsilon_{\mathbf{k}} - \Sigma_\infty]^{-1}`. The model subtraction cancels the :math:`1/\nu^2` tail of
        the correlation sum (remainder :math:`\sim 1/\nu^4`), while the large sum supplies the part beyond the stored
        box. The in-box trace is taken one chunk of the first momentum axis at a time
        (:data:`_MOMENTUM_CHUNK_ELEMENTS` entries; a lone momentum only when the whole grid is one, a trailing one
        joins the chunk before it), each momentum's trace bit for bit the one of all momenta at once.

        :param niv_asympt: Number of positive fermionic frequencies used for the asymptotic ("big") tail sum.
        :return: The potential energy per site.
        """
        smom0, smom1 = self._sigma.smom  # Sigma_inf, first tail coeff; both [o1, o2]

        # 1) Hartree: physical (tail-corrected) occupation, convergence factor exact.
        e_hartree = np.sum(smom0[None, None, None] * self._occ_k.swapaxes(-1, -2)).real

        # 2) In-box correlation Tr[(Sigma - Sigma_inf) G] (Sigma_inf counted above): contract the orbital trace with
        # einsum (g orbital-transposed) to avoid the transpose_orbitals deepcopy of _ek/_sigma; per momentum chunk, so
        # Sigma - Sigma_inf stays chunk-sized. einsum reduces a lone output entry in another order, so no chunk holds a
        # single momentum unless the whole grid is one.
        sigma_mat = self._sigma.decompress_q_dimension().mat
        g = self.decompress_q_dimension().mat
        trace_k = np.empty(np.broadcast_shapes(sigma_mat.shape, g.shape)[:-3], np.result_type(sigma_mat, smom0, g))
        n_first, single_plane = g.shape[0], int(np.prod(g.shape[1:3])) == 1
        step = max(2 if single_plane else 1, _MOMENTUM_CHUNK_ELEMENTS // int(np.prod(g.shape[1:])))
        bounds = list(range(0, n_first, step)) + [n_first]
        if single_plane and len(bounds) > 2 and bounds[-1] - bounds[-2] == 1:
            del bounds[-2]  # a lone trailing momentum joins the chunk before it
        for start, stop in zip(bounds[:-1], bounds[1:]):
            k = slice(start, stop)
            sigma = sigma_mat if sigma_mat.shape[0] == 1 else sigma_mat[k]
            trace_k[k] = np.einsum("...abv,...bav->...", sigma - smom0[..., None], g[k])
        e_corr = trace_k.sum().real / self._beta

        # 3) Analytic 1/v^2 model tail: replace the truncated box value by the large-box one.
        e_tail = self._model_epot(smom0, smom1, niv_asympt, self._beta) - self._model_epot(
            smom0, smom1, self.niv, self._beta
        )

        return (e_hartree + e_corr + e_tail) / self.nq_tot

    def _model_epot(self, smom0, smom1, niv, beta):
        r"""
        Evaluates the analytic :math:`1/\nu^2` model potential-energy tail :math:`\frac{1}{\beta}\sum_{\mathrm{k}}
        \mathrm{Tr}[(-\Sigma_1/\imath\nu) G_{\mathrm{mod}}]` over a frequency box of half width ``niv`` (used as the
        difference of a large and a small box in :meth:`get_epot`).

        :param smom0: Zeroth self-energy moment :math:`\Sigma_\infty`, shape ``[o1, o2]``.
        :param smom1: First self-energy tail coefficient :math:`\Sigma_1`, shape ``[o1, o2]``.
        :param niv: Number of positive fermionic frequencies.
        :param beta: Inverse temperature :math:`\beta`.
        :return: The model tail contribution to the potential energy (real scalar, not yet divided by ``nk_tot``).
        """
        h = (self._ek + smom0[None, None, None]).reshape(self.nq_tot, self.n_bands, self.n_bands)
        lam, u = np.linalg.eig(h)  # once per k
        u_inv = np.linalg.inv(u)
        smom1_rot = u_inv @ smom1 @ u  # rotate tail coeff into eigenbasis

        # the frequency sum is evaluated in bounded v-chunks: the single-pass form materialized several
        # [k, band, 2*niv] complex temporaries, which for the default niv_asympt = 50000 spikes to many GB
        iv_full = 1j * MFHelper.vn(niv, beta)
        step = max(1, _MODEL_EPOT_CHUNK_ELEMENTS // (self.nq_tot * self.n_bands))
        total = 0.0
        for i in range(0, len(iv_full), step):
            iv = iv_full[i : i + step]
            g_diag = 1.0 / (iv[None, :] + self._mu - lam[:, :, None])  # [k, band, v-chunk]
            # Tr[(-smom1/iv) G_mod] = -sum_i (smom1_rot)_ii * g_diag_i / iv
            total += (-np.einsum("kii,kiv->kv", smom1_rot, g_diag) / iv[None, :]).sum().real
        return total / beta

    def _get_gfull_mat(self) -> np.ndarray:
        r"""
        Builds the full momentum-dependent Green's function array :math:`[(\imath\nu + \mu) - \varepsilon(\mathbf{k}) -
        \Sigma^{\mathrm{k}}]^{-1}`.

        :return: The Green's function array, shape ``[kx, ky, kz, o1, o2, v]``.
        """
        iv_bands, mu_bands = self._get_g_params_local()
        iv_bands = iv_bands[None, None, None, ...]
        mu_bands = mu_bands[None, None, None, ...]

        sigma_mat = self._sigma.mat
        if len(self._sigma.mat.shape) == 3:  # (o1,o1,v)
            sigma_mat = sigma_mat[None, None, None, ...]
        mat = iv_bands + mu_bands - self._ek[..., None] - sigma_mat
        return self._invert_last_orbital_block(mat)

    def _get_gloc_mat(self) -> np.ndarray:
        """
        Builds the local (k-averaged) Green's function array.

        :return: The local Green's function array, shape ``[o1, o2, v]``.
        """
        return np.mean(self._get_gfull_mat(), axis=(0, 1, 2))

    def _get_g_params_local(self):
        r"""
        Projects the fermionic frequencies :math:`\imath\nu` and the chemical potential :math:`\mu` onto the diagonal
        of the orbital/band space.

        :return: The tuple ``(iv_bands, mu_bands)`` of diagonal frequency and chemical-potential arrays.
        """
        eye_bands = np.eye(self.n_bands, self.n_bands)
        iv = 1j * MFHelper.vn(self.niv, self._beta)
        iv_bands = iv[None, None, :] * eye_bands[..., None]
        mu_bands = self._mu * eye_bands[:, :, None]
        return iv_bands, mu_bands

    @staticmethod
    def load(
        filename: str,
        nk: tuple[int, int, int] = (1, 1, 1),
        full_niv_range: bool = True,
        has_compressed_q_dimension: bool = False,
        beta: float = None,
        mu: float = None,
    ) -> "GreensFunction":
        r"""
        Loads a :class:`GreensFunction` from a ``.npy`` file. The stored array is kept as it is, since no self-energy
        and no dispersion are passed; the default momentum layout is the decompressed one it is written in.

        :param filename: Path to the ``.npy`` file (loaded with ``allow_pickle=False``).
        :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
        :param full_niv_range: Whether the object spans the full (signed) fermionic range or only :math:`\nu \geq 0`.
        :param has_compressed_q_dimension: Whether the momentum is stored as a single compressed axis ``[k, ...]``
            (True) or as three separate axes ``[kx, ky, kz, ...]`` (False).
        :param beta: Inverse temperature :math:`\beta`.
        :param mu: Chemical potential :math:`\mu`.
        :return: The loaded :class:`GreensFunction`.
        """
        return GreensFunction(
            np.load(filename, allow_pickle=False),
            full_niv_range=full_niv_range,
            has_compressed_q_dimension=has_compressed_q_dimension,
            nk=nk,
            beta=beta,
            mu=mu,
        )

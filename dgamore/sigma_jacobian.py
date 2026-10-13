# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Exact Jacobian of the self-energy map the self-consistency loop iterates,

.. math:: F(x) = S\big(x;\ \mu[x],\ n[x, \mu[x]]\big),

with :math:`x` the self-energy on the core window, :math:`S` the proposal of
:func:`~dgamore.nonlocal_sde.calculate_sigma_proposal`, :math:`\mu[x]` the chemical potential of the filling the loop
holds and :math:`n` the occupations its Hartree-Fock term reads. :class:`ExactJacobian` evaluates the product
:math:`J\,\delta x` analytically at one self-energy by running the loop's own functions on the linearized objects, so
the product is the derivative of the code's map, not of an idealized one. :func:`leading_eigenpairs` computes the
leading eigenpairs of :math:`J` with ARPACK, the converged-point Jacobian of Gievers et al., arXiv:2609.11405, App. G4,
and :func:`certify_subspace` certifies the eigenpairs next to a few given directions with a short block Arnoldi, the
exact check the Jacobian tracker runs inside the loop.

The Jacobian is evaluated at the pure map: no lambda correction, susceptibility restriction or annealing mass, and the
momentum-dependent :math:`V^{\mathbf{q}}` Hartree-Fock offset of the shell held at its value (exact for :math:`V = 0`),
on the self-energies the loop can reach (:func:`sector_projector`).
"""

import contextlib
import hashlib
import os
from collections.abc import Callable

import mpi4py.MPI as MPI
import numpy as np
import scipy.sparse.linalg as spla

import dgamore.brillouin_zone as bz
import dgamore.config as config
import dgamore.mpi_utils as mpi_utils
from dgamore import memory_estimator, n_point_base, nonlocal_sde, symmetry_reduction
from dgamore.bubble_gen import BubbleGenerator
from dgamore.four_point import FourPoint
from dgamore.greens_function import _MODEL_EPOT_CHUNK_ELEMENTS, GreensFunction, _fermi_dirac_density
from dgamore.interaction import Interaction, LocalInteraction
from dgamore.jacobian_stabilization import _RITZ_GATE, _condition_numbers, _schur_ritz, to_mat, to_vec
from dgamore.matsubara_frequencies import MFHelper
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel, deferred_collection
from dgamore.self_energy import SelfEnergy

_CHANNELS = ((SpinChannel.DENS, 1.0), (SpinChannel.MAGN, 3.0))  # ladder channels and their weights in the kernel
_F_DC = ("f_dc_loc.npy", SpinChannel.MAGN, (4, 0, 1, 5, 2, 3, 6))  # the dc vertex, stored in its contraction order
_INVERSE_ORDER = (0, 5, 6, 1, 2, 4, 3)  # storage order of a one-fermion object after FourPoint.invert
_MAX_MATVECS = 150  # rough bound on the Jacobian-vector products of one ARPACK target
_CHECK_BASIS = memory_estimator.EXACT_CHECK_BASIS  # basis vectors, i.e. products, of one exact check
_DUPLICATE_OVERLAP = 0.9  # norm share of an eigenvector in the span of the kept ones of its value that repeats them
# compound slice size n_bands^2 * 2 niv_core from which recomputed three-leg vertices share the responses' factorization
_FUSED_SOLVE_MIN_COMPOUND = 320


def _squared_box_sum(sigma_mat: np.ndarray, mu: float, ek: np.ndarray, beta: float, complex_ek: bool) -> np.ndarray:
    r"""
    Returns :math:`\sum_{\nu}(G^{\mathrm{k}})^2` per momentum over the frequency box of ``sigma_mat``,
    :math:`G^{\mathrm{k}} = [(\imath\nu + \mu) - \varepsilon(\mathbf{k}) - \Sigma^{\mathrm{k}}]^{-1}`, i.e. minus
    the derivative of the box sum of the occupation with respect to :math:`\mu`, in double-precision momentum chunks;
    its real part for a real dispersion, as :meth:`GreensFunction._occupation_k` takes the box sum (the
    ``real_dispersion`` of :meth:`GreensFunction.get_occupation`).

    :param sigma_mat: The self-energy ``[k, o1, o2, 2 niv]``; a single momentum row broadcasts over ``ek``.
    :param mu: Chemical potential :math:`\mu`.
    :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})`, shape ``[k, o1, o2]``.
    :param beta: Inverse temperature :math:`\beta`.
    :param complex_ek: Whether the sums keep their imaginary part: the decision of the whole dispersion, so a real
        momentum slice of a complex dispersion keeps it as the occupation does; False returns the real part.
    :return: The sums ``[k, o1, o2]`` (complex128 for a complex dispersion, float64 otherwise).
    """
    nk, nb = ek.shape[0], ek.shape[-1]
    iv = 1j * MFHelper.vn(sigma_mat.shape[-1] // 2, beta)
    static = (iv[:, None, None] + mu) * np.eye(nb)
    out = np.empty((nk, nb, nb), dtype=np.complex128 if complex_ek else np.float64)
    for start, stop in mpi_utils.row_chunks(nk, 1, nb * nb * iv.size, _MODEL_EPOT_CHUNK_ELEMENTS):
        rows = sigma_mat if sigma_mat.shape[0] == 1 else sigma_mat[start:stop]
        g = np.linalg.inv(static[None] - ek[start:stop, None] - np.moveaxis(rows, -1, 1))
        squared = np.einsum("kvab,kvbc->kac", g, g, optimize=True)
        out[start:stop] = squared if complex_ek else squared.real
    return out


def _sandwich(g: np.ndarray, m: np.ndarray | None = None) -> np.ndarray:
    """
    Returns the orbital matrix product ``g @ m @ g`` per momentum and frequency (frequency axis last), or ``g @ g``
    without ``m``.

    :param g: The Green's function ``[..., o1, o2, v]``.
    :param m: The middle factor, same layout, or ``None``.
    :return: The product in the layout of ``g``.
    """
    gv = np.moveaxis(g, -1, -3)
    left = gv if m is None else gv @ np.moveaxis(m, -1, -3)
    return np.moveaxis(left @ gv, -3, -1)


def sector_projector(k_grid: bz.KGrid, keep: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    r"""
    Returns the projector onto the self-energies the self-consistency loop can reach, acting on complex sector arrays
    ``[nk_irr, o1, o2, ...]`` (irreducible momenta in ``k_grid.irrk_ind`` order). Every irreducible momentum is averaged
    over its little group, the symmetry operations that leave it fixed, with their orbital action,

    .. math:: (P y)^{\mathbf{k}} = \frac{1}{|H_{\mathbf{k}}|} \sum_{h \in H_{\mathbf{k}}} U_h\, y^{\mathbf{k}}\,
        U_h^\dagger,

    and the orbital pairs outside ``keep`` are zeroed. Only auto-discovered symmetries act on the orbitals, so on any
    other grid the average is the identity. The exact zeros of a symmetric self-energy form a pattern its own symmetry
    maps onto itself, so the average and the mask commute and their product is again a projector.

    :param k_grid: The momentum grid with the irreducible representatives and, in auto mode, the symmetry group.
    :param keep: Boolean ``[o1, o2]`` mask of the orbital pairs the loop populates.
    :return: The projector, mapping a complex sector array to one of the same shape.
    """
    stabilizers, counts = [], None
    if k_grid.is_auto:
        stabilizers = symmetry_reduction.stabilizer_unitaries(k_grid._auto_group, tuple(k_grid.nk), k_grid.irrk_ind)
        counts = np.bincount(np.concatenate([positions for positions, _ in stabilizers]), minlength=k_grid.nk_irr)
    mask = keep[None, :, :, None]

    def project(y: np.ndarray) -> np.ndarray:
        """
        Averages every irreducible momentum over its little group and zeroes the unpopulated orbital pairs.

        :param y: The complex sector array.
        :return: The projected array.
        """
        if not stabilizers:
            return y * mask
        out = np.zeros_like(y)
        for positions, u in stabilizers:
            out[positions] += np.einsum("ab,kbcv,dc->kadv", u, y[positions], u.conj(), optimize=True)
        return out / counts[:, None, None, None] * mask

    return project


def tracker_sector_maps(
    k_grid: bz.KGrid, niv_core: int, nb: int, beta: float
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    r"""
    Returns the maps between a core window of the loop's iterates and its weighted sector vector, the coordinates of
    the Jacobian tracker: the irreducible momenta and the positive core frequencies, each momentum scaled by
    :math:`\sqrt{2 m_{\mathbf{k}}}` with :math:`m_{\mathbf{k}}` its star size (``k_grid.irrk_count``) and the factor
    two the negative frequencies stand for, in the real layout of :func:`~dgamore.jacobian_stabilization.to_vec`. On
    the windows the loop iterates, unfolded from their irreducible momenta with the lattice symmetry and completed by
    the Matsubara Hermiticity, the weighted sector is an isometry: every norm and inner product the tracker forms is
    the one of the whole window.

    :param k_grid: The momentum grid of the loop.
    :param niv_core: Number of positive fermionic frequencies of the core window.
    :param nb: Number of bands.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The maps ``(to_sector, from_sector)``: the first takes a complex core window with its momenta in
        either layout to the real sector vector, the second a real sector vector back to the complex window with a
        compressed momentum axis, in the storage precision.
    """
    window_shape = (k_grid.nk_tot, nb, nb, 2 * niv_core)
    sector_shape = (k_grid.nk_irr, nb, nb, niv_core)
    weight = np.sqrt(2.0 * k_grid.irrk_count)[:, None, None, None]

    def to_sector(window: np.ndarray) -> np.ndarray:
        """Weighted real sector vector of a complex core window."""
        return to_vec(np.reshape(window, window_shape)[k_grid.irrk_ind, ..., niv_core:] * weight)

    def from_sector(vector: np.ndarray) -> np.ndarray:
        """Complex core window unfolded from a weighted real sector vector."""
        half = SelfEnergy(to_mat(vector, sector_shape) / weight, k_grid.nk, False, True, calc_smom=False, beta=beta)
        return half.map_to_full_bz(k_grid).to_full_niv_range().mat

    return to_sector, from_sector


class ExactJacobian:
    r"""
    The Jacobian of the loop's self-energy map at one self-energy as a collective Jacobian-vector product.

    The operator acts on the weighted sector vectors of the Jacobian tracker (:func:`tracker_sector_maps`), restricted
    by :meth:`project` to the self-energies the loop can reach: every product projects its input and its output
    (:func:`sector_projector`), so the eigenvalues are those of the map on those self-energies.

    One product :math:`J\,\delta x`, the proposal's stages linearized:

    1. rank 0: :math:`\delta\mu = -\partial_x N\cdot\delta x/\partial_\mu N` for the filling ``update_mu`` holds,
       and :math:`\delta n^{\mathbf{k}} = \frac{1}{\beta}\sum_{\nu}G^{\mathrm{k}}\,\delta x^{\mathrm{k}}
       \,G^{\mathrm{k}} + \kappa^{\mathbf{k}}\,\delta\mu` (its real part for a real dispersion, as the occupation);
    2. node roots: :math:`\delta G^{\mathrm{k}} = G^{\mathrm{k}}(\theta_{\mathrm{core}}\,\delta x^{\mathrm{k}} -
       \delta\mu)\,G^{\mathrm{k}}` on the loop box, transformed to real space once per column;
    3. per window of stored bosonic frequencies: :math:`\delta\chi^{\mathrm{q}\nu}_{0} = \chi_{0}[G, \delta G] +
       \chi_{0}[\delta G, G]`, the derivative of the bubble, which is quadratic in :math:`G` (see
       :meth:`~dgamore.bubble_gen.BubbleGenerator.create_generalized_chi0_q_fft_window`);
    4. per window and channel the Bethe-Salpeter system of :func:`~dgamore.nonlocal_sde.create_auxiliary_chi_r_q_sum`
       solved for :math:`t^{\mathrm{q}\nu}_{r} = (\chi^{\mathrm{q}\nu}_{0})^{-1}\,\delta\chi^{\mathrm{q}\nu}_{0}\,
       \gamma^{\mathrm{q}\nu}_{r}`, giving :math:`s^{\mathrm{q}\nu}_{r}`,
       :math:`\delta\gamma^{\mathrm{q}\nu}_{r} = (\chi^{\mathrm{q}\nu}_{0})^{-1}s^{\mathrm{q}\nu}_{r} -
       t^{\mathrm{q}\nu}_{r}` and the change :math:`\delta\hat\chi^{*\mathrm{q}}_{r}` of the shell-corrected sum,
       then :math:`\delta\chi^{\mathrm{q}}_{r} = (1 - \chi^{\mathrm{q}}_{r}\mathcal{U}^{\mathbf{q}}_{r})\,
       \delta\hat\chi^{*\mathrm{q}}_{r}\,(1 - \mathcal{U}^{\mathbf{q}}_{r}\chi^{\mathrm{q}}_{r})` and
       :math:`\delta K^{\mathrm{q}\nu}_{r} = (\delta\gamma^{\mathrm{q}\nu}_{r} - \delta\gamma^{\mathrm{q}\nu}_{r}
       \mathcal{U}^{\mathbf{q}}_{r}\chi^{\mathrm{q}}_{r} - \gamma^{\mathrm{q}\nu}_{r}\mathcal{U}^{\mathbf{q}}_{r}
       \delta\chi^{\mathrm{q}}_{r})\,\mathcal{U}^{\mathbf{q}}_{r}`, beside the double-counting kernel of
       :math:`\delta\chi^{\mathrm{q}\nu}_{0}`;
    5. per window, the column contraction of :class:`~dgamore.nonlocal_sde._ColumnSdeStream` of :math:`(K, \delta G)`
       and :math:`(\delta K, G)` into one output per column, and on rank 0 the Hartree-Fock term of
       :math:`\delta n^{\mathbf{k}}`;
    6. rank 0, for several orbitals on a lattice with real hoppings: the time-reversal average of the proposal
       (:meth:`~dgamore.self_energy.SelfEnergy.symmetrize_time_reversal`), which is linear.

    Every stage of steps 3 to 5 acts per momentum and bosonic frequency, and the products with an interaction take
    the bosonic frequency as a batch index, so a product does not depend on the window width.

    Held between products: per node the loop-box Green's function and its real-space window, the change window and
    the local vertices (unless loaded per window, see :meth:`_local_vertex`), per rank the kernel, the inverse core
    bubble, both channels' susceptibilities and three-leg vertices (the latter unless recomputed per window, see
    :meth:`_kernel_responses`) on its momenta, on rank 0 the filling derivatives and the Hartree-Fock interaction.

    The derivative treats the fitted moment :math:`\Sigma_\infty` of the loop box and of the DMFT box as fixed, which
    holds while their fit windows lie beyond the core window; the constructor warns when one reaches into it.
    """

    def __init__(
        self,
        sigma: SelfEnergy,
        mu: float,
        u_loc: LocalInteraction,
        v_nonloc: Interaction,
        v_nonloc_full: Interaction | None,
        sigma_dmft_full: SelfEnergy,
        mpi_dist_irrk: MpiDistributor,
        mpi_dist_fullbz: MpiDistributor,
        comm: MPI.Comm,
        chunk_budgets: memory_estimator.ChunkBudgets | None = None,
    ):
        r"""
        Builds the forward quantities at the self-energy ``sigma`` window by window of stored bosonic frequencies
        (collective over ``comm``).

        :param sigma: The loop's :class:`SelfEnergy` on its frequency box (full BZ, identical on every rank;
            compressed in place).
        :param mu: Chemical potential :math:`\mu` of ``sigma``.
        :param u_loc: The bare local interaction :math:`U`.
        :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}`, reduced to this rank's irreducible q-points.
        :param v_nonloc_full: The non-local interaction on the full q-grid (for the Hartree/Fock term); only read on
            rank 0.
        :param sigma_dmft_full: The DMFT :class:`SelfEnergy` supplying the shell frequencies (momentum-local).
        :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
        :param mpi_dist_fullbz: MPI distributor over the full BZ q-points.
        :param comm: The MPI communicator.
        :param chunk_budgets: Chunk byte budget, block width, window width and residency of the products (see
            :class:`~dgamore.memory_estimator.ChunkBudgets`); ``None`` gives the job-wide fair-share budget of
            :func:`~dgamore.nonlocal_sde._sde_chunk_budget`, one-column products, one window and held vertices.
        """
        box, k_grid, nb = config.box, config.lattice.k_grid, config.sys.n_bands
        beta = float(config.sys.beta)
        self._comm, self._dist = comm, mpi_dist_irrk
        self._u_loc, self._v_nonloc, self._v_nonloc_full = u_loc, v_nonloc, v_nonloc_full
        self._beta, self._mu = beta, float(mu)
        self._ek = config.lattice.hamiltonian.get_ek()
        # the occupation of a complex dispersion (decided on the whole dispersion, the same on every momentum slice)
        # and the proposal's time-reversal average, as the forward map has them
        self._complex_ek = np.iscomplexobj(np.real_if_close(self._ek))
        self._tr_average = nb > 1 and nonlocal_sde._has_time_reversal(self._ek)
        if any(niv - SelfEnergy._n_freq_fit(niv) < box.niv_core for niv in (sigma.niv, sigma_dmft_full.niv)):
            config.logger.warning(
                f"Exact Jacobian: the moment-fit window of the loop box (niv {sigma.niv}) or of the DMFT box (niv "
                f"{sigma_dmft_full.niv}) reaches into the core window (niv_core {box.niv_core}); the products hold "
                f"the fitted Sigma_inf fixed and miss its change."
            )
        self._sector_shape = (k_grid.nk_irr, nb, nb, box.niv_core)
        self.n_real = 2 * int(np.prod(self._sector_shape))
        self._to_sector, self._from_sector = tracker_sector_maps(k_grid, box.niv_core, nb, beta)
        self._project = None
        if comm.rank == 0:
            rows = sigma.mat.reshape(-1, nb, nb, sigma.mat.shape[-1])[
                ..., sigma.niv - box.niv_core : sigma.niv + box.niv_core
            ]
            keep = np.array([[bool(np.any(rows[:, a, b] != 0)) for b in range(nb)] for a in range(nb)])
            self._project = sector_projector(k_grid, keep)

        with deferred_collection():
            self._giwk, self._giwk_win, self._node_comm = nonlocal_sde._build_giwk_full(comm, sigma, mu, self._ek, beta)
            # the sector vectors of a product reach the node roots only, the ranks that unfold them
            self._roots_comm = comm.Split(0 if self._node_comm.rank == 0 else MPI.UNDEFINED)
            # the Hartree-Fock term of every product reads the same real-space interaction
            self._w_r = (u_loc + v_nonloc_full).fft(copy=False).decompress_q_dimension().mat if comm.rank == 0 else None
            sigma.compress_q_dimension()
            self._dn_dmu, self._kappa = self._filling_derivatives(sigma, sigma_dmft_full, mpi_dist_fullbz)

            budgets = memory_estimator.ChunkBudgets() if chunk_budgets is None else chunk_budgets
            self._sde_chunk = (
                nonlocal_sde._sde_chunk_budget(comm, self._node_comm) if chunk_budgets is None else budgets.exact
            )
            self._block = max(1, budgets.exact_block)
            self._vertices_per_window = budgets.exact_vertices_per_window
            n_w = box.niw_core + 1
            width = n_w if budgets.exact_window is None else max(1, min(n_w, budgets.exact_window))
            self._w_windows = [(w0, min(n_w, w0 + width)) for w0 in range(0, n_w, width)]
            self._vertices = {}
            # one real-space window per Green's function serves the bubbles and the contraction
            self._niv_r = box.niv_full + box.niw_core
            self._g_r, self._g_r_win = nonlocal_sde._build_rspace_giwk_window(
                self._giwk, self._node_comm, niv_window=self._niv_r
            )

            self._channels = []
            for channel, weight in _CHANNELS:
                u_r = u_loc.as_channel(channel)
                self._channels.append(
                    (weight, channel, v_nonloc.as_channel(channel) + u_r, self._inactive_pairs(channel, u_r))
                )
            self._kernel = self._held(SpinChannel.NONE, 1)
            self._gchi0_inv = self._held(SpinChannel.NONE, 1, _INVERSE_ORDER)
            self._chi = {channel: self._held(channel, 0) for _, channel, *_ in self._channels}
            self._vrg = {}
            if not budgets.exact_vrg_per_window:
                self._vrg = {channel: self._held(channel, 1) for _, channel, *_ in self._channels}
            for window in self._w_windows:
                self._build_window(self._bubbles(window)[0], window)

            shape, dtype = self._giwk.mat.shape, self._giwk.mat.dtype
            self._dg, self._dg_win = mpi_utils.allocate_node_shared_array(self._node_comm, shape, dtype)

    @contextlib.contextmanager
    def _local_vertex(
        self,
        name: str,
        channel: SpinChannel,
        axes: tuple[int, ...] | None = None,
        w_window: tuple[int, int] | None = None,
    ):
        """
        Yields a local vertex of the output folder, held once per node (collective), or the window ``w_window`` of its
        stored bosonic indices: a view of the held vertex, or a fresh load of that window of the file (see
        :func:`~dgamore.nonlocal_sde._load_node_shared_local_vertex`). A fresh load is kept when the vertices stay
        resident and freed at the end of the phase when they are loaded per window. The load is freed on the normal
        path only: an exception aborts the job instead of meeting another collective.

        :param name: File name of the vertex in the output folder.
        :param channel: Spin channel of the vertex (see :class:`SpinChannel`).
        :param axes: The axis order its window stores (see
            :func:`~dgamore.nonlocal_sde._load_node_shared_local_vertex`).
        :param w_window: The window ``(start, stop)`` of stored bosonic indices (half niw range), or ``None`` for the
            whole vertex.
        :return: A context yielding the :class:`LocalFourPoint`.
        """
        if name not in self._vertices:
            path = os.path.join(config.output.output_path, name)
            window = w_window if self._vertices_per_window else None
            vertex, win = nonlocal_sde._load_node_shared_local_vertex(
                self._node_comm, path, channel, axes=axes, w_window=window
            )
            if self._vertices_per_window:
                yield vertex
                vertex.mat = None
                nonlocal_sde._free_shared_window(win, self._node_comm)
                return
            self._vertices[name] = (vertex, win)
        vertex = self._vertices[name][0]
        yield vertex if w_window is None else vertex.take_wn_slice(*w_window, copy=False)

    def _inactive_pairs(self, channel: SpinChannel, u_r: LocalInteraction) -> np.ndarray:
        """
        Returns the orbital pairs without vertex of a channel (see
        :meth:`~dgamore.local_four_point.LocalFourPoint.orbital_pairs_without_vertex`), scanned once per operator: on
        the held vertex, or window by window of the operator's windows when the vertices are loaded per window
        (collective over the node, see :meth:`_local_vertex`), a pair being without vertex when it is so in every
        window.

        :param channel: Spin channel of the vertex (see :class:`SpinChannel`).
        :param u_r: The channel-projected local interaction :math:`U_{r}`.
        :return: The sorted flat indices of the pairs without vertex.
        """
        inactive = None
        for window in self._w_windows if self._vertices_per_window else [None]:
            with self._local_vertex(f"gamma_{channel.value}_loc.npy", channel, w_window=window) as gamma:
                pairs = gamma.orbital_pairs_without_vertex(u_r)
            inactive = pairs if inactive is None else np.intersect1d(inactive, pairs)
        return inactive

    def _held(self, channel: SpinChannel, num_vn: int, order: tuple[int, ...] | None = None) -> FourPoint:
        """
        Returns a zero object on this rank's irreducible q-points and every stored bosonic frequency (half niw range,
        core fermionic box), which the constructor fills window by window.

        :param channel: Spin channel of the object (see :class:`SpinChannel`).
        :param num_vn: Number of fermionic frequency axes (0 or 1).
        :param order: The axis order of its storage, ``None`` for its own.
        :return: The zero :class:`FourPoint` (compressed momentum axis).
        """
        box, nb = config.box, config.sys.n_bands
        shape = (self._dist.my_size, nb, nb, nb, nb, box.niw_core + 1) + (2 * box.niv_core,) * num_vn
        order = tuple(range(len(shape))) if order is None else order
        storage = np.zeros(tuple(shape[a] for a in order), dtype=n_point_base.DTYPE)
        return FourPoint(
            storage.transpose(np.argsort(order)), channel, config.lattice.k_grid.nk, 1, num_vn, False, True, True
        )

    def project(self, y: np.ndarray) -> np.ndarray:
        """
        Projects a real sector vector onto the self-energies the loop can reach (:func:`sector_projector`), on rank 0.

        :param y: The real sector vector.
        :return: The projected real sector vector.
        """
        return to_vec(self._project(to_mat(y, self._sector_shape)))

    def _filling_derivatives(
        self, sigma: SelfEnergy, sigma_dmft_full: SelfEnergy, mpi_dist_fullbz: MpiDistributor
    ) -> tuple[float | None, np.ndarray | None]:
        r"""
        Returns the derivatives of the filling and the occupations with respect to :math:`\mu` (collective): the
        filling of :func:`~dgamore.greens_function.get_total_fill` on the loop box, with its local model,

        .. math:: \partial_\mu N = 2\,\mathrm{tr}\Big[\beta\bar\rho(1 - \bar\rho) + \frac{1}{\beta}\sum_{\nu}
            \mathrm{Re}\big((\bar G^{\nu}_{\mathrm{mod}})^2 - \langle (G^{\mathrm{k}})^2\rangle_{\mathbf{k}}\big)\Big],

        and the k-resolved occupation of :func:`~dgamore.nonlocal_sde._update_occ_and_energies_distributed` on the
        DMFT box, :math:`\kappa^{\mathbf{k}} = \beta\rho^{\mathbf{k}}(1 - \rho^{\mathbf{k}}) + \frac{1}{\beta}
        \sum_{\nu}\big((G^{\mathrm{k}}_{\mathrm{mod}})^2 - (G^{\mathrm{k}})^2\big)` (complex Hermitian for a complex
        dispersion, its real part for a real one, as the occupation), both summed over the momentum slices of the
        ranks.

        :param sigma: The loop's :class:`SelfEnergy` (compressed momentum axis).
        :param sigma_dmft_full: The DMFT :class:`SelfEnergy` supplying the shell frequencies (momentum-local).
        :param mpi_dist_fullbz: MPI distributor over the full BZ q-points.
        :return: :math:`\partial_\mu N` and :math:`\kappa^{\mathbf{k}}` ``[kx, ky, kz, o1, o2]`` on rank 0; ``None``
            for both elsewhere.
        """
        beta, mu, nb, complex_ek = self._beta, self._mu, config.sys.n_bands, self._complex_ek
        eye = np.eye(nb)
        ek = self._ek.reshape(-1, nb, nb)
        ek_my = ek[mpi_dist_fullbz.my_slice]

        sigma_occ = nonlocal_sde._occupation_self_energy(sigma, sigma_dmft_full, mpi_dist_fullbz)
        smom0 = sigma_occ.smom[0]
        rho = _fermi_dirac_density((ek_my if complex_ek else ek_my.real) + smom0 - mu * eye, beta)
        model = np.broadcast_to(smom0[None, :, :, None], (1, nb, nb, sigma_occ.mat.shape[-1]))
        box_sum = _squared_box_sum(model, mu, ek_my, beta, complex_ek) - _squared_box_sum(
            sigma_occ.mat, mu, ek_my, beta, complex_ek
        )
        kappa_my = beta * rho @ (eye - rho) + box_sum / beta
        sigma_occ.free()
        kappa = nonlocal_sde._assemble_occupation(kappa_my.reshape(-1, 1, 1, nb, nb), mpi_dist_fullbz)[2]

        # only the real part of the trace enters the filling, so these sums stay real on every slice
        g2_sum = _squared_box_sum(sigma.mat[mpi_dist_fullbz.my_slice], mu, ek_my, beta, False).sum(axis=0)
        if mpi_dist_fullbz.mpi_size > 1:
            g2_sum = mpi_dist_fullbz.allreduce(g2_sum)
        if self._comm.rank != 0:
            return None, None
        smom0 = sigma.fit_smom()[0]
        hloc = np.mean(self._ek, axis=(0, 1, 2))
        rho = _fermi_dirac_density(hloc.real + smom0 - mu * eye, beta)
        model = np.broadcast_to(smom0[None, :, :, None], (1, nb, nb, sigma.mat.shape[-1]))
        model_sum = _squared_box_sum(model, mu, hloc[None], beta, False)[0]
        inner = beta * rho @ (eye - rho) + (model_sum - g2_sum / ek.shape[0]) / beta
        return float(2.0 * np.trace(inner).real), kappa

    def _bubbles(self, w_window: tuple[int, int], dg_rs: list[np.ndarray] | tuple = ()) -> list[FourPoint]:
        """
        Returns the bubble of :math:`G`, or with changes the bilinear bubble of every change, on one window of stored
        bosonic frequencies of the loop's box (collective, see
        :meth:`~dgamore.bubble_gen.BubbleGenerator.create_generalized_chi0_q_fft_window`).

        :param w_window: The window ``(start, stop)`` of stored bosonic indices (half niw range).
        :param dg_rs: The real-space windows of the changes.
        :return: The bubbles on this rank's irreducible q-points (full fermionic box, the window's frequencies).
        """
        return BubbleGenerator.create_generalized_chi0_q_fft_window(
            self._dist,
            self._g_r,
            config.box.niv_full,
            config.lattice.k_grid,
            self._beta,
            w_window,
            dg_rs,
            with_forward=not dg_rs,
        )

    def _split_bubble(self, gchi0_q: FourPoint, contraction: FourPoint) -> tuple[FourPoint, ...]:
        """
        Splits a bubble into the frequency sums the shell correction reads and its core box, as the proposal does, and
        attaches the interaction to its double-counting contraction once ``gchi0_q`` is freed, so the full box is gone
        before the attachment allocates its block. The attachment takes the bosonic frequency as a batch index (see
        :meth:`~dgamore.four_point.FourPoint.matmul`), so a window's kernel does not depend on the window.

        :param gchi0_q: The bubble on the full fermionic box.
        :param contraction: The double-counting vertex contraction of ``gchi0_q``.
        :return: The double-counting kernel, the sums over the full and the core box and the core-box bubble.
        """
        beta = self._beta
        full_sum = gchi0_q.sum_over_all_vn(beta).scale(1.0 / beta)
        core = gchi0_q.cut_niv(config.box.niv_core)
        gchi0_q.free()
        core_sum = core.sum_over_all_vn(beta).scale(1.0 / beta)
        return nonlocal_sde._attach_dc_interaction(contraction, self._u_loc, per_wn=True), full_sum, core_sum, core

    def _build_window(self, bubble: FourPoint, w_window: tuple[int, int]) -> None:
        r"""
        Fills one window of stored bosonic frequencies of the held quantities from the window's bubble of :math:`G`,
        which is freed (collective over the node when the local vertices are loaded per window, see
        :meth:`_local_vertex`): the double-counting kernel and, per channel, the kernel term of the three-leg vertex
        and the shell-corrected susceptibility enter the kernel, the inverse core bubble and the susceptibilities are
        kept, and the three-leg vertices too unless they are recomputed per window (see :meth:`_kernel_responses`).

        :param bubble: The bubble of :math:`G` on the full fermionic box and the window's frequencies.
        :param w_window: The window ``(start, stop)`` of stored bosonic indices (half niw range).
        :return: None.
        """
        beta, u_loc, (w0, w1) = self._beta, self._u_loc, w_window
        with self._local_vertex(*_F_DC, w_window=w_window) as f_dc:
            contraction = nonlocal_sde._contract_dc_vertex(f_dc, bubble)
        dc_kernel, full_sum, core_sum, core = self._split_bubble(bubble, contraction)
        contraction = None
        kernel = self._kernel.take_wn_slice(w0, w1, copy=False).add(dc_kernel, copy=False)
        dc_kernel.free()
        gchi0_inv = self._gchi0_inv.take_wn_slice(w0, w1, copy=False).add(core.invert(copy=False), copy=False)
        core.free()
        for weight, channel, u_r, inactive in self._channels:
            with self._local_vertex(f"gamma_{channel.value}_loc.npy", channel, w_window=w_window) as gamma:
                aux = nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_inv, u_loc, inactive=inactive)
            chi = nonlocal_sde.create_generalized_chi_q_with_shell_correction(
                aux.sum_over_all_vn(beta), full_sum, core_sum, u_loc, self._v_nonloc
            )
            chi = self._chi[channel].take_wn_slice(w0, w1, copy=False).add(chi, copy=False)
            vrg = nonlocal_sde.create_vrg_r_q(aux, gchi0_inv, copy=False)
            term = nonlocal_sde.calculate_kernel_r_q(vrg, chi, self._v_nonloc, u_loc, per_wn=True).scale(weight)
            kernel.add(term, copy=False)
            term.free()
            if channel in self._vrg:
                self._vrg[channel].take_wn_slice(w0, w1, copy=False).add(vrg, copy=False)
            vrg.free()
        full_sum.free()
        core_sum.free()

    def _greens_function(self, mat: np.ndarray) -> GreensFunction:
        """
        Wraps an array of the loop box, e.g. a node-shared window, into a :class:`GreensFunction` without copying.

        :param mat: The array ``[kx, ky, kz, o1, o2, 2 niv]``.
        :return: The :class:`GreensFunction` viewing ``mat``.
        """
        return GreensFunction(
            mat, None, self._ek, True, False, False, nk=self._ek.shape[:3], beta=self._beta, mu=self._mu
        )

    def _occupation_change(self, dx: np.ndarray) -> np.ndarray:
        r"""
        Returns :math:`\frac{1}{\beta}\sum_{\nu}G^{\mathrm{k}}\,\delta x^{\mathrm{k}}\,G^{\mathrm{k}}` per momentum
        over the core window, the occupation change at fixed :math:`\mu`, in momentum chunks (rank 0); its real part
        for a real dispersion, as the occupation takes it.

        :param dx: The window change ``[k, o1, o2, 2 niv_core]``.
        :return: The change ``[k, o1, o2]`` (complex128 for a complex dispersion, float64 otherwise).
        """
        g = self._giwk.mat.reshape(-1, *self._giwk.mat.shape[-3:])
        core = slice(self._giwk.niv - config.box.niv_core, self._giwk.niv + config.box.niv_core)
        out = np.empty(g.shape[:3], dtype=np.complex128 if self._complex_ek else np.float64)
        for start, stop in mpi_utils.row_chunks(g.shape[0], g.itemsize, g[0].size, memory_estimator.SLICE_CHUNK_BYTES):
            change = _sandwich(g[start:stop, ..., core], dx[start:stop])
            out[start:stop] = np.sum(change if self._complex_ek else change.real, axis=-1)
        return out / self._beta

    def _build_dg(self, dx: np.ndarray, dmu: float) -> None:
        r"""
        Writes :math:`\delta G^{\mathrm{k}} = G^{\mathrm{k}}(\theta_{\mathrm{core}}\,\delta x^{\mathrm{k}} -
        \delta\mu)\,G^{\mathrm{k}}` of the loop box into the node's window, in momentum chunks (node roots only).

        :param dx: The window change ``[k, o1, o2, 2 niv_core]``.
        :param dmu: The chemical-potential change.
        :return: None.
        """
        g = self._giwk.mat.reshape(-1, *self._giwk.mat.shape[-3:])
        core = slice(self._giwk.niv - config.box.niv_core, self._giwk.niv + config.box.niv_core)
        dg = self._dg.reshape(g.shape)
        for start, stop in mpi_utils.row_chunks(g.shape[0], g.itemsize, g[0].size, memory_estimator.SLICE_CHUNK_BYTES):
            rows = slice(start, stop)
            dg[rows] = _sandwich(g[rows])
            dg[rows] *= -dmu
            dg[rows, ..., core] += _sandwich(g[rows, ..., core], dx[rows])

    @staticmethod
    def _response_rhs(
        gchi0_inv: FourPoint, dcores: list[FourPoint]
    ) -> Callable[[int, int, FourPoint], list[FourPoint]]:
        r"""
        Returns the per-slice right-hand sides of the columns' Bethe-Salpeter systems for
        :func:`~dgamore.nonlocal_sde.create_auxiliary_chi_r_q_sum`: from the summed auxiliary susceptibility of the
        slice ``(i, w)``, that slice's three-leg vertex :math:`\gamma^{\mathrm{q}\nu}_{r}` (see
        :func:`~dgamore.nonlocal_sde.create_vrg_r_q`) and every column's :math:`t^{\mathrm{q}\nu}_{r} =
        (\chi^{\mathrm{q}\nu}_{0})^{-1}\,\delta\chi^{\mathrm{q}\nu}_{0}\,\gamma^{\mathrm{q}\nu}_{r}`. Momentum and
        bosonic frequency are batch indices of these products, so a slice's right-hand sides have the bits of its
        slice of the products on the whole window.

        :param gchi0_inv: The inverse core bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` on the window.
        :param dcores: The core-box bubble response :math:`\delta\chi^{\mathrm{q}\nu}_{0}` of every column on the
            window.
        :return: The callable ``(i, w, total) -> list[FourPoint]``, one right-hand side per column.
        """

        def slice_rhs(i: int, w: int, total: FourPoint) -> list[FourPoint]:
            """The right-hand sides of slice ``(i, w)`` from its summed auxiliary susceptibility ``total``."""

            def take(obj: FourPoint) -> FourPoint:
                """A view of the slice of a window object."""
                return obj.take_q_index_slice(i, i + 1, copy=False).take_wn_slice(w, w + 1, copy=False)

            g_inv = take(gchi0_inv)
            vrg = nonlocal_sde.create_vrg_r_q(total, g_inv)
            return [g_inv @ (take(dcore) @ vrg) for dcore in dcores]

        return slice_rhs

    def _kernel_responses(
        self, parts: list[tuple[FourPoint, FourPoint, FourPoint, FourPoint]], w_window: tuple[int, int]
    ) -> list[FourPoint]:
        r"""
        Returns the kernel responses :math:`\delta K^{\mathrm{q}\nu} = \delta K^{\mathrm{q}\nu}_{\mathrm{dc}} +
        \delta K^{\mathrm{q}\nu}_{\mathrm{d}} + 3\,\delta K^{\mathrm{q}\nu}_{\mathrm{m}}` of a piece's columns on one
        window of stored bosonic frequencies, in the contraction layout (collective over the node when the local
        vertices are loaded per window, see :meth:`_local_vertex`); the parts are freed. Per channel one
        Bethe-Salpeter pass solves the right-hand sides of every column against one factorization per slice. Without
        held three-leg vertices (:attr:`~dgamore.memory_estimator.ChunkBudgets.exact_vrg_per_window`) the same
        factorization solves the auxiliary sum first, whose three-leg vertex each slice builds its right-hand sides
        from (see :meth:`_response_rhs`), on compound slices of at least ``_FUSED_SOLVE_MIN_COMPOUND`` entries
        (:math:`B^2` times the core fermionic frequencies); below that, where the per-slice right-hand sides cost more
        than a second factorization, the sum and the right-hand sides are solved in two passes, with the same bits.

        :param parts: Per column the double-counting kernel response and the sums over the full and the core box and
            the core-box bubble of the bubble response on the window (see :meth:`_split_bubble`).
        :param w_window: The window ``(start, stop)`` of stored bosonic indices (half niw range).
        :return: The kernel responses on this rank's irreducible q-points and the window's frequencies, one per column.
        """
        beta, (w0, w1) = self._beta, w_window
        dkernels = [part[0].compress_q_dimension().to_half_niw_range() for part in parts]
        dcores = [dcore for *_, dcore in parts]
        gchi0_inv = self._gchi0_inv.take_wn_slice(w0, w1, copy=False)
        fused = config.sys.n_bands**2 * 2 * config.box.niv_core >= _FUSED_SOLVE_MIN_COMPOUND
        for weight, channel, u_r, inactive in self._channels:
            chi = self._chi[channel].take_wn_slice(w0, w1, copy=False)
            with self._local_vertex(f"gamma_{channel.value}_loc.npy", channel, w_window=w_window) as gamma:
                if channel in self._vrg or not fused:
                    if channel in self._vrg:
                        vrg = self._vrg[channel].take_wn_slice(w0, w1, copy=False)
                    else:
                        aux = nonlocal_sde.create_auxiliary_chi_r_q_sum(
                            gamma, gchi0_inv, self._u_loc, inactive=inactive
                        )
                        vrg = nonlocal_sde.create_vrg_r_q(aux, gchi0_inv, copy=False)
                    rhs = [gchi0_inv @ (dcore @ vrg) for dcore in dcores]
                    solved = nonlocal_sde.create_auxiliary_chi_r_q_sum(
                        gamma, gchi0_inv, self._u_loc, rhs=rhs, inactive=inactive
                    )
                else:
                    hook = self._response_rhs(gchi0_inv, dcores)
                    aux, solved = nonlocal_sde.create_auxiliary_chi_r_q_sum(
                        gamma, gchi0_inv, self._u_loc, inactive=inactive, slice_rhs=hook
                    )
                    vrg = nonlocal_sde.create_vrg_r_q(aux, gchi0_inv, copy=False)
                    # the right-hand sides the slices built, bit for bit
                    rhs = [gchi0_inv @ (dcore @ vrg) for dcore in dcores]
            for dkernel, (_, dfull_sum, dcore_sum, _), t, s in zip(dkernels, parts, rhs, solved):
                dvrg = (gchi0_inv @ s).sub(t, copy=False)
                t.free()
                dsum = s.scale(1.0 / beta).sum_over_all_vn(beta) + dfull_sum - dcore_sum
                s.free()
                dchi = dsum - chi.matmul(u_r, per_wn=True) @ dsum
                dchi = dchi - dchi.matmul(u_r, per_wn=True) @ chi
                dk = dvrg - dvrg.matmul(u_r, per_wn=True) @ chi - vrg.matmul(u_r, per_wn=True) @ dchi
                term = dk.matmul(u_r, per_wn=True).permute_orbitals("abcd->badc", copy=False).scale(weight)
                dkernel.add(term, copy=False)
            if channel not in self._vrg:
                vrg.free()
        for _, dfull_sum, dcore_sum, dcore in parts:
            dcore.free()
            dfull_sum.free()
            dcore_sum.free()
        return dkernels

    def _apply_change(self, y: np.ndarray) -> tuple[float, np.ndarray | None, np.ndarray | None]:
        r"""
        Writes the change :math:`\delta G` of a sector vector into the node's window (collective): the node roots
        unfold it, rank 0 evaluates the chemical-potential and occupation changes, and the node roots build
        :math:`\delta G^{\mathrm{k}}` (see :meth:`_build_dg`).

        :param y: The real sector vector on the node roots, ``None`` elsewhere.
        :return: The chemical-potential change (every rank) and the local and momentum-resolved occupation changes
            (rank 0; ``None`` elsewhere).
        """
        comm, node_root = self._comm, self._node_comm.rank == 0
        dx = self._from_sector(y) if node_root else None
        dmu, docc, docc_k = None, None, None
        if comm.rank == 0:
            dn_x = self._occupation_change(dx)
            dmu = -2.0 * np.trace(dn_x.mean(axis=0)).real / self._dn_dmu
            docc_k = dn_x.reshape(self._kappa.shape) + self._kappa * dmu
            docc = docc_k.mean(axis=(0, 1, 2))
        dmu = comm.bcast(dmu, root=0)
        if node_root:
            self._build_dg(dx, dmu)
        self._node_comm.Barrier()
        return dmu, docc, docc_k

    def _finalize(
        self, dsigma: SelfEnergy | None, docc: np.ndarray | None, docc_k: np.ndarray | None
    ) -> np.ndarray | None:
        r"""
        Returns the product of a column from its contraction (rank 0): the transform back to momentum space, the
        Hartree-Fock term of the occupation changes and the reduction to the sector, in place.

        :param dsigma: The column's contraction of :math:`(K, \delta G)` and :math:`(\delta K, G)` in real space (see
            :meth:`~dgamore.nonlocal_sde._ColumnSdeStream.finish`; rank 0).
        :param docc: The local occupation change (rank 0).
        :param docc_k: The momentum-resolved occupation change (rank 0).
        :return: The product as a real sector vector on rank 0; ``None`` elsewhere.
        """
        if self._comm.rank != 0:
            return None
        dsigma.ifft(copy=False).to_full_niv_range()
        hartree, fock = nonlocal_sde.get_hartree_fock(self._u_loc, self._v_nonloc_full, docc, docc_k, self._w_r)
        dsigma.add(hartree, copy=False).add(fock, copy=False)
        if self._tr_average:
            dsigma.compress_q_dimension().symmetrize_time_reversal()
        return self.project(self._to_sector(dsigma.mat))

    def matvec(self, y: np.ndarray | None) -> np.ndarray | None:
        r"""
        Returns :math:`J y` (collective over the communicator of the constructor), the one-column block of
        :meth:`matvec_block`: rank 0 passes the sector vector, every other rank ``None``.

        :param y: The real sector vector on rank 0.
        :return: :math:`J y` as a real sector vector on rank 0; ``None`` elsewhere.
        """
        ys = self.matvec_block(None if y is None else np.asarray(y, dtype=np.float64)[:, None])
        return None if ys is None else ys[:, 0]

    def matvec_block(self, ys: np.ndarray | None) -> np.ndarray | None:
        r"""
        Returns :math:`J Y` for a block of sector vectors (collective over the communicator of the constructor): rank 0
        passes the vectors as the columns of ``ys``, every other rank ``None``. The columns run in pieces of at most the
        block width the chunk budgets give (one column without them), each piece costing one Bethe-Salpeter pass per
        channel whatever its width (see :meth:`_piece_products`).

        :param ys: The real sector vectors as columns on rank 0.
        :return: :math:`J Y` as real sector vectors in columns on rank 0; ``None`` elsewhere.
        """
        comm = self._comm
        with deferred_collection():
            projected = None
            if comm.rank == 0:
                projected = np.column_stack([self.project(y) for y in np.asarray(ys, dtype=np.float64).T])
            width = comm.bcast(None if projected is None else projected.shape[1], root=0)
            columns = [None] * width
            if self._node_comm.rank == 0:
                columns = list(mpi_utils.bcast_rows(self._roots_comm, projected, root=0).T)
            products = []
            for start in range(0, width, self._block):
                products += self._piece_products(columns[start : start + self._block])
            return np.column_stack(products) if comm.rank == 0 else None

    def _piece_products(self, columns: list) -> list:
        r"""
        Returns the products of one piece of a block (collective, see :meth:`matvec_block`): the change of every
        column and its real-space window, then per window of stored bosonic frequencies the bubble responses, one
        Bethe-Salpeter pass per channel for the piece and the column contraction of the window, and finally per column
        the reduction of its contraction and its finish on rank 0.

        :param columns: The piece's real sector vectors on the node roots, ``None`` per column elsewhere.
        :return: The products as real sector vectors on rank 0, ``None`` per column elsewhere.
        """
        changes, dg_rs, wins = [], [], []
        for y in columns:
            changes.append(self._apply_change(y))
            dg_r, win = nonlocal_sde._build_rspace_giwk_window(
                self._greens_function(self._dg), self._node_comm, niv_window=self._niv_r
            )
            dg_rs.append(dg_r)
            wins.append(win)
        dg_r = win = None
        stream = nonlocal_sde._ColumnSdeStream(
            self._dist, len(columns), self._g_r.dtype, self._sde_chunk, by_frequency=True
        )
        terms = [(0, j, dg_r, self._niv_r) for j, dg_r in enumerate(dg_rs)]
        terms += [(1 + j, j, self._g_r, self._niv_r) for j in range(len(columns))]
        for w0, w1 in self._w_windows:
            responses = self._bubbles((w0, w1), dg_rs)
            with self._local_vertex(*_F_DC, w_window=(w0, w1)) as f_dc:
                parts = [
                    self._split_bubble(dchi0, nonlocal_sde._contract_dc_vertex(f_dc, dchi0)) for dchi0 in responses
                ]
            responses = None
            dkernels = self._kernel_responses(parts, (w0, w1))
            parts = None
            stream.add_group((w0, w1), [self._kernel.take_wn_slice(w0, w1, copy=False), *dkernels], terms)
            for dkernel in dkernels:
                dkernel.free()
        products = [self._finalize(stream.finish(j), *change[1:]) for j, change in enumerate(changes)]
        dg_rs = terms = None
        for win in wins:
            nonlocal_sde._free_shared_window(win, self._node_comm)
        return products

    def free(self) -> None:
        """
        Releases the held quantities, the shared windows and the node and node-root communicators (collective).

        :return: None.
        """
        node_comm = self._node_comm
        for obj in (self._kernel, self._gchi0_inv, *self._chi.values(), *self._vrg.values()):
            obj.free()
        self._chi, self._vrg, self._channels = {}, {}, []
        self._w_r = None
        for vertex, win in self._vertices.values():
            vertex.mat = None
            nonlocal_sde._free_shared_window(win, node_comm)
        self._vertices = {}
        self._g_r = self._dg = None
        for win in (self._g_r_win, self._dg_win):
            nonlocal_sde._free_shared_window(win, node_comm)
        if self._roots_comm != MPI.COMM_NULL:
            self._roots_comm.Free()
        self._giwk.mat = None
        nonlocal_sde._release_shared_giwk(self._giwk_win, node_comm)


def _merge_modes(theta: np.ndarray, vectors: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Returns the distinct eigenpairs among the ones ARPACK returned, with the conjugate partner of every complex
    eigenvalue (the operator is real), sorted by decreasing real part of :math:`\theta`; every complex pair is first
    taken to its member with a positive imaginary part (``vectors`` is conjugated in place). A pair repeats the kept
    ones when its value lies within ``100 tol max(1, |theta|)`` of theirs and the part of its eigenvector outside
    their span is at most :math:`\sqrt{1 - o^2}` of its norm, :math:`o` = ``_DUPLICATE_OVERLAP``, so a degenerate
    eigenspace keeps one vector per dimension.

    :param theta: The eigenvalues ARPACK returned, all targets concatenated.
    :param vectors: Their eigenvectors as columns (overwritten).
    :param tol: The ARPACK tolerance.
    :return: The distinct eigenpairs, values and eigenvectors.
    """
    complex_ = np.abs(theta.imag) > tol * np.maximum(1.0, np.abs(theta))
    lower = complex_ & (theta.imag < 0)
    theta = np.where(lower, theta.conj(), theta)
    for i in np.flatnonzero(lower):
        np.conjugate(vectors[:, i], out=vectors[:, i])
    keep = []
    for i, value in enumerate(theta):
        close = [j for j in keep if abs(value - theta[j]) <= 100 * tol * max(1.0, abs(value))]
        if close:
            gram = np.array([[np.vdot(vectors[:, a], vectors[:, b]) for b in close] for a in close])
            overlap = np.array([np.vdot(vectors[:, a], vectors[:, i]) for a in close])
            captured = np.vdot(overlap, np.linalg.lstsq(gram, overlap, rcond=None)[0]).real
            if captured > _DUPLICATE_OVERLAP**2 * np.vdot(vectors[:, i], vectors[:, i]).real:
                continue
        keep.append(i)
    sources = [(i, False) for i in keep] + [(i, True) for i in keep if complex_[i]]
    values = np.array([theta[i].conj() if conj else theta[i] for i, conj in sources], dtype=np.complex128)
    order = np.argsort(-values.real, kind="stable")
    out = np.empty((vectors.shape[0], order.size), dtype=vectors.dtype, order="F")
    for column, (i, conj) in zip(out.T, (sources[k] for k in order)):
        if conj:
            np.conjugate(vectors[:, i], out=column)
        else:
            column[:] = vectors[:, i]
    return values[order], out


def leading_eigenpairs(
    jac: ExactJacobian, comm: MPI.Comm, lr_start: np.ndarray | None = None, lm_start: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    r"""
    Computes leading eigenpairs of the Jacobian with ARPACK (``scipy.sparse.linalg.eigs``) on rank 0, while every
    other rank serves the collective products of :meth:`ExactJacobian.matvec` until rank 0 signals the end. Two
    targets of :data:`~dgamore.memory_estimator.EXACT_JACOBIAN_MODES` pairs each are searched, the largest real parts
    of the eigenvalues :math:`\theta` of :math:`J` (the least stable :math:`\lambda_\Pi = 1 - \theta`) and the largest
    moduli (the stiff modes). A target that does not converge within about ``_MAX_MATVECS`` products keeps its
    converged pairs, and a largest-real-part target whose pairs are all unstable is flagged, both with a warning. The
    residual bound of a pair is :math:`\max(100\,\epsilon, 10^{-10})\max(1, |\theta|)`, :math:`\epsilon` of the
    storage precision. Both targets start from the projection of a fixed random vector plus the given directions, so
    the second one repeats the first one's first ``ncv + 1`` vectors, whose kept products it takes without a product.

    :param jac: The operator (``n_real``, ``matvec``, ``project``).
    :param comm: The MPI communicator.
    :param lr_start: A nonzero real direction in the operator's sector coordinates on rank 0, folded into the start
        vector both targets share, e.g. the least-stable certified direction, or ``None``.
    :param lm_start: A nonzero real direction folded into the same shared start vector, e.g. the stiffest certified
        direction, or ``None``.
    :return: On rank 0 the tuple ``(lam_pi, res, u)`` sorted by the real part of :math:`\lambda_\Pi`, ``u`` the
        normalized complex eigenvectors as columns in the operator's sector coordinates; ``None`` elsewhere.
    """
    if comm.rank != 0:
        while comm.bcast(None, root=0):
            jac.matvec(None)
        return None

    logger = config.logger
    n = jac.n_real
    ncv = min(memory_estimator.EXACT_JACOBIAN_NCV, n)
    k = min(memory_estimator.EXACT_JACOBIAN_MODES, ncv - 2)
    tol = max(100 * np.finfo(n_point_base.DTYPE).eps, 1e-10)
    count, products = [0], {}

    def apply(y: np.ndarray) -> np.ndarray:
        """Returns a kept product, which leaves the store, or signals the serving ranks and returns the product."""
        key = hashlib.blake2b(np.ascontiguousarray(y)).digest()
        if key in products:
            return products.pop(key)
        comm.bcast(True, root=0)
        count[0] += 1
        product = jac.matvec(y)
        if which == "LR" and len(products) <= ncv:  # only the first target's products are repeated
            products[key] = product.copy()
        return product

    operator = spla.LinearOperator((n, n), matvec=apply, dtype=np.float64)
    v0 = np.random.default_rng(0).standard_normal(n)
    v0_norm = np.linalg.norm(v0)
    for direction in (lr_start, lm_start):
        if direction is not None:
            v0 += v0_norm / np.linalg.norm(direction) * direction
    v0 = jac.project(v0)
    found_theta, found_vectors = [], []
    for which in ("LR", "LM"):
        start = count[0]
        try:
            theta, vectors = spla.eigs(
                operator, k=k, which=which, v0=v0, ncv=ncv, tol=tol, maxiter=(_MAX_MATVECS - ncv) // (ncv - k) + 1
            )
        except spla.ArpackNoConvergence as err:
            theta, vectors = err.eigenvalues, err.eigenvectors
            logger.warning(
                f"Exact Jacobian: ARPACK ({which}) converged {theta.size} of {k} eigenpairs within "
                f"{count[0] - start} products."
            )
        bound = np.maximum(1e-2, tol * np.maximum(1.0, np.abs(theta)))
        if which == "LR" and theta.size and np.all((1.0 - theta).real < -bound):
            logger.warning(
                f"Exact Jacobian: all {theta.size} eigenpairs of the largest real part are unstable (Re lambda_Pi "
                f"below -max(0.01, res)); the Jacobian may have more unstable directions than these."
            )
        found_theta.append(theta)
        found_vectors.append(vectors)
    products.clear()
    # released on the normal path only: an exception aborts the job instead of meeting another collective
    comm.bcast(False, root=0)

    theta = np.concatenate(found_theta)
    vectors = np.empty((n, theta.size), dtype=np.complex128, order="F")
    np.concatenate(found_vectors, axis=1, out=vectors)
    found_vectors.clear()
    theta, vectors = _merge_modes(theta, vectors, tol)
    # a restart can leave the projected sector, where the operator vanishes; such a zero mode is no loop direction
    inside = [
        np.linalg.norm(jac.project(v.real) + 1j * jac.project(v.imag) - v) <= 0.5 * np.linalg.norm(v) for v in vectors.T
    ]
    if not all(inside):
        theta, vectors = theta[inside], vectors[:, inside]
    for column in vectors.T:
        column /= np.linalg.norm(column)
    lam_pi = 1.0 - theta
    logger.info(
        f"Exact Jacobian: {lam_pi.size} eigenpairs from {count[0]} products, lambda_Pi "
        f"{', '.join(f'{lam.real:+.4f}{lam.imag:+.4f}j' for lam in lam_pi)}."
    )
    return lam_pi, tol * np.maximum(1.0, np.abs(theta)), vectors


def _orthonormal_extension(basis: np.ndarray, size: int, block: np.ndarray, norms: list[float] | None = None) -> int:
    """
    Writes the part of every column of ``block`` orthogonal to the first ``size`` columns of ``basis`` and to the
    columns written before into the next free column of ``basis``, normalized and orthogonalized twice for stability,
    until ``basis`` is full; a column whose remainder is below ``1e-8`` of its reference norm adds nothing.

    :param basis: The preallocated basis, shape ``[n, k_max]``, whose first ``size`` columns are orthonormal (written
        in place).
    :param size: Number of orthonormal columns ``basis`` holds.
    :param block: The columns to append, shape ``[n, m]``.
    :param norms: The reference norm of every column, e.g. its norm before a projection; ``None`` takes the norms of
        the columns of ``block``.
    :return: The number of orthonormal columns ``basis`` holds afterwards.
    """
    norms = [np.linalg.norm(column) for column in block.T] if norms is None else norms
    for column, reference in zip(block.T, norms):
        if size == basis.shape[1]:
            break
        remainder, kept = basis[:, size], basis[:, :size]
        remainder[:] = column
        for _ in range(2):
            remainder -= kept @ (kept.T @ remainder)
        norm = np.linalg.norm(remainder)
        if norm > 1e-8 * reference:
            remainder /= norm
            size += 1
    return size


def _ritz_residual(
    basis: np.ndarray, images: np.ndarray, y: np.ndarray, theta: complex
) -> tuple[np.ndarray, np.ndarray, float]:
    r"""
    Returns the residual :math:`W y - \theta\, Q y` of a Ritz pair from real products only, :math:`Q y = Q\,
    \mathrm{Re}\,y + \imath\, Q\,\mathrm{Im}\,y` and likewise for :math:`W y`, with the norm of :math:`Q y`.

    :param basis: The orthonormal basis :math:`Q`, shape ``[n, m]``.
    :param images: The products :math:`W = J Q`, shape ``[n, m]``.
    :param y: The Ritz vector's coordinates in the basis, length ``m``.
    :param theta: The Ritz value.
    :return: The real and imaginary part of the residual and :math:`\lVert Q y\rVert`.
    """
    q_re, q_im = basis @ y.real, basis @ y.imag
    r_re, r_im = images @ y.real, images @ y.imag
    r_re -= theta.real * q_re
    r_re += theta.imag * q_im
    r_im -= theta.real * q_im
    r_im -= theta.imag * q_re
    return r_re, r_im, float(np.hypot(np.linalg.norm(q_re), np.linalg.norm(q_im)))


def certify_subspace(
    jac: ExactJacobian, start: np.ndarray | None, comm: MPI.Comm
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray] | None:
    r"""
    Certifies the eigenpairs of the Jacobian next to a few given directions with a short block Arnoldi on rank 0,
    while every other rank serves the collective block products of :meth:`ExactJacobian.matvec_block` until rank 0
    signals the end. The start columns are projected onto the sector and orthonormalized into the basis :math:`Q` (a
    column whose projection is below ``1e-8`` of its own norm adds nothing); after every block product
    :math:`W = J Q` the Rayleigh-Ritz of :math:`B = Q^{T} W` gives the Ritz pairs :math:`(\theta, y)` and their exact
    residuals

    .. math:: \mathrm{res} = \frac{\lVert W y - \theta\, Q y \rVert}{\lVert Q y \rVert}.

    The search stops once the pair with the largest real part of :math:`\theta` passes the tracker's Ritz gate,
    :math:`\mathrm{res} \leq g \max(1, |\theta|)` with :math:`g` its ``_RITZ_GATE``, or once the basis holds
    ``_CHECK_BASIS`` vectors; until then the real and the imaginary part of that pair's residual extend the basis.

    :param jac: The operator (``n_real``, ``matvec_block``, ``project``).
    :param start: The real start columns in the operator's sector coordinates, shape ``[n_real, m]``, on rank 0;
        ``None`` elsewhere.
    :param comm: The MPI communicator.
    :return: On rank 0 the tuple ``(lam_pi, res, u, n_products, kappa)`` of every Ritz pair of the final basis,
        sorted by the real part of :math:`\lambda_\Pi`, ``u`` the normalized complex eigenvectors as columns in the
        operator's sector coordinates and ``kappa`` the condition numbers of the values in :math:`B`, which turn the
        residuals into error bounds; ``None`` elsewhere.
    """
    if comm.rank != 0:
        while comm.bcast(None, root=0):
            jac.matvec_block(None)
        return None

    n = jac.n_real
    basis, images = np.empty((n, _CHECK_BASIS), order="F"), np.empty((n, _CHECK_BASIS), order="F")
    theta, y, res = np.zeros(0, dtype=np.complex128), np.zeros((0, 0), dtype=np.complex128), np.zeros(0)
    start = np.asarray(start)
    norms = [np.linalg.norm(column) for column in start.T]
    size = _orthonormal_extension(basis, 0, np.column_stack([jac.project(column) for column in start.T]), norms)
    done = 0
    while size > done:
        comm.bcast(True, root=0)
        images[:, done:size] = jac.matvec_block(basis[:, done:size])
        done = size
        q, w = basis[:, :done], images[:, :done]
        theta, y, _ = _schur_ritz(q.T @ w)
        res = np.empty(done)
        for j in range(done):
            r_re, r_im, ritz_norm = _ritz_residual(q, w, y[:, j], theta[j])
            res[j] = np.hypot(np.linalg.norm(r_re), np.linalg.norm(r_im)) / ritz_norm
        lead = int(np.argmax(theta.real))
        if res[lead] <= _RITZ_GATE * max(1.0, abs(theta[lead])):
            break
        size = _orthonormal_extension(basis, done, np.column_stack(_ritz_residual(q, w, y[:, lead], theta[lead])[:2]))
    # released on the normal path only: an exception aborts the job instead of meeting another collective
    comm.bcast(False, root=0)

    lam_pi = 1.0 - theta
    order = np.argsort(lam_pi.real)
    lam_pi, res, kappa = lam_pi[order], res[order], _condition_numbers(y)[order]
    u = np.empty((n, order.size), dtype=np.complex128, order="F")
    for column, j in zip(u.T, order):
        column.real = basis[:, :done] @ y[:, j].real
        column.imag = basis[:, :done] @ y[:, j].imag
        column /= np.linalg.norm(column)
    config.logger.info(
        f"Exact check: {done} products, lambda_Pi "
        f"{', '.join(f'{lam.real:+.4f}{lam.imag:+.4f}j (res {r:.1e})' for lam, r in zip(lam_pi, res))}."
    )
    return lam_pi, res, u, done, kappa

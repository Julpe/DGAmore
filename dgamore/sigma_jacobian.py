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
:math:`J\,\delta x` analytically at one self-energy: the loop's own functions run on the linearized objects (the
bubble through its polarization identity, the Bethe-Salpeter equation with the response as its right-hand side, the
Schwinger-Dyson contraction once with each factor linearized), so every half-range reconstruction of the map is
linearized with it and the product is the derivative of the code's map, not of an idealized one.
:func:`leading_eigenpairs` computes the leading eigenpairs of :math:`J` with ARPACK, the converged-point Jacobian of
Gievers et al., arXiv:2609.11405, App. G4, which the Jacobian tracker carries to a successor run (see
:meth:`~dgamore.jacobian_stabilization.JacobianTracker.install_exact_spectrum`), and :func:`certify_subspace`
certifies the eigenpairs next to a few given directions with a short block Arnoldi, the exact check the tracker runs
inside the loop (see :meth:`~dgamore.jacobian_stabilization.JacobianTracker.install_exact_check`).

The Jacobian is evaluated at the pure map: no lambda correction, susceptibility restriction or annealing mass, and
the momentum-dependent :math:`V^{\mathbf{q}}` Hartree-Fock offset of the shell held at its value (exact for
:math:`V = 0`). It acts on the self-energies the loop can reach (:func:`sector_projector`): with several orbitals the
symmetries that fix a momentum constrain its orbital matrix, and orbital pairs the converged self-energy leaves exactly
zero are never populated by the iteration, so neither symmetry-breaking nor unpopulated directions enter the spectrum.
"""

import contextlib
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
from dgamore.jacobian_stabilization import _RITZ_GATE, _schur_ritz, to_mat, to_vec
from dgamore.matsubara_frequencies import MFHelper
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel, deferred_collection
from dgamore.self_energy import SelfEnergy

_CHANNELS = ((SpinChannel.DENS, 1.0), (SpinChannel.MAGN, 3.0))  # ladder channels and their weights in the kernel
_F_DC = ("f_dc_loc.npy", SpinChannel.MAGN, (4, 0, 1, 5, 2, 3, 6))  # the dc vertex, stored in its contraction order
_MAX_MATVECS = 150  # rough bound on the Jacobian-vector products of one ARPACK target
_CHECK_BASIS = memory_estimator.EXACT_CHECK_BASIS  # basis vectors, i.e. products, of one exact check
_DUPLICATE_OVERLAP = 0.9  # |<u, u'>| of normalized eigenvectors above which two pairs of one value are one mode


def _squared_box_sum(sigma_mat: np.ndarray, mu: float, ek: np.ndarray, beta: float) -> np.ndarray:
    r"""
    Returns :math:`\sum_{\nu}(G^{\mathrm{k}})^2` per momentum over the frequency box of ``sigma_mat``,
    :math:`G^{\mathrm{k}} = [(\imath\nu + \mu) - \varepsilon(\mathbf{k}) - \Sigma^{\mathrm{k}}]^{-1}`, i.e. minus
    the derivative of the box sum of the occupation with respect to :math:`\mu`; its real part for a real
    dispersion, as the occupation of :meth:`GreensFunction.get_fill_nonlocal_from_sigma` takes it. The Dyson equation
    is solved in double precision in momentum chunks, as that method does.

    :param sigma_mat: The self-energy ``[k, o1, o2, 2 niv]``; a single momentum row broadcasts over ``ek``.
    :param mu: Chemical potential :math:`\mu`.
    :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})`, shape ``[k, o1, o2]``.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The sums ``[k, o1, o2]`` (complex128 for a complex dispersion, float64 otherwise).
    """
    nk, nb = ek.shape[0], ek.shape[-1]
    iv = 1j * MFHelper.vn(sigma_mat.shape[-1] // 2, beta)
    static = (iv[:, None, None] + mu) * np.eye(nb)
    complex_ek = np.iscomplexobj(np.real_if_close(ek))
    out = np.empty((nk, nb, nb), dtype=np.complex128 if complex_ek else np.float64)
    step = max(1, _MODEL_EPOT_CHUNK_ELEMENTS // (nb * nb * iv.size))
    for start in range(0, nk, step):
        stop = min(nk, start + step)
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
    stabilizers = (
        symmetry_reduction.stabilizer_unitaries(k_grid._auto_group, tuple(k_grid.nk), k_grid.irrk_ind)
        if k_grid.is_auto
        else []
    )
    counts = np.zeros(k_grid.nk_irr)
    for positions, _ in stabilizers:
        counts[positions] += 1.0
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
    the windows the loop iterates, unfolded from their irreducible momenta with the lattice symmetry (the orbital
    unitaries of an auto-discovered group included) and completed by the Matsubara Hermiticity, the weighted sector is
    an isometry: every norm and inner product the tracker forms is the one of the whole window, on a vector shorter by
    :math:`2 n_{\mathbf{k}} / n_{\mathbf{k}}^{\mathrm{irr}}`. A window is read at its irreducible momenta only, so no
    real copy of a whole window is made.

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

    The operator acts on the sector the iteration lives in: the self-energy on the irreducible momenta and the
    positive core frequencies, unfolded with the lattice symmetry (:meth:`SelfEnergy.map_to_full_bz`) and completed
    by its Matsubara Hermiticity (:meth:`SelfEnergy.to_full_niv_range`), and restricted by :meth:`project` to the
    self-energies the loop can reach: every product projects its input and its output (:func:`sector_projector`), so
    the eigenvalues of the sector operator are those of the map on those self-energies, and a direction that breaks
    the symmetry of a momentum's little group or populates an orbital pair the converged self-energy leaves zero is
    annihilated. A sector vector holds the weighted sector coordinates of the Jacobian tracker
    (:func:`tracker_sector_maps`), so the bases and residuals of the eigen-solvers are measured in the norm of the
    whole core window and their vectors pass to the tracker as they are.

    One product :math:`J\,\delta x`, the proposal's stages linearized:

    1. rank 0: :math:`\delta\mu = -\partial_x N\cdot\delta x/\partial_\mu N` for the filling ``update_mu`` holds,
       and :math:`\delta n^{\mathbf{k}} = \frac{1}{\beta}\sum_{\nu}G^{\mathrm{k}}\,\delta x^{\mathrm{k}}
       \,G^{\mathrm{k}} + \kappa^{\mathbf{k}}\,\delta\mu` (its real part for a real dispersion, as the occupation);
    2. node roots: :math:`\delta G^{\mathrm{k}} = G^{\mathrm{k}}(\theta_{\mathrm{core}}\,\delta x^{\mathrm{k}} -
       \delta\mu)\,G^{\mathrm{k}}` on the loop box;
    3. :math:`\delta\chi^{\mathrm{q}\nu}_{0} = (\chi^{\mathrm{q}\nu}_{0}[G + s\,\delta G] - \chi^{\mathrm{q}\nu}_{0}[G -
       s\,\delta G])/(2s)` with :math:`s = \lVert G\rVert/\lVert\delta G\rVert`, exact since the bubble is quadratic
       in :math:`G`;
    4. per channel the Bethe-Salpeter system of :func:`~dgamore.nonlocal_sde.create_auxiliary_chi_r_q_sum` solved for
       :math:`t^{\mathrm{q}\nu}_{r} = (\chi^{\mathrm{q}\nu}_{0})^{-1}\,\delta\chi^{\mathrm{q}\nu}_{0}\,
       \gamma^{\mathrm{q}\nu}_{r}`, giving :math:`s^{\mathrm{q}\nu}_{r}`,
       :math:`\delta\gamma^{\mathrm{q}\nu}_{r} = (\chi^{\mathrm{q}\nu}_{0})^{-1}s^{\mathrm{q}\nu}_{r} -
       t^{\mathrm{q}\nu}_{r}` and the change :math:`\delta\hat\chi^{*\mathrm{q}}_{r}` of the shell-corrected sum,
       then :math:`\delta\chi^{\mathrm{q}}_{r} = (1 - \chi^{\mathrm{q}}_{r}\mathcal{U}^{\mathbf{q}}_{r})\,
       \delta\hat\chi^{*\mathrm{q}}_{r}\,(1 - \mathcal{U}^{\mathbf{q}}_{r}\chi^{\mathrm{q}}_{r})` and
       :math:`\delta K^{\mathrm{q}\nu}_{r} = (\delta\gamma^{\mathrm{q}\nu}_{r} - \delta\gamma^{\mathrm{q}\nu}_{r}
       \mathcal{U}^{\mathbf{q}}_{r}\chi^{\mathrm{q}}_{r} - \gamma^{\mathrm{q}\nu}_{r}\mathcal{U}^{\mathbf{q}}_{r}
       \delta\chi^{\mathrm{q}}_{r})\,\mathcal{U}^{\mathbf{q}}_{r}`, beside the double-counting kernel of
       :math:`\delta\chi^{\mathrm{q}\nu}_{0}`;
    5. the contraction of :func:`~dgamore.nonlocal_sde._run_column_sde` once with :math:`(K, \delta G)` and once with
       :math:`(\delta K, G)`, and on rank 0 the Hartree-Fock term of :math:`\delta n^{\mathbf{k}}`;
    6. rank 0, for several orbitals on a lattice with real hoppings: the time-reversal average of the proposal
       (:meth:`~dgamore.self_energy.SelfEnergy.symmetrize_time_reversal`), which is linear.

    Held between products: the Green's function of the loop box and its real-space window (once per node), the
    local vertices (once per node, unless the driver has them loaded for the phase that reads them, see
    :meth:`_local_vertex`), the inverse core bubble, the three-leg vertices and dressed susceptibilities of
    both channels and the total kernel on this rank's momenta, and on rank 0 the filling derivatives.
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
        Builds the forward quantities at the self-energy ``sigma`` (collective over ``comm``).

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
        :param chunk_budgets: Chunk byte budgets of the auxiliary-susceptibility build and the self-energy
            contraction and the block width of the products (sized by the driver from the memory estimate); ``None``
            gives both builds the job-wide fair-share budget of :func:`~dgamore.nonlocal_sde._sde_chunk_budget` and
            the products one column at a time.
        """
        box, k_grid, nb = config.box, config.lattice.k_grid, config.sys.n_bands
        beta = float(config.sys.beta)
        self._comm, self._dist = comm, mpi_dist_irrk
        self._u_loc, self._v_nonloc_full = u_loc, v_nonloc_full
        self._beta, self._mu = beta, float(mu)
        self._ek = config.lattice.hamiltonian.get_ek()
        # the occupation of a complex dispersion and the proposal's time-reversal average, as the forward map has them
        self._complex_ek = np.iscomplexobj(np.real_if_close(self._ek))
        self._tr_average = nb > 1 and nonlocal_sde._has_time_reversal(self._ek)
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
            self._roots_comm = comm.Split(0 if self._node_comm.rank == 0 else 1)
            sigma.compress_q_dimension()
            self._dn_dmu, self._kappa = self._filling_derivatives(sigma, sigma_dmft_full, mpi_dist_fullbz)

            chunk = (
                nonlocal_sde._sde_chunk_budget(comm, self._node_comm) if chunk_budgets is None else chunk_budgets.exact
            )
            self._sde_chunk = self._aux_chunk = chunk
            self._block = 1 if chunk_budgets is None else max(1, chunk_budgets.exact_block)
            self._per_phase = chunk_budgets is not None and chunk_budgets.exact_vertices_per_phase
            self._vertices = {}

            gchi0_q = self._bubble(self._giwk)
            with self._local_vertex(*_F_DC) as f_dc:
                self._kernel = nonlocal_sde.calculate_sigma_dc_kernel(f_dc, gchi0_q, u_loc)
            full_sum, core_sum, core = self._split_bubble(gchi0_q)
            self._gchi0_inv = core.invert(copy=False)

            self._channels = []
            for channel, weight in _CHANNELS:
                with self._local_vertex(f"gamma_{channel.value}_loc.npy", channel) as gamma:
                    aux = nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, self._gchi0_inv, u_loc, self._aux_chunk)
                vrg = nonlocal_sde.create_vrg_r_q(aux, self._gchi0_inv)
                chi = nonlocal_sde.create_generalized_chi_q_with_shell_correction(
                    aux.sum_over_all_vn(beta), full_sum, core_sum, u_loc, v_nonloc
                )
                aux.free()
                self._kernel.add(nonlocal_sde.calculate_kernel_r_q(vrg, chi, v_nonloc, u_loc).scale(weight), copy=False)
                u_r = v_nonloc.as_channel(channel) + u_loc.as_channel(channel)
                self._channels.append((weight, channel, vrg, chi, u_r))
            full_sum.free()
            core_sum.free()

            giwk_cut, cut_win = nonlocal_sde._cut_and_reshare_giwk(
                self._giwk, self._giwk_win, self._node_comm, box.niv_core + box.niw_core
            )
            self._g_r, self._g_r_win = nonlocal_sde._build_rspace_giwk_window(giwk_cut, self._node_comm)
            self._g_r_niv = giwk_cut.niv
            if cut_win is not None:
                giwk_cut.mat = None
            nonlocal_sde._free_shared_window(cut_win, self._node_comm)

            shape, dtype = self._giwk.mat.shape, self._giwk.mat.dtype
            self._dg, self._dg_win = mpi_utils.allocate_node_shared_array(self._node_comm, shape, dtype)

    @contextlib.contextmanager
    def _local_vertex(self, name: str, channel: SpinChannel, axes: tuple[int, ...] | None = None):
        """
        Yields a local vertex of the output folder, held once per node (collective): the held window, or a fresh load
        of the file (see :func:`~dgamore.nonlocal_sde._load_node_shared_local_vertex`) that is kept when the vertices
        stay resident and freed at the end of the phase when they are loaded per phase. The load is freed on the
        normal path only: an exception aborts the job instead of meeting another collective.

        :param name: File name of the vertex in the output folder.
        :param channel: Its spin channel.
        :param axes: The axis order its window stores (see
            :func:`~dgamore.nonlocal_sde._load_node_shared_local_vertex`).
        :return: A context yielding the :class:`LocalFourPoint`.
        """
        if name in self._vertices:
            yield self._vertices[name][0]
            return
        path = os.path.join(config.output.output_path, name)
        vertex, win = nonlocal_sde._load_node_shared_local_vertex(self._node_comm, path, channel, axes=axes)
        yield vertex
        if self._per_phase:
            vertex.mat = None
            nonlocal_sde._free_shared_window(win, self._node_comm)
        else:
            self._vertices[name] = (vertex, win)

    def project(self, y: np.ndarray) -> np.ndarray:
        """
        Projects a real sector vector onto the self-energies the loop can reach (see :func:`sector_projector`), on
        rank 0.

        :param y: The real sector vector.
        :return: The projected real sector vector.
        """
        return to_vec(self._project(to_mat(y, self._sector_shape)))

    def _filling_derivatives(
        self, sigma: SelfEnergy, sigma_dmft_full: SelfEnergy, mpi_dist_fullbz: MpiDistributor
    ) -> tuple[float | None, np.ndarray]:
        r"""
        Returns the derivatives of the filling and the occupations with respect to :math:`\mu` (collective): the
        filling of :func:`~dgamore.greens_function.get_total_fill` on the loop box, with its local model,

        .. math:: \partial_\mu N = 2\,\mathrm{tr}\Big[\beta\bar\rho(1 - \bar\rho) + \frac{1}{\beta}\sum_{\nu}
            \mathrm{Re}\big((\bar G^{\nu}_{\mathrm{mod}})^2 - \langle (G^{\mathrm{k}})^2\rangle_{\mathbf{k}}\big)\Big],

        and the k-resolved occupation of :func:`~dgamore.nonlocal_sde._update_occ_and_energies_distributed` on the
        DMFT box, :math:`\kappa^{\mathbf{k}} = \beta\rho^{\mathbf{k}}(1 - \rho^{\mathbf{k}}) + \frac{1}{\beta}
        \sum_{\nu}\big((G^{\mathrm{k}}_{\mathrm{mod}})^2 - (G^{\mathrm{k}})^2\big)` (complex Hermitian for a complex
        dispersion, its real part for a real one, as the occupation), both summed over the momentum slices of the
        ranks. The fitted moments lie in the shell, beyond the core window.

        :param sigma: The loop's :class:`SelfEnergy` (compressed momentum axis).
        :param sigma_dmft_full: The DMFT :class:`SelfEnergy` supplying the shell frequencies (momentum-local).
        :param mpi_dist_fullbz: MPI distributor over the full BZ q-points.
        :return: :math:`\partial_\mu N` (on rank 0, ``None`` elsewhere) and :math:`\kappa^{\mathbf{k}}`
            ``[kx, ky, kz, o1, o2]`` on every rank.
        """
        beta, mu, nb = self._beta, self._mu, config.sys.n_bands
        eye = np.eye(nb)
        ek = self._ek.reshape(-1, nb, nb)
        ek_my = ek[mpi_dist_fullbz.my_slice]

        sigma_occ = nonlocal_sde._occupation_self_energy(sigma, sigma_dmft_full, mpi_dist_fullbz)
        smom0 = sigma_occ.smom[0]
        rho = _fermi_dirac_density(np.real_if_close(ek_my) + smom0 - mu * eye, beta)
        model = np.broadcast_to(smom0[None, :, :, None], (1, nb, nb, sigma_occ.mat.shape[-1]))
        box_sum = _squared_box_sum(model, mu, ek_my, beta) - _squared_box_sum(sigma_occ.mat, mu, ek_my, beta)
        kappa_my = beta * rho @ (eye - rho) + box_sum / beta
        sigma_occ.free()
        kappa = nonlocal_sde._assemble_occupation(kappa_my.reshape(-1, 1, 1, nb, nb), mpi_dist_fullbz)[2]

        # only the trace enters the filling, and the reduction must not carry a slice-dependent dtype
        g2_sum = _squared_box_sum(sigma.mat[mpi_dist_fullbz.my_slice], mu, ek_my, beta).real.sum(axis=0)
        if mpi_dist_fullbz.mpi_size > 1:
            g2_sum = mpi_dist_fullbz.allreduce(g2_sum)
        if self._comm.rank != 0:
            return None, kappa
        smom0 = sigma.fit_smom()[0]
        hloc = np.mean(self._ek, axis=(0, 1, 2))
        rho = _fermi_dirac_density(hloc.real + smom0 - mu * eye, beta)
        model = np.broadcast_to(smom0[None, :, :, None], (1, nb, nb, sigma.mat.shape[-1]))
        model_sum = _squared_box_sum(model, mu, hloc[None], beta)[0]
        inner = beta * rho @ (eye - rho) + (model_sum - g2_sum / ek.shape[0]) / beta
        return float(2.0 * np.trace(inner).real), kappa

    def _bubble(self, giwk: GreensFunction) -> FourPoint:
        """
        Returns the bubble of ``giwk`` on the loop's box (collective), as the proposal builds it.

        :param giwk: The momentum-dependent :class:`GreensFunction` of the loop box.
        :return: The bubble on this rank's irreducible q-points (full fermionic box, half niw range).
        """
        return BubbleGenerator.create_generalized_chi0_q_fft(
            self._dist,
            giwk,
            config.box.niw_core,
            config.box.niv_full,
            config.lattice.k_grid,
            self._beta,
            node_comm=self._node_comm,
        )

    def _split_bubble(self, gchi0_q: FourPoint) -> tuple[FourPoint, FourPoint, FourPoint]:
        """
        Splits a bubble into the frequency sums the shell correction reads and its core box, as the proposal does;
        ``gchi0_q`` is freed.

        :param gchi0_q: The bubble on the full fermionic box.
        :return: The sums over the full and the core box and the core-box bubble.
        """
        beta = self._beta
        full_sum = gchi0_q.sum_over_all_vn(beta).scale(1.0 / beta)
        core = gchi0_q.cut_niv(config.box.niv_core)
        gchi0_q.free()
        core_sum = core.sum_over_all_vn(beta).scale(1.0 / beta)
        return full_sum, core_sum, core

    def _greens_function(self, mat: np.ndarray) -> GreensFunction:
        """
        Wraps an array of the loop box, e.g. a node-shared window, into a :class:`GreensFunction` without copying.

        :param mat: The array ``[kx, ky, kz, o1, o2, 2 niv]``.
        :return: The :class:`GreensFunction` viewing ``mat``.
        """
        return GreensFunction(
            mat, None, self._ek, True, False, False, nk=self._ek.shape[:3], beta=self._beta, mu=self._mu
        )

    def _loop_box_rows(self) -> tuple[np.ndarray, slice]:
        """
        Returns the loop-box Green's function with a compressed momentum axis and the core window's frequency slice.

        :return: The view ``[k, o1, o2, 2 niv]`` of the node's window and the slice of the core frequencies.
        """
        mat, niv, niv_core = self._giwk.mat, self._giwk.niv, config.box.niv_core
        return mat.reshape(-1, *mat.shape[-3:]), slice(niv - niv_core, niv + niv_core)

    def _occupation_change(self, dx: np.ndarray) -> np.ndarray:
        r"""
        Returns :math:`\frac{1}{\beta}\sum_{\nu}G^{\mathrm{k}}\,\delta x^{\mathrm{k}}\,G^{\mathrm{k}}` per momentum
        over the core window, the occupation change at fixed :math:`\mu`, in momentum chunks (rank 0); its real part
        for a real dispersion, as the occupation takes it.

        :param dx: The window change ``[k, o1, o2, 2 niv_core]``.
        :return: The change ``[k, o1, o2]`` (complex128 for a complex dispersion, float64 otherwise).
        """
        g, core = self._loop_box_rows()
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
        g, core = self._loop_box_rows()
        dg = self._dg.reshape(g.shape)
        for start, stop in mpi_utils.row_chunks(g.shape[0], g.itemsize, g[0].size, memory_estimator.SLICE_CHUNK_BYTES):
            rows = slice(start, stop)
            dg[rows] = _sandwich(g[rows])
            dg[rows] *= -dmu
            dg[rows, ..., core] += _sandwich(g[rows, ..., core], dx[rows])

    def _bubble_response(self, y: np.ndarray, dmu: float, scale: float) -> FourPoint:
        r"""
        Returns :math:`\delta\chi^{\mathrm{q}\nu}_{0}` from the polarization of the bubble (collective): the node
        roots turn the change in the node's window into :math:`G^{\mathrm{k}} \pm s\,\delta G^{\mathrm{k}}` in place,
        rewriting the change before the second sign, and the bubble is built of each; the window holds no change
        afterwards.

        :param y: The real sector vector the window's change belongs to on the node roots, ``None`` elsewhere.
        :param dmu: Its chemical-potential change.
        :param scale: The factor :math:`s` (identical on every rank).
        :return: The bubble response on this rank's irreducible q-points (full fermionic box, half niw range).
        """
        response = None
        for sign in (1.0, -1.0):
            if self._node_comm.rank == 0:
                if sign < 0:
                    self._build_dg(self._from_sector(y), dmu)
                np.multiply(self._dg, sign * scale, out=self._dg)
                self._dg += self._giwk.mat
            self._node_comm.Barrier()
            bubble = self._bubble(self._greens_function(self._dg))
            if response is None:
                response = bubble
            else:
                response.sub(bubble, copy=False)
                bubble.free()
        return response.scale(0.5 / scale)

    def _kernel_responses(self, parts: list[tuple[FourPoint, FourPoint, FourPoint, FourPoint]]) -> list[FourPoint]:
        r"""
        Returns the kernel responses :math:`\delta K^{\mathrm{q}\nu} = \delta K^{\mathrm{q}\nu}_{\mathrm{dc}} +
        \delta K^{\mathrm{q}\nu}_{\mathrm{d}} + 3\,\delta K^{\mathrm{q}\nu}_{\mathrm{m}}` of a block of bubble
        responses, in the contraction layout (collective): per channel one Bethe-Salpeter pass solves the right-hand
        sides of every column against one factorization per slice (see
        :func:`~dgamore.nonlocal_sde.create_auxiliary_chi_r_q_sum`); the parts are freed. Every step after the bubble
        response is local in the momentum, so the pass walks this rank's momenta in the groups the solver's chunk
        budget gives and only each group's right-hand sides, solutions and kernel terms are alive at once.

        :param parts: Per column the double-counting kernel response and the sums over the full and the core box and
            the core-box bubble of the bubble response (see :meth:`_split_bubble`).
        :return: The kernel responses on this rank's irreducible q-points, one per column.
        """
        beta = self._beta
        dkernels = [part[0].compress_q_dimension().to_half_niw_range() for part in parts]
        n_q = self._gchi0_inv.current_shape[0]
        # one momentum's Bethe-Salpeter box, from the shape: a rank without momenta has no first row to measure
        per_q_box = (
            self._gchi0_inv.mat.itemsize
            * int(np.prod(self._gchi0_inv.current_shape[1:]))
            * self._gchi0_inv.current_shape[-1]
        )
        q_step = max(1, int(self._aux_chunk // max(1, per_q_box)))
        for weight, channel, vrg, chi, u_r in self._channels:
            with self._local_vertex(f"gamma_{channel.value}_loc.npy", channel) as gamma:
                for q_start in range(0, n_q, q_step):
                    q_stop = min(n_q, q_start + q_step)

                    def take(obj):
                        """A view of this group's momenta of a momentum-resolved object (the object for one group)."""
                        return obj if q_step >= n_q else obj.take_q_index_slice(q_start, q_stop, copy=False)

                    gchi0_inv, vrg_q, chi_q, u_q = take(self._gchi0_inv), take(vrg), take(chi), take(u_r)
                    rhs = [gchi0_inv @ (take(dcore) @ vrg_q) for *_, dcore in parts]
                    solved = nonlocal_sde.create_auxiliary_chi_r_q_sum(
                        gamma, gchi0_inv, self._u_loc, self._aux_chunk, rhs=rhs
                    )
                    for dkernel, (_, dfull_sum, dcore_sum, _), t, s in zip(dkernels, parts, rhs, solved):
                        dvrg = (gchi0_inv @ s).sub(t, copy=False)
                        t.free()
                        dsum = s.scale(1.0 / beta).sum_over_all_vn(beta) + take(dfull_sum) - take(dcore_sum)
                        s.free()
                        dchi = dsum - chi_q @ u_q @ dsum
                        dchi = dchi - dchi @ u_q @ chi_q
                        dk = dvrg - dvrg @ u_q @ chi_q - vrg_q @ u_q @ dchi
                        term = (dk @ u_q).permute_orbitals("abcd->badc", copy=False).scale(weight)
                        if q_step >= n_q:
                            dkernel.add(term, copy=False)
                        else:
                            np.add(dkernel.mat[q_start:q_stop], term.mat, out=dkernel.mat[q_start:q_stop])
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

    def _contraction(self, dkernel: FourPoint, docc: np.ndarray | None, docc_k: np.ndarray | None) -> np.ndarray | None:
        r"""
        Returns the product for the change in the node's window from its kernel response (collective): the
        contraction once with :math:`(K, \delta G)` and once with :math:`(\delta K, G)`, on rank 0 the Hartree-Fock
        term of the occupation changes and the reduction to the sector; ``dkernel`` is freed.

        :param dkernel: The kernel response of the change.
        :param docc: The local occupation change (rank 0).
        :param docc_k: The momentum-resolved occupation change (rank 0).
        :return: The product as a real sector vector on rank 0; ``None`` elsewhere.
        """
        box = config.box
        dg = self._greens_function(self._dg)
        dg_cut, dg_cut_win = nonlocal_sde._cut_and_reshare_giwk(
            dg, self._dg_win, self._node_comm, box.niv_core + box.niw_core
        )
        dg_r, dg_r_win = nonlocal_sde._build_rspace_giwk_window(dg_cut, self._node_comm)
        dg_r_niv = dg_cut.niv
        if dg_cut_win is not None:
            dg_cut.mat = None
        nonlocal_sde._free_shared_window(dg_cut_win, self._node_comm)
        propagator = nonlocal_sde._run_column_sde(self._kernel, self._dist, dg_r, dg_r_niv, self._sde_chunk)
        dg_r = None
        nonlocal_sde._free_shared_window(dg_r_win, self._node_comm)
        vertex = nonlocal_sde._run_column_sde(dkernel, self._dist, self._g_r, self._g_r_niv, self._sde_chunk)
        dkernel.free()
        if self._comm.rank != 0:
            return None

        dsigma = (propagator + vertex).ifft().to_full_niv_range()
        hartree, fock = nonlocal_sde.get_hartree_fock(self._u_loc, self._v_nonloc_full, docc, docc_k)
        dsigma = dsigma + hartree + fock
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
        passes the vectors as the columns of ``ys``, every other rank ``None``; the vectors reach only the node roots,
        which unfold them, the other ranks the column count. The columns run in pieces of at most
        the block width the chunk budgets give (one column without them), and every column of a piece runs the stages
        of one product, except that the Bethe-Salpeter systems of both channels take the right-hand sides of the whole
        piece at once, against one factorization per slice, so a piece costs one factorization pass per channel
        whatever its width. The node's window holds one change of the Green's function at a time, and the bubble
        response shifts it in place, so each column's change is written twice for its bubble response and again for
        its contraction.

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
        Returns the products of one piece of a block (collective, see :meth:`matvec_block`): the change and bubble
        response of every column, one Bethe-Salpeter pass per channel for the piece, and the contraction of every
        column.

        :param columns: The piece's real sector vectors on the node roots, ``None`` per column elsewhere.
        :return: The products as real sector vectors on rank 0, ``None`` per column elsewhere.
        """
        comm = self._comm
        parts = []
        with self._local_vertex(*_F_DC) as f_dc:
            for y in columns:
                dmu = self._apply_change(y)[0]
                norm = (np.linalg.norm(self._giwk.mat), np.linalg.norm(self._dg)) if comm.rank == 0 else None
                norm = comm.bcast(norm, root=0)
                scale = float(norm[0] / norm[1]) if norm[1] > 0 else 1.0
                dchi0 = self._bubble_response(y, dmu, scale)
                dkernel_dc = nonlocal_sde.calculate_sigma_dc_kernel(f_dc, dchi0, self._u_loc)
                parts.append((dkernel_dc, *self._split_bubble(dchi0)))
        products = []
        for y, dkernel in zip(columns, self._kernel_responses(parts)):
            _, docc, docc_k = self._apply_change(y)
            products.append(self._contraction(dkernel, docc, docc_k))
        return products

    def free(self) -> None:
        """
        Releases the held quantities, the shared windows and the node and node-root communicators (collective).

        :return: None.
        """
        node_comm = self._node_comm
        for _, _, vrg, chi, _ in self._channels:
            vrg.free()
            chi.free()
        self._channels = []
        for vertex, win in self._vertices.values():
            vertex.mat = None
            nonlocal_sde._free_shared_window(win, node_comm)
        self._vertices = {}
        self._kernel.free()
        self._gchi0_inv.free()
        self._g_r = self._dg = None
        for win in (self._g_r_win, self._dg_win):
            nonlocal_sde._free_shared_window(win, node_comm)
        self._roots_comm.Free()
        self._giwk.mat = None
        nonlocal_sde._release_shared_giwk(self._giwk_win, node_comm)


def _merge_modes(theta: np.ndarray, vectors: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Adds the conjugate partner of every complex eigenvalue (the operator is real) and drops repeated eigenpairs: a
    pair repeats an earlier one when its value lies within ``100 tol max(1, |theta|)`` of it and its eigenvector
    overlaps it by more than ``_DUPLICATE_OVERLAP``, so the partners of a degenerate value both stay.

    :param theta: The eigenvalues ARPACK returned, all targets concatenated.
    :param vectors: Their eigenvectors as columns.
    :param tol: The ARPACK tolerance.
    :return: The distinct eigenpairs, values and eigenvectors.
    """
    complex_ = np.abs(theta.imag) > tol * np.maximum(1.0, np.abs(theta))
    theta = np.concatenate((theta, theta[complex_].conj()))
    vectors = np.concatenate((vectors, vectors[:, complex_].conj()), axis=1)
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=0), np.finfo(np.float64).tiny)
    keep = []
    for i, value in enumerate(theta):
        close = [j for j in keep if abs(value - theta[j]) <= 100 * tol * max(1.0, abs(value))]
        if all(abs(np.vdot(unit[:, j], unit[:, i])) <= _DUPLICATE_OVERLAP for j in close):
            keep.append(i)
    return theta[keep], vectors[:, keep]


def leading_eigenpairs(jac: ExactJacobian, comm: MPI.Comm) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    r"""
    Computes leading eigenpairs of the Jacobian with ARPACK (``scipy.sparse.linalg.eigs``) on rank 0, while every
    other rank serves the collective products of :meth:`ExactJacobian.matvec` until rank 0 signals the end. Two
    targets of :data:`~dgamore.memory_estimator.EXACT_JACOBIAN_MODES` pairs each are searched: the largest real
    parts of the eigenvalues :math:`\theta` of :math:`J` (the modes the tracker flips, :math:`\lambda_\Pi = 1 -
    \theta` with the smallest real part) and the largest moduli (the stiff modes that bind the damping). A target
    that does not converge within about ``_MAX_MATVECS`` products contributes its converged pairs, with a warning.
    The residual bound of a pair is the ARPACK tolerance :math:`\max(100\,\epsilon, 10^{-10})\max(1, |\theta|)`,
    :math:`\epsilon` of the storage precision.

    :param jac: The operator (``n_real``, ``matvec``, ``project``).
    :param comm: The MPI communicator.
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
    count = [0]

    def apply(y: np.ndarray) -> np.ndarray:
        """Signals the serving ranks and returns the collective product :math:`J y`."""
        comm.bcast(True, root=0)
        count[0] += 1
        return jac.matvec(y)

    operator = spla.LinearOperator((n, n), matvec=apply, dtype=np.float64)
    v0 = jac.project(np.random.default_rng(0).standard_normal(n))
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
        found_theta.append(theta)
        found_vectors.append(vectors)
    # released on the normal path only: an exception aborts the job instead of meeting another collective
    comm.bcast(False, root=0)

    theta, vectors = _merge_modes(np.concatenate(found_theta), np.concatenate(found_vectors, axis=1), tol)
    # a restart can leave the projected sector, where the operator vanishes; such a zero mode is no loop direction
    inside = [
        np.linalg.norm(jac.project(v.real) + 1j * jac.project(v.imag) - v) <= 0.5 * np.linalg.norm(v) for v in vectors.T
    ]
    theta, vectors = theta[inside], vectors[:, inside]
    lam_pi = 1.0 - theta
    order = np.argsort(lam_pi.real)
    lam_pi, vectors = lam_pi[order], vectors[:, order]
    u = vectors / np.linalg.norm(vectors, axis=0)
    logger.info(
        f"Exact Jacobian: {lam_pi.size} eigenpairs from {count[0]} products, lambda_Pi "
        f"{', '.join(f'{lam.real:+.4f}{lam.imag:+.4f}j' for lam in lam_pi)}."
    )
    return lam_pi, tol * np.maximum(1.0, np.abs(theta[order])), u


def _orthonormal_extension(basis: np.ndarray, block: np.ndarray) -> np.ndarray:
    """
    Appends to ``basis`` the part of every column of ``block`` orthogonal to it and to the columns appended before,
    normalized and orthogonalized twice for stability; a column whose remainder is below ``1e-8`` of its own norm
    adds nothing.

    :param basis: Orthonormal columns, shape ``[n, k]``.
    :param block: The columns to append, shape ``[n, m]``.
    :return: The extended orthonormal basis.
    """
    for column in block.T:
        remainder = column.astype(np.float64, copy=True)
        for _ in range(2):
            remainder -= basis @ (basis.T @ remainder)
        norm = np.linalg.norm(remainder)
        if norm > 1e-8 * np.linalg.norm(column):
            basis = np.column_stack([basis, remainder / norm])
    return basis


def certify_subspace(
    jac: ExactJacobian, start: np.ndarray | None, comm: MPI.Comm
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    r"""
    Certifies the eigenpairs of the Jacobian next to a few given directions with a short block Arnoldi on rank 0,
    while every other rank serves the collective block products of :meth:`ExactJacobian.matvec_block` until rank 0
    signals the end, as in :func:`leading_eigenpairs`: the start columns and every later extension of the basis each
    go through one block product. The start columns are projected onto the sector (:meth:`ExactJacobian.project`)
    and orthonormalized into the basis :math:`Q`; after every block of products
    :math:`W = J Q` the Rayleigh-Ritz of the projected matrix :math:`B = Q^{T} W` gives the Ritz pairs
    :math:`(\theta, y)` and their residuals

    .. math:: \mathrm{res} = \frac{\lVert W y - \theta\, Q y \rVert}{\lVert Q y \rVert},

    exact since every column of :math:`W` is an exact product. The search stops once the pair with the largest real
    part of :math:`\theta`, i.e. the smallest real part of :math:`\lambda_\Pi = 1 - \theta`, passes the Ritz gate
    of the Jacobian tracker, :math:`\mathrm{res} \leq g \max(1, |\theta|)` with :math:`g` its ``_RITZ_GATE``, or once
    the basis holds ``_CHECK_BASIS`` vectors; until then the real and the imaginary part of that pair's residual
    extend the basis.

    :param jac: The operator (``n_real``, ``matvec_block``, ``project``).
    :param start: The real start columns in the operator's sector coordinates, shape ``[n_real, m]``, on rank 0;
        ``None`` elsewhere.
    :param comm: The MPI communicator.
    :return: On rank 0 the tuple ``(lam_pi, res, u, n_products)`` of every Ritz pair of the final basis, sorted by the
        real part of :math:`\lambda_\Pi`, ``u`` the normalized complex eigenvectors as columns in the operator's
        sector coordinates; ``None`` elsewhere.
    """
    if comm.rank != 0:
        while comm.bcast(None, root=0):
            jac.matvec_block(None)
        return None

    def apply(block: np.ndarray) -> np.ndarray:
        """Signals the serving ranks and returns the collective block product :math:`J Y`."""
        comm.bcast(True, root=0)
        return jac.matvec_block(block)

    n = jac.n_real
    theta, ritz, res = np.zeros(0, dtype=np.complex128), np.zeros((n, 0), dtype=np.complex128), np.zeros(0)
    images = np.zeros((n, 0))
    sector_start = np.column_stack([jac.project(column) for column in np.asarray(start).T])
    basis = _orthonormal_extension(np.zeros((n, 0)), sector_start)[:, :_CHECK_BASIS]
    while basis.shape[1] > images.shape[1]:
        images = np.column_stack([images, apply(basis[:, images.shape[1] :])])
        theta, y, _ = _schur_ritz(basis.T @ images)
        ritz = basis @ y
        residuals = images @ y - ritz * theta
        res = np.linalg.norm(residuals, axis=0) / np.linalg.norm(ritz, axis=0)
        lead = int(np.argmax(theta.real))
        if res[lead] <= _RITZ_GATE * max(1.0, abs(theta[lead])):
            break
        block = np.column_stack([residuals[:, lead].real, residuals[:, lead].imag])
        basis = _orthonormal_extension(basis, block)[:, :_CHECK_BASIS]
    # released on the normal path only: an exception aborts the job instead of meeting another collective
    comm.bcast(False, root=0)

    lam_pi = 1.0 - theta
    order = np.argsort(lam_pi.real)
    lam_pi, res, ritz = lam_pi[order], res[order], ritz[:, order]
    u = ritz / np.linalg.norm(ritz, axis=0)
    config.logger.info(
        f"Exact check: {images.shape[1]} products, lambda_Pi "
        f"{', '.join(f'{lam.real:+.4f}{lam.imag:+.4f}j (res {r:.1e})' for lam, r in zip(lam_pi, res))}."
    )
    return lam_pi, res, u, images.shape[1]

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
:meth:`~dgamore.jacobian_stabilization.JacobianTracker.install_exact_spectrum`).

The Jacobian is evaluated at the pure map: no lambda correction, susceptibility restriction or annealing mass, and
the momentum-dependent :math:`V^{\mathbf{q}}` Hartree-Fock offset of the shell held at its value (exact for
:math:`V = 0`).
"""

import os

import mpi4py.MPI as MPI
import numpy as np
import scipy.sparse.linalg as spla

import dgamore.config as config
import dgamore.mpi_utils as mpi_utils
from dgamore import memory_estimator, n_point_base, nonlocal_sde
from dgamore.bubble_gen import BubbleGenerator
from dgamore.four_point import FourPoint
from dgamore.greens_function import _MODEL_EPOT_CHUNK_ELEMENTS, GreensFunction, _fermi_dirac_density
from dgamore.interaction import Interaction, LocalInteraction
from dgamore.jacobian_stabilization import to_mat, to_vec
from dgamore.matsubara_frequencies import MFHelper
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel, deferred_collection
from dgamore.self_energy import SelfEnergy

_CHANNELS = ((SpinChannel.DENS, 1.0), (SpinChannel.MAGN, 3.0))  # ladder channels and their weights in the kernel
_MAX_MATVECS = 150  # rough bound on the Jacobian-vector products of one ARPACK target


def _squared_box_sum(sigma_mat: np.ndarray, mu: float, ek: np.ndarray, beta: float) -> np.ndarray:
    r"""
    Returns :math:`\sum_{\nu}\mathrm{Re}\,[(G^{\mathrm{k}})^2]` per momentum over the frequency box of ``sigma_mat``,
    :math:`G^{\mathrm{k}} = [(\imath\nu + \mu) - \varepsilon(\mathbf{k}) - \Sigma^{\mathrm{k}}]^{-1}`, i.e. minus
    the derivative of the box sum of the occupation with respect to :math:`\mu`. The Dyson equation is solved in
    double precision in momentum chunks, as :meth:`GreensFunction.get_fill_nonlocal_from_sigma` does.

    :param sigma_mat: The self-energy ``[k, o1, o2, 2 niv]``; a single momentum row broadcasts over ``ek``.
    :param mu: Chemical potential :math:`\mu`.
    :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})`, shape ``[k, o1, o2]``.
    :param beta: Inverse temperature :math:`\beta`.
    :return: The sums ``[k, o1, o2]`` (float64).
    """
    nk, nb = ek.shape[0], ek.shape[-1]
    iv = 1j * MFHelper.vn(sigma_mat.shape[-1] // 2, beta)
    static = (iv[:, None, None] + mu) * np.eye(nb)
    out = np.empty((nk, nb, nb))
    step = max(1, _MODEL_EPOT_CHUNK_ELEMENTS // (nb * nb * iv.size))
    for start in range(0, nk, step):
        stop = min(nk, start + step)
        rows = sigma_mat if sigma_mat.shape[0] == 1 else sigma_mat[start:stop]
        g = np.linalg.inv(static[None] - ek[start:stop, None] - np.moveaxis(rows, -1, 1))
        out[start:stop] = np.einsum("kvab,kvbc->kac", g, g, optimize=True).real
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


class ExactJacobian:
    r"""
    The Jacobian of the loop's self-energy map at one self-energy as a collective Jacobian-vector product.

    The operator acts on the sector the iteration lives in: the self-energy on the irreducible momenta and the
    positive core frequencies, unfolded with the lattice symmetry (:meth:`SelfEnergy.map_to_full_bz`) and completed
    by its Matsubara Hermiticity (:meth:`SelfEnergy.to_full_niv_range`); the map carries that sector into itself, so
    the eigenvalues of the sector operator are those of the map on the self-energies the loop can reach. A sector
    vector is the real vector of :func:`~dgamore.jacobian_stabilization.to_vec` of the complex sector array, and
    :meth:`expand` maps it to the tracker's vector of the whole core window.

    One product :math:`J\,\delta x`, the proposal's stages linearized:

    1. rank 0: :math:`\delta\mu = -\partial_x N\cdot\delta x/\partial_\mu N` for the filling ``update_mu`` holds,
       and :math:`\delta n^{\mathbf{k}} = \frac{1}{\beta}\sum_{\nu}\mathrm{Re}\,(G^{\mathrm{k}}\,\delta x^{\mathrm{k}}
       \,G^{\mathrm{k}}) + \kappa^{\mathbf{k}}\,\delta\mu`;
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
       :math:`(\delta K, G)`, and on rank 0 the Hartree-Fock term of :math:`\delta n^{\mathbf{k}}`.

    Held between products: the Green's function of the loop box and its real-space window (once per node), the
    local vertices (once per node), the inverse core bubble, the three-leg vertices and dressed susceptibilities of
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
            contraction (sized by the driver from the memory estimate); ``None`` gives both the job-wide fair-share
            budget of :func:`~dgamore.nonlocal_sde._sde_chunk_budget`.
        """
        box, k_grid, nb = config.box, config.lattice.k_grid, config.sys.n_bands
        beta = float(config.sys.beta)
        self._comm, self._dist = comm, mpi_dist_irrk
        self._u_loc, self._v_nonloc_full = u_loc, v_nonloc_full
        self._beta, self._mu = beta, float(mu)
        self._ek = config.lattice.hamiltonian.get_ek()
        self._sector_shape = (k_grid.nk_irr, nb, nb, box.niv_core)
        self.n_real = 2 * int(np.prod(self._sector_shape))

        with deferred_collection():
            self._giwk, self._giwk_win, self._node_comm = nonlocal_sde._build_giwk_full(comm, sigma, mu, self._ek, beta)
            sigma.compress_q_dimension()
            self._dn_dmu, self._kappa = self._filling_derivatives(sigma, sigma_dmft_full, mpi_dist_fullbz)

            chunk = (
                nonlocal_sde._sde_chunk_budget(comm, self._node_comm) if chunk_budgets is None else chunk_budgets.sde
            )
            self._sde_chunk = chunk
            self._aux_chunk = chunk if chunk_budgets is None else chunk_budgets.chiq_aux

            out = config.output.output_path
            gchi0_q = self._bubble(self._giwk)
            self._f_dc, self._f_dc_win = nonlocal_sde._load_node_shared_local_vertex(
                self._node_comm, os.path.join(out, "f_dc_loc.npy"), SpinChannel.MAGN, axes=(4, 0, 1, 5, 2, 3, 6)
            )
            self._kernel = nonlocal_sde.calculate_sigma_dc_kernel(self._f_dc, gchi0_q, u_loc)
            full_sum, core_sum, core = self._split_bubble(gchi0_q)
            self._gchi0_inv = core.invert(copy=False)

            self._channels = []
            for channel, weight in _CHANNELS:
                gamma, gamma_win = nonlocal_sde._load_node_shared_local_vertex(
                    self._node_comm, os.path.join(out, f"gamma_{channel.value}_loc.npy"), channel
                )
                aux = nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, self._gchi0_inv, u_loc, self._aux_chunk)
                vrg = nonlocal_sde.create_vrg_r_q(aux, self._gchi0_inv)
                chi = nonlocal_sde.create_generalized_chi_q_with_shell_correction(
                    aux.sum_over_all_vn(beta), full_sum, core_sum, u_loc, v_nonloc
                )
                aux.free()
                self._kernel.add(nonlocal_sde.calculate_kernel_r_q(vrg, chi, v_nonloc, u_loc).scale(weight), copy=False)
                u_r = v_nonloc.as_channel(channel) + u_loc.as_channel(channel)
                self._channels.append((weight, gamma, gamma_win, vrg, chi, u_r))
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
            self._g_shift, self._g_shift_win = mpi_utils.allocate_node_shared_array(self._node_comm, shape, dtype)

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
        \sum_{\nu}\mathrm{Re}\big((G^{\mathrm{k}}_{\mathrm{mod}})^2 - (G^{\mathrm{k}})^2\big)`, both summed over
        the momentum slices of the ranks. The fitted moments lie in the shell, beyond the core window.

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
        rho = _fermi_dirac_density(ek_my.real + smom0 - mu * eye, beta)
        model = np.broadcast_to(smom0[None, :, :, None], (1, nb, nb, sigma_occ.mat.shape[-1]))
        box_sum = _squared_box_sum(model, mu, ek_my, beta) - _squared_box_sum(sigma_occ.mat, mu, ek_my, beta)
        kappa_my = beta * rho @ (eye - rho) + box_sum / beta
        sigma_occ.free()
        kappa = nonlocal_sde._assemble_occupation(kappa_my.reshape(-1, 1, 1, nb, nb), mpi_dist_fullbz)[2]

        g2_sum = _squared_box_sum(sigma.mat[mpi_dist_fullbz.my_slice], mu, ek_my, beta).sum(axis=0)
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

    def _window(self, y: np.ndarray) -> np.ndarray:
        """
        Unfolds a sector vector to the core window of the whole BZ.

        :param y: The real sector vector.
        :return: The complex window ``[k, o1, o2, 2 niv_core]`` (compressed momentum axis).
        """
        half = SelfEnergy(
            to_mat(y, self._sector_shape), config.lattice.nk, False, True, calc_smom=False, beta=self._beta
        )
        return half.map_to_full_bz(config.lattice.k_grid).to_full_niv_range().mat

    def expand(self, y: np.ndarray) -> np.ndarray:
        """
        Maps a real sector vector to the real vector of the whole core window, in the layout the Jacobian tracker
        flattens the loop's iterates with (:func:`~dgamore.jacobian_stabilization.to_vec`).

        :param y: The real sector vector.
        :return: The real vector of the window.
        """
        return to_vec(self._window(y))

    def _loop_box_rows(self) -> tuple[np.ndarray, slice]:
        """
        Returns the loop-box Green's function with a compressed momentum axis and the core window's frequency slice.

        :return: The view ``[k, o1, o2, 2 niv]`` of the node's window and the slice of the core frequencies.
        """
        mat, niv, niv_core = self._giwk.mat, self._giwk.niv, config.box.niv_core
        return mat.reshape(-1, *mat.shape[-3:]), slice(niv - niv_core, niv + niv_core)

    def _occupation_change(self, dx: np.ndarray) -> np.ndarray:
        r"""
        Returns :math:`\frac{1}{\beta}\sum_{\nu}\mathrm{Re}\,(G^{\mathrm{k}}\,\delta x^{\mathrm{k}}\,G^{\mathrm{k}})`
        per momentum over the core window, the occupation change at fixed :math:`\mu`, in momentum chunks (rank 0).

        :param dx: The window change ``[k, o1, o2, 2 niv_core]``.
        :return: The change ``[k, o1, o2]`` (float64).
        """
        g, core = self._loop_box_rows()
        out = np.empty(g.shape[:3])
        for start, stop in mpi_utils.row_chunks(g.shape[0], g.itemsize, g[0].size, memory_estimator.SLICE_CHUNK_BYTES):
            out[start:stop] = np.sum(_sandwich(g[start:stop, ..., core], dx[start:stop]).real, axis=-1)
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

    def _bubble_response(self, scale: float) -> FourPoint:
        r"""
        Returns :math:`\delta\chi^{\mathrm{q}\nu}_{0}` from the polarization of the bubble (collective): the node
        roots write :math:`G^{\mathrm{k}} \pm s\,\delta G^{\mathrm{k}}` into the node's window and the bubble is built
        of each.

        :param scale: The factor :math:`s` (identical on every rank).
        :return: The bubble response on this rank's irreducible q-points (full fermionic box, half niw range).
        """
        response = None
        for sign in (1.0, -1.0):
            if self._node_comm.rank == 0:
                np.multiply(self._dg, sign * scale, out=self._g_shift)
                self._g_shift += self._giwk.mat
            self._node_comm.Barrier()
            bubble = self._bubble(self._greens_function(self._g_shift))
            if response is None:
                response = bubble
            else:
                response.sub(bubble, copy=False)
                bubble.free()
        return response.scale(0.5 / scale)

    def _kernel_response(self, dchi0: FourPoint) -> FourPoint:
        r"""
        Returns the kernel response :math:`\delta K^{\mathrm{q}\nu} = \delta K^{\mathrm{q}\nu}_{\mathrm{dc}} +
        \delta K^{\mathrm{q}\nu}_{\mathrm{d}} + 3\,\delta K^{\mathrm{q}\nu}_{\mathrm{m}}` to a bubble response, in the
        contraction layout (collective); ``dchi0`` is freed.

        :param dchi0: The bubble response on the full fermionic box.
        :return: The kernel response on this rank's irreducible q-points.
        """
        beta = self._beta
        dkernel = nonlocal_sde.calculate_sigma_dc_kernel(self._f_dc, dchi0, self._u_loc)
        dfull_sum, dcore_sum, dcore = self._split_bubble(dchi0)
        for weight, gamma, _, vrg, chi, u_r in self._channels:
            rhs = self._gchi0_inv @ (dcore @ vrg)
            solved = nonlocal_sde.create_auxiliary_chi_r_q_sum(
                gamma, self._gchi0_inv, self._u_loc, self._aux_chunk, rhs=rhs
            )
            dvrg = (self._gchi0_inv @ solved).sub(rhs, copy=False)
            rhs.free()
            dsum = solved.scale(1.0 / beta).sum_over_all_vn(beta) + dfull_sum - dcore_sum
            solved.free()
            dchi = dsum - chi @ u_r @ dsum
            dchi = dchi - dchi @ u_r @ chi
            dk = dvrg - dvrg @ u_r @ chi - vrg @ u_r @ dchi
            dkernel.add((dk @ u_r).permute_orbitals("abcd->badc", copy=False).scale(weight), copy=False)
        dcore.free()
        dfull_sum.free()
        dcore_sum.free()
        return dkernel

    def matvec(self, y: np.ndarray | None) -> np.ndarray | None:
        r"""
        Returns :math:`J y` (collective over the communicator of the constructor): rank 0 passes the sector vector,
        every other rank ``None``.

        :param y: The real sector vector on rank 0.
        :return: :math:`J y` as a real sector vector on rank 0; ``None`` elsewhere.
        """
        comm, node_root = self._comm, self._node_comm.rank == 0
        box, nb = config.box, config.sys.n_bands
        with deferred_collection():
            y = mpi_utils.bcast_rows(comm, np.asarray(y, dtype=np.float64) if comm.rank == 0 else None, root=0)
            dx = self._window(y) if node_root else None

            dmu, docc, docc_k = None, None, None
            if comm.rank == 0:
                dn_x = self._occupation_change(dx)
                dmu = -2.0 * np.trace(dn_x.mean(axis=0)) / self._dn_dmu
                docc_k = dn_x.reshape(self._kappa.shape) + self._kappa * dmu
                docc = docc_k.mean(axis=(0, 1, 2))
            dmu = comm.bcast(dmu, root=0)

            if node_root:
                self._build_dg(dx, dmu)
            self._node_comm.Barrier()
            norm = (np.linalg.norm(self._giwk.mat), np.linalg.norm(self._dg)) if comm.rank == 0 else None
            norm = comm.bcast(norm, root=0)
            scale = float(norm[0] / norm[1]) if norm[1] > 0 else 1.0

            dkernel = self._kernel_response(self._bubble_response(scale))

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
            if comm.rank != 0:
                return None

            dsigma = (propagator + vertex).ifft().to_full_niv_range()
            hartree, fock = nonlocal_sde.get_hartree_fock(self._u_loc, self._v_nonloc_full, docc, docc_k)
            dsigma = dsigma + hartree + fock
            window = dsigma.mat.reshape(-1, nb, nb, 2 * box.niv_core)
            return to_vec(window[config.lattice.k_grid.irrk_ind][..., box.niv_core :])

    def free(self) -> None:
        """
        Releases the held quantities, the shared windows and the node communicator (collective).

        :return: None.
        """
        node_comm = self._node_comm
        for _, gamma, gamma_win, vrg, chi, _ in self._channels:
            vrg.free()
            chi.free()
            gamma.mat = None
            nonlocal_sde._free_shared_window(gamma_win, node_comm)
        self._channels = []
        self._f_dc.mat = None
        nonlocal_sde._free_shared_window(self._f_dc_win, node_comm)
        self._kernel.free()
        self._gchi0_inv.free()
        self._g_r = self._dg = self._g_shift = None
        for win in (self._g_r_win, self._dg_win, self._g_shift_win):
            nonlocal_sde._free_shared_window(win, node_comm)
        self._giwk.mat = None
        nonlocal_sde._release_shared_giwk(self._giwk_win, node_comm)


def _merge_modes(theta: np.ndarray, vectors: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Adds the conjugate partner of every complex eigenvalue (the operator is real) and drops repeated eigenvalues.

    :param theta: The eigenvalues ARPACK returned, all targets concatenated.
    :param vectors: Their eigenvectors as columns.
    :param tol: The ARPACK tolerance; values within ``100 tol max(1, |theta|)`` count as one.
    :return: The distinct eigenvalues and their eigenvectors.
    """
    complex_ = np.abs(theta.imag) > tol * np.maximum(1.0, np.abs(theta))
    theta = np.concatenate((theta, theta[complex_].conj()))
    vectors = np.concatenate((vectors, vectors[:, complex_].conj()), axis=1)
    keep = []
    for i, value in enumerate(theta):
        if all(abs(value - theta[j]) > 100 * tol * max(1.0, abs(value)) for j in keep):
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

    :param jac: The operator (``n_real``, ``matvec``, ``expand``).
    :param comm: The MPI communicator.
    :return: On rank 0 the tuple ``(lam_pi, res, u)`` sorted by the real part of :math:`\lambda_\Pi`, ``u`` the
        normalized complex eigenvectors as columns in the vector layout of the tracker (see
        :meth:`ExactJacobian.expand`); ``None`` elsewhere.
    """
    if comm.rank != 0:
        while comm.bcast(None, root=0):
            jac.matvec(None)
        return None

    logger = config.logger
    n = jac.n_real
    k = min(memory_estimator.EXACT_JACOBIAN_MODES, n - 2)
    ncv = min(memory_estimator.EXACT_JACOBIAN_NCV, n - 1)
    tol = max(100 * np.finfo(n_point_base.DTYPE).eps, 1e-10)
    count = [0]

    def apply(y: np.ndarray) -> np.ndarray:
        """Signals the serving ranks and returns the collective product :math:`J y`."""
        comm.bcast(True, root=0)
        count[0] += 1
        return jac.matvec(y)

    operator = spla.LinearOperator((n, n), matvec=apply, dtype=np.float64)
    v0 = np.random.default_rng(0).standard_normal(n)
    found_theta, found_vectors = [], []
    try:
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
    finally:
        comm.bcast(False, root=0)

    theta, vectors = _merge_modes(np.concatenate(found_theta), np.concatenate(found_vectors, axis=1), tol)
    lam_pi = 1.0 - theta
    order = np.argsort(lam_pi.real)
    lam_pi, vectors = lam_pi[order], vectors[:, order]
    u = np.empty((jac.expand(np.zeros(n)).size, theta.size), dtype=np.complex128)
    for j, v in enumerate(vectors.T):
        u[:, j] = jac.expand(v.real) + 1j * jac.expand(v.imag)
        u[:, j] /= np.linalg.norm(u[:, j])
    logger.info(
        f"Exact Jacobian: {lam_pi.size} eigenpairs from {count[0]} products, lambda_Pi "
        f"{', '.join(f'{lam.real:+.4f}{lam.imag:+.4f}j' for lam in lam_pi)}."
    )
    return lam_pi, tol * np.maximum(1.0, np.abs(theta[order])), u

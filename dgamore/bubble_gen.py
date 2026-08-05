# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Generalized bare susceptibilities (the "bubbles"). :class:`BubbleGenerator` builds the products of two Green's
functions :math:`\chi_{0;1234}^{\omega\nu} = -\beta\, G_{14}^{\nu}\, G_{32}^{\nu-\omega}` in the particle-hole and particle-particle channels,
both local and momentum-dependent. The non-local versions are evaluated either by an FFT over the BZ or by a
direct momentum-shift einsum, distributed over MPI ranks and optionally accelerated on the GPU (CuPy).
"""

import numpy as np
import scipy as sp

import dgamore.mpi_utils as mpi_utils
from dgamore.brillouin_zone import KGrid
from dgamore.dga_logger import DgaLogger
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction
from dgamore.local_four_point import LocalFourPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel, FrequencyNotation


class BubbleGenerator:
    """
    Collection of static factory methods that build the generalized bare susceptibilities (bubbles) from a Green's
    function in the particle-hole and particle-particle channels, both local and momentum-dependent.
    """

    @staticmethod
    def create_generalized_chi0(g_dmft: GreensFunction, niw: int, niv: int, beta: float) -> LocalFourPoint:
        r"""
        Returns the local generalized bare susceptibility
        :math:`\chi_{0;1234}^{\omega\nu} = -\beta\, G_{14}^{\nu}\, G_{32}^{\nu-\omega}`.

        :param g_dmft: The local (DMFT) :class:`GreensFunction`.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param beta: Inverse temperature :math:`\beta`.
        :return: The local bubble as a :class:`LocalFourPoint` with one bosonic and one fermionic frequency axis.
        """
        wn = MFHelper.wn(niw)
        niv_range = np.arange(-niv, niv)

        # the local Green's function carries a single-momentum dimension; index it away for the orbital algebra
        g_mat = g_dmft.mat[0, 0, 0]
        g_mat_transposed = g_dmft.transpose_orbitals().mat[0, 0, 0]
        g_left_mat = g_mat[:, None, None, :, None, g_dmft.niv - niv : g_dmft.niv + niv]
        g_right_mat = g_mat_transposed[None, :, :, None, g_dmft.niv + niv_range[None, :] - wn[:, None]]
        return LocalFourPoint(-beta * g_left_mat * g_right_mat, SpinChannel.NONE, 1, 1).filter_small_values()

    @staticmethod
    def create_generalized_chi0_q_fft(
        mpi_dist_irrk: MpiDistributor,
        giwk: GreensFunction,
        niw: int,
        niv: int,
        k_grid: KGrid,
        beta: float,
        use_gpu: bool = False,
        node_comm=None,
    ) -> FourPoint:
        r"""
        Returns the momentum-dependent generalized bare susceptibility :math:`\chi^{\mathrm{q}\nu}_{0;1234} = -\beta
        \sum_{\mathbf{k}} G^{\mathrm{k}}_{14}\, G^{\mathrm{k}-\mathrm{q}}_{32}`, evaluated via an FFT over the BZ with
        preallocated buffers. On a multi-rank CPU run the bosonic-frequency loop is distributed across all ranks (see
        :meth:`_create_generalized_chi0_q_fft_distributed`); on a single rank or on the GPU the whole bubble is computed
        on rank 0 over the irreducible BZ and scattered across ranks.

        :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
        :param giwk: The momentum-dependent :class:`GreensFunction`.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param k_grid: The :class:`KGrid` over which the BZ sum/FFT is performed.
        :param beta: Inverse temperature :math:`\beta`.
        :param use_gpu: If True, compute with CuPy on the GPU; otherwise with NumPy on the CPU.
        :param node_comm: Optional node-local communicator; when given, the distributed CPU path builds its R-space
            Green's functions once per node in shared-memory windows.
        :return: The bubble as a :class:`FourPoint` over the irreducible BZ (compressed momentum, half niw range).
        """
        if not use_gpu and mpi_dist_irrk.comm.size > 1:
            return BubbleGenerator._create_generalized_chi0_q_fft_distributed(
                mpi_dist_irrk, giwk, niw, niv, k_grid, beta, node_comm
            )

        if use_gpu:
            import cupy as xp

            def ifftn_kxyz(arr):
                return xp.fft.ifftn(arr, axes=(0, 1, 2))

        else:
            import numpy as xp

            def ifftn_kxyz(arr):
                # scipy.fft keeps complex64 in place; numpy.fft would upcast to complex128 (double work + 2x buffer)
                return sp.fft.ifftn(arr, axes=(0, 1, 2), overwrite_x=True)

        order = "F" if use_gpu else "C"

        gchi0_q_mat = None
        if mpi_dist_irrk.my_rank == 0:
            nb = giwk.n_bands
            wn = MFHelper.wn(niw, return_only_positive=True)

            g_k = giwk.cut_niv(niv + niw)
            g_r = g_k.fft().decompress_q_dimension()
            g_r_rev = g_r.flip_momentum_axis(copy=True).transpose_orbitals()

            gchi0_q_mat = xp.zeros(
                (len(k_grid.irrk_ind), nb, nb, nb, nb, len(wn), 2 * niv), dtype=g_r.mat.dtype, order=order
            )
            chi_r_v_buffer = xp.empty((*k_grid.nk, nb, nb, nb, nb, 2 * niv), dtype=g_r.mat.dtype, order=order)

            giwk_niv = g_r.current_shape[-1] // 2
            start = giwk_niv - niv
            end = giwk_niv + niv
            g_r = xp.asarray(g_r.mat[..., start:end], order=order)
            g_r_rev = xp.asarray(g_r_rev.mat, order=order)

            for iw, wn_i in enumerate(wn):
                g_vw = g_r_rev[..., start - wn_i : end - wn_i]
                xp.multiply(g_r[:, :, :, :, None, None, :, :], g_vw[:, :, :, None, :, :, None, :], out=chi_r_v_buffer)
                gchi0_q_mat[..., iw, :] = ifftn_kxyz(chi_r_v_buffer).reshape((k_grid.nk_tot, nb, nb, nb, nb, 2 * niv))[
                    k_grid.irrk_ind
                ]

            gchi0_q_mat *= -beta / k_grid.nk_tot
            if use_gpu:
                gchi0_q_mat = xp.asnumpy(gchi0_q_mat)
        gchi0_q_mat = mpi_dist_irrk.scatter(gchi0_q_mat)

        return FourPoint(
            gchi0_q_mat, SpinChannel.NONE, k_grid.nk, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
        ).filter_small_values()

    @staticmethod
    def _create_generalized_chi0_q_fft_distributed(
        mpi_dist_irrk: MpiDistributor,
        giwk: GreensFunction,
        niw: int,
        niv: int,
        k_grid: KGrid,
        beta: float,
        node_comm=None,
    ) -> FourPoint:
        r"""
        Distributed-CPU evaluation of :meth:`create_generalized_chi0_q_fft`: the flattened bosonic-fermionic
        frequency columns :math:`(\omega, \nu)` are split across the ranks, each rank runs the R-space multiply and
        the full-grid in-place ``scipy.fft.ifftn`` (bit-identical to the rank-0 path) only for its columns - reading
        the R-space Green's function and its momentum-flipped, orbital-transposed partner from per-node shared-memory
        windows - and the per-column irreducible-BZ results are ring-exchanged onto the irr-BZ q-distribution
        (the same total bytes the former rank-0 scatter moved). The columns are processed in sub-chunks sized so
        that a node's combined full-grid buffers stay well below the former rank-0 footprint; the exchange is
        pipelined per sub-chunk on a globally derived schedule, so no rank ever holds more than its final irr-BZ
        slice plus one sub-chunk in flight.

        :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
        :param giwk: The momentum-dependent :class:`GreensFunction`.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param k_grid: The :class:`KGrid` over which the BZ sum/FFT is performed.
        :param beta: Inverse temperature :math:`\beta`.
        :param node_comm: Optional node-local communicator for the shared R-space Green's functions.
        :return: The bubble as a :class:`FourPoint` over the irreducible BZ (compressed momentum, half niw range).
        """
        comm = mpi_dist_irrk.comm
        rank, size = comm.rank, comm.size
        nb = giwk.n_bands
        nk_tot, nk_irr = k_grid.nk_tot, len(k_grid.irrk_ind)
        wn = MFHelper.wn(niw, return_only_positive=True)
        vpos = 2 * niv

        def build_g_r() -> np.ndarray:
            return giwk.cut_niv(niv + niw).fft(copy=False).decompress_q_dimension().mat

        def build_g_r_rev() -> np.ndarray:
            # flip_momentum_axis + transpose_orbitals of the R-space Green's function, on the raw array
            return np.roll(np.flip(g_r_mat, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2)).swapaxes(3, 4)

        if node_comm is not None:
            g_r_mat, g_r_win = mpi_utils.build_node_shared_array(node_comm, build_g_r)
            g_r_rev_mat, g_r_rev_win = mpi_utils.build_node_shared_array(node_comm, build_g_r_rev)
        else:
            g_r_mat, g_r_win = build_g_r(), None
            g_r_rev_mat, g_r_rev_win = np.ascontiguousarray(build_g_r_rev()), None

        giwk_niv = g_r_mat.shape[-1] // 2
        start, end = giwk_niv - niv, giwk_niv + niv

        # globally derived column schedule: contiguous (w, v)-column runs per rank, processed in equal-size
        # sub-chunk rounds so the pipelined ring exchange stays collectively matched on every rank
        total_cols = len(wn) * vpos
        col_bounds = np.linspace(0, total_cols, size + 1).astype(int)
        ncols_max = int(np.max(np.diff(col_bounds)))
        sub_cols = max(1, ncols_max * nk_irr // (6 * nk_tot), min(ncols_max, vpos // max(1, size)))
        n_rounds = -(-ncols_max // sub_cols) if ncols_max else 0

        my_q_slice = mpi_dist_irrk.slices[rank]
        my_nq = (my_q_slice.stop - my_q_slice.start) if my_q_slice is not None else 0
        gchi0_q_mat = np.empty((my_nq, nb, nb, nb, nb, len(wn), vpos), dtype=g_r_mat.dtype)
        gchi0_cols = gchi0_q_mat.reshape(my_nq, nb**4, total_cols)  # contiguous view, columns w-major

        def round_cols(r: int, rnd: int) -> tuple[int, int]:
            c0 = col_bounds[r] + rnd * sub_cols
            return c0, min(col_bounds[r + 1], c0 + sub_cols)

        buf = np.empty((*k_grid.nk, nb, nb, nb, nb, sub_cols), dtype=g_r_mat.dtype)
        for rnd in range(n_rounds):
            c0, c1 = round_cols(rank, rnd)
            n_mine = max(0, c1 - c0)
            chunk_irr = None
            if n_mine:
                buf_v = buf[..., :n_mine]
                for col in range(c0, c1):  # fill column-wise: one (w, v) pair per column
                    iw, iv = divmod(col, vpos)
                    np.multiply(
                        g_r_mat[:, :, :, :, None, None, :, start + iv],
                        g_r_rev_mat[:, :, :, None, :, :, None, start + iv - wn[iw]],
                        out=buf_v[..., col - c0],
                    )
                chunk_irr = np.ascontiguousarray(
                    # scipy.fft keeps complex64 in place; numpy.fft would upcast to complex128 (double work + 2x buffer)
                    sp.fft.ifftn(buf_v, axes=(0, 1, 2), overwrite_x=True).reshape(nk_tot, nb, nb, nb, nb, n_mine)[
                        k_grid.irrk_ind
                    ]
                )
                if my_nq:
                    gchi0_cols[:, :, c0:c1] = chunk_irr[my_q_slice].reshape(my_nq, nb**4, n_mine)

            # ring exchange of this round's sub-chunks onto the irr-BZ q-distribution
            for step in range(1, size):
                dst, src = (rank + step) % size, (rank - step) % size
                reqs, staging = [], None

                dst_slice = mpi_dist_irrk.slices[dst]
                n_dst = (dst_slice.stop - dst_slice.start) if dst_slice is not None else 0
                if n_mine and n_dst:
                    reqs += mpi_utils._isend_rows(comm, chunk_irr[dst_slice], dst, base_tag=500 + step)

                s0, s1 = round_cols(src, rnd)
                if my_nq and s1 > s0:
                    staging = np.empty((my_nq, nb, nb, nb, nb, s1 - s0), dtype=gchi0_q_mat.dtype)
                    reqs += mpi_utils._irecv_rows_into(comm, staging, src, base_tag=500 + step)

                if reqs:
                    mpi_utils.MPI.Request.Waitall(reqs)
                if staging is not None:
                    gchi0_cols[:, :, s0:s1] = staging.reshape(my_nq, nb**4, s1 - s0)

        del buf
        g_r_mat = g_r_rev_mat = None
        for win in (g_r_win, g_r_rev_win):
            if win is not None:
                node_comm.Barrier()
                win.Free()

        gchi0_q_mat *= -beta / nk_tot
        return FourPoint(
            gchi0_q_mat, SpinChannel.NONE, k_grid.nk, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
        ).filter_small_values()

    @staticmethod
    def create_generalized_chi0_q_fft_auto(
        mpi_dist_irrk: MpiDistributor,
        giwk: GreensFunction,
        niw: int,
        niv: int,
        k_grid: KGrid,
        beta: float,
        logger: DgaLogger,
        node_comm=None,
    ):
        r"""
        Dispatches :meth:`create_generalized_chi0_q_fft` to the GPU when CuPy and a usable CUDA device are available
        (assigning one GPU per MPI rank round-robin), otherwise falls back to the CPU.

        :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
        :param giwk: The momentum-dependent :class:`GreensFunction`.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param k_grid: The :class:`KGrid` over which the BZ sum/FFT is performed.
        :param beta: Inverse temperature :math:`\beta`.
        :param logger: Logger used to report whether GPU acceleration is used.
        :param node_comm: Optional node-local communicator for the distributed CPU path (see
            :meth:`create_generalized_chi0_q_fft`).
        :return: The bubble as a :class:`FourPoint` over the irreducible BZ.
        """
        cp = None
        try:
            import cupy as cp
        except ImportError:
            pass  # CuPy not installed -> CPU

        n_gpus = 0
        if cp is not None:
            try:
                n_gpus = cp.cuda.runtime.getDeviceCount()
            except cp.cuda.runtime.CUDARuntimeError:
                n_gpus = 0  # no usable CUDA driver/device -> CPU

        if n_gpus > 0 and cp.cuda.is_available():
            logger.info(f"CuPy detected {n_gpus} GPU(s). Using GPU acceleration for gchi0_q calculation.")

            gpu_id = mpi_dist_irrk.my_rank % n_gpus
            cp.cuda.Device(gpu_id).use()
            return BubbleGenerator.create_generalized_chi0_q_fft(
                mpi_dist_irrk, giwk, niw, niv, k_grid, beta, use_gpu=True
            )

        return BubbleGenerator.create_generalized_chi0_q_fft(
            mpi_dist_irrk, giwk, niw, niv, k_grid, beta, use_gpu=False, node_comm=node_comm
        )

    @staticmethod
    def create_generalized_chi0_q(
        giwk: GreensFunction,
        niw: int,
        niv: int,
        q_list: np.ndarray,
        q_grid: KGrid,
        beta: float,
        use_gpu: bool = False,
    ) -> FourPoint:
        r"""
        Returns the momentum-dependent generalized bare susceptibility :math:`\chi^{\mathrm{q}\nu}_{0;1234} = -\beta
        \sum_{\mathbf{k}} G^{\mathrm{k}}_{14}\, G^{\mathrm{k}-\mathrm{q}}_{32}`, evaluated by a direct momentum-shift
        and a fused einsum over the explicit list of q-points (preallocated buffers). The production bubble always
        runs the FFT evaluation; this direct form is its independent cross-check and the evaluator for explicit
        q-subsets.

        :param giwk: The momentum-dependent :class:`GreensFunction`.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param q_list: Array of integer q-point index triplets to compute.
        :param q_grid: The :class:`KGrid` providing the momentum normalization (``nk_tot``).
        :param beta: Inverse temperature :math:`\beta`.
        :param use_gpu: If True, compute with CuPy on the GPU; otherwise with NumPy on the CPU.
        :return: The bubble as a :class:`FourPoint` over the given q-points (compressed momentum, half niw range).
        """
        if use_gpu:
            import cupy as xp
        else:
            import numpy as xp

        order = "F" if use_gpu else "C"

        wn = MFHelper.wn(niw, return_only_positive=True)
        nb = giwk.n_bands
        nq = len(q_list)

        gchi0_q = xp.zeros((nq, nb, nb, nb, nb, len(wn), 2 * niv), dtype=giwk.mat.dtype, order=order)

        # g_left (the central niv window) and g_right (the full range, momentum-shifted per q) are both read-only,
        # so they can share one backing array instead of holding a full-size duplicate.
        g_full = xp.asarray(giwk.cut_niv(niv + niw).mat, order=order)
        giwk_niv = g_full.shape[-1] // 2

        g_r_buf = xp.empty_like(g_full)
        g_right = g_full
        g_left = g_full[..., giwk_niv - niv : giwk_niv + niv]

        # the precomputed contraction path is only used on the CPU; cupy.einsum optimizes internally
        path = True if use_gpu else xp.einsum_path("xyzadv,xyzcbv->abcdv", g_left, g_left, optimize="optimal")[0]
        kxs, kys, kzs = xp.arange(g_right.shape[0]), xp.arange(g_right.shape[1]), xp.arange(g_right.shape[2])

        for iq, q in enumerate(q_list):
            g_r_buf[...] = xp.take(g_right, (kxs - q[0]) % g_right.shape[0], axis=0)
            g_r_buf[...] = xp.take(g_r_buf, (kys - q[1]) % g_right.shape[1], axis=1)
            g_r_buf[...] = xp.take(g_r_buf, (kzs - q[2]) % g_right.shape[2], axis=2)

            for iw, wn_i in enumerate(wn):
                s = giwk_niv - niv - wn_i
                e = giwk_niv + niv - wn_i
                gchi0_q[iq, ..., iw, :] = xp.einsum("xyzadv,xyzcbv->abcdv", g_left, g_r_buf[..., s:e], optimize=path)

        gchi0_q *= -beta / q_grid.nk_tot
        if use_gpu:
            gchi0_q = xp.asnumpy(gchi0_q)
        return FourPoint(
            gchi0_q, SpinChannel.NONE, q_grid.nk, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
        ).filter_small_values()

    @staticmethod
    def create_generalized_chi0_pp_w0(g_dmft: GreensFunction, niv_pp: int, beta: float) -> LocalFourPoint:
        r"""
        Returns the local particle-particle bare bubble at :math:`\omega = 0`,
        :math:`\chi_{0;1234}^{\nu} = -\beta\, G_{14}^{\nu}\, G_{32}^{-\nu}`.

        :param g_dmft: The local (DMFT) :class:`GreensFunction`.
        :param niv_pp: Number of positive fermionic frequencies of the pp bubble.
        :param beta: Inverse temperature :math:`\beta`.
        :return: The local pp bubble as a :class:`LocalFourPoint` in pp notation at :math:`\omega = 0`.
        """
        g = g_dmft.cut_niv(niv_pp)
        # transpose_orbitals() returns a fresh private copy, so flip it in place (copy=False) instead of deepcopying
        # that throwaway again.
        gchi0_pp_w0 = (
            g.mat[0, 0, 0][:, None, None, :, :]
            * g.transpose_orbitals().flip_frequency_axis(-1, copy=False).mat[0, 0, 0][None, :, :, None, :]
        )
        return LocalFourPoint(
            -beta * gchi0_pp_w0[..., None, :], SpinChannel.NONE, 1, 1, frequency_notation=FrequencyNotation.PP
        ).filter_small_values()

    @staticmethod
    def create_generalized_chi0_q_pp_w0(giwk: GreensFunction, niv_pp: int, q_grid: KGrid) -> FourPoint:
        r"""
        Returns the momentum-dependent particle-particle bare bubble at :math:`\omega = 0`,
        :math:`\chi^{\mathrm{k}}_{0;1234} = G^{\mathrm{k}}_{14}\, G^{-\mathrm{k}}_{23}` with :math:`G^{-\mathrm{k}}_{23}
        = (G^{\mathrm{k}}_{32})^{*}`.
        Note that no factor of :math:`-\beta` is included here.

        :param giwk: The momentum-dependent :class:`GreensFunction`.
        :param niv_pp: Number of positive fermionic frequencies of the pp bubble.
        :param q_grid: The :class:`KGrid` defining the momentum grid.
        :return: The momentum-dependent pp bubble as a :class:`FourPoint` (no bosonic axis, pp notation, compressed q).
        """
        g = giwk.cut_niv(niv_pp).compress_q_dimension()
        # transpose_orbitals() returns a fresh private copy, so conjugate it in place to reuse its buffer.
        g_t = g.transpose_orbitals()
        np.conj(g_t.mat, out=g_t.mat)
        gchi0_q_pp_w0 = g.mat[:, :, None, None, :, :] * g_t.mat[:, None, :, :, None, :]

        return FourPoint(
            gchi0_q_pp_w0, SpinChannel.NONE, q_grid.nk, 0, 1, True, True, True, FrequencyNotation.PP
        ).filter_small_values()

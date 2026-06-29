# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Generalized bare susceptibilities (the "bubbles"). :class:`BubbleGenerator` builds the products of two Green's
functions :math:`\chi_{0;abcd} = -\beta\, G_{ad}\, G_{cb}` in the particle-hole and particle-particle channels,
both local and momentum-dependent. The non-local versions are evaluated either by an FFT over the BZ or by a
direct momentum-shift einsum, distributed over MPI ranks and optionally accelerated on the GPU (CuPy).
"""

import numpy as np

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
        :math:`\chi_{0;abcd}^{\omega\nu} = -\beta\, G_{ad}^{\nu}\, G_{cb}^{\nu-\omega}`.

        :param g_dmft: The local (DMFT) :class:`GreensFunction`.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param beta: Inverse temperature :math:`\beta`.
        :return: The local bubble as a :class:`LocalFourPoint` with one bosonic and one fermionic frequency axis.
        """
        wn = MFHelper.wn(niw)
        niv_range = np.arange(-niv, niv)

        g_left_mat = g_dmft.mat[:, None, None, :, None, g_dmft.niv - niv : g_dmft.niv + niv]
        g_right_mat = g_dmft.transpose_orbitals().mat[None, :, :, None, g_dmft.niv + niv_range[None, :] - wn[:, None]]
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
    ) -> FourPoint:
        r"""
        Returns the momentum-dependent generalized bare susceptibility
        :math:`\chi_{0;abcd}^{q\nu} = -\beta \sum_k G^{k}_{ad}\, G^{k-q}_{cb}`, evaluated via an FFT over the BZ with
        preallocated buffers. The result is computed on rank 0 over the irreducible BZ and scattered across ranks.

        :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
        :param giwk: The momentum-dependent :class:`GreensFunction`.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param k_grid: The :class:`KGrid` over which the BZ sum/FFT is performed.
        :param beta: Inverse temperature :math:`\beta`.
        :param use_gpu: If True, compute with CuPy on the GPU; otherwise with NumPy on the CPU.
        :return: The bubble as a :class:`FourPoint` over the irreducible BZ (compressed momentum, half niw range).
        """
        if use_gpu:
            import cupy as xp
        else:
            import numpy as xp

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
                gchi0_q_mat[..., iw, :] = xp.fft.ifftn(chi_r_v_buffer, axes=(0, 1, 2)).reshape(
                    (k_grid.nk_tot, nb, nb, nb, nb, 2 * niv)
                )[k_grid.irrk_ind]

            gchi0_q_mat *= -beta / k_grid.nk_tot
            if use_gpu:
                gchi0_q_mat = xp.asnumpy(gchi0_q_mat)
        gchi0_q_mat = mpi_dist_irrk.scatter(gchi0_q_mat)

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

        return BubbleGenerator.create_generalized_chi0_q_fft(mpi_dist_irrk, giwk, niw, niv, k_grid, beta, use_gpu=False)

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
        Returns the momentum-dependent generalized bare susceptibility
        :math:`\chi_{0;abcd}^{q\nu} = -\beta \sum_k G^{k}_{ad}\, G^{k-q}_{cb}`, evaluated by a direct momentum-shift
        and a fused einsum over the explicit list of q-points (preallocated buffers).

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
    def create_generalized_chi0_q_auto(
        mpi_distributor: MpiDistributor,
        giwk: GreensFunction,
        niw: int,
        niv: int,
        q_list: np.ndarray,
        q_grid: KGrid,
        beta: float,
        logger: DgaLogger,
    ):
        r"""
        Dispatches :meth:`create_generalized_chi0_q` to the GPU when CuPy and a usable CUDA device are available
        (assigning one GPU per MPI rank round-robin), otherwise falls back to the CPU.

        :param mpi_distributor: MPI distributor over the q-points (see :class:`MpiDistributor`).
        :param giwk: The momentum-dependent :class:`GreensFunction`.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param q_list: Array of integer q-point index triplets to compute.
        :param q_grid: The :class:`KGrid` providing the momentum normalization.
        :param beta: Inverse temperature :math:`\beta`.
        :param logger: Logger used to report whether GPU acceleration is used.
        :return: The bubble as a :class:`FourPoint` over the given q-points.
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

            gpu_id = mpi_distributor.my_rank % n_gpus
            cp.cuda.Device(gpu_id).use()
            return BubbleGenerator.create_generalized_chi0_q(giwk, niw, niv, q_list, q_grid, beta, use_gpu=True)

        return BubbleGenerator.create_generalized_chi0_q(giwk, niw, niv, q_list, q_grid, beta, use_gpu=False)

    @staticmethod
    def create_generalized_chi0_pp_w0(g_dmft: GreensFunction, niv_pp: int, beta: float) -> LocalFourPoint:
        r"""
        Returns the local particle-particle bare bubble at :math:`\omega = 0`,
        :math:`\chi_{0;abcd}^{\nu} = -\beta\, G_{ad}^{\nu}\, G_{cb}^{-\nu}`.

        :param g_dmft: The local (DMFT) :class:`GreensFunction`.
        :param niv_pp: Number of positive fermionic frequencies of the pp bubble.
        :param beta: Inverse temperature :math:`\beta`.
        :return: The local pp bubble as a :class:`LocalFourPoint` in pp notation at :math:`\omega = 0`.
        """
        g = g_dmft.cut_niv(niv_pp)
        # transpose_orbitals() returns a fresh private copy, so flip it in place (copy=False) instead of deepcopying
        # that throwaway again.
        gchi0_pp_w0 = (
            g.mat[:, None, None, :, :]
            * g.transpose_orbitals().flip_frequency_axis(-1, copy=False).mat[None, :, :, None, :]
        )
        return LocalFourPoint(
            -beta * gchi0_pp_w0[..., None, :], SpinChannel.NONE, 1, 1, frequency_notation=FrequencyNotation.PP
        ).filter_small_values()

    @staticmethod
    def create_generalized_chi0_q_pp_w0(giwk: GreensFunction, niv_pp: int, q_grid: KGrid) -> FourPoint:
        r"""
        Returns the momentum-dependent particle-particle bare bubble at :math:`\omega = 0`,
        :math:`\chi_{0;abcd}^{\vec{k}(\omega=0)\nu} = G_{ad}^{k}\, G_{bc}^{-k}` with :math:`G_{bc}^{-k} = G_{cb}^{*k}`.
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

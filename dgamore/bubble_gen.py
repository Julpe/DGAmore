# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import numpy as np

from dgamore.brillouin_zone import KGrid
from dgamore.dga_logger import DgaLogger
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction
from dgamore.local_four_point import LocalFourPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore.mpi_distributor import MpiDistributor
from dgamore.n_point_base import SpinChannel, FrequencyNotation


class BubbleGenerator:
    @staticmethod
    def create_generalized_chi0(g_dmft: GreensFunction, niw: int, niv: int, beta: float) -> LocalFourPoint:
        r"""
        Returns the generalized bare susceptibility :math:`\chi_{0;abcd}^{wv} = -\beta G_{ad}^{v} G_{cb}^{v-w}`.
        """
        wn = MFHelper.wn(niw)
        niv_range = np.arange(-niv, niv)

        g_left_mat = g_dmft.mat[:, None, None, :, None, g_dmft.niv - niv : g_dmft.niv + niv]
        g_right_mat = g_dmft.transpose_orbitals().mat[None, :, :, None, g_dmft.niv + niv_range[None, :] - wn[:, None]]
        return LocalFourPoint(-beta * g_left_mat * g_right_mat, SpinChannel.NONE, 1, 1).filter_small_values()

    @staticmethod
    def create_generalized_chi0_q_fft_cpu(
        mpi_dist_irrk: MpiDistributor, giwk: GreensFunction, niw: int, niv: int, k_grid: KGrid, beta: float
    ) -> FourPoint:
        """
        Returns χ₀^{qν}_{abcd} = -β ∑ₖ G^{k}_{ad} G^{k-q}_{cb}
        FFT-based CPU version with preallocated buffers.
        """
        gchi0_q_mat = None
        if mpi_dist_irrk.my_rank == 0:
            nb = giwk.n_bands
            wn = MFHelper.wn(niw, return_only_positive=True)

            g_k = giwk.cut_niv(niv + niw)
            g_r = g_k.fft().decompress_q_dimension()
            g_r_rev = g_r.flip_momentum_axis(copy=True).transpose_orbitals()

            gchi0_q_mat = np.zeros((len(k_grid.irrk_ind), nb, nb, nb, nb, len(wn), 2 * niv), dtype=g_r.mat.dtype)
            chi_r_v_buffer = np.empty((*k_grid.nk, nb, nb, nb, nb, 2 * niv), dtype=g_r.mat.dtype)

            giwk_niv = g_r.current_shape[-1] // 2
            start = giwk_niv - niv
            end = giwk_niv + niv
            g_r = g_r[..., start:end]

            for iw, wn_i in enumerate(wn):
                g_vw = g_r_rev[..., start - wn_i : end - wn_i]
                np.multiply(g_r[:, :, :, :, None, None, :, :], g_vw[:, :, :, None, :, :, None, :], out=chi_r_v_buffer)
                gchi0_q_mat[..., iw, :] = np.fft.ifftn(chi_r_v_buffer, axes=(0, 1, 2)).reshape(
                    (k_grid.nk_tot, nb, nb, nb, nb, 2 * niv)
                )[k_grid.irrk_ind]

            gchi0_q_mat *= -beta / k_grid.nk_tot
        gchi0_q_mat = mpi_dist_irrk.scatter(gchi0_q_mat)

        return FourPoint(
            gchi0_q_mat, SpinChannel.NONE, k_grid.nk, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
        ).filter_small_values()

    @staticmethod
    def create_generalized_chi0_q_fft_gpu(
        mpi_dist_irrk: MpiDistributor, giwk: GreensFunction, niw: int, niv: int, k_grid: KGrid, beta: float
    ) -> FourPoint:
        """
        Returns χ₀^{qν}_{abcd} = -β ∑ₖ G^{k}_{ad} G^{k-q}_{cb}
        FFT-based GPU version with preallocated buffers.
        """
        import cupy as cp

        gchi0_q_mat = None
        if mpi_dist_irrk.my_rank == 0:
            nb = giwk.n_bands
            wn = MFHelper.wn(niw, return_only_positive=True)

            g_k = giwk.cut_niv(niv + niw)
            g_r = g_k.fft().decompress_q_dimension()
            g_r_rev = g_r.flip_momentum_axis(copy=True).transpose_orbitals()

            gchi0_q_mat = cp.asarray(
                np.zeros((len(k_grid.irrk_ind), nb, nb, nb, nb, len(wn), 2 * niv), dtype=g_r.mat.dtype), order="F"
            )
            chi_r_v_buffer = cp.asarray(np.empty((*k_grid.nk, nb, nb, nb, nb, 2 * niv), dtype=g_r.mat.dtype), order="F")

            giwk_niv = g_r.current_shape[-1] // 2
            start = giwk_niv - niv
            end = giwk_niv + niv
            g_r = cp.asarray(g_r.mat[..., start:end], order="F")
            g_r_rev = cp.asarray(g_r_rev.mat, order="F")

            for iw, wn_i in enumerate(wn):
                g_vw = g_r_rev[..., start - wn_i : end - wn_i]
                cp.multiply(g_r[:, :, :, :, None, None, :, :], g_vw[:, :, :, None, :, :, None, :], out=chi_r_v_buffer)
                gchi0_q_mat[..., iw, :] = cp.fft.ifftn(chi_r_v_buffer, axes=(0, 1, 2)).reshape(
                    (k_grid.nk_tot, nb, nb, nb, nb, 2 * niv)
                )[k_grid.irrk_ind]

            gchi0_q_mat *= -beta / k_grid.nk_tot
            gchi0_q_mat = cp.asnumpy(gchi0_q_mat)
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
        """
        Automatically uses GPU if available, otherwise CPU.
        """
        try:
            import cupy as cp

            n_gpus = cp.cuda.runtime.getDeviceCount()

            if cp.cuda.is_available() and n_gpus > 0:
                logger.info(f"CuPy detected {n_gpus} GPU(s). Using GPU acceleration for gchi0_q calculation.")

                gpu_id = mpi_dist_irrk.my_rank % n_gpus
                cp.cuda.Device(gpu_id).use()
                return BubbleGenerator.create_generalized_chi0_q_fft_gpu(mpi_dist_irrk, giwk, niw, niv, k_grid, beta)
        except:
            # CuPy not installed or device could not be found
            pass

        return BubbleGenerator.create_generalized_chi0_q_fft_cpu(mpi_dist_irrk, giwk, niw, niv, k_grid, beta)

    @staticmethod
    def create_generalized_chi0_q_cpu(
        giwk: GreensFunction, niw: int, niv: int, q_list: np.ndarray, q_grid: KGrid, beta: float
    ) -> FourPoint:
        """
        Returns χ₀^{qν}_{abcd} = -β ∑ₖ G^{k}_{ad} G^{k-q}_{cb}
        Optimized CPU version with preallocated buffers.
        """
        wn = MFHelper.wn(niw, return_only_positive=True)
        nb = giwk.n_bands
        nq = len(q_list)

        gchi0_q = np.zeros((nq, nb, nb, nb, nb, len(wn), 2 * niv), dtype=giwk.mat.dtype)

        g_left = giwk.cut_niv(niv + niw).mat
        g_right = giwk.transpose_orbitals().cut_niv(niv + niw).mat
        giwk_niv = g_right.shape[-1] // 2

        g_r_buf = np.empty_like(g_left)
        g_left = g_left[..., giwk_niv - niv : giwk_niv + niv]

        path, _ = np.einsum_path("xyzadv,xyzcbv->abcdv", g_left, g_left, optimize="optimal")
        kxs, kys, kzs = np.arange(g_right.shape[0]), np.arange(g_right.shape[1]), np.arange(g_right.shape[2])

        for iq, q in enumerate(q_list):
            g_r_buf[...] = np.take(g_right, (kxs - q[0]) % g_right.shape[0], axis=0)
            g_r_buf[...] = np.take(g_r_buf, (kys - q[1]) % g_right.shape[1], axis=1)
            g_r_buf[...] = np.take(g_r_buf, (kzs - q[2]) % g_right.shape[2], axis=2)

            for iw, wn_i in enumerate(wn):
                s = giwk_niv - niv - wn_i
                e = giwk_niv + niv - wn_i
                gchi0_q[iq, ..., iw, :] = np.einsum("xyzadv,xyzcbv->abcdv", g_left, g_r_buf[..., s:e], optimize=path)

        gchi0_q *= -beta / q_grid.nk_tot
        return FourPoint(
            gchi0_q, SpinChannel.NONE, q_grid.nk, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
        ).filter_small_values()

    @staticmethod
    def create_generalized_chi0_q_gpu(
        giwk: GreensFunction, niw: int, niv: int, q_list: np.ndarray, q_grid: KGrid, beta: float
    ) -> FourPoint:
        """
        GPU version of χ₀^{qν}_{abcd} with preallocated buffers and fused einsum.
        """
        import cupy as cp

        wn = MFHelper.wn(niw, return_only_positive=True)
        nb = giwk.n_bands
        nq = len(q_list)

        gchi0_q = cp.zeros((nq, nb, nb, nb, nb, len(wn), 2 * niv), dtype=giwk.mat.dtype, order="F")

        g_left = cp.asarray(giwk.cut_niv(niv + niw).mat, order="F")
        g_right = cp.asarray(giwk.transpose_orbitals().cut_niv(niv + niw).mat, order="F")
        giwk_niv = g_right.shape[-1] // 2

        g_r_buf = cp.empty_like(g_left)
        g_left = g_left[..., giwk_niv - niv : giwk_niv + niv]

        kxs, kys, kzs = cp.arange(g_right.shape[0]), cp.arange(g_right.shape[1]), cp.arange(g_right.shape[2])

        for iq, q in enumerate(q_list):
            g_r_buf[...] = cp.take(g_right, (kxs - q[0]) % g_right.shape[0], axis=0)
            g_r_buf[...] = cp.take(g_r_buf, (kys - q[1]) % g_right.shape[1], axis=1)
            g_r_buf[...] = cp.take(g_r_buf, (kzs - q[2]) % g_right.shape[2], axis=2)

            for iw, wn_i in enumerate(wn):
                s = giwk_niv - niv - wn_i
                e = giwk_niv + niv - wn_i
                gchi0_q[iq, ..., iw, :] = cp.einsum("xyzadv,xyzcbv->abcdv", g_left, g_r_buf[..., s:e], optimize=True)

        gchi0_q *= -beta / q_grid.nk_tot
        return FourPoint(
            cp.asnumpy(gchi0_q), SpinChannel.NONE, q_grid.nk, 1, 1, False, True, True
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
        """
        Automatically uses GPU if available, otherwise CPU.
        """
        try:
            import cupy as cp

            n_gpus = cp.cuda.runtime.getDeviceCount()

            if cp.cuda.is_available() and n_gpus > 0:
                logger.info(f"CuPy detected {n_gpus} GPU(s). Using GPU acceleration for gchi0_q calculation.")

                gpu_id = mpi_distributor.my_rank % n_gpus
                cp.cuda.Device(gpu_id).use()
                return BubbleGenerator.create_generalized_chi0_q_gpu(giwk, niw, niv, q_list, q_grid, beta)
        except:
            # CuPy not installed or device could not be found
            pass

        return BubbleGenerator.create_generalized_chi0_q_cpu(giwk, niw, niv, q_list, q_grid, beta)

    @staticmethod
    def create_generalized_chi0_pp_w0(g_dmft: GreensFunction, niv_pp: int, beta: float) -> LocalFourPoint:
        r"""
        Returns the particle-particle bare bubble susceptibility from the Green's function. Returns the object with :math:`\omega = 0`.
        We have :math:`\chi_{0;abcd}^{\nu} = -\beta * G_{ad}^\nu * G_{bc}^{-\nu}`, where :math:`G_{bc}^{-\nu}=G_{cb}^{*\nu}`.
        """
        g = g_dmft.cut_niv(niv_pp)
        gchi0_pp_w0 = g.mat[:, None, None, :, :] * np.conj(g.transpose_orbitals().mat)[None, :, :, None, :]
        return LocalFourPoint(
            -beta * gchi0_pp_w0[..., None, :], SpinChannel.NONE, 1, 1, frequency_notation=FrequencyNotation.PP
        ).filter_small_values()

    @staticmethod
    def create_generalized_chi0_q_pp_w0(giwk: GreensFunction, niv_pp: int, q_grid: KGrid) -> FourPoint:
        r"""
        Returns the particle-particle bare bubble susceptibility from the Green's function. Returns the object with :math:`\omega = 0`.
        We have :math:`\chi_{0;abcd}^{\vec{k}(\omega=0)\nu} = G_{ad}^k * G_{bc}^{-k}` with :math:`G_{bc}^{-k} = G_{cb}^{*k}`. Attention:
        no factor of :math:`-\beta` is included here.
        """
        g = giwk.cut_niv(niv_pp).compress_q_dimension()
        gchi0_q_pp_w0 = g.mat[:, :, None, None, :, :] * np.conj(g.transpose_orbitals().mat)[:, None, :, :, None, :]

        return FourPoint(
            gchi0_q_pp_w0, SpinChannel.NONE, q_grid.nk, 0, 1, True, True, True, FrequencyNotation.PP
        ).filter_small_values()

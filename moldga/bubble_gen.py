import moldga.config as config
from moldga.greens_function import GreensFunction
from moldga.matsubara_frequencies import MFHelper
from moldga.local_four_point import LocalFourPoint
from moldga.four_point import FourPoint
from moldga.n_point_base import SpinChannel, FrequencyNotation
import numpy as np


class BubbleGenerator:
    @staticmethod
    def create_generalized_chi0(g_loc: GreensFunction, niw: int, niv: int) -> LocalFourPoint:
        r"""
        Returns the generalized bare susceptibility :math:`\chi_{0;abcd}^{wv} = -\beta G_{ad}^{v} G_{cb}^{v-w}`.
        """
        wn = MFHelper.wn(niw)
        niv_range = np.arange(-niv, niv)
        eye_bands = np.eye(g_loc.n_bands)

        g_left_mat = (
            g_loc.mat[:, None, None, :, None, g_loc.niv - niv : g_loc.niv + niv]
            * eye_bands[None, :, :, None, None, None]
        )
        g_right_mat = (
            g_loc.transpose_orbitals().mat[None, :, :, None, g_loc.niv + niv_range[None, :] - wn[:, None]]
            * eye_bands[:, None, None, :, None, None]
        )
        return LocalFourPoint(-config.sys.beta * g_left_mat * g_right_mat, SpinChannel.NONE, 1, 1)

    @staticmethod
    def create_generalized_chi0_q(giwk: GreensFunction, niw: int, niv: int, q_list: np.ndarray) -> FourPoint:
        r"""
        Returns :math:`\chi^{q\nu}_{0;abcd} = -\beta \sum_{k} G^{k}_{ad} * G^{k-q}_{cb}`.
        """
        wn = MFHelper.wn(niw, return_only_positive=True)
        gchi0_q = np.zeros((len(q_list),) + (giwk.n_bands,) * 4 + (len(wn), 2 * niv), dtype=giwk.mat.dtype)

        eye_bands = np.eye(giwk.n_bands)
        g_right = giwk.transpose_orbitals().cut_niv(niv + niw)
        g_left_mat = (
            giwk.cut_niv(niv + niw).mat[:, :, :, :, None, None, :, g_right.niv - niv : g_right.niv + niv]
            * eye_bands[None, None, None, None, :, :, None, None]
        )
        for idx_q, q in enumerate(q_list):
            g_right_mat = (
                np.roll(g_right.mat, [-i for i in q], axis=(0, 1, 2))[:, :, :, None, :, :, None, :]
                * eye_bands[None, None, None, :, None, None, :, None]
            )

            for idx_w, wn_i in enumerate(wn):
                start = g_right.niv - niv - wn_i
                end = g_right.niv + niv - wn_i
                gchi0_q[idx_q, ..., idx_w, :] = np.sum(g_left_mat * g_right_mat[..., start:end], axis=(0, 1, 2))

        gchi0_q *= -config.sys.beta / config.lattice.q_grid.nk_tot
        return FourPoint(
            gchi0_q, SpinChannel.NONE, config.lattice.nq, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
        )

    @staticmethod
    def create_generalized_chi0_q_cpu(giwk: GreensFunction, niw: int, niv: int, q_list: np.ndarray) -> FourPoint:
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

        gchi0_q *= -config.sys.beta / config.lattice.q_grid.nk_tot
        return FourPoint(
            gchi0_q, SpinChannel.NONE, config.lattice.nq, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
        )

    @staticmethod
    def create_generalized_chi0_q_gpu(giwk: GreensFunction, niw: int, niv: int, q_list: np.ndarray) -> FourPoint:
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
                gchi0_q[iq, ..., iw, :] += cp.einsum("xyzadv,xyzcbv->abcdv", g_left, g_r_buf[..., s:e], optimize=True)

        gchi0_q *= -config.sys.beta / config.lattice.q_grid.nk_tot
        return FourPoint(
            cp.asnumpy(gchi0_q),
            SpinChannel.NONE,
            config.lattice.nq,
            1,
            1,
            full_niw_range=False,
            has_compressed_q_dimension=True,
        )

    @staticmethod
    def create_generalized_chi0_q_auto(mpi_distributor, giwk, niw, niv, q_list):
        """
        Automatically uses GPU if available, otherwise CPU.
        """
        logger = config.logger

        try:
            import cupy as cp

            n_gpus = cp.cuda.runtime.getDeviceCount()

            if cp.cuda.is_available() and n_gpus > 0:
                logger.info(f"CuPy detected {n_gpus} GPU(s). Using GPU acceleration for gchi0_q calculation.")

                gpu_id = mpi_distributor.my_rank % n_gpus
                cp.cuda.Device(gpu_id).use()
                return BubbleGenerator.create_generalized_chi0_q_gpu(giwk, niw, niv, q_list)
        except ImportError:
            pass

        return BubbleGenerator.create_generalized_chi0_q_cpu(giwk, niw, niv, q_list)

    @staticmethod
    def create_generalized_chi0_pp_w0(g_loc: GreensFunction, niv_pp: int) -> LocalFourPoint:
        r"""
        Returns the particle-particle bare bubble susceptibility from the Green's function. Returns the object with :math:`\omega = 0`.
        We have :math:`\chi_{0;abcd}^{\vec{k}\nu} = -\beta * G_{ad}^k * G_{cb}^{\omega-k}`.
        """
        g = g_loc.cut_niv(niv_pp)
        eye_bands = np.eye(config.sys.n_bands)
        gchi0_pp_w0 = (g.mat[:, None, None, :, :] * eye_bands[None, :, :, None, None]) * (
            np.conj(g.mat)[None, :, :, None, :] * eye_bands[:, None, None, :, None]
        )
        return LocalFourPoint(
            -config.sys.beta * gchi0_pp_w0[..., None, :],
            SpinChannel.NONE,
            1,
            1,
            frequency_notation=FrequencyNotation.PP,
        )

    @staticmethod
    def create_generalized_chi0_q_pp_w0(giwk: GreensFunction, niv_pp: int) -> FourPoint:
        r"""
        Returns the particle-particle bare bubble susceptibility from the Green's function. Returns the object with :math:`\omega = 0`.
        We have :math:`\chi_{0;abcd}^{\vec{k}(\omega=0)\nu} = G_{ad}^k * G_{cb}^{-k}` with :math:`G_{cb}^{-k} = G_{bc}^{*k}`. Attention:
        no factor of :math:`-\beta` is included here.
        """
        g = giwk.cut_niv(niv_pp).compress_q_dimension()
        eye_bands = np.eye(g.n_bands)
        gchi0_q = (g.mat[:, :, None, None, :, :] * eye_bands[None, None, :, :, None, None]) * (
            np.conj(g.mat)[:, None, :, :, None, :] * eye_bands[None, :, None, None, :, None]
        )
        return FourPoint(
            gchi0_q,
            SpinChannel.NONE,
            config.lattice.nq,
            0,
            1,
            has_compressed_q_dimension=True,
            frequency_notation=FrequencyNotation.PP,
        )

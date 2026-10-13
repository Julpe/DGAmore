# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Generalized bare susceptibilities (the "bubbles"). :class:`BubbleGenerator` builds the products of two Green's
functions :math:`\chi_{0;1234}^{\omega\nu} = -\beta\, G_{14}^{\nu}\, G_{32}^{\nu-\omega}` in the particle-hole and particle-particle channels,
both local and momentum-dependent. The non-local versions are evaluated either by an FFT over the BZ or by a
direct momentum-shift einsum, distributed over MPI ranks.
"""

from collections.abc import Callable, Sequence

import numpy as np
import scipy as sp

import dgamore.mpi_utils as mpi_utils
from dgamore.brillouin_zone import KGrid
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction
from dgamore.local_four_point import LocalFourPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel, FrequencyNotation, zero_small_values

# bytes of one sub-chunk round's full-grid buffer in the distributed bubble: about one rank's L3 share
ROUND_BUFFER_BYTES = 4 * 1024**2


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
        node_comm=None,
        subtract_from: FourPoint | None = None,
    ) -> FourPoint:
        r"""
        Returns the momentum-dependent generalized bare susceptibility :math:`\chi^{\mathrm{q}\nu}_{0;1234} = -\beta
        \sum_{\mathbf{k}} G^{\mathrm{k}}_{14}\, G^{\mathrm{k}-\mathrm{q}}_{32}`, evaluated via an FFT over the BZ with
        preallocated buffers. On a multi-rank run the bosonic-frequency loop is distributed across all ranks (see
        :meth:`_create_generalized_chi0_q_fft_distributed`); on a single rank the whole bubble is computed on rank 0
        over the irreducible BZ and scattered across ranks. With ``subtract_from`` the bubble is subtracted from that
        bubble in place instead of returned: the multi-rank path scales, filters and subtracts every column block as
        it arrives, so the bubble's own array is never built; a single rank subtracts the finished bubble.

        :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
        :param giwk: The momentum-dependent :class:`GreensFunction`.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param k_grid: The :class:`KGrid` over which the BZ sum/FFT is performed.
        :param beta: Inverse temperature :math:`\beta`.
        :param node_comm: Optional node-local communicator; when given, the distributed path builds its R-space
            Green's functions once per node in shared-memory windows.
        :param subtract_from: A bubble of the same box on this rank's irreducible q-points that the bubble is
            subtracted from in place, or ``None``.
        :return: The bubble as a :class:`FourPoint` over the irreducible BZ (compressed momentum, half niw range), or
            ``subtract_from`` minus the bubble.
        """
        if mpi_dist_irrk.comm.size > 1:
            return BubbleGenerator._create_generalized_chi0_q_fft_distributed(
                mpi_dist_irrk, giwk, niw, niv, k_grid, beta, node_comm, subtract_from
            )

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
            g_r = np.asarray(g_r.mat[..., start:end], order="C")
            g_r_rev = np.asarray(g_r_rev.mat, order="C")

            for iw, wn_i in enumerate(wn):
                g_vw = g_r_rev[..., start - wn_i : end - wn_i]
                np.multiply(g_r[:, :, :, :, None, None, :, :], g_vw[:, :, :, None, :, :, None, :], out=chi_r_v_buffer)
                # scipy.fft keeps complex64 in place; numpy.fft would upcast to complex128 (double work + 2x buffer)
                chi_q_v = sp.fft.ifftn(chi_r_v_buffer, axes=(0, 1, 2), overwrite_x=True)
                gchi0_q_mat[..., iw, :] = chi_q_v.reshape((k_grid.nk_tot, nb, nb, nb, nb, 2 * niv))[k_grid.irrk_ind]

            gchi0_q_mat *= -beta / k_grid.nk_tot
        gchi0_q_mat = mpi_dist_irrk.scatter(gchi0_q_mat)

        bubble = FourPoint(
            gchi0_q_mat, SpinChannel.NONE, k_grid.nk, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
        ).filter_small_values()
        return bubble if subtract_from is None else subtract_from.sub(bubble, copy=False)

    @staticmethod
    def create_generalized_chi0_q_fft_window(
        mpi_dist_irrk: MpiDistributor,
        g_r: np.ndarray,
        niv: int,
        k_grid: KGrid,
        beta: float,
        w_window: tuple[int, int],
        dg_rs: Sequence[np.ndarray] = (),
        with_forward: bool = True,
    ) -> list[FourPoint]:
        r"""
        Returns the momentum-dependent bubbles of the stored bosonic frequencies :math:`\omega = w_0, \ldots, w_1 - 1`
        of the half niw range, ``w_window = (w0, w1)``, from frequency-first real-space windows built beforehand (see
        :func:`~dgamore.nonlocal_sde._build_rspace_giwk_window`), on any rank count (collective; every rank passes the
        same window and as many changes). With :math:`\chi_{0}[A, B]^{\mathrm{q}\nu}_{1234} =
        -\frac{\beta}{n_{\mathbf{k}}} \sum_{\mathbf{k}} A^{\mathrm{k}}_{14}\, B^{\mathrm{k}-\mathrm{q}}_{32}`, the
        first bubble is :math:`\chi_{0}[G, G]`, scaled and filtered as :meth:`create_generalized_chi0_q_fft` builds
        it, whose columns it equals bit for bit. For every window :math:`\delta G` of ``dg_rs`` the same column pass
        returns the bilinear bubble :math:`\chi_{0}[G, \delta G] + \chi_{0}[\delta G, G]`, the derivative of the
        unfiltered bubble in the direction :math:`\delta G`, scaled but not filtered. A column's values depend neither
        on the window nor on the rank count, so any split of the bosonic range into windows keeps every bit. The
        partner :math:`B^{\mathrm{k}-\mathrm{q}}_{32}` of a column is gathered from its window at the reversed momenta
        with the orbitals swapped, so no reversed copy of a window is held. A single rank transforms all fermionic
        columns of a bosonic frequency at once, with the same bits.

        :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
        :param g_r: The real-space window of :math:`G`, ``[2 n, n_R, o1, o2]`` around its central index ``n``, holding
            at least ``niv + w1 - 1`` positive fermionic frequencies; only read.
        :param niv: Number of positive fermionic frequencies of the bubbles.
        :param k_grid: The :class:`KGrid` over which the BZ sum/FFT is performed.
        :param beta: Inverse temperature :math:`\beta`.
        :param w_window: The stored bosonic index window ``(w0, w1)``, ``0 <= w0 < w1``.
        :param dg_rs: Real-space windows of the changes :math:`\delta G`, shaped and typed like ``g_r``; only read.
        :param with_forward: Whether the bubble of :math:`G` is returned too; without it only the bilinear bubbles are
            computed.
        :return: The bubble of :math:`G` (unless ``with_forward`` is False) followed by one bilinear bubble per entry
            of ``dg_rs``, each a
            :class:`FourPoint` on this rank's irreducible q-points (compressed momentum, half niw range holding the
            ``w1 - w0`` frequencies of the window, ``2 niv`` fermionic frequencies).
        :raises ValueError: If the window is empty or needs more fermionic frequencies than ``g_r`` holds.
        """
        w0, w1 = w_window
        nk, nk_tot = k_grid.nk, k_grid.nk_tot
        nb, dtype, center = g_r.shape[-1], g_r.dtype, g_r.shape[0] // 2
        if not 0 <= w0 < w1 or center < niv + w1 - 1:
            raise ValueError(
                f"The bosonic window ({w0}, {w1}) is empty or needs {niv + w1 - 1} positive fermionic frequencies; the "
                f"real-space window holds {center}."
            )

        # flat index of -R for every flat R: the partner of a column reads its window at the reversed momenta
        rev = np.ravel_multi_index(tuple(-np.indices(nk).reshape(3, -1) % np.reshape(nk, (3, 1))), nk)
        slabs = np.empty((2, nk_tot, nb, nb), dtype=dtype)
        # the second product of a bilinear column runs in blocks of momenta of at most one round buffer
        rows = max(1, ROUND_BUFFER_BYTES // (nb**4 * np.dtype(dtype).itemsize))
        block = np.empty((min(rows, nk_tot), nb, nb, nb, nb), dtype=dtype)
        windows = [g_r, *dg_rs] if with_forward else list(dg_rs)

        def partner(window: np.ndarray, f: int, slab: np.ndarray) -> np.ndarray:
            """Gathers ``window`` at -R and frequency index ``f`` into ``slab``, shaped [x, y, z, 1, o2, o3, 1]."""
            np.take(window[f], rev, axis=0, out=slab, mode="clip")
            return slab.reshape(*nk, nb, nb).swapaxes(-1, -2)[:, :, :, None, :, :, None]

        def fill(out: np.ndarray, part: int, iw: int, iv: int) -> None:
            """Writes the R-space product of the column (w0 + iw, v): G with its partner, or the bilinear sum."""
            f = center - niv + iv
            g_left = g_r[f].reshape(*nk, nb, nb)[:, :, :, :, None, None, :]
            g_rev = partner(g_r, f - w0 - iw, slabs[0])
            if with_forward and part == 0:
                np.multiply(g_left, g_rev, out=out)
                return
            dg_r = windows[part]
            np.multiply(g_left, partner(dg_r, f - w0 - iw, slabs[1]), out=out)
            BubbleGenerator._add_product(out, dg_r[f][:, :, None, None, :], g_rev, block)

        my_q_slice = mpi_dist_irrk.slices[mpi_dist_irrk.comm.rank]
        my_nq = my_q_slice.stop - my_q_slice.start
        n_w, n_v, n_parts = w1 - w0, 2 * niv, len(windows)
        mats = [np.empty((my_nq, nb, nb, nb, nb, n_w, n_v), dtype=dtype) for _ in range(n_parts)]
        cols = [np.reshape(mat, (my_nq, nb**4, n_w * n_v), copy=False) for mat in mats]

        def deliver(c0: int, c1: int, block: np.ndarray) -> None:
            """Writes a column block of this rank's momenta into the arrays of every part."""
            parts = block.reshape(my_nq, nb**4, n_parts, c1 - c0)
            for part, part_cols in enumerate(cols):
                part_cols[:, :, c0:c1] = parts[:, :, part]

        if mpi_dist_irrk.comm.size > 1:
            BubbleGenerator._run_columns(mpi_dist_irrk, k_grid, n_w, n_v, n_parts, nb, dtype, fill, deliver)
        else:
            parts = ([None] if with_forward else []) + list(dg_rs)
            BubbleGenerator._fill_window_on_one_rank(mats, g_r, parts, center - niv, w0, k_grid, rev)
        bubbles = []
        for mat in mats:
            mat *= -beta / nk_tot
            bubbles.append(
                FourPoint(mat, SpinChannel.NONE, nk, 1, 1, full_niw_range=False, has_compressed_q_dimension=True)
            )
        if with_forward:
            bubbles[0].filter_small_values()
        return bubbles

    @staticmethod
    def _add_product(out: np.ndarray, left: np.ndarray, right: np.ndarray, block: np.ndarray) -> None:
        """
        Adds the elementwise product of ``left`` and ``right`` to ``out`` in blocks of the leading momentum axes, each
        written into ``block`` first, so no second array of the size of ``out`` is allocated.

        :param out: The array ``[x, y, z, ...]`` added to in place.
        :param left: The left factor ``[n_R, ...]``, broadcast against ``out``'s trailing axes.
        :param right: The right factor ``[x, y, z, ...]``.
        :param block: The buffer of the products of one block ``[n_rows, ...]`` of ``n_R = x y z`` momenta.
        :return: None.
        """
        n_r = left.shape[0]
        rows = np.reshape(out, (n_r, *out.shape[3:]), copy=False)
        right = np.reshape(right, (n_r, *right.shape[3:]), copy=False)
        for r0 in range(0, n_r, block.shape[0]):
            r1 = min(n_r, r0 + block.shape[0])
            np.multiply(left[r0:r1], right[r0:r1], out=block[: r1 - r0])
            np.add(rows[r0:r1], block[: r1 - r0], out=rows[r0:r1])

    @staticmethod
    def _fill_window_on_one_rank(
        mats: list[np.ndarray],
        g_r: np.ndarray,
        parts: list[np.ndarray | None],
        first: int,
        w0: int,
        k_grid: KGrid,
        rev: np.ndarray,
    ) -> None:
        r"""
        Writes the unscaled bubbles of :meth:`create_generalized_chi0_q_fft_window` on a single rank, every fermionic
        column of one bosonic frequency at once: per part the product of :math:`G` with its partner, or the bilinear
        sum of a change. Every column is the same elementwise product as in the column pass, and the transform runs
        every momentum line on its own, so the bubbles have the column pass's bits.

        :param mats: The bubbles' arrays ``[nk_irr, o1, o2, o3, o4, n_w, n_v]``, written in place.
        :param g_r: The real-space window of :math:`G`.
        :param parts: Per bubble ``None`` for the bubble of :math:`G`, or the real-space window of the change.
        :param first: The window index of the bubbles' first fermionic frequency.
        :param w0: The first stored bosonic index of the window.
        :param k_grid: The :class:`KGrid` over which the BZ sum/FFT is performed.
        :param rev: The flat index of :math:`-\mathbf{R}` for every flat :math:`\mathbf{R}`.
        :return: None.
        """
        nk, nb, n_v = k_grid.nk, g_r.shape[-1], mats[0].shape[-1]
        buf = np.empty((*nk, nb, nb, nb, nb, n_v), dtype=g_r.dtype)
        rows = max(1, ROUND_BUFFER_BYTES // (buf.nbytes // k_grid.nk_tot))
        block = np.empty((min(rows, k_grid.nk_tot), *buf.shape[3:]), dtype=g_r.dtype)

        def left(window: np.ndarray) -> np.ndarray:
            """The window at the bubbles' fermionic frequencies, shaped [n_R, o1, 1, 1, o4, v]."""
            return np.moveaxis(window[first : first + n_v], 0, -1)[:, :, None, None, :, :]

        def partner(window: np.ndarray, iw: int) -> np.ndarray:
            """The window at -R and the shifted frequencies, orbitals swapped, shaped [x, y, z, 1, o2, o3, 1, v]."""
            gathered = np.take(window[first - w0 - iw : first - w0 - iw + n_v], rev, axis=1)
            return np.moveaxis(gathered.reshape(n_v, *nk, nb, nb).swapaxes(-1, -2), 0, -1)[:, :, :, None, :, :, None]

        g_left = left(g_r).reshape(*nk, nb, 1, 1, nb, n_v)
        for iw in range(mats[0].shape[-2]):
            g_rev = partner(g_r, iw)
            for mat, dg_r in zip(mats, parts):
                if dg_r is None:
                    np.multiply(g_left, g_rev, out=buf)
                else:
                    np.multiply(g_left, partner(dg_r, iw), out=buf)
                    BubbleGenerator._add_product(buf, left(dg_r), g_rev, block)
                transformed = sp.fft.ifftn(buf, axes=(0, 1, 2), overwrite_x=True)
                mat[..., iw, :] = transformed.reshape(k_grid.nk_tot, nb, nb, nb, nb, n_v)[k_grid.irrk_ind]

    @staticmethod
    def _create_generalized_chi0_q_fft_distributed(
        mpi_dist_irrk: MpiDistributor,
        giwk: GreensFunction,
        niw: int,
        niv: int,
        k_grid: KGrid,
        beta: float,
        node_comm=None,
        subtract_from: FourPoint | None = None,
    ) -> FourPoint:
        r"""
        Distributed evaluation of :meth:`create_generalized_chi0_q_fft`: the flattened bosonic-fermionic
        frequency columns :math:`(\omega, \nu)` are split across the ranks, each rank runs the R-space multiply and
        the full-grid in-place ``scipy.fft.ifftn`` (bit-identical to the rank-0 path) only for its columns - reading
        the R-space Green's function and its momentum-flipped, orbital-transposed partner from per-node shared-memory
        windows - and the per-column irreducible-BZ results are ring-exchanged onto the irr-BZ q-distribution
        (the same total bytes the former rank-0 scatter moved). The columns are processed in sub-chunks sized so
        that a node's combined full-grid buffers stay well below the former rank-0 footprint and a sub-chunk's buffer
        stays within ``ROUND_BUFFER_BYTES`` (its column-wise fill is a strided write, so a larger buffer would stream
        every cache line through memory once per column; a column above the budget runs alone); the exchange is
        pipelined per sub-chunk on a globally derived schedule, so no rank ever holds more than its final irr-BZ
        slice plus one sub-chunk in flight. With ``subtract_from`` every column block is scaled and filtered as the
        finished bubble is and subtracted from that bubble's columns as it arrives, so no slice of its own is
        allocated (the same elementwise operations, hence the bits of ``subtract_from.sub(bubble)``).

        :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
        :param giwk: The momentum-dependent :class:`GreensFunction`.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param k_grid: The :class:`KGrid` over which the BZ sum/FFT is performed.
        :param beta: Inverse temperature :math:`\beta`.
        :param node_comm: Optional node-local communicator for the shared R-space Green's functions.
        :param subtract_from: A bubble of the same box on this rank's irreducible q-points that the bubble is
            subtracted from in place, or ``None``.
        :return: The bubble as a :class:`FourPoint` over the irreducible BZ (compressed momentum, half niw range), or
            ``subtract_from`` minus the bubble.
        """
        nb = giwk.n_bands
        nk_tot = k_grid.nk_tot
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

        my_q_slice = mpi_dist_irrk.slices[mpi_dist_irrk.comm.rank]
        my_nq = (my_q_slice.stop - my_q_slice.start) if my_q_slice is not None else 0
        if subtract_from is None:
            gchi0_q_mat = np.empty((my_nq, nb, nb, nb, nb, len(wn), vpos), dtype=g_r_mat.dtype)
        else:
            gchi0_q_mat = subtract_from.mat
        gchi0_cols = np.reshape(gchi0_q_mat, (my_nq, nb**4, len(wn) * vpos), copy=False)  # contiguous, columns w-major

        def fill(out: np.ndarray, part: int, iw: int, iv: int) -> None:
            """Writes the R-space product of the column ``(w, v)``, the bubble's only part."""
            np.multiply(
                g_r_mat[:, :, :, :, None, None, :, start + iv],
                g_r_rev_mat[:, :, :, None, :, :, None, start + iv - wn[iw]],
                out=out,
            )

        def deliver(c0: int, c1: int, block: np.ndarray) -> None:
            """Writes a column block of this rank's momenta, or subtracts it scaled and filtered (``subtract_from``)."""
            cols = gchi0_cols[:, :, c0:c1]
            if subtract_from is None:
                cols[...] = block.reshape(cols.shape)
                return
            block *= -beta / nk_tot
            zero_small_values(block)
            cols -= block.reshape(cols.shape)

        BubbleGenerator._run_columns(mpi_dist_irrk, k_grid, len(wn), vpos, 1, nb, g_r_mat.dtype, fill, deliver)
        g_r_mat = g_r_rev_mat = None
        for win in (g_r_win, g_r_rev_win):
            if win is not None:
                node_comm.Barrier()
                win.Free()

        if subtract_from is not None:
            return subtract_from
        gchi0_q_mat *= -beta / nk_tot
        return FourPoint(
            gchi0_q_mat, SpinChannel.NONE, k_grid.nk, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
        ).filter_small_values()

    @staticmethod
    def _run_columns(
        mpi_dist_irrk: MpiDistributor,
        k_grid: KGrid,
        n_w: int,
        n_v: int,
        n_parts: int,
        nb: int,
        dtype: np.dtype,
        fill: Callable[[np.ndarray, int, int, int], None],
        deliver: Callable[[int, int, np.ndarray], None],
    ) -> None:
        r"""
        Runs the column evaluation of the distributed bubbles (collective). The ``n_w * n_v`` flattened
        bosonic-fermionic columns :math:`(\omega, \nu)`, bosonic-major, are split across the ranks in contiguous runs
        and processed in sub-chunk rounds of one full-grid buffer. Per round and part, ``fill(out, part, iw, iv)``
        writes the R-space product of each of the rank's columns into the buffer, the in-place ``scipy.fft.ifftn``
        transforms them and their irreducible-BZ rows are kept; the rows of all parts are ring-exchanged onto the
        irr-BZ q-distribution and handed to ``deliver(c0, c1, block)``, ``block = [q, o1, o2, o3, o4, n_parts * (c1 -
        c0)]`` holding this rank's momenta of the columns ``[c0, c1)``, part-major. The buffer stays within
        ``ROUND_BUFFER_BYTES`` (the column-wise fill is a strided write, so a larger buffer would stream every cache
        line through memory once per column; a column above the budget runs alone). The schedule depends only on the
        column counts, the box and the rank count, so the pipelined exchange stays collectively matched and no rank
        holds more than one round in flight. A column's values do not depend on the schedule: its product is
        elementwise and the transform runs every momentum line of every column on its own.

        :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
        :param k_grid: The :class:`KGrid` over which the BZ sum/FFT is performed.
        :param n_w: Number of bosonic frequencies of the columns.
        :param n_v: Number of fermionic frequencies of the columns.
        :param n_parts: Number of products per column.
        :param nb: Number of orbitals.
        :param dtype: Storage dtype of the products.
        :param fill: Writes the R-space product ``part`` of the column ``(iw, iv)`` into ``out``
            ``[x, y, z, o1, o2, o3, o4]``.
        :param deliver: Receives every column block of this rank's momenta.
        :return: None.
        """
        comm = mpi_dist_irrk.comm
        rank, size = comm.rank, comm.size
        nk_tot, nk_irr = k_grid.nk_tot, len(k_grid.irrk_ind)

        # globally derived column schedule: contiguous (w, v)-column runs per rank, processed in equal-size
        # sub-chunk rounds so the pipelined ring exchange stays collectively matched on every rank
        total_cols = n_w * n_v
        col_bounds = np.linspace(0, total_cols, size + 1).astype(int)
        ncols_max = int(np.max(np.diff(col_bounds)))
        sub_cols = max(1, ncols_max * nk_irr // (6 * nk_tot), min(ncols_max, n_v // max(1, size)))
        sub_cols = max(1, min(sub_cols, ROUND_BUFFER_BYTES // (nk_tot * nb**4 * np.dtype(dtype).itemsize)))
        n_rounds = -(-ncols_max // sub_cols) if ncols_max else 0

        my_q_slice = mpi_dist_irrk.slices[rank]
        my_nq = (my_q_slice.stop - my_q_slice.start) if my_q_slice is not None else 0

        def round_cols(r: int, rnd: int) -> tuple[int, int]:
            c0 = col_bounds[r] + rnd * sub_cols
            return c0, min(col_bounds[r + 1], c0 + sub_cols)

        buf = np.empty((*k_grid.nk, nb, nb, nb, nb, sub_cols), dtype=dtype)
        for rnd in range(n_rounds):
            c0, c1 = round_cols(rank, rnd)
            n_mine = max(0, c1 - c0)
            chunk_irr = None
            if n_mine:
                buf_v = buf[..., :n_mine]
                pieces = []
                for part in range(n_parts):
                    for col in range(c0, c1):  # fill column-wise: one (w, v) pair per column
                        fill(buf_v[..., col - c0], part, *divmod(col, n_v))
                    # scipy.fft keeps complex64 in place; numpy.fft would upcast to complex128 (double work + 2x buffer)
                    transformed = sp.fft.ifftn(buf_v, axes=(0, 1, 2), overwrite_x=True)
                    pieces.append(
                        np.ascontiguousarray(transformed.reshape(nk_tot, nb, nb, nb, nb, n_mine)[k_grid.irrk_ind])
                    )
                    transformed = None
                chunk_irr = pieces[0] if n_parts == 1 else np.concatenate(pieces, axis=-1)
                pieces = None
                if my_nq:
                    deliver(c0, c1, chunk_irr[my_q_slice])

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
                    staging = np.empty((my_nq, nb, nb, nb, nb, n_parts * (s1 - s0)), dtype=dtype)
                    reqs += mpi_utils._irecv_rows_into(comm, staging, src, base_tag=500 + step)

                if reqs:
                    mpi_utils.MPI.Request.Waitall(reqs)
                if staging is not None:
                    deliver(s0, s1, staging)

    @staticmethod
    def create_generalized_chi0_q_loop(
        giwk: GreensFunction,
        niw: int,
        niv: int,
        q_list: np.ndarray,
        q_grid: KGrid,
        beta: float,
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
        :return: The bubble as a :class:`FourPoint` over the given q-points (compressed momentum, half niw range).
        """
        wn = MFHelper.wn(niw, return_only_positive=True)
        nb = giwk.n_bands
        nq = len(q_list)

        gchi0_q = np.zeros((nq, nb, nb, nb, nb, len(wn), 2 * niv), dtype=giwk.mat.dtype)

        # g_left (the central niv window) and g_right (the full range, momentum-shifted per q) are both read-only,
        # so they can share one backing array instead of holding a full-size duplicate.
        g_full = np.asarray(giwk.cut_niv(niv + niw).mat, order="C")
        giwk_niv = g_full.shape[-1] // 2

        g_r_buf = np.empty_like(g_full)
        g_right = g_full
        g_left = g_full[..., giwk_niv - niv : giwk_niv + niv]

        path = np.einsum_path("xyzadv,xyzcbv->abcdv", g_left, g_left, optimize="optimal")[0]
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
        :math:`\chi^{\mathrm{k}}_{0;1234} = G^{\mathrm{k}}_{14}\, G^{-\mathrm{k}}_{23}`, the partner propagator read at
        the flipped momentum and fermionic frequency, :math:`-\mathrm{k} = (-\mathbf{k}, -\nu)`.
        Note that no factor of :math:`-\beta` is included here.

        :param giwk: The momentum-dependent :class:`GreensFunction`.
        :param niv_pp: Number of positive fermionic frequencies of the pp bubble.
        :param q_grid: The :class:`KGrid` defining the momentum grid.
        :return: The momentum-dependent pp bubble as a :class:`FourPoint` (no bosonic axis, pp notation, compressed q).
        """
        g = giwk.cut_niv(niv_pp).compress_q_dimension()
        # G_23 at (-k, -v); the same-momentum conj(G_32(k, v)) equals it only if G(-k) = G(k)
        g_minus = g.flip_momentum_axis(copy=True)
        gchi0_q_pp_w0 = g.mat[:, :, None, None, :, :] * g_minus.mat[:, None, :, :, None, ::-1]

        return FourPoint(
            gchi0_q_pp_w0, SpinChannel.NONE, q_grid.nk, 0, 1, True, True, True, FrequencyNotation.PP
        ).filter_small_values()

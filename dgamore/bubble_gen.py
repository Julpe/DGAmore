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

import time
from collections.abc import Callable, Sequence

import numpy as np
import scipy as sp

import dgamore.config as config
import dgamore.mpi_utils as mpi_utils
from dgamore.brillouin_zone import KGrid
from dgamore.four_point import FourPoint
from dgamore.greens_function import GreensFunction
from dgamore.local_four_point import LocalFourPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel, FrequencyNotation, zero_small_values

# bytes of the distributed bubble's full-grid buffer of one group of orbital components, and of the whole columns
# of a round: about one rank's L3 share
ROUND_BUFFER_BYTES = 4 * 1024**2

# bytes of received one-column blocks the distributed bubble hands to one deliver call: consecutive ring sources hold
# consecutive columns, so a call writes runs of whole cache lines, and its expanded copies stay within a few MiB
DELIVER_BATCH_BYTES = 2 * 1024**2

# seconds the steps of the first round's ring must wait for their transfers (a rank's lower quartile, the median over
# the ranks) before the later rounds' exchanges move behind the next round's columns: the helper's sleeps between its
# tests cost up to half a millisecond per step
BEHIND_STEP_SECONDS = 2e-3


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
        involution: np.ndarray | None = None,
        blocks: bool = False,
        release: Callable[[], None] | None = None,
    ) -> FourPoint | list[FourPoint]:
        r"""
        Returns the momentum-dependent generalized bare susceptibility :math:`\chi^{\mathrm{q}\nu}_{0;1234} = -\beta
        \sum_{\mathbf{k}} G^{\mathrm{k}}_{14}\, G^{\mathrm{k}-\mathrm{q}}_{32}`, evaluated via an FFT over the BZ with
        preallocated buffers. On a multi-rank run the bosonic-frequency loop is distributed across all ranks (see
        :meth:`_create_generalized_chi0_q_fft_distributed`); on a single rank the whole bubble is computed on rank 0
        over the irreducible BZ and scattered across ranks. When the Green's function obeys Hermiticity and time
        reversal on the box (see :meth:`_has_crossing_symmetries`), only the columns :meth:`_crossing_columns` leaves
        are transformed, about half of them, and every other column follows at the same momentum from the crossing
        relation :math:`\chi^{(\mathbf{q},\omega)\,\omega-\nu}_{0;1234} = [\chi^{(\mathbf{q},\omega)\,\nu}_{0;3412}]^*`
        as the conjugated rows of its computed partner with the orbital pairs swapped, so the computed columns keep
        their bits and the derived ones carry their partner's rounding, on one rank and on many alike. With
        ``subtract_from`` the bubble is subtracted from that bubble in place instead of returned: the multi-rank path
        scales, filters and subtracts every column block as it arrives, so the bubble's own array is never built; a
        single rank subtracts the finished bubble.

        When the Green's function is transposed by an orbital permutation of order two at every momentum (see
        :meth:`_orbital_transpose_permutation`), :math:`G^{\mathrm{k}}_{12} = G^{\mathrm{k}}_{\pi(2)\pi(1)}`, the
        bubble obeys :math:`\chi^{\mathrm{q}\nu}_{0;1234} = \chi^{\mathrm{q}\nu}_{0;\pi(4)\pi(3)\pi(2)\pi(1)}`
        at the same momentum and frequencies, so only one orbital component of each such pair is multiplied,
        transformed and exchanged and the other is its copy (see :meth:`_independent_components`).

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
        :param involution: An orbital permutation of order two that may transpose the Green's function (tried after
            the identity, see :meth:`_orbital_transpose_permutation`), or ``None``.
        :param blocks: Whether to return the bubble as a list of objects of consecutive bosonic windows (one per stored
            bosonic frequency on several ranks, the whole range on one rank), so a caller can consume and free it
            window by window; not with ``subtract_from``.
        :param release: Called once on every rank after the last read of ``giwk`` (its real-space transform and the
            symmetry checks are done) and before the first column, so the caller can free ``giwk``; or ``None``.
        :return: The bubble as a :class:`FourPoint` over the irreducible BZ (compressed momentum, half niw range), or
            ``subtract_from`` minus the bubble; with ``blocks`` the list of its bosonic windows.
        """
        if mpi_dist_irrk.comm.size > 1:
            return BubbleGenerator._create_generalized_chi0_q_fft_distributed(
                mpi_dist_irrk, giwk, niw, niv, k_grid, beta, node_comm, subtract_from, involution, blocks, release
            )

        gchi0_q_mat = None
        if mpi_dist_irrk.my_rank == 0:
            nb = giwk.n_bands
            wn = MFHelper.wn(niw, return_only_positive=True)

            g_k = giwk.cut_niv(niv + niw)
            g_r = g_k.fft().decompress_q_dimension()
            g_r_rev = g_r.flip_momentum_axis(copy=True).transpose_orbitals()

            pi = BubbleGenerator._orbital_transpose_permutation(giwk, niv + niw, involution)
            components = None if pi is None else BubbleGenerator._independent_components(nb, pi)
            comp_shape = (nb,) * 4 if components is None else (len(components[0]),)
            gchi0_q_mat = np.zeros((len(k_grid.irrk_ind), nb, nb, nb, nb, len(wn), 2 * niv), dtype=g_r.mat.dtype)
            gchi0_comps = gchi0_q_mat.reshape(len(k_grid.irrk_ind), nb**4, len(wn), 2 * niv)
            chi_r_v_buffer = np.empty((*k_grid.nk, *comp_shape, 2 * niv), dtype=g_r.mat.dtype)

            giwk_niv = g_r.current_shape[-1] // 2
            start = giwk_niv - niv
            end = giwk_niv + niv
            g_r = np.asarray(g_r.mat[..., start:end], order="C")
            g_r_rev = np.asarray(g_r_rev.mat, order="C")
            columns, partners = np.arange(len(wn) * 2 * niv), np.full(len(wn) * 2 * niv, -1)
            if BubbleGenerator._has_crossing_symmetries(giwk, niv + niw):
                columns, partners = BubbleGenerator._crossing_columns(wn, niv)
            if release is not None:
                release()

            for iw, wn_i in enumerate(wn):
                # the computed columns of a bosonic frequency are its first n_v fermionic ones
                mine = columns // (2 * niv) == iw
                n_v = np.count_nonzero(mine)
                buf = chi_r_v_buffer[..., :n_v]
                g_vw = g_r_rev[..., start - wn_i : start - wn_i + n_v]
                if components is None:
                    np.multiply(g_r[:, :, :, :, None, None, :, :n_v], g_vw[:, :, :, None, :, :, None, :], out=buf)
                else:
                    BubbleGenerator._fill_components(buf, g_r[..., :n_v], g_vw, components[0], nb)
                # scipy.fft keeps complex64 in place; numpy.fft would upcast to complex128 (double work + 2x buffer)
                chi_q_v = sp.fft.ifftn(buf, axes=(0, 1, 2), overwrite_x=True)
                chi_q_v = chi_q_v.reshape((k_grid.nk_tot, *comp_shape, n_v))[k_grid.irrk_ind]
                if components is None:
                    gchi0_q_mat[..., iw, :n_v] = chi_q_v
                else:
                    computed, derived, source = components
                    gchi0_comps[:, computed, iw, :n_v] = chi_q_v
                    gchi0_comps[:, derived, iw, :n_v] = chi_q_v[:, source]
                found = partners[mine] >= 0
                if found.any():
                    crossed = np.conj(gchi0_q_mat[..., iw, :n_v][..., found].transpose(0, 3, 4, 1, 2, 5))
                    gchi0_q_mat[..., iw, partners[mine][found] % (2 * niv)] = crossed

            gchi0_q_mat *= -beta / k_grid.nk_tot
        gchi0_q_mat = mpi_dist_irrk.scatter(gchi0_q_mat)

        bubble = FourPoint(
            gchi0_q_mat, SpinChannel.NONE, k_grid.nk, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
        ).filter_small_values()
        if blocks:
            return [bubble]
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
        it. Every column is computed here, so it equals that bubble's columns bit for bit, except the columns the
        crossing relation derives there on a :math:`G` with both crossing symmetries, which agree to rounding. For
        every window :math:`\delta G` of ``dg_rs`` the same column pass returns the bilinear bubble
        :math:`\chi_{0}[G, \delta G] + \chi_{0}[\delta G, G]`, the derivative of the
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

        # the partner of a column reads its window at the reversed momenta
        rev = BubbleGenerator._reversed_momenta(nk)
        slabs = np.empty((2, nk_tot, nb, nb), dtype=dtype)
        # the second product of a bilinear column runs in blocks of momenta of at most one round buffer
        rows = max(1, ROUND_BUFFER_BYTES // (nb**4 * np.dtype(dtype).itemsize))
        block = np.empty((min(rows, nk_tot), nb, nb, nb, nb), dtype=dtype)
        windows = [g_r, *dg_rs] if with_forward else list(dg_rs)

        def partner(window: np.ndarray, f: int, slab: np.ndarray) -> np.ndarray:
            """Gathers ``window`` at -R and frequency index ``f`` into ``slab``, shaped [x, y, z, 1, o2, o3, 1]."""
            np.take(window[f], rev, axis=0, out=slab, mode="clip")
            return slab.reshape(*nk, nb, nb).swapaxes(-1, -2)[:, :, :, None, :, :, None]

        def fill(out: np.ndarray, part: int, iw: int, iv: int, comps: slice) -> None:
            """Writes the R-space product of the column (w0 + iw, v): G with its partner, or the bilinear sum; columns
            run whole here, so ``comps`` holds every component."""
            out = out.reshape(*nk, nb, nb, nb, nb)
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

        def deliver(where: slice, block: np.ndarray) -> None:
            """Writes a column block of this rank's momenta into the arrays of every part."""
            parts = block.reshape(my_nq, nb**4, n_parts, -1)
            for part, part_cols in enumerate(cols):
                part_cols[:, :, where] = parts[:, :, part]

        if mpi_dist_irrk.comm.size > 1:
            BubbleGenerator._run_columns(mpi_dist_irrk, k_grid, n_w, n_v, n_parts, (nb,) * 4, dtype, fill, deliver)
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
        involution: np.ndarray | None = None,
        blocks: bool = False,
        release: Callable[[], None] | None = None,
    ) -> FourPoint | list[FourPoint]:
        r"""
        Distributed evaluation of :meth:`create_generalized_chi0_q_fft`: the flattened bosonic-fermionic
        frequency columns :math:`(\omega, \nu)` are split across the ranks, each rank runs the R-space multiply and
        the full-grid in-place ``scipy.fft.ifftn`` (bit-identical to the rank-0 path) only for its columns, and the
        per-column irreducible-BZ results are ring-exchanged onto the irr-BZ q-distribution (at most the bytes the
        former rank-0 scatter moved). The R-space Green's function is read from one per-node shared-memory window: per
        column a rank gathers its two factors orbital-major, :math:`G_{14}(\mathbf{R})` at :math:`\nu` and the partner
        :math:`G_{32}(-\mathbf{R})` at :math:`\nu - \omega` (the window at the reversed momenta, orbitals swapped), and
        multiplies and transforms the column in groups of orbital components of at most ``ROUND_BUFFER_BYTES`` (one
        component when a component is larger), so it never holds a whole full-grid column. The exchange runs per
        sub-chunk round of columns on a globally derived schedule, so no rank ever holds more than its final irr-BZ
        slice plus the rows of two rounds (see :meth:`_run_columns`). On a Green's function with both crossing
        symmetries only the crossing's computed columns are transformed and exchanged, and every rank derives their
        partners on its own momenta as the blocks arrive. With ``subtract_from`` every column block, computed or
        derived, is scaled and filtered as the finished bubble is and subtracted from that bubble's columns as it
        arrives, so no slice of its own is allocated (the same elementwise operations, hence the bits of
        ``subtract_from.sub(bubble)``). With ``blocks`` every stored bosonic frequency is held in an array of its own,
        with the same bits.

        On a Green's function an orbital permutation transposes, only the independent orbital components are
        multiplied (in blocks of momenta), transformed and exchanged, and every rank copies the others from them as
        the blocks arrive.

        :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
        :param giwk: The momentum-dependent :class:`GreensFunction`.
        :param niw: Number of positive bosonic frequencies.
        :param niv: Number of positive fermionic frequencies.
        :param k_grid: The :class:`KGrid` over which the BZ sum/FFT is performed.
        :param beta: Inverse temperature :math:`\beta`.
        :param node_comm: Optional node-local communicator for the shared R-space Green's functions.
        :param subtract_from: A bubble of the same box on this rank's irreducible q-points that the bubble is
            subtracted from in place, or ``None``.
        :param involution: An orbital permutation of order two that may transpose the Green's function, or ``None``.
        :param blocks: Whether to return one object per stored bosonic frequency; not with ``subtract_from``.
        :param release: Called once after the last read of ``giwk`` and before the first column, or ``None``.
        :return: The bubble as a :class:`FourPoint` over the irreducible BZ (compressed momentum, half niw range), or
            ``subtract_from`` minus the bubble; with ``blocks`` the list of its bosonic frequencies.
        :raises ValueError: If both ``blocks`` and ``subtract_from`` are given.
        """
        if blocks and subtract_from is not None:
            raise ValueError("A bubble subtracted in place cannot be returned in bosonic blocks.")
        nb = giwk.n_bands
        nk_tot = k_grid.nk_tot
        wn = MFHelper.wn(niw, return_only_positive=True)
        vpos = 2 * niv
        pi = BubbleGenerator._orbital_transpose_permutation(giwk, niv + niw, involution, mpi_dist_irrk.comm)
        components = None if pi is None else BubbleGenerator._independent_components(nb, pi)

        def build_g_r() -> np.ndarray:
            return giwk.cut_niv(niv + niw).fft(copy=False).decompress_q_dimension().mat

        if node_comm is not None:
            g_r_mat, g_r_win = mpi_utils.build_node_shared_array(node_comm, build_g_r)
        else:
            g_r_mat, g_r_win = build_g_r(), None

        start = g_r_mat.shape[-1] // 2 - niv

        my_q_slice = mpi_dist_irrk.slices[mpi_dist_irrk.comm.rank]
        my_nq = (my_q_slice.stop - my_q_slice.start) if my_q_slice is not None else 0
        if subtract_from is not None:
            mats = [subtract_from.mat]
        elif blocks:
            mats = [np.empty((my_nq, nb, nb, nb, nb, 1, vpos), dtype=g_r_mat.dtype) for _ in wn]
        else:
            mats = [np.empty((my_nq, nb, nb, nb, nb, len(wn), vpos), dtype=g_r_mat.dtype)]
        # contiguous, columns w-major; one stored bosonic frequency per array with blocks
        width = vpos if blocks else len(wn) * vpos
        mat_cols = [np.reshape(mat, (my_nq, nb**4, width), copy=False) for mat in mats]

        # chi0_1234(R) = G_14(R) G_32(-R): the flat factor pairs of every transformed component, the orbital swap
        # of the partner, and the column's two factors, orbital-major, gathered once per column
        o1, o2, o3, o4 = np.unravel_index(np.arange(nb**4) if components is None else components[0], (nb,) * 4)
        left_pair, right_pair = o1 * nb + o4, o2 * nb + o3
        swapped = np.arange(nb * nb).reshape(nb, nb).T.ravel()
        rev = BubbleGenerator._reversed_momenta(k_grid.nk)
        factors = np.empty((2, nb * nb, nk_tot), dtype=g_r_mat.dtype)
        gathered = [None]

        def fill(out: np.ndarray, part: int, iw: int, iv: int, comps: slice) -> None:
            """Writes the R-space products of the components ``comps`` of the column ``(w, v)``, the bubble's only part;
            the column's factors are gathered from the window when its first group is filled."""
            if gathered[0] != (iw, iv):
                np.copyto(factors[0], g_r_mat[..., start + iv].reshape(nk_tot, nb * nb).T)
                right = g_r_mat[..., start + iv - wn[iw]].reshape(nk_tot, nb * nb)
                for pair, source in enumerate(swapped):
                    np.take(right[:, source], rev, out=factors[1, pair], mode="clip")
                gathered[0] = (iw, iv)
            rows = np.reshape(out, (nk_tot, -1), copy=False)
            for i, c in enumerate(range(comps.start, comps.stop)):
                np.multiply(factors[0, left_pair[c]], factors[1, right_pair[c]], out=rows[:, i])

        def put(target: slice, values: np.ndarray) -> None:
            """Writes ``values [q, nb**4, n]`` into the columns ``target`` of one bosonic frequency, or subtracts them
            (``subtract_from``)."""
            b = target.start // width
            local = slice(target.start - b * width, target.stop - b * width)
            if subtract_from is None:
                mat_cols[b][:, :, local] = values
            else:
                mat_cols[b][:, :, local] -= values

        def deliver(where: slice | np.ndarray, block: np.ndarray) -> None:
            """Writes the column block ``[q, *comp, 1, n]`` of this rank's momenta, the orbital components copied from
            its independent ones and the crossing partners it derives, or subtracts all of them scaled and filtered
            (``subtract_from``), one slice per run of consecutive columns of one bosonic frequency."""
            block = block[..., 0, :]
            n = block.shape[-1]
            if components is not None:
                computed, derived, source = components
                every = np.empty((my_nq, nb**4, n), dtype=block.dtype)
                every[:, computed], every[:, derived] = block, block[:, source]
                block = every
            block = np.reshape(block, (my_nq, nb**4, n))
            if subtract_from is not None:
                block *= -beta / nk_tot
                zero_small_values(block)
            flat = np.arange(where.start, where.stop) if isinstance(where, slice) else where
            ends = np.flatnonzero((np.diff(flat) != 1) | (np.diff(flat // vpos) != 0)) + 1
            for o0, o1 in zip([0, *ends], [*ends, n]):
                c0, c1 = flat[o0], flat[o0] + o1 - o0
                put(slice(c0, c1), block[..., o0:o1])
                found = np.flatnonzero(partner_of[c0:c1] >= 0) if partner_of is not None else ()
                if len(found):
                    # the partner (w, w - v) of (w, v) at the same q: conjugated, orbital pairs 12 and 34 swapped;
                    # the partners of a run of one frequency form one descending run
                    crossed = block.reshape(my_nq, nb, nb, nb, nb, n)[..., o0 + found[0] : o0 + found[-1] + 1]
                    crossed = np.conjugate(crossed.transpose(0, 3, 4, 1, 2, 5)).reshape(my_nq, nb**4, -1)
                    put(slice(partner_of[c0 + found[-1]], partner_of[c0 + found[0]] + 1), crossed[..., ::-1])

        columns = partner_of = None
        if BubbleGenerator._has_crossing_symmetries(giwk, niv + niw, mpi_dist_irrk.comm):
            columns, partners = BubbleGenerator._crossing_columns(wn, niv)
            partner_of = np.full(len(wn) * vpos, -1)
            partner_of[columns] = partners
        if release is not None:
            release()
        comp_shape = (nb,) * 4 if components is None else (len(components[0]),)
        group = max(1, ROUND_BUFFER_BYTES // (nk_tot * g_r_mat.itemsize))
        BubbleGenerator._run_columns(
            mpi_dist_irrk, k_grid, len(wn), vpos, 1, comp_shape, g_r_mat.dtype, fill, deliver, columns, group
        )
        g_r_mat = None
        if g_r_win is not None:
            node_comm.Barrier()
            g_r_win.Free()

        if subtract_from is not None:
            return subtract_from
        bubbles = []
        for mat in mats:
            mat *= -beta / nk_tot
            bubbles.append(
                FourPoint(mat, SpinChannel.NONE, k_grid.nk, 1, 1, full_niw_range=False, has_compressed_q_dimension=True)
            )
            bubbles[-1].filter_small_values()
        return bubbles if blocks else bubbles[0]

    @staticmethod
    def _independent_components(nb: int, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        r"""
        Returns the orbital components of a bubble left to compute under :math:`\chi_{0;1234} =
        \chi_{0;\pi(4)\pi(3)\pi(2)\pi(1)}`: of every pair of components the relation maps onto each other the one
        with the smaller flat index ``((o1 nb + o2) nb + o3) nb + o4``, and every component it maps onto itself.

        :param nb: Number of orbitals.
        :param pi: The orbital permutation :math:`\pi` (an index array, its own inverse).
        :return: The tuple ``(computed, derived, source)``: the flat indices of the computed components, ascending,
            those of the derived ones, and per derived component the position of its partner among the computed
            ones; None when no component is derived.
        """
        o = np.indices((nb,) * 4).reshape(4, -1)
        partner = np.ravel_multi_index((pi[o[3]], pi[o[2]], pi[o[1]], pi[o[0]]), (nb,) * 4)
        flat = np.arange(nb**4)
        computed, derived = flat[flat <= partner], flat[flat > partner]
        return (computed, derived, np.searchsorted(computed, partner[derived])) if derived.size else None

    @staticmethod
    def _fill_components(out: np.ndarray, left: np.ndarray, right: np.ndarray, computed: np.ndarray, nb: int) -> None:
        r"""
        Writes the products :math:`L_{14} R_{23}` of the orbital components ``computed`` of two real-space factors
        into ``out``, in blocks of momenta of about 1 MiB of gathered factors, so no temporary of the size of ``out``
        is made.

        :param out: The products ``[x, y, z, component, ...]``; written in place.
        :param left: The factor ``[x, y, z, o1, o4, ...]``.
        :param right: The factor ``[x, y, z, o2, o3, ...]``, its trailing axes those of ``left``.
        :param computed: The flat indices ``((o1 nb + o2) nb + o3) nb + o4`` of the components.
        :param nb: Number of orbitals.
        :return: None.
        """
        o1, o2, o3, o4 = np.unravel_index(computed, (nb,) * 4)
        n_r = int(np.prod(out.shape[:3]))
        flat = np.reshape(out, (n_r, *out.shape[3:]), copy=False)
        left = np.reshape(left, (n_r, nb * nb, *left.shape[5:]), copy=False)
        right = np.reshape(right, (n_r, nb * nb, *right.shape[5:]), copy=False)
        rows = max(1, 2**20 // (flat[0].size * flat.itemsize))
        for r0 in range(0, n_r, rows):
            np.multiply(
                left[r0 : r0 + rows, o1 * nb + o4], right[r0 : r0 + rows, o2 * nb + o3], out=flat[r0 : r0 + rows]
            )

    @staticmethod
    def _orbital_transpose_permutation(
        giwk: GreensFunction, niv_window: int, involution: np.ndarray | None = None, comm=None
    ) -> np.ndarray | None:
        r"""
        Returns the orbital permutation :math:`\pi` of order two that transposes the Green's function at every
        momentum, :math:`G^{\mathrm{k}}_{12} = G^{\mathrm{k}}_{\pi(2)\pi(1)}` on the fermionic frequencies
        ``[-niv_window, niv_window)``, to ``1e-6`` relative to the largest :math:`|G|` there, or to 64 units of
        rounding of the storage precision when that is finer (complex128): the identity for a symmetric Green's
        function, else ``involution`` if it does. Time reversal with an inversion that permutes the orbitals gives
        such a relation. Collective over ``comm``: every rank checks its share of the momenta in blocks of rows, so
        no copy of its share is held, and all ranks return the same answer.

        :param giwk: The momentum-dependent :class:`GreensFunction`; only read.
        :param niv_window: Number of positive fermionic frequencies checked.
        :param involution: The permutation tried after the identity (an index array, its own inverse), or ``None``.
        :param comm: The MPI communicator, or ``None`` for a single rank.
        :return: The permutation as an index array; None for one band or when neither candidate holds.
        """
        nb = giwk.n_bands
        if nb == 1:
            return None
        candidates = [np.arange(nb)] + ([] if involution is None else [np.asarray(involution)])
        mat = np.reshape(giwk.mat, (-1, nb, nb, giwk.mat.shape[-1]), copy=False)
        mat = mat[..., giwk.niv - niv_window : giwk.niv + niv_window]
        size, rank = (comm.size, comm.rank) if comm is not None else (1, 0)
        first, last = rank * len(mat) // size, (rank + 1) * len(mat) // size
        diffs, scale = [0.0] * len(candidates), 0.0
        for r0 in range(first, last, 64):
            rows = mat[r0 : min(r0 + 64, last)]
            scale = max(scale, float(np.abs(rows).max()))
            for c, pi in enumerate(candidates):
                diffs[c] = max(diffs[c], float(np.abs(rows.swapaxes(1, 2) - rows[:, pi][:, :, pi]).max()))
        if size > 1:
            gathered = comm.allgather((diffs, scale))
            diffs = [max(values) for values in zip(*(d for d, _ in gathered))]
            scale = max(sc for _, sc in gathered)
        bound = min(1e-6, 64 * np.finfo(mat.dtype).eps) * scale
        return next((pi for pi, diff in zip(candidates, diffs) if diff <= bound), None)

    @staticmethod
    def _crossing_columns(wn: np.ndarray, niv: int) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Returns the bubble columns left to compute under the crossing relation
        :math:`\chi^{(\mathbf{q},\omega)\,\omega-\nu}_{0;1234} = [\chi^{(\mathbf{q},\omega)\,\nu}_{0;3412}]^*` and the
        partner of each. A column is the flat index ``iw * 2 niv + iv`` of the bosonic frequency ``wn[iw]`` and the
        fermionic frequency :math:`\nu_{iv - niv}`; the partner of ``iv`` at :math:`\omega_m` is
        ``2 niv + m - 1 - iv``. A column is computed unless its partner lies in the box below it, so every column is
        either computed or the partner of exactly one computed column, and the column paired with itself is computed.

        :param wn: The non-negative bosonic frequency indices of the stored columns.
        :param niv: Number of positive fermionic frequencies.
        :return: The tuple ``(columns, partners)``: the flat columns to compute, ascending, and per column the flat
            column of its partner, or ``-1`` when it has none in the box.
        """
        iv = np.arange(2 * niv)
        partner = 2 * niv + np.asarray(wn)[:, None] - 1 - iv
        flat = np.arange(len(wn))[:, None] * 2 * niv
        computed = partner >= iv
        partners = np.where((partner < 2 * niv) & (partner > iv), flat + partner, -1)
        return (flat + iv)[computed], partners[computed]

    @staticmethod
    def _has_crossing_symmetries(giwk: GreensFunction, niv_window: int, comm=None) -> bool:
        r"""
        Returns whether Hermiticity, :math:`G^{(\mathbf{k},-\nu)}_{12} = [G^{(\mathbf{k},\nu)}_{21}]^*`, and time
        reversal, :math:`G^{(-\mathbf{k},\nu)}_{12} = G^{(\mathbf{k},\nu)}_{21}`, the conditions of the bubble's
        crossing relation at fixed momentum, both hold on the fermionic frequencies ``[-niv_window, niv_window)`` to
        ``1e-6`` relative to the largest :math:`|G|` there, or to 64 units of rounding of the storage precision when
        that is finer (complex128), so a derived column agrees with the transformed one to rounding. Collective over
        ``comm``: every rank checks its share of the momenta in blocks of rows against the same rows and their
        :math:`-\mathbf{k}` rows, so no copy of its share is held, and all ranks return the same answer.

        :param giwk: The momentum-dependent :class:`GreensFunction`; only read.
        :param niv_window: Number of positive fermionic frequencies checked.
        :param comm: The MPI communicator, or ``None`` for a single rank.
        :return: Whether both symmetries hold on every momentum and checked frequency.
        """
        nb = giwk.n_bands
        mat = np.reshape(giwk.mat, (-1, nb, nb, giwk.mat.shape[-1]), copy=False)
        mat = mat[..., giwk.niv - niv_window : giwk.niv + niv_window]
        reversed_rows = BubbleGenerator._reversed_momenta(giwk.nq)
        size, rank = (comm.size, comm.rank) if comm is not None else (1, 0)
        first, last = rank * len(mat) // size, (rank + 1) * len(mat) // size
        diff = scale = 0.0
        for r0 in range(first, last, 64):
            rows = mat[r0 : min(r0 + 64, last)]
            diff = max(diff, float(np.abs(rows - np.conj(rows[..., ::-1].swapaxes(1, 2))).max()))
            diff = max(diff, float(np.abs(rows - mat[reversed_rows[r0 : r0 + len(rows)]].swapaxes(1, 2)).max()))
            scale = max(scale, float(np.abs(rows).max()))
        if size > 1:
            diff, scale = (max(values) for values in zip(*comm.allgather((diff, scale))))
        return diff <= min(1e-6, 64 * np.finfo(mat.dtype).eps) * scale

    @staticmethod
    def _reversed_momenta(nk: tuple) -> np.ndarray:
        r"""
        Returns the flat index of :math:`-\mathbf{R}` for every flat :math:`\mathbf{R}` of the grid ``nk`` (C order),
        which is also the flat index of :math:`-\mathbf{q}` on the transformed grid.

        :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
        :return: The flat reversed indices, one per grid point.
        """
        return np.ravel_multi_index(tuple(-np.indices(nk).reshape(3, -1) % np.reshape(nk, (3, 1))), nk)

    @staticmethod
    def _run_columns(
        mpi_dist_irrk: MpiDistributor,
        k_grid: KGrid,
        n_w: int,
        n_v: int,
        n_parts: int,
        comp_shape: tuple,
        dtype: np.dtype,
        fill: Callable[[np.ndarray, int, int, int, slice], None],
        deliver: Callable[[slice | np.ndarray, np.ndarray], None],
        columns: np.ndarray | None = None,
        group: int | None = None,
    ) -> None:
        r"""
        Runs the column evaluation of the distributed bubbles (collective). The flattened bosonic-fermionic columns
        :math:`(\omega, \nu)`, ``iw * n_v + iv`` bosonic-major (all ``n_w * n_v`` of them, or ``columns``), are
        processed in rounds, round-major: a round takes the next ``size * sub_cols`` positions of the column list, rank
        ``k`` the ``k``-th run of ``sub_cols`` of them. Per round, part and column the flat orbital components run in
        groups of at most ``group``: ``fill(out, part, iw, iv, comps)`` writes the R-space products of the components
        ``comps`` (a slice of the flat components) into one full-grid buffer ``out`` ``[x, y, z, n]``, the in-place
        ``scipy.fft.ifftn`` transforms them and their irreducible-BZ rows are kept, so a rank holds one group of one
        column at a time. The rows of all parts are ring-exchanged onto the irr-BZ q-distribution, one rank distance in
        flight per rank, and handed to ``deliver(where, block)``, ``block = [q, *comp_shape, n_parts, n]`` holding
        this rank's momenta of the ``n`` flat columns ``where`` (a slice, or an ascending index array with
        ``columns``). With one column per round the ring's consecutive sources hold consecutive columns, so their
        blocks are staged and handed over together, up to ``DELIVER_BATCH_BYTES`` per call. A round holds at most
        ``ROUND_BUFFER_BYTES`` of whole columns (a column above the budget runs alone). When MPI lets a helper thread
        call it (``MPI.THREAD_SERIALIZED`` or more) and three quarters of the first round's ring steps waited longer
        than ``BEHIND_STEP_SECONDS`` on most ranks (the median over the ranks of each rank's lower quartile: a slow
        link, not a late or descheduled rank), the exchange of every later round but the last runs on a helper thread
        while the next round is filled and transformed, so a rank then holds the irreducible rows of two rounds. The
        schedule depends only on the column counts, the box and the rank count, so the exchange stays collectively
        matched. A column's values depend neither on the schedule nor on the groups: its product is elementwise and
        the transform runs every momentum line of every component on its own.

        :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
        :param k_grid: The :class:`KGrid` over which the BZ sum/FFT is performed.
        :param n_w: Number of bosonic frequencies of the columns.
        :param n_v: Number of fermionic frequencies of the columns.
        :param n_parts: Number of products per column.
        :param comp_shape: Shape of the orbital axes of a column, ``(nb, nb, nb, nb)`` or a flat count of components.
        :param dtype: Storage dtype of the products.
        :param fill: Writes the R-space product ``part`` of the flat components ``comps`` of the column ``(iw, iv)``
            into ``out`` ``[x, y, z, comps.stop - comps.start]``.
        :param deliver: Receives every column block of this rank's momenta.
        :param columns: The flat columns to compute, ascending, or ``None`` for all of them.
        :param group: Number of flat components transformed at once, or ``None`` for whole columns.
        :return: None.
        """
        comm = mpi_dist_irrk.comm
        rank, size = comm.rank, comm.size
        nk_tot, nk_irr = k_grid.nk_tot, len(k_grid.irrk_ind)
        itemsize = np.dtype(dtype).itemsize

        # globally derived round-major schedule, so the ring exchange stays collectively matched on every rank
        total_cols = n_w * n_v if columns is None else len(columns)
        ncols_max = -(-total_cols // size)
        sub_cols = max(1, ncols_max * nk_irr // (6 * nk_tot), min(ncols_max, n_v // max(1, size)))
        n_comp = int(np.prod(comp_shape))
        sub_cols = max(1, min(sub_cols, ROUND_BUFFER_BYTES // (nk_tot * n_comp * itemsize)))
        n_rounds = -(-total_cols // (size * sub_cols))

        my_q_slice = mpi_dist_irrk.slices[rank]
        my_nq = (my_q_slice.stop - my_q_slice.start) if my_q_slice is not None else 0

        def round_cols(r: int, rnd: int) -> tuple[int, int]:
            c0 = min(total_cols, (rnd * size + r) * sub_cols)
            return c0, min(total_cols, c0 + sub_cols)

        def targets(c0: int, c1: int) -> slice | np.ndarray:
            """The flat columns of the round positions ``[c0, c1)``."""
            return slice(c0, c1) if columns is None else columns[c0:c1]

        # one-column rounds receive into the slots of a staging buffer, handed to deliver a batch of slots at a time
        slot_bytes = my_nq * n_comp * n_parts * itemsize
        n_slots = max(1, min(size - 1, DELIVER_BATCH_BYTES // slot_bytes)) if slot_bytes else 1
        staging = np.empty((n_slots, my_nq, *comp_shape, n_parts), dtype=dtype) if sub_cols == 1 else None

        def exchange(rnd: int, chunk_irr: np.ndarray | None, n_mine: int, wait: Callable[[list], None]) -> list:
            """Ring exchange of round ``rnd``'s rows onto the irr-BZ q-distribution, one rank distance per step; returns
            the seconds every step waited for its transfers."""
            # the round positions of the staged blocks, descending as the ring's sources do
            staged, waits = [], []

            def hand_over():
                if staged:
                    # the staged blocks reversed are the consecutive columns staged[-1] .. staged[0]
                    deliver(targets(staged[-1], staged[0] + 1), np.moveaxis(staging[len(staged) - 1 :: -1], 0, -1))
                    staged.clear()

            for step in range(1, size):
                dst, src = (rank + step) % size, (rank - step) % size
                reqs, block = [], None

                dst_slice = mpi_dist_irrk.slices[dst]
                n_dst = (dst_slice.stop - dst_slice.start) if dst_slice is not None else 0
                if n_mine and n_dst:
                    reqs += mpi_utils._isend_rows(comm, chunk_irr[dst_slice], dst, base_tag=500 + step)

                s0, s1 = round_cols(src, rnd)
                if my_nq and s1 > s0:
                    if staging is None:
                        block = np.empty((my_nq, *comp_shape, n_parts * (s1 - s0)), dtype=dtype)
                    else:
                        block = staging[len(staged)]
                    reqs += mpi_utils._irecv_rows_into(comm, block, src, base_tag=500 + step)

                if reqs:
                    t0 = time.perf_counter()
                    wait(reqs)
                    waits.append(time.perf_counter() - t0)
                if block is not None and staging is None:
                    deliver(targets(s0, s1), block.reshape(my_nq, *comp_shape, n_parts, s1 - s0))
                elif block is not None:
                    staged.append(s0)
                    # the next source holds the next lower column, unless the ring wraps around past rank 0
                    if len(staged) == n_slots or src == 0:
                        hand_over()
            hand_over()
            return waits

        # from round 1 on, a round's exchange may run behind the next round's columns: when MPI lets a helper thread
        # call it meanwhile and most steps of the first round waited long, i.e. on a slow link
        behind = size > 1 and n_rounds > 2 and mpi_utils.MPI.Query_thread() >= mpi_utils.MPI.THREAD_SERIALIZED
        join = None
        group = n_comp if group is None else max(1, min(n_comp, group))
        buf = np.empty(nk_tot * group, dtype=dtype)
        for rnd in range(n_rounds):
            c0, c1 = round_cols(rank, rnd)
            n_mine = max(0, c1 - c0)
            chunk_irr = None
            if n_mine:
                chunk_irr = np.empty((nk_irr, n_comp, n_parts * n_mine), dtype=dtype)
                for part in range(n_parts):
                    for col in range(c0, c1):  # one (w, v) pair per column, its components in groups
                        iw, iv = divmod(col if columns is None else columns[col], n_v)
                        for g0 in range(0, n_comp, group):
                            comps = slice(g0, min(n_comp, g0 + group))
                            out = buf[: nk_tot * (comps.stop - g0)].reshape(*k_grid.nk, comps.stop - g0)
                            fill(out, part, iw, iv, comps)
                            # scipy.fft keeps complex64 in place; numpy.fft would upcast to complex128
                            transformed = sp.fft.ifftn(out, axes=(0, 1, 2), overwrite_x=True)
                            rows = transformed.reshape(nk_tot, -1)[k_grid.irrk_ind]
                            chunk_irr[:, comps, part * n_mine + col - c0] = rows
                            transformed = rows = None
                chunk_irr = chunk_irr.reshape(nk_irr, *comp_shape, n_parts * n_mine)
                if my_nq:
                    deliver(targets(c0, c1), chunk_irr[my_q_slice].reshape(my_nq, *comp_shape, n_parts, n_mine))

            if join is not None:
                join()
            join = None
            if behind and 0 < rnd < n_rounds - 1:
                join = mpi_utils.in_background(exchange, rnd, chunk_irr, n_mine, mpi_utils.wait_idle)
                continue
            waits = exchange(rnd, chunk_irr, n_mine, mpi_utils.MPI.Request.Waitall)
            if behind and rnd == 0:
                # a late or descheduled rank lengthens a few steps; a slow link three quarters of them on most ranks
                wait = float(np.median(comm.allgather(float(np.percentile(waits, 25)) if waits else 0.0)))
                behind = wait > BEHIND_STEP_SECONDS
                config.logger.info(
                    f"Bubble column exchange: the first round's ring steps waited {1e3 * wait:.2f} ms (lower quartile, "
                    f"median over the ranks), so the later rounds run {'behind' if behind else 'in front of'} the "
                    f"next round's columns."
                )

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

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Labels of the gap functions the Eliashberg step returns: the spatial wave and the frequency parity of every orbital
block, and the ``s±`` test that reads the sign of the gap on the Fermi sheets of the bare bands. Pure array functions,
no communicator and no configuration.
"""

import numpy as np


def _classify_momentum_block(block: np.ndarray, purity: float) -> str:
    r"""
    Classifies the wave symmetry of one orbital block's momentum profile, returning ``s``, ``d``, ``dxy`` (``dxz``,
    ``dyz``), ``g``, ``p`` or ``x``.

    The block is decomposed into symmetry sectors rather than compared element-wise: for an involution :math:`X`,
    the fraction of the block's weight that is even under it is
    :math:`\lVert \tfrac{1}{2}(b + Xb)\rVert^{2}/\lVert b\rVert^{2}`, and the odd fraction is its complement. A
    sector counts as realized when its fraction reaches ``purity``. This tolerates the few percent of admixture
    that a converged gap generally carries, where an element-wise ``allclose`` would reject it outright.

    The involutions are the three coordinate inversions :math:`k_i \to -k_i` (in the ``np.roll(np.flip(...), 1)``
    convention of the Gamma-at-index-0 grid) and the axis exchanges :math:`k_i \leftrightarrow k_j`, the latter only
    for pairs of axes of equal length. All three exchanges are tested, not just :math:`k_x \leftrightarrow k_y`: on
    a cubic lattice a :math:`d`-wave living in the :math:`xz` plane is antisymmetric under
    :math:`k_x \leftrightarrow k_z` and carries no definite symmetry under :math:`k_x \leftrightarrow k_y` at all.

    Only the momentum profile enters here; the orbital indices are fixed by the caller. This is a diagnostic label, so
    unlike the mirrors of :func:`~dgamore.eliashberg_solver._mirror_operator` no orbital transformation accompanies the
    reflections.

    :param block: The momentum profile ``[kx, ky, kz]`` of one orbital block.
    :param purity: Minimum weight fraction for a symmetry sector to count as realized.
    :return: ``s`` (even under every inversion, symmetric under every applicable exchange), ``d`` (even under every
        inversion, antisymmetric under some exchange: the :math:`d_{x^2-y^2}` type), ``dxy`` (odd under
        :math:`k_x \to -k_x` and :math:`k_y \to -k_y`, even under the third, symmetric under their exchange; ``dxz``,
        ``dyz`` in the other planes), ``g`` (odd under two inversions, antisymmetric under their exchange), ``p`` (odd
        under the full inversion) or ``x``.
    """

    def even_fraction(arr: np.ndarray, mirrored: np.ndarray) -> float:
        norm = float(np.vdot(arr, arr).real)
        return float(np.vdot(arr + mirrored, arr + mirrored).real) / (4.0 * norm) if norm > 0 else 0.0

    def invert(arr: np.ndarray, axes: tuple) -> np.ndarray:
        return np.roll(np.flip(arr, axis=axes), shift=(1,) * len(axes), axis=axes)

    if float(np.vdot(block, block).real) <= 0:
        return "x"

    even_axis = [even_fraction(block, invert(block, (axis,))) for axis in range(3)]
    if all(frac >= purity for frac in even_axis):
        exchanges = [
            (i, j) for i, j in ((0, 1), (0, 2), (1, 2)) if block.shape[i] == block.shape[j] and block.shape[i] > 1
        ]
        swapped = [np.swapaxes(block, i, j) for i, j in exchanges]
        if any(even_fraction(block, -swap) >= purity for swap in swapped):  # antisymmetric under some exchange
            return "d"
        if all(even_fraction(block, swap) >= purity for swap in swapped):
            return "s"
        return "x"
    odd_axes = [axis for axis in range(3) if 1.0 - even_axis[axis] >= purity]
    if len(odd_axes) == 2 and even_axis[3 - sum(odd_axes)] >= purity:
        i, j = odd_axes
        if block.shape[i] == block.shape[j]:
            swapped = np.swapaxes(block, i, j)
            if even_fraction(block, swapped) >= purity:
                return f"d{'xyz'[i]}{'xyz'[j]}"
            if even_fraction(block, -swapped) >= purity:
                return "g"
        return "x"
    if even_fraction(block, -invert(block, (0, 1, 2))) >= purity:  # odd under the full inversion
        return "p"
    return "x"


def fermi_surface_bands(ek: np.ndarray, mu: float, orbitals: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Returns the band energies relative to the chemical potential and the band eigenvectors on the gap's orbitals,
    :math:`\varepsilon_n(\mathbf{k}) - \mu` and :math:`\psi_{an}(\mathbf{k})`, the input of the Fermi-surface sign test
    of :func:`classify_gap_symmetry`.

    :param ek: Band dispersion :math:`\varepsilon(\mathbf{k})` on the whole grid, shape ``[kx, ky, kz, o1, o2]``.
    :param mu: Chemical potential :math:`\mu`.
    :param orbitals: Indices of the orbitals the gap lives on (the correlated ones); ``None`` for every orbital.
    :return: ``(energies, vectors)`` of shapes ``[kx, ky, kz, n]`` and ``[kx, ky, kz, o, n]``.
    """
    energies, vectors = np.linalg.eigh(ek)
    return energies - mu, (vectors if orbitals is None else vectors[..., orbitals, :])


def _changes_sign_between_fermi_sheets(
    gap_slice: np.ndarray, bands: tuple[np.ndarray, np.ndarray], consistency: float = 0.5, floor: float = 0.05
) -> bool:
    r"""
    Whether a gap at one frequency has opposite signs on the Fermi sheets of two bands. The intraband gap
    :math:`\Delta_n(\mathbf{k}) = \psi_n(\mathbf{k})^\dagger \Delta(\mathbf{k})^{T} \psi_n(\mathbf{k})` (the band gap of
    the time-reversal gauge :math:`\psi(-\mathbf{k}) = \psi(\mathbf{k})^*` for the gap's first orbital at
    :math:`-\mathbf{k}`, independent of the eigenvector phases) is read on the momenta next to a sign change of
    :math:`\varepsilon_n - \mu` towards a grid neighbor, after the global phase that makes it most nearly real. A band
    counts when its sheet carries at least ``floor`` of the largest sheet's weight and keeps at least ``consistency`` of
    it in the signed sum (one sign on most of the sheet). Bands degenerate at a momentum to 1e-4 of the largest band
    energy all carry the trace over them there as their band gap, independent of the basis the eigensolver picked
    among them, so degenerate partners count with one sign, also where another band crosses them.

    :param gap_slice: The gap at one frequency, shape ``[kx, ky, kz, o1, o2]``.
    :param bands: The ``(energies, vectors)`` of :func:`fermi_surface_bands`.
    :param consistency: Minimum ratio of the signed to the absolute sum on a sheet.
    :param floor: Minimum weight of a sheet relative to the largest one.
    :return: True when two counted Fermi surfaces carry opposite signs.
    """
    energies, vectors = bands
    intra = np.einsum("xyzan,xyzba,xyzbn->xyzn", vectors.conj(), gap_slice, vectors)
    sheet = np.zeros(energies.shape, dtype=bool)
    for axis in range(3):
        if energies.shape[axis] > 1:
            crossing = energies * np.roll(energies, -1, axis=axis) < 0
            sheet |= crossing | np.roll(crossing, 1, axis=axis)
    if not sheet.any():
        return False
    # bands degenerate at a momentum all carry the trace over them there as their band gap
    degenerate = np.abs(energies[..., :, None] - energies[..., None, :]) <= 1e-4 * np.abs(energies).max()
    intra = np.einsum("xyznm,xyzm->xyzn", degenerate, intra)
    signed = (intra * np.exp(-0.5j * np.angle(np.sum(intra[sheet] ** 2)))).real
    sums = [
        (signed[..., n][sheet[..., n]].sum(), np.abs(intra[..., n][sheet[..., n]]).sum())
        for n in range(intra.shape[-1])
    ]
    largest = max(weight for _, weight in sums)
    signs = [
        np.sign(total) for total, weight in sums if weight > floor * largest and abs(total) >= consistency * weight
    ]
    return min(signs, default=0) < 0 < max(signs, default=0)


def classify_gap_symmetry(
    gap: np.ndarray,
    purity: float = 0.9,
    weight_floor: float = 0.01,
    bands: tuple[np.ndarray, np.ndarray] | None = None,
) -> str:
    r"""
    Classifies the momentum wave symmetry and Matsubara-frequency parity of a gap and returns a compact label of the
    form ``<wave><parity>``. The wave is ``s``, ``s±``, ``d`` (:math:`d_{x^2-y^2}` type), ``dxy`` (``dxz``, ``dyz``),
    ``g`` or ``p`` (``x`` if none of these match), and the parity sign is ``+`` (even in :math:`\nu`), ``-`` (odd) or
    empty (neither). The momentum tests are reflections of the grid axes, so the names presume the axes of a square or
    cubic lattice. With ``bands`` an ``s`` wave whose band gap has opposite signs on the Fermi sheets of two bands
    becomes ``s±`` (see :func:`_changes_sign_between_fermi_sheets`). The frequency parity is the sign of the
    global T Rayleigh quotient :math:`\langle \Delta, T\Delta \rangle / \langle \Delta, \Delta \rangle` (with
    :math:`T` the fermionic-frequency flip), so it is consistent with the parity diagnostics.

    Every orbital block :math:`(o_1, o_2)` carrying at least ``weight_floor`` of the heaviest block's weight is
    classified separately from its momentum profile at the first positive Matsubara frequency (see
    :func:`_classify_momentum_block`), because a multi-orbital gap need not carry the same wave in every block: the
    :math:`E_g` singlet of a cubic :math:`t_{2g}` system, for instance, is a :math:`d`-wave in each block but a
    *different* one each time, :math:`\cos k_x - \cos k_y` on :math:`d_{xy}` and its cyclic images on the other two.
    Blocks below the floor are skipped - the wave symmetry of a numerically negligible block is noise.

    Blocks that agree collapse to a single label, so the common case stays one token wide. When they disagree the
    label lists each distinct wave with the blocks realizing it, e.g. ``d+[00,11]|s+[22]``, ordered by weight; the
    number of entries is bounded by the number of distinct waves, never by the number of orbital blocks.

    :param gap: The gap array in the ``[kx, ky, kz, o1, o2, v]`` layout.
    :param purity: Minimum weight fraction for a symmetry sector to count as realized within a block.
    :param weight_floor: Minimum weight of a block relative to the heaviest one for it to be classified at all.
    :param bands: The band energies and eigenvectors of :func:`fermi_surface_bands`, for the ``s±`` test; ``None``
        skips it.
    :return: The ``<wave><parity>`` label, or ``"unknown"`` for an all-zero gap.
    """
    denom = np.vdot(gap, gap)
    if denom == 0:
        return "unknown"
    t = (np.vdot(gap, np.flip(gap, axis=-1)) / denom).real
    freq_label = "+" if t > 0.5 else ("-" if t < -0.5 else "")

    n_o1, n_o2 = gap.shape[3], gap.shape[4]
    weights = np.linalg.norm(gap.reshape(-1, n_o1, n_o2, gap.shape[-1]), axis=(0, -1))
    if weights.max() <= 0:
        return f"x{freq_label}"

    labeled: dict[str, list[tuple[float, str]]] = {}
    for o1 in range(n_o1):
        for o2 in range(n_o2):
            if weights[o1, o2] < weight_floor * weights.max():
                continue
            wave = _classify_momentum_block(gap[:, :, :, o1, o2, gap.shape[-1] // 2], purity)
            labeled.setdefault(wave, []).append((float(weights[o1, o2]), f"{o1}{o2}"))

    if not labeled:
        return f"x{freq_label}"
    if "s" in labeled and bands is not None and _changes_sign_between_fermi_sheets(gap[..., gap.shape[-1] // 2], bands):
        labeled["s±"] = labeled.pop("s")
    if len(labeled) == 1:
        return f"{next(iter(labeled))}{freq_label}"
    order = sorted(labeled.items(), key=lambda kv: -sum(w for w, _ in kv[1]))
    return "|".join(
        f"{wave}{freq_label}[{','.join(block for _, block in sorted(blocks, key=lambda wb: -wb[0]))}]"
        for wave, blocks in order
    )

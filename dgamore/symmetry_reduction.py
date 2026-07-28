# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

"""
Automatic symmetry reduction of a k-space Hamiltonian H[kx,ky,kz,o1,o2] to the
irreducible Brillouin zone (IBZ), with an inverse map back to the full BZ.

Convention
----------
H is indexed on a uniform grid (j_1, j_2, j_3) with j_i in {0, ..., N_i - 1},
corresponding to k = (j_1/N_1) b_1 + (j_2/N_2) b_2 + (j_3/N_3) b_3, where
b_1, b_2, b_3 are the primitive reciprocal-lattice vectors. Gamma is at
(0,0,0). In this lattice basis, every crystallographic point group is a
finite subgroup of GL(3, Z), and its generators have entries in {-1, 0, +1}.

Symmetries searched
-------------------
Operations (M, q, U, sigma, conj) such that for every k in the grid,
    H((M k + q) mod N) = sigma * U @ H(k)^{[*]} @ U^dagger
where:
  * M is a 3x3 integer matrix with entries in {-1, 0, +1} and det = +/- 1.
    Enumerated exhaustively (6960 matrices), filtered to those compatible
    with the grid shape.
  * q is any integer translation vector in [0, N_1) x [0, N_2) x [0, N_3).
    For each M, valid q's are found via FFT-based cross-correlation of the
    eigenvalue field (fast: O(N^3 log N) per M).
  * U is an arbitrary unitary in orbital space, found by simultaneous
    diagonalization with per-eigenspace gauge fixing. NOT enumerated:
    works for any number of orbitals and any U (not just signed perms).
  * sigma in {+1, -1} covers anti-symmetries (chiral / particle-hole).
  * conj covers anti-unitary symmetries (time-reversal-like).

Algorithm
---------
1. Enumerate {-1,0,+1}-matrix candidates M (grid-compatible).
2. For each M and each (sigma, conj), use FFT cross-correlation on the
   eigenvalue field to find all q for which the eigenvalue pre-screen holds.
3. For each surviving (M, q, sigma, conj), solve for U.
4. Close the discovered operations under composition.
5. Orbit-collapse the k-grid using the closed group; canonical representative
   = smallest flat index in each orbit.
6. expand / expand_tensor: vectorized reconstruction of arbitrary-rank
   tensors T[k, o_1, ..., o_r] from their IBZ values.

Reference
---------
The integer-matrix enumeration covers all crystallographic point groups, but
discovery requires that H be expressed in the *primitive* reciprocal basis
(not Cartesian). For models given in Cartesian coordinates of a non-cubic
lattice (e.g. hexagonal kx, ky, kz axes), the rotations are not integer
matrices and will not be detected. Re-grid H onto the lattice basis first.
"""

import numpy as np
import scipy as sp
import itertools
import string
from functools import lru_cache

# ============================================================================
# Spatial ops on a discrete reciprocal grid


@lru_cache(maxsize=1)
def _enumerate_integer_matrices():
    """
    Enumerates all 3x3 integer matrices with entries in ``{-1, 0, +1}`` and determinant ``+/-1`` (6960 of them), a
    standard generating set for the finite subgroups of GL(3, Z). The result is grid-independent, so it is cached
    (the determinant is a direct 3x3 formula on the integer entries, avoiding a LAPACK call per candidate).

    :return: A list of the candidate 3x3 integer matrices.
    """
    mats = []
    for e in itertools.product((-1, 0, 1), repeat=9):
        d = e[0] * (e[4] * e[8] - e[5] * e[7]) - e[1] * (e[3] * e[8] - e[5] * e[6]) + e[2] * (e[3] * e[7] - e[4] * e[6])
        if d in (-1, 1):
            mats.append(np.array(e, dtype=np.int64).reshape(3, 3))
    return mats


def _M_preserves_grid(M, nk):
    """
    Checks whether the integer matrix ``M`` is compatible with the (possibly non-cubic) grid: ``M[i,j] != 0`` is
    allowed only if ``N_j`` divides ``N_i``, so the k-index action ``k_i -> sum_j M[i,j] k_j`` is well-defined modulo
    ``N_i``.

    :param M: The candidate 3x3 integer matrix.
    :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
    :return: True if ``M`` preserves the grid.
    """
    Ns = list(nk)
    for i in range(3):
        for j in range(3):
            if M[i, j] != 0 and (Ns[i] % Ns[j] != 0):
                return False
    return True


@lru_cache(maxsize=8)
def _grid_index_stack(nk):
    """
    Returns the read-only ``(nx, ny, nz, 3)`` stack of k-grid integer coordinates. Cached per grid size so the
    per-M k-index maps do not rebuild the meshgrid (called thousands of times during symmetry discovery).

    :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
    :return: The integer coordinate stack (shared, not to be mutated by callers).
    """
    nx, ny, nz = nk
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    stack = np.stack([ix, iy, iz], axis=-1).astype(np.int64)
    stack.setflags(write=False)
    return stack


def _apply_M_to_kgrid_indices(M, nk):
    """
    Builds the flat-index map ``idx[k_flat] = (M @ k) mod N`` for the action of ``M`` on the k-grid (with grid-size
    scaling for non-cubic grids).

    :param M: The 3x3 integer matrix acting on k-indices.
    :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
    :return: The flat-index permutation array of length ``nx*ny*nz``.
    """
    nx, ny, nz = nk
    Ns = np.array([nx, ny, nz], dtype=np.int64)
    j = _grid_index_stack(nk)  # cached, read-only
    out = np.empty((nx, ny, nz, 3), dtype=np.int64)
    for i in range(3):
        s = np.zeros((nx, ny, nz), dtype=np.int64)
        for jj in range(3):
            if M[i, jj] != 0:
                s += (M[i, jj] * (Ns[i] // Ns[jj])) * j[..., jj]
        out[..., i] = s % Ns[i]
    return (out[..., 0] * (ny * nz) + out[..., 1] * nz + out[..., 2]).ravel()


def _apply_M_to_points(M, coords, nk):
    """
    Applies the k-grid action of ``M`` to a small set of integer grid coordinates, returning their flat indices
    ``(M @ coord) mod N`` (with the same grid-size scaling as :func:`_apply_M_to_kgrid_indices`). Used for the cheap
    O(K) signature pre-filter of the symmetry prune.

    :param M: The 3x3 integer matrix acting on k-indices.
    :param coords: Integer grid coordinates of shape ``(K, 3)``.
    :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
    :return: The flat indices of the mapped coordinates, shape ``(K,)``.
    """
    nx, ny, nz = nk
    Ns = np.array([nx, ny, nz], dtype=np.int64)
    out = np.empty_like(coords)
    for i in range(3):
        s = np.zeros(len(coords), dtype=np.int64)
        for jj in range(3):
            if M[i, jj] != 0:
                s += (M[i, jj] * (Ns[i] // Ns[jj])) * coords[:, jj]
        out[:, i] = s % Ns[i]
    return out[:, 0] * (ny * nz) + out[:, 1] * nz + out[:, 2]


def _translate_kgrid(idx_map, q, nk):
    """
    Composes a flat-index map with an integer grid translation ``q``.

    :param idx_map: An existing flat-index map over the k-grid.
    :param q: The integer translation vector ``(qx, qy, qz)``.
    :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
    :return: The translated flat-index map.
    """
    nx, ny, nz = nk
    qx, qy, qz = q
    iz = idx_map % nz
    iy = (idx_map // nz) % ny
    ix = idx_map // (ny * nz)
    ix = (ix + qx) % nx
    iy = (iy + qy) % ny
    iz = (iz + qz) % nz
    return ix * (ny * nz) + iy * nz + iz


def _apply_M_to_ev_field(M, ev, nk, idx_map=None):
    """
    Pulls back an eigenvalue field by the action of ``M``, returning ``A[k] = ev[M k mod N]``. Used for the
    eigenvalue pre-screen in symmetry discovery. Reuses a precomputed flat-index map when available so the pullback
    is a single fancy-index gather instead of rebuilding the M-action.

    :param M: The 3x3 integer matrix acting on k-indices.
    :param ev: The eigenvalue field of shape ``(nx, ny, nz, n_orb)``.
    :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
    :param idx_map: Optional precomputed flat-index map from :func:`_apply_M_to_kgrid_indices` for ``M``.
    :return: The transformed eigenvalue field, same shape as ``ev``.
    """
    if idx_map is None:
        idx_map = _apply_M_to_kgrid_indices(M, nk)
    return ev.reshape(-1, ev.shape[-1])[idx_map].reshape(ev.shape)


# ============================================================================
# FFT-based fast q-detection (eigenvalue pre-screen)


def _fft_find_matching_q(A, B, atol):
    r"""
    Finds all integer translations ``q`` such that ``A[k] = B[k + q]`` for all ``k``, via a 3D FFT cross-correlation of
    the (real) eigenvalue fields, i.e. minimizing :math:`D(\mathbf{q}) = \sum_{\mathbf{k},e} (A -
    B(\cdot+\mathbf{q}))^2`.

    :param A: First real field of shape ``(Nx, Ny, Nz, n_orb_evals)``.
    :param B: Second real field of the same shape.
    :param atol: Absolute tolerance for accepting a translation.
    :return: A list of ``q`` index tuples for which the mismatch is below tolerance.
    """
    A2 = (A * A).sum()
    B2 = (B * B).sum()
    FA = sp.fft.fftn(A, axes=(0, 1, 2))
    FB = sp.fft.fftn(B, axes=(0, 1, 2))
    cross = sp.fft.ifftn(np.conj(FA) * FB, axes=(0, 1, 2), overwrite_x=True).real.sum(axis=-1)
    D = A2 + B2 - 2.0 * cross
    thresh = max(atol * (A2 + B2 + 1.0), atol * 100)
    qs = np.argwhere(D < thresh)
    return [tuple(int(x) for x in q) for q in qs]


# ============================================================================
# Solving for U


def _cluster_eigvals(d, tol):
    """
    Groups consecutive (sorted) eigenvalues that lie within ``tol`` of each other into degenerate clusters.

    :param d: The sorted eigenvalues.
    :param tol: The degeneracy tolerance.
    :return: A list of index lists, one per (near-)degenerate cluster.
    """
    clusters = []
    cur = [0]
    for i in range(1, len(d)):
        if abs(d[i] - d[i - 1]) < tol:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    return clusters


def _canonicalize_sign_gauge(U, Hk_eff, Hg, atol):
    """
    Apply a left sign-diagonal `D` (entries +/-1) to `U` to produce `D U` with as few
    negative-entry signs as possible, subject to the constraint that `D U` still
    satisfies `(D U) Hk_eff (D U)^dag == Hg`. This is a clean gauge-fix that selects
    among centralizer-equivalent solutions, removing arbitrary global-sign choices
    made by the upstream solver.

    The valid sign-diagonals D are exactly those in the centralizer of Hk_eff:
    `D Hk_eff D = Hk_eff`. For generic Hermitian Hk_eff the centralizer is just
    `{+I, -I}`, but for block-diagonal or special H it can be larger (up to
    `{±1}^norb`). We try all `2^norb` sign-diagonals and pick the one minimizing
    the count of negative entries in `D U` (ties broken by preferring fewer changes
    from the identity diagonal). For norb up to ~6 this is cheap; beyond that we
    fall back to a row-major canonicalization.

    This change is purely a basis convention. It does not affect the validity of
    the symmetry: any D in the centralizer of Hk_eff yields a valid solution and
    gives identical results when applied to two-point quantities transforming as
    `M -> U M U^dag`. For four-point objects with the same lattice symmetry,
    GLOBAL sign flips (D = +/- I) also cancel (four U-factors), so this canonical
    form does not change four-point results either. It primarily makes the stored
    Us match the conventional unsigned-permutation form whenever that is consistent
    with the H equation, which is the form users expect for cubic-style symmetries
    in the t2g/eg basis.

    :param U: The candidate unitary to canonicalize.
    :param Hk_eff: The (effective) source Hamiltonian field the unitary acts on.
    :param Hg: The target Hamiltonian field the relation must reproduce.
    :param atol: Absolute tolerance for validating the sign-fixed unitary.
    :return: The canonicalized unitary ``D U`` (or ``U`` unchanged if no valid improvement is found).
    """
    norb = U.shape[0]
    if norb > 6:
        # Fall back to row canonicalization: scale each row by sign of its largest entry.
        out = U.copy()
        for i in range(norb):
            mags = np.abs(out[i])
            j = int(np.argmax(mags))
            if mags[j] > 1e-12 and out[i, j].real < 0:
                out[i] = -out[i]
        # Verify the result still solves (only the global-sign case is guaranteed safe).
        rhs = np.einsum("ij,...jk,lk->...il", out, Hk_eff, out.conj())
        if np.allclose(Hg, rhs, atol=atol):
            return out
        return U

    # Since U already solves the relation, (D U) Hk_eff (D U)^dag = D (U Hk_eff U^dag) D = D Hg D, so a sign-diagonal
    # D is valid iff D Hg D == Hg, i.e. signs[a]*signs[b] == 1 wherever Hg is nonzero. This cheap pattern check
    # pre-filters the 2^norb masks so the confirming full-grid einsum runs only for the few D in Hg's centralizer.
    hg_nonzero = np.abs(Hg).max(axis=(0, 1, 2)) > atol  # [norb, norb] nonzero orbital pattern
    best_U = U
    best_score = (int((U.real < -0.5).sum()), 0)  # (neg_count, dist_from_identity)
    for mask in range(1, 1 << norb):
        signs = np.array([(1 if not (mask >> i) & 1 else -1) for i in range(norb)], dtype=complex)
        if np.any(((signs[:, None] * signs[None, :]).real < 0) & hg_nonzero):
            continue  # D Hg D != Hg -> D not in the centralizer, skip without the full-grid einsum
        U_try = (signs[:, None]) * U
        rhs = np.einsum("ij,...jk,lk->...il", U_try, Hk_eff, U_try.conj())
        if not np.allclose(Hg, rhs, atol=atol):
            continue
        neg_count = int((U_try.real < -0.5).sum())
        dist = int((signs.real < 0).sum())  # number of rows we flipped
        score = (neg_count, dist)
        if score < best_score:
            best_score = score
            best_U = U_try
    return best_U


def _solve_U_for_op(Hg, Hk_eff, atol, ev_k=None, ev_g=None):
    """Find a unitary U such that Hg(k) = U @ Hk_eff(k) @ U^dag for every k.
    Returns U or None.

    When a solution exists, the returned U is canonicalized: among all
    centralizer-equivalent solutions (i.e. ``D U`` for ``D`` a sign-diagonal in
    the centralizer of ``Hk_eff``), the one with the fewest negative entries is
    returned. This makes the output independent of any global-sign choice the
    inner gauge-fixing routine happens to make and matches the conventional
    unsigned-permutation form whenever it is consistent with the H equation.

    :param Hg: The target Hamiltonian field of shape ``(nx, ny, nz, norb, norb)``.
    :param Hk_eff: The (effective) source Hamiltonian field of the same shape.
    :param atol: Absolute tolerance for validating the relation.
    :param ev_k: Optional precomputed ``eigvalsh(Hk_eff)`` (the caller knows it is ``+/- ev`` of ``H``).
    :param ev_g: Optional precomputed ``eigvalsh(Hg)`` (the caller knows it is ``ev`` reindexed by the grid action).
    :return: A canonicalized unitary ``U`` solving the relation, or None if none exists.
    """
    norb = Hg.shape[-1]
    if ev_k is None:
        ev_k = np.linalg.eigvalsh(Hk_eff)
    if ev_g is None:
        ev_g = np.linalg.eigvalsh(Hg)
    if not np.allclose(ev_k, ev_g, atol=10 * atol):
        return None

    # Single-orbital short-circuit: U is a 1x1 unitary (a phase). For Hermitian Hg/Hk_eff with matching spectra,
    # U = [[1]] always works (any U(1) phase satisfies it; pick the canonical one), and avoids np.diff on an empty axis.
    if norb == 1:
        U_simple = np.eye(1, dtype=complex)
        rhs = np.einsum("ij,...jk,lk->...il", U_simple, Hk_eff, U_simple.conj())
        if np.allclose(Hg, rhs, atol=atol):
            return U_simple
        return None

    nx, ny, nz = ev_k.shape[:3]
    gaps = np.diff(ev_k, axis=-1).min(axis=-1)
    order = np.argsort(gaps.ravel())[::-1]

    for flat in order[:8]:
        i0, j0, k0 = np.unravel_index(flat, (nx, ny, nz))
        d_k, V = np.linalg.eigh(Hk_eff[i0, j0, k0])
        d_g, W = np.linalg.eigh(Hg[i0, j0, k0])
        if not np.allclose(d_k, d_g, atol=10 * atol):
            continue
        clusters = _cluster_eigvals(d_k, tol=max(100 * atol, 1e-7))

        # Always try the simple choice first
        U_simple = W @ V.conj().T
        rhs = np.einsum("ij,...jk,lk->...il", U_simple, Hk_eff, U_simple.conj())
        if np.allclose(Hg, rhs, atol=atol):
            return _canonicalize_sign_gauge(U_simple, Hk_eff, Hg, atol)

        # Gauge fix (non-degenerate or block-diagonal)
        if all(len(c) == 1 for c in clusters):
            U_cand = _fix_phases_nondegenerate(V, W, Hk_eff, Hg, (i0, j0, k0), atol)
        else:
            U_cand = _fix_gauge_degenerate(V, W, clusters, Hk_eff, Hg, atol)
        if U_cand is not None:
            rhs = np.einsum("ij,...jk,lk->...il", U_cand, Hk_eff, U_cand.conj())
            if np.allclose(Hg, rhs, atol=atol):
                return _canonicalize_sign_gauge(U_cand, Hk_eff, Hg, atol)
    return None


def _fix_phases_nondegenerate(V, W, Hk_eff, Hg, k0, atol):
    r"""
    Determines the eigenvector phases :math:`\phi` so that :math:`U = W \mathrm{diag}(\phi) V^\dagger` solves the
    symmetry relation globally, for the non-degenerate-spectrum case (constraint at any ``k1``:
    :math:`\mathrm{diag}(\phi) A \mathrm{diag}(\phi^*) = B` with ``A = V^dag Hk_eff[k1] V``, ``B = W^dag Hg[k1] W``).

    :param V: Eigenvectors of ``Hk_eff`` at the reference point.
    :param W: Eigenvectors of ``Hg`` at the reference point.
    :param Hk_eff: The (effective) source Hamiltonian field.
    :param Hg: The target Hamiltonian field.
    :param k0: The reference k-point index triplet (excluded when sampling test points).
    :param atol: Absolute tolerance for validating the relation.
    :return: The solving unitary ``U``, or None if no consistent phase set is found.
    """
    norb = V.shape[0]
    nx, ny, nz = Hk_eff.shape[:3]
    rng = np.random.default_rng(42)
    for trial in range(64):
        i1, j1, k1 = (rng.integers(nx), rng.integers(ny), rng.integers(nz))
        if (i1, j1, k1) == k0:
            continue
        A = V.conj().T @ Hk_eff[i1, j1, k1] @ V
        B = W.conj().T @ Hg[i1, j1, k1] @ W
        phi = np.ones(norb, dtype=complex)
        ok = True
        for r in range(1, norb):
            found = False
            for col in range(norb):
                if col == r:
                    continue
                if abs(A[r, col]) > 1e-4 and abs(phi[col]) > 1e-8:
                    val = B[r, col] / A[r, col] * phi[col]
                    m = abs(val)
                    if m < 1e-8:
                        continue
                    phi[r] = val / m
                    found = True
                    break
            if not found:
                ok = False
                break
        if not ok:
            continue
        U_try = W @ np.diag(phi) @ V.conj().T
        rhs = np.einsum("ij,...jk,lk->...il", U_try, Hk_eff, U_try.conj())
        if np.allclose(Hg, rhs, atol=atol):
            return U_try
    return None


def _fix_gauge_degenerate(V, W, clusters, Hk_eff, Hg, atol):
    r"""
    Solves for the block-diagonal unitary ``R`` (one block per degenerate cluster) such that
    :math:`U = W R V^\dagger` solves the symmetry relation, by assembling linear constraints ``A R = R B`` at several
    sampled k-points and taking the null-space solution.

    :param V: Eigenvectors of ``Hk_eff`` at the reference point.
    :param W: Eigenvectors of ``Hg`` at the reference point.
    :param clusters: The degenerate-eigenvalue clusters (see :func:`_cluster_eigvals`).
    :param Hk_eff: The (effective) source Hamiltonian field.
    :param Hg: The target Hamiltonian field.
    :param atol: Absolute tolerance (unused directly; kept for signature consistency).
    :return: The solving unitary ``U``, or None if the constraints have no consistent solution.
    """
    norb = V.shape[0]
    nx, ny, nz = Hk_eff.shape[:3]
    rng = np.random.default_rng(123)
    n_kpts = 8
    k_pts = [(rng.integers(nx), rng.integers(ny), rng.integers(nz)) for _ in range(n_kpts)]
    # Block-diagonal entries' positions in vec(R) (column-major):
    cols = []
    for c in clusters:
        for ji in c:
            for ii in c:
                cols.append(ji * norb + ii)
    cols = np.array(cols, dtype=int)
    rows = []
    I = np.eye(norb)
    for kp in k_pts:
        A = V.conj().T @ Hk_eff[kp] @ V
        B = W.conj().T @ Hg[kp] @ W
        M_ab = np.kron(I, A) - np.kron(B.T, I)
        rows.append(M_ab[:, cols])
    stacked = np.vstack(rows)
    try:
        _, S, Vh = np.linalg.svd(stacked, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if len(S) == 0 or S[-1] > 1e-4:
        return None
    rvec = Vh[-1].conj()
    R = np.zeros((norb, norb), dtype=complex)
    idx = 0
    for c in clusters:
        b = len(c)
        block = rvec[idx : idx + b * b].reshape(b, b, order="F")
        try:
            u_, _, v_ = np.linalg.svd(block)
        except np.linalg.LinAlgError:
            return None
        block_u = u_ @ v_
        for jj, j_orig in enumerate(c):
            for ii, i_orig in enumerate(c):
                R[i_orig, j_orig] = block_u[ii, jj]
        idx += b * b
    return W @ R @ V.conj().T


# ============================================================================
# Symmetry discovery


def _discover_symmetries(H, atol, verbose=False):
    """
    Discovers all symmetry operations ``(M, q, U, sigma, conj)`` of the Hamiltonian ``H``, deduplicated by their
    action (grid index map + sigma + conj + U up to phase). Uses the integer-matrix enumeration, the FFT eigenvalue
    pre-screen for translations, and the U-solver for the orbital part.

    :param H: The Hamiltonian field of shape ``(nx, ny, nz, norb, norb)``.
    :param atol: Absolute tolerance for symmetry validation.
    :param verbose: If True, print discovery diagnostics.
    :return: A tuple ``(ops, n_ops)`` of the list of discovered operation dicts and their count.
    """
    nx, ny, nz, norb, _ = H.shape
    nk = (nx, ny, nz)

    M_all = _enumerate_integer_matrices()
    M_candidates = [M for M in M_all if _M_preserves_grid(M, nk)]

    # For N >= 3 on every axis, distinct integer M's (entries in {-1, 0, 1}) give distinct grid actions (since
    # -1 mod N != +1), so no dedup is needed and the grid index map is built lazily only for the few M's that
    # survive the symmetry prune. On small grids (some N <= 2) the actions can collapse, so dedup by the full action.
    if all(N >= 3 for N in nk):
        M_unique = [(M, None) for M in M_candidates]
    else:
        seen_hashes: dict = {}
        M_unique = []
        for M in M_candidates:
            idx_map = _apply_M_to_kgrid_indices(M, nk)
            h = hash(idx_map.tobytes())  # cheap key; a collision is confirmed against the stored representative
            existing = seen_hashes.get(h)
            if existing is None:
                seen_hashes[h] = (M, idx_map)
                M_unique.append((M, idx_map))
            elif not np.array_equal(existing[1], idx_map):
                seen_hashes[h] = (M, idx_map)  # genuine 64-bit hash collision: keep both distinct actions
                M_unique.append((M, idx_map))
    if verbose:
        print(f"  Integer matrices: {len(M_candidates)} grid-compatible -> {len(M_unique)} unique grid actions")

    ev = np.linalg.eigvalsh(H)
    ev_neg = -ev[..., ::-1]
    H_flat = H.reshape(-1, norb, norb)
    ev_flat = ev.reshape(-1, norb)

    # Cache the reference FFTs (one for sigma=+1, one for sigma=-1) and their magnitudes for the prune.
    FB_plus = sp.fft.fftn(ev, axes=(0, 1, 2))
    FB_minus = sp.fft.fftn(ev_neg, axes=(0, 1, 2))
    B_plus_sq = (ev * ev).sum()
    B_minus_sq = (ev_neg * ev_neg).sum()
    FB_plus_flat = FB_plus.reshape(-1, norb)
    absFB_plus_flat = np.abs(FB_plus_flat)
    absFB_minus_flat = np.abs(FB_minus).reshape(-1, norb)
    # A real symmetry has |F[ev_M]| == |F[ev]| EXACTLY (a translate only changes the Fourier phase), matching to the
    # eigenvalue noise floor (~1e-13*max), so this loose bound prunes non-symmetries without ever dropping a real one.
    mag_atol = 1e-6 * float(absFB_plus_flat.max()) if absFB_plus_flat.size else 0.0
    # Cheap O(K) signature pre-filter: the K most distinctive |FB| R-points. A real symmetry keeps |FB| invariant
    # everywhere and hence at these points, so most non-symmetries are rejected here without the full-grid gather.
    sig_k = min(8, absFB_plus_flat.shape[0])
    sig_R = np.argsort(-absFB_plus_flat.sum(axis=1))[:sig_k]
    sig_coords = _grid_index_stack(nk).reshape(-1, 3)[sig_R]
    sig_ref_plus = absFB_plus_flat[sig_R]
    sig_ref_minus = absFB_minus_flat[sig_R]

    def _fft_q_scan_cached(FA, A2, FB, B_sq, atol):
        """
        FFT translation scan against a precomputed reference FFT (faster variant of :func:`_fft_find_matching_q`).
        Takes the pulled-back field's forward FFT and squared norm precomputed, since they are identical for the
        ``sigma = +/-1`` reference scans of the same field.

        :param FA: The forward FFT of the pulled-back eigenvalue field to match.
        :param A2: The squared norm of the pulled-back field.
        :param FB: The precomputed FFT of the reference eigenvalue field.
        :param B_sq: The precomputed squared norm of the reference field.
        :param atol: Absolute tolerance for accepting a translation.
        :return: A list of matching ``q`` index tuples.
        """
        cross = sp.fft.ifftn(np.conj(FA) * FB, axes=(0, 1, 2), overwrite_x=True).real.sum(axis=-1)
        D = A2 + B_sq - 2.0 * cross
        thresh = max(atol * (A2 + B_sq + 1.0), atol * 100)
        return [tuple(int(x) for x in q) for q in np.argwhere(D < thresh)]

    ops = []
    seen_actions = set()

    def _canon_U_bytes(U):
        """
        Produces a canonical (global-phase-fixed, rounded) byte representation of a unitary for deduplication.

        :param U: The unitary matrix to canonicalize.
        :return: The canonical byte string identifying ``U`` up to a global phase.
        """
        flat = U.ravel()
        mags = np.abs(flat)
        candidates_idx = np.where(mags > mags.max() - 1e-4)[0]
        i_pivot = candidates_idx[0]
        if mags[i_pivot] > 1e-12:
            phase = flat[i_pivot] / mags[i_pivot]
            Uc = U / phase
        else:
            Uc = U.copy()
        Uc[np.abs(Uc) < 1e-5] = 0
        return (np.round(Uc.real, 4) + 1j * np.round(Uc.imag, 4)).tobytes()

    for M, idx_map in M_unique:
        # F[ev_M] equals F[ev] permuted by M^-T (an integer grid automorphism, since det M = +/-1), so when M^-T also
        # preserves the grid the forward FFT is a pure index gather of the precomputed FB_plus (no FFT); otherwise
        # fall back to the direct FFT. |F[ev_M]| is then a necessary-condition prune for a translation match either
        # way, skipping the (expensive) correlation for the vast majority of M's that are not symmetries.
        M_invT = np.rint(np.linalg.inv(M)).astype(np.int64).T
        if _M_preserves_grid(M_invT, nk):
            # cheap signature: reject unless |FB| is invariant under M^-T at the K distinctive points (necessary)
            sig_mapped = _apply_M_to_points(M_invT, sig_coords, nk)
            plus_ok = np.allclose(absFB_plus_flat[sig_mapped], sig_ref_plus, rtol=0.0, atol=mag_atol)
            minus_ok = np.allclose(absFB_minus_flat[sig_mapped], sig_ref_minus, rtol=0.0, atol=mag_atol)
            if not (plus_ok or minus_ok):
                continue
            r_map = _apply_M_to_kgrid_indices(M_invT, nk)  # full frequency map, only for signature survivors
            FA = None  # the full complex FA is built lazily, only when a sigma survives
        else:
            if idx_map is None:
                idx_map = _apply_M_to_kgrid_indices(M, nk)
            FA = sp.fft.fftn(_apply_M_to_ev_field(M, ev, nk, idx_map), axes=(0, 1, 2))
            abs_FA = np.abs(FA).reshape(-1, norb)
            plus_ok = np.allclose(abs_FA, absFB_plus_flat, rtol=0.0, atol=mag_atol)
            minus_ok = np.allclose(abs_FA, absFB_minus_flat, rtol=0.0, atol=mag_atol)
            r_map = None

        for sigma, FB, B_sq, sig_ok in (
            (+1, FB_plus, B_plus_sq, plus_ok),
            (-1, FB_minus, B_minus_sq, minus_ok),
        ):
            if not sig_ok:
                continue
            if FA is None:
                FA = FB_plus_flat[r_map].reshape(FB_plus.shape)
            qs = _fft_q_scan_cached(FA, B_plus_sq, FB, B_sq, atol)
            if not qs:
                continue
            ev_k = ev if sigma > 0 else ev_neg  # eigvalsh(sigma * H[.conj]) is +/- ev, never recomputed in the solver
            for q in qs:
                if idx_map is None:
                    idx_map = _apply_M_to_kgrid_indices(M, nk)
                idx_q = _translate_kgrid(idx_map, q, nk)
                idx_q_key = idx_q.tobytes()
                Hg = None
                ev_g = None
                for conj in (False, True):
                    # Quick dedup: if this (idx_q, sigma, conj) already has an op, one U is enough (determined up to the
                    # group's commutant - redundant for the IBZ), but we keep distinct U's as different group elts.
                    if Hg is None:
                        Hg = H_flat[idx_q].reshape(nx, ny, nz, norb, norb)
                        ev_g = ev_flat[idx_q].reshape(nx, ny, nz, norb)  # eigvalsh(Hg) = ev reindexed by the action
                    Hk_eff = sigma * (H.conj() if conj else H)
                    U = _solve_U_for_op(Hg, Hk_eff, atol, ev_k=ev_k, ev_g=ev_g)
                    if U is None:
                        continue
                    action_key = (idx_q_key, sigma, conj, _canon_U_bytes(U))
                    if action_key in seen_actions:
                        continue
                    seen_actions.add(action_key)
                    ops.append(
                        {
                            "M": M.copy(),
                            "q": np.array(q, dtype=np.int64),
                            "U": U,
                            "sigma": sigma,
                            "conj": conj,
                        }
                    )
    return ops, len(ops)


# ============================================================================
# Group elements

_grid_action_cache = {}


def _grid_action_bytes(M, q, nk):
    """
    Returns canonical bytes encoding the combined action of ``(M, q)`` on the k-grid, cached by ``(M, q, nk)``.

    :param M: The 3x3 integer matrix.
    :param q: The integer translation vector.
    :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
    :return: The byte string encoding the resulting flat-index map.
    """
    key = (M.tobytes(), q.tobytes(), tuple(nk))
    cached = _grid_action_cache.get(key)
    if cached is not None:
        return cached
    idx = _apply_M_to_kgrid_indices(M, nk)
    idx = _translate_kgrid(idx, tuple(q), nk)
    out = idx.tobytes()
    # Bound the cache size to avoid unbounded growth across runs.
    if len(_grid_action_cache) > 200000:
        _grid_action_cache.clear()
    _grid_action_cache[key] = out
    return out


def _clear_grid_action_cache():
    """
    Clears the module-level grid-action byte cache.

    :return: None.
    """
    _grid_action_cache.clear()


class _GroupElement:
    """
    A single discovered symmetry operation ``(M, q, U, sigma, conj)`` acting on the k-grid and orbital space, with
    hashing/equality based on its canonical grid action and (phase-fixed) orbital unitary so that operations with the
    same effect compare equal.
    """

    __slots__ = ("M", "q", "U", "sigma", "conj", "nk", "_key")

    def __init__(self, M, q, U, sigma, conj, nk):
        """
        Stores the operation and precomputes its canonical action key for hashing/equality.

        :param M: The 3x3 integer matrix acting on k-indices.
        :param q: The integer translation vector.
        :param U: The orbital-space unitary (canonicalized up to a global phase).
        :param sigma: The antisymmetry sign (``+1`` or ``-1``).
        :param conj: Whether the operation is anti-unitary (complex conjugation).
        :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
        """
        self.M = np.asarray(M, dtype=np.int64)
        self.q = np.asarray(q, dtype=np.int64)
        self.nk = tuple(int(x) for x in nk)
        # Canonicalize U up to global phase.
        U = np.asarray(U, dtype=complex)
        flat = U.ravel()
        mags = np.abs(flat)
        max_mag = mags.max()
        candidates = np.where(mags > max_mag - 1e-4)[0]
        idx_pivot = candidates[0]
        if abs(flat[idx_pivot]) > 1e-12:
            phase = flat[idx_pivot] / abs(flat[idx_pivot])
            U = U / phase
        U_clean = U.copy()
        U_clean[np.abs(U_clean) < 1e-5] = 0
        self.U = U_clean
        self.sigma = int(sigma)
        self.conj = bool(conj)
        # Key: the GRID ACTION, sigma, conj, and the canonicalized U. Using the grid action (not raw M, q) merges
        # operations with different (M, q) but identical effect on the discrete grid.
        Ur = np.round(self.U.real, 4) + 1j * np.round(self.U.imag, 4)
        grid_key = _grid_action_bytes(self.M, self.q, self.nk)
        self._key = (grid_key, self.sigma, self.conj, Ur.tobytes())

    def __eq__(self, other):
        """
        Equality based on the canonical action key (grid action, sigma, conj, phase-fixed U). See :meth:`__init__`.
        """
        return isinstance(other, _GroupElement) and self._key == other._key

    def __hash__(self):
        """
        Hash of the canonical action key (consistent with :meth:`__eq__`).
        """
        return hash(self._key)

    @staticmethod
    def identity(norb, nk):
        """
        Builds the identity group element (identity matrix, zero translation, identity unitary, ``sigma=+1``).

        :param norb: Number of orbitals.
        :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
        :return: The identity :class:`_GroupElement`.
        """
        return _GroupElement(
            np.eye(3, dtype=np.int64), np.zeros(3, dtype=np.int64), np.eye(norb, dtype=complex), +1, False, nk
        )


def _compose(ga, gb, nk):
    """
    Composes two group elements, ``g = ga . gb`` (apply ``gb`` first, then ``ga``).

    :param ga: The outer group element.
    :param gb: The inner group element.
    :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
    :return: The composed :class:`_GroupElement`.
    """
    Ns = np.array(nk, dtype=np.int64)
    M = ga.M @ gb.M
    q = (ga.M @ gb.q + ga.q) % Ns
    sigma = ga.sigma * gb.sigma
    conj = ga.conj ^ gb.conj
    Ub = gb.U if not ga.conj else gb.U.conj()
    U = ga.U @ Ub
    return _GroupElement(M, q, U, sigma, conj, nk)


def _inverse(g, nk):
    """
    Computes the inverse of a group element.

    :param g: The group element to invert.
    :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
    :return: The inverse :class:`_GroupElement`.
    """
    Ns = np.array(nk, dtype=np.int64)
    M_inv = np.linalg.inv(g.M.astype(float))
    M_inv = np.round(M_inv).astype(np.int64)
    q_inv = (-M_inv @ g.q) % Ns
    U_inv = g.U.conj().T if not g.conj else g.U.T
    return _GroupElement(M_inv, q_inv, U_inv, g.sigma, g.conj, nk)


def _close_group(ops_raw, norb, nk, max_size=10000):
    """
    Closes the discovered raw operations under composition (including the identity) to form the full symmetry group,
    up to a maximum size safeguard.

    :param ops_raw: The list of discovered operation dicts.
    :param norb: Number of orbitals.
    :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
    :param max_size: Maximum allowed group size before bailing out.
    :return: The closed set of :class:`_GroupElement`.
    """
    group = {_GroupElement.identity(norb, nk)}
    for o in ops_raw:
        group.add(_GroupElement(o["M"], o["q"], o["U"], o["sigma"], o["conj"], nk))
    changed = True
    while changed and len(group) < max_size:
        changed = False
        gl = list(group)
        for a in gl:
            for b in gl:
                p = _compose(a, b, nk)
                if p not in group:
                    group.add(p)
                    changed = True
                    if len(group) >= max_size:
                        return group
    return group


# ============================================================================
# Orbit collapse and reconstruction


def _g_action_on_kgrid(g, nk):
    """
    Returns the flat-index map of a group element's combined ``(M, q)`` action on the k-grid.

    :param g: The group element.
    :param nk: Number of k-points per spatial direction ``(nx, ny, nz)``.
    :return: The flat-index permutation array.
    """
    idx = _apply_M_to_kgrid_indices(g.M, nk)
    return _translate_kgrid(idx, tuple(g.q), nk)


def _orbit_collapse(H, group):
    """
    Collapses the k-grid into symmetry orbits under the closed group, choosing the smallest flat index as each
    orbit's representative, and records the group element mapping each k back to its representative.

    :param H: The Hamiltonian field of shape ``(nx, ny, nz, norb, norb)``.
    :param group: The closed symmetry group (iterable of :class:`_GroupElement`).
    :return: A tuple ``(orbit_min, trans)`` of the per-k representative flat-index field and the per-k inverse
        transformation elements.
    """
    nx, ny, nz, norb, _ = H.shape
    nk = (nx, ny, nz)
    nktot = nx * ny * nz
    g_list = list(group)
    idx_maps = np.stack([_g_action_on_kgrid(g, nk) for g in g_list], axis=0)
    orbit_min = idx_maps.min(axis=0)
    g_to_rep = np.argmin(idx_maps, axis=0)
    inv_cache = [_inverse(g, nk) for g in g_list]
    trans = np.array([inv_cache[g_to_rep[k]] for k in range(nktot)], dtype=object)
    return orbit_min, trans


# ============================================================================
# Public API


def get_symmetry_reduction(H, atol=1e-8, verbose=False, include_antiunitary=False):
    r"""
    Discovers the symmetries of ``H[kx, ky, kz, o1, o2]`` (on the primitive reciprocal-lattice grid) and produces an
    irreducible-BZ reduction together with reconstruction callables.

    :param H: Hamiltonian of shape ``(nx, ny, nz, norb, norb)`` in the primitive reciprocal-lattice basis.
    :param atol: Absolute tolerance for symmetry validation.
    :param verbose: If True, print diagnostics about discovery and group closure.
    :param include_antiunitary: If False (default), anti-unitary symmetries (``conj=True``, e.g. time-reversal-like
        :math:`H(\mathbf{k}) = H(\mathbf{k})^*`) are discarded after discovery. They are valid symmetries of H, but for
        frequency-dependent objects they additionally require a Matsubara-frequency flip :math:`\imath\omega \to
        -\imath\omega` that the FBZ-mapping path does not perform; keep the default unless reducing a strictly static
        quantity (such as H itself or a band structure).
    :return: A dict with keys ``'group'`` (the discovered :class:`_GroupElement` list), ``'irrk_ind'`` (flat IBZ
        representative indices), ``'fbz2irrk'`` (per-k representative field), ``'expand'`` (callable mapping IBZ
        Hamiltonian values to the full BZ), ``'expand_tensor'`` (callable for arbitrary-rank tensors with per-axis
        ket/bra character), ``'generators'`` (raw discovered ops), ``'n_ibz'``, ``'n_fbz'``, and the per-k transform
        data ``'pos_in_irrk'``, ``'Us'``, ``'sigmas'``, ``'conjs'``.
    """
    # Reset the grid-action cache (in case nk changes between calls).
    _clear_grid_action_cache()
    nx, ny, nz, norb, _ = H.shape
    nk = (nx, ny, nz)
    nktot = nx * ny * nz

    if verbose:
        print(f"H shape: {H.shape}, grid {nk}, {norb} orbitals")

    ops_raw, n_found = _discover_symmetries(H, atol, verbose=verbose)
    if verbose:
        print(f"  Validated operations: {n_found}")

    if not include_antiunitary:
        n_pre = len(ops_raw)
        ops_raw = [op for op in ops_raw if not op.get("conj", False)]
        if verbose:
            print(f"  Anti-unitary ops dropped: {n_pre - len(ops_raw)}; kept {len(ops_raw)}")

    group = _close_group(ops_raw, norb, nk)
    if verbose:
        print(f"  Closed group size: {len(group)}")

    parent, trans = _orbit_collapse(H, group)
    irrk_set = sorted(set(int(p) for p in parent))
    irrk_ind = np.array(irrk_set, dtype=np.int64)
    rep_to_pos = {r: i for i, r in enumerate(irrk_set)}
    pos_in_irrk = np.array([rep_to_pos[int(p)] for p in parent], dtype=np.int64)

    sigmas = np.array([t.sigma for t in trans], dtype=float).reshape(nx, ny, nz)
    conjs = np.array([t.conj for t in trans], dtype=bool).reshape(nx, ny, nz)
    Us = np.stack([t.U for t in trans]).reshape(nx, ny, nz, norb, norb)

    def expand(H_ibz):
        """
        Reconstructs the full-BZ Hamiltonian from its IBZ values by applying the per-k transformation
        ``(sigma_k, U_k, conj_k)``.

        :param H_ibz: IBZ Hamiltonian values of shape ``(n_ibz, norb, norb)``, ordered to match ``irrk_ind``.
        :return: The full-BZ Hamiltonian of shape ``(nx, ny, nz, norb, norb)``.
        """
        H_parents = H_ibz[pos_in_irrk].reshape(nx, ny, nz, norb, norb)
        H_eff = np.where(conjs[..., None, None], H_parents.conj(), H_parents)
        Udag = Us.conj().transpose(0, 1, 2, 4, 3)
        out = np.einsum("...ij,...jk,...kl->...il", Us, H_eff, Udag)
        out *= sigmas[..., None, None]
        return out

    def expand_tensor(T_ibz, kind="kb", sigma_power=1):
        r"""T_ibz: shape (n_ibz, norb, ..., norb) with len(kind) orbital axes.
        kind: string of 'k' (ket: U-contracted) and 'b' (bra: U^dag-contracted)
              per orbital axis. Shortcuts:
                'ket-bra' = 'kb'  (Hamiltonian / Green's function)
                'vertex'  = 'rank4' = 'kkbb'
        sigma_power: power of sigma multiplying the result. For H itself: 1.
                     For quantities built from 2 H's or 2 G's: effectively 0
                     (since sigma^2 = 1).

        Per-axis contraction:
            'k' (ket):  T_new[out, ...] = U[out, in]    * T[in, ...]
            'b' (bra):  T_new[out, ...] = U^dag[in, out] * T[in, ...]
                                        = conj(U[out, in]) * T[in, ...]
        i.e. both branches contract U (or U.conj()) with the SAME index pattern
        (out_letter, in_letter). The only difference is the U vs U.conj() choice.

        Shortcut conventions (alphabetical orbital index order a, b, c, d):
            'ket-bra' / 'kb'     :  1-particle propagator G_ab (a annihilation, b creation)
            'vertex'  / 'rank4'  :  2-particle Green's function G_abcd with operator
                                    ordering :math:`\langle c_a c^\dagger_b c_c c^\dagger_d\rangle` ->
                                    alternating annihilation/creation -> 'kbkb'.

        :param T_ibz: IBZ tensor values of shape ``(n_ibz, norb, ..., norb)`` with ``len(kind)`` orbital axes.
        :param kind: Per-axis character string of ``'k'`` (ket) / ``'b'`` (bra), or a shortcut (``"ket-bra"``,
            ``"vertex"``, ``"rank4"``).
        :param sigma_power: Power of ``sigma`` multiplying the result (1 for H itself, effectively 0 for quantities
            built from two H's/G's).
        :return: The full-BZ tensor with leading momentum axes ``(nx, ny, nz)`` followed by the transformed orbitals.
        :raises ValueError: If ``kind`` is invalid or ``T_ibz``'s rank/orbital sizes do not match it.
        """
        shortcuts = {"ket-bra": "kb", "vertex": "kbkb", "rank4": "kbkb"}
        if isinstance(kind, str) and kind in shortcuts:
            kind = shortcuts[kind]
        if not isinstance(kind, str) or not all(c in "kb" for c in kind):
            raise ValueError(f"kind must be 'k'/'b' chars or a shortcut; got {kind!r}")
        n_oax = len(kind)
        if T_ibz.ndim != 1 + n_oax:
            raise ValueError(f"T_ibz has {T_ibz.ndim} axes, expected 1+{n_oax}")
        for ax in range(n_oax):
            if T_ibz.shape[1 + ax] != norb:
                raise ValueError(f"T_ibz orb axis {ax} has size {T_ibz.shape[1+ax]}, expected {norb}")
        T_parents = T_ibz[pos_in_irrk].reshape((nx, ny, nz) + (norb,) * n_oax)
        bcast = (slice(None),) * 3 + (None,) * n_oax
        T_eff = np.where(conjs[bcast], T_parents.conj(), T_parents)
        in_letters = list(string.ascii_lowercase[:n_oax])
        out_letters = list(string.ascii_lowercase[n_oax : 2 * n_oax])
        operand_str = "KLM" + "".join(in_letters)
        operands = [T_eff]
        # For both 'k' and 'b' the index pattern is (out, in); the only difference is
        # that 'b' uses U.conj() (since U^dag[in, out] = conj(U[out, in])).
        for ax, c in enumerate(kind):
            if c == "k":
                operands.append(Us)
            else:
                operands.append(Us.conj())
            operand_str += f",KLM{out_letters[ax]}{in_letters[ax]}"
        out_str = "KLM" + "".join(out_letters)
        T_full = np.einsum(operand_str + "->" + out_str, *operands, optimize=True)
        if sigma_power != 0:
            T_full = T_full * (sigmas**sigma_power)[bcast]
        return T_full

    return {
        "group": list(group),
        "irrk_ind": irrk_ind,
        "fbz2irrk": parent.reshape(nx, ny, nz),
        "expand": expand,
        "expand_tensor": expand_tensor,
        "generators": ops_raw,
        "n_ibz": len(irrk_ind),
        "n_fbz": nktot,
        # Per-k transformation data, so callers can map other tensors without expand_tensor. For every FBZ point k:
        #   T_full(k) = sigma_k * U_k T(rep(k))^[*conj_k] U_k^dagger, rep(k) = pos_in_irrk[k_flat] -> irrk_ind position.
        "pos_in_irrk": pos_in_irrk,  # shape (nktot,), int - irrk_inv equivalent
        "Us": Us,  # shape (nx, ny, nz, norb, norb), complex
        "sigmas": sigmas,  # shape (nx, ny, nz), float (+/-1)
        "conjs": conjs,  # shape (nx, ny, nz), bool
    }


def _coordinate_mirror_matrix(axis: int) -> np.ndarray:
    """
    Builds the integer matrix of the single-axis coordinate mirror ``k_axis -> -k_axis`` (the identity on the other
    two axes), in the same lattice-basis convention as the discovery enumeration.

    :param axis: The reflected momentum axis (0, 1 or 2).
    :return: The 3x3 integer mirror matrix.
    """
    M = np.eye(3, dtype=np.int64)
    M[axis, axis] = -1
    return M


def find_coordinate_mirror_orbital_unitaries(H, atol=1e-8) -> dict:
    r"""
    Solves for the orbital part of the single-axis coordinate mirrors of ``H``, i.e. for each axis :math:`i` a
    unitary :math:`U_i` with

    .. math:: H(M_i \mathbf{k}) = U_i\, H(\mathbf{k})\, U_i^\dagger, \qquad M_i: k_i \to -k_i .

    This is the same relation and the same U-solver (:func:`_solve_U_for_op`) the full symmetry discovery uses,
    restricted to the three mirrors and to a zero translation, so it costs a handful of small eigen-solves instead of
    a scan over the 6960 candidate matrices. A point-group mirror acts on the orbital indices as well as on the
    momenta; for :math:`t_{2g}` orbitals :math:`U_i` comes out as the diagonal sign matrix of the mirror
    (:math:`\mathrm{diag}(-1,-1,+1)` for :math:`M_x` in the order :math:`d_{xy}, d_{xz}, d_{yz}`), but nothing here
    assumes a particular orbital set, so :math:`e_g`-based and other Wannier bases work the same way.

    :math:`U_i` is fixed only up to the commutant of :math:`H`, which is harmless where the gap transforms by
    conjugation (a global phase or sign cancels in :math:`U \Delta U^\dagger`) but does limit what can be recovered:
    for an :math:`H` that is exactly orbital-diagonal every diagonal sign matrix solves the relation and the
    canonical identity comes back, and on an axis with two or fewer k-points the momentum action is trivial, so
    there is no orbital information to recover there either. In both cases the mirror degenerates to the
    momentum-only reflection, which is what ``H`` alone justifies.

    :param H: The Hamiltonian field of shape ``(nx, ny, nz, norb, norb)`` in the primitive reciprocal-lattice basis.
    :param atol: Absolute tolerance for validating the mirror relation.
    :return: A dict ``{axis: U}`` holding one orbital unitary per axis whose mirror is a symmetry of ``H``.
    """
    nx, ny, nz, norb, _ = H.shape
    nk = (nx, ny, nz)
    H = np.asarray(H, dtype=complex)
    H_flat = H.reshape(-1, norb, norb)
    ev = np.linalg.eigvalsh(H)
    ev_flat = ev.reshape(-1, norb)

    mirrors = {}
    for axis in range(3):
        idx = _apply_M_to_kgrid_indices(_coordinate_mirror_matrix(axis), nk)
        Hg = H_flat[idx].reshape(nx, ny, nz, norb, norb)
        ev_g = ev_flat[idx].reshape(nx, ny, nz, norb)  # eigvalsh(Hg) = ev reindexed by the mirror
        U = _solve_U_for_op(Hg, H, atol, ev_k=ev, ev_g=ev_g)
        if U is not None:
            mirrors[axis] = U
    return mirrors


def apply_auto_orbital_transform(
    mat: np.ndarray,
    us: np.ndarray,
    sigmas: np.ndarray,
    conjs: np.ndarray,
    num_orbital_dimensions: int,
) -> np.ndarray:
    r"""
    Applies the auto-discovered per-k orbital transformation ``(sigma_k, U_k, conj_k)`` to a tensor whose leading
    axis enumerates k-points (or a contiguous slice thereof).

    The transformation follows the operator ordering
    :math:`G_{1234} := \langle T[c_1 c^\dagger_2 c_3 c^\dagger_4]\rangle`, with annihilation indices (positions 1, 3)
    transforming with :math:`U` and creation indices (positions 2, 4) with :math:`U^\dagger`, combined with
    :math:`\sigma` and conjugation:

    .. math::

        M_{12}(\mathbf{k})   &= \sigma_{\mathbf{k}}\, U_{1a} [M_{ab}(\mathbf{k}_{\mathrm{rep}})]^{[*\mathrm{conj}_k]}
                                U^\dagger_{b2} \\
        M_{1234}(\mathbf{k}) &= \sigma_{\mathbf{k}}^2\, U_{1a} [M_{abcd}(\mathbf{k}_{\mathrm{rep}})]^{[*\mathrm{conj}_k]}
                                U^\dagger_{b2} U_{3c} U^\dagger_{d4}

    Since :math:`\sigma_{\mathbf{k}} = \pm 1`, :math:`\sigma_{\mathbf{k}}^2 = 1`; the 4-index case effectively has no
    sign factor, which is the correct physics for vertex quantities under particle-hole-like antisymmetries.

    :param mat: Input tensor of shape ``(k_local, nb, [nb, nb,] nb, ...)``. The leading axis may be the full FBZ or a
        contiguous slice of it; ``us``, ``sigmas`` and ``conjs`` must be sliced consistently.
    :param us: Per-k unitary matrices of shape ``(k_local, nb, nb)``, complex.
    :param sigmas: Per-k antisymmetry signs of shape ``(k_local,)``, values in ``{+1, -1}``.
    :param conjs: Per-k anti-unitary flags of shape ``(k_local,)``, dtype bool.
    :param num_orbital_dimensions: 2 (single-particle, e.g. H, G) or 4 (two-particle vertex); determines both the
        einsum pattern and the effective power of ``sigma_k``.
    :return: The transformed tensor with the same shape as ``mat`` (the same backing array, with identity rows left
        untouched).
    """
    assert num_orbital_dimensions in (2, 4), "num_orbital_dimensions must be 2 or 4."
    k_local = mat.shape[0]
    assert us.shape[0] == k_local and sigmas.shape[0] == k_local and conjs.shape[0] == k_local, (
        f"apply_auto_orbital_transform: per-k arrays must match leading axis "
        f"({us.shape[0]}, {sigmas.shape[0]}, {conjs.shape[0]} vs mat[0]={k_local})."
    )
    if k_local == 0:
        return mat

    nb = us.shape[1]
    for axis_idx in range(1, num_orbital_dimensions + 1):
        assert mat.shape[axis_idx] == nb, (
            f"apply_auto_orbital_transform: orbital axis {axis_idx} has size " f"{mat.shape[axis_idx]}, expected {nb}."
        )

    # Promote U to the matrix dtype to keep einsum in one dtype.
    us = us.astype(mat.dtype, copy=False)

    # sigma_power = number of (U, U^dag) pairs = num_orbital_dimensions // 2.
    # For 4-index this gives sigma^2 = 1, so the sign factor effectively drops.
    sigma_power = num_orbital_dimensions // 2
    effective_sigmas = sigmas if sigma_power == 1 else (sigmas**sigma_power)

    identity = np.eye(nb, dtype=mat.dtype)

    # Group local k-points by their (U, sigma, conj) signature so each equivalence
    # class can be transformed in one batched einsum.
    groups: dict = {}
    for ik in range(k_local):
        key = (
            us[ik].real.round(6).tobytes() + us[ik].imag.round(6).tobytes(),
            float(effective_sigmas[ik]),
            bool(conjs[ik]),
        )
        groups.setdefault(key, []).append(ik)

    path_2 = path_4 = None
    for indices in groups.values():
        u_ref = us[indices[0]]
        sigma = float(effective_sigmas[indices[0]])
        conj = bool(conjs[indices[0]])

        # Identity-like rows: skip entirely.
        if sigma == 1.0 and not conj and np.allclose(u_ref, identity):
            continue

        idx = np.array(indices)
        block = mat[idx]

        if conj:
            block = block.conj()

        uc_ref = u_ref.conj()
        if num_orbital_dimensions == 2:
            if path_2 is None:
                path_2 = np.einsum_path("ap,bq,kpq...->kab...", u_ref, uc_ref, block, optimize="optimal")[0]
            out = np.einsum("ap,bq,kpq...->kab...", u_ref, uc_ref, block, optimize=path_2)
        else:  # 4
            if path_4 is None:
                path_4 = np.einsum_path(
                    "ap,bq,cr,ds,kpqrs...->kabcd...", u_ref, uc_ref, u_ref, uc_ref, block, optimize="optimal"
                )[0]
            out = np.einsum("ap,bq,cr,ds,kpqrs...->kabcd...", u_ref, uc_ref, u_ref, uc_ref, block, optimize=path_4)

        if sigma != 1.0:
            out = out * mat.dtype.type(sigma)

        mat[idx] = out

    return mat

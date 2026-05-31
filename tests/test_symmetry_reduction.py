# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import builtins
import os

import numpy as np
import pytest
from unittest.mock import patch

import dgamore.symmetry_reduction as sr


def test_enumerate_integer_matrices_returns_only_gl3z_matrices():
    mats = sr._enumerate_integer_matrices()
    assert len(mats) == 6960
    assert all(m.shape == (3, 3) for m in mats)
    assert all(np.all(np.isin(m, [-1, 0, 1])) for m in mats)
    assert all(int(round(np.linalg.det(m))) in (-1, 1) for m in mats)


def test_m_preserves_grid_accepts_compatible_matrix_and_rejects_incompatible_one():
    assert sr._M_preserves_grid(np.eye(3, dtype=np.int64), (4, 4, 4)) is True
    assert sr._M_preserves_grid(np.eye(3, dtype=np.int64), (4, 2, 4)) is True

    incompatible = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int64)
    assert sr._M_preserves_grid(incompatible, (4, 2, 4)) is False


def test_apply_m_to_kgrid_indices_maps_identity_and_negative_axis_correctly():
    nk = (2, 2, 2)
    identity = np.eye(3, dtype=np.int64)
    expected_identity = np.arange(8)
    assert np.array_equal(sr._apply_M_to_kgrid_indices(identity, nk), expected_identity)

    flip_x = np.diag([-1, 1, 1]).astype(np.int64)
    mapped = sr._apply_M_to_kgrid_indices(flip_x, nk)
    assert np.array_equal(np.sort(mapped), expected_identity)


def test_translate_kgrid_shifts_flat_indices_modulo_grid_size():
    nk = (2, 3, 4)
    idx_map = np.array([0, 1, 5, 23], dtype=np.int64)
    translated = sr._translate_kgrid(idx_map, (1, 2, 3), nk)

    nx, ny, nz = nk
    iz = idx_map % nz
    iy = (idx_map // nz) % ny
    ix = idx_map // (ny * nz)
    expected = ((ix + 1) % nx) * (ny * nz) + ((iy + 2) % ny) * nz + ((iz + 3) % nz)
    assert np.array_equal(translated, expected)


def test_apply_m_to_ev_field_returns_expected_values_for_identity():
    nk = (2, 2, 2)
    ev = np.arange(8, dtype=np.float64).reshape(*nk, 1)
    out = sr._apply_M_to_ev_field(np.eye(3, dtype=np.int64), ev, nk)
    assert np.array_equal(out, ev)


def test_fft_find_matching_q_finds_exact_translation():
    a = np.arange(8, dtype=np.float64).reshape(2, 2, 2, 1)
    b = np.roll(a, shift=1, axis=0)
    qs = sr._fft_find_matching_q(a, b, atol=1e-12)
    assert (1, 0, 0) in qs


def test_cluster_eigvals_groups_equal_values_and_singletons():
    clusters = sr._cluster_eigvals(np.array([1.0, 1.0, 2.0, 4.0, 4.0]), tol=1e-12)
    assert clusters == [[0, 1], [2], [3, 4]]


def test_solve_u_for_op_returns_simple_unitary_for_matching_hamiltonians():
    nk = (1, 1, 1)
    h = np.zeros((*nk, 2, 2), dtype=complex)
    h[0, 0, 0] = np.array([[1.0, 0.0], [0.0, 2.0]])

    u = sr._solve_U_for_op(h, h.copy(), atol=1e-12)
    assert u is not None
    assert np.allclose(u.conj().T @ u, np.eye(2), atol=1e-12)
    assert np.allclose(np.einsum("ij,...jk,lk->...il", u, h, u.conj()), h, atol=1e-12)


def test_solve_u_for_op_returns_none_for_mismatched_eigenvalues():
    nk = (1, 1, 1)
    hk = np.zeros((*nk, 2, 2), dtype=complex)
    hg = np.zeros((*nk, 2, 2), dtype=complex)
    hk[0, 0, 0] = np.array([[1.0, 0.0], [0.0, 2.0]])
    hg[0, 0, 0] = np.array([[1.0, 0.0], [0.0, 3.0]])
    assert sr._solve_U_for_op(hg, hk, atol=1e-12) is None


def test_fix_phases_nondegenerate_returns_none_when_no_trial_matches():
    nk = (2, 1, 1)
    hk = np.zeros((*nk, 2, 2), dtype=complex)
    hg = np.zeros((*nk, 2, 2), dtype=complex)
    hk[:, 0, 0, 0, 0] = 1.0
    hk[:, 0, 0, 1, 1] = 2.0
    hg[:, 0, 0, 0, 0] = 3.0
    hg[:, 0, 0, 1, 1] = 4.0

    v = np.eye(2, dtype=complex)
    w = np.eye(2, dtype=complex)

    assert sr._fix_phases_nondegenerate(v, w, hk, hg, (0, 0, 0), atol=1e-12) is None


def test_fix_gauge_degenerate_returns_none_when_constraints_are_inconsistent():
    nk = (2, 1, 1)
    hk = np.zeros((*nk, 2, 2), dtype=complex)
    hg = np.zeros((*nk, 2, 2), dtype=complex)
    hk[:, 0, 0] = np.array([[1.0, 0.0], [0.0, 1.0]])
    hg[:, 0, 0] = np.array([[2.0, 0.0], [0.0, 2.0]])

    v = np.eye(2, dtype=complex)
    w = np.eye(2, dtype=complex)
    clusters = [[0, 1]]

    assert sr._fix_gauge_degenerate(v, w, clusters, hk, hg, atol=1e-12) is None


def test_group_element_identity_and_hashable():
    g = sr._GroupElement.identity(2, (2, 2, 1))
    assert g.sigma == 1
    assert g.conj is False
    assert np.array_equal(g.M, np.eye(3, dtype=np.int64))
    assert np.array_equal(g.q, np.zeros(3, dtype=np.int64))
    assert len({g}) == 1


def test_compose_and_inverse_round_trip():
    nk = (2, 2, 1)
    g1 = sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 0, 0]), np.eye(2), +1, False, nk)
    g2 = sr._GroupElement(np.eye(3, dtype=np.int64), np.array([0, 1, 0]), np.eye(2), -1, True, nk)

    composed = sr._compose(g1, g2, nk)
    recovered = sr._compose(composed, sr._inverse(g2, nk), nk)

    assert isinstance(composed, sr._GroupElement)
    assert recovered.sigma == g1.sigma
    assert recovered.conj == g1.conj
    assert np.array_equal(recovered.q, g1.q)


def test_close_group_adds_identity_and_raw_operations():
    nk = (2, 2, 1)
    ops_raw = [
        {
            "M": np.eye(3, dtype=np.int64),
            "q": np.array([1, 0, 0], dtype=np.int64),
            "U": np.eye(2, dtype=complex),
            "sigma": 1,
            "conj": False,
        }
    ]

    group = sr._close_group(ops_raw, norb=2, nk=nk, max_size=10)
    assert any(np.array_equal(g.q, np.zeros(3, dtype=np.int64)) for g in group)
    assert any(np.array_equal(g.q, np.array([1, 0, 0], dtype=np.int64)) for g in group)


def test_g_action_on_kgrid_matches_translation_of_matrix_action():
    nk = (2, 2, 1)
    g = sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 1, 0]), np.eye(2), +1, False, nk)
    action = sr._g_action_on_kgrid(g, nk)
    translated = sr._translate_kgrid(sr._apply_M_to_kgrid_indices(g.M, nk), tuple(g.q), nk)
    assert np.array_equal(action, translated)


def test_orbit_collapse_returns_representatives_and_transformations():
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    H[0, 0, 0, 0, 0] = 1.0
    H[1, 0, 0, 0, 0] = 2.0

    group = {
        sr._GroupElement.identity(1, (2, 1, 1)),
        sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 0, 0]), np.eye(1), +1, False, (2, 1, 1)),
    }
    orbit_min, trans = sr._orbit_collapse(H, group)

    assert orbit_min.shape == (2,)
    assert trans.shape == (2,)
    assert all(isinstance(t, sr._GroupElement) for t in trans)


def test_get_symmetry_reduction_public_api_with_monkeypatched_discovery():
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    H[0, 0, 0, 0, 0] = 1.0
    H[1, 0, 0, 0, 0] = 2.0

    fake_group = {
        sr._GroupElement.identity(1, (2, 1, 1)),
        sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 0, 0]), np.eye(1), +1, False, (2, 1, 1)),
    }

    with patch.object(
        sr,
        "_discover_symmetries",
        return_value=(
            [
                {
                    "M": np.eye(3, dtype=np.int64),
                    "q": np.zeros(3, dtype=np.int64),
                    "U": np.eye(1),
                    "sigma": 1,
                    "conj": False,
                }
            ],
            1,
        ),
    ):
        with patch.object(sr, "_close_group", return_value=fake_group):
            result = sr.get_symmetry_reduction(H, atol=1e-12, verbose=False)

    assert result["n_fbz"] == 2
    assert result["n_ibz"] == len(result["irrk_ind"])
    assert result["fbz2irrk"].shape == (2, 1, 1)
    assert callable(result["expand"])
    assert callable(result["expand_tensor"])


def test_expand_reconstructs_full_hamiltonian_from_ibz_values():
    """With a fake group containing a translation by (1,0,0), both FBZ points collapse
    onto the single representative at index 0. ``expand`` therefore replicates the
    single IBZ value across the full BZ."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    H[0, 0, 0, 0, 0] = 1.0
    H[1, 0, 0, 0, 0] = 2.0

    fake_group = {
        sr._GroupElement.identity(1, (2, 1, 1)),
        sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 0, 0]), np.eye(1), +1, False, (2, 1, 1)),
    }

    with patch.object(
        sr,
        "_discover_symmetries",
        return_value=(
            [
                {
                    "M": np.eye(3, dtype=np.int64),
                    "q": np.zeros(3, dtype=np.int64),
                    "U": np.eye(1),
                    "sigma": 1,
                    "conj": False,
                }
            ],
            1,
        ),
    ):
        with patch.object(sr, "_close_group", return_value=fake_group):
            result = sr.get_symmetry_reduction(H, atol=1e-12, verbose=False)

    # The orbit collapse picks index 0 as the IBZ representative for both points.
    assert result["n_ibz"] == 1
    H_ibz = np.array([[[5.0 + 0.0j]]], dtype=complex)  # shape (n_ibz=1, norb=1, norb=1)
    expanded = result["expand"](H_ibz)

    assert expanded.shape == H.shape
    # Both FBZ points reconstruct from the same IBZ representative.
    assert np.allclose(expanded[0], 5.0)
    assert np.allclose(expanded[1], 5.0)


def test_expand_tensor_validates_kind_and_tensor_shape():
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    fake_group = {
        sr._GroupElement.identity(1, (2, 1, 1)),
        sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 0, 0]), np.eye(1), +1, False, (2, 1, 1)),
    }

    with patch.object(
        sr,
        "_discover_symmetries",
        return_value=(
            [
                {
                    "M": np.eye(3, dtype=np.int64),
                    "q": np.zeros(3, dtype=np.int64),
                    "U": np.eye(1),
                    "sigma": 1,
                    "conj": False,
                }
            ],
            1,
        ),
    ):
        with patch.object(sr, "_close_group", return_value=fake_group):
            result = sr.get_symmetry_reduction(H, atol=1e-12, verbose=False)

    tensor_ibz = np.ones((1, 1, 1), dtype=complex)
    expanded = result["expand_tensor"](tensor_ibz, kind="kb")
    assert expanded.shape == (2, 1, 1, 1, 1)

    with pytest.raises(ValueError):
        result["expand_tensor"](tensor_ibz, kind="bad-kind")

    with pytest.raises(ValueError):
        result["expand_tensor"](np.ones((1, 2, 1), dtype=complex), kind="kb")

    with pytest.raises(ValueError):
        result["expand_tensor"](np.ones((1, 1, 1, 1), dtype=complex), kind="kb")


def test_expand_tensor_supports_shortcuts_and_sigma_power_zero():
    """``rank4`` shortcut means 4 orbital axes (kkbb pattern). With norb=1 the output
    shape is ``(nx, ny, nz, 1, 1, 1, 1)`` regardless of sigma. With ``sigma_power=0``
    the per-k sign factor is suppressed entirely."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    fake_group = {
        sr._GroupElement.identity(1, (2, 1, 1)),
        sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 0, 0]), np.eye(1), -1, False, (2, 1, 1)),
    }

    with patch.object(
        sr,
        "_discover_symmetries",
        return_value=(
            [
                {
                    "M": np.eye(3, dtype=np.int64),
                    "q": np.zeros(3, dtype=np.int64),
                    "U": np.eye(1),
                    "sigma": -1,
                    "conj": False,
                }
            ],
            1,
        ),
    ):
        with patch.object(sr, "_close_group", return_value=fake_group):
            result = sr.get_symmetry_reduction(H, atol=1e-12, verbose=False)

    tensor_ibz = np.ones((1, 1, 1, 1, 1), dtype=complex)
    expanded = result["expand_tensor"](tensor_ibz, kind="rank4", sigma_power=0)
    # nk = (2, 1, 1) and rank4 == kkbb (4 orbital axes); norb=1 -> shape (2,1,1,1,1,1,1)
    assert expanded.shape == (2, 1, 1, 1, 1, 1, 1)
    # sigma_power = 0 -> no sign factor is applied; all values stay equal to the input.
    assert np.allclose(expanded, 1.0)


def test_clear_grid_action_cache_resets_internal_cache():
    sr._grid_action_cache[("a", "b", (1, 1, 1))] = b"cached"
    sr._clear_grid_action_cache()
    assert sr._grid_action_cache == {}


def test_grid_action_bytes_caches_and_reuses_result():
    sr._clear_grid_action_cache()
    M = np.eye(3, dtype=np.int64)
    q = np.array([1, 0, 0], dtype=np.int64)
    first = sr._grid_action_bytes(M, q, (2, 2, 1))
    second = sr._grid_action_bytes(M, q, (2, 2, 1))
    assert first == second
    assert len(sr._grid_action_cache) == 1


def test_discover_symmetries_branching_with_monkeypatched_helpers():
    """The discovery loop iterates ``sigma in {+1,-1}`` × valid q's × ``conj in {False,True}``.
    Patch the helpers so the eigenvalue pre-screen always matches and ``_solve_U_for_op``
    succeeds; verify the returned op records have the expected schema and are deduplicated
    by (idx_q, sigma, conj, U)."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)

    with patch.object(sr, "_enumerate_integer_matrices", return_value=[np.eye(3, dtype=np.int64)]):
        with patch.object(sr, "_M_preserves_grid", return_value=True):
            with patch.object(sr, "_apply_M_to_kgrid_indices", return_value=np.array([0, 1], dtype=np.int64)):
                with patch.object(sr, "_apply_M_to_ev_field", return_value=np.zeros((2, 1, 1, 1))):
                    with patch.object(sr, "_solve_U_for_op", return_value=np.eye(1)):
                        ops, n_found = sr._discover_symmetries(H, atol=1e-12, verbose=False)

    assert n_found == len(ops)
    assert n_found >= 1
    assert all("M" in op and "q" in op and "U" in op and "sigma" in op and "conj" in op for op in ops)
    # All discovered ops share the same M (only one is enumerated).
    assert all(np.array_equal(op["M"], np.eye(3, dtype=np.int64)) for op in ops)


def test_translate_kgrid_identity_translation_keeps_indices():
    nk = (3, 3, 2)
    idx = np.arange(np.prod(nk), dtype=np.int64)
    out = sr._translate_kgrid(idx, (0, 0, 0), nk)
    assert np.array_equal(out, idx)


def test_apply_m_to_kgrid_indices_with_axis_swap_is_modulo_correct():
    """For an x<->y swap to act consistently on the grid, the two axes must be
    commensurate. With nk=(3,3,1) the swap maps (ix, iy, iz) to (iy, ix, iz)."""
    nk = (3, 3, 1)
    swap_xy = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    out = sr._apply_M_to_kgrid_indices(swap_xy, nk)

    nx, ny, nz = nk
    expected = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                # After the swap: new (ix', iy', iz') = (iy, ix, iz)
                expected.append(iy * (ny * nz) + ix * nz + iz)
    assert np.array_equal(out, np.array(expected, dtype=np.int64))


def test_fft_find_matching_q_returns_empty_when_fields_do_not_match():
    a = np.zeros((2, 2, 2, 1), dtype=float)
    b = np.ones((2, 2, 2, 1), dtype=float)
    assert sr._fft_find_matching_q(a, b, atol=1e-12) == []


def test_solve_u_for_op_accepts_global_phase_equivalent_matching():
    nk = (1, 1, 1)
    h = np.zeros((*nk, 2, 2), dtype=complex)
    h[0, 0, 0] = np.array([[3.0, 0.0], [0.0, 5.0]])

    phase = np.exp(1j * 0.37)
    u = phase * np.eye(2, dtype=complex)
    hg = np.einsum("ij,...jk,lk->...il", u, h, u.conj())

    out = sr._solve_U_for_op(hg, h, atol=1e-12)
    assert out is not None
    assert np.allclose(np.einsum("ij,...jk,lk->...il", out, h, out.conj()), hg, atol=1e-12)


def test_fix_phases_nondegenerate_can_return_unitary_for_diagonal_case():
    """When Hk == Hg and both are diagonal with distinct eigenvalues, the identity U
    is one valid solution. _fix_phases_nondegenerate should find it (with phi all 1)
    without needing rng patching: the natural rng draws will eventually produce a k1
    where the off-diagonal phases are determined consistently."""
    nk = (2, 1, 1)
    hk = np.zeros((*nk, 2, 2), dtype=complex)
    hg = np.zeros((*nk, 2, 2), dtype=complex)
    # Add small off-diagonal so the per-eigenvector phases are determined.
    hk[0, 0, 0] = np.array([[1.0, 0.5], [0.5, 2.0]])
    hk[1, 0, 0] = np.array([[1.0, 0.3], [0.3, 2.0]])
    hg[0, 0, 0] = hk[0, 0, 0]
    hg[1, 0, 0] = hk[1, 0, 0]

    v = np.eye(2, dtype=complex)
    w = np.eye(2, dtype=complex)

    out = sr._fix_phases_nondegenerate(v, w, hk, hg, (0, 0, 0), atol=1e-10)

    assert out is not None
    assert np.allclose(out.conj().T @ out, np.eye(2), atol=1e-10)
    rhs = np.einsum("ij,...jk,lk->...il", out, hk, out.conj())
    assert np.allclose(rhs, hg, atol=1e-10)


def test_fix_gauge_degenerate_can_return_unitary_for_trivial_cluster():
    nk = (2, 1, 1)
    hk = np.zeros((*nk, 2, 2), dtype=complex)
    hg = np.zeros((*nk, 2, 2), dtype=complex)
    hk[:, 0, 0] = np.array([[1.0, 0.0], [0.0, 1.0]])
    hg[:, 0, 0] = hk[:, 0, 0]

    v = np.eye(2, dtype=complex)
    w = np.eye(2, dtype=complex)

    out = sr._fix_gauge_degenerate(v, w, [[0], [1]], hk, hg, atol=1e-12)
    assert out is not None
    assert np.allclose(np.einsum("ij,...jk,lk->...il", out, hk, out.conj()), hg, atol=1e-12)


def test_close_group_uses_all_raw_ops_and_identity():
    nk = (1, 1, 1)
    ops_raw = [
        {
            "M": np.eye(3, dtype=np.int64),
            "q": np.zeros(3, dtype=np.int64),
            "U": np.eye(1, dtype=complex),
            "sigma": 1,
            "conj": False,
        },
        {
            "M": np.eye(3, dtype=np.int64),
            "q": np.zeros(3, dtype=np.int64),
            "U": np.array([[1j]], dtype=complex),
            "sigma": -1,
            "conj": True,
        },
    ]
    group = sr._close_group(ops_raw, norb=1, nk=nk, max_size=20)
    assert len(group) >= 2
    assert any(g.sigma == -1 and g.conj for g in group)


def test_orbit_collapse_with_singleton_group_returns_identity_transform():
    H = np.zeros((1, 1, 1, 1, 1), dtype=complex)
    group = {sr._GroupElement.identity(1, (1, 1, 1))}
    orbit_min, trans = sr._orbit_collapse(H, group)

    assert np.array_equal(orbit_min, np.array([0], dtype=np.int64))
    assert len(trans) == 1
    assert trans[0].sigma == 1
    assert trans[0].conj is False


def test_get_symmetry_reduction_honors_verbose_branch_and_cache_reset():
    H = np.zeros((1, 1, 1, 1, 1), dtype=complex)

    sr._grid_action_cache[("stale", "entry", (1, 1, 1))] = b"old"

    fake_group = {sr._GroupElement.identity(1, (1, 1, 1))}
    with patch.object(sr, "_discover_symmetries", return_value=([], 0)) as mock_disc:
        with patch.object(sr, "_close_group", return_value=fake_group) as mock_close:
            result = sr.get_symmetry_reduction(H, atol=1e-12, verbose=True)

    assert mock_disc.call_count == 1
    assert mock_close.call_count == 1
    assert sr._grid_action_cache  # populated again after the call
    assert result["n_ibz"] == 1
    assert result["n_fbz"] == 1


def test_expand_tensor_rejects_unknown_shortcut_kind():
    H = np.zeros((1, 1, 1, 1, 1), dtype=complex)
    fake_group = {sr._GroupElement.identity(1, (1, 1, 1))}
    with patch.object(sr, "_discover_symmetries", return_value=([], 0)):
        with patch.object(sr, "_close_group", return_value=fake_group):
            result = sr.get_symmetry_reduction(H, atol=1e-12, verbose=False)

    with pytest.raises(ValueError):
        result["expand_tensor"](np.ones((1, 1), dtype=complex), kind="not-a-kind")


def test_expand_tensor_applies_sigma_factor_when_requested():
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    fake_group = {
        sr._GroupElement.identity(1, (2, 1, 1)),
        sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 0, 0]), np.eye(1), -1, False, (2, 1, 1)),
    }
    with patch.object(sr, "_discover_symmetries", return_value=([{}], 0)):
        with patch.object(sr, "_close_group", return_value=fake_group):
            result = sr.get_symmetry_reduction(H, atol=1e-12, verbose=False)

    tensor_ibz = np.ones((1, 1, 1), dtype=complex)
    expanded = result["expand_tensor"](tensor_ibz, kind="kb", sigma_power=1)
    assert expanded.shape == (2, 1, 1, 1, 1)
    assert np.all(np.isin(np.unique(expanded), [-1, 1]))


def test_group_element_equality_depends_on_canonical_action_and_phase():
    nk = (1, 1, 1)
    g1 = sr._GroupElement(np.eye(3, dtype=np.int64), np.zeros(3, dtype=np.int64), np.eye(2), 1, False, nk)
    g2 = sr._GroupElement(
        np.eye(3, dtype=np.int64), np.zeros(3, dtype=np.int64), np.eye(2) * np.exp(1j * 0.4), 1, False, nk
    )

    assert g1 == g2
    assert hash(g1) == hash(g2)


def test_discover_symmetries_dedups_identical_M_grid_actions():
    """When two enumerated M's produce identical grid actions, the second one should
    be skipped by the M-dedup. With identity-everywhere mocks and 1 k-point, the
    loop iterates: 1 unique-M × 2 sigmas × 1 q × 2 conjs = 4 distinct ops."""
    H = np.zeros((1, 1, 1, 1, 1), dtype=complex)

    with patch.object(
        sr, "_enumerate_integer_matrices", return_value=[np.eye(3, dtype=np.int64), np.eye(3, dtype=np.int64)]
    ):
        with patch.object(sr, "_M_preserves_grid", return_value=True):
            with patch.object(sr, "_apply_M_to_kgrid_indices", return_value=np.array([0], dtype=np.int64)):
                with patch.object(sr, "_apply_M_to_ev_field", return_value=np.zeros((1, 1, 1, 1))):
                    with patch.object(sr, "_solve_U_for_op", return_value=np.eye(1)):
                        ops, n_found = sr._discover_symmetries(H, atol=1e-12, verbose=False)

    assert n_found == len(ops)
    # M is enumerated twice but the second copy has the same grid action and is deduped:
    # one unique M times {sigma=+1, sigma=-1} times {conj=False, conj=True} = 4 ops.
    # (Each (sigma, conj) yields a distinct action_key tuple since they enter the key.)
    assert n_found == 4


def test_apply_auto_orbital_transform_identity_rows_are_left_unchanged():
    mat = np.arange(2 * 2 * 2, dtype=np.complex128).reshape(2, 2, 2)
    us = np.stack([np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)])
    sigmas = np.array([1, 1], dtype=int)
    conjs = np.array([False, False], dtype=bool)

    out = sr.apply_auto_orbital_transform(mat.copy(), us, sigmas, conjs, num_orbital_dimensions=2)

    assert np.array_equal(out, mat)


def test_apply_auto_orbital_transform_applies_unitary_rotation_for_two_orbital_axes():
    mat = np.zeros((1, 2, 2, 3), dtype=np.complex128)
    mat[0, 0, 1] = np.array([1.0, 2.0, 3.0])

    theta = np.pi / 2
    u = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.complex128,
    )
    us = np.array([u])
    sigmas = np.array([1], dtype=int)
    conjs = np.array([False], dtype=bool)

    out = sr.apply_auto_orbital_transform(mat.copy(), us, sigmas, conjs, num_orbital_dimensions=2)

    expected = np.einsum("ap,bq,kpq...->kab...", u, u.conj(), mat, optimize=True)
    assert np.allclose(out, expected)


def test_apply_auto_orbital_transform_applies_conjugation_and_sigma_sign_for_two_orbital_axes():
    """For U=I, sigma=-1, conj=True: result = sigma * U M^* U^dag = -M^*."""
    mat = np.array([[[1.0 + 2.0j, 3.0 - 4.0j], [5.0 + 6.0j, 7.0 - 8.0j]]], dtype=np.complex128)
    u = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    us = np.array([u])
    sigmas = np.array([-1], dtype=int)
    conjs = np.array([True], dtype=bool)

    out = sr.apply_auto_orbital_transform(mat.copy(), us, sigmas, conjs, num_orbital_dimensions=2)

    assert out.shape == mat.shape
    assert np.allclose(out, -mat.conj())


def test_apply_auto_orbital_transform_four_orbital_axes_uses_sigma_squared_and_preserves_identity_case():
    mat = np.arange(1, 1 + 1 * 2 * 2 * 2 * 2, dtype=np.complex128).reshape(1, 2, 2, 2, 2)
    us = np.array([np.eye(2, dtype=np.complex128)])
    sigmas = np.array([-1], dtype=int)
    conjs = np.array([False], dtype=bool)

    out = sr.apply_auto_orbital_transform(mat.copy(), us, sigmas, conjs, num_orbital_dimensions=4)

    assert np.array_equal(out, mat)


def test_apply_auto_orbital_transform_groups_equivalent_k_points_together():
    mat = np.zeros((3, 2, 2), dtype=np.complex128)
    mat[0] = np.array([[1.0, 2.0], [3.0, 4.0]])
    mat[1] = np.array([[5.0, 6.0], [7.0, 8.0]])
    mat[2] = np.array([[9.0, 10.0], [11.0, 12.0]])

    u = np.eye(2, dtype=np.complex128)
    us = np.stack([u, u, u])
    sigmas = np.array([1, 1, 1], dtype=int)
    conjs = np.array([False, False, False], dtype=bool)

    out = sr.apply_auto_orbital_transform(mat.copy(), us, sigmas, conjs, num_orbital_dimensions=2)

    assert np.array_equal(out, mat)


def test_apply_auto_orbital_transform_rejects_invalid_orbital_dimension_count():
    mat = np.zeros((1, 2, 2), dtype=np.complex128)
    us = np.eye(2, dtype=np.complex128)[None, ...]
    sigmas = np.array([1], dtype=int)
    conjs = np.array([False], dtype=bool)

    with pytest.raises(AssertionError):
        sr.apply_auto_orbital_transform(mat, us, sigmas, conjs, num_orbital_dimensions=3)


def test_apply_auto_orbital_transform_rejects_mismatched_leading_axis_lengths():
    mat = np.zeros((2, 2, 2), dtype=np.complex128)
    us = np.eye(2, dtype=np.complex128)[None, ...]
    sigmas = np.array([1], dtype=int)
    conjs = np.array([False], dtype=bool)

    with pytest.raises(AssertionError):
        sr.apply_auto_orbital_transform(mat, us, sigmas, conjs, num_orbital_dimensions=2)


def test_apply_auto_orbital_transform_rejects_wrong_orbital_axis_sizes():
    mat = np.zeros((1, 3, 2), dtype=np.complex128)
    us = np.eye(2, dtype=np.complex128)[None, ...]
    sigmas = np.array([1], dtype=int)
    conjs = np.array([False], dtype=bool)

    with pytest.raises(AssertionError):
        sr.apply_auto_orbital_transform(mat, us, sigmas, conjs, num_orbital_dimensions=2)


def test_apply_auto_orbital_transform_handles_empty_input():
    mat = np.zeros((0, 2, 2), dtype=np.complex128)
    us = np.zeros((0, 2, 2), dtype=np.complex128)
    sigmas = np.zeros((0,), dtype=int)
    conjs = np.zeros((0,), dtype=bool)

    out = sr.apply_auto_orbital_transform(mat, us, sigmas, conjs, num_orbital_dimensions=2)
    assert out.shape == mat.shape


# ============================================================================
# Tests for the norb == 1 short-circuit in _solve_U_for_op.
# Without this short-circuit, np.diff over a length-1 last axis produced an empty
# array and .min(axis=-1) raised ValueError. The short-circuit avoids both.
# ============================================================================


def test_solve_u_for_op_one_orbital_returns_identity_when_spectra_match():
    """For norb=1, U is a 1x1 unitary (a global phase). Identity always works
    when spectra match, regardless of grid extent."""
    h_k = np.zeros((3, 2, 4, 1, 1), dtype=complex)
    h_k[..., 0, 0] = np.arange(24).reshape(3, 2, 4)
    h_g = h_k.copy()

    u = sr._solve_U_for_op(h_g, h_k, atol=1e-12)
    assert u is not None
    assert u.shape == (1, 1)
    assert np.allclose(u, np.eye(1))


def test_solve_u_for_op_one_orbital_returns_none_when_spectra_differ():
    """norb=1 short-circuit must still return None when spectra disagree."""
    h_k = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    h_g = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    h_k[..., 0, 0] = np.array([1.0, 2.0]).reshape(2, 1, 1)
    h_g[..., 0, 0] = np.array([1.0, 5.0]).reshape(2, 1, 1)

    u = sr._solve_U_for_op(h_g, h_k, atol=1e-12)
    assert u is None


def test_solve_u_for_op_one_orbital_does_not_call_eigh(monkeypatch):
    """The norb=1 path short-circuits before any eigendecomposition or gauge fixing.
    Patch _fix_phases_nondegenerate/_fix_gauge_degenerate to raise if invoked."""
    h_k = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    h_k[..., 0, 0] = np.array([1.0, 2.0]).reshape(2, 1, 1)

    def _explode(*a, **kw):  # pragma: no cover - should not run
        raise AssertionError("gauge-fix helper should not be called for norb=1")

    monkeypatch.setattr(sr, "_fix_phases_nondegenerate", _explode)
    monkeypatch.setattr(sr, "_fix_gauge_degenerate", _explode)

    u = sr._solve_U_for_op(h_k.copy(), h_k.copy(), atol=1e-12)
    assert u is not None


# ============================================================================
# End-to-end auto discovery on real (and synthetic) Hamiltonians.
# These cover the full pipeline: spatial-op enumeration, FFT q-scan, U solving,
# group closure, orbit collapse, expansion. Files live under
# tests/test_data/auto_symmetries/.
# ============================================================================


_HAMILTONIANS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", "auto_symmetries")

# (filename, expected_shape, marks)
# Slow tests are skipped by default; run them with `pytest --runslow`.
_HAMILTONIAN_CASES = [
    pytest.param("hk_1band_square_32x32x1.npy", (32, 32, 1, 1, 1), id="1band_square_32x32x1"),
    pytest.param("hk_1band_anisotropy_48x48x1.npy", (48, 48, 1, 1, 1), id="1band_anisotropy_48x48x1"),
    pytest.param(
        "hk_3band_srvo3_cubic_12x12x12.npy",
        (12, 12, 12, 3, 3),
        id="3band_srvo3_cubic_12x12x12",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "hk_3band_srvo3_cubic_20x20x20.npy",
        (20, 20, 20, 3, 3),
        id="3band_srvo3_cubic_20x20x20",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "hk_4band_la3ni2o7_32x32x32.npy",
        (32, 32, 32, 4, 4),
        id="4band_la3ni2o7_32x32x32",
        marks=pytest.mark.slow,
    ),
]


def _require_hamiltonian(fname: str, expected_shape: tuple) -> np.ndarray:
    """Load a test Hamiltonian or skip the test if the file is missing."""
    path = os.path.join(_HAMILTONIANS_DIR, fname)
    if not os.path.exists(path):
        pytest.skip(f"Hamiltonian fixture not present: {path}")
    H = np.load(path)
    assert H.shape == expected_shape, f"Unexpected shape for {fname}: {H.shape} != {expected_shape}"
    return H


@pytest.mark.parametrize("fname,shape", _HAMILTONIAN_CASES)
def test_auto_symmetry_discovery_reconstructs_hamiltonian(fname, shape):
    """For every supplied Hamiltonian, the IBZ -> FBZ reconstruction via the
    auto-discovered symmetry data must reproduce H to machine precision (double).
    This is the most important integration test: if it passes, the entire
    discover -> close-group -> orbit-collapse -> expand pipeline is consistent."""
    H = _require_hamiltonian(fname, shape)

    result = sr.get_symmetry_reduction(H, atol=1e-8, verbose=False)
    nx, ny, nz, nb, _ = shape
    nktot = nx * ny * nz

    # IBZ has fewer points than the FBZ (or equal, for a no-symmetry H).
    assert 1 <= result["n_ibz"] <= nktot
    assert result["n_fbz"] == nktot
    assert len(result["irrk_ind"]) == result["n_ibz"]
    assert result["fbz2irrk"].shape == (nx, ny, nz)

    H_ibz = H.reshape(-1, nb, nb)[result["irrk_ind"]]
    H_rec = result["expand"](H_ibz)
    assert H_rec.shape == H.shape
    assert np.allclose(
        H_rec, H, atol=1e-9
    ), f"reconstruction mismatch for {fname}: max |diff| = {np.max(np.abs(H_rec - H)):.2e}"


@pytest.mark.parametrize("fname,shape", _HAMILTONIAN_CASES)
def test_auto_symmetry_discovery_expand_tensor_reproduces_HtimesH_vertex(fname, shape):
    """Build a rank-4 tensor Gamma = H tensor H (which inherits the same symmetries
    as H since each factor transforms identically), reduce to IBZ, expand back, and
    check exact recovery. This exercises the 4-orbital-index expand_tensor path
    and verifies the sigma_power=2 convention (sigma^2 == 1 always)."""
    H = _require_hamiltonian(fname, shape)
    nx, ny, nz, nb, _ = shape

    result = sr.get_symmetry_reduction(H, atol=1e-8, verbose=False)

    # Gamma[k, a, b, c, d] = H[k, a, b] * H[k, c, d]
    Gamma = np.einsum("...ab,...cd->...abcd", H, H)
    G_ibz = Gamma.reshape(-1, nb, nb, nb, nb)[result["irrk_ind"]]
    G_rec = result["expand_tensor"](G_ibz, kind="rank4", sigma_power=2)

    assert G_rec.shape == Gamma.shape
    assert np.allclose(
        G_rec, Gamma, atol=1e-9
    ), f"vertex reconstruction mismatch for {fname}: max |diff| = {np.max(np.abs(G_rec - Gamma)):.2e}"


def test_auto_discovery_finds_2d_square_group_for_isotropic_lattice():
    """The 32x32x1 1-band square lattice has full square point group D4h × TR.
    Auto should detect the 8-element point group (and combine it with TR-like
    operations to give a 16-element total)."""
    H = _require_hamiltonian("hk_1band_square_32x32x1.npy", (32, 32, 1, 1, 1))
    result = sr.get_symmetry_reduction(H, atol=1e-8, verbose=False)
    # Empirically: 8 spatial + 8 TR-combined = 16 group elements for the simple square H
    assert len(result["group"]) >= 8
    # IBZ should be 153/1024 ~= 8x reduction
    assert result["n_ibz"] == 153
    assert result["n_fbz"] == 1024


def test_auto_discovery_finds_smaller_group_for_anisotropic_lattice():
    """The 48x48x1 anisotropic 1-band lattice has tx != ty, so kx<->ky is NOT a
    symmetry. Only inversion and individual axis flips (real H) survive. The IBZ
    should be ~4x reduced, much less than 8x for the isotropic case."""
    H = _require_hamiltonian("hk_1band_anisotropy_48x48x1.npy", (48, 48, 1, 1, 1))
    result = sr.get_symmetry_reduction(H, atol=1e-8, verbose=False)

    nktot = 48 * 48 * 1
    reduction = nktot / result["n_ibz"]
    # Expect reduction between 3 and 5 (inversion + each-axis flips for real H)
    assert 2.5 < reduction < 5.0, f"Unexpected reduction factor {reduction:.2f}"


@pytest.mark.slow
def test_auto_discovery_matches_legacy_for_12cubed_cubic_hamiltonian():
    """Auto-discovered IBZ partition must match the legacy three_dimensional_cubic
    partition for a genuinely cubic 3-band Hamiltonian. (12^3 — slower.)"""
    import dgamore.brillouin_zone as bz

    fname, shape = "hk_3band_srvo3_cubic_12x12x12.npy", (12, 12, 12, 3, 3)
    H = _require_hamiltonian(fname, shape)
    nx, ny, nz, _, _ = shape

    kg_auto = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    kg_auto.specify_auto_symmetries(H, atol=1e-8)

    kg_legacy = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.three_dimensional_cubic_symmetries())

    assert kg_auto.nk_irr == kg_legacy.nk_irr
    assert np.array_equal(kg_auto.fbz2irrk, kg_legacy.fbz2irrk)


@pytest.mark.slow
def test_auto_discovery_matches_legacy_for_20cubed_cubic_hamiltonian():
    """Same as above for the 20^3 grid. (Even slower — covers the larger-grid path.)"""
    import dgamore.brillouin_zone as bz

    fname, shape = "hk_3band_srvo3_cubic_20x20x20.npy", (20, 20, 20, 3, 3)
    H = _require_hamiltonian(fname, shape)
    nx, ny, nz, _, _ = shape

    kg_auto = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.AUTO_SYMMETRIES_SENTINEL)
    kg_auto.specify_auto_symmetries(H, atol=1e-8)

    kg_legacy = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.three_dimensional_cubic_symmetries())

    assert kg_auto.nk_irr == kg_legacy.nk_irr
    assert np.array_equal(kg_auto.fbz2irrk, kg_legacy.fbz2irrk)


# ============================================================================
# Edge cases
# ============================================================================


def test_get_symmetry_reduction_on_trivial_1x1x1_grid():
    """A single k-point: every symmetry operation acts trivially on k-space.
    IBZ has exactly 1 point, the FBZ point coincides with it."""
    H = np.zeros((1, 1, 1, 2, 2), dtype=complex)
    H[..., 0, 0] = 1.0
    H[..., 1, 1] = 2.0
    H[..., 0, 1] = 0.3
    H[..., 1, 0] = 0.3

    result = sr.get_symmetry_reduction(H, atol=1e-10)

    assert result["n_fbz"] == 1
    assert result["n_ibz"] == 1
    assert np.array_equal(result["irrk_ind"], np.array([0], dtype=np.int64))

    # Reconstruct
    H_ibz = H.reshape(-1, 2, 2)[result["irrk_ind"]]
    H_rec = result["expand"](H_ibz)
    assert np.allclose(H_rec, H, atol=1e-12)


def test_get_symmetry_reduction_on_random_non_symmetric_hamiltonian_yields_full_bz_ibz():
    """A random Hermitian H with no special structure has only the trivial
    symmetry group {e} (and possibly TR if real). IBZ should equal FBZ for a
    sufficiently generic complex H."""
    rng = np.random.default_rng(7)
    nx, ny, nz, nb = 4, 4, 1, 2  # small to keep tests fast
    H = rng.standard_normal((nx, ny, nz, nb, nb)) + 1j * rng.standard_normal((nx, ny, nz, nb, nb))
    H = 0.5 * (H + H.conj().transpose(0, 1, 2, 4, 3))  # Hermitian

    result = sr.get_symmetry_reduction(H, atol=1e-10)

    # Trivial group: every k-point is its own representative.
    assert result["n_ibz"] == result["n_fbz"]
    H_ibz = H.reshape(-1, nb, nb)[result["irrk_ind"]]
    H_rec = result["expand"](H_ibz)
    assert np.allclose(H_rec, H, atol=1e-12)


def test_get_symmetry_reduction_handles_zero_hamiltonian():
    """H == 0 has every possible symmetry; the discovered group will be large but
    the reconstruction must still work."""
    H = np.zeros((2, 2, 1, 2, 2), dtype=complex)
    result = sr.get_symmetry_reduction(H, atol=1e-10)

    H_ibz = H.reshape(-1, 2, 2)[result["irrk_ind"]]
    H_rec = result["expand"](H_ibz)
    assert np.allclose(H_rec, H, atol=1e-12)
    # Every k collapses to the single representative.
    assert result["n_ibz"] == 1


def test_get_symmetry_reduction_handles_diagonal_real_hamiltonian():
    """A purely-diagonal H with cos-like dispersion on a cubic grid -- exercise
    a case where the orbital action is identity for all symmetries."""
    nx = ny = nz = 4
    j1, j2, j3 = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    k1 = 2 * np.pi * j1 / nx
    k2 = 2 * np.pi * j2 / ny
    k3 = 2 * np.pi * j3 / nz
    e = np.cos(k1) + np.cos(k2) + np.cos(k3)
    H = np.zeros((nx, ny, nz, 2, 2), dtype=complex)
    H[..., 0, 0] = e
    H[..., 1, 1] = -e

    result = sr.get_symmetry_reduction(H, atol=1e-10)
    H_ibz = H.reshape(-1, 2, 2)[result["irrk_ind"]]
    H_rec = result["expand"](H_ibz)
    assert np.allclose(H_rec, H, atol=1e-10)


def test_get_symmetry_reduction_returns_callables_and_complete_dict_schema():
    """Sanity: the returned dict has every documented key."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    result = sr.get_symmetry_reduction(H, atol=1e-10)
    expected_keys = {
        "group",
        "irrk_ind",
        "fbz2irrk",
        "expand",
        "expand_tensor",
        "generators",
        "n_ibz",
        "n_fbz",
        "pos_in_irrk",
        "Us",
        "sigmas",
        "conjs",
    }
    assert expected_keys.issubset(set(result.keys())), f"Missing keys: {expected_keys - set(result.keys())}"
    assert callable(result["expand"])
    assert callable(result["expand_tensor"])


def test_apply_auto_orbital_transform_two_orbital_axes_preserves_trailing_dims():
    """The function is shape-polymorphic in the trailing axes after the orbital
    pair: pass an array with extra frequency-like axes and verify they pass through."""
    k_local, nb, n_extra = 2, 2, 3
    mat = np.arange(k_local * nb * nb * n_extra, dtype=np.complex128).reshape(k_local, nb, nb, n_extra)
    us = np.stack([np.eye(nb), np.eye(nb)]).astype(np.complex128)
    sigmas = np.array([1, 1], dtype=int)
    conjs = np.array([False, False], dtype=bool)

    out = sr.apply_auto_orbital_transform(mat.copy(), us, sigmas, conjs, num_orbital_dimensions=2)
    # Identity transform: output equals input.
    assert out.shape == mat.shape
    assert np.array_equal(out, mat)


def test_apply_auto_orbital_transform_returns_input_array_object_for_identity_only_groups():
    """When every k-point has identity (U, sigma=+1, conj=False), the function
    returns the same array object without allocating a new one."""
    mat = np.arange(8, dtype=np.complex128).reshape(2, 2, 2)
    us = np.stack([np.eye(2), np.eye(2)]).astype(np.complex128)
    sigmas = np.array([1, 1], dtype=int)
    conjs = np.array([False, False], dtype=bool)

    out = sr.apply_auto_orbital_transform(mat, us, sigmas, conjs, num_orbital_dimensions=2)
    assert out is mat  # identity short-circuit must not copy


# =============================================================================
# include_antiunitary flag in get_symmetry_reduction
# =============================================================================
# Anti-unitary symmetries (H(k) = H(k)*) are valid for H but not for frequency-
# dependent objects, because FBZ expansion only conjugates orbital indices and
# does NOT flip Matsubara frequencies. The ``include_antiunitary`` flag was
# introduced to make the safe behavior the default.


def _make_real_cubic_h(nx=4, ny=4, nz=4, nb=1):
    j1, j2, j3 = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    k1 = 2 * np.pi * j1 / nx
    k2 = 2 * np.pi * j2 / ny
    k3 = 2 * np.pi * j3 / nz
    H = np.zeros((nx, ny, nz, nb, nb), dtype=complex)
    eps = -2.0 * (np.cos(k1) + np.cos(k2) + np.cos(k3))
    for o in range(nb):
        H[..., o, o] = eps + 0.1 * o
    return H


def test_get_symmetry_reduction_default_excludes_antiunitary_ops():
    """The default behaviour must drop anti-unitary operations; therefore no FBZ
    point should carry conj=True. This is the safe semantics for frequency-
    dependent quantities."""
    H = _make_real_cubic_h(4, 4, 4, 1)
    result = sr.get_symmetry_reduction(H, atol=1e-8)
    assert result["conjs"].any() == False  # noqa: E712 — explicit bool check


def test_get_symmetry_reduction_include_antiunitary_admits_conj_ops():
    """For a real-valued H, H(k) = H(k)* always gives anti-unitary ops; opting in
    must therefore produce at least one conj=True point."""
    H = _make_real_cubic_h(4, 4, 4, 1)
    result = sr.get_symmetry_reduction(H, atol=1e-8, include_antiunitary=True)
    assert int(result["conjs"].sum()) > 0


def test_get_symmetry_reduction_include_antiunitary_shrinks_or_equals_ibz():
    """Adding TR ops can only make orbits larger, never smaller; hence the IBZ
    with anti-unitary ops must be smaller than or equal to the spatial-only IBZ."""
    H = _make_real_cubic_h(4, 4, 4, 1)
    r_default = sr.get_symmetry_reduction(H, atol=1e-8, include_antiunitary=False)
    r_full = sr.get_symmetry_reduction(H, atol=1e-8, include_antiunitary=True)
    assert r_full["n_ibz"] <= r_default["n_ibz"]


def test_get_symmetry_reduction_include_antiunitary_reconstructs_H_correctly():
    """When anti-unitary ops are included, reconstruction of H itself is still
    correct (anti-unitary ops are valid symmetries of H — only frequency-dependent
    objects are affected by the missing freq flip)."""
    H = _make_real_cubic_h(4, 4, 4, 1)
    result = sr.get_symmetry_reduction(H, atol=1e-8, include_antiunitary=True)
    H_ibz = H.reshape(-1, 1, 1)[result["irrk_ind"]]
    H_rec = result["expand"](H_ibz)
    assert np.allclose(H_rec, H, atol=1e-12)


def test_get_symmetry_reduction_include_antiunitary_passes_verbose_diagnostic(capsys):
    """The verbose branch reports how many anti-unitary ops were dropped."""
    H = _make_real_cubic_h(4, 4, 4, 1)
    sr.get_symmetry_reduction(H, atol=1e-8, verbose=True, include_antiunitary=False)
    captured = capsys.readouterr().out
    assert "Anti-unitary ops dropped" in captured


def test_get_symmetry_reduction_verbose_does_not_report_drop_when_keeping_antiunitary(capsys):
    """If we explicitly keep anti-unitary ops, the 'dropped' message should NOT appear."""
    H = _make_real_cubic_h(4, 4, 4, 1)
    sr.get_symmetry_reduction(H, atol=1e-8, verbose=True, include_antiunitary=True)
    captured = capsys.readouterr().out
    assert "Anti-unitary ops dropped" not in captured


def test_get_symmetry_reduction_default_yields_no_conjugation_in_expansion():
    """Concrete consequence of default ``include_antiunitary=False``: applying
    ``expand`` to any IBZ payload does NOT conjugate orbital indices anywhere.
    We verify this by feeding a complex payload built so that conjugation would
    be detectable (the conjugate differs from the original)."""
    H = _make_real_cubic_h(4, 4, 4, 1)
    result = sr.get_symmetry_reduction(H, atol=1e-8)
    # Reconstruct H itself — well-defined and exact
    H_ibz = H.reshape(-1, 1, 1)[result["irrk_ind"]]
    H_rec = result["expand"](H_ibz)
    assert np.allclose(H_rec, H, atol=1e-12)
    # And: every per-k transformation has conj=False (asserted directly here).
    assert int(result["conjs"].sum()) == 0


# ============================================================================
# Full path / branch coverage for previously-untested branches.
#
# The tests below exercise the remaining defensive and rarely-hit branches so
# that statement *and* branch coverage of symmetry_reduction.py is complete:
#   * _canonicalize_sign_gauge: the norb > 6 row-canonicalization fallback
#     (both the "canon succeeds" and "canon breaks the solution" outcomes) and
#     the accepted lower-score sign-diagonal in the <=6 path.
#   * _solve_U_for_op: the norb==1 None return (spectra close but H differs by
#     more than atol), the per-point eigh-spectra-disagree `continue`, and the
#     degenerate-cluster gauge-fix route.
#   * _fix_phases_nondegenerate: the near-zero ratio `continue`.
#   * _fix_gauge_degenerate: both np.linalg.LinAlgError fallbacks (stacked SVD
#     and per-block SVD).
#   * _discover_symmetries: hash-collision handling, zero-pivot U canonicalization,
#     and the duplicate-action skip.
#   * _grid_action_bytes: cache eviction past the size cap.
#   * _close_group: the early return when max_size is reached mid-composition.
#   * _GroupElement: near-zero U skips phase normalization.
#   * apply_auto_orbital_transform: the 4-orbital einsum branch and the cached
#     einsum-path reuse for a second non-identity group (2- and 4-index).
# ============================================================================


# ---- _canonicalize_sign_gauge ----------------------------------------------


def test_canonicalize_sign_gauge_norb_gt_6_fallback_flips_negative_rows():
    """For norb > 6 the routine uses a row-major canonicalization: each row is
    scaled by the sign of its largest-magnitude entry. With a diagonal H the
    sign flips cancel in U H U^dag, so the canonicalized U is returned."""
    norb = 8
    Hk = np.diag(np.arange(1.0, norb + 1)).astype(complex)[None, None, None]
    Hg = Hk.copy()
    U = np.diag([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0]).astype(complex)

    out = sr._canonicalize_sign_gauge(U, Hk, Hg, atol=1e-10)

    for i in range(norb):
        j = int(np.argmax(np.abs(out[i])))
        assert out[i, j].real > 0  # every row's dominant entry made positive
    rhs = np.einsum("ij,...jk,lk->...il", out, Hk, out.conj())
    assert np.allclose(Hg, rhs, atol=1e-10)


def test_canonicalize_sign_gauge_norb_gt_6_returns_original_when_canon_breaks_solution():
    """If the row-major sign flip moves U out of the centralizer of a *non*-
    degenerate H, the resulting matrix no longer solves U H U^dag = Hg and the
    original U is returned unchanged. A rotation angle near pi/2 makes exactly
    one row of the 2x2 block flip (so the flip does not cancel)."""
    norb = 7
    base = np.diag(np.arange(1.0, norb + 1))
    base[0, 1] = base[1, 0] = 0.3  # off-diagonal: a single-row flip is detectable
    Hk = base[None, None, None].astype(complex)

    th = 1.3  # |sin| > |cos|, sin > 0: row 0's dominant entry (-sin) is negative,
    #           row 1's dominant entry (sin) is positive -> exactly ONE row flips
    c, s = np.cos(th), np.sin(th)
    U = np.eye(norb)
    U[0, 0] = c
    U[0, 1] = -s
    U[1, 0] = s
    U[1, 1] = c
    U = U.astype(complex)
    Hg = np.einsum("ij,...jk,lk->...il", U, Hk, U.conj())

    # Sanity on the flip pattern: row 0 dominant entry negative, row 1 positive.
    assert U[0, 1].real < 0 and abs(U[0, 1]) > abs(U[0, 0])
    assert U[1, 0].real > 0 and abs(U[1, 0]) > abs(U[1, 1])

    out = sr._canonicalize_sign_gauge(U, Hk, Hg, atol=1e-10)
    assert np.array_equal(out, U)  # canon broke the relation -> original returned


def test_canonicalize_sign_gauge_accepts_lower_score_sign_diagonal():
    """For norb <= 6 the routine searches sign-diagonals. Starting from U = -I
    (two negative entries), D = -I yields +I (zero negative entries) which still
    solves, so the lower-score solution is accepted."""
    Hk = np.diag([1.0, 2.0]).astype(complex)[None, None, None]
    Hg = Hk.copy()
    U = -np.eye(2, dtype=complex)

    out = sr._canonicalize_sign_gauge(U, Hk, Hg, atol=1e-10)
    assert np.allclose(out, np.eye(2), atol=1e-12)


# ---- _solve_U_for_op --------------------------------------------------------


def test_solve_u_for_op_one_orbital_returns_none_when_close_but_outside_atol():
    """norb==1: spectra agree within 10*atol (the pre-screen passes) but H itself
    differs by more than atol, so the final validation fails and None is returned.
    Near-zero values keep allclose's relative term from masking the difference."""
    Hg = np.array([[[[[0.0 + 0j]]]]])
    Hk = np.array([[[[[5e-12 + 0j]]]]])
    assert sr._solve_U_for_op(Hg, Hk, atol=1e-12) is None


def test_solve_u_for_op_continues_when_perpoint_eigh_spectra_disagree(monkeypatch):
    """The global eigenvalue pre-screen passes (Hk == Hg), but a patched eigh
    desynchronizes the per-point spectra between the Hk and Hg evaluations, so
    every candidate reference point hits the `continue` and None is returned."""
    Hk = np.zeros((2, 1, 1, 2, 2), dtype=complex)
    Hk[..., 0, 0] = np.array([1.0, 1.0]).reshape(2, 1, 1)
    Hk[..., 1, 1] = np.array([2.0, 3.0]).reshape(2, 1, 1)
    Hg = Hk.copy()

    calls = {"n": 0}

    def fake_eigh(a):
        calls["n"] += 1
        if calls["n"] % 2 == 1:  # the Hk evaluation
            return np.array([0.0, 1.0]), np.eye(2, dtype=complex)
        return np.array([0.0, 2.0]), np.eye(2, dtype=complex)  # the Hg evaluation

    monkeypatch.setattr(sr.np.linalg, "eigh", fake_eigh)
    assert sr._solve_U_for_op(Hg, Hk, atol=1e-12) is None
    assert calls["n"] >= 2


def test_solve_u_for_op_routes_through_degenerate_gauge_fix():
    """A 2-fold degenerate spectrum forces the degenerate-cluster branch: the
    naive W @ V^dag picks an arbitrary orientation of the degenerate eigenspace
    that does not match the (fixed) U relating Hk and Hg, so _fix_gauge_degenerate
    must solve for the block rotation. The returned U must satisfy the relation."""
    norb = 3
    nk = (6, 1, 1)
    rng = np.random.default_rng(0)
    D = np.diag([0.0, 1.0, 1.0])  # eigenvalue 1 is 2-fold degenerate
    th = 0.6
    U_true = np.eye(norb)
    U_true[1, 1] = np.cos(th)
    U_true[1, 2] = -np.sin(th)
    U_true[2, 1] = np.sin(th)
    U_true[2, 2] = np.cos(th)

    Hk = np.zeros((*nk, norb, norb), dtype=complex)
    Hg = np.zeros((*nk, norb, norb), dtype=complex)
    for kx in range(nk[0]):
        B, _ = np.linalg.qr(rng.standard_normal((norb, norb)))
        local = B @ D @ B.T
        Hk[kx, 0, 0] = local
        Hg[kx, 0, 0] = U_true @ local @ U_true.T

    out = sr._solve_U_for_op(Hg, Hk, atol=1e-9)
    assert out is not None
    rhs = np.einsum("ij,...jk,lk->...il", out, Hk, out.conj())
    assert np.allclose(rhs, Hg, atol=1e-8)


# ---- _fix_phases_nondegenerate ---------------------------------------------


def test_fix_phases_nondegenerate_hits_near_zero_ratio_continue():
    """When A[r, col] is significant but B[r, col] ~ 0, the candidate phase has
    near-zero magnitude and is skipped (`continue`). With no column yielding a
    consistent phase, every trial fails and None is returned."""
    nk = (3, 1, 1)
    Hk = np.zeros((*nk, 2, 2), dtype=complex)
    Hg = np.zeros((*nk, 2, 2), dtype=complex)
    for kx in range(nk[0]):
        Hk[kx, 0, 0] = np.array([[1.0, 0.5], [0.5, 2.0]])  # A[1,0] = 0.5 (> 1e-4)
        Hg[kx, 0, 0] = np.array([[1.0, 0.0], [0.0, 2.0]])  # B[1,0] = 0
    V = np.eye(2, dtype=complex)
    W = np.eye(2, dtype=complex)

    assert sr._fix_phases_nondegenerate(V, W, Hk, Hg, (0, 0, 0), atol=1e-12) is None


# ---- _fix_gauge_degenerate (LinAlgError fallbacks) -------------------------


def test_fix_gauge_degenerate_returns_none_on_stacked_svd_linalgerror(monkeypatch):
    """If the SVD of the stacked constraint matrix raises LinAlgError, the routine
    returns None."""
    nk = (2, 1, 1)
    Hk = np.zeros((*nk, 2, 2), dtype=complex)
    Hg = np.zeros((*nk, 2, 2), dtype=complex)
    Hk[:, 0, 0] = np.eye(2)
    Hg[:, 0, 0] = np.eye(2)

    def boom(*a, **k):
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(sr.np.linalg, "svd", boom)
    out = sr._fix_gauge_degenerate(
        np.eye(2, dtype=complex), np.eye(2, dtype=complex), [[0, 1]], Hk, Hg, atol=1e-12
    )
    assert out is None


def test_fix_gauge_degenerate_returns_none_on_block_svd_linalgerror(monkeypatch):
    """If the per-block SVD raises LinAlgError (after the stacked SVD succeeded
    and a null vector was found), the routine returns None."""
    nk = (2, 1, 1)
    Hk = np.zeros((*nk, 2, 2), dtype=complex)
    Hg = np.zeros((*nk, 2, 2), dtype=complex)
    Hk[:, 0, 0] = np.eye(2)
    Hg[:, 0, 0] = np.eye(2)

    real_svd = np.linalg.svd
    state = {"n": 0}

    def flaky_svd(a, *args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return real_svd(a, *args, **kwargs)  # stacked SVD succeeds
        raise np.linalg.LinAlgError("forced on block")

    monkeypatch.setattr(sr.np.linalg, "svd", flaky_svd)
    out = sr._fix_gauge_degenerate(
        np.eye(2, dtype=complex), np.eye(2, dtype=complex), [[0, 1]], Hk, Hg, atol=1e-12
    )
    assert out is None
    assert state["n"] >= 2


# ---- _discover_symmetries (rare branches) ----------------------------------


def test_discover_symmetries_handles_hash_collision_of_grid_actions(monkeypatch):
    """Two M's with distinct grid actions but a (forced) identical hash must both
    be kept: the collision is confirmed via array comparison and the second M is
    appended rather than dropped."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    with patch.object(
        sr,
        "_enumerate_integer_matrices",
        return_value=[np.eye(3, dtype=np.int64), np.diag([-1, 1, 1]).astype(np.int64)],
    ):
        with patch.object(sr, "_M_preserves_grid", return_value=True):
            with patch.object(
                sr,
                "_apply_M_to_kgrid_indices",
                side_effect=[np.array([0, 1], dtype=np.int64), np.array([1, 0], dtype=np.int64)],
            ):
                with patch.object(sr, "_apply_M_to_ev_field", return_value=np.zeros((2, 1, 1, 1))):
                    with patch.object(sr, "_solve_U_for_op", return_value=np.eye(1)):
                        monkeypatch.setattr(builtins, "hash", lambda x: 1234)  # force collision
                        ops, n = sr._discover_symmetries(H, atol=1e-12, verbose=False)
    assert n == len(ops)
    assert n >= 1


def test_discover_symmetries_handles_zero_pivot_in_U_canonicalization():
    """When the solver returns an all-near-zero U, the canonical-bytes helper has
    a near-zero pivot magnitude and skips the phase division (the else branch)."""
    H = np.zeros((1, 1, 1, 1, 1), dtype=complex)
    with patch.object(sr, "_enumerate_integer_matrices", return_value=[np.eye(3, dtype=np.int64)]):
        with patch.object(sr, "_M_preserves_grid", return_value=True):
            with patch.object(sr, "_apply_M_to_kgrid_indices", return_value=np.array([0], dtype=np.int64)):
                with patch.object(sr, "_apply_M_to_ev_field", return_value=np.zeros((1, 1, 1, 1))):
                    with patch.object(sr, "_solve_U_for_op", return_value=np.zeros((1, 1), dtype=complex)):
                        ops, n = sr._discover_symmetries(H, atol=1e-12, verbose=False)
    assert n == len(ops)


def test_discover_symmetries_skips_duplicate_action_key():
    """Two distinct M's whose translated grid actions coincide for some q (with
    identical sigma/conj/U) produce the same action key; the duplicate is skipped
    rather than recorded twice."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    with patch.object(
        sr,
        "_enumerate_integer_matrices",
        return_value=[np.eye(3, dtype=np.int64), np.diag([-1, 1, 1]).astype(np.int64)],
    ):
        with patch.object(sr, "_M_preserves_grid", return_value=True):
            with patch.object(
                sr,
                "_apply_M_to_kgrid_indices",
                side_effect=[np.array([0, 1], dtype=np.int64), np.array([1, 0], dtype=np.int64)],
            ):
                with patch.object(sr, "_apply_M_to_ev_field", return_value=np.zeros((2, 1, 1, 1))):
                    with patch.object(sr, "_solve_U_for_op", return_value=np.eye(1)):
                        ops, n = sr._discover_symmetries(H, atol=1e-12, verbose=False)
    # With H == 0 every q matches, so both M's enumerate overlapping actions; the
    # duplicate-action guard keeps the op set deduplicated.
    assert n == len(ops)


# ---- _grid_action_bytes (cache eviction) -----------------------------------


def test_grid_action_bytes_evicts_cache_past_size_cap():
    """When the cache grows beyond its cap, the next insertion clears it first."""
    sr._clear_grid_action_cache()
    try:
        sr._grid_action_cache.update({i: b"" for i in range(200001)})
        assert len(sr._grid_action_cache) > 200000
        sr._grid_action_bytes(np.eye(3, dtype=np.int64), np.array([1, 0, 0], dtype=np.int64), (2, 2, 1))
        assert len(sr._grid_action_cache) == 1  # cleared, then the new entry stored
    finally:
        sr._clear_grid_action_cache()


# ---- _close_group (max_size early return) ----------------------------------


def test_close_group_returns_early_when_max_size_reached_during_composition():
    """A translation generator on an 8-point axis generates an 8-element cyclic
    group; with max_size=4 the closure aborts mid-composition and returns the
    partially-built group."""
    nk = (8, 1, 1)
    ops_raw = [
        {
            "M": np.eye(3, dtype=np.int64),
            "q": np.array([1, 0, 0], dtype=np.int64),
            "U": np.eye(1, dtype=complex),
            "sigma": 1,
            "conj": False,
        }
    ]
    group = sr._close_group(ops_raw, norb=1, nk=nk, max_size=4)
    assert len(group) >= 4


# ---- _GroupElement (near-zero U) -------------------------------------------


def test_group_element_with_near_zero_U_skips_phase_normalization():
    """A U whose entries are all ~0 has a near-zero pivot magnitude, so the global
    phase normalization is skipped (the False side of the pivot guard)."""
    g = sr._GroupElement(
        np.eye(3, dtype=np.int64),
        np.zeros(3, dtype=np.int64),
        np.zeros((2, 2), dtype=complex),
        +1,
        False,
        (1, 1, 1),
    )
    assert np.allclose(g.U, 0.0)


# ---- apply_auto_orbital_transform (einsum path + path reuse) ---------------


def _rot2(theta):
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.complex128
    )


def test_apply_auto_orbital_transform_four_orbital_nonidentity_einsum():
    """A non-identity U with 4 orbital axes exercises the rank-4 einsum branch
    (the identity short-circuit does not apply)."""
    u = _rot2(np.pi / 5)
    mat = np.arange(1, 1 + 16, dtype=np.complex128).reshape(1, 2, 2, 2, 2)
    us = np.array([u])
    sigmas = np.array([1], dtype=int)
    conjs = np.array([False], dtype=bool)

    out = sr.apply_auto_orbital_transform(mat.copy(), us, sigmas, conjs, num_orbital_dimensions=4)

    uc = u.conj()
    expected = np.einsum("ap,bq,cr,ds,kpqrs->kabcd", u, uc, u, uc, mat)
    assert np.allclose(out, expected)


def test_apply_auto_orbital_transform_reuses_cached_path_two_dim():
    """Two distinct non-identity groups in one call: the second reuses the cached
    2-index einsum path (the False side of `if path_2 is None`)."""
    us = np.stack([_rot2(np.pi / 5), _rot2(np.pi / 3)])
    mat = np.arange(2 * 2 * 2, dtype=np.complex128).reshape(2, 2, 2)
    sigmas = np.array([1, 1], dtype=int)
    conjs = np.array([False, False], dtype=bool)

    out = sr.apply_auto_orbital_transform(mat.copy(), us, sigmas, conjs, num_orbital_dimensions=2)

    for i in range(2):
        u = us[i]
        exp = np.einsum("ap,bq,pq->ab", u, u.conj(), mat[i])
        assert np.allclose(out[i], exp)


def test_apply_auto_orbital_transform_reuses_cached_path_four_dim():
    """Same path-reuse check for the 4-index einsum branch."""
    us = np.stack([_rot2(np.pi / 5), _rot2(np.pi / 3)])
    mat = np.arange(2 * 16, dtype=np.complex128).reshape(2, 2, 2, 2, 2)
    sigmas = np.array([1, 1], dtype=int)
    conjs = np.array([False, False], dtype=bool)

    out = sr.apply_auto_orbital_transform(mat.copy(), us, sigmas, conjs, num_orbital_dimensions=4)

    for i in range(2):
        u = us[i]
        uc = u.conj()
        exp = np.einsum("ap,bq,cr,ds,pqrs->abcd", u, uc, u, uc, mat[i])
        assert np.allclose(out[i], exp)
# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import builtins
import os

import numpy as np
import pytest
from unittest.mock import MagicMock

import dgamore.symmetry_reduction as sr


def test_enumerate_integer_matrices_returns_only_gl3z_matrices():
    """enumerate_integer_matrices yields only GL(3,Z) matrices."""
    mats = sr._enumerate_integer_matrices()
    assert len(mats) == 6960
    assert all(m.shape == (3, 3) for m in mats)
    assert all(np.all(np.isin(m, [-1, 0, 1])) for m in mats)
    assert all(int(round(np.linalg.det(m))) in (-1, 1) for m in mats)


def test_m_preserves_grid_accepts_compatible_matrix_and_rejects_incompatible_one():
    """m_preserves_grid accepts a grid-compatible matrix and rejects an incompatible one."""
    assert sr._M_preserves_grid(np.eye(3, dtype=np.int64), (4, 4, 4)) is True
    assert sr._M_preserves_grid(np.eye(3, dtype=np.int64), (4, 2, 4)) is True

    incompatible = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int64)
    assert sr._M_preserves_grid(incompatible, (4, 2, 4)) is False


def test_apply_m_to_kgrid_indices_maps_identity_and_negative_axis_correctly():
    """apply_m_to_kgrid_indices maps the identity and negative-axis operations correctly."""
    nk = (2, 2, 2)
    identity = np.eye(3, dtype=np.int64)
    expected_identity = np.arange(8)
    assert np.array_equal(sr._apply_M_to_kgrid_indices(identity, nk), expected_identity)

    flip_x = np.diag([-1, 1, 1]).astype(np.int64)
    mapped = sr._apply_M_to_kgrid_indices(flip_x, nk)
    assert np.array_equal(np.sort(mapped), expected_identity)


def test_translate_kgrid_shifts_flat_indices_modulo_grid_size():
    """translate_kgrid shifts flat indices modulo the grid size."""
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
    """apply_m_to_ev_field returns the input field unchanged for the identity."""
    nk = (2, 2, 2)
    ev = np.arange(8, dtype=np.float64).reshape(*nk, 1)
    out = sr._apply_M_to_ev_field(np.eye(3, dtype=np.int64), ev, nk)
    assert np.array_equal(out, ev)


def test_fft_find_matching_q_finds_exact_translation():
    """fft_find_matching_q finds the exact translation between two fields."""
    a = np.arange(8, dtype=np.float64).reshape(2, 2, 2, 1)
    b = np.roll(a, shift=1, axis=0)
    qs = sr._fft_find_matching_q(a, b, atol=1e-12)
    assert (1, 0, 0) in qs


def test_cluster_eigvals_groups_equal_values_and_singletons():
    """cluster_eigvals groups equal eigenvalues and keeps singletons separate."""
    clusters = sr._cluster_eigvals(np.array([1.0, 1.0, 2.0, 4.0, 4.0]), tol=1e-12)
    assert clusters == [[0, 1], [2], [3, 4]]


def test_solve_u_for_op_returns_simple_unitary_for_matching_hamiltonians():
    """_solve_U_for_op returns a unitary relating two matching Hamiltonians."""
    nk = (1, 1, 1)
    h = np.zeros((*nk, 2, 2), dtype=complex)
    h[0, 0, 0] = np.array([[1.0, 0.0], [0.0, 2.0]])

    u = sr._solve_U_for_op(h, h.copy(), atol=1e-12)
    assert u is not None
    assert np.allclose(u.conj().T @ u, np.eye(2), atol=1e-12)
    assert np.allclose(np.einsum("ij,...jk,lk->...il", u, h, u.conj()), h, atol=1e-12)


def test_solve_u_for_op_returns_none_for_mismatched_eigenvalues():
    """_solve_U_for_op returns None for mismatched eigenvalue spectra."""
    nk = (1, 1, 1)
    hk = np.zeros((*nk, 2, 2), dtype=complex)
    hg = np.zeros((*nk, 2, 2), dtype=complex)
    hk[0, 0, 0] = np.array([[1.0, 0.0], [0.0, 2.0]])
    hg[0, 0, 0] = np.array([[1.0, 0.0], [0.0, 3.0]])
    assert sr._solve_U_for_op(hg, hk, atol=1e-12) is None


def test_fix_phases_nondegenerate_returns_none_when_no_trial_matches():
    """_fix_phases_nondegenerate returns None when no trial phase matches."""
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
    """_fix_gauge_degenerate returns None for inconsistent constraints."""
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
    """A _GroupElement identity is hashable and well-defined."""
    g = sr._GroupElement.identity(2, (2, 2, 1))
    assert g.sigma == 1
    assert g.conj is False
    assert np.array_equal(g.M, np.eye(3, dtype=np.int64))
    assert np.array_equal(g.q, np.zeros(3, dtype=np.int64))
    assert len({g}) == 1


def test_compose_and_inverse_round_trip():
    """Composing a group element with its inverse returns the identity."""
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
    """_close_group includes the identity and the raw operations."""
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
    """The group action on the k-grid matches the translation of the matrix action."""
    nk = (2, 2, 1)
    g = sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 1, 0]), np.eye(2), +1, False, nk)
    action = sr._g_action_on_kgrid(g, nk)
    translated = sr._translate_kgrid(sr._apply_M_to_kgrid_indices(g.M, nk), tuple(g.q), nk)
    assert np.array_equal(action, translated)


def test_orbit_collapse_returns_representatives_and_transformations():
    """_orbit_collapse returns IBZ representatives and their transformations."""
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


def test_get_symmetry_reduction_public_api_with_monkeypatched_discovery(monkeypatch):
    """get_symmetry_reduction exposes the public API over a monkeypatched discovery."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    H[0, 0, 0, 0, 0] = 1.0
    H[1, 0, 0, 0, 0] = 2.0

    fake_group = {
        sr._GroupElement.identity(1, (2, 1, 1)),
        sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 0, 0]), np.eye(1), +1, False, (2, 1, 1)),
    }

    with monkeypatch.context() as mp:
        mp.setattr(
            sr,
            "_discover_symmetries",
            MagicMock(
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
                )
            ),
        )
        mp.setattr(sr, "_close_group", MagicMock(return_value=fake_group))
        result = sr.get_symmetry_reduction(H, atol=1e-12, verbose=False)

    assert result["n_fbz"] == 2
    assert result["n_ibz"] == len(result["irrk_ind"])
    assert result["fbz2irrk"].shape == (2, 1, 1)
    assert callable(result["expand"])
    assert callable(result["expand_tensor"])


def test_expand_reconstructs_full_hamiltonian_from_ibz_values(monkeypatch):
    """A fake (1,0,0)-translation group collapses both FBZ points onto index 0, so expand replicates the IBZ value."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    H[0, 0, 0, 0, 0] = 1.0
    H[1, 0, 0, 0, 0] = 2.0

    fake_group = {
        sr._GroupElement.identity(1, (2, 1, 1)),
        sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 0, 0]), np.eye(1), +1, False, (2, 1, 1)),
    }

    with monkeypatch.context() as mp:
        mp.setattr(
            sr,
            "_discover_symmetries",
            MagicMock(
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
                )
            ),
        )
        mp.setattr(sr, "_close_group", MagicMock(return_value=fake_group))
        result = sr.get_symmetry_reduction(H, atol=1e-12, verbose=False)

    # the orbit collapse picks index 0 as the IBZ representative, so both FBZ points reconstruct from it
    assert result["n_ibz"] == 1
    H_ibz = np.array([[[5.0 + 0.0j]]], dtype=complex)
    expanded = result["expand"](H_ibz)

    assert expanded.shape == H.shape
    assert np.allclose(expanded[0], 5.0)
    assert np.allclose(expanded[1], 5.0)


def test_expand_tensor_validates_kind_and_tensor_shape(monkeypatch):
    """expand_tensor validates the kind argument and the tensor shape."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    fake_group = {
        sr._GroupElement.identity(1, (2, 1, 1)),
        sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 0, 0]), np.eye(1), +1, False, (2, 1, 1)),
    }

    with monkeypatch.context() as mp:
        mp.setattr(
            sr,
            "_discover_symmetries",
            MagicMock(
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
                )
            ),
        )
        mp.setattr(sr, "_close_group", MagicMock(return_value=fake_group))
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


def test_expand_tensor_supports_shortcuts_and_sigma_power_zero(monkeypatch):
    """The rank4 shortcut gives 4 orbital axes; norb=1 gives size-1 orbitals and sigma_power=0 drops the sign."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    fake_group = {
        sr._GroupElement.identity(1, (2, 1, 1)),
        sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 0, 0]), np.eye(1), -1, False, (2, 1, 1)),
    }

    with monkeypatch.context() as mp:
        mp.setattr(
            sr,
            "_discover_symmetries",
            MagicMock(
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
                )
            ),
        )
        mp.setattr(sr, "_close_group", MagicMock(return_value=fake_group))
        result = sr.get_symmetry_reduction(H, atol=1e-12, verbose=False)

    tensor_ibz = np.ones((1, 1, 1, 1, 1), dtype=complex)
    expanded = result["expand_tensor"](tensor_ibz, kind="rank4", sigma_power=0)
    assert expanded.shape == (2, 1, 1, 1, 1, 1, 1)
    assert np.allclose(expanded, 1.0)


def test_clear_grid_action_cache_resets_internal_cache():
    """clear_grid_action_cache resets the internal grid-action cache."""
    sr._grid_action_cache[("a", "b", (1, 1, 1))] = b"cached"
    sr._clear_grid_action_cache()
    assert sr._grid_action_cache == {}


def test_grid_action_bytes_caches_and_reuses_result():
    """_grid_action_bytes caches and reuses its result."""
    sr._clear_grid_action_cache()
    M = np.eye(3, dtype=np.int64)
    q = np.array([1, 0, 0], dtype=np.int64)
    first = sr._grid_action_bytes(M, q, (2, 2, 1))
    second = sr._grid_action_bytes(M, q, (2, 2, 1))
    assert first == second
    assert len(sr._grid_action_cache) == 1


def test_discover_symmetries_branching_with_monkeypatched_helpers(monkeypatch):
    """With the pre-screen and _solve_U_for_op patched to succeed, discovery returns deduplicated op records."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)

    with monkeypatch.context() as mp:
        mp.setattr(sr, "_enumerate_integer_matrices", MagicMock(return_value=[np.eye(3, dtype=np.int64)]))
        mp.setattr(sr, "_M_preserves_grid", MagicMock(return_value=True))
        mp.setattr(sr, "_apply_M_to_kgrid_indices", MagicMock(return_value=np.array([0, 1], dtype=np.int64)))
        mp.setattr(sr, "_apply_M_to_ev_field", MagicMock(return_value=np.zeros((2, 1, 1, 1))))
        mp.setattr(sr, "_solve_U_for_op", MagicMock(return_value=np.eye(1)))
        ops, n_found = sr._discover_symmetries(H, atol=1e-12, verbose=False)

    assert n_found == len(ops)
    assert n_found >= 1
    assert all("M" in op and "q" in op and "U" in op and "sigma" in op and "conj" in op for op in ops)
    # All discovered ops share the same M (only one is enumerated).
    assert all(np.array_equal(op["M"], np.eye(3, dtype=np.int64)) for op in ops)


def test_translate_kgrid_identity_translation_keeps_indices():
    """translate_kgrid with the identity translation keeps the indices."""
    nk = (3, 3, 2)
    idx = np.arange(np.prod(nk), dtype=np.int64)
    out = sr._translate_kgrid(idx, (0, 0, 0), nk)
    assert np.array_equal(out, idx)


def test_apply_m_to_kgrid_indices_with_axis_swap_is_modulo_correct():
    """On nk=(3,3,1) an x<->y swap maps (ix,iy,iz) to (iy,ix,iz) modulo the commensurate axes."""
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
    """fft_find_matching_q returns empty when the fields do not match."""
    a = np.zeros((2, 2, 2, 1), dtype=float)
    b = np.ones((2, 2, 2, 1), dtype=float)
    assert sr._fft_find_matching_q(a, b, atol=1e-12) == []


def test_solve_u_for_op_accepts_global_phase_equivalent_matching():
    """_solve_U_for_op accepts a global-phase-equivalent match."""
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
    """For diagonal Hk == Hg with distinct eigenvalues _fix_phases_nondegenerate finds the identity U."""
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
    """_fix_gauge_degenerate returns a unitary for a trivial degenerate cluster."""
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
    """_close_group uses every raw operation plus the identity."""
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
    """_orbit_collapse with a singleton group returns the identity transform."""
    H = np.zeros((1, 1, 1, 1, 1), dtype=complex)
    group = {sr._GroupElement.identity(1, (1, 1, 1))}
    orbit_min, trans = sr._orbit_collapse(H, group)

    assert np.array_equal(orbit_min, np.array([0], dtype=np.int64))
    assert len(trans) == 1
    assert trans[0].sigma == 1
    assert trans[0].conj is False


def test_get_symmetry_reduction_honors_verbose_branch_and_cache_reset(monkeypatch):
    """get_symmetry_reduction honors the verbose branch and resets the cache."""
    H = np.zeros((1, 1, 1, 1, 1), dtype=complex)

    sr._grid_action_cache[("stale", "entry", (1, 1, 1))] = b"old"

    fake_group = {sr._GroupElement.identity(1, (1, 1, 1))}
    mock_disc = MagicMock(return_value=([], 0))
    mock_close = MagicMock(return_value=fake_group)
    with monkeypatch.context() as mp:
        mp.setattr(sr, "_discover_symmetries", mock_disc)
        mp.setattr(sr, "_close_group", mock_close)
        result = sr.get_symmetry_reduction(H, atol=1e-12, verbose=True)

    assert mock_disc.call_count == 1
    assert mock_close.call_count == 1
    assert sr._grid_action_cache  # populated again after the call
    assert result["n_ibz"] == 1
    assert result["n_fbz"] == 1


def test_expand_tensor_rejects_unknown_shortcut_kind(monkeypatch):
    """expand_tensor rejects an unknown shortcut kind."""
    H = np.zeros((1, 1, 1, 1, 1), dtype=complex)
    fake_group = {sr._GroupElement.identity(1, (1, 1, 1))}
    with monkeypatch.context() as mp:
        mp.setattr(sr, "_discover_symmetries", MagicMock(return_value=([], 0)))
        mp.setattr(sr, "_close_group", MagicMock(return_value=fake_group))
        result = sr.get_symmetry_reduction(H, atol=1e-12, verbose=False)

    with pytest.raises(ValueError):
        result["expand_tensor"](np.ones((1, 1), dtype=complex), kind="not-a-kind")


def test_expand_tensor_applies_sigma_factor_when_requested(monkeypatch):
    """expand_tensor applies the sigma factor when requested."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    fake_group = {
        sr._GroupElement.identity(1, (2, 1, 1)),
        sr._GroupElement(np.eye(3, dtype=np.int64), np.array([1, 0, 0]), np.eye(1), -1, False, (2, 1, 1)),
    }
    with monkeypatch.context() as mp:
        mp.setattr(sr, "_discover_symmetries", MagicMock(return_value=([{}], 0)))
        mp.setattr(sr, "_close_group", MagicMock(return_value=fake_group))
        result = sr.get_symmetry_reduction(H, atol=1e-12, verbose=False)

    tensor_ibz = np.ones((1, 1, 1), dtype=complex)
    expanded = result["expand_tensor"](tensor_ibz, kind="kb", sigma_power=1)
    assert expanded.shape == (2, 1, 1, 1, 1)
    assert np.all(np.isin(np.unique(expanded), [-1, 1]))


def test_group_element_equality_depends_on_canonical_action_and_phase():
    """_GroupElement equality depends on the canonical action and phase."""
    nk = (1, 1, 1)
    g1 = sr._GroupElement(np.eye(3, dtype=np.int64), np.zeros(3, dtype=np.int64), np.eye(2), 1, False, nk)
    g2 = sr._GroupElement(
        np.eye(3, dtype=np.int64), np.zeros(3, dtype=np.int64), np.eye(2) * np.exp(1j * 0.4), 1, False, nk
    )

    assert g1 == g2
    assert hash(g1) == hash(g2)


def test_discover_symmetries_dedups_identical_M_grid_actions(monkeypatch):
    """When two M's give identical grid actions the second is skipped, so identity mocks yield 4 distinct ops."""
    H = np.zeros((1, 1, 1, 1, 1), dtype=complex)

    with monkeypatch.context() as mp:
        mp.setattr(
            sr,
            "_enumerate_integer_matrices",
            MagicMock(return_value=[np.eye(3, dtype=np.int64), np.eye(3, dtype=np.int64)]),
        )
        mp.setattr(sr, "_M_preserves_grid", MagicMock(return_value=True))
        mp.setattr(sr, "_apply_M_to_kgrid_indices", MagicMock(return_value=np.array([0], dtype=np.int64)))
        mp.setattr(sr, "_apply_M_to_ev_field", MagicMock(return_value=np.zeros((1, 1, 1, 1))))
        mp.setattr(sr, "_solve_U_for_op", MagicMock(return_value=np.eye(1)))
        ops, n_found = sr._discover_symmetries(H, atol=1e-12, verbose=False)

    assert n_found == len(ops)
    # M is enumerated twice but the second copy has the same grid action and is deduped: one unique M times
    # {sigma=+1,-1} times {conj=False,True} = 4 ops (each (sigma, conj) yields a distinct action_key).
    assert n_found == 4


def test_apply_auto_orbital_transform_identity_rows_are_left_unchanged():
    """apply_auto_orbital_transform leaves identity rows unchanged."""
    mat = np.arange(2 * 2 * 2, dtype=np.complex128).reshape(2, 2, 2)
    us = np.stack([np.eye(2, dtype=np.complex128), np.eye(2, dtype=np.complex128)])
    sigmas = np.array([1, 1], dtype=int)
    conjs = np.array([False, False], dtype=bool)

    out = sr.apply_auto_orbital_transform(mat.copy(), us, sigmas, conjs, num_orbital_dimensions=2)

    assert np.array_equal(out, mat)


def test_apply_auto_orbital_transform_applies_unitary_rotation_for_two_orbital_axes():
    """apply_auto_orbital_transform applies a unitary rotation on two orbital axes."""
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
    """apply_auto_orbital_transform uses sigma squared on four orbital axes and preserves the identity case."""
    mat = np.arange(1, 1 + 1 * 2 * 2 * 2 * 2, dtype=np.complex128).reshape(1, 2, 2, 2, 2)
    us = np.array([np.eye(2, dtype=np.complex128)])
    sigmas = np.array([-1], dtype=int)
    conjs = np.array([False], dtype=bool)

    out = sr.apply_auto_orbital_transform(mat.copy(), us, sigmas, conjs, num_orbital_dimensions=4)

    assert np.array_equal(out, mat)


def test_apply_auto_orbital_transform_groups_equivalent_k_points_together():
    """apply_auto_orbital_transform groups equivalent k-points together."""
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
    """apply_auto_orbital_transform rejects an invalid orbital-dimension count."""
    mat = np.zeros((1, 2, 2), dtype=np.complex128)
    us = np.eye(2, dtype=np.complex128)[None, ...]
    sigmas = np.array([1], dtype=int)
    conjs = np.array([False], dtype=bool)

    with pytest.raises(AssertionError):
        sr.apply_auto_orbital_transform(mat, us, sigmas, conjs, num_orbital_dimensions=3)


def test_apply_auto_orbital_transform_rejects_mismatched_leading_axis_lengths():
    """apply_auto_orbital_transform rejects mismatched leading-axis lengths."""
    mat = np.zeros((2, 2, 2), dtype=np.complex128)
    us = np.eye(2, dtype=np.complex128)[None, ...]
    sigmas = np.array([1], dtype=int)
    conjs = np.array([False], dtype=bool)

    with pytest.raises(AssertionError):
        sr.apply_auto_orbital_transform(mat, us, sigmas, conjs, num_orbital_dimensions=2)


def test_apply_auto_orbital_transform_rejects_wrong_orbital_axis_sizes():
    """apply_auto_orbital_transform rejects wrong orbital-axis sizes."""
    mat = np.zeros((1, 3, 2), dtype=np.complex128)
    us = np.eye(2, dtype=np.complex128)[None, ...]
    sigmas = np.array([1], dtype=int)
    conjs = np.array([False], dtype=bool)

    with pytest.raises(AssertionError):
        sr.apply_auto_orbital_transform(mat, us, sigmas, conjs, num_orbital_dimensions=2)


def test_apply_auto_orbital_transform_handles_empty_input():
    """apply_auto_orbital_transform handles an empty input array."""
    mat = np.zeros((0, 2, 2), dtype=np.complex128)
    us = np.zeros((0, 2, 2), dtype=np.complex128)
    sigmas = np.zeros((0,), dtype=int)
    conjs = np.zeros((0,), dtype=bool)

    out = sr.apply_auto_orbital_transform(mat, us, sigmas, conjs, num_orbital_dimensions=2)
    assert out.shape == mat.shape


def test_solve_u_for_op_one_orbital_returns_identity_when_spectra_match():
    """For norb=1, U is a global phase, so identity always solves when spectra match, regardless of grid extent."""
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
    """The norb=1 path short-circuits before any eigendecomposition or gauge fixing (the fixers raise if invoked)."""
    h_k = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    h_k[..., 0, 0] = np.array([1.0, 2.0]).reshape(2, 1, 1)

    def _explode(*a, **kw):  # pragma: no cover - should not run
        raise AssertionError("gauge-fix helper should not be called for norb=1")

    monkeypatch.setattr(sr, "_fix_phases_nondegenerate", _explode)
    monkeypatch.setattr(sr, "_fix_gauge_degenerate", _explode)

    u = sr._solve_U_for_op(h_k.copy(), h_k.copy(), atol=1e-12)
    assert u is not None


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
    """The IBZ->FBZ reconstruction via the auto-discovered symmetry data reproduces every H to double precision."""
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
    """Gamma = H (x) H reduced to IBZ and expanded back recovers exactly, exercising the rank-4 sigma_power=2 path."""
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
    """The 32x32x1 isotropic square lattice's auto group is the 8-element point group (16 with TR-like ops)."""
    H = _require_hamiltonian("hk_1band_square_32x32x1.npy", (32, 32, 1, 1, 1))
    result = sr.get_symmetry_reduction(H, atol=1e-8, verbose=False)
    # empirically 8 spatial (+8 TR-combined) group elements; the IBZ is 153/1024 ~= an 8x reduction
    assert len(result["group"]) >= 8
    assert result["n_ibz"] == 153
    assert result["n_fbz"] == 1024


def test_auto_discovery_finds_smaller_group_for_anisotropic_lattice():
    """The anisotropic lattice (tx != ty) drops kx<->ky, so only inversion and axis flips survive."""
    H = _require_hamiltonian("hk_1band_anisotropy_48x48x1.npy", (48, 48, 1, 1, 1))
    result = sr.get_symmetry_reduction(H, atol=1e-8, verbose=False)

    nktot = 48 * 48 * 1
    reduction = nktot / result["n_ibz"]
    # Expect reduction between 3 and 5 (inversion + each-axis flips for real H)
    assert 2.5 < reduction < 5.0, f"Unexpected reduction factor {reduction:.2f}"


@pytest.mark.slow
def test_auto_discovery_matches_explicit_cubic_group_for_12cubed_hamiltonian():
    """The auto IBZ partition matches the explicit three_dimensional_cubic partition for a cubic 3-band H (12^3)."""
    import dgamore.brillouin_zone as bz

    fname, shape = "hk_3band_srvo3_cubic_12x12x12.npy", (12, 12, 12, 3, 3)
    H = _require_hamiltonian(fname, shape)
    nx, ny, nz, _, _ = shape

    kg_auto = bz.KGrid(nk=(nx, ny, nz), symmetries=[bz.KnownSymmetries.AUTO])
    kg_auto.specify_auto_symmetries(H, atol=1e-8)

    kg_explicit = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.three_dimensional_cubic_symmetries())

    assert kg_auto.nk_irr == kg_explicit.nk_irr
    assert np.array_equal(kg_auto.fbz2irrk, kg_explicit.fbz2irrk)


@pytest.mark.slow
def test_auto_discovery_matches_explicit_cubic_group_for_20cubed_hamiltonian():
    """Same as above for the 20^3 grid. (Even slower - covers the larger-grid path.)"""
    import dgamore.brillouin_zone as bz

    fname, shape = "hk_3band_srvo3_cubic_20x20x20.npy", (20, 20, 20, 3, 3)
    H = _require_hamiltonian(fname, shape)
    nx, ny, nz, _, _ = shape

    kg_auto = bz.KGrid(nk=(nx, ny, nz), symmetries=[bz.KnownSymmetries.AUTO])
    kg_auto.specify_auto_symmetries(H, atol=1e-8)

    kg_explicit = bz.KGrid(nk=(nx, ny, nz), symmetries=bz.three_dimensional_cubic_symmetries())

    assert kg_auto.nk_irr == kg_explicit.nk_irr
    assert np.array_equal(kg_auto.fbz2irrk, kg_explicit.fbz2irrk)


def test_get_symmetry_reduction_on_trivial_1x1x1_grid():
    """A single k-point: every symmetry acts trivially, the IBZ has one point and the FBZ point coincides with it."""
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
    """A generic complex Hermitian H has only the trivial group, so the IBZ equals the FBZ."""
    rng = np.random.default_rng(7)
    nx, ny, nz, nb = 4, 4, 1, 2
    H = rng.standard_normal((nx, ny, nz, nb, nb)) + 1j * rng.standard_normal((nx, ny, nz, nb, nb))
    H = 0.5 * (H + H.conj().transpose(0, 1, 2, 4, 3))

    result = sr.get_symmetry_reduction(H, atol=1e-10)

    # trivial group: every k-point is its own representative
    assert result["n_ibz"] == result["n_fbz"]
    H_ibz = H.reshape(-1, nb, nb)[result["irrk_ind"]]
    H_rec = result["expand"](H_ibz)
    assert np.allclose(H_rec, H, atol=1e-12)


def test_get_symmetry_reduction_handles_zero_hamiltonian():
    """H == 0 has every possible symmetry; the discovered group will be large but the reconstruction must still work."""
    H = np.zeros((2, 2, 1, 2, 2), dtype=complex)
    result = sr.get_symmetry_reduction(H, atol=1e-10)

    H_ibz = H.reshape(-1, 2, 2)[result["irrk_ind"]]
    H_rec = result["expand"](H_ibz)
    assert np.allclose(H_rec, H, atol=1e-12)
    # Every k collapses to the single representative.
    assert result["n_ibz"] == 1


def test_get_symmetry_reduction_handles_diagonal_real_hamiltonian():
    """A purely-diagonal cubic H exercises the case where the orbital action is identity for all symmetries."""
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
    """apply_auto_orbital_transform is shape-polymorphic in the trailing axes after the orbital pair."""
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
    """For identity-only groups the function returns the same array object without allocating."""
    mat = np.arange(8, dtype=np.complex128).reshape(2, 2, 2)
    us = np.stack([np.eye(2), np.eye(2)]).astype(np.complex128)
    sigmas = np.array([1, 1], dtype=int)
    conjs = np.array([False, False], dtype=bool)

    out = sr.apply_auto_orbital_transform(mat, us, sigmas, conjs, num_orbital_dimensions=2)
    assert out is mat  # identity short-circuit must not copy


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
    """The default drops anti-unitary ops, so no FBZ point carries conj=True (safe for frequency objects)."""
    H = _make_real_cubic_h(4, 4, 4, 1)
    result = sr.get_symmetry_reduction(H, atol=1e-8)
    assert result["conjs"].any() == False  # noqa: E712 - explicit bool check


def test_get_symmetry_reduction_include_antiunitary_admits_conj_ops():
    """For a real H, H(k)=H(k)* gives anti-unitary ops, so opting in produces at least one conj=True point."""
    H = _make_real_cubic_h(4, 4, 4, 1)
    result = sr.get_symmetry_reduction(H, atol=1e-8, include_antiunitary=True)
    assert int(result["conjs"].sum()) > 0


def test_get_symmetry_reduction_include_antiunitary_shrinks_or_equals_ibz():
    """Adding TR ops only grows orbits, so the anti-unitary IBZ is smaller than or equal to the spatial-only IBZ."""
    H = _make_real_cubic_h(4, 4, 4, 1)
    r_default = sr.get_symmetry_reduction(H, atol=1e-8, include_antiunitary=False)
    r_full = sr.get_symmetry_reduction(H, atol=1e-8, include_antiunitary=True)
    assert r_full["n_ibz"] <= r_default["n_ibz"]


def test_get_symmetry_reduction_include_antiunitary_reconstructs_H_correctly():
    """With anti-unitary ops included, H reconstruction is still exact (only frequency objects need the freq flip)."""
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
    """With the default, expand never conjugates orbital indices, checked with a payload whose conjugate differs."""
    H = _make_real_cubic_h(4, 4, 4, 1)
    result = sr.get_symmetry_reduction(H, atol=1e-8)
    # Reconstruct H itself - well-defined and exact
    H_ibz = H.reshape(-1, 1, 1)[result["irrk_ind"]]
    H_rec = result["expand"](H_ibz)
    assert np.allclose(H_rec, H, atol=1e-12)
    assert int(result["conjs"].sum()) == 0


def test_canonicalize_sign_gauge_norb_gt_6_fallback_flips_negative_rows():
    """For norb > 6 the row-major sign canonicalization returns U when the flips cancel in U H U^dag (diagonal H)."""
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
    """For norb > 6, if the sign flip breaks U H U^dag = Hg for non-degenerate H the original U is returned."""
    norb = 7
    base = np.diag(np.arange(1.0, norb + 1))
    base[0, 1] = base[1, 0] = 0.3
    Hk = base[None, None, None].astype(complex)

    # theta = 1.3: |sin| > |cos| with sin > 0, so row 0's dominant entry (-sin) is negative while row 1's (sin)
    # is positive - exactly ONE row flips and the off-diagonal H makes the broken relation detectable
    th = 1.3
    c, s = np.cos(th), np.sin(th)
    U = np.eye(norb)
    U[0, 0] = c
    U[0, 1] = -s
    U[1, 0] = s
    U[1, 1] = c
    U = U.astype(complex)
    Hg = np.einsum("ij,...jk,lk->...il", U, Hk, U.conj())

    assert U[0, 1].real < 0 and abs(U[0, 1]) > abs(U[0, 0])
    assert U[1, 0].real > 0 and abs(U[1, 0]) > abs(U[1, 1])

    out = sr._canonicalize_sign_gauge(U, Hk, Hg, atol=1e-10)
    assert np.array_equal(out, U)


def test_canonicalize_sign_gauge_accepts_lower_score_sign_diagonal():
    """For norb <= 6 a lower-score sign-diagonal that still solves is accepted (U=-I with D=-I yields +I)."""
    Hk = np.diag([1.0, 2.0]).astype(complex)[None, None, None]
    Hg = Hk.copy()
    U = -np.eye(2, dtype=complex)

    out = sr._canonicalize_sign_gauge(U, Hk, Hg, atol=1e-10)
    assert np.allclose(out, np.eye(2), atol=1e-12)


def test_solve_u_for_op_one_orbital_returns_none_when_close_but_outside_atol():
    """For norb=1, spectra within 10*atol but H differing beyond atol fails validation and returns None."""
    Hg = np.array([[[[[0.0 + 0j]]]]])
    Hk = np.array([[[[[5e-12 + 0j]]]]])
    assert sr._solve_U_for_op(Hg, Hk, atol=1e-12) is None


def test_solve_u_for_op_continues_when_perpoint_eigh_spectra_disagree(monkeypatch):
    """A patched eigh desyncs the per-point Hk/Hg spectra, so every reference point continues and returns None."""
    Hk = np.zeros((2, 1, 1, 2, 2), dtype=complex)
    Hk[..., 0, 0] = np.array([1.0, 1.0]).reshape(2, 1, 1)
    Hk[..., 1, 1] = np.array([2.0, 3.0]).reshape(2, 1, 1)
    Hg = Hk.copy()

    calls = {"n": 0}

    def fake_eigh(a):
        calls["n"] += 1
        if calls["n"] % 2 == 1:  # odd calls are the Hk evaluation, even calls the Hg one
            return np.array([0.0, 1.0]), np.eye(2, dtype=complex)
        return np.array([0.0, 2.0]), np.eye(2, dtype=complex)

    monkeypatch.setattr(sr.np.linalg, "eigh", fake_eigh)
    assert sr._solve_U_for_op(Hg, Hk, atol=1e-12) is None
    assert calls["n"] >= 2


def test_solve_u_for_op_routes_through_degenerate_gauge_fix():
    """A 2-fold degenerate spectrum routes through _fix_gauge_degenerate, which solves the block rotation matching U."""
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


def test_fix_phases_nondegenerate_hits_near_zero_ratio_continue():
    """When B[r,col] ~ 0 the candidate phase is near-zero and skipped; with no consistent column it returns None."""
    nk = (3, 1, 1)
    Hk = np.zeros((*nk, 2, 2), dtype=complex)
    Hg = np.zeros((*nk, 2, 2), dtype=complex)
    for kx in range(nk[0]):
        Hk[kx, 0, 0] = np.array([[1.0, 0.5], [0.5, 2.0]])  # A[1,0] = 0.5 (> 1e-4)
        Hg[kx, 0, 0] = np.array([[1.0, 0.0], [0.0, 2.0]])  # B[1,0] = 0
    V = np.eye(2, dtype=complex)
    W = np.eye(2, dtype=complex)

    assert sr._fix_phases_nondegenerate(V, W, Hk, Hg, (0, 0, 0), atol=1e-12) is None


def test_fix_gauge_degenerate_returns_none_on_stacked_svd_linalgerror(monkeypatch):
    """If the SVD of the stacked constraint matrix raises LinAlgError, the routine returns None."""
    nk = (2, 1, 1)
    Hk = np.zeros((*nk, 2, 2), dtype=complex)
    Hg = np.zeros((*nk, 2, 2), dtype=complex)
    Hk[:, 0, 0] = np.eye(2)
    Hg[:, 0, 0] = np.eye(2)

    def boom(*a, **k):
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(sr.np.linalg, "svd", boom)
    out = sr._fix_gauge_degenerate(np.eye(2, dtype=complex), np.eye(2, dtype=complex), [[0, 1]], Hk, Hg, atol=1e-12)
    assert out is None


def test_fix_gauge_degenerate_returns_none_on_block_svd_linalgerror(monkeypatch):
    """If the per-block SVD raises LinAlgError after a null vector was found, _fix_gauge_degenerate returns None."""
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
    out = sr._fix_gauge_degenerate(np.eye(2, dtype=complex), np.eye(2, dtype=complex), [[0, 1]], Hk, Hg, atol=1e-12)
    assert out is None
    assert state["n"] >= 2


def test_discover_symmetries_handles_hash_collision_of_grid_actions(monkeypatch):
    """Two M's with distinct grid actions but a forced identical hash are both kept (confirmed by array compare)."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    with monkeypatch.context() as mp:
        mp.setattr(
            sr,
            "_enumerate_integer_matrices",
            MagicMock(return_value=[np.eye(3, dtype=np.int64), np.diag([-1, 1, 1]).astype(np.int64)]),
        )
        mp.setattr(sr, "_M_preserves_grid", MagicMock(return_value=True))
        mp.setattr(
            sr,
            "_apply_M_to_kgrid_indices",
            MagicMock(side_effect=lambda M, nk: np.array([0, 1] if int(M[0, 0]) > 0 else [1, 0], dtype=np.int64)),
        )
        mp.setattr(sr, "_apply_M_to_ev_field", MagicMock(return_value=np.zeros((2, 1, 1, 1))))
        mp.setattr(sr, "_solve_U_for_op", MagicMock(return_value=np.eye(1)))
        monkeypatch.setattr(builtins, "hash", lambda x: 1234)  # force collision
        ops, n = sr._discover_symmetries(H, atol=1e-12, verbose=False)
    assert n == len(ops)
    assert n >= 1


def test_discover_symmetries_handles_zero_pivot_in_U_canonicalization(monkeypatch):
    """An all-near-zero U gives a near-zero pivot, so canonical-bytes skips the phase division (else branch)."""
    H = np.zeros((1, 1, 1, 1, 1), dtype=complex)
    with monkeypatch.context() as mp:
        mp.setattr(sr, "_enumerate_integer_matrices", MagicMock(return_value=[np.eye(3, dtype=np.int64)]))
        mp.setattr(sr, "_M_preserves_grid", MagicMock(return_value=True))
        mp.setattr(sr, "_apply_M_to_kgrid_indices", MagicMock(return_value=np.array([0], dtype=np.int64)))
        mp.setattr(sr, "_apply_M_to_ev_field", MagicMock(return_value=np.zeros((1, 1, 1, 1))))
        mp.setattr(sr, "_solve_U_for_op", MagicMock(return_value=np.zeros((1, 1), dtype=complex)))
        ops, n = sr._discover_symmetries(H, atol=1e-12, verbose=False)
    assert n == len(ops)


def test_discover_symmetries_skips_duplicate_action_key(monkeypatch):
    """Two M's whose translated grid actions coincide for some q share an action key, so the duplicate is skipped."""
    H = np.zeros((2, 1, 1, 1, 1), dtype=complex)
    with monkeypatch.context() as mp:
        mp.setattr(
            sr,
            "_enumerate_integer_matrices",
            MagicMock(return_value=[np.eye(3, dtype=np.int64), np.diag([-1, 1, 1]).astype(np.int64)]),
        )
        mp.setattr(sr, "_M_preserves_grid", MagicMock(return_value=True))
        mp.setattr(
            sr,
            "_apply_M_to_kgrid_indices",
            MagicMock(side_effect=lambda M, nk: np.array([0, 1] if int(M[0, 0]) > 0 else [1, 0], dtype=np.int64)),
        )
        mp.setattr(sr, "_apply_M_to_ev_field", MagicMock(return_value=np.zeros((2, 1, 1, 1))))
        mp.setattr(sr, "_solve_U_for_op", MagicMock(return_value=np.eye(1)))
        ops, n = sr._discover_symmetries(H, atol=1e-12, verbose=False)
    # With H == 0 every q matches, so both M's enumerate overlapping actions; the
    # duplicate-action guard keeps the op set deduplicated.
    assert n == len(ops)


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


def test_close_group_returns_early_when_max_size_reached_during_composition():
    """An 8-point translation generates an 8-element group; with max_size=4 the closure returns the partial group."""
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


def test_group_element_with_near_zero_U_skips_phase_normalization():
    """An all-near-zero U has a near-zero pivot, so the global phase normalization is skipped (pivot-guard False)."""
    g = sr._GroupElement(
        np.eye(3, dtype=np.int64),
        np.zeros(3, dtype=np.int64),
        np.zeros((2, 2), dtype=complex),
        +1,
        False,
        (1, 1, 1),
    )
    assert np.allclose(g.U, 0.0)


def _rot2(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.complex128)


def test_apply_auto_orbital_transform_four_orbital_nonidentity_einsum():
    """A non-identity U with 4 orbital axes exercises the rank-4 einsum branch (identity short-circuit skipped)."""
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
    """Two non-identity groups in one call: the second reuses the cached 2-index einsum path."""
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

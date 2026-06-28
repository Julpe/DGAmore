# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import itertools
import os

import numpy as np

import dgamore.config as config
from dgamore.hamiltonian import Hamiltonian
from dgamore.interaction import Interaction
from dgamore.local_sde import get_local_hartree_fock
from dgamore.n_point_base import SpinChannel
from dgamore.nonlocal_sde import _init_mu_history, get_hartree_fock

LOCAL_SDE_DATA = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/local_sde"


def test_init_mu_history_fresh_uses_current_mu():
    """A fresh run (starting_iter == 0) seeds the history with the current (DMFT) chemical potential and leaves it."""
    config.sys.mu = 0.7
    config.self_consistency.previous_sc_path = ""

    mu_history = _init_mu_history(0)

    assert mu_history == [0.7]
    assert config.sys.mu == 0.7  # unchanged on a fresh run


def test_init_mu_history_from_previous_syncs_global_mu(monkeypatch, tmp_path):
    """Resuming from a previous run seeds the history with, and syncs config.sys.mu to, that run's last mu."""
    config.sys.mu = 0.3  # stale DMFT value that must be overwritten
    config.self_consistency.previous_sc_path = str(tmp_path)
    previous_mu = 1.5
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: np.array([0.9, 1.2, previous_mu]))

    mu_history = _init_mu_history(3)

    assert mu_history == [previous_mu]
    assert config.sys.mu == previous_mu  # global synced to the previous run's converged mu


def test_nonlocal_hartree_fock_matches_local_reference():
    """
    The non-local Hartree-Fock term must reduce to the (DMFT-validated) local Hartree-Fock at every k when the
    non-local interaction vanishes (V=0) and the occupation is k-independent. This covers the same orbital-index
    convention as the local test (:func:`dgamore.local_sde.get_local_hartree_fock`): the Hartree must pick up the
    inter-orbital density U' (stored at U_{abab}), not U_{aabb}. Reverting the ``qacbd`` contraction in
    ``get_hartree_fock`` makes this fail.
    """
    nb = 2
    nk = (2, 2, 1)
    nq_tot = int(np.prod(nk))

    config.lattice.nk = nk
    config.lattice.nq = nk
    config.sys.n_bands = nb

    occ = np.load(f"{LOCAL_SDE_DATA}/occ.npy", allow_pickle=False)
    config.sys.occ = occ
    config.sys.occ_k = np.broadcast_to(occ, nk + (nb, nb)).copy()  # k-independent occupation

    u_loc = Hamiltonian().read_umatrix(f"{LOCAL_SDE_DATA}/u_matrix.dat").get_local_u()
    v_nonloc = Interaction(
        np.zeros((nq_tot, nb, nb, nb, nb), dtype=u_loc.mat.dtype),
        SpinChannel.NONE,
        nk,
        has_compressed_q_dimension=True,
    )
    q_list = np.array(list(itertools.product(*[range(n) for n in nk])))

    hartree, fock = get_hartree_fock(u_loc, v_nonloc, q_list)
    hf_nonlocal = (hartree + fock)[..., 0]  # [nk_tot, nb, nb]

    sigma_hf_ref = np.load(f"{LOCAL_SDE_DATA}/sigma_HF.npy", allow_pickle=False)
    assert hf_nonlocal.shape == (nq_tot, nb, nb)
    assert np.allclose(hf_nonlocal, sigma_hf_ref[None, ...])
    # the same reference is the local Hartree-Fock, so the two SDE paths agree
    assert np.allclose(hf_nonlocal, get_local_hartree_fock(u_loc, occ)[None, ...])

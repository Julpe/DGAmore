# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import logging
import os
from unittest.mock import MagicMock

import mpi4py.MPI as MPI
import numpy as np
import pytest

import moldga.brillouin_zone as bz


def pytest_addoption(parser):
    """Register the --runslow flag so individual tests can be marked @pytest.mark.slow
    and skipped by default. CI can opt in via `pytest --runslow`."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked as slow (large-grid auto-symmetry discovery, etc.).",
    )


def pytest_configure(config):
    """Register the 'slow' marker so pytest doesn't warn about an unknown marker."""
    config.addinivalue_line("markers", "slow: mark test as slow (deselect with '-m \"not slow\"')")


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.slow tests unless --runslow was given."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def mock_does_not_delete_files(monkeypatch):
    # Make os.remove do nothing
    monkeypatch.setattr(os, "remove", lambda path: None)


@pytest.fixture(autouse=True)
def mock_does_not_create_folders(monkeypatch):
    # Make os.remove do nothing
    monkeypatch.setattr(os, "makedirs", lambda path: None)


@pytest.fixture(autouse=True)
def mock_numpy_save(monkeypatch):
    # Automatically mock numpy.save for all tests.
    def fake_save(file, arr, **kwargs):
        pass

    monkeypatch.setattr(np, "save", fake_save)
    monkeypatch.setattr(np, "savetxt", fake_save)
    yield


@pytest.fixture(autouse=True)
def mock_logger(monkeypatch):
    # Automatically mock logger.log for all tests.
    logger_mock = MagicMock()
    monkeypatch.setattr(logging, "getLogger", lambda name=None: logger_mock)
    monkeypatch.setattr(logging, "Logger", MagicMock(return_value=logger_mock))
    yield logger_mock


def create_default_config(config, folder: str):
    config.box.niw_core = -1
    config.box.niv_core = -1
    config.box.niv_shell = 10
    config.output.do_plotting = False
    config.lattice.nk = (4, 4, 1)
    config.lattice.nq = config.lattice.nk
    config.lattice.k_grid = bz.KGrid(config.lattice.nk, symmetries=bz.two_dimensional_square_symmetries())
    config.lattice.q_grid = config.lattice.k_grid
    config.lattice.type = "from_wannierHK"
    config.lattice.interaction_type = "kanamori_from_dmft"
    config.lattice.er_input = f"{folder}/wannier.hk"
    config.dmft.input_path = folder
    config.dmft.do_sym_v_vp = True
    config.dmft.n_ineq = 1
    config.dmft.ineq_ordering = [1]
    config.dmft.n_bands_per_ineq = []
    config.eliashberg.perform_eliashberg = False
    config.self_consistency.mixing = 1
    config.self_consistency.max_iter = 1


def create_comm_mock():
    comm_mock = MagicMock()

    # Fundamental MPI properties
    comm_mock.Get_size.return_value = 1
    comm_mock.Get_rank.return_value = 0
    comm_mock.size = 1
    comm_mock.rank = 0
    comm_mock.IN_PLACE = MPI.IN_PLACE

    # Internal state to track pending non-blocking operations
    # {tag: buffer_pointer}
    pending_irecvs = {}
    pending_isends = {}

    def mock_Irecv(buf, source, tag=0):
        if tag in pending_isends:
            np.copyto(buf, pending_isends[tag])
            del pending_isends[tag]
        else:
            pending_irecvs[tag] = buf
        return MPI.REQUEST_NULL

    def mock_Isend(buf, dest, tag=0):
        if tag in pending_irecvs:
            np.copyto(pending_irecvs[tag], buf)
            del pending_irecvs[tag]
        else:
            pending_isends[tag] = np.copy(buf)
        return MPI.REQUEST_NULL

    comm_mock.Irecv.side_effect = mock_Irecv
    comm_mock.Isend.side_effect = mock_Isend

    # --- Lowercase (Object) Loopback ---
    # Used for bcast/scatter/send_to_rank
    comm_mock.bcast.side_effect = lambda obj, root=0: obj
    comm_mock.allgather.side_effect = lambda obj: [obj]

    # Point-to-point object exchange
    obj_bus = {}

    def mock_send(obj, dest, tag=0):
        obj_bus[tag] = obj

    def mock_recv(source=0, tag=0):
        return obj_bus.get(tag, None)

    comm_mock.send.side_effect = mock_send
    comm_mock.recv.side_effect = mock_recv

    # --- Collective Uppercase (NumPy) ---
    def bcast_numpy(buf, root=0):
        return None  # Data already in-place for 1-rank

    def allreduce_numpy(sendbuf, recvbuf, op=None):
        if sendbuf is not MPI.IN_PLACE:
            np.copyto(recvbuf, sendbuf)
        return None

    comm_mock.Bcast.side_effect = bcast_numpy
    comm_mock.Allreduce.side_effect = allreduce_numpy

    # Split should return itself
    comm_mock.Split.return_value = comm_mock

    return comm_mock

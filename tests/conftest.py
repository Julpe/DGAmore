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


@pytest.fixture(autouse=True)
def mock_does_not_delete_files(monkeypatch):
    # Make os.remove do nothing
    monkeypatch.setattr(os, "remove", lambda path: None)


@pytest.fixture(autouse=True)
def mock_numpy_save(monkeypatch):
    # Automatically mock numpy.save for all tests.
    def fake_save(file, arr, **kwargs):
        pass

    monkeypatch.setattr(np, "save", fake_save)
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
    config.eliashberg.perform_eliashberg = False
    config.self_consistency.mixing = 1
    config.self_consistency.max_iter = 1


def create_comm_mock():
    comm_mock = MagicMock()
    comm_mock.Get_size.return_value = 1
    comm_mock.Get_rank.return_value = 0
    comm_mock.size = 1
    comm_mock.rank = 0
    comm_mock.barrier.return_value = None

    # lowercase bcast for arbitrary Python objects
    comm_mock.bcast.side_effect = lambda obj, root=0: obj

    # uppercase Bcast for NumPy arrays (in-place)
    def bcast_mock(buf, root=0):
        pass  # already in place, nothing to do for single rank

    comm_mock.Bcast.side_effect = bcast_mock

    # in-place Allreduce
    def allreduce_mock(sendbuf, recvbuf=None, op=None):
        if sendbuf is MPI.IN_PLACE:
            pass  # recvbuf is the array, already correct for single rank
        else:
            np.copyto(recvbuf, sendbuf)

    comm_mock.Allreduce.side_effect = allreduce_mock

    def gatherv_mock(sendbuf, recvlist, root=0):
        tot_result = recvlist[0]
        np.copyto(tot_result, sendbuf)
        return tot_result

    comm_mock.Gatherv.side_effect = gatherv_mock

    def scatterv_mock(sendlist, recvbuf, root=0):
        if sendlist is None:
            return recvbuf
        full_data = sendlist[0]
        np.copyto(recvbuf, full_data)
        return recvbuf

    comm_mock.Scatterv.side_effect = scatterv_mock

    return comm_mock

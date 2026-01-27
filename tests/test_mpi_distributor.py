import os
from unittest.mock import patch

import numpy as np
import pytest

from moldga import config
from moldga.dga_logger import DgaLogger
from moldga.mpi_distributor import MpiDistributor
from tests import conftest


@pytest.fixture
def setup():
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/end_2_end"
    comm_mock = conftest.create_comm_mock()

    with patch("mpi4py.MPI.COMM_WORLD", comm_mock):
        config.logger = DgaLogger(comm_mock, "./")
        conftest.create_default_config(config, folder)
        yield comm_mock


def test_scatter_distributes_data_correctly_among_ranks(setup):
    comm = setup
    ntasks = 10
    distributor = MpiDistributor.create_distributor(ntasks, comm)
    full_data = np.arange(ntasks)
    scattered_data = distributor.scatter(full_data, root=0)
    expected_data = full_data[distributor.my_slice]
    assert np.array_equal(scattered_data, expected_data)


def test_scatter_handles_empty_data(setup):
    comm = setup
    ntasks = 0
    distributor = MpiDistributor.create_distributor(ntasks, comm)
    full_data = np.array([])
    scattered_data = distributor.scatter(full_data, root=0)
    assert scattered_data.size == 0


def test_scatter_raises_error_for_non_numpy_array(setup):
    comm = setup
    ntasks = 10
    distributor = MpiDistributor.create_distributor(ntasks, comm)
    with pytest.raises(TypeError, match="full_data must be a numpy array or None"):
        distributor.scatter(full_data="not_an_array", root=0)


def test_scatter_handles_none_full_data(setup):
    comm = setup
    ntasks = 10
    distributor = MpiDistributor.create_distributor(ntasks, comm)
    scattered_data = distributor.scatter(full_data=None, root=0)
    assert scattered_data.shape == (distributor.my_size,)


def test_scatter_raises_error_for_mismatched_data_length(setup):
    comm = setup
    ntasks = 10
    distributor = MpiDistributor.create_distributor(ntasks, comm)
    full_data = np.arange(5)  # Mismatched length
    if distributor.my_rank == 0:
        with pytest.raises(ValueError, match="Mismatch in scatter!"):
            distributor.scatter(full_data, root=0)

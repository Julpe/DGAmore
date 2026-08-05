# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

from types import SimpleNamespace

import pytest

from dgamore.dga_logger import DgaLogger

from tests.conftest import create_comm_mock


def _obj(memory_in_gb: float):
    """Minimal stand-in for an IHaveMat object, exposing only the footprint the logger reads."""
    return SimpleNamespace(memory_usage_in_gb=memory_in_gb)


@pytest.fixture
def logger_and_stream(mock_logger):
    """A rank-0 logger together with the mocked stream logger recording its emitted messages."""
    return DgaLogger(create_comm_mock(), "./"), mock_logger


def _last_message(stream) -> str:
    """The message text of the most recent emitted log record."""
    return stream.log.call_args.args[1]


def test_log_memory_usage_single_copy_reports_the_bare_footprint(logger_and_stream):
    """A single copy is logged as its plain footprint, without a copy breakdown."""
    logger, stream = logger_and_stream
    logger.log_memory_usage("giwk", _obj(1.5))
    assert _last_message(stream).endswith("giwk use(s) (GB): 1.500000.")


def test_log_memory_usage_scales_by_the_number_of_copies(logger_and_stream):
    """Several copies are logged as the job-wide total plus the count-times-footprint breakdown."""
    logger, stream = logger_and_stream
    logger.log_memory_usage("giwk", _obj(1.5), 4)
    assert _last_message(stream).endswith("giwk use(s) (GB): 6.000000 (4 ranks x 1.500000).")


def test_log_memory_usage_labels_node_shared_copies(logger_and_stream):
    """A node-shared quantity is counted per node and labeled as such, not multiplied by the rank count."""
    logger, stream = logger_and_stream
    logger.log_memory_usage("giwk", _obj(13.0), 2, per="node")
    assert _last_message(stream).endswith("giwk use(s) (GB): 26.000000 (2 nodes x 13.000000).")


def test_log_memory_usage_scales_a_stand_in_object_up_to_the_reported_quantity(logger_and_stream):
    """A stand-in object holding a fraction of the quantity is scaled up before the copy count is applied."""
    logger, stream = logger_and_stream
    logger.log_memory_usage("Auxiliary susceptibility (dens)", _obj(0.5), 4, scale=160)
    assert _last_message(stream).endswith(
        "Auxiliary susceptibility (dens) use(s) (GB): 320.000000 (4 ranks x 80.000000)."
    )


def test_log_memory_usage_sums_a_sequence_of_objects(logger_and_stream):
    """A list of objects is summed into one copy before it is scaled by the copy count."""
    logger, stream = logger_and_stream
    logger.log_memory_usage("g_dmft & sigma_dmft", [_obj(0.25), _obj(0.75)], 2)
    assert _last_message(stream).endswith("g_dmft & sigma_dmft use(s) (GB): 2.000000 (2 ranks x 1.000000).")


def test_log_memory_usage_skips_none_entries_of_a_sequence(logger_and_stream):
    """``None`` entries of a sequence are ignored instead of raising."""
    logger, stream = logger_and_stream
    logger.log_memory_usage("g2_dens & g2_magn", [_obj(2.0), None])
    assert _last_message(stream).endswith("g2_dens & g2_magn use(s) (GB): 2.000000.")


def test_log_memory_usage_emits_nothing_without_an_object(logger_and_stream):
    """A ``None`` object and an all-``None`` sequence both emit no message at all."""
    logger, stream = logger_and_stream
    calls_before = stream.log.call_count
    logger.log_memory_usage("giwk", None, 4)
    logger.log_memory_usage("giwk", [None, None], 4)
    assert stream.log.call_count == calls_before


def test_log_memory_usage_respects_allowed_ranks(logger_and_stream):
    """A rank outside the allowed ranks stays silent."""
    logger, stream = logger_and_stream
    calls_before = stream.log.call_count
    logger.log_memory_usage("giwk", _obj(1.5), 4, allowed_ranks=(1,))
    assert stream.log.call_count == calls_before

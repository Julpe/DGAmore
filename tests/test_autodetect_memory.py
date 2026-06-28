# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
"""Tests for DGAmore.autodetect_memory_settings (node-total budget + floor-semantics wiring)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import dgamore.config as config
import dgamore.DGAmore as dgamore_main
from dgamore.memory_estimator import estimate_peaks
from tests.conftest import create_comm_mock

# the q-grid / box parameters the fake_system fixture installs, so tests can reproduce the driver's estimate
# niv_cut == min(niw_core + niv_full + 10, niv_dmft) == min(32, 50) == 32 with the fixture's niv_dmft below.
FIXTURE_PARAMS = dict(n_bands=1, nk_tot=64, nk_irr=10, niw_core=10, niv_core=10, niv_full=12, niv_cut=32, niv_pp=5)


@pytest.fixture
def fake_system(monkeypatch):
    """Sets up a minimal single-band config and a controllable psutil/MPI environment."""
    config.sys.n_bands = FIXTURE_PARAMS["n_bands"]
    config.box.niw_core = FIXTURE_PARAMS["niw_core"]
    config.box.niv_core = FIXTURE_PARAMS["niv_core"]
    config.box.niv_full = FIXTURE_PARAMS["niv_full"]
    config.box.niv_dmft = 50  # so niv_cut = min(niw_core + niv_full + 10, niv_dmft) = min(32, 50) = 32
    config.eliashberg.perform_eliashberg = False
    config.memory.save_memory_for_chi0q = False
    config.memory.save_memory_for_chiq_aux = False
    config.memory.save_memory_for_sde = False
    config.memory.save_memory_for_fq = False
    config.memory.save_memory_for_lanczos = False
    config.lattice.q_grid = SimpleNamespace(nk_tot=FIXTURE_PARAMS["nk_tot"], nk_irr=FIXTURE_PARAMS["nk_irr"])
    config.logger = MagicMock()

    monkeypatch.setattr(dgamore_main.MPI, "Get_processor_name", lambda: "node0", raising=False)

    def _set_available(num_bytes):
        monkeypatch.setattr(
            dgamore_main.psutil, "virtual_memory", lambda: SimpleNamespace(available=num_bytes), raising=False
        )

    return _set_available


def _mock_comm(size=1, allgather=None):
    """Single-process MagicMock comm with a working scalar allgather and the given size."""
    comm = create_comm_mock()
    comm.size = size
    comm.Get_size.return_value = size
    if allgather is not None:
        comm.allgather.side_effect = allgather
    return comm


def _node_total(branch, which, r, with_eliashberg=False):
    """Reproduces the driver's node total = r*(baseline+distributed) + single for one branch/path."""
    baseline, peaks = estimate_peaks(**FIXTURE_PARAMS, n_ranks=r, with_eliashberg=with_eliashberg)
    bp = peaks[branch]
    distributed = bp.off_distributed if which == "off" else bp.on_distributed
    single = bp.off_single if which == "off" else bp.on_single
    return r * (baseline + distributed) + single


def _all_node_totals(which, r, with_eliashberg):
    _, peaks = estimate_peaks(**FIXTURE_PARAMS, n_ranks=r, with_eliashberg=with_eliashberg)
    return [_node_total(k, which, r, with_eliashberg) for k in peaks]


def test_large_memory_keeps_all_flags_off(fake_system):
    """A large free-memory budget on a tiny problem leaves all lean flags off."""
    fake_system(64 * 1024**3)  # 64 GiB free, tiny problem -> nothing forced on
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert config.memory.save_memory_for_chiq_aux is False
    assert config.memory.save_memory_for_chi0q is False
    assert config.memory.save_memory_for_sde is False


def test_tiny_memory_forces_lean_flags_on(fake_system):
    """A budget below chiq_aux's fast-path node total forces its lean flag on without overflowing the node."""
    off = _node_total("chiq_aux", "off", r=1)
    max_on = max(_all_node_totals("on", r=1, with_eliashberg=False))
    budget = 0.5 * (max_on + off)
    fake_system(int(budget / 0.9))
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert config.memory.save_memory_for_chiq_aux is True


def test_user_true_is_preserved_as_floor(fake_system):
    """A user-set lean flag survives autodetection even when memory is plentiful."""
    fake_system(64 * 1024**3)  # plenty of memory; auto would leave it off
    config.memory.save_memory_for_chiq_aux = True
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert config.memory.save_memory_for_chiq_aux is True  # user setting survives


def test_chi0q_single_rank_peak_fits_under_node_total(fake_system):
    """chi0q's single-rank transient peak stays off against the node total while the heavier branch is forced on."""
    chi0q_off = _node_total("chi0q", "off", r=1)
    chiq_aux_off = _node_total("chiq_aux", "off", r=1)
    assert chi0q_off < chiq_aux_off  # sanity: chi0q is the lighter one here
    budget = 0.5 * (chi0q_off + chiq_aux_off)
    fake_system(int(budget / 0.9))
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert config.memory.save_memory_for_chi0q is False  # single-rank peak fits the node
    assert config.memory.save_memory_for_chiq_aux is True  # heavier branch still forced on


def test_more_ranks_per_node_raise_the_distributed_node_total(fake_system):
    """A distributed branch's node total scales with ranks-per-node, forcing chiq_aux on only at 2 co-located ranks."""
    config.memory.save_memory_for_chiq_aux = False
    one = _node_total("chiq_aux", "off", r=1)
    two = _node_total("chiq_aux", "off", r=2)
    assert two > one  # distributed block counted r times
    budget = 0.5 * (one + two)  # fits at 1 rank, overflows at 2
    avail = int(budget / 0.9)

    fake_system(avail)
    dgamore_main.autodetect_memory_settings(_mock_comm(size=1))
    assert config.memory.save_memory_for_chiq_aux is False

    config.memory.save_memory_for_chiq_aux = False
    fake_system(avail)
    dgamore_main.autodetect_memory_settings(_mock_comm(size=2, allgather=lambda obj: [obj, obj]))
    assert config.memory.save_memory_for_chiq_aux is True


def test_overflow_raises(fake_system):
    """A budget too small for even the lean path raises MemoryError."""
    fake_system(1024)  # 1 KiB free -> even the lean path cannot fit on the node
    with pytest.raises(MemoryError):
        dgamore_main.autodetect_memory_settings(_mock_comm())


def test_eliashberg_flags_considered_when_enabled(fake_system):
    """With Eliashberg enabled and ample memory, its lean flags stay off."""
    config.eliashberg.perform_eliashberg = True
    fake_system(64 * 1024**3)
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert config.memory.save_memory_for_fq is False
    assert config.memory.save_memory_for_lanczos is False


def test_eliashberg_flag_forced_on_under_pressure(fake_system):
    """Under memory pressure with Eliashberg enabled, the fq lean flag is forced on."""
    config.eliashberg.perform_eliashberg = True
    off = _node_total("fq", "off", r=1, with_eliashberg=True)
    max_on = max(_all_node_totals("on", r=1, with_eliashberg=True))
    budget = 0.5 * (max_on + off)
    fake_system(int(budget / 0.9))
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert config.memory.save_memory_for_fq is True


def test_autodetect_forwards_eliashberg_fq_flags(fake_system, monkeypatch):
    """The driver forwards save_fq and construct_fq_cheap from config.eliashberg into estimate_peaks."""
    config.eliashberg.perform_eliashberg = True
    monkeypatch.setattr(config.eliashberg, "save_fq", True, raising=False)
    monkeypatch.setattr(config.eliashberg, "construct_fq_cheap", True, raising=False)
    fake_system(64 * 1024**3)

    captured = {}
    real = dgamore_main.memory_estimator.estimate_peaks

    def spy(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", spy)
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert captured["save_fq"] is True
    assert captured["construct_fq_cheap"] is True

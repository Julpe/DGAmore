# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
"""Tests for DGAmore.autodetect_memory_settings (node-total budget wiring; the switches are autodetect-internal)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import dgamore.config as config
import dgamore.DGAmore as dgamore_main
from dgamore.memory_estimator import BranchPeak, estimate_peaks
from tests.conftest import create_comm_mock

# the q-grid / box parameters the fake_system fixture installs, so tests can reproduce the driver's estimate
# niv_cut == min(niw_core + niv_full + 10, niv_dmft) == min(32, 50) == 32 with the fixture's niv_dmft below.
FIXTURE_PARAMS = dict(n_bands=1, nk_tot=256, nk_irr=40, niw_core=10, niv_core=10, niv_full=12, niv_cut=32, niv_pp=5)


@pytest.fixture
def fake_system(monkeypatch):
    """Sets up a minimal single-band config and a controllable psutil/MPI environment."""
    config.sys.n_bands = FIXTURE_PARAMS["n_bands"]
    config.box.niw_core = FIXTURE_PARAMS["niw_core"]
    config.box.niv_core = FIXTURE_PARAMS["niv_core"]
    config.box.niv_full = FIXTURE_PARAMS["niv_full"]
    config.box.niv_dmft = 50  # so niv_cut = min(niw_core + niv_full + 10, niv_dmft) = min(32, 50) = 32
    config.eliashberg.perform_eliashberg = False
    config.lattice.k_grid = SimpleNamespace(nk_tot=FIXTURE_PARAMS["nk_tot"], nk_irr=FIXTURE_PARAMS["nk_irr"])
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
    """Reproduces the driver's node total incl. the shared-giwk credit for one branch/path on an r-rank node."""
    peaks = estimate_peaks(**FIXTURE_PARAMS, n_ranks=r, with_eliashberg=with_eliashberg)
    bp = peaks[branch]
    distributed = bp.off_distributed if which == "off" else bp.on_distributed
    single = bp.off_single if which == "off" else bp.on_single
    total = r * (bp.baseline + distributed) + single
    total -= (r - 1) * bp.giwk_shareable
    return total


def _all_node_totals(which, r, with_eliashberg):
    peaks = estimate_peaks(**FIXTURE_PARAMS, n_ranks=r, with_eliashberg=with_eliashberg)
    return [_node_total(k, which, r, with_eliashberg) for k in peaks]


def _mock_branch(
    baseline=0.0, giwk_shareable=0.0, off_distributed=0.0, off_single=0.0, on_distributed=0.0, on_single=0.0
):
    return BranchPeak(baseline, giwk_shareable, off_distributed, off_single, on_distributed, on_single)


def test_large_memory_passes_verification(fake_system):
    """A large free-memory budget on a tiny problem passes every branch verification without raising."""
    fake_system(64 * 1024**3)
    dgamore_main.autodetect_memory_settings(_mock_comm())


def test_chiq_aux_overflow_raises_while_lighter_branches_fit(fake_system):
    """A budget between the lighter branches and the chiq_aux node total raises on the heaviest branch."""
    chiq_aux = _node_total("chiq_aux", "off", r=1)
    floor = max(_node_total(k, "off", r=1) for k in ("chi0q", "sde", "local"))
    assert floor < chiq_aux  # sanity: chiq_aux is the heaviest branch here
    fake_system(int(0.5 * (floor + chiq_aux) / dgamore_main.NODE_MEMORY_FRACTION))
    with pytest.raises(MemoryError):
        dgamore_main.autodetect_memory_settings(_mock_comm())


def test_more_ranks_per_node_raise_the_distributed_node_total(fake_system):
    """A distributed branch's node total scales with ranks-per-node, overflowing only at 2 co-located ranks."""
    one = _node_total("chiq_aux", "off", r=1)
    two = _node_total("chiq_aux", "off", r=2)
    assert two > one  # distributed block counted r times (minus the one-copy giwk credit)
    budget = max(0.5 * (one + two), 1.01 * _node_total("local", "off", r=1))  # fits at 1 rank, overflows at 2
    avail = int(budget / dgamore_main.NODE_MEMORY_FRACTION)

    fake_system(avail)
    dgamore_main.autodetect_memory_settings(_mock_comm(size=1))

    fake_system(avail)
    with pytest.raises(MemoryError):
        dgamore_main.autodetect_memory_settings(_mock_comm(size=2, allgather=lambda obj: [obj, obj]))


def test_overflow_raises(fake_system):
    """A budget too small for even the lean path raises MemoryError."""
    fake_system(1024)  # 1 KiB free -> even the lean path cannot fit on the node
    with pytest.raises(MemoryError):
        dgamore_main.autodetect_memory_settings(_mock_comm())


def test_eliashberg_branches_verified_when_enabled(fake_system):
    """With Eliashberg enabled and ample memory, the verify-only branch checks pass without raising."""
    config.eliashberg.perform_eliashberg = True
    fake_system(64 * 1024**3)
    dgamore_main.autodetect_memory_settings(_mock_comm())


def test_autodetect_forwards_eliashberg_flags(fake_system, monkeypatch):
    """The driver forwards save_pairing_vertex and n_eig into estimate_peaks."""
    config.eliashberg.perform_eliashberg = True
    monkeypatch.setattr(config.eliashberg, "save_pairing_vertex", True, raising=False)
    monkeypatch.setattr(config.eliashberg, "n_eig", 3, raising=False)
    fake_system(64 * 1024**3)

    captured = {}
    real = dgamore_main.memory_estimator.estimate_peaks

    def spy(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", spy)
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert captured["save_pairing_vertex"] is True
    assert captured["n_eig"] == 3


def test_shared_giwk_credits_the_bubble_branch_node_total(fake_system, monkeypatch):
    """The always-on node-shared giwk window credits the deduplicated giwk to the chi0q node total."""
    fake_system(1)
    r = 4
    giwk_half = 1024.0**2
    baseline = 2.0 * giwk_half
    big = 50.0 * baseline
    tiny = _mock_branch(baseline=baseline, giwk_shareable=giwk_half)
    peaks = {
        "chi0q": _mock_branch(baseline=baseline, giwk_shareable=giwk_half, off_single=big),
        "chiq_aux": tiny,
        "sde": tiny,
    }
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: peaks)

    uncredited = r * baseline + big  # r*(baseline+off_distributed) + off_single, off_distributed == 0
    budget = uncredited - 0.5 * (r - 1) * giwk_half
    ranks = lambda obj: [obj] * r

    monkeypatch.setattr(
        dgamore_main.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=int(budget / dgamore_main.NODE_MEMORY_FRACTION)),
    )
    # the verify-only bubble raises without the credit at this budget, so completing is the assertion
    dgamore_main.autodetect_memory_settings(_mock_comm(size=r, allgather=ranks))


def test_shared_giwk_credits_a_non_bubble_sde_branch(fake_system, monkeypatch):
    """The giwk credit applies to every SDE-section branch, so the heavy chiq_aux fast path fits with sharing."""
    fake_system(1)
    r = 4
    giwk_half = 1024.0**2
    baseline = 2.0 * giwk_half
    big = 50.0 * baseline
    tiny = _mock_branch(baseline=baseline, giwk_shareable=giwk_half)
    peaks = {
        "chi0q": tiny,
        "chiq_aux": _mock_branch(baseline=baseline, giwk_shareable=giwk_half, off_distributed=big),
        "sde": tiny,
    }
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: peaks)

    uncredited = r * (baseline + big)  # off_single == 0
    budget = uncredited - 0.5 * (r - 1) * giwk_half
    ranks = lambda obj: [obj] * r

    monkeypatch.setattr(
        dgamore_main.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=int(budget / dgamore_main.NODE_MEMORY_FRACTION)),
    )
    # the verify-only branch raises without the credit at this budget, so completing is the assertion
    dgamore_main.autodetect_memory_settings(_mock_comm(size=r, allgather=ranks))


def test_eliashberg_branch_baseline_gets_no_giwk_credit(fake_system, monkeypatch):
    """A branch with giwk_shareable == 0 (private per-rank giwk_dga) gets no credit, so its estimate is not lowered."""
    fake_system(1)
    r = 4
    baseline = 2.0 * 1024.0**2
    peaks = {
        "chi0q": _mock_branch(baseline=baseline, giwk_shareable=baseline / 2, off_single=baseline),
        "fq": _mock_branch(baseline=baseline, off_distributed=baseline),
    }
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: peaks)

    budget = 4.5 * baseline  # credited chi0q fast 3.5 b fits; the uncredited fq verify (8 b) must fail
    monkeypatch.setattr(
        dgamore_main.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=int(budget / dgamore_main.NODE_MEMORY_FRACTION)),
    )
    with pytest.raises(MemoryError):
        dgamore_main.autodetect_memory_settings(_mock_comm(size=r, allgather=lambda obj: [obj] * r))


def test_non_solver_on_slots_are_ignored_by_verification(fake_system, monkeypatch):
    """Only the Eliashberg solver consults its on slots; an oversized on slot elsewhere must not raise."""
    fake_system(1)
    small, big = 1024.0**2, 100 * 1024.0**2
    peaks = {"chiq_aux": _mock_branch(baseline=small, off_distributed=small, on_distributed=big)}
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: peaks)
    monkeypatch.setattr(dgamore_main.psutil, "virtual_memory", lambda: SimpleNamespace(available=int(10 * small)))
    dgamore_main.autodetect_memory_settings(_mock_comm())


def test_flagless_sde_step_is_verified_and_raises_on_overflow(fake_system, monkeypatch):
    """The switch-less SDE FFT contraction is budget-checked: silent when it fits, raises the step on overflow."""
    fake_system(1)
    small, big = 1024.0**2, 100 * 1024.0**2
    peaks = {"sde": _mock_branch(baseline=small, off_distributed=big, on_distributed=big)}
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: peaks)

    monkeypatch.setattr(dgamore_main.psutil, "virtual_memory", lambda: SimpleNamespace(available=int(10 * big)))
    dgamore_main.autodetect_memory_settings(_mock_comm())  # fits -> no raise, no flag to set

    monkeypatch.setattr(dgamore_main.psutil, "virtual_memory", lambda: SimpleNamespace(available=int(10 * small)))
    with pytest.raises(MemoryError, match="Schwinger-Dyson"):
        dgamore_main.autodetect_memory_settings(_mock_comm())


def test_lanczos_single_rank_peak_doubled_on_single_node_multi_rank(fake_system, monkeypatch):
    """The doubled in-memory peak on a single node raises only when the grid fallback does not fit either."""
    fake_system(1)
    single = 10 * 1024.0**2
    budget = 1.5 * single  # fits one in-memory solver per node, not two concurrent ones
    monkeypatch.setattr(
        dgamore_main.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=int(budget / dgamore_main.NODE_MEMORY_FRACTION)),
    )

    both_too_big = {"lanczos": _mock_branch(off_single=single, on_distributed=single)}
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: both_too_big)
    with pytest.raises(MemoryError):
        dgamore_main.autodetect_memory_settings(_mock_comm(size=2, allgather=lambda obj: [obj, obj]))

    grid_rescues = {"lanczos": _mock_branch(off_single=single, on_distributed=0.1 * single)}
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: grid_rescues)
    dgamore_main.autodetect_memory_settings(_mock_comm(size=2, allgather=lambda obj: [obj, obj]))

    two_nodes = lambda obj: [("node0", obj[1]), ("node1", obj[1])]
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: both_too_big)
    dgamore_main.autodetect_memory_settings(_mock_comm(size=2, allgather=two_nodes))


def test_dgamore_excludes_osc_ucx_before_mpi_init():
    """DGAmore sets OMPI_MCA_osc to '^ucx' before importing mpi4py, muting the benign osc/ucx window warning."""
    src = open(dgamore_main.__file__, encoding="utf-8").read()
    assert 'os.environ.setdefault("OMPI_MCA_osc", "^ucx")' in src
    assert src.index('os.environ.setdefault("OMPI_MCA_osc", "^ucx")') < src.index("from mpi4py import MPI")


def test_local_step_overflow_raises_before_the_flag_loop(fake_system, monkeypatch):
    """A budget below the flag-less local single peak raises MemoryError naming the local Schwinger-Dyson step."""
    monkeypatch.setattr(config.box, "niv_full", 100)
    params = {**FIXTURE_PARAMS, "niv_full": 100, "niv_cut": 50}  # niv_cut = min(10 + 100 + 10, niv_dmft=50)
    peaks = estimate_peaks(**params, n_ranks=1, with_eliashberg=False)
    totals = {k: bp.baseline + bp.off_distributed + bp.off_single for k, bp in peaks.items()}
    local = totals.pop("local")
    assert max(totals.values()) < 0.9 * local
    fake_system(int(0.95 * local / dgamore_main.NODE_MEMORY_FRACTION))
    with pytest.raises(MemoryError, match="local Schwinger-Dyson"):
        dgamore_main.autodetect_memory_settings(_mock_comm())

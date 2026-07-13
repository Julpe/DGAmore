# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
"""Tests for DGAmore.autodetect_memory_settings (node-total budget + floor-semantics wiring)."""

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
    config.memory.save_memory_for_chi0q = False
    config.memory.save_memory_for_chiq_aux = False
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
    """Reproduces the driver's node total incl. the shared-giwk credit for one branch/path on an r-rank node."""
    peaks = estimate_peaks(**FIXTURE_PARAMS, n_ranks=r, with_eliashberg=with_eliashberg)
    bp = peaks[branch]
    distributed = bp.off_distributed if which == "off" else bp.on_distributed
    single = bp.off_single if which == "off" else bp.on_single
    total = r * (bp.baseline + distributed) + single
    if config.memory.use_shared_memory_common_obj:
        total -= (r - 1) * bp.giwk_shareable
    return total


def _all_node_totals(which, r, with_eliashberg):
    peaks = estimate_peaks(**FIXTURE_PARAMS, n_ranks=r, with_eliashberg=with_eliashberg)
    return [_node_total(k, which, r, with_eliashberg) for k in peaks]


def _mock_branch(
    baseline=0.0, giwk_shareable=0.0, off_distributed=0.0, off_single=0.0, on_distributed=0.0, on_single=0.0
):
    return BranchPeak(baseline, giwk_shareable, off_distributed, off_single, on_distributed, on_single)


def test_large_memory_keeps_all_flags_off(fake_system):
    """A large free-memory budget on a tiny problem leaves all lean flags off."""
    fake_system(64 * 1024**3)  # 64 GiB free, tiny problem -> nothing forced on
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert config.memory.save_memory_for_chiq_aux is False
    assert config.memory.save_memory_for_chi0q is False


def test_tiny_memory_forces_lean_flags_on(fake_system):
    """A budget below chiq_aux's fast-path node total forces its lean flag on without overflowing the node (the
    budget floor is the flag-less local step, which must always fit)."""
    off = _node_total("chiq_aux", "off", r=1)
    max_on = max(_all_node_totals("on", r=1, with_eliashberg=False))
    budget = max(0.5 * (max_on + off), 1.01 * _node_total("local", "off", r=1))
    fake_system(int(budget / dgamore_main.NODE_MEMORY_FRACTION))
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert config.memory.save_memory_for_chiq_aux is True


def test_user_true_is_preserved_as_floor(fake_system):
    """A user-set lean flag survives autodetection even when memory is plentiful."""
    fake_system(64 * 1024**3)  # plenty of memory; auto would leave it off
    config.memory.save_memory_for_chiq_aux = True
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert config.memory.save_memory_for_chiq_aux is True  # user setting survives


def test_chi0q_single_rank_peak_fits_under_node_total(fake_system):
    """chi0q's single-rank transient peak stays off against the node total while the heavier chiq_aux branch is
    forced on (the budget floor is the flag-less sde step, which must always fit)."""
    chi0q_off = _node_total("chi0q", "off", r=1)
    chiq_aux_off = _node_total("chiq_aux", "off", r=1)
    floor = max(chi0q_off, _node_total("sde", "off", r=1), _node_total("local", "off", r=1))
    assert floor < chiq_aux_off  # sanity: chiq_aux is the heaviest of the three here
    budget = 0.5 * (floor + chiq_aux_off)
    fake_system(int(budget / dgamore_main.NODE_MEMORY_FRACTION))
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert config.memory.save_memory_for_chi0q is False  # single-rank peak fits the node
    assert config.memory.save_memory_for_chiq_aux is True  # heavier branch still forced on


def test_more_ranks_per_node_raise_the_distributed_node_total(fake_system):
    """A distributed branch's node total scales with ranks-per-node, forcing chiq_aux on only at 2 co-located ranks."""
    config.memory.save_memory_for_chiq_aux = False
    one = _node_total("chiq_aux", "off", r=1)
    two = _node_total("chiq_aux", "off", r=2)
    assert two > one  # distributed block counted r times (minus the one-copy giwk credit)
    budget = max(0.5 * (one + two), 1.01 * _node_total("local", "off", r=1))  # fits at 1 rank, overflows at 2
    avail = int(budget / dgamore_main.NODE_MEMORY_FRACTION)

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
    fake_system(int(budget / dgamore_main.NODE_MEMORY_FRACTION))
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert config.memory.save_memory_for_fq is True


def test_autodetect_forwards_eliashberg_flags(fake_system, monkeypatch):
    """The driver forwards save_fq, construct_fq_cheap, save_pairing_vertex and n_eig into estimate_peaks."""
    config.eliashberg.perform_eliashberg = True
    monkeypatch.setattr(config.eliashberg, "save_fq", True, raising=False)
    monkeypatch.setattr(config.eliashberg, "construct_fq_cheap", True, raising=False)
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
    assert captured["save_fq"] is True
    assert captured["construct_fq_cheap"] is True
    assert captured["save_pairing_vertex"] is True
    assert captured["n_eig"] == 3


def test_memory_config_shares_giwk_by_default():
    """The node-shared giwk optimization is enabled by default (disable-able for the NUMA case)."""
    assert config.MemoryConfig().use_shared_memory_common_obj is True


def test_shared_giwk_credits_the_bubble_branch_node_total(fake_system, monkeypatch):
    """use_shared_memory_common_obj subtracts the deduplicated giwk copies from the chi0q (bubble) node total, so the fast
    FFT bubble fits at a budget where the replicated estimate would force the lean path on."""
    fake_system(1)  # installs the fixture config; the real budget is set per-branch below
    r = 4
    giwk_half = 1024.0**2
    baseline = 2.0 * giwk_half
    big = 50.0 * baseline  # chi0q fast is a single-rank transient and the binding branch; its lean path is tiny
    tiny = _mock_branch(baseline=baseline, giwk_shareable=giwk_half)
    peaks = {
        "chi0q": _mock_branch(baseline=baseline, giwk_shareable=giwk_half, off_single=big),
        "chiq_aux": tiny,
        "sde": tiny,
    }
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: peaks)

    uncredited = r * baseline + big  # r*(baseline+off_distributed) + off_single, off_distributed == 0
    budget = uncredited - 0.5 * (r - 1) * giwk_half  # between the credited (r->1 giwk) total and the uncredited one
    ranks = lambda obj: [obj] * r

    config.memory.use_shared_memory_common_obj = True
    config.memory.save_memory_for_chi0q = False
    monkeypatch.setattr(
        dgamore_main.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=int(budget / dgamore_main.NODE_MEMORY_FRACTION)),
    )
    dgamore_main.autodetect_memory_settings(_mock_comm(size=r, allgather=ranks))
    assert config.memory.save_memory_for_chi0q is False  # credited fast bubble fits

    config.memory.use_shared_memory_common_obj = False
    config.memory.save_memory_for_chi0q = False
    dgamore_main.autodetect_memory_settings(_mock_comm(size=r, allgather=ranks))
    assert config.memory.save_memory_for_chi0q is True  # uncredited fast bubble overflows -> lean forced on


def test_shared_giwk_credits_a_non_bubble_sde_branch(fake_system, monkeypatch):
    """The giwk credit applies to every SDE-section branch, not just the bubble: the heavy chiq_aux fast path fits
    with sharing where the replicated estimate would force it onto the lean per-q path."""
    fake_system(1)
    r = 4
    giwk_half = 1024.0**2
    baseline = 2.0 * giwk_half
    big = 50.0 * baseline  # chiq_aux fast is the binding branch (distributed); its lean path is trivially small here
    tiny = _mock_branch(baseline=baseline, giwk_shareable=giwk_half)
    peaks = {
        "chi0q": tiny,
        "chiq_aux": _mock_branch(baseline=baseline, giwk_shareable=giwk_half, off_distributed=big),
        "sde": tiny,
    }
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: peaks)

    uncredited = r * (baseline + big)  # off_single == 0
    budget = uncredited - 0.5 * (r - 1) * giwk_half  # between the credited (giwk r->1) and the uncredited total
    ranks = lambda obj: [obj] * r

    config.memory.use_shared_memory_common_obj = True
    config.memory.save_memory_for_chiq_aux = False
    monkeypatch.setattr(
        dgamore_main.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=int(budget / dgamore_main.NODE_MEMORY_FRACTION)),
    )
    dgamore_main.autodetect_memory_settings(_mock_comm(size=r, allgather=ranks))
    assert config.memory.save_memory_for_chiq_aux is False  # credited fast path fits

    config.memory.use_shared_memory_common_obj = False
    config.memory.save_memory_for_chiq_aux = False
    dgamore_main.autodetect_memory_settings(_mock_comm(size=r, allgather=ranks))
    assert config.memory.save_memory_for_chiq_aux is True  # uncredited fast path overflows -> lean forced on


def test_eliashberg_branch_baseline_gets_no_giwk_credit(fake_system, monkeypatch):
    """A branch with giwk_shareable == 0 (the eliashberg ones run on the private per-rank giwk_dga) is not credited."""
    fake_system(1)
    r = 4
    baseline = 2.0 * 1024.0**2
    peaks = {
        "chi0q": _mock_branch(baseline=baseline, giwk_shareable=baseline / 2, off_single=baseline),
        "fq": _mock_branch(baseline=baseline, off_distributed=baseline),
    }
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: peaks)

    # budget between the credited chi0q fast total (3.5 b) / the fq lean total (4 b) and the fq fast total (8 b);
    # the uncredited chi0q fast total (5 b) would also overflow, so chi0q staying off shows the credit was applied.
    budget = 4.5 * baseline
    config.memory.use_shared_memory_common_obj = True
    config.memory.save_memory_for_chi0q = False
    config.memory.save_memory_for_fq = False
    monkeypatch.setattr(
        dgamore_main.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=int(budget / dgamore_main.NODE_MEMORY_FRACTION)),
    )
    dgamore_main.autodetect_memory_settings(_mock_comm(size=r, allgather=lambda obj: [obj] * r))
    assert config.memory.save_memory_for_chi0q is False  # credited -> fits
    assert config.memory.save_memory_for_fq is True  # uncredited baseline + transient overflows -> lean forced on


def test_heavier_lean_path_does_not_raise_when_fast_fits(fake_system, monkeypatch):
    """A branch whose lean path peaks higher than its fast one must not raise as long as the fast path fits and the
    user did not force the lean flag."""
    fake_system(1)
    small, big = 1024.0**2, 100 * 1024.0**2
    peaks = {"chiq_aux": _mock_branch(baseline=small, off_distributed=small, on_distributed=big)}
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: peaks)
    monkeypatch.setattr(dgamore_main.psutil, "virtual_memory", lambda: SimpleNamespace(available=int(10 * small)))

    config.memory.save_memory_for_chiq_aux = False
    dgamore_main.autodetect_memory_settings(_mock_comm())
    assert config.memory.save_memory_for_chiq_aux is False  # fast path fits -> no raise despite the oversized lean path


def test_user_forced_lean_path_that_overflows_raises(fake_system, monkeypatch):
    """A user-forced lean flag whose path overflows the node raises even though the fast path would fit."""
    fake_system(1)
    small, big = 1024.0**2, 100 * 1024.0**2
    peaks = {"chiq_aux": _mock_branch(baseline=small, off_distributed=small, on_distributed=big)}
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: peaks)
    monkeypatch.setattr(dgamore_main.psutil, "virtual_memory", lambda: SimpleNamespace(available=int(10 * small)))

    config.memory.save_memory_for_chiq_aux = True
    with pytest.raises(MemoryError, match="fast path would fit"):
        dgamore_main.autodetect_memory_settings(_mock_comm())


def test_flagless_sde_step_is_verified_and_raises_on_overflow(fake_system, monkeypatch):
    """The switch-less Schwinger-Dyson FFT contraction is still budget-checked: it passes silently when it fits and
    raises a MemoryError naming the step when it overflows the node."""
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
    """On a single-node multi-rank job the singlet and triplet solves run concurrently on the same node, so the
    lanczos fast-path single-rank peak is doubled; spread over two nodes each carries only one solver."""
    fake_system(1)
    single = 10 * 1024.0**2
    peaks = {"lanczos": _mock_branch(baseline=1.0, off_single=single, on_distributed=1.0)}
    monkeypatch.setattr(dgamore_main.memory_estimator, "estimate_peaks", lambda **kw: peaks)
    budget = 1.5 * single  # fits one solver per node, not two
    monkeypatch.setattr(
        dgamore_main.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=int(budget / dgamore_main.NODE_MEMORY_FRACTION)),
    )

    config.memory.save_memory_for_lanczos = False
    dgamore_main.autodetect_memory_settings(_mock_comm(size=2, allgather=lambda obj: [obj, obj]))
    assert config.memory.save_memory_for_lanczos is True  # both solvers on one node -> doubled peak overflows

    config.memory.save_memory_for_lanczos = False
    two_nodes = lambda obj: [("node0", obj[1]), ("node1", obj[1])]
    dgamore_main.autodetect_memory_settings(_mock_comm(size=2, allgather=two_nodes))
    assert config.memory.save_memory_for_lanczos is False  # one solver per node -> fits


def test_dgamore_excludes_osc_ucx_before_mpi_init():
    """DGAmore defaults OMPI_MCA_osc to '^ucx' before importing mpi4py, muting the benign osc/ucx shared-window warning
    with no user configuration; the ordering (set before MPI_Init) is what makes it take effect."""
    src = open(dgamore_main.__file__, encoding="utf-8").read()
    assert 'os.environ.setdefault("OMPI_MCA_osc", "^ucx")' in src
    assert src.index('os.environ.setdefault("OMPI_MCA_osc", "^ucx")') < src.index("from mpi4py import MPI")


def test_local_step_overflow_raises_before_the_flag_loop(fake_system, monkeypatch):
    """A budget below the flag-less local single peak raises MemoryError naming the local Schwinger-Dyson step
    (a large shell box makes the rank-0 local step the dominant requirement)."""
    monkeypatch.setattr(config.box, "niv_full", 100)
    params = {**FIXTURE_PARAMS, "niv_full": 100, "niv_cut": 50}  # niv_cut = min(10 + 100 + 10, niv_dmft=50)
    peaks = estimate_peaks(**params, n_ranks=1, with_eliashberg=False)
    totals = {k: bp.baseline + bp.off_distributed + bp.off_single for k, bp in peaks.items()}
    local = totals.pop("local")
    assert max(totals.values()) < 0.9 * local  # sanity: every other requirement fits at this budget
    fake_system(int(0.95 * local / dgamore_main.NODE_MEMORY_FRACTION))
    with pytest.raises(MemoryError, match="local Schwinger-Dyson"):
        dgamore_main.autodetect_memory_settings(_mock_comm())

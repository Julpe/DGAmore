# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
"""Unit tests for the pure peak-memory estimator (no MPI, no psutil)."""

from dataclasses import replace

import numpy as np
import pytest

import dgamore.bubble_gen as bubble_gen
import dgamore.config as config
import dgamore.memory_estimator as memory_estimator
import dgamore.nonlocal_sde as nonlocal_sde
from dgamore.four_point import FourPoint
from dgamore.interaction import LocalInteraction
from dgamore.jacobian_stabilization import TRACKER_PAIRS
from dgamore.local_four_point import LocalFourPoint
from dgamore.memory_estimator import (
    ARPACK_EXTRA_VECTORS,
    CHI0Q_IFFTN_TRANSIENT_FACTOR,
    DTYPE_BYTES,
    EXACT_CHECK_BASIS,
    EXACT_JACOBIAN_MODES,
    EXACT_JACOBIAN_NCV,
    KERNEL_GROUP_TRANSIENT,
    KERNEL_MOMENTUM_GROUPS,
    LANCZOS_VERTEX_FACTOR,
    MAX_CHUNK_BUDGET_BYTES,
    MAX_SLICE_CHUNK_BYTES,
    OVERHEAD_FACTOR,
    RANK_BASELINE_BYTES,
    SDE_COLUMN_FACTOR,
    SDE_CONTRACTION_CHUNK_BYTES,
    SDE_W_BLOCK,
    SLICE_CHUNK_BYTES,
    BranchPeak,
    ChunkBudgets,
    _giwk_rspace,
    column_sde_schedule,
    column_sde_slabs,
    column_sde_stream_bytes,
    dynamic_chunk_budget,
    estimate_peaks,
    exact_jacobian_budgets,
    exact_window_stages,
    interaction_bytes,
    jacobian_tracker_bytes,
    max_chunk_budget,
    node_total,
    rank_tasks,
)
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import DTYPE, SpinChannel
from tests.conftest import FAKE_MPI, traced_peak

BASE = dict(
    n_bands=1,
    nk_tot=16 * 16,
    nk_irr=45,
    niw_core=30,
    niv_core=30,
    niv_full=40,
    niv_cut=80,  # min(niw_core + niv_full + 10, niv_dmft) == 80 here
    niv_dmft=120,
    niv_pp=15,
    n_ranks=4,
    with_eliashberg=False,
)

TINY = dict(
    n_bands=2,
    nk_tot=80,
    nk_irr=20,
    niw_core=4,
    niv_core=5,
    niv_full=6,
    niv_cut=15,
    niv_dmft=20,
    niv_pp=2,
    n_ranks=4,
    with_eliashberg=False,
)

SCALE = DTYPE_BYTES * OVERHEAD_FACTOR


def _peaks(**overrides):
    return estimate_peaks(**{**BASE, **overrides})


def _v_single(params):
    return SCALE * params["nk_tot"] * params["n_bands"] ** 4


# the node total of r equal ranks on rank 0's node: each holds the baseline + the distributed transient, plus one single
def _off_node_total(bp: BranchPeak, r):
    return r * (bp.baseline + bp.off_distributed) + bp.off_single


def test_constants():
    """DTYPE_BYTES tracks the global storage dtype and OVERHEAD_FACTOR defaults to 1.0 (no extra margin)."""
    assert DTYPE_BYTES == np.dtype(DTYPE).itemsize  # single source of truth: derived from the global storage dtype
    assert OVERHEAD_FACTOR == pytest.approx(1.0)


def test_keys_without_eliashberg():
    """Without Eliashberg the estimator reports the chi0q, chiq_aux, sde, sigma_loop, mu_update, energies, occupation
    and local branches."""
    keys = {"chi0q", "chiq_aux", "sde", "sigma_loop", "mu_update", "energies", "occupation", "local"}
    assert set(_peaks(with_eliashberg=False)) == keys


def test_keys_with_eliashberg():
    """With Eliashberg the estimator adds the fq and lanczos branches."""
    branches = {
        *("chi0q", "chiq_aux", "sde", "sigma_loop", "mu_update", "energies", "occupation", "fq", "lanczos", "local")
    }
    assert set(_peaks(with_eliashberg=True)) == branches


def test_every_branch_has_positive_baseline_and_off_transient():
    """The SDE-section branches and fq carry a baseline beyond the rank footprint, the others none; all a transient."""
    peaks = _peaks(with_eliashberg=True)
    for key, bp in peaks.items():
        assert isinstance(bp, BranchPeak)
        base = RANK_BASELINE_BYTES  # every rank's footprint sits in every baseline, the interaction in rank 0's slots
        assert bp.baseline > base if key in ("chi0q", "chiq_aux", "sde", "fq") else bp.baseline == base
        assert bp.off_distributed + bp.off_single > 0


def test_chi0q_fast_path_is_distributed_on_multi_rank_runs():
    """The chi0q fast path is column-distributed on multi-rank runs; only single-rank uses the rank-0 build."""
    multi = _peaks()["chi0q"]
    assert multi.off_distributed > 0.0 and multi.off_single == _v_single(BASE)
    single = _peaks(n_ranks=1)["chi0q"]
    assert single.off_distributed == 0.0 and single.off_single > _v_single(BASE)
    assert multi.on_distributed > 0.0 and multi.on_single == _v_single(BASE)


def test_chiq_aux_off_has_distributed_block_and_no_single_rank_gather():
    """The chiq_aux fast path holds a per-rank block; its single-rank slot is rank 0's full-grid interaction alone."""
    bp = _peaks()["chiq_aux"]
    assert bp.off_distributed > 0.0  # per-rank two-fermion block
    assert bp.off_single == _v_single(BASE)


def test_chi0q_distributed_peak_shrinks_with_more_ranks():
    """The column-distributed chi0q fast-path transient shrinks as the rank count grows."""
    assert _peaks(n_ranks=16)["chi0q"].off_distributed < _peaks(n_ranks=2)["chi0q"].off_distributed


def test_chiq_aux_distributed_block_shrinks_with_more_ranks():
    """The chiq_aux distributed block shrinks as the rank count grows."""
    assert _peaks(n_ranks=16)["chiq_aux"].off_distributed < _peaks(n_ranks=2)["chiq_aux"].off_distributed


def test_lanczos_single_rank_independent_of_rank_count_beyond_one():
    """The lanczos single-rank peak is rank-count-independent for multi-rank runs."""
    few = _peaks(n_ranks=2, with_eliashberg=True)["lanczos"].off_single
    many = _peaks(n_ranks=8, with_eliashberg=True)["lanczos"].off_single
    assert few == pytest.approx(many)


def test_lanczos_single_rank_run_adds_waiting_channel_vertex():
    """A single-rank run solves the channels sequentially and holds the waiting channel's gathered irr-BZ vertex."""
    p = {**BASE, "with_eliashberg": True}
    extra = SCALE * p["nk_irr"] * p["n_bands"] ** 4 * (2 * p["niv_pp"]) ** 2
    assert _peaks(n_ranks=1, with_eliashberg=True)["lanczos"].off_single == pytest.approx(
        _peaks(n_ranks=2, with_eliashberg=True)["lanczos"].off_single + extra
    )


def test_two_fermion_branches_dominate_node_total():
    """The two-fermion branches (chiq_aux, fq) dominate the per-node memory total."""
    peaks = _peaks(with_eliashberg=True)
    r = BASE["n_ranks"]
    totals = {k: _off_node_total(bp, r) for k, bp in peaks.items()}
    assert totals["chiq_aux"] > totals["chi0q"]
    assert totals["chiq_aux"] > totals["sde"]
    assert totals["fq"] > totals["sde"]


def test_node_total_monotonic_in_n_bands():
    """The node total grows with the number of bands."""
    r = BASE["n_ranks"]
    assert _off_node_total(_peaks(n_bands=2)["chiq_aux"], r) > _off_node_total(_peaks(n_bands=1)["chiq_aux"], r)


def test_overhead_scales_everything_linearly():
    """The overhead factor scales the baseline and every branch linearly."""
    peaks1 = estimate_peaks(**BASE, overhead=1.0)
    peaks2 = estimate_peaks(**BASE, overhead=2.0)
    assert peaks2["chiq_aux"].baseline == pytest.approx(2.0 * peaks1["chiq_aux"].baseline)
    assert peaks2["chiq_aux"].off_distributed == pytest.approx(2.0 * peaks1["chiq_aux"].off_distributed)


def test_fq_distributed_block_carries_the_pp_accumulator_beyond_chiq_aux():
    """The fq distributed transient exceeds chiq_aux's by the pp accumulator and the extra loaded inputs."""
    peaks = _peaks(with_eliashberg=True)
    assert peaks["fq"].on_distributed > peaks["chiq_aux"].on_distributed


def test_bubble_baseline_is_giwk_plus_sigma_old_at_niv_cut():
    """The chi0q baseline is giwk plus sigma_old at niv_cut, plus (multi-rank) the two node-shareable R-space Gs."""
    two_point = SCALE * 2 * (TINY["nk_tot"] * TINY["n_bands"] ** 2 * (2 * TINY["niv_cut"]))
    g_r_windows = SCALE * 2 * TINY["nk_tot"] * TINY["n_bands"] ** 2 * (2 * (TINY["niv_full"] + TINY["niw_core"]))
    base = RANK_BASELINE_BYTES
    assert estimate_peaks(**TINY)["chi0q"].baseline == pytest.approx(two_point + g_r_windows + base)
    assert estimate_peaks(**{**TINY, "n_ranks": 1})["chi0q"].baseline == pytest.approx(two_point + base)


def test_sde_section_baseline_uses_post_bubble_windows():
    """The chiq_aux baseline holds giwk at niv_core+niw_core, the loop Sigma at niv_cut and the local vertex."""
    nk, nb = TINY["nk_tot"], TINY["n_bands"]
    giwk = nk * nb**2 * 2 * (TINY["niv_core"] + TINY["niw_core"])
    sigma = nk * nb**2 * 2 * TINY["niv_cut"]
    peaks = estimate_peaks(**TINY)
    vf, vc = 2 * TINY["niv_full"], 2 * TINY["niv_core"]
    local_vertex = TINY["n_bands"] ** 4 * (TINY["niw_core"] + 1) * max(vf * vc, vc * vc)
    base = RANK_BASELINE_BYTES
    assert peaks["chiq_aux"].baseline == pytest.approx(SCALE * (giwk + sigma + local_vertex) + base)
    assert peaks["sde"].baseline == pytest.approx(SCALE * (2 * giwk + sigma) + base)
    assert peaks["chiq_aux"].giwk_shareable == pytest.approx(SCALE * (giwk + sigma + local_vertex))
    assert peaks["sde"].giwk_shareable == pytest.approx(SCALE * (2 * giwk + sigma))


def test_giwk_shareable_is_every_array_of_each_sde_section_baseline():
    """The loop Sigma is node-shared like the Green's functions, so only the per-rank footprint stays unshared."""
    peaks = _peaks(with_eliashberg=True)
    for key in ("chi0q", "chiq_aux", "sde"):
        assert peaks[key].giwk_shareable == pytest.approx(peaks[key].baseline - RANK_BASELINE_BYTES)


def test_mixing_history_sits_in_the_rank0_slot_of_every_proposal_branch():
    """Each mixing pair adds two core-box Sigma copies to every proposal branch's single slot, none after the loop."""
    pairs = 2 * BASE["nk_tot"] * BASE["n_bands"] ** 2 * 2 * BASE["niv_core"]
    linear, anderson = _peaks(niv_interp=40), _peaks(mixing_pairs=4, niv_interp=40)
    for key in ("chi0q", "chiq_aux", "sde"):
        assert anderson[key].off_single - linear[key].off_single == pytest.approx(SCALE * 4 * pairs)
        assert anderson[key].off_distributed == linear[key].off_distributed
    assert anderson["sigma_interp"].off_single == linear["sigma_interp"].off_single


def test_eliashberg_branches_share_only_the_local_vertex_of_the_pairing_vertex_build():
    """Only fq's local vertex is node-shared in the Eliashberg branches (sigma_dga freed, giwk_dga on rank 0)."""
    peaks = _peaks(with_eliashberg=True)
    giwk_dga = SCALE * BASE["nk_tot"] * BASE["n_bands"] ** 2 * 2 * BASE["niv_cut"]
    local_vertex = SCALE * BASE["n_bands"] ** 4 * (BASE["niw_core"] + 1) * (2 * BASE["niv_core"]) ** 2
    for key, shared in (("fq", local_vertex), ("lanczos", 0.0)):
        assert peaks[key].giwk_shareable == pytest.approx(shared)
        assert peaks[key].baseline == pytest.approx(RANK_BASELINE_BYTES + shared)
        assert peaks[key].off_single >= giwk_dga
        assert peaks[key].on_single >= giwk_dga


def test_bubble_baseline_depends_on_niv_cut_not_niv_full():
    """The single-rank chi0q baseline tracks niv_cut only; the multi-rank one also tracks niv_full via the R-space G."""
    assert _peaks(niv_full=40, n_ranks=1)["chi0q"].baseline == pytest.approx(
        _peaks(niv_full=400, n_ranks=1)["chi0q"].baseline
    )
    assert _peaks(niv_full=40)["chi0q"].baseline < _peaks(niv_full=400)["chi0q"].baseline
    assert _peaks(niv_cut=80)["chi0q"].baseline != pytest.approx(_peaks(niv_cut=800)["chi0q"].baseline)


def _chiq_aux_transient():
    """The modeled aux-chi transient: two compound slices, 128 + 4 n_bands^2 of their columns and two 256^2 tiles."""
    nb = TINY["n_bands"]
    n = nb**2 * 2 * TINY["niv_core"]
    return DTYPE_BYTES * (2 * n * n + (128 + 4 * nb**2) * n + 2 * 256**2)


def _fq_branch(budget, streaming=False):
    """The TINY fq branch: accumulator, loaded and group-copied bubbles, 3 windows + 6 one-fermion + 4 (9) slices."""
    nb, wp, vc, vpp = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_core"], 2 * TINY["niv_pp"]
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    nv, nw = (vc, wp) if streaming else (vpp, vpp)
    window_slice = DTYPE_BYTES * nb**4 * nv * nv
    chunk = min(max(budget, window_slice), qi * nw * window_slice)
    group = max(1, chunk // (nw * window_slice))
    residents = SCALE * (qi * nb**4 * vpp * vpp + 2 * (qi + group) * nb**4 * wp * vc)
    slices = (9 if streaming else 4) * DTYPE_BYTES * nb**4 * vc * vc
    return residents + OVERHEAD_FACTOR * (3 * chunk + 6 * (chunk * vc // nv**2) + slices), window_slice, qi * nw


def test_chiq_aux_block_is_the_largest_kernel_section_stage():
    """The chiq_aux transient is the largest of the cut, the attachment, the aux-sum solve and the grouped stages."""
    nb, wp, vc, vf = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_core"], 2 * TINY["niv_full"]
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    core, group = qi * nb**4 * wp * vc, -(-qi // KERNEL_MOMENTUM_GROUPS) * nb**4 * wp * vc
    expected = max(
        SCALE * (qi * nb**4 * wp * vf + 2 * core),
        SCALE * 4 * core,
        SCALE * 3 * core + OVERHEAD_FACTOR * _chiq_aux_transient(),
        SCALE * (3 * core + KERNEL_GROUP_TRANSIENT * group),
    )
    bp = estimate_peaks(**TINY)["chiq_aux"]
    assert bp.off_distributed == pytest.approx(expected)
    assert bp.on_distributed == bp.off_distributed


def test_chiq_aux_transient_takes_no_chunk_budget():
    """No budget moves the chiq_aux branch: the per-slice build holds the same two slices at any setting."""
    floor = estimate_peaks(**TINY)["chiq_aux"]
    for budgets in (ChunkBudgets(0, 0), ChunkBudgets(2**30, 2**30), ChunkBudgets(MAX_CHUNK_BUDGET_BYTES)):
        assert estimate_peaks(**TINY, chunk_budgets=budgets)["chiq_aux"] == floor


def test_fq_chunk_term_follows_the_passed_budget_up_to_the_rank_block():
    """The pairing-vertex transient follows its pp-box window chunk and momentum group, capped at the rank block."""
    params = {**TINY, "with_eliashberg": True}
    _, window_slice, block_slices = _fq_branch(0)
    for budget in (0, 3 * window_slice, block_slices * window_slice // 2, block_slices * window_slice):
        bp = estimate_peaks(**params, chunk_budgets=ChunkBudgets(fq=budget))["fq"]
        assert bp.off_distributed == pytest.approx(_fq_branch(budget)[0])
    capped = estimate_peaks(**params, chunk_budgets=ChunkBudgets(fq=MAX_CHUNK_BUDGET_BYTES))["fq"]
    assert capped.off_distributed == pytest.approx(_fq_branch(block_slices * window_slice)[0])


def test_streamed_full_vertex_models_the_whole_slice_inverse():
    """With save_fq the chunk is a core-box window and carries numpy's whole-slice inverse, beyond the band's."""
    _, window_slice, block_slices = _fq_branch(0, streaming=True)
    for budget in (0, block_slices * window_slice):
        params = {**TINY, "with_eliashberg": True, "chunk_budgets": ChunkBudgets(fq=budget)}
        band, streamed = (estimate_peaks(**params, save_fq=flag)["fq"].off_distributed for flag in (False, True))
        assert streamed == pytest.approx(_fq_branch(budget, streaming=True)[0]) and streamed > band


def test_max_chunk_budget_bisects_to_the_largest_fitting_budget():
    """The budget search returns the largest budget a monotone fit predicate accepts, to within 1 MiB."""
    limit = 3 * 2**30 + 12345
    found = max_chunk_budget(lambda budget: budget <= limit)
    assert limit - 2**20 <= found <= limit
    assert max_chunk_budget(lambda budget: True) == MAX_CHUNK_BUDGET_BYTES
    assert max_chunk_budget(lambda budget: False) == SLICE_CHUNK_BYTES
    assert max_chunk_budget(lambda budget: budget <= SLICE_CHUNK_BYTES) == SLICE_CHUNK_BYTES
    assert max_chunk_budget(lambda budget: budget <= 2**31, upper=2**30) == 2**30
    assert ChunkBudgets() == ChunkBudgets(SLICE_CHUNK_BYTES, SLICE_CHUNK_BYTES)
    assert RANK_BASELINE_BYTES > 0


def test_chi0q_fast_single_counts_buffer_ifftn_transient_and_g_copies():
    """The single-rank chi0q fast peak counts the multiply buffer, the ~2x ifftn transient and three G copies."""
    nb, wp, vf = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_full"]
    bubble_irr = TINY["nk_irr"] * nb**4 * wp * vf
    fft_buffers = (1 + CHI0Q_IFFTN_TRANSIENT_FACTOR) * TINY["nk_tot"] * nb**4 * vf
    gf_copies = 2 * TINY["nk_tot"] * nb**2 * (2 * (TINY["niv_full"] + TINY["niw_core"]))
    g_center = TINY["nk_tot"] * nb**2 * vf
    expected = SCALE * (bubble_irr + fft_buffers + gf_copies + g_center) + _v_single(TINY)
    assert estimate_peaks(**{**TINY, "n_ranks": 1})["chi0q"].off_single == pytest.approx(expected)


def test_chi0q_fast_distributed_is_bounded_by_the_result_slice():
    """The multi-rank chi0q fast peak is 1.5 times the per-rank irr result slice."""
    nb, wp, vf = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_full"]
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    expected = SCALE * 1.5 * qi * nb**4 * wp * vf
    assert estimate_peaks(**TINY)["chi0q"].off_distributed == pytest.approx(expected)


def test_column_sde_schedule_splits_the_blocks_of_each_frequency_into_contiguous_rank_runs():
    """Blocks cover w = 0..niw then w = -1..-niw in runs of 4; rank runs are contiguous; owners hold block 0."""
    blocks, bounds, owner = column_sde_schedule(niw=9, niv=3, n_ranks=4)
    positive, negative = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9)], [(1, 2, 3, 4), (5, 6, 7, 8), (9,)]
    assert blocks == [(False, b) for b in positive] + [(True, b) for b in negative] and SDE_W_BLOCK == 4
    assert bounds[0] == 0 and bounds[-1] == 3 * 6 and np.all(np.diff(bounds) >= 0)
    assert np.diff(bounds).max() - np.diff(bounds).min() <= 1
    for v in range(3):
        assert bounds[owner[v]] <= v * 6 < bounds[owner[v] + 1]
    _, many_bounds, many_owner = column_sde_schedule(niw=1, niv=1, n_ranks=5)
    assert many_bounds[-1] == 2 and np.diff(many_bounds).sum() == 2 and many_bounds[many_owner[0] + 1] > 0


def test_column_sde_slabs_counts_owned_results_a_foreign_sum_its_receive_buffer_and_the_task_buffers():
    """Rank 0 holds its owned sum, two task buffers and a receive for the middle rank's sum; one rank receives none."""
    assert column_sde_slabs(niw=2, niv=2, n_ranks=3) == 1 + 2 + 1
    assert column_sde_slabs(niw=2, niv=2, n_ranks=1) == 2 + 2


def test_column_sde_schedule_by_frequency_runs_single_frequency_blocks_in_ascending_order():
    """Blocks hold one w each, w before -w and none for -0; the split and the owners follow the task count."""
    blocks, bounds, owner = column_sde_schedule(niw=3, niv=2, n_ranks=3, by_frequency=True)
    singles = [(False, (0,))] + [(negative, (w,)) for w in (1, 2, 3) for negative in (False, True)]
    assert blocks == singles
    assert np.array_equal(bounds, [0, 4, 9, 14]) and np.array_equal(owner, [0, 1])
    assert column_sde_schedule(niw=0, niv=1, n_ranks=2, by_frequency=True)[0] == [(False, (0,))]


def test_column_sde_slabs_counts_every_output_and_the_carried_slabs():
    """Two outputs double the results and running sums and add a partial; carried slabs leave out the work buffers."""
    assert column_sde_slabs(niw=2, niv=2, n_ranks=3, n_out=2, by_frequency=True) == 2 * 2 + 3 + 1
    assert column_sde_slabs(niw=2, niv=2, n_ranks=3, n_out=2, by_frequency=True, carried=True) == 2 * 2
    assert column_sde_slabs(niw=2, niv=2, n_ranks=1, n_out=3, by_frequency=True) == 3 * 2 + 4


def test_column_sde_stream_bytes_are_the_carried_slabs_and_one_round_of_every_kernel_beside_the_column_work():
    """Carried slabs; a budget-sized round of all kernels' columns with one send copy, the column work and slabs."""
    nb = 2
    slab, column, full = DTYPE_BYTES * 80 * nb**2, DTYPE_BYTES * 20 * nb**4, DTYPE_BYTES * 80 * nb**4
    carried, transient = column_sde_stream_bytes(20, 80, nb, 9, 3, 4, 5, 3, 2, 2 * 3 * column)
    assert carried == column_sde_slabs(9, 3, 4, 2, by_frequency=True, carried=True) * slab
    work = column_sde_slabs(9, 3, 4, 2, by_frequency=True) * slab - carried
    columns = SDE_COLUMN_FACTOR * full + min(SDE_CONTRACTION_CHUNK_BYTES, full)
    assert np.allclose(transient, 2 * column * (3 + 5 / 20) + columns + work, rtol=0, atol=1e-6)
    floor = column_sde_stream_bytes(20, 80, nb, 9, 3, 4, 5, 3, 2, 0)[1]
    assert np.allclose(floor, column * (3 + 5 / 20) + columns + work, rtol=0, atol=1e-6)


def test_sde_transient_is_the_irr_kernel_one_round_the_column_work_and_the_slabs():
    """The sde transient is the irr kernel, one round of irr columns with its send copy, the column work and slabs."""
    nb, wp, vc = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_core"]
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    task = DTYPE_BYTES * TINY["nk_irr"] * nb**4 * SDE_W_BLOCK
    _, bounds, _ = column_sde_schedule(TINY["niw_core"], TINY["niv_core"], TINY["n_ranks"])
    per_round = min(max(1, SLICE_CHUNK_BYTES // task), int(np.diff(bounds).max()))
    column = DTYPE_BYTES * TINY["nk_tot"] * nb**4
    columns = SDE_COLUMN_FACTOR * column + min(SDE_CONTRACTION_CHUNK_BYTES, column)
    slabs = column_sde_slabs(TINY["niw_core"], TINY["niv_core"], TINY["n_ranks"]) * DTYPE_BYTES * TINY["nk_tot"] * nb**2
    expected = SCALE * qi * nb**4 * wp * vc + OVERHEAD_FACTOR * (
        per_round * task * (1 + qi / TINY["nk_irr"]) + columns + slabs
    )
    assert estimate_peaks(**TINY)["sde"].off_distributed == pytest.approx(expected)


def test_sde_round_follows_the_budget_in_whole_tasks_between_one_task_and_the_ranks_tasks():
    """The modeled round grows by one task per task-sized budget step, from one task up to the rank's whole run."""
    nb, qi = TINY["n_bands"], -(-TINY["nk_irr"] // TINY["n_ranks"])
    task = DTYPE_BYTES * TINY["nk_irr"] * nb**4 * SDE_W_BLOCK
    _, bounds, _ = column_sde_schedule(TINY["niw_core"], TINY["niv_core"], TINY["n_ranks"])
    most = int(np.diff(bounds).max())
    at = {
        b: estimate_peaks(**TINY, chunk_budgets=ChunkBudgets(sde=b))["sde"].off_distributed for b in (0, task, 2 * task)
    }
    step = OVERHEAD_FACTOR * task * (1 + qi / TINY["nk_irr"])
    assert at[0] == pytest.approx(at[task]) and at[2 * task] - at[task] == pytest.approx(step)
    whole = estimate_peaks(**TINY, chunk_budgets=ChunkBudgets(sde=MAX_CHUNK_BUDGET_BYTES))["sde"].off_distributed
    assert whole == pytest.approx(at[task] + (most - 1) * step)


def test_sde_single_holds_the_finalize_buffers_whatever_the_dmft_box():
    """The sde single-rank slot is rank 0's two core-box sigma buffers; the DMFT box does not enter it."""
    small = estimate_peaks(**{**TINY, "niv_dmft": TINY["niv_core"]})["sde"].off_single
    big = estimate_peaks(**{**TINY, "niv_dmft": 100 * TINY["niv_core"]})["sde"].off_single
    nb = TINY["n_bands"]
    assert small == big == pytest.approx(SCALE * 2 * TINY["nk_tot"] * nb**2 * 2 * TINY["niv_core"] + _v_single(TINY))


def test_energies_branch_holds_each_ranks_share_pair_and_its_largest_green_function_transient():
    """Every rank holds its share's sigma and Green's function on the DMFT box plus the largest of the Dyson,
    occupation and energy-tail transients; rank 0 the loop's residents of the chemical-potential update and the
    whole-grid moment-fit window (the top fifth of the box)."""
    params = {**TINY, "niv_dmft": 100 * TINY["niv_core"]}
    nb, share_k, box = params["n_bands"], -(-params["nk_tot"] // params["n_ranks"]), 2 * params["niv_dmft"]
    peaks = estimate_peaks(**params)
    bp = peaks["energies"]
    transients = memory_estimator._green_function_transients(share_k, nb, box)
    assert bp.off_distributed == pytest.approx(SCALE * 2 * share_k * nb**2 * box + max(transients))
    fit_window = SCALE * params["nk_tot"] * nb**2 * int(0.2 * params["niv_dmft"])
    assert bp.off_single == pytest.approx(peaks["mu_update"].off_single + fit_window)
    assert bp.baseline == pytest.approx(RANK_BASELINE_BYTES) and bp.giwk_shareable == 0.0
    assert (bp.on_distributed, bp.on_single) == (bp.off_distributed, bp.off_single)


def test_green_function_transients_are_bounded_by_the_chunk_sizes():
    """The Dyson, occupation and tail transients grow with the box up to their chunks and stay there."""
    small = memory_estimator._green_function_transients(4, 2, 40)
    assert small[0] == 16 * (2 * 4 * 4 * 40 + 4 * 4) and small[1] == int(16 * 3.5 * 4 * 4 * 40)
    big = memory_estimator._green_function_transients(10**4, 3, 2000)
    assert big[0] == 16 * (2 * memory_estimator.GF_FREQUENCY_CHUNK_ELEMENTS + 10**4 * 9)
    assert big[1] == int(16 * 3.5 * memory_estimator.GF_MOMENTUM_CHUNK_ELEMENTS)
    assert big[2] == 16 * 3 * memory_estimator.GF_FREQUENCY_CHUNK_ELEMENTS


def test_occupation_branch_adds_the_dmft_lattice_green_function_on_rank0_only_with_the_dmft_spectrum():
    """Every rank holds one momentum chunk's occupation transient; with the DMFT spectrum rank 0 also builds the
    whole-grid Green's function on the DMFT box."""
    nb, nk, box = TINY["n_bands"], TINY["nk_tot"], 2 * TINY["niv_dmft"]
    share_k = -(-nk // TINY["n_ranks"])
    plain = estimate_peaks(**TINY)["occupation"]
    assert plain.off_distributed == memory_estimator._green_function_transients(share_k, nb, box)[1]
    assert plain.off_single == _v_single(TINY) and plain.baseline == pytest.approx(RANK_BASELINE_BYTES)
    spectrum = estimate_peaks(**TINY, do_spectrum_dmft=True)["occupation"]
    dyson = memory_estimator._green_function_transients(nk, nb, box)[0]
    reload = SCALE * nk * nb**2 * (box + 2 * 2 * TINY["niv_cut"] + 2 * TINY["niv_core"])
    assert spectrum.off_single == pytest.approx(max(SCALE * nk * nb**2 * box + dyson, reload) + _v_single(TINY))
    assert spectrum.off_distributed == plain.off_distributed


def test_occupation_branch_holds_the_warm_start_on_rank0_and_its_mu_search_on_every_rank():
    """On a warm start rank 0 extends the whole-grid starting iterate onto the DMFT box next to its niv_cut copy and
    the occupation chunks, and every rank searches mu on its momenta."""
    nb, nk, box = TINY["n_bands"], TINY["nk_tot"], 2 * TINY["niv_dmft"]
    peaks = estimate_peaks(**TINY, warm_start=True)
    warm = peaks["occupation"]
    occupation = memory_estimator._green_function_transients(nk, nb, box)[1]
    sigma_full = nk * nb**2 * 2 * TINY["niv_cut"]
    assert warm.off_single == pytest.approx(SCALE * (nk * nb**2 * box + sigma_full) + occupation + _v_single(TINY))
    assert warm.off_distributed >= peaks["mu_update"].off_distributed
    assert estimate_peaks(**TINY)["occupation"].off_single == _v_single(TINY)


def test_sigma_loop_is_rank0_only_with_the_linear_mix_copies_or_the_accelerated_solve():
    """sigma_loop is rank 0's step alone: two niv_cut Sigmas plus the linear mix or the accelerated solve."""
    nk, nb = TINY["nk_tot"], TINY["n_bands"]
    core = nk * nb**2 * 2 * TINY["niv_core"]
    full = nk * nb**2 * 2 * TINY["niv_cut"]
    linear = SCALE * 2 * full + SCALE * 3 * full + _v_single(TINY)
    bp = estimate_peaks(**TINY)["sigma_loop"]
    assert bp.baseline == pytest.approx(RANK_BASELINE_BYTES) and bp.giwk_shareable == 0.0
    assert bp.off_distributed == 0.0 and bp.off_single == pytest.approx(linear)
    assert (bp.on_distributed, bp.on_single) == (bp.off_distributed, bp.off_single)
    solve = estimate_peaks(**TINY, mixing_pairs=4)["sigma_loop"].off_single
    assert solve == pytest.approx(SCALE * (8 * core + 2 * full) + 8 * core * (9 * 3 + 10) + _v_single(TINY))
    one_pair = estimate_peaks(**TINY, mixing_pairs=1)["sigma_loop"].off_single
    assert one_pair == pytest.approx(SCALE * 2 * core + linear)


def test_mu_update_holds_two_complex128_green_function_slices_per_rank_next_to_rank0_sigmas():
    """mu_update: every rank two complex128 G arrays on its momentum share, rank 0 two niv_cut Sigmas and history."""
    nk, nb, ranks = TINY["nk_tot"], TINY["n_bands"], TINY["n_ranks"]
    per_k = nb**2 * 2 * TINY["niv_cut"]
    bp = estimate_peaks(**TINY)["mu_update"]
    assert bp.baseline == pytest.approx(RANK_BASELINE_BYTES) and bp.giwk_shareable == 0.0
    assert bp.off_distributed == pytest.approx(16 * 2 * -(-nk // ranks) * per_k)
    assert bp.off_single == pytest.approx(SCALE * 2 * nk * per_k + _v_single(TINY))
    assert (bp.on_distributed, bp.on_single) == (bp.off_distributed, bp.off_single)
    history = estimate_peaks(**TINY, mixing_pairs=4)["mu_update"].off_single - bp.off_single
    assert history == pytest.approx(SCALE * 8 * nk * nb**2 * 2 * TINY["niv_core"])


def test_sigma_interp_branch_models_rank0_interpolating_the_irreducible_sigma():
    """With an interpolation target rank 0 re-grids and unfolds the irreducible Sigma, every rank at most its share."""
    nb, niv_interp = TINY["n_bands"], 2 * TINY["niv_cut"]
    per_q_source, per_q_target = nb**2 * 2 * TINY["niv_cut"], nb**2 * 2 * niv_interp
    source, target = TINY["nk_irr"] * per_q_source, TINY["nk_irr"] * per_q_target
    share = -(-TINY["nk_irr"] // TINY["n_ranks"])
    full = TINY["nk_tot"] * per_q_source
    unfold = SCALE * (target + TINY["nk_tot"] * per_q_target)
    assert "sigma_interp" not in estimate_peaks(**TINY)
    bp = estimate_peaks(**TINY, niv_interp=niv_interp)["sigma_interp"]
    assert bp.off_distributed == pytest.approx(
        share * (SCALE * per_q_source + 16 * (10 * per_q_source + 4 * per_q_target))
    )
    rank0 = SCALE * (full + source) + max(16 * (10 * source + 4 * target), unfold) + _v_single(TINY)
    assert bp.off_single == pytest.approx(rank0)
    assert (bp.on_distributed, bp.on_single) == (bp.off_distributed, bp.off_single)


def test_sde_off_and_on_slots_are_identical():
    """The sde step is single-path, so both path slots carry the same two-pass FFT estimate."""
    bp = _peaks()["sde"]
    assert bp.on_distributed == pytest.approx(bp.off_distributed)
    assert bp.on_single == pytest.approx(bp.off_single)


def test_fq_lean_includes_rank_local_accumulator_and_loads():
    """The fq lean transient grows with the per-rank q-count via the accumulator and the three 1-fermion loads."""
    few_ranks = _peaks(with_eliashberg=True, n_ranks=2)["fq"].on_distributed
    many_ranks = _peaks(with_eliashberg=True, n_ranks=8)["fq"].on_distributed
    assert few_ranks > many_ranks


def test_lanczos_fast_counts_layout_build_vertices_bubble_and_arpack_basis():
    """The lanczos fast peak holds the build-peak vertices, the pp bubble and the ARPACK workspace."""
    p = {**TINY, "with_eliashberg": True, "n_ranks": 4}
    nb, vpp = p["n_bands"], 2 * p["niv_pp"]
    vertex = p["nk_tot"] * nb**4 * vpp * vpp
    chi0 = p["nk_tot"] * nb**4 * vpp
    arpack = (memory_estimator.lanczos_ncv(1) + ARPACK_EXTRA_VECTORS) * p["nk_tot"] * nb**2 * vpp
    giwk_dga = p["nk_tot"] * nb**2 * 2 * p["niv_cut"]
    expected = SCALE * (LANCZOS_VERTEX_FACTOR * vertex + chi0 + arpack + giwk_dga) + _v_single(p)
    assert estimate_peaks(**p)["lanczos"].off_single == pytest.approx(expected)


def test_lanczos_lean_independent_of_irreducible_bz_size():
    """The lanczos lean transient is independent of the irreducible-BZ size."""
    a = _peaks(with_eliashberg=True, nk_irr=10)["lanczos"].on_distributed
    b = _peaks(with_eliashberg=True, nk_irr=60)["lanczos"].on_distributed
    assert a == pytest.approx(b)


def test_lanczos_grid_share_saturates_at_the_squared_task_count():
    """The grid vertex share shrinks past the row cap via columns and saturates at (2*niv_pp)^2 ranks."""
    n_freq = 2 * BASE["niv_pp"]
    at_rows = _peaks(with_eliashberg=True, n_ranks=n_freq)["lanczos"].on_distributed
    with_cols = _peaks(with_eliashberg=True, n_ranks=2 * n_freq)["lanczos"].on_distributed
    at_cap = _peaks(with_eliashberg=True, n_ranks=n_freq * n_freq)["lanczos"].on_distributed
    beyond = _peaks(with_eliashberg=True, n_ranks=2 * n_freq * n_freq)["lanczos"].on_distributed
    assert with_cols < at_rows
    assert beyond == pytest.approx(at_cap)


def test_lanczos_arpack_workspace_grows_with_n_eig():
    """Requesting more eigenpairs than the default ncv=20 basis grows the per-rank ARPACK workspace."""
    default = _peaks(with_eliashberg=True, n_eig=1)["lanczos"]
    many = _peaks(with_eliashberg=True, n_eig=30)["lanczos"]
    assert many.off_single > default.off_single


def test_lanczos_ncv_is_two_n_eig_plus_one_above_the_floor():
    """The Lanczos basis size is 2 n_eig + 1, floored at LANCZOS_NCV_FLOOR."""
    assert memory_estimator.lanczos_ncv(1) == memory_estimator.LANCZOS_NCV_FLOOR == 12
    assert memory_estimator.lanczos_ncv(12) == 25


def test_lanczos_team_bytes_on_the_wedge_holds_irreducible_windows(monkeypatch):
    """With wedge windows the per-channel term is the irreducible block instead of the full-BZ one."""
    monkeypatch.setattr(memory_estimator, "TEAM_BUILD_CHUNK_BYTES", 1000 * memory_estimator.DTYPE_BYTES)
    full = memory_estimator.lanczos_team_bytes(2, 64, 10, 3, 4, 2, 2, 3)
    wedge = memory_estimator.lanczos_team_bytes(2, 64, 10, 3, 4, 2, 2, 3, 10)
    vertex = memory_estimator._two_fermion_block(64, 2, 1, 6)
    irr = memory_estimator._two_fermion_block(10, 2, 1, 6)
    assert full - wedge == 2 * memory_estimator.DTYPE_BYTES * (vertex - irr)


def test_lanczos_team_bytes_counts_the_residual_block_of_the_block_iteration():
    """A block of three starting vectors adds two gap vectors to every sector's Krylov basis."""
    plain = memory_estimator.lanczos_team_bytes(2, 64, 10, 3, 4, 1, 2, 3)
    block = memory_estimator.lanczos_team_bytes(2, 64, 10, 3, 4, 1, 2, 3, block=3)
    assert block - plain == memory_estimator.DTYPE_BYTES * 2 * 2 * memory_estimator._giwk_rspace(64, 2, 6)


def test_lanczos_team_bytes_counts_windows_source_build_blocks_bubble_and_sectors(monkeypatch):
    """The team-solve node peak is the windows, the irr source beside the build blocks, the bubble and the sectors."""
    monkeypatch.setattr(memory_estimator, "TEAM_BUILD_CHUNK_BYTES", 1000 * memory_estimator.DTYPE_BYTES)
    one = memory_estimator.lanczos_team_bytes(2, 64, 10, 3, 4, 1, 2, 3)
    two = memory_estimator.lanczos_team_bytes(2, 64, 10, 3, 4, 2, 2, 3)
    more_sectors = memory_estimator.lanczos_team_bytes(2, 64, 10, 3, 4, 2, 4, 3)
    many_ranks = memory_estimator.lanczos_team_bytes(2, 64, 10, 3, 4, 1, 2, 10**6)
    vertex = memory_estimator._two_fermion_block(64, 2, 1, 6)
    irr = memory_estimator._two_fermion_block(10, 2, 1, 6)
    vectors = (
        memory_estimator.lanczos_ncv(4)
        + memory_estimator.TEAM_SCRATCH_VECTORS
        + 2 * 4
        + memory_estimator.TEAM_MATVEC_TRANSIENT_VECTORS
    )
    per_sector = vectors * memory_estimator._giwk_rspace(64, 2, 6)
    bubble = memory_estimator._bubble_block(64, 2, 1, 6)
    block = 64 * 2**4 * memory_estimator.team_build_columns(64, 2, 6)
    assert memory_estimator.team_build_columns(64, 2, 6) == 1
    assert two - one == memory_estimator.DTYPE_BYTES * vertex
    assert more_sectors - two == 2 * memory_estimator.DTYPE_BYTES * per_sector
    assert one == memory_estimator.DTYPE_BYTES * (vertex + irr + max(irr, 6 * block) + bubble + 2 * per_sector)
    assert many_ranks == memory_estimator.DTYPE_BYTES * (3 * vertex + irr + bubble + 2 * per_sector)


def test_save_pairing_vertex_enters_the_single_rank_gather_peak():
    """save_pairing_vertex gathers both irr-BZ pp vertices on one rank, raising the solver peak when it dominates."""
    p = {**TINY, "with_eliashberg": True}
    with_save = estimate_peaks(**{**p, "save_pairing_vertex": True})["lanczos"]
    without = estimate_peaks(**{**p, "save_pairing_vertex": False})["lanczos"]
    assert with_save.off_single >= without.off_single
    assert with_save.on_single > without.on_single  # the gather is a single-rank peak of the grid variant too


@pytest.mark.parametrize("niv_full, symmetrize", [(40, False), (40, True), (30, False)])
def test_local_step_is_flagless_single_rank_and_band_heavy(niv_full, symmetrize):
    """The local branch is verify-only, rank-independent and nb^4-heavy: the larger of the inversion and F phases."""
    bp = _peaks(niv_full=niv_full, symmetrize_orbitals=symmetrize)["local"]
    assert bp.baseline == pytest.approx(RANK_BASELINE_BYTES) and bp.giwk_shareable == 0.0 and bp.off_distributed == 0.0
    assert bp.off_single == bp.on_single > 0.0
    assert _peaks(n_ranks=16, niv_full=niv_full, symmetrize_orbitals=symmetrize)["local"].off_single == bp.off_single
    two_bands = _peaks(n_bands=2, niv_full=niv_full, symmetrize_orbitals=symmetrize)["local"].off_single
    assert two_bands == pytest.approx(16 * bp.off_single)
    wp, vc, vf = BASE["niw_core"] + 1, 2 * BASE["niv_core"], 2 * niv_full
    l_core, f_full = wp * vc * vc, wp * vf * vc
    expected = SCALE * max(4 * l_core + (3 if symmetrize else 2) * f_full, 6 * l_core + f_full) + _v_single(BASE)
    assert bp.off_single == pytest.approx(expected)


def test_dynamic_chunk_budget_scales_floors_and_caps():
    """The dynamic chunk budget grows with free memory per node rank, floored and capped at the class constants."""
    floor, cap = SLICE_CHUNK_BYTES, MAX_SLICE_CHUNK_BYTES
    assert dynamic_chunk_budget(total_bytes=1, node_ranks=48) == floor
    mid = dynamic_chunk_budget(total_bytes=300 * 2**30, node_ranks=48)
    assert floor < mid < cap
    assert dynamic_chunk_budget(total_bytes=4000 * 2**30, node_ranks=2) == cap
    assert dynamic_chunk_budget(total_bytes=600 * 2**30, node_ranks=48) == 2 * mid


def test_jacobian_tracker_residents_and_sector_history_join_the_loop_slots_and_its_transient_the_mixing_step():
    """With the tracker its residents and a sector-sized history fill the rank-0 slots, the solve has sector rows."""
    kw = dict(overhead=1.0, niv_interp=2 * BASE["niv_core"])
    core = DTYPE_BYTES * BASE["nk_tot"] * BASE["n_bands"] ** 2 * (2 * BASE["niv_core"])
    sector, k = BASE["nk_irr"] * BASE["n_bands"] ** 2 * (2 * BASE["niv_core"]), TRACKER_PAIRS - 1
    # window, predecessor, estimate, snapshot and held flips in the storage precision, the binding and the waiting pair
    # in complex64; four tall float64 arrays
    resident = (TRACKER_PAIRS * DTYPE_BYTES + DTYPE_BYTES * k + DTYPE_BYTES * 3 * k + 8 * 4 + 4 * 8 * k) * sector
    transient = 7 * np.dtype(np.float64).itemsize * sector * k
    assert jacobian_tracker_bytes(BASE["nk_irr"], BASE["n_bands"], 2 * BASE["niv_core"]) == (resident, transient)
    sigma_full = _giwk_rspace(BASE["nk_tot"], BASE["n_bands"], 2 * BASE["niv_cut"])
    for pairs in (0, 4, 40):
        off = _peaks(**kw, mixing_pairs=pairs)
        on = _peaks(**kw, mixing_pairs=pairs, with_jacobian_tracker=True)
        window, history = 2 * pairs * core, 2 * pairs * (DTYPE_BYTES // 2) * sector
        for key in ("chi0q", "chiq_aux", "sde", "mu_update"):
            assert on[key].off_single == pytest.approx(off[key].off_single - window + history + resident), key
            assert on[key].on_single == pytest.approx(off[key].on_single - window + history + resident), key
            assert on[key].off_distributed == pytest.approx(off[key].off_distributed), key
        assert on["sigma_interp"].off_single == pytest.approx(off["sigma_interp"].off_single + resident)
        solve = 4 * sector * (9 * (pairs - 1) + 10) if pairs > 1 else 0
        step = max(DTYPE_BYTES * 3 * sigma_full, solve, transient)
        expected = history + resident + 2 * core + DTYPE_BYTES * 2 * sigma_full + step + _v_single(BASE)
        assert on["sigma_loop"].off_single == pytest.approx(expected)
        assert (step == solve) == (pairs == 40)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_jacobian_tracker_residents_count_101_sector_columns_with_exact_checks_and_59_without_in_complex64(
    dtype, monkeypatch
):
    """With exact checks the tracker holds 101 float64 sector columns in complex64, 59 without, more in complex128."""
    monkeypatch.setattr(memory_estimator, "DTYPE_BYTES", np.dtype(dtype).itemsize)
    nk_irr, nb, nv = BASE["nk_irr"], BASE["n_bands"], 2 * BASE["niv_core"]
    column, storage = 8 * nk_irr * nb**2 * nv, np.dtype(dtype).itemsize / np.dtype(np.complex64).itemsize
    # the window, the predecessor file, the Ritz sets but the binding and the waiting pair, and the check candidate
    # double in a complex128 run
    exact = (101 + (7 + 12 + 38 + 2) * (storage - 1)) * column
    secant = (59 + (7 + 6 + 18) * (storage - 1)) * column
    assert jacobian_tracker_bytes(nk_irr, nb, nv, with_exact_jacobian=True) == (exact, 42 * column)
    assert jacobian_tracker_bytes(nk_irr, nb, nv) == (secant, 42 * column)
    both = _peaks(with_jacobian_tracker=True, with_exact_jacobian=True, overhead=1.0)
    tracker = _peaks(with_jacobian_tracker=True, overhead=1.0)
    for key in ("chi0q", "chiq_aux", "sde", "sigma_loop"):
        assert both[key].off_single == pytest.approx(tracker[key].off_single + exact - secant), key


# a grid whose core blocks dwarf the per-slice Bethe-Salpeter solve's fixed transient, so the solve's phase stays small
LARGE_GRID = dict(nk_tot=128 * 128, nk_irr=2145)


def test_bubble_round_bytes_is_the_bubble_generators_round_buffer():
    """The estimator's round of full-grid columns is the window bubble's round buffer."""
    assert memory_estimator.BUBBLE_ROUND_BYTES == bubble_gen.ROUND_BUFFER_BYTES


@pytest.mark.parametrize(
    "budgets",
    [ChunkBudgets(exact=0), ChunkBudgets(exact=0, exact_block=3, exact_window=2), ChunkBudgets(exact_window=1)],
)
@pytest.mark.parametrize("grid", [{}, LARGE_GRID, {**LARGE_GRID, "n_ranks": 1}])
def test_exact_jacobian_branch_is_its_residents_beside_the_largest_stage_of_a_window(budgets, grid):
    """Per rank: residents and carried slabs beside a window's bubbles, dc split, Bethe-Salpeter pass or contraction."""
    params = {**BASE, **grid}
    bp = _peaks(with_exact_jacobian=True, overhead=1.0, chunk_budgets=budgets, **grid)["exact_jacobian"]
    nb, nk_tot, nk_irr, niw, niv, nivf = (
        params[k] for k in ("n_bands", "nk_tot", "nk_irr", "niw_core", "niv_core", "niv_full")
    )
    qi, wp, m = -(-nk_irr // params["n_ranks"]), niw + 1, budgets.exact_block
    n_w = wp if budgets.exact_window is None else budgets.exact_window
    window_core, window_full = DTYPE_BYTES * qi * nb**4 * n_w * 2 * niv, DTYPE_BYTES * qi * nb**4 * n_w * 2 * nivf
    residents = DTYPE_BYTES * (4 * qi * nb**4 * wp * 2 * niv + 2 * qi * nb**4 * wp)
    carried, contraction = column_sde_stream_bytes(
        nk_irr, nk_tot, nb, niw, niv, params["n_ranks"], qi, m + 1, m, budgets.exact
    )
    column, round_bytes = DTYPE_BYTES * nk_tot * nb**4, memory_estimator.BUBBLE_ROUND_BYTES
    rounds = max(column, round_bytes) * (1 + (m + 1) * (2 * nk_irr + qi) / nk_tot)
    rounds += 4 * DTYPE_BYTES * nk_tot * nb**2 + round_bytes
    if params["n_ranks"] == 1:
        rounds = column * 2 * nivf + 2 * DTYPE_BYTES * nk_tot * nb**2 * 2 * nivf
        rounds += max(round_bytes, DTYPE_BYTES * nb**4 * 2 * nivf)
    bubble = (m + 1) * window_full + max(rounds, (DTYPE_BYTES + 3) * window_full / DTYPE_BYTES)
    dc, bse = exact_window_stages(qi, nb, n_w, niv, nivf, m, False)
    stages = (bubble, dc, bse, m * window_core + contraction)
    assert bp.off_distributed == pytest.approx(residents + carried + max(stages))


def test_jacobian_tracker_keeps_44_sector_columns_after_the_loop_with_exact_checks_in_complex64():
    """After the loop the tracker keeps snapshot, held flips, predecessor and reflector basis: 44 F in complex64."""
    nk_irr, nb, nv = BASE["nk_irr"], BASE["n_bands"], 2 * BASE["niv_core"]
    column, storage = 8 * nk_irr * nb**2 * nv, DTYPE_BYTES / np.dtype(np.complex64).itemsize
    after = jacobian_tracker_bytes(nk_irr, nb, nv, with_exact_jacobian=True, after_loop=True)
    assert after == ((44 + (12 + 16) * (storage - 1)) * column, 0)


@pytest.mark.parametrize("mixing_pairs, ncv", [(0, 40), (4, 40), (0, 400)])
def test_exact_jacobian_rank_0_slot_is_the_larger_of_an_in_loop_check_and_the_end_of_rung_solve(
    mixing_pairs, ncv, monkeypatch
):
    """Rank 0 holds a product beside the check, history, tracker and proposal, or ARPACK and the released tracker."""
    monkeypatch.setattr(memory_estimator, "EXACT_JACOBIAN_NCV", ncv)
    kw = dict(with_exact_jacobian=True, with_jacobian_tracker=True, mixing_pairs=mixing_pairs, overhead=1.0)
    bp = _peaks(**kw)["exact_jacobian"]
    nk_irr, nb, nv = BASE["nk_irr"], BASE["n_bands"], 2 * BASE["niv_core"]
    vector = 8 * nk_irr * nb**2 * nv
    sigma_core = DTYPE_BYTES * _giwk_rspace(BASE["nk_tot"], nb, nv)
    sigma_full = DTYPE_BYTES * _giwk_rspace(BASE["nk_tot"], nb, 2 * BASE["niv_cut"])
    product = 2 * sigma_core + DTYPE_BYTES * BASE["nk_tot"] * nb**4
    history = 2 * mixing_pairs * (DTYPE_BYTES // 2) * nk_irr * nb**2 * nv
    in_loop = history + jacobian_tracker_bytes(nk_irr, nb, nv, True)[0] + sigma_full
    in_loop += (4 * EXACT_CHECK_BASIS + 4) * vector
    end_of_rung = jacobian_tracker_bytes(nk_irr, nb, nv, True, after_loop=True)[0]
    end_of_rung += (2 * (ncv + 1) + 3 * EXACT_JACOBIAN_MODES + 11) * vector
    assert bp.off_single == pytest.approx(product + _v_single(BASE) + max(in_loop, end_of_rung))
    assert (end_of_rung > in_loop) == (ncv > 100)


@pytest.mark.parametrize("width_cost, width, window", [(1 / 16, 4, 31), (1 / 8, 4, 25), (1 / 3, 2, 8), (1.0, None, 1)])
def test_exact_jacobian_budgets_take_the_widest_block_the_widest_window_and_the_budget_they_leave(
    width_cost, width, window
):
    """The exact products take the widest block, then the widest window that fits, then the budget left."""
    line, loop = 3 * 1024**3, ChunkBudgets(sde=2, fq=3)

    def fits(trial: ChunkBudgets) -> bool:
        """Whether the trial's block, window and budget fit the line."""
        return 3 * trial.exact + (width_cost * trial.exact_block + 0.01 * trial.exact_window) * line <= line

    budgets = exact_jacobian_budgets(fits, loop, 31)
    assert (budgets.sde, budgets.fq) == (2, 3)
    if width is None:
        assert budgets == ChunkBudgets(2, 3, SLICE_CHUNK_BYTES, 1, 1, True, True)
    else:
        left = (line - (width_cost * width + 0.01 * window) * line) / 3
        assert (budgets.exact_block, budgets.exact_window) == (width, window) and not budgets.exact_vertices_per_window
        assert left - 2**20 <= budgets.exact <= left


@pytest.mark.parametrize(
    "local, vrg, residency, width",
    [
        (0.1, 0.1, (False, False), 4),
        (0.45, 0.1, (True, False), 4),
        (0.7, 0.1, (True, False), 4),
        (0.4, 0.4, (True, True), 4),
        (0.05, 0.42, (False, True), 4),
        (0.1, 0.7, (False, True), 4),
        (0.7, 0.7, (True, True), 4),
        (0.1, 0.0, (False, False), 4),
        (0.6, 0.0, (True, False), 4),
    ],
)
def test_exact_jacobian_budgets_take_the_cheapest_in_loop_check_that_fits(local, vrg, residency, width):
    """Fewest pieces per check first, recomputed three-leg vertices 2 % dearer, held local vertices on a tie."""
    line = 3 * 1024**3

    def fits(trial: ChunkBudgets) -> bool:
        """Whether the trial fits the line, each vertex set held unless loaded or recomputed per window."""
        held = (0.0 if trial.exact_vertices_per_window else local) + (0.0 if trial.exact_vrg_per_window else vrg)
        return 3 * trial.exact + (0.1 * trial.exact_block + 0.001 * (trial.exact_window - 1) + held) * line <= line

    budgets = exact_jacobian_budgets(fits, ChunkBudgets(), 31)
    assert (budgets.exact_vertices_per_window, budgets.exact_vrg_per_window) == residency
    assert budgets.exact_block == width and 1 <= budgets.exact_window <= 31 and fits(budgets)


@pytest.mark.parametrize("dtype_bytes, nodes, picked", [(8, 2, (4, False, False)), (16, 3, (4, True, True))])
def test_exact_jacobian_budgets_give_a_production_box_four_columns_per_piece(dtype_bytes, nodes, picked, monkeypatch):
    """At 50^3 momenta, 4 bands and 150 frequencies on 1 TiB nodes of 96 ranks a piece takes the widest block."""
    monkeypatch.setattr(memory_estimator, "DTYPE_BYTES", dtype_bytes)
    params = dict(BASE, n_bands=4, nk_tot=50**3, nk_irr=3276, niw_core=150, niv_core=150, niv_full=300, niv_cut=460)
    params.update(niv_dmft=1000, niv_pp=75, n_ranks=96 * nodes, with_jacobian_tracker=True, with_exact_jacobian=True)

    def fits(trial: ChunkBudgets) -> bool:
        """Whether rank 0's node of busiest ranks holds the exact branch within 95 % of 1 TiB."""
        bp = estimate_peaks(**params, chunk_budgets=trial)["exact_jacobian"]
        return node_total([(96, bp)], holds_rank_zero=True) <= 0.95 * 1024**4

    budgets = exact_jacobian_budgets(fits, ChunkBudgets(), 151)
    assert (budgets.exact_block, budgets.exact_vertices_per_window, budgets.exact_vrg_per_window) == picked
    assert fits(budgets) and not fits(replace(budgets, exact_window=budgets.exact_window + 1))


@pytest.mark.parametrize("window", [1, 4])
def test_exact_jacobian_branch_recomputing_the_three_leg_vertices_holds_two_core_blocks_fewer(window):
    """Three-leg vertices recomputed per window hold two core blocks fewer and add one window block to the pass."""
    held = ChunkBudgets(exact=0, exact_window=window)
    held_bp = _peaks(with_exact_jacobian=True, overhead=1.0, chunk_budgets=held, **LARGE_GRID)["exact_jacobian"]
    per_window = replace(held, exact_vrg_per_window=True)
    window_bp = _peaks(with_exact_jacobian=True, overhead=1.0, chunk_budgets=per_window, **LARGE_GRID)["exact_jacobian"]
    qi, nb, vc = -(-LARGE_GRID["nk_irr"] // BASE["n_ranks"]), BASE["n_bands"], 2 * BASE["niv_core"]
    core = DTYPE_BYTES * qi * nb**4 * (BASE["niw_core"] + 1) * vc
    window_block = DTYPE_BYTES * qi * nb**4 * window * vc
    bse = [
        exact_window_stages(qi, nb, window, BASE["niv_core"], BASE["niv_full"], 1, flag)[1] for flag in (False, True)
    ]
    assert bse[1] - bse[0] == pytest.approx(window_block)
    assert 2 * core - window_block <= held_bp.off_distributed - window_bp.off_distributed <= 2 * core
    assert (held_bp.baseline, held_bp.off_single) == (window_bp.baseline, window_bp.off_single)


@pytest.mark.parametrize("width, window, per_window", [(1, 1, False), (2, 4, False), (3, 31, True)])
def test_exact_window_stages_are_the_worst_response_of_the_split_and_the_pass_beside_the_solve(
    width, window, per_window
):
    """The split: a response in work beside the earlier ones' parts and the later ones; the pass beside the solve."""
    qi, nb, nc, nf = 9, 2, 30, 40
    vc, vf = 2 * nc, 2 * nf
    core, full = qi * nb**4 * window * vc, qi * nb**4 * window * vf
    part, one_q = (2 + 2 / vc) * core, nb**4 * window * (vf + 2 * vc)
    work = max(core + one_q, (2 + 1 / vc) * core)
    dc = max(
        max(i * part + (width - i) * full + work, i * part + (width - 1 - i) * full + (4 + 2 / vc) * core)
        for i in (0, width - 1)
    )
    bse = DTYPE_BYTES * (width * part + (6 + 2 * width + per_window) * core)
    bse += memory_estimator._exact_solve_transient(nb * nb * vc, nb, width)
    stages = exact_window_stages(qi, nb, window, nc, nf, width, per_window)
    assert np.allclose(stages, (DTYPE_BYTES * dc, bse), rtol=0, atol=1.0)
    assert exact_window_stages(0, nb, window, nc, nf, width, per_window) == (0.0, 0.0)


@pytest.mark.parametrize("o, niv, n_rhs", [(6, 20, 1), (6, 20, 4), (8, 10, 2), (8, 10, 4)])
def test_exact_solve_transient_bounds_the_traced_peak_of_the_right_hand_side_solve(o, niv, n_rhs, monkeypatch):
    """The modeled transient covers the traced per-slice solve of n_rhs systems above its solutions, up to 8 bands."""
    monkeypatch.setattr(config.sys, "beta", 9.0, raising=False)
    vn, n, rng = 2 * niv, o * o * 2 * niv, np.random.default_rng(5)
    gamma = LocalFourPoint(rng.standard_normal((o,) * 4 + (2, vn, vn)) + 0j, SpinChannel.DENS, 1, 2, False, True)
    # two momenta and two bosonic frequencies: every slice after the first meets its predecessor's right-hand sides
    shape, meta = (2,) + (o,) * 4 + (2, vn), (SpinChannel.NONE, (2, 1, 1), 1, 1, False, True, True)
    gchi0_q_inv = FourPoint(rng.standard_normal(shape) + 30.0, *meta)
    u_loc = LocalInteraction(np.ones((o,) * 4), SpinChannel.NONE)
    rhs = [FourPoint(rng.standard_normal(shape) + 0j, *meta) for _ in range(n_rhs)]
    nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_q_inv, u_loc, rhs=rhs)
    solved, peak = traced_peak(lambda: nonlocal_sde.create_auxiliary_chi_r_q_sum(gamma, gchi0_q_inv, u_loc, rhs=rhs))
    above = peak - sum(x.mat.nbytes for x in solved)
    assert 2 * n**2 * DTYPE_BYTES < above <= memory_estimator._exact_solve_transient(n, o, n_rhs)


def test_exact_jacobian_branch_grows_with_its_window_up_to_the_whole_range():
    """The exact branch grows with its window, monotone as the budget search needs, the whole range at the top."""
    peaks = [
        _peaks(with_exact_jacobian=True, chunk_budgets=ChunkBudgets(exact_window=w), **LARGE_GRID)["exact_jacobian"]
        for w in (1, 2, 8, 31, 40)
    ]
    totals = [_off_node_total(bp, 4) for bp in peaks]
    whole = _peaks(with_exact_jacobian=True, **LARGE_GRID)["exact_jacobian"]
    assert all(a < b for a, b in zip(totals[:3], totals[1:4])) and peaks[3] == peaks[4] == whole


def test_exact_jacobian_branch_is_present_only_with_the_flag_and_leaves_the_other_branches_alone():
    """The converged-point Jacobian adds its own branch, only when it is enabled, and changes no other branch."""
    off, on = _peaks(), _peaks(with_exact_jacobian=True)
    assert "exact_jacobian" not in off and set(on) == set(off) | {"exact_jacobian"}
    assert all(on[key] == off[key] for key in off)


def test_exact_jacobian_branch_grows_with_its_block_width_and_leaves_the_other_branches_alone():
    """A wider block of exact products raises the exact branch's per-rank peak and no other branch."""
    narrow = _peaks(with_exact_jacobian=True, chunk_budgets=ChunkBudgets(exact_block=1))
    wide = _peaks(with_exact_jacobian=True, chunk_budgets=ChunkBudgets(exact_block=4))
    assert wide["exact_jacobian"].off_distributed > narrow["exact_jacobian"].off_distributed
    assert all(wide[key] == narrow[key] for key in narrow if key != "exact_jacobian")


@pytest.mark.parametrize("width", [1, 3])
def test_exact_jacobian_branch_counts_its_node_windows_and_rank_0s_check_exactly(width):
    """Node: loop Sigma, G and change windows, the real-space windows of G and a piece, three vertices, the block."""
    budgets = ChunkBudgets(exact_block=width)
    bp = _peaks(with_exact_jacobian=True, overhead=1.0, chunk_budgets=budgets)["exact_jacobian"]
    nk, niv, niw, nivf, cut = BASE["nk_tot"], BASE["niv_core"], BASE["niw_core"], BASE["niv_full"], BASE["niv_cut"]
    sigma_full, sigma_core, sector = nk * 2 * cut, nk * 2 * niv, 2 * BASE["nk_irr"] * niv
    vertices = (niw + 1) * 2 * nivf * 2 * niv + 2 * (niw + 1) * (2 * niv) ** 2
    rspace = nk * 2 * (nivf + niw)
    windows = DTYPE_BYTES * (3 * sigma_full + (width + 1) * rspace + vertices) + 8 * EXACT_CHECK_BASIS * sector
    assert bp.giwk_shareable == pytest.approx(windows)
    assert bp.baseline == pytest.approx(windows + RANK_BASELINE_BYTES)
    arpack = 8 * (2 * (EXACT_JACOBIAN_NCV + 1) + 3 * EXACT_JACOBIAN_MODES + 11) * sector
    check = 8 * (4 * EXACT_CHECK_BASIS + 4) * sector
    product = DTYPE_BYTES * (2 * sigma_core + nk * BASE["n_bands"] ** 4)
    assert bp.off_single == pytest.approx(product + _v_single(BASE) + max(DTYPE_BYTES * sigma_full + check, arpack))


@pytest.mark.parametrize("window", [None, 4])
def test_exact_jacobian_branch_holds_one_local_vertex_window_when_they_load_per_window(window):
    """Vertices loaded per window leave the larger one's window on the node instead of all three, nothing else."""
    held = _peaks(with_exact_jacobian=True, chunk_budgets=ChunkBudgets(exact_window=window))["exact_jacobian"]
    budgets = ChunkBudgets(exact_window=window, exact_vertices_per_window=True)
    per_window = _peaks(with_exact_jacobian=True, chunk_budgets=budgets)["exact_jacobian"]
    wp, vc, vf = BASE["niw_core"] + 1, 2 * BASE["niv_core"], 2 * BASE["niv_full"]
    f_dc, gamma, n_w = wp * vf * vc, wp * vc * vc, window or wp
    saved = DTYPE_BYTES * (f_dc + 2 * gamma - f_dc * n_w / wp)
    assert f_dc > gamma and held.giwk_shareable - per_window.giwk_shareable == pytest.approx(saved)
    assert held.baseline - per_window.baseline == pytest.approx(saved)
    assert (per_window.off_distributed, per_window.off_single) == (held.off_distributed, held.off_single)


def test_exact_jacobian_branch_is_sized_by_its_own_budget_and_not_the_loops():
    """The exact products' contraction rounds follow ChunkBudgets.exact, not the loop's sde and fq budgets."""

    def exact(**budgets):
        budgets = ChunkBudgets(exact_window=1, **budgets)
        return _peaks(with_exact_jacobian=True, chunk_budgets=budgets, **LARGE_GRID)["exact_jacobian"].off_distributed

    small = exact(exact=0)
    assert exact(exact=MAX_CHUNK_BUDGET_BYTES) > small
    assert exact(exact=0, sde=MAX_CHUNK_BUDGET_BYTES, fq=MAX_CHUNK_BUDGET_BYTES) == small


def test_exact_jacobian_branch_grows_with_the_box_and_holds_the_eigenvectors_on_rank_0():
    """The branch grows with the momentum grid and the core box; rank 0 holds the Arnoldi basis and the eigenvectors."""
    small = _peaks(with_exact_jacobian=True)["exact_jacobian"]
    for bigger in (dict(nk_tot=2 * BASE["nk_tot"], nk_irr=2 * BASE["nk_irr"]), dict(niv_core=BASE["niv_core"] + 5)):
        big = _peaks(with_exact_jacobian=True, **bigger)["exact_jacobian"]
        assert _off_node_total(big, 4) > _off_node_total(small, 4), bigger
    sector = 2 * BASE["nk_irr"] * BASE["n_bands"] ** 2 * BASE["niv_core"]
    arnoldi = np.dtype(np.float64).itemsize * (4 * EXACT_CHECK_BASIS + 4) * sector
    assert small.off_single > arnoldi and small.giwk_shareable < small.baseline


def test_rank_0_alone_holds_the_full_grid_interaction_in_every_branch(monkeypatch):
    """The full-grid interaction sits in every branch's single-rank slots and in no baseline or distributed term."""
    kw = dict(with_eliashberg=True, with_jacobian_tracker=True, with_exact_jacobian=True, niv_interp=40)
    held = _peaks(**kw)
    assert interaction_bytes(BASE["n_bands"], BASE["nk_tot"]) == _v_single(BASE)
    monkeypatch.setattr(memory_estimator, "interaction_bytes", lambda *args, **kwargs: 0.0)
    none = _peaks(**kw)
    for key, bp in held.items():
        assert (bp.baseline, bp.off_distributed, bp.on_distributed) == (
            none[key].baseline,
            none[key].off_distributed,
            none[key].on_distributed,
        ), key
        assert bp.off_single - none[key].off_single == pytest.approx(_v_single(BASE)), key
        assert bp.on_single - none[key].on_single == pytest.approx(_v_single(BASE)), key


@pytest.mark.parametrize("n_tasks, n_ranks", [(7, 3), (6, 3), (1, 3), (0, 2), (45, 4), (3276, 192)])
def test_rank_tasks_are_the_counts_the_distributor_gives_each_rank(n_tasks, n_ranks):
    """rank_tasks reproduces MpiDistributor's split, the excess on the highest ranks, ranks without tasks included."""
    sizes = MpiDistributor(ntasks=n_tasks, comm=FAKE_MPI.Comm(n_ranks)).sizes
    assert [rank_tasks(n_tasks, n_ranks, rank) for rank in range(n_ranks)] == list(sizes)


def test_a_rank_with_fewer_momenta_holds_smaller_distributed_terms():
    """A rank's own irreducible and full-BZ counts drive its distributed terms; the busiest rank is the default."""
    params = dict(nk_tot=258, with_eliashberg=True, niv_interp=40)  # irreducible 11, 11, 11, 12; full BZ 64, 64, 65, 65
    default, low, high = _peaks(**params), _peaks(**params, rank=0), _peaks(**params, rank=3)
    assert high == default
    for key in ("chi0q", "chiq_aux", "sde", "fq", "sigma_interp", "mu_update", "energies", "occupation"):
        assert low[key].off_distributed < high[key].off_distributed, key
        assert (low[key].baseline, low[key].off_single) == (high[key].baseline, high[key].off_single), key


def test_nodes_without_rank_0_hold_only_their_node_roots_share_of_the_single_slots():
    """A node without rank 0 holds the loop Sigma its root receives and shares, nothing of rank 0's own arrays."""
    kw = dict(with_eliashberg=True, with_jacobian_tracker=True, with_exact_jacobian=True, mixing_pairs=4, niv_interp=40)
    peaks = _peaks(**kw)
    sigma_full = SCALE * _giwk_rspace(BASE["nk_tot"], BASE["n_bands"], 2 * BASE["niv_cut"])
    assert peaks["sigma_loop"].remote_single == pytest.approx(2 * sigma_full)
    for key in ("mu_update", "energies", "sigma_interp"):
        assert peaks[key].remote_single == pytest.approx(sigma_full), key
    for key in ("chi0q", "chiq_aux", "sde", "occupation", "local", "fq", "lanczos"):
        assert peaks[key].remote_single == 0.0, key


@pytest.mark.parametrize(
    "grid", [{}, dict(n_bands=4, nk_tot=50**3, nk_irr=3276, niv_core=150, niv_full=300, niv_cut=460, niv_dmft=1000)]
)
def test_exact_jacobian_branch_holds_a_roots_unfolding_or_its_change_and_one_dg_chunk_off_rank_0s_node(grid):
    """Off rank 0's node the exact branch holds a root's two-window unfolding or its change beside a dG chunk."""
    params = {**BASE, **grid}
    bp = _peaks(**params, with_jacobian_tracker=True, with_exact_jacobian=True)["exact_jacobian"]
    sigma_core, sigma_full = (
        SCALE * _giwk_rspace(params["nk_tot"], params["n_bands"], 2 * params[k]) for k in ("niv_core", "niv_cut")
    )
    expected = sigma_core + max(sigma_core, min(SLICE_CHUNK_BYTES, sigma_full))
    assert expected >= 2 * sigma_core and np.allclose(bp.remote_single, expected, rtol=0, atol=1.0)
    rank_0s, other = node_total([(4, bp)], holds_rank_zero=True), node_total([(4, bp)], holds_rank_zero=False)
    assert bp.off_single > bp.remote_single
    assert np.allclose(rank_0s - other, bp.off_single - bp.remote_single, rtol=0, atol=1.0)


def test_node_total_sums_each_ranks_own_peak_and_counts_rank_0s_single_only_on_its_node():
    """Each rank adds its own baseline and transient, the node one single (remote off rank 0's node), shared once."""
    a = BranchPeak(1.0, 0.5, 2.0, 10.0, 4.0, 20.0, remote_single=3.0)
    b = replace(a, off_distributed=5.0, on_distributed=7.0)
    assert node_total([(2, a), (1, b)], holds_rank_zero=True) == 2 * 3.0 + 6.0 + 10.0 - 2 * 0.5
    assert node_total([(2, a), (1, b)], holds_rank_zero=False) == 2 * 3.0 + 6.0 + 3.0 - 2 * 0.5
    assert node_total([(2, a), (1, b)], holds_rank_zero=True, grid=True) == 2 * 5.0 + 8.0 + 20.0 - 2 * 0.5
    anywhere = replace(a, remote_single=None)
    assert node_total([(3, anywhere)], holds_rank_zero=False) == node_total([(3, anywhere)], holds_rank_zero=True)

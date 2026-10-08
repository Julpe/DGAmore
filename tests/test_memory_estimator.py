# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
"""Unit tests for the pure peak-memory estimator (no MPI, no psutil)."""

import numpy as np
import pytest

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
    dynamic_chunk_budget,
    estimate_peaks,
    exact_jacobian_budgets,
    jacobian_tracker_bytes,
    max_chunk_budget,
)
from dgamore.n_point_base import DTYPE, SpinChannel
from tests.conftest import traced_peak

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


def _rank_base(params):
    return RANK_BASELINE_BYTES + SCALE * params["nk_tot"] * params["n_bands"] ** 4


# the node total used by the driver: every rank holds the branch baseline + the distributed transient, plus one single
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
        base = _rank_base(BASE)  # every rank's footprint and full-grid interaction sit in every baseline
        assert bp.baseline > base if key in ("chi0q", "chiq_aux", "sde", "fq") else bp.baseline == base
        assert bp.off_distributed + bp.off_single > 0


def test_chi0q_fast_path_is_distributed_on_multi_rank_runs():
    """The chi0q fast path is column-distributed on multi-rank runs; only single-rank uses the rank-0 build."""
    multi = _peaks()["chi0q"]
    assert multi.off_distributed > 0.0 and multi.off_single == 0.0
    single = _peaks(n_ranks=1)["chi0q"]
    assert single.off_distributed == 0.0 and single.off_single > 0.0
    assert multi.on_distributed > 0.0 and multi.on_single == 0.0


def test_chiq_aux_off_has_distributed_block_and_no_single_rank_gather():
    """The chiq_aux fast path holds a per-rank block; the irr->full-BZ map is p2p, so no single-rank gather remains."""
    bp = _peaks()["chiq_aux"]
    assert bp.off_distributed > 0.0  # per-rank two-fermion block
    assert bp.off_single == 0.0


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
    base = _rank_base(TINY)
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
    base = _rank_base(TINY)
    assert peaks["chiq_aux"].baseline == pytest.approx(SCALE * (giwk + sigma + local_vertex) + base)
    assert peaks["sde"].baseline == pytest.approx(SCALE * (2 * giwk + sigma) + base)
    assert peaks["chiq_aux"].giwk_shareable == pytest.approx(SCALE * (giwk + sigma + local_vertex))
    assert peaks["sde"].giwk_shareable == pytest.approx(SCALE * (2 * giwk + sigma))


def test_giwk_shareable_is_every_array_of_each_sde_section_baseline():
    """The loop Sigma is node-shared like the Green's functions, so only the per-rank footprint stays unshared."""
    peaks = _peaks(with_eliashberg=True)
    for key in ("chi0q", "chiq_aux", "sde"):
        assert peaks[key].giwk_shareable == pytest.approx(peaks[key].baseline - _rank_base(BASE))


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
        assert peaks[key].baseline == pytest.approx(_rank_base(BASE) + shared)
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


def test_chiq_aux_block_is_the_resident_one_fermion_blocks_plus_two_compound_slices():
    """The chiq_aux transient is three one-fermion blocks (sum, kernel, inverse bubble) plus the per-slice buffers."""
    nb, wp, vc = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_core"]
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    expected = SCALE * 3 * qi * nb**4 * wp * vc + OVERHEAD_FACTOR * _chiq_aux_transient()
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
    expected = SCALE * (bubble_irr + fft_buffers + gf_copies + g_center)
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
    assert small == big == pytest.approx(SCALE * 2 * TINY["nk_tot"] * nb**2 * 2 * TINY["niv_core"])


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
    assert bp.baseline == pytest.approx(_rank_base(params)) and bp.giwk_shareable == 0.0
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
    assert plain.off_single == 0.0 and plain.baseline == pytest.approx(_rank_base(TINY))
    spectrum = estimate_peaks(**TINY, do_spectrum_dmft=True)["occupation"]
    dyson = memory_estimator._green_function_transients(nk, nb, box)[0]
    reload = SCALE * nk * nb**2 * (box + 2 * 2 * TINY["niv_cut"] + 2 * TINY["niv_core"])
    assert spectrum.off_single == pytest.approx(max(SCALE * nk * nb**2 * box + dyson, reload))
    assert spectrum.off_distributed == plain.off_distributed


def test_occupation_branch_holds_the_warm_start_on_rank0_and_its_mu_search_on_every_rank():
    """On a warm start rank 0 extends the whole-grid starting iterate onto the DMFT box next to its niv_cut copy and
    the occupation chunks, and every rank searches mu on its momenta."""
    nb, nk, box = TINY["n_bands"], TINY["nk_tot"], 2 * TINY["niv_dmft"]
    peaks = estimate_peaks(**TINY, warm_start=True)
    warm = peaks["occupation"]
    occupation = memory_estimator._green_function_transients(nk, nb, box)[1]
    sigma_full = nk * nb**2 * 2 * TINY["niv_cut"]
    assert warm.off_single == pytest.approx(SCALE * (nk * nb**2 * box + sigma_full) + occupation)
    assert warm.off_distributed >= peaks["mu_update"].off_distributed
    assert estimate_peaks(**TINY)["occupation"].off_single == 0.0


def test_sigma_loop_is_rank0_only_with_the_linear_mix_copies_or_the_accelerated_solve():
    """sigma_loop is rank 0's step alone: two niv_cut Sigmas plus the linear mix or the accelerated solve."""
    nk, nb = TINY["nk_tot"], TINY["n_bands"]
    core = nk * nb**2 * 2 * TINY["niv_core"]
    full = nk * nb**2 * 2 * TINY["niv_cut"]
    linear = SCALE * 2 * full + SCALE * 3 * full
    bp = estimate_peaks(**TINY)["sigma_loop"]
    assert bp.baseline == pytest.approx(_rank_base(TINY)) and bp.giwk_shareable == 0.0
    assert bp.off_distributed == 0.0 and bp.off_single == pytest.approx(linear)
    assert (bp.on_distributed, bp.on_single) == (bp.off_distributed, bp.off_single)
    solve = estimate_peaks(**TINY, mixing_pairs=4)["sigma_loop"].off_single
    assert solve == pytest.approx(SCALE * (8 * core + 2 * full) + 8 * core * (9 * 3 + 10))
    one_pair = estimate_peaks(**TINY, mixing_pairs=1)["sigma_loop"].off_single
    assert one_pair == pytest.approx(SCALE * 2 * core + linear)


def test_mu_update_holds_two_complex128_green_function_slices_per_rank_next_to_rank0_sigmas():
    """mu_update: every rank two complex128 G arrays on its momentum share, rank 0 two niv_cut Sigmas and history."""
    nk, nb, ranks = TINY["nk_tot"], TINY["n_bands"], TINY["n_ranks"]
    per_k = nb**2 * 2 * TINY["niv_cut"]
    bp = estimate_peaks(**TINY)["mu_update"]
    assert bp.baseline == pytest.approx(_rank_base(TINY)) and bp.giwk_shareable == 0.0
    assert bp.off_distributed == pytest.approx(16 * 2 * -(-nk // ranks) * per_k)
    assert bp.off_single == pytest.approx(SCALE * 2 * nk * per_k)
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
    assert bp.off_single == pytest.approx(SCALE * (full + source) + max(16 * (10 * source + 4 * target), unfold))
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
    expected = SCALE * (LANCZOS_VERTEX_FACTOR * vertex + chi0 + arpack + giwk_dga)
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
    assert bp.baseline == pytest.approx(_rank_base(BASE)) and bp.giwk_shareable == 0.0 and bp.off_distributed == 0.0
    assert bp.off_single == bp.on_single > 0.0
    assert _peaks(n_ranks=16, niv_full=niv_full, symmetrize_orbitals=symmetrize)["local"].off_single == bp.off_single
    two_bands = _peaks(n_bands=2, niv_full=niv_full, symmetrize_orbitals=symmetrize)["local"].off_single
    assert two_bands == pytest.approx(16 * bp.off_single)
    wp, vc, vf = BASE["niw_core"] + 1, 2 * BASE["niv_core"], 2 * niv_full
    l_core, f_full = wp * vc * vc, wp * vf * vc
    expected = SCALE * max(4 * l_core + (3 if symmetrize else 2) * f_full, 6 * l_core + f_full)
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
    # window, predecessor, estimate, snapshot and held flips in the storage precision; four tall float64 arrays
    resident = (TRACKER_PAIRS * DTYPE_BYTES + DTYPE_BYTES * k + 3 * DTYPE_BYTES * k + 4 * 8 * k) * sector
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
        expected = history + resident + 2 * core + DTYPE_BYTES * 2 * sigma_full + step
        assert on["sigma_loop"].off_single == pytest.approx(expected)
        assert (step == solve) == (pairs == 40)


def test_jacobian_tracker_residents_count_97_sector_columns_with_exact_checks_and_55_without_in_complex64():
    """With exact checks the tracker holds 97 float64 sector columns (complex64 window, file, Ritz sets), 55 without."""
    nk_irr, nb, nv = BASE["nk_irr"], BASE["n_bands"], 2 * BASE["niv_core"]
    column, storage = 8 * nk_irr * nb**2 * nv, DTYPE_BYTES / np.dtype(np.complex64).itemsize
    # the window, the predecessor file, the Ritz sets and the check candidate double in a complex128 run
    exact = (97 + (7 + 12 + 38 + 2) * (storage - 1)) * column
    secant = (55 + (7 + 6 + 18) * (storage - 1)) * column
    assert jacobian_tracker_bytes(nk_irr, nb, nv, with_exact_jacobian=True) == (exact, 42 * column)
    assert jacobian_tracker_bytes(nk_irr, nb, nv) == (secant, 42 * column)
    both = _peaks(with_jacobian_tracker=True, with_exact_jacobian=True, overhead=1.0)
    tracker = _peaks(with_jacobian_tracker=True, overhead=1.0)
    for key in ("chi0q", "chiq_aux", "sde", "sigma_loop"):
        assert both[key].off_single == pytest.approx(tracker[key].off_single + exact - secant), key


# a grid whose core blocks dwarf the per-slice Bethe-Salpeter solve's fixed transient, so the solve's phase stays small
LARGE_GRID = dict(nk_tot=128 * 128, nk_irr=2145)


@pytest.mark.parametrize("niv_full, width", [(30, 1), (30, 2), (150, 1), (150, 2)])
def test_exact_jacobian_bubble_phase_is_the_response_beside_the_second_bubbles_buffers_or_the_dc_attachment(
    niv_full, width
):
    """The bubble phase is max(full + max(half a full block, 2 core), parts + 2 core) beside earlier columns' parts."""
    chunk_budgets = ChunkBudgets(exact=0, exact_block=width)
    params = dict(niv_full=niv_full, niv_cut=niv_full + 40, niv_dmft=400, overhead=1.0, chunk_budgets=chunk_budgets)
    bp = _peaks(with_exact_jacobian=True, **LARGE_GRID, **params)["exact_jacobian"]
    qi = -(-LARGE_GRID["nk_irr"] // BASE["n_ranks"])
    core = DTYPE_BYTES * qi * BASE["n_bands"] ** 4 * (BASE["niw_core"] + 1) * 2 * BASE["niv_core"]
    full, parts = core * niv_full / BASE["niv_core"], (2 + 1 / BASE["niv_core"]) * core
    bubble = max(full + max(0.5 * full, 2 * core), parts + 2 * core) + (width - 1) * parts
    assert bp.off_distributed == pytest.approx(4 * core + bubble)


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
    assert bp.off_single == pytest.approx(product + max(in_loop, end_of_rung))
    assert (end_of_rung > in_loop) == (ncv > 100)


@pytest.mark.parametrize("width_cost, width", [(1 / 8, 4), (1 / 3, 2), (1.0, None)])
def test_exact_jacobian_budgets_take_the_widest_block_that_fits_and_the_budget_it_leaves(width_cost, width):
    """The exact products take the widest block that fits at the floor and the budget left, apart from the loop's."""
    line, loop = 3 * 1024**3, ChunkBudgets(sde=2, fq=3)
    budgets = exact_jacobian_budgets(
        lambda trial: 3 * trial.exact + width_cost * line * trial.exact_block <= line, loop
    )
    assert (budgets.sde, budgets.fq) == (2, 3)
    if width is None:
        assert budgets == ChunkBudgets(2, 3, SLICE_CHUNK_BYTES, 1, True, True)
    else:
        left = (line - width_cost * line * width) / 3
        assert budgets.exact_block == width and not budgets.exact_vertices_per_phase
        assert left - 2**20 <= budgets.exact <= left


@pytest.mark.parametrize(
    "local, vrg, residency, width",
    [
        (0.1, 0.1, (False, False), 4),
        (0.45, 0.1, (False, False), 2),
        (0.7, 0.1, (True, False), 4),
        (0.4, 0.4, (True, False), 3),
        (0.1, 0.7, (False, True), 4),
        (0.7, 0.7, (True, True), 4),
        (0.1, 0.0, (False, False), 4),
        (0.6, 0.0, (False, False), 1),
        (0.7, 0.0, (True, False), 4),
    ],
)
def test_exact_jacobian_budgets_take_the_widest_block_of_the_cheapest_residency_that_fits(local, vrg, residency, width):
    """Both vertex sets held, local ones per phase, three-leg ones per group, then both: the first that fits at all."""
    line = 3 * 1024**3

    def fits(trial: ChunkBudgets) -> bool:
        """Whether the trial fits the line, each vertex set held unless loaded per phase or recomputed per group."""
        held = (0.0 if trial.exact_vertices_per_phase else local) + (0.0 if trial.exact_vrg_per_group else vrg)
        return 3 * trial.exact + (0.1 * trial.exact_block + held) * line <= line

    budgets = exact_jacobian_budgets(fits, ChunkBudgets())
    assert (budgets.exact_vertices_per_phase, budgets.exact_vrg_per_group) == residency
    assert budgets.exact_block == width and fits(budgets)


@pytest.mark.parametrize("budget, group_momenta", [(0, 1), (MAX_CHUNK_BUDGET_BYTES, 12)])
def test_exact_jacobian_branch_recomputing_the_three_leg_vertices_holds_two_core_blocks_fewer(budget, group_momenta):
    """Per-group three-leg vertices hold two residents fewer and add one momentum group's slice to the solve."""
    held = _peaks(with_exact_jacobian=True, overhead=1.0, chunk_budgets=ChunkBudgets(exact=budget))["exact_jacobian"]
    budgets = ChunkBudgets(exact=budget, exact_vrg_per_group=True)
    per_group = _peaks(with_exact_jacobian=True, overhead=1.0, chunk_budgets=budgets)["exact_jacobian"]
    qi = -(-BASE["nk_irr"] // BASE["n_ranks"])
    core = DTYPE_BYTES * qi * BASE["n_bands"] ** 4 * (BASE["niw_core"] + 1) * 2 * BASE["niv_core"]
    # one momentum per group at the floor, the rank's twelve at the cap
    assert held.off_distributed - per_group.off_distributed == pytest.approx((2 - group_momenta / qi) * core)
    assert (held.baseline, held.off_single) == (per_group.baseline, per_group.off_single)


@pytest.mark.parametrize("budget, width", [(0, 1), (0, 4), (MAX_CHUNK_BUDGET_BYTES, 1), (MAX_CHUNK_BUDGET_BYTES, 2)])
def test_exact_jacobian_solve_phase_is_the_parts_six_plus_two_m_group_slices_and_the_per_slice_transient(budget, width):
    """The solve phase: m column parts, 6 + 2 m group slices, the sum's slice transient and 4 m - 3 more columns."""
    bp = _peaks(with_exact_jacobian=True, overhead=1.0, chunk_budgets=ChunkBudgets(exact=budget, exact_block=width))
    nb, vc = BASE["n_bands"], 2 * BASE["niv_core"]
    qi, compound = -(-BASE["nk_irr"] // BASE["n_ranks"]), nb * nb * vc
    core = DTYPE_BYTES * qi * nb**4 * (BASE["niw_core"] + 1) * vc
    group = core / qi if budget == 0 else core
    # the frequency sum's two slices, 128 + 4 nb^2 columns and two tiles, then four nb^2 columns per right-hand side
    solve = DTYPE_BYTES * (2 * compound**2 + (128 + 4 * nb**2) * compound + 2 * 256**2)
    solve += DTYPE_BYTES * (4 * width - 3) * nb**2 * compound
    kernel = width * (2 + 2 / vc) * core + (6 + 2 * width) * group + solve
    assert bp["exact_jacobian"].off_distributed == pytest.approx(4 * core + kernel)


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


@pytest.mark.parametrize("per_group, residents", [(False, 4), (True, 2)])
def test_exact_jacobian_build_phase_holds_three_core_blocks_six_susceptibilities_and_its_own_three_leg_vertex(
    per_group, residents
):
    """The build's last channel holds 3 core blocks and 6 susceptibilities, and its three-leg vertex per group."""
    budgets = ChunkBudgets(exact=0, exact_vrg_per_group=per_group)
    bp = _peaks(with_exact_jacobian=True, overhead=1.0, chunk_budgets=budgets, **LARGE_GRID)["exact_jacobian"]
    vc = 2 * BASE["niv_core"]
    qi = -(-LARGE_GRID["nk_irr"] // BASE["n_ranks"])
    core = DTYPE_BYTES * qi * BASE["n_bands"] ** 4 * (BASE["niw_core"] + 1) * vc
    attachment, build = (4 + 2 / vc) * core, (3 + per_group + 6 / vc) * core
    assert bp.off_distributed == pytest.approx(residents * core + max(attachment, build))
    assert (build > attachment) == per_group


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


def test_exact_jacobian_branch_counts_its_node_windows_and_rank_0s_check_exactly():
    """Node: loop Sigma, G, dG, G_R, three vertices, the block, bubble windows; rank 0: a product and its solver."""
    bp = _peaks(with_exact_jacobian=True, overhead=1.0)["exact_jacobian"]
    nk, niv, niw, nivf, cut = BASE["nk_tot"], BASE["niv_core"], BASE["niw_core"], BASE["niv_full"], BASE["niv_cut"]
    sigma_full, sigma_core, sector = nk * 2 * cut, nk * 2 * niv, 2 * BASE["nk_irr"] * niv
    vertices = (niw + 1) * 2 * nivf * 2 * niv + 2 * (niw + 1) * (2 * niv) ** 2
    windows = DTYPE_BYTES * (3 * sigma_full + nk * 2 * (niv + niw) + vertices) + 8 * EXACT_CHECK_BASIS * sector
    bubble_windows = DTYPE_BYTES * 2 * nk * 2 * (nivf + niw)
    assert bp.giwk_shareable == pytest.approx(windows + bubble_windows)
    assert bp.baseline == pytest.approx(windows + bubble_windows + _rank_base(BASE))
    arpack = 8 * (2 * (EXACT_JACOBIAN_NCV + 1) + 3 * EXACT_JACOBIAN_MODES + 11) * sector
    check = 8 * (4 * EXACT_CHECK_BASIS + 4) * sector
    product = DTYPE_BYTES * (2 * sigma_core + nk * BASE["n_bands"] ** 4)
    assert bp.off_single == pytest.approx(product + max(DTYPE_BYTES * sigma_full + check, arpack))


def test_exact_jacobian_branch_holds_one_local_vertex_when_they_load_per_phase():
    """Vertices loaded per phase leave the larger one on the node instead of all three, and change nothing else."""
    held = _peaks(with_exact_jacobian=True)["exact_jacobian"]
    per_phase = _peaks(with_exact_jacobian=True, chunk_budgets=ChunkBudgets(exact_vertices_per_phase=True))
    per_phase = per_phase["exact_jacobian"]
    wp, vc, vf = BASE["niw_core"] + 1, 2 * BASE["niv_core"], 2 * BASE["niv_full"]
    f_dc, gamma = wp * vf * vc, wp * vc * vc
    assert f_dc > gamma and held.giwk_shareable - per_phase.giwk_shareable == pytest.approx(DTYPE_BYTES * 2 * gamma)
    assert held.baseline - per_phase.baseline == pytest.approx(held.giwk_shareable - per_phase.giwk_shareable)
    assert (per_phase.off_distributed, per_phase.off_single) == (held.off_distributed, held.off_single)


def test_exact_jacobian_branch_is_sized_by_its_own_budget_and_not_the_loops():
    """The exact products' transients follow ChunkBudgets.exact, not the loop's sde and fq budgets."""

    def exact(**budgets):
        return _peaks(with_exact_jacobian=True, chunk_budgets=ChunkBudgets(**budgets))["exact_jacobian"].off_distributed

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

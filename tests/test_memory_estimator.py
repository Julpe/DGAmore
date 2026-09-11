# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
"""Unit tests for the pure peak-memory estimator (no MPI, no psutil)."""

import numpy as np
import pytest

import dgamore.memory_estimator as memory_estimator
from dgamore.memory_estimator import (
    ARPACK_EXTRA_VECTORS,
    CHI0Q_IFFTN_TRANSIENT_FACTOR,
    DTYPE_BYTES,
    FQ_MATMUL_FACTOR,
    LANCZOS_VERTEX_FACTOR,
    MAX_CHUNK_BUDGET_BYTES,
    MAX_SLICE_CHUNK_BYTES,
    OVERHEAD_FACTOR,
    RANK_BASELINE_BYTES,
    SDE_CHUNK_FACTOR,
    SDE_HEADROOM_SHARE,
    SLICE_CHUNK_BYTES,
    BranchPeak,
    ChunkBudgets,
    dynamic_chunk_budget,
    estimate_peaks,
    max_chunk_budget,
)
from dgamore.n_point_base import DTYPE

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


# the node total used by the driver: every rank holds the branch baseline + the distributed transient, plus one single
def _off_node_total(bp: BranchPeak, r):
    return r * (bp.baseline + bp.off_distributed) + bp.off_single


def test_constants():
    """DTYPE_BYTES tracks the global storage dtype and OVERHEAD_FACTOR defaults to 1.0 (no extra margin)."""
    assert DTYPE_BYTES == np.dtype(DTYPE).itemsize  # single source of truth: derived from the global storage dtype
    assert OVERHEAD_FACTOR == pytest.approx(1.0)


def test_keys_without_eliashberg():
    """Without Eliashberg the estimator reports the chi0q, chiq_aux, sde, sigma_loop and local branches."""
    assert set(_peaks(with_eliashberg=False)) == {"chi0q", "chiq_aux", "sde", "sigma_loop", "local"}


def test_keys_with_eliashberg():
    """With Eliashberg the estimator adds the fq and lanczos branches."""
    assert set(_peaks(with_eliashberg=True)) == {"chi0q", "chiq_aux", "sde", "sigma_loop", "fq", "lanczos", "local"}


def test_every_branch_has_positive_baseline_and_off_transient():
    """SDE-section branches carry a positive baseline, Eliashberg branches none, and every branch has a fast"""
    peaks = _peaks(with_eliashberg=True)
    for key, bp in peaks.items():
        assert isinstance(bp, BranchPeak)
        base = RANK_BASELINE_BYTES  # every rank's interpreter/library footprint sits in every baseline
        assert bp.baseline > base if key in ("chi0q", "chiq_aux", "sde") else bp.baseline == base
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
    base = RANK_BASELINE_BYTES
    assert estimate_peaks(**TINY)["chi0q"].baseline == pytest.approx(two_point + g_r_windows + base)
    assert estimate_peaks(**{**TINY, "n_ranks": 1})["chi0q"].baseline == pytest.approx(two_point + base)


def test_sde_section_baseline_uses_post_bubble_windows():
    """The chiq_aux baseline holds giwk at niv_core+niw_core and the asymmetric local vertex; sde the R-space G copy."""
    nk, nb = TINY["nk_tot"], TINY["n_bands"]
    giwk = nk * nb**2 * 2 * (TINY["niv_core"] + TINY["niw_core"])
    sigma_old = nk * nb**2 * 2 * TINY["niv_core"]
    peaks = estimate_peaks(**TINY)
    vf, vc = 2 * TINY["niv_full"], 2 * TINY["niv_core"]
    local_vertex = TINY["n_bands"] ** 4 * (TINY["niw_core"] + 1) * max(vf * vc, vc * vc)
    base = RANK_BASELINE_BYTES
    assert peaks["chiq_aux"].baseline == pytest.approx(SCALE * (giwk + sigma_old + local_vertex) + base)
    assert peaks["sde"].baseline == pytest.approx(SCALE * (2 * giwk + sigma_old) + base)
    assert peaks["chiq_aux"].giwk_shareable == pytest.approx(SCALE * (giwk + local_vertex))
    assert peaks["sde"].giwk_shareable == pytest.approx(SCALE * 2 * giwk)


def test_giwk_shareable_is_the_giwk_part_of_each_sde_section_baseline():
    """giwk_shareable covers exactly the Green's-function part of the chi0q/chiq_aux/sde baselines."""
    peaks = _peaks(with_eliashberg=True)
    sigma_old = SCALE * BASE["nk_tot"] * BASE["n_bands"] ** 2 * (2 * BASE["niv_cut"])
    assert peaks["chi0q"].giwk_shareable == pytest.approx(peaks["chi0q"].baseline - sigma_old - RANK_BASELINE_BYTES)
    for key in ("chi0q", "chiq_aux", "sde"):
        assert 0 < peaks[key].giwk_shareable < peaks[key].baseline


def test_eliashberg_branches_are_not_giwk_shareable():
    """The fq/lanczos branches carry no baseline (sigma_dga freed, giwk_dga on the bubble rank), nothing node-shared."""
    peaks = _peaks(with_eliashberg=True)
    giwk_dga = SCALE * BASE["nk_tot"] * BASE["n_bands"] ** 2 * 2 * BASE["niv_cut"]
    for key in ("fq", "lanczos"):
        assert peaks[key].giwk_shareable == 0.0
        assert peaks[key].baseline == RANK_BASELINE_BYTES
        assert peaks[key].off_single >= giwk_dga
        assert peaks[key].on_single >= giwk_dga


def test_bubble_baseline_depends_on_niv_cut_not_niv_full():
    """The single-rank chi0q baseline tracks niv_cut only; the multi-rank one also tracks niv_full via the R-space G."""
    assert _peaks(niv_full=40, n_ranks=1)["chi0q"].baseline == pytest.approx(
        _peaks(niv_full=400, n_ranks=1)["chi0q"].baseline
    )
    assert _peaks(niv_full=40)["chi0q"].baseline < _peaks(niv_full=400)["chi0q"].baseline
    assert _peaks(niv_cut=80)["chi0q"].baseline != pytest.approx(_peaks(niv_cut=800)["chi0q"].baseline)


def _tiny_chunk_sizes():
    """(rank block, one momentum's box, one (q, w) compound slice) of the TINY two-fermion objects, in bytes."""
    nb, wp, vc = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_core"]
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    return DTYPE_BYTES * qi * nb**4 * wp * vc * vc, DTYPE_BYTES * nb**4 * wp * vc * vc, DTYPE_BYTES * nb**4 * vc * vc


def _chiq_aux_transient(chunk):
    """The modeled aux-chi chunk transient: window + sliced local vertex + LU slice copy + summed output."""
    _, per_q_box, one_slice = _tiny_chunk_sizes()
    return chunk + min(chunk, per_q_box) + one_slice + chunk // (2 * TINY["niv_core"])


def _fq_transient(chunk):
    """The modeled pairing-vertex chunk transient: max(matmul pair, window + two pp cuts) + sliced local vertex."""
    _, per_q_box, _ = _tiny_chunk_sizes()
    pp_ratio = (2 * TINY["niv_pp"] / (2 * TINY["niv_core"])) ** 2
    return max(FQ_MATMUL_FACTOR * chunk, chunk + int(2 * pp_ratio * chunk)) + min(chunk, per_q_box)


def test_chiq_aux_block_is_the_resident_one_fermion_blocks_plus_the_chunk_transient():
    """The chiq_aux transient holds three one-fermion blocks (sum, kernel, inverse bubble) plus the chunk transient."""
    nb, wp, vc = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_core"]
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    block, _, _ = _tiny_chunk_sizes()
    assert block < SLICE_CHUNK_BYTES  # the floor budget already walks the tiny block in one chunk
    expected = SCALE * 3 * qi * nb**4 * wp * vc + OVERHEAD_FACTOR * _chiq_aux_transient(block)
    bp = estimate_peaks(**TINY)["chiq_aux"]
    assert bp.off_distributed == pytest.approx(expected)
    assert bp.on_distributed == bp.off_distributed


def test_chiq_aux_chunk_term_follows_the_passed_budget_up_to_the_rank_block():
    """A larger aux-chi chunk budget raises only the chiq_aux transient, by the modeled per-chunk temporaries."""
    block, _, _ = _tiny_chunk_sizes()
    at = {b: estimate_peaks(**TINY, chunk_budgets=ChunkBudgets(chiq_aux=b)) for b in (block // 4, block // 2, block)}
    quarter, half, whole = (at[b]["chiq_aux"].off_distributed for b in (block // 4, block // 2, block))
    assert half - quarter == pytest.approx(
        OVERHEAD_FACTOR * (_chiq_aux_transient(block // 2) - _chiq_aux_transient(block // 4))
    )
    assert whole - half == pytest.approx(
        OVERHEAD_FACTOR * (_chiq_aux_transient(block) - _chiq_aux_transient(block // 2))
    )
    capped = estimate_peaks(**TINY, chunk_budgets=ChunkBudgets(chiq_aux=10 * block))["chiq_aux"]
    assert capped.off_distributed == pytest.approx(whole)
    for key in ("chi0q", "sde", "sigma_loop", "local"):
        assert at[block // 4][key] == at[block][key]


def test_chiq_aux_transient_counts_the_sliced_local_vertex_at_most_once_per_momentum_box():
    """Above one momentum's box the sliced local vertex stops growing with the chunk (q-groups share one window)."""
    block, per_q_box, one_slice = _tiny_chunk_sizes()
    assert per_q_box < block
    below, above = _chiq_aux_transient(per_q_box // 2), _chiq_aux_transient(2 * per_q_box)
    assert below == pytest.approx(2 * (per_q_box // 2) + one_slice + (per_q_box // 2) // (2 * TINY["niv_core"]))
    assert above == pytest.approx(3 * per_q_box + one_slice + (2 * per_q_box) // (2 * TINY["niv_core"]))


def test_sde_chunk_term_follows_the_passed_budget_between_one_row_and_the_full_bz_block():
    """The sde exchange transient follows its budget linearly between one full-BZ bosonic row and the whole block."""
    nb, wp, vc = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_core"]
    qt = -(-TINY["nk_tot"] // TINY["n_ranks"])
    row, block = DTYPE_BYTES * qt * nb**4 * vc, DTYPE_BYTES * qt * nb**4 * wp * vc
    slope = OVERHEAD_FACTOR * SDE_CHUNK_FACTOR
    two, three = (estimate_peaks(**TINY, chunk_budgets=ChunkBudgets(sde=n * row))["sde"] for n in (2, 3))
    assert three.off_distributed - two.off_distributed == pytest.approx(slope * row)
    floor, zero = (estimate_peaks(**TINY, chunk_budgets=ChunkBudgets(sde=b))["sde"] for b in (row, 0))
    assert floor.off_distributed == pytest.approx(zero.off_distributed)
    capped = estimate_peaks(**TINY, chunk_budgets=ChunkBudgets(sde=10 * block))["sde"]
    assert capped.off_distributed == pytest.approx(
        estimate_peaks(**TINY, chunk_budgets=ChunkBudgets(sde=block))["sde"].off_distributed
    )


def test_fq_chunk_term_follows_the_passed_budget_up_to_the_rank_block():
    """The pairing-vertex transient follows its modeled per-chunk temporaries, capped at the rank block."""
    nb, wp, vc, vpp = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_core"], 2 * TINY["niv_pp"]
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    block, _, _ = _tiny_chunk_sizes()
    params = {**TINY, "with_eliashberg": True}
    residents = SCALE * (qi * nb**4 * vpp * vpp + 3 * qi * nb**4 * wp * vc)
    for budget in (block // 4, block // 2, block):
        bp = estimate_peaks(**params, chunk_budgets=ChunkBudgets(fq=budget))["fq"]
        assert bp.off_distributed == pytest.approx(residents + OVERHEAD_FACTOR * _fq_transient(budget))
    capped = estimate_peaks(**params, chunk_budgets=ChunkBudgets(fq=10 * block))["fq"]
    assert capped.off_distributed == pytest.approx(residents + OVERHEAD_FACTOR * _fq_transient(block))


def test_chiq_aux_chunk_term_never_drops_below_one_compound_slice():
    """The build walks at least one (q, w) slice per chunk, so a zero budget still models one slice's transient."""
    nb, wp, vc = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_core"]
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    _, _, one_slice = _tiny_chunk_sizes()
    residents = SCALE * 3 * qi * nb**4 * wp * vc
    bp = estimate_peaks(**TINY, chunk_budgets=ChunkBudgets(chiq_aux=0))["chiq_aux"]
    assert bp.off_distributed == pytest.approx(residents + OVERHEAD_FACTOR * _chiq_aux_transient(one_slice))


def test_max_chunk_budget_bisects_to_the_largest_fitting_budget():
    """The budget search returns the largest budget a monotone fit predicate accepts, to within 1 MiB."""
    limit = 3 * 2**30 + 12345
    found = max_chunk_budget(lambda budget: budget <= limit)
    assert limit - 2**20 <= found <= limit
    assert max_chunk_budget(lambda budget: True) == MAX_CHUNK_BUDGET_BYTES
    assert max_chunk_budget(lambda budget: False) == SLICE_CHUNK_BYTES
    assert max_chunk_budget(lambda budget: budget <= SLICE_CHUNK_BYTES) == SLICE_CHUNK_BYTES
    assert max_chunk_budget(lambda budget: budget <= 2**31, upper=2**30) == 2**30
    assert 0.0 < SDE_HEADROOM_SHARE < 1.0
    assert ChunkBudgets() == ChunkBudgets(SLICE_CHUNK_BYTES, SLICE_CHUNK_BYTES, SLICE_CHUNK_BYTES)
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
    """The multi-rank chi0q fast peak is the per-rank irr result slice plus the bounded sub-chunk group."""
    nb, wp, vf = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_full"]
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    expected = SCALE * 1.5 * qi * nb**4 * wp * vf
    assert estimate_peaks(**TINY)["chi0q"].off_distributed == pytest.approx(expected)


def test_sde_transient_is_the_irr_kernel_plus_the_bounded_exchange_chunks():
    """The sde transient holds the retained irr kernel plus the chunk-capped exchanged full-BZ w-slices."""
    nb, wp, vc = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_core"]
    qt = -(-TINY["nk_tot"] // TINY["n_ranks"])
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    chunk = min(SLICE_CHUNK_BYTES, DTYPE_BYTES * qt * nb**4 * wp * vc)
    expected = SCALE * qi * nb**4 * wp * vc + OVERHEAD_FACTOR * SDE_CHUNK_FACTOR * chunk
    assert estimate_peaks(**TINY)["sde"].off_distributed == pytest.approx(expected)


def test_sde_single_covers_the_rank0_occupation_step():
    """The sde single-rank slot grows to the DMFT-box sigma/giwk pair once that exceeds the finalize buffers."""
    small = estimate_peaks(**{**TINY, "niv_dmft": TINY["niv_core"]})["sde"].off_single
    big = estimate_peaks(**{**TINY, "niv_dmft": 100 * TINY["niv_core"]})["sde"].off_single
    nb = TINY["n_bands"]
    assert big == pytest.approx(SCALE * 2 * TINY["nk_tot"] * nb**2 * 2 * 100 * TINY["niv_core"])
    assert small < big


def test_sde_chunk_term_is_capped_by_the_byte_budget():
    """Once the per-rank full-BZ kernel exceeds the byte budget, the sde transient stops growing with the grid."""
    nb, vc = TINY["n_bands"], 2 * TINY["niv_core"]
    qt = -(-4 * TINY["nk_tot"] // TINY["n_ranks"])
    budgets = ChunkBudgets(sde=DTYPE_BYTES * qt * nb**4 * vc)  # one bosonic row of the larger grid
    small = estimate_peaks(**TINY, chunk_budgets=budgets)["sde"].off_distributed
    big = estimate_peaks(**{**TINY, "nk_tot": 4 * TINY["nk_tot"]}, chunk_budgets=budgets)["sde"].off_distributed
    assert big == pytest.approx(small)


def test_sigma_loop_counts_the_private_proposal_per_rank_and_the_rank0_mixing_copies():
    """sigma_loop counts the per-rank proposal and tail concatenation plus rank 0's previous-iterate and mix copies."""
    nk, nb = TINY["nk_tot"], TINY["n_bands"]
    core = nk * nb**2 * 2 * TINY["niv_core"]
    full = nk * nb**2 * 2 * TINY["niv_cut"]
    bp = estimate_peaks(**TINY)["sigma_loop"]
    assert bp.baseline == RANK_BASELINE_BYTES and bp.giwk_shareable == 0.0
    assert bp.off_distributed == pytest.approx(SCALE * max(3 * core, core + full))
    assert bp.off_single == pytest.approx(SCALE * (core + 4 * full))
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
    arpack = (max(2 * 1 + 1, 20) + ARPACK_EXTRA_VECTORS) * p["nk_tot"] * nb**2 * vpp
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


def test_save_pairing_vertex_enters_the_single_rank_gather_peak():
    """save_pairing_vertex gathers both irr-BZ pp vertices on one rank, raising the solver peak when it dominates."""
    p = {**TINY, "with_eliashberg": True}
    with_save = estimate_peaks(**{**p, "save_pairing_vertex": True})["lanczos"]
    without = estimate_peaks(**{**p, "save_pairing_vertex": False})["lanczos"]
    assert with_save.off_single >= without.off_single
    assert with_save.on_single > without.on_single  # the gather is a single-rank peak of the grid variant too


def test_local_step_is_flagless_single_rank_and_band_heavy():
    """The local branch is verify-only, rank-independent and nb^4-heavy (both channels + the shell transient)."""
    from dgamore.memory_estimator import LOCAL_SHELL_INVERT_FACTOR

    bp = _peaks()["local"]
    assert bp.baseline == RANK_BASELINE_BYTES and bp.giwk_shareable == 0.0 and bp.off_distributed == 0.0
    assert bp.off_single == bp.on_single > 0.0
    assert _peaks(n_ranks=16)["local"].off_single == pytest.approx(bp.off_single)
    assert _peaks(n_bands=2)["local"].off_single == pytest.approx(16 * bp.off_single)
    wp, vc, vf = BASE["niw_core"] + 1, 2 * BASE["niv_core"], 2 * BASE["niv_full"]
    l_core, l_full = wp * vc * vc, wp * vf * vf
    expected = SCALE * (2 * (2 * l_core + l_full) + 2 * l_core + LOCAL_SHELL_INVERT_FACTOR * l_full)
    assert bp.off_single == pytest.approx(expected)


def test_dynamic_chunk_budget_scales_floors_and_caps():
    """The dynamic chunk budget grows with free memory per node rank, floored and capped at the class constants."""
    floor, cap = SLICE_CHUNK_BYTES, MAX_SLICE_CHUNK_BYTES
    assert dynamic_chunk_budget(total_bytes=1, node_ranks=48) == floor
    mid = dynamic_chunk_budget(total_bytes=300 * 2**30, node_ranks=48)
    assert floor < mid < cap
    assert dynamic_chunk_budget(total_bytes=4000 * 2**30, node_ranks=2) == cap
    assert dynamic_chunk_budget(total_bytes=600 * 2**30, node_ranks=48) == 2 * mid

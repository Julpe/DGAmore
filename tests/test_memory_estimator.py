# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
"""Unit tests for the pure peak-memory estimator (no MPI, no psutil)."""

import numpy as np
import pytest

from dgamore.memory_estimator import (
    ARPACK_EXTRA_VECTORS,
    CHI0Q_IFFTN_TRANSIENT_FACTOR,
    CHIQ_AUX_INVERT_FACTOR,
    DTYPE_BYTES,
    FQ_MATMUL_FACTOR,
    LANCZOS_VERTEX_FACTOR,
    OVERHEAD_FACTOR,
    SDE_FFT_KERNEL_FACTOR,
    BranchPeak,
    estimate_peaks,
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
    """Without Eliashberg the estimator reports the chi0q, chiq_aux and sde branches."""
    assert set(_peaks(with_eliashberg=False)) == {"chi0q", "chiq_aux", "sde", "local"}


def test_keys_with_eliashberg():
    """With Eliashberg the estimator adds the fq and lanczos branches."""
    assert set(_peaks(with_eliashberg=True)) == {"chi0q", "chiq_aux", "sde", "fq", "lanczos", "local"}


def test_every_branch_has_positive_baseline_and_off_transient():
    """SDE branches carry a positive per-rank baseline, Eliashberg branches none, and all have a fast-path transient."""
    peaks = _peaks(with_eliashberg=True)
    for key, bp in peaks.items():
        assert isinstance(bp, BranchPeak)
        assert bp.baseline > 0 if key in ("chi0q", "chiq_aux", "sde") else bp.baseline == 0.0
        assert bp.off_distributed + bp.off_single > 0


def test_chi0q_fast_path_is_distributed_on_multi_rank_runs():
    """The chi0q fast path is column-distributed on multi-rank runs and falls back to the rank-0 build on one rank."""
    multi = _peaks()["chi0q"]
    assert multi.off_distributed > 0.0 and multi.off_single == 0.0
    single = _peaks(n_ranks=1)["chi0q"]
    assert single.off_distributed == 0.0 and single.off_single > 0.0
    assert multi.on_distributed > 0.0 and multi.on_single == 0.0


def test_chiq_aux_off_has_distributed_block_and_no_single_rank_gather():
    """The chiq_aux fast path holds a per-rank block; the irr-to-full-BZ map is always the p2p exchange."""
    bp = _peaks()["chiq_aux"]
    assert bp.off_distributed > 0.0  # per-rank two-fermion block
    assert bp.off_single == 0.0


def test_lanczos_fast_path_is_single_rank_only():
    """The lanczos fast path is single-rank-only while its lean path is distributed."""
    bp = _peaks(with_eliashberg=True)["lanczos"]
    assert bp.off_distributed == 0.0 and bp.off_single > 0.0
    assert bp.on_distributed > 0.0 and bp.on_single > 0.0  # lean single: the root rank's full-BZ pp bubble


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


def test_fq_distributed_block_heavier_than_chiq_aux_block():
    """The fq distributed block is heavier than chiq_aux (2 vs 1 two-fermion blocks per q)."""
    assert FQ_MATMUL_FACTOR > CHIQ_AUX_INVERT_FACTOR
    peaks = _peaks(with_eliashberg=True)
    assert peaks["fq"].on_distributed > peaks["chiq_aux"].on_distributed


def test_bubble_baseline_is_giwk_plus_sigma_old_at_niv_cut():
    """The chi0q baseline is giwk_full plus sigma_old at niv_cut, plus shareable R-space copies on multi-rank runs."""
    two_point = SCALE * 2 * (TINY["nk_tot"] * TINY["n_bands"] ** 2 * (2 * TINY["niv_cut"]))
    g_r_windows = SCALE * 2 * TINY["nk_tot"] * TINY["n_bands"] ** 2 * (2 * (TINY["niv_full"] + TINY["niw_core"]))
    assert estimate_peaks(**TINY)["chi0q"].baseline == pytest.approx(two_point + g_r_windows)
    assert estimate_peaks(**{**TINY, "n_ranks": 1})["chi0q"].baseline == pytest.approx(two_point)


def test_sde_section_baseline_uses_post_bubble_windows():
    """The chiq_aux baseline holds post-bubble windows; the sde baseline adds the node-shareable R-space G copy."""
    nk, nb = TINY["nk_tot"], TINY["n_bands"]
    giwk = nk * nb**2 * 2 * (TINY["niv_core"] + TINY["niw_core"])
    sigma_old = nk * nb**2 * 2 * TINY["niv_core"]
    peaks = estimate_peaks(**TINY)
    local_vertex = TINY["n_bands"] ** 4 * (TINY["niw_core"] + 1) * (2 * TINY["niv_full"]) ** 2
    assert peaks["chiq_aux"].baseline == pytest.approx(SCALE * (giwk + sigma_old + local_vertex))
    assert peaks["sde"].baseline == pytest.approx(SCALE * (2 * giwk + sigma_old))
    assert peaks["chiq_aux"].giwk_shareable == pytest.approx(SCALE * (giwk + local_vertex))
    assert peaks["sde"].giwk_shareable == pytest.approx(SCALE * 2 * giwk)


def test_giwk_shareable_is_the_giwk_part_of_each_sde_section_baseline():
    """giwk_shareable covers exactly the Green's-function part of the chi0q, chiq_aux and sde baselines."""
    peaks = _peaks(with_eliashberg=True)
    sigma_old = SCALE * BASE["nk_tot"] * BASE["n_bands"] ** 2 * (2 * BASE["niv_cut"])
    assert peaks["chi0q"].giwk_shareable == pytest.approx(peaks["chi0q"].baseline - sigma_old)
    for key in ("chi0q", "chiq_aux", "sde"):
        assert 0 < peaks[key].giwk_shareable < peaks[key].baseline


def test_eliashberg_branches_are_not_giwk_shareable():
    """The fq and lanczos branches carry no per-rank baseline, so nothing is node-shared there."""
    peaks = _peaks(with_eliashberg=True)
    giwk_dga = SCALE * BASE["nk_tot"] * BASE["n_bands"] ** 2 * 2 * BASE["niv_cut"]
    for key in ("fq", "lanczos"):
        assert peaks[key].giwk_shareable == 0.0
        assert peaks[key].baseline == 0.0
        assert peaks[key].off_single >= giwk_dga
        assert peaks[key].on_single >= giwk_dga


def test_bubble_baseline_depends_on_niv_cut_not_niv_full():
    """The single-rank chi0q baseline tracks niv_cut; only the multi-rank R-space copy tracks niv_full."""
    assert _peaks(niv_full=40, n_ranks=1)["chi0q"].baseline == pytest.approx(
        _peaks(niv_full=400, n_ranks=1)["chi0q"].baseline
    )
    assert _peaks(niv_full=40)["chi0q"].baseline < _peaks(niv_full=400)["chi0q"].baseline
    assert _peaks(niv_cut=80)["chi0q"].baseline != pytest.approx(_peaks(niv_cut=800)["chi0q"].baseline)


def test_chiq_aux_off_block_is_two_rank_local_two_fermion_blocks():
    """The chiq_aux off-distributed block equals two rank-local two-fermion blocks."""
    nb, wp, vc = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_core"]
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    block = qi * nb**4 * wp * vc * vc
    assert estimate_peaks(**TINY)["chiq_aux"].off_distributed == pytest.approx(SCALE * CHIQ_AUX_INVERT_FACTOR * block)


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
    """The multi-rank chi0q fast peak counts the per-rank result slice plus the bounded sub-chunk group."""
    nb, wp, vf = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_full"]
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    expected = SCALE * 1.5 * qi * nb**4 * wp * vf
    assert estimate_peaks(**TINY)["chi0q"].off_distributed == pytest.approx(expected)


def test_sde_holds_two_kernels_plus_irr_kernel():
    """The sde FFT path counts the pass kernels plus the retained irr-BZ kernel; the R copy sits in the baseline."""
    nb, wp, vc = TINY["n_bands"], TINY["niw_core"] + 1, 2 * TINY["niv_core"]
    qt = -(-TINY["nk_tot"] // TINY["n_ranks"])
    qi = -(-TINY["nk_irr"] // TINY["n_ranks"])
    expected = SCALE * (SDE_FFT_KERNEL_FACTOR * qt * nb**4 * wp * vc + qi * nb**4 * wp * vc)
    assert estimate_peaks(**TINY)["sde"].off_distributed == pytest.approx(expected)


def test_sde_off_and_on_slots_are_identical():
    """The sde step has no save_memory switch, so both path slots carry the same two-pass FFT estimate."""
    bp = _peaks()["sde"]
    assert bp.on_distributed == pytest.approx(bp.off_distributed)
    assert bp.on_single == pytest.approx(bp.off_single)


def test_fq_lean_includes_rank_local_accumulator_and_loads():
    """The fq lean transient grows with the per-rank q-count via the accumulator and the three 1-fermion loads."""
    few_ranks = _peaks(with_eliashberg=True, n_ranks=2)["fq"].on_distributed
    many_ranks = _peaks(with_eliashberg=True, n_ranks=8)["fq"].on_distributed
    assert few_ranks > many_ranks


def test_fq_lean_accumulator_larger_when_save_fq():
    """save_fq keeps the full ph box, making the fq lean accumulator larger than the small pp box."""
    small = _peaks(with_eliashberg=True, save_fq=False)["fq"].on_distributed
    big = _peaks(with_eliashberg=True, save_fq=True)["fq"].on_distributed
    assert big > small


def test_fq_save_fq_gathers_whole_irr_vertex_on_one_rank():
    """save_fq gathers the whole irreducible-BZ two-fermion vertex on one rank in both fq paths."""
    p = {**TINY, "with_eliashberg": True}
    nb, wp, vc = p["n_bands"], p["niw_core"] + 1, 2 * p["niv_core"]
    giwk_dga = SCALE * p["nk_tot"] * nb**2 * 2 * p["niv_cut"]
    expected = SCALE * p["nk_irr"] * nb**4 * wp * vc * vc + giwk_dga
    peaks = estimate_peaks(**{**p, "save_fq": True})
    assert peaks["fq"].off_single == pytest.approx(expected)
    assert peaks["fq"].on_single == pytest.approx(expected)
    no_save = estimate_peaks(**{**p, "save_fq": False})
    assert no_save["fq"].off_single == pytest.approx(giwk_dga)
    assert no_save["fq"].on_single == pytest.approx(giwk_dga)


def test_lanczos_fast_counts_layout_build_vertices_bubble_and_arpack_basis():
    """The lanczos fast peak holds the vertex-factor blocks, the pp bubble and the ARPACK workspace."""
    p = {**TINY, "with_eliashberg": True, "n_ranks": 4}
    nb, vpp = p["n_bands"], 2 * p["niv_pp"]
    vertex = p["nk_tot"] * nb**4 * vpp * vpp
    chi0 = p["nk_tot"] * nb**4 * vpp
    arpack = (max(2 * 1 + 1, 20) + ARPACK_EXTRA_VECTORS) * p["nk_tot"] * nb**2 * vpp
    giwk_dga = p["nk_tot"] * nb**2 * 2 * p["niv_cut"]
    expected = SCALE * (LANCZOS_VERTEX_FACTOR * vertex + chi0 + arpack + giwk_dga)
    assert estimate_peaks(**p)["lanczos"].off_single == pytest.approx(expected)


def test_lanczos_lean_scales_with_full_bz():
    """The lanczos lean transient scales with the full BZ size (nk_tot), not the irreducible one."""
    small = _peaks(with_eliashberg=True, nk_tot=256)["lanczos"].on_distributed
    big = _peaks(with_eliashberg=True, nk_tot=512)["lanczos"].on_distributed
    assert big > small


def test_lanczos_lean_independent_of_irreducible_bz_size():
    """The lanczos lean transient is independent of the irreducible-BZ size."""
    a = _peaks(with_eliashberg=True, nk_irr=10)["lanczos"].on_distributed
    b = _peaks(with_eliashberg=True, nk_irr=60)["lanczos"].on_distributed
    assert a == pytest.approx(b)


def test_lanczos_lean_vertex_share_saturates_beyond_v_task_count():
    """The lean vertex share is bounded by the v-axis task count (2*niv_pp tasks), not the rank count."""
    at_tasks = _peaks(with_eliashberg=True, n_ranks=2 * BASE["niv_pp"])["lanczos"].on_distributed
    beyond = _peaks(with_eliashberg=True, n_ranks=8 * BASE["niv_pp"])["lanczos"].on_distributed
    assert beyond == pytest.approx(at_tasks)


def test_lanczos_arpack_workspace_grows_with_n_eig():
    """Requesting more eigenpairs than the default ncv=20 basis grows the per-rank ARPACK workspace."""
    default = _peaks(with_eliashberg=True, n_eig=1)["lanczos"]
    many = _peaks(with_eliashberg=True, n_eig=30)["lanczos"]
    assert many.off_single > default.off_single
    assert many.on_distributed > default.on_distributed


def test_save_pairing_vertex_sets_the_lean_single_rank_gather():
    """save_pairing_vertex gathers both irr-BZ pp vertices on one rank, dominating the lean single-rank peak."""
    p = {**TINY, "with_eliashberg": True}
    nb, vpp = p["n_bands"], 2 * p["niv_pp"]
    gather = SCALE * 2 * p["nk_irr"] * nb**4 * vpp * vpp
    chi0 = SCALE * p["nk_tot"] * nb**4 * vpp
    giwk_dga = SCALE * p["nk_tot"] * p["n_bands"] ** 2 * 2 * p["niv_cut"]
    with_save = estimate_peaks(**{**p, "save_pairing_vertex": True})["lanczos"]
    without = estimate_peaks(**{**p, "save_pairing_vertex": False})["lanczos"]
    assert with_save.on_single == pytest.approx(max(chi0, gather) + giwk_dga)
    assert without.on_single == pytest.approx(chi0 + giwk_dga)


def test_local_step_is_flagless_single_rank_and_band_heavy():
    """The local branch is verify-only, rank-count-independent and scales with nb^4 through the shell transient."""
    from dgamore.memory_estimator import LOCAL_SHELL_INVERT_FACTOR

    bp = _peaks()["local"]
    assert bp.baseline == 0.0 and bp.giwk_shareable == 0.0 and bp.off_distributed == 0.0
    assert bp.off_single == bp.on_single > 0.0
    assert _peaks(n_ranks=16)["local"].off_single == pytest.approx(bp.off_single)
    assert _peaks(n_bands=2)["local"].off_single == pytest.approx(16 * bp.off_single)
    wp, vc, vf = BASE["niw_core"] + 1, 2 * BASE["niv_core"], 2 * BASE["niv_full"]
    l_core, l_full = wp * vc * vc, wp * vf * vf
    expected = SCALE * (2 * (2 * l_core + l_full) + 2 * l_core + LOCAL_SHELL_INVERT_FACTOR * l_full)
    assert bp.off_single == pytest.approx(expected)

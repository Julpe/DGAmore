# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
"""Unit tests for the pure peak-memory estimator (no MPI, no psutil)."""

import numpy as np
import pytest

from dgamore.memory_estimator import (
    DTYPE_BYTES,
    OVERHEAD_FACTOR,
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


def _estimate(**overrides):
    return estimate_peaks(**{**BASE, **overrides})


def _peaks(**overrides):
    return _estimate(**overrides)[1]


def _baseline(**overrides):
    return _estimate(**overrides)[0]


# the node total used by the driver: every rank holds baseline + the distributed transient, plus one single-rank one
def _node_total(baseline, distributed, single, r):
    return r * (baseline + distributed) + single


def _off_node_total(bp: BranchPeak, baseline, r):
    return _node_total(baseline, bp.off_distributed, bp.off_single, r)


def test_constants():
    """DTYPE_BYTES tracks the global storage dtype and OVERHEAD_FACTOR is 1.1."""
    assert DTYPE_BYTES == np.dtype(DTYPE).itemsize  # single source of truth: derived from the global storage dtype
    assert OVERHEAD_FACTOR == pytest.approx(1.1)


def test_keys_without_eliashberg():
    """Without Eliashberg the estimator reports the chi0q, chiq_aux and sde branches."""
    assert set(_peaks(with_eliashberg=False)) == {"chi0q", "chiq_aux", "sde"}


def test_keys_with_eliashberg():
    """With Eliashberg the estimator adds the fq and lanczos branches."""
    assert set(_peaks(with_eliashberg=True)) == {"chi0q", "chiq_aux", "sde", "fq", "lanczos"}


def test_returns_baseline_and_branchpeaks():
    """estimate_peaks returns a positive baseline and a BranchPeak per branch."""
    baseline, peaks = _estimate(with_eliashberg=True)
    assert baseline > 0
    assert all(isinstance(bp, BranchPeak) for bp in peaks.values())


def test_every_branch_has_some_off_transient():
    """Every branch allocates a distributed or single-rank transient beyond the baseline."""
    for bp in _peaks(with_eliashberg=True).values():
        assert bp.off_distributed + bp.off_single > 0


def test_chi0q_fast_path_is_single_rank_only():
    """The chi0q fast path is single-rank-only while its lean path is distributed."""
    bp = _peaks()["chi0q"]
    assert bp.off_distributed == 0.0 and bp.off_single > 0.0
    assert bp.on_distributed > 0.0 and bp.on_single == 0.0


def test_chiq_aux_off_has_distributed_block_and_single_rank_gather():
    """The chiq_aux fast path holds a per-rank two-fermion block plus a single-rank full-BZ gather."""
    bp = _peaks()["chiq_aux"]
    assert bp.off_distributed > 0.0  # per-rank two-fermion block
    assert bp.off_single > 0.0  # full-BZ kernel gather on one rank


def test_lanczos_fast_path_is_single_rank_only():
    """The lanczos fast path is single-rank-only while its lean path is distributed."""
    bp = _peaks(with_eliashberg=True)["lanczos"]
    assert bp.off_distributed == 0.0 and bp.off_single > 0.0
    assert bp.on_distributed > 0.0 and bp.on_single == 0.0


def test_sde_and_fq_are_distributed_only():
    """The sde and fq branches are distributed-only with no single-rank transient."""
    peaks = _peaks(with_eliashberg=True)
    for key in ("sde", "fq"):
        bp = peaks[key]
        assert bp.off_single == 0.0 and bp.on_single == 0.0
        assert bp.off_distributed > 0.0


def test_chi0q_single_rank_peak_independent_of_rank_count():
    """The chi0q single-rank peak (built over the whole irreducible BZ) does not shrink with more ranks."""
    assert _peaks(n_ranks=2)["chi0q"].off_single == pytest.approx(_peaks(n_ranks=16)["chi0q"].off_single)


def test_chiq_aux_distributed_block_shrinks_with_more_ranks():
    """The chiq_aux distributed block shrinks as the rank count grows."""
    assert _peaks(n_ranks=16)["chiq_aux"].off_distributed < _peaks(n_ranks=2)["chiq_aux"].off_distributed


def test_lanczos_single_rank_independent_of_rank_count():
    """The lanczos single-rank peak is independent of the rank count."""
    few = _peaks(n_ranks=2, with_eliashberg=True)["lanczos"].off_single
    many = _peaks(n_ranks=8, with_eliashberg=True)["lanczos"].off_single
    assert few == pytest.approx(many)


def test_two_fermion_branches_dominate_node_total():
    """The two-fermion branches (chiq_aux, fq) dominate the per-node memory total."""
    baseline, peaks = _estimate(with_eliashberg=True)
    r = BASE["n_ranks"]
    totals = {k: _off_node_total(bp, baseline, r) for k, bp in peaks.items()}
    assert totals["chiq_aux"] > totals["chi0q"]
    assert totals["chiq_aux"] > totals["sde"]
    assert totals["fq"] > totals["sde"]


def test_node_total_monotonic_in_n_bands():
    """The node total grows with the number of bands."""
    r = BASE["n_ranks"]
    small = _off_node_total(_peaks(n_bands=1)["chiq_aux"], _baseline(n_bands=1), r)
    big = _off_node_total(_peaks(n_bands=2)["chiq_aux"], _baseline(n_bands=2), r)
    assert big > small


def test_overhead_scales_everything_linearly():
    """The overhead factor scales the baseline and every branch linearly."""
    base1, peaks1 = estimate_peaks(**BASE, overhead=1.0)
    base2, peaks2 = estimate_peaks(**BASE, overhead=2.0)
    assert base2 == pytest.approx(2.0 * base1)
    assert peaks2["chiq_aux"].off_distributed == pytest.approx(2.0 * peaks1["chiq_aux"].off_distributed)


def test_fq_distributed_block_heavier_than_chiq_aux_block():
    """The fq distributed block is heavier than chiq_aux (3 vs 2 two-fermion blocks per q)."""
    from dgamore.memory_estimator import CHIQ_AUX_INVERT_FACTOR, FQ_MATMUL_FACTOR

    assert FQ_MATMUL_FACTOR > CHIQ_AUX_INVERT_FACTOR
    peaks = _peaks(with_eliashberg=True)
    assert peaks["fq"].on_distributed > peaks["chiq_aux"].on_distributed


def test_baseline_is_giwk_plus_sigma_old_at_their_windows():
    """The baseline equals giwk_full plus sigma_old, both kept at the niv_cut window."""
    tiny = dict(BASE, n_bands=2, nk_tot=100, niw_core=5, niv_core=5, niv_full=7, niv_cut=22)
    giwk = tiny["nk_tot"] * tiny["n_bands"] ** 2 * (2 * tiny["niv_cut"])
    sigma_old = tiny["nk_tot"] * tiny["n_bands"] ** 2 * (2 * tiny["niv_cut"])
    expected = DTYPE_BYTES * OVERHEAD_FACTOR * (giwk + sigma_old)
    assert estimate_peaks(**tiny)[0] == pytest.approx(expected)


def test_baseline_depends_on_niv_cut_not_niv_full():
    """The baseline tracks niv_cut and is independent of niv_full when niv_cut is fixed."""
    assert _baseline(niv_full=40) == pytest.approx(_baseline(niv_full=400))
    assert _baseline(niv_cut=80) != pytest.approx(_baseline(niv_cut=800))


def test_chiq_aux_invert_factor_counts_construction_temporary():
    """CHIQ_AUX_INVERT_FACTOR is 2, counting the block-construction temporary kept live."""
    from dgamore.memory_estimator import CHIQ_AUX_INVERT_FACTOR

    assert CHIQ_AUX_INVERT_FACTOR == 2


def test_chiq_aux_off_block_is_two_rank_local_two_fermion_blocks():
    """The chiq_aux off-distributed block equals two rank-local two-fermion blocks."""
    p = dict(
        n_bands=2, nk_tot=80, nk_irr=20, niw_core=4, niv_core=5, niv_full=6, niv_cut=15, niv_pp=2, n_ranks=4,
        with_eliashberg=False,
    )
    _, peaks = estimate_peaks(**p)
    nb, wp, vc = p["n_bands"], p["niw_core"] + 1, 2 * p["niv_core"]
    qi = -(-p["nk_irr"] // p["n_ranks"])
    block = qi * nb**4 * wp * vc * vc
    scale = DTYPE_BYTES * OVERHEAD_FACTOR
    assert peaks["chiq_aux"].off_distributed == pytest.approx(scale * 2 * block)


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


def test_sde_lean_includes_green_function_copy():
    """The sde lean transient includes a full Green's-function copy at the niv_cut window."""
    a = _peaks(niv_cut=80)["sde"].on_distributed
    b = _peaks(niv_cut=160)["sde"].on_distributed
    assert b > a


def test_fq_lean_includes_rank_local_accumulator():
    """The fq lean transient grows with the per-rank q-count via the rank-local accumulator."""
    few_ranks = _peaks(with_eliashberg=True, n_ranks=2)["fq"].on_distributed
    many_ranks = _peaks(with_eliashberg=True, n_ranks=8)["fq"].on_distributed
    assert few_ranks > many_ranks


def test_fq_lean_accumulator_larger_when_save_fq():
    """save_fq keeps the full ph box, making the fq lean accumulator larger than the small pp box."""
    small = _peaks(with_eliashberg=True, save_fq=False)["fq"].on_distributed
    big = _peaks(with_eliashberg=True, save_fq=True)["fq"].on_distributed
    assert big > small


def test_fq_cheap_construction_shrinks_per_q_block():
    """construct_fq_cheap cuts the inputs to niv_pp, shrinking every per-q two-fermion block."""
    normal = _peaks(with_eliashberg=True, construct_fq_cheap=False)["fq"].on_distributed
    cheap = _peaks(with_eliashberg=True, construct_fq_cheap=True)["fq"].on_distributed
    assert cheap < normal


def test_chi0q_fast_single_counts_two_full_grid_buffers():
    """The chi0q fast single-rank peak counts both the multiply buffer and the ifftn output."""
    p = dict(
        n_bands=2, nk_tot=80, nk_irr=20, niw_core=4, niv_core=5, niv_full=6, niv_cut=15, niv_pp=2, n_ranks=4,
        with_eliashberg=False,
    )
    _, peaks = estimate_peaks(**p)
    nb, wp, vf = p["n_bands"], p["niw_core"] + 1, 2 * p["niv_full"]
    bubble_irr = p["nk_irr"] * nb**4 * wp * vf
    chi_r_v_buffer = p["nk_tot"] * nb**4 * vf  # preallocated multiply target
    ifftn_output = p["nk_tot"] * nb**4 * vf  # xp.fft.ifftn returns a second full-grid buffer of the same size
    gf_copies = 2 * p["nk_tot"] * nb**2 * (2 * (p["niv_full"] + p["niw_core"]))
    scale = DTYPE_BYTES * OVERHEAD_FACTOR
    expected = scale * (bubble_irr + chi_r_v_buffer + ifftn_output + gf_copies)
    assert peaks["chi0q"].off_single == pytest.approx(expected)

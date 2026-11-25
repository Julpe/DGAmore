# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Linearized Eliashberg equation solver. Starting from the ladder-DGA full vertex (saved per channel by the non-local SDE
step), this module assembles (through :mod:`~dgamore.pairing_vertex`) the particle-particle pairing vertex in the
singlet/triplet channels at :math:`\omega = 0`, optionally adds the local reducible diagrams, and solves the linearized
gap equation :math:`\lambda \Delta = \pm\frac{1}{2\beta n_{\mathbf{k}}}\, \Gamma^{\mathrm{pp}}\, \chi_0^{\mathrm{pp}}\,
\Delta` with a matrix-free ARPACK/Lanczos eigensolver: in memory when a node holds one channel's pairing vertex (as a
team solve over one node-shared vertex window per channel on a multi-rank job, sector after sector on a single rank) and
with one team of all ranks whose vertex windows, vectors and bubble are split over the nodes otherwise. The leading
eigenvalue :math:`\lambda` signals the pairing instability and the eigenvector is the gap function
:math:`\Delta^{\mathrm{k}}_{12}`. Equation numbers refer to the author's master's thesis (Chapter 4).
"""

import glob
import os
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache, partial

import mpi4py.MPI as MPI
import numpy as np
import scipy as sp
from threadpoolctl import ThreadpoolController, threadpool_limits

import dgamore.config as config
from dgamore import mpi_utils, output_files, pairing_vertex, sde_kernels
from dgamore.brillouin_zone import KGrid, KnownSymmetries
from dgamore.bubble_gen import BubbleGenerator
from dgamore.four_point import FourPoint
from dgamore.gap_function import GapFunction
from dgamore.greens_function import GreensFunction
from dgamore.interaction import LocalInteraction, Interaction
from dgamore import memory_estimator
from dgamore.memory_estimator import (
    lanczos_distributed_bytes,
    lanczos_ncv,
    lanczos_solver_bytes,
    lanczos_team_bytes,
    team_build_columns,
)
from dgamore.mpi_utils import MpiDistributor
from dgamore.n_point_base import SpinChannel, FrequencyNotation, DTYPE
from dgamore.self_energy import SelfEnergy
from dgamore.symmetry_reduction import (
    M_preserves_grid,
    close_group,
    find_coordinate_mirror_orbital_unitaries,
    find_inversion_orbital_unitary,
    forced_multiplicity,
    point_group_orbits,
)

EVEN_SECTOR_DEGENERACY = 1e-2  # largest relative difference of the leading singlet-even and triplet-even eigenvalues
# at which even_sectors_degenerate reports them as coinciding
SECTOR_RESIDUAL_WARNING = 1e-4  # residual (over max(|rho|, 1)) of a sector's leading gap under the unprojected kernel
# above which (or above ten times a larger eigensolver tolerance) the parity sectors are reported as no symmetry of it
LANCZOS_MATVECS = 3000  # matvec cap of one eigensolver attempt, after which _solve_loosening retries the sector
TOLERANCE_LOOSENING = 1e2  # factor of each of the at most two looser retries of an unconverged sector
WATCHDOG_SECOND_PASS_SECONDS = 5  # delay of the rank-file watchdog's second pass, for files of ranks killed mid-save
STAR_BATCH_BYTES = 2**20  # right-hand sides of the stars of the real-space wedge one stacked matrix product takes
# arguments: Eliashberg folder, abort marker, delay of the second pass; "done" on the pipe means a normal end
_WATCHDOG_SCRIPT = (
    'trap "" HUP INT TERM; read -r state; [ "$state" = done ] && exit 0; cd "$1" || exit 0; touch "$2"; '
    "find . -maxdepth 1 -name '*_rank_*.npy' -delete; sleep \"$3\"; find . -maxdepth 1 -name '*_rank_*.npy' -delete"
)
_watchdog: subprocess.Popen | None = None  # the rank-file watchdog of this process, see start_rank_file_watchdog


def delete_rank_files() -> None:
    """
    Deletes the per-rank intermediates of every rank (the ``*_rank_<r>.npy`` files) from the run's Eliashberg folder.
    A failed rank calls it before it aborts the job, while the other ranks may still be writing such files: it first
    places :data:`~dgamore.output_files.ABORT_MARKER` in the folder, after which
    :func:`~dgamore.nonlocal_sde.save_rank_file` adds no rank file, then deletes every rank file present. The folder
    and its other files stay. A missing folder is left as it is. Every OS error is suppressed, so the abort is reached.

    :return: None.
    """
    path = config.output.eliashberg_path
    if not os.path.isdir(path):
        return
    with suppress(OSError):
        open(os.path.join(path, output_files.ABORT_MARKER), "a").close()
    for name in glob.glob(os.path.join(path, "*_rank_*.npy")):
        with suppress(OSError):
            os.remove(name)


def start_rank_file_watchdog() -> subprocess.Popen:
    """
    Starts a shell, in a session of its own, that deletes the rank files of the run's Eliashberg folder as
    :func:`delete_rank_files` does once the calling process ends without :func:`stop_rank_file_watchdog`, so a job
    killed from outside (out of memory, time limit, SIGKILL) leaves none behind. The shell waits on a pipe from the
    calling process, which closes however that process ends, and ignores the hang-up, interrupt and termination signals
    a scheduler sends to the job before it kills it. A second pass after :data:`WATCHDOG_SECOND_PASS_SECONDS` deletes
    the files of ranks killed during a save.

    :return: The watchdog process.
    """
    global _watchdog
    _watchdog = subprocess.Popen(
        [
            "sh",
            "-c",
            _WATCHDOG_SCRIPT,
            "sh",
            config.output.eliashberg_path,
            output_files.ABORT_MARKER,
            str(WATCHDOG_SECOND_PASS_SECONDS),
        ],
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    return _watchdog


def stop_rank_file_watchdog() -> None:
    """
    Tells the watchdog of :func:`start_rank_file_watchdog` that the run ended normally, so it deletes nothing, and waits
    for it to exit. Without a watchdog it does nothing.

    :return: None.
    """
    if _watchdog is not None:
        _watchdog.communicate(b"done\n")


def _compute_once_per_node(node_comm: MPI.Comm | None, compute_fn):
    """
    Evaluates ``compute_fn`` on the node-local root rank only and broadcasts its (small) result to the node's other
    ranks. Used for quantities whose result is cheap but whose construction holds a multi-GB transient, so that
    transient exists once per node instead of once per rank. A ``None`` communicator evaluates locally.

    :param node_comm: The node-local communicator (e.g. from ``comm.Split_type(MPI.COMM_TYPE_SHARED)``), or ``None``.
    :param compute_fn: Zero-argument callable returning the result; invoked only on the node root.
    :return: The node root's result, on every rank of the node.
    """
    if node_comm is None:
        return compute_fn()
    result = compute_fn() if node_comm.Get_rank() == 0 else None
    return node_comm.bcast(result, root=0)


# --- Gap initialization ---
def get_initial_gap_function(shape: tuple, channel: SpinChannel) -> np.ndarray:
    """
    Generates the initial gap-function guess for the power iteration, seeded with the configured momentum symmetry
    (d-wave / p-wave-x / p-wave-y) and the corresponding frequency parity for the singlet/triplet channel; falls back
    to a random guess if no symmetry is configured or recognized.

    The random guess is drawn from a generator with a fixed seed rather than from the global ``np.random``
    state: an unseeded start vector makes a run irreproducible, and inside a degenerate multiplet - where the
    eigensolver is free to return any basis of the eigenspace - it changes which partners come back from one run to
    the next. It also keeps the seed identical on every rank without a broadcast.

    For a configured symmetry the negative-frequency half is set from the positive one reflected under the
    fermionic-frequency flip T (``np.flip`` over the last axis, pairing ``niv + j`` with ``niv - 1 - j``) and signed
    by the channel's required parity, so the seed is a T eigenvector even for a frequency-dependent form factor.

    :param shape: Target array shape ``[kx, ky, kz, o1, o2, v]`` of the gap function.
    :param channel: Pairing channel, either :attr:`SpinChannel.SING` or :attr:`SpinChannel.TRIP`.
    :return: The initial gap-function array (dtype ``DTYPE`` either way).
    :raises ValueError: If ``channel`` is neither SING nor TRIP.
    """
    if channel not in {SpinChannel.SING, SpinChannel.TRIP}:
        raise ValueError("Channel must be either SING or TRIP.")

    symmetry = config.eliashberg.symmetry
    symm = {
        "d-wave": lambda k: -np.cos(k[0])[:, None, None] + np.cos(k[1])[None, :, None],
        "p-wave-x": lambda k: np.sin(k[0])[:, None, None],
        "p-wave-y": lambda k: np.sin(k[1])[None, :, None],
    }
    v_sym = {
        "d-wave": "even" if channel == SpinChannel.SING else "odd",
        "p-wave-x": "odd" if channel == SpinChannel.SING else "even",
        "p-wave-y": "odd" if channel == SpinChannel.SING else "even",
    }

    if symmetry not in symm:
        return np.random.default_rng(42).random(shape).astype(DTYPE)

    niv = shape[-1] // 2
    gap0 = np.zeros(shape, dtype=DTYPE)
    gap0[..., niv:] = np.repeat(symm[symmetry](config.lattice.k_grid.grid)[:, :, :, None, None, None], niv, axis=-1)
    # reflect the positive-frequency half under T (np.flip) and sign it with the channel's parity
    gap0[..., :niv] = (1 if v_sym[symmetry] == "even" else -1) * np.flip(gap0[..., niv:], axis=-1)
    return gap0


# --- Physical-gap symmetry sectors (parity under T O_U + forced inversion parity) ---
def even_sectors_degenerate(results: dict) -> bool:
    r"""
    Returns whether the leading even-frequency singlet and triplet eigenvalues agree to the relative tolerance
    ``EVEN_SECTOR_DEGENERACY``, i.e. to 1 %. A gap counts as even-frequency when most of its weight is even in the
    frequency (see :func:`frequency_parts`): for one band these are the gaps of the even sectors, while with hybridized
    orbitals either parity sector can hold one (an orbital-antisymmetric even-frequency triplet is odd under
    :math:`T O_U`). False without parity resolution or without an even-frequency gap in either channel.

    :param results: The solver output ``{(channel, parity): (lambdas, gaps)}`` with its gaps (rank 0).
    :return: Whether the two leading even-frequency eigenvalues coincide.
    """
    leading = []
    for channel in (SpinChannel.SING, SpinChannel.TRIP):
        even = [
            float(np.real(lam))
            for parity in ("even", "odd")
            for lam, gap in zip(*results.get((channel, parity), ((), ())))
            if _even_frequency_weight(gap.mat) > 0.5
        ]
        if not even:
            return False
        leading.append(max(even))
    return abs(leading[0] - leading[1]) <= EVEN_SECTOR_DEGENERACY * max(abs(leading[0]), abs(leading[1]))


def _parity_sectors(resolve_parity_sectors: bool) -> list[tuple[str, int | None]]:
    r"""
    Returns the list of gap sectors to solve, each a ``(label, eps_T)`` pair where ``eps_T`` is the requested parity
    under the frequency flip times the orbital exchange, :math:`T O_U` (:math:`+1` even, :math:`-1` odd; the frequency
    parity for one band, see :func:`_project_gap_to_sector`), or ``None`` for the unprojected case. When
    ``resolve_parity_sectors`` is set, the even and odd sectors are both returned; otherwise the single unprojected
    sector is returned. The paired inversion parity is fixed by the Pauli constraint ``eps_P = sign * eps_T`` inside
    the solver, so it is not carried here.

    :param resolve_parity_sectors: Whether to split the gap into the even and odd sectors.
    :return: The ``[(label, eps_T), ...]`` sector list.
    """
    return [("even", 1), ("odd", -1)] if resolve_parity_sectors else [("none", None)]


def _sector_log_label(channel: SpinChannel, parities: list[str] | None = None) -> str:
    """
    Returns the name of the sectors a solver call covers, for log messages. Sector-aware naming keeps concurrently
    solving ranks distinguishable in the log, where a channel-only name would repeat verbatim once per sector.

    :param channel: The spin channel being solved.
    :param parities: The subset of parity labels this call handles, or ``None`` for all of them.
    :return: ``"the singlet channel"`` when the parity projection is off, otherwise the covered sectors, e.g.
        ``"the singlet/even & odd sectors"`` or ``"the triplet/odd sector"``.
    """
    labels = [label for label, _ in _parity_sectors(config.eliashberg.resolve_parity_sectors)]
    if parities is not None:
        labels = [label for label in labels if label in parities]
    if labels == ["none"]:
        return f"the {channel.value}let channel"
    return f"the {channel.value}let/{' & '.join(labels)} sector{'' if len(labels) == 1 else 's'}"


def _local_dmft_self_energy(sigma_dmft_per_ineq: list[SelfEnergy]) -> np.ndarray:
    r"""
    Returns the local DMFT self-energy on the correlated orbitals, block diagonal over the atoms of
    ``config.dmft.ineq_ordering``.

    :param sigma_dmft_per_ineq: The DMFT :class:`SelfEnergy` of each inequivalent atom.
    :return: The self-energy, shape ``[o1, o2, v]``.
    """
    blocks = [sigma_dmft_per_ineq[ineq - 1].mat[0, 0, 0] for ineq in config.dmft.ineq_ordering]
    sigma = np.zeros((sum(len(block) for block in blocks),) * 2 + blocks[0].shape[-1:], dtype=complex)
    start = 0
    for block in blocks:
        sigma[start : start + len(block), start : start + len(block)] = block
        start += len(block)
    return sigma


def _inversion_action(sigma_dmft: np.ndarray | None = None) -> tuple[np.ndarray | None, str | None]:
    r"""
    Returns the orbital action of the inversion on the orbitals of the gap (the correlated ones), found on
    :math:`H(\mathbf{k})` by :func:`~dgamore.symmetry_reduction.find_inversion_orbital_unitary`, with the reason the
    parity sectors are not exact, ``None`` when they are. They need inversion to be a symmetry of the whole pairing
    problem: of :math:`H(\mathbf{k})` (of its correlated block), of the local interaction, which has to keep its form
    under the action and may only couple orbitals whose inversion images lie in one unit cell (else the inverted
    interaction is no local one), and of the local DMFT self-energy when it is given. The non-local interaction is
    taken to share the symmetry of :math:`H(\mathbf{k})`, as in the symmetry reduction of the momenta. A multiple of
    the identity comes back as ``None`` (no action).

    :param sigma_dmft: The local DMFT self-energy on the correlated orbitals, shape ``[o1, o2, v]`` (see
        :func:`_local_dmft_self_energy`); ``None`` skips its test.
    :return: ``(u, None)`` with the constant ``[o, o]`` or momentum-dependent ``[kx, ky, kz, o, o]`` action (``None``
        for none), or ``(None, reason)``.
    """
    hamiltonian = config.lattice.hamiltonian
    ek = hamiltonian.get_ek(config.lattice.k_grid)
    # to 1e-4 of the largest element, the tolerance of the self-energy's symmetry tests: a wannier90 H(k) keeps
    # inversion only to the precision of its print
    u = find_inversion_orbital_unitary(ek, 1e-4, orbitals=config.sys.correlated_orbitals)
    if u is None:
        return None, "inversion is no symmetry of H(k) to 1e-4 of its largest element"
    if u.ndim == 2 and np.allclose(u, np.round(u), rtol=0.0, atol=1e-8):
        # a permutation of the orbitals with phases 1, -1, i, -i is taken exact, without the rounding of its discovery
        u = np.round(u)
    u_mat = hamiltonian.get_local_u().mat
    u0 = u if u.ndim == 2 else u[0, 0, 0]
    if u.ndim == 5:
        # U(0)^dagger U(k) = diag(e^{i k.s_a}): the Bloch phases of the cells the orbitals' images lie in
        phases = np.diagonal(u0.conj().T @ u, axis1=-2, axis2=-1).reshape(-1, len(u0))
        same_cell = np.isclose(phases[:, :, None], phases[:, None, :]).all(axis=0)
        a, b, c, d = np.nonzero(np.abs(u_mat) > 1e-12)
        if not (same_cell[a, b] & same_cell[a, c] & same_cell[a, d]).all():
            return None, "the local interaction couples orbitals whose inversion images lie in different unit cells"
    # the action carries the precision of H(k), so the interaction is compared to the same relative tolerance
    rotated = np.einsum("ia,jb,kc,ld,abcd->ijkl", u0, u0.conj(), u0, u0.conj(), u_mat)
    if not np.allclose(rotated, u_mat, rtol=0.0, atol=1e-4 * max(1.0, np.abs(u_mat).max())):
        return None, "the local interaction is not invariant under the orbital action of the inversion"
    if sigma_dmft is not None:
        rotated_sigma = np.einsum("ia,abv,jb->ijv", u0, sigma_dmft, u0.conj())
        if not np.allclose(rotated_sigma, sigma_dmft, rtol=0.0, atol=1e-4 * max(1.0, np.abs(sigma_dmft).max())):
            return None, "the DMFT self-energy is not invariant under the orbital action of the inversion"
    if u.ndim == 2 and np.allclose(u, u[0, 0] * np.eye(len(u))):
        return None, None
    return u, None


def check_parity_sectors(
    comm: MPI.Comm | None = None, sigma_dmft_per_ineq: list[SelfEnergy] | None = None, log: bool = True
) -> np.ndarray | None:
    r"""
    Prepares the parity sectors of the gap (``config.eliashberg.resolve_parity_sectors``): rank 0 looks for the
    orbital action of the inversion (see :func:`_inversion_action`) and broadcasts it, and every rank gets it back for
    the sector projectors (see :func:`_project_gap_to_sector`). Where inversion is no symmetry of the pairing problem
    the sectors would mix, so the flag is switched off with a warning and both channels are solved unprojected. The
    driver calls it before the memory estimate, which counts the sectors, when the run solves the gap, and
    :func:`solve` again for the action, so the momentum-dependent one is held only while the gap is solved. The
    driver's call also tests the DMFT self-energy (outside RPA mode, which reads none) and leaves the log line of the
    action to the second call.

    :param comm: The MPI communicator; ``None`` on a single rank.
    :param sigma_dmft_per_ineq: The DMFT :class:`SelfEnergy` of each inequivalent atom, read on rank 0 (see
        :func:`_local_dmft_self_energy`); ``None`` skips its test.
    :param log: Whether to log the action found.
    :return: The constant ``[o, o]`` or momentum-dependent ``[kx, ky, kz, o, o]`` action, ``None`` for none or without
        the sectors.
    """
    eliashberg = config.eliashberg
    if not eliashberg.resolve_parity_sectors:
        return None
    found = None
    if comm is None or comm.rank == 0:
        found = _inversion_action(None if sigma_dmft_per_ineq is None else _local_dmft_self_energy(sigma_dmft_per_ineq))
    if comm is not None and comm.size > 1:
        found = comm.bcast(found, root=0)
    u, reason = found
    if reason is not None:
        config.logger.warning(
            f"The parity sectors of the gap are switched off: {reason}, so the sectors would mix. Both channels are "
            "solved unprojected."
        )
        eliashberg.resolve_parity_sectors = False
        return None
    if log:
        action = (
            "leaves the orbitals unchanged"
            if u is None
            else (
                "acts on the orbitals by a constant unitary"
                if u.ndim == 2
                else "acts on the orbitals by a momentum-dependent unitary (it maps a site into another unit cell)"
            )
        )
        config.logger.info(f"Parity sectors of the gap: the inversion {action}.")
    return u


def frequency_parts(gap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Splits a gap into its even- and odd-frequency parts, :math:`\Delta^{\pm}(\nu) = \tfrac{1}{2}[\Delta(\nu) \pm
    \Delta(-\nu)]`. Once the orbitals hybridize, the frequency flip is no symmetry of the kernel and the gap of a
    parity sector holds both parts; their weights :math:`\lVert\Delta^{\pm}\rVert^2 / \lVert\Delta\rVert^2` add up to
    one.

    :param gap: The gap, the fermionic frequency (on a box symmetric about zero) as its last axis.
    :return: ``(even, odd)``, each of the shape of ``gap``.
    """
    mirror = np.flip(gap, axis=-1)
    return 0.5 * (gap + mirror), 0.5 * (gap - mirror)


def _even_frequency_weight(gap: np.ndarray) -> float:
    r"""The weight :math:`\lVert\Delta^{+}\rVert^2 / \lVert\Delta\rVert^2` of a gap's even-frequency part (see
    :func:`frequency_parts`), zero for a zero gap."""
    norm2 = np.vdot(gap, gap).real
    even = frequency_parts(gap)[0]
    return float(np.vdot(even, even).real / norm2) if norm2 > 0 else 0.0


def _log_sector_residual(gap: np.ndarray, raw: np.ndarray, label: str, ranks, reduce: Callable | None = None) -> None:
    r"""
    Logs the residual :math:`\lVert K\Delta - \rho\Delta\rVert / (\max(\lvert\rho\rvert, 1) \lVert\Delta\rVert)` of a
    projected sector's leading gap under the unprojected kernel :math:`K`, with the Rayleigh quotient
    :math:`\rho = \langle\Delta, K\Delta\rangle / \langle\Delta, \Delta\rangle` (the eigensolvers report only the real
    part of a complex eigenvalue). It is relative to the eigenvalue, but at most to the instability at 1: the rounding
    of a single-precision kernel is absolute, so a sector whose eigenvalue lies far below 1 cannot reach a small
    relative residual, and it matters no more for the instability. Where the sector is a symmetry of the kernel the
    residual vanishes up to rounding and the eigensolver tolerance ``eliashberg.epsilon``; above the larger of
    ``SECTOR_RESIDUAL_WARNING`` and ten times that tolerance a warning follows.

    :param gap: The leading gap of the sector (this rank's block of it on a team).
    :param raw: :math:`K\Delta` (the same block).
    :param label: The sector label for the log.
    :param ranks: The ranks that log.
    :param reduce: Sums a block quantity over the ranks holding the gap (a team's ``allreduce``); ``None`` for a whole
        gap.
    :return: None.
    """
    reduce = (lambda x: x) if reduce is None else reduce
    norm2 = float(np.real(reduce(np.vdot(gap, gap))))
    rayleigh = complex(reduce(np.vdot(gap, raw))) / norm2 if norm2 > 0 else 0j
    if rayleigh == 0:
        return  # an empty sector (every k equal to -k on a tiny grid) leaves no eigenvalue to check
    leak = raw - rayleigh * gap
    residual = np.sqrt(float(np.real(reduce(np.vdot(leak, leak)))) / norm2) / max(abs(rayleigh), 1.0)
    logger = config.logger
    logger.info(f"Leading gap of {label} under the unprojected kernel: residual {residual:.1e}.", allowed_ranks=ranks)
    # a Ritz vector keeps a residual of the eigensolver tolerance even where the sector is exact
    if residual > max(SECTOR_RESIDUAL_WARNING, 10 * config.eliashberg.epsilon):
        logger.warning(
            f"The leading gap of {label} is no eigenvector of the unprojected kernel (residual "
            f"{residual:.1e}): the parity sectors are no symmetry of this pairing kernel, and the sector eigenvalues "
            "belong to the projected kernel only. 'eliashberg.resolve_parity_sectors: False' solves the channels "
            "unprojected.",
            allowed_ranks=ranks,
        )


def _at_minus_k(a: np.ndarray) -> np.ndarray:
    r"""An array with the momentum grid on its three leading axes, read at :math:`-\mathbf{k}` on the periodic grid."""
    return np.roll(np.flip(a, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))


def _permute_orbital_pair(g: np.ndarray, u: np.ndarray) -> np.ndarray | None:
    r"""
    Returns :math:`U g U^T` on the orbital pair of an array ``[..., o1, o2, v]`` for a constant action whose every
    entry is exactly 0, 1, -1, i or -i with one entry per row, :math:`U_{ab} = s_a \delta_{b\,p(a)}`: the entries
    :math:`s_a s_b\, g_{p(a) p(b)}`, gathered without a sum; ``None`` for any other action.

    :param g: The array, its orbital pair on the two axes before the last.
    :param u: The constant ``[o, o]`` action.
    :return: The rotated array (a new one), or ``None``.
    """
    if not np.isin(u, (0, 1, -1, 1j, -1j)).all() or (np.count_nonzero(u, axis=1) != 1).any():
        return None
    perm = np.argmax(u != 0, axis=1)
    phases = np.outer(u[np.arange(len(u)), perm], u[np.arange(len(u)), perm])
    out = g[..., perm[:, None], perm, :]
    return out if (phases == 1).all() else out * phases.astype(g.dtype)[..., None]


def _rotate_orbital_pair(g: np.ndarray, u: np.ndarray | None) -> np.ndarray:
    r"""
    Applies the orbital action of the inversion to both orbitals of a gap ``[kx, ky, kz, o1, o2, v]``,
    :math:`U(\mathbf{k})\, g(\mathbf{k})\, U(-\mathbf{k})^T`, for a constant ``u`` (an involution, so
    :math:`U(-\mathbf{k}) = U`) or a momentum-dependent one of shape ``[kx, ky, kz, o, o]``; ``None`` is no action.
    A constant orbital permutation with phases is gathered (see :func:`_permute_orbital_pair`).
    """
    if u is None:
        return g
    u = u.astype(g.dtype, copy=False)
    if u.ndim == 2:
        permuted = _permute_orbital_pair(g, u)
        return np.einsum("ac,xyzcdv,bd->xyzabv", u, g, u, optimize=True) if permuted is None else permuted
    return np.einsum("xyzac,xyzcdv,xyzbd->xyzabv", u, g, _at_minus_k(u), optimize=True)


def _exchange_orbitals(g: np.ndarray, u: np.ndarray | None) -> np.ndarray:
    r""":math:`O_U`: the orbital transpose of a gap followed by the orbital action of the inversion,
    :math:`U(\mathbf{k})\, g(\mathbf{k})^T U(-\mathbf{k})^T`."""
    return _rotate_orbital_pair(g.swapaxes(3, 4), u)


def _invert(g: np.ndarray, u: np.ndarray | None) -> np.ndarray:
    r""":math:`P_U`: the inversion of a gap, :math:`\Delta(\mathbf{k}) \to U(\mathbf{k})\, \Delta(-\mathbf{k})\,
    U(-\mathbf{k})^T`."""
    return _rotate_orbital_pair(_at_minus_k(g), u)


def _project_gap_to_sector(
    vec: np.ndarray, gap_shape: tuple, eps_t: int, eps_p: int, u: np.ndarray | None = None
) -> np.ndarray:
    r"""
    Projects a flattened gap onto a physical symmetry sector by applying the two commuting Hermitian projectors
    :math:`\tfrac{1}{2}(1 + \varepsilon_T\, T O_U)` and :math:`\tfrac{1}{2}(1 + \varepsilon_P P_U)` in turn, where
    the involutions act on the orbital gap :math:`\Delta^{\nu}_{12}(\mathbf{k})` as
    :math:`(T\Delta)^{\nu}(\mathbf{k}) = \Delta^{-\nu}(\mathbf{k})` (fermionic-frequency flip),
    :math:`(O_U\Delta)^{\nu}(\mathbf{k}) = U(\mathbf{k}) (\Delta^{\nu}(\mathbf{k}))^T U(-\mathbf{k})^T` (orbital
    exchange) and :math:`(P_U\Delta)^{\nu}(\mathbf{k}) = U(\mathbf{k}) \Delta^{\nu}(-\mathbf{k}) U(-\mathbf{k})^T`
    (inversion), with :math:`U(\mathbf{k})` the orbital action of the inversion,
    :math:`H(-\mathbf{k}) = U(\mathbf{k}) H(\mathbf{k}) U(\mathbf{k})^\dagger` and :math:`U(\mathbf{k}) U(-\mathbf{k})
    = 1`, constant unless inversion maps a site into another unit cell (see :func:`check_parity_sectors`).
    The Pauli antisymmetry :math:`\hat{S}\,P\,O\,T\,\Delta = -\Delta` with the spin exchange :math:`\hat{S}` a scalar
    in the singlet/triplet basis fixes :math:`T O_U \Delta = \mathrm{sign}\, P_U \Delta` (``sign`` the channel sign),
    so once :math:`\varepsilon_T` is chosen the inversion parity is forced to :math:`\varepsilon_P = \mathrm{sign}\cdot
    \varepsilon_T`, and the sectors are the inversion-parity sectors of the channel. The pairing kernel conserves
    :math:`P_U` wherever inversion is a symmetry, so they are invariant subspaces; the frequency parity :math:`T` alone
    is no symmetry once the orbitals hybridize (even and odd frequencies mix), and for one band :math:`O_U` is trivial
    and :math:`\varepsilon_T` is the frequency parity.

    :param vec: The flattened gap vector.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param eps_t: The requested parity under :math:`T O_U`, :math:`+1` (even) or :math:`-1` (odd).
    :param eps_p: The forced inversion parity, :math:`\mathrm{sign}\cdot\varepsilon_T`.
    :param u: The orbital action of the inversion, or ``None`` for none (one band, inversion-even orbitals).
    :return: The projected flattened gap vector (dtype preserved).
    """
    g = vec.reshape(gap_shape)
    g = 0.5 * (g + eps_t * _exchange_orbitals(np.flip(g, axis=-1), u))
    g = 0.5 * (g + eps_p * _invert(g, u))
    return g.reshape(-1).astype(vec.dtype, copy=False)


def _project_frequency_block(
    gap: np.ndarray, v0: int, v1: int, eps_t: int, eps_p: int, u: np.ndarray | None = None
) -> np.ndarray:
    r"""
    Returns the fermionic-frequency block ``[..., v0:v1]`` of the sector projection of a gap ``[kx, ky, kz, o1, o2, v]``
    (see :func:`_project_gap_to_sector`), computed from that block and its frequency mirror alone: the :math:`T O_U`
    projector pairs :math:`\nu` with :math:`-\nu` and the :math:`P_U` projector acts within one frequency, so every
    element is formed by the same operations as in the projection of the whole gap: bit for bit without an orbital
    action and with an orbital permutation (see :func:`_permute_orbital_pair`), and to rounding with any other action,
    whose einsum picks its contraction path by the operand shapes.

    :param gap: The gap in shape ``[kx, ky, kz, o1, o2, v]``.
    :param v0: First frequency index of the block.
    :param v1: One past the last frequency index of the block.
    :param eps_t: The requested parity under :math:`T O_U`, :math:`+1` (even) or :math:`-1` (odd).
    :param eps_p: The forced inversion parity, :math:`\mathrm{sign}\cdot\varepsilon_T`.
    :param u: The orbital action of the inversion, or ``None`` for none.
    :return: The projected block, shape ``[kx, ky, kz, o1, o2, v1 - v0]``.
    """
    nv = gap.shape[-1]
    mirror = gap[..., nv - v1 : nv - v0][..., ::-1]  # np.flip(gap, axis=-1)[..., v0:v1]
    g = 0.5 * (gap[..., v0:v1] + eps_t * _exchange_orbitals(mirror, u))
    return 0.5 * (g + eps_p * _invert(g, u))


def _sector_seed(
    base_seed: np.ndarray, gap_shape: tuple, eps_t: int | None, eps_p: int | None, u: np.ndarray | None = None
) -> np.ndarray:
    r"""
    Returns the eigensolver's starting vector of one sector: the base seed projected onto the sector, with a
    deterministic random fallback when the projection of the seed collapses (a seed whose parity is orthogonal to
    the requested sector) and the nonzero base seed itself when the sector is empty on the grid (e.g. every
    :math:`\mathbf{k}` equals :math:`-\mathbf{k}`); the ``"none"`` sector (``eps_t`` of ``None``) uses the base
    seed unchanged. Identical on every rank for identical inputs.

    :param base_seed: The flattened initial gap seed.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param eps_t: The requested parity under :math:`T O_U`, or ``None`` for the raw kernel.
    :param eps_p: The forced inversion parity.
    :param u: The orbital action of the inversion (see :func:`_project_gap_to_sector`).
    :return: The flattened seed of the sector.
    """
    if eps_t is None:
        return base_seed
    seed = _project_gap_to_sector(base_seed, gap_shape, eps_t, eps_p, u)
    if np.linalg.norm(seed) >= 1e-10 * max(np.linalg.norm(base_seed), 1e-30):
        return seed
    rng = np.random.default_rng(0)
    fallback = (rng.standard_normal(gap_shape) + 1j * rng.standard_normal(gap_shape)).flatten()
    fallback = _project_gap_to_sector(fallback.astype(base_seed.dtype, copy=False), gap_shape, eps_t, eps_p, u)
    return fallback if np.linalg.norm(fallback) > 0 else base_seed


def _sector_seeds(
    base_seed: np.ndarray,
    gap_shape: tuple,
    eps_t: int | None,
    eps_p: int | None,
    n_seeds: int,
    u: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Returns the ``n_seeds`` starting vectors of one sector along a leading axis: the seed of :func:`_sector_seed`
    followed by random vectors of a fixed-seed generator projected onto the sector, identical on every rank.

    :param base_seed: The flattened initial gap seed.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param eps_t: The requested parity under :math:`T O_U`, or ``None`` for the raw kernel.
    :param eps_p: The forced inversion parity.
    :param n_seeds: Number of starting vectors.
    :param u: The orbital action of the inversion (see :func:`_project_gap_to_sector`).
    :return: The seeds, shape ``[n_seeds, n]``.
    """
    rng = np.random.default_rng(1)
    seeds = [_sector_seed(base_seed, gap_shape, eps_t, eps_p, u)]
    for _ in range(n_seeds - 1):
        extra = (rng.standard_normal(base_seed.size) + 1j * rng.standard_normal(base_seed.size)).astype(base_seed.dtype)
        seeds.append(extra if eps_t is None else _project_gap_to_sector(extra, gap_shape, eps_t, eps_p, u))
    return np.stack(seeds)


def _finish_sector(
    lambdas: np.ndarray,
    gaps: np.ndarray,
    gap_shape: tuple,
    channel: SpinChannel,
    nq: tuple,
    label: str,
    orbital_mirrors: dict,
    ranks: tuple,
) -> tuple[np.ndarray, list[GapFunction]]:
    r"""
    Orders one solved sector's eigenpairs by descending eigenvalue, symmetrizes degenerate gaps when configured,
    logs the eigenvalues and wraps the eigenvectors as :class:`GapFunction` objects (as many as the eigensolver
    returned, each a contiguous copy so the eigenvector matrix is released).

    :param lambdas: The eigenvalues.
    :param gaps: The eigenvectors as columns, ``[n, len(lambdas)]``.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param channel: The pairing channel (used to label outputs).
    :param nq: The momentum-grid shape carried onto each :class:`GapFunction`.
    :param label: The sector label for the log.
    :param orbital_mirrors: The orbital mirror unitaries for the degenerate-gap symmetrization.
    :param ranks: The ranks tuple used for logging.
    :return: ``(lambdas, [GapFunction, ...])``.
    """
    order = lambdas.argsort()[::-1]
    if not np.array_equal(order, np.arange(len(order))):
        lambdas, gaps = lambdas[order], gaps[:, order]
    if config.eliashberg.symmetrize_degenerate_gaps:
        gaps = symmetrize_degenerate_gaps(lambdas, gaps, gap_shape, orbital_mirrors=orbital_mirrors)
    plural = "" if config.eliashberg.n_eig == 1 else "s"
    config.logger.info(
        f"Largest eigenvalue{plural} for {label}: " + ", ".join(f"{lam:.6f}" for lam in lambdas), allowed_ranks=ranks
    )
    gaps = [np.ascontiguousarray(gaps[:, i]).reshape(gap_shape) for i in range(gaps.shape[1])]
    if gap_shape[3] > 1:
        weights = [_even_frequency_weight(gap) for gap in gaps]
        config.logger.info(
            f"Even-frequency weight of the gap{plural} for {label}: " + ", ".join(f"{w:.4f}" for w in weights),
            allowed_ranks=ranks,
        )
    return lambdas, [GapFunction(gap, channel, nq) for gap in gaps]


def _solve_loosening(solve: Callable[[float], tuple], tol: float, n_eig: int, label: str, ranks) -> tuple:
    r"""
    Returns the eigensolver's result at ``tol`` once it delivers ``n_eig`` eigenpairs within ``LANCZOS_MATVECS``
    operator applications, else retries at a tolerance ``TOLERANCE_LOOSENING`` times looser, twice at most, with a
    warning each time, and returns the last attempt whatever it delivered. In single precision a sector whose
    eigenvalues are small against the rest of the kernel cannot reach a relative tolerance below the kernel's absolute
    rounding, and the eigensolver would run to its matvec cap for nothing.

    :param solve: Runs the eigensolver at a tolerance; the first entry of its result holds the converged eigenvalues.
    :param tol: The tolerance of the first attempt.
    :param n_eig: Number of eigenpairs asked for.
    :param label: The sector label for the log.
    :param ranks: The ranks that log.
    :return: The result of the last attempt.
    """
    for attempt in range(3):
        result = solve(tol)
        if len(result[0]) >= n_eig or attempt == 2:
            return result
        config.logger.warning(
            f"Lanczos for {label} did not converge to {tol:.0e} within {LANCZOS_MATVECS} matvecs; retrying at "
            f"{tol * TOLERANCE_LOOSENING:.0e}.",
            allowed_ranks=ranks,
        )
        tol *= TOLERANCE_LOOSENING


def _team_arnoldi(
    apply_block: Callable,
    seed_block: np.ndarray,
    n_eig: int,
    ncv: int,
    tol: float,
    max_matvec: int,
    team_comm: MPI.Comm | None,
    n_block: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    r"""
    Krylov-Schur (restarted Arnoldi) iteration for the eigenpairs of largest real part of an operator whose vectors
    are split over the ranks of a team: every rank holds its own block of each basis vector, ``apply_block`` maps a
    rank's block of a vector to its block of the operator's image (a collective call over the team), and every
    inner product is a team allreduce, so all ranks walk the same iteration with the same scalars; the eigensolves of
    the small Rayleigh quotient (Ritz pairs, Schur reordering) run on the team's first rank and reach the others by
    broadcast, so a rank whose LAPACK rounds differently cannot leave the iteration. The basis is
    orthogonalized by classical Gram-Schmidt, repeated once when the first pass removed more than a fixed share of
    the new vector (:data:`REORTHOGONALIZE_BELOW`, the criterion ARPACK uses), so a well-conditioned step streams
    the basis twice instead of four times. A vector the second pass also cuts below that share, or an image that
    vanishes exactly, lies in the span of the basis (the Krylov space is invariant, e.g. a sector smaller than the
    basis or one the operator annihilates): the iteration then continues from a random direction orthogonalized
    against the basis, as ARPACK does, with the vanishing norm kept in the
    Rayleigh quotient so the Arnoldi relation stays exact. When the ``ncv``-vector basis is full, the Ritz pairs of the
    Rayleigh quotient are formed; a pair counts as converged when its residual estimate :math:`|\beta_m|\,|y_m|` is
    at most ``tol`` times the modulus of its Ritz value. Otherwise the Schur form of the Rayleigh quotient is
    reordered so that the ``n_eig`` wanted Ritz values and half of the rest lead, and the basis is compacted to
    those Schur vectors plus the residual direction before the iteration continues. Stops when the wanted pairs have
    converged or at the first full basis after ``max_matvec`` operator applications, returning the converged pairs in
    either case; the eigenvalues are returned as real parts, the way scipy's ``eigsh`` reports a complex operator's
    spectrum.

    With ``n_block > 1`` it is the block (band) variant: the basis starts from ``n_block`` orthonormalized seeds, the
    image of basis vector :math:`j` is orthogonalized against the first :math:`j + n_\mathrm{block}` vectors and
    becomes vector :math:`j + n_\mathrm{block}`, and the last ``n_block`` vectors form the residual block whose
    coupling to the Ritz vectors gives the residual estimates and carries over the restart. Every eigenspace of
    dimension up to ``n_block`` is then reached in exact arithmetic, where a single start vector reaches one direction
    of each; ``n_block = 1`` is the plain iteration.

    :param apply_block: The operator, ``block -> block`` on this rank's slice.
    :param seed_block: This rank's block of the starting vector, or of the ``n_block`` starting vectors along a
        leading axis.
    :param n_eig: Number of wanted eigenpairs.
    :param ncv: Size of the Krylov basis (``n_eig < ncv``).
    :param tol: Relative residual tolerance of a Ritz pair.
    :param max_matvec: Number of operator applications after which the next full basis ends the iteration.
    :param team_comm: The team communicator, or ``None`` for a single process holding the whole vectors.
    :param n_block: Number of starting vectors, the largest eigenvalue multiplicity resolved in exact arithmetic.
    :return: ``(lambdas, ritz, basis, n_matvec)``: the converged eigenvalues in descending order, their Ritz
        coefficients as columns (this rank's block of eigenvector ``i`` is ``ritz[:, i] @ basis``, of unit norm up to
        rounding), this rank's blocks of the basis vectors as rows and the number of operator applications.
    """
    dtype = seed_block.dtype
    seeds = seed_block.reshape(n_block, -1)
    basis = np.empty((ncv + n_block - 1, seeds.shape[1]), dtype=dtype)
    rayleigh = np.zeros((ncv + n_block, ncv), dtype=np.complex128)
    n_matvec = 0

    reduce = (lambda x: x) if team_comm is None else team_comm.allreduce

    def on_lead(fn: Callable):
        # the small eigensolves of the Rayleigh quotient run on the lead alone, so LAPACK rounding never splits the team
        return fn() if team_comm is None else team_comm.bcast(fn() if team_comm.Get_rank() == 0 else None, root=0)

    def dots(vectors: np.ndarray, w: np.ndarray) -> np.ndarray:
        # conjugating the one vector instead of the basis slice spares a basis-sized copy per call (bit-identical)
        return reduce((vectors @ w.conj()).conj())

    def norm(w: np.ndarray) -> float:
        return float(np.sqrt(reduce(float(np.vdot(w, w).real))))

    rng = np.random.default_rng(0 if team_comm is None else team_comm.Get_rank())

    def fresh_direction(n_basis: int) -> np.ndarray:
        # a random unit vector orthogonalized twice against the basis (all ranks draw their block together)
        w = (rng.standard_normal(basis.shape[1]) + 1j * rng.standard_normal(basis.shape[1])).astype(dtype)
        for _ in range(2):
            w = w - dots(basis[:n_basis], w) @ basis[:n_basis]
        return w / norm(w)

    basis[0] = seeds[0] / norm(seeds[0])
    for i in range(1, n_block):
        w = seeds[i] - dots(basis[:i], seeds[i]) @ basis[:i]
        w = w - dots(basis[:i], w) @ basis[:i]
        basis[i] = w / norm(w)
    del seed_block, seeds
    j = 0
    while True:
        w = apply_block(basis[j].copy()).reshape(-1).astype(dtype, copy=False)
        n_matvec += 1
        w_norm = norm(w)
        top = j + n_block
        h = dots(basis[:top], w)
        w = w - h @ basis[:top]
        beta, spent = norm(w), False
        if beta <= REORTHOGONALIZE_BELOW * w_norm:
            correction = dots(basis[:top], w)
            w = w - correction @ basis[:top]
            h += correction
            beta, previous = norm(w), beta
            spent = beta <= REORTHOGONALIZE_BELOW * previous
        rayleigh[:top, j] = h
        rayleigh[top, j] = beta
        if j + 1 < ncv:
            basis[top] = fresh_direction(top) if spent else w / beta
            j += 1
            continue
        # the basis is full: Ritz pairs of the Rayleigh quotient, largest real part first
        theta, ritz = on_lead(lambda: np.linalg.eig(rayleigh[:ncv]))
        order = np.argsort(-theta.real)
        theta, ritz = theta[order], ritz[:, order]
        residual = beta * np.abs(ritz[-1]) if n_block == 1 else np.linalg.norm(rayleigh[ncv:] @ ritz, axis=0)
        converged = residual <= tol * np.abs(theta)
        if converged[:n_eig].all() or n_matvec >= max_matvec:
            chosen = [i for i in range(n_eig) if converged[i]]
            return theta[chosen].real, ritz[:, chosen], basis[:ncv], n_matvec
        # Krylov-Schur restart: the wanted Ritz values and half of the rest lead the reordered Schur form
        threshold = theta.real[n_eig + (ncv - n_eig) // 2 - 1]
        schur, vectors, n_leading = on_lead(
            lambda: sp.linalg.schur(rayleigh[:ncv], output="complex", sort=lambda z: z.real >= threshold)
        )
        kept = min(n_leading, ncv - n_block)
        # compacted in column chunks, so the transient stays a fraction of the basis
        chunk = max(1, basis.shape[1] // 8)
        for c0 in range(0, basis.shape[1], chunk):
            basis[:kept, c0 : c0 + chunk] = vectors[:, :kept].T @ basis[:ncv, c0 : c0 + chunk]
        basis[kept : kept + n_block - 1] = basis[ncv:]
        basis[kept + n_block - 1] = fresh_direction(kept + n_block - 1) if spent else w / beta
        coupling = beta * vectors[-1, :kept] if n_block == 1 else rayleigh[ncv:] @ vectors[:, :kept]
        rayleigh[...] = 0.0
        rayleigh[:kept, :kept] = schur[:kept, :kept]
        rayleigh[kept : kept + n_block, :kept] = coupling
        j = kept


def _crossed_term(direct: np.ndarray, sign: int, v0: int = 0, v1: int | None = None) -> np.ndarray:
    r"""
    Forms the crossed term of the pairing kernel from the direct term :math:`D`, as the exchange of its outgoing
    pair: the gap's first orbital sits at :math:`-\mathrm{k}` (see :func:`_apply_gchi0_pp`), so the exchange is
    :math:`(E X)_{ab}(\mathrm{k}) = \mathrm{sign}\,X_{ba}(-\mathrm{k})` with :math:`-\mathrm{k} = (-\mathbf{k}, -\nu)`
    and the kernel is :math:`(1 + E) D`. It maps every gap onto the Pauli-allowed form :math:`E\Delta = \Delta` and
    agrees with the exchange-symmetrized gap equation on it, for any vertex. The momentum flip is the same index map
    in real space and in momentum space, so either form of the direct term may be passed.

    :param direct: The direct term ``[x, y, z, o1, o2, v]`` on the full fermionic box.
    :param sign: The channel sign (:math:`+1` singlet, :math:`-1` triplet).
    :param v0: First fermionic index of the returned block.
    :param v1: One past the last fermionic index of the returned block (``None``: the whole axis).
    :return: The crossed term on the fermionic block ``[v0, v1)``, a new array.
    """
    nv = direct.shape[-1]
    v1 = nv if v1 is None else v1
    crossed = _at_minus_k(np.flip(direct.swapaxes(3, 4)[..., nv - v1 : nv - v0], axis=-1))
    if sign != 1:
        crossed *= sign
    return crossed


def gap_parity_diagnostics(gap: np.ndarray, gap_shape: tuple, u: np.ndarray | None = None) -> dict[str, complex]:
    r"""
    Reports the parity Rayleigh quotients :math:`\langle \Delta, X \Delta \rangle / \langle \Delta, \Delta \rangle`
    of a flattened gap for the involutions ``T`` (frequency flip), ``P`` (momentum flip), ``O`` (orbital transpose),
    their product ``PO``, and the two the parity sectors fix (see :func:`_project_gap_to_sector`): ``TOU``, the
    frequency flip times the orbital exchange :math:`O_U`, and ``PU``, the inversion :math:`P_U`, with ``u`` the
    orbital action of the inversion (``None``: none, so they are ``T.O`` and ``P``). A pure-parity gap returns
    :math:`\pm 1` for the involutions it is an eigenvector of; the values certify the parity of a returned gap and
    expose any leakage.

    :param gap: The flattened gap vector.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param u: The orbital action of the inversion (see :func:`check_parity_sectors`).
    :return: A dict mapping ``"T"``, ``"P"``, ``"O"``, ``"PO"``, ``"TOU"``, ``"PU"`` to the Rayleigh quotient.
    """
    g = gap.reshape(gap_shape)
    ops = {
        "T": np.flip(g, axis=-1),
        "P": _at_minus_k(g),
        "O": g.swapaxes(3, 4),
        "PO": _at_minus_k(g.swapaxes(3, 4)),
        "TOU": _exchange_orbitals(np.flip(g, axis=-1), u),
        "PU": _invert(g, u),
    }
    denom = np.vdot(g, g)
    if denom == 0:
        return {name: 0j for name in ops}
    return {name: complex(np.vdot(g, op) / denom) for name, op in ops.items()}


# Testing/benchmark override: forces the node-distributed team solve even where a node holds the vertex windows.
FORCE_DISTRIBUTED_SOLVER: bool = False


# --- Eliashberg eigensolver (Lanczos / ARPACK) ---
@lru_cache(maxsize=1)
def _openblas_thread_slot_cap() -> int | None:
    r"""
    Returns the build-time thread capacity (``NUM_THREADS``) of the loaded OpenBLAS libraries, or ``None`` when no
    OpenBLAS is loaded. OpenBLAS reserves working-buffer slots for at most that many calling threads at build time;
    a process calling into it from more threads than that overflows into an auxiliary bookkeeping path
    ("precompiled NUM_THREADS exceeded" warning) that is unreliable under concurrency and crashes ("Bad memory
    unallocation!", segmentation faults), so every solver thread budget must stay at or below this capacity.
    ``openblas_set_num_threads`` clamps its argument to the build maximum, so probing with an oversized limit and
    reading the value back yields that maximum; the previous thread settings are restored afterwards. The result is
    cached - the capacity is a fixed property of the loaded libraries. Other BLAS implementations (MKL, BLIS) size
    their buffers per calling thread and need no cap.

    :return: The smallest build-time thread capacity among the loaded OpenBLAS libraries, ``None`` without OpenBLAS.
    """
    openblas_libs = ThreadpoolController().select(internal_api="openblas")
    if not openblas_libs.lib_controllers:
        return None
    with openblas_libs.limit(limits=1 << 15):
        return min(lib.num_threads for lib in openblas_libs.lib_controllers)


def _clamp_to_openblas_slot_cap(budget: int) -> int:
    r"""
    Clamps a thread budget to the loaded OpenBLAS build's thread capacity (see :func:`_openblas_thread_slot_cap`);
    a larger budget would call OpenBLAS from more threads than it has buffer slots for and crash. The budget passes
    through unchanged when no OpenBLAS is loaded.

    :param budget: The thread budget to clamp.
    :return: The budget, clamped to the OpenBLAS thread capacity if OpenBLAS is loaded.
    """
    cap = _openblas_thread_slot_cap()
    return budget if cap is None else min(budget, cap)


def _solver_thread_budget() -> int:
    r"""
    Returns the BLAS/FFT thread budget for the in-memory Lanczos solve: the size of this process's CPU affinity
    mask (at least 1), clamped to the OpenBLAS thread capacity (see :func:`_clamp_to_openblas_slot_cap`). During
    that solve only the one or two solver ranks work while the other ranks of the node wait at the post-solve
    broadcast, so the solver may use every core its affinity mask allows - the launcher's binding stays the single
    source of truth (under a strict one-core-per-rank binding this is 1 and the threading is a no-op). Falls back
    to 1 where the affinity API does not exist (non-Linux platforms).

    :return: The thread budget as an int.
    """
    try:
        budget = max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return 1
    return _clamp_to_openblas_slot_cap(budget)


def _chi0_to_matmul_layout(chi0_mat: np.ndarray) -> np.ndarray:
    r"""
    Returns the bare pp bubble in batched-matmul layout ``[x, y, z, v, o2, o2]`` (a view, no copy) for use with
    :func:`_apply_gchi0_pp`. ``chi0_mat`` is in the einsum layout ``[x, y, z, a, b, c, d, v]``.

    :param chi0_mat: The bubble array :math:`\chi_0^{\mathrm{pp}}`, shape ``[x, y, z, o, o, o, o, v]``.
    :return: A view reshaped/transposed to ``[x, y, z, v, o2, o2]`` (rows ``(a, b)``, columns ``(c, d)``).
    """
    nqx, nqy, nqz, nb = chi0_mat.shape[:4]
    v = chi0_mat.shape[-1]
    return np.moveaxis(chi0_mat.reshape(nqx, nqy, nqz, nb * nb, nb * nb, v), -1, 3)


_MIRROR_TOL = 1e-6  # tolerance on the deviation of a mirror eigenvalue from +/-1


def _gap_orbital_mirrors(n_bands: int) -> dict:
    r"""
    Collects the orbital part :math:`U_i` of the single-axis coordinate mirrors :math:`k_i \to -k_i` for the gap
    symmetrization (see :func:`_mirror_operator`) by solving for them from :math:`H(\mathbf{k})` with the same U-solver
    the automatic symmetry discovery uses (:func:`~dgamore.symmetry_reduction.find_coordinate_mirror_orbital_unitaries`,
    a handful of small eigen-solves on the cached :math:`H(\mathbf{k})`). Nothing is hard-coded per orbital set - the
    mirrors are read off the Hamiltonian, so :math:`t_{2g}`, :math:`e_g` and any other Wannier basis are covered alike,
    and no symmetry mode is assumed (the solve does not need ``symmetries: auto``).

    A single-orbital gap needs no orbital factor (a :math:`1 \times 1` unitary is a phase, and it cancels in
    :math:`U \Delta U^\dagger`), and a failure to determine the mirrors is not fatal: the symmetrization then
    reduces to momentum-only mirrors and falls back to the Loewdin basis wherever those do not resolve a multiplet.

    :param n_bands: Number of orbitals of the gap.
    :return: A dict ``{axis: U}`` (empty when there is nothing to apply or the mirrors cannot be determined; with
        uncorrelated orbitals an axis whose mirror is no symmetry of the correlated block maps to ``None``, with a
        warning).
    """
    if n_bands < 2:
        return {}

    orbitals = config.sys.correlated_orbitals
    try:
        ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
        mirrors = find_coordinate_mirror_orbital_unitaries(np.asarray(ek, dtype=np.complex128), orbitals=orbitals)
    except Exception as exc:  # never let a symmetry probe abort a solve; momentum-only mirrors still work
        log = config.logger.debug if orbitals is None else config.logger.warning
        log(f"Could not determine the orbital mirror matrices for the gap symmetrization: {exc}")
        return {}
    dropped = sorted(axis for axis, u in mirrors.items() if u is None)
    if dropped:
        config.logger.warning(
            f"The orbital mirrors of axes {dropped} are no symmetry of the correlated block; the gap symmetrization "
            f"reflects only the momenta there."
        )
    return mirrors


def _mirror_acts_on_orbitals(u: np.ndarray | None) -> bool:
    r"""
    Whether an orbital mirror matrix acts non-trivially on the gap. The gap transforms by conjugation,
    :math:`\Delta \to U \Delta U^\dagger`, which is the identity map exactly when :math:`U` is a multiple of the
    identity - a global phase cancels between the two factors.

    :param u: The orbital mirror matrix, or ``None``.
    :return: True if conjugation by ``u`` is not the identity map.
    """
    if u is None or u.shape[0] < 2:
        return False
    return not np.allclose(u, (np.trace(u) / u.shape[0]) * np.eye(u.shape[0]), atol=1e-8)


def _validated_orbital_mirrors(gap_shape: tuple, orbital_mirrors: dict | None) -> dict:
    r"""
    Normalizes the caller-supplied orbital mirror matrices, dropping entries that do not match the gap's orbital
    dimension (a mirror discovered on a different orbital set must never be applied silently).

    :param gap_shape: Full gap shape ``[kx, ky, kz, o1, o2, 2*niv_pp]``.
    :param orbital_mirrors: A dict ``{axis: U}`` of orbital mirror matrices, or ``None``.
    :return: The validated dict ``{axis: U}`` as complex arrays (empty when nothing usable is supplied).
    """
    n_orb = gap_shape[3]
    return {
        int(axis): np.asarray(u, dtype=np.complex128)
        for axis, u in (orbital_mirrors or {}).items()
        if u is not None and np.shape(u) == (n_orb, n_orb)
    }


def _mirror_operator(gap_shape: tuple, axis: int, u: np.ndarray | None):
    r"""
    Builds the single-axis mirror operator acting on a flattened gap column. A point-group mirror :math:`k_i \to -k_i`
    acts on the orbital indices as well as on the momenta, so with the orbital matrix :math:`U` of that mirror
    (:math:`H(M_i \mathbf{k}) = U H(\mathbf{k}) U^\dagger`, see
    :func:`~dgamore.symmetry_reduction.find_coordinate_mirror_orbital_unitaries`) the gap transforms as

    .. math:: \Delta_{o_1 o_2}(\mathbf{k}) \to \left[U\, \Delta(M_i \mathbf{k})\, U^\dagger\right]_{o_1 o_2} .

    For :math:`t_{2g}` orbitals :math:`U` is the diagonal sign matrix of the mirror and this reduces to the familiar
    :math:`s_{o_1} s_{o_2} \Delta_{o_1 o_2}(M_i \mathbf{k})`. Dropping the orbital factor (as a momentum-only reflection
    does) leaves an operator that is not a symmetry of the multi-orbital pairing kernel: a partner carrying weight in
    both orbital sectors then measures the orbital-diagonal minus the orbital-off-diagonal weight instead of :math:`\pm
    1`, and the cleanliness check in :func:`_orient_cluster_by_mirrors` rejects it.

    :param gap_shape: Full gap shape ``[kx, ky, kz, o1, o2, 2*niv_pp]``.
    :param axis: The reflected momentum axis (0, 1 or 2).
    :param u: The orbital matrix of that mirror, or ``None``/a multiple of the identity for a momentum-only mirror.
    :return: A callable mapping a flattened gap column to its mirrored image.
    """
    idx = (gap_shape[axis] - np.arange(gap_shape[axis])) % gap_shape[axis]
    take = [slice(None)] * len(gap_shape)
    take[axis] = idx
    take = tuple(take)

    if not _mirror_acts_on_orbitals(u):
        return lambda column: column.reshape(gap_shape)[take].ravel()

    return lambda column: np.einsum(
        "ap,...pqv,bq->...abv", u, column.reshape(gap_shape)[take], u.conj(), optimize=True
    ).ravel()


def _orient_cluster_by_mirrors(
    block: np.ndarray, gap_shape: tuple, tol: float = _MIRROR_TOL, orbital_mirrors: dict | None = None
) -> np.ndarray | None:
    r"""
    Rotates an orthonormal degenerate cluster onto the common eigenbasis of the single-axis coordinate mirrors
    :math:`k_i \to -k_i` and orders the partners lexicographically by the tuple of axes each one is odd under. The
    mirrors mutually commute, so one generic real combination of their projections into the cluster (weights
    :math:`1, \sqrt{2}, \sqrt{3}`, giving a distinct eigenvalue per sign pattern) shares their eigenvectors; each
    partner's mirror eigenvalues are read off as Rayleigh quotients. Single-axis :math:`p`-like partners sort
    ``x, y, z`` and two-axis :math:`d`-like partners (:math:`d_{xy}`, :math:`d_{xz}`, :math:`d_{yz}`) sort
    ``xy, xz, yz``.

    Each mirror acts on the orbital indices as well as on the momenta (see :func:`_mirror_operator`); an axis whose
    orbital matrix is non-trivial therefore resolves partners even when its momentum action does not, which is what
    makes purely local (momentum-independent) multiplets - a local :math:`T_{1g}` triplet, say - resolvable at all.

    Two partners sharing a sign pattern span a subspace the mirrors do not resolve: the combined matrix is
    degenerate there, so the eigenvectors within it are fixed by floating-point noise alone and the sort cannot
    separate them either. Coordinate mirrors alone cannot split an :math:`E_g` doublet, for instance, because an
    orbital-diagonal state picks up :math:`s_o s_o = +1` on every axis; separating those partners would take a
    three-fold rotation about :math:`[111]`. Rotating by noise-fixed eigenvectors would scramble the cluster, so
    such a cluster is rejected and the caller keeps its input basis.

    :param block: Orthonormal cluster, one flattened gap function per column.
    :param gap_shape: Full gap shape ``[kx, ky, kz, o1, o2, 2*niv_pp]``.
    :param tol: Tolerance on the deviation of a mirror Rayleigh quotient from :math:`\pm 1`.
    :param orbital_mirrors: A dict ``{axis: U}`` with the orbital part of each coordinate mirror; entries whose shape
        does not match the gap's orbital dimension are ignored, and a missing axis falls back to a momentum-only
        reflection (exact for a single-orbital gap, but not a kernel symmetry for a multi-orbital one).
    :return: The reordered cluster, or ``None`` when no axis is resolved, a partner is not a clean :math:`\pm 1`
        mirror eigenstate, or two partners share the same mirror sign pattern.
    """
    mirrors = _validated_orbital_mirrors(gap_shape, orbital_mirrors)
    # an axis resolves partners through its momentum reflection, its orbital matrix, or both
    axes = [axis for axis in (0, 1, 2) if gap_shape[axis] > 1 or _mirror_acts_on_orbitals(mirrors.get(axis))]
    if not axes:
        return None

    projected = []
    for axis in axes:
        reflect = _mirror_operator(gap_shape, axis, mirrors.get(axis))
        mirrored = np.stack([reflect(block[:, i]) for i in range(block.shape[1])], axis=1)
        m = block.conj().T @ mirrored
        projected.append(0.5 * (m + m.conj().T))

    weights = np.sqrt(np.arange(1, len(projected) + 1, dtype=float))  # 1, sqrt(2), sqrt(3): generic, deterministic
    _, vecs = np.linalg.eigh(sum(w * m for w, m in zip(weights, projected)))

    keys = []
    for col in range(vecs.shape[1]):
        c = vecs[:, col]
        signature = np.array([(c.conj() @ m @ c).real for m in projected])
        if np.any(np.abs(np.abs(signature) - 1.0) > tol):  # not a clean +/-1 mirror eigenstate
            return None
        keys.append(tuple(int(a) for a in np.flatnonzero(signature < 0.0)))

    if len(set(keys)) < len(keys):  # a shared sign pattern spans a subspace the mirrors leave unresolved
        return None

    block = block @ vecs
    return block[:, sorted(range(len(keys)), key=lambda col: keys[col])]


def symmetrize_degenerate_gaps(
    lambdas: np.ndarray, gaps: np.ndarray, gap_shape: tuple, tol: float = 1e-4, orbital_mirrors: dict | None = None
) -> np.ndarray:
    r"""
    Orthonormalizes the eigenvectors returned by the Lanczos solver within clusters of (near-)degenerate
    eigenvalues and rotates every cluster to a mirror-adapted basis. The pairing kernel is not Hermitian in the
    plain inner product, so ARPACK may return oblique (mutually non-orthogonal) combinations
    inside a degenerate cluster: the cluster subspace is symmetry-covariant, but the returned vectors then do not
    form the symmetry-adapted partners.

    Per cluster the following steps are applied: (i) Loewdin orthonormalization, i.e. :math:`S^{-1/2}` applied to
    the cluster overlap matrix :math:`S`, which yields the orthonormal basis closest to the input vectors. A cluster
    of (nearly) linearly dependent vectors is exempt from it and from the orientation of step (ii), since
    :math:`S^{-1/2}` would amplify exactly the noise that makes such vectors distinct; it is still normalized and
    phase-fixed, so step (iii) holds for every returned vector without exception; (ii)
    for doublets, the mirror operation

    .. math:: M_y: \Delta_{o_1 o_2}(k_x, k_y, k_z, \nu) \to
        \left[U_y\, \Delta(k_x, -k_y, k_z, \nu)\, U_y^\dagger\right]_{o_1 o_2}

    is diagonalized within the cluster, ordering the even (:math:`+1`, :math:`p_x`-like) partner first and the
    odd (:math:`-1`, :math:`p_y`-like) partner second, but only when the two partners come out as clean, oppositely
    signed :math:`\pm 1` eigenstates (an :math:`E_g` doublet, even under every coordinate mirror, is not resolved
    this way and keeps the Loewdin basis); every cluster of three or more members is handled by
    :func:`_orient_cluster_by_mirrors`, which diagonalizes the single-axis coordinate mirrors of the resolved axes
    simultaneously and orders the partners by the axes each one is odd under (:math:`p`-like as ``x, y, z``,
    two-axis :math:`d`-like as ``xy, xz, yz``), provided every partner is a clean :math:`\pm 1` eigenstate and no
    two partners share a sign pattern (otherwise the mirrors do not resolve the cluster and the Loewdin basis is
    kept); (iii) the global phase of every vector is fixed such that its first element (in flattened order) within a
    relative :math:`10^{-3}` of the largest magnitude is real and positive. The band lies far above the eigenvector
    noise of the single-precision solve, so the opposite-sign, symmetry-equivalent largest elements of a
    :math:`d`- or :math:`p`-wave gap are told apart by their position rather than by rounding, and the global sign does
    not depend on the rank count, BLAS or thread count. Eigenvalues are not modified; vectors of non-degenerate
    eigenvalues are only phase-fixed. Enabled via ``symmetrize_degenerate_gaps`` of
    :class:`~dgamore.config.EliashbergConfig`.

    The mirrors act on the orbital indices as well as on the momenta (:math:`U_y` above, see
    :func:`_mirror_operator`). Without that orbital factor the operator is not a symmetry of a multi-orbital pairing
    kernel, and any multiplet with weight in both the orbital-diagonal and the orbital-off-diagonal sector fails the
    :math:`\pm 1` check and falls back to the Loewdin basis; with it, purely local (momentum-independent) multiplets
    become resolvable too, which a momentum-only mirror cannot do because it acts on them as the identity.

    :param lambdas: Eigenvalues sorted in descending order.
    :param gaps: Eigenvector matrix ``[n, n_eig]`` with one flattened gap function per column.
    :param gap_shape: Full gap shape ``[kx, ky, kz, o1, o2, 2*niv_pp]``, used to locate the momentum axes.
    :param tol: Relative tolerance for clustering neighboring eigenvalues as degenerate.
    :param orbital_mirrors: A dict ``{axis: U}`` with the orbital part of each coordinate mirror, as produced by
        :func:`_gap_orbital_mirrors`. Omitting it reduces the mirrors to momentum-only reflections, which is exact
        for a single-orbital gap only.
    :return: The symmetrized eigenvector matrix ``[n, n_eig]``.
    """
    mirrors = _validated_orbital_mirrors(gap_shape, orbital_mirrors)
    mirror_y = _mirror_operator(gap_shape, 1, mirrors.get(1))

    clusters = [[0]]
    for i in range(1, gaps.shape[1]):
        if abs(lambdas[i] - lambdas[i - 1]) <= tol * max(abs(lambdas[i]), 1e-12):
            clusters[-1].append(i)
        else:
            clusters.append([i])

    gaps = gaps.copy()
    for cluster in clusters:
        block = gaps[:, cluster].astype(np.complex128)
        block /= np.linalg.norm(block, axis=0)

        independent = True
        if len(cluster) > 1:
            overlap = block.conj().T @ block
            eigs, u = np.linalg.eigh(overlap)
            # dependent vectors keep only the stable normalization and phase fix; S^{-1/2} and the mirror step
            # would amplify the noise that makes them distinct (see the step (i) note in the docstring)
            independent = bool(eigs.min() >= 1e-12)
            if independent:
                block = block @ (u @ np.diag(eigs**-0.5) @ u.conj().T)

        if independent and len(cluster) == 2:
            mirrored = np.stack([mirror_y(block[:, i]) for i in range(2)], axis=1)
            mirror_block = block.conj().T @ mirrored
            mirror_block = 0.5 * (mirror_block + mirror_block.conj().T)
            mirror_eigs, mirror_vecs = np.linalg.eigh(mirror_block)
            # rotate only when M_y resolves the doublet into clean, oppositely signed +/-1 eigenstates; otherwise
            # keep the Loewdin basis (an E_g doublet is even under every mirror, so noise would fix the rotation)
            if np.all(np.abs(np.abs(mirror_eigs) - 1.0) <= _MIRROR_TOL) and mirror_eigs[0] * mirror_eigs[1] < 0.0:
                # order the even (+1, p_x-like) partner first and the odd (-1, p_y-like) partner second
                block = block @ mirror_vecs[:, ::-1]
        elif independent and len(cluster) >= 3:
            oriented = _orient_cluster_by_mirrors(block, gap_shape, orbital_mirrors=mirrors)
            if oriented is not None:
                block = oriented

        for col in range(block.shape[1]):
            mags = np.abs(block[:, col])
            if mags.max() == 0:  # an all-zero vector has no phase to fix (and dividing by it would yield nan)
                continue
            # first element within 1e-3 of the largest modulus: the band lies far above the eigenvector noise, so
            # opposite-sign symmetry partners among the largest elements (d-, p-wave) are told apart by position
            phase = block[np.flatnonzero(mags >= mags.max() * (1.0 - 1e-3))[0], col]
            block[:, col] *= phase.conjugate() / abs(phase)
        gaps[:, cluster] = block

    return gaps


def _apply_gchi0_pp(
    chi0_mm: np.ndarray, gap: np.ndarray, n_bands: int, executor: ThreadPoolExecutor = None, n_workers: int = 1
) -> np.ndarray:
    r"""
    Batched-matmul equivalent of ``np.einsum("xyzabcdv,xyzcdv->xyzabv", chi0, gap)`` (multiply the gap by the bare pp
    bubble per momentum and frequency). ``np.matmul`` is both faster than ``np.einsum`` and far leaner here: einsum
    materializes a vertex-sized internal temporary, the matmul allocates only the gap-sized output. With an
    ``executor`` the batch is split into up to ``n_workers`` contiguous chunks of the leading momentum axis and
    contracted concurrently: the chunks are pure slices of ``chi0_mm`` (no reshape of the bubble that could
    silently copy it) and every worker writes its own slice of the one gap-sized output buffer, so the threaded
    path allocates exactly what the serial path does and the result is bit-equal to it.

    :param chi0_mm: The bubble in matmul layout from :func:`_chi0_to_matmul_layout`, shape ``[x, y, z, v, o2, o2]``.
    :param gap: The gap vector, reshapeable to ``[x, y, z, o, o, v]``.
    :param n_bands: Number of orbitals ``o``.
    :param executor: Optional thread pool for the momentum-batch parallel path (``None`` runs serially).
    :param n_workers: Number of contiguous momentum chunks when ``executor`` is given.
    :return: ``chi0 @ gap`` in shape ``[x, y, z, o, o, v]``.
    """
    nqx, nqy, nqz, v = chi0_mm.shape[0], chi0_mm.shape[1], chi0_mm.shape[2], chi0_mm.shape[3]
    oo = n_bands * n_bands
    gap_r = np.moveaxis(gap.reshape(nqx, nqy, nqz, oo, v), -1, 3)[..., None]  # [x, y, z, v, o2, 1]
    if executor is None:
        out = np.matmul(chi0_mm, gap_r)[..., 0]  # [x, y, z, v, o2]
        return np.moveaxis(out, 3, -1).reshape(nqx, nqy, nqz, n_bands, n_bands, v)

    out = np.empty((nqx, nqy, nqz, v, oo, 1), dtype=np.result_type(chi0_mm.dtype, gap_r.dtype))
    bounds = np.linspace(0, nqx, n_workers + 1).astype(int)
    futures = [
        executor.submit(np.matmul, chi0_mm[i:j], gap_r[i:j], out[i:j]) for i, j in zip(bounds[:-1], bounds[1:]) if j > i
    ]
    for future in futures:
        future.result()
    return np.moveaxis(out[..., 0], 3, -1).reshape(nqx, nqy, nqz, n_bands, n_bands, v)


def _gamma_to_matmul_layout(gamma_mat: np.ndarray, scale: float = 1.0) -> np.ndarray:
    r"""
    Materializes the pp pairing vertex in batched-matmul layout ``[x, y, z, o2*nv, o2*np]`` (rows ``(a, b, v)``,
    columns ``(c, d, p)``) for :func:`_apply_gamma_pp`, scaled by ``scale``. The einsum layout is
    ``[x, y, z, a, c, b, d, v, p]`` with the orbitals interleaved (``a, c, b, d``), so a transpose to
    ``(a, b, v, c, d, p)`` precedes the copying reshape; a scale factor other than one rides along with that single
    pass (the same one multiply per element as scaling the layout afterwards). The per-matvec ``np.matmul`` then
    allocates only the gap-sized output.

    :param gamma_mat: The pp vertex, shape ``[x, y, z, a, c, b, d, v, p]`` (``v`` may be a frequency slice).
    :param scale: Factor multiplied onto every element, e.g. the kernel prefactor (one leaves the values untouched).
    :return: A contiguous array ``[x, y, z, o2*nv, o2*np]`` in matmul layout.
    """
    nqx, nqy, nqz, nb = gamma_mat.shape[:4]
    nv, npp = gamma_mat.shape[-2], gamma_mat.shape[-1]
    out = np.empty((nqx, nqy, nqz, nb * nb * nv, nb * nb * npp), dtype=gamma_mat.dtype)
    transposed = np.transpose(gamma_mat, (0, 1, 2, 3, 5, 7, 4, 6, 8))  # [x,y,z,a,b,v,c,d,p]
    np.multiply(transposed, scale, out=out.reshape(nqx, nqy, nqz, nb, nb, nv, nb, nb, npp))
    return out


def _apply_gamma_pp(
    gamma_mm: np.ndarray, gap_gg: np.ndarray, n_bands: int, executor: ThreadPoolExecutor = None, n_workers: int = 1
) -> np.ndarray:
    r"""
    Batched-matmul equivalent of ``np.einsum("xyzacbdvp,xyzcdp->xyzabv", gamma, gap_gg)`` (contract the pairing vertex
    with the gap over ``(c, d, p)``). Faster and leaner than ``np.einsum`` (see :func:`_apply_gchi0_pp`). With an
    ``executor`` the momentum batch is split into ``n_workers`` contiguous chunks contracted concurrently (each
    worker writes its own output slice, so the result is bit-equal to the serial path): the contraction is a batch
    of many small per-k GEMV products, which parallelizes over the batch but not inside one product (a raised BLAS
    thread pool pays per-call synchronization on every small GEMV and runs slower).

    :param gamma_mm: The vertex in matmul layout from :func:`_gamma_to_matmul_layout`, shape ``[x, y, z, o2*nv, o2*np]``.
    :param gap_gg: The transformed gap, shape ``[x, y, z, c, d, p]``.
    :param n_bands: Number of orbitals ``o``.
    :param executor: Optional thread pool for the momentum-batch parallel path (``None`` runs serially).
    :param n_workers: Number of contiguous momentum chunks when ``executor`` is given.
    :return: ``gamma @ gap_gg`` in shape ``[x, y, z, o, o, nv]``.
    """
    nqx, nqy, nqz = gamma_mm.shape[:3]
    oo = n_bands * n_bands
    npp = gap_gg.shape[-1]
    nv = gamma_mm.shape[3] // oo
    gg_r = gap_gg.reshape(nqx, nqy, nqz, oo * npp)[..., None]  # [x, y, z, o2*np, 1]
    if executor is None:
        out = np.matmul(gamma_mm, gg_r)[..., 0]  # [x, y, z, o2*nv]
        return out.reshape(nqx, nqy, nqz, n_bands, n_bands, nv)

    nk = nqx * nqy * nqz
    mm_flat = gamma_mm.reshape(nk, oo * nv, oo * npp)
    gg_flat = gg_r.reshape(nk, oo * npp, 1)
    out = np.empty((nk, oo * nv, 1), dtype=gamma_mm.dtype)
    bounds = np.linspace(0, nk, n_workers + 1).astype(int)
    futures = [
        executor.submit(np.matmul, mm_flat[i:j], gg_flat[i:j], out[i:j])
        for i, j in zip(bounds[:-1], bounds[1:])
        if j > i
    ]
    for future in futures:
        future.result()
    return out[..., 0].reshape(nqx, nqy, nqz, n_bands, n_bands, nv)


@lru_cache(maxsize=1)
def _wedge_orbits(k_grid: KGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    r"""
    Groups the flat indices of the real-space grid into the stars of the grid's point-like symmetry operations
    (index permutations with orbital rotations, no translations, no antiunitary members), under which the
    Fourier-transformed vertex obeys :math:`\Gamma(P\mathbf{R}) = U_P\,\Gamma(\mathbf{R})\,U_P^\dagger` with the same
    rotation as in momentum space. On an explicit-symmetry grid these are the grid's own irreducible-BZ maps; on an
    auto-symmetry grid they come from :func:`~dgamore.symmetry_reduction.point_group_orbits`. The result of the last
    grid is cached.

    :param k_grid: The momentum grid (its index maps apply to the real-space grid alike).
    :return: ``(points, offsets, order, us)``: the representative grid index of every star, the star boundaries
        into ``order``, every grid index sorted by star, and per grid index the rotation carrying the representative
        onto it (``None`` when the grid applies no orbital rotation).
    """
    if k_grid.is_auto:
        rep, us = point_group_orbits(k_grid._auto_group, tuple(k_grid.nk))
        if np.allclose(us, np.round(us), rtol=0.0, atol=1e-8):
            # orbital permutations with phases are taken exact, as in KGrid.auto_point_group
            us = np.round(us)
        reps, inv = np.unique(rep, return_inverse=True)
    else:
        reps, inv, us = k_grid.irrk_ind, k_grid.irrk_inv.ravel(), None
    order = np.argsort(inv, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(np.bincount(inv, minlength=len(reps)))))
    return reps, offsets, order, us


_SYMMETRY_MATRICES = {
    KnownSymmetries.X_INV: np.diag([-1, 1, 1]),
    KnownSymmetries.Y_INV: np.diag([1, -1, 1]),
    KnownSymmetries.Z_INV: np.diag([1, 1, -1]),
    KnownSymmetries.X_Y_SYM: np.eye(3, dtype=np.int64)[[1, 0, 2]],
    KnownSymmetries.X_Z_SYM: np.eye(3, dtype=np.int64)[[2, 1, 0]],
    KnownSymmetries.Y_Z_SYM: np.eye(3, dtype=np.int64)[[0, 2, 1]],
    KnownSymmetries.X_Y_INV: np.diag([-1, -1, 1]),
}


def block_size(k_grid: KGrid, parity: int | None = None) -> int:
    r"""
    Returns the number of starting vectors the Eliashberg eigensolver takes with ``resolve_degenerate_multiplets``, so
    that every eigenvalue multiplicity the grid's symmetry forces is found in exact arithmetic (see :func:`~dgamore.symmetry_reduction.forced_multiplicity`):
    the discovered group on an auto-symmetry grid, the group generated by the listed operations otherwise. A
    single-vector Krylov method reaches one direction of each eigenspace, so the partners of a forced multiplet only
    enter through rounding and can be missing from the converged eigenpairs.

    :param k_grid: The momentum grid.
    :param parity: The momentum-inversion parity of the sector (one band), or ``None`` for every representation.
    :return: The block size, 1 when the group forces no multiplet or ``resolve_degenerate_multiplets`` is off.
    """
    if not config.eliashberg.resolve_degenerate_multiplets:
        return 1
    nk = tuple(k_grid.nk)
    if k_grid.is_auto:
        return forced_multiplicity(k_grid._auto_group, nk, parity)
    matrices = [_SYMMETRY_MATRICES.get(sym) for sym in k_grid.symmetries or []]
    ops = [
        {"M": m, "q": np.zeros(3, dtype=np.int64), "U": np.eye(1), "sigma": 1, "conj": False}
        for m in matrices
        if m is not None and M_preserves_grid(m, nk)
    ]
    return forced_multiplicity(close_group(ops, 1, nk), nk, parity)


def wedge_window_points(k_grid: KGrid) -> int:
    r"""
    Returns the number of real-space points a channel's vertex window holds in the team solve: the star
    representatives of the grid's point-like operations (see :func:`_wedge_orbits`), i.e. the whole grid when they
    reduce nothing.

    :param k_grid: The momentum grid.
    :return: The number of window points.
    """
    return len(_wedge_orbits(k_grid)[0])


def _build_shared_vertex_window(
    gamma_r_pp: FourPoint, node_comm: MPI.Comm, norm: float, wedge_points: np.ndarray | None = None
) -> tuple[np.ndarray, MPI.Win | None]:
    r"""
    Puts one channel's pairing vertex, in matmul layout, into a single MPI shared-memory window of its node. The
    node root holds the gathered irreducible-BZ vertex on entry and publishes it through a temporary node-shared
    window; every rank of the node then builds its share of the full-BZ matmul layout straight into the vertex
    window, one column block at a time (one fermionic frequency :math:`\nu` and a range of :math:`\nu'` columns,
    sized by :data:`~dgamore.memory_estimator.TEAM_BUILD_CHUNK_BYTES`): each block runs through the same
    :class:`FourPoint` chain as the single-rank build (map to the full BZ, Fourier transform over the momenta,
    reorder the legs ``abcd->badc``) and is written scaled by the kernel prefactor. Every step acts on each column
    independently, so the window equals the layout the single-rank solve builds bit for bit. With ``wedge_points``
    only the rows of those real-space points are kept (the star representatives of :func:`_wedge_orbits`). The
    node holds the windows, the irreducible source and one block per rank; the vertex is **consumed** on the node
    root.

    :param gamma_r_pp: The channel's pairing vertex, the whole irreducible BZ on the node root (ignored elsewhere).
    :param node_comm: The node-local communicator.
    :param norm: The kernel prefactor :math:`1 / (2 \beta n_{\mathbf{k}})` folded into the vertex.
    :param wedge_points: Flat indices of the real-space points to keep, or ``None`` for the whole grid.
    :return: ``(gamma_mm, win)`` - the shared matmul-layout vertex, ``[x, y, z, o2*nv, o2*np]`` on the full grid or
        ``[r, o2*nv, o2*np]`` on the wedge, and its window.
    """
    k_grid = config.lattice.k_grid
    size, rank = node_comm.Get_size(), node_comm.Get_rank()
    irr_shape, dtype = node_comm.bcast((gamma_r_pp.mat.shape, gamma_r_pp.mat.dtype) if rank == 0 else None, root=0)
    irr, irr_win = gamma_r_pp.mat, None
    if size > 1:
        irr, irr_win = mpi_utils.allocate_node_shared_array(node_comm, irr_shape, dtype)
        if rank == 0:
            irr[...] = gamma_r_pp.mat
            gamma_r_pp.free()
        node_comm.Barrier()

    nb, nv, npp = irr_shape[1], irr_shape[-2], irr_shape[-1]
    rows, cols = nb * nb * nv, nb * nb * npp
    shape = tuple(k_grid.nk) + (rows, cols) if wedge_points is None else (len(wedge_points), rows, cols)
    gamma_mm, win = mpi_utils.allocate_node_shared_array(node_comm, shape, dtype)
    target = gamma_mm.reshape(-1, nb, nb, nv, nb, nb, npp)
    columns = team_build_columns(k_grid.nk_tot, nb, npp)
    chunks = [(v, p0, min(p0 + columns, npp)) for v in range(nv) for p0 in range(0, npp, columns)]
    for i in np.array_split(np.arange(len(chunks)), size)[rank]:
        v, p0, p1 = chunks[i]
        block = FourPoint(
            irr[..., v, p0:p1], gamma_r_pp.channel, k_grid.nk, 0, 1, False, True, True, FrequencyNotation.PP
        )
        block = block.map_to_full_bz(k_grid, k_grid.nk).decompress_q_dimension().fft(False)
        block = block.permute_orbitals("abcd->badc", False).mat.reshape(-1, nb, nb, nb, nb, p1 - p0)
        if wedge_points is not None:
            block = block[wedge_points]
        np.multiply(np.transpose(block, (0, 1, 3, 2, 4, 5)), norm, out=target[:, :, :, v, :, :, p0:p1])
    del irr, target
    node_comm.Barrier()
    mpi_utils.free_shared_window(irr_win, node_comm)
    if size == 1:
        gamma_r_pp.free()
    return gamma_mm, win


def _member_rotations(orbits: tuple, r0: int, r1: int) -> np.ndarray | None:
    r"""
    Returns the orbital rotations of the members of the stars ``r0..r1`` of :func:`_wedge_orbits` in matmul layout,
    :math:`W_{(12),(ab)} = U^*_{1a} U^*_{2b}` with :math:`U` the rotation carrying the star's representative onto the
    member (both pairs of the matmul layout carry the conjugate unitary), or ``None`` when the grid applies none.

    :param orbits: The stars of :func:`_wedge_orbits`.
    :param r0: First star.
    :param r1: One past the last star.
    :return: The rotations, shape ``[members, o^2, o^2]`` in star order, or ``None``.
    """
    _, offsets, order, us = orbits
    if us is None:
        return None
    members = us[order[offsets[r0] : offsets[r1]]]
    oo = members.shape[-1] ** 2
    return np.einsum("kab,kcd->kacbd", members.conj(), members.conj()).reshape(len(members), oo, oo)


def _star_batches(offsets: np.ndarray, r0: int, r1: int, row_bytes: int) -> list[tuple[int, int]]:
    r"""
    Splits the stars ``r0..r1`` of :func:`_wedge_orbits` into runs of consecutive stars of one size whose right-hand
    sides take at most :data:`STAR_BATCH_BYTES` (a star above the budget runs alone), the batches of
    :func:`_contract_stars`.

    :param offsets: The star boundaries of :func:`_wedge_orbits`.
    :param r0: First star.
    :param r1: One past the last star.
    :param row_bytes: Bytes of one member's right-hand side.
    :return: The batches as ``(first star, one past the last star)``.
    """
    sizes = np.diff(offsets[r0 : r1 + 1])
    batches, a = [], r0
    while a < r1:
        size, b = sizes[a - r0], a + 1
        cap = max(1, STAR_BATCH_BYTES // int(size * row_bytes))
        while b < r1 and b - a < cap and sizes[b - r0] == size:
            b += 1
        batches.append((a, b))
        a = b
    return batches


def _contract_stars(gamma: np.ndarray, rhs: np.ndarray, w: np.ndarray | None, nv: int) -> np.ndarray:
    r"""
    Contracts consecutive stars of one size of the real-space wedge in one stacked matrix product: every
    representative's matmul-layout vertex times its members' right-hand sides as columns, rotated into the
    representative's orbital frame with :math:`W^\dagger` and out of it with :math:`W` (see
    :func:`_member_rotations`). Every star's product takes the operands a product of that star alone takes, so
    without rotations the result does not depend on the batches.

    :param gamma: The representatives' vertices, ``[stars, (a, b, v), (c, d, p)]``.
    :param rhs: The members' right-hand sides in star order, ``[members, (c, d, p)]``.
    :param w: The members' rotations ``[members, o^2, o^2]``, or ``None`` for none.
    :param nv: Number of fermionic frequencies.
    :return: The members' direct terms, ``[members, (a, b, v)]``.
    """
    n_stars, rows = gamma.shape[:2]
    oo = rows // nv
    if w is not None:
        rhs = np.matmul(w.conj().transpose(0, 2, 1), rhs.reshape(-1, oo, nv)).reshape(-1, rows)
    cols = rhs.reshape(n_stars, -1, rows).transpose(0, 2, 1)
    if w is not None:
        cols = np.ascontiguousarray(cols)
    out = np.matmul(gamma, cols).transpose(0, 2, 1).reshape(-1, rows)
    if w is not None:
        out = np.matmul(w, out.reshape(-1, oo, nv)).reshape(-1, rows)
    return out


def _team_matvec(
    team_comm: MPI.Comm,
    gamma_by_channel: dict[SpinChannel, np.ndarray],
    chi0_mm: np.ndarray,
    scratch: list[np.ndarray],
    gap_shape: tuple,
    n_bands: int,
    orbits: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None = None,
    u: np.ndarray | None = None,
) -> tuple[Callable, tuple[int, int]]:
    r"""
    Builds the pairing-kernel matvec of a team of ranks that share their node's vertex windows. Every rank holds one
    frequency block of the vectors and every matvec runs in lockstep stages over the team's gap-sized scratch windows
    (the gap, its dressed transform, the direct and the crossed term), each rank working on its own share: the sector
    projection of the input (see :func:`_project_frequency_block`), the bubble multiply and the momentum FFT on a
    contiguous block of fermionic frequencies, the vertex contraction on a contiguous block of momenta, the crossed term
    of the frequency block (see :func:`_crossed_term`, which reads the mirror block of the direct term, so it is staged
    in its window before any block is overwritten), its addition and the inverse FFT, and for a projected sector the
    projection of the result on that block. Every element is computed by the same operations as in the single-rank
    sector matvec, so the result is bit-identical to it without an orbital action or with an orbital permutation, and
    agrees to rounding with any other action (the orbital rotation's einsum picks its contraction path by the block's
    shape). With ``orbits`` the windows hold the irreducible wedge of the real-space grid and each rank contracts the
    stars of its irreducible points instead, the consecutive stars of one size in one stacked matrix product (see
    :func:`_contract_stars`) with the members' right-hand sides as columns (rotated into and out of the irreducible
    point's orbital frame on an auto-symmetry grid: the layout's row and column pairs both carry the conjugate unitary,
    so :math:`W_{(12),(ab)} = U^*_{1a} U^*_{2b}` enters as :math:`W^\dagger` on the right-hand sides and as :math:`W` on
    the result); the result then equals the single-rank one to rounding.

    :param team_comm: The team communicator (rank 0 is the lead).
    :param gamma_by_channel: The node-shared matmul-layout vertices per channel, ``[x, y, z, o2*nv, o2*np]`` on the
        full grid or ``[r, o2*nv, o2*np]`` on the wedge.
    :param chi0_mm: The node-shared bubble in matmul layout (see :func:`_chi0_to_matmul_layout`).
    :param scratch: Four gap-shaped team-shared arrays.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param n_bands: Number of orbitals ``o``.
    :param orbits: The stars of :func:`_wedge_orbits` for wedge windows, else ``None``.
    :param u: The orbital action of the inversion for the parity sectors (see :func:`_project_gap_to_sector`).
    :return: ``(apply_block, (v0, v1))``: the collective sector matvec on this rank's frequency block
        ``[..., v0:v1]`` of the gap, ``apply_block(channel, eps_t, eps_p, block) -> block`` (``eps_t`` of ``None``
        applies the raw kernel; every team rank calls it with the same sector), and the block bounds.
    """
    gap_w, gg_w, direct_w, crossed_w = scratch
    size, rank = team_comm.Get_size(), team_comm.Get_rank()
    nqx, nqy, nqz = gap_shape[:3]
    nk, nv, oo = nqx * nqy * nqz, gap_shape[-1], n_bands * n_bands
    rows = oo * nv
    columns = np.linspace(0, nv, size + 1).astype(int)
    v0, v1 = int(columns[rank]), int(columns[rank + 1])
    chunks = np.linspace(0, nk, size + 1).astype(int)
    k0, k1 = int(chunks[rank]), int(chunks[rank + 1])
    gg_flat = gg_w.reshape(nk, rows, 1)
    direct_flat = direct_w.reshape(nk, rows, 1)
    if orbits is not None:
        _, offsets, order, us = orbits
        # this rank's irreducible points, balanced by the number of star members (the matrix-product columns)
        bounds = np.searchsorted(offsets, np.linspace(0, nk, size + 1))
        r0, r1 = int(bounds[rank]), int(bounds[rank + 1])
        rotations = _member_rotations(orbits, r0, r1)
        batches = _star_batches(offsets, r0, r1, rows * gg_w.itemsize)

    def contract_stars(gamma_irr: np.ndarray) -> None:
        for a, b in batches:
            members = order[offsets[a] : offsets[b]]
            w = None if rotations is None else rotations[offsets[a] - offsets[r0] : offsets[b] - offsets[r0]]
            direct_flat[members, :, 0] = _contract_stars(gamma_irr[a:b], gg_flat[members, :, 0], w, nv)

    def run(channel: SpinChannel, eps_t: int | None, eps_p: int | None) -> None:
        gamma_flat = gamma_by_channel[channel].reshape(-1, rows, rows)
        sign = 1 if channel == SpinChannel.SING else -1
        team_comm.Barrier()
        if v1 > v0:
            block = gap_w[..., v0:v1] if eps_t is None else _project_frequency_block(gap_w, v0, v1, eps_t, eps_p, u)
            dressed = _apply_gchi0_pp(chi0_mm[:, :, :, v0:v1], block, n_bands)
            gg_w[..., v0:v1] = sp.fft.fftn(dressed, axes=(0, 1, 2), overwrite_x=True)
            del block, dressed
        team_comm.Barrier()
        if orbits is not None:
            contract_stars(gamma_flat)
        elif k1 > k0:
            np.matmul(gamma_flat[k0:k1], gg_flat[k0:k1], out=direct_flat[k0:k1])
        team_comm.Barrier()
        if v1 > v0:
            crossed_w[..., v0:v1] = _crossed_term(direct_w, sign, v0, v1)
        team_comm.Barrier()
        if v1 > v0:
            direct_w[..., v0:v1] += crossed_w[..., v0:v1]
            direct_w[..., v0:v1] = sp.fft.ifftn(direct_w[..., v0:v1], axes=(0, 1, 2), overwrite_x=True)
        team_comm.Barrier()
        if eps_t is not None:
            # the projected result goes to the (no longer needed) gap window; every rank reads the mirror block
            if v1 > v0:
                gap_w[..., v0:v1] = _project_frequency_block(direct_w, v0, v1, eps_t, eps_p, u)
            team_comm.Barrier()

    def apply_block(channel: SpinChannel, eps_t: int | None, eps_p: int | None, block: np.ndarray) -> np.ndarray:
        gap_w[..., v0:v1] = block.reshape(gap_shape[:-1] + (v1 - v0,))
        run(channel, eps_t, eps_p)
        return (direct_w if eps_t is None else gap_w)[..., v0:v1].copy()

    return apply_block, (v0, v1)


@dataclass(frozen=True)
class LanczosTeam:
    """
    One team of the Eliashberg team solve: the ranks of one node that solve the given sectors together, sharing
    their node's vertex windows.

    :ivar host: The node's hostname.
    :ivar node_root: The lowest rank of the node (the rank that builds the node's vertex windows).
    :ivar ranks: The team's ranks in ascending order; the first is the lead that drives the eigensolver.
    :ivar sectors: The ``(channel, parity)`` sectors the lead solves, one after the other.
    """

    host: str
    node_root: int
    ranks: tuple[int, ...]
    sectors: tuple[tuple[SpinChannel, str], ...]


def plan_lanczos_teams(
    comm: MPI.Comm, node_comm: MPI.Comm, available_bytes: int, niv_pp: int
) -> tuple[tuple[LanczosTeam, ...], ...] | None:
    r"""
    Plans the team solve: the ``(channel, parity)`` sectors are spread as evenly as possible over the nodes (one
    node hosts all of them, two nodes one channel each, four nodes one sector each; nodes beyond the sector count
    host nothing), each channel's pairing vertex is built into a shared window on every node that hosts one of its
    sectors, and a node's ranks are split into one team per sector hosted there (a node with fewer ranks than
    sectors gives every rank a team of one that solves several sectors in turn). The nodes are the groups of
    ``node_comm``, keyed by the lowest rank of each, so the plan and the windows it drives can never disagree on
    who shares memory with whom. Every hosting node is checked against its own free memory with
    :func:`~dgamore.memory_estimator.lanczos_team_bytes`: when a node cannot hold its share at once the channels are
    solved in two rounds, each spread over the nodes the same way, and when even that does not fit the plan is
    ``None`` (the node-distributed solve of :func:`plan_distributed_solve` takes over).

    :param comm: The MPI communicator.
    :param node_comm: The node-local communicator.
    :param available_bytes: This rank's free host memory, allgathered and reduced (minimum) per node.
    :param niv_pp: Number of positive fermionic frequencies of the pp box.
    :return: The rounds of the solve, each a tuple of teams that run concurrently, or ``None`` when no round fits.
    """
    node_ranks, node_host, node_available = mpi_utils.node_groups(comm, node_comm, available_bytes)
    nodes = list(node_ranks)
    parities = [label for label, _ in _parity_sectors(config.eliashberg.resolve_parity_sectors)]
    sectors = [(channel, parity) for channel in (SpinChannel.SING, SpinChannel.TRIP) for parity in parities]

    def spread(group: list[tuple[SpinChannel, str]]) -> dict[int, list[tuple[SpinChannel, str]]]:
        # ponytail: at most one node per sector; splitting one matvec across nodes is the distributed solve's job
        chunks = np.array_split(np.arange(len(group)), min(len(nodes), len(group)))
        return {nodes[i]: [group[j] for j in chunk] for i, chunk in enumerate(chunks)}

    def fits(plan: list[dict[int, list[tuple[SpinChannel, str]]]]) -> bool:
        for hosted in plan:
            for node, node_sectors in hosted.items():
                need = lanczos_team_bytes(
                    config.sys.n_bands,
                    config.lattice.k_grid.nk_tot,
                    config.lattice.k_grid.nk_irr,
                    niv_pp,
                    config.eliashberg.n_eig,
                    len({channel for channel, _ in node_sectors}),
                    len(node_sectors),
                    len(node_ranks[node]),
                    wedge_window_points(config.lattice.k_grid),
                    block_size(config.lattice.k_grid),
                )
                if need > node_available[node] * NODE_MEMORY_FRACTION:
                    return False
        return True

    plan = [spread(sectors)]
    if not fits(plan):
        plan = [spread([s for s in sectors if s[0] == channel]) for channel in (SpinChannel.SING, SpinChannel.TRIP)]
        if not fits(plan):
            return None

    rounds = []
    for hosted in plan:
        teams = []
        for node, node_sectors in hosted.items():
            ranks = node_ranks[node]
            n_teams = min(len(node_sectors), len(ranks))
            for i, group in enumerate(np.array_split(np.array(ranks), n_teams)):
                teams.append(
                    LanczosTeam(node_host[node], node, tuple(int(r) for r in group), tuple(node_sectors[i::n_teams]))
                )
        rounds.append(tuple(teams))
    return tuple(rounds)


@dataclass(frozen=True)
class DistributedLayout:
    r"""
    The split of the node-distributed team solve over the nodes, identical on every rank.

    :ivar nodes: The ranks of every node in ascending order, the nodes ordered by their lowest rank.
    :ivar freqs: Per node, its fermionic frequency indices in ascending order: the pairs :math:`(\nu, -\nu)` of a
        contiguous range of pairs, so that the flip of a node's frequency axis is the global mirror.
    :ivar points: Per node, the real-space grid indices of its window: the members of its stars in the order of
        :func:`_wedge_orbits`, or a contiguous range of the grid without wedge windows.
    :ivar stars: Per node, its range of stars of :func:`_wedge_orbits`, or ``None`` without wedge windows.
    :ivar builders: Number of a node's ranks that expand vertex columns at once while a window is built.
    """

    nodes: tuple[tuple[int, ...], ...]
    freqs: tuple[np.ndarray, ...]
    points: tuple[np.ndarray, ...]
    stars: tuple[tuple[int, int], ...] | None
    builders: int


def plan_distributed_solve(comm: MPI.Comm, node_comm: MPI.Comm, available_bytes: int, niv_pp: int) -> DistributedLayout:
    r"""
    Plans the node-distributed team solve, used when no node holds one channel's vertex windows: every node owns the
    pairs :math:`(\nu, -\nu)` of a contiguous range of the ``niv_pp`` pairs (nodes beyond the pair count own none)
    and a contiguous run of stars of the real-space wedge (see :func:`_wedge_orbits`), balanced by their members, or
    a contiguous range of the grid when the grid has no point-like operations. The nodes are the groups of
    ``node_comm`` (see :func:`plan_lanczos_teams`). As many of every node's ranks build vertex columns at once as
    every node holds by :func:`~dgamore.memory_estimator.lanczos_distributed_bytes` (rank 0's extras on its node),
    at most the column blocks per node.

    :param comm: The MPI communicator.
    :param node_comm: The node-local communicator.
    :param available_bytes: This rank's free host memory, allgathered and reduced (minimum) per node.
    :param niv_pp: Number of positive fermionic frequencies of the pp box.
    :return: The layout of the solve.
    :raises MemoryError: If one builder per node does not fit every node.
    """
    node_ranks, _, node_available = mpi_utils.node_groups(comm, node_comm, available_bytes)
    nodes = tuple(tuple(ranks) for ranks in node_ranks.values())
    n_nodes, k_grid = len(nodes), config.lattice.k_grid
    nk = k_grid.nk_tot
    pairs = np.array_split(np.arange(niv_pp), n_nodes)
    freqs = tuple(np.concatenate((niv_pp - 1 - p[::-1], niv_pp + p)) for p in pairs)
    window_points = wedge_window_points(k_grid)
    stars = None
    if window_points < nk:
        _, offsets, order, _ = _wedge_orbits(k_grid)
        cuts = np.searchsorted(offsets, np.linspace(0, nk, n_nodes + 1))
        stars = tuple((int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]))
        points = tuple(order[offsets[a] : offsets[b]] for a, b in stars)
    else:
        cuts = np.linspace(0, nk, n_nodes + 1).astype(int)
        points = tuple(np.arange(a, b) for a, b in zip(cuts[:-1], cuts[1:]))

    n_sectors = 2 * len(_parity_sectors(config.eliashberg.resolve_parity_sectors))
    npp = 2 * niv_pp
    n_chunks = npp * -(-npp // team_build_columns(nk, config.sys.n_bands, npp))

    def need(ranks: tuple[int, ...], builders: int) -> float:
        return lanczos_distributed_bytes(
            config.sys.n_bands,
            nk,
            k_grid.nk_irr,
            niv_pp,
            config.eliashberg.n_eig,
            n_nodes,
            n_sectors,
            window_points,
            block_size(k_grid),
            builders,
            0 in ranks,
        )

    top = min(min(len(ranks) for ranks in nodes), -(-n_chunks // n_nodes))
    fitting = [
        b
        for b in range(1, top + 1)
        if all(need(ranks, b) <= node_available[ranks[0]] * NODE_MEMORY_FRACTION for ranks in nodes)
    ]
    if not fitting:
        worst = max(need(ranks, 1) for ranks in nodes)
        raise MemoryError(
            f"The distributed Eliashberg solve needs {worst / 1024**3:.3f} GB on a node, which exceeds "
            f"{NODE_MEMORY_FRACTION:.0%} of that node's available memory. Use more nodes, fewer ranks per node, a "
            "smaller frequency box or k-grid."
        )
    return DistributedLayout(nodes, freqs, points, stars, max(fitting))


def _node_index(layout: DistributedLayout, rank: int) -> int:
    r"""
    Returns the index of the node holding ``rank`` in the layout of the node-distributed solve.

    :param layout: The layout of the solve.
    :param rank: The rank.
    :return: The node's index into ``layout.nodes``.
    """
    return next(i for i, ranks in enumerate(layout.nodes) if rank in ranks)


def _window_rows(layout: DistributedLayout, node: int) -> np.ndarray:
    r"""
    Returns the real-space grid indices of one node's window rows in the node-distributed solve: the representatives
    of its stars (see :func:`_wedge_orbits`), or its range of the grid without wedge windows.

    :param layout: The layout of the solve.
    :param node: The node's index.
    :return: The grid indices of the node's window rows.
    """
    if layout.stars is None:
        return layout.points[node]
    s0, s1 = layout.stars[node]
    return _wedge_orbits(config.lattice.k_grid)[0][s0:s1]


def _build_distributed_window(
    gamma_r_pp: FourPoint, mpi_dist: MpiDistributor, node_comm: MPI.Comm, layout: DistributedLayout, norm: float
) -> tuple[np.ndarray, MPI.Win | None]:
    r"""
    Builds one channel's pairing vertex, in matmul layout and scaled by the kernel prefactor, into one node-shared
    window per node holding that node's rows (see :func:`_window_rows`), without the vertex ever gathered on one
    rank. The momentum-distributed irreducible vertex is transposed (:func:`~dgamore.mpi_utils.transpose_columns`) so
    that the first ``layout.builders`` ranks of every node hold every irreducible momentum of their column blocks
    (one fermionic frequency :math:`\nu` and a range of :math:`\nu'` columns, sized by
    :func:`~dgamore.memory_estimator.team_build_columns`). In rounds of one block per builder, every block runs
    through the chain of :func:`_build_shared_vertex_window` (map to the full BZ, Fourier transform over the momenta,
    reorder the legs ``abcd->badc``), its rows of every node travel to one rank of that node, and that rank writes
    them scaled into the window. Every element is formed by the same operations as in the single-node window. The
    vertex is **consumed** on every rank.

    :param gamma_r_pp: This rank's irreducible momenta of the channel's pairing vertex (compressed momentum axis).
    :param mpi_dist: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param node_comm: The node-local communicator.
    :param layout: The layout of the solve.
    :param norm: The kernel prefactor :math:`1 / (2 \beta n_{\mathbf{k}})` folded into the vertex.
    :return: ``(gamma_mm, win)`` - this node's window ``[r, o2*nv, o2*np]`` and its MPI window.
    """
    comm, k_grid = mpi_dist.comm, config.lattice.k_grid
    rank, node = comm.rank, _node_index(layout, comm.rank)
    channel, nb, dtype = gamma_r_pp.channel, gamma_r_pp.n_bands, gamma_r_pp.mat.dtype
    nv, npp = gamma_r_pp.current_shape[-2], gamma_r_pp.current_shape[-1]
    rows = _window_rows(layout, node)
    gamma_mm, win = mpi_utils.allocate_node_shared_array(node_comm, (len(rows), nb * nb * nv, nb * nb * npp), dtype)
    target = gamma_mm.reshape(-1, nb, nb, nv, nb, nb, npp)

    columns = team_build_columns(k_grid.nk_tot, nb, npp)
    chunks = [(v, p0, min(p0 + columns, npp)) for v in range(nv) for p0 in range(0, npp, columns)]
    builders = [r for ranks in layout.nodes for r in ranks[: layout.builders]]
    owned = dict(zip(builders, np.array_split(np.arange(len(chunks)), len(builders))))
    cols_of = []
    for r in range(comm.size):
        spans = [chunks[i] for i in owned.get(r, [])]
        v_idx = [v for v, p0, p1 in spans for _ in range(p0, p1)]
        p_idx = [p for _, p0, p1 in spans for p in range(p0, p1)]
        cols_of.append(np.array([v_idx, p_idx], dtype=int).reshape(2, -1))
    received = mpi_utils.transpose_columns(gamma_r_pp.mat, mpi_dist, cols_of)
    gamma_r_pp.free()
    mine = owned.get(rank, [])
    starts = np.concatenate(([0], np.cumsum([chunks[i][2] - chunks[i][1] for i in mine]))).astype(int)

    for step in range(max(len(chunk_ids) for chunk_ids in owned.values())):
        reqs, sends, incoming = [], [], []
        if step < len(mine):
            v, p0, p1 = chunks[mine[step]]
            cols = received[..., starts[step] : starts[step + 1]]
            block = FourPoint(cols, channel, k_grid.nk, 0, 1, False, True, True, FrequencyNotation.PP)
            block = block.map_to_full_bz(k_grid, k_grid.nk).decompress_q_dimension().fft(False)
            block = block.permute_orbitals("abcd->badc", False).mat.reshape(-1, nb, nb, nb, nb, p1 - p0)
            for m, ranks in enumerate(layout.nodes):
                piece = np.ascontiguousarray(block[_window_rows(layout, m)])
                dest = ranks[mine[step] % len(ranks)]
                if dest == rank:
                    incoming.append((mine[step], piece))
                elif piece.size:
                    sends.append(piece)
                    reqs += mpi_utils.isend_rows(comm, piece, dest, base_tag=WINDOW_TAG)
            del block
        for builder, chunk_ids in owned.items():
            if (
                step < len(chunk_ids)
                and builder != rank
                and layout.nodes[node][chunk_ids[step] % len(layout.nodes[node])] == rank
            ):
                v, p0, p1 = chunks[chunk_ids[step]]
                piece = np.empty((len(rows), nb, nb, nb, nb, p1 - p0), dtype=dtype)
                if piece.size:
                    reqs += mpi_utils.irecv_rows_into(comm, piece, builder, base_tag=WINDOW_TAG)
                incoming.append((chunk_ids[step], piece))
        MPI.Request.Waitall(reqs)
        del sends
        for chunk_id, piece in incoming:
            v, p0, p1 = chunks[chunk_id]
            np.multiply(np.transpose(piece, (0, 1, 3, 2, 4, 5)), norm, out=target[:, :, :, v, :, :, p0:p1])
    del received, target
    node_comm.Barrier()
    return gamma_mm, win


def _distributed_scratch(
    node_comm: MPI.Comm, layout: DistributedLayout, rank: int, gap_shape: tuple, dtype
) -> tuple[list[np.ndarray], list]:
    r"""
    Allocates the node-shared arrays of the node-distributed matvec (see :func:`_node_team_matvec`): three arrays of
    the node's frequency share on the full momentum grid (the input or output, the transformed and later the direct
    term, the projection's intermediate and later the crossed term) and two of the node's star members at every
    frequency (the transformed gap and the direct term).

    :param node_comm: The node-local communicator.
    :param layout: The layout of the solve.
    :param rank: This rank.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param dtype: Storage dtype of the arrays.
    :return: ``(arrays, wins)`` - the five arrays and their MPI windows.
    """
    node = _node_index(layout, rank)
    nk, o, nv = int(np.prod(gap_shape[:3])), gap_shape[3], gap_shape[-1]
    shares = [(nk, o, o, len(layout.freqs[node]))] * 3 + [(len(layout.points[node]), o, o, nv)] * 2
    allocated = [mpi_utils.allocate_node_shared_array(node_comm, shape, dtype) for shape in shares]
    return [array for array, _ in allocated], [win for _, win in allocated]


def _rotate_rows(g: np.ndarray, u_k: np.ndarray | None, u_m: np.ndarray | None) -> np.ndarray:
    r"""
    Applies the orbital action of the inversion to both orbitals of a gap on momentum rows ``[k, o1, o2, v]``,
    :math:`U(\mathbf{k})\, g(\mathbf{k})\, U(-\mathbf{k})^T` (see :func:`_rotate_orbital_pair`), with a constant
    ``u_k`` (``u_m`` the same) or the rows' and their partners' momentum-dependent actions ``[k, o, o]``.

    :param g: The gap rows.
    :param u_k: The action at the rows, or ``None`` for none.
    :param u_m: The action at the partner rows :math:`-\mathbf{k}`.
    :return: The rotated rows (``g`` itself without an action).
    """
    if u_k is None:
        return g
    u_k, u_m = u_k.astype(g.dtype, copy=False), u_m.astype(g.dtype, copy=False)
    if u_k.ndim == 2:
        permuted = _permute_orbital_pair(g, u_k)
        return np.einsum("ac,kcdv,bd->kabv", u_k, g, u_m, optimize=True) if permuted is None else permuted
    return np.einsum("kac,kcdv,kbd->kabv", u_k, g, u_m, optimize=True)


def _node_team_matvec(
    comm: MPI.Comm,
    node_comm: MPI.Comm,
    layout: DistributedLayout,
    gamma_mm: np.ndarray,
    chi0_share: np.ndarray,
    arrays: list[np.ndarray],
    gap_shape: tuple,
    n_bands: int,
    u: np.ndarray | None = None,
) -> tuple[Callable, Callable, Callable, tuple[int, int]]:
    r"""
    Builds the pairing-kernel matvec of the node-distributed team (see :class:`DistributedLayout`) with the seeds and
    the eigenvector assembly of its layout. Every rank holds one momentum chunk of its node's frequency share of the
    vectors, and every matvec runs in stages over the node's shared arrays (:func:`_distributed_scratch`), separated
    by node barriers, with two exchanges between the nodes: the sector projection of the input on momentum chunks
    (the :math:`T O_U` part, then the inversion reading the partner rows at :math:`-\mathbf{k}`), the bubble
    multiply on the same chunks, the momentum FFT on (orbital pair, frequency) columns, the first exchange (every
    node receives the transformed gap of its window's momenta at every frequency), the vertex contraction on the
    node's stars (:func:`_contract_stars`) or momentum range, the second exchange (every node receives the direct term
    at every momentum at its own frequencies), the crossed term on momentum chunks (:func:`_crossed_term`; the
    momentum and the frequency partners lie on the node), its addition, the inverse FFT on columns and the projection
    of the result. In each exchange the node at distance :math:`d` sends from its node rank :math:`(d - 1)` modulo
    its size to the same node rank of the receiving node. Every element is computed by the same operations as in the
    single-rank sector matvec, so the result is bit-identical to it without an orbital action and wedge windows, and
    equal to rounding with them (the einsums of the orbital action and the star rotations pick their contraction
    paths by the operand shapes).

    The seeds: the node root draws the base seed (:func:`get_initial_gap_function`) and writes its node's frequency
    share, and every rank projects its chunk; when that projection collapses (measured by the global norms as in
    :func:`_sector_seed`) or several starting vectors are asked for, the node roots form the seeds of
    :func:`_sector_seeds` and every rank copies its chunk. The assembly: every rank writes its chunk into the node's
    share, and the node roots send their shares to rank 0.

    :param comm: The MPI communicator; all its ranks form the team (rank 0 is the lead).
    :param node_comm: The node-local communicator.
    :param layout: The layout of the solve.
    :param gamma_mm: This node's window of one channel's vertex (see :func:`_build_distributed_window`).
    :param chi0_share: This node's frequency share of the pp bubble, ``[x, y, z, o, o, o, o, f]`` (node-shared).
    :param arrays: The node-shared arrays of :func:`_distributed_scratch`.
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param n_bands: Number of orbitals ``o``.
    :param u: The orbital action of the inversion for the parity sectors (see :func:`_project_gap_to_sector`).
    :return: ``(apply_block, seed_block, collect, (k0, k1))``: the collective sector matvec on this rank's block (the
        momentum rows ``k0:k1`` of the node's frequency share, flattened), the seeds and the assembly for
        :func:`_solve_team_sectors`, and the block's momentum rows.
    """
    x_share, t_share, y_share, gin, dout = arrays
    rank, node = comm.rank, _node_index(layout, comm.rank)
    size, local = node_comm.Get_size(), node_comm.Get_rank()
    nqx, nqy, nqz = gap_shape[:3]
    nk, nv, oo = nqx * nqy * nqz, gap_shape[-1], n_bands * n_bands
    freqs, points = layout.freqs[node], layout.points[node]
    f = len(freqs)
    chunks = np.linspace(0, nk, size + 1).astype(int)
    k0, k1 = int(chunks[local]), int(chunks[local + 1])
    minus = _at_minus_k(np.arange(nk).reshape(nqx, nqy, nqz)).reshape(-1)[k0:k1]
    columns = np.array_split(np.arange(oo * f), size)[local]
    x, t, y = (a.reshape(nk, n_bands, n_bands, f) for a in (x_share, t_share, y_share))
    y_columns = y_share.reshape(nqx, nqy, nqz, oo, f)
    chi0_rows = np.moveaxis(chi0_share.reshape(nk, oo, oo, f), -1, 1)[k0:k1, None, None]
    gin_flat, dout_flat = gin.reshape(len(points), -1, 1), dout.reshape(len(points), -1, 1)
    u_k = u_m = u
    if u is not None and u.ndim > 2:
        u_flat = u.reshape(nk, n_bands, n_bands)
        u_k, u_m = u_flat[k0:k1], u_flat[minus]
    own = np.linspace(0, len(points), size + 1).astype(int)
    o0, o1 = int(own[local]), int(own[local + 1])
    if layout.stars is not None:
        orbits = _wedge_orbits(config.lattice.k_grid)
        offsets = orbits[1]
        s0, s1 = layout.stars[node]
        cuts = np.searchsorted(offsets[s0 : s1 + 1], np.linspace(offsets[s0], offsets[s1], size + 1))
        r0, r1 = s0 + int(cuts[local]), s0 + int(cuts[local + 1])
        rotations = _member_rotations(orbits, r0, r1)
        batches = _star_batches(offsets, r0, r1, gin_flat.shape[1] * gin_flat.itemsize)
    n_nodes = len(layout.nodes)
    distances = [d for d in range(1, n_nodes) if (d - 1) % size == local]

    def exchange(pack: Callable, shape_from: Callable, unpack: Callable, tag: int) -> None:
        reqs, sends, received = [], [], []
        for d in distances:
            dest, src = (node + d) % n_nodes, (node - d) % n_nodes
            block = pack(dest)
            if block.size:
                sends.append(block)
                reqs += mpi_utils.isend_rows(comm, block, layout.nodes[dest][(d - 1) % len(layout.nodes[dest])], tag)
            buf = np.empty(shape_from(src), dtype=y_share.dtype)
            if buf.size:
                received.append((src, buf))
                reqs += mpi_utils.irecv_rows_into(comm, buf, layout.nodes[src][(d - 1) % len(layout.nodes[src])], tag)
        MPI.Request.Waitall(reqs)
        for src, buf in received:
            unpack(src, buf)

    def gg_shape(src: int) -> tuple:
        return len(points), n_bands, n_bands, len(layout.freqs[src])

    def gg_unpack(src: int, buf: np.ndarray) -> None:
        gin[..., layout.freqs[src]] = buf

    def direct_shape(src: int) -> tuple:
        return len(layout.points[src]), n_bands, n_bands, f

    def direct_unpack(src: int, buf: np.ndarray) -> None:
        y[layout.points[src]] = buf

    def project(source: np.ndarray, eps_t: int, eps_p: int) -> np.ndarray:
        g = source[k0:k1]
        t[k0:k1] = 0.5 * (g + eps_t * _rotate_rows(np.flip(g, axis=-1).swapaxes(1, 2), u_k, u_m))
        node_comm.Barrier()
        return 0.5 * (t[k0:k1] + eps_p * _rotate_rows(t[minus], u_k, u_m))

    def transform(fft: Callable) -> None:
        for column in columns:
            pair, v = divmod(int(column), f)
            y_columns[..., pair, v] = fft(y_columns[..., pair, v], axes=(0, 1, 2))
        node_comm.Barrier()

    def contract() -> None:
        if layout.stars is None:
            np.matmul(gamma_mm[o0:o1], gin_flat[o0:o1], out=dout_flat[o0:o1])
            return
        for a, b in batches:
            a0, a1 = offsets[a] - offsets[s0], offsets[b] - offsets[s0]
            w = None if rotations is None else rotations[offsets[a] - offsets[r0] : offsets[b] - offsets[r0]]
            dout_flat[a0:a1, :, 0] = _contract_stars(gamma_mm[a - s0 : b - s0], gin_flat[a0:a1, :, 0], w, nv)

    def apply_block(channel: SpinChannel, eps_t: int | None, eps_p: int | None, block: np.ndarray) -> np.ndarray:
        x[k0:k1] = block.reshape(k1 - k0, n_bands, n_bands, f)
        node_comm.Barrier()
        dressed = x[k0:k1] if eps_t is None else project(x, eps_t, eps_p)
        y[k0:k1] = _apply_gchi0_pp(chi0_rows, dressed, n_bands).reshape(k1 - k0, n_bands, n_bands, f)
        del dressed
        node_comm.Barrier()
        transform(sp.fft.fftn)
        gin[o0:o1, ..., freqs] = y[points[o0:o1]]
        exchange(lambda m: y[layout.points[m]], gg_shape, gg_unpack, EXCHANGE_TAG)
        node_comm.Barrier()
        contract()
        node_comm.Barrier()
        y[points[o0:o1]] = dout[o0:o1, ..., freqs]
        exchange(
            lambda m: np.ascontiguousarray(dout[..., layout.freqs[m]]), direct_shape, direct_unpack, EXCHANGE_TAG + 50
        )
        node_comm.Barrier()
        # the crossed term reads the direct term at -k, so it is staged before any row is overwritten
        t[k0:k1] = np.flip(y[minus].swapaxes(1, 2), axis=-1)
        if channel != SpinChannel.SING:
            t[k0:k1] *= -1
        node_comm.Barrier()
        y[k0:k1] += t[k0:k1]
        node_comm.Barrier()
        transform(sp.fft.ifftn)
        out = y[k0:k1].copy() if eps_t is None else project(y, eps_t, eps_p)
        node_comm.Barrier()
        return out.reshape(-1)

    def seed_block(channel: SpinChannel, eps_t: int | None, eps_p: int | None, n_block: int) -> np.ndarray:
        base = get_initial_gap_function(gap_shape, channel) if local == 0 else None
        if local == 0:
            x[...] = base.reshape(nk, n_bands, n_bands, nv)[..., freqs]
        node_comm.Barrier()
        seed, collapsed = x[k0:k1].copy(), False
        if eps_t is not None:
            seed = project(x, eps_t, eps_p)
            norms = comm.allreduce(np.array([np.vdot(seed, seed).real, np.vdot(x[k0:k1], x[k0:k1]).real]))
            collapsed = bool(np.sqrt(norms[0]) < 1e-10 * max(np.sqrt(norms[1]), 1e-30))
        node_comm.Barrier()
        if n_block == 1 and not collapsed:
            return seed[None]
        seeds = _sector_seeds(base.reshape(-1), gap_shape, eps_t, eps_p, n_block, u) if local == 0 else None
        blocks = np.empty((n_block,) + seed.shape, dtype=seed.dtype)
        for b in range(n_block):
            if local == 0:
                x[...] = seeds[b].reshape(nk, n_bands, n_bands, nv)[..., freqs]
            node_comm.Barrier()
            blocks[b] = x[k0:k1]
            node_comm.Barrier()
        return blocks

    def collect(block: np.ndarray, out: np.ndarray | None) -> None:
        x[k0:k1] = block.reshape(k1 - k0, n_bands, n_bands, f)
        node_comm.Barrier()
        if local == 0 and rank != 0 and f:
            mpi_utils.send_rows(comm, x_share, 0, base_tag=GAP_TAG)
        if rank == 0:
            full = out.reshape(nk, n_bands, n_bands, nv)
            for m, ranks in enumerate(layout.nodes):
                shape = (nk, n_bands, n_bands, len(layout.freqs[m]))
                if m == node:
                    full[..., freqs] = x
                elif shape[-1]:
                    full[..., layout.freqs[m]] = mpi_utils.recv_rows_alloc(comm, shape, x.dtype, ranks[0], GAP_TAG)
        node_comm.Barrier()

    return apply_block, seed_block, collect, (k0, k1)


def _solve_sectors_distributed(
    mpi_dist_irrk: MpiDistributor,
    comm: MPI.Comm,
    node_comm: MPI.Comm,
    gamma_sing_pp: FourPoint,
    gamma_trip_pp: FourPoint,
    giwk_dga: GreensFunction | None,
    niv_pp: int,
    layout: DistributedLayout,
    u: np.ndarray | None = None,
) -> dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Solves every ``(channel, parity)`` sector with one team of all ranks spread over the nodes (see
    :class:`DistributedLayout`), for jobs where no node holds one channel's vertex windows. The pp bubble is built on
    rank 0, and every node root receives its node's frequency share of it (node-shared). The channels run one after
    the other: the channel's window is built across the nodes (:func:`_build_distributed_window`), its sectors run
    one after the other through :func:`_solve_team_sectors` with the matvec, seeds and assembly of
    :func:`_node_team_matvec`, and the window is freed before the next channel's build. Both pairing vertices are
    **consumed**. The eigenvalues of every sector reach every rank, the gap functions rank 0 only.

    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param comm: The MPI communicator.
    :param node_comm: The node-local communicator.
    :param gamma_sing_pp: The singlet pairing vertex (irr-BZ q-distributed on entry; consumed).
    :param gamma_trip_pp: The triplet pairing vertex (irr-BZ q-distributed on entry; consumed).
    :param giwk_dga: The DGA Green's function (held on rank 0; used to build the bubble).
    :param niv_pp: Number of positive fermionic frequencies of the pp box.
    :param layout: The layout of :func:`plan_distributed_solve`.
    :param u: The orbital action of the inversion for the parity sectors (see :func:`_project_gap_to_sector`).
    :return: ``{(channel, parity): (lambdas, gaps)}`` for every sector; ``lambdas`` on every rank, ``gaps`` on rank 0
        (an empty list elsewhere).
    """
    logger = config.logger
    rank, k_grid = comm.rank, config.lattice.k_grid
    node = _node_index(layout, rank)
    n_bands = gamma_sing_pp.n_bands
    gap_shape = k_grid.nk + (n_bands, n_bands, 2 * niv_pp)
    norm = 0.5 / k_grid.nk_tot / config.sys.beta

    # the pp bubble: built on rank 0, each node root receives the frequency share of its node
    gchi0_q_pp = None
    if rank == 0:
        gchi0_q_pp = BubbleGenerator.create_generalized_chi0_q_pp_w0(giwk_dga, niv_pp, k_grid).decompress_q_dimension()
        giwk_dga.free()
        logger.info("Created the bare bubble susceptibility in pp notation.")
    dtype = comm.bcast(gchi0_q_pp.mat.dtype if rank == 0 else None, root=0)
    share = None
    if node_comm.Get_rank() == 0:
        share = np.empty(k_grid.nk + (n_bands,) * 4 + (len(layout.freqs[node]),), dtype=dtype)
    for ranks, freqs in zip(layout.nodes, layout.freqs):
        if not len(freqs):
            continue
        if rank == ranks[0] == 0:
            share = np.ascontiguousarray(gchi0_q_pp.mat[..., freqs])
        elif rank == 0:
            mpi_utils.send_rows(comm, np.ascontiguousarray(gchi0_q_pp.mat[..., freqs]), ranks[0], base_tag=BUBBLE_TAG)
        elif rank == ranks[0]:
            mpi_utils.recv_rows_into(comm, share, 0, base_tag=BUBBLE_TAG)
    if gchi0_q_pp is not None:
        gchi0_q_pp.free()
    chi0_share, chi0_win = mpi_utils.build_node_shared_array(node_comm, lambda: share)
    del share

    arrays, scratch_wins = _distributed_scratch(node_comm, layout, rank, gap_shape, gamma_sing_pp.mat.dtype)
    parities = [label for label, _ in _parity_sectors(config.eliashberg.resolve_parity_sectors)]
    host = f"{len(layout.nodes)} node(s)"
    results: dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]] = {}
    for channel, gamma_pp in ((SpinChannel.SING, gamma_sing_pp), (SpinChannel.TRIP, gamma_trip_pp)):
        gamma_mm, win = _build_distributed_window(gamma_pp, mpi_dist_irrk, node_comm, layout, norm)
        logger.info(
            f"Gamma_pp_{channel.value} window ({'irreducible wedge' if layout.stars else 'full grid'}) split over "
            f"{host}: {gamma_mm.nbytes / 1024**3:.3f} GB on rank 0's node, built by {layout.builders} rank(s) per "
            f"node.",
            allowed_ranks=(0,),
        )
        apply_block, seed_block, collect, _ = _node_team_matvec(
            comm, node_comm, layout, gamma_mm, chi0_share, arrays, gap_shape, n_bands, u
        )
        team = LanczosTeam(host, 0, tuple(range(comm.size)), tuple((channel, parity) for parity in parities))
        # ponytail: one BLAS thread per rank, as in the team solve
        with threadpool_limits(limits=1):
            local = _solve_team_sectors(comm, team, apply_block, seed_block, collect, gap_shape)
        # every view into a window dies before the window is freed
        del apply_block, seed_block, collect, gamma_mm
        mpi_utils.free_shared_window(win, node_comm)
        for sector in team.sectors:
            mine = local.get(sector)
            results[sector] = (comm.bcast(mine[0] if mine is not None else None, root=0), mine[1] if mine else [])
    del arrays, chi0_share
    for win in scratch_wins + [chi0_win]:
        mpi_utils.free_shared_window(win, node_comm)
    return results


def _solve_sectors_in_teams(
    mpi_dist_irrk: MpiDistributor,
    comm: MPI.Comm,
    node_comm: MPI.Comm,
    gamma_sing_pp: FourPoint,
    gamma_trip_pp: FourPoint,
    giwk_dga: GreensFunction | None,
    niv_pp: int,
    rounds: tuple[tuple[LanczosTeam, ...], ...],
    u: np.ndarray | None = None,
) -> dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Solves every ``(channel, parity)`` sector with the teams of :func:`plan_lanczos_teams`. The pp bubble is built
    once on rank 0, shipped to the other hosting node roots and exposed through one shared window per node; per
    round, each channel's pairing vertex is gathered onto the root of every node hosting one of its sectors and
    written into a node-shared matmul-layout window there (see :func:`_build_shared_vertex_window`), so a node holds
    one copy per hosted channel instead of one per sector; every team then runs the lockstep matvec of
    :func:`_team_matvec` over that window while its lead drives the eigensolver. On a symmetry-reduced grid with
    unitary operations the windows hold the irreducible wedge of the real-space grid and the matvec contracts its
    stars (see :func:`_wedge_orbits`, :func:`_contract_stars`). Both pairing vertices are **consumed**. The
    eigenvalues of every sector reach every rank; the gap functions travel from the lead to rank 0 only (the rank
    that writes them), one blocking transfer per gap, their number following the eigenvalues actually returned.

    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param comm: The MPI communicator.
    :param node_comm: The node-local communicator.
    :param gamma_sing_pp: The singlet pairing vertex (irr-BZ q-distributed on entry; consumed).
    :param gamma_trip_pp: The triplet pairing vertex (irr-BZ q-distributed on entry; consumed).
    :param giwk_dga: The DGA Green's function (held on rank 0; used to build the bubble).
    :param niv_pp: Number of positive fermionic frequencies of the pp box.
    :param rounds: The team plan.
    :return: ``{(channel, parity): (lambdas, gaps)}`` for every sector; ``lambdas`` on every rank, ``gaps`` on rank 0
        (an empty list elsewhere).
    """
    logger = config.logger
    my_rank = comm.rank
    k_grid = config.lattice.k_grid
    norm = 0.5 / k_grid.nk_tot / config.sys.beta
    n_bands = gamma_sing_pp.n_bands
    gap_shape = k_grid.nk + (n_bands, n_bands, 2 * gamma_sing_pp.niv)
    my_node_root = node_comm.bcast(my_rank, root=0)
    hosting_roots = sorted({team.node_root for teams in rounds for team in teams})
    orbits = _wedge_orbits(k_grid) if wedge_window_points(k_grid) < k_grid.nk_tot else None
    wedge = orbits is not None

    # the pp bubble: built on rank 0, shipped to the other hosting node roots, one shared window per hosting node
    gchi0_q_pp = None
    if my_rank == 0:
        gchi0_q_pp = BubbleGenerator.create_generalized_chi0_q_pp_w0(giwk_dga, niv_pp, k_grid).decompress_q_dimension()
        giwk_dga.free()
        logger.info("Created the bare bubble susceptibility in pp notation.")
    for root in hosting_roots:
        if root == 0:
            continue
        if my_rank == 0:
            mpi_dist_irrk.send_to_rank(gchi0_q_pp, dest=root, base_tag=BUBBLE_TAG)
        elif my_rank == root:
            gchi0_q_pp = mpi_dist_irrk.recv_from_rank(source=0, base_tag=BUBBLE_TAG)
    chi0_mm, chi0_win = None, None
    if my_node_root in hosting_roots:
        chi0_shared, chi0_win = mpi_utils.build_node_shared_array(node_comm, lambda: gchi0_q_pp.mat)
        chi0_mm = _chi0_to_matmul_layout(chi0_shared)
        del chi0_shared
    if gchi0_q_pp is not None:
        gchi0_q_pp.free()

    results: dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]] = {}
    for teams in rounds:
        gamma_by_channel: dict[SpinChannel, np.ndarray] = {}
        vertex_wins = []
        for channel, gamma_pp in ((SpinChannel.SING, gamma_sing_pp), (SpinChannel.TRIP, gamma_trip_pp)):
            roots = sorted({team.node_root for team in teams if any(c == channel for c, _ in team.sectors)})
            if not roots:
                continue
            local = gamma_pp.mat
            for root in roots:
                gathered = mpi_dist_irrk.gather(local, root=root)
                if my_rank == root:
                    gamma_pp.mat = gathered
            # the gathers are complete; these names would otherwise keep the rank's share alive
            del gathered, local
            if my_rank not in roots:
                gamma_pp.free()
            if my_node_root in roots:
                gamma_by_channel[channel], win = _build_shared_vertex_window(
                    gamma_pp, node_comm, norm, None if orbits is None else orbits[0]
                )
                vertex_wins.append(win)
                logger.info(
                    f"Gamma_pp_{channel.value} window ({'irreducible wedge' if wedge else 'full grid'}): "
                    f"{gamma_by_channel[channel].nbytes / 1024**3:.3f} GB, shared by the {node_comm.Get_size()} "
                    f"rank(s) of its node.",
                    allowed_ranks=tuple(roots),
                )

        my_team = next((team for team in teams if my_rank in team.ranks), None)
        team_comm = comm.Split(teams.index(my_team) if my_team is not None else len(teams), my_rank)
        local: dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]] = {}
        if my_team is not None:
            dtype = next(iter(gamma_by_channel.values())).dtype
            scratch = [mpi_utils.allocate_node_shared_array(team_comm, gap_shape, dtype) for _ in range(4)]
            apply_block, (v0, v1) = _team_matvec(
                team_comm,
                gamma_by_channel,
                chi0_mm,
                [array for array, _ in scratch],
                gap_shape,
                n_bands,
                orbits,
                u,
            )
            # ponytail: one BLAS thread per team rank (production binds one rank per core); a per-rank executor
            # if few-rank jobs on wide nodes ever matter
            with threadpool_limits(limits=1):
                local = _solve_team_sectors(
                    team_comm,
                    my_team,
                    apply_block,
                    partial(_team_seed_block, team_comm, gap_shape, (v0, v1), u),
                    partial(_team_collect, team_comm, scratch[0][0], (v0, v1)),
                    gap_shape,
                )
            # every view into a window dies before the window is freed (the matvec closure holds the scratch
            # arrays and the vertex dict)
            scratch_wins = [win for _, win in scratch]
            del apply_block, scratch
            for win in scratch_wins:
                mpi_utils.free_shared_window(win, team_comm)
        team_comm.Free()

        for team in teams:
            for sector in team.sectors:
                lead, mine, tag = team.ranks[0], local.get(sector), SECTOR_TAG * (1 + len(results))
                lambdas = mpi_dist_irrk.bcast(mine[0] if mine is not None else None, root=lead)
                gaps = mine[1] if mine is not None else []
                if my_rank == lead != 0:
                    for gap in gaps:
                        mpi_dist_irrk.send_to_rank(gap, dest=0, base_tag=tag)
                    gaps = []
                elif my_rank == 0 != lead:
                    gaps = [mpi_dist_irrk.recv_from_rank(source=lead, base_tag=tag) for _ in range(len(lambdas))]
                results[sector] = (lambdas, gaps)
        gamma_by_channel.clear()
        for win in vertex_wins:
            mpi_utils.free_shared_window(win, node_comm)
    del chi0_mm
    mpi_utils.free_shared_window(chi0_win, node_comm)
    return results


def _team_seed_block(
    team_comm: MPI.Comm,
    gap_shape: tuple,
    bounds: tuple[int, int],
    u: np.ndarray | None,
    channel: SpinChannel,
    eps_t: int | None,
    eps_p: int | None,
    n_block: int,
) -> np.ndarray:
    r"""
    Returns this rank's frequency block ``[..., v0:v1]`` of one sector's ``n_block`` starting vectors (see
    :func:`_sector_seeds`): the team's first rank alone draws and projects the gap-sized seeds, and every rank copies
    its block from a team-shared window. Collective over the team.

    :param team_comm: The team communicator (rank 0 is the lead).
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param bounds: This rank's frequency block ``(v0, v1)``.
    :param u: The orbital action of the inversion for the parity sectors (see :func:`_project_gap_to_sector`).
    :param channel: The pairing channel of the seed.
    :param eps_t: The parity of the sector under :math:`T O_U`, or ``None`` for the raw kernel.
    :param eps_p: The forced inversion parity of the sector.
    :param n_block: Number of starting vectors.
    :return: The block of the seeds, shape ``[n_block, kx, ky, kz, o1, o2, v1 - v0]``.
    """
    v0, v1 = bounds
    seeds, win = mpi_utils.build_node_shared_array(
        team_comm,
        lambda: _sector_seeds(
            get_initial_gap_function(gap_shape, channel).flatten(), gap_shape, eps_t, eps_p, n_block, u
        ),
    )
    seed = np.ascontiguousarray(seeds.reshape((n_block,) + tuple(gap_shape))[..., v0:v1])
    del seeds
    mpi_utils.free_shared_window(win, team_comm)
    return seed


def _team_collect(
    team_comm: MPI.Comm, gap_w: np.ndarray, bounds: tuple[int, int], block: np.ndarray, out: np.ndarray | None
) -> None:
    r"""
    Assembles one vector from the frequency blocks of a team's ranks into ``out`` on the team's first rank, through
    the team's gap-shaped window. Collective over the team.

    :param team_comm: The team communicator (rank 0 is the lead).
    :param gap_w: The team's gap-shaped shared window.
    :param bounds: This rank's frequency block ``(v0, v1)``.
    :param block: This rank's block of the vector.
    :param out: The flattened destination on the lead, ``None`` elsewhere.
    :return: None.
    """
    v0, v1 = bounds
    gap_w[..., v0:v1] = block.reshape(gap_w.shape[:-1] + (v1 - v0,))
    team_comm.Barrier()
    if out is not None:
        out[...] = gap_w.reshape(-1)
    team_comm.Barrier()


def _solve_team_sectors(
    team_comm: MPI.Comm,
    team: LanczosTeam,
    apply_block: Callable,
    seed_block: Callable,
    collect: Callable,
    gap_shape: tuple,
) -> dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Solves the sectors of one team with the team-distributed Krylov-Schur iteration (:func:`_team_arnoldi`): every
    rank of the team runs the same iteration on its block of the vectors, with the sector seeds and projectors of
    the single-rank solve, and the converged eigenvectors are assembled onto the lead, which orders, symmetrizes and
    wraps them (see :func:`_finish_sector`). The lead alone holds the results.

    :param team_comm: The team communicator (rank 0 is the lead).
    :param team: The team and its sectors.
    :param apply_block: The collective sector matvec on this rank's block,
        ``apply_block(channel, eps_t, eps_p, block) -> block``.
    :param seed_block: Returns this rank's block of a sector's starting vectors along a leading axis,
        ``seed_block(channel, eps_t, eps_p, n_block)`` (collective).
    :param collect: Assembles a vector from the ranks' blocks into a flattened destination on the lead,
        ``collect(block, out)`` with ``out`` ``None`` on the other ranks (collective).
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :return: ``{(channel, parity): (lambdas, gaps)}`` on the lead, an empty dict elsewhere.
    """
    logger = config.logger
    lead = team_comm.Get_rank() == 0
    n_eig, tol = config.eliashberg.n_eig, config.eliashberg.epsilon
    ncv = min(lanczos_ncv(n_eig), int(np.prod(gap_shape)))
    orbital_mirrors = _gap_orbital_mirrors(gap_shape[3]) if config.eliashberg.symmetrize_degenerate_gaps else {}
    eps_by_parity = dict(_parity_sectors(config.eliashberg.resolve_parity_sectors))
    ranks = (team.ranks[0],)
    labels = [_sector_log_label(channel, [parity]) for channel, parity in team.sectors]
    logger.info(f"Lanczos team for {', '.join(labels)}: {len(team.ranks)} rank(s) on {team.host}.", allowed_ranks=ranks)
    results: dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]] = {}
    for (channel, parity), label in zip(team.sectors, labels):
        sign = 1 if channel == SpinChannel.SING else -1
        eps_t = eps_by_parity[parity]
        eps_p = None if eps_t is None else sign * eps_t
        logger.info(f"Starting Lanczos method for {label}.", allowed_ranks=ranks)
        n_block = block_size(config.lattice.k_grid, eps_p if gap_shape[3] == 1 else None)
        seed = seed_block(channel, eps_t, eps_p, n_block)
        attempts = []

        def attempt(t: float) -> tuple:
            attempts.append(
                _team_arnoldi(
                    lambda block: apply_block(channel, eps_t, eps_p, block),
                    seed,
                    n_eig,
                    ncv,
                    t,
                    LANCZOS_MATVECS,
                    team_comm,
                    n_block,
                )
            )
            return attempts[-1]

        lambdas, ritz, basis, _ = _solve_loosening(attempt, tol, n_eig, label, ranks)
        # the log counts every attempt, as the single-rank solve does
        logger.info(f"Lanczos for {label}: {sum(a[3] for a in attempts)} matvecs.", allowed_ranks=ranks)
        if len(lambdas) < n_eig:
            logger.warning(
                f"Lanczos for {label} did not converge within the matvec limit; keeping the {len(lambdas)} of "
                f"{n_eig} converged eigenpair(s).",
                allowed_ranks=ranks,
            )
        if eps_t is not None and len(lambdas):
            leading = ritz[:, int(np.argmax(lambdas))] @ basis
            raw = apply_block(channel, None, None, leading).reshape(-1)
            _log_sector_residual(leading, raw, label, ranks, team_comm.allreduce)
        # the eigenvectors travel to the lead one at a time
        gaps = np.empty((int(np.prod(gap_shape)), len(lambdas)), dtype=basis.dtype) if lead else None
        for i in range(len(lambdas)):
            block = ritz[:, i] @ basis
            block /= np.sqrt(team_comm.allreduce(float(np.vdot(block, block).real)))
            collect(block, gaps[:, i] if lead else None)
        if lead:
            results[(channel, parity)] = _finish_sector(
                lambdas, gaps, gap_shape, channel, tuple(gap_shape[:3]), label, orbital_mirrors, ranks
            )
    logger.info(f"Finished solving the Eliashberg equation for {', '.join(labels)}.", allowed_ranks=ranks)
    return results


def _solve_pairing_sectors(
    mv,
    gap_shape: tuple,
    sign: int,
    channel: SpinChannel,
    nq: tuple,
    executor,
    ranks,
    base_seed: np.ndarray,
    parities: list[str] | None = None,
    dtype: np.dtype = DTYPE,
    u: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Runs the ARPACK/Lanczos solve of the pairing kernel ``mv`` once per parity sector selected by
    ``config.eliashberg.resolve_parity_sectors`` and returns the leading ``n_eig`` eigenpairs of each. Every projected
    sector hands the matvec its Hermitian sector projector :math:`\Pi` (parity ``eps_T`` under :math:`T O_U` and the
    Pauli-forced inversion parity ``eps_P = sign * eps_T``, see :func:`_project_gap_to_sector`) and projects the seed
    with it, and the residual of its leading gap under the raw kernel is logged (see
    :func:`_log_sector_residual`); the ``"none"`` sector runs the raw kernel unchanged. A sector on which ARPACK stops
    with an error (a kernel that annihilates the seed, e.g. a sector it vanishes on) is solved by :func:`_team_arnoldi`
    instead. The passed ``executor`` (the momentum-batch thread pool, or ``None``) is shut down before returning.

    :param mv: The sector matvec ``mv(gap, eps_t, eps_p)`` (maps a full-length gap vector to a full-length gap
        vector, projected onto the sector unless ``eps_t`` is ``None``).
    :param gap_shape: The ``[kx, ky, kz, o1, o2, v]`` shape of the gap.
    :param sign: The channel sign (:math:`+1` singlet, :math:`-1` triplet).
    :param channel: The pairing channel (used to label outputs).
    :param nq: The momentum-grid shape carried onto each :class:`GapFunction`.
    :param executor: The momentum-batch thread pool (or ``None``); shut down on return.
    :param ranks: The ranks tuple used for logging.
    :param base_seed: The flattened initial gap seed, identical on every rank (drawn from a fixed-seed
        generator); projected into each sector by :func:`_sector_seed`.
    :param parities: An optional subset of parity labels to solve; ``None`` solves every configured sector.
    :param dtype: The dtype ``mv`` returns (that of the pairing vertex); declared on the eigensolver's operator so
        scipy does not probe it with an extra matvec.
    :param u: The orbital action of the inversion for the parity sectors (see :func:`_project_gap_to_sector`).
    :return: ``{parity_label: (lambdas, [GapFunction, ...])}`` for each solved sector.
    """
    logger = config.logger
    n_eig = config.eliashberg.n_eig
    shape_flat = int(np.prod(gap_shape))

    sectors = _parity_sectors(config.eliashberg.resolve_parity_sectors)
    if parities is not None:
        sectors = [(label, eps_t) for label, eps_t in sectors if label in parities]

    # the mirrors are a property of the orbital basis and the lattice, not of the sector: determine them once
    orbital_mirrors = _gap_orbital_mirrors(gap_shape[3]) if config.eliashberg.symmetrize_degenerate_gaps else {}

    results: dict[str, tuple[np.ndarray, list[GapFunction]]] = {}
    try:
        for parity, eps_t in sectors:
            eps_p = None if eps_t is None else sign * eps_t
            label = _sector_log_label(channel, [parity])
            logger.info(f"Starting Lanczos method for {label}.", allowed_ranks=ranks)
            n_matvec = 0

            def counted_matvec(gap: np.ndarray, eps_t=eps_t, eps_p=eps_p) -> np.ndarray:
                nonlocal n_matvec
                n_matvec += 1
                return mv(gap, eps_t, eps_p)

            mat = sp.sparse.linalg.LinearOperator(shape=(shape_flat, shape_flat), matvec=counted_matvec, dtype=dtype)
            n_block = block_size(config.lattice.k_grid, eps_p if gap_shape[3] == 1 else None)
            # BLAS is pinned to one thread for the solve (threadpool_limits resizes the live pool; an environment
            # change would be ignored) so the momentum-batch threads never nest BLAS threads underneath.
            with threadpool_limits(limits=1 if executor is not None else None):
                if n_block > 1:
                    # forced multiplets need the block iteration, run on the full vectors of this rank
                    lambdas, ritz, basis, _ = _solve_loosening(
                        lambda t: _team_arnoldi(
                            counted_matvec,
                            _sector_seeds(base_seed, gap_shape, eps_t, eps_p, n_block, u),
                            n_eig,
                            min(lanczos_ncv(n_eig), shape_flat),
                            t,
                            LANCZOS_MATVECS,
                            None,
                            n_block,
                        ),
                        config.eliashberg.epsilon,
                        n_eig,
                        label,
                        ranks,
                    )
                    gaps = (ritz.T @ basis).T
                    del basis
                    if len(lambdas) < n_eig:
                        logger.warning(
                            f"Lanczos for {label} did not converge within the matvec limit; keeping the "
                            f"{len(lambdas)} of {n_eig} converged eigenpair(s).",
                            allowed_ranks=ranks,
                        )
                else:

                    def solve(tol: float) -> tuple[np.ndarray, np.ndarray]:
                        seed = _sector_seed(base_seed, gap_shape, eps_t, eps_p, u)
                        ncv = min(lanczos_ncv(n_eig), shape_flat)
                        try:
                            # ARPACK counts restarts, each of at most ncv - n_eig products after the first basis
                            return sp.sparse.linalg.eigsh(
                                mat,
                                k=n_eig,
                                tol=tol,
                                v0=seed,
                                ncv=ncv,
                                which="LA",
                                maxiter=(LANCZOS_MATVECS - ncv) // (ncv - n_eig) + 1,
                            )
                        except sp.sparse.linalg.ArpackNoConvergence as exc:
                            # the converged subset is kept; the delivery sizes the gap transfer off what came back
                            return exc.eigenvalues.real, exc.eigenvectors
                        except sp.sparse.linalg.ArpackError as exc:
                            # ARPACK stops when the kernel annihilates the seed (e.g. a sector it vanishes on); the
                            # team iteration continues from a fresh direction instead
                            logger.info(
                                f"{label}: {exc}; continuing with the team Arnoldi iteration.", allowed_ranks=ranks
                            )
                            lambdas, ritz, basis, _ = _team_arnoldi(
                                counted_matvec, seed, n_eig, ncv, tol, LANCZOS_MATVECS, None
                            )
                            return lambdas, (ritz.T @ basis).T

                    lambdas, gaps = _solve_loosening(solve, config.eliashberg.epsilon, n_eig, label, ranks)
                    if len(lambdas) < n_eig:
                        logger.warning(
                            f"Lanczos for {label} did not converge within the matvec limit; keeping the "
                            f"{len(lambdas)} of {n_eig} converged eigenpair(s).",
                            allowed_ranks=ranks,
                        )
            logger.info(f"Lanczos for {label}: {n_matvec} matvecs.", allowed_ranks=ranks)
            if eps_t is not None and len(lambdas):
                leading = gaps[:, int(np.argmax(lambdas))]
                # the check's matvec runs on the momentum-batch threads as well, so BLAS stays pinned
                with threadpool_limits(limits=1 if executor is not None else None):
                    _log_sector_residual(leading, mv(leading, None, None), label, ranks)
            results[parity] = _finish_sector(lambdas, gaps, gap_shape, channel, nq, label, orbital_mirrors, ranks)
    finally:
        if executor is not None:
            executor.shutdown()

    logger.info(f"Finished solving the Eliashberg equation for the {channel.value}let channel.", allowed_ranks=ranks[0])
    return results


def solve_eliashberg_lanczos(
    gamma_r_pp: FourPoint,
    gchi0_q0_pp: FourPoint,
    ranks: tuple[int, int],
    parities: list[str] | None = None,
    u: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Solves the linearized Eliashberg equation for the leading superconducting eigenvalue(s) and gap function(s) using
    an ARPACK/Lanczos eigensolver, with the pairing kernel applied matrix-free via FFTs over the BZ. This in-memory
    variant holds the full-BZ pairing vertex on the solving rank. The passed pairing vertex is **consumed** (mapped
    to the full BZ and Fourier transformed in place, then freed once its matmul-layout copy is built).

    When ``config.eliashberg.resolve_parity_sectors`` is set, the matvec and the starting vector are, for each
    parity sector, sandwiched in the sector projector :math:`\Pi` (see
    :func:`_project_gap_to_sector`), so the eigensolver returns the leading eigenpairs of :math:`\Pi M \Pi` restricted
    to that sector; otherwise the raw kernel :math:`M` is run unchanged.

    :param gamma_r_pp: The pairing vertex :math:`\Gamma^{\mathrm{pp}}_{r}` (irreducible BZ, pp notation) for one
        channel; consumed by the solve.
    :param gchi0_q0_pp: The bare pp bubble :math:`\chi_0^{\mathrm{pp}}` at :math:`\omega = 0`.
    :param ranks: The ranks used for logging.
    :param parities: An optional subset of parity labels to solve (``None`` solves every configured sector).
    :param u: The orbital action of the inversion for the parity sectors (see :func:`_project_gap_to_sector`).
    :return: A dict ``{parity_label: (lambdas, gaps)}`` of the leading eigenvalues and :class:`GapFunction` objects
        per solved parity sector (a single ``"none"`` key when no projection is requested).
    """
    logger = config.logger
    import psutil

    logger.info(
        f"Starting to solve the Eliashberg equation for the {gamma_r_pp.channel.value}let channel.",
        allowed_ranks=ranks[0],
    )

    gamma_r_pp = gamma_r_pp.map_to_full_bz(config.lattice.k_grid, config.lattice.k_grid.nk).decompress_q_dimension()
    logger.log_memory_usage(f"Gamma_pp_{gamma_r_pp.channel.value}", gamma_r_pp, 1, allowed_ranks=ranks[0])

    gamma_r_pp = gamma_r_pp.fft(False)

    gap_shape = gamma_r_pp.nq + 2 * (gamma_r_pp.n_bands,) + (2 * gamma_r_pp.niv,)
    gchi0_q0_pp = gchi0_q0_pp.decompress_q_dimension()

    symmetry_label = config.eliashberg.symmetry.lower() if config.eliashberg.symmetry else "random"
    logger.info(
        f"Initialized the gap function as {symmetry_label} for {_sector_log_label(gamma_r_pp.channel, parities)}.",
        allowed_ranks=ranks,
    )

    n_bands = gamma_r_pp.n_bands
    norm = 0.5 / config.lattice.k_grid.nk_tot / config.sys.beta
    # the single rank spreads the bandwidth-bound matvec over every core its affinity mask allows (momentum-batch
    # threads, BLAS kept at 1)
    n_threads = _solver_thread_budget()
    executor = ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None
    logger.info(
        f"Solver thread budget for {_sector_log_label(gamma_r_pp.channel, parities)}: {n_threads} thread(s), from "
        f"the CPU affinity mask.",
        allowed_ranks=ranks,
    )

    chi0_mm = _chi0_to_matmul_layout(gchi0_q0_pp.mat)
    # The pairing vertex arrives in w2dynamics G2 leg order (c cdag c cdag), whereas _apply_gamma_pp expects the
    # TRIQS order (cdag c cdag c), see https://triqs.github.io/tprf/latest/theory/eliashberg.html
    # the kernel prefactor is folded into the persistent vertex once: both matvec terms inherit it by linearity, so
    # the per-matvec full-gap multiply is dropped.
    gamma_mm = _gamma_to_matmul_layout(gamma_r_pp.permute_orbitals("abcd->badc", False).mat, scale=norm)
    gamma_r_pp.free()
    resident_gb = psutil.Process().memory_info().rss / 1024**3
    logger.info(
        f"Solver rank resident set after the matmul-layout build: {resident_gb:.3f} GB "
        f"({_sector_log_label(gamma_r_pp.channel, parities)}).",
        allowed_ranks=ranks,
    )

    sign = 1 if gamma_r_pp.channel == SpinChannel.SING else -1

    def mv(gap: np.ndarray, eps_t: int | None, eps_p: int | None) -> np.ndarray:
        r"""
        Applies the pairing kernel of one sector to a flattened gap vector (the matrix-vector product for the
        eigensolver): projects onto the sector (unless ``eps_t`` is ``None``), multiplies by
        :math:`\chi_0^{\mathrm{pp}}`, FFTs to real space, contracts with the pairing vertex (the direct term, plus the
        crossed term formed from it by the exchange of the outgoing pair, see :func:`_crossed_term`), transforms back
        and projects again. The orbital contractions are batched ``np.matmul`` products and the BZ transforms run in
        place through ``scipy.fft`` (both threaded up to the solver thread budget).

        :param gap: The flattened gap vector.
        :param eps_t: The parity of the sector under :math:`T O_U`, or ``None`` for the raw kernel.
        :param eps_p: The forced inversion parity of the sector.
        :return: The flattened result of applying the sector kernel to ``gap``.
        """
        if eps_t is not None:
            gap = _project_gap_to_sector(gap, gap_shape, eps_t, eps_p, u)
        gap_gg = sp.fft.fftn(
            _apply_gchi0_pp(chi0_mm, gap, n_bands, executor, n_threads),
            axes=(0, 1, 2),
            overwrite_x=True,
            workers=n_threads,
        )
        gap_new = _apply_gamma_pp(gamma_mm, gap_gg, n_bands, executor, n_threads)
        gap_new += _crossed_term(gap_new, sign)
        gap_new = sp.fft.ifftn(gap_new, axes=(0, 1, 2), overwrite_x=True, workers=n_threads)
        out = gap_new.flatten()
        return out if eps_t is None else _project_gap_to_sector(out, gap_shape, eps_t, eps_p, u)

    base_seed = get_initial_gap_function(gap_shape, gamma_r_pp.channel).flatten()
    return _solve_pairing_sectors(
        mv,
        gap_shape,
        sign,
        gamma_r_pp.channel,
        gamma_r_pp.nq,
        executor,
        ranks,
        base_seed,
        parities,
        gamma_mm.dtype,
        u,
    )


# --- Eliashberg eigensolver (Lanczos / ARPACK) ---
def inverse_core_bubble(mpi_dist_irrk: MpiDistributor, node_comm: MPI.Comm | None) -> FourPoint:
    r"""
    Rebuilds the inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` of this rank's irreducible momenta on the
    core box from the Green's function the kernel step saved with its Eliashberg intermediates
    (:data:`~dgamore.output_files.GIWK_FILE_STEM`), with the bubble generator and orbital involution of that step, so
    the three-leg vertices and the inverse bubble share one Green's function. The Green's function is loaded once per
    node into a shared window (straight from the file into the window) and released before the inversion.

    :param mpi_dist_irrk: MPI distributor over the irreducible BZ q-points (see :class:`MpiDistributor`).
    :param node_comm: The node-local communicator, or ``None`` on a single rank.
    :return: The inverse bare bubble on the core box as a :class:`FourPoint`.
    """
    path = output_files.npy_path(config.output.eliashberg_path, output_files.GIWK_FILE_STEM)
    if node_comm is None:
        giwk_mat, win = np.load(path), None
    else:
        giwk_mat, win = mpi_utils.build_node_shared_array(node_comm, lambda: np.load(path, mmap_mode="r"))
    ek, orbitals = config.lattice.hamiltonian.get_ek(), config.sys.correlated_orbitals
    giwk = GreensFunction(
        giwk_mat, None, ek, True, False, nk=config.lattice.k_grid.nk, beta=config.sys.beta, orbitals=orbitals
    )
    chi0 = BubbleGenerator.create_generalized_chi0_q_fft(
        mpi_dist_irrk,
        giwk,
        config.box.niw_core,
        config.box.niv_core,
        config.lattice.k_grid,
        config.sys.beta,
        node_comm=node_comm,
        involution=sde_kernels.orbital_involution(ek, orbitals),
    )
    giwk.mat = None
    mpi_utils.free_shared_window(win, node_comm)
    return chi0.invert(copy=False)


def _solve_sectors_in_memory(
    gamma_sing_pp: FourPoint,
    gamma_trip_pp: FourPoint,
    giwk_dga: GreensFunction,
    niv_pp: int,
    parities: list[str],
    u: np.ndarray | None = None,
) -> dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]]:
    r"""
    Solves every ``(channel, parity)`` sector in turn on the single rank of the job (the in-memory path of a
    single-rank run): the pp bubble is built once from ``giwk_dga``, then the singlet and the triplet channel run
    through :func:`solve_eliashberg_lanczos` one after the other, each over the requested parity sectors. Both
    pairing vertices are **consumed**.

    :param gamma_sing_pp: The singlet pairing vertex (the whole irreducible BZ; consumed).
    :param gamma_trip_pp: The triplet pairing vertex (the whole irreducible BZ; consumed).
    :param giwk_dga: The DGA Green's function (used to build the bubble).
    :param niv_pp: Number of positive fermionic frequencies of the pp box.
    :param parities: The parity labels to solve.
    :param u: The orbital action of the inversion for the parity sectors (see :func:`_project_gap_to_sector`).
    :return: ``{(channel, parity): (lambdas, gaps)}`` for every sector.
    """
    gchi0_q_pp = BubbleGenerator.create_generalized_chi0_q_pp_w0(giwk_dga, niv_pp, config.lattice.k_grid)
    config.logger.info("Created the bare bubble susceptibility in pp notation.")
    results: dict[tuple[SpinChannel, str], tuple[np.ndarray, list[GapFunction]]] = {}
    for gamma_pp in (gamma_sing_pp, gamma_trip_pp):
        sectors = solve_eliashberg_lanczos(gamma_pp, gchi0_q_pp, (0,), parities, u)
        results.update({(gamma_pp.channel, parity): sectors[parity] for parity in parities})
    return results


def solve(
    giwk_dga: GreensFunction | None,
    g_dmft: GreensFunction,
    u_loc: LocalInteraction,
    v_nonloc: Interaction | None,
    comm: MPI.Comm,
    chunk_budgets: memory_estimator.ChunkBudgets | None = None,
):
    r"""
    Drives the Eliashberg step: assembles the singlet and triplet pairing vertices from the saved
    ladder-DGA full vertices (optionally adding the local reducible diagrams; in the bare-vertex modes
    ``config.misc.do_flex`` and ``config.misc.do_rpa`` from the RPA vertices, counting the bare up-down interaction
    once instead), then solves the linearized gap equation for each channel and returns the leading eigenvalues and
    gap functions. A
    single rank solves in memory, sector after sector; a multi-rank job solves in memory as concurrent (channel x
    parity) sector teams that share one vertex window per channel and node when a node holds such a window (see
    :func:`plan_lanczos_teams`, decided from the available node memory), and otherwise with one team of all ranks
    whose vertex windows, vectors and bubble are split over the nodes, sector after sector (see
    :func:`plan_distributed_solve`).

    :param giwk_dga: The converged momentum-dependent DGA :class:`GreensFunction` on rank 0 (the pp-bubble build
        there is its only reader); ``None`` on every other rank.
    :param g_dmft: The local (DMFT) :class:`GreensFunction` (used for the local diagrams).
    :param u_loc: The bare local interaction :math:`U`.
    :param v_nonloc: The non-local interaction :math:`V^{\mathbf{q}}` on the full q-grid; only read on rank 0.
    :param comm: The MPI communicator.
    :param chunk_budgets: Chunk byte budgets sized by the driver from the memory estimate, of which the pairing-vertex
        budget is used; ``None`` gives the fair-share budget of
        :func:`~dgamore.memory_estimator.dynamic_chunk_budget`.
    :return: ``(results, inversion)``: a dict keyed by ``(channel, parity_label)`` mapping to ``(lambdas, gaps)`` of
        the leading eigenvalues and :class:`GapFunction` objects for each solved parity sector, and the orbital action
        of the inversion the sectors were projected with (see :func:`check_parity_sectors`; ``None`` for none or
        without the sectors). The eigenvalues are present on every rank, the gap functions on rank 0 only (an empty
        list elsewhere). When ``config.eliashberg.resolve_parity_sectors`` is set the parity labels are ``"even"``
        and ``"odd"``, otherwise a single unprojected ``"none"`` sector is returned.
    """
    logger = config.logger
    import psutil

    mpi_dist_irrk = sde_kernels.irreducible_momentum_distributor(comm)
    v_nonloc = sde_kernels.scatter_irreducible_interaction(v_nonloc, mpi_dist_irrk)

    # the check may switch the sectors off, so it runs before they are fixed
    inversion = check_parity_sectors(comm)
    parities = [parity for parity, _ in _parity_sectors(config.eliashberg.resolve_parity_sectors)]
    niv_pp = min(config.box.niw_core // 2, config.box.niv_core // 2)

    # one sector's single-rank residency, from the memory_estimator formula the driver's fit check reads as well;
    # it is what a single rank holds, and the figure the solver log lines quote
    per_sector_bytes = lanczos_solver_bytes(
        config.sys.n_bands,
        config.lattice.k_grid.nk_tot,
        config.lattice.k_grid.nk_irr,
        niv_pp,
        config.eliashberg.n_eig,
        comm.size,
    )
    # the node budget honors the scheduler's cgroup memory limit (e.g. slurm --mem), like the driver's fit check
    node_budget = psutil.virtual_memory().available
    cgroup_limit = mpi_utils.cgroup_memory_limit()
    if cgroup_limit is not None:
        node_budget = min(node_budget, cgroup_limit)

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) if comm.size > 1 or FORCE_DISTRIBUTED_SOLVER else None
    # on a multi-rank job the team plan (an allgather, so identical on every rank) decides: it is None when no
    # node holds even one channel's windows, and the node-distributed solve takes over; a single rank solves in memory
    rounds = layout = None
    if comm.size > 1 and not FORCE_DISTRIBUTED_SOLVER:
        rounds = plan_lanczos_teams(comm, node_comm, node_budget, niv_pp)
    if FORCE_DISTRIBUTED_SOLVER or (comm.size > 1 and rounds is None):
        layout = plan_distributed_solve(comm, node_comm, node_budget, niv_pp)
        logger.info(
            f"Eliashberg solver: no node holds one channel's vertex windows -> {2 * len(parities)} (channel x parity) "
            f"sector(s) in turn by one team of {comm.size} rank(s) split over {len(layout.nodes)} node(s), one "
            f"channel's window at a time (niv_pp = {niv_pp}, node budget {node_budget / 1024**3:.3f} GB)."
        )
    elif rounds is None:
        logger.info(
            f"Eliashberg solver: {2 * len(parities)} (channel x parity) sector(s) in turn on the single rank, "
            f"holding at most {per_sector_bytes / 1024**3:.3f} GB (niv_pp = {niv_pp})."
        )
    else:
        hosts = sorted({team.host for teams in rounds for team in teams})
        sizes = sorted({len(team.ranks) for teams in rounds for team in teams})
        logger.info(
            f"Eliashberg solver: {2 * len(parities)} (channel x parity) sector(s) solved by "
            f"{sum(len(teams) for teams in rounds)} team(s) of {'-'.join(map(str, sizes))} rank(s) on "
            f"{len(hosts)} node(s), "
            f"{'all sectors at once' if len(rounds) == 1 else 'one channel at a time'}; one vertex window per "
            f"channel and hosting node (niv_pp = {niv_pp}, node budget {node_budget / 1024**3:.3f} GB)."
        )

    chunk_bytes = (
        memory_estimator.dynamic_chunk_budget(mpi_utils.job_memory_total(), node_comm.size if node_comm else 1)
        if chunk_budgets is None
        else chunk_budgets.fq
    )

    gchi0_q_inv = None
    if not config.misc.bare_vertex:
        gchi0_q_inv = inverse_core_bubble(mpi_dist_irrk, node_comm)
        logger.info("Rebuilt the inverse core bubble.")
    f_dens_pp = pairing_vertex.dispatch_full_vertex_calculation(
        SpinChannel.DENS, u_loc, v_nonloc, niv_pp, mpi_dist_irrk, chunk_bytes, node_comm, gchi0_q_inv
    )
    logger.info("Built the density full ladder vertex in pp notation.")
    f_magn_pp = pairing_vertex.dispatch_full_vertex_calculation(
        SpinChannel.MAGN, u_loc, v_nonloc, niv_pp, mpi_dist_irrk, chunk_bytes, node_comm, gchi0_q_inv
    )
    logger.info("Built the magnetic full ladder vertex in pp notation.")
    if gchi0_q_inv is not None:
        gchi0_q_inv.free()
    mpi_dist_irrk.barrier()
    if comm.rank == 0:
        output_files.delete_files(config.output.eliashberg_path, f"{output_files.GIWK_FILE_STEM}.npy")

    gamma_sing_pp = f_dens_pp.scale(0.5).sub(f_magn_pp.scale(1.5, copy=True), copy=False)
    del f_dens_pp
    gamma_sing_pp.channel = SpinChannel.SING
    logger.info("Calculated full ladder-vertex (singlet) in pp notation.")

    gamma_trip_pp = gamma_sing_pp.add(f_magn_pp.scale(2.0))
    gamma_trip_pp.channel = SpinChannel.TRIP
    f_magn_pp.free()
    logger.info("Calculated full ladder-vertex (triplet) in pp notation.")

    if config.misc.bare_vertex:
        # Eqs. (4.49)-(4.52) with the bare up-down vertex U as the local full vertex and no pp-reducible diagrams:
        # its minus in crossed-slot form, -(-U_{1432}), keeps one copy of U, which both RPA ladders start with
        u_crossed = u_loc.permute_orbitals("abcd->adcb")
        gamma_sing_pp.add(u_crossed, copy=False)
        gamma_trip_pp.add(u_crossed, copy=False)
    else:
        # the local diagrams are reduced from local vertices on the full asymptotic fermionic box; one rank per node
        # reads and reduces them and broadcasts the pp-box-sized results, so that transient exists once per node
        f_ud_loc_pp_w0, gamma_ud_loc_pp_w0, phi_ud_loc_pp_w0 = _compute_once_per_node(
            node_comm, lambda: pairing_vertex.create_local_ud_diagrams_pp_w0(g_dmft, niv_pp)
        )

        if mpi_dist_irrk.my_rank == 0:
            f_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="f_ud_loc_pp_w0")
            phi_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="phi_ud_loc_pp_w0")
            gamma_ud_loc_pp_w0.save(output_dir=config.output.eliashberg_path, name="gamma_ud_loc_pp_w0")
            logger.info("Saved local ud diagrams in pp notation to file.")

        del f_ud_loc_pp_w0, gamma_ud_loc_pp_w0

        # special treatment of local full vertex that is subtracted with a different frequency notation and is
        # different from the regular pp
        f_ud_loc_transf_w0 = _compute_once_per_node(
            node_comm, lambda: pairing_vertex.create_local_f_ud_transformed_w0(niv_pp)
        )

        # Eqs. (4.49)-(4.52): the assembled vertex holds the negative crossed slot, so the local full vertex enters
        # with a relative minus, the pp-reducible diagrams phi with a plus, both in crossed-slot form ((v, -v'), 1432).
        phi_ud_loc_pp_w0 = phi_ud_loc_pp_w0.flip_frequency_axis(-1, copy=False).permute_orbitals(
            "abcd->adcb", copy=False
        )
        delta_loc = phi_ud_loc_pp_w0.sub(f_ud_loc_transf_w0)
        gamma_sing_pp.add(delta_loc, copy=False)
        gamma_trip_pp.add(delta_loc, copy=False)
        del phi_ud_loc_pp_w0, f_ud_loc_transf_w0, delta_loc

    if config.eliashberg.save_pairing_vertex:
        gamma_sing_pp.mat = mpi_dist_irrk.gather(gamma_sing_pp.mat)
        gamma_trip_pp.mat = mpi_dist_irrk.gather(gamma_trip_pp.mat)
        if comm.rank == 0:
            gamma_sing_pp.save(
                output_dir=config.output.eliashberg_path, name=f"gamma_irrq_{gamma_sing_pp.channel.value}_pp"
            )
            gamma_trip_pp.save(
                output_dir=config.output.eliashberg_path, name=f"gamma_irrq_{gamma_trip_pp.channel.value}_pp"
            )
        gamma_sing_pp.mat = mpi_dist_irrk.scatter(gamma_sing_pp.mat)
        gamma_trip_pp.mat = mpi_dist_irrk.scatter(gamma_trip_pp.mat)
        logger.info(f"Saved singlet and triplet pairing vertices in pp notation in the irreducible BZ to file.")

    if layout is not None:
        results = _solve_sectors_distributed(
            mpi_dist_irrk, comm, node_comm, gamma_sing_pp, gamma_trip_pp, giwk_dga, niv_pp, layout, inversion
        )
    elif rounds is None:
        results = _solve_sectors_in_memory(gamma_sing_pp, gamma_trip_pp, giwk_dga, niv_pp, parities, inversion)
    else:
        results = _solve_sectors_in_teams(
            mpi_dist_irrk, comm, node_comm, gamma_sing_pp, gamma_trip_pp, giwk_dga, niv_pp, rounds, inversion
        )
    if node_comm is not None:
        node_comm.Free()

    return results, inversion


# Fraction of a node's available host memory the sector packing may occupy (mirrors DGAmore.NODE_MEMORY_FRACTION).
NODE_MEMORY_FRACTION: float = 0.95
# Base MPI tags of the point-to-point transfers of the sector solves: the pp bubble ship to the hosting node roots,
# and one block per delivered sector (its gaps at SECTOR_TAG * (1 + sector_index)), clear of the gathers' tags.
BUBBLE_TAG: int = 900
SECTOR_TAG: int = 1000
# Base MPI tags of the node-distributed solve: the window rows of the build, the two gap exchanges of a matvec
# (EXCHANGE_TAG and EXCHANGE_TAG + 50) and the node shares of an eigenvector sent to rank 0.
WINDOW_TAG: int = 1500
EXCHANGE_TAG: int = 1600
GAP_TAG: int = 1700

# The team Arnoldi step orthogonalizes a second time only when the first pass left less than this share of the
# vector's norm (ARPACK's DGKS test), so the basis is streamed twice per step instead of four times.
REORTHOGONALIZE_BELOW: float = 0.717

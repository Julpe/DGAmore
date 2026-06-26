# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
Global configuration singleton. This module holds the process-wide mutable state of a DGAmore run as module-level
instances of the ``*Config`` classes (``box``, ``lattice``, ``sys``, ``dmft``, ``eliashberg``,
``lambda_correction``, ``self_consistency``, ``self_energy_interpolation``, ``output``, ``memory``, ``ana_cont``,
and the ``logger``). :class:`ConfigParser` populates these from a YAML file on rank 0, after which they are
broadcast to all MPI ranks; most modules read their parameters directly off this module rather than receiving them
as arguments, so mutating a field changes behavior everywhere. Each ``*Config`` class documents its fields; the
example YAML config is ``dgamore/dga_config.yaml``.
"""

import numpy as np

import dgamore.brillouin_zone as bz
from dgamore.dga_logger import DgaLogger
from dgamore.hamiltonian import Hamiltonian


class InteractionConfig:
    r"""
    Stores the interaction parameters. Currently only ``udd``, ``vdd``, ``jdd`` are used (local and Kanamori-type
    interactions); the remaining parameters are reserved for future use.

    :ivar float udd: Intra-orbital Hubbard interaction :math:`U_{dd}` on the d orbitals.
    :ivar float udp: d-p inter-orbital Hubbard interaction.
    :ivar float upp: Intra-orbital Hubbard interaction :math:`U_{pp}` on the p orbitals.
    :ivar float uppod: Off-diagonal p-p Hubbard interaction.
    :ivar float jdd: Hund's exchange :math:`J_{dd}` on the d orbitals.
    :ivar float jdp: d-p exchange.
    :ivar float jpp: Hund's exchange :math:`J_{pp}` on the p orbitals.
    :ivar float jppod: Off-diagonal p-p exchange.
    :ivar float vdd: Inter-orbital interaction :math:`V_{dd}` on the d orbitals.
    :ivar float vpp: Inter-orbital interaction :math:`V_{pp}` on the p orbitals.
    """

    def __init__(self):
        self.udd: float = 0.0
        self.udp: float = 0.0
        self.upp: float = 0.0
        self.uppod: float = 0.0
        self.jdd: float = 0.0
        self.jdp: float = 0.0
        self.jpp: float = 0.0
        self.jppod: float = 0.0
        self.vdd: float = 0.0
        self.vpp: float = 0.0


class BoxConfig:
    r"""
    Stores the Matsubara frequency box sizes. The main quantities live in the core region; explicit asymptotics
    correct it with shell-region quantities. The full region is the sum of core and shell and exists for convenience.

    :ivar int niw_core: Number of positive bosonic core frequencies :math:`\omega`.
    :ivar int niv_core: Number of positive fermionic core frequencies :math:`\nu`.
    :ivar int niv_shell: Number of positive fermionic shell frequencies (asymptotic correction region).
    :ivar int niv_full: Number of positive fermionic full-region frequencies (core + shell).
    :ivar int niv_dmft: Number of positive fermionic frequencies available in the DMFT 1-particle input.
    """

    def __init__(self):
        self.niw_core: int = -1
        self.niv_core: int = -1
        self.niv_shell: int = 0
        self.niv_full: int = 0
        self.niv_dmft: int = 0


class LatticeConfig:
    """
    Stores the lattice parameters: the symmetries, lattice type, input Hamiltonian and input interaction. The k and
    q grids are built from the number of k/q points and the lattice symmetries. See ``dga_config.yaml`` or the
    author's master's thesis for details.

    :ivar list symmetries: The lattice symmetries (a list of :class:`KnownSymmetries` or the auto sentinel).
    :ivar str type: How the kinetic Hamiltonian is provided (e.g. ``"from_wannier90"``).
    :ivar er_input: Path(s) to the hopping input.
    :vartype er_input: str | list
    :ivar str interaction_type: How the interaction is provided (e.g. ``"one_band_from_dmft"``).
    :ivar interaction_input: Path(s) to the interaction input.
    :vartype interaction_input: str | list
    :ivar tuple nk: Number of k-points per spatial direction.
    :ivar tuple nq: Number of q-points per spatial direction (defaults to ``nk``).
    :ivar InteractionConfig interaction: The :class:`InteractionConfig`.
    :ivar Hamiltonian hamiltonian: The :class:`Hamiltonian` instance.
    :ivar KGrid k_grid: The k-space :class:`KGrid`.
    :ivar KGrid q_grid: The q-space :class:`KGrid`.
    """

    def __init__(self):
        self.symmetries: list[bz.KnownSymmetries] = bz.two_dimensional_square_symmetries()
        self.type: str = "from_wannier90"
        self.er_input: str | list = "/path/to/file"
        self.interaction_type: str = "one_band_from_dmft"
        self.interaction_input: str | list = ""
        self.nk: tuple[int, int, int] = (16, 16, 1)
        self.nq: tuple[int, int, int] = self.nk

        self.interaction: InteractionConfig = InteractionConfig()
        self.hamiltonian: Hamiltonian = Hamiltonian()
        self.k_grid: bz.KGrid = bz.KGrid(self.nk, self.symmetries)
        self.q_grid: bz.KGrid = bz.KGrid(self.nq, self.symmetries)


class SelfConsistencyConfig:
    """
    Stores the self-consistency-loop parameters: the maximum iteration count, the convergence criterion, the mixing
    parameter/scheme and continuation options. If ``previous_sc_path`` is set, the loop resumes from a previous run.
    The mixing scheme can be ``"linear"``, ``"pulay"`` or ``"anderson"`` (the latter two use an iteration history).

    :ivar int max_iter: Maximum number of self-consistency iterations.
    :ivar float epsilon: Relative-residual convergence threshold on the self-energy.
    :ivar float mixing: The mixing parameter :math:`\\alpha`.
    :ivar str mixing_strategy: The mixing scheme (``"linear"``, ``"pulay"`` or ``"anderson"``).
    :ivar int mixing_history_length: Number of past iterations used by the accelerated mixing schemes.
    :ivar str previous_sc_path: Path to a previous self-consistency run to resume from (empty to start fresh).
    :ivar bool use_interpolated_sigma: Whether to resume from the interpolated rather than the raw self-energy.
    :ivar bool use_lambda_correction: Whether the self-consistency loop applies the lambda correction.
    :ivar bool restrict_chi_phys: Whether to restrict the physical susceptibility to positive values.
    :ivar anderson_prev_res: Cached previous Anderson residual (internal use).
    :vartype anderson_prev_res: float | None
    """

    def __init__(self):
        self.max_iter: int = 20
        self.epsilon: float = 1e-4
        self.mixing: float = 0.2
        self.mixing_strategy: str = "linear"
        self.mixing_history_length: int = 3
        self.previous_sc_path: str = ""
        self.use_interpolated_sigma: bool = False
        self.use_lambda_correction: bool = False
        self.restrict_chi_phys: bool = False
        self.anderson_prev_res: float | None = None


class EliashbergConfig:
    """
    Stores the Eliashberg-step configuration: whether to run it, power-iteration settings, and saving options for the
    pairing/full vertex in pp notation.

    :ivar bool perform_eliashberg: Whether to solve the Eliashberg equation.
    :ivar bool save_pairing_vertex: Whether to save the singlet/triplet pairing vertices.
    :ivar bool save_fq: Whether to save the full ladder vertex (in ph notation) in the irreducible BZ.
    :ivar bool construct_fq_cheap: Whether to build the full vertex on the smaller pp frequency box (cheaper).
    :ivar int n_eig: Number of leading eigenvalues/gap functions to compute per channel.
    :ivar float epsilon: Convergence tolerance for the Lanczos eigensolver.
    :ivar str symmetry: Initial gap-function symmetry (``"d-wave"``, ``"p-wave-x"``, ``"p-wave-y"``, or ``"random"``).
    :ivar bool include_local_part: Whether to add the local reducible pp diagrams to the pairing vertex.
    :ivar str subfolder_name: Output subfolder name for Eliashberg results.
    """

    def __init__(self):
        self.perform_eliashberg: bool = False
        self.save_pairing_vertex: bool = False
        self.save_fq: bool = False
        self.construct_fq_cheap: bool = False
        self.n_eig: int = 4
        self.epsilon: float = 1e-6
        self.symmetry: str = "random"
        self.include_local_part: bool = True
        self.subfolder_name: str = "Eliashberg"


class LambdaCorrectionConfig:
    """
    Stores the lambda-correction configuration.

    :ivar bool perform_lambda_correction: Whether to apply the Moriya lambda correction.
    :ivar str type: Correction type: ``"sp"`` (magnetic channel only) or ``"spch"`` (density and magnetic channels).
    """

    def __init__(self):
        self.perform_lambda_correction: bool = False
        self.type: str = "spch"


class DmftConfig:
    r"""
    Stores the DMFT input-file parameters: the input path, the 1- and 2-particle data filenames, symmetrization
    options and the inequivalent-atom structure.

    :ivar str type: DMFT solver/format of the input (e.g. ``"w2dyn"``).
    :ivar str input_path: Directory containing the DMFT input files.
    :ivar str fname_1p: Filename of the 1-particle data.
    :ivar str fname_2p: Filename of the 2-particle data.
    :ivar bool do_sym_v_vp: Whether to symmetrize the 2-particle data with respect to :math:`(\nu, \nu')`.
    :ivar list symmetrize_orbitals: 1-based orbital indices to symmetrize over (empty for none).
    :ivar int n_ineq: Number of inequivalent atoms.
    :ivar list ineq_ordering: Ordering of the inequivalent atoms.
    :ivar list n_bands_per_ineq: Number of bands per inequivalent atom.
    """

    def __init__(self):
        self.type: str = "w2dyn"
        self.input_path: str = "./"
        self.fname_1p: str = "1p-data.hdf5"
        self.fname_2p: str = "g4iw_sym.hdf5"
        self.do_sym_v_vp: bool = True
        self.symmetrize_orbitals: list = []
        self.n_ineq: int = 1
        self.ineq_ordering: list[int] = [1]
        self.n_bands_per_ineq = []


class SystemConfig:
    r"""
    Stores the physical system parameters and derived quantities updated during the run.

    :ivar float beta: Inverse temperature :math:`\beta`.
    :ivar float mu: Chemical potential :math:`\mu` (updated during self-consistency).
    :ivar float mu_dmft: Chemical potential of the DMFT input.
    :ivar float n: Total filling :math:`n`.
    :ivar int n_bands: Number of bands.
    :ivar numpy.ndarray occ: Local (k-averaged) occupation matrix.
    :ivar numpy.ndarray occ_k: k-resolved occupation matrix.
    :ivar numpy.ndarray occ_dmft: Local occupation matrix from the DMFT input.
    :ivar list occ_dmft_per_ineq: DMFT occupation matrices per inequivalent atom.
    """

    def __init__(self):
        self.beta: float = 0.0
        self.mu: float = 0.0
        self.mu_dmft: float = 0.0
        self.n: float = 0.0
        self.n_bands: int = 0
        self.occ: np.ndarray = np.ndarray(0)
        self.occ_k: np.ndarray = np.ndarray(0)
        self.occ_dmft: np.ndarray = np.ndarray(0)
        self.occ_dmft_per_ineq: list[np.ndarray] = []


class SelfEnergyInterpolationConfig:
    r"""
    Stores the self-energy interpolation parameters (re-gridding to a different temperature/frequency box).

    :ivar bool do_interpolation: Whether to additionally save an interpolated self-energy each iteration.
    :ivar float beta_target: Target inverse temperature :math:`\beta` of the interpolation.
    :ivar int niv_target: Target number of positive fermionic frequencies of the interpolation.
    """

    def __init__(self):
        self.do_interpolation: bool = False
        self.beta_target: float = 1.0
        self.niv_target: int = 10


class OutputConfig:
    """
    Stores the output paths.

    :ivar str output_path: Directory where the main results are written.
    :ivar bool do_plotting: Whether to produce plots (rank 0 only).
    :ivar str plotting_path: Directory where plots are written.
    :ivar str plotting_subfolder_name: Subfolder name (under ``plotting_path``) for the plots.
    :ivar str eliashberg_path: Directory where Eliashberg results are written.
    """

    def __init__(self):
        self.output_path: str = ""
        self.do_plotting: bool = True
        self.plotting_path: str = "./Plots/"
        self.plotting_subfolder_name: str = "Plots"
        self.eliashberg_path: str = "./Eliashberg/"


class MemoryConfig:
    """
    Stores the speed-vs-memory trade-off switches. Each flag, when True, selects a slower but more memory-frugal code
    path for the corresponding quantity.

    :ivar bool save_memory_for_chi0q: Use the per-q einsum bubble instead of the FFT bubble.
    :ivar bool save_memory_for_chiq_aux: Use the per-q auxiliary-susceptibility path and per-rank BZ mapping.
    :ivar bool save_memory_for_sde: Use the q-loop self-energy contraction instead of the FFT one.
    :ivar bool save_memory_for_fq: Use the per-q full-vertex construction in the Eliashberg step.
    :ivar bool save_memory_for_lanczos: Use the frequency-distributed Lanczos solver.
    """

    def __init__(self):
        self.save_memory_for_chi0q: bool = False
        self.save_memory_for_chiq_aux: bool = False
        self.save_memory_for_sde: bool = False
        self.save_memory_for_fq: bool = False
        self.save_memory_for_lanczos: bool = False


class AnaContConfig:
    """
    Stores the analytic-continuation (maximum-entropy) configuration.

    :ivar bool do_spectrum_dga: Whether to continue the DGA Green's function to real frequencies.
    :ivar bool do_spectrum_dmft: Whether to continue the DMFT Green's function to real frequencies.
    :ivar int w_count: Number of real-frequency points.
    :ivar bool plot_spectrum: Whether to plot the resulting spectral function.
    :ivar list k_path: The k-path (list of ``(kx, ky, kz, label)`` tuples) for the spectral function.
    :ivar tuple energy_window: The real-frequency window ``(w_min, w_max)``.
    """

    def __init__(self):
        self.do_spectrum_dga: bool = False
        self.do_spectrum_dmft: bool = False
        self.w_count: int = 1001
        self.plot_spectrum: bool = False
        self.k_path: list[tuple[float, float, float, str]] = [
            (0.0, 0.0, 0.0, "Gamma"),  # default for cubic, must be in primitive k-space
            (0.0, 0.5, 0.0, "X"),
            (0.5, 0.5, 0.0, "M"),
            (0.0, 0.0, 0.0, "Gamma"),
        ]
        self.energy_window: tuple[float, float] = (-2, 3)


logger: DgaLogger
box: BoxConfig = BoxConfig()
lattice: LatticeConfig = LatticeConfig()
lambda_correction: LambdaCorrectionConfig = LambdaCorrectionConfig()
dmft: DmftConfig = DmftConfig()
sys: SystemConfig = SystemConfig()
output: OutputConfig = OutputConfig()
self_energy_interpolation: SelfEnergyInterpolationConfig = SelfEnergyInterpolationConfig()
self_consistency: SelfConsistencyConfig = SelfConsistencyConfig()
eliashberg: EliashbergConfig = EliashbergConfig()
memory: MemoryConfig = MemoryConfig()
ana_cont: AnaContConfig = AnaContConfig()

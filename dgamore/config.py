# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import numpy as np

import dgamore.brillouin_zone as bz
from dgamore.dga_logger import DgaLogger
from dgamore.hamiltonian import Hamiltonian


class InteractionConfig:
    """
    Class to store the interaction parameters. Currently, we only make use of udd, vdd, jdd for local and Kanamori-type
    interactions. Other parameters are currently not used.
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
    """
    Class to store the box sizes. The main quantities are available in the core region. Due to explicit asymptotics,
    we can correct the core region by shell-region quantities. The full region is the sum of the core and shell regions
    and the variable exists for convenience.
    """

    def __init__(self):
        self.niw_core: int = -1
        self.niv_core: int = -1
        self.niv_shell: int = 0
        self.niv_full: int = 0
        self.niv_dmft: int = 0


class LatticeConfig:
    """
    Class to store the lattice parameters. The lattice is defined by the symmetries, the type of lattice, the input
    Hamiltonian and the input interaction. The k and q grids are defined by the number of k and q points and the
    symmetries of the lattice. For more information, have a look at the file dga_config.yaml or check out my master's
    thesis.
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
    Class to store the self-consistency parameters. The self-consistency loop is controlled by the maximum number of
    iterations, the convergence criterion epsilon, the mixing parameter and the option to save the quantities throughout
    the self-consistency iteration. If previous_sc_path is set, the self-consistency will be started from the previous
    self-consistency iteration found in this path. We also added the possibility to change the mixing scheme from "linear" to
    (periodic) Pulay mixing, using a history of the last couple iterations if available.
    """

    def __init__(self):
        self.max_iter: int = 20
        self.epsilon: float = 1e-4
        self.mixing: float = 0.2
        self.mixing_strategy: str = "linear"
        self.mixing_history_length: int = 3
        self.previous_sc_path: str = ""
        self.use_lambda_correction: bool = False
        self.restrict_chi_phys: bool = False
        self.anderson_prev_res: float | None = None


class EliashbergConfig:
    """
    Class to store the configuration for the Eliashberg equation. It is defined by the option to perform the Eliashberg
    equation, the subfolder name, settings for the power iteration and the option to save the pairing vertex or
    the full vertex in pp notation.
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
    Class to store the configuration for the lambda correction. You can set the option to perform the lambda
    correction and the type. Currently available are "sp" and "spch" for the magnetic channel and density + magnetic
    channel, respectively.
    """

    def __init__(self):
        self.perform_lambda_correction: bool = False
        self.type: str = "spch"


class DmftConfig:
    """
    Class to store the DMFT input file parameters. The DMFT section contains the input path, the filenames
    for the 1-particle and 2-particle data and the option to symmetrize the 2-particle data with respect to v and v'.
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
    """
    Class to store the system parameters. It contains the number of bands, the inverse temperature beta, the
    chemical potential mu and the total filling n. The occupation numbers for the different bands are
    stored in the occ array for the local case and occ_k for the k-dependent occupation.
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
    """
    Class to store the interpolation parameters for the self-energy, such as target beta and target niv.
    """

    def __init__(self):
        self.do_interpolation: bool = False
        self.beta_target: float = 1.0
        self.niv_target: int = 10


class OutputConfig:
    """
    Class to store the output parameters. The output path is the path where the results are saved, the plotting path
    is the path where the plots are saved, and the plotting subfolder name is the name of the subfolder in the plotting
    path where the plots are saved to. The Eliashberg path is the absolute path where the Eliashberg results are saved.
    """

    def __init__(self):
        self.output_path: str = ""
        self.do_plotting: bool = True
        self.plotting_path: str = "./Plots/"
        self.plotting_subfolder_name: str = "Plots"
        self.eliashberg_path: str = "./Eliashberg/"


class MemoryConfig:
    """
    Class to store parameters that decide, whether the code should perform very fast (but memory-intensive) calculations
    or less memory-intensive calculations that are substantially slower.
    """

    def __init__(self):
        self.save_memory_for_chi0q: bool = False
        self.save_memory_for_chiq_aux: bool = False
        self.save_memory_for_sde: bool = False
        self.save_memory_for_fq: bool = False
        self.save_memory_for_lanczos: bool = False


class AnaContConfig:
    """
    Class that takes care of the configuration for the analytic continuation using the maximum entropy method.
    """

    def __init__(self):
        self.do_ana_cont_green_dga: bool = False
        self.do_ana_cont_green_dmft: bool = False
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

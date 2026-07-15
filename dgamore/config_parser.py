# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
YAML configuration parsing. :class:`ConfigParser` reads the run's YAML config (rank 0), broadcasts it to all MPI
ranks, and populates the global :mod:`dgamore.config` singletons section by section, falling back to the
``*Config`` defaults for any missing key/section. The config file location is taken from the ``-p``/``-c``
command-line arguments.
"""

import argparse
import os

from mpi4py import MPI
from ruamel.yaml import YAML

import dgamore.config as config
from dgamore.config import *
from dgamore.dga_logger import DgaLogger


class ConfigParser:
    """
    Parses the config file and builds the global ``dgamore.config`` singletons (``box``, ``lattice``, ``dmft``, ...). The configuration is then broadcast to all
    processes. The config file location can be specified with the path and/or name arguments when executing the main
    Python file.
    """

    def __init__(self):
        """
        Initializes an empty parser; the YAML content is loaded later by :meth:`parse_config`.
        """
        self._config_file = None

    def parse_config(self, comm: MPI.Comm = None, path: str = "./", name: str = "dga_config.yaml"):
        """
        Parses the config file on rank 0, broadcasts it to all ranks, and populates the global config singletons. The
        ``-c``/``-p`` command-line arguments override the ``name``/``path`` defaults.

        :param comm: The MPI communicator (rank 0 reads the file and broadcasts it).
        :param path: Default directory of the config file.
        :param name: Default config file name.
        :return: ``self`` (for chaining).
        """
        parser = argparse.ArgumentParser(
            prog="DGApy", description="Multi-orbital dynamical vertex approximation solver."
        )
        parser.add_argument("-c", "--config", nargs="?", default=name, type=str, help=" Config file name. ")
        parser.add_argument("-p", "--path", nargs="?", default=path, type=str, help=" Path to the config file. ")

        if comm.rank == 0:
            args = parser.parse_args()
            self._config_file = YAML().load(open(os.path.join(args.path, args.config)))

        config.logger = DgaLogger(comm)

        self._config_file = comm.bcast(self._config_file, root=0)

        self._build_config_from_file(self._config_file)
        return self

    def save_config_file(self, path: str = "./", name: str = "dga_config.yaml") -> None:
        """
        Dumps the loaded config back to a YAML file (typically into the output folder to record the used parameters).

        :param path: Directory to write the config file to.
        :param name: File name to write.
        :return: None.
        """
        with open(os.path.join(path, name), "w+") as file:
            YAML().dump(self._config_file, file)

    def _build_config_from_file(self, config_file):
        """
        Populates every global config singleton from the parsed YAML content.

        :param config_file: The parsed YAML mapping.
        :return: None.
        """
        config.dmft = self._build_dmft_config(config_file)
        config.output = self._build_output_config(config_file)
        config.self_consistency = self._build_self_consistency_config(config_file)
        config.stabilization = self._build_stabilization_config(config_file)
        config.eliashberg = self._build_eliashberg_config(config_file)
        config.lambda_correction = self._build_lambda_correction_config(config_file)
        config.box = self._build_box_config(config_file)
        config.lattice = self._build_lattice_config(config_file)
        config.self_energy_interpolation = self._build_self_energy_interpolation_config(config_file)
        config.sys = self._build_system_config(config_file)
        config.memory = self._build_memory_config(config_file)
        config.ana_cont = self._build_ana_cont_config(config_file)

    def _build_box_config(self, config_file) -> BoxConfig:
        """
        Builds the frequency-box config from the ``box_sizes`` section (defaults if absent), deriving ``niv_full``
        from the core and shell sizes.

        :param config_file: The parsed YAML mapping.
        :return: The populated :class:`BoxConfig`.
        """
        conf = BoxConfig()
        try:
            section = config_file["box_sizes"]
        except KeyError:
            config.logger.info(f"'box_sizes' section not found. Using default values.")
            return conf

        conf.niw_core = self._try_parse(section, "niw_core", conf.niw_core)
        conf.niv_core = self._try_parse(section, "niv_core", conf.niv_core)
        conf.niv_shell = self._try_parse(section, "niv_shell", conf.niv_shell)
        if conf.niv_shell <= 0:
            config.logger.info(f"'niv_shell' is set to {conf.niv_shell}. No asymptotics will be used.")
            conf.niv_shell = 0
        conf.niv_full = conf.niv_core + conf.niv_shell

        return conf

    def _build_lattice_config(self, config_file) -> LatticeConfig:
        """
        Builds the lattice config from the ``lattice`` section (defaults if absent): grid size, symmetries, the
        momentum grid, and the lattice/interaction input.

        :param config_file: The parsed YAML mapping.
        :return: The populated :class:`LatticeConfig`.
        """
        conf = LatticeConfig()
        try:
            section = config_file["lattice"]
        except KeyError:
            config.logger.info(f"'lattice' section not found. Using default values.")
            return conf

        conf.nk = self._try_parse(section, "nk", conf.nk)
        symmetries = self._try_parse(section, "symmetries", "two_dimensional_square")
        conf.symmetries = bz.get_lattice_symmetries_from_string(symmetries)

        conf.k_grid = bz.KGrid(conf.nk, conf.symmetries)

        conf.type = self._try_parse(section, "type", conf.type)
        conf.er_input = section["hr_input"]  # can be multiple types

        conf.interaction_type = self._try_parse(section, "interaction_type", conf.interaction_type)
        conf.interaction_input = self._try_parse(section, "interaction_input", conf.interaction_input)

        return conf

    def _build_dmft_config(self, config_file) -> DmftConfig:
        """
        Builds the DMFT input config from the ``dmft_input`` section (defaults if absent).

        :param config_file: The parsed YAML mapping.
        :return: The populated :class:`DmftConfig`.
        """
        conf = DmftConfig()
        try:
            section = config_file["dmft_input"]
        except KeyError:
            config.logger.info(f"'dmft_input' section not found. Using default values.")
            return conf

        conf.type = self._try_parse(section, "type", conf.type)
        conf.input_path = self._try_parse(section, "input_path", conf.input_path)
        conf.fname_1p = self._try_parse(section, "fname_1p", conf.fname_1p)
        conf.fname_2p = self._try_parse(section, "fname_2p", conf.fname_2p)
        conf.symmetrize_orbitals = self._try_parse(section, "symmetrize_orbitals", conf.symmetrize_orbitals)
        conf.n_ineq = self._try_parse(section, "n_ineq", conf.n_ineq)
        conf.ineq_ordering = self._try_parse(section, "ineq_ordering", conf.ineq_ordering)

        return conf

    def _build_system_config(self, _) -> SystemConfig:
        """
        Returns an empty system config; its fields are filled later by the main routine from the DMFT input.

        :param _: Unused (the parsed YAML mapping).
        :return: A default :class:`SystemConfig`.
        """
        return SystemConfig()

    def _build_output_config(self, config_file) -> OutputConfig:
        """
        Builds the output config from the ``output`` section (defaults if absent), defaulting ``output_path`` to the
        DMFT input path when unset.

        :param config_file: The parsed YAML mapping.
        :return: The populated :class:`OutputConfig`.
        """
        conf = OutputConfig()
        try:
            section = config_file["output"]
        except KeyError:
            config.logger.info(f"'output' section not found. Using default values.")
            return conf

        conf.do_plotting = self._try_parse(section, "do_plotting", conf.do_plotting)
        conf.plotting_subfolder_name = self._try_parse(section, "plotting_subfolder_name", conf.plotting_subfolder_name)
        conf.output_path = self._try_parse(section, "output_path", conf.output_path)

        if not conf.output_path or conf.output_path == "":
            config.logger.info(f"'output_path' not set in config. Setting 'output_path' = '{config.dmft.input_path}'.")
            conf.output_path = config.dmft.input_path

        return conf

    def _build_self_consistency_config(self, config_file) -> SelfConsistencyConfig:
        """
        Builds the self-consistency config from the ``self_consistency`` section (defaults if absent).

        :param config_file: The parsed YAML mapping.
        :return: The populated :class:`SelfConsistencyConfig`.
        """
        conf = SelfConsistencyConfig()
        try:
            section = config_file["self_consistency"]
        except KeyError:
            config.logger.info(f"'self_consistency' section not found. Using default values.")
            return conf

        conf.max_iter = self._try_parse(section, "max_iter", conf.max_iter)
        conf.epsilon = self._try_parse(section, "epsilon", conf.epsilon)
        conf.mixing = self._try_parse(section, "mixing", conf.mixing)
        conf.mixing_strategy = self._try_parse(section, "mixing_strategy", conf.mixing_strategy)
        conf.mixing_history_length = self._try_parse(section, "mixing_history_length", conf.mixing_history_length)
        conf.previous_sc_path = self._try_parse(section, "previous_sc_path", conf.previous_sc_path)
        conf.use_interpolated_sigma = self._try_parse(section, "use_interpolated_sigma", conf.use_interpolated_sigma)

        return conf

    def _build_stabilization_config(self, config_file) -> StabilizationConfig:
        """
        Builds the stabilization config from the ``stabilization`` section (defaults if absent).

        :param config_file: The parsed YAML mapping.
        :return: The populated :class:`StabilizationConfig`.
        """
        conf = StabilizationConfig()
        try:
            section = config_file["stabilization"]
        except KeyError:
            config.logger.info(f"'stabilization' section not found. Using default values.")
            return conf

        conf.use_lambda_correction = self._try_parse(section, "use_lambda_correction", conf.use_lambda_correction)
        conf.use_chi_phys_restriction = self._try_parse(
            section, "use_chi_phys_restriction", conf.use_chi_phys_restriction
        )
        conf.use_lambda_annealing = self._try_parse(section, "use_lambda_annealing", conf.use_lambda_annealing)

        return conf

    def _build_eliashberg_config(self, config_file) -> EliashbergConfig:
        """
        Builds the Eliashberg config from the ``eliashberg`` section (defaults if absent).

        :param config_file: The parsed YAML mapping.
        :return: The populated :class:`EliashbergConfig`.
        """
        conf = EliashbergConfig()
        try:
            section = config_file["eliashberg"]
        except KeyError:
            config.logger.info(f"'eliashberg' section not found. Using default values.")
            return conf

        conf.perform_eliashberg = self._try_parse(section, "perform_eliashberg", conf.perform_eliashberg)
        conf.save_pairing_vertex = self._try_parse(section, "save_pairing_vertex", conf.save_pairing_vertex)
        conf.save_fq = self._try_parse(section, "save_fq", conf.save_fq)
        conf.n_eig = self._try_parse(section, "n_eig", conf.n_eig)
        conf.epsilon = self._try_parse(section, "epsilon", conf.epsilon)
        conf.symmetry = self._try_parse(section, "symmetry", conf.symmetry)
        conf.include_local_part = self._try_parse(section, "include_local_part", conf.include_local_part)
        conf.symmetrize_degenerate_gaps = self._try_parse(
            section, "symmetrize_degenerate_gaps", conf.symmetrize_degenerate_gaps
        )
        conf.subfolder_name = self._try_parse(section, "subfolder_name", conf.subfolder_name)

        return conf

    def _build_lambda_correction_config(self, config_file):
        """
        Builds the lambda-correction config from the ``lambda_correction`` section (defaults if absent).

        :param config_file: The parsed YAML mapping.
        :return: The populated :class:`LambdaCorrectionConfig`.
        """
        conf = LambdaCorrectionConfig()
        try:
            section = config_file["lambda_correction"]
        except KeyError:
            config.logger.info(f"'lambda_correction' section not found. Using default values.")
            return conf

        conf.perform_lambda_correction = self._try_parse(
            section, "perform_lambda_correction", conf.perform_lambda_correction
        )
        conf.type = self._try_parse(section, "type", conf.type)

        return conf

    def _build_self_energy_interpolation_config(self, config_file):
        """
        Builds the self-energy interpolation config from the ``self_energy_interpolation`` section (defaults if absent).

        :param config_file: The parsed YAML mapping.
        :return: The populated :class:`SelfEnergyInterpolationConfig`.
        """
        conf = SelfEnergyInterpolationConfig()
        try:
            section = config_file["self_energy_interpolation"]
        except KeyError:
            config.logger.info(f"'self_energy_interpolation' section not found. Using default values.")
            return conf

        conf.do_interpolation = self._try_parse(section, "do_interpolation", conf.do_interpolation)
        conf.beta_target = self._try_parse(section, "target_beta", conf.beta_target)
        conf.niv_target = self._try_parse(section, "target_niv", conf.niv_target)

        return conf

    def _build_memory_config(self, config_file):
        """
        Builds the memory config from the ``memory`` section (defaults if absent).

        :param config_file: The parsed YAML mapping.
        :return: The populated :class:`MemoryConfig`.
        """
        conf = MemoryConfig()
        try:
            section = config_file["memory"]
        except KeyError:
            config.logger.info(f"'memory' section not found. Using default values.")
            return conf

        conf.save_memory_for_chi0q = self._try_parse(section, "save_memory_for_chi0q", conf.save_memory_for_chi0q)
        conf.save_memory_for_chiq_aux = self._try_parse(
            section, "save_memory_for_chiq_aux", conf.save_memory_for_chiq_aux
        )
        conf.save_memory_for_fq = self._try_parse(section, "save_memory_for_fq", conf.save_memory_for_fq)
        conf.save_memory_for_lanczos = self._try_parse(section, "save_memory_for_lanczos", conf.save_memory_for_lanczos)
        conf.use_shared_memory_common_obj = self._try_parse(
            section, "use_shared_memory_common_obj", conf.use_shared_memory_common_obj
        )

        return conf

    def _build_ana_cont_config(self, config_file):
        """
        Builds the analytic-continuation config from the ``ana_cont`` section (defaults if absent).

        :param config_file: The parsed YAML mapping.
        :return: The populated :class:`AnaContConfig`.
        """
        conf = AnaContConfig()
        try:
            section = config_file["ana_cont"]
        except KeyError:
            config.logger.info(f"'ana_cont' section not found. Using default values.")
            return conf

        conf.do_spectrum_dga = self._try_parse(section, "do_spectrum_dga", conf.do_spectrum_dga)
        conf.do_spectrum_dmft = self._try_parse(section, "do_spectrum_dmft", conf.do_spectrum_dmft)
        conf.w_count = self._try_parse(section, "w_count", conf.w_count)
        conf.plot_spectrum = self._try_parse(section, "plot_spectrum", conf.plot_spectrum)
        conf.k_path = self._try_parse(section, "k_path", conf.k_path)
        conf.energy_window = self._try_parse(section, "energy_window", conf.energy_window)

        return conf

    def _try_parse(self, config_section, key: str, default_value):
        """
        Reads ``key`` from a config section and coerces it to the type of ``default_value``, returning the default if
        the key is missing or cannot be coerced.

        :param config_section: The YAML sub-mapping to read from.
        :param key: The key to look up.
        :param default_value: The fallback value (its type also determines the coercion target).
        :return: The parsed value, or ``default_value`` on a missing key or coercion failure.
        """
        if key not in config_section:
            return default_value

        value_type = type(default_value)
        try:
            return value_type(config_section[key])
        except ValueError:
            config.logger.info(f"Could not parse value for {key}. Using default value: {default_value}.")
            return default_value

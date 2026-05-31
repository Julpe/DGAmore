# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import argparse
import os

from mpi4py import MPI
from ruamel.yaml import YAML

import moldga.config as config
from moldga.config import *
from moldga.dga_logger import DgaLogger


class ConfigParser:
    """
    Parses the config file and builds the DgaConfig singleton class. The Configuration is then broadcasted to all
    processes. The config file location can be specified with the path and/or name arguments when executing the main
    Python file.
    """

    def __init__(self):
        self._config_file = None

    def parse_config(self, comm: MPI.Comm = None, path: str = "./", name: str = "dga_config.yaml"):
        """
        Parses the config file and builds the DgaConfig singleton class. Broadcasts the configuration to all
        processes. The config file location can be specified with the path and/or name arguments when executing the
        main Python file. If the config file is not found, it will raise an error.
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
        Provides a way to dump the current config file to a separate location. Usually, the current configuration file
        is saved to the output folder to keep track of the used parameters.
        """
        with open(os.path.join(path, name), "w+") as file:
            YAML().dump(self._config_file, file)

    def _build_config_from_file(self, config_file):
        """
        Builds the full DgaConfig from the config file.
        """
        config.dmft = self._build_dmft_config(config_file)
        config.output = self._build_output_config(config_file)
        config.self_consistency = self._build_self_consistency_config(config_file)
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
        Builds the box config from the config file. Mainly concerned with the frequency boxes.
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
        Builds the lattice config from the config file. Mainly concerned with the lattice and interaction input.
        """
        conf = LatticeConfig()
        try:
            section = config_file["lattice"]
        except KeyError:
            config.logger.info(f"'lattice' section not found. Using default values.")
            return conf

        conf.nk = self._try_parse(section, "nk", conf.nk)

        if "nq" not in section:
            config.logger.info("'nq' not set in config. Setting 'nq' = 'nk'.")
            conf.nq = conf.nk
        else:
            conf.nq = self._try_parse(section, "nq", conf.nq)

        symmetries = self._try_parse(section, "symmetries", "two_dimensional_square")
        conf.symmetries = bz.get_lattice_symmetries_from_string(symmetries)

        conf.k_grid = bz.KGrid(conf.nk, conf.symmetries)
        conf.q_grid = bz.KGrid(conf.nq, conf.symmetries)

        conf.type = self._try_parse(section, "type", conf.type)
        conf.er_input = section["hr_input"]  # can be multiple types

        conf.interaction_type = self._try_parse(section, "interaction_type", conf.interaction_type)
        conf.interaction_input = self._try_parse(section, "interaction_input", conf.interaction_input)

        return conf

    def _build_dmft_config(self, config_file) -> DmftConfig:
        """
        Builds the DMFT config from the config file. Mainly concerned with input data.
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
        conf.do_sym_v_vp = self._try_parse(section, "do_sym_v_vp", conf.do_sym_v_vp)
        conf.symmetrize_orbitals = self._try_parse(section, "symmetrize_orbitals", conf.symmetrize_orbitals)
        conf.n_ineq = self._try_parse(section, "n_ineq", conf.n_ineq)
        conf.ineq_ordering = self._try_parse(section, "ineq_ordering", conf.ineq_ordering)

        return conf

    def _build_system_config(self, _) -> SystemConfig:
        """
        Builds the system config. This will be filled from the outside by the main routine.
        """
        return SystemConfig()

    def _build_output_config(self, config_file) -> OutputConfig:
        """
        Builds the output config from the config file. Mainly concerned with plotting and saving quantities.
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
        Builds the self-consistency config from the config file. Mainly concerned with the self-consistency loop.
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
        conf.use_lambda_correction = self._try_parse(section, "use_lambda_correction", conf.use_lambda_correction)
        conf.restrict_chi_phys = self._try_parse(section, "restrict_chi_phys", conf.restrict_chi_phys)

        return conf

    def _build_eliashberg_config(self, config_file) -> EliashbergConfig:
        """
        Builds the Eliashberg config from the config file. Mainly concerned with the Eliashberg equation.
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
        conf.construct_fq_cheap = self._try_parse(section, "construct_fq_cheap", conf.construct_fq_cheap)
        conf.n_eig = self._try_parse(section, "n_eig", conf.n_eig)
        conf.epsilon = self._try_parse(section, "epsilon", conf.epsilon)
        conf.symmetry = self._try_parse(section, "symmetry", conf.symmetry)
        conf.include_local_part = self._try_parse(section, "include_local_part", conf.include_local_part)
        conf.subfolder_name = self._try_parse(section, "subfolder_name", conf.subfolder_name)

        return conf

    def _build_lambda_correction_config(self, config_file):
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
        Builds the self-energy interpolation config from the config file.
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
        Builds the memory config from the config file.
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
        conf.save_memory_for_sde = self._try_parse(section, "save_memory_for_sde", conf.save_memory_for_sde)
        conf.save_memory_for_fq = self._try_parse(section, "save_memory_for_fq", conf.save_memory_for_fq)
        conf.save_memory_for_lanczos = self._try_parse(section, "save_memory_for_lanczos", conf.save_memory_for_lanczos)

        return conf

    def _build_ana_cont_config(self, config_file):
        """
        Builds the analytic continuation config from the config file.
        """
        conf = AnaContConfig()
        try:
            section = config_file["ana_cont"]
        except KeyError:
            config.logger.info(f"'ana_cont' section not found. Using default values.")
            return conf

        conf.do_ana_cont_green_dga = self._try_parse(section, "do_ana_cont_green_dga", conf.do_ana_cont_green_dga)
        conf.do_ana_cont_green_dmft = self._try_parse(section, "do_ana_cont_green_dmft", conf.do_ana_cont_green_dmft)
        conf.w_count = self._try_parse(section, "w_count", conf.w_count)
        conf.plot_spectrum = self._try_parse(section, "plot_spectrum", conf.plot_spectrum)
        conf.k_path = self._try_parse(section, "k_path", conf.k_path)
        conf.energy_window = self._try_parse(section, "energy_window", conf.energy_window)

        return conf

    def _try_parse(self, config_section, key: str, default_value):
        """
        Tries to parse the value for the key in the config_section. If it fails, the default_value is returned. Parses
        the value to the type of the param default_value.
        """
        if key not in config_section:
            return default_value

        value_type = type(default_value)
        try:
            return value_type(config_section[key])
        except ValueError:
            config.logger.info(f"Could not parse value for {key}. Using default value: {default_value}.")
            return default_value

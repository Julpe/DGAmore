# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import logging
import os
from datetime import datetime

import mpi4py.MPI as MPI


class DgaLogger:
    """
    A logger class that handles logging messages in a distributed environment using MPI. Currently, it logs messages
    to the stdout and can also be extended to log to a file.
    """

    def __init__(self, comm: MPI.Comm, output_path: str = "./", filename: str = "dga.log"):
        self._comm = comm
        self._output_path = output_path
        self._filename = filename
        self._filepath = os.path.join(output_path, filename)

        self._logger = logging.getLogger("dga_logger")
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(logging.StreamHandler())
        # self._logger.addHandler(logging.FileHandler(self._filepath))

        self._start_time = datetime.now()
        self.info("       Current-T        |    Elapsed-T    |    Message")

    @property
    def is_root(self) -> bool:
        return self._comm.rank == 0

    @property
    def total_elapsed_time(self) -> str:
        """
        Calculates the total elapsed time since the logger was initialized (i.e. almost the same as the runtime of the
        code).
        """
        delta = datetime.now() - self._start_time
        total_seconds = int(delta.total_seconds())
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = delta.microseconds // 1000

        return str(f"{days:02}-{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}")

    @property
    def current_time(self) -> str:
        """
        Returns the current time in the format "YYYY-MM-DD HH:MM:SS.sss".
        """
        return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))[:-3]

    def _log(self, message: str, level: int, allowed_ranks: tuple | int = (0,)):
        """
        Logs a message with a specific logging level, but only if the current rank is in the allowed ranks.
        """
        if isinstance(allowed_ranks, int):
            allowed_ranks = (allowed_ranks,)
        if self._comm.rank not in allowed_ranks or message is None or message == "" or level < 0:
            return
        self._logger.log(level, f"{self.current_time} | {self.total_elapsed_time} | {message}")

    def debug(self, message: str, allowed_ranks: tuple = (0,)):
        """
        Logs a debug message. This is intended for detailed debugging information that is not usually needed in production.
        """
        self._log("::DEBUG:: " + message, level=logging.DEBUG, allowed_ranks=allowed_ranks)

    def info(self, message: str, allowed_ranks: tuple = (0,)):
        """
        Logs an informational message. This is intended for general information about the program's execution.
        """
        self._log(message, level=logging.INFO, allowed_ranks=allowed_ranks)

    def warning(self, message: str, allowed_ranks: tuple = (0,)):
        """
        Logs a warning message. This is intended for situations that are not errors but may require attention.
        """
        self._log("::WARNING:: " + message, level=logging.WARNING, allowed_ranks=allowed_ranks)

    def log_memory_usage(self, obj_name: str, obj, n_exists: int = 1, allowed_ranks: tuple = (0,)):
        """
        Logs the memory usage of an object in gigabytes. This is useful for tracking memory consumption in distributed
        applications, especially when using MPI.
        """
        if obj is None:
            return
        self.info(f"{obj_name} use(s) (GB): {obj.memory_usage_in_gb * n_exists:.6f}", allowed_ranks=allowed_ranks)

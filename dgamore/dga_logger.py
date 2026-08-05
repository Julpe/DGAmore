# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
MPI-aware logging. :class:`DgaLogger` timestamps messages and, by default, only emits them on the root rank so
that parallel runs produce a single clean log stream.
"""

import logging
import os
from datetime import datetime

import mpi4py.MPI as MPI


class DgaLogger:
    """
    Handles logging messages in a distributed environment using MPI. Currently, it logs messages to stdout.
    """

    def __init__(self, comm: MPI.Comm, output_path: str = "./", filename: str = "dga.log"):
        """
        Sets up the underlying stream logger and records the start time for elapsed-time reporting.

        :param comm: The MPI communicator; the rank determines which processes actually emit a message.
        :param output_path: Directory for the (currently disabled) log file.
        :param filename: Name of the (currently disabled) log file.
        """
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
        """
        Whether the current process is the root rank.

        :return: True if the current MPI rank is the root rank (rank 0).
        """
        return self._comm.rank == 0

    @property
    def total_elapsed_time(self) -> str:
        """
        Calculates the total elapsed time since the logger was initialized (i.e. almost the same as the runtime of the
        code).

        :return: The elapsed time formatted as ``"DD-HH:MM:SS.mmm"``.
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
        The current wall-clock time as a formatted string.

        :return: The current wall-clock time formatted as ``"YYYY-MM-DD HH:MM:SS.sss"``.
        """
        return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))[:-3]

    def _log(self, message: str, level: int, allowed_ranks: tuple | int = (0,)):
        """
        Logs a message with a specific logging level, but only if the current rank is in the allowed ranks.

        :param message: The message to log; ignored if None or empty.
        :param level: The :mod:`logging` level (e.g. ``logging.INFO``); a negative level suppresses the message.
        :param allowed_ranks: The MPI rank(s) permitted to emit this message.
        :return: None.
        """
        if isinstance(allowed_ranks, int):
            allowed_ranks = (allowed_ranks,)
        if self._comm.rank not in allowed_ranks or message is None or message == "" or level < 0:
            return
        self._logger.log(level, f"{self.current_time} | {self.total_elapsed_time} | {message}")

    def debug(self, message: str, allowed_ranks: tuple = (0,)):
        """
        Logs a debug message. This is intended for detailed debugging information that is not usually needed in
        production.

        :param message: The message to log.
        :param allowed_ranks: The MPI rank(s) permitted to emit this message.
        :return: None.
        """
        self._log("::DEBUG:: " + message, level=logging.DEBUG, allowed_ranks=allowed_ranks)

    def info(self, message: str, allowed_ranks: tuple = (0,)):
        """
        Logs an informational message. This is intended for general information about the program's execution.

        :param message: The message to log.
        :param allowed_ranks: The MPI rank(s) permitted to emit this message.
        :return: None.
        """
        self._log(message, level=logging.INFO, allowed_ranks=allowed_ranks)

    def warning(self, message: str, allowed_ranks: tuple = (0,)):
        """
        Logs a warning message. This is intended for situations that are not errors but may require attention.

        :param message: The message to log.
        :param allowed_ranks: The MPI rank(s) permitted to emit this message.
        :return: None.
        """
        self._log("::WARNING:: " + message, level=logging.WARNING, allowed_ranks=allowed_ranks)

    def log_memory_usage(
        self, obj_name: str, obj, n_copies: int = 1, allowed_ranks: tuple = (0,), per: str = "rank", scale: float = 1.0
    ):
        """
        Logs the host memory a quantity occupies across the whole job in gigabytes, i.e. the footprint of one copy
        times the number of copies that are physically allocated, plus that breakdown whenever there is more than one.

        ``n_copies`` must count the copies that really exist: ``comm.size`` for a quantity every rank builds for
        itself or holds a distinct slice of, but only the **number of nodes** for one kept in a per-node
        shared-memory window (see :func:`dgamore.mpi_utils.build_node_shared_array`, counted by
        :func:`dgamore.mpi_utils.count_nodes`), since the ranks of a node map one and the same buffer. ``per`` labels
        what a copy belongs to in the log line.

        :param obj_name: Human-readable name used in the log line.
        :param obj: An object exposing ``memory_usage_in_gb`` (e.g. any :class:`IHaveMat`), or a list/tuple of such
            objects whose footprints are summed into one copy; ``None`` (also as an entry) is ignored.
        :param n_copies: Number of physically allocated copies of ``obj`` across the job.
        :param allowed_ranks: The MPI rank(s) permitted to emit this message.
        :param per: What a single copy belongs to, i.e. ``"rank"`` or ``"node"``; wording of the breakdown only.
        :param scale: Factor one copy's footprint is multiplied by, for a quantity reported through a stand-in
            object that holds a fraction of it (e.g. the frequency-summed auxiliary susceptibility standing in for
            the full two-fermion one, which the chunked build never materializes).
        :return: None.
        """
        objects = [o for o in (obj if isinstance(obj, (list, tuple)) else [obj]) if o is not None]
        if not objects:
            return
        per_copy = scale * sum(o.memory_usage_in_gb for o in objects)
        breakdown = "" if n_copies == 1 else f" ({n_copies} {per}s x {per_copy:.6f})"
        self.info(f"{obj_name} use(s) (GB): {per_copy * n_copies:.6f}{breakdown}.", allowed_ranks=allowed_ranks)

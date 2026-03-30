# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import h5py

import moldga.symmetrize_new as sym
from moldga.n_point_base import *


class W2dynFile:
    def __init__(self, fname: str):
        self._file = None
        self._fname = fname
        self.open()

    def __del__(self):
        """
        Destructor to ensure the file is closed properly.
        """
        self._file.close()

    def close(self):
        """
        Closes the HDF5 file.
        """
        self._file.close()

    def open(self):
        """
        Opens the HDF5 file in read mode.
        """
        self._file = h5py.File(self._fname, "r")

    def ineq_group(self, dmft_iter="dmft-last", ineq=1):
        """
        Returns the group string for a given DMFT iteration and ineq.
        """
        return dmft_iter + f"/ineq-{ineq:03}"

    def get_nd(self, ineq: int = 1) -> int:
        """
        Returns the number of d orbitals of the DMFT calculation.
        """
        return self._from_ineq_config("nd", ineq=ineq)

    def get_np(self, ineq: int = 1) -> int:
        """
        Returns the number of p orbitals of the DMFT calculation.
        """
        return self._from_ineq_config("np", ineq=ineq)

    def get_beta(self) -> float:
        r"""
        Returns the inverse temperature :math:`\beta`.
        """
        return self._file[".config"].attrs["general.beta"]

    def get_mu(self, dmft_iter: str = "dmft-last"):
        r"""
        Returns the chemical potential :math:`\mu`.
        """
        return self._file[dmft_iter + "/mu/value"][()]

    def get_totdens(self) -> float:
        """
        Returns the total particle density.
        """
        return self._file[".config"].attrs["general.totdens"]

    def get_jdd(self, ineq: int = 1) -> float:
        """
        Extracts the Hund's coupling for d orbitals.
        """
        return self._from_ineq_config("jdd", ineq=ineq)

    def get_jdp(self, ineq: int = 1) -> float:
        """
        Extracts the Hund's coupling between d and p orbitals.
        """
        return self._from_ineq_config("jdp", ineq=ineq)

    def get_jpp(self, ineq: int = 1) -> float:
        """
        Extracts the Hund's coupling for p orbitals.
        """
        return self._from_ineq_config("jpp", ineq=ineq)

    def get_jppod(self, ineq: int = 1) -> float:
        """
        Extracts the offdiagonal terms for jpp.
        """
        return self._from_ineq_config("jppod", ineq=ineq)

    def get_udd(self, ineq: int = 1) -> float:
        """
        Extracts the Hubbard U for d orbitals.
        """
        return self._from_ineq_config("udd", ineq=ineq)

    def get_udp(self, ineq: int = 1) -> float:
        """
        Extracts the Hubbard U between d and p orbitals.
        """
        return self._from_ineq_config("udp", ineq=ineq)

    def get_upp(self, ineq: int = 1) -> float:
        """
        Extracts the Hubbard U for p orbitals.
        """
        return self._from_ineq_config("upp", ineq=ineq)

    def get_uppod(self, ineq: int = 1):
        """
        Extracts the offdiagonal terms for upp.
        """
        return self._from_ineq_config("uppod", ineq=ineq)

    def get_vdd(self, ineq: int = 1) -> float:
        """
        Extracts the intersite interaction between d orbitals.
        """
        return self._from_ineq_config("vdd", ineq=ineq)

    def get_vpp(self, ineq: int = 1) -> float:
        """
        Extracts the intersite interaction between p orbitals.
        """
        return self._from_ineq_config("vpp", ineq=ineq)

    def get_siw(self, dmft_iter: str = "dmft-last", ineq: int = 1) -> list:
        """
        Extracts the DMFT self-energy in Matsubara frequency space as [band, spin, iv].
        """
        return self._file[self.ineq_group(dmft_iter=dmft_iter, ineq=ineq) + "/siw/value"][()]

    def get_giw(self, dmft_iter: str = "dmft-last", ineq: int = 1) -> list:
        """
        Extracts the DMFT Green's function in Matsubara frequency space as [band, spin, iv].
        """
        return self._file[self.ineq_group(dmft_iter=dmft_iter, ineq=ineq) + "/giw/value"][()]

    def get_occ(self, dmft_iter: str = "dmft-last", ineq: int = 1) -> list:
        """
        Extracts the occupation matrix as [band1, spin1, band2, spin2].
        """
        return self._file[self.ineq_group(dmft_iter=dmft_iter, ineq=ineq) + "/occ/value"][()]

    def get_rho1(self, dmft_iter: str = "dmft-last", ineq: int = 1) -> list:
        """
        Extracts the 1-particle density matrix as [band1, spin1, band2, spin2].
        """
        return self._file[self.ineq_group(dmft_iter=dmft_iter, ineq=ineq) + "/rho1/value"][()]

    def get_rho2(self, dmft_iter: str = "dmft-last", ineq: int = 1) -> list:
        """
        Extracts the 2-particle density matrix as [band1, spin1, band2, spin2, band3, spin3, band4, spin4].
        """
        return self._file[self.ineq_group(dmft_iter=dmft_iter, ineq=ineq) + "/rho2/value"][()]

    def _from_ineq_config(self, key: str, ineq: int = 1):
        """
        Extracts a value from the .config group for a given ineq.
        """
        return self._file[".config"].attrs[f"atoms.{ineq:1}.{key}"]

    def get_dc(self, dmft_iter: str = "dmft-last", ineq: int = 1) -> list:
        """
        Extracts the DMFT double-counting correction as [band, spin].
        """
        return self._file[self.ineq_group(dmft_iter=dmft_iter, ineq=ineq) + "/dc/value"][()]


class W2dynG4iwFile:
    def __init__(self, fname: str):
        self._fname = fname
        self._file = None
        self.open()

    def __del__(self):
        """
        Destructor to ensure the file is closed properly.
        """
        self._file.close()

    def close(self):
        """
        Closes the HDF5 file.
        """
        self._file.close()

    def open(self):
        """
        Opens the HDF5 file in read mode.
        """
        self._file = h5py.File(self._fname, "r")

    def read_g2_full_multiband(self, n_bands: int, ineq: int = 1, name: str = "dens") -> np.ndarray:
        """
        Reads the full two-particle Green's function from a w2dynamics vertex file and returns it as a numpy array.
        """
        # the next lines determine the size of g2, i.e. niw and niv
        channel_group_string = f"/ineq-{ineq:03}/{name}"
        niw_full = len(self._file[channel_group_string].keys())
        # 00000 is the first element. If it does not exist, there are no bosonic frequencies in the G2 and that would be weird
        first_index = int(next(iter(self._file[f"{channel_group_string}/00000"])))
        niv_full = len(
            self._file[f"{channel_group_string}/00000/{first_index:05}/value"][()]
        )  # extract niv from the size of the array

        g2 = np.zeros((n_bands,) * 4 + (niw_full,) + 2 * (niv_full,), dtype=np.complex128)
        for wn in range(niw_full):
            wn_group_string = f"{channel_group_string}/{wn:05}"
            for ind in self._file[wn_group_string].keys():
                bands = sym.index2component_band_4(n_bands, 4, int(ind))
                val = self._file[f"{wn_group_string}/{ind}/value"][()].T
                g2[bands[0], bands[1], bands[2], bands[3], wn, ...] = val

        return g2

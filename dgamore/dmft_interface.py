# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
from abc import ABC

import h5py
import numpy as np

import dgamore.config as config
import dgamore.symmetrize_new as sym
from dgamore.greens_function import GreensFunction
from dgamore.local_four_point import LocalFourPoint
from dgamore.n_point_base import SpinChannel
from dgamore.self_energy import SelfEnergy


class DMFTInterface(ABC):
    """
    Abstract interface for DMFT calculations. Reads the necessary quantities which are needed for a DGA calculation
    from the output files.
    """

    def get_beta(self) -> float:
        """
        Returns the inverse temperature from the DMFT calculation.
        """
        raise NotImplementedError()

    def get_mu(self) -> float:
        """
        Returns the chemical potential from the DMFT calculation.
        """
        raise NotImplementedError()

    def get_nd(self, ineq: int = 1) -> int:
        """
        Returns the number of interacting d-orbitals from DMFT.
        :param ineq: The index of the inequivalent atom.
        """
        raise NotImplementedError()

    def get_totdens(self) -> float:
        """
        Returns the total electron density from the DMFT calculation.
        """
        raise NotImplementedError()

    def get_occ(self, ineq: int = 1) -> np.ndarray:
        """
        Returns the orbital-resolved occupation from DMFT.
        :param ineq: The index of the inequivalent atom.
        """
        raise NotImplementedError()

    def get_udd(self, ineq: int = 1) -> float:
        """
        Returns the density-density interaction U from the DMFT calculation for the interacting d-orbitals.
        This is both used in simple density-density and Kanamori calculations for d-orbitals.
        :param ineq: The index of the inequivalent atom.
        """
        raise NotImplementedError()

    def get_jdd(self, ineq: int = 1) -> float:
        """
        Returns the Hund's coupling J from the DMFT calculation for the interacting d-orbitals.
        This is only supposed to be nonzero when the DMFT calculation uses a Kanamori interaction for the d-orbitals.
        :param ineq: The index of the inequivalent atom.
        """
        raise NotImplementedError()

    def get_vdd(self, ineq: int = 1) -> float:
        """
        Returns the inter-orbital repulsion V (often called U') from the DMFT calculation for the interacting
        d-orbitals.
        This is only supposed to be nonzero when the DMFT calculation uses a Kanamori interaction for the d-orbitals.
        :param ineq: The index of the inequivalent atom.
        """
        raise NotImplementedError()

    def get_dc(self, ineq: int = 1) -> float:
        """
        Returns the double-counting correction for the self-energy from DMFT.
        :param ineq: The index of the inequivalent atom.
        """
        raise NotImplementedError()

    def get_giw(self, ineq: int = 1) -> GreensFunction:
        """
        Returns the one-particle Green's function from DMFT.
        Attention: due to how the code handles the DMFT Green's function, it should be returned with an array of shape
        [nbands, nbands, 2*niv_dmft]
        :param ineq: The index of the inequivalent atom.
        """
        raise NotImplementedError()

    def get_siw(self, ineq: int = 1) -> SelfEnergy:
        """
        Returns the one-particle Self-Energy from DMFT.
        Note: This should be already updated with the double-counting correction here!
        Attention: due to how the code handles the DMFT Green's function, it should be returned with an array of shape
        [1, 1, 1, nbands, nbands, 2*niv_dmft]
        :param ineq: The index of the inequivalent atom.
        """
        raise NotImplementedError()

    def get_g2iw(self, channel: SpinChannel, ineq: int = 1) -> LocalFourPoint:
        """
        Returns the two-particle Green's function from DMFT.
        :param channel: The spin channel of the two-particle quantity (should be either density or magnetic).
        :param ineq: The index of the inequivalent atom.
        """
        raise NotImplementedError()


class W2dynInterface(DMFTInterface):
    """
    Interface for w2dynamics output files.
    """

    def __init__(self):
        self.file_1p = None
        self.file_2p = None
        self._open()

    def get_beta(self) -> float:
        """
        Returns the inverse temperature from the DMFT calculation.
        """
        return self.file_1p[".config"].attrs["general.beta"]

    def get_mu(self, dmft_iter: str = "dmft-last") -> float:
        """
        Returns the chemical potential from the DMFT calculation.
        :param dmft_iter The dmft iteration where the quantity will be taken from.
        """
        return self.file_1p[dmft_iter + "/mu/value"][()]

    def get_nd(self, ineq: int = 1) -> int:
        """
        Returns the number of interacting d-orbitals from DMFT.
        :param ineq: The index of the inequivalent atom.
        """
        return self._from_ineq_config("nd", ineq=ineq)

    def get_totdens(self, dmft_iter: str = "dmft-last") -> float:
        """
        Returns the total electron density from the DMFT calculation.
        :param dmft_iter The dmft iteration where the quantity will be taken from.
        """
        return self.file_1p[".config"].attrs["general.totdens"]

    def get_occ(self, ineq: int = 1, dmft_iter: str = "dmft-last") -> np.ndarray:
        """
        Returns the orbital-resolved occupation from DMFT.
        :param ineq: The index of the inequivalent atom.
        :param dmft_iter The dmft iteration where the quantity will be taken from.
        """
        rho1 = self.file_1p[self._ineq_group(ineq, dmft_iter) + "/rho1/value"][()]
        return 2 * np.mean(rho1, axis=(1, 3))

    def get_udd(self, ineq: int = 1) -> float:
        """
        Returns the density-density interaction U from the DMFT calculation for the interacting d-orbitals.
        This is both used in simple density-density and Kanamori calculations for d-orbitals.
        :param ineq: The index of the inequivalent atom.
        """
        return self._from_ineq_config("udd", ineq=ineq)

    def get_jdd(self, ineq: int = 1) -> float:
        """
        Returns the Hund's coupling J from the DMFT calculation for the interacting d-orbitals.
        This is only supposed to be nonzero when the DMFT calculation uses a Kanamori interaction for the d-orbitals.
        :param ineq: The index of the inequivalent atom.
        """
        return self._from_ineq_config("jdd", ineq=ineq)

    def get_vdd(self, ineq: int = 1) -> float:
        """
        Returns the inter-orbital repulsion V (often called U') from the DMFT calculation for the interacting
        d-orbitals.
        This is only supposed to be nonzero when the DMFT calculation uses a Kanamori interaction for the d-orbitals.
        :param ineq: The index of the inequivalent atom.
        """
        return self._from_ineq_config("vdd", ineq=ineq)

    def get_dc(self, ineq: int = 1, dmft_iter: str = "dmft-last") -> float:
        """
        Returns the double-counting correction for the self-energy from DMFT.
        :param ineq: The index of the inequivalent atom.
        :param dmft_iter The dmft iteration where the quantity will be taken from.
        """
        return self.file_1p[self._ineq_group(ineq, dmft_iter) + "/dc/value"][()]

    def get_giw(self, ineq: int = 1, dmft_iter: str = "dmft-last") -> GreensFunction:
        """
        Returns the one-particle Green's function from DMFT.
        Attention: due to how the code handles the DMFT Green's function, it should be returned with an array of shape
        [nbands, nbands, 2*niv_dmft]
        :param ineq: The index of the inequivalent atom.
        :param dmft_iter The dmft iteration where the quantity will be taken from.
        """
        giw = self.file_1p[self._ineq_group(ineq, dmft_iter) + "/giw/value"][()]  # [band, spin, niv]
        giw = np.mean(giw, axis=1)  # mean over spin
        return GreensFunction(self._extend_orbital(giw))

    def get_siw(self, ineq: int = 1, dmft_iter: str = "dmft-last") -> SelfEnergy:
        """
        Returns the one-particle Self-Energy from DMFT.
        Note: This should be already updated with the double-counting correction here!
        Attention: due to how the code handles the DMFT Green's function, it should be returned with an array of shape
        [1, 1, 1, nbands, nbands, 2*niv_dmft]
        :param ineq: The index of the inequivalent atom.
        :param dmft_iter The dmft iteration where the quantity will be taken from.
        """
        siw = self.file_1p[self._ineq_group(ineq, dmft_iter) + "/siw/value"][()]  # [band, spin, niv]
        siw = np.mean(siw, axis=1)  # mean over spin
        siw = self._extend_orbital(siw)[None, None, None, ...]
        siw_dc = np.mean(self.get_dc(ineq, dmft_iter), axis=-1)  # from [band, spin] to spin-mean
        siw_dc = self._extend_orbital(siw_dc)[None, None, None, ..., None]
        return SelfEnergy(siw, estimate_niv_core=True) + siw_dc

    def get_g2iw(self, channel: SpinChannel, ineq: int = 1) -> LocalFourPoint:
        """
        Returns the two-particle Green's function from DMFT.
        :param channel: The spin channel of the two-particle quantity (should be either density or magnetic).
        :param ineq: The index of the inequivalent atom.
        :raises ValueError: When entering an invalid spin channel that is not density or magnetic.
        """
        if channel not in (SpinChannel.DENS, SpinChannel.MAGN):
            raise ValueError(
                "The two-particle Green's function can only be retrieved for the density and magnetic spin channel."
            )

        # the next lines determine the size of g2, i.e. niw and niv
        channel_group_string = f"/ineq-{ineq:03}/{channel.value}"
        niw_full = len(self.file_2p[channel_group_string].keys())
        # 00000 is the first element. If it does not exist, there are no bosonic frequencies in the G2 and that would be weird
        first_index = int(next(iter(self.file_2p[f"{channel_group_string}/00000"])))
        niv_full = len(
            self.file_2p[f"{channel_group_string}/00000/{first_index:05}/value"][()]
        )  # extract niv from the size of the array

        n_bands = self.get_nd(ineq)
        g2 = np.zeros((n_bands,) * 4 + (niw_full,) + 2 * (niv_full,), dtype=np.complex128)
        for wn in range(niw_full):
            wn_group_string = f"{channel_group_string}/{wn:05}"
            for ind in self.file_2p[wn_group_string].keys():
                bands = sym.index2component_band(n_bands, 4, int(ind))
                val = self.file_2p[f"{wn_group_string}/{ind}/value"][()].T
                g2[bands[0], bands[1], bands[2], bands[3], wn, ...] = val

        return LocalFourPoint(g2, channel)

    def _extend_orbital(self, obj: np.ndarray) -> np.ndarray:
        """
        Extends the first dimension of an array to two dimensions, putting the values into the diagonal entries.
        :param obj: The array to extend.
        """
        return np.einsum("i...,ij->ij...", obj, np.eye(obj.shape[0]))

    def _ineq_group(self, ineq=1, dmft_iter="dmft-last"):
        """
        Returns the group string for a given DMFT iteration and ineq.
        :param ineq: The index of the inequivalent atom.
        :param dmft_iter The dmft iteration where the quantity will be taken from.
        """
        return dmft_iter + f"/ineq-{ineq:03}"

    def _from_ineq_config(self, key: str, ineq: int = 1):
        """
        Extracts a value with key 'key' from the .config group for a given ineq.
        """
        return self.file_1p[".config"].attrs[f"atoms.{ineq:1}.{key}"]

    def _open(self):
        """
        Opens the w2dynamics output files in read mode.
        """
        self.file_1p = h5py.File(os.path.join(config.dmft.input_path, config.dmft.fname_1p), "r")
        self.file_2p = h5py.File(os.path.join(config.dmft.input_path, config.dmft.fname_2p), "r")

    def _close(self):
        """
        Closes the w2dynamics output files.
        """
        self.file_1p.close()
        self.file_2p.close()

    def __enter__(self):
        """
        Context manager for w2dynamics output files.
        """
        self._open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager for w2dynamics output files.
        """
        self._close()

    def __del__(self):
        """
        Upon destruction, close the w2dynamics output files.
        """
        self._close()


class TriqsInterface(DMFTInterface):
    """
    Interface for TRIQS output files.
    """

    def __init__(self):
        raise NotImplementedError()


if __name__ == "__main__":
    string = "test"

    if string == "w2dyn":
        interface = W2dynInterface()
    elif string == "triqs":
        interface = TriqsInterface()
    else:
        raise ValueError("Unknown interface.")

    g = interface.get_giw()

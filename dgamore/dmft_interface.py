# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
"""
DMFT input interface. :class:`DMFTInterface` is the abstract contract for reading the quantities a DGA run needs
from a converged DMFT calculation (inverse temperature, chemical potential, fillings, interaction parameters,
the 1-particle Green's function/self-energy and the 2-particle Green's function). :class:`W2dynInterface`
implements it for w2dynamics HDF5 output (handling multiple inequivalent atoms); :class:`TriqsInterface` is a
placeholder. The 2-particle Green's function is expected in the w2dynamics frequency convention.
"""

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
    Abstract interface for DMFT calculations. Reads the quantities needed for a DGA calculation
    from the output files.
    """

    def get_beta(self) -> float:
        r"""
        Returns the inverse temperature from the DMFT calculation.

        :return: The inverse temperature :math:`\beta` from the DMFT calculation.
        :raises NotImplementedError: In the abstract base class.
        """
        raise NotImplementedError()

    def get_mu(self) -> float:
        r"""
        Returns the chemical potential from the DMFT calculation.

        :return: The chemical potential :math:`\mu` from the DMFT calculation.
        :raises NotImplementedError: In the abstract base class.
        """
        raise NotImplementedError()

    def get_nd(self, ineq: int = 1) -> int:
        """
        Returns the number of interacting d-orbitals from the DMFT calculation.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The number of interacting d-orbitals.
        :raises NotImplementedError: In the abstract base class.
        """
        raise NotImplementedError()

    def get_totdens(self) -> float:
        """
        Returns the total electron density from the DMFT calculation.

        :return: The total electron density from the DMFT calculation.
        :raises NotImplementedError: In the abstract base class.
        """
        raise NotImplementedError()

    def get_occ(self, ineq: int = 1) -> np.ndarray:
        """
        Returns the orbital-resolved occupation from the DMFT calculation.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The orbital-resolved occupation.
        :raises NotImplementedError: In the abstract base class.
        """
        raise NotImplementedError()

    def get_udd(self, ineq: int = 1) -> float:
        r"""
        Returns the density-density interaction :math:`U` for the interacting d-orbitals (used both for plain
        density-density and Kanamori interactions).

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The interaction :math:`U_{dd}`.
        :raises NotImplementedError: In the abstract base class.
        """
        raise NotImplementedError()

    def get_jdd(self, ineq: int = 1) -> float:
        r"""
        Returns the Hund's coupling :math:`J` for the interacting d-orbitals (nonzero only for Kanamori interactions).

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The Hund's coupling :math:`J_{dd}`.
        :raises NotImplementedError: In the abstract base class.
        """
        raise NotImplementedError()

    def get_vdd(self, ineq: int = 1) -> float:
        r"""
        Returns the inter-orbital repulsion :math:`V` (often :math:`U'`) for the interacting d-orbitals (nonzero only
        for Kanamori interactions).

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The inter-orbital repulsion :math:`V_{dd}`.
        :raises NotImplementedError: In the abstract base class.
        """
        raise NotImplementedError()

    def get_dc(self, ineq: int = 1) -> float:
        """
        Returns the double-counting correction for the self-energy from the DMFT calculation.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The double-counting correction for the self-energy.
        :raises NotImplementedError: In the abstract base class.
        """
        raise NotImplementedError()

    def get_giw(self, ineq: int = 1) -> GreensFunction:
        """
        Returns the one-particle Green's function from DMFT. It must be returned with an array of shape
        ``[nbands, nbands, 2*niv_dmft]``.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The local DMFT :class:`GreensFunction`.
        :raises NotImplementedError: In the abstract base class.
        """
        raise NotImplementedError()

    def get_siw(self, ineq: int = 1) -> SelfEnergy:
        """
        Returns the one-particle self-energy from DMFT, already including the double-counting correction. It must be
        returned with an array of shape ``[1, 1, 1, nbands, nbands, 2*niv_dmft]``.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The local DMFT :class:`SelfEnergy`.
        :raises NotImplementedError: In the abstract base class.
        """
        raise NotImplementedError()

    def get_g2iw(self, channel: SpinChannel, ineq: int = 1) -> LocalFourPoint:
        r"""
        Returns the two-particle Green's function from DMFT. The input must follow the w2dynamics frequency
        convention: :math:`\nu_1 = \nu`, :math:`\nu_2 = \nu-\omega`, :math:`\nu_3 = \nu'-\omega`, :math:`\nu_4 = \nu'`.

        :param channel: The spin channel of the two-particle quantity (density or magnetic).
        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The two-particle Green's function as a :class:`LocalFourPoint`.
        :raises NotImplementedError: In the abstract base class.
        """
        raise NotImplementedError()


class W2dynInterface(DMFTInterface):
    """
    Interface for w2dynamics output files.
    """

    def __init__(self):
        """
        Opens the w2dynamics 1- and 2-particle HDF5 files (from :mod:`dgamore.config`).
        """
        self.file_1p = None
        self.file_2p = None
        self._open()

    def get_beta(self) -> float:
        r"""
        Reads the inverse temperature from the w2dynamics config.

        :return: The inverse temperature :math:`\beta` from the DMFT calculation.
        """
        return self.file_1p[".config"].attrs["general.beta"]

    def get_mu(self, dmft_iter: str = "dmft-last") -> float:
        r"""
        Reads the chemical potential from the w2dynamics output.

        :param dmft_iter: The DMFT iteration to read from.
        :return: The chemical potential :math:`\mu`.
        """
        return self.file_1p[dmft_iter + "/mu/value"][()]

    def get_nd(self, ineq: int = 1) -> int:
        """
        Reads the number of interacting d-orbitals from the w2dynamics config.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The number of interacting d-orbitals.
        """
        return self._from_ineq_config("nd", ineq=ineq)

    def get_totdens(self, dmft_iter: str = "dmft-last") -> float:
        """
        Reads the total electron density from the w2dynamics config.

        :param dmft_iter: The DMFT iteration to read from.
        :return: The total electron density.
        """
        return self.file_1p[".config"].attrs["general.totdens"]

    def get_occ(self, ineq: int = 1, dmft_iter: str = "dmft-last") -> np.ndarray:
        """
        Reads and spin-sums the orbital-resolved occupation from the w2dynamics output.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :param dmft_iter: The DMFT iteration to read from.
        :return: The orbital-resolved occupation (spin-summed).
        """
        rho1 = self.file_1p[self._ineq_group(ineq, dmft_iter) + "/rho1/value"][()]
        return 2 * np.mean(rho1, axis=(1, 3))

    def get_udd(self, ineq: int = 1) -> float:
        r"""
        Reads the density-density interaction from the w2dynamics config.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The density-density interaction :math:`U_{dd}` (used for plain and Kanamori interactions).
        """
        return self._from_ineq_config("udd", ineq=ineq)

    def get_jdd(self, ineq: int = 1) -> float:
        r"""
        Reads the Hund's coupling from the w2dynamics config.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The Hund's coupling :math:`J_{dd}` (nonzero only for Kanamori interactions).
        """
        return self._from_ineq_config("jdd", ineq=ineq)

    def get_vdd(self, ineq: int = 1) -> float:
        r"""
        Reads the inter-orbital repulsion from the w2dynamics config.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The inter-orbital repulsion :math:`V_{dd}` (often :math:`U'`; nonzero only for Kanamori interactions).
        """
        return self._from_ineq_config("vdd", ineq=ineq)

    def get_dc(self, ineq: int = 1, dmft_iter: str = "dmft-last") -> float:
        """
        Reads the double-counting correction from the w2dynamics output.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :param dmft_iter: The DMFT iteration to read from.
        :return: The double-counting correction for the self-energy.
        """
        return self.file_1p[self._ineq_group(ineq, dmft_iter) + "/dc/value"][()]

    def get_giw(self, ineq: int = 1, dmft_iter: str = "dmft-last") -> GreensFunction:
        """
        Returns the spin-averaged one-particle Green's function from DMFT, extended to a diagonal orbital matrix of
        shape ``[nbands, nbands, 2*niv_dmft]``.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :param dmft_iter: The DMFT iteration to read from.
        :return: The local DMFT :class:`GreensFunction`.
        """
        giw = self.file_1p[self._ineq_group(ineq, dmft_iter) + "/giw/value"][()]  # [band, spin, niv]
        giw = np.mean(giw, axis=1)  # mean over spin
        return GreensFunction(self._extend_orbital(giw), nk=config.lattice.nk, beta=config.sys.beta)

    def get_siw(self, ineq: int = 1, dmft_iter: str = "dmft-last") -> SelfEnergy:
        """
        Returns the spin-averaged one-particle self-energy from DMFT, with the double-counting correction added,
        as an array of shape ``[1, 1, 1, nbands, nbands, 2*niv_dmft]``.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :param dmft_iter: The DMFT iteration to read from.
        :return: The local DMFT :class:`SelfEnergy` including the double-counting correction.
        """
        siw = self.file_1p[self._ineq_group(ineq, dmft_iter) + "/siw/value"][()]  # [band, spin, niv]
        siw = np.mean(siw, axis=1)  # mean over spin
        siw = self._extend_orbital(siw)[None, None, None, ...]
        siw_dc = np.mean(self.get_dc(ineq, dmft_iter), axis=-1)  # from [band, spin] to spin-mean
        siw_dc = self._extend_orbital(siw_dc)[None, None, None, ..., None]
        return SelfEnergy(siw, estimate_niv_core=True, beta=config.sys.beta) + siw_dc

    def get_g2iw(self, channel: SpinChannel, ineq: int = 1) -> LocalFourPoint:
        """
        Reads the two-particle Green's function from the w2dynamics 2-particle file, assembling the orbital/frequency
        array from the per-bosonic-frequency, per-orbital-component groups.

        :param channel: The spin channel of the two-particle quantity (density or magnetic).
        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The two-particle Green's function as a :class:`LocalFourPoint`.
        :raises ValueError: If ``channel`` is neither density nor magnetic.
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
        Promotes a band-resolved array to a diagonal orbital matrix (values placed on the orbital diagonal).

        :param obj: The array whose leading axis is the band index.
        :return: The array with the band axis expanded to a diagonal ``[i, j, ...]`` orbital pair.
        """
        return np.einsum("i...,ij->ij...", obj, np.eye(obj.shape[0]))

    def _ineq_group(self, ineq: int = 1, dmft_iter: str = "dmft-last") -> str:
        """
        Builds the HDF5 group path for a given DMFT iteration and inequivalent atom.

        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :param dmft_iter: The DMFT iteration to read from.
        :return: The HDF5 group path string.
        """
        return dmft_iter + f"/ineq-{ineq:03}"

    def _from_ineq_config(self, key: str, ineq: int = 1):
        """
        Reads an attribute from the ``.config`` group for a given inequivalent atom.

        :param key: The atom-config attribute name (without the ``atoms.<ineq>.`` prefix).
        :param ineq: The index of the inequivalent atom (for multi-site DMFT).
        :return: The attribute value.
        """
        return self.file_1p[".config"].attrs[f"atoms.{ineq:1}.{key}"]

    def _open(self) -> None:
        """
        Opens the w2dynamics 1- and 2-particle output files in read mode.

        :return: None.
        """
        self.file_1p = h5py.File(os.path.join(config.dmft.input_path, config.dmft.fname_1p), "r")
        self.file_2p = h5py.File(os.path.join(config.dmft.input_path, config.dmft.fname_2p), "r")

    def _close(self) -> None:
        """
        Closes the w2dynamics output files.

        :return: None.
        """
        self.file_1p.close()
        self.file_2p.close()

    def __enter__(self):
        """
        Context-manager entry; opens the w2dynamics output files (see :meth:`_open`).
        """
        self._open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context-manager exit; closes the w2dynamics output files (see :meth:`_close`).

        :param exc_type: Exception type if one was raised in the ``with`` block, else None.
        :param exc_val: Exception instance if one was raised, else None.
        :param exc_tb: Traceback if an exception was raised, else None.
        """
        self._close()

    def __del__(self):
        """
        Destructor; closes the w2dynamics output files (see :meth:`_close`).
        """
        self._close()


class TriqsInterface(DMFTInterface):
    """
    Interface for TRIQS output files.
    """

    def __init__(self):
        """
        Placeholder constructor for the (not yet implemented) TRIQS interface.

        :raises NotImplementedError: The TRIQS interface is not yet implemented.
        """
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

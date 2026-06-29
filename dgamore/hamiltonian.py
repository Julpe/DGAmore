# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems
r"""
Lattice model setup. :class:`Hamiltonian` assembles the (multi-orbital) Hubbard model: the real-space hopping
:math:`t_{ab}(R)`, the local interaction :math:`U_{abcd}` and the non-local interaction :math:`V_{abcd}(R)`. It
offers convenience builders (single-band, orbital-diagonal, Kanamori, 2D :math:`t`-:math:`t'`-:math:`t''`),
readers/writers for wannier90 / wien2k ``hr``/``hk`` and ``umatrix`` files, and Fourier transforms to obtain the
band dispersion :math:`\varepsilon_{ab}(k)` and momentum-dependent interaction :math:`V_{abcd}(q)`.
:class:`HoppingElement` and :class:`InteractionElement` are small validated containers for single hopping/
interaction entries.
"""

import itertools as it
import logging

import numpy as np
import pandas as pd

import dgamore.brillouin_zone as bz
from dgamore.interaction import LocalInteraction, Interaction
from dgamore.n_point_base import SpinChannel


class HoppingElement:
    """
    Data class to store a hopping element of the Hamiltonian. The hopping element is defined by the relative lattice
    vector ``r_lat``, the orbitals ``orbs`` and the value of the hopping ``value``. A hopping element represents a
    single line in a w2k file. Added this to make the code more readable and to avoid passing around lists of values.
    """

    def __init__(self, r_lat: list, orbs: list, value: float = 0.0):
        """
        Validates and stores a single hopping element.

        :param r_lat: The relative lattice vector as a list of 3 integers.
        :param orbs: The two (1-based) orbital indices ``[o1, o2]`` of the hopping.
        :param value: The hopping amplitude.
        :raises ValueError: If ``r_lat`` is not 3 integers, ``orbs`` is not 2 positive integers, or ``value`` is not
            a number.
        """
        if not (isinstance(r_lat, list) and len(r_lat) == 3 and all(isinstance(x, int) for x in r_lat)):
            raise ValueError("'r_lat' must be a list with exactly 3 integer elements.")
        if not (
            isinstance(orbs, list)
            and len(orbs) == 2
            and all(isinstance(x, int) for x in orbs)
            and all(orb > 0 for orb in orbs)
        ):
            raise ValueError("'orbs' must be a list with exactly 2 integer elements that are greater than 0.")
        if not isinstance(value, (int, float)):
            raise ValueError("'value' must be a valid number.")

        self.r_lat = tuple(r_lat)
        self.orbs = np.array(orbs, dtype=int)
        self.value = float(value)


class InteractionElement:
    """
    Data class to store an interaction element of the Hamiltonian. The interaction element is defined by the relative
    lattice vector ``r_lat``, the orbitals ``orbs`` and the value of the interaction ``value``. An interaction element
    represents a single entry in the interaction matrix. Added this to make the code more readable and to avoid passing
    around lists of values.
    """

    def __init__(self, r_lat: list[int], orbs: list[int], value: float):
        """
        Validates and stores a single interaction element.

        :param r_lat: The relative lattice vector as a list of 3 integers.
        :param orbs: The four (1-based) orbital indices ``[o1, o2, o3, o4]`` of the interaction.
        :param value: The interaction value.
        :raises ValueError: If ``r_lat`` is not 3 integers, ``orbs`` is not 4 positive integers, or ``value`` is not
            a number.
        """
        if not (isinstance(r_lat, list) and len(r_lat) == 3 and all(isinstance(x, int) for x in r_lat)):
            raise ValueError("'r_lat' must be a list with exactly 3 integer elements.")
        if not (
            isinstance(orbs, list)
            and len(orbs) == 4
            and all(isinstance(x, int) for x in orbs)
            and all(orb > 0 for orb in orbs)
        ):
            raise ValueError("'orbs' must be a list with exactly 4 integer elements that are greater than zero.")
        if not isinstance(value, (int, float)):
            raise ValueError("'value' must be a real number.")

        self.r_lat = tuple(r_lat)
        self.orbs = np.array(orbs, dtype=int)
        self.value = float(value)


class Hamiltonian:
    """
    Class to handle the Hamiltonian of the Hubbard model. Contains the hopping terms (``er``) and the local
    (``local_interaction``) and nonlocal interaction (``nonlocal_interaction``) terms. Has a few helper methods to
    simplify adding terms to the Hamiltonian.
    """

    def __init__(self):
        """
        Initializes an empty Hamiltonian; hopping and interaction terms are added afterwards via the builder/reader
        methods.
        """
        self._er = None
        self._ek = None
        self._er_r_grid = None
        self._er_orbs = None
        self._er_r_weights = None

        self._ur_local = None
        self._ur_nonlocal = None
        self._ur_r_grid = None
        self._ur_orbs = None
        self._ur_r_weights = None

        self._local_interaction = None
        self._nonlocal_interaction = None

    def single_band_interaction(self, u: float) -> "Hamiltonian":
        r"""
        Sets the local interaction for a single-band model from a single Hubbard :math:`U`.

        :param u: The Hubbard interaction :math:`U`.
        :return: ``self`` (for chaining).
        """
        return self.interaction_orbital_diagonal(u, 1)

    def interaction_orbital_diagonal(self, u: float, n_bands: int = 1) -> "Hamiltonian":
        r"""
        Sets a purely orbital-diagonal local interaction for a multi-band model: the interaction tensor is zero
        everywhere except the ``[0,0,0,0]`` element, which is set to ``u``.

        :param u: The Hubbard interaction :math:`U` on the first orbital.
        :param n_bands: Number of orbitals/bands.
        :return: ``self`` (for chaining).
        """
        interaction_elements = []
        for i, j, k, l in it.product(range(n_bands), repeat=4):
            if i == j == k == l == 0:
                interaction_elements.append(InteractionElement([0, 0, 0], [i + 1, j + 1, k + 1, l + 1], u))
            else:
                interaction_elements.append(InteractionElement([0, 0, 0], [i + 1, j + 1, k + 1, l + 1], 0))

        return self._add_interaction_term(interaction_elements)

    def kanamori_interaction_d(self, n_bands: int, udd: float, jdd: float, vdd: float = None) -> "Hamiltonian":
        r"""
        Adds the Kanamori interaction terms ONLY for d orbitals to the Hamiltonian.
        The interaction terms are defined by the Hubbard ``udd`` (U),
        the exchange interaction ``jdd`` (J) and the inter-orbital density-density interaction ``vdd`` (V or sometimes U').
        ``vdd`` is an optional parameter, if left empty, it is set to V=U-2J.

        :param n_bands: Number of d orbitals/bands.
        :param udd: The intra-orbital Hubbard interaction :math:`U_{dd}`.
        :param jdd: The Hund's exchange :math:`J_{dd}`.
        :param vdd: The inter-orbital interaction :math:`V_{dd}`; defaults to :math:`U_{dd} - 2 J_{dd}` if None.
        :return: ``self`` (for chaining).
        """
        return self.kanamori_interaction_dp(nd_bands=n_bands, udd=udd, jdd=jdd, vdd=vdd)

    def kanamori_interaction_p(self, n_bands: int, upp: float, jpp: float, vpp: float = None) -> "Hamiltonian":
        r"""
        Adds the Kanamori interaction terms ONLY for p orbitals to the Hamiltonian.
        The interaction terms are defined by the Hubbard ``upp`` (U),
        the exchange interaction ``jpp`` (J) and the inter-orbital density-density interaction ``vpp`` (V or sometimes U').
        ``vpp`` is an optional parameter, if left empty, it is set to V=U-2J.

        :param n_bands: Number of p orbitals/bands.
        :param upp: The intra-orbital Hubbard interaction :math:`U_{pp}`.
        :param jpp: The Hund's exchange :math:`J_{pp}`.
        :param vpp: The inter-orbital interaction :math:`V_{pp}`; defaults to :math:`U_{pp} - 2 J_{pp}` if None.
        :return: ``self`` (for chaining).
        """
        return self.kanamori_interaction_dp(np_bands=n_bands, upp=upp, jpp=jpp, vpp=vpp)

    def kanamori_interaction_dp(
        self,
        nd_bands: int = 0,
        np_bands: int = 0,
        udd: float = 0.0,
        upp: float = 0.0,
        udp: float = 0.0,
        jdd: float = 0.0,
        jpp: float = 0.0,
        jdp: float = 0.0,
        vdd: float = None,
        vpp: float = None,
    ) -> "Hamiltonian":
        r"""
        Adds the full Kanamori interaction terms for d and p orbitals to the Hamiltonian.
        The interaction terms are defined by the local interaction Hubbard U,
        the exchange interaction J and the inter-orbital density-density interaction V or sometimes U'.
        vdd (vpp) (vdp) are optional parameters, if left empty, they are set to V=U-2J.

        :param nd_bands: Number of d orbitals (placed first in the orbital ordering).
        :param np_bands: Number of p orbitals (placed after the d orbitals).
        :param udd: Intra-orbital Hubbard :math:`U_{dd}`.
        :param upp: Intra-orbital Hubbard :math:`U_{pp}`.
        :param udp: The d-p inter-orbital interaction (used as :math:`V_{dp}`).
        :param jdd: Hund's exchange :math:`J_{dd}`.
        :param jpp: Hund's exchange :math:`J_{pp}`.
        :param jdp: The d-p exchange :math:`J_{dp}`.
        :param vdd: Inter-orbital :math:`V_{dd}`; defaults to :math:`U_{dd} - 2 J_{dd}` if None.
        :param vpp: Inter-orbital :math:`V_{pp}`; defaults to :math:`U_{pp} - 2 J_{pp}` if None.
        :return: ``self`` (for chaining).
        """

        # Default V=U-2J
        if vdd is None:
            vdd = udd - 2 * jdd
        if vpp is None:
            vpp = upp - 2 * jpp

        def orb_type(i: int) -> str:
            """
            Classifies an orbital as a d- or p-orbital by its index.

            :param i: 0-based orbital index.
            :return: ``"d"`` if the orbital is among the first ``nd_bands`` (d orbitals), else ``"p"``.
            """
            return "d" if i < nd_bands else "p"  # d bands come first

        def get_params(o1: int, o2: int) -> tuple[float, float, float]:
            """
            Returns the ``(U, J, V)`` Kanamori parameters for an orbital pair according to their d/p types.

            :param o1: 0-based index of the first orbital.
            :param o2: 0-based index of the second orbital.
            :return: The ``(U, J, V)`` tuple for that orbital combination.
            """
            t1, t2 = orb_type(o1), orb_type(o2)

            if t1 == "d" and t2 == "d":
                return udd, jdd, vdd
            elif t1 == "p" and t2 == "p":
                return upp, jpp, vpp
            else:
                return 0, jdp, udp  # udp is used as "Vdp", since there is no Udp possible that fits U_llll

        r_loc = [0, 0, 0]
        n_tot = nd_bands + np_bands

        interaction_elements = []

        for a, b, c, d in it.product(range(n_tot), repeat=4):
            bands = [a + 1, b + 1, c + 1, d + 1]

            # choose parameters based on (a,b) pair (equivalently (c,d))
            u, j, v = get_params(a, b)

            if a == b == c == d:  # U_{llll}
                interaction_elements.append(InteractionElement(r_loc, bands, u))
            elif (a == d and b == c) or (a == b and c == d):  # U_{lmml}, U_{llmm}
                interaction_elements.append(InteractionElement(r_loc, bands, j))
            elif a == c and b == d:  # U_{lmlm}
                interaction_elements.append(InteractionElement(r_loc, bands, v))

        return self._add_interaction_term(interaction_elements)

    def kinetic_one_band_2d_t_tp_tpp(self, t: float, tp: float, tpp: float) -> "Hamiltonian":
        r"""
        Adds the kinetic terms for a one-band model in 2D with nearest (t), next-nearest (tp) and next-next-nearest
        (tpp) neighbor hopping.

        :param t: Nearest-neighbor hopping :math:`t`.
        :param tp: Next-nearest-neighbor hopping :math:`t'`.
        :param tpp: Next-next-nearest-neighbor hopping :math:`t''`.
        :return: ``self`` (for chaining).
        """
        orbs = [1, 1]
        hopping_elements = [
            HoppingElement(r_lat=[1, 0, 0], orbs=orbs, value=-t),
            HoppingElement(r_lat=[0, 1, 0], orbs=orbs, value=-t),
            HoppingElement(r_lat=[-1, 0, 0], orbs=orbs, value=-t),
            HoppingElement(r_lat=[0, -1, 0], orbs=orbs, value=-t),
            HoppingElement(r_lat=[1, 1, 0], orbs=orbs, value=-tp),
            HoppingElement(r_lat=[1, -1, 0], orbs=orbs, value=-tp),
            HoppingElement(r_lat=[-1, 1, 0], orbs=orbs, value=-tp),
            HoppingElement(r_lat=[-1, -1, 0], orbs=orbs, value=-tp),
            HoppingElement(r_lat=[2, 0, 0], orbs=orbs, value=-tpp),
            HoppingElement(r_lat=[0, 2, 0], orbs=orbs, value=-tpp),
            HoppingElement(r_lat=[-2, 0, 0], orbs=orbs, value=-tpp),
            HoppingElement(r_lat=[0, -2, 0], orbs=orbs, value=-tpp),
        ]

        return self._add_kinetic_term(hopping_elements)

    def read_hr_w2k(self, filename: str = "./wannier_hr.dat") -> "Hamiltonian":
        """
        Reads the ``wannier_hr.dat`` file from a wien2k hr file and sets the real-space kinetic Hamiltonian.
        This is then typically Fourier-transformed to momentum space to obtain the k-dependent band dispersion.

        :param filename: Path to the ``wannier_hr.dat`` file.
        :return: ``self`` (for chaining).
        """
        er_file = pd.read_csv(filename, skiprows=1, names=np.arange(15), sep=r"\s+", dtype=float, engine="python")
        n_bands = er_file.values[0][0].astype(int)
        nr = er_file.values[1][0].astype(int)

        tmp = np.reshape(er_file.values, (np.size(er_file.values), 1))
        tmp = tmp[~np.isnan(tmp)]

        self._er_r_weights = tmp[2 : 2 + nr].astype(int)
        self._er_r_weights = np.reshape(self._er_r_weights, (np.size(self._er_r_weights), 1))
        ns = 7
        n_tmp = np.size(tmp[2 + nr :]) // ns
        tmp = np.reshape(tmp[2 + nr :], (n_tmp, ns))

        self._er_r_grid = np.reshape(tmp[:, 0:3], (nr, n_bands, n_bands, 3))
        self._er_orbs = np.reshape(tmp[:, 3:5], (nr, n_bands, n_bands, 2))
        self._er = np.reshape(tmp[:, 5] + 1j * tmp[:, 6], (nr, n_bands, n_bands))
        return self

    def read_hk_w2k(self, fname: str, spin_sym: bool = True):
        r"""
        Reads a Hamiltonian :math:`H_{bb'}(k)` from a text file.

        Expects a text file with white-space separated values in the syntax as
        generated by wannier90:  the first line is a header with three integers,
        optionally followed by '#'-prefixed comment:

            <no of k-points> <no of wannier functions> <no of bands (ignored)>

        For each k-point, there is a header line with the x, y, z coordinates of the
        k-point, followed by <nbands> rows as lines of 2*<nbands> values each, which
        are the real and imaginary part of each column.

        :param fname: Path to the ``hk`` text file.
        :param spin_sym: If True, assume spin symmetry (no spin-orbit); if False, the file carries a doubled
            spin structure that is folded back to ``nbands``.
        :return: A pair ``(ham, kpoints)`` where ``ham`` is a new :class:`Hamiltonian` with ``ek`` set to the
            complex array :math:`H(k)` (k-point first, then band indices) and ``kpoints`` is the real array of
            k-point ``(x, y, z)`` components (transposed).
        :raises RuntimeError: If a spin-orbit Hamiltonian has an odd band count.
        :raises ValueError: If the file contains fewer k-points than declared in the header.
        """
        logger = logging.getLogger()
        hk_file = open(fname, "r", encoding="utf-8")
        spin_orbit = not spin_sym

        def nextline():
            """
            Reads the next file line, stripping any ``#`` comment, and splits it into whitespace-separated tokens.

            :return: The list of tokens on the line (empty if the line is blank/comment-only).
            """
            line = hk_file.readline()
            return line[: line.find("#")].split()

        # parse header
        header = nextline()
        if header[0] == "VERSION":
            logger.warning("Version 2 headers are obsolete (specify in input file!)")
            nkpoints, natoms = map(int, nextline())
            lines = np.array([nextline() for _ in range(natoms)], np.int_)
            nbands = np.sum(lines[:, :2])
            del lines, natoms
        elif len(header) != 3:
            logger.warning("Version 1 headers are obsolete (specify in input file!)")
            header = list(map(int, header))
            nkpoints = header[0]
            nbands = header[1] * (header[2] + header[3])
        else:
            nkpoints, nbands, _ = map(int, header)
        del header

        # nspins is the spin dimension for H(k); G(iw), Sigma(iw) etc. will always
        # be spin-dependent
        if spin_orbit:
            if nbands % 2:
                raise RuntimeError("Spin-structure of Hamiltonian!")
            nbands //= 2
            nspins = 2
        else:
            nspins = 1
        # GS: inside read_hamiltonian nspins is therefore equal to 1 if spin_orbit=0
        # GS: outside nspins is however always set to 2. Right?
        # GS: this also means that we have to set nspins internally in read_ImHyb too

        hk_file.flush()

        # parse data
        hk = np.fromfile(hk_file, sep=" ")
        hk = hk.reshape(-1, 3 + 2 * nbands**2 * nspins**2)
        kpoints_file = hk.shape[0]
        if kpoints_file > nkpoints:
            logger.warning("truncating Hk points")
        elif kpoints_file < nkpoints:
            raise ValueError(f"problem! {kpoints_file} < {nkpoints}")
        kpoints = hk[:nkpoints, :3]

        hk = hk[:nkpoints, 3:].reshape(nkpoints, nspins, nbands, nspins, nbands, 2)
        hk = hk[..., 0] + 1j * hk[..., 1]
        if not np.allclose(hk, hk.transpose(0, 3, 4, 1, 2).conj()):
            logger.warning("Hermiticity violation detected in Hk file")

        # go from wannier90/convert_Hamiltonian structure to our Green's function
        # convention
        hk = hk.transpose(0, 2, 1, 4, 3)
        hk_file.close()

        hk = np.squeeze(hk)
        ham = Hamiltonian()
        ham._ek = hk

        return ham, kpoints.T

    def write_hr_w2k(self, filename: str) -> None:
        """
        Writes the real-space kinetic Hamiltonian to a file in the wien2k ``hr`` format.

        :param filename: Output file path.
        :return: None.
        """
        n_columns = 15
        n_r = self._er.shape[0]
        n_bands = self._er.shape[-1]
        file = open(filename, "w", encoding="utf-8")
        file.write("# Written using the wannier module of the dga code\n")
        file.write(f"{n_bands} \n")
        file.write(f"{n_r} \n")

        for i in range(0, len(self._er_r_weights), n_columns):
            line = "    ".join(map(str, self._er_r_weights[i : i + n_columns, 0]))
            file.write("    " + line + "\n")
        hr = self._er.reshape(n_r * n_bands**2, 1)
        r_grid = self._er_r_grid.reshape(n_r * n_bands**2, 3).astype(int)
        orbs = self._er_orbs.reshape(n_r * n_bands**2, 2).astype(int)

        for i in range(0, n_r * n_bands**2):
            line = "{: 5d}{: 5d}{: 5d}{: 5d}{: 5d}{: 12.6f}{: 12.6f}".format(
                *r_grid[i, :], *orbs[i, :], hr[i, 0].real, hr[i, 0].imag
            )
            file.write(line + "\n")

    def write_hk_w2k(self, filename: str, k_grid: bz.KGrid, ek: np.ndarray = None) -> None:
        """
        Writes the k-space kinetic Hamiltonian to a file in the wien2k ``hk`` format.

        :param filename: Output file path.
        :param k_grid: The :class:`KGrid` defining the k-points to write.
        :param ek: Optional band dispersion to write; computed from the Hamiltonian on ``k_grid`` if None.
        :return: None.
        """
        if ek is None:
            ek = self.get_ek(k_grid)

        f = open(filename, "w", encoding="utf-8")
        n_bands = ek.shape[-1]

        ek = ek.reshape(k_grid.nk_tot, n_bands, n_bands)
        kmesh = k_grid.kmesh_list

        print(k_grid.nk_tot, n_bands, n_bands, file=f)
        for ik in range(k_grid.nk_tot):
            print(kmesh[0, ik], kmesh[1, ik], kmesh[2, ik], file=f)
            ek_slice = np.copy(ek[ik, ...])
            np.savetxt(
                f, ek_slice.view(float), fmt="%.12f", delimiter=" ", newline="\n", header="", footer="", comments="#"
            )

        f.close()

    def read_umatrix(self, filename: str) -> "Hamiltonian":
        """
        Reads a file and creates the interaction matrix from it. The file should contain the number of bands in the
        first line and the number of r values in the second line. From the third line onwards it should contain the
        interaction matrix entries. It looks very similar to the format of a wannier_hr.dat file. The format is:
        r_lat_x r_lat_y r_lat_z orb1 orb2 orb3 orb4 realvalue imagvalue, where r_lat is the relative lattice vector
        and orb1-4 are the orbital indices. The interaction is assumed to be purely real. The ordering of the entries
        themselves does not matter. Note: The file must not contain any comments or empty lines.

        :param filename: Path to the umatrix file.
        :return: ``self`` (for chaining).
        """
        umatrix_file = pd.read_csv(filename, skiprows=0, names=np.arange(15), sep=r"\s+", dtype=float, engine="python")
        values = umatrix_file.values

        nr = values[1][0].astype(int)
        values = values[~np.isnan(values)]

        n_cols = 9
        n_rows = np.size(values[2 + nr :]) // n_cols
        values = np.reshape(values[2 + nr :], (n_rows, n_cols))

        interaction_elements = []
        for i in range(len(values)):
            interaction_elements.append(
                InteractionElement(
                    r_lat=values[i, 0:3].astype(int).tolist(),
                    orbs=values[i, 3:7].astype(int).tolist(),
                    value=values[i, 7].astype(float),
                )
            )

        return self._add_interaction_term(interaction_elements)

    def get_ek(self, k_grid: bz.KGrid = None) -> np.ndarray:
        r"""
        Returns the band dispersion :math:`\varepsilon_{ab}(k)`, computing and caching it from the real-space hopping
        on ``k_grid`` if not already set.

        :param k_grid: The :class:`KGrid` to evaluate on (ignored if the dispersion is already cached/set).
        :return: The band dispersion array of shape ``[kx, ky, kz, o1, o2]``.
        """
        if self._ek is not None:
            return self._ek
        self._ek = self._convham_2_orbs(k_grid.kmesh.reshape(3, -1))
        n_bands = self._ek.shape[-1]
        self._ek = self._ek.reshape(*k_grid.nk, n_bands, n_bands)
        return self._ek

    def set_ek(self, ek: np.ndarray) -> "Hamiltonian":
        """
        Sets the band dispersion directly (bypassing the Fourier transform).

        :param ek: The band dispersion array.
        :return: ``self`` (for chaining).
        """
        self._ek = ek
        return self

    def get_local_u(self) -> LocalInteraction:
        """
        Returns the local interaction. Since the local interaction is momentum-independent, its momentum-space and
        real-space representations coincide.

        :return: The local interaction as a :class:`LocalInteraction`.
        """
        return LocalInteraction(self._ur_local)

    def get_vq(self, q_grid: bz.KGrid) -> "Interaction":
        r"""
        Returns the momentum-dependent non-local interaction :math:`V_{abcd}(q)` by Fourier-transforming the
        real-space interaction onto ``q_grid``.

        :param q_grid: The :class:`KGrid` defining the momentum grid.
        :return: The non-local interaction as an :class:`Interaction`.
        """
        uq = self._convham_4_orbs(q_grid.kmesh.reshape(3, -1))
        n_bands = uq.shape[-1]
        return Interaction(uq.reshape(q_grid.nk + (n_bands,) * 4), SpinChannel.NONE, q_grid.nk)

    def _add_kinetic_term(self, hopping_elements: list) -> "Hamiltonian":
        """
        Builds the real-space kinetic term (grid, orbitals, weights and ``er`` matrix) from a list of hopping
        elements.

        :param hopping_elements: List of :class:`HoppingElement` (or dicts convertible to one).
        :return: ``self`` (for chaining).
        :raises ValueError: If any hopping element is local (``r_lat == [0, 0, 0]``).
        """
        hopping_elements = self._parse_elements(hopping_elements, HoppingElement)

        if any(np.allclose(el.r_lat, [0, 0, 0]) for el in hopping_elements):
            raise ValueError("Local hopping is not allowed!")

        r_to_index, n_rp, n_orbs = self._prepare_lattice_indices_and_orbs(hopping_elements)

        self._er_r_grid = self._create_er_grid(r_to_index, n_orbs)
        self._er_orbs = self._create_er_orbs(n_rp, n_orbs)
        self._er_r_weights = np.ones(n_rp)[:, None]

        self._er = np.zeros((n_rp, n_orbs, n_orbs))
        for he in hopping_elements:
            self._insert_er_element(self._er, r_to_index, he.r_lat, *he.orbs, he.value)
        return self

    def _add_interaction_term(self, interaction_elements: list) -> "Hamiltonian":
        """
        Builds the local and non-local interaction terms from a list of interaction elements (the local part is
        stored separately, so the ``[0,0,0]`` lattice vector is removed from the non-local grid).

        :param interaction_elements: List of :class:`InteractionElement` (or dicts convertible to one).
        :return: ``self`` (for chaining).
        """
        interaction_elements = self._parse_elements(interaction_elements, InteractionElement)
        r_to_index, n_rp, n_orbs = self._prepare_lattice_indices_and_orbs(interaction_elements)
        # we need local interactions in a separate object, hence we do not care about the [0,0,0] r_lat in here
        n_rp -= 1
        r_to_index.pop((0, 0, 0))

        self._ur_r_grid = self._create_ur_grid(r_to_index, n_orbs)
        self._ur_orbs = self._create_ur_orbs(n_rp, n_orbs)
        self._ur_r_weights = np.ones(n_rp)[:, None]

        self._ur_local = np.zeros((n_orbs, n_orbs, n_orbs, n_orbs))
        self._ur_nonlocal = np.zeros((n_rp, n_orbs, n_orbs, n_orbs, n_orbs))
        for ie in interaction_elements:
            if np.allclose(ie.r_lat, [0, 0, 0]):
                self._insert_ur_element(self._ur_local, None, None, *ie.orbs, ie.value)
            else:
                self._insert_ur_element(self._ur_nonlocal, r_to_index, ie.r_lat, *ie.orbs, ie.value)
        return self

    def _create_er_grid(self, r_to_index: dict[list, int], n_orbs: int) -> np.ndarray:
        """
        Builds the real-space lattice-vector grid for the kinetic Hamiltonian.

        :param r_to_index: Mapping from lattice-vector tuple to its row index.
        :param n_orbs: Number of orbitals.
        :return: The grid array of shape ``[n_rp, o1, o2, 3]``.
        """
        n_rp = len(r_to_index)
        r_grid = np.zeros((n_rp, n_orbs, n_orbs, 3))
        for r_vec in r_to_index.keys():
            r_grid[r_to_index[r_vec], :, :, :] = r_vec
        return r_grid

    def _create_ur_grid(self, r_to_index: dict[list, int], n_orbs: int) -> np.ndarray:
        """
        Builds the real-space lattice-vector grid for the interaction Hamiltonian.

        :param r_to_index: Mapping from lattice-vector tuple to its row index.
        :param n_orbs: Number of orbitals.
        :return: The grid array of shape ``[n_rp, o1, o2, o3, o4, 3]``.
        """
        n_rp = len(r_to_index)
        r_grid = np.zeros((n_rp, n_orbs, n_orbs, n_orbs, n_orbs, 3))
        for r_vec in r_to_index.keys():
            r_grid[r_to_index[r_vec], :, :, :, :, :] = r_vec
        return r_grid

    def _create_er_orbs(self, n_rp: int, n_orbs: int) -> np.ndarray:
        """
        Builds the (1-based) orbital-index array for the kinetic Hamiltonian.

        :param n_rp: Number of real-space lattice vectors.
        :param n_orbs: Number of orbitals.
        :return: The orbital-index array of shape ``[n_rp, o1, o2, 2]``.
        """
        orbs = np.zeros((n_rp, n_orbs, n_orbs, 2))
        for r, io1, io2 in it.product(range(n_rp), range(n_orbs), range(n_orbs)):
            orbs[r, io1, io2, :] = np.array([io1 + 1, io2 + 1])
        return orbs

    def _create_ur_orbs(self, n_rp: int, n_orbs: int) -> np.ndarray:
        """
        Builds the (1-based) orbital-index array for the interaction Hamiltonian.

        :param n_rp: Number of real-space lattice vectors.
        :param n_orbs: Number of orbitals.
        :return: The orbital-index array of shape ``[n_rp, o1, o2, o3, o4, 4]``.
        """
        orbs = np.zeros((n_rp, n_orbs, n_orbs, n_orbs, n_orbs, 4))
        for r, io1, io2, io3, io4 in it.product(
            range(n_rp), range(n_orbs), range(n_orbs), range(n_orbs), range(n_orbs)
        ):
            orbs[r, io1, io2, io3, io4, :] = np.array([io1 + 1, io2 + 1, io3 + 1, io4 + 1])
        return orbs

    def _insert_er_element(
        self, er_mat: np.ndarray, r_to_index: dict[list, int], r_vec: list, orb1: int, orb2: int, hr_elem: float
    ) -> None:
        """
        Writes a single hopping value into the kinetic Hamiltonian matrix at the matching lattice/orbital position.

        :param er_mat: The kinetic Hamiltonian array to write into.
        :param r_to_index: Mapping from lattice-vector tuple to its row index.
        :param r_vec: The lattice vector of this element.
        :param orb1: First (1-based) orbital index.
        :param orb2: Second (1-based) orbital index.
        :param hr_elem: The hopping value.
        :return: None.
        """
        index = r_to_index[r_vec]
        er_mat[index, orb1 - 1, orb2 - 1] = hr_elem

    def _insert_ur_element(
        self,
        ur_mat: np.ndarray,
        r_to_index: dict[list, int] | None,
        r_vec: list | None,
        orb1: int,
        orb2: int,
        orb3: int,
        orb4: int,
        value: float,
    ) -> None:
        """
        Writes a single interaction value into the interaction Hamiltonian. If ``r_to_index``/``r_vec`` are None the
        value is treated as local (no lattice axis); otherwise it is placed at the matching lattice position.

        :param ur_mat: The interaction Hamiltonian array to write into (local or non-local).
        :param r_to_index: Mapping from lattice-vector tuple to its row index, or None for the local part.
        :param r_vec: The lattice vector of this element, or None for the local part.
        :param orb1: First (1-based) orbital index.
        :param orb2: Second (1-based) orbital index.
        :param orb3: Third (1-based) orbital index.
        :param orb4: Fourth (1-based) orbital index.
        :param value: The interaction value.
        :return: None.
        """
        if r_to_index is None or r_vec is None:
            ur_mat[orb1 - 1, orb2 - 1, orb3 - 1, orb4 - 1] = value
            return
        index = r_to_index[r_vec]
        ur_mat[index, orb1 - 1, orb2 - 1, orb3 - 1, orb4 - 1] = value

    def _convham_2_orbs(self, k_mesh: np.ndarray) -> np.ndarray:
        r"""
        Fourier-transforms the real-space hopping to the band dispersion :math:`\varepsilon_{ab}(k)`, looping over
        lattice vectors to keep the memory footprint low.

        :param k_mesh: The k-points as an array of shape ``[3, nk]``.
        :return: The band dispersion of shape ``[nk, o1, o2]``.
        """
        n_rp = self._er.shape[0]
        n_orbs = self._er.shape[1]
        nk = k_mesh.shape[1]

        result = np.zeros((n_orbs, n_orbs, nk), dtype=np.complex128)

        for r in range(n_rp):
            phase = np.exp(1j * np.tensordot(self._er_r_grid[r], k_mesh, axes=([2], [0])))
            phase /= float(self._er_r_weights[r, 0])
            result += phase * self._er[r, ..., None]

        return np.transpose(result, axes=(2, 0, 1))

    def _convham_4_orbs(self, k_mesh: np.ndarray) -> np.ndarray:
        r"""
        Fourier-transforms the real-space non-local interaction to :math:`V_{abcd}(q)`, looping over lattice vectors
        to keep the memory footprint low.

        :param k_mesh: The q-points as an array of shape ``[3, nk]``.
        :return: The momentum-dependent interaction of shape ``[nk, o1, o2, o3, o4]``.
        """
        n_rp = self._ur_nonlocal.shape[0]
        n_orbs = self._ur_nonlocal.shape[1]
        nk = k_mesh.shape[1]
        result = np.zeros((n_orbs, n_orbs, n_orbs, n_orbs, nk), dtype=np.complex128)

        for r in range(n_rp):
            phase = np.exp(1j * np.tensordot(self._ur_r_grid[r], k_mesh, axes=([4], [0])))
            phase /= float(self._ur_r_weights[r, 0])
            result += phase * self._ur_nonlocal[r, ..., None]

        return np.transpose(result, axes=(4, 0, 1, 2, 3))

    def _parse_elements(self, elements: list, element_type: type) -> list:
        """
        Normalizes a list of elements to instances of ``element_type``, constructing them from dicts where needed.

        :param elements: List of elements, each either an ``element_type`` instance or a dict of its kwargs.
        :param element_type: The target class (:class:`HoppingElement` or :class:`InteractionElement`).
        :return: A list of ``element_type`` instances.
        """
        if not all(isinstance(item, element_type) for item in elements):
            return [element_type(**element) for element in elements]
        return elements

    def _prepare_lattice_indices_and_orbs(self, elements: list) -> tuple:
        """
        Derives the lattice-vector indexing and dimensions shared by the Hamiltonian matrices from a list of elements.

        :param elements: List of hopping or interaction elements.
        :return: The tuple ``(r_to_index, n_rp, n_orbs)`` of the lattice-vector-to-index map, the number of lattice
            vectors, and the number of orbitals.
        """
        unique_lat_r = set([el.r_lat for el in elements])
        r_to_index = {tup: index for index, tup in enumerate(unique_lat_r)}
        n_rp = len(r_to_index)
        n_orbs = int(max(np.array([el.orbs for el in elements]).flatten()))
        return r_to_index, n_rp, n_orbs

    def _check_interaction_swapping_symmetry(self, uq_local: np.ndarray, uq_nonlocal: np.ndarray):
        r"""
        Checks the interaction swapping symmetry: :math:`U_{lmm'l'} = U_{mll'm'}` for the local part and
        :math:`V^{q}_{lmm'l'} = V^{-q}_{mll'm'}` for the non-local part.
        NOTE: The check for the non-local :math:`V^q` needs to be revised because a straight inversion of the first
        axis is not correct.

        :param uq_local: The local interaction tensor :math:`U_{abcd}`.
        :param uq_nonlocal: The non-local interaction tensor :math:`V_{abcd}(q)`.
        :return: None.
        :raises AssertionError: If either swapping symmetry is violated.
        """
        assert np.allclose(
            uq_local, np.einsum("abcd->badc", uq_local)
        ), "Swapping symmetry of the interaction is not satisfied!"

        assert np.allclose(
            uq_nonlocal, np.einsum("qabcd->qbadc", np.flip(uq_nonlocal, axis=0))
        ), "Swapping symmetry of the interaction is not satisfied!"

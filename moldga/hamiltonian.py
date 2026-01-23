import itertools as it
import logging

import numpy as np
import pandas as pd

import moldga.brillouin_zone as bz
from moldga.interaction import LocalInteraction, Interaction
from moldga.n_point_base import SpinChannel


class HoppingElement:
    """
    Data class to store a hopping element of the Hamiltonian. The hopping element is defined by the relative lattice
    vector 'r_lat', the orbitals 'orbs' and the value of the hopping 'value'. A hopping element represents a single line
    in a w2k file. Added this to make the code more readable and to avoid passing around lists of values.
    """

    def __init__(self, r_lat: list, orbs: list, value: float = 0.0):
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
    Data class to store an interaction element of the Hamiltonian. The interaction element is defined by the relative lattice
    vector 'r_lat', the orbitals 'orbs' and the value of the interaction 'value'. An interaction element represents a single entry
    in the interaction matrix. Added this to make the code more readable and to avoid passing around lists of values.
    """

    def __init__(self, r_lat: list[int], orbs: list[int], value: float):
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
    Class to handle the Hamiltonian of the Hubbard model. Contains the hopping terms ('er') and the local ('local_interaction')
    and nonlocal interaction ('nonlocal_interaction') terms. Has a few helper methods to simplify adding terms to the Hamiltonian.
    """

    def __init__(self):
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
        """
        Sets the local interaction term for a single band model from the input of a floating point number.
        """
        return self.interaction_orbital_diagonal(u, 1)

    def interaction_orbital_diagonal(self, u: float, n_bands: int = 1) -> "Hamiltonian":
        """
        Sets the local interaction term for a multi-band model from a floating point number.
        The interaction matrix is going to be zero everywhere except for the [n,n,n,n] element, which is set to 'u'.
        """
        interaction_elements = []
        for i, j, k, l in it.product(range(n_bands), repeat=4):
            if i == j == k == l == 0:
                interaction_elements.append(InteractionElement([0, 0, 0], [i + 1, j + 1, k + 1, l + 1], u))
            else:
                interaction_elements.append(InteractionElement([0, 0, 0], [i + 1, j + 1, k + 1, l + 1], 0))

        return self._add_interaction_term(interaction_elements)

    def kanamori_interaction(self, n_bands: int, udd: float, jdd: float, vdd: float = None) -> "Hamiltonian":
        """
        Adds the Kanamori interaction terms to the Hamiltonian. The interaction terms are defined by the Hubbard 'udd' (U),
        the exchange interaction 'jdd' (J) and the pair hopping 'vdd' (V or sometimes U'). 'vdd' is an optional parameter, if left empty,
        it is set to U-2J.
        """
        if vdd is None:
            vdd = udd - 2 * jdd

        r_loc = [0, 0, 0]

        interaction_elements = []
        for a, b, c, d in it.product(range(n_bands), repeat=4):
            bands = [a + 1, b + 1, c + 1, d + 1]
            if a == b == c == d:  # U_{llll}
                interaction_elements.append(InteractionElement(r_loc, bands, udd))
            elif (a == d and b == c) or (a == b and c == d):  # U_{lmml} or U_{llmm}
                interaction_elements.append(InteractionElement(r_loc, bands, jdd))
            elif a == c and b == d:  # U_{lmlm}
                interaction_elements.append(InteractionElement(r_loc, bands, vdd))

        return self._add_interaction_term(interaction_elements)

    def kinetic_one_band_2d_t_tp_tpp(self, t: float, tp: float, tpp: float) -> "Hamiltonian":
        """
        Adds the kinetic terms for a one-band model in 2D with nearest (t), next-nearest (tp) and next-next-nearest
        (tpp) neighbor hopping.
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
        Reads the 'wannier_hr.dat' file from a wien2k hr file and sets the real part of the kinetic Hamiltonian.
        This is then typically Fourier-transformed to momentum space to obtain the k-dependent band dispersion.
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
        """Reads a Hamiltonian f$ H_{bb'}(k) f$ from a text file.

        Expects a text file with white-space separated values in the syntax as
        generated by wannier90:  the first line is a header with three integers,
        optionally followed by '#'-prefixed comment:

            <no of k-points> <no of wannier functions> <no of bands (ignored)>

        For each k-point, there is a header line with the x, y, z coordinates of the
        k-point, followed by <nbands> rows as lines of 2*<nbands> values each, which
        are the real and imaginary part of each column.

        Returns: A pair (hk, kpoints):
         - hk       three-dimensional complex array of the Hamiltonian H(k),
                    where the first index corresponds to the k-point and the other
                    dimensions mark band indices.
         - kpoints  two-dimensional real array, which contains the
                    components (x,y,z) of the different kpoints.
        """
        logger = logging.getLogger()
        hk_file = open(fname, "r", encoding="utf-8")
        spin_orbit = not spin_sym

        def nextline():
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
        Write the real-space kinetic Hamiltonian in the format of w2k to a file.
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
        Write the k-space kinetic Hamiltonian in the format of w2k to a file.
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
        """
        Returns the band dispersion for the Hamiltonian on the input k-grid.
        """
        if self._ek is not None:
            return self._ek
        self._ek = self._convham_2_orbs(k_grid.kmesh.reshape(3, -1))
        n_bands = self._ek.shape[-1]
        self._ek = self._ek.reshape(*k_grid.nk, n_bands, n_bands)
        return self._ek

    def set_ek(self, ek: np.ndarray) -> "Hamiltonian":
        """
        Sets the band dispersion for the Hamiltonian and returns the Hamiltonian.
        """
        self._ek = ek
        return self

    def get_local_u(self) -> LocalInteraction:
        """
        Returns the local interaction term in momentum space. Note that due to the momentum-independence of the local
        interaction, the momentum space representation is the same as the real space representation.
        """
        return LocalInteraction(self._ur_local)

    def get_vq(self, q_grid: bz.KGrid) -> "Interaction":
        """
        Returns the nonlocal interaction term in momentum space. The nonlocal interaction is momentum-dependent.
        """
        uq = self._convham_4_orbs(q_grid.kmesh.reshape(3, -1))
        n_bands = uq.shape[-1]
        return Interaction(uq.reshape(*q_grid.nk + (n_bands,) * 4), SpinChannel.NONE, q_grid.nk)

    def _add_kinetic_term(self, hopping_elements: list) -> "Hamiltonian":
        """
        Creates the kinetic term of the Hamiltonian from a list of hopping elements.
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
        Creates the interaction term of the Hamiltonian from the interaction elements.
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
        Creates the real-space grid for the kinetic Hamiltonian.
        """
        n_rp = len(r_to_index)
        r_grid = np.zeros((n_rp, n_orbs, n_orbs, 3))
        for r_vec in r_to_index.keys():
            r_grid[r_to_index[r_vec], :, :, :] = r_vec
        return r_grid

    def _create_ur_grid(self, r_to_index: dict[list, int], n_orbs: int) -> np.ndarray:
        """
        Creates the real-space grid for the interaction Hamiltonian.
        """
        n_rp = len(r_to_index)
        r_grid = np.zeros((n_rp, n_orbs, n_orbs, n_orbs, n_orbs, 3))
        for r_vec in r_to_index.keys():
            r_grid[r_to_index[r_vec], :, :, :, :, :] = r_vec
        return r_grid

    def _create_er_orbs(self, n_rp: int, n_orbs: int) -> np.ndarray:
        """
        Creates the orbital indices for the kinetic Hamiltonian.
        """
        orbs = np.zeros((n_rp, n_orbs, n_orbs, 2))
        for r, io1, io2 in it.product(range(n_rp), range(n_orbs), range(n_orbs)):
            orbs[r, io1, io2, :] = np.array([io1 + 1, io2 + 1])
        return orbs

    def _create_ur_orbs(self, n_rp: int, n_orbs: int) -> np.ndarray:
        """
        Creates the orbital indices for the interaction Hamiltonian.
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
        Inserts a hopping element into the kinetic Hamiltonian.
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
        Inserts an interaction element into the interaction Hamiltonian.
        """
        if r_to_index is None or r_vec is None:
            ur_mat[orb1 - 1, orb2 - 1, orb3 - 1, orb4 - 1] = value
            return
        index = r_to_index[r_vec]
        ur_mat[index, orb1 - 1, orb2 - 1, orb3 - 1, orb4 - 1] = value

    def _convham_2_orbs(self, k_mesh: np.ndarray) -> np.ndarray:
        """
        Fourier transforms the kinetic Hamiltonian to momentum space (more memory-friendly).
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
        """
        Fourier transforms the interaction Hamiltonian to momentum space (more memory-friendly).
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
        Helper method. Parses the input elements to the correct data class.
        """
        if not all(isinstance(item, element_type) for item in elements):
            return [element_type(**element) for element in elements]
        return elements

    def _prepare_lattice_indices_and_orbs(self, elements: list) -> tuple:
        """
        Helper method. Prepares the indices and dimensions for the matrices contained in the Hamiltonian.
        """
        unique_lat_r = set([el.r_lat for el in elements])
        r_to_index = {tup: index for index, tup in enumerate(unique_lat_r)}
        n_rp = len(r_to_index)
        n_orbs = int(max(np.array([el.orbs for el in elements]).flatten()))
        return r_to_index, n_rp, n_orbs

    def _check_interaction_swapping_symmetry(self, uq_local: np.ndarray, uq_nonlocal: np.ndarray):
        """
        Checks the swapping symmetry on the local interaction matrix. This symmetry is defined as:
        U_{lmm'l'} = U_{mll'm'} for the local part and V^{q}_{lmm'l'} = V^{-q}_{mll'm'} for the nonlocal part.
        NOTE: The check for the nonlocal vq needs to be changed because a straight up inversion of the first axis is not
        correct.
        """
        assert np.allclose(
            uq_local, np.einsum("abcd->badc", uq_local)
        ), "Swapping symmetry of the interaction is not satisfied!"

        assert np.allclose(
            uq_nonlocal, np.einsum("qabcd->qbadc", np.flip(uq_nonlocal, axis=0))
        ), "Swapping symmetry of the interaction is not satisfied!"

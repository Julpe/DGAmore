"""
Module to handle operations within the (irreduzible) Brilloun zone. Copied over from Paul Worm's code.
Only modified the constant arrays and made enums out of them for type hinting.
"""

import warnings
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np


class KnownSymmetries(Enum):
    """
    Known symmetries of the Brillouin zone.
    """

    X_INV = "x-inv"
    Y_INV = "y-inv"
    Z_INV = "z-inv"
    X_Y_SYM = "x-y-sym"
    X_Z_SYM = "x-z-sym"
    Y_Z_SYM = "y-z-sym"
    X_Y_INV = "x-y-inv"


class KnownKPoints(Enum):
    """
    Known k-points in the Brillouin zone.
    """

    GAMMA = [0, 0, 0]
    X = [0.5, 0, 0]  # [np.pi, 0, 0]
    Y = [0, 0.5, 0]  # [0, np.pi, 0]
    M = [0.5, 0.5, 0]  # [np.pi, np.pi, 0]
    M2 = [0.25, 0.25, 0]  # [np.pi/2, np.pi/2, 0]
    Z = [0.0, 0.0, 0.5]  # [np.pi/2, np.pi/2, 0]
    R = [0.5, 0.0, 0.5]  # [np.pi/2, np.pi/2, 0]
    A = [0.5, 0.5, 0.5]  # [np.pi/2, np.pi/2, 0]


class Labels(Enum):
    """
    Labels for the k-points in the Brillouin zone.
    """

    GAMMA = r"$\Gamma$"
    X = "X"
    Y = "Y"
    M = "M"
    M2 = "M2"
    Z = "Z"
    R = "R"
    A = "A"


def fmmm_amam_symmetries() -> list[KnownSymmetries]:
    """
    Amam/Fmmm lattice symmetries in three dimensions.
    """
    return [KnownSymmetries.X_INV, KnownSymmetries.Y_INV, KnownSymmetries.Z_INV]


def two_dimensional_square_symmetries() -> list[KnownSymmetries]:
    """
    Two-dimensional square lattice symmetries.
    """
    return [KnownSymmetries.X_INV, KnownSymmetries.Y_INV, KnownSymmetries.X_Y_SYM]


def three_dimensional_cubic_symmetries() -> list[KnownSymmetries]:
    """
    Three-dimensional cubic lattice symmetries.
    """
    return [
        KnownSymmetries.X_INV,
        KnownSymmetries.Y_INV,
        KnownSymmetries.Z_INV,
        KnownSymmetries.X_Y_SYM,
        KnownSymmetries.X_Z_SYM,
        KnownSymmetries.Y_Z_SYM,
    ]


def two_dimensional_nematic_symmetries() -> list[KnownSymmetries]:
    """
    Two-dimensional nematic lattice symmetries.
    """
    return [KnownSymmetries.X_INV, KnownSymmetries.Y_INV]


def quasi_two_dimensional_square_symmetries() -> list[KnownSymmetries]:
    """
    Quasi-two-dimensional square lattice symmetries.
    """
    return [KnownSymmetries.X_INV, KnownSymmetries.Y_INV, KnownSymmetries.Z_INV, KnownSymmetries.X_Y_SYM]


def quasi_one_dimensional_square_symmetries() -> list[KnownSymmetries]:
    """
    Quasi-one-dimensional square lattice symmetries.
    """
    return [KnownSymmetries.X_INV, KnownSymmetries.Y_INV]


def simultaneous_x_y_inversion() -> list[KnownSymmetries]:
    """
    Simultaneous inversion in x and y direction.
    """
    return [KnownSymmetries.X_Y_INV]


def inv_sym(mat: np.ndarray, axis) -> None:
    """
    In-place inversion symmetry applied to mat along dimension axis
    assumes that the grid is from [0,2pi), hence 0 does not map.
    """
    assert axis in [0, 1, 2], f"axis = {axis} but must be in [0,1,2]"
    assert len(np.shape(mat)) >= 3, f"dim(mat) = {len(np.shape(mat))} but must be at least 3 dimensional"
    len_ax = np.shape(mat)[axis] // 2
    mod_2 = np.shape(mat)[axis] % 2
    if axis == 0:
        mat[len_ax + 1 :, :, :, ...] = mat[1 : len_ax + mod_2, :, :, ...][::-1]
    if axis == 1:
        mat[:, len_ax + 1 :, :, ...] = mat[:, 1 : len_ax + mod_2, :, ...][:, ::-1]
    if axis == 2:
        mat[:, :, len_ax + 1 :, ...] = mat[:, :, 1 : len_ax + mod_2, ...][:, :, ::-1]


def x_y_sym(mat: np.ndarray) -> None:
    """
    In-place x-y symmetry applied to matrix.
    """
    _pairwise_sym(mat, 0, 1)


def x_z_sym(mat: np.ndarray) -> None:
    """
    In-place x-z symmetry applied to matrix.
    """
    _pairwise_sym(mat, 0, 2)


def y_z_sym(mat: np.ndarray) -> None:
    """
    In-place y-z symmetry applied to matrix.
    """
    _pairwise_sym(mat, 1, 2)


def _pairwise_sym(mat: np.ndarray, axis_a: int, axis_b: int) -> None:
    """
    In-place symmetry swapping axis_a and axis_b applied to matrix.
    """
    assert axis_a in [0, 1, 2] and axis_b in [0, 1, 2], f"axes = {(axis_a, axis_b)} must be in [0,1,2]"
    assert len(np.shape(mat)) >= 3, f"dim(mat) = {len(np.shape(mat))} but must be at least 3 dimensional"
    if mat.shape[axis_a] == mat.shape[axis_b]:
        mat[...] = np.minimum(mat, mat.swapaxes(axis_a, axis_b))
    else:
        warnings.warn(f"Matrix not compatible for symmetry between axes {axis_a} and {axis_b}. Doing nothing.")


def x_y_inv(mat: np.ndarray) -> None:
    """
    Simultaneous inversion in x and y direction.
    """
    assert len(np.shape(mat)) >= 3, f"dim(mat) = {len(np.shape(mat))} but must be at least 3 dimensional"
    len_ax_x = np.shape(mat)[0] // 2
    mod_2_x = np.shape(mat)[0] % 2
    mat[len_ax_x + 1 :, 1:, :, ...] = mat[1 : len_ax_x + mod_2_x, 1:, :][::-1, ::-1, :, ...]


def apply_symmetry(mat: np.ndarray, sym: KnownSymmetries) -> None:
    """
    Applies a single symmetry to matrix.
    """
    assert sym in KnownSymmetries, f"sym = {sym} not in known symmetries {KnownSymmetries}."
    if sym == KnownSymmetries.X_INV:
        inv_sym(mat, 0)
    if sym == KnownSymmetries.Y_INV:
        inv_sym(mat, 1)
    if sym == KnownSymmetries.Z_INV:
        inv_sym(mat, 2)
    if sym == KnownSymmetries.X_Y_SYM:
        x_y_sym(mat)
    if sym == KnownSymmetries.X_Z_SYM:
        x_z_sym(mat)
    if sym == KnownSymmetries.Y_Z_SYM:
        y_z_sym(mat)
    if sym == KnownSymmetries.X_Y_INV:
        x_y_inv(mat)


def apply_symmetries(mat: np.ndarray, symmetries: list[KnownSymmetries]) -> None:
    """
    Applies symmetries to matrix in-place.
    """
    assert len(mat.shape) >= 3, f"dim(mat) = {len(np.shape(mat))} but must at least 3 dimensional"
    if not symmetries:
        return
    for sym in symmetries:
        apply_symmetry(mat, sym)


def get_lattice_symmetries_from_string(symmetry_string: str) -> list[KnownSymmetries]:
    """
    Return the lattice symmetries from a string.
    """
    if symmetry_string == "two_dimensional_square":
        return two_dimensional_square_symmetries()
    elif symmetry_string == "three_dimensional_cubic":
        return three_dimensional_cubic_symmetries()
    elif symmetry_string == "quasi_one_dimensional_square":
        return quasi_one_dimensional_square_symmetries()
    elif symmetry_string == "simultaneous_x_y_inversion":
        return simultaneous_x_y_inversion()
    elif symmetry_string == "quasi_two_dimensional_square_symmetries":
        return quasi_two_dimensional_square_symmetries()
    elif symmetry_string.lower() == "fmmm" or symmetry_string.lower() == "amam" or symmetry_string.lower() == "cmcm":
        return fmmm_amam_symmetries()
    elif not symmetry_string or symmetry_string == "none":
        return []
    elif isinstance(symmetry_string, (tuple, list)):
        symmetries = []
        for sym in symmetry_string:
            if sym not in [s.value for s in KnownSymmetries]:
                raise NotImplementedError(f"Symmetry {sym} not supported.")
            symmetries.append(KnownSymmetries(sym))
        return symmetries
    else:
        raise NotImplementedError(f"Symmetry {symmetry_string} not supported.")


class KGrid:
    """
    Class to build the k-grid for the Brillouin zone.
    """

    def __init__(self, nk: tuple = None, symmetries: list[KnownSymmetries] = None):
        self.kx = None  # kx-grid
        self.ky = None  # ky-grid
        self.kz = None  # kz-grid
        self.irrk_ind = None  # Index of the irreducible BZ points
        self.irrk_inv = None  # Index map back to the full BZ from the irreducible one
        self.irrk_count = None  # duplicity of each k-point in the irreducible BZ
        self.irr_kmesh = None  # k-meshgrid of the irreduzible BZ
        self.fbz2irrk = None  # index map from the full BZ to the irreduzible one
        self.symmetries = symmetries
        self.ind = None

        self.nk = nk
        self.set_k_axes()

        self.set_fbz2irrk()
        self.set_irrk_maps()
        self.set_irrk_mesh()

    def set_fbz2irrk(self) -> None:
        """
        Set the mapping from the full BZ to the irreducible one by applying the lattice symmetries.
        """
        self.fbz2irrk = np.reshape(np.arange(0, np.prod(self.nk)), self.nk)
        apply_symmetries(self.fbz2irrk, self.symmetries)

    def set_irrk_maps(self) -> None:
        """
        Set the mapping from the irreducible BZ to the full one and the inverse.
        """
        _, self.irrk_ind, self.irrk_inv, self.irrk_count = np.unique(
            self.fbz2irrk, return_index=True, return_inverse=True, return_counts=True
        )

    def set_irrk_mesh(self) -> None:
        """
        Set the k-meshgrid of the irreducible BZ.
        """
        self.irr_kmesh = np.array([self.kmesh[ax].flatten()[self.irrk_ind] for ax in range(len(self.nk))])

    @property
    def kx_shift(self) -> float:
        r"""
        Returns the kx grid shifted by :math:`pi` in the half-open interval i.e. :math:`[-\pi,\pi)`.
        """
        return self.kx - np.pi

    @property
    def ky_shift(self) -> float:
        r"""
        Returns the ky grid shifted by :math:`pi` in the half-open interval i.e. :math:`[-\pi,\pi)`.
        """
        return self.ky - np.pi

    @property
    def kz_shift(self) -> float:
        r"""
        Returns the kz grid shifted by :math:`pi` in the half-open interval i.e. :math:`[-\pi,\pi)`.
        """
        return self.kz - np.pi

    @property
    def kx_shift_closed(self) -> np.ndarray:
        r"""
        Returns the kx grid shifted by :math:`pi` in the closed interval i.e. :math:`[-\pi,\pi]`.
        """
        return np.array([*(self.kx - np.pi), -self.kx[0] + np.pi])

    @property
    def ky_shift_closed(self) -> np.ndarray:
        r"""
        Returns the ky grid shifted by :math:`pi` in the closed interval i.e. :math:`[-\pi,\pi]`.
        """
        return np.array([*(self.ky - np.pi), -self.ky[0] + np.pi])

    @property
    def kz_shift_closed(self) -> np.ndarray:
        r"""
        Returns the kz grid shifted by :math:`pi` in the closed interval i.e. :math:`[-\pi,\pi]`.
        """
        return np.array([*(self.kz - np.pi), -self.kz[0] + np.pi])

    @property
    def grid(self) -> tuple:
        """
        Returns the k-grid as a tuple of arrays.
        """
        return self.kx, self.ky, self.kz

    @property
    def nk_tot(self):
        """
        Returns the total number of k-points in the full BZ.
        """
        return np.prod(self.nk)

    @property
    def nk_irr(self) -> int:
        """
        Returns the number of k-points in the irreducible BZ.
        """
        return np.size(self.irrk_ind)

    @property
    def kmesh(self) -> np.ndarray:
        """
        Meshgrid of {kx,ky,kz}.
        """
        return np.array(np.meshgrid(self.kx, self.ky, self.kz, indexing="ij"))

    @property
    def kmesh_ind(self) -> np.ndarray:
        r"""
        Indices of {kx,ky,kz}
        Only works for meshes that go from 0 to :math:`2\pi`.
        """
        ind_x = np.arange(0, self.nk[0])
        ind_y = np.arange(0, self.nk[1])
        ind_z = np.arange(0, self.nk[2])
        return np.array(np.meshgrid(ind_x, ind_y, ind_z, indexing="ij"))

    @property
    def kmesh_list(self):
        """
        List of {kx,ky,kz}
        """
        return self.kmesh.reshape((3, -1))

    def set_k_axes(self) -> None:
        """
        Set the k-axes for the full BZ.
        """
        self.kx = np.linspace(0, 2 * np.pi, self.nk[0], endpoint=False)
        self.ky = np.linspace(0, 2 * np.pi, self.nk[1], endpoint=False)
        self.kz = np.linspace(0, 2 * np.pi, self.nk[2], endpoint=False)

    def get_q_list(self) -> np.ndarray:
        """
        Return list of all q-point indices in the BZ.
        """
        return np.array([self.kmesh_ind[i].flatten() for i in range(3)]).T

    def get_irrq_list(self) -> np.ndarray:
        """
        Return list of all q-point indices in the irreduzible BZ.
        """
        return np.array([self.kmesh_ind[i].flatten()[self.irrk_ind] for i in range(3)]).T


class KPath:
    """
    Object to generate paths in the Brillouin zone.
    Currently assumed that the BZ grid is from (0,2*pi)
    """

    def __init__(self, nk, path, kx=None, ky=None, kz=None, path_deliminator="-"):
        """
        nk: number of points in each dimension (tuple)
        path: desired path in the Brillouin zone (string)
        """
        self.path_deliminator = path_deliminator
        self.path = path
        self.nk = nk

        # Set k-grids:
        self.kx = self.set_kgrid(kx, nk[0])
        self.ky = self.set_kgrid(ky, nk[1])
        self.kz = self.set_kgrid(kz, nk[2])

        # Set the k-path:
        self.ckp = self.corner_k_points()
        self.kpts, self.nkp = self.build_k_path()
        self.k_val = self.get_kpath_val()
        self.k_points = self.get_kpoints()

    def get_kpath_val(self):
        k = [self.kx[self.kpts[:, 0]], self.kx[self.kpts[:, 1]], self.kx[self.kpts[:, 2]]]
        return k

    def set_kgrid(self, k_in, nk):
        if k_in is None:
            k = np.linspace(0, np.pi * 2, nk, endpoint=False)
        else:
            k = k_in
        return k

    @property
    def ckps(self):
        """Corner k-point strings"""
        return self.path.split(self.path_deliminator)

    @property
    def labels(self):
        """Labels of the k-points for plotting"""
        count = 0
        ckps = self.ckps
        labels = []
        for k_p in ckps:
            if k_p in [s.value for s in KnownSymmetries]:
                labels.append(Labels[k_p].value)
            else:
                labels.append(f"K{count}")
            count += 1
        return labels

    @property
    def x_ticks(self):
        """Return ticks values for plotting"""
        return self.k_axis[self.cind]

    @property
    def cind(self):
        return np.concatenate(([0], np.cumsum(self.nkp) - 1))

    @property
    def ikx(self):
        return self.kpts[:, 0]

    @property
    def iky(self):
        return self.kpts[:, 1]

    @property
    def ikz(self):
        return self.kpts[:, 2]

    @property
    def k_axis(self):
        return np.linspace(0, 1, np.sum(self.nkp), endpoint=True)

    @property
    def nk_tot(self):
        return np.sum(self.nkp)

    @property
    def nk_seg(self):
        return np.diff(self.cind)

    def get_kpoints(self):
        return np.array(self.k_val).T

    def corner_k_points(self):
        ckps = self.ckps
        ckp = np.zeros((np.size(ckps), 3))
        for i, kps in enumerate(ckps):
            if kps in [s.value for s in KnownSymmetries]:
                ckp[i, :] = KnownKPoints[kps].value
            else:
                ckp[i, :] = get_k_point_from_string(kps)

        return ckp

    def map_to_kpath(self, mat):
        """Map mat [kx,ky,kz,...] onto the k-path"""
        return mat[self.ikx, self.iky, self.ikz, ...]

    def build_k_path(self):
        k_path = []
        nkp = []
        nckp = np.shape(self.ckp)[0]
        for i in range(nckp - 1):
            segment, nkps = kpath_segment(self.ckp[i], self.ckp[i + 1], self.nk)
            nkp.append(nkps)
            if i == 0:
                k_path = segment
            else:
                k_path = np.concatenate((k_path, segment))
        return k_path, nkp

    def plot_kpoints(self, fname=None):
        plt.figure()
        plt.plot(self.kpts[:, 0], color="cornflowerblue", label="$k_x$")
        plt.plot(self.kpts[:, 1], color="firebrick", label="$k_y$")
        plt.plot(self.kpts[:, 2], color="seagreen", label="$k_z$")
        plt.legend()
        plt.xlabel("Path-index")
        plt.ylabel("k-index")
        if fname is not None:
            plt.savefig(fname + "_q_path.png", dpi=300)
        plt.show()

    def plot_kpath(self, mat, verbose=False, do_save=True, pdir="./", name="k_path", ylabel="Energy [t]"):
        """
        mat: [kx,ky,kz]
        """
        plt.figure()
        plt.xticks(self.x_ticks, self.labels)
        plt.vlines(self.x_ticks, np.min(mat), np.max(mat), ls="-", color="grey", alpha=0.8)
        plt.hlines(0, self.k_axis[0], self.k_axis[-1], ls="--", color="grey", alpha=0.8)
        plt.plot(self.k_axis, self.map_to_kpath(mat), "-k")
        plt.ylabel(ylabel)
        plt.xlim(self.k_axis[0], self.k_axis[-1])
        plt.ylim(np.min(mat), np.max(mat))
        if do_save:
            plt.savefig(pdir + "/" + name + ".png", dpi=300)
        if verbose:
            plt.show()
        plt.close()

    def get_bands(self, ek):
        """Return the bands along the k-path"""
        ek_kpath = self.map_to_kpath(ek)
        bands = np.zeros((ek_kpath.current_shape[:-1]))
        for i, eki in enumerate(ek_kpath):
            val, _ = np.linalg.eig(eki)
            bands[i, :] = np.sort(val).real
        return bands


def kpath_segment(k_start, k_end, nk):
    nkp = int(np.round(np.linalg.norm(k_start * nk - k_end * nk)))
    k_segment = (
        k_start[None, :] * nk + np.linspace(0, 1, nkp, endpoint=False)[:, None] * ((k_end - k_start) * nk)[None, :]
    )
    k_segment = np.round(k_segment).astype(int)
    for i, nki in enumerate(nk):
        ind = np.where(k_segment[:, i] >= nki)
        k_segment[ind, i] = k_segment[ind, i] - nki
    return k_segment, nkp


def get_k_point_from_string(string):
    scoords = string.split(" ")
    coords = np.array([float(sc) for sc in scoords])
    return coords


def get_bz_masks(nk):
    mask_1q = np.ones((nk, nk), dtype=int)
    mask_2q = np.ones((nk, nk), dtype=int)
    mask_3q = np.ones((nk, nk), dtype=int)
    mask_4q = np.ones((nk, nk), dtype=int)
    mask_3q[: nk // 2, : nk // 2] = 0
    mask_1q[nk // 2 :, : nk // 2] = 0
    mask_2q[nk // 2 :, nk // 2 :] = 0
    mask_4q[: nk // 2, nk // 2 :] = 0
    return [mask_1q, mask_2q, mask_3q, mask_4q]


def shift_mat_by_ind(mat, ind=(0, 0, 0)):
    """Structure of mat has to be {kx,ky,kz,...}"""
    return np.roll(mat, ind, axis=(0, 1, 2))


if __name__ == "__main__":
    pass

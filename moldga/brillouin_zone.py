"""
Module to handle operations within the (irreducible) Brillouin zone. Copied over from Paul Worm's code.
Only modified the constant arrays and made enums out of them for type hinting.
"""

import warnings
from enum import Enum

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


class KnownOrbitalBases(Enum):
    """
    Known orbital bases and their mirror symmetry transformation matrices.

    Each entry contains the unitary matrices U such that under a mirror symmetry g:
        O_ab(k) = U_aa' U*_bb' O_a'b'(k')
    where 'a' is an annihilation index and 'b' is a creation index,
    consistent with G_abcd := <T[c_a c†_b c_c c†_d]>.

    Orbital orderings:
        EG  : {d3z2-r2, dx2-y2}
        T2G : {dxz, dyz, dxy}
    """

    EG = "eg"
    T2G = "t2g"

    @staticmethod
    def from_string(s: str) -> "KnownOrbitalBases":
        s = s.strip().lower()
        for basis in KnownOrbitalBases:
            if s == basis.value:
                return basis
        raise ValueError(f"Unknown orbital basis '{s}'. " f"Supported bases: {[b.value for b in KnownOrbitalBases]}.")

    def get_mirror_rotations(self) -> dict["KnownSymmetries", np.ndarray]:
        """
        Returns the mirror symmetry orbital rotation matrices for this basis.
        Only mirror symmetries are included; inversion symmetries are always identity
        and are handled separately in build_orbital_rotations.
        """
        match self:
            case KnownOrbitalBases.EG:
                # eg orbitals: {d3z2-r2, dx2-y2}
                # kx <-> ky: dx2-y2 sign change, d3z2-r2 unchanged
                # kx <-> kz, ky <-> kz: mix d3z2-r2 and dx2-y2 via cubic rotation
                sqrt3 = np.sqrt(3)
                return {
                    KnownSymmetries.X_Y_SYM: np.array(
                        [[1, 0], [0, -1]], dtype=complex  # d3z2-r2 -> d3z2-r2  # dx2-y2  -> -dx2-y2
                    ),
                    KnownSymmetries.X_Z_SYM: np.array([[-1 / 2, sqrt3 / 2], [sqrt3 / 2, 1 / 2]], dtype=complex),
                    KnownSymmetries.Y_Z_SYM: np.array([[-1 / 2, -sqrt3 / 2], [-sqrt3 / 2, 1 / 2]], dtype=complex),
                }
            case KnownOrbitalBases.T2G:
                # NOTE: it depends on the ordering of the orbitals in the wannier files which matrices to choose!
                # if {dxy=0, dxz=1, dyz=2} (alphabetical ordering)
                # dxy lies in the xy plane -> invariant under X_Y_SYM
                # dxz lies in the xz plane -> invariant under X_Z_SYM
                # dyz lies in the yz plane -> invariant under Y_Z_SYM
                # X_Y_SYM (kx<->ky): dxz(1) <-> dyz(2), dxy(0) unchanged
                # X_Z_SYM (kx<->kz): dxy(0) <-> dyz(2), dxz(1) unchanged
                # Y_Z_SYM (ky<->kz): dxy(0) <-> dxz(1), dyz(2) unchanged
                return {
                    KnownSymmetries.X_Y_SYM: np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex),
                    KnownSymmetries.X_Z_SYM: np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=complex),
                    KnownSymmetries.Y_Z_SYM: np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex),
                }

                """
                # If {dxz=0, dxy=1, dyz=2}
                # X_Y_SYM (kx<->ky): dxz(0) <-> dyz(2), dxy(1) unchanged
                # X_Z_SYM (kx<->kz): dxy(1) <-> dyz(2), dxz(0) unchanged
                # Y_Z_SYM (ky<->kz): dxz(0) <-> dxy(1), dyz(2) unchanged
                return {
                    KnownSymmetries.X_Y_SYM: np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=complex),
                    KnownSymmetries.X_Z_SYM: np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex),
                    KnownSymmetries.Y_Z_SYM: np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex),
                }
                """


INVERSION_SYMMETRIES = frozenset(
    {KnownSymmetries.X_INV, KnownSymmetries.Y_INV, KnownSymmetries.Z_INV, KnownSymmetries.X_Y_INV}
)

MIRROR_SYMMETRIES = frozenset({KnownSymmetries.X_Y_SYM, KnownSymmetries.X_Z_SYM, KnownSymmetries.Y_Z_SYM})


class KnownKPoints(Enum):
    """
    Known k-points in the Brillouin zone.
    """

    GAMMA = (0.0, 0.0, 0.0)
    X = (0.5, 0.0, 0.0)
    Y = (0.0, 0.5, 0.0)
    Z = (0.0, 0.0, 0.5)
    M = (0.5, 0.5, 0.0)
    M2 = (0.25, 0.25, 0.0)
    R = (0.5, 0.0, 0.5)
    A = (0.5, 0.5, 0.5)


class Labels(Enum):
    """
    Labels for the k-points in the Brillouin zone.
    """

    GAMMA = ("gamma", r"$\Gamma$")
    X = ("x", "X")
    Y = ("y", "Y")
    Z = ("z", "Z")
    M = ("m", "M")
    M2 = ("m2", "M2")
    R = ("r", "R")
    A = ("a", "A")

    @property
    def key(self):
        return self.value[0]

    @property
    def latex(self):
        return self.value[1]

    @staticmethod
    def from_string(s: str):
        s = s.strip().lower()
        for label in Labels:
            if s == label.key:
                return label
        raise ValueError(f"Unknown label string: {s}")


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
    assert axis_a in [0, 1, 2] and axis_b in [0, 1, 2]
    assert mat.ndim >= 3
    if mat.shape[axis_a] == mat.shape[axis_b]:
        merged = np.minimum(mat, mat.swapaxes(axis_a, axis_b))
        mat[...] = np.minimum(merged, merged.swapaxes(axis_a, axis_b))
    else:
        warnings.warn(f"Matrix not compatible for symmetry between axes {axis_a} and {axis_b}. Doing nothing.")


def x_y_inv(mat: np.ndarray) -> None:
    """
    Simultaneous inversion in x and y direction.
    """
    assert mat.ndim >= 3, f"dim(mat) = {mat.ndim} but must be at least 3 dimensional"
    len_ax_x = mat.shape[0] // 2
    mod_2_x = mat.shape[0] % 2
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
    assert mat.ndim >= 3, f"dim(mat) = {mat.ndim} but must at least 3 dimensional"
    if not symmetries:
        return
    for sym in symmetries:
        apply_symmetry(mat, sym)


def get_lattice_symmetries_from_string(symmetry_string: str | tuple | list) -> list[KnownSymmetries]:
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
    elif not symmetry_string or symmetry_string == "none":
        return []

    try:
        import ast

        if not isinstance(symmetry_string, (tuple, list)):
            symmetry_string = ast.literal_eval(symmetry_string)
    except (ValueError, SyntaxError):
        raise ValueError("Symmetry does not exist or input cannot be parsed as a Python literal.")

    if isinstance(symmetry_string, (tuple, list)):
        symmetries = []
        for sym in symmetry_string:
            if sym.lower() not in [s.value.lower() for s in KnownSymmetries]:
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
        self.irrk_count = None  # Duplicity of each k-point in the irreducible BZ
        self.irr_kmesh = None  # k-meshgrid of the irreducible BZ
        self.fbz2irrk = None  # Index map from the full BZ to the irreducible one
        self.fbz2sym = None  # Index of symmetry operation mapping each FBZ point to its representative
        self.symmetries = symmetries
        self.ind = None
        self.orbital_rot_u = None

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
        Set the mapping from the irreducible BZ to the full one, the inverse, and the symmetry map.
        """
        _, self.irrk_ind, self.irrk_inv, self.irrk_count = np.unique(
            self.fbz2irrk, return_index=True, return_inverse=True, return_counts=True
        )
        self.fbz2sym = self._build_fbz2sym()

    def _build_orbital_rotations(
        self,
        nb: int,
        orbital_basis: str | KnownOrbitalBases = None,
        mirror_rotations: dict[KnownSymmetries, np.ndarray] = None,
    ) -> dict[KnownSymmetries, np.ndarray]:
        """
        Constructs orbital rotation matrices for each symmetry operation of this k-grid.

        Inversion symmetries always yield identity matrices since they do not mix orbitals.
        Mirror symmetry rotations are constructed from orbital_basis if provided,
        or from mirror_rotations if provided, or default to identity with a warning.
        orbital_basis takes precedence over mirror_rotations if both are given.

        Parameters
        ----------
        nb:
            Number of orbitals/bands.
        orbital_basis:
            KnownOrbitalBases enum or string specifying the orbital basis.
            Supported: 'eg', 't2g'.
        mirror_rotations:
            dict mapping mirror KnownSymmetries -> unitary (nb, nb) matrix.
            Used if orbital_basis is not provided.
        """
        identity = np.eye(nb, dtype=complex)

        if orbital_basis is not None and orbital_basis != "":
            if isinstance(orbital_basis, str):
                orbital_basis = KnownOrbitalBases.from_string(orbital_basis)
            mirror_rotations = orbital_basis.get_mirror_rotations()

        orbital_rotations = {}
        for sym in self.symmetries:
            if sym in INVERSION_SYMMETRIES:
                orbital_rotations[sym] = identity
            elif sym in MIRROR_SYMMETRIES:
                if mirror_rotations is not None and sym in mirror_rotations:
                    u = mirror_rotations[sym]
                    assert u.shape == (nb, nb), f"Rotation matrix for {sym} must be ({nb},{nb}), got {u.shape}."
                    assert np.allclose(u @ u.conj().T, identity), f"Rotation matrix for {sym} must be unitary."
                    orbital_rotations[sym] = u
                else:
                    warnings.warn(
                        f"No orbital rotation provided for mirror symmetry {sym}. "
                        f"Assuming identity, which is only correct if orbitals are "
                        f"invariant under this mirror. This may cause incorrect results."
                    )
                    orbital_rotations[sym] = identity

        return orbital_rotations

    def specify_orbital_basis(
        self,
        nb: int,
        orbital_basis: str | KnownOrbitalBases = None,
        mirror_rotations: dict[KnownSymmetries, np.ndarray] = None,
    ) -> None:
        """
        Builds orbital_rot_u[k] for every FBZ point k by replaying the exact same
        symmetry reduction sequence used in set_fbz2irrk, composing orbital rotation
        matrices in lockstep.

        For each k, orbital_rot_u[k] is the composed unitary U such that:
            M_ab(k) = U_aa' U*_bb' M_a'b'(k_irr)
        For irreducible points (own representatives), U = identity.

        Parameters
        ----------
        nb:
            Number of orbitals/bands.
        orbital_basis:
            KnownOrbitalBases enum or string specifying the orbital basis.
            Supported: 'eg', 't2g'.
        mirror_rotations:
            dict mapping mirror KnownSymmetries -> unitary (nb, nb) matrix.
            Used if orbital_basis is not provided.
        """
        if nb == 1:
            return

        identity = np.eye(nb, dtype=complex)
        orbital_rotations = self._build_orbital_rotations(nb, orbital_basis, mirror_rotations)

        self.orbital_rot_u = np.tile(identity, (self.nk_tot, 1, 1)).astype(complex)

        nx, ny, nz = self.nk

        def to_flat(ix, iy, iz):
            return (ix % nx) * ny * nz + (iy % ny) * nz + (iz % nz)

        def sym_image_flat(sym, flat_indices):
            """Return the flat index each point maps to under sym."""
            ix = flat_indices // (ny * nz)
            iy = (flat_indices % (ny * nz)) // nz
            iz = flat_indices % nz
            if sym == KnownSymmetries.X_INV:
                return to_flat(-ix, iy, iz)
            elif sym == KnownSymmetries.Y_INV:
                return to_flat(ix, -iy, iz)
            elif sym == KnownSymmetries.Z_INV:
                return to_flat(ix, iy, -iz)
            elif sym == KnownSymmetries.X_Y_SYM:
                return to_flat(iy, ix, iz)
            elif sym == KnownSymmetries.X_Z_SYM:
                return to_flat(iz, iy, ix)
            elif sym == KnownSymmetries.Y_Z_SYM:
                return to_flat(ix, iz, iy)
            elif sym == KnownSymmetries.X_Y_INV:
                return to_flat(-ix, -iy, iz)
            return flat_indices.copy()

        all_flat = np.arange(self.nk_tot)
        current_rep = all_flat.copy()

        for sym in self.symmetries:
            u_sym = orbital_rotations.get(sym, identity)
            images = sym_image_flat(sym, current_rep)  # where current_rep maps to under sym

            # np.minimum logic: update where image is smaller
            changed = images < current_rep

            # Compose orbital rotation for changed points
            self.orbital_rot_u[changed] = self.orbital_rot_u[changed] @ u_sym

            current_rep = np.minimum(current_rep, images)

        assert np.allclose(current_rep, self.fbz2irrk.ravel()), "Orbital basis replay does not match fbz2irrk!"

    def _build_fbz2sym(self) -> np.ndarray:
        """
        For each FBZ point, record the index (+1) of the first symmetry operation
        that maps it away from its original index. 0 = identity.
        """
        fbz2sym = np.zeros(np.prod(self.nk), dtype=int)

        for i_sym, sym in enumerate(self.symmetries):
            test = np.reshape(np.arange(0, np.prod(self.nk)), self.nk)
            apply_symmetry(test, sym)
            test_flat = test.ravel()

            changed = test_flat != np.arange(np.prod(self.nk))
            unrecorded = fbz2sym == 0
            fbz2sym[changed & unrecorded] = i_sym + 1

        return fbz2sym

    def set_irrk_mesh(self) -> None:
        """
        Set the k-meshgrid of the irreducible BZ.
        """
        self.irr_kmesh = np.array([self.kmesh[ax].flatten()[self.irrk_ind] for ax in range(len(self.nk))])

    @property
    def kx_shift(self) -> float:
        r"""
        Returns the kx grid shifted by :math:`\pi` in the half-open interval i.e. :math:`[-\pi,\pi)`.
        """
        return self.kx - np.pi

    @property
    def ky_shift(self) -> float:
        r"""
        Returns the ky grid shifted by :math:`\pi` in the half-open interval i.e. :math:`[-\pi,\pi)`.
        """
        return self.ky - np.pi

    @property
    def kz_shift(self) -> float:
        r"""
        Returns the kz grid shifted by :math:`\pi` in the half-open interval i.e. :math:`[-\pi,\pi)`.
        """
        return self.kz - np.pi

    @property
    def kx_shift_closed(self) -> np.ndarray:
        r"""
        Returns the kx grid shifted by :math:`\pi` in the closed interval i.e. :math:`[-\pi,\pi]`.
        """
        return np.array([*(self.kx - np.pi), -self.kx[0] + np.pi])

    @property
    def ky_shift_closed(self) -> np.ndarray:
        r"""
        Returns the ky grid shifted by :math:`\pi` in the closed interval i.e. :math:`[-\pi,\pi]`.
        """
        return np.array([*(self.ky - np.pi), -self.ky[0] + np.pi])

    @property
    def kz_shift_closed(self) -> np.ndarray:
        r"""
        Returns the kz grid shifted by :math:`\pi` in the closed interval i.e. :math:`[-\pi,\pi]`.
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
        Indices of {kx,ky,kz}.
        Only works for meshes that go from 0 to :math:`2\pi`.
        """
        ind_x = np.arange(0, self.nk[0])
        ind_y = np.arange(0, self.nk[1])
        ind_z = np.arange(0, self.nk[2])
        return np.array(np.meshgrid(ind_x, ind_y, ind_z, indexing="ij"))

    @property
    def kmesh_list(self):
        """
        List of {kx,ky,kz}.
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
        Return list of all q-point indices in the irreducible BZ.
        """
        return np.array([self.kmesh_ind[i].flatten()[self.irrk_ind] for i in range(3)]).T


class KPath:
    """
    Object to generate paths in the Brillouin zone.
    Currently assumed that the BZ grid is from (0,2*pi).
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
        label_map = {l.key: l.latex for l in Labels}
        count = 0
        labels = []
        for k_p in self.ckps:
            key = k_p.strip().lower()
            if key in label_map:
                labels.append(label_map[key])
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
        k_axis_pos = np.zeros(np.sum(self.nkp))
        ds = np.linalg.norm(self.kpts[1:] - self.kpts[:-1], ord=2, axis=1)
        k_axis_pos[1:] = np.cumsum(ds)
        return k_axis_pos / k_axis_pos[-1]

    @property
    def nk_tot(self):
        return np.sum(self.nkp)

    @property
    def nk_seg(self):
        return np.diff(self.cind)

    def get_kpoints(self):
        return np.array(self.k_val).T

    def corner_k_points(self):
        ckp = np.zeros((len(self.ckps), 3))
        label_values = {l.key for l in Labels}
        kpoint_map = {k.name.lower(): np.array(k.value) for k in KnownKPoints}
        for i, kps in enumerate(self.ckps):
            key = kps.strip().lower()
            if key in label_values:
                ckp[i, :] = kpoint_map[key]
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

    def get_bands(self, ek):
        """Return the bands along the k-path"""
        ek_kpath = self.map_to_kpath(ek)
        bands = np.zeros((ek_kpath.current_shape[:-1]))
        for i, eki in enumerate(ek_kpath):
            val, _ = np.linalg.eig(eki)
            bands[i, :] = np.sort(val).real
        return bands


def kpath_segment(k_start, k_end, nk):
    nkp = int(np.round(np.linalg.norm(k_start * nk - k_end * nk, ord=np.inf)))
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

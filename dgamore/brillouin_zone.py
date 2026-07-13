# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

"""
Module to handle operations within the (irreducible) Brillouin zone. Heavily inspired by DGApy.
"""

import warnings
from enum import Enum

import numpy as np


class KnownSymmetries(Enum):
    r"""
    Known symmetries of the Brillouin zone.

    :cvar X_INV: Inversion along :math:`k_x`.
    :cvar Y_INV: Inversion along :math:`k_y`.
    :cvar Z_INV: Inversion along :math:`k_z`.
    :cvar X_Y_SYM: Exchange symmetry between :math:`k_x` and :math:`k_y`.
    :cvar X_Z_SYM: Exchange symmetry between :math:`k_x` and :math:`k_z`.
    :cvar Y_Z_SYM: Exchange symmetry between :math:`k_y` and :math:`k_z`.
    :cvar X_Y_INV: Simultaneous inversion of :math:`k_x` and :math:`k_y`.
    :cvar AUTO: Deferral flag - discover the symmetry group from a Hamiltonian ``H(k)`` at runtime via
        :meth:`KGrid.specify_auto_symmetries`, rather than a fixed spatial operation.
    """

    X_INV = "x-inv"
    Y_INV = "y-inv"
    Z_INV = "z-inv"
    X_Y_SYM = "x-y-sym"
    X_Z_SYM = "x-z-sym"
    Y_Z_SYM = "y-z-sym"
    X_Y_INV = "x-y-inv"
    AUTO = "auto"


class KnownKPoints(Enum):
    r"""
    Known high-symmetry k-points in the Brillouin zone, as fractional coordinates.

    :cvar GAMMA: Zone center :math:`\Gamma = (0, 0, 0)`.
    :cvar X: :math:`(0.5, 0, 0)`.
    :cvar Y: :math:`(0, 0.5, 0)`.
    :cvar Z: :math:`(0, 0, 0.5)`.
    :cvar M: :math:`(0.5, 0.5, 0)`.
    :cvar M2: :math:`(0.25, 0.25, 0)`.
    :cvar R: :math:`(0.5, 0, 0.5)`.
    :cvar A: :math:`(0.5, 0.5, 0.5)`.
    :cvar T: :math:`(0, 0.5, 0.5)`.
    """

    GAMMA = (0.0, 0.0, 0.0)
    X = (0.5, 0.0, 0.0)
    Y = (0.0, 0.5, 0.0)
    Z = (0.0, 0.0, 0.5)
    M = (0.5, 0.5, 0.0)
    M2 = (0.25, 0.25, 0.0)
    R = (0.5, 0.0, 0.5)
    A = (0.5, 0.5, 0.5)
    T = (0.0, 0.5, 0.5)


class Labels(Enum):
    r"""
    Plot labels for the high-symmetry k-points, each a ``(key, latex)`` pair (lookup key and rendered LaTeX label).

    :cvar GAMMA: The :math:`\Gamma` point label.
    :cvar X: The X point label.
    :cvar Y: The Y point label.
    :cvar Z: The Z point label.
    :cvar M: The M point label.
    :cvar M2: The M2 point label.
    :cvar R: The R point label.
    :cvar A: The A point label.
    :cvar T: The T point label.
    """

    GAMMA = ("gamma", r"$\Gamma$")
    X = ("x", "X")
    Y = ("y", "Y")
    Z = ("z", "Z")
    M = ("m", "M")
    M2 = ("m2", "M2")
    R = ("r", "R")
    A = ("a", "A")
    T = ("t", "T")

    @property
    def key(self):
        """
        The lowercase lookup key of the label.

        :return: The lowercase string key of the label (first tuple element).
        """
        return self.value[0]

    @property
    def latex(self):
        """
        The LaTeX/plot string of the label.

        :return: The LaTeX/plot label string (second tuple element).
        """
        return self.value[1]

    @staticmethod
    def from_string(s: str):
        """
        Looks up a :class:`Labels` member by its string key (case-insensitive).

        :param s: The label key to look up.
        :return: The matching :class:`Labels` member.
        :raises ValueError: If no label matches ``s``.
        """
        s = s.strip().lower()
        for label in Labels:
            if s == label.key:
                return label
        raise ValueError(f"Unknown label string: {s}")


def two_dimensional_square_symmetries() -> list[KnownSymmetries]:
    """
    Returns the standard symmetry set of a two-dimensional square lattice.

    :return: The lattice symmetries of a two-dimensional square lattice.
    """
    return [KnownSymmetries.X_INV, KnownSymmetries.Y_INV, KnownSymmetries.X_Y_SYM]


def three_dimensional_cubic_symmetries() -> list[KnownSymmetries]:
    """
    Returns the standard symmetry set of a three-dimensional cubic lattice.

    :return: The lattice symmetries of a three-dimensional cubic lattice.
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
    Returns the standard symmetry set of a two-dimensional nematic lattice.

    :return: The lattice symmetries of a two-dimensional nematic lattice.
    """
    return [KnownSymmetries.X_INV, KnownSymmetries.Y_INV]


def quasi_two_dimensional_square_symmetries() -> list[KnownSymmetries]:
    """
    Returns the standard symmetry set of a quasi-two-dimensional square lattice.

    :return: The lattice symmetries of a quasi-two-dimensional square lattice.
    """
    return [KnownSymmetries.X_INV, KnownSymmetries.Y_INV, KnownSymmetries.Z_INV, KnownSymmetries.X_Y_SYM]


def quasi_one_dimensional_square_symmetries() -> list[KnownSymmetries]:
    """
    Returns the standard symmetry set of a quasi-one-dimensional square lattice.

    :return: The lattice symmetries of a quasi-one-dimensional square lattice.
    """
    return [KnownSymmetries.X_INV, KnownSymmetries.Y_INV]


def simultaneous_x_y_inversion() -> list[KnownSymmetries]:
    """
    Returns the symmetry set for a simultaneous x-and-y inversion.

    :return: The symmetry list for a simultaneous inversion in the x and y directions.
    """
    return [KnownSymmetries.X_Y_INV]


def inv_sym(mat: np.ndarray, axis) -> None:
    r"""
    Applies an inversion symmetry along ``axis`` to ``mat`` in place, assuming the grid runs over :math:`[0, 2\pi)`
    so that the zero point does not map.

    :param mat: The (at least 3D) array to symmetrize in place; the leading three axes are the momentum axes.
    :param axis: The momentum axis (0, 1 or 2) to invert.
    :return: None.
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
    Applies the x-y reflection symmetry to ``mat`` in place (see :func:`_pairwise_sym`).

    :param mat: The (at least 3D) array to symmetrize in place.
    :return: None.
    """
    _pairwise_sym(mat, 0, 1)


def x_z_sym(mat: np.ndarray) -> None:
    """
    Applies the x-z reflection symmetry to ``mat`` in place (see :func:`_pairwise_sym`).

    :param mat: The (at least 3D) array to symmetrize in place.
    :return: None.
    """
    _pairwise_sym(mat, 0, 2)


def y_z_sym(mat: np.ndarray) -> None:
    """
    Applies the y-z reflection symmetry to ``mat`` in place (see :func:`_pairwise_sym`).

    :param mat: The (at least 3D) array to symmetrize in place.
    :return: None.
    """
    _pairwise_sym(mat, 1, 2)


def _pairwise_sym(mat: np.ndarray, axis_a: int, axis_b: int) -> None:
    """
    Symmetrizes ``mat`` in place under the swap of two momentum axes by taking the element-wise minimum of the array
    and its axis-swapped version (used to collapse equivalent points to a single representative index). Does nothing
    (with a warning) if the two axes have different lengths.

    :param mat: The (at least 3D) array to symmetrize in place.
    :param axis_a: First momentum axis to swap.
    :param axis_b: Second momentum axis to swap.
    :return: None.
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
    Applies the simultaneous x-and-y inversion symmetry to ``mat`` in place.

    :param mat: The (at least 3D) array to symmetrize in place.
    :return: None.
    """
    assert mat.ndim >= 3, f"dim(mat) = {mat.ndim} but must be at least 3 dimensional"
    len_ax_x = mat.shape[0] // 2
    mod_2_x = mat.shape[0] % 2
    mat[len_ax_x + 1 :, 1:, :, ...] = mat[1 : len_ax_x + mod_2_x, 1:, :][::-1, ::-1, :, ...]


def apply_symmetry(mat: np.ndarray, sym: KnownSymmetries) -> None:
    """
    Applies a single known lattice symmetry to ``mat`` in place.

    :param mat: The (at least 3D) array to symmetrize in place.
    :param sym: The :class:`KnownSymmetries` operation to apply.
    :return: None.
    """
    assert sym in KnownSymmetries, f"sym = {sym} not in known symmetries {KnownSymmetries}."
    if sym == KnownSymmetries.AUTO:
        return  # deferral flag, not a spatial operation (symmetries are discovered by specify_auto_symmetries)
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
    Applies a sequence of lattice symmetries to ``mat`` in place (see :func:`apply_symmetry`).

    :param mat: The (at least 3D) array to symmetrize in place.
    :param symmetries: The list of :class:`KnownSymmetries` to apply in order (empty/None is a no-op).
    :return: None.
    """
    assert mat.ndim >= 3, f"dim(mat) = {mat.ndim} but must at least 3 dimensional"
    if not symmetries:
        return
    for sym in symmetries:
        apply_symmetry(mat, sym)


def get_lattice_symmetries_from_string(symmetry_string: str | tuple | list) -> list[KnownSymmetries]:
    """
    Returns the lattice symmetries from a string.

    The special string ``"auto"`` signals that symmetries should be auto-detected from a Hamiltonian ``H(k)`` at
    runtime via :meth:`specify_auto_symmetries`; it resolves to ``[KnownSymmetries.AUTO]``, which the KGrid recognizes
    to defer building ``fbz2irrk`` until :meth:`specify_auto_symmetries` is called.

    :param symmetry_string: A named preset (e.g. ``"two_dimensional_square"``), the special ``"auto"``, an empty
        string/``"none"``, or a list/tuple (or its string repr) of :class:`KnownSymmetries` values.
    :return: The corresponding list of :class:`KnownSymmetries` (``[KnownSymmetries.AUTO]`` for ``"auto"``, empty for
        none).
    :raises ValueError: If the string cannot be parsed as a known preset or a Python literal.
    :raises NotImplementedError: If a listed symmetry is not a known symmetry.
    """
    if not symmetry_string:
        return []

    if isinstance(symmetry_string, str):
        symmetry_string = symmetry_string.lower()
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
        elif symmetry_string == "auto":
            # KGrid recognizes the AUTO flag and defers symmetry reduction until specify_auto_symmetries(hk) is called.
            return [KnownSymmetries.AUTO]
        elif symmetry_string == "" or symmetry_string == "none":
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


def is_auto_symmetries(symmetries) -> bool:
    """
    Tests whether a symmetries value requests auto-detection (i.e. carries the :attr:`KnownSymmetries.AUTO` flag).

    :param symmetries: A symmetries value (a list/tuple of :class:`KnownSymmetries`, or anything else).
    :return: True if ``symmetries`` is a list/tuple containing :attr:`KnownSymmetries.AUTO`.
    """
    return isinstance(symmetries, (list, tuple)) and KnownSymmetries.AUTO in symmetries


class KGrid:
    """
    Class to build the k-grid for the Brillouin zone.

    The ``symmetries`` argument accepts the usual list of ``KnownSymmetries`` *and* the special "auto" mode (passed as
    ``[KnownSymmetries.AUTO]``, typically obtained from ``get_lattice_symmetries_from_string("auto")``). In auto mode
    the symmetry group is discovered from a Hamiltonian ``H(k)`` at runtime: instantiate the grid with the AUTO flag,
    then call :meth:`specify_auto_symmetries` with the Hamiltonian. Until that call the grid behaves as if no
    symmetries were applied (full BZ = IBZ).
    """

    def __init__(self, nk: tuple = None, symmetries: list[KnownSymmetries] = None):
        """
        Builds the k-axes and the irreducible-BZ maps from the grid size and symmetries.

        :param nk: Number of k-points per spatial direction, as a tuple ``(nx, ny, nz)``.
        :param symmetries: A list of :class:`KnownSymmetries` defining the irreducible BZ, or
            ``[KnownSymmetries.AUTO]`` to defer symmetry discovery to :meth:`specify_auto_symmetries`.
        """
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

        # Auto-discovered symmetry data, populated by specify_auto_symmetries(); when set, _map_to_full_bz uses these
        # to apply the per-k orbital transformation.
        self._auto_mode = is_auto_symmetries(symmetries)
        self._auto_us = None  # shape (nx, ny, nz, nb, nb), complex
        self._auto_sigmas = None  # shape (nx, ny, nz), float (+/-1)
        self._auto_conjs = None  # shape (nx, ny, nz), bool

        self.nk = nk
        self.set_k_axes()

        self.set_fbz2irrk()
        self.set_irrk_maps()
        self.set_irrk_mesh()

    def set_fbz2irrk(self) -> None:
        """
        Builds the full-BZ-to-irreducible-BZ index field by applying the lattice symmetries to the flat index grid.

        :return: None.
        """
        self.fbz2irrk = np.reshape(np.arange(0, np.prod(self.nk)), self.nk)
        apply_symmetries(self.fbz2irrk, self.symmetries)

    def set_irrk_maps(self) -> None:
        """
        Derives the irreducible-BZ index list, the inverse map back to the full BZ, the multiplicities, and the
        per-point symmetry map from ``fbz2irrk``.

        :return: None.
        """
        _, self.irrk_ind, self.irrk_inv, self.irrk_count = np.unique(
            self.fbz2irrk, return_index=True, return_inverse=True, return_counts=True
        )
        self.fbz2sym = self._build_fbz2sym()

    def specify_auto_symmetries(
        self,
        hk: np.ndarray,
        atol: float = 1e-8,
        verbose: bool = False,
        include_antiunitary: bool = False,
    ) -> None:
        r"""
        Auto-detects the symmetry group of the Hamiltonian ``hk`` and replay
        the IBZ reduction onto this grid.

        Only applicable when this ``KGrid`` was constructed in auto mode
        (``symmetries`` contains ``KnownSymmetries.AUTO``). Discovers all
        operations ``(M, q, U, sigma, conj)`` that leave H(k) invariant, then
        repopulates ``fbz2irrk``, ``irrk_ind``, ``irrk_inv``, ``irrk_count``,
        ``irr_kmesh``, and stores per-k transformation data used by
        ``_map_to_full_bz`` to apply the orbital transformation when expanding
        IBZ -> FBZ.

        :param hk: Complex Hermitian Hamiltonian of shape ``(nk[0], nk[1], nk[2], nb, nb)`` indexed on the same grid
            as this KGrid, with axes along the primitive reciprocal-lattice basis (fractional coordinate along
            :math:`b_i`).
        :param atol: Absolute tolerance for symmetry validation.
        :param verbose: If True, print diagnostics about the discovered group.
        :param include_antiunitary: If False (default), anti-unitary symmetries (``conj=True``, i.e.
            :math:`H(k) = H(k)^*`-style time-reversal) are dropped after discovery. They are valid symmetries of H
            but, for frequency-dependent quantities, additionally require a Matsubara-frequency flip that
            ``_map_to_full_bz`` does not perform; keeping ``False`` makes the IBZ-to-FBZ expansion safe for any object
            with the same lattice symmetry as H, at the cost of a possibly larger IBZ.
        :return: None.
        :raises RuntimeError: If the grid was not constructed in auto mode.
        :raises ValueError: If ``hk``'s shape does not match the grid or is not ``(nx, ny, nz, nb, nb)``.
        """
        if not self._auto_mode:
            raise RuntimeError(
                "specify_auto_symmetries() may only be called when the KGrid "
                "was constructed in auto mode (symmetries='auto')."
            )
        if hk.shape[:3] != tuple(self.nk):
            raise ValueError(f"Hamiltonian k-grid shape {hk.shape[:3]} does not match KGrid {tuple(self.nk)}.")
        if hk.ndim != 5 or hk.shape[3] != hk.shape[4]:
            raise ValueError(f"Hamiltonian must have shape (nx, ny, nz, nb, nb); got {hk.shape}.")

        # Late import to keep the dependency optional at module load time and
        # to avoid a circular import if symmetry_reduction ever imports from here.
        from dgamore.symmetry_reduction import get_symmetry_reduction

        res = get_symmetry_reduction(
            np.asarray(hk, dtype=np.complex128),
            atol=atol,
            verbose=verbose,
            include_antiunitary=include_antiunitary,
        )

        # Refresh the IBZ maps from the discovered orbits. ``fbz2irrk`` is the (nx,ny,nz) flat-index field; np.unique
        # on it recovers irrk_ind/irrk_inv/irrk_count exactly as the rest of the code expects.
        self.fbz2irrk = res["fbz2irrk"].astype(self.fbz2irrk.dtype, copy=True)
        _, self.irrk_ind, self.irrk_inv, self.irrk_count = np.unique(
            self.fbz2irrk, return_index=True, return_inverse=True, return_counts=True
        )
        self.set_irrk_mesh()

        # Stash per-k transformation tensors for use by _map_to_full_bz.
        self._auto_us = res["Us"]
        self._auto_sigmas = res["sigmas"]
        self._auto_conjs = res["conjs"]

        # fbz2sym is kept as built by the trivial set_fbz2irrk path on the auto sentinel; it is not consumed by
        # _map_to_full_bz in auto mode and remains for backwards compatibility only.

    @property
    def is_auto(self) -> bool:
        """
        Whether auto-discovered symmetry data is available on this grid.

        :return: True if this KGrid is in auto-discovered symmetry mode and :meth:`specify_auto_symmetries` has
            populated the transformation data.
        """
        return self._auto_mode and self._auto_us is not None

    def _build_fbz2sym(self) -> np.ndarray:
        """
        For each full-BZ point, records the index (+1) of the first symmetry operation that moves it away from its
        own index (0 means identity / no operation moved it).

        :return: The per-point symmetry-operation index array, flattened over the full BZ.
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
        Builds and stores the k-mesh restricted to the irreducible BZ.

        :return: None.
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
        The three k-axis arrays of the grid.

        :return: The k-grid as the tuple of axis arrays ``(kx, ky, kz)``.
        """
        return self.kx, self.ky, self.kz

    @property
    def nk_tot(self):
        """
        The total number of full-BZ k-points.

        :return: The total number of k-points in the full BZ.
        """
        return np.prod(self.nk)

    @property
    def nk_irr(self) -> int:
        """
        The number of irreducible-BZ k-points.

        :return: The number of k-points in the irreducible BZ.
        """
        return np.size(self.irrk_ind)

    @property
    def kmesh(self) -> np.ndarray:
        """
        The momentum meshgrid over the full BZ.

        :return: The meshgrid of ``{kx, ky, kz}`` (shape ``[3, nx, ny, nz]``, ``"ij"`` indexing).
        """
        return np.array(np.meshgrid(self.kx, self.ky, self.kz, indexing="ij"))

    @property
    def kmesh_ind(self) -> np.ndarray:
        r"""
        The integer index meshgrid over the full BZ.

        :return: The integer index meshgrid of ``{kx, ky, kz}``. Only valid for meshes spanning :math:`[0, 2\pi)`.
        """
        ind_x = np.arange(0, self.nk[0])
        ind_y = np.arange(0, self.nk[1])
        ind_z = np.arange(0, self.nk[2])
        return np.array(np.meshgrid(ind_x, ind_y, ind_z, indexing="ij"))

    @property
    def kmesh_list(self):
        """
        The flattened momentum meshgrid.

        :return: The k-meshgrid flattened to shape ``[3, nk_tot]``.
        """
        return self.kmesh.reshape((3, -1))

    def set_k_axes(self) -> None:
        r"""
        Builds the three k-axis arrays spanning :math:`[0, 2\pi)` for the full BZ.

        :return: None.
        """
        self.kx = np.linspace(0, 2 * np.pi, self.nk[0], endpoint=False)
        self.ky = np.linspace(0, 2 * np.pi, self.nk[1], endpoint=False)
        self.kz = np.linspace(0, 2 * np.pi, self.nk[2], endpoint=False)

    def get_q_list(self) -> np.ndarray:
        """
        Lists the integer index triplets of all full-BZ q-points.

        :return: The integer index triplets of all q-points in the full BZ, shape ``[nk_tot, 3]``.
        """
        return np.array([self.kmesh_ind[i].flatten() for i in range(3)]).T

    def get_irrq_list(self) -> np.ndarray:
        """
        Lists the integer index triplets of all irreducible-BZ q-points.

        :return: The integer index triplets of all q-points in the irreducible BZ, shape ``[nk_irr, 3]``.
        """
        return np.array([self.kmesh_ind[i].flatten()[self.irrk_ind] for i in range(3)]).T


class KPath:
    """
    Object to generate paths in the Brillouin zone.
    It is currently assumed that the BZ grid is from (0,2*pi).
    """

    def __init__(self, nk, path, kx=None, ky=None, kz=None, path_deliminator="-"):
        r"""
        Builds the k-axes and the discretized path (and its k-points) from the path string.

        :param nk: Number of k-points per spatial direction, as a tuple ``(nx, ny, nz)``.
        :param path: The desired path through the BZ as a delimiter-separated string of corner-point labels.
        :param kx: Optional explicit kx-axis array; a :math:`[0, 2\pi)` grid is built if None.
        :param ky: Optional explicit ky-axis array; a :math:`[0, 2\pi)` grid is built if None.
        :param kz: Optional explicit kz-axis array; a :math:`[0, 2\pi)` grid is built if None.
        :param path_deliminator: The delimiter separating corner-point labels in ``path``.
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
        """
        Maps the path indices to their k-axis values.

        :return: The k-axis values along the path as a list ``[kx_vals, ky_vals, kz_vals]``.
        """
        k = [self.kx[self.kpts[:, 0]], self.ky[self.kpts[:, 1]], self.kz[self.kpts[:, 2]]]
        return k

    def set_kgrid(self, k_in, nk):
        r"""
        Returns an explicit k-axis if given, otherwise builds a :math:`[0, 2\pi)` grid of ``nk`` points.

        :param k_in: Explicit k-axis array, or None to build a default grid.
        :param nk: Number of points in the default grid.
        :return: The k-axis array.
        """
        if k_in is None:
            k = np.linspace(0, np.pi * 2, nk, endpoint=False)
        else:
            k = k_in
        return k

    @property
    def ckps(self):
        """
        The corner-point label strings of the path.

        :return: The list of corner-point label strings obtained by splitting ``path``.
        """
        return self.path.split(self.path_deliminator)

    @property
    def labels(self):
        """
        The plot labels for the path corner points.

        :return: The plot labels (LaTeX where known) for the corner points along the path.
        """
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
        """
        The x-axis tick positions at the path corner points.

        :return: The x-axis tick positions (at the corner points) for plotting along the path.
        """
        return self.k_axis[self.cind]

    @property
    def cind(self):
        """
        The corner-point indices within the concatenated path.

        :return: The indices of the corner points within the concatenated path.
        """
        return np.concatenate(([0], np.cumsum(self.nkp) - 1))

    @property
    def ikx(self):
        """
        The kx index of each path point.

        :return: The integer kx index of each point along the path.
        """
        return self.kpts[:, 0]

    @property
    def iky(self):
        """
        The ky index of each path point.

        :return: The integer ky index of each point along the path.
        """
        return self.kpts[:, 1]

    @property
    def ikz(self):
        """
        The kz index of each path point.

        :return: The integer kz index of each point along the path.
        """
        return self.kpts[:, 2]

    @property
    def k_axis(self):
        """
        The normalized arc-length coordinate along the path.

        :return: The cumulative arc-length coordinate of each path point, normalized to ``[0, 1]`` (for plotting).
        """
        k_axis_pos = np.zeros(np.sum(self.nkp))
        ds = np.linalg.norm(self.kpts[1:] - self.kpts[:-1], ord=2, axis=1)
        k_axis_pos[1:] = np.cumsum(ds)
        return k_axis_pos / k_axis_pos[-1]

    @property
    def nk_tot(self):
        """
        The total number of path points.

        :return: The total number of points along the path.
        """
        return np.sum(self.nkp)

    @property
    def nk_seg(self):
        """
        The number of points per path segment.

        :return: The number of points in each path segment between corner points.
        """
        return np.diff(self.cind)

    def get_kpoints(self):
        """
        Stacks the path k-axis values into a coordinate array.

        :return: The path k-points as an array of shape ``[nk_tot, 3]``.
        """
        return np.array(self.k_val).T

    def corner_k_points(self):
        """
        Resolves the corner-point labels of the path to their fractional k-coordinates (known labels via
        :class:`KnownKPoints`, otherwise parsed from the string).

        :return: The corner k-points as an array of shape ``[n_corners, 3]``.
        """
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
        """
        Selects the values of a BZ-gridded array along the k-path.

        :param mat: Array indexed as ``[kx, ky, kz, ...]`` over the full BZ.
        :return: The array restricted to the path points (leading axis runs along the path).
        """
        return mat[self.ikx, self.iky, self.ikz, ...]

    def build_k_path(self):
        """
        Builds the discretized k-path by concatenating the segments between consecutive corner points.

        :return: A tuple ``(k_path, nkp)`` of the integer path-index array and the per-segment point counts.
        """
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
        """
        Diagonalizes the band dispersion at each path point to obtain the (sorted, real) band energies.

        :param ek: The band dispersion indexed as ``[kx, ky, kz, o1, o2]``.
        :return: The band energies along the path, shape ``[nk_path, n_bands]``.
        """
        ek_kpath = self.map_to_kpath(ek)
        bands = np.zeros((ek_kpath.current_shape[:-1]))
        for i, eki in enumerate(ek_kpath):
            val, _ = np.linalg.eig(eki)
            bands[i, :] = np.sort(val).real
        return bands


def kpath_segment(k_start, k_end, nk):
    """
    Builds the integer k-index points of a straight segment between two fractional corner points, wrapping indices
    back into the grid.

    :param k_start: Fractional coordinates of the segment start point.
    :param k_end: Fractional coordinates of the segment end point.
    :param nk: Number of k-points per spatial direction.
    :return: A tuple ``(k_segment, nkp)`` of the integer index array along the segment and the number of points.
    """
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
    """
    Parses a whitespace-separated coordinate string into a fractional k-point.

    :param string: A string of space-separated floats, e.g. ``"0.5 0.5 0.0"``.
    :return: The parsed coordinates as a numpy array.
    """
    scoords = string.split(" ")
    coords = np.array([float(sc) for sc in scoords])
    return coords

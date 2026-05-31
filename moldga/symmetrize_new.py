# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# moLDGA — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#          Eliashberg Equation Solver for Strongly Correlated Electron Systems

import gc
import glob
import itertools as it
import os
import re
import readline
from pathlib import Path

import h5py
import numpy as np


def index2component_general(num_bands: int, n_dims: int, ind: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the band and spin components corresponding to a compound index for four-legged objects.
    """
    bandspin = np.zeros(n_dims, dtype=np.int_)
    spin = np.zeros(n_dims, dtype=np.int_)
    band = np.zeros(n_dims, dtype=np.int_)
    ind_tmp = ind - 1
    tmp = (2 * num_bands) ** np.arange(n_dims, -1, -1)

    for i in range(n_dims):
        bandspin[i] = ind_tmp // tmp[i + 1]
        spin[i] = bandspin[i] % 2
        band[i] = bandspin[i] // 2
        ind_tmp -= tmp[i + 1] * bandspin[i]

    return bandspin, band, spin


def index2component_general_2dims(num_bands: int, ind: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the band and spin components corresponding to a compound index for two-legged objects.
    """
    return index2component_general(num_bands, 2, ind)


def index2component_general_4dims(num_bands: int, ind: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the band and spin components corresponding to a compound index for four-legged objects.
    """
    return index2component_general(num_bands, 4, ind)


def component2index_general(num_bands: int, n_dims: int, bands: list, spins: list) -> int:
    """
    Computes a compound index from band and spin indices for a n_dims-legged object.
    """
    assert num_bands > 0, "Number of bands has to be set to non-zero positive integers."

    n_spins = 2
    dims_bs = n_dims * (num_bands * n_spins,)
    dims_1 = (num_bands, n_spins)

    bandspin = np.ravel_multi_index((bands, spins), dims_1)
    return np.ravel_multi_index(bandspin, dims_bs) + 1


def component2index_general_2dims(num_bands: int, bands: list, spins: list) -> int:
    """
    Computes a compound index from band and spin indices for a two-legged object.
    """
    return component2index_general(num_bands, 2, bands, spins)


def component2index_general_4dims(num_bands: int, bands: list, spins: list) -> int:
    """
    Computes a compound index from band and spin indices for a four-legged object.
    """
    return component2index_general(num_bands, 4, bands, spins)


def index2component_band(num_bands: int, n_dims: int, ind: int) -> list:
    """
    Computes only orbital indices from a compound index for four-legged objects.
    """
    b = []
    ind_tmp = ind - 1
    for i in range(n_dims):
        b.append(ind_tmp // (num_bands ** (n_dims - i - 1)))
        ind_tmp = ind_tmp - b[i] * (num_bands ** (n_dims - i - 1))
    return b


def component2index_band(num_bands: int, n_dims: int, b: list) -> int:
    """
    Computes a compound index from only orbital indices for four-legged objects.
    """
    ind = 1
    for i in range(n_dims):
        ind = ind + num_bands ** (n_dims - i - 1) * b[i]
    return ind


def get_worm_components_2dims(num_bands: int, orbs: list[list[int]]) -> list[int]:
    """
    Returns the worm components for two-legged objects.
    """
    spins = [0, 0], [1, 1]
    component_indices = []
    for o in orbs:
        for s in spins:
            component_indices.append(int(component2index_general_2dims(num_bands, o, s)))
    return sorted(component_indices)


def get_worm_components_all_2dims(num_bands: int) -> list[int]:
    """
    Returns the list of worm components for a given number of bands for two-legged objects,
    where only relevant spin combinations for the
    density and magnetic channels in the case of SU(2) symmetry are picked.
    """
    orbs = [list(orb) for orb in it.product(range(num_bands), repeat=2)]
    return get_worm_components_2dims(num_bands, orbs)


def get_worm_components_partial_2dims(num_bands: int) -> list[int]:
    """
    Returns the list of worm components for a given number of bands for two-legged objects,
    where only relevant spin combinations for the
    density and magnetic channels in the case of SU(2) symmetry are picked.
    It only lists orbital-diagonal components.
    """
    orbs = [[orb, orb] for orb in range(num_bands)]
    return get_worm_components_2dims(num_bands, orbs)


def get_worm_components_4dims(num_bands: int, orbs: list[list[int]]) -> list[int]:
    """
    Returns the worm components for 4-legged objects.
    """
    spins = [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]
    component_indices = []
    for o in orbs:
        for s in spins:
            component_indices.append(int(component2index_general_4dims(num_bands, o, s)))
    return sorted(component_indices)


def get_worm_components_all_4dims(num_bands: int) -> list[int]:
    """
    Returns the list of worm components for a given number of bands for four-legged objecst,
    where only relevant spin combinations for the
    density and magnetic channels in the case of SU(2) symmetry are picked.
    """
    orbs = [list(orb) for orb in it.product(range(num_bands), repeat=4)]
    return get_worm_components_4dims(num_bands, orbs)


def get_worm_components_partial_4dims(num_bands: int) -> list[int]:
    """
    Returns the list of worm components for a given number of bands for four-legged objects,
    where only relevant spin combinations for the
    density and magnetic channels in the case of SU(2) symmetry are picked.
    It only lists worm components where the orbitals are not of type ijjj, jijj, jjij or jjji.
    """
    orbs = [
        list(orb)
        for orb in it.product(range(num_bands), repeat=4)
        if not (
            orb[1] == orb[2] == orb[3] != orb[0]  # ijjj
            or orb[0] == orb[2] == orb[3] != orb[1]  # jijj
            or orb[0] == orb[1] == orb[3] != orb[2]  # jjij
            or orb[0] == orb[1] == orb[2] != orb[3]  # jjji
        )
    ]
    return get_worm_components_4dims(num_bands, orbs)


def extract_g2_general(group_string: str, indices: list, file: h5py.File, niw: int, niv: int) -> tuple:
    r"""
    Extracts the two-particle Green's function components from the vertex file for given indices and group string.
    Returns the components :math:`G2_{\uparrow\uparrow\uparrow\uparrow}, G2_{\downarrow\downarrow\downarrow\downarrow}`,
    :math:`G2_{\downarrow\downarrow\uparrow\uparrow}, G2_{\uparrow\uparrow\downarrow\downarrow}`,
    :math:`G2_{\uparrow\downarrow\downarrow\uparrow}` and :math:`G2_{\downarrow\uparrow\uparrow\downarrow}`.
    """
    print(f"Nonzero number of elements of G2 in dataset: {len(indices)} / {n_bands ** 4 * 2**4}")

    elements = np.array([file[f"{group_string}/{idx}/value"] for idx in indices])
    elements = elements.transpose(0, -1, 1, 2)

    # for some reason, the elements are stored transposed in vv' in symmetrize_old.py
    # therefore, every time we have to read or write, we have to transpose in vv' (this is a different transpose than above)
    elements = elements.transpose(0, 1, 3, 2)

    # construct G2_dens and G2_magn for the output file
    bands, spins = zip(*(index2component_general(n_bands, 4, int(i))[1:3] for i in indices))

    # since we are SU(2) symmetric, we only have to pick out the elements where the spin is either
    # [0,0,0,0] or [1,1,1,1] for uu component, [0,0,1,1] or [1,1,0,0] for ud component and [0,1,1,0] or [1,0,0,1] for ud_bar component
    g2_uuuu, g2_dddd, g2_dduu, g2_uudd, g2_uddu, g2_duud = (
        np.zeros((n_bands, n_bands, n_bands, n_bands, 2 * niw + 1, 2 * niv, 2 * niv), dtype=np.complex64)
        for _ in range(6)
    )

    spin_dddd, spin_uuuu = [0, 0, 0, 0], [1, 1, 1, 1]
    spin_dduu, spin_uudd = [0, 0, 1, 1], [1, 1, 0, 0]
    spin_uddu, spin_duud = [1, 0, 0, 1], [0, 1, 1, 0]

    for a, b, c, d in it.product(range(n_bands), repeat=4):
        target_orbital = [a, b, c, d]
        print(f"Collecting G2 for orbitals {[t+1 for t in target_orbital]} ...")

        idx_dddd, idx_dduu, idx_uudd, idx_uuuu, idx_uddu, idx_duud = (
            next(
                (
                    idx
                    for idx, band in enumerate(bands)
                    if np.array_equal(band, target_orbital) and np.array_equal(spins[idx], spin)
                ),
                None,
            )
            for spin in (spin_dddd, spin_dduu, spin_uudd, spin_uuuu, spin_uddu, spin_duud)
        )

        if None in (idx_dddd, idx_dduu, idx_uudd, idx_uuuu, idx_uddu, idx_duud):
            continue

        for g2, idx in zip(
            (g2_uuuu, g2_dddd, g2_dduu, g2_uudd, g2_uddu, g2_duud),
            (idx_uuuu, idx_dddd, idx_dduu, idx_uudd, idx_uddu, idx_duud),
        ):
            g2[a, b, c, d] = elements[idx]

    return g2_uuuu, g2_dddd, g2_dduu, g2_uudd, g2_uddu, g2_duud


def save_to_file(g2_list: list[np.ndarray], names: list[str], niw: int, nb: int, ineq: int):
    """
    Saves the given g2 to the output file.
    """
    assert len(g2_list) == len(names)
    for wn in range(2 * niw + 1):
        for i, j, k, l in it.product(range(nb), repeat=4):
            idx = component2index_band(nb, 4, [i, j, k, l])
            for g2, name in zip(g2_list, names):
                output_file[f"ineq-{ineq:03}/{name}/{wn:05}/{idx:05}/value"] = g2[i, j, k, l, wn].transpose()


def get_niw_niv(vertex_file, g4iw_groupstring, indices):
    """
    Determines niw and niv from the shape of the first element in the vertex file.
    """
    first_element_shape = vertex_file[f"{g4iw_groupstring}/{indices[0]}/value"].shape
    assert first_element_shape[0] % 2 == 0, "The number of fermionic frequencies has to be even."
    assert first_element_shape[-1] % 2 != 0, "The number of bosonic frequencies has to be odd."
    return first_element_shape[-1] // 2, first_element_shape[0] // 2


def complete(text, state):
    expanded = os.path.expanduser(text)
    matches = [m + "/" if os.path.isdir(m) else m for m in glob.glob(expanded + "*")]

    try:
        return matches[state]
    except IndexError:
        return None


if __name__ == "__main__":
    default_filename = "Vertex.hdf5"
    default_output_filename = "g4iw_sym.hdf5"

    readline.parse_and_bind("tab: complete")
    readline.set_completer_delims(" \t\n;")
    readline.set_completer(complete)

    input_filename = input(f"Enter the DMFT vertex file name (default = {default_filename}): ")
    output_filename = input(f"Enter the output filename (default = {default_output_filename}): ")

    input_filename = input_filename.strip() if input_filename else default_filename
    output_filename = output_filename.strip() if output_filename else default_output_filename

    input_filename = str(Path(input_filename).with_suffix(".hdf5"))
    output_filename = str(Path(output_filename).with_suffix(".hdf5"))

    vertex_file = h5py.File(input_filename, "r")
    output_file = h5py.File(output_filename, "w")

    group = vertex_file["worm-last"]

    ineq_numbers = []
    for key in group.keys():
        match = re.match(r"ineq-(\d+)", key)
        if match:
            ineq_numbers.append(int(match.group(1)))

    ineq_numbers.sort()

    n_bands = int(vertex_file[".config"].attrs[f"atoms.1.nd"])

    for ineq in ineq_numbers:
        print("-----------------------------------------")
        print("Processing inequivalent atom number:", ineq)
        print("-----------------------------------------")
        g4iw_groupstring = f"worm-last/ineq-{ineq:03}/g4iw-worm"

        indices = None
        try:
            indices = list(vertex_file[g4iw_groupstring].keys())
        except KeyError:
            if ineq == max(ineq_numbers):
                print(f"WARNING: No g4iw-worm group found for atom {ineq} in the input file. Aborting.")
                exit()
            else:
                print(
                    f"WARNING: No g4iw-worm group found for atom {ineq} in the input file. Continuing with next atom."
                )
                continue

        niw, niv = get_niw_niv(vertex_file, g4iw_groupstring, indices)

        print("Number of bands:", n_bands)
        print("Number of fermionic Matsubara frequencies:", niv)
        print("Number of bosonic Matsubara frequencies:", niw)

        print(f"Extracting G2 for atom {ineq} ...")
        g2_uuuu, g2_dddd, g2_dduu, g2_uudd, g2_uddu, g2_duud = extract_g2_general(
            g4iw_groupstring, indices, vertex_file, niw, niv
        )
        print(f"G2 extracted. Calculating G2_dens and G2_magn for atom {ineq} ...")
        g2_dens = 0.5 * (g2_uuuu + g2_dddd + g2_uudd + g2_dduu)
        g2_magn = 0.5 * (g2_uddu + g2_duud)

        del g2_uuuu, g2_dddd, g2_dduu, g2_uudd, g2_uddu, g2_duud
        gc.collect()
        print(f"G2_dens and G2_magn for atom {ineq} calculated. Writing to file ...")

        save_to_file([g2_dens, g2_magn], ["dens", "magn"], niw, n_bands, ineq)
        del g2_dens, g2_magn
        gc.collect()
        print(f"G2_dens and G2_magn for atom {ineq} successfully written to file.")

    output_file.close()
    vertex_file.close()
    print(f"{len(ineq_numbers)} inequivalent atom(s) written to {output_filename}.")
    print("Done!")
    exit()

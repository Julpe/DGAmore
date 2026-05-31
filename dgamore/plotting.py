# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore — Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator

from dgamore.brillouin_zone import KGrid
from dgamore.gap_function import GapFunction
from dgamore.local_four_point import LocalFourPoint
from dgamore.local_n_point import LocalNPoint
from dgamore.matsubara_frequencies import MFHelper
from dgamore.n_point_base import IAmNonLocal


def add_afzb(ax=None, kx=None, ky=None, lw=1.0, marker=""):
    """
    Add visual lines to mark the antiferromagnetic zone-boundary to existing axis.
    """
    if np.any(kx < 0):
        ax.plot(np.linspace(-np.pi, 0, 101), np.linspace(0, np.pi, 101), "--k", lw=lw, marker=marker)
        ax.plot(np.linspace(-np.pi, 0, 101), np.linspace(0, -np.pi, 101), "--k", lw=lw, marker=marker)
        ax.plot(np.linspace(0, np.pi, 101), np.linspace(-np.pi, 0, 101), "--k", lw=lw, marker=marker)
        ax.plot(np.linspace(0, np.pi, 101), np.linspace(np.pi, 0, 101), "--k", lw=lw, marker=marker)
        ax.plot(kx, 0 * kx, "-k", lw=lw, marker=marker)
        ax.plot(0 * ky, ky, "-k", lw=lw, marker=marker)
    else:
        ax.plot(np.linspace(0, np.pi, 101), np.linspace(np.pi, 2 * np.pi, 101), "--k", lw=lw, marker=marker)
        ax.plot(np.linspace(np.pi, 0, 101), np.linspace(0, np.pi, 101), "--k", lw=lw, marker=marker)
        ax.plot(np.linspace(np.pi, 2 * np.pi, 101), np.linspace(0, np.pi, 101), "--k", lw=lw, marker=marker)
        ax.plot(np.linspace(np.pi, 2 * np.pi, 101), np.linspace(np.pi * 2, np.pi, 101), "--k", lw=lw, marker=marker)
        ax.plot(kx, np.pi * np.ones_like(kx), "-k", lw=lw, marker=marker)
        ax.plot(np.pi * np.ones_like(ky), ky, "-k", lw=lw, marker=marker)

    ax.set_xlim(kx[0], kx[-1])
    ax.set_ylim(ky[0], ky[-1])
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")


def find_zeros(mat: np.ndarray) -> np.ndarray:
    """
    Finds the zero crossings of a 2D matrix.
    """
    ind_x = np.arange(mat.shape[0])
    ind_y = np.arange(mat.shape[1])

    cs1 = plt.contour(ind_x, ind_y, mat.T.real, cmap="magma", levels=[0])
    paths = cs1.get_paths()
    plt.close()
    paths = np.atleast_1d(paths)
    vertices = []
    for path in paths:
        vertices.extend(path.vertices)
    return np.array(vertices, dtype=int)


def sigma_loc_checks(
    siw_arr: list[np.ndarray],
    labels: list[str],
    beta: float,
    output_dir: str = "./",
    show: bool = False,
    save: bool = True,
    name: str = "",
    xmax: float = 0,
) -> None:
    """
    siw_arr: list of local self-energies for routine plots.
    """
    if xmax == 0:
        xmax = 5 + 2 * beta
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 5))
    axes = axes.flatten()

    for i, siw in enumerate(siw_arr):
        vn = MFHelper.vn(np.size(siw) // 2)
        axes[0].plot(vn, siw.real, label=labels[i])
        axes[1].plot(vn, siw.imag, label=labels[i])
        axes[2].loglog(vn, siw.real, label=labels[i])
        axes[3].loglog(vn, np.abs(siw.imag), label=labels[i])

    for i in range(4):
        axes[i].set_xlabel(r"$\nu_n$")

    axes[0].set_ylabel(r"$\Re \Sigma(i\nu_n)$")
    axes[1].set_ylabel(r"$\Im \Sigma(i\nu_n)$")
    axes[2].set_ylabel(r"$\Re \Sigma(i\nu_n)$")
    axes[3].set_ylabel(r"$|\Im \Sigma(i\nu_n)|$")

    axes[0].set_xlim(0, xmax)
    axes[1].set_xlim(0, xmax)
    axes[2].set_xlim(None, xmax)
    axes[3].set_xlim(None, xmax)
    plt.legend()
    axes[1].set_ylim(None, 0)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f"sde_{name}_check.pdf"), bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    else:
        plt.close()


def chi_checks(
    chi_dens_list: list[np.ndarray],
    chi_magn_list: list[np.ndarray],
    beta: float,
    labels: list[str],
    e_kin: float,
    output_dir: str = "./",
    orbs=[0, 0, 0, 0],
    show: bool = False,
    save: bool = True,
    name: str = "",
):
    """
    Routine plots to inspect chi_dens and chi_magn
    """
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 5), dpi=500)
    axes = axes.flatten()
    niw_chi_input = np.size(chi_dens_list[0][*orbs, :])

    for i, cd in enumerate(chi_dens_list):
        axes[0].plot(MFHelper.wn(len(cd[*orbs, :]) // 2), cd[*orbs, :].real, label=labels[i])
    axes[0].set_ylabel(r"$\Re \chi(i\omega_n)_{dens}$")
    axes[0].legend()

    for i, cd in enumerate(chi_magn_list):
        axes[1].plot(MFHelper.wn(len(cd[*orbs, :]) // 2), cd[*orbs, :].real, label=labels[i])
    axes[1].set_ylabel(r"$\Re \chi(i\omega_n)_{magn}$")
    axes[1].legend()

    for i, cd in enumerate(chi_dens_list):
        axes[2].loglog(MFHelper.wn(len(cd[*orbs, :]) // 2), cd[*orbs, :].real, label=labels[i], ms=0)
    axes[2].loglog(
        MFHelper.wn(niw_chi_input),
        np.real(1 / (1j * MFHelper.wn(niw_chi_input, beta) + 0.000001) ** 2 * e_kin) * 2,
        ls="--",
        label="Asympt",
        ms=0,
    )
    axes[2].set_ylabel(r"$\Re \chi(i\omega_n)_{dens}$")
    axes[2].legend()

    for i, cd in enumerate(chi_magn_list):
        axes[3].loglog(MFHelper.wn(len(cd[*orbs, :]) // 2), cd[*orbs, :].real, label=labels[i], ms=0)
    axes[3].loglog(
        MFHelper.wn(niw_chi_input),
        np.real(1 / (1j * MFHelper.wn(niw_chi_input, beta) + 0.000001) ** 2 * e_kin) * 2,
        "--",
        label="Asympt",
        ms=0,
    )
    axes[3].set_ylabel(r"$\Re \chi(i\omega_n)_{magn}$")
    axes[3].legend()
    axes[0].set_xlim(-1, 10)
    axes[1].set_xlim(-1, 10)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f"chi_dens_magn_{name}.pdf"), bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    else:
        plt.close()


def plot_nu_nup(
    obj: LocalFourPoint,
    orbs: np.ndarray | list | tuple = (0, 0, 0, 0),
    omega: int = 0,
    do_save: bool = True,
    output_dir: str = "./",
    name: str = "Name",
    colormap: str = "RdBu",
    show: bool = False,
) -> None:
    """
    Plots the four-point object for a given set of orbitals and a given bosonic frequency.
    """
    if np.abs(omega) > obj.niw:
        raise ValueError(f"Omega {omega} out of range.")
    if len(orbs) != 4:
        raise ValueError("'orbs' needs to be of size 4.")

    fig, axes = plt.subplots(ncols=2, figsize=(7, 3), dpi=251)
    axes = axes.flatten()
    wn_list = MFHelper.wn(obj.niw)
    wn_index = np.argmax(wn_list == omega)
    mat = obj.mat[orbs[0], orbs[1], orbs[2], orbs[3], wn_index, ...]
    vn = MFHelper.vn(obj.niv)
    im1 = axes[0].pcolormesh(vn, vn, mat.real, cmap=colormap)
    im2 = axes[1].pcolormesh(vn, vn, mat.imag, cmap=colormap)
    axes[0].set_title(r"$\Re$")
    axes[1].set_title(r"$\Im$")
    for ax in axes:
        ax.set_xlabel(r"$\nu_p$")
        ax.set_ylabel(r"$\nu$")
        ax.set_aspect("equal")
    fig.suptitle(name)
    fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location="right", pad=0.05)
    fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location="right", pad=0.05)
    plt.tight_layout()
    if do_save:
        plt.savefig(os.path.join(output_dir, f"{name}_w{omega}.pdf"), bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    else:
        plt.close()


def plot_two_point_kx_ky(
    obj: LocalNPoint | IAmNonLocal,
    kx: np.ndarray,
    ky: np.ndarray,
    pi_shift: bool = True,
    title: str = "",
    name: str = "",
    orbs: np.ndarray | list | tuple = (0, 0),
    output_dir="./",
    cmap="magma",
    scatter=None,
    save: bool = True,
    show: bool = False,
):
    """
    Used to plot a two-point function in kx, ky space for given orbitals.
    """
    if len(orbs) != 2:
        raise ValueError("'orbs' needs to be of size 2.")

    mat = obj.shift_k_by_pi().mat if pi_shift else obj.mat

    niv = mat.shape[-1] // 2
    mat = mat[:, :, 0, orbs[0], orbs[1], niv]
    mat = np.concatenate([mat, mat[0:1, :, ...]], axis=0)
    mat = np.concatenate([mat, mat[:, 0:1, ...]], axis=1)

    fig, axes = plt.subplots(ncols=2, figsize=(7, 3), dpi=500)
    axes = axes.flatten()
    im1 = axes[0].pcolormesh(kx, ky, mat.T.real, cmap=cmap)
    im2 = axes[1].pcolormesh(kx, ky, mat.T.imag, cmap=cmap)
    axes[0].set_title(r"$\Re$" + title)
    axes[1].set_title(r"$\Im$" + title)

    tick_vals = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]

    for ax in axes:
        ax.set_xlabel(r"$k_x$")
        ax.set_aspect("equal")
        add_afzb(ax=ax, kx=kx, ky=ky, lw=1.0, marker="")
        ax.set_xticks(tick_vals)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_vals)
    axes[0].set_ylabel(r"$k_y$")
    axes[1].set_ylabel("")
    axes[0].set_yticklabels(tick_labels)
    axes[1].set_yticklabels([""] * len(tick_labels))

    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)  # 5% width, 10% padding
    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)  # 5% width, 10% padding

    # fig.suptitle(title)
    vmin1, vmax1 = im1.get_clim()
    vmin2, vmax2 = im2.get_clim()

    cb1 = fig.colorbar(im1, cax=cax1, ticks=np.linspace(vmin1, vmax1, 5))
    cb1.locator = ticker.MaxNLocator(nbins=5)
    cb1.update_ticks()
    cb2 = fig.colorbar(im2, cax=cax2, ticks=np.linspace(vmin2, vmax2, 5))
    cb2.locator = ticker.MaxNLocator(nbins=5)
    cb2.update_ticks()

    if scatter is not None:
        for ax in axes:
            colours = plt.cm.get_cmap(cmap)(np.linspace(0, 1, np.shape(scatter)[0]))
            ax.scatter(scatter[:, 0], scatter[:, 1], marker="o", c=colours)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    else:
        plt.close()


def plot_two_point_kx_ky_real_and_imag(
    obj: LocalNPoint | IAmNonLocal,
    kx: np.ndarray,
    ky: np.ndarray,
    pi_shift: bool = True,
    title: str = "",
    name: str = "",
    orbs: np.ndarray | list | tuple = (0, 0),
    output_dir="./",
    cmap="magma",
    save: bool = True,
    show: bool = False,
):
    """
    Used to plot a two-point function in kx, ky space for given orbitals in two separate files for the real part and
    imaginary part, respectively.
    """
    cm = 1.0 / 2.54
    if len(orbs) != 2:
        raise ValueError("'orbs' needs to be of size 2.")

    obj = obj.shift_k_by_pi() if pi_shift else obj
    for idx, mat in enumerate([obj.mat.real, obj.mat.imag]):

        niv = mat.shape[-1] // 2
        mat = mat[:, :, 0, orbs[0], orbs[1], niv]
        mat = np.concatenate([mat, mat[0:1, :, ...]], axis=0)
        mat = np.concatenate([mat, mat[:, 0:1, ...]], axis=1)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9 * cm, 9 * cm), dpi=500)
        im = axes.pcolormesh(kx, ky, mat.T, cmap=cmap)
        axes.set_title(rf"$\Re {title}$" if idx == 0 else rf"$\Im {title}$")

        tick_vals = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]

        axes.set_xlabel(r"$k_x$")
        axes.set_ylabel(r"$k_y$")
        axes.set_aspect("equal")

        axes.set_xticks(tick_vals)
        axes.set_xticklabels(tick_labels)
        axes.set_yticks(tick_vals)
        axes.set_yticklabels(tick_labels)

        divider1 = make_axes_locatable(axes)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)  # 5% width, 10% padding

        cb = fig.colorbar(im, cax=cax1)
        cb.locator = ticker.MaxNLocator(nbins=5)
        cb.update_ticks()
        plt.tight_layout()
        if save:
            plt.savefig(
                os.path.join(output_dir, f"{name}_{"real" if idx == 0 else "imag"}.pdf"),
                bbox_inches="tight",
                pad_inches=0.05,
            )
        if show:
            plt.show()
        else:
            plt.close()


def plot_two_point_kx_ky_with_fs_points(
    obj: LocalNPoint | IAmNonLocal,
    k_grid: KGrid,
    kx: np.ndarray,
    ky: np.ndarray,
    pi_shift: bool = True,
    title: str = "",
    name: str = "",
    orbs: np.ndarray | list | tuple = (0, 0),
    output_dir="./",
    cmap="magma",
    do_save: bool = True,
    show: bool = False,
):
    """
    Used to plot a two-point function in kx, ky space for given orbitals including Fermi surface points.
    """
    mat = obj.mat[..., 0, orbs[0], orbs[1], obj.niv][: obj.nq[0] // 2, : obj.nq[1] // 2]
    fs_ind = find_zeros(mat)
    n_fs = np.shape(fs_ind)[0]
    fs_ind = fs_ind[: n_fs // 2]
    fs_points = np.stack((k_grid.kx[fs_ind[:, 0]], k_grid.ky[fs_ind[:, 1]]), axis=1)
    plot_two_point_kx_ky(obj, kx, ky, pi_shift, title, name, orbs, output_dir, cmap, fs_points, do_save, show)


def plot_gap_function(
    obj: GapFunction,
    kx: np.ndarray,
    ky: np.ndarray,
    name: str = "",
    orbs: np.ndarray | list | tuple = (0, 0),
    output_dir="./",
    cmap="magma",
    scatter=None,
    do_save: bool = True,
    show: bool = False,
):
    """
    Used to plot the gap function in kx, ky space for given orbitals. Here, not the real and imaginary parts are plotted,
    but rather the two values at the two lowest positive and smallest negative Matsubara frequency, respectively. This
    helps investigating the symmetry of the gap function.
    """
    if len(orbs) != 2:
        raise ValueError("'orbs' needs to be of size 2.")

    gap = obj.shift_k_by_pi().mat
    niv_pp = gap.shape[-1] // 2
    gap = gap[:, :, 0, orbs[0], orbs[1], niv_pp - 1 : niv_pp + 1]
    gap = np.concatenate([gap, gap[0:1, :, ...]], axis=0)
    gap = np.concatenate([gap, gap[:, 0:1, ...]], axis=1)

    fig, axes = plt.subplots(ncols=2, figsize=(7, 3), dpi=500)
    axes = axes.flatten()
    im1 = axes[0].pcolormesh(kx, ky, gap[..., 0].T.real, cmap=cmap)
    im2 = axes[1].pcolormesh(kx, ky, gap[..., 1].T.real, cmap=cmap)
    axes[0].set_title(f"$\\Delta^{{k_x k_y k_z=0;\\nu=\\frac{{\\pi}}{{\\beta}}}}_{{{obj.channel.value[0]}}}$")
    axes[1].set_title(f"$\\Delta^{{k_x k_y k_z=0;\\nu=-\\frac{{\\pi}}{{\\beta}}}}_{{{obj.channel.value[0]}}}$")

    tick_vals = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]

    for ax in axes:
        ax.set_xlabel(r"$k_x$")
        ax.set_aspect("equal")
        add_afzb(ax=ax, kx=kx, ky=ky, lw=1.0, marker="")
        ax.set_xticks(tick_vals)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_vals)
    axes[0].set_ylabel(r"$k_y$")
    axes[1].set_ylabel("")
    axes[0].set_yticklabels(tick_labels)
    axes[1].set_yticklabels([""] * len(tick_labels))

    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)  # 5% width, 10% padding
    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)  # 5% width, 10% padding

    # fig.suptitle(title)
    vmin1, vmax1 = im1.get_clim()
    vmin2, vmax2 = im2.get_clim()

    fig.colorbar(im1, cax=cax1, ticks=np.linspace(vmin1, vmax1, 5))
    fig.colorbar(im2, cax=cax2, ticks=np.linspace(vmin2, vmax2, 5))

    if scatter is not None:
        for ax in axes:
            colours = plt.cm.get_cmap(cmap)(np.linspace(0, 1, np.shape(scatter)[0]))
            ax.scatter(scatter[:, 0], scatter[:, 1], marker="o", c=colours)
    plt.tight_layout()
    if do_save:
        plt.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    else:
        plt.close()


def plot_spectrum(
    a_w: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    high_sym_points: list[tuple[float, float, float, str]],
    energy_window: tuple[float, float],
    beta: float,
    title: str,
    fermi_energy: float = 0,
    output_dir="./",
    name: str = "",
    cmap="magma",
    do_save: bool = True,
    show: bool = False,
):
    """
    Plots the total (band-summed) spectral function along a high-symmetry path. A(k,w) is
    expected to be in band-diagonal space already.
    """
    n_per_seg = 200
    # Determine Grid Properties for wrapping
    k_axes = (kx, ky, kz)

    a_w = np.sum(a_w, axis=-2)  # sum over bands to get total spectral function

    periods = []
    for ax in k_axes:
        if len(ax) > 1:
            step = ax[1] - ax[0]
            periods.append(ax.max() - ax.min() + step)
        else:
            periods.append(2 * np.pi)
    periods = np.array(periods)

    k_mins = np.array([ax.min() for ax in k_axes])

    path_segments = []
    labels = [rf"$\Gamma$" if "gamma" in p[3].lower() else rf"${p[3]}$" for p in high_sym_points]
    points = np.array([p[:3] for p in high_sym_points])
    points *= periods

    for i in range(len(points) - 1):
        start, end = points[i], points[i + 1]
        # Linear interpolation between high-symmetry points
        seg = np.linspace(start, end, n_per_seg, endpoint=(i == len(points) - 2))
        path_segments.append(seg)
    full_path = np.vstack(path_segments)

    # Wrapping & Interpolation
    path_wrapped = k_mins + (full_path - k_mins) % periods

    nw = a_w.shape[-1]
    a_on_path = np.zeros((len(full_path), nw))

    # Create interpolator for each energy slice
    for i in range(nw):
        interp = RegularGridInterpolator(k_axes, a_w[..., i], bounds_error=False, fill_value=None)
        a_on_path[:, i] = interp(path_wrapped)

    a_on_path = np.nan_to_num(a_on_path)

    # Calculate X-axis (Arc Length)
    dists = np.sqrt(np.sum(np.diff(full_path, axis=0) ** 2, axis=1))
    arc = np.concatenate(([0], np.cumsum(dists)))
    tick_indices = [min(i * n_per_seg, len(arc) - 1) for i in range(len(labels))]
    tick_x = arc[tick_indices]

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    v_max = np.percentile(a_on_path, 98)

    w_axis = 5 * beta * np.tan(np.linspace(-np.pi / 2.1, np.pi / 2.1, num=nw, endpoint=True)) / np.tan(np.pi / 2.1)

    pm = ax.pcolormesh(
        arc, w_axis - fermi_energy, a_on_path.T, cmap=cmap, shading="gouraud", rasterized=True, vmin=0, vmax=v_max
    )

    # Formatting
    ax.set_xticks(tick_x)
    ax.set_xticklabels(labels)
    ax.set_xlim(arc[0], arc[-1])
    ax.set_ylim(*energy_window)
    ax.set_ylabel(r"$\omega - \varepsilon_{\mathrm{F}}$ [eV]")
    ax.set_title(title)

    # Guidelines
    for tx in tick_x:
        ax.axvline(tx, color="white", alpha=0.3, lw=0.8)
    ax.axhline(0, color="white", lw=1.2, ls="--", alpha=0.6)

    cbar = fig.colorbar(pm, pad=0.02)
    cbar.set_label(r"$A(\mathbf{k}, \omega)$")

    plt.tight_layout()
    if do_save:
        plt.savefig(os.path.join(output_dir, f"spectrum_{name}.png"), dpi=400)
    if show:
        plt.show()

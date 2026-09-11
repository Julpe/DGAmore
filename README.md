[![CI](https://github.com/Julpe/DGAmore/actions/workflows/CI.yml/badge.svg)](https://github.com/Julpe/DGAmore/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/Julpe/DGAmore/graph/badge.svg?token=O1E161NNHP)](https://codecov.io/github/Julpe/DGAmore)
[![Documentation Status](https://app.readthedocs.org/projects/dgamore/badge/?version=latest)](https://dgamore.readthedocs.io/en/latest/?badge=latest)

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logos/dgamore-lockup-tagline-dark.svg" />
    <img src="logos/dgamore-lockup-tagline-light.svg" alt="DGAmore" width="55%" />
  </picture>
</p>

`DGAmore` is a Python toolbox that computes the multi-orbital, self-consistent ladder Dynamical Vertex Approximation
and solves the Eliashberg equation for (strongly) correlated electron systems described by the multi-band Hubbard
model. Starting from the one- and two-particle output of a dynamical mean-field theory (DMFT) calculation, it
assembles the local vertex functions, solves the momentum-dependent ladder equations for the non-local self-energy
and, optionally, extracts the leading superconducting eigenvalues and gap functions. It relies on vectorized `numpy`
operations, parallelizes the heavy momentum-dependent work with `mpi4py`, and reads its DMFT input from HDF5 via
`h5py`. It combines and extends the one-shot multi-orbital [AbinitioDGA](https://doi.org/10.1016/j.cpc.2019.07.012)
and the single-band [DGApy](https://github.com/PaulWorm/DGApy), on which it is partially based.

For the implemented equations, see the author's
[Master's thesis](https://doi.org/10.34726/hss.2025.130528) (Chapters 3 and 4).

# Features

- Multi-orbital ladder DGA in the density and magnetic channels, with explicit high-frequency asymptotics of the
  vertex functions and support for non-local interactions and several inequivalent atoms.
- A self-consistency loop for the non-local self-energy with linear, Pulay and Anderson mixing, a cooldown procedure
  that warm-starts low-temperature runs from converged higher-temperature ones, and three stabilization techniques
  for the regime close to the Bethe-Salpeter instability; the one-shot Moriya lambda correction is available as well.
- The multi-orbital Eliashberg equation, yielding the leading singlet and triplet eigenvalues and gap functions,
  optionally resolved by frequency parity and sorted into symmetry-adapted multiplets.
- Optional analytic continuation of the DGA and DMFT Green's functions to real frequencies with the maximum entropy
  method.
- MPI parallelization over the irreducible Brillouin zone with automatic symmetry discovery, shared-memory windows for
  replicated objects, and an upfront check that a run fits the memory of the nodes it received.

# Documentation

**Full documentation is hosted at [dgamore.readthedocs.io](https://dgamore.readthedocs.io/en/latest).**

| Topic | Description |
| --- | --- |
| [Installation](https://dgamore.readthedocs.io/en/latest/installation.html) | Environment setup, MPI dependencies, and installing the package. |
| [Usage](https://dgamore.readthedocs.io/en/latest/usage.html) | Running the routine single-core, with MPI, and on a SLURM cluster. |
| [Configuration](https://dgamore.readthedocs.io/en/latest/configuration.html) | The YAML configuration file and its parameters. |
| [Cooldown and stabilization](https://dgamore.readthedocs.io/en/latest/cooldown.html) | Reaching low temperatures by chaining runs, and the stabilization techniques of the self-consistency cycle. |
| [Output files](https://dgamore.readthedocs.io/en/latest/output.html) | Every file a run writes, its array layout, and how to read it back. |
| [Contributing](https://dgamore.readthedocs.io/en/latest/contributing.html) | Reporting issues and submitting pull requests. |
| [API reference](https://dgamore.readthedocs.io/en/latest/api.html) | Module-by-module reference. |
| [About](https://dgamore.readthedocs.io/en/latest/about.html) | Background, citation, license, and contact. |

# Quick start

`DGAmore` needs Python 3.12 or newer and a working MPI installation; it is tested on Linux and macOS with Python 3.12
to 3.14 (Windows is not supported). Install `mpich` and `mpi4py`, then the package:

```bash
conda install -c conda-forge mpich mpi4py
git clone https://github.com/Julpe/DGAmore.git
cd DGAmore
pip install .
```

To check the installation, run `pytest tests` from the repository directory. A calculation needs three inputs: the
one- and two-particle output of a w2dynamics DMFT run (the two-particle file goes through the installed `symmetrize`
script first), a Hamiltonian in real space (Wannier90 format) or in momentum space, and a YAML configuration file.
Edit the configuration, then start the routine with `-p` pointing at the directory that holds it and `-c` naming it
(defaults: the current working directory and [dga_config.yaml](dgamore/dga_config.yaml)):

```bash
mpiexec -np 8 DGAmore -p /configs/ -c my_config.yaml   # or: DGAmore for a single-core test run
```

See the [installation](https://dgamore.readthedocs.io/en/latest/installation.html) and
[usage](https://dgamore.readthedocs.io/en/latest/usage.html) pages for the full instructions and an example SLURM
submit script, the [configuration](https://dgamore.readthedocs.io/en/latest/configuration.html) page for every
parameter, and the [cooldown](https://dgamore.readthedocs.io/en/latest/cooldown.html) page before attempting
low-temperature calculations.

# Contributing

Contributions are welcome. Please open an issue for bugs and feature requests, or submit a pull request. See
[CONTRIBUTING.md](CONTRIBUTING.md) for details.

# Citation and license

`DGAmore` is released under the MIT license. If you use it, please consider citing it together with the author's
[Master's thesis](https://doi.org/10.34726/hss.2025.130528). For questions, get in touch by
[e-mail](mailto:julian.peil@tuwien.ac.at).

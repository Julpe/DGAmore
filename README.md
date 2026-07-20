[![CI](https://github.com/Julpe/DGAmore/actions/workflows/CI.yml/badge.svg)](https://github.com/Julpe/DGAmore/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/Julpe/DGAmore/graph/badge.svg?token=O1E161NNHP)](https://codecov.io/github/Julpe/DGAmore)
[![Documentation Status](https://app.readthedocs.org/projects/dgamore/badge/?version=latest)](https://dgamore.readthedocs.io/en/latest/?badge=latest)

---

<p align="center">
  <img src="logos/DGAmore_light.png" alt="DGAmore" width="38%" />
  &nbsp;&nbsp;&nbsp;
  <img src="logos/DGAmore_dark.png" alt="DGAmore" width="38%" />
</p>

`DGAmore` is a Python toolbox that computes the multi-orbital, self-consistent ladder Dynamical Vertex Approximation
and solves the Eliashberg equation for (strongly) correlated electron systems described by the multi-band Hubbard
model. Starting from the one- and two-particle output of a dynamical mean-field theory (DMFT) calculation, it
assembles the local vertex functions, solves the momentum-dependent ladder equations for the non-local self-energy
and, optionally, extracts the leading superconducting eigenvalues and gap functions. It relies on vectorized `numpy`
operations, parallelizes the heavy momentum-dependent work with `mpi4py`, and reads its DMFT input from HDF5 via
`h5py`. It is partially based on [DGApy](https://github.com/PaulWorm/DGApy).

For the implemented equations, see the author's
[Master's thesis](https://doi.org/10.34726/hss.2025.130528) (Chapters 3 and 4).

# Documentation

**Full documentation is hosted at [dgamore.readthedocs.io](https://dgamore.readthedocs.io/en/latest).**

| Topic | Description |
| --- | --- |
| [Installation](https://dgamore.readthedocs.io/en/latest/installation.html) | Environment setup, MPI dependencies, and installing the package. |
| [Usage](https://dgamore.readthedocs.io/en/latest/usage.html) | Running the routine single-core, with MPI, and on a SLURM cluster. |
| [Configuration](https://dgamore.readthedocs.io/en/latest/configuration.html) | The YAML configuration file and its parameters. |
| [Contributing](https://dgamore.readthedocs.io/en/latest/contributing.html) | Reporting issues and submitting pull requests. |
| [API reference](https://dgamore.readthedocs.io/en/latest/api.html) | Module-by-module reference. |
| [About](https://dgamore.readthedocs.io/en/latest/about.html) | Background, citation, license, and contact. |

# Quick start

Install `mpich` and `mpi4py` (Python 3.12+ required), then install the package:

```bash
conda install -c conda-forge mpich mpi4py
git clone https://github.com/Julpe/DGAmore.git
cd DGAmore
pip install .
```

Configure a run by editing your configuration file, then execute the routine with `-p` to point at the directory
holding it and `-c` to name it (defaults: the current working directory and
[dga_config.yaml](dgamore/dga_config.yaml)):

```bash
mpiexec -np 8 DGAmore -p /configs/ -c my_config.yaml   # or: DGAmore for a single-core test run
```

See the [installation](https://dgamore.readthedocs.io/en/latest/installation.html) and
[usage](https://dgamore.readthedocs.io/en/latest/usage.html) pages for the full instructions and an example SLURM submit script.

# Contributing

Contributions are welcome. Please open an issue for bugs and feature requests, or submit a pull request. See
[CONTRIBUTING.md](CONTRIBUTING.md) for details.

# Citation and license

`DGAmore` is released under the MIT license. If you use it, please consider citing it together with the author's
[Master's thesis](https://doi.org/10.34726/hss.2025.130528). For questions, get in touch by
[e-mail](mailto:julian.peil@tuwien.ac.at).

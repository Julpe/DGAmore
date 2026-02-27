[![CI](https://github.com/Julpe/moLDGA/actions/workflows/CI.yml/badge.svg)](https://github.com/Julpe/moLDGA/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/Julpe/moLDGA/graph/badge.svg?token=O1E161NNHP)](https://codecov.io/github/Julpe/moLDGA)

---

## Introduction

`moLDGA` is a Python toolbox for performing the multi-orbital (self-consistent) Dynamical Vertex Approximation and 
Eliashberg equation for (strongly) correlated electron systems. It is partially based on 
[DGApy](https://github.com/PaulWorm/DGApy), which calculates the Dynamical Vertex Approximation for single-band models. 
If you are familiar with [DGApy](https://github.com/PaulWorm/DGApy), you will find the installation and usage to be very 
similar.

The code is written in Python and uses primarily `numpy` for numerical computations. It also uses `mpi4py` for 
parallelization and `h5py` for reading data in HDF5 format. The code is structured in a modular way, with different 
modules for different parts of the calculation.

Additionally, we feature extensive logging: every important step in the calculation will be logged. If `moLDGA` is 
started from a terminal, the logging will be done to the standard output. If the code is executed on a slurm-based 
cluster, one will find the logs in the job output file. The reason we employ a lot of logging is the ease of finding 
errors that might occur during a calculation.

For details regarding the implemented equations, please have a look at my 
[Master's thesis](https://doi.org/10.34726/hss.2025.130528), specifically Chapters 3 and 4.

--- 

## Installation

We recommend installing `moLDGA` in a virtual environment. There are many options to create virtual environments, 
however, we recommend `conda` or `miniconda`. We also recommend using Python 3.12 or higher, as we do not guarantee that 
the code will work with older versions of Python. Currently, the CI pipeline tests the code with Python 3.12, 3.13, and 
3.14 on both Linux and macOS, so we can guarantee that the code works with these versions of Python. We do not test the 
code on Windows, so we cannot guarantee that it will work there.

To install `conda` or `miniconda`, please follow the instructions on their respective websites:
- `conda`: [Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) or 
[macOS](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html).
- `miniconda`: [Linux](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2) or
[macOS](https://www.anaconda.com/docs/getting-started/miniconda/install#macos-2).

Once you have created your virtual environment, you have to clone the repository into any folder of choice. You can do 
this with the following command:
```bash
git clone https://github.com/Julpe/moLDGA.git
```
After switching to the `moLDGA` directory, i.e., where you find the file [setup.py](setup.py), you can install `moLDGA` 
and all of its requirements in normal mode with 
```bash 
pip install .
``` 
or in editable mode with 
```bash
pip install -e .
```
Editable mode is recommended if you want to make changes to the code, as it allows you to edit the code without having 
to reinstall it every time. Also, if new changes are made to the code, you can simply pull the latest version from the 
repository, and you will have the latest version of the code without having to reinstall it.

---

## Code Execution

The main entry point to the program is the file [dga_main.py](moldga/dga_main.py), which can be started with either
```bash
python dga_main.py
```
for single-core execution (mostly used for testing purposes) or
```bash
mpiexec -np <n_proc> python dga_main.py
```
for multi-core processing with MPI. Instead of `mpiexec`, one can also use `mpirun` or `srun` (if you are using a 
slurm-based cluster). The number of processes, `<n_proc>`, should be chosen according to the size of the problem and the
available computational resources.

There are two additional command line parameters available
- `-p`: This is used to specify the path to the configuration file, which contains all run-specific parameters. 
This is useful if one wants to store multiple configuration files in different directories. If this parameter is not 
set, the path defaults to the location of the repository directory.
- `-c`: This is used to specify the name of the configuration file one wants to load. It defaults to `dga_config.yaml`.

As an example, the following shell command runs the code using 8 MPI processes and loads the configuration file 
`my_config.yaml` from the path `/configs/`:

```bash
mpiexec -np 8 python dga_main.py -p /configs/ -c my_config.yaml
```

The following code snippet shows the content of an exemplary job submit script for a slurm-based cluster.
In this case, the configuration file is named `my_config.yaml`:

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -J <some_job_name>
#SBATCH --partition=<some_partition>
#SBATCH --qos=<some_qos>
#SBATCH --ntasks-per-node=96
#SBATCH -t 1:00:00

# Load the necessary modules, in this case activate the conda environment 
# with the preinstalled moLDGA package and its dependencies.
module purge
source <path to miniconda>/miniconda3/bin/activate <some_conda_env>

# Run the code with MPI, using all available tasks on the node.
mpiexec -n $SLURM_NTASKS python <path to code>/dga_main.py -p "<path to config>" -c "my_config.yaml"
```

---

## Configuration

To configure run parameters, one has to create or modify the existing configuration file, which is a YAML file. The 
default configuration file is [dga_config.yaml](moldga/dga_config.yaml), which can be found in the repository directory. 
Each entry in the configuration file is explained in detail in the file itself, so please refer to the file for more 
information on how to configure the code. Please note that each section in the configuration file is required, and the 
code will not run if any section is missing. The default values in the configuration file are set to reasonable 
defaults, so if you are unsure about any of the parameters, you can simply use the default values (except for the file 
paths to the input). If you are unsure about any of the parameters, please refer to the documentation in the 
configuration file or contact me via [E-Mail](mailto:julian.peil@tuwien.ac.at).

--- 

## Contributing

If you want to contribute to the development of moLDGA, please feel free to fork the repository and create a pull 
request with your changes. Before creating a pull request, please make sure that your code follows the existing coding 
style and that it is well-documented. Also, please make sure that your code does not break any existing functionality 
and that it is tested. We have set up a continuous integration (CI) pipeline that runs tests on every pull request, so 
if your code does not pass the tests, it will not be merged. Additionally, we have set up a code coverage tool that 
checks the coverage of the tests, so please make sure that your code is well-tested and that it has more than 85% 
coverage. If you are unsure about any of these points, please feel free to contact me via 
[E-Mail](mailto:julian.peil@tuwien.ac.at).

If you find any bugs or issues with the code, please feel free to create an issue in the repository and explain the 
problem in detail. If possible, please also provide a minimal example that reproduces the issue, so that we can easily 
understand and fix the problem. Alternatively, if you have a feature request or an idea for improving the code, please 
also create an issue and explain your idea in detail. If you are unsure about how to create or flag issues, feel free to 
contact me via [E-Mail](mailto:julian.peil@tuwien.ac.at).

We will try to fix an issue as soon as possible if it is a critical bug. However, if it is a feature request or minor 
bug that does not affect the overall functionality of the code, we will consider it for future development. If you 
create an issue, make sure to provide the correct label (bug, feature request, etc.), this will help us to keep track of 
the issues more easily.

---

## Citation and Acknowledgements

If you use this code, please consider citing it and also consider citing my 
[Master's thesis](https://doi.org/10.34726/hss.2025.130528). 

A big thanks goes to my colleagues, who have contributed to the development of this code, either by providing feedback 
or testing.

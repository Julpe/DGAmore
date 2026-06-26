Installation
============

DGAmore supports Python 3.12 and newer and is distributed as an installable package. It is developed and
continuously tested on Linux and macOS with Python 3.12, 3.13 and 3.14; Windows is not supported. Because the code
relies on MPI for parallelisation, the installation needs a working MPI implementation in addition to the Python
dependencies.

We recommend installing DGAmore into an isolated virtual environment. Any environment manager works, but ``conda``
or ``miniconda`` are the most convenient because they can also provide the MPI libraries. Installation instructions
for both are available for `Linux <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_
and `macOS <https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html>`_. After installing,
make sure the environment is activated and, for convenience, that the ``conda`` executable is on your ``PATH``.

First create and activate an environment, then install ``mpich`` and ``mpi4py`` so that the code can run under MPI:

.. code-block:: bash

   conda create -n py-dgamore python=3.13
   conda activate py-dgamore
   conda install -c conda-forge mpich mpi4py

Other MPI implementations such as ``openmpi`` and ``intel-mpi`` are also supported in place of ``mpich``.

Next, clone the repository into a directory of your choice (this requires ``git``):

.. code-block:: bash

   git clone https://github.com/Julpe/DGAmore.git

Finally, change into the ``DGAmore`` directory and install the package together with all of its dependencies. A
regular install is done with

.. code-block:: bash

   cd DGAmore
   pip install .

while an editable install is done with

.. code-block:: bash

   cd DGAmore
   pip install -e .

The editable install is recommended for development, as it lets you modify the source without reinstalling and lets
you pick up upstream changes simply by pulling the latest version. Please make sure no older cached versions of the
dependencies are reused, as this can lead to installation problems.

Installing the package also places the entry point ``DGAmore.py`` on your ``PATH``, so you can launch a run from
anywhere without specifying its full path. To verify the installation, you can optionally install ``pytest`` and run
the test suite:

.. code-block:: bash

   pip install pytest
   pytest

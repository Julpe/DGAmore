Usage
=====

Preparing the input
-------------------

The starting point of every DGAmore run is the result of a DMFT calculation, which currently must come from
w2dynamics, a continuous-time quantum Monte Carlo solver based on the hybridization expansion. A w2dynamics run
yields both the one-particle quantities and a two-particle output containing the four-point Green's functions.
Before DGAmore can read the latter, it has to be brought into the expected format: run the ``symmetrize`` script,
installed alongside the main entry point, which prompts for the input and output file names and writes only the
density and magnetic spin components of the two-particle Green's function. The original, unsymmetrized vertex file
is not needed afterwards.

Beyond the processed two-particle file, a run needs two more things: a configuration file describing all run-time
parameters, and a Hamiltonian. The latter can be supplied in real space (Wannier) or in momentum space as generated
by wien2wannier; for interaction types the code does not handle directly, an additional file specifying the local
and non-local interaction can be provided.

Running a calculation
---------------------

The main entry point of the program is ``DGAmore``. Because it is added to the Python environment as a standalone
executable, it can be invoked by name without its full path. For single-core execution, which is mainly intended for
testing, run:

.. code-block:: bash

   DGAmore

For a parallel run with MPI, use:

.. code-block:: bash

   mpiexec -np <n_proc> DGAmore

Instead of ``mpiexec`` you may also use ``mpirun`` or, on SLURM-based clusters, ``srun``. How many processes
``<n_proc>`` make sense depends on the problem size and the available hardware; more ranks also mean more
communication, so throwing processes at a small problem can slow it down rather than speed it up.

There are two optional command-line arguments: ``-p`` sets the directory holding the configuration file (handy when
configurations for several projects live in different places; the default is the current working directory), and
``-c`` names the configuration file itself, defaulting to ``dga_config.yaml``. As an example, the following command
runs the code with eight MPI processes and loads the configuration file ``my_config.yaml`` from ``/configs/``:

.. code-block:: bash

   mpiexec -np 8 DGAmore -p /configs/ -c my_config.yaml

On a SLURM-based cluster, a typical job submission script looks as follows:

.. code-block:: bash

   #!/bin/bash
   #SBATCH -N <n nodes>
   #SBATCH -J <some job name>
   #SBATCH --partition=<some partition>
   #SBATCH --qos=<some qos>
   #SBATCH --ntasks-per-node=<n proc>
   #SBATCH -t <time limit>
   #SBATCH -o log.txt
   #SBATCH -e log.txt

   # Load the necessary modules; here we activate the conda environment that
   # has the DGAmore package and its dependencies preinstalled.
   module purge
   source <path to miniconda>/miniconda3/bin/activate <your conda env>

   # Use MPI for parallelization, so restrict each task to a single thread.
   export OMP_NUM_THREADS=1

   # Recommended on SLURM-based clusters:
   srun DGAmore -p "<path to config>" -c "<name of config>.yaml"

   # Alternatively, with mpirun or mpiexec:
   mpirun -np $SLURM_NTASKS DGAmore -p "<path to config>" -c "<name of config>.yaml"

The ``-o`` and ``-e`` options set the files for the job output and errors; here both are written to the same file,
but separate files may be used instead. The results of a completed run are written to a subdirectory of the output
folder, whose name encodes run-specific parameters such as the momentum-grid size and the frequency box. The
:doc:`output` page lists every file such a run produces and the array layout of each stored quantity.

.. note::
   During the in-memory Eliashberg solve, the singlet and triplet channels (and, with
   ``resolve_frequency_parity``, their frequency-even and frequency-odd sectors) each get their own rank, so up to
   four solves run concurrently. How many actually run at once on a given node depends on its free host memory:
   the solver packs as many sector solves per node as fit, each solving rank holding one full pairing vertex, so a
   node with enough headroom runs all four while a memory-tight node runs fewer and does the rest sequentially.
   Spreading the ranks over several nodes therefore gives the most concurrency. Since only a handful of ranks
   compute while the rest wait, DGAmore also threads the solver ranks' matrix-vector products for exactly this
   phase, using as many threads as each rank's CPU affinity mask allows. The results are bit-identical to the
   single-threaded ones, and ``OMP_NUM_THREADS=1`` stays correct for the rest of the run. The
   frequency-distributed solve (``save_memory_for_lanczos``) is threaded the same way, except that every rank
   holding a frequency slice computes at once. There, each rank's thread budget is its affinity-mask size divided
   by the number of active ranks on its node, so shared cores are never oversubscribed and the cores of ranks left
   without a frequency slice are put to work automatically. All of this only helps if the launcher leaves the
   affinity mask wider than one core - with a strict one-core-per-rank binding it is a no-op. On Eliashberg-heavy
   runs, prefer a binding that lets the solver ranks spread (e.g. ``srun --cpu-bind=sockets`` or
   ``mpirun --bind-to socket``/``--bind-to none``).

The full set of run-time parameters is described on the :doc:`configuration` page.

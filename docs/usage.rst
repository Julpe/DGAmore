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
but separate files may be used instead. Should any rank fail, it writes its traceback to the error stream and
aborts the whole job, so the remaining ranks never wait for it indefinitely. Before the heavy steps start, DGAmore
verifies that the run fits the memory of every node it received; on a batch system this check honors the job's
cgroup memory limit (e.g. slurm's ``--mem``), so request as much memory as the job may actually use rather than
relying on the node total. The results of a completed run are written to a subdirectory of the output
folder, whose name encodes run-specific parameters such as the momentum-grid size and the frequency box. The
:doc:`output` page lists every file such a run produces and the array layout of each stored quantity.

.. note::
   The in-memory Eliashberg solve spreads the (channel, parity) sectors as evenly as possible over the nodes: one
   node hosts all of them, two nodes one channel each, four nodes one sector each (with ``resolve_frequency_parity``
   there are four sectors; nodes beyond that idle through the solve). Each channel's pairing vertex is built into
   an MPI shared-memory window on every node that hosts one of its sectors, by all of that node's ranks in column
   blocks; on a symmetry-reduced grid only the irreducible wedge of the real-space grid is kept (the vertex at every
   other point follows from a point-group operation), and each matrix-vector product then contracts one star of
   symmetry-related points at a time. The node's ranks are split into one team per sector hosted there, and the
   team runs the whole eigensolver together: a restarted Krylov-Schur iteration whose basis vectors are split over
   the team's ranks by fermionic frequency, so every rank handles its block of momenta in the vertex contractions
   and its block of frequencies in the sector projections, the bubble multiply, the Fourier transforms and the
   orthogonalizations. The eigenvalues agree with a single-rank solve to the solver tolerance (the matrix-vector
   product is bit-identical on a symmetry-free grid and equal to rounding on a reduced one), so ``OMP_NUM_THREADS=1``
   with one rank per core is the right binding here as everywhere else. When a node cannot hold its share of vertex
   windows at once, the channels are solved one after the other, each spread over the nodes the same way. On a
   single-rank run the sectors are solved in turn on that rank with scipy's eigensolver, with the matrix-vector
   products threaded over the rank's CPU affinity mask. In both in-memory solves, for one band with a frequency-even
   pp bubble the crossed term of a projected sector is formed from the direct one, which halves the vertex
   contractions. When no node
   holds even one channel's window, the solver instead distributes the vertex over a two-dimensional
   frequency-block grid spanning all ranks: the sectors then run sequentially on the whole grid, every rank holds
   and contracts one vertex block, and the eigensolver iterates in lockstep with one block-sized reduction and one
   gap-sized gather per iteration.

Memory is managed automatically: every heavy step runs a single chunk-bounded or distributed algorithm, and before
the heavy part of a run begins, DGAmore verifies from the memory available on every node together with an analytic
estimate of each step's peak (as a node total over all ranks placed there) that the run fits, and sizes the chunks
of the auxiliary-susceptibility build, the self-energy passes and the pairing-vertex build from the memory that
estimate leaves free on the tightest node - the runtime choices left are the Eliashberg solver's automatic
fallback from its in-memory solve to the block-distributed grid and whether a node holds every vertex window of
its sectors at once or one channel at a time.
Replicated full-grid objects are always deduplicated into one shared-memory window per node. There are no memory
switches in the configuration file; if some step does not fit, the run stops upfront with a :class:`MemoryError`
recommending more nodes, fewer ranks per node, or a smaller box.

The full set of run-time parameters is described on the :doc:`configuration` page.

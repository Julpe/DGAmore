Configuration
=============

Run parameters are specified through a YAML configuration file. You can either create a new file or modify the
default one, ``dga_config.yaml``, which is shipped with the repository. Its default entries are chosen to be
sensible for typical use cases, so whenever a parameter is unclear the default can usually be kept; the only
exceptions are the input file paths, which always have to be set explicitly.

Individual settings within a section, or even whole sections, may be omitted, in which case DGAmore falls back to
the predefined defaults listed below alongside the variable type the code expects. The order of the sections in the
file does not matter for parsing; the sections are discussed here in the order in which they appear in the default
``dga_config.yaml``.

Matsubara frequencies
---------------------

The first section governs the frequency box sizes used throughout the calculation.

.. code-block:: yaml

   box_sizes:
     niw_core: -1   # int
     niv_core: -1   # int
     niv_shell: 0   # int

The "core" region defines the frequency box on which the Bethe-Salpeter and Schwinger-Dyson equations are solved
explicitly, while the "shell" region sets the size of the asymptotic tails used for the vertex reconstruction
through the U-range method. Here ``niw_core`` is the number of positive bosonic Matsubara frequencies and
``niv_core`` the number of positive fermionic ones, so that the objects carry ``2 * niw_core + 1`` bosonic and
``2 * niv_core`` fermionic frequencies in total. Setting either core size to ``-1`` instructs the code to take the
full number of positive frequencies available from the DMFT calculation; if a smaller box is requested, the DMFT
vertices are cut down to the specified size.

Lattice and symmetries
-----------------------

The next section describes the Hamiltonian and the lattice symmetries of the system.

.. code-block:: yaml

   lattice:
     symmetries: "auto"                     # str | list[str]
     type: "from_wannier90"                 # str
     hr_input: "/path/to/file"              # str | list[float]
     interaction_type: "one_band_from_dmft" # str
     interaction_input: ""                  # str
     nk: [ 16, 16, 1 ]                      # list[int]
     # nq: [ 16, 16, 1 ]                    # list[int]

The ``symmetries`` field controls how the irreducible Brillouin zone is built. Entering ``auto`` enables the
automatic symmetry discovery, which probes a large number of combined momentum and orbital transformations,
constructs the irreducible zone together with its map onto the full zone, and always yields the smallest possible
number of irreducible momenta; it is the preferred choice. Alternatively, one of the predefined symmetry sets, such
as ``two_dimensional_square`` or ``three_dimensional_cubic``, can be supplied, or a list of individual symmetries
can be passed explicitly. The supported single operations are ``x-inv``, ``y-inv`` and ``z-inv`` (inverting the
respective momentum component), ``x-y-sym``, ``x-z-sym`` and ``y-z-sym`` (exchanging two components), and
``x-y-inv`` (inverting both in-plane components at once). Passing
``[ "x-inv", "y-inv", "z-inv", "x-y-sym", "x-z-sym", "y-z-sym" ]`` is, for instance, equivalent to
``three_dimensional_cubic``. The predefined and individual operations always act on the primitive reciprocal
lattice axes, so they map the grids incorrectly when not every calculated orbital shares the requested symmetry, or
when the primitive reciprocal lattice vectors are not orthogonal; the automatic discovery is free of both
limitations because it inspects the Hamiltonian directly.

The ``type`` field selects the form in which the kinetic part of the Hamiltonian is provided. With
``from_wannier90`` the ``hr_input`` field points to a real-space Hamiltonian file holding the hopping elements,
whereas ``from_wannierHK`` expects a file giving the Hamiltonian directly in momentum space. The third option,
``t_tp_tpp``, is available for single-band input only and reads a list of three floating-point values for the
nearest, next-nearest and third-nearest neighbour hoppings from ``hr_input``, for example ``[1.0, -0.25, 0.12]``.

Because DGAmore supports multi-orbital calculations with non-local interactions, an interaction type and input must
be specified as well. With ``one_band_from_dmft`` a single-band calculation is assumed and the local interaction
strength is read from the DMFT input, while ``kanamori_from_dmft`` takes the interaction values from DMFT and builds
the Kanamori interaction tensor; in both cases ``interaction_input`` is ignored. Choosing ``custom`` instead lets
the code read the interaction from the file referenced in ``interaction_input``, which is structured like a
real-space Hamiltonian file but with four orbital indices instead of two and may also encode non-local
interactions.

Finally, ``nk`` and ``nq`` set the sizes of the momentum grid for the one-particle quantities and for the
ladder, respectively. The ``nq`` field may be left unset, in which case ``nk`` is reused for the q-grid. Note that
the Eliashberg equation can only be solved when the two grids have the same size.

Self-consistency
----------------

This section controls the self-consistency cycle.

.. code-block:: yaml

   self_consistency:
     max_iter: 20                  # int
     epsilon: 1e-4                 # float
     mixing: 0.2                   # float
     mixing_strategy: "linear"     # str
     mixing_history_length: 3      # int
     previous_sc_path: ""          # str
     use_interpolated_sigma: False # bool
     use_lambda_correction: False  # bool
     restrict_chi_phys: False      # bool

The ``max_iter`` field puts an upper limit on the number of iterations; if the self-energy has not converged by
then, the loop stops and the program exits. The non-local self-energy is written to the output folder for every
iteration. Convergence itself is judged by ``epsilon``: rather than comparing two successive self-energies
directly, the code monitors the relative residual of the Schwinger-Dyson iteration, that is the norm of the change
in the self-energy between consecutive iterations divided by the norm of the previous one, taken over the full
momentum grid, all orbital combinations and the positive fermionic frequencies of the core box. The cycle is
considered converged once this residual drops below ``epsilon``; in addition, the chemical potential is required to
change by less than a small temperature-dependent threshold between iterations.

The ``mixing`` parameter is a floating-point number between zero and one that sets the weight of the new
self-energy in the update. The accompanying ``mixing_strategy`` selects between ``linear``, ``pulay`` and
``anderson`` mixing; the latter two build a prediction from the previous ``mixing_history_length`` self-energies and
fall back to linear mixing while fewer iterations than the history length are available. The ``previous_sc_path``
field points to the folder of an earlier, possibly unconverged, self-consistency run: the code then resumes from
its last iterations and continues converging, applying Pulay or Anderson mixing right away if enough previous
iterations are available. Enabling ``use_interpolated_sigma`` makes the cycle start from the interpolated
self-energy of the previous run, with the interpolation itself configured in the
:ref:`self-energy interpolation section <self-energy-interpolation>`.

It is furthermore possible to apply the lambda correction throughout the entire self-consistency cycle for
single-band data, which can help stabilise convergence or reveal how a lambda-corrected cycle changes the results.
The correction type is taken from the ``lambda_correction`` section, and ``perform_lambda_correction`` there must be
set to ``True`` for it to take effect. Lastly, ``restrict_chi_phys`` clamps the inverse physical susceptibilities to
positive values, replacing any negative value by a small positive one. This sometimes stabilises early iterations
but generally yields unphysical results, so once the self-energy converges with this option enabled and iterations
remain, the option is switched off automatically and the cycle continues until convergence is reached again or
``max_iter`` is hit.

Lambda correction
------------------

DGAmore implements two lambda-correction schemes, both restricted to the single-band case.

.. code-block:: yaml

   lambda_correction:
     perform_lambda_correction: False # bool
     type: "spch"                     # str

When ``perform_lambda_correction`` is ``True``, the code applies the correction selected in ``type``: ``spch``
corrects both the density and the magnetic susceptibility, whereas ``sp`` corrects only the magnetic one.

DMFT input
----------

This section specifies the DMFT result files and the structure of the vertex.

.. code-block:: yaml

   dmft_input:
     type: "w2dyn"             # str
     input_path: "./"          # str
     fname_1p: "1p-data.hdf5"  # str
     fname_2p: "g4iw_sym.hdf5" # str
     do_sym_v_vp: True         # bool
     symmetrize_orbitals: []   # list[int] | list[list[int]]
     n_ineq: 1                 # int
     ineq_ordering: [ 1 ]      # list[int]

Currently only w2dynamics output is supported, though support for further DMFT solvers is planned. The result files
are expected in the folder given by ``input_path``, with ``fname_1p`` and ``fname_2p`` naming the one- and
two-particle results. Because the two-particle Green's function is symmetric under the exchange of its two fermionic
frequencies, this symmetrisation is enforced explicitly when ``do_sym_v_vp`` is ``True``.

The ``symmetrize_orbitals`` field allows the local DMFT quantities, namely the self-energy and the one- and
two-particle Green's functions, to be symmetrised over orbitals, which is well defined because these quantities are
purely local. For the three orbitals of a material such as :math:`\mathrm{SrVO}_3`, entering ``[1, 2, 3]``
symmetrises all three with respect to one another. Subsets and multiple subsets are equally possible: for a
four-orbital case in which orbitals one and three as well as orbitals two and four are locally equivalent, one
enters ``[[1, 3], [2, 4]]``.

The last two settings were introduced for multi-layered materials such as :math:`\mathrm{La}_3\mathrm{Ni}_2\mathrm{O}_7`.
There the DGA calculation uses a four-orbital model, while at the DMFT level the symmetry of the system lets one
treat pairs of orbitals as equivalent atoms, producing two-orbital DMFT quantities that must be mapped back onto the
full four-band orbital-diagonal space. The ``n_ineq`` field gives the number of inequivalent atoms treated in DMFT
(one for :math:`\mathrm{La}_3\mathrm{Ni}_2\mathrm{O}_7`), and ``ineq_ordering`` specifies how these are distributed
across the orbital-diagonal entries of the Hamiltonian; setting it to ``[1, 1]`` correctly populates the four-orbital
space by repeating the first atom's data. The entries may be ordered arbitrarily or repeated to suit different
layered geometries, as long as the total number of orbitals they represent matches the number of bands in the
Hamiltonian input.

Eliashberg equation
-------------------

Superconducting properties are obtained by solving the linearised Eliashberg equation, configured here.

.. code-block:: yaml

   eliashberg:
     perform_eliashberg: False    # bool
     save_pairing_vertex: False   # bool
     save_fq: False               # bool
     construct_fq_cheap: False    # bool
     n_eig: 4                     # int
     epsilon: 1e-6                # float
     symmetry: "random"           # str
     include_local_part: True     # bool
     subfolder_name: "Eliashberg" # str

The equation is solved only when ``perform_eliashberg`` is ``True``. Enabling ``save_pairing_vertex`` or ``save_fq``
writes the pairing vertex or the full ladder vertex on the irreducible Brillouin zone to the output folder. Since
assembling the full ladder vertex is very memory-intensive, ``construct_fq_cheap`` offers a variant that builds it
on a reduced fermionic frequency box and needs only a quarter of the memory; this slightly changes the eigenvalues
and gap functions but remains qualitatively reliable, as verified across many test cases, and is recommended only
for datasets that push the memory limits.

The equation is solved with a Lanczos algorithm based on the ARPACK routines, retrieving the ``n_eig`` largest
eigenvalues and the corresponding gap functions to an accuracy of ``epsilon``. The ``symmetry`` field sets the
starting vector of the iteration: entering ``"d-wave"``, for example, begins from a gap function with d-wave
symmetry, but ``"random"`` is sufficient most of the time. The pairing vertex requires local reducible diagrams,
which can be skipped by setting ``include_local_part`` to ``False``; this is only advisable when s-wave symmetry is
not expected, as these diagrams become relevant in that case. Finally, the results are written to a subfolder named
according to ``subfolder_name``.

.. _self-energy-interpolation:

Self-energy interpolation
-------------------------

When a low-temperature parameter set is hard to converge, a common strategy is to bootstrap the solution from a
converged result at a higher temperature, interpolating the self-energy to the target temperature and frequency grid
before entering the self-consistency cycle.

.. code-block:: yaml

   self_energy_interpolation:
     do_interpolation: False # bool
     target_beta: 1.0        # float
     target_niv: 10          # int

The interpolation runs when ``do_interpolation`` is ``True``. It applies a linear inter- or extrapolation for the
lowest frequencies and a PCHIP interpolation for the remaining ones, and writes the resulting self-energy for each
iteration at the new inverse temperature ``target_beta`` and the new number of positive fermionic frequencies
``target_niv`` to the output folder.

Output
------

This section collects further output settings.

.. code-block:: yaml

   output:
     output_path: ""                  # str
     do_plotting: True                # bool
     plotting_subfolder_name: "Plots" # str

When ``output_path`` is empty, the results are written to a subfolder created next to the DMFT input files;
otherwise they go to the folder it names. If ``do_plotting`` is enabled, a few quantities such as the self-energy
and the Green's function are plotted with matplotlib into the subfolder named by ``plotting_subfolder_name``.

Analytic continuation
---------------------

The code integrates support for continuing Green's functions from the Matsubara axis to the real-frequency axis
through the ``ana_cont`` package.

.. code-block:: yaml

   ana_cont:
     do_spectrum_dga: False                                     # bool
     do_spectrum_dmft: False                                    # bool
     w_count: 1001                                              # int
     plot_spectrum: True                                        # bool
     k_path: [ [0.0, 0.0, 0.0, "Gamma"], [0.0, 0.5, 0.0, "X"] ] # list[tuple]
     energy_window: [ -2, 3 ]                                   # list[float]

The flags ``do_spectrum_dga`` and ``do_spectrum_dmft`` toggle the continuation of the DGA and DMFT
Green's functions, respectively. The procedure yields the spectral function over a symmetric interval of ``w_count``
real-frequency points. When ``plot_spectrum`` is enabled, the spectral function is plotted along the path given in
``k_path``, a list of tuples whose first three elements are the coordinates of a high-symmetry point in the
primitive reciprocal lattice and whose fourth element is its label (the example above is truncated relative to the
default path). The ``energy_window`` parameter sets the bounds of the frequency axis in the resulting spectral
plots.

Memory efficiency
-----------------

For very large parameter sets memory becomes the main bottleneck, owing to the vectorised nature of the implemented
equations. This section therefore exposes more memory-efficient algorithms for five of the heaviest steps.

.. code-block:: yaml

   memory:
     save_memory_for_chi0q: False    # bool
     save_memory_for_chiq_aux: False # bool
     save_memory_for_sde: False      # bool
     save_memory_for_fq: False       # bool
     save_memory_for_lanczos: False  # bool

These switches control, in turn, the construction of the bare bubble susceptibility, of the auxiliary
susceptibility entering the Schwinger-Dyson equation, of the Schwinger-Dyson equation itself, of the full ladder
vertices for the Eliashberg equation, and of the Lanczos algorithm. Enabling any of them increases the runtime
substantially because of the additional Python and MPI communication. Under the hood, the bubble and the
Schwinger-Dyson steps use fast Fourier transforms when their switches are left at ``False``; this gives by far the
largest speed-up while barely affecting the memory footprint, so it can be kept at the default in almost all cases.
The largest memory savings come from the auxiliary-susceptibility and full-ladder-vertex switches, which shrink
those objects considerably, whereas the Lanczos switch matters only for extremely large parameter sets and can
usually stay disabled.

In practice these switches rarely need to be set by hand. Before the heavy part of a run begins, DGAmore inspects
the memory available on every node together with an analytic estimate of the peak memory each of the five steps
consumes, accounting for how the momentum points are distributed across the MPI ranks that share a node. Whenever
the default, faster variant of a step would not fit, the corresponding switch is turned on automatically. The
estimate is evaluated as a node total: it sums, over all ranks placed on a node, the data that each rank keeps
resident throughout the calculation plus the transient peak of the step in question, and requires the result to stay
below ninety percent of that node's available memory. Because the switches act process-wide, the most constrained
node decides whether a given switch is enabled.

A switch that is explicitly set to ``True`` in the configuration is always honoured: the automatic detection can
only enable additional switches, never turn off one that was requested. Conversely, if even the most memory-frugal
variant of a required step does not fit on some node, the run stops immediately with a ``MemoryError`` that
recommends using more nodes, fewer ranks per node, or a smaller frequency box or momentum grid, rather than failing
unpredictably partway through.

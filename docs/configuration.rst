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
nearest, next-nearest and third-nearest neighbor hoppings from ``hr_input``, for example ``[1.0, -0.25, 0.12]``.

Because DGAmore supports multi-orbital calculations with non-local interactions, an interaction type and input must
be specified as well. With ``one_band_from_dmft`` a single-band calculation is assumed and the local interaction
strength is read from the DMFT input, while ``kanamori_from_dmft`` takes the interaction values from DMFT and builds
the Kanamori interaction tensor; in both cases ``interaction_input`` is ignored. Choosing ``custom`` instead lets
the code read the interaction from the file referenced in ``interaction_input``, which is structured like a
real-space Hamiltonian file but with four orbital indices instead of two and may also encode non-local
interactions.

Finally, ``nk`` sets the size of the momentum grid, which is shared by the one-particle quantities and the
ladder (the q-grid always equals the k-grid).

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

The ``max_iter`` field puts an upper limit on the number of iterations; if the self-energy has not converged by
then, the loop stops and the program exits. The non-local self-energy is written to the output folder for every
iteration. Convergence itself is judged by ``epsilon`` through the relative *step* residual of the Schwinger-Dyson
iteration - the norm of the change of the mixed self-energy between consecutive iterations, divided by the norm of
the previous one - taken over the full momentum grid, all orbital combinations and the positive fermionic
frequencies of the core box (reported in the log every iteration); in addition, the chemical potential is required
to change by less than a small temperature-dependent threshold between iterations.

The ``mixing`` parameter is a floating-point number between zero and one that sets the weight of the new
self-energy in the update. The accompanying ``mixing_strategy`` selects between ``linear``, ``pulay`` and
``anderson`` mixing; the latter two build a prediction from the previous ``mixing_history_length`` self-energies and
fall back to linear mixing while fewer iterations than the history length are available. The ``previous_sc_path``
field points to the folder of an earlier, possibly unconverged, self-consistency run: the code then resumes from
its last iterations and continues converging, applying Pulay or Anderson mixing right away if enough previous
iterations are available. Enabling ``use_interpolated_sigma`` makes the cycle start from the interpolated
self-energy of the previous run, with the interpolation itself configured in the
:ref:`self-energy interpolation section <self-energy-interpolation>`.

.. _stabilization:

Stabilization
-------------

This section collects the convergence-stabilization options of the self-consistency cycle.

.. code-block:: yaml

   stabilization:
     use_lambda_correction: False      # bool
     use_chi_phys_restriction: False   # bool
     use_lambda_annealing: False       # bool

Setting ``use_lambda_correction`` to ``True`` applies the lambda correction to the physical susceptibilities
in every iteration of the cycle. The correction is dispatched by the band count (and the choice is logged):
single-band data uses the scalar Moriya correction with the scheme taken from the ``type`` field of the
:ref:`lambda correction section <lambda-correction>`. The Moriya lambda correction has formally only been derived
for single-band models; for multi-band data the code therefore applies the similar, heuristic scheme described
below purely to stabilize convergence (it always corrects both channels). In both cases the correction acts as a
*releasing scaffold*: the loop converges at a relaxed
(tenfold) threshold with the correction on, then automatically disables it, resets the mixing history and converges
the *uncorrected* self-consistent solution to the full ``epsilon``, so the corrected susceptibilities never enter
the final result. It is independent of the one-shot ``perform_lambda_correction`` of the
:ref:`lambda correction section <lambda-correction>`, which takes precedence when both are enabled.

The multi-band scheme mimics the scalar correction without being derived from it: instead of one
scalar per channel it calibrates a full :math:`N_o^2 \times N_o^2` real-symmetric mass matrix :math:`\Lambda_r` per
channel (both density and magnetic), added to the compound inverse susceptibility so that the momentum- and
frequency-summed corrected susceptibility matches the local (impurity) sum rule component by component. The number
of conditions matches the number of parameters exactly, so the calibration is a well-posed matrix root problem
solved by a damped Newton iteration (in the physically relevant regime the map is strictly monotone, hence the root
is unique); it runs in double precision and is line-searched to keep the static susceptibility gap positive. The
momentum sum is taken over the *full* Brillouin zone rather than by weighting the irreducible-zone value with its
multiplicity, because symmetry-related momenta carry orbitally-rotated susceptibility matrices that only combine
correctly once the star is summed explicitly. For a single band the matrix collapses to a scalar and the scheme
reduces to the ordinary lambda correction.

The option ``use_chi_phys_restriction`` regularizes the physical
susceptibilities: per momentum and bosonic frequency, the eigenvalues of the Hermitian part of the inverse
susceptibility are floored at a small positive value (a negative eigenvalue of the inverse marks a crossed pole of
the Bethe-Salpeter equation), while healthy eigenpairs - including legitimately negative off-diagonal matrix
elements - pass through unchanged.
This stabilizes cycles that would otherwise converge onto an unphysical branch, but the floored blocks are a
regularization rather than physics, so once the self-energy converges with this option enabled and iterations
remain, the option is switched off automatically and the cycle continues until convergence is reached again or
``max_iter`` is hit. The restricted phase only serves as a scaffold for the unrestricted one, so it is converged to
a relaxed threshold of ten times ``epsilon``; at the release the mixing history is reset (the accelerated schemes
must not extrapolate across the discontinuity) and the unrestricted phase then converges to the full ``epsilon``.
The log reports the number of floored eigenvalues per iteration - releasing is only promising once that count has
decayed to zero - and the minimum static eigenvalue of every channel's susceptibility, which certifies whether the
final state is physical. Combining ``use_chi_phys_restriction`` with the lambda correction is not recommended (the lambda
correction would calibrate its sum rule on eigenvalue-floored susceptibilities); if both are enabled, the lambda
correction takes precedence and ``use_chi_phys_restriction`` is disabled automatically with a warning.

Setting ``use_lambda_annealing`` to ``True`` protects the cycle with the *lambda-annealing scaffold*: a bosonic
mass :math:`\lambda` is added to the inverse physical susceptibility of every channel, which damps the
susceptibility and keeps the Bethe-Salpeter pole at bay - a sum-rule-free alternative to the lambda
correction. A single *shared* mass is used for all channels (they are coupled through the self-energy,
so per-channel masses would chase each other to unphysical values); it is sized automatically from the worst
channel's static (:math:`\omega=0`) gap, raised toward its target with damping so it tracks the self-energy's own
relaxation, and clamped at a ceiling past which the pole is too deep for the scaffold (the log then advises a warmer
start). The mass is never chosen by the user: a healthy susceptibility leaves the scaffold inert, and the mass is
halved between converged phases and re-armed automatically if a pole reopens mid-run. Phases converge at ten times ``epsilon``; only a converged phase with **all masses at exactly zero**
counts as the final result, so the returned self-energy is always pure self-consistency - the scaffold merely
guides the iteration there. If the pure phase cannot converge, the run stops at ``max_iter`` with the honest
verdict that no stable physical fixed point was found. It is mutually exclusive with the other
susceptibility-reshaping options - the lambda correction and ``use_chi_phys_restriction`` (the sum rule must not be
calibrated on mass-shifted susceptibilities, and the two scaffolds would fight); if combined, a lambda correction
or ``use_chi_phys_restriction`` takes precedence and the annealing scaffold is disabled with a warning.

.. _lambda-correction:

Lambda correction
------------------

This section controls the one-shot lambda correction.

.. code-block:: yaml

   lambda_correction:
     perform_lambda_correction: False # bool
     type: "spch"                     # str

When ``perform_lambda_correction`` is ``True``, the code performs a one-shot DGA with lambda correction:
``max_iter`` is overridden to 1 and ``mixing`` to 1.0, and the correction is applied once, dispatched by the band
count exactly like ``stabilization.use_lambda_correction`` (over which it takes precedence when both are enabled);
recall that for multi-band data this is the heuristic stabilization scheme of the
:ref:`stabilization section <stabilization>`, not a derived lambda correction.
For single-band data the ``type`` field selects the correction scheme: ``spch`` corrects both the density and the
magnetic susceptibility, whereas ``sp`` corrects only the magnetic one; the multi-band scheme always
corrects both channels.

DMFT input
----------

This section specifies the DMFT result files and the structure of the vertex.

.. code-block:: yaml

   dmft_input:
     type: "w2dyn"             # str
     input_path: "./"          # str
     fname_1p: "1p-data.hdf5"  # str
     fname_2p: "g4iw_sym.hdf5" # str
     symmetrize_orbitals: []   # list[int] | list[list[int]]
     n_ineq: 1                 # int
     ineq_ordering: [ 1 ]      # list[int]

Currently only w2dynamics output is supported, though support for further DMFT solvers is planned. The result files
are expected in the folder given by ``input_path``, with ``fname_1p`` and ``fname_2p`` naming the one- and
two-particle results. The two-particle Green's function is always symmetrized under the exchange of its two
fermionic frequencies :math:`(\nu, \nu')` on load: this is the time-reversal-plus-inversion symmetry that makes the
right three-leg vertex the first-frequency-summed transpose of the left one, so both are obtained from a single
auxiliary susceptibility sum.

The ``symmetrize_orbitals`` field allows the local DMFT quantities, namely the self-energy and the one- and
two-particle Green's functions, to be symmetrized over orbitals, which is well defined because these quantities are
purely local. For the three orbitals of a material such as :math:`\mathrm{SrVO}_3`, entering ``[1, 2, 3]``
symmetrizes all three with respect to one another. Subsets and multiple subsets are equally possible: for a
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

Superconducting properties are obtained by solving the linearized Eliashberg equation, configured here.

.. code-block:: yaml

   eliashberg:
     perform_eliashberg: False    # bool
     save_pairing_vertex: False   # bool
     save_fq: False               # bool
     n_eig: 4                     # int
     epsilon: 1e-6                # float
     symmetry: "random"           # str
     include_local_part: True     # bool
     symmetrize_degenerate_gaps: True # bool
     resolve_frequency_parity: True # bool
     subfolder_name: "Eliashberg" # str

The equation is solved only when ``perform_eliashberg`` is ``True``. Enabling ``save_pairing_vertex`` or ``save_fq``
writes the pairing vertex or the full ladder vertex on the irreducible Brillouin zone to the output folder.

The equation is solved with a Lanczos algorithm based on the ARPACK routines, retrieving the ``n_eig`` largest
eigenvalues and the corresponding gap functions to an accuracy of ``epsilon``. The ``symmetry`` field sets the
starting vector of the iteration: entering ``"d-wave"``, for example, begins from a gap function with d-wave
symmetry, but ``"random"`` is sufficient most of the time. The pairing vertex includes local reducible diagrams by default,
which can be skipped by setting ``include_local_part`` to ``False``; this is only advisable when s-wave symmetry is
not expected, as these diagrams become relevant in that case. With ``symmetrize_degenerate_gaps`` enabled (the
default), gap functions belonging to (near-)degenerate eigenvalues are orthogonalized with a Loewdin scheme and
degenerate doublets are rotated to their mirror-adapted (:math:`p_x`/:math:`p_y`-like) partners.

The ``resolve_frequency_parity`` field controls whether the physical gap-symmetry sectors are returned. The
unprojected eigensolver leaks onto the globally dominant eigenvector, mixing frequency-even and frequency-odd modes;
projecting the trial vector onto a fixed symmetry sector at every iteration separates them.

The orbital gap :math:`\Delta_{12}(k, \nu)` (with the spin part folded into the channel) acts on three commuting
involutions built from the same array operations as the pairing kernel: the frequency flip
:math:`(T\Delta)_{12}(k, \nu) = \Delta_{12}(k, -\nu)`, the momentum flip
:math:`(P\Delta)_{12}(k, \nu) = \Delta_{12}(-k, \nu)` and the orbital transpose
:math:`(O\Delta)_{12}(k, \nu) = \Delta_{21}(k, \nu)`. The Pauli principle requires the Cooper-pair amplitude to be
antisymmetric under the combined exchange of spin, momentum, orbital and frequency,

.. math::

   \hat{S}\, P\, O\, T\, \Delta = -\Delta ,

where :math:`\hat{S}` is the spin exchange (a scalar :math:`s_S = -1` for the singlet, :math:`+1` for the triplet).
Writing :math:`G \equiv P\,O\,T`, physical gaps are eigenvectors :math:`G\Delta = g_S\,\Delta` with
:math:`g_S = -s_S`, which equals the channel sign ``sign`` (:math:`+1` singlet, :math:`-1` triplet) that the kernel
already carries. Choosing the frequency parity :math:`\varepsilon_T \in \{+1\ (\text{even}), -1\ (\text{odd})\}`
therefore fixes the combined momentum-orbital parity through
:math:`\varepsilon_{PO} = \mathrm{sign}\cdot\varepsilon_T`:

.. list-table::
   :header-rows: 1

   * - channel
     - ``sign``
     - :math:`\varepsilon_T` (even / odd)
     - forced :math:`\varepsilon_{PO}`
   * - singlet
     - :math:`+1`
     - :math:`+1` / :math:`-1`
     - :math:`+1` / :math:`-1`
   * - triplet
     - :math:`-1`
     - :math:`+1` / :math:`-1`
     - :math:`-1` / :math:`+1`

The gap is projected onto the sector with the commuting Hermitian projector (only the product :math:`P\,O` is fixed,
never :math:`P` and :math:`O` separately)

.. math::

   \Pi = \tfrac{1}{2}\left(1 + \varepsilon_T\, T\right)\,\tfrac{1}{2}\left(1 + \varepsilon_{PO}\, P\,O\right) ,

applied to both the matrix-vector product and the starting vector at every iteration. Setting
``resolve_frequency_parity`` to ``True`` returns the frequency-even and frequency-odd sectors for each of the
singlet and triplet channels, while ``False`` (the default) returns the overall leading eigenpairs without any
projection.
Clean frequency-parity separation additionally requires the pairing vertex to satisfy the time-reversal-plus-inversion
symmetry :math:`\Gamma^{\nu\nu'}(q) = \Gamma^{(-\nu)(-\nu')}(-q)`. Sector-resolved gap functions are written as
``gap_<channel>_<parity>_<i>`` (for example ``gap_sing_even_1``), while the unprojected case keeps the
``gap_<channel>_<i>`` naming. A ``gap_parity.txt`` file is written alongside them with one line per saved gap: a
compact wave-symmetry label (``s`` / ``d`` / ``p``, or ``x`` if unclassified, followed by ``+`` / ``-`` for the
frequency parity) obtained from the dominant orbital-diagonal momentum structure, followed by the measured parity
Rayleigh quotients :math:`\langle \Delta, X\Delta \rangle / \langle \Delta, \Delta \rangle` for
:math:`X \in \{T, P, O, P O\}`. Finally, the results are written to a subfolder named according to ``subfolder_name``.

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
lowest frequencies and a PCHIP interpolation for the remaining ones. The resulting self-energy, now at the new
inverse temperature ``target_beta`` and the new number of positive fermionic frequencies ``target_niv``, is written
to the output folder for each iteration.

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

For very large parameter sets memory becomes the main bottleneck, owing to the vectorized nature of the implemented
equations. This section therefore exposes more memory-efficient algorithms for four of the heaviest steps.

.. code-block:: yaml

   memory:
     save_memory_for_chi0q: False    # bool
     save_memory_for_chiq_aux: False # bool
     save_memory_for_fq: False       # bool
     save_memory_for_lanczos: False  # bool
     use_shared_memory_common_obj: True    # bool

The first four switches control, in turn, the construction of the bare bubble susceptibility, of the auxiliary
susceptibility entering the Schwinger-Dyson equation, of the full ladder vertices for the Eliashberg equation, and
of the Lanczos algorithm. Enabling any of them increases the runtime substantially because of the additional
Python-level looping and MPI communication. Under the hood, the bubble uses fast Fourier transforms when its switch
is left at ``False``; this gives by far the largest speed-up while barely affecting the memory footprint, so it can
be kept at the default in almost all cases. The Schwinger-Dyson equation itself has no switch: it always runs its
Fourier-transformed form, which processes the two bosonic frequency halves in separate passes and thereby needs no
more memory than the alternative momentum-loop formulation would. The largest memory savings come from the
auxiliary-susceptibility and full-ladder-vertex switches, which shrink those objects considerably, whereas the
Lanczos switch matters only for extremely large parameter sets and can usually stay disabled.

In practice these switches rarely need to be set by hand. Before the heavy part of a run begins, DGAmore inspects
the memory available on every node together with an analytic estimate of the peak memory each of the heavy steps
(including the switch-less Schwinger-Dyson contraction) consumes, accounting for how the momentum points are
distributed across the MPI ranks that share a node. Whenever
the default, faster variant of a step would not fit, the corresponding switch is turned on automatically. The
estimate is evaluated as a node total: it sums, over all ranks placed on a node, the data that each rank keeps
resident throughout the calculation plus the transient peak of the step in question, and requires the result to stay
below ninety-seven percent of that node's available memory. Because the switches act process-wide, the most constrained
node decides whether a given switch is enabled.

A switch that is explicitly set to ``True`` in the configuration is always honored: the automatic detection can
only enable additional switches, never turn off one that was requested. Conversely, if the variant of a step that is
about to run does not fit on some node - because neither of its variants fits, or because a switch forced by the
configuration selects a variant that does not - the run stops immediately with a ``MemoryError`` that recommends
using more nodes, fewer ranks per node, or a smaller frequency box or momentum grid, rather than failing
unpredictably partway through.

The final switch, ``use_shared_memory_common_obj``, is of a different kind and is enabled by default. The full-grid lattice
Green's function is identical on every rank, yet each rank would normally rebuild and store its own copy, so a node
running many ranks holds that (large) array many times over. When this switch is on, the ranks that share a physical
node instead keep a single copy in one MPI shared-memory window: the Dyson inversion is performed only by that node's
first rank and the others map the same buffer read-only. The real-space copy of the Green's function used by the
Schwinger-Dyson contraction is deduplicated through the same mechanism (built once per node, with each rank keeping
only its slice of real-space points). The node topology is discovered automatically at runtime, so nothing about the
cluster needs to be configured. Independently of this switch, the Eliashberg step no longer holds replicated
full-grid objects: ``sigma_dga`` is freed on every rank before the step (it is already saved to disk) and
``giwk_dga`` survives only on the rank that builds the pairing bubble.

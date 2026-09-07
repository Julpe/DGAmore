.. _cooldown:

Cooldown and stabilization
==========================

At high temperature the self-consistency cycle of the :doc:`configuration` page is well behaved. The ladder
susceptibilities sit far from their poles, and the DMFT self-energy is already close to the solution, so a cold
start converges without much trouble. Neither holds once the temperature drops. The fixed point drifts away from
the DMFT input, the static susceptibility of the dominant channel creeps toward the pole of the Bethe-Salpeter
equation, and a cycle started cold from DMFT needs many more iterations or does not settle at all. DGAmore offers two
tools for this regime. A *cooldown* is a chain of runs at decreasing temperature, each one warm-started from the
converged self-energy of the previous, warmer run, so that the cycle only has to close a small remaining gap. The
*stabilization techniques* reshape the physical susceptibility while the cycle iterates in order to keep it on the
physical branch of the ladder, and they withdraw again before the result counts. This page covers the cooldown
first, then the three stabilization techniques.

The cooldown procedure
----------------------

A cooldown is a ladder of self-consistent runs at inverse temperatures :math:`\beta_1 < \beta_2 < \ldots`; each run
is called a rung below. A rung is an ordinary DGAmore run with its own DMFT input at its own temperature. What sets
it apart from a stand-alone run is only the starting point of the self-consistency cycle: instead of the DMFT
self-energy, the cycle starts from the converged self-energy of the previous rung, re-gridded from that rung's
Matsubara frequencies onto its own. Neighboring temperatures have neighboring fixed points, so the warm start lands
close to the solution. Fewer iterations are needed, and the iteration is less likely to wander onto an unphysical
branch on the way.

The whole procedure hinges on the predecessor being *converged*. A converged self-energy from a somewhat higher
temperature is the best approximation to the lower-temperature solution one can get. It already carries the
momentum dependence and the low-frequency structure the DMFT self-energy lacks, and it places the next rung inside
the basin of attraction of its fixed point, where the cycle needs a fraction of the iterations of a cold start and
frequently converges where a cold start would not. Each converged rung thus makes the next, colder one easier, and
this is how the ladder reaches temperatures a direct calculation cannot. An unconverged predecessor gives none of
that. It passes on its own remaining distance from the fixed point on top of the temperature step, and the colder
rung has to pay for both.

What a rung needs
~~~~~~~~~~~~~~~~~

A rung needs three things before it can start.

1. **A DMFT calculation at the rung's temperature.** A run reads its inverse temperature from the DMFT input, not
   from the configuration file. The local vertex functions depend on temperature and are never interpolated, so
   every rung recomputes the local Schwinger-Dyson step from its own one- and two-particle files, prepared with the
   ``symmetrize`` script as usual. All rungs describe the same model, i.e. the same Hamiltonian, interaction and
   filling; only the temperature changes.

2. **A converged previous rung.** The warm start is only as good as its source. Chaining from a rung that stopped at
   ``max_iter`` throws away the advantage the procedure is built on. The technical minimum is lower: the predecessor
   must have completed its cycle, because the chemical potential is read from its ``mu_history.npy``, and that file
   is only written when the cycle ends, whether by convergence or by exhausting ``max_iter``. A run that was killed
   while iterating cannot serve as a predecessor at all.

3. **The predecessor's self-energy on this rung's grid.** The hand-over reads the interpolated self-energy the
   previous rung wrote, and that file only exists if the predecessor ran with the interpolation switched on and aimed
   at this rung: ``do_interpolation`` set to ``True``, ``target_beta`` equal to this rung's inverse temperature, and
   ``target_niv`` at least this rung's ``niv_core`` (the value it resolves to, i.e. the size of the DMFT vertex box
   when it is left at ``-1``).

Configuring a rung
~~~~~~~~~~~~~~~~~~

Apart from the DMFT input path, a rung's configuration differs from a stand-alone run in two sections. The
self-consistency section names the predecessor and asks for its interpolated self-energy; the interpolation section
points at the *next* rung. The two files below show this for the last step of a single-band ladder that ends at
:math:`\beta = 25`: the configuration of the :math:`\beta = 20` rung, which writes the hand-over self-energy, and
that of the :math:`\beta = 25` rung, which starts from it. The DMFT data of the two temperatures are assumed to live
in ``/data/beta20/`` and ``/data/beta25/``, and the :math:`\beta = 20` rung was itself chained from a
:math:`\beta = 15` rung in ``/data/beta15/``. Sections that are omitted, such as ``ana_cont`` here, fall back to
their defaults.

.. _cooldown-config-beta20:

The rung at :math:`\beta = 20`, ``dga_config_beta20.yaml``:

.. code-block:: yaml

   box_sizes:
     niw_core: 60
     niv_core: 60
     niv_shell: 50

   lattice:
     symmetries: "auto"
     type: "from_wannier90"
     hr_input: "/data/wannier_hr.dat"
     interaction_type: "one_band_from_dmft"
     interaction_input: ""
     nk: [ 64, 64, 1 ]

   self_consistency:
     max_iter: 60
     epsilon: 1e-6
     mixing: 0.4
     mixing_strategy: "anderson"
     mixing_history_length: 4
     previous_sc_path: "/data/beta15/LDGA_Nk4096_Nq4096_wc60_vc60_vs50" # run folder of the converged beta = 15 rung
     use_interpolated_sigma: True                                        # start from its interpolated self-energy

   stabilization:
     use_lambda_correction: False
     use_chi_phys_restriction: False
     use_lambda_annealing: False

   lambda_correction:
     perform_lambda_correction: False # has to stay False in a ladder, see below
     type: "spch"

   dmft_input:
     type: "w2dyn"
     input_path: "/data/beta20/" # this rung's own DMFT files; beta = 20 is read from them
     fname_1p: "1p-data.hdf5"
     fname_2p: "g4iw_sym.hdf5"
     symmetrize_orbitals: []
     n_ineq: 1
     ineq_ordering: [ 1 ]

   eliashberg:
     perform_eliashberg: True
     save_pairing_vertex: False
     save_fq: False
     n_eig: 4
     epsilon: 1e-6
     symmetry: "random"
     symmetrize_degenerate_gaps: True
     resolve_frequency_parity: True
     subfolder_name: "Eliashberg"

   self_energy_interpolation:
     do_interpolation: True # write the hand-over self-energy in every iteration ...
     target_beta: 25.0      # ... on the Matsubara grid of the beta = 25 rung ...
     target_niv: 60         # ... with at least that rung's niv_core positive frequencies

   output:
     output_path: "" # empty: the run folder is created inside input_path
     do_plotting: True
     plotting_subfolder_name: "Plots"

.. _cooldown-config-beta25:

The rung at :math:`\beta = 25`, ``dga_config_beta25.yaml``, differs in exactly three places, marked below:

.. code-block:: yaml

   box_sizes:
     niw_core: 60
     niv_core: 60
     niv_shell: 50

   lattice:
     symmetries: "auto"
     type: "from_wannier90"
     hr_input: "/data/wannier_hr.dat"
     interaction_type: "one_band_from_dmft"
     interaction_input: ""
     nk: [ 64, 64, 1 ]

   self_consistency:
     max_iter: 60
     epsilon: 1e-6
     mixing: 0.4
     mixing_strategy: "anderson"
     mixing_history_length: 4
     previous_sc_path: "/data/beta20/LDGA_Nk4096_Nq4096_wc60_vc60_vs50" # CHANGED: the converged beta = 20 rung
     use_interpolated_sigma: True

   stabilization:
     use_lambda_correction: False
     use_chi_phys_restriction: False
     use_lambda_annealing: False

   lambda_correction:
     perform_lambda_correction: False
     type: "spch"

   dmft_input:
     type: "w2dyn"
     input_path: "/data/beta25/" # CHANGED: this rung's own DMFT files
     fname_1p: "1p-data.hdf5"
     fname_2p: "g4iw_sym.hdf5"
     symmetrize_orbitals: []
     n_ineq: 1
     ineq_ordering: [ 1 ]

   eliashberg:
     perform_eliashberg: True
     save_pairing_vertex: False
     save_fq: False
     n_eig: 4
     epsilon: 1e-6
     symmetry: "random"
     symmetrize_degenerate_gaps: True
     resolve_frequency_parity: True
     subfolder_name: "Eliashberg"

   self_energy_interpolation:
     do_interpolation: False # CHANGED: last rung; set True with target_beta: 30.0 to continue the ladder
     target_beta: 25.0
     target_niv: 60

   output:
     output_path: ""
     do_plotting: True
     plotting_subfolder_name: "Plots"

Each rung is then started as usual, one after the other:

.. code-block:: bash

   mpiexec -np 16 DGAmore -p /configs/ -c dga_config_beta20.yaml
   mpiexec -np 16 DGAmore -p /configs/ -c dga_config_beta25.yaml

The settings deserve a closer look, section by section.

**Box sizes and lattice.** These describe the model and are the same on both rungs. The box sizes are given
explicitly rather than as ``-1`` so that the next rung's ``niv_core`` is known when ``target_niv`` is chosen; with
``-1`` one would have to look up the vertex box of the :math:`\beta = 25` DMFT file first. The momentum grid may
differ between rungs, since the hand-over re-samples the self-energy, but the Hamiltonian and interaction may not.

**The chain.** ``previous_sc_path`` names the *run folder* of the previous rung, not the folder holding its DMFT
data. A run creates that folder inside ``output_path`` (empty here, so inside ``input_path``) and names it after
the grid and the box, ``LDGA_Nk<nk_tot>_Nq<nk_tot>_wc<niw_core>_vc<niv_core>_vs<niv_shell>``; with the
:math:`64 \times 64 \times 1` grid and the box above that is ``LDGA_Nk4096_Nq4096_wc60_vc60_vs50``. A rerun with the
same settings does not overwrite an existing folder but gets a ``_1``, ``_2``, ... suffix, so after repeating a
rung make sure the path names the folder you mean. ``use_interpolated_sigma`` is ``True`` on both rungs, because
each of them starts from a warmer predecessor. The first rung of the ladder is the only one that leaves
``previous_sc_path`` empty and starts cold from DMFT; it still has the interpolation switched on.

**The interpolation.** On the :math:`\beta = 20` rung, ``do_interpolation`` is on and ``target_beta`` is
``25.0``, the inverse temperature of the DMFT data in ``/data/beta25/``. The two have to match exactly: the
hand-over file carries no metadata, so the :math:`\beta = 25` rung takes its content at face value, and a mismatch
goes unnoticed. ``target_niv`` is ``60``, the ``niv_core`` of the next rung. A larger value is harmless: the next
rung cuts the file to its core box anyway and fills the tail from its own DMFT self-energy, so a generous value
only costs disk space. A smaller value means the DMFT self-energy also has to fill the rest of the core box. The
interpolated self-energy is written in every iteration, and the next rung picks the file with the highest
iteration number, i.e. the converged one if the rung converged. The ratio :math:`25/20 = 1.25` keeps the
re-gridding mild: only the lowest target frequency lands on the straight-line guess. On the :math:`\beta = 25`
rung the interpolation is switched off because the ladder ends there; to go on to :math:`\beta = 30` one would
leave it on with ``target_beta: 30.0``.

**The mixing.** Both rungs use Anderson mixing with ``mixing: 0.4`` and a history of four pairs. After the
hand-over the first four iterations mix linearly at :math:`0.4` while pairs of the new map accumulate (the log
counts them: ``Using the last k (iterate, proposal) pairs of the mixing history.``), and from the fifth iteration
on the log reports ``Anderson acceleration applied (m=4, alpha=0.400, ...)``. The damping of :math:`0.4` is a
compromise. The default of :math:`0.2` is on the weak side for a low-temperature rung, where the quasi-Newton
correction needs room to cancel the expanding direction, whereas noticeably larger values make the first linear
steps big enough to risk overshooting the pole. A history of four is a modest step up from the default of three.
Neither number is universal; they are the levers to turn when a rung struggles, and cheap to explore on the warm
rungs.

**Threshold and budget.** ``epsilon: 1e-6`` is a reasonable choice when the Eliashberg eigenvalues of every rung
are of interest, since they react to the low-frequency self-energy that a looser threshold leaves undetermined.
``max_iter: 60`` gives each rung room to converge; the iteration count of the :math:`\beta = 25` rung continues
where the :math:`\beta = 20` rung stopped, so if that one converged at iteration 87, the log of the next rung starts
at iteration 88 and may run to 147.

**Stabilization.** All three flags are off in a first attempt. Should the :math:`\beta = 25` rung stall or bounce
and report pole warnings, one of them is switched on for that rung alone; the flags are per rung, and a rung that
needed a scaffold can still serve as a converged predecessor. ``perform_lambda_correction`` in the lambda correction
section has to stay ``False``. It turns a run into a one-shot calculation by forcing ``max_iter`` to 1 and
``mixing`` to 1.0, and the ladder is then silently broken; the only sign is the line
``Performing one-shot DGA with lambda correction`` in the log.

**Eliashberg.** Solving the Eliashberg equation on every rung yields the temperature dependence of the leading
eigenvalues along the ladder. The eigenvalues of a rung are only meaningful if that rung converged.

Resuming a run *at the same temperature* to add iterations uses the same mechanism with ``use_interpolated_sigma``
set to ``False``; the run then continues from the predecessor's raw last iterate instead of the interpolated one.

What happens at the hand-over
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the start of the cycle, the predecessor's result becomes the rung's starting point in the following steps.

* The cycle picks the file ``sigma_dga_interpolated_*_iteration_<i>.npy`` with the highest iteration number in
  ``previous_sc_path`` (without ``use_interpolated_sigma`` it looks for the raw ``sigma_dga_iteration_<i>.npy``
  files instead). The file carries no metadata, so its content is taken to be a self-energy at the rung's own
  inverse temperature. A predecessor whose ``target_beta`` does not match this rung's temperature therefore goes
  *undetected* and quietly mislabels the frequencies.
* The self-energy is cut to the rung's ``niv_core``. If the rung uses a different momentum grid, it is re-sampled
  onto it, exactly by striding when the new grid is a sub-lattice of the old one and by band-limited Fourier
  interpolation otherwise, so the grid may be refined along the ladder. Beyond the core box, the rung's own DMFT
  self-energy supplies the tail, as in any cold run.
* The chemical potential starts at the last entry of the predecessor's ``mu_history.npy`` and gets re-adjusted to
  the filling from the first iteration on.
* The iteration count carries on. A predecessor that ended at iteration :math:`N` hands over to iteration
  :math:`N + 1`, the rung performs up to ``max_iter`` further iterations on top, and its per-iteration files keep
  that numbering.
* The accelerated mixing schemes start with an empty history. For the first ``mixing_history_length`` iterations
  the rung mixes linearly with the configured ``mixing`` while (iterate, proposal) pairs of the *new* map
  accumulate; only then do Pulay or Anderson take over. Secant pairs from the previous temperature are deliberately
  left behind. They describe a different map, and an accelerated scheme fed with them explains the new residual
  through the old map and parks the iteration at the predecessor's solution while the step residual reports
  convergence.

The re-gridding treats real and imaginary parts separately on the full signed frequency axis. The innermost
frequencies, where the grid is sparsest, are interpolated linearly and everything above them with shape-preserving
PCHIP splines. Target frequencies smaller in magnitude than the lowest source frequency fall onto the straight line
between the lowest negative and the lowest positive source frequency. Since a shape-preserving interpolant never
overshoots its data, a causal self-energy stays causal under the hand-over.

How demanding the re-gridding is depends on the *ratio* of the two inverse temperatures, not on their difference.
The Matsubara grids :math:`\nu_n = (2n + 1)\pi/\beta` of two temperatures are related by a pure rescaling. Which
source frequencies bracket a given target frequency, and how many target frequencies drop below the lowest source
frequency and must be taken from the straight-line guess, is therefore fixed by :math:`\beta_{k+1}/\beta_k` alone.
Going from :math:`\beta = 5` to :math:`10` is the same interpolation task as going from :math:`20` to :math:`40`,
while :math:`20` to :math:`25` is far milder than :math:`5` to :math:`10` despite the equal difference. To see why,
note that the source run has data only at its own Matsubara frequencies. Its lowest positive one,
:math:`\pi/\beta_k`, has nothing below it but its negative partner, so every target frequency of smaller magnitude
has to come from the straight line joining the two. A target frequency :math:`(2m + 1)\pi/\beta_{k+1}` lies below
:math:`\pi/\beta_k` exactly when :math:`2m + 1 < \beta_{k+1}/\beta_k`. For :math:`m = 0` this always holds when
cooling; the second target frequency only drops into the gap once the ratio exceeds three, the third once it
exceeds five. Below a ratio of three the straight-line guess is thus confined to the single lowest target
frequency, and everything above it is interpolated between actual source values.

The only positive evidence that the chain engaged is the log line

.. code-block:: text

   Using previous calculation and starting the self-consistency loop at iteration N.

with an iteration number that continues from the predecessor. If ``previous_sc_path`` does not exist, or holds no
file matching the requested pattern, the run starts cold from DMFT *without any warning* and finishes with
plausible-looking output. The usual way to end up there is a predecessor that ran with ``do_interpolation`` set to
``False`` while the successor asks for ``use_interpolated_sigma``. The two flags live in different sections and
nothing cross-validates them, so check the log of every rung.

Reaching self-consistency
~~~~~~~~~~~~~~~~~~~~~~~~~

A rung counts as converged once the relative step residual of the mixed self-energy on the core box drops below
``epsilon`` and the chemical potential has settled between two iterations. The cycle then reports

.. code-block:: text

   Self-consistency of sigma and mu reached at iteration N.

and stops. Otherwise it runs until ``max_iter`` is used up, the last iteration reads ``Self-consistency not
reached.``, and the returned self-energy is simply the last iterate. Such a rung is *not* converged. It should
neither serve as the predecessor of the next rung nor be fed to the Eliashberg step, whose eigenvalues on an
unconverged iterate mean nothing. A few practices keep a ladder trustworthy.

* **Start warm, and chain only from converged rungs.** Begin at a temperature where the cycle still converges cold
  from DMFT, converge that rung fully, and let every further rung inherit a converged predecessor. Nothing helps a
  low-temperature rung as much as a converged warm start, and an unconverged one is a liability rather than a head
  start.
* **Step in ratios, not differences.** Since the hand-over depends on :math:`\beta_{k+1}/\beta_k` alone (see
  above), lay the ladder out geometrically rather than in equal steps of :math:`\beta`. Consecutive ratios between
  about 1.25 and 1.5 are a typical choice; beyond a ratio of three the second target frequency, too, ends up on the
  straight-line guess. The warm start has to land inside the basin of attraction of the new fixed point. A rung
  that needs many more iterations than its predecessor, or that reports pole warnings (see below) in its first
  iterations, was probably stepped too far, and an intermediate temperature should be inserted.
* **Budget the iterations.** Near the cold end of the ladder a rung may need tens of iterations, and that count is
  part of the physics rather than a nuisance. Set ``max_iter`` generously; a rung that converges late beats one that
  stops early.
* **Treat the mixing as a convergence lever.** Near an instability the self-energy map has an expanding direction,
  and damped iteration multiplies the error along it by more than one at any damping. A smaller ``mixing`` slows
  the divergence down but cannot remove it. Only the quasi-Newton correction of the accelerated schemes can cancel
  such a mode, and heavy damping starves exactly that correction. Anderson or Pulay mixing with moderate damping
  and a history of a few pairs is therefore the recommended setting for a ladder. Both ``mixing`` and
  ``mixing_history_length`` decide whether and how fast a rung converges, and they are worth tuning on the warm
  rungs, where experiments are cheap.
* **Choose the threshold for the observable.** The step residual says something about the returned iterate, not
  about the fixed-point equation. A direction along which the map contracts very slowly contributes almost nothing
  to the step, yet it can still separate states with different low-frequency self-energies, chemical potentials and
  pairing eigenvalues. Quantities that depend on the lowest Matsubara frequencies, above all the Eliashberg
  eigenvalues near a transition, need a tighter ``epsilon`` than the self-energy itself.
* **Verify, do not trust the flag.** Three checks certify a converged rung. First, the
  ``Minimum static compound eigenvalue of chi_phys`` reported for both channels on the final iteration is
  non-negative up to a small offset from the shell truncation, and the last iterations carry no warning about an
  unphysical (past-pole) branch. Second, the result does not depend on the route. Reaching the same temperature
  through a different predecessor, a different step or a cold start (where one still converges) has to reproduce
  the self-energy and the Eliashberg eigenvalues; a state that remembers how it was reached is not a solution of the
  fixed-point equation. Third, the result survives a refinement of the momentum grid. A grid that is too coarse can
  drag the ladder through a pole the physics does not have, and it shifts the temperature at which the unstabilized
  cycle stops converging.

When a rung does not converge, there are three remedies, in increasing order of intervention. The rung can be
resumed at the same temperature: point ``previous_sc_path`` at its own output folder with ``use_interpolated_sigma``
set to ``False``, and it gets another ``max_iter`` iterations. The temperature ratio can be reduced by inserting an
intermediate rung. Or, once the ladder approaches the instability, one of the stabilization techniques described
next is switched on. The symptoms that call for it are repeated warnings about an unphysical (past-pole) branch and
a step residual that stalls or bounces instead of decaying. Ladder plus scaffold is the standard route into the
low-temperature regime: the ladder keeps the start close to the solution, the scaffold keeps the iteration on the
physical branch on the way there.

Stabilization techniques
------------------------

The self-consistency cycle iterates the map :math:`\Sigma \mapsto \Sigma'` that builds the ladder from the current
self-energy and evaluates the Schwinger-Dyson equation with it. The physical susceptibility
:math:`\chi^{\mathrm{q}}_{\mathrm{r}}` that enters this map solves the Bethe-Salpeter equation, and that solution
has a pole where the static compound block :math:`\chi^{(\mathbf{q},\omega=0)}_{\mathrm{r}}` stops being positive
semi-definite. On cooling, the dominant channel, usually the magnetic one, moves toward this pole, and two things go
wrong at once. Close to the pole a small change of the self-energy changes the susceptibility enormously, so the map
acquires a direction along which it expands instead of contracting, and damped iteration cannot converge along it.
And an iterate that overshoots the pole lands on an unphysical branch of the ladder, where the static
susceptibility has negative eigenvalues and the self-energy built from it cannot be trusted. The code watches for
the second event on every iteration. It logs the minimum static compound eigenvalue of each channel's
susceptibility and warns that the ladder sits on an unphysical branch once that value drops below a small negative
bound.

All three stabilization techniques act on :math:`\chi^{\mathrm{q}}_{\mathrm{r}}` before it enters the self-energy
kernel, and all three add a bosonic mass to its inverse,

.. math::

   \chi^{\mathrm{q}}_{\mathrm{r}} \;\to\; \left[\left(\chi^{\mathrm{q}}_{\mathrm{r}}\right)^{-1} +
   \lambda\right]^{-1} ,

which lowers the susceptibility, moves the pole out of reach and makes the map contractive again. They differ in
how :math:`\lambda` is chosen and in how it is withdrawn at the end, since a mass-shifted susceptibility is not the
physics one is after. All of them are built as *scaffolds*: they shape the map while the iteration approaches the
solution and are taken away before the result counts. Four mechanics are common to all of them.

* **Relaxed threshold.** While a scaffold shapes the map, self-energy convergence is judged at ten times
  ``epsilon``. Converging a scaffolded map to full precision would only spend iterations on an intermediate object.
  The chemical potential criterion stays as it is.
* **Release and history reset.** Once a scaffolded phase converges, the scaffold is removed (or reduced, in the case
  of the annealing) and the cycle carries on with the changed map. Every such switch resets the accelerated mixing
  history, because secant pairs of the previous map would extrapolate across the discontinuity; the next
  ``mixing_history_length`` iterations mix linearly while pairs of the new map accumulate.
* **Only the pure phase counts.** The result is the fixed point of the unmodified map, converged to the full
  ``epsilon``. If ``max_iter`` runs out before that, the run is not converged. Should the scaffold happen to be
  released on the very last iteration, the log also warns that the returned self-energy is a scaffolded-phase
  result and not pure self-consistency.
* **Mutual exclusivity.** The three options modify the same susceptibility and cannot be combined. A sum rule must
  not be calibrated on floored or mass-shifted blocks, and two scaffolds would fight over the same object. If
  several are enabled, the lambda correction wins over ``use_chi_phys_restriction``, which wins over
  ``use_lambda_annealing``, and the losing options are disabled with a warning. The one-shot
  ``perform_lambda_correction`` of the :ref:`lambda correction section <lambda-correction>` is something else
  entirely. It runs a single iteration with the correction applied and left in place, which makes it a one-shot DΓA
  rather than a stabilization of the cycle, and when it is enabled it overrides all three techniques.

Lambda correction as a releasing scaffold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``use_lambda_correction`` the mass is the Moriya lambda correction, calibrated in every iteration so that the
momentum- and frequency-summed corrected susceptibility reproduces the local sum rule of the impurity,

.. math::

   \frac{1}{\beta\, n_{\mathbf{q}}} \sum_{\mathrm{q}} \left[\left(\chi^{\mathrm{q}}_{\mathrm{r}}\right)^{-1} +
   \lambda_{\mathrm{r}}\right]^{-1} = \frac{1}{\beta} \sum_{\omega} \chi^{\omega}_{\mathrm{r},\mathrm{loc}} .

Near its pole the ladder overestimates the susceptibility. The sum rule caps the total weight it may accumulate, so
the calibrated mass comes out just large enough to pull the static susceptibility back from the pole. Which
correction runs depends on the band count. For a single band, :math:`\lambda_{\mathrm{r}}` is one number per
channel, found by a Newton iteration that starts just above the value at which the corrected static susceptibility
would diverge. The ``type`` field of the lambda correction section selects whether both channels are corrected
(``spch``) or only the magnetic one (``sp``, with the density sum rule folded into the magnetic target). For several
bands the mass is a real-symmetric :math:`n_{\mathrm{o}}^2 \times n_{\mathrm{o}}^2` matrix
:math:`\Lambda_{\mathrm{r}}` per channel that matches the sum rule component by component. It comes from a damped
Newton iteration whose line search keeps the static susceptibility gap positive; both channels are always corrected,
and the momentum sum runs over the full Brillouin zone because symmetry-related momenta carry orbitally rotated
susceptibility matrices. The single-band scheme is the derived Moriya correction. The matrix scheme is a heuristic
that mimics it, and inside the cycle both do the same job of stabilizing the iteration.

The schedule has two phases. The correction is applied in every iteration until the cycle converges at the relaxed
threshold. At that point the log announces

.. code-block:: text

   ATTENTION: Self-consistency with the lambda correction reached (at 10x epsilon). Disabling the correction
   and continuing to the pure fixed point with a reset mixing history.

and the cycle goes on to converge the uncorrected map to the full ``epsilon``. The calibrated mass (its Frobenius
norm in the matrix case) is logged every iteration and appended to a text file in the output folder, so the strength
of the intervention can be followed throughout the scaffolded phase. By construction, the scaffold only works if the
pure fixed point is stable once the iteration gets close to it. At temperatures where the pure map has an expanding
direction even right next to its fixed point, the released phase tears away again and the run ends at ``max_iter``
with the verdict that no stable pure fixed point was found from that start. The converged lambda-corrected iterate
of the release iteration is then still on disk as ``sigma_dga_iteration_<i>.npy``. It is a well-defined object in
its own right, but a lambda-corrected DΓA solution rather than a self-consistent one.

Eigenvalue restriction of the susceptibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The option ``use_chi_phys_restriction`` calibrates nothing. For every momentum and bosonic frequency it takes the
compound :math:`n_{\mathrm{o}}^2 \times n_{\mathrm{o}}^2` block of the inverse susceptibility, splits it into
Hermitian and skew-Hermitian parts, diagonalizes the Hermitian part and floors its eigenvalues at a small positive
value (:math:`10^{-4}`), leaves the skew-Hermitian part alone, and inverts the block back. A negative eigenvalue of
the inverse is the signature of a crossed pole. Flooring it pins the corresponding mode of the susceptibility at the
inverse of the floor, while every healthy eigenpair passes through untouched, including negative off-diagonal
*matrix elements*, which are perfectly legitimate in a multi-orbital susceptibility and which an elementwise clamp
would destroy. For a single band the block is a scalar and the restriction reduces to clamping negative values of
the inverse. In the language of the mass shift above, :math:`\lambda` is a positive operator living on the poled
modes alone, just strong enough to lift each of them to the floor.

The two-phase schedule is the same as for the lambda correction. The restriction stays active until the cycle
converges at the relaxed threshold, is then released with a reset mixing history, and the unrestricted phase
converges to the full ``epsilon``. The per-iteration log line

.. code-block:: text

   Restricted physical susceptibility (magn): floored N eigenvalues of the inverse. Releasing the restriction
   is only safe once this count decays to zero.

is the diagnostic to watch. The restriction only bites where an eigenvalue of the inverse falls below the floor, so
on a healthy susceptibility the count is zero and the option changes nothing. During a scaffolded phase the count
should decay to zero as the self-energy relaxes. A release while :math:`N` is still large removes a regularization
the iterate still leans on, and the unrestricted phase will then most likely relapse. The floored blocks are a
regularization, not physics, and the self-energy of the restricted phase is biased wherever they act, so the
restricted iterate is never a result in itself.

Lambda annealing
~~~~~~~~~~~~~~~~

With ``use_lambda_annealing`` the mass is a single scalar :math:`\lambda`, shared by all channels and added to the
compound diagonal of the inverse susceptibility at every bosonic frequency. As an identity shift it is basis
independent, needs no sum rule and is safe for any number of orbitals. One mass for all channels, rather than one
per channel, is a deliberate choice. The channels are coupled through the self-energy, so a large mass on one
channel distorts :math:`\Sigma` and can push another channel's gap negative, and per-channel masses then chase each
other to unphysical values. A single mass sized from the worst channel protects all of them at once and cannot
ratchet.

The size is measured, never chosen by the user. In every iteration the scaffold computes each channel's *static
gap*, the smallest eigenvalue of the Hermitian part of :math:`(\chi^{(\mathbf{q},\omega=0)}_{\mathrm{r}})^{-1}` over
all momenta (reduced across the MPI ranks), and takes the most negative of them as the worst gap :math:`g`. Only
the static slice enters, because that is where the positivity statement lives; the full-frequency spectrum of the
inverse carries a large negative baseline from the shell truncation that would inflate the mass by orders of
magnitude. Whenever the shifted worst gap :math:`g + \lambda` is still negative, the mass moves toward the target
:math:`1.5\,|g|`, but only by half the remaining distance per iteration. It thus changes on the self-energy's own
relaxation timescale, and every measured gap belongs to a settled :math:`\Sigma`. The mass is clamped at a ceiling
past which the pole is too deep for the scaffold; hitting it produces a warning that recommends a warmer start.

The schedule advances once per iteration, after the convergence decision, and exactly one action applies. On the
first measured iteration the scaffold initializes. If every gap is healthy it stays inert at zero mass and the run
proceeds as a plain self-consistency; otherwise it performs a first damped bump. From then on, a shifted gap that is
still negative triggers another bump. This takes precedence over everything else and also re-arms a scaffold whose
mass had already reached zero, should a pole reopen mid-run. When a phase has instead converged at the relaxed
threshold with a healthy shifted gap, the mass is halved, and below a small floor it snaps to exactly zero. Every
change of the mass changes the map, so it resets the mixing history, and the converged verdict of the previous phase
never ends the run. Only a converged phase at exactly zero mass counts as the final result, which the log marks with

.. code-block:: text

   Lambda annealed to zero - continuing with pure self-consistency at full epsilon (only that result counts as
   converged).

Compared with the two-phase scaffolds, the annealing descends to the pure map through a sequence of nearby converged
states rather than a single release. That is gentler and fully automatic, but it usually costs more iterations. The
per-channel log lines report the applied mass together with the measured gap, and every schedule step is logged
with its reason, so the descent can be followed from the log alone.

Choosing a technique
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * -
     - ``use_lambda_correction``
     - ``use_chi_phys_restriction``
     - ``use_lambda_annealing``
   * - mass :math:`\lambda`
     - calibrated by the local sum rule; scalar per channel for one band, matrix per channel otherwise
     - eigenvalue floor on the inverse, acting only on poled modes
     - one measured scalar shared by all channels
   * - physical content
     - derived Moriya correction (single band); heuristic mimic (multi-band)
     - none, a regularization
     - none, a homotopy toward the pure map
   * - phases
     - two: corrected, then pure
     - two: restricted, then pure
     - several: mass halved between converged phases until zero
   * - acts on a healthy susceptibility
     - always, the sum rule shifts it regardless
     - only where an eigenvalue of the inverse falls below the floor
     - only while a static gap is negative
   * - typical use
     - the strongest pull; also the right choice when the corrected state itself is of interest
     - a first, minimal intervention when only a few modes cross the pole
     - a gradual descent when the pure map is expected to be stable but hard to reach

None of the three is tied to a cooldown; each can just as well support a cold run at a temperature where the plain
cycle fails. Whatever the technique, the result is certified the same way as above: a converged pure phase, a
non-negative static susceptibility in both channels on the final iteration, no pole warnings at the end, and
agreement with a second route to the same temperature.

Summary
-------

The cooldown, step by step:

1. Run DMFT at every temperature of the planned ladder, for the same Hamiltonian, interaction and filling, and pass
   each two-particle file through the ``symmetrize`` script.
2. Lay the ladder out geometrically, with consecutive ratios :math:`\beta_{k+1}/\beta_k` of about 1.25 to 1.5, and
   start at a temperature where the cycle still converges cold from DMFT.
3. Configure the first rung with an empty ``previous_sc_path``, ``do_interpolation: True``, ``target_beta`` equal to
   the second rung's inverse temperature and ``target_niv`` at least the second rung's ``niv_core`` (the
   :ref:`self-energy interpolation section <self-energy-interpolation>` lists these keys). Use Anderson or Pulay
   mixing with moderate damping, a generous ``max_iter`` and an ``epsilon`` that suits the observables you are
   after. Keep ``perform_lambda_correction`` at ``False``. The example file
   :ref:`dga_config_beta20.yaml <cooldown-config-beta20>` shows all of this for a later rung; for the first rung
   only its ``previous_sc_path`` would be empty.
4. Run the rung and confirm in the log that it converged (``Self-consistency of sigma and mu reached``), that the
   final ``Minimum static compound eigenvalue of chi_phys`` of both channels is not negative beyond the small
   truncation offset, and that the last iterations show no past-pole warning. Note the run folder
   ``LDGA_Nk<nk_tot>_Nq<nk_tot>_wc<niw_core>_vc<niv_core>_vs<niv_shell>`` it created.
5. Configure the next rung: ``input_path`` set to its own DMFT data, ``previous_sc_path`` set to the predecessor's
   run folder, ``use_interpolated_sigma: True``, and the interpolation section aimed at the rung after it. Leave
   the interpolation off on the last rung. The example file :ref:`dga_config_beta25.yaml <cooldown-config-beta25>`
   marks the three entries that change from one rung to the next.
6. Run it and check the log for ``Using previous calculation and starting the self-consistency loop at iteration
   N`` with an iteration number that continues from the predecessor; without that line the rung started cold.
7. If the rung stops at ``max_iter``, do not chain from it. Either resume it at the same temperature
   (``previous_sc_path`` pointing at its own folder, ``use_interpolated_sigma: False``), insert an intermediate
   temperature, or switch on one flag of the :ref:`stabilization section <stabilization>` for that rung and rerun
   it.
8. Repeat steps 4 to 7 down the ladder. Solve the Eliashberg equation on converged rungs only.
9. Before quoting results, reproduce at least one rung by a second route or on a finer momentum grid and confirm
   that the self-energy and the Eliashberg eigenvalues agree.

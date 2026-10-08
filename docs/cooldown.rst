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
     interaction_type: "from_dmft"
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
     interaction_type: "from_dmft"
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

**The interpolation.** On the :math:`\beta = 20` rung, ``do_interpolation`` is on and ``target_beta`` is ``25.0``, the
inverse temperature of the DMFT data in ``/data/beta25/``. The two have to match exactly: the hand-over file carries
no metadata, so the :math:`\beta = 25` rung takes its content at face value, and a mismatch goes unnoticed.
``target_niv`` is ``60``, the ``niv_core`` of the next rung. A larger value is harmless: the next rung cuts the file
to its core box anyway and fills the tail from its own DMFT self-energy, so a generous value only costs disk space. A
smaller value means the DMFT self-energy also has to fill the rest of the core box. The interpolated self-energy is
written once, from the rung's final self-energy, the converged one if the rung converged. The ratio :math:`25/20 =
1.25` keeps the re-gridding mild: only the lowest target frequency has to be extrapolated below the source grid. On
the :math:`\beta = 25` rung the interpolation is switched off because the ladder ends there; to go on to :math:`\beta
= 30` one would leave it on with ``target_beta: 30.0``.

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

* The cycle looks for ``sigma_dga_interpolated_beta<b>_niv<n>.npy`` in ``previous_sc_path`` and, if several exist,
  takes the one whose ``<b>`` is closest to the rung's inverse temperature. Without ``use_interpolated_sigma`` it
  takes the raw ``sigma_dga_iteration_<i>.npy`` with the highest ``<i>`` instead. Either way the iteration count
  continues from the highest raw iterate. Beyond its name the file carries no metadata, so its content is taken at
  face value as a self-energy at the rung's own temperature; a predecessor aimed at the wrong ``target_beta`` is
  only caught when a better-matching file sits next to it.
* The raw iterates ``sigma_dga_iteration_<i>.npy`` live in the ``Sigma_Iterates`` subfolder of the run folder (a
  predecessor written before that subfolder existed keeps them in the run folder itself, which is searched as the
  fallback), together with the core window of the un-mixed proposals ``sigma_dga_proposal_iteration_<i>.npy``, which
  a run writes only with ``use_jacobian_stabilization`` on and which are never read back.
* The self-energy is cut to the rung's ``niv_core``. If the rung uses a different momentum grid, it is re-sampled
  onto it, exactly by striding when the new grid is a sub-lattice of the old one and by band-limited Fourier
  interpolation otherwise, so the grid may be refined along the ladder. Beyond the core box, the rung's own DMFT
  self-energy supplies the tail, as in any cold run.
* The chemical potential starts at the last entry of the predecessor's ``mu_history.npy`` and is re-solved before
  the first iteration, so that the starting self-energy has the filling of the rung's DMFT lattice Green's function.
  Every later iteration holds that same filling. The predecessor's own filling at its chemical potential is never
  the target: it drifts along a chain of rungs.
* The mixing history starts empty, so the first steps of an accelerated scheme are damped steps. With
  ``use_jacobian_stabilization`` on both runs, the predecessor's certified Jacobian spectrum is handed over as well:
  its run folder holds ``jacobian.npz``, and the rung installs the damping bound of that certified spectrum
  before its first step, together with a reflection on the directions that were already unstable or that the two
  preceding rungs extrapolate across the boundary, and the own damping of every carried direction, the ones within a
  small band of the boundary included, each of them
  re-gridded to the rung's frequencies like the self-energy (without the pole fit, and zeroed beyond the
  predecessor's highest frequency instead of a DMFT tail), on the same momentum grid and symmetry reduction only; on
  another grid the eigenvalues and their damping bound are carried without vectors (see the
  :ref:`Jacobian tracking <cooldown-jacobian>` section). For a direction that re-gridding is an approximation and
  not an exact map: the spline derivatives depend nonlinearly on the data, so the real and the imaginary part of a
  carried vector are re-gridded on their own and the weighting inside the column pair of a complex mode shifts a
  little, and the branch extrapolation below the innermost frequency was designed for a self-energy. Without the
  flag on the predecessor, nothing but the self-energy and the chemical potential is handed over.
* The iteration count carries on. A predecessor that ended at iteration :math:`N` hands over to iteration :math:`N +
  1`, the rung performs up to ``max_iter`` further iterations on top, and its per-iteration files keep that numbering.
* The accelerated mixing schemes start with an empty history. For the first ``mixing_history_length`` iterations the
  rung mixes linearly while (iterate, proposal) pairs of the *new* map accumulate, with the configured ``mixing`` or,
  with ``use_jacobian_stabilization`` on, at the damping bound ``p_eff`` of the tracker, never above ``mixing``, which a
  carried spectrum can lower before the first step; only then do Pulay or Anderson take over. Secant pairs from the
  previous temperature are deliberately left behind. They describe a different map, and an accelerated scheme fed with
  them explains the new residual through the old map and parks the iteration at the predecessor's solution while the
  step residual reports convergence.

The re-gridding treats real and imaginary parts separately on the full signed frequency axis. The innermost
frequencies, where the grid is sparsest, are interpolated linearly and everything above them with shape-preserving
PCHIP splines; between source points a shape-preserving interpolant never overshoots its data, so a causal self-energy
stays causal there. Target frequencies smaller in magnitude than the lowest source frequency are extrapolated with
PCHIP from the same-sign branch alone rather than across :math:`\nu = 0`, because the odd imaginary part does not
vanish at the origin, :math:`\mathrm{Im}\,\Sigma(i0^+) = -\Gamma(0)`; nothing is clipped, so a sign change below the
innermost frequency is followed. At momenta whose diagonal self-energy is non-causal at the innermost frequency,
:math:`\mathrm{Im}\,\Sigma_{11}(i\nu_0) > 0`, that branch extrapolation is replaced by a minimal pole representation
(MiniPole) of the momentum's own self-energy with its static part removed and restored, the fits distributed over the
ranks; a fit is kept only if it places no pole in the upper half plane and stays within one source step of the branch
values, otherwise the branch values remain. The log reports how many of the flagged irreducible k-points accepted the
fit.

How demanding the re-gridding is depends on the *ratio* of the two inverse temperatures, not on their difference.
The Matsubara grids :math:`\nu_n = (2n + 1)\pi/\beta` of two temperatures are related by a pure rescaling. Which
source frequencies bracket a given target frequency, and how many target frequencies drop below the lowest source
frequency and must be extrapolated from the branch, is therefore fixed by :math:`\beta_{k+1}/\beta_k` alone.
Going from :math:`\beta = 5` to :math:`10` is the same interpolation task as going from :math:`20` to :math:`40`,
while :math:`20` to :math:`25` is far milder than :math:`5` to :math:`10` despite the equal difference. To see why,
note that the source run has data only at its own Matsubara frequencies. Its lowest positive one,
:math:`\pi/\beta_k`, has nothing below it but its negative partner, so every target frequency of smaller magnitude
has to be extrapolated from the same-sign branch. A target frequency :math:`(2m + 1)\pi/\beta_{k+1}` lies below
:math:`\pi/\beta_k` exactly when :math:`2m + 1 < \beta_{k+1}/\beta_k`. For :math:`m = 0` this always holds when
cooling; the second target frequency only drops into the gap once the ratio exceeds three, the third once it
exceeds five. Below a ratio of three the branch extrapolation is thus confined to the single lowest target
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

and stops. Otherwise it runs until ``max_iter`` is used up, the last iteration reads ``Self-consistency not reached.``,
and the returned self-energy is simply the last iterate. Such a rung is *not* converged. It should neither serve as the
predecessor of the next rung nor be fed to the Eliashberg step, whose eigenvalues on an unconverged iterate mean
nothing. The step measures how far the mixing moved, not how far the map is from its fixed point: every iteration's
convergence line also carries the ``map residual`` :math:`\lVert S(\Sigma) - \Sigma\rVert / \lVert\Sigma\rVert`, which
keeps falling with the step on a healthy rung and ends at a multiple of ``epsilon`` set by the damping. An accelerated
mixing can stagnate instead, typically next to a direction the map has turned unstable: its steps shrink while the map
residual stays put or grows, the step test fires, and the cycle warns ``Converged on the step residual ..., but the map
residual ... is still ... and no lower than in the previous iteration: the steps have stalled`` when the map residual
lies above ten times ``epsilon`` and has not fallen since the previous iteration. Such a rung has not reached a fixed
point either. A flip of the unstable direction (see :ref:`cooldown-jacobian`) lets the mixing converge there properly.
With ``use_jacobian_stabilization`` the damped steps run at the tracker's damping bound ``p_eff`` instead of
``mixing`` and are shorter by that ratio, so the step residual of a damped step is read at the configured mixing,
multiplied by ``mixing / p_eff`` before it is compared with ``epsilon``, and the verdict means the same distance from
the fixed point with and without the flag; the convergence line then adds ``, <x> at the configured mixing``. An
accelerated step is compared as it is. The tracker can also hold the verdict back for a minimum number of iterations
(see :ref:`cooldown-jacobian`). A few practices keep a ladder trustworthy.

* **Start warm, and chain only from converged rungs.** Begin at a temperature where the cycle still converges cold
  from DMFT, converge that rung fully, and let every further rung inherit a converged predecessor. Nothing helps a
  low-temperature rung as much as a converged warm start, and an unconverged one is a liability rather than a head
  start.
* **Step in ratios, not differences.** Since the hand-over depends on :math:`\beta_{k+1}/\beta_k` alone (see
  above), lay the ladder out geometrically rather than in equal steps of :math:`\beta`. Consecutive ratios between
  about 1.25 and 1.5 are a typical choice; beyond a ratio of three the second target frequency, too, ends up on the
  branch extrapolation. The warm start has to land inside the basin of attraction of the new fixed point. A rung
  that needs many more iterations than its predecessor, or that reports pole warnings (see below) in its first
  iterations, was probably stepped too far, and an intermediate temperature should be inserted.
* **Budget the iterations.** Near the cold end of the ladder a rung may need tens of iterations, and that count is
  part of the physics rather than a nuisance. Set ``max_iter`` generously; a rung that converges late beats one that
  stops early.
* **Treat the mixing as a convergence lever.** Near an instability the self-energy map has an expanding direction,
  and damped iteration multiplies the error along it by more than one at any damping. A smaller ``mixing`` slows
  the divergence down but cannot remove it. Without the flip of the Jacobian tracking below, only the quasi-Newton
  correction of the accelerated schemes can cancel such a mode, and heavy damping starves exactly that correction.
  Anderson or Pulay mixing with moderate damping and a history of a few pairs is therefore the recommended setting
  for a ladder. Both ``mixing`` and
  ``mixing_history_length`` decide whether and how fast a rung converges, and they are worth tuning on the warm
  rungs, where experiments are cheap. Whether a rung struggles with an overshooting mode, which a smaller
  ``mixing`` cures, or with an expanding one, which it cannot, is what the Jacobian tracking described below reads
  off the cycle's own iterates.
* **Choose the threshold for the observable.** The step residual says something about the returned iterate, not
  about the fixed-point equation. A direction along which the map contracts very slowly contributes almost nothing
  to the step, yet it can still separate states with different low-frequency self-energies, chemical potentials and
  pairing eigenvalues. Quantities that depend on the lowest Matsubara frequencies, above all the Eliashberg
  eigenvalues near a transition, need a tighter ``epsilon`` than the self-energy itself.
* **Verify, do not trust the flag.** Three checks certify a converged rung. First, the
  ``Lowest eigenvalue of the static susceptibility`` reported for both channels on the final iteration is non-negative
  up to a small offset from the shell truncation, unless the material genuinely orders in that channel (the charge
  susceptibility of a charge-ordered material can be negative). Second, the result does not depend on the route.
  Reaching the same temperature through a different predecessor, a different step or a cold start (where one still
  converges) has to reproduce the self-energy and the Eliashberg eigenvalues; a state that remembers how it was reached
  is not a solution of the fixed-point equation. Third, the result survives a refinement of the momentum grid. A grid
  that is too coarse can drag the ladder through a pole the physics does not have, and it shifts the temperature at
  which the unstabilized cycle stops converging.

When a rung does not converge, there are three remedies, in increasing order of intervention. The rung can be resumed
at the same temperature: point ``previous_sc_path`` at its own output folder with ``use_interpolated_sigma`` set to
``False``, and it gets another ``max_iter`` iterations. The temperature ratio can be reduced by inserting an
intermediate rung. The rung can be restarted with a smaller ``mixing`` for its first steps, which removes a bad first
step. Or, once the
ladder approaches the instability, one of the stabilization techniques described next is switched on. Before reaching
for a scaffold it pays to rerun the rung with ``use_jacobian_stabilization`` switched on, which costs no extra
evaluation of the map and writes the leading eigenvalues of the map to the log every iteration; the :ref:`Jacobian
tracking <cooldown-jacobian>` section below explains how to read them. The symptoms that call for it are a lowest
static susceptibility eigenvalue that turns negative in a channel the material does not order in, and a step residual
that stalls or bounces instead of decaying. Ladder
plus scaffold is the standard route into the low-temperature regime: the ladder keeps the start close to the solution,
the scaffold keeps the iteration on the physical branch on the way there.

Stabilization techniques
------------------------

The self-consistency cycle iterates the map :math:`\Sigma \mapsto \Sigma'` that builds the ladder from the current
self-energy and evaluates the Schwinger-Dyson equation with it. The physical susceptibility
:math:`\chi^{\mathrm{q}}_{\mathrm{r}}` that enters this map solves the Bethe-Salpeter equation, and that solution has a
pole where the static compound block :math:`\chi^{(\mathbf{q},\omega=0)}_{\mathrm{r}}` stops being positive
semi-definite. On cooling, the dominant channel, usually the magnetic one, moves toward this pole, and two things go
wrong at once. Close to the pole a small change of the self-energy changes the susceptibility enormously, so the map
acquires a direction along which it expands instead of contracting, and damped iteration cannot converge along it. And
an iterate that overshoots the pole lands on an unphysical branch of the ladder, where the static susceptibility has
negative eigenvalues and the self-energy built from it cannot be trusted. The code logs what to watch on every
iteration. It reports the lowest eigenvalue of each channel's static susceptibility as plain information, without a
warning, since a negative value is not by itself a sign of the unphysical branch: the charge susceptibility of a
genuinely charge-ordered material can be negative. On the same line it logs the lowest eigenvalue of the static inverse
compound susceptibility, the distance of the static Bethe-Salpeter equation from its pole: it shrinks toward zero as a
channel approaches its instability, and a negative value means the pole has been crossed. A pole can also sit at a
finite frequency while the static slice stays healthy, so the code logs the largest compound norm at the first bosonic
frequency next to the static one as well, and warns once it exceeds twice the static value: a bosonic susceptibility
decreases with :math:`|\omega|`, and a first-frequency value above the static one is a pole of the ladder at
:math:`\omega_1` that the static monitor cannot see.

For a single band the loop also predicts that pole before each proposal, from the iterate alone: at :math:`\omega_1`
the density Bethe-Salpeter matrix is dominated by the bubble element of the pair :math:`(\pi T, -\pi T)`, and a pole
ring enters the Brillouin zone once the ratio
:math:`R = \gamma_{\mathrm{loc}}\,\beta\,\mathrm{Re}\langle G^{(\mathbf{k},\pi T)} G^{(\mathbf{k},-\pi T)}
\rangle_{\mathbf{k}}` reaches one, with :math:`\gamma_{\mathrm{loc}}` the local density vertex at that element. Every
iteration logs ``First-frequency density pole ratio R = <R> (a pole ring of the density ladder at w_1 lies inside the
Brillouin zone once R >= 1).``, and from :math:`R \geq 1` on a warning says that the iterate has lost too much local
scattering at :math:`\pi T` for the DMFT vertex, a threshold that mixing and damping do not move.

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
  iteration mixes linearly, after which the accelerated scheme resumes on the pairs of the new map and its history
  grows back to ``mixing_history_length``. A change of the reflected directions of the Jacobian tracking below resets
  the history in the same way.
* **Only the pure phase counts.** The result is the fixed point of the unmodified map, converged to the full
  ``epsilon``. If ``max_iter`` runs out before that, the run is not converged. Should
  the scaffold happen to be released on the very last iteration, the log also warns that the returned self-energy is a
  scaffolded-phase result and not pure self-consistency.
* **Mutual exclusivity.** The three options modify the same susceptibility and cannot be combined. A sum rule must
  not be calibrated on floored or mass-shifted blocks, and two scaffolds would fight over the same object. If
  several are enabled, the lambda correction wins over ``use_chi_phys_restriction``, which wins over
  ``use_lambda_annealing``, and the losing options are disabled with a warning. The one-shot
  ``perform_lambda_correction`` of the :ref:`lambda correction section <lambda-correction>` is something else
  entirely. It runs a single iteration with the correction applied and left in place, which makes it a one-shot DΓA
  rather than a stabilization of the cycle, and when it is enabled it overrides all three techniques.

A fourth option, ``use_jacobian_stabilization``, is not a scaffold and leaves the susceptibility alone; it acts on
the iteration instead of on the map and composes with any of the three. It has its own section next.

.. _cooldown-jacobian:

Jacobian tracking
~~~~~~~~~~~~~~~~~

Whether damped iteration can converge to the fixed point is decided by the Jacobian of the map :math:`\Sigma \mapsto
\Sigma'` there. Write :math:`\lambda_\Pi` for the eigenvalues of :math:`1 - \partial\Sigma'/ \partial\Sigma`. Linear
mixing with parameter :math:`p` multiplies the error along such a direction by :math:`1 - p\lambda_\Pi` per iteration,
so a direction with a positive real part of :math:`\lambda_\Pi` is tamed by a small enough :math:`p`, whereas one with a
negative real part expands at every :math:`p` and is exactly the expanding direction of the previous section. With
``use_jacobian_stabilization`` the cycle estimates the leading :math:`\lambda_\Pi` from the (iterate, proposal) pairs
the mixing records anyway, a secant Rayleigh-Ritz estimate on the last seven pairs that costs no additional evaluation
of the map, and refreshes it every iteration. Each estimate carries a residual; only estimates whose residual passes a
gate count as *certified*, and every decision is based on certified ones alone. Past the gate an eigenvalue is read with
its error bound, the residual times the condition number of the eigenvalue in the projected map, which equals the
residual for a normal map and widens every band below on a non-normal one; that bound is the ``res`` of the bands, the
predictions, the log line and ``jacobian.npz``. Two things follow from the estimate. A
direction certified with a negative real part is *flipped* on the iteration that certifies it: the tracker updates on
that iteration's pair before the mixing acts, so that iteration's proposal residual is already reflected on it, which
turns the sign of the damping on that direction and makes the physical fixed point attractive there, the modified
iteration of arXiv:2606.04936, whose stabilization acts from the first step. The flip also lands ahead of the crossing:
the certified directions of consecutive estimates are identified by the overlap of their eigenvectors (a complex pair
by its real plane), and a direction whose real part fell monotonically over three consecutive matched estimates is
flipped on the estimate whose linear extrapolation one iteration ahead lies below the boundary by more than the
undecidable band plus an error bar (the larger error bound of the last two estimates plus twice the scatter of the
three points about a line), logged as ``Jacobian tracker: predicted crossing of lambda_Pi=<v> (extrapolated <p>, band
<m>, error bar <d>), flipped ahead of its crossing.``; only the real part is extrapolated, since a complex pair whose
modulus factor passes one while its real part is still positive is the case the damping bound cures. A direction whose
eigenvalue jumps through a pole, from a certified positive to a certified negative real part at a modulus beyond 5 with
the step of its inverse continuing the previous one, is flipped on the iteration of the jump (``Jacobian tracker: pole
crossing of lambda_Pi from <a> to <b>, flipped at once.``), and a direction approaching a pole with a growing modulus
beyond 5 is left alone until it has jumped (``Jacobian tracker: pole-type approach of lambda_Pi=<v> (modulus <r>
growing), no prediction.``), since flipping it before the pole would make it diverge. A predicted flip goes through the
same reflection, refusals and releases as a certified one and is undone by the calm release once the real part has
turned back up and the mode is certified stable inside the reflection. And the largest damping the certified spectrum
allows is applied to every damped step, i.e. to linear mixing and to the linear warm-up and fallbacks of the accelerated
schemes, never above the configured ``mixing``; an Anderson or Pulay step keeps the configured value, because a
whole-spectrum bound describes damped iteration and not a quasi-Newton step. The safety factor ``c`` in ``(0, 1)`` of
that bound (arXiv:2606.04936 Eq. 13, smaller values widening the basin) is taken at the vertex of the stability
parabola, arXiv:2609.11405 App. F. The bound departs from Eq. 13 in two places. It never drops below a floor of
:math:`0.01`, and once the measured spectrum asks for less the log warns that every damped step runs at the floor and
that a certified direction the floor cannot contract, :math:`2|\mathrm{Re}\,\lambda_\Pi|/|\lambda_\Pi|^2 \leq 0.01`,
contracts only where the per-direction damping below carries it. And a certified mode whose real part lies inside its
undecidable band :math:`\max(10^{-2}, \mathrm{res})` on the unstable side enters with the band in place of its real
part, the loosest bound compatible with either sign, so that a near-neutral mode does not freeze the iteration, while
one measured on the stable side binds with its measured real part, as Eq. 13 asks. The damping is never raised again
within a run, so a single estimate that measures such a mode just above zero, inside a band that does not resolve its
sign, holds every later damped step at the small value or the floor it asks for. That scalar applies only to the
directions the estimate does not certify: on a damped step every certified direction is taken out of the mixed
result and given the step its own eigenvalue allows instead, capped at one rather than at ``mixing`` and with the
reversed sign where it is flipped (Eq. 27 of arXiv:2606.04936, which carries the same cap, and Eq. 21 of
arXiv:2609.11405, which does not; the latter's App. G1 splits the stabilized step from the damped one by frequency, the
code by the certified span), so a stiff certified mode is damped hard while a flat one advances its whole residual in a
single step, whatever the damping did to it. An Anderson or Pulay step is
left alone: its secant model of the certified span comes from the same secants that certified the span, so its step
there is the better informed one, and overwriting it with a Picard damping pins the flat certified directions to the
damped rate. The one exception is the direction of the step on the reflection. A damped iteration on the reflected map
converges only to fixed points that are stable for it, which the flips make of the physical one, while a quasi-Newton
step can be drawn to any fixed point of the map, the unphysical one included. So while a reflection is installed, an
Anderson or Pulay step whose component on the reflected subspace points against that of the flipped damped step is
replaced by the stabilized damped step at ``p_eff``, with the warning ``Anderson step opposes the flipped damped step
on the reflected subspace - taking the stabilized damped step instead.`` (``Pulay step ...`` likewise). The
per-direction damping gave no speed-up for the parquet iteration of arXiv:2606.04936 and looked less
stable there than uniform damping, while arXiv:2609.11405 finds that it helps TRILEX and SBE, whose iterated object
carries the self-energy and the screened interaction (in SBE the couplings beside them), and that uniform mixing is more
stable for MBE, which that paper attributes to MBE being affected by vertex divergences the way the parquet iteration
is. Ladder DGA iterates the self-energy alone, but the beta ladder takes it close to those divergences, so its benefit
here is a hypothesis for the ladder to confirm and not a settled result. That per-direction damping is installed on the
first iteration that certifies a direction, with or without a flip, so a run that never flips is still accelerated on
its certified flat directions, and once installed it stays in force until the next iteration that certifies something
rebuilds it (an iteration that certifies nothing leaves it in place), until the residual grows on three consecutive
iterations, which releases it and holds it off for three further informative iterations (an iteration frozen below the
noise floor does not count) before the estimate may install it again, until the calm release of the reflection beside it
drops it until the next certifying iteration, or until a scaffold pauses the flips. The projector that takes the
certified span out of the damped step is the oblique one along the uncertified subspace, so an uncertified direction
keeps exactly the mixed step; where the two subspaces are too close to separate that way, the orthogonal projector is
used instead and the log records it. The per-direction damping is rebuilt from the current estimate on every iteration
that certifies something, so a direction certified stable again takes its own positive damping right away, and the
persistence rule above (the flip lands on the certifying iteration, held off only after a growth release) gates the
reversed sign in it exactly as it gates the reflection; until that rule admits a flip and its reflection is built, the
direction is no part of the map and keeps the step the mixing gave it. The
same holds for a certified direction whose real part is negative but inside its undecidable band: its sign is not
resolved, so the map gives it the damping of the damped step itself, the step the mixing gives it, instead of a
positive damping of up to one that would push it outward; the map of a carried set or of an exact check below treats
such a mode the same way.

The log carries one line per iteration once three pairs exist,

.. code-block:: text

   Jacobian tracker: lambda_Pi=+4.4529+0.0000j (res 4.9e-02, certified, stable), lambda_Pi=+2.6498-2.0341j
   (res 5.4e-01, uncertified), ...; 0 flipped, p_eff=0.2246, rho=0.0000.

listing every flipped mode and the least-stable certified mode, the one with the smallest real part that approaches
the boundary first, and filling the remaining places up to four by descending modulus, each with its error bound and
verdict (``certified`` with ``stable``, ``flip``, ``predicted flip``, ``marginal`` or, for a clearly unstable mode while
a scaffold pauses flips, ``unstable, suppressed``, or ``uncertified``), then the number of modes this estimate flips,
the damping ``p_eff`` the whole spectrum allows and the convergence rate ``rho`` the certified directions predict under
the damping each of them actually takes. The flip count is the decision of that update's estimate, not the size of
the reflection in force: a carried or held reflection whose direction the estimate no longer sees logs ``0 flipped``
while it acts, and a flip whose reflection was refused still counts. The summary line at convergence,
``Jacobian tracker at convergence: <k> flipped directions, p_eff=<p>, rho=<r>.``, counts the reflected directions
installed instead.

The tracker also keeps a converging run going for a minimum number of iterations, the ``ceil(1 / (eps p))`` of
arXiv:2609.11405 App. G4, taken per certified mode with ``eps`` its distance from the stability boundary (a mode inside
its undecidable band counts with the band :math:`\max(10^{-2}, \mathrm{res})`, the smallest distance it is known to
have) and ``p`` the damping that mode actually gets, i.e. the damping of the installed per-direction map where it acts
on the mode (a flat mode at a damping of one needs ``ceil(1 / eps)`` iterations) and the undamped mixing the spectrum
allows elsewhere; the count divides every damping, the map's own included, by the vertex constant to recover ``p`` and
clips it at one, the largest count over the modes is the minimum, and it is capped at 15000. The pure fixed point is
not declared before the minimum over the certified modes outside their undecidable band has passed since the run's
start, so that a flat direction can carry the iterate far enough from where it began; a run that meets the convergence
test earlier logs ``Self-consistency of sigma and mu reached at iteration <N>, <M> iterations after the start, below
the minimum <K> the certified modes outside their undecidable band ask for (p_eff = <p>): continuing, so that a flat
direction can settle.`` and keeps iterating, up to ``max_iter``. A mode inside its band does not hold the convergence
back, since its band asks for a hundred iterations or more at any damping (a band of 0.01 alone asks for 100 at a
damping of one) whatever its real part is. When the count over every certified mode, the in-band ones included,
exceeds the iterations the run took, through such a mode or because ``max_iter`` cut the minimum, the run converges
with the warning ``Jacobian tracker: converged after <M> iterations, below the minimum <K> ...`` that a flat direction
may not have settled.

Reading the log line is the point of the flag as much as the flip is. Certified
eigenvalues with large positive real parts (values of 4 to 10 are common near the instability) mean the map overshoots:
damped iteration bounces along those directions, a smaller ``mixing`` or the automatic bound cures it, and Anderson
mixing handles it once its secant history describes the map, which it does not in the first steps of a rung. A certified
negative real part means the pure fixed point repels along that direction; the reflection is applied, and a scaffold is
the alternative if the flip does not carry the rung. Estimates that stay ``uncertified`` for many iterations, with
residuals of order one and eigenvalues jumping between iterations, mean that the iterate is not in a neighborhood where
the map is linear, typically because it crosses a pole every few iterations; the tracker then correctly does nothing,
and the remedy is the ladder step or a scaffold, not the flip. The tracker also freezes once the step between iterates
drops to the rounding noise of the stored self-energy, :math:`10^{3}\,\epsilon_{\mathrm{mach}}` of the storage
precision, which is :math:`1.2 \cdot 10^{-4}` relative for single-precision iterates and
:math:`2.2 \cdot 10^{-13}` for double-precision ones. That sits
just above the default ``epsilon`` of :math:`10^{-4}`, so the tracker is frozen over the final approach of a
converging rung and a tighter ``epsilon`` only lengthens that stretch. The last estimate stays in force, but it is the
estimate from before the freeze, not the Jacobian at the converged point: a flat direction that dominates only the
final approach is never certified from single-precision iterates, and resolving it takes double-precision iterates
with a tighter ``epsilon``.

The tracker keeps its vectors on the symmetry sector of the self-energy, the irreducible momenta and the positive
frequencies, each weighted by the square root of the number of momenta and frequencies it stands for, so every norm it
forms is the norm of the whole core window while its memory scales with the irreducible wedge. The exact Jacobian below
works in the same coordinates, and so are the eigenvectors written to ``jacobian.npz``. With Pulay or Anderson mixing,
the accelerated-mixing history of a tracker run is kept on the same sector: the least-squares correction is solved
there and unfolded once onto the core window, while the part of the window the sector does not hold (inputs that break
the lattice symmetry) takes the linear step. The tracker records the core window alone. With a non-zero
:math:`V^{\mathbf{q}}` the shell beyond it carries a Hartree-Fock offset that the loop recomputes from every iterate's
occupations, a feedback the measured spectrum leaves out (and the exact Jacobian below,
which holds the offset fixed, as well), so the loop warns once, before its first iteration: ``Jacobian stabilization
with a non-zero V^q: the loop recomputes the V^q Hartree-Fock offset of the shell from every iterate's occupations,
while the tracker records the core window alone, so the measured spectrum and the one in jacobian.npz leave out its
feedback.`` (with ``and the exact Jacobian holds the offset fixed`` after ``alone`` when ``use_exact_jacobian`` is on).

``use_exact_jacobian`` removes that limit for the spectrum a rung hands on: at the pure fixed point the Jacobian is
evaluated exactly (the converged-point Jacobian of arXiv:2609.11405 App. G4), and its eigenpairs with the smallest real
parts of :math:`\lambda_\Pi` and with the largest moduli replace the certified spectrum in ``jacobian.npz`` (marked
``exact``), restricted to the self-energies the loop keeps: the lattice symmetry, including what it does to the orbitals
at a momentum it leaves fixed, the Matsubara Hermiticity, and the orbital pairs the converged self-energy populates. The
log says ``Exact Jacobian: <n> eigenpairs from <m> products, lambda_Pi ...``; a search that does not converge within
about 150 products contributes its converged pairs, with a warning. Both searches start from one vector that holds the
least-stable and the stiffest certified direction of the tracker beside a fixed random one, so the second search
repeats the opening Arnoldi factorization of the first and is served those 41 products from memory instead of
recomputing them. When every pair the search for the smallest real part of :math:`\lambda_\Pi` returns is unstable,
the log warns ``Exact Jacobian: all <n> eigenpairs of the largest real part are unstable (...); the Jacobian may have
more unstable directions than these.``, since the search then holds no stable mode to show that it reached past the
unstable ones.
An exact mode past the flip margin that the reflection in force does not hold is unstable for the damped iteration at
the converged point, which the run then reached by another route; it is warned about (``Jacobian tracker: the exact
spectrum at the converged point has unstable modes the installed reflector does not hold, lambda_Pi <v>: ...``) and the
next rung carries it as a flip. A reflected flip no exact mode matches stays in the file beside the exact spectrum with
the value it was flipped at, so a search that misses it does not drop it. The derivative holds the fitted
high-frequency moment :math:`\Sigma_\infty` of the loop box and of the DMFT box fixed, which is exact while their fit
windows, the top fifth of each box and at least four frequencies, lie beyond the core window; the operator warns
(``Exact Jacobian: the moment-fit window of the loop box ... reaches into the core window ...``) when one does not.
The next rung's carried flips, cross-rung extrapolation and damping bound then rest on eigenvalues without a secant
residual and without the freeze above.

The same flag acts inside the loop where the estimate alone cannot certify a mode. A window of Anderson steps mixes
many modes, so the least-stable Ritz vector can keep its direction for many iterations while its value wanders and its
residual stays above the gate. When the least-stable mode of two consecutive estimates lies below the storage band
(:math:`\mathrm{Re}\,\lambda_\Pi < 0.1`), matches by its vector and is still uncertified, the loop runs a short exact
Arnoldi from that vector before the iteration's step, every rank taking part; a run carried from an exact spectrum does
the same on the carried modes (every mode below :math:`\mathrm{Re}\,\lambda_\Pi = 1` keeps its vector there) before its
first step. Certified pairs act at once through the path of a carried set: a negative real part is flipped, a pair of
undecidable negative sign takes the damping of the damped step, calm updates do not release the set while the estimate
certifies nothing stable inside it, a flip the reflection already holds stays in the set, and a flip the estimate
certifies later extends the set instead of replacing it, taking the place of the member it matches where the check
found that member stable.
The log says ``Jacobian
tracker: exact check requested for ...``, then ``Exact check: <m> products, lambda_Pi ...`` and ``Jacobian tracker:
exact-check set installed on ...``. A check costs one build of the operator plus at most ten products, and a run
makes at most five of them besides the one at its start; a check that certifies its lead pair and finds every
certified pair on the stable side of its undecidable band gives its slot back (``Jacobian tracker: the exact check
found every certified pair stable, its slot re-armed.``), at most five times per run, so the budget is kept for a mode
that turns out unstable, also when the checks follow a stable mode drifting towards the boundary, and a run makes at
most ten mid-rung checks; one that leaves its lead uncertified, certifies no pair or certifies a pair of undecidable
or negative sign keeps its slot. Its products run in blocks, the start directions and then each extension of the
search, and the Bethe-Salpeter systems of a block share one factorization per channel, so a
block costs little more than one product where that solve dominates the product; the memory detection sizes the widest
block that fits every node, at most four products, and logs it with the chunk budgets. The operator holds the three
local vertices of the Bethe-Salpeter and double-counting kernels between its products while they fit; on a node too
small for that it loads each vertex for the phase of the product that reads it, which costs one read of the three
vertex files per piece of a block, and so per product for the solve at the end of a rung, which takes one column at a
time; the log then says ``Exact-Jacobian products ... local vertices loaded per phase``. Each rank also holds both
channels' three-leg vertices on its momenta; when the operator does not fit with them (tried with the local vertices
held first, then loaded per phase), every momentum group of a piece's Bethe-Salpeter pass recomputes its own three-leg
vertices from the auxiliary susceptibility, one more factorization pass per channel and piece with the same products,
and the log adds ``three-leg vertices recomputed per momentum group``. With them recomputed, the local vertices are
again held if they fit and loaded per phase otherwise; the exact Jacobian is disabled only when one column does not fit
with both.

Every eigenvalue an iteration's estimate has (up to six, largest modulus first) and its error bound are also written to
``jacobian.npz`` in the run folder, as the arrays ``eigenvalues`` and ``eigenvalue_residuals``, one row per iteration,
``nan`` where no estimate exists, and the damping the tracker ran at goes into the same file as ``damping``, one
``p_eff`` per iteration, for plotting the approach to the instability. Row 0 is the run's first iteration
(``starting_iter + 1`` on a resumed rung), and iterations measured while a susceptibility-reshaping or annealing
scaffold shaped the map are included, unlike the certified spectrum that joins the file when the loop ends, which holds
the last certification made with flips allowed (by the estimate or by an exact check), the set the run carried in, the
exact spectrum at the converged point, or no mode at all. A flipped direction contracts under its reversed damping and
leaves the secant window, so that last certification often no longer sees a flip the reflection in force still holds;
such flips are written beside it, with the values they were certified at. The file is stored uncompressed, since its
vector columns barely compress, and every rewrite goes to a temporary file that is renamed onto ``jacobian.npz``, so a
run killed while writing leaves the previous version intact.

A run that starts from a predecessor with the flag on begins with that predecessor's spectrum (the log says
``Jacobian tracker: carried <k> certified modes (lambda_Pi ...), <m> with a usable vector, p_eff=<p>.`` and, when the
set is installed, ``Jacobian tracker: carried set installed on <n> directions, <m> of them reflected.``). Every
certified mode whose real part lies below the storage band of :math:`+0.1` carries a vector, a band distinct from the
flip band :math:`-\max(10^{-2}, \mathrm{res})`, so the set holds the flipped modes together with the stable ones
close to the boundary; it acts through the reflection of the flipped columns and through the per-direction damping
built on all of them, a flipped direction taking its own reversed damping and a stable one its own positive damping
instead of the scalar bound of the step, while one whose negative real part lies inside its undecidable band takes the
scalar bound ``p_eff``, as in the estimate's own map. Where two carried columns of opposite sign lie too close for any
bounded weighting to reverse one and keep the other, the per-direction damping of the set is refused and the
reflection alone carries the
install, logged as ``Jacobian tracker: carried set installed on its reflector alone over <m> directions, its signs
carrying no per-direction damping.``; a flipped carried column lying in the span of the others beyond the condition
cap refuses the whole set instead, reflection or not, logged as ``Jacobian tracker: no column was collected from the
carried modes, or a flipped one lies in the span of the others beyond the condition cap 1000, nothing installed.``,
while a stable column in that position is dropped from the per-direction damping alone and the rest of the set is
installed, logged as ``Jacobian tracker: <k> stable carried column(s) of lambda_Pi <v> lie in the span of the others
beyond the condition cap 1000 and are dropped.``
When there is no predecessor spectrum, the log notes
``No jacobian.npz in <path>; nothing carried.``, a predecessor whose
``jacobian.npz`` holds only the per-iteration traces, the mark of a run that never ended its loop, is passed over with
``<path> holds no certified spectrum (the run did not end its loop); nothing carried.``; a ``jacobian.npz`` that
cannot be read, an empty or truncated file, carries nothing either, with the warning ``<path> could not be read
(<error>); nothing carried.``, and so does one that lacks a key of the spectrum layout, with the warning ``<path> lacks
the key '<key>' of the spectrum layout; nothing carried.`` and the tracker left as it was; a predecessor
spectrum recorded for a different orbital count is refused with the warning
``The carried Jacobian spectrum belongs to another orbital count; nothing carried.``; one recorded on another momentum
grid or symmetry reduction, whose coordinates the vectors do not share, carries its eigenvalues and their damping
bound without vectors, with the warning
``The carried Jacobian spectrum belongs to another momentum grid or symmetry reduction; its eigenvalues are carried
without vectors.``; and a predecessor that did
not reach the pure fixed point is still carried, with the warning
``The predecessor's Jacobian spectrum in <path> belongs to a run that did not reach the pure fixed point.`` The bound
the carried spectrum allows applies from the first step to every damped step; a file that holds no certified mode
carries no bound, and the rung starts at the configured ``mixing``. The damping the predecessor ended with is written to
the file as a record and is not read back, so that a single early transient cannot pin every later rung of a ladder to
the value it forced; where the predecessor's tracker ended well below the configured ``mixing``, lowering ``mixing`` for
the next rung is a deliberate choice, not an automatic one. An accelerated step keeps the configured ``mixing``
whether or not a reflection is installed. A carried reflection, like the tracker's own, is held until a certified
stable mode lies inside it on three consecutive updates without a flip candidate, until the residual grows on three
consecutive updates, until the tracker's own certified flip replaces it, until a scaffold pauses the flips, or until the
run converges and the loop ends; the carried direction is the one the first iterations cannot yet certify, and its
absence from the estimate releases nothing. A scaffold pauses a carried flip rather than discarding
it: a reflection carried in while a scaffold is already on waits and is installed on the first update after the release,
and one installed before an annealing mass appears is withdrawn while the mass shapes the map and re-installed once
flips are allowed again. The forced release lands on the first update the scaffold shapes, whether or not that update
yields an estimate, so a reflector carried in at iteration 1 is withdrawn on the first iteration after the annealing
mass appears; the re-install happens only on an update that clears the tracker's own early gates (three recorded pairs
after a window restart, a step above the noise floor), two to three iterations after the mass goes. An exact-check set
released that way comes back as one. A refined momentum grid carries the eigenvalues and their bound without vectors.
A carried mode
still on the stable side of the boundary is flipped ahead of its crossing when two rungs say it will cross, the
preemptive flip arXiv:2606.04936 recommends for a carried Jacobian (it flips the modes with a very small positive real
part or approaching a pole; the supplement of the PRL instead identifies the mode by a cusp of the eigenvalue at the
converged points and applies the flip to the region beyond it, a cusp the carry-in reports but does not act on, see
below): the file a rung
writes holds, beside each mode's eigenvalue at the rung's own :math:`\beta_2`, the value of the mode it matched among
the predecessor's carried columns at :math:`\beta_1` (``lam_prev``, ``res_prev``, ``beta_prev``, matched by the same
eigenvector overlap once both live on the rung's window), and the next rung at :math:`\beta_3` extrapolates the real
part linearly in :math:`\beta`, :math:`\mathrm{Re}\,\lambda^{(2)} + (\mathrm{Re}\,\lambda^{(2)} -
\mathrm{Re}\,\lambda^{(1)})\,(\beta_3 - \beta_2) / (\beta_2 - \beta_1)`, and pre-flips a falling mode whose
extrapolation lies below the boundary by more than the undecidable band plus the larger error bound times one plus the
step ratio, logged as ``Jacobian tracker: cross-rung prediction: lambda_Pi=<v> at beta=<b2> from <v1> at beta=<b1>
extrapolates to <p> at beta=<b3>, flipped ahead of its crossing.``. A mode whose inverse, extrapolated the same way,
changes sign before :math:`\beta_3` while its modulus grew beyond 5 sits past a pole at the new rung and is pre-flipped
as well (the same line with ``(past the pole)``); the damping bound is then taken over the predicted spectrum too. The
extrapolation is refused, and the mode carried as before, when the new rung is more than twice the previous step away
(``Jacobian tracker: lambda_Pi=<v> has a step ratio <r> in beta above the cap 2, no cross-rung prediction.``), when
the predecessor did not reach the pure fixed point (``Jacobian tracker: the predecessor did not reach the pure fixed
point, no cross-rung prediction.``), or when the mode has no matched predecessor value. The linear extrapolation reads
a mode that approached the boundary and receded as a stable one, the cusp the PRL identifies a crossing by, so the
carry-in from a converged predecessor warns about every matched mode whose real part lay within the storage band of
zero at :math:`\beta_1` and rose again at :math:`\beta_2` by more than its error bar, the larger of the two bounds:
``Jacobian tracker: the carried lambda_Pi=<v> at beta=<b2> rose again from <v1> at beta=<b1>, close to the boundary: a
mode that touched the boundary between the rungs and receded, which can mean the predecessor converged beyond a
crossing; it is carried as measured, not flipped.`` A carried spectrum whose modes all lie above the storage band
stores no vector at all: it carries its damping bound and nothing else.

Four limits are worth knowing. The flip presumes an iterate inside the linear neighborhood of the physical fixed
point, as the underlying method does; it cannot carry an iteration across a pole surface, where the map is
singular, and it cannot resolve a direction whose contribution to the step lies below the rounding noise. The
reflection is built from the invariant subspace the flipped directions span, so an unstable direction that overlaps
a certified stable one (the rule rather than the exception for this non-normal map) is flipped while that stable one
keeps its own damping instead of the reflection; where a flipped and a stable eigenvalue nearly coincide, the two
subspaces cannot be separated, no reflection is installed, and the log says so, while a pair close enough to make
that separation steep but not singular is reflected orthogonally instead, which the log also records. Two certified
eigenvalues that nearly coincide are likewise not separable per direction and share one damping, the smaller of
theirs. And where the per-direction map would come out larger than twice the damping it encodes, the two groups of
blocks its largest coupling joins are merged, one merge after the other, until the map lies inside that cap: each
coupled group takes one uniform damping, the smallest of its members, while every block no strong coupling reaches
keeps its own, and the log names the count of coupled blocks and the damping of each group (``Jacobian tracker:
nonuniform damping not separable on <n> coupled certified blocks of the span (...), uniform damping <p> used on each
coupled group of them.``). A uniform damping keeps the sign of every certified direction on its own eigenvector, not on
the axis of the basis the span happens to be written in, which means that two certified directions of opposite sign
lying closer than about fifty degrees have no bounded uniform damping either: no linear map can reverse one of two
nearly parallel directions and fix the other. When even the whole span in one group lies above the cap, the
per-direction damping is refused outright, the log says so, and every certified direction takes the damped step at the
scalar bound instead. What stays in force there is the reflection. It keeps its oblique form, which reverses the
flipped direction and fixes a stable neighbor exactly, until its projector reaches the obliquity cap, a norm of ten;
for a single stable neighbor at angle :math:`t` that norm is :math:`1/\sin t`, so the cap fires only below about six
degrees, while with several stable neighbors the norm follows the angle to their whole span and the cap can fire at
larger
pairwise angles (about 12 to 14 degrees on the fixtures). Beyond the cap the orthogonal reflection takes over, which
reverses the flipped directions exactly and, being unable to fix a stable neighbor that close, reverses those too: a
stable direction at angle :math:`t` sees a Rayleigh quotient of :math:`-\cos 2t`, between -0.88 and -0.996 on the
small non-normal maps this was checked on, so its residual enters the step with the opposite sign instead of its
own. What bounds that is the damping the reflected residual then reaches the mixing at, ``p_eff``
on a damped step and the configured ``mixing`` on an accelerated one, and the fact that the per-direction step never
overwrites an accelerated step.
Flips are paused while a scaffold shapes the map, since the scaffolded map is not the physical one, and resume
after the release. And a
flipped direction is dropped again once the estimate certifies a stable mode inside the reflection on three
consecutive updates without a flip candidate (``Jacobian tracker: a certified stable mode lies in the tracked basis,
tracked basis released``, or the carried or exact-check basis), the mark of a wrong flip, since a stable direction
flipped by a noisy estimate grows under the reflection and is then measured stable inside it, or when the residual
keeps growing with the flip in force. A correct flip is not dropped when its direction has contracted out of the
secant window: its absence from the estimate says nothing, and the reflection stays, as the papers keep a flip in place
once it is made. The growth release can still cycle: it resets the persistence streak and holds re-installs off for
three more updates, after which a mode that certifies again is flipped anew, at a cost in iterations. Such a cycle also
slows the accelerated mixing: every install and every release of a reflection is an event that restarts the Pulay or
Anderson history from the pairs after it and costs one damped step (the per-direction damping alone is none), so a flip
that is installed and released every few updates turns a good part of the steps into damped ones and keeps the
accelerated history short, a safe fallback but a slow one. The tracker follows at most six directions, the Ritz values
of its window; when every value of a full window is certified and flipped, more unstable directions than it holds may
exist outside it, and it warns once per run, ``Jacobian tracker: every Ritz value of a full window of 6 is certified
and flipped; the window may hold fewer unstable directions than the map has, and the ones outside it are not
reflected.``
The per-direction damping itself
reaches the damped steps only, so a rung that spends most of its iterations on accelerated steps sees it on its warm-up
and its fallbacks alone; the reflection, a preconditioner of the map rather than a step, reaches every iteration. And
the tracker is blind at the start of a run
unless a spectrum is carried in: a cold run needs three pairs before it estimates anything, and pairs taken over
steps that grow from one iteration to the next do not certify, so the bound cannot protect the first steps of a
run without a carried predecessor. The flag is disabled with a warning in a one-shot run (``max_iter: 1`` or the
one-shot ``perform_lambda_correction``), which has no iteration history to read.

Lambda correction as a releasing scaffold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``use_lambda_correction`` the mass is the Moriya lambda correction, calibrated in every iteration so that the
momentum- and frequency-summed corrected susceptibility reproduces the local sum rule of the impurity,

.. math::

   \frac{1}{\beta\, n_{\mathbf{q}}} \sum_{\mathrm{q}} \left[\left(\chi^{\mathrm{q}}_{\mathrm{r}}\right)^{-1} +
   \lambda_{\mathrm{r}}\right]^{-1} = \frac{1}{\beta} \sum_{\omega} \chi^{\omega}_{\mathrm{r},\mathrm{loc}} .

Near its pole the ladder overestimates the susceptibility. The sum rule caps the total weight it may accumulate, so
the calibrated mass comes out just large enough to pull the static susceptibility back from the pole. Which correction
runs depends on the band count. For a single band, :math:`\lambda_{\mathrm{r}}` is one number per channel. Inside the
cycle the search is confined to one branch of the sum rule and the bracket starts from the previous iteration's
:math:`\lambda`, because a warm-started state can carry a negative susceptibility at a finite frequency whose pole
lies above the static bound, and an unbounded search then jumps between poles instead of converging. Every entry of
the susceptibility contributes a pole to the sum rule at :math:`-1/\chi^{\mathrm{q}}_{\mathrm{r}}`, and the sum
rule falls off from one pole to the next. The lower end of the branch is the larger of two bounds. The static bound
lies above every pole of the :math:`\omega = 0` slice, because only there is the corrected static susceptibility
positive at every momentum, which is what the correction exists to secure. The finite-frequency bound, the largest
:math:`-\mathrm{Re}(1/\chi^{\mathrm{q}}_{\mathrm{r}})` over the entries at :math:`\omega \neq 0` with a positive real
part, keeps the real part of each of those entries positive after the correction. It lies above the static bound only
when such an entry has :math:`|\chi^{(\mathbf{q},\omega_n)}_{\mathrm{r}}|^2 / \mathrm{Re}\,
\chi^{(\mathbf{q},\omega_n)}_{\mathrm{r}} > \max_{\mathbf{q}'} \chi^{(\mathbf{q}',\omega=0)}_{\mathrm{r}}`: for a real
susceptibility when the entry rises above the static maximum, as at the first-frequency density pole described above,
and for a complex one also through a large imaginary part; both bounds then lie below zero, so the branch
contains :math:`\lambda = 0`, and the root found there keeps every positive entry positive instead of the one below
its pole that would turn it negative. The branch is closed at the next pole above its lower end, which only the
negative entries place there, at any bosonic frequency. Both ends coincide with the values the one-shot search walks
whenever the static bound is the lower end and no negative entry sits above it. Anchoring the branch at the largest
pole of all entries alike instead does not work: the high-frequency tail of a computed susceptibility is near zero
and slightly negative, an artifact of the truncated vertex box, and every one of those entries places a pole
thousands above the physical root, so the search reports no root at all. The applied value is logged with the suffix
``(bounded, warm start from <lambda>)``, ``none`` on the first iteration. When the sum rule has no sign change on the
branch, the log warns ``Lambda correction found no sum-rule root on the branch (<lower>, <upper>); keeping <which>.``
(preceded by ``Lambda correction could not bracket the sum-rule root inside the branch.`` when an open branch gave
no sign change within reach), and the iteration keeps, in this order, the previous iteration's :math:`\lambda` when
it lies inside the branch (``<which>`` reads ``the previous iteration's lambda <value>``), zero when zero does
(``zero (no correction)``), the branch midpoint on a closed branch (``the branch midpoint``), or the lower end plus 0.1
on an open one (``the lower end plus delta``); such a fallback is never stored as the next warm start. The one-shot
correction of
``perform_lambda_correction`` keeps its Newton iteration started just above the static bound. The ``type`` field of
the lambda correction section selects whether both channels are corrected (``spch``) or only the magnetic one
(``sp``, with the density sum rule folded into the magnetic target). For several bands the mass is a
real-symmetric :math:`n_{\mathrm{o}}^2 \times n_{\mathrm{o}}^2`
matrix :math:`\Lambda_{\mathrm{r}}` per channel that matches the sum rule component by component. It comes from a
damped Newton iteration whose line search keeps the static susceptibility gap positive; both channels are always
corrected, and the momentum sum runs over the full Brillouin zone because symmetry-related momenta carry orbitally
rotated susceptibility matrices. The single-band scheme is the derived Moriya correction. The matrix scheme is a
heuristic that mimics it, and inside the cycle both do the same job of stabilizing the iteration.

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
of the release iteration is then still on disk as ``Sigma_Iterates/sigma_dga_iteration_<i>.npy``. It is a
well-defined object in
its own right, but a lambda-corrected DΓA solution rather than a self-consistent one.

Eigenvalue restriction of the susceptibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The option ``use_chi_phys_restriction`` calibrates nothing. It enforces the two bounds a bosonic susceptibility
obeys at every momentum: positive semi-definite at the static frequency, and no larger than the static value at a
finite frequency, since :math:`\chi^{(\mathbf{q},\omega_n)}` decreases with :math:`|\omega_n|`. At the static
frequency it takes the compound :math:`n_{\mathrm{o}}^2 \times n_{\mathrm{o}}^2` block of the inverse susceptibility,
splits it into Hermitian and skew-Hermitian parts, diagonalizes the Hermitian part and floors its eigenvalues, leaves
the skew-Hermitian part alone, and inverts the block back. A negative eigenvalue of the inverse is the signature of a
crossed pole. The floor is half the smallest inverse eigenvalue of the healthy static blocks of that channel, taken
over all momenta and ranks, so a crossed mode is pinned at twice the largest healthy static susceptibility instead of
at an arbitrary large value, which would kick the self-energy violently; when no static block is healthy the floor is
:math:`10^{-2}`. At every finite frequency it diagonalizes the Hermitian part of the susceptibility block itself and
clips its eigenvalues into :math:`[-c, c]`, with :math:`c` the largest static eigenvalue at that momentum, and bounds
the skew-Hermitian part, which vanishes for a physical susceptibility but spikes with the pole, the same way. A value
beyond the static one is a pole of the ladder at that frequency, such as the first-frequency density pole a
self-consistent rung can develop while its static slice stays healthy; the symmetric window leaves the small negative
values that the truncation of the frequency box produces at high frequencies alone. Every block within both bounds
passes through bit for bit, including negative off-diagonal *matrix elements*, which are perfectly legitimate in a
multi-orbital susceptibility and which an elementwise clamp would destroy. For a single band the blocks are scalars, and
the restriction reduces to pinning a negative static value and clipping the real and imaginary parts of the
finite-frequency values. In the language of the mass shift above, :math:`\lambda` is an operator living on the poled
modes alone, just strong enough to bring each of them back within its bound.

The two-phase schedule is the same as for the lambda correction. The restriction stays active until the cycle
converges at the relaxed threshold, is then released with a reset mixing history, and the unrestricted phase
converges to the full ``epsilon``. The per-iteration log line

.. code-block:: text

   Restricted physical susceptibility (dens): N eigenvalues restricted (static inverse floored, finite-frequency
   values clipped to the static maximum). Releasing the restriction is only safe once this count decays to zero.

is the diagnostic to watch. The restriction only bites where one of the two bounds is violated, so on a healthy
susceptibility the count is zero and the option changes nothing. During a scaffolded phase the count
should decay to zero as the self-energy relaxes. A release while :math:`N` is still large removes a regularization
the iterate still leans on, and the unrestricted phase will then most likely relapse. The restricted blocks are a
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
     - crossed static modes pinned at twice the largest healthy value, finite-frequency values clipped to the
       static maximum, acting only on poled modes
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
     - only where a static eigenvalue of the inverse falls below the floor or a finite-frequency value exceeds
       the static one
     - only while a static gap is negative
   * - typical use
     - the strongest pull; also the right choice when the corrected state itself is of interest
     - a first, minimal intervention when only a few modes cross the pole
     - a gradual descent when the pure map is expected to be stable but hard to reach

The Jacobian tracking is not in the table because it is not a fourth scaffold: the three above reshape the
susceptibility and change the map, while the tracking leaves the map alone and changes the iteration that walks it.
It is therefore the first thing to switch on, and the only one that costs no evaluation of the map: a rung whose
certified spectrum holds no unstable direction never flips, but it still differs from a run without the flag, since
the measured bound lowers the damping of its damped steps, the per-direction damping takes over the certified
directions of those steps, and the convergence test reads those steps at the configured mixing and waits for the
minimum iteration count of the certified spectrum.
What it buys is a rung whose start sits near a direction the map has already turned unstable, the usual situation
after a large cooling step, where the reflection and the measured bound keep the iteration on the physical branch
until the mixing can take over. What it does not do is choose the fixed point: it makes the physical one attractive
where the estimate reaches, and a start that contracts into another one still lands there.

None of the three is tied to a cooldown; each can just as well support a cold run at a temperature where the plain
cycle fails, and the Jacobian tracking can accompany any of them. Whatever the technique, the result is certified the
same way as above: a converged pure phase, a non-negative static susceptibility in both channels on the final
iteration, no pole warnings at the end (the first-frequency one included), no warning that the leading singlet-even
and triplet-even Eliashberg eigenvalues agree to within 1% when the frequency parity is resolved (a hint rather than
a verdict, since two physical sectors can meet by accident, and expected for an SU(2N)-symmetric interaction), and
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
   final ``Lowest eigenvalue of the static susceptibility`` of both channels is not negative beyond the small
   truncation offset (unless the material genuinely orders in that channel). Look at the Eliashberg sectors
   as well: coinciding sectors and a jump from the previous rung are the signature of a wrong fixed point. Note the
   run folder
   ``LDGA_Nk<nk_tot>_Nq<nk_tot>_wc<niw_core>_vc<niv_core>_vs<niv_shell>`` it created, and the ``p_eff`` of the
   tracker's convergence line.
5. Configure the next rung: ``input_path`` set to its own DMFT data, ``previous_sc_path`` set to the predecessor's
   run folder, ``use_interpolated_sigma: True``, and the interpolation section aimed at the rung after it. Leave
   the interpolation off on the last rung. The example file :ref:`dga_config_beta25.yaml <cooldown-config-beta25>`
   marks the three entries that change from one rung to the next. If the predecessor's tracker ended well below
   the configured ``mixing``, a smaller ``mixing`` for this rung is worth considering. With the flag
   on, the rung also reads the predecessor's ``jacobian.npz``; nothing has to be configured for that.
6. Run it and check the log for ``Using previous calculation and starting the self-consistency loop at iteration
   N`` with an iteration number that continues from the predecessor; without that line the rung started cold.
7. If the rung stops at ``max_iter``, do not chain from it. Either resume it at the same temperature
   (``previous_sc_path`` pointing at its own folder, ``use_interpolated_sigma: False``), insert an intermediate
   temperature, or switch on one flag of the :ref:`stabilization section <stabilization>` for that rung and rerun
   it; rerunning with ``use_jacobian_stabilization`` first shows in the log whether the map overshoots or expands.
   If the rung converged but its Eliashberg sectors look like those of step 4, do not resume it either; rerun it
   from its predecessor with a changed start.
8. Repeat steps 4 to 7 down the ladder. Solve the Eliashberg equation on converged rungs only.
9. Before quoting results, reproduce at least one rung by a second route or on a finer momentum grid and confirm
   that the self-energy and the Eliashberg eigenvalues agree.

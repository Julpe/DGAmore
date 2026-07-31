Output files
============

Every quantity a run produces is written as a plain NumPy ``.npy`` file (created with ``allow_pickle=False``, so the
files contain nothing but the raw array). All of them end up under the ``output_path`` of the ``output``
configuration section, in a run-specific subdirectory whose name encodes the momentum grid and the frequency box,
following the pattern ``LDGA_Nk<nk_tot>_Nq<nk_tot>_wc<niw_core>_vc<niv_core>_vs<niv_shell>``. That directory is
referred to as ``output_path`` below and holds the main results; the quantities of the Eliashberg step go to the
subfolder ``eliashberg_path`` and the figures to the subfolder ``plotting_path``, both named in the configuration
file and described on the :doc:`configuration` page.

Because a ``.npy`` file stores only the array, none of the metadata that the corresponding class carries survives the
round trip: the spin channel, the frequency notation, the momentum layout and the two frequency ranges have to be
supplied again when the file is read back. The conventions below fix the layout of every stored array, and the tables
list the layout of each individual quantity.

Storage conventions
-------------------

The following symbols are used throughout this page:

* :math:`n_{\mathrm{o}}` (written ``no`` in the tables) is the number of bands (orbitals) of the full multi-band
  problem,
* ``(nkx, nky, nkz)`` is the momentum grid ``nk``, :math:`n_{\mathbf{q}}^{\mathrm{irr}}` (written ``nq_irr``) the
  number of momenta in the irreducible Brillouin zone and ``nq_rank`` the number of them held by one MPI rank,
* ``niw_core``, ``niv_core``, ``niv_full`` (``= niv_core + niv_shell``) and ``niv_dmft`` are the frequency-box sizes
  of the ``box_sizes`` configuration section,
* ``niv_pp = min(niw_core // 2, niv_core // 2)`` is the fermionic box of the particle-particle (Eliashberg)
  quantities.

**Axis order.** Every stored array follows the same order: first the momentum axes (absent for purely local
quantities), then the orbital axes, then the bosonic frequency axis (if the quantity has one), then the fermionic
frequency axes (up to two).

**Momentum.** The momentum-dependent quantities are stored in one of two layouts, depending on the type of object:

* The single-particle (two-point) quantities - the self-energy, the Green's function and the gap function - are
  stored in the **full Brillouin zone** with three **separate** momentum axes, that is ``[kx, ky, kz, ...]``.
* The vertex-like (four-point) quantities - the susceptibilities, the three-leg vertices, the bubble and the ladder
  and pairing vertices - are stored in the **irreducible Brillouin zone** with a single **compressed** momentum axis,
  that is ``[q, ...]`` with :math:`n_{\mathbf{q}}^{\mathrm{irr}}` entries. Their momenta are the irreducible
  representatives in the order of ``k_grid.irrk_ind``. To recover the full zone, use
  :meth:`~dgamore.n_point_base.IAmNonLocal.map_to_full_bz` rather than plain indexing with ``k_grid.irrk_inv``: on
  grids whose symmetries were discovered automatically, symmetry-related momenta additionally carry an orbital
  rotation that only that method applies.

**Orbitals.** Two-point quantities carry two orbital axes, four-point quantities carry four, each of length
:math:`n_{\mathrm{o}}`. The index convention is the one of the operator ordering
:math:`G_{1234} = \langle T[c_1 c^\dagger_2 c_3 c^\dagger_4]\rangle`, so the first and third orbital index belong to
an annihilation operator and the second and fourth to a creation operator.

**Bosonic frequencies.** A quantity with a bosonic frequency axis is always saved over the **positive slice only**,
that is over :math:`\omega \geq 0`. The axis therefore has ``niw_core + 1`` entries and index ``i`` corresponds to
:math:`\omega_i = 2 i \pi / \beta`, with index ``0`` being :math:`\omega = 0`. The negative half is not stored because
it follows from the time-reversal symmetry

.. math:: F^{-\omega,-\nu,-\nu'}_{1234} = \left(F^{\omega\nu\nu'}_{1234}\right)^{*} .

Use :meth:`~dgamore.local_n_point.LocalNPoint.to_full_niw_range` to restore the signed bosonic axis after loading, or
:meth:`~dgamore.local_n_point.LocalNPoint.to_negative_niw_range` to obtain the negative block by itself.

**Fermionic frequencies.** Fermionic axes are stored over the full, signed range. An axis with ``niv`` positive
frequencies therefore has ``2 * niv`` entries, running from :math:`\nu = -\mathrm{niv}` to
:math:`\nu = \mathrm{niv} - 1`, so index ``i`` corresponds to :math:`\nu_i = (2 (i - \mathrm{niv}) + 1) \pi / \beta`.
The two fermionic axes of a quantity need not have the same length: the local full vertex used for the
double-counting correction is stored on an asymmetric box, since one of its indices has to be summed over the full
asymptotic region. The summed frequency is the second argument of the vertex, which the double-counting kernel
reads off the stored (full-box) first axis via the compound symmetry
:math:`F^{\omega\nu\nu'}_{1234} = F^{\omega\nu'\nu}_{4321}` of the symmetrized local vertex.

**Data type.** The n-point objects store their array as ``complex64`` to save memory, and the files inherit that
data type. The few purely real outputs (the chemical-potential history and the spectral functions) are stored as
``float64`` and ``float32`` respectively.

Reading a quantity back
-----------------------

The four-point classes provide a ``load`` method whose defaults already match the conventions above: one bosonic and
two fermionic axes, the half bosonic range, the full fermionic range, particle-hole notation and, for the
momentum-dependent class, a compressed momentum axis. Everything that deviates from those defaults has to be passed
explicitly, since the file itself carries no metadata. The examples below assume

.. code-block:: python

   import numpy as np

   from dgamore.four_point import FourPoint
   from dgamore.gap_function import GapFunction
   from dgamore.greens_function import GreensFunction
   from dgamore.local_four_point import LocalFourPoint
   from dgamore.n_point_base import FrequencyNotation, SpinChannel
   from dgamore.self_energy import SelfEnergy

   run = "LDGA_Nk1024_Nq1024_wc60_vc60_vs200"   # the run's output directory
   eliashberg = f"{run}/Eliashberg"             # its Eliashberg subfolder, named in the configuration
   nk = (32, 32, 1)                             # the momentum grid the run used
   beta = 12.5                                  # the run's inverse temperature

Local four-point quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only the channel and the number of frequency axes distinguish these calls, plus the frequency notation for the
particle-particle diagrams:

.. code-block:: python

   # two fermionic axes (the default): irreducible vertex, generalized susceptibility, two-particle Green's function
   gamma_dens = LocalFourPoint.load(f"{run}/gamma_dens_loc.npy", SpinChannel.DENS)
   gchi_magn = LocalFourPoint.load(f"{run}/gchi_magn_loc.npy", SpinChannel.MAGN)
   g2_dens = LocalFourPoint.load(f"{run}/g2_dens_loc.npy", SpinChannel.DENS)

   # the double-counting vertex takes the same arguments; only its two fermionic axes have different lengths
   f_dc = LocalFourPoint.load(f"{run}/f_dc_loc.npy", SpinChannel.MAGN)

   # one fermionic axis: three-leg vertex and bare bubble
   vrg_dens = LocalFourPoint.load(f"{run}/vrg_dens_loc.npy", SpinChannel.DENS, num_vn_dimensions=1)
   gchi0 = LocalFourPoint.load(f"{run}/gchi0_loc.npy", num_vn_dimensions=1)

   # no fermionic axis: physical susceptibility
   chi_dens = LocalFourPoint.load(f"{run}/chi_dens_loc.npy", SpinChannel.DENS, num_vn_dimensions=0)

   # particle-particle diagrams at omega = 0; gamma keeps a single-entry bosonic axis, f and phi have none
   gamma_ud_pp = LocalFourPoint.load(
       f"{eliashberg}/gamma_ud_loc_pp_w0.npy", SpinChannel.UD, frequency_notation=FrequencyNotation.PP
   )
   f_ud_pp = LocalFourPoint.load(
       f"{eliashberg}/f_ud_loc_pp_w0.npy",
       SpinChannel.UD,
       num_wn_dimensions=0,
       frequency_notation=FrequencyNotation.PP,
   )

   # the file holds omega >= 0 only, so restore the signed bosonic axis before using the object
   gamma_dens = gamma_dens.to_full_niw_range()

Momentum-dependent four-point quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The momentum grid is the third positional argument. The stored momentum axis is compressed and covers the
irreducible zone, which is what the ``has_compressed_q_dimension=True`` default expresses:

.. code-block:: python

   # no fermionic axis: physical and RPA susceptibility
   chi_phys_dens = FourPoint.load(f"{run}/chi_phys_q_dens.npy", SpinChannel.DENS, nk, num_vn_dimensions=0)
   chi_rpa_magn = FourPoint.load(f"{run}/chi_rpa_q_magn.npy", SpinChannel.MAGN, nk, num_vn_dimensions=0)

   # two fermionic axes: the full ladder vertex written with save_fq
   f_q_dens = FourPoint.load(f"{run}/f_irrq_dens.npy", SpinChannel.DENS, nk)

   # pairing vertex: particle-particle notation without a bosonic axis
   gamma_sing = FourPoint.load(
       f"{eliashberg}/gamma_irrq_sing_pp.npy",
       SpinChannel.SING,
       nk,
       num_wn_dimensions=0,
       frequency_notation=FrequencyNotation.PP,
   )

Unfolding such an object to the full zone needs the same :class:`~dgamore.brillouin_zone.KGrid` the run used, because
the momentum axis only stores one representative per star. With a fixed symmetry list the grid is built from that list
alone, but a run configured with ``symmetries: auto`` discovered its group from the Hamiltonian, so the grid has to
rediscover it from the same ``wannier_hr.dat``:

.. code-block:: python

   from dgamore.brillouin_zone import KGrid, KnownSymmetries
   from dgamore.hamiltonian import Hamiltonian

   # KnownSymmetries.AUTO defers the reduction; until it is resolved the grid behaves as if the zone were unreduced
   k_grid = KGrid(nk, [KnownSymmetries.AUTO])

   # evaluate H(k) of the run's hopping file on that grid and let the grid discover its symmetry group
   ek = Hamiltonian().read_hr_w2k(f"{run}/wannier_hr.dat").get_ek(k_grid)
   k_grid.specify_auto_symmetries(ek)

   # restore the signed bosonic axis and unfold to the full zone
   chi_phys_dens = chi_phys_dens.to_full_niw_range().map_to_full_bz(k_grid)

The discovery has to happen before the unfold: on an auto grid the symmetry-related momenta of a star carry an orbital
rotation, and :meth:`~dgamore.n_point_base.IAmNonLocal.map_to_full_bz` applies it from the data
``specify_auto_symmetries`` stores on the grid. Unfolding against a grid whose group was never resolved silently
replicates the irreducible values instead.

The per-rank intermediates are read the same way, but their momentum axis holds only the momenta of the writing
rank instead of the whole irreducible zone. They are therefore loaded with the default grid, and their leading axis
must be treated as an opaque chunk rather than as a momentum index:

.. code-block:: python

   gchi0_q_inv = FourPoint.load(f"{eliashberg}/gchi0_q_inv_rank_0.npy", num_vn_dimensions=1)
   vrg_q_dens = FourPoint.load(f"{eliashberg}/vrg_q_dens_rank_0.npy", SpinChannel.DENS, num_vn_dimensions=1)

Two-point quantities
~~~~~~~~~~~~~~~~~~~~

The two-point classes provide the same ``load`` method, each returning its own class. They carry no bosonic axis and
their momentum default is the decompressed layout these files are written in, so only the momentum grid and the
extra arguments the concrete behavior needs are left to supply - ``beta`` for the self-energy moments and the
Green's function, the pairing channel for the gap:

.. code-block:: python

   sigma_dga = SelfEnergy.load(f"{run}/sigma_dga.npy", nk=nk, beta=beta)

   # the local quantities are stored on a single momentum point, so nk keeps its default
   sigma_dmft = SelfEnergy.load(f"{run}/sigma_dmft.npy", beta=beta)
   g_dmft = GreensFunction.load(f"{run}/g_dmft.npy", beta=beta)

   giwk_dga = GreensFunction.load(f"{run}/giwk_dga.npy", nk=nk, beta=beta)
   gap = GapFunction.load(f"{eliashberg}/gap_sing_even_1.npy", SpinChannel.SING, nk)

Local quantities
----------------

These files are written to ``output_path`` by rank 0 after the local Schwinger-Dyson step. The four-point
quantities among them carry no momentum axis at all. The two-point ones do: the DMFT Green's function and the two
self-energies descend from :class:`~dgamore.two_point.TwoPoint`, so they are momentum-dependent objects evaluated on
a single momentum point and keep three leading axes of length one.

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - File
     - Quantity
     - Array layout
   * - ``g2_dens_loc.npy``, ``g2_magn_loc.npy``
     - Two-particle DMFT Green's function :math:`G^{(2)}_{\mathrm{r}}`
     - ``[no, no, no, no, niw_core + 1, 2 niv_core, 2 niv_core]``
   * - ``gchi_dens_loc.npy``, ``gchi_magn_loc.npy``
     - Generalized susceptibility :math:`\chi_{\mathrm{r}}`
     - ``[no, no, no, no, niw_core + 1, 2 niv_core, 2 niv_core]``
   * - ``gamma_dens_loc.npy``, ``gamma_magn_loc.npy``
     - Irreducible vertex :math:`\Gamma_{\mathrm{r}}`
     - ``[no, no, no, no, niw_core + 1, 2 niv_core, 2 niv_core]``
   * - ``f_dens_loc.npy``, ``f_magn_loc.npy``
     - Full vertex :math:`F_{\mathrm{r}}`, both fermionic axes on the core box
     - ``[no, no, no, no, niw_core + 1, 2 niv_core, 2 niv_core]``
   * - ``f_dc_loc.npy``
     - Magnetic full vertex for the double-counting kernel, first fermionic axis on the full asymptotic box
     - ``[no, no, no, no, niw_core + 1, 2 niv_full, 2 niv_core]``
   * - ``vrg_dens_loc.npy``, ``vrg_magn_loc.npy``
     - Three-leg vertex :math:`\gamma_{\mathrm{r}}`
     - ``[no, no, no, no, niw_core + 1, 2 niv_core]``
   * - ``gchi0_loc.npy``
     - Bare bubble :math:`\chi_{0}` (only written if the Eliashberg step runs)
     - ``[no, no, no, no, niw_core + 1, 2 niv_full]``
   * - ``chi_dens_loc.npy``, ``chi_magn_loc.npy``
     - Physical susceptibility :math:`\chi^{\omega}_{\mathrm{r}}`
     - ``[no, no, no, no, niw_core + 1]``
   * - ``g_dmft.npy``
     - Local DMFT Green's function
     - ``[1, 1, 1, no, no, 2 niv_dmft]``
   * - ``sigma_dmft.npy``
     - Local DMFT self-energy
     - ``[1, 1, 1, no, no, 2 niv_dmft]``
   * - ``siw_dga_local.npy``
     - Local self-energy of the local DΓA step
     - ``[1, 1, 1, no, no, 2 niv_core]``

Momentum-dependent quantities
-----------------------------

These files are written to ``output_path`` during and after the self-consistency loop. The susceptibilities live in
the irreducible zone with a compressed momentum axis; the single-particle quantities live in the full zone with three
separate momentum axes.

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - File
     - Quantity
     - Array layout
   * - ``chi_phys_q_dens.npy``, ``chi_phys_q_magn.npy``
     - Physical ladder susceptibility :math:`\chi^{\mathrm{q}}_{\mathrm{r}}` after the lambda correction
     - ``[nq_irr, no, no, no, no, niw_core + 1]``
   * - ``chi_rpa_q_dens.npy``, ``chi_rpa_q_magn.npy``
     - RPA susceptibility, written once in the first iteration
     - ``[nq_irr, no, no, no, no, niw_core + 1]``
   * - ``f_irrq_dens.npy``, ``f_irrq_magn.npy``
     - Full ladder vertex :math:`F^{\mathrm{q}\nu\nu'}_{\mathrm{r}}` (only written with ``save_fq``)
     - ``[nq_irr, no, no, no, no, niw_core + 1, 2 niv_core, 2 niv_core]``
   * - ``sigma_dga.npy``
     - Converged non-local self-energy :math:`\Sigma^{\mathrm{k}}_{12}`
     - ``[nkx, nky, nkz, no, no, 2 niv]``
   * - ``giwk_dga.npy``
     - Non-local Green's function built from the converged self-energy
     - ``[nkx, nky, nkz, no, no, 2 niv]``
   * - ``g_latt_dmft.npy``
     - Lattice Green's function built from the DMFT self-energy
     - ``[nkx, nky, nkz, no, no, 2 niv]``
   * - ``sigma_dga_iteration_<i>.npy``
     - Self-energy after self-consistency iteration ``i`` (also read back when a run is resumed)
     - ``[nkx, nky, nkz, no, no, 2 niv]``
   * - | ``sigma_dga_interpolated``
       | ``_beta<b>_niv<n>_iteration_<i>.npy``
     - The same self-energy interpolated to the target temperature and frequency box
     - ``[nkx, nky, nkz, no, no, 2 n]``
   * - ``mu_history.npy``
     - Chemical potential of every self-consistency iteration, ``float64``
     - ``[n_iterations]``

The fermionic box ``niv`` of the single-particle quantities is the one the Schwinger-Dyson step produces and is
generally larger than ``niv_core``; it is not a configuration parameter, so read it off the stored array.

Eliashberg quantities
---------------------

These files are written to ``eliashberg_path``. The vertices among them use the particle-particle frequency notation
at :math:`\omega = 0`, which is why they have no bosonic axis (or a single-entry one).

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - File
     - Quantity
     - Array layout
   * - ``gap_<channel>_<i>.npy``
     - The ``i``-th leading gap function of the singlet or triplet channel, written when
       ``resolve_frequency_parity`` is disabled
     - ``[nkx, nky, nkz, no, no, 2 niv_pp]``
   * - ``gap_<channel>_<parity>_<i>.npy``
     - The ``i``-th leading gap function of the given channel and frequency-parity sector (``even`` or ``odd``),
       written when ``resolve_frequency_parity`` is enabled
     - ``[nkx, nky, nkz, no, no, 2 niv_pp]``
   * - ``gamma_irrq_sing_pp.npy``, ``gamma_irrq_trip_pp.npy``
     - Pairing vertex of the singlet and triplet channel (only written with ``save_pairing_vertex``)
     - ``[nq_irr, no, no, no, no, 2 niv_pp, 2 niv_pp]``
   * - ``gamma_ud_loc_pp_w0.npy``
     - Local up-down irreducible vertex in pp notation; its bosonic axis holds the single entry
       :math:`\omega = 0`
     - ``[no, no, no, no, 1, 2 niv_pp, 2 niv_pp]``
   * - ``f_ud_loc_pp_w0.npy``, ``phi_ud_loc_pp_w0.npy``
     - Local up-down full vertex and pp-reducible diagrams, written with ``include_local_part``
     - ``[no, no, no, no, 2 niv_pp, 2 niv_pp]``

Per-rank intermediates
----------------------

When the Eliashberg step is enabled, the self-consistency loop additionally dumps the ingredients of the ladder
vertex to ``eliashberg_path``, one file per MPI rank, so that the vertex can be assembled after the loop has
converged. Their leading axis holds only the irreducible momenta of the writing rank, they are overwritten in every
iteration, and they are deleted once the vertex has been built.

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - File
     - Quantity
     - Array layout
   * - ``gchi0_q_inv_rank_<r>.npy``
     - Inverse bare bubble :math:`(\chi^{\mathrm{q}\nu}_{0})^{-1}` on the core box
     - ``[nq_rank, no, no, no, no, niw_core + 1, 2 niv_core]``
   * - ``vrg_q_<channel>_rank_<r>.npy``
     - Three-leg vertex :math:`\gamma^{\mathrm{q}\nu}_{\mathrm{r}}`
     - ``[nq_rank, no, no, no, no, niw_core + 1, 2 niv_core]``
   * - ``vrg_q_<channel>_right_rank_<r>.npy``
     - The corresponding right (left-summed) three-leg vertex
     - ``[nq_rank, no, no, no, no, niw_core + 1, 2 niv_core]``
   * - ``chi_phys_q_<channel>_rank_<r>.npy``
     - Physical susceptibility of that rank's momenta
     - ``[nq_rank, no, no, no, no, niw_core + 1]``

Real-frequency and plain-text output
------------------------------------

The analytic continuation and a few diagnostics do not follow the n-point layout:

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - File
     - Quantity
     - Array layout
   * - ``spectral_function_dga.npy``
     - DΓA spectral function :math:`A(\mathbf{k}, \omega)` in the full Brillouin zone, ``float32``
       (written when ``do_spectrum_dga`` is enabled)
     - ``[nkx, nky, nkz, no, w_count]``
   * - ``spectral_function_dmft.npy``
     - DMFT spectral function on the same real-frequency grid and in the same momentum layout
       (written when ``do_spectrum_dmft`` is enabled)
     - ``[nkx, nky, nkz, no, w_count]``
   * - ``oz_coeff.txt``
     - Ornstein-Zernike fit of the magnetic susceptibility, one line per orbital combination with the columns
       ``o1 o2 o3 o4 A xi``
     - text
   * - ``eigenvalues.txt``
     - Leading Eliashberg eigenvalues, one comma-separated line per channel and parity sector
     - text
   * - ``gap_parity.txt``
     - One line per saved gap function with its wave-symmetry label and the measured parity expectation values
     - text

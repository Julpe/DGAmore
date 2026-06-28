DGAmore
=======

.. image:: https://github.com/Julpe/DGAmore/actions/workflows/CI.yml/badge.svg
   :target: https://github.com/Julpe/DGAmore/actions/workflows/CI.yml
   :alt: CI status

.. image:: https://codecov.io/github/Julpe/DGAmore/graph/badge.svg?token=O1E161NNHP
   :target: https://codecov.io/github/Julpe/DGAmore
   :alt: Code coverage

Welcome
-------

DGAmore is a Python toolbox that computes the multi-orbital, self-consistent ladder Dynamical Vertex
Approximation and solves the Eliashberg equation for (strongly) correlated electron systems described by the
multi-band Hubbard model. Starting from the one- and two-particle output of a dynamical mean-field theory (DMFT)
calculation, it assembles the local vertex functions, solves the momentum-dependent ladder equations to obtain the
non-local self-energy, and, optionally, extracts the leading superconducting eigenvalues and gap functions in the
singlet and triplet channels. It is built for usability and a minimal setup effort, is openly developed on GitHub
under the MIT license, and is extensively unit- and end-to-end tested on Linux and macOS.

The code combines and extends two earlier approaches, the one-shot multi-orbital
`AbinitioDGA <https://doi.org/10.1016/j.cpc.2019.07.012>`_ and the single-band
`DGApy <https://doi.org/10.5281/zenodo.10406493>`_, on which it is partially based. On top of these it adds
explicit multi-orbital high-frequency asymptotics,
a self-consistency loop with prediction-based self-energy mixing, and the multi-orbital Eliashberg equation, which
together make it well suited to studying the low-temperature physics of multi-orbital materials such as the bilayer
nickelate :math:`\mathrm{La}_3\mathrm{Ni}_2\mathrm{O}_7`. Numerically it relies on vectorized NumPy operations, distributes the heavy momentum-dependent
work across cores with MPI, and exploits lattice and frequency symmetries together with an asymptotic treatment of
the vertex functions to keep the memory footprint manageable. Throughout a run it emits detailed logs, so that
progress and any problems are easy to follow both in a terminal and in cluster job output.

The toolbox is organised as a pipeline of focused modules. A run is orchestrated by the entry point
:mod:`~dgamore.DGAmore` and configured through a process-wide configuration object in :mod:`~dgamore.config`,
populated from a YAML file by :mod:`~dgamore.config_parser`. Input handling lives in :mod:`~dgamore.dga_io` and
:mod:`~dgamore.dmft_interface`, which read the one- and two-particle quantities from the DMFT output, while
:mod:`~dgamore.hamiltonian` builds the kinetic dispersion and interaction tensors and :mod:`~dgamore.brillouin_zone`
together with :mod:`~dgamore.symmetry_reduction` set up the irreducible momentum grid. The physical quantities
themselves are thin wrappers around NumPy arrays defined in :mod:`~dgamore.n_point_base` and its descendants, among
them :mod:`~dgamore.local_four_point`, :mod:`~dgamore.four_point`, :mod:`~dgamore.greens_function`,
:mod:`~dgamore.self_energy`, :mod:`~dgamore.interaction` and :mod:`~dgamore.gap_function`, which carry the orbital,
frequency and momentum bookkeeping. The local vertices and the local self-energy are assembled in
:mod:`~dgamore.local_sde`, the bare bubble in :mod:`~dgamore.bubble_gen`, and the momentum-dependent ladder
self-energy in :mod:`~dgamore.nonlocal_sde`, with an optional Moriya correction in
:mod:`~dgamore.lambda_correction`. Pairing properties are obtained in :mod:`~dgamore.eliashberg_solver`, and an
optional continuation to real frequencies is provided by :mod:`~dgamore.max_ent`. The remaining modules support the
computation: :mod:`~dgamore.matsubara_frequencies` handles frequency-index arithmetic,
:mod:`~dgamore.mpi_utils` manages the parallel work distribution (message chunking, the work distributor and the
data-movement routines), :mod:`~dgamore.dga_logger` provides the structured logging, and
:mod:`~dgamore.plotting` produces the diagnostic figures.

For the physics and the precise equations behind the implementation, see the accompanying paper and the author's
Master's thesis, both linked on the :doc:`about` page. The pages listed in the sidebar walk through installation,
running a calculation, and contributing, while the :doc:`api` documents every module in detail.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   configuration
   contributing
   about
   api

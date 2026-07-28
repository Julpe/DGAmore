# Contributing to DGAmore

First of all, thank you for taking the time to contribute. `DGAmore` is an openly developed research code, and it
grows and improves through the people who use it, report problems with it, and extend it. Every kind of contribution
is appreciated, whether that is a detailed bug report, a question that uncovers something unclear in the
documentation, a small fix, or a substantial new feature. You do not need to be an expert in the underlying physics or
in the entire code base to help out, and if anything below is unclear or feels like a hurdle, please do not hesitate to
get in touch by [e-mail](mailto:julian.peil@tuwien.ac.at). We are happy to help you get started.

> The same information is also available in the [online documentation](https://dgamore.readthedocs.io/en/latest/contributing.html).

## Ways to contribute

There are many ways to make `DGAmore` better, and not all of them involve writing code:

- **Report a bug.** If something does not behave the way you expect, letting us know is already a valuable
  contribution.
- **Suggest a feature.** Ideas for new functionality or for making existing functionality more convenient are always
  welcome.
- **Improve the documentation.** Clarifications, corrections, and examples help the next person who comes along.
- **Contribute code.** Fixes and new features are very welcome, from one-line corrections to larger additions.

## Reporting bugs and requesting features

If you run into a bug, please [open an issue](https://github.com/Julpe/DGAmore/issues) and describe what happened in
as much detail as you can. A short, self-contained example that reproduces the problem is enormously helpful, as it
lets us understand and fix the issue much more quickly, but please open an issue even if you cannot provide one: a
clear description of what you did, what you expected, and what happened instead already goes a long way.

Feature requests and ideas for improving the code are just as welcome. When you open an issue, it helps a lot if you
add an appropriate label (such as *bug* or *feature request*), so that the open items are easy to keep track of. If you
are not sure how to do that, no problem at all, just describe what you have in mind and we will sort out the details
together.

We try to look at critical bugs as soon as we can. Feature requests and smaller issues that do not affect the overall
functionality are noted and considered for future development, so even if something is not picked up right away, it is
not forgotten.

## Setting up a development environment

`DGAmore` requires Python 3.12 or newer, together with `mpich` and `mpi4py`. Once you have
[installed the dependencies](https://dgamore.readthedocs.io/en/latest/installation.html), the easiest way to work on
the code is to fork the repository, clone your fork, and install the package in editable mode. Editable mode means
that your changes take effect immediately, without having to reinstall the package every time:

```bash
git clone https://github.com/<your-username>/DGAmore.git
cd DGAmore
pip install -e .
```

From here you are ready to make changes, run the code, and run the tests.

## Submitting changes

When you would like to contribute code, the following workflow keeps things smooth for everyone:

1. **Create a branch** on your fork for your changes, so that your work is easy to follow and review.
2. **Write your code and add tests for it.** New functionality should come with tests that cover it, and existing
   functionality should keep working. The test suite lives in the `tests/` directory, and `tests/conftest.py`
   provides a number of fixtures (including an in-process fake MPI runtime) that make it easier to write tests without
   a full MPI setup.
3. **Format your changes** with [Black](https://black.readthedocs.io), using a line length of 120 characters. A plain
   run picks up the project configuration automatically:
   ```bash
   black .
   ```
4. **Run the tests locally** before opening your pull request, so that any problems surface early:
   ```bash
   pytest tests                       # fast suite (skips tests marked slow)
   pytest tests --runslow             # full suite, as run in CI
   ```
   The coverage run additionally needs `pytest-cov`, which is not part of the runtime dependencies:
   ```bash
   pip install pytest-cov
   pytest tests --runslow --cov=dgamore --cov-report=term-missing --cov-fail-under=85   # coverage, as CI runs it
   ```
5. **Open a pull request** against the `main` branch, with a short description of what you changed and why. If your
   pull request is related to an existing issue, mentioning it helps connect the two.

A continuous integration pipeline runs on every pull request. Independent workflows check that the code is
Black-formatted, build the documentation, spell-check the sources, and run the full test suite across Python 3.12 to
3.14 on both Linux and macOS. This is there to catch regressions, not to be a
gatekeeper, so please do not worry if something turns red on the first try; it is a normal part of the process, and we
are glad to help you get it passing. The pipeline also requires the overall test coverage to stay at **at least 85%**,
and the build fails if it drops below that threshold. Beyond the overall figure, the new or changed code in a pull
request (the *patch*) must itself be covered to **at least 85%**, so please add tests for what you write rather than
relying on the rest of the code base to carry the average.

## Coding style

To keep the code base consistent and easy to read, it helps to stay close to the style that is already there. In
practice that means keeping the `numpy` and physics notation used throughout the code (the variable names mirror the
author's [Master's thesis](https://doi.org/10.34726/hss.2025.130528)), documenting public classes and methods, using
type hints, and applying the Black formatting described above. It is also easier to review changes that stay focused on
one thing, so where possible please avoid reformatting or restructuring unrelated code in the same pull request.

All prose and identifiers use **American English** (`normalize`, `color`, `behavior`, `parallelization`), in the code,
in docstrings and comments, and in the documentation. The spell-checking job in CI runs with the `en-us` locale, so
British spellings are reported as typos.

Formulas in docstrings, comments and documentation follow one notation for momenta and frequencies:

- An **upright** `\mathrm{q}` and `\mathrm{k}` are the compound momentum-frequency indices
  `\mathrm{q} = (\mathbf{q}, \omega)` and `\mathrm{k} = (\mathbf{k}, \nu)`. Lattice quantities carry them as a
  superscript, and the frequency they already contain is not repeated: write `\chi^{\mathrm{q}\nu}_{0;1234}` (one free
  fermionic index left over), `G^{\mathrm{k}}_{12}` and `\Sigma^{\mathrm{k}}_{12}`. A sum over such an index,
  `\sum_{\mathrm{q}}`, runs over momentum and frequency together.
- A **bold** `\mathbf{q}` and `\mathbf{k}` are momentum vectors alone. Use them for quantities without any frequency
  dependence (`\varepsilon_{12}(\mathbf{k})`, `H(\mathbf{k})`, `V^{\mathbf{q}}_{1234}`), for sums that run over
  momentum only (`\sum_{\mathbf{k}}`), and for the momentum counts `n_{\mathbf{q}}` and `n_{\mathbf{k}}`.
- Counts are a lowercase `n` with the counted object as the subscript: `n_{\mathbf{q}}`, `n_{\mathbf{k}}`,
  `n_{\mathrm{o}}` (orbitals), `n_{\mathbf{q}}^{\mathrm{irr}}` (momenta in the irreducible zone).
- Subscripts and superscripts that are words rather than indices are upright: `E_{\mathrm{kin}}`,
  `\chi^{\mathrm{phys};\mathrm{q}}_{r}`, `\mathbf{k}_{\mathrm{rep}}`.

The orbital-index convention (free indices `1234`, summed indices `a, b, c, ...` in order of first appearance) is
described in the docstrings that use it.

## A final word

If you are ever unsure about any of this, about how to set things up, how to label an issue, or whether an idea is
worth pursuing, please just reach out by [e-mail](mailto:julian.peil@tuwien.ac.at). Questions are always welcome, and
helping new contributors get going is part of what keeps the project alive. Thank you again for contributing to
`DGAmore`.

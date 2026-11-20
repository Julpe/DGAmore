# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import os
import subprocess
import sys

import dgamore


def test_pipeline_modules_import_no_plotting_or_table_libraries():
    """The pipeline modules import without loading matplotlib, pandas, mini_pole, kneed or readline."""
    code = (
        "import sys, dgamore.nonlocal_sde, dgamore.eliashberg_solver, dgamore.self_energy, dgamore.hamiltonian, "
        "dgamore.dga_io; print(sorted(m for m in ('matplotlib', 'pandas', 'mini_pole', 'kneed', 'readline') "
        "if m in sys.modules))"
    )
    # a fresh interpreter, since this test session has imported these libraries already
    env = {**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(dgamore.__file__))}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"

# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import pickle

import pytest

import dgamore.config as config


def test_frozen_section_rejects_writes_and_unfreeze_restores_them():
    """After freeze() a section raises on any attribute assignment and unfreeze() makes it writable again."""
    config.freeze()
    with pytest.raises(AttributeError, match="frozen"):
        config.box.niv_core = 5
    config.unfreeze()
    config.box.niv_core = 5
    assert config.box.niv_core == 5


def test_freeze_covers_every_section_except_the_runtime_sys_section():
    """freeze() latches every configuration section while the runtime-state sys section stays writable."""
    config.freeze()
    for section in config._frozen_sections():
        with pytest.raises(AttributeError, match="frozen"):
            section.some_attribute = 1
    config.sys.mu = 0.7
    assert config.sys.mu == 0.7
    config.unfreeze()


def test_freeze_reaches_nested_sections():
    """Freezing recurses into nested sections: lattice.interaction is latched and released together with its parent."""
    config.freeze()
    with pytest.raises(AttributeError, match="frozen"):
        config.lattice.interaction.udd = 3.0
    config.unfreeze()
    config.lattice.interaction.udd = 3.0
    assert config.lattice.interaction.udd == 3.0


def test_reinstantiated_section_starts_writable():
    """A freshly constructed section is always writable, so the per-test singleton reset never leaks a frozen state."""
    config.freeze()
    fresh = config.BoxConfig()
    fresh.niv_core = 3
    assert fresh.niv_core == 3
    config.unfreeze()


def test_pickled_frozen_section_stays_frozen():
    """A frozen section survives a pickle round trip (the MPI broadcast path) with the latch intact."""
    section = config.BoxConfig()
    section.freeze()
    clone = pickle.loads(pickle.dumps(section))
    with pytest.raises(AttributeError, match="frozen"):
        clone.niv_core = 1

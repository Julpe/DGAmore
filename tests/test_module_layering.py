# SPDX-FileCopyrightText: 2025-2026 Julian Peil <julian.peil@tuwien.ac.at>
# SPDX-License-Identifier: MIT
#
# DGAmore - Multi-Orbital Ladder Dynamical Vertex Approximation (LDGA) &
#           Eliashberg Equation Solver for Strongly Correlated Electron Systems

import ast
import fnmatch
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dgamore"
MODULES = {path.stem for path in PACKAGE.glob("*.py") if path.stem != "__init__"}

KNOWN_PRIVATE_USES = set()
KNOWN_CYCLES = []
FORBIDDEN_EDGES = {
    ("sigma_jacobian", "nonlocal_sde"),
    ("eliashberg_solver", "nonlocal_sde"),
    ("sde_kernels", "nonlocal_sde"),
    ("sde_kernels", "sigma_jacobian"),
    ("sde_kernels", "eliashberg_solver"),
    ("mixing", "nonlocal_sde"),
    ("gap_classification", "eliashberg_solver"),
    ("gap_classification", "config"),
    ("gap_classification", "mpi_utils"),
    ("pairing_vertex", "eliashberg_solver"),
}
STAGE_FILE_NAME = re.compile(
    r"[a-z0-9_]+_(dens|magn|\{\})_loc(\.npy)?|[a-z0-9_{}]+_rank_\{\}(\.npy)?|f_dc_loc(\.npy)?"
    r"|\.aborting|giwk_core_rank_0|chi_phys_q_(dens|magn|\{\})(\.npy)?|g_latt_dmft(\.npy)?|g_dmft(\.npy)?"
    r"|sigma_dga(\.npy)?"
)


def _scanned_modules() -> dict[str, ast.Module]:
    """Returns the parsed source of every package module outside the coverage omit list, keyed by module name."""
    omit = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["coverage"]["run"]["omit"]
    return {
        path.stem: ast.parse(path.read_text())
        for path in sorted(PACKAGE.glob("*.py"))
        if path.stem != "__init__" and not any(fnmatch.fnmatch(str(path), pattern) for pattern in omit)
    }


def _imports(tree: ast.Module) -> tuple[dict[str, str], set[tuple[str, str]]]:
    """Returns the package-module aliases a module imports and the ``(owner, name)`` pairs it imports by name."""
    aliases, names = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "dgamore" and len(parts) == 2 and parts[1] in MODULES:
                    aliases[alias.asname or alias.name] = parts[1]
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "dgamore":
            parts = node.module.split(".")
            for alias in node.names:
                if len(parts) == 1 and alias.name in MODULES:
                    aliases[alias.asname or alias.name] = alias.name
                elif len(parts) == 2 and parts[1] in MODULES:
                    names.add((parts[1], alias.name))
    return aliases, names


def _import_graph() -> dict[str, set[str]]:
    """Returns, for every scanned module, the package modules it imports at module level or inside a function."""
    graph = {}
    for module, tree in _scanned_modules().items():
        aliases, names = _imports(tree)
        graph[module] = (set(aliases.values()) | {owner for owner, _ in names}) - {module}
    return graph


def _reachable(graph: dict[str, set[str]], start: str) -> set[str]:
    """Returns every module reachable from ``start`` along one or more import edges."""
    seen, stack = set(), [start]
    while stack:
        for target in graph.get(stack.pop(), ()):
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


def test_no_module_uses_a_private_name_of_another():
    """Cross-module reads of underscore names are exactly the known ones."""
    found = set()
    for module, tree in _scanned_modules().items():
        aliases, names = _imports(tree)
        found |= {(module, owner, name) for owner, name in names if name.startswith("_")}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                if node.attr.startswith("_") and not node.attr.startswith("__"):
                    found.add((module, aliases[node.value.id], node.attr))
    assert found == KNOWN_PRIVATE_USES


def test_import_graph_cycles_are_the_known_ones():
    """The import cycles among the package modules are exactly the known ones."""
    graph = _import_graph()
    reach = {module: _reachable(graph, module) for module in graph}
    cycles = {
        frozenset(m for m in graph if m in reach[start] and start in reach[m])
        for start in graph
        if start in reach[start]
    }
    assert sorted(sorted(cycle) for cycle in cycles) == sorted(sorted(cycle) for cycle in KNOWN_CYCLES)


def test_stage_file_names_are_spelled_only_in_output_files():
    """No module but output_files spells a file name that one stage writes and another reads."""
    found = []
    for module, tree in _scanned_modules().items():
        if module == "output_files":
            continue
        skip = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
                skip.add(id(body[0].value))
            if isinstance(node, ast.JoinedStr):
                skip |= {id(value) for value in node.values}
        for node in ast.walk(tree):
            if id(node) in skip:
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
            elif isinstance(node, ast.JoinedStr):
                text = "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
            else:
                continue
            if STAGE_FILE_NAME.fullmatch(text):
                found.append(f"{module}:{node.lineno}: {text}")
    assert found == []


def test_forbidden_imports_are_absent():
    """No module imports a module it must stay independent of."""
    graph = _import_graph()
    present = {(importer, imported) for importer, imported in FORBIDDEN_EDGES if imported in graph.get(importer, ())}
    assert present == set()

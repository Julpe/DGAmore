import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "DGAmore"
author = "Julian Peil"
copyright = "2026, Julian Peil"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True

# Native/optional dependencies that need not (and on a docs runner cannot easily) be installed to build the docs.
# autodoc only needs to *import* the package; mocking these keeps the docs build light and runner-agnostic.
autodoc_mock_imports = ["mpi4py", "cupy"]

# Cross-reference the standard library and the scientific stack.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

autodoc_typehints = "both"
autodoc_class_signature = "mixed"
autodoc_typehints_format = "short"
autodoc_class_attributes = False
# autodoc_member_order = "bysource"

python_use_unqualified_type_names = True

napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "pydata_sphinx_theme"  # "alabaster"  # "furo"
# Serve the logos directly from the repository's top-level ``logos/`` folder (copied into the build output only,
# never duplicated in the docs tree).
html_static_path = ["../logos"]
html_title = "DGAmore"

# Light/dark-aware navbar logo: the light-background logo is shown in light mode, the dark one in dark mode.
html_theme_options = {
    "show_nav_level": 1,
    "navbar_align": "content",
    # Move the section links out of the top header; the full navigation lives permanently in the left sidebar
    # (see the ``_templates/sidebar-nav-bs.html`` override, which renders the toctree from the root).
    "navbar_center": [],
    "navigation_with_keys": True,
    "logo": {
        "image_light": "_static/DGAmore_light.png",
        "image_dark": "_static/DGAmore_dark.png",
        "alt_text": "DGAmore",
    },
}


def _skip_class_data_attributes(app, what, name, obj, skip, options):
    """Hide plain *class* data attributes (incl. enum members) from autodoc to avoid duplicate/ugly entries.

    The ``*Config`` classes document their fields with ``:ivar:`` and the ``Enum`` classes document their members
    with ``:cvar:`` in the class docstring. With ``members: True``, autodoc would *additionally* emit a standalone
    entry for each attribute/enum member, listing everything twice (and rendering enum members as ``NAME = 'value'``).
    This filters out class data attributes so the ``:ivar:``/``:cvar:`` lists are the single source. Methods and
    properties are always kept, as is module-level data (constants, singletons).
    """
    if skip:
        return None
    if inspect.isroutine(obj) or isinstance(obj, (property, staticmethod, classmethod)):
        return None  # keep methods, functions and properties
    if what == "class":
        return True  # hide class data attributes and enum members (documented via :ivar:/:cvar:)
    return None


def setup(app):
    """Register the autodoc member filter (see :func:`_skip_class_data_attributes`)."""
    app.connect("autodoc-skip-member", _skip_class_data_attributes)

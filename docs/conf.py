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

html_theme_options = {
    "show_nav_level": 2,
    "navbar_align": "content",
    "navigation_with_keys": True,
}

html_theme = "pydata_sphinx_theme"  # "alabaster"  # "furo"

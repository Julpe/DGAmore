import inspect
import os
import sys
import typing

from sphinx.ext.autodoc.mock import _MockObject

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
autodoc_mock_imports = ["mpi4py", "psutil"]

# Mocked attributes are not classes, so a union annotation over them (``MPI.Comm | None``) raises a TypeError while
# autodoc imports the package. Teaching the mock the union operators keeps such annotations importable and rendered.
_MockObject.__or__ = lambda self, other: typing.Union[self, other]
_MockObject.__ror__ = lambda self, other: typing.Union[other, self]

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
# ``_static`` holds the docs' own assets (custom.css); ``../logos`` serves the logos directly from the repository's
# top-level ``logos/`` folder. Both are merged into the build output's ``_static/`` directory.
html_static_path = ["_static", "../logos"]
html_css_files = ["custom.css"]
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
        "image_light": "_static/dgamore-lockup-tagline-light.svg",
        "image_dark": "_static/dgamore-lockup-tagline-dark.svg",
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


def _keep_sidebar_on_root(app, pagename, templatename, context, doctree):
    """Keep the navigation sidebar on the root/landing page.

    The pydata theme strips the sidebar ``TocTree`` (and so collapses the primary sidebar, rendering the toctree
    inline in the page body) whenever ``suppress_sidebar_toctree()`` is truthy. For the root document that helper
    always returns ``True``, because the root page has no ancestor section to anchor a second-level TocTree on.
    We override that single context callable for the root page only, so the full navigation - rendered from the
    document root by the ``_templates/sidebar-nav-bs.html`` override (``startdepth=0``) - stays in the left sidebar
    on the welcome page just like on every other page.
    """
    if pagename == app.config.root_doc:
        context["suppress_sidebar_toctree"] = lambda *args, **kwargs: False


def setup(app):
    """Register the autodoc member filter (see :func:`_skip_class_data_attributes`) and the root-page sidebar fix."""
    app.connect("autodoc-skip-member", _skip_class_data_attributes)
    # Run after the theme's own ``add_toctree_functions`` (priority 500) so we override the context callable it sets.
    app.connect("html-page-context", _keep_sidebar_on_root, priority=900)

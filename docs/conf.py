"""Sphinx configuration for the artisatomic documentation."""

import os
import sys
from importlib.metadata import version as get_version
from pathlib import Path

project = "artisatomic"
author = "Luke J. Shingles and the ARTIS Collaboration"
copyright = f"%Y, {author}"  # ruff: ignore[builtin-variable-shadowing]
release = get_version("artisatomic")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinxcontrib.programoutput",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]

# The docstrings put names in single backticks, as in Markdown.
default_role = "code"

myst_heading_anchors = 3

autosummary_generate = True
autodoc_default_options = {"members": True, "member-order": "bysource"}
autodoc_typehints = "description"
# Only the hdf5_to_parquet module needs h5py, and the docs group does not install it.
autodoc_mock_imports = ["h5py"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
}

html_theme = "furo"
html_title = f"artisatomic {release}"

# The program-output directive starts the console scripts. The Python that runs Sphinx can be in a
# virtual environment that is not on PATH, and argparse wraps the help text to the terminal width.
os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
os.environ["COLUMNS"] = "80"

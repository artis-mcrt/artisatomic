"""Sphinx configuration for the artisatomic documentation."""

import os
import sys
from importlib.metadata import version as get_version
from pathlib import Path

project = "artisatomic"
author = "Luke J. Shingles and the ARTIS Collaboration"
# the copyright holder of LICENSE
copyright = "2016-%Y, the ARTIS Collaboration"  # ruff: ignore[builtin-variable-shadowing]
release = get_version("artisatomic")
version = release

extensions = [
    "myst_parser",
    "sphinxcontrib.programoutput",
]

# All pages are MyST Markdown. Sphinx reads no docstring, so the docs contain no reStructuredText.
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]

myst_heading_anchors = 3

html_theme = "furo"
html_title = f"artisatomic {release}"

# The program-output directive starts the console scripts. The Python that runs Sphinx can be in a
# virtual environment that is not on PATH, and argparse wraps the help text to the terminal width.
os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
os.environ["COLUMNS"] = "80"

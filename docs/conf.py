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
    "sphinx_copybutton",
    "sphinxcontrib.programoutput",
]

# All pages are MyST Markdown. Sphinx reads no docstring, so the docs contain no reStructuredText.
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]

myst_heading_anchors = 3

# The canonical link of a page names the version that a search engine must show.
# The versions of the site then do not compete with each other.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# The "Edit this page" link of each page points at the branch or the tag that Read the Docs builds.
# A build of a pull request gives the number of the request, which is not a branch.
# The link then points at the main branch.
source_branch = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")
if os.environ.get("READTHEDOCS_VERSION_TYPE") == "external":
    source_branch = "main"

html_theme = "furo"
html_title = f"artisatomic {release}"
html_theme_options = {
    "source_repository": "https://github.com/artis-mcrt/artisatomic/",
    "source_branch": source_branch,
    "source_directory": "docs/",
}

# The program-output directive starts the console scripts.
# The Python that runs Sphinx can be in a virtual environment that is not on PATH.
# argparse wraps the help text to the terminal width.
os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
os.environ["COLUMNS"] = "80"

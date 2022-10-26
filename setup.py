#!/usr/bin/env python3
# mypy: ignore-errors
"""Atomic data tools for the ARTIS 3D supernova radiative transfer code."""
import datetime
import os
import sys
from pathlib import Path

from setuptools import find_packages
from setuptools import setup
from setuptools.command.test import test as TestCommand
from setuptools_scm import get_version

setup(
    name="artisatomic",
    version=get_version(),
    use_scm_version=True,
    author="Luke Shingles",
    author_email="luke.shingles@gmail.com",
    packages=find_packages(),
    url="https://github.com/artis-mcrt/artisatomic",
    description="Tools to create an atomic database for use with ARTIS.",
    long_description=(Path(__file__).absolute().parent / "README.md").open("rt").read(),
    long_description_content_type="text/markdown",
    install_requires=(Path(__file__).absolute().parent / "requirements.txt").open("rt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "makeartisatomicfiles = artisatomic:main",
            "makerecombratefile = artisatomic.makerecombratefile:main",
            "convertartisatomictopythonrt = artisatomic.converttopythonrt:main",
        ]
    },
    python_requires=">==3.9",
    setup_requires=["psutil>=5.9.0", "setuptools>=45", "setuptools_scm[toml]>=6.2", "wheel"],
    tests_require=["pytest", "pytest-runner", "pytest-cov"],
)

#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Build an ARTIS atomic database from published atomic data sets.

Each data source has its own read*.py module; this package selects a handler per ion, calls it,
and writes the combined result to adata.txt, transitiondata.txt and phixsdata_v2.txt.

The implementation lives in submodules (base, levelnames, ionhandlers, phixs, iondata, output,
cli), and every caller imports from the submodule that defines the name. This module therefore
re-exports nothing: the package has no external API callers, and a re-export here would be a
second name for something that already has one. Python binds a submodule to the package when
something imports it, so `from artisatomic import readqubdata` needs no line here either.
"""

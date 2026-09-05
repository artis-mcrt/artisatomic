"""Build an ARTIS atomic database from published atomic data sets.

Each data source has its own read*.py module. This package selects a handler for each ion, calls
it, and writes the combined result to adata.txt, transitiondata.txt and phixsdata_v2.txt.

The implementation lives in the submodules (base, levelnames, ionhandlers, phixs, iondata,
output, cli). Every caller imports from the submodule that defines the name. This module
therefore re-exports nothing. The package has no external API callers, and a re-export here
would be a second name for a name that already exists. Python binds a submodule to the package
when any module imports it. Therefore `from artisatomic import readqubdata` needs no line here.
"""

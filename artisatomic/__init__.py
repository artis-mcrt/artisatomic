#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Build an ARTIS atomic database from published atomic data sets.

Each data source has its own read*.py module; this package selects a handler per ion, calls it,
and writes the combined result to adata.txt, transitiondata.txt and phixsdata_v2.txt.

The implementation lives in submodules (base, levelnames, ionhandlers, phixs, iondata, output,
cli). This module re-exports names so the internal command-line scripts and the tests can use the
flat artisatomic.name interface. The package has no external API callers, so a name gets a
re-export only when an internal script or a test requires it. Each name is imported under its own
name (`import x as x`), which is what marks it as a re-export rather than an incidental import.
Submodules and readers import base-level helpers and constants with
`from artisatomic.base import ...` (safe from circularity, since base imports nothing from the
package).
"""

from artisatomic import groundstatesonlynist as groundstatesonlynist
from artisatomic import readboyledata as readboyledata
from artisatomic import readdreamdata as readdreamdata
from artisatomic import readfacdata as readfacdata
from artisatomic import readfloers25data as readfloers25data
from artisatomic import readhillierdata as readhillierdata
from artisatomic import readkuruczdata as readkuruczdata
from artisatomic import readlisbondata as readlisbondata
from artisatomic import readmonsdata as readmonsdata
from artisatomic import readqubdata as readqubdata
from artisatomic import readtanakajpltdata as readtanakajpltdata
from artisatomic.base import compression_extensions as compression_extensions
from artisatomic.base import elsymbols as elsymbols
from artisatomic.base import empty_transitions_schema as empty_transitions_schema
from artisatomic.base import find_file_check_extension as find_file_check_extension
from artisatomic.base import get_nist_ionization_energies_ev as get_nist_ionization_energies_ev
from artisatomic.base import hc_in_ev_cm as hc_in_ev_cm
from artisatomic.base import isfloat as isfloat
from artisatomic.base import levelid_of_fileindex_map as levelid_of_fileindex_map
from artisatomic.base import leveltuples_to_pldataframe as leveltuples_to_pldataframe
from artisatomic.base import log_and_print as log_and_print
from artisatomic.base import parallel_map as parallel_map
from artisatomic.base import path_for_log as path_for_log
from artisatomic.base import PYDIR as PYDIR
from artisatomic.base import resolve_transition_levelids as resolve_transition_levelids
from artisatomic.base import roman_numerals as roman_numerals
from artisatomic.base import split_element_ionstage_str as split_element_ionstage_str
from artisatomic.base import xopen_check_extension as xopen_check_extension
from artisatomic.cli import main as main
from artisatomic.ionhandlers import add_handler_if_not_set as add_handler_if_not_set
from artisatomic.ionhandlers import parse_ion_handlers as parse_ion_handlers
from artisatomic.levelnames import get_config_parity as get_config_parity
from artisatomic.levelnames import has_merged_orbital as has_merged_orbital
from artisatomic.levelnames import interpret_configuration as interpret_configuration
from artisatomic.output import add_level_ids_forbidden as add_level_ids_forbidden
from artisatomic.output import write_adata as write_adata
from artisatomic.output import write_phixs_data as write_phixs_data
from artisatomic.phixs import match_hydrogenic_phixs as match_hydrogenic_phixs
from artisatomic.phixs import reduce_phixs_tables as reduce_phixs_tables
from artisatomic.phixs import reduce_phixs_tables_worker as reduce_phixs_tables_worker

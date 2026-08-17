#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Build an ARTIS atomic database from published atomic data sets.

Each data source has its own read*.py module; this package selects a handler per ion, calls it,
and writes the combined result to adata.txt, transitiondata.txt and phixsdata_v2.txt.

The implementation lives in submodules (base, levelnames, ionhandlers, phixs, iondata, output,
cli); everything is re-exported here so readers and scripts can keep using the flat
artisatomic.name interface. Each name is imported under its own name (`import x as x`), which is
what marks it as a re-export rather than an incidental import. Submodules and readers import
base-level helpers and constants with `from artisatomic.base import ...` (safe from circularity,
since base imports nothing from the package); the flat `artisatomic.name` attribute access
remains for facade-level names.
"""

from artisatomic import groundstatesonlynist as groundstatesonlynist
from artisatomic import readboyledata as readboyledata
from artisatomic import readdreamdata as readdreamdata
from artisatomic import readfacdata as readfacdata
from artisatomic import readfloers25data as readfloers25data
from artisatomic import readhillierdata as readhillierdata
from artisatomic import readkuruczdata as readkuruczdata
from artisatomic import readlisbondata as readlisbondata
from artisatomic import readqubdata as readqubdata
from artisatomic import readtanakajpltdata as readtanakajpltdata
from artisatomic.base import atomic_weights as atomic_weights
from artisatomic.base import compression_extensions as compression_extensions
from artisatomic.base import elsymbols as elsymbols
from artisatomic.base import empty_transitions_schema as empty_transitions_schema
from artisatomic.base import find_file_check_extension as find_file_check_extension
from artisatomic.base import get_nist_ionization_energies_ev as get_nist_ionization_energies_ev
from artisatomic.base import h_in_ev_seconds as h_in_ev_seconds
from artisatomic.base import hc_in_ev_angstrom as hc_in_ev_angstrom
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
from artisatomic.base import ryd_to_ev as ryd_to_ev
from artisatomic.base import split_element_ionstage_str as split_element_ionstage_str
from artisatomic.base import xopen_check_extension as xopen_check_extension
from artisatomic.cli import main as main
from artisatomic.cli import process_files as process_files
from artisatomic.iondata import IonData as IonData
from artisatomic.iondata import read_ion_data as read_ion_data
from artisatomic.iondata import resolve_photoion_targetfractions as resolve_photoion_targetfractions
from artisatomic.iondata import simple_handler_readers as simple_handler_readers
from artisatomic.ionhandlers import add_handler_if_not_set as add_handler_if_not_set
from artisatomic.ionhandlers import drop_handlers as drop_handlers
from artisatomic.ionhandlers import get_ion_handlers as get_ion_handlers
from artisatomic.ionhandlers import parse_ion_handlers as parse_ion_handlers
from artisatomic.ionhandlers import sort_ion_handlers as sort_ion_handlers
from artisatomic.levelnames import get_config_parity as get_config_parity
from artisatomic.levelnames import get_parity_from_config as get_parity_from_config
from artisatomic.levelnames import has_merged_orbital as has_merged_orbital
from artisatomic.levelnames import interpret_configuration as interpret_configuration
from artisatomic.output import add_level_ids_forbidden as add_level_ids_forbidden
from artisatomic.output import clear_files as clear_files
from artisatomic.output import write_adata as write_adata
from artisatomic.output import write_compositionfile as write_compositionfile
from artisatomic.output import write_output_files as write_output_files
from artisatomic.output import write_phixs_data as write_phixs_data
from artisatomic.output import write_transition_data as write_transition_data
from artisatomic.phixs import match_hydrogenic_phixs as match_hydrogenic_phixs
from artisatomic.phixs import reduce_phixs_tables as reduce_phixs_tables
from artisatomic.phixs import reduce_phixs_tables_worker as reduce_phixs_tables_worker

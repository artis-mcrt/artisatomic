#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Build an ARTIS atomic database from published atomic data sets.

Each data source has its own read*.py module; this package selects a handler per ion, calls it,
and writes the combined result to adata.txt, transitiondata.txt and phixsdata_v2.txt.

The implementation lives in submodules (base, levelnames, ionhandlers, phixs, iondata, output,
cli); everything is re-exported here so readers and scripts can keep using the flat
artisatomic.name interface.
"""

from artistools.constants import h_ev_s as h_in_ev_seconds

from artisatomic import groundstatesonlynist
from artisatomic import readboyledata
from artisatomic import readdreamdata
from artisatomic import readfacdata
from artisatomic import readfloers25data
from artisatomic import readhillierdata
from artisatomic import readkuruczdata
from artisatomic import readnahardata
from artisatomic import readqubdata
from artisatomic import readtanakajpltdata
from artisatomic.base import atomic_weights
from artisatomic.base import compression_extensions
from artisatomic.base import elsymbols
from artisatomic.base import empty_transitions_schema
from artisatomic.base import find_file_check_extension
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import hc_in_ev_angstrom
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import isfloat
from artisatomic.base import leveltuples_to_pldataframe
from artisatomic.base import log_and_print
from artisatomic.base import parallel_map
from artisatomic.base import path_for_log
from artisatomic.base import PYDIR
from artisatomic.base import roman_numerals
from artisatomic.base import ryd_to_ev
from artisatomic.base import split_element_ionstage_str
from artisatomic.base import xopen_check_extension
from artisatomic.cli import main
from artisatomic.cli import process_files
from artisatomic.iondata import IonData
from artisatomic.iondata import read_ion_data
from artisatomic.iondata import simple_handler_readers
from artisatomic.ionhandlers import add_handler_if_not_set
from artisatomic.ionhandlers import drop_handlers
from artisatomic.ionhandlers import get_default_handler
from artisatomic.ionhandlers import get_ion_handlers
from artisatomic.ionhandlers import get_ion_stage
from artisatomic.ionhandlers import sort_ion_handlers
from artisatomic.levelnames import get_parity_from_config
from artisatomic.levelnames import interpret_configuration
from artisatomic.output import add_level_ids_forbidden
from artisatomic.output import clear_files
from artisatomic.output import write_adata
from artisatomic.output import write_compositionfile
from artisatomic.output import write_output_files
from artisatomic.output import write_phixs_data
from artisatomic.output import write_transition_data
from artisatomic.phixs import match_hydrogenic_phixs
from artisatomic.phixs import reduce_phixs_tables
from artisatomic.phixs import reduce_phixs_tables_worker

__all__ = [
    "PYDIR",
    "IonData",
    "add_handler_if_not_set",
    "add_level_ids_forbidden",
    "atomic_weights",
    "clear_files",
    "compression_extensions",
    "drop_handlers",
    "elsymbols",
    "empty_transitions_schema",
    "find_file_check_extension",
    "get_default_handler",
    "get_ion_handlers",
    "get_ion_stage",
    "get_nist_ionization_energies_ev",
    "get_parity_from_config",
    "groundstatesonlynist",
    "h_in_ev_seconds",
    "hc_in_ev_angstrom",
    "hc_in_ev_cm",
    "interpret_configuration",
    "isfloat",
    "leveltuples_to_pldataframe",
    "log_and_print",
    "main",
    "match_hydrogenic_phixs",
    "parallel_map",
    "path_for_log",
    "process_files",
    "read_ion_data",
    "readboyledata",
    "readdreamdata",
    "readfacdata",
    "readfloers25data",
    "readhillierdata",
    "readkuruczdata",
    "readnahardata",
    "readqubdata",
    "readtanakajpltdata",
    "reduce_phixs_tables",
    "reduce_phixs_tables_worker",
    "roman_numerals",
    "ryd_to_ev",
    "simple_handler_readers",
    "sort_ion_handlers",
    "split_element_ionstage_str",
    "write_adata",
    "write_compositionfile",
    "write_output_files",
    "write_phixs_data",
    "write_transition_data",
    "xopen_check_extension",
]

if __name__ == "__main__":
    main()

"""Read a single ion's levels, transitions, and photoionisation data from its source dataset."""

import argparse
import typing as t
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

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
from artisatomic.base import elsymbols
from artisatomic.base import empty_transitions_schema
from artisatomic.base import leveltuples_to_pldataframe
from artisatomic.base import log_and_print
from artisatomic.base import path_for_log
from artisatomic.base import roman_numerals
from artisatomic.base import ryd_to_ev
from artisatomic.ionhandlers import get_default_handler
from artisatomic.phixs import match_hydrogenic_phixs

# import artisatomic.readlisbondata as readlisbondata


class IonData(t.NamedTuple):
    """Levels, transitions, and photoionisation data for a single ion, read from one of the source datasets."""

    ion_stage: int
    handler: str
    ionization_energy_ev: float
    dfenergylevels: pl.DataFrame
    dftransitions: pl.DataFrame
    transition_count_of_level_name: dict[str, int]
    upsilondict: dict[tuple[int, int], float]
    # core state id n is at index n - 1 ([] if unavailable)
    nahar_core_states: list[readnahardata.NaharCoreState]
    # keys are (2S+1, L, parity, index in symmetry), values are strings of electron configuration
    nahar_configurations: dict[tuple[int, int, int, int], str]
    # None where a level has no photoionisation data (and None entirely if none was read)
    hillier_photoion_targetconfigs: list[list[tuple[str, float]] | None] | None
    photoionization_crosssections: npt.NDArray[np.float64]  # cross sections in Mb, indexed by level id
    photoionization_targetfractions: list[list[tuple[int, float]]]  # indexed by level id
    photoionization_thresholds_ev: npt.NDArray[np.float64]  # indexed by level id


# handlers whose readers share a common call signature and return
# (ionization_energy_ev, energy_levels, transitions, transition_count_of_level_name)
# with an additional upsilondict for qub_data
simple_handler_readers: dict[str, Callable[..., tuple[t.Any, ...]]] = {
    "boyle": lambda atomic_number, ion_stage, _flog: readboyledata.read_levels_and_transitions(
        atomic_number, ion_stage
    ),
    "kurucz": readkuruczdata.read_levels_and_transitions,
    "dream": readdreamdata.read_levels_and_transitions,  # DREAM database of Z >= 57
    # "lisbon": readlisbondata.read_levels_and_transitions,
    "floers25calib": lambda atomic_number, ion_stage, flog: readfloers25data.read_levels_and_transitions(
        atomic_number, ion_stage, flog, calibrated=True
    ),
    "floers25uncalib": lambda atomic_number, ion_stage, flog: readfloers25data.read_levels_and_transitions(
        atomic_number, ion_stage, flog, calibrated=False
    ),
    "fac": readfacdata.read_levels_and_transitions,  # early version of floers25 calib data
    "tanakajplt": readtanakajpltdata.read_levels_and_transitions,  # Tanaka Japan-Lithuania database of 26 <= Z <= 88
    "gsnist": groundstatesonlynist.read_ground_levels,  # ground states taken from NIST
    "qub_data": readqubdata.read_qub_levels_and_transitions,  # also returns an upsilondict
}


def read_ion_data(
    atomic_number: int, ion_stage_entry: int | tuple[int, str], is_top_ion: bool, args: argparse.Namespace
) -> IonData:
    """Read a single ion's data from its source dataset."""
    if isinstance(ion_stage_entry, int):
        ion_stage = ion_stage_entry
        handler = get_default_handler(atomic_number, ion_stage)
    else:
        ion_stage, handler = ion_stage_entry

    ionization_energy_ev = 0.0
    transition_count_of_level_name: dict[str, int] = {}
    upsilondict: dict[tuple[int, int], float] = {}
    nahar_core_states: list[readnahardata.NaharCoreState] = []
    nahar_configurations: dict[tuple[int, int, int, int], str] = {}
    hillier_photoion_targetconfigs: list[list[tuple[str, float]] | None] | None = None
    # empty until a handler below reads photoionisation data (and left empty for the top ion)
    photoionization_crosssections: npt.NDArray[np.float64] = np.empty((0, args.nphixspoints))  # in Mb
    photoionization_targetfractions: list[list[tuple[int, float]]] = []
    photoionization_thresholds_ev: npt.NDArray[np.float64] = np.empty(0)

    logfilepath = Path(
        args.output_folder, args.output_folder_logs, f"{elsymbols[atomic_number].lower()}{ion_stage:d}.txt"
    )
    with logfilepath.open("w", encoding="utf-8") as flog:
        log_and_print(
            flog,
            f"\n===========> Z={atomic_number} {elsymbols[atomic_number]} {roman_numerals[ion_stage]} input:",
        )
        log_and_print(flog, f"Source handler: {handler}")
        if handler == "qub_cobalt":
            if ion_stage in {3, 4}:  # QUB levels and transitions, or single-level Co IV
                (
                    ionization_energy_ev,
                    energy_levels,
                    transitions,
                    transition_count_of_level_name,
                    upsilondict,
                ) = readqubdata.read_qub_levels_and_transitions(atomic_number, ion_stage, flog)
            else:  # hillier levels and transitions
                # if ion_stage == 2:
                #     upsilondict = readstoreydata.read_storey_2016_upsilondata(flog)
                (
                    ionization_energy_ev,
                    energy_levels,
                    transitions,
                    transition_count_of_level_name,
                ) = readhillierdata.read_levels_and_transitions(atomic_number, ion_stage, flog)

            if not is_top_ion and not args.nophixs:  # don't get cross sections for top ion
                (
                    photoionization_crosssections,
                    photoionization_targetfractions,
                    photoionization_thresholds_ev,
                ) = readqubdata.read_qub_photoionizations(
                    atomic_number, ion_stage, levelcount=len(energy_levels), args=args, flog=flog
                )

        elif handler == "nahar":
            path_nahar_energy_file = f"atomic-data-nahar/{elsymbols[atomic_number].lower()}{ion_stage:d}.en.ls.txt"
            path_nahar_px_file = f"atomic-data-nahar/{elsymbols[atomic_number].lower()}{ion_stage:d}.ptpx.txt"
            (
                nahar_energy_levels,
                nahar_core_states,
                nahar_configurations,
                nahar_ionization_potential_rydberg,
            ) = readnahardata.read_nahar_energy_level_file(path_nahar_energy_file, atomic_number, ion_stage, flog)
            ionization_energy_ev = nahar_ionization_potential_rydberg * ryd_to_ev

            # keys are (2S+1, L, parity, indexinsymmetry), values are lists of
            # (energy in Rydberg, cross section in Mb) tuples
            nahar_phixs_tables: dict[tuple[int, int, int, int], npt.NDArray[np.float64]] = {}
            thresholds_ev_dict: dict[tuple[int, int, int, int], float] = {}
            if not is_top_ion:  # don't get cross sections for top ion
                log_and_print(flog, f"Reading {path_for_log(path_nahar_px_file)}")
                nahar_phixs_tables, thresholds_ev_dict = readnahardata.read_nahar_phixs_tables(
                    path_nahar_px_file, atomic_number, ion_stage, args
                )

            (energy_levels, photoionization_crosssections, photoionization_thresholds_ev) = (
                readnahardata.build_nahar_levels_and_phixs(
                    nahar_energy_levels, nahar_phixs_tables, thresholds_ev_dict, args, flog
                )
            )
            # the Nahar data set gives no bound-bound transitions
            transitions = pl.DataFrame(schema=empty_transitions_schema)

        elif handler == "cmfgen":  # Hillier CMFGEN data only
            (
                ionization_energy_ev,
                energy_levels,
                transitions,
                transition_count_of_level_name,
            ) = readhillierdata.read_levels_and_transitions(atomic_number, ion_stage, flog)

            if len(upsilondict) == 0:
                upsilondict = readhillierdata.read_coldata(atomic_number, ion_stage, energy_levels, flog, args)

            if not is_top_ion and not args.nophixs:  # don't get cross sections for top ion
                (
                    photoionization_crosssections,
                    hillier_photoion_targetconfigs,
                    photoionization_thresholds_ev,
                ) = readhillierdata.read_phixs_tables(atomic_number, ion_stage, energy_levels, args, flog)
            else:
                hillier_photoion_targetconfigs = None

        elif handler in simple_handler_readers:
            result = simple_handler_readers[handler](atomic_number, ion_stage, flog)
            if len(result) == 5:
                (ionization_energy_ev, energy_levels, transitions, transition_count_of_level_name, upsilondict) = result
            else:
                (ionization_energy_ev, energy_levels, transitions, transition_count_of_level_name) = result

        else:
            raise ValueError(f"Unknown handler: {handler}")

    dfenergylevels = leveltuples_to_pldataframe(energy_levels)

    # the len() == 0 test is what limits the estimate to ions the handler gave nothing for: an ion
    # with even one cross-section table is left alone, so measured data is never replaced. The top
    # ion is excluded because there is no upper ion for it to photoionise to.
    if (
        not is_top_ion
        and not args.nophixs
        and len(photoionization_crosssections) == 0
        and args.nlevels_hydrogenic_for_unknown_phixs > 0
    ):
        (
            photoionization_crosssections,
            photoionization_targetfractions,
            photoionization_thresholds_ev,
        ) = match_hydrogenic_phixs(atomic_number, dfenergylevels, ionization_energy_ev, handler, args)

    dftransitions = transitions if isinstance(transitions, pl.DataFrame) else pl.DataFrame(transitions)

    return IonData(
        ion_stage=ion_stage,
        handler=handler,
        ionization_energy_ev=ionization_energy_ev,
        dfenergylevels=dfenergylevels,
        dftransitions=dftransitions,
        transition_count_of_level_name=transition_count_of_level_name,
        upsilondict=upsilondict,
        nahar_core_states=nahar_core_states,
        nahar_configurations=nahar_configurations,
        hillier_photoion_targetconfigs=hillier_photoion_targetconfigs,
        photoionization_crosssections=photoionization_crosssections,
        photoionization_targetfractions=photoionization_targetfractions,
        photoionization_thresholds_ev=photoionization_thresholds_ev,
    )

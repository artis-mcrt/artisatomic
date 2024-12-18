#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import contextlib
import glob
import itertools
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import typing as t
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import argcomplete
import numpy as np
import pandas as pd
import polars as pl
from scipy import integrate
from scipy import interpolate

from artisatomic import groundstatesonlynist
from artisatomic import readboyledata
from artisatomic import readcarsusdata
from artisatomic import readdreamdata
from artisatomic import readfacdata
from artisatomic import readfloers25data
from artisatomic import readhillierdata
from artisatomic import readnahardata
from artisatomic import readqubdata
from artisatomic import readtanakajpltdata
from artisatomic.manual_matches import hillier_name_replacements
from artisatomic.manual_matches import nahar_configuration_replacements

# import artisatomic.readlisbondata as readlisbondata

PYDIR = Path(__file__).parent.resolve()
atomicdata = pd.read_csv(PYDIR / "atomic_properties.txt", sep=r"\s+", comment="#")
atomicdata = atomicdata.apply(lambda x: x.fillna(x.number / 0.45), axis=1)  # estimate unknown atomic mass as Z / 0.45
elsymbols = ["n", *list(atomicdata["symbol"].values)]
atomic_weights = ["n", *list(atomicdata["mass"].values)]

roman_numerals = (
    "",
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
    "XVII",
    "XVIII",
    "XIX",
    "XX",
)


def get_ion_handlers() -> list[tuple[int, list[int | tuple[int, str]]]]:
    inputhandlersfile = Path("artisatomicionhandlers.json")

    if inputhandlersfile.exists():
        print(f"Reading {inputhandlersfile}")
        return json.load(inputhandlersfile.open(encoding="utf-8"))

    ion_handlers: list[tuple[int, list[int | tuple[int, str]]]] = []
    ion_handlers = [
        (26, [1, 2, 3, 4, 5]),
        (27, [2, 3, 4]),
        (28, [2, 3, 4, 5]),
    ]

    # ion_handlers = [
    #     (2, [(3, "boyle")]),
    #     (38, [(1, "carsus"), (2, "carsus"), (3, "carsus")]),
    #     (39, [(1, "carsus"), (2, "carsus")]),
    #     (40, [(1, "carsus"), (2, "carsus"), (3, "carsus")]),
    #     (70, [(5, "gsnist")]),
    #     (92, [(2, "fac"), (3, "fac")]),
    #     (94, [(2, "fac"), (3, "fac")]),
    # ]

    # include everything we have data for
    # ion_handlers = readhillierdata.extend_ion_list(ion_handlers, maxionstage=5, include_hydrogen=False)
    # ion_handlers = readcarsusdata.extend_ion_list(ion_handlers)
    # ion_handlers = readdreamdata.extend_ion_list(ion_handlers)
    # ion_handlers = readfacdata.extend_ion_list(ion_handlers)
    # ion_handlers = readfloers25data.extend_ion_list(ion_handlers, calibrated=True)
    # ion_handlers = readtanakajpltdata.extend_ion_list(ion_handlers)
    # ion_handlers = groundstatesonlynist.extend_ion_list(ion_handlers)

    return ion_handlers


USE_QUB_COBALT = False

ryd_to_ev = 13.605693122994232
hc_in_ev_cm = 0.0001239841984332003
hc_in_ev_angstrom = 12398.419843320025
h_in_ev_seconds = 4.135667696923859e-15


def drop_handlers(list_ions: list[int | tuple[int, str]]) -> list[int]:
    """Replace [(ion_stage, 'handler1'), (ion_stage2, 'handler2'), ion_stage3] with [ion_stage1, ion_stage2, ion_stage3]."""
    list_out = []
    for ion_stage in list_ions:
        if isinstance(ion_stage, int):
            list_out.append(ion_stage)
        else:
            list_out.append(ion_stage[0])

    return list_out


def add_dummy_zero_level(dflevels: pl.DataFrame) -> pl.DataFrame:
    # keep the zero index as null since we use 1-index level indicies
    anycolname = dflevels.columns[0]
    return pl.concat(
        [
            pl.DataFrame({anycolname: [None]}, schema={anycolname: dflevels.schema[anycolname]}),
            dflevels,
        ],
        how="diagonal",
    )


def leveltuples_to_pldataframe(energy_levels) -> pl.DataFrame:
    if isinstance(energy_levels, pl.DataFrame):
        dflevels = energy_levels
        assert energy_levels["energyabovegsinpercm"].item(0) is None

    else:
        dflevels = add_dummy_zero_level(
            pl.DataFrame(energy_levels[1:]),
        )

    if "levelid" not in dflevels.columns:
        dflevels = dflevels.with_row_index(name="levelid")

    return dflevels.with_columns(pl.col("levelid").cast(pl.Int64))


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Plot estimated spectra from bound-bound transitions.",
        )
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Produce an ARTIS atomic database by combining Hillier and Nahar data sets.",
        )
        parser.add_argument("-output_folder", action="store", default="artis_files", help="Folder for output files")
        parser.add_argument(
            "-output_folder_logs", action="store", default="atomic_data_logs", help="Folder for log files"
        )
        parser.add_argument(
            "-nphixspoints", type=int, default=100, help="Number of cross section points to save in output"
        )
        parser.add_argument(
            "-phixsnuincrement",
            type=float,
            default=0.03,
            help="Fraction of nu_edge incremented for each cross section point",
        )
        parser.add_argument(
            "-optimaltemperature",
            type=int,
            default=6000,
            help=(
                "(Electron and excitation) temperature at which recombination rate "
                "should be constant when downsampling cross sections"
            ),
        )
        parser.add_argument(
            "-electrontemperature",
            type=int,
            default=6000,
            help="Temperature for choosing effective collision strengths",
        )
        parser.add_argument(
            "--nophixs", action="store_true", help="Don't generate cross sections and write to phixsdata_v2.txt file"
        )

        parser.add_argument(
            "--use_hydrogenic_for_unknown_phixs",
            action="store_true",
            help="Use hydrogenic cross sections for ions with unknown cross sections",
        )

        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    ion_handlers = get_ion_handlers()

    assert len(ion_handlers) > 0
    readhillierdata.read_hyd_phixsdata()

    os.makedirs(args.output_folder, exist_ok=True)

    log_folder = os.path.join(args.output_folder, args.output_folder_logs)
    if os.path.exists(log_folder):
        # delete any existing log files
        logfiles = glob.glob(os.path.join(log_folder, "*.txt"))
        for logfile in logfiles:
            os.remove(logfile)
            print("deleting", logfile)
    else:
        os.makedirs(log_folder, exist_ok=True)

    json.dump(obj=ion_handlers, fp=Path(args.output_folder, "artisatomicionhandlers.json").open("w"))
    write_compositionfile(ion_handlers, args)
    clear_files(args)
    process_files(ion_handlers, args)


def clear_files(args: argparse.Namespace) -> None:
    # clear out the file contents, so these can be appended to later
    with (
        open(os.path.join(args.output_folder, "adata.txt"), "w"),
        open(os.path.join(args.output_folder, "transitiondata.txt"), "w"),
        open(os.path.join(args.output_folder, "phixsdata_v2.txt"), "w") as fphixs,
    ):
        fphixs.write(f"{args.nphixspoints:d}\n")
        fphixs.write(f"{args.phixsnuincrement:14.7e}\n")


def process_files(ion_handlers: list[tuple[int, list[int | tuple[int, str]]]], args: argparse.Namespace) -> None:
    for elementindex, (atomic_number, listions) in enumerate(ion_handlers):
        if not listions:
            continue

        nahar_core_states = [[None] for _ in listions]  # list of named tuples (naharcorestaterow)
        hillier_photoion_targetconfigs: list = [[] for _ in listions]

        # keys are (2S+1, L, parity), values are strings of electron configuration
        nahar_configurations: list[dict[tuple, str]] = [{} for _ in listions]

        # keys are (2S+1, L, parity, indexinsymmetry), values are lists of (energy
        # in Rydberg, cross section in Mb) tuples
        nahar_phixs_tables: list[dict[tuple, list[tuple]]] = [{} for _ in listions]

        ionization_energy_ev = [0.0 for _ in listions]
        thresholds_ev_dict: list[dict] = [{} for _ in listions]

        # list of named tuples (hillier_transition_row)
        transitions: list = [[] for _ in listions]
        transition_count_of_level_name: list[dict] = [{} for _ in listions]
        upsilondicts: list[dict] = [{} for _ in listions]

        energy_levels: list = [[] for _ in listions]
        dfenergylevels_allions: list = [pl.DataFrame() for _ in listions]
        # index matches level id
        photoionization_thresholds_ev: list = [[] for _ in listions]
        photoionization_crosssections: list = [[] for _ in listions]  # list of cross section in Mb
        photoionization_targetfractions: list = [[] for _ in listions]

        for i, ion_stage in enumerate(listions):
            handler = None

            if not isinstance(ion_stage, int):
                ion_stage, handler = ion_stage

            if handler is None:
                if atomic_number == 2 and ion_stage == 3:
                    handler = "boyle"
                elif USE_QUB_COBALT and atomic_number == 27:
                    handler = "qub_cobalt"
                elif False:  # Nahar only, usually just for testing purposes
                    handler = "nahar"
                # elif atomic_number in [
                #     26,
                # ]:  # atomic_number in [8, 26] and os.path.isfile(path_nahar_energy_file):  # Hillier/Nahar hybrid
                #     handler = "hiller_nahar"
                elif atomic_number <= 28 or atomic_number == 56:  # Hillier data only
                    handler = "cmfgen"
                elif atomic_number >= 57:  # DREAM database of Z > 57
                    handler = "dream"
                else:
                    handler = "carsus"

            logfilepath = Path(
                args.output_folder, args.output_folder_logs, f"{elsymbols[atomic_number].lower()}{ion_stage:d}.txt"
            )
            with logfilepath.open("w") as flog:
                log_and_print(
                    flog,
                    f"\n===========> Z={atomic_number} {elsymbols[atomic_number]} {roman_numerals[ion_stage]} input:",
                )
                log_and_print(flog, f"Source handler: {handler}")

                path_nahar_energy_file = f"atomic-data-nahar/{elsymbols[atomic_number].lower()}{ion_stage:d}.en.ls.txt"
                path_nahar_px_file = f"atomic-data-nahar/{elsymbols[atomic_number].lower()}{ion_stage:d}.ptpx.txt"

                # upsilondatafilenames = {(26, 2): 'fe_ii_upsilon-data.txt', (26, 3): 'fe_iii_upsilon-data.txt'}
                # if (atomic_number, ion_stage) in upsilondatafilenames:
                #     upsilonfilename = os.path.join('atomic-data-tiptopbase',
                #                                    upsilondatafilenames[(atomic_number, ion_stage)])
                #     log_and_print(flog, f'Reading effective collision strengths from {upsilonfilename}')
                #     upsilondatadf = pd.read_csv(upsilonfilename,
                #                                 names=["Z", "ion_stage", "lower", "upper", "upsilon"],
                #                                 index_col=False, header=None, sep=" ")
                #     if len(upsilondatadf) > 0:
                #         for _, row in upsilondatadf.iterrows():
                #             lower = int(row['lower'])
                #             upper = int(row['upper'])
                #             if upper < lower:
                #                 print(f'Problem in {upsilondatafilenames[(atomic_number, ion_stage)]}, lower {lower} upper {upper}. Swapping lower and upper')
                #                 old_lower = lower
                #                 lower = upper
                #                 upper = old_lower
                #             if (lower, upper) not in upsilondicts[i]:
                #                 upsilondicts[i][(lower, upper)] = row['upsilon']
                #             else:
                #                 log_and_print(flog, f"Duplicate upsilon value for transition {lower:d} to {upper:d} keeping {upsilondicts[i][(lower, upper)]:5.2e} instead of using {row['upsilon']:5.2e}")

                # if False:
                #     pass
                hillier_photoion_targetconfigs[i] = None

                # Call readHedata for He III
                if handler == "boyle":
                    (
                        ionization_energy_ev[i],
                        energy_levels[i],
                        transitions[i],
                        transition_count_of_level_name[i],
                    ) = readboyledata.read_levels_and_transitions(atomic_number, ion_stage)

                elif handler == "qub_cobalt":
                    if ion_stage in [3, 4]:  # QUB levels and transitions, or single-level Co IV
                        (
                            ionization_energy_ev[i],
                            energy_levels[i],
                            transitions[i],
                            transition_count_of_level_name[i],
                            upsilondicts[i],
                        ) = readqubdata.read_qub_levels_and_transitions(atomic_number, ion_stage, flog)
                    else:  # hillier levels and transitions
                        # if ion_stage == 2:
                        #     upsilondicts[i] = read_storey_2016_upsilondata(flog)
                        (
                            ionization_energy_ev[i],
                            energy_levels[i],
                            transitions[i],
                            transition_count_of_level_name[i],
                            hillier_levelnamesnoJ_matching_term,
                        ) = readhillierdata.read_levels_and_transitions(atomic_number, ion_stage, flog)

                    if i < len(listions) - 1 and not args.nophixs:  # don't get cross sections for top ion
                        (
                            photoionization_crosssections[i],
                            photoionization_targetfractions[i],
                            photoionization_thresholds_ev[i],
                        ) = readqubdata.read_qub_photoionizations(
                            atomic_number, ion_stage, energy_levels[i], args, flog
                        )

                elif handler == "nahar":
                    (
                        nahar_energy_levels,
                        nahar_core_states[i],
                        nahar_level_index_of_state,
                        nahar_configurations[i],
                        ionization_energy_ev[i],
                    ) = readnahardata.read_nahar_energy_level_file(
                        path_nahar_energy_file, atomic_number, ion_stage, flog
                    )

                    if i < len(listions) - 1:  # don't get cross sections for top ion
                        log_and_print(flog, f"Reading {path_nahar_px_file}")
                        nahar_phixs_tables[i], thresholds_ev_dict[i] = readnahardata.read_nahar_phixs_tables(
                            path_nahar_px_file, atomic_number, ion_stage, args
                        )

                    hillier_levelnamesnoJ_matching_term = defaultdict(list)
                    hillier_transitions: list = []
                    hillier_energy_levels = [None]
                    (
                        energy_levels[i],
                        transitions[i],
                        photoionization_crosssections[i],
                        photoionization_thresholds_ev[i],
                    ) = combine_hillier_nahar(
                        hillier_energy_levels,
                        hillier_levelnamesnoJ_matching_term,
                        hillier_transitions,
                        nahar_energy_levels,
                        nahar_level_index_of_state,
                        nahar_configurations[i],
                        nahar_phixs_tables[i],
                        thresholds_ev_dict[i],
                        args,
                        flog,
                        useallnaharlevels=True,
                    )

                    print(energy_levels[i][:3])

                elif handler == "hillier_nahar":  # Hillier/Nahar hybrid
                    (
                        nahar_energy_levels,
                        nahar_core_states[i],
                        nahar_level_index_of_state,
                        nahar_configurations[i],
                        nahar_ionization_potential_rydberg,
                    ) = readnahardata.read_nahar_energy_level_file(
                        path_nahar_energy_file, atomic_number, ion_stage, flog
                    )

                    (
                        ionization_energy_ev[i],
                        hillier_energy_levels,
                        hillier_transitions,
                        transition_count_of_level_name[i],
                        hillier_levelnamesnoJ_matching_term,
                    ) = readhillierdata.read_levels_and_transitions(atomic_number, ion_stage, flog)

                    if i < len(listions) - 1:  # don't get cross sections for top ion
                        log_and_print(flog, f"Reading {path_nahar_px_file}")
                        nahar_phixs_tables[i], thresholds_ev_dict[i] = readnahardata.read_nahar_phixs_tables(
                            path_nahar_px_file, atomic_number, ion_stage, args
                        )

                    (
                        energy_levels[i],
                        transitions[i],
                        photoionization_crosssections[i],
                        photoionization_thresholds_ev[i],
                    ) = combine_hillier_nahar(
                        hillier_energy_levels,
                        hillier_levelnamesnoJ_matching_term,
                        hillier_transitions,
                        nahar_energy_levels,
                        nahar_level_index_of_state,
                        nahar_configurations[i],
                        nahar_phixs_tables[i],
                        thresholds_ev_dict[i],
                        args,
                        flog,
                    )
                    # reading the collision data (in terms of level names) must be done after the data sets have
                    # been combined so that the level numbers are correct
                    if len(upsilondicts[i]) == 0:
                        upsilondicts[i] = readhillierdata.read_coldata(
                            atomic_number, ion_stage, energy_levels[i], flog, args
                        )

                elif handler == "cmfgen":  # Hillier CMFGEN data only
                    (
                        ionization_energy_ev[i],
                        energy_levels[i],
                        transitions[i],
                        transition_count_of_level_name[i],
                        hillier_levelnamesnoJ_matching_term,
                    ) = readhillierdata.read_levels_and_transitions(atomic_number, ion_stage, flog)

                    if len(upsilondicts[i]) == 0:
                        upsilondicts[i] = readhillierdata.read_coldata(
                            atomic_number, ion_stage, energy_levels[i], flog, args
                        )

                    if i < len(listions) - 1 and not args.nophixs:  # don't get cross sections for top ion
                        (
                            photoionization_crosssections[i],
                            hillier_photoion_targetconfigs[i],
                            photoionization_thresholds_ev[i],
                        ) = readhillierdata.read_phixs_tables(atomic_number, ion_stage, energy_levels[i], args, flog)
                    else:
                        hillier_photoion_targetconfigs[i] = None

                elif handler == "carsus":  # tardis Carsus
                    (
                        ionization_energy_ev[i],
                        energy_levels[i],
                        transitions[i],
                        transition_count_of_level_name[i],
                    ) = readcarsusdata.read_levels_and_transitions(atomic_number, ion_stage, flog)

                elif handler == "dream":  # DREAM database of Z >= 57
                    (
                        ionization_energy_ev[i],
                        energy_levels[i],
                        transitions[i],
                        transition_count_of_level_name[i],
                    ) = readdreamdata.read_levels_and_transitions(atomic_number, ion_stage, flog)

                # elif handler == "lisbon":
                #     (
                #         ionization_energy_ev[i],
                #         energy_levels[i],
                #         transitions[i],
                #         transition_count_of_level_name[i],
                #     ) = readlisbondata.read_levels_and_transitions(atomic_number, ion_stage, flog)

                elif handler in {"floers25calib", "floers25uncalib"}:
                    (
                        ionization_energy_ev[i],
                        energy_levels[i],
                        transitions[i],
                        transition_count_of_level_name[i],
                    ) = readfloers25data.read_levels_and_transitions(
                        atomic_number, ion_stage, flog, calibrated=(handler == "floers25calib")
                    )

                elif handler == "fac":
                    # early version of floers25 calib data
                    (
                        ionization_energy_ev[i],
                        energy_levels[i],
                        transitions[i],
                        transition_count_of_level_name[i],
                    ) = readfacdata.read_levels_and_transitions(atomic_number, ion_stage, flog)

                elif handler == "tanakajplt":  # Tanaka Japan-Lithuania database of 26 <= Z <= 88
                    (
                        ionization_energy_ev[i],
                        energy_levels[i],
                        transitions[i],
                        transition_count_of_level_name[i],
                    ) = readtanakajpltdata.read_levels_and_transitions(atomic_number, ion_stage, flog)

                elif handler == "gsnist":  # ground states taken from NIST
                    (
                        ionization_energy_ev[i],
                        energy_levels[i],
                        transitions[i],
                        transition_count_of_level_name[i],
                    ) = groundstatesonlynist.read_ground_levels(atomic_number, ion_stage, flog)

                else:
                    raise ValueError(f"Unknown handler: {handler}")

            dfenergylevels_allions[i] = leveltuples_to_pldataframe(energy_levels[i])

            if (
                i < len(listions) - 1
                and not args.nophixs
                and len(photoionization_crosssections[i]) == 0
                and args.use_hydrogenic_for_unknown_phixs
            ):  # don't get cross sections for top ion
                (
                    photoionization_crosssections[i],
                    photoionization_targetfractions[i],
                    photoionization_thresholds_ev[i],
                ) = match_hydrogenic_phixs(
                    atomic_number, dfenergylevels_allions[i], ionization_energy_ev[i], handler, args
                )

        dftransitions_allions = [t if isinstance(t, pl.DataFrame) else pl.DataFrame(t) for t in transitions]

        write_output_files(
            elementindex,
            dfenergylevels_allions,
            dftransitions_allions,
            upsilondicts,
            ionization_energy_ev,
            transition_count_of_level_name,
            nahar_core_states,
            nahar_configurations,
            hillier_photoion_targetconfigs,
            photoionization_thresholds_ev,
            photoionization_targetfractions,
            photoionization_crosssections,
            ion_handlers,
            args,
        )


def read_storey_2016_upsilondata(flog) -> dict[tuple[int, int], float]:
    upsilondict = {}

    filename = "atomic-data-storey/storetetal2016-co-ii.txt"
    log_and_print(flog, f"Reading effective collision strengths from {filename}")

    with open(filename) as fstoreydata:
        found_tablestart = False
        while True:
            line = fstoreydata.readline()
            if not line:
                break

            if found_tablestart:
                row = line.split()

                if len(row) <= 5:
                    break

                lower = int(row[0])
                upper = int(row[1])
                upsilon = float(row[11])
                upsilondict[(lower, upper)] = upsilon
            if line.startswith(
                "--	--	------	------	------	------	------	------	------	------	------	------	------	------	------"
            ):
                found_tablestart = True

    return upsilondict


def combine_hillier_nahar(
    hillier_energy_levels,
    hillier_levelnamesnoJ_matching_term,
    hillier_transitions,
    nahar_energy_levels,
    nahar_level_index_of_state,
    nahar_configurations,
    nahar_phixs_tables,
    thresholds_ev_dict,
    args,
    flog,
    useallnaharlevels=False,
):
    added_nahar_levels = []
    photoionization_crosssections = []
    photoionization_thresholds_ev = []
    levelids_of_levelnamenoJ = defaultdict(list)

    if useallnaharlevels:
        added_nahar_levels = nahar_energy_levels[1:]
    else:
        for levelid, energy_level in enumerate(hillier_energy_levels[1:], 1):
            levelnamenoJ = energy_level.levelname.split("[")[0]
            levelids_of_levelnamenoJ[levelnamenoJ].append(levelid)

        # match up Nahar states given in phixs data with Hillier levels, adding
        # missing levels as necessary
        def energy_if_available(state_tuple):
            if state_tuple in nahar_level_index_of_state:
                return hc_in_ev_cm * nahar_energy_levels[nahar_level_index_of_state[state_tuple]].energyabovegsinpercm

            return 999999.0

        phixs_state_tuples_sorted = sorted(nahar_phixs_tables.keys(), key=energy_if_available)
        for state_tuple in phixs_state_tuples_sorted:
            twosplusone, l, parity, indexinsymmetry = state_tuple
            hillier_level_ids_matching_this_nahar_state = []

            if state_tuple in nahar_level_index_of_state:
                nahar_energy_level = nahar_energy_levels[nahar_level_index_of_state[state_tuple]]
                nahar_energyabovegsinev = hc_in_ev_cm * nahar_energy_level.energyabovegsinpercm
            # else:
            # nahar_energy_level = None
            # nahar_energyabovegsinev = 999999.

            nahar_configuration_this_state = "_CONFIG NOT FOUND_"
            flog.write("\n")
            if state_tuple in nahar_configurations:
                nahar_configuration_this_state = nahar_configurations[state_tuple]

                if nahar_configuration_this_state.strip() in nahar_configuration_replacements:
                    nahar_configuration_this_state = nahar_configuration_replacements[
                        nahar_configurations[state_tuple].strip()
                    ]
                    flog.write(
                        f"Replacing Nahar configuration of '{nahar_configurations[state_tuple]}' with"
                        f" '{nahar_configuration_this_state}'\n"
                    )

            if hillier_levelnamesnoJ_matching_term[(twosplusone, l, parity)]:
                # match the electron configurations from the levels with matching terms
                if nahar_configuration_this_state != "_CONFIG NOT FOUND_":
                    level_match_scores = []
                    for levelname in hillier_levelnamesnoJ_matching_term[(twosplusone, l, parity)]:
                        altlevelname = hillier_name_replacements.get(levelname, levelname)

                        # set zero if already matched this level to something
                        match_score = (
                            0
                            if hillier_energy_levels[levelids_of_levelnamenoJ[levelname][0]].indexinsymmetry >= 0
                            else score_config_match(altlevelname, nahar_configuration_this_state)
                        )

                        avghillierenergyabovegsinev = weightedavgenergyinev(
                            hillier_energy_levels, levelids_of_levelnamenoJ[levelname]
                        )
                        if nahar_energyabovegsinev < 999:
                            # reduce the score by 30% for every eV of energy difference (up to 100%)
                            match_score *= 1 - min(1, 0.3 * abs(avghillierenergyabovegsinev - nahar_energyabovegsinev))

                        level_match_scores.append([levelname, match_score])

                    level_match_scores.sort(key=lambda x: -x[1])
                    best_match_score = level_match_scores[0][1]
                    if best_match_score > 0:
                        best_levelname = level_match_scores[0][0]
                        core_state_id = nahar_energy_levels[nahar_level_index_of_state[state_tuple]].corestateid

                        confignote = nahar_configurations[state_tuple]

                        if nahar_configuration_this_state != confignote:
                            confignote += f" replaced by {nahar_configuration_this_state}"

                        for levelid in levelids_of_levelnamenoJ[best_levelname]:
                            hillierlevel = hillier_energy_levels[levelid]
                            # print(hillierlevel.twosplusone, hillierlevel.l, hillierlevel.parity, hillierlevel.levelname)
                            hillier_energy_levels[levelid] = hillier_energy_levels[levelid]._replace(
                                twosplusone=twosplusone,
                                l=l,
                                parity=parity,
                                indexinsymmetry=indexinsymmetry,
                                corestateid=core_state_id,
                                naharconfiguration=confignote,
                                matchscore=best_match_score,
                            )
                            hillier_level_ids_matching_this_nahar_state.append(levelid)
                    # else:
                    #     print("no match for", nahar_configuration_this_state)
                else:
                    log_and_print(
                        flog,
                        f"No electron configuration for {twosplusone:d}{lchars[l]}{['e', 'o'][parity]} index"
                        f" {indexinsymmetry:d}",
                    )
            else:
                flog.write(f"No Hillier levels with term {twosplusone:d}{lchars[l]}{['e', 'o'][parity]}\n")

            if not hillier_level_ids_matching_this_nahar_state:
                flog.write(
                    "No matched Hillier levels for Nahar cross section of"
                    f" {twosplusone:d}{lchars[l]}{['e', 'o'][parity]} index"
                    f" {indexinsymmetry:d} '{nahar_configuration_this_state}' "
                )

                # now find the Nahar level and add it to the new list
                if state_tuple in nahar_level_index_of_state:
                    nahar_energy_level = nahar_energy_levels[nahar_level_index_of_state[state_tuple]]
                    nahar_energy_eV = nahar_energy_level.energyabovegsinpercm * hc_in_ev_cm
                    flog.write(f"(E = {nahar_energy_eV:.3f} eV, g = {nahar_energy_level.g:.1f})\n")

                    if nahar_energy_eV < 0.002:
                        flog.write(" but prevented duplicating the ground state\n")
                    else:
                        added_nahar_levels.append(
                            nahar_energy_level._replace(
                                naharconfiguration=nahar_configurations.get(state_tuple, "UNKNOWN CONFIG")
                            )
                        )
                else:
                    flog.write(" (and no matching entry in Nahar energy table, so can't be added)\n")
            else:  # there are Hillier levels matched to this state
                nahar_energy_level = nahar_energy_levels[nahar_level_index_of_state[state_tuple]]
                nahar_energyabovegsinev = hc_in_ev_cm * nahar_energy_level.energyabovegsinpercm
                # avghillierthreshold = weightedavgthresholdinev(
                #    hillier_energy_levels, hillier_level_ids_matching_this_nahar_state)
                # strhilliermatchesthreshold = '[' + ', '.join(
                #     ['{0} ({1:.3f} eV)'.format(hillier_energy_levels[k].levelname,
                #                                hc_in_ev_angstrom / float(hillier_energy_levels[k].lambdaangstrom))
                #      for k in hillier_level_ids_matching_this_nahar_state]) + ']'

                flog.write(
                    "Matched Nahar phixs for {:d}{}{} index {:d} '{}' (E = {:.3f} eV, g = {:.1f}) to \n".format(
                        twosplusone,
                        lchars[l],
                        ["e", "o"][parity],
                        indexinsymmetry,
                        nahar_configuration_this_state,
                        nahar_energyabovegsinev,
                        nahar_energy_level.g,
                    )
                )

                if len(hillier_level_ids_matching_this_nahar_state) > 1:
                    avghillierenergyabovegsinev = weightedavgenergyinev(
                        hillier_energy_levels, hillier_level_ids_matching_this_nahar_state
                    )
                    sumhillierstatweights = sum(
                        hillier_energy_levels[levelid].g for levelid in hillier_level_ids_matching_this_nahar_state
                    )
                    flog.write(f"<E> = {avghillierenergyabovegsinev:.3f} eV, g_sum = {sumhillierstatweights:.1f}: \n")
                    if abs(nahar_energyabovegsinev / avghillierenergyabovegsinev - 1) > 0.5:
                        flog.write("ENERGY DISCREPANCY WARNING\n")

                strhilliermatches = "\n".join(
                    [
                        f"{hillier_energy_levels[k].levelname} ({hc_in_ev_cm * float(hillier_energy_levels[k].energyabovegsinpercm):.3f} eV, g = {hillier_energy_levels[k].g:.1f}, match_score = {hillier_energy_levels[k].matchscore:.1f})"
                        for k in hillier_level_ids_matching_this_nahar_state
                    ]
                )

                flog.write(strhilliermatches + "\n")

    energy_levels = hillier_energy_levels + added_nahar_levels

    log_and_print(
        flog,
        f"Included {len(hillier_energy_levels) - 1} levels from Hillier dataset and added"
        f" {len(added_nahar_levels)} levels from Nahar phixs tables for a total of {len(energy_levels) - 1} levels",
    )

    # sort the concatenated energy level list by energy
    print("Sorting levels by energy...")
    energy_levels.sort(key=lambda x: float(getattr(x, "energyabovegsinpercm", "-inf")))

    if len(nahar_phixs_tables.keys()) > 0:
        photoionization_crosssections = np.zeros(
            (len(energy_levels), args.nphixspoints)
        )  # this probably gets overwritten anyway
        photoionization_thresholds_ev = np.zeros(len(energy_levels))

        # process the phixs tables and attach them to any matching levels in the output list

        if not args.nophixs:
            reduced_phixs_dict = reduce_phixs_tables(
                nahar_phixs_tables, args.optimaltemperature, args.nphixspoints, args.phixsnuincrement
            )

            for (twosplusone, l, parity, indexinsymmetry), phixstable in reduced_phixs_dict.items():
                foundamatch = False
                for levelid, energylevel in enumerate(energy_levels[1:], 1):
                    if (
                        int(energylevel.twosplusone) == twosplusone
                        and int(energylevel.l) == l
                        and int(energylevel.parity) == parity
                        and int(energylevel.indexinsymmetry) == indexinsymmetry
                    ):
                        photoionization_crosssections[levelid] = phixstable
                        photoionization_thresholds_ev[levelid] = thresholds_ev_dict[
                            (twosplusone, l, parity, indexinsymmetry)
                        ]
                        foundamatch = (
                            True  # there could be more than one match, but this flags there being at least one
                        )

                if not foundamatch:
                    log_and_print(
                        flog,
                        "No Hillier or Nahar state to match with photoionization crosssection of"
                        f" {twosplusone:d}{lchars[l]}{['e', 'o'][parity]} index {indexinsymmetry:d}",
                    )

    return energy_levels, hillier_transitions, photoionization_crosssections, photoionization_thresholds_ev


def log_and_print(flog, strout):
    print(strout)
    flog.write(strout + "\n")


def isfloat(value: t.Any) -> bool:
    try:
        float(value.replace("D", "E"))
    except ValueError:
        return False

    return True


# split a list into evenly sized chunks
def chunks(listin: list, chunk_size: int) -> list:
    return [listin[i : i + chunk_size] for i in range(0, len(listin), chunk_size)]


@lru_cache(maxsize=1)
def get_nist_ionization_energies_ev() -> dict[tuple[int, int], float]:
    """Get a dictionary where dictioniz[(atomic_number, ion_sage)] = ionization_energy_ev."""
    dfnist = pd.read_csv(
        PYDIR / "nist_ionization.txt",
        sep="\t",
        usecols=["At. num", "Ion Charge", "Ionization Energy (a) (eV)"],
    )

    dictioniz = {}
    for atomic_number, ion_charge, ioniz_ev in dfnist[
        ["At. num", "Ion Charge", "Ionization Energy (a) (eV)"]
    ].itertuples(index=False):
        with contextlib.suppress(ValueError):
            ion_stage = int(ion_charge) + 1
            dictioniz[(int(atomic_number), ion_stage)] = ioniz_ev
    return dictioniz


def match_hydrogenic_phixs(atomic_number: int, energy_levels, ionization_energy_ev: float, ion_handler: str, args):
    dict_get_n_func = {
        "tanakajplt": readtanakajpltdata.get_level_valence_n,
        "carsus": readcarsusdata.get_level_valence_n,
        "fac": readfacdata.get_level_valence_n,
        "floers25calib": readfloers25data.get_level_valence_n,
        "floers25uncalib": readfloers25data.get_level_valence_n,
    }
    if ion_handler not in dict_get_n_func:
        print(
            f"WARNING: Can't assign hydrogenic photoionization cross sections because I don't know how to find principle quantum numbers for {ion_handler} levels"
        )
        return [], [], []

    get_n = dict_get_n_func[ion_handler]
    print(f"using hydrogenic photoionization cross sections for Z={atomic_number} {elsymbols[atomic_number]}")

    alpha_squared = 0.0072973525643**2  # fine structure constant squared
    mc_squared = 0.5109989461 * 1e6  # electron mass in eV

    photoionization_crosssections = np.zeros((energy_levels.height, args.nphixspoints))
    photoionization_targetfractions: list = [[] for _ in range(energy_levels.height)]
    photoionization_thresholds_ev = np.zeros(energy_levels.height)
    phixstables = {}
    for level in energy_levels[1:].iter_rows(named=True):
        lowerlevelid = level["levelid"]
        if lowerlevelid > 100:
            # limit levels with hydrogenic photoionization cross sections
            break
        en_ev = hc_in_ev_cm * level["energyabovegsinpercm"]
        threshold_ev = ionization_energy_ev - en_ev
        photoionization_thresholds_ev[lowerlevelid] = threshold_ev
        lambda_angstrom = hc_in_ev_angstrom / threshold_ev
        if lambda_angstrom <= 0.0:
            continue

        n = get_n(level["levelname"])
        effective_charge_squared = threshold_ev * 2 * (n**2) / alpha_squared / mc_squared
        phixstables[lowerlevelid] = (
            readhillierdata.get_hydrogenic_n_phixstable(lambda_angstrom=lambda_angstrom, n=n) / effective_charge_squared
        )
        photoionization_targetfractions[lowerlevelid] = [(1, 1.0)]

    reduced_phixs_dict = reduce_phixs_tables(
        phixstables, args.optimaltemperature, args.nphixspoints, args.phixsnuincrement
    )
    for lowerlevelid, reduced_phixs_table in reduced_phixs_dict.items():
        photoionization_crosssections[lowerlevelid] = reduced_phixs_table

    return photoionization_crosssections, photoionization_targetfractions, photoionization_thresholds_ev


def reduce_phixs_tables(
    dicttables, optimaltemperature: float, nphixspoints: int, phixsnuincrement: float, hideoutput: bool = False
) -> dict:
    """Receives a dictionary, with each item being a 2D array of energy and cross section points
    Returns a dictionary with the items having been downsampled into a 1D array.

    Units don't matter, but the first (lowest) energy point is assumed to be the threshold energy
    """
    out_q: t.Any = mp.Queue()
    procs = []

    if not hideoutput:
        print(f"Processing {len(dicttables.keys()):d} phixs tables")
    nprocs = os.cpu_count()
    assert nprocs is not None
    keylist = dicttables.keys()
    for procnum in range(nprocs):
        dicttablesslice = itertools.islice(dicttables.items(), procnum, len(keylist), nprocs)
        procs.append(
            mp.Process(
                target=reduce_phixs_tables_worker,
                args=(dicttablesslice, optimaltemperature, nphixspoints, phixsnuincrement, out_q),
            )
        )
        procs[-1].start()

    dictout: dict = {}
    for _ in procs:
        subdict = out_q.get()
        # print("a process returned {:d} items".format(len(subdict.keys())))
        dictout |= subdict

    for proc in procs:
        proc.join()

    return dictout


# this method downsamples the photoionization cross section table to a
# regular grid while keeping the recombination rate integral constant
# (assuming that the temperature matches)
def reduce_phixs_tables_worker(
    dicttables: itertools.islice,
    optimaltemperature: float,
    nphixspoints: int,
    phixsnuincrement: float,
    out_q: queue.Queue,
) -> None:
    dictout = {}

    ryd_to_hz = 3289841960250880.5
    h_over_kb_in_K_sec = 4.799243073366221e-11

    # proportional to recombination rate
    # nu0 = 1e16
    # fac = math.exp(h_over_kb_in_K_sec * nu0 / optimaltemperature)

    def integrand(nu):
        return (nu**2) * math.exp(-h_over_kb_in_K_sec * nu / optimaltemperature)

    # def integrand_vec(nu_list):
    #    return [(nu ** 2) * math.exp(- h_over_kb_in_K_sec * (nu - nu0) / optimaltemperature)
    #            for nu in nu_list]

    integrand_vec = np.vectorize(integrand)

    xgrid = np.linspace(1.0, 1.0 + phixsnuincrement * (nphixspoints + 1), num=nphixspoints + 1, endpoint=False)

    # for key in keylist:
    #   tablein = dicttables[key]
    for key, tablein in dicttables:
        # # filter zero points out of the table
        # firstnonzeroindex = 0
        # for i, point in enumerate(tablein):
        #     if point[1] != 0.:
        #         firstnonzeroindex = i
        #         break
        # if firstnonzeroindex != 0:
        #     tablein = tablein[firstnonzeroindex:]

        # table says zero threshold, so avoid divide by zero
        if tablein[0][0] == 0.0:
            dictout[key] = np.zeros(nphixspoints)
            continue

        threshold_old_ryd = tablein[0][0]
        # tablein is an array of pairs (energy, phixs cross section)

        # nu0 = tablein[0][0] * ryd_to_hz

        arr_sigma_out = np.empty(nphixspoints)
        # x is nu/nu_edge

        sigma_interp = interpolate.interp1d(tablein[:, 0], tablein[:, 1], kind="linear", assume_sorted=True)

        for i, _ in enumerate(xgrid[:-1]):
            iprevious = max(i - 1, 0)
            enlow = 0.5 * (xgrid[iprevious] + xgrid[i]) * threshold_old_ryd
            enhigh = 0.5 * (xgrid[i] + xgrid[i + 1]) * threshold_old_ryd

            # start of interval interpolated point, Nahar points, and end of interval interpolated point
            samples_in_interval = tablein[(enlow <= tablein[:, 0]) & (tablein[:, 0] <= enhigh)]

            if len(samples_in_interval) == 0 or ((samples_in_interval[0, 0] - enlow) / enlow) > 1e-20:
                if i == 0 and len(samples_in_interval) != 0:
                    print(
                        f"adding first point {enlow:.4e} {samples_in_interval[0, 0]:.4e} {(samples_in_interval[0, 0] - enlow) / enlow:.4e}"
                    )
                if enlow <= tablein[-1][0]:
                    new_crosssection = sigma_interp(enlow)
                    if new_crosssection < 0:
                        print("negative extrap")
                else:
                    # assume power law decay after last point
                    new_crosssection = tablein[-1][1] * (tablein[-1][0] / enlow) ** 3
                samples_in_interval = np.vstack([[enlow, new_crosssection], samples_in_interval])

            if (
                len(samples_in_interval) == 0
                or ((enhigh - samples_in_interval[-1, 0]) / samples_in_interval[-1, 0]) > 1e-20
            ):
                if enhigh <= tablein[-1][0]:
                    new_crosssection = sigma_interp(enhigh)
                    if new_crosssection < 0:
                        print("negative extrap")
                else:
                    new_crosssection = (
                        tablein[-1][1] * (tablein[-1][0] / enhigh) ** 3
                    )  # assume power law decay after last point

                samples_in_interval = np.vstack([samples_in_interval, [enhigh, new_crosssection]])

            nsamples = len(samples_in_interval)

            # integralnosigma, err = integrate.fixed_quad(integrand_vec, enlow, enhigh, n=250)
            # integralwithsigma, err = integrate.fixed_quad(
            #    lambda x: sigma_interp(x) * integrand_vec(x), enlow, enhigh, n=250)

            # this is incredibly fast, but maybe not accurate
            # integralnosigma, err = integrate.quad(integrand, enlow, enhigh, epsrel=1e-2)
            # integralwithsigma, err = integrate.quad(
            #    lambda x: sigma_interp(x) * integrand(x), enlow, enhigh, epsrel=1e-2)

            if nsamples >= 50 or enlow > tablein[-1][0]:
                arr_energyryd = samples_in_interval[:, 0]
                arr_sigma_megabarns = samples_in_interval[:, 1]
            else:
                nsteps = 50  # was 500
                arr_energyryd = np.linspace(enlow, enhigh, num=nsteps, endpoint=False)
                arr_sigma_megabarns = np.interp(arr_energyryd, tablein[:, 0], tablein[:, 1])

            integrand_vals = integrand_vec(arr_energyryd * ryd_to_hz)
            if np.any(integrand_vals):
                sigma_integrand_vals = [
                    sigma * integrand_val
                    for sigma, integrand_val in zip(arr_sigma_megabarns, integrand_vals, strict=True)
                ]

                integralnosigma = integrate.trapezoid(integrand_vals, arr_energyryd)
                integralwithsigma = integrate.trapezoid(sigma_integrand_vals, arr_energyryd)

            else:
                integralnosigma = 1.0
                integralwithsigma = np.average(arr_sigma_megabarns)

            if integralwithsigma > 0 and integralnosigma > 0:
                arr_sigma_out[i] = integralwithsigma / integralnosigma
            elif integralwithsigma == 0:
                arr_sigma_out[i] = 0.0
            else:
                print("Math error: ", i, nsamples, arr_sigma_megabarns[i], integralwithsigma, integralnosigma)
                print(samples_in_interval)
                print(arr_sigma_out[i - 1])
                print(arr_sigma_out[i])
                print(arr_sigma_out[i + 1])
                arr_sigma_out[i] = 0.0
                # sys.exit()

        dictout[key] = arr_sigma_out  # output a 1D list of cross sections

    # return dictout
    out_q.put(dictout)


def check_forbidden(levela, levelb) -> bool:
    return levela.parity == levelb.parity


def weightedavgenergyinev(energylevels_thision, ids) -> float:
    genergysum = 0.0
    gsum = 0.0
    for levelid in ids:
        statisticalweight = float(energylevels_thision[levelid].g)
        genergysum += statisticalweight * hc_in_ev_cm * float(energylevels_thision[levelid].energyabovegsinpercm)
        gsum += statisticalweight
    return genergysum / gsum


def weightedavgthresholdinev(energylevels_thision, ids) -> float:
    genergysum = 0.0
    gsum = 0.0
    for levelid in ids:
        statisticalweight = float(energylevels_thision[levelid].g)
        genergysum += statisticalweight * hc_in_ev_angstrom / float(energylevels_thision[levelid].lambdaangstrom)
        gsum += statisticalweight
    return genergysum / gsum


alphabets = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
reversedalphabets = "zyxwvutsrqponmlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA "
lchars = "SPDFGHIKLMNOPQRSTUVWXYZ"


# reads a Hillier level name and returns the term
# tuple (twosplusone, l, parity)
def get_term_as_tuple(config: str) -> tuple[int, int, int]:
    config = config.split("[")[0]

    if "{" in config and "}" in config:  # JJ coupling, no L and S
        if config[-1] == "e":
            return (-1, -1, 0)

        if config[-1] == "o":
            return (-1, -1, 1)

        print(f"WARNING: Can't read parity from JJ coupling state '{config}'")
        return (-1, -1, -1)

    lposition = -1
    for charpos, char in reversed(list(enumerate(config))):
        if char in lchars:
            lposition = charpos
            l = lchars.index(char)
            break
    if lposition < 0:
        if config[-1] == "e":
            return (-1, -1, 0)
        if config[-1] == "o":
            return (-1, -1, 1)
    try:
        twosplusone = int(config[lposition - 1])  # could this be two digits long?
        if lposition + 1 > len(config) - 1:
            parity = 0
        elif config[lposition + 1] == "o":
            parity = 1
        elif config[lposition + 1] == "e":
            parity = 0
        elif config[lposition + 2] == "o":
            parity = 1
        elif config[lposition + 2] == "e":
            parity = 0
        else:
            twosplusone = -1
            l = -1
            parity = -1
    #        sys.exit()
    except ValueError:
        twosplusone = -1
        l = -1
        parity = -1
    return (twosplusone, l, parity)


# e.g. turn '(4F)' into (4, 3, -1)
# or '(4F1) into (4, 3, 1)
def interpret_parent_term(strin: str) -> tuple[int, int, int]:
    strin = strin.strip("()")
    lposition = -1
    for charpos, char in reversed(list(enumerate(strin))):
        if char in lchars:
            lposition = charpos
            l = lchars.index(char)
            break
    if lposition < 0:
        return (-1, -1, -1)

    twosplusone = int(strin[:lposition].lstrip(alphabets))  # could this be two digits long?

    jvalue = (
        int(strin[lposition + 1 :]) if lposition < len(strin) - 1 and strin[lposition + 1 :] not in ["e", "o"] else -1
    )
    return (twosplusone, l, jvalue)


# e.g. convert "3d64s  (6D ) 8p  j5Fo" to "3d64s8p_5Fo",
# similar to Hillier style "3d6(5D)4s8p_5Fo" but without the parent term
# (and mysterious letter before the term if present)
def reduce_configuration(instr: str) -> str:
    if instr == "-1":
        return "-1"
    instr = instr.split("[")[0]  # remove trailing bracketed J value

    if instr[-1] not in ["o", "e"]:
        instr = instr + "e"  # last character being S,P,D, etc means even
    if str.isdigit(instr[-2]):  # J value is in the term, so remove it
        instr = instr[:-2] + instr[-1]

    outstr = remove_bracketed_part(instr)
    outstr += "_"
    outstr += instr[-3:-1]
    outstr += "o" if instr[-1] == "o" else "e"
    return outstr


def remove_bracketed_part(instr: str) -> str:
    """Operates on a string by removing anything between parentheses (including the parentheses)
    e.g. remove_bracketed_part('AB(CD)EF') = 'ABEF'.
    """
    outstr = ""
    in_brackets = False
    for char in instr[:-4]:
        if char in (" ", "_"):
            continue
        if char == "(":
            in_brackets = True
        elif char == ")":
            in_brackets = False
        elif not in_brackets:
            outstr += char
    return outstr


def interpret_configuration(instr_orig: str) -> tuple[list[str], int, int, int, int]:
    max_n = 20  # maximum possible principle quantum number n
    instr = instr_orig
    instr = instr.split("[")[0]  # remove trailing bracketed J value

    if instr[-1] in lchars:
        term_parity = 0  # even
    else:
        term_parity = [0, 1][(instr[-1] == "o")]
        instr = instr[:-1]

    term_twosplusone = -1
    term_l = -1
    indexinsymmetry = -1

    while instr:
        if instr[-1] in lchars:
            term_l = lchars.index(instr[-1])
            instr = instr[:-1]
            break
        if not str.isdigit(instr[-1]):
            term_parity = (
                term_parity + 2
            )  # this accounts for things like '3d7(4F)6d_5Pbe' in the Hillier levels. Shouldn't match these
        instr = instr[:-1]

    if str.isdigit(instr[-1]):
        term_twosplusone = int(instr[-1])
        instr = instr[:-1]

    if instr[-1] == "_":
        instr = instr[:-1]
    elif instr[-1] in alphabets and (
        (len(instr) < 2 or not str.isdigit(instr[-2])) or (len(instr) < 3 or instr[-3] in lchars.lower())
    ):
        # to catch e.g., '3d6(5D)6d4Ge[9/2]' occupation piece 6d, not index d
        # and 3d7b2Fe is at index b, (keep it from conflicting into the orbital occupation)
        indexinsymmetry = reversedalphabets.index(instr[-1]) + 1 if term_parity == 1 else alphabets.index(instr[-1]) + 1
        instr = instr[:-1]

    electron_config: list[str] = []
    if not instr.startswith("Eqv st"):
        while instr:
            if instr[-1].upper() in lchars:
                try:
                    startpos = -3 if len(instr) >= 3 and str.isdigit(instr[-3]) and int(instr[-3:-1]) < max_n else -2
                except ValueError:
                    startpos = (
                        -3  # this tripped on '4sp(3P)_7Po[2]'. just pretend 4sp is an orbital and occupation number
                    )

                electron_config.insert(0, instr[startpos:])
                instr = instr[:startpos]
            elif instr[-1] == ")":
                left_bracket_pos = instr.rfind("(")
                str_parent_term = instr[left_bracket_pos:].replace(" ", "")
                electron_config.insert(0, str_parent_term)
                instr = instr[:left_bracket_pos]
            elif str.isdigit(instr[-1]):  # probably the number of electrons in an orbital
                if instr[-2].upper() in lchars:
                    startpos = -4 if len(instr) >= 4 and str.isdigit(instr[-4]) and int(instr[-4:-2]) < max_n else -3
                    electron_config.insert(0, instr[-3:])
                    instr = instr[:-3]
                else:
                    # print('Unknown character ' + instr[-1])
                    instr = instr[:-1]
            elif instr[-1] in ["_", " "]:
                instr = instr[:-1]
            else:
                # print('Unknown character ' + instr[-1])
                instr = instr[:-1]

    # return '{0} {1}{2}{3} index {4}'.format(electron_config, term_twosplusone, lchars[term_l], ['e', 'o'][term_parity], indexinsymmetry)
    return electron_config, term_twosplusone, term_l, term_parity, indexinsymmetry


def get_parity_from_config(instr) -> int:
    configsplit = interpret_configuration(instr)[0]
    lsum = 0
    for orbitalstr in configsplit:
        l = lchars.lower().index(orbitalstr[1])
        nelec = int(orbitalstr[2:]) if len(orbitalstr[2:]) > 0 else 1
        lsum += l * nelec

    return lsum % 2


def score_config_match(config_a, config_b):
    if config_a.split("[")[0] == config_b.split("[")[0]:
        return 100

    electron_config_a, term_twosplusone_a, term_l_a, term_parity_a, indexinsymmetry_a = interpret_configuration(
        config_a
    )
    electron_config_b, term_twosplusone_b, term_l_b, term_parity_b, indexinsymmetry_b = interpret_configuration(
        config_b
    )

    if term_twosplusone_a != term_twosplusone_b or term_l_a != term_l_b or term_parity_a != term_parity_b:
        return 0
    if (
        indexinsymmetry_a != -1
        and indexinsymmetry_b != -1
        and ("0s" not in electron_config_a and "0s" not in electron_config_b)
    ):
        if indexinsymmetry_a == indexinsymmetry_b:
            return 100  # exact match between Hillier and Nahar

        return 0  # both correspond to Nahar states but do not match

    if electron_config_a == electron_config_b:
        return 99

    if len(electron_config_a) > 0 and len(electron_config_b) > 0:
        parent_term_match = 0.5  # 0 is definite mismatch, 0.5 is consistent, 1 is definite match
        parent_term_index_a, parent_term_index_b = -1, -1
        matched_pieces = 0
        if "0s" in electron_config_a or "0s" in electron_config_b:
            matched_pieces += 0.5  # make sure 0s states gets matched to something
        index_a, index_b = 0, 0

        non_term_pieces_a = sum([1 for a in electron_config_a if not a.startswith("(")])
        non_term_pieces_b = sum([1 for b in electron_config_b if not b.startswith("(")])
        # go through the configuration piece by piece
        while index_a < len(electron_config_a) and index_b < len(electron_config_b):
            piece_a = electron_config_a[index_a]  # an orbital electron count or a parent term
            piece_b = electron_config_b[index_b]  # an orbital electron count or a parent term

            if piece_a.startswith("(") or piece_b.startswith("("):
                if piece_a.startswith("("):
                    if parent_term_index_a == -1:
                        parent_term_index_a = index_a
                    index_a += 1
                if piece_b.startswith("("):
                    if parent_term_index_b == -1:
                        parent_term_index_b = index_b
                    index_b += 1
            else:  # orbital occupation piece
                if piece_a == piece_b:
                    matched_pieces += 1
                # elif '0s' in [piece_a, piece_b]:  # wildcard piece
                #     matched_pieces += 0.5
                # pass
                # else:
                #     return 0

                index_a += 1
                index_b += 1

        if parent_term_index_a != -1 and parent_term_index_b != -1:
            parent_term_a = interpret_parent_term(electron_config_a[parent_term_index_a])
            parent_term_b = interpret_parent_term(electron_config_b[parent_term_index_b])
            if parent_term_index_a == parent_term_index_b:
                if parent_term_a == parent_term_b:
                    parent_term_match = 1.0
                elif parent_term_a[:2] == parent_term_b[:2] and -1 in [
                    parent_term_a[2],
                    parent_term_b[2],
                ]:  # e.g., '(3F1)' matches '(3F)'
                    # strip J values from the parent term
                    parent_term_match = 0.75
                else:
                    parent_term_match = 0.0
                    return 0
            else:  # parent terms occur at different locations. are they consistent?
                orbitaldiff = (
                    electron_config_b[parent_term_index_a:parent_term_index_b]
                    if parent_term_index_b > parent_term_index_a
                    else electron_config_a[parent_term_index_b:parent_term_index_a]
                )

                maxldiff = 0
                maxspindiff = 0  # two times s
                for orbital in orbitaldiff:
                    for pos, char in enumerate(orbital):
                        if char in lchars.lower():
                            maxldiff += lchars.lower().index(char)
                            occupation = orbital[pos + 1 :]
                            if len(occupation) > 0:
                                maxspindiff += int(occupation)
                            else:
                                maxspindiff += 1
                            break

                spindiff = abs(parent_term_a[0] - parent_term_b[0])
                ldiff = abs(parent_term_a[1] - parent_term_b[1])
                if spindiff > maxspindiff or ldiff > maxldiff:  # parent terms are inconsistent -> no match
                    # print(orbitaldiff, spindiff, maxspindiff, ldiff, maxldiff, config_a, config_b)
                    parent_term_match = 0.0
                    return 0
        score = int(98 * matched_pieces / max(non_term_pieces_a, non_term_pieces_b) * parent_term_match)
        return score

    return 5  # term matches but no electron config available or it's an Eqv state...0s type


def add_level_ids_forbidden(dfenergylevels_ion: pl.DataFrame, dftransitions_ion: pl.DataFrame) -> pl.DataFrame:
    if "upperlevel" not in dftransitions_ion.columns:
        dftransitions_ion = dftransitions_ion.join(
            dfenergylevels_ion.select(pl.col("levelid").alias("upperlevel"), pl.col("levelname").alias("nameto")),
            on="nameto",
        )

    if "lowerlevel" not in dftransitions_ion.columns:
        dftransitions_ion = dftransitions_ion.join(
            dfenergylevels_ion.select(pl.col("levelid").alias("lowerlevel"), pl.col("levelname").alias("namefrom")),
            on="namefrom",
        )

    if "forbidden" not in dftransitions_ion.columns:
        dftransitions_ion = (
            dftransitions_ion.join(
                dfenergylevels_ion.select(
                    pl.col("levelid").alias("lowerlevel"), pl.col("parity").alias("lower_parity")
                ),
                on="lowerlevel",
            )
            .join(
                dfenergylevels_ion.select(
                    pl.col("levelid").alias("upperlevel"), pl.col("parity").alias("upper_parity")
                ),
                on="upperlevel",
            )
            .with_columns(forbidden=pl.col("lower_parity") == pl.col("upper_parity"))
        )
    return dftransitions_ion


def write_output_files(
    elementindex,
    dfenergylevels_allions,
    dftransitions_allions: list[pl.DataFrame],
    upsilondicts,
    ionization_energies,
    transition_count_of_level_name,
    nahar_core_states,
    nahar_configurations,
    hillier_photoion_targetconfigs,
    photoionization_thresholds_ev,
    photoionization_targetfractions,
    photoionization_crosssections,
    ion_handlers,
    args,
):
    atomic_number, listions = ion_handlers[elementindex]

    for i, ion_stage in enumerate(listions):
        with contextlib.suppress(TypeError):
            if len(ion_stage) == 2:
                ion_stage, handler = ion_stage
        upsilondict = upsilondicts[i]
        ionstr = f"{elsymbols[atomic_number]} {roman_numerals[ion_stage]}"

        flog = open(
            os.path.join(
                args.output_folder, args.output_folder_logs, f"{elsymbols[atomic_number].lower()}{ion_stage:d}.txt"
            ),
            "a",
        )

        log_and_print(flog, f"\n===========> Z={atomic_number} {ionstr} output:")

        dfenergylevels_ion = dfenergylevels_allions[i]
        dftransitions_ion = dftransitions_allions[i]

        dftransitions_ion = add_level_ids_forbidden(dfenergylevels_ion, dftransitions_ion)

        unused_upsilon_transitions = set(upsilondicts[i].keys()).difference(
            dftransitions_ion[["lowerlevel", "upperlevel"]].iter_rows(named=False)
        )

        log_and_print(flog, f"Adding in {len(unused_upsilon_transitions):d} extra transitions with only upsilon values")

        if unused_upsilon_transitions:
            dfupsilon_only_transitions = pl.DataFrame(
                list(unused_upsilon_transitions),
                schema=(("lowerlevel", pl.Int64), ("upperlevel", pl.Int64)),
                orient="row",
            ).with_columns(A=0.0)
            for id_lower, id_upper in dfupsilon_only_transitions[["lowerlevel", "upperlevel"]].iter_rows(named=False):
                namefrom = dfenergylevels_ion["levelname"][id_upper]
                nameto = dfenergylevels_ion["levelname"][id_lower]

                transition_count_of_level_name[i][namefrom] += 1
                transition_count_of_level_name[i][nameto] += 1

            dfupsilon_only_transitions = add_level_ids_forbidden(dfenergylevels_ion, dfupsilon_only_transitions)
            dftransitions_ion = pl.concat([dftransitions_ion, dfupsilon_only_transitions], how="diagonal_relaxed")

        dftransitions_ion = dftransitions_ion.with_columns(
            pl.struct(["lowerlevel", "upperlevel", "forbidden"])
            .map_elements(
                lambda row, upsilondict=upsilondict: upsilondict.get(  # type: ignore[misc]
                    (row["lowerlevel"], row["upperlevel"]),
                    -2.0 if row["forbidden"] else -1.0,
                ),
                return_dtype=pl.Float64,
            )
            .alias("coll_str")
        )

        with open(os.path.join(args.output_folder, "adata.txt"), "a") as fatommodels:
            write_adata(
                fatommodels,
                atomic_number,
                ion_stage,
                dfenergylevels_allions[i],
                ionization_energies[i],
                transition_count_of_level_name[i],
                flog,
            )

        dftransitions_ion = dftransitions_ion.sort(by=("lowerlevel", "upperlevel"))
        with open(os.path.join(args.output_folder, "transitiondata.txt"), "a") as ftransitiondata:
            write_transition_data(
                ftransitiondata,
                atomic_number,
                ion_stage,
                dftransitions_ion,
                flog,
            )

        if i < len(listions) - 1 and not args.nophixs:  # ignore the top ion
            if len(photoionization_targetfractions[i]) < 1:
                if len(nahar_core_states[i]) > 1:
                    photoionization_targetfractions[i] = readnahardata.get_photoiontargetfractions(
                        dfenergylevels_allions[i],
                        dfenergylevels_allions[i + 1],
                        nahar_core_states[i],
                        nahar_configurations[i + 1],
                        flog,
                    )
                else:
                    photoionization_targetfractions[i] = readhillierdata.get_photoiontargetfractions(
                        dfenergylevels_allions[i], dfenergylevels_allions[i + 1], hillier_photoion_targetconfigs[i]
                    )

            with open(os.path.join(args.output_folder, "phixsdata_v2.txt"), "a") as fphixs:
                write_phixs_data(
                    fphixs,
                    atomic_number,
                    ion_stage,
                    photoionization_crosssections[i],
                    photoionization_targetfractions[i],
                    photoionization_thresholds_ev[i],
                    args,
                    flog,
                )

        flog.close()


def write_adata(
    fatommodels,
    atomic_number: int,
    ion_stage: int,
    dfenergylevels: pl.DataFrame,
    ionization_energy: float,
    transition_count_of_level_name,
    flog,
) -> None:
    log_and_print(flog, f"Writing {dfenergylevels.height-1} levels to 'adata.txt'")
    fatommodels.write(f"{atomic_number:12d}{ion_stage:12d}{dfenergylevels.height-1:12d}{ionization_energy:15.7f}\n")

    for energylevel in dfenergylevels[1:].iter_rows(named=True):
        transitioncount = (
            transition_count_of_level_name.get(energylevel["levelname"], 0) if "levelname" in energylevel else 0
        )

        level_comment = ""
        if "levelname" in energylevel:
            hlevelname = energylevel["levelname"]
            if hlevelname in hillier_name_replacements:
                # hlevelname += ' replaced by {0}'.format(hillier_name_replacements[hlevelname])
                hlevelname = hillier_name_replacements[hlevelname]
            level_comment = hlevelname.ljust(27)
        else:
            level_comment = " " * 27

        if "indexinsymmetry" in energylevel:
            if energylevel["indexinsymmetry"] >= 0:
                level_comment += (
                    f'Nahar: {energylevel["twosplusone"]:d}{lchars[energylevel["l"]]:}{["e", "o"][energylevel["parity"]]:} index'
                    f" {energylevel["indexinsymmetry"]:}"
                )
                if "naharconfiguration" in energylevel:
                    config = energylevel["naharconfiguration"]
                    if energylevel["naharconfiguration"].strip() in nahar_configuration_replacements:
                        config += f" replaced by {nahar_configuration_replacements[energylevel["naharconfiguration"].strip()]}"
                    level_comment += f" '{config}'"
                else:
                    level_comment += " (no config)"
        else:
            level_comment = level_comment.rstrip()

        fatommodels.write(
            f"{energylevel["levelid"]:5d} {hc_in_ev_cm * float(energylevel["energyabovegsinpercm"]):19.16f} {float(energylevel["g"]):8.3f} {transitioncount:4d} {level_comment:}\n"
        )

    fatommodels.write("\n")


def write_transition_data(
    ftransitiondata,
    atomic_number: int,
    ion_stage: int,
    dftransitions_ion: pl.DataFrame,
    flog,
) -> None:
    log_and_print(flog, f"Writing {dftransitions_ion.height} transitions to 'transitiondata.txt'")

    ftransitiondata.write(f"{atomic_number:7d}{ion_stage:7d}{dftransitions_ion.height:12d}\n")

    for levelid_lower, levelid_upper, A, coll_str, forbidden in dftransitions_ion[
        ["lowerlevel", "upperlevel", "A", "coll_str", "forbidden"]
    ].iter_rows():
        assert levelid_lower < levelid_upper

        ftransitiondata.write(f"{levelid_lower:4d} {levelid_upper:4d} {float(A):11.5e} {coll_str:9.2e} {forbidden:d}\n")

    ftransitiondata.write("\n")

    num_forbidden_transitions = dftransitions_ion.filter(pl.col("forbidden")).height

    num_collision_strengths_applied = dftransitions_ion.filter(pl.col("coll_str") > 0).height

    log_and_print(
        flog,
        f"  output {dftransitions_ion.height:d} transitions of which {num_forbidden_transitions:d} are forbidden and"
        f" {num_collision_strengths_applied:d} have collision strengths",
    )


def write_phixs_data(
    fphixs,
    atomic_number: int,
    ion_stage: int,
    photoionization_crosssections,
    photoionization_targetfractions,
    photoionization_thresholds_ev,
    args,
    flog,
) -> None:
    log_and_print(flog, f"Writing {len(photoionization_crosssections)} phixs tables to 'phixsdata2.txt'")
    flog.write(
        f"Downsampling cross sections assuming T={args.optimaltemperature} Kelvin, "
        f"nphixspoints={args.nphixspoints}, phixsnuincrement={args.phixsnuincrement}\n"
    )

    if len(photoionization_crosssections) >= 2 and photoionization_crosssections[1][0] == 0.0:
        log_and_print(flog, "ERROR: ground state has zero photoionization cross section")
        sys.exit()

    for lowerlevelid, targetlist in enumerate(photoionization_targetfractions[1:], 1):
        if not targetlist:
            continue
        threshold_ev = photoionization_thresholds_ev[lowerlevelid]
        if len(targetlist) <= 1 and targetlist[0][1] > 0.99:
            upperionlevelid = targetlist[0][0] if len(targetlist) > 0 else 1

            fphixs.write(
                f"{atomic_number:12d}{ion_stage + 1:12d}{upperionlevelid:8d}{ion_stage:12d}{lowerlevelid:8d}{threshold_ev:16.6E}\n"
            )
        else:
            fphixs.write(
                f"{atomic_number:12d}{ion_stage + 1:12d}{-1:8d}{ion_stage:12d}{lowerlevelid:8d}{threshold_ev:16.6E}\n"
            )
            fphixs.write(f"{len(targetlist):8d}\n")
            probability_sum = 0.0
            for upperionlevelid, targetprobability in targetlist:
                fphixs.write(f"{upperionlevelid:8d}{targetprobability:12f}\n")
                probability_sum += targetprobability
            if abs(probability_sum - 1.0) > 0.00001:
                print(f"STOP! phixs fractions sum to {probability_sum:.5f} != 1.0")
                print(targetlist)
                print(f"level id {lowerlevelid}")
                sys.exit()

        for crosssection in photoionization_crosssections[lowerlevelid]:
            fphixs.write(f"{crosssection:16.8E}\n")


def write_compositionfile(
    ion_handlers: list[tuple[int, list[int | tuple[int, str]]]], args: argparse.Namespace
) -> None:
    print("Writing compositiondata.txt")
    with open(os.path.join(args.output_folder, "compositiondata.txt"), "w") as fcomp:
        fcomp.write(f"{len(ion_handlers):d}\n")
        fcomp.write("0\n0\n")
        for atomic_number, listions in ion_handlers:
            listions_nohandlers: list[int] = drop_handlers(listions)
            ion_stage_min: int = 0
            ion_stage_max: int = 0
            nions: int = 0
            if listions_nohandlers:
                ion_stage_min = min(listions_nohandlers)
                ion_stage_max = max(listions_nohandlers)
                nions = ion_stage_max - ion_stage_min + 1

            fcomp.write(
                f"{atomic_number:d}  {nions:d}  {ion_stage_min:d}  {ion_stage_max:d}  "
                f"-1 0.0 {atomic_weights[atomic_number]:.4f}\n"
            )


if __name__ == "__main__":
    # print(interpret_configuration('3d64s_4H'))
    # print(interpret_configuration('3d6(3H)4sa4He[11/2]'))
    # print(score_config_match('3d64s_4H','3d6(3H)4sa4He[11/2]'))
    main()

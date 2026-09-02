#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Command-line entry point: build an ARTIS atomic database from the configured ions and handlers."""

import argparse
import json
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete

from artisatomic import readhillierdata
from artisatomic.iondata import read_ion_data
from artisatomic.iondata import resolve_photoion_targetfractions
from artisatomic.ionhandlers import get_ion_handlers
from artisatomic.output import clear_files
from artisatomic.output import write_compositionfile
from artisatomic.output import write_output_files


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Write an ARTIS atomic database from the configured ions and handlers."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Produce an ARTIS atomic database from published atomic data sets.",
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
            "-nlevels_hydrogenic_for_unknown_phixs",
            type=int,
            default=100,
            help=(
                "Consider this many of the lowest levels of any ion whose handler supplied no"
                " cross sections at all, and estimate a hydrogenic one for each, or 0 to disable."
                " Negative values are rejected. Fewer tables than this can result, because a level"
                " at or above the ionization energy is skipped but still counts towards the limit."
                " An ion with even one cross section from its data source is left untouched, so"
                " this never replaces or extends measured data. Excludes the top ion, which has no"
                " upper ion to photoionise to."
            ),
        )

        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    # 0 is the way to switch the estimate off, so a negative value is a typo rather than a
    # quieter way of saying the same thing
    if args.nlevels_hydrogenic_for_unknown_phixs < 0:
        msg = f"-nlevels_hydrogenic_for_unknown_phixs must not be negative, got {args.nlevels_hydrogenic_for_unknown_phixs}"
        raise ValueError(msg)

    ion_handlers = get_ion_handlers()

    if not ion_handlers:
        # not an assert: an empty selection writes an empty database rather than failing, and
        # get_ion_handlers() reads a file, so this validates input and must survive python -O
        msg = "No ions selected. artisatomicionhandlers.json is empty, or no reader found any data."
        raise ValueError(msg)

    # only the cross sections use these tables, and reading them means parsing the whole H I
    # oscillator file plus two hydrogenic data files. The CMFGEN fit types 2, 3 and 8 need them,
    # and so does the hydrogenic estimate for the other handlers. A run with neither does not
    # need the CMFGEN data at all.
    handlers_in_run = {handler for _atomic_number, listions in ion_handlers for _ion_stage, handler in listions}
    needs_hydrogenic_tables = args.nlevels_hydrogenic_for_unknown_phixs > 0 or bool(
        handlers_in_run & {"cmfgen", "qub_cobalt"}
    )
    if not args.nophixs and needs_hydrogenic_tables:
        readhillierdata.read_hyd_phixsdata()

    Path(args.output_folder).mkdir(exist_ok=True, parents=True)

    log_folder = Path(args.output_folder) / args.output_folder_logs
    if log_folder.exists():
        # delete any existing log files
        for logfile in sorted(log_folder.glob("*.txt")):
            logfile.unlink(missing_ok=True)
            print("deleting", logfile)
    else:
        Path(log_folder).mkdir(exist_ok=True, parents=True)

    # a record of what this run used, written beside the logs. It is NOT the file
    # get_ion_handlers() reads: that one is ./artisatomicionhandlers.json, in the working
    # directory. Copy this one there to repeat a run exactly, as the CI workflow does.
    with Path(log_folder, "artisatomicionhandlers.json").open("w", encoding="utf-8") as f:
        json.dump(obj=ion_handlers, fp=f)
    write_compositionfile(ion_handlers, args)
    clear_files(args)
    process_files(ion_handlers, args)


def process_files(ion_handlers: list[tuple[int, list[tuple[int, str]]]], args: argparse.Namespace) -> None:
    """Read every configured ion and append it to the output files, one element at a time.

    Ion stages are processed in ascending order so that each ion's photoionisation targets, which
    are levels of the next ion up, are already known, and so the top ion can be identified and
    given no cross sections.
    """
    for atomic_number, listions in ion_handlers:
        if not listions:
            continue

        iondatalist = [
            read_ion_data(atomic_number, ion_stage_entry, is_top_ion=(i == len(listions) - 1), args=args)
            for i, ion_stage_entry in enumerate(listions)
        ]

        if not args.nophixs:
            resolve_photoion_targetfractions(
                iondatalist, atomic_number, Path(args.output_folder, args.output_folder_logs)
            )

        write_output_files(atomic_number, iondatalist, args)


if __name__ == "__main__":
    main()

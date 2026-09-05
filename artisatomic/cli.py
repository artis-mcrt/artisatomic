#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Command-line entry point: build an ARTIS atomic database from the configured ions and handlers."""

import argparse
import json
from pathlib import Path

import argcomplete

from artisatomic.iondata import read_ion_data
from artisatomic.iondata import resolve_photoion_targetfractions
from artisatomic.ionhandlers import get_ion_handlers
from artisatomic.output import clear_files
from artisatomic.output import write_compositionfile
from artisatomic.output import write_output_files


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser. Every option has a default, so parse_args([]) gives a full namespace."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Produce an ARTIS atomic database from published atomic data sets.",
    )
    parser.add_argument("-output_folder", action="store", default="artis_files", help="Folder for output files")
    parser.add_argument("-output_folder_logs", action="store", default="atomic_data_logs", help="Folder for log files")
    parser.add_argument("-nphixspoints", type=int, default=100, help="Number of cross section points to save in output")
    parser.add_argument(
        "-phixsnuincrement",
        type=float,
        default=0.03,
        help="Step between two cross section points, as a fraction of nu_edge",
    )
    parser.add_argument(
        "-optimaltemperature",
        type=int,
        default=6000,
        help=(
            "(Electron and excitation) temperature in K. When artisatomic downsamples the cross sections,"
            " it keeps the recombination rate constant at this temperature."
        ),
    )
    parser.add_argument(
        "-electrontemperature",
        type=int,
        default=6000,
        help="Temperature in K at which artisatomic selects the effective collision strengths",
    )
    parser.add_argument(
        "--nophixs", action="store_true", help="Do not generate cross sections. Do not write phixsdata_v2.txt."
    )

    parser.add_argument(
        "-nlevels_hydrogenic_for_unknown_phixs",
        type=int,
        default=100,
        help=(
            "Estimate a hydrogenic cross section for this many of the lowest levels (by energy) of an ion"
            " whose handler supplied no cross sections. Give 0 to disable the estimate. The program"
            " rejects a negative value. The result can have fewer tables than this number. A level at or"
            " above the ionisation energy gets no table but still counts towards the limit."
            " An ion with one or more cross sections from its data source keeps them unchanged, so"
            " the estimate never replaces or extends measured data. The option does not apply to the"
            " top ion, which has no upper ion to photoionise to."
        ),
    )
    return parser


def main() -> None:
    """Write an ARTIS atomic database from the configured ions and handlers."""
    parser = build_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # 0 switches the estimate off. A negative value is therefore a typo and not a second way to
    # switch it off.
    if args.nlevels_hydrogenic_for_unknown_phixs < 0:
        msg = f"-nlevels_hydrogenic_for_unknown_phixs must not be negative, got {args.nlevels_hydrogenic_for_unknown_phixs}"
        raise ValueError(msg)

    ion_handlers = get_ion_handlers()

    if not ion_handlers:
        # Not an assert: an empty selection writes an empty database and does not fail. The function
        # get_ion_handlers() reads a file, so this check validates input and must survive python -O.
        msg = "No ions selected. artisatomicionhandlers.json is empty, or no reader found any data."
        raise ValueError(msg)

    Path(args.output_folder).mkdir(exist_ok=True, parents=True)

    log_folder = Path(args.output_folder) / args.output_folder_logs
    if log_folder.exists():
        # delete any existing log files
        for logfile in sorted(log_folder.glob("*.txt")):
            logfile.unlink(missing_ok=True)
            print("deleting", logfile)
    else:
        Path(log_folder).mkdir(exist_ok=True, parents=True)

    # A record of what this run used, beside the logs. It is NOT the file
    # get_ion_handlers() reads: that one is ./artisatomicionhandlers.json, in the working
    # directory. Copy this one there to repeat a run exactly, as the CI workflow does.
    with Path(log_folder, "artisatomicionhandlers.json").open("w", encoding="utf-8") as f:
        json.dump(obj=ion_handlers, fp=f)
    write_compositionfile(ion_handlers, args)
    clear_files(args)
    process_files(ion_handlers, args)


def process_files(ion_handlers: list[tuple[int, list[tuple[int, str]]]], args: argparse.Namespace) -> None:
    """Read every configured ion and append it to the output files, one element at a time.

    The loop processes the ion stages from the lowest to the highest. Each ion's photoionisation
    targets are levels of the next ion up, so the loop knows them already. The order also
    identifies the top ion, which gets no cross sections.
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

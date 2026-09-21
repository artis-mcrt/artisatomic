#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Command-line entry point: build an ARTIS atomic database from the configured ions and handlers."""

import argparse
import json
from pathlib import Path

import argcomplete

from artisatomic import readadasdata
from artisatomic.base import check_ion_stages_contiguous
from artisatomic.base import log_path
from artisatomic.iondata import read_ion_data
from artisatomic.iondata import resolve_photoion_targetfractions
from artisatomic.ionhandlers import get_ion_handlers
from artisatomic.ionhandlers import inputhandlersfile
from artisatomic.output import clear_files
from artisatomic.output import write_compositionfile
from artisatomic.output import write_output_files

# the record of the ions and the handlers of a run, in the output folder
handlersrecordname = "artisatomicionhandlers_used.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser. Every option has a default, so parse_args([]) gives a full namespace."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Produce an ARTIS atomic database from published atomic data sets.",
    )
    parser.add_argument("-output_folder", action="store", default="artis_files", help="Folder for output files")
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
        "-minionstage", type=int, default=1, help="Do not include an ion below this ion stage. 1 is the neutral atom"
    )
    parser.add_argument("-maxionstage", type=int, default=5, help="Do not include an ion above this ion stage")
    parser.add_argument(
        "-maxatomicnumber", type=int, default=None, help="Do not include an element above this atomic number"
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

    # argparse applies no default to an attribute that the namespace holds already, so an ion
    # limit that the command line omits stays None. main() finds the given limits from that, and
    # then applies the default value of each other limit.
    ionlimits = ("minionstage", "maxionstage", "maxatomicnumber")
    args = parser.parse_args(namespace=argparse.Namespace(**dict.fromkeys(ionlimits)))
    ionlimits_given = [name for name in ionlimits if getattr(args, name) is not None]
    for name in ionlimits:
        if getattr(args, name) is None:
            setattr(args, name, parser.get_default(name))

    # artisatomicionhandlers.json holds the ion stages and the atomic numbers already, so that
    # file selects the ions itself. Not an assert: this validates the command line and must
    # survive python -O.
    if ionlimits_given and inputhandlersfile.exists():
        options = ", ".join(f"-{name}" for name in ionlimits_given)
        msg = (
            f"{inputhandlersfile.resolve()} exists, so that file selects the ions."
            f" Remove the file. As an alternative, remove {options}."
        )
        raise ValueError(msg)

    # Ion stage 1 is the neutral atom, and hydrogen is atomic number 1. A smaller limit selects no
    # ion at all, and is therefore a typo.
    for name in ionlimits:
        limit = getattr(args, name)
        if limit is not None and limit < 1:
            msg = f"-{name} must be 1 or more, got {limit}"
            raise ValueError(msg)

    if args.minionstage > args.maxionstage:
        msg = f"-minionstage {args.minionstage} is above -maxionstage {args.maxionstage}, so no ion remains"
        raise ValueError(msg)

    # 0 switches the estimate off. A negative value is therefore a typo and not a second way to
    # switch it off.
    if args.nlevels_hydrogenic_for_unknown_phixs < 0:
        msg = f"-nlevels_hydrogenic_for_unknown_phixs must not be negative, got {args.nlevels_hydrogenic_for_unknown_phixs}"
        raise ValueError(msg)

    # get_ion_handlers() finds the ADAS ions in this directory, so the rename comes first
    readadasdata.rename_old_adas_directory()
    ion_handlers = get_ion_handlers(
        minionstage=args.minionstage, maxionstage=args.maxionstage, maxatomicnumber=args.maxatomicnumber
    )

    if not ion_handlers:
        # Not an assert: an empty selection writes an empty database and does not fail. The function
        # get_ion_handlers() reads a file, so this check validates input and must survive python -O.
        msg = (
            "No ions selected. The ion limits exclude every ion, artisatomicionhandlers.json is"
            " empty, or no reader found any data."
        )
        raise ValueError(msg)

    # The readers can offer an element that has a gap in its ion stages. -maxionstage 6 gives
    # Sr I-IV and Sr VI, because no data source here holds Sr V. write_compositionfile() rejects
    # such a gap. This check runs first, because the code below deletes the logs of the last run.
    check_ion_stages_contiguous(ion_handlers)

    Path(args.output_folder).mkdir(exist_ok=True, parents=True)

    # this empties the log of the last run. The passes of each ion append to the file.
    log_path(args.output_folder).write_text("", encoding="utf-8")
    remove_old_log_folder(Path(args.output_folder))

    # A record of what this run used, beside the output files. Its name is not the name of the
    # file that get_ion_handlers() reads (./artisatomicionhandlers.json). A run into the working
    # directory, or a later run from inside an output folder, must not take the record of an
    # earlier run as its ion selection. Copy the record to ./artisatomicionhandlers.json to repeat
    # a run exactly. It holds the ions that the limits kept, so the repeat run must not give a
    # limit again.
    with Path(args.output_folder, handlersrecordname).open("w", encoding="utf-8") as f:
        json.dump(obj=ion_handlers, fp=f)
    write_compositionfile(ion_handlers, args)
    clear_files(args)
    process_files(ion_handlers, args)


def remove_old_log_folder(output_folder: Path) -> None:
    """Remove the log files that an earlier release wrote to the folder atomic_data_logs.

    That release wrote one log file for each ion, and a copy of the ion handlers, into this folder.
    A user could take such a file for a record of the new run. The function removes each .txt
    file of the folder and the copy of the ion handlers, as that release did at the start of each
    run. It then removes the folder if the folder is empty.
    """
    old_log_folder = output_folder / "atomic_data_logs"
    if not old_log_folder.is_dir():
        return
    for oldfile in sorted([*old_log_folder.glob("*.txt"), old_log_folder / "artisatomicionhandlers.json"]):
        if oldfile.is_file():
            print("deleting", oldfile)
            oldfile.unlink()
    if not any(old_log_folder.iterdir()):
        old_log_folder.rmdir()


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
            resolve_photoion_targetfractions(iondatalist, atomic_number, log_path(args.output_folder))

        write_output_files(atomic_number, iondatalist, args)


if __name__ == "__main__":
    main()

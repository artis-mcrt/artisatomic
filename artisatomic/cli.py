#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Command-line entry point: build an ARTIS atomic database from the configured ions and handlers."""

import argparse
import contextlib
import json
import math
import os
import shutil
import signal
import socket
import sys
import tempfile
import typing as t
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import argcomplete

# The functions import the modules of the package only when they need them. The package imports
# polars and numpy, which took 0.3 s, so "--help" and the tab completion waited for them.

# The name of the ion handlers file that a run reads from the working directory.
# ionhandlers.inputhandlersfile holds the same name, and a test compares the two. The parser needs
# the name for its help text, and an import of ionhandlers there would import polars.
defaulthandlersfilename = "artisatomicionhandlers.json"
# the record of the ions and the handlers of a run, in the output folder
handlersrecordname = "artisatomicionhandlers_used.json"
# the log of a run that failed, in the output folder. The log of the earlier run stays beside its output.
failedlogname = "artisatomiclog_failed.txt"


class VersionAction(argparse.Action):
    """Print the version of the package and stop. The import of importlib.metadata waits until the user gives the option."""

    def __init__(self, option_strings: list[str], dest: str, **kwargs: t.Any) -> None:
        """Make an option that takes no value and puts no attribute into the namespace."""
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)

    def __call__(self, parser: argparse.ArgumentParser, *_args: object, **_kwargs: object) -> None:
        """Print the version and stop the program with status 0."""
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version

        try:
            packageversion = version("artisatomic")
        except PackageNotFoundError:
            packageversion = "has no version, because the environment holds no metadata of the package"
        print(f"artisatomic {packageversion}")
        parser.exit()


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Add the default value to the help of each option, except for a default of None.

    None is no value that a user can give. The help of such an option says what the program does
    with no such option.
    """

    @t.override
    def _get_help_string(self, action: argparse.Action) -> str | None:
        """Give the help of the option, with the default value if it is not None."""
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser. Every option has a default, so parse_args([]) gives a full namespace.

    Each option has a form with one dash and a form with two dashes. The job scripts of the users
    give the form with one dash.
    """
    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=(
            "Produce an ARTIS atomic database from published atomic data sets. The program writes adata.txt,"
            " compositiondata.txt, transitiondata.txt and phixsdata_v2.txt into the output folder."
        ),
        epilog=(
            "The built-in ion selection needs the data sets of its handlers. --listions shows the selection, and"
            " --listhandlers shows the data source of each handler. The documentation is at"
            " https://artisatomic.readthedocs.io"
        ),
    )
    parser.add_argument("--version", "-version", action=VersionAction, help="Show the version of artisatomic and stop")
    parser.add_argument(
        "-output_folder", "--output-folder", action="store", default="artis_files", help="Folder for output files"
    )
    parser.add_argument(
        "-nphixspoints",
        "--nphixspoints",
        type=int,
        default=100,
        help="Number of cross section points; at least two",
    )
    parser.add_argument(
        "-phixsnuincrement",
        "--phixsnuincrement",
        type=float,
        default=0.03,
        help="Step between two cross section points, as a fraction of nu_edge",
    )
    parser.add_argument(
        "-optimaltemperature",
        "--optimaltemperature",
        type=int,
        default=6000,
        help=(
            "(Electron and excitation) temperature in K. When artisatomic downsamples the cross sections,"
            " it keeps the recombination rate constant at this temperature."
        ),
    )
    parser.add_argument(
        "-electrontemperature",
        "--electrontemperature",
        type=int,
        default=6000,
        help="Temperature in K at which artisatomic selects the effective collision strengths",
    )
    parser.add_argument(
        "--nophixs",
        "-nophixs",
        action="store_true",
        help="Do not generate cross sections. Do not write phixsdata_v2.txt.",
    )

    parser.add_argument(
        "-ionhandlersfile",
        "--ionhandlersfile",
        type=Path,
        default=None,
        help=(
            "JSON file that selects the ions and the handler of each ion. With no such option, the program reads"
            f" {defaulthandlersfilename} in the working directory if that file exists. With no file, the program"
            " uses the built-in ion selection."
        ),
    )
    parser.add_argument(
        "-minionstage",
        "--minionstage",
        type=int,
        default=1,
        help=(
            "Do not include an ion of the built-in ion selection below this ion stage. 1 is the neutral atom."
            " The program rejects this option with an ion handlers file."
        ),
    )
    parser.add_argument(
        "-maxionstage",
        "--maxionstage",
        type=int,
        default=5,
        help=(
            "Do not include an ion of the built-in ion selection above this ion stage."
            " The program rejects this option with an ion handlers file."
        ),
    )
    parser.add_argument(
        "-maxatomicnumber",
        "--maxatomicnumber",
        type=int,
        default=None,
        help=(
            "Do not include an element of the built-in ion selection above this atomic number. With no such option,"
            " the program applies no limit to the atomic number. The program rejects this option with an ion"
            " handlers file."
        ),
    )
    parser.add_argument(
        "--listions",
        "-listions",
        action="store_true",
        help=(
            "Print the ion selection in the format of an ion handlers file, and stop. The program reads no ion"
            " and writes no output file."
        ),
    )
    parser.add_argument(
        "--listhandlers",
        "-listhandlers",
        action="store_true",
        help="Print each handler with its data source, and stop.",
    )

    parser.add_argument(
        "-nlevels_hydrogenic_for_unknown_phixs",
        "--nlevels-hydrogenic-for-unknown-phixs",
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


def get_builtin_ion_handlers(
    minionstage: int | None, maxionstage: int | None, maxatomicnumber: int | None
) -> list[tuple[int, list[tuple[int, str]]]]:
    """Give the built-in ion selection of ionhandlers.get_builtin_ion_handlers(), with the import only at the call."""
    from artisatomic import ionhandlers

    return ionhandlers.get_builtin_ion_handlers(
        minionstage=minionstage, maxionstage=maxionstage, maxatomicnumber=maxatomicnumber
    )


def format_ion_handlers(ion_handlers: list[tuple[int, list[tuple[int, str]]]]) -> str:
    """Give the ion selection as the text of an ion handlers file, with one line for each element."""
    lines = [f"    {json.dumps([atomic_number, listions])}" for atomic_number, listions in ion_handlers]
    return "[\n" + ",\n".join(lines) + "\n]\n"


def format_handlers() -> str:
    """Give one paragraph for each handler: the name, then the data source of the handler.

    The text is ASCII, as in the comment blocks of the output files. A standard output with an
    encoding such as cp1252 cannot take each character of the references.
    """
    from artisatomic.base import to_ascii
    from artisatomic.iondata import handlers

    return "\n".join(f"{name}\n    {to_ascii(handler.summary)}\n" for name, handler in sorted(handlers.items()))


def cmfgen_data_needed(ion_handlers: list[tuple[int, list[tuple[int, str]]]], args: argparse.Namespace) -> bool:
    """Whether the run reads the CMFGEN data set for certain.

    Each ion of the cmfgen and cmfgen_qubphixs handlers reads its files, except H II. The reader
    builds H II from constants, and H II is the top ion of H, so it needs no cross section.

    The hydrogenic estimate of an ion below the top ion reads the hydrogen tables of the data set.
    An ion gets the estimate if its handler has a parser for its level names and no cross sections
    for the ion. --nophixs and -nlevels_hydrogenic_for_unknown_phixs 0 switch the estimate off.
    Handler.phixs_ions names the ions with cross sections of a handler such as adas. For a
    different handler with cross sections, only the data shows if an ion gets the estimate.
    read_hyd_phixsdata() checks the folder then.
    """
    from artisatomic.iondata import handlers

    estimate = not args.nophixs and args.nlevels_hydrogenic_for_unknown_phixs > 0
    for atomic_number, listions in ion_handlers:
        for index, (ion_stage, handlername) in enumerate(listions):
            if handlername in {"cmfgen", "cmfgen_qubphixs"} and (atomic_number, ion_stage) != (1, 2):
                return True
            handler = handlers[handlername]
            is_top_ion = index == len(listions) - 1
            gives_phixs = handler.read_phixs is not None and (
                handler.phixs_ions is None or (atomic_number, ion_stage) in handler.phixs_ions
            )
            if estimate and not is_top_ion and handler.get_level_valence_n is not None and not gives_phixs:
                return True
    return False


def check_arguments(
    args: argparse.Namespace, handlersfile: Path | None, ionlimits: tuple[str, ...], ionlimits_given: list[str]
) -> None:
    """Raise ValueError for an option value, or for a set of options, that the run cannot use.

    handlersfile is the ion handlers file that find_handlers_file() gives.
    """
    # An ion handlers file holds the ion stages and the atomic numbers already, so that file selects
    # the ions itself. Not an assert: this validates the command line and must survive python -O.
    if ionlimits_given and handlersfile is not None:
        options = ", ".join(f"-{name}" for name in ionlimits_given)
        remedy = (
            f"Remove {options}."
            if args.ionhandlersfile is not None
            else f"Remove the file. As an alternative, remove {options}."
        )
        # absolute() and not resolve(): the message names the path that the user gave, and not the
        # target of a link, which "Remove the file" would then delete
        msg = f"{handlersfile.absolute()} exists, so that file selects the ions. {remedy}"
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

    # A bad value can give NaN cross sections, or the collision strengths of the lowest temperature,
    # with no error
    for name in ("nphixspoints", "phixsnuincrement", "optimaltemperature", "electrontemperature"):
        value = getattr(args, name)
        if not (math.isfinite(value) and value > 0):
            msg = f"-{name} must be a finite number more than 0, got {value}"
            raise ValueError(msg)

    if args.nphixspoints < 2:
        msg = "-nphixspoints must be 2 or more. ARTIS needs two points for a recombination integral."
        raise ValueError(msg)


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

    if args.listhandlers:
        print(format_handlers(), end="")
        return

    # The import loads numpy and polars. It comes before the try block, so an error of the
    # environment gives its traceback and not a usage error.
    from artisatomic.ionhandlers import find_handlers_file
    from artisatomic.ionhandlers import read_handlers_file

    # parser.error() prints the usage and one line, and stops with status 2. A traceback would hide
    # the line that names the option.
    try:
        handlersfile = find_handlers_file(args.ionhandlersfile)
        check_arguments(args, handlersfile, ionlimits, ionlimits_given)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    from artisatomic import readadasdata
    from artisatomic.base import check_ion_stages_contiguous

    # With --listions, the standard output holds the selection only, so a user can write it to a file.
    with contextlib.redirect_stdout(sys.stderr if args.listions else sys.stdout):
        # The rename of the old ADAS folder can move the ion handlers file that the checks found, so
        # the read comes before the rename. The run does not search for the file again.
        ion_handlers = None
        if handlersfile is not None:
            try:
                ion_handlers = read_handlers_file(handlersfile)
            except (ValueError, TypeError) as error:
                parser.error(str(error))
        # get_builtin_ion_handlers() finds the ADAS ions in the new folder, so the rename comes first
        readadasdata.rename_old_adas_directory()
        if ion_handlers is None:
            ion_handlers = get_builtin_ion_handlers(
                minionstage=args.minionstage, maxionstage=args.maxionstage, maxatomicnumber=args.maxatomicnumber
            )

    if not any(ions for _, ions in ion_handlers):
        # Not an assert: an empty selection writes an empty database and does not fail. An ion
        # handlers file can give it, so this check validates input and must survive python -O.
        msg = (
            "No ions selected. The ion limits exclude every ion, the ion handlers file names"
            " no ion, or no reader found any data."
        )
        raise ValueError(msg)

    # The readers can offer an element that has a gap in its ion stages. -maxionstage 6 gives
    # Sr I-IV and Sr VI, because no data source here holds Sr V. write_compositionfile() rejects
    # such a gap. This check runs first, because the read of the ions can take hours.
    check_ion_stages_contiguous(ion_handlers)

    if args.listions:
        print(format_ion_handlers(ion_handlers), end="")
        return

    # The run reads the CMFGEN files only after it wrote compositiondata.txt, and after it read the
    # ions of each lower element.
    if cmfgen_data_needed(ion_handlers, args):
        from artisatomic import readhillierdata

        readhillierdata.check_data_folder()

    outputfolder = Path(args.output_folder)
    outputfolder.mkdir(exist_ok=True, parents=True)
    remove_stale_work_folders(outputfolder)

    # The run writes into a new folder in the output folder, and moves the files at the end. A run
    # that fails then leaves the output of the earlier run. The level ids of one file refer to the
    # others. The name holds the host and the process id, so a later run can find a stale folder.
    workfolder = Path(
        tempfile.mkdtemp(prefix=f"{workfolder_prefix}{os.getpid()}@{socket.gethostname()}@", dir=outputfolder)
    )
    # The signal handler stays active until the end of the install, so a SIGTERM during the moves
    # also restores the files of the earlier run.
    with stop_signals_as_exit():
        try:
            write_files(ion_handlers, argparse.Namespace(**{**vars(args), "output_folder": str(workfolder)}))
        except BaseException:
            with signals_held(), output_folder_lock(outputfolder):
                discard_run(workfolder, outputfolder)
            raise

        # Two runs into one output folder can end at the same time. The lock lets only one run change
        # the files in the output folder, so the folder holds the files of one run. A signal waits
        # until the end of the install and its cleanup, also during the wait for the lock. Between the
        # last move and the cleanup, no handler could tell a full install from a part install.
        with signals_held(), output_folder_lock(outputfolder):
            try:
                earlierfolder = install_files(workfolder, outputfolder)
            except BaseException:
                discard_run(workfolder, outputfolder)
                raise

            # The output folder holds the full new output now. A failure of the cleanup must not make
            # the run fail, because the files of the earlier run are gone from the output folder.
            try:
                shutil.rmtree(earlierfolder)
                workfolder.rmdir()
                (outputfolder / failedlogname).unlink(missing_ok=True)
                remove_old_log_folder(outputfolder, keep=handlersfile)
            except OSError as error:
                print(f"WARNING: The run wrote all output files, but the cleanup of the output folder failed: {error}")


workfolder_prefix = ".artisatomic_run_"


def signals_to_take_over(names: tuple[str, ...]) -> list[int]:
    """Give the signals of these names whose handler the run can replace and put back.

    A signal that the process ignores stays ignored: nohup makes a run ignore SIGHUP, so that the
    run continues after a logout. A handler that code outside Python installed has no Python
    object (signal.getsignal() gives None), so signal.signal() cannot put it back.
    """
    signums = [getattr(signal, name) for name in names if hasattr(signal, name)]
    handlers = {signum: signal.getsignal(signum) for signum in signums}
    # by identity: a handler can be an object with no hash, which a membership test cannot take
    return [signum for signum, handler in handlers.items() if handler is not signal.SIG_IGN and handler is not None]


@contextmanager
def stop_signals_as_exit() -> Generator[None]:
    """Turn SIGTERM and SIGHUP into SystemExit, so that the cleanup of a failed run runs for them also.

    A batch system sends SIGTERM at the end of the time of a job. Python stops at such a signal with
    no cleanup, and the work folder stayed in the output folder. signals_to_take_over() gives the
    signals that the function leaves alone.
    """
    signums = signals_to_take_over(("SIGTERM", "SIGHUP"))

    def raise_exit(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    try:
        previous = {signum: signal.signal(signum, raise_exit) for signum in signums}
    except ValueError:
        # signal.signal() works only in the main thread
        yield
        return
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@contextmanager
def signals_held() -> Generator[None]:
    """Hold SIGINT, SIGTERM and SIGHUP until the end of the block, and then raise each held signal again.

    A handler records each signal during the block. A signal mask is not enough, because the system
    can give a signal to another thread, and Python then runs the handler in the main thread.
    signals_to_take_over() gives the signals that the function leaves alone.
    """
    signums = signals_to_take_over(("SIGINT", "SIGTERM", "SIGHUP"))
    held: list[int] = []

    def hold(signum: int, _frame: object) -> None:
        held.append(signum)

    try:
        previous = {signum: signal.signal(signum, hold) for signum in signums}
    except ValueError:
        # signal.signal() works only in the main thread
        yield
        return
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        # the handler of the block around this one acts on the signal now, as if it came at this point
        for signum in dict.fromkeys(held):
            signal.raise_signal(signum)


def remove_stale_work_folders(outputfolder: Path) -> None:
    """Remove the work folders of earlier runs on this host whose process no longer exists.

    A run that the system kills (SIGKILL), or that stops while it removes its work folder, leaves the
    folder behind. A folder of another host, or of an earlier release with no process id in its name,
    gets a warning only, because the function cannot know that no run uses it.
    """
    hostname = socket.gethostname()
    for folder in sorted(outputfolder.glob(f"{workfolder_prefix}*")):
        # the name is <prefix><pid>@<host>@<random part of mkdtemp()>
        nameparts = folder.name.removeprefix(workfolder_prefix).split("@")
        if len(nameparts) == 3 and nameparts[1] == hostname and nameparts[0].isdigit():
            pid = int(nameparts[0])
            if pid == os.getpid() or process_exists(pid):
                continue
            print(f"Removing the work folder of a run that stopped: {folder}")
            shutil.rmtree(folder, ignore_errors=True)
        else:
            print(f"WARNING: {folder} is the work folder of another run. Remove it if no run uses it.")


def process_exists(pid: int) -> bool:
    """Whether a process with this id exists on this host. On Windows the function always gives True.

    On Windows, os.kill() does not test a process: it sends a console event or stops the process.
    """
    if sys.platform == "win32":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def output_folder_lock(outputfolder: Path) -> Generator[None]:
    """Hold an exclusive lock on the output folder. A second run waits until the first run releases it.

    The operating system releases the lock when the process stops, so a run that crashes leaves no
    lock. Windows has no flock(), so there the function gives no lock. Some file systems refuse
    flock() (Lustre with the option noflock, NFS with no lock daemon), and a folder with no read
    permission refuses the open. There the function gives a warning and no lock, so a run does not
    fail after all of its work.
    """
    try:
        import fcntl
    except ImportError:
        yield
        return
    folderfd = None
    try:
        try:
            folderfd = os.open(outputfolder, os.O_RDONLY)
            fcntl.flock(folderfd, fcntl.LOCK_EX)
        except OSError as error:
            print(
                f"WARNING: {outputfolder} gives no lock ({error}). The run continues with no lock."
                " Two runs that end at one time can mix their files."
            )
        yield
    finally:
        if folderfd is not None:
            os.close(folderfd)


def discard_run(workfolder: Path, outputfolder: Path) -> None:
    """Remove the work folder of a run that failed. Keep its log as failedlogname in the output folder."""
    from artisatomic.base import log_path

    if log_path(workfolder).is_file():
        log_path(workfolder).replace(outputfolder / failedlogname)
    shutil.rmtree(workfolder)
    print(f"The run failed. The output folder keeps the files of the earlier run. See {failedlogname}.")


def install_files(workfolder: Path, outputfolder: Path) -> Path:
    """Move the files of the run from workfolder into outputfolder. Give the folder that holds the earlier files.

    The files of the earlier run go to a new folder first. If a move fails, the function moves the
    new files back to workfolder, restores the earlier files, and raises the error again.
    phixsdata_v2.txt goes to the new folder also, because --nophixs writes no new one. Its level
    ids belong to the adata.txt of the earlier run.
    """
    earlierfolder = Path(tempfile.mkdtemp(prefix=".artisatomic_earlier_", dir=outputfolder))
    newnames = sorted(newfile.name for newfile in workfolder.iterdir())
    installednames: list[str] = []
    try:
        for name in sorted({*newnames, "phixsdata_v2.txt"}):
            if (outputfolder / name).exists():
                (outputfolder / name).replace(earlierfolder / name)
        for name in newnames:
            (workfolder / name).replace(outputfolder / name)
            installednames.append(name)
    except BaseException:
        for name in installednames:
            (outputfolder / name).replace(workfolder / name)
        for earlierfile in earlierfolder.iterdir():
            earlierfile.replace(outputfolder / earlierfile.name)
        earlierfolder.rmdir()
        raise
    return earlierfolder


def write_files(ion_handlers: list[tuple[int, list[tuple[int, str]]]], args: argparse.Namespace) -> None:
    """Write all output files of the run, the log file, and the record of the ion handlers to args.output_folder."""
    from artisatomic.base import log_path
    from artisatomic.output import clear_files
    from artisatomic.output import write_compositionfile

    # this makes the log file. The passes of each ion append to the file.
    log_path(args.output_folder).write_text("", encoding="utf-8")

    # A record of what this run used, beside the output files. Its name is not the name of the
    # file that find_handlers_file() finds (./artisatomicionhandlers.json). A run into the working
    # directory, or a later run from inside an output folder, must not take the record of an
    # earlier run as its ion selection. Give the record to -ionhandlersfile to repeat a run exactly.
    # It holds the ions that the limits kept, so the repeat run must not give a limit again.
    with Path(args.output_folder, handlersrecordname).open("w", encoding="utf-8") as f:
        json.dump(obj=ion_handlers, fp=f)
    write_compositionfile(ion_handlers, args)
    clear_files(args)
    process_files(ion_handlers, args)


def is_same_file(path: Path, other: Path | None) -> bool:
    """Whether the two paths name one file. A path that does not exist names no file."""
    if other is None:
        return False
    try:
        return path.samefile(other)
    except OSError:
        return False


def remove_old_log_folder(output_folder: Path, keep: Path | None = None) -> None:
    """Remove the log files that an earlier release wrote to the folder atomic_data_logs.

    That release wrote one log file for each ion, and a copy of the ion handlers, into this folder.
    A user could take such a file for a record of the new run. The function removes each .txt
    file of the folder and the copy of the ion handlers. That release did the same at the start
    of each run. The function then removes the folder if the folder is empty.

    keep is the ion handlers file of the run. The function keeps that file, also if it is the copy
    of the ion handlers in this folder.
    """
    old_log_folder = output_folder / "atomic_data_logs"
    if not old_log_folder.is_dir():
        return
    for oldfile in sorted([*old_log_folder.glob("*.txt"), old_log_folder / "artisatomicionhandlers.json"]):
        if oldfile.is_file() and not is_same_file(oldfile, keep):
            print("deleting", oldfile)
            oldfile.unlink()
    # rmdir() fails on a symbolic link to a folder, so such a link stays
    if not old_log_folder.is_symlink() and not any(old_log_folder.iterdir()):
        old_log_folder.rmdir()


def process_files(ion_handlers: list[tuple[int, list[tuple[int, str]]]], args: argparse.Namespace) -> None:
    """Read every configured ion and append it to the output files, one element at a time.

    The loop processes the ion stages from the lowest to the highest. Each ion's photoionisation
    targets are levels of the next ion up, so the loop knows them already. The order also
    identifies the top ion, which gets no cross sections.
    """
    from artisatomic.base import log_path
    from artisatomic.iondata import read_ion_data
    from artisatomic.iondata import resolve_photoion_targetfractions
    from artisatomic.output import write_output_files

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

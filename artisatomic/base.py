"""Element data, physical constants, and small utilities shared by the data-source readers."""

import atexit
import io
import itertools
import math
import multiprocessing as mp
import operator
import os
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

PYDIR = Path(__file__).parent.resolve()

# Read once at import. Every reader that has a test data sample keys its data path on this flag.
# A change to the variable after the import must not redirect only some of the readers.
TESTMODE = os.environ.get("ARTISATOMIC_TESTMODE") == "1"


def _read_atomic_properties() -> tuple[list[str], list[float]]:
    """Read the element symbols and masses from atomic_properties.txt, in atomic number order.

    A plain split of each line reads the file, not pandas. Every spawned worker imports this
    module, and nothing else on that import path uses pandas.
    """
    with (PYDIR / "atomic_properties.txt").open(encoding="utf-8") as fproperties:
        rows = [line.split() for line in fproperties if line.strip() and not line.startswith("#")]
    columns = rows[0]
    symbolcolumn, numbercolumn, masscolumn = columns.index("symbol"), columns.index("number"), columns.index("mass")
    symbols = [row[symbolcolumn] for row in rows[1:]]
    # estimate an unknown atomic mass ("NA" in the file) as Z / 0.45
    masses = [float(row[masscolumn]) if row[masscolumn] != "NA" else int(row[numbercolumn]) / 0.45 for row in rows[1:]]
    return symbols, masses


_symbols, _masses = _read_atomic_properties()
elsymbols = ["n", *_symbols]
atomic_weights = ["n", *_masses]

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

# the single copy of each constant for the whole package
ryd_to_ev = 13.605693122994232
hc_in_ev_cm = 0.0001239841984332003
hc_in_ev_angstrom = 12398.419843320025
h_in_ev_seconds = 4.135667696923859e-15
ryd_to_hz = 3289841960250880.5  # equal to ryd_to_ev / h_in_ev_seconds
h_over_kb_in_K_sec = 4.799243073366221e-11
# A = gf / (gf_to_a_coefficient * g_upper * lambda_angstrom^2) with A in s^-1
gf_to_a_coefficient = 1.49919e-16


def split_element_ionstage_str(ionstr: str) -> tuple[int, int]:
    """Split a string like 'FeII' into (atomic_number, ion_stage).

    A split on `ionstr.rstrip("IVX")` destroys the symbols of the elements whose symbols contain
    only those letters: V (vanadium) and I (iodine). Instead find the split point where
    the prefix is an element symbol and the suffix is a Roman numeral. Element symbols have a
    lowercase second letter and Roman numerals are uppercase, so the match is unambiguous.
    """
    for splitpos in range(1, len(ionstr)):
        elsym, ion_stage_roman = ionstr[:splitpos], ionstr[splitpos:]
        if elsym in elsymbols and ion_stage_roman in roman_numerals[1:]:
            return elsymbols.index(elsym), roman_numerals.index(ion_stage_roman)

    msg = f"Could not split '{ionstr}' into an element symbol and a Roman numeral ion stage"
    raise ValueError(msg)


# The id-keyed transition columns write_transition_data() needs, so an empty frame still carries
# them. Name-keyed frames get lowerlevel/upperlevel from add_level_ids_forbidden() instead.
empty_transitions_schema = pl.Schema({"lowerlevel": pl.Int64, "upperlevel": pl.Int64, "A": pl.Float64})

# The level columns that write_adata() and add_level_ids_forbidden() read, for an ion whose reader
# gave no levels at all.
empty_levels_schema = pl.Schema(
    {"levelname": pl.String, "energyabovegsinpercm": pl.Float64, "g": pl.Float64, "parity": pl.Int64}
)


class EnergyLevel(t.NamedTuple):
    """One energy level of a data set that gives no other per-level column."""

    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: int | None  # None where the data set gives no parity


class Transition(t.NamedTuple):
    """One bound-bound transition, keyed by zero-based level id."""

    lowerlevel: int
    upperlevel: int
    A: float


class PhixsData(t.NamedTuple):
    """The photoionisation cross sections of one ion, indexed by zero-based level id.

    A data source with no cross sections for the ion gives empty arrays, not zero-filled ones. The
    function iondata.read_ion_data() reads an empty cross section array as "no data" and applies the
    hydrogenic estimate. A reader gives the targets of a level in one of two forms. CMFGEN names
    the upper ion's levels with their fractions, and get_photoiontargetfractions() resolves the
    names after the run reads the upper ion. QUB gives the upper ion's level ids with their
    fractions.
    """

    crosssections: npt.NDArray[np.float64]  # (levelcount, nphixspoints), in Mb
    thresholds_ev: npt.NDArray[np.float64]  # (levelcount,)
    targetconfigs: list[list[tuple[str, float]] | None] | None = None
    targetfractions: list[list[tuple[int, float]]] | None = None


def transition_count_of_level(dftransitions: pl.DataFrame, levelcount: int) -> list[int]:
    """Count the transitions that touch each level, indexed by zero-based level id, for adata.txt.

    Both levels of a transition count, and a level with no transition gets 0. The writer calls
    this on the final transition frame of the ion, after every join and filter, so the counts
    agree with transitiondata.txt. Keyed by id and not by name: a data set can give two levels
    one name, and a name-keyed count merged those levels.
    """
    if dftransitions.is_empty():
        return [0] * levelcount
    levelids = pl.concat([dftransitions["lowerlevel"], dftransitions["upperlevel"]])
    # not an assert: this guards written output, and a level id outside the level list would
    # otherwise raise a bare IndexError below
    if (levelids < 0).any() or (levelids >= levelcount).any():
        msg = f"transitions name level ids {levelids.min()} to {levelids.max()}, but the ion has {levelcount} levels"
        raise ValueError(msg)
    return np.bincount(levelids.to_numpy(), minlength=levelcount).tolist()


def leveltuples_to_pldataframe(energy_levels) -> pl.DataFrame:
    """Convert a list of level tuples (or a DataFrame) into a DataFrame with a zero-based levelid column.

    Level ids are zero-based everywhere in memory. The write_*() functions apply the 1-based
    numbering of the output files.
    """
    if isinstance(energy_levels, pl.DataFrame):
        dflevels = energy_levels
    elif energy_levels:
        dflevels = pl.DataFrame(energy_levels)
    else:
        # An empty list carries no column names, and write_adata() selects the level columns
        # by name. An ion with no levels therefore gets the columns that every reader supplies.
        dflevels = pl.DataFrame(schema=empty_levels_schema)

    if "levelid" not in dflevels.columns:
        dflevels = dflevels.with_row_index(name="levelid")

    dflevels = dflevels.with_columns(pl.col("levelid").cast(pl.Int64))

    # Other code indexes the frame by level id, so a reader-supplied levelid must be contiguous
    # and zero-based. Not an assert: input validation must survive python -O.
    if not dflevels["levelid"].equals(pl.int_range(dflevels.height, dtype=pl.Int64, eager=True)):
        msg = "level ids must be contiguous and start at zero"
        raise ValueError(msg)

    return dflevels


def levelid_of_fileindex_map(fileindices: Iterable[t.Any], sourcename: str) -> dict[int, int]:
    """Map each level's index in its source file to its zero-based level id.

    Readers that re-sort their level list by energy need this, because their transitions still
    name their levels by the file's numbering. Pass the file indices in the sorted level order,
    i.e. fileindices[n] is the file index of the level that ended up at level id n.

    A duplicated file index would overwrite its entry and misroute every transition that
    references it. This function rejects the duplicate here. Otherwise it would appear later as a
    transition on the wrong level.
    """
    fileindices = list(fileindices)
    levelid_of_fileindex = {int(fileindex): levelid for levelid, fileindex in enumerate(fileindices)}

    # not an assert: input validation must survive python -O
    if len(levelid_of_fileindex) != len(fileindices):
        msg = (
            f"Duplicate level indices in {sourcename}: {len(fileindices)} levels but only"
            f" {len(levelid_of_fileindex)} unique indices"
        )
        raise ValueError(msg)

    return levelid_of_fileindex


def resolve_transition_levelids(
    fileindex_lower: t.Any, fileindex_upper: t.Any, levelid_of_fileindex: dict[int, int], sourcename: str
) -> tuple[int, int]:
    """Resolve one transition's file-numbered levels to zero-based level ids, lower id first.

    The function raises on an index that names no level, and does not skip it. A reader whose
    transition and level files disagree about the numbering (0- or 1-based, for example) would
    otherwise drop every transition. It would then write an empty ion without an error.
    """
    try:
        lowerlevel = levelid_of_fileindex[int(fileindex_lower)]
        upperlevel = levelid_of_fileindex[int(fileindex_upper)]
    except KeyError as exc:
        msg = (
            f"Transition {fileindex_lower} -> {fileindex_upper} in {sourcename} names level index {exc.args[0]}."
            f" None of the {len(levelid_of_fileindex)} levels of the level file has that index."
            " The transition file and the level file can disagree about the level numbering."
        )
        raise ValueError(msg) from exc

    # The reader re-sorted the levels by energy, so a transition can name them in either order.
    # transitiondata.txt lists the lower id first.
    return (lowerlevel, upperlevel) if lowerlevel < upperlevel else (upperlevel, lowerlevel)


def ion_log_path(log_folder: str | Path, atomic_number: int, ion_stage: int) -> Path:
    """Path of the per-ion log file. The read pass writes it, and the write pass appends to it."""
    return Path(log_folder, f"{elsymbols[atomic_number].lower()}{ion_stage:d}.txt")


def log_and_print(flog, strout):
    """Write a line to both stdout and this ion's log file."""
    print(strout)
    flog.write(strout + "\n")


def path_for_log(filepath: str | Path) -> str:
    """Render an input data path relative to the repository root where possible.

    The log files must not depend on the location of the repository checkout, so an absolute
    path would be wrong there. Paths outside the repository (some readers load data from elsewhere)
    come back unchanged.
    """
    try:
        return str(Path(filepath).resolve().relative_to(PYDIR.parent))
    except ValueError:
        return str(filepath)


def fortran_float(text: str) -> float:
    """Convert a number that a Fortran program wrote, where the exponent letter can be D."""
    return float(text.replace("D", "E"))


def isfloat(value: t.Any) -> bool:
    """Whether a string parses as a float, with Fortran's D exponent (1.5D-3) permitted.

    The CMFGEN oscillator reader calls this once for each field of every line (5.3M times for the
    cmfgen test set). The function therefore calls replace() only when the field contains a D.
    This avoids a copy of every field.
    """
    try:
        float(value.replace("D", "E") if "D" in value else value)
    except ValueError:
        return False

    return True


compression_extensions = ("", ".zst", ".gz", ".xz")


def find_file_check_extension(filename: str | Path) -> Path | None:
    """Find a data file by its plain name, accepting any of the compressed variants of that name.

    Returns None if neither the plain name nor any compressed form exists. A caller that treats
    a missing file as "no data for this ion" can then say so and does not open the file.
    """
    return next((path for ext in compression_extensions if (path := Path(f"{filename}{ext}")).is_file()), None)


def find_file_check_extension_or_raise(filename: str | Path) -> Path:
    """Find a data file by its plain name, and raise if no form of the name exists."""
    filepath = find_file_check_extension(filename)
    if filepath is None:
        filepaths = [f"{filename}{ext}" for ext in compression_extensions]
        msg = f"Could not find any of the following files:\n  {'\n  '.join(filepaths)}."
        raise FileNotFoundError(msg)

    return filepath


def xopen_check_extension(filename: str | Path, **kwargs: t.Any) -> t.IO[t.Any]:
    """Open a data file, or a compressed variant of the name if the plain file does not exist.

    The data sets ship some files compressed and some not, and the set of compressed files varies
    between downloads. Callers name the plain file, and this function finds the form that is
    present.
    """
    from xopen import xopen

    return xopen(find_file_check_extension_or_raise(filename), **kwargs)


def rewrite_file_as_utf8(filename: str | Path) -> bool:
    """Rewrite a data file as utf-8 if it is not utf-8, and return True if it did so.

    CMFGEN writes an author's name with an accent, which leaves a few of its files in
    iso-8859-1. Neither Python nor polars reads such a file, so a reader that meets one
    converts it once, in place, and reads it again.
    """
    from xopen import xopen

    filepath = find_file_check_extension_or_raise(filename)
    with xopen(filepath, "rb") as fin:
        filebytes = fin.read()

    try:
        filebytes.decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        return False

    print(f"{filepath} is not utf-8. artisatomic rewrites the file as utf-8 from iso-8859-1.")
    # every byte is a character in iso-8859-1, so this decode cannot fail
    text = filebytes.decode("iso-8859-1")
    try:
        with xopen(filepath, "wt", encoding="utf-8") as fout:
            fout.write(text)
    except OSError as exc:
        msg = (
            f"Could not rewrite {filepath} as utf-8: {exc}\n"
            f"Convert the file by hand. Then run this again:\n"
            f"  iconv -f iso-8859-1 -t utf-8 '{filepath}' > tmp && mv tmp '{filepath}'"
        )
        raise RuntimeError(msg) from exc

    return True


def scan_file_lines(filename: str | Path, skip_lines: int = 0) -> pl.LazyFrame:
    """Read a text file into a lazy frame that holds one line in each row of a "line" column.

    polars cuts the columns out of every line at once, with str.slice() or str.extract_all().
    That is much faster than pandas read_fwf(), or read_csv() with a regular expression
    separator. Neither of those has a C parser, so each reads one line at a time in Python.

    The caller names the plain file, as for xopen_check_extension(). polars reads a plain, a
    gzip, or a zstd file itself. It cannot read the xz form, which xopen decompresses into
    memory instead.
    """
    filepath = find_file_check_extension_or_raise(filename)

    csv_options: dict[str, t.Any] = {
        # a separator that the data files cannot contain keeps each whole line in one column
        "separator": "\x1f",
        "has_header": False,
        "new_columns": ["line"],
        "quote_char": None,
        "infer_schema_length": 0,
        "skip_lines": skip_lines,
    }

    if filepath.suffix == ".xz":
        from xopen import xopen

        with xopen(filepath, "rb") as fin:
            return pl.read_csv(io.BytesIO(fin.read()), **csv_options).lazy()

    return pl.scan_csv(filepath, **csv_options)


NIST_IONIZATION_PATH = PYDIR / "nist_ionization.txt.zst"


def parse_nist_ionization_table(text: str) -> tuple[list[str], dict[tuple[int, int], float]]:
    """Parse the tab-separated NIST table of ionisation energies.

    The result holds the provenance lines at the top of the table (the lines that start with "#")
    and the energies, keyed by (atomic_number, ion_stage). The footnotes of the table follow the
    data rows, after a line that starts with "Notes:". A blank energy means that NIST lists none,
    so the ion gets no entry. Any other row that does not parse raises a ValueError.
    """
    lines = text.splitlines()
    provenance = [line.removeprefix("#").strip() for line in lines if line.startswith("#")]
    datalines = [line for line in lines if not line.startswith("#")]
    for index, line in enumerate(datalines):
        if line.startswith("Notes:"):
            datalines = datalines[:index]
            break

    dfnist = pl.read_csv(
        io.StringIO("\n".join(datalines)),
        separator="\t",
        columns=["At. num", "Ion Charge", "Ionization Energy (a) (eV)"],
        infer_schema=False,
    ).fill_null("")
    energies = {}
    for atomic_number, ion_charge, ioniz_ev in dfnist.iter_rows():
        if not ioniz_ev:
            continue  # a blank energy means that NIST lists none
        try:
            key = (int(atomic_number), int(ion_charge) + 1)
            energy_ev = float(ioniz_ev)
        except ValueError as err:
            msg = f"The NIST table has a row that does not parse: {atomic_number!r} {ion_charge!r} {ioniz_ev!r}"
            raise ValueError(msg) from err
        if not math.isfinite(energy_ev):
            msg = f"The NIST table has a non-finite energy for Z={atomic_number} charge {ion_charge}"
            raise ValueError(msg)
        energies[key] = energy_ev
    return provenance, energies


@lru_cache(maxsize=1)
def _read_nist_ionization_table() -> tuple[list[str], dict[tuple[int, int], float]]:
    """Read the NIST table that the package ships, artisatomic/nist_ionization.txt.zst."""
    from xopen import xopen

    with xopen(NIST_IONIZATION_PATH, encoding="utf-8") as fnist:
        return parse_nist_ionization_table(fnist.read())


def get_nist_ionization_energies_ev() -> dict[tuple[int, int], float]:
    """Get a dictionary where dictioniz[(atomic_number, ion_stage)] = ionization_energy_ev."""
    return _read_nist_ionization_table()[1]


def get_nist_ionization_provenance() -> list[str]:
    """Get the provenance lines of the NIST table: the source, the query, and the date."""
    return _read_nist_ionization_table()[0]


_process_pool: ProcessPoolExecutor | None = None


def get_process_pool() -> ProcessPoolExecutor:
    """Get the one process pool for the whole run, and create it on the first use.

    A new pool costs about 0.6 s however small the batch, because "spawn" makes every worker
    re-import this package and its numpy/polars/pandas dependencies. A build asks for one pool
    for each ion and each photoionisation file. A pool for each call therefore spent most of its
    time in startup. The peak memory does not change (the same workers, alive for longer).
    """
    global _process_pool
    if _process_pool is None:
        # an explicit spawn context rather than mp.set_start_method(force=True), which reached
        # outside this function to change the default for the whole process
        _process_pool = ProcessPoolExecutor(mp_context=mp.get_context("spawn"))
        atexit.register(_process_pool.shutdown, wait=False, cancel_futures=True)

    return _process_pool


def parallel_map[ResultType](
    fn: Callable[..., ResultType],
    *iterables: Iterable[t.Any],
    chunksize: int | None = None,
) -> list[ResultType]:
    """Execute a parallel map with a progress bar, with threads on a free-threading python and processes otherwise.

    Every iterable must have the same length. Executor.map() and thread_map() stop at the
    shortest iterable, as zip() does. A short iterable would therefore drop the tail of the work
    without an error, and the three paths below would not drop the same items.

    The signature accepts no other keywords. thread_map() accepts a dozen more that shape the
    pool it builds (max_workers, timeout, mp_context, ...). The run-wide pool from
    get_process_pool() cannot honour them, and tqdm() on the other path rejects them. A forwarded
    keyword would apply the caller's intent on a free-threading build and raise on a stock one.
    """
    # use a thread pool if we have no GIL (free threading)
    use_multiprocessing = sys._is_gil_enabled()  # ruff: ignore[private-member-access]

    # Materialise the iterables to measure the work. The chunk size and the serial cutoff below
    # both need a length, and the callers pass views and generators.
    lists = [list(iterable) for iterable in iterables]
    lengths = [len(x) for x in lists]
    # not an assert: this check decides how much of the work runs, so it must survive python -O
    if len(set(lengths)) > 1:
        msg = f"parallel_map() received iterables of different lengths: {lengths}"
        raise ValueError(msg)
    nitems = lengths[0] if lengths else 0

    # Even with the pool already up, a handful of items costs more in IPC than the work itself.
    # This path does them here (readqubdata reduces four cross section tables at a time).
    if nitems <= 32:
        return list(itertools.starmap(fn, zip(*lists, strict=True)))

    if chunksize is None:
        # Without a chunk size, items go to the workers one at a time, and the IPC for each item
        # costs more than the work. Above 1000 items, tqdm warns about it.
        chunksize = max(1, nitems // (mp.cpu_count() * 4))

    # disable=None means "disable on non-TTY". The bar is for a person who watches a build, and
    # it redraws by carriage return. A redirected run or a CI capture therefore got a line of
    # partial bars mixed with the real output for every call. The thread_map() path forwards this
    # to tqdm.
    if use_multiprocessing:
        from tqdm import tqdm

        results = list(
            tqdm(get_process_pool().map(fn, *lists, chunksize=chunksize), total=nitems, disable=None)  # ty:ignore[no-matching-overload]
        )
    else:
        from tqdm.contrib.concurrent import thread_map

        results = thread_map(fn, *lists, chunksize=chunksize, total=nitems, disable=None)

    assert isinstance(results, list)
    return results


def drop_handlers(list_ions: list[tuple[int, str]]) -> list[int]:
    """Replace [(ion_stage1, 'handler1'), (ion_stage2, 'handler2')] with [ion_stage1, ion_stage2]."""
    return [ion_stage for ion_stage, _handler in list_ions]


def sort_ion_handlers(
    ion_handlers: list[tuple[int, list[tuple[int, str]]]],
) -> list[tuple[int, list[tuple[int, str]]]]:
    """Sort by atomic number, and each element's ions by ion stage.

    process_files() relies on the ion stages in order from lowest to highest to identify the top
    ion and to find each ion's photoionisation target. This function normalises the order before
    main() writes the handler list to artisatomicionhandlers.json and passes it to
    write_compositionfile().

    This function rejects a duplicated element or ion stage, because the code downstream trusts
    the list. The ion count of write_compositionfile() is max - min + 1, so it cannot see a
    duplicate. A duplicate would therefore make compositiondata.txt disagree with the other output
    files.
    """
    atomic_numbers = [atomic_number for atomic_number, _listions in ion_handlers]
    if len(set(atomic_numbers)) != len(atomic_numbers):
        duplicates = sorted({z for z in atomic_numbers if atomic_numbers.count(z) > 1})
        msg = f"The ion handlers list contains more than one entry for Z={duplicates}."
        raise ValueError(msg)
    for atomic_number, listions in ion_handlers:
        ion_stages = [ion_stage for ion_stage, _handler in listions]
        if len(set(ion_stages)) != len(ion_stages):
            duplicates = sorted({s for s in ion_stages if ion_stages.count(s) > 1})
            msg = f"The ion handlers list contains ion stage {duplicates} more than once for Z={atomic_number}."
            raise ValueError(msg)

    return sorted(
        ((atomic_number, sorted(listions, key=operator.itemgetter(0))) for atomic_number, listions in ion_handlers),
        key=operator.itemgetter(0),
    )


def add_handler_if_not_set(
    ion_handlers: list[tuple[int, list[tuple[int, str]]]],
    atomic_number: int | str,
    ion_stage: int | str,
    handler: str,
) -> list[tuple[int, list[tuple[int, str]]]]:
    """Return a new ion_handlers list with (ion_stage, handler) added unless the ion is already present.

    The function does not modify the input list, so the caller must use the return value.
    """
    # Readers derive these from pandas/numpy data, and json.dump() in main() cannot serialise
    # numpy integers. Normalise them here and not in each caller.
    atomic_number = int(atomic_number)
    ion_stage = int(ion_stage)

    ion_handlers_out: list[tuple[int, list[tuple[int, str]]]] = []
    found_element = False
    for tmp_atomic_number, list_ions_handlers in ion_handlers:
        list_ions_handlers_out: list[tuple[int, str]] = list(list_ions_handlers)
        if tmp_atomic_number == atomic_number:
            found_element = True
            if ion_stage not in drop_handlers(list_ions_handlers_out):
                list_ions_handlers_out.append((ion_stage, handler))
        ion_handlers_out.append((tmp_atomic_number, list_ions_handlers_out))

    if not found_element:
        ion_handlers_out.append((atomic_number, [(ion_stage, handler)]))

    return sort_ion_handlers(ion_handlers_out)

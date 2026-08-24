"""Element data, physical constants, and small utilities shared by the data-source readers."""

import atexit
import io
import itertools
import math
import multiprocessing as mp
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import pandas as pd
import polars as pl

PYDIR = Path(__file__).parent.resolve()
atomicdata = pd.read_csv(PYDIR / "atomic_properties.txt", sep=r"\s+", comment="#")
# estimate unknown atomic mass as Z / 0.45. Only the mass column: a row-wise fillna() filled every
# column with it, including the densities and radii, which are not masses and are not read here.
atomicdata["mass"] = atomicdata["mass"].fillna(atomicdata["number"] / 0.45)
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

# the single copy of each constant for the whole package
ryd_to_ev = 13.605693122994232
hc_in_ev_cm = 0.0001239841984332003
hc_in_ev_angstrom = 12398.419843320025
h_in_ev_seconds = 4.135667696923859e-15


def split_element_ionstage_str(ionstr: str) -> tuple[int, int]:
    """Split a string like 'FeII' into (atomic_number, ion_stage).

    Splitting on `ionstr.rstrip("IVX")` destroys the symbols of the elements whose symbols are
    made only of those letters: V (vanadium) and I (iodine). Instead find the split point where
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


def leveltuples_to_pldataframe(energy_levels) -> pl.DataFrame:
    """Convert a list of level tuples (or a DataFrame) into a DataFrame with a zero-based levelid column.

    Level ids are zero-based everywhere in memory; the 1-based numbering of the output files is
    applied by the write_*() functions.
    """
    dflevels = energy_levels if isinstance(energy_levels, pl.DataFrame) else pl.DataFrame(energy_levels)

    if "levelid" not in dflevels.columns:
        dflevels = dflevels.with_row_index(name="levelid")

    dflevels = dflevels.with_columns(pl.col("levelid").cast(pl.Int64))

    # the frame is indexed by level id elsewhere, so a reader-supplied levelid must be contiguous
    # and zero-based. Not an assert: input validation must survive python -O.
    if not dflevels["levelid"].equals(pl.int_range(dflevels.height, dtype=pl.Int64, eager=True)):
        msg = "level ids must be contiguous and start at zero"
        raise ValueError(msg)

    return dflevels


def levelid_of_fileindex_map(fileindices: Iterable[t.Any], sourcename: str) -> dict[int, int]:
    """Map each level's index in its source file to its zero-based level id.

    Readers whose level list is re-sorted by energy need this, because their transitions still
    name their levels by the file's numbering. Pass the file indices in the sorted level order,
    i.e. fileindices[n] is the file index of the level that ended up at level id n.

    A duplicated file index would overwrite its entry and misroute every transition referencing
    it, so that is rejected here rather than surfacing as a transition on the wrong level.
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

    Raises rather than skipping an index that names no level: a reader whose transition and level
    files disagree about the numbering (0- vs 1-based, say) would otherwise drop every transition
    and write a silently empty ion instead of failing.
    """
    try:
        lowerlevel = levelid_of_fileindex[int(fileindex_lower)]
        upperlevel = levelid_of_fileindex[int(fileindex_upper)]
    except KeyError as exc:
        msg = (
            f"Transition {fileindex_lower} -> {fileindex_upper} in {sourcename} names level index {exc.args[0]},"
            f" which is not one of the {len(levelid_of_fileindex)} levels read. The transition and level files"
            " may disagree about the level numbering."
        )
        raise ValueError(msg) from exc

    # the levels were re-sorted by energy, so a transition can name them either way round.
    # transitiondata.txt is written with the lower id first.
    return (lowerlevel, upperlevel) if lowerlevel < upperlevel else (upperlevel, lowerlevel)


def ion_log_path(log_folder: str | Path, atomic_number: int, ion_stage: int) -> Path:
    """Path of the per-ion log file, written by the reading pass and appended to by the writing pass."""
    return Path(log_folder, f"{elsymbols[atomic_number].lower()}{ion_stage:d}.txt")


def log_and_print(flog, strout):
    """Write a line to both stdout and this ion's log file."""
    print(strout)
    flog.write(strout + "\n")


def path_for_log(filepath: str | Path) -> str:
    """Render an input data path relative to the repository root where possible.

    The log files are compared by checksum in CI, so an absolute path would make them depend on
    where the repository happens to be checked out. Paths outside the repository (some readers
    load data from elsewhere) are returned unchanged.
    """
    try:
        return str(Path(filepath).resolve().relative_to(PYDIR.parent))
    except ValueError:
        return str(filepath)


def isfloat(value: t.Any) -> bool:
    """Whether a string parses as a float, accepting Fortran's D exponent (1.5D-3).

    Called once per field of every line of the CMFGEN oscillator files (5.3M times for the cmfgen
    test set), so the replace() is done only when there is a D to replace rather than allocating
    a copy of every field.
    """
    try:
        float(value.replace("D", "E") if "D" in value else value)
    except ValueError:
        return False

    return True


compression_extensions = ("", ".zst", ".gz", ".xz")


def find_file_check_extension(filename: str | Path) -> Path | None:
    """Find a data file by its plain name, accepting any of the compressed variants of that name.

    Returns None if neither the plain name nor any compressed form exists, so that callers which
    treat a missing file as "no data for this ion" can say so without opening it.
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
    """Open a data file, trying the compressed variants of the name if it does not exist.

    The data sets ship some files compressed and some not, and which ones varies between
    downloads, so callers name the plain file and this finds whichever form is present.
    """
    from xopen import xopen

    return xopen(find_file_check_extension_or_raise(filename), **kwargs)


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
    """Parse the tab-separated NIST table of ionization energies.

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

    dfnist = pd.read_csv(
        io.StringIO("\n".join(datalines)),
        sep="\t",
        usecols=["At. num", "Ion Charge", "Ionization Energy (a) (eV)"],
        dtype=str,
        keep_default_na=False,
    )
    energies = {}
    for atomic_number, ion_charge, ioniz_ev in dfnist.itertuples(index=False):
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
    """Get the one process pool for the whole run, creating it on first use.

    Building a pool costs about 0.6 s however small the batch, because "spawn" makes every worker
    re-import this package and its numpy/polars/pandas dependencies. A build asks for one pool per
    ion per photoionisation file, so a pool per call spent most of its time starting up. Peak
    memory is unchanged (the same workers, alive for longer).
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
    """Execute a parallel map with a progress bar using either multithreading (for free-threading python) or multiprocessing.

    Every iterable must be the same length. Executor.map() and thread_map() stop at the shortest,
    as zip() does, so an accidentally short one would silently drop the tail of the work instead
    of failing, and the three paths below would not even drop the same items.

    The keywords above are all there are. thread_map() absorbs a dozen more that shape the pool
    it builds (max_workers, timeout, mp_context, ...), none of which the run-wide pool from
    get_process_pool() can honour, and which tqdm() on the other path rejects: forwarding them
    would apply the caller's intent on a free-threading build and raise on a stock one.
    """
    # use a thread pool if we have no GIL (free threading)
    use_multiprocessing = sys._is_gil_enabled()  # ruff: ignore[private-member-access]

    # materialise so the work can be sized: the chunk size and the serial cutoff below both need
    # a length, and the callers pass views and generators
    lists = [list(iterable) for iterable in iterables]
    lengths = [len(x) for x in lists]
    # not an assert: this decides how much of the work is done, so it must survive python -O
    if len(set(lengths)) > 1:
        msg = f"parallel_map() was given iterables of different lengths: {lengths}"
        raise ValueError(msg)
    nitems = lengths[0] if lengths else 0

    # even with the pool already up, handing a handful of items to it costs more in IPC than doing
    # them here (readqubdata reduces four cross-section tables at a time)
    if nitems <= 32:
        return list(itertools.starmap(fn, zip(*lists, strict=True)))

    if chunksize is None:
        # without a chunk size, items go to the workers one at a time and the IPC per item costs
        # more than the work; tqdm warns about it above 1000 items
        chunksize = max(1, nitems // (mp.cpu_count() * 4))

    # disable=None means "disable on non-TTY": the bar is for someone watching a build, and it
    # redraws by carriage return, so a redirected run or a CI capture got a line of half-drawn
    # bars interleaved with the real output for every call. thread_map() forwards this to tqdm.
    if use_multiprocessing:
        from tqdm import tqdm

        results = list(
            tqdm(get_process_pool().map(fn, *lists, chunksize=chunksize), total=nitems, disable=None)  # ty:ignore[no-matching-overload]
        )
    else:
        from tqdm.contrib.concurrent import thread_map

        results = thread_map(fn, *lists, chunksize=chunksize, total=nitems, disable=None)  # type: ignore[arg-type] # zuban: ignore[no-untyped-call]

    assert isinstance(results, list)
    return results

"""Element data, physical constants, and small utilities shared by the data-source readers."""

import contextlib
import multiprocessing as mp
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import pandas as pd
import polars as pl

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
    """Whether a string parses as a float, accepting Fortran's D exponent (1.5D-3)."""
    try:
        float(value.replace("D", "E"))
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


def xopen_check_extension(filename: str | Path, **kwargs: t.Any) -> t.IO[t.Any]:
    """Open a data file, trying the compressed variants of the name if it does not exist.

    The data sets ship some files compressed and some not, and which ones varies between
    downloads, so callers name the plain file and this finds whichever form is present.
    """
    from xopen import xopen

    filepath = find_file_check_extension(filename)
    if filepath is None:
        filepaths = [f"{filename}{ext}" for ext in compression_extensions]
        msg = f"Could not find any of the following files:\n  {'\n  '.join(filepaths)}."
        raise FileNotFoundError(msg)

    return xopen(filepath, **kwargs)


# split a list into evenly sized chunks
def chunks[T](listin: list[T], chunk_size: int) -> list[list[T]]:
    """Split a list into consecutive chunks of at most chunk_size items."""
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
            dictioniz[int(atomic_number), ion_stage] = ioniz_ev
    return dictioniz


def parallel_map[ResultType](
    fn: Callable[..., ResultType],
    *iterables: Iterable[t.Any],
    **kwargs: t.Any,
) -> list[ResultType]:
    """Execute a parallel map with a progress bar using either multithreading (for free-threading python) or multiprocessing."""
    # use a thread pool if we have no GIL (free threading)
    use_multiprocessing = sys._is_gil_enabled()  # ruff: ignore[private-member-access]

    if use_multiprocessing:
        mp.set_start_method("spawn", force=True)
        from tqdm.contrib.concurrent import process_map

        results = process_map(fn, *iterables, **kwargs)  # type: ignore[arg-type] # zuban: ignore[no-untyped-call]
    else:
        from tqdm.contrib.concurrent import thread_map

        results = thread_map(fn, *iterables, **kwargs)  # type: ignore[arg-type] # zuban: ignore[no-untyped-call]

    assert isinstance(results, list)
    return results

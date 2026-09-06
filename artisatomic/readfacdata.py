"""Read levels and transitions from FAC and cFAC output, an early version of the Floers+25 data."""

import os
import re
import string
from pathlib import Path

import polars as pl

from artisatomic.base import add_handler_if_not_set
from artisatomic.base import elsymbols
from artisatomic.base import EnergyLevel
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import levelid_of_fileindex_map
from artisatomic.base import log_and_print
from artisatomic.base import path_for_log
from artisatomic.base import resolve_transition_levelids
from artisatomic.base import roman_numerals
from artisatomic.base import scan_file_lines
from artisatomic.base import split_element_ionstage_str
from artisatomic.base import Transition
from artisatomic.levelnames import parse_orbital_n

USE_CALIBRATED = True


def get_basepath() -> Path:
    """Return the directory that holds the OptimizedFAC lanthanide data.

    The data is not part of this repository. ARTISATOMIC_FAC_PATH overrides the directory that
    this function searches, as ARTISATOMIC_LISBON_PATH does for the Lisbon files. The default is
    the Google Drive mount path. That is where the shared drive appears on the machine of the
    original author.
    """
    calibstr = "_calibrated" if USE_CALIBRATED else ""
    default = Path.home() / "Google Drive/Shared drives/Atomic Data Group/OptimizedFACdata"
    return Path(os.environ.get("ARTISATOMIC_FAC_PATH", default)) / f"OptimizedFAC_lanthanides{calibstr}"


def parse_fixed_width(
    filename: Path | str, skip_lines: int, columns: list[tuple[str, int, int, type[pl.DataType]]]
) -> pl.DataFrame:
    """Cut fixed-width columns out of an FAC or cFAC ascii table.

    Each entry of columns names the column, its first character, its last character (exclusive),
    and its type. The table starts after skip_lines lines that hold text.

    The count of skip_lines does not include a blank line. FAC writes one in the header of every
    file, and another after the last row of a table. A line of spaces counts as blank, because
    the columns of such a line are all empty.

    See scan_file_lines() for why this repository cuts fixed-width columns this way.
    """
    return (
        scan_file_lines(filename)
        # keep a line that holds a character other than a space. A blank line gives null here,
        # which the filter drops
        .filter(pl.col("line").str.contains(r"\S"))
        .slice(skip_lines)
        .select(
            # a blank field, and a line too short to reach the field, both give a null
            pl.col("line").str.slice(start, end - start).str.strip_chars().replace("", None).cast(dtype).alias(name)
            for name, start, end, dtype in columns
        )
        .collect()
    )


def check_no_nulls(table: pl.DataFrame, sourcename: str) -> pl.DataFrame:
    """Stop the run if a column of a parsed FAC table holds a null.

    A null means that a line stopped before the end of the column, or that the field was blank.
    Every column that this reader cuts must have a value in every row. A null that passes here
    reaches int() or the output writer, which report neither the file nor the column.
    """
    # not an assert: a truncated data file must stop the run, and must name the column
    nullcounts = {name: count for name, count in table.null_count().row(0, named=True).items() if count > 0}
    if nullcounts:
        msg = f"{sourcename} has rows with no value: {nullcounts}. A line is shorter than its columns."
        raise ValueError(msg)

    return table


def GetLevels_FAC(filename: Path | str) -> pl.DataFrame:
    """Parse the level table of an FAC ascii output file (fixed-width, FAC column layout)."""
    columns: list[tuple[str, int, int, type[pl.DataType]]] = [
        ("Ilev", 0, 7, pl.Int64),
        ("Energy_ev", 14, 30, pl.Float64),
        ("P", 30, 31, pl.Int64),
        ("2J", 38, 43, pl.Int64),
        ("Configs", 76, 125, pl.String),
    ]
    levels_FAC = parse_fixed_width(filename, skip_lines=11, columns=columns)

    # the FAC layout separates the parts of a configuration with a full stop
    return finish_levels(levels_FAC.with_columns(Config=pl.col("Configs").str.replace_all(".", " ", literal=True)))


def GetLevels_cFAC(filename: Path | str) -> pl.DataFrame:
    """Parse the level table of a cFAC ascii output file, whose columns differ from FAC's."""
    columns: list[tuple[str, int, int, type[pl.DataType]]] = [
        ("Ilev", 0, 7, pl.Int64),
        ("Energy_ev", 14, 30, pl.Float64),
        ("P", 30, 31, pl.Int64),
        ("2J", 38, 43, pl.Int64),
        ("Configs", 43, 150, pl.String),
    ]
    levels_cFAC = parse_fixed_width(filename, skip_lines=11, columns=columns)

    # the cFAC layout puts a second field after the configuration, with two or more spaces between
    return finish_levels(levels_cFAC.with_columns(Config=pl.col("Configs").str.replace(r"\s{2,}.*$", "")))


def finish_levels(levels: pl.DataFrame) -> pl.DataFrame:
    """Derive the columns that read_levels_data() takes, the same way for the FAC and cFAC layouts."""
    check_no_nulls(levels, "The FAC levels file")

    # remove only a lone occupation of 1 ("6s1" -> "6s"); occupations of 10-14 keep their digits.
    # The regex engine of polars supports no lookaround, so this runs in Python. A level table
    # holds a few thousand rows, so the cost is small
    lone_occupation_1 = re.compile(r"(?<=[spdfg])1(?![0-9])")
    return levels.select(
        pl.col("Ilev"),
        pl.col("Config").map_elements(lambda s: lone_occupation_1.sub("", s), return_dtype=pl.String),
        pl.col("P"),
        (pl.col("2J") + 1).alias("g"),
        pl.col("Energy_ev"),
        # numpy divides, not polars: polars multiplies by the reciprocal, which differs by one
        # unit in the last place. write_adata() prints 16 decimals, so that difference shows
        energypercm=pl.Series(levels["Energy_ev"].to_numpy() / hc_in_ev_cm),
    )


def GetLevels(filename: Path | str) -> pl.DataFrame:
    """Get a dataframe of every energy level in the ascii level output of FAC or cFAC.

    The caller drops the levels above the ionisation energy and keeps their Ilev values. The
    values show whether a transition names a dropped level or an unknown level.
    """
    headerlines: list[str] = []
    with Path(filename).open(encoding="utf-8") as f:
        headerlines.extend(f.readline() for _ in range(10))

    # headerlines[7] holds the ground state and headerlines[5] the ion charge. This function needs neither.
    version_FAC = headerlines[0].split(" ")[0]
    print("FAC/cFAC: ", version_FAC)
    if version_FAC == "FAC":
        levels = GetLevels_FAC(filename)
    elif version_FAC == "cFAC":
        levels = GetLevels_cFAC(filename)
    else:
        msg = "No FAC-like code detected on output file"
        raise ValueError(msg)

    return levels


def GetLines_FAC(filename: Path | str) -> pl.DataFrame:
    """Parse the transition table of an FAC ascii output file."""
    # the A column takes the leading "-" of a negative Monopole in the last column, which the
    # cast cannot read. strip_chars_end() removes it. It strips the right only, so a negative A
    # keeps its sign
    columns: list[tuple[str, int, int, type[pl.DataType]]] = [
        ("Upper", 0, 7, pl.Int64),
        ("Lower", 11, 17, pl.Int64),
        ("A", 49, 63, pl.String),
    ]
    lines = parse_fixed_width(filename, skip_lines=12, columns=columns).with_columns(
        pl.col("A").str.strip_chars_end(" -").replace("", None).cast(pl.Float64)
    )
    return check_no_nulls(lines, "The FAC transitions file")


def GetLines_cFAC(filename: Path | str) -> pl.DataFrame:
    """Parse the transition table of a cFAC ascii output file."""
    columns: list[tuple[str, int, int, type[pl.DataType]]] = [
        ("Upper", 0, 6, pl.Int64),
        ("Lower", 10, 16, pl.Int64),
        ("A", 61, 75, pl.Float64),
    ]
    return check_no_nulls(parse_fixed_width(filename, skip_lines=12, columns=columns), "The cFAC transitions file")


def GetLines(filename: Path | str) -> pl.DataFrame:
    """Get a dataframe of the transitions extracted from ascii level output of cFAC and csv and dat files.

    Parameters
    ----------
    filename : str
        Filename of cFAC ascii output for the transitions
    """
    headerlines: list[str] = []
    with Path(filename).open(encoding="utf-8") as f:
        headerlines.extend(f.readline() for _ in range(11))
    # headerlines[8], [10] and [5] hold the ground state, multipole and ion charge. This function needs none.
    version_FAC = headerlines[0].split(" ")[0]

    if version_FAC == "FAC":
        lines = GetLines_FAC(filename)
    elif version_FAC == "cFAC":
        lines = GetLines_cFAC(filename)
    else:
        msg = "No FAC-like code detected on output file"
        raise ValueError(msg)

    return lines


def extend_ion_list(ion_handlers):
    """Add every ion with an FAC data file to ion_handlers under the "fac" handler."""
    basepath = get_basepath()
    # not an assert: this reports a missing data directory and must survive python -O. It also
    # names the environment variable, which a silent failure of the glob would not
    if not basepath.is_dir():
        msg = (
            f"FAC data directory {basepath} not found."
            " Set ARTISATOMIC_FAC_PATH to the directory that holds the OptimizedFAC_lanthanides* folders."
        )
        raise FileNotFoundError(msg)

    for s in basepath.glob("**/*.lev.asc"):
        ionstr = s.parts[-1].lstrip(string.digits).removesuffix(".lev.asc").removesuffix("_calib")
        atomic_number, ion_stage = split_element_ionstage_str(ionstr)
        ion_handlers = add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "fac")

    # add_handler_if_not_set() keeps the list sorted by atomic number, matching the other readers
    return ion_handlers


def read_levels_data(dflevels):
    """Convert the FAC level table to level tuples, in the energy order of the sorted frame.

    Also returns the map from the file's Ilev to the zero-based level id, which read_lines_data()
    needs because the sort by energy reorders the levels.
    """
    # sort by energy, so the level order is now the frame's order alone. The sort is stable, so
    # levels of one energy keep the file's order. Their ids then do not depend on the sort algorithm
    dflevels = dflevels.sort("energypercm", maintain_order=True)

    energy_levels = [
        # Config is not unique (levels of one configuration differ in J), so append the FAC level
        # index. The configuration stays first, for get_level_valence_n() and the adata.txt comment.
        EnergyLevel(
            levelname=f"{row['Config']} Ilev={int(row['Ilev'])}",
            parity=row["P"],
            g=row["g"],
            energyabovegsinpercm=float(row["energypercm"]),
        )
        for row in dflevels.iter_rows(named=True)
    ]

    return energy_levels, levelid_of_fileindex_map(dflevels["Ilev"], "the FAC levels file")


def read_lines_data(dflines, ilev_enlevelindex_map, ilevs_above_ionization: set[int], flog):
    """Convert FAC lines to transitions referencing zero-based level ids.

    The reader skips a line that names a level above the ionisation energy, because the level
    list stops there. A line that names an Ilev that the level file does not have is an error.
    The two files then disagree about the numbering, and a skip would empty the ion without a
    message. The reader orders the two levels with the lower id first.
    """
    transitions = []
    skipped_count = 0

    for row in dflines.iter_rows(named=True):
        if int(row["Lower"]) in ilevs_above_ionization or int(row["Upper"]) in ilevs_above_ionization:
            skipped_count += 1
            continue

        # not an assert: this decides between which levels the output writes a transition, so it
        # must survive python -O. It also names the offending Ilev values instead of a bare failure
        lowerlevel, upperlevel = resolve_transition_levelids(
            row["Lower"], row["Upper"], ilev_enlevelindex_map, "the FAC transitions file"
        )

        transitions.append(Transition(lowerlevel=lowerlevel, upperlevel=upperlevel, A=row["A"]))

    if skipped_count > 0:
        log_and_print(
            flog, f"WARNING: skipped {skipped_count:d} transitions that reference a level above the ionisation energy"
        )

    return transitions


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the FAC data set, an early version of the Floers+25 calibrated data."""
    elsym = elsymbols[atomic_number]
    ion_stage_roman = roman_numerals[ion_stage]

    ionstr = f"{atomic_number}{elsym}{ion_stage_roman}{'_calib' if USE_CALIBRATED else ''}"
    ion_folder = get_basepath() / ionstr
    levels_file = ion_folder / f"{ionstr}.lev.asc"
    lines_file = ion_folder / f"{ionstr}.tr.asc"

    if atomic_number == 92 and ion_stage in {2, 3}:
        # U II and U III come from a separate convergence study, which sits beside the
        # OptimizedFAC folders rather than inside them
        ionstr = f"{elsym}{ion_stage_roman}_convergence_t22_n30_calibrated"
        ion_folder = get_basepath().parent.parent / "Paper_Nd_U" / "FAC" / ionstr
        levels_file = ion_folder / f"{ionstr}.lev.asc"
        lines_file = ion_folder / f"{ionstr}.tr.asc"

    log_and_print(
        flog,
        f"Reading FAC/cFAC data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman}) from"
        f" {path_for_log(ion_folder)}",
    )

    ionization_energy_in_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]

    if not levels_file.is_file():
        msg = f"FAC levels file {levels_file} not found"
        raise FileNotFoundError(msg)
    dfalllevels = GetLevels(filename=levels_file)
    # drop the levels above the ionisation energy, but keep their Ilev values. With them,
    # read_lines_data() knows whether a transition names a dropped level or an unknown level
    # fill_null(False): a null energy compares as null, which would drop the level from the kept
    # levels and from the set below. The level would then be in neither, and every transition that
    # names it would stop the run with a message about the transitions file
    above_ionization = (pl.col("energypercm") > (ionization_energy_in_ev / hc_in_ev_cm)).fill_null(False)
    ilevs_above_ionization = {int(ilev) for ilev in dfalllevels.filter(above_ionization)["Ilev"]}
    dflevels = dfalllevels.filter(~above_ionization)

    # the map associates the file indices with the energy-sorted level ids (0 indexed)
    energy_levels, ilev_enlevelindex_map = read_levels_data(dflevels)

    log_and_print(flog, f"Read {len(energy_levels):d} levels")

    if not lines_file.is_file():
        msg = f"FAC transitions file {lines_file} not found"
        raise FileNotFoundError(msg)
    dflines = GetLines(filename=lines_file)

    transitions = read_lines_data(dflines, ilev_enlevelindex_map, ilevs_above_ionization, flog)

    log_and_print(flog, f"Read {len(transitions)} transitions")

    return ionization_energy_in_ev, energy_levels, transitions


def get_level_valence_n(levelname: str) -> int | None:
    """Principal quantum number of the valence electron, read from an FAC level name.

    Returns None for a name that it cannot parse. The caller, match_hydrogenic_phixs(), then
    gives the level no estimate and writes a warning to the ion log.

    Kept separate from the other readers' versions. Each data source names its levels
    differently, so a shared parser would have to guess the convention of each name.
    """
    # level names are "<configuration> Ilev=<index>", and the configuration itself contains
    # spaces. Drop the index suffix first, then take the last orbital
    part = levelname.split(" Ilev=", maxsplit=1)[0].rsplit(" ", maxsplit=1)[-1]
    return parse_orbital_n(part)

"""Read levels and transitions from the Floers+25 data set, calibrated or uncalibrated."""

import re
import string
from pathlib import Path

import polars as pl

from artisatomic.base import add_handler_if_not_set
from artisatomic.base import compression_extensions
from artisatomic.base import elsymbols
from artisatomic.base import find_file_check_extension
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import log_and_print
from artisatomic.base import PYDIR
from artisatomic.base import roman_numerals
from artisatomic.base import scan_file_lines
from artisatomic.base import split_element_ionstage_str
from artisatomic.base import TESTMODE
from artisatomic.base import xopen_check_extension
from artisatomic.levelnames import parse_orbital_n


def get_basepath(withforbidden: bool) -> Path:
    """Directory that holds the Floers+25 level and transitions files.

    The private OutputFiles_withforbidden directory holds the newer data with forbidden
    transitions. The public OutputFiles directory holds the published data. In the test mode,
    every request resolves to the test_sample directory from testdata.tar.xz.
    """
    datapath = PYDIR / ".." / "atomic-data-floers25"
    if TESTMODE:
        return datapath / "test_sample"
    return datapath / ("OutputFiles_withforbidden" if withforbidden else "OutputFiles")


def extend_ion_list(ion_handlers, calibrated=True):
    """Add every ion with a Floers+25 data file to ion_handlers.

    The handler priority from highest to lowest is floers25calibwithforbidden, floers25calib,
    and floers25uncalib. An ion with no calibrated data gets its uncalibrated version. With
    calibrated=False, this function adds only the uncalibrated handler.
    """
    basepath_public = get_basepath(withforbidden=False)
    basepath_private = get_basepath(withforbidden=True)
    # not an assert: input validation must survive python -O
    if not basepath_public.is_dir() and not basepath_private.is_dir():
        searched = " or ".join(str(p) for p in dict.fromkeys((basepath_public, basepath_private)))
        msg = (
            f"Found no Floers+25 data directory at {searched}. Run"
            " atomic-data-floers25/setup_floers25_data.sh to download the data. For the test"
            " mode, extract testdata.tar.xz instead."
        )
        raise FileNotFoundError(msg)

    # in the test mode the private path equals the public path, so no private search may run
    use_private = not TESTMODE and basepath_private.is_dir()
    if not use_private and not TESTMODE:
        print(f"Floers+25: skipped the private directory {basepath_private} because it does not exist")

    # the searches run in priority order, because add_handler_if_not_set() keeps the first
    # handler that matches an ion
    searches: list[tuple[str, str, Path]] = []
    if calibrated and use_private:
        searches.append(("floers25calibwithforbidden", "calib", basepath_private))
    if calibrated:
        searches.append(("floers25calib", "calib", basepath_public))
    if use_private:
        searches.append(("floers25uncalib", "uncalib", basepath_private))
    searches.append(("floers25uncalib", "uncalib", basepath_public))

    for handlername, calibstr, basepath in searches:
        for ext in compression_extensions:
            for s in basepath.glob(f"*_levels_{calibstr}.txt{ext}"):
                ionstr = s.name.lstrip(string.digits).split("_")[0]
                atomic_number, ion_stage = split_element_ionstage_str(ionstr)
                ion_handlers = add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, handlername)

    return ion_handlers


def read_table_header(filepath: Path) -> tuple[list[str], int]:
    """Read the column names of the data table that starts after the third '----' line.

    The second value is the number of lines before the first data row, which the scan skips.
    """
    linecount = 0
    dashrowcount = 0
    with xopen_check_extension(filepath) as f:
        for line in f:
            linecount += 1
            if line.startswith("--"):
                dashrowcount += 1
                if dashrowcount == 3:
                    break
        # a blank line between the rule and the names holds no column name
        columns: list[str] = []
        while dashrowcount == 3 and not columns:
            line = f.readline()
            linecount += 1
            if not line:
                break
            columns = line.split()

    if dashrowcount < 3 or not columns:
        msg = f"Did not find the expected data table in {filepath}"
        raise ValueError(msg)

    return columns, linecount


def read_dashed_table(filepath: Path, usecols: list[str]) -> pl.DataFrame:
    """Read the data table that starts after the third '----' line, and cut it into columns.

    The tables look like a fixed-width format, but they are not one. A cell holds a right-aligned
    value with a minimum width. A value that is wider than its cell takes the space in front of
    it. It thus moves the rest of the line to the right.

    The Config_Upper value "4f2.5d1.6s1.6s2" of Tb II does this. A column is therefore not at a
    fixed character position, and this function splits each line on whitespace.

    polars cuts the tokens out of every line at once. That is much faster than a parser that reads
    one line at a time. The caller names the columns that it uses. The other columns stay in the
    list of tokens and never become a column. This keeps the memory low on a wide file. Every
    column of the result holds strings, because the tables mix text and numbers.
    """
    columns, skip_lines = read_table_header(filepath)
    missing = [name for name in usecols if name not in columns]
    # not an assert: input validation must survive python -O
    if missing:
        msg = f"The data table in {filepath} has no {missing} of the columns {columns}"
        raise ValueError(msg)

    # the name is not a column name of the data, so it cannot hide a column that the caller asked
    # for. A leading space keeps it different from every name that split() can give.
    countcol = " tokencount"
    lftable = (
        scan_file_lines(filepath, skip_lines=skip_lines)
        .select(parts=pl.col("line").str.extract_all(r"\S+"))
        # a blank line inside the table gives a null in every column, which the callers reject
        .filter(pl.col("parts").list.len() > 0)
        .select(
            pl.col("parts").list.len().alias(countcol),
            *[pl.col("parts").list.get(columns.index(name), null_on_oob=True).alias(name) for name in usecols],
        )
    )

    try:
        # the streaming engine halves the peak memory of a multi-gigabyte file. It is also faster.
        dftable = lftable.collect(engine="streaming")
    except pl.exceptions.NoDataError:
        # a table can hold a header and no data row, e.g. an ion with no line of one type
        dftable = pl.DataFrame(schema={countcol: pl.UInt32} | dict.fromkeys(usecols, pl.String))
    except pl.exceptions.PolarsError as exc:
        # the parser names no file, so name it here: a run reads more than a hundred of them
        msg = f"Could not read the data table in {filepath}: {exc}"
        raise ValueError(msg) from exc

    # A row with more tokens than the header moves every value into the column on its left. No
    # later test can find that. A row with fewer tokens has an empty cell, and keeps the columns
    # in front of that cell. The level tables leave the LS2 cell of a high-l level empty. The
    # callers take no column after LS2, so the function permits such a row.
    #
    # Not an assert: input validation must survive python -O.
    nlongrows = dftable.select((pl.col(countcol) > len(columns)).sum()).item()
    if nlongrows > 0:
        msg = f"{nlongrows} rows of the data table in {filepath} have more than {len(columns)} tokens"
        raise ValueError(msg)

    return dftable.drop(countcol)


def read_transitions_file(filepath: Path) -> pl.DataFrame:
    """Read one Floers+25 transitions file into the lowerlevel, upperlevel, A and forbidden columns.

    The Type column decides the forbidden flag, so this function keeps a row for each line and
    does not merge the rows yet. It drops the Type strings, because a large file has millions of
    rows.
    """
    dffile = read_dashed_table(filepath, usecols=["Lower", "Upper", "A", "Type"])

    # a null means an unreadable cell, and it must not decide a flag or become a zero rate.
    # Not an assert: input validation must survive python -O.
    for colname in ("Lower", "Upper", "A", "Type"):
        if dffile[colname].null_count() > 0:
            msg = f"Unreadable {colname} values in {filepath}"
            raise ValueError(msg)

    # the forbidden flag below trusts the Type column, so an unknown type must stop the run
    # rather than count as forbidden. A file has few distinct types, so test those and not each row.
    for transitiontype in dffile["Type"].unique().to_list():
        if re.fullmatch(r"[EM][0-9]+", transitiontype) is None:
            msg = f"Unknown transition type {transitiontype!r} in {filepath}"
            raise ValueError(msg)

    # Int32 holds every level index of the data set, and it halves the memory of the two columns
    return dffile.select(
        lowerlevel=pl.col("Lower").cast(pl.Int32),
        upperlevel=pl.col("Upper").cast(pl.Int32),
        A=pl.col("A").cast(pl.Float64),
        forbidden=pl.col("Type") != "E1",
    )


def read_levels_and_transitions(
    atomic_number: int, ion_stage: int, flog, calibrated: bool, withforbidden: bool = False
):
    """Read one ion from the Floers+25 data set.

    The ionisation energy comes from NIST rather than the file. Configurations are not unique
    (levels of one configuration differ by J), so level names combine the configuration, J and
    the file's index. The function checks the level indices, because a gap would silently
    misattach transitions. The function discards a transition to a level that the levels file
    does not list, with a warning in the log.
    """
    elsym = elsymbols[atomic_number]
    ion_stage_roman = roman_numerals[ion_stage]
    calibstr = "calib" if calibrated else "uncalib"
    ionstr = f"{atomic_number}{elsym}{ion_stage_roman}"

    # the handler name selects the directory. The floers25uncalib handler has no "withforbidden"
    # variant, so it searches the private directory and then the public directory.
    if withforbidden or calibrated or TESTMODE:
        basepaths = [get_basepath(withforbidden=withforbidden)]
    else:
        basepaths = [get_basepath(withforbidden=True), get_basepath(withforbidden=False)]

    levels_file = next(
        (
            found
            for searchpath in basepaths
            if (found := find_file_check_extension(searchpath / f"{ionstr}_levels_{calibstr}.txt")) is not None
        ),
        None,
    )
    if levels_file is None:
        searched = " or ".join(str(searchpath / f"{ionstr}_levels_{calibstr}.txt*") for searchpath in basepaths)
        msg = f"Found no Floers+25 levels file for {ionstr}. Searched {searched}"
        raise FileNotFoundError(msg)
    basepath = levels_file.parent

    # the original Floers+25 format has a single transitions file. The newer format has one
    # file for each transition type, for example _E1, _E2, and _M1.
    lines_file = find_file_check_extension(basepath / f"{ionstr}_transitions_{calibstr}.txt")

    # a file can exist in a plain form and in a compressed form at the same time. Keep one path
    # for each name. The extension list is in priority order, so the plain form wins.
    pertype_file_of_name: dict[str, Path] = {}
    for ext in compression_extensions:
        for filepath in basepath.glob(f"{ionstr}_transitions_{calibstr}_*.txt{ext}"):
            pertype_file_of_name.setdefault(filepath.name.removesuffix(ext), filepath)
    pertype_files = [pertype_file_of_name[name] for name in sorted(pertype_file_of_name)]

    if lines_file is not None and pertype_files:
        msg = f"Found both {lines_file.name} and per-type transitions files in {basepath}. Remove one of the forms."
        raise ValueError(msg)
    transition_files = [lines_file] if lines_file is not None else pertype_files
    if not transition_files:
        msg = f"Found no Floers+25 transitions files for {ionstr} ({calibstr}) in {basepath}"
        raise FileNotFoundError(msg)

    log_and_print(
        flog,
        f"Reading Floers+25 {calibstr}rated data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman}) from {basepath.name}/{levels_file.name} and {len(transition_files)} transitions files",
    )

    ionization_energy_in_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]

    # the levels files of the data sets do not all carry the same columns, so name the ones used.
    # J keeps its "5/2" form as a string, which the g column below reads.
    dflevels = read_dashed_table(levels_file, usecols=["Index", "Energy", "J", "Parity", "Configuration"])

    # a null means an empty cell, and a level needs each of these values
    # not an assert: input validation must survive python -O
    for colname in dflevels.columns:
        if dflevels[colname].null_count() > 0:
            msg = f"Unreadable {colname} values in {levels_file}"
            raise ValueError(msg)

    # every expression here reads the input frame, so g reads the file's own J string
    dflevels = dflevels.with_columns(
        pl.col("Index").cast(pl.Int64),
        pl.col("Energy").cast(pl.Float64),
        pl.col("Parity").cast(pl.Int64),
        pl.when(pl.col("J").str.ends_with("/2"))
        .then(pl.col("J").str.strip_suffix("/2").cast(pl.Int32) + 1)
        .otherwise(
            pl.col("J").str.strip_suffix("/2").cast(pl.Int32) * 2 + 1
        )  # the strip_suffix is not necessary here (J does not end in "/2") but it prevents a polars error
        .alias("g"),
    )

    # the file indexes the levels from zero in file order, and the transitions refer to those
    # indices. So a gap would misattach them. Not an assert: input validation must survive python -O.
    if dflevels["Index"].to_list() != list(range(dflevels.height)):
        msg = f"Level indices in {levels_file} are not contiguous and zero-based"
        raise ValueError(msg)

    # Configuration is not unique (levels of one configuration differ by J), so append the
    # contiguous index. The configuration stays first, for get_level_valence_n() and the
    # adata.txt comment.
    dflevels = dflevels.with_columns(
        levelname=pl.format("{} J={} index={}", pl.col("Configuration"), pl.col("J"), pl.col("Index"))
    )

    log_and_print(flog, f"Read {dflevels.height:d} levels")

    # the files keep their order, so the merge below adds the A values in the same order for
    # each run. rechunk=False: the merge reads the rows once, so a copy into one chunk gains nothing
    dftransitions = pl.concat(
        [read_transitions_file(transition_file) for transition_file in transition_files], rechunk=False
    )

    log_and_print(flog, f"Read {dftransitions.height} transitions")

    # some transitions files reference levels that the levels file does not list, for example
    # the private Ce III set. Discard those rows with a warning: they cannot attach to a level.
    inrange = pl.col("lowerlevel").is_between(0, dflevels.height - 1) & pl.col("upperlevel").is_between(
        0, dflevels.height - 1
    )
    ndiscarded = dftransitions.filter(~inrange).height
    if ndiscarded > 0:
        log_and_print(
            flog,
            f"WARNING: Discarded {ndiscarded} transitions of {ionstr} that reference levels outside"
            f" 0..{dflevels.height - 1}",
        )
        dftransitions = dftransitions.filter(inrange)

    # some files give the initial state first, so a row can carry the higher level in the Lower
    # column. Swap those rows into energy order: the merge and the output want lowerlevel first.
    nreversed = dftransitions.filter(pl.col("lowerlevel") > pl.col("upperlevel")).height
    if nreversed > 0:
        log_and_print(flog, f"Swapped the level order of {nreversed} reversed transitions")
        dftransitions = dftransitions.with_columns(
            lowerlevel=pl.min_horizontal("lowerlevel", "upperlevel"),
            upperlevel=pl.max_horizontal("lowerlevel", "upperlevel"),
        )

    # the file's Lower/Upper indices are already zero-based. They agree with the level ids in
    # memory. Merge the rows that share a level pair. Add their A values. ARTIS reads each row
    # as one transition, so a duplicate pair would double a line.
    # The Type column decides the forbidden flag. A merged row is forbidden only when no E1 line
    # contributes to it, so M1, E2, and any higher multipole count as forbidden.
    # Each level pair occurs once after the merge, and output.py sorts the rows on the pair. The
    # order of the groups thus does not change the output, and the merge does not have to keep it.
    dftransitions = dftransitions.group_by(["upperlevel", "lowerlevel"]).agg(
        A=pl.col("A").sum(), forbidden=pl.col("forbidden").all()
    )

    # cross-check the Type column against the Laporte rule, which an E1 line must obey. The level
    # indices are contiguous from zero, as checked above, so a level index is also a row number.
    parity_of_index = dflevels["Parity"]
    dfallowed = dftransitions.filter(~pl.col("forbidden"))
    n_paritymatch = (
        parity_of_index.gather(dfallowed["lowerlevel"]) == parity_of_index.gather(dfallowed["upperlevel"])
    ).sum()
    if n_paritymatch > 0:
        log_and_print(flog, f"WARNING: {n_paritymatch} E1 transitions connect two levels with the same parity")

    # use standard artisatomic column names. The forbidden flag comes from the Type column
    # above, so add_level_ids_forbidden() does not derive it from the parity.
    dflevels = dflevels.select(
        levelname=pl.col("levelname"),
        parity=pl.col("Parity"),
        g=pl.col("g"),
        energyabovegsinpercm=pl.col("Energy"),
    )

    return ionization_energy_in_ev, dflevels, dftransitions


def get_level_valence_n(levelname: str) -> int | None:
    """Principal quantum number of the valence electron, read from a Floers+25 level name.

    Returns None for a name that it cannot parse. The caller, match_hydrogenic_phixs(), then
    gives the level no estimate and writes a warning to the ion log.

    Kept separate from the other readers' versions. Each data source names its levels
    differently, so a shared parser would have to guess the convention of each name.
    """
    # level names are "<configuration> J=<J> index=<index>", so drop everything after the config
    part = levelname.split(" ", maxsplit=1)[0].rsplit(".", maxsplit=1)[-1]
    return parse_orbital_n(part)

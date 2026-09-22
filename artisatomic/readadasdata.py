#!/usr/bin/env python3
"""Read levels, transitions and collision strengths from files in the ADAS adf04 format.

Authors at QUB (Queen's University Belfast) made the Co, Sr I and Fe files. The Ca III file comes
from OPEN-ADAS (https://open.adas.ac.uk). The reader had the name readqubdata before, and the
handlers had the names "qub" and "qub_cobalt" (see ionhandlers.renamed_handlers).

The Sr I file comes from Dougan, D. J., McElroy, N. E., Ballance, C. P., Ramsbottom, C. A. (2025),
MNRAS, 541, 367-383, doi:10.1093/mnras/staf1013. The Co data in co_tyndall comes from a private
communication (see atomic-data-adas/README.txt).
"""

import re
import string
import typing as t
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl

# the "cmfgen_qubphixs" handler takes the CMFGEN phot files for an ion with no QUB cross sections.
# readhillierdata imports nothing from this module, so the import is not circular.
from artisatomic import readhillierdata
from artisatomic.base import add_handlers_if_not_set
from artisatomic.base import compression_extensions
from artisatomic.base import empty_transitions_schema
from artisatomic.base import find_file_check_extension
from artisatomic.base import fixed_width_column
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import ion_filename_pattern
from artisatomic.base import ion_label
from artisatomic.base import ions_from_filenames
from artisatomic.base import log_and_print
from artisatomic.base import log_comment
from artisatomic.base import log_detail
from artisatomic.base import log_source
from artisatomic.base import nist_ionization_energy_comment
from artisatomic.base import path_for_log
from artisatomic.base import path_in_data_folder
from artisatomic.base import PhixsData
from artisatomic.base import PYDIR
from artisatomic.base import TESTMODE
from artisatomic.base import xopen_check_extension
from artisatomic.levelnames import convert_eissner_to_standard
from artisatomic.levelnames import eissner_shell_orders
from artisatomic.levelnames import eissner_total_l_is_possible
from artisatomic.levelnames import expand_standard_config
from artisatomic.levelnames import get_config_parity
from artisatomic.levelnames import is_eissner_config
from artisatomic.levelnames import lchars
from artisatomic.levelnames import lchars_lower
from artisatomic.levelnames import looks_like_eissner_config
from artisatomic.levelnames import split_count_and_n
from artisatomic.phixs import combine_phixs_routes
from artisatomic.phixs import PHIXS_TARGET_FRACTION_CUT
from artisatomic.phixs import reduce_phixs_tables

adasfolder = PYDIR / ".." / "atomic-data-adas"
adaspath = adasfolder.resolve()


def _move_into(source: Path, target: Path) -> list[Path]:
    """Move source to target, and merge two directories. Return each path that stays at the source."""
    if not target.exists() and not target.is_symlink():
        source.rename(target)
        return []
    if source.is_dir() and not source.is_symlink() and target.is_dir():
        staying = [path for entry in sorted(source.iterdir()) for path in _move_into(entry, target / entry.name)]
        if not staying:
            source.rmdir()
        return staying
    if source.name == ".DS_Store":
        # the Finder writes this file into each directory, so the target always has one
        source.unlink()
        return []
    return [source]


def _move_old_data_directory(oldpath: Path, newpath: Path) -> None:
    if oldpath.is_symlink():
        if newpath.exists() or newpath.is_symlink():
            print(f"WARNING: {oldpath} is a symbolic link. Move its files to {newpath}, then remove the link.")
        else:
            oldpath.rename(newpath)
            print(f"Renamed the symbolic link {oldpath} to {newpath}")
        return
    if not oldpath.is_dir():
        return
    staying = _move_into(oldpath, newpath)
    print(f"Moved the files of {oldpath} to {newpath}")
    if staying:
        names = ", ".join(str(path.relative_to(oldpath)) if path != oldpath else "." for path in staying[:5])
        more = f" and {len(staying) - 5} more" if len(staying) > 5 else ""
        print(f"WARNING: {oldpath} keeps these files, because {newpath} has files with the same names: {names}{more}")


def rename_old_data_directory(oldpath: Path, newpath: Path) -> None:
    """Rename the data directory from its old name, atomic-data-qub, to the new name.

    After an update of the repository, the new directory holds the tracked files, and the old
    directory holds the files that Git does not track. The function then moves each of those
    files. It does not replace a file of the new directory, and it gives a warning for each file
    that stays. The data of a symbolic link is in a different place, so the function renames the
    link and does not move that data. A failure gives a warning, because the conversion can
    continue without the old directory.
    """
    try:
        _move_old_data_directory(oldpath, newpath)
    except OSError as error:
        print(f"WARNING: could not move the files of {oldpath} to {newpath}: {error}")


# not resolved: a symbolic link must stay a link, so that the function above can find it
old_adaspath = (PYDIR / "..").resolve() / "atomic-data-qub"


def rename_old_adas_directory() -> None:
    """Rename the old data directory of this reader. The two paths come from the module at this time, so a test can set them."""
    rename_old_data_directory(old_adaspath, adaspath)


tyndall_co3_path = (adaspath / ("co_tyndall_test_sample" if TESTMODE else "co_tyndall")).resolve()

# the "source:" line of the comment blocks in the output files (see Handler.description in iondata.py)
description = (
    "files in the ADAS adf04 format, in the folder atomic-data-adas of the artisatomic repository (see its README.txt)"
)
description_co4 = (
    "the ground level of Co IV (3d6 5D4) as a constant in the code of artisatomic, with no transition, and the"
    " ionisation energy of the NIST Atomic Spectra Database, https://physics.nist.gov/asd, doi:10.18434/T4W30F"
)


def description_of_ion(atomic_number: int, ion_stage: int) -> str:
    """Give the "source:" line of an ion. No file gives the one level of Co IV."""
    return description_co4 if (atomic_number, ion_stage) == (27, 4) else description


# The origin of the files of one element or one ion. atomic-data-adas/README.txt holds the same
# facts, so change the two together. Co IV has no entry, because no file gives its one level.
qub_origin = "Authors at Queen's University Belfast (QUB) made the {} files."
ion_origins: dict[tuple[int, int | None], str] = {
    (27, 3): qub_origin.format("Co III") + " They come from a private communication.",
    (26, None): qub_origin.format("Fe"),
    (38, 1): (
        "Authors at Queen's University Belfast (QUB) made the Sr I file: Dougan, D. J., McElroy, N. E.,"
        " Ballance, C. P., Ramsbottom, C. A. (2025), MNRAS, 541, 367-383, doi:10.1093/mnras/staf1013"
    ),
    (
        20,
        3,
    ): "The Ca III file comes from OPEN-ADAS, https://open.adas.ac.uk (atomic-data-adas/20_3.txt gives its address).",
}
qub_cobalt_phixs_description = "the Co cross sections of Queen's University Belfast (private communication)"

# the name of a data file, e.g. 26_2.adf04 or 26_2.adf04.zst
adas_filename_pattern = ion_filename_pattern(".adf04")


class ADASTransitionRow(t.NamedTuple):
    """One ADAS bound-bound transition.

    nameto is the name of the upper level, and namefrom is the name of the lower level. The row
    carries the level ids, so add_level_ids_forbidden() does not join on the names.
    """

    lowerlevel: int
    upperlevel: int
    A: float
    nameto: str
    namefrom: str
    lambdaangstrom: float


class ADASEnergyLevel(t.NamedTuple):
    """One energy level of an ADAS calculation."""

    levelname: str
    adas_id: int
    twosplusone: int
    l: int
    j: float
    energyabovegsinpercm: float
    g: float
    parity: int | None  # None where the configuration determines no parity


def extend_ion_list(
    ion_handlers,
    *,
    minionstage: int | None = None,
    maxionstage: int | None = None,
    maxatomicnumber: int | None = None,
):
    """Add every ion with an ADAS adf04 file to ion_handlers under the "adas" handler."""
    # the files ship compressed or plain, so match every form of the name that a reader accepts
    adasfiles = [f for ext in compression_extensions for f in adaspath.glob(f"*_*.adf04{ext}")]
    # each name holds the atomic number and the ion stage, e.g. 26_2.adf04
    adasions = ions_from_filenames(adasfiles, adas_filename_pattern)

    return add_handlers_if_not_set(
        ion_handlers,
        adasions,
        "adas",
        minionstage=minionstage,
        maxionstage=maxionstage,
        maxatomicnumber=maxatomicnumber,
    )


adf04_section_end = "-1"


def adf04_first_field(line: str) -> str:
    """Return the first whitespace-separated field of the line, or "" when it holds none."""
    fields = line.split(maxsplit=1)
    return fields[0] if fields else ""


def is_adf04_terminator(line: str) -> bool:
    """Report whether the line is the row that ends an adf04 section.

    The level block and the collision block each end with such a row. Writers pad it
    differently, so the test is on the first field and not on a fixed column. The loop over the
    collision rows makes the same test on the first field, which it needs for a second test.
    """
    return adf04_first_field(line) == adf04_section_end


# These values give the fixed columns of the specification. A collision row is a1,i3,i4,16e8.2.
# The process code and the two file indices fill the first 8 columns, and each value has 8
# columns. The line with the temperatures is f5.1,i5,6x,14e8.2: ZEFF (the effective charge), ITYP
# (the type of the collision data), and the temperatures from column 17.
adf04_index_width = 4
adf04_row_prefix_width = 2 * adf04_index_width
adf04_value_width = 8
adf04_ityp_columns = slice(5, 10)
adf04_temperature_offset = 16
# the columns a1,i3 of a collision row hold an upper file index of 999 at most
adf04_largest_i3 = 10 ** (adf04_index_width - 1) - 1

# adf04 writes the exponent with no "E": 1.23-04 means 1.23e-04. The "E" goes only after a digit
# or a point. A leading sign and a number that already has an "E" stay as they are.
adf04_exponent_pattern = r"([0-9.])([-+])"


def adf04_float(offset: int, length: int) -> pl.Expr:
    """Return an expression for the fixed columns as a float. A blank or unreadable field gives a null."""
    return (
        fixed_width_column(offset, length)
        .str.replace_all(adf04_exponent_pattern, "${1}E${2}")
        .cast(pl.Float64, strict=False)
    )


def adf04_number(text: str) -> float:
    """Convert a number in the ADAS form, where the exponent has a sign and no letter ("5.00+03")."""
    return float(re.sub(adf04_exponent_pattern, r"\1E\2", text))


def adf04_file_index(offset: int, length: int, prefix: str = "") -> pl.Expr:
    """Return an expression for a file index in the fixed columns. Other text gives a null.

    Fortran writes an integer at the right of its columns, with no sign and no zero at the left.
    A different text, for example "005" or "12 ", shows that the columns hold something else.
    The prefix is a pattern for the characters that come before the integer.
    """
    pattern = rf"^{prefix} *([1-9][0-9]*)$"
    return pl.col("line").str.slice(offset, length).str.extract(pattern, 1).cast(pl.Int64, strict=False)


def adf04_upper_file_index(levelcount: int) -> pl.Expr:
    """Return an expression for the upper file index of a collision row.

    Columns 1 to 4 hold the process code (a blank, "1", "2" or "3") and the file index. These
    columns cannot hold a file index above 999. A file with more levels thus has no column for the
    process code, and columns 1 to 4 are the file index. A text such as "1 23" is a process code
    and a file index in each file.
    """
    index_after_code = adf04_file_index(0, adf04_index_width, prefix="[ 123]")
    if levelcount <= adf04_largest_i3:
        return index_after_code
    return pl.coalesce(adf04_file_index(0, adf04_index_width), index_after_code)


# The specification writes each value with a decimal point. Some files omit it. The lookahead
# stops a match in the middle of a number, so "4.10+05" does not give 4.10.
decimal_number_pattern = r"\d+(?:\.\d*)?(?![0-9.,+\-EeDd])"

# The groups are: IZ (the ion charge), IZ0 (the atomic number), IZ1 (the ion stage) and the
# ionisation potential. The specification lets a file omit the parent term after the ionisation
# potential. Some files also leave the element symbol blank.
adf04_header_regex = re.compile(rf"\s*[A-Za-z]{{0,2}}\s*\+\s*(\d+)\s+(\d+)\s+(\d+)\s+({decimal_number_pattern})")

# The groups are: the file index, the configuration, the multiplicity (2S+1), L as a hexadecimal
# digit, J, and the energy above the ground level. The configuration column has no fixed width
# or format, so ".*?" captures it. It is not greedy, because the text after the energy can have
# a second group of the form "(n)L(J)".
adf04_level_regex = re.compile(
    rf"\s*(\d+)\s+(.*?)\s+\((\d+)\)([0-9A-Fa-f])\(\s*({decimal_number_pattern})\)\s+({decimal_number_pattern})"
)


def _read_adf04_header(line: str, atomic_number: int, ion_stage: int, filepath: str | Path) -> float:
    """Return the ionisation energy in eV from the adf04 header. The header must name the requested ion."""
    headermatch = adf04_header_regex.match(line)
    if headermatch is None:
        msg = f"Cannot read the adf04 header line in {filepath}: {line.rstrip()!r}"
        raise ValueError(msg)
    header_ion_charge = int(headermatch[1])
    header_atomic_number = int(headermatch[2])
    header_ion_stage = int(headermatch[3])
    ionization_energy_percm = float(headermatch[4])

    if atomic_number != header_atomic_number:
        msg = f"Atomic number ({atomic_number}) does not match that read from {filepath} ({header_atomic_number})"
        raise ValueError(msg)
    if ion_stage != header_ion_stage:
        msg = f"Ion stage ({ion_stage}) does not match that read from {filepath} ({header_ion_stage})"
        raise ValueError(msg)
    if header_ion_charge + 1 != header_ion_stage:
        msg = (
            f"The header of {filepath} gives the ion charge {header_ion_charge} and the ion stage"
            f" {header_ion_stage}. The ion stage must be the ion charge plus 1."
        )
        raise ValueError(msg)

    return ionization_energy_percm * hc_in_ev_cm


def _read_adf04_temperatures(line: str, filepath: str | Path) -> tuple[list[str], list[float]]:
    """Return the temperature fields of the line that holds ZEFF, ITYP and the temperatures, as text and as numbers.

    The fields must be in the fixed columns of the specification, because the reader takes each
    collision row from the same columns. A file in a different layout stops here with an error.
    Without this check, such a file gives rows with no value or with a wrong value.
    """
    line = line.rstrip()
    if not line:
        msg = f"{filepath} ends before the line that gives the temperatures of the collision strengths"
        raise ValueError(msg)
    # ITYP=3 gives an upsilon value for each electron temperature. ITYP=1 gives a collision
    # strength for each value of a threshold parameter, and this reader cannot use them.
    ityp_text = line[adf04_ityp_columns].strip()
    if not ityp_text.isascii() or not ityp_text.isdigit() or int(ityp_text) != 3:
        msg = (
            f"{filepath} does not give an upsilon value for each electron temperature."
            f" The adf04 ITYP field must be 3, and it is {ityp_text!r}"
        )
        raise ValueError(msg)

    temperatures = [
        line[start : start + adf04_value_width].strip()
        for start in range(adf04_temperature_offset, len(line), adf04_value_width)
    ]
    # not an assert: input validation must survive python -O
    if not temperatures:
        msg = f"{filepath} names no temperatures for its collision strengths"
        raise ValueError(msg)
    if temperatures != line[adf04_temperature_offset:].split():
        msg = (
            f"{filepath} does not give the temperatures in the fixed columns of the adf04 specification"
            f" (f5.1,i5,6x,14e8.2): {line!r}"
        )
        raise ValueError(msg)
    try:
        return temperatures, [adf04_number(text) for text in temperatures]
    except ValueError:
        msg = f"{filepath} has a temperature that is not a number: {line!r}"
        raise ValueError(msg) from None


def _eissner_order_of_file(levels: list[tuple[str, int]], filepath: str | Path, flog) -> str | None:
    """Return the order of the Eissner shell characters of a file, or None for a file in standard notation.

    Each level is its configuration and its total L. The notation is a property of the file. A
    label such as "21" in a file with standard notation is also a valid Eissner configuration.
    A decision for each level therefore gives wrong level names.
    """
    configs = [config for config, _total_l in levels]
    # A blank field has no notation, so it does not count.
    if 2 * sum(looks_like_eissner_config(config) for config in configs) <= sum(bool(config) for config in configs):
        return None

    # The total L of a level shows the order. The count of all levels decides, because one level
    # with a wrong L in the file must not stop the run.
    agreement = {
        order: sum(
            is_eissner_config(config, order) and eissner_total_l_is_possible(config, total_l, order)
            for config, total_l in levels
        )
        for order in eissner_shell_orders
    }
    order = max(agreement, key=lambda name: agreement[name])
    counts = ", ".join(
        f"{count} of {len(levels)} levels agree with the {name} order" for name, count in agreement.items()
    )
    log_comment(flog, ("adata",), f"The reader found Eissner notation in the electron configurations ({counts})")

    # The digits of a defective Eissner configuration must not become the name of a level. A
    # blank field or a label is not an Eissner configuration, and the reader keeps its text.
    for config in configs:
        if looks_like_eissner_config(config) and not is_eissner_config(config, order):
            msg = f"{filepath} uses Eissner notation, but the reader cannot read the configuration {config!r}"
            raise ValueError(msg)
    labels = [config for config in configs if not looks_like_eissner_config(config)]
    if labels:
        log_comment(
            flog, ("adata",), f"WARNING: {len(labels)} levels have no Eissner configuration, for example {labels[0]!r}."
        )
    if agreement[order] < len(levels) - len(labels):
        log_comment(
            flog,
            ("adata",),
            f"WARNING: The shells of {len(levels) - len(labels) - agreement[order]} levels cannot give their total L."
            " The file possibly uses a different order of the Eissner shell characters.",
        )
    return order


def _split_at_terms(config: str) -> Iterator[tuple[str, bool]]:
    """Yield each part of a configuration, and True if the part is a term in parentheses.

    The column of the configuration has 18 characters, so a term can have no closing parenthesis.
    """
    depth = 0
    start = 0
    for position, char in enumerate(config):
        if char == "(":
            if depth == 0:
                yield config[start:position], False
                start = position
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
            if depth == 0:
                yield config[start : position + 1], True
                start = position + 1
    yield config[start:], depth > 0


def _standardise_config(config: str, *, eissner_order: str | None) -> str:
    """Return the configuration in standard notation. eissner_order is None for a file in standard notation."""
    config = config.strip()
    if eissner_order is not None and is_eissner_config(config, eissner_order):
        return convert_eissner_to_standard(config, eissner_order)

    # Each term in parentheses stays in upper case, e.g. "3D6(5D)4DA" becomes "3d6(5D)4d10".
    return "".join(
        part if is_term else expand_standard_config(part.lower()) for part, is_term in _split_at_terms(config)
    )


def read_adf04(
    filepath: str | Path,
    flog,
    electrontemperature: float,
    atomic_number: int,
    ion_stage: int,
    *,
    contents: str = "The levels, the transitions and the collision strengths",
    origin: str | None = None,
) -> tuple[float, list[ADASEnergyLevel], dict[tuple[int, int], float], pl.DataFrame]:
    """Read levels and effective collision strengths from an ADAS adf04 file.

    The collision strengths come from the tabulated temperature nearest to electrontemperature,
    as readhillierdata.read_coldata() picks them for the CMFGEN files.

    The comment blocks name the file as the source of contents. origin is the sentence of
    ion_origins for the file, which follows the file name in the blocks.

    Returns four values:
    - the ionisation energy in eV;
    - the levels;
    - a dict of upsilon values keyed by a (lower, upper) pair of zero-based level ids;
    - the parsed collision rows.

    The caller takes the A-values from that frame, which saves a second read and a second parse
    of the file. The file numbers levels from one, and the rest of the code looks up id n at
    list index n - 1. The reader therefore checks that the file indices are contiguous and 1-based.
    """
    energylevels: list[ADASEnergyLevel] = []
    upsilondict: dict[tuple[int, int], float] = {}
    ionization_energy_ev = 0.0
    log_comment(flog, ("adata", "transitiondata"), f"{contents} come from {path_in_data_folder(filepath, adasfolder)}.")
    if origin is not None:
        log_comment(flog, ("adata", "transitiondata"), origin)
    with xopen_check_extension(filepath) as fleveltrans:
        line = fleveltrans.readline()
        ionization_energy_ev = _read_adf04_header(line, atomic_number, ion_stage, filepath)
        log_and_print(flog, f"The file gives an ionisation energy of {ionization_energy_ev:.7f} eV.")
        # A note between two 'C-' rule lines can sit inside the level block, and the reader skips
        # its lines. The loops stop at the '-1' rows, so the reader never reads a note after the
        # collision block.
        atomic_group_note = False
        levelrows: list[tuple[str, ...]] = []
        while True:
            line = fleveltrans.readline()
            if not line or is_adf04_terminator(line):
                break
            if line.startswith("C-"):
                atomic_group_note = not atomic_group_note
                continue
            if atomic_group_note:
                continue

            levelmatch = adf04_level_regex.match(line)
            if levelmatch is None:
                msg = f"Cannot read the adf04 level line in {filepath}: {line.rstrip()!r}"
                raise ValueError(msg)
            levelrows.append(levelmatch.groups())

        eissner_order = _eissner_order_of_file(
            [(config, int(l_hex, 16)) for _index, config, _multiplicity, l_hex, _j, _energy in levelrows],
            filepath,
            flog,
        )

        for adas_id, config, multiplicity, l_hex, j, energy_percm in levelrows:
            config = _standardise_config(config, eissner_order=eissner_order)
            energylevel = ADASEnergyLevel(
                config, int(adas_id), int(multiplicity), int(l_hex, 16), float(j), float(energy_percm), 0.0, 0
            )

            # hasterm=False: an adf04 name is all configuration, because the file keeps 2S+1 and
            # L in their own columns (read just above). A cut of a term off the end would lose
            # the last orbital of '3S2 3P6 3D5 4P1'. It would also read the bare '5s2' as a term
            # and not as an orbital. That is how every level of some files came out even.
            parity = get_config_parity(config, hasterm=False)

            levelname = energylevel.levelname + "_{:d}{:}{:}[{:d}/2]_id={:}".format(
                energylevel.twosplusone,
                lchars[energylevel.l],
                # the name keeps the old even/odd letter where the parity is unknown. So
                # adata.txt does not depend on a distinction that the parity column now makes.
                ["e", "o"][parity if parity is not None else 0],
                int(2 * energylevel.j),
                energylevel.adas_id,
            )

            g = 2 * energylevel.j + 1
            energylevel = energylevel._replace(g=g, parity=parity, levelname=levelname)
            energylevels.append(energylevel)

            # the transition and upsilon tables use these file indices and the rest of the code
            # looks up id n at index n - 1. A non-contiguous file would therefore misattach every
            # transition. Not an assert: input validation must survive python -O.
            if energylevel.adas_id != len(energylevels):
                msg = (
                    f"adf04 file index {energylevel.adas_id} found at position {len(energylevels)} in {filepath}."
                    " The file indices must be contiguous and start at 1."
                )
                raise ValueError(msg)

        temperatures, temperature_values = _read_adf04_temperatures(fleveltrans.readline(), filepath)

        # ADAS writes the other processes with a letter in column 1: R for recombination, S and I
        # for ionisation, P for proton impact. A comment line starts with a letter too. A blank
        # line is not a bad row, so the counter skips it.
        collision_lines: list[str] = []
        skipped_rows = 0
        for line in fleveltrans:
            firstfield = adf04_first_field(line)
            if firstfield == adf04_section_end:
                break
            if not firstfield:
                continue
            if firstfield[0].isalpha():
                skipped_rows += 1
                continue
            collision_lines.append(line)

        nearest_index = min(
            range(len(temperatures)), key=lambda index: abs(temperature_values[index] - electrontemperature)
        )
        log_comment(
            flog,
            ("transitiondata",),
            f"The collision strengths are the values at {temperature_values[nearest_index]:.0f} K. The collision data"
            f" file gives these temperatures [K]: {', '.join(f'{t:.10g}' for t in temperature_values)}.",
        )

        # A split at whitespace fails where two values touch, for example "2.81-01-3.01-02".
        # A cut of only the wanted fields needs about a third of the memory of a cut of every
        # line into all of its columns.
        upsilon_offset = adf04_row_prefix_width + adf04_value_width * (1 + nearest_index)
        has_indices = pl.col("upper").is_not_null() & pl.col("lower").is_not_null()
        # A row that stops before the selected temperature has no upsilon there. It is a good
        # row, and the caller takes its A-value.
        is_short = has_indices & (fixed_width_column(upsilon_offset, adf04_value_width).str.len_chars() == 0)
        rowcount = len(collision_lines)
        collisiondf = (
            pl.DataFrame({"line": collision_lines}, schema={"line": pl.String})
            .with_columns(
                adf04_upper_file_index(len(energylevels)).alias("upper"),
                adf04_file_index(adf04_index_width, adf04_index_width).alias("lower"),
                adf04_float(adf04_row_prefix_width, adf04_value_width).alias("avalue"),
                adf04_float(upsilon_offset, adf04_value_width).alias("upsilon"),
            )
            # The caller makes a transition from each row of this frame, so a row that the reader
            # cannot parse must not be in it.
            .filter(is_short | (has_indices & pl.col("upsilon").is_not_null()))
            .select("upper", "lower", "avalue", "upsilon")
        )
        goodrows = collisiondf.drop_nulls(subset=["upsilon"])
        short_rows = collisiondf.height - goodrows.height
        unreadable_rows = rowcount - collisiondf.height
        if rowcount and collisiondf.is_empty():
            msg = (
                f"The reader could not parse any of the {rowcount} collision rows of {filepath}. Each row must have"
                ' the process code " ", "1", "2" or "3" and the fixed columns of the adf04 specification'
                " (a1,i3,i4,16e8.2)."
            )
            raise ValueError(msg)

        for lower, upper, upsilon in goodrows.select("lower", "upper", "upsilon").iter_rows():
            lower, upper = min(lower, upper), max(lower, upper)
            # a raise rather than an assert: this validates an input file, and the check
            # must survive python -O. Equal ids would store a self-transition.
            if not 1 <= lower < upper <= len(energylevels):
                msg = (
                    f"collision strength file indices {lower}, {upper} in {filepath} are outside"
                    f" the file's {len(energylevels)} levels"
                )
                raise ValueError(msg)

            # the file index starts at one; level ids are zero-based in memory. The log
            # messages keep the file indices, because they are about the file's contents.
            levelidpair = (lower - 1, upper - 1)
            if levelidpair not in upsilondict:
                upsilondict[levelidpair] = upsilon
            else:
                log_detail(
                    flog,
                    ("transitiondata",),
                    "duplicate upsilon",
                    f"Duplicate upsilon value for transition {lower:d} to {upper:d}. The reader keeps"
                    f" {upsilondict[levelidpair]:5.2e} and ignores {upsilon:5.2e}",
                )

    log_and_print(flog, f"The reader got {len(energylevels):d} levels.")
    if skipped_rows:
        log_comment(
            flog,
            ("transitiondata",),
            f"The reader skipped {skipped_rows:d} collision rows that are not an electron impact excitation.",
        )
    if unreadable_rows:
        log_comment(
            flog, ("transitiondata",), f"The reader skipped {unreadable_rows:d} collision rows that it could not parse."
        )
    if short_rows:
        warning = "" if upsilondict else "WARNING: no collision row has an upsilon at the selected temperature. "
        log_comment(
            flog,
            ("transitiondata",),
            f"{warning}{short_rows:d} collision rows have no value at the selected temperature.",
        )

    return ionization_energy_ev, energylevels, upsilondict, collisiondf


def append_adas_transition(adas_energylevels, adas_transitions, id_lower, id_upper, A, filepath) -> None:
    """Validate one radiative transition row and append it to the transition list.

    The ids are the file indices, which start at 1. The columns of a file do not always give the lower
    level first, so the function sorts the pair. read_adf04() sorts each collision pair the same
    way. A reversed pair would give a transition that the upsilon join misses.
    """
    id_lower, id_upper = min(id_lower, id_upper), max(id_lower, id_upper)
    # a raise rather than an assert: this validates an input file. A non-positive
    # id would wrap to the wrong level through a negative index. An id one past the
    # end would raise a bare IndexError that names neither the file nor the transition.
    if not 1 <= id_lower <= len(adas_energylevels) or not 1 <= id_upper <= len(adas_energylevels):
        msg = (
            f"transition file indices {id_lower}, {id_upper} in {filepath} are outside"
            f" the file's {len(adas_energylevels)} levels"
        )
        raise ValueError(msg)
    # read_adf04() makes the same check for a collision pair. Without it, the failure comes from
    # the writer, after adata.txt already holds the ion.
    if id_lower == id_upper:
        msg = f"transition in {filepath} has the same file index {id_lower} for the two levels"
        raise ValueError(msg)
    # the file numbers levels from one; level ids are zero-based in memory
    id_lower -= 1
    id_upper -= 1
    level_upper = adas_energylevels[id_upper]
    level_lower = adas_energylevels[id_lower]
    levelname_upper = level_upper.levelname
    levelname_lower = level_lower.levelname
    delta_percm = level_upper.energyabovegsinpercm - level_lower.energyabovegsinpercm
    lamdaangstrom = 1.0e8 / delta_percm if delta_percm != 0.0 else -1.0
    transition = ADASTransitionRow(
        lowerlevel=id_lower,
        upperlevel=id_upper,
        A=A,
        nameto=levelname_upper,
        namefrom=levelname_lower,
        lambdaangstrom=lamdaangstrom,
    )
    adas_transitions.append(transition)


def read_photoionizations(atomic_number, ion_stage, dfenergylevels, args, flog) -> PhixsData:
    """Read the cross sections of an ion of the "adas" handler. Only Co III, with its QUB levels, has data."""
    fill_arrays = _fill_co3_phixs if (atomic_number, ion_stage) == (27, 3) else None
    return _read_qub_phixs(fill_arrays, atomic_number, ion_stage, dfenergylevels.height, args, flog)


def read_cmfgen_qubphixs_photoionizations(atomic_number, ion_stage, dfenergylevels, args, flog) -> PhixsData:
    """Read the cross sections of an ion of the "cmfgen_qubphixs" handler.

    The levels of such an ion come from CMFGEN. Co II has QUB cross sections for those levels. Each
    other ion takes the CMFGEN phot files.
    """
    if (atomic_number, ion_stage) == (27, 2):
        return _read_qub_phixs(_fill_co2_phixs, atomic_number, ion_stage, dfenergylevels.height, args, flog)
    return readhillierdata.read_phixs_tables(atomic_number, ion_stage, dfenergylevels, args, flog)


def read_adas_levels_and_transitions(atomic_number, ion_stage, flog, args):
    """Read one ion from an adf04 file, or from the QUB Co data.

    args gives -electrontemperature, which picks the tabulated collision strengths.

    The function reads the per-ion adf04 files, the Co III files in the co_tyndall directory, and
    the single level of Co IV. The Co III and the Co IV data have their own layouts. Also returns
    the effective collision strengths, so this reader
    supplies an upsilondict. Most other readers leave another module to fill it.
    """
    # the plain name, not the found path: read_adf04() logs the name that it receives. The
    # tested log file carries the plain name for a plain file and for a compressed file.
    atom_filepath = adaspath / f"{atomic_number}_{ion_stage}.adf04"

    origin = ion_origins.get((atomic_number, ion_stage)) or ion_origins.get((atomic_number, None))

    if (atomic_number == 27) and (ion_stage == 3):
        # Co III takes its A-values from a separate file, so the collision rows are not needed
        ionization_energy_ev, adas_energylevels, upsilondict, _ = read_adf04(
            tyndall_co3_path / "adf04_v1",
            flog,
            args.electrontemperature,
            atomic_number,
            ion_stage,
            contents="The levels and the collision strengths",
            origin=origin,
        )

        adas_transitions: list[ADASTransitionRow] | pl.DataFrame = []
        transitionfile = tyndall_co3_path / "adf04rad_v1"
        log_comment(
            flog, ("transitiondata",), f"The transitions come from {path_in_data_folder(transitionfile, adasfolder)}."
        )
        with xopen_check_extension(transitionfile) as ftrans:
            for line in ftrans:
                row = line.split()
                id_upper = int(row[0])
                id_lower = int(row[1])
                A = float(row[2])
                if A > 2e-30:
                    append_adas_transition(
                        adas_energylevels,
                        adas_transitions,
                        id_lower,
                        id_upper,
                        A,
                        transitionfile,
                    )

    elif (atomic_number == 27) and (ion_stage == 4):
        # one level, the 3d6 5D4 ground state, with g = 2J + 1 as read_adf04() derives it
        adas_energylevels: list[ADASEnergyLevel] = [ADASEnergyLevel("groundstate", 1, 5, 2, 4.0, 0.0, 2 * 4.0 + 1, 0)]
        log_comment(
            flog,
            ("adata", "transitiondata"),
            "The reader holds the single level of Co IV, with no transition. No file gives it.",
        )
        adas_transitions = pl.DataFrame(schema=empty_transitions_schema)
        upsilondict: dict[tuple[int, int], float] = {}
        ionization_energy_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]
        log_comment(flog, ("adata",), nist_ionization_energy_comment)
        log_and_print(flog, f"The NIST table gives an ionisation energy of {ionization_energy_ev} eV.")

    elif find_file_check_extension(atom_filepath) is not None:
        # the same test that extend_ion_list() makes when it discovers these ions with a glob of
        # adaspath. So an adf04 file that discovery registers is one that this reader accepts.
        ionization_energy_ev, adas_energylevels, upsilondict, collisiondf = read_adf04(
            atom_filepath, flog, args.electrontemperature, atomic_number, ion_stage, origin=origin
        )

        adas_transitions: list[ADASTransitionRow] | pl.DataFrame = []

        # a radiative transition is a collision row with both file indices and an A-value. The width
        # of a line does not identify such a row, because a row can be one character shorter than
        # the widest.
        transitiondf = collisiondf.filter(
            pl.col("upper").is_not_null(), pl.col("lower").is_not_null(), pl.col("avalue") > 2e-30
        )

        # append_adas_transition() sorts each pair of file indices. So a file that gives the two
        # columns in the opposite order needs no special case here. The W II file does that.
        for id_upper, id_lower, A in transitiondf.select("upper", "lower", "avalue").iter_rows():
            append_adas_transition(
                adas_energylevels,
                adas_transitions,
                id_lower,
                id_upper,
                A,
                atom_filepath,
            )

    else:
        msg = f"No ADAS data available for Z={atomic_number} ion_stage {ion_stage} (no file {atom_filepath})"
        raise ValueError(msg)

    log_and_print(flog, f"The reader got {len(adas_transitions):d} transitions.")

    return ionization_energy_ev, adas_energylevels, adas_transitions, upsilondict


def _fill_co2_phixs(
    atomic_number,
    ion_stage,
    _levelcount: int,
    args,
    flog,
    photoionization_crosssections,
    photoionization_thresholds_ev,
    photoionization_targetfractions,
) -> None:
    """Fill the arrays with the QUB cross sections of Co II. The tables are for the CMFGEN levels of Co II."""
    log_comment(
        flog,
        ("phixsdata",),
        f"The cross sections come from the files 1 to 8 in {path_in_data_folder(tyndall_co3_path, adasfolder)}.",
    )
    for lowerlevelid in range(8):
        # the name of a cross section file is the level's number in the source data, which
        # counts from one
        filename = tyndall_co3_path / f"{lowerlevelid + 1:d}.gz"
        log_and_print(flog, f"The cross sections of level {lowerlevelid + 1} come from {path_for_log(filename)}.")
        ntargets = 4  # just the 4Fe ground quartet (the file has 40 target columns)
        # One space separates the columns, and every field is a number. So a null means that
        # the columns are not where the read expects them. A read of the first five columns
        # of the 41 costs a third of the time of a cut of every line into its parts.
        columnnames = ["energy", *(f"target{column}" for column in range(1, ntargets + 1))]
        photdata = (
            pl.scan_csv(filename, separator=" ", has_header=False, infer_schema_length=0)
            .select(pl.nth(column).cast(pl.Float64).alias(name) for column, name in enumerate(columnnames))
            .collect()
        )
        if photdata.null_count().sum_horizontal().item() > 0:
            msg = f"A value is missing in {filename}, so the columns are not in their expected positions."
            raise ValueError(msg)
        phixstables = {}

        # column n of the file holds the cross section to the upper ion's level id n - 1
        for targetcolumn in range(1, ntargets + 1):
            targetname = f"target{targetcolumn}"
            phixstable = photdata.filter(pl.col(targetname) > 0.0).select("energy", targetname).to_numpy()
            if len(phixstable) == 0:
                # nothing positive in this column, so there is no table to downsample. A skip
                # here leaves the target out of the fractions below, which is what a zero cross
                # section means. reduce_phixs_tables() would index an empty array and fail.
                log_detail(
                    flog,
                    ("phixsdata",),
                    "target with no positive cross section",
                    f"WARNING: level {lowerlevelid} has no positive cross section to target"
                    f" {targetcolumn - 1}, so the reader drops that target",
                )
                continue
            phixstables[targetcolumn] = phixstable

        reduced_phixs_dict = reduce_phixs_tables(
            phixstables,
            args.optimaltemperature,
            args.nphixspoints,
            args.phixsnuincrement,
            label=f"{ion_label(atomic_number, ion_stage)} QUB level id {lowerlevelid}",
        )
        combined = combine_phixs_routes(
            [(targetcolumn - 1, reduced) for targetcolumn, reduced in reduced_phixs_dict.items()]
        )
        if not combined.fractions:
            # the code assigns nothing for this level, so write_phixs_data() will skip it
            log_detail(
                flog,
                ("phixsdata",),
                "level with zero cross section to each target",
                f"WARNING: level {lowerlevelid + 1} has a zero cross section to each target, so it gets no table.",
            )
            continue
        for target, factor in combined.dropped:
            log_detail(
                flog,
                ("phixsdata",),
                "target below the cut",
                f"The cross section of level {lowerlevelid + 1} to the upper level {target + 1} sums to"
                f" {factor:.4e} Mb, less than {PHIXS_TARGET_FRACTION_CUT:.0%} of the total, so the output drops"
                " that target.",
            )

        # NaN, the arrays' initial value, says: the threshold energy comes from the level
        # energies, not from the first energy point of the cross section table
        photoionization_thresholds_ev[lowerlevelid] = np.nan
        photoionization_targetfractions[lowerlevelid] = combined.fractions
        photoionization_crosssections[lowerlevelid] = combined.table


def _fill_co3_phixs(
    atomic_number,
    ion_stage,
    levelcount: int,
    args,
    flog,
    photoionization_crosssections,
    photoionization_thresholds_ev,
    photoionization_targetfractions,
) -> None:
    """Fill the arrays with the QUB cross sections of Co III. The table is for the QUB levels of Co III."""
    # photoionize to a single level ion

    phixsvalues_const = [
        9.3380692,
        7.015829602,
        5.403975231,
        4.250372872,
        3.403086443,
        2.766835319,
        2.279802051,
        1.900685772,
        1.601177846,
        1.361433037,
        1.16725865,
        1.008321909,
        0.8769787,
        0.76749151,
        0.675496904,
        0.597636429,
        0.531296609,
        0.474423066,
        0.425385805,
        0.382880364,
        0.345854415,
        0.313452694,
        0.284975256,
        0.259845541,
        0.237585722,
        0.217797532,
        0.200147231,
        0.184353724,
        0.17017913,
        0.157421217,
        0.145907331,
        0.135489462,
        0.126040239,
        0.117449648,
        0.109622338,
        0.102475382,
        0.095936439,
        0.089942202,
        0.084437113,
        0.079372279,
        0.074704554,
        0.070395769,
        0.066412076,
        0.062723384,
        0.059302883,
        0.056126637,
        0.053173226,
        0.050423446,
        0.047860046,
        0.045467498,
        0.043231802,
        0.041140312,
        0.039181587,
        0.037345256,
        0.035621907,
        0.034002983,
        0.032480693,
        0.031047932,
        0.029698215,
        0.028425611,
        0.027224692,
        0.026090478,
        0.025018404,
        0.02400427,
        0.023044216,
        0.022134683,
        0.021272391,
        0.020454314,
        0.019677652,
        0.018939819,
        0.018238416,
        0.017571225,
        0.016936183,
        0.016331377,
        0.01575503,
        0.015205486,
        0.014681206,
        0.014180754,
        0.013702792,
        0.013246071,
        0.012809423,
        0.012391758,
        0.011992055,
        0.011609359,
        0.011242775,
        0.010891464,
        0.010554639,
        0.010231561,
        0.009921535,
        0.009623909,
        0.009338069,
        0.009063438,
        0.008799471,
        0.008545656,
        0.00830151,
        0.008066575,
        0.007840423,
        0.007622646,
        0.00741286,
        0.007210703,
    ]

    if abs(args.nphixspoints - 100) < 0.5 and abs(args.phixsnuincrement - 0.1) < 0.001:
        phixsvalues = np.array(phixsvalues_const)
    else:
        # the stop of 10.95 makes arange produce all 100 grid points from 1.0 to 10.9. A stop
        # of 10.9 produced 99 points, and the strict flag then dropped the table's last point.
        dict_phixstable = {"gs": np.array(list(zip(np.arange(1.0, 10.95, 0.1), phixsvalues_const, strict=True)))}
        phixsvalues = reduce_phixs_tables(
            dict_phixstable,
            args.optimaltemperature,
            args.nphixspoints,
            args.phixsnuincrement,
            label=f"{ion_label(atomic_number, ion_stage)} QUB constant table",
        )["gs"]

    log_comment(
        flog,
        ("phixsdata",),
        "The reader holds one cross section table for the ground quartet of Co III. Each higher level gets a table"
        " of zeros.",
    )
    # unlike the Co II branch above, every level deliberately gets a phixs entry. The ground
    # quartet gets the tabulated cross section, and the higher levels get an explicit
    # all-zero table.
    for levelid in range(levelcount):
        photoionization_thresholds_ev[levelid] = np.nan
        photoionization_targetfractions[levelid] = [(0, 1.0)]  # the upper ion's ground state
        if levelid < 4:
            photoionization_crosssections[levelid] = phixsvalues


def _read_qub_phixs(fill_arrays, atomic_number, ion_stage, levelcount: int, args, flog) -> PhixsData:
    """Read the photoionisation cross sections for one ion, downsampled onto the output grid.

    Returns the cross sections, the threshold energies and the upper-ion target fractions per
    level, all indexed by zero-based level id. Levels with no data keep an empty target list,
    which is how write_phixs_data() knows to skip them.

    fill_arrays is None for an ion with no QUB data. Such an ion gets the empty arrays, not
    zero-filled ones. The caller reads an empty cross section array as "no data", and then applies
    the hydrogenic estimate.
    """
    if fill_arrays is None:
        log_comment(
            flog,
            ("phixsdata",),
            "The ADAS data has no photoionisation cross sections for this ion.",
        )
        return PhixsData(np.empty((0, args.nphixspoints)), np.empty(0), targetfractions=[])
    log_source(flog, ("phixsdata",), "the cross sections", qub_cobalt_phixs_description)
    photoionization_crosssections = np.zeros((levelcount, args.nphixspoints))
    photoionization_targetfractions: list[list[tuple[int, float]]] = [[] for _ in range(levelcount)]
    photoionization_thresholds_ev = np.full(levelcount, np.nan)
    fill_arrays(
        atomic_number,
        ion_stage,
        levelcount,
        args,
        flog,
        photoionization_crosssections,
        photoionization_thresholds_ev,
        photoionization_targetfractions,
    )
    return PhixsData(
        photoionization_crosssections, photoionization_thresholds_ev, targetfractions=photoionization_targetfractions
    )


def get_level_valence_n(levelname: str) -> int | None:
    """Principal quantum number of the valence electron, read from an ADAS level name.

    Returns None for a name that it cannot parse. The caller, match_hydrogenic_phixs(), then
    gives the level no estimate and writes a warning to the log file.

    Kept separate from the other readers' versions. Each data source names its levels
    differently, so a shared parser would have to guess the convention of each name.
    """
    namesplit = levelname.split("_")
    # lower(): adf04 writes some orbitals in upper case ('3S2 3P6 3D5 4P1'), and the orbital
    # tests below compare against the lower-case orbital letters only
    part = namesplit[0].strip().lower()
    # `part` is empty for a name that starts with '_'. The removal of a parent term at the end of
    # the name below can empty it too. So test it again before every part[-1], which prevents an IndexError.
    if len(namesplit) < 2 or not part:
        return None

    if part[-1] == ")" and "(" in part:
        part = part[: part.rfind("(")]

    if not part:
        return None

    if part[-1] not in lchars_lower:
        # the last character must be the number of electrons in the orbital: remove it
        if not part[-1].isdigit():
            return None
        part = part.rstrip(string.digits)
    if not part:
        return None
    valenceorbital = part[-1]
    part = part.strip(lchars_lower)

    # the last run of digits of the label, with the character in front of it. The pattern
    # matches digits only, which split_count_and_n() requires.
    nmatch = re.search(r"(\D?)(\d+)$", part)
    if nmatch is None:
        return None

    # a lower-case orbital letter before the number means that the number is an electron count
    # of the previous orbital, then n. For example, the '24' in '3d24s' is two electrons and
    # n=4. The same rule as readkuruczdata applies, and it also reads '5s111s1' (5s1 11s1) as
    # n = 11. A space in front of the run separates two shells, so the run holds n alone
    if nmatch[1] and nmatch[1] in lchars_lower:
        return split_count_and_n(nmatch[1], nmatch[2], valenceorbital)

    return int(nmatch[2])

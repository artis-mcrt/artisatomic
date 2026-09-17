#!/usr/bin/env python3
"""Read levels, transitions and collision strengths from the QUB (Queen's University Belfast) data.

The Sr I file comes from Dougan, D. J., McElroy, N. E., Ballance, C. P., Ramsbottom, C. A. (2025),
MNRAS, 541, 367-383, doi:10.1093/mnras/staf1013. The Co data in co_tyndall comes from a private
communication (see atomic-data-qub/README.txt).
"""

import re
import string
import typing as t
from pathlib import Path

import numpy as np
import polars as pl

# the "qub_cobalt" handler reads the stages that the QUB data does not cover from the CMFGEN
# files. readhillierdata imports nothing from this module, so the import is not circular.
from artisatomic import readhillierdata
from artisatomic.base import add_handlers_if_not_set
from artisatomic.base import compression_extensions
from artisatomic.base import elsymbols
from artisatomic.base import empty_transitions_schema
from artisatomic.base import find_file_check_extension
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import ion_filename_pattern
from artisatomic.base import ions_from_filenames
from artisatomic.base import log_and_print
from artisatomic.base import path_for_log
from artisatomic.base import PhixsData
from artisatomic.base import PYDIR
from artisatomic.base import roman_numerals
from artisatomic.base import TESTMODE
from artisatomic.base import xopen_check_extension
from artisatomic.levelnames import convert_eissner_to_standard
from artisatomic.levelnames import get_config_parity
from artisatomic.levelnames import is_eissner_config
from artisatomic.levelnames import lchars
from artisatomic.levelnames import split_count_and_n
from artisatomic.phixs import combine_phixs_routes
from artisatomic.phixs import PHIXS_TARGET_FRACTION_CUT
from artisatomic.phixs import reduce_phixs_tables

qubpath = (PYDIR / ".." / "atomic-data-qub").resolve()
tyndall_co3_path = (qubpath / ("co_tyndall_test_sample" if TESTMODE else "co_tyndall")).resolve()

# the name of a data file, e.g. 26_2.adf04 or 26_2.adf04.zst
qub_filename_pattern = ion_filename_pattern(".adf04")


class QUBTransitionRow(t.NamedTuple):
    """One QUB bound-bound transition.

    nameto is the name of the upper level, and namefrom is the name of the lower level. The row
    carries the level ids, so add_level_ids_forbidden() does not join on the names.
    """

    lowerlevel: int
    upperlevel: int
    A: float
    nameto: str
    namefrom: str
    lambdaangstrom: float


class QUBEnergyLevel(t.NamedTuple):
    """One energy level of a QUB calculation."""

    levelname: str
    qub_id: int
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
    """Add every ion with a QUB adf04 file to ion_handlers under the "qub" handler."""
    # the files ship compressed or plain, so match every form of the name that a reader accepts
    qubfiles = [f for ext in compression_extensions for f in qubpath.glob(f"*_*.adf04{ext}")]
    # each name holds the atomic number and the ion stage, e.g. 26_2.adf04
    qubions = ions_from_filenames(qubfiles, qub_filename_pattern)

    return add_handlers_if_not_set(
        ion_handlers,
        qubions,
        "qub",
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
    differently, so the test is on the first field and not on a fixed column.
    """
    return adf04_first_field(line) == adf04_section_end


def adf04_field(index: int) -> pl.Expr:
    """Return an expression for the whitespace-separated field at index of the "line" column."""
    return pl.col("line").str.extract(rf"^\s*(?:\S+\s+){{{index}}}(\S+)", 1)


def adf04_float(index: int) -> pl.Expr:
    """Return an expression for the field at index as a float.

    adf04 writes the exponent with no "E": 1.23-04 means 1.23e-04. The replacement puts the "E"
    only after a digit or a point. A leading sign and a field that already has an "E" stay as
    they are.
    """
    return adf04_field(index).str.replace_all(r"([0-9.])([-+])", "${1}E${2}").cast(pl.Float64, strict=False)


def adf04_number(text: str) -> float:
    """Convert a number in the ADAS form, where the exponent has a sign and no letter ("5.00+03")."""
    return float(re.sub(r"(?<=[0-9.])([-+])", r"E\1", text))


decimal_number_pattern = r"\d+\.\d*"

adf04_header_regex = re.compile(rf"\s*[A-Za-z]{{1,2}}\s*\+\s*\d+\s+(\d+)\s+(\d+)\s+({decimal_number_pattern})\(.*\)")

# The groups are: qub_id, config, multiplicity (2S+1), L as a hexadecimal digit, J, energy above the ground level.
# The configuration column has no fixed width or format, so ".*" captures it.
adf04_level_regex = re.compile(
    rf"\s*(\d+)\s+(.*)\s+\((\d+)\)([0-9A-Fa-f])\(\s*({decimal_number_pattern})\)\s+({decimal_number_pattern})"
)


def _read_adf04_header(line: str, atomic_number: int, ion_stage: int, filepath: str | Path) -> float:
    """Return the ionisation energy in eV from the adf04 header. The header must name the requested ion."""
    headermatch = adf04_header_regex.match(line)
    if headermatch is None:
        msg = f"Cannot read the adf04 header line in {filepath}: {line.rstrip()!r}"
        raise ValueError(msg)
    header_atomic_number = int(headermatch[1])
    header_ion_stage = int(headermatch[2])
    ionization_energy_percm = float(headermatch[3])

    if atomic_number != header_atomic_number:
        msg = f"Atomic number ({atomic_number}) does not match that read from {filepath} ({header_atomic_number})"
        raise ValueError(msg)
    if ion_stage != header_ion_stage:
        msg = f"Ion stage ({ion_stage}) does not match that read from {filepath} ({header_ion_stage})"
        raise ValueError(msg)

    return ionization_energy_percm * hc_in_ev_cm


def _standardise_config(config: str) -> tuple[str, bool]:
    """Return the configuration in standard notation. The flag is True if the input was Eissner notation."""
    config = config.strip()

    if is_eissner_config(config):
        return convert_eissner_to_standard(config), True

    # The term in parentheses stays in upper case, e.g. "4P65S2(1S)" becomes "4p65s2(1S)".
    head, sep, tail = config.partition("(")
    return head.lower() + sep + tail, False


def read_adf04(
    filepath: str | Path, flog, electrontemperature: float, atomic_number: int, ion_stage: int
) -> tuple[float, list[QUBEnergyLevel], dict[tuple[int, int], float], pl.DataFrame]:
    """Read levels and effective collision strengths from an ADAS adf04 file.

    The collision strengths come from the tabulated temperature nearest to electrontemperature,
    as readhillierdata.read_coldata() picks them for the CMFGEN files.

    Returns four values:
    - the ionisation energy in eV;
    - the levels;
    - a dict of upsilon values keyed by a (lower, upper) pair of zero-based level ids;
    - the parsed collision rows.

    The caller takes the A-values from that frame, which saves a second read and a second parse
    of the file. The file numbers levels from one, and the rest of the code looks up id n at
    list index n - 1. The reader therefore checks that the level ids are contiguous and 1-based.
    """
    energylevels: list[QUBEnergyLevel] = []
    upsilondict: dict[tuple[int, int], float] = {}
    ionization_energy_ev = 0.0
    log_and_print(flog, f"Reading {path_for_log(filepath)}")
    uses_eissner_notation = False
    with xopen_check_extension(filepath) as fleveltrans:
        line = fleveltrans.readline()
        ionization_energy_ev = _read_adf04_header(line, atomic_number, ion_stage, filepath)
        # A note between two 'C-' rule lines can sit inside the level block, and the reader skips
        # its lines. The loops stop at the '-1' rows, so the reader never reads a note after the
        # collision block.
        atomic_group_note = False
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
            qub_id, config, multiplicity, l_hex, j, energy_percm = levelmatch.groups()
            config, was_eissner_notation = _standardise_config(config)
            if not uses_eissner_notation and was_eissner_notation:
                uses_eissner_notation = True
                log_and_print(flog, "Eissner notation detected for electron configuration")

            energylevel = QUBEnergyLevel(
                config, int(qub_id), int(multiplicity), int(l_hex, 16), float(j), float(energy_percm), 0.0, 0
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
                energylevel.qub_id,
            )

            g = 2 * energylevel.j + 1
            energylevel = energylevel._replace(g=g, parity=parity, levelname=levelname)
            energylevels.append(energylevel)

            # the transition and upsilon tables use these 1-based ids and the rest of the code
            # looks up id n at index n - 1. A non-contiguous file would therefore misattach every
            # transition. Not an assert: input validation must survive python -O.
            if energylevel.qub_id != len(energylevels):
                msg = (
                    f"adf04 level id {energylevel.qub_id} found at position {len(energylevels)} in {filepath}."
                    " Level ids must be contiguous and start at 1."
                )
                raise ValueError(msg)

        upsilonheader = fleveltrans.readline().split()
        temperatures = upsilonheader[2:]

        # ADAS writes auxiliary rows with a process code in the first field: R for recombination,
        # S and I for ionisation, P for proton impact. Only a row that starts with a level id is
        # a collision strength. A blank line is not a bad row, so the counter skips it.
        collision_lines: list[str] = []
        skipped_rows = 0
        for line in fleveltrans:
            firstfield = adf04_first_field(line)
            if firstfield == adf04_section_end:
                break
            if not firstfield:
                continue
            if not firstfield.isdigit():
                skipped_rows += 1
                continue
            collision_lines.append(line)

        # not an assert: input validation must survive python -O
        if not temperatures:
            msg = f"{filepath} names no temperatures for its collision strengths"
            raise ValueError(msg)
        temperature_values = [adf04_number(text) for text in temperatures]
        nearest_index = min(
            range(len(temperatures)), key=lambda index: abs(temperature_values[index] - electrontemperature)
        )
        log_and_print(
            flog,
            f"Selecting {temperature_values[nearest_index]:.0f} K for the collision strengths from"
            f" {', '.join(temperatures)}",
        )

        # each collision row holds these fields in order:
        #  - upper,
        #  - lower,
        #  - A-value,
        #  - one upsilon for each temperature,
        #  - the infinite-energy (Born) limit.
        # A cut of only the wanted fields needs about a third of the memory of a split of every
        # line into all of its columns.
        upsilonindex = 3 + nearest_index
        collisiondf = pl.DataFrame({"line": collision_lines}, schema={"line": pl.String}).select(
            adf04_field(0).cast(pl.Int64, strict=False).alias("upper"),
            adf04_field(1).cast(pl.Int64, strict=False).alias("lower"),
            adf04_float(2).alias("avalue"),
            adf04_float(upsilonindex).alias("upsilon"),
        )

        # a row that is too short, or that holds a value this cannot read, gives a null
        goodrows = collisiondf.drop_nulls(subset=["lower", "upper", "upsilon"])
        unreadable_rows = collisiondf.height - goodrows.height

        for lower, upper, upsilon in goodrows.select("lower", "upper", "upsilon").iter_rows():
            lower, upper = min(lower, upper), max(lower, upper)
            # a raise rather than an assert: this validates an input file, and the check
            # must survive python -O. Equal ids would store a self-transition.
            if not 1 <= lower < upper <= len(energylevels):
                msg = (
                    f"collision strength level ids {lower}, {upper} in {filepath} are outside"
                    f" the file's {len(energylevels)} levels"
                )
                raise ValueError(msg)

            # the file numbers levels from one; level ids are zero-based in memory. The log
            # messages keep the file's ids, because they are about the file's contents.
            levelidpair = (lower - 1, upper - 1)
            if levelidpair not in upsilondict:
                upsilondict[levelidpair] = upsilon
            else:
                log_and_print(
                    flog,
                    f"Duplicate upsilon value for transition {lower:d} to {upper:d}. The reader keeps"
                    f" {upsilondict[levelidpair]:5.2e} and ignores {upsilon:5.2e}",
                )

    log_and_print(flog, f"Read {len(energylevels):d} levels")
    log_and_print(flog, f"Read {len(upsilondict):d} effective collision strengths")
    if skipped_rows:
        log_and_print(flog, f"Skipped rows without a numeric level id: {skipped_rows:d}")
    if unreadable_rows:
        log_and_print(flog, f"Skipped collision rows that the reader could not parse: {unreadable_rows:d}")

    return ionization_energy_ev, energylevels, upsilondict, collisiondf


def append_qub_transition(qub_energylevels, qub_transitions, id_lower, id_upper, A, filepath) -> None:
    """Validate one radiative transition row and append it to the transition list.

    The ids are the file's 1-based level ids. The columns of a file do not always give the lower
    level first, so the function sorts the pair. read_adf04() sorts each collision pair the same
    way. A reversed pair would give a transition that the upsilon join misses.
    """
    id_lower, id_upper = min(id_lower, id_upper), max(id_lower, id_upper)
    # a raise rather than an assert: this validates an input file. A non-positive
    # id would wrap to the wrong level through a negative index. An id one past the
    # end would raise a bare IndexError that names neither the file nor the transition.
    if not 1 <= id_lower <= len(qub_energylevels) or not 1 <= id_upper <= len(qub_energylevels):
        msg = (
            f"transition level ids {id_lower}, {id_upper} in {filepath} are outside"
            f" the file's {len(qub_energylevels)} levels"
        )
        raise ValueError(msg)
    # the file numbers levels from one; level ids are zero-based in memory
    id_lower -= 1
    id_upper -= 1
    level_upper = qub_energylevels[id_upper]
    level_lower = qub_energylevels[id_lower]
    levelname_upper = level_upper.levelname
    levelname_lower = level_lower.levelname
    delta_percm = level_upper.energyabovegsinpercm - level_lower.energyabovegsinpercm
    lamdaangstrom = 1.0e8 / delta_percm if delta_percm != 0.0 else -1.0
    transition = QUBTransitionRow(
        lowerlevel=id_lower,
        upperlevel=id_upper,
        A=A,
        nameto=levelname_upper,
        namefrom=levelname_lower,
        lambdaangstrom=lamdaangstrom,
    )
    qub_transitions.append(transition)


# the ion stages that the QUB Co data covers: the Co III adf04 files and the single-level
# Co IV. For the other stages of a "qub_cobalt" ion, read_cobalt_levels_and_transitions() below
# takes the CMFGEN reader. read_qub_levels_and_transitions() has one branch for each stage in
# this set, so a new stage needs an entry here and a branch there.
qub_cobalt_stages: frozenset[int] = frozenset({3, 4})

# the ions whose photoionisation cross sections the QUB Co data covers, one branch each in
# read_qub_photoionizations(). read_cobalt_photoionizations() takes the CMFGEN phot files for
# every other stage of a "qub_cobalt" ion that has CMFGEN levels.
qub_phixs_ions: frozenset[tuple[int, int]] = frozenset({(27, 2), (27, 3)})


def read_cobalt_levels_and_transitions(atomic_number, ion_stage, flog, args):
    """Read one ion of the "qub_cobalt" handler: the QUB lists for its stages, the CMFGEN lists otherwise.

    Returns the same four values as read_qub_levels_and_transitions(). The CMFGEN collision
    strengths of a CMFGEN stage are the fourth value, as the QUB ones are for a QUB stage.
    """
    if ion_stage in qub_cobalt_stages:
        return read_qub_levels_and_transitions(atomic_number, ion_stage, flog, args)
    ionization_energy_ev, dflevels, dftransitions = readhillierdata.read_levels_and_transitions(
        atomic_number, ion_stage, flog
    )
    upsilondict = readhillierdata.read_coldata(atomic_number, ion_stage, dflevels, args, flog)
    return ionization_energy_ev, dflevels, dftransitions, upsilondict


def read_cobalt_photoionizations(atomic_number, ion_stage, dfenergylevels, args, flog) -> PhixsData:
    """Read the cross sections of a "qub_cobalt" ion: from the QUB data where it has them, else from CMFGEN.

    A stage with QUB levels stays on the QUB path even without QUB cross sections. Its levels
    carry no threshold wavelengths, so the CMFGEN phot files cannot apply to them.
    """
    if ion_stage in qub_cobalt_stages or (atomic_number, ion_stage) in qub_phixs_ions:
        return read_qub_photoionizations(
            atomic_number, ion_stage, levelcount=dfenergylevels.height, args=args, flog=flog
        )
    return readhillierdata.read_phixs_tables(atomic_number, ion_stage, dfenergylevels, args, flog)


def read_qub_levels_and_transitions(atomic_number, ion_stage, flog, args):
    """Read one ion from the QUB calculations.

    args gives -electrontemperature, which picks the tabulated collision strengths.

    The function reads the per-ion adf04 files, the Co III files in the co_tyndall directory, and
    the single level of Co IV. The Co III and the Co IV data have their own layouts. Also returns
    the effective collision strengths, so this reader
    supplies an upsilondict. Most other readers leave another module to fill it.
    """
    # the plain name, not the found path: read_adf04() logs the name that it receives. The
    # tested log files carry the plain name for a plain file and for a compressed file.
    atom_filepath = qubpath / f"{atomic_number}_{ion_stage}.adf04"

    if (atomic_number == 27) and (ion_stage == 3):
        # Co III takes its A-values from a separate file, so the collision rows are not needed
        ionization_energy_ev, qub_energylevels, upsilondict, _ = read_adf04(
            tyndall_co3_path / "adf04_v1", flog, args.electrontemperature, atomic_number, ion_stage
        )

        qub_transitions: list[QUBTransitionRow] | pl.DataFrame = []
        transitionfile = tyndall_co3_path / "adf04rad_v1"
        with xopen_check_extension(transitionfile) as ftrans:
            for line in ftrans:
                row = line.split()
                id_upper = int(row[0])
                id_lower = int(row[1])
                A = float(row[2])
                if A > 2e-30:
                    append_qub_transition(
                        qub_energylevels,
                        qub_transitions,
                        id_lower,
                        id_upper,
                        A,
                        transitionfile,
                    )

    elif (atomic_number == 27) and (ion_stage == 4):
        # one level, the 3d6 5D4 ground state, with g = 2J + 1 as read_adf04() derives it
        qub_energylevels: list[QUBEnergyLevel] = [QUBEnergyLevel("groundstate", 1, 5, 2, 4.0, 0.0, 2 * 4.0 + 1, 0)]
        qub_transitions = pl.DataFrame(schema=empty_transitions_schema)
        upsilondict: dict[tuple[int, int], float] = {}
        ionization_energy_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]
        log_and_print(flog, f"ionisation energy: {ionization_energy_ev} eV (NIST)")

    elif find_file_check_extension(atom_filepath) is not None:
        # the same test that extend_ion_list() makes when it discovers these ions with a glob of
        # qubpath. So an adf04 file that discovery registers is one that this reader accepts.
        ionization_energy_ev, qub_energylevels, upsilondict, collisiondf = read_adf04(
            atom_filepath, flog, args.electrontemperature, atomic_number, ion_stage
        )

        qub_transitions: list[QUBTransitionRow] | pl.DataFrame = []

        # a radiative transition is a collision row with both level ids and an A-value. The width
        # of a line does not identify such a row, because a row can be one character shorter than
        # the widest.
        transitiondf = collisiondf.filter(
            pl.col("upper").is_not_null(), pl.col("lower").is_not_null(), pl.col("avalue") > 2e-30
        )

        # append_qub_transition() sorts each pair of level ids. So a file that gives the two
        # columns in the opposite order needs no special case here. The W II file does that.
        for id_upper, id_lower, A in transitiondf.select("upper", "lower", "avalue").iter_rows():
            append_qub_transition(
                qub_energylevels,
                qub_transitions,
                id_lower,
                id_upper,
                A,
                atom_filepath,
            )

    else:
        msg = f"No QUB data available for Z={atomic_number} ion_stage {ion_stage} (no file {atom_filepath})"
        raise ValueError(msg)

    log_and_print(flog, f"Read {len(qub_transitions):d} transitions")

    return ionization_energy_ev, qub_energylevels, qub_transitions, upsilondict


def read_qub_photoionizations(atomic_number, ion_stage, levelcount: int, args, flog) -> PhixsData:
    """Read QUB photoionisation cross sections for one ion, downsampled onto the output grid.

    Returns the cross sections, the threshold energies and the upper-ion target fractions per
    level, all indexed by zero-based level id. Levels with no data keep an empty target list,
    which is how write_phixs_data() knows to skip them.

    An ion that this function has no data for gets the empty arrays, not zero-filled ones. The
    caller reads an empty cross section array as "no data", and then applies the hydrogenic
    estimate. A zero-filled array would pass as data and leave the ion with no cross sections.
    """
    photoionization_crosssections = np.zeros((levelcount, args.nphixspoints))
    # levels stay empty (write_phixs_data() skips them) unless the code below assigns real data
    photoionization_targetfractions: list[list[tuple[int, float]]] = [[] for _ in range(levelcount)]
    photoionization_thresholds_ev = np.full(levelcount, np.nan)

    if atomic_number == 27 and ion_stage == 2:
        for lowerlevelid in range(8):
            # the name of a cross section file is the level's number in the source data, which
            # counts from one
            filename = tyndall_co3_path / f"{lowerlevelid + 1:d}.gz"
            log_and_print(flog, f"Reading {path_for_log(filename)}")
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
                    log_and_print(
                        flog,
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
                label=f"Z={atomic_number} {elsymbols[atomic_number]} {roman_numerals[ion_stage]} QUB level id {lowerlevelid}",
            )
            combined = combine_phixs_routes(
                [(targetcolumn - 1, reduced) for targetcolumn, reduced in reduced_phixs_dict.items()]
            )
            if not combined.fractions:
                # the code assigns nothing for this level, so write_phixs_data() will skip it
                log_and_print(
                    flog, f"WARNING: all photoionisation targets for level {lowerlevelid} have zero cross section"
                )
                continue
            for target, factor in combined.dropped:
                log_and_print(
                    flog,
                    f"level {lowerlevelid}: target {target} is below the {PHIXS_TARGET_FRACTION_CUT:.0%} cut"
                    f" with {factor:.4e} Mb, so its route drops out",
                )

            # NaN, the arrays' initial value, says: the threshold energy comes from the level
            # energies, not from the first energy point of the cross section table
            photoionization_thresholds_ev[lowerlevelid] = np.nan
            photoionization_targetfractions[lowerlevelid] = combined.fractions
            photoionization_crosssections[lowerlevelid] = combined.table

    elif atomic_number == 27 and ion_stage == 3:
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
                label=f"Z={atomic_number} {elsymbols[atomic_number]} {roman_numerals[ion_stage]} QUB constant table",
            )["gs"]

        # unlike the Co II branch above, every level deliberately gets a phixs entry. The ground
        # quartet gets the tabulated cross section, and the higher levels get an explicit
        # all-zero table.
        for levelid in range(levelcount):
            photoionization_thresholds_ev[levelid] = np.nan
            photoionization_targetfractions[levelid] = [(0, 1.0)]  # the upper ion's ground state
            if levelid < 4:
                photoionization_crosssections[levelid] = phixsvalues

    else:
        log_and_print(flog, f"WARNING: no QUB photoionisation data for Z={atomic_number} ion_stage {ion_stage}")
        return PhixsData(np.empty((0, args.nphixspoints)), np.empty(0), targetfractions=[])

    return PhixsData(
        photoionization_crosssections, photoionization_thresholds_ev, targetfractions=photoionization_targetfractions
    )


def get_level_valence_n(levelname: str) -> int | None:
    """Principal quantum number of the valence electron, read from a QUB level name.

    Returns None for a name that it cannot parse. The caller, match_hydrogenic_phixs(), then
    gives the level no estimate and writes a warning to the ion log.

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

    if part[-1] not in lchars.lower():
        # the last character must be the number of electrons in the orbital: remove it
        if not part[-1].isdigit():
            return None
        part = part.rstrip(string.digits)
    if not part:
        return None
    valenceorbital = part[-1]
    part = part.strip(lchars.lower())

    # the last run of digits of the label, with the character in front of it. The pattern
    # matches digits only, which split_count_and_n() requires.
    nmatch = re.search(r"(\D?)(\d+)$", part)
    if nmatch is None:
        return None

    # a lower-case orbital letter before the number means that the number is an electron count
    # of the previous orbital, then n. For example, the '24' in '3d24s' is two electrons and
    # n=4. The same rule as readkuruczdata applies, and it also reads '5s111s1' (5s1 11s1) as
    # n = 11. A space in front of the run separates two shells, so the run holds n alone
    if nmatch[1] and nmatch[1] in lchars.lower():
        return split_count_and_n(nmatch[1], nmatch[2], valenceorbital)

    return int(nmatch[2])

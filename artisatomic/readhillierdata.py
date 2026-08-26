"""Read levels, transitions, collision strengths and cross sections from Hillier's CMFGEN data."""

import os
import re
import sys
import typing as t
from collections import defaultdict
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from string import ascii_uppercase

import numpy as np
import numpy.typing as npt
import polars as pl

import artisatomic
from artisatomic.base import elsymbols
from artisatomic.base import h_in_ev_seconds
from artisatomic.base import hc_in_ev_angstrom
from artisatomic.base import rewrite_file_as_utf8
from artisatomic.base import ryd_to_ev
from artisatomic.base import scan_file_lines
from artisatomic.levelnames import lchars

# need to also include collision strengths from e.g., o2col.dat

# Column layout of the level table in the oldest oscillator files, which carry no header line
# (no "!Format date"). Its fourth column is a threshold FREQUENCY, not an energy.
hillier_rowformat_noheader = "levelname g energyabovegsinpercm freqtentothe15hz lambdaangstrom hillierlevelid"


# The files disagree on which columns they carry, so only these are kept and every ion gets the
# same frame. Neither parity nor J is in the file: both are read from the level name, and both
# decide forbidden-ness.
class HillierEnergyLevel(t.NamedTuple):
    """One energy level read from a CMFGEN oscillator file."""

    levelname: str
    g: float
    energyabovegsinpercm: float
    lambdaangstrom: float
    hillierlevelid: int
    parity: int | None  # None where the level has no definite parity, e.g. a merged '1___'
    j: float | None  # None where the name carries no J, as a term-resolved level does not


class HillierTransition(t.NamedTuple):
    """One bound-bound transition read from a CMFGEN oscillator file, keyed by level name."""

    namefrom: str
    nameto: str
    f: float
    A: float
    lambdaangstrom: float
    i: int
    j: int
    hilliertransitionid: int


# every polars column is nullable, so an optional field needs no separate dtype
_pl_dtype_of = {
    str: pl.String,
    float: pl.Float64,
    int: pl.Int64,
    int | None: pl.Int64,
    float | None: pl.Float64,
}

# derived from the NamedTuples so the row classes stay the single source of the frame layouts
hillier_level_schema = pl.Schema(
    {name: _pl_dtype_of[fieldtype] for name, fieldtype in HillierEnergyLevel.__annotations__.items()}
)
hillier_transition_schema = pl.Schema(
    {name: _pl_dtype_of[fieldtype] for name, fieldtype in HillierTransition.__annotations__.items()}
)

# every schema column except the two read from the level name comes straight out of the table
hillier_required_filecolumns = tuple(colname for colname in hillier_level_schema if colname not in {"parity", "j"})


class IonFiles(t.NamedTuple):
    """The CMFGEN files holding one ion's levels, cross sections and collision data."""

    folder: str
    levelstransitionsfilename: str
    photfilenames: tuple[str, ...]
    coldatafilename: str


def phot_data_names(count: int) -> tuple[str, ...]:
    """Get the standard CMFGEN photoionisation file names for one ion."""
    return tuple(f"phot_data_{ascii_uppercase[index]}" for index in range(count))


default_ion_files = IonFiles("19apr23", "osc_data", phot_data_names(1), "col_data")

default_ion_stages: dict[int, Iterable[int]] = {
    6: range(1, 7),  # C
    7: range(5, 8),  # N
    8: (2, 3, 5, 6, 7, 8),  # O
    10: range(1, 9),  # Ne
    11: range(1, 10),  # Na
    12: range(1, 11),  # Mg
    13: range(1, 12),  # Al
    14: (1, *range(3, 13)),  # Si
    15: range(2, 12),  # P (I not in CMFGEN)
    16: range(1, 11),  # S
    17: range(4, 8),  # Cl (only ions IV to VII)
    18: range(1, 11),  # Ar
    19: range(1, 12),  # K
    20: range(1, 12),  # Ca
    21: range(1, 4),  # Sc (only I-III are in CMFGEN)
    22: range(2, 4),  # Ti (only II and III are in CMFGEN)
    24: range(1, 15),  # Cr
    25: range(2, 8),  # Mn (Mn I is not in CMFGEN)
    26: (2, 3, *range(5, 17)),  # Fe
    27: range(1, 10),  # Co
    28: range(1, 17),  # Ni
}

ions_data = {
    (atomic_number, ion_stage): default_ion_files
    for atomic_number, ion_stages in default_ion_stages.items()
    for ion_stage in ion_stages
}

# keys are (atomic number, ion stage)
ions_data |= {
    # H
    (1, 1): IonFiles("5dec96", "hi_osc.dat", ("hiphot.dat",), "hicol.dat"),
    (1, 2): IonFiles("", "", (), ""),
    # He
    (2, 1): IonFiles("11may07", "heioscdat_a7.dat_old", ("heiphot_a7.dat",), "heicol.dat"),
    (2, 2): IonFiles("5dec96", "he2_osc.dat", ("he2phot.dat",), "he2col.dat"),
    # C
    (6, 2): IonFiles("19apr23", "osc_data", phot_data_names(2), "col_data"),
    (6, 3): IonFiles("19apr23", "osc_data", phot_data_names(2), "col_data"),
    # N
    (7, 1): IonFiles("19apr23", "osc_data", phot_data_names(4), "col_data"),
    (7, 2): IonFiles("19apr23", "osc_data", phot_data_names(2), "col_data"),
    (7, 3): IonFiles("19apr23", "osc_data", phot_data_names(2), "col_data"),
    (7, 4): IonFiles("19apr23", "osc_data", phot_data_names(2), "col_data"),
    # O
    (8, 1): IonFiles("19apr23", "osc_data", phot_data_names(2), "col_data"),
    (8, 4): IonFiles("19apr23", "osc_data", phot_data_names(2), "col_data"),
    # F
    (9, 2): IonFiles("tst", "fin_osc", ("phot_data_a", "phot_data_b", "phot_data_c"), ""),
    (9, 3): IonFiles("tst", "fin_osc", ("phot_data_a", "phot_data_b", "phot_data_c", "phot_data_d"), ""),
    # Si
    (14, 2): IonFiles("19apr23", "osc_data", phot_data_names(2), "col_data"),
    # P
    (15, 4): IonFiles("19apr23", "osc_data", phot_data_names(2), "col_data"),
    # Ti IV has dummy files with a single level.
    # (22, 4): IonFiles("18oct00", "tkiv_osc.dat", ("phot_data.dat",), "col_guess.dat"),
    # V (only V I is in CMFGEN and it has a single level)
    # (23, 1): IonFiles("27may10", "vi_osc", ("vi_phot.dat",), "col_guess.dat"),
    # Fe
    (26, 1): IonFiles("19apr23", "osc_data", ("REV_PHOT_DATA",), "col_data"),
    (26, 4): IonFiles("19apr23", "feiv_osc_rev2", phot_data_names(1), "col_data"),
    # Cu, Zn and above are not in CMGFEN?
    # Ba
    # (56, 2): IonFiles("19apr23", "osc_data", phot_data_names(1), "col_data"),
}

elsymboltohilliercode = {
    "H": "HYD",
    "He": "HE",
    "C": "CARB",
    "N": "NIT",
    "O": "OXY",
    "F": "FLU",
    "Ne": "NEON",
    "Na": "NA",
    "Mg": "MG",
    "Al": "AL",
    "Si": "SIL",
    "P": "PHOS",
    "S": "SUL",
    "Cl": "CHL",
    "Ar": "ARG",
    "K": "POT",
    "Ca": "CA",
    "Sc": "SCAN",
    "Ti": "TIT",
    "V": "VAN",
    "Cr": "CHRO",
    "Mn": "MAN",
    "Fe": "FE",
    "Co": "COB",
    "Ni": "NICK",
    "Ba": "BAR",
}


atomic_number_to_hillier_code = {elsymbols.index(k): v for (k, v) in elsymboltohilliercode.items()}


class VY95PhixsFitRow(t.NamedTuple):
    """One Verner & Yakovlev (1995) analytic photoionization cross-section fit."""

    n: int
    l: int
    E_th_eV: float
    E_0: float
    sigma_0: float
    y_a: float
    P: float
    y_w: float


# keys are (n, l), values are energy in Rydberg or cross_section in Megabarns.
# Stored as arrays because get_hydrogenic_nl_phixstable() is called once per level per ion.
hyd_phixs_energygrid_ryd: dict[tuple[int, int], np.ndarray] = {}
hyd_phixs: dict[tuple[int, int], np.ndarray] = {}

# keys are n quantum number
hyd_gaunt_energygrid_ryd: dict[int, list[float]] = {}
hyd_gaunt_factor: dict[int, list[float]] = {}

# Maximum hydrogenic principal quantum number
max_hyd_l_n, max_hyd_gaunt_n = -1, -1


def hillier_ion_folder(atomic_number, ion_stage):
    """Directory of one ion's CMFGEN data, e.g. atomic_21jun23/FE/II for Fe II."""
    return str(
        (
            artisatomic.PYDIR
            / ".."
            / "atomic-data-hillier"
            / "atomic_21jun23"
            / atomic_number_to_hillier_code[atomic_number]
            / artisatomic.roman_numerals[ion_stage]
        ).resolve()
    )


def get_level_parity(config: str) -> int:
    """Parity of a Hillier level name: 0 even, 1 odd, -1 no definite parity.

    The trailing 'e'/'o' of the name is the parity wherever there is one, which is almost always:
    it agrees with every level name in the CMFGEN set whose term is otherwise readable, and it is
    the only thing that gets the intermediate-coupling names right ('3d5(4D)4po[3]', where the
    term letter belongs to the parent term '(4D)' rather than to the level).

    Names with no suffix fall back to summing l over the occupied orbitals. That leaves the
    levels that merge sub-levels of both parities and so have no parity to read: CMFGEN's merged
    high-l levels ('2s2_13w_2W'), its merged n-levels ('1___', '13___', g = 2n^2) and He I's
    merged singlets and triplets ('8SNG', '8TRP'). Those are -1, which
    add_level_ids_forbidden() never counts as a parity match.
    """
    config = config.split("[", maxsplit=1)[0]
    if not config:
        return -1

    # first, because a merged level spans both parities whatever the rest of the name says
    if artisatomic.has_merged_orbital(config):
        return -1

    if config[-1] == "e":
        return 0
    if config[-1] == "o":
        return 1

    configparity = artisatomic.get_config_parity(config)
    return -1 if configparity is None else configparity


def get_level_j(levelname: str, g: float | None = None) -> float | None:
    """J of a Hillier level name, or None where the name does not state one.

    CMFGEN writes J in brackets at the end of a J-resolved level's name, as a whole number or a
    half-integer fraction: '3d6_a5De[4]', '3d5(4D)4po[9/2]'. A term-resolved level has no
    bracket, and its g counts every J of the term, so nothing can be recovered from that either.

    The last bracketed or braced group is the one read. In pair coupling the brace holds K and
    the bracket J ('2p5(2P*<1/2>)3d_2{3/2}o[1]'); a name with a brace and nothing after it gives
    J there ('2s2_2p(2P<1/2>)4f_2{5/2}e'). Both come out right that way, checked against g for
    every such level in the corpus.

    Not every trailing bracket is a J. Si X and S X number their levels in brackets instead
    ('2p3p3Pe[3]' with g = 5, and a 3P term has no J = 3), and a few single levels elsewhere do
    the same. Pass the level's g and the value is returned only where g == 2J + 1, which is
    what J means; anything else is some other bracketed number and gives None.
    """
    match = re.search(r"[\[{](\d+)(?:/(\d+))?[\]}]\w*$", levelname)
    if match is None:
        return None

    numerator, denominator = match.group(1), match.group(2)
    if denominator is None:
        j = float(numerator)
    elif denominator == "2":
        j = float(numerator) / 2.0
    else:
        return None  # only halves are physical, so this is some other bracketed quantity

    if g is not None and abs(g - (2 * j + 1)) > 1e-6:
        return None  # the bracket held something other than J

    return j


def get_term_as_tuple(config: str) -> tuple[int, int, int]:
    """Read the LS term of a Hillier level name as (2S+1, L, parity), parity 0 even and 1 odd.

    Returns -1 for any component that cannot be read, which readers log rather than treat as an
    error. The parity is worked out separately by get_level_parity(), so it can still come back
    when the term itself is unreadable.
    """
    parity = get_level_parity(config)
    config = config.split("[", maxsplit=1)[0]

    if "{" in config and "}" in config:  # JJ coupling, no L and S
        if parity < 0:
            print(f"WARNING: Can't read parity from JJ coupling state '{config}'")
        return (-1, -1, parity)

    lposition = -1
    l = -1
    for charpos, char in reversed(list(enumerate(config))):
        if char in lchars:
            lposition = charpos
            l = lchars.index(char)
            break
    # lposition == 0 leaves no room for the multiplicity, and config[-1] would quietly wrap round
    # to the end of the name and read some other character as it
    if lposition < 1:
        return (-1, -1, parity)

    # The only L character can belong to a parenthesised parent term rather than to the level, as
    # in '3d5(4D)4po[3]', where reporting the '(4D)' would describe the parent and not this level.
    # An unclosed '(' before it is what tells them apart.
    if config.rfind("(", 0, lposition) > config.rfind(")", 0, lposition):
        return (-1, -1, parity)

    # a malformed name just means the term is unreadable, which says nothing about the parity
    try:
        twosplusone = int(config[lposition - 1])  # could this be two digits long?
    except ValueError:
        return (-1, -1, parity)

    return (twosplusone, l, parity)


def parse_transition_lines(dflines: pl.LazyFrame, filename: Path) -> pl.DataFrame:
    """Read the oscillator strengths table of a CMFGEN file into a transition frame.

    A transition line names its two levels, and a level name can hold a dash. The line is
    therefore cut at its first dash and at its last dash. The first dash separates the two
    names, and the last one separates the two level numbers of the "i-j" column. The parts
    between them keep their dashes, which the exponents of the f and the A values need.

    The file marks no end of the table. A line that is not a transition, e.g. a title, gives
    parts that fit neither layout below, and the filter drops it.
    """
    # only the first table is read, and a spelling mistake in a title must not hide the second one
    tablestart = (
        dflines.with_row_index()
        .filter(pl.col("line").str.contains(r"^\s*Osci(l|ll)ator strengths"))
        .select("index")
        .head(1)
        .collect()
    )
    if tablestart.height > 0:
        dflines = dflines.head(tablestart.item())

    # a line with no dash is doubled, as the parts of the empty middle are joined either side of it
    linewithspaces = (
        pl.when(pl.col("line").str.contains("-", literal=True))
        # the second pattern can only match at the last dash, and costs much less than the
        # equivalent greedy "^(.*)-" does
        .then(pl.col("line").str.replace("-", " ", literal=True).str.replace(r"-([^-]*)$", " ${1}"))
        .otherwise(pl.col("line") + "  " + pl.col("line"))
    )

    def part(index: int) -> pl.Expr:
        return pl.col("parts").list.get(index, null_on_oob=True)

    def as_float(index: int) -> pl.Expr:
        # the files write an exponent as D as well as E
        return part(index).str.replace_all("D", "E", literal=True).cast(pl.Float64, strict=False)

    dftransitions = (
        dflines.select(parts=linewithspaces.str.extract_all(r"\S+"))
        .with_columns(
            partcount=pl.col("parts").list.len(),
            f=as_float(2),
            A=as_float(3),
        )
        # the last two columns are absent from some files, which the transition id column follows.
        # A line that carries no number where the f and the A values belong is not a transition.
        .filter(
            ((pl.col("partcount") == 8) | ((pl.col("partcount") >= 10) & (part(-1) == "|")))
            & pl.col("f").is_not_null()
            & pl.col("A").is_not_null()
        )
        .select(
            namefrom=part(0),
            nameto=part(1),
            f=pl.col("f"),
            A=pl.col("A"),
            # the wavelength column holds a dash where the transition has no measured wavelength
            lambdaangstrom=part(4).cast(pl.Float64, strict=False).fill_null(-1.0),
            i=part(5).str.strip_chars_end("-").cast(pl.Int64),
            j=part(6).cast(pl.Int64),
            # the id column is masked before the cast, not after: the other layout holds a bar
            # there, and both arms of a when() are evaluated
            hilliertransitionid=pl.when(pl.col("partcount") == 8)
            .then(part(7))
            .cast(pl.Int64)
            .fill_null(pl.int_range(pl.len(), dtype=pl.Int64) + 1),
        )
        .collect()
    )

    for hilliertransitionid, entrynumber in (
        dftransitions.select("hilliertransitionid")
        .with_row_index("entrynumber", offset=1)
        .filter(pl.col("hilliertransitionid") != pl.col("entrynumber"))
        # with_row_index() puts its column first, so the two are put back in the unpacked order
        .select("hilliertransitionid", "entrynumber")
        .iter_rows()
    ):
        print(f"{filename} WARNING: Transition id {hilliertransitionid:d} found at entry number {entrynumber:d}")

    return dftransitions.cast(hillier_transition_schema)


def read_levels_and_transitions(
    atomic_number: int, ion_stage: int, flog
) -> tuple[float, pl.DataFrame, pl.DataFrame, defaultdict[str, int]]:
    """Read one ion, and rewrite its file as utf-8 first if CMFGEN wrote it in iso-8859-1.

    Python raises a UnicodeDecodeError for such a file and polars raises a ComputeError, so
    both reads of read_levels_and_transitions_from_file() are covered here. The file is
    converted once, so the second call reads it.
    """
    try:
        return read_levels_and_transitions_from_file(atomic_number, ion_stage, flog)
    except (UnicodeDecodeError, pl.exceptions.ComputeError):
        # a failure that the encoding does not explain belongs to the caller
        if not rewrite_file_as_utf8(hillier_osc_filename(atomic_number, ion_stage)):
            raise

    return read_levels_and_transitions_from_file(atomic_number, ion_stage, flog)


def hillier_osc_filename(atomic_number: int, ion_stage: int) -> Path:
    """Path of one ion's CMFGEN oscillator file, which holds its levels and its transitions."""
    return Path(
        hillier_ion_folder(atomic_number, ion_stage),
        ions_data[atomic_number, ion_stage].folder,
        ions_data[atomic_number, ion_stage].levelstransitionsfilename,
    )


def read_levels_and_transitions_from_file(
    atomic_number: int, ion_stage: int, flog
) -> tuple[float, pl.DataFrame, pl.DataFrame, defaultdict[str, int]]:
    """Read one ion's levels and bound-bound transitions from its CMFGEN oscillator file.

    Returns the ionization energy in eV, a level frame, a transition frame, and the number of
    transitions touching each level name. Levels with no transitions are dropped. The level
    table's columns are read from the header the file carries above it, so the layout varies by
    ion; see hillier_rowformat_noheader for the oldest files, which have no header.

    Transitions are keyed by level name here. add_level_ids_forbidden() joins the level ids on
    afterwards, once the level frame has been given its ids.
    """
    transition_count_of_level_name: defaultdict[str, int] = defaultdict(int)
    hillier_ionization_energy_ev = 0.0

    if atomic_number == 1 and ion_stage == 2:
        # a bare proton: a single dummy level, and no oscillator file to read
        return (
            hillier_ionization_energy_ev,
            pl.DataFrame(
                [
                    HillierEnergyLevel(
                        levelname="I",
                        g=10.0,
                        energyabovegsinpercm=0.0,
                        lambdaangstrom=0.0,
                        hillierlevelid=1,
                        parity=0,
                        j=None,  # a bare nucleus, with no transitions for a J to bear on
                    )
                ],
                schema=hillier_level_schema,
                orient="row",
            ),
            pl.DataFrame(schema=hillier_transition_schema),
            transition_count_of_level_name,
        )

    filename = hillier_osc_filename(atomic_number, ion_stage)

    artisatomic.log_and_print(flog, f"Reading {artisatomic.path_for_log(filename)}")

    levelrows: list[HillierEnergyLevel] = []
    levels_without_parity: list[str] = []

    prev_line = ""
    # threads=0 decompresses in this process. The default starts a process that writes into a
    # pipe, which reports a broken pipe when the reader stops at the end of the level table.
    fhillierosc = artisatomic.xopen_check_extension(filename, threads=0)
    # the two loops below read the header and the levels. polars reads the transitions, and
    # starts at the line after the last line that they read.
    linesread = 0
    expected_energy_levels = -1
    expected_transitions = -1
    row_format_energy_level = None
    format_date = "NOT_SPECIFIED"
    for line in fhillierosc:
        linesread += 1
        row = line.split()
        if (
            re.match(r"x*\*{5,}", line) and prev_line
        ):  # The x is not a mistake, one of the lines of stars somewhere starts with an x and breaks otherwise
            if atomic_number == 26 and ion_stage == 8:  # Fe VIII has its own bespoke header...
                print("Fe VIII has a bespoke header")
                row_format_energy_level = (
                    "levelname g energyabovegsinpercm thresholdenergyev freqtentothe15hz lambdaangstrom hillierlevelid"
                )
            else:
                headerline = prev_line
                headerline = headerline.replace("ID", "hillierlevelid")
                headerline = headerline.replace("E(cm^-1)", "energyabovegsinpercm")
                headerline = headerline.replace("10^15 Hz", "freqtentothe15hz")
                headerline = headerline.replace("eV", "thresholdenergyev")
                headerline = headerline.replace("Lam(A)", "lambdaangstrom")
                headerline = headerline.replace("ARAD", "arad")
                row_format_energy_level = "levelname " + " ".join(headerline.lower().split())

            print("File contains columns:")
            print(f"  {row_format_energy_level}")
        elif line.rstrip().endswith("!Number of energy levels"):
            expected_energy_levels = int(row[0])
            artisatomic.log_and_print(flog, f"File specifies {expected_energy_levels:d} levels")
        elif line.rstrip().endswith("!Number of transitions"):
            expected_transitions = int(row[0])
            artisatomic.log_and_print(flog, f"File specifies {expected_transitions:d} transitions")
        elif len(row) == 3 and row[1] == "!Format" and row[2] == "date":
            format_date = row[0]
            print(f"Format date: {format_date}")

        if expected_energy_levels >= 0 and not row:
            break
        prev_line = line.strip()

    if not row_format_energy_level:
        # the files that predate the header also predate the format date line
        if format_date != "NOT_SPECIFIED":
            msg = f"{filename} gives a format date of {format_date} but carries no column header"
            raise ValueError(msg)
        row_format_energy_level = hillier_rowformat_noheader
        print("File has no column header, assuming columns:")
        print(f"  {row_format_energy_level}")

    # the file columns vary by ion, so find where the ones we keep sit in each row
    headercolumns = row_format_energy_level.split()
    colindex = {colname: index for index, colname in enumerate(headercolumns)}
    levelcolcount = len(headercolumns)
    if len(colindex) != levelcolcount:
        # a repeated header token would silently shadow an earlier column's position
        msg = f"Level table header of {filename} contains duplicate column names: {row_format_energy_level}"
        raise ValueError(msg)
    missingcolumns = [colname for colname in hillier_required_filecolumns if colname not in colindex]
    if missingcolumns:
        msg = (
            f"Level table of {filename} is missing the {', '.join(missingcolumns)} column(s):"
            f" it has {row_format_energy_level}"
        )
        raise ValueError(msg)

    for line in fhillierosc:
        linesread += 1
        row = line.split()
        # check for right number of columns and that are all numbers except first column
        if len(row) == levelcolcount and all(map(artisatomic.isfloat, row[1:])):
            hillierlevelid = int(row[colindex["hillierlevelid"]].lstrip("-"))
            levelname = row[colindex["levelname"]]
            energyabovegsinpercm = float(row[colindex["energyabovegsinpercm"]].replace("D", "E"))
            lambdaangstrom = float(row[colindex["lambdaangstrom"]].replace("D", "E"))
            (twosplusone, _l, parity) = get_term_as_tuple(levelname)
            ismerged = parity < 0
            isjjcoupled = "{" in levelname and "}" in levelname

            if ismerged:
                # No definite parity: a merged level, which is normal CMFGEN, or a name we
                # could not read. Null rather than a number, so that add_level_ids_forbidden()
                # cannot match it against another level's absent parity.
                parity = None
                levels_without_parity.append(levelname)

            levelrows.append(
                HillierEnergyLevel(
                    levelname=levelname,
                    g=float(row[colindex["g"]]),
                    energyabovegsinpercm=energyabovegsinpercm,
                    lambdaangstrom=lambdaangstrom,
                    hillierlevelid=hillierlevelid,
                    parity=parity,
                    j=get_level_j(levelname, g=float(row[colindex["g"]])),
                )
            )

            # -1 indicates that the term could not be interpreted. JJ-coupled names have no
            # LS term by construction and merged levels are summarised once below, so neither
            # is worth a line here; what is left is a name we expected to read and could not.
            if twosplusone == -1 and atomic_number > 1 and not isjjcoupled and not ismerged:
                artisatomic.log_and_print(flog, f"Can't find LS term in Hillier level name '{levelname}'")

            # if this is the ground state
            if energyabovegsinpercm < 1.0:
                hillier_ionization_energy_ev = hc_in_ev_angstrom / lambdaangstrom

            if hillierlevelid != len(levelrows):
                artisatomic.log_and_print(
                    flog,
                    f"Hillier levels mismatch: id {len(levelrows):d} found at entry number {hillierlevelid:d}",
                )
                sys.exit(1)

        if re.match(r"^\s*Osci(l|ll)ator strengths", line) and len(levelrows) > 0:
            break

    artisatomic.log_and_print(flog, f"Read {len(levelrows):d} levels")
    if levels_without_parity:
        # Normal for ions with merged levels, so this is a count and a sample rather than a
        # warning per level: H I and He II are merged all the way down.
        artisatomic.log_and_print(
            flog,
            f"{len(levels_without_parity):d} of {len(levelrows):d} levels have no definite parity"
            f" (every transition touching one is treated as permitted), e.g."
            f" {', '.join(levels_without_parity[:5])}",
        )
    if len(levelrows) != expected_energy_levels:
        msg = f"{filename} declares {expected_energy_levels} levels but {len(levelrows)} were read"
        raise ValueError(msg)

    fhillierosc.close()

    # the rest of the file holds the transitions, which polars reads rather than Python: one ion
    # carries half a million of them
    dftransitions = parse_transition_lines(scan_file_lines(filename, skip_lines=linesread), filename)

    for namecolumn in ("namefrom", "nameto"):
        for levelname, count in dftransitions[namecolumn].value_counts().iter_rows():
            transition_count_of_level_name[levelname] += count

    artisatomic.log_and_print(flog, f"Read {dftransitions.height:d} transitions")
    if dftransitions.height != expected_transitions:
        msg = f"{filename} declares {expected_transitions} transitions but {dftransitions.height} were read"
        raise ValueError(msg)

    # filter out levels with no transitions: a name is only counted above when one is read for it
    dfhillier_energy_levels = pl.DataFrame(levelrows, schema=hillier_level_schema, orient="row").filter(
        pl.col("levelname").is_in(set(transition_count_of_level_name))
    )

    return (
        hillier_ionization_energy_ev,
        dfhillier_energy_levels,
        dftransitions,
        transition_count_of_level_name,
    )


# cross section types
phixs_type_labels = {
    0: "Constant (always zero?) [constant]",
    1: "Seaton formula fit [sigma_o, alpha, beta]",
    2: "Hydrogenic split l (z states, n > 11) [n, l_start, l_end]",
    3: "Hydrogenic pure n level (all l, n >= 13) [scale, n]",
    4: "Used for CIV rates from Leobowitz (JQSRT 1972,12,299) (6 numbers)",
    5: "Opacity project fits (from Peach, Saraph, and Seaton (1988) (5 numbers)",
    6: "Hummer fits to the opacity cross-sections for HeI",
    7: "Modified Seaton formula fit (cross section zero until offset edge)",
    8: "Modified hydrogenic split l (cross-section zero until offset edge) [n,l_start,l_end,nu_o]",
    9: "Verner & Yakolev 1995 ground state fits (multiple shells)",
    20: "Opacity Project: smoothed [number of data points]",
    21: "Opacity Project: scaled, smoothed [number of data points]",
    22: "energy is in units of threshold, cross section in Megabarns? [number of data points]",
}


def read_phixs_tables(
    atomic_number, ion_stage, dfenergy_levels: pl.DataFrame, args, flog
) -> tuple[npt.NDArray[np.float64], list[list[tuple[str, float]] | None], npt.NDArray[np.float64]]:
    """Read one ion's CMFGEN photoionization cross sections, downsampled onto the output grid.

    Returns the cross sections, the target configurations and their fractions per level, and the
    threshold energies, all indexed by zero-based level id. A level with no data keeps None in
    the target list, which get_photoiontargetfractions() relies on to tell it apart from a level
    whose targets all came out zero. The files give fit coefficients rather than tabulated cross
    sections for most levels; see phixs_type_labels for the fit types and the get_*_phixstable()
    functions that evaluate them.
    """
    msg = "Hydrogenic tables not loaded; call read_hyd_phixsdata() before read_phixs_tables()"
    if max_hyd_l_n == -1:
        raise RuntimeError(msg)
    if max_hyd_gaunt_n == -1:
        raise RuntimeError(msg)

    # pulled out of the frame once: the loops below index these per level, per cross-section table
    levelcount = dfenergy_levels.height
    levelnames: list[str] = dfenergy_levels["levelname"].to_list()
    lambdaangstroms: list[float] = dfenergy_levels["lambdaangstrom"].to_list()

    # first level index of each name, with and without the [J] suffix: tables are matched to levels
    # by name, and rescanning the level list per table would be O(levels x tables)
    firstlevelindex_of_levelname: dict[str, int] = {}
    firstlevelindex_of_levelnamenoJ: dict[str, int] = {}
    for levelindex, levelname in enumerate(levelnames):
        firstlevelindex_of_levelname.setdefault(levelname, levelindex)
        firstlevelindex_of_levelnamenoJ.setdefault(levelname.split("[")[0], levelindex)

    # this gets partially overwritten anyway
    photoionization_crosssections = np.zeros((levelcount, args.nphixspoints))
    photoionization_thresholds_ev = np.full(levelcount, np.nan)
    # None means "no photoionisation data for this level", which get_photoiontargetfractions()
    # relies on, so this must not be initialised to empty lists
    photoionization_targetconfig_fractions: list[list[tuple[str, float]] | None] = [None] * levelcount

    # charge of the ion left behind, i.e. CMFGEN's ZION (= ZXzV from the oscillator file, which
    # equals the ionisation stage for every ion here)
    zion = ion_stage
    photfilenames = ions_data[atomic_number, ion_stage].photfilenames
    phixstables = [{} for _ in photfilenames]
    phixstargets = ["" for _ in photfilenames]
    reduced_phixs_dict = {}
    # the target whose table is the one kept in reduced_phixs_dict, so the normalisation below can
    # divide by that target's fraction and recover the level's total cross section, plus that
    # table's threshold cross section, which is what decides between competing targets
    kepttarget_of_levelname: dict[str, str] = {}
    keptthreshold_of_levelname: dict[str, float] = {}
    phixs_targetconfigfactors_of_levelname = defaultdict(list)
    num_levelnames_with_zero_crosssection = 0

    j_splitting_on = False  # hopefully this is either on or off for all photoion files associated with a given ion

    # sets, not lists: only the distinct level count per type is reported, and a list needed a
    # linear scan per record to dedupe, which is quadratic in the level count of the ion
    phixs_type_levels: defaultdict[int, set[str]] = defaultdict(set)
    unknown_phixs_types = []
    for filenum, photfilename in enumerate(photfilenames):
        filename = Path(
            hillier_ion_folder(atomic_number, ion_stage), ions_data[atomic_number, ion_stage].folder, photfilename
        )

        artisatomic.log_and_print(flog, f"Reading {artisatomic.path_for_log(filename)}")

        with artisatomic.xopen_check_extension(filename) as fhillierphot:
            lowerlevelindex = -1
            lowerlevelname = ""
            targetlevelname = ""
            numpointsexpected = 0
            pointnumber = 0
            crosssectiontype = -1
            fitcoefficients: list[t.Any] = []

            # Used to skip problematic lines in Fe VIII and Ni X phot_data_A (see below)
            in_header = False

            for line in fhillierphot:
                row = line.split()

                # Every marker below is a trailing "!..." comment, so a line with no '!' at all
                # cannot match any of them. 94% of a phot file is two-column cross-section data,
                # and testing that once here skips seven " ".join() calls per such line.
                has_marker = "!" in line

                if has_marker and len(row) >= 2 and " ".join(row[-4:]) == "!Final state in ion":
                    # this is not used because the upper ion's levels are not known at this time
                    targetlevelname = row[0]
                    artisatomic.log_and_print(flog, "Photoionisation target: " + targetlevelname)
                    if "[" in targetlevelname:
                        print("STOP! target level contains a bracket (is J-split?)")
                        sys.exit(1)
                    if targetlevelname in phixstargets:
                        print("STOP! Multiple phixs files for the same target configuration")
                        sys.exit(1)
                    phixstargets[filenum] = targetlevelname

                if has_marker and len(row) >= 2 and " ".join(row[-3:]) == "!Split J levels":
                    if row[0].lower() == "true":
                        j_splitting_on = True
                        artisatomic.log_and_print(flog, "File specifies J-splitting enabled")
                    elif row[0].lower() == "false":
                        if j_splitting_on:
                            print("STOP! J-splitting disabled here, but was previously enabled for this ion")
                            sys.exit(1)
                        j_splitting_on = False
                    else:
                        print(f'STOP! J-splitting not true or false: "{row[0]}"')
                        sys.exit(1)

                if has_marker and (
                    (len(row) >= 2 and " ".join(row[-2:]) == "!Configuration name")
                    or " ".join(row[-3:]) == "!Configuration name [*]"
                ):
                    if not in_header:
                        artisatomic.log_and_print(
                            flog, f"WARNING: no photoionisation target ({line.strip()}), skipping to the next line"
                        )
                        continue  # We are probably in Fe VIII or Ni X phot_data_A, where there a bunch of lines before the header that end in "!Configuration name" and confuse things...

                    lowerlevelname = row[0]
                    # with J splitting the name (including any [J] suffix) maps to exactly one
                    # level; without it, strip the suffix so the table covers the configuration
                    if not j_splitting_on and "[" in lowerlevelname:
                        lowerlevelname = lowerlevelname.split("[")[0]
                    fitcoefficients = []
                    numpointsexpected = 0
                    # first matching level (without J splitting, several may differ by J). A name
                    # with no matching level falls back to index 0, so its fit is evaluated at the
                    # ground state's threshold wavelength, but that table is never used: the
                    # levelindices_of_matchname mapping at the end of this function is keyed the
                    # same way, so it finds no level for the name and drops the table. The phot
                    # files routinely cover levels the oscillator file does not (1145 of them for
                    # Co II), which is why this is a silent fallback rather than an error.
                    lowerlevelindex = (
                        firstlevelindex_of_levelname if j_splitting_on else firstlevelindex_of_levelnamenoJ
                    ).get(lowerlevelname, 0)
                    if not targetlevelname:
                        print("ERROR: no upper level name")
                        sys.exit(1)

                if has_marker and len(row) >= 2 and " ".join(row[-3:]) == "!Screened nuclear charge":
                    # CMFGEN's ZION comes from the oscillator file: RDPHOT_GEN_V2 never reads
                    # this field, and the two disagree for 29 shipped files. Keep ion_stage (which
                    # matches the oscillator value for every ion in ions_data) and just report it.
                    zion_from_photfile = int(float(row[0]))
                    if zion_from_photfile != ion_stage:
                        artisatomic.log_and_print(
                            flog,
                            f"WARNING: ignoring screened nuclear charge {zion_from_photfile} in {photfilename},"
                            f" which disagrees with ion_stage {ion_stage}",
                        )

                if has_marker and len(row) >= 2 and " ".join(row[1:]) == "!Number of cross-section points":
                    numpointsexpected = int(row[0])
                    pointnumber = 0

                if has_marker and len(row) >= 2 and " ".join(row[1:]) == "!Cross-section unit":
                    in_header = True  # All phot_data_* in 19apr23 have this line
                    if row[0] != "Megabarns":
                        print(f"Wrong cross-section unit: {row[0]}")
                        sys.exit(1)

                # a line carrying a "!..." marker has text in it, so it can never be all floats.
                # Short-circuiting on that skips the parse attempt on every header line.
                row_is_all_floats = not has_marker and all(map(artisatomic.isfloat, row))
                if crosssectiontype == 0:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace("D", "E")))

                        if fitcoefficients[-1] != 0.0:
                            print("ERROR: Cross section type 0 has non-zero number after it")
                            sys.exit(1)

                elif crosssectiontype == 1:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace("D", "E")))
                        if len(fitcoefficients) == 3:
                            lambda_angstrom = abs(lambdaangstroms[lowerlevelindex])
                            phixstables[filenum][lowerlevelname] = get_seaton_phixstable(
                                lambda_angstrom, *fitcoefficients
                            )
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])

                elif crosssectiontype == 2:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(int(float(row[0].replace("D", "E"))))
                        if len(fitcoefficients) == 3:
                            n, l_start, l_end = fitcoefficients

                            if n > max_hyd_l_n:
                                artisatomic.log_and_print(
                                    flog, f"WARNING: n ({n}) > max_hyd_l_n ({max_hyd_l_n}), skipping table"
                                )
                                continue

                            if l_end > n - 1:
                                artisatomic.log_and_print(flog, f"ERROR: can't have l_end = {l_end} > n - 1 = {n - 1}")
                            else:
                                lambda_angstrom = abs(lambdaangstroms[lowerlevelindex])
                                phixstables[filenum][lowerlevelname] = get_hydrogenic_nl_phixstable(
                                    lambda_angstrom, n, l_start, l_end
                                )
                                numpointsexpected = len(phixstables[filenum][lowerlevelname])

                elif crosssectiontype == 3:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace("D", "E")))
                        if len(fitcoefficients) == 2:
                            scale, n = fitcoefficients

                            if n > max_hyd_gaunt_n:
                                artisatomic.log_and_print(
                                    flog, f"WARNING: n ({n}) > max_hyd_l_n ({max_hyd_gaunt_n}), skipping table"
                                )
                                continue

                            n = int(n)
                            lambda_angstrom = abs(lambdaangstroms[lowerlevelindex])
                            # scale the cross sections but not the energy grid
                            phixstable = get_hydrogenic_n_phixstable(lambda_angstrom, n)
                            phixstable[:, 1] *= scale
                            phixstables[filenum][lowerlevelname] = phixstable

                            numpointsexpected = len(phixstables[filenum][lowerlevelname])

                elif crosssectiontype == 5:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace("D", "E")))
                        if len(fitcoefficients) == 5:
                            lambda_angstrom = abs(lambdaangstroms[lowerlevelindex])
                            phixstables[filenum][lowerlevelname] = get_opproject_phixstable(
                                lambda_angstrom, *fitcoefficients
                            )
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])

                elif crosssectiontype == 6:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace("D", "E")))
                        if len(fitcoefficients) == 8:
                            lambda_angstrom = abs(lambdaangstroms[lowerlevelindex])
                            phixstables[filenum][lowerlevelname] = get_hummer_phixstable(
                                lambda_angstrom, *fitcoefficients
                            )
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])

                elif crosssectiontype == 7:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace("D", "E")))
                        if len(fitcoefficients) == 4:
                            lambda_angstrom = abs(lambdaangstroms[lowerlevelindex])
                            phixstables[filenum][lowerlevelname] = get_seaton_phixstable(
                                lambda_angstrom, *fitcoefficients
                            )
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])

                elif crosssectiontype == 8:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        if len(fitcoefficients) <= 2:
                            # first three params should be integers
                            fitcoefficients.append(int(float(row[0].replace("D", "E"))))
                        else:
                            # fourth param is a float
                            fitcoefficients.append(float(row[0].replace("D", "E")))

                        if len(fitcoefficients) == 4:
                            n, l_start, l_end, nu_o = fitcoefficients

                            if n > max_hyd_l_n:
                                artisatomic.log_and_print(
                                    flog, f"WARNING: n ({n}) > max_hyd_l_n ({max_hyd_l_n}), skipping table"
                                )
                                continue

                            if l_end > n - 1:
                                artisatomic.log_and_print(flog, f"ERROR: can't have l_end = {l_end} > n - 1 = {n - 1}")
                            else:
                                lambda_angstrom = abs(lambdaangstroms[lowerlevelindex])
                                phixstables[filenum][lowerlevelname] = get_hydrogenic_nl_phixstable(
                                    lambda_angstrom, n, l_start, l_end, nu_o=nu_o, zion=zion
                                )
                                numpointsexpected = len(phixstables[filenum][lowerlevelname])

                elif crosssectiontype == 9:
                    if len(row) == 8 and numpointsexpected > 0:
                        fitcoefficients.append(
                            VY95PhixsFitRow(int(row[0]), int(row[1]), *[float(x.replace("D", "E")) for x in row[2:]])
                        )

                        if len(fitcoefficients) * 8 == numpointsexpected:
                            lambda_angstrom = abs(lambdaangstroms[lowerlevelindex])
                            phixstables[filenum][lowerlevelname] = get_vy95_phixstable(lambda_angstrom, fitcoefficients)
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])

                elif crosssectiontype in {20, 21, 22}:  # sampled data points
                    if len(row) == 2 and row_is_all_floats and lowerlevelname:
                        if lowerlevelname not in phixstables[filenum]:
                            phixstables[filenum][lowerlevelname] = np.zeros((numpointsexpected, 2))

                        lambda_angstrom = abs(lambdaangstroms[lowerlevelindex])
                        thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev
                        enryd = float(row[0].replace("D", "E"))

                        if crosssectiontype in {
                            20,
                            21,
                            22,
                        }:  # the x value is actually a fraction of the threshold, not an energy
                            if pointnumber == 0 and abs(enryd - 1.0) > 0.5:
                                print(
                                    f"{lowerlevelname} cross section type:{crosssectiontype}, {enryd:.3f} is not near"
                                    f" one? might be energy instead? E_threshold = {thresholdenergyryd:.3f} Ry"
                                )
                            enryd *= thresholdenergyryd

                        xspoint = enryd, float(row[1].replace("D", "E"))
                        phixstables[filenum][lowerlevelname][pointnumber] = xspoint

                        if pointnumber > 0:
                            curenergy = phixstables[filenum][lowerlevelname][pointnumber][0]
                            prevenergy = phixstables[filenum][lowerlevelname][pointnumber - 1][0]
                            if curenergy == prevenergy:
                                print(
                                    f"WARNING: photoionization table for {lowerlevelname} first column duplicated "
                                    f"energy value of {prevenergy}"
                                )
                            elif curenergy < prevenergy:
                                print(
                                    f"ERROR: photoionization table for {lowerlevelname} first column decreases "
                                    f"with energy {prevenergy} followed by {curenergy}"
                                )
                                print(phixstables[filenum][lowerlevelname])
                                sys.exit(1)
                        pointnumber += 1

                elif crosssectiontype != -1:
                    if crosssectiontype not in unknown_phixs_types:
                        unknown_phixs_types.append(crosssectiontype)
                    fitcoefficients = []
                    lowerlevelname = ""
                    numpointsexpected = 0

                if has_marker and len(row) >= 2 and " ".join(row[1:]) == "!Type of cross-section":
                    crosssectiontype = int(row[0])
                    phixs_type_levels[crosssectiontype].add(lowerlevelname)

                if len(row) == 0:
                    # for the tabulated types the array is preallocated to the declared row
                    # count, so a short table leaves trailing zeros: compare the points read
                    if (
                        crosssectiontype in {20, 21, 22}
                        and lowerlevelname
                        and lowerlevelname in phixstables[filenum]
                        and pointnumber != numpointsexpected
                    ):
                        print(
                            f"photoionization_crosssections mismatch: expecting {numpointsexpected:d} rows but found"
                            f" {pointnumber:d}"
                        )
                        print(
                            f"A={atomic_number}, ion_stage={ion_stage}, lowerlevel={lowerlevelname},"
                            f" crosssectiontype={crosssectiontype}"
                        )
                        sys.exit(1)
                    lowerlevelname = ""
                    crosssectiontype = -1
                    numpointsexpected = 0

        reduced_phixstables_onetarget = artisatomic.reduce_phixs_tables(
            phixstables[filenum], args.optimaltemperature, args.nphixspoints, args.phixsnuincrement
        )

        for lowerlevelname, reduced_phixstable in reduced_phixstables_onetarget.items():
            try:
                phixs_at_threshold = reduced_phixstable[np.nonzero(reduced_phixstable)][0]
            except IndexError:
                # The cross section is zero everywhere on the output grid, so the level gets no
                # photoionization. For type 8 (offset) this happens when the offset edge
                # nu_edge + nu_o lies beyond the grid that reduce_phixs_tables() samples.
                num_levelnames_with_zero_crosssection += 1
                artisatomic.log_and_print(
                    flog, f"WARNING: No non-zero cross section points for {lowerlevelname}, so it will have no phixs"
                )
            else:
                phixs_targetconfigfactors_of_levelname[lowerlevelname].append(
                    (
                        phixstargets[filenum],
                        phixs_at_threshold,
                    )
                )

                # Every ion with more than one photoionisation file has one file per final state of
                # the upper ion, and a level is usually present in all of them, so a second table
                # for a level is the normal multi-target case rather than an error. Every target is
                # recorded above; only one table can be written per level, so keep the one with the
                # largest threshold cross section. The normalisation below divides it by that
                # target's fraction, recovering the level's total rather than one target's share.
                if lowerlevelname in reduced_phixs_dict:
                    artisatomic.log_and_print(
                        flog,
                        f"{lowerlevelname} has a cross section table in more than one photoionisation file."
                        f" Target {phixstargets[filenum]} gives {phixs_at_threshold:.4e} Mb at threshold against"
                        f" {keptthreshold_of_levelname[lowerlevelname]:.4e} Mb for {kepttarget_of_levelname[lowerlevelname]}.",
                    )
                if phixs_at_threshold > keptthreshold_of_levelname.get(lowerlevelname, 0.0):
                    reduced_phixs_dict[lowerlevelname] = reduced_phixstable
                    kepttarget_of_levelname[lowerlevelname] = phixstargets[filenum]
                    keptthreshold_of_levelname[lowerlevelname] = phixs_at_threshold

    # summarised once for the ion, not once per photoionisation file: the counts below accumulate
    # over every file, so logging them inside that loop repeated them with partial totals
    for crosssectiontype in sorted(phixs_type_levels.keys()):
        # .get(): the branch below fires for exactly the types this parser does not handle, which
        # are the ones least likely to have a label, so indexing would fail while reporting them
        typelabel = phixs_type_labels.get(crosssectiontype, "unrecognised cross-section type")
        if crosssectiontype in unknown_phixs_types:
            artisatomic.log_and_print(
                flog,
                f"WARNING {len(phixs_type_levels[crosssectiontype])} levels with UNKNOWN cross-section type"
                f" {crosssectiontype}: {typelabel}",
            )
        else:
            artisatomic.log_and_print(
                flog,
                f"{len(phixs_type_levels[crosssectiontype])} levels with cross-section type {crosssectiontype}:"
                f" {typelabel}",
            )

    if num_levelnames_with_zero_crosssection > 0:
        artisatomic.log_and_print(
            flog,
            f"WARNING: {num_levelnames_with_zero_crosssection} level names have a cross section that is zero"
            " everywhere on the output energy grid, so those levels get no photoionization",
        )

    # normalise the target factors into fractions
    phixs_targetconfigfractions_of_levelname = defaultdict(list)
    for lowerlevelname, reduced_phixstable in reduced_phixs_dict.items():
        target_configfactors_nofilter = phixs_targetconfigfactors_of_levelname[lowerlevelname]
        # the factors are arbitrary and need to be normalised into fractions

        # filter out low fraction targets
        factor_sum_nofilter = sum(x[1] for x in target_configfactors_nofilter)

        if factor_sum_nofilter > 0.0:
            # if these are false, it's probably all zeros, so leave it and "send" it to the ground state
            target_configfactors = [x for x in target_configfactors_nofilter if (x[1] / factor_sum_nofilter > 0.01)]

            if len(target_configfactors) == 0:
                # every target was below the 1% cut, so keep them all rather than dividing by zero
                artisatomic.log_and_print(
                    flog,
                    f"WARNING: all photoionisation targets for {lowerlevelname} are below the 1% cut"
                    f" ({target_configfactors_nofilter}), so keeping them unfiltered",
                )
                target_configfactors = target_configfactors_nofilter

            factor_sum = sum(x[1] for x in target_configfactors)

            for target_config, target_factor in target_configfactors:
                target_fraction = target_factor / factor_sum
                phixs_targetconfigfractions_of_levelname[lowerlevelname].append((target_config, target_fraction))

            # The kept table is one target's cross section, but write_phixs_data() writes it as the
            # level's total and splits it over every target by the fractions above, so divide by
            # the kept target's own fraction first. Without this a level whose kept target holds
            # 50% would have both targets' rates halved. readqubdata does the same with
            # max_fraction. The kept target has the largest factor, so it is never below the 1% cut
            # and .get() only misses if its factor was zero, where there is nothing to rescale.
            kept_fraction = dict(phixs_targetconfigfractions_of_levelname[lowerlevelname]).get(
                kepttarget_of_levelname[lowerlevelname]
            )
            # not in-place: reduce_phixs_tables() hands out these arrays and they must not be
            # mutated behind the caller's back
            if kept_fraction:
                reduced_phixs_dict[lowerlevelname] = reduced_phixstable / kept_fraction

    # map the non-J-split cross sections onto J-split levels. A table matches every level sharing
    # the configuration, so index the level list by match name once rather than rescanning it.
    levelindices_of_matchname: defaultdict[str, list[int]] = defaultdict(list)
    for levelindex, levelname in enumerate(levelnames):
        levelindices_of_matchname[levelname if j_splitting_on else levelname.split("[")[0]].append(levelindex)

    for lowerlevelname_a, phixstable in reduced_phixs_dict.items():
        for levelindex in levelindices_of_matchname[lowerlevelname_a]:
            photoionization_crosssections[levelindex] = phixstable
            # .get() rather than __getitem__: a level whose target factors all came out zero is
            # absent, and an empty list would look like "has data, no targets" to  get_photoiontargetfractions()
            photoionization_targetconfig_fractions[levelindex] = phixs_targetconfigfractions_of_levelname.get(
                lowerlevelname_a
            )
            # abs() as at the phixs-fit sites above: CMFGEN writes a negative Lam(A) for some
            # levels, and that sign would read as readqubdata's "no threshold value" sentinel.
            # A zero Lam(A) gives no threshold at all rather than dividing by zero; the level
            # keeps its NaN and write_phixs_data() skips it.
            if lambdaangstroms[levelindex] != 0.0:
                photoionization_thresholds_ev[levelindex] = hc_in_ev_angstrom / abs(lambdaangstroms[levelindex])

    return photoionization_crosssections, photoionization_targetconfig_fractions, photoionization_thresholds_ev


def get_seaton_phixstable(lambda_angstrom, sigmat, beta, s, nu_o=None):
    """Evaluate a Seaton formula fit (CMFGEN cross-section type 1, or type 7 when nu_o is given).

    Returns (energy in Rydberg, cross section in Megabarns) pairs. With nu_o the edge is offset,
    so the cross section is zero until the offset threshold.
    """
    energygrid = np.arange(0, 1.0, 0.001)

    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    energy_div_threshold = 1 + 20 * (energygrid**2)

    if nu_o is None:
        threshold_div_energy = energy_div_threshold**-1
        crosssection = sigmat * (beta + (1 - beta) * threshold_div_energy) * (threshold_div_energy**s)
    else:
        # type 7
        # include Christian Vogl's python adaption of CMFGEN sub_phot_gen.f:
        # Altered 07-Oct-2015 : Bug fix for Type 7 (modified Seaton formula).
        #                       Offset was being added to the current frequency instead
        #                       of the ionization edge.

        threshold_energy_ev = hc_in_ev_angstrom / lambda_angstrom
        offset_threshold_div_energy = (energy_div_threshold**-1) * (
            1 + (nu_o * 1e15 * h_in_ev_seconds) / threshold_energy_ev
        )

        # the cross section is zero until the offset edge is reached. np.where evaluates both
        # arms, which is safe here: the ratio is positive everywhere, so the discarded arm has
        # no domain error to raise.
        crosssection = np.where(
            offset_threshold_div_energy < 1.0,
            sigmat * (beta + (1 - beta) * offset_threshold_div_energy) * offset_threshold_div_energy**s,
            0.0,
        )

    return np.column_stack([energy_div_threshold * thresholdenergyryd, crosssection])


# test: for n = 5, l_start = 4, l_end = 4 (2s2_5g_2Ge level of C II)
# 2.18 eV threshold cross section is near 4.37072813 Mb, great!
@cache
def get_hydrogenic_sigma_summed_over_l(n: int, l_start: int, l_end: int) -> np.ndarray:
    """Sum of (2l + 1) * sigma(n, l) over l_start <= l <= l_end, on the tabulated U grid.

    Depends only on the quantum numbers, not on the level, so cache it: the callers below run
    once per level per ion.
    """
    arr_sigma_summed_over_l = np.zeros(len(hyd_phixs_energygrid_ryd[n, l_start]))
    for l in range(l_start, l_end + 1):
        if not np.array_equal(hyd_phixs_energygrid_ryd[n, l], hyd_phixs_energygrid_ryd[n, l_start]):
            print("TABLE MISMATCH")
            sys.exit(1)
        arr_sigma_summed_over_l += (2 * l + 1) * hyd_phixs[n, l]

    return arr_sigma_summed_over_l


def get_hydrogenic_nl_phixstable(lambda_angstrom, n, l_start, l_end, nu_o=None, zion=None):
    """Hydrogenic split-l cross section table (CMFGEN cross-section types 2 and 8).

    With nu_o given this is type 8 ("modified hydrogenic split l"), where the cross section is
    zero below an offset edge nu_edge + nu_o and the tabulated hydrogenic cross section is read
    at U = nu / (nu_edge + nu_o) instead of nu / nu_edge. Type 8 needs the ion charge zion,
    since CMFGEN scales it by 1 / zion**2 rather than by the (n_eff / (n * zion))**2 of type 2.

    See SUB_PHOT_GEN in CMFGEN's newsubs/sub_phot_gen.f.
    """
    assert l_start >= 0
    assert l_end <= n - 1
    energygrid = hyd_phixs_energygrid_ryd[n, l_start]
    phixstable = np.empty((len(energygrid), 2))

    thresholdenergyev = hc_in_ev_angstrom / lambda_angstrom
    thresholdenergyryd = thresholdenergyev / ryd_to_ev

    # U values at which the hydrogenic cross sections are tabulated
    u_grid = energygrid / energygrid[0]

    arr_sigma_summed_over_l = get_hydrogenic_sigma_summed_over_l(n, l_start, l_end)

    l_degeneracy = (l_end - l_start + 1) * (l_end + l_start + 1)  # == sum of (2l + 1) from l_start to l_end

    if nu_o is None:
        # type 2: the output energies coincide with the tabulated U values, so no interpolation
        # 1 / thresholdenergyryd / n**2 is CMFGEN's (NEF / (n * ZION))**2
        scale_factor = 1 / thresholdenergyryd / (n**2) / l_degeneracy
        arr_sigma = arr_sigma_summed_over_l * scale_factor
    else:
        assert zion is not None, "the ion charge is required for offset (type 8) hydrogenic cross sections"
        # type 8: CMFGEN ignores the n_eff correction and uses 1 / zion**2 (see sub_phot_gen.f)
        scale_factor = 1 / (zion**2) / l_degeneracy
        e_o_ev = nu_o * 1e15 * h_in_ev_seconds
        # energy / (E_o + E_threshold), i.e. U measured from the offset edge rather than the true edge
        u_offset = thresholdenergyev * u_grid / (e_o_ev + thresholdenergyev)

        # CMFGEN interpolates log10(cross section) linearly in log10(U) on a geometric U grid,
        # reproducing its RJ = LOG10(U) / L_DEL_U indexing. u_offset <= u_grid for e_o_ev >= 0,
        # so the table end is never extrapolated past.
        with np.errstate(divide="ignore"):
            log_sigma = np.log10(arr_sigma_summed_over_l)
        arr_sigma = 10 ** np.interp(np.log10(u_offset), np.log10(u_grid), log_sigma) * scale_factor

        # the cross section is zero until the offset edge is reached
        arr_sigma[u_offset < 1.0] = 0.0

    phixstable[:, 0] = u_grid * thresholdenergyryd
    phixstable[:, 1] = arr_sigma

    return phixstable


# test: hydrogen n = 1: 13.606 eV threshold cross section is near 6.3029 Mb
# test: hydrogen n = 5: 2.72 eV threshold cross section is near 37.0 Mb?? can't find a source for this
# give the same results as get_hydrogenic_nl_phixstable(lambda_angstrom, n, 0, n - 1)
def get_hydrogenic_n_phixstable(lambda_angstrom, n):
    """Evaluate a hydrogenic cross section for a whole shell (CMFGEN type 3), all l of one n.

    Returns (energy in Rydberg, cross section in Megabarns) pairs. The Kramers scale factor
    already accounts for the effective charge, so the result must not be rescaled by the caller.
    """
    energygrid = np.asarray(hyd_gaunt_energygrid_ryd[n])

    thresholdenergyev = hc_in_ev_angstrom / lambda_angstrom
    thresholdenergyryd = thresholdenergyev / ryd_to_ev

    scale_factor = 7.91 / thresholdenergyryd / n

    energydivthreshold = energygrid / energygrid[0]

    crosssection = np.where(
        energydivthreshold > 0,
        scale_factor * np.asarray(hyd_gaunt_factor[n]) / energydivthreshold**3,
        0.0,
    )

    return np.column_stack([energydivthreshold * thresholdenergyryd, crosssection])


# Peach, Saraph, and Seaton (1988)
def get_opproject_phixstable(lambda_angstrom, a, b, c, d, e):
    """Evaluate an Opacity Project fit of Peach, Saraph and Seaton (1988) (CMFGEN type 5).

    Returns (energy in Rydberg, cross section in Megabarns) pairs.
    """
    energygrid = np.arange(0, 1.0, 0.001)

    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    energydivthreshold = 1 + 20 * (energygrid**2)
    u = energydivthreshold

    x = np.log10(np.minimum(u, e))

    crosssection = 10 ** (a + x * (b + x * (c + x * d)))
    # above the break the fit is continued with a 1/u^2 tail
    crosssection = np.where(u > e, crosssection * (e / u) ** 2, crosssection)

    return np.column_stack([energydivthreshold * thresholdenergyryd, crosssection])


# only applies to helium
# the threshold cross sections seems ok, but energy dependence could be slightly wrong
# what is the h parameter that is not used??
def get_hummer_phixstable(lambda_angstrom, a, b, c, d, e, f, g, h):  # ruff: ignore[unused-function-argument]
    """Evaluate Hummer's fit to the He I opacity cross sections (CMFGEN type 6).

    Returns (energy in Rydberg, cross section in Megabarns) pairs. A cubic in log10(E/E_th)
    below the break at e, a straight line above it.
    """
    energygrid = np.arange(0, 1.0, 0.001)

    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    energydivthreshold = 1 + 20 * (energygrid**2)

    x = np.log10(energydivthreshold)

    crosssection = np.where(x < e, 10 ** (((d * x + c) * x + b) * x + a), 10 ** (f + g * x))

    return np.column_stack([energydivthreshold * thresholdenergyryd, crosssection])


def get_vy95_phixstable(lambda_angstrom, fitcoefficients):
    """Verner & Yakovlev (1995) multi-shell ground-state fits (CMFGEN cross-section type 9).

    Each shell contributes only above its own threshold E_th, and the fit is evaluated at the
    actual photon energy. See the type-9 branch of SUB_PHOT_GEN in CMFGEN's
    newsubs/sub_phot_gen.f, where U = FREQ / CROSS_A(LMIN+3) / EV_TO_HZ is the photon energy in
    eV divided by E_0, and each shell after the first is gated on FREQ >= EV_TO_HZ * E_th_eV.
    """
    energygrid = np.arange(0, 1.0, 0.001)
    thresholdenergyev = hc_in_ev_angstrom / lambda_angstrom
    thresholdenergyryd = thresholdenergyev / ryd_to_ev

    energydivthreshold = 1 + 20 * (energygrid**2)
    energy_ev = energydivthreshold * thresholdenergyev

    crosssection = np.zeros(len(energygrid))
    for shellnum, params in enumerate(fitcoefficients):
        y = energy_ev / params.E_0
        P = params.P
        Q = 5.5 + params.l - 0.5 * params.P
        y_a = params.y_a
        y_w = params.y_w
        shellcrosssection = params.sigma_0 * ((y - 1) ** 2 + y_w**2) * (y**-Q) * ((1 + np.sqrt(y / y_a)) ** -P)
        # the first shell starts at the level's own ionization edge, later (inner) shells
        # only contribute above their own threshold
        if shellnum > 0:
            shellcrosssection = np.where(energy_ev < params.E_th_eV, 0.0, shellcrosssection)
        crosssection += shellcrosssection

    return np.column_stack([energydivthreshold * thresholdenergyryd, crosssection])


def read_coldata(atomic_number, ion_stage, dfenergy_levels: pl.DataFrame, flog, args):
    """Read one ion's CMFGEN effective collision strengths at the requested electron temperature.

    Returns a dict of upsilon values keyed by a (lower, upper) pair of zero-based level ids.
    Where the file gives one value for a whole term but the level list is J-split, the value is
    shared over the term's J levels in proportion to g_i * g_j, so that summing over the term
    recovers the file's value.
    """
    t_scale_factor = 1e4  # Hiller temperatures are given as T_4
    upsilondict: dict[tuple[int, int], float] = {}
    coldatafilename = ions_data[atomic_number, ion_stage].coldatafilename
    if not coldatafilename:
        artisatomic.log_and_print(flog, "No collisional data file specified")
        return upsilondict

    levelnames: list[str] = dfenergy_levels["levelname"].to_list()
    gvalues: list[float] = dfenergy_levels["g"].to_list()

    found_nonjsplit_transition = False
    level_ids_of_level_name = {}
    for levelid, levelname in enumerate(levelnames):
        levelnamenoJ = levelname.split("[")[0]
        if levelname != levelnamenoJ:  # levels are J split
            level_ids_of_level_name[levelname] = [levelid]
        elif not found_nonjsplit_transition:
            artisatomic.log_and_print(flog, "Found at least one transition specifying level name with no J value")
            found_nonjsplit_transition = True

        # keep track of the level ids of states that differ by J only
        # in case the collisional data level names are not J split
        try:
            level_ids_of_level_name[levelnamenoJ].append(levelid)
        except KeyError:
            level_ids_of_level_name[levelnamenoJ] = [levelid]

    # total statistical weight per term, for sharing a term-resolved collision strength over its
    # J levels. Depends only on the level list, so build it once.
    g_sum_of_level_name = {
        levelname: sum(gvalues[levelid] for levelid in levelids)
        for levelname, levelids in level_ids_of_level_name.items()
    }

    filename = (
        Path(hillier_ion_folder(atomic_number, ion_stage))
        / ions_data[atomic_number, ion_stage].folder
        / coldatafilename
    )
    artisatomic.log_and_print(flog, f"Reading {artisatomic.path_for_log(filename)}")
    coll_lines_in = 0
    number_expected_transitions = -1
    with artisatomic.xopen_check_extension(filename) as fcoldata:
        header_row: list[str] = []
        temperature_index = -1
        num_expected_t_values = -1
        for line in fcoldata:
            row = line.split()
            if len(line.strip()) == 0:
                continue  # skip blank lines

            if (
                header_row != []
                and temperature_index != -1
                and num_expected_t_values != -1
                and re.match(r"^\*{5,}+", line.strip())
            ):
                artisatomic.log_and_print(
                    flog, "WARNING: Found line of *'s after reading header, assuming that's the end of the table"
                )
                break  # Some files have lines of stars at the end, if we see one of these just exit (e.g. Na VI, Ne V)

            if line.startswith(("dln_OMEGA_dlnT = T/OMEGA* dOMEGAdt for HE2", "Johnson values")):  # found in col_ariii
                break

            if line.lstrip().startswith(r"Transition\T"):  # found the header row
                header_row = row
                if len(header_row) != num_expected_t_values + 1:
                    artisatomic.log_and_print(
                        flog,
                        f"WARNING: Expected {num_expected_t_values:d} temperature values, but header has"
                        f" {len(header_row):d} columns",
                    )

                    # Sc I and III have most of their temperatures commented out, so the number of expected temperatures is actually correct
                    # This will not catch cases with a commented header where len(header_row) == num_expected_t_values + 1 (no such cases exist as far as I am aware)
                    if "!" in header_row:
                        artisatomic.log_and_print(
                            flog,
                            f"Some temperatures are commented out, assuming header is correct, num_expected_t_values={num_expected_t_values:d}",
                        )
                    else:
                        num_expected_t_values = len(header_row) - 1
                        artisatomic.log_and_print(
                            flog,
                            f"Assuming header is incorrect and setting num_expected_t_values={num_expected_t_values:d}",
                        )

                temperatures = row[-num_expected_t_values:]
                artisatomic.log_and_print(
                    flog,
                    "Temperatures available for effective collision strengths (units of"
                    f" {t_scale_factor:.1e} K):\n{', '.join(temperatures)}",
                )
                match_sorted_temperatures = sorted(
                    temperatures,
                    key=lambda t: abs(float(t.replace("D", "E")) * t_scale_factor - args.electrontemperature),
                )
                best_temperature = match_sorted_temperatures[0]
                temperature_index = temperatures.index(best_temperature)
                artisatomic.log_and_print(
                    flog, f"Selecting {float(temperatures[temperature_index].replace('D', 'E')) * t_scale_factor:.3f} K"
                )
                continue

            if len(row) >= 2:
                row_two_to_end = " ".join(row[1:])

                if row_two_to_end.startswith("!Number of transitions"):
                    number_expected_transitions = int(row[0])
                elif row_two_to_end.startswith("!Number of T values OMEGA tabulated at"):
                    num_expected_t_values = int(row[0])
                elif (
                    row_two_to_end.startswith("!Scaling factor for OMEGA (non-file values)")
                    and float(row[0].replace("D", "E")) != 1.0
                ):
                    artisatomic.log_and_print(flog, "ERROR: non-zero scaling factor for OMEGA. what does this mean?")
                    sys.exit(1)

            if header_row != []:
                namefromnameto = "".join(row[:-num_expected_t_values])
                upsilonvalues = row[-num_expected_t_values:]

                if "-" in namefromnameto:
                    namefrom, nameto = map(str.strip, namefromnameto.split("-"))
                else:
                    # Assume there is just a space between them (as is the case in Ni XIV)
                    namefrom, nameto = row[:2]
                upsilon = float(upsilonvalues[temperature_index].replace("D", "E"))
                coll_lines_in += 1

                # any level lookup below can raise KeyError for a name not in the level list,
                # so guard the whole block rather than each lookup
                try:  # ruff: ignore[too-many-statements-in-try-clause]
                    if level_ids_of_level_name[namefrom][0] > level_ids_of_level_name[nameto][0]:
                        artisatomic.log_and_print(
                            flog,
                            f"WARNING: Swapping transition levels {namefrom} {level_ids_of_level_name[namefrom]} "
                            f"-> {nameto} {level_ids_of_level_name[nameto]}.",
                        )
                        namefrom, nameto = nameto, namefrom

                    # add forbidden collisions between states within lower and upper terms if
                    # the upper and lower levels have no J specified
                    for id_lower in level_ids_of_level_name[namefrom]:
                        for id_lower2 in level_ids_of_level_name[namefrom]:
                            if id_lower < id_lower2 and (id_lower, id_lower2) not in upsilondict:
                                upsilondict[id_lower, id_lower2] = -2.0

                    for id_upper in level_ids_of_level_name[nameto]:
                        for id_upper2 in level_ids_of_level_name[nameto]:
                            if id_upper < id_upper2 and (id_upper, id_upper2) not in upsilondict:
                                upsilondict[id_upper, id_upper2] = -2.0

                    # A term-resolved collision strength is shared over the J levels of both
                    # terms in proportion to their statistical weights:
                    #     upsilon_ij = upsilon_term * (g_i / g_lower_term) * (g_j / g_upper_term)  # ruff: ignore[commented-out-code]
                    # so that sum_ij upsilon_ij = upsilon_term, which is what makes the total
                    # term-to-term rate right (ARTIS builds the rate from upsilon_ij / g_i).
                    lower_g_sum = g_sum_of_level_name[namefrom]
                    upper_g_sum = g_sum_of_level_name[nameto]

                    for id_lower in level_ids_of_level_name[namefrom]:
                        for id_upper in level_ids_of_level_name[nameto]:
                            if id_lower == id_upper:
                                continue
                            upsilonscaled = (
                                upsilon * (gvalues[id_lower] / lower_g_sum) * (gvalues[id_upper] / upper_g_sum)
                            )
                            # upsilon is symmetric and the output wants lower id < upper id; the
                            # terms' J levels can interleave, so order the key rather than
                            # dropping the pairs that come out reversed
                            key = (min(id_lower, id_upper), max(id_lower, id_upper))
                            if key in upsilondict and upsilondict[key] >= 0.0:
                                artisatomic.log_and_print(
                                    flog,
                                    f"ERROR: Duplicate collisional transition from {namefrom} <->"
                                    f" {nameto} ({key[0]} -> {key[1]}). Keeping existing collision strength of"
                                    f" {upsilondict[key]:.2e} instead of new value of"
                                    f" {upsilonscaled:.2e}.",
                                )
                            else:
                                upsilondict[key] = upsilonscaled

                except KeyError:
                    unlisted_from_message = " (unlisted)" if namefrom not in level_ids_of_level_name else ""
                    unlisted_to_message = " (unlisted)" if nameto not in level_ids_of_level_name else ""
                    artisatomic.log_and_print(
                        flog,
                        f"Discarding upsilon={upsilon:.3f} for {namefrom}{unlisted_from_message} ->"
                        f" {nameto}{unlisted_to_message}",
                    )

    if number_expected_transitions < 0:
        artisatomic.log_and_print(flog, "WARNING: no '!Number of transitions' line found in collision data file")
    elif coll_lines_in < number_expected_transitions:
        print(
            f"ERROR: file specified {number_expected_transitions:d} transitions, but only {coll_lines_in:d} were found"
        )
        sys.exit(1)
    elif coll_lines_in > number_expected_transitions:
        artisatomic.log_and_print(
            flog,
            f"WARNING: file specified {number_expected_transitions:d} transitions, but {coll_lines_in:d} were found",
        )
    else:
        artisatomic.log_and_print(flog, f"Read {coll_lines_in} effective collision strengths")
        artisatomic.log_and_print(flog, f"Output {len(upsilondict)} effective collision strengths")

    return upsilondict


def strip_name_separators(levelname: str) -> str:
    """Remove the characters that a phot file and an oscillator file put between the parts of a name.

    A phot file does not always separate the parts of a target name the way the oscillator file of
    the upper ion does. F II names a target '2s2_2p3(2Do)' where F III has '2s2_2p3_2Do', and O IV
    names '2s2p3Po' where O V has '2s_2p_3Po'. The underscore and the parentheses carry no other
    meaning in these names. A comparison with them removed still matches the shells and the term
    exactly. Over the whole CMFGEN corpus, no two level names of one ion become the same string.
    get_photoiontargetfractions() rejects a name part that matches more than one level name.
    """
    return levelname.replace("_", "").replace("(", "").replace(")", "")


def get_photoiontargetfractions(
    dfenergy_levels,
    dfenergy_levels_upperion,
    hillier_photoion_targetconfigs: list[list[tuple[str, float]] | None] | None,
    flog=None,
) -> list[list[tuple[int, float]]]:
    """Resolve each level's photoionisation targets from configuration names to upper-ion ids.

    Returns, per zero-based level id, a list of (upper ion level id, fraction) pairs. Each upper
    ion level occurs one time only: two target configurations that come to the same level get one
    entry with the summed fraction. A target configuration that names several J-split levels of
    the upper ion shares its fraction over them in proportion to their statistical weights. For a
    name that matches no level, the comparison runs again with the separators removed. If that
    also fails, the target becomes the upper ion's ground state. A level with no cross-section
    data (None) keeps an empty target list, and write_phixs_data() skips it.

    In the second comparison, each part of the name must come to one level name of the upper ion.
    A part that matches more than one name is ambiguous, and this function raises a ValueError.
    """
    targetlist: list[list[tuple[int, float]]] = [[] for _ in range(dfenergy_levels.height)]
    targetlist_of_targetconfig: dict[str, list[tuple[int, float]]] = {}

    if hillier_photoion_targetconfigs is None:
        return targetlist

    # The comparison is per target configuration, not per level, so pull the names out one time.
    uppernamenoj_of_levelid = [levelname.split("[")[0] for levelname in dfenergy_levels_upperion["levelname"].to_list()]
    strippednames_of_levelid = [strip_name_separators(levelname) for levelname in uppernamenoj_of_levelid]

    for lowerlevelid in range(dfenergy_levels.height):
        targetconfig_fractions = hillier_photoion_targetconfigs[lowerlevelid]
        if targetconfig_fractions is None:
            continue  # photoionisation flagged as not available

        # The dict key is the upper ion level id. Two target configurations of one level can come
        # to the same upper ion level. ARTIS gives each entry of the list its own target.
        fraction_of_upperlevelid: defaultdict[int, float] = defaultdict(float)

        for targetconfig, targetconfig_fraction in targetconfig_fractions:
            if targetconfig not in targetlist_of_targetconfig:
                # sometimes the target has a slash, e.g. '3d7_4Fe/3d7_a4Fe'
                # so split on the slash and match all parts
                targetconfiglist = targetconfig.split("/")
                upperionlevelids = [
                    upperlevelid
                    for upperlevelid, upperlevelnamenoj in enumerate(uppernamenoj_of_levelid)
                    if upperlevelnamenoj in targetconfiglist
                ]
                if not upperionlevelids:
                    # The two files do not always separate the parts of a name the same way.
                    # Compare again with the separators removed before the ground state fallback.
                    targetconfiglist_stripped = [strip_name_separators(part) for part in targetconfiglist]
                    upperionlevelids = [
                        upperlevelid
                        for upperlevelid, strippedname in enumerate(strippednames_of_levelid)
                        if strippedname in targetconfiglist_stripped
                    ]
                    if upperionlevelids:
                        # The share by statistical weight is correct over the J levels of one
                        # name, and over the names of a slash list. A part that matches two names
                        # would split its fraction between two different levels. No ion of the
                        # corpus has such a part, and this check keeps that true. Not an assert:
                        # it guards written output and must survive -O.
                        for strippedpart in targetconfiglist_stripped:
                            partnames = sorted(
                                {
                                    uppernamenoj_of_levelid[levelid]
                                    for levelid in upperionlevelids
                                    if strippednames_of_levelid[levelid] == strippedpart
                                }
                            )
                            if len(partnames) > 1:
                                msg = (
                                    f"Photoionisation target part '{strippedpart}' of '{targetconfig}' matched"
                                    f" more than one level name of the upper ion with the name separators"
                                    f" removed: {partnames}. The part is ambiguous, so artisatomic cannot"
                                    " share the fraction."
                                )
                                raise ValueError(msg)
                        matchednames = sorted({uppernamenoj_of_levelid[levelid] for levelid in upperionlevelids})
                        artisatomic.log_and_print(
                            flog,
                            f"Photoionisation target '{targetconfig}' matched {matchednames} of the upper ion"
                            " with the name separators removed",
                        )
                if not upperionlevelids:
                    artisatomic.log_and_print(
                        flog,
                        f"WARNING: photoionisation target '{targetconfig}' matched no level of the upper ion,"
                        " so the upper ion's ground state is the target",
                    )
                    upperionlevelids = [0]  # the upper ion's ground state

                summed_statistical_weights = sum(
                    float(dfenergy_levels_upperion["g"][levelid]) for levelid in upperionlevelids
                )
                targetlist_of_targetconfig[targetconfig] = [
                    (upperionlevelid, dfenergy_levels_upperion["g"][upperionlevelid] / summed_statistical_weights)
                    for upperionlevelid in upperionlevelids
                ]

            for upperlevelid, statweight_fraction in targetlist_of_targetconfig[targetconfig]:
                fraction_of_upperlevelid[upperlevelid] += targetconfig_fraction * statweight_fraction

        # the upper ion's ground state where no target was matched at all
        targetlist[lowerlevelid] = list(fraction_of_upperlevelid.items()) or [(0, 1.0)]

    return targetlist


def read_hyd_phixsdata():
    """Load the hydrogenic photoionization tables that the type 2, 3 and 8 fits interpolate.

    Fills the module-level hyd_phixs / hyd_gaunt tables, so it must be called before any
    get_hydrogenic_*_phixstable(). Thresholds are taken from the H I level list, which is
    therefore required to be indexed by principal quantum number.
    """
    # the cached (2l+1)-weighted sums come from the tables filled in below, so a reload (the
    # tests call this repeatedly) must not leave sums from the previous tables
    get_hydrogenic_sigma_summed_over_l.cache_clear()

    with Path(os.devnull).open("w", encoding="utf-8") as devnull:
        (
            _hillier_ionization_energy_ev,
            dfhillier_energy_levels,
            _transitions,
            _transition_count_of_level_name,
        ) = read_levels_and_transitions(1, 1, devnull)

    # the tables below are indexed by principal quantum number, so the H I level list must not
    # have been filtered (read_levels_and_transitions drops levels with no transitions)
    assert dfhillier_energy_levels["hillierlevelid"].to_list() == list(range(1, dfhillier_energy_levels.height + 1)), (
        "H I level list is not indexed by principal quantum number, so the hydrogenic"
        " cross-section thresholds would be taken from the wrong levels"
    )
    # indexed by n - 1, i.e. the ionisation threshold wavelength of the level with principal
    # quantum number n
    lambdaangstrom_of_n = dfhillier_energy_levels["lambdaangstrom"].to_list()

    hyd_filename = hillier_ion_folder(1, 1) + "/5dec96/hyd_l_data.dat"
    print(f"Reading hydrogen photoionization cross sections from {hyd_filename}")
    max_n = -1
    l_start_u = 0.0
    l_del_u = 0.0
    with artisatomic.xopen_check_extension(hyd_filename) as fhyd:
        for line in fhyd:
            row = line.split()
            if " ".join(row[1:]) == "!Maximum principal quantum number":
                max_n = int(row[0])
                global max_hyd_l_n
                max_hyd_l_n = max_n

            if " ".join(row[1:]) == "!L_ST_U":
                l_start_u = float(row[0].replace("D", "E"))

            if " ".join(row[1:]) == "!L_DEL_U":
                l_del_u = float(row[0].replace("D", "E"))

            if max_n >= 0 and not line.strip():
                break

        for line in fhyd:
            if not line.strip():
                continue

            n, l, num_points = (int(x) for x in line.split())
            e_threshold_ev = hc_in_ev_angstrom / lambdaangstrom_of_n[n - 1]

            xs_values: list[float] = []
            for line in fhyd:
                values_thisline = [float(x) for x in line.split()]
                xs_values += values_thisline
                if len(xs_values) == num_points:
                    break
                if len(xs_values) > num_points:
                    print(
                        f"ERROR: too many datapoints for (n,l)=({n},{l}), expected {num_points} but found"
                        f" {len(xs_values)}"
                    )
                    sys.exit(1)

            hyd_phixs_energygrid_ryd[n, l] = np.array(
                [e_threshold_ev / ryd_to_ev * 10 ** (l_start_u + l_del_u * index) for index in range(num_points)]
            )
            # cross sections in Megabarns: the table holds log10(sigma) in CMFGEN's internal
            # unit of 1e-10 cm^2, and 1 Mb = 1e-18 cm^2
            hyd_phixs[n, l] = np.array([10 ** (8 + logxs) for logxs in xs_values])

    hyd_filename = hillier_ion_folder(1, 1) + "/5dec96/gbf_n_data.dat"
    print(f"Reading hydrogen Gaunt factors from {hyd_filename}")
    max_n = -1
    n_start_u = 0.0
    n_del_u = 0.0
    with artisatomic.xopen_check_extension(hyd_filename) as fhyd:
        for line in fhyd:
            row = line.split()
            if " ".join(row[1:]) == "!Maximum principal quantum number":
                max_n = int(row[0])
                global max_hyd_gaunt_n
                max_hyd_gaunt_n = max_n

            if len(row) > 1:
                if row[1] == "!N_ST_U":
                    n_start_u = float(row[0].replace("D", "E"))
                elif row[1] == "!N_DEL_U":
                    n_del_u = float(row[0].replace("D", "E"))

            if max_n >= 0 and not line.strip():
                break

        for line in fhyd:
            if not line.strip():
                continue

            n, num_points = (int(x) for x in line.split())
            e_threshold_ev = hc_in_ev_angstrom / lambdaangstrom_of_n[n - 1]

            gaunt_values: list[float] = []
            for line in fhyd:
                values_thisline = [float(x) for x in line.split()]
                gaunt_values += values_thisline
                if len(gaunt_values) == num_points:
                    break
                if len(gaunt_values) > num_points:
                    print(f"ERROR: too many datapoints for n={n}, expected {num_points} but found {len(gaunt_values)}")
                    sys.exit(1)

            hyd_gaunt_energygrid_ryd[n] = [
                e_threshold_ev / ryd_to_ev * 10 ** (n_start_u + n_del_u * index) for index in range(num_points)
            ]
            hyd_gaunt_factor[n] = gaunt_values  # cross sections in Megabarns


def extend_ion_list(
    ion_handlers: list[tuple[int, list[tuple[int, str]]]],
    maxionstage: int | None = None,
    include_hydrogen: bool | None = False,
):
    """Add every ion with CMFGEN data to ion_handlers under the "cmfgen" handler.

    Hydrogen is excluded by default: its levels are also the source of the hydrogenic
    photoionisation tables used as a fallback for other elements.
    """
    for atomic_number, ion_stage in ions_data:
        if maxionstage is not None and ion_stage > maxionstage:
            continue  # skip
        if not include_hydrogen and atomic_number == 1:
            continue  # skip
        ion_handlers = artisatomic.add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "cmfgen")

    return ion_handlers

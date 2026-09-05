"""Read levels, transitions, collision strengths and cross sections from Hillier's CMFGEN data."""

import os
import re
import typing as t
from collections import defaultdict
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from string import ascii_uppercase

import numpy as np
import numpy.typing as npt
import polars as pl

from artisatomic.base import add_handler_if_not_set
from artisatomic.base import elsymbols
from artisatomic.base import fortran_float
from artisatomic.base import h_in_ev_seconds
from artisatomic.base import hc_in_ev_angstrom
from artisatomic.base import isfloat
from artisatomic.base import log_and_print
from artisatomic.base import path_for_log
from artisatomic.base import PYDIR
from artisatomic.base import rewrite_file_as_utf8
from artisatomic.base import roman_numerals
from artisatomic.base import ryd_to_ev
from artisatomic.base import scan_file_lines
from artisatomic.base import xopen_check_extension
from artisatomic.levelnames import get_config_parity
from artisatomic.levelnames import has_merged_orbital
from artisatomic.levelnames import lchars
from artisatomic.levelnames import parse_orbital_n

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
    22: range(2, 4),  # Ti (Ti IV is left out, see ions_data below)
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
    # Ti IV: the 19apr23 files hold 126 levels and 1000 transitions, but the collision file is a
    # header only, and the 18oct00 files hold a single level. Not included.
    # (22, 4): IonFiles("19apr23", "osc_data", phot_data_names(1), "col_data"),
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
            PYDIR
            / ".."
            / "atomic-data-hillier"
            / "atomic_21jun23"
            / atomic_number_to_hillier_code[atomic_number]
            / roman_numerals[ion_stage]
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
    if has_merged_orbital(config):
        return -1

    if config[-1] == "e":
        return 0
    if config[-1] == "o":
        return 1

    configparity = get_config_parity(config)
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
    # collect once: the frame comes from scan_csv, and every collect() of the lazy frame reads
    # and decompresses the file again. The table-start search and the parse below share one read.
    dflines_eager = dflines.collect()

    # only the first table is read, and a spelling mistake in a title must not hide the second one
    tablestart = (
        dflines_eager.with_row_index()
        .filter(pl.col("line").str.contains(r"^\s*Osci(l|ll)ator strengths"))
        .select("index")
        .head(1)
    )
    if tablestart.height > 0:
        dflines_eager = dflines_eager.head(tablestart.item())
    dflines = dflines_eager.lazy()

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
            lambdaangstrom=as_float(4).fill_null(-1.0),
            i=part(5).str.strip_chars_end("-").cast(pl.Int64),
            j=part(6).cast(pl.Int64),
            # the id column is masked before the cast, not after: the other layout holds a bar
            # there, and both arms of a when() are evaluated. Only the 8-part layout carries the
            # id there: in the wide layout the column holds another number in some files (O III
            # 19apr23 has 0, 0, 1, 2, ...), so the position fills in.
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


def read_levels_and_transitions(atomic_number: int, ion_stage: int, flog) -> tuple[float, pl.DataFrame, pl.DataFrame]:
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
) -> tuple[float, pl.DataFrame, pl.DataFrame]:
    """Read one ion's levels and bound-bound transitions from its CMFGEN oscillator file.

    Returns the ionization energy in eV, a level frame, and a transition frame. Levels with no
    transitions are dropped. The level
    table's columns are read from the header the file carries above it, so the layout varies by
    ion; see hillier_rowformat_noheader for the oldest files, which have no header.

    Transitions are keyed by level name here. add_level_ids_forbidden() joins the level ids on
    afterwards, once the level frame has been given its ids.
    """
    hillier_ionization_energy_ev = 0.0

    if atomic_number == 1 and ion_stage == 2:
        # a bare proton: a single dummy level, and no oscillator file to read
        return (
            hillier_ionization_energy_ev,
            pl.DataFrame(
                [
                    HillierEnergyLevel(
                        levelname="I",
                        # one state: ARTIS divides the Saha ratio of H I to H II by this g
                        g=1.0,
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
        )

    filename = hillier_osc_filename(atomic_number, ion_stage)

    log_and_print(flog, f"Reading {path_for_log(filename)}")

    levelrows: list[HillierEnergyLevel] = []
    levels_without_parity: list[str] = []

    prev_line = ""
    # the loop below reads the header and the levels. polars reads the transitions, and starts
    # at the line after the last line that the loop read.
    linesread = 0
    expected_energy_levels = -1
    expected_transitions = -1
    row_format_energy_level = None
    format_date = "NOT_SPECIFIED"
    # threads=0 decompresses in this process. The default starts a process that writes into a
    # pipe, which reports a broken pipe when the reader stops at the end of the level table.
    with xopen_check_extension(filename, threads=0) as fhillierosc:
        for line in fhillierosc:
            linesread += 1
            row = line.split()
            if (
                re.match(r"x*\*{5,}", line) and prev_line
            ):  # The x is not a mistake, one of the lines of stars somewhere starts with an x and breaks otherwise
                if atomic_number == 26 and ion_stage == 8:  # Fe VIII has its own bespoke header...
                    print("Fe VIII has a bespoke header")
                    row_format_energy_level = "levelname g energyabovegsinpercm thresholdenergyev freqtentothe15hz lambdaangstrom hillierlevelid"
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
                log_and_print(flog, f"File specifies {expected_energy_levels:d} levels")
            elif line.rstrip().endswith("!Number of transitions"):
                expected_transitions = int(row[0])
                log_and_print(flog, f"File specifies {expected_transitions:d} transitions")
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
            if len(row) == levelcolcount and all(map(isfloat, row[1:])):
                hillierlevelid = int(row[colindex["hillierlevelid"]].lstrip("-"))
                levelname = row[colindex["levelname"]]
                energyabovegsinpercm = fortran_float(row[colindex["energyabovegsinpercm"]])
                lambdaangstrom = fortran_float(row[colindex["lambdaangstrom"]])
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
                    log_and_print(flog, f"Can't find LS term in Hillier level name '{levelname}'")

                # the ground state gives the ionization energy. The first level below 1 cm^-1 is
                # the ground state: a second one is a J level of the same split term. CMFGEN writes
                # a negative Lam(A) for some levels, hence the abs().
                if energyabovegsinpercm < 1.0 and hillier_ionization_energy_ev == 0.0:
                    if lambdaangstrom == 0.0:
                        msg = f"Level '{levelname}' has Lam(A) = 0, so the ionization energy cannot be read from it"
                        raise ValueError(msg)
                    hillier_ionization_energy_ev = hc_in_ev_angstrom / abs(lambdaangstrom)

                if hillierlevelid != len(levelrows):
                    msg = f"Hillier levels mismatch: id {hillierlevelid:d} found at entry number {len(levelrows):d}"
                    raise ValueError(msg)

            if re.match(r"^\s*Osci(l|ll)ator strengths", line) and len(levelrows) > 0:
                break

    log_and_print(flog, f"Read {len(levelrows):d} levels")
    if levels_without_parity:
        # Normal for ions with merged levels, so this is a count and a sample rather than a
        # warning per level: H I and He II are merged all the way down.
        log_and_print(
            flog,
            f"{len(levels_without_parity):d} of {len(levelrows):d} levels have no definite parity"
            f" (every transition touching one is treated as permitted), e.g."
            f" {', '.join(levels_without_parity[:5])}",
        )
    if len(levelrows) != expected_energy_levels:
        msg = f"{filename} declares {expected_energy_levels} levels but {len(levelrows)} were read"
        raise ValueError(msg)

    # not an assert: this guards the ionization energy that adata.txt gets. H II returns above
    # with 0.0, because a bare nucleus has no level to read one from.
    if hillier_ionization_energy_ev == 0.0:
        msg = f"{filename} has no level below 1 cm^-1, so the ionization energy could not be read"
        raise ValueError(msg)

    # the rest of the file holds the transitions, which polars reads rather than Python: one ion
    # carries half a million of them
    dftransitions = parse_transition_lines(scan_file_lines(filename, skip_lines=linesread), filename)

    log_and_print(flog, f"Read {dftransitions.height:d} transitions")
    if dftransitions.height != expected_transitions:
        msg = f"{filename} declares {expected_transitions} transitions but {dftransitions.height} were read"
        raise ValueError(msg)

    # filter out levels with no transitions
    names_with_transitions = pl.concat([dftransitions["namefrom"], dftransitions["nameto"]]).unique()
    dfhillier_energy_levels = pl.DataFrame(levelrows, schema=hillier_level_schema, orient="row").filter(
        pl.col("levelname").is_in(names_with_transitions)
    )

    return hillier_ionization_energy_ev, dfhillier_energy_levels, dftransitions


# the energy grid of the analytic fits (types 1, 5, 6, 7 and 9) as a multiple of the threshold
# energy: 1000 points from the threshold to 21 times the threshold, denser near the threshold
fit_energy_div_threshold = 1 + 20 * (np.arange(0, 1.0, 0.001) ** 2)

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


class PhotFileReader:
    """Read the photoionization files of one CMFGEN ion, one file at a time.

    A file is a header, then one block for each level. A block starts with three marker
    lines: "!Configuration name", "!Type of cross-section" and "!Number of cross-section
    points". The data rows follow, and a blank line ends the block. The rows of a fit type
    hold one coefficient each. The rows of a tabulated type (20, 21, 22) hold an energy in
    units of the threshold and a cross section in Megabarns.

    One polars pass cuts every line into its tokens and parses the first two as floats. 94% of
    a phot file is two-column data. A Python loop over those lines was the largest cost of a
    cmfgen build after the downsampling. The lines with a "!..." marker and the blank lines
    are the events. read_file() walks those in Python and takes the data rows of each block as
    a slice of the parsed columns.

    The ion-wide state stays on the instance across the files. That state is the tables and
    the target of each file, the J-splitting mode, and the level names of each type.
    """

    def __init__(
        self,
        atomic_number: int,
        ion_stage: int,
        photfilecount: int,
        lambdaangstroms: list[float],
        firstlevelindex_of_levelname: dict[str, int],
        firstlevelindex_of_levelnamenoJ: dict[str, int],
        flog,
    ) -> None:
        """Set up the reader for one ion. The level lookups come from read_phixs_tables()."""
        self.atomic_number = atomic_number
        self.ion_stage = ion_stage
        # charge of the ion left behind, i.e. CMFGEN's ZION (= ZXzV from the oscillator file,
        # which equals the ionisation stage for every ion here)
        self.zion = ion_stage
        self.lambdaangstroms = lambdaangstroms
        self.firstlevelindex_of_levelname = firstlevelindex_of_levelname
        self.firstlevelindex_of_levelnamenoJ = firstlevelindex_of_levelnamenoJ
        self.flog = flog

        self.phixstables: list[dict[str, np.ndarray]] = [{} for _ in range(photfilecount)]
        self.phixstargets: list[str] = ["" for _ in range(photfilecount)]
        self.j_splitting_on = False
        # the level matching after the files uses one mode for the whole ion, so every phot
        # file of the ion must declare the same J-splitting mode. None means no file declared
        # one yet.
        self.j_splitting_seen: bool | None = None
        # sets, not lists: only the distinct level count per type is reported, and a list
        # needed a linear scan per record to dedupe, which is quadratic in the level count
        self.phixs_type_levels: defaultdict[int, set[str]] = defaultdict(set)
        self.unknown_phixs_types: list[int] = []

        # the state of the file and the block being read, reset by read_file()
        self.filenum = 0
        self.photfilename = ""
        self.lowerlevelindex = -1
        self.lowerlevelname = ""
        self.targetlevelname = ""
        self.numpointsexpected = 0
        self.crosssectiontype = -1
        self.fitcoefficients: list[t.Any] = []
        # the points of the current tabulated block, in file order
        self.pending_energyryd: list[np.ndarray] = []
        self.pending_sigma: list[np.ndarray] = []
        # the name and the declared size of the block those points belong to. A file can start
        # the next block with no blank line. The state then names the next level already.
        self.pending_levelname = ""
        self.pending_numpoints = 0
        self.thresholdenergyryd = 0.0
        # Used to skip problematic lines in Fe VIII and Ni X phot_data_A (see read_file)
        self.in_header = False
        self.lines = pl.Series("line", [], dtype=pl.String)
        self.ncols = np.empty(0, dtype=np.int64)
        self.f0 = np.empty(0)
        self.f1 = np.empty(0)

    def read_file(self, filenum: int, filename: Path, photfilename: str) -> None:
        """Read one phot file into self.phixstables[filenum] and self.phixstargets[filenum]."""
        self.filenum = filenum
        self.photfilename = photfilename
        self.lowerlevelindex = -1
        self.lowerlevelname = ""
        self.targetlevelname = ""
        self.numpointsexpected = 0
        self.crosssectiontype = -1
        self.fitcoefficients = []
        self.pending_energyryd = []
        self.pending_sigma = []
        self.pending_levelname = ""
        self.pending_numpoints = 0
        self.thresholdenergyryd = 0.0
        self.in_header = False

        self.lines = scan_file_lines(filename).collect()["line"].fill_null("")
        is_event = self.lines.str.contains("!", literal=True) | (self.lines.str.strip_chars().str.len_chars() == 0)
        event_rows = np.flatnonzero(is_event.to_numpy())
        event_lines: list[str] = self.lines.filter(is_event).to_list()
        dftokens = (
            self.lines.to_frame()
            .select(parts=pl.col("line").str.extract_all(r"\S+"))
            .select(
                pl.col("parts").list.len().alias("ncols"),
                # Fortran writes a D exponent. A token that is not a float gives NaN.
                *(
                    pl.col("parts")
                    .list.get(column, null_on_oob=True)
                    .str.replace("D", "E", literal=True)
                    .cast(pl.Float64, strict=False)
                    .alias(f"f{column}")
                    for column in (0, 1)
                ),
            )
        )
        self.ncols = dftokens["ncols"].to_numpy()
        self.f0 = dftokens["f0"].to_numpy()
        self.f1 = dftokens["f1"].to_numpy()

        segment_start = 0
        for event_row, line in zip(event_rows, event_lines, strict=True):
            # the data rows between the previous event and this one belong to the current block
            self.take_data_rows(segment_start, int(event_row))
            segment_start = int(event_row) + 1
            self.take_event_line(line)

        self.take_data_rows(segment_start, len(self.lines))
        self.finish_tabulated_block(validate=False)

        # a file with no "!Cross-section unit" line never leaves the header state, so every
        # level block was skipped with a warning and the file gave no cross sections
        if not self.in_header:
            msg = f"{photfilename} has no '!Cross-section unit' line, so none of its cross sections were read"
            raise ValueError(msg)

    def take_event_line(self, line: str) -> None:
        """Apply one blank line or one line with a "!..." marker to the state."""
        row = line.split()
        if not row:
            # a blank line ends the block, and the points read must match the declared count
            self.finish_tabulated_block(validate=True)
            self.lowerlevelname = ""
            self.crosssectiontype = -1
            self.numpointsexpected = 0
            return

        if len(row) >= 2 and " ".join(row[-4:]) == "!Final state in ion":
            # this is not used because the upper ion's levels are not known at this time
            self.targetlevelname = row[0]
            log_and_print(self.flog, "Photoionisation target: " + self.targetlevelname)
            if "[" in self.targetlevelname:
                msg = f"target level {self.targetlevelname} contains a bracket (is J-split?)"
                raise ValueError(msg)
            if self.targetlevelname in self.phixstargets:
                msg = f"Multiple phixs files for the same target configuration {self.targetlevelname}"
                raise ValueError(msg)
            self.phixstargets[self.filenum] = self.targetlevelname

        if len(row) >= 2 and " ".join(row[-3:]) == "!Split J levels":
            if row[0].lower() in {"true", "false"}:
                new_j_splitting_on = row[0].lower() == "true"
                if self.j_splitting_seen is not None and new_j_splitting_on != self.j_splitting_seen:
                    msg = "The ion's phot files disagree about J-splitting"
                    raise ValueError(msg)
                self.j_splitting_seen = new_j_splitting_on
                self.j_splitting_on = new_j_splitting_on
                if self.j_splitting_on:
                    log_and_print(self.flog, "File specifies J-splitting enabled")
            else:
                msg = f'J-splitting not true or false: "{row[0]}"'
                raise ValueError(msg)

        if (len(row) >= 2 and " ".join(row[-2:]) == "!Configuration name") or " ".join(
            row[-3:]
        ) == "!Configuration name [*]":
            if not self.in_header:
                log_and_print(
                    self.flog, f"WARNING: no photoionisation target ({line.strip()}), skipping to the next line"
                )
                # Fe VIII and Ni X phot_data_A have lines before the header that end in
                # "!Configuration name" and are not level blocks
                return

            self.lowerlevelname = row[0]
            # with J splitting the name (including any [J] suffix) maps to exactly one
            # level; without it, strip the suffix so the table covers the configuration
            if not self.j_splitting_on and "[" in self.lowerlevelname:
                self.lowerlevelname = self.lowerlevelname.split("[")[0]
            self.fitcoefficients = []
            self.numpointsexpected = 0
            # take the first matching level (without J splitting, several can differ by
            # J). A name with no matching level falls back to index 0, so the fit uses
            # the ground state's threshold wavelength. Nothing uses that table: the
            # levelindices_of_matchname mapping in read_phixs_tables() has the same key,
            # finds no level for the name, and drops the table. The phot files routinely
            # cover levels that the oscillator file does not (1145 of them for Co II), so
            # this is a silent fallback rather than an error.
            self.lowerlevelindex = (
                self.firstlevelindex_of_levelname if self.j_splitting_on else self.firstlevelindex_of_levelnamenoJ
            ).get(self.lowerlevelname, 0)
            if not self.targetlevelname:
                msg = f"{self.photfilename} names a level before its '!Final state in ion' line"
                raise ValueError(msg)

        if len(row) >= 2 and " ".join(row[-3:]) == "!Screened nuclear charge":
            # CMFGEN's ZION comes from the oscillator file: RDPHOT_GEN_V2 never reads
            # this field, and the two disagree for 29 shipped files. Keep ion_stage (which
            # matches the oscillator value for every ion in ions_data) and just report it.
            zion_from_photfile = int(fortran_float(row[0]))
            if zion_from_photfile != self.ion_stage:
                log_and_print(
                    self.flog,
                    f"WARNING: ignoring screened nuclear charge {zion_from_photfile} in {self.photfilename},"
                    f" which disagrees with ion_stage {self.ion_stage}",
                )

        if len(row) >= 2 and " ".join(row[1:]) == "!Number of cross-section points":
            # a new block of points starts, so a block that no blank line ended is stored as is
            self.finish_tabulated_block(validate=False)
            self.numpointsexpected = int(row[0])

        if len(row) >= 2 and " ".join(row[1:]) == "!Cross-section unit":
            self.in_header = True  # All phot_data_* in 19apr23 have this line
            if row[0] != "Megabarns":
                msg = f"Wrong cross-section unit: {row[0]}"
                raise ValueError(msg)

        if len(row) >= 2 and " ".join(row[1:]) == "!Type of cross-section":
            self.crosssectiontype = int(row[0])
            self.phixs_type_levels[self.crosssectiontype].add(self.lowerlevelname)
            # dropped here, once, and not on every marker line: a check that ran on the next
            # block's "!Configuration name" line cleared that block's name when no blank line
            # separated the two blocks
            if not self.type_is_known():
                self.note_unknown_type()

    def type_is_known(self) -> bool:
        """Whether the current cross-section type is one that this reader evaluates."""
        return self.crosssectiontype in phixs_fit_functions or self.crosssectiontype in {0, 2, 3, 8, 9, 20, 21, 22}

    def note_unknown_type(self) -> None:
        """Record a cross-section type that this reader does not evaluate, and drop its block."""
        if self.crosssectiontype not in self.unknown_phixs_types:
            self.unknown_phixs_types.append(self.crosssectiontype)
        self.fitcoefficients = []
        self.lowerlevelname = ""
        self.numpointsexpected = 0

    def finish_tabulated_block(self, validate: bool) -> None:
        """Store the points of the tabulated block that ends here.

        A blank line ends a block, and then the point count must match the declared one.
        Without a blank line a short block is stored as read, with a warning.
        """
        if not self.pending_energyryd:
            return
        energyryd = np.concatenate(self.pending_energyryd)
        sigma = np.concatenate(self.pending_sigma)
        self.pending_energyryd = []
        self.pending_sigma = []
        if len(energyryd) != self.pending_numpoints and (validate or len(energyryd) > self.pending_numpoints):
            msg = (
                f"Z={self.atomic_number}, ion_stage={self.ion_stage}, lowerlevel={self.pending_levelname},"
                f" crosssectiontype={self.crosssectiontype}: expecting {self.pending_numpoints:d}"
                f" cross-section rows but found {len(energyryd):d}"
            )
            raise ValueError(msg)
        if len(energyryd) < self.pending_numpoints:
            log_and_print(
                self.flog,
                f"WARNING: {self.pending_levelname} declares {self.pending_numpoints:d} cross-section rows but"
                f" the block ends after {len(energyryd):d}",
            )
        # the rows the file gave, and no zero rows up to the declared count: the downsampling
        # takes the energy column as sorted, and a zero energy at the end of the table is not
        self.phixstables[self.filenum][self.pending_levelname] = np.column_stack((energyryd, sigma))

    def take_data_rows(self, start: int, end: int) -> None:
        """Read the data rows start to end - 1 into the current block."""
        if self.crosssectiontype == -1 or start >= end:
            return
        seg_ncols = self.ncols[start:end]
        seg_f0 = self.f0[start:end]

        if self.crosssectiontype in {20, 21, 22}:
            self.take_tabulated_rows(seg_ncols, seg_f0, self.f1[start:end])
        elif self.crosssectiontype == 9:
            if self.numpointsexpected > 0:
                # tolist(): a list of Python ints, which is what the Series index takes
                for rowindex in (np.flatnonzero(seg_ncols == 8) + start).tolist():
                    self.take_vy95_row(self.lines[rowindex].split())
        elif not self.type_is_known():
            self.note_unknown_type()
        elif self.numpointsexpected > 0:
            # the fit types take one float on each row
            for value in seg_f0[(seg_ncols == 1) & ~np.isnan(seg_f0)]:
                self.take_fit_coefficient(float(value))

    def take_tabulated_rows(self, seg_ncols: np.ndarray, seg_f0: np.ndarray, seg_f1: np.ndarray) -> None:
        """Add the two-column rows of a tabulated block (types 20, 21, 22) to the pending points."""
        if not self.lowerlevelname:
            return
        ispoint = (seg_ncols == 2) & ~np.isnan(seg_f0) & ~np.isnan(seg_f1)
        if not ispoint.any():
            return
        x = seg_f0[ispoint]
        if not self.pending_energyryd:
            self.pending_levelname = self.lowerlevelname
            self.pending_numpoints = self.numpointsexpected
            lambda_angstrom = abs(self.lambdaangstroms[self.lowerlevelindex])
            self.thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev
            # for these types the x value is a fraction of the threshold, not an energy
            if abs(x[0] - 1.0) > 0.5:
                print(
                    f"{self.lowerlevelname} cross section type:{self.crosssectiontype}, {x[0]:.3f} is not near"
                    f" one? might be energy instead? E_threshold = {self.thresholdenergyryd:.3f} Ry"
                )
        energyryd = x * self.thresholdenergyryd
        # the order test includes the last point of the block read so far
        allenergy = (
            energyryd if not self.pending_energyryd else np.concatenate(([self.pending_energyryd[-1][-1]], energyryd))
        )
        steps = np.diff(allenergy)
        decreasing = np.flatnonzero(steps < 0)
        if len(decreasing) > 0:
            msg = (
                f"photoionization table for {self.lowerlevelname} first column decreases "
                f"with energy {allenergy[decreasing[0]]} followed by {allenergy[decreasing[0] + 1]}"
            )
            raise ValueError(msg)
        for index in np.flatnonzero(steps == 0):
            print(
                f"WARNING: photoionization table for {self.lowerlevelname} first column duplicated "
                f"energy value of {allenergy[index]}"
            )
        self.pending_energyryd.append(energyryd)
        self.pending_sigma.append(seg_f1[ispoint])

    def take_vy95_row(self, row: list[str]) -> None:
        """Add one eight-column row of a type 9 (Verner & Yakovlev 1995) block."""
        self.fitcoefficients.append(VY95PhixsFitRow(int(row[0]), int(row[1]), *[fortran_float(x) for x in row[2:]]))
        if len(self.fitcoefficients) * 8 == self.numpointsexpected:
            lambda_angstrom = abs(self.lambdaangstroms[self.lowerlevelindex])
            self.store_table(get_vy95_phixstable(lambda_angstrom, self.fitcoefficients))

    def store_table(self, table: np.ndarray) -> None:
        """Keep the evaluated table of the current level, and close the block to more coefficients."""
        self.phixstables[self.filenum][self.lowerlevelname] = table
        self.numpointsexpected = len(table)

    def take_fit_coefficient(self, value: float) -> None:
        """Add one coefficient of a fit type, and evaluate the fit when the type's count is reached."""
        crosssectiontype = self.crosssectiontype
        fitcoefficients = self.fitcoefficients
        flog = self.flog

        if crosssectiontype == 0:
            fitcoefficients.append(value)
            if value != 0.0:
                msg = f"Cross section type 0 of {self.lowerlevelname} has a non-zero number after it"
                raise ValueError(msg)
            return

        if crosssectiontype in phixs_fit_functions:
            # types 1, 5, 6 and 7 share one shape: single-float rows fill fitcoefficients
            # until the type's count is reached, and one call then builds the table
            fitcoefficients.append(value)
            ncoefficients, fitfunc = phixs_fit_functions[crosssectiontype]
            if len(fitcoefficients) == ncoefficients:
                lambda_angstrom = abs(self.lambdaangstroms[self.lowerlevelindex])
                self.store_table(fitfunc(lambda_angstrom, *fitcoefficients))
            return

        if crosssectiontype == 2:
            fitcoefficients.append(int(value))
            if len(fitcoefficients) == 3:
                n, l_start, l_end = fitcoefficients
                if n > max_hyd_l_n:
                    log_and_print(flog, f"WARNING: n ({n}) > max_hyd_l_n ({max_hyd_l_n}), skipping table")
                elif l_end > n - 1:
                    log_and_print(flog, f"ERROR: can't have l_end = {l_end} > n - 1 = {n - 1}")
                else:
                    lambda_angstrom = abs(self.lambdaangstroms[self.lowerlevelindex])
                    self.store_table(get_hydrogenic_nl_phixstable(lambda_angstrom, n, l_start, l_end))
            return

        if crosssectiontype == 3:
            fitcoefficients.append(value)
            if len(fitcoefficients) == 2:
                scale, n = fitcoefficients
                if n > max_hyd_gaunt_n:
                    log_and_print(flog, f"WARNING: n ({n}) > max_hyd_gaunt_n ({max_hyd_gaunt_n}), skipping table")
                    return
                lambda_angstrom = abs(self.lambdaangstroms[self.lowerlevelindex])
                # scale the cross sections but not the energy grid
                phixstable = get_hydrogenic_n_phixstable(lambda_angstrom, int(n))
                phixstable[:, 1] *= scale
                self.store_table(phixstable)
            return

        if crosssectiontype == 8:
            # the first three parameters are integers, the fourth is a float
            fitcoefficients.append(int(value) if len(fitcoefficients) <= 2 else value)
            if len(fitcoefficients) == 4:
                n, l_start, l_end, nu_o = fitcoefficients
                if n > max_hyd_l_n:
                    log_and_print(flog, f"WARNING: n ({n}) > max_hyd_l_n ({max_hyd_l_n}), skipping table")
                elif l_end > n - 1:
                    log_and_print(flog, f"ERROR: can't have l_end = {l_end} > n - 1 = {n - 1}")
                else:
                    lambda_angstrom = abs(self.lambdaangstroms[self.lowerlevelindex])
                    self.store_table(
                        get_hydrogenic_nl_phixstable(lambda_angstrom, n, l_start, l_end, nu_o=nu_o, zion=self.zion)
                    )


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
    # phixs imports this module for get_hydrogenic_n_phixstable(), so a module-level import
    # of reduce_phixs_tables here would be circular
    from artisatomic.phixs import reduce_phixs_tables

    # pulled out of the frame once: the loops below index these per level, per cross-section table
    levelcount = dfenergy_levels.height

    photfilenames = ions_data[atomic_number, ion_stage].photfilenames
    if not photfilenames:
        # empty arrays, not zero-filled ones: read_ion_data() reads an empty cross-section array
        # as "no data" and applies the hydrogenic estimate. A zero-filled array passed as data.
        log_and_print(flog, "No photoionisation files for this ion")
        return np.empty((0, args.nphixspoints)), [None] * levelcount, np.empty(0)

    # the type 2, 3 and 8 fits interpolate the hydrogenic tables
    read_hyd_phixsdata()

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

    reader = PhotFileReader(
        atomic_number,
        ion_stage,
        len(photfilenames),
        lambdaangstroms,
        firstlevelindex_of_levelname,
        firstlevelindex_of_levelnamenoJ,
        flog,
    )
    reduced_phixs_dict = {}
    # the target whose table is the one kept in reduced_phixs_dict, so the normalisation below can
    # divide by that target's fraction and recover the level's total cross section, plus that
    # table's threshold cross section, which is what decides between competing targets
    kepttarget_of_levelname: dict[str, str] = {}
    keptthreshold_of_levelname: dict[str, float] = {}
    phixs_targetconfigfactors_of_levelname = defaultdict(list)
    num_levelnames_with_zero_crosssection = 0

    for filenum, photfilename in enumerate(photfilenames):
        filename = Path(
            hillier_ion_folder(atomic_number, ion_stage), ions_data[atomic_number, ion_stage].folder, photfilename
        )

        log_and_print(flog, f"Reading {path_for_log(filename)}")
        reader.read_file(filenum, filename, photfilename)

        reduced_phixstables_onetarget = reduce_phixs_tables(
            reader.phixstables[filenum], args.optimaltemperature, args.nphixspoints, args.phixsnuincrement
        )

        for lowerlevelname, reduced_phixstable in reduced_phixstables_onetarget.items():
            # The first non-zero point of the grid, not index 0. A table can be zero at the
            # nominal threshold: a type 8 offset fit, or a tabulated type whose data starts
            # above nu_edge. Such a table has its own edge further up the grid, and the reader
            # takes its cross section there. Two targets of one level can thus meet at
            # different photon energies (N II 2s_2p2(4Pe)3s_5Pe: nu/nu_edge = 2.47 against
            # 1.0). That is the accepted choice: each target's branching factor is its cross
            # section at its own edge.
            try:
                phixs_at_threshold = reduced_phixstable[np.nonzero(reduced_phixstable)][0]
            except IndexError:
                # The cross section is zero everywhere on the output grid, so the level gets no
                # photoionization. For type 8 (offset) this happens when the offset edge
                # nu_edge + nu_o lies beyond the grid that reduce_phixs_tables() samples.
                num_levelnames_with_zero_crosssection += 1
                log_and_print(
                    flog, f"WARNING: No non-zero cross section points for {lowerlevelname}, so it will have no phixs"
                )
            else:
                phixs_targetconfigfactors_of_levelname[lowerlevelname].append(
                    (
                        reader.phixstargets[filenum],
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
                    log_and_print(
                        flog,
                        f"{lowerlevelname} has a cross section table in more than one photoionisation file."
                        f" Target {reader.phixstargets[filenum]} gives {phixs_at_threshold:.4e} Mb at threshold against"
                        f" {keptthreshold_of_levelname[lowerlevelname]:.4e} Mb for {kepttarget_of_levelname[lowerlevelname]}.",
                    )
                if phixs_at_threshold > keptthreshold_of_levelname.get(lowerlevelname, 0.0):
                    reduced_phixs_dict[lowerlevelname] = reduced_phixstable
                    kepttarget_of_levelname[lowerlevelname] = reader.phixstargets[filenum]
                    keptthreshold_of_levelname[lowerlevelname] = phixs_at_threshold

    # summarised once for the ion, not once per photoionisation file: the counts below accumulate
    # over every file, so logging them inside that loop repeated them with partial totals
    for crosssectiontype in sorted(reader.phixs_type_levels.keys()):
        # .get(): the branch below fires for exactly the types this parser does not handle, which
        # are the ones least likely to have a label, so indexing would fail while reporting them
        typelabel = phixs_type_labels.get(crosssectiontype, "unrecognised cross-section type")
        if crosssectiontype in reader.unknown_phixs_types:
            log_and_print(
                flog,
                f"WARNING {len(reader.phixs_type_levels[crosssectiontype])} levels with UNKNOWN cross-section type"
                f" {crosssectiontype}: {typelabel}",
            )
        else:
            log_and_print(
                flog,
                f"{len(reader.phixs_type_levels[crosssectiontype])} levels with cross-section type {crosssectiontype}:"
                f" {typelabel}",
            )

    if num_levelnames_with_zero_crosssection > 0:
        log_and_print(
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
                log_and_print(
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
        levelindices_of_matchname[levelname if reader.j_splitting_on else levelname.split("[")[0]].append(levelindex)

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
    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    energy_div_threshold = fit_energy_div_threshold

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
            msg = f"The hydrogenic energy grids of (n, l) = ({n}, {l}) and ({n}, {l_start}) differ"
            raise ValueError(msg)
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
    read_hyd_phixsdata()
    if n < 1 or n > max_hyd_gaunt_n:
        # a bare KeyError on the module dict would name neither the table nor its range
        msg = f"The hydrogenic tables cover n = 1 to {max_hyd_gaunt_n}, not n = {n}"
        raise ValueError(msg)
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
    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    energydivthreshold = fit_energy_div_threshold
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
    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    energydivthreshold = fit_energy_div_threshold

    x = np.log10(energydivthreshold)

    crosssection = np.where(x < e, 10 ** (((d * x + c) * x + b) * x + a), 10 ** (f + g * x))

    return np.column_stack([energydivthreshold * thresholdenergyryd, crosssection])


# the cross-section types whose data rows are single floats collected into fitcoefficients:
# {crosssectiontype: (coefficient count, fit function)}. read_phixs_tables() dispatches on this.
phixs_fit_functions = {
    1: (3, get_seaton_phixstable),
    5: (5, get_opproject_phixstable),
    6: (8, get_hummer_phixstable),
    7: (4, get_seaton_phixstable),
}


def get_vy95_phixstable(lambda_angstrom, fitcoefficients):
    """Verner & Yakovlev (1995) multi-shell ground-state fits (CMFGEN cross-section type 9).

    Each shell contributes only above its own threshold E_th, and the fit is evaluated at the
    actual photon energy. See the type-9 branch of SUB_PHOT_GEN in CMFGEN's
    newsubs/sub_phot_gen.f, where U = FREQ / CROSS_A(LMIN+3) / EV_TO_HZ is the photon energy in
    eV divided by E_0, and each shell after the first is gated on FREQ >= EV_TO_HZ * E_th_eV.
    """
    thresholdenergyev = hc_in_ev_angstrom / lambda_angstrom
    thresholdenergyryd = thresholdenergyev / ryd_to_ev

    energydivthreshold = fit_energy_div_threshold
    energy_ev = energydivthreshold * thresholdenergyev

    crosssection = np.zeros(len(energydivthreshold))
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


def get_level_valence_n(levelname: str) -> int | None:
    """Principal quantum number of the valence electron, read from a CMFGEN level name.

    The last orbital of the configuration is the valence one: '2s2_2p3(4So)3p_5Pe[1]' gives 3,
    '3d5(4D)4po[3]' gives 4, and a merged shell such as '2s2_18w_2W' gives 18. Returns None for a
    name with no readable orbital ('1___', '8SNG'). The caller, match_hydrogenic_phixs(), then
    gives the level no estimate and writes a warning to the ion log.

    An orbital is digits and a lower-case orbital letter. A term is a digit, an upper-case letter
    and an optional e or o, so it never matches, and a seniority letter ('_a6De') follows no digit.
    """
    orbitals = re.findall(r"\d+[a-z]", levelname.split("[", maxsplit=1)[0])
    return parse_orbital_n(orbitals[-1]) if orbitals else None


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
        log_and_print(flog, "No collisional data file specified")
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
            log_and_print(flog, "Found at least one transition specifying level name with no J value")
            found_nonjsplit_transition = True

        # keep track of the level ids of states that differ by J only
        # in case the collisional data level names are not J split
        level_ids_of_level_name.setdefault(levelnamenoJ, []).append(levelid)

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
    log_and_print(flog, f"Reading {path_for_log(filename)}")
    coll_lines_in = 0
    number_expected_transitions = -1
    # the within-term pair loops below insert all of a name's pairs at its first mention, so
    # later mentions of the name can skip both loops
    names_expanded: set[str] = set()
    with xopen_check_extension(filename) as fcoldata:
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
                log_and_print(
                    flog, "WARNING: Found line of *'s after reading header, assuming that's the end of the table"
                )
                break  # Some files have lines of stars at the end, if we see one of these just exit (e.g. Na VI, Ne V)

            if line.startswith(("dln_OMEGA_dlnT = T/OMEGA* dOMEGAdt for HE2", "Johnson values")):  # found in col_ariii
                break

            if line.lstrip().startswith(r"Transition\T"):  # found the header row
                header_row = row
                if len(header_row) != num_expected_t_values + 1:
                    log_and_print(
                        flog,
                        f"WARNING: Expected {num_expected_t_values:d} temperature values, but header has"
                        f" {len(header_row):d} columns",
                    )

                    # Sc I and III have most of their temperatures commented out, so the
                    # number of expected temperatures is correct there. This test does not
                    # catch a commented header with len(header_row) == num_expected_t_values + 1.
                    # No known file has one.
                    if "!" in header_row:
                        log_and_print(
                            flog,
                            f"Some temperatures are commented out, assuming header is correct, num_expected_t_values={num_expected_t_values:d}",
                        )
                    else:
                        num_expected_t_values = len(header_row) - 1
                        log_and_print(
                            flog,
                            f"Assuming header is incorrect and setting num_expected_t_values={num_expected_t_values:d}",
                        )

                # a header can comment out part of its temperature list ('0.2 100.0 ! 0.5 ...'),
                # and the slice below counts from the end of the row, so it read the commented
                # labels as the real ones. Cut the row at the comment first. header_row keeps the
                # full row: the commented-out test above needs the '!'.
                # cut at the first comment token, whether "!" stands alone or is attached ("!0.5")
                row = row[: next((i for i, token in enumerate(row) if token.startswith("!")), len(row))]
                temperatures = row[-num_expected_t_values:]
                log_and_print(
                    flog,
                    "Temperatures available for effective collision strengths (units of"
                    f" {t_scale_factor:.1e} K):\n{', '.join(temperatures)}",
                )
                best_temperature = min(
                    temperatures,
                    key=lambda t: abs(fortran_float(t) * t_scale_factor - args.electrontemperature),
                )
                temperature_index = temperatures.index(best_temperature)
                log_and_print(
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
                    and fortran_float(row[0]) != 1.0
                ):
                    msg = f"scaling factor for OMEGA is {row[0]}, not 1. The reader does not apply a scaling factor."
                    raise ValueError(msg)

            if header_row != []:
                namefromnameto = "".join(row[:-num_expected_t_values])
                upsilonvalues = row[-num_expected_t_values:]

                if "-" in namefromnameto:
                    namefrom, nameto = map(str.strip, namefromnameto.split("-"))
                else:
                    # Assume there is just a space between them (as is the case in Ni XIV)
                    namefrom, nameto = row[:2]
                upsilon = fortran_float(upsilonvalues[temperature_index])
                coll_lines_in += 1

                # the collision file can name a level that the oscillator file does not have.
                # A membership test, not a try/except KeyError around the whole block: that
                # reported any KeyError below as an unlisted level.
                unlisted = [name for name in (namefrom, nameto) if name not in level_ids_of_level_name]
                if unlisted:
                    unlisted_from_message = " (unlisted)" if namefrom in unlisted else ""
                    unlisted_to_message = " (unlisted)" if nameto in unlisted else ""
                    log_and_print(
                        flog,
                        f"Discarding upsilon={upsilon:.3f} for {namefrom}{unlisted_from_message} ->"
                        f" {nameto}{unlisted_to_message}",
                    )
                    continue
                if level_ids_of_level_name[namefrom][0] > level_ids_of_level_name[nameto][0]:
                    log_and_print(
                        flog,
                        f"WARNING: Swapping transition levels {namefrom} {level_ids_of_level_name[namefrom]} "
                        f"-> {nameto} {level_ids_of_level_name[nameto]}.",
                    )
                    namefrom, nameto = nameto, namefrom

                # add forbidden collisions between states within lower and upper terms if
                # the upper and lower levels have no J specified
                for name in (namefrom, nameto):
                    if name not in names_expanded:
                        for id1 in level_ids_of_level_name[name]:
                            for id2 in level_ids_of_level_name[name]:
                                if id1 < id2 and (id1, id2) not in upsilondict:
                                    upsilondict[id1, id2] = -2.0
                        names_expanded.add(name)

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
                        upsilonscaled = upsilon * (gvalues[id_lower] / lower_g_sum) * (gvalues[id_upper] / upper_g_sum)
                        # upsilon is symmetric and the output wants lower id < upper id; the
                        # terms' J levels can interleave, so order the key rather than
                        # dropping the pairs that come out reversed
                        key = (min(id_lower, id_upper), max(id_lower, id_upper))
                        if key in upsilondict and upsilondict[key] >= 0.0:
                            log_and_print(
                                flog,
                                f"ERROR: Duplicate collisional transition from {namefrom} <->"
                                f" {nameto} ({key[0]} -> {key[1]}). Keeping existing collision strength of"
                                f" {upsilondict[key]:.2e} instead of new value of"
                                f" {upsilonscaled:.2e}.",
                            )
                        else:
                            upsilondict[key] = upsilonscaled

    if number_expected_transitions < 0:
        log_and_print(flog, "WARNING: no '!Number of transitions' line found in collision data file")
    elif coll_lines_in < number_expected_transitions:
        msg = f"file specified {number_expected_transitions:d} transitions, but only {coll_lines_in:d} were found"
        raise ValueError(msg)
    elif coll_lines_in > number_expected_transitions:
        log_and_print(
            flog,
            f"WARNING: file specified {number_expected_transitions:d} transitions, but {coll_lines_in:d} were found",
        )
    else:
        log_and_print(flog, f"Read {coll_lines_in} effective collision strengths")
        log_and_print(flog, f"Output {len(upsilondict)} effective collision strengths")

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
    photoion_targetconfigs: list[list[tuple[str, float]] | None] | None,
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

    def logprint(strout: str) -> None:
        """Write to stdout, and to the ion log when the caller gave one."""
        if flog is None:
            print(strout)
        else:
            log_and_print(flog, strout)

    targetlist: list[list[tuple[int, float]]] = [[] for _ in range(dfenergy_levels.height)]
    targetlist_of_targetconfig: dict[str, list[tuple[int, float]]] = {}

    if photoion_targetconfigs is None:
        return targetlist

    # The comparison is per target configuration, not per level. Pull the names out one time.
    uppernamenoj_of_levelid = [levelname.split("[")[0] for levelname in dfenergy_levels_upperion["levelname"].to_list()]
    strippednames_of_levelid = [strip_name_separators(levelname) for levelname in uppernamenoj_of_levelid]

    for lowerlevelid in range(dfenergy_levels.height):
        targetconfig_fractions = photoion_targetconfigs[lowerlevelid]
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
                        names_of_strippedname: defaultdict[str, set[str]] = defaultdict(set)
                        for levelid in upperionlevelids:
                            names_of_strippedname[strippednames_of_levelid[levelid]].add(
                                uppernamenoj_of_levelid[levelid]
                            )
                        for strippedpart, partnames in names_of_strippedname.items():
                            if len(partnames) > 1:
                                msg = (
                                    f"Photoionisation target part '{strippedpart}' of '{targetconfig}' matched"
                                    f" more than one level name of the upper ion with the name separators"
                                    f" removed: {sorted(partnames)}. The part is ambiguous, so artisatomic"
                                    " cannot share the fraction."
                                )
                                logprint(f"ERROR: {msg}")
                                raise ValueError(msg)
                        matchednames = sorted(
                            {name for partnames in names_of_strippedname.values() for name in partnames}
                        )
                        logprint(
                            f"Photoionisation target '{targetconfig}' matched {matchednames} of the upper ion"
                            " with the name separators removed"
                        )
                if not upperionlevelids:
                    logprint(
                        f"WARNING: photoionisation target '{targetconfig}' matched no level of the upper ion,"
                        " so the upper ion's ground state is the target"
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


def read_hyd_phixsdata(force: bool = False) -> None:
    """Load the hydrogenic photoionization tables that the type 2, 3 and 8 fits interpolate.

    Fills the module-level hyd_phixs / hyd_gaunt tables. The functions that read the tables
    call this first, so a caller needs no call of its own. A second call does nothing unless
    force is set. Thresholds are taken from the H I level list, which is therefore required to
    be indexed by principal quantum number.
    """
    global max_hyd_l_n, max_hyd_gaunt_n
    if max_hyd_l_n != -1 and max_hyd_gaunt_n != -1 and not force:
        return

    # the cached (2l+1)-weighted sums come from the tables filled in below, so a reload must
    # not leave sums from the previous tables
    get_hydrogenic_sigma_summed_over_l.cache_clear()

    with Path(os.devnull).open("w", encoding="utf-8") as devnull:
        _hillier_ionization_energy_ev, dfhillier_energy_levels, _transitions = read_levels_and_transitions(
            1, 1, devnull
        )

    # the tables below are indexed by principal quantum number, so the H I level list must not
    # have been filtered (read_levels_and_transitions drops levels with no transitions). Not an
    # assert: every hydrogenic threshold depends on it, so it must survive python -O.
    if dfhillier_energy_levels["hillierlevelid"].to_list() != list(range(1, dfhillier_energy_levels.height + 1)):
        msg = (
            "H I level list is not indexed by principal quantum number, so the hydrogenic"
            " cross-section thresholds would be taken from the wrong levels"
        )
        raise ValueError(msg)
    # indexed by n - 1, i.e. the ionisation threshold wavelength of the level with principal
    # quantum number n
    lambdaangstrom_of_n = dfhillier_energy_levels["lambdaangstrom"].to_list()

    hyd_filename = hillier_ion_folder(1, 1) + "/5dec96/hyd_l_data.dat"
    print(f"Reading hydrogen photoionization cross sections from {hyd_filename}")
    max_n = -1
    l_start_u = 0.0
    l_del_u = 0.0
    with xopen_check_extension(hyd_filename) as fhyd:
        for line in fhyd:
            row = line.split()
            if " ".join(row[1:]) == "!Maximum principal quantum number":
                max_n = int(row[0])
                max_hyd_l_n = max_n

            if " ".join(row[1:]) == "!L_ST_U":
                l_start_u = fortran_float(row[0])

            if " ".join(row[1:]) == "!L_DEL_U":
                l_del_u = fortran_float(row[0])

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
                    msg = f"too many datapoints for (n,l)=({n},{l}), expected {num_points} but found {len(xs_values)}"
                    raise ValueError(msg)

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
    with xopen_check_extension(hyd_filename) as fhyd:
        for line in fhyd:
            row = line.split()
            if " ".join(row[1:]) == "!Maximum principal quantum number":
                max_n = int(row[0])
                max_hyd_gaunt_n = max_n

            if len(row) > 1:
                if row[1] == "!N_ST_U":
                    n_start_u = fortran_float(row[0])
                elif row[1] == "!N_DEL_U":
                    n_del_u = fortran_float(row[0])

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
                    msg = f"too many datapoints for n={n}, expected {num_points} but found {len(gaunt_values)}"
                    raise ValueError(msg)

            hyd_gaunt_energygrid_ryd[n] = [
                e_threshold_ev / ryd_to_ev * 10 ** (n_start_u + n_del_u * index) for index in range(num_points)
            ]
            hyd_gaunt_factor[n] = gaunt_values


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
        ion_handlers = add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "cmfgen")

    return ion_handlers

"""Read levels and transitions from the Kurucz gfall line lists."""

import itertools
import re
import string
from pathlib import Path

import polars as pl

from artisatomic.base import find_file_check_extension
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import gf_to_a_coefficient
from artisatomic.base import leveltuples_to_pldataframe
from artisatomic.base import log_and_print
from artisatomic.base import path_for_log
from artisatomic.base import PYDIR
from artisatomic.base import scan_file_lines
from artisatomic.base import TESTMODE

kuruczdatapath = (PYDIR / ".." / "atomic-data-kurucz").resolve()
if TESTMODE:
    kuruczdatapath /= "test_sample"


def parse_gfall(fname: str) -> pl.LazyFrame:
    """Parse one Kurucz gfall line list into a frame of transitions with their two levels.

    Each gfall row is a transition carrying both of its levels inline, in a fixed-width Fortran
    format. The two levels are ordered into lower/upper by energy here, since the file lists
    them in an arbitrary order. A negative energy in the file means a predicted (rather than
    measured) level, which is recorded in a "theoretical" flag and the magnitude kept.
    """
    # Code derived from the GFALL reader of carsus
    # https://github.com/tardis-sn/carsus/blob/master/carsus/io/kurucz/gfall.py
    gfall_fortran_format = (
        "F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,"
        "A10,F6.2,F6.2,F6.2,A4,I2,I2,I3,F6.3,I3,F6.3,I5,I5,"
        "1X,I1,A1,1X,I1,A1,I1,A3,I5,I5,I6"
    )

    gfall_columns = [
        "wavelength_nm",
        "loggf",
        "z_dot_ioncharge",
        "energyabovegsinpercm_first",
        "j_first",
        "blank1",
        "label_first",
        "energyabovegsinpercm_second",
        "j_second",
        "blank2",
        "label_second",
        "log_gamma_rad",
        "log_gamma_stark",
        "log_gamma_vderwaals",
        "ref",
        "nlte_level_no_first",
        "nlte_level_no_second",
        "isotope",
        "log_f_hyperfine",
        "isotope2",
        "log_iso_abundance",
        "hyper_shift_first",
        "hyper_shift_second",
        "blank3",
        "hyperfine_f_first",
        "hyperfine_note_first",
        "blank4",
        "hyperfine_f_second",
        "hyperfine_note_second",
        "line_strength_class",
        "line_code",
        "lande_g_first",
        "lande_g_second",
        "isotopic_shift",
    ]
    number_match = re.compile(r"\d+(\.\d+)?")
    type_match = re.compile(r"[FIXA]")
    type_dict = {"F": pl.Float64, "I": pl.Int64, "X": pl.String, "A": pl.String}
    field_types = [type_dict[item] for item in number_match.sub("", gfall_fortran_format).split(",")]

    field_widths = list(map(int, re.sub(r"\.\d+", "", type_match.sub("", gfall_fortran_format)).split(",")))
    # each field starts after the fields before it, so the last width starts no field
    field_offsets = list(itertools.accumulate(field_widths[:-1], initial=0))

    # read each line whole, then cut the fixed-width fields out of it
    gfall = scan_file_lines(fname).select(
        # a blank field, and a line too short to reach the field, both give a null
        pl.col("line").str.slice(offset, width).str.strip_chars().replace("", None).cast(dtype).alias(name)
        for name, offset, width, dtype in zip(gfall_columns, field_offsets, field_widths, field_types, strict=True)
    )

    gfall = gfall.drop_nulls(["z_dot_ioncharge", "energyabovegsinpercm_first", "energyabovegsinpercm_second"])
    double_columns = [col.replace("_first", "") for col in gfall.collect_schema().names() if col.endswith("first")]

    # due to the fact that energy is stored in 1/cm
    gfall = gfall.with_columns(
        order_lower_upper=pl.col("energyabovegsinpercm_first").abs() < pl.col("energyabovegsinpercm_second").abs()
    )
    gfall = gfall.with_columns(
        pl.when(pl.col("order_lower_upper"))
        .then(f"{column}_first")
        .otherwise(f"{column}_second")
        .alias(f"{column}_lower")
        for column in double_columns
    ).with_columns(
        pl.when(pl.col("order_lower_upper"))
        .then(f"{column}_second")
        .otherwise(f"{column}_first")
        .alias(f"{column}_upper")
        for column in double_columns
    )

    # Clean labels. str.replace_all(), not Expr.replace(): the latter swaps whole values that
    # equal the literal string "\s+", so the internal whitespace runs the gfall columns are padded
    # with ('s4d  1D') were never collapsed and went into the level names as-is.
    # fill_null(""): a blank label parsed to null, and a null is_in() result made filter() drop
    # the row, so U II lost 495 of its 595 lines. Only the three pseudo-level labels are ignored.
    ignored_labels = ["AVERAGE", "ENERGIES", "CONTINUUM"]
    gfall = gfall.with_columns(
        pl.col("label_lower").str.strip_chars().str.replace_all(r"\s+", " ").fill_null(""),
        pl.col("label_upper").str.strip_chars().str.replace_all(r"\s+", " ").fill_null(""),
    ).filter(
        (pl.col("label_lower").is_in(ignored_labels).not_()) & (pl.col("label_upper").is_in(ignored_labels).not_())
    )

    gfall = gfall.with_columns(
        energyabovegsinpercm_lower_predicted=pl.col("energyabovegsinpercm_lower") < 0,
        energyabovegsinpercm_lower=pl.col("energyabovegsinpercm_lower").abs(),
        energyabovegsinpercm_upper_predicted=pl.col("energyabovegsinpercm_upper") < 0,
        energyabovegsinpercm_upper=pl.col("energyabovegsinpercm_upper").abs(),
    )

    return gfall.with_columns(atomic_number=pl.col("z_dot_ioncharge").cast(pl.Int64)).with_columns(
        ion_charge=((pl.col("z_dot_ioncharge") - pl.col("atomic_number")) * 100).round().cast(pl.Int64),
    )


def find_gfall(atomic_number: int, ion_charge: int) -> Path:
    """Locate one ion's Kurucz line list, trying the extendedatoms and zztar layouts.

    Raises FileNotFoundError if the ion has no file, which is how callers detect that Kurucz
    has no data for it.
    """
    stems = [
        kuruczdatapath / "extendedatoms" / f"gf{atomic_number:02d}{ion_charge:02d}.lines",
        kuruczdatapath / "extendedatoms" / f"gf{atomic_number:02d}{ion_charge:02d}z.lines",
        kuruczdatapath / "zztar" / f"gf{atomic_number:02d}{ion_charge:02d}.all",
    ]
    for stem in stems:
        path_gfall = find_file_check_extension(stem)
        if path_gfall is not None:
            return path_gfall.resolve()

    msg = f"No Kurucz file for Z={atomic_number} ion_charge {ion_charge}."
    raise FileNotFoundError(msg)


def read_levels_and_transitions(atomic_number: int, ion_stage: int, flog) -> tuple[float, pl.DataFrame, pl.DataFrame]:
    """Read one ion from the Kurucz line lists.

    The files are transition lists rather than level lists, so the levels are recovered by
    taking the distinct lower and upper levels of every transition. The ionisation energy comes
    from NIST rather than the file.
    """
    ion_charge = ion_stage - 1

    log_and_print(flog, f"Using Kurucz for Z={atomic_number} ion_stage {ion_stage}")

    path_gfall = find_gfall(atomic_number, ion_charge)
    log_and_print(flog, f"Reading {path_for_log(path_gfall)}")

    gfall = parse_gfall(fname=str(path_gfall))
    column_renames = {
        "energyabovegsinpercm_{0}": "energyabovegsinpercm",
        "j_{0}": "j",
        "label_{0}": "label",
        "energyabovegsinpercm_{0}_predicted": "theoretical",
    }

    transition_columns = [
        "atomic_number",
        "ion_charge",
        "energyabovegsinpercm_lower",
        "j_lower",
        "energyabovegsinpercm_upper",
        "j_upper",
        "wavelength_nm",
        "loggf",
        # kept only for the duplicate-line test below, and dropped by the final select
        "label_lower",
        "label_upper",
        "isotope",
        "isotope2",
        "log_f_hyperfine",
        "hyperfine_f_lower",
        "hyperfine_f_upper",
        "hyper_shift_lower",
        "hyper_shift_upper",
    ]
    # The levels and the transitions come from the same rows, so read those rows once. Each
    # collect() of the lazy frame reads and parses the file again, and the file can be 150 MB.
    # The levels need the two columns below as well, and the transitions need no other column.
    dfgfall = gfall.select(
        [
            *transition_columns,
            "energyabovegsinpercm_lower_predicted",
            "energyabovegsinpercm_upper_predicted",
        ]
    ).collect()

    # One file holds one ion. The atomic number and the ion charge both come from the file's
    # z_dot_ioncharge column, so a second ion changes one of them. This test reads the rows that
    # are in memory: on the lazy frame it read and parsed the whole file a second time.
    if dfgfall.select(pl.n_unique("atomic_number"), pl.n_unique("ion_charge")).row(0) != (1, 1):
        msg = f"Expected exactly one unique ion in file {path_gfall}, but found multiple"
        raise ValueError(msg)

    gfall = dfgfall.lazy()

    e_lower_levels = gfall.rename({key.format("lower"): value for key, value in column_renames.items()})
    e_upper_levels = gfall.rename({key.format("upper"): value for key, value in column_renames.items()})

    selected_columns = ["atomic_number", "ion_charge", "energyabovegsinpercm", "j", "label", "theoretical"]
    dflevels = (
        pl.concat([e_lower_levels.select(selected_columns), e_upper_levels.select(selected_columns)])
        # maintain_order so that which label survives for a duplicated (energy, j) is reproducible;
        # without it the level names in adata.txt can differ from run to run
        .unique(["energyabovegsinpercm", "j"], keep="first", maintain_order=True)
        .sort("energyabovegsinpercm", "j")
        .select(
            pl.col("energyabovegsinpercm"),
            pl.col("j"),
            levelname=(
                pl.col("label")
                + ",enpercm="
                + pl.col("energyabovegsinpercm").cast(pl.Utf8)
                + ",j="
                + pl.col("j").cast(pl.String)
            ),
            g=2 * pl.col("j") + 1,
        )
        .collect()
    )
    dflevels = leveltuples_to_pldataframe(dflevels).with_columns(
        # this data set supplies no parities, and a null one never matches another, so
        # add_level_ids_forbidden() leaves every transition permitted
        parity=pl.lit(None, dtype=pl.Int64)
    )
    log_and_print(flog, f"Read {len(dflevels):d} levels")

    transitions = (
        gfall.select(transition_columns)
        # gfall lists some lines twice, once at the observed wavelength and once at the Ritz one
        # (Y II has one such pair at 241.7267 and 241.7308 nm, both loggf = 0). ARTIS adds the A
        # values of two rows that share a level pair, so a repeat would double the line.
        #
        # A row only counts as a repeat if the levels, the labels AND the strength all match.
        # Each on its own keeps rows that are separate lines:
        #  - Sr I has 785 rows sharing a level pair with different labels, whose strengths follow
        #    the spin rule, so they are lines whose levels the (energy, J) key above merged.
        #  - Sr III has 46 rows sharing a level pair and both labels whose loggf still differs,
        #    by as much as -1.911 against -5.742, and Sr II in the zztar layout has five more.
        # Dropping either kind would delete a real transition, and in Sr III would keep the
        # weaker of the two.
        .with_columns(gf=10 ** pl.col("loggf"))
        .join(
            dflevels.lazy().select(
                energyabovegsinpercm_lower=pl.col("energyabovegsinpercm"),
                j_lower=pl.col("j"),
                levelid_lower=pl.col("levelid"),
            ),
            on=["energyabovegsinpercm_lower", "j_lower"],
            how="left",
        )
        .join(
            dflevels.lazy().select(
                energyabovegsinpercm_upper=pl.col("energyabovegsinpercm"),
                j_upper=pl.col("j"),
                levelid_upper=pl.col("levelid"),
            ),
            on=["energyabovegsinpercm_upper", "j_upper"],
            how="left",
        )
        .with_columns(
            # wavelengths are in nanometers, so multiply by 10 to get Angstroms
            A=pl.col("gf")
            / (gf_to_a_coefficient * (2 * pl.col("j_upper") + 1) * (pl.col("wavelength_nm") * 10.0).pow(2))
        )
        .collect()
    )

    transitions_in = transitions.height
    transitions = transitions.unique(
        [
            "energyabovegsinpercm_lower",
            "j_lower",
            "energyabovegsinpercm_upper",
            "j_upper",
            "label_lower",
            "label_upper",
            "loggf",
            # an isotope or hyperfine component is its own line and can share everything above
            # with another, so the fields that tell them apart belong in the identity too
            "isotope",
            "isotope2",
            "log_f_hyperfine",
            "hyperfine_f_lower",
            "hyperfine_f_upper",
            "hyper_shift_lower",
            "hyper_shift_upper",
        ],
        keep="first",
        maintain_order=True,
    )
    if transitions.height < transitions_in:
        log_and_print(flog, f"Dropped {transitions_in - transitions.height:d} lines that gfall lists more than once")

    # the level ids follow a sort on (energy, J), while the file's pair was ordered by energy
    # alone, so two levels of one energy can come out with the higher id first: order the ids
    transitions = transitions.select(
        upperlevel=pl.max_horizontal("levelid_lower", "levelid_upper"),
        lowerlevel=pl.min_horizontal("levelid_lower", "levelid_upper"),
        A=pl.col("A"),
    )

    log_and_print(flog, f"Read {len(transitions):d} transitions")

    ionization_energy_in_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]
    log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")

    return ionization_energy_in_ev, dflevels, transitions


def get_level_valence_n(levelname: str) -> int | None:
    """Principal quantum number of the valence electron, read from a Kurucz level label.

    Returns None when the label cannot be parsed. The caller, match_hydrogenic_phixs(), then
    gives the level no estimate and writes a warning to the ion log. A guessed n would give the
    level a cross section of the wrong size without a trace in the output.

    A label can end in a parent term ("6s6p*(3P*)"), an odd-parity mark ("*"), or a prime that
    marks a second series ("d5p'"). All come after the valence orbital, so they are removed
    before the orbital is read.

    Kept separate from the other readers' versions: each data source names its levels
    differently, so a shared parser would have to guess which convention it is looking at.
    """
    namesplit = levelname.replace("  ", " ").split(" ")
    if len(namesplit) < 2 or not (part := namesplit[-2]):
        return None

    if part.endswith(")") and "(" in part:
        part = part[: part.rfind("(")]
    part = part.rstrip("*'")
    if not part:
        return None

    if part[-1] not in "spdfghijklmnopqr":
        # end of string is a number of electrons in the orbital, not a principal quantum number, so remove it
        if not part[-1].isdigit():
            return None
        part = part.rstrip(string.digits)

    # the digits before the valence orbital letter. A Kurucz label writes the electron count of
    # the shell before them without a space: "s25p" is 5s2 5p, "f36s" is 4f3 6s and "f125d" is
    # 4f12 5d. So a run of digits that follows an orbital letter starts with that count. A
    # two-digit run that ends in 0 is a two-digit n ("s10d" is 5s 10d), because no shell has
    # n = 0. A lower-case letter is an orbital letter here: the term letters are upper case.
    nmatch = re.search(r"([a-z]?)(\d+)[a-z]$", part)
    if nmatch is None:
        return None
    digits = nmatch.group(2)
    if nmatch.group(1):
        if len(digits) == 2 and digits[1] != "0":
            digits = digits[1:]
        elif len(digits) == 3:
            # a two-digit count and a one-digit n ("f125d"), unless that n would be 0: then a
            # one-digit count and a two-digit n ("s210d" is 5s2 10d)
            digits = digits[2:] if digits[2] != "0" else digits[1:]
        elif len(digits) > 3:
            return None
    return int(digits)

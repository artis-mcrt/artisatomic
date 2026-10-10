"""Read levels and transitions from the Kurucz gfall line lists (http://kurucz.harvard.edu/linelists/gfall/).

The reference for the line lists is Kurucz, R. L. (2017), Can. J. Phys., 95, 825-827,
doi:10.1139/cjp-2016-0794.
"""

import itertools
import re
import string
from pathlib import Path

import polars as pl

from artisatomic.base import find_file_check_extension
from artisatomic.base import fixed_width_column
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import gf_to_a_coefficient
from artisatomic.base import leveltuples_to_pldataframe
from artisatomic.base import log_and_print
from artisatomic.base import log_comment
from artisatomic.base import log_detail
from artisatomic.base import nist_ionization_energy_comment
from artisatomic.base import path_in_data_folder
from artisatomic.base import PYDIR
from artisatomic.base import scan_file_lines
from artisatomic.base import TESTMODE
from artisatomic.levelnames import lchars
from artisatomic.levelnames import split_count_and_n

kuruczfolder = PYDIR / ".." / "atomic-data-kurucz"
kuruczdatapath = kuruczfolder.resolve()
if TESTMODE:
    kuruczdatapath /= "test_sample"

# the "source:" line of the comment blocks in the output files (see Handler.description in iondata.py)
description = (
    "the Kurucz gfall line lists, http://kurucz.harvard.edu/linelists/gfall/. Kurucz, R. L. (2017), Can. J. Phys., 95,"
    " 825-827, doi:10.1139/cjp-2016-0794"
)


def parse_gfall(fname: str) -> pl.LazyFrame:
    """Parse one Kurucz gfall line list into a frame of transitions with their two levels.

    Each gfall row is a transition that carries both of its levels inline, in a fixed-width
    Fortran format. This function orders the two levels into lower/upper by energy, because the
    file lists them in an arbitrary order. A negative energy in the file means a predicted (not
    measured) level. The parser records that in a "theoretical" flag and keeps the magnitude.
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

    # gfall08oct17 (and the 2016 and 2017 versions before it) writes the loggf of the Fe I line at
    # 448.8906 nm as "-1 72". The space is at the position of the decimal point of the F7.3 field.
    # The source of the line gives log gf = -1.72 (Den Hartog, E. A., Ruffoni, M. P., Lawler, J. E.,
    # Pickering, J. C., Lind, K., & Brewer, N. R. 2014, ApJS, 215, 23, doi:10.1088/0067-0049/215/2/23).
    # So the reader reads a space at that position, between two digits, as the decimal point. A Fortran
    # read ignores the space and gives -0.172, and gfallvac08oct17 has that value.
    loggf_offset, loggf_width = field_offsets[1], field_widths[1]
    loggf_text = pl.col("line").str.slice(loggf_offset, loggf_width)
    loggf_point_missing = loggf_text.str.contains(r"^..\d \d")
    loggf_point_repaired = (
        pl.when(loggf_point_missing)
        .then(loggf_text.str.slice(0, 3) + "." + loggf_text.str.slice(4))
        .otherwise(loggf_text)
        .str.strip_chars()
    )

    # read each line whole, then cut the fixed-width fields out of it
    gfall = scan_file_lines(fname).select(
        *(
            # a blank field, and a line too short to reach the field, both give a null
            (loggf_point_repaired if name == "loggf" else fixed_width_column(offset, width))
            .replace("", None)
            .cast(dtype)
            .alias(name)
            for name, offset, width, dtype in zip(gfall_columns, field_offsets, field_widths, field_types, strict=True)
        ),
        loggf_point_missing=loggf_point_missing,
    )

    gfall = gfall.drop_nulls(["z_dot_ioncharge", "energyabovegsinpercm_first", "energyabovegsinpercm_second"])
    double_columns = [col.replace("_first", "") for col in gfall.collect_schema().names() if col.endswith("first")]

    # compare the magnitudes: a negative energy marks a predicted level, and the sign does not
    # order the levels
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
    # equal the literal string "\s+", so it cannot collapse the whitespace runs that pad the gfall
    # columns ('s4d  1D'). fill_null(""): a blank label parses to null, and a null is_in() result
    # makes filter() drop the row. The filter removes only the three pseudo-level labels.
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
    """Locate one ion's Kurucz line list in the extendedatoms layout or the zztar layout.

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
            return path_gfall

    msg = f"No Kurucz file for Z={atomic_number} ion_charge {ion_charge}."
    raise FileNotFoundError(msg)


# the LS term at the end of a level label, for example "3D" of "d5s a3D" or "3P" of "s4p *3P". The term
# can have a "?" after it, as in "(3F)9p 2F?". An extendedatoms label can end with a number, as in "B(1D)2F 2".
label_term_regex = re.compile(rf"(\d{{1,2}})([{lchars}])\??(?: \d+)?$")


def possible_j_expr(side: str, nelectrons: int) -> pl.Expr:
    """Give True where the J of the level of one side of a gfall row is a J that the level can have.

    2J is odd for an odd number of electrons and even for an even number. Where the label ends with
    an LS term, J must also be in the range |L - S| to L + S.
    """
    j = pl.col(f"j_{side}")
    twoj = (2 * j).round().cast(pl.Int64)
    term = pl.col(f"label_{side}").str.extract_groups(label_term_regex.pattern)
    twos = term.struct.field("1").cast(pl.Int64, strict=False) - 1
    twol = 2 * term.struct.field("2").replace_strict(list(lchars), list(range(len(lchars))), default=None)
    in_term_range = ((twol - twos).abs() <= twoj) & (twoj <= twol + twos)
    return (twoj % 2 == nelectrons % 2) & in_term_range.fill_null(value=True)


def fix_impossible_j(dfgfall: pl.DataFrame, nelectrons: int, flog) -> pl.DataFrame:
    """Give a level a possible J where a gfall row gives it a J that it cannot have.

    Some rows give a known level a J of 0.0 in place of its J. The reader keys the levels on the
    energy and J. Such a row therefore made an extra level with g = 1 at the energy of the real
    level. The row also took an A from the wrong g. If exactly one level with a possible J has the
    same energy, the row takes that J and the label of that level. If more than one has (an
    unresolved fine structure), the reader cannot tell the level of the row, and it drops the row.
    """
    sides = ("lower", "upper")
    possible_levels = pl.concat(
        dfgfall.filter(possible_j_expr(side, nelectrons)).select(
            energy=pl.col(f"energyabovegsinpercm_{side}"),
            possiblej=pl.col(f"j_{side}"),
            possiblelabel=pl.col(f"label_{side}"),
        )
        for side in sides
    ).unique(maintain_order=True)
    candidates = possible_levels.group_by("energy", maintain_order=True).agg(
        pl.col("possiblej").first(),
        pl.col("possiblelabel").first(),
        pl.col("possiblej").n_unique().alias("ncandidates"),
    )

    dfgfall = dfgfall.with_row_index("gfallrow")
    nchanged = 0
    ambiguousrows: set[int] = set()
    kept_levels: set[tuple[float, float]] = set()
    for side in sides:
        impossible = dfgfall.filter(possible_j_expr(side, nelectrons).not_()).select(
            "gfallrow", energy=pl.col(f"energyabovegsinpercm_{side}"), j=pl.col(f"j_{side}")
        )
        kept_levels.update(impossible.join(candidates, on="energy", how="anti").select("energy", "j").iter_rows())
        impossible = impossible.join(candidates, on="energy", how="inner")
        ambiguousrows.update(impossible.filter(pl.col("ncandidates") > 1)["gfallrow"].to_list())
        newj = impossible.filter(pl.col("ncandidates") == 1).select(
            "gfallrow", newj=pl.col("possiblej"), newlabel=pl.col("possiblelabel")
        )
        nchanged += newj.height
        dfgfall = (
            dfgfall.join(newj, on="gfallrow", how="left", maintain_order="left")
            .with_columns(
                pl.coalesce("newj", f"j_{side}").alias(f"j_{side}"),
                pl.coalesce("newlabel", f"label_{side}").alias(f"label_{side}"),
            )
            .drop("newj", "newlabel")
        )

    if nchanged > 0:
        log_comment(
            flog,
            ("adata", "transitiondata"),
            f"The file gives {nchanged:d} levels of its lines a J that the level cannot have. The reader gave each one"
            " the J of the one level at the same energy with a possible J.",
        )
    if kept_levels:
        log_comment(
            flog,
            ("adata", "transitiondata"),
            f"WARNING: {len(kept_levels):d} levels keep a J that they cannot have, because no other level has the same"
            " energy.",
        )
    if ambiguousrows:
        log_comment(
            flog,
            ("transitiondata",),
            f"The reader dropped {len(ambiguousrows):d} lines. Each one has a level with a J that it cannot have, and"
            " more than one level at the same energy has a possible J.",
        )
    return dfgfall.filter(pl.col("gfallrow").is_in(sorted(ambiguousrows)).not_()).drop("gfallrow")


# a component level is at most this far from the level that it belongs to. In gfall08oct17, the
# energies of the components of one level spread over 0.6 cm^-1 at most (Li I 7f 2F7/2).
component_shift_tolerance_percm = 1.0


def combine_line_components(dfgfall: pl.DataFrame, flog) -> pl.DataFrame:
    """Combine the isotope and hyperfine components of each line into one line.

    gfall gives a component the gf value of the whole line. Two more fields give the log of its
    isotope fraction and the log of its hyperfine fraction. A component names its isotope. The
    level energies of a component include its isotope shift and its hyperfine shift. A negative
    energy there is a shift below the level, and not a predicted level. Without this function, each
    component is a full line between two sublevels of its own. ARTIS adds the A values of the
    transitions of a level pair, and it has no isotope or hyperfine levels.

    A component level gets the energy of the nearest level with the same label and J in the lines
    that are not split. That level must be near enough. The labels are not unique, so the label
    and J alone cannot identify a level. Each other group of component levels with the same label
    and J and near energies gets the mean energy of the group, weighted by gf.

    Some files give a line as a whole line and also as the components of one isotope. The function
    then keeps the whole line. It also drops a combined line between two sublevels of one level,
    because such a line joins a level to itself.
    """
    fractions = [pl.col(column).fill_null(0.0) for column in ("log_f_hyperfine", "log_iso_abundance")]
    # a fraction of one gives a log of 0, so the isotope also marks a component
    is_component = (pl.col("isotope").fill_null(0) != 0) | pl.any_horizontal(fraction != 0.0 for fraction in fractions)
    components = dfgfall.filter(is_component)
    if components.is_empty():
        return dfgfall
    wholelines = dfgfall.filter(is_component.not_())

    components = components.with_row_index("componentrow").with_columns(
        gf=10 ** (pl.col("loggf") + pl.sum_horizontal(fractions))
    )
    sides = ("lower", "upper")
    occurrences = pl.concat(
        components.select(
            "componentrow",
            "gf",
            side=pl.lit(side),
            label=pl.col(f"label_{side}"),
            j=pl.col(f"j_{side}"),
            energyabs=pl.col(f"energyabovegsinpercm_{side}"),
            energy=pl.when(pl.col(f"energyabovegsinpercm_{side}_predicted"))
            .then(-pl.col(f"energyabovegsinpercm_{side}"))
            .otherwise(pl.col(f"energyabovegsinpercm_{side}")),
        )
        for side in sides
    )
    known_levels = pl.concat(
        wholelines.select(
            label=pl.col(f"label_{side}"), j=pl.col(f"j_{side}"), knownenergy=pl.col(f"energyabovegsinpercm_{side}")
        )
        for side in sides
    ).unique()

    matched = (
        occurrences.join(known_levels, on=["label", "j"], how="inner")
        .with_columns(distance=(pl.col("knownenergy") - pl.col("energyabs")).abs())
        .filter(pl.col("distance") <= component_shift_tolerance_percm)
        .sort("componentrow", "side", "distance", "knownenergy")
        .group_by("componentrow", "side", maintain_order=True)
        .agg(mergedenergy=pl.col("knownenergy").first())
    )
    unmatched = (
        occurrences.join(matched, on=["componentrow", "side"], how="anti")
        .sort("label", "j", "energy")
        .with_columns(
            group=(
                (pl.col("label") != pl.col("label").shift())
                | (pl.col("j") != pl.col("j").shift())
                | (pl.col("energy") - pl.col("energy").shift() > component_shift_tolerance_percm)
            )
            .fill_null(value=True)
            .cum_sum()
        )
        .with_columns(
            # rounded to the precision of the energies in gfall (0.001 cm^-1)
            mergedenergy=((pl.col("energy") * pl.col("gf")).sum() / pl.col("gf").sum()).over("group").round(3).abs()
        )
        .select("componentrow", "side", "mergedenergy")
    )
    # The ground level is at 0 cm^-1. If no whole line gives it, the mean of its component levels
    # is a hyperfine shift above 0, so the lowest group of component levels gets 0.
    hasgroundline = not wholelines.filter(
        pl.any_horizontal(pl.col(f"energyabovegsinpercm_{side}") == 0.0 for side in sides)
    ).is_empty()
    lowestmerged = unmatched.select(pl.col("mergedenergy").min()).item()
    if not hasgroundline and lowestmerged is not None and lowestmerged <= component_shift_tolerance_percm:
        unmatched = unmatched.with_columns(
            mergedenergy=pl.when(pl.col("mergedenergy") == lowestmerged).then(0.0).otherwise(pl.col("mergedenergy"))
        )
    mergedenergies = pl.concat([matched, unmatched])
    for side in sides:
        components = components.drop(f"energyabovegsinpercm_{side}").join(
            mergedenergies.filter(pl.col("side") == side).select(
                "componentrow", pl.col("mergedenergy").alias(f"energyabovegsinpercm_{side}")
            ),
            on="componentrow",
            how="left",
            maintain_order="left",
        )

    # with the labels: the reader keeps two levels with the same energy and J but other labels
    linekey = [
        "energyabovegsinpercm_lower",
        "j_lower",
        "label_lower",
        "energyabovegsinpercm_upper",
        "j_upper",
        "label_upper",
    ]
    # each component carries the gf value of its whole line, so two lines between the same levels
    # stay two transitions, as two whole lines do
    combined = components.group_by("atomic_number", "ion_charge", *linekey, "loggf", maintain_order=True).agg(
        pl.col("gf").sum()
    )
    ncombined = combined.height
    combined = combined.filter(
        (pl.col("energyabovegsinpercm_lower") != pl.col("energyabovegsinpercm_upper"))
        | (pl.col("j_lower") != pl.col("j_upper"))
    )
    nselfline = ncombined - combined.height
    combined = combined.join(wholelines.select(*linekey, "loggf"), on=[*linekey, "loggf"], how="anti")
    nrepeat = ncombined - nselfline - combined.height
    log_comment(
        flog,
        ("adata", "transitiondata"),
        f"The reader combined {components.height:d} isotope and hyperfine components into {ncombined:d}"
        " lines. A component has the gf value of the whole line times its isotope fraction and its hyperfine"
        " fraction.",
    )
    if nrepeat > 0 or nselfline > 0:
        log_comment(
            flog,
            ("transitiondata",),
            f"The reader dropped {nrepeat:d} combined lines that the file also gives as a whole line, and"
            f" {nselfline:d} combined lines that join a level to itself.",
        )
    combined = combined.with_columns(
        loggf=pl.col("gf").log10(),
        energyabovegsinpercm_lower_predicted=pl.lit(value=False),
        energyabovegsinpercm_upper_predicted=pl.lit(value=False),
    )
    return pl.concat([wholelines, combined], how="diagonal_relaxed").select(wholelines.columns)


def read_levels_and_transitions(atomic_number: int, ion_stage: int, flog) -> tuple[float, pl.DataFrame, pl.DataFrame]:
    """Read one ion from the Kurucz line lists.

    The files are transition lists, not level lists. So this function recovers the levels as the
    distinct lower and upper levels of every transition. The ionisation energy comes from NIST
    rather than the file.
    """
    ion_charge = ion_stage - 1

    log_and_print(flog, f"The Kurucz reader reads Z={atomic_number} ion_stage {ion_stage}.")

    path_gfall = find_gfall(atomic_number, ion_charge)
    log_comment(
        flog,
        ("adata", "transitiondata"),
        f"The levels and the transitions come from {path_in_data_folder(path_gfall, kuruczfolder)}.",
    )

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
        "loggf",
        # kept only for the duplicate-line test below, and dropped by the final select
        "label_lower",
        "label_upper",
    ]
    # The levels and the transitions come from the same rows, so read those rows once. Each
    # collect() of the lazy frame reads and parses the file again, and the file can be 150 MB.
    # The levels need the predicted flags as well, and combine_line_components() needs the
    # isotope fields. The reader reads no other field. gfall08oct17 writes F = 10 as "A" in the
    # hyperfine F fields, so a read of those fields stops the read of V I, Mn I, Co I and Nb I-II.
    dfgfall = gfall.select(
        [
            *transition_columns,
            "energyabovegsinpercm_lower_predicted",
            "energyabovegsinpercm_upper_predicted",
            "isotope",
            "log_f_hyperfine",
            "log_iso_abundance",
            "loggf_point_missing",
        ]
    ).collect()

    for row in dfgfall.filter(pl.col("loggf_point_missing")).iter_rows(named=True):
        log_detail(
            flog,
            ("transitiondata",),
            "loggf decimal point",
            f"The loggf field of the line between the levels at {row['energyabovegsinpercm_lower']} and"
            f" {row['energyabovegsinpercm_upper']} cm^-1 has a space in place of the decimal point. The reader"
            f" reads it as {row['loggf']}.",
        )
    dfgfall = dfgfall.drop("loggf_point_missing")

    # One file holds one ion. The atomic number and the ion charge both come from the file's
    # z_dot_ioncharge column, so a second ion changes one of them. This test reads the rows in
    # memory. A test on the lazy frame would read and parse the whole file again.
    if dfgfall.is_empty():
        msg = (
            f"{path_gfall} has no line between two levels that the reader can use. The reader ignores"
            " the levels with the labels AVERAGE, ENERGIES and CONTINUUM."
        )
        raise ValueError(msg)
    if dfgfall.select(pl.n_unique("atomic_number"), pl.n_unique("ion_charge")).row(0) != (1, 1):
        msg = f"Expected exactly one unique ion in file {path_gfall}, but found multiple"
        raise ValueError(msg)

    dfgfall = combine_line_components(dfgfall, flog)
    dfgfall = fix_impossible_j(dfgfall, atomic_number - ion_charge, flog)
    gfall = dfgfall.lazy()

    e_lower_levels = gfall.rename({key.format("lower"): value for key, value in column_renames.items()})
    e_upper_levels = gfall.rename({key.format("upper"): value for key, value in column_renames.items()})

    selected_columns = ["atomic_number", "ion_charge", "energyabovegsinpercm", "j", "label", "theoretical"]
    dflevels = (
        pl.concat([e_lower_levels.select(selected_columns), e_upper_levels.select(selected_columns)])
        # maintain_order makes the label that survives for a duplicated (energy, j) reproducible.
        # Without it the level names in adata.txt can differ from run to run.
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
    # ARTIS takes the first level as the ground level. A hydrogenic ion has n-averaged levels with
    # the label AVERAGE, and the filter of parse_gfall() removes all lines of its ground level.
    lowestenergy = dflevels.select(pl.col("energyabovegsinpercm").min()).item()
    if lowestenergy != 0.0:
        msg = (
            f"The lowest level of {path_gfall} that the reader can use is at {lowestenergy} cm^-1 and not at 0."
            " The ion would have no ground level. The reader ignores the levels with the labels AVERAGE,"
            " ENERGIES and CONTINUUM."
        )
        raise ValueError(msg)
    log_and_print(flog, f"The reader got {len(dflevels):d} levels.")

    transitions = (
        gfall.select(transition_columns)
        # gfall lists some lines twice, once at the observed wavelength and once at the Ritz one.
        # Y II has one such pair at 241.7267 and 241.7308 nm, both with loggf = 0. ARTIS adds
        # the A values of two rows that share a level pair, so a repeat would double the line.
        #
        # A row only counts as a repeat if the levels, the labels AND the strength all match.
        # Each on its own keeps rows that are separate lines:
        #  - Sr I has 785 rows that share a level pair but have different labels. Their strengths
        #    follow the spin rule, so they are lines whose levels the (energy, J) key above merged.
        #  - Sr III has 46 rows that share a level pair and both labels but differ in loggf, by
        #    as much as -1.911 against -5.742. Sr II in the zztar layout has five more.
        # A drop of either kind would delete a real transition, and in Sr III it would keep the
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
            # polars gives a join no defined row order without this. The unique() below keeps the
            # first of two duplicates, and the writer keeps tied rows in this order.
            maintain_order="left",
        )
        .join(
            dflevels.lazy().select(
                energyabovegsinpercm_upper=pl.col("energyabovegsinpercm"),
                j_upper=pl.col("j"),
                levelid_upper=pl.col("levelid"),
            ),
            on=["energyabovegsinpercm_upper", "j_upper"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            # The vacuum wavelength in Angstrom from the level energies, not the wavelength field.
            # That field is an air wavelength above 200 nm, and gfall caps it at 999999.9999 nm.
            # Sr I has 102 lines at the cap, with A too large by up to 1.6e6.
            A=pl.col("gf")
            / (
                gf_to_a_coefficient
                * (2 * pl.col("j_upper") + 1)
                * (1e8 / (pl.col("energyabovegsinpercm_upper") - pl.col("energyabovegsinpercm_lower")).abs()).pow(2)
            )
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
        ],
        keep="first",
        maintain_order=True,
    )
    if transitions.height < transitions_in:
        log_comment(
            flog,
            ("transitiondata",),
            f"The reader dropped {transitions_in - transitions.height:d} lines that gfall lists more than one time.",
        )

    # the level ids follow a sort on (energy, J), but the file ordered its pair by energy alone.
    # So two levels of one energy can come out with the higher id first: order the ids.
    transitions = transitions.select(
        upperlevel=pl.max_horizontal("levelid_lower", "levelid_upper"),
        lowerlevel=pl.min_horizontal("levelid_lower", "levelid_upper"),
        A=pl.col("A"),
    )

    log_and_print(flog, f"The reader got {len(transitions):d} transitions.")

    ionization_energy_in_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]
    log_comment(flog, ("adata",), nist_ionization_energy_comment)
    log_and_print(flog, f"The NIST table gives an ionisation energy of {ionization_energy_in_ev} eV.")

    return ionization_energy_in_ev, dflevels, transitions


def get_level_valence_n(levelname: str) -> int | None:
    """Principal quantum number of the valence electron, read from a Kurucz level label.

    Returns None for a label that it cannot parse. The caller, match_hydrogenic_phixs(), then
    gives the level no estimate and writes a warning to the log file. A guessed n would give the
    level a cross section of the wrong size without a trace in the output.

    A label can end in a parent term ("6s6p*(3P*)"), an odd-parity mark ("*"), or a prime that
    marks a second series ("d5p'"). All come after the valence orbital, so the parser removes
    them before it reads the orbital.

    Kept separate from the other readers' versions. Each data source names its levels
    differently, so a shared parser would have to guess the convention of each name.
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
        # the end of the string is a number of electrons in the orbital, not a principal quantum number: remove it
        if not part[-1].isdigit():
            return None
        part = part.rstrip(string.digits)

    # the digits before the valence orbital letter. A Kurucz label writes the electron count of
    # the shell before them without a space. "s25p" is 5s2 5p, "f36s" is 4f3 6s and "f125d" is
    # 4f12 5d. So a run of digits that follows an orbital letter starts with that count. The
    # extendedatoms labels write the first orbital with its n and no count: "5s14d" is 5s 14d.
    # A lower-case letter is an orbital letter here: the term letters are upper case.
    nmatch = re.search(r"(?:(\d)?([a-z]))?(\d+)([a-z])$", part)
    if nmatch is None:
        return None
    if nmatch.group(1) and nmatch.group(2) and len(nmatch.group(3)) <= 2:
        # n is at most two digits, so a three-digit run holds a count too ("3d104s" is 3d10 4s)
        return int(nmatch.group(3))
    return split_count_and_n(nmatch.group(2) or "", nmatch.group(3), nmatch.group(4))

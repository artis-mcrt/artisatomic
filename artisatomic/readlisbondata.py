"""Read levels and transitions from the Lisbon Atomic Group data set (the "lisbon" handler)."""

import os
import typing as t
from pathlib import Path

import polars as pl

from artisatomic.base import elsymbols
from artisatomic.base import find_file_check_extension_or_raise
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import gf_to_a_coefficient
from artisatomic.base import levelid_of_fileindex_map
from artisatomic.base import log_and_print
from artisatomic.base import PYDIR
from artisatomic.base import resolve_transition_levelids
from artisatomic.base import roman_numerals
from artisatomic.base import split_levels_above_ionization
from artisatomic.base import Transition


def read_levels_csv(filename: Path | str) -> pl.DataFrame:
    """Read the levels CSV of one ion, past the eight lines of provenance.

    The column names come from Andreas Floers' code in git.gsi.de:nucastro/opacities.git.

    skip_lines, not skip_rows: skip_rows reads CSV rows, so a quote character in the provenance
    text would swallow the header. infer_schema_length=None reads the whole column, as pandas did.
    A sample of the first rows can give Int64 to a float column.

    The first column of the CSV numbers the levels. The lines name their levels by POSITION. So
    the reader writes that position into a column of its own. It does not use the CSV column as a
    key. The position stays with the level through the drop of the levels above the ionisation
    energy and through the sort by energy.
    """
    return (
        pl.read_csv(find_file_check_extension_or_raise(filename), skip_lines=8, infer_schema_length=None)
        .select(
            energy=pl.col("Energy[cm^-1]"),
            j=0.5 * (pl.col("g") - 1),
            label=pl.col("RelConfig"),
        )
        .with_row_index("fileposition")
    )


def read_lines_csv(filename: Path | str) -> pl.DataFrame:
    """Read the transitions CSV of one ion, past the eight lines of provenance.

    See read_levels_csv() for the reason that this reader skips lines and reads the whole column.
    """
    return pl.read_csv(find_file_check_extension_or_raise(filename), skip_lines=8, infer_schema_length=None).select(
        level_index_lower=pl.col("Lower"),
        level_index_upper=pl.col("Upper"),
        gf=pl.col("gf"),
        # keep the wavelength in Angstrom: the gf-to-A constant in read_lines_data() expects Angstrom
        wavelength=pl.col("Wavelength[Ang]"),
    )


def get_levelname(row, fileindex: int):
    """Name a Lisbon level from its label, J and file index.

    The label alone is not unique, and neither is the label with J. In Nd II, most levels share
    their relativistic configuration and J with another level. The file index makes the name
    unique, as the FAC, Floers+25 and MONS readers do with theirs.
    """
    return f"{row['label']}, j={row['j']}, index={fileindex}"


def read_levels_data(dflevels):
    """Convert the Lisbon level table to level tuples, sorted by energy.

    Also returns the map from the file index to the zero-based level id, which
    read_lines_data() needs because the sort by energy reorders the levels.

    The lines name their levels by POSITION in the levels file, which read_levels_csv() writes
    into the fileposition column. That position keys the map. Without it, the map would depend on
    a levels CSV whose numbers happen to start at zero.

    This data set supplies no parities, so every level's parity is null and the Laporte rule
    never fires. It does supply J, which is part of each level name, so the delta J rule alone
    decides whether a transition is forbidden here.

    read_levels_and_transitions() rejects a level with no energy before it calls this function.
    """
    # each level carries its file position through the sort. The sort is stable, so levels of one
    # energy keep the order of the file
    dflevels = dflevels.sort("energy", maintain_order=True)

    energy_levels = [
        EnergyLevelTuple(
            levelname=get_levelname(row, row["fileposition"]),
            parity=None,  # no parity in this data set, so the Laporte rule cannot fire
            j=float(row["j"]),
            g=2 * row["j"] + 1,
            energyabovegsinpercm=float(row["energy"]),
        )
        for row in dflevels.iter_rows(named=True)
    ]

    return energy_levels, levelid_of_fileindex_map(dflevels["fileposition"], "the Lisbon levels file")


def read_lines_data(energy_levels, dflines, levelid_of_fileindex):
    """Convert Lisbon lines to transitions referencing zero-based level ids.

    The lines name their levels by position in the levels file, and read_levels_data() sorted the
    levels by energy. So the reader maps every position through levelid_of_fileindex and does not
    use it directly. A line that names a level that does not exist is an error, not something to
    skip. The caller removes the lines that name a level above the ionisation energy first.

    A = gf / (gf_to_a_coefficient * g_upper * wavelength^2) with the wavelength in Angstrom, as
    in readkuruczdata and readmonsdata. g_upper is the g of the level that is the upper level
    after the reader resolves the ids. It is not the g of the level that the file labels "Upper".
    The file can list a pair in the reverse order, and the swap must not leave A with the wrong g.
    """
    transitions = []

    for row in dflines.iter_rows(named=True):
        lowerlevel, upperlevel = resolve_transition_levelids(
            row["level_index_lower"], row["level_index_upper"], levelid_of_fileindex, "the Lisbon transitions file"
        )

        A = row["gf"] / (gf_to_a_coefficient * energy_levels[upperlevel].g * row["wavelength"] ** 2)
        transitions.append(Transition(lowerlevel=lowerlevel, upperlevel=upperlevel, A=A))

    return transitions


class EnergyLevelTuple(t.NamedTuple):
    """One Lisbon energy level."""

    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: int | None  # None where the data set gives no parity
    j: float  # the level's J, which is part of its name and the source of its g


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the Lisbon data set.

    The CSV files are not part of artisatomic. ARTISATOMIC_LISBON_PATH overrides the directory
    that the reader searches.
    """
    elsym = elsymbols[atomic_number]
    ion_stage_roman = roman_numerals[ion_stage]

    # not an assert: the reader knows the file layout of these ions only, and the check must
    # survive python -O. A different element or stage would give a confusing file-not-found error
    if elsym not in {"Nd", "U"}:
        msg = f"The Lisbon data set holds Nd and U only, not {elsym} (Z={atomic_number})"
        raise ValueError(msg)
    if ion_stage not in {2, 3}:
        msg = f"The Lisbon data set holds ion stages 2 and 3 only, not ion stage {ion_stage}"
        raise ValueError(msg)

    print(f"Reading Lisbon data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman})")

    # the Lisbon CSVs are not part of this repository, so the location is configurable. This check
    # comes first: polars would otherwise report only the missing file, not what to set
    lisbonpath = Path(os.environ.get("ARTISATOMIC_LISBON_PATH", PYDIR / ".." / "atomic-data-lisbon")).resolve()
    if not lisbonpath.is_dir():
        msg = (
            f"Lisbon data directory {lisbonpath} not found. Set ARTISATOMIC_LISBON_PATH to the directory that holds"
            " the per-ion <El>/<El><Stage>/<El><Stage>_Levels.csv and _Transitions.csv files."
        )
        raise FileNotFoundError(msg)

    # from NIST, as every other reader whose data set carries no ionisation energy does. This was
    # -1, which went into adata.txt verbatim as the ion's ionisation energy.
    ionization_energy_in_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]
    log_and_print(flog, f"ionisation energy: {ionization_energy_in_ev} eV")

    iondir = lisbonpath / elsym / f"{elsym}{ion_stage_roman}"
    dfalllevels = read_levels_csv(iondir / f"{elsym}{ion_stage_roman}_Levels.csv")
    # not an assert: an empty frame would write an ion with no levels. The pandas reader that this
    # replaced raised a KeyError here, as the DREAM reader's guard does
    if dfalllevels.is_empty():
        msg = f"The Lisbon data has no levels for Z={atomic_number} ion_stage {ion_stage}"
        raise ValueError(msg)

    # not an assert: a blank energy sorts before every number in polars, so it would take level
    # id 0 and shift every other id. float() on it then reports neither the file nor the row.
    # This runs before the drop below, which keeps such a level but cannot see it
    if dfalllevels["energy"].null_count() > 0:
        msg = "The Lisbon levels file has a level with no energy"
        raise ValueError(msg)

    # drop the levels above the ionisation energy, as the FAC reader does, but keep their file
    # positions. With them, the filter below knows whether a line names a dropped level
    dflevels, filepositions_above_ionization = split_levels_above_ionization(
        dfalllevels, "energy", "fileposition", ionization_energy_in_ev, flog
    )

    # not an assert: an ion with no bound level would go to the output as an empty ion
    if dflevels.is_empty():
        msg = (
            f"Every one of the {dfalllevels.height} levels of Z={atomic_number} ion_stage {ion_stage} is above the"
            f" ionisation energy of {ionization_energy_in_ev} eV."
        )
        raise ValueError(msg)

    # the map associates the file indices with the energy-sorted level ids (0 indexed)
    energy_levels, levelid_of_fileindex = read_levels_data(dflevels)

    log_and_print(flog, f"Read {len(energy_levels):d} levels")

    dfalllines = read_lines_csv(iondir / f"{elsym}{ion_stage_roman}_Transitions.csv")

    # a line that names a dropped level goes with it, because that level has no level id
    dflines = dfalllines.filter(
        ~pl.col("level_index_lower").is_in(filepositions_above_ionization)
        & ~pl.col("level_index_upper").is_in(filepositions_above_ionization)
    )
    if dflines.height < dfalllines.height:
        log_and_print(
            flog,
            f"WARNING: skipped {dfalllines.height - dflines.height:d} transitions that reference a level above the"
            " ionisation energy",
        )

    transitions = read_lines_data(energy_levels, dflines, levelid_of_fileindex)

    # the writer accepts an ion with no transition, so this is a warning and not an error
    if not transitions and dfalllines.height > 0:
        log_and_print(
            flog,
            f"WARNING: every one of the {dfalllines.height} transitions of Z={atomic_number} ion_stage {ion_stage}"
            f" names a level above the ionisation energy of {ionization_energy_in_ev} eV."
            " The ion goes to the output with no transitions.",
        )

    log_and_print(flog, f"Read {len(transitions):d} transitions")

    return ionization_energy_in_ev, energy_levels, transitions

"""Read helium levels and transitions from the Boyle AOIFE data set.

The data set belongs to the helium work of Boyle, A., Sim, S. A., Hachinger, S., Kerzendorf, W.
(2017), A&A, 599, A46, doi:10.1051/0004-6361/201629712.
"""

import typing as t
from functools import cache

import polars as pl

from artisatomic.base import hc_in_ev_cm
from artisatomic.base import log_comment
from artisatomic.base import path_for_log
from artisatomic.base import PYDIR

# artisatomic/hdf5_to_parquet.py writes one Parquet file into this folder for each table of aoife.hdf5
datafolder = PYDIR / ".." / "atomic-data-helium-boyle" / "aoife"

# the "source:" line of the comment blocks in the output files (see Handler.description in iondata.py)
description = (
    "the AOIFE helium data set. Boyle, A., Sim, S. A., Hachinger, S., Kerzendorf, W. (2017), A&A, 599, A46,"
    " doi:10.1051/0004-6361/201629712"
)


@cache
def read_aoife_table(tablename: str) -> pl.DataFrame:
    """Read one table of the AOIFE data set, once, on first use."""
    path = datafolder / f"{tablename}.parquet"
    # not an assert: the user must convert aoife.hdf5 once, and the message says how
    if not path.exists():
        msg = f"{path} does not exist. Make it with artisatomic/hdf5_to_parquet.py from aoife.hdf5."
        raise FileNotFoundError(msg)
    return pl.read_parquet(path)


class EnergyLevelRow(t.NamedTuple):
    """One level of the AOIFE levels_data table, with the derived fields appended.

    The table's energy column is the energy above the ground state in eV. The reader converts it
    to cm^-1 once, in energyabovegsinpercm.
    """

    atomic_number: float
    ion_number: float
    level_number: float
    g: float
    metastable: float
    energyabovegsinpercm: float
    parity: int | None  # None where the data set gives no parity
    levelname: str


class TransitionTuple(t.NamedTuple):
    """One bound-bound transition of the AOIFE lines_data table."""

    atomic_number: float
    ion_stage: float
    lowerlevel: int
    upperlevel: int
    A: float
    lambdaangstrom: float


def read_ionization_data(atomic_number, ion_stage):
    """Ionisation energy in eV of one ion, from the AOIFE ionization_data table.

    The ion_number of the table is the charge after the ionisation, so it is equal to ion_stage.
    He III is a bare nucleus, so the table has no row for it. The function uses a sentinel instead.
    """
    if (atomic_number, ion_stage) == (2, 3):
        return 999999.0

    # item() stops the run if the table has no row or more than one row for the ion
    return (
        read_aoife_table("ionization_data")
        .filter((pl.col("atomic_number") == atomic_number) & (pl.col("ion_number") == ion_stage))
        .get_column("ionization_energy")
        .item()
    )


def read_levels_data(atomic_number, ion_stage):
    """Read one ion's energy levels from the AOIFE levels_data table.

    The table numbers ion stages from zero, so the reader matches ion_stage against ion_number + 1.
    Levels have no spectroscopic names, so the name of each level holds its zero-based level number.
    """
    dflevels = read_aoife_table("levels_data").filter(
        (pl.col("atomic_number") == atomic_number) & (pl.col("ion_number") == ion_stage - 1)
    )

    energy_levels = [
        # named fields, not *row plus three positional extras: a bare 0 in the extras would become
        # the parity of every level
        EnergyLevelRow(
            atomic_number=row["atomic_number"],
            ion_number=row["ion_number"],
            level_number=row["level_number"],
            g=row["g"],
            metastable=row["metastable"],
            # the AOIFE energy column is in eV (He I 1s2s 3S is 19.8196)
            energyabovegsinpercm=row["energy"] / hc_in_ev_cm,
            # No parity: this data set supplies none. add_level_ids_forbidden() marks a transition
            # forbidden when its two levels share one parity, and helium has many permitted ones. A
            # null parity never matches another, here as in the other readers whose data set has
            # no parities.
            parity=None,
            # int() as read_lines_data() does, so the two agree on the name whatever type the
            # table stores the level number in
            levelname=f"level{int(row['level_number']):05d}",
        )
        for row in dflevels.iter_rows(named=True)
    ]

    # not an assert: read_lines_data() takes the table's level numbers as the zero-based level ids
    # that leveltuples_to_pldataframe() assigns by row. The two numberings must agree
    level_numbers = [int(level.level_number) for level in energy_levels]
    if level_numbers != list(range(len(energy_levels))):
        msg = (
            f"the AOIFE levels of Z={atomic_number} ion_stage {ion_stage} are not numbered 0 to"
            f" {len(energy_levels) - 1} in table order"
        )
        raise ValueError(msg)

    return energy_levels


def read_lines_data(atomic_number, ion_stage):
    """Read one ion's bound-bound transitions from the AOIFE lines_data table.

    The table's level numbers are already zero-based, the same as the level ids in memory. The
    table gives no collision strengths.
    """
    dflines = read_aoife_table("lines_data").filter(
        (pl.col("atomic_number") == atomic_number) & (pl.col("ion_number") == ion_stage - 1)
    )

    transitions = []
    for row in dflines.iter_rows(named=True):
        # transitiondata.txt has the lower id first, so this code swaps a pair that the table
        # lists in the reverse order. The table stores the level numbers as floats
        levelid_lower = min(int(row["level_number_lower"]), int(row["level_number_upper"]))
        levelid_upper = max(int(row["level_number_lower"]), int(row["level_number_upper"]))
        transitions.append(
            TransitionTuple(
                row["atomic_number"], row["ion_number"], levelid_lower, levelid_upper, row["A_ul"], row["wavelength"]
            )
        )

    return transitions


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion for the "boyle" handler, which covers helium only."""
    assert atomic_number == 2
    log_comment(
        flog,
        ("adata", "transitiondata"),
        f"The levels and the transitions come from the tables in {path_for_log(datafolder)}.",
    )
    transitions = read_lines_data(atomic_number, ion_stage)

    ionization_energy_in_ev = read_ionization_data(atomic_number, ion_stage)

    energy_levels = read_levels_data(atomic_number, ion_stage)

    return ionization_energy_in_ev, energy_levels, transitions

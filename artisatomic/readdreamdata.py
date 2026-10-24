"""Read levels and transitions from the DREAM database of lanthanides and actinides.

DREAM is the Database on Rare Earths At Mons university: Quinet, P., Palmeri, P. (2020), Atoms, 8,
18, doi:10.3390/atoms8020018.
"""

from functools import cache

import numpy as np
import polars as pl

from artisatomic.base import add_handlers_if_not_set
from artisatomic.base import converted_parquet_path
from artisatomic.base import EnergyLevel
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import log_and_print
from artisatomic.base import log_comment
from artisatomic.base import nist_ionization_energy_comment
from artisatomic.base import path_in_data_folder
from artisatomic.base import PYDIR
from artisatomic.base import Transition

dreamfolder = PYDIR / ".." / "atomic-data-dream"
# the HDF5 file of the DREAM parser of Andreas Floers
dreamhdf5path = dreamfolder / "DREAM_atomic_data_20241106-1325.h5"
# artisatomic/hdf5_to_parquet.py writes this file from the HDF5 file
dreamdatapath = dreamhdf5path.with_suffix("") / "atomic_data.parquet"

# the "source:" line of the comment blocks in the output files (see Handler.description in iondata.py)
description = (
    "the DREAM database (Database on Rare Earths At Mons university) of Z >= 57. Quinet, P., Palmeri, P. (2020),"
    " Atoms, 8, 18, doi:10.3390/atoms8020018"
)


@cache
def read_dreamdata() -> pl.DataFrame:
    """Read the DREAM line list, once, on first use."""
    # only the columns that the reader uses, so a file that lacks one stops the run here
    columns = ["Z", "C", "Lower_Level", "Lower_Type", "Lower_J", "Upper_Level", "Upper_Type", "Upper_J", "gA"]
    return pl.read_parquet(converted_parquet_path(dreamdatapath, dreamhdf5path), columns=columns).with_columns(
        Lower_g=2 * pl.col("Lower_J") + 1, Upper_g=2 * pl.col("Upper_J") + 1
    )


def extend_ion_list(
    ion_handlers,
    *,
    minionstage: int | None = None,
    maxionstage: int | None = None,
    maxatomicnumber: int | None = None,
):
    """Add every ion in the DREAM line list to ion_handlers under the "dream" handler."""
    dreamions = [
        (atomic_number, charge + 1)
        for atomic_number, charge in read_dreamdata().select("Z", "C").unique(maintain_order=True).iter_rows()
    ]

    return add_handlers_if_not_set(
        ion_handlers,
        dreamions,
        "dream",
        minionstage=minionstage,
        maxionstage=maxionstage,
        maxatomicnumber=maxatomicnumber,
    )


def energytuple(energy, leveltype, g) -> EnergyLevel:
    """Build one DREAM level from its energy, its type "(o)" or "(e)", and its statistical weight.

    DREAM levels have no spectroscopic names, so the name of each level holds the energy, parity
    and statistical weight that identify it. level_frame() keys the levels on the same three
    values.
    """
    # not a default of 0 for any other text: the Laporte rule would get a parity the file did
    # not state
    if leveltype not in {"(o)", "(e)"}:
        msg = f"DREAM level type {leveltype!r} is not '(o)' or '(e)'"
        raise ValueError(msg)
    parity = 1 if leveltype == "(o)" else 0
    paritystr = "odd" if parity == 1 else "even"
    energyabovegsinpercm = float(energy)

    levelname = f"enpercm={energy},{paritystr},g={g}"
    return EnergyLevel(levelname=levelname, parity=parity, g=g, energyabovegsinpercm=energyabovegsinpercm)


def energytuplefromrow(row, prefix):
    """Build the lower or upper level of one DREAM line, selected by prefix ("Lower"/"Upper")."""
    return energytuple(row[prefix + "_Level"], row[prefix + "_Type"], row[prefix + "_g"])


# the columns of a line that identify one of its levels, without the "Lower_" or "Upper_" prefix
level_key_columns = ("Type", "Level", "g")


def level_frame(dflines: pl.DataFrame) -> pl.DataFrame:
    """Recover the level list from a DREAM line list, which has no separate level table.

    Each line carries both of its levels inline, so the levels are the distinct lower and upper
    levels over all lines, sorted by energy. The frame holds the key columns of each level, with
    the values of the line list, and its zero-based level id. read_levels_and_transitions() joins
    the lines to the frame on the key columns, so the join needs no Python loop over the lines.
    """
    keys = pl.concat(
        [
            dflines.select(pl.col(f"{prefix}_{column}").alias(column) for column in level_key_columns)
            for prefix in ("Lower", "Upper")
        ]
    )
    # keep="first" pins the order that decides the ids of two levels of one energy. The sort is
    # stable, so levels of one energy keep that order
    return (
        keys.unique(keep="first", maintain_order=True)
        .sort(pl.col("Level").cast(pl.Float64), maintain_order=True)
        .with_row_index("levelid")
        .with_columns(pl.col("levelid").cast(pl.Int64))
    )


def read_levels_data(dflevels: pl.DataFrame) -> list[EnergyLevel]:
    """Give the levels of a frame of level_frame() as tuples, in the order of the frame.

    The tuples validate the level type of every level, so a line with a different type stops the
    run here.
    """
    # with_row_index() puts the level id first
    return [energytuple(energy, leveltype, g) for _levelid, leveltype, energy, g in dflevels.iter_rows()]


def read_lines_data(dfiondata):
    """Convert DREAM lines to transitions referencing zero-based level ids."""
    transitions = []

    # numpy columns, not iterrows(): iterrows() builds a Series for each of the 10^5 lines of an ion.
    # read_levels_data() sorted the levels by energy, and transitiondata.txt has the lower id
    # first. So this code swaps a pair that the file lists in the reverse order.
    lowerindices = np.minimum(dfiondata["Lower_index"].to_numpy(), dfiondata["Upper_index"].to_numpy())
    upperindices = np.maximum(dfiondata["Lower_index"].to_numpy(), dfiondata["Upper_index"].to_numpy())
    # g_upper is the g of the level the file calls upper, because gA is that level's product
    A_values = dfiondata["gA"].to_numpy() / dfiondata["Upper_g"].to_numpy()

    for lowerindex, upperindex, A in zip(lowerindices.tolist(), upperindices.tolist(), A_values.tolist(), strict=True):
        transitions.append(Transition(lowerlevel=lowerindex, upperlevel=upperindex, A=A))

    return transitions


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the DREAM database of Z >= 57."""
    charge = ion_stage - 1
    dfiondata = read_dreamdata().filter((pl.col("Z") == atomic_number) & (pl.col("C") == charge))
    # not an assert: an ion that the database does not hold would give an empty level list and an
    # ion with no lines in the output
    if dfiondata.is_empty():
        msg = f"The DREAM database has no lines for Z={atomic_number} ion_stage {ion_stage}"
        raise ValueError(msg)
    # the HDF5 file and not the Parquet file: a user of the output files can get only the HDF5 file
    log_comment(
        flog,
        ("adata", "transitiondata"),
        f"The levels and the transitions come from {path_in_data_folder(dreamhdf5path, dreamfolder)}.",
    )

    dflevels = level_frame(dfiondata)
    energy_levels = read_levels_data(dflevels)

    # a join on the key columns, not a Python loop over the lines: an ion has 10^5 lines
    for prefix in ("Lower", "Upper"):
        dfiondata = dfiondata.join(
            dflevels.select(
                *(pl.col(column).alias(f"{prefix}_{column}") for column in level_key_columns),
                pl.col("levelid").alias(f"{prefix}_index"),
            ),
            on=[f"{prefix}_{column}" for column in level_key_columns],
            how="left",
            maintain_order="left",
        )
        # not an assert: level_frame() built the level list from the same frame as the lines. A
        # miss means they disagree, so the code must not map it silently
        if dfiondata[f"{prefix}_index"].null_count() > 0:
            msg = f"A DREAM line names a {prefix} level that is not in the level list"
            raise ValueError(msg)

    transitions = read_lines_data(dfiondata)

    # DREAM has no ionisation energies, so take them from NIST as the other handlers do
    ionization_energy_in_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]
    log_comment(flog, ("adata",), nist_ionization_energy_comment)
    log_and_print(flog, f"The NIST table gives an ionisation energy of {ionization_energy_in_ev} eV.")

    log_and_print(flog, f"The reader got {len(energy_levels):d} levels.")

    return ionization_energy_in_ev, energy_levels, transitions

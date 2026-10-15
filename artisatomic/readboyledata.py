"""Read helium levels and transitions from the Boyle AOIFE data set.

The data set belongs to the helium work of Boyle, A., Sim, S. A., Hachinger, S., Kerzendorf, W.
(2017), A&A, 599, A46, doi:10.1051/0004-6361/201629712.
"""

from functools import cache

import polars as pl

from artisatomic.base import check_integer_columns
from artisatomic.base import converted_parquet_path
from artisatomic.base import EnergyLevel
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import levelid_of_fileindex_map
from artisatomic.base import log_comment
from artisatomic.base import path_in_data_folder
from artisatomic.base import PYDIR
from artisatomic.base import resolve_transition_levelid_columns

boylefolder = PYDIR / ".." / "atomic-data-helium-boyle"
hdf5path = boylefolder / "aoife.hdf5"
# artisatomic/hdf5_to_parquet.py writes one Parquet file into this folder for each table of aoife.hdf5
datafolder = hdf5path.with_suffix("")

# the "source:" line of the comment blocks in the output files (see Handler.description in iondata.py)
description = (
    "the AOIFE helium data set. Boyle, A., Sim, S. A., Hachinger, S., Kerzendorf, W. (2017), A&A, 599, A46,"
    " doi:10.1051/0004-6361/201629712"
)


@cache
def read_aoife_table(tablename: str) -> pl.DataFrame:
    """Read the helium rows of one table of the AOIFE data set, once, on first use.

    The data set holds many elements, but the boyle handler reads only helium.
    """
    path = converted_parquet_path(datafolder / f"{tablename}.parquet", hdf5path)
    return pl.scan_parquet(path).filter(pl.col("atomic_number") == 2).collect()


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
    Levels have no spectroscopic names, so the name of each level holds its file index. The
    function also gives the map from the file index of each level to its level id.
    """
    sourcename = "the AOIFE levels_data table"
    dflevels = read_aoife_table("levels_data").filter(
        (pl.col("atomic_number") == atomic_number) & (pl.col("ion_number") == ion_stage - 1)
    )
    check_integer_columns(dflevels, ["level_number"], sourcename)

    energy_levels = [
        EnergyLevel(
            levelname=f"level{level_number:05d}",
            # the AOIFE energy column is in eV (He I 1s2s 3S is 19.8196). A division in Python and not
            # in polars: polars multiplies by the reciprocal, which changes the last bit of some energies
            energyabovegsinpercm=energy / hc_in_ev_cm,
            g=g,
            # No parity: this data set supplies none. add_level_ids_forbidden() marks a transition
            # forbidden when its two levels share one parity, and helium has many permitted ones. A
            # null parity never matches another, here as in the other readers whose data set has
            # no parities.
            parity=None,
        )
        for level_number, energy, g in dflevels.select("level_number", "energy", "g").iter_rows()
    ]

    return energy_levels, levelid_of_fileindex_map(dflevels["level_number"], sourcename)


def read_lines_data(atomic_number, ion_stage, levelid_of_fileindex):
    """Read one ion's bound-bound transitions from the AOIFE lines_data table, with zero-based level ids.

    The table gives the file indices of the levels as floats. The table gives no collision
    strengths.
    """
    sourcename = "the AOIFE lines_data table"
    levelcolumns = ["level_number_lower", "level_number_upper"]
    dflines = read_aoife_table("lines_data").filter(
        (pl.col("atomic_number") == atomic_number) & (pl.col("ion_number") == ion_stage - 1)
    )

    for column in levelcolumns:
        values = dflines[column]
        # not an assert: a cast to an integer would cut a file index such as 3.5 to 3 without a message
        if values.dtype.is_float() and not (values.is_finite() & (values == values.floor())).all():
            msg = f"{sourcename} gives a file index in the column {column} that is not a whole number"
            raise ValueError(msg)

    dflevelids = resolve_transition_levelid_columns(
        dflines.with_columns(pl.col(levelcolumns).cast(pl.Int64)),
        "level_number_lower",
        "level_number_upper",
        levelid_of_fileindex,
        sourcename,
    )
    return dflevelids.with_columns(A=dflines["A_ul"])


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion for the "boyle" handler, which covers helium only."""
    assert atomic_number == 2
    # the HDF5 file and not the Parquet files: a user of the output files can get only the HDF5 file
    log_comment(
        flog,
        ("adata", "transitiondata"),
        f"The levels and the transitions come from {path_in_data_folder(hdf5path, boylefolder)}.",
    )
    energy_levels, levelid_of_fileindex = read_levels_data(atomic_number, ion_stage)

    transitions = read_lines_data(atomic_number, ion_stage, levelid_of_fileindex)

    ionization_energy_in_ev = read_ionization_data(atomic_number, ion_stage)

    return ionization_energy_in_ev, energy_levels, transitions

"""Read ground states only, from the NIST ground-state table.

The table is an export of the NIST Atomic Spectra Database: Kramida, A., Ralchenko, Yu., Reader, J.
and NIST ASD Team, https://physics.nist.gov/asd, doi:10.18434/T4W30F.
"""

import typing as t
from functools import cache

import polars as pl

from artisatomic.base import add_handlers_if_not_set
from artisatomic.base import EnergyLevel
from artisatomic.base import log_and_print
from artisatomic.base import log_comment
from artisatomic.base import path_for_log
from artisatomic.base import PYDIR

datafilepath = PYDIR / ".." / "atomic-data-groundstatesonlynist" / "groundstates.dat"

# the "source:" line of the comment blocks in the output files (see Handler.description in iondata.py)
description = (
    "ground states only, from the NIST Atomic Spectra Database. Kramida, A., Ralchenko, Yu., Reader, J. and NIST ASD"
    " Team, https://physics.nist.gov/asd, doi:10.18434/T4W30F"
)


@cache
def read_groundstates_table() -> pl.DataFrame:
    """Read the whole NIST ground-state table once. Every ion reads its row from this frame."""
    return pl.read_csv(datafilepath, separator="\t")


def read_ground_levels(atomic_number, ion_stage, flog):
    """Read the ground state of one ion from the NIST ground-state table.

    This handler supplies a single level per ion and never any transitions. An ion that uses it
    contributes only its ground state and ionisation energy to the output.
    """
    # the transitiondata.txt block needs its source file also, although this handler gives no transition
    log_comment(flog, ("adata", "transitiondata"), f"Reading {path_for_log(datafilepath)}")
    groundstatesdata = read_groundstates_table()

    this_ion = groundstatesdata.filter(
        (pl.col("Z") == atomic_number) & (pl.col("ion") == ion_stage),
    )

    # not an assert: the bare IndexError from an empty selection names neither Z nor the stage
    if this_ion.is_empty():
        msg = f"groundstates.dat has no row for Z={atomic_number} ion_stage {ion_stage}"
        raise ValueError(msg)
    ionization_energy_in_ev = this_ion["IonizationEnergy"].item(0)
    log_and_print(flog, f"ionisation energy: {ionization_energy_in_ev} eV")
    energy_levels = [
        EnergyLevel(
            levelname=this_ion["config"].item(0),
            parity=0,
            g=this_ion["g"].item(0),
            energyabovegsinpercm=0.0,
        ),
    ]
    transitions: list[t.Any] = []

    return ionization_energy_in_ev, energy_levels, transitions


def extend_ion_list(
    ion_handlers,
    *,
    minionstage: int | None = None,
    maxionstage: int | None = None,
    maxatomicnumber: int | None = None,
):
    """Add every ion in the NIST ground-state table to ion_handlers under the "gsnist" handler."""
    groundstatesdata = read_groundstates_table()

    # add_handlers_if_not_set() returns a new list and does not change its argument
    return add_handlers_if_not_set(
        ion_handlers,
        groundstatesdata.select("Z", "ion").iter_rows(),
        "gsnist",
        minionstage=minionstage,
        maxionstage=maxionstage,
        maxatomicnumber=maxatomicnumber,
    )

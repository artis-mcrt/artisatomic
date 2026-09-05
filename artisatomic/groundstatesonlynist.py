"""Read ground states only, from the NIST ground-state table."""

import typing as t
from functools import cache

import pandas as pd

from artisatomic.base import add_handler_if_not_set
from artisatomic.base import EnergyLevel
from artisatomic.base import log_and_print
from artisatomic.base import PYDIR

datafilepath = PYDIR / ".." / "atomic-data-groundstatesonlynist" / "groundstates.dat"


@cache
def read_groundstates_table() -> pd.DataFrame:
    """Read the whole NIST ground-state table once. Every ion reads its row from this frame."""
    return pd.read_csv(datafilepath, delimiter="\t")


def read_ground_levels(atomic_number, ion_stage, flog):
    """Read the ground state of one ion from the NIST ground-state table.

    This handler supplies a single level per ion and never any transitions. An ion that uses it
    contributes only its ground state and ionisation energy to the output.
    """
    print(f"Reading NIST ground state data for Z={atomic_number} ion_stage {ion_stage} from groundstates.dat")
    groundstatesdata = read_groundstates_table()

    this_ion = groundstatesdata.loc[(groundstatesdata["Z"] == atomic_number) & (groundstatesdata["ion"] == ion_stage)]

    # not an assert: the bare IndexError from an empty selection names neither Z nor the stage
    if this_ion.empty:
        msg = f"groundstates.dat has no row for Z={atomic_number} ion_stage {ion_stage}"
        raise ValueError(msg)
    ionization_energy_in_ev = this_ion["IonizationEnergy"].to_numpy()[0]
    log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")
    energy_levels = [
        EnergyLevel(
            levelname=this_ion["config"].to_numpy()[0],
            parity=0,
            g=this_ion["g"].to_numpy()[0],
            energyabovegsinpercm=0.0,
        ),
    ]
    transitions: list[t.Any] = []  # this handler provides ground states only, so never any transitions

    return ionization_energy_in_ev, energy_levels, transitions


def extend_ion_list(ion_handlers):
    """Add every ion in the NIST ground-state table to ion_handlers under the "gsnist" handler."""
    groundstatesdata = read_groundstates_table()

    for _index, row in groundstatesdata.iterrows():
        # add_handler_if_not_set() returns a new list and does not change its argument. It also
        # normalises the pandas numpy integers to plain ints
        ion_handlers = add_handler_if_not_set(ion_handlers, row["Z"], row["ion"], "gsnist")

    return ion_handlers

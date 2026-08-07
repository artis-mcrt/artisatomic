"""Read ground states only, from the NIST ground-state table."""

import os.path
import typing as t
from collections import defaultdict
from pathlib import Path

import pandas as pd

import artisatomic


class EnergyLevel(t.NamedTuple):
    """A ground state read from the NIST table."""

    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: float


datafilepath = Path(
    os.path.dirname(os.path.abspath(__file__)), "..", "atomic-data-groundstatesonlynist", "groundstates.dat"
)


def read_ground_levels(atomic_number, ion_stage, flog):
    """Read the ground state of one ion from the NIST ground-state table.

    This handler supplies a single level per ion and never any transitions, so an ion using it
    contributes only its ground state and ionization energy to the output.
    """
    print(f"Reading NIST ground state data for Z={atomic_number} ion_stage {ion_stage} from groundstates.dat")
    groundstatesdata = pd.read_csv(datafilepath, delimiter="\t")

    this_ion = groundstatesdata.loc[(groundstatesdata["Z"] == atomic_number) & (groundstatesdata["ion"] == ion_stage)]

    ionization_energy_in_ev = this_ion["IonizationEnergy"].to_numpy()[0]
    artisatomic.log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")
    energy_levels = [
        EnergyLevel(
            levelname=this_ion["config"].to_numpy()[0],
            parity=0,
            g=this_ion["g"].to_numpy()[0],
            energyabovegsinpercm=0.0,
        ),
    ]
    transitions: list[t.Any] = []  # this handler provides ground states only, so never any transitions
    transition_count_of_level_name: defaultdict[str, int] = defaultdict(int)

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name


def extend_ion_list(ion_handlers):
    """Add every ion in the NIST ground-state table to ion_handlers under the "gsnist" handler."""
    groundstatesdata = pd.read_csv(datafilepath, delimiter="\t")

    for _index, row in groundstatesdata.iterrows():
        # add_handler_if_not_set() returns a new list rather than mutating its argument,
        # and normalises the pandas numpy integers to plain ints
        ion_handlers = artisatomic.add_handler_if_not_set(ion_handlers, row["Z"], row["ion"], "gsnist")

    return ion_handlers

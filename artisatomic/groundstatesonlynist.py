import os.path
import typing as t
from collections import defaultdict
from pathlib import Path

import pandas as pd

import artisatomic

hc_in_ev_cm = 0.0001239841984332003


class EnergyLevel(t.NamedTuple):
    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: float


datafilepath = Path(
    os.path.dirname(os.path.abspath(__file__)), "..", "atomic-data-groundstatesonlynist", "groundstates.dat"
)


def read_ground_levels(atomic_number, ion_stage, flog):
    print(f"Reading NIST ground state data for Z={atomic_number} ion_stage {ion_stage} from groundstates.dat")
    groundstatesdata = pd.read_csv(datafilepath, delimiter="\t")

    this_ion = groundstatesdata.loc[(groundstatesdata["Z"] == atomic_number) & (groundstatesdata["ion"] == ion_stage)]

    ionization_energy_in_ev = this_ion["IonizationEnergy"].to_numpy()[0]
    artisatomic.log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")
    energy_levels = [
        None,
        EnergyLevel(
            levelname=this_ion["config"].to_numpy()[0],
            parity=0,
            g=this_ion["g"].to_numpy()[0],
            energyabovegsinpercm=0.0,
        ),
    ]
    transitions = []
    transition_count_of_level_name = defaultdict(int)

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name


def extend_ion_list(ion_handlers):
    groundstatesdata = pd.read_csv(datafilepath, delimiter="\t")

    for _index, row in groundstatesdata.iterrows():
        # int() because pandas hands back numpy integers, which json.dump() cannot serialise
        # when main() writes artisatomicionhandlers.json
        atomic_number = int(row["Z"])
        ion_stage = int(row["ion"])
        # add_handler_if_not_set() returns a new list rather than mutating its argument
        ion_handlers = artisatomic.add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "gsnist")

    return ion_handlers

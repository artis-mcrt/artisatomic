import os.path
from pathlib import Path
import typing as t
import pandas as pd
from astropy import constants as const
from collections import defaultdict

import artisatomic

hc_in_ev_cm = (const.h * const.c).to("eV cm").value

class EnergyLevel(t.NamedTuple):
    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: float


datafilepath = Path(os.path.dirname(os.path.abspath(__file__)), "..", "atomic-data-groundstatesonlynist", "groundstates.dat")


def read_ground_levels(atomic_number, ion_stage, flog):
    print(f"Reading NIST ground state data for Z={atomic_number} ion_stage {ion_stage} from groundstates.dat")
    groundstatesdata = pd.read_csv(datafilepath, delimiter="\t")

    this_ion = groundstatesdata.loc[(groundstatesdata["Z"] == atomic_number) & (groundstatesdata["ion"] == ion_stage)]

    ionization_energy_in_ev = this_ion["IonizationEnergy"].to_numpy()[0]
    artisatomic.log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")
    energy_levels = [None]
    energy_levels.append(EnergyLevel(levelname=this_ion["config"].to_numpy()[0], parity=0, g=this_ion["g"].to_numpy()[0],
                                 energyabovegsinpercm=0.0))

    transitions = []
    transition_count_of_level_name = defaultdict(int)

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name


def extend_ion_list(ion_handlers):
    groundstatesdata = pd.read_csv(datafilepath, delimiter="\t")

    for index, row in groundstatesdata.iterrows():
        atomic_number = row["Z"]
        ion_stage = row["ion"]
        found_element = False
        for tmp_atomic_number, list_ions_handlers in ion_handlers:
            if tmp_atomic_number == atomic_number:
                # add an ion that is not present in the element's list
                if ion_stage not in [x[0] if hasattr(x, "__getitem__") else x for x in list_ions_handlers]:
                    list_ions_handlers.append((ion_stage, "gsnist"))
                    list_ions_handlers.sort(key=lambda x: x[0] if hasattr(x, "__getitem__") else x)
                found_element = True

        if not found_element:
            ion_handlers.append(
                (atomic_number,
                [(ion_stage, "gsnist")],)
            )
    ion_handlers.sort(key=lambda x: x[0])
    return ion_handlers

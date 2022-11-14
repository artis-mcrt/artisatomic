import os
from collections import defaultdict
from collections import namedtuple
from pathlib import Path

from astropy import constants as const

import artisatomic
from artisatomic.cFACReader import GetLevels
from artisatomic.cFACReader import GetLines

# from astropy import units as u
# import os.path
# import pandas as pd
# from carsus.util import parse_selected_species


hc_in_ev_cm = (const.h * const.c).to("eV cm").value
BASEPATH = "/Volumes/GoogleDrive/Shared drives/Atomic Data Group/opacities/SystematicCalculations"


def extend_ion_list(listelements):
    for s in Path(BASEPATH).glob("**/*.lev.asc"):
        ionstr = s.parts[-1].removesuffix(".lev.asc")
        elsym = s.parts[-3]
        ion_stage_roman = ionstr.removeprefix(elsym)
        atomic_number = artisatomic.elsymbols.index(elsym)

        ion_stage = artisatomic.roman_numerals.index(ion_stage_roman)

        if atomic_number <= 30 or ion_stage != 2:  # TODO: remove??
            continue

        found_element = False
        for tmp_atomic_number, list_ions in listelements:
            if tmp_atomic_number == atomic_number:
                if ion_stage not in [x[0] if len(x) > 0 else x for x in list_ions]:
                    list_ions.append((ion_stage, "fac"))
                    list_ions.sort()
                found_element = True
        if not found_element:
            listelements.append(
                (
                    atomic_number,
                    [(ion_stage, "fac")],
                )
            )

    listelements.sort(key=lambda x: x[0])
    # print(listelements)
    return listelements


def read_levels_data(dflevels):
    energy_level_tuple = namedtuple("energylevel", "levelname energyabovegsinpercm g parity")

    energy_levels = []

    for index, row in dflevels.iterrows():
        parity = row["P"]
        energyabovegsinpercm = float(row["Energy[cm^-1]"])
        g = row["g"]

        newlevel = energy_level_tuple(
            levelname=row["Config rel"], parity=parity, g=g, energyabovegsinpercm=energyabovegsinpercm
        )
        energy_levels.append(newlevel)

    energy_levels.sort(key=lambda x: x.energyabovegsinpercm)

    return ["IGNORE"] + energy_levels


def read_lines_data(energy_levels, dflines):
    transitions = []
    transition_count_of_level_name = defaultdict(int)
    transitiontuple = namedtuple("transition", "lowerlevel upperlevel A coll_str")

    for index, row in dflines.iterrows():
        lowerlevel = int(row["Lower"]) + 1
        upperlevel = int(row["Upper"]) + 1
        A = row["TR_rate[1/s]"]

        transtuple = transitiontuple(lowerlevel=lowerlevel, upperlevel=upperlevel, A=A, coll_str=-1)

        transition_count_of_level_name[energy_levels[lowerlevel].levelname] += 1
        transition_count_of_level_name[energy_levels[upperlevel].levelname] += 1

        transitions.append(transtuple)

    return transitions, transition_count_of_level_name


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    # ion_charge = ion_stage - 1
    elsym = artisatomic.elsymbols[atomic_number]
    ion_stage_roman = artisatomic.roman_numerals[ion_stage]

    ion_folder = BASEPATH + f"/{elsym}/{elsym}{ion_stage_roman}"
    levels_file = ion_folder + f"/{elsym}{ion_stage_roman}.lev.asc"
    lines_file = ion_folder + f"/{elsym}{ion_stage_roman}.tr.asc"

    print(f"Reading FAC data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman}) from {ion_folder}")

    dflevels = GetLevels(filename=levels_file, Z=atomic_number, Get_csv=False, Get_dat=False)
    # print(dflevels)

    energy_levels = read_levels_data(dflevels)

    dflines = GetLines(filename=lines_file, Z=atomic_number, Get_csv=False, Get_dat=False)
    # print(dflines)

    transitions, transition_count_of_level_name = read_lines_data(energy_levels, dflines)

    ionization_energy_in_ev = -1

    artisatomic.log_and_print(flog, f"Read {len(energy_levels[1:]):d} levels and {len(transitions)} transitions")

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name

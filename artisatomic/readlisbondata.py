import os.path
from collections import defaultdict
from collections import namedtuple
from pathlib import Path

import pandas as pd
from astropy import constants as const

import artisatomic

# from astropy import units as u


hc_in_ev_cm = (const.h * const.c).to("eV cm").value


class LisbonReader:
    """Copied from Andreas Floers code in git.gsi.de:nucastro/opacities.git
    Class for extracting levels and lines from the Lisbon Atomic Group.

    Mimics the GFALLReader class.

    Attributes
    ----------
    levels : DataFrame
    lines : DataFrame

    """

    def __init__(self, data, priority=10):
        """Parameters
        ----------
        data : dict
            Dictionary containing one dictionary per species with
            keys `levels` and `lines`.

        priority: int, optional
            Priority of the current data source, by default 10.
        """
        self.priority = priority
        self._get_levels_lines(data)

    def _get_levels_lines(self, data):
        """Generates `levels` and `lines` DataFrames.

        Parameters
        ----------
        data : dict
            Dictionary containing one dictionary per species with
            keys `levels` and `lines`.
        """
        from carsus.util import parse_selected_species

        lvl_list = []
        lns_list = []
        for ion, parser in data.items():
            atomic_number = parse_selected_species(ion)[0][0]
            ion_charge = parse_selected_species(ion)[0][1]
            levels_data = pd.read_csv(parser["levels"], skiprows=8, index_col=0)
            levels = pd.DataFrame()
            levels["energy"] = levels_data["Energy[cm^-1]"]
            levels["j"] = 0.5 * (levels_data["g"] - 1)
            levels["label"] = levels_data["RelConfig"]
            levels["method"] = "meas"
            levels["priority"] = self.priority
            levels["atomic_number"] = atomic_number
            levels["ion_charge"] = ion_charge
            levels["level_index"] = levels.index
            levels = levels.set_index(["atomic_number", "ion_charge", "level_index"])
            lvl_list.append(levels)

            lines_data = pd.read_csv(parser["lines"], skiprows=8)  # index_col=0
            lines = pd.DataFrame()
            lines["level_index_lower"] = lines_data["Lower"]
            lines["level_index_upper"] = lines_data["Upper"]
            lines["atomic_number"] = atomic_number
            lines["ion_charge"] = ion_charge
            lines["energy_lower"] = levels.iloc[lines["level_index_lower"]]["energy"].values
            # for r in lines.iterrows():
            #     print(r)
            #     print(levels.iloc[r['level_index_upper']]['energy'])
            lines["energy_upper"] = levels.iloc[lines["level_index_upper"]]["energy"].values
            lines["gf"] = lines_data["gf"]
            lines["j_lower"] = levels.iloc[lines["level_index_lower"]]["j"].values
            lines["j_upper"] = levels.iloc[lines["level_index_upper"]]["j"].values
            lines["wavelength"] = lines_data["Wavelength[Ang]"] / 10.0
            lines = lines.set_index(["atomic_number", "ion_charge", "level_index_lower", "level_index_upper"])
            lns_list.append(lines)
        levels = pd.concat(lvl_list)
        lines = pd.concat(lns_list)
        self.levels = levels
        self.lines = lines


# def extend_ion_list(ion_handlers):
#     selected_ions = [(38, 1)]
#     for atomic_number, charge in selected_ions:
#         assert (atomic_number, charge) in gfall_reader.ions
#         ion_stage = charge + 1
#
#         found_element = False
#         for (tmp_atomic_number, list_ions) in ion_handlers:
#             if tmp_atomic_number == atomic_number:
#                 if ion_stage not in list_ions:
#                     list_ions.append((ion_stage, 'carsus'))
#                     list_ions.sort()
#                 found_element = True
#
#         if not found_element:
#             ion_handlers.append((atomic_number, [(ion_stage, 'carsus')],))
#
#     ion_handlers.sort(key=lambda x: x[0])
#
#     return ion_handlers


def get_levelname(row):
    return f"{row.label}, j={row.j}"


def read_levels_data(dflevels):
    energy_level_tuple = namedtuple("energylevel", "levelname energyabovegsinpercm g parity")

    energy_levels = []

    for index, row in dflevels.iterrows():
        parity = -index  # give a unique parity so that all transitions are permitted
        energyabovegsinpercm = float(row.energy)
        g = 2 * row.j + 1
        newlevel = energy_level_tuple(
            levelname=get_levelname(row), parity=parity, g=g, energyabovegsinpercm=energyabovegsinpercm
        )
        energy_levels.append(newlevel)

    energy_levels.sort(key=lambda x: x.energyabovegsinpercm)

    return [None, *energy_levels]


def read_lines_data(energy_levels, dflines):
    transitions = []
    transition_count_of_level_name = defaultdict(int)
    transitiontuple = namedtuple("transition", "lowerlevel upperlevel A coll_str")

    for (lowerindex, upperindex), row in dflines.iterrows():
        lowerlevel = lowerindex + 1
        upperlevel = upperindex + 1

        transtuple = transitiontuple(lowerlevel=lowerlevel, upperlevel=upperlevel, A=row.A, coll_str=-1)

        # print(line)
        transition_count_of_level_name[energy_levels[lowerlevel].levelname] += 1
        transition_count_of_level_name[energy_levels[upperlevel].levelname] += 1

        transitions.append(transtuple)

    return transitions, transition_count_of_level_name


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    ion_charge = ion_stage - 1
    elsym = artisatomic.elsymbols[atomic_number]
    ion_stage_roman = artisatomic.roman_numerals[ion_stage]

    assert elsym in ["Nd", "U"]
    assert ion_stage in [2, 3]

    print(f"Reading Lisbon data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman})")

    LISPATH = "/Users/luke/Dropbox/GitHub/opacities/SystematicCalculations"

    # nd_2_lvl = LISPATH + '/Nd/NdIII/NdIII_Levels.csv'
    # nd_2_lns = LISPATH + '/Nd/NdIII/NdIII_Transitions.csv'
    # u_2_lvl = LISPATH + '/U/UIII/UIII_Levels.csv'
    # u_2_lns = LISPATH + '/U/UIII/UIII_Transitions.csv'
    # lisbon_data = {'Nd 2': {'levels': nd_2_lvl, 'lines': nd_2_lns},
    #                'U 2': {'levels': u_2_lvl, 'lines': u_2_lns}}

    lisbon_data = {
        f"{elsym} {ion_charge}": {
            "levels": LISPATH + f"/{elsym}/{elsym}{ion_stage_roman}/{elsym}{ion_stage_roman}_Levels.csv",
            "lines": LISPATH + f"/{elsym}/{elsym}{ion_stage_roman}/{elsym}{ion_stage_roman}_Transitions.csv",
        }
    }

    lisbon_reader = LisbonReader(lisbon_data)

    dflevels = lisbon_reader.levels.loc[atomic_number, ion_charge]
    energy_levels = read_levels_data(dflevels)

    # for x in energy_levels:
    #     print(x)
    dflines = lisbon_reader.lines.loc[atomic_number, ion_charge]
    dflines.eval("A = gf / (1.49919e-16 * (2 * j_upper + 1) * wavelength ** 2)", inplace=True)

    # print(dflines)

    transitions, transition_count_of_level_name = read_lines_data(energy_levels, dflines)

    ionization_energy_in_ev = -1

    artisatomic.log_and_print(flog, f"Read {len(energy_levels[1:]):d} levels")

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name

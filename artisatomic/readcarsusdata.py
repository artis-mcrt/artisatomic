from carsus.io.nist import NISTWeightsComp, NISTIonizationEnergies
from carsus.io.kurucz import GFALLReader
# from carsus.io.zeta import KnoxLongZeta
# from carsus.io.chianti_ import ChiantiReader
# from carsus.io.output import TARDISAtomData

from collections import defaultdict, namedtuple
from astropy import constants as const
import artisatomic
# from astropy import units as u
import os.path
from pathlib import Path
import pandas as pd


hc_in_ev_cm = (const.h * const.c).to('eV cm').value

# ionization_energies = NISTIonizationEnergies('Sr')
gfall_reader = None


# def extend_ion_list(listelements):
#     selected_ions = [(38, 1)]
#     for atomic_number, charge in selected_ions:
#         assert (atomic_number, charge) in gfall_reader.ions
#         ion_stage = charge + 1
#
#         found_element = False
#         for (tmp_atomic_number, list_ions) in listelements:
#             if tmp_atomic_number == atomic_number:
#                 if ion_stage not in list_ions:
#                     list_ions.append((ion_stage, 'carsus'))
#                     list_ions.sort()
#                 found_element = True
#
#         if not found_element:
#             listelements.append((atomic_number, [(ion_stage, 'carsus')],))
#
#     listelements.sort(key=lambda x: x[0])
#
#     return listelements

def get_levelname(row):
    return f'{row.label},enpercm={row.energy},j={row.j}'


def read_levels_data(dflevels):
    energy_level_tuple = namedtuple(
        'energylevel', 'levelname energyabovegsinpercm g parity')

    energy_levels = []

    for index, row in dflevels.iterrows():

        parity = - index  # give a unique parity so that all transitions are permitted
        energyabovegsinpercm = float(row.energy)
        g = 2 * row.j + 1
        newlevel = energy_level_tuple(
            levelname=get_levelname(row), parity=parity, g=g, energyabovegsinpercm=energyabovegsinpercm)
        energy_levels.append(newlevel)

    energy_levels.sort(key=lambda x: x.energyabovegsinpercm)

    return ['IGNORE'] + energy_levels


def read_lines_data(energy_levels, dflines):
    transitions = []
    transition_count_of_level_name = defaultdict(int)
    transitiontuple = namedtuple('transition', 'lowerlevel upperlevel A coll_str')

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

    print(f'Reading CARSUS database for Z={atomic_number} ion_stage {ion_stage}')

    path_gfall = str((Path(__file__).parent.absolute() / '..' / 'atomic-data-kurucz' / 'gfall_latest.dat').resolve())
    gfall_reader = GFALLReader(ions=f'{artisatomic.elsymbols[atomic_number]} {ion_charge}', fname=path_gfall)

    dflevels = gfall_reader.extract_levels().loc[atomic_number, ion_charge]

    # print(dflevels)

    # from nistasd import NISTLines
    #
    # nist = NISTLines(spectrum='Ce')
    # energy_levels = nist.get_energy_level_data()
    # print("energy_levels.keys() = ", energy_levels['Ce I'])
    # for ion_stage in energy_levels:
    #     print("Number of energy levels: {0} for {1}".format(len(energy_levels[ion_stage]), ion_stage))
    #     df = pd.DataFrame(energy_levels[ion_stage])
    #     print(df)

    energy_levels = read_levels_data(dflevels)

    # for x in energy_levels:
    #     print(x)

    dflines = gfall_reader.extract_lines().loc[atomic_number, ion_charge]

    dflines.eval('A = gf / (1.49919e-16 * (2 * j_upper + 1) * wavelength ** 2)   ', inplace=True)

    # print(dflines)

    transitions, transition_count_of_level_name = read_lines_data(
        energy_levels, dflines)

    # ionization_energy_in_ev = read_ionization_data(atomic_number, ion_stage)
    ionization_energy_in_ev = -1

    # artisatomic.log_and_print(flog, f'Read {len(energy_levels[1:]):d} levels')

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name

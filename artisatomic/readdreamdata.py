import h5py
from collections import defaultdict, namedtuple
from astropy import constants as const
import artisatomic
# from astropy import units as u
import os.path
from pathlib import Path
import pandas as pd

dreamdatapath = Path(os.path.dirname(os.path.abspath(__file__)), "..",
                     "atomic-data-dream", "DREAM_atomic_data_20210217-1633.h5")
dreamdata = None
hc_in_ev_cm = (const.h * const.c).to('eV cm').value


def init_dreamdata():
    global dreamdata
    if dreamdata is None:
        dreamdatapath = Path(os.path.dirname(os.path.abspath(__file__)), "..",
                             "atomic-data-dream", "DREAM_atomic_data_20210217-1633.h5")
        dreamdata = pd.read_hdf(dreamdatapath) if dreamdatapath.exists() else None


def extend_ion_list(listelements):
    init_dreamdata()

    for atomic_number, charge in dreamdata.index.unique():
        ion_stage = charge + 1

        found_element = False
        for (tmp_atomic_number, list_ions) in listelements:
            if tmp_atomic_number == atomic_number:
                if ion_stage not in list_ions:
                    list_ions.append((ion_stage, 'dream'))
                    list_ions.sort()
                found_element = True

        if not found_element:
            listelements.append((atomic_number, [(ion_stage, 'dream')],))

    listelements.sort(key=lambda x: x[0])

    return listelements


def energytuplefromrow(row, prefix):
    energy, type, g = row[prefix + '_Level'], row[prefix + '_Type'], row[prefix + '_g']
    dream_energy_level_row = namedtuple(
        'energylevel', 'levelname energyabovegsinpercm g parity')

    parity = 1 if type == '(o)' else 0
    paritystr = "odd" if parity == 1 else "even"
    energyabovegsinpercm = float(energy)

    levelname = f"enpercm={energy},{paritystr},g={g}"
    newlevel = dream_energy_level_row(levelname=levelname, parity=parity, g=g,
                                      energyabovegsinpercm=energyabovegsinpercm)
    return newlevel


def read_levels_data(dflines):
    energy_levels = []

    for prefix in ['Lower', 'Upper']:
        for index, row in dflines.drop_duplicates(
                subset=[prefix + '_Type', prefix + '_Level', prefix + '_g']).iterrows():

            leveltuple = energytuplefromrow(row, prefix)
            if leveltuple not in energy_levels:
                energy_levels.append(leveltuple)

    energy_levels.sort(key=lambda x: x.energyabovegsinpercm)

    return ['IGNORE'] + energy_levels


def read_lines_data(atomic_number, ion_stage, dfiondata, energy_levels):
    transitions = []
    transition_count_of_level_name = defaultdict(int)
    transitiontuple = namedtuple('transition', 'lowerlevel upperlevel A coll_str')

    for index, row in dfiondata.iterrows():
        coll_str = -1  # TODO
        lowerindex = row['Lower_index']
        upperindex = row['Upper_index']
        A = row['gA'] / row['Upper_g']  # TODO: is this correct?
        transtuple = transitiontuple(lowerlevel=lowerindex, upperlevel=upperindex, A=A, coll_str=coll_str)

        # print(line)
        transition_count_of_level_name[energy_levels[lowerindex].levelname] += 1
        transition_count_of_level_name[energy_levels[upperindex].levelname] += 1

        transitions.append(transtuple)

    return transitions, transition_count_of_level_name


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    init_dreamdata()
    charge = ion_stage - 1
    dfiondata = dreamdata.loc[(atomic_number, charge)]
    print(f'Reading DREAM database for Z={atomic_number} ion_stage {ion_stage}')
    print(dfiondata)

    # from nistasd import NISTLines
    #
    # nist = NISTLines(spectrum='Ce')
    # energy_levels = nist.get_energy_level_data()
    # print("energy_levels.keys() = ", energy_levels['Ce I'])
    # for ion_stage in energy_levels:
    #     print("Number of energy levels: {0} for {1}".format(len(energy_levels[ion_stage]), ion_stage))
    #     df = pd.DataFrame(energy_levels[ion_stage])
    #     print(df)

    energy_levels = read_levels_data(dfiondata)

    # for x in energy_levels:
    #     print(x)
    def get_level_index(row, prefix):
        leveltuple = energytuplefromrow(row, prefix)
        if leveltuple in energy_levels:
            return energy_levels.index(leveltuple)
        assert False
        return -1

    dfiondata.insert(2, "Lower_index",
                     dfiondata.apply(lambda row: get_level_index(row, prefix='Lower'), axis=1).values, allow_duplicates=True)

    dfiondata.insert(2, "Upper_index",
                     dfiondata.apply(lambda row: get_level_index(row, prefix='Upper'), axis=1).values, allow_duplicates=True)

    transitions, transition_count_of_level_name = read_lines_data(atomic_number, ion_stage, dfiondata, energy_levels)

    # ionization_energy_in_ev = read_ionization_data(atomic_number, ion_stage)
    ionization_energy_in_ev = -1

    # artisatomic.log_and_print(flog, f'Read {len(energy_levels[1:]):d} levels')

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name

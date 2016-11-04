#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
from collections import namedtuple, defaultdict
from astropy import constants as const
from astropy import units as u
import makeartisatomicfiles as artisatomic
from manual_matches import hillier_name_replacements
ryd_to_ev = u.rydberg.to('eV')

hc_in_ev_cm = (const.h * const.c).to('eV cm').value
hc_in_ev_angstrom = (const.h * const.c).to('eV angstrom').value
h_in_ev_seconds = const.h.to('eV s').value
lchars = 'SPDFGHIKLMNOPQRSTUVWXYZ'
PYDIR = os.path.dirname(os.path.abspath(__file__))
elsymbols = ['n'] + list(pd.read_csv(os.path.join(PYDIR, 'elements.csv'))['symbol'].values)


# need to also include collision strengths from e.g., o2col.dat

# keys are (atomic number, ion stage)
path_hillier_osc_file = {
    (8, 1): '20sep11/oi_osc_mchf',
    (8, 2): '23mar05/o2osc_fin.dat',
    (8, 3): '15mar08/oiiiosc',
    (26, 1): '29apr04/fei_osc',
    (26, 2): '16nov98/fe2osc_nahar_kurucz.dat',
    (26, 3): '30oct12/FeIII_OSC',
    (26, 4): '18oct00/feiv_osc_rev2.dat',
    (26, 5): '18oct00/fev_osc.dat',
    (27, 2): '15nov11/fin_osc_bound',
}

hillier_row_format_energy_level = {
    (8, 1): 'levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad c4 c6',
    (8, 2): 'levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4',
    (8, 3): 'levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4',
    (26, 1): 'levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4',
    (26, 2): 'levelname g energyabovegsinpercm freqtentothe15hz lambdaangstrom hillierlevelid',
    (26, 3): 'levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad c4 c6',
    (26, 4): 'levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4',
    (26, 5): 'levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4',
    (27, 2): 'levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4',
}

elsymboltohilliercode = {
    'H': 'HYD', 'He': 'HE', 'C': 'CARB', 'N': 'NIT',
    'O': 'OXY', 'F': 'FLU', 'Ne': 'NEON', 'Na': 'SOD',
    'Mg': 'MG', 'Al': 'ALUM', 'Si': 'SIL', 'P': 'PHOS',
    'S': 'SUL', 'Cl': 'CHL', 'Ar': 'ARG', 'K': 'POT',
    'Ca': 'CAL', 'Sc': 'SCAN', 'Ti': 'TIT', 'V': 'VAN',
    'Cr': 'CHRO', 'Mn': 'MAN', 'Fe': 'FE', 'Co': 'COB',
    'Ni': 'NICK'
}


# hilliercodetoelsymbol = {v : k for (k,v) in elsymboltohilliercode.items()}
# hilliercodetoatomic_number = {k : elsymbols.index(v) for (k,v) in hilliercodetoelsymbol.items()}

atomic_number_to_hillier_code = {elsymbols.index(k): v for (k, v) in elsymboltohilliercode.items()}


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    row_format_energy_level = hillier_row_format_energy_level[(atomic_number, ion_stage)]
    hillier_ion_folder = ('atomic-data-hillier/atomic/' + atomic_number_to_hillier_code[atomic_number] +
                          '/' + artisatomic.roman_numerals[ion_stage] + '/')
    osc_file = path_hillier_osc_file[(atomic_number, ion_stage)]
    filename = os.path.join(hillier_ion_folder, osc_file)
    print('Reading ' + filename)
    hillier_energy_level_row = namedtuple(
        'energylevel', row_format_energy_level + ' corestateid twosplusone l parity indexinsymmetry naharconfiguration matchscore')
    hillier_transition_row = namedtuple(
        'transition', 'namefrom nameto f A lambdaangstrom i j hilliertransitionid lowerlevel upperlevel coll_str')
    hillier_energy_levels = ['IGNORE']
    hillier_level_ids_matching_term = defaultdict(list)
    transition_count_of_level_name = defaultdict(int)
    hillier_ionization_energy_ev = 0.0
    transitions = []

    if os.path.isfile(filename):
        with open(filename, 'r') as fhillierosc:
            for line in fhillierosc:
                row = line.split()

                # check for right number of columns and that are all numbers except first column
                if len(row) == len(row_format_energy_level.split()) and all(map(artisatomic.isfloat, row[1:])):
                    hillier_energy_level = hillier_energy_level_row(*row, 0, -1, -1, -1, -1, '', -1)

                    hillierlevelid = int(hillier_energy_level.hillierlevelid.lstrip('-'))
                    levelname = hillier_energy_level.levelname
                    if levelname not in hillier_name_replacements:
                        (twosplusone, l, parity) = artisatomic.get_term_as_tuple(levelname)
                    else:
                        (twosplusone, l, parity) = artisatomic.get_term_as_tuple(hillier_name_replacements[levelname])

                    hillier_energy_level = hillier_energy_level._replace(
                        hillierlevelid=hillierlevelid,
                        energyabovegsinpercm=float(hillier_energy_level.energyabovegsinpercm),
                        g=float(hillier_energy_level.g),
                        twosplusone=twosplusone,
                        l=l,
                        parity=parity
                    )

                    hillier_energy_levels.append(hillier_energy_level)

                    if twosplusone == -1:
                        # -1 indicates that the term could not be interpreted
                        if parity == -1:
                            artisatomic.log_and_print(flog, "Can't find LS term in Hillier level name '" + levelname + "'")
                        # else:
                            # artisatomic.log_and_print(flog, "Can't find LS term in Hillier level name '{0:}' (parity is {1:})".format(levelname, parity))
                    else:
                        if levelname not in hillier_level_ids_matching_term[(twosplusone, l, parity)]:
                            hillier_level_ids_matching_term[(twosplusone, l, parity)].append(hillierlevelid)

                    # if this is the ground state
                    if float(hillier_energy_levels[-1].energyabovegsinpercm) < 1.0:
                        hillier_ionization_energy_ev = hc_in_ev_angstrom / \
                            float(hillier_energy_levels[-1].lambdaangstrom)

                    if hillierlevelid != len(hillier_energy_levels) - 1:
                        print('Hillier levels mismatch: id {0:d} found at entry number {1:d}'.format(
                            len(hillier_energy_levels) - 1, hillierlevelid))
                        sys.exit()

                if line.startswith('                        Oscillator strengths'):
                    break

            # defined_transition_ids = []
            for line in fhillierosc:
                if line.startswith('                        Oscillator strengths'):
                    break
                linesplitdash = line.split('-')
                row = (linesplitdash[0] + ' ' + '-'.join(linesplitdash[1:-1]) +
                       ' ' + linesplitdash[-1]).split()

                if len(row) == 8 and all(map(artisatomic.isfloat, row[2:4])):
                    transition = hillier_transition_row(row[0], row[1],
                                                        float(row[2]),  # f
                                                        float(row[3]),  # A
                                                        float(row[4]),  # lambda
                                                        int(row[5]),  # i
                                                        int(row[6]),  # j
                                                        int(row[7]),  # hilliertransitionid
                                                        -1,  # lowerlevel
                                                        -1,  # upperlevel
                                                        -99)  # coll_str

                    if True:  # or int(transition.hilliertransitionid) not in defined_transition_ids: #checking for duplicates massively slows down the code
                        #                    defined_transition_ids.append(int(transition.hilliertransitionid))
                        transitions.append(transition)
                        transition_count_of_level_name[transition.namefrom] += 1
                        transition_count_of_level_name[transition.nameto] += 1

                        if int(transition.hilliertransitionid) != len(transitions):
                            print(path_hillier_osc_file + ', WARNING: Transition id {0:d} found at entry number {1:d}'.format(
                                int(transition.hilliertransitionid), len(transitions)))
                            sys.exit()
                    else:
                        artisatomic.log_and_print(flog, 'FATAL: multiply-defined Hillier transition: {0} {1}'.format(
                            transition.namefrom, transition.nameto))
                        sys.exit()
        artisatomic.log_and_print(flog, 'Read {:d} transitions'.format(len(transitions)))

    return hillier_ionization_energy_ev, hillier_energy_levels, transitions, transition_count_of_level_name, hillier_level_ids_matching_term


# this is out of date, so make sure this produces valid output before using
def read_hillier_phixs_tables(hillier_ion_folder, photoion_xs_file, atomic_number, ion_stage):
    photoionization_crosssections = []
    with open(hillier_ion_folder + photoion_xs_file, 'r') as fhillierphot:
        upperlevelid = -1
        truncatedlowerlevelname = ''
        numpointsexpected = 0
        crosssectiontype = '-1'
        seatonfittingcoefficients = []

        for line in fhillierphot:
            row = line.split()
    #            print(row)

            if len(row) >= 2 and ' '.join(row[1:]) == '!Final state in ion':
                # upperlevelid = level_ids_of_energy_level_name_no_brackets[row[0]][0]
                upperlevelid = -1

            if len(row) >= 2 and ' '.join(row[1:]) == '!Configuration name':
                truncatedlowerlevelname = row[0]
                for lowerlevelid in level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname]:
                    photoionization_crosssections[lowerlevelid] = []
                seatonfittingcoefficients = []
                numpointsexpected = 0

            if len(row) >= 2 and ' '.join(row[1:]) == '!Number of cross-section points':
                numpointsexpected = int(row[0])

            if len(row) >= 2 and ' '.join(row[1:]) == '!Cross-section unit' and row[0] != 'Megabarns':
                    print('Wrong cross-section unit: ' + row[0])
                    sys.exit()

            if crosssectiontype in ['20', '21'] and len(row) == 2 and all(map(isfloat, row)) and lowerlevelid != -1:
                for lowerlevelid in level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname]:
                    # WOULD NEED TO FIX THIS LINE SO THAT ENERGY IS IN ABSOLUTE UNITS OF RYDBERG FOR THIS TO WORK
                    photoionization_crosssections[lowerlevelid].append((float(row[0].replace('D', 'E')), float(row[1].replace('D', 'E'))))
                    if (len(photoionization_crosssections[lowerlevelid]) > 1 and
                       photoionization_crosssections[lowerlevelid][-1][0] <= photoionization_crosssections[lowerlevelid][-2][0]):
                        print('ERROR: photoionization table first column not monotonically increasing')
                        sys.exit()
            # elif True: #if want to ignore the types below
                # pass
            elif crosssectiontype == '1' and len(row) == 1 and artisatomic.isfloat(row[0]) and numpointsexpected > 0:
                seatonfittingcoefficients.append(float(row[0].replace('D', 'E')))
                if len(seatonfittingcoefficients) == 3:
                    (sigmat, beta, s) = seatonfittingcoefficients
                    for lowerlevelid in level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname]:
                        for c in np.arange(0, 1.0, 0.01):
                            energydivthreshold = 1 + 20 * (c ** 2)
                            thresholddivenergy = energydivthreshold ** -1
                            crosssection = sigmat * (beta + (1 - beta)*(thresholddivenergy)) * (thresholddivenergy ** s)
                            photoionization_crosssections[lowerlevelid].append((energydivthreshold*thresholddivenergy/ryd_to_ev, crosssection))
    #                    print('Using Seaton formula values for lower level {0:d}'.format(lowerlevelid))
                    numpointsexpected = len(photoionization_crosssections[lowerlevelid])
            elif crosssectiontype == '7' and len(row) == 1 and artisatomic.isfloat(row[0]) and numpointsexpected > 0:
                seatonfittingcoefficients.append(float(row[0].replace('D', 'E')))
                if len(seatonfittingcoefficients) == 4:
                    (sigmat, beta, s, nuo) = seatonfittingcoefficients
                    for lowerlevelid in level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname]:
                        for c in np.arange(0, 1.0, 0.01):
                            energydivthreshold = 1 + 20 * (c ** 2)
                            thresholdenergyev = hc_in_ev_angstrom / float(hillier_energy_levels[i][lowerlevelid].lambdaangstrom)
                            energyoffsetdivthreshold = energydivthreshold + (nuo*1e15*h_in_ev_seconds)/thresholdenergyev
                            thresholddivenergyoffset = energyoffsetdivthreshold ** -1
                            if thresholddivenergyoffset < 1.0:
                                crosssection = sigmat * (beta + (1 - beta)*(thresholddivenergyoffset)) * (thresholddivenergyoffset ** s)
                            else:
                                crosssection = 0

                            photoionization_crosssections[lowerlevelid].append((energydivthreshold*thresholddivenergy/ryd_to_ev, crosssection))
    #                    print('Using modified Seaton formula values for lower level {0:d}'.format(lowerlevelid))
                    numpointsexpected = len(photoionization_crosssections[lowerlevelid])

            if len(row) >= 2 and ' '.join(row[1:]) == '!Type of cross-section':
                crosssectiontype = row[0]
                if crosssectiontype not in ['1', '7', '20', '21']:
                    if crosssectiontype != '-1':
                        print('Warning: Unknown cross-section type: "{0}"'.format(crosssectiontype))
    #                                sys.exit()
                    truncatedlowerlevelname = ''
                    crosssectiontype = '-1'
                    numpointsexpected = 0

            if len(row) == 0:
                if (truncatedlowerlevelname != '' and
                        numpointsexpected != len(photoionization_crosssections[level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname][0]])):
                    print('photoionization_crosssections mismatch: expecting {0:d} rows but found {1:d}'.format(
                        numpointsexpected, len(photoionization_crosssections[level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname][0]])))
                    print('A={0}, ion_stage={1}, lowerlevel={2}, crosssectiontype={3}'.format(
                        atomic_number, ion_stage, truncatedlowerlevelname, crosssectiontype))
                    print('matching level ids: ', level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname])
                    print(photoionization_crosssections[level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname][0]])
                    sys.exit()
                seatonfittingcoefficients = []
                truncatedlowerlevelname = ''
                numpointsexpected = 0
    return photoionization_crosssections

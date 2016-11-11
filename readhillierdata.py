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

# need to also include collision strengths from e.g., o2col.dat

hillier_rowformata = 'levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4'
hillier_rowformatb = 'levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad c4 c6'
hillier_rowformatc = 'levelname g energyabovegsinpercm freqtentothe15hz lambdaangstrom hillierlevelid'

# keys are (atomic number, ion stage)
ion_files = namedtuple('ion_files', ['folder', 'levelstransitionsfilename', 'energylevelrowformat', 'photfilenames'])

ions_data = {
    # O
    (8, 1): ion_files('20sep11', 'oi_osc_mchf', hillier_rowformatb, ['']),
    (8, 2): ion_files('23mar05', 'o2osc_fin.dat', hillier_rowformata, ['']),
    (8, 3): ion_files('15mar08', 'oiiiosc', hillier_rowformata, ['']),

    # F
    (9, 2): ion_files('tst', 'fin_osc', hillier_rowformata, ['phot_data_a', 'phot_data_b', 'phot_data_c']),
    (9, 3): ion_files('tst', 'fin_osc', hillier_rowformata, ['phot_data_a', 'phot_data_b', 'phot_data_c', 'phot_data_d']),

    # Ca
    (20, 1): ion_files('5aug97', 'cai_osc.dat', hillier_rowformatc, ['cai_phot_a.dat']),
    (20, 2): ion_files('30oct12', 'ca2_osc.dat', hillier_rowformatc, ['ca2_phot_a.dat']),
    (20, 3): ion_files('10apr99', 'osc_op_sp.dat', hillier_rowformatc, ['phot_smooth.dat']),
    (20, 4): ion_files('10apr99', 'osc_op_sp.dat', hillier_rowformatc, ['phot_smooth.dat']),

    # Fe
    (26, 1): ion_files('29apr04', 'fei_osc', hillier_rowformata, ['']),
    (26, 2): ion_files('16nov98', 'fe2osc_nahar_kurucz.dat', hillier_rowformatc, ['']),
    (26, 3): ion_files('30oct12', 'FeIII_OSC', hillier_rowformatb, ['']),
    (26, 4): ion_files('18oct00', 'feiv_osc_rev2.dat', hillier_rowformata, ['']),
    (26, 5): ion_files('18oct00', 'fev_osc.dat', hillier_rowformata, ['']),

    # Co
    (27, 2): ion_files('15nov11', 'fin_osc_bound', hillier_rowformata, ['']),

    # Ni
    (28, 2): ion_files('30oct12', 'nkii_osc.dat', hillier_rowformata, ['30oct12/phot_data']),
    (28, 3): ion_files('27aug12', 'nkiii_osc.dat', hillier_rowformatb, ['']),
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

ryd_to_ev = u.rydberg.to('eV')
hc_in_ev_cm = (const.h * const.c).to('eV cm').value
hc_in_ev_angstrom = (const.h * const.c).to('eV angstrom').value
h_in_ev_seconds = const.h.to('eV s').value
lchars = 'SPDFGHIKLMNOPQRSTUVWXYZ'
PYDIR = os.path.dirname(os.path.abspath(__file__))
elsymbols = ['n'] + list(pd.read_csv(os.path.join(PYDIR, 'elements.csv'))['symbol'].values)

# hilliercodetoelsymbol = {v : k for (k,v) in elsymboltohilliercode.items()}
# hilliercodetoatomic_number = {k : elsymbols.index(v) for (k,v) in hilliercodetoelsymbol.items()}

atomic_number_to_hillier_code = {elsymbols.index(k): v for (k, v) in elsymboltohilliercode.items()}


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    row_format_energy_level = ions_data[(atomic_number, ion_stage)].energylevelrowformat
    filename = os.path.join(hillier_ion_folder(atomic_number, ion_stage),
                            ions_data[(atomic_number, ion_stage)].folder,
                            ions_data[(atomic_number, ion_stage)].levelstransitionsfilename)
    artisatomic.log_and_print(flog, 'Reading ' + filename)
    hillier_energy_level_row = namedtuple(
        'energylevel', row_format_energy_level + ' corestateid twosplusone l parity indexinsymmetry naharconfiguration matchscore')
    hillier_transition_row = namedtuple(
        'transition', 'namefrom nameto f A lambdaangstrom i j hilliertransitionid lowerlevel upperlevel coll_str')
    hillier_energy_levels = ['IGNORE']
    hillier_level_ids_matching_term = defaultdict(list)
    transition_count_of_level_name = defaultdict(int)
    hillier_ionization_energy_ev = 0.0
    transitions = []

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
                        print(filename + ', WARNING: Transition id {0:d} found at entry number {1:d}'.format(
                            int(transition.hilliertransitionid), len(transitions)))
                        sys.exit()
                else:
                    artisatomic.log_and_print(flog, 'FATAL: multiply-defined Hillier transition: {0} {1}'.format(
                        transition.namefrom, transition.nameto))
                    sys.exit()
    artisatomic.log_and_print(flog, 'Read {:d} transitions'.format(len(transitions)))

    return hillier_ionization_energy_ev, hillier_energy_levels, transitions, transition_count_of_level_name, hillier_level_ids_matching_term


def read_phixs_tables(atomic_number, ion_stage, energy_levels, args, flog):
    photoionization_crosssections = np.zeros((len(energy_levels), args.nphixspoints))  # this probably gets overwritten anyway
    photoionization_targetfractions = [[(1, 1.0)] for _ in energy_levels]
    # return np.zeros((len(energy_levels), args.nphixspoints)), photoionization_targetfractions  # TODO: replace with real data

    phixstables = defaultdict(list)
    for photfilename in ions_data[(atomic_number, ion_stage)].photfilenames:
        filename = os.path.join(hillier_ion_folder(atomic_number, ion_stage), photfilename)
        artisatomic.log_and_print(flog, 'Reading ' + filename)
        with open(filename, 'r') as fhillierphot:
            lowerlevelid = -1
            truncatedlowerlevelname = ''
            # upperlevelname = ''
            numpointsexpected = 0
            crosssectiontype = '-1'
            seatonfittingcoefficients = []

            for line in fhillierphot:
                row = line.split()

                # if len(row) >= 2 and ' '.join(row[1:]) == '!Final state in ion': # TODO: this must not be ignored!
                #     upperlevelname = row[0]  # this is not used because the upper ion's levels are not known at this time

                if len(row) >= 2 and ' '.join(row[1:]) == '!Configuration name':
                    truncatedlowerlevelname = row[0]
                    seatonfittingcoefficients = []
                    numpointsexpected = 0
                    lowerlevelid = 0
                    for levelid, energy_level in enumerate(energy_levels[1:], 1):
                        this_levelnamenoj = energy_level.levelname.split('[')[0]
                        if this_levelnamenoj == truncatedlowerlevelname:
                            lowerlevelid = levelid
                            break
                    # print('Reading level {0} '{1}''.format(lowerlevelid, truncatedlowerlevelname))

                if len(row) >= 2 and ' '.join(row[1:]) == '!Number of cross-section points':
                    numpointsexpected = int(row[0])
                    pointnumber = 0
                    phixstables[truncatedlowerlevelname] = np.zeros((numpointsexpected, 2))

                if len(row) >= 2 and ' '.join(row[1:]) == '!Cross-section unit' and row[0] != 'Megabarns':
                        print('Wrong cross-section unit: ' + row[0])
                        sys.exit()

                row_is_all_floats = all(map(artisatomic.isfloat, row))
                if (crosssectiontype in ['20', '21'] and len(row) == 2 and
                        row_is_all_floats and truncatedlowerlevelname != ''):
                    point = float(row[0].replace('D', 'E')), float(row[1].replace('D', 'E'))
                    phixstables[truncatedlowerlevelname][pointnumber] = point

                    if (pointnumber > 0 and
                            phixstables[truncatedlowerlevelname][pointnumber][0] <= phixstables[truncatedlowerlevelname][pointnumber - 1][0]):
                        print('ERROR: photoionization table first column not monotonically increasing')
                        sys.exit()

                    pointnumber += 1
                # elif True: #if want to ignore the types below
                    # pass

                elif crosssectiontype == '1' and len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                    seatonfittingcoefficients.append(float(row[0].replace('D', 'E')))
                    if len(seatonfittingcoefficients) == 3:
                        phixstables[truncatedlowerlevelname] = get_seaton_phixs_table(*seatonfittingcoefficients)
                        numpointsexpected = len(phixstables[truncatedlowerlevelname])
                        print('Using Seaton formula values for lower level {0}'.format(truncatedlowerlevelname))

                elif crosssectiontype in ['7', '8'] and len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                    seatonfittingcoefficients.append(float(row[0].replace('D', 'E')))
                    if len(seatonfittingcoefficients) == 4:
                        phixstables[truncatedlowerlevelname] = get_seaton_phixs_table(
                            *seatonfittingcoefficients, float(energy_levels[lowerlevelid].lambdaangstrom))
                        numpointsexpected = len(phixstables[truncatedlowerlevelname])
                        print('Using modified Seaton formula values for lower level {0}'.format(truncatedlowerlevelname))

                if len(row) >= 2 and ' '.join(row[1:]) == '!Type of cross-section':
                    crosssectiontype = row[0]
                    if crosssectiontype not in ['1', '7', '8', '20', '21']:
                        if crosssectiontype != '-1':
                            print('Warning: Unknown cross-section type: {0}'.format(crosssectiontype))
                            # sys.exit()
                        truncatedlowerlevelname = ''
                        crosssectiontype = '-1'
                        numpointsexpected = 0

                if len(row) == 0:
                    if (truncatedlowerlevelname != '' and
                            numpointsexpected != len(phixstables[truncatedlowerlevelname])):
                        print('photoionization_crosssections mismatch: expecting {0:d} rows but found {1:d}'.format(
                            numpointsexpected, len(phixstables[truncatedlowerlevelname])))
                        print('A={0}, ion_stage={1}, lowerlevel={2}, crosssectiontype={3}'.format(
                            atomic_number, ion_stage, truncatedlowerlevelname, crosssectiontype))
                        sys.exit()
                    seatonfittingcoefficients = []
                    truncatedlowerlevelname = ''
                    numpointsexpected = 0

        reduced_phixs_dict = artisatomic.reduce_phixs_tables(phixstables, args)
        for key, phixstable in reduced_phixs_dict.items():
            for levelid, energy_level in enumerate(energy_levels[1:], 1):
                levelnamenoj = energy_level.levelname.split('[')[0]
                if levelnamenoj == key:
                    photoionization_crosssections[levelid] = phixstable

    return photoionization_crosssections, photoionization_targetfractions


def get_seaton_phixs_table(sigmat, beta, s, nuo=None, lambda_angstrom=None):
    energygrid = np.arange(0, 1.0, 0.001)
    phixstable = np.empty((len(energygrid), 2))

    for index, c in enumerate(energygrid):
        energydivthreshold = 1 + 20 * (c ** 2)

        if nuo is None:
            thresholddivenergy = energydivthreshold ** -1
            crosssection = sigmat * (beta + (1 - beta) * (thresholddivenergy)) * (thresholddivenergy ** s)
        else:
            thresholdenergyev = hc_in_ev_angstrom / lambda_angstrom
            energyoffsetdivthreshold = energydivthreshold + (nuo * 1e15 * h_in_ev_seconds) / thresholdenergyev
            thresholddivenergyoffset = energyoffsetdivthreshold ** -1
            if thresholddivenergyoffset < 1.0:
                crosssection = sigmat * (beta + (1 - beta) * (thresholddivenergyoffset)) * (thresholddivenergyoffset ** s)
            else:
                crosssection = 0.

        phixstable[index] = energydivthreshold, crosssection
    return phixstable


def hillier_ion_folder(atomic_number, ion_stage):
    return ('atomic-data-hillier/atomic/' + atomic_number_to_hillier_code[atomic_number] + '/' + artisatomic.roman_numerals[ion_stage] + '/')

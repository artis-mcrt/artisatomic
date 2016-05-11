#!/usr/bin/env python3
import collections
import os
import sys
import math
import argparse
from astropy import constants as const
from astropy import units as u
import numpy as np
import pandas as pd

pydir = os.path.dirname(os.path.abspath(__file__))
elsymbols = ['n'] + list(pd.read_csv(
    os.path.join(pydir, 'elements.csv'))['symbol'].values)

roman_numerals = ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
                  'X', 'XI','XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII',
                  'XIX', 'XX')

elsymboltohilliercode = {
    'H': 'HYD',  'He': 'HE',
    'C': 'CARB', 'N': 'NIT',
    'O': 'OXY',  'F': 'FLU',
    'Ne': 'NEON', 'Na': 'SOD',
    'Mg': 'MG',   'Al': 'ALUM',
    'Si': 'SIL',  'P': 'PHOS',
    'S': 'SUL',  'Cl': 'CHL',
    'Ar': 'ARG',  'K': 'POT',
    'Ca': 'CAL',  'Sc': 'SCAN',
    'Ti': 'TIT',  'V': 'VAN',
    'Cr': 'CHRO', 'Mn': 'MAN',
    'Fe': 'FE',   'Co': 'COB',
    'Ni': 'NICK'}

nahar_configuration_replacements = {
    'Eqv st (0S ) 0s  a3P':  '2s2_2p4_3Pe',  # O I groundstate
    '2s22p2 (3P ) 0s  z4So': '2s2_2p3_4So',  # O II groundstate
    '2s22p  (2Po) 0s  a3P':  '2s2_2p2_3Pe',  # O III groundstate
    'Eqv st (0S ) 0s  a5D':  '3d6_4s2_5De',  # Fe I groundstate
    '3d5    (6S ) 0s  a5D':  '3d6_5De',  # Fe III groundstate
    'Eqv st (0S ) 0s  a6S':  '3d5_6Se',  # Fe IV groundstate
    'Eqv st (0S ) 0s  a4G':  '3d5_4Ge'  # Fe IV state
}

# need to also include collision strengths from e.g., o2col.dat
listelements = \
    [
#        (8, (
#            (1, '20sep11/oi_osc_mchf',               'NONE',
#             'hilliername g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad c4 c6'),
#            (2, '23mar05/o2osc_fin.dat',             'NONE',
#             'hilliername g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4'),
#            (3, '15mar08/oiiiosc',                   'NONE',
#             'hilliername g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4')
#        )),
        (26,
         (
             (1, '29apr04/fei_osc',                   '29apr04/phot_smooth_3000',
              'hilliername g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4'),
             (2, '16nov98/fe2osc_nahar_kurucz.dat',   '24may96/phot_op.dat',
              'hilliername g energyabovegsinpercm freqtentothe15hz lambdaangstrom hillierlevelid'),
             (3, '30oct12/FeIII_OSC',                 '30oct12/phot_sm_3000.dat',
              'hilliername g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad c4 c6')
#             (4, '18oct00/feiv_osc_rev2.dat',         '18oct00/phot_sm_3000.dat',
#              'hilliername g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4'),
#             (5, '18oct00/fev_osc.dat',               '18oct00/phot_sm_3000.dat',
#              'hilliername g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4')
         ))
    ]

ryd_to_ev = u.rydberg.to('eV')
h_over_kb_in_kelvin_seconds = (const.h / const.k_B).to('K s').value
ryd_to_erg = u.rydberg.to('erg')
h_in_erg_seconds = const.h.to('erg s').value  # Planck constant

hc_in_ev_cm = (const.h * const.c).to('eV cm').value
hc_in_ev_angstrom = (const.h * const.c).to('eV angstrom').value
h_in_ev_seconds = const.h.to('eV s').value

# hilliercodetoelsymbol = {v : k for (k,v) in elsymboltohilliercode.items()}
# hilliercodetoatomic_number = {k : elsymbols.index(v) for (k,v) in hilliercodetoelsymbol.items()}

atomic_number_to_hillier_code = {elsymbols.index(k): v for (k, v) in elsymboltohilliercode.items()}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Produce an ARTIS atomic database by combining Hiller and '
                    'Nahar data sets.')
    parser.add_argument('-output_folder', action='store', default='artis_files',
        help='')
    parser.add_argument('-output_folder_logs', action='store',
        default='atomic_data_logs', help='')
    parser.add_argument('-output_folder_transition_guide', action='store',
        default='transition_guide', help='')
    parser.add_argument('-nphixspoints', type=int, default=100,
        help='Number of cross section points to save in output')
    parser.add_argument('-nphixsnuincrement', type=float, default=0.1,
        help='Fraction of edge frequency incremented for each cross section '
             'point')
    parser.add_argument('-integralstepfactor', type=int, default=5000,
        help='Number of steps in integral (times something)')
    parser.add_argument('-optimaltemperature', type=int, default=3000,
        help='Temperature to use when downsampling cross sections')

    args = parser.parse_args()

    clear_files(args)
    process_files(args)


def clear_files(args):
    # clear out the file contents, so these can be appended to later
    with open(os.path.join(args.output_folder, 'adata.txt'), 'w') as fatommodels, \
            open(os.path.join(args.output_folder, 'transitiondata.txt'), 'w') as ftransitiondata, \
            open(os.path.join(args.output_folder, 'phixsdata_v2.txt'), 'w') as fphixs:
        fphixs.write('{0:d}\n'.format(args.nphixspoints))
        fphixs.write('{0:14.7e}\n'.format(args.nphixsnuincrement))


def process_files(args):
    global atomic_number, ionization_stage, i, listions
    global nahar_core_states, nahar_energy_levels, nahar_configurations, nahar_phixs_tables, nahar_ionization_potential_rydberg
    global nahar_core_state_id_of_nahar_state, nahar_level_index_of_state
    global hillier_energy_levels
    global transition_count_of_hillier_level_name, hillier_ionization_energy_ev
    global hillier_level_ids_matching_term
    global energy_levels, transitions, photoionization_crosssections
    global level_id_of_level_name
    global flog

    # for hillierelname in ('IRON',):
    #    d = './hillieratomic/' + hillierelname

    for e, (atomic_number, listions) in enumerate(listelements):

        nahar_energy_levels = [['IGNORE'] for x in listions]  # list of named tuples (naharcorestaterow)
        nahar_core_states = [['IGNORE'] for x in listions]  # list of named tuples (naharcorestaterow)

        # keys are (2S+1, L, parity, indexinsymmetry), values are integer id of Nahar core state
        nahar_core_state_id_of_nahar_state = [{} for x in listions]

        # keys are (2S+1, L, parity), values are strings of electron configuration
        nahar_configurations = [{} for x in listions]

        # keys are (2S+1, L, parity, indexinsymmetry), values are lists of (energy
        # in Rydberg, cross section in Mb) tuples
        nahar_phixs_tables = [{} for x in listions]

        nahar_ionization_potential_rydberg = [0.0 for x in listions]
        nahar_level_index_of_state = [{} for x in listions]

        # list of named tuples (first element is IGNORE to discard 0th index)
        hillier_energy_levels = [['IGNORE'] for x in listions]

        # keys are tuples (2S+1, L, parity), item is a list of truncated level names in energy order
        hillier_level_ids_matching_term = [{} for x in listions]

        hillier_ionization_energy_ev = [0.0 for x in listions]

        # list of named tuples (hillier_transition_row)
        transitions = [['IGNORE'] for x in listions]
        transition_count_of_hillier_level_name = [{} for x in listions]

        level_id_of_level_name = [{} for x in listions]
        energy_levels = [[] for x in listions]
        # keys are hillier level ids, values are lists of (energy in Rydberg,
        # cross section in Mb) tuples
        photoionization_crosssections = [[] for x in listions]

        for i, (ionization_stage, path_hillier_osc_file, path_nahar_px_file,
                hillier_row_format_energy_level) in enumerate(listions):

            with open(os.path.join(args.output_folder, args.output_folder_logs,
                                   '{0}{1:d}.txt'.format(
                                       elsymbols[atomic_number].lower(),
                                       ionization_stage)),
                      'w') as flog:

                hillier_ion_folder = 'atomic-data-hillier/atomic/' + \
                    atomic_number_to_hillier_code[atomic_number] + \
                    '/' + roman_numerals[ionization_stage] + '/'

                log_and_print('==============> {0} {1}:'.format(
                    elsymbols[atomic_number], roman_numerals[ionization_stage]))

                path_nahar_energy_file = 'atomic-data-nahar/{0}{1:d}.en.ls.txt'.format(
                    elsymbols[atomic_number].lower(), ionization_stage)
                log_and_print('Reading ' + path_nahar_energy_file)
                read_nahar_energy_level_file(path_nahar_energy_file)

                log_and_print('Reading ' + hillier_ion_folder + path_hillier_osc_file)
                read_hillier_levels_and_transitions_file(
                    hillier_ion_folder, path_hillier_osc_file, hillier_row_format_energy_level)

                if i < len(listions) - 1:  # don't get cross sections for top ion
                    path_nahar_px_file = 'atomic-data-nahar/{0}{1:d}.px.txt'.format(
                        elsymbols[atomic_number].lower(), ionization_stage)
                    log_and_print('Reading ' + path_nahar_px_file)
                    read_nahar_phixs_tables(path_nahar_px_file)

                combine_sources(args)

                # Alternatively use Hillier phixs tables, but BEWARE this probably doesn't work anymore since the code has changed a lot
                # print('Reading ' + hillier_ion_folder + path_nahar_px_file)
                # read_hillier_phixs_tables(hillier_ion_folder,path_nahar_px_file)

        with open(os.path.join(args.output_folder, 'adata.txt'), 'a') as fatommodels, \
                open(os.path.join(args.output_folder, 'transitiondata.txt'), 'a') as ftransitiondata, \
                open(os.path.join(args.output_folder, 'phixsdata_v2.txt'), 'a') as fphixs, \
                open(os.path.join(args.output_folder_transition_guide, 'transitions_{}.txt'.format(elsymbols[atomic_number])), 'w') as ftransitionguide:
            print('\nStarting output stage:')

            ftransitionguide.write('{0:16s} {1:12s} {2:3s} {3:9s} {4:17s} {5:17s} {6:10s} {7:25s} {8:25s} {9:17s} {10:17s} {11:19s}\n'.format(
                'lambda_angstroms', 'A', 'Z', 'ion_stage', 'lower_energy_Ev', 'lower_statweight', 'forbidden', 'lower_level', 'upper_level', 'upper_statweight', 'upper_energy_Ev', 'upper_has_permitted'))

            for i, (ionization_stage, path_hillier_osc_file, path_nahar_px_file,
                    hillier_row_format_energy_level) in enumerate(listions):

                ionstr = '{0} {1}'.format(elsymbols[atomic_number],
                                          roman_numerals[ionization_stage])

                flog = open(
                    os.path.join(args.output_folder,
                                 args.output_folder_logs,
                                 '{0}{1:d}.txt'.format(
                         elsymbols[atomic_number].lower(), ionization_stage)),
                    'a')

                print('==============> ' + ionstr + ':')

                write_adata(fatommodels, i, args)

                write_transition_data(ftransitiondata, ftransitionguide, i,
                                      args)

                write_phixs_data(fphixs, i, args)


def read_nahar_energy_level_file(path_nahar_energy_file):
    nahar_energy_level_row = collections.namedtuple(
        'energylevel', 'indexinsymmetry TC corestateid elecn elecl energyreltoionpotrydberg twosplusone l parity energyabovegsinpercm g')
    naharcorestaterow = collections.namedtuple(
        'naharcorestate', 'nahar_core_state_id configuration term energyrydberg')
    with open(path_nahar_energy_file, 'r') as fenlist:
        while True:
            line = fenlist.readline()
            if not line:
                print('End of file before data section')
                sys.exit()
            if line.startswith(' i) Table of target/core states in the wavefunction expansion'):
                break

        while True:
            line = fenlist.readline()
            if not line:
                print('End of file before end of core states table')
                sys.exit()
            if line.startswith(' no of target/core states in WF'):
                numberofcorestates = int(line.split('=')[1])
                break
        fenlist.readline()  # blank line
        fenlist.readline()  # ' target states and energies:'
        fenlist.readline()  # blank line

        nahar_core_states[i] = [()] * (numberofcorestates + 1)
        for c in range(1, numberofcorestates + 1):
            row = fenlist.readline().split()
            nahar_core_states[i][c] = naharcorestaterow._make(
                (int(row[0]), row[1], row[2], float(row[3])))
            if int(nahar_core_states[i][c].nahar_core_state_id) != c:
                print('Nahar levels mismatch: id {0:d} found at entry number {1:d}'.format(
                    c, int(nahar_core_states[i][c].nahar_core_state_id)))
                sys.exit()

        while True:
            line = fenlist.readline()
            if not line:
                print('End of file before data section ii)')
                sys.exit()
            if line.startswith('ii) Table of bound (negative) state energies (with spectroscopic notation)'):
                break

        found_table = False
        while True:
            line = fenlist.readline()
            if not line:
                print('End of file before end of state table')
                sys.exit()

            if line.startswith(' Ion ground state'):
                nahar_ionization_potential_rydberg[i] = -float(line.split('=')[2])
                flog.write('Ionization potential = {0} Ryd\n'.format(
                    nahar_ionization_potential_rydberg[i]))

            if line.startswith(' ') and len(line) > 36 and isfloat(line[29:29 + 8]):
                found_table = True
                state = line[1:22]
#               energy = float(line[29:29+8])
                twosplusone = int(state[18])
                l = lchars.index(state[19])

                if state[20] == 'o':
                    parity = 1
                    indexinsymmetry = reversedalphabets.index(state[17]) + 1
                else:
                    parity = 0
                    indexinsymmetry = alphabets.index(state[17]) + 1

                # print(state,energy,twosplusone,l,parity,indexinsymmetry)
                nahar_configurations[i][(twosplusone, l, parity, indexinsymmetry)] = state
            else:
                if found_table:
                    break

        while True:
            line = fenlist.readline()
            if not line:
                print('End of file before table iii header')
                sys.exit()
            if line.startswith('iii) Table of complete set (both negative and positive) of energies'):
                break
        line = fenlist.readline()  # line of ----------------------
        while True:
            line = fenlist.readline()
            if not line:
                print('End of file before table iii starts')
                sys.exit()
            if line.startswith('------------------------------------------------------'):
                break

        row = fenlist.readline().split()  # line of atomic number and electron number, unless
        while len(row) == 0:  # some extra blank lines might exist
            row = fenlist.readline().split()

        if atomic_number != int(row[0]) or ionization_stage != int(row[0]) - int(row[1]):
            print('Wrong atomic number or ionization stage in Nahar energy file',
                  atomic_number, int(row[0]), ionization_stage, int(row[0]) - int(row[1]))
            sys.exit()

        while True:
            line = fenlist.readline()
            if not line:
                print('End of file before table iii finished')
                sys.exit()
            row = line.split()
            if '-'.join(row) == '0-0-0-0':
                break
            twosplusone = int(row[0])
            l = int(row[1])
            parity = int(row[2])
            number_of_states_in_symmetry = int(row[3])

            line = fenlist.readline()  # core state number and energy

            for s in range(number_of_states_in_symmetry):
                row = fenlist.readline().split()
                indexinsymmetry = int(row[0])
                nahar_core_state_id = int(row[2])
                if nahar_core_state_id < 1 or nahar_core_state_id > len(nahar_core_states[i]):
                    flog.write("Core state id of {0:d}{1}{2} index {3:d} is invalid (={4:d}, Ncorestates={5:d})\n".format(
                        twosplusone, lchars[l], ['e', 'o'][parity],
                        indexinsymmetry, nahar_core_state_id,
                        len(nahar_core_states[i])))

                    if nahar_core_state_id == 0:
                        flog.write("(setting the core state to 1 instead)\n")
                        nahar_core_state_id = 1

                nahar_energy_levels[i].append(nahar_energy_level_row._make(
                    row + [twosplusone, l, parity, -1.0, 0]))
                energyabovegsinpercm = \
                    (nahar_ionization_potential_rydberg[i] +
                     float(nahar_energy_levels[i][-1].energyreltoionpotrydberg)) * \
                    ryd_to_ev / hc_in_ev_cm

                nahar_energy_levels[i][-1] = nahar_energy_levels[i][-1]._replace(
                    indexinsymmetry=indexinsymmetry,
                    corestateid=nahar_core_state_id,
                    energyreltoionpotrydberg=float(
                        nahar_energy_levels[i][-1].energyreltoionpotrydberg),
                    energyabovegsinpercm=energyabovegsinpercm,
                    g=twosplusone * (2 * l + 1)
                )
                nahar_level_index_of_state[i][(twosplusone, l, parity, indexinsymmetry)] = len(
                    nahar_energy_levels[i]) - 1

#                if float(nahar_energy_levels[i][-1].energyreltoionpotrydberg) >= 0.0:
#                    nahar_energy_levels[i].pop()

    # end reading Nahar energy file
    return


def read_hillier_levels_and_transitions_file(hillier_ion_folder, path_hillier_osc_file, hillier_row_format_energy_level):
    hillier_energy_level_row = collections.namedtuple(
        'energylevel', hillier_row_format_energy_level + ' corestateid twosplusone l parity indexinsymmetry naharconfiguration')
    hillier_transition_row = collections.namedtuple(
        'transition', 'namefrom nameto f A lambdaangstrom i j hilliertransitionid')
    with open(hillier_ion_folder + path_hillier_osc_file, 'r') as fhillierosc:
        for line in fhillierosc:
            row = line.split()

            # check for right number of columns and that are all numbers except first column
            if len(row) == len(hillier_row_format_energy_level.split()) and all(map(isfloat, row[1:])):
                hillier_energy_level = hillier_energy_level_row._make(row + [0, -1, -1, -1, -1, ''])

                levelname = hillier_energy_level.hilliername
                transition_count_of_hillier_level_name[i][levelname] = 0
                hillierlevelid = int(hillier_energy_level.hillierlevelid.lstrip('-'))
                energyabovegsinpercm = float(hillier_energy_level.energyabovegsinpercm)

                (twosplusone, l, parity) = get_term_as_tuple(levelname)
                hillier_energy_level = hillier_energy_level._replace(
                    hillierlevelid=hillierlevelid,
                    energyabovegsinpercm=energyabovegsinpercm,
                    twosplusone=twosplusone,
                    l=l,
                    parity=parity
                )

                hillier_energy_levels[i].append(hillier_energy_level)

                if twosplusone == -1:
                    # -1 indicates that the term could not be interpreted
                    log_and_print("Can't find term in Hillier level name '" + levelname + "'")
                else:
                    if (twosplusone, l, parity) in hillier_level_ids_matching_term[i].keys():
                        if levelname not in hillier_level_ids_matching_term[i][(twosplusone, l, parity)]:
                            hillier_level_ids_matching_term[i][
                                (twosplusone, l, parity)].append(hillierlevelid)
                    else:
                        hillier_level_ids_matching_term[i][(twosplusone, l, parity)] = [
                            hillierlevelid, ]

                # if this is the ground state
                if float(hillier_energy_levels[i][-1].energyabovegsinpercm) < 1.0:
                    hillier_ionization_energy_ev[i] = hc_in_ev_angstrom / \
                        float(hillier_energy_levels[i][-1].lambdaangstrom)

                if hillierlevelid != len(hillier_energy_levels[i]) - 1:
                    print('Hillier levels mismatch: id {0:d} found at entry number {1:d}'.format(
                        len(hillier_energy_levels[i]) - 1, hillierlevelid))
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

            if len(row) == 8 and all(map(isfloat, row[2:4])):
                transition = hillier_transition_row._make(row)

                if True:  # or int(transition.hilliertransitionid) not in defined_transition_ids: #checking for duplicates massively slows down the code
                    #                    defined_transition_ids.append(int(transition.hilliertransitionid))
                    transitions[i].append(transition)
                    transition_count_of_hillier_level_name[i][transition.namefrom] += 1
                    transition_count_of_hillier_level_name[i][transition.nameto] += 1

                    if int(transition.hilliertransitionid) != len(transitions[i]) - 1:
                        print(hillier_ion_folder + path_hillier_osc_file +
                              ', WARNING: Transition id {0:d} found at entry number {1:d}'.format
                              (
                                  int(transition.hilliertransitionid),
                                  len(transitions[i]) - 1)
                              )
#                    sys.exit()
                else:
                    log_and_print('FATAL: multiply-defined Hillier transition: {0} {1}'
                             .format(transition.namefrom, transition.nameto))
                    sys.exit()
    log_and_print('Read {:d} transitions'.format(len(transitions[i]) - 1))
    return


def read_nahar_phixs_tables(path_nahar_px_file):
    with open(path_nahar_px_file, 'r') as fenlist:
        while True:
            line = fenlist.readline()
            if not line:
                print('End of file before data section')
                sys.exit()
            if line.startswith('----------------------------------------------'):
                break

        line = ""
        while len(line.strip()) == 0:
            line = fenlist.readline()
        row = line.split()
        if atomic_number != int(row[0]) or ionization_stage != int(row[0]) - int(row[1]):
            print('Wrong atomic number or ionization stage in Nahar file',
                  atomic_number, int(row[0]), ionization_stage,
                  int(row[0]) - int(row[1]))
            sys.exit()

        while True:
            line = fenlist.readline()
            row = line.split()

            if not line or sum(map(float, row)) == 0:
                break
            twosplusone = int(row[0])
            l = int(row[1])
            parity = int(row[2])
            indexinsymmetry = int(row[3])
            line = fenlist.readline()
            number_of_points = int(line.split()[1])
            nahar_binding_energy_rydberg = float(fenlist.readline().split()[0])

            nahar_phixs_tables[i][
                (twosplusone, l, parity, indexinsymmetry)] = np.empty(
                    (number_of_points, 2))

            for p in range(number_of_points):
                row = fenlist.readline().split()
                nahar_phixs_tables[i][
                    (twosplusone, l, parity, indexinsymmetry)][p] = (
                        float(row[0]), float(row[1]))

                # ensure that energies are monotonically increasing
#                if (len(nahar_phixs_tables[i][(twosplusone,l,parity,indexinsymmetry)]) > 1 and
#                    nahar_phixs_tables[i][(twosplusone,l,parity,indexinsymmetry)][-1][0] <= nahar_phixs_tables[i][(twosplusone,l,parity,indexinsymmetry)][-2][0]):
#                    nahar_phixs_tables[i][(twosplusone,l,parity,indexinsymmetry)].pop()

    return


def combine_sources(args):
    # hillier_energy_levels[i] = ['IGNORE'] #TESTING only
    # hillier_level_ids_matching_term[i] = {} #TESTING only
    added_nahar_levels = []
    # match up Nahar states given in phixs data with Hillier levels, adding
    # missing levels as necessary
    for (twosplusone, l, parity, indexinsymmetry) in nahar_phixs_tables[i].keys():
        state_tuple = (twosplusone, l, parity, indexinsymmetry)
        hillier_level_ids_matching_this_nahar_state = []

        nahar_configuration_this_state = '_CONFIG NOT FOUND_'

        if state_tuple in nahar_configurations[i]:
            nahar_configuration_this_state = nahar_configurations[i][state_tuple]

            if nahar_configuration_this_state.strip() in nahar_configuration_replacements.keys():
                nahar_configuration_this_state = nahar_configuration_replacements[
                    nahar_configurations[i][state_tuple].strip()]
                log_and_print("Replacing Nahar configuration of '{0}' with '{1}'".format(
                    nahar_configurations[i][state_tuple], nahar_configuration_this_state))

        if (twosplusone, l, parity) in hillier_level_ids_matching_term[i].keys():
                # match the electron configurations
            if nahar_configuration_this_state != '_CONFIG NOT FOUND_':
                for levelid in hillier_level_ids_matching_term[i][(twosplusone, l, parity)]:
                    levelname = hillier_energy_levels[i][levelid].hilliername
                    if reduce_configuration(levelname) == reduce_configuration(nahar_configuration_this_state) and \
                            hillier_energy_levels[i][levelid].indexinsymmetry < 1:  # make sure this Hillier level hasn't already been matched to a Nahar state

                        core_state_id = nahar_energy_levels[i][
                            nahar_level_index_of_state[i][state_tuple]].corestateid

                        confignote = nahar_configurations[i][state_tuple]

                        if nahar_configuration_this_state != confignote:
                            confignote += " manually replaced by {:}".format(nahar_configuration_this_state)

                        hillier_energy_levels[i][levelid] = hillier_energy_levels[i][levelid]._replace(
                            twosplusone=twosplusone, l=l, parity=parity,
                            indexinsymmetry=indexinsymmetry, corestateid=core_state_id,
                            naharconfiguration=confignote)
                        hillier_level_ids_matching_this_nahar_state.append(levelid)
            else:
                log_and_print("No electron configuration for {0:d}{1}{2} index {3:d}".format(
                    twosplusone, lchars[l], ['e', 'o'][parity], indexinsymmetry))
        else:
            flog.write("No Hillier levels with term {0:d}{1}{2}\n".format(
                twosplusone, lchars[l], ['e', 'o'][parity]))

        naharthresholdrydberg = nahar_phixs_tables[i][state_tuple][0][0]
        if len(hillier_level_ids_matching_this_nahar_state) == 0:
            flog.write("No matched Hillier levels for Nahar cross section of {0:d}{1}{2} index {3:d} [{4}] ".format(twosplusone, lchars[l], ['e', 'o'][parity], indexinsymmetry,
                                                                                                                    (nahar_configurations[i][state_tuple] if state_tuple in nahar_configurations[i] else 'CONFIG NOT FOUND')) +
                       '(E_threshold={0:.2f} eV)'.format(naharthresholdrydberg * ryd_to_ev) + "\n")
            # now find the Nahar level and add it to the new list
            if state_tuple in nahar_level_index_of_state[i]:
                nahar_energy_level = nahar_energy_levels[i][
                    nahar_level_index_of_state[i][state_tuple]]
                if nahar_energy_level.energyabovegsinpercm * hc_in_ev_cm < 0.002:
                    flog.write(" but prevented duplicating the ground state\n")
                else:
                    added_nahar_levels.append(nahar_energy_level)
            else:
                flog.write(" (and no matching entry in Nahar energy table, so can't be added)\n")
        else:
            nahar_energyabovegsinev = hc_in_ev_cm * \
                nahar_energy_levels[i][nahar_level_index_of_state[
                    i][state_tuple]].energyabovegsinpercm
            # avghillierthreshold = weightedavgthresholdinev(
            #    hillier_energy_levels[i], hillier_level_ids_matching_this_nahar_state)
            strhilliermatchesthreshold = '[' + ', '.join(['{0} ({1:.3f} eV)'.format(hillier_energy_levels[i][k].hilliername, hc_in_ev_angstrom /
                                                                                    float(hillier_energy_levels[i][k].lambdaangstrom)) for k in hillier_level_ids_matching_this_nahar_state]) + ']'
            avghillierenergyabovegsinev = weightedavgenergyinev(
                hillier_energy_levels[i], hillier_level_ids_matching_this_nahar_state)

            strhilliermatchesenergy = '[' + ', '.join(['{0} ({1:.3f} eV)'.format(hillier_energy_levels[i][k].hilliername, hc_in_ev_cm * float(
                hillier_energy_levels[i][k].energyabovegsinpercm)) for k in hillier_level_ids_matching_this_nahar_state]) + ']'

            flog.write("Matched Nahar phixs for {0:d}{1}{2} index {3:d} '{4}' ".format(twosplusone, lchars[l], ['e', 'o'][parity], indexinsymmetry,
                                                                                       nahar_configuration_this_state) +
                       '({0:.3f} eV) to '.format(nahar_energyabovegsinev) +
                       '<E>={0:.3f} eV: '.format(avghillierenergyabovegsinev) + strhilliermatchesenergy + '\n')

    energy_levels[i] = hillier_energy_levels[i] + added_nahar_levels

    log_and_print('Included {0} levels from Hillier dataset and added {1} levels from Nahar phixs tables for a total of {2} levels'.format(
        len(hillier_energy_levels[i]) - 1, len(added_nahar_levels), len(energy_levels[i]) - 1))

    # sort the combined energy level list by energy
    print('Sorting level list...')
    energy_levels[i].sort(key=lambda x: float(x.energyabovegsinpercm)
                          if hasattr(x, 'energyabovegsinpercm') else float('-inf'))

    # generate a mapping of level names to the final output level id numbers
    print('Mapping level names to ID numbers...')
    for levelid in range(1, len(energy_levels[i])):
        if hasattr(energy_levels[i][levelid], 'hilliername'):
            level_id_of_level_name[i][energy_levels[i][levelid].hilliername] = levelid

    photoionization_crosssections[i] = np.zeros((len(energy_levels[i]), args.nphixspoints))

    print('Processing phixs tables...')
    # process the phixs tables and attach them to any matching levels in the output list
    for (twosplusone, l, parity, indexinsymmetry) in nahar_phixs_tables[i].keys():
        reduced_phixs_list = reduce_phixs_table(
            nahar_phixs_tables[i][(twosplusone, l, parity, indexinsymmetry)],
            args)
        foundamatch = False
        for levelid in range(1, len(energy_levels[i])):
            if (int(energy_levels[i][levelid].twosplusone) == twosplusone and
                    int(energy_levels[i][levelid].l) == l and
                    int(energy_levels[i][levelid].parity) == parity and
                    int(energy_levels[i][levelid].indexinsymmetry) == indexinsymmetry):
                photoionization_crosssections[i][levelid] = list(reduced_phixs_list)
                foundamatch = True  # there could be more than one match, but this flags there being at least one

        if not foundamatch:
            log_and_print("No Hiller or Nahar state to match with photoionization crosssection of {0:d}{1}{2} index {3:d}".format(
                twosplusone, lchars[l], ['e', 'o'][parity], indexinsymmetry))
    return


def log_and_print(strout):
    print(strout)
    flog.write(strout + "\n")


def isfloat(value):
    try:
        float(value.replace('D', 'E'))
        return True
    except ValueError:
        return False


# this method downsamples the photoionization cross section table to a
# regular grid while keeping the recombination rate integral constant
# (assuming that the temperature matches)
def reduce_phixs_table(listin, args):  # listin is a list of pairs (energy, phixs cross section)
    listout = []
    xgrid = np.arange(1.0, 1.0 + args.nphixsnuincrement * (args.nphixspoints + 1), args.nphixsnuincrement)
#    xgrid = list(filter(lambda x:x < (listin[-1][0]/listin[0][0]),xgrid))

    for i in range(len(xgrid) - 1):
        enlow = xgrid[i] * listin[0][0]
        enhigh = xgrid[i + 1] * listin[0][0]

        listenergyryd = np.linspace(enlow, enhigh, num=args.nphixsnuincrement *
                                    args.integralstepfactor, endpoint=False)  # change back to *5000
        # dnu = (listenergyryd[1] - listenergyryd[0]) * ryd_to_erg / h_in_erg_seconds

        listsigma_bf_megabarns = np.interp(listenergyryd, listin[:, :1].flatten(), listin[
                                           :, 1:].flatten(), right=-1)
        # this method is incredibly slow!
        # listsigma_bf_megabarns = np.interp(listenergyryd,*zip(*listin),right=-1)

        integralnosigma = 0.0
        integralwithsigma = 0.0
        nu0 = listenergyryd[0] * ryd_to_erg / h_in_erg_seconds
        previntegrand = nu0 ** 2
        for j in range(1, len(listenergyryd)):
            energyryd = listenergyryd[j]
            sigma_bf_megabarns = listsigma_bf_megabarns[j]
            if sigma_bf_megabarns < 0:  # negative value means we're in the extrapolation regime
                # assume a power law decline from the highest-energy cross section point
                sigma_bf_megabarns = (listin[-1][0] / energyryd) ** 3 * listin[-1][1]
            nu = energyryd * ryd_to_erg / h_in_erg_seconds

            integrandnosigma = (nu ** 2) * math.exp(-h_over_kb_in_kelvin_seconds *
                                                    (nu - nu0) / args.optimaltemperature)
            integralcontribution = (integrandnosigma + previntegrand) / 2.0
            integralnosigma += integralcontribution
            integralwithsigma += integralcontribution * sigma_bf_megabarns
            previntegrand = integrandnosigma

        if integralnosigma > 0:
            listout.append((integralwithsigma / integralnosigma))
        else:
            listout.append((0.0))
            print('probable underflow')

    return listout[:args.nphixspoints]  # output is a 1D list of cross sections


def weightedavgenergyinev(energy_levelsthision, ids):
    genergysum = 0.0
    gsum = 0.0
    for levelid in ids:
        statisticalweight = float(energy_levelsthision[levelid].g)
        genergysum += statisticalweight * hc_in_ev_cm * \
            float(energy_levelsthision[levelid].energyabovegsinpercm)
        gsum += statisticalweight
    return genergysum / gsum


def weightedavgthresholdinev(energy_levelsthision, ids):
    genergysum = 0.0
    gsum = 0.0
    for levelid in ids:
        statisticalweight = float(energy_levelsthision[levelid].g)
        genergysum += statisticalweight * hc_in_ev_angstrom / \
            float(energy_levelsthision[levelid].lambdaangstrom)
        gsum += statisticalweight
    return genergysum / gsum

alphabets = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
reversedalphabets = 'zyxwvutsrqponmlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA '
lchars = 'SPDFGHIKLMNOPQRSTUVWXYZ'


# reads a Hillier level name and returns the term tuple (twosplusone, l, parity)
def get_term_as_tuple(config):
    config = config.split('[')[0]
    for charpos in reversed(range(len(config))):
        if config[charpos] in lchars:
            lposition = charpos
            l = lchars.index(config[charpos])
            break

    twosplusone = int(config[lposition - 1])  # could this be two digits long?
    if lposition + 1 > len(config) - 1:
        parity = 0
    elif config[lposition + 1] == 'o':
        parity = 1
    elif config[lposition + 1] == 'e':
        parity = 0
    elif config[lposition + 2] == 'o':
        parity = 1
    elif config[lposition + 2] == 'e':
        parity = 0
    else:
        twosplusone = -1
        l = -1
        parity = -1
#        sys.exit()
    return (twosplusone, l, parity)


# e.g. convert "3d64s  (6D ) 8p  j5Fo" to "3d64s8p_5Fo",
# similar to Hillier style "3d6(5D)4s8p_5Fo" but without the parent term
# (and mysterious letter before the term if present)
def reduce_configuration(instr):
    if instr == "-1":
        return "-1"
    instr = instr.split('[')[0]
    if instr[-1] in lchars:
        instr = instr + 'e'
    outstr = ""
    in_brackets = False
    for char in instr[:-4]:
        if char == ' ' or char == '_':
            continue
        if char == '(':
            in_brackets = True
        if not in_brackets:
            outstr += char
        if char == ')':
            in_brackets = False

    outstr += '_'
    outstr += instr[-3:-1]
    if instr[-1] == 'o':
        outstr += 'o'
    else:
        outstr += 'e'
    return outstr


def write_adata(fatommodels, i, args):
    log_and_print("writing to 'adata.txt'")
    fatommodels.write('{0:12d}{1:12d}{2:12d}{3:15.7f}\n'.format(
        atomic_number, ionization_stage, len(energy_levels[i]) - 1, hillier_ionization_energy_ev[i]))

    for levelid in range(1, len(energy_levels[i])):
        energylevel = energy_levels[i][levelid]
        if hasattr(energylevel, 'hilliername') and energylevel.hilliername in transition_count_of_hillier_level_name[i]:
            transitioncount = transition_count_of_hillier_level_name[
                i][energylevel.hilliername]
        else:
            transitioncount = 0

        level_comment = ""
        try:
            level_comment = "Hiller: '{:}', ".format(energylevel.hilliername)
        except AttributeError:
            pass

        try:
            level_comment += 'Nahar: {:d}{:}{:} index {:}'.format(
                energylevel.twosplusone,
                lchars[energylevel.l],
                ['e', 'o'][energylevel.parity],
                energylevel.indexinsymmetry)
        except AttributeError:
            pass

        try:
            level_comment += " '{:}'".format(energylevel.naharconfiguration)
        except AttributeError:
            level_comment += ' (no config)'

        fatommodels.write('{:7d}{:25.16f}{:25.16f}{:7d}     {:}\n'.format(levelid, hc_in_ev_cm * float(
            energylevel.energyabovegsinpercm), float(energylevel.g), transitioncount, level_comment))
    fatommodels.write('\n')


def write_transition_data(ftransitiondata, ftransitionguide, i, args):
    log_and_print("writing to 'transitiondata.txt'")
    num_forbidden_transitions = 0
    ftransitiondata.write('{0:7d}{1:7d}{2:12d}\n'.format(
        atomic_number, ionization_stage, len(transitions[i]) - 1))

    level_ids_with_permitted_down_transitions = set()
    for transitionid in range(1, len(transitions[i])):
        transition = transitions[i][transitionid]
        levelid_from = level_id_of_level_name[i][transition.namefrom]
        levelid_to = level_id_of_level_name[i][transition.nameto]
        forbidden = (energy_levels[i][levelid_from].parity ==
                     energy_levels[i][levelid_to].parity)
        if not forbidden:
            level_ids_with_permitted_down_transitions.add(
                levelid_to)  # hopefully 'to_level' is the upper level

    for transitionid in range(1, len(transitions[i])):
        transition = transitions[i][transitionid]
        levelid_from = level_id_of_level_name[i][transition.namefrom]
        levelid_to = level_id_of_level_name[i][transition.nameto]
        forbidden = (energy_levels[i][levelid_from].parity ==
                     energy_levels[i][levelid_to].parity)
        if forbidden:
            num_forbidden_transitions += 1
            coll_str = -2
            flog.write('Forbidden transition: lambda_angstrom= {:7.1f}, {:25s} to {:25s}\n'.format(
                float(transition.lambdaangstrom), transition.namefrom, transition.nameto))
        else:
            coll_str = -1

        if True:  # ionization_stage in [1,2,3]:
            ftransitionguide.write('{0:16.1f} {1:12E} {2:3d} {3:9d} {4:17.2f} {5:17.4f} {6:10b} {7:25s} {8:25s} {9:17.2f} {10:17.4f} {11:19b}\n'.format(abs(float(transition.lambdaangstrom)), float(transition.A), atomic_number, ionization_stage, hc_in_ev_cm * float(energy_levels[i][
                                   levelid_from].energyabovegsinpercm), float(energy_levels[i][levelid_from].g), forbidden, transition.namefrom, transition.nameto, float(energy_levels[i][levelid_to].g), hc_in_ev_cm * float(energy_levels[i][levelid_to].energyabovegsinpercm), levelid_to in level_ids_with_permitted_down_transitions))

        ftransitiondata.write('{0:12d}{1:7d}{2:12d}{3:25.16E} {4:.1f}\n'.format(
            transitionid, levelid_from, levelid_to, float(transition.A), coll_str))
    ftransitiondata.write('\n')
    log_and_print('Included {0:d} transitions, of which {1:d} are forbidden'.format(
        len(transitions[i]) - 1, num_forbidden_transitions))


def write_phixs_data(fphixs, i, args):
    upper_level_ids_of_core_state_id = {}
    if i < len(listions) - 1:  # ignore the top ion
        log_and_print("writing to 'phixsdata2.txt'")
        flog.write('Downsampling cross sections assuming T={0} Kelvin\n'.format(
            args.optimaltemperature))
        for lowerlevelid in range(1, len(energy_levels[i])):
            upperionlevelids = []
            # find the upper level ids from the Nahar core state
            core_state_id = int(energy_levels[i][lowerlevelid].corestateid)
            if core_state_id > 0 and core_state_id < len(nahar_core_states[i]):

                if core_state_id not in upper_level_ids_of_core_state_id.keys():  # go find matching levels if they haven't been found yet
                    upper_level_ids_of_core_state_id[core_state_id] = []
                    nahar_core_state = nahar_core_states[i][core_state_id]
                    nahar_core_state_reduced_configuration = reduce_configuration(
                        nahar_core_state.configuration + '_' + nahar_core_state.term)

                    for upperlevelid in range(1, len(energy_levels[i + 1])):
                        upperlevel = energy_levels[i + 1][upperlevelid]
                        if hasattr(upperlevel, 'hilliername'):
                            upperlevelconfig = upperlevel.hilliername
                        else:
                            state_tuple = (int(upperlevel.twosplusone), int(upperlevel.l), int(
                                upperlevel.parity), int(upperlevel.indexinsymmetry))
                            if state_tuple in nahar_configurations[i + 1]:
                                upperlevelconfig = nahar_configurations[
                                    i + 1][state_tuple]
                            else:
                                upperlevelconfig = "-1"

                        if reduce_configuration(upperlevelconfig) == nahar_core_state_reduced_configuration:
                            upper_level_ids_of_core_state_id[
                                core_state_id].append(upperlevelid)
                            log_and_print("Matched core state {0} '{1}_{2}' to upper ion level {3} '{4}'".format(
                                core_state_id,
                                nahar_core_state.configuration,
                                nahar_core_state.term,
                                upperlevelid,
                                upperlevelconfig))

                    if len(upper_level_ids_of_core_state_id[core_state_id]) == 0:
                        upper_level_ids_of_core_state_id[core_state_id] = [1]
                        log_and_print("No upper level matched core state {0} '{1}_{2}' (reduced string: '{3}')".format(
                            core_state_id,
                            nahar_core_state.configuration,
                            nahar_core_state.term,
                            nahar_core_state_reduced_configuration))

                upperionlevelids = upper_level_ids_of_core_state_id[core_state_id]
            else:
                upperionlevelids = [1]

            # upperionlevelids = [1] #force ionization to ground state

            if upperionlevelids == [1]:
                fphixs.write('{0:12d}{1:12d}{2:8d}{3:12d}{4:8d}\n'.format(
                    atomic_number, ionization_stage + 1, 1, ionization_stage, lowerlevelid))
            else:
                fphixs.write('{0:12d}{1:12d}{2:8d}{3:12d}{4:8d}\n'.format(
                    atomic_number, ionization_stage + 1, -1, ionization_stage, lowerlevelid))
                fphixs.write('{0:8d}\n'.format(len(upperionlevelids)))

                summed_statistical_weights = sum(
                    [float(energy_levels[i + 1][id].g) for id in upperionlevelids])
                for upperionlevelid in sorted(upperionlevelids):
                    phixsprobability = float(
                        energy_levels[i + 1][upperionlevelid].g) / summed_statistical_weights
                    fphixs.write('{0:8d}{1:12f}\n'.format(
                        upperionlevelid, phixsprobability))

            for crosssection in photoionization_crosssections[i][lowerlevelid]:
                fphixs.write('{0:16.8E}\n'.format(crosssection))


if __name__ == "__main__":
    main()


"""
# this is out of date, so make sure this produces valid output before using
def read_hillier_phixs_tables(hillier_ion_folder, path_nahar_px_file):
    with open(hillier_ion_folder + path_nahar_px_file,'r') as fhillierphot:
        upperlevelid = -1
        truncatedlowerlevelname = ''
        numpointsexpected = 0
        crosssectiontype = '-1'
        seatonfittingcoefficients = []

        for line in fhillierphot:
            row = line.split()
#            print(row)

            if len(row) >= 2 and ' '.join(row[1:]) == '!Final state in ion':
                #upperlevelid = level_ids_of_energy_level_name_no_brackets[row[0]][0]
                upperlevelid = -1

            if len(row) >= 2 and ' '.join(row[1:]) == '!Configuration name':
                truncatedlowerlevelname = row[0]
                for lowerlevelid in level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname]:
                    photoionization_crosssections[i][lowerlevelid] = []
                seatonfittingcoefficients = []
                numpointsexpected = 0

            if len(row) >= 2 and ' '.join(row[1:]) == '!Number of cross-section points':
                numpointsexpected = int(row[0])

            if len(row) >= 2 and ' '.join(row[1:]) == '!Cross-section unit' and row[0] != 'Megabarns':
                    print('Wrong cross-section unit: ' + row[0])
                    sys.exit()

            if crosssectiontype in ['20','21'] and len(row) == 2 and all(map(isfloat, row)) and lowerlevelid != -1:
                for lowerlevelid in level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname]:
                    #WOULD NEED TO FIX THIS LINE SO THAT ENERGY IS IN ABSOLUTE UNITS OF RYDBERG FOR THIS TO WORK
                    photoionization_crosssections[i][lowerlevelid].append( (float(row[0].replace('D','E')),float(row[1].replace('D','E'))) )
                    if (len(photoionization_crosssections[i][lowerlevelid]) > 1 and
                       photoionization_crosssections[i][lowerlevelid][-1][0] <= photoionization_crosssections[i][lowerlevelid][-2][0]):
                        print('ERROR: photoionization table first column not monotonically increasing')
                        sys.exit()
            #elif True: #if want to ignore below types
                #pass
            elif crosssectiontype == '1' and len(row) == 1 and isfloat(row[0]) and numpointsexpected > 0:
                seatonfittingcoefficients.append(float(row[0].replace('D', 'E')))
                if len(seatonfittingcoefficients) == 3:
                    (sigmat,beta,s) = seatonfittingcoefficients
                    for lowerlevelid in level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname]:
                        for c in np.arange(0,1.0,0.01):
                            energydivthreshold = 1 + 20 * (c ** 2)
                            thresholddivenergy = energydivthreshold ** -1
                            crosssection = sigmat * (beta + (1 - beta)*(thresholddivenergy)) * (thresholddivenergy ** s)
                            photoionization_crosssections[i][lowerlevelid].append( (energydivthreshold*thresholddivenergy/ryd_to_ev, crosssection) )
    #                    print('Using Seaton formula values for lower level {0:d}'.format(lowerlevelid))
                    numpointsexpected = len(photoionization_crosssections[i][lowerlevelid])
            elif crosssectiontype == '7' and len(row) == 1 and isfloat(row[0]) and numpointsexpected > 0:
                seatonfittingcoefficients.append(float(row[0].replace('D', 'E')))
                if len(seatonfittingcoefficients) == 4:
                    (sigmat,beta,s,nuo) = seatonfittingcoefficients
                    for lowerlevelid in level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname]:
                        for c in np.arange(0,1.0,0.01):
                            energydivthreshold = 1 + 20 * (c ** 2)
                            thresholdenergyev = hc_in_ev_angstrom / float(hillier_energy_levels[i][lowerlevelid].lambdaangstrom)
                            energyoffsetdivthreshold = energydivthreshold + (nuo*1e15*h_in_ev_seconds)/thresholdenergyev
                            thresholddivenergyoffset = energyoffsetdivthreshold ** -1
                            if thresholddivenergyoffset < 1.0:
                                crosssection = sigmat * (beta + (1 - beta)*(thresholddivenergyoffset)) * (thresholddivenergyoffset ** s)
                            else:
                                crosssection = 0

                            photoionization_crosssections[i][lowerlevelid].append( (energydivthreshold*thresholddivenergy/ryd_to_ev, crosssection) )
    #                    print('Using modified Seaton formula values for lower level {0:d}'.format(lowerlevelid))
                    numpointsexpected = len(photoionization_crosssections[i][lowerlevelid])

            if len(row) >= 2 and ' '.join(row[1:]) == '!Type of cross-section':
                crosssectiontype = row[0]
                if crosssectiontype not in ['1','7','20','21']:
                    if crosssectiontype != '-1':
                        print('Warning: Unknown cross-section type: "{0}"'.format(crosssectiontype))
#                                sys.exit()
                    truncatedlowerlevelname = ''
                    crosssectiontype = '-1'
                    numpointsexpected = 0

            if len(row) == 0:
                if (truncatedlowerlevelname != '' and
                    numpointsexpected != len(photoionization_crosssections[i][level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname][0]])):
                    print('photoionization_crosssections mismatch: expecting {0:d} rows but found {1:d}'.format(
                        numpointsexpected,len(photoionization_crosssections[i][level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname][0]])))
                    print('A={0}, ionization_stage={1}, lowerlevel={2}, crosssectiontype={3}'.format(
                        atomic_number,ionization_stage,truncatedlowerlevelname,crosssectiontype))
                    print('matching level ids: ',level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname])
                    print(photoionization_crosssections[i][level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname][0]])
                    sys.exit()
                seatonfittingcoefficients = []
                truncatedlowerlevelname = ''
                numpointsexpected = 0
    return
"""

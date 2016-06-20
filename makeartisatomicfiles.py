#!/usr/bin/env python3
from collections import namedtuple, defaultdict
import os
import sys
import math
import argparse

from astropy import constants as const
from astropy import units as u
import numpy as np
import pandas as pd
from scipy import integrate
# from scipy import interpolate

PYDIR = os.path.dirname(os.path.abspath(__file__))
elsymbols = ['n'] + list(pd.read_csv(os.path.join(PYDIR, 'elements.csv'))['symbol'].values)

roman_numerals = (
    '', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
    'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX')

elsymboltohilliercode = {
    'H': 'HYD', 'He': 'HE', 'C': 'CARB', 'N': 'NIT',
    'O': 'OXY', 'F': 'FLU', 'Ne': 'NEON', 'Na': 'SOD',
    'Mg': 'MG', 'Al': 'ALUM', 'Si': 'SIL', 'P': 'PHOS',
    'S': 'SUL', 'Cl': 'CHL', 'Ar': 'ARG', 'K': 'POT',
    'Ca': 'CAL', 'Sc': 'SCAN', 'Ti': 'TIT', 'V': 'VAN',
    'Cr': 'CHRO', 'Mn': 'MAN', 'Fe': 'FE', 'Co': 'COB',
    'Ni': 'NICK'
}

nahar_configuration_replacements = {
    'Eqv st (0S ) 0s  a3P':  '2s2_2p4_3Pe',  # O I groundstate
    '2s22p2 (3P ) 0s  z4So': '2s2_2p3_4So',  # O II groundstate
    '2s22p  (2Po) 0s  a3P':  '2s2_2p2_3Pe',  # O III groundstate
    'Eqv st (0S ) 0s  a5D':  '3d6_4s2_5De',  # Fe I groundstate
    '3d5    (6S ) 0s  a5D':  '3d6_5De',  # Fe III groundstate
    'Eqv st (0S ) 0s  a4P':  '3d5_4Pe',  # Fe III level
    'Eqv st (0S ) 0s  a4F': '3d5_4Fe',  # Fe III level
    'Eqv st (0S ) 0s  a6S':  '3d5_6Se',  # Fe IV groundstate
    'Eqv st (0S ) 0s  a4G':  '3d5_4Ge'  # Fe IV state
}

# need to also include collision strengths from e.g., o2col.dat

path_hillier_osc_file = {
    (8, 1): '20sep11/oi_osc_mchf',
    (8, 2): '23mar05/o2osc_fin.dat',
    (8, 3): '15mar08/oiiiosc',
    (26, 1): '29apr04/fei_osc',
    (26, 2): '16nov98/fe2osc_nahar_kurucz.dat',
    (26, 3): '30oct12/FeIII_OSC',
    (26, 4): '18oct00/feiv_osc_rev2.dat',
    (26, 5): '18oct00/fev_osc.dat',
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
}

listelements = [
    (26, [1, 2, 3, 4, 5]),
    # (27, [3])
]

ryd_to_ev = u.rydberg.to('eV')

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
    parser.add_argument(
        '-output_folder', action='store',
        default='artis_files', help='Folder for output files')
    parser.add_argument(
        '-output_folder_logs', action='store',
        default='atomic_data_logs', help='Folder for log files')
    parser.add_argument(
        '-output_folder_transition_guide', action='store',
        default='transition_guide', help='')
    parser.add_argument(
        '-nphixspoints', type=int, default=100,
        help='Number of cross section points to save in output')
    parser.add_argument(
        '-nphixsnuincrement', type=float, default=0.1,
        help='Fraction of nu_edge incremented for each cross section point')
    parser.add_argument(
        '-integralstepfactor', type=int, default=4,
        help='Proportional to the number of steps in the integral for'
             'downsampling cross sections')
    parser.add_argument(
        '-optimaltemperature', type=int, default=3000,
        help='Temperature to use when downsampling cross sections')

    args = parser.parse_args()

    clear_files(args)
    process_files(args)


def clear_files(args):
    # clear out the file contents, so these can be appended to later
    with open(os.path.join(args.output_folder, 'adata.txt'), 'w'), \
            open(os.path.join(args.output_folder, 'transitiondata.txt'), 'w'), \
            open(os.path.join(args.output_folder, 'phixsdata_v2.txt'), 'w') as fphixs:
        fphixs.write('{0:d}\n'.format(args.nphixspoints))
        fphixs.write('{0:14.7e}\n'.format(args.nphixsnuincrement))


def process_files(args):
    global flog

    # for hillierelname in ('IRON',):
    #    d = './hillieratomic/' + hillierelname

    for elementindex, (atomic_number, listions) in enumerate(listelements):
        if not listions:
            continue

        nahar_core_states = [['IGNORE'] for x in listions]  # list of named tuples (naharcorestaterow)

        # keys are (2S+1, L, parity), values are strings of electron configuration
        nahar_configurations = [{} for x in listions]

        # keys are (2S+1, L, parity, indexinsymmetry), values are lists of (energy
        # in Rydberg, cross section in Mb) tuples
        nahar_phixs_tables = [{} for x in listions]

        ionization_energy_ev = [0.0 for x in listions]

        # list of named tuples (hillier_transition_row)
        transitions = [[] for x in listions]
        transition_count_of_level_name = [{} for x in listions]

        energy_levels = [[] for x in listions]
        # keys are hillier level ids, values are lists of (energy in Rydberg,
        # cross section in Mb) tuples
        photoionization_crosssections = [[] for x in listions]

        for i, ion_stage in enumerate(listions):
            energy_levels[i], transitions[i] = read_qub_levels_and_transitions(atomic_number, ion_stage)
            if len(energy_levels[i]) > 1:
                pass
            else:
                with open(os.path.join(args.output_folder, args.output_folder_logs,
                                       '{0}{1:d}.txt'.format(elsymbols[atomic_number].lower(), ion_stage)),
                          'w') as flog:

                    hillier_ion_folder = 'atomic-data-hillier/atomic/' + \
                        atomic_number_to_hillier_code[atomic_number] + \
                        '/' + roman_numerals[ion_stage] + '/'

                    log_and_print('==============> {0} {1}:'.format(
                        elsymbols[atomic_number], roman_numerals[ion_stage]))

                    path_nahar_energy_file = 'atomic-data-nahar/{0}{1:d}.en.ls.txt'.format(
                        elsymbols[atomic_number].lower(), ion_stage)

                    log_and_print('Reading ' + path_nahar_energy_file)
                    (nahar_energy_levels, nahar_core_states[i],
                     nahar_level_index_of_state, nahar_configurations[i]) = \
                        read_nahar_energy_level_file(path_nahar_energy_file, atomic_number, i, ion_stage)

                    if (atomic_number, ion_stage) in path_hillier_osc_file:
                        osc_file = path_hillier_osc_file[(atomic_number, ion_stage)]
                        log_and_print('Reading ' + hillier_ion_folder + osc_file)
                        (ionization_energy_ev[i], hillier_energy_levels,
                         hillier_transitions,
                         transition_count_of_level_name[i],
                         hillier_level_ids_matching_term) = \
                            read_hillier_levels_and_transitions_file(
                                hillier_ion_folder + osc_file,
                                hillier_row_format_energy_level[(atomic_number, ion_stage)], i)

                    if i < len(listions) - 1:  # don't get cross sections for top ion
                        path_nahar_px_file = 'atomic-data-nahar/{0}{1:d}.px.txt'.format(
                            elsymbols[atomic_number].lower(), ion_stage)

                        log_and_print('Reading ' + path_nahar_px_file)
                        nahar_phixs_tables[i] = read_nahar_phixs_tables(
                            path_nahar_px_file, atomic_number, i, ion_stage)

                    (energy_levels[i], transitions[i],
                     photoionization_crosssections[i]) = \
                        combine_hillier_nahar(
                            i, hillier_energy_levels,
                            hillier_level_ids_matching_term,
                            hillier_transitions,
                            nahar_energy_levels,
                            nahar_level_index_of_state,
                            nahar_configurations[i], nahar_phixs_tables[i],
                            args)

                    # Alternatively use Hillier phixs tables, but BEWARE this
                    # probably doesn't work anymore since the code has changed a lot
                    # print('Reading ' + hillier_ion_folder + path_hillier_phixs_file[(atomic_number, ion_stage)])
                    # read_hillier_phixs_tables(hillier_ion_folder,path_hillier_phixs_file[(atomic_number, ion_stage)])

        write_output_files(elementindex, energy_levels, transitions,
                           ionization_energy_ev,
                           transition_count_of_level_name,
                           nahar_core_states, nahar_configurations,
                           photoionization_crosssections, args)


def read_nahar_energy_level_file(path_nahar_energy_file, atomic_number, i, ion_stage):
    nahar_energy_level_row = namedtuple(
        'energylevel', 'indexinsymmetry TC corestateid elecn elecl energyreltoionpotrydberg twosplusone l parity energyabovegsinpercm g naharconfiguration')
    naharcorestaterow = namedtuple(
        'naharcorestate', 'nahar_core_state_id configuration term energyrydberg')
    nahar_configurations = {}
    nahar_energy_levels = ['IGNORE']
    nahar_level_index_of_state = {}
    nahar_core_states = []
    if os.path.isfile(path_nahar_energy_file):
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

            nahar_core_states = [()] * (numberofcorestates + 1)
            for c in range(1, numberofcorestates + 1):
                row = fenlist.readline().split()
                nahar_core_states[c] = naharcorestaterow(
                    int(row[0]), row[1], row[2], float(row[3]))
                if int(nahar_core_states[c].nahar_core_state_id) != c:
                    print('Nahar levels mismatch: id {0:d} found at entry number {1:d}'.format(
                        c, int(nahar_core_states[c].nahar_core_state_id)))
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
                    nahar_ionization_potential_rydberg = -float(line.split('=')[2])
                    flog.write('Ionization potential = {0:.4f} eV\n'.format(
                        nahar_ionization_potential_rydberg * ryd_to_ev))

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
                    nahar_configurations[(twosplusone, l, parity, indexinsymmetry)] = state
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

            if atomic_number != int(row[0]) or ion_stage != int(row[0]) - int(row[1]):
                print('Wrong atomic number or ionization stage in Nahar energy file',
                      atomic_number, int(row[0]), ion_stage, int(row[0]) - int(row[1]))
                sys.exit()

            while True:
                line = fenlist.readline()
                if not line:
                    print('End of file before table iii finished')
                    sys.exit()
                row = line.split()
                if row == ['0', '0', '0', '0']:
                    break
                twosplusone = int(row[0])
                l = int(row[1])
                parity = int(row[2])
                number_of_states_in_symmetry = int(row[3])

                line = fenlist.readline()  # core state number and energy

                for _ in range(number_of_states_in_symmetry):
                    row = fenlist.readline().split()
                    indexinsymmetry = int(row[0])
                    nahar_core_state_id = int(row[2])
                    if nahar_core_state_id < 1 or nahar_core_state_id > len(nahar_core_states):
                        flog.write("Core state id of {0:d}{1}{2} index {3:d} is invalid (={4:d}, Ncorestates={5:d}). Setting core state to 1 instead.\n".format(
                            twosplusone, lchars[l], ['e', 'o'][parity],
                            indexinsymmetry, nahar_core_state_id,
                            len(nahar_core_states)))
                        nahar_core_state_id = 1

                    nahar_energy_levels.append(nahar_energy_level_row(*row, twosplusone, l, parity, -1.0, 0, ''))

                    energyabovegsinpercm = \
                        (nahar_ionization_potential_rydberg +
                         float(nahar_energy_levels[-1].energyreltoionpotrydberg)) * \
                        ryd_to_ev / hc_in_ev_cm

                    nahar_energy_levels[-1] = nahar_energy_levels[-1]._replace(
                        indexinsymmetry=indexinsymmetry,
                        corestateid=nahar_core_state_id,
                        energyreltoionpotrydberg=float(nahar_energy_levels[-1].energyreltoionpotrydberg),
                        energyabovegsinpercm=energyabovegsinpercm,
                        g=twosplusone * (2 * l + 1)
                    )
                    nahar_level_index_of_state[(twosplusone, l, parity, indexinsymmetry)] = len(nahar_energy_levels) - 1

    #                if float(nahar_energy_levels[-1].energyreltoionpotrydberg) >= 0.0:
    #                    nahar_energy_levels.pop()

    return (nahar_energy_levels, nahar_core_states, nahar_level_index_of_state, nahar_configurations)


def read_hillier_levels_and_transitions_file(path_hillier_osc_file,
                                             hillier_row_format_energy_level, i):
    hillier_energy_level_row = namedtuple(
        'energylevel', hillier_row_format_energy_level + ' corestateid twosplusone l parity indexinsymmetry naharconfiguration matchscore')
    hillier_transition_row = namedtuple(
        'transition', 'namefrom nameto f A lambdaangstrom i j hilliertransitionid')
    hillier_energy_levels = ['IGNORE']
    hillier_level_ids_matching_term = defaultdict(list)
    transition_count_of_level_name = defaultdict(int)
    hillier_ionization_energy_ev = 0.0
    transitions = ['IGNORE']

    if os.path.isfile(path_hillier_osc_file):
        with open(path_hillier_osc_file, 'r') as fhillierosc:
            for line in fhillierosc:
                row = line.split()

                # check for right number of columns and that are all numbers except first column
                if len(row) == len(hillier_row_format_energy_level.split()) and all(map(isfloat, row[1:])):
                    hillier_energy_level = hillier_energy_level_row(*row, 0, -1, -1, -1, -1, '', -1)

                    hillierlevelid = int(hillier_energy_level.hillierlevelid.lstrip('-'))
                    levelname = hillier_energy_level.levelname

                    (twosplusone, l, parity) = get_term_as_tuple(levelname)
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
                        log_and_print("Can't find term in Hillier level name '" + levelname + "'")
                    else:
                        if levelname not in hillier_level_ids_matching_term[(twosplusone, l, parity)]:
                            hillier_level_ids_matching_term[
                                (twosplusone, l, parity)].append(hillierlevelid)

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

                if len(row) == 8 and all(map(isfloat, row[2:4])):
                    transition = hillier_transition_row(row[0], row[1],
                                                        float(row[2]),  # f
                                                        float(row[3]),  # A
                                                        float(row[4]),  # lambda
                                                        int(row[5]),  # i
                                                        int(row[6]),  # j
                                                        int(row[7]))

                    if True:  # or int(transition.hilliertransitionid) not in defined_transition_ids: #checking for duplicates massively slows down the code
                        #                    defined_transition_ids.append(int(transition.hilliertransitionid))
                        transitions.append(transition)
                        transition_count_of_level_name[transition.namefrom] += 1
                        transition_count_of_level_name[transition.nameto] += 1

                        if int(transition.hilliertransitionid) != len(transitions) - 1:
                            print(path_hillier_osc_file + ', WARNING: Transition id {0:d} found at entry number {1:d}'.format(
                                int(transition.hilliertransitionid), len(transitions) - 1))
                            sys.exit()
                    else:
                        log_and_print('FATAL: multiply-defined Hillier transition: {0} {1}'
                                      .format(transition.namefrom, transition.nameto))
                        sys.exit()
        log_and_print('Read {:d} transitions'.format(len(transitions) - 1))

    return hillier_ionization_energy_ev, hillier_energy_levels, transitions, transition_count_of_level_name, hillier_level_ids_matching_term


def read_nahar_phixs_tables(path_nahar_px_file, atomic_number, i, ion_stage):
    nahar_phixs_tables = {}
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
        if atomic_number != int(row[0]) or ion_stage != int(row[0]) - int(row[1]):
            print('Wrong atomic number or ionization stage in Nahar file',
                  atomic_number, int(row[0]), ion_stage,
                  int(row[0]) - int(row[1]))
            sys.exit()

        while True:
            line = fenlist.readline()
            row = line.split()

            if not line or sum(map(float, row)) == 0:
                break

            twosplusone, l, parity, indexinsymmetry = int(row[0]), int(row[1]), int(row[2]), int(row[3])

            number_of_points = int(fenlist.readline().split()[1])
            nahar_binding_energy_rydberg = float(fenlist.readline().split()[0])

            nahar_phixs_tables[(twosplusone, l, parity, indexinsymmetry)] = \
                np.array([list(map(float, fenlist.readline().split())) for p in range(number_of_points)])
            #    np.genfromtxt(fenlist, max_rows=number_of_points)

    return nahar_phixs_tables


def read_qub_levels_and_transitions(atomic_number, ion_stage):
    qub_energy_level_row = namedtuple(
        'energylevel', 'levelname qub_id twosplusone l j energyabovegsinpercm g parity')
    qub_transition_row = namedtuple(
        'transition', 'qub_id_up qub_id_lo A namefrom nameto lambdaangstrom')

    qub_energylevels = ['IGNORE']
    qub_transitions = ['IGNORE']
    if (atomic_number == 27) and (ion_stage == 3):
        with open('atomic-data-qub/adf04_v1', 'r') as fleveltrans:
            line = fleveltrans.readline()
            while True:
                line = fleveltrans.readline()
                if not line or line.startswith('   -1'):
                    break
                energylevel = qub_energy_level_row(
                    line[5:21], int(line[:5]), int(line[25:26]),
                    int(line[27:28]), float(line[29:33]), float(line[34:55]),
                    0.0, 0)  # TODO: PARITY IS WRONG!
                g = energylevel.twosplusone * (2 * energylevel.l + 1)
                energylevel = energylevel._replace(g=g)
                qub_energylevels.append(energylevel)

        with open('atomic-data-qub/adf04rad_v1', 'r') as ftrans:
            for line in ftrans:
                row = line.split()
                id_upper = int(row[0])
                id_lower = int(row[1])
                A = float(row[2])
                namefrom = qub_energylevels[id_upper].levelname
                nameto = qub_energylevels[id_lower].levelname
                lamdaangstrom = 1.e8 / (qub_energylevels[id_upper].energyabovegsinpercm - qub_energylevels[id_lower].energyabovegsinpercm)
                transition = qub_transition_row(id_upper, id_lower, A, namefrom, nameto, lamdaangstrom)
                qub_transitions.append(transition)

    return qub_energylevels, qub_transitions


def combine_hillier_nahar(i, hillier_energy_levels, hillier_level_ids_matching_term,
                          hillier_transitions,
                          nahar_energy_levels, nahar_level_index_of_state, nahar_configurations, nahar_phixs_tables, args):
    # hillier_energy_levels[i] = ['IGNORE'] #TESTING only
    # hillier_level_ids_matching_term[i] = {} #TESTING only
    added_nahar_levels = []
    photoionization_crosssections = []
    # match up Nahar states given in phixs data with Hillier levels, adding
    # missing levels as necessary
    for (twosplusone, l, parity, indexinsymmetry) in nahar_phixs_tables:
        state_tuple = (twosplusone, l, parity, indexinsymmetry)
        hillier_level_ids_matching_this_nahar_state = []

        nahar_configuration_this_state = '_CONFIG NOT FOUND_'
        flog.write("\n")
        if state_tuple in nahar_configurations:
            nahar_configuration_this_state = nahar_configurations[state_tuple]

            if nahar_configuration_this_state.strip() in nahar_configuration_replacements:
                nahar_configuration_this_state = nahar_configuration_replacements[
                    nahar_configurations[state_tuple].strip()]
                log_and_print("Replacing Nahar configuration of '{0}' with '{1}'".format(
                    nahar_configurations[state_tuple], nahar_configuration_this_state))

        if hillier_level_ids_matching_term[(twosplusone, l, parity)]:
                # match the electron configurations
            if nahar_configuration_this_state != '_CONFIG NOT FOUND_':
                best_match_score = 0.
                for levelid in hillier_level_ids_matching_term[(twosplusone, l, parity)]:
                    match_score = score_config_match(hillier_energy_levels[levelid].levelname, nahar_configuration_this_state)
                    best_match_score = max(best_match_score, match_score)
                if best_match_score > 0.:
                    for levelid in hillier_level_ids_matching_term[(twosplusone, l, parity)]:
                        levelname = hillier_energy_levels[levelid].levelname
                        match_score = score_config_match(hillier_energy_levels[levelid].levelname, nahar_configuration_this_state)
                        # if reduce_configuration(levelname) == reduce_configuration(nahar_configuration_this_state) and \
                        if match_score == best_match_score and \
                                hillier_energy_levels[levelid].indexinsymmetry < 1:  # make sure this Hillier level hasn't already been matched to a Nahar state

                            core_state_id = nahar_energy_levels[nahar_level_index_of_state[state_tuple]].corestateid

                            confignote = nahar_configurations[state_tuple]

                            if nahar_configuration_this_state != confignote:
                                confignote += " manually replaced by {:}".format(nahar_configuration_this_state)

                            hillier_energy_levels[levelid] = hillier_energy_levels[levelid]._replace(
                                twosplusone=twosplusone, l=l, parity=parity,
                                indexinsymmetry=indexinsymmetry,
                                corestateid=core_state_id,
                                naharconfiguration=confignote,
                                matchscore=match_score)
                            hillier_level_ids_matching_this_nahar_state.append(levelid)
            else:
                log_and_print("No electron configuration for {0:d}{1}{2} index {3:d}".format(
                    twosplusone, lchars[l], ['e', 'o'][parity], indexinsymmetry))
        else:
            flog.write("No Hillier levels with term {0:d}{1}{2}\n".format(
                twosplusone, lchars[l], ['e', 'o'][parity]))

        if not hillier_level_ids_matching_this_nahar_state:
            naharthresholdrydberg = nahar_phixs_tables[state_tuple][0][0]
            flog.write("No matched Hillier levels for Nahar cross section of {0:d}{1}{2} index {3:d} '{4}' ".format(
                twosplusone, lchars[l], ['e', 'o'][parity], indexinsymmetry,
                nahar_configuration_this_state))

            # now find the Nahar level and add it to the new list
            if state_tuple in nahar_level_index_of_state:
                nahar_energy_level = nahar_energy_levels[nahar_level_index_of_state[state_tuple]]
                energy_eV = nahar_energy_level.energyabovegsinpercm * hc_in_ev_cm
                flog.write('(E = {0:.3f} eV, g = {1:.1f})'.format(energy_eV, nahar_energy_level.g) + "\n")

                if energy_eV < 0.002:
                    flog.write(" but prevented duplicating the ground state\n")
                else:
                    added_nahar_levels.append(nahar_energy_level._replace(naharconfiguration=nahar_configuration_this_state))
            else:
                flog.write(" (and no matching entry in Nahar energy table, so can't be added)\n")
        else:  # there are Hillier levels matched to this state
            nahar_energy_level = nahar_energy_levels[nahar_level_index_of_state[state_tuple]]
            nahar_energyabovegsinev = hc_in_ev_cm * \
                nahar_energy_level.energyabovegsinpercm
            # avghillierthreshold = weightedavgthresholdinev(
            #    hillier_energy_levels, hillier_level_ids_matching_this_nahar_state)
            # strhilliermatchesthreshold = '[' + ', '.join(
            #     ['{0} ({1:.3f} eV)'.format(hillier_energy_levels[k].levelname,
            #                                hc_in_ev_angstrom / float(hillier_energy_levels[k].lambdaangstrom))
            #      for k in hillier_level_ids_matching_this_nahar_state]) + ']'

            flog.write("Matched Nahar phixs for {0:d}{1}{2} index {3:d} '{4}' (E = {5:.3f} eV, g = {6:.1f}) to \n".format(
                twosplusone, lchars[l], ['e', 'o'][parity], indexinsymmetry, nahar_configuration_this_state, nahar_energyabovegsinev, nahar_energy_level.g))

            if len(hillier_level_ids_matching_this_nahar_state) > 1:
                avghillierenergyabovegsinev = weightedavgenergyinev(hillier_energy_levels, hillier_level_ids_matching_this_nahar_state)
                sumhillierstatweights = sum([hillier_energy_levels[levelid].g for levelid in hillier_level_ids_matching_this_nahar_state])
                flog.write('<E> = {0:.3f} eV, sum(g) = {1:.1f}: \n'.format(avghillierenergyabovegsinev, sumhillierstatweights))

            strhilliermatches = '\n'.join(['{0} ({1:.3f} eV, g = {2:.1f}, match_score = {3:.1f})'.format(hillier_energy_levels[k].levelname, hc_in_ev_cm * float(
                hillier_energy_levels[k].energyabovegsinpercm), hillier_energy_levels[k].g, hillier_energy_levels[k].matchscore) for k in hillier_level_ids_matching_this_nahar_state])

            flog.write(strhilliermatches + '\n')

    energy_levels = hillier_energy_levels + added_nahar_levels

    log_and_print('Included {0} levels from Hillier dataset and added {1} levels from Nahar phixs tables for a total of {2} levels'.format(
        len(hillier_energy_levels) - 1, len(added_nahar_levels), len(energy_levels) - 1))

    # sort the concatenated energy level list by energy
    print('Sorting level list...')
    energy_levels.sort(key=lambda x: float(getattr(x, 'energyabovegsinpercm', '-inf')))

    # generate a mapping of level names to the final output level id numbers

    photoionization_crosssections = np.zeros((len(energy_levels), args.nphixspoints))

    print('Processing phixs tables...')
    # process the phixs tables and attach them to any matching levels in the output list
    reduced_phixs_list = reduce_phixs_tables(nahar_phixs_tables, args)
    for (twosplusone, l, parity, indexinsymmetry), phixstable in reduced_phixs_list.items():
        foundamatch = False
        for levelid, energylevel in enumerate(energy_levels[1:], 1):
            if (int(energylevel.twosplusone) == twosplusone and
                    int(energylevel.l) == l and
                    int(energylevel.parity) == parity and
                    int(energylevel.indexinsymmetry) == indexinsymmetry):
                photoionization_crosssections[levelid] = phixstable
                foundamatch = True  # there could be more than one match, but this flags there being at least one

        if not foundamatch:
            log_and_print("No Hiller or Nahar state to match with photoionization crosssection of {0:d}{1}{2} index {3:d}".format(
                twosplusone, lchars[l], ['e', 'o'][parity], indexinsymmetry))
    return energy_levels, hillier_transitions, photoionization_crosssections


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
def reduce_phixs_tables(dicttables, args):
    dictout = {}

    ryd_to_hz = (u.rydberg / const.h).to('Hz').value
    h_over_kb_in_K_sec = (const.h / const.k_B).to('K s').value

    # proportional to recombination rate
    # nu0 = 1e13
    # fac = math.exp(h_over_kb_in_K_sec * nu0 / args.optimaltemperature)
    def integrand(nu):
        return (nu + nu) * math.exp(- h_over_kb_in_K_sec *
                                    (nu - nu0) / args.optimaltemperature)

    # def integrand_vec(nu_list):
    #    return [(nu + nu) * math.exp(- h_over_kb_in_K_sec *
    #                                 (nu - nu0) / args.optimaltemperature)
    #            for nu in nu_list]

    integrand_vec = np.vectorize(integrand)

    xgrid = np.linspace(1.0,
                        1.0 + args.nphixsnuincrement * (
                            args.nphixspoints + 1),
                        num=args.nphixspoints + 1, endpoint=False)

    for state_tuple, tablein in dicttables.items():
        # tablein is an array of pairs (energy, phixs cross section)

        nu0 = tablein[0][0] * ryd_to_hz

        arr_sigma_out = np.empty(args.nphixspoints)
        # x is nu/nu_edge

        # sigma_interp = interpolate.interp1d(tablein[:, 0], tablein[:, 1],
        #                                    kind='linear', assume_sorted=True)

        for i, _ in enumerate(xgrid[:-1]):
            enlow = xgrid[i] * tablein[0][0]
            enhigh = xgrid[i + 1] * tablein[0][0]

            nsteps = (args.nphixspoints - i + 1) * args.integralstepfactor

            # integralnosigma, err = integrate.fixed_quad(integrand_vec, enlow, enhigh, n=250)
            # integralwithsigma, err = integrate.fixed_quad(
            #    lambda x: sigma_interp(x) * integrand_vec(x), enlow, enhigh, n=250)

            # this is incredibly fast, but maybe not accurate
            # integralnosigma, err = integrate.quad(integrand, enlow, enhigh, epsrel=1e-2)
            # integralwithsigma, err = integrate.quad(
            #    lambda x: sigma_interp(x) * integrand(x), enlow, enhigh, epsrel=1e-2)

            arr_energyryd = np.linspace(enlow, enhigh,
                                        num=nsteps, endpoint=False)

            arr_sigma_megabarns = np.interp(arr_energyryd, tablein[:, 0],
                                            tablein[:, 1])

            integrand_vals = integrand_vec(arr_energyryd * ryd_to_hz)
            sigma_integrand_vals = [sigma * integrand_val
                                    for integrand_val, sigma
                                    in zip(integrand_vals, arr_sigma_megabarns)]

            integralnosigma = integrate.simps(integrand_vals, arr_energyryd)
            integralwithsigma = integrate.simps(sigma_integrand_vals, arr_energyryd)

            if integralnosigma > 0:
                arr_sigma_out[i] = (integralwithsigma / integralnosigma)
            else:
                arr_sigma_out[i] = 0

        dictout[state_tuple] = arr_sigma_out  # output a 1D list of cross sections

    return dictout


def weightedavgenergyinev(energylevels_thision, ids):
    genergysum = 0.0
    gsum = 0.0
    for levelid in ids:
        statisticalweight = float(energylevels_thision[levelid].g)
        genergysum += statisticalweight * hc_in_ev_cm * \
            float(energylevels_thision[levelid].energyabovegsinpercm)
        gsum += statisticalweight
    return genergysum / gsum


def weightedavgthresholdinev(energylevels_thision, ids):
    genergysum = 0.0
    gsum = 0.0
    for levelid in ids:
        statisticalweight = float(energylevels_thision[levelid].g)
        genergysum += statisticalweight * hc_in_ev_angstrom / \
            float(energylevels_thision[levelid].lambdaangstrom)
        gsum += statisticalweight
    return genergysum / gsum


alphabets = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
reversedalphabets = 'zyxwvutsrqponmlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA '
lchars = 'SPDFGHIKLMNOPQRSTUVWXYZ'


# reads a Hillier level name and returns the term
# tuple (twosplusone, l, parity)
def get_term_as_tuple(config):
    config = config.split('[')[0]
    for charpos, char in reversed(list(enumerate(config))):
        if char in lchars:
            lposition = charpos
            l = lchars.index(char)
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


# e.g. turn '(4F)' into (4, 3, -1)
# or '(4F1) into (4, 3, 1)
def interpret_parent_term(strin):
    strin = strin.strip('()')
    lposition = -1
    for charpos, char in reversed(list(enumerate(strin))):
        if char in lchars:
            lposition = charpos
            l = lchars.index(char)
            break
    if lposition < 0:
        return (-1, -1, -1)

    twosplusone = int(strin[:lposition].lstrip(alphabets))  # could this be two digits long?
    if lposition < len(strin)-1:
        jvalue = strin[lposition+1:]
    else:
        jvalue = -1
    return (twosplusone, l, jvalue)


# e.g. convert "3d64s  (6D ) 8p  j5Fo" to "3d64s8p_5Fo",
# similar to Hillier style "3d6(5D)4s8p_5Fo" but without the parent term
# (and mysterious letter before the term if present)
def reduce_configuration(instr):
    if instr == "-1":
        return "-1"
    instr = instr.split('[')[0]  # remove trailing bracketed J value
    if instr[-1] in lchars:
        instr = instr + 'e'  # last character being S,P,D, etc means even

    outstr = remove_bracketed_part(instr)
    outstr += '_'
    outstr += instr[-3:-1]
    if instr[-1] == 'o':
        outstr += 'o'
    else:
        outstr += 'e'
    return outstr


def remove_bracketed_part(instr):
    outstr = ""
    in_brackets = False
    for char in instr[:-4]:
        if char == ' ' or char == '_':
            continue
        elif char == '(':
            in_brackets = True
        elif char == ')':
            in_brackets = False
        elif not in_brackets:
            outstr += char
    return outstr


def interpret_configuration(instr_orig):
    max_n = 20  # maximum possible principle quantum number n
    instr = instr_orig
    instr = instr.split('[')[0]  # remove trailing bracketed J value

    if instr[-1] in lchars:
        term_parity = 0  # even
    else:
        term_parity = [0, 1][(instr[-1] == 'o')]
        instr = instr[:-1]

    term_twosplusone = -1
    term_l = -1
    indexinsymmetry = -1

    while instr:
        if instr[-1] in lchars:
            term_l = lchars.index(instr[-1])
            instr = instr[:-1]
            break
        else:
            term_parity = term_parity + 2  # this accounts for things like '3d7(4F)6d_5Pbe' in the Hiller levels. Shouldn't match these
        instr = instr[:-1]

    if str.isdigit(instr[-1]):
        term_twosplusone = int(instr[-1])
        instr = instr[:-1]

    if instr[-1] == '_':
        instr = instr[:-1]
    elif instr[-1] in alphabets and (len(instr) < 2 or not str.isdigit(instr[-2])):  # to catch e.g., '3d6(5D)6d4Ge[9/2]' 6d, not index d
        if term_parity == 1:
            indexinsymmetry = reversedalphabets.index(instr[-1]) + 1
        else:
            indexinsymmetry = alphabets.index(instr[-1]) + 1
        instr = instr[:-1]

    electron_config = []
    if not instr.startswith('Eqv st'):
        while instr:
            if instr[-1].upper() in lchars:
                try:
                    if len(instr) >= 3 and str.isdigit(instr[-3]) and int(instr[-3:-1]) < max_n:
                        startpos = -3
                    else:
                        startpos = -2
                except ValueError:
                    startpos = -3  # this tripped on '4sp(3P)_7Po[2]'. just pretend 4sp is an orbital and occupation number

                electron_config.insert(0, instr[startpos:])
                instr = instr[:startpos]
            elif instr[-1] == ')':
                left_bracket_pos = instr.rfind('(')
                str_parent_term = instr[left_bracket_pos:].replace(" ", "")
                electron_config.insert(0, str_parent_term)
                instr = instr[:left_bracket_pos]
            elif str.isdigit(instr[-1]):  # probably the number of electrons in an orbital
                if instr[-2].upper() in lchars:
                    if len(instr) >= 4 and str.isdigit(instr[-4]) and int(instr[-4:-2]) < max_n:
                        startpos = -4
                    else:
                        startpos = -3
                    electron_config.insert(0, instr[-3:])
                    instr = instr[:-3]
                else:
                    # print('Unknown character ' + instr[-1])
                    instr = instr[:-1]
            elif instr[-1] in ['_', ' ']:
                instr = instr[:-1]
            else:
                # print('Unknown character ' + instr[-1])
                instr = instr[:-1]

    # return '{0} {1}{2}{3} index {4}'.format(electron_config, term_twosplusone, lchars[term_l], ['e', 'o'][term_parity], indexinsymmetry)
    return electron_config, term_twosplusone, term_l, term_parity, indexinsymmetry


def score_config_match(config_a, config_b):
    electron_config_a, term_twosplusone_a, term_l_a, term_parity_a, indexinsymmetry_a = interpret_configuration(config_a)
    electron_config_b, term_twosplusone_b, term_l_b, term_parity_b, indexinsymmetry_b = interpret_configuration(config_b)

    if term_twosplusone_a != term_twosplusone_b or term_l_a != term_l_b or term_parity_a != term_parity_b:
        return 0
    elif indexinsymmetry_a != -1 and indexinsymmetry_b != -1:
        if indexinsymmetry_a == indexinsymmetry_b:
            return 100  # exact match between Hillier and Nahar
        else:
            return 0  # both correspond to Nahar states but do not match
    elif electron_config_a == electron_config_b:
        return 100
    elif len(electron_config_a) > 0 and len(electron_config_b) > 0:
        parent_term_match = 0.5  # 0 is definite mismatch, 0.5 is consistent, 1 is definite match
        parent_term_index_a, parent_term_index_b = -1, -1
        matched_pieces = 0
        index_a, index_b = 0, 0

        non_term_pieces_a = sum([1 for a in electron_config_a if not a.startswith('(')])
        non_term_pieces_b = sum([1 for b in electron_config_a if not b.startswith('(')])
        # go through the configuration piece by piece
        while index_a < len(electron_config_a) and index_b < len(electron_config_b):
            piece_a = electron_config_a[index_a]  # an orbital electron count or a parent term
            piece_b = electron_config_b[index_b]  # an orbital electron count or a parent term

            if piece_a.startswith('(') or piece_b.startswith('('):
                if piece_a.startswith('('):
                    if parent_term_index_a == -1:
                        parent_term_index_a = index_a
                    index_a += 1
                if piece_b.startswith('('):
                    if parent_term_index_b == -1:
                        parent_term_index_b = index_b
                    index_b += 1
            else:  # orbital occupation piece
                if piece_a == piece_b:
                    matched_pieces += 1
                elif '0s' in [piece_a, piece_b]:  # wildcard piece
                    pass
                else:
                    return 0
                index_a += 1
                index_b += 1

        if parent_term_index_a != -1 and parent_term_index_b != -1:
            parent_term_a = interpret_parent_term(electron_config_a[parent_term_index_a])
            parent_term_b = interpret_parent_term(electron_config_b[parent_term_index_b])
            if parent_term_index_a == parent_term_index_b:
                if parent_term_a == parent_term_b:
                    parent_term_match = 1.
                elif parent_term_a[:2] == parent_term_b[:2] and -1 in [parent_term_a[2], parent_term_b[2]]:  # e.g., '(3F1)' matches '(3F)'
                    # strip J values from the parent term
                    parent_term_match = 0.75
                else:
                    parent_term_match = 0.
                    return 0
            else:  # parent terms occur at different locations. are they consistent?
                if parent_term_index_b > parent_term_index_a:
                    orbitaldiff = electron_config_b[parent_term_index_a:parent_term_index_b]
                else:
                    orbitaldiff = electron_config_a[parent_term_index_b:parent_term_index_a]

                maxldiff = 0
                maxspindiff = 0  # two times s
                for orbital in orbitaldiff:
                    for pos, char in enumerate(orbital):
                        if char in lchars.lower():
                            maxldiff += lchars.lower().index(char)
                            occupation = orbital[pos+1:]
                            if len(occupation) > 0:
                                maxspindiff += int(occupation)
                            else:
                                maxspindiff += 1
                            break

                spindiff = abs(parent_term_a[0] - parent_term_b[0])
                ldiff = abs(parent_term_a[1] - parent_term_b[1])
                if spindiff > maxspindiff or ldiff > maxldiff:  # parent terms are inconsistent -> no match
                    # print(orbitaldiff, spindiff, maxspindiff, ldiff, maxldiff, config_a, config_b)
                    parent_term_match = 0.
                    return 0

        score = int(99 * matched_pieces / max(non_term_pieces_a, non_term_pieces_b) * parent_term_match)
        return score
    else:
        return 5  # term matches but no electron config available or it's an Eqv state...0s type

    print("WHAT?")
    sys.exit()
    return -1


def write_output_files(elementindex, energy_levels, transitions,
                       ionization_energies,
                       transition_count_of_level_name,
                       nahar_core_states, nahar_configurations,
                       photoionization_crosssections, args):
    global flog
    atomic_number, listions = listelements[elementindex]

    print('\nStarting output stage:')

    with open(os.path.join(args.output_folder, 'adata.txt'), 'a') as fatommodels, \
            open(os.path.join(args.output_folder, 'transitiondata.txt'), 'a') as ftransitiondata, \
            open(os.path.join(args.output_folder, 'phixsdata_v2.txt'), 'a') as fphixs, \
            open(os.path.join(args.output_folder_transition_guide, 'transitions_{}.txt'.format(elsymbols[atomic_number])), 'w') as ftransitionguide:

        ftransitionguide.write('{0:>16s} {1:>12s} {2:>3s} {3:>9s} {4:>17s} {5:>17s} {6:>10s} {7:25s} {8:25s} {9:>17s} {10:>17s} {11:>19s}\n'.format(
            'lambda_angstroms', 'A', 'Z', 'ion_stage', 'lower_energy_Ev', 'lower_statweight', 'forbidden', 'lower_level', 'upper_level', 'upper_statweight', 'upper_energy_Ev', 'upper_has_permitted'))

        for i, ion_stage in enumerate(listions):

            ionstr = '{0} {1}'.format(elsymbols[atomic_number],
                                      roman_numerals[ion_stage])

            flog = open(
                os.path.join(args.output_folder,
                             args.output_folder_logs,
                             '{0}{1:d}.txt'.format(elsymbols[atomic_number].lower(), ion_stage)),
                'a')

            print('==============> ' + ionstr + ':')

            write_adata(fatommodels, atomic_number, ion_stage, energy_levels[i], ionization_energies[i], transition_count_of_level_name[i], args)

            write_transition_data(ftransitiondata, ftransitionguide, atomic_number, ion_stage, energy_levels[i], transitions[i], args)

            if i < len(listions) - 1:  # ignore the top ion
                write_phixs_data(fphixs, i, atomic_number, ion_stage, energy_levels[i], energy_levels[i+1], nahar_core_states[i], nahar_configurations, photoionization_crosssections[i], args)


def write_adata(fatommodels, atomic_number, ion_stage, energy_levels, ionization_energy, transition_count_of_level_name, args):
    log_and_print("writing to 'adata.txt'")
    fatommodels.write('{0:12d}{1:12d}{2:12d}{3:15.7f}\n'.format(
        atomic_number, ion_stage,
        len(energy_levels) - 1,
        ionization_energy))

    for levelid, energylevel in enumerate(energy_levels[1:], 1):

        if hasattr(energylevel, 'levelname'):
            transitioncount = transition_count_of_level_name.get(
                energylevel.levelname, 0)
        else:
            transitioncount = 0

        level_comment = ""
        try:
            level_comment = "Hiller: '{:}', ".format(energylevel.levelname).ljust(35)
        except AttributeError:
            level_comment = " " * 35

        try:
            if energylevel.indexinsymmetry >= 0:
                level_comment += 'Nahar: {:d}{:}{:} index {:}'.format(
                    energylevel.twosplusone,
                    lchars[energylevel.l],
                    ['e', 'o'][energylevel.parity],
                    energylevel.indexinsymmetry)
                try:
                    level_comment += " '{:}'".format(energylevel.naharconfiguration)
                except AttributeError:
                    level_comment += ' (no config)'
        except AttributeError:
            pass

        fatommodels.write('{:7d}{:25.16f}{:25.16f}{:7d}     {:}\n'.format(
            levelid, hc_in_ev_cm * float(energylevel.energyabovegsinpercm),
            float(energylevel.g), transitioncount, level_comment))

    fatommodels.write('\n')


def write_transition_data(ftransitiondata, ftransitionguide, atomic_number, ion_stage, energy_levels, transitions, args):
    level_id_of_level_name = {}
    for levelid in range(1, len(energy_levels)):
        if hasattr(energy_levels[levelid], 'levelname'):
            level_id_of_level_name[energy_levels[levelid].levelname] = levelid

    log_and_print("writing to 'transitiondata.txt'")
    num_forbidden_transitions = 0
    ftransitiondata.write('{0:7d}{1:7d}{2:12d}\n'.format(
        atomic_number, ion_stage, len(transitions) - 1))

    level_ids_with_permitted_down_transitions = set()
    for transitionid in range(1, len(transitions)):
        transition = transitions[transitionid]
        levelid_from = level_id_of_level_name[transition.namefrom]
        levelid_to = level_id_of_level_name[transition.nameto]
        forbidden = (energy_levels[levelid_from].parity ==
                     energy_levels[levelid_to].parity)
        if not forbidden:
            level_ids_with_permitted_down_transitions.add(
                levelid_to)  # hopefully 'to_level' is the upper level

    for transitionid, transition in enumerate(transitions[1:], 1):
        levelid_from = level_id_of_level_name[transition.namefrom]
        levelid_to = level_id_of_level_name[transition.nameto]
        forbidden = (energy_levels[levelid_from].parity ==
                     energy_levels[levelid_to].parity)
        if forbidden:
            num_forbidden_transitions += 1
            coll_str = -2
            flog.write('Forbidden transition: lambda_angstrom= {:7.1f}, {:25s} to {:25s}\n'.format(
                float(transition.lambdaangstrom), transition.namefrom,
                transition.nameto))
        else:
            coll_str = -1

        if True:  # ion_stage in [1,2,3]:
            ftransitionguide.write('{0:16.1f} {1:12E} {2:3d} {3:9d} {4:17.2f} {5:17.4f} {6:10b} {7:25s} {8:25s} {9:17.2f} {10:17.4f} {11:19b}\n'.format(
                abs(float(transition.lambdaangstrom)), float(transition.A),
                atomic_number, ion_stage,
                hc_in_ev_cm * float(
                    energy_levels[levelid_from].energyabovegsinpercm),
                float(energy_levels[levelid_from].g), forbidden,
                transition.namefrom, transition.nameto,
                float(energy_levels[levelid_to].g),
                hc_in_ev_cm * float(
                    energy_levels[levelid_to].energyabovegsinpercm),
                levelid_to in level_ids_with_permitted_down_transitions))

        ftransitiondata.write(
            '{0:12d}{1:7d}{2:12d}{3:25.16E} {4:.1f}\n'.format(
                transitionid, levelid_from, levelid_to,
                float(transition.A), coll_str))

    ftransitiondata.write('\n')
    log_and_print(
        'Included {0:d} transitions, of which {1:d} are forbidden'.format(
            len(transitions) - 1, num_forbidden_transitions))


def write_phixs_data(fphixs, i, atomic_number, ion_stage, energy_levels,
                     energy_levels_upperion, nahar_core_states,
                     nahar_configurations, photoionization_crosssections,
                     args):
    log_and_print("writing to 'phixsdata2.txt'")
    flog.write('Downsampling cross sections assuming T={0} Kelvin\n'.format(
        args.optimaltemperature))
    upper_level_ids_of_core_state_id = defaultdict(list)

    for lowerlevelid, energy_level in enumerate(energy_levels[1:], 1):
        upperionlevelids = []
        # find the upper level ids from the Nahar core state
        upperionlevelids = get_photoion_upperlevelids(
            energy_level, energy_levels_upperion,
            nahar_core_states, nahar_configurations[i + 1],
            upper_level_ids_of_core_state_id)

        # upperionlevelids = [1] #force ionization to ground state

        if upperionlevelids == [1]:
            fphixs.write('{0:12d}{1:12d}{2:8d}{3:12d}{4:8d}\n'.format(
                atomic_number, ion_stage + 1, 1, ion_stage, lowerlevelid))
        else:
            fphixs.write('{0:12d}{1:12d}{2:8d}{3:12d}{4:8d}\n'.format(
                atomic_number, ion_stage + 1, -1, ion_stage, lowerlevelid))
            fphixs.write('{0:8d}\n'.format(len(upperionlevelids)))

            summed_statistical_weights = sum(
                [float(energy_levels_upperion[id].g) for id in upperionlevelids])
            for upperionlevelid in sorted(upperionlevelids):
                phixsprobability = (energy_levels_upperion[upperionlevelid].g /
                                    summed_statistical_weights)
                fphixs.write('{0:8d}{1:12f}\n'.format(
                    upperionlevelid, phixsprobability))

        for crosssection in photoionization_crosssections[lowerlevelid]:
            fphixs.write('{0:16.8E}\n'.format(crosssection))


def get_photoion_upperlevelids(energy_level, energy_levels_upperion,
                               nahar_core_states, nahar_configurations_upperion,
                               upper_level_ids_of_core_state_id):

    core_state_id = int(energy_level.corestateid)
    if core_state_id > 0 and core_state_id < len(nahar_core_states):

        if not upper_level_ids_of_core_state_id[core_state_id]:
            # go find matching levels if they haven't been found yet
            nahar_core_state = nahar_core_states[core_state_id]
            nahar_core_state_reduced_configuration = reduce_configuration(
                nahar_core_state.configuration + '_' + nahar_core_state.term)

            for upperlevelid, upperlevel in enumerate(energy_levels_upperion[1:], 1):
                if hasattr(upperlevel, 'levelname'):
                    upperlevelconfig = upperlevel.levelname
                else:
                    state_tuple = (int(upperlevel.twosplusone), int(upperlevel.l), int(
                        upperlevel.parity), int(upperlevel.indexinsymmetry))
                    upperlevelconfig = nahar_configurations_upperion.get(state_tuple, '-1')

                if reduce_configuration(upperlevelconfig) == nahar_core_state_reduced_configuration:
                    upper_level_ids_of_core_state_id[core_state_id].append(upperlevelid)
                    flog.write("Matched core state {0} '{1}_{2}' to upper ion level {3} '{4}'\n".format(
                        core_state_id,
                        nahar_core_state.configuration,
                        nahar_core_state.term,
                        upperlevelid,
                        upperlevelconfig))

            if not upper_level_ids_of_core_state_id[core_state_id]:
                upper_level_ids_of_core_state_id[core_state_id] = [1]
                log_and_print("No upper level matched core state {0} '{1}_{2}' (reduced string: '{3}')".format(
                    core_state_id,
                    nahar_core_state.configuration,
                    nahar_core_state.term,
                    nahar_core_state_reduced_configuration))

        upperionlevelids = upper_level_ids_of_core_state_id[core_state_id]
    else:
        upperionlevelids = [1]

    return upperionlevelids


if __name__ == "__main__":
    # print(interpret_configuration('3d64s_4H'))
    # print(interpret_configuration('3d6(3H)4sa4He[11/2]'))
    # print(score_config_match('3d64s_4H','3d6(3H)4sa4He[11/2]'))
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
                    print('A={0}, ion_stage={1}, lowerlevel={2}, crosssectiontype={3}'.format(
                        atomic_number,ion_stage,truncatedlowerlevelname,crosssectiontype))
                    print('matching level ids: ',level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname])
                    print(photoionization_crosssections[i][level_ids_of_energy_level_name_no_brackets[i][truncatedlowerlevelname][0]])
                    sys.exit()
                seatonfittingcoefficients = []
                truncatedlowerlevelname = ''
                numpointsexpected = 0
    return
"""

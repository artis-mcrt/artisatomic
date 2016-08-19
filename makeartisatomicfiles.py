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
from scipy import interpolate
from manual_matches import nahar_configuration_replacements, hillier_name_replacements
# from readnaharfiles import get_naharphotoion_upperlevelids, get_nahar_targetfractions

PYDIR = os.path.dirname(os.path.abspath(__file__))
elsymbols = ['n'] + list(pd.read_csv(os.path.join(PYDIR, 'elements.csv'))['symbol'].values)

roman_numerals = (
    '', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
    'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX'
)

elsymboltohilliercode = {
    'H': 'HYD', 'He': 'HE', 'C': 'CARB', 'N': 'NIT',
    'O': 'OXY', 'F': 'FLU', 'Ne': 'NEON', 'Na': 'SOD',
    'Mg': 'MG', 'Al': 'ALUM', 'Si': 'SIL', 'P': 'PHOS',
    'S': 'SUL', 'Cl': 'CHL', 'Ar': 'ARG', 'K': 'POT',
    'Ca': 'CAL', 'Sc': 'SCAN', 'Ti': 'TIT', 'V': 'VAN',
    'Cr': 'CHRO', 'Mn': 'MAN', 'Fe': 'FE', 'Co': 'COB',
    'Ni': 'NICK'
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

listelements = [
  # (8, [1, 2, 3]),
  (26, [1, 2, 3, 4, 5]),
  (27, [2, 3, 4])
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
        description='Produce an ARTIS atomic database by combining Hillier and Nahar data sets.')
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
        '-phixsnuincrement', type=float, default=0.1,
        help='Fraction of nu_edge incremented for each cross section point')
    parser.add_argument(
        '-optimaltemperature', type=int, default=3000,
        help='Temperature to use when downsampling cross sections')
    parser.add_argument(
        '--nophixs', action='store_true',
        help='Don''t generate cross sections and write to phixsdata_v2.txt file')

    args = parser.parse_args()

    log_folder = os.path.join(args.output_folder, args.output_folder_logs)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    clear_files(args)
    process_files(args)


def clear_files(args):
    # clear out the file contents, so these can be appended to later
    with open(os.path.join(args.output_folder, 'adata.txt'), 'w'), \
            open(os.path.join(args.output_folder, 'transitiondata.txt'), 'w'):
        if not args.nophixs:
            with open(os.path.join(args.output_folder, 'phixsdata_v2.txt'), 'w') as fphixs:
                fphixs.write('{0:d}\n'.format(args.nphixspoints))
                fphixs.write('{0:14.7e}\n'.format(args.phixsnuincrement))


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
        upsilondicts = [{} for x in listions]

        energy_levels = [[] for x in listions]
        # keys are level ids, values are lists of (energy in Rydberg,
        # cross section in Mb) tuples
        photoionization_crosssections = [[] for _ in listions]
        photoionization_targetfractions = [[] for _ in listions]

        for i, ion_stage in enumerate(listions):
            logfilepath = os.path.join(args.output_folder, args.output_folder_logs,
                                       '{0}{1:d}.txt'.format(elsymbols[atomic_number].lower(), ion_stage))
            hillier_ion_folder = ('atomic-data-hillier/atomic/' + atomic_number_to_hillier_code[atomic_number] +
                                  '/' + roman_numerals[ion_stage] + '/')

            if atomic_number == 26:
                upsilondatafilenames = {2: 'fe_ii_upsilon-data.txt', 3: 'fe_iii_upsilon-data.txt'}
                if ion_stage in upsilondatafilenames:
                    upsilondatadf = pd.read_csv(os.path.join('atomic-data-tiptopbase', upsilondatafilenames[ion_stage]),
                                                names=["Z", "ion_stage", "lower", "upper", "upsilon"],
                                                index_col=False, header=None, sep=" ")
                    if len(upsilondatadf) > 0:
                        for _, row in upsilondatadf.iterrows():
                            lower = int(row['lower'])
                            upper = int(row['upper'])
                            if upper <= lower:
                                print("Problem in {0}, lower {1} upper {2}. Skipping".format(upsilondatafilenames[ion_stage], lower, upper))
                            else:
                                if (lower, upper) not in upsilondicts[i]:
                                    upsilondicts[i][(lower, upper)] = row['upsilon']
                                else:
                                    log_and_print("Duplicate upsilon value for transition {0:d} to {1:d} keeping {2:5.2e} instead of using {3:5.2e}".format(
                                        lower, upper, upsilondicts[i][(lower, upper)], row['upsilon']))

            if atomic_number == 27:
                with open(logfilepath, 'w') as flog:
                    log_and_print('==============> {0} {1}:'.format(elsymbols[atomic_number], roman_numerals[ion_stage]))

                    if ion_stage in [3, 4]:
                        (ionization_energy_ev[i], energy_levels[i],
                         transitions[i], transition_count_of_level_name[i],
                         upsilondicts[i]) = read_qub_levels_and_transitions(atomic_number, ion_stage)
                    else:
                        osc_file = path_hillier_osc_file[(atomic_number, ion_stage)]
                        print('Reading ' + hillier_ion_folder + osc_file)

                        (ionization_energy_ev[i], energy_levels[i],
                         transitions[i], transition_count_of_level_name[i],
                         hillier_level_ids_matching_term) = read_hillier_levels_and_transitions(
                             hillier_ion_folder + osc_file,
                             hillier_row_format_energy_level[(atomic_number, ion_stage)], i)

                    if i < len(listions) - 1 and not args.nophixs:  # don't get cross sections for top ion
                        photoionization_crosssections[i], photoionization_targetfractions[i] = read_qub_photoionizations(atomic_number, ion_stage, energy_levels[i], args)

            else:
                with open(logfilepath, 'w') as flog:
                    log_and_print('==============> {0} {1}:'.format(elsymbols[atomic_number], roman_numerals[ion_stage]))

                    path_nahar_energy_file = 'atomic-data-nahar/{0}{1:d}.en.ls.txt'.format(
                        elsymbols[atomic_number].lower(), ion_stage)

                    (nahar_energy_levels, nahar_core_states[i],
                     nahar_level_index_of_state, nahar_configurations[i]) = read_nahar_energy_level_file(
                         path_nahar_energy_file, atomic_number, i, ion_stage)

                    if (atomic_number, ion_stage) in path_hillier_osc_file:
                        osc_file = path_hillier_osc_file[(atomic_number, ion_stage)]
                        log_and_print('Reading ' + hillier_ion_folder + osc_file)

                        (ionization_energy_ev[i], hillier_energy_levels,
                         hillier_transitions,
                         transition_count_of_level_name[i],
                         hillier_level_ids_matching_term) = \
                            read_hillier_levels_and_transitions(
                                hillier_ion_folder + osc_file,
                                hillier_row_format_energy_level[(atomic_number, ion_stage)], i)

                    if i < len(listions) - 1:  # don't get cross sections for top ion
                        path_nahar_px_file = 'atomic-data-nahar/{0}{1:d}.px.txt'.format(
                            elsymbols[atomic_number].lower(), ion_stage)

                        log_and_print('Reading ' + path_nahar_px_file)
                        nahar_phixs_tables[i] = read_nahar_phixs_tables(path_nahar_px_file, atomic_number, i, ion_stage, args)

                    (energy_levels[i], transitions[i], photoionization_crosssections[i]) = combine_hillier_nahar(
                        i, hillier_energy_levels, hillier_level_ids_matching_term, hillier_transitions,
                        nahar_energy_levels, nahar_level_index_of_state, nahar_configurations[i],
                        nahar_phixs_tables[i], upsilondicts[i], args)

                    # Alternatively use Hillier phixs tables, but BEWARE this
                    # probably doesn't work anymore since the code has changed a lot
                    # print('Reading ' + hillier_ion_folder + path_hillier_phixs_file[(atomic_number, ion_stage)])
                    # read_hillier_phixs_tables(hillier_ion_folder,path_hillier_phixs_file[(atomic_number, ion_stage)])

        write_output_files(elementindex, energy_levels, transitions, upsilondicts,
                           ionization_energy_ev,
                           transition_count_of_level_name,
                           nahar_core_states, nahar_configurations,
                           photoionization_targetfractions, photoionization_crosssections, args)


def read_nahar_energy_level_file(path_nahar_energy_file, atomic_number, i, ion_stage):
    nahar_energy_level_row = namedtuple(
        'energylevel', 'indexinsymmetry TC corestateid elecn elecl energyreltoionpotrydberg twosplusone l parity energyabovegsinpercm g naharconfiguration')
    naharcorestaterow = namedtuple(
        'naharcorestate', 'nahar_core_state_id configuration term energyrydberg')
    nahar_configurations = {}
    nahar_energy_levels = ['IGNORE']
    nahar_level_index_of_state = {}
    nahar_core_states = []

    if not os.path.isfile(path_nahar_energy_file):
        log_and_print(path_nahar_energy_file + ' does not exist')
    else:
        log_and_print('Reading ' + path_nahar_energy_file)
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


def read_hillier_levels_and_transitions(path_hillier_osc_file, hillier_row_format_energy_level, i):
    hillier_energy_level_row = namedtuple(
        'energylevel', hillier_row_format_energy_level + ' corestateid twosplusone l parity indexinsymmetry naharconfiguration matchscore')
    hillier_transition_row = namedtuple(
        'transition', 'namefrom nameto f A lambdaangstrom i j hilliertransitionid lowerlevel upperlevel coll_str')
    hillier_energy_levels = ['IGNORE']
    hillier_level_ids_matching_term = defaultdict(list)
    transition_count_of_level_name = defaultdict(int)
    hillier_ionization_energy_ev = 0.0
    transitions = []

    if os.path.isfile(path_hillier_osc_file):
        with open(path_hillier_osc_file, 'r') as fhillierosc:
            for line in fhillierosc:
                row = line.split()

                # check for right number of columns and that are all numbers except first column
                if len(row) == len(hillier_row_format_energy_level.split()) and all(map(isfloat, row[1:])):
                    hillier_energy_level = hillier_energy_level_row(*row, 0, -1, -1, -1, -1, '', -1)

                    hillierlevelid = int(hillier_energy_level.hillierlevelid.lstrip('-'))
                    levelname = hillier_energy_level.levelname
                    if levelname not in hillier_name_replacements:
                        (twosplusone, l, parity) = get_term_as_tuple(levelname)
                    else:
                        (twosplusone, l, parity) = get_term_as_tuple(hillier_name_replacements[levelname])

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
                            log_and_print("Can't find LS term in Hillier level name '" + levelname + "'")
                        # else:
                            # log_and_print("Can't find LS term in Hillier level name '{0:}' (parity is {1:})".format(levelname, parity))
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

                if len(row) == 8 and all(map(isfloat, row[2:4])):
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
                        log_and_print('FATAL: multiply-defined Hillier transition: {0} {1}'
                                      .format(transition.namefrom, transition.nameto))
                        sys.exit()
        log_and_print('Read {:d} transitions'.format(len(transitions)))

    return hillier_ionization_energy_ev, hillier_energy_levels, transitions, transition_count_of_level_name, hillier_level_ids_matching_term


def read_nahar_phixs_tables(path_nahar_px_file, atomic_number, i, ion_stage, args):
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
            _ = float(fenlist.readline().split()[0])  # nahar_binding_energy_rydberg

            if not args.nophixs:
                phixsarray = np.array([list(map(float, fenlist.readline().split())) for p in range(number_of_points)])
            else:
                for p in range(number_of_points):
                    fenlist.readline()
                phixsarray = np.zeros((2, 2))

            nahar_phixs_tables[(twosplusone, l, parity, indexinsymmetry)] = phixsarray

            #    np.genfromtxt(fenlist, max_rows=number_of_points)

    return nahar_phixs_tables


def get_naharphotoion_upperlevelids(energy_level, energy_levels_upperion, nahar_core_states,
                                    nahar_configurations_upperion, upper_level_ids_of_core_state_id):

    core_state_id = int(energy_level.corestateid)
    if core_state_id > 0 and core_state_id < len(nahar_core_states):

        if not upper_level_ids_of_core_state_id[core_state_id]:
            # go find matching levels if they haven't been found yet
            nahar_core_state = nahar_core_states[core_state_id]
            nahar_core_state_reduced_configuration = reduce_configuration(
                nahar_core_state.configuration + '_' + nahar_core_state.term)
            core_state_energy_ev = nahar_core_state.energyrydberg * ryd_to_ev
            flog.write("\nMatching core state {0} '{1}_{2}' E={3:0.3f} eV to:\n".format(
                core_state_id, nahar_core_state.configuration, nahar_core_state.term, core_state_energy_ev))

            candidate_upper_levels = {}
            for upperlevelid, upperlevel in enumerate(energy_levels_upperion[1:], 1):
                if hasattr(upperlevel, 'levelname'):
                    upperlevelconfig = upperlevel.levelname
                else:
                    state_tuple = (int(upperlevel.twosplusone), int(upperlevel.l),
                                   int(upperlevel.parity), int(upperlevel.indexinsymmetry))
                    upperlevelconfig = nahar_configurations_upperion.get(state_tuple, '-1')
                energyev = upperlevel.energyabovegsinpercm * hc_in_ev_cm

                if reduce_configuration(upperlevelconfig) == nahar_core_state_reduced_configuration:  # this ignores parent term
                    ediff = energyev - core_state_energy_ev
                    upperlevelconfignoj = upperlevelconfig.split('[')[0]
                    if upperlevelconfignoj not in candidate_upper_levels:
                        candidate_upper_levels[upperlevelconfignoj] = [[], []]
                    candidate_upper_levels[upperlevelconfignoj][1].append(upperlevelid)
                    candidate_upper_levels[upperlevelconfignoj][0].append(ediff)
                    flog.write("Upper ion level {0} '{1}' E = {2:.4f} E_diff={3:.4f}\n".format(upperlevelid, upperlevelconfig, energyev, ediff))

            best_ediff = float('inf')
            best_match_upperlevelids = []
            for _, (ediffs, upperlevelids) in candidate_upper_levels.items():
                avg_ediff = abs(sum(ediffs)/len(ediffs))
                if avg_ediff < best_ediff:
                    best_ediff = avg_ediff
                    best_match_upperlevelids = upperlevelids

            flog.write('Best matching levels: {0}\n'.format(best_match_upperlevelids))

            upper_level_ids_of_core_state_id[core_state_id] = best_match_upperlevelids

            # after matching process, still no upper levels matched!
            if not upper_level_ids_of_core_state_id[core_state_id]:
                upper_level_ids_of_core_state_id[core_state_id] = [1]
                log_and_print("No upper levels matched. Defaulting to level 1 (reduced string: '{3}')".format(
                    nahar_core_state_reduced_configuration))

        upperionlevelids = upper_level_ids_of_core_state_id[core_state_id]
    else:
        upperionlevelids = [1]

    return upperionlevelids


def get_nahar_targetfractions(i, energy_levels, energy_levels_upperion, nahar_core_states, nahar_configurations):
    targetlist = [[] for _ in energy_levels]
    upper_level_ids_of_core_state_id = defaultdict(list)
    for lowerlevelid, energy_level in enumerate(energy_levels[1:], 1):
        # find the upper level ids from the Nahar core state
        upperionlevelids = get_naharphotoion_upperlevelids(energy_level, energy_levels_upperion, nahar_core_states, nahar_configurations[i + 1], upper_level_ids_of_core_state_id)

        summed_statistical_weights = sum([float(energy_levels_upperion[id].g) for id in upperionlevelids])
        for upperionlevelid in sorted(upperionlevelids):
            phixsprobability = (energy_levels_upperion[upperionlevelid].g / summed_statistical_weights)
            targetlist[lowerlevelid].append((upperionlevelid, phixsprobability))

    return targetlist


def read_qub_levels_and_transitions(atomic_number, ion_stage):
    qub_energy_level_row = namedtuple(
        'energylevel', 'levelname qub_id twosplusone l j energyabovegsinpercm g parity')
    qub_transition_row = namedtuple(
        'transition', 'lowerlevel upperlevel A nameto namefrom lambdaangstrom coll_str')
    transition_count_of_level_name = defaultdict(int)
    qub_energylevels = ['IGNORE']
    qub_transitions = []
    upsilondict = {}
    ionization_energy_ev = 0.

    if (atomic_number == 27) and (ion_stage == 3):
        log_and_print('Reading atomic-data-qub/adf04_v1')
        with open('atomic-data-qub/adf04_v1', 'r') as fleveltrans:
            line = fleveltrans.readline()
            row = line.split()
            ionization_energy_ev = float(row[4].split('(')[0]) * hc_in_ev_cm
            while True:
                line = fleveltrans.readline()
                if not line or line.startswith('   -1'):
                    break
                config = line[5:21].strip()
                energylevel = qub_energy_level_row(
                    config, int(line[:5]), int(line[25:26]),
                    int(line[27:28]), float(line[29:33]), float(line[34:55]),
                    0.0, 0)
                parity = get_parity_from_config(config)

                levelname = energylevel.levelname + '_{0:d}{1:}{2:}[{3:d}/2]_id={4:}'.format(
                    energylevel.twosplusone, lchars[energylevel.l],
                    ['e', 'o'][energylevel.parity], int(2 * energylevel.j), energylevel.qub_id)

                g = energylevel.twosplusone * (2 * energylevel.l + 1)
                energylevel = energylevel._replace(
                    g=g, parity=parity, levelname=levelname)
                qub_energylevels.append(energylevel)

            upsilonheader = fleveltrans.readline().split()
            list_tempheaders = ['upsT={0:}'.format(x) for x in upsilonheader[2:]]
            list_headers = ['upper', 'lower', 'ignore'] + list_tempheaders
            qubupsilondf_alltemps = pd.read_csv(fleveltrans, index_col=False, delim_whitespace=True,
                                                comment="C", names=list_headers, dtype={'lower': np.int, 'upper': np.int}.update({z: np.float64 for z in list_headers[2:]}),
                                                error_bad_lines=False, skip_blank_lines=True, keep_default_na=False)
            qubupsilondf_alltemps.query('upper!=-1', inplace=True)
            for _, row in qubupsilondf_alltemps.iterrows():
                lower = int(row['lower'])
                upper = int(row['upper'])
                upsilon = float(row['upsT=5.01+03'].replace('-', 'E').replace('+', 'E'))
                if (lower, upper) not in upsilondict:
                    upsilondict[(lower, upper)] = upsilon
                else:
                    log_and_print("Duplicate upsilon value for transition {0:d} to {1:d} keeping {2:5.2e} instead of using {3:5.2e}".format(
                        lower, upper, upsilondict[(lower, upper)], upsilon))

        with open('atomic-data-qub/adf04rad_v1', 'r') as ftrans:
            for line in ftrans:
                row = line.split()
                id_upper = int(row[0])
                id_lower = int(row[1])
                A = float(row[2])
                if A > 2e-30:
                    namefrom = qub_energylevels[id_upper].levelname
                    nameto = qub_energylevels[id_lower].levelname
                    forbidden = check_forbidden(qub_energylevels[id_upper], qub_energylevels[id_lower])
                    transition_count_of_level_name[namefrom] += 1
                    transition_count_of_level_name[nameto] += 1
                    lamdaangstrom = 1.e8 / (qub_energylevels[id_upper].energyabovegsinpercm - qub_energylevels[id_lower].energyabovegsinpercm)
                    if (id_lower, id_upper) in upsilondict:
                        coll_str = upsilondict[(id_lower, id_upper)]
                    elif forbidden:
                        coll_str = -2.
                    else:
                        coll_str = -1.
                    transition = qub_transition_row(id_lower, id_upper, A, namefrom, nameto, lamdaangstrom, coll_str)
                    qub_transitions.append(transition)

    if (atomic_number == 27) and (ion_stage == 4):
        ionization_energy_ev = 54.9000015
        qub_energylevels.append(qub_energy_level_row('groundstate', 1, 0, 0, 0, 0., 10, 0))

    return ionization_energy_ev, qub_energylevels, qub_transitions, transition_count_of_level_name, upsilondict


def read_qub_photoionizations(atomic_number, ion_stage, energy_levels, args):
    photoionization_crosssections = np.zeros((len(energy_levels), args.nphixspoints))
    # photoionization_crosssections = [[] for _ in energy_levels]
    photoionization_targetfractions = [[(1, 1.)] for _ in energy_levels]

    if atomic_number == 27 and ion_stage == 2:
        for lowerlevelid in [1, 2, 3, 4, 5, 6, 7, 8]:
            photdata = pd.read_csv('atomic-data-qub/{0:d}.gz'.format(lowerlevelid), delim_whitespace=True, header=None)
            phixstables = {}
            ntargets = 40
            for targetlevel in range(1, ntargets + 1):
                phixstables[targetlevel] = photdata.loc[photdata[:][targetlevel] > 0.][[0, targetlevel]].values

            reduced_phixs_dict = reduce_phixs_tables(phixstables, args)
            target_scalefactors = np.zeros(ntargets + 1)
            for upperlevelid in reduced_phixs_dict:
                # take the ratio of cross sections at the threshold energyies
                target_scalefactors[upperlevelid] = reduced_phixs_dict[upperlevelid][0] / reduced_phixs_dict[1][0]
            targetfractions = target_scalefactors / sum(target_scalefactors)

            photoionization_targetfractions[lowerlevelid] = [(upperlevelid, targetfractions[upperlevelid]) for upperlevelid in range(1, ntargets + 1)]
            photoionization_crosssections[lowerlevelid] = reduced_phixs_dict[1] / targetfractions[1]

    if atomic_number == 27 and ion_stage == 3:
        for lowerlevelid in range(1, len(energy_levels)):
            photoionization_targetfractions[lowerlevelid] = [(1, 1.)]
            if lowerlevelid <= 4:
                photoionization_crosssections[lowerlevelid] = np.array([9.3380692, 7.015829602, 5.403975231, 4.250372872, 3.403086443, 2.766835319, 2.279802051, 1.900685772, 1.601177846, 1.361433037, 1.16725865, 1.008321909, 0.8769787, 0.76749151, 0.675496904, 0.597636429, 0.531296609, 0.474423066, 0.425385805, 0.382880364, 0.345854415, 0.313452694, 0.284975256, 0.259845541, 0.237585722, 0.217797532, 0.200147231, 0.184353724, 0.17017913, 0.157421217, 0.145907331, 0.135489462, 0.126040239, 0.117449648, 0.109622338, 0.102475382, 0.095936439, 0.089942202, 0.084437113, 0.079372279, 0.074704554, 0.070395769, 0.066412076, 0.062723384, 0.059302883, 0.056126637, 0.053173226, 0.050423446, 0.047860046, 0.045467498, 0.043231802, 0.041140312, 0.039181587, 0.037345256, 0.035621907, 0.034002983, 0.032480693, 0.031047932, 0.029698215, 0.028425611, 0.027224692, 0.026090478, 0.025018404, 0.02400427, 0.023044216, 0.022134683, 0.021272391, 0.020454314, 0.019677652, 0.018939819, 0.018238416, 0.017571225, 0.016936183, 0.016331377, 0.01575503, 0.015205486, 0.014681206, 0.014180754, 0.013702792, 0.013246071, 0.012809423, 0.012391758, 0.011992055, 0.011609359, 0.011242775, 0.010891464, 0.010554639, 0.010231561, 0.009921535, 0.009623909, 0.009338069, 0.009063438, 0.008799471, 0.008545656, 0.00830151, 0.008066575, 0.007840423, 0.007622646, 0.00741286, 0.007210703])

    return photoionization_crosssections, photoionization_targetfractions


def combine_hillier_nahar(i, hillier_energy_levels, hillier_level_ids_matching_term, hillier_transitions,
                          nahar_energy_levels, nahar_level_index_of_state, nahar_configurations, nahar_phixs_tables, upsilondict,
                          args):
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
                flog.write("Replacing Nahar configuration of '{0}' with '{1}'\n".format(
                    nahar_configurations[state_tuple], nahar_configuration_this_state))

        if hillier_level_ids_matching_term[(twosplusone, l, parity)]:
                # match the electron configurations
            if nahar_configuration_this_state != '_CONFIG NOT FOUND_':
                best_match_score = 0.
                for levelid in hillier_level_ids_matching_term[(twosplusone, l, parity)]:
                    levelname = hillier_energy_levels[levelid].levelname
                    if levelname in hillier_name_replacements:
                        levelname = hillier_name_replacements[levelname]

                    match_score = score_config_match(levelname, nahar_configuration_this_state)
                    best_match_score = max(best_match_score, match_score)

                if best_match_score > 0:
                    for levelid in hillier_level_ids_matching_term[(twosplusone, l, parity)]:
                        hlevelname = hillier_energy_levels[levelid].levelname
                        if hlevelname in hillier_name_replacements:
                            hlevelname = hillier_name_replacements[hlevelname]
                        match_score = score_config_match(hlevelname, nahar_configuration_this_state)

                        if match_score == best_match_score and \
                                hillier_energy_levels[levelid].indexinsymmetry < 1:  # make sure this Hillier level hasn't already been matched to a Nahar state

                            core_state_id = nahar_energy_levels[nahar_level_index_of_state[state_tuple]].corestateid

                            confignote = nahar_configurations[state_tuple]

                            if nahar_configuration_this_state != confignote:
                                confignote += " replaced by {:}".format(nahar_configuration_this_state)

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
                    added_nahar_levels.append(nahar_energy_level._replace(naharconfiguration=nahar_configurations[state_tuple]))
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
                flog.write('<E> = {0:.3f} eV, g_sum = {1:.1f}: \n'.format(avghillierenergyabovegsinev, sumhillierstatweights))

            strhilliermatches = '\n'.join(['{0} ({1:.3f} eV, g = {2:.1f}, match_score = {3:.1f})'.format(hillier_energy_levels[k].levelname, hc_in_ev_cm * float(
                hillier_energy_levels[k].energyabovegsinpercm), hillier_energy_levels[k].g, hillier_energy_levels[k].matchscore) for k in hillier_level_ids_matching_this_nahar_state])

            flog.write(strhilliermatches + '\n')

    energy_levels = hillier_energy_levels + added_nahar_levels

    log_and_print('Included {0} levels from Hillier dataset and added {1} levels from Nahar phixs tables for a total of {2} levels'.format(
        len(hillier_energy_levels) - 1, len(added_nahar_levels), len(energy_levels) - 1))

    # sort the concatenated energy level list by energy
    print('Sorting level list...')
    energy_levels.sort(key=lambda x: float(getattr(x, 'energyabovegsinpercm', '-inf')))

    photoionization_crosssections = np.zeros((len(energy_levels), args.nphixspoints))  # this probably gets overwritten anyway

    if not args.nophixs:
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
                log_and_print("No Hillier or Nahar state to match with photoionization crosssection of {0:d}{1}{2} index {3:d}".format(
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
    nu0 = 1e13
    # fac = math.exp(h_over_kb_in_K_sec * nu0 / args.optimaltemperature)

    def integrand(nu):
        return (nu ** 2) * math.exp(- h_over_kb_in_K_sec * nu / args.optimaltemperature)

    # def integrand_vec(nu_list):
    #    return [(nu ** 2) * math.exp(- h_over_kb_in_K_sec * (nu - nu0) / args.optimaltemperature)
    #            for nu in nu_list]

    integrand_vec = np.vectorize(integrand)

    xgrid = np.linspace(1.0,
                        1.0 + args.phixsnuincrement * (args.nphixspoints + 1),
                        num=args.nphixspoints + 1, endpoint=False)

    for key, tablein in dicttables.items():
        # tablein is an array of pairs (energy, phixs cross section)

        nu0 = tablein[0][0] * ryd_to_hz

        arr_sigma_out = np.empty(args.nphixspoints)
        # x is nu/nu_edge

        sigma_interp = interpolate.interp1d(tablein[:, 0], tablein[:, 1], kind='linear', assume_sorted=True)

        for i, _ in enumerate(xgrid[:-1]):
            enlow = xgrid[i] * tablein[0][0]
            enhigh = xgrid[i + 1] * tablein[0][0]

            # start of interval interpolated point, Nahar points, and end of interval interpolated point
            samples_in_interval = tablein[(enlow <= tablein[:, 0]) & (tablein[:, 0] <= enhigh)]

            if len(samples_in_interval) == 0 or ((samples_in_interval[0, 0] - enlow)/enlow) > 1e-20:
                if i == 0:
                    print('adding first point {0:.4e} {1:.4e} {2:.4e}'.format(enlow, samples_in_interval[0, 0], ((samples_in_interval[0, 0] - enlow)/enlow)))
                if (enlow <= tablein[-1][0]):
                    new_crosssection = sigma_interp(enlow)
                    if new_crosssection < 0:
                        print('negative extrap')
                else:
                    new_crosssection = tablein[-1][1] * (tablein[-1][0] / enlow) ** 3  # assume power law decay after last point
                samples_in_interval = np.vstack([[enlow, new_crosssection], samples_in_interval])

            if len(samples_in_interval) == 0 or ((enhigh - samples_in_interval[-1, 0])/samples_in_interval[-1, 0]) > 1e-20:
                if (enhigh <= tablein[-1][0]):
                    new_crosssection = sigma_interp(enhigh)
                    if new_crosssection < 0:
                        print('negative extrap')
                else:
                    new_crosssection = tablein[-1][1] * (tablein[-1][0] / enhigh) ** 3  # assume power law decay after last point

                samples_in_interval = np.vstack([samples_in_interval, [enhigh, new_crosssection]])

            nsamples = len(samples_in_interval)

            # integralnosigma, err = integrate.fixed_quad(integrand_vec, enlow, enhigh, n=250)
            # integralwithsigma, err = integrate.fixed_quad(
            #    lambda x: sigma_interp(x) * integrand_vec(x), enlow, enhigh, n=250)

            # this is incredibly fast, but maybe not accurate
            # integralnosigma, err = integrate.quad(integrand, enlow, enhigh, epsrel=1e-2)
            # integralwithsigma, err = integrate.quad(
            #    lambda x: sigma_interp(x) * integrand(x), enlow, enhigh, epsrel=1e-2)

            if nsamples >= 500 or enlow > tablein[-1][0]:
                arr_energyryd = samples_in_interval[:, 0]
                arr_sigma_megabarns = samples_in_interval[:, 1]
            else:
                nsteps = 500
                arr_energyryd = np.linspace(enlow, enhigh, num=nsteps, endpoint=False)
                arr_sigma_megabarns = np.interp(arr_energyryd, tablein[:, 0], tablein[:, 1])

            integrand_vals = integrand_vec(arr_energyryd * ryd_to_hz)
            sigma_integrand_vals = [sigma * integrand_val
                                    for integrand_val, sigma
                                    in zip(integrand_vals, arr_sigma_megabarns)]

            integralnosigma = integrate.trapz(integrand_vals, arr_energyryd)
            integralwithsigma = integrate.trapz(sigma_integrand_vals, arr_energyryd)

            if integralwithsigma > 0 and integralnosigma > 0:
                arr_sigma_out[i] = (integralwithsigma / integralnosigma)
            elif integralwithsigma == 0:
                arr_sigma_out[i] = 0.
            else:
                print('Math error: ', i, nsamples, arr_sigma_megabarns[i], integralwithsigma, integralnosigma)
                print(samples_in_interval)
                print(arr_sigma_out[i-1])
                print(arr_sigma_out[i])
                print(arr_sigma_out[i+1])
                arr_sigma_out[i] = 0.
                # sys.exit()

        dictout[key] = arr_sigma_out  # output a 1D list of cross sections

    return dictout


def check_forbidden(levela, levelb):
    return levela.parity == levelb.parity


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

    if '{' in config and '}' in config:  # JJ coupling, no L and S
        if config[-1] == 'e':
            return (-1, -1, 0)
        elif config[-1] == 'o':
            return (-1, -1, 1)
        else:
            log_and_print("Can't read parity from JJ coupling state '" + config + "'")
            return (-1, -1, -1)
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

    if lposition < len(strin)-1 and strin[lposition+1:] != 'e':
        jvalue = int(strin[lposition+1:])
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

    if instr[-1] not in ['o', 'e']:
        instr = instr + 'e'  # last character being S,P,D, etc means even
    if str.isdigit(instr[-2]):  # J value is in the term, so remove it
        instr = instr[:-2] + instr[-1]

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
        elif not str.isdigit(instr[-1]):
            term_parity = term_parity + 2  # this accounts for things like '3d7(4F)6d_5Pbe' in the Hillier levels. Shouldn't match these
        instr = instr[:-1]

    if str.isdigit(instr[-1]):
        term_twosplusone = int(instr[-1])
        instr = instr[:-1]

    if instr[-1] == '_':
        instr = instr[:-1]
    elif instr[-1] in alphabets and ((len(instr) < 2 or not str.isdigit(instr[-2])) or (len(instr) < 3 or instr[-3] in lchars.lower())):
        # to catch e.g., '3d6(5D)6d4Ge[9/2]' occupation piece 6d, not index d
        # and 3d7b2Fe is at index b, (keep it from conflicting into the orbital occupation)
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


def get_parity_from_config(instr):
    configsplit = interpret_configuration(instr)[0]
    lsum = 0
    for orbitalstr in configsplit:
        l = lchars.lower().index(orbitalstr[1])
        if len(orbitalstr[2:]) > 0:
            nelec = int(orbitalstr[2:])
        else:
            nelec = 1
        lsum += l * nelec

    return lsum % 2


def score_config_match(config_a, config_b):
    if config_a.split('[')[0] == config_b.split('[')[0]:
        return 100

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
        return 99
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

        score = int(98 * matched_pieces / max(non_term_pieces_a, non_term_pieces_b) * parent_term_match)
        return score
    else:
        return 5  # term matches but no electron config available or it's an Eqv state...0s type

    print("WHAT?")
    sys.exit()
    return -1


def write_output_files(elementindex, energy_levels, transitions, upsilondicts,
                       ionization_energies,
                       transition_count_of_level_name,
                       nahar_core_states, nahar_configurations,
                       photoionization_targetfractions,
                       photoionization_crosssections, args):
    global flog
    atomic_number, listions = listelements[elementindex]
    upsilon_transition_row = namedtuple('transition', 'lowerlevel upperlevel A nameto namefrom lambdaangstrom coll_str')

    print('\nStarting output stage:')

    with open(os.path.join(args.output_folder, 'adata.txt'), 'a') as fatommodels, \
            open(os.path.join(args.output_folder, 'transitiondata.txt'), 'a') as ftransitiondata, \
            open(os.path.join(args.output_folder, 'phixsdata_v2.txt'), 'a') as fphixs, \
            open(os.path.join(args.output_folder_transition_guide, 'transitions_{}.txt'.format(elsymbols[atomic_number])), 'w') as ftransitionguide:

        ftransitionguide.write('{0:>16s} {1:>12s} {2:>3s} {3:>9s} {4:>17s} {5:>17s} {6:>10s} {7:25s}  {8:25s} {9:>17s} {10:>17s} {11:>19s}\n'.format(
            'lambda_angstroms', 'A', 'Z', 'ion_stage', 'lower_energy_Ev', 'lower_statweight', 'forbidden', 'lower_level', 'upper_level', 'upper_statweight', 'upper_energy_Ev', 'upper_has_permitted'))

        for i, ion_stage in enumerate(listions):
            upsilondict = upsilondicts[i]
            ionstr = '{0} {1}'.format(elsymbols[atomic_number],
                                      roman_numerals[ion_stage])

            flog = open(os.path.join(args.output_folder, args.output_folder_logs,
                                     '{0}{1:d}.txt'.format(elsymbols[atomic_number].lower(), ion_stage)), 'a')

            print('==============> ' + ionstr + ':')

            level_id_of_level_name = {}
            for levelid in range(1, len(energy_levels[i])):
                if hasattr(energy_levels[i][levelid], 'levelname'):
                    level_id_of_level_name[energy_levels[i][levelid].levelname] = levelid

            unused_upsilon_transitions = set(upsilondicts[i].keys())
            for transitionid, transition in enumerate(transitions[i]):
                updaterequired = False
                if hasattr(transition, 'upperlevel') and transition.upperlevel >= 0:
                    id_lower = transition.lowerlevel
                    id_upper = transition.upperlevel
                else:
                    id_upper = level_id_of_level_name[transition.nameto]
                    id_lower = level_id_of_level_name[transition.namefrom]
                    updaterequired = True
                unused_upsilon_transitions.discard((id_lower, id_upper))

                coll_str = transition.coll_str
                if coll_str < 5:
                    forbidden = check_forbidden(energy_levels[i][id_lower], energy_levels[i][id_upper])
                    if (id_lower, id_upper) in upsilondict:
                        coll_str = upsilondict[(id_lower, id_upper)]
                    elif forbidden:
                        coll_str = -2.
                    else:
                        coll_str = -1.
                    updaterequired = True
                if updaterequired:
                    transitions[i][transitionid] = transition._replace(
                        lowerlevel=id_lower, upperlevel=id_upper,
                        coll_str=coll_str)

            print("Adding in {0:d} extra transitions with upsilon values".format(len(unused_upsilon_transitions)))
            for (id_lower, id_upper) in unused_upsilon_transitions:
                namefrom = energy_levels[i][id_upper].levelname
                nameto = energy_levels[i][id_lower].levelname
                A = 0.
                lamdaangstrom = 1.e8 / (energy_levels[i][id_upper].energyabovegsinpercm - energy_levels[i][id_lower].energyabovegsinpercm)
                # upsilon = upsilondict[(id_lower, id_upper)]
                transition_count_of_level_name[i][namefrom] += 1
                transition_count_of_level_name[i][nameto] += 1
                coll_str = upsilondict[(id_lower, id_upper)]

                transition = upsilon_transition_row(id_lower, id_upper, A, namefrom, nameto, lamdaangstrom, coll_str)
                transitions[i].append(transition)

            transitions[i].sort(key=lambda x: (getattr(x, 'lowerlevel', -99), getattr(x, 'upperlevel', -99)))

            write_adata(fatommodels, atomic_number, ion_stage, energy_levels[i], ionization_energies[i], transition_count_of_level_name[i], args)

            write_transition_data(ftransitiondata, ftransitionguide, atomic_number, ion_stage, energy_levels[i], transitions[i], upsilondicts[i], args)

            if i < len(listions) - 1 and not args.nophixs:  # ignore the top ion
                if len(photoionization_targetfractions[i]) < 1:
                    photoionization_targetfractions[i] = get_nahar_targetfractions(i, energy_levels[i], energy_levels[i+1], nahar_core_states[i], nahar_configurations)
                write_phixs_data(fphixs, i, atomic_number, ion_stage, energy_levels[i], photoionization_crosssections[i], photoionization_targetfractions[i], args)


def write_adata(fatommodels, atomic_number, ion_stage, energy_levels, ionization_energy, transition_count_of_level_name, args):
    log_and_print("writing to 'adata.txt'")
    fatommodels.write('{0:12d}{1:12d}{2:12d}{3:15.7f}\n'.format(
        atomic_number, ion_stage,
        len(energy_levels) - 1,
        ionization_energy))

    for levelid, energylevel in enumerate(energy_levels[1:], 1):

        if hasattr(energylevel, 'levelname'):
            transitioncount = transition_count_of_level_name.get(energylevel.levelname, 0)
        else:
            transitioncount = 0

        level_comment = ""
        try:
            hlevelname = energylevel.levelname
            if hlevelname in hillier_name_replacements:
                # hlevelname += ' replaced by {0}'.format(hillier_name_replacements[hlevelname])
                hlevelname = hillier_name_replacements[hlevelname]
            level_comment = hlevelname.ljust(30)
        except AttributeError:
            level_comment = " " * 30

        try:
            if energylevel.indexinsymmetry >= 0:
                level_comment += 'Nahar: {:d}{:}{:} index {:}'.format(
                    energylevel.twosplusone,
                    lchars[energylevel.l],
                    ['e', 'o'][energylevel.parity],
                    energylevel.indexinsymmetry)
                try:
                    config = energylevel.naharconfiguration
                    if energylevel.naharconfiguration.strip() in nahar_configuration_replacements:
                        config += ' replaced by {0}'.format(nahar_configuration_replacements[energylevel.naharconfiguration.strip()])
                    level_comment += " '{0}'".format(config)
                except AttributeError:
                    level_comment += ' (no config)'
        except AttributeError:
            pass

        fatommodels.write('{:7d}{:25.16f}{:25.16f}{:7d}     {:}\n'.format(
            levelid, hc_in_ev_cm * float(energylevel.energyabovegsinpercm),
            float(energylevel.g), transitioncount, level_comment))

    fatommodels.write('\n')


def write_transition_data(ftransitiondata, ftransitionguide, atomic_number, ion_stage, energy_levels,
                          transitions, upsilondict, args):
    log_and_print("writing to 'transitiondata.txt'")

    num_forbidden_transitions = 0
    num_collision_strengths_applied = 0
    ftransitiondata.write('{0:7d}{1:7d}{2:12d}\n'.format(
        atomic_number, ion_stage, len(transitions)))

    level_ids_with_permitted_down_transitions = set()
    for transitionid, transition in enumerate(transitions):
        levelid_lower = transition.lowerlevel
        levelid_upper = transition.upperlevel
        forbidden = (energy_levels[levelid_lower].parity == energy_levels[levelid_upper].parity)

        if not forbidden:
            level_ids_with_permitted_down_transitions.add(levelid_upper)

    for transitionid, transition in enumerate(transitions):
        levelid_lower = transition.lowerlevel
        levelid_upper = transition.upperlevel
        coll_str = transition.coll_str

        if coll_str > 0:
            num_collision_strengths_applied += 1

        forbidden = (energy_levels[levelid_lower].parity == energy_levels[levelid_upper].parity)

        if forbidden:
            num_forbidden_transitions += 1
            flog.write('Forbidden transition: lambda_angstrom= {:7.1f}, {:25s} to {:25s}\n'.format(
                float(transition.lambdaangstrom), transition.namefrom,
                transition.nameto))

        ftransitionguide.write('{0:16.1f} {1:12E} {2:3d} {3:9d} {4:17.2f} {5:17.4f} {6:10b} {7:25s} {8:25s} {9:17.2f} {10:17.4f} {11:19b}\n'.format(
            abs(float(transition.lambdaangstrom)), float(transition.A),
            atomic_number, ion_stage,
            hc_in_ev_cm * float(
                energy_levels[levelid_lower].energyabovegsinpercm),
            float(energy_levels[levelid_lower].g), forbidden,
            transition.namefrom, transition.nameto,
            float(energy_levels[levelid_upper].g),
            hc_in_ev_cm * float(
                energy_levels[levelid_upper].energyabovegsinpercm),
            levelid_upper in level_ids_with_permitted_down_transitions))

        ftransitiondata.write(
            '{0:9d}{1:6d}{2:6d}{3:18.10E} {4:9.2e}\n'.format(
                transitionid + 1, levelid_lower, levelid_upper,
                float(transition.A), coll_str))

    ftransitiondata.write('\n')
    log_and_print('Wrote out {0:d} transitions, of which {1:d} are forbidden and {2:d} had collision strengths'.format(
        len(transitions), num_forbidden_transitions, num_collision_strengths_applied))


def write_phixs_data(fphixs, i, atomic_number, ion_stage, energy_levels,
                     photoionization_crosssections, photoionization_targetfractions, args):
    log_and_print("writing to 'phixsdata2.txt'")
    flog.write('Downsampling cross sections assuming T={0} Kelvin\n'.format(args.optimaltemperature))

    for lowerlevelid in range(1, len(energy_levels)):

        if len(photoionization_targetfractions[lowerlevelid]) <= 1 and photoionization_targetfractions[lowerlevelid][0][1] > 0.99:
            if len(photoionization_targetfractions[lowerlevelid]) > 0:
                upperionlevelid = photoionization_targetfractions[lowerlevelid][0][0]
            else:
                upperionlevelid = 1

            fphixs.write('{0:12d}{1:12d}{2:8d}{3:12d}{4:8d}\n'.format(
                atomic_number, ion_stage + 1, upperionlevelid, ion_stage, lowerlevelid))
        else:
            targetlist = photoionization_targetfractions[lowerlevelid]
            fphixs.write('{0:12d}{1:12d}{2:8d}{3:12d}{4:8d}\n'.format(
                atomic_number, ion_stage + 1, -1, ion_stage, lowerlevelid))
            fphixs.write('{0:8d}\n'.format(len(targetlist)))
            for upperionlevelid, targetprobability in targetlist:
                fphixs.write('{0:8d}{1:12f}\n'.format(upperionlevelid, targetprobability))

        for crosssection in photoionization_crosssections[lowerlevelid]:
            fphixs.write('{0:16.8E}\n'.format(crosssection))


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

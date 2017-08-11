#!/usr/bin/env python3

import os
import sys
from collections import defaultdict, namedtuple

import numpy as np
from astropy import constants as const
from astropy import units as u

import artisatomic

ryd_to_ev = u.rydberg.to('eV')

hc_in_ev_cm = (const.h * const.c).to('eV cm').value
hc_in_ev_angstrom = (const.h * const.c).to('eV angstrom').value
h_in_ev_seconds = const.h.to('eV s').value

alphabets = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
reversedalphabets = 'zyxwvutsrqponmlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA '
lchars = 'SPDFGHIKLMNOPQRSTUVWXYZ'

term_index_tuple = namedtuple('term_index', 'twosplusone lval parity indexinsymmetry')


def read_nahar_energy_level_file(path_nahar_energy_file, atomic_number, ion_stage, flog):
    nahar_energy_level_row = namedtuple(
        'energylevel', 'indexinsymmetry TC corestateid elecn elecl energyreltoionpotrydberg twosplusone l parity energyabovegsinpercm g naharconfiguration')
    nahar_configurations = {}
    nahar_energy_levels = ['IGNORE']
    nahar_level_index_of_state = {}
    nahar_core_states = []

    if not os.path.isfile(path_nahar_energy_file):
        artisatomic.log_and_print(flog, path_nahar_energy_file + ' does not exist')
    else:
        artisatomic.log_and_print(flog, 'Reading ' + path_nahar_energy_file)
        with open(path_nahar_energy_file, 'r') as fenlist:
            nahar_core_states = read_nahar_core_states(fenlist)

            nahar_configurations, nahar_ionization_potential_rydberg = read_nahar_configurations(fenlist, flog)

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
                    print('End of file before table iii ends')
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
                    print('End of file before table finished')
                    sys.exit()
                row = line.split()
                if row == ['0', '0', '0', '0']:
                    break
                twosplusone = int(row[0])
                l_val = int(row[1])
                parity = int(row[2])
                number_of_states_in_symmetry = int(row[3])

                line = fenlist.readline()  # core state number and energy

                for _ in range(number_of_states_in_symmetry):
                    row = fenlist.readline().split()
                    indexinsymmetry = int(row[0])
                    nahar_core_state_id = int(row[2])
                    if nahar_core_state_id < 1 or nahar_core_state_id > len(nahar_core_states):
                        flog.write("Core state id of {0:d}{1}{2} index {3:d} is invalid (={4:d}, Ncorestates={5:d}). Setting core state to 1 instead.\n".format(
                            twosplusone, lchars[l_val], ['e', 'o'][parity],
                            indexinsymmetry, nahar_core_state_id,
                            len(nahar_core_states)))
                        nahar_core_state_id = 1

                    nahar_energy_levels.append(nahar_energy_level_row(*row, twosplusone, l_val, parity, -1.0, 0, ''))

                    energyabovegsinpercm = \
                        (nahar_ionization_potential_rydberg +
                         float(nahar_energy_levels[-1].energyreltoionpotrydberg)) * \
                        ryd_to_ev / hc_in_ev_cm

                    nahar_energy_levels[-1] = nahar_energy_levels[-1]._replace(
                        indexinsymmetry=indexinsymmetry,
                        corestateid=nahar_core_state_id,
                        energyreltoionpotrydberg=float(nahar_energy_levels[-1].energyreltoionpotrydberg),
                        energyabovegsinpercm=energyabovegsinpercm,
                        g=twosplusone * (2 * l_val + 1)
                    )
                    nahar_level_index_of_state[(twosplusone, l_val, parity, indexinsymmetry)] = len(nahar_energy_levels) - 1

    #                if float(nahar_energy_levels[-1].energyreltoionpotrydberg) >= 0.0:
    #                    nahar_energy_levels.pop()

    return (nahar_energy_levels, nahar_core_states, nahar_level_index_of_state, nahar_configurations, nahar_ionization_potential_rydberg)


def read_nahar_core_states(fenlist):
    naharcorestaterow = namedtuple('naharcorestate', 'nahar_core_state_id configuration term energyrydberg')
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
            print(f'Nahar levels mismatch: id {c:d} found at entry'
                  f' number {int(nahar_core_states[c].nahar_core_state_id):d}')
            sys.exit()
    return nahar_core_states


def read_nahar_phixs_tables(path_nahar_px_file, atomic_number, ion_stage, args):
    nahar_phixs_tables = {}
    thresholds_ev_dict = {}
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
            binding_energy_ryd = float(fenlist.readline().split()[0])
            thresholds_ev_dict[(twosplusone, l, parity, indexinsymmetry)] = binding_energy_ryd * 13.605698065

            if not args.nophixs:
                phixsarray = np.array([list(map(float, fenlist.readline().split())) for p in range(number_of_points)])
            else:
                for _ in range(number_of_points):
                    fenlist.readline()
                phixsarray = np.zeros((2, 2))

            nahar_phixs_tables[(twosplusone, l, parity, indexinsymmetry)] = phixsarray

    return nahar_phixs_tables, thresholds_ev_dict


def read_nahar_configurations(fenlist, flog):
    nahar_configurations = {}
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
            nahar_ionization_potential_rydberg_str = line.split('=')[2]
            if 'E' not in nahar_ionization_potential_rydberg_str:
                nahar_ionization_potential_rydberg_str = nahar_ionization_potential_rydberg_str.replace('+', 'E+')
            nahar_ionization_potential_rydberg = -float(nahar_ionization_potential_rydberg_str)
            flog.write(f'Ionization potential = {nahar_ionization_potential_rydberg * ryd_to_ev:.4f} eV\n')

        if line.startswith(' ') and len(line) > 36 and artisatomic.isfloat(line[29:29 + 8]):
            found_table = True
            state = line[1:22]
#               energy = float(line[29:29+8])
            twosplusone = int(state[18])
            l_val = lchars.index(state[19])

            if state[20] == 'o':
                parity = 1
                indexinsymmetry = reversedalphabets.index(state[17]) + 1
            else:
                parity = 0
                indexinsymmetry = alphabets.index(state[17]) + 1

            # print(state,energy,twosplusone,l,parity,indexinsymmetry)
            nahar_configurations[(twosplusone, l_val, parity, indexinsymmetry)] = state
        else:
            if found_table:
                break

    return nahar_configurations, nahar_ionization_potential_rydberg


def get_naharphotoion_upperlevelids(energy_level, energy_levels_upperion, nahar_core_states,
                                    nahar_configurations_upperion, upper_level_ids_of_core_state_id, flog):
    """
        Returns a list of upper level id numbers for a given energy level's photoionisation processes
    """
    # core_state_id = int(energy_level.corestateid)
    core_state_id = 1  # temporary fix
    if core_state_id > 0 and core_state_id < len(nahar_core_states):

        if not upper_level_ids_of_core_state_id[core_state_id]:
            # go find matching levels if they haven't been found yet
            nahar_core_state = nahar_core_states[core_state_id]
            nahar_core_state_reduced_configuration = artisatomic.reduce_configuration(
                nahar_core_state.configuration + '_' + nahar_core_state.term)
            core_state_energy_ev = nahar_core_state.energyrydberg * ryd_to_ev
            flog.write(
                f"\nMatching core state {core_state_id} '{nahar_core_state.configuration}_{nahar_core_state.term}' E={core_state_energy_ev:0.3f} eV to:\n")

            candidate_upper_levels = {}
            for upperlevelid, upperlevel in enumerate(energy_levels_upperion[1:], 1):
                if hasattr(upperlevel, 'levelname'):
                    upperlevelconfig = upperlevel.levelname
                else:
                    state_tuple = (int(upperlevel.twosplusone), int(upperlevel.l),
                                   int(upperlevel.parity), int(upperlevel.indexinsymmetry))
                    upperlevelconfig = nahar_configurations_upperion.get(state_tuple, '-1')
                energyev = upperlevel.energyabovegsinpercm * hc_in_ev_cm

                # this ignores parent term
                if artisatomic.reduce_configuration(upperlevelconfig) == nahar_core_state_reduced_configuration:
                    ediff = energyev - core_state_energy_ev
                    upperlevelconfignoj = upperlevelconfig.split('[')[0]
                    if upperlevelconfignoj not in candidate_upper_levels:
                        candidate_upper_levels[upperlevelconfignoj] = [[], []]
                    candidate_upper_levels[upperlevelconfignoj][1].append(upperlevelid)
                    candidate_upper_levels[upperlevelconfignoj][0].append(ediff)
                    flog.write(
                        f"Upper ion level {upperlevelid} '{upperlevelconfig}' E = {energyev:.4f} E_diff={ediff:.4f}\n")

            best_ediff = float('inf')
            best_match_upperlevelids = []
            for _, (ediffs, upperlevelids) in candidate_upper_levels.items():
                avg_ediff = abs(sum(ediffs) / len(ediffs))
                if avg_ediff < best_ediff:
                    best_ediff = avg_ediff
                    best_match_upperlevelids = upperlevelids

            flog.write(f'Best matching levels: {best_match_upperlevelids}\n')

            upper_level_ids_of_core_state_id[core_state_id] = best_match_upperlevelids

            # after matching process, still no upper levels matched!
            if not upper_level_ids_of_core_state_id[core_state_id]:
                upper_level_ids_of_core_state_id[core_state_id] = [1]
                artisatomic.log_and_print(
                    flog, f"No upper levels matched. Defaulting to level 1 (reduced string: '{nahar_core_state_reduced_configuration}')")

        upperionlevelids = upper_level_ids_of_core_state_id[core_state_id]
    else:
        upperionlevelids = [1]

    return upperionlevelids


def get_photoiontargetfractions(energy_levels, energy_levels_upperion, nahar_core_states, nahar_configurations_upperion, flog):
    targetlist = [[] for _ in energy_levels]
    upper_level_ids_of_core_state_id = defaultdict(list)
    for lowerlevelid, energy_level in enumerate(energy_levels[1:], 1):
        # find the upper level ids from the Nahar core state
        upperionlevelids = get_naharphotoion_upperlevelids(energy_level, energy_levels_upperion, nahar_core_states,
                                                           nahar_configurations_upperion,
                                                           upper_level_ids_of_core_state_id, flog)

        summed_statistical_weights = sum([float(energy_levels_upperion[id].g) for id in upperionlevelids])
        for upperionlevelid in sorted(upperionlevelids):
            phixsprobability = (energy_levels_upperion[upperionlevelid].g / summed_statistical_weights)
            targetlist[lowerlevelid].append((upperionlevelid, phixsprobability))

    return targetlist

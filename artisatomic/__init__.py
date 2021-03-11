#!/usr/bin/env python3
import argparse
import glob
import itertools
import math
import multiprocessing as mp
import os
import sys
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from scipy import integrate, interpolate

import artisatomic.readhillierdata as readhillierdata
import artisatomic.readnahardata as readnahardata
import artisatomic.readqubdata as readqubdata
from artisatomic.manual_matches import (hillier_name_replacements, nahar_configuration_replacements)
import artisatomic.readboyledata as readboyledata
import artisatomic.readdreamdata as readdreamdata

PYDIR = os.path.dirname(os.path.abspath(__file__))
atomicdata = pd.read_csv(os.path.join(PYDIR, 'atomic_properties.txt'), delim_whitespace=True, comment='#')
elsymbols = ['n'] + list(atomicdata['symbol'].values)
atomic_weights = ['n'] + list(atomicdata['mass'].values)

roman_numerals = (
    '', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
    'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX'
)

listelements = [
    # (2, [1, 2]),
    # (8, [1, 2, 3, 4]),
    # (14, [1, 2, 3, 4]),
    # (16, [1, 2, 3, 4]),
    # (20, [1, 2, 3, 4]),
    (26, [1, 2, 3, 4, 5]),
    (27, [2, 3, 4]),
    (28, [2, 3, 4, 5]),
    # (56, [2]),
    # (58, [2]),
]

# include everything we have data for
# listelements = readhillierdata.extend_ion_list(listelements)
# listelements = readdreamdata.extend_ion_list(listelements)

USE_QUB_COBALT = False

ryd_to_ev = u.rydberg.to('eV')
hc_in_ev_cm = (const.h * const.c).to('eV cm').value
hc_in_ev_angstrom = (const.h * const.c).to('eV angstrom').value
h_in_ev_seconds = const.h.to('eV s').value


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot estimated spectra from bound-bound transitions.')
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
            '-nphixspoints', type=int, default=100,
            help='Number of cross section points to save in output')
        parser.add_argument(
            '-phixsnuincrement', type=float, default=0.03,
            help='Fraction of nu_edge incremented for each cross section point')
        parser.add_argument(
            '-optimaltemperature', type=int, default=6000,
            help='(Electron and excitation) temperature at which recombination rate '
                 'should be constant when downsampling cross sections')
        parser.add_argument(
            '-electrontemperature', type=int, default=6000,
            help='Temperature for choosing effective collision strengths')
        parser.add_argument(
            '--nophixs', action='store_true',
            help='Don''t generate cross sections and write to phixsdata_v2.txt file')
        parser.add_argument(
            '--plotphixs', action='store_true',
            help='Generate cross section plots')

        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    readhillierdata.read_hyd_phixsdata()

    os.makedirs(args.output_folder, exist_ok=True)
    log_folder = os.path.join(args.output_folder, args.output_folder_logs)
    if os.path.exists(log_folder):
        # delete any existing log files
        logfiles = glob.glob(os.path.join(log_folder, '*.txt'))
        for logfile in logfiles:
            os.remove(logfile)
            print(logfile)
    else:
        os.makedirs(log_folder, exist_ok=True)

    write_compositionfile(listelements, args)
    clear_files(args)
    process_files(listelements, args)


def clear_files(args):
    # clear out the file contents, so these can be appended to later
    with open(os.path.join(args.output_folder, 'adata.txt'), 'w'), \
            open(os.path.join(args.output_folder, 'transitiondata.txt'), 'w'):
        if not args.nophixs:
            with open(os.path.join(args.output_folder, 'phixsdata_v2.txt'), 'w') as fphixs:
                fphixs.write(f'{args.nphixspoints:d}\n')
                fphixs.write(f'{args.phixsnuincrement:14.7e}\n')


def process_files(listelements, args):
    for elementindex, (atomic_number, listions) in enumerate(listelements):
        if not listions:
            continue

        nahar_core_states = [['IGNORE'] for x in listions]  # list of named tuples (naharcorestaterow)
        hillier_photoion_targetconfigs = [[] for x in listions]

        # keys are (2S+1, L, parity), values are strings of electron configuration
        nahar_configurations = [{} for x in listions]

        # keys are (2S+1, L, parity, indexinsymmetry), values are lists of (energy
        # in Rydberg, cross section in Mb) tuples
        nahar_phixs_tables = [{} for x in listions]

        ionization_energy_ev = [0.0 for x in listions]
        thresholds_ev_dict = [{} for x in listions]

        # list of named tuples (hillier_transition_row)
        transitions = [[] for x in listions]
        transition_count_of_level_name = [{} for x in listions]
        upsilondicts = [{} for x in listions]

        energy_levels = [[] for x in listions]
        # index matches level id
        photoionization_thresholds_ev = [[] for _ in listions]
        photoionization_crosssections = [[] for _ in listions]  # list of cross section in Mb
        photoionization_targetfractions = [[] for _ in listions]

        for i, ion_stage in enumerate(listions):
            logfilepath = os.path.join(args.output_folder, args.output_folder_logs,
                                       f'{elsymbols[atomic_number].lower()}{ion_stage:d}.txt')
            with open(logfilepath, 'w') as flog:
                log_and_print(flog, f'\n===========> {elsymbols[atomic_number]} {roman_numerals[ion_stage]} input:')

                path_nahar_energy_file = f'atomic-data-nahar/{elsymbols[atomic_number].lower()}{ion_stage:d}.en.ls.txt'
                path_nahar_px_file = f'atomic-data-nahar/{elsymbols[atomic_number].lower()}{ion_stage:d}.ptpx.txt'

                # upsilondatafilenames = {(26, 2): 'fe_ii_upsilon-data.txt', (26, 3): 'fe_iii_upsilon-data.txt'}
                # if (atomic_number, ion_stage) in upsilondatafilenames:
                #     upsilonfilename = os.path.join('atomic-data-tiptopbase',
                #                                    upsilondatafilenames[(atomic_number, ion_stage)])
                #     log_and_print(flog, f'Reading effective collision strengths from {upsilonfilename}')
                #     upsilondatadf = pd.read_csv(upsilonfilename,
                #                                 names=["Z", "ion_stage", "lower", "upper", "upsilon"],
                #                                 index_col=False, header=None, sep=" ")
                #     if len(upsilondatadf) > 0:
                #         for _, row in upsilondatadf.iterrows():
                #             lower = int(row['lower'])
                #             upper = int(row['upper'])
                #             if upper < lower:
                #                 print(f'Problem in {upsilondatafilenames[(atomic_number, ion_stage)]}, lower {lower} upper {upper}. Swapping lower and upper')
                #                 old_lower = lower
                #                 lower = upper
                #                 upper = old_lower
                #             if (lower, upper) not in upsilondicts[i]:
                #                 upsilondicts[i][(lower, upper)] = row['upsilon']
                #             else:
                #                 log_and_print(flog, f"Duplicate upsilon value for transition {lower:d} to {upper:d} keeping {upsilondicts[i][(lower, upper)]:5.2e} instead of using {row['upsilon']:5.2e}")

                # if False:
                #     pass

                # Call readHedata for He III
                if atomic_number == 2 and ion_stage == 3:

                    (ionization_energy_ev[i], energy_levels[i], transitions[i],
                     transition_count_of_level_name[i]) = readboyledata.read_levels_and_transitions(
                        atomic_number, ion_stage, flog)

                elif USE_QUB_COBALT and atomic_number == 27:
                    if ion_stage in [3, 4]:  # QUB levels and transitions, or single-level Co IV
                        (ionization_energy_ev[i], energy_levels[i],
                         transitions[i], transition_count_of_level_name[i],
                         upsilondicts[i]) = readqubdata.read_qub_levels_and_transitions(atomic_number, ion_stage, flog)
                    else:  # hillier levels and transitions
                        # if ion_stage == 2:
                        #     upsilondicts[i] = read_storey_2016_upsilondata(flog)
                        (ionization_energy_ev[i], energy_levels[i], transitions[i],
                         transition_count_of_level_name[i], hillier_levelnamesnoJ_matching_term) = readhillierdata.read_levels_and_transitions(
                             atomic_number, ion_stage, flog)

                    if i < len(listions) - 1 and not args.nophixs:  # don't get cross sections for top ion
                        (photoionization_crosssections[i], photoionization_targetfractions[i],
                         photoionization_thresholds_ev[i]) = readqubdata.read_qub_photoionizations(
                            atomic_number, ion_stage, energy_levels[i], args, flog)

                elif False:  # Nahar only, usually just for testing purposes
                    (nahar_energy_levels, nahar_core_states[i],
                     nahar_level_index_of_state, nahar_configurations[i],
                     ionization_energy_ev[i]) = readnahardata.read_nahar_energy_level_file(
                         path_nahar_energy_file, atomic_number, ion_stage, flog)

                    if i < len(listions) - 1:  # don't get cross sections for top ion
                        log_and_print(flog, f'Reading {path_nahar_px_file}')
                        nahar_phixs_tables[i], thresholds_ev_dict[i] = readnahardata.read_nahar_phixs_tables(
                            path_nahar_px_file, atomic_number, ion_stage, args)

                    hillier_levelnamesnoJ_matching_term = defaultdict(list)
                    hillier_transitions = []
                    hillier_energy_levels = ['IGNORE']
                    (energy_levels[i], transitions[i],
                     photoionization_crosssections[i], photoionization_thresholds_ev[i]) = combine_hillier_nahar(
                        hillier_energy_levels, hillier_levelnamesnoJ_matching_term, hillier_transitions,
                        nahar_energy_levels, nahar_level_index_of_state, nahar_configurations[i],
                        nahar_phixs_tables[i], thresholds_ev_dict[i], args, flog, useallnaharlevels=True)

                    print(energy_levels[i][0:3])

                elif False and atomic_number in [26, ]:  # atomic_number in [8, 26] and os.path.isfile(path_nahar_energy_file):  # Hillier/Nahar hybrid
                    (nahar_energy_levels, nahar_core_states[i],
                     nahar_level_index_of_state, nahar_configurations[i],
                     nahar_ionization_potential_rydberg) = readnahardata.read_nahar_energy_level_file(
                         path_nahar_energy_file, atomic_number, ion_stage, flog)

                    (ionization_energy_ev[i], hillier_energy_levels, hillier_transitions,
                     transition_count_of_level_name[i], hillier_levelnamesnoJ_matching_term) = \
                        readhillierdata.read_levels_and_transitions(atomic_number, ion_stage, flog)

                    if i < len(listions) - 1:  # don't get cross sections for top ion
                        log_and_print(flog, f'Reading {path_nahar_px_file}')
                        nahar_phixs_tables[i], thresholds_ev_dict[i] = readnahardata.read_nahar_phixs_tables(
                            path_nahar_px_file, atomic_number, ion_stage, args)

                    (energy_levels[i], transitions[i],
                     photoionization_crosssections[i], photoionization_thresholds_ev[i]) = combine_hillier_nahar(
                        hillier_energy_levels, hillier_levelnamesnoJ_matching_term, hillier_transitions,
                        nahar_energy_levels, nahar_level_index_of_state, nahar_configurations[i],
                        nahar_phixs_tables[i], thresholds_ev_dict[i], args, flog)
                    # reading the collision data (in terms of level names) must be done after the data sets have
                    # been combined so that the level numbers are correct
                    if len(upsilondicts[i]) == 0:
                        upsilondicts[i] = readhillierdata.read_coldata(atomic_number, ion_stage, energy_levels[i], flog, args)

                elif atomic_number <= 56:  # Hillier data only

                    (ionization_energy_ev[i], energy_levels[i], transitions[i],
                     transition_count_of_level_name[i], hillier_levelnamesnoJ_matching_term) = (
                          readhillierdata.read_levels_and_transitions(atomic_number, ion_stage, flog))

                    if len(upsilondicts[i]) == 0:
                        upsilondicts[i] = readhillierdata.read_coldata(
                            atomic_number, ion_stage, energy_levels[i], flog, args)

                    if i < len(listions) - 1 and not args.nophixs:  # don't get cross sections for top ion
                        (photoionization_crosssections[i], hillier_photoion_targetconfigs[i],
                         photoionization_thresholds_ev[i]) = readhillierdata.read_phixs_tables(
                            atomic_number, ion_stage, energy_levels[i], args, flog)

                else:  # DREAM database of Z > 57

                    (ionization_energy_ev[i], energy_levels[i], transitions[i],
                     transition_count_of_level_name[i]) = readdreamdata.read_levels_and_transitions(
                        atomic_number, ion_stage, flog)

        write_output_files(elementindex, energy_levels, transitions, upsilondicts,
                           ionization_energy_ev, transition_count_of_level_name,
                           nahar_core_states, nahar_configurations, hillier_photoion_targetconfigs,
                           photoionization_thresholds_ev, photoionization_targetfractions,
                           photoionization_crosssections, args)


def read_storey_2016_upsilondata(flog):
    upsilondict = {}

    filename = 'atomic-data-storey/storetetal2016-co-ii.txt'
    log_and_print(flog, f'Reading effective collision strengths from {filename}')

    with open(filename, 'r') as fstoreydata:
        found_tablestart = False
        while True:
            line = fstoreydata.readline()
            if not line:
                break

            if found_tablestart:
                row = line.split()

                if len(row) > 5:
                    lower = int(row[0])
                    upper = int(row[1])
                    upsilon = float(row[11])
                    upsilondict[(lower, upper)] = upsilon
                else:
                    break

            if line.startswith('--	--	------	------	------	------	------	------	------	------	------	------	------	------	------'):
                found_tablestart = True

    return upsilondict


def combine_hillier_nahar(hillier_energy_levels, hillier_levelnamesnoJ_matching_term, hillier_transitions,
                          nahar_energy_levels, nahar_level_index_of_state, nahar_configurations, nahar_phixs_tables,
                          thresholds_ev_dict, args, flog, useallnaharlevels=False):
    added_nahar_levels = []
    photoionization_crosssections = []
    photoionization_thresholds_ev = []
    levelids_of_levelnamenoJ = defaultdict(list)

    if useallnaharlevels:
        added_nahar_levels = nahar_energy_levels[1:]
    else:
        for levelid, energy_level in enumerate(hillier_energy_levels[1:], 1):
            levelnamenoJ = energy_level.levelname.split('[')[0]
            levelids_of_levelnamenoJ[levelnamenoJ].append(levelid)

        # match up Nahar states given in phixs data with Hillier levels, adding
        # missing levels as necessary
        def energy_if_available(state_tuple):
            if state_tuple in nahar_level_index_of_state:
                return hc_in_ev_cm * nahar_energy_levels[nahar_level_index_of_state[state_tuple]].energyabovegsinpercm
            else:
                return 999999.

        phixs_state_tuples_sorted = sorted(nahar_phixs_tables.keys(), key=energy_if_available)
        for state_tuple in phixs_state_tuples_sorted:
            twosplusone, l, parity, indexinsymmetry = state_tuple
            hillier_level_ids_matching_this_nahar_state = []

            if state_tuple in nahar_level_index_of_state:
                nahar_energy_level = nahar_energy_levels[nahar_level_index_of_state[state_tuple]]
                nahar_energyabovegsinev = hc_in_ev_cm * nahar_energy_level.energyabovegsinpercm
            # else:
                # nahar_energy_level = None
                # nahar_energyabovegsinev = 999999.

            nahar_configuration_this_state = '_CONFIG NOT FOUND_'
            flog.write("\n")
            if state_tuple in nahar_configurations:
                nahar_configuration_this_state = nahar_configurations[state_tuple]

                if nahar_configuration_this_state.strip() in nahar_configuration_replacements:
                    nahar_configuration_this_state = nahar_configuration_replacements[
                        nahar_configurations[state_tuple].strip()]
                    flog.write(f"Replacing Nahar configuration of '{nahar_configurations[state_tuple]}' with '{nahar_configuration_this_state}'\n")

            if hillier_levelnamesnoJ_matching_term[(twosplusone, l, parity)]:
                # match the electron configurations from the levels with matching terms
                if nahar_configuration_this_state != '_CONFIG NOT FOUND_':
                    level_match_scores = []
                    for levelname in hillier_levelnamesnoJ_matching_term[(twosplusone, l, parity)]:
                        if levelname in hillier_name_replacements:
                            altlevelname = hillier_name_replacements[levelname]
                        else:
                            altlevelname = levelname

                        if hillier_energy_levels[levelids_of_levelnamenoJ[levelname][0]].indexinsymmetry >= 0:
                            # already matched this level to something
                            match_score = 0
                        else:
                            match_score = score_config_match(altlevelname, nahar_configuration_this_state)

                        avghillierenergyabovegsinev = weightedavgenergyinev(hillier_energy_levels, levelids_of_levelnamenoJ[levelname])
                        if nahar_energyabovegsinev < 999:
                            # reduce the score by 30% for every eV of energy difference (up to 100%)
                            match_score *= 1 - min(1, 0.3 * abs(avghillierenergyabovegsinev - nahar_energyabovegsinev))

                        level_match_scores.append([levelname, match_score])

                    level_match_scores.sort(key=lambda x: -x[1])
                    best_match_score = level_match_scores[0][1]
                    if best_match_score > 0:
                        best_levelname = level_match_scores[0][0]
                        core_state_id = nahar_energy_levels[nahar_level_index_of_state[state_tuple]].corestateid

                        confignote = nahar_configurations[state_tuple]

                        if nahar_configuration_this_state != confignote:
                            confignote += f' replaced by {nahar_configuration_this_state}'

                        for levelid in levelids_of_levelnamenoJ[best_levelname]:
                            hillierlevel = hillier_energy_levels[levelid]
                            # print(hillierlevel.twosplusone, hillierlevel.l, hillierlevel.parity, hillierlevel.levelname)
                            hillier_energy_levels[levelid] = hillier_energy_levels[levelid]._replace(
                                twosplusone=twosplusone, l=l, parity=parity,
                                indexinsymmetry=indexinsymmetry,
                                corestateid=core_state_id,
                                naharconfiguration=confignote,
                                matchscore=best_match_score)
                            hillier_level_ids_matching_this_nahar_state.append(levelid)
                    else:
                        pass
                        # print("no match for", nahar_configuration_this_state)
                else:
                    log_and_print(flog, f"No electron configuration for {twosplusone:d}{lchars[l]}{['e', 'o'][parity]} index {indexinsymmetry:d}")
            else:
                flog.write(f"No Hillier levels with term {twosplusone:d}{lchars[l]}{['e', 'o'][parity]}\n")

            if not hillier_level_ids_matching_this_nahar_state:
                flog.write(f"No matched Hillier levels for Nahar cross section of {twosplusone:d}{lchars[l]}{['e', 'o'][parity]} index {indexinsymmetry:d} '{nahar_configuration_this_state}' ")

                # now find the Nahar level and add it to the new list
                if state_tuple in nahar_level_index_of_state:
                    nahar_energy_level = nahar_energy_levels[nahar_level_index_of_state[state_tuple]]
                    nahar_energy_eV = nahar_energy_level.energyabovegsinpercm * hc_in_ev_cm
                    flog.write(f'(E = {nahar_energy_eV:.3f} eV, g = {nahar_energy_level.g:.1f})\n')

                    if nahar_energy_eV < 0.002:
                        flog.write(" but prevented duplicating the ground state\n")
                    else:
                        added_nahar_levels.append(nahar_energy_level._replace(naharconfiguration=nahar_configurations.get(state_tuple, 'UNKNOWN CONFIG')))
                else:
                    flog.write(" (and no matching entry in Nahar energy table, so can't be added)\n")
            else:  # there are Hillier levels matched to this state
                nahar_energy_level = nahar_energy_levels[nahar_level_index_of_state[state_tuple]]
                nahar_energyabovegsinev = hc_in_ev_cm * nahar_energy_level.energyabovegsinpercm
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
                    flog.write(f'<E> = {avghillierenergyabovegsinev:.3f} eV, g_sum = {sumhillierstatweights:.1f}: \n')
                    if abs(nahar_energyabovegsinev / avghillierenergyabovegsinev - 1) > 0.5:
                        flog.write(f'ENERGY DISCREPANCY WARNING\n')

                strhilliermatches = '\n'.join(['{0} ({1:.3f} eV, g = {2:.1f}, match_score = {3:.1f})'.format(hillier_energy_levels[k].levelname, hc_in_ev_cm * float(
                    hillier_energy_levels[k].energyabovegsinpercm), hillier_energy_levels[k].g, hillier_energy_levels[k].matchscore) for k in hillier_level_ids_matching_this_nahar_state])

                flog.write(strhilliermatches + '\n')

    energy_levels = hillier_energy_levels + added_nahar_levels

    log_and_print(flog, f'Included {len(hillier_energy_levels) - 1} levels from Hillier dataset and added {len(added_nahar_levels)} levels from Nahar phixs tables for a total of {len(energy_levels) - 1} levels')

    # sort the concatenated energy level list by energy
    print('Sorting levels by energy...')
    energy_levels.sort(key=lambda x: float(getattr(x, 'energyabovegsinpercm', '-inf')))

    if len(nahar_phixs_tables.keys()) > 0:
        photoionization_crosssections = np.zeros((len(energy_levels), args.nphixspoints))  # this probably gets overwritten anyway
        photoionization_thresholds_ev = np.zeros((len(energy_levels)))

        # process the phixs tables and attach them to any matching levels in the output list

        if not args.nophixs:
            reduced_phixs_dict = reduce_phixs_tables(
                nahar_phixs_tables, args.optimaltemperature, args.nphixspoints, args.phixsnuincrement)

            for (twosplusone, l, parity, indexinsymmetry), phixstable in reduced_phixs_dict.items():
                foundamatch = False
                for levelid, energylevel in enumerate(energy_levels[1:], 1):
                    if (int(energylevel.twosplusone) == twosplusone and
                            int(energylevel.l) == l and
                            int(energylevel.parity) == parity and
                            int(energylevel.indexinsymmetry) == indexinsymmetry):
                        photoionization_crosssections[levelid] = phixstable
                        photoionization_thresholds_ev[levelid] = thresholds_ev_dict[(twosplusone, l, parity, indexinsymmetry)]
                        foundamatch = True  # there could be more than one match, but this flags there being at least one

                if not foundamatch:
                    log_and_print(flog, f"No Hillier or Nahar state to match with photoionization crosssection of {twosplusone:d}{lchars[l]}{['e', 'o'][parity]} index {indexinsymmetry:d}")

    return energy_levels, hillier_transitions, photoionization_crosssections, photoionization_thresholds_ev


def log_and_print(flog, strout):
    print(strout)
    flog.write(strout + "\n")


def isfloat(value):
    try:
        float(value.replace('D', 'E'))
        return True
    except ValueError:
        return False


# split a list into evenly sized chunks
def chunks(listin, chunk_size):
    return [listin[i:i + chunk_size] for i in range(0, len(listin), chunk_size)]


def reduce_phixs_tables(dicttables, optimaltemperature, nphixspoints, phixsnuincrement, hideoutput=False):
    """
        Receives a dictionary, with each item being a 2D array of energy and cross section points
        Returns a dictionary with the items having been downsampled into a 1D array

        Units don't matter, but the first (lowest) energy point is assumed to be the threshold energy
    """
    out_q = mp.Queue()
    procs = []

    if not hideoutput:
        print(f"Processing {len(dicttables.keys()):d} phixs tables")
    nprocs = os.cpu_count()
    keylist = dicttables.keys()
    for procnum in range(nprocs):
        dicttablesslice = itertools.islice(dicttables.items(), procnum, len(keylist), nprocs)
        procs.append(mp.Process(target=reduce_phixs_tables_worker, args=(
            dicttablesslice, optimaltemperature, nphixspoints, phixsnuincrement, out_q)))
        procs[-1].start()

    dictout = {}
    for procnum in range(len(procs)):
        subdict = out_q.get()
        # print("a process returned {:d} items".format(len(subdict.keys())))
        dictout.update(subdict)

    for proc in procs:
        proc.join()

    return dictout


# this method downsamples the photoionization cross section table to a
# regular grid while keeping the recombination rate integral constant
# (assuming that the temperature matches)
def reduce_phixs_tables_worker(
    dicttables, optimaltemperature, nphixspoints, phixsnuincrement, out_q):

    dictout = {}

    ryd_to_hz = (u.rydberg / const.h).to('Hz').value
    h_over_kb_in_K_sec = (const.h / const.k_B).to('K s').value

    # proportional to recombination rate
    # nu0 = 1e16
    # fac = math.exp(h_over_kb_in_K_sec * nu0 / optimaltemperature)

    def integrand(nu):
        return (nu ** 2) * math.exp(- h_over_kb_in_K_sec * nu / optimaltemperature)

    # def integrand_vec(nu_list):
    #    return [(nu ** 2) * math.exp(- h_over_kb_in_K_sec * (nu - nu0) / optimaltemperature)
    #            for nu in nu_list]

    integrand_vec = np.vectorize(integrand)

    xgrid = np.linspace(1.0, 1.0 + phixsnuincrement * (nphixspoints + 1),
                        num=nphixspoints + 1, endpoint=False)

    # for key in keylist:
    #   tablein = dicttables[key]
    for key, tablein in dicttables:
        # # filter zero points out of the table
        # firstnonzeroindex = 0
        # for i, point in enumerate(tablein):
        #     if point[1] != 0.:
        #         firstnonzeroindex = i
        #         break
        # if firstnonzeroindex != 0:
        #     tablein = tablein[firstnonzeroindex:]

        # table says zero threshold, so avoid divide by zero
        if tablein[0][0] == 0.:
            dictout[key] = np.zeros(nphixspoints)
            continue

        threshold_old_ryd = tablein[0][0]
        # tablein is an array of pairs (energy, phixs cross section)

        # nu0 = tablein[0][0] * ryd_to_hz

        arr_sigma_out = np.empty(nphixspoints)
        # x is nu/nu_edge

        sigma_interp = interpolate.interp1d(tablein[:, 0], tablein[:, 1], kind='linear', assume_sorted=True)

        for i, _ in enumerate(xgrid[:-1]):
            iprevious = max(i - 1, 0)
            enlow = 0.5 * (xgrid[iprevious] + xgrid[i]) * threshold_old_ryd
            enhigh = 0.5 * (xgrid[i] + xgrid[i + 1]) * threshold_old_ryd

            # start of interval interpolated point, Nahar points, and end of interval interpolated point
            samples_in_interval = tablein[(enlow <= tablein[:, 0]) & (tablein[:, 0] <= enhigh)]

            if len(samples_in_interval) == 0 or ((samples_in_interval[0, 0] - enlow)/enlow) > 1e-20:
                if i == 0:
                    if len(samples_in_interval) != 0:
                        print('adding first point {0:.4e} {1:w.4e} {2:.4e}'.format(
                            enlow, samples_in_interval[0, 0], ((samples_in_interval[0, 0] - enlow) / enlow)))
                if enlow <= tablein[-1][0]:
                    new_crosssection = sigma_interp(enlow)
                    if new_crosssection < 0:
                        print('negative extrap')
                else:
                    # assume power law decay after last point
                    new_crosssection = tablein[-1][1] * (tablein[-1][0] / enlow) ** 3
                samples_in_interval = np.vstack([[enlow, new_crosssection], samples_in_interval])

            if len(samples_in_interval) == 0 or ((enhigh - samples_in_interval[-1, 0]) / samples_in_interval[-1, 0]) > 1e-20:
                if enhigh <= tablein[-1][0]:
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

            if nsamples >= 50 or enlow > tablein[-1][0]:
                arr_energyryd = samples_in_interval[:, 0]
                arr_sigma_megabarns = samples_in_interval[:, 1]
            else:
                nsteps = 50  # was 500
                arr_energyryd = np.linspace(enlow, enhigh, num=nsteps, endpoint=False)
                arr_sigma_megabarns = np.interp(arr_energyryd, tablein[:, 0], tablein[:, 1])

            integrand_vals = integrand_vec(arr_energyryd * ryd_to_hz)
            if np.any(integrand_vals):
                sigma_integrand_vals = [sigma * integrand_val
                                        for sigma, integrand_val
                                        in zip(arr_sigma_megabarns, integrand_vals)]

                integralnosigma = integrate.trapz(integrand_vals, arr_energyryd)
                integralwithsigma = integrate.trapz(sigma_integrand_vals, arr_energyryd)

            else:
                integralnosigma = 1.0
                integralwithsigma = np.average(arr_sigma_megabarns)

            if integralwithsigma > 0 and integralnosigma > 0:
                arr_sigma_out[i] = (integralwithsigma / integralnosigma)
            elif integralwithsigma == 0:
                arr_sigma_out[i] = 0.
            else:
                print('Math error: ', i, nsamples, arr_sigma_megabarns[i], integralwithsigma, integralnosigma)
                print(samples_in_interval)
                print(arr_sigma_out[i - 1])
                print(arr_sigma_out[i])
                print(arr_sigma_out[i + 1])
                arr_sigma_out[i] = 0.
                # sys.exit()

        dictout[key] = arr_sigma_out  # output a 1D list of cross sections

    # return dictout
    out_q.put(dictout)


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
        genergysum += statisticalweight * hc_in_ev_angstrom / float(energylevels_thision[levelid].lambdaangstrom)
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
            print(f"WARNING: Can't read parity from JJ coupling state '{config}'")
            return (-1, -1, -1)

    lposition = -1
    for charpos, char in reversed(list(enumerate(config))):
        if char in lchars:
            lposition = charpos
            l = lchars.index(char)
            break
    if lposition < 0:
        if config[-1] == 'e':
            return (-1, -1, 0)
        elif config[-1] == 'o':
            return (-1, -1, 1)
    try:
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
    except ValueError:
        twosplusone = -1
        l = -1
        parity = -1
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

    if lposition < len(strin) - 1 and strin[lposition + 1:] not in ['e', 'o']:
        jvalue = int(strin[lposition + 1:])
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
    """
        Operates on a string by removing anything between parentheses (including the parentheses)
        e.g. remove_bracketed_part('AB(CD)EF') = 'ABEF'
    """
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
    elif indexinsymmetry_a != -1 and indexinsymmetry_b != -1 and ('0s' not in electron_config_a and '0s' not in electron_config_b):
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
        if '0s' in electron_config_a or '0s' in electron_config_b:
            matched_pieces += 0.5  # make sure 0s states gets matched to something
        index_a, index_b = 0, 0

        non_term_pieces_a = sum([1 for a in electron_config_a if not a.startswith('(')])
        non_term_pieces_b = sum([1 for b in electron_config_b if not b.startswith('(')])
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
                # elif '0s' in [piece_a, piece_b]:  # wildcard piece
                #     matched_pieces += 0.5
                    # pass
                # else:
                #     return 0

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
                       hillier_photoion_targetconfigs,
                       photoionization_thresholds_ev,
                       photoionization_targetfractions,
                       photoionization_crosssections, args):
    atomic_number, listions = listelements[elementindex]
    upsilon_transition_row = namedtuple('transition', 'lowerlevel upperlevel A nameto namefrom lambdaangstrom coll_str')

    for i, ion_stage in enumerate(listions):
        upsilondict = upsilondicts[i]
        ionstr = f'{elsymbols[atomic_number]} {roman_numerals[ion_stage]}'

        flog = open(os.path.join(args.output_folder, args.output_folder_logs,
                                 f'{elsymbols[atomic_number].lower()}{ion_stage:d}.txt'), 'a')

        log_and_print(flog, '\n===========> ' + ionstr + ' output:')

        level_id_of_level_name = {}
        for levelid in range(1, len(energy_levels[i])):
            if hasattr(energy_levels[i][levelid], 'levelname'):
                level_id_of_level_name[energy_levels[i][levelid].levelname] = levelid

        unused_upsilon_transitions = set(upsilondicts[i].keys())  # start with the full set and remove used ones
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

        log_and_print(flog, f"Adding in {len(unused_upsilon_transitions):d} extra transitions with only upsilon values")
        for (id_lower, id_upper) in unused_upsilon_transitions:
            namefrom = energy_levels[i][id_upper].levelname
            nameto = energy_levels[i][id_lower].levelname
            A = 0.
            try:
                lamdaangstrom = 1.e8 / (energy_levels[i][id_upper].energyabovegsinpercm - energy_levels[i][id_lower].energyabovegsinpercm)
            except ZeroDivisionError:
                lamdaangstrom = -1
            # upsilon = upsilondict[(id_lower, id_upper)]
            transition_count_of_level_name[i][namefrom] += 1
            transition_count_of_level_name[i][nameto] += 1
            coll_str = upsilondict[(id_lower, id_upper)]

            transition = upsilon_transition_row(id_lower, id_upper, A, namefrom, nameto, lamdaangstrom, coll_str)
            transitions[i].append(transition)

        transitions[i].sort(key=lambda x: (getattr(x, 'lowerlevel', -99), getattr(x, 'upperlevel', -99)))

        with open(os.path.join(args.output_folder, 'adata.txt'), 'a') as fatommodels:
            write_adata(fatommodels, atomic_number, ion_stage, energy_levels[i], ionization_energies[i], transition_count_of_level_name[i], args, flog)

        with open(os.path.join(args.output_folder, 'transitiondata.txt'), 'a') as ftransitiondata:
            write_transition_data(ftransitiondata, atomic_number, ion_stage, energy_levels[i], transitions[i], upsilondicts[i], args, flog)

        if i < len(listions) - 1 and not args.nophixs:  # ignore the top ion
            if len(photoionization_targetfractions[i]) < 1:
                if len(nahar_core_states[i]) > 1:
                    photoionization_targetfractions[i] = readnahardata.get_photoiontargetfractions(energy_levels[i], energy_levels[i+1], nahar_core_states[i], nahar_configurations[i + 1], flog)
                else:
                    photoionization_targetfractions[i] = readhillierdata.get_photoiontargetfractions(energy_levels[i], energy_levels[i+1], hillier_photoion_targetconfigs[i], flog)

            with open(os.path.join(args.output_folder, 'phixsdata_v2.txt'), 'a') as fphixs:
                write_phixs_data(fphixs, atomic_number, ion_stage, energy_levels[i], photoionization_crosssections[i], photoionization_targetfractions[i], photoionization_thresholds_ev[i], args, flog)

        flog.close()


def write_adata(fatommodels, atomic_number, ion_stage, energy_levels, ionization_energy, transition_count_of_level_name, args, flog):
    log_and_print(flog, "Writing to 'adata.txt'")
    fatommodels.write(f'{atomic_number:12d}{ion_stage:12d}{len(energy_levels) - 1:12d}{ionization_energy:15.7f}\n')

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
            level_comment = hlevelname.ljust(27)
        except AttributeError:
            level_comment = " " * 27

        try:
            if energylevel.indexinsymmetry >= 0:
                level_comment += f'Nahar: {energylevel.twosplusone:d}{lchars[energylevel.l]:}{["e", "o"][energylevel.parity]:} index {energylevel.indexinsymmetry:}'
                try:
                    config = energylevel.naharconfiguration
                    if energylevel.naharconfiguration.strip() in nahar_configuration_replacements:
                        config += f' replaced by {nahar_configuration_replacements[energylevel.naharconfiguration.strip()]}'
                    level_comment += f" '{config}'"
                except AttributeError:
                    level_comment += ' (no config)'
        except AttributeError:
            level_comment = level_comment.rstrip()

        fatommodels.write(f'{levelid:5d} {hc_in_ev_cm * float(energylevel.energyabovegsinpercm):19.16f} {float(energylevel.g):8.3f} {transitioncount:4d} {level_comment:}\n')

    fatommodels.write('\n')


def write_transition_data(ftransitiondata, atomic_number, ion_stage, energy_levels,
                          transitions, upsilondict, args, flog):
    log_and_print(flog, "Writing to 'transitiondata.txt'")

    num_forbidden_transitions = 0
    num_collision_strengths_applied = 0
    ftransitiondata.write(f'{atomic_number:7d}{ion_stage:7d}{len(transitions):12d}\n')

    level_ids_with_permitted_down_transitions = set()
    for transition in transitions:
        levelid_lower = transition.lowerlevel
        levelid_upper = transition.upperlevel
        forbidden = (energy_levels[levelid_lower].parity == energy_levels[levelid_upper].parity)

        if not forbidden:
            level_ids_with_permitted_down_transitions.add(levelid_upper)

    for transition in transitions:
        levelid_lower = transition.lowerlevel
        levelid_upper = transition.upperlevel
        coll_str = transition.coll_str

        if coll_str > 0:
            num_collision_strengths_applied += 1

        forbidden = (energy_levels[levelid_lower].parity == energy_levels[levelid_upper].parity)

        if forbidden:
            num_forbidden_transitions += 1
            # flog.write(f'Forbidden transition: lambda_angstrom= {float(transition.lambdaangstrom):7.1f}, {transition.namefrom:25s} to {transition.nameto:25s}\n')

        # ftransitiondata.write('{0:4d} {1:4d} {2:16.10E} {3:9.2e} {4:d}\n'.format(
        #     levelid_lower, levelid_upper, float(transition.A), coll_str, forbidden))
        ftransitiondata.write(f'{levelid_lower:4d} {levelid_upper:4d} {float(transition.A):16.10E} {coll_str:9.2e} {forbidden:d}\n')

    ftransitiondata.write('\n')

    log_and_print(flog, f'Output {len(transitions):d} transitions of which {num_forbidden_transitions:d} are forbidden and {num_collision_strengths_applied:d} have collision strengths')


def write_phixs_data(fphixs, atomic_number, ion_stage, energy_levels,
                     photoionization_crosssections, photoionization_targetfractions,
                     photoionization_thresholds_ev, args, flog):
    log_and_print(flog, "Writing to 'phixsdata2.txt'")
    flog.write(f'Downsampling cross sections assuming T={args.optimaltemperature} Kelvin, '
               f'nphixspoints={args.nphixspoints}, phixsnuincrement={args.phixsnuincrement}\n')

    if photoionization_crosssections[1][0] == 0.:
        log_and_print(flog, 'ERROR: ground state has zero photoionization cross section')
        sys.exit()

    for lowerlevelid, targetlist in enumerate(photoionization_targetfractions[1:], 1):
        threshold_ev = photoionization_thresholds_ev[lowerlevelid]
        if len(targetlist) <= 1 and targetlist[0][1] > 0.99:
            if len(targetlist) > 0:
                upperionlevelid = targetlist[0][0]
            else:
                upperionlevelid = 1

            fphixs.write(f'{atomic_number:12d}{ion_stage + 1:12d}{upperionlevelid:8d}{ion_stage:12d}{lowerlevelid:8d}{threshold_ev:16.6E}\n')
        else:
            fphixs.write(f'{atomic_number:12d}{ion_stage + 1:12d}{-1:8d}{ion_stage:12d}{lowerlevelid:8d}{threshold_ev:16.6E}\n')
            fphixs.write(f'{len(targetlist):8d}\n')
            probability_sum = 0.
            for upperionlevelid, targetprobability in targetlist:
                fphixs.write(f'{upperionlevelid:8d}{targetprobability:12f}\n')
                probability_sum += targetprobability
            if abs(probability_sum - 1.) > 0.00001:
                print(f'STOP! phixs fractions sum to {probability_sum:.5f} != 1.0')
                print(targetlist)
                print(f'level id {lowerlevelid} {energy_levels[lowerlevelid].levelname}')
                sys.exit()

        for crosssection in photoionization_crosssections[lowerlevelid]:
            fphixs.write(f'{crosssection:16.8E}\n')


def write_compositionfile(listelements, args):
    with open(os.path.join(args.output_folder, 'compositiondata.txt'), 'w') as fcomp:
        fcomp.write(f'{len(listelements):d}\n')
        fcomp.write(f'0\n0\n')
        for (atomic_number, listions) in listelements:
            ion_stage_min = min(listions)
            ion_stage_max = max(listions)
            nions = ion_stage_max - ion_stage_min + 1
            fcomp.write(f'{atomic_number:d}  {nions:d}  {ion_stage_min:d}  {ion_stage_max:d}  '
                        f'-1 0.0 {atomic_weights[atomic_number]:.4f}\n')


if __name__ == "__main__":
    # print(interpret_configuration('3d64s_4H'))
    # print(interpret_configuration('3d6(3H)4sa4He[11/2]'))
    # print(score_config_match('3d64s_4H','3d6(3H)4sa4He[11/2]'))
    main()

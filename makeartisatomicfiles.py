#!/usr/bin/env python3
from collections import namedtuple
import itertools
import os
import sys
import math
import multiprocessing as mp
import argparse

from astropy import constants as const
from astropy import units as u
import numpy as np
import pandas as pd
from scipy import integrate
from scipy import interpolate
from manual_matches import nahar_configuration_replacements, hillier_name_replacements
import readnahardata
import readqubdata
import readhillierdata

PYDIR = os.path.dirname(os.path.abspath(__file__))
elsymbols = ['n'] + list(pd.read_csv(os.path.join(PYDIR, 'elements.csv'))['symbol'].values)

roman_numerals = (
    '', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
    'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX'
)

listelements = [
    (8, [1, 2, 3]),
    (26, [1, 2, 3, 4, 5]),
    (27, [2, 3, 4]),
    # (28, [1, 2]),
]

ryd_to_ev = u.rydberg.to('eV')

hc_in_ev_cm = (const.h * const.c).to('eV cm').value
hc_in_ev_angstrom = (const.h * const.c).to('eV angstrom').value
h_in_ev_seconds = const.h.to('eV s').value


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
            with open(logfilepath, 'w') as flog:
                log_and_print(flog, '\n==============> {0} {1}:'.format(elsymbols[atomic_number],
                                                                        roman_numerals[ion_stage]))

                upsilondatafilenames = {(26, 2): 'fe_ii_upsilon-data.txt', (26, 3): 'fe_iii_upsilon-data.txt'}
                if (atomic_number, ion_stage) in upsilondatafilenames:
                    upsilonfilename = os.path.join('atomic-data-tiptopbase',
                                                   upsilondatafilenames[(atomic_number, ion_stage)])
                    log_and_print(flog, 'Reading effective collision strengths from ' + upsilonfilename)
                    upsilondatadf = pd.read_csv(upsilonfilename,
                                                names=["Z", "ion_stage", "lower", "upper", "upsilon"],
                                                index_col=False, header=None, sep=" ")
                    if len(upsilondatadf) > 0:
                        for _, row in upsilondatadf.iterrows():
                            lower = int(row['lower'])
                            upper = int(row['upper'])
                            if upper < lower:
                                print("Problem in {0}, lower {1} upper {2}. Swapping lower and upper".format(
                                    upsilondatafilenames[(atomic_number, ion_stage)], lower, upper))
                                old_lower = lower
                                lower = upper
                                upper = old_lower
                            if (lower, upper) not in upsilondicts[i]:
                                upsilondicts[i][(lower, upper)] = row['upsilon']
                            else:
                                log_and_print(flog, "Duplicate upsilon value for transition {0:d} to {1:d} keeping {2:5.2e} instead of using {3:5.2e}".format(
                                    lower, upper, upsilondicts[i][(lower, upper)], row['upsilon']))

                if atomic_number == 27:
                    if ion_stage in [3, 4]:
                        (ionization_energy_ev[i], energy_levels[i],
                         transitions[i], transition_count_of_level_name[i],
                         upsilondicts[i]) = readqubdata.read_qub_levels_and_transitions(atomic_number, ion_stage, flog)
                    else:
                        (ionization_energy_ev[i], energy_levels[i],
                         transitions[i], transition_count_of_level_name[i],
                         hillier_level_ids_matching_term) = readhillierdata.read_levels_and_transitions(
                             atomic_number, ion_stage, flog)

                        if ion_stage == 2:
                            upsilondicts[i] = read_storey_2016_upsilondata(flog)

                    if i < len(listions) - 1 and not args.nophixs:  # don't get cross sections for top ion
                        photoionization_crosssections[i], photoionization_targetfractions[i] = readqubdata.read_qub_photoionizations(atomic_number, ion_stage, energy_levels[i], args, flog)

                else:
                    path_nahar_energy_file = 'atomic-data-nahar/{0}{1:d}.en.ls.txt'.format(
                        elsymbols[atomic_number].lower(), ion_stage)

                    (nahar_energy_levels, nahar_core_states[i],
                     nahar_level_index_of_state, nahar_configurations[i]) = readnahardata.read_nahar_energy_level_file(
                         path_nahar_energy_file, atomic_number, i, ion_stage, flog)

                    if (atomic_number, ion_stage) in readhillierdata.path_hillier_osc_file:

                        (ionization_energy_ev[i], hillier_energy_levels,
                         hillier_transitions,
                         transition_count_of_level_name[i],
                         hillier_level_ids_matching_term) = \
                            readhillierdata.read_levels_and_transitions(atomic_number, ion_stage, flog)

                    if i < len(listions) - 1:  # don't get cross sections for top ion
                        path_nahar_px_file = 'atomic-data-nahar/{0}{1:d}.px.txt'.format(
                            elsymbols[atomic_number].lower(), ion_stage)

                        log_and_print(flog, 'Reading ' + path_nahar_px_file)
                        nahar_phixs_tables[i] = readnahardata.read_nahar_phixs_tables(path_nahar_px_file, atomic_number, ion_stage, args)

                    (energy_levels[i], transitions[i], photoionization_crosssections[i]) = combine_hillier_nahar(
                        i, hillier_energy_levels, hillier_level_ids_matching_term, hillier_transitions,
                        nahar_energy_levels, nahar_level_index_of_state, nahar_configurations[i],
                        nahar_phixs_tables[i], args, flog)

                    # Alternatively use Hillier phixs tables, but BEWARE this
                    # probably doesn't work anymore since the code has changed a lot
                    # print('Reading ' + hillier_ion_folder + path_hillier_phixs_file[(atomic_number, ion_stage)])
                    # read_hillier_phixs_tables(hillier_ion_folder, path_hillier_phixs_file[(atomic_number, ion_stage)], atomic_number, ion_stage)

        write_output_files(elementindex, energy_levels, transitions, upsilondicts,
                           ionization_energy_ev, transition_count_of_level_name,
                           nahar_core_states, nahar_configurations,
                           photoionization_targetfractions, photoionization_crosssections, args)


def read_storey_2016_upsilondata(flog):
    upsilondict = {}

    filename = 'atomic-data-storey/storetetal2016-co-ii.txt'
    log_and_print(flog, 'Reading effective collision strengths from ' + filename)

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


def combine_hillier_nahar(i, hillier_energy_levels, hillier_level_ids_matching_term, hillier_transitions,
                          nahar_energy_levels, nahar_level_index_of_state, nahar_configurations, nahar_phixs_tables,
                          args, flog):
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
                log_and_print(flog, "No electron configuration for {0:d}{1}{2} index {3:d}".format(
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

    log_and_print(flog, 'Included {0} levels from Hillier dataset and added {1} levels from Nahar phixs tables for a total of {2} levels'.format(
        len(hillier_energy_levels) - 1, len(added_nahar_levels), len(energy_levels) - 1))

    # sort the concatenated energy level list by energy
    print('Sorting level list...')
    energy_levels.sort(key=lambda x: float(getattr(x, 'energyabovegsinpercm', '-inf')))

    if len(nahar_phixs_tables.keys()) > 0:
        photoionization_crosssections = np.zeros((len(energy_levels), args.nphixspoints))  # this probably gets overwritten anyway

        if not args.nophixs:
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
                    log_and_print(flog, "No Hillier or Nahar state to match with photoionization crosssection of {0:d}{1}{2} index {3:d}".format(
                        twosplusone, lchars[l], ['e', 'o'][parity], indexinsymmetry))

    return energy_levels, hillier_transitions, photoionization_crosssections


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


def reduce_phixs_tables(dicttables, args):
    out_q = mp.Queue()
    procs = []

    print("Processing {0:d} phixs tables".format(len(dicttables.keys())))
    nprocs = os.cpu_count()
    keylist = dicttables.keys()
    for procnum in range(nprocs):
        dicttablesslice = itertools.islice(dicttables.items(), procnum, len(keylist), nprocs)
        procs.append(mp.Process(target=reduce_phixs_tables_worker, args=(dicttablesslice, args, out_q)))
        procs[-1].start()

    dictout = {}
    for procnum in range(len(procs)):
        subdict = out_q.get()
        dictout.update(subdict)
        # print("a process returned {:d} items".format(len(subdict.keys())))

    for proc in procs:
        proc.join()

    return dictout


# this method downsamples the photoionization cross section table to a
# regular grid while keeping the recombination rate integral constant
# (assuming that the temperature matches)
def reduce_phixs_tables_worker(dicttables, args, out_q):
    dictout = {}

    ryd_to_hz = (u.rydberg / const.h).to('Hz').value
    h_over_kb_in_K_sec = (const.h / const.k_B).to('K s').value

    # proportional to recombination rate
    # nu0 = 1e13
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

    # for key in keylist:
    #   tablein = dicttables[key]
    for key, tablein in dicttables:
        # tablein is an array of pairs (energy, phixs cross section)

        # nu0 = tablein[0][0] * ryd_to_hz

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
                if enlow <= tablein[-1][0]:
                    new_crosssection = sigma_interp(enlow)
                    if new_crosssection < 0:
                        print('negative extrap')
                else:
                    new_crosssection = tablein[-1][1] * (tablein[-1][0] / enlow) ** 3  # assume power law decay after last point
                samples_in_interval = np.vstack([[enlow, new_crosssection], samples_in_interval])

            if len(samples_in_interval) == 0 or ((enhigh - samples_in_interval[-1, 0])/samples_in_interval[-1, 0]) > 1e-20:
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
            log_and_print(flog, "Can't read parity from JJ coupling state '" + config + "'")
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
    atomic_number, listions = listelements[elementindex]
    upsilon_transition_row = namedtuple('transition', 'lowerlevel upperlevel A nameto namefrom lambdaangstrom coll_str')

    print('\nStarting output stage:')
    with open(os.path.join(args.output_folder_transition_guide, 'transitions_{}.txt'.format(elsymbols[atomic_number])), 'w') as ftransitionguide:
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

        log_and_print(flog, "Adding in {0:d} extra transitions with upsilon values".format(len(unused_upsilon_transitions)))
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

        with open(os.path.join(args.output_folder, 'adata.txt'), 'a') as fatommodels:
            write_adata(fatommodels, atomic_number, ion_stage, energy_levels[i], ionization_energies[i], transition_count_of_level_name[i], args, flog)

        with open(os.path.join(args.output_folder, 'transitiondata.txt'), 'a') as ftransitiondata, \
                open(os.path.join(args.output_folder_transition_guide, 'transitions_{}.txt'.format(elsymbols[atomic_number])), 'a') as ftransitionguide:
            write_transition_data(ftransitiondata, ftransitionguide, atomic_number, ion_stage, energy_levels[i], transitions[i], upsilondicts[i], args, flog)

        if i < len(listions) - 1 and not args.nophixs:  # ignore the top ion
            if len(photoionization_targetfractions[i]) < 1:
                photoionization_targetfractions[i] = readnahardata.get_nahar_targetfractions(i, energy_levels[i], energy_levels[i+1], nahar_core_states[i], nahar_configurations, flog)
            with open(os.path.join(args.output_folder, 'phixsdata_v2.txt'), 'a') as fphixs:
                write_phixs_data(fphixs, i, atomic_number, ion_stage, energy_levels[i], photoionization_crosssections[i], photoionization_targetfractions[i], args, flog)

        flog.close()


def write_adata(fatommodels, atomic_number, ion_stage, energy_levels, ionization_energy, transition_count_of_level_name, args, flog):
    log_and_print(flog, "writing to 'adata.txt'")
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
            level_comment = hlevelname.ljust(27)
        except AttributeError:
            level_comment = " " * 27

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
            level_comment = level_comment.rstrip()

        fatommodels.write('{0:5d} {1:19.16f} {2:7.3f} {3:4d} {4:}\n'.format(
            levelid, hc_in_ev_cm * float(energylevel.energyabovegsinpercm),
            float(energylevel.g), transitioncount, level_comment))

    fatommodels.write('\n')


def write_transition_data(ftransitiondata, ftransitionguide, atomic_number, ion_stage, energy_levels,
                          transitions, upsilondict, args, flog):
    log_and_print(flog, "writing to 'transitiondata.txt'")

    num_forbidden_transitions = 0
    num_collision_strengths_applied = 0
    ftransitiondata.write('{0:7d}{1:7d}{2:12d}\n'.format(atomic_number, ion_stage, len(transitions)))

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

        ftransitiondata.write('{0:4d} {1:4d} {2:16.10E} {3:9.2e} {4:d}\n'.format(
            levelid_lower, levelid_upper, float(transition.A), coll_str, forbidden))

    ftransitiondata.write('\n')

    log_and_print(flog, 'Wrote out {0:d} transitions, of which {1:d} are forbidden and {2:d} had collision strengths'.format(
        len(transitions), num_forbidden_transitions, num_collision_strengths_applied))


def write_phixs_data(fphixs, i, atomic_number, ion_stage, energy_levels,
                     photoionization_crosssections, photoionization_targetfractions, args, flog):
    log_and_print(flog, "writing to 'phixsdata2.txt'")
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



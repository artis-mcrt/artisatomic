#!/usr/bin/env python3

from collections import namedtuple, defaultdict
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
import makeartisatomicfiles as artisatomic
ryd_to_ev = u.rydberg.to('eV')

hc_in_ev_cm = (const.h * const.c).to('eV cm').value
hc_in_ev_angstrom = (const.h * const.c).to('eV angstrom').value
h_in_ev_seconds = const.h.to('eV s').value
lchars = 'SPDFGHIKLMNOPQRSTUVWXYZ'


def read_qub_levels_and_transitions(atomic_number, ion_stage, flog):
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
        artisatomic.log_and_print(flog, 'Reading atomic-data-qub/adf04_v1')
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
                    int(line[27:28]), float(line[29:33]), float(line[34:55]), 0.0, 0)
                parity = artisatomic.get_parity_from_config(config)

                levelname = energylevel.levelname + '_{0:d}{1:}{2:}[{3:d}/2]_id={4:}'.format(
                    energylevel.twosplusone, lchars[energylevel.l],
                    ['e', 'o'][parity], int(2 * energylevel.j), energylevel.qub_id)

                g = (2 * energylevel.j + 1)
                energylevel = energylevel._replace(g=g, parity=parity, levelname=levelname)
                qub_energylevels.append(energylevel)

            upsilonheader = fleveltrans.readline().split()
            list_tempheaders = ['upsT={0:}'.format(x) for x in upsilonheader[2:]]
            list_headers = ['upper', 'lower', 'ignore'] + list_tempheaders
            qubupsilondf_alltemps = pd.read_csv(fleveltrans, index_col=False, delim_whitespace=True,
                                                comment="C", names=list_headers,
                                                dtype={'lower': np.int, 'upper': np.int}.update({z: np.float64 for z in list_headers[2:]}),
                                                error_bad_lines=False, skip_blank_lines=True, keep_default_na=False)
            qubupsilondf_alltemps.query('upper!=-1', inplace=True)
            for _, row in qubupsilondf_alltemps.iterrows():
                lower = int(row['lower'])
                upper = int(row['upper'])
                upsilon = float(row['upsT=5.01+03'].replace('-', 'E-').replace('+', 'E+'))
                if (lower, upper) not in upsilondict:
                    upsilondict[(lower, upper)] = upsilon
                else:
                    artisatomic.log_and_print(flog, "Duplicate upsilon value for transition {0:d} to {1:d} keeping {2:5.2e} instead of using {3:5.2e}".format(
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
                    forbidden = artisatomic.check_forbidden(qub_energylevels[id_upper], qub_energylevels[id_lower])
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


def read_qub_photoionizations(atomic_number, ion_stage, energy_levels, args, flog):
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

            reduced_phixs_dict = artisatomic.reduce_phixs_tables(phixstables, args)
            target_scalefactors = np.zeros(ntargets + 1)
            for upperlevelid in reduced_phixs_dict:
                # take the ratio of cross sections at the threshold energyies
                target_scalefactors[upperlevelid] = reduced_phixs_dict[upperlevelid][0]
                # target_scalefactors[upperlevelid] = np.average(reduced_phixs_dict[upperlevelid])

            scalefactorsum = sum(target_scalefactors)
            photoionization_targetfractions[lowerlevelid] = []
            max_fraction = 0.
            upperlevelid_withmaxfraction = 1
            for upperlevelid, target_scalefactor in enumerate(target_scalefactors[1:], 1):
                target_fraction = target_scalefactor / scalefactorsum
                if target_fraction > max_fraction:
                    upperlevelid_withmaxfraction = upperlevelid
                    max_fraction = target_fraction
                if target_fraction > 0.001:
                    photoionization_targetfractions[lowerlevelid].append((upperlevelid, target_fraction))

            photoionization_crosssections[lowerlevelid] = reduced_phixs_dict[upperlevelid_withmaxfraction] / max_fraction
    elif atomic_number == 27 and ion_stage == 3:
        for lowerlevelid in range(1, len(energy_levels)):
            photoionization_targetfractions[lowerlevelid] = [(1, 1.)]
            if lowerlevelid <= 4:
                photoionization_crosssections[lowerlevelid] = np.array([9.3380692, 7.015829602, 5.403975231, 4.250372872, 3.403086443, 2.766835319, 2.279802051, 1.900685772, 1.601177846, 1.361433037, 1.16725865, 1.008321909, 0.8769787, 0.76749151, 0.675496904, 0.597636429, 0.531296609, 0.474423066, 0.425385805, 0.382880364, 0.345854415, 0.313452694, 0.284975256, 0.259845541, 0.237585722, 0.217797532, 0.200147231, 0.184353724, 0.17017913, 0.157421217, 0.145907331, 0.135489462, 0.126040239, 0.117449648, 0.109622338, 0.102475382, 0.095936439, 0.089942202, 0.084437113, 0.079372279, 0.074704554, 0.070395769, 0.066412076, 0.062723384, 0.059302883, 0.056126637, 0.053173226, 0.050423446, 0.047860046, 0.045467498, 0.043231802, 0.041140312, 0.039181587, 0.037345256, 0.035621907, 0.034002983, 0.032480693, 0.031047932, 0.029698215, 0.028425611, 0.027224692, 0.026090478, 0.025018404, 0.02400427, 0.023044216, 0.022134683, 0.021272391, 0.020454314, 0.019677652, 0.018939819, 0.018238416, 0.017571225, 0.016936183, 0.016331377, 0.01575503, 0.015205486, 0.014681206, 0.014180754, 0.013702792, 0.013246071, 0.012809423, 0.012391758, 0.011992055, 0.011609359, 0.011242775, 0.010891464, 0.010554639, 0.010231561, 0.009921535, 0.009623909, 0.009338069, 0.009063438, 0.008799471, 0.008545656, 0.00830151, 0.008066575, 0.007840423, 0.007622646, 0.00741286, 0.007210703])

    return photoionization_crosssections, photoionization_targetfractions

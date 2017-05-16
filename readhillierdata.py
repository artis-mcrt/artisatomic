#!/usr/bin/env python3

import math
import os
import sys
from collections import defaultdict, namedtuple
from contracts import contract

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import makeartisatomicfiles as artisatomic
from manual_matches import hillier_name_replacements

# need to also include collision strengths from e.g., o2col.dat

hillier_rowformat_a = 'levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad gam2 gam4'
hillier_rowformat_b = 'levelname g energyabovegsinpercm freqtentothe15hz thresholdenergyev lambdaangstrom hillierlevelid arad c4 c6'
hillier_rowformat_c = 'levelname g energyabovegsinpercm freqtentothe15hz lambdaangstrom hillierlevelid'

# keys are (atomic number, ion stage)
ion_files = namedtuple('ion_files', ['folder', 'levelstransitionsfilename',
                                     'energylevelrowformat', 'photfilenames', 'coldatafilename'])

ions_data = {
    # H
    (1, 1): ion_files('5dec96', 'hi_osc.dat', hillier_rowformat_c, ['hiphot.dat'], 'hicol.dat'),
    (1, 2): ion_files('', '', hillier_rowformat_c, [''], ''),

    # He
    (2, 1): ion_files('11may07', 'heioscdat_a7.dat_old', hillier_rowformat_a, ['heiphot_a7.dat'], 'heicol.dat'),
    (2, 2): ion_files('5dec96', 'he2_osc.dat', hillier_rowformat_c, ['he2phot.dat'], 'he2col.dat'),

    # C
    (6, 1): ion_files('12dec04', 'ci_split_osc', hillier_rowformat_a, ['phot_smooth_50'], 'cicol.dat'),
    (6, 2): ion_files('30oct12', 'c2osc_rev.dat', hillier_rowformat_b, ['phot_sm_3000.dat'], 'c2col.dat'),
    (6, 3): ion_files('23dec04', 'ciiiosc_st_split_big.dat', hillier_rowformat_a, ['ciiiphot_sm_a_500.dat', 'ciiiphot_sm_b_500.dat'], 'ciiicol.dat'),
    (6, 4): ion_files('30oct12', 'civosc_a12_split.dat', hillier_rowformat_b, ['civphot_a12.dat'], 'civcol.dat'),

    # N
    (7, 1): ion_files('12sep12', 'ni_osc', hillier_rowformat_b, ['niphot_a.dat', 'niphot_b.dat', 'niphot_c.dat', 'niphot_d.dat'], 'ni_col'),
    (7, 2): ion_files('23jan06', 'fin_osc', hillier_rowformat_b, ['phot_sm_3000'], 'n2col.dat'),
    (7, 3): ion_files('24mar07', 'niiiosc_rev.dat', hillier_rowformat_a, ['phot_sm_0_A.dat', 'phot_sm_0_B.dat'], 'niiicol.dat'),

    # O
    (8, 1): ion_files('20sep11', 'oi_osc_mchf', hillier_rowformat_b, ['phot_nosm_A', 'phot_nosm_B'], 'oi_col'),
    (8, 2): ion_files('23mar05', 'o2osc_fin.dat', hillier_rowformat_a, ['phot_sm_3000.dat'], 'o2col.dat'),
    (8, 3): ion_files('15mar08', 'oiiiosc', hillier_rowformat_a, ['phot_sm_0'], 'col_data_oiii_butler_2012.dat'),
    (8, 4): ion_files('19nov07', 'fin_osc', hillier_rowformat_a, ['phot_sm_50_A', 'phot_sm_50_B'], 'col_oiv'),

    # F
    (9, 2): ion_files('tst', 'fin_osc', hillier_rowformat_a, ['phot_data_a', 'phot_data_b', 'phot_data_c'], ''),
    (9, 3): ion_files('tst', 'fin_osc', hillier_rowformat_a, ['phot_data_a', 'phot_data_b', 'phot_data_c', 'phot_data_d'], ''),

    # Ne
    (10, 1): ion_files('9sep11', 'fin_osc', hillier_rowformat_b, ['fin_phot'], 'col_guess'),
    (10, 2): ion_files('19nov07', 'fin_osc', hillier_rowformat_a, ['phot_nosm'], 'col_neii'),
    (10, 3): ion_files('19nov07', 'fin_osc', hillier_rowformat_a, ['phot_nosm'], 'col_neiii'),
    (10, 4): ion_files('1dec99', 'fin_osc.dat', hillier_rowformat_c, ['phot_sm_3000.dat'], 'col_data.dat'),

    # Na
    (11, 1): ion_files('5aug97', 'nai_osc_split.dat', hillier_rowformat_c, ['nai_phot_a.dat'], 'col_guess.dat'),
    (11, 2): ion_files('15feb01', 'na2_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    (11, 3): ion_files('15feb01', 'naiiiosc_rev.dat', hillier_rowformat_a, ['phot_sm_3000.dat'], 'col_guess.dat'),
    (11, 4): ion_files('15feb01', 'naivosc_rev.dat', hillier_rowformat_a, ['phot_sm_3000.dat'], 'col_data.dat'),

    # Mg
    (12, 1): ion_files('5aug97', 'mgi_osc_split.dat', hillier_rowformat_c, ['mgi_phot_a.dat'], 'mgicol.dat'),
    (12, 2): ion_files('30oct12', 'mg2_osc_split.dat', hillier_rowformat_b, ['mg2_phot_a.dat'], 'mg2col.dat'),
    (12, 3): ion_files('20jun01', 'mgiiiosc_rev.dat', hillier_rowformat_a, ['phot_sm_3000.dat'], 'col_guess.dat'),
    (12, 4): ion_files('20jun01', 'mgivosc_rev.dat', hillier_rowformat_a, ['phot_sm_3000.dat'], 'col_guess.dat'),

    # Al
    (13, 1): ion_files('29jul10', 'fin_osc', hillier_rowformat_a, ['phot_smooth_0'], 'col_data'),
    (13, 2): ion_files('5aug97', 'al2_osc_split.dat', hillier_rowformat_c, ['al2_phot_a.dat'], 'al2col.dat'),
    (13, 3): ion_files('30oct12', 'aliii_osc_split.dat', hillier_rowformat_b, ['aliii_phot_a.dat'], 'aliii_col_data.dat'),
    (13, 4): ion_files('23oct02', 'fin_osc', hillier_rowformat_b, ['phot_sm_3000.dat'], 'col_guess'),

    # Si
    (14, 1): ion_files('23nov11', 'SiI_OSC', hillier_rowformat_b, ['SiI_PHOT_DATA'], 'col_data'),
    (14, 2): ion_files('30oct12', 'si2_osc_nahar', hillier_rowformat_b, ['phot_op.dat'], 'si2_col'),
    (14, 3): ion_files('5dec96b', 'osc_op_split_rev.dat_1jun12', hillier_rowformat_a, ['phot_op.dat'], 'col_data'),
    (14, 4): ion_files('30oct12', 'osc_op_split.dat', hillier_rowformat_b, ['phot_op.dat'], 'col_data.dat'),

    # P (IV and V are the only ions in CMFGEN)
    (15, 4): ion_files('15feb01', 'pivosc_rev.dat', hillier_rowformat_a, ['phot_data_a.dat', 'phot_data_b.dat'], 'col_guess.dat'),
    (15, 5): ion_files('15feb01', 'pvosc_rev.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),

    # S
    (16, 1): ion_files('24nov11', 'SI_OSC', hillier_rowformat_b, ['SI_PHOT_DATA'], 'col_data'),
    (16, 2): ion_files('30oct12', 's2_osc', hillier_rowformat_b, ['phot_sm_3000'], 's2_col'),
    (16, 3): ion_files('30oct12', 'siiiosc_fin', hillier_rowformat_a, ['phot_nosm'], 'col_siii'),
    (16, 4): ion_files('19nov07', 'sivosc_fin', hillier_rowformat_a, ['phot_nosm'], 'col_siv'),

    # Cl (only ions IV to VII)
    # (17, 4): ion_files('15feb01', 'clivosc_fin.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_data.dat'),

    # Ar
    (18, 1): ion_files('9sep11', 'fin_osc', hillier_rowformat_b, ['phot_nosm'], 'col_guess'),
    (18, 2): ion_files('9sep11', 'fin_osc', hillier_rowformat_b, ['phot_nosm'], 'col_data'),
    (18, 3): ion_files('19nov07', 'fin_osc', hillier_rowformat_a, ['phot_nosm'], 'col_ariii'),
    (18, 4): ion_files('1dec99', 'fin_osc.dat', hillier_rowformat_c, ['phot_sm_3000.dat'], 'col_data.dat'),

    # K
    (19, 1): ion_files('4mar12', 'fin_osc', hillier_rowformat_b, ['phot_ki'], 'COL_DATA'),
    (19, 2): ion_files('4mar12', 'fin_osc', hillier_rowformat_b, ['phot_k2'], 'COL_DATA'),

    # Ca
    (20, 1): ion_files('5aug97', 'cai_osc_split.dat', hillier_rowformat_c, ['cai_phot_a.dat'], 'caicol.dat'),
    (20, 2): ion_files('30oct12', 'ca2_osc_split.dat', hillier_rowformat_c, ['ca2_phot_a.dat'], 'ca2col.dat'),
    (20, 3): ion_files('10apr99', 'osc_op_sp.dat', hillier_rowformat_c, ['phot_smooth.dat'], 'col_guess.dat'),
    (20, 4): ion_files('10apr99', 'osc_op_sp.dat', hillier_rowformat_c, ['phot_smooth.dat'], 'col_guess.dat'),

    # Sc (only II and III are in CMFGEN)
    (21, 2): ion_files('01jul13', 'fin_osc', hillier_rowformat_a, ['phot_nosm'], 'col_data'),
    (21, 3): ion_files('3dec12', 'fin_osc', hillier_rowformat_a, ['phot_nosm'], ''),

    # Ti (only II and III are in CMFGEN, IV has dummy files with a single level)
    (22, 2): ion_files('18oct00', 'tkii_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    (22, 3): ion_files('18oct00', 'tkiii_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    (22, 4): ion_files('18oct00', 'tkiv_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),

    # V (only V I is in CMFGEN and it has a single level)
    # (23, 1): ion_files('27may10', 'vi_osc', hillier_rowformat_a, ['vi_phot.dat'], 'col_guess.dat'),

    # Cr
    (24, 1): ion_files('10aug12', 'cri_osc.dat', hillier_rowformat_b, ['phot_data.dat'], 'col_guess.dat'),
    (24, 2): ion_files('15aug12', 'crii_osc.dat', hillier_rowformat_b, ['phot_data.dat'], 'col_data.dat'),
    (24, 3): ion_files('18oct00', 'criii_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    (24, 4): ion_files('18oct00', 'criv_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    (24, 5): ion_files('18oct00', 'crv_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),

    # Mn (Mn I is not in CMFGEN)
    (25, 2): ion_files('18oct00', 'mnii_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    (25, 3): ion_files('18oct00', 'mniii_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    (25, 4): ion_files('18oct00', 'mniv_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    (25, 5): ion_files('18oct00', 'mnv_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),

    # Fe
    (26, 1): ion_files('07sep16', 'fei_osc', hillier_rowformat_b, ['phot_smooth_3000'], 'col_data'),
    (26, 2): ion_files('10sep16', 'fe2_osc', hillier_rowformat_b, ['phot_op.dat'], 'fe2_col.dat'),
    (26, 3): ion_files('30oct12', 'FeIII_OSC', hillier_rowformat_b, ['phot_sm_3000.dat'], 'col_data.dat'),
    (26, 4): ion_files('18oct00', 'feiv_osc_rev2.dat', hillier_rowformat_a, ['phot_sm_3000.dat'], 'col_data.dat'),
    (26, 5): ion_files('18oct00', 'fev_osc.dat', hillier_rowformat_a, ['phot_sm_3000.dat'], 'col_guess.dat'),

    # Co
    (27, 2): ion_files('15nov11', 'fin_osc_bound', hillier_rowformat_a, ['phot_nosm'], 'Co2_COL_DATA'),
    (27, 3): ion_files('30oct12', 'coiii_osc.dat', hillier_rowformat_b, ['phot_nosm'], 'col_data.dat'),
    # (27, 4): ion_files('4jan12', 'coiv_osc.dat', hillier_rowformat_b, ['phot_data'], 'col_data.dat'),
    # (27, 5): ion_files('18oct00', 'cov_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),

    # Ni
    (28, 2): ion_files('30oct12', 'nkii_osc.dat', hillier_rowformat_a, ['phot_data'], 'col_data_bautista'),
    (28, 3): ion_files('27aug12', 'nkiii_osc.dat', hillier_rowformat_b, ['phot_data.dat'], 'col_data.dat'),
    (28, 4): ion_files('18oct00', 'nkiv_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),
    (28, 5): ion_files('18oct00', 'nkv_osc.dat', hillier_rowformat_a, ['phot_data.dat'], 'col_guess.dat'),

    # Cu, Zn and above are not in CMGFEN?
}

elsymboltohilliercode = {
    'H': 'HYD', 'He': 'HE', 'C': 'CARB', 'N': 'NIT',
    'O': 'OXY', 'F': 'FLU', 'Ne': 'NEON', 'Na': 'NA',
    'Mg': 'MG', 'Al': 'AL', 'Si': 'SIL', 'P': 'PHOS',
    'S': 'SUL', 'Cl': 'CHL', 'Ar': 'ARG', 'K': 'POT',
    'Ca': 'CA', 'Sc': 'SCAN', 'Ti': 'TIT', 'V': 'VAN',
    'Cr': 'CHRO', 'Mn': 'MAN', 'Fe': 'FE', 'Co': 'COB',
    'Ni': 'NICK'
}

ryd_to_ev = u.rydberg.to('eV')
hc_in_ev_cm = (const.h * const.c).to('eV cm').value
hc_in_ev_angstrom = (const.h * const.c).to('eV angstrom').value
h_in_ev_seconds = const.h.to('eV s').value
lchars = 'SPDFGHIKLMNOPQRSTUVWXYZ'
PYDIR = os.path.dirname(os.path.abspath(__file__))
atomicdata = pd.read_csv(os.path.join(PYDIR, 'atomic_properties.txt'), delim_whitespace=True, comment='#')
elsymbols = ['n'] + list(atomicdata['symbol'].values)

# hilliercodetoelsymbol = {v : k for (k,v) in elsymboltohilliercode.items()}
# hilliercodetoatomic_number = {k : elsymbols.index(v) for (k,v) in hilliercodetoelsymbol.items()}

atomic_number_to_hillier_code = {elsymbols.index(k): v for (k, v) in elsymboltohilliercode.items()}

vy95_phixsfitrow = namedtuple('vy95phixsfit', ['n', 'l', 'E_th_eV', 'E_0', 'sigma_0', 'y_a', 'P', 'y_w'])

# keys are (n, l), values are energy in Rydberg or cross_section in Megabarns
hyd_phixs_energygrid_ryd, hyd_phixs = {}, {}
hyd_gaunt_energygrid_ryd, hyd_gaunt_factor = {}, {}  # keys are n quantum number


def hillier_ion_folder(atomic_number, ion_stage):
    return ('atomic-data-hillier/atomic/' + atomic_number_to_hillier_code[atomic_number] + '/' + artisatomic.roman_numerals[ion_stage] + '/')


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    hillier_energy_levels = ['IGNORE']
    hillier_level_ids_matching_term = defaultdict(list)
    transition_count_of_level_name = defaultdict(int)
    hillier_ionization_energy_ev = 0.0
    transitions = []

    if atomic_number == 1 and ion_stage == 2:
        ionization_energy_ev = 0.
        qub_energy_level_row = namedtuple(
            'energylevel', 'levelname qub_id twosplusone l j energyabovegsinpercm g parity')

        hillier_energy_levels.append(qub_energy_level_row('I', 1, 0, 0, 0, 0., 10, 0))
        return hillier_ionization_energy_ev, hillier_energy_levels, transitions, transition_count_of_level_name, hillier_level_ids_matching_term

    row_format_energy_level = ions_data[(atomic_number, ion_stage)].energylevelrowformat
    filename = os.path.join(hillier_ion_folder(atomic_number, ion_stage),
                            ions_data[(atomic_number, ion_stage)].folder,
                            ions_data[(atomic_number, ion_stage)].levelstransitionsfilename)
    artisatomic.log_and_print(flog, 'Reading ' + filename)
    hillier_energy_level_row = namedtuple(
        'energylevel', row_format_energy_level + ' corestateid twosplusone l parity indexinsymmetry naharconfiguration matchscore')
    hillier_transition_row = namedtuple(
        'transition', 'namefrom nameto f A lambdaangstrom i j hilliertransitionid lowerlevel upperlevel coll_str')

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
                    energyabovegsinpercm=float(hillier_energy_level.energyabovegsinpercm.replace('D', 'E')),
                    g=float(hillier_energy_level.g),
                    twosplusone=twosplusone,
                    l=l,
                    parity=parity
                )

                hillier_energy_levels.append(hillier_energy_level)

                if twosplusone == -1 and atomic_number > 1:
                    # -1 indicates that the term could not be interpreted
                    if parity == -1:
                        artisatomic.log_and_print(flog, f"Can't find LS term in Hillier level name '{levelname}'")
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
                    print(
                        f'Hillier levels mismatch: id {len(hillier_energy_levels) - 1:d} found at entry number {hillierlevelid:d}')
                    sys.exit()

            if line.startswith('                        Oscillator strengths'):
                break

        artisatomic.log_and_print(flog, f'Read {len(hillier_energy_levels[1:]):d} levels')

        # defined_transition_ids = []
        for line in fhillierosc:
            if line.startswith('                        Oscillator strengths'):  # only allow one table
                break
            linesplitdash = line.split('-')
            row = (linesplitdash[0] + ' ' + '-'.join(linesplitdash[1:-1]) +
                   ' ' + linesplitdash[-1]).split()

            if (len(row) == 8 or (len(row) >= 10 and row[-1] == '|')) and all(map(artisatomic.isfloat, row[2:4])):
                try:
                    lambda_value = float(row[4])
                except ValueError:
                    lambda_value = -1

                if len(row) == 8:
                    hilliertransitionid = int(row[7])
                else:
                    hilliertransitionid = len(transitions) + 1

                transition = hillier_transition_row(row[0], row[1],
                                                    float(row[2]),  # f
                                                    float(row[3]),  # A
                                                    lambda_value,
                                                    int(row[5]),  # i
                                                    int(row[6]),  # j
                                                    hilliertransitionid,  # hilliertransitionid
                                                    -1,  # lowerlevel
                                                    -1,  # upperlevel
                                                    -99)  # coll_str

                if True:  # or int(transition.hilliertransitionid) not in defined_transition_ids: #checking for duplicates massively slows down the code
                    #                    defined_transition_ids.append(int(transition.hilliertransitionid))
                    transitions.append(transition)
                    transition_count_of_level_name[transition.namefrom] += 1
                    transition_count_of_level_name[transition.nameto] += 1

                    if int(transition.hilliertransitionid) != len(transitions):
                        print(
                            f'{filename} WARNING: Transition id {int(transition.hilliertransitionid):d} found at entry number {len(transitions):d}')
                        sys.exit()
                else:
                    artisatomic.log_and_print(
                        flog, f'FATAL: multiply-defined Hillier transition: {transition.namefrom} {transition.nameto}')
                    sys.exit()
    artisatomic.log_and_print(flog, f'Read {len(transitions):d} transitions')

    return hillier_ionization_energy_ev, hillier_energy_levels, transitions, transition_count_of_level_name, hillier_level_ids_matching_term


# cross section types
phixs_type_labels = {
    0: 'Constant (always zero?) [constant]',
    1: 'Seaton formula fit [sigma_o, alpha, beta]',
    2: 'Hydrogenic split l (z states, n > 11) [n, l_start, l_end]',
    3: 'Hydrogenic pure n level (all l, n >= 13) [scale, n]',
    4: 'Used for CIV rates from Leobowitz (JQSRT 1972,12,299) (6 numbers)',
    5: 'Opacity project fits (from Peach, Sraph, and Seaton (1988) (5 numbers)',
    6: 'Hummer fits to the opacity cross-sections for HeI',
    7: 'Modified Seaton formula fit (cross section zero until offset edge)',
    8: 'Modified hydrogenic split l (cross-section zero until offset edge) [n,l_start,l_end,nu_o]',
    9: 'Verner & Yakolev 1995 ground state fits (multiple shells)',
    20: 'Opacity Project: smoothed [number of data points]',
    21: 'Opacity Project: scaled, smoothed [number of data poits]',
    22: 'energy is in units of threshold, cross section in Megabarns? [number of data points]',
}


def read_phixs_tables(atomic_number, ion_stage, energy_levels, args, flog):
    # this gets partially overwritten anyway
    photoionization_crosssections = np.zeros((len(energy_levels), args.nphixspoints))
    photoionization_targetconfig_fractions = [[] for _ in energy_levels]
    # return np.zeros((len(energy_levels), args.nphixspoints)),
    # photoionization_targetfractions  # TODO: replace with real data

    n_eff = ion_stage - 1  # effective nuclear charge (with be replaced with value in file if available)
    photfilenames = ions_data[(atomic_number, ion_stage)].photfilenames
    phixstables = [{} for _ in photfilenames]
    phixstargets = ['' for _ in photfilenames]
    reduced_phixs_dict = {}
    phixs_targetconfigfactors_of_levelname = defaultdict(list)

    j_splitting_on = False  #  hopefully this is either on or off for all photoion files asssociated with a given ion

    phixs_type_levels = defaultdict(list)
    unknown_phixs_types = []
    for filenum, photfilename in enumerate(photfilenames):
        if photfilename == '':
            continue
        filename = os.path.join(hillier_ion_folder(atomic_number, ion_stage),
                                ions_data[(atomic_number, ion_stage)].folder,
                                photfilename)
        artisatomic.log_and_print(flog, 'Reading ' + filename)
        with open(filename, 'r') as fhillierphot:
            lowerlevelid = -1
            lowerlevelname = ''
            # upperlevelname = ''
            numpointsexpected = 0
            crosssectiontype = -1
            fitcoefficients = []

            for line in fhillierphot:
                row = line.split()

                if len(row) >= 2 and ' '.join(row[-4:]) == '!Final state in ion':
                    # this is not used because the upper ion's levels are not known at this time
                    targetlevelname = row[0]
                    artisatomic.log_and_print(flog, 'Photoionisation target: ' + targetlevelname)
                    if '[' in targetlevelname:
                        print('STOP! target level contains a bracket (is J-split?)')
                        sys.exit()
                    if targetlevelname in phixstargets:
                        print("STOP! Multiple phixs files for the same target configuration")
                        sys.exit()
                    phixstargets[filenum] = targetlevelname

                if len(row) >= 2 and ' '.join(row[-3:]) == '!Split J levels':
                    if row[0].lower() == 'true':
                        j_splitting_on = True
                        artisatomic.log_and_print(flog, 'File specifies J-splitting enabled')
                    elif row[0].lower() == 'false':
                        if j_splitting_on:
                            print('STOP! J-splitting disabled here, but was previously enabled for this ion')
                            sys.exit()
                        j_splitting_on = False
                    else:
                        print(f'STOP! J-splitting not true or false: "{row[0]}"')
                        sys.exit()

                if len(row) >= 2 and ' '.join(row[-2:]) == '!Configuration name' or ' '.join(row[-3:]) == '!Configuration name [*]':
                    lowerlevelname = row[0]
                    if '[' in lowerlevelname:
                        lowerlevelname.split('[')[0]
                    fitcoefficients = []
                    numpointsexpected = 0
                    lowerlevelid = 1
                    for levelid, energy_level in enumerate(energy_levels[1:], 1):
                        this_levelnamenoj = energy_level.levelname.split('[')[0]
                        if this_levelnamenoj == lowerlevelname:
                            lowerlevelid = levelid
                            break
                    if targetlevelname == '':
                        print("ERROR: no upper level name")
                        sys.exit()
                    # print(f"Reading level {lowerlevelid} '{lowerlevelname}'")

                if len(row) >= 2 and ' '.join(row[-3:]) == '!Screened nuclear charge':
                    # 'Screened nuclear charge' appears mislabelled in the CMFGEN database
                    # it is really an ionisation stage
                    n_eff = int(float(row[0])) - 1

                if len(row) >= 2 and ' '.join(row[1:]) == '!Number of cross-section points':
                    numpointsexpected = int(row[0])
                    pointnumber = 0

                if len(row) >= 2 and ' '.join(row[1:]) == '!Cross-section unit' and row[0] != 'Megabarns':
                    print(f'Wrong cross-section unit: {row[0]}')
                    sys.exit()

                row_is_all_floats = all(map(artisatomic.isfloat, row))
                if crosssectiontype == 0:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace('D', 'E')))

                        if fitcoefficients[-1] != 0.0:
                            print("ERROR: Cross section type 0 has non-zero number after it")
                            sys.exit()

                        # phixstables[filenum][lowerlevelname] = np.zeros((numpointsexpected, 2))

                elif crosssectiontype == 1:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace('D', 'E')))
                        if len(fitcoefficients) == 3:
                            lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                            phixstables[filenum][lowerlevelname] = get_seaton_phixstable(
                                lambda_angstrom, *fitcoefficients)
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])
                            # artisatomic.log_and_print(flog, 'Using Seaton formula values for level {0}'.format(lowerlevelname))

                elif crosssectiontype == 2:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(int(float(row[0])))
                        if len(fitcoefficients) == 3:
                            n, l_start, l_end = fitcoefficients
                            if l_end > n - 1:
                                artisatomic.log_and_print(
                                    flog, "ERROR: can't have l_end = {0} > n - 1 = {1}".format(l_end, n - 1))
                            else:
                                lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                                phixstables[filenum][lowerlevelname] = get_hydrogenic_nl_phixstable(
                                    lambda_angstrom, n, l_start, l_end)
                                numpointsexpected = len(phixstables[filenum][lowerlevelname])
                            # artisatomic.log_and_print(flog, 'Using Hydrogenic split l formula values for level {0}'.format(lowerlevelname))

                elif crosssectiontype == 3:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0]))
                        if len(fitcoefficients) == 2:
                            scale, n = fitcoefficients
                            n = int(n)
                            lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                            phixstables[filenum][lowerlevelname] = scale * \
                                get_hydrogenic_n_phixstable(lambda_angstrom, n)

                            numpointsexpected = len(phixstables[filenum][lowerlevelname])
                            # artisatomic.log_and_print(flog, 'Using Hydrogenic pure n formula values for level {0}'.format(lowerlevelname))
                            # print(lowerlevelname)
                            # print(phixstables[filenum][lowerlevelname][:10])
                            # print(get_hydrogenic_nl_phixstable(lambda_angstrom, n, 0, n - 1)[:10])

                elif crosssectiontype == 5:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace('D', 'E')))
                        if len(fitcoefficients) == 5:
                            lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                            phixstables[filenum][lowerlevelname] = get_opproject_phixstable(
                                lambda_angstrom, *fitcoefficients)
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])

                            # artisatomic.log_and_print(flog, 'Using OP project formula values for level {0}'.format(lowerlevelname))

                elif crosssectiontype == 6:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace('D', 'E')))
                        if len(fitcoefficients) == 8:
                            lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                            phixstables[filenum][lowerlevelname] = get_hummer_phixstable(
                                lambda_angstrom, *fitcoefficients)
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])

                            # print(lowerlevelname, "HUMMER")
                            # print(fitcoefficients)
                            # print(phixstables[lowerlevelname][::5])
                            # print(phixstables[lowerlevelname][-10:])

                            # artisatomic.log_and_print(flog, 'Using Hummer formula values for level {0}'.format(lowerlevelname))

                elif crosssectiontype == 7:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(float(row[0].replace('D', 'E')))
                        if len(fitcoefficients) == 4:
                            lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                            phixstables[filenum][lowerlevelname] = get_seaton_phixstable(
                                lambda_angstrom, *fitcoefficients)
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])
                            # log_and_print(flog, 'Using modified Seaton formula values for level {0}'.format(lowerlevelname))

                elif crosssectiontype == 8:
                    if len(row) == 1 and row_is_all_floats and numpointsexpected > 0:
                        fitcoefficients.append(int(float(row[0].replace('D', 'E'))))
                        if len(fitcoefficients) == 4:
                            n, l_start, l_end, nu_o = fitcoefficients
                            if l_end > n - 1:
                                artisatomic.log_and_print(flog, f"ERROR: can't have l_end = {l_end} > n - 1 = {n - 1}")
                            else:
                                lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                                phixstables[filenum][lowerlevelname] = get_hydrogenic_nl_phixstable(lambda_angstrom, n,
                                                                                                    l_start, l_end, nu_o=nu_o)
                                # log_and_print(flog, 'Using offset Hydrogenic split l formula values for level {0}'.format(lowerlevelname))
                                numpointsexpected = len(phixstables[filenum][lowerlevelname])

                elif crosssectiontype == 9:
                    if len(row) == 8 and numpointsexpected > 0:
                        fitcoefficients.append(vy95_phixsfitrow(int(row[0]), int(
                            row[1]), *[float(x.replace('D', 'E')) for x in row[2:]]))

                        if len(fitcoefficients) * 8 == numpointsexpected:
                            lambda_angstrom = abs(float(energy_levels[lowerlevelid].lambdaangstrom))
                            phixstables[filenum][lowerlevelname] = get_vy95_phixstable(lambda_angstrom, fitcoefficients)
                            numpointsexpected = len(phixstables[filenum][lowerlevelname])
                            # artisatomic.log_and_print(flog, 'Using Verner & Yakolev 1995 formula values for level {0}'.format(lowerlevelname))

                elif crosssectiontype in [20, 21, 22]:  # sampled data points
                    if len(row) == 2 and row_is_all_floats and lowerlevelname != '':
                        if lowerlevelname not in phixstables[filenum]:
                            phixstables[filenum][lowerlevelname] = np.zeros((numpointsexpected, 2))
                        xspoint = float(row[0].replace('D', 'E')), float(row[1].replace('D', 'E'))
                        phixstables[filenum][lowerlevelname][pointnumber] = xspoint

                        if pointnumber > 0:
                            curenergy = phixstables[filenum][lowerlevelname][pointnumber][0]
                            prevenergy = phixstables[filenum][lowerlevelname][pointnumber - 1][0]
                            if curenergy == prevenergy:
                                print(
                                    f'WARNING: photoionization table for {lowerlevelname} first column duplicated '
                                    f'energy value of {prevenergy}')
                            elif curenergy < prevenergy:
                                print(f'ERROR: photoionization table for {lowerlevelname} first column decreases '
                                      f'with energy {prevenergy} followed by {curenergy}')
                                print(phixstables[filenum][lowerlevelname])
                                sys.exit()

                        pointnumber += 1

                elif crosssectiontype != -1:
                    if crosssectiontype not in unknown_phixs_types:
                        unknown_phixs_types.append(crosssectiontype)
                    fitcoefficients = []
                    lowerlevelname = ''
                    numpointsexpected = 0

                if len(row) >= 2 and ' '.join(row[1:]) == '!Type of cross-section':
                    crosssectiontype = int(row[0])
                    if lowerlevelname not in phixs_type_levels[crosssectiontype]:
                        phixs_type_levels[crosssectiontype].append(lowerlevelname)

                if len(row) == 0:
                    if (lowerlevelname != '' and lowerlevelname in phixstables and
                            targetlevelname in phixstables[lowerlevelname] and
                            numpointsexpected != len(phixstables[filenum][lowerlevelname])):
                        print(
                            f'photoionization_crosssections mismatch: expecting {numpointsexpected:d} rows but found {len(phixstables[filenum][lowerlevelname]):d}')
                        print(
                            f'A={atomic_number}, ion_stage={ion_stage}, lowerlevel={lowerlevelname}, crosssectiontype={crosssectiontype}')
                        sys.exit()
                    lowerlevelname = ''
                    crosssectiontype = -1
                    numpointsexpected = 0

        for crosssectiontype in sorted(phixs_type_levels.keys()):
            if crosssectiontype in unknown_phixs_types:
                artisatomic.log_and_print(
                    flog, f'WARNING {len(phixs_type_levels[crosssectiontype])} levels with UNKOWN cross-section type {crosssectiontype}: {phixs_type_labels[crosssectiontype]}')
            else:
                artisatomic.log_and_print(
                    flog, f'{len(phixs_type_levels[crosssectiontype])} levels with cross-section type {crosssectiontype}: {phixs_type_labels[crosssectiontype]}')

        reduced_phixstables_onetarget = artisatomic.reduce_phixs_tables(phixstables[filenum], args)

        for lowerlevelname, reduced_phixstable in reduced_phixstables_onetarget.items():
            phixs_threshold = reduced_phixstable[0]
            phixs_targetconfigfactors_of_levelname[lowerlevelname].append(
                (phixstargets[filenum], phixs_threshold))

            # add the new phixs table, or replace the
            # existing one if this target has a larger threshold cross section
            if lowerlevelname not in reduced_phixs_dict or \
                    phixs_threshold > reduced_phixs_dict[lowerlevelname][0]:
                reduced_phixs_dict[lowerlevelname] = reduced_phixstables_onetarget[lowerlevelname]

    # normalise the target factors and scale the phixs table
    phixs_targetconfigfractions_of_levelname = defaultdict(list)
    for lowerlevelname, reduced_phixstable in reduced_phixs_dict.items():
        target_configfactors_nofilter = phixs_targetconfigfactors_of_levelname[lowerlevelname]
        # the factors are arbitary and need to be normalised into fractions

        # filter out low fraction targets
        factor_sum_nofilter = sum([x[1] for x in target_configfactors_nofilter])
        if factor_sum_nofilter > 0.:  # else, it's probably all zeros, so leave it and "send" it to the ground state
            target_configfactors = [x for x in target_configfactors_nofilter if (x[1] / factor_sum_nofilter > 0.01)]
            if len(target_configfactors) == 0:
                print("ERRORHERE", lowerlevelname, target_configfactors_nofilter)
                print(reduced_phixstable)
            max_factor = max([x[1] for x in target_configfactors])
            factor_sum = sum([x[1] for x in target_configfactors])

            for target_config, target_factor in target_configfactors:
                target_fraction = target_factor / factor_sum
                phixs_targetconfigfractions_of_levelname[lowerlevelname].append((target_config, target_fraction))

            # e.g. if the target with the highest fraction has 50%, the cross sections need to be multiplied by two
            reduced_phixs_dict[lowerlevelname] = reduced_phixstable / (max_factor / factor_sum)

    # now the non-J-split cross sections are mapped onto J-split levels
    for lowerlevelname_a, phixstable in reduced_phixs_dict.items():
        for levelid, energy_level in enumerate(energy_levels[1:], 1):
            if j_splitting_on:
                levelname_b = energy_level.levelname
            else:
                levelname_b = energy_level.levelname.split('[')[0]

            if levelname_b == lowerlevelname_a:  # due to J splitting, we may match multiple levels here
                photoionization_crosssections[levelid] = phixstable
                photoionization_targetconfig_fractions[levelid] = phixs_targetconfigfractions_of_levelname[levelname_b]

    return photoionization_crosssections, photoionization_targetconfig_fractions


@contract
def get_seaton_phixstable(lambda_angstrom: 'float,>0', sigmat, beta, s, nu_o=None) -> 'array':
    energygrid = np.arange(0, 1.0, 0.001)
    phixstable = np.empty((len(energygrid), 2))

    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    for index, c in enumerate(energygrid):
        energydivthreshold = 1 + 20 * (c ** 2)

        if nu_o is None:
            thresholddivenergy = energydivthreshold ** -1
            crosssection = sigmat * (beta + (1 - beta) * (thresholddivenergy)) * (thresholddivenergy ** s)
        else:
            thresholdenergyev = hc_in_ev_angstrom / lambda_angstrom
            energyoffsetdivthreshold = energydivthreshold + (nu_o * 1e15 * h_in_ev_seconds) / thresholdenergyev
            thresholddivenergyoffset = energyoffsetdivthreshold ** -1
            if thresholddivenergyoffset < 1.0:
                crosssection = sigmat * (beta + (1 - beta) * (thresholddivenergyoffset)) * \
                    (thresholddivenergyoffset ** s)
            else:
                crosssection = 0.

        phixstable[index] = energydivthreshold * thresholdenergyryd, crosssection
    return phixstable


# test: for n = 5, l_start = 4, l_end = 4 (2s2_5g_2Ge level of C II)
# 2.18 eV threshold cross section is near 4.37072813 Mb, great!
def get_hydrogenic_nl_phixstable(lambda_angstrom, n, l_start, l_end, nu_o=None):
    assert l_start >= 0
    assert l_end <= n - 1
    energygrid = hyd_phixs_energygrid_ryd[(n, l_start)]
    phixstable = np.empty((len(energygrid), 2))

    thresholdenergyev = hc_in_ev_angstrom / lambda_angstrom
    thresholdenergyryd = thresholdenergyev / ryd_to_ev

    scale_factor = 1 / thresholdenergyryd / (n ** 2) / ((l_end - l_start + 1) * (l_end + l_start + 1))
    # scale_factor = 1.0

    for index, energy_ryd in enumerate(energygrid):
        energydivthreshold = energy_ryd / energygrid[0]
        if nu_o is None:
            U = energydivthreshold
        else:
            E_o = (nu_o * 1e15 * h_in_ev_seconds)
            U = thresholdenergyev * energydivthreshold / (E_o + thresholdenergyev)  # energy / (E_0 + E_threshold)
        if U > 0:
            crosssection = 0.
            for l in range(l_start, l_end + 1):
                if not np.array_equal(hyd_phixs_energygrid_ryd[(n, l)], energygrid):
                    print("TABLE MISMATCH")
                    sys.exit()
                crosssection += (2 * l + 1) * hyd_phixs[(n, l)][index]
            crosssection = crosssection * scale_factor
        else:
            crosssection = 0.
        phixstable[index][0] = energydivthreshold * thresholdenergyryd  # / ryd_to_ev
        phixstable[index][1] = crosssection

    return phixstable


# test: hydrogen n = 1: 13.606 eV threshold cross section is near 6.3029 Mb
# test: hydrogen n = 5: 2.72 eV threshold cross section is near 37.0 Mb?? can't find a source for this
# give the same results as get_hydrogenic_nl_phixstable(lambda_angstrom, n, 0, n - 1)
def get_hydrogenic_n_phixstable(lambda_angstrom, n):
    energygrid = hyd_gaunt_energygrid_ryd[n]
    phixstable = np.empty((len(energygrid), 2))

    thresholdenergyev = hc_in_ev_angstrom / lambda_angstrom
    thresholdenergyryd = thresholdenergyev / ryd_to_ev

    scale_factor = 7.91 / thresholdenergyryd / n

    for index, energy_ryd in enumerate(energygrid):
        energydivthreshold = energy_ryd / energygrid[0]

        if energydivthreshold > 0:
            crosssection = scale_factor * hyd_gaunt_factor[n][index] / (energydivthreshold) ** 3
        else:
            crosssection = 0.

        phixstable[index][0] = energydivthreshold * thresholdenergyryd  # / ryd_to_ev
        phixstable[index][1] = crosssection

    return phixstable


# Peach, Sraph, and Seaton (1988)
def get_opproject_phixstable(lambda_angstrom, a, b, c, d, e):
    energygrid = np.arange(0, 1.0, 0.001)
    phixstable = np.empty((len(energygrid), 2))

    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    for index, c in enumerate(energygrid):
        energydivthreshold = 1 + 20 * (c ** 2)
        u = energydivthreshold

        x = math.log10(min(u, e))

        crosssection = 10 ** (a + x * (b + x * (c + x * d)))
        if u > e:
            crosssection *= (e / u) ** 2

        phixstable[index] = energydivthreshold * thresholdenergyryd, crosssection

    return phixstable


# only applies to helium
# the threshold cross sections seems ok, but energy dependence could be slightly wrong
# what is the h parameter that is not used??
def get_hummer_phixstable(lambda_angstrom, a, b, c, d, e, f, g, h):
    energygrid = np.arange(0, 1.0, 0.001)
    phixstable = np.empty((len(energygrid), 2))

    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    for index, c in enumerate(energygrid):
        energydivthreshold = 1 + 20 * (c ** 2)

        thresholddivenergy = energydivthreshold ** -1
        x = math.log10(energydivthreshold)

        if x < e:
            crosssection = 10 ** (((d * x + c) * x + b) * x + a)
        else:
            crosssection = 10 ** (f + g * x)

        phixstable[index] = energydivthreshold * thresholdenergyryd, crosssection

    return phixstable


def get_vy95_phixstable(lambda_angstrom, fitcoefficients):
    energygrid = np.arange(0, 1.0, 0.001)
    phixstable = np.empty((len(energygrid), 2))
    thresholdenergyryd = hc_in_ev_angstrom / lambda_angstrom / ryd_to_ev

    for index, c in enumerate(energygrid):
        energydivthreshold = 1 + 20 * (c ** 2)

        crosssection = 0.
        for params in fitcoefficients:
            y = energydivthreshold * params.E_th_eV / params.E_0  # E / E_0
            P = params.P
            Q = 5.5 + params.l - 0.5 * params.P
            y_a = params.y_a
            y_w = params.y_w
            crosssection += params.sigma_0 * ((y - 1) ** 2 + y_w ** 2) * (y ** -Q) * ((1 + math.sqrt(y / y_a)) ** -P)

        phixstable[index] = energydivthreshold * thresholdenergyryd, crosssection
    return phixstable


def read_coldata(atomic_number, ion_stage, energy_levels, flog, args):
    t_scale_factor = 1e4  # Hiller temperatures are given as T_4
    upsilondict = {}
    coldatafilename = ions_data[(atomic_number, ion_stage)].coldatafilename
    if coldatafilename == '':
        artisatomic.log_and_print(flog, 'No collisional data file specified')
        return upsilondict

    discarded_transitions = 0

    level_id_of_level_name = {}
    for levelid in range(1, len(energy_levels)):
        if hasattr(energy_levels[levelid], 'levelname'):
            level_id_of_level_name[energy_levels[levelid].levelname] = levelid

    filename = os.path.join(hillier_ion_folder(atomic_number, ion_stage),
                            ions_data[(atomic_number, ion_stage)].folder,
                            coldatafilename)
    artisatomic.log_and_print(flog, 'Reading ' + filename)
    with open(filename, 'r') as fcoldata:
        header_row = []
        temperature_index = -1
        for line in fcoldata:
            row = line.split()
            if len(line.strip()) == 0:
                continue  #  skip blank lines

            if (line.startswith("dln_OMEGA_dlnT = T/OMEGA* dOMEGAdt for HE2") or  # found in he2col.dat
                    line.startswith("Johnson values")):  # found in col_ariii
                break

            if line.lstrip().startswith('Transition\T'):  # found the header row
                header_row = row
                if len(header_row) != num_expected_t_values + 1:
                    artisatomic.log_and_print(
                        flog, f'WARNING: Expected {num_expected_t_values:d} temperature values, but header has {len(header_row):d} columns')
                    num_expected_t_values = len(header_row) - 1
                    artisatomic.log_and_print(
                        flog, f'Assuming header is incorrect and setting num_expected_t_values={num_expected_t_values:d}')

                temperatures = row[-num_expected_t_values:]
                artisatomic.log_and_print(
                    flog, f'Temperatures available for effective collision strengths (units of {t_scale_factor:.1e} K):\n{", ".join(temperatures)}')
                match_sorted_temperatures = sorted(temperatures,
                                                   key=lambda t: abs(float(t.replace('D', 'E')) * t_scale_factor - args.electrontemperature))
                best_temperature = match_sorted_temperatures[0]
                temperature_index = temperatures.index(best_temperature)
                artisatomic.log_and_print(
                    flog, f"Selecting {float(temperatures[temperature_index].replace('D', 'E')) * t_scale_factor:.3f} K")
                continue

            if len(row) >= 2:
                row_two_to_end = ' '.join(row[1:])

                if row_two_to_end == '!Number of transitions':
                    number_expected_transitions = int(row[0])
                elif row_two_to_end.startswith('!Number of T values OMEGA tabulated at'):
                    num_expected_t_values = int(row[0])
                elif row_two_to_end == '!Scaling factor for OMEGA (non-file values)' and float(row[0]) != 1.0:
                    artisatomic.log_and_print(flog, 'ERROR: non-zero scaling factor for OMEGA. what does this mean?')
                    sys.exit()

            if header_row != []:
                namefromnameto = "".join(row[:-num_expected_t_values])
                upsilonvalues = row[-num_expected_t_values:]

                namefrom, nameto = map(str.strip, namefromnameto.split('-'))
                upsilon = float(upsilonvalues[temperature_index].replace('D', 'E'))
                try:
                    id_lower = level_id_of_level_name[namefrom]
                    id_upper = level_id_of_level_name[nameto]
                    if id_lower > id_upper:
                        artisatomic.log_and_print(
                            flog, f'WARNING: Transition ids are backwards {namefrom} (level {id_lower:d}) -> {nameto} (level {id_upper:d})...swapping levels')
                        id_lower, namefrom, id_upper, nameto = id_upper, nameto, id_lower, namefrom

                    if id_lower == id_upper:
                        artisatomic.log_and_print(
                            flog, f'WARNING: Transition ids are equal? {namefrom} (level {id_lower:d}) -> {nameto} (level {id_upper:d})...discarding')
                        discarded_transitions += 1
                    elif (id_lower, id_upper) in upsilondict:
                        print(f'ERROR: Duplicate transition from {namefrom} -> {nameto}')
                        sys.exit()
                    else:
                        upsilondict[(id_lower, id_upper)] = upsilon
                    # print(namefrom, nameto, upsilon)
                except KeyError:
                    unlisted_from_message = ' (unlisted)' if namefrom not in level_id_of_level_name else ''
                    unlisted_to_message = ' (unlisted)' if nameto not in level_id_of_level_name else ''
                    artisatomic.log_and_print(
                        flog, f'Discarding upsilon={upsilon:.3f} for {namefrom}{unlisted_from_message} -> {nameto}{unlisted_to_message}')
                    discarded_transitions += 1

    if len(upsilondict) + discarded_transitions < number_expected_transitions:
        print(
            f'ERROR: file specified {number_expected_transitions:d} transitions, but only {len(upsilondict) + discarded_transitions:d} were found')
        sys.exit()
    elif len(upsilondict) + discarded_transitions > number_expected_transitions:
        artisatomic.log_and_print(
            flog, f'WARNING: file specified {number_expected_transitions:d} transitions, but {len(upsilondict) + discarded_transitions:d} were found')
    else:
        artisatomic.log_and_print(
            flog, f'Read {len(upsilondict) + discarded_transitions} effective collision strengths ')

    return upsilondict


def get_photoiontargetfractions(energy_levels, energy_levels_upperion, hillier_photoion_targetconfigs, flog):
    targetlist = [[] for _ in energy_levels]
    targetlist_of_targetconfig = defaultdict(list)

    for lowerlevelid, energy_level in enumerate(energy_levels[1:], 1):
        for targetconfig, targetconfig_fraction in hillier_photoion_targetconfigs[lowerlevelid]:
            if targetconfig not in targetlist_of_targetconfig:
                # sometimes the target has a slash, e.g. '3d7_4Fe/3d7_a4Fe'
                # so split on the slash and match all parts
                targetconfiglist = targetconfig.split('/')
                upperionlevelids = []
                for upperlevelid, upper_energy_level in enumerate(energy_levels_upperion[1:], 1):
                    upperlevelnamenoj = upper_energy_level.levelname.split('[')[0]
                    if upperlevelnamenoj in targetconfiglist:
                        upperionlevelids.append(upperlevelid)
                if not upperionlevelids:
                    upperionlevelids = [1]
                targetlist_of_targetconfig[targetconfig] = []

                summed_statistical_weights = sum([float(energy_levels_upperion[index].g) for index in upperionlevelids])
                for upperionlevelid in sorted(upperionlevelids):
                    statweight_fraction = energy_levels_upperion[upperionlevelid].g / summed_statistical_weights
                    targetlist_of_targetconfig[targetconfig].append((upperionlevelid, statweight_fraction))

            for upperlevelid, statweight_fraction in targetlist_of_targetconfig[targetconfig]:
                targetlist[lowerlevelid].append((upperlevelid, targetconfig_fraction * statweight_fraction))

        if len(targetlist[lowerlevelid]) == 0:
            targetlist[lowerlevelid].append((1, 1.0))

    return targetlist


def read_hyd_phixsdata():
    hillier_ionization_energy_ev, hillier_energy_levels, transitions, transition_count_of_level_name, hillier_level_ids_matching_term = read_levels_and_transitions(
        1, 1, open('/dev/null', 'w'))

    hyd_filename = 'atomic-data-hillier/atomic/HYD/I/5dec96/hyd_l_data.dat'
    print(f'Reading hydrogen photoionization cross sections from {hyd_filename}')
    max_n = -1
    l_start_u = 0.
    with open(hyd_filename, 'r') as fhyd:
        for line in fhyd:
            row = line.split()
            if ' '.join(row[1:]) == '!Maximum principal quantum number':
                max_n = int(row[0])

            if ' '.join(row[1:]) == '!L_ST_U':
                l_start_u = float(row[0].replace('D', 'E'))

            if ' '.join(row[1:]) == '!L_DEL_U':
                l_del_u = float(row[0].replace('D', 'E'))

            if max_n >= 0 and line.strip() == '':
                break

        for line in fhyd:
            if line.strip() == '':
                continue

            n, l, num_points = [int(x) for x in line.split()]
            e_threshold_ev = hc_in_ev_angstrom / float(hillier_energy_levels[n].lambdaangstrom)

            xs_values = []
            for line in fhyd:
                values_thisline = [float(x) for x in line.split()]
                xs_values = xs_values + values_thisline
                if len(xs_values) == num_points:
                    break
                elif len(xs_values) > num_points:
                    print(
                        f'ERROR: too many datapoints for (n,l)=({n},{l}), expected {num_points} but found {len(xs_values)}')
                    sys.exit()

            hyd_phixs_energygrid_ryd[(n, l)] = [e_threshold_ev / ryd_to_ev * 10 **
                                                (l_start_u + l_del_u * index) for index in range(num_points)]
            hyd_phixs[(n, l)] = [10 ** (8 + logxs) for logxs in xs_values]  # cross sections in Megabarns
            #hyd_phixs_f = interpolate.interp1d(hyd_energydivthreholdgrid[(n, l)], hyd_phixs[(n, l)], kind='linear', assume_sorted=True)

    hyd_filename = 'atomic-data-hillier/atomic/HYD/I/5dec96/gbf_n_data.dat'
    print(f'Reading hydrogen Gaunt factors from {hyd_filename}')
    max_n = -1
    l_start_u = 0.
    with open(hyd_filename, 'r') as fhyd:
        for line in fhyd:
            row = line.split()
            if ' '.join(row[1:]) == '!Maximum principal quantum number':
                max_n = int(row[0])

            if len(row) > 1:
                if row[1] == '!N_ST_U':
                    n_start_u = float(row[0].replace('D', 'E'))
                elif row[1] == '!N_DEL_U':
                    n_del_u = float(row[0].replace('D', 'E'))

            if max_n >= 0 and line.strip() == '':
                break

        for line in fhyd:
            if line.strip() == '':
                continue

            n, num_points = [int(x) for x in line.split()]
            e_threshold_ev = hc_in_ev_angstrom / float(hillier_energy_levels[n].lambdaangstrom)

            gaunt_values = []
            for line in fhyd:
                values_thisline = [float(x) for x in line.split()]
                gaunt_values = gaunt_values + values_thisline
                if len(gaunt_values) == num_points:
                    break
                elif len(gaunt_values) > num_points:
                    print(f'ERROR: too many datapoints for n={n}, expected {num_points} but found {len(gaunt_values)}')
                    sys.exit()

            hyd_gaunt_energygrid_ryd[n] = [e_threshold_ev / ryd_to_ev * 10 **
                                           (n_start_u + n_del_u * index) for index in range(num_points)]
            hyd_gaunt_factor[n] = gaunt_values  # cross sections in Megabarns


def extend_ion_list(listelements):
    for (atomic_number, ion_stage) in ions_data.keys():
        if atomic_number == 1:
            continue  # skip hydrogen
        found_element = False
        for (tmp_atomic_number, list_ions) in listelements:
            if tmp_atomic_number == atomic_number:
                if ion_stage not in list_ions:
                    list_ions.append(ion_stage)
                    list_ions.sort()
                found_element = True
        if not found_element:
            listelements.append((atomic_number, [ion_stage],))
    listelements.sort(key=lambda x: x[0])
    return listelements

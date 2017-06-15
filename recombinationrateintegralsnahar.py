#!/usr/bin/env python3
# import itertools
import math
# import os
import sys
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
# import numexpr as ne
import numpy as np
# import pandas as pd
# from astropy import constants as const
# from astropy import units as u
from scipy import integrate
# from scipy import interpolate

import makeartisatomicfiles as artisatomic

# PYDIR = os.path.dirname(os.path.abspath(__file__))
# atomicdata = pd.read_csv(os.path.join(PYDIR, 'atomic_properties.txt'), delim_whitespace=True, comment='#')
# elsymbols = ['n'] + list(atomicdata['symbol'].values)

corestatetuple = namedtuple('corestate', 'twosplusone lval parity energyryd config')
nahar_recomb_level = namedtuple('naharrecomblevel', 'strlevelid twosplusone lval parity indexinsymmetry energyryd zz icx')

number_of_multiplicity = {
    'singlet': 1,
    'doublet': 2,
    'triplet': 3,
    'quartet': 4,
    'quintet': 5,
    'sextet': 6,
    'septet': 7,
}

CLIGHT = 2.99792458e10  # Speed of light.
CLIGHT2 = 8.9875518e20  # Speed of light squared.
H = 6.6260755e-27  # Planck constant in erg seconds
RYD = 2.1798741e-11  # Rydberg to ergs
KB = 1.38065e-16  # Boltzmann constant
SAHACONST = 2.0706659e-16  # Saha constant
MH = 1.67352e-24      # Mass of hydrogen atom [g]
ME = 9.1093897e-28    # Mass of free electron [g]

CLIGHTSQUARED = 8.9875518e20
TWOOVERCLIGHTSQUARED = 2.2253001e-21
TWOHOVERCLIGHTSQUARED = 1.4745007e-47
CLIGHTSQUAREDOVERTWOH = 6.7819570e46
HOVERKB = 4.799243681748932e-11
FOURPI = 1.256637061600000e+01
ONEOVER4PI = 7.957747153555701e-02
HCLIGHTOVERFOURPI = 1.580764662876770e-17
OSCSTRENGTHCONVERSION = 1.3473837e+21


def levelmatch(levela, levelb):
    return (
        levela.twosplusone == levelb.twosplusone and
        levela.lval == levelb.lval and
        levela.indexinsymmetry == levelb.indexinsymmetry)


def reduce_and_reconstruct_phixs_tables(dicttables, optimaltemperature, nphixspoints, phixsnuincrement):
    print('Reducing phixs tables')
    dictout = {}

    xgrid = np.linspace(1.0, 1.0 + phixsnuincrement * (nphixspoints + 1),
                        num=nphixspoints + 1, endpoint=False)

    dicttables_reduced = artisatomic.reduce_phixs_tables(
        dicttables, optimaltemperature, nphixspoints, phixsnuincrement, hideoutput=False)

    for key, phixstable_reduced in dicttables_reduced.items():
        e_threshold_ryd = dicttables[key][0][0]
        dictout[key] = np.array(list(zip(xgrid[:-1] * e_threshold_ryd, phixstable_reduced)))

    return dictout


def read_recombrate_file(atomicnumber, ionstage):
    filename = f'atomic-data-nahar/{artisatomic.elsymbols[atomicnumber].lower()}{ionstage}.rrc.ls.txt'
    print(f'Reading {filename}')
    with open(filename, 'r') as filein:
        temperatures = []
        recomblevels = []
        recombrates = {}
        symmetrymatches = defaultdict(int)
        corestates = {}
        indexinsymmetryoflevel = defaultdict(int)
        while True:
            line = filein.readline()
            row = line.split()

            if corestates and line.startswith('------------------------------------------------------------------'):
                break

            elif line.rstrip().rstrip(':').endswith('target information'):
                lower_twosplusone = number_of_multiplicity.get(row[1].lower(), -1)  # -1 means these are not split into different multiplicities
                while True:
                    line = filein.readline()
                    row = line.split()
                    if (len(row) >= 3 and row[0] == f'{atomicnumber:d}' and
                            row[1] == f'{atomicnumber - ionstage:d}' and row[2] == 'T'):
                        break
                ncorestates = int(filein.readline())
                corestates[lower_twosplusone] = []
                for _ in range(ncorestates):
                    row = filein.readline().split()
                    if len(row) == 6:
                        row = row[1:]  # truncate the core state number in some files (fe2.rrc.ls.txt)
                    corestate = corestatetuple(int(row[0]), int(row[1]), int(row[2]), float(row[3]), row[4])
                    corestates[lower_twosplusone].append(corestate)

        # recombined states and their energies
        while True:
            line = filein.readline()
            row = line.split()
            if len(row) == 4 and all(map(artisatomic.isfloat, row)):
                energyryd = float(row[1])
                strlevelid = row[2]
                statweight = int(float(row[3]))

                if len(strlevelid) > 8:
                    twosplusone = int(strlevelid[0])
                    lval = int(strlevelid[1:3])
                    parity = int(strlevelid[3])
                    zz = int(strlevelid[4:6])
                    icx = int(strlevelid[6:8])

                    if statweight != twosplusone * (lval * 2 + 1):
                        print('Error: stat weight inconsistent with (2S + 1)(2L + 1)')
                        sys.exit()

                    symmetrymatches[(twosplusone, lval, parity)] += 1
                    indexinsymmetryoflevel[strlevelid] = symmetrymatches[(twosplusone, lval, parity)]
                    recomb_level = nahar_recomb_level(strlevelid=strlevelid,
                                                      twosplusone=twosplusone, lval=lval, parity=parity,
                                                      indexinsymmetry=indexinsymmetryoflevel[strlevelid],
                                                      energyryd=energyryd, zz=zz, icx=icx)
                    recomblevels.append(recomb_level)

            if not line or line.startswith('iii) State-specific RRC'):
                break

        while True:
            line = filein.readline()
            if not line or line.startswith(' level-id         BE'):
                break

        while True:
            line = filein.readline()
            temperatures.extend([float(token.strip('(K)')) for token in line.split() if token != '(Ry)'])
            if not line or line.strip() == '':
                break

        current_level = "-1"
        while True:
            line = filein.readline()
            if not line:
                break
            elif line.strip() == '':
                continue
            row = line.split()

            if '.' in row[0] and len(row[0]) == 13:
                current_level = row[0]
                recombrates[current_level] = [float(rate) for rate in row[2:]]
            elif current_level != "-1" and len(recombrates[current_level]) < len(temperatures):
                recombrates[current_level].extend([float(rate) for rate in row])

    return np.array(temperatures), recombrates, corestates, recomblevels


def get_phixslist(atomicnumber, ionstage, partial=False):
    pt = 'pt' if partial else ''
    filename = f'atomic-data-nahar/{artisatomic.elsymbols[atomicnumber].lower()}{ionstage}.{pt}px.txt'
    print(f'Reading {filename}')
    with open(filename, 'r') as filein:
        phixslist = {}
        binding_energy_ryd = {}

        while True:
            line = filein.readline()
            if line.startswith(f'   {atomicnumber}   {atomicnumber - ionstage}    P'):
                break

        while True:
            row = filein.readline().split()
            if len(row) != 4 or row == ['0', '0', '0', '0']:
                break

            levelid = tuple(int(x) for x in row)
            row = filein.readline().split()
            strnpoints = row[1]
            npoints = int(strnpoints)
            be, _ = filein.readline().split()  # binding energy
            binding_energy_ryd[levelid] = float(be)
            phixslist[levelid] = np.zeros((npoints, 2))

            for n in range(npoints):
                row = filein.readline().split()
                if len(row) != 2:
                    print("ERROR!")
                    sys.exit()
                phixslist[levelid][n][0] = float(row[0])
                phixslist[levelid][n][1] = float(row[1])

    return phixslist, binding_energy_ryd


def plot_phixs(phixstable, ptphixstable):
    outputfile = 'phixs.pdf'
    fig, axis = plt.subplots(
        1, 1, sharex=True, figsize=(6, 6),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    # axis.annotate(f'Timestep {timestepmin:d} to {timestepmax:d} (t={time_days_min} to '
    #               f'{time_days_max})\nCell {modelgridindex:d}',
    #               xy=(0.02, 0.96), xycoords='axes fraction',
    #               horizontalalignment='left', verticalalignment='top', fontsize=8)
    import scipy.signal

    ylist = [x[1] for x in phixstable]
    ylist_smooth = scipy.signal.savgol_filter(ylist, 349, 3)
    # ylist_smooth = scipy.signal.resample(ylist, 10000)

    axis.plot([x[0] for x in phixstable], ylist_smooth, linewidth=2)

    ylist2 = [x[1] for x in ptphixstable]
    ylist2_smooth = scipy.signal.savgol_filter(ylist2, 349, 3)
    # ylist_smooth = scipy.signal.resample(ylist, 10000)

    axis.plot([x[0] for x in ptphixstable], ylist2_smooth, linewidth=1)

    # axis.set_xlabel(r'Wavelength in ($\AA$)')
    # axis.set_ylabel(r'Wavelength out ($\AA$)')
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    axis.set_xlim(xmax=4)
    axis.set_yscale('log')
    # axis.set_ylim(ymin=xmin, ymax=xmax)

    # axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 13})

    print(f'Saving to {outputfile:s}')
    fig.savefig(outputfile, format='pdf')
    plt.close()


def recomb_integral_euler(phixslist, T):
    integral = 0.0
    for i in range(0, len(phixslist) - 1):
        nu = phixslist[i][0] * RYD / H
        dnu = (phixslist[i + 1][0] - phixslist[i][0]) * RYD / H
        sigma_bf = phixslist[i][1] * 1e-18  # from Megabarn to cm^2
        dnu2 = dnu / 10.0
        # print(nu, nu + dnu, dnu2)
        for nu2 in np.arange(nu, nu + dnu, dnu2):
            # sigma_bf = np.interp(nu2, [nu, nu + dnu], [phixslist[i][1], phixslist[i + 1][1]]) * 1e-18
            sigma_bf = phixslist[i][1] * 1e-18  # nearest to the left

            integral += TWOOVERCLIGHTSQUARED * sigma_bf * (nu2 ** 2) * math.exp(-HOVERKB * nu2 / T) * dnu2
    return integral


def calculate_level_alpha(phixslist, g_lower, g_upper, T, kind="trapz"):
    E_threshold = phixslist[0][0] * RYD
    # sahafactor = float(ne.evaluate("g_lower / g_upper * SAHACONST * (T ** -1.5) * exp(E_threshold / KB / T)"))
    sahafactor = g_lower / g_upper * SAHACONST * (T ** -1.5) * math.exp(E_threshold / KB / T)
    # sahafactor2 = g_lower / g_upper * SAHACONST * math.exp(E_threshold / KB / T - 1.5 * math.log(T))

    if kind == "euler":
        integral = recomb_integral_euler(phixslist, T)
    else:
        def integrand(en_ryd, sigma_megabarns):
            nu = en_ryd * RYD / H
            sigmabf = sigma_megabarns * 1e-18
            return TWOOVERCLIGHTSQUARED * sigmabf * pow(nu, 2) * math.exp(-HOVERKB * nu / T)

        arr_nu = [en_ryd * RYD / H for en_ryd in phixslist[:, 0]]
        arr_integrand = [integrand(en_ryd, sigma_megabarns) for en_ryd, sigma_megabarns in phixslist]

        if kind == "simps":
            integral = integrate.simps(arr_integrand, arr_nu)
        elif kind == "trapz":
            integral = integrate.trapz(arr_integrand, arr_nu)
        else:
            print(f"UNKNOWN INTEGRAL KIND: {kind}")
            sys.exit()

    alpha = 4 * math.pi * sahafactor * integral

    # attempt at using Nahar form of recombination rate integral
    # def integrand_nahar(en, sigma_megabarns):
    #     epsilon = en - E_threshold
    #     sigmabf = sigma_megabarns * 1e-18
    #     return (en ** 2) * sigmabf * math.exp(- HOVERKB * epsilon / T)

    # arr_en = [en_ryd * RYD for en_ryd, _ in phixslist]
    # arr_epsilon = [en - E_threshold for en in arr_en]
    # arr_sigmamb = [sigma_megabarns for _, sigma_megabarns in phixslist]
    # arr_integrand = [integrand_nahar(en, sigma_megabarns) for en, sigma_megabarns in zip(arr_en, arr_sigmamb)]
    # integral_trapz = integrate.trapz(arr_integrand, arr_epsilon)
    #
    # factor = g_lower / g_upper * 2 / (KB * T * math.sqrt(2 * math.pi * (ME ** 3) * KB * CLIGHT2 * T))
    #
    # print(f'    Nahar integral_trapz alpha: {factor * integral_trapz:11.3e}  = '
    #       f'{factor * integral_trapz / alpha_nahar:6.3f} * Nahar')
    return alpha


def main():
    # T_goal = 6.31E+03
    T_goal = 6000
    nphixspoints = 50
    phixsnuincrement = 0.03

    atomicnumber = 26
    ionstage = 2
    do_reduced_list = True

    ionstr = artisatomic.elsymbols[atomicnumber] + ' ' + artisatomic.roman_numerals[ionstage]
    temperatures, recombrates, corestates, recomblevels = read_recombrate_file(atomicnumber, ionstage)
    T_index = (np.abs(temperatures - T_goal)).argmin()

    T = temperatures[T_index]  # Temperature in Kelvin

    print(f'T(K):               {T:11.3e}')
    print(f'nphixspoints:       {nphixspoints:11.3f}')
    print(f'phixsnuincrement:   {phixsnuincrement:11.3f}')

    phixsall, binding_energy_ryd_all = get_phixslist(atomicnumber, ionstage, partial=True)

    # for testing, select a level
    # recomblevels = [x for x in recomblevels if x.twosplusone == 4 and x.lval == 3 and x.parity == 0 and x.indexinsymmetry == 1]
    # phixsall = {k: v for k, v in phixsall.items() if k == (4, 3, 0, 1)}

    if do_reduced_list:
        phixslist_reduced = reduce_and_reconstruct_phixs_tables(phixsall, 3000, nphixspoints, phixsnuincrement)

    calculated_alpha_sum = defaultdict(float)

    # list the levels defined in the phix file missing from the rrc file
    # extrarecomblevels = []
    # for lowerlevel in phixsall:
    #     recomblevelfound = False
    #     for recomblevel in recomblevels:
    #         if levelmatch(recomblevel, lowerlevel):
    #             recomblevelfound = True
    #             break
    #     if not recomblevelfound:
    #         print(lowerlevel)
    #         recomb_level = nahar_recomb_level(strlevelid="PX LEVEL",
    #                                           twosplusone=lowerlevel.twosplusone, lval=lowerlevel.lval,
    #                                           parity=lowerlevel.parity,
    #                                           indexinsymmetry=lowerlevel.indexinsymmetry,
    #                                           energyryd=-99., zz=-1, icx=1)
    #         extrarecomblevels.append(recomb_level)

    # recomblevels.extend(extrarecomblevels)
    sorted_lowerlevels = list(sorted(recomblevels, key=lambda x: x.energyryd))
    for lowerlevel in sorted_lowerlevels[:]:
        # ignore unbound states
        if lowerlevel.energyryd > 0.:
            continue

        # some energy levels have no recombination rates listed in rrc file
        if lowerlevel.strlevelid not in recombrates.keys():
            continue

        if lowerlevel.twosplusone in corestates and lowerlevel.icx <= len(corestates[lowerlevel.twosplusone]):
            corestate = corestates[lowerlevel.twosplusone][lowerlevel.icx - 1]
        elif -1 in corestates and lowerlevel.icx <= len(corestates[-1]):
            corestate = corestates[-1][lowerlevel.icx - 1]
        else:
            corestate = 'NOT FOUND'

        if lowerlevel.twosplusone in corestates:
            uppergroundstate = corestates[lowerlevel.twosplusone][0]
        else:
            uppergroundstate = corestates[list(corestates.keys())[0]][0]
        g_upper = uppergroundstate.twosplusone * (uppergroundstate.lval * 2 + 1)

        term_index = (
            lowerlevel.twosplusone, lowerlevel.lval, lowerlevel.parity, lowerlevel.indexinsymmetry)

        try:
            binding_energy_ryd = binding_energy_ryd_all[term_index]
        except KeyError:
            print(f'State {lowerlevel} from rrc file is not found in px file')
            continue

        E_threshold = phixsall[term_index][0][0] * RYD

        g_lower = term_index[0] * (2 * term_index[1] + 1)
        # g_lower = 9

        termstr = (f'{term_index[0]}{artisatomic.lchars[term_index[1]]}' +
                   ("e" if term_index[2] == 0 else "o"))

        print(f'\n{ionstr} {lowerlevel.strlevelid} {term_index} {termstr}')
        print(f'  icx {lowerlevel.icx} {corestate}')
        print(f'  g_upper:              {g_upper:12.3f}')
        print(f'  g_lower:              {g_lower:12.3f}')
        print(f'  First point (Ry):     {E_threshold / RYD:12.4e}')
        print(f'  Binding energy (Ry):  {binding_energy_ryd:12.4e}')
        print(f'  Level energy (Ry):    {lowerlevel.energyryd:12.4e}')

        if lowerlevel.strlevelid in recombrates:
            alpha_nahar = recombrates[lowerlevel.strlevelid][T_index]
        else:
            alpha_nahar = -1
        print(f'\n  Nahar alpha:           {alpha_nahar:12.3e}')

        # plot_phixs(phixsall[lowerlevel], ptphixsall[lowerlevel])

        list_phixslists = [phixsall]
        if do_reduced_list:
            list_phixslists.append(phixslist_reduced)

        for reduced, phixsdict in enumerate(list_phixslists):
            print(f'\n  {len(phixsdict[term_index]):d} points in list')
            # print(phixsdict[term_index])
            for kind in ['euler', 'trapz']:
                tag = ('reduced_' if reduced == 1 else '') + kind
                alpha = calculate_level_alpha(
                    phixsdict[term_index], g_lower, g_upper, T, kind=kind)
                calculated_alpha_sum[tag] += alpha

                print(f'    integral_{tag} alpha: {alpha:11.3e} ', end='')
                if alpha_nahar > 0:
                    print(f'= {alpha / alpha_nahar:6.3f} * Nahar     '
                          f'(abs error {abs(alpha - alpha_nahar):6.1e})')
                else:
                    print()

    nahar_recomb_total = 0
    for _, recombrates_thislevel in recombrates.items():
        nahar_recomb_total += recombrates_thislevel[T_index]

    print(f'\nSummed alphas (ion Alpha low-n):')
    print(f'  Nahar:                    {nahar_recomb_total:11.3e}')
    for tag, alpha in calculated_alpha_sum.items():
        print(f'  Calculated {tag + ":":14} {alpha:11.3e} = {alpha / nahar_recomb_total:7.4f} * Nahar')


if __name__ == "__main__":
    main()

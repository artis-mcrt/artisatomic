#!/usr/bin/env python3
import itertools
import math
import multiprocessing as mp
import os
import sys
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from scipy import integrate, interpolate

import makeartisatomicfiles as artisatomic

# PYDIR = os.path.dirname(os.path.abspath(__file__))
# atomicdata = pd.read_csv(os.path.join(PYDIR, 'atomic_properties.txt'), delim_whitespace=True, comment='#')
# elsymbols = ['n'] + list(atomicdata['symbol'].values)

nahar_level = namedtuple('naharlevel', 'twosplusone lval parity indexinsymmetry')
corestatetuple = namedtuple('corestate', 'twosplusone lval parity energyryd config')
nahar_recomb_level = namedtuple('naharrecomblevel', 'levelid twosplusone lval parity indexinsymmetry energyryd zz icx')

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


def reduce_and_reconstruct_phixs_tables(dicttables, optimaltemperature, nphixspoints, phixsnuincrement):
    print('Reducing phixs tables')
    dictout = {}

    xgrid = np.linspace(1.0, 1.0 + phixsnuincrement * (nphixspoints + 1),
                        num=nphixspoints + 1, endpoint=False)
    for key, phixstable_orig in dicttables.items():
        phixstable = artisatomic.reduce_phixs_tables({'GS': phixstable_orig},
                                                     optimaltemperature, nphixspoints, phixsnuincrement,
                                                     hideoutput=True)['GS']
        dictout[key] = np.array(list(zip(xgrid[:-1] * phixstable_orig[0][0], phixstable[:])))

    return dictout


def read_recombrate_file(atomicnumber, ionstage):
    filename = f'atomic-data-nahar/{artisatomic.elsymbols[atomicnumber].lower()}{ionstage}.rrc.ls.txt'
    # print(f'Reading {filename}')
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

            if len(corestates.keys()) > 0 and line.startswith('------------------------------------------------------------------'):
                break

            elif line.rstrip().endswith('target information'):
                lower_twosplusone = number_of_multiplicity[row[1].lower()]
                while True:
                    line = filein.readline()
                    row = line.split()
                    if len(row) >= 3 and row[0] == f'{atomicnumber:d}' and row[1] == f'{atomicnumber - ionstage:d}' and row[2] == 'T':
                        break
                ncorestates = int(filein.readline())
                corestates[lower_twosplusone] = []
                for _ in range(ncorestates):
                    row = filein.readline().split()
                    corestate = corestatetuple(int(row[0]), int(row[1]), int(row[2]), float(row[3]), row[4])
                    corestates[lower_twosplusone].append(corestate)

        # recombined states and their energies
        while True:
            line = filein.readline()
            row = line.split()
            if len(row) == 4 and all(map(artisatomic.isfloat, row)):
                energyryd = float(row[1])
                strlevelid = row[2]
                statweight = float(row[3])

                twosplusone = int(strlevelid[0])
                lval = int(strlevelid[1:3])
                parity = int(strlevelid[3])
                zz = int(strlevelid[4:6])
                icx = int(strlevelid[6:8])

                symmetrymatches[(twosplusone, lval, parity)] += 1
                indexinsymmetryoflevel[strlevelid] = symmetrymatches[(twosplusone, lval, parity)]
                recomb_level = nahar_recomb_level(levelid=strlevelid,
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

    return np.array(temperatures), recombrates, corestates, recomblevels, indexinsymmetryoflevel


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

            levelid = nahar_level(*[int(x) for x in row])
            _, strnpoints = filein.readline().split()
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


def recomb_integral_euler(phixslist, T):
    integral = 0.0
    for i in range(0, len(phixslist) - 1):
        nu = phixslist[i][0] * RYD / H
        dnu = (phixslist[i + 1][0] - phixslist[i][0]) * RYD / H
        sigma_bf = phixslist[i][1] * 1e-18  # from Megabarn to cm^2
        dnu2 = dnu / 100.0
        for nu2 in np.arange(nu, nu + dnu, dnu2):
            # sigma_bf = np.interp(nu2, [nu, nu + dnu], [phixslist[i][1], phixslist[i + 1][1]]) * 1e-18
            sigma_bf = phixslist[i][1] * 1e-18  # nearest to the left

            integral += TWOOVERCLIGHTSQUARED * sigma_bf * (nu2 ** 2) * math.exp(-HOVERKB * nu2 / T) * dnu2
    return integral


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


def main():
    T_goal = 6.31E+03
    nphixspoints = 100
    phixsnuincrement = 0.1

    atomicnumber = 28
    ionstage = 2

    temperatures, recombrates, corestates, recomblevels, indexinsymmetryoflevel = read_recombrate_file(atomicnumber, ionstage)
    T_index = (np.abs(temperatures - T_goal)).argmin()

    T = temperatures[T_index]  # Temperature in Kelvin

    print(f'T(K):               {T:11.3e}')
    print(f'nphixspoints:       {nphixspoints:11.3f}')
    print(f'phixsnuincrement:   {phixsnuincrement:11.3f}')

    # count = 0
    # for transitionstr in recombrates.keys():
    #     icx = int(transitionstr[6:8])
    #     # if icx == 1:
    #     count += 1
    # print(f'Count: {count}')

    phixsall, binding_energy_ryd_all = get_phixslist(atomicnumber, ionstage, partial=True)
    phixslist_reduced = reduce_and_reconstruct_phixs_tables(phixsall, 3000, nphixspoints, phixsnuincrement)
    # phixslist_reduced = phixsall

    ion_alpha = 0.
    ion_alpha_reduced = 0.
    for recomblevel in recomblevels:

        # ignore unbound states
        if recomblevel.energyryd > 0.:
            continue

        strlevelid = recomblevel.levelid

        # some energy levels have no recombination rates listed in rrc file
        if strlevelid not in recombrates.keys():
            continue

        twosplusone = recomblevel.twosplusone
        lval = recomblevel.lval
        parity = recomblevel.parity
        icx = recomblevel.icx

        # index in symmetry??
        indexinsymmetry = indexinsymmetryoflevel[strlevelid]

        if icx <= len(corestates[twosplusone]):
            corestate = corestates[twosplusone][icx - 1]
        else:
            corestate = 'DOES NOT EXIST'

        uppergroundstate = corestates[twosplusone][0]
        g_upper = uppergroundstate.twosplusone * (uppergroundstate.lval * 2 + 1)

        lowerlevel = nahar_level(twosplusone, lval, parity, indexinsymmetry)

        try:
            binding_energy_ryd = binding_energy_ryd_all[lowerlevel]
        except KeyError:
            print(f'State {recomblevel} from rrc file is not found in px file')
            continue

        E_threshold = phixsall[lowerlevel][0][0] * RYD

        g_lower = lowerlevel.twosplusone * (2 * lowerlevel.lval + 1)
        # sahafactor = float(ne.evaluate("g_lower / g_upper * SAHACONST * (T ** -1.5) * exp(E_threshold / KB / T)"))
        sahafactor = g_lower / g_upper * SAHACONST * (T ** -1.5) * math.exp(E_threshold / KB / T)
        # sahafactor2 = g_lower / g_upper * SAHACONST * math.exp(E_threshold / KB / T - 1.5 * math.log(T))

        print(f'\n{strlevelid}')
        print(f'  icx {icx}, {corestate}')
        print(f'  g_upper:              {g_upper:12.3f}')
        print(f'  g_lower:              {g_lower:12.3f}')
        print(f'  First point (Ry):     {E_threshold / RYD:12.4e}')
        print(f'  Binding energy (Ry):  {binding_energy_ryd:12.4e}')
        print(f'  Level energy (Ry):    {recomblevel.energyryd:12.4e}')
        print(f'  Saha factor:          {sahafactor:12.3e}')

        alpha_nahar = recombrates[strlevelid][T_index]
        print(f'\n  Nahar alpha:           {alpha_nahar:12.3e}')

        # plot_phixs(phixsall[lowerlevel], ptphixsall[lowerlevel])

        for reduced, phixslist in enumerate([phixsall[lowerlevel], phixslist_reduced[lowerlevel]]):
            print(f'\n  {len(phixslist):d} points in list')

            # integral_euler = recomb_integral_euler(phixslist, T)

            def integrand(en_ryd, sigma_megabarns):
                nu = en_ryd * RYD / H
                sigmabf = sigma_megabarns * 1e-18
                return TWOOVERCLIGHTSQUARED * sigmabf * pow(nu, 2) * math.exp(-HOVERKB * nu / T)

            arr_nu = [en_ryd * RYD / H for en_ryd in phixslist[:, 0]]
            arr_integrand = [integrand(en_ryd, sigma_megabarns) for en_ryd, sigma_megabarns in phixslist]
            # integral_simps = integrate.simps(arr_integrand, arr_nu)
            integral_trapz = integrate.trapz(arr_integrand, arr_nu)

            factor = 4 * math.pi * sahafactor
            # print(f'    integral_euler alpha: {factor * integral_euler:11.3e} = {factor * integral_euler / alpha_nahar:6.3f} * Nahar')
            # print(f'    integral_simps alpha: {factor * integral_simps:11.3e} = {factor * integral_simps / alpha_nahar:6.3f} * Nahar')
            print(f'    integral_trapz alpha: {factor * integral_trapz:11.3e} '
                  f'= {factor * integral_trapz / alpha_nahar:6.3f} * Nahar     '
                  f'(abs error {abs(factor * integral_trapz - alpha_nahar):6.1e})')
            if reduced == 0:
                ion_alpha += factor * integral_trapz
            else:
                ion_alpha_reduced += factor * integral_trapz

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
            # print(f'    Nahar integral_trapz alpha: {factor * integral_trapz:11.3e}  = {factor * integral_trapz / alpha_nahar:6.3f} * Nahar')

    nahar_recomb_total = 0
    for _, recombrates_thislevel in recombrates.items():
        nahar_recomb_total += recombrates_thislevel[T_index]

    print(f'\nNahar ion Alpha:                    {nahar_recomb_total:13.3e}')
    print(f'Calculated ion Alpha:                 {ion_alpha:11.3e} = {ion_alpha / nahar_recomb_total:7.4f} * Nahar')
    print(f'Calculated ion Alpha (reduced phixs): {ion_alpha_reduced:11.3e} = {ion_alpha_reduced / nahar_recomb_total:7.4f} * Nahar')


if __name__ == "__main__":
    main()

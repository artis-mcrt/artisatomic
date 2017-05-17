#!/usr/bin/env python3
import math
import os
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from scipy import integrate, interpolate

PYDIR = os.path.dirname(os.path.abspath(__file__))
atomicdata = pd.read_csv(os.path.join(PYDIR, '..', 'atomic_properties.txt'), delim_whitespace=True, comment='#')
elsymbols = ['n'] + list(atomicdata['symbol'].values)


def reduce_phixs_tables(dicttables, nphixspoints, phixsnuincrement, optimaltemperature):
    dictout = {}

    ryd_to_hz = (u.rydberg / const.h).to('Hz').value
    h_over_kb_in_K_sec = (const.h / const.k_B).to('K s').value

    # proportional to recombination rate
    # nu0 = 1e16
    # fac = math.exp(h_over_kb_in_K_sec * nu0 / args.optimaltemperature)

    def integrand(nu):
        return (nu ** 2) * math.exp(- h_over_kb_in_K_sec * nu / optimaltemperature)

    integrand_vec = np.vectorize(integrand)

    xgrid = np.linspace(1.0, 1.0 + phixsnuincrement * (nphixspoints + 1),
                        num=nphixspoints + 1, endpoint=False)

    # for key in keylist:
    #   tablein = dicttables[key]
    for key, tablein in dicttables.items():
        # tablein is an array of pairs (energy, phixs cross section)

        # filter zero points out of the table
        firstnonzeroindex = 0
        for i, point in enumerate(tablein):
            if point[1] != 0.:
                firstnonzeroindex = i
                break
        if firstnonzeroindex != 0:
            tablein = tablein[firstnonzeroindex:]

        # table says zero threshold, so avoid divide by zero
        if tablein[0][0] == 0.:
            dictout[key] = np.zeros(nphixspoints)
            continue

        # nu0 = tablein[0][0] * ryd_to_hz

        arr_sigma_out = np.empty(nphixspoints)
        # x is nu/nu_edge

        sigma_interp = interpolate.interp1d(tablein[:, 0], tablein[:, 1], kind='linear', assume_sorted=True)

        for i, _ in enumerate(xgrid[:-1]):
            enlow = xgrid[i] * tablein[0][0]
            enhigh = xgrid[i + 1] * tablein[0][0]

            # start of interval interpolated point, Nahar points, and end of interval interpolated point
            samples_in_interval = tablein[(enlow <= tablein[:, 0]) & (tablein[:, 0] <= enhigh)]

            if len(samples_in_interval) == 0 or ((samples_in_interval[0, 0] - enlow)/enlow) > 1e-20:
                if i == 0:
                    print('adding first point {0:.4e} {1:.4e} {2:.4e}'.format(
                        enlow, samples_in_interval[0, 0], ((samples_in_interval[0, 0] - enlow)/enlow)))
                if enlow <= tablein[-1][0]:
                    new_crosssection = sigma_interp(enlow)
                    if new_crosssection < 0:
                        print('negative extrap')
                else:
                    # assume power law decay after last point
                    new_crosssection = tablein[-1][1] * (tablein[-1][0] / enlow) ** 3
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
                nsteps = 50  # was 500
                arr_energyryd = np.linspace(enlow, enhigh, num=nsteps, endpoint=False)

                # arr_sigma_megabarns = np.interp(arr_energyryd, tablein[:, 0], tablein[:, 1])

                # finterp = interpolate.interp1d(tablein[:, 0], tablein[:, 1], kind='nearest')
                # arr_sigma_megabarns = [finterp(en) for en in arr_energyryd]

                arr_sigma_megabarns = np.zeros(len(arr_energyryd))
                for index, en in enumerate(arr_energyryd):
                    arr_sigma_megabarns[index] = 0.0
                    for point in tablein:
                        if point[0] < en:
                            arr_sigma_megabarns[index] = point[1]
                        else:
                            break

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
                print(arr_sigma_out[i-1])
                print(arr_sigma_out[i])
                print(arr_sigma_out[i+1])
                arr_sigma_out[i] = 0.
                # sys.exit()

        dictout[key] = np.array(list(zip(xgrid[:-1] * tablein[0][0], arr_sigma_out)))

    return dictout


def get_recombratetable(atomicnumber, ionstage):
    temperatures = []
    recombrates = defaultdict(list)
    with open(f'../atomic-data-nahar/{elsymbols[atomicnumber].lower()}{ionstage}.rrc.ls.txt', 'r') as filein:
        while True:
            line = filein.readline()
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

                recombrates[current_level].extend([float(rate) for rate in row[2:]])
            elif current_level != "-1" and len(recombrates[current_level]) < len(temperatures):
                recombrates[current_level].extend([float(rate) for rate in row])

    return np.array(temperatures), recombrates


def get_phixslist(atomicnumber, ionstage, lowerlevel):
    phixslist = []
    with open(f'../atomic-data-nahar/{elsymbols[atomicnumber].lower()}{ionstage}.px.txt', 'r') as filein:
        while True:
            line = filein.readline()
            if line.startswith(f'    {lowerlevel.twosplusone}    {lowerlevel.l}'
                               f'    {lowerlevel.parity}    {lowerlevel.indexinsymmetry}'):
                print(line)
                break
        filein.readline()
        filein.readline()
        while True:
            line = filein.readline()
            if not line or len(line.split()) != 2:
                break
            row = line.split()
            phixslist.append([float(row[0]), float(row[1])])

    return phixslist


def main():
    CLIGHT = 2.99792458e10  # Speed of light.
    CLIGHT2 = 8.9875518e20  # Speed of light squared.
    H = 6.6260755e-27  # Planck constant in erg seconds
    RYD = 2.1798741e-11  # Rydberg to ergs
    KB = 1.38065e-16  # Boltzmann constant
    SAHACONST = 2.0706659e-16  # Saha constant

    CLIGHTSQUARED = 8.9875518e20
    TWOOVERCLIGHTSQUARED = 2.2253001e-21
    TWOHOVERCLIGHTSQUARED = 1.4745007e-47
    CLIGHTSQUAREDOVERTWOH = 6.7819570e46
    HOVERKB = 4.799243681748932e-11
    FOURPI = 1.256637061600000e+01
    ONEOVER4PI = 7.957747153555701e-02
    HCLIGHTOVERFOURPI = 1.580764662876770e-17
    OSCSTRENGTHCONVERSION = 1.3473837e+21


    nahar_level = namedtuple('naharlevel', 'twosplusone l parity indexinsymmetry')

    # Fe II -> III ground state to ground state
    # lowerlevel = nahar_level(6, 2, 0, 1)
    # g_upper = 25

    # Fe III -> IV ground state to ground state
    lowerlevel = nahar_level(5, 2, 0, 1)
    g_upper = 6
    atomicnumber = 26
    ionstage = 3

    T_goal = 6000
    nphixspoints = 100
    phixsnuincrement = 0.02

    temperatures, recombrates = get_recombratetable(atomicnumber, ionstage)
    T_index = (np.abs(temperatures - T_goal)).argmin()

    T = temperatures[T_index]  # Temperature in Kelvin

    transitionstr = '50202401.0000'
    print(f'{transitionstr} alpha:   {recombrates[transitionstr][T_index]:11.3e}')

    recomb_total = 0
    for _, recombrates_thislevel in recombrates.items():
        recomb_total += recombrates_thislevel[T_index]
    print(f'Total ion recomb rate: {recomb_total:11.3e}')

    phixslistorig = get_phixslist(atomicnumber, ionstage, lowerlevel)

    E_threshold = phixslistorig[0][0] * RYD

    g_lower = lowerlevel.twosplusone * (2 * lowerlevel.l + 1)
    sahafactor = g_lower / g_upper * SAHACONST * (T ** -1.5) * math.exp(E_threshold / KB / T)

    print(f'nphixspoints:    {nphixspoints:d}')
    print(f'phixsnuincrement:{phixsnuincrement:11.3e}')
    print(f'T(K):            {T:11.3e}')
    print(f'E_threshold(Ry): {E_threshold / RYD:11.3e}')
    print(f'sahafactor:      {sahafactor:11.3e}')

    phixslist_reduced = reduce_phixs_tables({'GS': np.array(phixslistorig)}, nphixspoints, phixsnuincrement, T)['GS']
    for phixslist in (phixslistorig, phixslist_reduced):
        print(f'\n{len(phixslist):d} points in list')

        nu = phixslist[0][0] * RYD / H
        sigma_bf = phixslist[0][1] * 1e-18

        integral = 0.0
        for i in range(0, len(phixslist) - 1):
            nu = phixslist[i][0] * RYD / H
            dnu = (phixslist[i + 1][0] - phixslist[i][0]) * RYD / H
            sigma_bf = phixslist[i][1] * 1e-18  # from Megabarn to cm^2
            dnu2 = dnu / 10.0
            for nu2 in np.arange(nu, nu + dnu, dnu2):
                sigma_bf = np.interp(nu2, [nu, nu + dnu], [phixslist[i][1], phixslist[i + 1][1]]) * 1e-18

                integral += TWOOVERCLIGHTSQUARED * sigma_bf * (nu2 ** 2) * math.exp(
                    -HOVERKB * nu2 / T) * dnu2

        def integrand(en_ryd, sigma_megabarns):
            nu = en_ryd * RYD / H
            sigmabf = sigma_megabarns * 1e-18
            return TWOOVERCLIGHTSQUARED * sigmabf * pow(nu, 2) * math.exp(-HOVERKB * nu / T)

        arr_nu = [en_ryd * RYD / H for en_ryd, _ in phixslist]
        arr_integrand = [integrand(en_ryd, sigma_megabarns) for en_ryd, sigma_megabarns in phixslist]
        integral_simps = integrate.simps(arr_integrand, arr_nu)
        integral_trapz = integrate.trapz(arr_integrand, arr_nu)

        print(f'integral part:   {integral:11.3e}  alpha: {4 * math.pi * sahafactor * integral:11.3e}')
        print(f'integral_simps:  {integral_simps:11.3e}  alpha: {4 * math.pi * sahafactor * integral_simps:11.3e}')
        print(f'integral_trapz:  {integral_trapz:11.3e}  alpha: {4 * math.pi * sahafactor * integral_trapz:11.3e}')


if __name__ == "__main__":
    main()

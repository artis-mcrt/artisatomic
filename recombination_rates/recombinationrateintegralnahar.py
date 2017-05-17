#!/usr/bin/env python3
import math
# import os
# import struct
# import sys

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import numpy as np
from collections import namedtuple

def main():
    CLIGHT = 2.99792458e10  # Speed of light.
    CLIGHT2 = 8.9875518e20  # Speed of light squared.
    H = 6.6260755e-27  # Planck constant in erg seconds
    ME = 9.1093897e-28  # Mass of free electron.
    MEV = 1.6021772e-6  # MeV to ergs
    PARSEC = 3.086e18  # 1 pc
    EV = 1.6021772e-12  # eV to ergs
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

    T = 1000  # Temperature in Kelvin

    nahar_level = namedtuple('naharlevel', 'twosplusone l parity indexinsymmetry')

    phixslistorig = []

    with open('../atomic-data-nahar/fe2.px.txt', 'r') as filein:
        while True:
            line = filein.readline()
            if line.startswith('    6    2    0    1'):
                break
        filein.readline()
        filein.readline()
        while True:
            line = filein.readline()
            if not line or len(line.split()) != 2:
                break
            row = line.split()
            phixslistorig.append([float(row[0]), float(row[1])])

    E_threshold = phixslistorig[0][0] * RYD

    lower = nahar_level(6, 2, 0, 1)
    g_lower = lower.twosplusone * (2 * lower.l + 1)
    sahafactor = g_lower / 25 * SAHACONST * (T**-1.5) * math.exp(E_threshold / KB / T)
    print('{0:14.6e}'.format(RYD / H))
    print('E_threshold(Ry): {0:14.6e}'.format(E_threshold / RYD))
    print('T(K):           {0:.1f}'.format(T))
    print('sahafactor:      {0:14.6e}'.format(sahafactor))

    for phixslist in (phixslistorig, ):  # smoothphixslist(phixslistorig,T)):
        print('\n{0:d} points in list'.format(len(phixslist)))

        nu = phixslist[0][0] * RYD / H
        sigma_bf = phixslist[0][1] * 1e-18

        integral = 0.0
        for i in range(0, len(phixslist) - 1):
            nu = phixslist[i][0] * RYD / H
            dnu = (phixslist[i + 1][0] - phixslist[i][0]) * RYD / H
            sigma_bf = phixslist[i][1] * 1e-18  # from Megabarn to cm^2
            dnu2 = dnu / 2000.0
            for nu2 in np.arange(nu, nu + dnu, dnu2):
                sigma_bf = np.interp(nu2, [nu, nu + dnu], [
                    phixslist[i][1], phixslist[i + 1][1]
                ]) * 1e-18
                integral += TWOOVERCLIGHTSQUARED * sigma_bf * (nu2**2) * math.exp(
                    -HOVERKB * nu2 / T) * dnu2
        alpha_sp = 4 * math.pi * sahafactor * integral

        print('Integral part: {0:14.6e}'.format(integral))
        print('alpha_sp:      {0:14.6e}'.format(alpha_sp))

if __name__ == "__main__":
    main()
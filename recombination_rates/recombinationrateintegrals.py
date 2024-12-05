#!/usr/bin/env python3
import math
import sys

import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

CLIGHT = 2.99792458e10  # /*Speed of light. */
CLIGHT2 = 8.9875518e20  # /*Speed of light squared. */
H = 6.6260755e-27  # /* Planck constant */
MH = 1.67352e-24  # /* Mass of hydrogen atom. */
ME = 9.1093897e-28  # /* Mass of free electron. */
QE = 4.80325e-10  # /* //MK: Elementary charge in cgs units*/
PI = math.pi  # /* PI - obviously. */
MEV = 1.6021772e-6  # /* MeV to ergs */
DAY = 86400.0  # /* day to seconds */
SIGMA_T = 6.6524e-25  # /* Thomson cross-section */
THOMSON_LIMIT = 1e-2  # /* Limit below which e-scattering is Thomson.*/
PARSEC = 3.086e18  # /* 1 pc */
EV = 1.6021772e-12  # /// eV to ergs
KB = 1.38065e-16  # /// Boltzmann constant
STEBO = 5.670400e-5  # /// Stefan-Boltzmann constant (data taken from NIST http://physics.nist.gov/cgi-bin/cuu/Value?eqsigma)
SAHACONST = 2.0706659e-16  # /// Saha constant

CLIGHTSQUARED = 8.9875518e20
TWOOVERCLIGHTSQUARED = 2.2253001e-21
TWOHOVERCLIGHTSQUARED = 1.4745007e-47
CLIGHTSQUAREDOVERTWOH = 6.7819570e46
HOVERKB = 4.799243681748932e-11
FOURPI = 1.256637061600000e01
ONEOVER4PI = 7.957747153555701e-02
HCLIGHTOVERFOURPI = 1.580764662876770e-17
OSCSTRENGTHCONVERSION = 1.3473837e21

T = 3000  # Temperature in Kelvin
atomicnumber = 26  # from
ionstage = 4  # from
levelnumber = 1  # from
E_thresholdinev = 54.801  # Fe I level 200
nu_edge = E_thresholdinev * EV / H
phixslist = []
phixsintegral = 0.0
lastpoint = []

print(f"Z={atomicnumber:d},fromionstage={ionstage:d},fromlevel={levelnumber:d}")
with open("example_run/phixsdata.txt", encoding="utf-8") as filein:
    while True:
        line = filein.readline()
        if not line:
            break
        headerrow = line.split()
        if (
            len(headerrow) == 6
            and int(headerrow[0]) == atomicnumber
            and int(headerrow[3]) == ionstage
            and int(headerrow[4]) == levelnumber
        ):
            for i in range(int(headerrow[5])):
                row = filein.readline().split()
                energyabovegsinev = float(row[0]) * 13.6
                sigma_bf = float(row[1])

                if energyabovegsinev < 9.0 * E_thresholdinev:
                    phixslist.append([energyabovegsinev, sigma_bf])
        elif len(headerrow) == 6:
            for i in range(int(headerrow[5])):
                filein.readline()
print(len(phixslist), " points")
if len(phixslist) == 0:
    sys.exit()
lastpoint = phixslist[-1]
# print(lastpoint[0]+E_thresholdinev,9*E_thresholdinev)
for energyabovegsinev in np.linspace(lastpoint[0], 9.0 * E_thresholdinev, num=100, endpoint=True):
    nulastpoint = (lastpoint[0] + E_thresholdinev) * EV / H
    nu = (energyabovegsinev + E_thresholdinev) * EV / H
    sigma_bf = lastpoint[1] * pow(nulastpoint / nu, 3)
    phixslist.append([energyabovegsinev, sigma_bf])

for i in range(1, len(phixslist)):
    energyabovegsinev = phixslist[i][0]
    sigma_bf = phixslist[i][1] * 1e-18

    nu = (energyabovegsinev + E_thresholdinev) * EV / H
    dnu = (phixslist[i][0] - phixslist[i - 1][0]) * EV / H
    x = TWOOVERCLIGHTSQUARED * sigma_bf * (nu**2) * math.exp(-HOVERKB * nu / T)
    # in formula this looks like
    # x = sigma_bf/H/nu * 2*H*pow(nu,3)/pow(CLIGHT,2) * exp(-H*nu/KB/T);
    phixsintegral += x * dnu

print(f"Integral: {phixsintegral:14.6e}")

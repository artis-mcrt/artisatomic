#!/usr/bin/env python3
# import itertools
import glob
# import math
import os
import sys
from collections import namedtuple
from pathlib import Path

# import numexpr as ne
import numpy as np
import pandas as pd
import pidly

import artistools as at
# import artisatomic

from astropy import constants as const
# from astropy import units as u

# selectedelements = [27]
selectedelements = None  # select all

OSCSTRENGTHCONVERSION = 1.3473837e+21
# = ME * pow(CLIGHT,3) / (8 * pow(QE*PI, 2)) * A_ul

def main():
    modelpath = Path()
    hc = (const.h * const.c).to('eV Angstrom').value

    ionlist = []
    compositiondata = at.get_composition_data(modelpath)
    for index, elementrow in compositiondata.iterrows():
        if selectedelements is None or elementrow.Z in selectedelements:
            ionlist.extend([
                (int(elementrow.Z), i) for i in range(int(elementrow.lowermost_ionstage), int(elementrow.uppermost_ionstage) + 1)])

            with open(f"Z{elementrow.Z:.0f}_levels.py.txt", "w") as levelfile:
                levelfile.write('#         z ion lvl ion_pot   ex_energy   g  rad_rate  \n')

            with open(f"Z{elementrow.Z:.0f}_lines.py.txt", "w") as linefile:
                linefile.write("""#
# f values from Menzel + Pekeris 1935 MNRAS 96 77
# supplemented with Gaunt factors from Baker + Menzel 1938 ApJ 88 52
#
# z = element, ion= ionstage, f = osc. str., gl(gu) = stat. we. lower(upper) level
# el(eu) = energy lower(upper) level (eV), ll(lu) = lvl index lower(upper) level
#        z ion       lambda      f         gl  gu    el          eu        ll   lu
""")

            with open(f"Z{elementrow.Z:.0f}_phot.py.txt", "w") as photfile:
                pass

    print(ionlist)
    adata = at.get_levels(modelpath, ionlist=tuple(ionlist),
                          get_transitions=True, get_photoionisations=True)

    for _, ion in adata.iterrows():
        atomic_number = int(ion.Z)
        if selectedelements is None or atomic_number in selectedelements:
            ionstage = int(ion.ion_stage)
            print(f"Doing Z={atomic_number:2.0f} ionstage {ionstage}")

            with open(f"Z{atomic_number:.0f}_levels.py.txt", "a") as levelfile:
                for levelindex, level in ion.levels.iterrows():
                    A_down_sum = ion.transitions.query('upper == @levelindex', inplace=False).A.sum()
                    rad_rate = 1. / A_down_sum if A_down_sum > 0. else 1e99
                    lineout = f'LevMacro {atomic_number:2.0f} {ionstage:2.0f} {levelindex + 1:4.0f} {level.energy_ev - ion.ion_pot:9.5f} {level.energy_ev:9.5f} {level.g:3.0f} {rad_rate:9.2e}\n'

                    levelfile.write(lineout)

            dftransitions = ion.transitions.copy()
            if not dftransitions.empty:
                with open(f"Z{atomic_number:.0f}_lines.py.txt", "a") as linefile:
                    dftransitions.eval('upper_g = @ion.levels.loc[upper].g.values', inplace=True)
                    dftransitions.eval('lower_g = @ion.levels.loc[lower].g.values', inplace=True)

                    dftransitions.eval('upper_energy_ev = @ion.levels.loc[upper].energy_ev.values', inplace=True)
                    dftransitions.eval('lower_energy_ev = @ion.levels.loc[lower].energy_ev.values', inplace=True)
                    dftransitions.eval('lambda_angstroms = @hc / (upper_energy_ev - lower_energy_ev)', inplace=True)

                    c_angps = const.c.to('angstrom/s').value
                    dftransitions.eval('fosc = upper_g / lower_g * @OSCSTRENGTHCONVERSION / (@c_angps / lambda_angstroms) ** 2 * A', inplace=True)

                    for _, transition in dftransitions.iterrows():
                        lineout = f'LinMacro {atomic_number:4.0f} {ionstage:3.0f} {transition.lambda_angstroms:19.5f} {transition.fosc:9.5e} {transition.upper_g:5.0f} {transition.lower_g:5.0f} {transition.lower_energy_ev:9.5f} {transition.upper_energy_ev:9.5f} {transition.lower + 1:5.0f} {transition.upper + 1:5.0f}\n'

                        linefile.write(lineout)

            with open(f"Z{atomic_number:.0f}_phot.py.txt", "a") as photfile:
                for levelindex, level in ion.levels.iterrows():
                    for upperlevelindex, targetfrac in level.phixstargetlist:
                        photfile.write(f'PhotMacS       {atomic_number:.0f} {ionstage:7.0f} {levelindex + 1:7.0f} {upperlevelindex + 1:7.0f} {ion.ion_pot:15.6f} {len(level.phixstable):7.0f}\n')
                        for x, xs in level.phixstable:
                            # the photoionisation cross-sections in the database are given in Mbarn = 1e6 * 1e-28m^2
                            # to convert to cgs units multiply by 1e-18
                            photfile.write(f'PhotMac {x * ion.ion_pot:15.6f} {xs * targetfrac * 1e-18:15.7e}\n')

if __name__ == "__main__":
    main()
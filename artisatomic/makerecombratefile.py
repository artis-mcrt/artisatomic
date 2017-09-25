#!/usr/bin/env python3
# import itertools
import glob
# import math
import os
import sys
from collections import namedtuple

# import numexpr as ne
import numpy as np
import pandas as pd
import pidly

import artistools as at
import artisatomic

# from astropy import constants as const
# from astropy import units as u


def read_nahar_rrcfile(filename):
    print(f'Reading {filename}')

    header_row = []
    with open(filename, 'r') as filein:
        while True:
            line = filein.readline()
            if line.strip().startswith('TOTAL RECOMBINATION RATE'):
                line = filein.readline()
                line = filein.readline()
                header_row = filein.readline().strip().replace(' n)', '-n)').split()
                break

        if not header_row:
            print("ERROR: no header found")
            sys.exit()

        index_logt = header_row.index('log(T)')
        index_low_n = header_row.index('RRC(low-n)')
        index_tot = header_row.index('RRC(total)')

        recomb_tuple = namedtuple("recomb_tuple", ['logT', 'RRC_low_n', 'RRC_total'])
        records = []
        for line in filein:
            row = line.split()
            if row:
                if len(row) != len(header_row):
                    print('Row contains wrong number of items for header:')
                    print(header_row)
                    print(row)
                    sys.exit()
                records.append(recomb_tuple(
                    *[float(row[index]) for index in [index_logt, index_low_n, index_tot]]))

    dfrecombrates = pd.DataFrame.from_records(records, columns=recomb_tuple._fields)
    return dfrecombrates


def main():
    dir(pidly.IDL)
    idl = pidly.IDL('/usr/local/ssw/gen/setup/ssw_idl')
    dir(idl)
    idl.setecho("False")

    atomic_number = 26
    ion_stage = 1

    dfcomposition = at.get_composition_data('artis_files/compositiondata.txt')

    with open(os.path.join('artis_files', f'recombrates.txt'), 'w') as frecombrates:
        for _, comprow in dfcomposition.iterrows():
            atomic_number = int(comprow.Z)
            for lowerionstage in range(int(comprow.lowermost_ionstage), int(comprow.uppermost_ionstage)):
                upperionstage = lowerionstage + 1
                print(f'{artisatomic.elsymbols[atomic_number]} {upperionstage}->{lowerionstage}')

                print(atomic_number, lowerionstage)
                rrcfiles = glob.glob(
                    f'atomic-data-nahar/{artisatomic.elsymbols[atomic_number].lower()}{lowerionstage}.rrc*.txt')

                if atomic_number == 28:
                    frecombrates.write(f'{atomic_number} {upperionstage} {len(arr_logT_e)}\n')
                    # Shull & Steenberg 1982
                    if lowerionstage == 1:
                        A_rad, X_rad = 3.60e-13, 0.700
                    elif lowerionstage == 2:
                        A_rad, X_rad = 1.00e-12, 0.700
                    elif lowerionstage == 3:
                        A_rad, X_rad = 1.40e-12, 0.700
                    elif lowerionstage == 4:
                        A_rad, X_rad = 1.60e-12, 0.700
                    elif lowerionstage == 5:
                        A_rad, X_rad = 3.85e-12, 0.746
                    elif lowerionstage == 6:
                        A_rad, X_rad = 9.05e-12, 0.682

                    arr_logT_e = np.arange(1.0, 9.1, 0.1)
                    for logT_e in arr_logT_e:
                        T_e = 10 ** logT_e
                        rrc = A_rad * (T_e / 1e4) ** - X_rad
                        frecombrates.write(f"{logT_e:.1f} {-1.0} {rrc}\n")

                elif rrcfiles:  # use Nahar's vlaues if available
                    filename = rrcfiles[0]
                    ionstr = os.path.basename(filename).split('.')[0]  # should be something like 'fe2'
                    elsymbol = ionstr.rstrip('0123456789')
                    lowerionstage = int(ionstr[len(elsymbol):])
                    upperionstage = lowerionstage + 1
                    atomic_number = artisatomic.elsymbols.index(elsymbol.title())
                    dfrecombrates = read_nahar_rrcfile(filename)
                    frecombrates.write(f'{atomic_number} {upperionstage} {len(dfrecombrates)}\n')
                    for _, row in dfrecombrates.iterrows():
                        frecombrates.write(f"{row['logT']} {row['RRC_low_n']} {row['RRC_total']}\n")

                else:  # use Chianti
                    arr_logT_e = np.arange(1.0, 9.1, 0.1)
                    frecombrates.write(f'{atomic_number} {upperionstage} {len(arr_logT_e)}\n')
                    for logT_e in arr_logT_e:
                        idl(f"rrc = RECOMB_RATE('{artisatomic.elsymbols[atomic_number].lower()}_{upperionstage}', {10 ** logT_e})")
                        try:
                            rrc = float(idl.rrc[0])
                        except IndexError:
                            rrc = float(idl.rrc)
                        frecombrates.write(f"{logT_e:.1f} {-1.0} {rrc}\n")

    idl.close()


if __name__ == "__main__":
    main()

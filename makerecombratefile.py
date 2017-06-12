#!/usr/bin/env python3
# import itertools
import math
import os
import glob
import sys
from collections import defaultdict, namedtuple

# import numexpr as ne
import numpy as np
import pandas as pd
# from astropy import constants as const
# from astropy import units as u

import makeartisatomicfiles as artisatomic


def get_ionrecombrates_fromfile(filename, atomicnumber, ionstage):
    # filename = f'atomic-data-nahar/{artisatomic.elsymbols[atomicnumber].lower()}{ionstage}.rrc.ls.txt'
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
    # T_goal = 6.31E+03
    atomic_number = 26
    ion_stage = 1

    with open(os.path.join('artis_files', f'recombrates.txt'), 'w') as frecombrates:
        rrcfiles = glob.glob('atomic-data-nahar/*rrc*.txt')
        for filename in rrcfiles:
            ionstr = os.path.basename(filename).split('.')[0]  # should be something like 'fe2'
            elsymbol = ionstr.rstrip('0123456789')
            lowerionstage = int(ionstr[len(elsymbol):])
            upperionstage = lowerionstage + 1
            atomic_number = artisatomic.elsymbols.index(elsymbol.title())
            dfrecombrates = get_ionrecombrates_fromfile(filename, atomic_number, ion_stage)
            frecombrates.write(f'{atomic_number} {upperionstage} {len(dfrecombrates)}\n')
            for _, row in dfrecombrates.iterrows():
                frecombrates.write(f"{row['logT']} {row['RRC_low_n']} {row['RRC_total']}\n")


if __name__ == "__main__":
    main()

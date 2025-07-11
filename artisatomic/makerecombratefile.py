#!/usr/bin/env python3
# import itertools
import glob
import os
import sys
from collections import namedtuple
from pathlib import Path

import ChiantiPy.core as ch
import numpy as np
import pandas as pd
from artistools import get_composition_data

import artisatomic


def read_nahar_rrcfile(filename, noprint=False):
    if not noprint:
        print(f"  reading {filename}")

    header_row = []
    with open(filename) as filein:
        while True:
            line = filein.readline()
            if line.strip().startswith("TOTAL RECOMBINATION RATE"):
                line = filein.readline()
                line = filein.readline()
                header_row = filein.readline().strip().replace(" n)", "-n)").split()
                break

        if not header_row:
            print("ERROR: no header found")
            sys.exit()

        index_logt = header_row.index("log(T)")
        index_low_n = header_row.index("RRC(low-n)")
        index_tot = header_row.index("RRC(total)")

        recomb_tuple = namedtuple("recomb_tuple", ["logT", "RRC_low_n", "RRC_total"])
        records = []
        for line in filein:
            if row := line.split():
                if len(row) != len(header_row):
                    print("Row contains wrong number of items for header:")
                    print(header_row)
                    print(row)
                    sys.exit()
                records.append(recomb_tuple(*[float(row[index]) for index in [index_logt, index_low_n, index_tot]]))

    return pd.DataFrame.from_records(records, columns=recomb_tuple._fields)


def main():
    # Shull & Steenberg 1982
    A_rad, X_rad = {}, {}
    A_rad[26, 1], X_rad[26, 1] = 1.42e-13, 0.891
    A_rad[26, 2], X_rad[26, 2] = 1.02e-12, 0.843
    A_rad[26, 3], X_rad[26, 3] = 3.32e-12, 0.746
    A_rad[26, 4], X_rad[26, 4] = 7.80e-12, 0.682
    A_rad[26, 5], X_rad[26, 5] = 1.51e-11, 0.699
    A_rad[26, 6], X_rad[26, 6] = 2.62e-11, 0.728

    A_rad[28, 1], X_rad[28, 1] = 3.60e-13, 0.700
    A_rad[28, 2], X_rad[28, 2] = 1.00e-12, 0.700
    A_rad[28, 3], X_rad[28, 3] = 1.40e-12, 0.700
    A_rad[28, 4], X_rad[28, 4] = 1.60e-12, 0.700
    A_rad[28, 5], X_rad[28, 5] = 3.85e-12, 0.746
    A_rad[28, 6], X_rad[28, 6] = 9.05e-12, 0.682

    dfcomposition = get_composition_data("artis_files/compositiondata.txt")

    with open(os.path.join("artis_files", "recombrates.txt"), "w") as frecombrates:
        for _, comprow in dfcomposition.iterrows():
            atomic_number = int(comprow.Z)
            for lowerionstage in range(int(comprow["lowermost_ion_stage"]), int(comprow["uppermost_ion_stage"])):
                upperionstage = lowerionstage + 1
                print(f"Z={atomic_number} {artisatomic.elsymbols[atomic_number]} {upperionstage}->{lowerionstage}")

                # if atomic_number == 28:  # Pure Shull & Steenberg 1982
                #     arr_logT_e = np.arange(1.0, 9.1, 0.1)
                #     frecombrates.write(f'{atomic_number} {upperionstage} {len(arr_logT_e)}\n')
                #     for logT_e in arr_logT_e:
                #         T_e = 10 ** logT_e
                #         rrc = A_rad[atomic_number, lowerionstage] * (T_e / 1e4) ** - X_rad[atomic_number, lowerionstage]
                #         frecombrates.write(f"{logT_e:.1f} {-1.0} {rrc}\n")

                if rrcfiles := glob.glob(
                    f"atomic-data-nahar/{artisatomic.elsymbols[atomic_number].lower()}{lowerionstage}.rrc*.txt"
                ):  # use Nahar's values if available
                    naharfilename = rrcfiles[0]
                    ionstr = Path(naharfilename).name.split(".")[0]  # should be something like 'fe2'
                    elsymbol = ionstr.rstrip("0123456789")
                    lowerionstage = int(ionstr[len(elsymbol) :])
                    upperionstage = lowerionstage + 1
                    atomic_number = artisatomic.elsymbols.index(elsymbol.title())
                    dfrecombrates = read_nahar_rrcfile(naharfilename)
                    frecombrates.write(f"{atomic_number} {upperionstage} {len(dfrecombrates)}\n")
                    for _, row in dfrecombrates.iterrows():
                        frecombrates.write(f"{row['logT']} {row['RRC_low_n']} {row['RRC_total']}\n")

                # elif atomic_number == 28 and lowerionstage >= 3:
                #     # Get Nahar's boost factors relative to SS82 for Fe, and apply them to the SS82 rates for Ni
                #     rrcfiles = glob.glob(
                #         f'atomic-data-nahar/fe{lowerionstage - 2}.rrc*.txt')
                #     dfrecombrates = read_nahar_rrcfile(rrcfiles[0], noprint=True)
                #     frecombrates.write(f'{atomic_number} {upperionstage} {len(dfrecombrates)}\n')
                #     for _, row in dfrecombrates.iterrows():
                #         T_e = 10 ** row['logT']
                #         rrc_fe_ss82 = A_rad[26, lowerionstage - 2] * (T_e / 1e4) ** - X_rad[26, lowerionstage - 2]
                #         rrc_ni_ss82 = A_rad[28, lowerionstage] * (T_e / 1e4) ** - X_rad[28, lowerionstage]
                #         rrc_total = rrc_ni_ss82 * (row['RRC_total'] / rrc_fe_ss82)
                #         print(f"Log T {row['logT']} Nahar/SS1982 boost factor: {(row['RRC_total'] / rrc_fe_ss82)}")
                #         frecombrates.write(f"{row['logT']} {-1.0} {rrc_total}\n")

                else:  # use Chianti with ChiantiPy
                    print("  using Chianti")
                    arr_logT_e = np.arange(1.0, 9.1, 0.1)
                    frecombrates.write(f"{atomic_number} {upperionstage} {len(arr_logT_e)}\n")
                    arr_temperature = 10**arr_logT_e
                    ion = ch.ion(
                        f"{artisatomic.elsymbols[atomic_number].lower()}_{upperionstage}", temperature=arr_temperature
                    )
                    ion.rrRate()
                    arr_rrc = ion.RrRate["rate"]
                    ion.drRate()
                    arr_drc = ion.DrRate["rate"]
                    frecombrates.writelines(
                        f"{logT_e:.1f} {-1.0} {arr_rrc[i]}\n" for i, logT_e in enumerate(arr_logT_e)
                    )


if __name__ == "__main__":
    main()

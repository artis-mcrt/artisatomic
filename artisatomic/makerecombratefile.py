#!/usr/bin/env python3
"""Write recombrates.txt from the Nahar total recombination rate files."""

import typing as t
from pathlib import Path

import ChiantiPy.core as ch
import numpy as np
import pandas as pd
from artistools import get_composition_data

from artisatomic.base import elsymbols
from artisatomic.base import PYDIR


class RecombRow(t.NamedTuple):
    """One row of a Nahar .rrc total recombination rate table."""

    logT: float  # ruff: ignore[mixed-case-variable-in-class-scope]  # the field name is the DataFrame column read back below
    RRC_low_n: float
    RRC_total: float


def read_nahar_rrcfile(filename, noprint=False):
    """Read a Nahar total recombination rate file (.rrc) as a table of temperature and rate."""
    if not noprint:
        print(f"  reading {filename}")

    header_row: list[str] = []
    with Path(filename).open(encoding="utf-8") as filein:
        while True:
            line = filein.readline()
            if not line:  # end of file, otherwise a file without the marker would loop forever
                break
            if line.strip().startswith("TOTAL RECOMBINATION RATE"):
                line = filein.readline()
                line = filein.readline()
                header_row = filein.readline().strip().replace(" n)", "-n)").split()
                break

        if not header_row:
            msg = "no header found"
            raise ValueError(msg)

        index_logt = header_row.index("log(T)")
        index_low_n = header_row.index("RRC(low-n)")
        index_tot = header_row.index("RRC(total)")

        records = []
        for line in filein:
            if row := line.split():
                if len(row) != len(header_row):
                    msg = f"Row contains wrong number of items for header:\n{header_row}\n{row}"
                    raise ValueError(msg)
                records.append(RecombRow(*[float(row[index]) for index in [index_logt, index_low_n, index_tot]]))

    return pd.DataFrame(records)


def main():
    """Write recombrates.txt from the Nahar recombination rate files."""
    artis_files_path = PYDIR.parent / "artis_files"
    dfcomposition = get_composition_data(artis_files_path / "compositiondata.txt")

    with Path(artis_files_path / "recombrates.txt").open(mode="w", encoding="utf-8") as frecombrates:
        for Z, lowermost_ion_stage, uppermost_ion_stage in dfcomposition.select(
            "Z", "lowermost_ion_stage", "uppermost_ion_stage"
        ).iter_rows():
            atomic_number = int(Z)
            for lowerionstage in range(int(lowermost_ion_stage), int(uppermost_ion_stage)):
                upperionstage = lowerionstage + 1
                print(f"Z={atomic_number} {elsymbols[atomic_number]} {upperionstage}->{lowerionstage}")

                # the glob is anchored to the repository, so the entry point finds the Nahar
                # files from any working directory; sorted() makes the choice deterministic
                # when more than one file matches
                rrcfiles = sorted(
                    (PYDIR.parent / "atomic-data-nahar").glob(
                        f"{elsymbols[atomic_number].lower()}{lowerionstage}.rrc*.txt"
                    )
                )
                if rrcfiles:  # use Nahar's values if available
                    naharfilename = rrcfiles[0]
                    dfrecombrates = read_nahar_rrcfile(naharfilename)
                    frecombrates.write(f"{atomic_number} {upperionstage} {len(dfrecombrates)}\n")
                    frecombrates.writelines(
                        f"{row['logT']} {row['RRC_low_n']} {row['RRC_total']}\n" for _, row in dfrecombrates.iterrows()
                    )
                else:  # use Chianti with ChiantiPy
                    print("  using Chianti")
                    arr_logT_e = np.arange(1.0, 9.1, 0.1)
                    frecombrates.write(f"{atomic_number} {upperionstage} {len(arr_logT_e)}\n")
                    arr_temperature = 10**arr_logT_e
                    ion = ch.ion(f"{elsymbols[atomic_number].lower()}_{upperionstage}", temperature=arr_temperature)
                    ion.rrRate()
                    arr_rrc = ion.RrRate["rate"]
                    ion.drRate()
                    arr_drc = ion.DrRate["rate"]
                    # the third column is the total recombination rate, matching RRC(total) used
                    # for the Nahar files above, so dielectronic recombination must be included
                    frecombrates.writelines(
                        f"{logT_e:.1f} {-1.0} {arr_rrc[i] + arr_drc[i]}\n" for i, logT_e in enumerate(arr_logT_e)
                    )


if __name__ == "__main__":
    main()

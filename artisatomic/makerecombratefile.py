#!/usr/bin/env python3
"""Write recombrates.txt from the Nahar total recombination rate files."""

import argparse
import importlib
import typing as t
from pathlib import Path

import numpy as np
from artistools import get_composition_data

from artisatomic.base import elsymbols
from artisatomic.base import PYDIR


class RecombRow(t.NamedTuple):
    """One row of a Nahar .rrc total recombination rate table."""

    logT: float  # ruff: ignore[mixed-case-variable-in-class-scope]  # the name matches the Nahar header
    RRC_low_n: float
    RRC_total: float


def read_nahar_rrcfile(filename, noprint=False) -> list[RecombRow]:
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
            msg = "the file has no header row"
            raise ValueError(msg)

        index_logt = header_row.index("log(T)")
        index_low_n = header_row.index("RRC(low-n)")
        index_tot = header_row.index("RRC(total)")

        records = []
        for line in filein:
            if row := line.split():
                if len(row) != len(header_row):
                    msg = f"The row does not have the number of items of the header:\n{header_row}\n{row}"
                    raise ValueError(msg)
                records.append(RecombRow(*[float(row[index]) for index in [index_logt, index_low_n, index_tot]]))

    return records


def import_chianti_core(firstion: str) -> t.Any:
    """Import ChiantiPy.core, and name the missing module if the import fails.

    The import is not at the top of the file. ChiantiPy takes 0.6 s to import and writes four
    warning lines. A run with a Nahar file for every ion never calls this function.

    ChiantiPy is in the optional chianti extra, so a checkout without that extra has no module
    to resolve. import_module() keeps the checkers away from the name. A plain import statement
    needs a suppression comment, and pyrefly then reports that comment as unused when the extra
    is present.

    ChiantiPy imports matplotlib and scipy at module level, so err.name names the module that is
    absent. firstion names the first ion that has no Nahar file.
    """
    try:
        return importlib.import_module("ChiantiPy.core")
    except ModuleNotFoundError as err:
        msg = (
            f"No Nahar file for {firstion}, so the rates come from Chianti. The module"
            f" {err.name} is not installed. Install the chianti extra:"
            " uv sync --frozen --extra chianti"
        )
        raise ModuleNotFoundError(msg) from err


def main():
    """Write recombrates.txt from the Nahar recombination rate files."""
    parser = argparse.ArgumentParser(description=__doc__)
    # the same default as makeartisatomicfiles, relative to the working directory, so the two
    # scripts read and write one folder
    parser.add_argument(
        "-output_folder", default="artis_files", type=Path, help="folder of compositiondata.txt and the output file"
    )
    args = parser.parse_args()
    artis_files_path = Path(args.output_folder)

    # the Nahar files are in no checkout of this repository. Without this line, an absent
    # directory sends every ion to Chianti and the output gives no sign of it
    naharpath = PYDIR.parent / "atomic-data-nahar"
    if naharpath.is_dir():
        print(f"Nahar data directory: {naharpath}")
    else:
        print(f"Nahar data directory {naharpath} not found. Every ion takes its rates from Chianti.")

    dfcomposition = get_composition_data(artis_files_path / "compositiondata.txt")

    # The source of every ion comes first, and the output file opens after it. A glob is cheap,
    # and an absent python module must leave no truncated recombrates.txt behind.
    ionsources: list[tuple[int, int, Path | None]] = []
    for Z, lowermost_ion_stage, uppermost_ion_stage in dfcomposition.select(
        "Z", "lowermost_ion_stage", "uppermost_ion_stage"
    ).iter_rows():
        atomic_number = int(Z)
        for lowerionstage in range(int(lowermost_ion_stage), int(uppermost_ion_stage)):
            # the glob starts at the repository, so the entry point finds the Nahar files
            # from any working directory. sorted() makes the choice deterministic when
            # more than one file matches.
            rrcfiles = sorted(naharpath.glob(f"{elsymbols[atomic_number].lower()}{lowerionstage}.rrc*.txt"))
            ionsources.append((atomic_number, lowerionstage, rrcfiles[0] if rrcfiles else None))

    firstchiantiion = next(
        (
            f"Z={atomic_number} {elsymbols[atomic_number]} ion stage {lowerionstage}"
            for atomic_number, lowerionstage, naharfilename in ionsources
            if naharfilename is None
        ),
        None,
    )
    ch = import_chianti_core(firstchiantiion) if firstchiantiion is not None else None

    with Path(artis_files_path / "recombrates.txt").open(mode="w", encoding="utf-8") as frecombrates:
        for atomic_number, lowerionstage, naharfilename in ionsources:
            upperionstage = lowerionstage + 1
            print(f"Z={atomic_number} {elsymbols[atomic_number]} {upperionstage}->{lowerionstage}")

            if naharfilename is not None:
                recombrates = read_nahar_rrcfile(naharfilename)
                frecombrates.write(f"{atomic_number} {upperionstage} {len(recombrates)}\n")
                frecombrates.writelines(f"{row.logT} {row.RRC_low_n} {row.RRC_total}\n" for row in recombrates)
            else:
                assert ch is not None
                print("  source: Chianti")
                arr_logT_e = np.arange(1.0, 9.1, 0.1)
                frecombrates.write(f"{atomic_number} {upperionstage} {len(arr_logT_e)}\n")
                arr_temperature = 10**arr_logT_e
                ion = ch.ion(f"{elsymbols[atomic_number].lower()}_{upperionstage}", temperature=arr_temperature)
                ion.rrRate()
                arr_rrc = ion.RrRate["rate"]
                ion.drRate()
                arr_drc = ion.DrRate["rate"]
                # the third column is the total recombination rate, the same as RRC(total) of
                # the Nahar files above, so the sum must include dielectronic recombination
                frecombrates.writelines(
                    f"{logT_e:.1f} {-1.0} {arr_rrc[i] + arr_drc[i]}\n" for i, logT_e in enumerate(arr_logT_e)
                )


if __name__ == "__main__":
    main()

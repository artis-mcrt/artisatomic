#!/usr/bin/env python3
"""Write recombrates.txt from the Nahar total recombination rate files."""

import argparse
import importlib
import math
import re
import typing as t
from itertools import pairwise
from pathlib import Path

import numpy as np
from artistools.atomic import get_composition_data

from artisatomic.base import elsymbols
from artisatomic.base import open_for_atomic_write
from artisatomic.base import PYDIR


class RecombRow(t.NamedTuple):
    """One row of a Nahar .rrc total recombination rate table."""

    logT: float  # ruff: ignore[mixed-case-variable-in-class-scope]  # the name matches the Nahar header
    RRC_low_n: float
    RRC_total: float


def read_nahar_rrcfile(filename) -> list[RecombRow]:
    """Read a Nahar total recombination rate file (.rrc) as a table of temperature and rate."""
    print(f"  reading {filename}")

    header_row: list[str] = []
    with Path(filename).open(encoding="utf-8") as filein:
        while True:
            line = filein.readline()
            if not line:  # end of file, otherwise a file without the marker would loop forever
                break
            if line.strip().startswith("TOTAL RECOMBINATION RATE"):
                filein.readline()
                filein.readline()
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

    # ARTIS stops for an ion with no temperature in its block, so a file with no rows must stop here
    if not records:
        msg = f"{filename} has a header row and no data rows"
        raise ValueError(msg)

    for row in records:
        if not all(math.isfinite(value) for value in row) or row.RRC_low_n < 0.0 or row.RRC_total < 0.0:
            msg = f"{filename} has a non-finite temperature or a negative or non-finite rate."
            raise ValueError(msg)
    if any(right.logT <= left.logT for left, right in pairwise(records)):
        msg = f"The temperatures in {filename} must increase without a duplicate."
        raise ValueError(msg)
    return records


def find_nahar_file(naharpath: Path, atomic_number: int, lowerionstage: int) -> Path | None:
    """Give the Nahar file of the recombination to the ion stage lowerionstage, or None if there is none.

    The glob also finds a copy that a sync client made, for example "fe2.rrc (1).txt". Such a copy
    can have an earlier version of the data, so the function skips a name with a space or a
    bracket, with a warning. sorted() makes the choice deterministic when more than one file remains.
    """
    prefix = f"{elsymbols[atomic_number].lower()}{lowerionstage}"
    name_regex = re.compile(rf"{re.escape(prefix)}\.rrc[A-Za-z0-9._-]*\.txt")
    rrcfiles = []
    for rrcfile in sorted(naharpath.glob(f"{prefix}.rrc*.txt")):
        if name_regex.fullmatch(rrcfile.name) is None:
            print(f"WARNING: The program skipped the file {rrcfile.name}, because the name is not {prefix}.rrc*.txt")
            continue
        rrcfiles.append(rrcfile)
    return rrcfiles[0] if rrcfiles else None


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


def checked_rate_array(values, count: int, source: str) -> np.ndarray:
    """Require one finite, non-negative rate for each requested temperature."""
    rates = np.asarray(values, dtype=float)
    if rates.shape != (count,) or not np.isfinite(rates).all() or (rates < 0.0).any():
        msg = f"{source} must give {count} finite, non-negative rates."
        raise ValueError(msg)
    return rates


def chianti_recombination_rates(ch, atomic_number: int, upperionstage: int, temperatures: np.ndarray) -> np.ndarray:
    """Read both Chianti rates and require their source parameters before use."""
    ionname = f"{elsymbols[atomic_number].lower()}_{upperionstage}"
    ion = ch.ion(ionname, temperature=temperatures)
    ion.rrRate()
    if not getattr(ion, "RrParams", None):
        msg = f"Chianti has no radiative recombination parameters for {ionname}. Supply a Nahar rate file."
        raise ValueError(msg)
    rr = checked_rate_array(ion.RrRate["rate"], len(temperatures), f"Chianti radiative recombination of {ionname}")
    ion.drRate()
    # A bare nucleus has no bound electron to excite during dielectronic recombination.
    if upperionstage != atomic_number + 1 and not getattr(ion, "DrParams", None):
        msg = f"Chianti has no dielectronic recombination parameters for {ionname}. Supply a Nahar rate file."
        raise ValueError(msg)
    dr = checked_rate_array(ion.DrRate["rate"], len(temperatures), f"Chianti dielectronic recombination of {ionname}")
    try:
        with np.errstate(over="raise"):
            total = rr + dr
    except FloatingPointError as error:
        msg = f"The total recombination rate of {ionname} exceeds the finite range."
        raise ValueError(msg) from error
    return checked_rate_array(total, len(temperatures), f"Chianti total recombination of {ionname}")


# the largest atomic number of the CHIANTI database
max_chianti_atomic_number = 30


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

    # the source of every ion comes first, so the program imports ChiantiPy only if an ion needs it
    ionsources: list[tuple[int, int, Path | None]] = []
    for Z, lowermost_ion_stage, uppermost_ion_stage in dfcomposition.select(
        "Z", "lowermost_ion_stage", "uppermost_ion_stage"
    ).iter_rows():
        atomic_number = int(Z)
        # naharpath starts at the repository, so the entry point finds the Nahar files from any
        # working directory
        ionsources.extend(
            (atomic_number, lowerionstage, find_nahar_file(naharpath, atomic_number, lowerionstage))
            for lowerionstage in range(int(lowermost_ion_stage), int(uppermost_ion_stage))
        )

    # CHIANTI holds the ions of Z <= 30 only, and ChiantiPy gives an error that names no ion for
    # another one. ARTIS keeps its own rate for an ion with no block in recombrates.txt.
    for atomic_number, lowerionstage, naharfilename in ionsources:
        if naharfilename is None and atomic_number > max_chianti_atomic_number:
            print(
                f"WARNING: Z={atomic_number} {elsymbols[atomic_number]} ion stage {lowerionstage + 1} has no Nahar file,"
                " and CHIANTI has no data above Z = 30. recombrates.txt gives no rates for this ion."
            )
    ionsources = [source for source in ionsources if source[2] is not None or source[0] <= max_chianti_atomic_number]

    firstchiantiion = next(
        (
            f"Z={atomic_number} {elsymbols[atomic_number]} ion stage {lowerionstage}"
            for atomic_number, lowerionstage, naharfilename in ionsources
            if naharfilename is None
        ),
        None,
    )
    ch = import_chianti_core(firstchiantiion) if firstchiantiion is not None else None

    # the earlier recombrates.txt stays if a reader or Chianti fails
    with open_for_atomic_write(artis_files_path / "recombrates.txt") as frecombrates:
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
                arr_temperature = 10**arr_logT_e
                arr_total = chianti_recombination_rates(ch, atomic_number, upperionstage, arr_temperature)
                frecombrates.write(f"{atomic_number} {upperionstage} {len(arr_logT_e)}\n")
                # the third column is the total recombination rate, the same as RRC(total) of
                # the Nahar files above, so the sum must include dielectronic recombination
                frecombrates.writelines(f"{logT_e:.1f} {-1.0} {arr_total[i]}\n" for i, logT_e in enumerate(arr_logT_e))


if __name__ == "__main__":
    main()

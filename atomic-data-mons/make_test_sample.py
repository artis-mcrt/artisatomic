#!/usr/bin/env python3
"""Write the reduced MONS archives that the tests/mons checksum test reads.

The published archives are too large for the repository: outggf_Ln_V--VII.zip alone is 21.7 GB.
This script cuts a sample from them. It keeps the lowest levels of each ion that the ion list
names, and every transition with both of its levels in that set. The line order of the original
files does not change, so the sample also tests the sort that the reader applies.

Run it from the atomic-data-mons directory, with both full archives present:

    uv run python make_test_sample.py

Regenerate tests/mons/checksums.txt afterwards, as tests/README.md describes.
"""

import zipfile
from pathlib import Path

# one entry for each ion of the sample, with the number of levels to keep
SAMPLE_IONS = {"Ce_V": 450, "Ce_VI": 400}

LEVELS_ARCHIVE = "outglv_Ln_V--VII.zip"
TRANSITIONS_ARCHIVE = "outggf_Ln_V--VII.zip"


def main() -> None:
    """Write one reduced archive of levels and one of transitions into test_sample/."""
    basepath = Path(__file__).parent.resolve()
    outputpath = basepath / "test_sample"
    outputpath.mkdir(exist_ok=True)

    with (
        zipfile.ZipFile(basepath / LEVELS_ARCHIVE) as ziplevels_in,
        zipfile.ZipFile(basepath / TRANSITIONS_ARCHIVE) as ziptransitions_in,
        zipfile.ZipFile(outputpath / LEVELS_ARCHIVE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as ziplevels_out,
        zipfile.ZipFile(
            outputpath / TRANSITIONS_ARCHIVE, "w", zipfile.ZIP_DEFLATED, compresslevel=9
        ) as ziptransitions_out,
    ):
        for ion, levelcount in SAMPLE_IONS.items():
            levelsmember = f"outglv_Ln_V--VII/outglv_0_{ion}"
            transitionsmember = f"outggf_Ln_V--VII/outggf_sorted_{ion}"

            levellines = ziplevels_in.read(levelsmember).decode().splitlines(keepends=True)
            energies1000percm = [float(line.split(",")[0]) for line in levellines]
            # the file has no energy order, so the cut is an energy limit and not a line count
            energylimit = sorted(energies1000percm)[levelcount - 1] * 1000
            keptlevels = [
                line for line, energy in zip(levellines, energies1000percm, strict=True) if energy * 1000 <= energylimit
            ]

            kepttransitions = []
            for line in ziptransitions_in.read(transitionsmember).decode().splitlines(keepends=True):
                wavelength_A, lowerenergy1000percm, _ = (float(x) for x in line.split(","))
                lowerenergypercm = lowerenergy1000percm * 1000
                upperenergypercm = lowerenergypercm + 1e8 / wavelength_A
                # the limit gets a margin, because the transition file rounds the energies
                if max(lowerenergypercm, upperenergypercm) <= energylimit + 1.0:
                    kepttransitions.append(line)

            print(f"{ion}: levels {len(keptlevels)}/{len(levellines)}, transitions {len(kepttransitions)}")
            ziplevels_out.writestr(levelsmember, "".join(keptlevels))
            ziptransitions_out.writestr(transitionsmember, "".join(kepttransitions))


if __name__ == "__main__":
    main()

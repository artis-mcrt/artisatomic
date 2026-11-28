#!/usr/bin/env python3
"""Write the reduced QUB Co II cross section files that the tests/adas checksum test reads.

The full files 1.gz to 8.gz in co_tyndall hold 200000 rows and 41 columns each: the photon energy
and one cross section for each of the 40 target states of photo_input. They are too large for the
repository. This script cuts a sample from them. It keeps each row whose index is a multiple of
ROW_STEP, and the last row. It also keeps the first row of each target column with a cross
section above zero, which is the threshold of that channel. The order of the rows does not change.
The script also copies photo_input.

Run it from the atomic-data-adas directory, with the full files present:

    uv run python make_co_test_sample.py

Regenerate tests/adas/checksums.txt afterwards, as tests/README.md describes.
"""

import gzip
from pathlib import Path

ROW_STEP = 2000
LEVEL_FILES = range(1, 9)


def main() -> None:
    """Write the sample of each cross section file, and photo_input, into co_tyndall_test_sample/."""
    basepath = Path(__file__).parent.resolve()
    sourcepath = basepath / "co_tyndall"
    outputpath = basepath / "co_tyndall_test_sample"
    for levelnumber in LEVEL_FILES:
        with gzip.open(sourcepath / f"{levelnumber}.gz", "rt", encoding="utf-8") as fin:
            lines = fin.read().splitlines()
        keep = set(range(0, len(lines), ROW_STEP)) | {len(lines) - 1}
        ncolumns = len(lines[0].split())
        opened = [False] * ncolumns
        for index, line in enumerate(lines):
            for column, field in enumerate(line.split()[1:], start=1):
                if not opened[column] and float(field) > 0.0:
                    opened[column] = True
                    keep.add(index)
            if all(opened[1:]):
                break
        # mtime=0 and no file name in the header, so a second run gives the same bytes
        with (
            (outputpath / f"{levelnumber}.gz").open("wb") as fout,
            gzip.GzipFile(filename="", mode="wb", fileobj=fout, mtime=0) as fgz,
        ):
            fgz.write(("\n".join(lines[index] for index in sorted(keep)) + "\n").encode("utf-8"))
        print(f"{levelnumber}.gz: {len(keep)} of {len(lines)} rows, {ncolumns} columns")
    text = (sourcepath / "photo_input").read_text(encoding="utf-8")
    with (
        (outputpath / "photo_input.gz").open("wb") as fout,
        gzip.GzipFile(filename="", mode="wb", fileobj=fout, mtime=0) as fgz,
    ):
        fgz.write(text.encode("utf-8"))
    print("photo_input.gz")


if __name__ == "__main__":
    main()

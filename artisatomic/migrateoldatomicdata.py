#!/usr/bin/env python3
import os
import shutil

from pathlib import Path


def write_sorted_transitionblock(fileout, transitionblock):
    transitionblock.sort()
    for (levelid_lower, levelid_upper, A, coll_str, forbidden) in transitionblock:
        fileout.write(f'{levelid_lower:4d} {levelid_upper:4d} {A} {coll_str:9.2e} {forbidden:d}\n')


def main():
    inputfolder = Path('/Volumes/GoogleDrive/Shared drives/ARTIS/artis-atomic_private/artis_classic_atomicdata')
    outputfolder = Path(__file__).parent.absolute() / 'artis_files'

    outputfolder.mkdir(parents=True, exist_ok=True)

    print(f"migrating old atomic data")
    print(f"  input folder:  {inputfolder}")
    print(f"  output folder: {outputfolder}")

    # same format is valid
    shutil.copy(inputfolder / 'ATOM.MODELS', outputfolder / 'adata.txt')
    print(f"copied ATOM.MODELS to adata.txt (old format is valid)")

    # mark all lines as permitted
    with open(inputfolder / 'LINELIST', 'r') as flinelist:
        path_transdata = outputfolder / 'transitiondata.txt'
        print(f"writing {path_transdata}")
        transitionblock = []
        firstblock = True
        with path_transdata.open('w') as ftransdata:
            while line := flinelist.readline():
                row = line.split()

                if len(row) == 3:  # ion header row
                    write_sorted_transitionblock(ftransdata, transitionblock)
                    transitionblock = []

                    atomic_number, ion_stage, ion_ntrans = [int(x) for x in row]
                    if not firstblock:
                        ftransdata.write('\n')
                    firstblock = False
                    ftransdata.write(f'{atomic_number:7d}{ion_stage:7d}{ion_ntrans:12d}')
                    transitionblock = []

                elif len(row) == 4:  # transition row
                    transid = int(row[0])
                    levelid_lower = int(row[1])
                    levelid_upper = int(row[2])
                    A = row[3]

                    coll_str = -1.  # no collision strength available
                    forbidden = 0  # 0 in the forbidden column means permitted

                    assert transid <= ion_ntrans
                    # assert levelid_lower < levelid_upper

                    if levelid_lower > levelid_upper:
                        # print(f"Flipping lower/upper {atomic_number=}, {ion_stage=}, {ion_ntrans=}: {line}")
                        levelid_lower, levelid_upper = levelid_upper, levelid_lower

                    transitionblock.append((levelid_lower, levelid_upper, A, coll_str, forbidden))

                elif not line.strip():  # blank line is ok
                    ftransdata.write(line)

                else:
                    print(f"Unknown line content: {line}")
                    assert False

            write_sorted_transitionblock(ftransdata, transitionblock)
            transitionblock = []

    # prepend new lines for table size and increment
    with open(inputfolder / 'PHIXS', 'r') as fadataold:
        path_phixsdata = outputfolder / 'phixsdata_v2.txt'
        print(f"writing {path_phixsdata}")
        with path_phixsdata.open('w') as fadatanew:
            fadatanew.write('100\n')
            fadatanew.write('0.1\n')
            while line := fadataold.readline():
                fadatanew.write(line)


if __name__ == "__main__":
    main()
import artisatomic


def read_storey_2016_upsilondata(flog) -> dict[tuple[int, int], float]:
    """Read the Storey et al. (2016) Co II effective collision strengths.

    Returns a dict of upsilon values keyed by a (lower, upper) pair of level ids, using the
    file's own numbering.
    """
    upsilondict = {}

    filename = "atomic-data-storey/storetetal2016-co-ii.txt"
    artisatomic.log_and_print(flog, f"Reading effective collision strengths from {artisatomic.path_for_log(filename)}")

    with open(filename) as fstoreydata:
        found_tablestart = False
        while True:
            line = fstoreydata.readline()
            if not line:
                break

            if found_tablestart:
                row = line.split()

                if len(row) <= 5:
                    break

                lower = int(row[0])
                upper = int(row[1])
                upsilon = float(row[11])
                upsilondict[(lower, upper)] = upsilon
            if line.startswith(
                "--	--	------	------	------	------	------	------	------	------	------	------	------	------	------"
            ):
                found_tablestart = True

    return upsilondict

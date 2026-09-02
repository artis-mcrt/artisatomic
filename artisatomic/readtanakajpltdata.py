"""Read levels and transitions from the Tanaka et al. Japan-Lithuania database."""

import re

import polars as pl

from artisatomic.base import add_handler_if_not_set
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import log_and_print
from artisatomic.base import PYDIR
from artisatomic.base import scan_file_lines

jpltpath = (PYDIR / ".." / "atomic-data-tanaka-jplt" / "data_v2.1").resolve()


def extend_ion_list(ion_handlers, maxionstage=None):
    """Add every ion with a Tanaka et al. Japan-Lithuania data file to ion_handlers."""
    tanakaions = sorted(
        [tuple(int(x) for x in f.parts[-1].split(".")[0].split("_")) for f in jpltpath.glob("*_*.txt*")]
    )
    if maxionstage is not None:
        tanakaions = [ion for ion in tanakaions if ion[1] <= maxionstage]

    for atomic_number, ion_stage in tanakaions:
        ion_handlers = add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "tanakajplt")

    return ion_handlers


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the Tanaka et al. Japan-Lithuania database.

    Levels and transitions are returned as DataFrames. The file numbers levels from one and
    quotes g_u * A rather than A, so ids are shifted to the zero-based convention used in
    memory and the rate is divided by the upper level's statistical weight. Self-transitions
    (equal upper and lower level) appear in some files and are dropped with a warning.
    """
    filename = f"{atomic_number}_{ion_stage}.txt"
    print(f"Reading Tanaka et al. Japan-Lithuania database for Z={atomic_number} ion_stage {ion_stage} from {filename}")
    # the header holds at most 6 lines before the "# Z ion_stage" line, and 5 lines after it.
    # A blank line reads as a null, which strip() cannot take.
    headerlines = [(line or "").strip() for line in scan_file_lines(jpltpath / filename).slice(0, 12).collect()["line"]]

    for linenumber, readlinein in enumerate(headerlines[:7]):
        if linenumber < 3:
            log_and_print(flog, readlinein)

        if readlinein == f"# {atomic_number} {ion_stage}":  # search for this line. Header info can be different
            break
    assert readlinein == f"# {atomic_number} {ion_stage}"

    levelcount, transitioncount = (int(x) for x in headerlines[linenumber + 1].removeprefix("# ").split())
    log_and_print(flog, f"levels: {levelcount}")
    log_and_print(flog, f"transitions: {transitioncount}")

    ionization_energy_in_ev = float(headerlines[linenumber + 3].removeprefix("# IP = "))
    log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")
    assert headerlines[linenumber + 4] == "# Energy levels"
    expected_column_headers = ["#", "num", "weight", "parity", "E(eV)", "configuration"]
    read_column_headers = headerlines[linenumber + 5].split()  # v2.1 has extra column
    assert all(item in read_column_headers for item in expected_column_headers)

    # the level section starts on the line after the column headers
    dflines = scan_file_lines(jpltpath / filename, skip_lines=linenumber + 6)

    # the transitions follow the levels, after a section title that some files leave out
    sectionheaders = dflines.slice(levelcount, 2).collect()["line"].to_list()
    line = sectionheaders[0].strip()
    assert line in {"# Transitions", "# num_u   num_l   wavelength(nm)     g_u*A      log(g_l*f)"}
    if line == "# Transitions":
        assert sectionheaders[1].strip() == "# num_u   num_l   wavelength(nm)     g_u*A      log(g_l*f)"
    transitionsectionstart = levelcount + (2 if line == "# Transitions" else 1)

    dflevels = (
        dflines.slice(0, levelcount)
        .select(
            levelid=pl.col("line").str.slice(0, 7).str.strip_chars(),
            g=pl.col("line").str.slice(7, 8).str.strip_chars(),
            parity=pl.col("line").str.slice(15, 4).str.strip_chars(),
            energy_ev=pl.col("line").str.slice(19, 15).str.strip_chars(),
            configuration=pl.col("line").str.slice(34).str.strip_chars(),
        )
        .select(
            energyabovegsinpercm=pl.col("energy_ev").cast(pl.Float64) / hc_in_ev_cm,
            parity=pl.when(pl.col("parity") == "odd").then(1).otherwise(0),
            g=pl.col("g").cast(pl.Float64),
            # Every expression in one select() reads the INPUT frame, so both columns below
            # are the file's own values, not the ones being computed alongside them: the name
            # gets the file's 1-based number rather than the zero-based levelid, and the
            # 'even'/'odd' text rather than the 0/1 parity. Both are wanted here — the name is
            # a human-readable comment in adata.txt.
            levelname=pl.format("{},{},{}", pl.col("levelid"), pl.col("parity"), pl.col("configuration")),
            levelid=pl.col("levelid").cast(pl.Int64) - 1,
        )
        # an empty line holds no level, and gives a null in every column. The count test below
        # rejects the file if such a line falls inside the section rather than after it.
        .filter(pl.col("levelid").is_not_null())
        .collect()
    )

    assert dflevels.height == levelcount

    dftransitions = (
        dflines.slice(transitionsectionstart)
        .select(
            # the file numbers levels from one; level ids are zero-based in memory
            lowerlevel=pl.col("line").str.slice(7, 8).str.strip_chars().cast(pl.Int64) - 1,
            upperlevel=pl.col("line").str.slice(0, 7).str.strip_chars().cast(pl.Int64) - 1,
            g_u_times_A=pl.col("line").str.slice(30, 13).str.strip_chars().cast(pl.Float64),
        )
        # the file may end with a blank line, which holds no transition
        .filter(pl.col("lowerlevel").is_not_null())
        .collect()
    )

    assert dftransitions.height == transitioncount

    # a level number outside the level section would vanish in the inner joins of
    # add_level_ids_forbidden() without a message, while adata.txt still counts the transition.
    # Not an assert: input validation must survive python -O.
    if not dftransitions.is_empty():
        levelid_min = int(dftransitions.select(pl.min_horizontal("lowerlevel", "upperlevel").min()).item())
        levelid_max = int(dftransitions.select(pl.max_horizontal("lowerlevel", "upperlevel").max()).item())
        if levelid_min < 0 or levelid_max >= levelcount:
            msg = (
                f"The JPLT transitions of Z={atomic_number} ion_stage {ion_stage} name level numbers"
                f" {levelid_min + 1} to {levelid_max + 1}, but the file has {levelcount} levels"
            )
            raise ValueError(msg)

    dftransitions = (
        dftransitions.join(
            dflevels.select(g_u=pl.col("g"), upperlevel=pl.col("levelid")),
            on="upperlevel",
            how="left",
            maintain_order="left",
        )
        .with_columns(A=pl.col("g_u_times_A") / pl.col("g_u"))
        .select(["lowerlevel", "upperlevel", "A"])
    )
    dftransitions_filtered = dftransitions.filter(pl.col("lowerlevel") != pl.col("upperlevel"))
    if dftransitions.height != dftransitions_filtered.height:
        log_and_print(flog, "WARNING: dropped rows where upper and lower levels are equal")
        dftransitions = dftransitions_filtered

    # count after the self-transition filter, so the counts in adata.txt agree with the
    # transitions present in transitiondata.txt
    transition_count_of_levelid: dict[int, int] = dict(
        pl.concat([dftransitions["lowerlevel"], dftransitions["upperlevel"]]).value_counts().iter_rows()
    )
    transition_count_of_level_name = {
        levelname: transition_count_of_levelid.get(levelid, 0)
        for levelid, levelname in dflevels.select("levelid", "levelname").iter_rows(named=False)
    }

    return ionization_energy_in_ev, dflevels, dftransitions, transition_count_of_level_name


def get_level_valence_n(levelname: str):
    """Principal quantum number of the valence electron, read from a JPLT level name.

    Kept separate from the other readers' versions: each data source names its levels
    differently, so a shared parser would have to guess which convention it is looking at.

    data_v2.1 mixes two conventions: the original relativistic one, "{  4s+ 2  4p- 1 }",
    where the valence orbital heads the last double-space-separated token, and the
    LS-coupled one of the 2024 Ge-sequence files, "3s(2).3p(6).3d(10).4s.4p(3)2D_3D",
    where it is the last dot-separated shell before the term label.
    """
    if "{" in levelname:
        n = int(levelname.rsplit("  ", maxsplit=1)[-1].split(" ", maxsplit=1)[0].rstrip("spdfg+-"))
    else:
        lastshell = levelname.rsplit(" ", maxsplit=1)[-1].rsplit(".", maxsplit=1)[-1].partition("_")[0]
        nmatch = re.match(r"\d+", lastshell)
        assert nmatch is not None, f"Could not parse valence n from level name: {levelname}"
        n = int(nmatch.group())
    assert n >= 0
    assert n < 20
    return n

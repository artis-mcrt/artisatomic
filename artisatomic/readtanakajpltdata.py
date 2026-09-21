"""Read levels and transitions from the Japan-Lithuania opacity database for kilonovae.

The database papers are Tanaka, M., Kato, D., Gaigalas, G., Kawaguchi, K. (2020), MNRAS, 496,
1369-1392, doi:10.1093/mnras/staa1576 (version 1) and Kato, D., Tanaka, M., Gaigalas, G.,
Kitovienė, L., Rynkun, P. (2024), MNRAS, 535, 2670-2686, doi:10.1093/mnras/stae2504 (version 2).
The second line of each data file names the paper of that ion.
"""

import re

import polars as pl

from artisatomic.base import add_handlers_if_not_set
from artisatomic.base import fixed_width_column
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import ion_filename_pattern
from artisatomic.base import ions_from_filenames
from artisatomic.base import log_and_print
from artisatomic.base import log_comment
from artisatomic.base import path_for_log
from artisatomic.base import PYDIR
from artisatomic.base import scan_file_lines

jpltpath = (PYDIR / ".." / "atomic-data-tanaka-jplt" / "data_v2.1").resolve()

# the "source:" line of the comment blocks in the output files (see Handler.description in iondata.py)
description = (
    "the Japan-Lithuania opacity database for kilonovae, of 26 <= Z <= 88. Tanaka, M., Kato, D., Gaigalas, G.,"
    " Kawaguchi, K. (2020), MNRAS, 496, 1369-1392, doi:10.1093/mnras/staa1576 (version 1), and Kato, D., Tanaka, M.,"
    " Gaigalas, G., Kitovienė, L., Rynkun, P. (2024), MNRAS, 535, 2670-2686, doi:10.1093/mnras/stae2504 (version 2)"
)

# the name of a data file, e.g. 26_2.txt or 26_2.txt.zst
jplt_filename_pattern = ion_filename_pattern(".txt")


def extend_ion_list(
    ion_handlers,
    *,
    minionstage: int | None = None,
    maxionstage: int | None = None,
    maxatomicnumber: int | None = None,
):
    """Add every ion with a Tanaka et al. Japan-Lithuania data file to ion_handlers."""
    # each name holds the atomic number and the ion stage, e.g. 26_2.txt
    tanakaions = ions_from_filenames(jpltpath.glob("*_*.txt*"), jplt_filename_pattern)

    return add_handlers_if_not_set(
        ion_handlers,
        tanakaions,
        "tanakajplt",
        minionstage=minionstage,
        maxionstage=maxionstage,
        maxatomicnumber=maxatomicnumber,
    )


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the Tanaka et al. Japan-Lithuania database.

    The function returns the levels and the transitions as DataFrames. The file numbers levels
    from one and quotes g_u * A rather than A. So the reader shifts the ids to the zero-based
    convention in memory. It also divides the rate by the statistical weight of the upper level.
    Self-transitions (equal upper and lower level) appear in some files. The reader drops them
    with a warning.
    """
    filename = f"{atomic_number}_{ion_stage}.txt"
    log_comment(
        flog,
        ("adata", "transitiondata"),
        f"Reading {path_for_log(jpltpath / filename, relative_to=jpltpath.parent.parent)}",
    )

    def require(condition: bool, message: str) -> None:
        # not an assert: input validation must survive python -O
        if not condition:
            msg = f"{filename}: {message}"
            raise ValueError(msg)

    # the header holds at most 6 lines before the "# Z ion_stage" line, and 5 lines after it.
    # A blank line reads as a null, which strip() cannot take.
    headerlines = [(line or "").strip() for line in scan_file_lines(jpltpath / filename).slice(0, 12).collect()["line"]]

    for linenumber, readlinein in enumerate(headerlines[:7]):
        if linenumber < 3:
            log_comment(flog, ("adata", "transitiondata"), readlinein)

        if readlinein == f"# {atomic_number} {ion_stage}":  # search for this line. Header info can be different
            break
    require(readlinein == f"# {atomic_number} {ion_stage}", f"no '# {atomic_number} {ion_stage}' line in the header")

    levelcount, transitioncount = (int(x) for x in headerlines[linenumber + 1].removeprefix("# ").split())
    log_and_print(flog, f"levels: {levelcount}")
    log_and_print(flog, f"transitions: {transitioncount}")

    ionization_energy_in_ev = float(headerlines[linenumber + 3].removeprefix("# IP = "))
    log_and_print(flog, f"ionisation energy: {ionization_energy_in_ev} eV")
    require(headerlines[linenumber + 4] == "# Energy levels", "no '# Energy levels' line after the ionisation energy")
    expected_column_headers = ["#", "num", "weight", "parity", "E(eV)", "configuration"]
    read_column_headers = headerlines[linenumber + 5].split()  # v2.1 has extra column
    require(
        all(item in read_column_headers for item in expected_column_headers),
        f"the level column headers {read_column_headers} lack one of {expected_column_headers}",
    )

    # the level section starts on the line after the column headers
    dflines = scan_file_lines(jpltpath / filename, skip_lines=linenumber + 6)

    # the transitions follow the levels, after a section title that some files leave out
    sectionheaders = dflines.slice(levelcount, 2).collect()["line"].to_list()
    line = sectionheaders[0].strip()
    transitionheader = "# num_u   num_l   wavelength(nm)     g_u*A      log(g_l*f)"
    require(line in {"# Transitions", transitionheader}, f"unexpected line after the level section: {line!r}")
    if line == "# Transitions":
        require(sectionheaders[1].strip() == transitionheader, "no transition column header after '# Transitions'")
    transitionsectionstart = levelcount + (2 if line == "# Transitions" else 1)

    dflevels = (
        dflines.slice(0, levelcount)
        # a line with no text holds no level: an empty line reads as a null and a line of spaces
        # as "". Neither has a character left. The count test below rejects the file if such a
        # line falls inside the section rather than after it.
        .filter(pl.col("line").str.strip_chars().str.len_chars() > 0)
        .select(
            levelid=fixed_width_column(0, 7),
            g=fixed_width_column(7, 8),
            parity=fixed_width_column(15, 4),
            energy_ev=fixed_width_column(19, 15),
            configuration=fixed_width_column(34),
        )
        .select(
            energyabovegsinpercm=pl.col("energy_ev").cast(pl.Float64) / hc_in_ev_cm,
            # odd -> 1, even -> 0, and anything else stays null. The check below rejects a null:
            # a silent 0 would give the Laporte rule a parity the file did not state
            parity=pl.when(pl.col("parity") == "odd").then(1).when(pl.col("parity") == "even").then(0),
            g=pl.col("g").cast(pl.Float64),
            # Every expression in one select() reads the INPUT frame. So both columns below hold
            # the file's own values, not the values that the same select() computes. The name
            # gets the file's 1-based number rather than the zero-based levelid, and the
            # 'even'/'odd' text rather than the 0/1 parity. Both are the desired values here: the
            # name is a human-readable comment in adata.txt.
            levelname=pl.format("{},{},{}", pl.col("levelid"), pl.col("parity"), pl.col("configuration")),
            levelid=pl.col("levelid").cast(pl.Int64) - 1,
        )
        .collect()
    )

    require(
        dflevels.height == levelcount, f"the header declares {levelcount} levels but the file has {dflevels.height}"
    )
    require(dflevels["parity"].null_count() == 0, "a level has a parity that is not 'odd' or 'even'")

    dftransitions = (
        dflines.slice(transitionsectionstart)
        # the file may end with a blank line, which holds no transition
        .filter(pl.col("line").str.strip_chars().str.len_chars() > 0)
        # split on white space, not at fixed positions. A wavelength of 1e9 nm or more is one
        # character wider than its field. It moves g_u*A one place right (Fe II 455 -> 454)
        .with_columns(fields=pl.col("line").str.extract_all(r"\S+"))
        .select(
            # the file numbers levels from one; level ids are zero-based in memory
            lowerlevel=pl.col("fields").list.get(1).cast(pl.Int64) - 1,
            upperlevel=pl.col("fields").list.get(0).cast(pl.Int64) - 1,
            g_u_times_A=pl.col("fields").list.get(3).cast(pl.Float64),
        )
        .collect()
    )

    require(
        dftransitions.height == transitioncount,
        f"the header declares {transitioncount} transitions but the file has {dftransitions.height}",
    )

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
        # the file names the upper level first, but transitiondata.txt has the lower id first. So
        # this select swaps a pair that the file lists in the reverse order
        .select(
            lowerlevel=pl.min_horizontal("lowerlevel", "upperlevel"),
            upperlevel=pl.max_horizontal("lowerlevel", "upperlevel"),
            A=pl.col("A"),
        )
    )
    dftransitions_filtered = dftransitions.filter(pl.col("lowerlevel") != pl.col("upperlevel"))
    if dftransitions.height != dftransitions_filtered.height:
        log_comment(flog, ("transitiondata",), "WARNING: dropped rows where upper and lower levels are equal")
        dftransitions = dftransitions_filtered

    return ionization_energy_in_ev, dflevels, dftransitions


def get_level_valence_n(levelname: str) -> int | None:
    """Principal quantum number of the valence electron, read from a JPLT level name.

    Returns None for a name that it cannot parse. The caller, match_hydrogenic_phixs(), then
    gives the level no estimate and writes a warning to the ion log.

    Kept separate from the other readers' versions. Each data source names its levels
    differently, so a shared parser would have to guess the convention of each name.

    data_v2.1 mixes two conventions. In the original relativistic one, "{  4s+ 2  4p- 1 }",
    the valence orbital heads the last shell token. In the LS-coupled one of the 2024 files,
    "4s2_4p6_4f2 4s(2).4p(6).4d(10)1S0_1S.4f(2)3H1_3H.5s(2).5p(6)_3H", the configuration column
    before the LS term gives it. The valence orbital is the last n-letter pair of that column. A
    token can glue two orbitals ("3d10_4s4p4"), so the last underscore-separated token is not
    always one orbital. The LS term is no guide: the lanthanide files write the closed 5s(2).5p(6)
    shells after the open 4f shell.
    """
    if "{" in levelname:
        # a two-digit n has one leading space ("6p+ 4 10s+ 1"), so read every shell by pattern
        shells = re.findall(r"(\d+)[a-z][+-]?\s+\d+", levelname)
        return int(shells[-1]) if shells else None

    configuration = levelname.split(",", maxsplit=2)[-1].split()
    if len(configuration) < 2:
        # the name gives the LS term alone. No name of data_v2.1 has that shape
        return None

    # the configuration column can glue two orbitals ("4p5s" is 4p 5s), so the last
    # n-letter pair is the valence orbital, not the last underscore-separated token
    shells = re.findall(r"(\d+)[a-z]", configuration[0])
    return int(shells[-1]) if shells else None

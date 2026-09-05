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

    The function returns the levels and the transitions as DataFrames. The file numbers levels
    from one and quotes g_u * A rather than A. So the reader shifts the ids to the zero-based
    convention in memory. It also divides the rate by the statistical weight of the upper level.
    Self-transitions (equal upper and lower level) appear in some files. The reader drops them
    with a warning.
    """
    filename = f"{atomic_number}_{ion_stage}.txt"
    print(f"Reading Tanaka et al. Japan-Lithuania database for Z={atomic_number} ion_stage {ion_stage} from {filename}")

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
            log_and_print(flog, readlinein)

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
            levelid=pl.col("line").str.slice(0, 7).str.strip_chars(),
            g=pl.col("line").str.slice(7, 8).str.strip_chars(),
            parity=pl.col("line").str.slice(15, 4).str.strip_chars(),
            energy_ev=pl.col("line").str.slice(19, 15).str.strip_chars(),
            configuration=pl.col("line").str.slice(34).str.strip_chars(),
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
        .select(
            # the file numbers levels from one; level ids are zero-based in memory
            lowerlevel=pl.col("line").str.slice(7, 8).str.strip_chars().cast(pl.Int64) - 1,
            upperlevel=pl.col("line").str.slice(0, 7).str.strip_chars().cast(pl.Int64) - 1,
            g_u_times_A=pl.col("line").str.slice(30, 13).str.strip_chars().cast(pl.Float64),
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
        log_and_print(flog, "WARNING: dropped rows where upper and lower levels are equal")
        dftransitions = dftransitions_filtered

    return ionization_energy_in_ev, dflevels, dftransitions


def get_level_valence_n(levelname: str) -> int | None:
    """Principal quantum number of the valence electron, read from a JPLT level name.

    Returns None when it cannot parse the name. The caller, match_hydrogenic_phixs(), then
    gives the level no estimate and writes a warning to the ion log.

    This parser stays separate from the versions of the other readers. Each data source names
    its levels differently, so a shared parser would have to guess the convention of the name.

    data_v2.1 mixes two conventions. In the original relativistic one, "{  4s+ 2  4p- 1 }",
    the valence orbital heads the last double-space-separated token. In the LS-coupled one of
    the 2024 Ge-sequence files, "3s(2).3p(6).3d(10).4s.4p(3)2D_3D", it is the last
    dot-separated shell before the term label.
    """
    if "{" in levelname:
        lastshell = levelname.rsplit("  ", maxsplit=1)[-1].split(" ", maxsplit=1)[0]
    else:
        lastshell = levelname.rsplit(" ", maxsplit=1)[-1].rsplit(".", maxsplit=1)[-1].partition("_")[0]
    nmatch = re.match(r"\d+", lastshell)
    return int(nmatch.group()) if nmatch is not None else None

"""Read levels and transitions from the Japan-Lithuania opacity database for kilonovae.

The database papers are Tanaka, M., Kato, D., Gaigalas, G., Kawaguchi, K. (2020), Systematic opacity
calculations for kilonovae, MNRAS, 496, 1369-1392, doi:10.1093/mnras/staa1576 (version 1) and
Kato, D., Tanaka, M., Gaigalas, G., Kitovienė, L., Rynkun, P. (2024), Systematic opacity calculations
for kilonovae - II. Improved atomic data for singly ionized lanthanides, MNRAS, 535, 2670-2686,
doi:10.1093/mnras/stae2504 (version 2).
The second line of each data file names the paper of that ion.
"""

import re

import polars as pl

from artisatomic.base import add_handlers_if_not_set
from artisatomic.base import elsymbols
from artisatomic.base import fixed_width_column
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import hc_in_ev_angstrom
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import ion_filename_pattern
from artisatomic.base import ions_from_filenames
from artisatomic.base import log_and_print
from artisatomic.base import log_comment
from artisatomic.base import path_in_data_folder
from artisatomic.base import PYDIR
from artisatomic.base import roman_numerals
from artisatomic.base import scan_file_lines

jpltfolder = PYDIR / ".." / "atomic-data-tanaka-jplt"
# not resolved below the data folder: a link there must not put its target into a comment line
jpltpath = jpltfolder.resolve() / "data_v2.1"

# the "source:" line of the comment blocks in the output files (see Handler.description in iondata.py)
description = (
    "the Japan-Lithuania opacity database for kilonovae. Tanaka, M., Kato, D., Gaigalas, G.,"
    " Kawaguchi, K. (2020), Systematic opacity calculations for kilonovae, MNRAS, 496, 1369-1392,"
    " doi:10.1093/mnras/staa1576 (version 1), and Kato, D., Tanaka, M., Gaigalas, G., Kitovienė, L., Rynkun, P. (2024),"
    " Systematic opacity calculations for kilonovae - II. Improved atomic data for singly ionized lanthanides,"
    " MNRAS, 535, 2670-2686, doi:10.1093/mnras/stae2504 (version 2)"
)

# the name of a data file, e.g. 26_2.txt or 26_2.txt.zst
jplt_filename_pattern = ion_filename_pattern(".txt")

# a header value that differs from the NIST value by more than this fraction gets a warning
ionization_energy_warning_fraction = 0.02


def extend_ion_list(
    ion_handlers,
):
    """Add every ion with a Tanaka et al. Japan-Lithuania data file to ion_handlers."""
    # each name holds the atomic number and the ion stage, e.g. 26_2.txt
    tanakaions = ions_from_filenames(jpltpath.glob("*_*.txt*"), jplt_filename_pattern)

    return add_handlers_if_not_set(
        ion_handlers,
        tanakaions,
        "tanakajplt",
    )


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the Tanaka et al. Japan-Lithuania database.

    The function returns the levels and the transitions as DataFrames. The file numbers levels
    from one and quotes g_u * A rather than A. So the reader shifts the ids to the zero-based
    convention in memory. It also divides the rate by the statistical weight of the upper level.
    Self-transitions (equal upper and lower level) appear in some files. The reader drops them
    with a warning.
    """
    # the plain name, also for a compressed file, so the output files do not depend on the compression
    filename = f"{atomic_number}_{ion_stage}.txt"
    log_comment(
        flog,
        ("adata", "transitiondata"),
        f"The levels and the transitions come from {path_in_data_folder(jpltpath / filename, jpltfolder)}.",
    )

    def require(condition: bool, message: str) -> None:
        # not an assert: input validation must survive python -O
        if not condition:
            msg = f"{filename}: {message}"
            raise ValueError(msg)

    # the header holds at most 6 lines before the "# Z ion_stage" line, and 5 lines after it.
    # A blank line reads as a null, which strip() cannot take.
    headerlines = [(line or "").strip() for line in scan_file_lines(jpltpath / filename).slice(0, 12).collect()["line"]]

    # search for the "# Z ion_stage" line. The lines before it can differ between the files.
    linenumber = next(
        (number for number, line in enumerate(headerlines[:7]) if line == f"# {atomic_number} {ion_stage}"), -1
    )
    require(linenumber >= 0, f"no '# {atomic_number} {ion_stage}' line in the header")

    # The lines before the "# Z ion_stage" line name the database version and the paper of the
    # ion. The last of them repeats the ion name. A paper line can continue on a line with no #,
    # so join such a line to the line before it. The log file and the comment blocks get each
    # joined line with no # of its own.
    # Two files write the paper line and the ion name on one line ("... 248 (2020) 17.# Eu II"), so
    # the reader cuts the ion name from the end of the last line, and drops that line if nothing is left.
    ionname = f"# {elsymbols[atomic_number]} {roman_numerals[ion_stage]}"
    beforeion = headerlines[:linenumber]
    if beforeion and beforeion[-1].endswith(ionname):
        beforeion[-1] = beforeion[-1].removesuffix(ionname).strip()
        if not beforeion[-1]:
            beforeion = beforeion[:-1]
    provenance: list[str] = []
    for readlinein in beforeion:
        if readlinein.startswith("#") or not provenance:
            provenance.append(readlinein.removeprefix("#").strip())
        else:
            provenance[-1] = f"{provenance[-1]} {readlinein}"
    for text in provenance:
        log_comment(flog, ("adata", "transitiondata"), text)

    levelcount, transitioncount = (int(x) for x in headerlines[linenumber + 1].removeprefix("# ").split())
    log_and_print(flog, f"The file header declares {levelcount} levels.")
    log_and_print(flog, f"The file header declares {transitioncount} transitions.")

    ionization_energy_in_ev = float(headerlines[linenumber + 3].removeprefix("# IP = "))
    log_and_print(flog, f"The file header gives an ionisation energy of {ionization_energy_in_ev} eV.")
    nist_ev = get_nist_ionization_energies_ev().get((atomic_number, ion_stage))
    if nist_ev is not None and abs(ionization_energy_in_ev / nist_ev - 1.0) > ionization_energy_warning_fraction:
        log_comment(
            flog,
            ("adata",),
            f"WARNING: the file gives an ionisation energy of {ionization_energy_in_ev} eV, and the NIST value is"
            f" {nist_ev} eV. The output keeps the value of the file.",
        )
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
    sectionheaders = [(text or "").strip() for text in dflines.slice(levelcount, 2).collect()["line"].to_list()]
    require(len(sectionheaders) >= 1, "the file ends after the level section")
    line = sectionheaders[0]
    transitionheader = "# num_u   num_l   wavelength(nm)     g_u*A      log(g_l*f)"
    require(line in {"# Transitions", transitionheader}, f"unexpected line after the level section: {line!r}")
    if line == "# Transitions":
        require(sectionheaders[1:] == [transitionheader], "no transition column header after '# Transitions'")
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
            # the weight column of the file is 2J + 1
            j=(pl.col("g").cast(pl.Float64) - 1) / 2,
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
            wavelength_nm=pl.col("fields").list.get(2).cast(pl.Float64),
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
        levelid_min, levelid_max = dftransitions.select(
            levelid_min=pl.min_horizontal("lowerlevel", "upperlevel").min(),
            levelid_max=pl.max_horizontal("lowerlevel", "upperlevel").max(),
        ).row(0)
        if levelid_min < 0 or levelid_max >= levelcount:
            msg = (
                f"The JPLT transitions of Z={atomic_number} ion_stage {ion_stage} name level numbers"
                f" {levelid_min + 1} to {levelid_max + 1}, but the file has {levelcount} levels"
            )
            raise ValueError(msg)

    check_wavelengths_against_levels(dftransitions, dflevels, filename, flog)

    # a row that repeats the levels, the wavelength and g_u*A of an earlier row is the same line two
    # times. ARTIS adds the A values of two rows of one level pair, so it would double the line.
    nrows = dftransitions.height
    dftransitions = dftransitions.unique(
        ["upperlevel", "lowerlevel", "wavelength_nm", "g_u_times_A"], keep="first", maintain_order=True
    )
    if dftransitions.height < nrows:
        log_comment(
            flog,
            ("transitiondata",),
            f"WARNING: The file gives {nrows - dftransitions.height:d} transition rows two times. The reader drops each"
            " repeat, because ARTIS adds the A values of two rows of one level pair.",
        )

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
        log_comment(
            flog,
            ("transitiondata",),
            "WARNING: The reader dropped the transitions whose upper level and lower level are the same.",
        )
        dftransitions = dftransitions_filtered

    log_rule_breaks(dftransitions, dflevels, flog)

    # The file lists electric dipole transitions only (readme_v2.1.txt), so no transition is
    # forbidden. The reader gives the flag itself, so the writer does not derive it from the level labels.
    return ionization_energy_in_ev, dflevels, dftransitions.with_columns(forbidden=pl.lit(value=False))


def log_rule_breaks(dftransitions: pl.DataFrame, dflevels: pl.DataFrame, flog) -> None:
    """Report the transitions whose levels break a selection rule of an electric dipole transition.

    The two rules are a change of the parity, and |delta J| <= 1 with no J = 0 -> J = 0. The
    transitions are electric dipole transitions, so a break shows a wrong level number in the
    file. 14 GRASP files of the lanthanides have such transitions (1.7% of the transitions of
    data_v2.1). A level less than 0.3 cm-1 from the named level fits 91% to 100% of them in each file.
    """
    if dftransitions.is_empty():
        return
    labels = dflevels.select("levelid", "parity", "j")
    nbreaks = (
        dftransitions.join(
            labels.rename({"levelid": "lowerlevel", "parity": "lower_parity", "j": "lower_j"}), on="lowerlevel"
        )
        .join(labels.rename({"levelid": "upperlevel", "parity": "upper_parity", "j": "upper_j"}), on="upperlevel")
        .filter(
            (pl.col("lower_parity") == pl.col("upper_parity"))
            | ((pl.col("upper_j") - pl.col("lower_j")).abs() > 1)
            | ((pl.col("upper_j") == 0) & (pl.col("lower_j") == 0))
        )
        .height
    )
    if nbreaks:
        log_comment(
            flog,
            ("transitiondata",),
            f"WARNING: {nbreaks} of {dftransitions.height} transitions break the parity rule or the delta J rule of an"
            " electric dipole transition. The file lists electric dipole transitions only, so the output writes each"
            " transition as permitted. Such a transition can belong to a different level of almost the same energy.",
        )


# a wavelength more than this fraction away from the energy difference of its two levels does not fit them
wavelength_tolerance_fraction = 0.01
# the absolute part of the tolerance [eV], for the energies that the file gives to 6 decimal places
wavelength_tolerance_ev = 3e-6
# a file with more rows than this fraction that do not fit their levels stops the read
wavelength_misfit_limit_fraction = 0.01


def check_wavelengths_against_levels(dftransitions: pl.DataFrame, dflevels: pl.DataFrame, filename: str, flog) -> None:
    """Compare the wavelength of each transition with the energy difference of the two levels that it names.

    The file gives the wavelength in its own column, so the comparison finds a transition list that
    does not use the numbers of the level list. The GRASP file of As II (grasp_v2.1/33_2.txt) has
    such a list: 825 of its 931 rows do not fit. Each other file of data_v2.1 has at most 0.01% of
    such rows, which are pairs of levels that are nearly at one energy.
    """
    if dftransitions.is_empty():
        return
    energies = dflevels.select("levelid", energy_ev=pl.col("energyabovegsinpercm") * hc_in_ev_cm)
    deltae = (
        dftransitions.join(energies.rename({"levelid": "upperlevel", "energy_ev": "e_upper"}), on="upperlevel")
        .join(energies.rename({"levelid": "lowerlevel", "energy_ev": "e_lower"}), on="lowerlevel")
        .select(
            deltae=(pl.col("e_upper") - pl.col("e_lower")).abs(),
            photon=hc_in_ev_angstrom / (10.0 * pl.col("wavelength_nm")),
        )
    )
    difference = (pl.col("photon") - pl.col("deltae")).abs()
    nmisfit = deltae.filter(
        (difference > wavelength_tolerance_fraction * pl.col("deltae")) & (difference > wavelength_tolerance_ev)
    ).height
    if nmisfit > wavelength_misfit_limit_fraction * dftransitions.height:
        msg = (
            f"{filename}: {nmisfit} of {dftransitions.height} transitions have a wavelength that does not fit the energy"
            " difference of the two levels that they name, so the transition list does not use the level numbers of"
            " the level list. The GRASP file 33_2.txt (As II) of grasp_v2.1 has this defect. setup_jplt_data.sh"
            " keeps the data_v2.0 file of that ion: delete data_v2.1/33_2.txt.zst and run the script again."
        )
        raise ValueError(msg)
    if nmisfit:
        log_comment(
            flog,
            ("transitiondata",),
            f"WARNING: {nmisfit} of {dftransitions.height} transitions have a wavelength more than"
            f" {wavelength_tolerance_fraction:.0%} away from the energy difference of their two levels.",
        )


def get_level_valence_n(levelname: str) -> int | None:
    """Principal quantum number of the valence electron, read from a JPLT level name.

    Returns None for a name that it cannot parse. The caller, match_hydrogenic_phixs(), then
    gives the level no estimate and writes a warning to the log file.

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

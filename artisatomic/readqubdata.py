#!/usr/bin/env python3
"""Read levels, transitions and collision strengths from the QUB (Queen's University Belfast) data."""

import string
import typing as t
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from artisatomic.base import add_handler_if_not_set
from artisatomic.base import compression_extensions
from artisatomic.base import empty_transitions_schema
from artisatomic.base import find_file_check_extension
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import log_and_print
from artisatomic.base import path_for_log
from artisatomic.base import PYDIR
from artisatomic.base import TESTMODE
from artisatomic.base import xopen_check_extension
from artisatomic.levelnames import get_config_parity
from artisatomic.levelnames import lchars
from artisatomic.phixs import reduce_phixs_tables

qubpath = (PYDIR / ".." / "atomic-data-qub").resolve()
tyndall_co3_path = (qubpath / ("co_tyndall_test_sample" if TESTMODE else "co_tyndall")).resolve()


class QUBTransitionRow(t.NamedTuple):
    """One QUB bound-bound transition.

    nameto is the UPPER level's name and namefrom the LOWER level's, matching the columns
    add_level_ids_forbidden() joins on to recover upperlevel and lowerlevel.
    """

    lowerlevel: int
    upperlevel: int
    A: float
    nameto: str
    namefrom: str
    lambdaangstrom: float


class QUBEnergyLevel(t.NamedTuple):
    """One energy level of a QUB calculation."""

    levelname: str
    qub_id: int
    twosplusone: int
    l: int
    j: float
    energyabovegsinpercm: float
    g: float
    parity: int | None  # None where the configuration determines no parity


def extend_ion_list(ion_handlers):
    """Add every ion with a QUB adf04 file to ion_handlers under the "qub" handler."""
    # the files ship compressed or plain, so match every form of the name that a reader accepts
    qubfiles = [f for ext in compression_extensions for f in qubpath.glob(f"*_*.adf04{ext}")]
    qubions = sorted({tuple(int(x) for x in f.name.split(".")[0].split("_")) for f in qubfiles})
    for atomic_number, ion_stage in qubions:
        ion_handlers = add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "qub")

    return ion_handlers


adf04_section_end = "-1"


def adf04_first_field(line: str) -> str:
    """Return the first whitespace-separated field of the line, or "" when it holds none."""
    fields = line.split(maxsplit=1)
    return fields[0] if fields else ""


def is_adf04_terminator(line: str) -> bool:
    """Report whether the line is the row that ends an adf04 section.

    The level block and the collision block each end with such a row. Writers pad it
    differently, so the test is on the first field and not on a fixed column.
    """
    return adf04_first_field(line) == adf04_section_end


def adf04_field(index: int) -> pl.Expr:
    """Return an expression for the whitespace-separated field at index of the "line" column."""
    return pl.col("line").str.extract(rf"^\s*(?:\S+\s+){{{index}}}(\S+)", 1)


def adf04_float(index: int) -> pl.Expr:
    """Return an expression for the field at index as a float.

    adf04 writes the exponent with no "E": 1.23-04 means 1.23e-04. The "E" goes in only after a
    digit or a point, so a leading sign and a field that already has an "E" stay as they are.
    """
    return adf04_field(index).str.replace_all(r"([0-9.])([-+])", "${1}E${2}").cast(pl.Float64, strict=False)


def adf04_level_layout(line: str) -> str:
    """Name the column layout of one adf04 level line, "standard" or "tyndall".

    The standard layout puts the (2S+1) group at column 28, after a 22-column configuration. The
    Tyndall Co III and Fe III files put it at column 24, after a 16-column configuration. The
    layout is a property of the file, not of the element, so it is read from the line.
    """
    if line[28:29] == "(" and line[30:31] == ")":
        return "standard"
    if line[24:25] == "(" and line[26:27] == ")":
        return "tyndall"
    msg = f"adf04 level line has no (2S+1) group at column 24 or 28: {line.rstrip()!r}"
    raise ValueError(msg)


def read_adf04(
    filepath: str | Path, atomic_number: int, ion_stage: int, flog
) -> tuple[float, list[QUBEnergyLevel], dict[tuple[int, int], float], pl.DataFrame]:
    """Read levels and effective collision strengths from an ADAS adf04 file.

    Returns the ionization energy in eV, the levels, a dict of upsilon values keyed by a
    (lower, upper) pair of zero-based level ids, and the parsed collision rows. The caller takes
    the A-values from that frame, which saves a second read and a second parse of the file. The
    file numbers levels from one, and the rest of the code looks up id n at list index n - 1, so
    the level ids are validated as contiguous and 1-based. Collision strengths are taken at one
    temperature, chosen per element.
    """
    energylevels: list[QUBEnergyLevel] = []
    upsilondict: dict[tuple[int, int], float] = {}
    ionization_energy_ev = 0.0
    log_and_print(flog, f"Reading {path_for_log(filepath)}")
    with xopen_check_extension(filepath) as fleveltrans:
        line = fleveltrans.readline()
        row = line.split()
        ionization_energy_ev = float(row[4].split("(")[0]) * hc_in_ev_cm
        # The calculation-details section at the end of each file has no standard format. The
        # reader skips it when it follows the data blocks. A note at any other position is not
        # handled.
        atomic_group_note = False
        while True:
            line = fleveltrans.readline()
            if not line or is_adf04_terminator(line):
                break
            if line.startswith("C-"):
                atomic_group_note = not atomic_group_note
                continue
            if atomic_group_note:
                continue

            if adf04_level_layout(line) == "tyndall":
                config = line[5:21].strip()
                energylevel = QUBEnergyLevel(
                    config,
                    int(line[:5]),
                    int(line[25:26]),
                    int(line[27:28]),
                    float(line[29:33]),
                    float(line[34:55]),
                    0.0,
                    0,
                )

            else:
                # the whole configuration, with any parent term and any orbital after it: a name
                # cut at the parent term lost the orbital that followed it
                config = line[5:27].strip()
                energylevel = QUBEnergyLevel(
                    config,
                    int(line[:5]),
                    int(line[29:30]),
                    int(line[31:32], 16),
                    float(line[33:37]),
                    float(line[39:59]),
                    0.0,
                    0,
                )
            config_for_parity = config
            # hasterm=False: an adf04 name is all configuration, because the file keeps 2S+1 and
            # L in their own columns (read just above). Stripping a term off the end would lose
            # the last orbital of '3S2 3P6 3D5 4P1', and would read the bare '5s2' as a term
            # rather than an orbital, which is how every level of some files came out even.
            parity = get_config_parity(config_for_parity, hasterm=False)

            levelname = energylevel.levelname + "_{:d}{:}{:}[{:d}/2]_id={:}".format(
                energylevel.twosplusone,
                lchars[energylevel.l],
                # the name keeps the old even/odd letter where the parity is unknown, so that
                # adata.txt does not depend on a distinction the parity column now makes
                ["e", "o"][parity if parity is not None else 0],
                int(2 * energylevel.j),
                energylevel.qub_id,
            )

            g = 2 * energylevel.j + 1
            energylevel = energylevel._replace(g=g, parity=parity, levelname=levelname)
            energylevels.append(energylevel)

            # the transition and upsilon tables use these 1-based ids and the rest of the code
            # looks up id n at index n - 1, so a non-contiguous file would misattach every
            # transition. Not an assert: input validation must survive python -O.
            if energylevel.qub_id != len(energylevels):
                msg = (
                    f"adf04 level id {energylevel.qub_id} found at position {len(energylevels)} in {filepath}."
                    " Level ids must be contiguous and start at 1."
                )
                raise ValueError(msg)

        upsilonheader = fleveltrans.readline().split()
        temperatures = upsilonheader[2:]

        # ADAS writes auxiliary rows with a process code in the first field: R for recombination,
        # S and I for ionization, P for proton impact. Only a row that starts with a level id is
        # a collision strength. A blank line is not a bad row, so it is not counted.
        collision_lines: list[str] = []
        skipped_rows = 0
        for line in fleveltrans:
            firstfield = adf04_first_field(line)
            if firstfield == adf04_section_end:
                break
            if not firstfield:
                continue
            if not firstfield.isdigit():
                skipped_rows += 1
                continue
            collision_lines.append(line)

        # Co, W I and II rates are calculated at different temperatures
        # Should be handled in a less approximate way in the future
        if atomic_number == 27:
            strtemperature = "5.01+03"
        elif atomic_number == 60 and ion_stage == 2:
            strtemperature = "4.50+03"
        elif atomic_number == 74 and ion_stage == 1:
            strtemperature = "5.80+03"
        elif atomic_number == 74 and ion_stage == 2:
            strtemperature = "4.00+03"
        else:
            strtemperature = "5.00+03"

        if strtemperature not in temperatures:
            msg = (
                f"{filepath} holds no collision strengths at {strtemperature} K."
                f" The file's temperatures are: {' '.join(temperatures)}"
            )
            raise ValueError(msg)

        # each collision row is upper, lower, A-value, one upsilon for each temperature, then
        # the infinite-energy (Born) limit. Cutting out the wanted fields costs about a third of
        # the memory of splitting every line into all of its columns.
        upsilonindex = 3 + temperatures.index(strtemperature)
        collisiondf = pl.DataFrame({"line": collision_lines}, schema={"line": pl.String}).select(
            adf04_field(0).cast(pl.Int64, strict=False).alias("upper"),
            adf04_field(1).cast(pl.Int64, strict=False).alias("lower"),
            adf04_float(2).alias("avalue"),
            adf04_float(upsilonindex).alias("upsilon"),
        )

        # a row that is too short, or that holds a value this cannot read, gives a null
        goodrows = collisiondf.drop_nulls(subset=["lower", "upper", "upsilon"])
        unreadable_rows = collisiondf.height - goodrows.height

        for lower, upper, upsilon in goodrows.select("lower", "upper", "upsilon").iter_rows():
            lower, upper = min(lower, upper), max(lower, upper)
            # a raise rather than an assert: this validates an input file, and the check
            # must survive python -O. Equal ids would store a self-transition.
            if not 1 <= lower < upper <= len(energylevels):
                msg = (
                    f"collision strength level ids {lower}, {upper} in {filepath} are outside"
                    f" the file's {len(energylevels)} levels"
                )
                raise ValueError(msg)

            # the file numbers levels from one; level ids are zero-based in memory. The log
            # messages keep the file's numbering, since they are about the file's contents.
            levelidpair = (lower - 1, upper - 1)
            if levelidpair not in upsilondict:
                upsilondict[levelidpair] = upsilon
            else:
                log_and_print(
                    flog,
                    f"Duplicate upsilon value for transition {lower:d} to {upper:d} keeping"
                    f" {upsilondict[levelidpair]:5.2e} instead of using {upsilon:5.2e}",
                )

    log_and_print(flog, f"Read {len(energylevels):d} levels")
    log_and_print(flog, f"Read {len(upsilondict):d} effective collision strengths")
    if skipped_rows:
        log_and_print(flog, f"Skipped rows without a numeric level id: {skipped_rows:d}")
    if unreadable_rows:
        log_and_print(flog, f"Skipped collision rows that could not be read: {unreadable_rows:d}")

    return ionization_energy_ev, energylevels, upsilondict, collisiondf


def append_qub_transition(qub_energylevels, qub_transitions, id_lower, id_upper, A, filepath) -> None:
    """Validate one radiative transition row and append it to the transition list.

    The ids are the file's 1-based level ids.
    """
    # a raise rather than an assert: this validates an input file. A non-positive
    # id would wrap to the wrong level via negative indexing, and one past the end
    # would raise a bare IndexError naming neither the file nor the transition.
    if not 1 <= id_lower <= len(qub_energylevels) or not 1 <= id_upper <= len(qub_energylevels):
        msg = (
            f"transition level ids {id_lower}, {id_upper} in {filepath} are outside"
            f" the file's {len(qub_energylevels)} levels"
        )
        raise ValueError(msg)
    # the file numbers levels from one; level ids are zero-based in memory
    id_lower -= 1
    id_upper -= 1
    level_upper = qub_energylevels[id_upper]
    level_lower = qub_energylevels[id_lower]
    levelname_upper = level_upper.levelname
    levelname_lower = level_lower.levelname
    delta_percm = level_upper.energyabovegsinpercm - level_lower.energyabovegsinpercm
    lamdaangstrom = 1.0e8 / delta_percm if delta_percm != 0.0 else -1.0
    transition = QUBTransitionRow(
        lowerlevel=id_lower,
        upperlevel=id_upper,
        A=A,
        nameto=levelname_upper,
        namefrom=levelname_lower,
        lambdaangstrom=lamdaangstrom,
    )
    qub_transitions.append(transition)


# the ion stages that the QUB Co data covers: the Co III adf04 files and the single-level
# Co IV. For the other stages of a "qub_cobalt" ion, iondata.read_ion_data() falls back to
# the CMFGEN reader. read_qub_levels_and_transitions() below has one branch for each stage
# in this set, so a new stage needs an entry here and a branch there.
qub_cobalt_stages: frozenset[int] = frozenset({3, 4})

# the ions whose photoionisation cross sections the QUB Co data covers, one branch each in
# read_qub_photoionizations(). iondata.read_ion_data() takes the CMFGEN phot files for every
# other stage of a "qub_cobalt" ion.
qub_phixs_ions: frozenset[tuple[int, int]] = frozenset({(27, 2), (27, 3)})


def read_qub_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the QUB calculations.

    Covers both the newer per-ion adf04 calculations and the older Co II/III/IV data sets, which
    have their own file layouts. Also returns the effective collision strengths, so this reader
    supplies an upsilondict where most others leave it to be filled in elsewhere.
    """
    # the plain name, not the found path: read_adf04() logs the name it is given, and the
    # tested log files carry the plain name whether or not the file on disk is compressed
    atom_filepath = qubpath / f"{atomic_number}_{ion_stage}.adf04"

    if (atomic_number == 27) and (ion_stage == 3):
        # Co III takes its A-values from a separate file, so the collision rows are not needed
        ionization_energy_ev, qub_energylevels, upsilondict, _ = read_adf04(
            tyndall_co3_path / "adf04_v1", atomic_number, ion_stage, flog
        )

        qub_transitions: list[QUBTransitionRow] | pl.DataFrame = []
        transitionfile = tyndall_co3_path / "adf04rad_v1"
        with xopen_check_extension(transitionfile) as ftrans:
            for line in ftrans:
                row = line.split()
                id_upper = int(row[0])
                id_lower = int(row[1])
                A = float(row[2])
                if A > 2e-30:
                    append_qub_transition(
                        qub_energylevels,
                        qub_transitions,
                        id_lower,
                        id_upper,
                        A,
                        transitionfile,
                    )

    elif (atomic_number == 27) and (ion_stage == 4):
        # one level, the 3d6 5D4 ground state: g = 2J + 1 = 9, as the CMFGEN Co IV level list gives it
        qub_energylevels: list[QUBEnergyLevel] = [QUBEnergyLevel("groundstate", 1, 0, 0, 0, 0.0, 9, 0)]
        qub_transitions = pl.DataFrame(schema=empty_transitions_schema)
        upsilondict: dict[tuple[int, int], float] = {}
        ionization_energy_ev = 54.9000015

    elif find_file_check_extension(atom_filepath) is not None:
        # the same test that extend_ion_list() makes when it discovers these ions by globbing
        # qubpath, so an adf04 file that discovery registers is one that this reader accepts
        ionization_energy_ev, qub_energylevels, upsilondict, collisiondf = read_adf04(
            atom_filepath, atomic_number, ion_stage, flog
        )

        qub_transitions: list[QUBTransitionRow] | pl.DataFrame = []

        # W II file has the first two columns swapped around from the standard order
        uppercolumn, lowercolumn = ("lower", "upper") if (atomic_number, ion_stage) == (74, 2) else ("upper", "lower")
        # a radiative transition is a collision row with both level ids and an A-value. The
        # rows were selected by the width of the line before, which dropped a row that was one
        # character shorter than the widest without a count.
        transitiondf = collisiondf.filter(
            pl.col("upper").is_not_null(), pl.col("lower").is_not_null(), pl.col("avalue") > 2e-30
        )

        for id_upper, id_lower, A in transitiondf.select(uppercolumn, lowercolumn, "avalue").iter_rows():
            append_qub_transition(
                qub_energylevels,
                qub_transitions,
                id_lower,
                id_upper,
                A,
                atom_filepath,
            )

    else:
        msg = f"No QUB data available for Z={atomic_number} ion_stage {ion_stage} (no file {atom_filepath})"
        raise ValueError(msg)

    log_and_print(flog, f"Read {len(qub_transitions):d} transitions")

    return ionization_energy_ev, qub_energylevels, qub_transitions, upsilondict


def read_qub_photoionizations(
    atomic_number, ion_stage, levelcount: int, args, flog
) -> tuple[npt.NDArray[np.float64], list[list[tuple[int, float]]], npt.NDArray[np.float64]]:
    """Read QUB photoionization cross sections for one ion, downsampled onto the output grid.

    Returns the cross sections, the upper-ion target fractions per level, and the threshold
    energies, all indexed by zero-based level id. Levels with no data keep an empty target list,
    which is how write_phixs_data() knows to skip them.

    An ion that this function has no data for gets the empty arrays, not zero-filled ones. The
    caller reads an empty cross-section array as "no data", and then applies the hydrogenic
    estimate. A zero-filled array would pass as data and leave the ion with no cross sections.
    """
    photoionization_crosssections = np.zeros((levelcount, args.nphixspoints))
    # levels stay empty (write_phixs_data() skips them) unless real data is assigned below
    photoionization_targetfractions: list[list[tuple[int, float]]] = [[] for _ in range(levelcount)]
    photoionization_thresholds_ev = np.full(levelcount, np.nan)

    if atomic_number == 27 and ion_stage == 2:
        for lowerlevelid in range(8):
            # the cross-section files are named after the level's number in the source data,
            # which counts from one
            filename = tyndall_co3_path / f"{lowerlevelid + 1:d}.gz"
            log_and_print(flog, f"Reading {path_for_log(filename)}")
            ntargets = 4  # just the 4Fe ground quartet (the file has 40 target columns)
            # One space separates the columns, and every field is a number, so a null means the
            # columns are not where the read expects them. Reading the first five columns of the
            # 41 costs a third of the time that cutting every line into its parts does.
            columnnames = ["energy", *(f"target{column}" for column in range(1, ntargets + 1))]
            photdata = (
                pl.scan_csv(filename, separator=" ", has_header=False, infer_schema_length=0)
                .select(pl.nth(column).cast(pl.Float64).alias(name) for column, name in enumerate(columnnames))
                .collect()
            )
            if photdata.null_count().sum_horizontal().item() > 0:
                msg = f"Columns of {filename} are not where they are expected: a value is missing."
                raise ValueError(msg)
            phixstables = {}

            # column n of the file holds the cross section to the upper ion's level id n - 1
            for targetcolumn in range(1, ntargets + 1):
                targetname = f"target{targetcolumn}"
                phixstable = photdata.filter(pl.col(targetname) > 0.0).select("energy", targetname).to_numpy()
                if len(phixstable) == 0:
                    # nothing positive in this column, so there is no table to downsample. Skipping
                    # here leaves the target out of the fractions below, which is what a zero cross
                    # section means anyway; reduce_phixs_tables() would index off an empty array.
                    log_and_print(
                        flog,
                        f"WARNING: level {lowerlevelid} has no positive cross section to target"
                        f" {targetcolumn - 1}, so that target is dropped",
                    )
                    continue
                phixstables[targetcolumn] = phixstable

            reduced_phixs_dict = reduce_phixs_tables(
                phixstables, args.optimaltemperature, args.nphixspoints, args.phixsnuincrement
            )
            target_scalefactors = np.zeros(ntargets)
            targetcolumn_withmaxfraction = 1
            max_scalefactor = 0.0
            for targetcolumn, reduced_phixstable in reduced_phixs_dict.items():
                # take the ratio of cross sections at the threshold energies
                scalefactor = reduced_phixstable[0]
                target_scalefactors[targetcolumn - 1] = scalefactor
                if scalefactor > max_scalefactor:
                    targetcolumn_withmaxfraction = targetcolumn
                    max_scalefactor = scalefactor

            scalefactorsum = sum(target_scalefactors)
            if scalefactorsum <= 0.0:
                # nothing was assigned for this level, so write_phixs_data() will skip it
                log_and_print(
                    flog, f"WARNING: all photoionisation targets for level {lowerlevelid} have zero cross section"
                )
                continue
            target_scalefactors = [x if (x / scalefactorsum > 0.02) else 0.0 for x in target_scalefactors]
            scalefactorsum = sum(target_scalefactors)

            # NaN, the arrays' initial value, says: the threshold energy comes from the level
            # energies, not from the first energy point of the cross-section table
            photoionization_thresholds_ev[lowerlevelid] = np.nan
            for upperlevelid, target_scalefactor in enumerate(target_scalefactors):
                target_fraction = target_scalefactor / scalefactorsum
                if target_fraction > 0.001:
                    photoionization_targetfractions[lowerlevelid].append((upperlevelid, target_fraction))

            max_fraction = max_scalefactor / scalefactorsum
            photoionization_crosssections[lowerlevelid] = (
                reduced_phixs_dict[targetcolumn_withmaxfraction] / max_fraction
            )

    elif atomic_number == 27 and ion_stage == 3:
        # photoionize to a single level ion

        phixsvalues_const = [
            9.3380692,
            7.015829602,
            5.403975231,
            4.250372872,
            3.403086443,
            2.766835319,
            2.279802051,
            1.900685772,
            1.601177846,
            1.361433037,
            1.16725865,
            1.008321909,
            0.8769787,
            0.76749151,
            0.675496904,
            0.597636429,
            0.531296609,
            0.474423066,
            0.425385805,
            0.382880364,
            0.345854415,
            0.313452694,
            0.284975256,
            0.259845541,
            0.237585722,
            0.217797532,
            0.200147231,
            0.184353724,
            0.17017913,
            0.157421217,
            0.145907331,
            0.135489462,
            0.126040239,
            0.117449648,
            0.109622338,
            0.102475382,
            0.095936439,
            0.089942202,
            0.084437113,
            0.079372279,
            0.074704554,
            0.070395769,
            0.066412076,
            0.062723384,
            0.059302883,
            0.056126637,
            0.053173226,
            0.050423446,
            0.047860046,
            0.045467498,
            0.043231802,
            0.041140312,
            0.039181587,
            0.037345256,
            0.035621907,
            0.034002983,
            0.032480693,
            0.031047932,
            0.029698215,
            0.028425611,
            0.027224692,
            0.026090478,
            0.025018404,
            0.02400427,
            0.023044216,
            0.022134683,
            0.021272391,
            0.020454314,
            0.019677652,
            0.018939819,
            0.018238416,
            0.017571225,
            0.016936183,
            0.016331377,
            0.01575503,
            0.015205486,
            0.014681206,
            0.014180754,
            0.013702792,
            0.013246071,
            0.012809423,
            0.012391758,
            0.011992055,
            0.011609359,
            0.011242775,
            0.010891464,
            0.010554639,
            0.010231561,
            0.009921535,
            0.009623909,
            0.009338069,
            0.009063438,
            0.008799471,
            0.008545656,
            0.00830151,
            0.008066575,
            0.007840423,
            0.007622646,
            0.00741286,
            0.007210703,
        ]

        if abs(args.nphixspoints - 100) < 0.5 and abs(args.phixsnuincrement - 0.1) < 0.001:
            phixsvalues = np.array(phixsvalues_const)
        else:
            # the stop of 10.95 makes arange produce all 100 grid points from 1.0 to 10.9; a stop
            # of 10.9 produced 99 and the strict flag then dropped the table's last point
            dict_phixstable = {"gs": np.array(list(zip(np.arange(1.0, 10.95, 0.1), phixsvalues_const, strict=True)))}
            phixsvalues = reduce_phixs_tables(
                dict_phixstable, args.optimaltemperature, args.nphixspoints, args.phixsnuincrement
            )["gs"]

        # unlike the Co II branch above, every level deliberately gets a phixs entry: the ground
        # quartet gets the tabulated cross section and higher levels an explicit all-zero table
        for levelid in range(levelcount):
            photoionization_thresholds_ev[levelid] = np.nan
            photoionization_targetfractions[levelid] = [(0, 1.0)]  # the upper ion's ground state
            if levelid < 4:
                photoionization_crosssections[levelid] = phixsvalues

    else:
        log_and_print(flog, f"WARNING: no QUB photoionization data for Z={atomic_number} ion_stage {ion_stage}")
        return np.empty((0, args.nphixspoints)), [], np.empty(0)

    return photoionization_crosssections, photoionization_targetfractions, photoionization_thresholds_ev


def get_level_valence_n(levelname: str) -> int | None:
    """Principal quantum number of the valence electron, read from a QUB level name.

    Returns None when the name cannot be parsed. The caller, match_hydrogenic_phixs(), then
    gives the level no estimate and writes a warning to the ion log.

    Kept separate from the other readers' versions: each data source names its levels
    differently, so a shared parser would have to guess which convention it is looking at.
    """
    namesplit = levelname.split("_")
    # lower(): adf04 writes some orbitals in upper case ('3S2 3P6 3D5 4P1'), and the orbital
    # tests below compare against the lower-case orbital letters only
    part = namesplit[0].strip().lower()
    # `part` is empty for a name that starts with '_', and stripping a leading parent term below
    # can empty it too, so re-test it before every part[-1] rather than raising IndexError
    if len(namesplit) < 2 or not part:
        return None

    if part[-1] == ")" and "(" in part:
        part = part[: part.rfind("(")]

    if not part:
        return None

    if part[-1] not in lchars.lower():
        # the last character must be the number of electrons in the orbital: remove it
        if not part[-1].isdigit():
            return None
        part = part.rstrip(string.digits)
    part = part.strip(lchars.lower())

    # inefficient way to find the last number in a string
    for i in range(len(part)):
        try:
            n = int(part[i:])
        except ValueError:
            continue
        else:
            # a lower-case orbital letter before the number means that the number is an
            # electron count of the previous orbital followed by n, e.g. the '24' in '3d24s'
            # is two electrons and n=4. The same rule as readkuruczdata: a two-digit run that
            # ends in 0 is a two-digit n ('5s10d'), and a three-digit run is a two-digit count
            # and a one-digit n ('4f145d') unless that n would be 0 ('5s210d').
            if i > 0 and part[i - 1] in lchars.lower():
                digits = part[i:]
                if len(digits) == 2 and digits[1] != "0":
                    digits = digits[1:]
                elif len(digits) == 3:
                    digits = digits[2:] if digits[2] != "0" else digits[1:]
                elif len(digits) > 3:
                    return None
                n = int(digits)
            return n

    return None

#!/usr/bin/env python3
"""Read levels, transitions and collision strengths from the QUB (Queen's University Belfast) data."""

import os
import string
import typing as t
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

import artisatomic
from artisatomic.base import hc_in_ev_cm
from artisatomic.levelnames import lchars

tyndall_co3_path = (
    Path(__file__).parent.resolve()
    / ".."
    / "atomic-data-qub"
    / ("co_tyndall_test_sample" if os.environ.get("ARTISATOMIC_TESTMODE") == "1" else "co_tyndall")
).resolve()


class QUBTransitionRow(t.NamedTuple):
    """One QUB bound-bound transition."""

    lowerlevel: int
    upperlevel: int
    A: float
    nameto: str
    namefrom: str
    lambdaangstrom: float
    coll_str: float


class QUBEnergyLevel(t.NamedTuple):
    """One energy level of a QUB calculation."""

    levelname: str
    qub_id: int
    twosplusone: int
    l: int
    j: float
    energyabovegsinpercm: float
    g: float
    parity: int


qubpath = (Path(__file__).parent.resolve() / ".." / "atomic-data-qub").resolve()


def check_forbidden(levela: QUBEnergyLevel, levelb: QUBEnergyLevel) -> bool:
    """Whether a transition between two levels is forbidden, i.e. they have the same parity."""
    return levela.parity == levelb.parity


def extend_ion_list(ion_handlers):
    """Add every ion with a QUB adf04 file to ion_handlers under the "qub_data" handler."""
    qubions = sorted([tuple(int(x) for x in f.parts[-1].split(".")[0].split("_")) for f in qubpath.glob("*_*.adf04")])
    for atomic_number, ion_stage in qubions:
        ion_handlers = artisatomic.add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "qub_data")

    return ion_handlers


def read_adf04(
    filepath: str | Path, atomic_number: int, ion_stage: int, flog
) -> tuple[float, list[QUBEnergyLevel], dict[tuple[int, int], float]]:
    """Read levels and effective collision strengths from an ADAS adf04 file.

    Returns the ionization energy in eV, the levels, and a dict of upsilon values keyed by a
    (lower, upper) pair of zero-based level ids. The file numbers levels from one, and the rest
    of the code looks up id n at list index n - 1, so the level ids are validated as contiguous
    and 1-based. Collision strengths are taken at one temperature, chosen per element.
    """
    energylevels: list[QUBEnergyLevel] = []
    upsilondict: dict[tuple[int, int], float] = {}
    ionization_energy_ev = 0.0
    artisatomic.log_and_print(flog, f"Reading {artisatomic.path_for_log(filepath)}")
    with artisatomic.xopen_check_extension(filepath) as fleveltrans:
        line = fleveltrans.readline()
        row = line.split()
        ionization_energy_ev = float(row[4].split("(")[0]) * hc_in_ev_cm
        # Formatting of the calculation details section at the bottom of each file is not standardised
        # This ignores it, provided it's put in the correct place. Needs made more robust later.
        # Currently if it's somewhere random: ¯\_(ツ)_/¯
        atomic_group_note = False
        while True:
            line = fleveltrans.readline()
            if not line or line.startswith("   -1"):
                break
            if line.startswith("C-"):
                atomic_group_note = not atomic_group_note
                continue
            if atomic_group_note:
                continue

            if atomic_number == 27:
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
                config_full = line[5:27].strip()
                # Accounting for files which have multiple sets of brackets in this string
                # Some files include parent terms
                config_parts = config_full.split(")")
                relevant_config = config_parts[-2].strip() + ")" if len(config_parts) > 1 else config_full
                config = relevant_config
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
            parity = artisatomic.get_parity_from_config(config)

            levelname = energylevel.levelname + "_{:d}{:}{:}[{:d}/2]_id={:}".format(
                energylevel.twosplusone,
                lchars[energylevel.l],
                ["e", "o"][parity],
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
        list_tempheaders = [f"upsT={x:}" for x in upsilonheader[2:]]
        # each collision row is upper, lower, A-value, one upsilon per temperature, then the
        # infinite-energy (Born) limit. Name that last column so pandas does not drop one to fit.
        list_headers = ["upper", "lower", "avalue", *list_tempheaders, "born_limit"]
        qubupsilondf_alltemps = pd.read_csv(
            fleveltrans,
            index_col=False,
            sep=r"\s+",
            comment="C",
            names=list_headers,
            dtype={"lower": int, "upper": int},
            on_bad_lines="skip",
            skip_blank_lines=True,
            keep_default_na=False,
        )
        qubupsilondf_alltemps = qubupsilondf_alltemps.query("upper!=-1")
        for _, row in qubupsilondf_alltemps.iterrows():
            lower = row["lower"]
            upper = row["upper"]
            lower, upper = min(lower, upper), max(lower, upper)
            assert upper > lower

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
            strupsilon = str(row[f"upsT={strtemperature}"])
            upsilon = float(strupsilon.replace("-", "E-").replace("+", "E+"))
            # the file numbers levels from one; level ids are zero-based in memory. The log
            # messages keep the file's numbering, since they are about the file's contents.
            levelidpair = (lower - 1, upper - 1)
            if levelidpair not in upsilondict:
                upsilondict[levelidpair] = upsilon
            else:
                artisatomic.log_and_print(
                    flog,
                    f"Duplicate upsilon value for transition {lower:d} to {upper:d} keeping"
                    f" {upsilondict[levelidpair]:5.2e} instead of using {upsilon:5.2e}",
                )

    artisatomic.log_and_print(flog, f"Read {len(energylevels):d} levels")

    return ionization_energy_ev, energylevels, upsilondict


def read_qub_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the QUB calculations.

    Covers both the newer per-ion adf04 calculations and the older Co II/III/IV data sets, which
    have their own file layouts. Also returns the effective collision strengths, so this reader
    supplies an upsilondict where most others leave it to be filled in elsewhere.
    """
    new_qub_calculations = {
        (38, 1),
        (38, 2),
        (38, 3),
        (38, 4),
        (38, 5),
        (39, 2),
        (39, 3),
        (40, 1),
        (40, 2),
        (40, 3),
        (52, 1),
        (52, 2),
        (52, 3),
        (52, 4),
        (52, 5),
        (60, 2),
        (74, 1),
        (74, 2),
        (74, 3),
        (78, 1),
        (78, 2),
        (78, 3),
        (79, 1),
        (79, 2),
        (79, 3),
    }

    if (atomic_number == 27) and (ion_stage == 3):
        ionization_energy_ev, qub_energylevels, upsilondict = read_adf04(
            tyndall_co3_path / "adf04_v1", atomic_number, ion_stage, flog
        )

        qub_transitions = []
        transition_count_of_level_name = defaultdict(int)
        transitionfile = tyndall_co3_path / "adf04rad_v1"
        with artisatomic.xopen_check_extension(transitionfile) as ftrans:
            for line in ftrans:
                row = line.split()
                id_upper = int(row[0])
                id_lower = int(row[1])
                A = float(row[2])
                if A > 2e-30:
                    # a raise rather than an assert: this validates an input file, and a
                    # non-positive id would otherwise wrap to the wrong level via negative indexing
                    if id_lower < 1 or id_upper < 1:
                        msg = f"non-positive transition level ids {id_lower}, {id_upper} in {transitionfile}"
                        raise ValueError(msg)
                    # the file numbers levels from one; level ids are zero-based in memory
                    id_lower -= 1
                    id_upper -= 1
                    level_upper = qub_energylevels[id_upper]
                    level_lower = qub_energylevels[id_lower]
                    namefrom = level_upper.levelname
                    nameto = level_lower.levelname
                    # WARNING replace with correct selection rules!
                    forbidden = level_upper.parity == level_lower.parity
                    transition_count_of_level_name[namefrom] += 1
                    transition_count_of_level_name[nameto] += 1
                    delta_percm = level_upper.energyabovegsinpercm - level_lower.energyabovegsinpercm
                    lamdaangstrom = 1.0e8 / delta_percm if delta_percm != 0.0 else -1.0
                    if (id_lower, id_upper) in upsilondict:
                        coll_str = upsilondict[id_lower, id_upper]
                    elif forbidden:
                        coll_str = -2.0
                    else:
                        coll_str = -1.0
                    transition = QUBTransitionRow(id_lower, id_upper, A, namefrom, nameto, lamdaangstrom, coll_str)
                    qub_transitions.append(transition)

    elif (atomic_number == 27) and (ion_stage == 4):
        transition_count_of_level_name = defaultdict(int)
        qub_energylevels: list[QUBEnergyLevel] = [QUBEnergyLevel("groundstate", 1, 0, 0, 0, 0.0, 10, 0)]
        qub_transitions = pl.DataFrame(schema=artisatomic.empty_transitions_schema)
        upsilondict: dict[tuple[int, int], float] = {}
        ionization_energy_ev = 54.9000015

    elif (atomic_number, ion_stage) in new_qub_calculations:
        atom_filepath = Path("atomic-data-qub", f"{atomic_number}_{ion_stage}.adf04")
        ionization_energy_ev, qub_energylevels, upsilondict = read_adf04(atom_filepath, atomic_number, ion_stage, flog)

        qub_transitions = []
        transition_count_of_level_name = defaultdict(int)
        # two passes are needed (collision strengths are found by line length, known only after
        # seeing every line), so read the lines once: a decompressing stream may not seek back
        with artisatomic.xopen_check_extension(atom_filepath) as ftrans:
            translines = ftrans.readlines()

        atom_group_note = False
        longest_line_length = 0
        # Use the length of lines to locate collision strengths
        # Ignoring notes section between C- markers
        for line in translines:
            if line.startswith("C-"):
                atom_group_note = not atom_group_note
                continue
            if atom_group_note:
                continue
            longest_line_length = max(longest_line_length, len(line))

        for line in translines:
            if line.startswith("C-"):
                atom_group_note = not atom_group_note
                continue
            if atom_group_note:
                continue

            rowlen = line
            row = line.split()

            if len(rowlen) == longest_line_length:
                # W II file has the first two columns swapped around from the standard order
                if atomic_number == 74 and ion_stage == 2:
                    id_upper = int(row[1])
                    id_lower = int(row[0])
                else:
                    id_upper = int(row[0])
                    id_lower = int(row[1])
                A = float(row[2].replace("-", "E-").replace("+", "E+"))
                if A > 2e-30:
                    # a raise rather than an assert: this validates an input file. A non-positive
                    # id would wrap to the wrong level via negative indexing, and one past the end
                    # would raise a bare IndexError naming neither the file nor the transition.
                    if not 1 <= id_lower <= len(qub_energylevels) or not 1 <= id_upper <= len(qub_energylevels):
                        msg = (
                            f"transition level ids {id_lower}, {id_upper} in {atom_filepath} are outside"
                            f" the file's {len(qub_energylevels)} levels"
                        )
                        raise ValueError(msg)
                    # the file numbers levels from one; level ids are zero-based in memory
                    id_lower -= 1
                    id_upper -= 1
                    level_upper = qub_energylevels[id_upper]
                    level_lower = qub_energylevels[id_lower]
                    namefrom = level_upper.levelname
                    nameto = level_lower.levelname
                    forbidden = check_forbidden(level_upper, level_lower)
                    transition_count_of_level_name[namefrom] += 1
                    transition_count_of_level_name[nameto] += 1
                    delta_percm = level_upper.energyabovegsinpercm - level_lower.energyabovegsinpercm
                    lamdaangstrom = 1.0e8 / delta_percm if delta_percm != 0.0 else -1.0
                    if (id_lower, id_upper) in upsilondict:
                        coll_str = upsilondict[id_lower, id_upper]
                    elif forbidden:
                        coll_str = -2.0
                    else:
                        coll_str = -1.0
                    transition = QUBTransitionRow(id_lower, id_upper, A, namefrom, nameto, lamdaangstrom, coll_str)
                    qub_transitions.append(transition)

    else:
        msg = f"No QUB data available for Z={atomic_number} ion_stage {ion_stage}"
        raise ValueError(msg)

    artisatomic.log_and_print(flog, f"Read {len(qub_transitions):d} transitions")

    return ionization_energy_ev, qub_energylevels, qub_transitions, transition_count_of_level_name, upsilondict


def read_qub_photoionizations(
    atomic_number, ion_stage, levelcount: int, args, flog
) -> tuple[npt.NDArray[np.float64], list[list[tuple[int, float]]], npt.NDArray[np.float64]]:
    """Read QUB photoionization cross sections for one ion, downsampled onto the output grid.

    Returns the cross sections, the upper-ion target fractions per level, and the threshold
    energies, all indexed by zero-based level id. Levels with no data keep an empty target list,
    which is how write_phixs_data() knows to skip them.
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
            artisatomic.log_and_print(flog, f"Reading {artisatomic.path_for_log(filename)}")
            photdata = pd.read_csv(filename, sep=r"\s+", header=None)
            phixstables = {}
            ntargets = 4  # just the 4Fe ground quartet (the file has 40 target columns)

            # column n of the file holds the cross section to the upper ion's level id n - 1
            for targetcolumn in range(1, ntargets + 1):
                phixstables[targetcolumn] = photdata.loc[photdata[:][targetcolumn] > 0.0][[0, targetcolumn]].to_numpy()

            reduced_phixs_dict = artisatomic.reduce_phixs_tables(
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
                artisatomic.log_and_print(
                    flog, f"WARNING: all photoionisation targets for level {lowerlevelid} have zero cross section"
                )
                continue
            target_scalefactors = [x if (x / scalefactorsum > 0.02) else 0.0 for x in target_scalefactors]
            scalefactorsum = sum(target_scalefactors)

            # -1.0 sentinel: the threshold energy comes from the level energies, not from the
            # first energy point of the cross-section table
            photoionization_thresholds_ev[lowerlevelid] = -1.0
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
            dict_phixstable = {"gs": np.array(list(zip(np.arange(1.0, 10.9, 0.1), phixsvalues_const, strict=False)))}
            phixsvalues = artisatomic.reduce_phixs_tables(
                dict_phixstable, args.optimaltemperature, args.nphixspoints, args.phixsnuincrement
            )["gs"]

        # unlike the Co II branch above, every level deliberately gets a phixs entry: the ground
        # quartet gets the tabulated cross section and higher levels an explicit all-zero table
        for levelid in range(levelcount):
            photoionization_thresholds_ev[levelid] = -1.0
            photoionization_targetfractions[levelid] = [(0, 1.0)]  # the upper ion's ground state
            if levelid < 4:
                photoionization_crosssections[levelid] = phixsvalues

    return photoionization_crosssections, photoionization_targetfractions, photoionization_thresholds_ev


def get_level_valence_n(levelname: str):
    """Principal quantum number of the valence electron, read from a QUB level name.

    Falls back to n=1 with a warning when the name cannot be parsed, since a missing n only
    affects the hydrogenic photoionisation estimate rather than the level itself.

    Kept separate from the other readers' versions: each data source names its levels
    differently, so a shared parser would have to guess which convention it is looking at.
    """
    namesplit = levelname.split("_")
    part = namesplit[0].strip()
    if len(namesplit) < 2:
        print(f"WARNING: Could not find valence n in {levelname}. Using n=1")
        return 1

    if part[-1] == ")" and "(" in part:
        part = part[: part.rfind("(")]

    if part[-1] not in lchars.lower():
        # Last character must be mumber of electrons in the orbital: remove it
        if not part[-1].isdigit():
            print(f"WARNING: Could not find n in {levelname}. Using n=1")
            return 1
        part = part.rstrip(string.digits)
    part = part.strip(lchars.lower())

    valance_n = None

    # Inefficient way to find the last number in a string
    for i in range(len(part)):
        try:
            n = int(part[i:])
        except ValueError:
            continue
        else:
            assert n >= 0
            valance_n = n
            n_start_index = i  # Track where the number starts
            if n_start_index > 0 and part[n_start_index - 1] in lchars.lower():
                str_valance_n = str(valance_n)
                # Strip the first digit of n
                valance_n = int(str_valance_n[1:] if len(str_valance_n) > 1 else str_valance_n)
            assert valance_n < 50
        return valance_n
    if valance_n is None:
        print(f"WARNING: Could not find n in {levelname}. Using n=1")
    return 1

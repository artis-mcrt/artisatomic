#!/usr/bin/env python3
from collections import defaultdict
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from xopen import xopen

import artisatomic

tyndall_co3_path = (Path(__file__).parent.resolve() / ".." / "atomic-data-qub" / "co_tyndall").resolve()

ryd_to_ev = 13.605693122994232

hc_in_ev_cm = 0.0001239841984332003
hc_in_ev_angstrom = 12398.419843320025
h_in_ev_seconds = 4.135667696923859e-15
lchars = "SPDFGHIKLMNOPQRSTUVWXYZ"

qub_energy_level_row = namedtuple(
    "qub_energy_level_row", "levelname qub_id twosplusone l j energyabovegsinpercm g parity"
)

qubpath = (Path(__file__).parent.resolve() / ".." / "atomic-data-qub").resolve()


def extend_ion_list(ion_handlers):
    qubions = sorted([tuple(int(x) for x in f.parts[-1].split(".")[0].split("_")) for f in qubpath.glob("*_*.adf04")])
    for atomic_number, ion_stage in qubions:
        found_element = False
        for tmp_atomic_number, list_ions_handlers in ion_handlers:
            if tmp_atomic_number == atomic_number:
                # add an ion that is not present in the element's list
                if ion_stage not in [x[0] if hasattr(x, "__getitem__") else x for x in list_ions_handlers]:
                    list_ions_handlers.append((ion_stage, "qub_data"))
                    list_ions_handlers.sort(key=lambda x: x[0] if hasattr(x, "__getitem__") else x)
                found_element = True

        if not found_element:
            ion_handlers.append(
                (
                    atomic_number,
                    [(ion_stage, "qub_data")],
                )
            )

    ion_handlers.sort(key=lambda x: x[0])

    return ion_handlers


def read_adf04(filepath, atomic_number, ion_stage, flog):
    if not Path(filepath).is_file() and Path(f"{filepath}.gz").is_file():
        filepath = f"{filepath}.gz"
    energylevels = [None]
    upsilondict = {}
    ionization_energy_ev = 0.0
    artisatomic.log_and_print(flog, f"Reading {filepath}")
    with xopen(filepath) as fleveltrans:
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
                energylevel = qub_energy_level_row(
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
                energylevel = qub_energy_level_row(
                    config,
                    int(line[:5]),
                    int(line[29:30]),
                    int(line[31:32]),
                    float(line[33:37]),
                    float(line[38:59]),
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

        upsilonheader = fleveltrans.readline().split()
        list_tempheaders = [f"upsT={x:}" for x in upsilonheader[2:]]
        list_headers = ["upper", "lower", "ignore", *list_tempheaders]
        qubupsilondf_alltemps = pd.read_csv(
            fleveltrans,
            index_col=False,
            sep=r"\s+",
            comment="C",
            names=list_headers,
            dtype={"lower": int, "upper": int}.update(dict.fromkeys(list_headers[2:], float)),
            on_bad_lines="skip",
            skip_blank_lines=True,
            keep_default_na=False,
        )
        qubupsilondf_alltemps = qubupsilondf_alltemps.query("upper!=-1")
        for _, row in qubupsilondf_alltemps.iterrows():
            lower = int(row["lower"])
            upper = int(row["upper"])
            lower, upper = min(lower, upper), max(lower, upper)
            assert upper > lower

            # Co, W I and II rates are calculated at different temperatures
            # Should be handled in a less approximate way in the future
            if atomic_number == 27:
                upsilon = float(row["upsT=5.01+03"].replace("-", "E-").replace("+", "E+"))
            elif atomic_number == 74 and ion_stage == 1:
                upsilon = float(row["upsT=5.80+03"].replace("-", "E-").replace("+", "E+"))
            elif atomic_number == 74 and ion_stage == 2:
                upsilon = float(row["upsT=4.00+03"].replace("-", "E-").replace("+", "E+"))
            else:
                upsilon = float(row["upsT=5.00+03"].replace("-", "E-").replace("+", "E+"))
            if (lower, upper) not in upsilondict:
                upsilondict[(lower, upper)] = upsilon
            else:
                artisatomic.log_and_print(
                    flog,
                    f"Duplicate upsilon value for transition {lower:d} to {upper:d} keeping"
                    f" {(upsilondict[(lower, upper)],):5.2e} instead of using {upsilon:5.2e}",
                )

    artisatomic.log_and_print(flog, f"Read {len(energylevels[1:]):d} levels")

    return ionization_energy_ev, energylevels, upsilondict


def read_qub_levels_and_transitions(atomic_number, ion_stage, flog):
    qub_transition_row = namedtuple("transition", "lowerlevel upperlevel A nameto namefrom lambdaangstrom coll_str")

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
        if not Path(transitionfile).is_file() and Path(f"{transitionfile}.gz").is_file():
            transitionfile = f"{transitionfile}.gz"
        with xopen(transitionfile) as ftrans:
            for line in ftrans:
                row = line.split()
                id_upper = int(row[0])
                id_lower = int(row[1])
                A = float(row[2])
                if A > 2e-30:
                    namefrom = qub_energylevels[id_upper].levelname
                    nameto = qub_energylevels[id_lower].levelname
                    # WARNING replace with correct selection rules!
                    forbidden = qub_energylevels[id_upper].parity == qub_energylevels[id_lower].parity
                    transition_count_of_level_name[namefrom] += 1
                    transition_count_of_level_name[nameto] += 1
                    lamdaangstrom = 1.0e8 / (
                        qub_energylevels[id_upper].energyabovegsinpercm
                        - qub_energylevels[id_lower].energyabovegsinpercm
                    )
                    if (id_lower, id_upper) in upsilondict:
                        coll_str = upsilondict[(id_lower, id_upper)]
                    elif forbidden:
                        coll_str = -2.0
                    else:
                        coll_str = -1.0
                    transition = qub_transition_row(id_lower, id_upper, A, namefrom, nameto, lamdaangstrom, coll_str)
                    qub_transitions.append(transition)

    elif (atomic_number == 27) and (ion_stage == 4):
        transition_count_of_level_name = defaultdict(int)
        qub_energylevels = [None]
        qub_transitions = pl.DataFrame(schema={"lowerlevel": pl.Int64, "upperlevel": pl.Int64, "A": pl.Float64})
        upsilondict = {}
        ionization_energy_ev = 54.9000015
        qub_energylevels.append(qub_energy_level_row("groundstate", 1, 0, 0, 0, 0.0, 10, 0))

    elif (atomic_number, ion_stage) in new_qub_calculations:
        atom_file = f"{atomic_number}_{ion_stage}.adf04"
        ionization_energy_ev, qub_energylevels, upsilondict = read_adf04(
            Path("atomic-data-qub", atom_file), atomic_number, ion_stage, flog
        )

        qub_transitions = []
        transition_count_of_level_name = defaultdict(int)
        with open(Path("atomic-data-qub", atom_file)) as ftrans:
            ftrans.seek(0)
            atom_group_note = False
            longest_line_length = 0
            # Use the length of lines to locate collision strengths
            # Ignoring notes section between C- markers
            for line in ftrans:
                if line.startswith("C-"):
                    atom_group_note = not atom_group_note
                    continue
                if atom_group_note:
                    continue
                longest_line_length = max(longest_line_length, len(line))
            ftrans.seek(0)
            for line in ftrans:
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
                        namefrom = qub_energylevels[id_upper].levelname
                        nameto = qub_energylevels[id_lower].levelname
                        forbidden = artisatomic.check_forbidden(qub_energylevels[id_upper], qub_energylevels[id_lower])
                        transition_count_of_level_name[namefrom] += 1
                        transition_count_of_level_name[nameto] += 1
                        lamdaangstrom = 1.0e8 / (
                            qub_energylevels[id_upper].energyabovegsinpercm
                            - qub_energylevels[id_lower].energyabovegsinpercm
                        )
                        if (id_lower, id_upper) in upsilondict:
                            coll_str = upsilondict[(id_lower, id_upper)]
                        elif forbidden:
                            coll_str = -2.0
                        else:
                            coll_str = -1.0
                        transition = qub_transition_row(
                            id_lower, id_upper, A, namefrom, nameto, lamdaangstrom, coll_str
                        )
                        qub_transitions.append(transition)

    artisatomic.log_and_print(flog, f"Read {len(qub_transitions):d} transitions")

    return ionization_energy_ev, qub_energylevels, qub_transitions, transition_count_of_level_name, upsilondict


def read_qub_photoionizations(atomic_number, ion_stage, energy_levels, args, flog):
    photoionization_crosssections = np.zeros((len(energy_levels), args.nphixspoints))
    photoionization_targetfractions = [[(1, 1.0)] for _ in energy_levels]
    photoionization_thresholds_ev = np.zeros(len(energy_levels))

    if atomic_number == 27 and ion_stage == 2:
        for lowerlevelid in [1, 2, 3, 4, 5, 6, 7, 8]:
            filename = tyndall_co3_path / f"{lowerlevelid:d}.gz"
            artisatomic.log_and_print(flog, f"Reading {filename}")
            photdata = pd.read_csv(filename, sep=r"\s+", header=None)
            phixstables = {}
            # ntargets = 40
            ntargets = 4  # just the 4Fe ground quartet
            photoionization_thresholds_ev[lowerlevelid] = -1.0
            # photoionization_thresholds_ev[lowerlevelid] = photdata.loc[0][0]

            for targetlevel in range(1, ntargets + 1):
                phixstables[targetlevel] = photdata.loc[photdata[:][targetlevel] > 0.0][[0, targetlevel]].to_numpy()

            reduced_phixs_dict = artisatomic.reduce_phixs_tables(
                phixstables, args.optimaltemperature, args.nphixspoints, args.phixsnuincrement
            )
            target_scalefactors = np.zeros(ntargets + 1)
            upperlevelid_withmaxfraction = 1
            max_scalefactor = 0.0
            for upperlevelid in reduced_phixs_dict:
                # take the ratio of cross sections at the threshold energies
                scalefactor = reduced_phixs_dict[upperlevelid][0]
                target_scalefactors[upperlevelid] = scalefactor
                # target_scalefactors[upperlevelid] = np.average(reduced_phixs_dict[upperlevelid])
                if scalefactor > max_scalefactor:
                    upperlevelid_withmaxfraction = upperlevelid
                    max_scalefactor = scalefactor

            scalefactorsum = sum(target_scalefactors)
            target_scalefactors = [x if (x / scalefactorsum > 0.02) else 0.0 for x in target_scalefactors]
            scalefactorsum = sum(target_scalefactors)

            photoionization_targetfractions[lowerlevelid] = []
            for upperlevelid, target_scalefactor in enumerate(target_scalefactors[1:], 1):
                target_fraction = target_scalefactor / scalefactorsum
                if target_fraction > 0.001:
                    photoionization_targetfractions[lowerlevelid].append((upperlevelid, target_fraction))

            max_fraction = max_scalefactor / scalefactorsum
            photoionization_crosssections[lowerlevelid] = (
                reduced_phixs_dict[upperlevelid_withmaxfraction] / max_fraction
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

        for lowerlevelid in range(1, len(energy_levels)):
            photoionization_thresholds_ev[lowerlevelid] = -1.0
            photoionization_targetfractions[lowerlevelid] = [(1, 1.0)]
            if lowerlevelid <= 4:
                photoionization_crosssections[lowerlevelid] = phixsvalues

    return photoionization_crosssections, photoionization_targetfractions, photoionization_thresholds_ev


def get_level_valence_n(levelname: str):
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
        part = part.rstrip("0123456789")
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

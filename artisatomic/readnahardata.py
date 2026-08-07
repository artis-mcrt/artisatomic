"""Read levels, configurations and photoionization cross sections from Nahar's NORAD data."""

import sys
import typing as t
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import polars as pl

import artisatomic

ryd_to_ev = 13.605693122994232

hc_in_ev_cm = 0.0001239841984332003
hc_in_ev_angstrom = 12398.419843320025
h_in_ev_seconds = 4.135667696923859e-15

alphabets = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
reversedalphabets = "zyxwvutsrqponmlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA "
lchars = "SPDFGHIKLMNOPQRSTUVWXYZ"


class NaharCoreState(t.NamedTuple):
    """One target/core state of the wavefunction expansion, i.e. a state of the upper ion."""

    nahar_core_state_id: int
    configuration: str
    term: str
    energyrydberg: float


class NaharEnergyLevel(t.NamedTuple):
    """One energy level read from a Nahar .en.ls.txt file."""

    # Nahar levels have no spectroscopic name of their own, so this is built from the level's
    # symmetry and configuration; write_adata() writes it as the level comment in adata.txt
    levelname: str
    indexinsymmetry: int
    # "T" for a valence electron state, "C" for an equivalent electron state
    TC: str
    corestateid: int
    elecn: int
    elecl: int
    energyreltoionpotrydberg: float
    twosplusone: int
    l: int
    parity: int
    energyabovegsinpercm: float
    g: int
    naharconfiguration: str


# derived from the row class so it stays the single source of the frame layout
nahar_level_schema = pl.Schema(
    {
        name: {str: pl.String, float: pl.Float64, int: pl.Int64}[fieldtype]
        for name, fieldtype in NaharEnergyLevel.__annotations__.items()
    }
)


def build_nahar_levels_and_phixs(
    nahar_energy_levels: list[NaharEnergyLevel],
    nahar_phixs_tables,
    thresholds_ev_dict,
    args,
    flog,
) -> tuple[pl.DataFrame, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Build the level table for a Nahar-only ion and attach its photoionization cross sections."""
    # the explicit schema keeps an empty level list (e.g. a missing energy file) producing a frame
    # with the columns that the sort below and write_adata() expect
    dfenergy_levels = pl.DataFrame(nahar_energy_levels, schema=nahar_level_schema, orient="row")

    levels_with_config = (dfenergy_levels["naharconfiguration"] != "UNKNOWN CONFIG").sum()
    artisatomic.log_and_print(
        flog,
        f"Included {dfenergy_levels.height} levels from Nahar dataset"
        f" ({levels_with_config} with an electron configuration)",
    )

    # Level ids are assigned from the row order, so ties must keep the order they were read in:
    # polars sorts unstably by default.
    print("Sorting levels by energy...")
    dfenergy_levels = dfenergy_levels.sort("energyabovegsinpercm", maintain_order=True)

    # stay empty unless there are Nahar phixs tables to attach to the level list
    photoionization_crosssections: npt.NDArray[np.float64] = np.empty((0, args.nphixspoints))
    photoionization_thresholds_ev: npt.NDArray[np.float64] = np.empty(0)

    if nahar_phixs_tables:
        photoionization_crosssections = np.zeros((dfenergy_levels.height, args.nphixspoints))
        photoionization_thresholds_ev = np.full(dfenergy_levels.height, np.nan)

        if not args.nophixs:
            reduced_phixs_dict = artisatomic.reduce_phixs_tables(
                nahar_phixs_tables, args.optimaltemperature, args.nphixspoints, args.phixsnuincrement
            )

            # there can be more than one level per state, and they all get the same table
            levelindices_of_state: dict[tuple[int, int, int, int], list[int]] = {}
            for levelindex, levelstate in enumerate(
                dfenergy_levels.select("twosplusone", "l", "parity", "indexinsymmetry").iter_rows()
            ):
                levelindices_of_state.setdefault(levelstate, []).append(levelindex)

            for state_tuple, phixstable in reduced_phixs_dict.items():
                matching_levelindices = levelindices_of_state.get(state_tuple, [])
                # a state right at the ionization threshold has nothing to ionize from, so leave
                # it with no threshold energy and let write_phixs_data() skip it
                threshold_ev = thresholds_ev_dict[state_tuple]
                for levelindex in matching_levelindices:
                    photoionization_crosssections[levelindex] = phixstable
                    if threshold_ev > 0.0:
                        photoionization_thresholds_ev[levelindex] = threshold_ev

                if not matching_levelindices:
                    twosplusone, l, parity, indexinsymmetry = state_tuple
                    artisatomic.log_and_print(
                        flog,
                        "No Nahar state to match with photoionization crosssection of"
                        f" {twosplusone:d}{lchars[l]}{['e', 'o'][parity]} index {indexinsymmetry:d}",
                    )

    return dfenergy_levels, photoionization_crosssections, photoionization_thresholds_ev


def read_nahar_energy_level_file(
    path_nahar_energy_file, atomic_number, ion_stage, flog
) -> tuple[
    list[NaharEnergyLevel],
    list[NaharCoreState],
    dict[tuple[int, int, int, int], str],
    float,
]:
    """Read one ion's Nahar energy level file (.en.ls.txt).

    Returns the levels, the core states, the electron configurations of the bound states, and the
    ionization potential in Rydberg. A missing file is logged and returns empty data rather than
    raising, so an ion with no Nahar data simply contributes nothing.
    """
    # state tuples are (2S+1, L, parity, index in symmetry)
    nahar_configurations: dict[tuple[int, int, int, int], str] = {}
    nahar_energy_levels: list[NaharEnergyLevel] = []
    nahar_core_states: list[NaharCoreState] = []
    nahar_ionization_potential_rydberg = -1.0

    # the plain name may not be on disk when the compressed form is, so ask the same resolver that
    # xopen_check_extension() uses rather than testing the plain name with os.path.isfile()
    if artisatomic.find_file_check_extension(path_nahar_energy_file) is None:
        artisatomic.log_and_print(flog, f"{artisatomic.path_for_log(path_nahar_energy_file)} does not exist")
    else:
        artisatomic.log_and_print(flog, f"Reading {artisatomic.path_for_log(path_nahar_energy_file)}")
        with artisatomic.xopen_check_extension(path_nahar_energy_file) as fenlist:
            nahar_core_states = read_nahar_core_states(fenlist)

            nahar_configurations, nahar_ionization_potential_rydberg = read_nahar_configurations(fenlist, flog)

            # every level energy below is measured from this, so a missing value would silently
            # offset the whole level list (and be written to adata.txt as a negative ionization energy)
            if nahar_ionization_potential_rydberg <= 0.0:
                msg = (
                    f"No ' Ion ground state' line found in {path_nahar_energy_file}, so the ionization"
                    " potential is unknown and the level energies cannot be computed"
                )
                raise ValueError(msg)

            while True:
                line = fenlist.readline()
                if not line:
                    print("End of file before table iii header")
                    sys.exit()
                if line.startswith("iii) Table of complete set (both negative and positive) of energies"):
                    break
            line = fenlist.readline()  # line of ----------------------
            while True:
                line = fenlist.readline()
                if not line:
                    print("End of file before table iii ends")
                    sys.exit()
                if line.startswith("------------------------------------------------------"):
                    break

            row = fenlist.readline().split()  # line of atomic number and electron number, unless
            while len(row) == 0:  # some extra blank lines might exist
                row = fenlist.readline().split()

            if atomic_number != int(row[0]) or ion_stage != int(row[0]) - int(row[1]):
                print(
                    "Wrong atomic number or ionization stage in Nahar energy file",
                    atomic_number,
                    int(row[0]),
                    ion_stage,
                    int(row[0]) - int(row[1]),
                )
                sys.exit()

            while True:
                line = fenlist.readline()
                if not line:
                    print("End of file before table finished")
                    sys.exit()
                row = line.split()
                if row == ["0", "0", "0", "0"]:
                    break
                twosplusone = int(row[0])
                l_val = int(row[1])
                parity = int(row[2])
                number_of_states_in_symmetry = int(row[3])

                line = fenlist.readline()  # core state number and energy

                for _ in range(number_of_states_in_symmetry):
                    row = fenlist.readline().split()
                    indexinsymmetry = int(row[0])
                    # a letter, not a number: reading it as one would shift every column after it
                    TC = row[1]
                    if TC not in {"T", "C"}:
                        msg = f"Expected 'T' or 'C' in column 2 of the table iii energy row {row}, but found {TC!r}"
                        raise ValueError(msg)
                    nahar_core_state_id = int(row[2])
                    elecn = int(row[3])
                    elecl = int(row[4])
                    energyreltoionpotrydberg = float(row[5])
                    if nahar_core_state_id < 1 or nahar_core_state_id > len(nahar_core_states):
                        flog.write(
                            "Core state id of {:d}{}{} index {:d} is invalid (={:d}, Ncorestates={:d}). Setting"
                            " core state to 1 instead.\n".format(
                                twosplusone,
                                lchars[l_val],
                                ["e", "o"][parity],
                                indexinsymmetry,
                                nahar_core_state_id,
                                len(nahar_core_states),
                            )
                        )
                        nahar_core_state_id = 1

                    energyabovegsinpercm = (
                        (nahar_ionization_potential_rydberg + energyreltoionpotrydberg) * ryd_to_ev / hc_in_ev_cm
                    )

                    # the spectroscopic table covers only the bound states, so the levels above the
                    # ionization threshold have no known configuration
                    naharconfiguration = nahar_configurations.get(
                        (twosplusone, l_val, parity, indexinsymmetry), "UNKNOWN CONFIG"
                    )
                    nahar_energy_levels.append(
                        NaharEnergyLevel(
                            levelname=(
                                f"Nahar: {twosplusone:d}{lchars[l_val]}{['e', 'o'][parity]}"
                                f" index {indexinsymmetry} '{naharconfiguration}'"
                            ),
                            indexinsymmetry=indexinsymmetry,
                            TC=TC,
                            corestateid=nahar_core_state_id,
                            elecn=elecn,
                            elecl=elecl,
                            energyreltoionpotrydberg=energyreltoionpotrydberg,
                            twosplusone=twosplusone,
                            l=l_val,
                            parity=parity,
                            energyabovegsinpercm=energyabovegsinpercm,
                            g=twosplusone * (2 * l_val + 1),
                            naharconfiguration=naharconfiguration,
                        )
                    )

    #                if float(nahar_energy_levels[-1].energyreltoionpotrydberg) >= 0.0:
    #                    nahar_energy_levels.pop()

    return (
        nahar_energy_levels,
        nahar_core_states,
        nahar_configurations,
        nahar_ionization_potential_rydberg,
    )


def read_nahar_core_states(fenlist) -> list[NaharCoreState]:
    """Read table i, the target/core states of the wavefunction expansion.

    Reads from the current position of the open file and leaves it just past the table. Core
    state id n is stored at index n - 1, and the file's numbering is checked against that.
    """
    while True:
        line = fenlist.readline()
        if not line:
            print("End of file before data section")
            sys.exit()
        if line.startswith(" i) Table of target/core states in the wavefunction expansion"):
            break

    while True:
        line = fenlist.readline()
        if not line:
            print("End of file before end of core states table")
            sys.exit()
        if line.startswith(" no of target/core states in WF"):
            numberofcorestates = int(line.split("=")[1])
            break
    fenlist.readline()  # blank line
    fenlist.readline()  # ' target states and energies:'
    fenlist.readline()  # blank line

    # core state id n is stored at index n - 1
    nahar_core_states: list[NaharCoreState] = []
    for c in range(1, numberofcorestates + 1):
        row = fenlist.readline().split()
        state = NaharCoreState(int(row[0]), row[1], row[2], float(row[3]))
        nahar_core_states.append(state)
        if int(state.nahar_core_state_id) != c:
            print(f"Nahar levels mismatch: id {c:d} found at entry number {int(state.nahar_core_state_id):d}")
            sys.exit()
    return nahar_core_states


def read_nahar_phixs_tables(path_nahar_px_file, atomic_number, ion_stage, args):
    """Read Nahar photoionization cross sections, keyed by (2S+1, L, parity, index in symmetry).

    Returns the cross-section tables and each state's threshold energy in eV. The tables are
    (energy in Rydberg, cross section in Megabarns) pairs, at the file's own energy resolution;
    reduce_phixs_tables() downsamples them onto the output grid later.
    """
    nahar_phixs_tables = {}
    thresholds_ev_dict = {}
    with artisatomic.xopen_check_extension(path_nahar_px_file) as fenlist:
        while True:
            line = fenlist.readline()
            if not line:
                print("End of file before data section")
                sys.exit()
            if line.startswith("----------------------------------------------"):
                break

        line = ""
        while not line.strip():
            line = fenlist.readline()
        row = line.split()
        if atomic_number != int(row[0]) or ion_stage != int(row[0]) - int(row[1]):
            print(
                "Wrong atomic number or ionization stage in Nahar file",
                atomic_number,
                int(row[0]),
                ion_stage,
                int(row[0]) - int(row[1]),
            )
            sys.exit()

        while True:
            line = fenlist.readline()
            row = line.split()

            if not line:
                break
            if not row:
                continue  # a blank line is not the end of the table (sum([]) == 0 used to break here)
            if len(row) < 4 or not all(map(artisatomic.isfloat, row)):
                break  # trailing text after the table rather than another state
            if sum(map(float, row)) == 0:
                break  # the "0 0 0 0" terminator

            twosplusone, l, parity, indexinsymmetry = int(row[0]), int(row[1]), int(row[2]), int(row[3])

            number_of_points = int(fenlist.readline().split()[1])
            binding_energy_ryd = float(fenlist.readline().split()[0])
            thresholds_ev_dict[twosplusone, l, parity, indexinsymmetry] = binding_energy_ryd * ryd_to_ev

            if not args.nophixs:
                phixsarray = np.array([list(map(float, fenlist.readline().split())) for _ in range(number_of_points)])
            else:
                for _ in range(number_of_points):
                    fenlist.readline()
                phixsarray = np.zeros((2, 2))

            nahar_phixs_tables[twosplusone, l, parity, indexinsymmetry] = phixsarray

    return nahar_phixs_tables, thresholds_ev_dict


def read_nahar_configurations(fenlist, flog) -> tuple[dict[tuple[int, int, int, int], str], float]:
    """Read table ii, the bound states with spectroscopic notation, and the ionization potential.

    Returns the electron configurations keyed by (2S+1, L, parity, index in symmetry), and the
    ionization potential in Rydberg. Reads from the current position of the open file and leaves
    it just past the table. Only bound states appear here, so levels above the ionization
    threshold have no entry. The index in symmetry comes from the seniority letter that prefixes
    each term, ascending for even parity and descending for odd.
    """
    nahar_configurations: dict[tuple[int, int, int, int], str] = {}
    nahar_ionization_potential_rydberg = -1.0
    while True:
        line = fenlist.readline()
        if not line:
            print("End of file before data section ii)")
            sys.exit()
        if line.startswith("ii) Table of bound (negative) state energies (with spectroscopic notation)"):
            break

    found_table = False
    while True:
        line = fenlist.readline()
        if not line:
            print("End of file before end of state table")
            sys.exit()

        if line.startswith(" Ion ground state"):
            nahar_ionization_potential_rydberg_str = line.split("=")[2]
            if "E" not in nahar_ionization_potential_rydberg_str:
                nahar_ionization_potential_rydberg_str = nahar_ionization_potential_rydberg_str.replace("+", "E+")
            nahar_ionization_potential_rydberg = -float(nahar_ionization_potential_rydberg_str)
            flog.write(f"Ionization potential = {nahar_ionization_potential_rydberg * ryd_to_ev:.4f} eV\n")

        if line.startswith(" ") and len(line) > 36 and artisatomic.isfloat(line[29 : 29 + 8]):
            found_table = True
            state = line[1:22]
            #               energy = float(line[29:29+8])
            twosplusone = int(state[18])
            l_val = lchars.index(state[19])

            if state[20] == "o":
                parity = 1
                indexinsymmetry = reversedalphabets.index(state[17]) + 1
            else:
                parity = 0
                indexinsymmetry = alphabets.index(state[17]) + 1

            # print(state,energy,twosplusone,l,parity,indexinsymmetry)
            nahar_configurations[twosplusone, l_val, parity, indexinsymmetry] = state
        elif found_table:
            break

    return nahar_configurations, nahar_ionization_potential_rydberg


def reduce_configuration(instr: str) -> str:
    """Normalise a configuration for comparison, e.g. "3d64s  (6D ) 8p  j5Fo" -> "3d64s8p_5Fo".

    Drops the parent term, the seniority letter and any J value, giving a form close to the
    Hillier style "3d6(5D)4s8p_5Fo" but without the parent term. Used to compare a Nahar core
    state against an upper-ion level name, which the two data sets spell differently.
    """
    if instr == "-1":
        return "-1"
    instr = instr.split("[", maxsplit=1)[0]  # remove trailing bracketed J value

    if instr[-1] not in {"o", "e"}:
        instr += "e"  # last character being S,P,D, etc means even
    if str.isdigit(instr[-2]):  # J value is in the term, so remove it
        instr = instr[:-2] + instr[-1]

    outstr = remove_bracketed_part(instr)
    outstr += "_"
    outstr += instr[-3:-1]
    outstr += "o" if instr[-1] == "o" else "e"
    return outstr


def remove_bracketed_part(instr: str) -> str:
    """Remove anything between parentheses, and the parentheses themselves.

    e.g. remove_bracketed_part('AB(CD)EF') = 'ABEF'.
    """
    outstr = ""
    in_brackets = False
    for char in instr[:-4]:
        if char in {" ", "_"}:
            continue
        if char == "(":
            in_brackets = True
        elif char == ")":
            in_brackets = False
        elif not in_brackets:
            outstr += char
    return outstr


def get_naharphotoion_upperlevelids(
    dfenergy_levels_upperion,
    nahar_core_states,
    nahar_configurations_upperion,
    upper_level_ids_of_core_state_id,
    flog,
):
    """Get the upper level id numbers for a given energy level's photoionisation processes."""
    # core_state_id = int(energy_level.corestateid)
    core_state_id = 1  # temporary fix
    if core_state_id > 0 and core_state_id <= len(nahar_core_states):
        if not upper_level_ids_of_core_state_id[core_state_id]:
            # go find matching levels if they haven't been found yet
            nahar_core_state = nahar_core_states[core_state_id - 1]
            nahar_core_state_reduced_configuration = reduce_configuration(
                nahar_core_state.configuration + "_" + nahar_core_state.term
            )
            core_state_energy_ev = nahar_core_state.energyrydberg * ryd_to_ev
            flog.write(
                f"\nMatching core state {core_state_id} '{nahar_core_state.configuration}_{nahar_core_state.term}'"
                f" E={core_state_energy_ev:0.3f} eV to:\n"
            )

            # per configuration: the energy differences and the upper ion level ids
            candidate_upper_levels: dict[str, tuple[list[float], list[int]]] = {}
            for upperlevel in dfenergy_levels_upperion.iter_rows(named=True):
                upperlevelid = upperlevel["levelid"]
                if "levelname" in upperlevel:
                    upperlevelconfig = upperlevel["levelname"]
                else:
                    state_tuple = (
                        int(upperlevel["twosplusone"]),
                        int(upperlevel["l"]),
                        int(upperlevel["parity"]),
                        int(upperlevel["indexinsymmetry"]),
                    )
                    upperlevelconfig = nahar_configurations_upperion.get(state_tuple, "-1")
                energyev = upperlevel["energyabovegsinpercm"] * hc_in_ev_cm

                # this ignores parent term
                if reduce_configuration(upperlevelconfig) == nahar_core_state_reduced_configuration:
                    ediff = energyev - core_state_energy_ev
                    upperlevelconfignoj = upperlevelconfig.split("[")[0]
                    if upperlevelconfignoj not in candidate_upper_levels:
                        candidate_upper_levels[upperlevelconfignoj] = ([], [])
                    candidate_upper_levels[upperlevelconfignoj][1].append(upperlevelid)
                    candidate_upper_levels[upperlevelconfignoj][0].append(ediff)
                    flog.write(
                        f"Upper ion level {upperlevelid} '{upperlevelconfig}' E = {energyev:.4f} E_diff={ediff:.4f}\n"
                    )

            best_ediff = float("inf")
            best_match_upperlevelids: list[int] = []
            for ediffs, upperlevelids in candidate_upper_levels.values():
                avg_ediff = abs(sum(ediffs) / len(ediffs))
                if avg_ediff < best_ediff:
                    best_ediff = avg_ediff
                    best_match_upperlevelids = upperlevelids

            flog.write(f"Best matching levels: {best_match_upperlevelids}\n")

            upper_level_ids_of_core_state_id[core_state_id] = best_match_upperlevelids

            # after matching process, still no upper levels matched!
            if not upper_level_ids_of_core_state_id[core_state_id]:
                upper_level_ids_of_core_state_id[core_state_id] = [0]  # the upper ion's ground state
                artisatomic.log_and_print(
                    flog,
                    "No upper levels matched. Defaulting to the ground state (reduced string:"
                    f" '{nahar_core_state_reduced_configuration}')",
                )

        upperionlevelids = upper_level_ids_of_core_state_id[core_state_id]
    else:
        upperionlevelids = [0]  # the upper ion's ground state

    return upperionlevelids


def get_photoiontargetfractions(
    dfenergy_levels,
    dfenergy_levels_upperion,
    nahar_core_states: list[NaharCoreState],
    nahar_configurations_upperion: dict[tuple[int, int, int, int], str],
    flog,
) -> list[list[tuple[int, float]]]:
    """Resolve each level's photoionisation targets from its core state to upper-ion level ids.

    Returns, per zero-based level id, a list of (upper ion level id, fraction) pairs. A level's
    core state names the upper-ion state it ionises to; where that matches several J-split levels
    of the upper ion, the fraction is shared over them in proportion to their statistical
    weights. Levels whose core state cannot be matched fall back to the upper ion's ground state.
    """
    targetlist: list[list[tuple[int, float]]] = [[] for _ in range(dfenergy_levels.height)]
    upper_level_ids_of_core_state_id = defaultdict(list)
    for lowerlevelid in range(dfenergy_levels.height):
        # find the upper level ids from the Nahar core state
        upperionlevelids = get_naharphotoion_upperlevelids(
            dfenergy_levels_upperion,
            nahar_core_states,
            nahar_configurations_upperion,
            upper_level_ids_of_core_state_id,
            flog,
        )

        summed_statistical_weights = sum(float(dfenergy_levels_upperion["g"][levelid]) for levelid in upperionlevelids)
        for upperionlevelid in sorted(upperionlevelids):
            phixsprobability = dfenergy_levels_upperion["g"][upperionlevelid] / summed_statistical_weights
            targetlist[lowerlevelid].append((upperionlevelid, phixsprobability))

    return targetlist

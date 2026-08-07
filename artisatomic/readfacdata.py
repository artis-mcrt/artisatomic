"""Read levels and transitions from FAC and cFAC output, an early version of the Floers+25 data."""

import re
import string
import typing as t
from collections import defaultdict
from pathlib import Path

import pandas as pd

import artisatomic

USE_CALIBRATED = True

BASEPATH = str(
    Path.home()
    / f"Google Drive/Shared drives/Atomic Data Group/OptimizedFACdata/OptimizedFAC_lanthanides{'_calibrated' if USE_CALIBRATED else ''}"
)


# Constants
me = 9.10938e-28  # grams
NA = 6.0221409e23  # mol^-1
cspeed = 29979245800  # cm/s
kB = 0.6950356  # cm-1 K
echarge = 4.8e-10  # statC
hc = 4.1357e-15 * cspeed


def GetLevels_FAC(filename: Path | str) -> pd.DataFrame:
    """Parse the level table of an FAC ascii output file (fixed-width, FAC column layout)."""
    widths = [(0, 7), (7, 14), (14, 30), (30, 31), (32, 38), (38, 43), (44, 76), (76, 125), (127, 200)]
    names = ["Ilev", "Ibase", "Energy_ev", "P", "VNL", "2J", "Configs_no", "Configs", "Config rel"]

    levels_FAC = pd.read_fwf(filename, header=10, index_col=False, colspecs=widths, names=names, engine="pyarrow")

    levels_FAC["Config"] = levels_FAC["Configs"].apply(lambda x: " ".join(x.split(".")))
    levels_FAC["Config rel"] = levels_FAC["Config rel"].apply(
        lambda x: x.replace(".", " ") if isinstance(x, str) else x
    )
    levels_FAC["g"] = levels_FAC["2J"] + 1

    levels_FAC = levels_FAC[["Ilev", "Config", "Config rel", "P", "2J", "g", "Energy_ev"]]

    levels_FAC["Config"] = levels_FAC["Config"].apply(lambda s: s.replace("1", ""))
    levels_FAC["energypercm"] = levels_FAC["Energy_ev"] / hc

    assert isinstance(levels_FAC, pd.DataFrame)
    return levels_FAC


def GetLevels_cFAC(filename: Path | str) -> pd.DataFrame:
    """Parse the level table of a cFAC ascii output file, whose columns differ from FAC's."""
    widths = [(0, 7), (7, 14), (14, 30), (30, 31), (32, 38), (38, 43), (43, 150)]
    names = ["Ilev", "Ibase", "Energy_ev", "P", "VNL", "2J", "Configs"]

    levels_cFAC = pd.read_fwf(filename, header=10, index_col=False, colspecs=widths, names=names, engine="pyarrow")

    levels_cFAC["Config"] = levels_cFAC["Configs"].apply(lambda x: re.split(r"\s{2,}", x)[0])
    levels_cFAC["Config rel"] = levels_cFAC["Configs"].apply(lambda x: re.split(r"\s{2,}", x)[1])

    levels_cFAC["g"] = levels_cFAC["2J"] + 1

    levels_cFAC = levels_cFAC[["Ilev", "Config", "Config rel", "P", "g", "Energy_ev"]]

    levels_cFAC["Config"] = levels_cFAC["Config"].apply(lambda s: s.replace("1", ""))
    levels_cFAC["energypercm"] = [en_ev / hc for en_ev in levels_cFAC["Energy_ev"]]

    assert isinstance(levels_cFAC, pd.DataFrame)
    return levels_cFAC


def GetLevels(filename: Path | str, ionization_energy_in_ev: float) -> pd.DataFrame:
    """Get a dataframe of the energy levels extracted from ascii level output of cFAC and csv and dat files.

    Parameters
    ----------
    filename : str
        Filename of cFAC ascii output for the energy levels
    """
    headerlines: list[str] = []
    with Path(filename).open(encoding="utf-8") as f:
        headerlines.extend(f.readline() for _ in range(10))

    # headerlines[7] holds the ground state and headerlines[5] the ion charge; neither is needed here
    version_FAC = headerlines[0].split(" ")[0]
    print("FAC/cFAC: ", version_FAC)
    if version_FAC == "FAC":
        levels = GetLevels_FAC(filename)
    elif version_FAC == "cFAC":
        levels = GetLevels_cFAC(filename)
    else:
        msg = "No FAC-like code detected on output file"
        raise ValueError(msg)

    levels = levels[levels["energypercm"] <= (ionization_energy_in_ev / hc)]
    assert isinstance(levels, pd.DataFrame)

    return levels


def GetLines_FAC(filename: Path | str) -> pd.DataFrame:
    """Parse the transition table of an FAC ascii output file."""
    names = ["Upper", "2J1", "Lower", "2J2", "DeltaE[eV]", "gf", "A", "Monopole"]

    widths = [(0, 7), (7, 11), (11, 17), (17, 21), (21, 35), (35, 49), (49, 63), (63, 77)]
    trans_FAC = pd.read_fwf(filename, header=11, index_col=False, colspecs=widths, names=names, engine="pyarrow")
    trans_FAC["Wavelength[Ang]"] = trans_FAC["DeltaE[eV]"].apply(lambda en_ev: (hc / en_ev) * 1e8)
    trans_FAC["DeltaE[cm^-1]"] = trans_FAC["DeltaE[eV]"] / hc
    trans_FAC["A"] = trans_FAC["A"].apply(lambda tr: float(tr.rstrip(" -")))
    trans_FAC = trans_FAC[["Upper", "Lower", "DeltaE[eV]", "DeltaE[cm^-1]", "Wavelength[Ang]", "gf", "A"]]
    assert isinstance(trans_FAC, pd.DataFrame)
    return trans_FAC


def GetLines_cFAC(filename: Path | str) -> pd.DataFrame:
    """Parse the transition table of a cFAC ascii output file."""
    names = ["Upper", "2J1", "Lower", "2J2", "DeltaE[eV]", "UTAdiff", "gf", "A", "Monopole"]

    widths = [(0, 6), (6, 10), (10, 16), (16, 21), (21, 35), (35, 47), (47, 61), (61, 75), (75, 89)]
    trans_cFAC = pd.read_fwf(filename, header=11, index_col=False, colspecs=widths, names=names, engine="pyarrow")
    trans_cFAC["Wavelength[Ang]"] = trans_cFAC["DeltaE[eV]"].apply(lambda en_ev: (hc / en_ev) * 1e8)
    trans_cFAC["DeltaE[cm^-1]"] = trans_cFAC["DeltaE[eV]"] / hc
    trans_cFAC = trans_cFAC[["Upper", "Lower", "DeltaE[eV]", "DeltaE[cm^-1]", "Wavelength[Ang]", "gf", "A"]]
    assert isinstance(trans_cFAC, pd.DataFrame)
    return trans_cFAC.astype({"Upper": "int64", "Lower": "int64"})


def GetLines(filename: Path | str) -> pd.DataFrame:
    """Get a dataframe of the transitions extracted from ascii level output of cFAC and csv and dat files.

    Parameters
    ----------
    filename : str
        Filename of cFAC ascii output for the transitions
    """
    headerlines: list[str] = []
    with Path(filename).open(encoding="utf-8") as f:
        headerlines.extend(f.readline() for _ in range(11))
    # headerlines[8], [10] and [5] hold the ground state, multipole and ion charge; none is needed here
    version_FAC = headerlines[0].split(" ")[0]

    if version_FAC == "FAC":
        lines = GetLines_FAC(filename)
    elif version_FAC == "cFAC":
        lines = GetLines_cFAC(filename)
    else:
        msg = "No FAC-like code detected on output file"
        raise ValueError(msg)

    return lines


def extend_ion_list(ion_handlers):
    """Add every ion with an FAC data file to ion_handlers under the "fac" handler."""
    assert Path(BASEPATH).is_dir()
    for s in Path(BASEPATH).glob("**/*.lev.asc"):
        ionstr = s.parts[-1].lstrip(string.digits).removesuffix(".lev.asc").removesuffix("_calib")
        atomic_number, ion_stage = artisatomic.split_element_ionstage_str(ionstr)
        ion_handlers = artisatomic.add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "fac")

    # add_handler_if_not_set() keeps the list sorted by atomic number, matching the other readers
    return ion_handlers


class FACEnergyLevel(t.NamedTuple):
    """One energy level of an FAC calculation."""

    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: int


def read_levels_data(dflevels):
    """Convert the FAC level table to level tuples, in the energy order of the sorted frame.

    Also returns the map from the file's Ilev to the zero-based level id, which read_lines_data()
    needs because sorting by energy reorders the levels.
    """
    energy_levels = []
    ilev_enlevelindex_map = {}

    dflevels = dflevels.sort_values(by="energypercm", ignore_index=True)
    for index, row in dflevels.iterrows():
        ilev_enlevelindex_map[int(row["Ilev"])] = index

        # Config is not unique (levels of one configuration differ in J), so append the FAC level
        # index. The configuration stays first, for get_level_valence_n() and the adata.txt comment.
        newlevel = FACEnergyLevel(
            levelname=f"{row['Config']} Ilev={int(row['Ilev'])}",
            parity=row["P"],
            g=row["g"],
            energyabovegsinpercm=float(row["energypercm"]),
        )
        energy_levels.append(newlevel)

    # a duplicated Ilev would overwrite its map entry and misroute every transition referencing
    # it. Not an assert: input validation must survive python -O.
    if len(ilev_enlevelindex_map) != len(energy_levels):
        msg = f"Duplicate Ilev values in FAC levels file: {len(energy_levels)} rows but only {len(ilev_enlevelindex_map)} unique Ilev"
        raise ValueError(msg)

    return energy_levels, ilev_enlevelindex_map


class FACTransition(t.NamedTuple):
    """One bound-bound transition of an FAC calculation, keyed by zero-based level id."""

    lowerlevel: int
    upperlevel: int
    A: float
    coll_str: float


def read_lines_data(energy_levels, dflines, ilev_enlevelindex_map):
    """Convert FAC lines to transitions referencing zero-based level ids.

    Lines referencing an Ilev with no level are skipped. Returns the transitions and the number
    of them touching each level name.
    """
    transitions = []
    transition_count_of_level_name = defaultdict(int)

    for _, row in dflines.iterrows():
        try:
            lowerlevel = ilev_enlevelindex_map[int(row["Lower"])]
            upperlevel = ilev_enlevelindex_map[int(row["Upper"])]
        except KeyError:
            continue
        assert lowerlevel < upperlevel

        transtuple = FACTransition(lowerlevel=lowerlevel, upperlevel=upperlevel, A=row["A"], coll_str=-1)

        transition_count_of_level_name[energy_levels[lowerlevel].levelname] += 1
        transition_count_of_level_name[energy_levels[upperlevel].levelname] += 1

        transitions.append(transtuple)

    return transitions, transition_count_of_level_name


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the FAC data set, an early version of the Floers+25 calibrated data."""
    elsym = artisatomic.elsymbols[atomic_number]
    ion_stage_roman = artisatomic.roman_numerals[ion_stage]

    ionstr = f"{atomic_number}{elsym}{ion_stage_roman}{'_calib' if USE_CALIBRATED else ''}"
    ion_folder = BASEPATH + f"/{ionstr}"
    levels_file = ion_folder + f"/{ionstr}.lev.asc"
    lines_file = ion_folder + f"/{ionstr}.tr.asc"

    if atomic_number == 92 and ion_stage in {2, 3}:
        ion_folder = str(
            Path.home()
            / f"Google Drive/Shared drives/Atomic Data Group/Paper_Nd_U/FAC/{elsym}{ion_stage_roman}_convergence_t22_n30_calibrated"
        )
        levels_file = f"{ion_folder}/{elsym}{ion_stage_roman}_convergence_t22_n30_calibrated.lev.asc"
        lines_file = f"{ion_folder}/{elsym}{ion_stage_roman}_convergence_t22_n30_calibrated.tr.asc"

    artisatomic.log_and_print(
        flog,
        f"Reading FAC/cFAC data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman}) from"
        f" {artisatomic.path_for_log(ion_folder)}",
    )

    ionization_energy_in_ev = artisatomic.get_nist_ionization_energies_ev()[atomic_number, ion_stage]

    assert Path(levels_file).exists()
    dflevels = GetLevels(filename=levels_file, ionization_energy_in_ev=ionization_energy_in_ev)

    # map associates source file level numbers with energy-sorted level numbers (0 indexed)
    energy_levels, ilev_enlevelindex_map = read_levels_data(dflevels)

    artisatomic.log_and_print(flog, f"Read {len(energy_levels):d} levels")

    assert Path(lines_file).exists()
    dflines = GetLines(filename=lines_file)

    transitions, transition_count_of_level_name = read_lines_data(energy_levels, dflines, ilev_enlevelindex_map)

    artisatomic.log_and_print(flog, f"Read {len(transitions)} transitions")

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name


def get_level_valence_n(levelname: str):
    """Principal quantum number of the valence electron, read from an FAC level name.

    Kept separate from the other readers' versions: each data source names its levels
    differently, so a shared parser would have to guess which convention it is looking at.
    """
    # level names are "<configuration> Ilev=<index>" and the configuration is itself
    # space-separated, so drop the index suffix before taking the last orbital
    part = levelname.split(" Ilev=", maxsplit=1)[0].rsplit(" ", maxsplit=1)[-1]
    if part[-1] not in "spdfg":
        # end of string is a number of electrons in the orbital, not a principal quantum number, so remove it
        assert part[-1].isdigit()
        part = part.rstrip(string.digits)
    return int(part.rstrip("spdfg"))

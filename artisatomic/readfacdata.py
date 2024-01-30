import re
from collections import defaultdict
from collections import namedtuple
from pathlib import Path

import pandas as pd

import artisatomic

# from astropy import units as u
# import os.path
# import pandas as pd
# from carsus.util import parse_selected_species

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


def GetLevels_FAC(filename: Path) -> pd.DataFrame:
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

    return levels_FAC


def GetLevels_cFAC(filename: Path) -> pd.DataFrame:
    widths = [(0, 7), (7, 14), (14, 30), (30, 31), (32, 38), (38, 43), (43, 150)]
    names = ["Ilev", "Ibase", "Energy_ev", "P", "VNL", "2J", "Configs"]

    levels_cFAC = pd.read_fwf(filename, header=10, index_col=False, colspecs=widths, names=names, engine="pyarrow")

    levels_cFAC["Config"] = levels_cFAC["Configs"].apply(lambda x: re.split(r"\s{2,}", x)[0])
    levels_cFAC["Config rel"] = levels_cFAC["Configs"].apply(lambda x: re.split(r"\s{2,}", x)[1])

    levels_cFAC["g"] = levels_cFAC["2J"] + 1

    levels_cFAC = levels_cFAC[["Ilev", "Config", "Config rel", "P", "g", "Energy_ev"]]

    levels_cFAC["Config"] = levels_cFAC["Config"].apply(lambda s: s.replace("1", ""))
    levels_cFAC["energypercm"] = [en_ev / hc for en_ev in levels_cFAC["Energy_ev"]]

    return levels_cFAC


def GetLevels(filename: Path, Z: int, ionization_energy_in_ev: float) -> pd.DataFrame:
    """Returns a dataframe of the energy levels extracted from ascii level output of cFAC and csv and dat files of the data.

    Parameters
    ----------
    data : str
        Filename of cFAC ascii output for the energy levels

    Z: int
        Ion atomic number

    """
    headerlines: list[str] = []
    with open(filename) as f:
        headerlines.extend(f.readline() for _ in range(10))

    GState = headerlines[7][8:]
    IonStage = Z - int(float(headerlines[5][6:].strip()))
    version_FAC = headerlines[0].split(" ")[0]
    print("FAC/cFAC: ", version_FAC)
    if version_FAC == "FAC":
        levels = GetLevels_FAC(filename)
    elif version_FAC == "cFAC":
        levels = GetLevels_cFAC(filename)
    else:
        raise ValueError("No FAC-like code detected on output file")

    levels = levels[levels["energypercm"] <= (ionization_energy_in_ev / hc)]

    return levels


def GetLines_FAC(filename: Path) -> pd.DataFrame:
    names = ["Upper", "2J1", "Lower", "2J2", "DeltaE[eV]", "gf", "A", "Monopole"]

    widths = [(0, 7), (7, 11), (11, 17), (17, 21), (21, 35), (35, 49), (49, 63), (63, 77)]
    trans_FAC = pd.read_fwf(filename, header=11, index_col=False, colspecs=widths, names=names, engine="pyarrow")
    trans_FAC["Wavelength[Ang]"] = trans_FAC["DeltaE[eV]"].apply(lambda en_ev: (hc / en_ev) * 1e8)
    trans_FAC["DeltaE[cm^-1]"] = trans_FAC["DeltaE[eV]"] / hc
    trans_FAC["A"] = trans_FAC["A"].apply(lambda tr: float(tr.rstrip(" -")))
    trans_FAC = trans_FAC[["Upper", "Lower", "DeltaE[eV]", "DeltaE[cm^-1]", "Wavelength[Ang]", "gf", "A"]]
    return trans_FAC


def GetLines_cFAC(filename: Path) -> pd.DataFrame:
    names = ["Upper", "2J1", "Lower", "2J2", "DeltaE[eV]", "UTAdiff", "gf", "A", "Monopole"]

    widths = [(0, 6), (6, 10), (10, 16), (16, 21), (21, 35), (35, 47), (47, 61), (61, 75), (75, 89)]
    trans_cFAC = pd.read_fwf(filename, header=11, index_col=False, colspecs=widths, names=names, engine="pyarrow")
    trans_cFAC["Wavelength[Ang]"] = trans_cFAC["DeltaE[eV]"].apply(lambda en_ev: (hc / en_ev) * 1e8)
    trans_cFAC["DeltaE[cm^-1]"] = trans_cFAC["DeltaE[eV]"] / hc
    trans_cFAC = trans_cFAC[["Upper", "Lower", "DeltaE[eV]", "DeltaE[cm^-1]", "Wavelength[Ang]", "gf", "A"]]
    trans_cFAC = trans_cFAC.astype({"Upper": "int64", "Lower": "int64"})
    return trans_cFAC


def GetLines(filename: Path, Z: int) -> pd.DataFrame:
    """Returns a dataframe of the transitions extracted from ascii level output of cFAC and csv and dat files of the data.

    Parameters
    ----------
    data : str
        Filename of cFAC ascii output for the transitions

    Z: int
        Ion atomic number

    """
    headerlines: list[str] = []
    with open(filename) as f:
        headerlines.extend(f.readline() for _ in range(11))
    GState = headerlines[8][8:]
    Multi = headerlines[10][9:]
    IonStage = Z - int(float(headerlines[5][6:].strip()))
    version_FAC = headerlines[0].split(" ")[0]

    if version_FAC == "FAC":
        lines = GetLines_FAC(filename)
    elif version_FAC == "cFAC":
        lines = GetLines_cFAC(filename)
    else:
        raise ValueError("No FAC-like code detected on output file")

    return lines


def extend_ion_list(ion_handlers):
    for s in Path(BASEPATH).glob("**/*.lev.asc"):
        ionstr = s.parts[-1].lstrip("0123456789").removesuffix(".lev.asc").removesuffix("_calib")
        elsym = ionstr.rstrip("IVX")
        ion_stage_roman = ionstr.removeprefix(elsym)
        atomic_number = artisatomic.elsymbols.index(elsym)

        ion_stage = artisatomic.roman_numerals.index(ion_stage_roman)

        found_element = False
        for tmp_atomic_number, list_ions in ion_handlers:
            if tmp_atomic_number == atomic_number:
                if ion_stage not in [x[0] if len(x) > 0 else x for x in list_ions]:
                    list_ions.append((ion_stage, "fac"))
                    list_ions.sort()
                found_element = True
        if not found_element:
            ion_handlers.append(
                (
                    atomic_number,
                    [(ion_stage, "fac")],
                )
            )

    ion_handlers.sort(key=lambda x: x[0])
    # print(ion_handlers)
    return ion_handlers


def read_levels_data(dflevels):
    energy_level_tuple = namedtuple("energylevel", "levelname energyabovegsinpercm g parity")

    energy_levels = []
    ilev_enlevelindex_map = {}

    dflevels = dflevels.sort_values(by="energypercm", ignore_index=True)
    for index, row in dflevels.iterrows():
        ilev_enlevelindex_map[int(row["Ilev"])] = index

        newlevel = energy_level_tuple(
            levelname=row["Config rel"], parity=row["P"], g=row["g"], energyabovegsinpercm=float(row["energypercm"])
        )
        energy_levels.append(newlevel)

    energy_levels.sort(key=lambda x: x.energyabovegsinpercm)

    return [None, *energy_levels], ilev_enlevelindex_map


def read_lines_data(energy_levels, dflines, ilev_enlevelindex_map, flog):
    transitions = []
    transition_count_of_level_name = defaultdict(int)
    transitiontuple = namedtuple("transition", "lowerlevel upperlevel A coll_str")

    for index, row in dflines.iterrows():
        try:
            lowerlevel = ilev_enlevelindex_map[int(row["Lower"])] + 1
            upperlevel = ilev_enlevelindex_map[int(row["Upper"])] + 1
        except KeyError:
            continue
        assert lowerlevel < upperlevel

        transtuple = transitiontuple(lowerlevel=lowerlevel, upperlevel=upperlevel, A=row["A"], coll_str=-1)

        transition_count_of_level_name[energy_levels[lowerlevel].levelname] += 1
        transition_count_of_level_name[energy_levels[upperlevel].levelname] += 1

        transitions.append(transtuple)

    return transitions, transition_count_of_level_name


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    # ion_charge = ion_stage - 1
    elsym = artisatomic.elsymbols[atomic_number]
    ion_stage_roman = artisatomic.roman_numerals[ion_stage]

    ionstr = f"{atomic_number}{elsym}{ion_stage_roman}{'_calib' if USE_CALIBRATED else ''}"
    ion_folder = BASEPATH + f"/{ionstr}"
    levels_file = ion_folder + f"/{ionstr}.lev.asc"
    lines_file = ion_folder + f"/{ionstr}.tr.asc"

    if atomic_number == 92 and ion_stage in [2, 3]:
        ion_folder = str(
            Path.home()
            / f"Google Drive/Shared drives/Atomic Data Group/Paper_Nd_U/FAC/{elsym}{ion_stage_roman}_convergence_t22_n30_calibrated"
        )
        levels_file = f"{ion_folder}/{elsym}{ion_stage_roman}_convergence_t22_n30_calibrated.lev.asc"
        lines_file = f"{ion_folder}/{elsym}{ion_stage_roman}_convergence_t22_n30_calibrated.tr.asc"

    artisatomic.log_and_print(
        flog,
        f"Reading FAC/cFAC data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman}) from"
        f" {ion_folder}",
    )

    ionization_energy_in_ev = artisatomic.get_nist_ionization_energies_ev()[(atomic_number, ion_stage)]

    assert Path(levels_file).exists()
    dflevels = GetLevels(filename=levels_file, Z=atomic_number, ionization_energy_in_ev=ionization_energy_in_ev)
    # print(dflevels)

    # map associates source file level numbers with energy-sorted level numbers (0 indexed)
    energy_levels, ilev_enlevelindex_map = read_levels_data(dflevels)

    artisatomic.log_and_print(flog, f"Read {len(energy_levels[1:]):d} levels")

    assert Path(lines_file).exists()
    dflines = GetLines(filename=lines_file, Z=atomic_number)

    transitions, transition_count_of_level_name = read_lines_data(energy_levels, dflines, ilev_enlevelindex_map, flog)

    artisatomic.log_and_print(flog, f"Read {len(transitions)} transitions")

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name

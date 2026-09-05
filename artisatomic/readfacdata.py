"""Read levels and transitions from FAC and cFAC output, an early version of the Floers+25 data."""

import os
import re
import string
from pathlib import Path

import pandas as pd

from artisatomic.base import add_handler_if_not_set
from artisatomic.base import elsymbols
from artisatomic.base import EnergyLevel
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import hc_in_ev_cm
from artisatomic.base import levelid_of_fileindex_map
from artisatomic.base import log_and_print
from artisatomic.base import path_for_log
from artisatomic.base import resolve_transition_levelids
from artisatomic.base import roman_numerals
from artisatomic.base import split_element_ionstage_str
from artisatomic.base import Transition
from artisatomic.levelnames import parse_orbital_n

USE_CALIBRATED = True


def get_basepath() -> Path:
    """Return the directory that holds the OptimizedFAC lanthanide data.

    The data is not part of this repository. ARTISATOMIC_FAC_PATH overrides the directory that
    this function searches, as ARTISATOMIC_LISBON_PATH does for the Lisbon files. The default is
    the Google Drive mount path. That is where the shared drive appears on the machine of the
    original author.
    """
    calibstr = "_calibrated" if USE_CALIBRATED else ""
    default = Path.home() / "Google Drive/Shared drives/Atomic Data Group/OptimizedFACdata"
    return Path(os.environ.get("ARTISATOMIC_FAC_PATH", default)) / f"OptimizedFAC_lanthanides{calibstr}"


def GetLevels_FAC(filename: Path | str) -> pd.DataFrame:
    """Parse the level table of an FAC ascii output file (fixed-width, FAC column layout)."""
    widths = [(0, 7), (7, 14), (14, 30), (30, 31), (32, 38), (38, 43), (44, 76), (76, 125), (127, 200)]
    names = ["Ilev", "Ibase", "Energy_ev", "P", "VNL", "2J", "Configs_no", "Configs", "Config rel"]

    levels_FAC = pd.read_fwf(filename, header=10, index_col=False, colspecs=widths, names=names, engine="pyarrow")

    levels_FAC["Config"] = levels_FAC["Configs"].apply(lambda x: " ".join(x.split(".")))
    return finish_levels(levels_FAC)


def GetLevels_cFAC(filename: Path | str) -> pd.DataFrame:
    """Parse the level table of a cFAC ascii output file, whose columns differ from FAC's."""
    widths = [(0, 7), (7, 14), (14, 30), (30, 31), (32, 38), (38, 43), (43, 150)]
    names = ["Ilev", "Ibase", "Energy_ev", "P", "VNL", "2J", "Configs"]

    levels_cFAC = pd.read_fwf(filename, header=10, index_col=False, colspecs=widths, names=names, engine="pyarrow")

    levels_cFAC["Config"] = levels_cFAC["Configs"].apply(lambda x: re.split(r"\s{2,}", x)[0])
    return finish_levels(levels_cFAC)


def finish_levels(levels: pd.DataFrame) -> pd.DataFrame:
    """Derive the columns that read_levels_data() takes, the same way for the FAC and cFAC layouts."""
    levels["g"] = levels["2J"] + 1
    # remove only a lone occupation of 1 ("6s1" -> "6s"); occupations of 10-14 keep their digits
    levels["Config"] = levels["Config"].apply(lambda s: re.sub(r"(?<=[spdfg])1(?![0-9])", "", s))
    levels["energypercm"] = levels["Energy_ev"] / hc_in_ev_cm

    levels = levels[["Ilev", "Config", "P", "g", "Energy_ev", "energypercm"]]
    assert isinstance(levels, pd.DataFrame)
    return levels


def GetLevels(filename: Path | str) -> pd.DataFrame:
    """Get a dataframe of every energy level in the ascii level output of FAC or cFAC.

    The caller drops the levels above the ionisation energy and keeps their Ilev values. The
    values show whether a transition names a dropped level or an unknown level.
    """
    headerlines: list[str] = []
    with Path(filename).open(encoding="utf-8") as f:
        headerlines.extend(f.readline() for _ in range(10))

    # headerlines[7] holds the ground state and headerlines[5] the ion charge. This function needs neither.
    version_FAC = headerlines[0].split(" ")[0]
    print("FAC/cFAC: ", version_FAC)
    if version_FAC == "FAC":
        levels = GetLevels_FAC(filename)
    elif version_FAC == "cFAC":
        levels = GetLevels_cFAC(filename)
    else:
        msg = "No FAC-like code detected on output file"
        raise ValueError(msg)

    return levels


def GetLines_FAC(filename: Path | str) -> pd.DataFrame:
    """Parse the transition table of an FAC ascii output file."""
    names = ["Upper", "2J1", "Lower", "2J2", "DeltaE[eV]", "gf", "A", "Monopole"]

    widths = [(0, 7), (7, 11), (11, 17), (17, 21), (21, 35), (35, 49), (49, 63), (63, 77)]
    trans_FAC = pd.read_fwf(filename, header=11, index_col=False, colspecs=widths, names=names, engine="pyarrow")
    # read_fwf() infers the A column as float64 when no row carries the leading "-" of a
    # negative Monopole in the last column. It infers str when one row does. Only the str form
    # needs the "-" removed.
    if not pd.api.types.is_numeric_dtype(trans_FAC["A"]):
        trans_FAC["A"] = pd.to_numeric(trans_FAC["A"].str.rstrip(" -"))
    trans_FAC = trans_FAC[["Upper", "Lower", "A"]]
    assert isinstance(trans_FAC, pd.DataFrame)
    return trans_FAC


def GetLines_cFAC(filename: Path | str) -> pd.DataFrame:
    """Parse the transition table of a cFAC ascii output file."""
    names = ["Upper", "2J1", "Lower", "2J2", "DeltaE[eV]", "UTAdiff", "gf", "A", "Monopole"]

    widths = [(0, 6), (6, 10), (10, 16), (16, 21), (21, 35), (35, 47), (47, 61), (61, 75), (75, 89)]
    trans_cFAC = pd.read_fwf(filename, header=11, index_col=False, colspecs=widths, names=names, engine="pyarrow")
    trans_cFAC = trans_cFAC[["Upper", "Lower", "A"]]
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
    # headerlines[8], [10] and [5] hold the ground state, multipole and ion charge. This function needs none.
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
    basepath = get_basepath()
    # not an assert: this reports a missing data directory and must survive python -O. It also
    # names the environment variable, which a silent failure of the glob would not
    if not basepath.is_dir():
        msg = (
            f"FAC data directory {basepath} not found."
            " Set ARTISATOMIC_FAC_PATH to the directory holding the OptimizedFAC_lanthanides* folders."
        )
        raise FileNotFoundError(msg)

    for s in basepath.glob("**/*.lev.asc"):
        ionstr = s.parts[-1].lstrip(string.digits).removesuffix(".lev.asc").removesuffix("_calib")
        atomic_number, ion_stage = split_element_ionstage_str(ionstr)
        ion_handlers = add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "fac")

    # add_handler_if_not_set() keeps the list sorted by atomic number, matching the other readers
    return ion_handlers


def read_levels_data(dflevels):
    """Convert the FAC level table to level tuples, in the energy order of the sorted frame.

    Also returns the map from the file's Ilev to the zero-based level id, which read_lines_data()
    needs because the sort by energy reorders the levels.
    """
    # astype(float) first: the level order is now the frame's order alone. A column that arrived
    # as strings (pd.read_fwf can yield those) would sort lexicographically. The sort is stable, so
    # levels of one energy keep the file's order. Their ids then do not depend on the sort
    # algorithm.
    dflevels = dflevels.astype({"energypercm": float}).sort_values(by="energypercm", kind="stable", ignore_index=True)

    energy_levels = [
        # Config is not unique (levels of one configuration differ in J), so append the FAC level
        # index. The configuration stays first, for get_level_valence_n() and the adata.txt comment.
        EnergyLevel(
            levelname=f"{row['Config']} Ilev={int(row['Ilev'])}",
            parity=row["P"],
            g=row["g"],
            energyabovegsinpercm=float(row["energypercm"]),
        )
        for _index, row in dflevels.iterrows()
    ]

    return energy_levels, levelid_of_fileindex_map(dflevels["Ilev"], "the FAC levels file")


def read_lines_data(dflines, ilev_enlevelindex_map, ilevs_above_ionization: set[int], flog):
    """Convert FAC lines to transitions referencing zero-based level ids.

    The reader skips a line that names a level above the ionisation energy, because the level
    list stops there. A line that names an Ilev that the level file does not have is an error.
    The two files then disagree about the numbering, and a skip would empty the ion without a
    message. The reader orders the two levels with the lower id first.
    """
    transitions = []
    skipped_count = 0

    for _, row in dflines.iterrows():
        if int(row["Lower"]) in ilevs_above_ionization or int(row["Upper"]) in ilevs_above_ionization:
            skipped_count += 1
            continue

        # not an assert: this decides between which levels the output writes a transition, so it
        # must survive python -O. It also names the offending Ilev values instead of a bare failure
        lowerlevel, upperlevel = resolve_transition_levelids(
            row["Lower"], row["Upper"], ilev_enlevelindex_map, "the FAC transitions file"
        )

        transitions.append(Transition(lowerlevel=lowerlevel, upperlevel=upperlevel, A=row["A"]))

    if skipped_count > 0:
        log_and_print(
            flog, f"WARNING: skipped {skipped_count:d} transitions that reference a level above the ionization energy"
        )

    return transitions


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the FAC data set, an early version of the Floers+25 calibrated data."""
    elsym = elsymbols[atomic_number]
    ion_stage_roman = roman_numerals[ion_stage]

    ionstr = f"{atomic_number}{elsym}{ion_stage_roman}{'_calib' if USE_CALIBRATED else ''}"
    ion_folder = get_basepath() / ionstr
    levels_file = ion_folder / f"{ionstr}.lev.asc"
    lines_file = ion_folder / f"{ionstr}.tr.asc"

    if atomic_number == 92 and ion_stage in {2, 3}:
        # U II and U III come from a separate convergence study, which sits beside the
        # OptimizedFAC folders rather than inside them
        ionstr = f"{elsym}{ion_stage_roman}_convergence_t22_n30_calibrated"
        ion_folder = get_basepath().parent.parent / "Paper_Nd_U" / "FAC" / ionstr
        levels_file = ion_folder / f"{ionstr}.lev.asc"
        lines_file = ion_folder / f"{ionstr}.tr.asc"

    log_and_print(
        flog,
        f"Reading FAC/cFAC data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman}) from"
        f" {path_for_log(ion_folder)}",
    )

    ionization_energy_in_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]

    if not levels_file.is_file():
        msg = f"FAC levels file {levels_file} not found"
        raise FileNotFoundError(msg)
    dfalllevels = GetLevels(filename=levels_file)
    # drop the levels above the ionisation energy, but keep their Ilev values. With them,
    # read_lines_data() knows whether a transition names a dropped level or an unknown level
    above_ionization = dfalllevels["energypercm"] > (ionization_energy_in_ev / hc_in_ev_cm)
    ilevs_above_ionization = {int(ilev) for ilev in dfalllevels.loc[above_ionization, "Ilev"]}
    dflevels = dfalllevels.loc[~above_ionization]

    # map associates source file level numbers with energy-sorted level numbers (0 indexed)
    energy_levels, ilev_enlevelindex_map = read_levels_data(dflevels)

    log_and_print(flog, f"Read {len(energy_levels):d} levels")

    if not lines_file.is_file():
        msg = f"FAC transitions file {lines_file} not found"
        raise FileNotFoundError(msg)
    dflines = GetLines(filename=lines_file)

    transitions = read_lines_data(dflines, ilev_enlevelindex_map, ilevs_above_ionization, flog)

    log_and_print(flog, f"Read {len(transitions)} transitions")

    return ionization_energy_in_ev, energy_levels, transitions


def get_level_valence_n(levelname: str) -> int | None:
    """Principal quantum number of the valence electron, read from an FAC level name.

    Returns None when it cannot parse the name. The caller, match_hydrogenic_phixs(), then
    gives the level no estimate and writes a warning to the ion log.

    This parser stays separate from the versions of the other readers. Each data source names
    its levels differently, so a shared parser would have to guess the convention of the name.
    """
    # level names are "<configuration> Ilev=<index>", and the configuration itself contains
    # spaces. Drop the index suffix first, then take the last orbital
    part = levelname.split(" Ilev=", maxsplit=1)[0].rsplit(" ", maxsplit=1)[-1]
    return parse_orbital_n(part)

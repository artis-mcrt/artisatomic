"""Read levels and transitions from the Lisbon Atomic Group data set (the "lisbon" handler)."""

import os
import typing as t
from collections import defaultdict
from pathlib import Path

import pandas as pd

from artisatomic.base import elsymbols
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import gf_to_a_coefficient
from artisatomic.base import levelid_of_fileindex_map
from artisatomic.base import log_and_print
from artisatomic.base import PYDIR
from artisatomic.base import resolve_transition_levelids
from artisatomic.base import roman_numerals


class LisbonReader:
    """Extract levels and lines from the Lisbon Atomic Group data.

    Copied from Andreas Floers' code in git.gsi.de:nucastro/opacities.git, and mimics the
    GFALLReader class.

    Attributes
    ----------
    levels : DataFrame
    lines : DataFrame
    """

    def __init__(self, data) -> None:
        """Store the Lisbon data.

        Parameters
        ----------
        data : dict
            Dictionary containing one dictionary per species with
            keys `levels` and `lines`.
        """
        self._get_levels_lines(data)

    def _get_levels_lines(self, data):
        """Generate the `levels` and `lines` DataFrames.

        Parameters
        ----------
        data : dict
            Dictionary containing one dictionary per species with
            keys `atomic_number`, `ion_charge`, `levels` and `lines`.
        """
        lvl_list = []
        lns_list = []
        # the caller passes the element and the charge directly. The alternative was to format
        # them into the key and to parse them back with carsus.util.parse_selected_species(),
        # which made an undeclared optional dependency a hard requirement of this reader.
        for parser in data.values():
            atomic_number = parser["atomic_number"]
            ion_charge = parser["ion_charge"]
            levels_data = pd.read_csv(parser["levels"], skiprows=8, index_col=0)
            levels = pd.DataFrame()
            levels["energy"] = levels_data["Energy[cm^-1]"]
            levels["j"] = 0.5 * (levels_data["g"] - 1)
            levels["label"] = levels_data["RelConfig"]
            levels["atomic_number"] = atomic_number
            levels["ion_charge"] = ion_charge
            levels["level_index"] = levels.index
            levels = levels.set_index(["atomic_number", "ion_charge", "level_index"])
            lvl_list.append(levels)

            lines_data = pd.read_csv(parser["lines"], skiprows=8)  # index_col=0
            lines = pd.DataFrame()
            lines["level_index_lower"] = lines_data["Lower"]
            lines["level_index_upper"] = lines_data["Upper"]
            lines["atomic_number"] = atomic_number
            lines["ion_charge"] = ion_charge
            lines["gf"] = lines_data["gf"]
            lines["j_upper"] = levels.iloc[lines["level_index_upper"]]["j"].to_numpy()
            # keep the wavelength in Angstrom: the gf-to-A constant in
            # read_levels_and_transitions() expects Angstrom
            lines["wavelength"] = lines_data["Wavelength[Ang]"]
            lines = lines.set_index(["atomic_number", "ion_charge", "level_index_lower", "level_index_upper"])
            lns_list.append(lines)
        levels = pd.concat(lvl_list)
        # pd.concat() of the untyped list above narrows to Never, so a type checker reads this as dead
        lines = pd.concat(lns_list)
        self.levels = levels
        self.lines = lines


def get_levelname(row):
    """Name a Lisbon level from its label and J, since the label alone is not unique."""
    return f"{row.label}, j={row.j}"


def read_levels_data(dflevels):
    """Convert the Lisbon level table to level tuples, sorted by energy.

    Also returns the map from the file's level index to the zero-based level id, which
    read_lines_data() needs because sorting by energy reorders the levels.

    The lines name their levels POSITIONALLY: LisbonReader reads their energies with
    levels.iloc[lines["level_index_lower"]], which is a position in the file-ordered frame, not an
    index label. So the map is keyed by that position, which reset_index() below makes explicit
    rather than relying on the levels CSV happening to be numbered from zero.

    This data set supplies no parities, so every level's parity is null and the Laporte rule
    never fires. It does supply J, which its level names are keyed on, so the delta J rule is
    what decides forbidden-ness here.
    """
    # sort first, so that the ids handed out below are the ones the levels keep. The index is
    # reset to the row position beforehand, so sorting leaves each level carrying its file position
    dflevels = dflevels.reset_index(drop=True).sort_values(by="energy", kind="stable")

    energy_levels = [
        EnergyLevelTuple(
            levelname=get_levelname(row),
            parity=None,  # no parity in this data set, so the Laporte rule cannot fire
            j=float(row.j),
            g=2 * row.j + 1,
            energyabovegsinpercm=float(row.energy),
        )
        for levelid, (_fileposition, row) in enumerate(dflevels.iterrows())
    ]

    return energy_levels, levelid_of_fileindex_map(dflevels.index, "the Lisbon levels file")


def read_lines_data(energy_levels, dflines, levelid_of_fileindex):
    """Convert Lisbon lines to transitions referencing zero-based level ids.

    The lines name their levels by position in the levels file, and read_levels_data() sorted the
    levels by energy, so every one is mapped through levelid_of_fileindex rather than used
    directly. A line naming a level that does not exist is an error, not something to skip.

    Returns the transitions and the number of them touching each level name.
    """
    transitions = []
    transition_count_of_level_name = defaultdict(int)

    for (fileindex_lower, fileindex_upper), row in dflines.iterrows():
        lowerlevel, upperlevel = resolve_transition_levelids(
            fileindex_lower, fileindex_upper, levelid_of_fileindex, "the Lisbon transitions file"
        )

        transtuple = TransitionTuple(lowerlevel=lowerlevel, upperlevel=upperlevel, A=row.A, coll_str=-1)

        transition_count_of_level_name[energy_levels[lowerlevel].levelname] += 1
        transition_count_of_level_name[energy_levels[upperlevel].levelname] += 1

        transitions.append(transtuple)

    return transitions, transition_count_of_level_name


class EnergyLevelTuple(t.NamedTuple):
    """One Lisbon energy level."""

    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: int | None  # None where the data set gives no parity
    j: float  # the level's J, which its name is keyed on and its g derived from


class TransitionTuple(t.NamedTuple):
    """One Lisbon bound-bound transition."""

    lowerlevel: int
    upperlevel: int
    A: float
    coll_str: float


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the Lisbon data set.

    The CSV files are not bundled with artisatomic; ARTISATOMIC_LISBON_PATH overrides where they
    are looked for.
    """
    ion_charge = ion_stage - 1
    elsym = elsymbols[atomic_number]
    ion_stage_roman = roman_numerals[ion_stage]

    assert elsym in {"Nd", "U"}
    assert ion_stage in {2, 3}

    print(f"Reading Lisbon data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman})")

    # the Lisbon CSVs are not part of this repository, so the location is configurable and
    # checked up front: pandas would otherwise report only the missing file, not what to set
    lisbonpath = Path(os.environ.get("ARTISATOMIC_LISBON_PATH", PYDIR / ".." / "atomic-data-lisbon")).resolve()
    if not lisbonpath.is_dir():
        msg = (
            f"Lisbon data directory {lisbonpath} not found. Set ARTISATOMIC_LISBON_PATH to the directory holding the"
            " per-ion <El>/<El><Stage>/<El><Stage>_Levels.csv and _Transitions.csv files."
        )
        raise FileNotFoundError(msg)

    iondir = lisbonpath / elsym / f"{elsym}{ion_stage_roman}"
    lisbon_data = {
        f"{elsym} {ion_charge}": {
            "atomic_number": atomic_number,
            "ion_charge": ion_charge,
            "levels": str(iondir / f"{elsym}{ion_stage_roman}_Levels.csv"),
            "lines": str(iondir / f"{elsym}{ion_stage_roman}_Transitions.csv"),
        }
    }

    lisbon_reader = LisbonReader(lisbon_data)

    dflevels = lisbon_reader.levels.loc[atomic_number, ion_charge]
    # the map associates source file level indices with energy-sorted level ids (0 indexed)
    energy_levels, levelid_of_fileindex = read_levels_data(dflevels)

    dflines = lisbon_reader.lines.loc[atomic_number, ion_charge]
    # gf to A with the wavelength in Angstrom, as in readkuruczdata and readmonsdata
    dflines = dflines.assign(
        A=dflines["gf"] / (gf_to_a_coefficient * (2 * dflines["j_upper"] + 1) * dflines["wavelength"] ** 2)
    )

    transitions, transition_count_of_level_name = read_lines_data(energy_levels, dflines, levelid_of_fileindex)

    # from NIST, as every other reader whose data set carries no ionization energy does. This was
    # -1, which went into adata.txt verbatim as the ion's ionization energy.
    ionization_energy_in_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]
    log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")

    log_and_print(flog, f"Read {len(energy_levels):d} levels")

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name

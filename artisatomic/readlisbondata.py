"""Read levels and transitions from the Lisbon data set. Not currently wired into the dispatch."""

import typing as t
from collections import defaultdict

import pandas as pd

import artisatomic


class LisbonReader:
    """Extract levels and lines from the Lisbon Atomic Group data.

    Copied from Andreas Floers' code in git.gsi.de:nucastro/opacities.git, and mimics the
    GFALLReader class.

    Attributes
    ----------
    levels : DataFrame
    lines : DataFrame
    """

    def __init__(self, data, priority=10) -> None:
        """Store the Lisbon data and its priority.

        Parameters
        ----------
        data : dict
            Dictionary containing one dictionary per species with
            keys `levels` and `lines`.

        priority: int, optional
            Priority of the current data source, by default 10.
        """
        self.priority = priority
        self._get_levels_lines(data)

    def _get_levels_lines(self, data):
        """Generate the `levels` and `lines` DataFrames.

        Parameters
        ----------
        data : dict
            Dictionary containing one dictionary per species with
            keys `levels` and `lines`.
        """
        # carsus is an optional extra that is not a declared dependency of this package
        from carsus.util import parse_selected_species  # ruff: ignore[unsorted-imports] # ty:ignore[unresolved-import] # pyright: ignore[reportMissingImports] # pyrefly: ignore[missing-import]

        lvl_list = []
        lns_list = []
        for ion, parser in data.items():
            atomic_number = parse_selected_species(ion)[0][0]
            ion_charge = parse_selected_species(ion)[0][1]
            levels_data = pd.read_csv(parser["levels"], skiprows=8, index_col=0)
            levels = pd.DataFrame()
            levels["energy"] = levels_data["Energy[cm^-1]"]
            levels["j"] = 0.5 * (levels_data["g"] - 1)
            levels["label"] = levels_data["RelConfig"]
            levels["method"] = "meas"
            levels["priority"] = self.priority
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
            lines["energy_lower"] = levels.iloc[lines["level_index_lower"]]["energy"].to_numpy()
            lines["energy_upper"] = levels.iloc[lines["level_index_upper"]]["energy"].to_numpy()
            lines["gf"] = lines_data["gf"]
            lines["j_lower"] = levels.iloc[lines["level_index_lower"]]["j"].to_numpy()
            lines["j_upper"] = levels.iloc[lines["level_index_upper"]]["j"].to_numpy()
            lines["wavelength"] = lines_data["Wavelength[Ang]"] / 10.0
            lines = lines.set_index(["atomic_number", "ion_charge", "level_index_lower", "level_index_upper"])
            lns_list.append(lines)
        levels = pd.concat(lvl_list)
        lines = pd.concat(lns_list)  # pyright: ignore[reportUnreachable]
        self.levels = levels
        self.lines = lines


def get_levelname(row):
    """Name a Lisbon level from its label and J, since the label alone is not unique."""
    return f"{row.label}, j={row.j}"


def read_levels_data(dflevels):
    """Convert the Lisbon level table to level tuples, sorted by energy.

    Every level is given a distinct parity so that add_level_ids_forbidden() marks none of the
    transitions forbidden: this data set does not supply parities.
    """
    energy_levels = []

    for index, row in dflevels.iterrows():
        parity = -index  # give a unique parity so that all transitions are permitted
        energyabovegsinpercm = float(row.energy)
        g = 2 * row.j + 1
        newlevel = EnergyLevelTuple(
            levelname=get_levelname(row), parity=parity, g=g, energyabovegsinpercm=energyabovegsinpercm
        )
        energy_levels.append(newlevel)

    energy_levels.sort(key=lambda x: x.energyabovegsinpercm)

    return energy_levels


def read_lines_data(energy_levels, dflines):
    """Convert Lisbon lines to transitions referencing zero-based level ids.

    Returns the transitions and the number of them touching each level name.
    """
    transitions = []
    transition_count_of_level_name = defaultdict(int)

    for (lowerlevel, upperlevel), row in dflines.iterrows():
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
    parity: int


class TransitionTuple(t.NamedTuple):
    """One Lisbon bound-bound transition."""

    lowerlevel: int
    upperlevel: int
    A: float
    coll_str: float


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the Lisbon data set. Not currently wired into the handler dispatch."""
    ion_charge = ion_stage - 1
    elsym = artisatomic.elsymbols[atomic_number]
    ion_stage_roman = artisatomic.roman_numerals[ion_stage]

    assert elsym in {"Nd", "U"}
    assert ion_stage in {2, 3}

    print(f"Reading Lisbon data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman})")

    LISPATH = "/Users/luke/Dropbox/GitHub/opacities/SystematicCalculations"

    lisbon_data = {
        f"{elsym} {ion_charge}": {
            "levels": f"{LISPATH}/{elsym}/{elsym}{ion_stage_roman}/{elsym}{ion_stage_roman}_Levels.csv",
            "lines": f"{LISPATH}/{elsym}/{elsym}{ion_stage_roman}/{elsym}{ion_stage_roman}_Transitions.csv",
        }
    }

    lisbon_reader = LisbonReader(lisbon_data)

    dflevels = lisbon_reader.levels.loc[atomic_number, ion_charge]
    energy_levels = read_levels_data(dflevels)

    dflines = lisbon_reader.lines.loc[atomic_number, ion_charge]
    dflines = dflines.eval("A = gf / (1.49919e-16 * (2 * j_upper + 1) * wavelength ** 2)")

    transitions, transition_count_of_level_name = read_lines_data(energy_levels, dflines)

    ionization_energy_in_ev = -1

    artisatomic.log_and_print(flog, f"Read {len(energy_levels):d} levels")

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name

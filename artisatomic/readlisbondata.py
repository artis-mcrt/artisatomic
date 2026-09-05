"""Read levels and transitions from the Lisbon Atomic Group data set (the "lisbon" handler)."""

import os
import typing as t
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
from artisatomic.base import Transition


class LisbonReader:
    """Extract levels and lines from the Lisbon Atomic Group data.

    This class is a copy of Andreas Floers' code in git.gsi.de:nucastro/opacities.git. It mimics
    the GFALLReader class.

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
            Dictionary that holds one dictionary per species with
            the keys `levels` and `lines`.
        """
        self._get_levels_lines(data)

    def _get_levels_lines(self, data):
        """Generate the `levels` and `lines` DataFrames.

        Parameters
        ----------
        data : dict
            Dictionary that holds one dictionary per species with
            the keys `atomic_number`, `ion_charge`, `levels` and `lines`.
        """
        lvl_list = []
        lns_list = []
        # the caller passes the element and the charge directly. The alternative was to format
        # them into the key and to parse them back with carsus.util.parse_selected_species().
        # That made an undeclared optional dependency a hard requirement of this reader.
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
            # keep the wavelength in Angstrom: the gf-to-A constant in read_lines_data()
            # expects Angstrom
            lines["wavelength"] = lines_data["Wavelength[Ang]"]
            lines = lines.set_index(["atomic_number", "ion_charge", "level_index_lower", "level_index_upper"])
            lns_list.append(lines)
        levels = pd.concat(lvl_list)
        # pd.concat() of the untyped list above narrows to Never, so a type checker reads this as dead
        lines = pd.concat(lns_list)
        self.levels = levels
        self.lines = lines


def get_levelname(row, fileindex: int):
    """Name a Lisbon level from its label, J and file index.

    The label alone is not unique, and neither is the label with J. In Nd II, most levels share
    their relativistic configuration and J with another level. The file index makes the name
    unique, as the FAC, Floers+25 and MONS readers do with theirs.
    """
    return f"{row.label}, j={row.j}, index={fileindex}"


def read_levels_data(dflevels):
    """Convert the Lisbon level table to level tuples, sorted by energy.

    Also returns the map from the file's level index to the zero-based level id, which
    read_lines_data() needs because the sort by energy reorders the levels.

    The lines name their levels by POSITION. LisbonReader reads their energies with
    levels.iloc[lines["level_index_lower"]], which is a position in the file-ordered frame, not an
    index label. So the key of the map is that position. The reset_index() below makes this
    explicit. Without it, the map would depend on a levels CSV whose numbers happen to start at
    zero.

    This data set supplies no parities, so every level's parity is null and the Laporte rule
    never fires. It does supply J, which is part of each level name, so the delta J rule alone
    decides whether a transition is forbidden here.
    """
    # sort first, so that the ids handed out below are the ones the levels keep. The reset of the
    # index to the row position comes first, so each level carries its file position through the sort
    dflevels = dflevels.reset_index(drop=True).sort_values(by="energy", kind="stable")

    energy_levels = [
        EnergyLevelTuple(
            levelname=get_levelname(row, fileposition),
            parity=None,  # no parity in this data set, so the Laporte rule cannot fire
            j=float(row.j),
            g=2 * row.j + 1,
            energyabovegsinpercm=float(row.energy),
        )
        for fileposition, row in dflevels.iterrows()
    ]

    return energy_levels, levelid_of_fileindex_map(dflevels.index, "the Lisbon levels file")


def read_lines_data(energy_levels, dflines, levelid_of_fileindex):
    """Convert Lisbon lines to transitions referencing zero-based level ids.

    The lines name their levels by position in the levels file, and read_levels_data() sorted the
    levels by energy. So the reader maps every position through levelid_of_fileindex and does not
    use it directly. A line that names a level that does not exist is an error, not something to
    skip.

    A = gf / (gf_to_a_coefficient * g_upper * wavelength^2) with the wavelength in Angstrom, as
    in readkuruczdata and readmonsdata. g_upper is the g of the level that is the upper level
    after the reader resolves the ids. It is not the g of the level that the file labels "Upper".
    The file can list a pair in the reverse order, and the swap must not leave A with the wrong g.
    """
    transitions = []

    for (fileindex_lower, fileindex_upper), row in dflines.iterrows():
        lowerlevel, upperlevel = resolve_transition_levelids(
            fileindex_lower, fileindex_upper, levelid_of_fileindex, "the Lisbon transitions file"
        )

        A = row.gf / (gf_to_a_coefficient * energy_levels[upperlevel].g * row.wavelength**2)
        transitions.append(Transition(lowerlevel=lowerlevel, upperlevel=upperlevel, A=A))

    return transitions


class EnergyLevelTuple(t.NamedTuple):
    """One Lisbon energy level."""

    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: int | None  # None where the data set gives no parity
    j: float  # the level's J, which is part of its name and the source of its g


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the Lisbon data set.

    The CSV files are not part of artisatomic. ARTISATOMIC_LISBON_PATH overrides the directory
    that the reader searches.
    """
    ion_charge = ion_stage - 1
    elsym = elsymbols[atomic_number]
    ion_stage_roman = roman_numerals[ion_stage]

    assert elsym in {"Nd", "U"}
    assert ion_stage in {2, 3}

    print(f"Reading Lisbon data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman})")

    # the Lisbon CSVs are not part of this repository, so the location is configurable. This check
    # comes first: pandas would otherwise report only the missing file, not what to set
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

    transitions = read_lines_data(energy_levels, dflines, levelid_of_fileindex)

    # from NIST, as every other reader whose data set carries no ionisation energy does. This was
    # -1, which went into adata.txt verbatim as the ion's ionisation energy.
    ionization_energy_in_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]
    log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")

    log_and_print(flog, f"Read {len(energy_levels):d} levels")

    return ionization_energy_in_ev, energy_levels, transitions

"""Read levels and transitions from the DREAM database of lanthanides and actinides."""

import numpy as np
import pandas as pd

from artisatomic.base import add_handler_if_not_set
from artisatomic.base import EnergyLevel
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import log_and_print
from artisatomic.base import PYDIR
from artisatomic.base import Transition

# the h5 file comes from Andreas Floers's DREAM parser
dreamdatapath = PYDIR / ".." / "atomic-data-dream" / "DREAM_atomic_data_20241106-1325.h5"
dreamdata: pd.DataFrame | None = None


def init_dreamdata():
    """Load the DREAM line list into the module-level cache, once per process."""
    global dreamdata
    if dreamdata is not None:
        return
    hdfdata = pd.read_hdf(dreamdatapath)
    assert isinstance(hdfdata, pd.DataFrame)
    dreamdata = hdfdata
    dreamdata = dreamdata.assign(Lower_g=lambda row: 2 * row["Lower_J"] + 1, Upper_g=lambda row: 2 * row["Upper_J"] + 1)


def extend_ion_list(ion_handlers):
    """Add every ion in the DREAM line list to ion_handlers under the "dream" handler."""
    init_dreamdata()
    assert dreamdata is not None
    for atomic_number, charge in dreamdata.index.unique():  # ty:ignore[possibly-missing-attribute]
        ion_stage = charge + 1
        ion_handlers = add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "dream")

    return ion_handlers


def energytuplefromrow(row, prefix):
    """Build the lower or upper level of one DREAM line, selected by prefix ("Lower"/"Upper").

    DREAM levels have no spectroscopic names, so the name of each level holds the energy, parity
    and statistical weight that identify it. read_levels_data() deduplicates on the same three
    values.
    """
    energy, leveltype, g = row[prefix + "_Level"], row[prefix + "_Type"], row[prefix + "_g"]

    # not a default of 0 for any other text: the Laporte rule would get a parity the file did
    # not state
    if leveltype not in {"(o)", "(e)"}:
        msg = f"DREAM level type {leveltype!r} is not '(o)' or '(e)'"
        raise ValueError(msg)
    parity = 1 if leveltype == "(o)" else 0
    paritystr = "odd" if parity == 1 else "even"
    energyabovegsinpercm = float(energy)

    levelname = f"enpercm={energy},{paritystr},g={g}"
    return EnergyLevel(levelname=levelname, parity=parity, g=g, energyabovegsinpercm=energyabovegsinpercm)


def read_levels_data(dflines):
    """Recover the level list from a DREAM line list, which has no separate level table.

    Each line carries both of its levels inline, so the levels are the distinct lower and upper
    levels over all lines, sorted by energy.
    """
    # a set for the membership test, not `not in energy_levels`: that was a linear scan of the
    # list per candidate. The build of the level list then cost O(levels^2)
    seen: set[EnergyLevel] = set()
    energy_levels = []

    for prefix in ["Lower", "Upper"]:
        for _, row in dflines.drop_duplicates(subset=[prefix + "_Type", prefix + "_Level", prefix + "_g"]).iterrows():
            leveltuple = energytuplefromrow(row, prefix)
            if leveltuple not in seen:
                seen.add(leveltuple)
                energy_levels.append(leveltuple)

    energy_levels.sort(key=lambda x: x.energyabovegsinpercm)

    return energy_levels


def read_lines_data(dfiondata):
    """Convert DREAM lines to transitions referencing zero-based level ids."""
    transitions = []

    # numpy columns, not iterrows(): that built a Series for each of the 10^5 lines of an ion.
    # read_levels_data() sorted the levels by energy, and transitiondata.txt has the lower id
    # first. So this code swaps a pair that the file lists in the reverse order.
    lowerindices = np.minimum(dfiondata["Lower_index"].to_numpy(), dfiondata["Upper_index"].to_numpy())
    upperindices = np.maximum(dfiondata["Lower_index"].to_numpy(), dfiondata["Upper_index"].to_numpy())
    # g_upper is the g of the level the file calls upper, because gA is that level's product
    A_values = dfiondata["gA"].to_numpy() / dfiondata["Upper_g"].to_numpy()

    for lowerindex, upperindex, A in zip(lowerindices.tolist(), upperindices.tolist(), A_values.tolist(), strict=True):
        transitions.append(Transition(lowerlevel=lowerindex, upperlevel=upperindex, A=A))

    return transitions


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the DREAM database of Z >= 57."""
    init_dreamdata()
    assert dreamdata is not None
    charge = ion_stage - 1
    # a list of one key, not .loc[atomic_number, charge]: that gives a Series for an ion with a
    # single line. The frame methods below then fail
    dfiondata = dreamdata.loc[[(atomic_number, charge)]].reset_index(drop=True)  # ty:ignore[possibly-missing-attribute]
    print(f"Reading DREAM database for Z={atomic_number} ion_stage {ion_stage}")

    energy_levels = read_levels_data(dfiondata)

    # a dict, not energy_levels.index(): that scanned the level list once per level of every
    # line. The id resolution then cost O(lines x levels) for a database of Z >= 57 lanthanides
    levelid_of_leveltuple = {leveltuple: levelid for levelid, leveltuple in enumerate(energy_levels)}

    def get_level_index(row, prefix):
        """Return the zero-based level id of the row's level."""
        leveltuple = energytuplefromrow(row, prefix)
        levelid = levelid_of_leveltuple.get(leveltuple)
        if levelid is None:
            # not an assert: read_levels_data() built the level list from the same frame as the
            # lines. A miss means they disagree, so the code must not map it silently
            msg = f"DREAM line names a {prefix} level that is not in the level list: {leveltuple}"
            raise ValueError(msg)
        return levelid

    # a list over the row records, not DataFrame.apply(axis=1): apply builds a Series per row.
    # That cost tens of microseconds for each of the 10^5 lines of a lanthanide ion
    rows = dfiondata.to_dict("records")
    dfiondata.insert(2, "Lower_index", [get_level_index(row, prefix="Lower") for row in rows], allow_duplicates=True)
    dfiondata.insert(2, "Upper_index", [get_level_index(row, prefix="Upper") for row in rows], allow_duplicates=True)

    transitions = read_lines_data(dfiondata)

    # DREAM has no ionisation energies, so take them from NIST as the other handlers do
    ionization_energy_in_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]
    log_and_print(flog, f"ionisation energy: {ionization_energy_in_ev} eV")

    log_and_print(flog, f"Read {len(energy_levels):d} levels")

    return ionization_energy_in_ev, energy_levels, transitions

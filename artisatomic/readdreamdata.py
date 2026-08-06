import os.path
from collections import defaultdict
from collections import namedtuple
from pathlib import Path

import pandas as pd

import artisatomic

# the h5 file comes from Andreas Floers's DREAM parser
dreamdatapath = Path(
    os.path.dirname(os.path.abspath(__file__)), "..", "atomic-data-dream", "DREAM_atomic_data_20241106-1325.h5"
)
dreamdata: pd.DataFrame | None = None
hc_in_ev_cm = 0.0001239841984332003


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
        ion_handlers = artisatomic.add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "dream")

    return ion_handlers


def energytuplefromrow(row, prefix):
    """Build the lower or upper level of one DREAM line, selected by prefix ("Lower"/"Upper").

    DREAM levels have no spectroscopic names, so each is named after the energy, parity and
    statistical weight that identify it, which is what read_levels_data() deduplicates on.
    """
    energy, leveltype, g = row[prefix + "_Level"], row[prefix + "_Type"], row[prefix + "_g"]
    dream_energy_level_row = namedtuple("dream_energy_level_row", "levelname energyabovegsinpercm g parity")

    parity = 1 if leveltype == "(o)" else 0
    paritystr = "odd" if parity == 1 else "even"
    energyabovegsinpercm = float(energy)

    levelname = f"enpercm={energy},{paritystr},g={g}"
    newlevel = dream_energy_level_row(
        levelname=levelname, parity=parity, g=g, energyabovegsinpercm=energyabovegsinpercm
    )
    return newlevel


def read_levels_data(dflines):
    """Recover the level list from a DREAM line list, which has no separate level table.

    Each line carries both of its levels inline, so the levels are the distinct lower and upper
    levels over all lines, sorted by energy.
    """
    energy_levels = []

    for prefix in ["Lower", "Upper"]:
        for index, row in dflines.drop_duplicates(
            subset=[prefix + "_Type", prefix + "_Level", prefix + "_g"]
        ).iterrows():
            leveltuple = energytuplefromrow(row, prefix)
            if leveltuple not in energy_levels:
                energy_levels.append(leveltuple)

    energy_levels.sort(key=lambda x: x.energyabovegsinpercm)

    return energy_levels


def read_lines_data(dfiondata, energy_levels):
    """Convert DREAM lines to transitions referencing zero-based level ids.

    Returns the transitions and the number of them touching each level name.
    """
    transitions = []
    transition_count_of_level_name = defaultdict(int)
    transitiontuple = namedtuple("transitiontuple", "lowerlevel upperlevel A coll_str")

    for index, row in dfiondata.iterrows():
        lowerindex = row["Lower_index"]
        upperindex = row["Upper_index"]
        A = row["gA"] / row["Upper_g"]  # TODO: is this correct?
        transtuple = transitiontuple(lowerlevel=lowerindex, upperlevel=upperindex, A=A, coll_str=-1)

        # print(line)
        transition_count_of_level_name[energy_levels[lowerindex].levelname] += 1
        transition_count_of_level_name[energy_levels[upperindex].levelname] += 1

        transitions.append(transtuple)

    return transitions, transition_count_of_level_name


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the DREAM database of Z >= 57."""
    init_dreamdata()
    assert dreamdata is not None
    charge = ion_stage - 1
    dfiondata = dreamdata.loc[atomic_number, charge]  # ty:ignore[possibly-missing-attribute]
    print(f"Reading DREAM database for Z={atomic_number} ion_stage {ion_stage}")
    print(dfiondata)

    # from nistasd import NISTLines
    #
    # nist = NISTLines(spectrum='Ce')
    # energy_levels = nist.get_energy_level_data()
    # print("energy_levels.keys() = ", energy_levels['Ce I'])
    # for ion_stage in energy_levels:
    #     print("Number of energy levels: {0} for {1}".format(len(energy_levels[ion_stage]), ion_stage))
    #     df = pd.DataFrame(energy_levels[ion_stage])
    #     print(df)

    energy_levels = read_levels_data(dfiondata)

    # for x in energy_levels:
    #     print(x)
    def get_level_index(row, prefix):
        """Return the zero-based level id of the row's level."""
        leveltuple = energytuplefromrow(row, prefix)
        if leveltuple in energy_levels:
            return energy_levels.index(leveltuple)
        raise AssertionError

    dfiondata.insert(
        2,
        "Lower_index",
        dfiondata.apply(lambda row: get_level_index(row, prefix="Lower"), axis=1).to_numpy(),
        allow_duplicates=True,
    )

    dfiondata.insert(
        2,
        "Upper_index",
        dfiondata.apply(lambda row: get_level_index(row, prefix="Upper"), axis=1).to_numpy(),
        allow_duplicates=True,
    )

    transitions, transition_count_of_level_name = read_lines_data(dfiondata, energy_levels)

    # DREAM has no ionization energies, so take them from NIST as the other handlers do
    ionization_energy_in_ev = artisatomic.get_nist_ionization_energies_ev()[(atomic_number, ion_stage)]
    artisatomic.log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")

    artisatomic.log_and_print(flog, f"Read {len(energy_levels):d} levels")

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name

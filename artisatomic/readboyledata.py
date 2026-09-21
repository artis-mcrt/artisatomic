"""Read helium levels and transitions from the Boyle AOIFE data set.

The data set belongs to the helium work of Boyle, A., Sim, S. A., Hachinger, S., Kerzendorf, W.
(2017), A&A, 599, A46, doi:10.1051/0004-6361/201629712.
"""

import typing as t
from functools import cache

from artisatomic.base import hc_in_ev_cm
from artisatomic.base import log_comment
from artisatomic.base import path_for_log
from artisatomic.base import PYDIR

datafilepath = PYDIR / ".." / "atomic-data-helium-boyle" / "aoife.hdf5"

# the "source:" line of the comment blocks in the output files (see Handler.description in iondata.py)
description = (
    "the AOIFE helium data set. Boyle, A., Sim, S. A., Hachinger, S., Kerzendorf, W. (2017), A&A, 599, A46,"
    " doi:10.1051/0004-6361/201629712"
)


@cache
def get_aoife_dataset():
    """Open the AOIFE HDF5 file, once, on first use.

    The open happens here and not at import. iondata.py imports this module for the handler
    registry. So an open at import would hold a file handle for the whole of every run, whichever
    handlers the user selected. Returns None when the file (or h5py) is absent. The readers below
    use that to report that this data set is unavailable.
    """
    try:
        import h5py
    except ModuleNotFoundError:
        return None

    return h5py.File(datafilepath, "r") if datafilepath.exists() else None


class EnergyLevelRow(t.NamedTuple):
    """One level of the AOIFE levels_data table, with the derived fields appended.

    The table's energy column is the energy above the ground state in eV. The reader converts it
    to cm^-1 once, in energyabovegsinpercm.
    """

    atomic_number: float
    ion_number: float
    level_number: float
    g: float
    metastable: float
    energyabovegsinpercm: float
    parity: int | None  # None where the data set gives no parity
    levelname: str


class TransitionTuple(t.NamedTuple):
    """One bound-bound transition of the AOIFE lines_data table."""

    atomic_number: float
    ion_stage: float
    lowerlevel: int
    upperlevel: int
    A: float
    lambdaangstrom: float


def read_ionization_data(atomic_number, ion_stage):
    """Ionisation energy in eV of one ion, from the AOIFE HDF5 file.

    He III is a bare nucleus, so the file has no entry for it. The function uses a sentinel instead.
    """
    aoife_dataset = get_aoife_dataset()
    assert aoife_dataset is not None, "the boyle handler needs the AOIFE HDF5 file"
    ionization_data = aoife_dataset["/ionization_data"]

    ionization_dict = {}
    for atomic_num, ion_number, ionization_energy in ionization_data:
        ion_dict = {ion_number: ionization_energy}
        if atomic_num in ionization_dict:
            ionization_dict[atomic_num].update(ion_dict)
        else:
            ionization_dict[atomic_num] = ion_dict
    ionization_dict[2][3] = 999999.0  # He III

    return ionization_dict[atomic_number][ion_stage]


def read_levels_data(atomic_number, ion_stage):
    """Read one ion's energy levels from the AOIFE HDF5 file.

    The file numbers ion stages from zero, so the reader matches ion_stage against ion_number + 1.
    Levels have no spectroscopic names, so the name of each level holds its zero-based level number.
    """
    aoife_dataset = get_aoife_dataset()
    assert aoife_dataset is not None, "the boyle handler needs the AOIFE HDF5 file"
    levels_data = aoife_dataset["/levels_data"]

    energy_levels: list[EnergyLevelRow] = []

    for rowtuple in levels_data:
        atomic_num, ion_number, level_number, energy_ev, g, metastable = rowtuple

        if int(atomic_num) != atomic_number or int(ion_number) != ion_stage - 1:
            continue

        # named fields, not *rowtuple plus three positional extras: a bare 0 in the extras would
        # become the parity of every level
        energy_levels.append(
            EnergyLevelRow(
                atomic_number=atomic_num,
                ion_number=ion_number,
                level_number=level_number,
                g=g,
                metastable=metastable,
                # the AOIFE energy column is in eV (He I 1s2s 3S is 19.8196)
                energyabovegsinpercm=energy_ev / hc_in_ev_cm,
                # No parity: this data set supplies none. add_level_ids_forbidden() marks a
                # transition forbidden when its two levels share one parity, and helium has many
                # permitted ones. A null parity never matches another, here as in the other readers
                # whose data set has no parities.
                parity=None,
                # int() as read_lines_data() does, so the two agree on the name whatever dtype the
                # file stores the level number in
                levelname=f"level{int(level_number):05d}",
            )
        )

    # not an assert: read_lines_data() takes the file's level numbers as the zero-based level ids
    # that leveltuples_to_pldataframe() assigns by row. The two numberings must agree
    level_numbers = [int(level.level_number) for level in energy_levels]
    if level_numbers != list(range(len(energy_levels))):
        msg = (
            f"the AOIFE levels of Z={atomic_number} ion_stage {ion_stage} are not numbered 0 to"
            f" {len(energy_levels) - 1} in file order"
        )
        raise ValueError(msg)

    return energy_levels


def read_lines_data(atomic_number, ion_stage):
    """Read one ion's bound-bound transitions from the AOIFE HDF5 file.

    The file's level numbers are already zero-based, the same as the level ids in memory. The
    file gives no collision strengths.
    """
    aoife_dataset = get_aoife_dataset()
    assert aoife_dataset is not None, "the boyle handler needs the AOIFE HDF5 file"
    lines_data = aoife_dataset["/lines_data"]

    transitions = []

    for rowtuple in lines_data:
        (
            _line_id,
            wavelength,
            atomic_num,
            ion_number,
            _f_ul,
            _f_lu,
            level_number_lower,
            level_number_upper,
            _nu,
            _B_lu,
            _B_ul,
            A_ul,
        ) = rowtuple

        if int(atomic_num) != atomic_number or int(ion_number) != ion_stage - 1:
            continue
        # transitiondata.txt has the lower id first, so this code swaps a pair that the file lists
        # in the reverse order.
        levelid_lower = min(int(level_number_lower), int(level_number_upper))
        levelid_upper = max(int(level_number_lower), int(level_number_upper))
        transitions.append(TransitionTuple(atomic_num, ion_number, levelid_lower, levelid_upper, A_ul, wavelength))

    return transitions


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion for the "boyle" handler, which covers helium only."""
    assert atomic_number == 2
    log_comment(flog, ("adata", "transitiondata"), f"Reading {path_for_log(datafilepath)}")
    transitions = read_lines_data(atomic_number, ion_stage)

    ionization_energy_in_ev = read_ionization_data(atomic_number, ion_stage)

    energy_levels = read_levels_data(atomic_number, ion_stage)

    return ionization_energy_in_ev, energy_levels, transitions

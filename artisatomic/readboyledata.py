"""Read helium levels and transitions from the Boyle AOIFE data set."""

import os.path
from collections import defaultdict
from collections import namedtuple
from pathlib import Path

datafilepath = Path(os.path.dirname(os.path.abspath(__file__)), "..", "atomic-data-helium-boyle", "aoife.hdf5")

try:
    import h5py  # pyright: ignore[reportMissingTypeStubs]

    aoife_dataset = h5py.File(datafilepath, "r") if datafilepath.exists() else None
except ModuleNotFoundError:
    aoife_dataset = None


def read_ionization_data(atomic_number, ion_stage):
    """Ionization energy in eV of one ion, from the AOIFE HDF5 file.

    He III is a bare nucleus, so the file has no entry for it and a sentinel is used instead.
    """
    assert aoife_dataset is not None
    ionization_data = aoife_dataset["/ionization_data"]

    ionization_dict = {}
    for atomic_num, ion_number, ionization_energy in ionization_data:  # pyright: ignore[reportGeneralTypeIssues]
        ion_dict = {ion_number: ionization_energy}
        if atomic_num in ionization_dict:
            ionization_dict[atomic_num].update(ion_dict)
        else:
            ionization_dict[atomic_num] = ion_dict
    ionization_dict[2][3] = 999999.0  # He III

    return ionization_dict[atomic_number][ion_stage]


def read_levels_data(atomic_number, ion_stage):
    """Read one ion's energy levels from the AOIFE HDF5 file.

    The file numbers ion stages from zero, so ion_stage is matched against ion_number + 1.
    Levels have no spectroscopic names, so each is named after its zero-based level number,
    in the same format read_lines_data() uses to count transitions per level.
    """
    assert aoife_dataset is not None
    levels_data = aoife_dataset["/levels_data"]

    energy_level_row = namedtuple(
        "energy_level_row",
        "atomic_number ion_number level_number energy g metastable energyabovegsinpercm parity levelname",
    )
    energy_levels: list[energy_level_row] = []

    for rowtuple in levels_data:  # pyright: ignore[reportGeneralTypeIssues]
        _atomic_num, _ion_number, level_number, energyabovegsinpercm, _g, _metastable = rowtuple
        # the six unpacked columns plus three more make up the nine fields, which the type
        # checkers cannot count through the star-unpacking
        energy_level = energy_level_row(*rowtuple, energyabovegsinpercm, 0, f"level{level_number:05d}")  # pyrefly: ignore [bad-argument-count] # ty:ignore[too-many-positional-arguments] # pyright: ignore[reportCallIssue]

        if int(energy_level.atomic_number) != atomic_number or int(energy_level.ion_number) != ion_stage - 1:
            continue
        energy_levels.append(energy_level)

    return energy_levels


def read_lines_data(atomic_number, ion_stage):
    """Read one ion's bound-bound transitions from the AOIFE HDF5 file.

    Returns the transitions and the number of them touching each level name. The file's level
    numbers are already zero-based, matching the level ids used in memory. No collision
    strengths are available, so every transition gets the -1 "unknown" sentinel.
    """
    assert aoife_dataset is not None
    lines_data = aoife_dataset["/lines_data"]

    transitions = []
    transition_count_of_level_name = defaultdict(int)
    transition_tuple = namedtuple(
        "transition_tuple", "atomic_number ion_stage lowerlevel upperlevel A lambdaangstrom coll_str"
    )

    for rowtuple in lines_data:  # pyright: ignore[reportGeneralTypeIssues]
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

        coll_str = -1  # TODO
        # the file's level numbers are already zero-based, matching the level ids used in memory
        line = transition_tuple(
            atomic_num, ion_number, int(level_number_lower), int(level_number_upper), A_ul, wavelength, coll_str
        )
        if int(atomic_num) != atomic_number or int(ion_number) != ion_stage - 1:
            continue
        # print(line)
        # must match the levelname format used in read_levels_data
        transition_count_of_level_name[f"level{int(level_number_lower):05d}"] += 1
        transition_count_of_level_name[f"level{int(level_number_upper):05d}"] += 1

        transitions.append(line)

    return transitions, transition_count_of_level_name


def read_levels_and_transitions(atomic_number, ion_stage):
    """Read one ion for the "boyle" handler, which covers helium only."""
    assert atomic_number == 2
    # artisatomic.log_and_print(flog, 'Reading atomic-data-He')
    transitions, transition_count_of_level_name = read_lines_data(atomic_number, ion_stage)

    ionization_energy_in_ev = read_ionization_data(atomic_number, ion_stage)
    # ionization_energy_in_ev = -1

    energy_levels = read_levels_data(atomic_number, ion_stage)
    # artisatomic.log_and_print(flog, f'Read {len(energy_levels):d} levels')

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name

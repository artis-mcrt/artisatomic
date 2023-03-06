from collections import defaultdict
from collections import namedtuple
from pathlib import Path

import h5py
import pandas as pd
from astropy import constants as const

import artisatomic

# from astropy import units as u

# the h5 file comes from Andreas Floers's DREAM parser
jpltpath = (Path(__file__).parent.resolve() / ".." / "atomic-data-tanaka-jplt" / "data").resolve()
hc_in_ev_cm = (const.h * const.c).to("eV cm").value


def extend_ion_list(listelements):
    tanakaions = sorted(
        [tuple([int(x) for x in f.parts[-1].removesuffix(".txt").split("_")]) for f in jpltpath.glob("*_*.txt")]
    )

    for atomic_number, ion_stage in tanakaions:
        found_element = False
        for tmp_atomic_number, list_ions_handlers in listelements:
            if tmp_atomic_number == atomic_number:
                # add an ion that is not present in the element's list
                if ion_stage not in [x[0] if hasattr(x, "__getitem__") else x for x in list_ions_handlers]:
                    list_ions_handlers.append((ion_stage, "tanakajplt"))
                    list_ions_handlers.sort(key=lambda x: x[0] if hasattr(x, "__getitem__") else x)
                found_element = True

        if not found_element:
            listelements.append(
                (
                    atomic_number,
                    [(ion_stage, "tanakajplt")],
                )
            )

    listelements.sort(key=lambda x: x[0])

    return listelements


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    filename = f"{atomic_number}_{ion_stage}.txt"
    print(f"Reading Tanaka et al. Japan-Lithuania database for Z={atomic_number} ion_stage {ion_stage} from {filename}")
    with open(jpltpath / filename) as fin:
        artisatomic.log_and_print(flog, fin.readline().strip())
        artisatomic.log_and_print(flog, fin.readline().strip())
        artisatomic.log_and_print(flog, fin.readline().strip())
        assert fin.readline().strip() == f"# {atomic_number} {ion_stage}"
        levelcount, transitioncount = [int(x) for x in fin.readline().removeprefix("# ").split()]
        artisatomic.log_and_print(flog, f"levels: {levelcount}")
        artisatomic.log_and_print(flog, f"transitions: {transitioncount}")

        fin.readline()
        str_ip_line = fin.readline()
        ionization_energy_in_ev = float(str_ip_line.removeprefix("# IP = "))
        artisatomic.log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")
        assert fin.readline().strip() == "# Energy levels"
        assert fin.readline().strip() == "# num  weight parity      E(eV)      configuration"

        with pd.read_fwf(
            fin,
            chunksize=levelcount,
            nrows=levelcount,
            colspecs=[(0, 7), (7, 15), (15, 19), (19, 34), (34, None)],
            names=["num", "weight", "parity", "energy_ev", "configuration"],
        ) as reader:
            # dflevels = pd.concat(reader, ignore_index=True)

            dflevels = reader.get_chunk(levelcount)
            # print(dflevels)

            energy_level_tuple = namedtuple("energylevel", "levelname energyabovegsinpercm g parity")
            energy_levels = [None]
            for row in dflevels.itertuples(index=False):
                parity = 1 if row.parity.strip() == "odd" else 0
                energyabovegsinpercm = float(row.energy_ev / hc_in_ev_cm)
                g = float(row.weight)

                levelname = f"{row.num},{row.parity},{row.configuration.strip()}"
                energy_levels.append(
                    energy_level_tuple(
                        levelname=levelname, parity=parity, g=g, energyabovegsinpercm=energyabovegsinpercm
                    )
                )
                # print(energy_levels[-1])
        assert len(energy_levels[1:]) == levelcount

        line = fin.readline().strip()
        assert line == "# Transitions" or line == "# num_u   num_l   wavelength(nm)     g_u*A      log(g_l*f)"
        if line == "# Transitions":
            assert fin.readline().strip() == "# num_u   num_l   wavelength(nm)     g_u*A      log(g_l*f)"
        dftransitions = pd.read_fwf(
            fin,
            colspecs=[(0, 7), (7, 15), (15, 30), (30, 43), (43, None)],
            names=["num_u", "num_l", "wavelength", "g_u_times_A", "log(g_l*f)"],
        )
        # dftransitions = reader.get_chunk(transitioncount)

        # print(dftransitions)
        transitiontuple = namedtuple("transition", "lowerlevel upperlevel A coll_str")
        transitions = []
        transition_count_of_level_name = defaultdict(int)

        for row in dftransitions.itertuples(index=False):
            A = float(row.g_u_times_A) / energy_levels[row.num_u].g
            coll_str = -1 if (energy_levels[row.num_u].parity != energy_levels[row.num_l].parity) else -2
            transitions.append(transitiontuple(lowerlevel=row.num_l, upperlevel=row.num_u, A=A, coll_str=coll_str))

            transition_count_of_level_name[energy_levels[row.num_u].levelname] += 1
            transition_count_of_level_name[energy_levels[row.num_l].levelname] += 1

            # print(transitions[-1])

        assert len(transitions) == transitioncount

    # artisatomic.log_and_print(flog, f'Read {len(energy_levels[1:]):d} levels')

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name

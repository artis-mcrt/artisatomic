import typing as t
from collections import defaultdict
from collections import namedtuple
from pathlib import Path

import artisatomic

hc_in_ev_cm = 0.0001239841984332003

gfall_reader: t.Any = None


def find_gfall(atomic_number: int, ion_charge: int) -> Path:
    path_gfall = (
        Path(__file__).parent.absolute()
        / ".."
        / "atomic-data-kurucz"
        / "extendedatoms"
        / f"gf{atomic_number:02d}{ion_charge:02d}.lines"
    ).resolve()

    if not path_gfall.is_file():
        path_gfall = (
            Path(__file__).parent.absolute()
            / ".."
            / "atomic-data-kurucz"
            / "extendedatoms"
            / f"gf{atomic_number:02d}{ion_charge:02d}z.lines"
        ).resolve()

    if not path_gfall.is_file():
        path_gfall = (
            Path(__file__).parent.absolute()
            / ".."
            / "atomic-data-kurucz"
            / "zztar"
            / f"gf{atomic_number:02d}{ion_charge:02d}.all"
        ).resolve()

    if not path_gfall.is_file():
        raise FileNotFoundError(f"No Kurucz file for Z={atomic_number} ion_charge {ion_charge}")

    return path_gfall


# def extend_ion_list(ion_handlers):
#     selected_ions = [(38, 1)]
#     for atomic_number, charge in selected_ions:
#         assert (atomic_number, charge) in gfall_reader.ions
#         ion_stage = charge + 1
#
#         found_element = False
#         for (tmp_atomic_number, list_ions) in ion_handlers:
#             if tmp_atomic_number == atomic_number:
#                 if ion_stage not in list_ions:
#                     list_ions.append((ion_stage, 'carsus'))
#                     list_ions.sort()
#                 found_element = True
#
#         if not found_element:
#             ion_handlers.append((atomic_number, [(ion_stage, 'carsus')],))
#
#     ion_handlers.sort(key=lambda x: x[0])
#
#     return ion_handlers


def get_levelname(row) -> str:
    return f"{row.label},enpercm={row.energy},j={row.j}"


def read_levels_data(dflevels):
    energy_level_tuple = namedtuple("energylevel", "levelname energyabovegsinpercm g parity")

    energy_levels = []

    for index, row in dflevels.iterrows():
        parity = -index  # give a unique parity so that all transitions are permitted
        energyabovegsinpercm = float(row.energy)
        g = 2 * row.j + 1
        newlevel = energy_level_tuple(
            levelname=get_levelname(row), parity=parity, g=g, energyabovegsinpercm=energyabovegsinpercm
        )
        energy_levels.append(newlevel)

    energy_levels.sort(key=lambda x: x.energyabovegsinpercm)

    return [None, *energy_levels]


def read_lines_data(energy_levels, dflines):
    transitions = []
    transition_count_of_level_name = defaultdict(int)
    transitiontuple = namedtuple("transition", "lowerlevel upperlevel A coll_str")

    for (lowerindex, upperindex), row in dflines.iterrows():
        lowerlevel = lowerindex + 1
        upperlevel = upperindex + 1

        transtuple = transitiontuple(lowerlevel=lowerlevel, upperlevel=upperlevel, A=row.A, coll_str=-1)

        # print(line)
        transition_count_of_level_name[energy_levels[lowerlevel].levelname] += 1
        transition_count_of_level_name[energy_levels[upperlevel].levelname] += 1

        transitions.append(transtuple)

    return transitions, transition_count_of_level_name


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    ion_charge = ion_stage - 1

    artisatomic.log_and_print(flog, f"Using Kurucz via CARSUS for Z={atomic_number} ion_stage {ion_stage}")

    from carsus.io.kurucz import GFALLReader

    # path_gfall = (Path(__file__).parent.absolute() / ".." / "atomic-data-kurucz" / "gfall.dat").resolve()
    path_gfall = find_gfall(atomic_number, ion_charge)
    artisatomic.log_and_print(flog, f"Reading {path_gfall}")
    gfall_reader = GFALLReader(
        ions=f"{artisatomic.elsymbols[atomic_number]} {ion_charge}",
        fname=str(path_gfall),
        unique_level_identifier=["energy", "j"],
    )

    dflevels = gfall_reader.extract_levels().loc[atomic_number, ion_charge]

    energy_levels = read_levels_data(dflevels)

    artisatomic.log_and_print(flog, f"Read {len(energy_levels[1:]):d} levels")

    dflines = gfall_reader.extract_lines().loc[atomic_number, ion_charge]

    # wavelengths are in nanometers, so multiply by 10 to get Angstroms
    dflines["A"] = dflines["gf"] / (1.49919e-16 * (2 * dflines["j_upper"] + 1) * (dflines["wavelength"] * 10.0) ** 2)

    transitions, transition_count_of_level_name = read_lines_data(energy_levels, dflines)
    artisatomic.log_and_print(flog, f"Read {len(transitions):d} transitions")

    ionization_energy_in_ev = artisatomic.get_nist_ionization_energies_ev()[(atomic_number, ion_stage)]
    artisatomic.log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name


def get_level_valence_n(levelname: str):
    namesplit = levelname.replace("  ", " ").split(" ")
    if len(namesplit) < 2 or not (part := namesplit[-2]):
        print(f"WARNING: Could not find n in {levelname}. Using n=1")
        return 1

    if part[-1] not in "spdfghijklmnopqr":
        # end of string is a number of electrons in the orbital, not a principal quantum number, so remove it

        if not part[-1].isdigit():
            print(f"WARNING: Could not find n in {levelname}. Using n=1")
            return 1
        part = part.rstrip("0123456789")
    part = part.strip("spdfghijklmnopqr")

    # inefficient way to find the last number in a string
    for i in range(len(part)):
        try:
            n = int(part[i:])
        except ValueError:
            continue
        else:
            assert n >= 0
            assert n < 50
            return n

    print(f"WARNING: Could not find n in {levelname}. Using n=1")
    return 1

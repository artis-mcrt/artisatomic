import os.path
from pathlib import Path
import typing as t
import pandas as pd
from astropy import constants as const
from astropy import units as u
from collections import defaultdict
import numpy as np
import zipfile

# import artisatomic

hc_in_ev_cm = (const.h * const.c).to("eV cm").value
hc_in_ev_angstrom = (const.h * const.c).to("eV angstrom").value


class EnergyLevel(t.NamedTuple):
    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: float


class TransitionTuple(t.NamedTuple):
    lowerlevel: int
    upperlevel: int
    A: float
    coll_str: float


datafilepath = Path(os.path.dirname(os.path.abspath(__file__)), "..", "atomic-data-mons")


# outggf_Ln_V-VII.zip folder:
#     45 files outggf for each lanthanide between the V and VII spectra:
#     first column is wavelength of the E1 transition (A),
#     second column is the lower energy level of the transition (1000 cm^-1)
#     third column is the oscillator strength
#
# outglv_Ln_V--VII.zip folder:
#     45 files outglv for each lanthanide between the V and VII spectra:
#     first column is the energy of levels (1000 cm^-1)
#     second column is the total angular momentum (J-value)


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    # TODO: generalise ################


    datafilename_energylevels = "outglv_0_La_V"
    # datafilename_energylevels = f"outglv_0_{artisatomic.elsymbols[atomic_number]}_{artisatomic.roman_numerals[ion_stage]}"
    datafilename_transitions = "outggf_sorted_La_V"
    ###################################

    # Read first file
    energy_levels1000percm, j_arr = np.loadtxt(datafilepath / datafilename_energylevels, unpack=True, delimiter=",")
    artisatomic.log_and_print(flog, f"levels: {len(energy_levels1000percm)}")

    energiesabovegsinpercm = energy_levels1000percm * 1000

    g_arr = 2 * j_arr + 1
    print(g_arr[374], len(g_arr))

    # Sort table by energy levels
    dfenergylevels = pd.DataFrame.from_dict({"energiesabovegsinpercm": energiesabovegsinpercm, "g": g_arr})
    dfenergylevels = dfenergylevels.sort_values("energiesabovegsinpercm")
    print(dfenergylevels)

    energiesabovegsinpercm = dfenergylevels["energiesabovegsinpercm"].values
    g_arr = dfenergylevels["g"].values

    parity = None
    energy_levels = [None]

    for levelindex, (g, energyabovegsinpercm) in enumerate(zip(g_arr, energiesabovegsinpercm)):
        energy_levels.append(
            EnergyLevel(levelname=str(levelindex), parity=parity, g=g, energyabovegsinpercm=energyabovegsinpercm)
        )

    # Read next file
    transition_wavelength_A, energy_levels_lower_1000percm, oscillator_strength = np.loadtxt(
        datafilepath / datafilename_transitions, unpack=True, delimiter=","
    )
    artisatomic.log_and_print(flog, f"transitions: {len(energy_levels_lower_1000percm)}")

    energy_levels_lower_percm = energy_levels_lower_1000percm * 1000
    print(transition_wavelength_A, energy_levels_lower_percm, oscillator_strength)

    # Get index of lower level of transition
    lowerlevel = np.array(
        [
            (
                np.abs(
                    energiesabovegsinpercm - energylevellower
                )  # get closest energy in energy level array to lower level
            ).argmin()  # then get the index with argmin()
            for energylevellower in energy_levels_lower_percm
        ]
    )

    ionization_energy_in_ev = artisatomic.get_nist_ionization_energies_ev()[(atomic_number, ion_stage)]
    artisatomic.log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")

    # get energy of upper level of transition
    energy_levels_lower_ev = energy_levels_lower_percm * hc_in_ev_cm
    transitionenergyev = hc_in_ev_angstrom / transition_wavelength_A
    assert max(transitionenergyev) < ionization_energy_in_ev
    energy_levels_upper_ev = transitionenergyev + energy_levels_lower_ev
    energy_levels_upper_percm = energy_levels_upper_ev / hc_in_ev_cm

    # Get index of upper level of transition
    upperlevel = np.array(
        [
            (
                np.abs(energiesabovegsinpercm - energylevelupper)  # get closest energy in energy level array
            ).argmin()  # then get the index with argmin()
            for energylevelupper in energy_levels_upper_percm
        ]
    )

    # Get A value from oscillator strength
    A_ul = np.array(
        [
            (
                (8 * np.pi**2 * const.e.value**2)
                / (
                    const.m_e.value * const.c.value * (transition_wavelength_A[transitionnumber] / 1e10) ** 2
                )  # convert wavelength from angstrom to m
                * (g_arr[l] / g_arr[u])
                * oscillator_strength[transitionnumber]
            )
            for transitionnumber, (l, u) in enumerate(zip(lowerlevel, upperlevel))
        ]
    )

    transitions = [
        TransitionTuple(
            lowerlevel=lowerlevel[transitionnumber],
            upperlevel=upperlevel[transitionnumber],
            A=A_ul[transitionnumber],
            coll_str=-1,
        )
        for transitionnumber, _ in enumerate(lowerlevel)
    ]

    transition_count_of_level_name = defaultdict(int)
    for level_number_lower, level_number_upper in zip(lowerlevel, upperlevel):
        transition_count_of_level_name[level_number_lower] += 1
        transition_count_of_level_name[level_number_upper] += 1

    assert len(transitions) == len(
        energy_levels_lower_1000percm
    )  # check number of transitions is the same as the number read in

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name


read_levels_and_transitions(atomic_number=57, ion_stage=5, flog=None)
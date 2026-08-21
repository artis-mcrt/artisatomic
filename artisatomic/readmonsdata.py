"""Read levels and transitions from the Carvajal Gallego et al. (University of Mons) lanthanide V-VII data set.

Data: https://doi.org/10.5281/zenodo.10635803 (see atomic-data-mons/readme.txt for the file formats).
"""

import os
import zipfile

import numpy as np
import polars as pl

import artisatomic
from artisatomic.base import elsymbols
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import log_and_print
from artisatomic.base import PYDIR
from artisatomic.base import roman_numerals

datafilepath = (PYDIR / ".." / "atomic-data-mons").resolve()
if os.environ.get("ARTISATOMIC_TESTMODE") == "1":
    # a reduced Ce V and Ce VI model cut from the full archives (see atomic-data-mons/readme.txt)
    datafilepath /= "test_sample"

# the archives hold La-Lu (Z=57-71) in ion stages V-VII
mons_atomic_numbers = range(57, 72)
mons_ion_stages = (5, 6, 7)


def extend_ion_list(ion_handlers):
    """Add every lanthanide V-VII ion of the MONS data set to ion_handlers."""
    for atomic_number in mons_atomic_numbers:
        for ion_stage in mons_ion_stages:
            ion_handlers = artisatomic.add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "mons")

    return ion_handlers


def get_nearest_level_indices(sorted_energies: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """Return the index of the closest value in sorted_energies for each value in energies.

    The transition file gives the lower level energy and the wavelength, so both levels of a
    transition are found by their energy. The search is binary, so the full data set (up to ten
    million transitions per ion) does not need a distance matrix. A tie goes to the lower index.
    """
    idx_right = np.searchsorted(sorted_energies, energies).clip(1, len(sorted_energies) - 1)
    idx_left = idx_right - 1
    dist_left = np.abs(energies - sorted_energies[idx_left])
    dist_right = np.abs(energies - sorted_energies[idx_right])
    return np.where(dist_left <= dist_right, idx_left, idx_right)


def read_levels_and_transitions(atomic_number: int, ion_stage: int, flog):
    """Read one ion from the MONS zip archives.

    The level file gives the energy (in 1000 cm^-1) and J of each level, in no particular order.
    The transition file gives the wavelength, the lower level energy and the oscillator strength
    of each E1 transition. The levels of a transition are found by their energy: the upper level
    energy is the lower level energy plus the photon energy. The level names are the energies.
    The data set has no parity, and every transition is E1, so no transition is forbidden.
    The ionization energy comes from NIST.
    """
    elsym = elsymbols[atomic_number]
    ionstr = roman_numerals[ion_stage]

    with (
        zipfile.ZipFile(datafilepath / "outglv_Ln_V--VII.zip", "r") as ziparchive_outglv,
        ziparchive_outglv.open(f"outglv_Ln_V--VII/outglv_0_{elsym}_{ionstr}") as datafile_energylevels,
    ):
        energy_levels1000percm, j_arr = np.loadtxt(datafile_energylevels, unpack=True, delimiter=",")
    log_and_print(flog, f"levels: {len(energy_levels1000percm)}")

    sortorder = np.argsort(energy_levels1000percm, kind="stable")
    energiesabovegsinpercm = energy_levels1000percm[sortorder] * 1000
    g_arr = 2 * j_arr[sortorder] + 1

    dflevels = pl.DataFrame(
        {
            "levelname": [str(energy) for energy in energiesabovegsinpercm],
            "energyabovegsinpercm": energiesabovegsinpercm,
            "g": g_arr,
        }
    ).with_columns(parity=pl.lit(None, dtype=pl.Int64))

    with (
        zipfile.ZipFile(datafilepath / "outggf_Ln_V--VII.zip", "r") as ziparchive_outggf,
        ziparchive_outggf.open(f"outggf_Ln_V--VII/outggf_sorted_{elsym}_{ionstr}") as datafile_transitions,
    ):
        transition_wavelength_A, energy_levels_lower_1000percm, oscillator_strength = np.loadtxt(
            datafile_transitions, unpack=True, delimiter=","
        )
    log_and_print(flog, f"transitions: {len(energy_levels_lower_1000percm)}")

    energy_levels_lower_percm = energy_levels_lower_1000percm * 1000
    energy_levels_upper_percm = energy_levels_lower_percm + 1e8 / transition_wavelength_A
    lowerlevels = get_nearest_level_indices(energiesabovegsinpercm, energy_levels_lower_percm)
    upperlevels = get_nearest_level_indices(energiesabovegsinpercm, energy_levels_upper_percm)

    # the energies in the transition file are rounded, so a small mismatch is expected. A large
    # mismatch means that a transition refers to a level that is not in the level file.
    maxmismatch_percm = max(
        np.abs(energiesabovegsinpercm[lowerlevels] - energy_levels_lower_percm).max(),
        np.abs(energiesabovegsinpercm[upperlevels] - energy_levels_upper_percm).max(),
    )
    log_and_print(flog, f"largest difference between a transition energy and its level: {maxmismatch_percm:.3g} cm^-1")
    if maxmismatch_percm > 1.0:
        log_and_print(flog, "WARNING: a transition does not match a level to within 1 cm^-1")

    ionization_energy_in_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]
    log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV (NIST)")

    # A = f / (g_u / g_l * 1.3473837e21 / nu^2) with nu in Hz and the wavelength in Angstrom
    OSCSTRENGTHCONVERSION = 1.3473837e21
    c_angps = 2.99792458e18
    A_ul = oscillator_strength / (
        g_arr[upperlevels] / g_arr[lowerlevels] * OSCSTRENGTHCONVERSION / (c_angps / transition_wavelength_A) ** 2
    )

    # level ids are zero-based in memory
    dftransitions = pl.DataFrame({"lowerlevel": lowerlevels, "upperlevel": upperlevels, "A": A_ul}).with_columns(
        pl.col("lowerlevel").cast(pl.Int64), pl.col("upperlevel").cast(pl.Int64)
    )

    transition_count_of_levelid: dict[int, int] = dict(
        pl.concat([dftransitions["lowerlevel"], dftransitions["upperlevel"]]).value_counts().iter_rows()
    )
    transition_count_of_level_name = {
        levelname: transition_count_of_levelid.get(levelid, 0)
        for levelid, levelname in enumerate(dflevels["levelname"].to_list())
    }

    return ionization_energy_in_ev, dflevels, dftransitions, transition_count_of_level_name

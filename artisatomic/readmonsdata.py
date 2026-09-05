"""Read levels and transitions from the Carvajal Gallego et al. (University of Mons) lanthanide V-VII data set.

Data: https://doi.org/10.5281/zenodo.10635803 (see atomic-data-mons/readme.txt for the file formats).
"""

import zipfile

import numpy as np
import polars as pl

from artisatomic.base import add_handler_if_not_set
from artisatomic.base import elsymbols
from artisatomic.base import empty_transitions_schema
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import gf_to_a_coefficient
from artisatomic.base import log_and_print
from artisatomic.base import PYDIR
from artisatomic.base import roman_numerals
from artisatomic.base import TESTMODE

datafilepath = (PYDIR / ".." / "atomic-data-mons").resolve()
if TESTMODE:
    # a reduced Ce V and Ce VI sample cut from the full archives (see tests/README.md)
    datafilepath /= "test_sample"

levels_archive = "outglv_Ln_V--VII.zip"
transitions_archive = "outggf_Ln_V--VII.zip"

# The transition file quotes the lower level energy to about 8 significant digits, so it differs
# from the level file by up to 0.02 cm^-1. A larger difference means that the two files disagree.
MATCH_TOLERANCE_PERCM = 0.05


def levels_member(atomic_number: int, ion_stage: int) -> str:
    """Name of the level file of one ion in the levels archive."""
    return f"outglv_Ln_V--VII/outglv_0_{elsymbols[atomic_number]}_{roman_numerals[ion_stage]}"


def transitions_member(atomic_number: int, ion_stage: int) -> str:
    """Name of the transition file of one ion in the transitions archive."""
    return f"outggf_Ln_V--VII/outggf_sorted_{elsymbols[atomic_number]}_{roman_numerals[ion_stage]}"


def read_csv_columns(archivename: str, membername: str, columncount: int) -> list[np.ndarray]:
    """Read the columns of one comma-separated member of a zip archive.

    The files pad the fields with spaces, which polars cannot convert to a float directly. The
    reader takes each column as text, removes the spaces, and then converts it.
    """
    columnnames = [f"column{i}" for i in range(columncount)]
    with zipfile.ZipFile(datafilepath / archivename) as ziparchive, ziparchive.open(membername) as datafile:
        dfcolumns = pl.read_csv(
            datafile,
            has_header=False,
            new_columns=columnnames,
            separator=",",
            schema_overrides=dict.fromkeys(columnnames, pl.String),
        ).select(pl.col(name).str.strip_chars().cast(pl.Float64) for name in columnnames)

    return [dfcolumns[name].to_numpy() for name in columnnames]


def extend_ion_list(ion_handlers):
    """Add every ion with a MONS level file to ion_handlers.

    The archive holds La-Lu (Z=57-71) in the ion stages V-VII. The list comes from the archive
    itself, so a run offers only the ions that the downloaded data covers.
    """
    with zipfile.ZipFile(datafilepath / levels_archive) as ziparchive:
        membernames = set(ziparchive.namelist())

    for atomic_number in range(57, 72):
        for ion_stage in (5, 6, 7):
            if levels_member(atomic_number, ion_stage) in membernames:
                ion_handlers = add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "mons")

    return ion_handlers


def get_nearest_level_indices(sorted_energies: np.ndarray, energies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the closest and the second closest level for each energy.

    The transition file gives the lower level energy and the wavelength, so the reader finds both
    levels of a transition by their energy. The search is binary, so the full data set (up to 45
    million transitions per ion) needs no distance matrix. A tie goes to the lower index.
    """
    if len(sorted_energies) < 2:
        return np.zeros(len(energies), dtype=np.int64), np.zeros(len(energies), dtype=np.int64)

    idx_right = np.searchsorted(sorted_energies, energies).clip(1, len(sorted_energies) - 1)
    idx_left = idx_right - 1
    leftisnearer = np.abs(energies - sorted_energies[idx_left]) <= np.abs(energies - sorted_energies[idx_right])

    return np.where(leftisnearer, idx_left, idx_right), np.where(leftisnearer, idx_right, idx_left)


def read_levels_and_transitions(atomic_number: int, ion_stage: int, flog):
    """Read one ion from the MONS zip archives.

    The level file gives the energy (in 1000 cm^-1) and J of each level, in no particular order.
    The transition file gives the wavelength, the lower level energy and the weighted oscillator
    strength of each E1 transition. The reader finds the levels of a transition by their energy:
    the upper level energy is the lower level energy plus the photon energy. Each level name is
    the energy plus the zero-based level id, so two levels with one energy keep separate names.
    NIST supplies the ionisation energy.
    """
    energy_levels1000percm, j_arr = read_csv_columns(levels_archive, levels_member(atomic_number, ion_stage), 2)
    log_and_print(flog, f"levels: {len(energy_levels1000percm)}")

    sortorder = np.argsort(energy_levels1000percm, kind="stable")
    energiesabovegsinpercm = energy_levels1000percm[sortorder] * 1000
    j_arr = j_arr[sortorder]
    g_arr = 2 * j_arr + 1

    dflevels = pl.DataFrame(
        {
            # the id makes the name unique: two levels with one energy would otherwise share a name
            "levelname": [f"{energy},id={levelid}" for levelid, energy in enumerate(energiesabovegsinpercm)],
            "energyabovegsinpercm": energiesabovegsinpercm,
            "g": g_arr,
            "j": j_arr,
        }
    ).with_columns(parity=pl.lit(None, dtype=pl.Int64))

    transition_wavelength_A, energy_levels_lower_1000percm, weighted_oscillator_strength = read_csv_columns(
        transitions_archive, transitions_member(atomic_number, ion_stage), 3
    )
    log_and_print(flog, f"transitions: {len(energy_levels_lower_1000percm)}")

    energy_levels_lower_percm = energy_levels_lower_1000percm * 1000
    energy_levels_upper_percm = energy_levels_lower_percm + 1e8 / transition_wavelength_A
    lowerlevels, lowerlevels_second = get_nearest_level_indices(energiesabovegsinpercm, energy_levels_lower_percm)
    upperlevels, upperlevels_second = get_nearest_level_indices(energiesabovegsinpercm, energy_levels_upper_percm)

    maxmismatch_percm = max(
        np.abs(energiesabovegsinpercm[lowerlevels] - energy_levels_lower_percm).max(),
        np.abs(energiesabovegsinpercm[upperlevels] - energy_levels_upper_percm).max(),
    )
    log_and_print(flog, f"largest difference between a transition energy and its level: {maxmismatch_percm:.3g} cm^-1")
    if maxmismatch_percm > MATCH_TOLERANCE_PERCM:
        msg = (
            f"A MONS transition of Z={atomic_number} ion stage {ion_stage} is {maxmismatch_percm:.3g} cm^-1"
            f" from its closest level. The tolerance is {MATCH_TOLERANCE_PERCM} cm^-1."
            " The level file and the transition file do not match."
        )
        raise ValueError(msg)

    # a second level inside the match tolerance makes the match a guess. The reader keeps the
    # closest level, and the count says how much of the ion depends on that choice.
    ambiguouscount = int(
        (np.abs(energy_levels_lower_percm - energiesabovegsinpercm[lowerlevels_second]) <= MATCH_TOLERANCE_PERCM).sum()
        + (
            np.abs(energy_levels_upper_percm - energiesabovegsinpercm[upperlevels_second]) <= MATCH_TOLERANCE_PERCM
        ).sum()
    )
    if ambiguouscount > 0:
        log_and_print(flog, f"WARNING: {ambiguouscount} level matches have a second level equally close")

    ionization_energy_in_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]
    log_and_print(flog, f"ionisation energy: {ionization_energy_in_ev} eV (NIST)")

    # the third column of the transition file is gf, not f: single lines reach gf = 25. The sum of
    # gf / g_lower over the lines of one level reaches the electron count, while the sum of gf does
    # not.
    A_ul = weighted_oscillator_strength / (gf_to_a_coefficient * g_arr[upperlevels] * transition_wavelength_A**2)

    # level ids are zero-based in memory
    dftransitions = pl.DataFrame(
        {"lowerlevel": lowerlevels, "upperlevel": upperlevels, "A": A_ul}, schema=empty_transitions_schema
    )

    return ionization_energy_in_ev, dflevels, dftransitions

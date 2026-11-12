"""Read levels and transitions from the University of Mons lanthanide V-VII data set.

Paper: Carvajal Gallego, H., Deprince, J., Maison, L., Palmeri, P., Quinet, P. (2024), A&A, 685, A91,
doi:10.1051/0004-6361/202347723. Data: https://doi.org/10.5281/zenodo.10635803 (see
atomic-data-mons/readme.txt for the file formats).
"""

import io
import zipfile
from collections.abc import Iterator

import numpy as np
import polars as pl

from artisatomic.base import add_handlers_if_not_set
from artisatomic.base import elsymbols
from artisatomic.base import empty_transitions_schema
from artisatomic.base import gf_to_a_coefficient
from artisatomic.base import log_and_print
from artisatomic.base import log_comment
from artisatomic.base import nist_ionization_energy_ev
from artisatomic.base import path_in_data_folder
from artisatomic.base import PYDIR
from artisatomic.base import roman_numerals
from artisatomic.base import TESTMODE

monsfolder = PYDIR / ".." / "atomic-data-mons"
datafilepath = monsfolder.resolve()
if TESTMODE:
    # a reduced Ce V and Ce VI sample cut from the full archives (see tests/README.md)
    datafilepath /= "test_sample"

levels_archive = "outglv_Ln_V--VII.zip"
transitions_archive = "outggf_Ln_V--VII.zip"

# the "source:" line of the comment blocks in the output files (see Handler.description in iondata.py)
description = (
    "the University of Mons data set of the lanthanides V-VII. Carvajal Gallego, H., Deprince, J., Maison, L.,"
    " Palmeri, P., Quinet, P. (2024), A&A, 685, A91, doi:10.1051/0004-6361/202347723. Data set:"
    " doi:10.5281/zenodo.10635803"
)

# The transition file adds a tag to each level energy (see tagged_level_energies()), so it differs
# from the level file by the tag. The largest tag of the full level archive is 0.034 cm^-1. A larger
# difference means that the two files disagree.
MATCH_TOLERANCE_PERCM = 0.05

# The tagged energies of the two files agree to 2e-13 (1000 cm^-1) in the sample.
TAG_TOLERANCE_1000PERCM = 1e-9


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
    with zipfile.ZipFile(datafilepath / archivename) as ziparchive, ziparchive.open(membername) as datafile:
        return parse_csv_columns(datafile, archivename, membername, columncount)


# A batch ends at a complete row. The source files contain numeric fields without embedded newlines.
TRANSITION_BATCH_BYTES = 8 * 1024 * 1024


def iter_csv_columns(archivename: str, membername: str, columncount: int) -> Iterator[list[np.ndarray]]:
    """Read one archive member in batches of complete rows."""
    with zipfile.ZipFile(datafilepath / archivename) as ziparchive, ziparchive.open(membername) as datafile:
        remainder = b""
        while chunk := datafile.read(TRANSITION_BATCH_BYTES):
            complete, separator, remainder = (remainder + chunk).rpartition(b"\n")
            if separator:
                yield parse_csv_columns(io.BytesIO(complete + separator), archivename, membername, columncount)
        if remainder:
            yield parse_csv_columns(io.BytesIO(remainder), archivename, membername, columncount)


def parse_csv_columns(datafile, archivename: str, membername: str, columncount: int) -> list[np.ndarray]:
    """Read finite numeric columns from a complete table or a batch of rows."""
    columnnames = [f"column{i}" for i in range(columncount)]
    dfcolumns = (
        pl.read_csv(
            datafile,
            has_header=False,
            new_columns=columnnames,
            separator=",",
            schema_overrides=dict.fromkeys(columnnames, pl.String),
        )
        .lazy()
        .select(pl.col(name).str.strip_chars().cast(pl.Float64) for name in columnnames)
        .collect()
    )

    if dfcolumns.select(pl.all().is_finite().fill_null(value=False).all()).row(0) != (True,) * columncount:
        msg = f"{membername} in {archivename} has a value that is not a finite number, or an empty value"
        raise ValueError(msg)
    return [dfcolumns[name].to_numpy() for name in columnnames]


def extend_ion_list(
    ion_handlers,
):
    """Add every ion with a MONS level file to ion_handlers.

    The archive holds La-Lu (Z=57-71) in the ion stages V-VII. The list comes from the archive
    itself, so a run offers only the ions that the downloaded data covers.
    """
    with zipfile.ZipFile(datafilepath / levels_archive) as ziparchive:
        membernames = set(ziparchive.namelist())

    monsions = [
        (atomic_number, ion_stage)
        for atomic_number in range(57, 72)
        for ion_stage in (5, 6, 7)
        if levels_member(atomic_number, ion_stage) in membernames
    ]

    return add_handlers_if_not_set(
        ion_handlers,
        monsions,
        "mons",
    )


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
    nearest = np.where(leftisnearer, idx_left, idx_right)

    # the second closest level is a neighbour of the closest level, and it can be on the same side of the energy
    below = (nearest - 1).clip(0)
    above = (nearest + 1).clip(max=len(sorted_energies) - 1)
    distance_below = np.where(nearest > 0, np.abs(energies - sorted_energies[below]), np.inf)
    distance_above = np.where(above > nearest, np.abs(energies - sorted_energies[above]), np.inf)

    return nearest, np.where(distance_below <= distance_above, below, above)


def tagged_level_energies(energies1000percm: np.ndarray, j_arr: np.ndarray) -> np.ndarray:
    """Give the energy of each level with the tag of the transition file, in 1000 cm^-1.

    The transition file gives the energy of a level as E + J * 1e-6 + n * 1e-8. n is the position
    of the level in its block of one J in the level file, from 1. The tag thus names one level,
    also where two levels have almost the same energy.
    """
    levelpositions = np.arange(len(j_arr))
    newblock = np.r_[True, j_arr[1:] != j_arr[:-1]]
    blockstarts = np.maximum.accumulate(np.where(newblock, levelpositions, 0))
    return energies1000percm + j_arr * 1e-6 + (levelpositions - blockstarts + 1) * 1e-8


def match_tagged_levels(
    tagged1000percm: np.ndarray, sorted_energies_percm: np.ndarray, energies1000percm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the level of each transition energy.

    tagged1000percm holds the tagged energy of each level id. The match uses the tagged energy
    first. An energy with no tagged level inside TAG_TOLERANCE_1000PERCM takes the closest level
    energy. The function gives the level ids, the positions of the energies without a tag, and the
    second closest level id of each such energy. Only those energies get the second search, because
    an ion can have 45 million transitions.
    """
    order = np.argsort(tagged1000percm, kind="stable")
    sorted_tagged = tagged1000percm[order]
    if len(order) > 1:
        right = np.searchsorted(sorted_tagged, energies1000percm).clip(1, len(order) - 1)
        left = right - 1
        leftisnearer = np.abs(energies1000percm - sorted_tagged[left]) <= np.abs(
            energies1000percm - sorted_tagged[right]
        )
        levels = order[np.where(leftisnearer, left, right)]
        del right, left, leftisnearer
    else:
        levels = np.zeros(len(energies1000percm), int)
    untagged = np.flatnonzero(np.abs(tagged1000percm[levels] - energies1000percm) > TAG_TOLERANCE_1000PERCM)
    closest, second = get_nearest_level_indices(sorted_energies_percm, energies1000percm[untagged] * 1000)
    levels[untagged] = closest
    return levels, untagged, second


def read_levels_and_transitions(atomic_number: int, ion_stage: int, flog):
    """Read one ion from the MONS zip archives.

    The level file gives the energy (in 1000 cm^-1) and J of each level, in no particular order.
    The transition file gives the wavelength, the lower level energy and the weighted oscillator
    strength of each E1 transition. The reader finds the levels of a transition by their tagged
    energy (see tagged_level_energies()). The upper level energy is the lower level energy plus the
    photon energy. Each level name is
    the energy plus the zero-based level id, so two levels with one energy keep separate names.
    NIST supplies the ionisation energy.
    """
    # the path of the archive, then the name of its member
    levelspath = path_in_data_folder(datafilepath / levels_archive, monsfolder)
    log_comment(flog, ("adata",), f"The levels come from {levelspath}/{levels_member(atomic_number, ion_stage)}.")
    energy_levels1000percm, j_arr = read_csv_columns(levels_archive, levels_member(atomic_number, ion_stage), 2)
    log_and_print(flog, f"The reader got {len(energy_levels1000percm)} levels.")

    sortorder = np.argsort(energy_levels1000percm, kind="stable")
    tagged_levels1000percm = tagged_level_energies(energy_levels1000percm, j_arr)[sortorder]
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

    transitionspath = path_in_data_folder(datafilepath / transitions_archive, monsfolder)
    log_comment(
        flog,
        ("transitiondata",),
        f"The transitions come from {transitionspath}/{transitions_member(atomic_number, ion_stage)}.",
    )
    batches: list[pl.DataFrame] = []
    maxmismatch_percm = 0.0
    untaggedcount = 0
    ambiguouscount = 0
    for columns in iter_csv_columns(transitions_archive, transitions_member(atomic_number, ion_stage), 3):
        batch, mismatch, untagged, ambiguous = match_transition_batch(
            columns, tagged_levels1000percm, energiesabovegsinpercm, g_arr
        )
        batches.append(batch)
        maxmismatch_percm = max(maxmismatch_percm, mismatch)
        untaggedcount += untagged
        ambiguouscount += ambiguous
    dftransitions = pl.concat(batches, rechunk=False) if batches else pl.DataFrame(schema=empty_transitions_schema)
    log_and_print(flog, f"The reader got {dftransitions.height} transitions.")

    log_comment(
        flog,
        ("transitiondata",),
        "The reader finds the levels of a transition by the tagged energy E + J * 1e-6 + n * 1e-8 (1000 cm^-1) of the"
        " transition file. n is the position of the level in its block of one J in the level file, from 1. The"
        f" largest difference between the energies of the two files is {maxmismatch_percm:.3g} cm^-1.",
    )
    if untaggedcount > 0:
        log_comment(
            flog,
            ("transitiondata",),
            f"WARNING: {untaggedcount} of {2 * dftransitions.height} level matches found no tagged energy. They take the"
            " closest level energy.",
        )
    if maxmismatch_percm > MATCH_TOLERANCE_PERCM:
        msg = (
            f"A MONS transition of Z={atomic_number} ion stage {ion_stage} is {maxmismatch_percm:.3g} cm^-1"
            f" from its closest level. The tolerance is {MATCH_TOLERANCE_PERCM} cm^-1."
            " The level file and the transition file do not match."
        )
        raise ValueError(msg)

    if ambiguouscount > 0:
        log_comment(
            flog,
            ("transitiondata",),
            f"WARNING: For {ambiguouscount} level matches, a second level is equally close in energy.",
        )

    ionization_energy_in_ev = nist_ionization_energy_ev(atomic_number, ion_stage, flog)

    return ionization_energy_in_ev, dflevels, dftransitions


def match_transition_batch(
    columns: list[np.ndarray], tagged_levels1000percm: np.ndarray, energiesabovegsinpercm: np.ndarray, g_arr: np.ndarray
) -> tuple[pl.DataFrame, float, int, int]:
    """Match one batch and give its transitions and summary values."""
    transition_wavelength_A, energy_levels_lower_1000percm, weighted_oscillator_strength = columns
    energy_levels_upper_1000percm = energy_levels_lower_1000percm + 1e5 / transition_wavelength_A
    energy_levels_lower_percm = energy_levels_lower_1000percm * 1000
    energy_levels_upper_percm = energy_levels_upper_1000percm * 1000
    lowerlevels, lower_untagged, lowerlevels_second = match_tagged_levels(
        tagged_levels1000percm, energiesabovegsinpercm, energy_levels_lower_1000percm
    )
    upperlevels, upper_untagged, upperlevels_second = match_tagged_levels(
        tagged_levels1000percm, energiesabovegsinpercm, energy_levels_upper_1000percm
    )

    maxmismatch_percm = max(
        np.abs(energiesabovegsinpercm[lowerlevels] - energy_levels_lower_percm).max(),
        np.abs(energiesabovegsinpercm[upperlevels] - energy_levels_upper_percm).max(),
    )
    ambiguouscount = int(
        (
            np.abs(energy_levels_lower_percm[lower_untagged] - energiesabovegsinpercm[lowerlevels_second])
            <= MATCH_TOLERANCE_PERCM
        ).sum()
        + (
            np.abs(energy_levels_upper_percm[upper_untagged] - energiesabovegsinpercm[upperlevels_second])
            <= MATCH_TOLERANCE_PERCM
        ).sum()
    )
    untaggedcount = len(lower_untagged) + len(upper_untagged)
    A_ul = weighted_oscillator_strength / (gf_to_a_coefficient * g_arr[upperlevels] * transition_wavelength_A**2)
    dftransitions = pl.DataFrame(
        {"lowerlevel": lowerlevels, "upperlevel": upperlevels, "A": A_ul}, schema=empty_transitions_schema
    )
    return dftransitions, float(maxmismatch_percm), untaggedcount, ambiguouscount

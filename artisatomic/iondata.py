"""Read a single ion's levels, transitions, and photoionisation data from its source dataset."""

import argparse
import contextlib
import itertools
import typing as t
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from artisatomic import groundstatesonlynist
from artisatomic import readboyledata
from artisatomic import readdreamdata
from artisatomic import readfacdata
from artisatomic import readfloers25data
from artisatomic import readhillierdata
from artisatomic import readkuruczdata
from artisatomic import readlisbondata
from artisatomic import readmonsdata
from artisatomic import readqubdata
from artisatomic import readtanakajpltdata
from artisatomic.base import elsymbols
from artisatomic.base import ion_log_path
from artisatomic.base import leveltuples_to_pldataframe
from artisatomic.base import log_and_print
from artisatomic.base import roman_numerals
from artisatomic.phixs import match_hydrogenic_phixs


@dataclass(slots=True)
class IonData:
    """Levels, transitions, and photoionisation data for a single ion, read from one of the source datasets.

    A dataclass rather than a NamedTuple because resolve_photoion_targetfractions() fills in the
    target fractions after the whole element has been read.
    """

    ion_stage: int
    handler: str
    # the top ion has no upper ion, so it gets neither cross sections nor a phixs block. Recorded
    # here at read time so the reading, resolving and writing passes cannot disagree about it.
    is_top_ion: bool
    ionization_energy_ev: float
    dfenergylevels: pl.DataFrame
    dftransitions: pl.DataFrame
    transition_count_of_level_name: dict[str, int]
    upsilondict: dict[tuple[int, int], float]
    # None where a level has no photoionisation data (and None entirely if none was read).
    # readhillierdata.get_photoiontargetfractions() matches these names to the upper ion's
    # levels. It follows the CMFGEN conventions for a name. That reader therefore resolves
    # this field for every handler, and returns an empty result when the field is None.
    photoion_targetconfigs: list[list[tuple[str, float]] | None] | None
    photoionization_crosssections: npt.NDArray[np.float64]  # cross sections in Mb, indexed by level id
    photoionization_targetfractions: list[list[tuple[int, float]]]  # indexed by level id
    photoionization_thresholds_ev: npt.NDArray[np.float64]  # indexed by level id


@dataclass(frozen=True, slots=True)
class SimpleHandler:
    """One entry of the simple-handler registry.

    The reader takes (atomic_number, ion_stage, flog) and returns (ionization_energy_ev,
    energy_levels, transitions, transition_count_of_level_name), with an upsilondict appended
    when returns_upsilondict is set. The shapes differ by design, so the Callable stays untyped
    in its return. get_level_valence_n is the handler's own level-name parser. The
    hydrogenic photoionisation estimate uses it. None means that the handler has no parser.
    """

    read_levels_and_transitions: Callable[..., tuple[t.Any, ...]]
    get_level_valence_n: Callable[[str], int] | None = None
    returns_upsilondict: bool = False


simple_handlers: dict[str, SimpleHandler] = {
    "boyle": SimpleHandler(
        lambda atomic_number, ion_stage, _flog: readboyledata.read_levels_and_transitions(atomic_number, ion_stage)
    ),
    "kurucz": SimpleHandler(readkuruczdata.read_levels_and_transitions, readkuruczdata.get_level_valence_n),
    "dream": SimpleHandler(readdreamdata.read_levels_and_transitions),  # DREAM database of Z >= 57
    "lisbon": SimpleHandler(readlisbondata.read_levels_and_transitions),
    "floers25calibwithforbidden": SimpleHandler(
        partial(readfloers25data.read_levels_and_transitions, calibrated=True, withforbidden=True),
        readfloers25data.get_level_valence_n,
    ),
    "floers25calib": SimpleHandler(
        partial(readfloers25data.read_levels_and_transitions, calibrated=True),
        readfloers25data.get_level_valence_n,
    ),
    "floers25uncalib": SimpleHandler(
        partial(readfloers25data.read_levels_and_transitions, calibrated=False),
        readfloers25data.get_level_valence_n,
    ),
    # fac reads an early version of the floers25 calib data
    "fac": SimpleHandler(readfacdata.read_levels_and_transitions, readfacdata.get_level_valence_n),
    "mons": SimpleHandler(
        readmonsdata.read_levels_and_transitions  # Carvajal Gallego et al. (University of Mons) lanthanides V-VII
    ),
    "tanakajplt": SimpleHandler(
        readtanakajpltdata.read_levels_and_transitions,  # Tanaka Japan-Lithuania database of 26 <= Z <= 88
        readtanakajpltdata.get_level_valence_n,
    ),
    "gsnist": SimpleHandler(groundstatesonlynist.read_ground_levels),  # ground states taken from NIST
    "qub_data": SimpleHandler(
        readqubdata.read_qub_levels_and_transitions,
        readqubdata.get_level_valence_n,
        returns_upsilondict=True,
    ),
}


def read_ion_data(
    atomic_number: int, ion_stage_entry: tuple[int, str], is_top_ion: bool, args: argparse.Namespace
) -> IonData:
    """Read a single ion's data from its source dataset.

    Every ion names the handler that reads it; there is no default per element.
    """
    ion_stage, handler = ion_stage_entry
    simplehandler = simple_handlers.get(handler)

    # no default for ionization_energy_ev: every handler branch below sets it, and an unknown
    # handler raises, so a branch that forgets should fail rather than write a 0 eV threshold
    transition_count_of_level_name: dict[str, int] = {}
    upsilondict: dict[tuple[int, int], float] = {}
    photoion_targetconfigs: list[list[tuple[str, float]] | None] | None = None
    # empty until a handler below reads photoionisation data (and left empty for the top ion)
    photoionization_crosssections: npt.NDArray[np.float64] = np.empty((0, args.nphixspoints))  # in Mb
    photoionization_targetfractions: list[list[tuple[int, float]]] = []
    photoionization_thresholds_ev: npt.NDArray[np.float64] = np.empty(0)

    logfilepath = ion_log_path(Path(args.output_folder, args.output_folder_logs), atomic_number, ion_stage)
    with logfilepath.open("w", encoding="utf-8") as flog:
        log_and_print(
            flog,
            f"\n===========> Z={atomic_number} {elsymbols[atomic_number]} {roman_numerals[ion_stage]} input:",
        )
        log_and_print(flog, f"Source handler: {handler}")
        if handler == "qub_cobalt":
            if ion_stage in readqubdata.qub_cobalt_stages:
                (
                    ionization_energy_ev,
                    energy_levels,
                    transitions,
                    transition_count_of_level_name,
                    upsilondict,
                ) = readqubdata.read_qub_levels_and_transitions(atomic_number, ion_stage, flog)
            else:  # hillier levels and transitions
                (
                    ionization_energy_ev,
                    energy_levels,
                    transitions,
                    transition_count_of_level_name,
                ) = readhillierdata.read_levels_and_transitions(atomic_number, ion_stage, flog)

            if not is_top_ion and not args.nophixs:  # don't get cross sections for top ion
                (
                    photoionization_crosssections,
                    photoionization_targetfractions,
                    photoionization_thresholds_ev,
                ) = readqubdata.read_qub_photoionizations(
                    atomic_number, ion_stage, levelcount=len(energy_levels), args=args, flog=flog
                )

        elif handler == "cmfgen":  # Hillier CMFGEN data only
            (
                ionization_energy_ev,
                energy_levels,
                transitions,
                transition_count_of_level_name,
            ) = readhillierdata.read_levels_and_transitions(atomic_number, ion_stage, flog)

            upsilondict = readhillierdata.read_coldata(atomic_number, ion_stage, energy_levels, flog, args)

            if not is_top_ion and not args.nophixs:  # don't get cross sections for top ion
                (
                    photoionization_crosssections,
                    photoion_targetconfigs,
                    photoionization_thresholds_ev,
                ) = readhillierdata.read_phixs_tables(atomic_number, ion_stage, energy_levels, args, flog)

        elif simplehandler is not None:
            result = simplehandler.read_levels_and_transitions(atomic_number, ion_stage, flog)
            if simplehandler.returns_upsilondict:
                (ionization_energy_ev, energy_levels, transitions, transition_count_of_level_name, upsilondict) = result
            else:
                (ionization_energy_ev, energy_levels, transitions, transition_count_of_level_name) = result

        else:
            msg = f"Unknown handler: {handler}"
            raise ValueError(msg)

    dfenergylevels = leveltuples_to_pldataframe(energy_levels)

    # the len() == 0 test is what limits the estimate to ions the handler gave nothing for: an ion
    # with even one cross-section table is left alone, so measured data is never replaced. The top
    # ion is excluded because there is no upper ion for it to photoionise to.
    if (
        not is_top_ion
        and not args.nophixs
        and len(photoionization_crosssections) == 0
        and args.nlevels_hydrogenic_for_unknown_phixs > 0
    ):
        get_level_valence_n = simplehandler.get_level_valence_n if simplehandler is not None else None
        (
            photoionization_crosssections,
            photoionization_targetfractions,
            photoionization_thresholds_ev,
        ) = match_hydrogenic_phixs(
            atomic_number, dfenergylevels, ionization_energy_ev, handler, get_level_valence_n, args
        )

    dftransitions = transitions if isinstance(transitions, pl.DataFrame) else pl.DataFrame(transitions)

    return IonData(
        ion_stage=ion_stage,
        handler=handler,
        is_top_ion=is_top_ion,
        ionization_energy_ev=ionization_energy_ev,
        dfenergylevels=dfenergylevels,
        dftransitions=dftransitions,
        transition_count_of_level_name=transition_count_of_level_name,
        upsilondict=upsilondict,
        photoion_targetconfigs=photoion_targetconfigs,
        photoionization_crosssections=photoionization_crosssections,
        photoionization_targetfractions=photoionization_targetfractions,
        photoionization_thresholds_ev=photoionization_thresholds_ev,
    )


def resolve_photoion_targetfractions(
    iondatalist: list[IonData], atomic_number: int | None = None, log_folder: str | Path | None = None
) -> None:
    """Fill in the photoionisation target fractions of each ion whose reader supplied none.

    An ion's targets are levels of the next ion up, so the fractions can only be resolved once the
    whole element has been read. The top ion is left alone (it has no upper ion to photoionise
    to), as is any ion whose reader already gave per-level fractions.

    Call this before write_output_files() unless cross sections are switched off entirely; the
    writer needs the fractions and does not resolve them itself. Give atomic_number and log_folder
    to append the resolve messages to each ion's log file.
    """
    # a half-given pair is always a caller bug, so fail loudly rather than skip the log silently
    if (atomic_number is None) != (log_folder is None):
        msg = "give both atomic_number and log_folder, or neither"
        raise ValueError(msg)

    if not iondatalist:
        return

    # the ions must be one element's, in ascending stage order: each is resolved against the next
    # entry as its upper ion. A top ion anywhere but last means the list is not that, and the
    # levels being read as targets would belong to the wrong ion.
    if not iondatalist[-1].is_top_ion or any(iondata.is_top_ion for iondata in iondatalist[:-1]):
        msg = (
            "iondatalist must be one element's ions in ascending ion stage order, with only the last being the top ion"
        )
        raise ValueError(msg)

    for iondata, upperiondata in itertools.pairwise(iondatalist):
        if not iondata.photoionization_targetfractions:
            # The reading pass closed the per-ion log. Reopen it in append mode.
            logcontext = (
                ion_log_path(log_folder, atomic_number, iondata.ion_stage).open("a", encoding="utf-8")
                if log_folder is not None and atomic_number is not None
                else contextlib.nullcontext()
            )
            with logcontext as flog:
                iondata.photoionization_targetfractions = readhillierdata.get_photoiontargetfractions(
                    iondata.dfenergylevels,
                    upperiondata.dfenergylevels,
                    iondata.photoion_targetconfigs,
                    flog=flog,
                )

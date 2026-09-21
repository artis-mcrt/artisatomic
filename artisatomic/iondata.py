"""Read a single ion's levels, transitions, and photoionisation data from its source data set."""

import argparse
import contextlib
import itertools
import typing as t
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from artisatomic import groundstatesonlynist
from artisatomic import readadasdata
from artisatomic import readboyledata
from artisatomic import readdreamdata
from artisatomic import readfacdata
from artisatomic import readfloers25data
from artisatomic import readhillierdata
from artisatomic import readkuruczdata
from artisatomic import readlisbondata
from artisatomic import readmonsdata
from artisatomic import readtanakajpltdata
from artisatomic.base import elsymbols
from artisatomic.base import empty_comments
from artisatomic.base import ion_log_path
from artisatomic.base import IonLog
from artisatomic.base import leveltuples_to_pldataframe
from artisatomic.base import log_and_print
from artisatomic.base import log_comment
from artisatomic.base import PhixsData
from artisatomic.base import roman_numerals
from artisatomic.phixs import match_hydrogenic_phixs


@dataclass(slots=True)
class IonData:
    """Levels, transitions, and photoionisation data for a single ion, read from one of the source data sets.

    A dataclass rather than a NamedTuple because resolve_photoion_targetfractions() fills in the
    target fractions after the run reads the whole element.
    """

    ion_stage: int
    handler: str
    # The top ion has no upper ion, so it gets neither cross sections nor a phixs block. The read
    # pass records it here, so the read, resolve and write passes cannot disagree about it.
    is_top_ion: bool
    ionization_energy_ev: float
    dfenergylevels: pl.DataFrame
    dftransitions: pl.DataFrame
    upsilondict: dict[tuple[int, int], float]
    # None where a level has no photoionisation data (and None entirely if the reader read none).
    # readhillierdata.get_photoiontargetfractions() matches these names to the upper ion's
    # levels. It follows the CMFGEN conventions for a name. That reader therefore resolves
    # this field for every handler, and returns an empty result when the field is None.
    photoion_targetconfigs: list[list[tuple[str, float]] | None] | None
    photoionization_crosssections: npt.NDArray[np.float64]  # cross sections in Mb, indexed by level id
    photoionization_targetfractions: list[list[tuple[int, float]]]  # indexed by level id
    photoionization_thresholds_ev: npt.NDArray[np.float64]  # indexed by level id
    # The lines for the comment block of each output file (see base.IonLog). The read pass fills
    # them, and the resolve pass and the write pass add to them.
    comments: dict[str, list[str]] = field(default_factory=empty_comments)


@dataclass(frozen=True, slots=True)
class Handler:
    """One entry of the handler registry: the functions that read one data source.

    read_levels_and_transitions takes (atomic_number, ion_stage, flog), and args as well when
    reader_takes_args is set. It returns (ionization_energy_ev, energy_levels, transitions), with
    an upsilondict appended when returns_upsilondict is set. The writer counts the transitions of
    each level itself, from the final transition frame, so a reader returns no counts. The shapes
    differ by design, so the Callable stays untyped in its return.

    get_level_valence_n is the handler's own level-name parser. The hydrogenic photoionisation
    estimate uses it. None leaves the ion without cross sections, and match_hydrogenic_phixs()
    then writes a warning.

    read_coldata, when set, takes (atomic_number, ion_stage, dfenergylevels, args, flog) and
    returns the collision strengths of a data set that keeps them in their own file. They add to
    the upsilondict of the reader.

    read_phixs, when set, takes the same arguments and returns the PhixsData of the ion. Without
    it, the hydrogenic estimate is the only source of cross sections.

    description is the name of the data source of the levels and the transitions, with its
    reference where the repository holds one. read_ion_data() records it as the "source:" line of
    adata.txt and transitiondata.txt. A function that gives cross sections records the "source:"
    line of phixsdata_v2.txt itself, because the source can be different for each ion of a handler.
    """

    description: str
    read_levels_and_transitions: Callable[..., tuple[t.Any, ...]]
    get_level_valence_n: Callable[[str], int | None] | None = None
    returns_upsilondict: bool = False
    reader_takes_args: bool = False
    read_coldata: Callable[..., dict[tuple[int, int], float]] | None = None
    read_phixs: Callable[..., PhixsData] | None = None


handlers: dict[str, Handler] = {
    "boyle": Handler(
        readboyledata.description,
        readboyledata.read_levels_and_transitions,
    ),
    "kurucz": Handler(
        readkuruczdata.description,
        readkuruczdata.read_levels_and_transitions,
        readkuruczdata.get_level_valence_n,
    ),
    "dream": Handler(
        readdreamdata.description,
        readdreamdata.read_levels_and_transitions,
    ),
    "lisbon": Handler(readlisbondata.description, readlisbondata.read_levels_and_transitions),
    "floers25calibwithforbidden": Handler(
        readfloers25data.description + " (calibrated, with the forbidden lines)",
        partial(readfloers25data.read_levels_and_transitions, calibrated=True, withforbidden=True),
        readfloers25data.get_level_valence_n,
    ),
    "floers25calib": Handler(
        readfloers25data.description + " (calibrated)",
        partial(readfloers25data.read_levels_and_transitions, calibrated=True),
        readfloers25data.get_level_valence_n,
    ),
    "floers25uncalib": Handler(
        readfloers25data.description + " (uncalibrated)",
        partial(readfloers25data.read_levels_and_transitions, calibrated=False),
        readfloers25data.get_level_valence_n,
    ),
    "fac": Handler(
        readfacdata.description,
        readfacdata.read_levels_and_transitions,
        readfacdata.get_level_valence_n,
    ),
    "mons": Handler(
        readmonsdata.description,
        readmonsdata.read_levels_and_transitions,
    ),
    "tanakajplt": Handler(
        readtanakajpltdata.description,
        readtanakajpltdata.read_levels_and_transitions,
        readtanakajpltdata.get_level_valence_n,
    ),
    "gsnist": Handler(
        groundstatesonlynist.description,
        groundstatesonlynist.read_ground_levels,
    ),
    # The adf04 files tabulate the collision strengths at several temperatures, and
    # -electrontemperature picks one, so the reader takes args.
    # Only the QUB Co III data has cross sections. An ion with none gets the hydrogenic estimate.
    "adas": Handler(
        readadasdata.description,
        readadasdata.read_adas_levels_and_transitions,
        readadasdata.get_level_valence_n,
        returns_upsilondict=True,
        reader_takes_args=True,
        read_phixs=readadasdata.read_photoionizations,
    ),
    # levels, collision strengths and cross sections
    "cmfgen": Handler(
        readhillierdata.description,
        readhillierdata.read_levels_and_transitions,
        readhillierdata.get_level_valence_n,
        read_coldata=readhillierdata.read_coldata,
        read_phixs=readhillierdata.read_phixs_tables,
    ),
    # CMFGEN levels, transitions and collision strengths, with the QUB cross sections for Co II.
    # The QUB Co II tables are for the CMFGEN levels of Co II.
    "cmfgen_qubphixs": Handler(
        readhillierdata.description,
        readhillierdata.read_levels_and_transitions,
        readhillierdata.get_level_valence_n,
        read_coldata=readhillierdata.read_coldata,
        read_phixs=readadasdata.read_cmfgen_qubphixs_photoionizations,
    ),
}

# every handler name that read_ion_data() dispatches. parse_ion_handlers() checks a JSON file
# against this before the run writes any output file.
known_handlers: frozenset[str] = frozenset(handlers)


def read_ion_data(
    atomic_number: int, ion_stage_entry: tuple[int, str], is_top_ion: bool, args: argparse.Namespace
) -> IonData:
    """Read a single ion's data from its source data set.

    Every ion names the handler that reads it; there is no default per element.
    """
    ion_stage, handler = ion_stage_entry
    handlerspec = handlers.get(handler)
    if handlerspec is None:
        msg = f"Unknown handler: {handler}"
        raise ValueError(msg)

    upsilondict: dict[tuple[int, int], float] = {}
    photoion_targetconfigs: list[list[tuple[str, float]] | None] | None = None
    # empty until a handler below reads photoionisation data (and empty for the top ion)
    photoionization_crosssections: npt.NDArray[np.float64] = np.empty((0, args.nphixspoints))  # in Mb
    photoionization_targetfractions: list[list[tuple[int, float]]] = []
    photoionization_thresholds_ev: npt.NDArray[np.float64] = np.empty(0)

    logfilepath = ion_log_path(Path(args.output_folder, args.output_folder_logs), atomic_number, ion_stage)
    with logfilepath.open("w", encoding="utf-8") as logstream:
        flog = IonLog(logstream)
        log_and_print(
            flog,
            f"\n===========> Z={atomic_number} {elsymbols[atomic_number]} {roman_numerals[ion_stage]} input:",
        )
        log_and_print(flog, f"Source handler: {handler}")
        log_comment(flog, ("adata", "transitiondata"), f"source: {handlerspec.description}")
        result = (
            handlerspec.read_levels_and_transitions(atomic_number, ion_stage, flog, args)
            if handlerspec.reader_takes_args
            else handlerspec.read_levels_and_transitions(atomic_number, ion_stage, flog)
        )
        if handlerspec.returns_upsilondict:
            (ionization_energy_ev, energy_levels, transitions, upsilondict) = result
        else:
            (ionization_energy_ev, energy_levels, transitions) = result

        dfenergylevels = leveltuples_to_pldataframe(energy_levels)

        if handlerspec.read_coldata is not None:
            upsilondict.update(handlerspec.read_coldata(atomic_number, ion_stage, dfenergylevels, args, flog))

        # the top ion has no upper ion to photoionise to, so it gets no cross sections
        if not is_top_ion and not args.nophixs and handlerspec.read_phixs is not None:
            phixs = handlerspec.read_phixs(atomic_number, ion_stage, dfenergylevels, args, flog)
            photoionization_crosssections = phixs.crosssections
            photoionization_thresholds_ev = phixs.thresholds_ev
            photoion_targetconfigs = phixs.targetconfigs
            photoionization_targetfractions = phixs.targetfractions or []

        # The len() == 0 test limits the estimate to the ions for which the handler gave nothing.
        # An ion with even one cross section table keeps its data, so the estimate never replaces
        # measured data. The top ion gets no estimate because it has no upper ion to photoionise to.
        if (
            not is_top_ion
            and not args.nophixs
            and len(photoionization_crosssections) == 0
            and args.nlevels_hydrogenic_for_unknown_phixs > 0
        ):
            get_level_valence_n = handlerspec.get_level_valence_n
            (
                photoionization_crosssections,
                photoionization_targetfractions,
                photoionization_thresholds_ev,
            ) = match_hydrogenic_phixs(
                atomic_number, dfenergylevels, ionization_energy_ev, handler, get_level_valence_n, args, flog
            )

    dftransitions = transitions if isinstance(transitions, pl.DataFrame) else pl.DataFrame(transitions)

    return IonData(
        ion_stage=ion_stage,
        handler=handler,
        is_top_ion=is_top_ion,
        ionization_energy_ev=ionization_energy_ev,
        dfenergylevels=dfenergylevels,
        dftransitions=dftransitions,
        upsilondict=upsilondict,
        photoion_targetconfigs=photoion_targetconfigs,
        photoionization_crosssections=photoionization_crosssections,
        photoionization_targetfractions=photoionization_targetfractions,
        photoionization_thresholds_ev=photoionization_thresholds_ev,
        comments=flog.comments,
    )


def resolve_photoion_targetfractions(
    iondatalist: list[IonData], atomic_number: int | None = None, log_folder: str | Path | None = None
) -> None:
    """Fill in the photoionisation target fractions of each ion whose reader supplied none.

    An ion's targets are levels of the next ion up. This function can therefore resolve the
    fractions only after the run has read the whole element. It does not change the top ion (it
    has no upper ion to photoionise to), or an ion whose reader already gave per-level fractions.

    Call this before write_output_files() unless the user switched cross sections off entirely.
    The writer needs the fractions and does not resolve them itself. Give atomic_number and
    log_folder to append the resolve messages to each ion's log file.
    """
    # a half-given pair is always a caller bug, so the function raises
    if (atomic_number is None) != (log_folder is None):
        msg = "give both atomic_number and log_folder, or neither"
        raise ValueError(msg)

    if not iondatalist:
        return

    # The ions must be one element's, in stage order from lowest to highest. The loop resolves each ion against
    # the next entry as its upper ion. A top ion anywhere but last means the list breaks that
    # rule, and the target levels would belong to the wrong ion.
    if not iondatalist[-1].is_top_ion or any(iondata.is_top_ion for iondata in iondatalist[:-1]):
        msg = (
            "iondatalist must hold the ions of one element in ascending ion stage order."
            " Only the last ion can be the top ion."
        )
        raise ValueError(msg)

    for iondata, upperiondata in itertools.pairwise(iondatalist):
        if not iondata.photoionization_targetfractions:
            # The read pass closed the per-ion log. Reopen it in append mode. The IonLog below adds
            # to the comment lists of the read pass.
            logcontext = (
                ion_log_path(log_folder, atomic_number, iondata.ion_stage).open("a", encoding="utf-8")
                if log_folder is not None and atomic_number is not None
                else contextlib.nullcontext()
            )
            with logcontext as logstream:
                iondata.photoionization_targetfractions = readhillierdata.get_photoiontargetfractions(
                    iondata.dfenergylevels,
                    upperiondata.dfenergylevels,
                    iondata.photoion_targetconfigs,
                    flog=None if logstream is None else IonLog(logstream, iondata.comments),
                )

"""Select which ions to process and which data-source handler reads each one.

An ion_handlers list holds (atomic_number, ions) pairs, where each ion is an (ion_stage,
handler_name) tuple. Every ion names its handler: which source an ion is read from is never
inferred from its atomic number.
"""

import json
import operator
import typing as t
from pathlib import Path

from artisatomic import readfloers25data
from artisatomic import readhillierdata
from artisatomic import readqubdata
from artisatomic import readtanakajpltdata


def get_ion_handlers() -> list[tuple[int, list[tuple[int, str]]]]:
    """Get the ions to process and the handler to read each one with.

    Read from artisatomicionhandlers.json when that file exists, so a run can be repeated
    exactly; otherwise built from the hard-coded selection below plus whatever the readers'
    extend_ion_list() functions find data for.
    """
    inputhandlersfile = Path("artisatomicionhandlers.json")

    if inputhandlersfile.exists():
        print(f"Reading {inputhandlersfile}")
        with inputhandlersfile.open(encoding="utf-8") as f:
            return sort_ion_handlers(parse_ion_handlers(json.load(f)))

    ion_handlers: list[tuple[int, list[tuple[int, str]]]] = [
        (38, [(1, "kurucz"), (2, "kurucz"), (3, "kurucz")]),
        (39, [(1, "kurucz"), (2, "kurucz")]),
        (40, [(1, "kurucz"), (2, "kurucz"), (3, "kurucz")]),
    ]

    # include everything we have data for.
    # The first to associate with an ion will be the handler used, so the order of these calls matters.
    # readdreamdata, readfacdata and groundstatesonlynist also offer extend_ion_list(), and can be
    # added to this sequence to pull in everything they have data for.
    ion_handlers = readqubdata.extend_ion_list(ion_handlers)
    ion_handlers = readhillierdata.extend_ion_list(ion_handlers, maxionstage=5, include_hydrogen=True)
    ion_handlers = readfloers25data.extend_ion_list(ion_handlers, calibrated=True)
    ion_handlers = readtanakajpltdata.extend_ion_list(ion_handlers, maxionstage=5)

    return sort_ion_handlers(ion_handlers)


def parse_ion_handlers(loaded: t.Any) -> list[tuple[int, list[tuple[int, str]]]]:
    """Convert the JSON form of an ion_handlers list into tuples, rejecting entries with no handler.

    json.load() gives nested lists, and a hand-written file can carry a bare ion stage from when
    the handler was optional. Both are caught here rather than several frames later, where a bare
    stage would surface only as an unpacking TypeError naming neither the element nor the file.
    """
    ion_handlers: list[tuple[int, list[tuple[int, str]]]] = []
    for atomic_number, listions in loaded:
        ions: list[tuple[int, str]] = []
        for entry in listions:
            if isinstance(entry, int):
                msg = (
                    f"Z={atomic_number} ion stage {entry} in artisatomicionhandlers.json names no handler."
                    " Every ion must be given as [ion_stage, handler]."
                )
                raise TypeError(msg)
            ion_stage, handler = entry
            ions.append((int(ion_stage), str(handler)))
        ion_handlers.append((int(atomic_number), ions))

    return ion_handlers


def sort_ion_handlers(
    ion_handlers: list[tuple[int, list[tuple[int, str]]]],
) -> list[tuple[int, list[tuple[int, str]]]]:
    """Sort by atomic number, and each element's ions by ion stage.

    process_files() relies on ascending ion stages to identify the top ion and to find each ion's
    photoionisation target, so normalise the order here, before the handler list is written to
    artisatomicionhandlers.json and passed to write_compositionfile().
    """
    return sorted(
        ((atomic_number, sorted(listions, key=operator.itemgetter(0))) for atomic_number, listions in ion_handlers),
        key=operator.itemgetter(0),
    )


def drop_handlers(list_ions: list[tuple[int, str]]) -> list[int]:
    """Replace [(ion_stage1, 'handler1'), (ion_stage2, 'handler2')] with [ion_stage1, ion_stage2]."""
    return [ion_stage for ion_stage, _handler in list_ions]


def add_handler_if_not_set(
    ion_handlers: list[tuple[int, list[tuple[int, str]]]],
    atomic_number: int | str,
    ion_stage: int | str,
    handler: str,
) -> list[tuple[int, list[tuple[int, str]]]]:
    """Return a new ion_handlers list with (ion_stage, handler) added unless the ion is already present.

    The input list is not modified, so the return value must be used.
    """
    # readers derive these from pandas/numpy data, and json.dump() in main() cannot serialise
    # numpy integers, so normalise here rather than in each caller
    atomic_number = int(atomic_number)
    ion_stage = int(ion_stage)

    ion_handlers_out: list[tuple[int, list[tuple[int, str]]]] = []
    found_element = False
    for tmp_atomic_number, list_ions_handlers in ion_handlers:
        list_ions_handlers_out: list[tuple[int, str]] = list(list_ions_handlers)
        if tmp_atomic_number == atomic_number:
            found_element = True
            if ion_stage not in [x[0] for x in list_ions_handlers_out]:
                # add an ion that is not present in the element's list
                list_ions_handlers_out.append((ion_stage, handler))
        ion_handlers_out.append((tmp_atomic_number, list_ions_handlers_out))

    if not found_element:
        ion_handlers_out.append((atomic_number, [(ion_stage, handler)]))

    return sort_ion_handlers(ion_handlers_out)

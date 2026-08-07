"""Select which ions to process and which data-source handler reads each one.

An ion_handlers list holds (atomic_number, ions) pairs, where each ion is either a bare ion stage
(the default handler applies) or an (ion_stage, handler_name) tuple.
"""

import json
import operator
from pathlib import Path

from artisatomic import readfloers25data
from artisatomic import readhillierdata
from artisatomic import readqubdata
from artisatomic import readtanakajpltdata

USE_QUB_COBALT = False


def get_ion_handlers() -> list[tuple[int, list[int | tuple[int, str]]]]:
    """Get the ions to process and the handler to read each one with.

    Read from artisatomicionhandlers.json when that file exists, so a run can be repeated
    exactly; otherwise built from the hard-coded selection below plus whatever the readers'
    extend_ion_list() functions find data for.
    """
    inputhandlersfile = Path("artisatomicionhandlers.json")

    if inputhandlersfile.exists():
        print(f"Reading {inputhandlersfile}")
        with inputhandlersfile.open(encoding="utf-8") as f:
            return sort_ion_handlers(json.load(f))

    ion_handlers: list[tuple[int, list[int | tuple[int, str]]]] = []

    ion_handlers = [
        #     (2, [(3, "boyle")]),
        (38, [(1, "kurucz"), (2, "kurucz"), (3, "kurucz")]),
        (39, [(1, "kurucz"), (2, "kurucz")]),
        (40, [(1, "kurucz"), (2, "kurucz"), (3, "kurucz")]),
    ]

    # include everything we have data for
    ion_handlers = readqubdata.extend_ion_list(ion_handlers)
    ion_handlers = readhillierdata.extend_ion_list(ion_handlers, maxionstage=5, include_hydrogen=True)
    # ion_handlers = readdreamdata.extend_ion_list(ion_handlers)
    # ion_handlers = readfacdata.extend_ion_list(ion_handlers)
    ion_handlers = readfloers25data.extend_ion_list(ion_handlers, calibrated=True)
    ion_handlers = readtanakajpltdata.extend_ion_list(ion_handlers, maxionstage=5)
    # ion_handlers = groundstatesonlynist.extend_ion_list(ion_handlers)

    return sort_ion_handlers(ion_handlers)


def get_ion_stage(entry: int | tuple[int, str]) -> int:
    """Ion stage of an ion_handlers entry, which is either a bare ion stage or (ion_stage, handler)."""
    return entry if isinstance(entry, int) else entry[0]


def sort_ion_handlers(
    ion_handlers: list[tuple[int, list[int | tuple[int, str]]]],
) -> list[tuple[int, list[int | tuple[int, str]]]]:
    """Sort by atomic number, and each element's ions by ion stage.

    process_files() relies on ascending ion stages to identify the top ion and to find each ion's
    photoionisation target, so normalise the order here, before the handler list is written to
    artisatomicionhandlers.json and passed to write_compositionfile().
    """
    return sorted(
        ((atomic_number, sorted(listions, key=get_ion_stage)) for atomic_number, listions in ion_handlers),
        key=operator.itemgetter(0),
    )


def drop_handlers(list_ions: list[int | tuple[int, str]]) -> list[int]:
    """Replace [(ion_stage, 'handler1'), (ion_stage2, 'handler2'), ion_stage3] with [ion_stage1, ion_stage2, ion_stage3]."""
    return [get_ion_stage(ion_stage) for ion_stage in list_ions]


def add_handler_if_not_set(
    ion_handlers: list[tuple[int, list[int | tuple[int, str]]]],
    atomic_number: int | str,
    ion_stage: int | str,
    handler: str,
) -> list[tuple[int, list[int | tuple[int, str]]]]:
    """Return a new ion_handlers list with (ion_stage, handler) added unless the ion is already present.

    The input list is not modified, so the return value must be used.
    """
    # readers derive these from pandas/numpy data, and json.dump() in main() cannot serialise
    # numpy integers, so normalise here rather than in each caller
    atomic_number = int(atomic_number)
    ion_stage = int(ion_stage)

    ion_handlers_out: list[tuple[int, list[int | tuple[int, str]]]] = []
    found_element = False
    for tmp_atomic_number, list_ions_handlers in ion_handlers:
        list_ions_handlers_out: list[int | tuple[int, str]] = list(list_ions_handlers)
        if tmp_atomic_number == atomic_number:
            found_element = True
            if ion_stage not in [get_ion_stage(x) for x in list_ions_handlers_out]:
                # add an ion that is not present in the element's list
                list_ions_handlers_out.append((ion_stage, handler))
        ion_handlers_out.append((tmp_atomic_number, list_ions_handlers_out))

    if not found_element:
        new_element_ions: list[int | tuple[int, str]] = [(ion_stage, handler)]
        ion_handlers_out.append((atomic_number, new_element_ions))

    return sort_ion_handlers(ion_handlers_out)


def get_default_handler(atomic_number: int, ion_stage: int) -> str:
    """Get the data source to use for an ion when the handler list does not name one."""
    if atomic_number == 2 and ion_stage == 3:
        return "boyle"
    if USE_QUB_COBALT and atomic_number == 27:
        return "qub_cobalt"
    if atomic_number <= 28 or atomic_number == 56:  # Hillier data only
        return "cmfgen"
    # a QUB calculation is preferred over the Kurucz line lists, and over DREAM for W, Pt and Au
    if (atomic_number, ion_stage) in readqubdata.default_handler_ions:
        return "qub_data"
    if atomic_number >= 57:  # DREAM database of Z > 57
        return "dream"
    return "kurucz"

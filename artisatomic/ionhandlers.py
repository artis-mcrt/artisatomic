"""Select which ions to process and which data-source handler reads each one.

An ion_handlers list holds (atomic_number, ions) pairs, where each ion is an (ion_stage,
handler_name) tuple. Every ion names its handler. The code never infers the source of an ion
from its atomic number.
"""

import json
import typing as t
from pathlib import Path

from artisatomic import readfloers25data
from artisatomic import readhillierdata
from artisatomic import readqubdata
from artisatomic import readtanakajpltdata
from artisatomic.base import sort_ion_handlers
from artisatomic.iondata import known_handlers


def get_ion_handlers() -> list[tuple[int, list[tuple[int, str]]]]:
    """Get the ions to process and the handler to read each one with.

    The function reads artisatomicionhandlers.json when that file exists, so the user can repeat
    a run exactly. Otherwise it builds the list from the hard-coded selection below plus every
    ion for which the readers' extend_ion_list() functions find data.
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

    # Include every ion that has data.
    # The first call that adds an ion sets its handler, so the order of these calls matters.
    # readdreamdata, readfacdata, readmonsdata and groundstatesonlynist also offer extend_ion_list(). Add them to
    # this sequence to include every ion that they have data for.
    ion_handlers = readqubdata.extend_ion_list(ion_handlers)
    ion_handlers = readhillierdata.extend_ion_list(ion_handlers, maxionstage=5, include_hydrogen=True)
    ion_handlers = readfloers25data.extend_ion_list(ion_handlers, calibrated=True)
    ion_handlers = readtanakajpltdata.extend_ion_list(ion_handlers, maxionstage=5)

    return sort_ion_handlers(ion_handlers)


# Old handler names and their new names. A file from before the rename still names the old one,
# so this map keeps those files readable.
renamed_handlers = {"qub_data": "qub"}


def parse_ion_handlers(loaded: t.Any) -> list[tuple[int, list[tuple[int, str]]]]:
    """Convert the JSON form of an ion_handlers list into tuples, and reject a malformed entry.

    json.load() gives nested lists. A hand-written file can carry a bare ion stage, a misspelt
    handler name, or an entry of the wrong shape. A bare ion stage dates from when the handler was
    optional. This function rejects all three before the run writes any output file. The function
    read_ion_data() would reject the name only after the run wrote compositiondata.txt and the
    earlier elements. An unpacking error several frames later would name neither the element nor
    the file.
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
            try:
                ion_stage, handler = entry
            except (TypeError, ValueError):
                msg = (
                    f"Z={atomic_number} entry {entry!r} in artisatomicionhandlers.json is not an"
                    " [ion_stage, handler] pair."
                )
                raise TypeError(msg) from None
            handlername = renamed_handlers.get(str(handler), str(handler))
            if handlername not in known_handlers:
                msg = (
                    f"Z={atomic_number} ion stage {ion_stage} in artisatomicionhandlers.json names the unknown"
                    f" handler {handlername!r}. The handlers are: {', '.join(sorted(known_handlers))}."
                )
                raise ValueError(msg)
            ions.append((int(ion_stage), handlername))
        ion_handlers.append((int(atomic_number), ions))

    return ion_handlers

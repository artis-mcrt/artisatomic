"""Select which ions to process and which data-source handler reads each one.

An ion_handlers list holds (atomic_number, ions) pairs, where each ion is an (ion_stage,
handler_name) tuple. Every ion names its handler. The code never infers the source of an ion
from its atomic number.
"""

import json
import typing as t
from pathlib import Path

from artisatomic import readadasdata
from artisatomic import readfloers25data
from artisatomic import readhillierdata
from artisatomic import readkuruczdata
from artisatomic import readtanakajpltdata
from artisatomic.base import add_handlers_if_not_set
from artisatomic.base import elsymbols
from artisatomic.base import sort_ion_handlers
from artisatomic.base import TESTMODE
from artisatomic.iondata import known_handlers

# The file that holds a full ion selection, in the working directory. main() rejects an ion limit
# that comes with this file, so both modules name it here.
inputhandlersfile = Path("artisatomicionhandlers.json")


def keep_ions_within_limits(
    ion_handlers: list[tuple[int, list[tuple[int, str]]]],
    minionstage: int | None,
    maxionstage: int | None,
    maxatomicnumber: int | None,
) -> list[tuple[int, list[tuple[int, str]]]]:
    """Drop each ion outside the limits, and each element that then has no ion. None applies no limit."""
    kept = [
        (
            atomic_number,
            [
                ion
                for ion in listions
                if (minionstage is None or ion[0] >= minionstage) and (maxionstage is None or ion[0] <= maxionstage)
            ],
        )
        for atomic_number, listions in ion_handlers
        if maxatomicnumber is None or atomic_number <= maxatomicnumber
    ]
    return [(atomic_number, listions) for atomic_number, listions in kept if listions]


def get_ion_handlers(
    minionstage: int | None, maxionstage: int | None, maxatomicnumber: int | None
) -> list[tuple[int, list[tuple[int, str]]]]:
    """Get the ions to process and the handler to read each one with.

    The function reads artisatomicionhandlers.json when that file exists, so the user can repeat
    a run exactly. That file holds the ion stages and the atomic numbers already, so no limit
    applies to it. Otherwise the function builds the built-in selection: the ions below plus every
    ion for which the readers' extend_ion_list() functions find data.

    minionstage, maxionstage and maxatomicnumber limit that built-in selection. A value of None
    applies no limit. The caller states all three, because build_parser() holds their default values.
    """
    if inputhandlersfile.exists():
        print(f"Reading {inputhandlersfile}")
        with inputhandlersfile.open(encoding="utf-8") as f:
            return sort_ion_handlers(parse_ion_handlers(json.load(f)))

    # Include every ion that has data.
    # The first call that adds an ion sets its handler, so the order of these calls matters.
    # readdreamdata, readfacdata, readmonsdata and groundstatesonlynist also offer extend_ion_list(). Add them to
    # this sequence to offer their ions too. A reader whose files start above the default -maxionstage needs a
    # higher value.
    #
    # The ADAS ions come before the Kurucz ions. The ADAS Sr I file gives collision strengths,
    # and the Kurucz line list of Sr I gives none.
    ion_handlers = readadasdata.extend_ion_list([])
    kuruczions = [
        (atomic_number, ion_stage)
        for atomic_number, ion_stages in ((38, (1, 2, 3)), (39, (1, 2)), (40, (1, 2, 3)))
        for ion_stage in ion_stages
    ]
    ion_handlers = add_handlers_if_not_set(ion_handlers, kuruczions, "kurucz")
    ion_handlers = readhillierdata.extend_ion_list(ion_handlers)
    ion_handlers = readfloers25data.extend_ion_list(ion_handlers, calibrated=True)
    ion_handlers = readtanakajpltdata.extend_ion_list(ion_handlers)

    ion_handlers = keep_ions_within_limits(ion_handlers, minionstage, maxionstage, maxatomicnumber)

    # The read of the lighter elements can take an hour, so a missing file stops the run here. Only the
    # ions that the Kurucz handler keeps count: ADAS can take Sr I, and the limits can exclude an ion.
    # The test mode reads the Kurucz sample, which holds the files of Sr I-II and Y I-II only.
    if not TESTMODE:
        for atomic_number, listions in ion_handlers:
            for ion_stage, handler in listions:
                if handler != "kurucz":
                    continue
                try:
                    readkuruczdata.find_gfall(atomic_number, ion_stage - 1)
                except FileNotFoundError as exc:
                    msg = (
                        f"The built-in selection takes Z={atomic_number} ion_stage={ion_stage} from the Kurucz line"
                        " lists, but the file is missing. Download the Kurucz files (atomic-data-kurucz/url.txt), or"
                        f" give an ion handlers file.\n{exc}"
                    )
                    raise FileNotFoundError(msg) from exc

    return sort_ion_handlers(ion_handlers)


# Old handler names and their new names. A file from before the rename still names the old one,
# so this map keeps those files readable.
# Each old name gives the new name for an ion stage, and the key None gives the name for each
# other ion stage. The old cobalt handler read QUB levels for stages 3 and 4, CMFGEN levels with
# QUB cross sections for stage 2, and CMFGEN data for each other stage.
renamed_handlers: dict[str, dict[int | None, str]] = {
    "qub_data": {None: "adas"},
    "qub": {None: "adas"},
    "qub_cobalt": {2: "cmfgen_qubphixs", 3: "adas", 4: "adas", None: "cmfgen"},
}


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
    for element in loaded:
        try:
            atomic_number, listions = element
        except (TypeError, ValueError):
            msg = f"The entry {element!r} in artisatomicionhandlers.json is not an [atomic_number, [ions]] pair."
            raise TypeError(msg) from None
        # a JSON true is a Python bool, which is an int, and int() cuts 26.9 to 26
        if (
            not isinstance(atomic_number, int)
            or isinstance(atomic_number, bool)
            or not 1 <= atomic_number < len(elsymbols)
        ):
            msg = f"The atomic number {atomic_number!r} in artisatomicionhandlers.json is not a whole number from 1 to {len(elsymbols) - 1}."
            raise ValueError(msg)
        ions: list[tuple[int, str]] = []
        for entry in listions:
            if isinstance(entry, int):
                msg = (
                    f"Z={atomic_number} ion stage {entry} in artisatomicionhandlers.json names no handler."
                    " Write every ion as [ion_stage, handler]."
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
            if not isinstance(ion_stage, int) or isinstance(ion_stage, bool) or not 1 <= ion_stage <= atomic_number + 1:
                msg = (
                    f"Z={atomic_number} ion stage {ion_stage!r} in artisatomicionhandlers.json is not a whole number"
                    f" from 1 to {atomic_number + 1}."
                )
                raise ValueError(msg)
            newnames = renamed_handlers.get(str(handler), {None: str(handler)})
            handlername = newnames.get(ion_stage, newnames[None])
            if handlername not in known_handlers:
                msg = (
                    f"Z={atomic_number} ion stage {ion_stage} in artisatomicionhandlers.json names the unknown"
                    f" handler {handlername!r}. The handlers are: {', '.join(sorted(known_handlers))}."
                )
                raise ValueError(msg)
            ions.append((ion_stage, handlername))
        ion_handlers.append((atomic_number, ions))

    return ion_handlers

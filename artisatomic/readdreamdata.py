"""Read levels and transitions from the DREAM database of lanthanides and actinides."""

# the pandas HDFStore format holds a block of Python objects as a pickle, so this module must
# unpickle it. The file comes from the DREAM parser that the user runs, as every other data
# file of this package does
import pickle  # ruff: ignore[suspicious-pickle-import]
from pathlib import Path

import h5py
import numpy as np
import polars as pl

from artisatomic.base import add_handler_if_not_set
from artisatomic.base import EnergyLevel
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import log_and_print
from artisatomic.base import PYDIR
from artisatomic.base import Transition

# the h5 file comes from Andreas Floers's DREAM parser
dreamdatapath = PYDIR / ".." / "atomic-data-dream" / "DREAM_atomic_data_20241106-1325.h5"
dreamdata: pl.DataFrame | None = None


def read_pandas_hdfstore(path: Path) -> pl.DataFrame:
    """Read one frame from a pandas HDFStore of the "fixed" format.

    The DREAM parser writes its line list with the to_hdf() method of pandas. That format holds
    the frame as blocks. Each block has a data set of the values and a data set of the column
    names. This function reads them with h5py, which keeps pandas out of the dependencies.

    The function matches the values to the columns by name, so the order of the blocks does not
    matter. pandas pickles a block of Python objects and the levels of the index, so this
    function unpickles those. A file of any other format raises a ValueError, because the block
    layout is the only layout that this function reads.
    """

    def unpickle(dataset) -> np.ndarray:
        # pandas stores a column of Python objects as one pickle, in a pytables VLArray
        return pickle.loads(dataset[0].tobytes())  # ruff: ignore[suspicious-pickle-usage]

    def as_text(value) -> str:
        # h5py gives a byte string for a written attribute and a str for one it decoded itself
        return value.decode("utf-8") if isinstance(value, bytes | np.bytes_) else str(value)

    def names_of(dataset) -> list[str]:
        return [as_text(name) for name in dataset[:]]

    with h5py.File(path, "r") as h5file:
        groupnames = list(h5file.keys())
        # not an assert: a file that holds no frame, or more than one, must fail with its name
        if len(groupnames) != 1:
            msg = f"{path} holds {len(groupnames)} groups. The DREAM reader expects exactly one frame."
            raise ValueError(msg)
        group = h5file[groupnames[0]]

        pandas_type = as_text(group.attrs.get("pandas_type", b""))
        if pandas_type != "frame":
            msg = (
                f"{path} has pandas_type {pandas_type!r}, not 'frame'."
                " Write it with DataFrame.to_hdf(..., format='fixed')."
            )
            raise ValueError(msg)

        columns: dict[str, np.ndarray] = {}

        # the levels of the index come first, so the frame starts with Z and the ion charge
        for level in range(int(group.attrs["axis1_nlevels"])):
            leveldata = group[f"axis1_level{level}"]
            # pandas pickles a level of Python objects only. It writes a level of numbers plain,
            # as it does a block of numbers
            levelvalues = unpickle(leveldata) if leveldata.dtype == object else leveldata[:]
            codes = group[f"axis1_label{level}"][:]
            # not an assert: pandas writes -1 for a row that the level does not hold, and numpy
            # reads -1 as the last value. Such a row would join the wrong ion without a message
            if codes.min() < 0:
                msg = f"{path} index level {level} has a row with no value. pandas writes -1 for such a row."
                raise ValueError(msg)
            columns[as_text(leveldata.attrs["name"])] = levelvalues[codes]

        blockvalues: dict[str, np.ndarray] = {}
        for block in range(int(group.attrs["nblocks"])):
            values = group[f"block{block}_values"]
            names = names_of(group[f"block{block}_items"])
            array = unpickle(values) if values.dtype == object else values[:]
            # each block holds one column for each of its names, so a mismatch means that the
            # file does not have the layout that this function reads
            if array.shape[1] != len(names):
                msg = f"{path} block {block} has {array.shape[1]} columns for {len(names)} names"
                raise ValueError(msg)
            for index, name in enumerate(names):
                blockvalues[name] = array[:, index]

        # axis0 holds the columns in the order of the frame, which the blocks do not keep
        columns |= {name: blockvalues[name] for name in names_of(group["axis0"])}

    return pl.DataFrame({name: series_from_array(values) for name, values in columns.items()})


def series_from_array(values: np.ndarray) -> pl.Series:
    """Make a polars series from one column of a pandas HDFStore.

    pandas gives a column of Python objects as an array of dtype object. polars reads such an
    array as the Object type, which its expressions cannot filter or group. A list of the same
    values lets polars find the type, which gives Int64 for the ion charge and the level indices.

    A column whose rows hold more than one type keeps the Object type. The DREAM line list has
    one such column, CF, which holds a float in most rows and a string in the others.
    """
    if values.dtype != object:
        return pl.Series(values)
    try:
        return pl.Series(values.tolist())
    except (TypeError, pl.exceptions.PolarsError):
        # polars raises TypeError for a mixed column today. The except covers its own error class
        # too, so a later polars that raises SchemaError still reaches the Object fallback
        return pl.Series(values, dtype=pl.Object)


def init_dreamdata():
    """Load the DREAM line list into the module-level cache, once per process."""
    global dreamdata
    if dreamdata is not None:
        return
    # drop CF: no reader uses it, and its rows hold a float or a string, so it stays an Object
    # column that no polars expression can read
    dreamdata = (
        read_pandas_hdfstore(dreamdatapath)
        .drop("CF", strict=False)
        .with_columns(Lower_g=2 * pl.col("Lower_J") + 1, Upper_g=2 * pl.col("Upper_J") + 1)
    )


def extend_ion_list(
    ion_handlers, minionstage: int | None = None, maxionstage: int | None = None, maxatomicnumber: int | None = None
):
    """Add every ion in the DREAM line list to ion_handlers under the "dream" handler."""
    init_dreamdata()
    assert dreamdata is not None
    for atomic_number, charge in dreamdata.select("Z", "C").unique(maintain_order=True).iter_rows():
        ion_stage = charge + 1
        ion_handlers = add_handler_if_not_set(
            ion_handlers, atomic_number, ion_stage, "dream", minionstage, maxionstage, maxatomicnumber
        )

    return ion_handlers


def energytuplefromrow(row, prefix):
    """Build the lower or upper level of one DREAM line, selected by prefix ("Lower"/"Upper").

    DREAM levels have no spectroscopic names, so the name of each level holds the energy, parity
    and statistical weight that identify it. read_levels_data() deduplicates on the same three
    values.
    """
    energy, leveltype, g = row[prefix + "_Level"], row[prefix + "_Type"], row[prefix + "_g"]

    # not a default of 0 for any other text: the Laporte rule would get a parity the file did
    # not state
    if leveltype not in {"(o)", "(e)"}:
        msg = f"DREAM level type {leveltype!r} is not '(o)' or '(e)'"
        raise ValueError(msg)
    parity = 1 if leveltype == "(o)" else 0
    paritystr = "odd" if parity == 1 else "even"
    energyabovegsinpercm = float(energy)

    levelname = f"enpercm={energy},{paritystr},g={g}"
    return EnergyLevel(levelname=levelname, parity=parity, g=g, energyabovegsinpercm=energyabovegsinpercm)


def read_levels_data(dflines):
    """Recover the level list from a DREAM line list, which has no separate level table.

    Each line carries both of its levels inline, so the levels are the distinct lower and upper
    levels over all lines, sorted by energy.
    """
    # a set for the membership test, not `not in energy_levels`: that was a linear scan of the
    # list per candidate. The build of the level list then cost O(levels^2)
    seen: set[EnergyLevel] = set()
    energy_levels = []

    for prefix in ["Lower", "Upper"]:
        subset = [prefix + "_Type", prefix + "_Level", prefix + "_g"]
        # keep="first" pins the order that decides the ids of two levels of one energy
        for row in dflines.unique(subset=subset, keep="first", maintain_order=True).iter_rows(named=True):
            leveltuple = energytuplefromrow(row, prefix)
            if leveltuple not in seen:
                seen.add(leveltuple)
                energy_levels.append(leveltuple)

    energy_levels.sort(key=lambda x: x.energyabovegsinpercm)

    return energy_levels


def read_lines_data(dfiondata):
    """Convert DREAM lines to transitions referencing zero-based level ids."""
    transitions = []

    # numpy columns, not iterrows(): that built a Series for each of the 10^5 lines of an ion.
    # read_levels_data() sorted the levels by energy, and transitiondata.txt has the lower id
    # first. So this code swaps a pair that the file lists in the reverse order.
    lowerindices = np.minimum(dfiondata["Lower_index"].to_numpy(), dfiondata["Upper_index"].to_numpy())
    upperindices = np.maximum(dfiondata["Lower_index"].to_numpy(), dfiondata["Upper_index"].to_numpy())
    # g_upper is the g of the level the file calls upper, because gA is that level's product
    A_values = dfiondata["gA"].to_numpy() / dfiondata["Upper_g"].to_numpy()

    for lowerindex, upperindex, A in zip(lowerindices.tolist(), upperindices.tolist(), A_values.tolist(), strict=True):
        transitions.append(Transition(lowerlevel=lowerindex, upperlevel=upperindex, A=A))

    return transitions


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    """Read one ion from the DREAM database of Z >= 57."""
    init_dreamdata()
    assert dreamdata is not None
    charge = ion_stage - 1
    dfiondata = dreamdata.filter((pl.col("Z") == atomic_number) & (pl.col("C") == charge))
    # not an assert: an ion that the database does not hold would give an empty level list and an
    # ion with no lines in the output. The pandas reader that this replaced raised a KeyError
    if dfiondata.is_empty():
        msg = f"The DREAM database has no lines for Z={atomic_number} ion_stage {ion_stage}"
        raise ValueError(msg)
    print(f"Reading DREAM database for Z={atomic_number} ion_stage {ion_stage}")

    energy_levels = read_levels_data(dfiondata)

    # a dict, not energy_levels.index(): that scanned the level list once per level of every
    # line. The id resolution then cost O(lines x levels) for a database of Z >= 57 lanthanides
    levelid_of_leveltuple = {leveltuple: levelid for levelid, leveltuple in enumerate(energy_levels)}

    def get_level_index(row, prefix):
        """Return the zero-based level id of the row's level."""
        leveltuple = energytuplefromrow(row, prefix)
        levelid = levelid_of_leveltuple.get(leveltuple)
        if levelid is None:
            # not an assert: read_levels_data() built the level list from the same frame as the
            # lines. A miss means they disagree, so the code must not map it silently
            msg = f"DREAM line names a {prefix} level that is not in the level list: {leveltuple}"
            raise ValueError(msg)
        return levelid

    # a list over the rows, not a call for each column: the level id needs the whole row
    rows = list(dfiondata.iter_rows(named=True))
    dfiondata = dfiondata.with_columns(
        Lower_index=pl.Series([get_level_index(row, prefix="Lower") for row in rows], dtype=pl.Int64),
        Upper_index=pl.Series([get_level_index(row, prefix="Upper") for row in rows], dtype=pl.Int64),
    )

    transitions = read_lines_data(dfiondata)

    # DREAM has no ionisation energies, so take them from NIST as the other handlers do
    ionization_energy_in_ev = get_nist_ionization_energies_ev()[atomic_number, ion_stage]
    log_and_print(flog, f"ionisation energy: {ionization_energy_in_ev} eV")

    log_and_print(flog, f"Read {len(energy_levels):d} levels")

    return ionization_energy_in_ev, energy_levels, transitions

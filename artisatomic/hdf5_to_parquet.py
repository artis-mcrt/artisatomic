#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["h5py>=3.10.0", "numpy>=2.2.6", "polars>=2.0.0"]
# ///
"""Convert each table of an HDF5 file to a Parquet file.

The readers of artisatomic read Parquet files with polars, so artisatomic needs no HDF5 library.
Run this script once for each HDF5 file of a data set:

    artisatomic/hdf5_to_parquet.py atomic-data-helium-boyle/aoife.hdf5

The script writes the tables into a folder that has the name of the HDF5 file without its
extension. The path of a table in that folder is the path of its object in the HDF5 file. The
script does not change or delete the HDF5 file. It reads two layouts:

- A group with a pandas_type attribute holds a frame that DataFrame.to_hdf() of pandas wrote in
  the "fixed" format. The script rebuilds the frame without pandas.
- Each other data set becomes a table. The fields of a compound type become the columns.

The HDF5 attributes of an object go into the key-value metadata of its Parquet file.
"""

import argparse
import importlib

# the pandas "fixed" format holds a block of Python objects as a pickle, so this script must
# unpickle it. The user gives the file, so it is a trusted input
import pickle  # ruff: ignore[suspicious-pickle-import]
import sys
from collections.abc import Iterator
from collections.abc import Sequence
from itertools import starmap
from pathlib import Path

import numpy as np
import polars as pl

# h5py 3.16 has no wheel for Python 3.15, so the project environment does not hold it. An import
# statement would make the type checkers report a missing module
h5py = importlib.import_module("h5py")


def as_text(value) -> str:
    """Give an HDF5 attribute or name as a str.

    h5py gives a byte string for a written attribute and a str for one that it decoded itself.
    """
    return value.decode("utf-8") if isinstance(value, bytes | np.bytes_) else str(value)


def series_from_array(name: str, values: np.ndarray) -> pl.Series:
    """Make a polars series from one column, with a type that Parquet can hold.

    polars reads an array of dtype object as the Object type, which Parquet cannot hold. A list
    of the same values lets polars find the type, which gives Int64 for Python integers. A column
    whose rows hold more than one type becomes a String column. The DREAM line list has one such
    column, CF, which holds a float in most rows and a string in the others.
    """
    if values.dtype.kind == "S":
        return pl.Series(name, [as_text(value) for value in values.tolist()], dtype=pl.String)
    if values.dtype != object:
        return pl.Series(name, values)
    if all(isinstance(value, bytes) for value in values.tolist()):
        return pl.Series(name, [as_text(value) for value in values.tolist()], dtype=pl.String)
    try:
        return pl.Series(name, values.tolist())
    except (TypeError, pl.exceptions.PolarsError):
        # polars raises TypeError for a mixed column today. The except covers its own error class
        # too, so a later polars that raises SchemaError still reaches the String fallback
        return pl.Series(name, [None if value is None else str(value) for value in values.tolist()], dtype=pl.String)


def read_pandas_frame(group) -> pl.DataFrame:
    """Read one frame from a group that pandas wrote in the "fixed" format.

    That format holds the frame as blocks. Each block has a data set of the values and a data set
    of the column names. pandas pickles a block of Python objects and the levels of the index,
    so this function unpickles those. The function matches the values to the columns by name, so
    the order of the blocks does not matter. A group of any other pandas format raises a
    ValueError, because the block layout is the only layout that this function reads.
    """

    def unpickle(dataset) -> np.ndarray:
        # pandas stores a column of Python objects as one pickle, in a pytables VLArray
        return pickle.loads(dataset[0].tobytes())  # ruff: ignore[suspicious-pickle-usage]

    def names_of(dataset) -> list[str]:
        return [as_text(name) for name in dataset[:]]

    pandas_type = as_text(group.attrs.get("pandas_type", b""))
    if pandas_type != "frame":
        msg = (
            f"{group.file.filename}:{group.name} has pandas_type {pandas_type!r}, not 'frame'."
            " Write it with DataFrame.to_hdf(..., format='fixed')."
        )
        raise ValueError(msg)
    # a frame with one plain index has no axis1_nlevels. The script reads only the index of
    # several levels, because no data set of artisatomic has the other kind
    if "axis1_nlevels" not in group.attrs:
        msg = f"{group.file.filename}:{group.name} has no index of several levels. The script reads no other index."
        raise ValueError(msg)

    columns: dict[str, np.ndarray] = {}

    # the levels of the index come first, so the DREAM frame starts with Z and the ion charge
    for level in range(int(group.attrs["axis1_nlevels"])):
        leveldata = group[f"axis1_level{level}"]
        # pandas pickles a level of Python objects only. It writes a level of numbers plain, as it
        # does a block of numbers
        levelvalues = unpickle(leveldata) if leveldata.dtype == object else leveldata[:]
        codes = group[f"axis1_label{level}"][:]
        # not an assert: pandas writes -1 for a row that the level does not hold, and numpy reads
        # -1 as the last value. Such a row would get the wrong value without a message
        if codes.min() < 0:
            msg = f"{group.file.filename}:{group.name} index level {level} has a row with no value."
            raise ValueError(msg)
        columns[as_text(leveldata.attrs["name"])] = levelvalues[codes]

    blockvalues: dict[str, np.ndarray] = {}
    for block in range(int(group.attrs["nblocks"])):
        values = group[f"block{block}_values"]
        names = names_of(group[f"block{block}_items"])
        array = unpickle(values) if values.dtype == object else values[:]
        # each block holds one column for each of its names, so a mismatch means that the group
        # does not have the layout that this function reads
        if array.shape[1] != len(names):
            msg = (
                f"{group.file.filename}:{group.name} block {block} has {array.shape[1]} columns for {len(names)} names"
            )
            raise ValueError(msg)
        for index, name in enumerate(names):
            blockvalues[name] = array[:, index]

    # axis0 holds the columns in the order of the frame, which the blocks do not keep
    columns |= {name: blockvalues[name] for name in names_of(group["axis0"])}

    return pl.DataFrame(list(starmap(series_from_array, columns.items())))


def read_dataset(dataset) -> pl.DataFrame | None:
    """Read one data set as a table, or give None for a shape that is not a table.

    The fields of a compound type become the columns. A data set of one dimension gives one
    column, "value". A data set of two dimensions gives one column for each index of the second
    dimension, "column_0", "column_1", and so on.
    """
    values = dataset[()]
    if not isinstance(values, np.ndarray) or values.ndim not in {1, 2}:
        return None
    if values.dtype.names is not None:
        if values.ndim != 1:
            return None
        return pl.DataFrame([series_from_array(name, values[name]) for name in values.dtype.names])
    if values.ndim == 1:
        return pl.DataFrame([series_from_array("value", values)])
    return pl.DataFrame([series_from_array(f"column_{index}", values[:, index]) for index in range(values.shape[1])])


def metadata_of(h5object) -> dict[str, str]:
    """Give the HDF5 attributes of an object as text, for the key-value metadata of Parquet."""
    metadata = {}
    for key, value in h5object.attrs.items():
        # an attribute with no value, as pytables writes for the TITLE of each node
        if isinstance(value, h5py.Empty):
            continue
        metadata[key] = str(value.tolist()) if isinstance(value, np.ndarray) else as_text(value)
    return metadata


def tables_of(group) -> Iterator[tuple[str, pl.DataFrame | None, dict[str, str]]]:
    """Give the path, the table and the metadata of each object below group.

    The table is None for a data set that has no table shape.
    """
    for h5object in group.values():
        if isinstance(h5object, h5py.Group):
            if "pandas_type" in h5object.attrs:
                yield h5object.name, read_pandas_frame(h5object), metadata_of(h5object)
            else:
                yield from tables_of(h5object)
        else:
            yield h5object.name, read_dataset(h5object), metadata_of(h5object)


def convert_file(path: Path) -> list[Path]:
    """Write each table of one HDF5 file as a Parquet file, and give the paths of those files."""
    outputfolder = path.with_suffix("")
    outputpaths = []
    with h5py.File(path, "r") as h5file:
        for name, table, metadata in tables_of(h5file):
            if table is None:
                print(f"{path}:{name} has no table shape. The script skips it.")
                continue
            outputpath = outputfolder / f"{name.lstrip('/')}.parquet"
            outputpath.parent.mkdir(parents=True, exist_ok=True)
            table.write_parquet(outputpath, metadata=metadata)
            print(f"{path}:{name} -> {outputpath} ({table.height} rows, {table.width} columns)")
            outputpaths.append(outputpath)
    return outputpaths


def main(args: Sequence[str] | None = None) -> None:
    """Convert each HDF5 file that the command line names."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("paths", nargs="+", type=Path, help="the HDF5 files to convert")
    for path in parser.parse_args(args).paths:
        convert_file(path)


if __name__ == "__main__":
    main(sys.argv[1:])

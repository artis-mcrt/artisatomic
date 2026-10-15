#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.15"
# dependencies = ["h5py>=3.10.0", "numpy>=2.2.6", "polars>=2.0.0"]
# ///
"""Convert each table of an HDF5 file to a Parquet file.

The readers of artisatomic read Parquet files with polars, so artisatomic needs no HDF5 library.
Run this script once for each HDF5 file, and again after a change to the HDF5 file:

    artisatomic/hdf5_to_parquet.py atomic-data-helium-boyle/aoife.hdf5

uv installs the dependencies in the header above. h5py 3.16 has no wheel for Python 3.15, so the
header makes uv use an earlier Python. The script writes the tables into a folder that has the
name of the HDF5 file without its extension. It replaces the earlier content of that folder. The
path of a table in the folder is the path of its object in the HDF5 file. The script does not
change or delete the HDF5 file.

The script reads two layouts:

- A group with a pandas_type attribute holds a frame that DataFrame.to_hdf() of pandas wrote in
  the "fixed" format. The script rebuilds the frame without pandas.
- Each other HDF5 data set becomes a table. The fields of a compound type become the columns.

The key-value metadata of each Parquet file holds these items:

- the HDF5 attributes of the object, with their names;
- the attributes of each group above the object, with the path of the group before the name;
- the name of the HDF5 file (hdf5_file) and the SHA-256 hash of its content (hdf5_file_sha256).

The readers compare hdf5_file_sha256 with the HDF5 file, and they stop if the HDF5 file changed.
"""

import argparse
import hashlib
import importlib
import io
import json
import math
import numbers

# the pandas "fixed" format holds a block of Python objects as a pickle, so this script must
# unpickle it. ArrayUnpickler permits only the names of numpy arrays
import pickle  # ruff: ignore[suspicious-pickle-import]
import shutil
import tempfile
import unicodedata
from collections import Counter
from itertools import starmap
from pathlib import Path
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
from types import ModuleType
from typing import override

import numpy as np
import polars as pl
from polars.datatypes import DataTypeClass


class NoTableError(ValueError):
    """An HDF5 data set that cannot become a table, which the script skips."""


def import_h5py() -> ModuleType:
    """Import h5py, which the dependencies of the artisatomic package do not include.

    An import statement would make the type checkers report a missing module. h5py is in the
    optional extra hdf5, which the type checkers do not install.
    """
    try:
        return importlib.import_module("h5py")
    except ModuleNotFoundError as exc:
        msg = (
            "hdf5_to_parquet.py needs h5py. Run it as artisatomic/hdf5_to_parquet.py, and uv installs h5py."
            " Or install the extra hdf5 with Python 3.14: uv sync --python 3.14 --extra hdf5."
        )
        raise ModuleNotFoundError(msg) from exc


class ArrayUnpickler(pickle.Unpickler):  # ruff: ignore[suspicious-pickle-usage]
    """Unpickle only the numpy arrays that pandas writes for a block or an index level.

    An unpickle calls each function that the pickle names. The HDF5 file can come from a download,
    so the class permits only the names that a pickled numpy array needs.
    """

    permitted = frozenset(
        {
            ("numpy", "dtype"),
            ("numpy", "ndarray"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "scalar"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"),
        }
    )

    @override
    def find_class(self, module: str, name: str, /):
        """Give the class or function that the pickle names, if it is a permitted name."""
        if (module, name) not in self.permitted:
            msg = f"The pickle names {module}.{name}. The script unpickles only numpy arrays."
            raise pickle.UnpicklingError(msg)
        found = super().find_class(module, name)
        return checked_scalar(found) if name == "scalar" else found


def checked_scalar(scalar):
    """Give a version of numpy scalar() that refuses a numpy type that holds Python objects.

    scalar() unpickles the data of such a type with pickle.loads(). That inner unpickle does not use
    ArrayUnpickler, so it could call any function.
    """

    def scalar_of_no_objects(dtype: object, *args: object) -> object:
        if not isinstance(dtype, np.dtype) or dtype.hasobject:
            msg = f"The pickle has a numpy scalar of the type {dtype}. The script refuses a scalar of Python objects."
            raise pickle.UnpicklingError(msg)
        return scalar(dtype, *args)

    return scalar_of_no_objects


def unpickle(data: bytes):
    """Unpickle data with ArrayUnpickler."""
    return ArrayUnpickler(io.BytesIO(data)).load()


def as_text(value, encoding: str = "utf-8") -> str:
    """Give an HDF5 name or attribute as a str.

    h5py gives a byte string for a written attribute and a str for one that it decoded itself.
    """
    return value.decode(encoding) if isinstance(value, bytes) else str(value)


def holds_object(dtype: pl.DataType | DataTypeClass) -> bool:
    """Give True if a polars type is Object or holds Object. Parquet cannot hold that type."""
    if dtype == pl.Object:
        return True
    if isinstance(dtype, pl.List | pl.Array):
        return holds_object(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(holds_object(field.dtype) for field in dtype.fields)
    return False


def bytes_series(name: str, values: np.ndarray, encoding: str) -> pl.Series:
    """Make a String series from byte strings, or a Binary series if the bytes do not decode."""
    try:
        return pl.Series(name, np.strings.decode(values, encoding))
    except UnicodeDecodeError:
        return pl.Series(name, values)


def object_series(name: str, values: np.ndarray, encoding: str) -> pl.Series:
    """Make a polars series from an array of Python objects, as pandas pickles them.

    pandas uses a float NaN for a missing value, so a NaN becomes null. A mix of integers and
    floats becomes Float64. A mix of numbers and text becomes a String column. The DREAM line list
    has one such column, CF, which holds a float in most rows and a string in the others.
    """
    rows = [None if isinstance(value, float) and math.isnan(value) else value for value in values.tolist()]
    present = [value for value in rows if value is not None]
    if present and all(isinstance(value, bytes) for value in present):
        try:
            return pl.Series(name, [None if value is None else value.decode(encoding) for value in rows])
        except UnicodeDecodeError:
            return pl.Series(name, rows, dtype=pl.Binary)
    try:
        return pl.Series(name, rows)
    except (TypeError, pl.exceptions.PolarsError):
        # polars takes the type from the first value, so [2.5, 1] gives Float64 and [1, 2.5] fails
        pass
    if all(isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_) for value in present):
        return pl.Series(name, rows, dtype=pl.Float64, strict=False)
    return pl.Series(name, [None if value is None else as_text(value, encoding) for value in rows], dtype=pl.String)


def series_from_array(name: str, values: np.ndarray, encoding: str = "utf-8") -> pl.Series:
    """Make a polars series from one column, with a type that Parquet can hold.

    The function decodes byte strings with the encoding. Bytes in a different encoding stay bytes.
    The function raises NoTableError for a type that Parquet cannot hold, for example a complex
    number.
    """
    if values.dtype.kind == "S":
        series = bytes_series(name, values, encoding)
    elif values.dtype == object:
        series = object_series(name, values, encoding)
    else:
        try:
            series = pl.Series(name, values)
        except (TypeError, ValueError, pl.exceptions.PolarsError) as exc:
            msg = f"the column {name} has the numpy type {values.dtype}, which polars cannot read"
            raise NoTableError(msg) from exc
    if holds_object(series.dtype):
        msg = f"the column {name} has the type {series.dtype}, which Parquet cannot hold"
        raise NoTableError(msg)
    return series


def bitfield_as_bool(dataset, values: np.ndarray) -> np.ndarray:
    """Give the values of a bool array that PyTables wrote as an HDF5 bitfield, as bool values.

    h5py reads a bitfield as unsigned integers. The function changes only values of 0 and 1, so a
    bitfield of flags keeps its values.
    """
    h5py = import_h5py()
    if dataset.id.get_type().get_class() == h5py.h5t.BITFIELD and np.isin(values, (0, 1)).all():
        return values.astype(bool)
    return values


def pandas_values(dataset, typeattribute: str) -> np.ndarray:
    """Read one HDF5 data set of a pandas group, with the numpy type that pandas recorded.

    pandas stores a datetime64 or a timedelta64 array as int64 values. The attribute value_type of
    a block and the attribute kind of an index level hold the type. pandas stores an empty array
    as one placeholder element, and the attribute shape holds the real shape.
    """
    attrs = dataset.attrs
    where = f"{dataset.file.filename}:{dataset.name}"
    if "tz" in attrs:
        msg = f"{where} holds times with a time zone. The script reads no time zone."
        raise ValueError(msg)
    typename = as_text(attrs.get(typeattribute, b""))
    if typename in {"date", "category"} or typename.startswith("period"):
        msg = f"{where} has the pandas type {typename}. The script reads no such type."
        raise ValueError(msg)

    if "shape" in attrs:
        shape = attrs["shape"]
        valuetype = as_text(attrs.get("value_type", b"object"))
        values = np.empty(
            unpickle(shape) if isinstance(shape, bytes) else tuple(shape),
            dtype=object if valuetype in {"str", "string"} else np.dtype(valuetype),
        )
    elif dataset.dtype == object:
        values = unpickle(dataset[0].tobytes())
    else:
        values = bitfield_as_bool(dataset, dataset[:])

    # pandas stores the transpose of a block, with the rows first, and sets the attribute transposed.
    # It does not transpose an empty block, so the function does it here
    if not attrs.get("transposed", False) and values.ndim == 2:
        values = values.T
    if typename.startswith(("datetime64", "timedelta64")):
        values = values.view(typename)
    return values


def read_pandas_frame(group) -> pl.DataFrame:
    """Read one frame from a group that pandas wrote in the "fixed" format.

    That format holds the frame as blocks. Each block has an HDF5 data set of the values and an
    HDF5 data set of the column names. The function matches the values to the columns by name, so
    the order of the blocks does not matter. A group of a different pandas format raises a
    ValueError, because the block layout is the only layout that this function reads.
    """
    where = f"{group.file.filename}:{group.name}"
    pandas_type = as_text(group.attrs.get("pandas_type", b""))
    if pandas_type != "frame":
        msg = f"{where} has pandas_type {pandas_type!r}, not 'frame'. Write it with DataFrame.to_hdf(..., format='fixed')."
        raise ValueError(msg)
    # a frame with one plain index has no axis1_nlevels. The script reads only the index of
    # several levels, because no data source of artisatomic has the other kind
    if "axis1_nlevels" not in group.attrs:
        msg = f"{where} has no index of several levels. The script reads no other index."
        raise ValueError(msg)
    encoding = as_text(group.attrs.get("encoding", b"UTF-8"))

    def names_of(dataset) -> list[str]:
        return [as_text(name, encoding) for name in pandas_values(dataset, "kind").tolist()]

    serieslist: list[pl.Series] = []

    # the levels of the index come first, so the DREAM frame starts with Z and the ion charge
    for level in range(int(group.attrs["axis1_nlevels"])):
        leveldata = group[f"axis1_level{level}"]
        levelvalues = pandas_values(leveldata, "kind")
        codes = pandas_values(group[f"axis1_label{level}"], "value_type")
        # not an assert: pandas writes -1 for a row that the level does not hold, and numpy reads
        # -1 as the last value. Such a row would get the wrong value without a message
        if codes.size and codes.min() < 0:
            msg = f"{where} index level {level} has a row with no value."
            raise ValueError(msg)
        levelname = as_text(leveldata.attrs["name"])
        # pandas stores the pickle of None as the name of a level with no name. The reset_index()
        # method of pandas calls such a level level_0, level_1, and so on
        if levelname == "N.":
            levelname = f"level_{level}"
        serieslist.append(series_from_array(levelname, levelvalues[codes], encoding))

    blockvalues: dict[str, np.ndarray] = {}
    for block in range(int(group.attrs["nblocks"])):
        names = names_of(group[f"block{block}_items"])
        array = pandas_values(group[f"block{block}_values"], "value_type")
        # pandas 3 stores the block of one column of its str type as an array of one dimension
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        # one column for each name. A mismatch means that the group has a different layout
        if array.shape[1] != len(names):
            msg = f"{where} block {block} has {array.shape[1]} columns for {len(names)} names"
            raise ValueError(msg)
        blockvalues |= {name: array[:, index] for index, name in enumerate(names)}

    # axis0 holds the columns in the order of the frame, which the blocks do not keep
    columnnames = names_of(group["axis0"])
    allnames = [series.name for series in serieslist] + columnnames
    duplicates = sorted(name for name, count in Counter(allnames).items() if count > 1)
    # not an assert: a Parquet file holds each name once, so the data of one column would be lost
    if duplicates:
        msg = f"{where} has more than one index level or column with the name {', '.join(duplicates)}."
        raise ValueError(msg)
    serieslist += [series_from_array(name, blockvalues[name], encoding) for name in columnnames]

    return pl.DataFrame(serieslist)


def table_from_array(values) -> tuple[pl.DataFrame, dict[str, str]]:
    """Make a table from the values of one HDF5 data set, with metadata for the labels of enums.

    The fields of a compound type become the columns. An array of one dimension gives one column,
    "value". An array of two dimensions gives one column for each index of the second dimension,
    "column_0", "column_1", and so on. The function raises NoTableError for a different shape.
    """
    if not isinstance(values, np.ndarray) or values.ndim not in {1, 2}:
        msg = "has no table shape"
        raise NoTableError(msg)
    if values.dtype.names is not None:
        if values.ndim != 1:
            msg = "has no table shape"
            raise NoTableError(msg)
        columns = {name: values[name] for name in values.dtype.names}
    elif values.ndim == 1:
        columns = {"value": values}
    else:
        columns = {f"column_{index}": values[:, index] for index in range(values.shape[1])}

    # h5py keeps the labels of an enum in the metadata of the numpy type, and polars drops them
    metadata = {
        f"{name} enum": json.dumps(enum, default=int)
        for name, column in columns.items()
        if column.dtype.metadata and (enum := column.dtype.metadata.get("enum"))
    }
    return pl.DataFrame(list(starmap(series_from_array, columns.items()))), metadata


def read_dataset(dataset) -> tuple[pl.DataFrame, dict[str, str]]:
    """Read one HDF5 data set as a table (see table_from_array())."""
    values = dataset[()]
    if isinstance(values, np.ndarray) and values.dtype.names is None:
        values = bitfield_as_bool(dataset, values)
    return table_from_array(values)


def attribute_text(value, h5file) -> str:
    """Give one HDF5 attribute as text.

    A byte that is not part of UTF-8 text becomes a backslash escape. An array becomes JSON text.
    A reference becomes the path of the object that it names.
    """
    h5py = import_h5py()
    if isinstance(value, h5py.Reference):
        return h5file[value].name if value else ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="backslashreplace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "S":
            value = np.strings.decode(value, "utf-8", "backslashreplace")
        return json.dumps(value.tolist(), default=str)
    return str(value)


def metadata_of(h5object) -> dict[str, str]:
    """Give the HDF5 attributes of an object as text, for the key-value metadata of Parquet."""
    h5py = import_h5py()
    return {
        key: attribute_text(value, h5object.file)
        for key, value in h5object.attrs.items()
        # an attribute with no value, as pytables writes for the TITLE of each node
        if not isinstance(value, h5py.Empty)
    }


def group_metadata(h5file, objectpath: str) -> dict[str, str]:
    """Give the attributes of each group above an object, with the path of the group before the name.

    For example, the attributes md5 and uuid1 of the root group of aoife.hdf5 become /md5 and /uuid1.
    """
    return {
        f"{grouppath.as_posix().rstrip('/')}/{key}": value
        for grouppath in PurePosixPath(objectpath).parents
        for key, value in metadata_of(h5file[grouppath.as_posix()]).items()
    }


def tables_of(h5file) -> list[tuple[str, pl.DataFrame, dict[str, str]]]:
    """Read each table of an HDF5 file, with the path and the metadata of the table.

    visititems() visits each object one time, also an object with two hard links. It follows no
    soft link and no external link. It visits a group before the objects in the group. The
    function skips the objects in a pandas group, because the frame of the group holds their data.
    """
    h5py = import_h5py()
    tables: list[tuple[str, pl.DataFrame, dict[str, str]]] = []
    pandasgroups: list[str] = []

    def visit(name: str, h5object) -> None:
        if any(name.startswith(f"{groupname}/") for groupname in pandasgroups):
            return
        if isinstance(h5object, h5py.Group):
            if "pandas_type" in h5object.attrs:
                pandasgroups.append(name)
                tables.append((h5object.name, read_pandas_frame(h5object), metadata_of(h5object)))
        # a committed datatype holds no data
        elif isinstance(h5object, h5py.Dataset):
            try:
                table, enummetadata = read_dataset(h5object)
            except NoTableError as exc:
                print(f"{h5object.file.filename}:{h5object.name} {exc}. The script skips it.")
            else:
                tables.append((h5object.name, table, metadata_of(h5object) | enummetadata))

    h5file.visititems(visit)
    return tables


def output_folder_of(path: Path) -> Path:
    """Give the output folder of an HDF5 file: the path of the file without its extension."""
    # not an assert: without an extension, the output folder would have the path of the HDF5 file
    if not path.suffix:
        msg = f"{path} has no extension. The script names the output folder after the file without its extension."
        raise ValueError(msg)
    return path.with_suffix("")


def check_output_paths(path: Path, relativepaths: list[str]) -> None:
    """Stop the run if two tables get one output file, or if a table goes outside the output folder.

    HDF5 names are case-sensitive, but the default file systems of macOS and Windows are not.
    Windows also reads a backslash as a separator. macOS gives one file to two Unicode forms of one
    name, for example a composed and a decomposed letter. Such names would give one file, and one
    table would be lost.
    """

    def filekey(relativepath: str) -> str:
        normalised = unicodedata.normalize("NFC", relativepath.replace("\\", "/"))
        # Windows removes the periods and the spaces at the end of each part of a path
        parts = [part.rstrip(". ") for part in normalised.split("/")]
        return unicodedata.normalize("NFC", "/".join(parts).casefold())

    counts = Counter(map(filekey, relativepaths))
    clashes = sorted(relativepath for relativepath in relativepaths if counts[filekey(relativepath)] > 1)
    if clashes:
        msg = f"{path} has tables that would get one output file: {', '.join(clashes)}."
        raise ValueError(msg)
    # the two path types: Windows also reads a backslash as a separator, and C: as a drive
    unsafe = [
        relativepath
        for relativepath in relativepaths
        if PurePosixPath(relativepath).is_absolute()
        or PureWindowsPath(relativepath).anchor
        or {".", ".."} & {*PurePosixPath(relativepath).parts, *PureWindowsPath(relativepath).parts}
    ]
    if unsafe:
        msg = f"{path} has tables whose path goes outside the output folder: {', '.join(unsafe)}."
        raise ValueError(msg)


def write_tables(outputfolder: Path, tables: list[tuple[str, pl.DataFrame, dict[str, str]]]) -> None:
    """Write the tables into a new folder, and then put the new folder in the place of outputfolder.

    An error leaves the earlier output folder as it was. The script makes the output folder, so it
    replaces the folder as a whole. A table that the HDF5 file no longer holds goes away too. The
    function stops if the earlier folder holds a file that is not a Parquet file. The script did
    not write such a file. The function permits only .DS_Store, which the Finder of macOS writes.
    """
    # the script makes only a folder at this path, also for an input such as data.parquet.h5
    if outputfolder.is_symlink() or (outputfolder.exists() and not outputfolder.is_dir()):
        msg = f"{outputfolder} is not a folder that the script made."
        raise FileExistsError(msg)
    if outputfolder.exists():
        otherfiles = [
            filepath
            for filepath in outputfolder.rglob("*")
            if not filepath.is_dir() and filepath.suffix != ".parquet" and filepath.name != ".DS_Store"
        ]
        if otherfiles:
            msg = f"{outputfolder} holds files that the script did not write, for example {otherfiles[0]}."
            raise FileExistsError(msg)

    # a new folder with a unique name, so the function deletes no folder that it did not make. The
    # earlier output folder stays in it as a backup until the new tables have their place
    workfolder = Path(tempfile.mkdtemp(prefix=f"{outputfolder.name}.partial-", dir=outputfolder.parent))
    newfolder = workfolder / "new"
    # an HDF5 file with no table gives an empty output folder
    newfolder.mkdir()
    backupfolder = workfolder / "old"
    keepworkfolder = False
    try:
        for relativepath, table, metadata in tables:
            outputpath = newfolder / relativepath
            outputpath.parent.mkdir(parents=True, exist_ok=True)
            table.write_parquet(outputpath, metadata=metadata)
        if outputfolder.exists():
            outputfolder.rename(backupfolder)
        try:
            newfolder.rename(outputfolder)
        except BaseException as exc:
            if backupfolder.exists():
                try:
                    backupfolder.rename(outputfolder)
                except OSError as restoreexc:
                    # the backup holds the only copy of the earlier tables, so the function keeps it
                    keepworkfolder = True
                    msg = f"The script could not put back the earlier tables. They are in {backupfolder}."
                    raise RuntimeError(msg) from restoreexc
            raise exc from None
    finally:
        if not keepworkfolder:
            shutil.rmtree(workfolder, ignore_errors=True)


def convert_file(path: Path) -> list[Path]:
    """Write each table of one HDF5 file as a Parquet file, and give the paths of those files.

    The function reads all the tables before it writes one, so an error in one table writes nothing.
    """
    h5py = import_h5py()
    outputfolder = output_folder_of(path)
    with path.open("rb") as hdf5file:
        filemetadata = {"hdf5_file": path.name, "hdf5_file_sha256": hashlib.file_digest(hdf5file, "sha256").hexdigest()}
    with h5py.File(path, "r") as h5file:
        tables = [
            (f"{name.lstrip('/')}.parquet", table, group_metadata(h5file, name) | metadata | filemetadata)
            for name, table, metadata in tables_of(h5file)
        ]
    check_output_paths(path, [relativepath for relativepath, _, _ in tables])
    write_tables(outputfolder, tables)
    for relativepath, table, _ in tables:
        print(f"{path} -> {outputfolder / relativepath} ({table.height} rows, {table.width} columns)")
    return [outputfolder / relativepath for relativepath, _, _ in tables]


def main() -> None:
    """Convert each HDF5 file that the command line names."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("paths", nargs="+", type=Path, help="the HDF5 files to convert")
    paths = parser.parse_args().paths
    # two input files with one output folder: the second conversion would delete the tables of the first
    outputfolders = Counter(str(output_folder_of(path).resolve()).casefold() for path in paths)
    clashes = [str(path) for path in paths if outputfolders[str(output_folder_of(path).resolve()).casefold()] > 1]
    if clashes:
        parser.error(f"these files have one output folder: {', '.join(clashes)}")
    for path in paths:
        convert_file(path)


if __name__ == "__main__":
    main()

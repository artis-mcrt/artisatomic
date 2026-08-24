"""Read levels and transitions from the Floers+25 data set, calibrated or uncalibrated."""

import os
import string
import typing as t
from pathlib import Path

import pandas as pd
import polars as pl

import artisatomic


def get_basepath(withforbidden: bool) -> Path:
    """Directory that holds the Floers+25 level and transition tables.

    The private OutputFiles_withforbidden directory holds the newer data with forbidden
    transitions. The public OutputFiles directory holds the published data.
    """
    dirname = "OutputFiles_withforbidden" if withforbidden else "OutputFiles"
    return artisatomic.PYDIR / ".." / "atomic-data-floers25" / dirname


def in_testmode() -> bool:
    """Say whether the test mode is active. Test mode uses only the public data."""
    return os.environ.get("ARTISATOMIC_TESTMODE") == "1"


def extend_ion_list(ion_handlers, calibrated=True):
    """Add every ion with a Floers+25 data file to ion_handlers.

    The handler priority from highest to lowest is floers25calibwithforbidden, floers25calib,
    and floers25uncalib. An ion with no calibrated data gets its uncalibrated version rather
    than being left out. With calibrated=False, only the uncalibrated handler is added.
    """
    basepath_public = get_basepath(withforbidden=False)
    basepath_private = get_basepath(withforbidden=True)
    # not an assert: input validation must survive python -O
    if not basepath_public.is_dir():
        msg = f"Directory {basepath_public} does not exist. Run atomic-data-floers25/setup_floers25_data.sh to download the data."
        raise FileNotFoundError(msg)

    # the searches run in priority order, because add_handler_if_not_set() keeps the first
    # handler that matches an ion
    calibsearches = [
        ("floers25calibwithforbidden", "calib", basepath_private),
        ("floers25calib", "calib", basepath_public),
    ]
    searches = (calibsearches if calibrated else []) + [
        ("floers25uncalib", "uncalib", basepath_private),
        ("floers25uncalib", "uncalib", basepath_public),
    ]

    for handlername, calibstr, basepath in searches:
        if basepath == basepath_private and (in_testmode() or not basepath.is_dir()):
            continue
        for ext in artisatomic.compression_extensions:
            for s in basepath.glob(f"*_levels_{calibstr}.txt{ext}"):
                ionstr = s.name.lstrip(string.digits).split("_")[0]
                atomic_number, ion_stage = artisatomic.split_element_ionstage_str(ionstr)
                ion_handlers = artisatomic.add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, handlername)

    return ion_handlers


class FloersEnergyLevel(t.NamedTuple):
    """One energy level of the Floers+25 data set."""

    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: int


def read_dashed_table(filepath: Path, dtype: dict[str, t.Any] | None = None) -> pl.DataFrame:
    """Read the whitespace-separated data table that starts after the third '----' line."""
    dashrowcount = 0
    with artisatomic.xopen_check_extension(filepath) as f:
        for line in f:
            if line.startswith("--"):
                dashrowcount += 1
                if dashrowcount == 3:
                    break
        if dashrowcount < 3:
            msg = f"Did not find the expected data table in {filepath}"
            raise ValueError(msg)
        return pl.from_pandas(pd.read_csv(f, sep=r"\s+", dtype_backend="pyarrow", dtype=dtype))


def read_levels_and_transitions(
    atomic_number: int, ion_stage: int, flog, calibrated: bool, withforbidden: bool = False
):
    """Read one ion from the Floers+25 data set.

    The ionization energy comes from NIST rather than the file. Configurations are not unique
    (levels of one configuration differ by J), so level names combine the configuration, J and
    the file's index. Both the level indices and the transitions' references to them are
    validated, since a gap or an out-of-range index would silently misattach transitions.
    """
    elsym = artisatomic.elsymbols[atomic_number]
    ion_stage_roman = artisatomic.roman_numerals[ion_stage]
    calibstr = "calib" if calibrated else "uncalib"
    ionstr = f"{atomic_number}{elsym}{ion_stage_roman}"

    # the handler name selects the directory. The floers25uncalib handler has no "withforbidden"
    # variant, so it searches the private directory and then the public directory.
    if withforbidden or calibrated:
        basepaths = [get_basepath(withforbidden=withforbidden)]
    elif in_testmode():
        basepaths = [get_basepath(withforbidden=False)]
    else:
        basepaths = [get_basepath(withforbidden=True), get_basepath(withforbidden=False)]

    levels_file = next(
        (
            found
            for searchpath in basepaths
            if (found := artisatomic.find_file_check_extension(searchpath / f"{ionstr}_levels_{calibstr}.txt"))
            is not None
        ),
        None,
    )
    if levels_file is None:
        searched = " or ".join(str(searchpath / f"{ionstr}_levels_{calibstr}.txt*") for searchpath in basepaths)
        msg = f"Found no Floers+25 levels file for {ionstr}. Searched {searched}"
        raise FileNotFoundError(msg)
    basepath = levels_file.parent

    # the original Floers+25 format has a single transitions file. The newer format has one
    # file for each transition type, for example _E1, _E2, and _M1.
    lines_file = artisatomic.find_file_check_extension(basepath / f"{ionstr}_transitions_{calibstr}.txt")

    # a table can exist in a plain form and in a compressed form at the same time. Resolve each
    # name once through the extension list, so no table is read twice.
    pertype_names = sorted(
        {
            filepath.name.removesuffix(ext)
            for ext in artisatomic.compression_extensions
            for filepath in basepath.glob(f"{ionstr}_transitions_{calibstr}_*.txt{ext}")
        }
    )
    pertype_files = [
        found for name in pertype_names if (found := artisatomic.find_file_check_extension(basepath / name)) is not None
    ]

    if lines_file is not None and pertype_files:
        msg = f"Found both {lines_file.name} and per-type transitions files in {basepath}. Remove one of the forms."
        raise ValueError(msg)
    transition_files = [lines_file] if lines_file is not None else pertype_files
    if not transition_files:
        msg = f"Found no Floers+25 transitions files for {ionstr} ({calibstr}) in {basepath}"
        raise FileNotFoundError(msg)

    artisatomic.log_and_print(
        flog,
        f"Reading Floers+25 {calibstr}rated data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman}) from {basepath.name}/{levels_file.name} and {len(transition_files)} transition files",
    )

    ionization_energy_in_ev = artisatomic.get_nist_ionization_energies_ev()[atomic_number, ion_stage]

    dflevels = read_dashed_table(levels_file, dtype={"J": str})

    dflevels = dflevels.with_columns(
        pl.when(pl.col("J").str.ends_with("/2"))
        .then(pl.col("J").str.strip_suffix("/2").cast(pl.Int32) + 1)
        .otherwise(
            pl.col("J").str.strip_suffix("/2").cast(pl.Int32) * 2 + 1
        )  # the strip_suffix should not be needed (does not end in "/2" but prevents a polars error)
        .alias("g")
    )

    # the levels are indexed from zero in file order and the transitions refer to those indices,
    # so a gap would misattach them. Not an assert: input validation must survive python -O.
    if dflevels["Index"].to_list() != list(range(dflevels.height)):
        msg = f"Level indices in {levels_file} are not contiguous and zero-based"
        raise ValueError(msg)

    # Configuration is not unique (levels of one configuration differ by J), so append the
    # contiguous index. The configuration stays first, for get_level_valence_n() and the
    # adata.txt comment.
    dflevels = dflevels.with_columns(
        levelname=pl.format("{} J={} index={}", pl.col("Configuration"), pl.col("J"), pl.col("Index"))
    )

    artisatomic.log_and_print(flog, f"Read {dflevels.height:d} levels")

    # keep only the used columns before the concatenation. This also makes the tables
    # compatible: pandas infers a string type for every column of a table with no data rows.
    dftransitions = pl.concat(
        [
            read_dashed_table(transition_file).select(
                lowerlevel=pl.col("Lower").cast(pl.Int64),
                upperlevel=pl.col("Upper").cast(pl.Int64),
                A=pl.col("A").cast(pl.Float64),
            )
            for transition_file in transition_files
        ]
    )

    artisatomic.log_and_print(flog, f"Read {dftransitions.height} transitions")

    # an out-of-range level reference would be silently dropped by the joins in
    # add_level_ids_forbidden(). Not an assert: input validation must survive python -O.
    if dftransitions.height > 0:
        transition_level_indices = pl.concat([dftransitions["lowerlevel"], dftransitions["upperlevel"]])
        min_index = t.cast("int", transition_level_indices.min())
        max_index = t.cast("int", transition_level_indices.max())
        if min_index < 0 or max_index >= dflevels.height:
            msg = (
                f"Transition level indices for {ionstr} in {basepath} span {min_index}..{max_index}, outside the"
                f" level table's 0..{dflevels.height - 1}"
            )
            raise ValueError(msg)

    # the file's Lower/Upper indices are already zero-based, matching the level ids used in memory.
    # Merge the rows that share a level pair. Add their A values. ARTIS reads each row as one
    # transition, so a duplicate pair would double a line.
    dftransitions = dftransitions.group_by(["upperlevel", "lowerlevel"], maintain_order=True).agg(A=pl.col("A").sum())

    # count after the merge of duplicate level pairs. The counts in adata.txt then agree with
    # transitiondata.txt. Count per level index, not per configuration string: several levels
    # share a configuration.
    transition_count_of_levelindex: dict[int, int] = dict(
        pl.concat([dftransitions["lowerlevel"], dftransitions["upperlevel"]]).value_counts().iter_rows()
    )
    transition_count_of_level_name = {
        levelname: transition_count_of_levelindex.get(index, 0)
        for index, levelname in dflevels.select("Index", "levelname").iter_rows()
    }

    # use standard artisatomic column names.
    # the levels carry a parity, so add_level_ids_forbidden() can derive the forbidden flag from it
    dflevels = dflevels.select(
        levelname=pl.col("levelname"),
        parity=pl.col("Parity"),
        g=pl.col("g"),
        energyabovegsinpercm=pl.col("Energy"),
    )

    return ionization_energy_in_ev, dflevels, dftransitions, transition_count_of_level_name


def get_level_valence_n(levelname: str):
    """Principal quantum number of the valence electron, read from a Floers+25 level name.

    Kept separate from the other readers' versions: each data source names its levels
    differently, so a shared parser would have to guess which convention it is looking at.
    """
    # level names are "<configuration> J=<J> index=<index>", so drop everything after the config
    part = levelname.split(" ", maxsplit=1)[0].rsplit(".", maxsplit=1)[-1]
    if part[-1] not in "spdfg":
        # end of string is a number of electrons in the orbital, not a principal quantum number, so remove it
        assert part[-1].isdigit()
        part = part.rstrip(string.digits)
    return int(part.rstrip("spdfg"))

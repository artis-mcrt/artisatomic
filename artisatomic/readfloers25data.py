import typing as t
from pathlib import Path

import pandas as pd
import polars as pl

import artisatomic


def get_basepath() -> Path:
    return artisatomic.PYDIR / ".." / "atomic-data-floers25" / "OutputFiles"


def extend_ion_list(ion_handlers, calibrated=True):
    BASEPATH = get_basepath()
    assert BASEPATH.is_dir()
    # if calibrated is requested, also add uncalibrated data where calibrated data is not available
    calibflags = [True, False] if calibrated else [False]
    for searchcalib in calibflags:
        calibstr = "calib" if searchcalib else "uncalib"
        handlername = f"floers25{calibstr}"
        for s in BASEPATH.glob(f"*_levels_{calibstr}.txt*"):
            ionstr = s.name.lstrip("0123456789").split("_")[0]
            atomic_number, ion_stage = artisatomic.split_element_ionstage_str(ionstr)
            ion_handlers = artisatomic.add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, handlername)

    return ion_handlers


class FloersEnergyLevel(t.NamedTuple):
    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: int


def read_levels_and_transitions(atomic_number: int, ion_stage: int, flog, calibrated: bool):
    # ion_charge = ion_stage - 1
    elsym = artisatomic.elsymbols[atomic_number]
    ion_stage_roman = artisatomic.roman_numerals[ion_stage]
    calibstr = "calib" if calibrated else "uncalib"

    BASEPATH = get_basepath()
    ionstr = f"{atomic_number}{elsym}{ion_stage_roman}"
    levels_file = BASEPATH / f"{ionstr}_levels_{calibstr}.txt"
    lines_file = BASEPATH / f"{ionstr}_transitions_{calibstr}.txt"

    artisatomic.log_and_print(
        flog,
        f"Reading Floers+25 {calibstr}rated data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman}) from {levels_file.name} and {lines_file.name}",
    )

    ionization_energy_in_ev = artisatomic.get_nist_ionization_energies_ev()[(atomic_number, ion_stage)]

    dashrowcount = 0
    with artisatomic.xopen_check_extension(levels_file) as f:
        for line in f:
            if line.startswith("--"):
                dashrowcount += 1
                if dashrowcount == 3:  # data table starts after the '----' lines
                    break

        dflevels = pl.from_pandas(pd.read_csv(f, sep=r"\s+", dtype_backend="pyarrow", dtype={"J": str}))
    if dashrowcount < 3:
        raise ValueError(f"Did not find expected data table in {levels_file}")

    dflevels = dflevels.with_columns(
        pl.when(pl.col("J").str.ends_with("/2"))
        .then(pl.col("J").str.strip_suffix("/2").cast(pl.Int32) + 1)
        .otherwise(
            pl.col("J").str.strip_suffix("/2").cast(pl.Int32) * 2 + 1
        )  # the strip_suffix should not be needed (does not end in "/2" but prevents a polars error)
        .alias("g")
    )

    # The level table is indexed from zero in file order and the transitions refer to those
    # indices, so a gap would silently attach transitions to the wrong levels. Not an assert:
    # this validates an input file and must not disappear under python -O.
    if dflevels["Index"].to_list() != list(range(dflevels.height)):
        msg = f"Level indices in {levels_file} are not contiguous and zero-based"
        raise ValueError(msg)

    # Configuration is not unique (levels of the same configuration differ by J), so build a
    # unique level name from it. Index is contiguous, so including it guarantees uniqueness.
    # The configuration stays first so that get_level_valence_n() and the adata.txt comment
    # column both still start with it.
    dflevels = dflevels.with_columns(
        levelname=pl.format("{} J={} index={}", pl.col("Configuration"), pl.col("J"), pl.col("Index"))
    )

    artisatomic.log_and_print(flog, f"Read {dflevels.height:d} levels")

    dashrowcount = 0
    with artisatomic.xopen_check_extension(lines_file) as f:
        for line in f:
            if line.startswith("--"):
                dashrowcount += 1
                if dashrowcount == 3:  # data table starts after the '----' lines
                    break

        dftransitions = pl.from_pandas(pd.read_csv(f, sep=r"\s+", dtype_backend="pyarrow"))
    if dashrowcount < 3:
        raise ValueError(f"Did not find expected data table in {lines_file}")

    artisatomic.log_and_print(flog, f"Read {dftransitions.height} transitions")

    # Transitions reference levels by zero-based index, and an out-of-range reference would be
    # silently dropped by the level-id joins in add_level_ids_forbidden(), leaving an incomplete
    # database. Not an assert: this validates an input file and must not disappear under python -O.
    if dftransitions.height > 0:
        transition_level_indices = pl.concat([dftransitions["Lower"], dftransitions["Upper"]])
        min_index = t.cast("int", transition_level_indices.min())
        max_index = t.cast("int", transition_level_indices.max())
        if min_index < 0 or max_index >= dflevels.height:
            msg = (
                f"Transition level indices in {lines_file} span {min_index}..{max_index}, outside the"
                f" level table's 0..{dflevels.height - 1}"
            )
            raise ValueError(msg)

    # count per level index, not per configuration string: several levels share a configuration
    transition_count_of_levelindex: dict[int, int] = dict(
        pl.concat([dftransitions["Lower"], dftransitions["Upper"]]).value_counts().iter_rows()
    )
    transition_count_of_level_name = {
        levelname: transition_count_of_levelindex.get(index, 0)
        for index, levelname in dflevels.select("Index", "levelname").iter_rows()
    }

    # use standard artisatomic column names

    dflevels = dflevels.select(
        levelname=pl.col("levelname"),
        parity=pl.col("Parity"),
        g=pl.col("g"),
        energyabovegsinpercm=pl.col("Energy"),
    )

    # the levels carry a parity, so let add_level_ids_forbidden() derive the forbidden flag from it
    dftransitions = dftransitions.select(lowerlevel=pl.col("Lower") + 1, upperlevel=pl.col("Upper") + 1, A=pl.col("A"))

    return ionization_energy_in_ev, dflevels, dftransitions, transition_count_of_level_name


def get_level_valence_n(levelname: str):
    # level names are "<configuration> J=<J> index=<index>", so drop everything after the config
    part = levelname.split(" ", maxsplit=1)[0].rsplit(".", maxsplit=1)[-1]
    if part[-1] not in "spdfg":
        # end of string is a number of electrons in the orbital, not a principal quantum number, so remove it
        assert part[-1].isdigit()
        part = part.rstrip("0123456789")
    return int(part.rstrip("spdfg"))

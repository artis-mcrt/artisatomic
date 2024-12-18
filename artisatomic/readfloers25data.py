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
            elsym = ionstr.rstrip("IVX")
            ion_stage_roman = ionstr.removeprefix(elsym)
            atomic_number = artisatomic.elsymbols.index(elsym)

            ion_stage = artisatomic.roman_numerals.index(ion_stage_roman)

            found_element = False
            for tmp_atomic_number, list_ions in ion_handlers:
                if tmp_atomic_number == atomic_number:
                    if ion_stage not in [x[0] if len(x) > 0 else x for x in list_ions]:
                        list_ions.append((ion_stage, handlername))
                        list_ions.sort()
                    found_element = True
            if not found_element:
                ion_handlers.append(
                    (
                        atomic_number,
                        [(ion_stage, handlername)],
                    )
                )

    ion_handlers.sort(key=lambda x: x[0])

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

    dflevels = pl.from_pandas(
        pd.read_csv(levels_file, sep=r"\s+", skiprows=18, dtype_backend="pyarrow", dtype={"J": str})
    ).with_columns(
        pl.when(pl.col("J").str.ends_with("/2"))
        .then(pl.col("J").str.strip_suffix("/2").cast(pl.Int32) + 1)
        .otherwise(
            pl.col("J").str.strip_suffix("/2").cast(pl.Int32) * 2 + 1
        )  # the strip_suffix should not be needed (does not end in "/2" but prevents a polars error)
        .alias("g")
    )

    dflevels = dflevels.with_columns(pl.col("J").str.strip_suffix("/2").cast(pl.Float32).alias("2J"))

    artisatomic.log_and_print(flog, f"Read {dflevels.height:d} levels")

    dftransitions = pl.from_pandas(pd.read_csv(lines_file, sep=r"\s+", skiprows=28, dtype_backend="pyarrow"))

    artisatomic.log_and_print(flog, f"Read {dftransitions.height} transitions")

    transition_count_of_level_name = {
        config: (
            dftransitions.filter(pl.col("Config_Lower") == config).height
            + dftransitions.filter(pl.col("Config_Upper") == config).height
        )
        for config in dflevels["Configuration"]
    }

    # use standard artisatomic column names and convert to 1-indexed levels

    dflevels = artisatomic.add_dummy_zero_level(
        dflevels.select(
            levelname=pl.col("Configuration"),
            parity=pl.col("Parity"),
            g=pl.col("g"),
            energyabovegsinpercm=pl.col("Energy"),
        )
    )

    dftransitions = dftransitions.select(
        lowerlevel=pl.col("Lower") + 1, upperlevel=pl.col("Upper") + 1, A=pl.col("A"), forbidden=pl.lit(False)
    )

    # this check is slow
    # assert sum(transition_count_of_level_name.values()) == len(transitions) * 2

    return ionization_energy_in_ev, dflevels, dftransitions, transition_count_of_level_name


def get_level_valence_n(levelname: str):
    part = levelname.split(".")[-1]
    if part[-1] not in "spdfg":
        # end of string is a number of electrons in the orbital, not a principal quantum number, so remove it
        assert part[-1].isdigit()
        part = part.rstrip("0123456789")
    return int(part.rstrip("spdfg"))

import os
import re
from pathlib import Path

import numpy as np
import polars as pl

import artisatomic

kuruczdatapath = Path(__file__).parent.absolute() / ".." / "atomic-data-kurucz"
if os.environ.get("ARTISATOMIC_TESTMODE") == "1":
    kuruczdatapath = kuruczdatapath / "test_sample"


def parse_gfall(fname: str) -> pl.LazyFrame:
    # Code derived from the GFALL reader of carsus
    # https://github.com/tardis-sn/carsus/blob/master/carsus/io/kurucz/gfall.py
    gfall_fortran_format = (
        "F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,"
        "A10,F6.2,F6.2,F6.2,A4,I2,I2,I3,F6.3,I3,F6.3,I5,I5,"
        "1X,I1,A1,1X,I1,A1,I1,A3,I5,I5,I6"
    )

    gfall_columns = [
        "wavelength_nm",
        "loggf",
        "z_dot_ioncharge",
        "energyabovegsinpercm_first",
        "j_first",
        "blank1",
        "label_first",
        "energyabovegsinpercm_second",
        "j_second",
        "blank2",
        "label_second",
        "log_gamma_rad",
        "log_gamma_stark",
        "log_gamma_vderwaals",
        "ref",
        "nlte_level_no_first",
        "nlte_level_no_second",
        "isotope",
        "log_f_hyperfine",
        "isotope2",
        "log_iso_abundance",
        "hyper_shift_first",
        "hyper_shift_second",
        "blank3",
        "hyperfine_f_first",
        "hyperfine_note_first",
        "blank4",
        "hyperfine_f_second",
        "hyperfine_note_second",
        "line_strength_class",
        "line_code",
        "lande_g_first",
        "lande_g_second",
        "isotopic_shift",
    ]
    number_match = re.compile(r"\d+(\.\d+)?")
    type_match = re.compile(r"[FIXA]")
    type_dict = {"F": np.float64, "I": np.int64, "X": str, "A": str}
    field_types = tuple(type_dict[item] for item in number_match.sub("", gfall_fortran_format).split(","))

    field_widths = list(map(int, re.sub(r"\.\d+", "", type_match.sub("", gfall_fortran_format)).split(",")))

    import pandas as pd

    gfall = (
        pl.from_pandas(
            pd.read_fwf(
                fname,
                widths=field_widths,
                skip_blank_lines=True,
                names=gfall_columns,
                dtypes=dict(zip(gfall_columns, field_types, strict=True)),
                compression="infer",
                dtype_backend="pyarrow",
            )
        )
        .lazy()
        .drop_nulls(["z_dot_ioncharge", "energyabovegsinpercm_first", "energyabovegsinpercm_second"])
    )
    double_columns = [col.replace("_first", "") for col in gfall.collect_schema().names() if col.endswith("first")]

    # due to the fact that energy is stored in 1/cm
    gfall = gfall.with_columns(
        order_lower_upper=pl.col("energyabovegsinpercm_first").abs() < pl.col("energyabovegsinpercm_second").abs()
    )
    gfall = gfall.with_columns(
        pl.when(pl.col("order_lower_upper"))
        .then(f"{column}_first")
        .otherwise(f"{column}_second")
        .alias(f"{column}_lower")
        for column in double_columns
    ).with_columns(
        pl.when(pl.col("order_lower_upper"))
        .then(f"{column}_second")
        .otherwise(f"{column}_first")
        .alias(f"{column}_upper")
        for column in double_columns
    )

    # Clean labels
    ignored_labels = ["AVERAGE", "ENERGIES", "CONTINUUM"]
    gfall = gfall.with_columns(
        pl.col("label_lower").str.strip_chars().replace(r"\s+", " "),
        pl.col("label_upper").str.strip_chars().replace(r"\s+", " "),
    ).filter(
        (pl.col("label_lower").is_in(ignored_labels).not_()) & (pl.col("label_upper").is_in(ignored_labels).not_())
    )

    gfall = gfall.with_columns(
        energyabovegsinpercm_lower_predicted=pl.col("energyabovegsinpercm_lower") < 0,
        energyabovegsinpercm_lower=pl.col("energyabovegsinpercm_lower").abs(),
        energyabovegsinpercm_upper_predicted=pl.col("energyabovegsinpercm_upper") < 0,
        energyabovegsinpercm_upper=pl.col("energyabovegsinpercm_upper").abs(),
    )

    gfall = gfall.with_columns(atomic_number=pl.col("z_dot_ioncharge").cast(pl.Int64)).with_columns(
        ion_charge=((pl.col("z_dot_ioncharge") - pl.col("atomic_number")) * 100).round().cast(pl.Int64),
    )
    if gfall.select(pl.n_unique("z_dot_ioncharge")).collect().item() != 1:
        raise ValueError(f"Expected exactly one unique ion in file {fname}, but found multiple")

    return gfall


def find_gfall(atomic_number: int, ion_charge: int) -> Path:
    extended_atoms_filenames = [
        f"gf{atomic_number:02d}{ion_charge:02d}.lines.zst",
        f"gf{atomic_number:02d}{ion_charge:02d}.lines",
        f"gf{atomic_number:02d}{ion_charge:02d}z.lines.zst",
        f"gf{atomic_number:02d}{ion_charge:02d}z.lines",
    ]
    for filename in extended_atoms_filenames:
        path_gfall = (kuruczdatapath / "extendedatoms" / filename).resolve()
        if path_gfall.is_file():
            return path_gfall
    zztar_filenames = [f"gf{atomic_number:02d}{ion_charge:02d}.all", f"gf{atomic_number:02d}{ion_charge:02d}.all.zst"]
    for filename in zztar_filenames:
        path_gfall = (kuruczdatapath / "zztar" / filename).resolve()
        if path_gfall.is_file():
            return path_gfall

    raise FileNotFoundError(f"No Kurucz file for Z={atomic_number} ion_charge {ion_charge}.")


def read_levels_and_transitions(
    atomic_number: int, ion_stage: int, flog
) -> tuple[float, pl.DataFrame, pl.DataFrame, dict[str, int]]:
    ion_charge = ion_stage - 1

    artisatomic.log_and_print(flog, f"Using Kurucz for Z={atomic_number} ion_stage {ion_stage}")

    path_gfall = find_gfall(atomic_number, ion_charge)
    artisatomic.log_and_print(flog, f"Reading {path_gfall}")

    gfall = parse_gfall(fname=str(path_gfall))
    column_renames = {
        "energyabovegsinpercm_{0}": "energyabovegsinpercm",
        "j_{0}": "j",
        "label_{0}": "label",
        "energyabovegsinpercm_{0}_predicted": "theoretical",
    }

    e_lower_levels = gfall.rename({key.format("lower"): value for key, value in column_renames.items()})
    e_upper_levels = gfall.rename({key.format("upper"): value for key, value in column_renames.items()})

    selected_columns = ["atomic_number", "ion_charge", "energyabovegsinpercm", "j", "label", "theoretical"]
    dflevels = (
        pl.concat([e_lower_levels.select(selected_columns), e_upper_levels.select(selected_columns)])
        .unique(["energyabovegsinpercm", "j"], keep="first")
        .sort(["energyabovegsinpercm", "j", "label"])
        .with_row_index("levelid")
        .select(
            pl.col("energyabovegsinpercm"),
            pl.col("j"),
            levelname=(
                pl.col("label")
                + ",enpercm="
                + pl.col("energyabovegsinpercm").cast(pl.Utf8)
                + ",j="
                + pl.col("j").cast(pl.String)
            ),
            g=2 * pl.col("j") + 1,
        )
        .sort("energyabovegsinpercm", "j")
        .collect()
    )
    dflevels = (
        artisatomic.add_dummy_zero_level(dflevels)
        .with_row_index("levelid")
        .with_columns(pl.col("levelid").cast(pl.Int64))
        .with_columns(parity=-pl.col("levelid"))  # give a unique parity so that all transitions are permitted
    )
    artisatomic.log_and_print(flog, f"Read {len(dflevels) - 1:d} levels")

    transitions = (
        gfall.select(
            [
                "atomic_number",
                "ion_charge",
                "energyabovegsinpercm_lower",
                "j_lower",
                "energyabovegsinpercm_upper",
                "j_upper",
                "wavelength_nm",
                "loggf",
            ]
        )
        .with_columns(gf=10 ** pl.col("loggf"))
        .drop("loggf")
        .join(
            dflevels.lazy().select(
                energyabovegsinpercm_lower=pl.col("energyabovegsinpercm"),
                j_lower=pl.col("j"),
                levelid_lower=pl.col("levelid"),
            ),
            on=["energyabovegsinpercm_lower", "j_lower"],
            how="left",
        )
        .join(
            dflevels.lazy().select(
                energyabovegsinpercm_upper=pl.col("energyabovegsinpercm"),
                j_upper=pl.col("j"),
                levelid_upper=pl.col("levelid"),
            ),
            on=["energyabovegsinpercm_upper", "j_upper"],
            how="left",
        )
        .with_columns(
            # wavelengths are in nanometers, so multiply by 10 to get Angstroms
            A=pl.col("gf") / (1.49919e-16 * (2 * pl.col("j_upper") + 1) * (pl.col("wavelength_nm") * 10.0).pow(2))
        )
        .select(
            upperlevel=pl.col("levelid_upper"),
            lowerlevel=pl.col("levelid_lower"),
            A=pl.col("A"),
            coll_str=pl.lit(-1),
        )
        .collect()
    )

    transition_count_of_level_name = {
        levelname: (
            transitions.select(((pl.col("lowerlevel") == levelid) | (pl.col("upperlevel") == levelid)).sum()).item()
        )
        for levelid, levelname in dflevels.select("levelid", "levelname").iter_rows(named=False)
    }
    artisatomic.log_and_print(flog, f"Read {len(transitions):d} transitions")

    ionization_energy_in_ev = artisatomic.get_nist_ionization_energies_ev()[(atomic_number, ion_stage)]
    artisatomic.log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")

    return ionization_energy_in_ev, dflevels, transitions, transition_count_of_level_name


def get_level_valence_n(levelname: str):
    namesplit = levelname.replace("  ", " ").split(" ")
    if len(namesplit) < 2 or not (part := namesplit[-2]):
        print(f"WARNING: Could not find n in {levelname}. Using n=1")
        return 1

    if part[-1] not in "spdfghijklmnopqr":
        # end of string is a number of electrons in the orbital, not a principal quantum number, so remove it

        if not part[-1].isdigit():
            print(f"WARNING: Could not find n in {levelname}. Using n=1")
            return 1
        part = part.rstrip("0123456789")
    part = part.strip("spdfghijklmnopqr")

    # inefficient way to find the last number in a string
    for i in range(len(part)):
        try:
            n = int(part[i:])
        except ValueError:
            continue
        else:
            assert n >= 0
            assert n < 50
            return n

    print(f"WARNING: Could not find n in {levelname}. Using n=1")
    return 1

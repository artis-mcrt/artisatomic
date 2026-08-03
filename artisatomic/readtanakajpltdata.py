from pathlib import Path

import pandas as pd
import polars as pl

import artisatomic

jpltpath = (Path(__file__).parent.resolve() / ".." / "atomic-data-tanaka-jplt" / "data_v2.1").resolve()


def extend_ion_list(ion_handlers, maxionstage=None):
    tanakaions = sorted(
        [tuple(int(x) for x in f.parts[-1].split(".")[0].split("_")) for f in jpltpath.glob("*_*.txt*")]
    )
    if maxionstage is not None:
        tanakaions = [ion for ion in tanakaions if ion[1] <= maxionstage]

    for atomic_number, ion_stage in tanakaions:
        ion_handlers = artisatomic.add_handler_if_not_set(ion_handlers, atomic_number, ion_stage, "tanakajplt")

    return ion_handlers


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    filename = f"{atomic_number}_{ion_stage}.txt"
    print(f"Reading Tanaka et al. Japan-Lithuania database for Z={atomic_number} ion_stage {ion_stage} from {filename}")
    with artisatomic.xopen_check_extension(jpltpath / filename) as fin:
        for linenumber in range(7):
            readlinein = fin.readline().strip()
            if linenumber < 3:
                artisatomic.log_and_print(flog, readlinein)

            if readlinein == f"# {atomic_number} {ion_stage}":  # search for this line. Header info can be different
                break
        assert readlinein == f"# {atomic_number} {ion_stage}"
        levelcount, transitioncount = (int(x) for x in fin.readline().removeprefix("# ").split())
        artisatomic.log_and_print(flog, f"levels: {levelcount}")
        artisatomic.log_and_print(flog, f"transitions: {transitioncount}")

        fin.readline()
        str_ip_line = fin.readline()
        ionization_energy_in_ev = float(str_ip_line.removeprefix("# IP = "))
        artisatomic.log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")
        assert fin.readline().strip() == "# Energy levels"
        expected_column_headers = ["#", "num", "weight", "parity", "E(eV)", "configuration"]
        read_column_headers = fin.readline().strip().split()  # v2.1 has extra column
        assert all(item in read_column_headers for item in expected_column_headers)

        with pd.read_fwf(
            fin,
            chunksize=levelcount,
            nrows=levelcount,
            colspecs=[(0, 7), (7, 15), (15, 19), (19, 34), (34, 34 + 5000)],
            names=["levelid", "g", "parity", "energy_ev", "configuration"],
        ) as reader:
            hc_in_ev_cm = 0.0001239841984332003

            dflevels = pl.from_pandas(reader.get_chunk(levelcount)).select(
                energyabovegsinpercm=pl.col("energy_ev").cast(pl.Float64) / hc_in_ev_cm,
                parity=pl.when(pl.col("parity").str.strip_chars() == "odd").then(1).otherwise(0),
                g=pl.col("g").cast(pl.Float64),
                levelname=pl.format(
                    "{},{},{}", pl.col("levelid"), pl.col("parity"), pl.col("configuration").str.strip_chars()
                ),
                levelid=pl.col("levelid").cast(pl.Int64),
            )

        assert dflevels.height == levelcount

        line = fin.readline().strip()
        assert line in ("# Transitions", "# num_u   num_l   wavelength(nm)     g_u*A      log(g_l*f)")
        if line == "# Transitions":
            assert fin.readline().strip() == "# num_u   num_l   wavelength(nm)     g_u*A      log(g_l*f)"
        dftransitions = pl.from_pandas(
            pd.read_fwf(
                fin,
                colspecs=[(0, 7), (7, 15), (15, 30), (30, 43), (43, 43 + 5000)],
                names=["upperlevel", "lowerlevel", "wavelength", "g_u_times_A", "log(g_l*f)"],
                dtype_backend="pyarrow",
            )
        ).select(
            pl.col("lowerlevel").cast(pl.Int64),
            pl.col("upperlevel").cast(pl.Int64),
            pl.col("g_u_times_A").cast(pl.Float64),
        )

    transition_count_of_levelid: dict[int, int] = dict(
        pl.concat([dftransitions["lowerlevel"], dftransitions["upperlevel"]]).value_counts().iter_rows()
    )
    transition_count_of_level_name = {
        levelname: transition_count_of_levelid.get(levelid, 0)
        for levelid, levelname in dflevels.select("levelid", "levelname").iter_rows(named=False)
    }
    assert dftransitions.height == transitioncount

    dftransitions = (
        dftransitions.join(
            dflevels.select(g_u=pl.col("g"), upperlevel=pl.col("levelid")),
            on="upperlevel",
            how="left",
            maintain_order="left",
        )
        .with_columns(A=pl.col("g_u_times_A") / pl.col("g_u"))
        .select(["lowerlevel", "upperlevel", "A"])
    )
    dftransitions_filtered = dftransitions.filter(pl.col("lowerlevel") != pl.col("upperlevel"))
    if dftransitions.height != dftransitions_filtered.height:
        artisatomic.log_and_print(flog, "WARNING: dropped rows where upper and lower levels are equal")
        dftransitions = dftransitions_filtered

    return ionization_energy_in_ev, dflevels, dftransitions, transition_count_of_level_name


def get_level_valence_n(levelname: str):
    n = int(levelname.rsplit("  ", maxsplit=1)[-1].split(" ", maxsplit=1)[0].rstrip("spdfg+-"))
    assert n >= 0
    assert n < 20
    return n

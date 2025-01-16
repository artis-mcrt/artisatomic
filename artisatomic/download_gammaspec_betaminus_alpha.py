#!/usr/bin/env python3
import io
import math
from pathlib import Path

import artistools as at
import numpy as np
import polars as pl
import requests

colreplacements = {
    "Rad Int.": "intensity",
    "Rad Ene.": "radiationenergy_kev",
    "Rad subtype": "radsubtype",
    "Par. Elevel": "parent_elevel",
    "Dec Mode": "decaymode",
    "T1/2 (num)": "halflife_s",
}


def process_table(z: int, a: int, outpath: Path, textdata, dfbetaminus: pl.DataFrame, dfalpha: pl.DataFrame):
    textdata = textdata.replace("**********", "0.")
    # print(textdata)
    # match

    startindex = textdata.find("<pre>") + len("<pre>")
    endindex = textdata.rfind("</pre>")
    strtable = textdata[startindex:endindex].strip()
    strheader = strtable.strip().split("\n")[0].strip()
    # print(strheader)
    assert (
        strheader
        == "A  	Element	Z  	N  	Par. Elevel	Unc. 	JPi       	Dec Mode	T1/2 (txt)    	T1/2 (num)       "
        " 	Daughter	Radiation	Rad subtype 	Rad Ene.  	Unc       	EP Ene.   	Unc       	Rad Int.  	Unc      "
        " 	Dose        	Unc"
    )

    dfnuclide = pl.read_csv(io.StringIO(strtable), separator="\t", schema_overrides={"Par. Elevel": pl.Utf8})

    newcols: list[str] = []
    for colname in dfnuclide.columns:
        colname = colname.strip()
        if colname.startswith("Unc"):
            colname = f"{newcols[-1]} (Unc)"
        if colname in colreplacements:
            colname = colreplacements[colname]
        newcols.append(colname)
    dfnuclide.columns = newcols
    dfnuclide = dfnuclide.with_columns(pl.col(pl.Utf8).str.strip_chars()).with_columns(
        pl.col("parent_elevel").str.replace("0.0", "0"),
        pl.col("radiationenergy_kev").cast(pl.Float64),
        pl.col("intensity").cast(pl.Float64),
        pl.col("halflife_s").cast(pl.Float64),
    )

    found_groundlevel = False
    for (parelevel,), dfdecay in dfnuclide.group_by("parent_elevel"):
        assert isinstance(parelevel, str)
        try:
            is_groundlevel = float(parelevel) == 0.0
        except ValueError:
            is_groundlevel = False
        print(f"  parent_Elevel: {parelevel} is_groundlevel: {is_groundlevel}")
        if not is_groundlevel:
            continue

        found_groundlevel = True

        dfgammadecays = dfdecay.filter(
            (pl.col("Radiation") == "G") & pl.col("radsubtype").is_in(["", "Annihil."]) & (pl.col("intensity") > 0.0)
        )

        maybedfbetaminusrow = dfbetaminus.filter(pl.col("Z") == z).filter(pl.col("A") == a)
        maybedfalpharow = dfalpha.filter(pl.col("Z") == z).filter(pl.col("A") == a).limit(1)
        nndc_halflife = None
        if not dfgammadecays.is_empty():
            nndc_halflife = dfgammadecays["halflife_s"].item(0)
            print(f"                     NNDC half-life: {nndc_halflife:7.1e} s")

        if maybedfbetaminusrow.height > 0:
            halflife = maybedfbetaminusrow["tau[s]"].item() * math.log(2)
            strwarn = (
                " WARNING!!!!!!"
                if (nndc_halflife is not None and not np.isclose(nndc_halflife, halflife, rtol=0.1))
                else ""
            )
            print(f"      betaminusdecays.txt half-life: {halflife:7.1e} s {strwarn}")

        if maybedfalpharow.height > 0:
            halflife = maybedfalpharow["halflife_s"].item()
            strwarn = (
                " WARNING!!!!!!"
                if (nndc_halflife is not None and not np.isclose(nndc_halflife, halflife, rtol=0.1))
                else ""
            )
            print(f"          alphadecays.txt half-life: {halflife:7.1e} s {strwarn}")

        e_gamma = (dfgammadecays["radiationenergy_kev"] * dfgammadecays["intensity"] / 100.0).sum()
        print(f"                   NNDC Egamma: {e_gamma:7.1f} keV")

        if maybedfbetaminusrow.height > 0:
            file_e_gamma = maybedfbetaminusrow["E_gamma[MeV]"].item() * 1000
            strwarn = "" if np.isclose(e_gamma, file_e_gamma, rtol=0.1) else " WARNING!!!!!!"
            print(f"    betaminusdecays.txt Egamma: {file_e_gamma:7.1f} keV {strwarn}")

        elif maybedfalpharow.height > 0:
            file_e_gamma = maybedfalpharow["E_gamma[MeV]"].item() * 1000
            strwarn = "" if np.isclose(e_gamma, file_e_gamma, rtol=0.1) else " WARNING!!!!!!"
            print(f"        alphadecays.txt Egamma: {file_e_gamma:7.1f} keV {strwarn}")

        dfout = pl.DataFrame(
            {
                "energy_mev": dfgammadecays["radiationenergy_kev"] / 1000.0,
                "intensity": dfgammadecays["intensity"] / 100.0,
            }
        ).sort("energy_mev")
        if len(dfout) > 0:
            with outpath.open("w", encoding="utf-8") as fout:
                fout.write(f"{len(dfout)}\n")
                for energy_mev, intensity in dfout[["energy_mev", "intensity"]].iter_rows():
                    fout.write(f"{energy_mev:5.3f}  {intensity:6.4f}\n")

                print(f"Saved {outpath.name}")
        else:
            print("empty DataFrame")
    if not found_groundlevel:
        print("  ERROR! did not find ground level")


def main():
    elsymbols = at.get_elsymbolslist()

    outfolder = Path(__file__).parent.parent.absolute() / "artis_files" / "data"
    outfolder.mkdir(parents=True, exist_ok=True)

    dfbetaminus = (
        pl.read_csv(
            at.get_config()["path_datadir"] / "betaminusdecays.txt",
            separator=" ",
            comment_prefix="#",
            has_header=False,
            new_columns=["A", "Z", "Q[MeV]", "E_gamma[MeV]", "E_elec[MeV]", "E_neutrino[MeV]", "tau[s]"],
        )
        .filter(pl.col("Q[MeV]") > 0.0)
        .filter(pl.col("tau[s]") > 0.0)
    )

    assert dfbetaminus.height == dfbetaminus.unique(("Z", "A")).height

    dfalpha = (
        pl.read_csv(
            at.get_config()["path_datadir"] / "alphadecays.txt",
            separator=" ",
            comment_prefix="#",
            has_header=False,
            new_columns=[
                "A",
                "Z",
                "branch_alpha",
                "branch_beta",
                "halflife_s",
                "Q_total_alphadec[MeV]",
                "Q_total_betadec[MeV]",
                "E_alpha[MeV]",
                "E_gamma[MeV]",
                "E_beta[MeV]",
            ],
        )
        .filter(pl.col("halflife_s") > 0.0)
        .unique(("Z", "A"), keep="last", maintain_order=True)
    )
    assert dfalpha.height == dfalpha.unique(("Z", "A")).height

    nuclist = sorted(list(dfbetaminus.select(["Z", "A"]).iter_rows()) + list(dfalpha.select(["Z", "A"]).iter_rows()))

    for z, a in nuclist:
        strnuclide = elsymbols[z].lower() + str(a)
        filename = f"{strnuclide}_lines.txt"
        outpath = outfolder / filename
        # if outpath.is_file():
        #     # print(f"  {filename} already exists. skipping...")
        #     continue
        print(f"\n(Z={z}) {strnuclide}")

        url = f"https://www.nndc.bnl.gov/nudat3/decaysearchdirect.jsp?nuc={strnuclide}&unc=standard&out=file"

        with requests.Session() as s:
            textdata = s.get(url).text
            if "<pre>" not in textdata:
                print(f"  no table data returned from {url}")
            else:
                process_table(
                    z=z,
                    a=a,
                    outpath=outpath,
                    textdata=textdata,
                    dfbetaminus=dfbetaminus,
                    dfalpha=dfalpha,
                )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download ENDF decay data and write the gamma spectra that ARTIS uses."""

import io
import math

import artistools as at
import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from artisatomic.base import elsymbols
from artisatomic.base import PYDIR

colreplacements = {
    "Rad Int.": "intensity",
    "Rad Ene.": "radiationenergy_kev",
    "Rad subtype": "radsubtype",
    "Par. Elevel": "parent_elevel",
    "Dec Mode": "decaymode",
    "T1/2 (num)": "halflife_s",
}


def main():
    """Fetch a NuDat3 decay table for every nuclide in betaminusdecays.txt and alphadecays.txt.

    The script writes the gamma lines of each nuclide to artis_files/data/gamma_<nuclide>.txt.
    """
    outfolder = PYDIR.parent / "artis_files" / "data"
    outfolder.mkdir(parents=True, exist_ok=True)

    dfbetaminus = (
        pl.read_csv(
            at.get_path("datadir") / "betaminusdecays.txt",
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
            at.get_path("datadir") / "alphadecays.txt",
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

    # a set: the script downloads a nuclide in both lists once
    nuclist = sorted(set(dfbetaminus.select(["Z", "A"]).iter_rows()) | set(dfalpha.select(["Z", "A"]).iter_rows()))
    # one session for the whole run, and a retry on a transient failure. Then one 5xx or a
    # dropped connection does not end the run after N nuclides.
    nuclides_without_table: list[str] = []
    with requests.Session() as session:
        session.mount(
            "https://",
            HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))),
        )
        for z, a in nuclist:
            strnuclide = elsymbols[z].lower() + str(a)
            nucoutfilepath = outfolder / f"gamma_{strnuclide}.txt"
            print(f"\n(Z={z}) {strnuclide}")

            url = f"https://www.nndc.bnl.gov/nudat3/decaysearchdirect.jsp?nuc={strnuclide}&unc=standard&out=file"
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
            }
            # a timeout, so a stalled connection stops the run and does not hold it forever. A status
            # check, so the script does not parse an error page as a table.
            with session.get(url, headers=headers, timeout=120) as response:
                response.raise_for_status()
                textdata = response.text
                if "<pre>" not in textdata:
                    print(f"  NuDat returned no table data from {url}")
                    nuclides_without_table.append(strnuclide)
                    continue

                textdata = textdata.replace("**********", "0.")
                # match

                startindex = textdata.find("<pre>") + len("<pre>")
                endindex = textdata.rfind("</pre>")
                strtable = textdata[startindex:endindex].strip()
                strheader = strtable.strip().split("\n")[0].strip()
                assert (
                    strheader
                    == "A  	Element	Z  	N  	Par. Elevel	Unc. 	JPi       	Dec Mode	T1/2 (txt)    	T1/2 (num)       "
                    " 	Daughter	Radiation	Rad subtype 	Rad Ene.  	Unc       	EP Ene.   	Unc       	Rad Int.  	Unc      "
                    " 	Dose        	Unc"
                )

                dfnuclide = pl.read_csv(
                    io.StringIO(strtable), separator="\t", schema_overrides={"Par. Elevel": pl.Utf8}
                )

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
                    # every spelling of a zero level ("0", "0.0", "0.00") becomes one value. Then the
                    # lines of the ground level form one group below, and not two groups that each
                    # rewrite the file. A blank or non-numeric level stays as it is.
                    pl.when(pl.col("parent_elevel").cast(pl.Float64, strict=False) == 0.0)
                    .then(pl.lit("0"))
                    .otherwise(pl.col("parent_elevel"))
                    .alias("parent_elevel"),
                    pl.col("radiationenergy_kev").cast(pl.Float64),
                    pl.col("intensity").cast(pl.Float64),
                    pl.col("halflife_s").cast(pl.Float64),
                )

                found_groundlevel = False
                for (parelevel,), dfdecay in dfnuclide.group_by("parent_elevel"):
                    # a blank parent level reads as null, and a level that is not a number is not
                    # the ground level either
                    try:
                        is_groundlevel = parelevel is not None and float(parelevel) == 0.0
                    except ValueError:
                        is_groundlevel = False
                    print(f"  parent_Elevel: {parelevel} is_groundlevel: {is_groundlevel}")
                    if not is_groundlevel:
                        continue

                    found_groundlevel = True

                    dfgammadecays = dfdecay.filter(
                        (pl.col("Radiation") == "G")
                        # fill_null: an empty cell reads as null, and is_in() on a null drops the row
                        & pl.col("radsubtype").fill_null("").is_in(["", "Annihil."])
                        & (pl.col("intensity") > 0.0)
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
                            if (nndc_halflife is not None and not math.isclose(nndc_halflife, halflife, rel_tol=0.1))
                            else ""
                        )
                        print(f"      betaminusdecays.txt half-life: {halflife:7.1e} s {strwarn}")

                    if maybedfalpharow.height > 0:
                        halflife = maybedfalpharow["halflife_s"].item()
                        strwarn = (
                            " WARNING!!!!!!"
                            if (nndc_halflife is not None and not math.isclose(nndc_halflife, halflife, rel_tol=0.1))
                            else ""
                        )
                        print(f"          alphadecays.txt half-life: {halflife:7.1e} s {strwarn}")

                    e_gamma = (dfgammadecays["radiationenergy_kev"] * dfgammadecays["intensity"] / 100.0).sum()
                    print(f"                   NNDC Egamma: {e_gamma:7.1f} keV")

                    if maybedfbetaminusrow.height > 0:
                        file_e_gamma = maybedfbetaminusrow["E_gamma[MeV]"].item() * 1000
                        strwarn = "" if math.isclose(e_gamma, file_e_gamma, rel_tol=0.1) else " WARNING!!!!!!"
                        print(f"    betaminusdecays.txt Egamma: {file_e_gamma:7.1f} keV {strwarn}")

                    elif maybedfalpharow.height > 0:
                        file_e_gamma = maybedfalpharow["E_gamma[MeV]"].item() * 1000
                        strwarn = "" if math.isclose(e_gamma, file_e_gamma, rel_tol=0.1) else " WARNING!!!!!!"
                        print(f"        alphadecays.txt Egamma: {file_e_gamma:7.1f} keV {strwarn}")

                    dfout = pl.DataFrame(
                        {
                            "energy_mev": dfgammadecays["radiationenergy_kev"] / 1000.0,
                            "intensity": dfgammadecays["intensity"] / 100.0,
                        }
                    ).sort("energy_mev")
                    if len(dfout) > 0:
                        # write to a temporary name and then rename. A run that stops part way
                        # through then leaves the previous file complete and not truncated.
                        tmpoutfilepath = nucoutfilepath.with_suffix(".tmp")
                        with tmpoutfilepath.open("w", encoding="utf-8") as fout:
                            fout.write(f"{len(dfout)}\n")
                            for energy_mev, intensity in dfout[["energy_mev", "intensity"]].iter_rows():
                                fout.write(f"{energy_mev:5.3f}  {intensity:6.4f}\n")
                        tmpoutfilepath.replace(nucoutfilepath)
                        print(f"Saved {nucoutfilepath.name}")
                    else:
                        print("empty DataFrame")
                if not found_groundlevel:
                    print("  ERROR! the table has no ground level")

    # a run that skipped a nuclide must not end with an exit code of 0 and a missing file
    if nuclides_without_table:
        msg = f"NuDat returned no table for {len(nuclides_without_table)} nuclides: {', '.join(nuclides_without_table)}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()

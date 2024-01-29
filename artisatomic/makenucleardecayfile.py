#!/usr/bin/env python3
import io
import sys
from pathlib import Path

import artistools as at
import pandas as pd
import requests

# import numpy as np


def main():
    PYDIR = Path(__file__).parent.resolve()
    atomicdata = pd.read_csv(PYDIR / "atomic_properties.txt", sep=r"\s+", comment="#")
    elsymbols = ["n", *list(atomicdata["symbol"].values)]

    outfolder = Path(__file__).parent.absolute() / "artis_files" / "data"
    outfolder.mkdir(parents=True, exist_ok=True)
    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 999

    nuclist = [
        # (27, 56),
        # (27, 57),
        # (28, 56),
        # (28, 57),
        # (30, 71),
        # (92, 238)
        # (47, 104),
    ]

    dfmodel, _, _ = at.inputmodel.get_modeldata(
        "/Volumes/GoogleDrive/My"
        " Drive/Archive/Mergers/SFHo_long/SFHo_long_snapshot/artismodel_SFHo_long-radius-entropy_0p05d"
    )
    nuclist = [at.get_z_a_nucname(c) for c in dfmodel.columns if c.startswith("X_") and not c.endswith("Fegroup")]

    colreplacements = {
        "Rad Int.": "intensity",
        "Rad Ene.": "radiationenergy_kev",
        "Rad subtype": "radsubtype",
        "Par. Elevel": "parent_elevel",
    }

    for z, a in nuclist:
        strnuclide = elsymbols[z].lower() + str(a)
        print(f"\n(Z={z}) {strnuclide}")
        outpath = outfolder / f"{strnuclide}_lines.txt"
        if outpath.is_file():
            print(f"  {outpath} already exists. skipping...")
            continue

        url = f"https://www.nndc.bnl.gov/nudat3/decaysearchdirect.jsp?nuc={strnuclide}&unc=standard&out=file"

        with requests.Session() as s:
            textdata = s.get(url).text
            textdata = textdata.replace("**********", "0.")
            # print(textdata)
            # match
            if "<pre>" not in textdata:
                print(f"  no table data returned from {url}")
                continue

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
            # dfnuclide = pd.read_fwf(io.StringIO(strtable))
            dfnuclide = pd.read_csv(io.StringIO(strtable), delimiter="\t", dtype={"Par. Elevel": str})
            newcols = []
            for index, colname in enumerate(dfnuclide.columns):
                colname = colname.strip()
                if colname.startswith("Unc"):
                    colname = newcols[-1] + " (Unc)"
                if colname in colreplacements:
                    colname = colreplacements[colname]
                newcols.append(colname)
            dfnuclide.columns = newcols
            dfnuclide["Dec Mode"] = dfnuclide["Dec Mode"].str.strip()
            dfnuclide["Radiation"] = dfnuclide["Radiation"].str.strip()
            dfnuclide["radsubtype"] = dfnuclide["radsubtype"].str.strip()
            dfnuclide["parent_elevel"] = dfnuclide["parent_elevel"].str.strip()

            found_groundlevel = False
            for parelevel, dfdecay in dfnuclide.groupby("parent_elevel"):
                try:
                    is_groundlevel = float(parelevel) == 0.0
                except ValueError:
                    is_groundlevel = False
                print(f"  parent_Elevel: {parelevel} is_groundlevel: {is_groundlevel}")
                if not is_groundlevel:
                    continue
                else:
                    found_groundlevel = True
                dfgammadecays = dfnuclide.query(
                    "Radiation == 'G' and (radsubtype == '' or radsubtype == 'Annihil.') and intensity >= 0.15",
                    inplace=False,
                )
                # print(dfgammadecays)

                dfout = pd.DataFrame()
                dfout["energy_mev"] = dfgammadecays.radiationenergy_kev.values / 1000.0
                dfout["intensity"] = dfgammadecays.intensity.values / 100.0

                # if positrons are emitted, there are two 511 keV gamma rays per positron decay
                # dfannihil = dfnuclide.query("Radiation == 'G' and radsubtype == 'Annihil.'", inplace=False)
                # posbranchfrac = 0. if dfannihil.empty else dfannihil.intensity.sum() / 100. / 2.
                # endecay_positrons_mev = 0.
                # dfrad_e = dfnuclide.query("Radiation == 'BP' or Radiation == 'E'")
                # endecay_positrons_mev = (dfrad_e.radiationenergy_kev * dfrad_e.intensity).sum() / 100.

                dfout.sort_values(by="energy_mev", ascending=True, inplace=True, ignore_index=True)

                # combine identical energy gamma ray intensities
                # aggregation_functions = {'energy_mev': 'first', 'intensity': 'sum'}
                # dfout = dfout.groupby(dfout['energy_mev']).aggregate(aggregation_functions)

                # print(dfout)
                if len(dfout) > 0:
                    with open(outpath, "w") as fout:
                        # fout.write(f'{len(dfout)}  {posbranchfrac:.3f}  {endecay_positrons_mev:.3f}\n')
                        fout.write(f"{len(dfout)}\n")
                        for _, row in dfout.iterrows():
                            fout.write(f"{row.energy_mev:5.3f}  {row.intensity:6.4f}\n")
                        print(f"Saved {outpath}")
                else:
                    print("empty DataFrame")
            if not found_groundlevel:
                print("  ERROR! did not find ground level")


if __name__ == "__main__":
    main()

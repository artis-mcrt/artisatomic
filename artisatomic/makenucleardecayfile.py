#!/usr/bin/env python3
import io
import requests
import sys
from pathlib import Path
# import numpy as np
import pandas as pd
import artistools as at


def main():
    PYDIR = Path(__file__).parent.resolve()
    atomicdata = pd.read_csv(PYDIR / 'atomic_properties.txt', delim_whitespace=True, comment='#')
    elsymbols = ['n'] + list(atomicdata['symbol'].values)

    outfolder = Path(__file__).parent.absolute() / 'artis_files' / 'data'
    outfolder.mkdir(parents=True, exist_ok=True)
    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 999

    nuclist = [
        (27, 56),
        # (27, 57),
        # (28, 56),
        # (28, 57),
        # (30, 71),
        # (92, 238)
    ]

    colreplacements = {
        'Rad Int.': 'intensity',
        'Rad Ene.': 'radiationenergy_kev',
        'Rad subtype': 'radsubtype',
    }

    for z, a in nuclist:
        strnuclide = elsymbols[z].lower() + str(a)

        url = f'https://www.nndc.bnl.gov/nudat3/decaysearchdirect.jsp?nuc={strnuclide}&unc=standard&out=file'

        outpath = outfolder / f'{strnuclide}_lines.txt'
        print(f'Writing {outpath}')
        with open(outpath, 'wt') as fout, requests.Session() as s:
            textdata = s.get(url).text
            # print(textdata)
            # match
            if '<pre>' not in textdata:
                print('ERROR!')
                sys.exit(1)
            else:
                startindex = textdata.find('<pre>') + len('<pre>')
                endindex = textdata.rfind('</pre>')
                strtable = textdata[startindex:endindex].strip()
                strheader = strtable.strip().split('\n')[0].strip()
                # print(strheader)
                assert strheader == 'A  	Element	Z  	N  	Par. Elevel	Unc. 	JPi       	Dec Mode	T1/2 (txt)    	T1/2 (num)        	Daughter	Radiation	Rad subtype 	Rad Ene.  	Unc       	EP Ene.   	Unc       	Rad Int.  	Unc       	Dose        	Unc'
                # dfnuclide = pd.read_fwf(io.StringIO(strtable))
                dfnuclide = pd.read_csv(io.StringIO(strtable), delimiter='\t')
                newcols = []
                for index, colname in enumerate(dfnuclide.columns):
                    colname = colname.strip()
                    if colname.startswith('Unc'):
                        colname = newcols[-1] + ' (Unc)'
                    if colname in colreplacements:
                        colname = colreplacements[colname]
                    newcols.append(colname)
                dfnuclide.columns = newcols
                dfnuclide['Dec Mode'] = dfnuclide['Dec Mode'].str.strip()
                dfnuclide['Radiation'] = dfnuclide['Radiation'].str.strip()
                dfnuclide['radsubtype'] = dfnuclide['radsubtype'].str.strip()
                dfnuclide.query("`Par. Elevel` == 0")
                print(dfnuclide)

                dfgammadecays = dfnuclide.query("Radiation == 'G' and (radsubtype == '' or radsubtype == 'Annihil.') and intensity >= 0.15", inplace=False)
                print(dfgammadecays)

                dfout = pd.DataFrame()
                dfout['energy_mev'] = dfgammadecays.radiationenergy_kev.values / 1000.
                dfout['intensity'] = dfgammadecays.intensity.values / 100.

                # if positrons are emitted, there are two 511 keV gamma rays per positron decay
                # dfannihil = dfnuclide.query("Radiation == 'G' and radsubtype == 'Annihil.'", inplace=False)
                # posbranchfrac = 0. if dfannihil.empty else dfannihil.intensity.sum() / 100. / 2.
                # endecay_positrons_mev = 0.
                # dfrad_e = dfnuclide.query("Radiation == 'BP' or Radiation == 'E'")
                # endecay_positrons_mev = (dfrad_e.radiationenergy_kev * dfrad_e.intensity).sum() / 100.

                dfout.sort_values(by='energy_mev', ascending=True, inplace=True, ignore_index=True)

                # combine identical energy gamma ray intensities
                # aggregation_functions = {'energy_mev': 'first', 'intensity': 'sum'}
                # dfout = dfout.groupby(dfout['energy_mev']).aggregate(aggregation_functions)

                # print(dfout)
                # fout.write(f'{len(dfout)}  {posbranchfrac:.3f}  {endecay_positrons_mev:.3f}\n')
                fout.write(f'{len(dfout)}\n')
                for _, row in dfout.iterrows():
                    fout.write(f'{row.energy_mev:5.3f}  {row.intensity:6.4f}\n')
                print(f'Saved {outpath}')


if __name__ == "__main__":
    main()

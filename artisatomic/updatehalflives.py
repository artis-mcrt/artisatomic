#!/usr/bin/env python3
import io
import math
import requests

from pathlib import Path
import numpy as np
import pandas as pd

import artisatomic


def main():
    outfolder = Path(__file__).parent.absolute() / 'artis_files' / 'data'
    outfolder.mkdir(parents=True, exist_ok=True)
    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 999

    dfbetaminus = pd.read_csv(outfolder / 'betaminusdecays.txt',
                              delim_whitespace=True, comment='#',
                              names=['A', 'Z', 'Q[MeV]', 'Egamma[MeV]', 'Eelec[MeV]',
                                     'Eneutrino[MeV]', 'tau[s]'])

    for index, row in dfbetaminus.query('Z == 83').iterrows():
        z = int(row['Z'])
        a = int(row['A'])
        strnuclide = artisatomic.elsymbols[z].lower() + str(a)
        print(f'\n(Z={z}) {strnuclide}')

        url = f'https://www.nndc.bnl.gov/nudat3/decaysearchdirect.jsp?nuc={strnuclide}&unc=standard&out=file'

        with requests.Session() as s:
            textdata = s.get(url).text
            # print(textdata)
            # match
            if '<pre>' in textdata:
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
                    newcols.append(colname)
                dfnuclide.columns = newcols
                dfnuclide['Dec Mode'] = dfnuclide['Dec Mode'].str.strip()
                dfnuclide['Radiation'] = dfnuclide['Radiation'].str.strip()
                dfnuclide['Rad subtype'] = dfnuclide['Rad subtype'].str.strip()
                dfnuclide.query("`Par. Elevel` == 0")
                if not dfnuclide.empty:
                    print('  file half-life: ', row['tau[s]'] * math.log(2))
                    print('  NNDC half-life: ', dfnuclide.iloc[0]['T1/2 (num)'])
                # dfbetaminus = dfnuclide.query("`Dec Mode` == 'B-'")
                # if not dfbetaminus.empty:
                    print('  file Eelec: ', row['Eelec[MeV]'] * 1000, 'keV')
                    dfrad_e = dfnuclide.query("Radiation == 'BM' or Radiation == 'E'")
                    e_elec = (dfrad_e['Rad Ene.'] * dfrad_e['Rad Int.']).sum() / 100.
                    print('  NNDC Eelec: ', e_elec, 'keV')

                    print('  file Egamma: ', row['Egamma[MeV]'] * 1000, 'keV')
                    dfrad_g = dfnuclide.query("Radiation == 'G' and `Rad subtype` == ''")
                    e_gamma = (dfrad_g['Rad Ene.'] * dfrad_g['Rad Int.']).sum() / 100.
                    print('  NNDC Egamma: ', e_gamma, 'keV')

                    # print('  file Ealpha: ', row['Eelec[MeV]'] * 1000, 'keV')
                    dfrad_a = dfnuclide.query("Radiation == 'A' and `Rad subtype` == ''")
                    e_alpha = (dfrad_a['Rad Ene.'] * dfrad_a['Rad Int.']).sum() / 100.
                    print('  NNDC Ealpha: ', e_alpha, 'keV')
                # print(dfnuclide)
            else:
                pass
                # print('no data returned')


if __name__ == "__main__":
    main()

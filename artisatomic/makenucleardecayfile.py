#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

import artisatomic

from nuclear.io.nndc import download_decay_radiation, store_decay_radiation, get_decay_radiation_database


def main():
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

    for z, a in nuclist:
        strnuclide = artisatomic.elsymbols[z].lower() + str(a)

        print(outfolder / f'{strnuclide}_lines.txt')
        with open(outfolder / f'{strnuclide}_lines.txt', 'wt') as fout:
            download_needed = False
            try:
                decay_rad_db, meta = get_decay_radiation_database()
                if decay_rad_db.query('isotope == @strnuclide').empty:
                    download_needed = True
            except FileNotFoundError:
                download_needed = True
                pass
            if download_needed:
                nuc, nuc_meta = download_decay_radiation(strnuclide)
                print(nuc)
                store_decay_radiation(strnuclide, force_update=True)
                decay_rad_db, meta = get_decay_radiation_database()

            assert meta[meta['key'] == 'energy_column_unit']['value'][0] == 'keV'
            assert meta[meta['key'] == 'intensity_column_unit']['value'][0] == '%'

            dfgammadecays = decay_rad_db.query("type=='gamma_rays' and intensity > .1", inplace=False)

            dfout = pd.DataFrame()
            dfout['energy_mev'] = dfgammadecays.energy.values / 1000.
            dfout['intensity'] = dfgammadecays.intensity.values / 100.

            # if positrons are emitted, add two 511 keV gamma rays per positron decay
            print(decay_rad_db)
            print(meta)
            dfpos = decay_rad_db.query('type == "e+"', inplace=False)
            posbranchfrac = dfpos.intensity.sum() / 100.
            endecay_positrons_mev = 0.
            if posbranchfrac > .01:
                dfout = dfout.append(pd.DataFrame({'energy_mev': [0.511], 'intensity': [posbranchfrac * 2]}))
                endecay_positrons_mev = np.dot(dfpos.intensity.values / 100., dfpos['energy'].values)

            dfout.sort_values(by='energy_mev', ascending=True, inplace=True, ignore_index=True)

            # combine identical energy gamma ray intensities
            # aggregation_functions = {'energy_mev': 'first', 'intensity': 'sum'}
            # dfout = dfout.groupby(dfout['energy_mev']).aggregate(aggregation_functions)

            # print(dfout)
            # fout.write(f'{len(dfout)}  {posbranchfrac:.3f}  {endecay_positrons_mev:.3f}\n')
            fout.write(f'{len(dfout)}\n')
            for _, row in dfout.iterrows():
                fout.write(f'{row.energy_mev:6.4f}  {row.intensity:7.5f}\n')


if __name__ == "__main__":
    main()

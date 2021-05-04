#!/usr/bin/env python3

import numpy as np
import pandas as pd

from nuclear.io.nndc import download_decay_radiation, store_decay_radiation, get_decay_radiation_database


def main():
    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 999

    nuclide = 'ni56'

    decay_rad_db, meta = get_decay_radiation_database()
    if decay_rad_db.query('isotope == @nuclide').empty:
        nuc, nuc_meta = download_decay_radiation(nuclide)
        store_decay_radiation(nuclide)
        decay_rad_db, meta = get_decay_radiation_database()

    assert meta[meta['key'] == 'energy_column_unit']['value'][0] == 'keV'
    assert meta[meta['key'] == 'intensity_column_unit']['value'][0] == '%'

    dfgammadecays = decay_rad_db.query("type=='gamma_rays' and intensity > .1")

    dfout = pd.DataFrame()
    dfout['energy_mev'] = dfgammadecays.energy.values / 1000.
    dfout['intensity'] = dfgammadecays.intensity.values / 100.

    # if positrons are emitted, add two 511 keV gamma rays per positron decay
    dfpos = decay_rad_db.query('type == "e+"')
    posbranchfrac = dfpos.intensity.sum() / 100.
    endecay_positrons_mev = 0.
    if posbranchfrac > .01:
        dfout = dfout.append(pd.DataFrame({'energy_mev': [0.511], 'intensity': [posbranchfrac * 2]}))
        endecay_positrons_mev = np.dot(dfpos.intensity.values / 100., dfpos['energy'].values)

    dfout.sort_values(by='energy_mev', ascending=True, inplace=True, ignore_index=True)

    # combine identical energy gamma ray intensities
    # aggregation_functions = {'energy_mev': 'first', 'intensity': 'sum'}
    # dfout = dfout.groupby(dfout['energy_mev']).aggregate(aggregation_functions)

    print(dfout)
    with open(nuclide + '_lines.txt', 'wt') as fout:
        fout.write(f'{len(dfout)}  {posbranchfrac:.3f}  {endecay_positrons_mev:.3f}\n')
        for _, row in dfout.iterrows():
            fout.write(f'{row.energy_mev:6.4f}  {row.intensity:7.5f}\n')


if __name__ == "__main__":
    main()

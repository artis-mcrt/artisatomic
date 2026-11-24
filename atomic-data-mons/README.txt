The University of Mons data set covers La to Lu in ion stages V to VII.
The script setup_mons_data.sh downloads two archives. Each archive contains one file for each of the 45 ions.
The data record is at https://doi.org/10.5281/zenodo.10635803.

The Zenodo data record gives these columns for the transition archive outggf_Ln_V--VII.zip.

1. Wavelength of the electric dipole transition in angstroms.
2. Energy of the lower level in units of 1000 cm^-1.
3. Oscillator strength.

The reader interprets column 3 as the weighted oscillator strength g_l f_lu.
Here f_lu is the absorption oscillator strength and g_l is the statistical weight of the lower level.

The level archive outglv_Ln_V--VII.zip has two columns.

1. Level energy in units of 1000 cm^-1.
2. Total angular momentum J.

The test_sample/ folder contains a sample of both archives. See tests/README.md for the checksum test.

Reference

Carvajal Gallego, H., Deprince, J., Maison, L., Palmeri, P., and Quinet, P. (2024).
"Overview of the contributions from all lanthanide elements to kilonova opacity in the temperature range
from 25 000 to 40 000 K". Astronomy & Astrophysics, 685, A91, pp. 1-16. https://doi.org/10.1051/0004-6361/202347723

Save the helium data set of Aoife Boyle as aoife.hdf5 in this folder. Then convert it to Parquet
from the root of the repository:

    artisatomic/hdf5_to_parquet.py atomic-data-helium-boyle/aoife.hdf5

The script writes one Parquet file for each table into the folder aoife. Run the script again after
you replace aoife.hdf5. The boyle handler compares the size of aoife.hdf5 with the size that each
Parquet file records. It stops if the two sizes are different.

The boyle handler reads He I, He II and He III from the Parquet files. He III is a bare nucleus. The
levels_data table has one level for it, and the ionization_data table has no row for it. No built-in
ion selection uses the handler. Name it in artisatomicionhandlers.json to use it.

The data set belongs to the helium work of Boyle, A., Sim, S. A., Hachinger, S., Kerzendorf, W.
(2017), A&A, 599, A46, doi:10.1051/0004-6361/201629712.

link:
https://qubstudentcloud-my.sharepoint.com/:u:/g/personal/40102503_ads_qub_ac_uk/EXTe1IioIQ9Fuan1lUYl3iIBHQG-6u3ThvEo6zBq7L2qPw?e=P8PtTJ

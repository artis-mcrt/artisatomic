DREAM, the Database on Rare Earths At Mons university: <https://hosting.umons.ac.be/html/agif/databases/dream.html>.
Paper: Quinet, P., Palmeri, P. (2020), Atoms, 8, 18, doi:10.3390/atoms8020018.

Make the HDF5 file with <https://github.com/afloers/DREAM_atomic_data>. Save it as
`DREAM_atomic_data_20241106-1325.h5` in this folder. Then convert it to Parquet from the root of the
repository:

```sh
artisatomic/hdf5_to_parquet.py atomic-data-dream/DREAM_atomic_data_20241106-1325.h5
```

The script writes `DREAM_atomic_data_20241106-1325/atomic_data.parquet`, and `readdreamdata.py` reads
that file. Run the script again after you replace the HDF5 file. The reader compares the size of the
HDF5 file with the size that the Parquet file records. It stops if the two sizes are different.

DREAM, the Database on Rare Earths At Mons university: <https://hosting.umons.ac.be/html/agif/databases/dream.html>.
Paper: Quinet, P., Palmeri, P. (2020), Atoms, 8, 18, doi:10.3390/atoms8020018.

Generate the h5 file with <https://github.com/afloers/DREAM_atomic_data>. Save it as
`DREAM_atomic_data_20241106-1325.h5` in this directory. Then convert it to Parquet once:

```sh
artisatomic/hdf5_to_parquet.py atomic-data-dream/DREAM_atomic_data_20241106-1325.h5
```

The script writes `DREAM_atomic_data_20241106-1325/atomic_data.parquet`. `readdreamdata.py` reads
that file. On Python 3.15, h5py needs an HDF5 library on the computer, because h5py 3.16 has no
wheel for Python 3.15. If it has none, run the script with `uv run --python 3.14`.

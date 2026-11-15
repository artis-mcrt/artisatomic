# Commands

The package installs four console scripts. This page gives the help text of each script that has an
argument parser.

## makeartisatomicfiles

`makeartisatomicfiles` writes adata.txt, compositiondata.txt, phixsdata_v2.txt and transitiondata.txt.
The command `python -m artisatomic` starts the same program.

```{program-output} makeartisatomicfiles --help
```

## makeartisrecombratefile

```{program-output} makeartisrecombratefile --help
```

## makeartischargetransferfile

```{program-output} makeartischargetransferfile --help
```

The command fits the tables of Sterling & Stancil (2011), A&A, 535, A117, doi:10.1051/0004-6361/201117584.
It stops if the largest relative error of a fit, or of a flat value, is above `-maxfiterror`.
The message names each such reaction with its error, and the command writes no file.
The error is relative to the source points that the fit used.
It does not measure the accuracy of the source data.

Seven fits of the tracked source files are above the default limit of 0.25, so the default command stops.
Examine the errors before you give a larger limit.
`-maxfiterror inf` accepts each error.
The header of chargetransfer.txt records the limit of the run, and the comment of each line gives its error.

## makeartisgammaspecfiles

`makeartisgammaspecfiles` downloads the NuDat3 decay tables and writes a gamma spectrum for each nuclide that has gamma lines.

```{program-output} makeartisgammaspecfiles --help
```

## hdf5_to_parquet.py

`artisatomic/hdf5_to_parquet.py` is a script and not a console script. It converts an HDF5 file to
Parquet files for the DREAM and AOIFE readers. The docstring at the top of the script explains the
layouts that it reads.

```{program-output} python -m artisatomic.hdf5_to_parquet --help
```

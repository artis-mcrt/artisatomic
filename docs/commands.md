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

The command rejects a fit whose maximum relative error exceeds `-maxfiterror`.
The default is `0.25`, which uses the earlier warning limit of 25%.
This limit controls acceptance of the fit. It does not measure the accuracy of the source data.
The error uses the same source points as the fit. The check also applies to a flat approximation.

The current source data contain fits above this limit, so the default command stops before it writes the output.
The error names each affected reaction and gives its maximum relative error.
Review those errors before you increase the limit or use `--allow-large-fit-error`.
The output header records the selected limit and the override.
The override keeps the existing coefficients. It does not improve the fits.

A different fit model or separate temperature intervals need a separate scientific review.
The present output format has one fit for each reaction, so this command does not select another model automatically.

## makeartisgammaspecfiles

`makeartisgammaspecfiles` has no options. It downloads the NuDat3 decay tables and writes a gamma
spectrum for each nuclide that has gamma lines.

## hdf5_to_parquet.py

`artisatomic/hdf5_to_parquet.py` is a script and not a console script. It converts an HDF5 file to
Parquet files for the DREAM and AOIFE readers. The docstring at the top of the script explains the
layouts that it reads.

```{program-output} python -m artisatomic.hdf5_to_parquet --help
```

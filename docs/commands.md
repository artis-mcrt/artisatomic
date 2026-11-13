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

The command checks the temperature order and the rate values before it replaces recombrates.txt.
Missing Chianti parameters stop the command. A bare nucleus needs no dielectronic recombination parameters.

## makeartischargetransferfile

```{program-output} makeartischargetransferfile --help
```

## makeartisgammaspecfiles

`makeartisgammaspecfiles` has no options. It downloads the NuDat3 decay tables and writes a gamma
spectrum for each nuclide that has gamma lines.
The output uses 17 significant digits for each energy and intensity. It keeps every line with a positive intensity.
This precision preserves the parsed values through a file round trip. The command rejects a selected line with an invalid energy or intensity.
A missing intensity stops the command, because an unmeasured contribution has no known value. A finite zero intensity gives no line.

## hdf5_to_parquet.py

`artisatomic/hdf5_to_parquet.py` is a script and not a console script. It converts an HDF5 file to
Parquet files for the DREAM and AOIFE readers. The docstring at the top of the script explains the
layouts that it reads.

```{program-output} python -m artisatomic.hdf5_to_parquet --help
```

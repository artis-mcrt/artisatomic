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

The command stops if a Nahar table has a value that is not finite, or temperatures that do not increase.
It also stops if Chianti has no recombination parameters for an ion, because ChiantiPy then gives a rate of zero.
A bare nucleus needs no dielectronic parameters. The earlier recombrates.txt stays when the command stops.

## makeartischargetransferfile

```{program-output} makeartischargetransferfile --help
```

## makeartisgammaspecfiles

`makeartisgammaspecfiles` downloads the NuDat3 decay tables and writes a gamma spectrum for each nuclide that has gamma lines.
Each file gives the energy in MeV and the intensity for each decay of each gamma line of the ground level.
Each number is the shortest text that reads back as the same 64-bit number, and the file holds each line with a
positive intensity. The lines of the file thus keep the total energy of the lines in the NuDat table.

```{program-output} makeartisgammaspecfiles --help
```

## hdf5_to_parquet.py

`artisatomic/hdf5_to_parquet.py` is a script and not a console script. It converts an HDF5 file to
Parquet files for the DREAM and AOIFE readers. The docstring at the top of the script explains the
layouts that it reads.

```{program-output} python -m artisatomic.hdf5_to_parquet --help
```

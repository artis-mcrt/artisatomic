# Commands

The package installs four console scripts. This page gives the help text of each script that has an
argument parser.

## makeartisatomicfiles

`makeartisatomicfiles` writes adata.txt, compositiondata.txt, phixsdata_v2.txt and transitiondata.txt.
The command `python -m artisatomic` starts the same program.

```{program-output} makeartisatomicfiles --help
```

### Photoionisation interpolation

Set `-phixsinterpolation` to the cross section mode of the ARTIS run:

- `linear` is the default. Use it if `PHIXS_CLASSIC_NO_INTERPOLATION` is false in ARTIS.
  ARTIS then joins the points of a table with straight lines.
- `classic`: use it if `PHIXS_CLASSIC_NO_INTERPOLATION` is true in ARTIS.
  ARTIS then holds each value constant up to the next point.

The file comment of phixsdata_v2.txt records the mode. ARTIS does not read the mode from that comment.

The reduction preserves the recombination integral of each source table at `-optimaltemperature`.
ARTIS integrates from the threshold to the last point of a table, and the reduction uses the same range.
In linear mode, each value is an average of the source with the weight of the linear basis function of its point.
In classic mode, each value is an average of the source over the interval from its point to the next point.
The last classic value is for the interval above the last point. ARTIS also scales the nu^-3 tail with it.

A coarse grid still changes the shape of the cross section and the rates at other temperatures.
The combination of routes with different shapes can also change the rate of a target.
Use more points or a smaller frequency step if those errors are important.

## makeartisrecombratefile

```{program-output} makeartisrecombratefile --help
```

## makeartischargetransferfile

```{program-output} makeartischargetransferfile --help
```

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

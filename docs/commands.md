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

- `linear` is the default. Use it when `PHIXS_CLASSIC_NO_INTERPOLATION` is false.
- `classic` uses constant intervals. Use it when `PHIXS_CLASSIC_NO_INTERPOLATION` is true.

The output comment records this choice. ARTIS does not select its mode from that comment.
The reduction preserves each source table's recombination integral at `-optimaltemperature`.
The integral starts at the threshold and ends at the last stored point, as in ARTIS.
The power-law tail beyond that point is outside the integral.

Linear mode uses positive weighted averages over the linear basis functions of the stored points.
Those functions sum to one, which preserves the integral and gives non-negative values for a non-negative source.
Classic mode uses weighted averages over intervals that start at each stored point.
Its last value sets the tail only, so it takes the source value at the last point.

A coarse grid still changes the cross section shape and the rates at other temperatures.
The combination of routes with different shapes can also change a target's rate.
Use more points or a smaller frequency step when those errors matter.

## makeartisrecombratefile

```{program-output} makeartisrecombratefile --help
```

## makeartischargetransferfile

```{program-output} makeartischargetransferfile --help
```

## makeartisgammaspecfiles

`makeartisgammaspecfiles` has no options. It downloads the NuDat3 decay tables and writes a gamma
spectrum for each nuclide that has gamma lines.

## hdf5_to_parquet.py

`artisatomic/hdf5_to_parquet.py` is a script and not a console script. It converts an HDF5 file to
Parquet files for the DREAM and AOIFE readers. The docstring at the top of the script explains the
layouts that it reads.

```{program-output} python -m artisatomic.hdf5_to_parquet --help
```

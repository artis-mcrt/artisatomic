# Commands

The package installs four console scripts. This page gives the help text of each script.

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
If those errors are important, decrease `-phixsnuincrement` and increase `-nphixspoints` together.
More points with the same step only extend the frequency range of a table.

## makeartisrecombratefile

`makeartisrecombratefile` writes recombrates.txt into the output folder of `makeartisatomicfiles`.

```{program-output} makeartisrecombratefile --help
```

The repository does not hold the Nahar files. The command looks for them in the folder
`atomic-data-nahar` in the repository root. The name of a file starts with the element symbol in lower case
and the ion stage of the ion after the recombination. Then come `.rrc`, optional characters, and `.txt`.
For example, `fe2.rrc.txt` holds the rates of the recombination to Fe II. The command skips a name
that holds a space or a bracket. With no such folder, each ion with an atomic number of 30
or less takes the ChiantiPy rates, if CHIANTI has radiative recombination parameters for it.
The command gives a warning that names each ion with no such parameters, and recombrates.txt
has no rates for that ion. The command also gives a warning for an ion with no dielectronic
recombination parameters, if the ion is not a bare nucleus. The rate of that ion is the
radiative rate only.

ChiantiPy reads the CHIANTI database, which is a different download. The environment variable
`XUVTOP` gives the folder of the database to ChiantiPy.

## makeartischargetransferfile

`makeartischargetransferfile` writes chargetransfer.txt. The default output folder is
`artis_files/data` in the repository, and not in the working directory. Git tracks the source files
in `atomic-data-chargetransfer`, so the command needs no download.

```{program-output} makeartischargetransferfile --help
```

## makeartisgammaspecfiles

`makeartisgammaspecfiles` downloads the NuDat3 decay tables and writes a gamma spectrum for each nuclide that has gamma lines.
The command needs a network connection. It reads the nuclides from `betaminusdecays.txt` and
`alphadecays.txt` in the data folder of artistools. It writes each file `gamma_<nuclide>.txt` into
`artis_files/data` in the repository, and no option changes that folder.

```{program-output} makeartisgammaspecfiles --help
```

## hdf5_to_parquet.py

`artisatomic/hdf5_to_parquet.py` is a script and not a console script. It converts an HDF5 file to
Parquet files for the DREAM and AOIFE readers. The docstring at the top of the script explains the
layouts that it reads.

```{program-output} python -m artisatomic.hdf5_to_parquet --help
```

## Tab completion

`makeartisatomicfiles`, `makeartisrecombratefile` and `makeartischargetransferfile` use
[argcomplete](https://kislyuk.github.io/argcomplete/). To complete the options of a command in bash
or zsh, run this line:

```sh
eval "$(register-python-argcomplete makeartisatomicfiles)"
```

To use the completion in each new shell, add the line to the startup file of the shell.

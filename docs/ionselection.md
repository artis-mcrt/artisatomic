# Ion selection

`makeartisatomicfiles` has no configuration interface for the ion selection. An ion handlers file selects the ions and the handler that reads each ion. With no such file, the program uses the built-in ion selection.

## The ion handlers file

The program looks for the file in this order:

1. the file that the option `-ionhandlersfile` names;
2. `artisatomicionhandlers.json` in the working directory;
3. no file, which gives the built-in ion selection.

The file is a JSON list with one entry for each element. An entry is the atomic number and a list of `[ion stage, handler]` pairs. Ion stage 1 is the neutral atom. This example selects Ca III and Co II–IV:

```json
[
    [20, [[3, "adas"]]],
    [27, [[2, "cmfgen_qubphixs"], [3, "adas"], [4, "adas"]]]
]
```

These rules apply:

- The selection must contain at least one ion.
- An element can have an empty ion list if another element contains an ion.
- The ion stages of an element must have no gap.
- Each handler name must be one of the names on the page [Data sources](datasources.md). The program also accepts the old names `qub`, `qub_data` and `qub_cobalt`, and changes each one to a name of that page.

The run stops before it writes an output file if the file breaks a rule. The run also stops if the command gives `-minionstage`, `-maxionstage` or `-maxatomicnumber` with an ion handlers file, because the file selects the ions itself.

Each set in `tests/` except `chargetransfer` holds an example file. Most sets need the test mode, because they read the samples in the repository. The `cmfgen_lowz` set needs only the CMFGEN data set:

```sh
makeartisatomicfiles -ionhandlersfile tests/cmfgen_lowz/artisatomicionhandlers.json
```

```{warning}
A file with the name `artisatomicionhandlers.json` in the working directory controls every run in that directory that has no `-ionhandlersfile` option. Use `-ionhandlersfile` with a different file name. As an alternative, remove the file after the run.
```

## The built-in ion selection

With no ion handlers file, the program selects each ion for which one of these handlers finds data. The first handler that offers an ion reads that ion.

1. `adas`: each ion with a file `<atomic number>_<ion stage>.adf04` in `atomic-data-adas`, except Ca III.
2. `kurucz`: Sr I–III, Y I–II and Zr I–III. The `adas` handler reads Sr I if its file exists.
3. `cmfgen`: each ion of the CMFGEN data set that the reader knows.
4. `floers25calibwithforbidden`, `floers25calib` and `floers25uncalib`: each ion with a Floers+25 file.
5. `tanakajplt`: each ion with a file in `atomic-data-tanaka-jplt/data_v2.1`.

The options `-minionstage` (default 1), `-maxionstage` (default 5) and `-maxatomicnumber` (no limit) then remove ions from the selection.

The built-in selection needs these data sets. The page [Data sources](datasources.md) says how to get each one.

- The CMFGEN data set. If the folder does not exist, the run stops with the name of the download script.
- The Floers+25 data set. The run stops if neither the public folder nor the private folder exists. This rule applies for each value of `-maxatomicnumber`.
- The Kurucz line lists of the selected ions of Sr, Y and Zr. The run stops before it reads an ion if a file does not exist. `-maxatomicnumber 37` or less removes these ions.

The selection also takes the JPLT ions if the folder `atomic-data-tanaka-jplt/data_v2.1` exists. With no such folder, the selection has no JPLT ion, and the program gives no warning. The ADAS ions also come only from the files that exist.

No data source of the built-in selection holds Sr V. Thus the check of the ion stages stops a selection that holds Sr IV and Sr VI. An example is `-maxionstage 6` with the default values of the other two limits.

## Show the selection

`--listions` prints the selection and stops. The program reads no ion and writes no output file. The standard output holds the selection only, in the format of an ion handlers file. Thus the output is a start for a file that you edit:

```sh
makeartisatomicfiles --listions -maxatomicnumber 30 > my_ions.json
makeartisatomicfiles -ionhandlersfile my_ions.json
```

Do not write the output to `artisatomicionhandlers.json` in the working directory. The shell empties that file before the program reads it, and the program then stops.

## Repeat a run

A run that ends with no error writes `artisatomicionhandlers_used.json` into the output folder, in the format of an ion handlers file. Give that record to a later run to select the same ions and handlers:

```sh
makeartisatomicfiles -ionhandlersfile artis_files/artisatomicionhandlers_used.json -output_folder artis_files_repeat
```

Do not give an ion limit to the later run. The record holds only the ions that the limits kept.

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
- Each handler name must be one of the names on the page [Data sources](datasources.md).

The run stops before it writes an output file if the file breaks a rule. The run also stops if the command gives `-minionstage`, `-maxionstage` or `-maxatomicnumber` with an ion handlers file, because the file selects the ions itself.

Each set in `tests/` except `chargetransfer` holds an example file. For example:

```sh
makeartisatomicfiles -ionhandlersfile tests/kurucz/artisatomicionhandlers.json
```

```{warning}
A file with the name `artisatomicionhandlers.json` in the working directory controls every run in that directory that has no `-ionhandlersfile` option. Use `-ionhandlersfile` with a different file name, or remove the file after the run.
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

- The CMFGEN data set. The run stops with the name of the setup script if the folder does not exist.
- The Floers+25 data set. The run stops if neither the public folder nor the private folder exists. This rule applies for each value of `-maxatomicnumber`.
- The Kurucz line lists of the selected ions of Sr, Y and Zr. The run stops before it reads an ion if a file does not exist. `-maxatomicnumber 37` or less removes these ions.

No data source of the built-in selection holds Sr V. Thus `-maxionstage 6` or more stops at the check of the ion stages, unless `-maxatomicnumber` is 37 or less.

## Show the selection

`--listions` prints the selection and stops. The program reads no ion and writes no output file. The standard output holds the selection only, in the format of an ion handlers file. Thus the output is a start for a file that you edit:

```sh
makeartisatomicfiles --listions -maxatomicnumber 30 > my_ions.json
makeartisatomicfiles -ionhandlersfile my_ions.json
```

## Repeat a run

Each run writes `artisatomicionhandlers_used.json` into the output folder, in the format of an ion handlers file. Give that record to a later run to select the same ions and handlers:

```sh
makeartisatomicfiles -ionhandlersfile artis_files/artisatomicionhandlers_used.json -output_folder artis_files_repeat
```

The record holds the ions that the limits kept, so the later run must give no ion limit.

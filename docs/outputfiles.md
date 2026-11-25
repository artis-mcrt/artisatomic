# Output files

`makeartisatomicfiles` writes these files into the output folder (`-output_folder`, default `artis_files`):

| file | content |
|---|---|
| `adata.txt` | the levels of each ion |
| `compositiondata.txt` | the elements and the range of ion stages of each element |
| `transitiondata.txt` | the transitions of each ion, with the collision strengths |
| `phixsdata_v2.txt` | the photoionisation cross sections. `--nophixs` writes no such file, and removes the file of an earlier run from the output folder. |
| `artisatomiclog.txt` | the log of the run, for all ions |
| `artisatomicionhandlers_used.json` | the record of the ions and the handlers of the run |

ARTIS reads the first four files in the table.

The program checks the options and the ion selection before it creates a log.
If these checks fail, read the error message in the terminal.

During conversion, the program writes to a work folder. It moves the files into the output folder after a successful run.
If a reader fails, the terminal shows the error. The program saves the log as `artisatomiclog_failed.txt` and keeps the output of the previous successful run.

## Comments in the output files
`adata.txt` and `transitiondata.txt` start with a file comment. In `phixsdata_v2.txt`, the file comment follows the point count and the frequency step.
The file comment explains the fields and gives the options that apply to all ions. Level numbers and ion stages start at 1.
For example, `-optimaltemperature` sets the temperature for the reduction of the cross sections.
The option `-electrontemperature` sets the temperature for the collision strengths.

The file comment gives the creation time in UTC. Set `SOURCE_DATE_EPOCH` for a run whose files must be the same as the files of an earlier run. `ARTISATOMIC_TESTMODE=1` gives a fixed time of 1970-01-01T00:00:00Z, and the test mode comes before `SOURCE_DATE_EPOCH`.

adata.txt, transitiondata.txt and phixsdata_v2.txt have a comment block before the data of each ion. Each line of a comment block starts with `#`. A comment block gives:

- the ion and the handler;
- the data source with its reference (the `source:` line);
- the source files;
- the choices and the warnings for that file.

A comment block does not repeat a number of the header line of the ion, for example the count of levels or the ionisation energy.

In `phixsdata_v2.txt`, only ions with a cross section table have a comment block. The comment blocks contain ASCII characters only.

artisatomic writes only numbers in `compositiondata.txt` for compatibility with older ARTIS releases.
ARTIS v2026.11.9 accepts both comment lines and comments after data in this file. [ARTIS reader change](https://github.com/artis-mcrt/artis/pull/616)

The log file `artisatomiclog.txt` holds the same lines for all ions. Some kinds of detail line occur for many levels or transitions of one ion. For such a kind, the comment block gives only the first line, with the count. The log file has each of those lines. The log file is in the output folder, beside `artisatomicionhandlers_used.json`, which records the ions and the handlers of the run. Give that record to `-ionhandlersfile` to repeat the run.

ARTIS v2023.10 and later skip comment blocks in `adata.txt`, `transitiondata.txt` and `phixsdata_v2.txt`.
For an older release, run these commands in the folder with the three files.

```sh
for f in adata.txt transitiondata.txt phixsdata_v2.txt; do grep -v '^#' "$f" > "$f.tmp" && mv "$f.tmp" "$f"; done
```

artistools v2026.9.22 and later skip the comment blocks.

## Files of the other commands

| file | command | folder |
|---|---|---|
| `recombrates.txt` | `makeartisrecombratefile` | the output folder (`-output_folder`, default `artis_files`) |
| `chargetransfer.txt` | `makeartischargetransferfile` | `-output_folder`, default `artis_files/data` in the repository |
| `gamma_<nuclide>.txt` | `makeartisgammaspecfiles` | `artis_files/data` in the repository |

The page [Commands](commands.md) gives the details of each command.

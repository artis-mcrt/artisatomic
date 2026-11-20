# Output files

`makeartisatomicfiles` writes these files into the output folder (`-output_folder`, default `artis_files`):

| file | content |
|---|---|
| `adata.txt` | the levels of each ion |
| `compositiondata.txt` | the elements and the range of ion stages of each element |
| `transitiondata.txt` | the transitions of each ion, with the collision strengths |
| `phixsdata_v2.txt` | the photoionisation cross sections. `--nophixs` writes no such file. |
| `artisatomiclog.txt` | the log of the run, for all ions |
| `artisatomicionhandlers_used.json` | the record of the ions and the handlers of the run |

ARTIS reads the first four files. The file comment at the top of adata.txt, transitiondata.txt and phixsdata_v2.txt explains each field of that file.

The run writes all files into a work folder in the output folder. It moves the files into the output folder only at the end of a good run. A run that fails keeps the files of the earlier run, and puts its log into `artisatomiclog_failed.txt`.

## Comments in the output files
adata.txt, transitiondata.txt and phixsdata_v2.txt start with a file comment. The file comment explains each field of the file. It says that level numbers and ion stages start at 1, and not at 0. It also gives the options of the run that apply to all ions. Two examples are the temperature of the cross section downsample (`-optimaltemperature`) and the temperature of the collision strengths (`-electrontemperature`). In phixsdata_v2.txt the file comment comes after the first two numbers, because ARTIS reads them with no comment skip.

The file comment gives the creation time in UTC. Set `SOURCE_DATE_EPOCH` for a run whose files must be the same as the files of an earlier run. `ARTISATOMIC_TESTMODE=1` gives a fixed time of 1970-01-01T00:00:00Z, and the test mode comes before `SOURCE_DATE_EPOCH`.

adata.txt, transitiondata.txt and phixsdata_v2.txt have a comment block before the data of each ion. Each line of a comment block starts with `#`. A comment block gives:

- the ion and the handler;
- the data source with its reference (the `source:` line);
- the source files;
- the choices and the warnings for that file.

A comment block does not repeat a number of the header line of the ion, for example the count of levels or the ionisation energy.

In phixsdata_v2.txt, an ion with no cross section table has no comment block. The log file `artisatomiclog.txt` holds the same lines for all ions. A kind of detail line that occurs for many levels or transitions of one ion gives one line in the comment block: the first such line, with the count. The log file has each of those lines. The log file is in the output folder, beside `artisatomicionhandlers_used.json`, which records the ions and the handlers of the run. Give that record to `-ionhandlersfile` to repeat the run. compositiondata.txt has no comment block, because ARTIS reads it with no comment skip. The comment blocks contain ASCII characters only.

ARTIS v2023.10 and later skip the comment blocks. An older ARTIS release stops on them. Remove them from all three files for such a release. `grep -v` only prints the lines. Write the result to a new file, then replace the old file:

```sh
for f in adata.txt transitiondata.txt phixsdata_v2.txt; do grep -v '^#' "$f" > "$f.tmp" && mv "$f.tmp" "$f"; done
```

artistools needs a version that skips the comment blocks.

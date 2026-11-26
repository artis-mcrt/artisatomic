# Output checksum test sets

The test workflow runs each atomic data set with its own ion handlers file. It checks the four output files against
`checksums.txt`. The charge transfer set has a separate job. [Test workflow](https://github.com/artis-mcrt/artisatomic/blob/main/.github/workflows/test.yml)

Each MD5 checksum covers one complete file, including its comments. The `mons` set has no photoionisation tables.
The file comment in `phixsdata_v2.txt` records `-optimaltemperature`.
If you change this option from 6000 K to 7000 K for `mons`, only that file gets a different checksum.

Run this example from the repository root. The `mons` set uses the sample that Git tracks.
`python -m artisatomic` starts the same program as `makeartisatomicfiles`.

```bash
testset=mons
ARTISATOMIC_TESTMODE=1 uv run -m artisatomic \
    --ionhandlersfile "tests/$testset/artisatomicionhandlers.json" \
    --output-folder "tests/$testset/output"
(cd "tests/$testset/output" && md5sum -c ../checksums.txt)
```

If a checksum differs, compare the new file with the original file.
For a change to comments only, compare both files after you remove the `#` lines with `grep -v '^#'`.
The data rows must match. For an intended change to a data row, verify the new values.
Update the reference checksums only after you verify the changes for that set.

```bash
(cd "tests/$testset/output" && md5sum adata.txt compositiondata.txt phixsdata_v2.txt transitiondata.txt > ../checksums.txt)
```

Name the four files explicitly. The output folder also contains the log file `artisatomiclog.txt`.

`ARTISATOMIC_TESTMODE=1` selects the test samples and sets the creation time in file comments to 1970-01-01T00:00:00Z.
The test mode takes precedence over `SOURCE_DATE_EPOCH`. For the Floers+25 set, extract `testdata.tar.xz` in
`atomic-data-floers25` first. The test mode uses the sample in place of the public and private data sets.

Do not copy the file of a set to the repository root. With no `-ionhandlersfile` option,
`find_handlers_file()` finds `artisatomicionhandlers.json` in the working directory if that file exists.
Thus a copy left in the repository root silently overrides the built-in ion selection of every later
local run. A later run that gives `-minionstage`, `-maxionstage` or `-maxatomicnumber` stops with an
error instead.

Of the seven atomic data sets, only `mons` runs without the CMFGEN data set.
The hydrogenic estimate uses the CMFGEN tables for H I. The MONS level names contain no configuration for this estimate.
The `jplt` set also needs the JPLT download. The remaining input files come from the repository.

When a set needs a new ion, add the ion to a sample if possible.
For example, `testdata.tar.xz` contains the ions for the Floers+25 tests. The full Floers+25 data set uses 5.5 GiB.

## What each set is for

The ion lists are not arbitrary. Each one was chosen to reach code that no other set reaches, so
trimming a set for speed can silently delete coverage while looking like a routine checksum update.

| set | ions | why these |
|---|---|---|
| `adas` | Ca III, Co II–IV, Sr I | The `adas` handler and the `cmfgen_qubphixs` handler, from the committed `co_tyndall_test_sample/`. Co II uses `cmfgen_qubphixs`: CMFGEN levels with the QUB cross sections. Co III uses `adas` on the QUB adf04 file. Co IV is a placeholder of one level, because no file gives it. **Sr I** uses the `adas` handler on the committed `38_1.adf04.zst`. That file is complete, so it carries the `-1` row that ends the collision block, and the Co sample stops before that row. The Sr I file and the Ca III file use zstd compression. The Co sample files are gzip compressed. Sr I is the top ion, so it writes no phixs block. Every one of its 1596 output transitions carries a collision strength, and 224 of them come from an upsilon value with no A-value. **Ca III** uses the `adas` handler on the committed `20_3.adf04.zst`, a file from OPEN-ADAS in LS coupling. It is the only file that gives the configurations in Eissner notation, so it tests the conversion to standard notation. Each of its 86 levels is an LS term. The XJ field gives the statistical weight of the term, so the levels have no J. It writes 3164 transitions, and 1824 of them come from an upsilon value with no A-value. |
| `cmfgen` | O I–IV, Fe I–V, Co II–IV, Ni II–V | The main CMFGEN set. **O I and O III** carry term-named collision data against J-split levels (O II names each level with its J), so they are what drives the term-to-J upsilon redistribution into the output (O III's `col_data` names all of its rows but three by term). O I has two photoionisation files, and its levels carry both merge markers. Stages above O II exist so O II is not the top ion and its cross sections are read at all. |
| `cmfgen_lowz` | H I–II, He I–III, C I–IV, N I–III, F II–III, P IV–V | The awkward corners of the CMFGEN reader. **N I** has four phot files and **F II** three, the tests of the multi-file merge beyond two. F III also has four, but it is the top ion of the set, so the run reads none of them. **H II, He III, F II and F III** get no collision strengths, because `ions_data` names no collision data file for them. The files `FLU/II/tst/col_data.dat` and `FLU/III/tst/col_data.dat` hold the values of Ne III and Ne IV. **H II and He III** are bare nuclei. The single dummy level of each skips the oscillator file. He III makes He II a lower ion, so the run reads the cross sections of He II. **H I and He II** have no `Format date` line, so they take the `hillier_rowformat_noheader` legacy branch. **C II and N III** supply the `13w` merge-marker configurations, **He I and C IV** the `5z` ones. |
| `floers25` | La III (uncalibrated), Yb II–III (calibrated) | Both Floers+25 branches — the reader opens a different pair of filenames for each. **Yb II** is deliberately not a top ion, so the floers25 level names are also exercised through the hydrogenic phixs estimate. These are the smallest ions that cover both branches — 176 KB of `testdata.tar.xz`, against 73 MB of transitions for Dy III alone. |
| `jplt` | Se I–IV + Se V, Nb I–IV | **Se II and III** are the only ions in the matrix using the v2.1 LS-term level-name format; Se I, Se IV and all of Nb use the older format. This matters beyond level names: JPLT supplies no photoionisation data, so `get_level_valence_n()` feeds the hydrogenic estimate written to `phixsdata_v2.txt`. **Se V** is read by `gsnist`, which is that handler's only coverage. Se is read by two handlers, as Co is in the `adas` set. |
| `kurucz` | Sr I–II, Y I–II | The Kurucz gfall reader, from the committed `test_sample/`. |
| `mons` | Ce V–VI | The MONS lanthanide reader, from the committed `test_sample/`. `atomic-data-mons/make_test_sample.py` cut the sample from the 21.7 GB archive. It holds the lowest 450 levels of Ce V and 400 of Ce VI. It also holds every transition between two of those levels. The set has two ions, so Ce VI is the top ion. The level names carry no configuration, so artisatomic writes no hydrogenic phixs estimate. |

## Adding a set

Add a separate set if the new ions are unrelated to the ions of an existing set. The workflow runs the sets in parallel.

## The charge transfer set

`chargetransfer/` is not a matrix entry: the `chargetransfer` job of the workflow runs
`python -m artisatomic.makechargetransferfile` and checks `chargetransfer.txt` against
`checksums.txt`. The source files are tracked in `atomic-data-chargetransfer`, so the job downloads
nothing. The test `test_chargetransfer_output_matches_the_checksum` checks the same checksum under
pytest. Regenerate the checksum with:

```bash
PYTHONPATH="$PWD" uv run makeartischargetransferfile -output_folder tests/chargetransfer/output
(cd tests/chargetransfer/output && md5sum *.txt > ../checksums.txt)
```

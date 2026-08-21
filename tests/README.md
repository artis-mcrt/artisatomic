# Output checksum test sets

Each directory here is one entry of the `tests` matrix in
[../.github/workflows/test.yml](../.github/workflows/test.yml). The job copies the set's
`artisatomicionhandlers.json` to the repository root, runs `python -m artisatomic` under coverage
(`-m` is the form `coverage run` can name), and checks the four output files against
`checksums.txt`. That is the same `main()` the `makeartisatomicfiles` console script calls, so the
recipe below reproduces it.

Verification is whole-file MD5, so **any** change to level naming, sorting, phixs downsampling or a
default argument invalidates every checksum at once. Regenerate a set with:

```bash
export ARTISATOMIC_TESTMODE=1 PYTHONPATH="$PWD"
cp tests/<name>/artisatomicionhandlers.json .
uv run makeartisatomicfiles -output_folder tests/<name>/output
rm artisatomicionhandlers.json
(cd tests/<name>/output && md5sum *.txt > ../checksums.txt)
```

`ARTISATOMIC_TESTMODE=1` is what redirects the Kurucz, QUB and MONS readers to their committed
`test_sample/` directories, so it is required — the workflow sets it globally. The `rm` matters:
`get_ion_handlers()` prefers `artisatomicionhandlers.json` whenever it exists, so a copy left in the
repository root silently overrides the built-in ion selection of every later local run.

**Every** set needs the CMFGEN corpus, not just the `cmfgen` ones, which is why the workflow's
CMFGEN setup step is the one not gated on `matrix.testname` (the comment there says why). `jplt`
downloads its own corpus on top; Kurucz, QUB, MONS and Floers+25 come from committed samples. When a set
needs a new ion, prefer adding it to a committed sample over introducing another download: the
published Floers+25 corpus is 5.5 GiB, but `testdata.tar.xz` carries only the ions the tests name.

## What each set is for

The ion lists are not arbitrary. Each one was chosen to reach code that no other set reaches, so
trimming a set for speed can silently delete coverage while looking like a routine checksum update.

| set | ions | why these |
|---|---|---|
| `cmfgen` | O I–IV, Fe I–V, Co II–IV, Ni II–V | The main CMFGEN set. **O I, O II and O III** all carry term-named collision data against J-split levels, so they are what drives the term-to-J upsilon redistribution into the output (O III's `col_data_oiii_butler_2012.dat` is the purest case — every row is term-named). O I has two photoionisation files, and its levels carry both merge markers. Stages above O II exist so O II is not the top ion and its cross sections are read at all. |
| `cmfgen_lowz` | H I–II, He I–II, C I–IV, N I–III, F II–III, P IV–V | The awkward corners of the CMFGEN reader. **N I and F III** have four phot files and **F II** three, the tests of the multi-file merge beyond two. **F II/F III** are the only ions with no collision data file. **H II** is the bare proton, whose single dummy level skips the oscillator file. **H I and He II** have no `Format date` line, so they take the `hillier_rowformat_noheader` legacy branch. **C II and N III** supply the `13w` merge-marker configurations, **He I and C IV** the `5z` ones. |
| `floers25` | La III (uncalibrated), Yb II–III (calibrated) | Both Floers+25 branches — the reader opens a different pair of filenames for each. **Yb II** is deliberately not a top ion, so the floers25 level names are also exercised through the hydrogenic phixs estimate. These are the smallest ions that cover both branches — 176 KB of `testdata.tar.xz`, against 73 MB of transitions for Dy III alone. |
| `jplt` | Se I–IV + Se V, Nb I–IV | **Se II and III** are the only ions in the matrix using the v2.1 LS-term level-name format; Se I, Se IV and all of Nb use the older format. This matters beyond level names: JPLT supplies no photoionisation data, so `get_level_valence_n()` feeds the hydrogenic estimate written to `phixsdata_v2.txt`. **Se V** is read by `gsnist`, which is both that handler's only coverage and the only element read by two handlers. |
| `kurucz` | Sr I–II, Y I–II | The Kurucz gfall reader, from the committed `test_sample/`. |
| `mons` | Ce V–VI | The MONS lanthanide reader, from the committed `test_sample/`. `atomic-data-mons/make_test_sample.py` cut the sample from the 21.7 GB archive. It holds the lowest 450 levels of Ce V and 400 of Ce VI. It also holds every transition between two of those levels. The set has two ions, so Ce VI is the top ion. The level names carry no configuration, so artisatomic writes no hydrogenic phixs estimate. |
| `qub` | Co II–IV | The `qub_cobalt` handler, from the committed `co_tyndall_test_sample/`. Co II comes from CMFGEN, Co III–IV from the QUB adf04 files. |

## Adding a set

Prefer a new set over extending an existing one when the ions are unrelated: matrix entries run in
parallel, so a new set is free in wall clock, while extending a set churns its four checksums.

## The charge transfer set

`chargetransfer/` is not a matrix entry: the `chargetransfer` job of the workflow runs
`python -m artisatomic.makechargetransferfile` and checks `chargetransfer.txt` against
`checksums.txt`. The source files are tracked in `atomic-data-chargetransfer`, so the job downloads
nothing. The test `test_chargetransfer_output_matches_the_checksum` checks the same checksum under
pytest. Regenerate the checksum with:

```bash
PYTHONPATH="$PWD" uv run makechargetransferfile -outputfile tests/chargetransfer/output/chargetransfer.txt
(cd tests/chargetransfer/output && md5sum chargetransfer.txt > ../checksums.txt)
```

# Output checksum test sets

Each directory here is one entry of the `tests` matrix in
[../.github/workflows/test.yml](../.github/workflows/test.yml). The job copies the set's
`artisatomicionhandlers.json` to the repository root, runs `makeartisatomicfiles`, and checks the
four output files against `checksums.txt`.

Verification is whole-file MD5, so **any** change to level naming, sorting, phixs downsampling or a
default argument invalidates every checksum at once. Regenerate a set with:

```bash
ARTISATOMIC_TESTMODE=1 PYTHONPATH="$PWD" sh -c 'cp tests/<name>/artisatomicionhandlers.json . && uv run makeartisatomicfiles -output_folder tests/<name>/output && (cd tests/<name>/output && md5sum *.txt > ../checksums.txt)'
```

`ARTISATOMIC_TESTMODE=1` is what redirects the Kurucz and QUB readers to their committed
`test_sample/` directories, so it is required — the workflow sets it globally.

## What each set is for

The ion lists are not arbitrary. Each one was chosen to reach code that no other set reaches, so
trimming a set for speed can silently delete coverage while looking like a routine checksum update.

| set | ions | why these |
|---|---|---|
| `cmfgen` | O I–IV, Fe I–V, Co II–IV, Ni II–V | The main CMFGEN set. **O III** is the only ion whose collision data is term-resolved (`col_data_oiii_butler_2012.dat`), so it is the only output-level test of the term-to-J upsilon redistribution. O I has two photoionisation files. Stages above O II exist so O II is not the top ion and its cross sections are read at all. |
| `cmfgen_lowz` | H I–II, He I–II, C I–IV, N I–III, F II–III, P IV–V | The awkward corners of the CMFGEN reader. **N I** has four phot files and **F II** three, the only tests of the multi-file merge beyond two. **F II/F III** are the only ions with no collision data file. **H II** is the bare proton, whose single dummy level skips the oscillator file. **N I** supplies the `13w`/`5z` merge-marker configurations. **He I** uses the `_old` oscillator format. |
| `floers25` | Dy III (uncalibrated) | The uncalibrated Floers+25 branch, from the committed `testdata.tar.xz`. |
| `floers25calib` | La III, Yb II–III (calibrated) | The calibrated branch, downloaded by `setup_floers25_data.sh`. **Yb II** is deliberately not a top ion, so the floers25 level names are also exercised through the hydrogenic phixs estimate. |
| `jplt` | Se I–IV + Se V, Nb I–IV | **Se II and III** are the only ions in the matrix using the v2.1 LS-term level-name format; Se I, Se IV and all of Nb use the older format. This matters beyond level names: JPLT supplies no photoionisation data, so `get_level_valence_n()` feeds the hydrogenic estimate written to `phixsdata_v2.txt`. **Se V** is read by `gsnist`, which is both that handler's only coverage and the only element read by two handlers. |
| `kurucz` | Sr I–II, Y I–II | The Kurucz gfall reader, from the committed `test_sample/`. |
| `qub` | Co II–IV | The `qub_cobalt` handler, from the committed `co_tyndall_test_sample/`. Co II comes from CMFGEN, Co III–IV from the QUB adf04 files. |

## Adding a set

Add the directory with an `artisatomicionhandlers.json`, add the name to `matrix.testname`, add any
data-setup step it needs, generate `checksums.txt`, and add a row above saying what it covers.

Prefer a new set over extending an existing one when the ions are unrelated: matrix entries run in
parallel, so a new set is free in wall clock, while extending a set churns its four checksums.

# artisatomic
[![Build and test](https://github.com/artis-mcrt/artisatomic/actions/workflows/test.yml/badge.svg)](https://github.com/artis-mcrt/artisatomic/actions/workflows/test.yml)

>The Python package converts atomic data from several sources (for example CMFGEN, Kurucz, JPLT, ADAS) into the ARTIS format.

The documentation is at <https://artisatomic.readthedocs.io>.

## Installation and Development
First clone the repository, for example:
```sh
git clone https://github.com/artis-mcrt/artisatomic.git
cd artisatomic
```

To use a uv project virtual environment with locked dependency versions run:
```sh
uv sync --frozen
source .venv/bin/activate
```

Or to install into the system environment with pip:
```sh
python3 -m pip install --group dev -e .
```

Then install the pre-commit hooks once:
```sh
prek install
```

## Usage
Run "makeartisatomicfiles" at the command-line to create adata.txt, compositiondata.txt, phixsdata_v2.txt, and transitiondata.txt. The tool has no configuration interface for the ion selection. To change ions or data sources, edit the Python code or supply an `artisatomicionhandlers.json` file. The options `-minionstage` (default 1), `-maxionstage` (default 5) and `-maxatomicnumber` (no limit) also limit the built-in ion selection. No data source of the built-in selection holds Sr V, so `-maxionstage 6` or more stops at the check of the ion stages, unless `-maxatomicnumber` is 37 or less.

### The ion handlers file
`artisatomicionhandlers.json` in the working directory selects the ions and the handler that reads each ion. When the file exists, the run ignores the built-in ion selection. The run stops if the command also gives `-minionstage`, `-maxionstage` or `-maxatomicnumber`, because the file selects the ions itself.

The file is a JSON list with one entry for each element. An entry is the atomic number and a list of `[ion stage, handler]` pairs. Ion stage 1 is the neutral atom. This example selects Ca III and Co II–IV:

```json
[
    [20, [[3, "adas"]]],
    [27, [[2, "cmfgen_qubphixs"], [3, "adas"], [4, "adas"]]]
]
```

The selection must contain at least one ion. An element can have an empty ion list if another element contains an ion.

The ion stages of an element must have no gap. The handlers are:

- `adas`
- `boyle`
- `cmfgen`
- `cmfgen_qubphixs`
- `dream`
- `fac`
- `floers25calib`
- `floers25calibwithforbidden`
- `floers25uncalib`
- `gsnist`
- `kurucz`
- `lisbon`
- `mons`
- `tanakajplt`

The `handlers` registry in `artisatomic/iondata.py` gives the reader of each handler. The run stops before it writes an output file if the file names an unknown handler. Each set in `tests/` except `chargetransfer` holds an example file.

Each run writes `artisatomicionhandlers_used.json` into the output folder, in the same format. Copy that record to `artisatomicionhandlers.json` to repeat the run. Remove `artisatomicionhandlers.json` after the run. The file controls every later run in that directory.

### Comments in the output files
adata.txt, transitiondata.txt and phixsdata_v2.txt start with a file comment. The file comment explains each field of the file. It says that level numbers and ion stages start at 1, and not at 0. It also gives the options of the run that apply to all ions. Two examples are the temperature of the cross section downsample (`-optimaltemperature`) and the temperature of the collision strengths (`-electrontemperature`). In phixsdata_v2.txt the file comment comes after the first two numbers, because ARTIS reads them with no comment skip.

The file comment gives the creation time in UTC. Set `SOURCE_DATE_EPOCH` for a run whose files must be the same as the files of an earlier run. `ARTISATOMIC_TESTMODE=1` gives a fixed time of 1970-01-01T00:00:00Z, and the test mode comes before `SOURCE_DATE_EPOCH`.

adata.txt, transitiondata.txt and phixsdata_v2.txt have a comment block before the data of each ion. Each line of a comment block starts with `#`. A comment block gives:

- the ion and the handler;
- the data source with its reference (the `source:` line);
- the source files;
- the choices and the warnings for that file.

A comment block does not repeat a number of the header line of the ion, for example the count of levels or the ionisation energy.

In phixsdata_v2.txt, an ion with no cross section table has no comment block. The log file `artisatomiclog.txt` holds the same lines for all ions. A kind of detail line that occurs for many levels or transitions of one ion gives one line in the comment block: the first such line, with the count. The log file has each of those lines. The log file is in the output folder, beside `artisatomicionhandlers_used.json`, which records the ions and the handlers of the run. Copy that record to `artisatomicionhandlers.json` in the working directory to repeat the run. compositiondata.txt has no comment block, because ARTIS reads it with no comment skip. The comment blocks contain ASCII characters only.

ARTIS v2023.10 and later skip the comment blocks. An older ARTIS release stops on them. Remove them from all three files for such a release. `grep -v` only prints the lines. Write the result to a new file, then replace the old file:

```sh
for f in adata.txt transitiondata.txt phixsdata_v2.txt; do grep -v '^#' "$f" > "$f.tmp" && mv "$f.tmp" "$f"; done
```

artistools needs a version that skips the comment blocks.

The package installs three more commands:

- `makeartisrecombratefile` writes recombrates.txt from the Nahar recombination rate files. An ion with no Nahar file takes the ChiantiPy rates, which need the `chianti` extra (`uv sync --frozen --extra chianti`).
- `makeartischargetransferfile` writes chargetransfer.txt, the charge transfer rate file.
- `makeartisgammaspecfiles` downloads the NuDat3 decay tables and writes a gamma spectrum for each nuclide that has gamma lines.

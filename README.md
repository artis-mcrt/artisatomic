# artisatomic
[![Build and test](https://github.com/artis-mcrt/artisatomic/actions/workflows/test.yml/badge.svg)](https://github.com/artis-mcrt/artisatomic/actions/workflows/test.yml)
[![Documentation](https://readthedocs.org/projects/artisatomic/badge/?version=latest)](https://artisatomic.readthedocs.io)

>The Python package converts atomic data from several sources (for example CMFGEN, Kurucz, JPLT, ADAS) into the ARTIS format.

The documentation is at <https://artisatomic.readthedocs.io>. It has these pages:

- [Ion selection](https://artisatomic.readthedocs.io/en/latest/ionselection.html): how to select the ions and the handler of each ion.
- [Data sources](https://artisatomic.readthedocs.io/en/latest/datasources.html): the data set of each handler and how to get it.
- [Output files](https://artisatomic.readthedocs.io/en/latest/outputfiles.html): the files of a run and their comments.
- [Commands](https://artisatomic.readthedocs.io/en/latest/commands.html): the help text of each command.
- [Development](https://artisatomic.readthedocs.io/en/latest/development.html): the tests, the checks and the code layout.

## Installation
The package needs Python 3.13 or later. The data folders are in the repository, so clone the repository:
```sh
git clone https://github.com/artis-mcrt/artisatomic.git
cd artisatomic
```

To use a [uv](https://docs.astral.sh/uv/) project virtual environment with locked dependency versions run:
```sh
uv sync --frozen
source .venv/bin/activate
```

Or to install into the current environment with pip:
```sh
python3 -m pip install -e .
```

## Quick start
The repository holds the download scripts and not the data. This example needs only the CMFGEN data set. The download is 363 MB, and the data set uses 1.8 GB.

```sh
(cd atomic-data-hillier && . ./setup_cmfgen_data.sh)
makeartisatomicfiles -ionhandlersfile tests/cmfgen_lowz/artisatomicionhandlers.json
```

The file `tests/cmfgen_lowz/artisatomicionhandlers.json` selects some ions of H, He, C, N, F and P. The run writes these files into the folder `artis_files`:

- `adata.txt`, `compositiondata.txt`, `transitiondata.txt` and `phixsdata_v2.txt`, which ARTIS reads;
- `artisatomiclog.txt`, the log of the run;
- `artisatomicionhandlers_used.json`, the record of the ions and the handlers of the run.

Copy the four data files to the folder of an ARTIS model.

## Usage
`makeartisatomicfiles` writes the four data files. An ion handlers file selects the ions and the handler that reads each ion. This example selects Ca III and Co II–IV:

```json
[
    [20, [[3, "adas"]]],
    [27, [[2, "cmfgen_qubphixs"], [3, "adas"], [4, "adas"]]]
]
```

Give the file with `-ionhandlersfile`. With no such option, the program reads `artisatomicionhandlers.json` in the working directory, if that file exists. With no file, the program uses the built-in ion selection. The built-in selection needs the CMFGEN data set and the Floers+25 data set. The options `-minionstage` (default 1), `-maxionstage` (default 5) and `-maxatomicnumber` (no limit) limit the built-in selection. The page [Ion selection](https://artisatomic.readthedocs.io/en/latest/ionselection.html) gives the rules.

These commands give information and write no output file:

| command | result |
|---|---|
| `makeartisatomicfiles --help` | each option with its default value |
| `makeartisatomicfiles --listhandlers` | each handler with its data source |
| `makeartisatomicfiles --listions` | the ion selection, in the format of an ion handlers file |
| `makeartisatomicfiles --version` | the version of the package |

Each option has a form with one dash and a form with two dashes, for example `-maxionstage` and `--maxionstage`.

### Other commands
The package installs three more commands:

- `makeartisrecombratefile` writes recombrates.txt from the Nahar recombination rate files. It reads compositiondata.txt in the output folder, so run `makeartisatomicfiles` first. An ion with no Nahar file takes the ChiantiPy rates, which need the `chianti` extra (`uv sync --frozen --extra chianti`).
- `makeartischargetransferfile` writes chargetransfer.txt, the charge transfer rate file.
- `makeartisgammaspecfiles` downloads the NuDat3 decay tables and writes a gamma spectrum for each nuclide that has gamma lines.

### Tab completion
The commands with options use [argcomplete](https://kislyuk.github.io/argcomplete/). To complete the options of a command in bash or zsh, run this line, or add it to the startup file of the shell:

```sh
eval "$(register-python-argcomplete makeartisatomicfiles)"
```

## Development
Install the development tools, then install the pre-commit hooks once:
```sh
uv sync --frozen
prek install
```

With pip, the command is `python3 -m pip install --group dev -e .`. The page [Development](https://artisatomic.readthedocs.io/en/latest/development.html) gives the commands for the tests and the checks.

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
The package needs Python 3.13 or later. The data folders are in the repository, and the program looks for them beside the package folder. Thus the package must run from a clone. The two install commands below keep the package in the clone. Clone the repository:
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
The repository holds the download scripts and not the data. This example needs only the CMFGEN data set. The download is 363 MiB, and the data set uses 1.8 GiB. The page [Data sources](https://artisatomic.readthedocs.io/en/latest/datasources.html) lists the tools that each download script calls.

```sh
(cd atomic-data-hillier && . ./setup_cmfgen_data.sh)
makeartisatomicfiles -ionhandlersfile tests/cmfgen_lowz/artisatomicionhandlers.json
```

The file `tests/cmfgen_lowz/artisatomicionhandlers.json` selects some ions of H, He, C, N, F and P. The run writes these files into the folder `artis_files`:

- `adata.txt`, `compositiondata.txt`, `transitiondata.txt` and `phixsdata_v2.txt`, which ARTIS reads;
- `artisatomiclog.txt`, the log of the run;
- `artisatomicionhandlers_used.json`, the record of the ions and the handlers of the run.

Copy the four output files to the folder of an ARTIS model.

## Usage
`makeartisatomicfiles` writes the four output files. An ion handlers file selects the ions and the handler that reads each ion. This example selects O I–III and Ca III:

```json
[
    [8, [[1, "cmfgen"], [2, "cmfgen"], [3, "cmfgen"]]],
    [20, [[3, "adas"]]]
]
```

The example needs only the CMFGEN data set. The `adas` handler reads Ca III from a file that Git tracks.

Give the file with `-ionhandlersfile`. With no such option, the program reads `artisatomicionhandlers.json` in the working directory, if that file exists. With no file, the program uses the built-in ion selection. The built-in ion selection needs the CMFGEN data set, the Floers+25 data set and the Kurucz line lists. The options `-minionstage` (default 1), `-maxionstage` (default 5) and `-maxatomicnumber` (no limit) limit the built-in ion selection. The page [Ion selection](https://artisatomic.readthedocs.io/en/latest/ionselection.html) gives the rules.

These commands give information and write no output file:

| command | result |
|---|---|
| `makeartisatomicfiles --help` | each option with its default value |
| `makeartisatomicfiles --listhandlers` | each handler with its data source |
| `makeartisatomicfiles --listions` | the ion selection, in the format of an ion handlers file |
| `makeartisatomicfiles --version` | the version of the package |

Each option has a form with one dash and a form with two dashes, for example `-maxionstage` and `--maxionstage`. In the form with two dashes, `-output_folder` and `-nlevels_hydrogenic_for_unknown_phixs` use hyphens: `--output-folder` and `--nlevels-hydrogenic-for-unknown-phixs`.

### Other commands
The package installs three more commands:

- `makeartisrecombratefile` writes recombrates.txt into the output folder, from the Nahar recombination rate files. It reads compositiondata.txt in the output folder. Run `makeartisatomicfiles` before this command. The repository does not hold the Nahar files. The command looks for them in the folder `atomic-data-nahar` in the repository root. An ion with no Nahar file takes the ChiantiPy rates if its atomic number is 30 or less. The ChiantiPy rates need the `chianti` extra (`uv sync --frozen --extra chianti`) and the CHIANTI database. recombrates.txt has no rates for each other ion with no Nahar file, and ARTIS then uses its own rates for that ion.
- `makeartischargetransferfile` writes chargetransfer.txt, the charge transfer rate file. The default output folder is `artis_files/data` in the repository, and not in the working directory.
- `makeartisgammaspecfiles` downloads the NuDat3 decay tables and writes a gamma spectrum for each nuclide that has gamma lines. It needs a network connection and has no option. It writes the files into `artis_files/data` in the repository.

The page [Commands](https://artisatomic.readthedocs.io/en/latest/commands.html) gives the details of each command.

### Tab completion
The commands with options use [argcomplete](https://kislyuk.github.io/argcomplete/). To complete the options of a command in bash or zsh, run this line:

```sh
eval "$(register-python-argcomplete makeartisatomicfiles)"
```

To use the completion in each new shell, add the line to the startup file of the shell.

## Development
Install the development tools:
```sh
uv sync --frozen
```

Then install the pre-commit hooks one time:
```sh
prek install
```

With pip, the command is `python3 -m pip install --group dev -e .`. The page [Development](https://artisatomic.readthedocs.io/en/latest/development.html) gives the commands for the tests and the checks.

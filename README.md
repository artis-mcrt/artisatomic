# artisatomic
[![Build and test](https://github.com/artis-mcrt/artisatomic/actions/workflows/test.yml/badge.svg)](https://github.com/artis-mcrt/artisatomic/actions/workflows/test.yml)

>The Python package converts atomic data from several sources (for example CMFGEN, Kurucz, JPLT, ADAS) into the ARTIS format.

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
Run "makeartisatomicfiles" at the command-line to create adata.txt, compositiondata.txt, phixsdata_v2.txt, and transitiondata.txt. The tool has no configuration interface for the ion selection. To change ions or data sources, edit the Python code or supply an `artisatomicionhandlers.json` file. The options `-minionstage` (default 1), `-maxionstage` (default 5) and `-maxatomicnumber` (no limit) also limit the built-in ion selection.

### Comment blocks in the output files
adata.txt, transitiondata.txt and phixsdata_v2.txt have a block of `#` comment lines before the data of each ion. The block gives the handler, the data source with its reference, the source files, and the counts and warnings for that file. The log files in `atomic_data_logs` hold the same lines and more detail. compositiondata.txt has no comment, because ARTIS reads it with no comment skip.

ARTIS v2023.10 and later skip these comment lines. An older ARTIS release stops on them. Remove them for such a release, for example with `grep -v '^#' adata.txt`. artistools needs a version that skips the comment lines.

The package installs three more commands:

- `makeartisrecombratefile` writes recombrates.txt from the Nahar recombination rate files. An ion with no Nahar file takes the ChiantiPy rates, which need the `chianti` extra (`uv sync --frozen --extra chianti`).
- `makeartischargetransferfile` writes chargetransfer.txt, the charge transfer rate file.
- `makeartisgammaspecfiles` downloads the NuDat3 decay tables and writes a gamma spectrum for each nuclide that has gamma lines.

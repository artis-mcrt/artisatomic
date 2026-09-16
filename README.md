# artisatomic
[![Build and test](https://github.com/artis-mcrt/artisatomic/actions/workflows/test.yml/badge.svg)](https://github.com/artis-mcrt/artisatomic/actions/workflows/test.yml)

>The Python package converts atomic data from several sources (for example CMFGEN, Kurucz, JPLT, QUB) into the ARTIS format.

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
Run "makeartisatomicfiles" at the command-line to create adata.txt, compositiondata.txt, phixsdata_v2.txt, and transitiondata.txt. The tool is not user friendly by design. To change ions or data sources, edit the Python code or supply an `artisatomicionhandlers.json` file. The options `-minionstage` (default 1), `-maxionstage` (default 5) and `-maxatomicnumber` (no limit) also limit the built-in ion selection.

The package installs three more commands:

- `makeartisrecombratefile` writes recombrates.txt from the Nahar recombination rate files. An ion with no Nahar file takes the ChiantiPy rates, which need the `chianti` extra (`uv sync --frozen --extra chianti`).
- `makeartischargetransferfile` writes chargetransfer.txt, the charge transfer rate file.
- `makeartisgammaspecfiles` downloads the NuDat3 decay tables and writes a gamma spectrum for each nuclide that has gamma lines.

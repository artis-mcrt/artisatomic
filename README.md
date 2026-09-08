# artisatomic
[![Build and test](https://github.com/artis-mcrt/artisatomic/actions/workflows/test.yml/badge.svg)](https://github.com/artis-mcrt/artisatomic/actions/workflows/test.yml)

>The python package converts atomic data into ARTIS format from several sources (e.g., CMFGEN, NORAD)

## Installation and Development
First clone the repository, for example:
```sh
git clone https://github.com/artis-mcrt/artisatomic.git
cd artisatomic
prek install
```

To use a uv project virtual environment with locked dependency versions run:
```sh
uv sync --frozen
source .venv/bin/activate
uv pip install -e .[dev]
```

Or to install into the system environment with pip:
```sh
python3 -m pip install --group dev -e .
```

## Usage
Run "makeartisatomicfiles" at the command-line to create adata.txt, compositiondata.txt, phixsdata_v2.txt, and transitiondata.txt. This code is not user friendly and requires manual editing of the Python scripts to change ions and data sources. The options `-minionstage` (default 1), `-maxionstage` (default 5) and `-maxatomicnumber` (no limit) also limit the built-in ion selection.

The package installs three more commands:

- `makeartisrecombratefile` writes recombrates.txt from the Nahar recombination rate files.
- `makeartischargetransferfile` writes the charge transfer rate files.
- `makeartisgammaspecfiles` downloads the ENDF decay data and writes a gamma spectrum for each nuclide.

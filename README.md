# artis-atomic

>The python package converts atomic data into ARTIS format from several sources (e.g., CMFGEN, NORAD)

## Installation
First clone the repository, for example:
```sh
git clone https://github.com/artis-mcrt/artisatomic.git
cd artistatomic
```

To use a uv project virtual environment with locked dependency versions run:
```sh
uv sync --frozen
source .venv/bin/activate
uv pip install -e .[dev]
```

Or to install into the system environment with pip:
```sh
python3 -m pip install -e .[dev]
```

## Usage
Run "makeartisatomicfiles" at the command-line to create adata.txt, compositiondata.txt, phixsdata_v2.txt, and transitiondata.txt. This code is not user friendly and requires manual editing of the Python scripts to change ions and data sources.

[![Build and test](https://github.com/artis-mcrt/artisatomic/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/artis-mcrt/artisatomic/actions/workflows/pythonapp.yml)

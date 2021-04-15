# artis-atomic

>The python package converts atomic data into ARTIS format from several sources (e.g., CMFGEN, NORAD)

## Installation
First clone the repository, for example:
```sh
git clone https://github.com/artis-mcrt/artisatomic.git
```
Then from within the repository directory run:
```sh
python3 -m pip install -e .
```

NOTE: if you get an error on macOS >= 11 when installing tables, run brew install hdf5 c-blosc

## Usage
Run "makeartisatomicfiles" at the command-line to create adata.txt, compositiondata.txt, phixsdata_v2.txt, and transitiondata.txt. This code is not user friendly at all are requires manual editing of the Python scripts to change ions and data sources.

[![Build and test](https://github.com/artis-mcrt/artisatomic/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/artis-mcrt/artisatomic/actions/workflows/pythonapp.yml)

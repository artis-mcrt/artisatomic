# Data sources

A handler reads the data of one ion from one data source. The repository holds a data folder for each data source, except FAC and Lisbon. Each data folder holds a README file. Some data folders also hold a download script. Most data folders do not hold the data itself.

`makeartisatomicfiles --listhandlers` prints each handler with its data source, and with the reference where the repository holds one.

## Handlers

The column "built-in" says if the built-in ion selection uses the handler. An ion handlers file can name each handler.

| handler | data source | data folder | built-in |
|---|---|---|---|
| `adas` | files in the ADAS adf04 format | `atomic-data-adas` | yes |
| `boyle` | the AOIFE helium data set | `atomic-data-helium-boyle` | no |
| `cmfgen` | the CMFGEN atomic data compilation of Hillier | `atomic-data-hillier` | yes |
| `cmfgen_qubphixs` | CMFGEN levels of Co II with the QUB cross sections | `atomic-data-hillier` and `atomic-data-adas` | no |
| `dream` | the DREAM database | `atomic-data-dream` | no |
| `fac` | FAC and cFAC output, an early version of the calibrated Floers+25 data. U II and U III come from the convergence study of Nd and U. | the folder that `ARTISATOMIC_FAC_PATH` names, and the folder `Paper_Nd_U` beside it | no |
| `floers25calib` | the Floers+25 data set, calibrated | `atomic-data-floers25` | yes |
| `floers25calibwithforbidden` | a later version of the Floers+25 data set, which is not public | `atomic-data-floers25` | yes, if the private folder exists |
| `floers25uncalib` | the Floers+25 data set, uncalibrated. The handler reads the later version that is not public first, if the private folder exists. | `atomic-data-floers25` | yes, for an ion with no calibrated file |
| `gsnist` | the ground level of an ion, from the NIST Atomic Spectra Database | `atomic-data-groundstatesonlynist` | no |
| `kurucz` | the Kurucz gfall line lists | `atomic-data-kurucz` | yes |
| `lisbon` | the Lisbon Atomic Group data set of Nd II–III and U II–III | `atomic-data-lisbon`, or the folder that `ARTISATOMIC_LISBON_PATH` names | no |
| `mons` | the University of Mons data set of the lanthanides V–VII | `atomic-data-mons` | no |
| `tanakajplt` | the Japan-Lithuania opacity database for kilonovae | `atomic-data-tanaka-jplt` | yes |

The `handlers` registry in `artisatomic/iondata.py` gives the reader of each handler.

## Get the data

Run each command in the data folder of the handler, unless the step names a different folder.

| handler | step | size |
|---|---|---|
| `adas` | Git tracks the Ca III file and the Sr I file. The repository does not hold the Co and Fe files. ARTIS members get them from the shared drive of the collaboration. | |
| `boyle` | Save the data set as `aoife.hdf5`. `README.txt` of the folder holds a link to the data set. Then run `artisatomic/hdf5_to_parquet.py atomic-data-helium-boyle/aoife.hdf5` in the repository root. | |
| `cmfgen` | `. ./setup_cmfgen_data.sh` | 363 MiB download, 1.8 GiB after the extraction |
| `cmfgen_qubphixs` | Do the step of `cmfgen`. Then get the Co files of `atomic-data-adas`, as the row of `adas` says. | |
| `dream` | Make the HDF5 file as `README.md` of the folder says. Then run `artisatomic/hdf5_to_parquet.py` on that file in the repository root. | |
| `fac` | Set `ARTISATOMIC_FAC_PATH` to the folder that holds the folder `OptimizedFAC_lanthanides_calibrated`. U II and U III also need the folder `Paper_Nd_U` beside the folder that `ARTISATOMIC_FAC_PATH` names. The repository does not hold the data. | |
| `floers25calib`, `floers25uncalib` | `./setup_floers25_data.sh` | 1.6 GiB download. The extraction needs about 7 GiB of free space. |
| `floers25calibwithforbidden` | The data is not public. `README.txt` of the folder says where the folder `OutputFiles_withforbidden` must be. | |
| `gsnist` | No step. Git tracks `groundstates.dat`. | |
| `kurucz` | Download the zztar files with the `curl` and `tar` commands in `README.txt` of the folder. The built-in ion selection also needs Sr III and Zr III. Download `gf3802.lines` and `gf4002.lines` into `extendedatoms`, as `README.txt` says. | 84 MiB download, and 26 MiB for the two files of `extendedatoms` |
| `lisbon` | Put the data into `atomic-data-lisbon`. As an alternative, set `ARTISATOMIC_LISBON_PATH` to the folder of the data. The repository does not hold the data. | |
| `mons` | `./setup_mons_data.sh` | 20.2 GiB download |
| `tanakajplt` | `. ./setup_jplt_data.sh` | 1.4 GiB download, 0.7 GiB after the extraction |

The scripts `setup_floers25_data.sh` and `setup_mons_data.sh` start with `#!/usr/bin/env zsh`. On a computer with no zsh, run each script with bash. For example: `bash setup_floers25_data.sh`.

Each download script calls these tools. Install them before you run the script.

| script | tools |
|---|---|
| `setup_cmfgen_data.sh` | `curl`, `md5sum`, `tar` with `xz`, `iconv`, `zstd` |
| `setup_floers25_data.sh` | `curl`, `unzip`, `zstd` |
| `setup_mons_data.sh` | `curl`, `unzip` |
| `setup_jplt_data.sh` | `curl`, `md5sum`, `tar`, `zstd` |
| `setup_chargetransfer_data.sh` in `atomic-data-chargetransfer` | `curl` |

`artisatomic/hdf5_to_parquet.py` starts with a `uv run --script` line, so [uv](https://docs.astral.sh/uv/) installs its dependencies. The page [Commands](commands.md) shows its help text.

The README file of each data folder gives the reference of the data set and the layout of the files.

An ion with no cross sections in its data source gets a hydrogenic estimate (see the option `-nlevels_hydrogenic_for_unknown_phixs`). The estimate reads the hydrogen tables of the CMFGEN data set. Thus a run of a different handler can also need the CMFGEN data set. If the data set is absent, the run stops with the name of the download script. `--nophixs` gives no estimate. A handler with no parser for its level names, for example `mons`, also gives no estimate.

## Environment variables

| variable | effect |
|---|---|
| `ARTISATOMIC_FAC_PATH` | The folder that holds the folder `OptimizedFAC_lanthanides_calibrated`. The folder `Paper_Nd_U` of U II and U III is beside this folder. The default is a shared drive folder of the collaboration. |
| `ARTISATOMIC_LISBON_PATH` | The folder that holds the Lisbon data. The default is `atomic-data-lisbon` in the repository. Git does not track that folder. |
| `ARTISATOMIC_TESTMODE` | The value `1` starts the test mode. The Kurucz, ADAS, MONS and Floers+25 readers then read the samples in the repository. Git tracks the Floers+25 sample only as `atomic-data-floers25/testdata.tar.xz`. Extract that archive first. The file comments get a creation time of 1970-01-01T00:00:00Z. |
| `SOURCE_DATE_EPOCH` | The creation time of the file comments, in seconds since 1970. The test mode comes before this variable. |

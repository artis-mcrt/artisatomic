# Data sources

A handler reads the data of one ion from one data source. The repository holds a data folder for each data source. Most data folders hold a download script and a README file, and not the data itself.

`makeartisatomicfiles --listhandlers` prints each handler with the full reference of its data source.

## Handlers

The column "built-in" says if the built-in ion selection uses the handler. An ion handlers file can name each handler.

| handler | data source | data folder | built-in |
|---|---|---|---|
| `adas` | files in the ADAS adf04 format | `atomic-data-adas` | yes |
| `boyle` | the AOIFE helium data set | `atomic-data-helium-boyle` | no |
| `cmfgen` | the CMFGEN atomic data compilation of Hillier | `atomic-data-hillier` | yes |
| `cmfgen_qubphixs` | CMFGEN levels of Co II with the QUB cross sections | `atomic-data-hillier` and `atomic-data-adas` | no |
| `dream` | the DREAM database | `atomic-data-dream` | no |
| `fac` | FAC and cFAC output, an early version of the calibrated Floers+25 data | a folder outside the repository | no |
| `floers25calib` | the Floers+25 data set, calibrated | `atomic-data-floers25` | yes |
| `floers25calibwithforbidden` | a later version of the Floers+25 data set, which is not public | `atomic-data-floers25` | yes, if the private folder exists |
| `floers25uncalib` | the Floers+25 data set, uncalibrated | `atomic-data-floers25` | yes, for an ion with no calibrated file |
| `gsnist` | the ground level of an ion, from the NIST Atomic Spectra Database | `atomic-data-groundstatesonlynist` | no |
| `kurucz` | the Kurucz gfall line lists | `atomic-data-kurucz` | yes |
| `lisbon` | the Lisbon Atomic Group data set of Nd II–III and U II–III | a folder outside the repository | no |
| `mons` | the University of Mons data set of the lanthanides V–VII | `atomic-data-mons` | no |
| `tanakajplt` | the Japan-Lithuania opacity database for kilonovae | `atomic-data-tanaka-jplt` | yes |

The `handlers` registry in `artisatomic/iondata.py` gives the reader of each handler.

## Get the data

Each command in this table runs in the data folder of the handler.

| handler | step | size |
|---|---|---|
| `adas` | Git tracks the Ca III file and the Sr I file. ARTIS members get the Co and Fe files from the shared drive of the collaboration. | |
| `boyle` | Save the data set as `aoife.hdf5`. Then convert it with `artisatomic/hdf5_to_parquet.py`. | |
| `cmfgen` | `. ./setup_cmfgen_data.sh` | 363 MB download, 1.8 GB on disk |
| `dream` | Make the HDF5 file as `README.md` of the folder says. Then convert it with `artisatomic/hdf5_to_parquet.py`. | |
| `fac` | Set `ARTISATOMIC_FAC_PATH` to the folder of the data. The repository does not hold the data. | |
| `floers25calib`, `floers25uncalib` | `./setup_floers25_data.sh` | 5.5 GiB download |
| `gsnist` | No step. Git tracks `groundstates.dat`. | |
| `kurucz` | Download the files with the `curl` and `tar` commands in `README.txt` of the folder. | 88 MB download |
| `lisbon` | Set `ARTISATOMIC_LISBON_PATH` to the folder of the data. The repository does not hold the data. | |
| `mons` | `./setup_mons_data.sh` | 21.7 GB download |
| `tanakajplt` | `. ./setup_jplt_data.sh` | 0.7 GB on disk |

The README file of each data folder gives the reference of the data set and the layout of the files.

An ion with no cross sections in its data source gets a hydrogenic estimate (see the option `-nlevels_hydrogenic_for_unknown_phixs`). The estimate reads the hydrogen tables of the CMFGEN data set. Thus a run of a different handler can also need the CMFGEN data set. The run then stops with the name of the setup script if the data set is absent. `--nophixs` gives no estimate. A handler with no parser for its level names, for example `mons`, also gives no estimate.

## Environment variables

| variable | effect |
|---|---|
| `ARTISATOMIC_FAC_PATH` | The folder that holds the FAC data. The default is a shared drive folder of the collaboration. |
| `ARTISATOMIC_LISBON_PATH` | The folder that holds the Lisbon data. The default is `atomic-data-lisbon` in the repository. |
| `ARTISATOMIC_TESTMODE` | The value `1` starts the test mode. The Kurucz, ADAS, MONS and Floers+25 readers then read the samples that Git tracks. The file comments get a creation time of 1970-01-01T00:00:00Z. |
| `SOURCE_DATE_EPOCH` | The creation time of the file comments, in seconds since 1970. The test mode comes before this variable. |

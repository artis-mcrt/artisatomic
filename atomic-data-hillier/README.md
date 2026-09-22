CMFGEN atomic data compilation of D. J. Hillier: <http://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm>.
Paper: Hillier, D. J., Miller, D. L. (1998), ApJ, 496, 407-427, doi:10.1086/305350.

The readers use the atomic_21jun23 data set only. `setup_cmfgen_data.sh` downloads
`atomic_data_21jun23.tar.xz` from the GitHub release of this repository, checks the MD5 sum,
extracts it to `atomic_21jun23/`, and converts the iso-8859-1 files to utf-8:

```sh
. ./setup_cmfgen_data.sh
```

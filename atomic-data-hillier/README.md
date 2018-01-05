Download the atomic data from
<http://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm>

As of Jan 5, 2018 the latest version is at:
<http://kookaburra.phyast.pitt.edu/hillier/cmfgen_files/atomic_data_15nov16.tar.gz>

From the command-line you can download the file with curl.
```sh
curl -O http://kookaburra.phyast.pitt.edu/hillier/cmfgen_files/atomic_data_15nov16.tar.gz
```

Extract the tarball in this directory (which should go into the subfolder called 'atomic').

```sh
tar -xvzf atomic_data_15nov16.tar.gz
```
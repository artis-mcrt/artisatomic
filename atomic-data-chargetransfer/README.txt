Source files of the charge transfer rates for makeartischargetransferfile. Git tracks all of them.

setup_chargetransfer_data.sh downloads the files again. Run it in this folder:

    ./setup_chargetransfer_data.sh

The files and their references:

- ctrecombdata.dat and ctiondata.dat: the Cloudy master data files (gitlab.nublado.org). They carry
  the fits of Kingdon, J. B., Ferland, G. J. (1996), ApJS, 106, 205-211, doi:10.1086/192335, with
  later updates of Cloudy.
- ar85_ct2.dat: the table of Arnaud, M., Rothenflug, R. (1985), A&AS, 60, 425-457,
  bibcode 1985A&AS...60..425A, from the ASCII file ct2.dat of D. Verner.
- ss11_table4.dat and ss11_table5.dat: the CDS tables of Sterling, N. C., Stancil, P. C. (2011),
  A&A, 535, A117, doi:10.1051/0004-6361/201117584.

The docstring of artisatomic/makechargetransferfile.py explains how the program uses each file.

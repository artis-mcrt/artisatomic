Atomic data in the ADAS adf04 format. This directory had the name atomic-data-qub before.
makeartisatomicfiles moves the files of an atomic-data-qub directory into this directory. It does
not replace a file, and it does not move the data of a symbolic link.

The data has two origins:
- Authors at QUB (Queen's University Belfast) made the Co files (co_tyndall), the Sr I file and the
  Fe files. Most of them come from a private communication of the QUB group.
- The Ca III file 20_3.adf04.zst comes from OPEN-ADAS (https://open.adas.ac.uk). The file 20_3.txt
  gives its address.

ARTIS members find the files that Git does not track in the Google Drive folder.

The Sr I file 38_1.adf04.zst has its reference in 38_1.bib: Dougan, D. J., McElroy, N. E.,
Ballance, C. P., Ramsbottom, C. A. (2025), MNRAS, 541, 367-383, doi:10.1093/mnras/staf1013.

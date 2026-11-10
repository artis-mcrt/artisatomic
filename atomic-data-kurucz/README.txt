The Kurucz gfall line lists for the "kurucz" handler.
Reference: Kurucz, R. L. (2017), Can. J. Phys., 95, 825-827, doi:10.1139/cjp-2016-0794.

readkuruczdata.py looks for the file of an ion in this order. ZZ is the atomic number and CC is
the ion charge, so gf3801 is Sr II.

1. extendedatoms/gfZZCC.lines
2. extendedatoms/gfZZCCz.lines
3. zztar/gfZZCC.all

The reader also finds a compressed form of each name, for example gf3801.lines.zst.

The extendedatoms files come from the folder of each ion on the Kurucz web site, for example
http://kurucz.harvard.edu/atoms/3801/. extendedatoms/url.txt gives the address of the parent
folder. Download gfZZCC.lines, or gfZZCCz.lines if the folder has no gfZZCC.lines.

The zztar files come from one tar file of 88 MB with 265 gfZZCC.all files. url.txt gives its
address. Download it and extract it into zztar/:

    curl -o zztar.tar http://kurucz.harvard.edu/linelists/gfall/zztar
    mkdir -p zztar && tar -xf zztar.tar -C zztar

Git does not track the downloaded files. .gitignore lists them.

The built-in ion selection reads Sr II-III, Y I-II and Zr I-III with this handler. It reads Sr I
with the adas handler when atomic-data-adas holds the Sr I file, and with this handler if not. With
ARTISATOMIC_TESTMODE=1 the reader uses test_sample/ instead. That folder holds Sr I-II and
Y I-II for the kurucz checksum set in tests/.

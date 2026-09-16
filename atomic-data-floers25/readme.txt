Paper:
Flörs, A., da Silva, R. F., Marques, J. P., Sampaio, J. M., & Martínez-Pinedo, G. (2026).
Calibrated lanthanide atomic data for kilonova radiative transfer: Atomic structure and opacities.
Physical Review D, 113(6), 063041. https://doi.org/10.1103/jxqw-7ynk
https://ui.adsabs.harvard.edu/abs/2026PhRvD.113f3041F/abstract

Data set:
Flörs, A. GSI Database for Kilonova Radiative Transfer. Zenodo.
All versions (resolves to the latest): https://doi.org/10.5281/zenodo.15835360

setup_floers25_data.sh and url.txt download record 19335084.
The earlier record 15835361 (DOI 10.5281/zenodo.15835361) is the same dataset, one version back.

setup_floers25_data.sh creates OutputFiles from the Zenodo archives. The reader also looks for a
private directory OutputFiles_withforbidden, which holds the newer data with forbidden transitions.
Link it from the shared drive when you have access, e.g.
ln -s "/path/to/Shared drives/Floers Forbidden Lines Prerelease do not share" OutputFiles_withforbidden
ARTISATOMIC_TESTMODE=1 makes the reader use test_sample/ from testdata.tar.xz instead of both.

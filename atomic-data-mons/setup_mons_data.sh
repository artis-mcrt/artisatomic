#!/usr/bin/env zsh

set -x

# outggf_Ln_V--VII.zip is 21.7 GB. Both archives are read in place and do not need to be extracted.
if [ ! -f outglv_Ln_V--VII.zip ]; then curl -O https://zenodo.org/records/10635803/files/outglv_Ln_V--VII.zip; fi
if [ ! -f outggf_Ln_V--VII.zip ]; then curl -O https://zenodo.org/records/10635803/files/outggf_Ln_V--VII.zip; fi

set +x

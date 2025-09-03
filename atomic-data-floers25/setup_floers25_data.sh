#!/usr/bin/env zsh

if [[ ! -f OutputFiles/57LaII_levels_calib.txt && ! -f OutputFiles/57LaII_levels_calib.txt.zst ]]; then

  if [[ ! -f GSI_lanthanides_calibrated_levels.zip ]]; then curl -O https://zenodo.org/records/15835361/files/GSI_lanthanides_calibrated_levels.zip?download=1; fi

  md5sum -c GSI_lanthanides_calibrated_levels.zip.md5
  unzip -j -d OutputFiles GSI_lanthanides_calibrated_levels.zip
  zstd --rm -f -T0 -15 -v OutputFiles/*.txt

fi

if [[ ! -f OutputFiles/57LaII_transitions_calib.txt && ! -f OutputFiles/57LaII_transitions_calib.txt.zst ]]; then

  if [[ ! -f GSI_lanthanides_calibrated_transitions.zip ]]; then curl -O https://zenodo.org/records/15835361/files/GSI_lanthanides_calibrated_transitions.zip?download=1; fi

  md5sum -c GSI_lanthanides_calibrated_transitions.zip.md5
  unzip -j -d OutputFiles GSI_lanthanides_calibrated_transitions.zip
  zstd --rm -f -T0 -15 -v OutputFiles/*.txt

fi

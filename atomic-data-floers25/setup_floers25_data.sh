#!/usr/bin/env zsh

if [[ ! -f OutputFiles/57LaII_levels_calib.txt && ! -f OutputFiles/57LaII_levels_calib.txt.zst ]]; then

  if [[ ! -f GSI_lanthanides_calibrated_levels.zip ]]; then curl -O https://zenodo.org/records/19335084/files/GSI_lanthanides_calibrated_levels.zip?download=1; fi

  unzip -j -o -d OutputFiles GSI_lanthanides_calibrated_levels.zip
  rm -f OutputFiles/.DS_Store OutputFiles/.*
  zstd --rm -f -T0 -15 -v OutputFiles/*.txt

fi

if [[ ! -f OutputFiles/57LaII_transitions_calib.txt && ! -f OutputFiles/57LaII_transitions_calib.txt.zst ]]; then

  if [[ ! -f GSI_lanthanides_calibrated_transitions.zip ]]; then curl -O https://zenodo.org/records/19335084/files/GSI_lanthanides_calibrated_transitions.zip?download=1; fi

  unzip -j -o -d OutputFiles GSI_lanthanides_calibrated_transitions.zip
  rm -f OutputFiles/.DS_Store OutputFiles/.*
  zstd --rm -f -T0 -15 -v OutputFiles/*.txt

fi

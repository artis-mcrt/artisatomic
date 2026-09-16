#!/usr/bin/env zsh

set -x

if [ ! -f data_v2.1/33_2.txt.zst ]; then

  if [ ! -f data_v2.0.tar.gz ]; then curl -O http://dpc.nifs.ac.jp/DB/Opacity-Database/data/data_v2.0.tar.gz; fi
  md5sum -c data_v2.0.tar.gz.md5

  if [ ! -f grasp_v2.1.tar.gz ]; then curl -O http://dpc.nifs.ac.jp/DB/Opacity-Database/data/grasp_v2.1.tar.gz; fi
  md5sum -c grasp_v2.1.tar.gz.md5

  mkdir -p data_v2.1

  # v2.1 holds only some ions, so the script extracts it over the v2.0 files
  tar -xvf data_v2.0.tar.gz -C data_v2.1 --strip-components=1

  tar -xvf grasp_v2.1.tar.gz -C data_v2.1 --strip-components=1

fi
zstd --rm -f -T0 -v -5 data_v2.1/*.txt || true

set +x

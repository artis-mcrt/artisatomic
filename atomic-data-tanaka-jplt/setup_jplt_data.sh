#!/usr/bin/env zsh

set -x

if [ ! -f data_v1.1/26_1.txt.zst ]; then

  if [ ! -f data_v1.1.tar.gz ]; then curl -O http://dpc.nifs.ac.jp/DB/Opacity-Database/data/data_v1.1.tar.gz; fi

  mkdir -p data_v1.1
  md5sum -c data_v1.1.tar.gz.md5
  tar -xf data_v1.1.tar.gz -C data_v1.1
  zstd --rm -f -v data_v1.1/*.txt

fi

set +x
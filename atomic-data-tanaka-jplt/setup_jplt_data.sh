#!/usr/bin/env zsh

# if [ ! -f data_v1.1/26_1.txt.zst ]; then

#   if [ ! -f data_v1.1.tar.gz ]; then curl -O http://dpc.nifs.ac.jp/DB/Opacity-Database/data/data_v1.1.tar.gz; fi

#   mkdir -p data_v1.1
#   md5sum -c data_v1.1.tar.gz.md5
#   tar -xf data_v1.1.tar.gz -C data_v1.1

# fi
# zstd --rm -f -T0 -12 -v data_v1.1/*.txt

if [ ! -f data_v2.0/26_1.txt.zst ]; then

  if [ ! -f data_v2.0.tar.gz ]; then curl -O http://dpc.nifs.ac.jp/DB/Opacity-Database/data/data_v2.0.tar.gz; fi

  mkdir -p data_v2.0
  md5sum -c data_v2.0.tar.gz.md5
  tar -xf data_v2.0.tar.gz

fi
zstd --rm -f -T0 -12 -v data_v2.0/*.txt

# v2.1 is not complete, so should be overlaid on v2.0 files
if [ ! -f data_v2.1/26_1.txt.zst ]; then

    rm -rf data_v2.1
    mkdir -p data_v2.1
    rsync -av data_v2.0/ data_v2.1/

fi

if [ ! -f data_v2.1/33_2.txt.zst ]; then

  if [ ! -f grasp_v2.1.tar.gz ]; then curl -O http://dpc.nifs.ac.jp/DB/Opacity-Database/data/grasp_v2.1.tar.gz; fi

  mkdir -p data_v2.1
  md5sum -c grasp_v2.1.tar.gz.md5
  tar -xf grasp_v2.1.tar.gz --transform 's!^grasp_v2.1\($\|/\)!data_v2.1\1!'

fi
zstd --rm -f -T0 -12 -v data_v2.1/*.txt

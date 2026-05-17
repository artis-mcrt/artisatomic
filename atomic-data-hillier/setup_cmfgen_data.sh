#!/usr/bin/env zsh

set -x

# version="15nov16"
version="21jun23"

if [ ! -f atomic_data_$version.tar.xz ]; then curl -O -L https://github.com/artis-mcrt/artisatomic/releases/download/v2026.5.17/atomic_data_$version.tar.xz; fi

md5sum -c atomic_data_$version.tar.xz.md5
tar -xJf atomic_data_$version.tar.xz
mv atomic/ atomic_$version/
# rsync -a atomic_diff/ atomic_$version/

find atomic_$version ! -name "*.zst" -size +10M -exec zstd -12 -v -T0 --rm {} \; || true

set +x

#!/usr/bin/env zsh

set -x

if [ ! -f atomic_data_15nov16.tar.xz ]; then curl -O https://theory.gsi.de/~lshingle/artis_http_public/artisatomic/atomic-data-hillier/atomic_data_15nov16.tar.xz; fi

md5sum -c atomic_data_15nov16.tar.xz.md5
tar -xf atomic_data_15nov16.tar.xz
rsync -a atomic_diff/ atomic/

set +x
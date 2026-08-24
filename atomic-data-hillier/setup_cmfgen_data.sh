#!/usr/bin/env zsh

set -x

# version="15nov16"
version="21jun23"

if [ ! -f atomic_data_$version.tar.xz ]; then curl -O -L https://github.com/artis-mcrt/artisatomic/releases/download/v2026.5.17/atomic_data_$version.tar.xz; fi

md5sum -c atomic_data_$version.tar.xz.md5
tar -xJf atomic_data_$version.tar.xz
mv atomic/ atomic_$version/
# rsync -a atomic_diff/ atomic_$version/

# CMFGEN writes an author's name with an accent, which leaves a few files in iso-8859-1.
# The readers expect utf-8, so convert those files once, here.
find atomic_$version -type f -print0 | while IFS= read -r -d "" datafile; do
  # a file that holds a NUL byte is not text, e.g. a PDF, an object file or a .DS_Store
  LC_ALL=C tr -d "\000" < "$datafile" | cmp -s - "$datafile" || continue
  iconv -f utf-8 -t utf-8 "$datafile" > /dev/null 2>&1 && continue
  iconv -f iso-8859-1 -t utf-8 "$datafile" > "$datafile.utf8" && mv "$datafile.utf8" "$datafile"
  echo "converted $datafile to utf-8"
done

find atomic_$version ! -name "*.zst" -size +10M -exec zstd -12 -v -T0 --rm {} \; || true

set +x

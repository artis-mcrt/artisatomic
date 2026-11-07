#!/usr/bin/env zsh

# Download the CMFGEN atomic data from the GitHub release of this repository, check its MD5 sum,
# and extract it to atomic_21jun23. Run the script from this folder. CI sources it with bash, so a
# failed step ends with "return 1" and not with set -e, which would also stop a shell that sources it.

set -x

version="21jun23"
archive="atomic_data_$version.tar.xz"

if [ ! -f "$archive" ]; then
  # -f: an HTTP error must not leave an error page under the name of the archive
  curl -fL --retry 3 -o "$archive.part" "https://github.com/artis-mcrt/artisatomic/releases/download/v2026.5.17/$archive" \
    || { rm -f "$archive.part"; echo "ERROR: the download of $archive failed" >&2; return 1 2>/dev/null || exit 1; }
  mv "$archive.part" "$archive"
fi

md5sum -c "$archive.md5" \
  || { echo "ERROR: $archive has a wrong MD5 sum. Delete it and run this script again." >&2; return 1 2>/dev/null || exit 1; }

# The mark is the last step. A script that stops before it does the remaining steps at the next
# run. Each step after the extraction skips the files that it already changed.
marker="atomic_$version/setup_complete.txt"
if [ -f "$marker" ]; then
  echo "$marker exists, so the script does not extract $archive again. Delete the folder to extract it again."
else
  if [ ! -d "atomic_$version" ]; then
    tar -xJf "$archive" || { echo "ERROR: the extraction of $archive failed" >&2; return 1 2>/dev/null || exit 1; }
    mv atomic/ "atomic_$version/"
  fi

  # CMFGEN writes an author's name with an accent, which leaves a few files in iso-8859-1.
  # The readers expect utf-8, so convert those files once, here. The loop runs in a subshell in
  # bash, so it records a failed conversion in a file and not in a variable.
  failedlist="atomic_$version.conversion_failed.txt"
  rm -f "$failedlist"
  find "atomic_$version" -type f -print0 | while IFS= read -r -d "" datafile; do
    # a file that holds a NUL byte is not text, e.g. a PDF, an object file or a .DS_Store
    LC_ALL=C tr -d "\000" < "$datafile" | cmp -s - "$datafile" || continue
    iconv -f utf-8 -t utf-8 "$datafile" > /dev/null 2>&1 && continue
    if iconv -f iso-8859-1 -t utf-8 "$datafile" > "$datafile.utf8" && mv "$datafile.utf8" "$datafile"; then
      echo "converted $datafile to utf-8"
    else
      rm -f "$datafile.utf8"
      echo "$datafile" >> "$failedlist"
    fi
  done
  if [ -s "$failedlist" ]; then
    echo "ERROR: the conversion to utf-8 failed for the files in $failedlist. Run this script again." >&2
    return 1 2>/dev/null || exit 1
  fi

  find "atomic_$version" ! -name "*.zst" -size +10M -exec zstd -12 -v -T0 --rm {} \; || true

  echo "The setup of atomic_$version is complete." > "$marker"
fi

set +x

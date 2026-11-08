#!/usr/bin/env zsh

# Download the JPLT data (versions 2.0 and 2.1) and extract it to data_v2.1. Run the script from
# this folder. CI sources it with bash, so a failed step ends with "return 1" and not with set -e,
# which would also stop a shell that sources it.

set -x

if [ ! -f data_v2.1/33_2.txt.zst ]; then

  for archive in data_v2.0.tar.gz grasp_v2.1.tar.gz; do
    if [ ! -f "$archive" ]; then
      # -f: an HTTP error must not leave an error page under the name of the archive
      curl -fL --retry 3 -o "$archive.part" "http://dpc.nifs.ac.jp/DB/Opacity-Database/data/$archive" \
        || { rm -f "$archive.part"; echo "ERROR: the download of $archive failed" >&2; return 1 2>/dev/null || exit 1; }
      mv "$archive.part" "$archive"
    fi
    md5sum -c "$archive.md5" \
      || { echo "ERROR: $archive has a wrong MD5 sum. Delete it and run this script again." >&2; return 1 2>/dev/null || exit 1; }
  done

  mkdir -p data_v2.1

  # v2.1 holds only some ions, so the script extracts it over the v2.0 files
  tar -xvf data_v2.0.tar.gz -C data_v2.1 --strip-components=1 \
    || { echo "ERROR: the extraction of data_v2.0.tar.gz failed" >&2; return 1 2>/dev/null || exit 1; }

  # The transition list of grasp_v2.1/33_2.txt (As II) does not use the level numbers of its level
  # list, so the v2.0 file of As II stays. readtanakajpltdata.py stops on the v2.1 file.
  tar -xvf grasp_v2.1.tar.gz -C data_v2.1 --strip-components=1 --exclude='*/33_2.txt' \
    || { echo "ERROR: the extraction of grasp_v2.1.tar.gz failed" >&2; return 1 2>/dev/null || exit 1; }

fi
# find and not a glob: zsh stops a script at a glob that matches no file
find data_v2.1 -maxdepth 1 -name '*.txt' -type f -exec zstd --rm -f -T0 -v -5 {} +

set +x

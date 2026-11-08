#!/usr/bin/env zsh

# Download the public Floers+25 data (Zenodo record 19335084) into OutputFiles, and compress each
# table with zstd. The floers25calib handler reads the calibrated files, and floers25uncalib reads
# the uncalibrated files. An archive counts as done only after its last step, so run the script
# again after a stop. To download an archive again, delete its marker OutputFiles/.<archive>.done.

set -e
cd "$(dirname "$0")"

record="https://zenodo.org/records/19335084/files"
mkdir -p OutputFiles

for archive in GSI_lanthanides_calibrated_levels GSI_lanthanides_calibrated_transitions \
  GSI_lanthanides_uncalibrated_levels GSI_lanthanides_uncalibrated_transitions; do
  if [[ -f "OutputFiles/.$archive.done" ]]; then
    echo "$archive is done, skipped"
    continue
  fi
  if [[ ! -f "$archive.zip" ]]; then
    echo "downloading $archive.zip"
    curl -fL --retry 3 -o "$archive.zip.part" "$record/$archive.zip?download=1"
    # a server that sends an error page with a 200 status must not leave a bad file behind
    if ! unzip -tqq "$archive.zip.part" > /dev/null 2>&1; then
      rm -f "$archive.zip.part"
      echo "ERROR: $archive.zip is not a zip archive" >&2
      exit 1
    fi
    mv "$archive.zip.part" "$archive.zip"
  fi
  # -j puts each table directly into OutputFiles. The __MACOSX members are AppleDouble files.
  unzip -j -o -q -d OutputFiles "$archive.zip" -x '__MACOSX/*'
  find OutputFiles -maxdepth 1 -name '*.txt' -type f -exec zstd --rm -f -q -T0 -15 {} +
  touch "OutputFiles/.$archive.done"
done

#!/usr/bin/env zsh

# Download the MONS lanthanide V-VII archives. artisatomic reads both archives in place, so do
# not extract them. outggf_Ln_V--VII.zip is 21.7 GB. To download a file again, delete it first.

set -e
cd "$(dirname "$0")"

download() {
  local file=$1
  if [ -f "$file" ]; then
    echo "$file exists, skipped"
    return
  fi
  echo "downloading $file"
  curl -fsSL --retry 3 -o "$file.part" "https://zenodo.org/records/10635803/files/$file"
  # a server that sends an error page with a 200 status must not leave a bad file behind
  if ! unzip -tqq "$file.part" > /dev/null 2>&1; then
    rm -f "$file.part"
    echo "ERROR: $file is not a zip archive" >&2
    exit 1
  fi
  mv "$file.part" "$file"
}

download outglv_Ln_V--VII.zip
download outggf_Ln_V--VII.zip

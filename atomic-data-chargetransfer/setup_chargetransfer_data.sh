#!/usr/bin/env zsh

# Download the source files of makechargetransferfile. The repository tracks the files, so the
# script is for an update. It skips a file that exists in plain or compressed form. To download
# a file again, delete it first.

set -e
cd "$(dirname "$0")"

download() {
  local file=$1
  local url=$2
  if [ -f "$file" ] || [ -f "$file.zst" ]; then
    echo "$file exists, skipped"
    return
  fi
  echo "downloading $file"
  curl -fsSL --retry 3 -o "$file.part" "$url"
  # a server that sends an error page with a 200 status must not leave a bad file behind. The
  # check skips blanks and a byte order mark before the first character.
  firstchar=$(head -c 64 "$file.part" | LC_ALL=C tr -d ' \t\r\n\357\273\277' | head -c 1)
  if [ "$firstchar" = "<" ]; then
    rm -f "$file.part"
    echo "ERROR: $url returned markup instead of data" >&2
    exit 1
  fi
  mv "$file.part" "$file"
}

# Cloudy: the fits of Kingdon & Ferland (1996) for reactions with hydrogen, plus the Cloudy updates
download ctrecombdata.dat "https://gitlab.nublado.org/cloudy/cloudy/-/raw/master/data/ctrecombdata.dat"
download ctiondata.dat "https://gitlab.nublado.org/cloudy/cloudy/-/raw/master/data/ctiondata.dat"

# Arnaud & Rothenflug (1985): recombination with neutral helium, from the file of D. Verner
download ar85_ct2.dat "https://www.pa.uky.edu/~verner/dima/ct/ct2.dat"

# Sterling & Stancil (2011): Ge, Se, Br, Kr, Rb, and Xe with hydrogen, from the CDS
download ss11_table4.dat "https://cdsarc.cds.unistra.fr/ftp/J/A+A/535/A117/table4.dat"
download ss11_table5.dat "https://cdsarc.cds.unistra.fr/ftp/J/A+A/535/A117/table5.dat"

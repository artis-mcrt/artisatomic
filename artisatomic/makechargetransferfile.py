#!/usr/bin/env python3
"""Write the file of charge transfer rates for ARTIS (data/chargetransfer.txt in the artis repository).

The script downloads the published fits and rate tables, converts them into one fit form, and
writes one file with one reaction for each line. The sources are:

- The Cloudy master data files ctrecombdata.dat and ctiondata.dat (gitlab.nublado.org). They carry
  the fits of Kingdon & Ferland (1996), ApJS, 106, 205 (KF96) for reactions with hydrogen, plus the
  later updates that Cloudy applies to individual reactions. The script compares each row against
  the KF96 paper tables (transcribed below) and names each update in the comment of its line.
- The table of Arnaud & Rothenflug (1985), A&AS, 60, 425 (AR85) for recombination with neutral
  helium, from the ASCII file ct2.dat of D. Verner (pa.uky.edu/~verner).
- The CDS tables of Sterling & Stancil (2011), A&A, 535, A117 (SS11) for the n-capture elements
  Ge, Se, Br, Kr, Rb, and Xe with hydrogen. SS11 publish tabulated k(T) values and no fit
  coefficients, so the script fits their tables over 1e3 to 4e4 K.
- Estimates for the reactions of a heavy ion with a neutral heavy atom. The kilonova ejecta
  hold no hydrogen and need these reactions. No publication gives these rates, so the script
  takes the ionization energies of Sc to U from the NIST ASD. It keeps the exothermic
  reactions with a small energy defect. Each one gets the near-resonant rate of 1e-9 cm3/s
  (Melius 1974). The reverse reactions come from detailed balance in ARTIS.

Every entry uses the KF96 fit form (their equation 7, from AR85):
  k = a * 1e-9 * t4^b * (1 + c * exp(d * t4)) * exp(-eexp/T)  [cm3/s],  t4 = T / 1e4 K.
See the header of the output file for the column definitions. The chargetransfer.cc module of ARTIS
reads the file and generates Landau-Zener estimates for the reactions that the file does not cover.
"""

import argparse
import math
import typing as t
from itertools import starmap
from pathlib import Path

import requests

from artisatomic.base import elsymbols

SOURCE_URLS = {
    "ctrecombdata.dat": "https://gitlab.nublado.org/cloudy/cloudy/-/raw/master/data/ctrecombdata.dat",
    "ctiondata.dat": "https://gitlab.nublado.org/cloudy/cloudy/-/raw/master/data/ctiondata.dat",
    "ar85_ct2.dat": "https://www.pa.uky.edu/~verner/dima/ct/ct2.dat",
    "ss11_table4.dat": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/535/A117/table4.dat",
    "ss11_table5.dat": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/535/A117/table5.dat",
    "nist_ie_sc_to_u.dat": (
        "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Sc-U&submit=Retrieve+Data"
        "&units=1&format=3&order=0&at_num_out=on&sp_name_out=on&ion_charge_out=on&e_out=0"
    ),
}

# KF96 Table 1 and the Table 2 totals, transcribed from the paper: (Z, q) -> (a, b, c, d).
# q is the ion charge before the electron capture, and a is in 1e-9 cm3/s. The script uses these
# values only to detect and name the Cloudy updates, so the temperature ranges are not stored.
KF96_REC = {
    (2, 1): (7.47e-6, 2.06, 9.93, -3.89),
    (2, 2): (1.00e-5, 0.00, 0.00, 0.00),
    (3, 2): (1.26, 0.96, 3.02, -0.65),
    (3, 3): (1.00e-5, 0.00, 0.00, 0.00),
    (4, 2): (1.00e-5, 0.00, 0.00, 0.00),
    (4, 3): (1.00e-5, 0.00, 0.00, 0.00),
    (4, 4): (5.17, 0.82, -0.69, -1.12),
    (5, 2): (2.00e-2, 0.00, 0.00, 0.00),
    (5, 3): (1.00e-5, 0.00, 0.00, 0.00),
    (5, 4): (5.27e-1, 0.76, -0.63, -1.17),
    (6, 1): (1.76e-9, 8.33, 4278.78, -6.41),
    (6, 2): (1.67e-4, 2.79, 304.72, -4.07),
    (6, 3): (3.25, 0.21, 0.19, -3.29),
    (6, 4): (332.46, -0.11, -9.95e-1, -1.58e-3),
    (7, 1): (1.01e-3, -0.29, -0.92, -8.38),
    (7, 2): (3.05e-1, 0.60, 2.65, -0.93),
    (7, 3): (4.54, 0.57, -0.65, -0.89),
    (7, 4): (3.28, 0.52, -0.52, -0.19),
    (8, 1): (1.04, 3.15e-2, -0.61, -9.73),
    (8, 2): (1.04, 0.27, 2.02, -5.92),
    (8, 3): (3.98, 0.26, 0.56, -2.62),
    (8, 4): (2.52e-1, 0.63, 2.08, -4.16),
    (9, 2): (1.00e-5, 0.00, 0.00, 0.00),
    (9, 3): (9.86, 0.29, -0.21, -1.15),
    (9, 4): (7.15e-1, 1.21, -0.70, -0.85),
    (10, 2): (1.00e-5, 0.00, 0.00, 0.00),
    (10, 3): (14.73, 4.52e-2, -0.84, -0.31),
    (10, 4): (6.47, 0.54, 3.59, -5.22),
    (11, 2): (1.00e-5, 0.00, 0.00, 0.00),
    (11, 3): (1.33, 1.15, 1.20, -0.32),
    (11, 4): (1.01e-1, 1.34, 10.05, -6.41),
    (12, 2): (8.58e-5, 2.49e-3, 2.93e-2, -4.33),
    (12, 3): (6.49, 0.53, 2.82, -7.63),
    (12, 4): (6.36, 0.55, 3.86, -5.19),
    (13, 2): (1.00e-5, 0.00, 0.00, 0.00),
    (13, 3): (7.11e-5, 4.12, 1.72e4, -22.24),
    (13, 4): (7.52e-1, 0.77, 6.24, -5.67),
    (14, 2): (1.23, 0.24, 3.17, 4.18e-3),
    (14, 3): (4.90e-1, -8.74e-2, -0.36, -0.79),
    (14, 4): (7.58, 0.37, 1.06, -4.09),
    (15, 2): (1.74e-4, 3.84, 36.06, -0.97),
    (15, 3): (9.46e-2, -5.58e-2, 0.77, -6.43),
    (15, 4): (5.37, 0.47, 2.21, -8.52),
    (16, 1): (3.82e-7, 11.10, 2.57e4, -8.22),
    (16, 2): (1.00e-5, 0.00, 0.00, 0.00),
    (16, 3): (2.29, 4.02e-2, 1.59, -6.06),
    (16, 4): (6.44, 0.13, 2.69, -5.69),
    (17, 2): (1.00e-5, 0.00, 0.00, 0.00),
    (17, 3): (1.88, 0.32, 1.77, -5.70),
    (17, 4): (7.27, 0.29, 1.04, -10.14),
    (18, 2): (1.00e-5, 0.00, 0.00, 0.00),
    (18, 3): (4.57, 0.27, -0.18, -1.57),
    (18, 4): (6.37, 0.85, 10.21, -6.22),
    (19, 2): (1.00e-5, 0.00, 0.00, 0.00),
    (19, 3): (4.76, 0.44, -0.56, -0.88),
    (19, 4): (1.00e-5, 0.00, 0.00, 0.00),
    (20, 3): (3.17e-2, 2.12, 12.06, -0.40),
    (20, 4): (2.68, 0.69, -0.68, -4.47),
    (21, 3): (7.22e-3, 2.34, 411.50, -13.24),
    (21, 4): (1.20e-1, 1.48, 4.00, -9.33),
    (22, 3): (6.34e-1, 6.87e-3, 0.18, -8.04),
    (22, 4): (4.37e-3, 1.25, 40.02, -8.05),
    (23, 2): (1.00e-5, 0.00, 0.00, 0.00),
    (23, 3): (5.12, -2.18e-2, -0.24, -0.83),
    (23, 4): (1.96e-1, -8.53e-3, 0.28, -6.46),
    (24, 2): (5.27e-1, 0.61, -0.89, -3.56),
    (24, 3): (10.90, 0.24, 0.26, -11.94),
    (24, 4): (1.18, 0.20, 0.77, -7.09),
    (25, 2): (1.65e-1, 6.80e-3, 6.44e-2, -9.70),
    (25, 3): (14.20, 0.34, -0.41, -1.19),
    (25, 4): (4.43e-1, 0.91, 10.76, -7.49),
    (26, 2): (1.26, 7.72e-2, -0.41, -7.31),
    (26, 3): (3.42, 0.51, -2.06, -8.99),
    (26, 4): (14.60, 3.57e-2, -0.92, -0.37),
    (27, 2): (5.30, 0.24, -0.91, -0.47),
    (27, 3): (3.26, 0.87, 2.85, -9.23),
    (27, 4): (1.03, 0.58, -0.89, -0.66),
    (28, 2): (1.05, 1.28, 6.54, -1.81),
    (28, 3): (9.73, 0.35, 0.90, -5.33),
    (28, 4): (6.14, 0.25, -0.91, -0.42),
    (29, 2): (1.47e-3, 3.51, 23.91, -0.93),
    (29, 3): (9.26, 0.37, 0.40, -10.73),
    (29, 4): (11.59, 0.20, 0.80, -6.62),
    (30, 2): (1.00e-5, 0.00, 0.00, 0.00),
    (30, 3): (6.96e-4, 4.24, 26.06, -1.24),
    (30, 4): (1.33e-2, 1.56, -0.92, -1.20),
}

# KF96 Table 4: endothermic recombination reactions with no fit, (Z, q) -> deltaE in eV
KF96_ENDOTHERMIC = {(20, 2): -1.73, (21, 2): -0.80, (22, 2): -0.02}

# KF96 Table 3, transcribed from the paper: (Z, q) -> (a, b, c, d, dE/k in 1e4 K)
KF96_ION = {
    (3, 0): (2.81e-3, 2.00, 221.36, -47.64, 0.0),
    (6, 0): (1.00e-5, 0.00, 0.00, 0.00, 0.0),
    (7, 0): (4.55e-3, -0.29, -0.92, -8.38, 1.086),
    (8, 0): (7.40e-2, 0.47, 24.37, -0.74, 0.023),
    (12, 0): (9.76e-3, 3.14, 55.54, -1.12, 0.0),  # ruff: ignore[math-constant]  # 3.14 is the published fit value, not pi
    (12, 1): (7.60e-5, 0.00, -1.97, -4.32, 1.670),
    (14, 1): (4.10e-1, 0.24, 3.17, 4.18e-3, 3.178),
    (16, 0): (1.00e-5, 0.00, 0.00, 0.00, 0.0),
    (24, 1): (4.39, 0.61, -0.89, -3.56, 3.349),
    (25, 1): (2.83e-1, 6.80e-3, 6.44e-2, -9.70, 2.368),
    (26, 1): (2.10, 7.72e-2, -0.41, -7.31, 3.005),
    (27, 1): (1.20e-2, 3.49, 24.41, -1.26, 4.044),
}

# Notes on the AR85 helium rows where the Cloudy source uses a different expression, decoded from
# the Cloudy file source/atmdat_char_tran.cpp: (Z, q) -> note
CLOUDY_HE_NOTES = {
    (6, 2): "absent in Cloudy (no CharExcRecTo[He][C][1] entry)",
    (6, 3): "Cloudy: 4.6e-19*Te^2 (Butler & Dalgarno 1980b) = AR85",
    (6, 4): "Cloudy: 1e-14 = AR85",
    (7, 2): "Cloudy replaces with 0.8e-10 const (Sun et al.; Fang & Kwong 1997)",
    (7, 3): "Cloudy: 1.5e-10 const = AR85",
    (7, 4): "Cloudy replaces with 2e-9 const (Feickert 1984; Rittby 1984)",
    (8, 2): "Cloudy: 3.2e-14*Te^0.95 = AR85 to 2 digits",
    (8, 3): "Cloudy: 1e-9 const (drops the c term)",
    (8, 4): "Cloudy: 6e-10 const (drops the c term)",
    (10, 2): "Cloudy: 1e-14 = AR85",
    (10, 3): "Cloudy: 1e-16*Te^0.5 (AR85 has b=0.51)",
    (10, 4): "Cloudy: 1.7e-11*Te^0.5 (AR85 adds the c term)",
    (12, 3): "Cloudy: 7.5e-10 const (drops the c term)",
    (12, 4): "Cloudy: 1.4e-10*Te^0.30",
    (14, 3): "Cloudy scales AR85 by 1.3: 1.95e-12*Te^0.7 (Fang & Kwong 1997)",
    (14, 4): "Cloudy replaces with 2.54e-11*Te^0.45 (Opradolce et al. 1985)",
    (16, 3): "Cloudy: 1.1e-11*Te^0.5 (AR85 has b=0.56)",
    (16, 4): "Cloudy: 4.8e-14*Te^0.30 = AR85 at 1e4 K",
    (18, 2): "Cloudy: 1.3e-10 = AR85",
    (18, 3): "Cloudy: 1e-14 = AR85",
    (18, 4): "Cloudy: 1.6e-8*Te^-0.30 = AR85",
}

# the temperature grid of the SS11 CDS tables [K]
SS11_TGRID = [
    100, 200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000,
    10000, 20000, 40000, 60000, 80000, 100000, 200000, 400000, 600000, 800000,
]  # fmt: skip

# the n-capture elements that SS11 cover
MAXERR_WARN = 0.25  # a fit error above this fraction gets a warning in the report

# The near-resonant rate estimate for a reaction between two heavy species (Melius 1974).
# The unit is 1e-9 cm3/s, which is the unit of the fit coefficient a.
METAL_METAL_RATE = 1.0
# An exothermic reaction with an energy defect up to this value gets the estimate. A larger
# defect can not reach a level of the products near resonance, so the reaction stays slow and
# ARTIS makes its own Landau-Zener estimate instead.
METAL_METAL_MAX_DEFECT_EV = 4.0
# the acceptor charges of the estimate; higher stages are rare in the kilonova nebular phase
METAL_METAL_ACCEPTOR_CHARGES = (1, 2, 3)

SS11_ZNUM = {elsymbol: elsymbols.index(elsymbol) for elsymbol in ("Ge", "Se", "Br", "Kr", "Rb", "Xe")}

OUTPUT_HEADER = """\
# Fits of the rate coefficients for charge transfer. The comment of each line names its source.
# The artisatomic script makechargetransferfile downloads the sources and generates this file.
# The first non-comment line gives the number of reactions.
# Each reaction line has the columns:
#   Z_acc ionstage_acc Z_don ionstage_don a b c d eexp tmin tmax autoreverse
# The acceptor ion (Z_acc, ionstage_acc) captures an electron from the donor ion (Z_don, ionstage_don).
# The rate coefficient is k = a * 1e-9 * t4^b * (1 + c * exp(d * t4)) * exp(-eexp/T) [cm3/s].
# t4 is T/1e4 K with T clamped into [tmin, tmax]; the factor exp(-eexp/T) uses the true T, and eexp is in K.
# autoreverse 1 lets the code add the reverse reaction from detailed balance when the forward
# reaction releases energy. Use 0 when the file holds an explicit fit for the reverse direction.
"""


class CTEntry(t.NamedTuple):
    """One reaction line of the output file."""

    z_acc: int
    ionstage_acc: int
    z_don: int
    ionstage_don: int
    a: float  # in 1e-9 cm3/s
    b: float
    c: float
    d: float
    eexp: float  # in K
    tmin: float
    tmax: float
    comment: str
    autoreverse: int = 1  # set_autoreverse_flags() gives the value that the output file holds


def download(filekey: str, cachedir: Path, *, refresh: bool = False) -> str:
    """Return the text of a source file, from the cache folder or from a fresh download."""
    cachefile = cachedir / filekey
    if cachefile.is_file() and not refresh:
        print(f"  using cached {cachefile}")
        return cachefile.read_text(encoding="utf-8")
    url = SOURCE_URLS[filekey]
    print(f"  downloading {url}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    # a server that sends an error page with a 200 status must not poison the cache folder
    if response.text.lstrip().startswith("<"):
        msg = f"{url} returned markup instead of data"
        raise RuntimeError(msg)
    cachedir.mkdir(parents=True, exist_ok=True)
    cachefile.write_text(response.text, encoding="utf-8")
    return response.text


def fnum(x: float) -> str:
    """Format a number with up to six significant digits, matching the source file precision."""
    return f"{x:.6g}"


def isclose_published(x: float, y: float) -> bool:
    """Report whether a Cloudy value matches a paper value to the printed precision."""
    if x == y:
        return True
    if x == 0.0 or y == 0.0:
        return False
    return abs(x - y) <= 5e-3 * max(abs(x), abs(y))


def read_cloudy_table(text: str, ncols: int) -> dict[tuple[int, int], list[float]]:
    """Parse a Cloudy charge transfer data file into {(Z, rowindex + 1): column values}.

    The files hold four rows for each element from H to Zn, after a magic number line. For the
    recombination file, the row index + 1 is the ion charge before the capture. For the ionisation
    file, the row index + 1 is the charge after the electron loss.
    """
    lines = text.split("\n")
    assert lines[0].split()[0] in {"201903041", "201903042"}, "unexpected Cloudy data file magic number"
    rows = {}
    for k, line in enumerate(line for line in lines[1:] if line.strip()):
        vals = [float(x) for x in line.split()]
        assert len(vals) == ncols, f"expected {ncols} columns: {line}"
        nelem, ionindex = divmod(k, 4)
        rows[nelem + 1, ionindex + 1] = vals
    assert len(rows) == 120, "expected 30 elements with 4 rows each"
    return rows


def ionlabel(z: int, q: int) -> str:
    """Return a label like Fe+2 or Fe0 for an element with the given charge."""
    return f"{elsymbols[z]}{'+' + str(q) if q > 0 else '0'}"


def get_kf96_h_entries(cachedir: Path, *, refresh: bool = False) -> tuple[list[CTEntry], list[str]]:
    """Build the entries for the reactions with hydrogen from the Cloudy data files.

    Also return a report that lists each Cloudy row that differs from the KF96 paper value.
    """
    rec = read_cloudy_table(download("ctrecombdata.dat", cachedir, refresh=refresh), 7)
    ion = read_cloudy_table(download("ctiondata.dat", cachedir, refresh=refresh), 8)
    report = []
    entries = []
    for (z, q), vals in sorted(rec.items()):
        a, b, c, d, tmin, tmax, deltae_ev = vals
        if tmax == 0.0:
            continue  # Cloudy has no entry for this (Z, q)
        label = f"{ionlabel(z, q)} + H0 -> {ionlabel(z, q - 1)} + H+"
        kf = KF96_REC.get((z, q))
        if a == 0.0 and (z, q) in KF96_ENDOTHERMIC:
            # Cloudy leaves the energy column at zero for these rows, so take the paper value
            deltae_ev = KF96_ENDOTHERMIC[z, q]
            note = "endothermic, no fit (KF96 Table 4)"
        elif kf is None:
            note = "no KF96 entry"
            report.append(f"rec Z={z} q={q}: Cloudy addition with no KF96 entry")
        elif all(starmap(isclose_published, zip((a, b, c, d), kf, strict=True))):
            note = "KF96"
        else:
            published = " ".join(f"{name}={fnum(val)}" for name, val in zip("abcd", kf, strict=True))
            note = f"Cloudy update; KF96 published {published}"
            report.append(f"rec Z={z} q={q}: {note}")
        comment = f"{label}; deltaE={fnum(deltae_ev)} eV; {note}"
        entries.append(CTEntry(z, q + 1, 1, 1, a, b, c, d, 0.0, tmin, tmax, comment))

    for (z, q1), vals in sorted(ion.items()):
        a, b, c, d, tmin, tmax, de4, deficit_ev = vals
        if a == 0.0:
            continue
        q = q1 - 1  # the charge before the ion loses the electron
        label = f"{ionlabel(z, q)} + H+ -> {ionlabel(z, q + 1)} + H0"
        kf_ion = KF96_ION.get((z, q))
        if kf_ion is None:
            note = "no KF96 entry (Cloudy addition)"
            report.append(f"ion Z={z} q={q}: Cloudy addition with no KF96 entry")
        elif all(starmap(isclose_published, zip((a, b, c, d), kf_ion[:4], strict=True))) and isclose_published(
            de4, kf_ion[4]
        ):
            note = "KF96 Table 3"
        else:
            published = " ".join(f"{name}={fnum(val)}" for name, val in zip("abcd", kf_ion[:4], strict=True))
            note = f"Cloudy update; KF96 published {published} dE/k={fnum(kf_ion[4])}(1e4 K)"
            report.append(f"ion Z={z} q={q}: {note}")
        comment = f"{label}; energy deficit={fnum(deficit_ev)} eV; {note}"
        entries.append(CTEntry(1, 2, z, q + 1, a, b, c, d, de4 * 1e4, tmin, tmax, comment))

    return entries, report


def read_nist_ionisation_energies(text: str) -> dict[tuple[int, int], float]:
    """Parse a NIST ASD tab-delimited ionization energy table into {(Z, ion charge): energy in eV}.

    The prefix and suffix columns mark an interpolated or theoretical value. The estimates that
    use this table accept those values.
    """
    energies = {}
    lines = text.splitlines()
    assert lines[0].startswith("At. num"), "unexpected NIST ASD header"
    for line in lines[1:]:
        if line.startswith("Notes:"):
            break  # the footnotes of the table follow the data rows
        if not line.strip():
            continue
        fields = [field.strip().strip('"') for field in line.split("\t")]
        z, charge, energy = fields[0], fields[2], fields[4]
        if not energy:
            continue
        energies[int(z), int(charge.removeprefix("+"))] = float(energy)
    return energies


def get_metal_metal_entries(cachedir: Path, *, refresh: bool = False) -> list[CTEntry]:
    """Build estimates for the capture of an electron by a heavy ion from a neutral heavy atom.

    The kilonova ejecta hold no hydrogen, so the reactions of this file's other sources do not
    occur there. No publication gives the heavy-element rates. The heavy species have many low
    levels. An exothermic reaction with an energy defect below METAL_METAL_MAX_DEFECT_EV can
    therefore reach a level of the products near resonance. Such a reaction gets the
    near-resonant rate of 1e-9 cm3/s (Melius 1974). The donor is always neutral, because the
    Coulomb repulsion between two positive ions makes their reactions slow.
    """
    energies = read_nist_ionisation_energies(download("nist_ie_sc_to_u.dat", cachedir, refresh=refresh))
    donors = sorted(z for (z, charge) in energies if charge == 0)
    entries = []
    for z_acc in donors:
        for charge_acc in METAL_METAL_ACCEPTOR_CHARGES:
            if (z_acc, charge_acc - 1) not in energies:
                continue
            # the capture releases the ionization energy of the product ion
            for z_don in donors:
                if z_don == z_acc and charge_acc == 1:
                    continue  # the products equal the reactants, so the reaction changes nothing
                deltae_ev = energies[z_acc, charge_acc - 1] - energies[z_don, 0]
                if not 0.0 < deltae_ev <= METAL_METAL_MAX_DEFECT_EV:
                    continue
                label = f"{ionlabel(z_acc, charge_acc)} + {ionlabel(z_don, 0)} -> {ionlabel(z_acc, charge_acc - 1)} + {ionlabel(z_don, 1)}"
                comment = (
                    f"{label}; deltaE={fnum(deltae_ev)} eV; near-resonant estimate (Melius 1974), NIST ASD energies"
                )
                entries.append(
                    CTEntry(z_acc, charge_acc + 1, z_don, 1, METAL_METAL_RATE, 0.0, 0.0, 0.0, 0.0, 1000, 40000, comment)
                )
    return entries


def get_he_entries(cachedir: Path, *, refresh: bool = False) -> list[CTEntry]:
    """Build the entries for recombination with neutral helium from the AR85 table."""
    entries = []
    for line in download("ar85_ct2.dat", cachedir, refresh=refresh).split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        z, nelectrons = int(parts[0]), int(parts[1])
        a, b, c, d, tmin, tmax = (float(x) for x in parts[2:8])
        q = z - nelectrons + 1  # the file lists the electron count of the product ion
        label = f"{ionlabel(z, q)} + He0 -> {ionlabel(z, q - 1)} + He+"
        note = CLOUDY_HE_NOTES.get((z, q), "")
        comment = f"AR85; {label}" + (f"; {note}" if note else "")
        entries.append(CTEntry(z, q + 1, 2, 1, a, b, c, d, 0.0, tmin, tmax, comment))
    return entries


def read_cds_totals(text: str) -> dict[tuple[str, int], list[float]]:
    """Sum the final-state resolved channels of an SS11 CDS table into rates for each (element, charge).

    The CDS tables floor each channel at the radiative rate 1e-14 cm3/s, while the paper floors the
    total. To reproduce the totals printed in the paper, exclude the channel values that sit exactly
    on the floor, then floor the sum.
    """
    channelvalues: dict[tuple[str, int], list[list[float]]] = {}
    for rawline in text.split("\n"):
        if not rawline.strip():
            continue
        line = rawline.rstrip().ljust(212)
        assert len(line) == 212, f"unexpected line length in the CDS table: {rawline}"
        elsymbol = line[0:2].strip()
        q = int(line[3])
        vals = [float(line[33 + 9 * i : 41 + 9 * i]) for i in range(len(SS11_TGRID))]
        cols = channelvalues.setdefault((elsymbol, q), [[] for _ in SS11_TGRID])
        for i, v in enumerate(vals):
            cols[i].append(v)

    totals = {}
    for key, cols in channelvalues.items():
        row = []
        for col in cols:
            total = sum(v for v in col if v != 1.00e-14)
            row.append(max(total, 1.00e-14))
        totals[key] = row
    return totals


class SS11Fit(t.NamedTuple):
    """One fit of a tabulated SS11 rate curve, with its quality and its validity range."""

    a: float  # in cm3/s
    b: float
    eexp: float  # in K
    maxerr: float  # the largest relative error over the usable points
    npts: int  # the count of usable points; below four, a is a flat value and not a fit
    tmin: float  # the span of the usable points, which the output clamps T into
    tmax: float


def fit_ss11_curve(ks: list[float]) -> SS11Fit:
    """Fit ln k = ln a + b ln t4 - eexp/T to a tabulated SS11 rate curve.

    The usable points lie between 1e3 and 4e4 K and above the 1e-14 floor: charge transfer
    competes with the other processes only in the nebular low-temperature regime, and the fit
    form cannot span the floor-then-steep-rise shape of some reactions over the full tabulated
    range. tmin and tmax hold the span of the usable points. Below tmin, the reader clamps t4
    and the Boltzmann factor keeps the true temperature, so the rate falls like the tabulated
    values do. A curve with fewer than four usable points gets the flat value of its 2e4 K
    entry, together with the error of that value over the usable points.
    """
    pts = [(temp, k) for temp, k in zip(SS11_TGRID, ks, strict=True) if 1000 <= temp <= 40000 and k > 1.2e-14]
    if len(pts) < 4:
        flat = ks[SS11_TGRID.index(20000)]
        maxerr = max((abs(flat / k - 1.0) for _, k in pts), default=0.0)
        return SS11Fit(flat, 0.0, 0.0, maxerr, len(pts), 20000, 20000)

    def solve_normal_equations(use_eexp: bool) -> list[float]:
        ncoeff = 3 if use_eexp else 2
        ata = [[0.0] * ncoeff for _ in range(ncoeff)]
        aty = [0.0] * ncoeff
        for temp, k in pts:
            row = [1.0, math.log(temp / 1e4)] + ([-1.0 / temp] if use_eexp else [])
            y = math.log(k)
            for i in range(ncoeff):
                for j in range(ncoeff):
                    ata[i][j] += row[i] * row[j]
                aty[i] += row[i] * y
        for i in range(ncoeff):
            pivot = max(range(i, ncoeff), key=lambda r: abs(ata[r][i]))
            ata[i], ata[pivot] = ata[pivot], ata[i]
            aty[i], aty[pivot] = aty[pivot], aty[i]
            for r in range(i + 1, ncoeff):
                factor = ata[r][i] / ata[i][i]
                for col in range(i, ncoeff):
                    ata[r][col] -= factor * ata[i][col]
                aty[r] -= factor * aty[i]
        x = [0.0] * ncoeff
        for i in reversed(range(ncoeff)):
            x[i] = (aty[i] - sum(ata[i][j] * x[j] for j in range(i + 1, ncoeff))) / ata[i][i]
        return x

    lna, b, eexp = solve_normal_equations(use_eexp=True)
    if eexp < 0.0:
        # a negative Boltzmann temperature would grow without limit towards low T, so drop the term
        lna, b = solve_normal_equations(use_eexp=False)
        eexp = 0.0
    a = math.exp(lna)
    # snap the degenerate values of a flat curve to zero
    if abs(b) < 1e-6:
        b = 0.0
    if eexp < 1e-3:
        eexp = 0.0
    maxerr = max(abs((a * (temp / 1e4) ** b * math.exp(-eexp / temp)) / k - 1.0) for temp, k in pts)
    return SS11Fit(a, b, eexp, maxerr, len(pts), pts[0][0], pts[-1][0])


def get_ss11_entries(cachedir: Path, *, refresh: bool = False) -> tuple[list[CTEntry], list[str]]:
    """Build the n-capture element entries from fits to the SS11 CDS rate tables."""
    rec_totals = read_cds_totals(download("ss11_table4.dat", cachedir, refresh=refresh))
    ion_totals = read_cds_totals(download("ss11_table5.dat", cachedir, refresh=refresh))
    entries = []
    report = []
    for totals, is_recombination in ((rec_totals, True), (ion_totals, False)):
        for (elsymbol, q), ks in sorted(totals.items(), key=lambda kv: (SS11_ZNUM[kv[0][0]], kv[0][1])):
            z = SS11_ZNUM[elsymbol]
            fit = fit_ss11_curve(ks)
            direction = "rec" if is_recombination else "ion"
            warning = "  WARNING: the fit error is large" if fit.maxerr > MAXERR_WARN else ""
            report.append(
                f"{direction} {elsymbol} q={q}: a={fit.a:.3e} b={fit.b:.3f} eexp={fit.eexp:.0f}"
                f" maxerr={fit.maxerr * 100:.0f}% npts={fit.npts}{warning}"
            )
            label = (
                f"{ionlabel(z, q)} + H0 -> {ionlabel(z, q - 1)} + H+"
                if is_recombination
                else f"{ionlabel(z, q)} + H+ -> {ionlabel(z, q + 1)} + H0"
            )
            if fit.npts >= 4:
                source = (
                    f"fit to their tabulated k(T) over {fnum(fit.tmin)}-{fnum(fit.tmax)} K,"
                    f" max fit error {fit.maxerr * 100:.0f}%"
                )
            elif fit.npts > 0:
                source = (
                    f"flat value from their 2e4 K entry"
                    f" ({fit.npts} usable points over 1e3-4e4 K, max error {fit.maxerr * 100:.0f}%)"
                )
            else:
                source = "flat value from their 2e4 K entry (no usable fit points over 1e3-4e4 K)"
            comment = f"SS11; {label}; {source}"
            reaction = (z, q + 1, 1, 1) if is_recombination else (1, 2, z, q + 1)
            entries.append(CTEntry(*reaction, fit.a / 1e-9, fit.b, 0.0, 0.0, fit.eexp, fit.tmin, fit.tmax, comment))
    return entries, report


def set_autoreverse_flags(entries: list[CTEntry]) -> list[CTEntry]:
    """Set the reverse reaction flag of each entry from the other reactions that the file holds.

    The reverse of "A + D -> A(-1) + D(+1)" is "D(+1) + A(-1) -> D + A". An entry whose reverse
    reaction has its own fit gets a zero, which stops the ARTIS code from adding that reverse a
    second time. Every other entry gets a one, so detailed balance supplies its reverse.
    """
    reactions = {(e.z_acc, e.ionstage_acc, e.z_don, e.ionstage_don) for e in entries}
    return [
        entry._replace(
            autoreverse=0
            if (entry.z_don, entry.ionstage_don + 1, entry.z_acc, entry.ionstage_acc - 1) in reactions
            else 1
        )
        for entry in entries
    ]


def write_chargetransfer_file(entries: list[CTEntry], outpath: Path) -> None:
    """Write the assembled entries in the format that the ARTIS chargetransfer.cc reader expects."""
    with outpath.open("w", encoding="utf-8") as fout:
        fout.write(OUTPUT_HEADER)
        fout.write(f"{len(entries)}\n")
        for e in entries:
            fout.write(
                f"{e.z_acc:3d} {e.ionstage_acc:2d} {e.z_don:3d} {e.ionstage_don:2d} {e.a:12.6g} {e.b:10.6g}"
                f" {e.c:10.6g} {e.d:10.6g} {e.eexp:9.6g} {e.tmin:8.6g} {e.tmax:8.6g} {e.autoreverse}"
                f" # {e.comment}\n"
            )
    print(f"wrote {len(entries)} reactions to {outpath}")


def main() -> None:
    """Download the sources of the charge transfer rates and write chargetransfer.txt for ARTIS."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaultoutpath = Path(__file__).parent.parent.absolute() / "artis_files" / "data" / "chargetransfer.txt"
    defaultcachedir = Path(__file__).parent.parent.absolute() / "atomic-data-chargetransfer"
    parser.add_argument("-outputfile", default=defaultoutpath, type=Path, help="path of the output file")
    parser.add_argument("-cachedir", default=defaultcachedir, type=Path, help="folder for the downloaded source files")
    parser.add_argument("-refresh", action="store_true", help="download the source files again and replace the cache")
    args = parser.parse_args()

    print("Reactions with hydrogen (Cloudy carrying KF96 plus updates):")
    kf96_entries, kf96_report = get_kf96_h_entries(args.cachedir, refresh=args.refresh)
    print("Recombination with neutral helium (AR85):")
    he_entries = get_he_entries(args.cachedir, refresh=args.refresh)
    print("n-capture elements with hydrogen (SS11):")
    ss11_entries, ss11_report = get_ss11_entries(args.cachedir, refresh=args.refresh)
    print("Heavy ion with neutral heavy atom (near-resonant estimates):")
    metal_entries = get_metal_metal_entries(args.cachedir, refresh=args.refresh)
    print(f"  {len(metal_entries)} reactions with an energy defect in (0, {METAL_METAL_MAX_DEFECT_EV}] eV")

    print("\nCloudy rows that differ from the KF96 paper tables:")
    for line in kf96_report:
        print(f"  {line}")
    print("\nSS11 fit quality:")
    for line in ss11_report:
        print(f"  {line}")
    print()

    args.outputfile.parent.mkdir(parents=True, exist_ok=True)
    entries = set_autoreverse_flags(kf96_entries + he_entries + ss11_entries + metal_entries)
    write_chargetransfer_file(entries, args.outputfile)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write the file of charge transfer rates for ARTIS (data/chargetransfer.txt in the artis repository).

The script reads the published fits and rate tables from the folder atomic-data-chargetransfer.
It converts them into one fit form and writes one reaction for each line. The script
setup_chargetransfer_data.sh in that folder downloads the tables. The sources are:

- The Cloudy master data files ctrecombdata.dat and ctiondata.dat (gitlab.nublado.org). They carry
  the fits of Kingdon & Ferland (1996), ApJS, 106, 205 (KF96) for reactions with hydrogen. Cloudy
  adds later updates to individual reactions. The script compares each row against the KF96 paper
  tables (transcribed below) and names each update in the comment of its line.
- The table of Arnaud & Rothenflug (1985), A&AS, 60, 425 (AR85) for recombination with neutral
  helium, from the ASCII file ct2.dat of D. Verner (pa.uky.edu/~verner).
- The CDS tables of Sterling & Stancil (2011), A&A, 535, A117 (SS11). They cover the n-capture
  elements Ge, Se, Br, Kr, Rb, and Xe with hydrogen. SS11 publish tabulated k(T) values and no
  fit coefficients, so the script fits their tables over 1e3 to 4e4 K.
- Estimates for the reactions of a heavy ion with a neutral heavy atom, and for the reactions
  of any element with hydrogen that the sources above do not cover. The kilonova ejecta consist
  mostly of heavy elements, so these reactions matter there. No publication gives these rates,
  so the script takes the ionization energies from the NIST table of this package. It keeps the
  exothermic reactions with a small energy defect. Each one gets the near-resonant rate of
  1e-9 cm3/s (Melius 1974). The reverse reactions come from detailed balance in ARTIS.

Every entry uses the KF96 fit form (their equation 7, from AR85):
  k = a * 1e-9 * t4^b * (1 + c * exp(d * t4)) * exp(-eexp/T)  [cm3/s],  t4 = T / 1e4 K.
See the header of the output file for the column definitions. The chargetransfer.cc module of ARTIS
reads the file and generates Landau-Zener estimates for the reactions that the file does not cover.
"""

import argparse
import math
import typing as t
from collections import Counter
from collections.abc import Set as AbstractSet
from itertools import starmap
from pathlib import Path

import numpy as np

from artisatomic.base import elsymbols
from artisatomic.base import get_nist_ionization_energies_ev
from artisatomic.base import get_nist_ionization_provenance
from artisatomic.base import xopen_check_extension

# KF96 Table 1 and the Table 2 totals, transcribed from the paper: (Z, q) -> (a, b, c, d).
# q is the ion charge before the electron capture, and a is in 1e-9 cm3/s. The script uses these
# values only to detect and name the Cloudy updates, so the table omits the temperature ranges.
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

# Notes on the AR85 helium rows where the Cloudy source uses a different expression. The notes
# come from the Cloudy file source/atmdat_char_tran.cpp: (Z, q) -> note
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

MAXERR_WARN = 0.25  # a fit error above this fraction gets a warning in the report

# The fit window of the SS11 curves [K]. Charge transfer competes with the other processes only
# in the nebular low-temperature regime.
SS11_FIT_TMIN = 1000
SS11_FIT_TMAX = 40000
SS11_FLOOR = 1.00e-14  # the radiative floor of the SS11 tables [cm3/s]
SS11_USABLE_MIN = 1.2e-14  # a tabulated total above this value is a usable fit point [cm3/s]
SS11_MIN_POINTS = 4  # a curve with fewer usable points gets a flat value instead of a fit
SS11_FLAT_TEMP = 20000  # the entry that supplies the flat value [K]

# The near-resonant rate estimate for a reaction between two heavy species (Melius 1974).
# The unit is 1e-9 cm3/s, which is the unit of the fit coefficient a.
METAL_METAL_RATE = 1.0
# An exothermic reaction with an energy defect up to this value gets the estimate. A larger
# defect can not reach a level of the products near resonance. The reaction then stays slow,
# and ARTIS makes its own Landau-Zener estimate.
METAL_METAL_MAX_DEFECT_EV = 4.0
# the acceptor charges of the estimate; higher stages are rare in the kilonova nebular phase
METAL_METAL_ACCEPTOR_CHARGES = (1, 2, 3)
# The elements of the estimates. The bounds are parameters of this script. The estimates with a
# neutral heavy atom start at Sc, because the light elements have few low levels. Both kinds end
# at Lr, the last actinide.
METAL_METAL_ZMIN = 21
ESTIMATE_ZMAX = 103
# The range of use of the estimates [K]. A flat value has no temperature dependence, so the clamp
# has no effect; the values document the nebular range only.
METAL_METAL_TMIN = 1000
METAL_METAL_TMAX = 40000

# the n-capture elements that SS11 cover
SS11_ZNUM = {elsymbol: elsymbols.index(elsymbol) for elsymbol in ("Ge", "Se", "Br", "Kr", "Rb", "Xe")}


def format_header(source_counts: Counter[str]) -> str:
    """Build the comment block of the output file: the format, the sources, and the parameters."""
    lines = [
        "Fits of the rate coefficients for charge transfer, for the chargetransfer.cc module of ARTIS.",
        "The artisatomic script makechargetransferfile generates this file. Do not edit it by hand.",
        "",
        "FORMAT",
        "The first non-comment line gives the number of reactions. Each reaction line has the columns:",
        "  Z_acc ionstage_acc Z_don ionstage_don a b c d eexp tmin tmax autoreverse",
        "The acceptor ion (Z_acc, ionstage_acc) captures an electron from the donor ion (Z_don, ionstage_don).",
        "The rate coefficient is k = a * 1e-9 * t4^b * (1 + c * exp(d * t4)) * exp(-eexp/T) [cm3/s].",
        "t4 is T/1e4 K with T clamped into [tmin, tmax]; the factor exp(-eexp/T) uses the true T, and eexp is in K.",
        "For a flat entry (b = c = eexp = 0) the clamp has no effect.",
        "autoreverse 1 lets the code add the reverse reaction from detailed balance when the forward",
        "reaction releases energy. It is 0 when this file holds a fit for the reverse reaction.",
        "The comment of each line starts with the tag of its source, then names the reaction.",
        "",
        "SOURCES",
        f"Cloudy ({source_counts['Cloudy']} reactions): reactions with hydrogen, from the Cloudy data files",
        "  https://gitlab.nublado.org/cloudy/cloudy/-/raw/master/data/ctrecombdata.dat",
        "  https://gitlab.nublado.org/cloudy/cloudy/-/raw/master/data/ctiondata.dat",
        "  They hold the fits of Kingdon & Ferland (1996), ApJS, 106, 205 (KF96) plus the later updates of",
        "  Cloudy. The comment of a line names each update, with the values that KF96 published.",
        f"AR85 ({source_counts['AR85']} reactions): recombination with neutral helium, from the table of",
        "  Arnaud & Rothenflug (1985), A&AS, 60, 425, in the file ct2.dat of D. Verner",
        "  https://www.pa.uky.edu/~verner/dima/ct/ct2.dat",
        f"SS11 ({source_counts['SS11']} reactions): Ge, Se, Br, Kr, Rb, and Xe with hydrogen, from the CDS tables of",
        "  Sterling & Stancil (2011), A&A, 535, A117",
        "  https://cdsarc.cds.unistra.fr/ftp/J/A+A/535/A117/table4.dat",
        "  https://cdsarc.cds.unistra.fr/ftp/J/A+A/535/A117/table5.dat",
        "  SS11 publish tabulated k(T) values and no fit coefficients. This script fits their tables, see",
        "  PARAMETERS. The tables resolve the final state; the fit uses the sum over the final states.",
        f"Estimate ({source_counts['Estimate']} reactions): a heavy ion with a neutral heavy atom, for the kilonova ejecta,",
        "  which consist mostly of heavy elements, and any ion with a hydrogen atom or a proton with any neutral",
        "  atom. No publication gives these rates. Each exothermic reaction with a small energy defect, which no",
        "  other source of this file covers, gets the near-resonant rate of Melius (1974), J. Phys. B, 7, 1692.",
        "  The energy defect comes from the ground-state ionization",
        "  energies of the NIST table that artisatomic ships (nist_ionization.txt.zst):",
        *[f"    {line}" for line in get_nist_ionization_provenance()],
        f"  With a neutral heavy atom, the elements are {elsymbols[METAL_METAL_ZMIN]} to {elsymbols[ESTIMATE_ZMAX]} (Z = {METAL_METAL_ZMIN} to {ESTIMATE_ZMAX}). With hydrogen,",
        f"  the elements are He to {elsymbols[ESTIMATE_ZMAX]}.",
        "",
        "PARAMETERS",
        "SS11 fits: ln k = ln a + b ln t4 - eexp/T, so c = d = 0. A fit uses the tabulated totals between",
        f"  {SS11_FIT_TMIN} and {SS11_FIT_TMAX} K that lie above {SS11_USABLE_MIN:g} cm3/s. The tables floor each total at",
        f"  {SS11_FLOOR:g} cm3/s. tmin and tmax of a fitted entry span the points that the fit used. A curve with",
        f"  fewer than {SS11_MIN_POINTS} usable points gets the flat value of its {SS11_FLAT_TEMP} K entry, with tmin = tmax =",
        f"  {SS11_FLAT_TEMP} K. The comment of each line gives the largest relative error of the fit or the flat value",
        "  over the usable points. eexp of an SS11 entry is a fit coefficient and not a measured energy.",
        "  Below tmin, an entry with eexp > 0 keeps falling and an entry with eexp = 0 holds its value at tmin.",
        f"Estimates: rate {METAL_METAL_RATE:g}e-9 cm3/s, flat in T, for an exothermic reaction with an energy defect",
        f"  in (0, {METAL_METAL_MAX_DEFECT_EV:g}] eV. The acceptor charge is one of {METAL_METAL_ACCEPTOR_CHARGES}, or 1 for a proton. The",
        "  donor is always neutral, because the Coulomb repulsion between two positive ions makes their reactions",
        "  slow. The estimate assumes that a level of the products lies near the energy defect; it does not",
        f"  check the level lists. tmin and tmax are {METAL_METAL_TMIN}-{METAL_METAL_TMAX} K, the range of use.",
        "  ARTIS makes a Landau-Zener estimate for every reaction that this file does not hold.",
        "autoreverse: set from the full list of reactions in this file, not from one source alone.",
    ]
    return "".join(f"# {line}".rstrip() + "\n" for line in lines)


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


def read_source(filekey: str, sourcedir: Path) -> str:
    """Return the text of one source file from the data folder, plain or zstd compressed."""
    try:
        with xopen_check_extension(sourcedir / filekey, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError as err:
        msg = (
            f"{err} The repository tracks the source files, so the checkout is incomplete."
            " Run setup_chargetransfer_data.sh in atomic-data-chargetransfer. The script downloads the file again."
        )
        raise FileNotFoundError(msg) from err
    print(f"  read {filekey} from {sourcedir}")
    return text


def fnum(x: float) -> str:
    """Format a number with up to six significant digits, the precision of the source files."""
    return f"{x:.6g}"


def isclose_published(x: float, y: float) -> bool:
    """Report whether a Cloudy value matches a paper value to the printed precision.

    The tolerance is the half-ulp of a three-digit value near 1.0. A two-digit paper value can fail
    it when Cloudy holds more digits than the paper printed; the current files hold no such row.
    """
    return math.isclose(x, y, rel_tol=5e-3)


def matches_kf96(values: tuple[float, ...], published: tuple[float, ...]) -> bool:
    """Report whether the Cloudy values match the published values to the printed precision."""
    return all(starmap(isclose_published, zip(values, published, strict=True)))


def kf96_published_text(published: tuple[float, ...]) -> str:
    """Name the published fit values, for the comment of a row that Cloudy changed."""
    return " ".join(f"{name}={fnum(val)}" for name, val in zip("abcd", published, strict=True))


def read_cloudy_table(text: str, ncols: int) -> dict[tuple[int, int], list[float]]:
    """Parse one charge transfer file of Cloudy into {(Z, rowindex + 1): column values}.

    The files hold four rows for each element from H to Zn, after a magic number line. For the
    recombination file, the row index + 1 is the ion charge before the capture. For the ionization
    file, the row index + 1 is the charge after the electron loss.
    """
    lines = text.split("\n")
    assert lines[0].split()[0] in {"201903041", "201903042"}, "unexpected magic number in the Cloudy file"
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


def get_kf96_h_entries(sourcedir: Path) -> tuple[list[CTEntry], list[str]]:
    """Build the entries for the reactions with hydrogen from the Cloudy data files.

    Also return a report that lists each Cloudy row that differs from the KF96 paper value.
    """
    rec = read_cloudy_table(read_source("ctrecombdata.dat", sourcedir), 7)
    ion = read_cloudy_table(read_source("ctiondata.dat", sourcedir), 8)
    report = []
    entries = []
    for (z, q), vals in sorted(rec.items()):
        a, b, c, d, tmin, tmax, deltae_ev = vals
        if tmax == 0.0:
            continue  # Cloudy has no entry for this (Z, q)
        if a == 0.0:
            # KF96 Table 4 lists these endothermic reactions with no fit. A zero rate in the file
            # would stop ARTIS from its own estimate, so the file holds no entry for them.
            paper_deltae = KF96_ENDOTHERMIC.get((z, q), "unknown")
            report.append(f"rec Z={z} q={q}: endothermic, no fit (KF96 Table 4, deltaE={paper_deltae} eV); not written")
            continue
        label = f"{ionlabel(z, q)} + H0 -> {ionlabel(z, q - 1)} + H+"
        kf = KF96_REC.get((z, q))
        if kf is None:
            note = "no KF96 entry"
            report.append(f"rec Z={z} q={q}: Cloudy addition with no KF96 entry")
        elif matches_kf96((a, b, c, d), kf):
            note = "KF96"
        else:
            note = f"Cloudy update; KF96 published {kf96_published_text(kf)}"
            report.append(f"rec Z={z} q={q}: {note}")
        # the energy column of Cloudy is the defect of the dominant channel, with the sign of the file
        comment = f"Cloudy; {label}; Cloudy deltaE={fnum(deltae_ev)} eV; {note}"
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
        elif matches_kf96((a, b, c, d, de4), kf_ion):
            note = "KF96 Table 3"
        else:
            note = f"Cloudy update; KF96 published {kf96_published_text(kf_ion[:4])} dE/k={fnum(kf_ion[4])}(1e4 K)"
            report.append(f"ion Z={z} q={q}: {note}")
        comment = f"Cloudy; {label}; energy deficit={fnum(deficit_ev)} eV; {note}"
        entries.append(CTEntry(1, 2, z, q + 1, a, b, c, d, de4 * 1e4, tmin, tmax, comment))

    return entries, report


def get_estimate_entries(covered: AbstractSet[tuple[int, int, int, int]] = frozenset()) -> list[CTEntry]:
    """Build estimates for the capture of an electron from a neutral atom, with a small energy defect.

    The kilonova ejecta consist mostly of heavy elements, so the reactions between heavy species
    matter there. No publication gives the heavy-element rates. The heavy species have many low
    levels. An exothermic reaction with an energy defect below METAL_METAL_MAX_DEFECT_EV can
    therefore reach a level of the products near resonance. Such a reaction gets the
    near-resonant rate of 1e-9 cm3/s (Melius 1974). The donor is always neutral, because the
    Coulomb repulsion between two positive ions makes their reactions slow.

    Three kinds of reaction get an estimate:

    - a heavy ion (Sc to Lr) captures from a neutral heavy atom (Sc to Lr);
    - an ion of any element up to Lr captures from a hydrogen atom;
    - a proton captures from a neutral atom of any element up to Lr.

    A reaction in `covered`, the keys (Z_acc, stage_acc, Z_don, stage_don) of the other sources,
    gets no estimate. A published rate therefore replaces its estimate. The ionization energies
    come from the NIST table that the package ships, artisatomic/nist_ionization.txt.zst.
    """
    energies = {(z, stage - 1): energy_ev for (z, stage), energy_ev in get_nist_ionization_energies_ev().items()}
    ie_hydrogen_ev = energies[1, 0]
    entries = []

    def add(z_acc: int, charge_acc: int, z_don: int, deltae_ev: float) -> None:
        """Add the capture by (z_acc, charge_acc) from the neutral atom z_don when the defect is small."""
        if (z_acc, charge_acc + 1, z_don, 1) in covered or not 0.0 < deltae_ev <= METAL_METAL_MAX_DEFECT_EV:
            return
        label = f"{ionlabel(z_acc, charge_acc)} + {ionlabel(z_don, 0)} -> {ionlabel(z_acc, charge_acc - 1)} + {ionlabel(z_don, 1)}"
        comment = f"Estimate; {label}; deltaE={fnum(deltae_ev)} eV; near-resonant rate, NIST ASD energies"
        entries.append(
            CTEntry(
                z_acc,
                charge_acc + 1,
                z_don,
                1,
                METAL_METAL_RATE,
                0.0,
                0.0,
                0.0,
                0.0,
                METAL_METAL_TMIN,
                METAL_METAL_TMAX,
                comment,
            )
        )

    heavy = [z for (z, charge) in sorted(energies) if charge == 0 and METAL_METAL_ZMIN <= z <= ESTIMATE_ZMAX]
    for z_acc in heavy:
        for charge_acc in METAL_METAL_ACCEPTOR_CHARGES:
            if (z_acc, charge_acc - 1) not in energies:
                continue
            # the capture releases the ionization energy of the product ion
            energy_released_ev = energies[z_acc, charge_acc - 1]
            for z_don in heavy:
                if z_don == z_acc and charge_acc == 1:
                    continue  # the products equal the reactants, so the reaction changes nothing
                add(z_acc, charge_acc, z_don, energy_released_ev - energies[z_don, 0])

    for z in sorted({z for (z, charge) in energies if 1 < z <= ESTIMATE_ZMAX}):
        # an ion captures from a hydrogen atom
        for charge_acc in METAL_METAL_ACCEPTOR_CHARGES:
            if (z, charge_acc - 1) in energies:
                add(z, charge_acc, 1, energies[z, charge_acc - 1] - ie_hydrogen_ev)
        # a proton captures from the neutral atom
        if (z, 0) in energies:
            add(1, 1, z, ie_hydrogen_ev - energies[z, 0])
    return entries


def get_he_entries(sourcedir: Path) -> list[CTEntry]:
    """Build the entries for recombination with neutral helium from the AR85 table."""
    entries = []
    for line in read_source("ar85_ct2.dat", sourcedir).split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        assert len(parts) == 8, f"expected 8 columns in the AR85 table: {line}"
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
    total. The code excludes the channel values that sit exactly on the floor. Then it floors the
    sum. This reproduces the totals that the paper prints.
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
            total = sum(v for v in col if v != SS11_FLOOR)
            row.append(max(total, SS11_FLOOR))
        totals[key] = row
    return totals


class SS11Fit(t.NamedTuple):
    """One fit of a tabulated SS11 rate curve, with its quality and its validity range."""

    a: float  # in cm3/s
    b: float
    eexp: float  # in K
    maxerr: float  # the largest relative error over the usable points
    npts: int  # the count of usable points; below SS11_MIN_POINTS, a is a flat value and not a fit
    tmin: float  # the span of the usable points, which the output clamps T into
    tmax: float


def fit_ss11_curve(ks: list[float]) -> SS11Fit:
    """Fit ln k = ln a + b ln t4 - eexp/T to a tabulated SS11 rate curve.

    The usable points lie in the fit window and above SS11_USABLE_MIN. Charge transfer competes
    with the other processes only in the nebular low-temperature regime. Also, the fit form cannot
    span the floor-then-steep-rise shape of some reactions over the full tabulated range. tmin and
    tmax hold the span of the usable points.

    Below tmin, the reader clamps t4 and the Boltzmann factor keeps the true temperature. A fit
    with eexp > 0 therefore keeps falling, like the tabulated values; a fit with eexp = 0 holds
    its value at tmin. A curve with fewer than SS11_MIN_POINTS usable points gets the flat value
    of its SS11_FLAT_TEMP entry, with tmin = tmax = SS11_FLAT_TEMP. The result includes the error
    of that value over the usable points.
    """
    pts = [
        (temp, k)
        for temp, k in zip(SS11_TGRID, ks, strict=True)
        if SS11_FIT_TMIN <= temp <= SS11_FIT_TMAX and k > SS11_USABLE_MIN
    ]
    if len(pts) < SS11_MIN_POINTS:
        flat = ks[SS11_TGRID.index(SS11_FLAT_TEMP)]
        maxerr = max((abs(flat / k - 1.0) for _, k in pts), default=0.0)
        return SS11Fit(flat, 0.0, 0.0, maxerr, len(pts), SS11_FLAT_TEMP, SS11_FLAT_TEMP)

    temps = np.array([temp for temp, _ in pts], dtype=float)
    logk = np.log([k for _, k in pts])
    design = np.column_stack([np.ones_like(temps), np.log(temps / 1e4), -1.0 / temps])
    lna, b, eexp = map(float, np.linalg.lstsq(design, logk, rcond=None)[0])
    if eexp < 0.0:
        # a negative Boltzmann temperature would grow without limit towards low T, so drop the term
        lna, b = map(float, np.linalg.lstsq(design[:, :2], logk, rcond=None)[0])
        eexp = 0.0
    a = math.exp(lna)
    # set the degenerate values of a flat curve to zero
    if abs(b) < 1e-6:
        b = 0.0
    if eexp < 1e-3:
        eexp = 0.0
    maxerr = max(abs((a * (temp / 1e4) ** b * math.exp(-eexp / temp)) / k - 1.0) for temp, k in pts)
    return SS11Fit(a, b, eexp, maxerr, len(pts), pts[0][0], pts[-1][0])


def get_ss11_entries(sourcedir: Path) -> tuple[list[CTEntry], list[str]]:
    """Build the n-capture element entries from fits to the SS11 CDS rate tables."""
    rec_totals = read_cds_totals(read_source("ss11_table4.dat", sourcedir))
    ion_totals = read_cds_totals(read_source("ss11_table5.dat", sourcedir))
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
            window = f"{fnum(SS11_FIT_TMIN)}-{fnum(SS11_FIT_TMAX)} K"
            if fit.npts >= SS11_MIN_POINTS:
                source = (
                    f"fit to their tabulated k(T) over {fnum(fit.tmin)}-{fnum(fit.tmax)} K,"
                    f" max fit error {fit.maxerr * 100:.0f}%"
                )
            elif fit.npts > 0:
                source = (
                    f"flat value from their {fnum(SS11_FLAT_TEMP)} K entry"
                    f" ({fit.npts} usable points over {window}, max error {fit.maxerr * 100:.0f}%)"
                )
            else:
                source = f"flat value from their {fnum(SS11_FLAT_TEMP)} K entry (no usable fit points over {window})"
            comment = f"SS11; {label}; {source}"
            reaction = (z, q + 1, 1, 1) if is_recombination else (1, 2, z, q + 1)
            entries.append(CTEntry(*reaction, fit.a / 1e-9, fit.b, 0.0, 0.0, fit.eexp, fit.tmin, fit.tmax, comment))
    return entries, report


def set_autoreverse_flags(entries: list[CTEntry]) -> list[CTEntry]:
    """Set the reverse reaction flag of each entry from the other reactions that the file holds.

    The reverse of "A + D -> A(-1) + D(+1)" is "D(+1) + A(-1) -> D + A". An entry whose reverse
    reaction has its own fit gets a zero, so the ARTIS code does not add that reverse a second
    time. Every other entry gets a one, so detailed balance supplies its reverse. The file must
    hold each reaction once, because the reader takes the first fit that it finds.
    """
    keys = [(e.z_acc, e.ionstage_acc, e.z_don, e.ionstage_don) for e in entries]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    assert not duplicates, f"reactions that appear more than once: {duplicates[:5]}"
    reactions = set(keys)
    return [
        entry._replace(
            autoreverse=0
            if (entry.z_don, entry.ionstage_don + 1, entry.z_acc, entry.ionstage_acc - 1) in reactions
            else 1
        )
        for entry in entries
    ]


def write_chargetransfer_file(entries: list[CTEntry], outpath: Path) -> None:
    """Write the assembled entries in the format that chargetransfer.cc of ARTIS reads."""
    # each comment starts with the tag of its source, so the header counts come from the entries
    source_counts = Counter(e.comment.split(";", 1)[0] for e in entries)
    with outpath.open("w", encoding="utf-8") as fout:
        fout.write(format_header(source_counts))
        fout.write(f"{len(entries)}\n")
        for e in entries:
            fout.write(
                f"{e.z_acc:3d} {e.ionstage_acc:2d} {e.z_don:3d} {e.ionstage_don:2d} {e.a:12.6g} {e.b:10.6g}"
                f" {e.c:10.6g} {e.d:10.6g} {e.eexp:9.6g} {e.tmin:8.6g} {e.tmax:8.6g} {e.autoreverse}"
                f" # {e.comment}\n"
            )
    print(f"wrote {len(entries)} reactions to {outpath}")


def main() -> None:
    """Read the sources of the charge transfer rates and write chargetransfer.txt for ARTIS."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaultoutpath = Path(__file__).parent.parent.absolute() / "artis_files" / "data" / "chargetransfer.txt"
    defaultsourcedir = Path(__file__).parent.parent.absolute() / "atomic-data-chargetransfer"
    parser.add_argument("-outputfile", default=defaultoutpath, type=Path, help="path of the output file")
    parser.add_argument(
        "-sourcedir",
        default=defaultsourcedir,
        type=Path,
        help="folder of the source files; setup_chargetransfer_data.sh in that folder downloads them",
    )
    args = parser.parse_args()

    print("Reactions with hydrogen (Cloudy carrying KF96 plus updates):")
    kf96_entries, kf96_report = get_kf96_h_entries(args.sourcedir)
    print("Recombination with neutral helium (AR85):")
    he_entries = get_he_entries(args.sourcedir)
    print("n-capture elements with hydrogen (SS11):")
    ss11_entries, ss11_report = get_ss11_entries(args.sourcedir)
    print("Near-resonant estimates (heavy ion with neutral heavy atom, and any element with hydrogen):")
    published_entries = kf96_entries + he_entries + ss11_entries
    covered = {(e.z_acc, e.ionstage_acc, e.z_don, e.ionstage_don) for e in published_entries}
    metal_entries = get_estimate_entries(covered)
    print(f"  {len(metal_entries)} reactions with an energy defect in (0, {METAL_METAL_MAX_DEFECT_EV}] eV")

    print("\nCloudy rows that differ from the KF96 paper tables:")
    for line in kf96_report:
        print(f"  {line}")
    print("\nSS11 fit quality:")
    for line in ss11_report:
        print(f"  {line}")
    print()

    args.outputfile.parent.mkdir(parents=True, exist_ok=True)
    entries = set_autoreverse_flags(published_entries + metal_entries)
    write_chargetransfer_file(entries, args.outputfile)


if __name__ == "__main__":
    main()

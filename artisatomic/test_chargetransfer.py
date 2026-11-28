"""Tests of makechargetransferfile, the generator of the charge transfer file for ARTIS.

Most tests build small source tables in memory and patch read_source(). Two tests read the
committed files in atomic-data-chargetransfer, so a change of their format fails here.
"""

import hashlib
import math
import sys
import typing as t
from collections.abc import Callable
from pathlib import Path

import pytest
from xopen import xopen

from artisatomic import makechargetransferfile


@pytest.fixture
def patch_sources(monkeypatch) -> Callable[[dict[str, str]], None]:
    """Give a function that makes read_source() return the given text for each file name."""

    def _patch(tables: dict[str, str]) -> None:
        monkeypatch.setattr(makechargetransferfile, "read_source", lambda filekey, _sourcedir: tables[filekey])

    return _patch


def _ss11_cds_line(elsymbol: str, q: int, label: str, rates: list[float]) -> str:
    """Build one fixed-column line of an SS11 CDS table."""
    header = f"{elsymbol:<2} {q}    {label}".ljust(32)
    assert len(header) == 32, "the label must fit in the fixed columns of the CDS format"
    return header + "".join(f" {rate:.2E}" for rate in rates)


def test_makechargetransferfile_cds_totals_and_fit():
    """read_cds_totals adds the channels above the radiative floor, and the fit recovers a power law."""
    powerlaw = [3.0e-10 * (temp / 1e4) ** 0.5 for temp in makechargetransferfile.SS11_TGRID]
    floor = [1.00e-14] * len(makechargetransferfile.SS11_TGRID)
    subfloor = [5.00e-21] * len(makechargetransferfile.SS11_TGRID)
    text = "\n".join(
        [
            _ss11_cds_line("Ge", 1, "4s^2^4p^2^ ^3^P", powerlaw),
            _ss11_cds_line("Ge", 1, "4s^2^4p^2^ ^1^D", floor),
            _ss11_cds_line("Ge", 2, "4s^2^4p ^2^P^o^", floor),
            _ss11_cds_line("Ge", 3, "4s4p ^1^P^o^", subfloor),
        ]
    )
    totals = makechargetransferfile.read_cds_totals(text)

    # a channel that sits on the floor carries no rate, so only the other channel counts
    assert totals["Ge", 1] == pytest.approx(powerlaw, rel=0.01)
    # every channel on the floor gives the floor itself, not zero
    assert totals["Ge", 2] == pytest.approx([1.00e-14] * len(makechargetransferfile.SS11_TGRID))
    # a reaction that needs energy has no channel on the floor, so its total stays below the floor
    assert totals["Ge", 3] == pytest.approx([5.00e-21] * len(makechargetransferfile.SS11_TGRID))

    tgrid = makechargetransferfile.SS11_TGRID
    fit = makechargetransferfile.fit_ss11_curve(totals["Ge", 1])
    assert fit.a == pytest.approx(3.0e-10, rel=0.01)
    assert fit.b == pytest.approx(0.5, rel=0.01)
    assert fit.eexp == 0.0
    assert fit.maxerr < 0.01
    assert fit.npts == 8
    # the eight usable points span the full window, so the fit is valid over all of it
    assert (fit.tmin, fit.tmax) == (1000, 40000)

    # the floor curve has no usable points, so the fit is flat at the 2e4 K value
    assert makechargetransferfile.fit_ss11_curve(totals["Ge", 2]) == (
        1.00e-14,
        0.0,
        0.0,
        0.0,
        0,
        20000,
        20000,
        0.0,
        0.0,
    )

    # A curve with one usable point gets the flat 2e4 K value. The result must report the error
    # of that value over the usable points.
    onepoint = [2.5e-13 if temp == 40000 else 1.00e-14 for temp in makechargetransferfile.SS11_TGRID]
    fallback = makechargetransferfile.fit_ss11_curve(onepoint)
    assert fallback.a == pytest.approx(1.0e-14)
    assert fallback.npts == 1
    assert fallback.maxerr == pytest.approx(0.96)
    assert (fallback.tmin, fallback.tmax, fallback.c, fallback.d) == (20000, 20000, 0.0, 0.0)

    # A curve that needs energy has no floor, so two values above zero give a flat value also.
    # get_ss11_entries() writes no line for such a curve.
    nofloor = [2.5e-13 if temp == 40000 else (5.0e-14 if temp == 20000 else 0.0) for temp in tgrid]
    assert makechargetransferfile.fit_ss11_floor_curve(nofloor, usable_min=0.0) is None
    assert makechargetransferfile.fit_ss11_curve(nofloor, usable_min=0.0).c == 0.0


def ctfit_rate(coefficients: t.Sequence[float], temperature: float) -> float:
    """Evaluate a line of chargetransfer.txt as evaluate_ctfit() of ARTIS does (chargetransfer.cc), in cm3/s."""
    a, b, c, d, eexp, tmin, tmax = coefficients
    t4 = min(max(temperature, tmin), tmax) / 1e4
    rate = a * 1e-9 * t4**b * (1.0 + c * math.exp(d * t4))
    return rate * math.exp(-eexp / temperature) if eexp > 0.0 else rate


def test_makechargetransferfile_keeps_the_temperature_dependence_of_a_sparse_floor_curve(tmp_path):
    """Ge3+ + H has two values above the floor in the fit range, and the written line must give both.

    The flat value of the 2e4 K entry is 3.36e-14 cm3/s, 18 times below the table at 4e4 K. The
    table is on the radiative floor of 1e-14 cm3/s up to 1e4 K, so the line must go to that floor.
    """
    sourcedir = Path(makechargetransferfile.PYDIR).parent / "atomic-data-chargetransfer"
    entries, report = makechargetransferfile.get_ss11_entries(sourcedir)
    (entry,) = [e for e in entries if (e.z_acc, e.ionstage_acc, e.z_don, e.ionstage_don) == (32, 4, 1, 1)]
    assert entry.comment == (
        "SS11; radiative floor plus an exponential, fit to their 2 values above the floor;"
        " max fit error 47% over 1000-40000 K"
    )
    assert any(line.startswith("rec Ge q=3: a=1.000e-14 b=0.000 c=9.392e-02 d=1.612") for line in report)

    # the numbers as ARTIS reads them from the file
    makechargetransferfile.write_chargetransfer_file(makechargetransferfile.set_autoreverse_flags(entries), tmp_path)
    (line,) = [
        line
        for line in (tmp_path / "chargetransfer.txt").read_text(encoding="utf-8").splitlines()
        if line.startswith("32 4 1 1 ")
    ]
    columns = line.split("#")[0].split()
    coefficients = [float(column) for column in columns[4:11]]
    assert coefficients[1] == coefficients[4] == 0.0
    assert coefficients[5:] == [1000.0, 40000.0]
    assert columns[11] == "1"

    assert ctfit_rate(coefficients, 20000.0) == pytest.approx(3.36e-14, rel=1e-4)
    assert ctfit_rate(coefficients, 40000.0) == pytest.approx(6.03e-13, rel=1e-4)
    # between the two values the rate rises
    assert 3.36e-14 < ctfit_rate(coefficients, 30000.0) < 6.03e-13
    # below 2e4 K the table is on the floor. The line stays above the floor and within 48% of it.
    for temperature in (1000.0, 2000.0, 4000.0, 6000.0, 8000.0, 10000.0):
        assert 1.0e-14 < ctfit_rate(coefficients, temperature) < 1.48e-14
    assert ctfit_rate(coefficients, 1000.0) == pytest.approx(1.11e-14, rel=0.01)
    # outside tmin and tmax the line holds its value, as each other line of the file
    assert ctfit_rate(coefficients, 100.0) == ctfit_rate(coefficients, 1000.0)
    assert ctfit_rate(coefficients, 1.0e5) == ctfit_rate(coefficients, 40000.0)


def test_fit_ss11_floor_curve_fits_three_points_by_least_squares():
    """Two usable points give their rates exactly. Three points give the line of least squares in ln(k/floor - 1)."""
    tgrid = makechargetransferfile.SS11_TGRID
    floor = makechargetransferfile.SS11_FLOOR
    c, d = 0.05, 1.5

    def curve(firsttemp: int) -> list[float]:
        return [floor * (1.0 + c * math.exp(d * temp / 1e4)) if temp >= firsttemp else floor for temp in tgrid]

    for firsttemp, npts in ((20000, 2), (10000, 3)):
        fit = makechargetransferfile.fit_ss11_curve(curve(firsttemp))
        assert (fit.a, fit.b, fit.eexp, fit.npts) == (floor, 0.0, 0.0, npts)
        assert (fit.c, fit.d) == pytest.approx((c, d))
        assert (fit.tmin, fit.tmax) == (1000, 40000)
    # four usable points take the fit of ln k, with c = d = 0
    assert makechargetransferfile.fit_ss11_curve(curve(8000)).c == 0.0


def test_makechargetransferfile_ss11_autoreverse(patch_sources, tmp_path):
    """A SS11 reaction keeps autoreverse=1 unless the tables hold a fit for its reverse.

    SS11 Table 5 gives the loss of an electron only for the neutral atom. Every recombination
    above the first ion stage therefore needs detailed balance for its reverse reaction.
    """
    rates = [3.0e-10 * (temp / 1e4) ** 0.5 for temp in makechargetransferfile.SS11_TGRID]
    tables = {
        "ss11_table4.dat": "\n".join(
            [
                _ss11_cds_line("Ge", 1, "4s^2^4p^2^ ^3^P", rates),
                _ss11_cds_line("Ge", 2, "4s^2^4p ^2^P^o^", rates),
            ]
        ),
        "ss11_table5.dat": _ss11_cds_line("Ge", 0, "4s^2^4p ^2^P^o^", rates),
    }
    patch_sources(tables)

    entries, _report = makechargetransferfile.get_ss11_entries(tmp_path)
    autoreverse_of_reaction = {
        (e.z_acc, e.ionstage_acc, e.z_don, e.ionstage_don): e.autoreverse
        for e in makechargetransferfile.set_autoreverse_flags(entries)
    }

    # Ge+1 + H0 -> Ge0 + H+ has the Table 5 row of Ge0 as its explicit reverse
    assert autoreverse_of_reaction[32, 2, 1, 1] == 0
    # Ge+2 + H0 -> Ge+1 + H+ has no fit for the reverse, so the code must add it
    assert autoreverse_of_reaction[32, 3, 1, 1] == 1
    # Ge0 + H+ -> Ge+1 + H0 has the Table 4 row of Ge+1 as its explicit reverse
    assert autoreverse_of_reaction[1, 2, 32, 1] == 0


def test_makechargetransferfile_ss11_fits_a_reaction_that_needs_energy(patch_sources, tmp_path):
    """A reaction that needs energy gets a fit to each table value above zero, and not a flat value.

    The curve 1e-12 exp(-60000 K / T) cm3/s has two values above the usable limit of 1.2e-14 cm3/s.
    A flat value of its 2e4 K entry is 3e-12 times too large at 2000 K. The reverse reaction has
    its own row, so both lines get autoreverse = 0.
    """
    # the table gives zero below 1000 K, where the value needs an exponent of three digits
    boltzmann = [
        1.0e-12 * math.exp(-60000.0 / temp) if temp >= 1000 else 0.0 for temp in makechargetransferfile.SS11_TGRID
    ]
    forward = [1.0e-12] * len(makechargetransferfile.SS11_TGRID)
    patch_sources(
        {
            "ss11_table4.dat": _ss11_cds_line("Se", 1, "4s^2^4p^4^ ^3^P", boltzmann),
            "ss11_table5.dat": _ss11_cds_line("Se", 0, "4s^2^4p^3^ ^2^P^o^", forward),
        }
    )

    entries, report = makechargetransferfile.get_ss11_entries(tmp_path)
    recombination, ionisation = makechargetransferfile.set_autoreverse_flags(entries)
    assert (recombination.z_acc, recombination.ionstage_acc, recombination.z_don) == (34, 2, 1)
    # the columns a and eexp: a is in units of 1e-9 cm3/s. The table has three significant digits.
    assert recombination.a == pytest.approx(1.0e-3, rel=0.02)
    assert recombination.eexp == pytest.approx(60000.0, rel=0.01)
    assert (recombination.tmin, recombination.tmax) == (1000, 40000)
    assert "takes each value above zero" in recombination.comment
    assert (recombination.autoreverse, ionisation.autoreverse) == (0, 0)
    assert not any("no line" in line for line in report)

    # a curve with fewer than four values above zero has no line
    fewvalues = [
        value if temp >= 20000 else 0.0
        for temp, value in zip(makechargetransferfile.SS11_TGRID, boltzmann, strict=True)
    ]
    patch_sources({"ss11_table4.dat": _ss11_cds_line("Se", 1, "4s^2^4p^4^ ^3^P", fewvalues), "ss11_table5.dat": ""})
    entries, report = makechargetransferfile.get_ss11_entries(tmp_path)
    assert entries == []
    assert "no line" in report[0]


def _cloudy_table(magic: str, ncols: int, rows: dict[tuple[int, int], list[float]]) -> str:
    """Build the text of a charge transfer file of Cloudy, with an empty row for each ion not given."""
    lines = [f"  {magic}"]
    lines += [
        "  ".join(f"{value:.5e}" for value in rows.get((atomic_number, ionindex), [0.0] * ncols))
        for atomic_number in range(1, 31)
        for ionindex in range(1, 5)
    ]
    return "\n".join(lines)


def test_makechargetransferfile_kf96_autoreverse(patch_sources, tmp_path):
    """A Cloudy reaction keeps autoreverse=1 unless the files hold a fit for its reverse.

    Cloudy leaves an empty row for a reaction that it does not carry, in either direction. Such a
    row gives no reverse fit, so the code must add the reverse reaction from detailed balance. A
    KF96 Table 4 row (a zero rate with a temperature range) gets no entry at all.
    """
    fit = [2.0, 0.5, 0.0, 0.0]
    tables = {
        # S+1 + H0 -> S0 + H+, Fe+3 + H0 -> Fe+2 + H+, and the endothermic Ca+2 + H0 with no fit
        "ctrecombdata.dat": _cloudy_table(
            "201903042",
            7,
            {
                (16, 1): [*fit, 1e3, 3e4, 1.0],
                (26, 3): [*fit, 1e3, 3e4, 1.0],
                (20, 2): [0.0, 0.0, 0.0, 0.0, 10.0, 1e9, 0.0],
            },
        ),
        # S0 + H+ -> S+1 + H0 and Li0 + H+ -> Li+1 + H0
        "ctiondata.dat": _cloudy_table(
            "201903041", 8, {(16, 1): [*fit, 1e3, 3e4, 0.0, 1.0], (3, 1): [*fit, 1e3, 3e4, 0.0, 1.0]}
        ),
    }
    patch_sources(tables)

    entries, report = makechargetransferfile.get_kf96_h_entries(tmp_path)
    autoreverse_of_reaction = {
        (e.z_acc, e.ionstage_acc, e.z_don, e.ionstage_don): e.autoreverse
        for e in makechargetransferfile.set_autoreverse_flags(entries)
    }

    # the S reactions are a pair, so each one is the explicit reverse of the other
    assert autoreverse_of_reaction[16, 2, 1, 1] == 0
    assert autoreverse_of_reaction[1, 2, 16, 1] == 0
    # Fe+3 + H0 -> Fe+2 + H+ has no ionisation row for Fe+2. Li0 + H+ -> Li+1 + H0 has no
    # recombination row for Li+1.
    assert autoreverse_of_reaction[26, 4, 1, 1] == 1
    assert autoreverse_of_reaction[1, 2, 3, 1] == 1
    # a zero rate would stop ARTIS from its own estimate, so the script reports the Ca+2 row and does not write it
    assert (20, 3, 1, 1) not in autoreverse_of_reaction
    assert any("Z=20 q=2" in line and "Table 4" in line for line in report)


def test_makechargetransferfile_ar85_helium_entries(patch_sources, tmp_path):
    """The AR85 file gives the electron count of the product ion, which needs a conversion."""
    text = " 8  6  1.00E+00  0.00E+00  1.25E+00 -5.80E+00  1.00E+03  3.00E+04"
    patch_sources({"ar85_ct2.dat": text})

    (entry,) = makechargetransferfile.set_autoreverse_flags(makechargetransferfile.get_he_entries(tmp_path))

    # the product O+2 holds the six electrons of the file, so O+3 is the ion that captures one
    assert (entry.z_acc, entry.ionstage_acc, entry.z_don, entry.ionstage_don) == (8, 4, 2, 1)
    assert (entry.a, entry.b, entry.c, entry.d) == (1.0, 0.0, 1.25, -5.8)
    assert (entry.tmin, entry.tmax) == (1000.0, 30000.0)
    # the file holds no fit for He+ + O+2, so the code must add that reverse reaction
    assert entry.autoreverse == 1


def test_set_autoreverse_flags_rejects_a_duplicate_reaction():
    """The assembled list must hold each reaction once, because the reader takes the first fit."""
    entry = makechargetransferfile.CTEntry(26, 2, 1, 1, 1.0, 0.0, 0.0, 0.0, 0.0, 1e3, 4e4, "Test; Fe+1 + H0")
    with pytest.raises(ValueError, match=r"more than once"):
        makechargetransferfile.set_autoreverse_flags([entry, entry])


def test_committed_source_files_build_the_expected_entries():
    """The committed source files parse, and the assembled list holds the expected count for each source.

    This test fails when a new download changes the format of a table, which the other tests
    cannot see because they patch read_source().
    """
    sourcedir = Path(makechargetransferfile.__file__).parent.parent / "atomic-data-chargetransfer"
    kf96_entries, _ = makechargetransferfile.get_kf96_h_entries(sourcedir)
    he_entries = makechargetransferfile.get_he_entries(sourcedir)
    ss11_entries, _ = makechargetransferfile.get_ss11_entries(sourcedir)
    assert (len(kf96_entries), len(he_entries), len(ss11_entries)) == (100, 21, 36)
    entries = makechargetransferfile.set_autoreverse_flags(kf96_entries + he_entries + ss11_entries)
    assert all(e.comment.split(";", 1)[0] in {"Cloudy", "AR85", "SS11"} for e in entries)


def test_read_source_reads_the_compressed_file(tmp_path):
    """read_source finds the zstd compressed form of a source file, and names the setup script otherwise."""
    with xopen(tmp_path / "table.dat.zst", "wt", encoding="utf-8") as f:
        f.write("1 2 3\n")

    assert makechargetransferfile.read_source("table.dat", tmp_path) == "1 2 3\n"

    with pytest.raises(FileNotFoundError, match=r"setup_chargetransfer_data\.sh"):
        makechargetransferfile.read_source("absent.dat", tmp_path)


def test_chargetransfer_output_matches_the_checksum(tmp_path, monkeypatch):
    """main() writes the file that tests/chargetransfer/checksums.txt records, byte for byte.

    The CI job 'test chargetransfer' checks the same file with md5sum. Regenerate the checksum
    after a deliberate change of the output. tests/README.md gives the commands.
    """
    monkeypatch.setattr(sys, "argv", ["makeartischargetransferfile", "-output_folder", str(tmp_path)])
    makechargetransferfile.main()

    checksumfile = Path(makechargetransferfile.__file__).parent.parent / "tests" / "chargetransfer" / "checksums.txt"
    expected = {name: digest for digest, name in (line.split() for line in checksumfile.read_text().splitlines())}
    assert set(expected) == {"chargetransfer.txt"}
    for name, digest in expected.items():
        assert hashlib.md5((tmp_path / name).read_bytes(), usedforsecurity=False).hexdigest() == digest, name

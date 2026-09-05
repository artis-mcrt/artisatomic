"""Tests of makechargetransferfile, the generator of the charge transfer file for ARTIS.

Most tests build small source tables in memory and patch read_source(). One test reads the
committed files in atomic-data-chargetransfer, so a change of their format fails here.
"""

import hashlib
import sys
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
    # the paper floors the total, so a sum below the floor also becomes the floor
    assert totals["Ge", 3] == pytest.approx([1.00e-14] * len(makechargetransferfile.SS11_TGRID))

    fit = makechargetransferfile.fit_ss11_curve(totals["Ge", 1])
    assert fit.a == pytest.approx(3.0e-10, rel=0.01)
    assert fit.b == pytest.approx(0.5, rel=0.01)
    assert fit.eexp == 0.0
    assert fit.maxerr < 0.01
    assert fit.npts == 8
    # the eight usable points span the full window, so the fit is valid over all of it
    assert (fit.tmin, fit.tmax) == (1000, 40000)

    # the floor curve has no usable points, so the fit is flat at the 2e4 K value
    assert makechargetransferfile.fit_ss11_curve(totals["Ge", 2]) == (1.00e-14, 0.0, 0.0, 0.0, 0.0, 20000, 20000)

    # a curve with fewer than four usable points gets the flat 2e4 K value. The result must
    # report the error of that value over the usable points.
    steep = [
        2.5e-13 if temp == 40000 else (5.0e-14 if temp == 20000 else 1.00e-14)
        for temp in makechargetransferfile.SS11_TGRID
    ]
    fallback = makechargetransferfile.fit_ss11_curve(steep)
    assert fallback.a == pytest.approx(5.0e-14)
    assert fallback.npts == 2
    assert fallback.maxerr == pytest.approx(0.8)
    assert (fallback.tmin, fallback.tmax) == (20000, 20000)


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
    monkeypatch.setattr(sys, "argv", ["makechargetransferfile", "-output_folder", str(tmp_path)])
    makechargetransferfile.main()

    checksumfile = Path(makechargetransferfile.__file__).parent.parent / "tests" / "chargetransfer" / "checksums.txt"
    expected = {name: digest for digest, name in (line.split() for line in checksumfile.read_text().splitlines())}
    assert set(expected) == {"chargetransfer.txt"}
    for name, digest in expected.items():
        assert hashlib.md5((tmp_path / name).read_bytes(), usedforsecurity=False).hexdigest() == digest, name

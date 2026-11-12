"""Test the complete gamma and recombination commands without network access or external databases."""

import math
from types import SimpleNamespace
from typing import Self

import numpy as np
import polars as pl
import pytest
import requests

from artisatomic import download_gammaspec_betaminus_alpha as gamma
from artisatomic import makerecombratefile as recomb

NUDAT_HEADER = (
    "A  \tElement\tZ  \tN  \tPar. Elevel\tUnc. \tJPi       \tDec Mode\tT1/2 (txt)    \tT1/2 (num)       "
    " \tDaughter\tRadiation\tRad subtype \tRad Ene.  \tUnc       \tEP Ene.   \tUnc       \tRad Int.  \tUnc      "
    " \tDose        \tUnc"
)


def nudat_table(rows: list[tuple[str, str, str, str]]) -> str:
    """Make a complete NuDat response with explicit energy, intensity, parent level, and radiation type."""
    lines = [NUDAT_HEADER]
    for energy, intensity, parent, radiation in rows:
        lines.append(
            f"44\tSc\t21\t23\t{parent}\t\t0+\tB-\t1 s\t1\tTi\t{radiation}\t\t{energy}\t\t\t\t{intensity}\t\t\t"
        )
    return "<pre>" + "\n".join(lines) + "</pre>"


@pytest.fixture
def gamma_run(tmp_path, monkeypatch):
    """Supply local decay lists and a controlled response to the complete gamma command."""
    monkeypatch.setattr("sys.argv", ["makeartisgammaspecfiles"])
    data = tmp_path / "input"
    data.mkdir()
    (data / "betaminusdecays.txt").write_text("44 21 1.0 0.0 0.0 0.0 1.4426950408889634\n")
    (data / "alphadecays.txt").write_text("44 21 0 0 0 0 0 0 0 0\n")
    monkeypatch.setattr(gamma.at, "get_path", lambda _name: data)
    monkeypatch.setattr(gamma, "PYDIR", tmp_path / "package")
    monkeypatch.setattr(gamma, "nuclides_with_artis_hardcoded_decay_data", ())
    state = SimpleNamespace(text="", error=None, requests=0)

    class Response:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        @property
        def text(self):
            return state.text

        @staticmethod
        def raise_for_status() -> None:
            if state.error is not None:
                raise state.error

    class Session:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        @staticmethod
        def mount(*_args: object) -> None:
            pass

        @staticmethod
        def get(_url, **_kwargs: object) -> Response:
            state.requests += 1
            return Response()

    monkeypatch.setattr(gamma.requests, "Session", Session)
    output = tmp_path / "artis_files" / "data"
    output.mkdir(parents=True)
    return state, output / "gamma_sc44.txt"


def test_gamma_command_preserves_weak_lines_and_energy_precision(gamma_run):
    """The file must retain weak lines and the precision of the parsed energies and intensities."""
    state, path = gamma_run
    state.text = nudat_table(
        [
            ("1234.56789", "12.3456789", "0", "G"),
            ("0.4", "0.0001", "0.00", "G"),
            ("9999", "100", "10", "G"),
            ("5", "100", "0", "B-"),
        ]
    )
    gamma.main()
    lines = path.read_text().splitlines()
    assert lines[0] == "2"
    rows = [tuple(map(float, line.split())) for line in lines[1:]]
    assert rows[0] == pytest.approx((0.0004, 0.000001), rel=1e-15)
    assert rows[1] == pytest.approx((1.23456789, 0.123456789), rel=1e-15)
    assert math.fsum(energy * intensity for energy, intensity in rows) == pytest.approx(
        0.0004 * 0.000001 + 1.23456789 * 0.123456789, rel=1e-15
    )
    assert state.requests == 1


@pytest.mark.parametrize(
    ("energy", "intensity"),
    [("0", "1"), ("-1", "1"), ("NaN", "1"), ("1", "inf"), ("1", "-inf"), ("1", "NaN"), ("1", "-1"), ("1", "")],
)
def test_gamma_command_keeps_the_previous_file_for_invalid_values(gamma_run, energy, intensity):
    """An invalid selected line must not replace the earlier spectrum."""
    state, path = gamma_run
    path.write_text("earlier spectrum")
    state.text = nudat_table([(energy, intensity, "0", "G")])
    with pytest.raises(ValueError, match="invalid energy or intensity"):
        gamma.main()
    assert path.read_text() == "earlier spectrum"


@pytest.mark.parametrize("case", ["request", "header", "table", "ground"])
def test_gamma_command_keeps_the_previous_file_after_a_source_failure(gamma_run, case):
    """A failed request or an unusable table must leave the earlier spectrum intact."""
    state, path = gamma_run
    path.write_text("earlier spectrum")
    if case == "request":
        state.error = requests.HTTPError("The request failed.")
        error = requests.HTTPError
    elif case == "header":
        state.text = "<pre>unknown header</pre>"
        error = ValueError
    elif case == "table":
        state.text = "No table"
        error = SystemExit
    else:
        state.text = nudat_table([("1", "1", "10", "G")])
        error = SystemExit
    with pytest.raises(error):
        gamma.main()
    assert path.read_text() == "earlier spectrum"


def test_gamma_command_removes_an_earlier_file_for_a_valid_empty_spectrum(gamma_run):
    """A valid ground-state table without gamma lines must remove the earlier spectrum."""
    state, path = gamma_run
    path.write_text("earlier spectrum")
    state.text = nudat_table([("1", "1", "0", "B-")])
    gamma.main()
    assert not path.exists()


@pytest.fixture
def recomb_run(tmp_path, monkeypatch):
    """Supply a small composition and a controlled Chianti object to the complete recombination command."""
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr("sys.argv", ["makeartisrecombratefile", "-output_folder", str(output)])
    monkeypatch.setattr(recomb, "PYDIR", tmp_path / "package")
    state = SimpleNamespace(z=2, lower=1, rr=2.0, dr=3.0, rr_source=True, dr_source=True, rr_length=None)
    monkeypatch.setattr(
        recomb,
        "get_composition_data",
        lambda _path: pl.DataFrame(
            {"Z": [state.z], "lowermost_ion_stage": [state.lower], "uppermost_ion_stage": [state.lower + 1]}
        ),
    )

    class Ion:
        def __init__(self, temperature) -> None:
            self.count = len(temperature)

        def rrRate(self):
            if state.rr_source:
                self.RrParams = {"rrtype": 1}
            self.RrRate = {"rate": np.full(state.rr_length or self.count, state.rr)}

        def drRate(self):
            if state.dr_source:
                self.DrParams = {"drtype": 1}
            self.DrRate = {"rate": np.full(self.count, state.dr)}

    core = SimpleNamespace(ion=lambda _name, temperature: Ion(temperature))
    monkeypatch.setattr(recomb, "import_chianti_core", lambda _ion: core)
    return state, output / "recombrates.txt", tmp_path


def test_recombination_command_writes_both_chianti_rates(recomb_run):
    """Each total must include the radiative and dielectronic rates at the requested temperature."""
    _state, path, _root = recomb_run
    recomb.main()
    lines = path.read_text().splitlines()
    assert lines[0] == "2 2 81"
    rows = [tuple(map(float, line.split())) for line in lines[1:]]
    assert [row[0] for row in rows] == [round(1 + i / 10, 1) for i in range(81)]
    assert all(row[1:] == (-1.0, 5.0) for row in rows)


@pytest.mark.parametrize("case", ["rr_source", "dr_source", "negative", "nonfinite", "length", "overflow"])
def test_recombination_command_keeps_the_previous_file_for_invalid_chianti_data(recomb_run, case):
    """Absent source data or invalid rate values must not replace the earlier rate file."""
    state, path, _root = recomb_run
    path.write_text("earlier rates")
    if case in {"rr_source", "dr_source"}:
        setattr(state, case, False)
    elif case == "negative":
        state.rr = -1.0
    elif case == "nonfinite":
        state.rr = float("nan")
    elif case == "length":
        state.rr_length = 2
    else:
        state.rr = state.dr = np.finfo(float).max
    with pytest.raises(ValueError, match=r"parameters|finite|rates"):
        recomb.main()
    assert path.read_text() == "earlier rates"


def test_recombination_command_accepts_no_dielectronic_source_for_a_bare_nucleus(recomb_run):
    """A bare nucleus needs only the radiative source."""
    state, path, _root = recomb_run
    state.z = 1
    state.dr_source = False
    state.dr = 0.0
    recomb.main()
    assert path.read_text().splitlines()[1] == "1.0 -1.0 2.0"


def test_recombination_command_reads_nahar_without_chianti(recomb_run, monkeypatch):
    """A Nahar table supplies the complete command without a Chianti import."""
    _state, path, root = recomb_run
    source = root / "atomic-data-nahar"
    source.mkdir()
    (source / "he1.rrc.txt").write_text(
        "TOTAL RECOMBINATION RATE\n\n\nlog(T) RRC(low n) RRC(total)\n1.0 2e-12 3e-12\n2.0 4e-12 5e-12\n"
    )
    monkeypatch.setattr(recomb, "import_chianti_core", lambda _ion: pytest.fail("Chianti must not load."))
    recomb.main()
    assert path.read_text().splitlines() == ["2 2 2", "1.0 2e-12 3e-12", "2.0 4e-12 5e-12"]


@pytest.mark.parametrize("rows", ["1 2 3\n1 4 5", "2 2 3\n1 4 5", "NaN 2 3", "1 -2 3", "1 2 inf"])
def test_nahar_validation_rejects_invalid_temperatures_or_rates(tmp_path, rows):
    """The temperature order and the finite, non-negative rates are constraints of the output table."""
    path = tmp_path / "rates.txt"
    path.write_text("TOTAL RECOMBINATION RATE\n\n\nlog(T) RRC(low n) RRC(total)\n" + rows)
    with pytest.raises(ValueError, match=r"temperatures|non-finite"):
        recomb.read_nahar_rrcfile(path)


def test_recombination_command_accepts_zero_rates_with_source_parameters(recomb_run):
    """A valid zero rate must remain distinct from absent source data."""
    state, path, _root = recomb_run
    state.rr = state.dr = 0.0
    recomb.main()
    assert all(line.split()[2] == "0.0" for line in path.read_text().splitlines()[1:])


def test_gamma_command_accepts_a_finite_zero_intensity(gamma_run):
    """A finite zero intensity gives no line and can remove an earlier spectrum."""
    state, path = gamma_run
    path.write_text("earlier spectrum")
    state.text = nudat_table([("1", "0", "0", "G")])
    gamma.main()
    assert not path.exists()

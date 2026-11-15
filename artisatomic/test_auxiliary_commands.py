"""Test the complete gamma and recombination commands with no network request and no Chianti database."""

import io
import math
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from artisatomic import download_gammaspec_betaminus_alpha as gamma
from artisatomic import makerecombratefile as recomb

NUDAT_HEADER = (
    "A  \tElement\tZ  \tN  \tPar. Elevel\tUnc. \tJPi       \tDec Mode\tT1/2 (txt)    \tT1/2 (num)       "
    " \tDaughter\tRadiation\tRad subtype \tRad Ene.  \tUnc       \tEP Ene.   \tUnc       \tRad Int.  \tUnc      "
    " \tDose        \tUnc"
)


def nudat_page(rows: list[tuple[str, str, str, str]]) -> str:
    """Give a NuDat page of Sc44 with one row for each (energy in keV, intensity in percent, parent level, radiation)."""
    lines = [NUDAT_HEADER]
    lines.extend(
        f"44\tSc\t21\t23\t{parent}\t\t2+\tEC\t4 h\t14000\tCa\t{radiation}\t\t{energy}\t\t\t\t{intensity}\t\t\t"
        for energy, intensity, parent, radiation in rows
    )
    return "<pre>" + "\n".join(lines) + "</pre>"


@pytest.fixture
def gamma_run(tmp_path, monkeypatch):
    """Give the gamma command two small decay tables and a fixed NuDat page as the response to each request."""
    (tmp_path / "betaminusdecays.txt").write_text("44 21 1.0 0.5 0.3 0.2 100.0\n", encoding="utf-8")
    (tmp_path / "alphadecays.txt").write_text("210 84 1 0 0 5 0 5 0 0\n", encoding="utf-8")
    page = SimpleNamespace(text="", status=200)

    def send(_request: requests.PreparedRequest, **_kwargs: object) -> requests.Response:
        response = requests.Response()
        response.status_code = page.status
        response.raw = io.BytesIO(page.text.encode())
        return response

    monkeypatch.setattr("sys.argv", ["makeartisgammaspecfiles"])
    monkeypatch.setattr(gamma, "PYDIR", tmp_path / "artisatomic")
    monkeypatch.setattr(gamma.at, "get_path", lambda _name: tmp_path)
    # the session of the command sends each https request through this stand-in for HTTPAdapter
    monkeypatch.setattr(gamma, "HTTPAdapter", lambda **_kwargs: SimpleNamespace(send=send, close=lambda: None))
    return page, tmp_path / "artis_files" / "data" / "gamma_sc44.txt"


def test_gamma_file_keeps_each_number_and_the_total_energy(gamma_run):
    """The file gives each number as the shortest text of a 64-bit float, so it keeps the total energy of the table.

    The table has a line below 0.001 MeV and a line below 5e-5 per decay. The file holds only the gamma lines of
    the ground level. The tolerance of 1e-15 permits the rounding of the two unit conversions and of the sum.
    """
    page, path = gamma_run
    lines_kev_percent = [("0.4", "12.5"), ("1157.002", "99.875"), ("2656.41", "0.0012")]
    page.text = nudat_page(
        [(energy, intensity, "0.0", "G") for energy, intensity in lines_kev_percent]
        + [("271.24", "86.7", "270.95", "G"), ("632.0", "94.27", "0.0", "B+")]
    )

    gamma.main()

    count, *rows = path.read_text(encoding="utf-8").splitlines()
    expected = [(float(energy) / 1000.0, float(intensity) / 100.0) for energy, intensity in lines_kev_percent]
    assert int(count) == len(expected)
    assert all(repr(float(text)) == text for row in rows for text in row.split())
    written = [tuple(map(float, row.split())) for row in rows]
    assert written == [pytest.approx(line, rel=1e-15, abs=0.0) for line in expected]
    assert math.fsum(energy * intensity for energy, intensity in written) == pytest.approx(
        math.fsum(energy * intensity for energy, intensity in expected), rel=1e-15, abs=0.0
    )


@pytest.mark.parametrize(
    ("text", "status", "error"),
    [
        (nudat_page([("1157.002", "99.875", "0.0", "G")]), 503, requests.HTTPError),
        ("<pre>A\tElement\tZ</pre>", 200, ValueError),
        ("NuDat has no data for this nuclide.", 200, SystemExit),
        (nudat_page([("271.24", "86.7", "270.95", "G")]), 200, SystemExit),
    ],
    ids=["failed request", "unknown header", "no table", "no ground level"],
)
def test_gamma_command_keeps_the_earlier_file_when_the_source_fails(gamma_run, text, status, error):
    """A failed request or a page with no usable table stops the command and keeps the file of an earlier run."""
    page, path = gamma_run
    page.text, page.status = text, status
    path.parent.mkdir(parents=True)
    path.write_text("earlier spectrum", encoding="utf-8")

    with pytest.raises(error):
        gamma.main()

    assert path.read_text(encoding="utf-8") == "earlier spectrum"


@pytest.mark.parametrize("earlier_file", [True, False])
def test_gamma_command_leaves_no_file_for_an_empty_spectrum(gamma_run, earlier_file):
    """A ground level with no gamma line gets no file, and the command deletes the file of an earlier run."""
    page, path = gamma_run
    page.text = nudat_page([("632.0", "94.27", "0.0", "B+"), ("1157.002", "0", "0.0", "G")])
    if earlier_file:
        path.parent.mkdir(parents=True)
        path.write_text("earlier spectrum", encoding="utf-8")

    gamma.main()

    assert not path.exists()


class ChiantiIon:
    """A stand-in for a ChiantiPy ion. Like ChiantiPy, it gives a rate of zero and no parameters for an absent file."""

    def __init__(self, rates: dict[str, object], temperature: np.ndarray) -> None:
        """Keep the rates of the test and the temperatures of the command."""
        self.rates = rates
        self.temperature = temperature

    def rate(self, process: str) -> dict[str, object]:
        """Give the rate of one process: an array of the test as it is, or one value at each temperature."""
        value = self.rates.get(process)
        if isinstance(value, np.ndarray):
            return {"rate": value}
        return {"rate": np.full_like(self.temperature, 0.0 if value is None else value)}

    def rrRate(self) -> None:
        """Set the radiative rate, and the parameters only if the test gives a rate."""
        if self.rates.get("rr") is not None:
            self.RrParams = {"rrtype": 1}
        self.RrRate = self.rate("rr")

    def drRate(self) -> None:
        """Set the dielectronic rate, and the parameters only if the test gives a rate."""
        if self.rates.get("dr") is not None:
            self.DrParams = {"drtype": 1}
        self.DrRate = self.rate("dr")


@pytest.fixture
def recomb_run(tmp_path, monkeypatch):
    """Give the recombination command C I to C III, a Nahar file for C I, and Chianti rates that the test sets."""
    output = tmp_path / "artis_files"
    output.mkdir()
    (output / "compositiondata.txt").write_text("1\n0\n0\n6 3 1 3 -1 0.0 12.0107\n", encoding="utf-8")
    nahar = tmp_path / "atomic-data-nahar"
    nahar.mkdir()
    naharfile = nahar / "c1.rrc.txt"
    nahar_header = "TOTAL RECOMBINATION RATE\n\n\nlog(T) RRC(low n) RRC(total)\n"
    naharfile.write_text(nahar_header + "1.0 2e-12 3e-12\n1.1 4e-12 5e-12\n", encoding="utf-8")
    rates: dict[str, object] = {"rr": 2.0, "dr": 3.0}
    chianti = SimpleNamespace(ion=lambda _name, temperature: ChiantiIon(rates, temperature))

    monkeypatch.setattr("sys.argv", ["makeartisrecombratefile", "-output_folder", str(output)])
    monkeypatch.setattr(recomb, "PYDIR", tmp_path / "artisatomic")
    monkeypatch.setattr(recomb, "import_chianti_core", lambda _firstion: chianti)
    return SimpleNamespace(
        rates=rates, output=output / "recombrates.txt", naharfile=naharfile, nahar_header=nahar_header
    )


def test_recombination_command_writes_the_nahar_rates_and_the_chianti_rates(recomb_run):
    """C I has a Nahar file, and C II takes the sum of the two Chianti rates at each temperature."""
    recomb.main()

    lines = recomb_run.output.read_text(encoding="utf-8").splitlines()
    assert lines[:4] == ["6 2 2", "1.0 2e-12 3e-12", "1.1 4e-12 5e-12", "6 3 81"]
    assert lines[4:] == [f"{1.0 + 0.1 * index:.1f} -1.0 5.0" for index in range(81)]


@pytest.mark.parametrize(
    ("atomic_number", "rates", "total"),
    [(6, {"rr": 0.0, "dr": 0.0}, 0.0), (1, {"rr": 2e-12, "dr": None}, 2e-12)],
    ids=["zero rates with parameters", "bare nucleus with no dielectronic parameters"],
)
def test_chianti_rates_accept_a_zero_rate_that_has_parameters(atomic_number, rates, total):
    """A rate of zero is valid when Chianti has the parameters, or when a bare nucleus needs none."""
    temperatures = np.array([1e3, 1e4])
    chianti = SimpleNamespace(ion=lambda _name, temperature: ChiantiIon(rates, temperature))

    result = recomb.chianti_total_rates(chianti, atomic_number, atomic_number + 1, temperatures)

    assert result.tolist() == [total, total]


@pytest.mark.parametrize(
    ("rates", "message"),
    [
        ({"rr": None}, "no radiative recombination parameters for c_3"),
        ({"dr": None}, "no dielectronic recombination parameters for c_3"),
        ({"rr": np.ones(2)}, "one finite rate for each of the 81 temperatures of c_3"),
        ({"rr": math.nan}, "one finite rate for each of the 81 temperatures of c_3"),
    ],
    ids=["no radiative parameters", "no dielectronic parameters", "wrong array length", "rate not finite"],
)
def test_recombination_command_keeps_the_earlier_file_for_invalid_chianti_data(recomb_run, rates, message):
    """Absent Chianti parameters, a wrong array length or a rate that is not finite stops the command before output."""
    recomb_run.rates.update(rates)
    recomb_run.output.write_text("earlier rates", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        recomb.main()

    assert recomb_run.output.read_text(encoding="utf-8") == "earlier rates"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("1.1 2e-12 3e-12\n1.0 4e-12 5e-12\n", "increasing order"),
        ("1.0 2e-12 3e-12\n1.0 4e-12 5e-12\n", "increasing order"),
        ("1.0 2e-12 nan\n", "not finite"),
        ("1.0 inf 3e-12\n", "not finite"),
    ],
    ids=["decreasing temperature", "repeated temperature", "NaN rate", "infinite rate"],
)
def test_recombination_command_keeps_the_earlier_file_for_an_invalid_nahar_table(recomb_run, rows, message):
    """A Nahar table with a wrong temperature order or a value that is not finite stops the command before output."""
    recomb_run.naharfile.write_text(recomb_run.nahar_header + rows, encoding="utf-8")
    recomb_run.output.write_text("earlier rates", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        recomb.main()

    assert recomb_run.output.read_text(encoding="utf-8") == "earlier rates"

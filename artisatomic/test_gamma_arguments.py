"""Check the gamma command: its arguments, its decay tables and its use of the NuDat responses, with no network."""

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from artisatomic import download_gammaspec_betaminus_alpha as gamma


@pytest.mark.parametrize(("argument", "status"), [("--help", 0), ("--unknown", 2)])
def test_arguments_stop_before_conversion(monkeypatch, capsys, argument, status):
    """Help and invalid arguments must leave the data and output files unchanged."""

    def fail_operation(*_args: object, **_kwargs: object):
        msg = "The argument check must precede conversion."
        raise AssertionError(msg)

    monkeypatch.setattr("sys.argv", ["makeartisgammaspecfiles", argument])
    monkeypatch.setattr(Path, "mkdir", fail_operation)
    monkeypatch.setattr(gamma.at, "get_path", fail_operation)
    monkeypatch.setattr(gamma.requests, "Session", fail_operation)
    with pytest.raises(SystemExit) as error:
        gamma.main()
    assert error.value.code == status
    captured = capsys.readouterr()
    assert "usage:" in captured.out + captured.err


def write_decay_tables(folder: Path, duplicated: str) -> None:
    """Write the two decay tables into the folder, with two rows for one nuclide in the duplicated table."""
    rows = {"betaminusdecays.txt": "44 21 1.0 0.5 0.3 0.2 100.0\n", "alphadecays.txt": "210 84 1 0 2.0 5 0 5 0 0\n"}
    rows[duplicated] *= 2
    for filename, text in rows.items():
        (folder / filename).write_text(text, encoding="utf-8")


class NetworkRequestError(Exception):
    """The command got to its first network request."""


def stop_at_request(*_args: object, **_kwargs: object):
    """Stand in for requests.Session."""
    raise NetworkRequestError


def test_a_nuclide_with_two_beta_minus_rows_stops_the_command(monkeypatch, tmp_path):
    """A beta-minus table with two rows for one nuclide stops the command before a network request."""
    write_decay_tables(tmp_path, "betaminusdecays.txt")
    monkeypatch.setattr("sys.argv", ["makeartisgammaspecfiles"])
    monkeypatch.setattr(gamma, "PYDIR", tmp_path / "artisatomic")
    monkeypatch.setattr(gamma.at, "get_path", lambda _name: tmp_path)
    monkeypatch.setattr(gamma.requests, "Session", stop_at_request)
    with pytest.raises(ValueError, match="The beta minus table holds a nuclide more than one time"):
        gamma.main()


def test_a_nuclide_with_two_alpha_rows_does_not_stop_the_command(monkeypatch, tmp_path):
    """An alpha table with two rows for one nuclide lets the command continue to the network request."""
    write_decay_tables(tmp_path, "alphadecays.txt")
    monkeypatch.setattr("sys.argv", ["makeartisgammaspecfiles"])
    monkeypatch.setattr(gamma, "PYDIR", tmp_path / "artisatomic")
    monkeypatch.setattr(gamma.at, "get_path", lambda _name: tmp_path)
    monkeypatch.setattr(gamma.requests, "Session", stop_at_request)
    with pytest.raises(NetworkRequestError):
        gamma.main()


def nudat_row(parent_elevel: str, radiation: str, energy_kev: str, intensity: str) -> str:
    """Give one row of a NuDat decay table of Sc-44, with the fields in the order of the header."""
    fields = ["44", "Sc", "21", "23", parent_elevel, "", "2+", "EC", "3.97 H 4", "14292.0", "44Ca", radiation, ""]
    return "\t".join([*fields, energy_kev, "", "", "", intensity, "", "", ""])


def nudat_response(rows: list[str], *, complete: bool = True) -> str:
    """Give the text of a NuDat response with these rows. A response that is not complete has no </pre>."""
    table = "\n".join([gamma.nudat_header, *rows])
    return f"<html><body><pre>\n{table}\n" + ("</pre></body></html>\n" if complete else "")


def run_with_responses(monkeypatch, tmp_path: Path, sc44_response: str) -> Path:
    """Run the command with a stand-in for the network, and give the path of the Sc-44 spectrum.

    The decay tables hold only Sc-44, which is also the nuclide with hard-coded decay data in ARTIS.
    """
    (tmp_path / "betaminusdecays.txt").write_text("44 21 1.0 0.5 0.3 0.2 100.0\n", encoding="utf-8")
    (tmp_path / "alphadecays.txt").write_text("44 21 1 0 2.0 5 0 5 0 0\n", encoding="utf-8")

    def get(url: str, **_kwargs: object) -> contextlib.AbstractContextManager[SimpleNamespace]:
        assert "nuc=sc44&" in url
        return contextlib.nullcontext(SimpleNamespace(text=sc44_response, raise_for_status=lambda: None))

    def session() -> contextlib.AbstractContextManager[SimpleNamespace]:
        return contextlib.nullcontext(SimpleNamespace(mount=lambda *_args: None, get=get))

    monkeypatch.setattr("sys.argv", ["makeartisgammaspecfiles"])
    monkeypatch.setattr(gamma, "PYDIR", tmp_path / "artisatomic")
    monkeypatch.setattr(gamma.at, "get_path", lambda _name: tmp_path)
    monkeypatch.setattr(gamma.requests, "Session", session)
    spectrumpath = tmp_path / "artis_files" / "data" / "gamma_sc44.txt"
    spectrumpath.parent.mkdir(parents=True)
    spectrumpath.write_text("the spectrum of an earlier run\n", encoding="utf-8")
    return spectrumpath


sc44_rows = [nudat_row("0.0", "G", "1157.020", "99.9"), nudat_row("0.0", "G", "1499.46", "0.908")]


def test_a_complete_ground_level_table_replaces_the_spectrum(monkeypatch, tmp_path):
    """A complete table with a ground level gives the gamma lines in MeV, sorted by energy."""
    spectrumpath = run_with_responses(monkeypatch, tmp_path, nudat_response(list(reversed(sc44_rows))))
    gamma.main()
    assert spectrumpath.read_text(encoding="utf-8") == "2\n1.157  0.9990\n1.499  0.0091\n"


def test_a_truncated_table_keeps_the_earlier_spectrum(monkeypatch, tmp_path):
    """A response that stops after one complete row must not give a spectrum of one line."""
    spectrumpath = run_with_responses(monkeypatch, tmp_path, nudat_response(sc44_rows[:1], complete=False))
    with pytest.raises(SystemExit, match="NuDat returned an incomplete table for 1 nuclides: sc44"):
        gamma.main()
    assert spectrumpath.read_text(encoding="utf-8") == "the spectrum of an earlier run\n"


def test_an_unknown_parent_level_is_not_the_ground_level(monkeypatch, tmp_path):
    """A parent level of asterisks is unknown. Its lines must not replace the spectrum of the ground level."""
    rows = [nudat_row("**********", "G", "271.24", "86.7")]
    spectrumpath = run_with_responses(monkeypatch, tmp_path, nudat_response(rows))
    with pytest.raises(SystemExit, match="The NuDat table of 1 nuclides has no ground level: sc44"):
        gamma.main()
    assert spectrumpath.read_text(encoding="utf-8") == "the spectrum of an earlier run\n"


def test_a_line_of_an_unknown_parent_level_stays_out_of_the_ground_level_spectrum(monkeypatch, tmp_path):
    """The ground level keeps its own lines when the table also has a level with no known energy."""
    rows = [*sc44_rows, nudat_row("**********", "G", "271.24", "86.7")]
    spectrumpath = run_with_responses(monkeypatch, tmp_path, nudat_response(rows))
    gamma.main()
    assert spectrumpath.read_text(encoding="utf-8") == "2\n1.157  0.9990\n1.499  0.0091\n"


def test_missing_values_of_other_fields_do_not_become_zero(monkeypatch, tmp_path):
    """A line with no intensity gives no line. A line with an intensity and no energy stops the nuclide."""
    rows = [*sc44_rows, nudat_row("0.0", "G", "2656.5", "**********")]
    spectrumpath = run_with_responses(monkeypatch, tmp_path, nudat_response(rows))
    gamma.main()
    assert spectrumpath.read_text(encoding="utf-8") == "2\n1.157  0.9990\n1.499  0.0091\n"

    rows = [*sc44_rows, nudat_row("0.0", "G", "**********", "5.0")]
    spectrumpath.unlink()
    spectrumpath.parent.rmdir()
    spectrumpath = run_with_responses(monkeypatch, tmp_path, nudat_response(rows))
    with pytest.raises(SystemExit, match="NuDat returned an incomplete table for 1 nuclides: sc44"):
        gamma.main()
    assert spectrumpath.read_text(encoding="utf-8") == "the spectrum of an earlier run\n"

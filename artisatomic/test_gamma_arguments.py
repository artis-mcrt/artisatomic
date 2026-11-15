"""Check the argument and decay table checks of the gamma command, which precede each network request."""

from pathlib import Path

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
